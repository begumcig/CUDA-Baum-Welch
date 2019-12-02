/*
* Implementation of Baum - Welch algorithm for training the 
* transition and emission probabilities for a Hidden Markov Model.
*
* N = # of hidden states
* K = # of output states
* L = length of the observance sequence
*
* O = observence sequence
* A = transition matrix
* B = emission matrix
* pi = prior probabilities

*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <regex.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include <helper_cuda.h>

#define DEBUG 0
#define ASSERT 1
#define debug_print(...)                  \
	do                                    \
	{                                     \
		if (DEBUG)                        \
			fprintf(stderr, __VA_ARGS__); \
	} while (0)
#define IDX(x, y, N) (x * N + y)
#define BLOCK_SIZE 16
#define MAX_BLOCK_SZ 1024

int N, K, L;
int *O, *dO;
float *A, *B, *pi, *alpha, *beta, *xi, *gamm;
float *dA, *dB, *dpi, *dalpha,*dbeta, *dxi, *dgamm;
float *obs_seq, *dobs_seq, *dgammn, *gammn, *dgammn_col;
float *dB_obs, *B_obs;


//transition, emission matrices, the observence sequence and the prior probability array 
//can be put into texture memory for faster access, since they are read-only.
texture<float, 1, cudaReadModeElementType> text_A;
texture<float, 1, cudaReadModeElementType> text_B;
texture<float, 1, cudaReadModeElementType> text_pi;
texture<int, 1, cudaReadModeElementType> text_O;

__device__ float add_logs(float, float);
__device__ float sl_add_logs(float, float);



//su an icin alpha row row okuyo digerleri column column okuyo. 
//coalescing icin matrixlerin transposeunu almayi deneyebilirsin.
__global__ void first_forward(int N, int K, float * alpha)
{
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	alpha[IDX(0, x, N)] = tex1Dfetch(text_pi,x) + tex1Dfetch(text_B,IDX(x, tex1Dfetch(text_O, 0), K)); 

}

__global__ void first_backward(int L, int N, float * beta){
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	beta[IDX(L-1, x, N)] = 0;
}

//idea: tz just to compute state transition array.
__global__ void forward_step(int N, int K, int step, float * alpha){
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int x = bx * BLOCK_SIZE + tx;
	float sum = logf(0);
	int observation = tex1Dfetch(text_O, step);

	//bunu iceri tasiyip dene
	__shared__ float salpha[BLOCK_SIZE];
	__shared__ float sB[BLOCK_SIZE];
	__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];

	int i,j;
	for(i = 0; i < N; i+= BLOCK_SIZE){
		// if i + tx < N
		
		salpha[tx] = alpha[IDX((step-1),(i + tx), N)];
		sB[tx] = tex1Dfetch(text_B, IDX(x, observation, K));
		for(j = 0; j < BLOCK_SIZE; j++)
			sA[j][tx] = tex1Dfetch(text_A, IDX(i + j, x, N));

		__syncthreads();
		for(j = 0; j < BLOCK_SIZE; j++)
			sum = add_logs(sum, (salpha[j] + sA[j][tx] + sB[tx]));

		__syncthreads();
	}

	alpha[IDX(step, x, N)] = sum;
}

__global__ void backward_step(int N, int K, int step, float * beta){
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int x = bx * BLOCK_SIZE + tx;
	float sum = logf(0);

	__shared__ float sbeta[BLOCK_SIZE];
	__shared__ float sB[BLOCK_SIZE];

	int i,j;
	for(i = 0; i < N; i+= BLOCK_SIZE){
		sbeta[tx] = beta[IDX((step +1),(i + tx),N)];
		sB[tx] = tex1Dfetch(text_B, IDX((i + tx), tex1Dfetch(text_O, (step+1)), K));

		__syncthreads();

		for(j = 0; j < BLOCK_SIZE; j++)
			sum = add_logs(sum, (sbeta[j] + tex1Dfetch(text_A, IDX(x, i+j, N)) + sB[j]));

		__syncthreads();
 	}

 	beta[IDX(step, x, N)] = sum;
 	//printf("%.4f is the result for thread %d step %d \n", sum, x, step);


}

__global__ void baum_gamma_init(float * obs_seq, int L){
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int y = bx * BLOCK_SIZE + tx;
	int x = by * BLOCK_SIZE + ty;

	obs_seq[IDX(x,y,L)] = logf(0);

}

__global__ void baum_gamma_first(float * obs_seq, float * alpha, float * beta, float* gamm, float * gamm_col, int L, int N){
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int y = bx * BLOCK_SIZE + tx;
	int x = by * BLOCK_SIZE + ty;

	obs_seq[IDX(tex1Dfetch(text_O, x), x, L)] = 0;
	float alph = alpha[IDX(x,y, N)];
	float bet = beta[IDX(x,y, N)];

	gamm[IDX(x, y, N)] =  alph + bet;
	//column version of gamma to use in reduce sum for the next step.
	gamm_col[IDX(y,x,L)] = alph + bet;


}

__global__ void baum_xi_first(float * obs_seq, float * B_obs, int K, int N, int L){
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; 		
    int by = blockIdx.y;
    int tx = threadIdx.x;		
    int ty = threadIdx.y;

	int y = bx * BLOCK_SIZE + tx; // L-1
	int x = by * BLOCK_SIZE + ty; // N

	// Initialize accumulator to 0
	float sum = logf(0);

	//B = N * K
	//obs_seq = K * L-1
	//load B into As and obs_seq as Bs

	for (int k = 0; k < (BLOCK_SIZE + K - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + tx < K && x < N){
             As[ty][tx] = tex1Dfetch(text_B, (x*K + k*BLOCK_SIZE + tx));
             //printf("assssssss %d %d %.4f \n", ty, tx, As[ty][tx] );
         }
         else
             As[ty][tx] = logf(0);

         if (k*BLOCK_SIZE + ty < K && y <(L-1)){
             Bs[ty][tx] = obs_seq[(k*BLOCK_SIZE + ty)* L + y+1];
             //printf("bssssssss %d %d %.4f \n", ty, tx, Bs[ty][tx] );
         }
         else
             Bs[ty][tx] = logf(0);

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n){
         	if(Bs[n][tx] != logf(0))
             sum = add_logs(sum, As[ty][n] + Bs[n][tx]);
             	//printf("%.4f, %.4f, %d, %d , %d \n",expf(sum), expf(As[ty][n] + Bs[n][tx]), As[ty][n], Bs[n][tx], ty);
         }

         __syncthreads();
    }

    if (x < N && y < L-1){
        B_obs[y*N +x] = sum;
        //printf("%d, %d, %.4e \n",x,y,sum);
    }


}


__global__ void baum_gamma(float* gamm,  float* obs_seq,
                                     float* gammn, float * d_total_sum, int K, int L, int N)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; 		
    int by = blockIdx.y;
    int tx = threadIdx.x;		
    int ty = threadIdx.y;

	int y = bx * BLOCK_SIZE + tx;
	int x = by * BLOCK_SIZE + ty;

	// Initialize accumulator to 0
	float sum = logf(0);

	//obs_seq * gamma = transpose (gamma nominator)
	// K x L  * L x N
	//grid size K by N 
	// acol L arow K bcol N brow L ccol N crow K


	for (int k = 0; k < (BLOCK_SIZE + L - 1)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + tx < L && x < K){
             As[ty][tx] = obs_seq[x*L + k*BLOCK_SIZE + tx];
             //printf("assssssss %d %d %.4f \n", ty, tx, As[ty][tx] );
         }
         else
             As[ty][tx] = logf(0);

         if (k*BLOCK_SIZE + ty < L && y < N){
             Bs[ty][tx] = gammn[(k*BLOCK_SIZE + ty)* N + y];
             //printf("bssssssss %d %d %.4f \n", ty, tx, Bs[ty][tx] );
         }
         else
             Bs[ty][tx] = logf(0);

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n){
         	if(As[ty][n]!=logf(0))
             sum = add_logs(sum, As[ty][n] + Bs[n][tx]);
             	//printf("%.4f, %.4f, %d, %d , %d \n",expf(sum), expf(As[ty][n] + Bs[n][tx]), As[ty][n], Bs[n][tx], ty);
         }

         __syncthreads();
    }

    if (x < K && y < N){
        gamm[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = sum - d_total_sum[y];
        //printf("%d, %d, %.4e \n",x,y,sum);
    }
}

__global__ void baum_xi(float * alpha, float * beta, float * B_obs, float * xi, int K, int L, int N)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Cs[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Ds[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x; 		
    int by = blockIdx.y;
    int tx = threadIdx.x;		
    int ty = threadIdx.y;

	int y = bx * BLOCK_SIZE + tx;
	int x = by * BLOCK_SIZE + ty;

	// Initialize accumulator to 0
	float sum = logf(0);

	//alpha * B_obs . beta . A
	// L-1 x N  * L-1 x N
	//grid size K by N 
	// acol L arow K bcol N brow L ccol N crow K

	Ds[ty][tx] = tex1Dfetch(text_A, IDX(x,y,N));


	for (int k = 0; k < (BLOCK_SIZE + L - 2)/BLOCK_SIZE; k++) {

         if (k*BLOCK_SIZE + ty < (L-1) && y < N){
             As[tx][ty] = alpha[(k*BLOCK_SIZE + ty)* N + y];
             //printf("assssssss %d %d %.4f \n", ty, tx, As[ty][tx] );
         }
         else
             As[tx][ty] = logf(0);

         if (k*BLOCK_SIZE + ty < (L-1) && y < N){
             Bs[ty][tx] = beta[(k*BLOCK_SIZE + ty+1)* N + y];
             Cs[ty][tx] = B_obs[(k*BLOCK_SIZE + ty)* N + y];
             //printf("bssssssss %d %d %.4f \n", ty, tx, Bs[ty][tx] );
         }
         else{
             Bs[ty][tx] = logf(0);
             Cs[ty][tx] = logf(0);
         }

         __syncthreads();

         for (int n = 0; n < BLOCK_SIZE; ++n){
             sum = add_logs(sum, As[ty][n] + Bs[n][tx] + Cs[n][tx] + Ds[ty][tx]);
             	//printf("%.4f, %.4f, %d, %d , %d \n",expf(sum), expf(As[ty][n] + Bs[n][tx]), As[ty][n], Bs[n][tx], ty);
         }

         __syncthreads();
    }

    if (x < N && y < N){
        xi[((blockIdx.y * blockDim.y + threadIdx.y)*N) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = sum;
        //printf("%d, %d, %.4e \n",x,y,sum);
    }
}


__global__
void block_sum_reduce(float* const d_block_sums, 
	float * const d_in,
	const unsigned int d_in_len)
{
	extern __shared__ float s_out[];

	unsigned int max_elems_per_block = blockDim.x * 2;
	unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	
	// Zero out shared memory
	// Especially important when padding shmem for
	//  non-power of 2 sized input
	s_out[threadIdx.x] = logf(0);
	s_out[threadIdx.x + blockDim.x] = logf(0);
	d_block_sums[blockIdx.x] = logf(0);

	__syncthreads();

	// Copy d_in to shared memory per block
	if (glbl_tid < d_in_len)
	{
		s_out[threadIdx.x] = d_in[glbl_tid];
		if (glbl_tid + blockDim.x < d_in_len)
			s_out[threadIdx.x + blockDim.x] = d_in[glbl_tid + blockDim.x];
	}
	__syncthreads();

	// Actually do the reduction
	for (unsigned int s = blockDim.x; s > 0; s >>= 1) {
		if (tid < s) {
			if(s_out[tid + s] != logf(0))
			s_out[tid] = add_logs(s_out[tid], s_out[tid + s]);
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0)
		d_block_sums[blockIdx.x] = s_out[0];
}


void gpu_sum_reduce(float * d_in, int L, cudaStream_t stream, float * d_total_sum)
{
	
		//float total_sum;
		//checkCudaErrors(cudaMallocHost((void**) &total_sum, sizeof(float)));
		//total_sum = logf(0);

		unsigned int block_sz = MAX_BLOCK_SZ; 
		unsigned int max_elems_per_block = block_sz * 2; // due to binary tree nature of algorithm
		// NVIDIA's reduceX()
		//unsigned int max_elems_per_block = block_sz;
		
		unsigned int grid_sz = 0;
		if (L <= max_elems_per_block)
		{
			grid_sz = (unsigned int)std::ceil(float(L) / float(max_elems_per_block));
		}
		else
		{
			grid_sz = L / max_elems_per_block;
			if (L % max_elems_per_block != 0)
				grid_sz++;
		}

		// Allocate memory for array of total sums produced by each block
		// Array length must be the same as number of blocks / grid size
		float* d_block_sums;
		checkCudaErrors(cudaMalloc(&d_block_sums, sizeof(float) * grid_sz));
		//checkCudaErrors(cudaMemset(d_block_sums, logf(0), sizeof(float) * grid_sz));

		// Sum data allocated for each block
		block_sum_reduce<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block, stream>>>(d_block_sums, d_in, L);
		//reduce4<<<grid_sz, block_sz, sizeof(unsigned int) * block_sz>>>(d_block_sums, d_in, N);
		//print_d_array(d_block_sums, grid_sz);

		// Sum each block's total sums (to get global total sum)
		// Use basic implementation if number of total sums is <= 2048
		// Else, recurse on this same function
		if (grid_sz <= max_elems_per_block)
		{
			
			//checkCudaErrors(cudaMemset(d_total_sum, logf(0), sizeof(float)));
			block_sum_reduce<<<1, block_sz, sizeof(float) * max_elems_per_block, stream>>>(d_total_sum, d_block_sums, grid_sz);
			//reduce4<<<1, block_sz, sizeof(unsigned int) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);
			//checkCudaErrors(cudaMemcpyAsync(&total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
			//checkCudaErrors(cudaFree(d_total_sum));
		}
		else
		{
			//float * d_in_block_sums;
			//checkCudaErrors(cudaMalloc(&d_in_block_sums, sizeof(float) * grid_sz));
			//checkCudaErrors(cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(float) * grid_sz, cudaMemcpyDeviceToDevice));
			gpu_sum_reduce(d_block_sums, grid_sz, stream, d_total_sum);
			//checkCudaErrors(cudaFree(d_in_block_sums));
		}

		checkCudaErrors(cudaFree(d_block_sums));
	
}


__device__ float add_logs(float x, float y) {
  if (y <= x)
    return x + log1pf(expf(y - x));
  else
    return y + log1pf(expf(x - y));
}

//slower version of add logs; to avoid getting NaN
__device__ float sl_add_logs(float x, float y) {
	return(log( exp((double)x) + exp((double)y)) );
}


int main(int argc, char *argv[]){
	int c;
	char *file_path = NULL;
	opterr = 0;
	


	while ((c = getopt(argc, argv, "f:")) != -1)
	{
		switch (c)
		{
		case 'f':
			file_path = optarg;
			break;
		case '?':
			fprintf(stderr, "Unknown option character.\n");
			exit(EXIT_FAILURE);
		default:
			abort();
		}
	}

	FILE *input_file = fopen(file_path, "r");
	size_t size_line = 0, length_line = 0;
	char *line = NULL;
	char *p = NULL;

	//remove white space lines.
	regex_t regex;
	regcomp(&regex, "^[\t\n ]*$", 0);
	bool first = true, b_pi = false, b_A = false, b_B = false, b_O = false;

	c = 0;
	while ((length_line = getline(&line, &size_line, input_file)) != -1)
	{
		if (regexec(&regex, line, 0, NULL, 0) == 0 || line[0] == '#')
			continue;

		//read the values of N, K and L.
		if (first)
		{
			b_pi = true;
			first = false;
			sscanf(line, "%d %d %d", &N, &K, &L);
			debug_print("%d, %d, %d \n", N, K, L);
			A = (float *)malloc(sizeof(float) * N * N);
			checkCudaErrors(cudaMallocHost((void**) &xi, sizeof(float) * N * N));
			B = (float *)malloc(sizeof(float) * N * K);
			pi = (float *)malloc(sizeof(float) * N);
			checkCudaErrors( cudaMallocHost((void**)&alpha, sizeof(float) * N * L));
			checkCudaErrors( cudaMallocHost((void**)&beta, sizeof(float) * N * L));
			checkCudaErrors(cudaMallocHost((void**) &gamm, sizeof(float) * N * K));
			checkCudaErrors(cudaMallocHost((void**) &B_obs, sizeof(float) * N * (L-1)));
			O = (int *)malloc(sizeof(int) * L);
			continue;
		}
		//read pi.
		else if (b_pi)
		{
			b_pi = false;
			b_A = true;
			p = strtok(line, " ");
			int i = 0;
			while (p != NULL && strcmp(p, "\n") != 0)
			{
				pi[i++] = logf(atof(p));
				debug_print("%f ", pi[i - 1]);
				p = strtok(NULL, " ");
			}
			assert(i == N);
			debug_print("\n");
			continue;
		}
		//read A.
		else if (b_A)
		{
			p = strtok(line, " ");
			int i = 0;
			while (p != NULL && strcmp(p, "\n") != 0)
			{
				A[IDX(c, i++, N)] = logf(atof(p));
				debug_print("%f ", A[IDX(c, i - 1, N)]);
				p = strtok(NULL, " ");
			}
			assert(i == N);
			debug_print("\n");
			c++;
			if (c == N)
			{
				c = 0;
				b_A = false;
				b_B = true;
			}
			continue;
		}
		//read B
		else if (b_B)
		{
			p = strtok(line, " ");
			int i = 0;
			while (p != NULL && strcmp(p, "\n") != 0)
			{
				B[IDX(c, i++, K)] = logf(atof(p));
				debug_print("%f ", B[IDX(c, i - 1, K)]);
				p = strtok(NULL, " ");
			}
			assert(i == K);
			debug_print("\n");
			c++;
			if (c == N)
			{
				b_B = false;
				b_O = true;
				c = 0;
			}
			continue;
		}

		else if (b_O)
		{
			b_O = false;
			p = strtok(line, " ");
			int i = 0;
			while (p != NULL && strcmp(p, "\n") != 0)
			{
				O[i++] = atoi(p);
				debug_print("%d ", O[i - 1]);
				p = strtok(NULL, " ");
			}
			assert(i == L);
			debug_print("\n");
			continue;
		}
	}
	fclose(input_file);

	cudaStream_t stream_forw, stream_back, stream_gamm, stream_xi;
	checkCudaErrors(cudaStreamCreate(&stream_forw));
	checkCudaErrors(cudaStreamCreate(&stream_back));
	checkCudaErrors(cudaStreamCreate(&stream_gamm));
	checkCudaErrors(cudaStreamCreate(&stream_xi));


	//bu kismi da paralelize et.
	size_t offset = 0;
	checkCudaErrors(cudaMalloc((void**)&dA, sizeof(float)*N*N));
	cudaMemcpy(dA, A, sizeof(float)*N*N, cudaMemcpyHostToDevice);

	text_A.addressMode[0] = cudaAddressModeBorder;
	text_A.addressMode[1] = cudaAddressModeBorder;
	text_A.filterMode = cudaFilterModePoint;
	text_A.normalized = false;

	cudaBindTexture(&offset, text_A, dA, sizeof(float)*N*N);


	checkCudaErrors(cudaMalloc((void**)&dB, sizeof(float)*N * K));
	cudaMemcpy(dB, B, sizeof(float)*N * K, cudaMemcpyHostToDevice);

	text_B.addressMode[0] = cudaAddressModeBorder;
	text_B.addressMode[1] = cudaAddressModeBorder;
	text_B.filterMode = cudaFilterModePoint;
	text_B.normalized = false;

	cudaBindTexture(&offset, text_B, dB, sizeof(float)*N*K);

	checkCudaErrors(cudaMalloc((void**)&dpi, sizeof(float)*N));
	cudaMemcpy(dpi, pi, sizeof(float)*N, cudaMemcpyHostToDevice);

	text_pi.addressMode[0] = cudaAddressModeBorder;
	text_pi.addressMode[1] = cudaAddressModeBorder;
	text_pi.filterMode = cudaFilterModePoint;
	text_pi.normalized = false;

	cudaBindTexture(&offset, text_pi, dpi, sizeof(float)*N);

	checkCudaErrors(cudaMalloc((void**)&dO, sizeof(int)* L));
	cudaMemcpy(dO, O, sizeof(int) *L, cudaMemcpyHostToDevice);

	text_O.addressMode[0] = cudaAddressModeBorder;
	text_O.addressMode[1] = cudaAddressModeBorder;
	text_O.filterMode = cudaFilterModePoint;
	text_O.normalized = false;

	cudaBindTexture(&offset, text_O, dO, sizeof(int)*L);

	//allocate memory for forward and backward algorithms on the device.
	checkCudaErrors( cudaMalloc((void**)&dalpha, sizeof(float) * N * L));
	checkCudaErrors( cudaMalloc((void**)&dbeta, sizeof(float) * N * L));

	dim3 dimBlock(min(N, BLOCK_SIZE));
	dim3 dimGrid(ceil((float) N / dimBlock.x)); 
	debug_print("%f, grid size for initial forward alg \n", ceil((float) N / dimBlock.x));

	//first step of forward algorithm.
	//olasilik: bu kadar buyuk bi allocation yapmak yerine her tur N kadarlik memory allocate edip 
	//sonraki turlarda surekli bi onceki turun sonuclarini gpuya tasimak
	
	//allocate memory for the gamma array of expectation step.
	checkCudaErrors( cudaMalloc((void**)&dgammn, sizeof(float) * N * L));
	checkCudaErrors( cudaMalloc((void**)&dgammn_col, sizeof(float) * N * L));
	checkCudaErrors( cudaMalloc((void**)&dgamm, sizeof(float) * N * K));
	checkCudaErrors( cudaMalloc((void**)&dxi, sizeof(float) * N * N));

	dim3 dimBObs(min(L, BLOCK_SIZE), min(K, BLOCK_SIZE));
	dim3 dimGObs(ceil((float) L / dimBObs.x) , ceil((float) K/ dimBObs.y));

	//initalize the observance sequence 
	checkCudaErrors(cudaMalloc((void**)&dobs_seq, sizeof(float) * K * L));
	baum_gamma_init<<<dimGObs, dimBObs, 0, stream_gamm>>>(dobs_seq, L);
	
	//calisma sirasina bak
	first_forward<<<dimGrid, dimBlock, 0, stream_forw>>>(N,K, dalpha);
	checkCudaErrors(cudaMemcpyAsync(alpha,dalpha,N * sizeof(float),cudaMemcpyDeviceToHost, stream_forw));

	first_backward<<<dimGrid, dimBlock,0, stream_back>>>(L, N, dbeta);
	checkCudaErrors(cudaMemcpyAsync(beta,dbeta,N * sizeof(float),cudaMemcpyDeviceToHost, stream_back));


	
	//olasilik: her seferinde bir onceki turu texture memoryye yuklemek?
	//olasilik: texture bind etme isini de asama asama yapmak
	for (int i = 1; i < L; i++){
		forward_step<<<dimGrid, dimBlock, 0, stream_forw>>>(N, K, i, dalpha);
		checkCudaErrors(cudaMemcpyAsync(alpha + i * N,dalpha + i * N ,N * sizeof(float),cudaMemcpyDeviceToHost, stream_forw));
		backward_step<<<dimGrid, dimBlock, 0, stream_back>>>(N, K, (L-1-i), dbeta);
		checkCudaErrors(cudaMemcpyAsync(beta + (L-1-i) * N, dbeta + (L-1-i) * N ,N * sizeof(float),cudaMemcpyDeviceToHost, stream_back));

	}
	cudaDeviceSynchronize();


	dim3 dimBGamma(min(N, BLOCK_SIZE), min(L, BLOCK_SIZE));
	dim3 dimGGamma(ceil((float) N / dimBGamma.x) , ceil((float) L / dimBGamma.y));
	//initialize the matrices to calculate gamma array.
	baum_gamma_first<<<dimGGamma, dimBGamma, 0, stream_gamm>>>(dobs_seq, dalpha, dbeta, dgammn, dgammn_col, L, N);

	dim3 dimBxiF(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGxiF(ceil((float)(L-1) / dimBxiF.x), ceil((float) N/ dimBxiF.y));

	checkCudaErrors( cudaMalloc((void**)&dB_obs, sizeof(float) * N * (L-1)));
	cudaDeviceSynchronize();

	baum_xi_first<<<dimGxiF, dimBxiF, 0, stream_xi>>>(dobs_seq, dB_obs, K, N, L);
	
	//checkCudaErrors(cudaMemcpyAsync(B_obs, dB_obs, sizeof(float) * (L-1)* N, cudaMemcpyDeviceToHost, stream_xi));

	cudaStream_t streams[N];
	float * d_total_sum;
	checkCudaErrors(cudaMalloc(&d_total_sum, sizeof(float) * N));

	//calculate the denominator (gamma sum)
	for(int i = 0; i < N; i++){
		cudaStreamCreate(&streams[i]);
		gpu_sum_reduce(dgammn_col + i * L, L, streams[i], d_total_sum + i);
		//printf("here is your sum hon %.4f\n", s);
	}

	cudaDeviceSynchronize();	

	//calculate gamma
	dim3 dimBGammaD(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGGammaD(ceil((float) N / dimBGammaD.x) , ceil((float) K / dimBGammaD.y));
	baum_gamma<<<dimGGammaD, dimBGammaD, 0, stream_gamm>>>(dgamm, dobs_seq, dgammn, d_total_sum, K, L, N );

	//calcuate xi

	dim3 dimBxi(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGxi(ceil((float) N / dimBGammaD.x) , ceil((float) N / dimBGammaD.y));
	baum_xi<<<dimGxi, dimBxi, 0, stream_xi>>>(dalpha, dbeta, dB_obs, dxi, K,L,N);

	
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(gamm, dgamm, sizeof(float) * K * N, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(xi, dxi, sizeof(float) * N * N, cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	
/*

	//TODO don't forget to free host memory, device memory, texture memory and pinned memory biatch.
	printf("gamma probability array \n");
	for(int i = 0; i < L; i++){
		for(int j = 0; j < N; j++){
			printf("%.4f ", (gammn[IDX(i,j,N)]));
		}
		printf("\n");
	}

	printf("gamma observations probability array \n");
	for(int i = 0; i < K; i++){
		for(int j = 0; j < L; j++){
			printf("%.4f ", (obs_seq[IDX(i,j,L)]));
		}
		printf("\n");
	}
*/
	printf("gamma nom probability array \n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < K; j++){
			printf("%.4f ", (gamm[IDX(j,i,N)]));
		}
		printf("\n");
	}
/*
	printf("dbobs \n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < (L-1); j++){
			printf("%.4f ", (B_obs[IDX(j,i,(N))]));
		}
		printf("\n");
	}*/


		printf("xi nom probability array \n");
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			printf("%.4f ", (xi[IDX(i,j,N)]));
		}
		printf("\n");
	}

	debug_print("Forward probability array \n");
	for(int i = 0; i < L; i++){
		for(int j = 0; j < N; j++){
			debug_print("%.4e ", exp((double)alpha[IDX(i,j,N)]));
		}
		debug_print("\n");
	}

	debug_print("\n\n\n\n\nBackward probability array \n");
  for(int j = 0; j < L; j++){
    for(int k= 0; k < N; k++){
      debug_print("%.4e  ", exp((double)beta[IDX(j,k, N)]));
    }
    debug_print("\n"); 
  }




	
	

}