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

#define DEBUG 1
#define ASSERT 1
#define debug_print(...)                  \
	do                                    \
	{                                     \
		if (DEBUG)                        \
			fprintf(stderr, __VA_ARGS__); \
	} while (0)
#define IDX(x, y, N) (x * N + y)
#define BLOCK_SIZE 16

int N, K, L;
int *O, *dO;
float *A, *B, *pi, *alpha, *beta, *xi, *gamm;
float *dA, *dB, *dpi, *dalpha, *dbeta, *dxi, *dgamm;

//transition, emission matrices, the observence sequence and the prior probability array 
//can be put into texture memory for faster access, since they are read-only.
texture<float, 1, cudaReadModeElementType> text_A;
texture<float, 1, cudaReadModeElementType> text_B;
texture<float, 1, cudaReadModeElementType> text_pi;
texture<int, 1, cudaReadModeElementType> text_O;


//su an icin alpha row row okuyo digerleri column column okuyo. 
//coalescing icin matrixlerin transposeunu almayi deneyebilirsin.
__global__ void first_forward(int N, int K, float * alpha)
{
	int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	alpha[IDX(0, x, N)] = tex1Dfetch(text_pi,x) + tex1Dfetch(text_B,IDX(x, tex1Dfetch(text_O, 0), K)); 
	//debug_print("%d, pi:%.4f, B:%.4f, O:%d \n, result:%.4f ", x, tex1Dfetch(text_pi,x), tex1Dfetch(text_B,IDX(x, tex1Dfetch(text_O, 0), K)), tex1Dfetch(text_O, 0), alpha[IDX(0, x, N)]);
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
			xi = (float *)malloc(sizeof(float) * N * N);
			B = (float *)malloc(sizeof(float) * N * K);
			pi = (float *)malloc(sizeof(float) * N);
			checkCudaErrors( cudaMallocHost((void**)&alpha, sizeof(float) * N * L));
			checkCudaErrors( cudaMallocHost((void**)&beta, sizeof(float) * N * L));
			gamm = (float *)malloc(sizeof(float) * K * N);
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

	cudaStream_t stream_forw, stream_back;
	checkCudaErrors(cudaStreamCreate(&stream_forw));
	checkCudaErrors(cudaStreamCreate(&stream_back));


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

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(ceil((float) N / dimBlock.x)); 
	debug_print("%f, grid size for initial forward alg \n", ceil((float) N / dimBlock.x));

	//first step of forward algorithm.
	checkCudaErrors( cudaMalloc((void**)&dalpha, sizeof(float) * N * L));
	
	//TODO buraya senkron koymam gerekir mi?? probably not. yine de aklinda bulunsun.
	
	first_forward<<<dimGrid, dimBlock, 0, stream_forw>>>(N,K, dalpha);

	checkCudaErrors(cudaMemcpyAsync(alpha,dalpha,N * sizeof(float),cudaMemcpyDeviceToHost, stream_forw));

	cudaDeviceSynchronize();

	//TODO don't forget to free host memory, device memory, texture memory and pinned memory biatch.
	
	debug_print("\n\n\n\n\nInitial forward probabilities \n");
	for (int j = 0; j < N; j++)
	{
		debug_print("%.4e ", exp(alpha[IDX(0, j, N)]));
	}
	debug_print("\n");



	
	

}