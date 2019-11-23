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

#define DEBUG 1
#define ASSERT 1
#define debug_print(...)                  \
	do                                    \
	{                                     \
		if (DEBUG)                        \
			fprintf(stderr, __VA_ARGS__); \
	} while (0)
#define IDX(x, y, N) (x * N + y)

int N, K, L;
int *O;
float *A, *B, *pi, *alpha, *beta, *xi, *gamm;

float add_logs(float x, float y);
void forward_alg();
void backward_alg();
void baum_welch_alg();

int main(int argc, char *argv[])
{

	//Implemented according to Speech and Language Processing. Daniel Jurafsky & James H. Martin,
	//Appendix Chapter A: Hidden Markov Models.
	//Pdf of their book can be found at https://web.stanford.edu/~jurafsky/slp3/edbook_oct162019.pdf.

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

	//Reading the input file.
	//Lines that start with # will be ignored.
	//The order of the input file should be as follows:
	//(N, # of hidden state variables; K, # of observed variables; L, #length of the observence sequence)
	//Initial probability distribution Pi (Pi should be of length N)
	//Initial transition matrix A (A should be N * N)
	//Initial emission matrix B (B should be N * K)
	//The observence sequence O (Notice that the elements of O should be numbers
	//between 0 and K-1)

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
			alpha = (float *)malloc(sizeof(float) * L * N);
			beta = (float *)malloc(sizeof(float) * L * N);
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

	baum_welch_alg();

	//printing results.

	/*
  debug_print("Forward probability array \n");
	for(int i = 0; i < L; i++){
		for(int j = 0; j < N; j++){
			debug_print("%.4e ", exp(alpha[IDX(j,i,L)]));
		}
		debug_print("\n");
	}


  debug_print("\n\n\n\n\nBackward probability array \n");
  for(int j = 0; j < L; j++){
    for(int k= 0; k < N; k++){
      debug_print("%.4e  ", exp(beta[IDX(k,j, L)]));
    }
    debug_print("\n"); 
  }*/

	debug_print("\n\n\n\n\nFinal forward probabilities \n");
	for (int j = 0; j < N; j++)
	{
		debug_print("%.4e ", exp(alpha[IDX(j, L - 1, L)]));
	}
	debug_print("\n");

	debug_print("\n\n\n\n\nFinal backward probabilities \n");
	for (int j = 0; j < N; j++)
	{
		debug_print("%.4e ", exp(beta[IDX(j, 0, L)]));
	}
	debug_print("\n");

	debug_print("\n\n\n\n\nAfter first loop, The transition array \n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			debug_print("%.4e, ", exp(A[IDX(i, j, N)]));
		}
		debug_print("\n");
	}
	debug_print("\n");

	debug_print("\n\n\n\n\nAfter first loop, The emission array \n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			debug_print("%.4e, ", exp(B[IDX(i, j, K)]));
		}
		debug_print("\n");
	}
	debug_print("\n");

	debug_print("\n\n\n\n\nAfter first loop, prior probabilities \n");
	for (int j = 0; j < N; j++)
	{
		debug_print("%.4e ", exp(pi[j]));
	}
	debug_print("\n");
}

float add_logs(float x, float y)
{
	if (y <= x)
		return x + log1pf(expf(y - x));
	else
		return y + log1pf(expf(x - y));
}

//Given the transition (A), prior (pi) and emission (B) probabilities,
//we are calculating the likelihood of the observence sequence P(O | A,B,pi).
//The observence sequence is of length L. There are N hidden states.
//alpha[i, t] is the possibility of being in hidden state i at time t with the
//observence sequence O_1.....O_t, given model parameters.
//alpha[i, t] = P(O_1.....O_t, s_t = i | A,B,pi)
//Once we calculate alpha [i, L] we can sum alpha [i, L] from 1--> N to get P(O | A,B, pi)
//This implementation uses log likelihood to avoid vanishing to 0.
void forward_alg()
{
	//initialization step
	//alpha[n, 1] = pi_n * B_n(O_1)
	for (int j = 0; j < N; j++)
		alpha[IDX(j, 0, L)] = pi[j] + B[IDX(j, O[0], K)];

	//recursion step
	//alpha[n, t] = sum k: 1-->N (alpha[k, t-1] * A_kn * B_n(O_t))
	float p;
	float sum;
	for (int i = 1; i < L; i++)
	{
		for (int j = 0; j < N; j++)
		{
			sum = logf(0);
			for (int k = 0; k < N; k++)
			{
				p = alpha[IDX(k, i - 1, L)] + A[IDX(k, j, N)] + B[IDX(j, O[i], K)];
				sum = add_logs(sum, p);
			}
			alpha[IDX(j, i, L)] = sum;
		}
	}
}

void backward_alg()
{
	//initialization
	//beta[n, L] = 1
	//log(1) = 0
	for (int i = 0; i < N; i++)
		beta[IDX(i, L - 1, L)] = 0;

	//recursion
	float p;
	float sum;
	for (int t = L - 2; t >= 0; t--)
	{
		for (int i = 0; i < N; i++)
		{
			sum = logf(0);
			for (int j = 0; j < N; j++)
			{
				p = A[IDX(i, j, N)] + B[IDX(j, O[t + 1], K)] + beta[IDX(j, t + 1, L)];
				sum = add_logs(sum, p);
			}
			beta[IDX(i, t, L)] = sum;
		}
	}
}

void baum_welch_alg()
{
	float p, g, pi_sum;
	float xi_sum[N], gamma_sum[N];
	forward_alg();
	backward_alg();

	pi_sum = logf(0);

	//initialize to 0.
	for (int j = 0; j < N; j++)
	{
		xi_sum[j] = logf(0);
		gamma_sum[j] = logf(0);
	}

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			xi[IDX(i, j, N)] = logf(0);
		}
		for (int j = 0; j < K; j++)
		{
			gamm[IDX(i, j, K)] = logf(0);
		}
	}

	for (int t = 0; t < L - 1; t++)
	{
		for (int i = 0; i < N; i++)
		{
			g = alpha[IDX(i, t, L)] + beta[IDX(i, t, L)];
			gamm[IDX(i, O[t], K)] = add_logs(gamm[IDX(i, O[t], K)], g);
			gamma_sum[i] = add_logs(gamma_sum[i], g);

			for (int j = 0; j < N; j++)
			{
				p = alpha[IDX(i, t, L)] + A[IDX(i, j, N)] + B[IDX(j, O[t + 1], K)] + beta[IDX(j, t + 1, L)];
				xi[IDX(i, j, N)] = add_logs(xi[IDX(i, j, N)], p);
				xi_sum[i] = add_logs(xi_sum[i], p);
			}
		}
	}

	for (int i = 0; i < N; i++)
	{
		g = alpha[IDX(i, L - 1, L)] + beta[IDX(i, L - 1, L)];
		gamm[IDX(i, O[L - 1], K)] = add_logs(gamm[IDX(i, O[L - 1], K)], g);
		gamma_sum[i] = add_logs(gamma_sum[i], g);

		pi_sum = add_logs(pi_sum, alpha[IDX(i, 0, L)] + beta[IDX(i, 0, L)]);
	}

	for (int i = 0; i < N; i++)
	{
		pi[i] = alpha[IDX(i, 0, L)] + beta[IDX(i, 0, L)] - pi_sum;
		for (int j = 0; j < N; j++)
			A[IDX(i, j, N)] = xi[IDX(i, j, N)] - xi_sum[i];
		for (int j = 0; j < K; j++)
		{
			B[IDX(i, j, K)] = gamm[IDX(i, j, K)] - gamma_sum[i];
		}
	}
}