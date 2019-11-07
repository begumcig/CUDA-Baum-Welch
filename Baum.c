#include <stdio.h>
#include <stdlib.h>

#define DEBUG 1
#define IDX(x, y, N) (x * N + y)



void forward_alg(float * T, float * B, float * pi, int * seq, float * alpha, int l_O, int n_emission, int n_states)
{	
	//initialization step
	for(int j = 0; j < n_states; j++)
	{
		//alpha[s,0] = pi[s] * B[s,O[0]]
		alpha[IDX(j,0, l_O)] = pi[j] * B[IDX(j,seq[0], n_emission)];
	}
	
	//recursion step
	for(int i = 1; i < l_O; i ++)
	{
		for (int j = 0; j < n_states; j++)
		{
			float sum = 0;
			for(int k = 0; k < n_states; k++)
			{
				sum += alpha[IDX(k, i-1, l_O)] * T[IDX(k, j, n_states)] * B[IDX(j, seq[i], n_emission)];
			}
			alpha[IDX(j, i, l_O)] = sum;
		}
	}
}

int main(int argc, char *argv[]){
	const int n_states = 2;
	const int n_emission = 3;
	const int l_O = 2;

	//TEST
	//Example taken from Speech and Language Processing. Daniel Jurafsky & James H. Martin,
	//Appendix Chapter A: Hidden Markov Models.
	//Pdf of their book can be found at https://web.stanford.edu/~jurafsky/slp3/edbook_oct162019.pdf.


	float * T = (float *) malloc(sizeof(float) * n_states * n_states);
	float * B = (float *) malloc(sizeof(float) * n_emission * n_states);
	float * pi = (float *) malloc(sizeof(float) * n_states);
	float * alpha = (float *)malloc(sizeof(float) * l_O  * n_states);

	int * sequence = (int *) malloc(sizeof(int) * l_O);
	sequence[0] = 2;
	sequence[1] = 0;

	T[IDX(0,0,n_states)] = 0.6;
	T[IDX(0,1,n_states)] = 0.4;
	T[IDX(1,0,n_states)] = 0.5;
	T[IDX(1,1,n_states)] = 0.5;

	B[IDX(0,0,n_emission)] = 0.2;
	B[IDX(0,1,n_emission)] = 0.4;
	B[IDX(0,2,n_emission)] = 0.4;
	B[IDX(1,0,n_emission)] = 0.5;
	B[IDX(1,1,n_emission)] = 0.4;
	B[IDX(1,2,n_emission)] = 0.1;

	pi[0] = 0.8;
	pi[1] = 0.2;

	forward_alg(T,B,pi,sequence,alpha,l_O,n_emission,n_states);
	
	if(DEBUG){
		for(int i = 0; i < n_states; i++){
			for(int j = 0; j < l_O; j++){
				printf("%f ",alpha[IDX(i,j,l_O)]);
			}
			printf("\n");
		}
	}

	
	


	

}