/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 *    Edited by Erin Cummings and Zeyad Al Awwad
 */ 
#define _GNU_SOURCE
#include <string.h>
#include <tmmintrin.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

int BLOCK_SIZE = 64;
int BLOCK_SIZE2 = 32;
 
#define min(a,b) (((a)<(b))?(a):(b)) 
//Function for second level of cache blocking
static void do_block2 (int lda, int M2, int N2, int K2, double* restrict A0, double* restrict B0, double* restrict C)
{
	double A[2*K2];
	double B[N2*K2];
	for (int i=0; i<N2; i++)
	{
		memcpy(&B[i*K2], B0+i*lda, K2*sizeof(double));
	}//end i
	__m128d register a00_a01, a10_a11, a02_a03, a12_a13, a04_a05, a14_a15, a06_a07, a16_a17, cij, cij2, bk0_bk0, bk1_bk1, temp1, temp2, temp3; //moved to one single line for alignment optimization
    /* For each row i of A */
    for (int i = 0, len=M2-1; i < len; i+=2) {
    /* For each column j of B */
    memcpy(&A[0], A0+i*lda, K2*sizeof(double)); //allocate space for A
    memcpy(&A[K2], A0+(i+1)*lda, K2*sizeof(double)); //prefetching instructions
        for (int k = 0, len2=K2-7; k < len2; k+=8)
        {
            /* Compute C(i,j) */
            a00_a01 = _mm_load_pd(A+k);
            a10_a11 = _mm_load_pd(A + M2+k);
            a02_a03 = _mm_load_pd(A+k+2);
            a12_a13 = _mm_load_pd(A + M2+k+2);
	    a04_a05 = _mm_load_pd(A+k+4);
	    a14_a15 = _mm_load_pd(A + M2+k+4);
	    a06_a07 = _mm_load_pd(A+k+6);
	    a16_a17 = _mm_load_pd(A + M2+k+6);
            
            for (int j = 0, len3=N2-1; j < len3; j+=2)
            {
                bk0_bk0 = _mm_load_pd(B + j*N2+k);
                bk1_bk1 = _mm_load_pd(B + (j+1)*N2+k);
             	cij = _mm_load_pd(C+i*lda+j); // moved load up further from operations
             	cij2 = _mm_load_pd(C + (i+1)*lda+j );
        	temp1 = _mm_mul_pd(a00_a01, bk0_bk0);
                temp2 = _mm_mul_pd(a00_a01, bk1_bk1);
		temp1 = _mm_hadd_pd(temp1, temp2);      // Lock
                temp2 = _mm_mul_pd(a10_a11, bk0_bk0);
                bk0_bk0 = _mm_load_pd( B + j*N2+k+2 );
                temp3 = _mm_mul_pd(a10_a11, bk1_bk1);
                bk1_bk1 = _mm_load_pd( B + (j+1)*N2+k+2 );
                temp2 = _mm_hadd_pd(temp2, temp3);      // Lock

                temp3 = _mm_mul_pd(a02_a03, bk0_bk0);
                temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a02_a03, bk1_bk1));
                temp1 = _mm_add_pd(temp1, temp3);
                temp3 = _mm_mul_pd(a12_a13, bk0_bk0);
                bk0_bk0 = _mm_load_pd(B + j*N2+k+4);
                temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a12_a13, bk1_bk1));
                bk1_bk1 = _mm_load_pd(B + (j+1)*N2+k+4);
                temp2 = _mm_add_pd(temp2, temp3);
                
		temp3 = _mm_mul_pd(a04_a05, bk0_bk0);
		temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a04_a05, bk1_bk1));
		temp1 = _mm_add_pd(temp1, temp3);
		temp3 = _mm_mul_pd(a14_a15, bk0_bk0);
		bk0_bk0 = _mm_load_pd(B + j*N2+k+6);
		temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a14_a15, bk1_bk1));
		bk1_bk1 = _mm_load_pd(B + (j+1)*N2+k+6);
		temp2 = _mm_add_pd(temp2, temp3);

		temp3 = _mm_mul_pd(a06_a07, bk0_bk0);
		temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a06_a07, bk1_bk1));
		temp1 = _mm_add_pd(temp1, temp3);
		temp3 = _mm_mul_pd(a16_a17, bk0_bk0);
		temp3 = _mm_hadd_pd(temp3, _mm_mul_pd(a16_a17, bk1_bk1));
		temp2 = _mm_add_pd(temp2, temp3);
		
                cij = _mm_add_pd(cij, temp1);
                _mm_store_pd( C+i*lda+j, cij);
                
                cij2 = _mm_add_pd(cij2, temp2);
                _mm_store_pd( C+(i+1)*lda+j, cij2);
                
            }
        }
    }
}

/* This performs a smaller dgemm operation on the first level block
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block1 (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C, int i0, int j0, int k0)
{
    /* For each row i of A */
    for (int i = 0; i < M; i += BLOCK_SIZE2)
    /* For each column j of B */
        for (int j = 0; j < N; j += BLOCK_SIZE2)
        {
            /* Compute C(i,j) */
            for (int k = 0; k < K; k += BLOCK_SIZE2)
            {
                int M2 = min (BLOCK_SIZE2, lda-i-i0);
                int N2 = min (BLOCK_SIZE2, lda-j-j0);
                int K2 = min (BLOCK_SIZE2, lda-k-k0);		                
                /* Perform second level block dgemm */
                do_block2(lda, M2, N2, K2, A + i*lda + k, B + j*lda + k, C + i*lda + j);	
            }//end for k
        }//end for j
}//end do_block1

/* This routine performs a padded dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm_padded (int lda0, double* restrict A0, double* restrict B0, double* restrict C0)
{

    int diff = BLOCK_SIZE2 - lda0%BLOCK_SIZE2;
    int lda = lda0+diff;
    double A[lda*lda];
    double B[lda*lda];
    double C[lda*lda];
    
    memset(A, 0, sizeof(A));
    memset(B, 0, sizeof(B));
    memset(C, 0, sizeof(C));
        
    for (int i=0; i<lda0; i++)
    {
        memcpy( A + i*lda, A0 + i*lda0, lda0*sizeof(double) );
        memcpy( B + i*lda, B0 + i*lda0, lda0*sizeof(double) );
    }//end i
    
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            /* Correct block dimensions if block "goes off edge of" the matrix */
            int M = min (BLOCK_SIZE, lda-i);
            int N = min (BLOCK_SIZE, lda-j);
            int K = min (BLOCK_SIZE, lda-k); //could unroll this loop

            /* Perform individual block dgemm */
            do_block1(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j, i, j ,k);    
        }//end k
    }//end i
    
    for (int i = 0; i < lda0; ++i) {
        for (int j = 0; j < lda0; ++j) {
            B0[i*lda0+j] = B[j*lda+i];
        }//end j
    }//end i
    
    for (int i=0; i<lda0; i++)
    {
        memcpy( C0 + i*lda0, C + i*lda, lda0*sizeof(double) );
    }//end i
}
//Original square dgemm without padding
void square_dgemm_original (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    /* For each block-row of A */
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
    /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        /* Accumulate block dgemms into block of C */
        for (int k = 0; k < lda; k += BLOCK_SIZE)
        {
            /* Correct block dimensions if block "goes off edge of" the matrix */
            int M = min (BLOCK_SIZE, lda-i);
            int N = min (BLOCK_SIZE, lda-j);
            int K = min (BLOCK_SIZE, lda-k); 

            /* Perform individual block1 dgemm */
            do_block1(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j, i, j ,k);
        }//end k
    }//end i
    for (int i = 0; i < lda; ++i) {
        for (int j = i+1; j < lda; j++) {
            double t = B[i*lda+j];
            B[i*lda+j] = B[j*lda+i];
            B[j*lda+i] = t;
        }//end j
    }//end i
}//end square dgemm original

/* This routine performs a square dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    if (lda%BLOCK_SIZE2==0) 
    {
    	BLOCK_SIZE = 512;
    	BLOCK_SIZE2 = 64;
    	square_dgemm_original(lda, A, B, C);	
    }//end if
    //if we need to call the padded version
    else 
    {
    	square_dgemm_padded(lda, A, B, C);
    	
    }//end else

}//end square_dgemm
