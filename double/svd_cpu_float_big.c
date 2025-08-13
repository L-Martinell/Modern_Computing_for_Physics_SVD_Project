#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "power_iteration.h"

// FORTRAN style LAPACK sgesvd declaration
extern void sgesvd_(char *jobu, char *jobvt, int *m, int *n, 
                    float *a, int *lda, float *s, 
                    float *u, int *ldu, float *vt, int *ldvt,
                    float *work, int *lwork, int *info);

int main(){
    srand(13);
    int m = 2048, n = 1024;
    int num_SV = (m < n) ? m : n;                                                           // Number of singular values: minimum of m and n

    // ################# CUSTOM SVD IMPLEMENTATION #################
    float *A             = (float*)malloc(sizeof(float) * m * n);                           // Matrix A (m x n)
    float *T             = (float*)malloc(sizeof(float) * n * m);                           // Matrix T = A^T (n x m)
    float *ATA           = (float*)malloc(sizeof(float) * n * n);                           // Matrix A^T A = TA
    float *A_rec         = (float*)malloc(sizeof(float) * m * n);                           // Reconstructed matrix A after going through SVD
    float *VT_custom     = (float*)malloc(sizeof(float) * n * n);                           // Right singular matrix
    float *U_custom      = (float*)malloc(sizeof(float) * m * m);                           // Left singular matrix
    float *Sigma_custom  = (float*)malloc(sizeof(float) * m * n);                 	    // Matrix of singular values
    float *lambda_list   = (float*)malloc(sizeof(float) * num_SV);                          // List of eigenvalues of the matrix ATA
    float *USigma_custom = (float*)malloc(sizeof(float) * m * n);                           // Intermediate product U Sigma

    // Generate the random matrix A and the initial guess b
    generate_matrix(m, n, A);

    // ################# LAPACK SVD IMPLEMENTATION #################
    float *A_cm = (float*)malloc(sizeof(float) * m * n);                                    // Pass this version of the matrix A to LAPACK
    Row2ColMajor(m, n, A, A_cm);                                                            // Copy the data... in a column-major ordering!

    float *s      = (float*)malloc(sizeof(float) * n);                                      // Vector of singular values (per LAPACK)
    float *Sigma  = (float*)malloc(sizeof(float) * m * n);                                  // Singular value matrix
    float *U_cm   = (float*)malloc(sizeof(float) * m * m);                                  // Left singular matrix (per LAPACK), col-major indexing
    float *U      = (float*)malloc(sizeof(float) * m * m);                                  // Left singular matrix (per LAPACK), row-major indexing
    float *VT_cm  = (float*)malloc(sizeof(float) * n * n);                                  // Right singular matrix (per LAPACK), col-major indexing
    float *VT     = (float*)malloc(sizeof(float) * n * n);                                  // Right singular matrix (per LAPACK), row-major indexing
    float *USigma = (float*)malloc(sizeof(float) * m * n);                                  // Intermediate product U Sigma
    float *A_rec2 = (float*)malloc(sizeof(float) * m * n);                                  // Reconstructed matrix (per LAPACK)

    // LAPACKE ancillary variables
    int info;
    char jobu  = 'A';
    char jobvt = 'A';
    int LDA    = m;
    int LDU    = m;
    int LDVT   = n;
    int lwork  = -1;
    float work_query;

    sgesvd_(&jobu, &jobvt, 
            &m, &n, 
            A_cm, &LDA, 
            s,
            U_cm, &LDU, 
            VT_cm, &LDVT,
            &work_query, &lwork, &info);

    if(info != 0){
        printf("Workspace query failed with info = %d\n", info);
        exit(1);
    }

    lwork = (int)work_query;
    float *work = (float*)malloc(sizeof(float) * lwork);

    clock_t start_LAPACK = clock();                                                         // Start measuring time for the LAPACK implementation
    sgesvd_(&jobu, &jobvt,
            &m, &n,
            A_cm, &LDA,
            s,
            U_cm, &LDU,
            VT_cm, &LDVT,
            work, &lwork, &info);
    clock_t end_LAPACK = clock();

    if(info > 0){
        printf("SVD did not converge.\n");
    } else if(info < 0){
        printf("Invalid argument at position %d\n", -info);
    }

    free(work);

    // Convert the matrices to row-major indexing for multiplications
    Col2RowMajor(m, m, U_cm, U);
    Col2RowMajor(n, n, VT_cm, VT);

    // Reconstruct the matrix A = U Sigma V^T
    BuildSigma(m, n, num_SV, s, Sigma, false);
    MatMul(m, n, m, U, Sigma, USigma);
    MatMul(m, n, n, USigma, VT, A_rec2);

    // End measuring time for the LAPACK implementation
    double elapsed_LAPACK = (double)(end_LAPACK - start_LAPACK) / CLOCKS_PER_SEC;

    // Calculate the difference between original and reconstructed matrices
    float deviation_LAPACK = MatrixDeviation(m, n, A, A_rec2);

    printf("Deviation (LAPACK implementation): %f\n", deviation_LAPACK);
    printf("Time elapsed (LAPACK implementation): %.10lf s\n", elapsed_LAPACK);

    /*
    FILE *benchmark = fopen("benchmarking_float.txt", "a");
    fprintf(benchmark, "Matrix size: %d x %d, Frobenius norm: %f, time elapsed (SVD CPU float, LAPACK): %.10lf s\n", m, n, deviation_LAPACK, elapsed_LAPACK);
    fclose(benchmark)
    */

    printf("Matrix size: %d x %d\n", m, n);

    // Free the memory!
    free(A);
    free(ATA);
    free(T);
    free(VT_custom);
    free(U_custom);
    free(A_rec);
    free(A_rec2);
    free(A_cm);
    free(USigma);
    free(Sigma_custom);
    free(U_cm);
    free(U);
    free(VT_cm);
    free(VT);
    free(s);
    free(lambda_list);

    return 0;
}
