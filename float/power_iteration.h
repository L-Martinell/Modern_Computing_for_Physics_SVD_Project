#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <float.h>
#include "matrix_utility.h"

float ScalarProduct(int n, float *v, float *u){
    // This function returns the scalar product of two vectors lambda = v * u
    // In a matrix formalism, it can be seen as vT u, so the output is a scalar
    float lambda = 0.0f;

    for(int i = 0; i < n; i++){
        lambda += v[i] * u[i];
    }
    return lambda;
}

void OuterProduct(int n, float *v, float *u, float *A){
    // This function returns the matrix product of two vectors A = v (x) u
    // In a matrix formalism, it can be seend as v uT, so the output is a matrix
    // The output matrix is going to be a n x n matrix
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            A[i * n + j] = v[i] * u[j];
        }
    }
}

float RayleighQuotient(int n, float *A, float *b){
    // This function calculates the Rayleigh quotient:
    // lambda = (b^T A b) / (b^T b)
    float lambda = 0.0f, numerator = 0.0f, denominator = 0.0f;
    float *tmp = (float*)malloc(sizeof(float) * n);

    MatVecMul(n, n, A, b, tmp);
    numerator = ScalarProduct(n, b, tmp);
    denominator = ScalarProduct(n, b, b);
    lambda = numerator / denominator;
    free(tmp);
    return lambda;
}

void PowerIteration(int n, float *A, float *b, float *lambda, float accuracy, bool status){
    // This function is a single step in the power iteration algorithm, i.e. returns the
    // leading eigenvalue and the corresponding eigenvector of the matrix A
    
    float *Ab = (float*)malloc(sizeof(float) * n);
    float previous_lambda;
    int max_iterations = 500, counter = 0;              	// Cap the highest amount of iterations at 500
    float norm;
    *lambda = FLT_MIN;
    
    do {
        previous_lambda = *lambda;

        // Calculate Ab and ||Ab||
        MatVecMul(n, n, A, b, Ab);
        norm = NormVector(Ab, n);
        
        // Calculate b_{k+1}
        for(int i = 0; i < n; i++){
            b[i] = Ab[i] / norm;
        }
        *lambda = RayleighQuotient(n, A, b);
        counter++;
        if(counter == max_iterations - 1) break;
    } while(fabsf(*lambda - previous_lambda) > accuracy);
    free(Ab);

    if(status){
        printf("Number of iterations: %d\n", counter);
    }
}


void Deflation(int n, int num_SV, float *A, float *VT, float *lambda_list, float accuracy, bool status){
    // This function calculates the eigenvalues of a matrix using the deflation algorithm
    for(int L = 0; L < num_SV; L++){

        float *q = (float*)malloc(sizeof(float) * n);
        generate_vector(n, q);
        normalize_vector(n, q);

        PowerIteration(n, A, q, &lambda_list[L], accuracy, status);

        // Copy b into the matrix VT
        for(int i = 0; i < n; i++){
            VT[i * num_SV + L] = q[i];
        }

        // Compute b^T b; bb^T
        float *qqT = (float*)malloc(sizeof(float) * n * n);         // b * b^T
        float qTq, qTq2;                                            // b^T * b and (b^T * b)^2
        qTq = ScalarProduct(n, q, q);
        qTq2 = qTq * qTq;
        if(qTq2 == 0.0f){
            printf("qTq2 = 0 for eigenvalue number %d", L + 1);
            continue;
        }
        OuterProduct(n, q, q, qqT);

        // Compute the next step matrix
        float *A_old = (float*)malloc(sizeof(float) * n * n);
        memcpy(A_old, A, sizeof(float) * n * n);
        for(int i = 0; i < n; i++){
            for(int j = 0; j < n; j++){
                A[i * n + j] = A_old[i * n + j] - ( lambda_list[L] / qTq2 ) * qqT[i * n + j];
            }
        }
        
        free(q);
        free(qqT);
        free(A_old);
    }
}

void BuildSigma(int m, int n, int num_SV, float *lambda_list, float *Sigma, bool Eigenvalues){
    // Build the Sigma matrix from the list of eigenvalues of A^T A => Remember to go through the hoop of extracting the square root!
    // Initialize the matrix with all 0's
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            Sigma[i * n + j] = 0.0f;
        }
    }
    // Fill the diagonal 
    for(int i = 0; i < n; i++){
        if(Eigenvalues){
            Sigma[i * n + i] = sqrtf(lambda_list[i]);
        }
        else{
            Sigma[i * n + i] = lambda_list[i];
        }
    }
}

void MoorePenroseInverse(int m, int n, float *Sigma, float *SigmaMP){
    // Computes the Moore-Penrose pseudoinverse matrix of Sigma
    // Due to the particular definition of Sigma, this computation is much easier (phew!)
    float *tmp = (float*)malloc(sizeof(float) * m * n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(i == j && Sigma[i * n + j] != 0.0f){
                tmp[i * n + j] = 1.0f/Sigma[i * n + j];
            }
            else{
                tmp[i * n + j] = 0.0f;
            }
        }
    }

    // Transpose the matrix
    transpose(m, n, tmp, SigmaMP);
    free(tmp);
    return;
}

void ComputeLeftMatrix(int m, int n, float *Sigma, float *A, float *VT, float *U){
    // Computes the left singular matrix U using the previously defined 
    // functions (most importantly, the MoorePenroseInverse function)
    float *V       = (float*)malloc(sizeof(float) * n * n);
    float *SigmaMP = (float*)malloc(sizeof(float) * n * m);
    float *AV      = (float*)malloc(sizeof(float) * m * n);             // Stores the data for intermediate calculations
    transpose(n, n, VT, V);
    MoorePenroseInverse(m, n, Sigma, SigmaMP);
    MatMul(m, n, n, A, V, AV);
    MatMul(m, m, n, AV, SigmaMP, U);

    free(V);
    free(SigmaMP);
    free(AV);
}
