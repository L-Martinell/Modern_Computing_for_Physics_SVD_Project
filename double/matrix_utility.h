#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void generate_matrix(int m, int n, float *A){
    /*
    C function that generates a random matrix with entries between -1 and 1
    */
    if(m <= 0 || n <= 0){
        printf("Error: Either the number of rows or columns is non-positive.\n");
        return;
    }
    // Maximum and minimum numbers available
    float max = 1.0f;
    float min = -1.0f;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float r = ((float)rand() / RAND_MAX);
            A[i * n + j] = r * (max - min) + min;
        }
    }
    return;
}

void generate_vector(int n, float *b){
    /*
    C function that generates a random vector with entries between -1 and 1
    */
    if(n <= 0)
    {
        printf("Error: The size of the vector is non-positive.\n");
        return;
    }
    float max = 1.0f;
    float min = -1.0f;
    for(int i = 0; i < n; i++)
    {
        float r = ((float)rand() / RAND_MAX);
        b[i] = r * (max - min) + min;
    }
    return;
}


/*
void normalize_vector(int n, float *b){
    float sum_b = 0.0f;
    for(int i = 0; i < n; i++){
        sum_b += b[i];
    }
    for(int i = 0; i < n; i++){
        b[i] /= sum_b;
    }
    return;
}
*/


void transpose(int m, int n, float *A, float *T){
    /*
    C function to transpose the matrix A into the matrix T
    */
   for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
        T[j * m + i] = A[i * n + j];
    }
   }
   return;
}

void MatMul(int m, int n, int k, float *A, float *B, float *C){
    /*
    C function to perform matrix-matrix multiplication
    */
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            C[i * n + j] = 0.0f;
            for(int l = 0; l < k; l++){
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
    return;
}

void MatVecMul(int m, int n, float *A, float *b, float *c){
    /*
    C function to perform matrix-vector multiplication
    */
   for(int i = 0; i < m; i++){
    c[i] = 0.0f;
   }
   for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
        c[i] += A[i * n + j] * b[j];
    }
   }
   return;
}

float NormMatrix(float *A, int m, int n){
    /*
    C function to calculate the standard Norm-2 of a matrix
    */
    float sum = 0.0f;
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            sum += A[i * n + j] * A[i * n + j];
        }
    }
    return sqrtf(sum);
}

float NormVector(float *a, int n){
    /*
    C-function to calculate the standard Norm-2 of a vector
    */
   float sum = 0.0f;
   for(int i = 0; i < n; i++){
    sum += a[i] * a[i];
   }
   return sqrtf(sum);
}


void normalize_vector(int n, float *b){
    /*
    Alternative normalization, this time to normalize to 1
    */
    float norm = NormVector(b, n);
    for(int i = 0; i < n; i++){
        b[i] /= norm;
    }
    return;
}



float MatrixDeviation(int m, int n, float *A, float *B)
{
    /*
        Calculates the deviation of matrix A from B as
        dev = || A - B ||
        The two matrices must share the same dimension m x n
    */
   float dev = 0.0f;
   for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
        float diff = A[i * n + j] - B[i * n + j];
        dev += diff * diff;
    }
   }
   return sqrtf(dev);
}

void Row2ColMajor(int m, int n, float *A_rm, float *A_cm){
    // This function converts a matrix from row- to column-major ordering
    // This additional hoop has to be jumped through because of the version
    // of LAPACK available on the Jetson Nano
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            A_cm[j * m + i] = A_rm[i * n + j];
        }
    }
    return;
}

void Col2RowMajor(int m, int n, float *A_cm, float *A_rm){
    // This function converts a matrix from row- to column-major ordering
    // This additional hoop has to be jumped through because I didn't have
    // the foresight to write every function in column major
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            A_rm[i * n + j] = A_cm[j * m + i];
        }
    }
    return;
}
