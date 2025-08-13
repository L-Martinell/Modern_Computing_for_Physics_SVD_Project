#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <string.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "matrix_utility.h"

#define BLOCK_ROWS 8
#define TILE_SIZE 32

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)


#define CUDA_CHECK_MSG(call, msg) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      printf("CUDA error at %s:%d: %s | %s\n", __FILE__, __LINE__, cudaGetErrorString(err), msg); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

void checkCuda(cudaError_t result){
    if(result != cudaSuccess){
        printf("Cuda runtime error: %s\n", cudaGetErrorString(result));
        exit(-1);
    }
}

void checkCuSolver(cusolverStatus_t status){
    if(status != CUSOLVER_STATUS_SUCCESS){
        printf("cuSolver error\n");
        exit(-1);
    }
}


__global__ void MatrixTransposeKernel(int m, int n, float *A, float *T)
{
    // CUDA kernel to transpose a matrix. Useful bc cusolverDn expects a column-major indexing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < m && col < n)
    {
        T[col * m + row] = A[row * n + col];
    }
}

__global__ void MatrixTransposeSHARED(int m, int n, float *A, float *T)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // avoid bank conflicts

    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Read into shared memory
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        int y = row + i;
        if (col < n && y < m && (threadIdx.y + i) < TILE_SIZE)
        {
            tile[threadIdx.y + i][threadIdx.x] = A[y * n + col];
        }
    }

    __syncthreads();

    // Transpose and write to global memory
    col = blockIdx.y * TILE_SIZE + threadIdx.x; // transposed block offset
    row = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        int y = row + i;
        if (col < m && y < n && (threadIdx.y + i) < TILE_SIZE)
        {
            T[y * m + col] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}


__global__ void BuildSigmaKernel(int m, int n, int num_SV, float *lambda_list, float *Sigma, bool Eigenvalues){
    // Builds the matrix Sigma starting from the eigenvalues of A^T A or the singular values of A
    // NOTE -- This kernel works best with a 1D block, since it accesses data from a vector
    // Additionally, it requires that the matrix Sigma that is passed to it is initialized to 0's
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_SV){
        if(Eigenvalues){
            Sigma[index * n + index] = sqrtf(lambda_list[index]);
        }
        else{
            Sigma[index * n + index] = lambda_list[index];
        }
    }
}

__global__ void MatrixMultiplicationSHARED(int m, int n, int k, float *A, float *B, float *C){
    // Multiplies two matrices A and B, not necessarily square, leveraging on shared memory

    // Declare shared memory
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    // Tile rows and columns
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Global rows and columns
    int globalRow = blockIdx.y * TILE_SIZE + row;
    int globalCol = blockIdx.x * TILE_SIZE + col;

    // Accumulator for values in matrix C
    float Cval = 0.0f;

    // Loop over the tiles in the "k" ("contracted") dimension
    int num_tiles_k = (k + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over the tiles
    for(int i = 0; i < num_tiles_k; i++){
        int A_idx = i * TILE_SIZE + col, B_idx = i * TILE_SIZE + row;

        // Load tile_A into shared memory
        if(globalRow < m && A_idx < k){
            tile_A[row][col] = A[globalRow * k + A_idx];
        }
        else{       // Protects from tile sizes not being exact multiples of the original matrix's size
            tile_A[row][col] = 0.0f;
        }

        // Load tile_B into shared memory
        if(B_idx < k && globalCol < n){
            tile_B[row][col] = B[B_idx * n + globalCol];
        }
        else{
            tile_B[row][col] = 0.0f;
        }

        __syncthreads();

        // Compute the partial results
        for(int j = 0; j < TILE_SIZE; j++){
            Cval += tile_A[row][j] * tile_B[j][col];
        }
        __syncthreads();
    }
    // Write results back to global memory
    if(globalRow < m && globalCol < n){
        C[globalRow * n + globalCol] = Cval;
    }
    return;
}


__global__ void MatrixSubtractionSquaredKernel(int m, int n, float *A, float *B, float *C){
    // This kernel calculates the square of the elementwise difference of two matrices A and B
    // C = (A - B)^2

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < m && col < n){
        float diff = A[row * n + col] - B[row * n + col];
        C[row * n + col] = diff * diff;
    }
}

__global__ void MatrixReductionKernel(int m, int n, float *A, float *R){
    // This kernel performs the reduction of the matrix A

    extern __shared__ float partial_sum[];

    int tidx  = threadIdx.x;
    int bidx  = blockIdx.x;
    int bdim  = blockDim.x;
    int index = 2 * bdim * bidx;

    if(index >= m * n) return;

    partial_sum[tidx] = A[index + tidx];
    partial_sum[tidx + bdim] = A[index + tidx + bdim];

    for(int stride = bdim; stride > 0; stride /= 2){
        __syncthreads();
        if(tidx < stride){
            partial_sum[tidx] += partial_sum[tidx + stride];
        }
    }
    __syncthreads();
    if(tidx == 0){
        R[bidx] = partial_sum[0];
    }
}

int main()
{
    srand(13);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int m = 4096, n = 2048;
    int num_SV = (m < n) ? m : n;
    float *A     = (float*)malloc(sizeof(float) * m * n);

    generate_matrix(m, n, A);

    // Find the singular values of A, as well as the left and right matrices U and VT
    // Define the vector to keep the singular values, as well as the matrices U and VT
    float *s  = (float*)malloc(sizeof(float) * n);

    // Define the device quantities
    float *dA, *ds, *dU, *dVT;                                          // Row-major ordered matrices (useful for following calculations)
    // float *dA_cm;
    float *dU_cm, *dVT_cm;                                              // Column-major ordered matrices
    float *dA_rec;
    float *dSigma;                                                      // Singular value matrix
    float *dUSigma;

    // Allocate memory in the device (GPU)
    checkCuda(cudaMalloc((void **)&dA, sizeof(float) * m * n));
    checkCuda(cudaMalloc((void **)&ds, sizeof(float) * n));
    checkCuda(cudaMalloc((void **)&dU, sizeof(float) * m * m));
    checkCuda(cudaMalloc((void **)&dVT, sizeof(float)* n * n));
    checkCuda(cudaMalloc((void **)&dA_rec, sizeof(float) * m * n));
    checkCuda(cudaMalloc((void **)&dU_cm, sizeof(float) * m * m));
    checkCuda(cudaMalloc((void **)&dVT_cm, sizeof(float)* n * n));
    checkCuda(cudaMalloc((void **)&dSigma, sizeof(float) * m * n));
    checkCuda(cudaMalloc((void **)&dUSigma, sizeof(float) * m * n));

    // Transfer the data from the host to the device
    checkCuda(cudaMemcpy(dA, A, sizeof(float) * m * n, cudaMemcpyHostToDevice));

    // Define the cuSolver object and create it
    cusolverDnHandle_t solver = NULL;
    checkCuSolver(cusolverDnCreate(&solver));

    // Define the worker quantities in the device
    int lwork = 0;
    cusolverDnSgesvd_bufferSize(solver, m, n, &lwork);
    float *dWork;
    cudaMalloc((void **)&dWork, sizeof(float) * lwork);

    int *dInfo;
    cudaMalloc((void **)&dInfo, sizeof(int));

    // ######################################################## CUDA HEAVY STUFF ########################################################
    // Now transpose the matrix A
    float *dAT;
    cudaMalloc((void **)&dAT, sizeof(float) * n * m);

    dim3 blockSize(32, 32);
    dim3 gridSize(n / blockSize.x, m / blockSize.y);

    dim3 blockSize_SHARED(TILE_SIZE, BLOCK_ROWS);
    dim3 gridSize_A( ( n + TILE_SIZE - 1 ) / TILE_SIZE,
                     ( m + TILE_SIZE - 1 ) / TILE_SIZE);

    // Start measuring time from here!
    cudaEventRecord(start);

    MatrixTransposeSHARED<<<gridSize_A, blockSize_SHARED>>>(m, n, dA, dAT);

    // Now we can use dAT as the matrix to solve with cuSolver
    cusolverStatus_t status = cusolverDnSgesvd(
                                                solver,
                                                'A', 'A',
                                                m, n,
                                                dAT, m,
                                                ds,
                                                dU_cm, m,
                                                dVT_cm, n,
                                                dWork, lwork,
                                                NULL,
                                                dInfo);
    int info_gpu = 0;
    cudaMemcpy(&info_gpu, dInfo, sizeof(int), cudaMemcpyDeviceToHost);
    if(info_gpu != 0){
        printf("SVD failed, info = %d\n", info_gpu);
        exit(-1);
    }

    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cusolverDnSgesvd failed with status %d\n", status);
    }

    // ############################ RECONSTRUCT MATRIX A ############################
    // A_rec = U * Sigma * VT

    // ############## BUILD MATRIX SIGMA ##############
    // Initialize dSigma to have all 0s
    int sigmaGridSize = 8;
    int sigmaThreadsPerBlock = num_SV / sigmaGridSize;

    checkCuda(cudaMemset(dSigma, 0.0f, sizeof(float) * m * n));
    BuildSigmaKernel<<<sigmaGridSize, sigmaThreadsPerBlock>>>(m, n, num_SV, ds, dSigma, false);
    CUDA_CHECK_MSG(cudaGetLastError(), "after BuildSigmaKernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ############## FROBENIUS NORM ##############
    // ####### TRANSPOSE MATRICES #######
    dim3 gridSize_mm((m + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
    dim3 gridSize_nn((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    MatrixTransposeSHARED<<<gridSize_mm, blockSize>>>(m, m, dU_cm, dU);
    MatrixTransposeSHARED<<<gridSize_nn, blockSize>>>(n, n, dVT_cm, dVT);
    
    // ####### MULTIPLY MATRICES #######
    // U * Sigma = USigma first, then USigma * VT = A_rec
    dim3 blockSize2(TILE_SIZE, TILE_SIZE);
    dim3 gridSize2((n + blockSize2.x - 1) / blockSize2.x, (m + blockSize2.y - 1) / blockSize2.y);
    MatrixMultiplicationSHARED<<<gridSize2, blockSize>>>(m, n, m, dU, dSigma, dUSigma);
    CUDA_CHECK_MSG(cudaGetLastError(), "after MatrixMultiplicationKernel (U * Sigma)");
    CUDA_CHECK(cudaDeviceSynchronize());

    MatrixMultiplicationSHARED<<<gridSize2, blockSize>>>(m, n, n, dUSigma, dVT, dA_rec);
    CUDA_CHECK_MSG(cudaGetLastError(), "after MatrixMultiplicationKernel (USigma * VT)");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ####### SUBTRACTION SQUARED #######
    float *Red = (float *)malloc(sizeof(float) * n);
    float *dDiff, *dRed;
    cudaMalloc((void**)&dDiff, sizeof(float) * m * n);
    cudaMalloc((void**)&dRed, sizeof(float) * n);

    MatrixSubtractionSquaredKernel<<<gridSize, blockSize>>>(m, n, dA, dA_rec, dDiff);
    CUDA_CHECK_MSG(cudaGetLastError(), "after MatrixSubtractionSquaredKernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    // ####### REDUCTION #######
    int reductionThreadsPerBlock = m / 4, reductionGridSize = 4 * n;                // Make sure to cover the whole matrix while also not exceeding limitations
    MatrixReductionKernel<<<reductionGridSize, reductionThreadsPerBlock, 2 * reductionThreadsPerBlock * sizeof(float)>>>(m, n, dDiff, dRed);
    CUDA_CHECK_MSG(cudaGetLastError(), "after MatrixReductionKernel");
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stop recording time here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Fetch the data back to the host
    checkCuda(cudaMemcpy(s, ds, sizeof(float) * n, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(Red, dRed, sizeof(float) * n, cudaMemcpyDeviceToHost));

    float FNorm = 0.0f;
    for(int i = 0; i < n; i++){
        FNorm += Red[i];
    }
    FNorm = sqrtf(FNorm);
    printf("Frobenius norm: %f\n", FNorm);

    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Elapsed time: %f s\n", elapsed_time/1000.0);
    printf("Matrix size: %d x %d\n", m, n);

    /*
    FILE *benchmark = fopen("benchmarking_float.txt", "a");
    fprintf(benchmark, "Matrix size: %d x %d, time elapsed (SVD GPU float, sharedMem): %.10lf s; ", m, n, elapsed_time/1000.0);
    fprintf(benchmark, "Tile size: %d; ", TILE_SIZE);
    fprintf(benchmark, "MatrixTranspositionKernel (matrix A): gridSize = (%d, %d), blockSize = (%d, %d); ", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    fprintf(benchmark, "MatrixTranspositionKernel (matrix U): gridSize = (%d, %d), blockSize = (%d, %d); ", gridSize_mm.x, gridSize_mm.y, blockSize.x, blockSize.y);
    fprintf(benchmark, "MatrixTranspositionKernel (matrix V): gridSize = (%d, %d), blockSize = (%d, %d); ", gridSize_nn.x, gridSize_nn.y, blockSize.x, blockSize.y);
    fprintf(benchmark, "MatrixMultiplicationKernel: gridSize = (%d, %d), blockSize = (%d, %d); ", gridSize2.x, gridSize2.y, blockSize.x, blockSize.y);
    fprintf(benchmark, "MatrixReductionKernel: gridSize = %d, blockSize = %d, shared Memory size = %ld", reductionGridSize, reductionThreadsPerBlock, 2 * reductionThreadsPerBlock * sizeof(float));
    fprintf(benchmark, "\n");
    */

    // Free the memory
    free(A);
    free(s);
    free(Red);

    cudaFree(dA);
    cudaFree(ds);
    cudaFree(dU);
    cudaFree(dVT);
    cudaFree(dWork);
    cudaFree(dInfo);
    cudaFree(dSigma);
    cudaFree(dA_rec);
    cudaFree(dU_cm);
    cudaFree(dVT_cm);
    cudaFree(dUSigma);
    cudaFree(dDiff);
    cudaFree(dRed);
    cudaFree(dAT);

    cusolverDnDestroy(solver);

    return 0;
}