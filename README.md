# Modern_Computing_for_Physics_SVD_Project
Final project for the course Modern Computing for Physics, part of the master's degree cycle Physics of Data delivered at UNIPD

## Outline
The project's goal is to develop a pipeline that allows to carry out a singular value decomposition (SVD) on non-square matrices of varying sizes (256x128, 512x256, 1024x512, 2048x1024, and 4096x2048) leveraging both on a naïve CPU implementation and a more optimized GPU implementation that uses a combination of some custom CUDA kernels to perform some basic matrix operations, as well as the LAPACK implementation of the SVD algorithm. The folder float contains the following files:
- matrix_utility.h: A series of small, helper functions that perform some basic matrix operations (such as matrix-matrix multiplication, generating random matrices, transposition, etc...)
- power_iteration.h: A series of functions that are used to find the singular values and, in general, carry out the deflation algorithm on CPU 
- svd_cpu_float.c: SVD carried out exclusively on CPU. Tested for 256x128, 512x256, and 1024x512 matrices
- svd_cpu_float_big.c: SVD carried out exclusively on CPU. Tested for 2048x1024 matrices. The main difference compared to the previous code is that it doesn't use a custom implementation of the SVD
- svd_cuda_float.cu: SVD carried out on GPU. Tested for 256x128, 512x256, and 1024x512 matrices
- svd_cuda_float_big.cu: SVD carried out on GPU. Tested for 2048x1024 and 4096x2048 matrices
- svd_cuda_float_sharedMem.cu: SVD carried out on GPU. Tested for 256x128, 512x256, and 1024x512 matrices. Leverages more on shared memory than the other file (svd_cuda_float.cu).
- svs_cuda_float_big_sharedMem.cu: SVD carried out on GPU. Tested for 2048x1024 and 4096x2048 matrices. Leverages more on shared memory than the other file (svd_cuda_float_big.cu).
- float_benchmark.txt: The collection of the time taken to run the codes. Optimized for a human reading, so doesn't follow the classic .csv structure and could use some polish in that regard
Analogous files have been produced to use double precision floating point numbers and are contained in the double subfolder.

