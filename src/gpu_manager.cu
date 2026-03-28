/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */



#include "gpu_manager.hpp"
#include "int1e.hpp"
#include "int2e.hpp"
#include "utils.hpp" // THROW_EXCEPTION
#include "int2c2e.hpp"
#include "int3c2e.hpp"
#include "int2e_direct.hpp"
#include "rys_eri_direct.hpp"
#include "device_host_memory.hpp" // For tracked_gansu::tracked_cudaMalloc/tracked_cudaFree
#include "gpu_kernels.hpp"
#include "rys_eri.hpp"
#include "rys_eri_mp2.hpp"

#include "gradients.hpp"
#include "rys_hessian_g.hpp"

#include <vector>    // std::vector
#include <tuple>     // std::tuple
#include <algorithm> // std::reverse, std::sort
#include <numeric>   // std::iota
#include <fstream>

#include <cuda_runtime.h> // for int2 type
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace gansu::gpu{






/**
 * @brief Performs eigenvalue decomposition on a symmetric matrix.
 * 
 * This function computes the eigenvalues and eigenvectors of the matrix, using the cuSOLVER library.
 *
 * @param d_matrix Device pointer to the input symmetric matrix.
 * @param d_eigenvalues Device pointer to store the eigenvalues.
 * @param d_eigenvectors Device pointer to store the eigenvectors.
 * @param size Size of the matrix (size x size).
 * @return Error status (0 if successful).
 * @details Since the eigenvectors are stored in the same memory as the input matrix, the input matrix is copied to a temporary matrix before.
 */
int eigenDecomposition(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size) {
    //cusolverManager cusolver;
    cusolverDnHandle_t cusolverHandle = GPUHandle::cusolver();
    cusolverDnParams_t cusolverParams = GPUHandle::cusolverParams();

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;
    real_t* d_workspace=nullptr;
    real_t* h_workspace=nullptr;

    cudaError_t err;
    
    // Query the workspace sizes of the device and host memory
    cusolverDnXsyevd_bufferSize(
        cusolverHandle, cusolverParams,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        size, CUDA_R_64F, d_eigenvalues, 
        size, CUDA_R_64F, d_workspace, 
        CUDA_R_64F,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost
    );
    // workspace allocation
    err = gansu::tracked_cudaMalloc(&d_workspace, workspaceInBytesOnDevice);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMallocHost(&h_workspace, workspaceInBytesOnHost);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate host memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // allocate return value for the error status        
    int* d_info;
    err = gansu::tracked_cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for error status: ") + std::string(cudaGetErrorString(err)));
    }

    // temporary matrix allocation for d_matrix since the eigenvectors will be stored in the same memory of d_matrix
    real_t* d_temp_matrix;
    err = gansu::tracked_cudaMalloc(&d_temp_matrix, size * size * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }

    
    // copy the d_matrix since the eigenvectors will be stored in the same memory
    cudaMemcpy(d_temp_matrix, d_matrix, size * size * sizeof(real_t), cudaMemcpyDeviceToDevice);
    
    // Perform eigenvalue decomposition
    cusolverDnXsyevd(
        cusolverHandle, cusolverParams,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER,
        size, CUDA_R_64F, d_temp_matrix, 
        size, CUDA_R_64F, d_eigenvalues, 
        CUDA_R_64F,
        d_workspace, workspaceInBytesOnDevice,
        h_workspace, workspaceInBytesOnHost,
        d_info
    );
    
    // Copy the eigenvectors to d_eigenvectors
    cudaMemcpy(d_eigenvectors, d_temp_matrix, size * size * sizeof(real_t), cudaMemcpyDeviceToDevice);
    
    // transpose the eigenvectors since the eigenvectors are stored by column-major order
    transposeMatrixInPlace(d_eigenvectors, size);
    
    // return the error status
    int h_info;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);


    // free the temporary memory
    gansu::tracked_cudaFree(d_temp_matrix);
    gansu::tracked_cudaFree(d_workspace);
    cudaFreeHost(h_workspace);
    gansu::tracked_cudaFree(d_info);

    return h_info; // 0 if successful
}

// LAPACK dgeev: non-symmetric eigenvalue decomposition (Fortran interface)
extern "C" {
    void dgeev_(const char* jobvl, const char* jobvr, const int* n,
                double* a, const int* lda, double* wr, double* wi,
                double* vl, const int* ldvl, double* vr, const int* ldvr,
                double* work, const int* lwork, int* info);
}

/**
 * @brief Eigenvalue decomposition of a general (non-symmetric) matrix
 *
 * Uses LAPACK dgeev on CPU (subspace matrices are small, typically ≤30×30).
 * Eigenvalues are sorted by real part (ascending).
 * Eigenvectors stored in the same transposed layout as eigenDecomposition
 * (eigenvector i at stride=size, offset=i).
 *
 * @param d_matrix Input matrix (column-major, n×n) — not modified
 * @param d_eigenvalues Output real parts of eigenvalues (sorted ascending) [size]
 * @param d_eigenvectors Output right eigenvectors [size × size], transposed layout
 * @param size Matrix dimension
 * @return 0 if successful
 */
int eigenDecompositionNonSymmetric(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size) {
    // Copy matrix to host (dgeev overwrites input)
    std::vector<real_t> h_A((size_t)size * size);
    cudaMemcpy(h_A.data(), d_matrix, (size_t)size * size * sizeof(real_t), cudaMemcpyDeviceToHost);

    // LAPACK output arrays
    std::vector<real_t> h_WR(size);  // real parts of eigenvalues
    std::vector<real_t> h_WI(size);  // imaginary parts of eigenvalues
    std::vector<real_t> h_VR((size_t)size * size);  // right eigenvectors

    // Query optimal workspace size
    int n = size;
    int lda = size;
    int ldvr = size;
    int ldvl = 1;
    int lwork = -1;
    int info = 0;
    real_t work_query;

    dgeev_("N", "V", &n,
           h_A.data(), &lda, h_WR.data(), h_WI.data(),
           nullptr, &ldvl, h_VR.data(), &ldvr,
           &work_query, &lwork, &info);

    lwork = (int)work_query;
    std::vector<real_t> h_work(lwork);

    // Solve
    dgeev_("N", "V", &n,
           h_A.data(), &lda, h_WR.data(), h_WI.data(),
           nullptr, &ldvl, h_VR.data(), &ldvr,
           h_work.data(), &lwork, &info);

    if (info != 0) {
        return info;
    }

    // Filter: keep only eigenvalues with negligible imaginary part
    // Complex eigenvalues from dgeev come in conjugate pairs; skip both
    double imag_threshold = 1e-6;
    std::vector<int> real_indices;
    real_indices.reserve(size);
    for (int i = 0; i < size; i++) {
        if (std::abs(h_WI[i]) < imag_threshold) {
            real_indices.push_back(i);
        }
    }

    // Sort real eigenvalues by real part (ascending)
    std::sort(real_indices.begin(), real_indices.end(),
              [&h_WR](int a, int b) { return h_WR[a] < h_WR[b]; });

    int n_real = (int)real_indices.size();

    // Build sorted eigenvalue array and reordered eigenvector matrix
    // Output layout matches eigenDecomposition: transposed, eigenvector i at stride=size, offset=i
    std::vector<real_t> h_sorted_evals(size, 1e30);  // fill unused slots with large value
    std::vector<real_t> h_sorted_evecs((size_t)size * size, 0.0);

    for (int i = 0; i < n_real; i++) {
        int orig = real_indices[i];
        h_sorted_evals[i] = h_WR[orig];
        // h_VR column-major: column orig = eigenvector orig
        // Transposed output: row i = eigenvector i
        // Column-major element (i, j) = h_sorted_evecs[i + j*size]
        for (int j = 0; j < size; j++) {
            h_sorted_evecs[i + j * size] = h_VR[j + orig * size];
        }
    }

    // Copy sorted results to device
    cudaMemcpy(d_eigenvalues, h_sorted_evals.data(),
               size * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eigenvectors, h_sorted_evecs.data(),
               (size_t)size * size * sizeof(real_t), cudaMemcpyHostToDevice);

    return 0;
}

/**
 * @brief Compute only the k smallest eigenvalues/eigenvectors of a symmetric matrix
 *
 * Uses cusolverDnXsyevdx with index range [1, num_eigenvalues].
 * Much faster than full eigenDecomposition when k << n.
 *
 * @param d_matrix Input symmetric matrix (column-major, n×n)
 * @param d_eigenvalues Output eigenvalues (size: num_eigenvalues)
 * @param d_eigenvectors Output eigenvectors (row-major after transpose, n×n buffer required)
 * @param size Matrix dimension n
 * @param num_eigenvalues Number of smallest eigenvalues to compute (k)
 * @return 0 if successful
 */
int partialEigenDecomposition(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size, const int num_eigenvalues) {
    cusolverDnHandle_t cusolverHandle = GPUHandle::cusolver();
    cusolverDnParams_t cusolverParams = GPUHandle::cusolverParams();

    size_t workspaceInBytesOnDevice;
    size_t workspaceInBytesOnHost;
    real_t* d_workspace = nullptr;
    real_t* h_workspace = nullptr;

    // Copy input matrix (will be overwritten with eigenvectors)
    real_t* d_temp_matrix;
    gansu::tracked_cudaMalloc(&d_temp_matrix, (size_t)size * size * sizeof(real_t));
    cudaMemcpy(d_temp_matrix, d_matrix, (size_t)size * size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Index range: 1-based, compute eigenvalues il..iu
    int64_t il = 1;
    int64_t iu = num_eigenvalues;
    real_t vl = 0.0, vu = 0.0;  // unused for RANGE_I
    int64_t h_meig = 0;

    // Query workspace size
    cusolverDnXsyevdx_bufferSize(
        cusolverHandle, cusolverParams,
        CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER,
        size, CUDA_R_64F, d_temp_matrix,
        size, &vl, &vu, il, iu, &h_meig,
        CUDA_R_64F, d_eigenvalues,
        CUDA_R_64F,
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost
    );

    gansu::tracked_cudaMalloc(&d_workspace, workspaceInBytesOnDevice);
    cudaMallocHost(&h_workspace, workspaceInBytesOnHost);

    int* d_info;
    gansu::tracked_cudaMalloc(&d_info, sizeof(int));

    // Compute partial eigendecomposition
    cusolverDnXsyevdx(
        cusolverHandle, cusolverParams,
        CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_UPPER,
        size, CUDA_R_64F, d_temp_matrix,
        size, &vl, &vu, il, iu, &h_meig,
        CUDA_R_64F, d_eigenvalues,
        CUDA_R_64F,
        d_workspace, workspaceInBytesOnDevice,
        h_workspace, workspaceInBytesOnHost,
        d_info
    );

    // Copy eigenvectors: first h_meig columns of d_temp_matrix → d_eigenvectors
    // Store as full n×n to maintain same layout as eigenDecomposition
    cudaMemcpy(d_eigenvectors, d_temp_matrix, (size_t)size * size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Transpose eigenvectors (same convention as eigenDecomposition)
    transposeMatrixInPlace(d_eigenvectors, size);

    int h_info_val;
    cudaMemcpy(&h_info_val, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    gansu::tracked_cudaFree(d_temp_matrix);
    gansu::tracked_cudaFree(d_workspace);
    cudaFreeHost(h_workspace);
    gansu::tracked_cudaFree(d_info);

    return h_info_val;
}

/**
 * @brief Computes the product of a matrix and a matrix using cuBLAS.
 * 
 * This function computes the product of a matrix and a matrix using cuBLAS.
 * 
 * @param d_matrix_A Device pointer to the N x N matrix stored by row-major order
 * @param d_matrix_B Device pointer to the N x N matrix stored by row-major order
 * @param d_matrix_C Device pointer to store the result stored by row-major order
 * @param size Size of the matrix (size x size)
 * @param transpose_A Flag to transpose matrix A, default is false
 * @param transpose_B Flag to transpose matrix B, default is false
 * @param initialize_C_to_zero Flag to initialize the matrix C to zero before the computation, default is true. If false, the matrix C is added to the product.
 * @details The matrix product is computed as \f$ C += AB \f$.
 * @details If the flag initialize_C_to_zero is true, the matrix C is initialized to zero before the computation.
 */
 void matrixMatrixProduct(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size, const bool transpose_A, const bool transpose_B, const bool accumulate){
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    const double alpha = 1.0;
    double beta = 0.0;

    if (!accumulate){
        cudaMemset(d_matrix_C, 0, size * size * sizeof(double));
        // beta = 0.0 for initialization
        beta = 0.0; // redundant, but for clarity
    }else{
        // beta = 1.0 for accumulation
        beta = 1.0;
    }

    const cublasOperation_t transA = (transpose_A) ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t transB = (transpose_B) ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemm(
        cublasHandle, 
        transB, transA, 
        size, size, size, 
        &alpha, 
        d_matrix_B, size, 
        d_matrix_A, size, 
        &beta, 
        d_matrix_C, size
    );

}

/**
 * @brief Computes the product of two rectangular matrices using cuBLAS.
 *
 * Computes C = alpha * op(A) * op(B) + beta * C
 * where op(X) = X if transpose=false, X^T if transpose=true.
 *
 * @param d_A  Row-major matrix. Without transpose: M×K. With transpose: K×M.
 * @param d_B  Row-major matrix. Without transpose: K×N. With transpose: N×K.
 * @param d_C  Row-major output matrix M×N.
 * @param M    Number of rows of op(A) and C.
 * @param N    Number of columns of op(B) and C.
 * @param K    Number of columns of op(A) / rows of op(B).
 * @param transpose_A  If true, A is K×M and transposed.
 * @param transpose_B  If true, B is N×K and transposed.
 * @param accumulate   If true, C += alpha*op(A)*op(B). If false, C = alpha*op(A)*op(B).
 * @param alpha        Scalar multiplier (default 1.0).
 */
void matrixMatrixProductRect(const double* d_A, const double* d_B, double* d_C,
                              const int M, const int N, const int K,
                              const bool transpose_A, const bool transpose_B,
                              const bool accumulate, const double alpha)
{
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    double beta = accumulate ? 1.0 : 0.0;

    // cuBLAS uses column-major. For row-major A(M×K), cuBLAS sees it as
    // column-major A'(K×M). So we compute C' = B' * A' to get row-major C.
    // Row-major op(A)[M×K]: stored as M×K. cuBLAS sees K×M column-major.
    // Row-major op(B)[K×N]: stored as K×N. cuBLAS sees N×K column-major.
    // We want row-major C[M×N]: cuBLAS sees N×M column-major.
    // cuBLAS call: C' = B' * A' → (N×M) = (N×K) * (K×M)

    // Leading dimensions in memory (row-major storage):
    // A stored as (transpose_A ? K×M : M×K) → lda = (transpose_A ? M : K)
    // B stored as (transpose_B ? N×K : K×N) → ldb = (transpose_B ? K : N)
    const int lda = transpose_A ? M : K;  // physical row width of A
    const int ldb = transpose_B ? K : N;  // physical row width of B
    const int ldc = N;                     // physical row width of C

    // cuBLAS sees row-major as column-major transposed.
    // For row-major A[M×K] without transpose:
    //   cuBLAS sees col-major A'[K×M], which is A^T in cuBLAS notation → CUBLAS_OP_T to undo
    //   But we need op(A)=A, so cuBLAS must apply T to get A from A'.
    // For row-major A[K×M] with transpose:
    //   cuBLAS sees col-major A'[M×K], which is A^T in cuBLAS notation → CUBLAS_OP_N gives A^T
    //
    // Simpler approach: C = op(A)*op(B) in row-major
    // ↔ C^T = op(B)^T * op(A)^T in col-major
    // ↔ cuBLAS(C^T[N×M]) = cuBLAS_op_B * cuBLAS_op_A
    //
    // For cuBLAS: we call cublasDgemm with:
    //   op_cuB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N  (for B in C^T = B^T * A^T)
    //   op_cuA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N  (for A in C^T = B^T * A^T)
    //   But row-major→col-major already gives transpose, so we need to invert:
    //   cuBLAS_opB for B^T in col-major: if transpose_B=false, B is K×N row-major = N×K col-major,
    //     we need B^T[N×K] col-major → CUBLAS_OP_T on the N×K col-major → gives K×N which is wrong.
    //
    // Let me use the standard row-major trick directly:
    // C[M×N] = op(A)[M×K] * op(B)[K×N]  in row-major
    // Equivalent to: C_col[N×M] = op(B)_col[N×K] * op(A)_col[K×M] in col-major
    //
    // A is stored row-major. Without transpose: shape M×K, col-major view: K×M.
    //   op(A) in row-major = A[M×K]. In col-major: we see A'[K×M].
    //   For op(A)_col[K×M] we need CUBLAS_OP_N on A' → gives K×M. ✓
    //   → cuBLAS_opA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N
    //
    // Similarly: cuBLAS_opB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N

    const cublasOperation_t cuOpB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t cuOpA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;

    // cublasDgemm(handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc)
    cublasDgemm(
        cublasHandle,
        cuOpB, cuOpA,
        N, M, K,
        &alpha,
        d_B, ldb,
        d_A, lda,
        &beta,
        d_C, ldc
    );
}

/**
 * @brief Strided batched DGEMM for row-major matrices.
 *
 * Computes C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i] for i=0..batchCount-1,
 * where A[i] = d_A + i*strideA, B[i] = d_B + i*strideB, C[i] = d_C + i*strideC.
 * Set strideA=0 to broadcast the same A matrix across all batches.
 */
void matrixMatrixProductBatched(const double* d_A, const double* d_B, double* d_C,
                                 const int M, const int N, const int K,
                                 const long long strideA, const long long strideB, const long long strideC,
                                 const int batchCount,
                                 const bool transpose_A, const bool transpose_B,
                                 const bool accumulate, const double alpha)
{
    cublasHandle_t cublasHandle = GPUHandle::cublas();
    double beta = accumulate ? 1.0 : 0.0;

    const int lda = transpose_A ? M : K;
    const int ldb = transpose_B ? K : N;
    const int ldc = N;

    // Row-major → col-major: C^T = op(B)^T * op(A)^T
    const cublasOperation_t cuOpB = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t cuOpA = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublasDgemmStridedBatched(
        cublasHandle,
        cuOpB, cuOpA,
        N, M, K,
        &alpha,
        d_B, ldb, strideB,
        d_A, lda, strideA,
        &beta,
        d_C, ldc, strideC,
        batchCount
    );
}

/**
 * @brief Computes the weighted sum of two matrices using cuBLAS.
 * @param d_matrix_A Device pointer to the size x size matrix
 * @param d_matrix_B Device pointer to the size x size matrix
 * @param d_matrix_C Device pointer to store the result
 * @param weight_A Weight of the matrix A
 * @param weight_B Weight of the matrix B
 * @param size Size of the matrix (size x size)
 * @details The matrix weighted sum is computed as \f$ C = \alpha A + \beta B \f$.
 */
void weightedMatrixSum(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const double weight_A, const double weight_B, const int size) {
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    const double alpha = weight_A;
    const double beta = weight_B;

    cublasDgeam(
        cublasHandle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        size, size, 
        &alpha, d_matrix_A, size, 
        &beta, d_matrix_B, size, 
        d_matrix_C, size
    );
}

/**
* @brief Computes the addition of two matrices using cuBLAS.
* @param d_matrix_A Device pointer to the size x size matrix
* @param d_matrix_B Device pointer to the size x size matrix
* @param d_matrix_C Device pointer to store the result
* @param size Size of the matrix (size x size)
* @details The matrix subtraction is computed as \f$ C = A + B \f$.
*/
void matrixAddition(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size) {
   weightedMatrixSum(d_matrix_A, d_matrix_B, d_matrix_C, 1.0, 1.0, size);
}


/**
 * @brief Computes the subtraction of two matrices using cuBLAS.
 * @param d_matrix_A Device pointer to the size x size matrix
 * @param d_matrix_B Device pointer to the size x size matrix
 * @param d_matrix_C Device pointer to store the result
 * @param size Size of the matrix (size x size)
 * @details The matrix subtraction is computed as \f$ C = A - B \f$.
 */
void matrixSubtraction(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size) {
    weightedMatrixSum(d_matrix_A, d_matrix_B, d_matrix_C, 1.0, -1.0, size);
}

void matrixSubtractionInPlace(const double* d_matrix_A, double* d_matrix_B, double* d_matrix_C, const int size) {
    weightedMatrixSum(d_matrix_A, d_matrix_B, d_matrix_C, 1.0, -1.0, size);
}

/**
 * @brief Computes the inner product of two vectors using cuBLAS.
 * @param d_vector_A Device pointer to the vector A
 * @param d_vector_B Device pointer to the vector B
 * @param size Size of the vector
 * @return The inner product of the two vectors
 * @details The inner product is computed as \f$ result = \sum_{i=1}^{size} A_i B_i \f$.
 */
double innerProduct(const double* d_vector_A, const double* d_vector_B, const int size) {
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    double result;
    cublasDdot(
        cublasHandle, 
        size, 
        d_vector_A, 1, 
        d_vector_B, 1, 
        &result
    );
    return result;
}


/**
 * @brief Computes the inverse of the square root of the vector.
 * 
 * This function computes the inverse of the square root of each value of the vector.
 * 
 * @param d_vectors Device pointer to the vector.
 * @param size Number of the vector.
 */
void invertSqrtElements(real_t* d_vectors, const size_t size, const double threshold) {
    size_t blockSize = 256;
    size_t numBlocks = (size + blockSize - 1) / blockSize;
    inverseSqrt_kernel<<<numBlocks, blockSize>>>(d_vectors, size, threshold);
}

/**
 * @brief Computes the square root of the vector.
 * 
 * This function computes the root of each value of the vector.
 * 
 * @param d_vectors Device pointer to the vector.
 * @param size Number of the vector.
 */
void sqrtElements(real_t* d_vectors, const size_t size) {
    size_t blockSize = 256;
    size_t numBlocks = (size + blockSize - 1) / blockSize;
    sqrt_kernel<<<numBlocks, blockSize>>>(d_vectors, size);
}

/**
 * @brief Transpose a matrix in place.
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix (size x size)
 * @details This function transposes a matrix in place using shared memory.
 * @details The size of the matrix is size x size.
 */
 void transposeMatrixInPlace(real_t* d_matrix, const int size) {
    dim3 blockSize(WARP_SIZE, WARP_SIZE);
    dim3 gridSize((size + WARP_SIZE - 1) / WARP_SIZE, (size + WARP_SIZE - 1) / WARP_SIZE);
    transposeMatrixInPlace_kernel<<<gridSize, blockSize>>>(d_matrix, size);
}

/**
 * @brief Make a diagonal matrix from the vector.
 * @param d_vector Device pointer to the vector of size size.
 * @param d_matrix Device pointer to store the diagonal matrix of size size x size.
 * @param size Size of the vector and the matrix.
 * @details This function creates a diagonal matrix, in which the diagonal elements are the elements of the vector.
 */
void makeDiagonalMatrix(const real_t* d_vector, real_t* d_matrix, const int size) {
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    // Set the matrix to zero
    cudaMemset(d_matrix, 0, size * size * sizeof(real_t));
    // Set the diagonal elements to the eigenvalues
    cublasDcopy(cublasHandle, size, d_vector, 1, d_matrix, size+1);
}

/**
 * @brief Compute the trace of a matrix (the sum of the diagonal elements)
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix (size x size)
 * @return Trace of the matrix (the sum of the diagonal elements)
 */
 real_t computeMatrixTrace(const real_t* d_matrix, const int size) 
 {
    cudaError_t err;

    double* d_trace;
    err = gansu::tracked_cudaMalloc(&d_trace, sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for trace: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_trace, 0, sizeof(real_t));

    real_t h_trace = 0.0;
    
    const int num_threads_per_block = 1024;
    const int num_blocks = (size + num_threads_per_block - 1) / num_threads_per_block;
    getMatrixTrace<<<num_blocks, num_threads_per_block>>>(d_matrix, d_trace, size);
    cudaMemcpy(&h_trace, d_trace, sizeof(real_t), cudaMemcpyDeviceToHost);

    gansu::tracked_cudaFree(d_trace);
    return h_trace;
}





/**
 * @brief Compute Core Hamiltonian Matrix (one electron integrals)
 * @param shell_type_infos Information of the shell types
 * @param d_primitive_shells Device pointer to the primitive shells
 * @param d_boys_grid Device pointer to the grid values of the Boys function
 * @param d_cgto_normalization_factors Device pointer to the normalization factors of the contracted Gaussian-type orbitals
 * @param d_overlap_matrix Device pointer to the overlap matrix to store the result
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix to store the result
 * @param num_atoms Number of atoms
 * @param num_basis Number of basis functions
 * @details This function computes the core Hamiltonian matrix and the overlap matrix.
 */
void computeCoreHamiltonianMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, Atom* d_atoms, PrimitiveShell* d_primitive_shells, real_t* d_boys_grid, 
                                    real_t* d_cgto_normalization_factors, real_t* d_overlap_matrix, real_t* d_core_hamiltonian_matrix, const int num_atoms, const int num_basis, const std::string int1e_method, const bool verbose) {
    // compute the core Hamiltonian matrix
    const int threads_per_block = 128; // the number of threads per block

    const int shell_type_count = shell_type_infos.size();

    cudaMemset(d_overlap_matrix, 0, sizeof(real_t)*num_basis*num_basis);
    cudaMemset(d_core_hamiltonian_matrix, 0, sizeof(real_t)*num_basis*num_basis);

    
    // make multi stream
    const int N = (shell_type_count)*(shell_type_count+1) /2;
    std::vector<cudaStream_t> streams(N);
    std::vector<cudaStream_t> V_streams(N);

    for (int i = 0; i < N; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
        err = cudaStreamCreate(&V_streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }

    }

    // Call the kernel functions from (s0|s1),... (e.g. (f|f), (d|f), (d|d), (s|d), (p|d), (d|d) for s, p, d, f shells)
    for (int s0 = shell_type_count-1; s0 >= 0; s0--) {
        for (int s1 = shell_type_count-1; s1 >= s0; s1--) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];

            const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count); // the number of pairs of primitive shells = the number of threads
            const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block; // the number of blocks

            if(verbose){
                std::cout << "(" << shell_type_to_shell_name(s0) << "|" << shell_type_to_shell_name(s1) << "): ";
                std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
                std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
                std::cout << "|[a|b]|=" << num_shell_pairs << ", ";
                std::cout << "num_blocks: " << num_blocks << std::endl;
            }

            int index = (2*(shell_type_count-1)-s0+1)*s0 / 2 + s1;

            // call the kernel functions
            get_overlap_kinetic_kernel(s0, s1, int1e_method)<<<num_blocks, threads_per_block, 0, streams[index]>>>(d_overlap_matrix, d_core_hamiltonian_matrix, d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, num_shell_pairs, num_basis);
            get_nuclear_attraction_kernel(s0, s1, int1e_method)<<<num_blocks, threads_per_block, 0, V_streams[index]>>>(d_core_hamiltonian_matrix, d_primitive_shells, d_cgto_normalization_factors, d_atoms, num_atoms, shell_s0, shell_s1, num_shell_pairs, num_basis, d_boys_grid);
        }
    }
    // syncronize streams
    cudaDeviceSynchronize();

    const int num_blocks_sym = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    matrixSymmetrization<<<num_blocks_sym, threads_per_block>>>(d_overlap_matrix, num_basis);
    matrixSymmetrization<<<num_blocks_sym, threads_per_block>>>(d_core_hamiltonian_matrix, num_basis);

    // destory streams
    for (int i = 0; i < N; i++) {
        cudaStreamDestroy(streams[i]);
        cudaStreamDestroy(V_streams[i]);
    }

}

int get_index_2to1_horizontal(int i, int j, const int n)
{
    if (i > j) std::swap(i, j);
    return j - static_cast<int>(i * (i - 2 * n + 1) / 2);
}


size_t makeShellPairTypeInfo(const std::vector<ShellTypeInfo>& shell_type_infos, std::vector<ShellPairTypeInfo>& shell_pair_type_infos)
{
    // Make shell-pair type infos: 
    const int shell_type_count = shell_type_infos.size();

    size_t num_primitive_shell_pairs = 0;
    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        for (int s1 = s0; s1 < shell_type_count; ++s1) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const size_t num_bra = (s0 == s1) ? shell_s0.count * (shell_s0.count + 1) / 2 : shell_s0.count * shell_s1.count;
            shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)] = {num_bra, num_primitive_shell_pairs};
            num_primitive_shell_pairs += num_bra;
        }
    }

    return num_primitive_shell_pairs;
}

void computeERIMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,  real_t* d_eri_matrix, const real_t* d_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const bool verbose) {

    // Zero-initialize ERI matrix (kernels use atomicAdd)
    cudaMemset(d_eri_matrix, 0, (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));

    // compute the electron repulsion integrals
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_type_infos.size();

    // Call the kernel functions from (ss|ss),... (e.g. (ss|ss), (ss|sp), (ss|pp), (sp|sp), (sp|pp), (pp|pp) for s and p shells)

    // list shell-quadruples for sorted shell-type (s0, s1, s2, s3)
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    // reverse the order of the shell_quadruples to make it sorted by (s0, s1, s2, s3)
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());


    // make multi stream
    const int num_kernels = shell_quadruples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket; // equal to the number of threads
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block; // the number of blocks

        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;

        gpu::get_eri_kernel(s0, s1, s2, s3)<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_eri_matrix, d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, num_basis, d_boys_grid, head_bra, head_ket);
    
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2) << shell_type_to_shell_name(s3) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s3) << "|=" << shell_s1.count << ", ";
            std::cout << "|bra|= " << num_bra << ", " ;
            std::cout << "|ket|= " << num_ket << ", " ;
            std::cout << "|braket|= " << num_braket << ", " ;
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}


// ============================================================
//  Direct MP2: Half-transform AO ERI with MO coefficients
//  H(mu,nu,la,i) = sum_sigma (mu nu | la sigma) * C(sigma, i)
//  Uses RysERI_half_transform kernel for all shell types.
//  TODO: GPU最適化 — s/p特化カーネルを使えば高速化可能
// ============================================================
void computeHalfTransformedERI(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_boys_grid,
    const real_t* d_cgto_normalization_factors,
    real_t* d_half,
    const real_t* d_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const int num_basis,
    const real_t* d_C,
    int i_start,
    int block_occ)
{
    const int threads_per_block = 256;
    const int shell_type_count = shell_type_infos.size();

    // Generate shell quadruples (same as computeERIMatrix)
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a)
        for (int b = a; b < shell_type_count; ++b)
            for (int c = 0; c < shell_type_count; ++c)
                for (int d = c; d < shell_type_count; ++d)
                    if (a < c || (a == c && b <= d))
                        shell_quadruples.emplace_back(a, b, c, d);
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    const int num_kernels = shell_quadruples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; ++i) cudaStreamCreate(&streams[i]);

    int stream_id = 0;
    for (const auto& quadruple : shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo ss0 = shell_type_infos[s0];
        const ShellTypeInfo ss1 = shell_type_infos[s1];
        const ShellTypeInfo ss2 = shell_type_infos[s2];
        const ShellTypeInfo ss3 = shell_type_infos[s3];

        const size_t num_bra = (s0 == s1) ? ss0.count*(ss0.count+1)/2 : ss0.count*ss1.count;
        const size_t num_ket = (s2 == s3) ? ss2.count*(ss2.count+1)/2 : ss2.count*ss3.count;
        const size_t num_braket = ((s0 == s2) && (s1 == s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket;
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block;

        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;

        gpu::RysERI_half_transform<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
            d_half, d_C, i_start, block_occ,
            d_primitive_shells, d_cgto_normalization_factors,
            ss0, ss1, ss2, ss3,
            num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors,
            num_basis, d_boys_grid, head_bra, head_ket);
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < num_kernels; i++) cudaStreamDestroy(streams[i]);
}


/**
 * @brief Computes the coefficient matrix from the Fock matrix and the transformation matrix.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_coefficient_matrix Device pointer to store the coefficient matrix
 * @param num_basis Number of basis functions
 * @param d_orbital_energies Device pointer to store the orbital energies, default is nullptr. If nullptr, the orbital energies are stored in the temporary memory allcated inside, otherwise, the orbital energies are stored in the given device memory.
 * @details This function computes the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem \f$FC = SCE \f$.
 * @details To transform the generalized eigenvalue problem to the standard eigenvalue problem \f$FC = CE \f$, the transformation matrix.
 */
void computeCoefficientMatrix(const real_t* d_fock_matrix, const real_t* d_transform_matrix, real_t* d_coefficient_matrix, const int num_basis, real_t* d_orbital_energies) {
    // allocate temporary memory
    real_t* d_tempMatrix = nullptr;
    real_t* d_tempSymFockMatrix = nullptr;
    real_t* d_tempEigenvectors = nullptr;
    real_t* d_tempEigenvalues = nullptr; // if d_orbital_energies is nullptr, the eigenvalues are stored in d_tempEigenvalues

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempSymFockMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary symmetrized Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempEigenvectors, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary eigenvectors: ") + std::string(cudaGetErrorString(err)));
    }

    if (d_orbital_energies == nullptr){
        err = gansu::tracked_cudaMalloc(&d_tempEigenvalues, num_basis * sizeof(real_t));
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary eigenvalues: ") + std::string(cudaGetErrorString(err)));
        }
    } else {
        d_tempEigenvalues = d_orbital_energies;
    }


    // calculate the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem FC = SCE
    // symmetrize the Fock matrix F' = X^T F X
    // temp = X^T F
    matrixMatrixProduct(
        d_transform_matrix, d_fock_matrix, d_tempMatrix, num_basis,
        true, // transpose the transformation matrix X
        false
    );
    // F' = temp X
    matrixMatrixProduct(
        d_tempMatrix, d_transform_matrix, d_tempSymFockMatrix, num_basis,
        false, 
        false
    );

    // diagonalize the symmetrized Fock matrix F'C' = C'E
    eigenDecomposition(d_tempSymFockMatrix, d_tempEigenvalues, d_tempEigenvectors, num_basis);

    // obtain the coefficient matrix from the eigenvectors C = X C'
    matrixMatrixProduct(
        d_transform_matrix, d_tempEigenvectors, d_coefficient_matrix, num_basis,
        false, 
        false
    );

    // free the temporary memory
    gansu::tracked_cudaFree(d_tempMatrix);
    gansu::tracked_cudaFree(d_tempSymFockMatrix);
    gansu::tracked_cudaFree(d_tempEigenvectors);

    if (d_orbital_energies == nullptr){
        gansu::tracked_cudaFree(d_tempEigenvalues);
    }
}

/**
 * @brief Compute the density matrix for the restricted Hartree-Fock method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix Device pointer to store the density matrix
 * @param num_electron Number of electrons
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix, each of orbitals has zero or two electrons, using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals, two by two.
 * @details The density matrix is given by \f$ D_{\mu\nu} = 2 \sum_{i=1}^{N/2} C_{\mu i} C_{\nu i} \f$.
 */
void computeDensityMatrix_RHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    computeDensityMatrix_RHF_kernel<<<num_blocks, threads_per_block>>>(
        d_coefficient_matrix,
        d_density_matrix,
        num_electron,
        num_basis
    );
}


/**
 * @brief Compute the density matrix (alpha or beta spin) for the unrestricted Hartree-Fock method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix for the alpha spin or beta spin
 * @param d_density_matrix Device pointer to store the density matrix for the alpha spin or beta spin
 * @param num_spin Number of electrons for the alpha spin or beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix, each of orbitals has zero or one electoron using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals, one by one.
 * @details The density matrix is given by \f$ D_{\mu\nu} = \sum_{i=1}^{N} C_{\mu i} C_{\nu i} \f$.
 */
void computeDensityMatrix_UHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    computeDensityMatrix_UHF_kernel<<<num_blocks, threads_per_block>>>(
        d_coefficient_matrix,
        d_density_matrix,
        num_electron,
        num_basis
    );
}


/**
 * @brief Compute the density matrix (open- or closed-shell) for the ROHF method.
 * @param d_coefficient_matrix Device pointer to the coefficient matrix for the alpha spin or beta spin
 * @param d_density_matrix_closed Device pointer to store the density matrix for the closed-shell orbitals
 * @param d_density_matrix_open Device pointer to store the density matrix for the open-shell orbitals
 * @param d_density_matrix Device pointer to store the density matrix (sum of the closed- and open-shell orbitals)
 * @param num_closed Number of closed-shell orbitals
 * @param num_open Number of open-shell orbitals
 * @details This function computes the density matrix, each of orbitals has two (closed), one (open), zero (virtual) electoron using the coefficient matrix.
 * @details Electrons are allocated from the lowest energy orbitals.
 */
void computeDensityMatrix_ROHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix_closed, real_t* d_density_matrix_open, real_t* d_density_matrix, const int num_closed, const int num_open, const int num_basis) {
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    computeDensityMatrix_ROHF_kernel<<<num_blocks, threads_per_block>>>(
        d_coefficient_matrix,
        d_density_matrix_closed,
        d_density_matrix_open,
        d_density_matrix,
        num_closed,
        num_open,
        num_basis
    );
}


/**
 * @brief Compute the Fock matrix for the restricted Hartree-Fock method.
 * @param d_density_matrix Device pointer to the density matrix
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix Device pointer to store the Fock matrix
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 * @details The Fock matrix is given by \f$ F_{\mu\nu} = H_{\mu\nu} + \sum_{\lambda\sigma} D_{\lambda\sigma} ((\mu\nu|\lambda\sigma) - {1 \over 2}(\nu\sigma|\mu\lambda)) \f$.
 */
void computeFockMatrix_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, const int num_basis) {
    const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
    const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
    if (threadsPerBlock > 1024) {
        THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
    }
    const int num_blocks = num_basis * num_basis;
    //const int num_blocks = num_basis * (num_basis + 1) / 2;
    dim3 blocks(num_blocks);
    dim3 threads(WARP_SIZE, warpsPerBlock);
    computeFockMatrix_RHF_kernel<<<blocks, threads>>>(d_density_matrix, d_core_hamiltonian_matrix, d_eri, d_fock_matrix, num_basis);
}


/**
 * @brief Compute the Fock matrix for the unrestricted Hartree-Fock method.
 * @param d_density_matrix_a Device pointer to the density matrix for the alpha spin
 * @param d_density_matrix_b Device pointer to the density matrix for the beta spin
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix_a Device pointer to store the Fock matrix for the alpha spin
 * @param d_fock_matrix_b Device pointer to store the Fock matrix for the beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix (alpha and beta spins) using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 * @details The Fock matrix is given by 
 *          \f$ F_{\mu\nu}^\alpha = H_{\mu\nu} + \sum_{\lambda\sigma} (D_{\lambda\sigma}^\alpha + D_{\lambda\sigma}^\beta) (\mu\nu|\lambda\sigma) - D_{\lambda\sigma}^\alpha (\nu\sigma|\mu\lambda) \f$.
 *          \f$ F_{\mu\nu}^\beta  = H_{\mu\nu} + \sum_{\lambda\sigma} (D_{\lambda\sigma}^\alpha + D_{\lambda\sigma}^\beta) (\mu\nu|\lambda\sigma) - D_{\lambda\sigma}^\beta  (\nu\sigma|\mu\lambda) \f$.
 */
void computeFockMatrix_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis) {
    const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
    const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
    if (threadsPerBlock > 1024) {
        THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
    }
    const int num_blocks = num_basis * num_basis;
    //const int num_blocks = num_basis * (num_basis + 1) / 2;
    dim3 blocks(num_blocks);
    dim3 threads(WARP_SIZE, warpsPerBlock);
    computeFockMatrix_UHF_kernel<<<blocks, threads>>>(d_density_matrix_a, d_density_matrix_b, d_core_hamiltonian_matrix, d_eri, d_fock_matrix_a, d_fock_matrix_b, num_basis);
}





/**
 * @brief Compute the Fock matrix for the ROHF method.
 * @param d_density_matrix_closed Device pointer to the density matrix for the closed-shell orbitals
 * @param d_density_matrix_open Device pointer to the density matrix for the open-shell orbitals
 * @param core_hamiltonian Device pointer to the core Hamiltonian matrix
 * @param d_eri Device pointer to the electron repulsion integrals
 * @param d_fock_matrix_closed Device pointer to store the Fock matrix for the closed-shell orbitals
 * @param d_fock_matrix_open Device pointer to store the Fock matrix for the open-shell orbitals
 * @param d_fock_matrix Device pointer to store the unified Fock matrix
 * @details This function computes the Fock matrix using the density matrix, core Hamiltonian matrix, and electron repulsion integrals.
 */
void computeFockMatrix_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_coefficient_matrix, const real_t* d_overlap_matrix, const real_t* d_eri, const ROHF_ParameterSet ROH_parameters, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, real_t* d_fock_matrix, const int num_closed, const int num_open, const int num_basis) {
    real_t* d_temp_F_MO_closed = nullptr; // Fock matrix for the closed-shell MO 
    real_t* d_temp_F_MO_open = nullptr; // Fock matrix for the open-shell MO
    real_t* d_temp_R_MO = nullptr; /// unified Fock matrix R_MO
    real_t* d_temp_matrix1 = nullptr;
    real_t* d_temp_matrix2 = nullptr;

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_temp_F_MO_closed, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary Fock matrix for closed-shell orbitals: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_F_MO_open, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary Fock matrix for open-shell orbitals: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_R_MO, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary unified Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_matrix1, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 1: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_matrix2, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 2: ") + std::string(cudaGetErrorString(err)));
    }


    { // compute the Fock matrices for the closed- and open-shell orbitals
        const int warpsPerBlock = (num_basis + WARP_SIZE - 1) / WARP_SIZE;
        const int threadsPerBlock = WARP_SIZE * warpsPerBlock;
        if (threadsPerBlock > 1024) {
            THROW_EXCEPTION("Too many contracted Gauss-type orbitals.");
        }
        const int num_blocks = num_basis * num_basis;
        //const int num_blocks = num_basis * (num_basis + 1) / 2;
        dim3 blocks(num_blocks);
        dim3 threads(WARP_SIZE, warpsPerBlock);
        computeFockMatrix_ROHF_kernel<<<blocks, threads>>>(d_density_matrix_closed, d_density_matrix_open, d_core_hamiltonian_matrix, d_eri, d_fock_matrix_closed, d_fock_matrix_open, num_basis);
    }

    { // Transforms the Fock matrices from AO to the MO 
        // F_MO_closed = C^T F_AO_closed C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_closed, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_closed, num_basis, false, false);
        
        // F_MO_open = C F_AO_open C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_open, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_open, num_basis, false, false);
    }

    { // compute the unified Fock matrix R_MO
        const size_t num_elements = num_basis * (num_basis+1) / 2;
        const size_t threads_per_block = 256;
        const size_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        computeUnifiedFockMatrix_ROHF_kernel<<<num_blocks, threads_per_block>>>(d_temp_F_MO_closed, d_temp_F_MO_open, ROH_parameters, d_temp_R_MO, num_closed, num_open, num_basis);
    }

    { // Transform the unified Fock matrix from MO to AO by F_AO = S*C*R_MO*C^T*S
        // temp1 = S*C
        matrixMatrixProduct(d_overlap_matrix, d_coefficient_matrix, d_temp_matrix1, num_basis, false, false);
        // temp2 = temp1 * R_MO
        matrixMatrixProduct(d_temp_matrix1, d_temp_R_MO, d_temp_matrix2, num_basis, false, false);
        // temp1 = temp2 * C^T
        matrixMatrixProduct(d_temp_matrix2, d_coefficient_matrix, d_temp_matrix1, num_basis, false, true);
        // temp2 = temp1 * S
        matrixMatrixProduct(d_temp_matrix1, d_overlap_matrix, d_fock_matrix, num_basis, false, false);
    }

    // free the temporary memory
    gansu::tracked_cudaFree(d_temp_F_MO_closed);
    gansu::tracked_cudaFree(d_temp_F_MO_open);
    gansu::tracked_cudaFree(d_temp_R_MO);
    gansu::tracked_cudaFree(d_temp_matrix1);
    gansu::tracked_cudaFree(d_temp_matrix2);

}


/**
 * @brief Compute the energy for the restricted HF.
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param num_basis Number of basis functions
 * @return Energy
 */
real_t computeEnergy_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix, const int num_basis) {
   
    real_t energy = 0.0;
    energy += innerProduct(d_density_matrix, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix, d_fock_matrix,             num_basis * num_basis);
    energy *= 0.5;

    return energy;
}

/**
 * @brief Compute the energy for the unrestricted HF.
 * @param d_density_matrix_a Device pointer to the density matrix for the alpha spin
 * @param d_density_matrix_b Device pointer to the density matrix for the beta spin
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix_a Device pointer to the Fock matrix for the alpha spin
 * @param d_fock_matrix_b Device pointer to the Fock matrix for the beta spin
 * @param num_basis Number of basis functions
 * @return Energy
 */
real_t computeEnergy_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_a, const real_t* d_fock_matrix_b, const int num_basis) {
    real_t energy = 0.0;

    energy += innerProduct(d_density_matrix_a, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix_a, d_fock_matrix_a,           num_basis * num_basis);
    
    energy += innerProduct(d_density_matrix_b, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy += innerProduct(d_density_matrix_b, d_fock_matrix_b,           num_basis * num_basis);
    
    return 0.5 * energy;
}


/**
 * @brief Compute the energy for the ROHF method.
 * @param d_density_matrix_closed Device pointer to the density matrix for the closed orbitals
 * @param d_density_matrix_open Device pointer to the density matrix for the open orbitals
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_fock_matrix_closed Device pointer to the Fock matrix for the closed orbitals
 * @param d_fock_matrix_open Device pointer to the Fock matrix for the open orbitals
 * @param num_basis Number of basis functions
 * @return Energy
*/
real_t computeEnergy_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_closed, const real_t* d_fock_matrix_open, const int num_basis) {
    real_t energy = 0.0;

    energy +=       innerProduct(d_density_matrix_closed, d_core_hamiltonian_matrix, num_basis * num_basis);
    energy +=       innerProduct(d_density_matrix_closed, d_fock_matrix_closed,      num_basis * num_basis);

    energy +=       innerProduct(d_density_matrix_open, d_core_hamiltonian_matrix,   num_basis * num_basis);
    energy += 2.0 * innerProduct(d_density_matrix_open, d_fock_matrix_open,          num_basis * num_basis); // Note: factor 2.0 only here

    return 0.5 * energy;
}




/**
 * @brief Compute the optimal damping factor for RHF.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_prev_fock_matrix Device pointer to the previous Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_prev_density_matrix Device pointer to the previous density matrix
 * @param num_basis Number of basis functions
 * @return Optimal damping factor
 * @details This function computes the optimal damping factor for the restricted Hartree-Fock method.
 * @details The damping factor is given as follows:
 * @details \f$ s = \mathrm{Tr}[F_{\mathrm{old}}(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
 * @details \f$ c = \mathrm{Tr}[(F_{\mathrm{new}} - F_{\mathrm{old}})(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
 * @details \f$ \alpha = 1 \f$ if \f$ c \le - \frac{s}{2} \f$, otherwise \f$ \alpha = -\frac{s}{2c} \f$
 */
real_t computeOptimalDampingFactor_RHF(const real_t* d_fock_matrix, const real_t* d_prev_fock_matrix, const real_t* d_density_matrix, const real_t* d_prev_density_matrix, const int num_basis) {
    // allocate temporary memory
    real_t* d_tempDiffFockMatrix = nullptr;
    real_t* d_tempDiffDensityMatrix = nullptr;
    real_t* d_tempMatrix = nullptr;

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_tempDiffFockMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary difference Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempDiffDensityMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary difference density matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // calculate the difference between the Fock matrices
    // \f$ F_{\mathrm{diff}} = F_{\mathrm{new}} - F_{\mathrm{old}}  \f$
    matrixSubtraction(d_fock_matrix, d_prev_fock_matrix, d_tempDiffFockMatrix, num_basis);

    // calculate the difference between the density matrices
    // \f$D_{\mathrm{diff}} = D_{\mathrm{new}} - D_{\mathrm{old}} \f$
    matrixSubtraction(d_density_matrix, d_prev_density_matrix, d_tempDiffDensityMatrix, num_basis);

    // calculate the trace of the product of the difference matrices
    // \f$ s = \mathrm{Tr}[F_{\mathrm{old}}(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
    real_t s = 0.0;
    matrixMatrixProduct(
        d_prev_fock_matrix, d_tempDiffDensityMatrix, d_tempMatrix,
        num_basis, false, false
    );
    s = computeMatrixTrace(d_tempMatrix, num_basis);

    // \f$ c = \mathrm{Tr}[(F_{\mathrm{new}} - F_{\mathrm{old}})(D_{\mathrm{new}} - D_{\mathrm{old}})] \f$
    real_t c = 0.0;
    matrixMatrixProduct(
        d_tempDiffFockMatrix, d_tempDiffDensityMatrix, d_tempMatrix,
        num_basis, false, false
    );
    c = computeMatrixTrace(d_tempMatrix, num_basis);

    real_t alpha;
    //std::cout << "s = " << s << ", c = " << c << std::endl;
    if (c <= -s/2.0) {
        alpha = 1.0;
    } else {
        alpha = -0.5 * s / c;
    }

    // free the temporary memory
    gansu::tracked_cudaFree(d_tempDiffFockMatrix);
    gansu::tracked_cudaFree(d_tempDiffDensityMatrix);
    gansu::tracked_cudaFree(d_tempMatrix);


    return alpha;
}


/**
 * @brief Compute the optimal damping factor for ROHF.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_prev_fock_matrix Device pointer to the previous Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_prev_density_matrix Device pointer to the previous density matrix
 * @param num_basis Number of basis functions
 * @return Optimal damping factor
 * @details This function just calls the function computeOptimalDampingFactor_RHF.
 */
 real_t computeOptimalDampingFactor_ROHF(const real_t* d_fock_matrix, const real_t* d_prev_fock_matrix, const real_t* d_density_matrix, const real_t* d_prev_density_matrix, const int num_basis) {
    return computeOptimalDampingFactor_RHF(d_fock_matrix, d_prev_fock_matrix, d_density_matrix, d_prev_density_matrix, num_basis);
 }


/**
 * @brief Update the Fock/density matrix using the damping factor.
 * @param d_matrix_old Device pointer to the previous Fock matrix
 * @param d_matrix_new Device pointer to the current Fock matrix
 * @param alpha Damping factor
 * @details This function updates the Fock matrix using the damping factor.
 * @details The updated Fock matrix is given by \f$ F_{\mathrm{new}} = (1-\alpha)F_{\mathrm{old}} + \alpha F_{\mathrm{new}} \f$.
 * @details The current Fock matrix is overwritten with the updated Fock matrix. \f$ F_{\mathrm{old}} = F_{\mathrm{new}} \f$
 */
void damping(real_t* d_matrix_old, real_t* d_matrix_new, const real_t alpha, int num_basis) {
    real_t* d_tempMatrix;

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }

    weightedMatrixSum(d_matrix_old, d_matrix_new, d_tempMatrix, 1.0-alpha, alpha, num_basis);

    cudaMemcpy(d_matrix_old, d_tempMatrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_matrix_new, d_tempMatrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToDevice);

    gansu::tracked_cudaFree(d_tempMatrix);
}


/**
 * @brief Compute the DIIS error matrix for RHF, UHF, ROHF.
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_density_matrix Device pointer to the density matrix
 * @param d_diis_error_matrix Device pointer to store the DIIS error matrix
 * @param num_basis Number of basis functions
 * @details This function computes the DIIS error matrix.
 * @details The DIIS error matrix is given by \f$ E = FPS - SPF \f$.
 */
void computeDIISErrorMatrix(const real_t* d_overlap_matrix, const real_t* d_transform_matrix, const real_t* d_fock_matrix, const real_t* d_density_matrix, real_t* d_diis_error_matrix, const int num_basis, const bool is_include_transform) {
    real_t* d_tempFPS;
    real_t* d_tempSPF;
    real_t* d_tempMatrix1;

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_tempFPS, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary FPS matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempSPF, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary SPF matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_tempMatrix1, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 1: ") + std::string(cudaGetErrorString(err)));
    }

    // tempFPS = FPS
    matrixMatrixProduct(d_fock_matrix, d_density_matrix, d_tempMatrix1, num_basis, false, false);
    matrixMatrixProduct(d_tempMatrix1, d_overlap_matrix, d_tempFPS, num_basis, false, false);

    // tempSPF = SPF
    matrixMatrixProduct(d_overlap_matrix, d_density_matrix, d_tempMatrix1, num_basis, false, false);
    matrixMatrixProduct(d_tempMatrix1, d_fock_matrix, d_tempSPF, num_basis, false, false);

    // DIIS error matrix = FPS - SPF
    matrixSubtraction(d_tempFPS, d_tempSPF, d_diis_error_matrix, num_basis);

    if(is_include_transform){
        // tempSPF = X(FPS-SPF)
        matrixMatrixProduct(d_transform_matrix, d_diis_error_matrix, d_tempFPS, num_basis, false, false);

        // DIIS error matrix = X(FPS-SPF)X^T
        matrixMatrixProduct(d_tempFPS, d_transform_matrix, d_diis_error_matrix, num_basis, false, true);
    }

    gansu::tracked_cudaFree(d_tempMatrix1);
    gansu::tracked_cudaFree(d_tempFPS);
    gansu::tracked_cudaFree(d_tempSPF);

}



/**
 * @brief Compute the Fock matrix by DIIS for RHF, UHF, ROHF.
 * @param d_error_matrices Device pointer to the error matrices
 * @param d_fock_matrices Device pointer to the Fock matrices
 * @param d_new_fock_matrix Device pointer to store the new Fock matrix
 * @param num_prev Number of previous Fock matrices
 * @param num_basis Number of basis functions
 * @details This function computes the Fock matrix by DIIS.
 */
void computeFockMatrixDIIS(real_t* d_error_matrices, real_t* d_fock_matrices, real_t* d_new_fock_matrix, const int num_prev, const int num_basis){
    if (num_prev <= 1){
        THROW_EXCEPTION("DIIS requires at least two previous Fock matrices.");
    }

    const int num_size = num_prev + 1;

    // Create the DIIS matrix
    real_t* d_DIIS_matrix;
    real_t* h_DIIS_matrix = new real_t[num_size * num_size];
    if (h_DIIS_matrix == nullptr) {
        THROW_EXCEPTION("Failed to allocate host memory for DIIS matrix.");
    }

    cudaError_t err;

    err = gansu::tracked_cudaMalloc(&d_DIIS_matrix, num_size * num_size * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for DIIS matrix: ") + std::string(cudaGetErrorString(err)));
    }

    for (int i = 0; i < num_prev; i++){
        for (int j = i; j < num_prev; j++){
            real_t e = innerProduct(&d_error_matrices[i*num_basis*num_basis], &d_error_matrices[j*num_basis*num_basis], num_basis * num_basis);
            h_DIIS_matrix[i * num_size + j] = e;
            h_DIIS_matrix[j * num_size + i] = e;
        }
        h_DIIS_matrix[i * num_size + num_prev] = -1.0;
        h_DIIS_matrix[num_prev * num_size + i] = -1.0;
    }
    h_DIIS_matrix[num_prev * num_size + num_prev] = 0.0;


    cudaMemcpy(d_DIIS_matrix, h_DIIS_matrix, num_size * num_size * sizeof(real_t), cudaMemcpyHostToDevice);

    
    // Create the right-hand side vector
    real_t* h_DIIS_rhs = new real_t[num_size];
    if (h_DIIS_rhs == nullptr) {
        THROW_EXCEPTION("Failed to allocate host memory for DIIS right-hand side vector.");
    }
    real_t* d_DIIS_rhs;
    err = gansu::tracked_cudaMalloc(&d_DIIS_rhs, num_size * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for DIIS right-hand side vector: ") + std::string(cudaGetErrorString(err)));
    }

    for (int i = 0; i < num_prev; i++){
        h_DIIS_rhs[i] = 0.0;
    }
    h_DIIS_rhs[num_prev] = -1.0;
    

    cudaMemcpy(d_DIIS_rhs, h_DIIS_rhs, num_size * sizeof(real_t), cudaMemcpyHostToDevice);

    // Solve the linear equation on the device
    cusolverManager cusolver;

    // get the workspace size
    int work_size;
    cusolverDnDgetrf_bufferSize(cusolver.cusolverHandle, num_size, num_size, d_DIIS_matrix, num_size, &work_size);

    // allocate the workspace
    real_t* d_work;
    err = gansu::tracked_cudaMalloc(&d_work, work_size * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // pivot array and info
    int* d_pivot = nullptr;
    int* d_info = nullptr;
    err = gansu::tracked_cudaMalloc(&d_pivot, num_size * sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for pivot array: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for info array: ") + std::string(cudaGetErrorString(err)));
    }

    // LU factorization
    cusolverDnDgetrf(cusolver.cusolverHandle, num_size, num_size, d_DIIS_matrix, num_size, d_work, d_pivot, d_info);

    // solve the linear equation
    cusolverDnDgetrs(cusolver.cusolverHandle, CUBLAS_OP_N, num_size, 1, d_DIIS_matrix, num_size, d_pivot, d_DIIS_rhs, num_size, d_info);

    // copy the result to the host
    cudaMemcpy(h_DIIS_rhs, d_DIIS_rhs, num_size * sizeof(real_t), cudaMemcpyDeviceToHost);


    // compute the DIIS Fock matrix (\f$ F_{\mathrm{new}} = \sum_{i=1}^{N} c_i F_i \f$)
    // F = c_1 F_1 + c_2 F_2
    weightedMatrixSum(&d_fock_matrices[0*num_basis*num_basis], &d_fock_matrices[1*num_basis*num_basis], d_new_fock_matrix, h_DIIS_rhs[0], h_DIIS_rhs[1], num_basis);
    for (int i = 2; i < num_prev; i++){
        weightedMatrixSum(d_new_fock_matrix, &d_fock_matrices[i*num_basis*num_basis], d_new_fock_matrix, 1.0, h_DIIS_rhs[i], num_basis);
    }

    // free the memory
    gansu::tracked_cudaFree(d_DIIS_matrix);
    gansu::tracked_cudaFree(d_DIIS_rhs);
    gansu::tracked_cudaFree(d_work);
    gansu::tracked_cudaFree(d_pivot);
    gansu::tracked_cudaFree(d_info);

    delete[] h_DIIS_matrix;
    delete[] h_DIIS_rhs;

}



/**
 * @brief Computes the coefficient matrix from the Fock matrix and the transformation matrix.
 * @param d_fock_matrix Device pointer to the Fock matrix
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_transform_matrix Device pointer to the transformation matrix
 * @param d_coefficient_matrix Device pointer to store the coefficient matrix
 * @param num_basis Number of basis functions
 * @details This function computes the coefficient matrix using the eigenvectors of the Fock matrix by solving the generalized eigenvalue problem \f$FC = SCE \f$.
 * @details To transform the generalized eigenvalue problem to the standard eigenvalue problem \f$FC = CE \f$, the transformation matrix.
 */
 void computeInitialCoefficientMatrix_GWH(const real_t* d_core_hamiltonian_matrix, const real_t* d_overlap_matrix, const real_t* d_transform_matrix, real_t* d_coefficient_matrix, const int num_basis) {
    const real_t cx = 1.75;

    cudaError_t err;

    // allocate temporary memory
    real_t* d_temp_FockMatrix = nullptr;
    real_t* h_temp_FockMatrix = new real_t[num_basis * num_basis];
    err = gansu::tracked_cudaMalloc(&d_temp_FockMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // Compute the initial Fock matrix
    size_t threads_per_block = 256;
    size_t num_blocks = (num_basis * num_basis + threads_per_block - 1) / threads_per_block;
    computeInitialFockMatrix_GWH_kernel<<<num_blocks, threads_per_block>>>(d_core_hamiltonian_matrix, d_overlap_matrix, d_temp_FockMatrix, num_basis, cx);

    // Diagonalize the Fock matrix
    computeCoefficientMatrix(d_temp_FockMatrix, d_transform_matrix, d_coefficient_matrix, num_basis);

    // free the temporary memory
    gansu::tracked_cudaFree(d_temp_FockMatrix);

}


/**
 * @brief Computes the inverse of an N x N matrix stored in device memory.
 * 
 * This function overwrites the input matrix with its inverse using LU decomposition.
 * The original matrix is destroyed in the process.
 * 
 * @param d_A Pointer to the N x N matrix in device memory (input).
 * @param N The size of the matrix (number of rows/columns).
 */
void invertMatrix(double* d_A, const int N) {
    //cusolverManager cusolver;
    cusolverDnHandle_t cusolverHandle = GPUHandle::cusolver();

    int *d_ipiv, *d_info;
    double *d_work;
    int lwork;

    cudaError_t err;
    
    err = gansu::tracked_cudaMalloc(&d_ipiv, N * sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for pivot array: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for info array: ") + std::string(cudaGetErrorString(err)));
    }

    // Get workspace size for LU decomposition
    cusolverDnDgetrf_bufferSize(cusolverHandle, N, N, d_A, N, &lwork);
    err = gansu::tracked_cudaMalloc(&d_work, lwork * sizeof(double));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // Perform LU decomposition
    cusolverDnDgetrf(cusolverHandle, N, N, d_A, N, d_work, d_ipiv, d_info);

    // Allocate and initialize an identity matrix on the device
    double *d_I;
    err = gansu::tracked_cudaMalloc(&d_I, N * N * sizeof(double));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for identity matrix: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_I, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        double one = 1.0;
        cudaMemcpy(d_I + i * N + i, &one, sizeof(double), cudaMemcpyHostToDevice);
    }

    // Solve for the inverse using the LU decomposition
    cusolverDnDgetrs(cusolverHandle, CUBLAS_OP_N, N, N, d_A, N, d_ipiv, d_I, N, d_info);

    // Copy the result back to d_A (overwrite original matrix with its inverse)
    cudaMemcpy(d_A, d_I, N * N * sizeof(double), cudaMemcpyDeviceToDevice);

    // Cleanup
    gansu::tracked_cudaFree(d_ipiv);
    gansu::tracked_cudaFree(d_info);
    gansu::tracked_cudaFree(d_work);
    gansu::tracked_cudaFree(d_I);
}




/**
 * @brief Performs Cholesky decomposition on an N x N symmetric positive definite matrix in device memory.
 *
 * The input matrix is overwritten with the result. The decomposition produces a lower triangular
 * matrix L such that A = L * L^T.
 *
 * @param d_A Pointer to the N x N matrix in device memory (input/output).
 * @param N The size of the matrix (number of rows/columns).
 */
void choleskyDecomposition(double* d_A, const int N) {
    //cusolverManager cusolver;
    cusolverDnHandle_t cusolverHandle = GPUHandle::cusolver();

    int *d_info;
    double *d_work;
    int lwork;

    cudaError_t err;

    // Allocate device memory for error info
    err = gansu::tracked_cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for info array: ") + std::string(cudaGetErrorString(err)));
    }

    // Get workspace size
    cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, &lwork);
    err = gansu::tracked_cudaMalloc(&d_work, lwork * sizeof(double));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // Perform Cholesky decomposition (A -> L, overwriting lower triangular part)
    cusolverDnDpotrf(cusolverHandle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, d_work, lwork, d_info);

    // Set zero to the upper triangular part
    const int num_threads = 256;
    const int num_blocks = (N * N + num_threads - 1) / num_threads;

    setZeroUpperTriangle<<<num_blocks, num_threads>>>(d_A, N);


    // Cleanup
    gansu::tracked_cudaFree(d_work);
    gansu::tracked_cudaFree(d_info);
}



/**
 * @brief Solve the equation AX = B for X, where A is a lower triangular matrix.
 * The input matrix B is overwritten with the result X. 
 * @param d_A Pointer to the row x row lower triangular matrix in device memory.
 * @param d_B Pointer to the row x col matrix in device memory (input/output).
 * @param row The number of rows.
 * @param row The number of columns.
 */
void solve_lower_triangular(double* d_A, double* d_B, int row, int col){
    cublasHandle_t cublasHandle = GPUHandle::cublas();


    // 転置
    transposeMatrixInPlace(d_A, row);

    cudaError_t err;

    double *d_tmp;
    err = gansu::tracked_cudaMalloc((void**)&d_tmp, sizeof(double) * row * col);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }


    const double alpha = 1.0;
    const double beta = 0.0; //これ必要

    cublasDgeam(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        row, col,
        &alpha,
        d_B, col,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_tmp, row
    );

    // // Solve A * X = B → X overwrites B
    cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_LEFT,        
        CUBLAS_FILL_MODE_LOWER, 
        CUBLAS_OP_N,            
        CUBLAS_DIAG_NON_UNIT,   
        row,                   
        col,                   
        &alpha,
        d_A, row,                  
        d_tmp, row                  
    );

    cublasDgeam(
        cublasHandle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        col, row,
        &alpha,
        d_tmp, row,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B, col
    );

    gansu::tracked_cudaFree(d_tmp);
}


inline void writeMatrixToFile(std::string filename, double* array, size_t size) {
    std::ofstream outFile(filename);
    if (!outFile) 
        throw std::runtime_error("ファイルを書き込み用に開けませんでした");
    
    for (size_t i = 0; i < size; i++) {
        outFile << std::fixed << std::setprecision(15) << array[i] << "\n";
    }
}

/**
 * @brief Compute the intermediate matrix B for the RI approximation.
 * @param shell_type_infos Information about the basis functions
 * @param d_primitive_shells Pointer to the primitive shells in device memory
 * @param d_cgto_normalization_factors Pointer to the normalization factors of the CGTOs in device memory
 * @param auxiliary_shell_type_infos Information about the auxiliary basis functions
 * @param d_auxiliary_primitive_shells Pointer to the primitive shells of the auxiliary basis functions in device memory
 * @param d_auxiliary_cgto_normalization_factors Pointer to the normalization factors of the auxiliary CGTOs in device memory
 * @param d_intermediate_matrix_B Pointer to the intermediate matrix B in device memory
 * @param num_basis Number of basis functions
 * @param num_auxiliary_basis Number of auxiliary basis functions
 * @param d_boyst_grid The grid values of the precomputed Boys function
 * @param verbose Whether to print additional information
 * @details This function computes the intermediate matrix B for the RI approximation.
 * @details (1) \f$ A_{pq} = (p|q) \f$. (two-center ERIs)
 * @details (2) \f$ A^{-1} = LL^T \f$. (Cholesky decomposition)
 * @details (3) \f$ B_{\mu\nu}^p = \sum_{q}^{M_\textrm{aux}} (\mu\nu|q)L_{qp}.
 */
void compute_RI_IntermediateMatrixB(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_normalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_normalization_factors, 
    real_t* d_intermediate_matrix_B, 
    const size_t2* d_primitive_shell_pair_indices,
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const int num_basis, 
    const int num_auxiliary_basis, 
    const real_t* d_boys_grid, 
    const bool verbose) 
{
    cudaError_t err;

    // Allocate device memory for the two-center ERIs
    real_t* d_two_center_eri;
    err = gansu::tracked_cudaMalloc(&d_two_center_eri, num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for two-center ERIs: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_two_center_eri, 0.0, num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t));

    // Compute the two-center ERIs of the auxiliary basis functions
    computeTwoCenterERIs(
        auxiliary_shell_type_infos, 
        d_auxiliary_primitive_shells, 
        d_auxiliary_cgto_normalization_factors, 
        d_two_center_eri, 
        num_auxiliary_basis,
        d_boys_grid,
        d_auxiliary_schwarz_upper_bound_factors,
        schwarz_screening_threshold,
        verbose);


    // Cholesky decomposition of the inverse of the two-center ERI matrix (it is overwritten with the result)
    choleskyDecomposition(d_two_center_eri, num_auxiliary_basis);

    // Allocate device memory for the three-center ERIs
    real_t* d_three_center_eri;
    err = gansu::tracked_cudaMalloc(&d_three_center_eri, (size_t)num_basis * (size_t)num_basis * (size_t)num_auxiliary_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for three-center ERIs: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_three_center_eri, 0.0, num_basis * num_basis * (size_t)num_auxiliary_basis * sizeof(real_t));

    // Compute the three-center ERIs of the auxiliary basis functions and the basis functions
    computeThreeCenterERIs(
        shell_type_infos, 
        shell_pair_type_infos,
        d_primitive_shells, 
        d_cgto_normalization_factors, 
        auxiliary_shell_type_infos, 
        d_auxiliary_primitive_shells, 
        d_auxiliary_cgto_normalization_factors, 
        d_three_center_eri, 
        d_primitive_shell_pair_indices,
        num_basis,
        num_auxiliary_basis,
        d_boys_grid,
        d_schwarz_upper_bound_factors,
        d_auxiliary_schwarz_upper_bound_factors,
        schwarz_screening_threshold,
        verbose);

   
    // Compute the intermediate matrix B
    solve_lower_triangular(d_two_center_eri, d_three_center_eri, num_auxiliary_basis, num_basis*num_basis);
    cudaMemcpy(d_intermediate_matrix_B, d_three_center_eri, sizeof(real_t) * num_auxiliary_basis*num_basis*num_basis, cudaMemcpyDeviceToDevice);


    gansu::tracked_cudaFree(d_two_center_eri);
    gansu::tracked_cudaFree(d_three_center_eri);

}

void computeIntermediateMatrixB(
    const real_t* d_three_center_eri, 
    const real_t* d_two_center_eri, 
    real_t* d_intermediate_matrix_B, 
    const int num_basis, 
    const int num_auxiliary_basis)
{
    // B_{\mu\nu}^p = \sum_{q}^{M_\textrm{aux}} (\mu\nu|q)L_{qp}
    // B[p][\mu][\nu] = \sum_{q}^{M_\textrm{aux}} T[q][\mu][\nu] * L[q][p]

    const int num_threads = 256;
    const int num_blocks = (num_auxiliary_basis * num_basis * num_basis + num_threads - 1) / num_threads;
    computeRIIntermediateMatrixB_kernel<<<num_blocks, num_threads>>>(d_three_center_eri, d_two_center_eri, d_intermediate_matrix_B, num_basis, num_auxiliary_basis);
}



//* With density matrix
void computeFockMatrix_RI_RHF_with_density_matrix(
    const real_t* d_density_matrix, 
    const real_t* d_core_hamiltonian_matrix, 
    const real_t* d_intermediate_matrix_B, 
    real_t* d_fock_matrix, 
    const int num_basis, 
    const int num_auxiliary_basis, 
    real_t* d_J, 
    real_t* d_K, 
    real_t* d_W, 
    real_t* d_T, 
    real_t* d_V){
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    double alpha = 1.0;
    double beta = 0.0;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);
    // J = sum(W[i] * B[i])
    weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    cublasDgemmStridedBatched(
        cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis, num_basis,
        &alpha,
        d_density_matrix, num_basis, 0,
        d_intermediate_matrix_B, num_basis, num_basis*num_basis,
        &beta,
        d_T, num_basis, num_basis*num_basis,
        num_auxiliary_basis
    );
    cublasDgemmStridedBatched(
        cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis, num_basis,
        &alpha,
        d_T, num_basis, num_basis*num_basis,
        d_intermediate_matrix_B, num_basis, num_basis*num_basis,
        &beta,
        d_V, num_basis, num_basis*num_basis,
        num_auxiliary_basis
    );
    // K = sum(V^p)
    sum_matrices_kernel<<<num_blocks, num_threads>>>(d_K, d_V, num_basis, num_auxiliary_basis); 

    ////////////////////////////////// compute Fock matrix //////////////////////////////////
    // F = H + J - (1/2)*K
    computeFockMatrix_RI_RHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, num_basis);
    cudaDeviceSynchronize();
}
/**/






//* With coefficient matrix
__global__ void packThreeDimensionalTensorX(
    const real_t* d_X_in, real_t* d_X_out, 
    const int num_basis, const int num_auxiliary_basis, const int num_occ)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_size = (size_t)num_auxiliary_basis * num_basis * num_occ;

    if (idx >= total_size) {
        return;
    }

    const int p = idx / (num_basis * num_occ);
    const int mu = (idx % (num_basis * num_occ)) / num_occ;
    const int k = (idx % (num_basis * num_occ)) % num_occ;

    d_X_out[((size_t)num_auxiliary_basis * num_occ) * mu + num_occ * p + k] = d_X_in[idx];
}


void computeFockMatrix_RI_RHF_with_coefficient_matrix(
    const real_t* d_coefficient_matrix, 
    const real_t* d_density_matrix, 
    const real_t* d_core_hamiltonian_matrix, 
    const real_t* d_intermediate_matrix_B, 
    real_t* d_fock_matrix, 
    const int num_basis, 
    const int num_auxiliary_basis, 
    const int num_occ, 
    real_t* d_J, 
    real_t* d_K, 
    real_t* d_W, 
    real_t* d_X, 
    real_t* d_X_packed)
{
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    double alpha = 1.0;
    double beta = 0.0;
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);
    // J = sum(W[i] * B[i])
    weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    cudaMemset(d_K, 0, sizeof(real_t) * num_basis * num_basis);
    cublasDgemmStridedBatched(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ, num_basis, num_basis,
        &alpha,
        d_coefficient_matrix, num_basis, 0,
        d_intermediate_matrix_B, num_basis, num_basis * num_basis,
        &beta,
        d_X, num_occ, num_basis * num_occ,
        num_auxiliary_basis
    );
    packThreeDimensionalTensorX<<<(num_auxiliary_basis * num_basis * num_occ + num_threads - 1) / num_threads, num_threads>>>(d_X, d_X_packed, num_basis, num_auxiliary_basis, num_occ);
    alpha = 2.0;
    cublasDgemm(
        cublasHandle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis, num_occ * num_auxiliary_basis,
        &alpha,
        d_X_packed, num_occ * num_auxiliary_basis,
        d_X_packed, num_occ * num_auxiliary_basis,
        &beta,
        d_K, num_basis
    );

    ////////////////////////////////// compute Fock matrix //////////////////////////////////
    // F = H + J - (1/2)*K
    computeFockMatrix_RI_RHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, num_basis);
    cudaDeviceSynchronize();
}
/**/






void computeFockMatrix_RI_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis, const int num_auxiliary_basis){
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    cudaError_t err;

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    real_t* d_J = nullptr;
    real_t* d_density_matrix = nullptr;
    err = gansu::tracked_cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_density_matrix, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for density matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // D = D_a + D_b
    matrixAddition(d_density_matrix_a, d_density_matrix_b, d_density_matrix, num_basis);

    // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
    real_t* d_W = nullptr;
    err = gansu::tracked_cudaMalloc(&d_W, sizeof(real_t)*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for W vector: ") + std::string(cudaGetErrorString(err)));
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

    // J = sum(W[i] * B[i])
    weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);


    // free the memory
    gansu::tracked_cudaFree(d_W);
    gansu::tracked_cudaFree(d_density_matrix);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_T = nullptr;
    real_t* d_V = nullptr;
    err = gansu::tracked_cudaMalloc(&d_T, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for T matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_V, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for V matrix: ") + std::string(cudaGetErrorString(err)));
    }

    ////////////// compute Ka-matrix //////////////
    real_t* d_Ka = nullptr;
    err = gansu::tracked_cudaMalloc(&d_Ka, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Ka matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // T^p = B^p Da^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix_a, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }

    // V^p = B^p (T^p)^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // Ka = sum(V^p)
    sum_matrices_kernel<<<num_blocks, num_threads>>>(d_Ka, d_V, num_basis, num_auxiliary_basis); 

    ////////////// compute Kb-matrix //////////////
    real_t* d_Kb = nullptr;
    err = gansu::tracked_cudaMalloc(&d_Kb, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Kb matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // T^p = B^p Da^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix_b, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }

    // V^p = B^p (T^p)^T
    // Note: cublasDgemmBatched should be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // Kb = sum(V^p)
    sum_matrices_kernel<<<num_blocks, num_threads>>>(d_Kb, d_V, num_basis, num_auxiliary_basis); 




    // free the memory
    gansu::tracked_cudaFree(d_T);
    gansu::tracked_cudaFree(d_V);

    ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // F_a = H + J - K_a
    // F_b = H + J - K_b
    computeFockMatrix_RI_UHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Ka, d_fock_matrix_a, num_basis);
    computeFockMatrix_RI_UHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Kb, d_fock_matrix_b, num_basis);


    // free the memory
    gansu::tracked_cudaFree(d_J);
    gansu::tracked_cudaFree(d_Ka);
    gansu::tracked_cudaFree(d_Kb);
}



void computeFockMatrix_RI_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_coefficient_matrix, const real_t* d_overlap_matrix, const real_t* d_intermediate_matrix_B, const ROHF_ParameterSet ROH_parameters, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, real_t* d_fock_matrix, const int num_closed, const int num_open, const int num_basis, const int num_auxiliary_basis){
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    cudaError_t err;

    real_t* d_temp_F_MO_closed = nullptr; // Fock matrix for the closed-shell MO 
    real_t* d_temp_F_MO_open = nullptr; // Fock matrix for the open-shell MO
    real_t* d_temp_R_MO = nullptr; /// unified Fock matrix R_MO
    real_t* d_temp_matrix1 = nullptr;
    real_t* d_temp_matrix2 = nullptr;
    err = gansu::tracked_cudaMalloc(&d_temp_F_MO_closed, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for closed-shell Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_F_MO_open, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for open-shell Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_R_MO, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for unified Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_matrix1, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 1: ") + std::string(cudaGetErrorString(err)));
    }
    err = gansu::tracked_cudaMalloc(&d_temp_matrix2, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 2: ") + std::string(cudaGetErrorString(err)));
    }

    {// compute the Fock matrices for the closed- and open-shell orbitals using RI approximation

        // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
        const int num_threads = 256;
        const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

        ////////////////////////////////// compute J-matrix //////////////////////////////////
        real_t* d_J = nullptr;
        real_t* d_density_matrix = nullptr;
        err = gansu::tracked_cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
        }
        err = gansu::tracked_cudaMalloc(&d_density_matrix, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for density matrix: ") + std::string(cudaGetErrorString(err)));
        }

        // D = D_closed + D_open
        matrixAddition(d_density_matrix_closed, d_density_matrix_open, d_density_matrix, num_basis);

        // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
        real_t* d_W = nullptr;
        err = gansu::tracked_cudaMalloc(&d_W, sizeof(real_t)*num_auxiliary_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for W vector: ") + std::string(cudaGetErrorString(err)));
        }

        double alpha = 1.0;
        double beta = 0.0;

        cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

        // J = sum(W[i] * B[i])
        weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);


        // free the memory
        gansu::tracked_cudaFree(d_W);

        ////////////////////////////////// compute Kclosed-matrix //////////////////////////////////
        real_t* d_T = nullptr;
        real_t* d_V = nullptr;
        err = gansu::tracked_cudaMalloc(&d_T, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for T matrix: ") + std::string(cudaGetErrorString(err)));
        }
        err = gansu::tracked_cudaMalloc(&d_V, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for V matrix: ") + std::string(cudaGetErrorString(err)));
        }

        ////////////// compute Kclosed-matrix //////////////
        real_t* d_Kclosed = nullptr;
        err = gansu::tracked_cudaMalloc(&d_Kclosed, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for Kclosed matrix: ") + std::string(cudaGetErrorString(err)));
        }

        // T^p = B^p Da^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
        }

        // V^p = B^p (T^p)^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
        }

        // Kclosed = sum(V^p)
        sum_matrices_kernel<<<num_blocks, num_threads>>>(d_Kclosed, d_V, num_basis, num_auxiliary_basis); 

    
        ////////////// compute Kopen-matrix //////////////
        real_t* d_Kopen = nullptr;
        err = gansu::tracked_cudaMalloc(&d_Kopen, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for Kopen matrix: ") + std::string(cudaGetErrorString(err)));
        }

        // D = 0.5*D_closed + D_open
        weightedMatrixSum(d_density_matrix_closed, d_density_matrix_open, d_density_matrix, 0.5, 1.0, num_basis);

        // T^p = B^p Da^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
        }

        // V^p = B^p (T^p)^T
        // Note: cublasDgemmBatched should be used?
        for(int p=0; p<num_auxiliary_basis; p++){
            matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
        }

        // Kclosed = sum(V^p)
        sum_matrices_kernel<<<num_blocks, num_threads>>>(d_Kopen, d_V, num_basis, num_auxiliary_basis); 

    
        // free the memory
        gansu::tracked_cudaFree(d_T);
        gansu::tracked_cudaFree(d_V);
        


        ////////////////////////////////// compute Fock matrix //////////////////////////////////

        // Fclosed = H + J - 0.5*Kclosed
        // Fopen = 0.5*(H + J - Kopen)
        computeFockMatrix_RI_ROHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Kclosed, d_Kopen, d_fock_matrix_closed, d_fock_matrix_open, num_basis);


        // free the memory
        gansu::tracked_cudaFree(d_J);
        gansu::tracked_cudaFree(d_Kclosed);
        gansu::tracked_cudaFree(d_Kopen);
        gansu::tracked_cudaFree(d_density_matrix);
    }



    { // Transforms the Fock matrices from AO to the MO 
        // F_MO_closed = C^T F_AO_closed C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_closed, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_closed, num_basis, false, false);
        
        // F_MO_open = C F_AO_open C
        matrixMatrixProduct(d_coefficient_matrix, d_fock_matrix_open, d_temp_matrix1, num_basis, true, false);
        matrixMatrixProduct(d_temp_matrix1, d_coefficient_matrix, d_temp_F_MO_open, num_basis, false, false);
    }

    { // compute the unified Fock matrix R_MO
        const size_t num_elements = num_basis * (num_basis+1) / 2;
        const size_t threads_per_block = 256;
        const size_t num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
        computeUnifiedFockMatrix_ROHF_kernel<<<num_blocks, threads_per_block>>>(d_temp_F_MO_closed, d_temp_F_MO_open, ROH_parameters, d_temp_R_MO, num_closed, num_open, num_basis);
    }

    { // Transform the unified Fock matrix from MO to AO by F_AO = S*C*R_MO*C^T*S
        // temp1 = S*C
        matrixMatrixProduct(d_overlap_matrix, d_coefficient_matrix, d_temp_matrix1, num_basis, false, false);
        // temp2 = temp1 * R_MO
        matrixMatrixProduct(d_temp_matrix1, d_temp_R_MO, d_temp_matrix2, num_basis, false, false);
        // temp1 = temp2 * C^T
        matrixMatrixProduct(d_temp_matrix2, d_coefficient_matrix, d_temp_matrix1, num_basis, false, true);
        // temp2 = temp1 * S
        matrixMatrixProduct(d_temp_matrix1, d_overlap_matrix, d_fock_matrix, num_basis, false, false);
    }

    // free the temporary memory
    gansu::tracked_cudaFree(d_temp_F_MO_closed);
    gansu::tracked_cudaFree(d_temp_F_MO_open);
    gansu::tracked_cudaFree(d_temp_R_MO);
    gansu::tracked_cudaFree(d_temp_matrix1);
    gansu::tracked_cudaFree(d_temp_matrix2);

}

void computeTwoCenterERIs(
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_normalization_factors, 
    real_t* d_two_center_eri, 
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const bool verbose)
{
    // ここに２中心積分を計算するコードを書く    
    const int threads_per_block = 128;
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    std::vector<std::pair<int, int>> shell_pairs;

    // a <= b の2つ組を作る
    for (int a = 0; a < auxiliary_shell_type_count; ++a) {
        for (int b = a; b < auxiliary_shell_type_count; ++b) {
            shell_pairs.emplace_back(a, b);
        }
    }

    // (a + b) の降順にソート
    std::sort(shell_pairs.begin(), shell_pairs.end(),
        [](const auto& lhs, const auto& rhs) {
            return (lhs.first + lhs.second) > (rhs.first + rhs.second);  // 降順
        });

    // make multi stream
    const int num_kernels = shell_pairs.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }

    // for-loop for sorted shell-type (s0, s1)
    int stream_id = 0;
    for(const auto& pair: shell_pairs) {
        int s0, s1;
        std::tie(s0, s1) = pair;

        const ShellTypeInfo shell_s0 = auxiliary_shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = auxiliary_shell_type_infos[s1];

        const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count); // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block; // the number of blocks

        // real_t*, PrimitiveShell*, real_t*, ShellTypeInfo, ShellTypeInfo, int, int
        gpu::get_2center_kernel(s0, s1)<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_two_center_eri, d_auxiliary_primitive_shells, 
                                                                            d_auxiliary_cgto_normalization_factors, 
                                                                            shell_s0, shell_s1, 
                                                                            num_shell_pairs, 
                                                                            d_auxiliary_schwarz_upper_bound_factors,
                                                                            schwarz_screening_threshold,
                                                                            num_auxiliary_basis, 
                                                                            d_boys_grid);
    
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << "|" << shell_type_to_shell_name(s1) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|[a|b]|=" << num_shell_pairs << ", ";
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

inline int calcIdx_triangular_(int a, int b, int N){
    return (int)(a*N - (a*(a-1))/2) + (b-a);
}

void computeThreeCenterERIs(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_normalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_normalization_factors, 
    real_t* d_three_center_eri, 
    const size_t2* d_primitive_shell_pair_indices,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    const bool verbose)
{
    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();

    // Call the kernel functions from (ss|s),... (e.g. (ss|s), (ss|p), (sp|s), (sp|p), (pp|s), (pp|p) for s and p shells)

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& triple: shell_triples) {
        int s0, s1, s2;
        std::tie(s0, s1, s2) = triple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

        //const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const size_t num_tasks = ( (s0==s1) ? ((size_t)shell_s0.count*(shell_s0.count+1)/2) : ((size_t)shell_s0.count*shell_s1.count) ) * (size_t)shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const size_t num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
        
        gpu::get_3center_kernel(s0, s1, s2)<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_three_center_eri, d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors, 
                                                                                shell_s0, shell_s1, shell_s2, 
                                                                                num_tasks, num_basis, 
                                                                                &d_primitive_shell_pair_indices[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                &d_schwarz_upper_bound_factors[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                d_auxiliary_schwarz_upper_bound_factors,
                                                                                schwarz_screening_threshold,
                                                                                num_auxiliary_basis,
                                                                                d_boys_grid);
    
        if(verbose){
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2)<< "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s2.count << ", ";
            std::cout << "|[ab|c]|=" << num_tasks << ", ";
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }

    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

/**
 * @brief Compute the Schwarz upper bounds for the shell pairs.
 * @param shell_type_infos Information about the basis functions
 * @param shell_pair_type_infos Information about the shell pairs
 * @param d_primitive_shells Pointer to the primitive shells in device memory
 * @param d_boys_grid Pointer to the precomputed grid values of the Boys function in device memory
 * @param d_cgto_normalization_factors Pointer to the normalization factors of the CGTOs in device memory
 * @param d_upper_bound_factors Pointer to store the upper bound factors in device memory to be stored
 * @param verbose Whether to print additional information
 * @details This function computes the Schwarz upper bounds for the shell pairs.
 */
void computeSchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_normalization_factors, 
    real_t* d_upper_bound_factors, 
    const bool verbose)
{
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_type_infos.size();

    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        for (int s1 = s0; s1 < shell_type_count; ++s1) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const size_t head = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
            const size_t num_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].count;
            const size_t num_blocks = (num_bra + threads_per_block - 1) / threads_per_block; // the number of blocks

            gpu::get_schwarz_kernel(s0, s1)<<<num_blocks, threads_per_block>>>(d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, head, num_bra, d_boys_grid, d_upper_bound_factors);
        }
    }
}


/**
 * @brief Compute the Schwarz upper bounds for the shell pairs.
 * @param shell_aux_type_infos Information about the auxiliary primitive shells
 * @param d_primitive_shells_aux Pointer to the auxiliary primitive shells in device memory
 * @param d_boys_grid Pointer to the precomputed grid values of the Boys function in device memory
 * @param d_cgto_aux_normalization_factors Pointer to the normalization factors of the auxiliary CGTOs in device memory
 * @param d_upper_bound_factors_aux Pointer to store the upper bound factors in device memory to be stored
 * @param verbose Whether to print additional information
 * @details This function computes the Schwarz upper bounds for the shell pairs.
 */
void computeAuxiliarySchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& shell_aux_type_infos, 
    const PrimitiveShell* d_primitive_shells_aux, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_aux_normalization_factors, 
    real_t* d_upper_bound_factors_aux, 
    const bool verbose)
{
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_aux_type_infos.size();

    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        const ShellTypeInfo shell_s0 = shell_aux_type_infos[s0];
        const size_t head = shell_s0.start_index;
        const size_t num_bra = shell_s0.count;
        const size_t num_blocks = (num_bra + threads_per_block - 1) / threads_per_block; // the number of blocks
        gpu::get_schwarz_aux_kernel(s0)<<<num_blocks, threads_per_block>>>(d_primitive_shells_aux, d_cgto_aux_normalization_factors, shell_s0, head, num_bra, d_boys_grid, d_upper_bound_factors_aux);        
    }
}


void checkMatrixSum(const real_t* d_matrix, const size_t num_basis)
{
    real_t* h_matrix;
    cudaMallocHost(&h_matrix, sizeof(real_t) * num_basis * num_basis);
    cudaMemcpy(h_matrix, d_matrix, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToHost);
    real_t matrix_sum = 0.0;
    for (int i = 0; i < num_basis * num_basis; ++i) {
        matrix_sum += h_matrix[i];
    }
    printf("sum of matrix: %.12lf\n", matrix_sum);
    cudaFreeHost(h_matrix);
}



__global__ void initializeMinSkippedColumns(
    int* g_min_skipped_column, int num_bra_groups, int num_ket_groups)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < num_bra_groups) g_min_skipped_column[idx] = num_ket_groups;
}


__global__ void computeFockMatrix_DFT_kernel(const double* d_core_hamiltonian_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_Fock_matrix[id];
}



/*
void computeFockMatrix_Direct_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const int2* d_primitive_shell_pair_indices,
    const real_t* d_cgto_normalization_factors, 
    const real_t* d_boys_grid, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    real_t* d_fock_matrix,
    const int num_basis,
    std::vector<int*>& d_global_counters,
    std::vector<int*>& d_min_skipped_columns,
    real_t* d_fock_matrix_replicas,
    const int num_fock_replicas,
    const int verbose)
{
    // compute the electron repulsion integrals
    const int num_threads_per_block = 256;
    const int shell_type_count = shell_type_infos.size();

    //cudaMemset(d_fock_matrix, 0.0, sizeof(real_t) * num_basis * num_basis);
    cudaMemset(d_fock_matrix_replicas, 0, sizeof(real_t) * num_basis * num_basis * num_fock_replicas);

    // list shell-quadruples for sorted shell-type (s0, s1, s2, s3)
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    // reverse the order of the shell_quadruples to make it sorted by (s0, s1, s2, s3)
    // e.g. (pp|pp), (sp|pp), (sp|sp), (ss|pp), (ss|sp), (ss|ss) for s and p shells
    //std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    // make multi stream
    const int num_kernels = shell_quadruples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int kernel_idx = 0;
    const int task_group_size = 16;
    const int num_cuda_blocks = 256;
    for (const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        //std::cout << "num_bra: " << num_bra << ", num_ket: " << num_ket << std::endl;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket; // equal to the number of threads
        const size_t num_blocks = (num_braket + num_threads_per_block - 1) / num_threads_per_block; // the number of blocks
        //std::cout << "num_braket: " << num_braket << std::endl;

        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;
        //std::cout << "head_bra: " << head_bra << ", head_ket: " << head_ket << std::endl;

        if (s0 <= 1 && s1 <= 1 && s2 <= 1 && s3 <= 1) {
            // initialzie global counters and minimum skipped columns for dynamic screening
            const int num_bra_groups = (num_bra + task_group_size - 1) / task_group_size;
            const int num_ket_groups = (num_ket + task_group_size - 1) / task_group_size;
            const int num_init_blocks = (num_bra_groups + num_threads_per_block - 1) / num_threads_per_block;
            //cudaMemset(d_global_counters[kernel_idx], 0, sizeof(int) * num_bra_groups);
            cudaMemsetAsync(d_global_counters[kernel_idx], 0, sizeof(int) * num_bra_groups, streams[kernel_idx]);
            initializeMinSkippedColumns<<<num_init_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>(d_min_skipped_columns[kernel_idx], num_bra_groups, num_ket_groups);

            //gpu::get_eri_kernel_direct(s0, s1, s2, s3)<<<num_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
            //    (d_fock_matrix_replicas, d_primitive_shells, d_primitive_shell_pair_indices, 
            //     d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
            //     num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
            //     num_basis, d_boys_grid, d_density_matrix, head_bra, head_ket, num_fock_replicas);
            gpu::get_eri_kernel_dynamic(s0, s1, s2, s3)<<<num_cuda_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                (d_fock_matrix_replicas, d_primitive_shells, d_primitive_shell_pair_indices, 
                 d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 schwarz_screening_threshold, d_schwarz_upper_bound_factors, num_basis, 
                 d_boys_grid, d_density_matrix, d_global_counters[kernel_idx], d_min_skipped_columns[kernel_idx],
                 head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
        }
        else {
            RysERI_direct<<<num_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                (d_fock_matrix_replicas, d_density_matrix, d_primitive_shells, num_fock_replicas, 
                 d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
                 d_primitive_shell_pair_indices, num_basis, d_boys_grid, head_bra, head_ket);
        }
        kernel_idx++;
    
        if (verbose) {
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2) << shell_type_to_shell_name(s3) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s3) << "|=" << shell_s1.count << ", ";
            std::cout << "|bra|= " << num_bra << ", " ;
            std::cout << "|ket|= " << num_ket << ", " ;
            std::cout << "|braket|= " << num_braket << ", " ;
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }
    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }

    const int num_blocks_fock = ((num_basis * (num_basis + 1) / 2) + num_threads_per_block - 1) / num_threads_per_block;
    composeFockMatrix<<<num_blocks_fock, num_threads_per_block>>>(d_fock_matrix, d_fock_matrix_replicas, d_core_hamiltonian_matrix, num_basis, num_fock_replicas);

    cudaDeviceSynchronize();
}
/**/





//*
__global__ void densityMatrixDifferenceShellPairsKernel(
    real_t* g_density_matrix_diff_shell, 
    const real_t* g_density_matrix_diff, 
    const PrimitiveShell* g_primitive_shells, 
    const int2* g_primitive_shell_pair_indices, 
    const int num_primitive_shells, 
    const int num_basis)
{
    const int serial = blockDim.x * blockIdx.x + threadIdx.x;
    if (serial >= (num_primitive_shells * (num_primitive_shells + 1) / 2)) {
        return;
    }
    //const int shell_index_a = serial / num_primitive_shells;
    //const int shell_index_b = serial % num_primitive_shells;
    const int2 shell_pair_index = g_primitive_shell_pair_indices[serial];
    const int shell_index_a = shell_pair_index.x;
    const int shell_index_b = shell_pair_index.y;
    const PrimitiveShell a = g_primitive_shells[shell_index_a];
    const PrimitiveShell b = g_primitive_shells[shell_index_b];

    const int shell_size_a = (a.shell_type == 0) ? 1 : 3;
    const int shell_size_b = (b.shell_type == 0) ? 1 : 3;

    //real_t diff_abs;
    real_t max_value = 0.0;
    for (int i = 0; i < shell_size_a; ++i) {
        for (int j = 0; j < shell_size_b; ++j) {
            const int mu = a.basis_index + i;
            const int nu = b.basis_index + j;
            //diff_abs = fabs(g_density_matrix_diff[num_basis * mu + nu]);
            //if (diff_abs > max_value) {
            //    max_value = diff_abs;
            //}
            //const real_t density_diff = g_density_matrix_diff[(mu < nu) ? (num_basis * mu + nu) : (num_basis * nu + mu)];
            const real_t density_diff = g_density_matrix_diff[num_basis * mu + nu];
            max_value = fmax(max_value, fabs(density_diff));
        }
    }
    g_density_matrix_diff_shell[num_primitive_shells * shell_index_a + shell_index_b] = max_value;
    if (shell_index_a != shell_index_b) {
        g_density_matrix_diff_shell[num_primitive_shells * shell_index_b + shell_index_a] = max_value;
    }
}

void makeDensityMatrixDifferenceShellPairs(
    real_t* d_density_matrix_diff_shell, 
    const real_t* d_density_matrix_diff, 
    const PrimitiveShell* d_primitive_shells, 
    const int2* d_primitive_shell_pair_indices, 
    const int num_primitive_shells, 
    const int num_basis)
{
    const int threads_per_block = 256;
    const int num_primitive_shell_pairs = num_primitive_shells * (num_primitive_shells + 1) / 2;
    const int num_blocks = (num_primitive_shell_pairs + threads_per_block - 1) / threads_per_block;
    densityMatrixDifferenceShellPairsKernel<<<num_blocks, threads_per_block>>>(d_density_matrix_diff_shell, d_density_matrix_diff, d_primitive_shells, d_primitive_shell_pair_indices, num_primitive_shells, num_basis);
    cudaDeviceSynchronize();
}

void computeFockMatrix_Direct_RHF(
    const real_t* d_density_matrix,
    real_t* d_density_matrix_diff,
    real_t* d_density_matrix_diff_shell,
    const real_t* d_core_hamiltonian_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const int2* d_primitive_shell_pair_indices,
    const real_t* d_cgto_normalization_factors, 
    const real_t* d_boys_grid, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t schwarz_screening_threshold,
    real_t* d_fock_matrix,
    real_t* d_fock_matrix_prev,
    const int num_basis,
    std::vector<int*>& d_global_counters,
    std::vector<int*>& d_min_skipped_columns,
    real_t* d_fock_matrix_replicas,
    const int num_fock_replicas,
    const int verbose,
    bool& is_first_call)
{
    if (is_first_call) {
        cudaMemset(d_fock_matrix_prev, 0, sizeof(real_t) * num_basis * num_basis);
        cudaMemset(d_density_matrix_diff, 0, sizeof(real_t) * num_basis * num_basis);
    }
    // D_diff = D_new - D_old
    matrixSubtractionInPlace(d_density_matrix, d_density_matrix_diff, d_density_matrix_diff, num_basis);
    int num_primitive_shells = 0;
    for (const auto& x : shell_type_infos) {
        num_primitive_shells += x.count;
    }
    makeDensityMatrixDifferenceShellPairs(d_density_matrix_diff_shell, d_density_matrix_diff, d_primitive_shells, d_primitive_shell_pair_indices, num_primitive_shells, num_basis);
    cudaDeviceSynchronize();

    //int h_num_screened_shell_quartets = 0;
    //int* d_num_screened_shell_quartets;
    //cudaMalloc(&d_num_screened_shell_quartets, sizeof(int));
    //cudaMemset(d_num_screened_shell_quartets, 0, sizeof(int));

    // compute the electron repulsion integrals
    const int num_threads_per_block = 256;
    const int shell_type_count = shell_type_infos.size();
    cudaMemset(d_fock_matrix_replicas, 0, sizeof(real_t) * num_basis * num_basis * num_fock_replicas);

    // list shell-quadruples for sorted shell-type (s0, s1, s2, s3)
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    // reverse the order of the shell_quadruples to make it sorted by (s0, s1, s2, s3)
    // e.g. (pp|pp), (sp|pp), (sp|sp), (ss|pp), (ss|sp), (ss|ss) for s and p shells
    //std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    // make multi stream
    const int num_kernels = shell_quadruples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int kernel_idx = 0;
    const int task_group_size = 16;
    const int num_cuda_blocks = 256;
    for (const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket; // equal to the number of threads
        const size_t num_blocks = (num_braket + num_threads_per_block - 1) / num_threads_per_block; // the number of blocks
        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;

        if (s0 <= 1 && s1 <= 1 && s2 <= 1 && s3 <= 1) {
            // initialzie global counters and minimum skipped columns for dynamic screening
            const int num_bra_groups = (num_bra + task_group_size - 1) / task_group_size;
            const int num_ket_groups = (num_ket + task_group_size - 1) / task_group_size;
            const int num_init_blocks = (num_bra_groups + num_threads_per_block - 1) / num_threads_per_block;
            //cudaMemset(d_global_counters[kernel_idx], 0, sizeof(int) * num_bra_groups);
            cudaMemsetAsync(d_global_counters[kernel_idx], 0, sizeof(int) * num_bra_groups, streams[kernel_idx]);
            initializeMinSkippedColumns<<<num_init_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>(d_min_skipped_columns[kernel_idx], num_bra_groups, num_ket_groups);

            //if (s0 == 0 && s1 == 0 && s2 == 0 && s3 == 0) {
            if (false) {
                //ssss2e_dynamic_test<<<num_cuda_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                //    (d_fock_matrix_replicas, d_primitive_shells, d_primitive_shell_pair_indices, 
                //     d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                //     schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
                //     num_basis, num_primitive_shells, 
                //     d_boys_grid, d_density_matrix_diff, d_density_matrix_diff_shell, 
                //     d_global_counters[kernel_idx], d_min_skipped_columns[kernel_idx],
                //     head_bra, head_ket, num_bra, num_ket, num_fock_replicas, d_num_screened_shell_quartets);
            }
            else {
                //gpu::get_eri_kernel_direct(s0, s1, s2, s3)<<<num_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                //    (d_fock_matrix_replicas, d_primitive_shells, d_primitive_shell_pair_indices, 
                //     d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                //     num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
                //     num_basis, d_boys_grid, d_density_matrix, head_bra, head_ket, num_fock_replicas);
                gpu::get_eri_kernel_dynamic(s0, s1, s2, s3)<<<num_cuda_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                    (d_fock_matrix_replicas, d_primitive_shells, d_primitive_shell_pair_indices, 
                     d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                     schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
                     num_basis, num_primitive_shells, 
                     d_boys_grid, d_density_matrix_diff, d_density_matrix_diff_shell, 
                     d_global_counters[kernel_idx], d_min_skipped_columns[kernel_idx],
                     head_bra, head_ket, num_bra, num_ket, num_fock_replicas);
                }
        }
        else {
            RysERI_direct<<<num_blocks, num_threads_per_block, 0, streams[kernel_idx]>>>
                (d_fock_matrix_replicas, d_density_matrix_diff, d_primitive_shells, num_fock_replicas, 
                 d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, 
                 num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, 
                 d_primitive_shell_pair_indices, num_basis, d_boys_grid, head_bra, head_ket);
        }
        kernel_idx++;
    
        if (verbose) {
            std::cout << "(" << shell_type_to_shell_name(s0) << shell_type_to_shell_name(s1) << "|" << shell_type_to_shell_name(s2) << shell_type_to_shell_name(s3) << "): ";
            std::cout << "|" << shell_type_to_shell_name(s0) << "|=" << shell_s0.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s1) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s2) << "|=" << shell_s1.count << ", ";
            std::cout << "|" << shell_type_to_shell_name(s3) << "|=" << shell_s1.count << ", ";
            std::cout << "|bra|= " << num_bra << ", " ;
            std::cout << "|ket|= " << num_ket << ", " ;
            std::cout << "|braket|= " << num_braket << ", " ;
            std::cout << "num_blocks: " << num_blocks << std::endl;
        }
    }
    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }

    const int num_blocks_fock = ((num_basis * (num_basis + 1) / 2) + num_threads_per_block - 1) / num_threads_per_block;
    //composeFockMatrix<<<num_blocks_fock, num_threads_per_block>>>(d_fock_matrix, d_fock_matrix_replicas, d_core_hamiltonian_matrix, num_basis, num_fock_replicas, is_first_call);
    composeFockMatrix<<<num_blocks_fock, num_threads_per_block>>>(d_fock_matrix_prev, d_fock_matrix_replicas, d_core_hamiltonian_matrix, num_basis, num_fock_replicas, is_first_call);
    cudaMemcpy(d_fock_matrix, d_fock_matrix_prev, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToDevice);

    //cudaDeviceSynchronize();
    // update D_old = D_new for the next iteration
    cudaMemcpy(d_density_matrix_diff, d_density_matrix, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToDevice);

    if (is_first_call) {
        is_first_call = false;
    }

    //cudaMemcpy(&h_num_screened_shell_quartets, d_num_screened_shell_quartets, sizeof(int), cudaMemcpyDeviceToHost);
    //std::cout << "Number of screened shell quartets: " << h_num_screened_shell_quartets << std::endl;
}
/**/









void computeMullikenPopulation_RHF(
        const real_t* d_density_matrix,
        const real_t* d_overlap_matrix,
        real_t* mulliken_population_basis,
        const int num_basis
    )
{

    cudaError_t err;

    real_t* d_mulliken_population = nullptr;
    err = gansu::tracked_cudaMalloc(&d_mulliken_population, num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Mulliken population: ") + std::string(cudaGetErrorString(err)));
    }

    // Compute the diagonal elements of the product of the density matrix and the overlap matrix
    const size_t threads_per_block = 256;
    const size_t num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
    compute_diagonal_of_product<<<num_blocks, threads_per_block>>>(
        d_density_matrix, 
        d_overlap_matrix, 
        d_mulliken_population, 
        num_basis
    );

    // Copy the result to the host
    cudaMemcpy(mulliken_population_basis, d_mulliken_population, num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Free the memory for the temporary matrix
    gansu::tracked_cudaFree(d_mulliken_population);
}

void computeMullikenPopulation_UHF(
        const real_t* d_density_matrix_a,
        const real_t* d_density_matrix_b,
        const real_t* overlap_matrix,
        real_t* mulliken_population_basis,
        const int num_basis
    )
{
    cudaError_t err;

    real_t* d_mulliken_population = nullptr;
    err = gansu::tracked_cudaMalloc(&d_mulliken_population, num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Mulliken population: ") + std::string(cudaGetErrorString(err)));
    }

    // Compute the diagonal elements of the product of the density matrix and the overlap matrix
    const size_t threads_per_block = 256;
    const size_t num_blocks = (num_basis + threads_per_block - 1) / threads_per_block;
    compute_diagonal_of_product_sum<<<num_blocks, threads_per_block>>>(
        d_density_matrix_a, 
        d_density_matrix_b, 
        overlap_matrix, 
        d_mulliken_population, 
        num_basis
    );

    // Copy the result to the host
    cudaMemcpy(mulliken_population_basis, d_mulliken_population, num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Free the memory for the temporary matrix
    gansu::tracked_cudaFree(d_mulliken_population);
}


void computeDensityOverlapMatrix(
        const real_t* d_density_matrix,
        const real_t* d_overlap_matrix,
        real_t* result_matrix,
        const int num_basis
    )
{
    cudaError_t err;

    real_t* d_result_matrix = nullptr;
    err = gansu::tracked_cudaMalloc(&d_result_matrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Density-Overlap-Matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // result_matrix = D * S
    matrixMatrixProduct(d_density_matrix, d_overlap_matrix, d_result_matrix, num_basis, false, false);

    // Data transfer to host
    cudaMemcpy(result_matrix, d_result_matrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Free the memory for the temporary matrix
    gansu::tracked_cudaFree(d_result_matrix);
}

void computeSqrtOverlapDensitySqrtOverlapMatrix(
        const real_t* d_density_matrix,
        const real_t* d_sqrt_overlap_matrix,
        real_t* result_matrix,
        const int num_basis
    )
{
    cudaError_t err;

    // Compute the intermediate matrix: temp_matrix = S^(1/2) * D * S^(1/2)

    // Eigen decomposition of the overlap matrix S = U * diag(eigval) * U^T
    real_t* d_eigval = nullptr;
    err = gansu::tracked_cudaMalloc(&d_eigval, sizeof(real_t)*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for eigen values: ") + std::string(cudaGetErrorString(err)));
    }
    real_t* d_eigvec = nullptr;
    err = gansu::tracked_cudaMalloc(&d_eigvec, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for eigen vectors: ") + std::string(cudaGetErrorString(err)));
    }
    gpu::eigenDecomposition(d_sqrt_overlap_matrix, d_eigval, d_eigvec, num_basis);


    // Compute diag(eigval)^(1/2)
    gpu::sqrtElements(d_eigval, num_basis);


    // Make diag(eigval)^(1/2) matrix
    real_t* d_sqrt_eigval_matrix = nullptr;
    err = gansu::tracked_cudaMalloc(&d_sqrt_eigval_matrix, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for sqrt eigen value matrix: ") + std::string(cudaGetErrorString(err)));
    }
    gpu::makeDiagonalMatrix(d_eigval, d_sqrt_eigval_matrix, num_basis);

    // temp_matrix = U * diag(eigval)^(1/2) * U^T
    real_t* d_temp_matrix = nullptr;
    err = gansu::tracked_cudaMalloc(&d_temp_matrix, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }
    // temp_matrix = U * diag(eigval)^(1/2)
    gpu::matrixMatrixProduct(d_eigvec, d_sqrt_eigval_matrix, d_temp_matrix, num_basis, false, false);
    // S_sqrt = temp_matrix * U^T
    real_t* d_S_sqrt = nullptr;
    err = gansu::tracked_cudaMalloc(&d_S_sqrt, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for S_sqrt matrix: ") + std::string(cudaGetErrorString(err)));
    }
    gpu::matrixMatrixProduct(d_temp_matrix, d_eigvec, d_S_sqrt, num_basis, false, true);

    // result_matrix = S^(1/2) * D * S^(1/2)
    real_t* d_result_matrix = nullptr;
    err = gansu::tracked_cudaMalloc(&d_result_matrix, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for result matrix: ") + std::string(cudaGetErrorString(err)));
    }
    gpu::matrixMatrixProduct(d_S_sqrt, d_density_matrix, d_temp_matrix, num_basis, false, false);
    gpu::matrixMatrixProduct(d_temp_matrix, d_S_sqrt, d_result_matrix, num_basis, false, false);

    // Data transfer to host
    cudaMemcpy(result_matrix, d_result_matrix, sizeof(real_t)*num_basis*num_basis, cudaMemcpyDeviceToHost);

    // Free the memory for the temporary matrices
    gansu::tracked_cudaFree(d_eigval);
    gansu::tracked_cudaFree(d_eigvec);
    gansu::tracked_cudaFree(d_sqrt_eigval_matrix);
    gansu::tracked_cudaFree(d_temp_matrix);
    gansu::tracked_cudaFree(d_S_sqrt);
    gansu::tracked_cudaFree(d_result_matrix);

}




void constructERIHash(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_boys_grid,
    const real_t* d_cgto_normalization_factors,
    const real_t* d_schwarz_upper_bound_factors,
    real_t schwarz_screening_threshold,
    unsigned long long* d_hash_keys,
    real_t* d_hash_values,
    size_t hash_capacity_mask,
    const int num_basis,
    const bool verbose)
{
    const int threads_per_block = 256;
    const int shell_type_count = shell_type_infos.size();

    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a)
        for (int b = a; b < shell_type_count; ++b)
            for (int c = 0; c < shell_type_count; ++c)
                for (int d = c; d < shell_type_count; ++d)
                    if (a < c || (a == c && b <= d))
                        shell_quadruples.emplace_back(a, b, c, d);

    const int num_kernels = shell_quadruples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; ++i)
        cudaStreamCreate(&streams[i]);

    int stream_id = 0;
    for (const auto& quadruple : shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0 == s1) ? shell_s0.count * (shell_s0.count + 1) / 2 : shell_s0.count * shell_s1.count;
        const size_t num_ket = (s2 == s3) ? shell_s2.count * (shell_s2.count + 1) / 2 : shell_s2.count * shell_s3.count;
        const size_t num_braket = ((s0 == s2) && (s1 == s3)) ? num_bra * (num_bra + 1) / 2 : num_bra * num_ket;
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block;

        const size_t head_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
        const size_t head_ket = shell_pair_type_infos[get_index_2to1_horizontal(s2, s3, shell_type_count)].start_index;

        int a = s0, b = s1, c = s2, dd = s3;
        if (a > b) std::swap(a, b);
        if (c > dd) std::swap(c, dd);
        if (a > c || (a == c && b > dd)) { std::swap(a, c); std::swap(b, dd); }

#define HASH_KERNEL_ARGS \
    d_hash_keys, d_hash_values, hash_capacity_mask, \
    d_primitive_shells, d_cgto_normalization_factors, \
    shell_s0, shell_s1, shell_s2, shell_s3, \
    num_braket, schwarz_screening_threshold, d_schwarz_upper_bound_factors, \
    num_basis, d_boys_grid, head_bra, head_ket, hash_capacity_mask

        if      (a==0 && b==0 && c==0 && dd==0) ssss2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else if (a==0 && b==0 && c==0 && dd==1) sssp2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else if (a==0 && b==0 && c==1 && dd==1) sspp2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else if (a==0 && b==1 && c==0 && dd==1) spsp2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else if (a==0 && b==1 && c==1 && dd==1) sppp2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else if (a==1 && b==1 && c==1 && dd==1) pppp2e_hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);
        else RysERI_Hash<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(HASH_KERNEL_ARGS);

#undef HASH_KERNEL_ARGS
    }

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after Hash ERI kernels: " << cudaGetErrorString(err) << std::endl;
    }
    for (int i = 0; i < num_kernels; i++)
        cudaStreamDestroy(streams[i]);
}


// constructERICOO is deprecated — use constructERIHash instead


void computeFockMatrix_Hash_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const unsigned long long* d_coo_keys,
    const real_t* d_coo_values,
    size_t num_entries,
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    const int matrix_size = num_basis * num_basis;
    const int threads_per_block = 256;
    const int num_fock_replicas = 8;

    // Allocate and zero Fock replicas
    real_t* d_fock_replicas = nullptr;
    gansu::tracked_cudaMalloc(&d_fock_replicas, matrix_size * num_fock_replicas * sizeof(real_t));
    cudaMemset(d_fock_replicas, 0, matrix_size * num_fock_replicas * sizeof(real_t));

    // Push ERI contributions into replicas
    const int num_blocks = (num_entries + threads_per_block - 1) / threads_per_block;
    computeFockMatrix_COO_Push_RHF_kernel<<<num_blocks, threads_per_block>>>(
        d_density_matrix, d_coo_keys, d_coo_values, num_entries,
        d_fock_replicas, num_basis, num_fock_replicas);

    // Sum replicas + add core Hamiltonian into final Fock matrix
    cudaMemset(d_fock_matrix, 0, matrix_size * sizeof(real_t));
    const int num_utm = num_basis * (num_basis + 1) / 2;
    const int num_blocks_compose = (num_utm + threads_per_block - 1) / threads_per_block;
    composeFockMatrix<<<num_blocks_compose, threads_per_block>>>(
        d_fock_matrix, d_fock_replicas, d_core_hamiltonian_matrix,
        num_basis, num_fock_replicas, true);

    gansu::tracked_cudaFree(d_fock_replicas);
}

void computeFockMatrix_Hash_Indexed_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const unsigned long long* d_hash_keys,
    const real_t* d_hash_values,
    const size_t* d_nonzero_indices,
    size_t num_nonzero,
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    const int matrix_size = num_basis * num_basis;
    const int threads_per_block = 256;
    const int num_fock_replicas = 8;

    real_t* d_fock_replicas = nullptr;
    gansu::tracked_cudaMalloc(&d_fock_replicas, matrix_size * num_fock_replicas * sizeof(real_t));
    cudaMemset(d_fock_replicas, 0, matrix_size * num_fock_replicas * sizeof(real_t));

    const int num_blocks = (num_nonzero + threads_per_block - 1) / threads_per_block;
    computeFockMatrix_Hash_Push_Indexed_RHF_kernel<<<num_blocks, threads_per_block>>>(
        d_density_matrix, d_hash_keys, d_hash_values, d_nonzero_indices, num_nonzero,
        d_fock_replicas, num_basis, num_fock_replicas);

    cudaMemset(d_fock_matrix, 0, matrix_size * sizeof(real_t));
    const int num_utm = num_basis * (num_basis + 1) / 2;
    const int num_blocks_compose = (num_utm + threads_per_block - 1) / threads_per_block;
    composeFockMatrix<<<num_blocks_compose, threads_per_block>>>(
        d_fock_matrix, d_fock_replicas, d_core_hamiltonian_matrix,
        num_basis, num_fock_replicas, true);

    gansu::tracked_cudaFree(d_fock_replicas);
}

void computeFockMatrix_Hash_FullScan_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const unsigned long long* d_hash_keys,
    const real_t* d_hash_values,
    size_t hash_capacity,
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    const int matrix_size = num_basis * num_basis;
    const int threads_per_block = 256;
    const int num_fock_replicas = 8;

    real_t* d_fock_replicas = nullptr;
    gansu::tracked_cudaMalloc(&d_fock_replicas, matrix_size * num_fock_replicas * sizeof(real_t));
    cudaMemset(d_fock_replicas, 0, matrix_size * num_fock_replicas * sizeof(real_t));

    const size_t num_blocks = (hash_capacity + threads_per_block - 1) / threads_per_block;
    computeFockMatrix_Hash_Push_RHF_kernel<<<num_blocks, threads_per_block>>>(
        d_density_matrix, d_hash_keys, d_hash_values, hash_capacity,
        d_fock_replicas, num_basis, num_fock_replicas);

    cudaMemset(d_fock_matrix, 0, matrix_size * sizeof(real_t));
    const int num_utm = num_basis * (num_basis + 1) / 2;
    const int num_blocks_compose = (num_utm + threads_per_block - 1) / threads_per_block;
    composeFockMatrix<<<num_blocks_compose, threads_per_block>>>(
        d_fock_matrix, d_fock_replicas, d_core_hamiltonian_matrix,
        num_basis, num_fock_replicas, true);

    gansu::tracked_cudaFree(d_fock_replicas);
}






// before = |after - before|
__global__ void calcDiffMatrix(const real_t* d_matrix_after, real_t* d_matrix_before, int size) {
    size_t id = blockIdx.x*blockDim.x + threadIdx.x;
    if(id >= size * size) return;

    d_matrix_before[id] = d_matrix_after[id] - d_matrix_before[id];
}

__global__ void makeIdentityMatrix(real_t* d_I, int size) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size * size) return;

    int row = id % size;
    int col = id / size;
    d_I[id] = (row == col) ? 1.0 : 0.0;
}

void computeInverseByDtrsm(real_t* two_center_eris, real_t* two_center_eris_inverse, int num_auxiliary_basis){
    cublasHandle_t cublasHandle = GPUHandle::cublas();

    const int num_threads = 1024;
    const int num_blocks = (num_auxiliary_basis * num_auxiliary_basis + num_threads - 1) / num_threads;
    makeIdentityMatrix<<<num_blocks, num_threads>>>(two_center_eris_inverse, num_auxiliary_basis);
    cudaDeviceSynchronize();

    const real_t alpha = 1.0;

    cublasDtrsm(
        cublasHandle,
        CUBLAS_SIDE_LEFT,        
        CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N,            
        CUBLAS_DIAG_NON_UNIT,   
        num_auxiliary_basis,                   
        num_auxiliary_basis,                   
        &alpha,
        two_center_eris, num_auxiliary_basis,                  
        two_center_eris_inverse, num_auxiliary_basis                  
    );
}




void compute_RI_Direct_c_array(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_nomalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
    real_t* d_c, 
    const real_t* d_density_matrix,
    const size_t2* d_primitive_shell_pair_indices,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const double schwarz_screening_threshold, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const bool verbose)
{
const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    // Call the kernel functions from (ss|s),... (e.g. (ss|s), (ss|p), (sp|s), (sp|p), (pp|s), (pp|p) for s and p shells)

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<cudaStream_t> streams(num_kernels);

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& triple: shell_triples) {
        int s0, s1, s2;
        std::tie(s0, s1, s2) = triple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

        const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
        
        direct_ri_c_J_kernel_t c_kernel;

        if(s0==0 && s1==0 && s2==0) c_kernel = compute_RI_Direct_c_kernel_sss;
        else if(s0==0 && s1==0 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ssp;
        else if(s0==0 && s1==0 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ssd;
        else if(s0==0 && s1==0 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ssf;
        else if(s0==0 && s1==1 && s2==0) c_kernel = compute_RI_Direct_c_kernel_sps;
        else if(s0==0 && s1==1 && s2==1) c_kernel = compute_RI_Direct_c_kernel_spp;
        else if(s0==0 && s1==1 && s2==2) c_kernel = compute_RI_Direct_c_kernel_spd;
        else if(s0==0 && s1==1 && s2==3) c_kernel = compute_RI_Direct_c_kernel_spf;
        else if(s0==1 && s1==1 && s2==0) c_kernel = compute_RI_Direct_c_kernel_pps;
        else if(s0==1 && s1==1 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ppp;
        else if(s0==1 && s1==1 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ppd;
        else if(s0==1 && s1==1 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ppf;
        #if defined(COMPUTE_D_BASIS)
        else if(s0==0 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_sds;
        else if(s0==0 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_sdp;
        else if(s0==0 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_sdd;
        else if(s0==0 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_sdf;
        else if(s0==1 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_pds;
        else if(s0==1 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_pdp;
        else if(s0==1 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_pdd;
        else if(s0==1 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_pdf;
        else if(s0==2 && s1==2 && s2==0) c_kernel = compute_RI_Direct_c_kernel_dds;
        else if(s0==2 && s1==2 && s2==1) c_kernel = compute_RI_Direct_c_kernel_ddp;
        else if(s0==2 && s1==2 && s2==2) c_kernel = compute_RI_Direct_c_kernel_ddd;
        else if(s0==2 && s1==2 && s2==3) c_kernel = compute_RI_Direct_c_kernel_ddf;
        #endif
        else c_kernel = compute_RI_Direct_c_kernel;

        // c_kernel = compute_RI_Direct_c_kernel;


        c_kernel<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_c, d_density_matrix, d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors, 
                                                                                shell_s0, shell_s1, shell_s2, 
                                                                                num_tasks, num_basis, 
                                                                                &d_primitive_shell_pair_indices[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                &d_schwarz_upper_bound_factors[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                d_auxiliary_schwarz_upper_bound_factors,
                                                                                schwarz_screening_threshold,
                                                                                num_auxiliary_basis,
                                                                                d_boys_grid);
    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

void compute_RI_Direct_J_matrix(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_nomalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
    real_t* d_J, 
    const real_t* d_t,
    const size_t2* d_primitive_shell_pair_indices,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const double schwarz_screening_threshold, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const bool verbose)
{
const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    // Call the kernel functions from (ss|s),... (e.g. (ss|s), (ss|p), (sp|s), (sp|p), (pp|s), (pp|p) for s and p shells)

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<cudaStream_t> streams(num_kernels);

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& triple: shell_triples) {
        int s0, s1, s2;
        std::tie(s0, s1, s2) = triple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

        const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
        
        direct_ri_c_J_kernel_t J_kernel;

        if(s0==0 && s1==0 && s2==0) J_kernel = compute_RI_Direct_J_kernel_sss;
        else if(s0==0 && s1==0 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ssp;
        else if(s0==0 && s1==0 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ssd;
        else if(s0==0 && s1==0 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ssf;
        else if(s0==0 && s1==1 && s2==0) J_kernel = compute_RI_Direct_J_kernel_sps;
        else if(s0==0 && s1==1 && s2==1) J_kernel = compute_RI_Direct_J_kernel_spp;
        else if(s0==0 && s1==1 && s2==2) J_kernel = compute_RI_Direct_J_kernel_spd;
        else if(s0==0 && s1==1 && s2==3) J_kernel = compute_RI_Direct_J_kernel_spf;
        else if(s0==1 && s1==1 && s2==0) J_kernel = compute_RI_Direct_J_kernel_pps;
        else if(s0==1 && s1==1 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ppp;
        else if(s0==1 && s1==1 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ppd;
        else if(s0==1 && s1==1 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ppf;
        #if defined(COMPUTE_D_BASIS)
        else if(s0==0 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_sds;
        else if(s0==0 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_sdp;
        else if(s0==0 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_sdd;
        else if(s0==0 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_sdf;
        else if(s0==1 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_pds;
        else if(s0==1 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_pdp;
        else if(s0==1 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_pdd;
        else if(s0==1 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_pdf;
        else if(s0==2 && s1==2 && s2==0) J_kernel = compute_RI_Direct_J_kernel_dds;
        else if(s0==2 && s1==2 && s2==1) J_kernel = compute_RI_Direct_J_kernel_ddp;
        else if(s0==2 && s1==2 && s2==2) J_kernel = compute_RI_Direct_J_kernel_ddd;
        else if(s0==2 && s1==2 && s2==3) J_kernel = compute_RI_Direct_J_kernel_ddf;
        #endif
        else J_kernel = compute_RI_Direct_J_kernel;

        // J_kernel = compute_RI_Direct_J_kernel;

        J_kernel<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_J, d_t, d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors, 
                                                                                shell_s0, shell_s1, shell_s2, 
                                                                                num_tasks, num_basis, 
                                                                                &d_primitive_shell_pair_indices[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                &d_schwarz_upper_bound_factors[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                d_auxiliary_schwarz_upper_bound_factors,
                                                                                schwarz_screening_threshold,
                                                                                num_auxiliary_basis,
                                                                                d_boys_grid);
    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}



void computeFockMatrix_RI_Direct_RHF(const real_t* d_density_matrix, const real_t* d_coefficient_matrix,
                                    const real_t* d_L_inv, 
                                    real_t* d_decomposed_two_center_eris,
                                    const real_t* d_core_hamiltonian_matrix, 
                                    real_t* d_fock_matrix, 
                                    real_t* d_coefficient_matrix_prev,
                                    real_t* h_Z_tensor_prev,
                                    const std::vector<ShellTypeInfo>& shell_type_infos, 
                                    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
                                    const PrimitiveShell* h_primitive_shells, 
                                    const PrimitiveShell* d_primitive_shells, 
                                    const real_t* d_cgto_normalization_factors, 
                                    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
                                    const PrimitiveShell* d_auxiliary_primitive_shells, 
                                    const real_t* d_auxiliary_cgto_normalization_factors, 
                                    const size_t2* d_primitive_shell_pair_indices,
                                    const int num_basis,
                                    const int num_auxiliary_basis,
                                    const int num_electrons,
                                    const int num_primitive_shells,
                                    const real_t* d_boys_grid,
                                    const double schwarz_screening_threshold, 
                                    const real_t* d_schwarz_upper_bound_factors,
                                    const real_t* d_auxiliary_schwarz_upper_bound_factors,
                                    const bool verbose){


    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();


    cudaError_t err;

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    real_t* d_J = nullptr;
    err = gansu::tracked_cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_J, 0.0, sizeof(real_t)*num_basis*num_basis);



    // compute c_q = \sum_{a b} D_{a b} (q|ab)
    real_t *d_c = nullptr;
    err = gansu::tracked_cudaMalloc(&d_c, sizeof(real_t)*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for c vector: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_c, 0.0, sizeof(real_t)*num_auxiliary_basis);


    // cublas関数ように、column-majorにしておく
    transposeMatrixInPlace(d_decomposed_two_center_eris, num_auxiliary_basis);


    // cを求める
    compute_RI_Direct_c_array(shell_type_infos,
                              shell_pair_type_infos,
                              d_primitive_shells,
                              d_cgto_normalization_factors,
                              auxiliary_shell_type_infos,
                              d_auxiliary_primitive_shells,
                              d_auxiliary_cgto_normalization_factors,
                              d_c,
                              d_density_matrix,
                              d_primitive_shell_pair_indices,
                              num_basis,
                              num_auxiliary_basis,
                              d_boys_grid,
                              schwarz_screening_threshold,
                              d_schwarz_upper_bound_factors,
                              d_auxiliary_schwarz_upper_bound_factors,
                              verbose);
    cudaDeviceSynchronize();





    // Ly=cをyについて解く   
    cublasDtrsv(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER, 
        CUBLAS_OP_N,            
        CUBLAS_DIAG_NON_UNIT,   
        num_auxiliary_basis,                   
        d_decomposed_two_center_eris, num_auxiliary_basis,                  
        d_c, 1               
    );

    // L^T t = y をtについて解く
    cublasDtrsv(
        cublasHandle,       
        CUBLAS_FILL_MODE_LOWER, 
        CUBLAS_OP_T,            
        CUBLAS_DIAG_NON_UNIT,   
        num_auxiliary_basis,                                     
        d_decomposed_two_center_eris, num_auxiliary_basis,                  
        d_c, 1                
    );



    // Jmu nu = ()
    compute_RI_Direct_J_matrix(shell_type_infos,
                              shell_pair_type_infos,
                              d_primitive_shells,
                              d_cgto_normalization_factors,
                              auxiliary_shell_type_infos,
                              d_auxiliary_primitive_shells,
                              d_auxiliary_cgto_normalization_factors,
                              d_J,
                              d_c,
                              d_primitive_shell_pair_indices,
                              num_basis,
                              num_auxiliary_basis,
                              d_boys_grid,
                              schwarz_screening_threshold,
                              d_schwarz_upper_bound_factors,
                              d_auxiliary_schwarz_upper_bound_factors,
                              verbose);
    cudaDeviceSynchronize();

    gansu::tracked_cudaFree(d_c);


    transposeMatrixInPlace(d_decomposed_two_center_eris, num_auxiliary_basis);


    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_K = nullptr;
    err = gansu::tracked_cudaMalloc(&d_K, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for K matrix: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_K, 0.0, sizeof(real_t)*num_basis*num_basis);


    real_t* d_Z = nullptr;
    err = gansu::tracked_cudaMalloc(&d_Z, sizeof(real_t)*num_basis*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Z matrix: ") + std::string(cudaGetErrorString(err)));
    }

    real_t* d_Z_prev = nullptr;
    err = gansu::tracked_cudaMalloc(&d_Z_prev, sizeof(real_t)*num_basis*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Z_prev matrix: ") + std::string(cudaGetErrorString(err)));
    }


    real_t* d_W_diff = nullptr;
    err = gansu::tracked_cudaMalloc(&d_W_diff, sizeof(real_t)*num_basis*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for W matrix: ") + std::string(cudaGetErrorString(err)));
    }


    //Cとの差分を求める
    real_t* d_coefficient_matrix_diff;
    gansu::tracked_cudaMalloc((void**)&d_coefficient_matrix_diff, sizeof(real_t) * num_basis * num_basis);
    cudaMemcpy(d_coefficient_matrix_diff, d_coefficient_matrix_prev, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToDevice);
    calcDiffMatrix<<< ((num_basis * num_basis) + 1024 - 1) / 1024, 1024>>>(d_coefficient_matrix, d_coefficient_matrix_diff, num_basis); //d_coefficient_matrix_diff = |C_new - C_prev|
    
    
    transposeMatrixInPlace(d_coefficient_matrix_diff, num_basis);
    transposeMatrixInPlace(d_coefficient_matrix_prev, num_basis);



    const double alpha = 1.0, gamma = 2.0;

    const int row = num_basis, col = num_auxiliary_basis;


    cudaStream_t stream_main, stream_sub;
    cudaStreamCreateWithFlags(&stream_main, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream_sub, cudaStreamNonBlocking);




    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<cudaStream_t> streams(num_kernels);
    for(auto& stream : streams) cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    




    cublasSetStream(cublasHandle, stream_main);
    for(int iter = 0; iter < num_electrons/2; iter++) {
        cudaMemset(d_W_diff, 0.0, sizeof(real_t)*num_basis*num_auxiliary_basis);
        
        // d_schwarz_upper_bound_factors
        cudaDeviceSynchronize();

        // for-loop for sorted shell-type (s0, s1, s2, s3)
        int stream_id = 0;
        for(const auto& triple: shell_triples) {
            int s0, s1, s2;
            std::tie(s0, s1, s2) = triple;


            int bra_id = calcIdx_triangular_(s0, s1, shell_type_count);



            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

            const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
            const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
            
            direct_ri_w_kernel_t W_kernel;

            if(s0==0 && s1==0 && s2==0) W_kernel = compute_RI_Direct_W_kernel_sss;
            else if(s0==0 && s1==0 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ssp;
            else if(s0==0 && s1==0 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ssd;
            else if(s0==0 && s1==0 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ssf;
            else if(s0==0 && s1==1 && s2==0) W_kernel = compute_RI_Direct_W_kernel_sps;
            else if(s0==0 && s1==1 && s2==1) W_kernel = compute_RI_Direct_W_kernel_spp;
            else if(s0==0 && s1==1 && s2==2) W_kernel = compute_RI_Direct_W_kernel_spd;
            else if(s0==0 && s1==1 && s2==3) W_kernel = compute_RI_Direct_W_kernel_spf;
            else if(s0==1 && s1==1 && s2==0) W_kernel = compute_RI_Direct_W_kernel_pps;
            else if(s0==1 && s1==1 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ppp;
            else if(s0==1 && s1==1 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ppd;
            else if(s0==1 && s1==1 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ppf;
            #if defined(COMPUTE_D_BASIS)            
            else if(s0==0 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_sds;
            else if(s0==0 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_sdp;
            else if(s0==0 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_sdd;
            else if(s0==0 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_sdf;
            else if(s0==1 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_pds;
            else if(s0==1 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_pdp;
            else if(s0==1 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_pdd;
            else if(s0==1 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_pdf;
            else if(s0==2 && s1==2 && s2==0) W_kernel = compute_RI_Direct_W_kernel_dds;
            else if(s0==2 && s1==2 && s2==1) W_kernel = compute_RI_Direct_W_kernel_ddp;
            else if(s0==2 && s1==2 && s2==2) W_kernel = compute_RI_Direct_W_kernel_ddd;
            else if(s0==2 && s1==2 && s2==3) W_kernel = compute_RI_Direct_W_kernel_ddf;
            #endif
            else W_kernel = compute_RI_Direct_W_kernel;
            // W_kernel = compute_RI_Direct_W_kernel;

            W_kernel<<<num_blocks, threads_per_block, 0, streams[stream_id]>>>(d_W_diff, &d_coefficient_matrix_diff[iter * num_basis], d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                                    d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors, 
                                                                                                    shell_s0, shell_s1, shell_s2, 
                                                                                                    num_tasks, num_basis, 
                                                                                                    &d_primitive_shell_pair_indices[shell_pair_type_infos[bra_id].start_index],
                                                                                                    &d_schwarz_upper_bound_factors[shell_pair_type_infos[bra_id].start_index],
                                                                                                    d_auxiliary_schwarz_upper_bound_factors,
                                                                                                    schwarz_screening_threshold,
                                                                                                    num_auxiliary_basis,
                                                                                                    iter,
                                                                                                    d_boys_grid);

            stream_id++;
        }


    


        cudaMemcpyAsync(d_Z_prev, &h_Z_tensor_prev[(size_t)iter * num_basis * num_auxiliary_basis], sizeof(real_t) * num_basis * num_auxiliary_basis, cudaMemcpyHostToDevice, stream_sub);

        cudaDeviceSynchronize();



        cublasDgemm(
            cublasHandle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            num_auxiliary_basis,
            num_basis,
            num_auxiliary_basis,
            &alpha,
            d_L_inv, num_auxiliary_basis,
            d_W_diff, num_auxiliary_basis,
            &alpha,
            d_Z_prev, num_auxiliary_basis
        );
        




        cudaDeviceSynchronize();




        cublasDgemm(
            cublasHandle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            row,
            row,
            col,
            &gamma,
            d_Z_prev, col,
            d_Z_prev, col,
            &alpha,
            d_K, row
        );



        cudaMemcpyAsync(&h_Z_tensor_prev[(size_t)iter * num_basis * num_auxiliary_basis], d_Z_prev, sizeof(real_t) * num_basis*num_auxiliary_basis, cudaMemcpyDeviceToHost, stream_sub);
        
        cudaDeviceSynchronize();
    }
    cublasSetStream(cublasHandle, 0);
  


    for(auto& stream : streams) cudaStreamDestroy(stream);


    cudaMemcpyAsync(d_coefficient_matrix_prev, d_coefficient_matrix, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToDevice, stream_sub);
    // cudaMemcpyAsync(d_K_matrix_prev, d_K, sizeof(real_t) * num_basis * num_basis, cudaMemcpyDeviceToDevice, stream_sub);

    // ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // // F = H + J - (1/2)*K
    computeFockMatrix_RI_RHF_kernel<<<num_blocks, num_threads, 0, stream_main>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, num_basis);

    cudaDeviceSynchronize();

    cudaStreamDestroy(stream_main);
    cudaStreamDestroy(stream_sub);
    
    // free the memory
    gansu::tracked_cudaFree(d_J);
    gansu::tracked_cudaFree(d_K);
    gansu::tracked_cudaFree(d_Z);
    gansu::tracked_cudaFree(d_W_diff);
    gansu::tracked_cudaFree(d_Z_prev);  

    gansu::tracked_cudaFree(d_coefficient_matrix_diff);
}







// ============================================================
//  RI-Direct Fock build v2: BLAS-only J/K construction.
//  Computes raw 3-center ERIs ONCE, then B via L-solve,
//  then J and K entirely via DGEMM.
//
//  1. (μν|Q) via get_3center_kernel (default stream, int types for kernel args)
//  2. B = L^{-1} × (μν|Q) via solve_lower_triangular
//  3. J: W = B·D, J = B^T·W  (cublasDgemv)
//  4. X_q = C^T·B_q           (cublasDgemmStridedBatched)
//  5. K = 2·X_packed^T·X_packed (cublasDgemm)
//  6. F = H + J - K/2
// ============================================================
void computeFockMatrix_RI_Direct_v2(
    const real_t* d_density_matrix, const real_t* d_coefficient_matrix,
    const real_t* d_two_center_eris_cholesky,  // L (Cholesky factor, row-major)
    const real_t* d_L_inv_precomputed,         // L^{-1} (precomputed, row-major) — kept for API compat
    const real_t* d_core_hamiltonian_matrix, real_t* d_fock_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors,
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors,
    const size_t2* d_primitive_shell_pair_indices,
    int num_basis, int num_auxiliary_basis, int num_occ,
    const real_t* d_boys_grid,
    double schwarz_screening_threshold,
    const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors,
    bool verbose)
{
    cublasHandle_t handle = GPUHandle::cublas();
    // Use int for kernel arguments (get_3center_kernel expects int, NOT size_t)
    const int nao = num_basis;
    const int naux = num_auxiliary_basis;
    const size_t nao2 = (size_t)nao * nao;
    const double one = 1.0, zero = 0.0, two = 2.0;

    // -----------------------------------------------
    // Step 1: Compute raw 3-center ERIs (μν|Q) → d_B
    //   IMPORTANT: kernel expects int for num_basis/num_auxiliary_basis.
    //   Passing size_t would misalign subsequent pointer arguments.
    // -----------------------------------------------
    real_t* d_B = nullptr;
    gansu::tracked_cudaMalloc(&d_B, (size_t)naux * nao2 * sizeof(real_t));

    {
        const int threads_per_block = 128;
        const int shell_type_count = shell_type_infos.size();
        const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();

        for (int s0 = 0; s0 < shell_type_count; ++s0) {
            for (int s1 = s0; s1 < shell_type_count; ++s1) {
                for (int s2 = 0; s2 < auxiliary_shell_type_count; ++s2) {
                    const ShellTypeInfo shell_s0 = shell_type_infos[s0];
                    const ShellTypeInfo shell_s1 = shell_type_infos[s1];
                    const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

                    const int64_t num_tasks = ((s0 == s1)
                        ? ((int64_t)shell_s0.count * (shell_s0.count + 1) / 2)
                        : ((int64_t)shell_s0.count * shell_s1.count))
                        * (int64_t)shell_s2.count;
                    const int num_blocks = (int)((num_tasks + threads_per_block - 1) / threads_per_block);

                    const int pair_idx = calcIdx_triangular_(s0, s1, shell_type_count);

                    // 3-center kernels expect RELATIVE bra indices (local within type)
                    // and add start_index internally.  Direct RI's pair indices are ABSOLUTE,
                    // so pass start_index=0 for bra shells to avoid double-counting.
                    ShellTypeInfo shell_s0_nooff = shell_s0;  shell_s0_nooff.start_index = 0;
                    ShellTypeInfo shell_s1_nooff = shell_s1;  shell_s1_nooff.start_index = 0;

                    gpu::get_3center_kernel(s0, s1, s2)<<<num_blocks, threads_per_block>>>(
                        d_B, d_primitive_shells, d_auxiliary_primitive_shells,
                        d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors,
                        shell_s0_nooff, shell_s1_nooff, shell_s2,
                        num_tasks, nao,
                        &d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index],
                        &d_schwarz_upper_bound_factors[shell_pair_type_infos[pair_idx].start_index],
                        d_auxiliary_schwarz_upper_bound_factors,
                        schwarz_screening_threshold, naux, d_boys_grid);
                }
            }
        }
        cudaDeviceSynchronize();
    }

    // -----------------------------------------------
    // Step 2: B = L^{-1} × (μν|Q) via solve_lower_triangular
    // -----------------------------------------------
    real_t* d_L_copy = nullptr;
    gansu::tracked_cudaMalloc(&d_L_copy, (size_t)naux * naux * sizeof(real_t));
    cudaMemcpy(d_L_copy, d_two_center_eris_cholesky, (size_t)naux * naux * sizeof(real_t), cudaMemcpyDeviceToDevice);
    solve_lower_triangular(d_L_copy, d_B, naux, nao2);
    gansu::tracked_cudaFree(d_L_copy);

    // -----------------------------------------------
    // Step 3: J via cublasDgemv
    // -----------------------------------------------
    real_t* d_J = nullptr;
    real_t* d_W = nullptr;
    gansu::tracked_cudaMalloc(&d_J, nao2 * sizeof(real_t));
    gansu::tracked_cudaMalloc(&d_W, naux * sizeof(real_t));

    cublasDgemv(handle, CUBLAS_OP_T, nao2, naux, &one, d_B, nao2, d_density_matrix, 1, &zero, d_W, 1);
    cublasDgemv(handle, CUBLAS_OP_N, nao2, naux, &one, d_B, nao2, d_W, 1, &zero, d_J, 1);
    gansu::tracked_cudaFree(d_W);

    // -----------------------------------------------
    // Step 4: X_q(i, ν) = Σ_μ C(μ,i) B_q(μ,ν) via cublasDgemmStridedBatched
    // -----------------------------------------------
    real_t* d_X = nullptr;
    gansu::tracked_cudaMalloc(&d_X, (size_t)naux * nao * num_occ * sizeof(real_t));

    cublasDgemmStridedBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ, nao, nao,
        &one,
        d_coefficient_matrix, nao, 0,
        d_B, nao, (long long)nao * nao,
        &zero,
        d_X, num_occ, (long long)nao * num_occ,
        naux);

    gansu::tracked_cudaFree(d_B);

    // -----------------------------------------------
    // Step 5: K = 2·X_packed^T·X_packed
    // -----------------------------------------------
    real_t* d_X_packed = nullptr;
    gansu::tracked_cudaMalloc(&d_X_packed, (size_t)naux * nao * num_occ * sizeof(real_t));

    {
        const size_t total = (size_t)naux * nao * num_occ;
        const int th = 256;
        const int bl = (total + th - 1) / th;
        packThreeDimensionalTensorX<<<bl, th>>>(d_X, d_X_packed, nao, naux, num_occ);
    }
    gansu::tracked_cudaFree(d_X);

    real_t* d_K = nullptr;
    gansu::tracked_cudaMalloc(&d_K, nao2 * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nao, nao, naux * num_occ,
                &two, d_X_packed, naux * num_occ,
                d_X_packed, naux * num_occ,
                &zero, d_K, nao);
    gansu::tracked_cudaFree(d_X_packed);

    // -----------------------------------------------
    // Step 6: F = H + J - K/2
    // -----------------------------------------------
    {
        const int th = 256;
        const int bl = (nao2 + th - 1) / th;
        computeFockMatrix_RI_RHF_kernel<<<bl, th>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, nao);
        cudaDeviceSynchronize();
    }

    gansu::tracked_cudaFree(d_J);
    gansu::tracked_cudaFree(d_K);
}


/// @brief Check the validity and contents of the W matrix.
/// @param label 
/// @param W_matrix 
/// @param num_basis 
void print_W_Matrix(const char* label, const real_t* W_matrix, int num_basis){
    std::cout << "=== " << label << " ===" << std::endl;
    std::cout << "[\n";
    for(int i=0; i<num_basis; i++) {
        for(int j=0; j<num_basis; j++){
            if (j == 0) std::cout <<  "  [";

            std::cout << std::right << std::setfill(' ') << std::setw(10) << std::fixed << std::setprecision(6) << W_matrix[i*num_basis + j];

            if (j != num_basis - 1) std::cout << ",";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n\n";
}


/// @brief Check the validity and contents of the Gradient Matrix.
/// @param label 
/// @param grad 
/// @param num_atoms 
void printGradientMatrix(const char* label, const double* grad, int num_atoms) {
    std::cout << std::setfill(' '); 
    std::cout << "=== " << label << " ===" << std::endl;
    std::cout << "[\n";

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;

    for (int i = 0; i < num_atoms; ++i) {
        std::cout << "  [" 
                  << std::setw(14) << std::fixed << std::setprecision(8) << grad[3*i + 0] << ", "
                  << std::setw(14) << std::fixed << std::setprecision(8) << grad[3*i + 1] << ", "
                  << std::setw(14) << std::fixed << std::setprecision(8) << grad[3*i + 2] << " ]";
        if (i != num_atoms - 1) std::cout << ",";
        std::cout << "\n";
        sum_x += grad[3*i + 0];
        sum_y += grad[3*i + 1];
        sum_z += grad[3*i + 2];
    }
    std::cout << "]\n";

    std::cout << std::fixed << std::setprecision(8) << "(x, y, z) = (" << sum_x << ", " << sum_y << ", " << sum_z << ")" << std::endl;

    const double tol = 1e-8;
    std::cout << "Check if sums are zero: " << ((std::fabs(sum_x) < tol && std::fabs(sum_y) < tol && std::fabs(sum_z) < tol) ? "YES" : "NO") << std::endl << std::endl;
}


// 核座標微分をGPUで実装する前処理(重なり部分の係数Wの計算)
void compute_W(real_t* d_W_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_electron)
{
    const int threads_per_block = 128;
    const int blocks = (num_basis*num_basis + threads_per_block - 1) / threads_per_block;
    compute_W_Matrix_kernel<<<blocks, threads_per_block>>>(d_W_matrix, d_coefficient_matrix, d_orbital_energies, num_electron, num_basis);
}




// 各分子積分の微分を同時に計算
void computeMolucularGradients(double* d_grad_total, double* d_grad_N, double* d_grad_S, double* d_grad_K, double* d_grad_V, double* d_grad_G, real_t* d_W_matrix,
                                const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms, 
                                const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const PrimitiveShell* d_primitive_shells, 
                                const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const int num_atoms, const int num_basis, const int num_electron, const bool verbose)
{
    // block size, thread sizeの指定
    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();

    // 2電子部分の微分の前処理
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    // multi streamの作成
    int stream_id = 0;
    const int num_kernels = shell_quadruples.size() + 3*((shell_type_count)*(shell_type_count+1)/2) + 1;
    std::vector<cudaStream_t> streams(num_kernels);
    for(int i=0; i<num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }
    

    // 2電子部分の微分
    for(const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket; // equal to the number of threads
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block; // the number of blocks

        get_compute_gradients_repulsion()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_G, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, shell_s2, shell_s3, num_braket, num_basis, d_boys_grid);
    }

    // 1電子部分の微分
    for (int s0 = shell_type_count-1; s0 >= 0; s0--) {
        for (int s1 = shell_type_count-1; s1 >= s0; s1--) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];

            const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count); // the number of pairs of primitive shells = the number of threads
            const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block; // the number of blocks
            
            get_compute_gradients_overlap()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_S, d_W_matrix, d_primitive_shells, d_cgto_normalization_factors, num_basis, shell_s0, shell_s1, num_shell_pairs);
            get_compute_gradients_kinetic()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_K, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors, num_basis, shell_s0, shell_s1, num_shell_pairs);
            get_compute_gradients_nuclear()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_V, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors, d_atoms, num_atoms, num_basis, shell_s0, shell_s1, num_shell_pairs, d_boys_grid);
        }
    }

    const int NR_blocks = (num_atoms * num_atoms + threads_per_block - 1) / threads_per_block;
    compute_nuclear_repulsion_gradient_kernel<<<NR_blocks, threads_per_block, 0, streams[stream_id]>>>(d_grad_N, d_atoms, num_atoms);

    // syncronize streams
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        abort();
    }

    // destory streams
    for(int i=0; i<num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // 微分の影響を合計
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;

    cudaMemcpy(d_grad_total, d_grad_N, sizeof(double) * 3*num_atoms, cudaMemcpyDeviceToDevice);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_S, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_K, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_V, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_G, 1, d_grad_total, 1);

    cublasDestroy(handle);
}




// エネルギー微分を計算する関数
std::vector<double> computeEnergyGradient_RHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
                                const Atom* d_atoms, const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies,
                                const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,
                                const int num_atoms, const int num_basis, const int num_electron, const bool verbose)
{
    // メモリサイズ
    const int n = 3*num_atoms; // 配列のサイズ
    const size_t wmat_bytes = num_basis * num_basis * sizeof(real_t);
    const size_t gradients_bytes = n * sizeof(double);  // dx, dy, dz の計算結果を1次元配列に格納

    // CPU側のメモリ確保
    double* grad_total = nullptr;
    cudaMallocHost((void**)&grad_total, gradients_bytes);

    // GPU側のメモリ確保
    real_t* d_W_matrix = nullptr;
    double* d_grad_N = nullptr;
    double* d_grad_S = nullptr;
    double* d_grad_K = nullptr;
    double* d_grad_V = nullptr;
    double* d_grad_G = nullptr;
    double* d_grad_total = nullptr;

    cudaMalloc(&d_W_matrix, wmat_bytes);
    cudaMalloc(&d_grad_N, gradients_bytes);
    cudaMalloc(&d_grad_S, gradients_bytes);
    cudaMalloc(&d_grad_K, gradients_bytes);
    cudaMalloc(&d_grad_V, gradients_bytes);
    cudaMalloc(&d_grad_G, gradients_bytes);
    cudaMalloc(&d_grad_total, gradients_bytes);

    // GPUメモリの初期化
    cudaMemset(d_W_matrix, 0, wmat_bytes);
    cudaMemset(d_grad_N, 0, gradients_bytes);
    cudaMemset(d_grad_S, 0, gradients_bytes);
    cudaMemset(d_grad_K, 0, gradients_bytes);
    cudaMemset(d_grad_V, 0, gradients_bytes);
    cudaMemset(d_grad_G, 0, gradients_bytes);
    cudaMemset(d_grad_total, 0, gradients_bytes);

    // コールスタックのサイズを増加
    size_t stackSize = 64 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    // 重なりの係数Wの計算
    compute_W(d_W_matrix, d_coefficient_matrix, d_orbital_energies, num_basis, num_electron);

    // 各分子積分の微分を同時に計算
    computeMolucularGradients(d_grad_total, d_grad_N, d_grad_S, d_grad_K, d_grad_V, d_grad_G, d_W_matrix,
                              shell_type_infos, shell_pair_type_infos, d_atoms,
                              d_density_matrix, d_coefficient_matrix, d_orbital_energies, d_primitive_shells,
                              d_boys_grid, d_cgto_normalization_factors, num_atoms, num_basis, num_electron, verbose);

    // CPU側へ結果コピー
    cudaMemcpy(grad_total, d_grad_total, gradients_bytes, cudaMemcpyDeviceToHost);

    // Copy to std::vector for return
    std::vector<double> gradient(grad_total, grad_total + n);

    // GPUメモリの解放
    cudaFree(d_W_matrix);
    cudaFree(d_grad_N);
    cudaFree(d_grad_S);
    cudaFree(d_grad_K);
    cudaFree(d_grad_V);
    cudaFree(d_grad_G);
    cudaFree(d_grad_total);

    // CPUメモリの解放
    cudaFreeHost(grad_total);

    return gradient;
}



// UHF版: 各分子積分の微分を同時に計算
void computeMolucularGradients_UHF(double* d_grad_total, double* d_grad_N, double* d_grad_S, double* d_grad_K, double* d_grad_V, double* d_grad_G,
                                    real_t* d_W_total, const real_t* d_D_total,
                                    const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms,
                                    const real_t* d_density_matrix_a, const real_t* d_density_matrix_b,
                                    const PrimitiveShell* d_primitive_shells,
                                    const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,
                                    const int num_atoms, const int num_basis, const bool verbose)
{
    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();

    // 2電子部分の微分の前処理
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    // multi streamの作成
    int stream_id = 0;
    const int num_kernels = shell_quadruples.size() + 3*((shell_type_count)*(shell_type_count+1)/2) + 1;
    std::vector<cudaStream_t> streams(num_kernels);
    for(int i=0; i<num_kernels; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to create CUDA stream: ") + std::string(cudaGetErrorString(err)));
        }
    }

    // 2電子部分の微分 (UHF版: alpha/beta密度行列を別々に渡す)
    for(const auto& quadruple: shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = shell_type_infos[s2];
        const ShellTypeInfo shell_s3 = shell_type_infos[s3];

        const size_t num_bra = (s0==s1) ? shell_s0.count*(shell_s0.count+1)/2 : shell_s0.count*shell_s1.count;
        const size_t num_ket = (s2==s3) ? shell_s2.count*(shell_s2.count+1)/2 : shell_s2.count*shell_s3.count;
        const size_t num_braket = ((s0==s2) && (s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket;
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block;

        get_compute_gradients_repulsion_uhf()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
            d_grad_G, d_density_matrix_a, d_density_matrix_b,
            d_primitive_shells, d_cgto_normalization_factors,
            shell_s0, shell_s1, shell_s2, shell_s3, num_braket, num_basis, d_boys_grid);
    }

    // 1電子部分の微分 (D_total, W_totalを使用 — RHFカーネルを再利用)
    for (int s0 = shell_type_count-1; s0 >= 0; s0--) {
        for (int s1 = shell_type_count-1; s1 >= s0; s1--) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];

            const int num_shell_pairs = (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count);
            const int num_blocks = (num_shell_pairs + threads_per_block - 1) / threads_per_block;

            get_compute_gradients_overlap()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_S, d_W_total, d_primitive_shells, d_cgto_normalization_factors, num_basis, shell_s0, shell_s1, num_shell_pairs);
            get_compute_gradients_kinetic()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_K, d_D_total, d_primitive_shells, d_cgto_normalization_factors, num_basis, shell_s0, shell_s1, num_shell_pairs);
            get_compute_gradients_nuclear()<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_grad_V, d_D_total, d_primitive_shells, d_cgto_normalization_factors, d_atoms, num_atoms, num_basis, shell_s0, shell_s1, num_shell_pairs, d_boys_grid);
        }
    }

    const int NR_blocks = (num_atoms * num_atoms + threads_per_block - 1) / threads_per_block;
    compute_nuclear_repulsion_gradient_kernel<<<NR_blocks, threads_per_block, 0, streams[stream_id]>>>(d_grad_N, d_atoms, num_atoms);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        abort();
    }

    for(int i=0; i<num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }

    // 微分の影響を合計
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0;

    cudaMemcpy(d_grad_total, d_grad_N, sizeof(double) * 3*num_atoms, cudaMemcpyDeviceToDevice);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_S, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_K, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_V, 1, d_grad_total, 1);
    cublasDaxpy(handle, 3*num_atoms, &alpha, d_grad_G, 1, d_grad_total, 1);

    cublasDestroy(handle);
}



// UHF版エネルギー微分を計算する関数
std::vector<double> computeEnergyGradient_UHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
                                const Atom* d_atoms,
                                const real_t* d_density_matrix_a, const real_t* d_density_matrix_b,
                                const real_t* d_coefficient_matrix_a, const real_t* d_coefficient_matrix_b,
                                const real_t* d_orbital_energies_a, const real_t* d_orbital_energies_b,
                                const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,
                                const int num_atoms, const int num_basis, const int num_alpha, const int num_beta, const bool verbose)
{
    const int n = 3*num_atoms;
    const size_t mat_bytes = num_basis * num_basis * sizeof(real_t);
    const size_t gradients_bytes = n * sizeof(double);

    // CPU側のメモリ確保
    double* grad_total = nullptr;
    cudaMallocHost((void**)&grad_total, gradients_bytes);

    // GPU側のメモリ確保
    real_t* d_W_a = nullptr;
    real_t* d_W_b = nullptr;
    real_t* d_W_total = nullptr;
    real_t* d_D_total = nullptr;
    double* d_grad_N = nullptr;
    double* d_grad_S = nullptr;
    double* d_grad_K = nullptr;
    double* d_grad_V = nullptr;
    double* d_grad_G = nullptr;
    double* d_grad_total = nullptr;

    cudaMalloc(&d_W_a, mat_bytes);
    cudaMalloc(&d_W_b, mat_bytes);
    cudaMalloc(&d_W_total, mat_bytes);
    cudaMalloc(&d_D_total, mat_bytes);
    cudaMalloc(&d_grad_N, gradients_bytes);
    cudaMalloc(&d_grad_S, gradients_bytes);
    cudaMalloc(&d_grad_K, gradients_bytes);
    cudaMalloc(&d_grad_V, gradients_bytes);
    cudaMalloc(&d_grad_G, gradients_bytes);
    cudaMalloc(&d_grad_total, gradients_bytes);

    cudaMemset(d_W_a, 0, mat_bytes);
    cudaMemset(d_W_b, 0, mat_bytes);
    cudaMemset(d_W_total, 0, mat_bytes);
    cudaMemset(d_grad_N, 0, gradients_bytes);
    cudaMemset(d_grad_S, 0, gradients_bytes);
    cudaMemset(d_grad_K, 0, gradients_bytes);
    cudaMemset(d_grad_V, 0, gradients_bytes);
    cudaMemset(d_grad_G, 0, gradients_bytes);
    cudaMemset(d_grad_total, 0, gradients_bytes);

    // コールスタックのサイズを増加
    size_t stackSize = 64 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    // D_total = Da + Db
    cudaMemcpy(d_D_total, d_density_matrix_a, mat_bytes, cudaMemcpyDeviceToDevice);
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double one = 1.0;
        cublasDaxpy(handle, num_basis * num_basis, &one, d_density_matrix_b, 1, d_D_total, 1);
        cublasDestroy(handle);
    }

    // W_a = 2 * Σ_k Ca_ik * Ca_jk * εa_k (既存カーネルは num_electron/2 でループ、結果に2.0を掛ける)
    // W_b = 2 * Σ_k Cb_ik * Cb_jk * εb_k
    // UHFでは W_total = Σ_k Ca_ik*Ca_jk*εa_k + Σ_k Cb_ik*Cb_jk*εb_k = (W_a + W_b) / 2
    compute_W(d_W_a, d_coefficient_matrix_a, d_orbital_energies_a, num_basis, 2 * num_alpha);
    compute_W(d_W_b, d_coefficient_matrix_b, d_orbital_energies_b, num_basis, 2 * num_beta);
    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double half = 0.5;
        // W_total = 0.5 * W_a
        cudaMemcpy(d_W_total, d_W_a, mat_bytes, cudaMemcpyDeviceToDevice);
        cublasDscal(handle, num_basis * num_basis, &half, d_W_total, 1);
        // W_total += 0.5 * W_b
        cublasDaxpy(handle, num_basis * num_basis, &half, d_W_b, 1, d_W_total, 1);
        cublasDestroy(handle);
    }

    // 各分子積分の微分を同時に計算
    computeMolucularGradients_UHF(d_grad_total, d_grad_N, d_grad_S, d_grad_K, d_grad_V, d_grad_G,
                                   d_W_total, d_D_total,
                                   shell_type_infos, shell_pair_type_infos, d_atoms,
                                   d_density_matrix_a, d_density_matrix_b,
                                   d_primitive_shells, d_boys_grid, d_cgto_normalization_factors,
                                   num_atoms, num_basis, verbose);

    // CPU側へ結果コピー
    cudaMemcpy(grad_total, d_grad_total, gradients_bytes, cudaMemcpyDeviceToHost);
    std::vector<double> gradient(grad_total, grad_total + n);

    // GPUメモリの解放
    cudaFree(d_W_a);
    cudaFree(d_W_b);
    cudaFree(d_W_total);
    cudaFree(d_D_total);
    cudaFree(d_grad_N);
    cudaFree(d_grad_S);
    cudaFree(d_grad_K);
    cudaFree(d_grad_V);
    cudaFree(d_grad_G);
    cudaFree(d_grad_total);
    cudaFreeHost(grad_total);

    return gradient;
}





void compute_RI_Direct_Z_matrix(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_nomalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
    real_t* d_Z, 
    const real_t* d_C,
    const real_t* d_L_inv,
    const size_t2* d_primitive_shell_pair_indices,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* d_boys_grid,
    const double schwarz_screening_threshold, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    int iter,
    const bool verbose)
{
const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();
    const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();


    // Call the kernel functions from (ss|s),... (e.g. (ss|s), (ss|p), (sp|s), (sp|p), (pp|s), (pp|p) for s and p shells)

    // list shell-triples for sorted shell-type (s0, s1, s2)
    std::vector<std::tuple<int, int, int>> shell_triples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < auxiliary_shell_type_count; ++c) {
                shell_triples.emplace_back(a, b, c);
            }
        }
    }
    // sort by sum (a + b + c) in descending order
    std::sort(shell_triples.begin(), shell_triples.end(),
        [](const auto& lhs, const auto& rhs) {
            int sum_lhs = std::get<0>(lhs) + std::get<1>(lhs) + std::get<2>(lhs);
            int sum_rhs = std::get<0>(rhs) + std::get<1>(rhs) + std::get<2>(rhs);
            return sum_lhs > sum_rhs;  // 降順
        });


    // make multi stream
    const int num_kernels = shell_triples.size();
    std::vector<cudaStream_t> streams(num_kernels);

    // for-loop for sorted shell-type (s0, s1, s2, s3)
    int stream_id = 0;
    for(const auto& triple: shell_triples) {
        int s0, s1, s2;
        std::tie(s0, s1, s2) = triple;

        const ShellTypeInfo shell_s0 = shell_type_infos[s0];
        const ShellTypeInfo shell_s1 = shell_type_infos[s1];
        const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

        const int num_tasks = ( (s0==s1) ? (shell_s0.count*(shell_s0.count+1)/2) : (shell_s0.count*shell_s1.count) ) * shell_s2.count; // the number of pairs of primitive shells = the number of threads
        const int num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block; // the number of blocks
        
        compute_RI_Direct_Z_kernel<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_Z, d_C, d_L_inv, d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                                d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors, 
                                                                                                shell_s0, shell_s1, shell_s2, 
                                                                                                num_tasks, num_basis, 
                                                                                                &d_primitive_shell_pair_indices[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                                &d_schwarz_upper_bound_factors[shell_pair_type_infos[calcIdx_triangular_(s0, s1, shell_type_count)].start_index],
                                                                                                d_auxiliary_schwarz_upper_bound_factors,
                                                                                                schwarz_screening_threshold,
                                                                                                num_auxiliary_basis,
                                                                                                iter,
                                                                                                d_boys_grid);
    }

    // syncronize streams
    cudaDeviceSynchronize();

    // destory streams
    for (int i = 0; i < num_kernels; i++) {
        cudaStreamDestroy(streams[i]);
    }
}


__global__ void addMatrix(real_t* d_K, real_t* d_K_iter, int num_elements) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_elements) return;

    d_K[id] += 2.0*d_K_iter[id];
}


void computeInitialFockMatrix_RI_Direct_RHF(const real_t* d_density_matrix, const real_t* d_C,
                                    const real_t* d_L_inv, 
                                    const real_t* d_core_hamiltonian_matrix, 
                                    real_t* d_fock_matrix, 
                                    const std::vector<ShellTypeInfo>& shell_type_infos, 
                                    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, 
                                    const PrimitiveShell* h_primitive_shells, 
                                    const PrimitiveShell* d_primitive_shells, 
                                    const real_t* d_cgto_normalization_factors, 
                                    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
                                    const PrimitiveShell* d_auxiliary_primitive_shells, 
                                    const real_t* d_auxiliary_cgto_normalization_factors, 
                                    const size_t2* d_primitive_shell_pair_indices,
                                    size_t2* h_primitive_shell_pair_indices_for_SAD_K_computation,
                                    const size_t2* d_primitive_shell_pair_indices_for_SAD_K_computation,
                                    const int num_basis,
                                    const int num_auxiliary_basis,
                                    const int num_electrons,
                                    const int num_primitive_shells,
                                    const real_t* d_boys_grid,
                                    const double schwarz_screening_threshold, 
                                    const real_t* d_schwarz_upper_bound_factors,
                                    const real_t* d_auxiliary_schwarz_upper_bound_factors,
                                    const bool verbose,
                                    real_t* d_decomposed_two_center_eris){
    //cublasManager cublas;
    cublasHandle_t cublasHandle = GPUHandle::cublas();



    cudaError_t err;

    // the following is used in the two kernels. So, if necessary, it should be changed for each kernel.
    const int num_threads = 256;
    const int num_blocks = (num_basis * num_basis + num_threads - 1) / num_threads;

    ////////////////////////////////// compute J-matrix //////////////////////////////////
    real_t* d_J = nullptr;
    err = cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_J, 0.0, sizeof(real_t)*num_basis*num_basis);


    // compute c_q = \sum_{a b} D_{a b} (q|ab)
    real_t *d_c = nullptr;
    err = cudaMalloc(&d_c, sizeof(real_t)*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for c vector: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_c, 0.0, sizeof(real_t)*num_auxiliary_basis);


    // cublas関数ように、column-majorにしておく
    transposeMatrixInPlace(d_decomposed_two_center_eris, num_auxiliary_basis);


    // cを求める
    compute_RI_Direct_c_array(shell_type_infos,
                              shell_pair_type_infos,
                              d_primitive_shells,
                              d_cgto_normalization_factors,
                              auxiliary_shell_type_infos,
                              d_auxiliary_primitive_shells,
                              d_auxiliary_cgto_normalization_factors,
                              d_c,
                              d_density_matrix,
                              d_primitive_shell_pair_indices,
                              num_basis,
                              num_auxiliary_basis,
                              d_boys_grid,
                              schwarz_screening_threshold,
                              d_schwarz_upper_bound_factors,
                              d_auxiliary_schwarz_upper_bound_factors,
                              verbose);
    cudaDeviceSynchronize();





    // Ly=cをyについて解く   
    cublasDtrsv(
        cublasHandle,
        CUBLAS_FILL_MODE_LOWER, 
        CUBLAS_OP_N,            
        CUBLAS_DIAG_NON_UNIT,   
        num_auxiliary_basis,                   
        d_decomposed_two_center_eris, num_auxiliary_basis,                  
        d_c, 1               
    );

    // L^T t = y をtについて解く
    cublasDtrsv(
        cublasHandle,       
        CUBLAS_FILL_MODE_LOWER, 
        CUBLAS_OP_T,            
        CUBLAS_DIAG_NON_UNIT,   
        num_auxiliary_basis,                                     
        d_decomposed_two_center_eris, num_auxiliary_basis,                  
        d_c, 1                
    );



    // Jmu nu = ()
    compute_RI_Direct_J_matrix(shell_type_infos,
                              shell_pair_type_infos,
                              d_primitive_shells,
                              d_cgto_normalization_factors,
                              auxiliary_shell_type_infos,
                              d_auxiliary_primitive_shells,
                              d_auxiliary_cgto_normalization_factors,
                              d_J,
                              d_c,
                              d_primitive_shell_pair_indices,
                              num_basis,
                              num_auxiliary_basis,
                              d_boys_grid,
                              schwarz_screening_threshold,
                              d_schwarz_upper_bound_factors,
                              d_auxiliary_schwarz_upper_bound_factors,
                              verbose);
    cudaDeviceSynchronize();

    cudaFree(d_c);


    transposeMatrixInPlace(d_decomposed_two_center_eris, num_auxiliary_basis);



    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_K = nullptr;
    err = cudaMalloc(&d_K, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for K matrix: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_K, 0.0, sizeof(real_t)*num_basis*num_basis);

    real_t* d_K_iter = nullptr;
    err = cudaMalloc(&d_K_iter, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for K matrix: ") + std::string(cudaGetErrorString(err)));
    }


    real_t* d_Z = nullptr;
    err = cudaMalloc(&d_Z, sizeof(real_t)*num_basis*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Z matrix: ") + std::string(cudaGetErrorString(err)));
    }

    const double alpha = 1.0, beta = 0.0;

    const int row = num_basis, col = num_auxiliary_basis;

    for(int iter = 0; iter < num_electrons/2; iter++) {
        // printf("Iter [%d / %d]-----------------\n",iter, num_electrons/2);
        cudaMemset(d_Z, 0.0, sizeof(real_t)*num_basis*num_auxiliary_basis);

        // printf("    3-center\n");
        compute_RI_Direct_Z_matrix(shell_type_infos,
                        shell_pair_type_infos,
                        d_primitive_shells,
                        d_cgto_normalization_factors,
                        auxiliary_shell_type_infos,
                        d_auxiliary_primitive_shells,
                        d_auxiliary_cgto_normalization_factors,
                        d_Z,
                        d_C,
                        d_L_inv,
                        d_primitive_shell_pair_indices,
                        num_basis,
                        num_auxiliary_basis,
                        d_boys_grid,
                        schwarz_screening_threshold,
                        d_schwarz_upper_bound_factors,
                        d_auxiliary_schwarz_upper_bound_factors,
                        iter,
                        verbose);

        // cudaDeviceSynchronize();
    
        // printf("    DGEMM\n");
        cublasDgemm(
            cublasHandle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            row,
            row,
            col,
            &alpha,
            d_Z, col,
            d_Z, col,
            &beta,
            d_K_iter, row
        );

        // printf("    Add\n");
        addMatrix<<< num_blocks, num_threads >>>(d_K, d_K_iter, num_basis*num_basis);
        // addMatrix<<< num_blocks, num_threads >>>(d_K, d_Z, num_basis, num_auxiliary_basis);
        // cudaDeviceSynchronize();
    }



    // ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // // F = H + J - (1/2)*K
    computeFockMatrix_RI_RHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, num_basis);
    // cudaMemcpy(d_fock_matrix, d_J, sizeof(real_t)*num_basis*num_basis, cudaMemcpyDeviceToDevice);

    // free the memory
    cudaFree(d_J);
    cudaFree(d_K);
    cudaFree(d_K_iter);
    cudaFree(d_Z);
}







void computeSchwarzUpperBounds_for_SAD_K_computation(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_normalization_factors, 
    real_t* d_upper_bound_factors, 
    ShellPairSorter* d_upper_bound_factors_for_SAD_K_computation, 
    const size_t num_primitive_shells, 
    const bool verbose)
{
    const int threads_per_block = 256; // the number of threads per block
    const int shell_type_count = shell_type_infos.size();

    for (int s0 = 0; s0 < shell_type_count; ++s0) {
        for (int s1 = s0; s1 < shell_type_count; ++s1) {
            const ShellTypeInfo shell_s0 = shell_type_infos[s0];
            const ShellTypeInfo shell_s1 = shell_type_infos[s1];
            const size_t head = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].start_index;
            const size_t num_bra = shell_pair_type_infos[get_index_2to1_horizontal(s0, s1, shell_type_count)].count;
            const size_t num_blocks = (num_bra + threads_per_block - 1) / threads_per_block; // the number of blocks

            gpu::get_schwarz_upper_bound_factors_general_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, head, num_bra, num_primitive_shells, d_boys_grid, d_upper_bound_factors, d_upper_bound_factors_for_SAD_K_computation);
        }
    }
}

// Skeleton Hessian (without CPHF response term)
// H_skel = d²V_nn/dRdR + D·(d²T/dRdR + d²V/dRdR) - W·d²S/dRdR + d²G/dRdR
std::vector<double> computeSkeletonHessian_RHF(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const Atom* d_atoms,
    const real_t* d_density_matrix,
    const real_t* d_coefficient_matrix,
    const real_t* d_orbital_energies,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_boys_grid,
    const real_t* d_cgto_normalization_factors,
    const int num_atoms, const int num_basis, const int num_electron, const bool verbose)
{
    const int ndim = 3 * num_atoms;
    const size_t hess_bytes = ndim * ndim * sizeof(double);
    const int threads_per_block = 128;
    const int shell_type_count = shell_type_infos.size();

    // Allocate Hessian on GPU
    double* d_hessian = nullptr;
    cudaMalloc(&d_hessian, hess_bytes);
    cudaMemset(d_hessian, 0, hess_bytes);

    // W matrix for overlap Hessian
    const size_t wmat_bytes = num_basis * num_basis * sizeof(real_t);
    real_t* d_W_matrix = nullptr;
    cudaMalloc(&d_W_matrix, wmat_bytes);
    cudaMemset(d_W_matrix, 0, wmat_bytes);
    compute_W(d_W_matrix, d_coefficient_matrix, d_orbital_energies, num_basis, num_electron);

    // Stack size for complex kernels
    size_t stackSize = 64 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    // --- 2-electron Hessian ---
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a)
        for (int b = a; b < shell_type_count; ++b)
            for (int c = 0; c < shell_type_count; ++c)
                for (int d = c; d < shell_type_count; ++d)
                    if (a < c || (a == c && b <= d))
                        shell_quadruples.emplace_back(a, b, c, d);
    std::reverse(shell_quadruples.begin(), shell_quadruples.end());

    int stream_id = 0;
    const int num_kernels = shell_quadruples.size() + 3*((shell_type_count)*(shell_type_count+1)/2) + 1;
    std::vector<cudaStream_t> streams(num_kernels);
    for (int i = 0; i < num_kernels; i++)
        cudaStreamCreate(&streams[i]);

    for (const auto& quadruple : shell_quadruples) {
        int s0, s1, s2, s3;
        std::tie(s0, s1, s2, s3) = quadruple;
        const ShellTypeInfo ss0 = shell_type_infos[s0], ss1 = shell_type_infos[s1];
        const ShellTypeInfo ss2 = shell_type_infos[s2], ss3 = shell_type_infos[s3];
        const size_t num_bra = (s0==s1) ? ss0.count*(ss0.count+1)/2 : ss0.count*ss1.count;
        const size_t num_ket = (s2==s3) ? ss2.count*(ss2.count+1)/2 : ss2.count*ss3.count;
        const size_t num_braket = ((s0==s2)&&(s1==s3)) ? num_bra*(num_bra+1)/2 : num_bra*num_ket;
        const int num_blocks = (num_braket + threads_per_block - 1) / threads_per_block;

        Rys_compute_hessian_two_electron<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
            d_hessian, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors,
            ss0, ss1, ss2, ss3, num_braket, num_basis, num_atoms, d_boys_grid);
    }

    // --- 1-electron Hessian ---
    for (int s0 = shell_type_count-1; s0 >= 0; s0--) {
        for (int s1 = shell_type_count-1; s1 >= s0; s1--) {
            const ShellTypeInfo ss0 = shell_type_infos[s0], ss1 = shell_type_infos[s1];
            const int num_pairs = (s0==s1) ? (ss0.count*(ss0.count+1)/2) : (ss0.count*ss1.count);
            const int num_blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

            compute_hessian_overlap<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
                d_hessian, d_W_matrix, d_primitive_shells, d_cgto_normalization_factors,
                num_basis, num_atoms, ss0, ss1, num_pairs);
            compute_hessian_kinetic<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
                d_hessian, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors,
                num_basis, num_atoms, ss0, ss1, num_pairs);
            compute_hessian_nuclear_attraction<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(
                d_hessian, d_density_matrix, d_primitive_shells, d_cgto_normalization_factors,
                d_atoms, num_basis, num_atoms, ss0, ss1, num_pairs, d_boys_grid);
        }
    }

    // --- Nuclear repulsion Hessian ---
    const int num_pairs = num_atoms * (num_atoms + 1) / 2;
    const int NR_blocks = (num_pairs + threads_per_block - 1) / threads_per_block;
    compute_hessian_nuclear_repulsion<<<NR_blocks, threads_per_block, 0, streams[stream_id]>>>(
        d_hessian, d_atoms, num_atoms);

    cudaDeviceSynchronize();

    for (int i = 0; i < num_kernels; i++)
        cudaStreamDestroy(streams[i]);

    // Copy to host
    std::vector<double> hessian(ndim * ndim);
    cudaMemcpy(hessian.data(), d_hessian, hess_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_hessian);
    cudaFree(d_W_matrix);

    return hessian;
}

} // namespace gansu::gpu
