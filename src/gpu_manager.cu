/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
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

#include <vector>    // std::vector
#include <tuple>     // std::tuple
#include <algorithm> // std::reverse
#include <fstream>

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
    err = cudaMalloc(&d_workspace, workspaceInBytesOnDevice);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMallocHost(&h_workspace, workspaceInBytesOnHost);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate host memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // allocate return vale for the error status        
    int* d_info;
    err = cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for error status: ") + std::string(cudaGetErrorString(err)));
    }

    // temporary matrix allocation for d_matrix since the eigenvectors will be stored in the same memory of d_matrix
    real_t* d_temp_matrix;
    err = cudaMalloc(&d_temp_matrix, size * size * sizeof(real_t));
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
    cudaFree(d_temp_matrix);
    cudaFree(d_workspace);
    cudaFree(h_workspace);
    cudaFree(d_info);

    return h_info; // 0 if successful
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
    const double beta = 0.0;

    if (!accumulate){
        cudaMemset(d_matrix_C, 0, size * size * sizeof(double));
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
 real_t computeMatrixTrace(const real_t* d_matrix, const int size) {
    if(size > 1024){ // 1024 is the maximum number of threads per block. If the size is larger than 1024, two or more blocks are required.
        THROW_EXCEPTION("Too many basis functions.");
    }

    cudaError_t err;

    double* d_trace;
    err = cudaMalloc(&d_trace, sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for trace: ") + std::string(cudaGetErrorString(err)));
    }

    real_t h_trace = 0.0;
    
    getMatrixTrace<<<1, size>>>(d_matrix, d_trace, size);
    cudaMemcpy(&h_trace, d_trace, sizeof(real_t), cudaMemcpyDeviceToHost);

    cudaFree(d_trace);
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
void computeCoreHamiltonianMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, Atom* d_atoms, PrimitiveShell* d_primitive_shells, real_t* d_boys_grid, real_t* d_cgto_normalization_factors, real_t* d_overlap_matrix, real_t* d_core_hamiltonian_matrix, const int num_atoms, const int num_basis, const bool verbose) {
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
            // printf("(s0,s1) = (%d, %d), idx = %d\n", s0, s1, index);


            // call the kernel functions
            get_overlap_kinetic_kernel(s0, s1)<<<num_blocks, threads_per_block, 0, streams[index]>>>(d_overlap_matrix, d_core_hamiltonian_matrix, d_primitive_shells, d_cgto_normalization_factors, shell_s0, shell_s1, num_shell_pairs, num_basis);
            get_nuclear_attraction_kernel(s0, s1)<<<num_blocks, threads_per_block, 0, V_streams[index]>>>(d_core_hamiltonian_matrix, d_primitive_shells, d_cgto_normalization_factors, d_atoms, num_atoms, shell_s0, shell_s1, num_shell_pairs, num_basis, d_boys_grid);
        }
    }
    // syncronize streams
    cudaDeviceSynchronize();

    dim3 blocks(int((num_basis + 31) / 32), int((num_basis + 31) / 32));
    dim3 threads(32,32);
    Matrix_Symmetrization<<<blocks, threads>>>(d_overlap_matrix, num_basis);
    Matrix_Symmetrization<<<blocks, threads>>>(d_core_hamiltonian_matrix, num_basis);

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

    err = cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempSymFockMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary symmetrized Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempEigenvectors, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary eigenvectors: ") + std::string(cudaGetErrorString(err)));
    }

    if (d_orbital_energies == nullptr){
        err = cudaMalloc(&d_tempEigenvalues, num_basis * sizeof(real_t));
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
    cudaFree(d_tempMatrix);
    cudaFree(d_tempSymFockMatrix);
    cudaFree(d_tempEigenvectors);

    if (d_orbital_energies == nullptr){
        cudaFree(d_tempEigenvalues);
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

    err = cudaMalloc(&d_temp_F_MO_closed, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary Fock matrix for closed-shell orbitals: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_F_MO_open, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary Fock matrix for open-shell orbitals: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_R_MO, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary unified Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_matrix1, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 1: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_matrix2, num_basis * num_basis * sizeof(real_t));
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
    cudaFree(d_temp_F_MO_closed);
    cudaFree(d_temp_F_MO_open);
    cudaFree(d_temp_R_MO);
    cudaFree(d_temp_matrix1);
    cudaFree(d_temp_matrix2);

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

    err = cudaMalloc(&d_tempDiffFockMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary difference Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempDiffDensityMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary difference density matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
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
    cudaFree(d_tempDiffFockMatrix);
    cudaFree(d_tempDiffDensityMatrix);
    cudaFree(d_tempMatrix);


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

    err = cudaMalloc(&d_tempMatrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix: ") + std::string(cudaGetErrorString(err)));
    }

    weightedMatrixSum(d_matrix_old, d_matrix_new, d_tempMatrix, 1.0-alpha, alpha, num_basis);

    cudaMemcpy(d_matrix_old, d_tempMatrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_matrix_new, d_tempMatrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToDevice);

    cudaFree(d_tempMatrix);
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

    err = cudaMalloc(&d_tempFPS, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary FPS matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempSPF, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary SPF matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_tempMatrix1, num_basis * num_basis * sizeof(real_t));
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

    cudaFree(d_tempMatrix1);
    cudaFree(d_tempFPS);
    cudaFree(d_tempSPF);

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

    err = cudaMalloc(&d_DIIS_matrix, num_size * num_size * sizeof(real_t));
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
    err = cudaMalloc(&d_DIIS_rhs, num_size * sizeof(real_t));
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
    err = cudaMalloc(&d_work, work_size * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // pivot array and info
    int* d_pivot = nullptr;
    int* d_info = nullptr;
    err = cudaMalloc(&d_pivot, num_size * sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for pivot array: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_info, sizeof(int));
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
    cudaFree(d_DIIS_matrix);
    cudaFree(d_DIIS_rhs);
    cudaFree(d_work);
    cudaFree(d_pivot);
    cudaFree(d_info);

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
    err = cudaMalloc(&d_temp_FockMatrix, num_basis * num_basis * sizeof(real_t));
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
    cudaFree(d_temp_FockMatrix);

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
    
    err = cudaMalloc(&d_ipiv, N * sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for pivot array: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for info array: ") + std::string(cudaGetErrorString(err)));
    }

    // Get workspace size for LU decomposition
    cusolverDnDgetrf_bufferSize(cusolverHandle, N, N, d_A, N, &lwork);
    err = cudaMalloc(&d_work, lwork * sizeof(double));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for workspace: ") + std::string(cudaGetErrorString(err)));
    }

    // Perform LU decomposition
    cusolverDnDgetrf(cusolverHandle, N, N, d_A, N, d_work, d_ipiv, d_info);

    // Allocate and initialize an identity matrix on the device
    double *d_I;
    err = cudaMalloc(&d_I, N * N * sizeof(double));
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
    cudaFree(d_ipiv);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaFree(d_I);
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
    err = cudaMalloc(&d_info, sizeof(int));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for info array: ") + std::string(cudaGetErrorString(err)));
    }

    // Get workspace size
    cusolverDnDpotrf_bufferSize(cusolverHandle, CUBLAS_FILL_MODE_UPPER, N, d_A, N, &lwork);
    err = cudaMalloc(&d_work, lwork * sizeof(double));
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
    cudaFree(d_work);
    cudaFree(d_info);
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
    err = cudaMalloc((void**)&d_tmp, sizeof(double) * row * col);
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

    cudaFree(d_tmp);
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
 * @param d_auxiliary_cgto_nomalization_factors Pointer to the normalization factors of the auxiliary CGTOs in device memory
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
    const real_t* d_auxiliary_cgto_nomalization_factors, 
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
    err = cudaMalloc(&d_two_center_eri, num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for two-center ERIs: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_two_center_eri, 0.0, num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t));

    // Compute the two-center ERIs of the auxiliary basis functions
    computeTwoCenterERIs(
        auxiliary_shell_type_infos, 
        d_auxiliary_primitive_shells, 
        d_auxiliary_cgto_nomalization_factors, 
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
    err = cudaMalloc(&d_three_center_eri, num_basis * num_basis * num_auxiliary_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for three-center ERIs: ") + std::string(cudaGetErrorString(err)));
    }
    cudaMemset(d_three_center_eri, 0.0, num_basis * num_basis * num_auxiliary_basis * sizeof(real_t));

    // Compute the three-center ERIs of the auxiliary basis functions and the basis functions
    computeThreeCenterERIs(
        shell_type_infos, 
        shell_pair_type_infos,
        d_primitive_shells, 
        d_cgto_normalization_factors, 
        auxiliary_shell_type_infos, 
        d_auxiliary_primitive_shells, 
        d_auxiliary_cgto_nomalization_factors, 
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


    cudaFree(d_two_center_eri);
    cudaFree(d_three_center_eri);

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




void computeFockMatrix_RI_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix, const int num_basis, const int num_auxiliary_basis){
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


    // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
    real_t* d_W = nullptr;
    err = cudaMalloc(&d_W, sizeof(real_t)*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for W vector: ") + std::string(cudaGetErrorString(err)));
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

    // J = sum(W[i] * B[i])
    weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);


    // free the memory
    cudaFree(d_W);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_K = nullptr;
    err = cudaMalloc(&d_K, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for K matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // T^p = B^p D^T
    real_t* d_T = nullptr;
    err = cudaMalloc(&d_T, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for T matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // Note: cublasDgemmBatched shoul be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], d_density_matrix, &d_T[p*num_basis*num_basis], num_basis, false, true);
    }


    // V^p = B^p (T^p)^T
    real_t* d_V = nullptr;
    err = cudaMalloc(&d_V, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for V matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // Note: cublasDgemmBatched shoul be used?
    for(int p=0; p<num_auxiliary_basis; p++){
        matrixMatrixProduct(&d_intermediate_matrix_B[p*num_basis*num_basis], &d_T[p*num_basis*num_basis], &d_V[p*num_basis*num_basis], num_basis, false, true);
    }

    // K = sum(V^p)
    sum_matrices_kernel<<<num_blocks, num_threads>>>(d_K, d_V, num_basis, num_auxiliary_basis); 

    // free the memory
    cudaFree(d_T);
    cudaFree(d_V);


    ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // F = H + J - (1/2)*K
    computeFockMatrix_RI_RHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_K, d_fock_matrix, num_basis);

    // free the memory
    cudaFree(d_J);
    cudaFree(d_K);
}

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
    err = cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_density_matrix, sizeof(real_t)*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for density matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // D = D_a + D_b
    matrixAddition(d_density_matrix_a, d_density_matrix_b, d_density_matrix, num_basis);

    // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
    real_t* d_W = nullptr;
    err = cudaMalloc(&d_W, sizeof(real_t)*num_auxiliary_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for W vector: ") + std::string(cudaGetErrorString(err)));
    }

    double alpha = 1.0;
    double beta = 0.0;

    cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

    // J = sum(W[i] * B[i])
    weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);


    // free the memory
    cudaFree(d_W);
    cudaFree(d_density_matrix);

    ////////////////////////////////// compute K-matrix //////////////////////////////////
    real_t* d_T = nullptr;
    real_t* d_V = nullptr;
    err = cudaMalloc(&d_T, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for T matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_V, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for V matrix: ") + std::string(cudaGetErrorString(err)));
    }

    ////////////// compute Ka-matrix //////////////
    real_t* d_Ka = nullptr;
    err = cudaMalloc(&d_Ka, sizeof(real_t)*num_basis*num_basis);
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
    err = cudaMalloc(&d_Kb, sizeof(real_t)*num_basis*num_basis);
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
    cudaFree(d_T);
    cudaFree(d_V);

    ////////////////////////////////// compute Fock matrix //////////////////////////////////

    // F_a = H + J - K_a
    // F_b = H + J - K_b
    computeFockMatrix_RI_UHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Ka, d_fock_matrix_a, num_basis);
    computeFockMatrix_RI_UHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Kb, d_fock_matrix_b, num_basis);


    // free the memory
    cudaFree(d_J);
    cudaFree(d_Ka);
    cudaFree(d_Kb);
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
    err = cudaMalloc(&d_temp_F_MO_closed, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for closed-shell Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_F_MO_open, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for open-shell Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_R_MO, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for unified Fock matrix: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_matrix1, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for temporary matrix 1: ") + std::string(cudaGetErrorString(err)));
    }
    err = cudaMalloc(&d_temp_matrix2, num_basis * num_basis * sizeof(real_t));
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
        err = cudaMalloc(&d_J, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for J matrix: ") + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_density_matrix, sizeof(real_t)*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for density matrix: ") + std::string(cudaGetErrorString(err)));
        }

        // D = D_closed + D_open
        matrixAddition(d_density_matrix_closed, d_density_matrix_open, d_density_matrix, num_basis);

        // W = B D (Matrix(M_aux x M^2 matrix) * Vector (M^2 x 1) )
        real_t* d_W = nullptr;
        err = cudaMalloc(&d_W, sizeof(real_t)*num_auxiliary_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for W vector: ") + std::string(cudaGetErrorString(err)));
        }

        double alpha = 1.0;
        double beta = 0.0;

        cublasDgemv(cublasHandle, CUBLAS_OP_T, num_basis*num_basis, num_auxiliary_basis, &alpha, d_intermediate_matrix_B, num_basis*num_basis, d_density_matrix, 1, &beta, d_W, 1);

        // J = sum(W[i] * B[i])
        weighted_sum_matrices_kernel<<<num_blocks, num_threads>>>(d_J, d_intermediate_matrix_B, d_W, num_basis, num_auxiliary_basis);


        // free the memory
        cudaFree(d_W);

        ////////////////////////////////// compute Kclosed-matrix //////////////////////////////////
        real_t* d_T = nullptr;
        real_t* d_V = nullptr;
        err = cudaMalloc(&d_T, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for T matrix: ") + std::string(cudaGetErrorString(err)));
        }
        err = cudaMalloc(&d_V, sizeof(real_t)*num_auxiliary_basis*num_basis*num_basis);
        if (err != cudaSuccess) {
            THROW_EXCEPTION(std::string("Failed to allocate device memory for V matrix: ") + std::string(cudaGetErrorString(err)));
        }

        ////////////// compute Kclosed-matrix //////////////
        real_t* d_Kclosed = nullptr;
        err = cudaMalloc(&d_Kclosed, sizeof(real_t)*num_basis*num_basis);
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
        err = cudaMalloc(&d_Kopen, sizeof(real_t)*num_basis*num_basis);
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
        cudaFree(d_T);
        cudaFree(d_V);
        


        ////////////////////////////////// compute Fock matrix //////////////////////////////////

        // Fclosed = H + J - 0.5*Kclosed
        // Fopen = 0.5*(H + J - Kopen)
        computeFockMatrix_RI_ROHF_kernel<<<num_blocks, num_threads>>>(d_core_hamiltonian_matrix, d_J, d_Kclosed, d_Kopen, d_fock_matrix_closed, d_fock_matrix_open, num_basis);


        // free the memory
        cudaFree(d_J);
        cudaFree(d_Kclosed);
        cudaFree(d_Kopen);
        cudaFree(d_density_matrix);
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
    cudaFree(d_temp_F_MO_closed);
    cudaFree(d_temp_F_MO_open);
    cudaFree(d_temp_R_MO);
    cudaFree(d_temp_matrix1);
    cudaFree(d_temp_matrix2);

}

void computeTwoCenterERIs(
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
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
                                                                            d_auxiliary_cgto_nomalization_factors, 
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
    const real_t* d_cgto_nomalization_factors, 
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, 
    const PrimitiveShell* d_auxiliary_primitive_shells, 
    const real_t* d_auxiliary_cgto_nomalization_factors, 
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
        
        gpu::get_3center_kernel(s0, s1, s2)<<<num_blocks, threads_per_block, 0, streams[stream_id++]>>>(d_three_center_eri, d_primitive_shells, d_auxiliary_primitive_shells, 
                                                                                d_cgto_nomalization_factors, d_auxiliary_cgto_nomalization_factors, 
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



void computeFockMatrix_Direct_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_cgto_nomalization_factors, 
    const real_t* d_boys_grid, 
    const real_t* d_schwarz_upper_bound_factors,
    const real_t  schwarz_screening_threshold,
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    // ここにDirect-SCF(RHF)のFock行列計算を書く
    THROW_EXCEPTION("Not implemented yet.");
}


void computeMullikenPopulation_RHF(
        const real_t* d_density_matrix,
        const real_t* d_overlap_matrix,
        real_t* mulliken_population_basis,
        const int num_basis
    )
{

    cudaError_t err;

    real_t* d_mulliken_population = nullptr;
    err = cudaMalloc(&d_mulliken_population, num_basis * sizeof(real_t));
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
    cudaFree(d_mulliken_population);
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
    err = cudaMalloc(&d_mulliken_population, num_basis * sizeof(real_t));
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
    cudaFree(d_mulliken_population);
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
    err = cudaMalloc(&d_result_matrix, num_basis * num_basis * sizeof(real_t));
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for Density-Overlap-Matrix: ") + std::string(cudaGetErrorString(err)));
    }

    // result_matrix = D * S
    matrixMatrixProduct(d_density_matrix, d_overlap_matrix, d_result_matrix, num_basis, false, false);

    // Data transfer to host
    cudaMemcpy(result_matrix, d_result_matrix, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Free the memory for the temporary matrix
    cudaFree(d_result_matrix);
}



void constructERIHash(
    const std::vector<ShellTypeInfo>& shell_type_infos, 
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, 
    const real_t* d_boys_grid, 
    const real_t* d_cgto_normalization_factors, 
    // Hash memoryへのポインタ
    const bool verbose)
{
    // ここにERIを計算してハッシュメモリに追加するコードを書く（GPUカーネルを呼ぶ）
    //const int threads_per_block = 256; // the number of threads per block
    // ...
    // GPUカーネルはgpu_kernels.hppにプロトタイプ宣言、gpu_kernels.cuに実装を記述
    //constructERIHash_kernel<<<1, threads_per_block>>>(shell_type_infos, shell_pair_type_infos, d_primitive_shells, d_cgto_normalization_factors, /* Hash memory へのポインタ, */ verbose);

    THROW_EXCEPTION("Not implemented yet.");

}


void computeFockMatrix_Hash_RHF(
    const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    // Hash memoryへのポインタ
    real_t* d_fock_matrix,
    const int num_basis,
    const int verbose)
{
    // ここにERIハッシュを使ってFock行列を計算するコードを書く
    // 
    // computeFockMatrix_Hash_RHF_kernel<<<1, 256>>>(d_density_matrix, d_core_hamiltonian_matrix, /* Hash memory へのポインタ */, d_fock_matrix, num_basis);


    THROW_EXCEPTION("Not implemented yet.");
}


} // namespace gansu::gpu