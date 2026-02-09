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

#include "test_operators.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include <cublas_v2.h>

namespace gansu {

// ========== CUDA Kernels ==========

/**
 * @brief Kernel for diagonal matrix-vector product: y = D * x
 */
__global__ void apply_diagonal_kernel(
    const real_t* d_diagonal,
    const real_t* d_input,
    real_t* d_output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        d_output[idx] = d_diagonal[idx] * d_input[idx];
    }
}

/**
 * @brief Kernel for diagonal preconditioner: y = x / D
 */
__global__ void apply_diagonal_preconditioner_kernel(
    const real_t* d_diagonal,
    const real_t* d_input,
    real_t* d_output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        real_t diag_val = d_diagonal[idx];
        // Avoid division by zero
        d_output[idx] = (fabs(diag_val) > 1e-12) ? d_input[idx] / diag_val : 0.0;
    }
}

// ========== DiagonalOperator Implementation ==========

DiagonalOperator::DiagonalOperator(const std::vector<real_t>& diagonal)
    : dim_(diagonal.size()), d_diagonal_(nullptr)
{
    if (dim_ <= 0) {
        THROW_EXCEPTION("DiagonalOperator: dimension must be positive");
    }

    // Allocate device memory
    tracked_cudaMalloc(&d_diagonal_, dim_ * sizeof(real_t));

    // Copy diagonal to device
    cudaError_t err = cudaMemcpy(d_diagonal_, diagonal.data(),
                                  dim_ * sizeof(real_t),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        tracked_cudaFree(d_diagonal_);
        THROW_EXCEPTION("DiagonalOperator: failed to copy diagonal to device");
    }
}

DiagonalOperator::~DiagonalOperator() {
    if (d_diagonal_) {
        tracked_cudaFree(d_diagonal_);
    }
}

void DiagonalOperator::apply(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("DiagonalOperator::apply: null pointer");
    }

    int num_blocks = (dim_ + 255) / 256;
    int threads_per_block = 256;

    apply_diagonal_kernel<<<num_blocks, threads_per_block>>>(
        d_diagonal_, d_input, d_output, dim_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("DiagonalOperator::apply: kernel launch failed: ") +
                       cudaGetErrorString(err));
    }
}

void DiagonalOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("DiagonalOperator::apply_preconditioner: null pointer");
    }

    int num_blocks = (dim_ + 255) / 256;
    int threads_per_block = 256;

    apply_diagonal_preconditioner_kernel<<<num_blocks, threads_per_block>>>(
        d_diagonal_, d_input, d_output, dim_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("DiagonalOperator::apply_preconditioner: kernel launch failed: ") +
                       cudaGetErrorString(err));
    }
}

// ========== SymmetricMatrixOperator Implementation ==========

SymmetricMatrixOperator::SymmetricMatrixOperator(const std::vector<real_t>& matrix, int n)
    : dim_(n), d_matrix_(nullptr), d_diagonal_(nullptr)
{
    if (dim_ <= 0) {
        THROW_EXCEPTION("SymmetricMatrixOperator: dimension must be positive");
    }

    if (matrix.size() != static_cast<size_t>(dim_ * dim_)) {
        THROW_EXCEPTION("SymmetricMatrixOperator: matrix size mismatch");
    }

    // Allocate device memory for matrix
    tracked_cudaMalloc(&d_matrix_, dim_ * dim_ * sizeof(real_t));

    // Copy matrix to device
    cudaError_t err = cudaMemcpy(d_matrix_, matrix.data(),
                                  dim_ * dim_ * sizeof(real_t),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        tracked_cudaFree(d_matrix_);
        THROW_EXCEPTION("SymmetricMatrixOperator: failed to copy matrix to device");
    }

    // Extract and store diagonal for preconditioner
    std::vector<real_t> diagonal(dim_);
    for (int i = 0; i < dim_; ++i) {
        diagonal[i] = matrix[i * dim_ + i];
    }

    tracked_cudaMalloc(&d_diagonal_, dim_ * sizeof(real_t));
    err = cudaMemcpy(d_diagonal_, diagonal.data(),
                     dim_ * sizeof(real_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        tracked_cudaFree(d_matrix_);
        tracked_cudaFree(d_diagonal_);
        THROW_EXCEPTION("SymmetricMatrixOperator: failed to copy diagonal to device");
    }
}

SymmetricMatrixOperator::~SymmetricMatrixOperator() {
    if (d_matrix_) {
        tracked_cudaFree(d_matrix_);
    }
    if (d_diagonal_) {
        tracked_cudaFree(d_diagonal_);
    }
}

void SymmetricMatrixOperator::apply(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("SymmetricMatrixOperator::apply: null pointer");
    }

    // Use cuBLAS for matrix-vector product: output = matrix * input
    // Dgemv: y = alpha * A * x + beta * y
    // Here: output = 1.0 * matrix * input + 0.0 * output

    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    // cuBLAS uses column-major, but our matrix is row-major
    // So we compute: output = matrix^T * input (which equals matrix * input for symmetric matrix)
    cublasStatus_t status = cublasDgemv(
        handle,
        CUBLAS_OP_T,          // Transpose (to handle row-major as column-major)
        dim_,                 // Number of rows of matrix
        dim_,                 // Number of columns of matrix
        &alpha,               // Scalar alpha
        d_matrix_,            // Matrix A
        dim_,                 // Leading dimension of A
        d_input,              // Vector x
        1,                    // Increment for x
        &beta,                // Scalar beta
        d_output,             // Vector y
        1                     // Increment for y
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        THROW_EXCEPTION("SymmetricMatrixOperator::apply: cublasDgemv failed");
    }
}

void SymmetricMatrixOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!d_input || !d_output) {
        THROW_EXCEPTION("SymmetricMatrixOperator::apply_preconditioner: null pointer");
    }

    int num_blocks = (dim_ + 255) / 256;
    int threads_per_block = 256;

    apply_diagonal_preconditioner_kernel<<<num_blocks, threads_per_block>>>(
        d_diagonal_, d_input, d_output, dim_
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("SymmetricMatrixOperator::apply_preconditioner: kernel launch failed: ") +
                       cudaGetErrorString(err));
    }
}

} // namespace gansu
