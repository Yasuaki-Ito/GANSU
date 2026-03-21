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

/**
 * @file adc2_full_operator.cu
 * @brief GPU implementation of ADC(2) operator in the full singles+doubles space
 *
 * σ = [M11  M12] [R1]
 *     [M21  D2 ] [R2]
 *
 * σ1 = M11·R1 + M12·R2     (singles block)
 * σ2 = M21·R1 + D2·R2      (doubles block, D2 is diagonal)
 *
 * Uses kernels from adc2_operator.cu for M12·x2 and M21·x1.
 */

#include "adc2_full_operator.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"

namespace gansu {

// Forward declarations of kernels defined in adc2_operator.cu
__global__ void adc2_apply_M21_x1_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_x1,
    real_t* __restrict__ d_sigma2,
    int nocc, int nvir);

__global__ void adc2_apply_M12_x2_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_x2,
    real_t* __restrict__ d_sigma1,
    int nocc, int nvir);

/**
 * @brief σ2 += D2 · R2 (element-wise multiply by diagonal)
 */
__global__ void adc2_full_apply_D2_kernel(
    const real_t* __restrict__ d_D2,
    const real_t* __restrict__ d_R2,
    real_t* __restrict__ d_sigma2,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    d_sigma2[idx] += d_D2[idx] * d_R2[idx];
}

/**
 * @brief Build full-space diagonal: [diag(M11) | D2]
 */
__global__ void adc2_full_build_diagonal_kernel(
    const real_t* __restrict__ d_M11,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_diagonal,
    int singles_dim, int doubles_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = singles_dim + doubles_dim;
    if (idx >= total) return;

    if (idx < singles_dim) {
        // Diagonal of M11 (column-major): M11[idx + idx * singles_dim]
        d_diagonal[idx] = d_M11[(size_t)idx + (size_t)idx * singles_dim];
    } else {
        d_diagonal[idx] = d_D2[idx - singles_dim];
    }
}

/**
 * @brief Preconditioner: output[i] = input[i] / diagonal[i]
 */
__global__ void adc2_full_preconditioner_kernel(
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t diag = d_diagonal[idx];
    d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
}


ADC2FullOperator::ADC2FullOperator(const ADC2Operator& adc2_op)
    : adc2_op_(adc2_op),
      singles_dim_(adc2_op.get_singles_dim()),
      doubles_dim_(adc2_op.get_doubles_dim()),
      total_dim_(adc2_op.get_singles_dim() + adc2_op.get_doubles_dim()),
      d_diagonal_(nullptr)
{
    // Allocate and build diagonal
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));

    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    adc2_full_build_diagonal_kernel<<<blocks, threads>>>(
        adc2_op_.get_M11(), adc2_op_.get_D2(),
        d_diagonal_, singles_dim_, doubles_dim_);
    cudaDeviceSynchronize();
}

ADC2FullOperator::~ADC2FullOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void ADC2FullOperator::apply(const real_t* d_input, real_t* d_output) const {
    int threads = 256;

    const real_t* d_R1 = d_input;                    // [singles_dim]
    const real_t* d_R2 = d_input + singles_dim_;     // [doubles_dim]
    real_t* d_sigma1 = d_output;                     // [singles_dim]
    real_t* d_sigma2 = d_output + singles_dim_;      // [doubles_dim]

    int nocc = adc2_op_.get_nocc();
    int nvir = adc2_op_.get_nvir();

    // --- σ1 = M11 · R1 ---
    {
        const real_t alpha = 1.0;
        const real_t beta = 0.0;
        cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                    singles_dim_, singles_dim_, &alpha,
                    adc2_op_.get_M11(), singles_dim_,
                    d_R1, 1,
                    &beta, d_sigma1, 1);
    }

    // --- σ1 += M12 · R2 --- (via δ-structure kernel, accumulates into d_sigma1)
    {
        int blocks = (singles_dim_ + threads - 1) / threads;
        adc2_apply_M12_x2_kernel<<<blocks, threads>>>(
            adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
            d_R2, d_sigma1,
            nocc, nvir);
        cudaDeviceSynchronize();
    }

    // --- σ2 = M21 · R1 --- (via δ-structure kernel)
    {
        int blocks = (doubles_dim_ + threads - 1) / threads;
        adc2_apply_M21_x1_kernel<<<blocks, threads>>>(
            adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
            d_R1, d_sigma2,
            nocc, nvir);
        cudaDeviceSynchronize();
    }

    // --- σ2 += D2 · R2 --- (diagonal)
    {
        int blocks = (doubles_dim_ + threads - 1) / threads;
        adc2_full_apply_D2_kernel<<<blocks, threads>>>(
            adc2_op_.get_D2(), d_R2, d_sigma2, doubles_dim_);
        cudaDeviceSynchronize();
    }
}

void ADC2FullOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    adc2_full_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, total_dim_);
}

} // namespace gansu
