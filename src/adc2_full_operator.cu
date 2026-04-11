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

    if (!gpu::gpu_available()) {
        for (int idx = 0; idx < total_dim_; idx++) {
            if (idx < singles_dim_) {
                d_diagonal_[idx] = adc2_op_.get_M11()[(size_t)idx + (size_t)idx * singles_dim_];
            } else {
                d_diagonal_[idx] = adc2_op_.get_D2()[idx - singles_dim_];
            }
        }
    } else {
        int threads = 256;
        int blocks = (total_dim_ + threads - 1) / threads;
        adc2_full_build_diagonal_kernel<<<blocks, threads>>>(
            adc2_op_.get_M11(), adc2_op_.get_D2(),
            d_diagonal_, singles_dim_, doubles_dim_);
        cudaDeviceSynchronize();
    }
}

ADC2FullOperator::~ADC2FullOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void ADC2FullOperator::apply(const real_t* d_input, real_t* d_output) const {
    const real_t* d_R1 = d_input;                    // [singles_dim]
    const real_t* d_R2 = d_input + singles_dim_;     // [doubles_dim]
    real_t* d_sigma1 = d_output;                     // [singles_dim]
    real_t* d_sigma2 = d_output + singles_dim_;      // [doubles_dim]

    int nocc = adc2_op_.get_nocc();
    int nvir = adc2_op_.get_nvir();

    if (!gpu::gpu_available()) {
        // CPU fallback: σ1 = M11 · R1
        const real_t* M11 = adc2_op_.get_M11();
        #pragma omp parallel for
        for (int row = 0; row < singles_dim_; row++) {
            real_t sum = 0.0;
            for (int col = 0; col < singles_dim_; col++)
                sum += M11[(size_t)col * singles_dim_ + row] * d_R1[col];
            d_sigma1[row] = sum;
        }

        // CPU fallback: σ1 += M12 · R2  (matches adc2_apply_M12_x2_kernel)
        const real_t* d_eri_vvov = adc2_op_.get_eri_vvov();
        const real_t* d_eri_ooov = adc2_op_.get_eri_ooov();
        int vov = nvir * nocc * nvir;
        int vv = nvir * nvir;
        #pragma omp parallel for
        for (int idx = 0; idx < singles_dim_; idx++) {
            int K = idx / nvir;
            int E = idx % nvir;
            real_t val = 0.0;
            // Term A: Σ_{J,C,D} [2·(EC|JD) - (DE|JC)] · R2(K,J,C,D)
            for (int J = 0; J < nocc; J++)
                for (int C = 0; C < nvir; C++)
                    for (int D = 0; D < nvir; D++) {
                        real_t eri1 = d_eri_vvov[(size_t)E * vov + (size_t)C * nocc * nvir + J * nvir + D];
                        real_t eri2 = d_eri_vvov[(size_t)D * vov + (size_t)E * nocc * nvir + J * nvir + C];
                        real_t x2v = d_R2[(size_t)K * nocc * vv + (size_t)J * vv + C * nvir + D];
                        val += (2.0 * eri1 - eri2) * x2v;
                    }
            // Term B: Σ_{I,J,D} [(JK|ID) - 2·(IK|JD)] · R2(I,J,E,D)
            for (int I = 0; I < nocc; I++)
                for (int J = 0; J < nocc; J++)
                    for (int D = 0; D < nvir; D++) {
                        real_t eri1 = d_eri_ooov[(size_t)J * nocc * nocc * nvir + (size_t)K * nocc * nvir + I * nvir + D];
                        real_t eri2 = d_eri_ooov[(size_t)I * nocc * nocc * nvir + (size_t)K * nocc * nvir + J * nvir + D];
                        real_t x2v = d_R2[(size_t)I * nocc * vv + (size_t)J * vv + E * nvir + D];
                        val += (eri1 - 2.0 * eri2) * x2v;
                    }
            d_sigma1[idx] += val;
        }

        // CPU fallback: σ2 = M21 · R1  (matches adc2_apply_M21_x1_kernel)
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int I = idx / (nocc * vv);
            int rem = idx % (nocc * vv);
            int J = rem / vv;
            rem = rem % vv;
            int C = rem / nvir;
            int D = rem % nvir;
            real_t val = 0.0;
            // Term 1: Σ_E (EC|JD)·R1[I,E]
            for (int E = 0; E < nvir; E++)
                val += d_eri_vvov[(size_t)E * vov + (size_t)C * nocc * nvir + J * nvir + D] * d_R1[I * nvir + E];
            // Term 2: Σ_E (ED|IC)·R1[J,E]
            for (int E = 0; E < nvir; E++)
                val += d_eri_vvov[(size_t)E * vov + (size_t)D * nocc * nvir + I * nvir + C] * d_R1[J * nvir + E];
            // Term 3: -Σ_K (IK|JD)·R1[K,C]
            for (int K = 0; K < nocc; K++)
                val -= d_eri_ooov[(size_t)I * nocc * nocc * nvir + (size_t)K * nocc * nvir + J * nvir + D] * d_R1[K * nvir + C];
            // Term 4: -Σ_K (JK|IC)·R1[K,D]
            for (int K = 0; K < nocc; K++)
                val -= d_eri_ooov[(size_t)J * nocc * nocc * nvir + (size_t)K * nocc * nvir + I * nvir + C] * d_R1[K * nvir + D];
            d_sigma2[idx] = val;
        }

        // CPU fallback: σ2 += D2 · R2
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++)
            d_sigma2[idx] += adc2_op_.get_D2()[idx] * d_R2[idx];

        return;
    }

    int threads = 256;

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

    if (adc2_op_.is_dense_M12()) {
        // --- Dense path: σ1 += M12 · R2, σ2 = M21 · R1 via cuBLAS DGEMV ---
        {
            const real_t alpha = 1.0;
            const real_t beta = 1.0;  // accumulate onto σ1
            cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                        singles_dim_, doubles_dim_, &alpha,
                        adc2_op_.get_M12(), singles_dim_,
                        d_R2, 1,
                        &beta, d_sigma1, 1);
        }
        {
            const real_t alpha = 1.0;
            const real_t beta = 0.0;
            cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                        doubles_dim_, singles_dim_, &alpha,
                        adc2_op_.get_M21(), doubles_dim_,
                        d_R1, 1,
                        &beta, d_sigma2, 1);
        }
    } else {
        // --- Kernel-based path: exploit δ-structure ---
        // σ1 += M12 · R2
        {
            int blocks = (singles_dim_ + threads - 1) / threads;
            adc2_apply_M12_x2_kernel<<<blocks, threads>>>(
                adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
                d_R2, d_sigma1,
                nocc, nvir);
            cudaDeviceSynchronize();
        }
        // σ2 = M21 · R1
        {
            int blocks = (doubles_dim_ + threads - 1) / threads;
            adc2_apply_M21_x1_kernel<<<blocks, threads>>>(
                adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
                d_R1, d_sigma2,
                nocc, nvir);
            cudaDeviceSynchronize();
        }
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
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; idx++) {
            real_t diag = d_diagonal_[idx];
            d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
        }
        return;
    }
    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    adc2_full_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, total_dim_);
}

} // namespace gansu
