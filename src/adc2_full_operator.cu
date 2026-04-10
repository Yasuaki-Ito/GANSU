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

        // CPU fallback: σ1 += M12 · R2
        const real_t* d_eri_vvov = adc2_op_.get_eri_vvov();
        const real_t* d_eri_ooov = adc2_op_.get_eri_ooov();
        #pragma omp parallel for
        for (int ia = 0; ia < singles_dim_; ia++) {
            int i = ia / nvir;
            int a = ia % nvir;
            real_t sigma = 0.0;
            for (int j = 0; j < nocc; j++)
                for (int b = 0; b < nvir; b++) {
                    real_t r2_ijab = d_R2[(size_t)i * nocc * nvir * nvir + (size_t)j * nvir * nvir + (size_t)a * nvir + b];
                    real_t r2_jiab = d_R2[(size_t)j * nocc * nvir * nvir + (size_t)i * nvir * nvir + (size_t)a * nvir + b];
                    for (int c = 0; c < nvir; c++) {
                        real_t vvov_cb_j_a_val = d_eri_vvov[(size_t)c * nvir * nocc * nvir + (size_t)b * nocc * nvir + (size_t)j * nvir + a];
                        sigma += vvov_cb_j_a_val * (2.0 * r2_ijab - r2_jiab);
                    }
                    for (int k = 0; k < nocc; k++) {
                        real_t ooov_kj_i_b_val = d_eri_ooov[(size_t)k * nocc * nocc * nvir + (size_t)j * nocc * nvir + (size_t)i * nvir + b];
                        sigma -= ooov_kj_i_b_val * (2.0 * r2_ijab - r2_jiab);
                    }
                }
            d_sigma1[ia] += sigma;
        }

        // CPU fallback: σ2 = M21 · R1
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int ii = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int jj = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int aa = rem / nvir;
            int bb = rem % nvir;
            real_t sigma = 0.0;
            for (int c = 0; c < nvir; c++) {
                real_t x1_ic = d_R1[ii * nvir + c];
                real_t x1_jc = d_R1[jj * nvir + c];
                real_t vvov_cab_j = d_eri_vvov[(size_t)c * nvir * nocc * nvir + (size_t)aa * nocc * nvir + (size_t)jj * nvir + bb];
                real_t vvov_cba_i = d_eri_vvov[(size_t)c * nvir * nocc * nvir + (size_t)bb * nocc * nvir + (size_t)ii * nvir + aa];
                sigma += vvov_cab_j * (2.0 * x1_ic - x1_jc) - vvov_cba_i * x1_jc + vvov_cab_j * x1_jc
                       - vvov_cab_j * x1_jc;
                // Simplified: follow the kernel logic exactly
            }
            // Re-do with proper kernel logic
            sigma = 0.0;
            for (int c = 0; c < nvir; c++) {
                real_t vv_c_a_j_b = d_eri_vvov[(size_t)c * nvir * nocc * nvir + (size_t)aa * nocc * nvir + (size_t)jj * nvir + bb];
                real_t vv_c_b_i_a = d_eri_vvov[(size_t)c * nvir * nocc * nvir + (size_t)bb * nocc * nvir + (size_t)ii * nvir + aa];
                sigma += (2.0 * vv_c_a_j_b - vv_c_b_i_a) * d_R1[ii * nvir + c];
                sigma += (2.0 * vv_c_b_i_a - vv_c_a_j_b) * d_R1[jj * nvir + c];
            }
            for (int k = 0; k < nocc; k++) {
                real_t oo_k_j_i_b = d_eri_ooov[(size_t)k * nocc * nocc * nvir + (size_t)jj * nocc * nvir + (size_t)ii * nvir + bb];
                real_t oo_k_i_j_a = d_eri_ooov[(size_t)k * nocc * nocc * nvir + (size_t)ii * nocc * nvir + (size_t)jj * nvir + aa];
                sigma -= (2.0 * oo_k_j_i_b - oo_k_i_j_a) * d_R1[k * nvir + aa];
                sigma -= (2.0 * oo_k_i_j_a - oo_k_j_i_b) * d_R1[k * nvir + bb];
            }
            d_sigma2[idx] = sigma;
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
