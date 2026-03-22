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
 * @file adc2x_operator.cu
 * @brief GPU implementation of ADC(2)-x (extended) operator
 *
 * ADC(2)-x = ADC(2)-s + first-order off-diagonal M22 terms.
 *
 * Reuses ADC2Operator for M11, M12, M21 (identical to ADC(2)-s).
 * Adds first-order M22 corrections derived from spin-orbital:
 *   (1/2)<ab||cd> r2^{cd}_{ij} + (1/2)<kl||ij> r2^{ab}_{kl} + <ak||ic> r2^{bc}_{jk}
 *
 * After singlet RHF spin integration:
 *   oooo: Σ_{kl} (ki|lj) r^{ab}_{kl}
 *   vvvv: Σ_{cd} (ac|bd) r^{cd}_{ij}
 *   voov: Σ_{kc} [(ai|kc) - (ac|ki)] r^{bc}_{jk}
 *
 * σ1 = M11·R1 + M12·R2   (same as ADC(2)-s full operator)
 * σ2 = M21·R1 + D2·R2 + V·R2  (D2 diagonal + V off-diagonal)
 */

#include <cstdio>
#include <cmath>
#include <vector>

#include "adc2x_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"

namespace gansu {

// Reuse ERI extraction kernels from eom_mp2_operator.cu
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int);

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


// ========================================================================
//  M22 first-order off-diagonal kernel
// ========================================================================

/**
 * @brief σ2 += D2·R2 + V·R2 (diagonal + first-order off-diagonal)
 *
 * D2 (diagonal): (ε_a + ε_b - ε_i - ε_j) × r2[i,j,a,b]
 *
 * V (first-order off-diagonal, from RHF spin integration):
 *   oooo: Σ_{kl} (ik|jl) × r2[k,l,a,b]
 *   vvvv: Σ_{cd} (ac|bd) × r2[i,j,c,d]
 *   voov (8 terms, verified against PySCF radc_ee.py):
 *     +2(jb|kc) r^{ac}_{ik} - (kj|bc) r^{ac}_{ik}
 *     -(jb|kc) r^{ca}_{ik}  - (kj|ac) r^{cb}_{ik}
 *     +2(ia|kc) r^{bc}_{jk} - (ki|ac) r^{bc}_{jk}
 *     -(ki|bc) r^{ca}_{jk}  - (ia|kc) r^{cb}_{jk}
 *
 * ERI layout:
 *   ovov[i,a,k,c] = (ia|kc)
 *   oovv[k,i,a,c] = (ki|ac)
 *   oooo[i,k,j,l] = (ik|jl)
 *   vvvv[a,c,b,d] = (ac|bd)
 */
__global__ void adc2x_M22_R2_kernel(
    const real_t* __restrict__ d_D2,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_oooo,
    const real_t* __restrict__ d_eri_vvvv,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_R2,
    real_t* __restrict__ d_sigma2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doubles_dim = nocc * nocc * nvir * nvir;
    if (idx >= doubles_dim) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    real_t sigma = 0.0;

    // D2 diagonal: (ε_a + ε_b - ε_i - ε_j) × r2[i,j,a,b]
    sigma += d_D2[idx] * d_R2[idx];

    // oooo: Σ_{kl} (ik|jl) × r2[k,l,a,b]
    for (int k = 0; k < nocc; k++) {
        for (int l = 0; l < nocc; l++) {
            // oooo[i,k,j,l] = (ik|jl)
            real_t ik_jl = d_eri_oooo[(size_t)i * nocc * nocc * nocc +
                                      (size_t)k * nocc * nocc +
                                      (size_t)j * nocc + l];
            real_t r2_klab = d_R2[(size_t)k * nocc * nvir * nvir +
                                  (size_t)l * nvir * nvir +
                                  (size_t)a * nvir + b];
            sigma += ik_jl * r2_klab;
        }
    }

    // vvvv: Σ_{cd} (ac|bd) × r2[i,j,c,d]
    for (int c = 0; c < nvir; c++) {
        for (int d = 0; d < nvir; d++) {
            // vvvv[a,c,b,d] = (ac|bd)
            real_t ac_bd = d_eri_vvvv[(size_t)a * nvir * nvir * nvir +
                                      (size_t)c * nvir * nvir +
                                      (size_t)b * nvir + d];
            real_t r2_ijcd = d_R2[(size_t)i * nocc * nvir * nvir +
                                  (size_t)j * nvir * nvir +
                                  (size_t)c * nvir + d];
            sigma += ac_bd * r2_ijcd;
        }
    }

    // voov: 8 terms from RHF spin integration
    // ERIs: ovov[p,q,r,s] = (pq|rs), oovv[p,q,r,s] = (pq|rs)
    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    size_t oovv_stride = (size_t)nocc * nvir * nvir;
    for (int k = 0; k < nocc; k++) {
        for (int c = 0; c < nvir; c++) {
            // ERI values
            real_t jb_kc = d_eri_ovov[(size_t)j * ovov_stride +
                                      (size_t)b * nocc * nvir +
                                      (size_t)k * nvir + c];
            real_t ia_kc = d_eri_ovov[(size_t)i * ovov_stride +
                                      (size_t)a * nocc * nvir +
                                      (size_t)k * nvir + c];
            real_t kj_bc = d_eri_oovv[(size_t)k * oovv_stride +
                                      (size_t)j * nvir * nvir +
                                      (size_t)b * nvir + c];
            real_t kj_ac = d_eri_oovv[(size_t)k * oovv_stride +
                                      (size_t)j * nvir * nvir +
                                      (size_t)a * nvir + c];
            real_t ki_ac = d_eri_oovv[(size_t)k * oovv_stride +
                                      (size_t)i * nvir * nvir +
                                      (size_t)a * nvir + c];
            real_t ki_bc = d_eri_oovv[(size_t)k * oovv_stride +
                                      (size_t)i * nvir * nvir +
                                      (size_t)b * nvir + c];

            // R2 values: r2[p,q,r,s] at index p*nocc*nvir*nvir + q*nvir*nvir + r*nvir + s
            real_t r2_ik_ac = d_R2[(size_t)i * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)a * nvir + c];
            real_t r2_ik_ca = d_R2[(size_t)i * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)c * nvir + a];
            real_t r2_ik_cb = d_R2[(size_t)i * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)c * nvir + b];
            real_t r2_jk_bc = d_R2[(size_t)j * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)b * nvir + c];
            real_t r2_jk_ca = d_R2[(size_t)j * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)c * nvir + a];
            real_t r2_jk_cb = d_R2[(size_t)j * nocc * nvir * nvir +
                                   (size_t)k * nvir * nvir +
                                   (size_t)c * nvir + b];

            // 8 voov terms:
            sigma += 2.0 * jb_kc * r2_ik_ac;   // +2(jb|kc) r^{ac}_{ik}
            sigma -=       kj_bc * r2_ik_ac;   //  -(kj|bc) r^{ac}_{ik}
            sigma -=       jb_kc * r2_ik_ca;   //  -(jb|kc) r^{ca}_{ik}
            sigma -=       kj_ac * r2_ik_cb;   //  -(kj|ac) r^{cb}_{ik}
            sigma += 2.0 * ia_kc * r2_jk_bc;   // +2(ia|kc) r^{bc}_{jk}
            sigma -=       ki_ac * r2_jk_bc;   //  -(ki|ac) r^{bc}_{jk}
            sigma -=       ki_bc * r2_jk_ca;   //  -(ki|bc) r^{ca}_{jk}
            sigma -=       ia_kc * r2_jk_cb;   //  -(ia|kc) r^{cb}_{jk}
        }
    }

    d_sigma2[idx] += sigma;
}


/**
 * @brief Build full-space diagonal: [diag(M11) | D2]
 * Same as ADC(2)-s (D2 approximation for doubles preconditioner)
 */
__global__ void adc2x_build_diagonal_kernel(
    const real_t* __restrict__ d_M11,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_diagonal,
    int singles_dim, int doubles_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = singles_dim + doubles_dim;
    if (idx >= total) return;

    if (idx < singles_dim) {
        d_diagonal[idx] = d_M11[(size_t)idx + (size_t)idx * singles_dim];
    } else {
        d_diagonal[idx] = d_D2[idx - singles_dim];
    }
}

/**
 * @brief Preconditioner: output[i] = input[i] / diagonal[i]
 */
__global__ void adc2x_preconditioner_kernel(
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


// ========================================================================
//  ADC2XOperator Implementation
// ========================================================================

ADC2XOperator::ADC2XOperator(const ADC2Operator& adc2_op,
                             const real_t* d_eri_mo, int nao)
    : adc2_op_(adc2_op),
      nocc_(adc2_op.get_nocc()),
      nvir_(adc2_op.get_nvir()),
      singles_dim_(adc2_op.get_singles_dim()),
      doubles_dim_(adc2_op.get_doubles_dim()),
      total_dim_(adc2_op.get_singles_dim() + adc2_op.get_doubles_dim()),
      d_eri_oooo_(nullptr),
      d_eri_vvvv_(nullptr),
      d_eri_oovv_(nullptr),
      d_diagonal_(nullptr)
{
    int threads = 256;
    int blocks;

    // Extract additional ERI blocks needed for M22 first-order terms
    size_t oooo_size = (size_t)nocc_ * nocc_ * nocc_ * nocc_;
    tracked_cudaMalloc(&d_eri_oooo_, oooo_size * sizeof(real_t));
    blocks = (oooo_size + threads - 1) / threads;
    eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_, nao);

    size_t vvvv_size = (size_t)nvir_ * nvir_ * nvir_ * nvir_;
    tracked_cudaMalloc(&d_eri_vvvv_, vvvv_size * sizeof(real_t));
    blocks = (vvvv_size + threads - 1) / threads;
    eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_, nvir_, nao);

    size_t oovv_size = (size_t)nocc_ * nocc_ * nvir_ * nvir_;
    tracked_cudaMalloc(&d_eri_oovv_, oovv_size * sizeof(real_t));
    blocks = (oovv_size + threads - 1) / threads;
    eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_, nvir_, nao);

    cudaDeviceSynchronize();

    // Build diagonal: [diag(M11) | D2]
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));
    blocks = (total_dim_ + threads - 1) / threads;
    adc2x_build_diagonal_kernel<<<blocks, threads>>>(
        adc2_op_.get_M11(), adc2_op_.get_D2(),
        d_diagonal_, singles_dim_, doubles_dim_);
    cudaDeviceSynchronize();
}

ADC2XOperator::~ADC2XOperator() {
    if (d_eri_oooo_) tracked_cudaFree(d_eri_oooo_);
    if (d_eri_vvvv_) tracked_cudaFree(d_eri_vvvv_);
    if (d_eri_oovv_) tracked_cudaFree(d_eri_oovv_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}


void ADC2XOperator::apply(const real_t* d_input, real_t* d_output) const {
    int threads = 256;

    const real_t* d_R1 = d_input;
    const real_t* d_R2 = d_input + singles_dim_;
    real_t* d_sigma1 = d_output;
    real_t* d_sigma2 = d_output + singles_dim_;

    // --- σ1 = M11 · R1 (DGEMV) ---
    {
        const real_t alpha = 1.0;
        const real_t beta = 0.0;
        cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                    singles_dim_, singles_dim_, &alpha,
                    adc2_op_.get_M11(), singles_dim_,
                    d_R1, 1,
                    &beta, d_sigma1, 1);
    }

    // --- σ1 += M12 · R2 (kernel from ADC2Operator) ---
    {
        int blocks = (singles_dim_ + threads - 1) / threads;
        adc2_apply_M12_x2_kernel<<<blocks, threads>>>(
            adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
            d_R2, d_sigma1,
            nocc_, nvir_);
        cudaDeviceSynchronize();
    }

    // --- σ2 = M21 · R1 (kernel from ADC2Operator, writes to d_sigma2) ---
    {
        int blocks = (doubles_dim_ + threads - 1) / threads;
        adc2_apply_M21_x1_kernel<<<blocks, threads>>>(
            adc2_op_.get_eri_vvov(), adc2_op_.get_eri_ooov(),
            d_R1, d_sigma2,
            nocc_, nvir_);
        cudaDeviceSynchronize();
    }

    // --- σ2 += D2·R2 + V·R2 (M22 with first-order off-diagonal terms) ---
    {
        int blocks = (doubles_dim_ + threads - 1) / threads;
        adc2x_M22_R2_kernel<<<blocks, threads>>>(
            adc2_op_.get_D2(),
            adc2_op_.get_eri_ovov(),
            d_eri_oooo_, d_eri_vvvv_, d_eri_oovv_,
            d_R2, d_sigma2,
            nocc_, nvir_);
        cudaDeviceSynchronize();
    }
}


void ADC2XOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    adc2x_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, total_dim_);
}

} // namespace gansu
