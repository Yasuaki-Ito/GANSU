/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ri_adc2_schur_operator.cu
 * @brief RI-factored ADC(2) Schur complement — GPU DGEMM implementation
 *
 * Sigma = M11·x + Σ_{I,J} M12[·,(I,J,·,·)] × W_{IJCD} × M21[(I,J,·,·),·] × x
 *
 * Algorithm per (I,J) pair (4 DGEMM + 2 kernels + 1 DGEMM per J):
 *   Precompute (once): φ^P_C(I), ψ^P_C(I) for all I
 *   Per J: eri_vvov_J = B_ab × b_J^T  [vv × nvir]
 *   Per (I,J):
 *     α_I = φ_I - ψ_I, β_J = φ_J - ψ_J
 *     R = α_I × b_J^T + b_I × β_J^T   [2 DGEMM]
 *     W = R / (ω + εI + εJ - εC - εD)   [kernel]
 *     Group A: σ[I,E] += Σ_{CD} [2(EC|JD)-(DE|JC)]×W  [kernel]
 *     Group B: σ += W × [(JK|ID)-2(IK|JD)]^T            [2 DGEMM + 1 DGEMM]
 */

#include "ri_adc2_schur_operator.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

using gpu::GPUHandle;

// ============================================================
//  CUDA Kernels
// ============================================================

// W_all[(I*nvir+C) + D*ov] = R_all[(I*nvir+C) + D*ov] / (omega + eps_occ[I] + eps_occ_J - eps_vir[C] - eps_vir[D])
// Layout: col-major [ov × nvir]
__global__ void ri_adc2_weight_all_kernel(
    const double* __restrict__ R_all,
    double* __restrict__ W_all,
    const double* __restrict__ eps_occ,
    const double* __restrict__ eps_vir,
    double omega_J, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * nvir) return;
    int ic = idx % ov;  // I*nvir + C
    int D = idx / ov;
    int I = ic / nvir;
    int C = ic % nvir;
    double denom = omega_J + eps_occ[I] - eps_vir[C] - eps_vir[D];
    if (fabs(denom) < 1e-12) denom = (denom >= 0.0) ? 1e-12 : -1e-12;
    W_all[idx] = R_all[idx] / denom;
}

// Group A contraction for ALL I: σ[I*nvir+E] += Σ_{C,D} [2(EC|JD) - (DE|JC)] × W_all[I*nvir+C, D]
// eri_vvov[FC, D] col-major [vv × nvir]: eri[F*nvir+C + D*vv] = (FC|JD)
// W_all col-major [ov × nvir]: W_all[(I*nvir+C) + D*ov]
__global__ void ri_adc2_group_A_all_kernel(
    const double* __restrict__ eri_vvov,
    const double* __restrict__ W_all,
    double* __restrict__ sigma,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;  // one thread per (I, E)
    int I = idx / nvir;
    int E = idx % nvir;
    int vv = nvir * nvir;

    double coul = 0.0, xc = 0.0;
    for (int D = 0; D < nvir; D++) {
        for (int C = 0; C < nvir; C++) {
            double w = W_all[(I * nvir + C) + D * ov];
            coul += eri_vvov[E * nvir + C + D * vv] * w;  // (EC|JD)
            xc   += eri_vvov[D * nvir + E + C * vv] * w;  // (DE|JC)
        }
    }
    sigma[idx] += 2.0 * coul - xc;
}

// Group B contraction for ALL I: σ[K*nvir+E] += Σ_{I,D} [(JK|ID)-2(IK|JD)] × W_all[I*nvir+E, D]
// ooov1_all[K, I*nvir+D] = (JK|ID), col-major [nocc × ov]
// ooov2_all[I*nocc+K, D] = (IK|JD), col-major [oo × nvir]
// W_all[I*nvir+E, D] col-major [ov × nvir]
__global__ void ri_adc2_group_B_all_kernel(
    const double* __restrict__ ooov1_all,
    const double* __restrict__ ooov2_all,
    const double* __restrict__ W_all,
    double* __restrict__ sigma,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    int oo = nocc * nocc;
    if (idx >= ov) return;  // one thread per (K, E)
    int K = idx / nvir;
    int E = idx % nvir;

    double val = 0.0;
    for (int I = 0; I < nocc; I++) {
        for (int D = 0; D < nvir; D++) {
            double jk_id = ooov1_all[K + (size_t)(I * nvir + D) * nocc];  // [nocc × ov] col-major
            double ik_jd = ooov2_all[(I * nocc + K) + (size_t)D * oo];    // [oo × nvir] col-major
            double w = W_all[(I * nvir + E) + (size_t)D * ov];            // [ov × nvir] col-major
            val += (jk_id - 2.0 * ik_jd) * w;
        }
    }
    sigma[idx] += val;
}

// ============================================================
//  Constructor
// ============================================================
RIADC2SchurOperator::RIADC2SchurOperator(
    const real_t* d_B_ia,
    const real_t* d_B_ab,
    const real_t* d_B_ij,
    const real_t* d_M11,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int naux)
    : nocc_(nocc), nvir_(nvir), naux_(naux),
      ov_(nocc * nvir), vv_(nvir * nvir), oo_(nocc * nocc),
      d_B_ia_(d_B_ia), d_B_ab_(d_B_ab), d_B_ij_(d_B_ij)
{
    std::vector<double> eps(nocc + nvir);
    cudaMemcpy(eps.data(), d_orbital_energies, (nocc + nvir) * sizeof(double), cudaMemcpyDeviceToHost);
    eps_occ_.assign(eps.begin(), eps.begin() + nocc);
    eps_vir_.assign(eps.begin() + nocc, eps.end());

    // Upload orbital energies to device
    tracked_cudaMalloc(&d_eps_occ_dev_, nocc * sizeof(real_t));
    tracked_cudaMalloc(&d_eps_vir_dev_, nvir * sizeof(real_t));
    cudaMemcpy(d_eps_occ_dev_, eps_occ_.data(), nocc * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps_vir_dev_, eps_vir_.data(), nvir * sizeof(double), cudaMemcpyHostToDevice);

    // Copy M11
    tracked_cudaMalloc(&d_M11_, (size_t)ov_ * ov_ * sizeof(real_t));
    cudaMemcpy(d_M11_, d_M11, (size_t)ov_ * ov_ * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // D1 and diagonal
    tracked_cudaMalloc(&d_D1_, ov_ * sizeof(real_t));
    tracked_cudaMalloc(&d_diagonal_, ov_ * sizeof(real_t));
    compute_D1();
    compute_diagonal();

    // Precomputed intermediates
    size_t phi_size = (size_t)nvir * nocc * naux;
    tracked_cudaMalloc(&d_phi_, phi_size * sizeof(real_t));
    tracked_cudaMalloc(&d_psi_, phi_size * sizeof(real_t));

    // Per-J workspace (I loop eliminated)
    tracked_cudaMalloc(&d_alpha_all_, (size_t)ov_ * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_beta_, (size_t)nvir * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_R_all_, (size_t)ov_ * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_W_all_, (size_t)ov_ * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_vvov_, (size_t)vv_ * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_ooov1_all_, (size_t)nocc * ov_ * sizeof(real_t));
    tracked_cudaMalloc(&d_ooov2_all_, (size_t)oo_ * nvir * sizeof(real_t));

    std::cout << "[RI-ADC(2)-Schur] Initialized: nocc=" << nocc << " nvir=" << nvir
              << " naux=" << naux << " ov=" << ov_ << std::endl;
}

RIADC2SchurOperator::~RIADC2SchurOperator() {
    tracked_cudaFree(d_M11_);
    tracked_cudaFree(d_D1_);
    tracked_cudaFree(d_diagonal_);
    tracked_cudaFree(d_eps_occ_dev_);
    tracked_cudaFree(d_eps_vir_dev_);
    tracked_cudaFree(d_phi_);
    tracked_cudaFree(d_psi_);
    tracked_cudaFree(d_alpha_all_);
    tracked_cudaFree(d_beta_);
    tracked_cudaFree(d_R_all_);
    tracked_cudaFree(d_W_all_);
    tracked_cudaFree(d_eri_vvov_);
    tracked_cudaFree(d_ooov1_all_);
    tracked_cudaFree(d_ooov2_all_);
}

void RIADC2SchurOperator::compute_D1() {
    std::vector<double> D1(ov_);
    for (int i = 0; i < nocc_; i++)
        for (int a = 0; a < nvir_; a++)
            D1[i * nvir_ + a] = eps_vir_[a] - eps_occ_[i];
    cudaMemcpy(d_D1_, D1.data(), ov_ * sizeof(double), cudaMemcpyHostToDevice);
}

void RIADC2SchurOperator::compute_diagonal() {
    cudaMemcpy(d_diagonal_, d_D1_, ov_ * sizeof(double), cudaMemcpyDeviceToDevice);
}

void RIADC2SchurOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    std::vector<double> D1(ov_), input(ov_);
    cudaMemcpy(D1.data(), d_D1_, ov_ * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.data(), d_input, ov_ * sizeof(double), cudaMemcpyDeviceToHost);
    for (int ia = 0; ia < ov_; ia++) {
        double denom = D1[ia] - omega_;
        if (std::abs(denom) < 1e-10) denom = (denom >= 0) ? 1e-10 : -1e-10;
        input[ia] /= denom;
    }
    cudaMemcpy(d_output, input.data(), ov_ * sizeof(double), cudaMemcpyHostToDevice);
}

// ============================================================
//  CORE: sigma = M_eff(ω) · x — GPU DGEMM implementation
// ============================================================
void RIADC2SchurOperator::apply(const real_t* d_input, real_t* d_output) const {
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double one = 1.0, zero = 0.0, neg_one = -1.0, two = 2.0;
    const int threads = 256;

    // Step 0: σ = M11 · x
    cublasDgemv(handle, CUBLAS_OP_N, ov_, ov_,
                &one, d_M11_, ov_, d_input, 1, &zero, d_output, 1);

    // Step 1: Precompute φ and ψ using batched DGEMM
    // φ^P[C,I] = Σ_F B_ab^P[C,F] × x[F,I]  → cublasDgemmStridedBatched
    // ψ^P[C,I] = Σ_L x[C,L] × B_ij^P[L,I]^T → cublasDgemmStridedBatched
    // Layout: phi/psi[C + I*nvir + P*nvir*nocc]
    cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nvir_, nocc_, nvir_,
        &one,
        d_B_ab_, nvir_, (long long)vv_,
        d_input, nvir_, 0LL,
        &zero,
        d_phi_, nvir_, (long long)nvir_ * nocc_,
        naux_);

    cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nvir_, nocc_, nocc_,
        &one,
        d_input, nvir_, 0LL,
        d_B_ij_, nocc_, (long long)oo_,
        &zero,
        d_psi_, nvir_, (long long)nvir_ * nocc_,
        naux_);

    // Step 2: Compute α_all = φ - ψ as [ov × naux] contiguous
    // phi/psi layout: [C + I*nvir + P*nvir*nocc] = [(nvir*nocc) × naux] col-major with lda=nvir*nocc
    // α_all layout: [ov × naux] col-major with lda=ov (=nvir*nocc, same!)
    // So α_all = phi - psi element-wise (same layout!)
    {
        int total = ov_ * naux_;
        cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    total, 1,
                    &one, d_phi_, total,
                    &neg_one, d_psi_, total,
                    d_alpha_all_, total);
    }

    // Step 3: J loop (I loop eliminated)
    const int lda_ov = ov_;
    const int lda_oo = oo_;
    const int lda_phi = nvir_ * nocc_;

    // eps_occ already on device as d_eps_occ_dev_

    for (int J = 0; J < nocc_; J++) {
        const double* b_J = &d_B_ia_[J * nvir_];  // [nvir × naux], lda=ov

        // eri_vvov_J = B_ab × b_J^T: [vv × nvir]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    vv_, nvir_, naux_,
                    &one, d_B_ab_, vv_,
                    b_J, lda_ov,
                    &zero, d_eri_vvov_, vv_);

        // β_J = φ_J - ψ_J: [nvir × naux]
        cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    nvir_, naux_,
                    &one, &d_phi_[J * nvir_], lda_phi,
                    &neg_one, &d_psi_[J * nvir_], lda_phi,
                    d_beta_, nvir_);

        // R_all = α_all × b_J^T: [ov × naux] × [naux × nvir] → [ov × nvir]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    ov_, nvir_, naux_,
                    &one, d_alpha_all_, ov_,
                    b_J, lda_ov,
                    &zero, d_R_all_, ov_);

        // R_all += B_ia × β_J^T: [ov × naux] × [naux × nvir] → [ov × nvir]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    ov_, nvir_, naux_,
                    &one, d_B_ia_, ov_,
                    d_beta_, nvir_,
                    &one, d_R_all_, ov_);

        // W_all = R_all / (ω + εI + εJ - εC - εD) for all I
        {
            double omega_J = omega_ + eps_occ_[J];
            int total = ov_ * nvir_;
            int blk = (total + threads - 1) / threads;
            ri_adc2_weight_all_kernel<<<blk, threads>>>(
                d_R_all_, d_W_all_, d_eps_occ_dev_, d_eps_vir_dev_,
                omega_J, nocc_, nvir_);
        }

        // Group A for all I: σ[I*nvir+E] += Σ_{CD} [2(EC|JD)-(DE|JC)] × W_all[I*nvir+C, D]
        {
            int blk = (ov_ + threads - 1) / threads;
            ri_adc2_group_A_all_kernel<<<blk, threads>>>(
                d_eri_vvov_, d_W_all_, d_output, nocc_, nvir_);
        }

        // ooov1_all[K, I*nvir+D] = (JK|ID) = b_ij_J × B_ia^T: [nocc × ov]
        const double* b_ij_J = &d_B_ij_[J * nocc_];
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    nocc_, ov_, naux_,
                    &one, b_ij_J, lda_oo,
                    d_B_ia_, ov_,
                    &zero, d_ooov1_all_, nocc_);

        // ooov2_all[I*nocc+K, D] = (IK|JD) = B_ij × b_J^T: [oo × nvir]
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    oo_, nvir_, naux_,
                    &one, d_B_ij_, oo_,
                    b_J, lda_ov,
                    &zero, d_ooov2_all_, oo_);

        // Group B for all I: σ[K*nvir+E] += Σ_{I,D} [(JK|ID)-2(IK|JD)] × W_all[I*nvir+E, D]
        {
            int blk = (ov_ + threads - 1) / threads;
            ri_adc2_group_B_all_kernel<<<blk, threads>>>(
                d_ooov1_all_, d_ooov2_all_, d_W_all_, d_output, nocc_, nvir_);
        }

    } // J loop
}

// ============================================================
//  Kernels for M11 build
// ============================================================

// CIS matrix + OOVV exchange: M11[ia,jb] = δ_ij δ_ab(εa-εi) + 2(ia|jb) - (ij|ab)
// ovov: [ov × ov] col-major, ovov[ia + jb*ov] = (ia|jb)
// oovv: [oo × vv] col-major, oovv[(i*nocc+j) + (a*nvir+b)*oo] = (ij|ab)
__global__ void ri_adc2_build_cis_kernel(
    const double* __restrict__ ovov,
    const double* __restrict__ oovv,
    const double* __restrict__ eps,
    double* __restrict__ M11,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    int oo = nocc * nocc;
    if (idx >= ov * ov) return;

    int ia = idx % ov;
    int jb = idx / ov;
    int i = ia / nvir, a = ia % nvir;
    int j = jb / nvir, b = jb % nvir;

    double val = 2.0 * ovov[ia + (size_t)jb * ov];
    val -= oovv[(i * nocc + j) + (size_t)(a * nvir + b) * oo];
    if (i == j && a == b) val += eps[nocc + a] - eps[i];
    M11[idx] = val;
}

// T2[i,j,a,b] = ovov(i,a,j,b) / (εi+εj-εa-εb)
// Stored as 4D: t2[i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b]
// ovov: [ov × ov] col-major: (ia|jb) at ovov[(i*nvir+a) + (j*nvir+b)*ov]
__global__ void ri_adc2_build_t2_kernel(
    const double* __restrict__ ovov,
    const double* __restrict__ eps,
    double* __restrict__ t2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dd = nocc * nocc * nvir * nvir;
    int ov = nocc * nvir;
    if (idx >= dd) return;

    int b = idx % nvir;
    int a = (idx / nvir) % nvir;
    int j = (idx / (nvir * nvir)) % nocc;
    int i = idx / (nocc * nvir * nvir);

    double denom = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
    double ia_jb = ovov[(i * nvir + a) + (size_t)(j * nvir + b) * ov];
    t2[idx] = ia_jb / denom;
}

// Convert ovov [ov×ov] col-major → ovov 4D [i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]
__global__ void ri_adc2_ovov_to_4d_kernel(
    const double* __restrict__ ovov_matrix,
    double* __restrict__ ovov_4d,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    int ia = idx % ov;
    int jb = idx / ov;
    int i = ia / nvir, a = ia % nvir;
    int j = jb / nvir, b = jb % nvir;

    // 4D index: ovov[i][a][j][b] = i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b
    size_t idx_4d = (size_t)i * nvir * nocc * nvir + (size_t)a * nocc * nvir + j * nvir + b;
    ovov_4d[idx_4d] = ovov_matrix[idx];
}

// Add self-energy: M11[ia,jb] += -δ_ab Σ_oo[i,j] + δ_ij Σ_vv[a,b]
__global__ void ri_adc2_add_self_energy_kernel(
    const double* __restrict__ sigma_oo,
    const double* __restrict__ sigma_vv,
    double* __restrict__ M11,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    int ia = idx % ov;
    int jb = idx / ov;
    int i = ia / nvir, a = ia % nvir;
    int j = jb / nvir, b = jb % nvir;

    double correction = 0.0;
    if (a == b) correction -= sigma_oo[i * nocc + j];
    if (i == j) correction += sigma_vv[a * nvir + b];
    M11[idx] += correction;
}

// ISR correction kernel (same logic as adc2_operator.cu, duplicated for link independence)
__global__ void ri_m11_ISR_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_ISR_corr,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;
    int ia_idx = idx % ov, jb_idx = idx / ov;
    int i = ia_idx / nvir, a = ia_idx % nvir;
    int j = jb_idx / nvir, b = jb_idx % nvir;
    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t val = 0.0;
    for (int k = 0; k < nocc; k++)
        for (int c = 0; c < nvir; c++) {
            real_t t2_kiac = d_t2[(size_t)k*nocc*nvir*nvir + (size_t)i*nvir*nvir + (size_t)a*nvir + c];
            real_t t2_ikac = d_t2[(size_t)i*nocc*nvir*nvir + (size_t)k*nvir*nvir + (size_t)a*nvir + c];
            real_t ovov_kbjc = d_eri_ovov[(size_t)k*ovov_stride + (size_t)b*nocc*nvir + (size_t)j*nvir + c];
            real_t ovov_jbkc = d_eri_ovov[(size_t)j*ovov_stride + (size_t)b*nocc*nvir + (size_t)k*nvir + c];
            val += 0.5 * t2_kiac * ovov_kbjc;
            val += 2.0 * t2_ikac * ovov_jbkc;
            val -= t2_ikac * ovov_kbjc;
            val -= t2_kiac * ovov_jbkc;
            real_t t2_kjbc = d_t2[(size_t)k*nocc*nvir*nvir + (size_t)j*nvir*nvir + (size_t)b*nvir + c];
            real_t t2_jkbc = d_t2[(size_t)j*nocc*nvir*nvir + (size_t)k*nvir*nvir + (size_t)b*nvir + c];
            real_t ovov_kaic = d_eri_ovov[(size_t)k*ovov_stride + (size_t)a*nocc*nvir + (size_t)i*nvir + c];
            real_t ovov_iakc = d_eri_ovov[(size_t)i*ovov_stride + (size_t)a*nocc*nvir + (size_t)k*nvir + c];
            val += 0.5 * t2_kjbc * ovov_kaic;
            val += 2.0 * t2_jkbc * ovov_iakc;
            val -= t2_jkbc * ovov_kaic;
            val -= t2_kjbc * ovov_iakc;
        }
    d_ISR_corr[idx] = val;
}

// Σ_oo kernel
__global__ void ri_m11_sigma_oo_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_oo,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc) return;
    int i = idx / nocc, j = idx % nocc;
    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t v1 = 0.0, v2 = 0.0;
    for (int k = 0; k < nocc; k++)
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++) {
                real_t t2_ikab = d_t2[(size_t)i*nocc*nvir*nvir + (size_t)k*nvir*nvir + (size_t)a*nvir + b];
                real_t ja_kb = d_eri_ovov[(size_t)j*ovov_stride + (size_t)a*nocc*nvir + (size_t)k*nvir + b];
                real_t jb_ka = d_eri_ovov[(size_t)j*ovov_stride + (size_t)b*nocc*nvir + (size_t)k*nvir + a];
                v1 += t2_ikab * (ja_kb - 0.5 * jb_ka);
                real_t t2_jkab = d_t2[(size_t)j*nocc*nvir*nvir + (size_t)k*nvir*nvir + (size_t)a*nvir + b];
                real_t ia_kb = d_eri_ovov[(size_t)i*ovov_stride + (size_t)a*nocc*nvir + (size_t)k*nvir + b];
                real_t ib_ka = d_eri_ovov[(size_t)i*ovov_stride + (size_t)b*nocc*nvir + (size_t)k*nvir + a];
                v2 += t2_jkab * (ia_kb - 0.5 * ib_ka);
            }
    d_sigma_oo[idx] = v1 + v2;
}

// Σ_vv kernel
__global__ void ri_m11_sigma_vv_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_vv,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir) return;
    int a = idx / nvir, b = idx % nvir;
    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t v1 = 0.0, v2 = 0.0;
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            for (int c = 0; c < nvir; c++) {
                real_t t2_ijac = d_t2[(size_t)i*nocc*nvir*nvir + (size_t)j*nvir*nvir + (size_t)a*nvir + c];
                real_t ib_jc = d_eri_ovov[(size_t)i*ovov_stride + (size_t)b*nocc*nvir + (size_t)j*nvir + c];
                real_t jb_ic = d_eri_ovov[(size_t)j*ovov_stride + (size_t)b*nocc*nvir + (size_t)i*nvir + c];
                v1 += t2_ijac * (-ib_jc + 0.5 * jb_ic);
                real_t t2_ijbc = d_t2[(size_t)i*nocc*nvir*nvir + (size_t)j*nvir*nvir + (size_t)b*nvir + c];
                real_t ia_jc = d_eri_ovov[(size_t)i*ovov_stride + (size_t)a*nocc*nvir + (size_t)j*nvir + c];
                real_t ja_ic = d_eri_ovov[(size_t)j*ovov_stride + (size_t)a*nocc*nvir + (size_t)i*nvir + c];
                v2 += t2_ijbc * (-ia_jc + 0.5 * ja_ic);
            }
    d_sigma_vv[idx] = v1 + v2;
}

// ============================================================
//  Static: Build M11 entirely from RI 3-index integrals (GPU)
// ============================================================
void RIADC2SchurOperator::build_M11_from_RI(
    real_t* d_M11_out,
    const real_t* d_B_ia, const real_t* d_B_ab, const real_t* d_B_ij,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int naux)
{
    const int ov = nocc * nvir;
    const int vv = nvir * nvir;
    const int oo = nocc * nocc;
    const int dd = nocc * nocc * nvir * nvir;
    const size_t matrix_size = (size_t)ov * ov;
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double one = 1.0, zero = 0.0;
    const int threads = 256;

    // --- Step 1: Build OVOV = B_ia × B_ia^T [ov × ov] on device ---
    real_t* d_ovov = nullptr;
    tracked_cudaMalloc(&d_ovov, matrix_size * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                ov, ov, naux,
                &one, d_B_ia, ov, d_B_ia, ov,
                &zero, d_ovov, ov);

    // --- Step 2: Build OOVV = B_ij × B_ab^T [oo × vv] on device ---
    real_t* d_oovv = nullptr;
    tracked_cudaMalloc(&d_oovv, (size_t)oo * vv * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                oo, vv, naux,
                &one, d_B_ij, oo, d_B_ab, vv,
                &zero, d_oovv, oo);

    // --- Step 3: CIS matrix on device ---
    {
        int blocks = (matrix_size + threads - 1) / threads;
        ri_adc2_build_cis_kernel<<<blocks, threads>>>(
            d_ovov, d_oovv, d_orbital_energies, d_M11_out, nocc, nvir);
    }
    tracked_cudaFree(d_oovv);

    // --- Step 4: Convert OVOV to 4D layout for existing ISR/Sigma kernels ---
    // 4D: ovov_4d[i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b] = (ia|jb)
    real_t* d_ovov_4d = nullptr;
    tracked_cudaMalloc(&d_ovov_4d, matrix_size * sizeof(real_t));
    {
        int blocks = (matrix_size + threads - 1) / threads;
        ri_adc2_ovov_to_4d_kernel<<<blocks, threads>>>(d_ovov, d_ovov_4d, nocc, nvir);
    }
    tracked_cudaFree(d_ovov);

    // --- Step 5: Build T2 in 4D layout ---
    real_t* d_t2 = nullptr;
    tracked_cudaMalloc(&d_t2, (size_t)dd * sizeof(real_t));
    {
        // Rebuild ovov matrix temporarily for t2 kernel (uses [ov×ov] layout)
        real_t* d_ovov_mat = nullptr;
        tracked_cudaMalloc(&d_ovov_mat, matrix_size * sizeof(real_t));
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    ov, ov, naux,
                    &one, d_B_ia, ov, d_B_ia, ov,
                    &zero, d_ovov_mat, ov);
        int blocks = (dd + threads - 1) / threads;
        ri_adc2_build_t2_kernel<<<blocks, threads>>>(
            d_ovov_mat, d_orbital_energies, d_t2, nocc, nvir);
        tracked_cudaFree(d_ovov_mat);
    }

    // --- Step 6: ISR correction (reuse existing GPU kernel) ---
    real_t* d_ISR_corr = nullptr;
    tracked_cudaMalloc(&d_ISR_corr, matrix_size * sizeof(real_t));
    {
        int blocks = (matrix_size + threads - 1) / threads;
        ri_m11_ISR_kernel<<<blocks, threads>>>(
            d_t2, d_ovov_4d, d_ISR_corr, nocc, nvir);
        cudaDeviceSynchronize();
    }
    cublasDaxpy(handle, (int)matrix_size, &one, d_ISR_corr, 1, d_M11_out, 1);
    tracked_cudaFree(d_ISR_corr);

    // --- Step 7: Σ_oo (reuse existing GPU kernel) ---
    real_t* d_sigma_oo = nullptr;
    tracked_cudaMalloc(&d_sigma_oo, (size_t)oo * sizeof(real_t));
    {
        int blocks = (oo + threads - 1) / threads;
        ri_m11_sigma_oo_kernel<<<blocks, threads>>>(
            d_t2, d_ovov_4d, d_sigma_oo, nocc, nvir);
        cudaDeviceSynchronize();
    }

    // --- Step 8: Σ_vv (reuse existing GPU kernel) ---
    real_t* d_sigma_vv = nullptr;
    tracked_cudaMalloc(&d_sigma_vv, (size_t)vv * sizeof(real_t));
    {
        int blocks = (vv + threads - 1) / threads;
        ri_m11_sigma_vv_kernel<<<blocks, threads>>>(
            d_t2, d_ovov_4d, d_sigma_vv, nocc, nvir);
        cudaDeviceSynchronize();
    }

    // --- Step 9: Add Σ corrections to M11 ---
    {
        int blocks = (matrix_size + threads - 1) / threads;
        ri_adc2_add_self_energy_kernel<<<blocks, threads>>>(
            d_sigma_oo, d_sigma_vv, d_M11_out, nocc, nvir);
    }

    tracked_cudaFree(d_sigma_oo);
    tracked_cudaFree(d_sigma_vv);
    tracked_cudaFree(d_t2);
    tracked_cudaFree(d_ovov_4d);

    std::cout << "[RI-ADC(2)] M11 built from RI (GPU: CIS + ISR + Sigma_oo + Sigma_vv)" << std::endl;
}

} // namespace gansu
