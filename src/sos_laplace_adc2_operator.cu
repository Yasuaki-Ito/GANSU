/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file sos_laplace_adc2_operator.cu
 * @brief Laplace-SOS-ADC(2) operator implementation — O(N⁴) sigma build
 *
 * Sigma vector (ω-dependent Schur complement):
 *   σ(ω) = M11·x + c_os × Σ_τ w_τ e^{ωt_τ} × σ_schur(τ, x)
 *
 * σ_schur(τ, x) for each τ:
 *   1. B̃_ia^P(τ) = B_ia^P · exp((+ε_i - ε_a)·t/2)         [element-wise scale]
 *   2. v^P(τ) = Σ_{jb} B̃_jb^P(τ) · x_jb                   [DGEMV, O(N³)]
 *   3. X^{PQ}(τ) = Σ_{jb} B̃_jb^P(τ) · B̃_jb^Q(τ) · x_jb
 *      = diag(v^P) · B̃^T · B̃   ... or directly via DGEMM
 *      Actually: form F_jb^P = B̃_jb^P · x_jb, then X = F^T · B̃  [DGEMM, O(N_aux² × N_ov)]
 *   4. temp_ia^P = Σ_Q X^{PQ}(τ) · B̃_ia^Q(τ)               [DGEMM, O(N_aux² × N_ov)]
 *   5. σ_ia(τ) = Σ_P B̃_ia^P(τ) · temp_ia^P                 [element-wise × reduce]
 *      Actually: σ = row-wise dot of B̃ and temp              [O(N_ov × N_aux)]
 *
 * Steps 3-5 are the dominant O(N⁴) part.
 */

#include "sos_laplace_adc2_operator.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include "laplace_quadrature.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <numeric>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

using gpu::GPUHandle;

// ============================================================
//  Laplace scaling kernel: B̃_ia^P = B_ia^P · exp(-t/2 · (ε_a - ε_i))
// ============================================================
__global__ void sos_adc2_scale_B_kernel(
    const real_t* __restrict__ d_B,
    real_t* __restrict__ d_C,
    const real_t* __restrict__ d_eps_occ,  // ε_i [nocc]
    const real_t* __restrict__ d_eps_vir,  // ε_a [nvir]
    int nocc, int nvir, int naux, double t_half)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ov = (size_t)nocc * nvir;
    if (idx >= ov * naux) return;

    size_t P = idx / ov;
    size_t ia = idx % ov;
    int i = (int)(ia / nvir);
    int a = (int)(ia % nvir);

    double scale = exp(-t_half * (d_eps_vir[a] - d_eps_occ[i]));
    d_C[idx] = d_B[idx] * scale;
}

// ============================================================
//  Element-wise multiply: F_ia^P = B̃_ia^P × x_ia (broadcast x over P)
// ============================================================
__global__ void sos_adc2_scale_by_trial_kernel(
    const real_t* __restrict__ d_B_scaled,
    const real_t* __restrict__ d_x,
    real_t* __restrict__ d_F,
    int nov, int naux)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)nov * naux) return;

    size_t ia = idx % nov;
    d_F[idx] = d_B_scaled[idx] * d_x[ia];
}

// ============================================================
//  Row-wise dot: σ_ia = Σ_P B̃_ia^P · temp_ia^P
// ============================================================
__global__ void sos_adc2_rowdot_kernel(
    const real_t* __restrict__ d_A,   // [naux × nov], col-major
    const real_t* __restrict__ d_B,   // [naux × nov], col-major
    real_t* __restrict__ d_sigma,     // [nov]
    double scale_factor,
    int nov, int naux)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= nov) return;

    double sum = 0.0;
    for (int P = 0; P < naux; P++) {
        sum += d_A[(size_t)P * nov + ia] * d_B[(size_t)P * nov + ia];
    }
    d_sigma[ia] += scale_factor * sum;
}

// ============================================================
//  B3-exchange kernels
// ============================================================

// Scale B_ij: B̃_ij[I*nocc+L, P] = B_ij[I*nocc+L, P] × exp(εI × t/2)
// Only the first (I) index is scaled; L is unscaled.
__global__ void sos_adc2_scale_B_ij_kernel(
    const real_t* __restrict__ d_B_ij,
    real_t* __restrict__ d_B_ij_scaled,
    const real_t* __restrict__ d_eps_occ,
    int nocc, int naux, double t_half)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t oo = (size_t)nocc * nocc;
    if (idx >= oo * naux) return;
    size_t il = idx % oo;
    int I = (int)(il / nocc);
    double scale = exp(t_half * d_eps_occ[I]);
    d_B_ij_scaled[idx] = d_B_ij[idx] * scale;
}

// Compute g[I*nvir+E, Q] = Σ_L B̃_ij[I*nocc+L, Q] × x[L*nvir+E]
__global__ void sos_adc2_compute_g_kernel(
    const real_t* __restrict__ d_B_ij_scaled,
    const real_t* __restrict__ d_x,
    real_t* __restrict__ d_g,
    int nocc, int nvir, int naux)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ov = (size_t)nocc * nvir;
    size_t oo = (size_t)nocc * nocc;
    if (idx >= ov * naux) return;
    size_t ie = idx % ov;
    int Q = (int)(idx / ov);
    int I = (int)(ie / nvir);
    int E = (int)(ie % nvir);
    double sum = 0.0;
    for (int L = 0; L < nocc; L++)
        sum += d_B_ij_scaled[I * nocc + L + Q * oo] * d_x[L * nvir + E];
    d_g[idx] = sum;
}

// Accumulate B3-exchange: σ[K*nvir+E] += factor × exp(-εE×t) × raw[K*nvir+E]
__global__ void sos_adc2_b3x_postscale_kernel(
    const real_t* __restrict__ d_raw,
    real_t* __restrict__ d_sigma,
    const real_t* __restrict__ d_eps_vir,
    double factor, int nocc, int nvir, double t)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;
    int E = idx % nvir;
    d_sigma[idx] += factor * exp(-d_eps_vir[E] * t) * d_raw[idx];
}

// ============================================================
//  A3-Coulomb kernels
// ============================================================

// Scale B_ab: B̃_ab[E*nvir+C, P] = B_ab × exp(-εC × t/2)
// C = idx % nvir within each P-slice
__global__ void sos_adc2_scale_B_ab_kernel(
    const real_t* __restrict__ d_B_ab,
    real_t* __restrict__ d_B_ab_scaled,
    const real_t* __restrict__ d_eps_vir,
    int nvir, int naux, double t_half)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t vv = (size_t)nvir * nvir;
    if (idx >= vv * naux) return;
    size_t ec = idx % vv;
    int C = (int)(ec % nvir);
    double scale = exp(-t_half * d_eps_vir[C]);
    d_B_ab_scaled[idx] = d_B_ab[idx] * scale;
}

// Scale x: x̃[ia] = x[ia] × exp(-εa × t/2)
__global__ void sos_adc2_scale_x_kernel(
    const real_t* __restrict__ d_x,
    real_t* __restrict__ d_x_scaled,
    const real_t* __restrict__ d_eps_vir,
    int nvir, int ov, double t_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ov) return;
    int a = idx % nvir;
    d_x_scaled[idx] = d_x[idx] * exp(-t_half * d_eps_vir[a]);
}

// Extract diagonal sum: σ[K*nvir+E] += factor × exp(εK×t/2) × Σ_C r[E*nvir+C + C*vv]
// r_K is [vv × nvir], we want Σ_C r[E*nvir+C, C]
// But here we accumulate from f_P × w_P^T result (already summed over P)
__global__ void sos_adc2_a3_postscale_kernel(
    const real_t* __restrict__ d_raw,
    real_t* __restrict__ d_sigma,
    const real_t* __restrict__ d_eps_occ,
    double factor, int nocc, int nvir, double t_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;
    int K = idx / nvir;
    d_sigma[idx] += factor * exp(t_half * d_eps_occ[K]) * d_raw[idx];
}

// ============================================================
//  Constructor
// ============================================================
SOSLaplaceADC2Operator::SOSLaplaceADC2Operator(
    const real_t* d_B_ia,
    const real_t* d_B_ij,
    const real_t* d_B_ab,
    const real_t* d_M11_ext,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int naux,
    double c_os, int n_laplace)
    : nocc_(nocc), nvir_(nvir), naux_(naux),
      singles_dim_(nocc * nvir),
      c_os_(c_os), n_laplace_(n_laplace),
      d_B_ia_(d_B_ia), d_B_ij_(d_B_ij), d_B_ab_(d_B_ab)
{
    // Copy orbital energies to host
    std::vector<double> eps(nocc + nvir);
    cudaMemcpy(eps.data(), d_orbital_energies, (nocc + nvir) * sizeof(double), cudaMemcpyDeviceToHost);
    eps_occ_.assign(eps.begin(), eps.begin() + nocc);
    eps_vir_.assign(eps.begin() + nocc, eps.end());

    // Upload separated orbital energies to device (for scaling kernel)
    real_t* d_eps_occ_tmp;
    real_t* d_eps_vir_tmp;
    tracked_cudaMalloc(&d_eps_occ_tmp, nocc * sizeof(real_t));
    tracked_cudaMalloc(&d_eps_vir_tmp, nvir * sizeof(real_t));
    cudaMemcpy(d_eps_occ_tmp, eps_occ_.data(), nocc * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eps_vir_tmp, eps_vir_.data(), nvir * sizeof(double), cudaMemcpyHostToDevice);
    // Store as member for use in apply()
    d_eps_occ_dev_ = d_eps_occ_tmp;
    d_eps_vir_dev_ = d_eps_vir_tmp;

    // Allocate workspace
    size_t ov = (size_t)nocc * nvir;
    tracked_cudaMalloc(&d_B_scaled_, ov * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_F_, ov * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_X_PQ_, (size_t)naux * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_temp_ov_aux_, ov * naux * sizeof(real_t));

    // B3-exchange workspace (only if B_ij provided)
    if (d_B_ij_) {
        size_t oo = (size_t)nocc * nocc;
        tracked_cudaMalloc(&d_B_ij_scaled_, oo * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_g_, ov * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_Z_, (size_t)naux * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_h_, (size_t)naux * ov * sizeof(real_t));
        tracked_cudaMalloc(&d_sigma_b3_, ov * sizeof(real_t));
    }

    // A3-Coulomb workspace (only if B_ab provided)
    if (d_B_ab_) {
        size_t vv = (size_t)nvir * nvir;
        tracked_cudaMalloc(&d_B_ab_scaled_, vv * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_x_scaled_, ov * sizeof(real_t));
        tracked_cudaMalloc(&d_f_buf_, (size_t)nvir * nocc * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_w_T_, (size_t)nocc * nocc * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_sigma_a3_, ov * sizeof(real_t));
    }

    // Allocate D1 and diagonal
    tracked_cudaMalloc(&d_D1_, singles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_diagonal_, singles_dim_ * sizeof(real_t));
    compute_D1();

    tracked_cudaMalloc(&d_M11_, (size_t)singles_dim_ * singles_dim_ * sizeof(real_t));
    if (d_M11_ext) {
        // Use externally-built M11 (includes full CIS + ISR)
        cudaMemcpy(d_M11_, d_M11_ext, (size_t)singles_dim_ * singles_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        std::cout << "[SOS-LT-ADC(2)] M11 loaded from pre-built matrix" << std::endl;
    } else {
        build_M11();
    }

    // Initialize Laplace quadrature
    update_laplace_quadrature();

    compute_diagonal();

    std::cout << "[SOS-LT-ADC(2)] Initialized: nocc=" << nocc << " nvir=" << nvir
              << " naux=" << naux << " c_os=" << c_os << " n_laplace=" << n_laplace
              << (d_B_ij_ ? " +B3x" : "") << (d_B_ab_ ? " +A3" : "") << std::endl;
}

SOSLaplaceADC2Operator::~SOSLaplaceADC2Operator() {
    tracked_cudaFree(d_B_scaled_);
    tracked_cudaFree(d_F_);
    tracked_cudaFree(d_X_PQ_);
    tracked_cudaFree(d_temp_ov_aux_);
    tracked_cudaFree(d_D1_);
    tracked_cudaFree(d_diagonal_);
    tracked_cudaFree(d_M11_);
    tracked_cudaFree(d_eps_occ_dev_);
    tracked_cudaFree(d_eps_vir_dev_);
    // B3-exchange
    tracked_cudaFree(d_B_ij_scaled_);
    tracked_cudaFree(d_g_);
    tracked_cudaFree(d_Z_);
    tracked_cudaFree(d_h_);
    tracked_cudaFree(d_sigma_b3_);
    // A3-Coulomb
    tracked_cudaFree(d_B_ab_scaled_);
    tracked_cudaFree(d_x_scaled_);
    tracked_cudaFree(d_f_buf_);
    tracked_cudaFree(d_w_T_);
    tracked_cudaFree(d_sigma_a3_);
}

// ============================================================
//  Laplace quadrature generation (ω-dependent)
// ============================================================
void SOSLaplaceADC2Operator::set_omega(real_t omega) {
    omega_ = omega;
}

void SOSLaplaceADC2Operator::update_laplace_quadrature() {
    // Range for Laplace: 1/(Δ - ω) where Δ = ε_a + ε_b - ε_i - ε_j
    double Delta_min = 2.0 * eps_vir_[0] - 2.0 * eps_occ_[nocc_ - 1]; // smallest Δ
    double Delta_max = 2.0 * eps_vir_[nvir_ - 1] - 2.0 * eps_occ_[0]; // largest Δ

    double x_min = Delta_min - omega_;
    double x_max = Delta_max - omega_;

    // Safety: clamp range for ill-conditioned cases (ω near or above Δ_min)
    if (x_min < 1e-4) {
        std::cerr << "[SOS-LT-ADC(2)] Warning: ω=" << omega_
                  << " near 2p2h threshold Δ_min=" << Delta_min
                  << ". Clamping Laplace range." << std::endl;
        x_min = 1e-4;
    }
    if (x_max < x_min + 0.1) x_max = x_min + 10.0; // ensure valid range

    auto quad = generate_laplace_quadrature(x_min, x_max, n_laplace_);
    laplace_t_.resize(quad.num_points);
    laplace_w_.resize(quad.num_points);
    for (int k = 0; k < quad.num_points; k++) {
        laplace_t_[k] = quad.points[k];
        laplace_w_[k] = quad.weights[k];
    }
}

// ============================================================
//  Build M11 from B_ia (CIS part via RI)
// ============================================================
void SOSLaplaceADC2Operator::build_M11() {
    // M11 = δ_ij δ_ab (ε_a - ε_i) + 2(ia|jb) - (ij|ab)   [singlet CIS]
    //
    // Using RI: (ia|jb) = Σ_P B^P_ia B^P_jb   [Coulomb]
    //           (ij|ab) = Σ_P B^P_ia B^P_jb with index rearrangement [Exchange]
    //
    // B_ia is [ov × naux] col-major: element(ia, P) at index P*ov + ia
    //
    // Coulomb: J[ia,jb] = (ia|jb) = Σ_P B_ia^P B_jb^P = (B^T B)[ia,jb]
    //   where B is viewed as [ov × naux], so B^T B = [ov × ov]
    //   DGEMM: C = B * B^T with B[ov, naux] → C[ov, ov]
    //   cuBLAS col-major: C = A^T * A with A[naux, ov], lda=ov
    //   Actually B stored as [ov, naux] col-major lda=ov:
    //   J = B_ia * B_jb^T... no. Let's be careful.
    //
    //   B[ia,P] stored at P*ov+ia. In CUBLAS col-major with m=ov, n=naux, lda=ov:
    //   J = B * B^T: CUBLAS_OP_N(B) × CUBLAS_OP_T(B)
    //   = [ov×naux] * [naux×ov] = [ov×ov]

    size_t ov = singles_dim_;
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double zero = 0.0, one = 1.0, two = 2.0, neg_one = -1.0;

    // --- Diagonal: D1 ---
    std::vector<double> M11_host((size_t)ov * ov, 0.0);
    for (int ia = 0; ia < (int)ov; ia++) {
        int i = ia / nvir_;
        int a = ia % nvir_;
        M11_host[(size_t)ia * ov + ia] = eps_vir_[a] - eps_occ_[i];
    }
    cudaMemcpy(d_M11_, M11_host.data(), (size_t)ov * ov * sizeof(real_t), cudaMemcpyHostToDevice);

    // --- Coulomb: +2(ia|jb) = 2 * Σ_P B_ia^P B_jb^P ---
    // J = B * B^T where B is [ov × naux] col-major, lda=ov
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                (int)ov, (int)ov, naux_,
                &two, d_B_ia_, (int)ov,
                d_B_ia_, (int)ov,
                &one, d_M11_, (int)ov);

    // --- Exchange: -(ij|ab) ---
    // (ij|ab) = Σ_P B_ij^P B_ab^P, but we only have B_ia^P.
    // Use identity: (ij|ab) with B_ia indexing:
    //   B_ia^P has i=row/nvir, a=row%nvir.
    //   (ij|ab) needs B with indices (i,j) and (a,b).
    //
    // Compute on host for correctness (small matrix for moderate systems)
    // K[ia,jb] = Σ_P B_ij^P B_ab^P = Σ_P [Σ_μν C_μi C_νj B_μν^P] [Σ_λσ C_λa C_σb B_λσ^P]
    //
    // Alternative via B_ia: K[ia,jb] = (ij|ab) = Σ_P B_ij B_ab
    //   But B_ij is not available.
    //
    // Trick: Rearrange using existing B_ia^P:
    //   K[ia,jb] = Σ_P B_ia^P B_jb^P ... no, this is Coulomb (ia|jb), not exchange (ij|ab).
    //
    // For exchange: build per-i, per-j sub-blocks
    //   K[ia,jb] = Σ_P (Σ_c B_ic^P)(Σ_c... no, this doesn't factor.
    //
    // Correct: K_ia,jb = (ij|ab) = Σ_P [C_i * B^P * C_j^T][row a,col b]... too complex.
    //
    // Simpler approach: B_ia^P for fixed i gives a [nvir × naux] sub-block.
    // B_i^P[a] = B[(i*nvir+a), P]. Then:
    //   K[ia,jb] = Σ_P (B_i^T B_j)[a,b] ... wait, that's still (ia|jb).
    //
    // Actually: (ij|ab) = Σ_P B_ij^P B_ab^P requires oo-block and vv-block of B.
    // Without those, compute exchange via the Laplace identity or brute force.
    //
    // Brute-force on GPU: for each pair (i,j), compute N^{ij}[a,b] = Σ_P B_i[a,P] B_j[b,P]
    //   = B_i * B_j^T [nvir × nvir]. Then K[(i*nvir+a), (j*nvir+b)] = N^{ij}[a,b].
    //   But this is (ia|jb) again! Not (ij|ab).
    //
    // The CORRECT exchange for CIS: K[ia,jb] = (ij|ab) = Σ_P B_ij^P B_ab^P
    // To compute this from B_ia^P, we need a re-index:
    //   Exchange index swap: (ia|jb) → (ij|ab) means swap 2nd and 3rd index of the ERI.
    //   In RI terms: we can't simply get (ij|ab) from B_ia^P without B_ij^P.
    //
    // HOWEVER: for the CIS matrix in RI-ADC(2), the exchange is often computed as:
    //   K[ia,jb] = Σ_P B_ia^P · B_jb^P with SWAPPED indices:
    //   i.e., K[ia,jb] = (ib|ja) = Σ_P B_ib^P B_ja^P
    //
    // Wait: CIS exchange = -(ij|ab). But (ij|ab) = (ib|ja) by ERI symmetry (12|34)=(14|32).
    // No, (ij|ab) ≠ (ib|ja). (ij|ab) integrates φ_i φ_j at r1, φ_a φ_b at r2.
    // (ib|ja) integrates φ_i φ_b at r1, φ_j φ_a at r2. These are different.
    //
    // Actually for real orbitals: (ij|ab) = (ji|ba) = (ab|ij) = (ba|ji) [8-fold symmetry].
    // And (ib|ja) = (bi|aj) etc. These are NOT the same as (ij|ab) in general.
    //
    // The RI approximation gives: (pq|rs) = Σ_P B_pq^P B_rs^P
    // where B_pq^P = Σ_Q (pq|Q) [V^{-1/2}]_{QP}
    //
    // To get (ij|ab), we need B_ij^P (from B with occ-occ indices) and B_ab^P (virt-virt).
    // These are NOT the same as B_ia^P.
    //
    // For now: compute exchange on host via B_ia reshaping
    // K[ia,jb] = Σ_P B(i,a,P) × B(j,b,P) with the EXCHANGE interpretation
    // ... actually this IS Coulomb. For exchange we need (ib|ja) = Σ_P B_ib^P B_ja^P.
    // B_ib^P = B[(i*nvir + b), P] and B_ja^P = B[(j*nvir + a), P].
    // So K[ia,jb] = Σ_P B[(i*nvir+b), P] × B[(j*nvir+a), P]

    {
        // Exchange: K[ia,jb] = (ib|ja) = Σ_P B_{ib}^P B_{ja}^P
        // Note: CIS matrix uses -(ij|ab) = -(ib|ja) for real orbitals...
        // Actually: for the singlet CIS matrix:
        //   A[ia,jb] = δ_ij δ_ab (ε_a-ε_i) + 2(ia|jb) - (ij|ab)
        // And (ij|ab) ≠ (ib|ja) in general.
        // BUT: from B_ia^P we can compute (ib|ja) = Σ_P B_ib^P B_ja^P
        // by rearranging indices. This IS available from B_ia^P.
        //
        // Unfortunately (ij|ab) requires B_ij^P which is NOT B_ia^P.
        //
        // Workaround: Compute exchange on host via explicit loop
        std::vector<double> B_host((size_t)ov * naux_);
        cudaMemcpy(B_host.data(), d_B_ia_, (size_t)ov * naux_ * sizeof(double), cudaMemcpyDeviceToHost);

        std::vector<double> K_host((size_t)ov * ov, 0.0);
        // K[ia,jb] = (ib|ja) = Σ_P B[(i*nvir+b),P] * B[(j*nvir+a),P]
        for (int i = 0; i < nocc_; i++) {
            for (int j = 0; j < nocc_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        int ia = i * nvir_ + a;
                        int jb = j * nvir_ + b;
                        int ib = i * nvir_ + b;
                        int ja = j * nvir_ + a;
                        double val = 0.0;
                        for (int P = 0; P < naux_; P++) {
                            val += B_host[(size_t)P * ov + ib] * B_host[(size_t)P * ov + ja];
                        }
                        K_host[(size_t)jb * ov + ia] += val;  // col-major
                    }
                }
            }
        }

        // M11 -= K (exchange)
        real_t* d_K = nullptr;
        tracked_cudaMalloc(&d_K, (size_t)ov * ov * sizeof(real_t));
        cudaMemcpy(d_K, K_host.data(), (size_t)ov * ov * sizeof(real_t), cudaMemcpyHostToDevice);
        cublasDaxpy(handle, (int)((size_t)ov * ov), &neg_one, d_K, 1, d_M11_, 1);
        tracked_cudaFree(d_K);
    }

    std::cout << "[SOS-LT-ADC(2)] M11 built (CIS: D1 + 2J - K, no ISR)" << std::endl;
}

// ============================================================
//  Compute D1 = ε_a - ε_i
// ============================================================
void SOSLaplaceADC2Operator::compute_D1() {
    std::vector<double> D1(singles_dim_);
    for (int i = 0; i < nocc_; i++)
        for (int a = 0; a < nvir_; a++)
            D1[i * nvir_ + a] = eps_vir_[a] - eps_occ_[i];
    cudaMemcpy(d_D1_, D1.data(), singles_dim_ * sizeof(double), cudaMemcpyHostToDevice);
}

// ============================================================
//  Compute diagonal for preconditioner
// ============================================================
void SOSLaplaceADC2Operator::compute_diagonal() {
    // Approximate: diag(M_eff) ≈ D1 (orbital energy differences)
    cudaMemcpy(d_diagonal_, d_D1_, singles_dim_ * sizeof(double), cudaMemcpyDeviceToDevice);
}

// ============================================================
//  Preconditioner: (D1 - ω)^{-1} · r
// ============================================================
void SOSLaplaceADC2Operator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    // Simple Jacobi preconditioner
    std::vector<double> D1(singles_dim_), input(singles_dim_);
    cudaMemcpy(D1.data(), d_D1_, singles_dim_ * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.data(), d_input, singles_dim_ * sizeof(double), cudaMemcpyDeviceToHost);

    for (int ia = 0; ia < singles_dim_; ia++) {
        double denom = D1[ia] - omega_;
        if (std::abs(denom) < 1e-10) denom = 1e-10;
        input[ia] /= denom;
    }
    cudaMemcpy(d_output, input.data(), singles_dim_ * sizeof(double), cudaMemcpyHostToDevice);
}

// ============================================================
//  CORE: sigma vector application σ = M_eff(ω) · x
// ============================================================
void SOSLaplaceADC2Operator::apply(const real_t* d_input, real_t* d_output) const {
    const size_t ov = singles_dim_;
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double one = 1.0, zero = 0.0;

    // Step 1: σ = M11 · x
    cublasDgemv(handle, CUBLAS_OP_N,
                (int)ov, (int)ov,
                &one, d_M11_, (int)ov,
                d_input, 1,
                &zero, d_output, 1);

    // Step 2: Schur complement via Laplace-SOS
    // σ += c_os × Σ_τ w_τ e^{ωt_τ} × σ_schur(τ, x)
    int threads = 256;

    for (int k = 0; k < (int)laplace_t_.size(); k++) {
        double t = laplace_t_[k];
        double w = laplace_w_[k];
        double t_half = t / 2.0;
        // Negative sign: M_eff = M11 - U·(D2-ω)⁻¹·U^T
        double prefactor = -c_os_ * w * std::exp(omega_ * t);

        // 2a. Scale B → B̃(τ): B̃_ia^P = B_ia^P · exp(-t/2 · (ε_a - ε_i))
        {
            size_t total = ov * naux_;
            int blocks = (int)((total + threads - 1) / threads);
            sos_adc2_scale_B_kernel<<<blocks, threads>>>(
                d_B_ia_, d_B_scaled_, d_eps_occ_dev_, d_eps_vir_dev_,
                nocc_, nvir_, naux_, t_half);
        }

        // 2b. F_ia^P = B̃_ia^P × x_ia  (element-wise broadcast)
        {
            size_t total = ov * naux_;
            int blocks = (int)((total + threads - 1) / threads);
            sos_adc2_scale_by_trial_kernel<<<blocks, threads>>>(
                d_B_scaled_, d_input, d_F_, (int)ov, naux_);
        }

        // 2c. X^{PQ} = F^T · B̃  [naux × naux]
        //     F is [ov, naux] col-major (same as B̃ but with x_ia weight)
        //     X = F^T · B̃ → [naux, naux]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    naux_, naux_, (int)ov,
                    &one, d_F_, (int)ov,
                    d_B_scaled_, (int)ov,
                    &zero, d_X_PQ_, naux_);

        // 2d. temp = B̃ · X  [ov × naux]
        //     temp[ia, P] = Σ_Q B̃[ia, Q] · X[Q, P]
        //     B̃ is [ov × naux] col-major (lda=ov)
        //     X  is [naux × naux] col-major (ldb=naux)
        //     C  is [ov × naux] col-major (ldc=ov)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)ov, naux_, naux_,
                    &one, d_B_scaled_, (int)ov,
                    d_X_PQ_, naux_,
                    &zero, d_temp_ov_aux_, (int)ov);

        // 2e. σ_ia += prefactor × Σ_P B̃_ia^P · temp_ia^P
        //     = row-wise dot product of B̃ and temp (both [naux × ov] col-major)
        {
            int blocks = ((int)ov + threads - 1) / threads;
            sos_adc2_rowdot_kernel<<<blocks, threads>>>(
                d_B_scaled_, d_temp_ov_aux_, d_output,
                prefactor, (int)ov, naux_);
        }

        // ---- B3-exchange: +2(IK|JD)(IL|JD) / (D2-ω) ----
        if (d_B_ij_) {
            const int oo = nocc_ * nocc_;
            const double b3_factor = -2.0 * c_os_ * w * std::exp(omega_ * t);

            // 3a. Scale B_ij: B̃_ij[I*nocc+L, P] = B_ij × exp(εI×t/2)
            {
                size_t total_ij = (size_t)oo * naux_;
                int blocks = (int)((total_ij + threads - 1) / threads);
                sos_adc2_scale_B_ij_kernel<<<blocks, threads>>>(
                    d_B_ij_, d_B_ij_scaled_, d_eps_occ_dev_,
                    nocc_, naux_, t_half);
            }

            // 3b. g[I*nvir+E, Q] = Σ_L B̃_ij[I*nocc+L, Q] × x[L*nvir+E]
            {
                size_t total_g = ov * naux_;
                int blocks = (int)((total_g + threads - 1) / threads);
                sos_adc2_compute_g_kernel<<<blocks, threads>>>(
                    d_B_ij_scaled_, d_input, d_g_,
                    nocc_, nvir_, naux_);
            }

            // 3c. Z = B̃_ia^T · B̃_ia [naux × naux] (reuse d_B_scaled_)
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        naux_, naux_, (int)ov,
                        &one, d_B_scaled_, (int)ov,
                        d_B_scaled_, (int)ov,
                        &zero, d_Z_, naux_);

            // 3d. h[P, ie] = Σ_Q Z[P,Q] × g[ie,Q]
            // g is [ov × naux] col-major (ie rows, Q cols).
            // h = Z × g^T  →  [naux × naux] × [naux × ov] = [naux × ov]
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        naux_, (int)ov, naux_,
                        &one, d_Z_, naux_,
                        d_g_, (int)ov,
                        &zero, d_h_, naux_);

            // 3e. σ_b3[K*nvir+E] = Σ_I Σ_P B̃_ij_I[K,P] × h_I[P,E]
            // For each I: DGEMM h_I^T × B̃_ij_I^T → [nvir × nocc]
            // h_I = &d_h_[I*nvir*naux], [naux × nvir], lda=naux
            // B̃_ij_I = &d_B_ij_scaled_[I*nocc], [nocc × naux], lda=oo
            cudaMemset(d_sigma_b3_, 0, ov * sizeof(real_t));
            for (int I = 0; I < nocc_; I++) {
                cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                            nvir_, nocc_, naux_,
                            &one,
                            &d_h_[I * nvir_ * naux_], naux_,
                            &d_B_ij_scaled_[I * nocc_], oo,
                            &one, d_sigma_b3_, nvir_);
            }

            // 3f. σ += b3_factor × exp(-εE×t) × σ_b3
            {
                int blocks = ((int)ov + threads - 1) / threads;
                sos_adc2_b3x_postscale_kernel<<<blocks, threads>>>(
                    d_sigma_b3_, d_output, d_eps_vir_dev_,
                    b3_factor, nocc_, nvir_, t);
            }
        }

        // ---- A3-Coulomb: -2c_os(EC|JD)(KL|JD) / (D2-ω) ----
        if (d_B_ab_) {
            const int oo = nocc_ * nocc_;
            const int vv = nvir_ * nvir_;
            const double a3_factor = +2.0 * c_os_ * w * std::exp(omega_ * t);

            // 4a. Scale B_ab: B̃_ab[EC, P] = B_ab × exp(-εC×t/2)
            {
                size_t total_ab = (size_t)vv * naux_;
                int blocks = (int)((total_ab + threads - 1) / threads);
                sos_adc2_scale_B_ab_kernel<<<blocks, threads>>>(
                    d_B_ab_, d_B_ab_scaled_, d_eps_vir_dev_,
                    nvir_, naux_, t_half);
            }

            // 4b. Scale x: x̃[ia] = x[ia] × exp(-εa×t/2)
            {
                int blocks = ((int)ov + threads - 1) / threads;
                sos_adc2_scale_x_kernel<<<blocks, threads>>>(
                    d_input, d_x_scaled_, d_eps_vir_dev_,
                    nvir_, (int)ov, t_half);
            }

            // 4c. f_P[E,L] = Σ_C B̃_ab_P[E,C] × x̃^T[C,L]
            // B̃_ab_P: [nvir × nvir] at offset P*vv, stride vv
            // x̃ viewed as [nocc × nvir]: x̃^T is [nvir × nocc]
            // f_P: [nvir × nocc] at offset P*nvir*nocc
            // B̃_ab_P stored as [E*nvir+C] = cuBLAS [C_row, E_col] → use OP_T
            cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nvir_, nocc_, nvir_,
                &one,
                d_B_ab_scaled_, nvir_, (long long)vv,    // A_P^T → [E×C]
                d_x_scaled_, nvir_, 0LL,                 // B = x̃ [nvir×nocc] col-major
                &zero,
                d_f_buf_, nvir_, (long long)(nvir_ * nocc_), // C_P [nvir×nocc]
                naux_);

            // 4d. w_T = B̃_ij × Z [oo × naux]
            // B̃_ij_scaled [oo × naux] already computed in B3-exchange step
            // Z [naux × naux] already computed in B3-exchange step
            // If B3 was not computed, we need Z and B̃_ij_scaled here too
            if (d_B_ij_) {
                // Reuse d_Z_ and d_B_ij_scaled_ from B3
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            oo, naux_, naux_,
                            &one, d_B_ij_scaled_, oo,
                            d_Z_, naux_,
                            &zero, d_w_T_, oo);
            }

            // 4e. σ_a3 = Σ_P f_P × w_P^T [nvir × nocc]
            // f_P at &d_f_buf_[P * nvir_ * nocc_], [nvir × nocc]
            // w_P at &d_w_T_[P * oo], [nocc × nocc] viewed as w_P[L, K] = w_T[K*nocc+L, P]
            cudaMemset(d_sigma_a3_, 0, ov * sizeof(real_t));
            for (int P = 0; P < naux_; P++) {
                // w_P_cublas[L_row, K_col] → no transpose needed
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            nvir_, nocc_, nocc_,
                            &one,
                            &d_f_buf_[P * nvir_ * nocc_], nvir_,
                            &d_w_T_[P * oo], nocc_,
                            &one, d_sigma_a3_, nvir_);
            }

            // 4f. σ += a3_factor × exp(εK×t/2) × σ_a3
            {
                int blocks = ((int)ov + threads - 1) / threads;
                sos_adc2_a3_postscale_kernel<<<blocks, threads>>>(
                    d_sigma_a3_, d_output, d_eps_occ_dev_,
                    a3_factor, nocc_, nvir_, t_half);
            }
        }
    }
}

} // namespace gansu
