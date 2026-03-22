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
 * @file adc2_operator.cu
 * @brief GPU implementation of ADC(2) operator via ω-dependent Schur complement
 *
 * All formulas verified numerically against PySCF EE-ADC(2).
 *
 * M_eff(ω) = M11 + M12 · diag(1/(ω - D2)) · M21
 *
 * M11 = CIS + ISR_corr - δ_ab × Σ_oo[i,j] + δ_ij × Σ_vv[a,b]
 *   Singlet ISR_corr = (2P1 - P2 - P5 + 0.5P6) + transpose
 *   Triplet ISR_corr = (0.5P6) + transpose = 0.5*(P6 + P6^T)
 *   where:
 *     P1[ia,jb] = Σ_{kc} t2[i,k,a,c]·(jb|kc)
 *     P2[ia,jb] = Σ_{kc} t2[i,k,a,c]·(kb|jc)
 *     P5[ia,jb] = Σ_{kc} t2[k,i,a,c]·(jb|kc)  (t2 occ-swapped)
 *     P6[ia,jb] = Σ_{kc} t2[k,i,a,c]·(kb|jc)  (t2 occ-swapped)
 *   Σ_oo[i,j] = Σ_{k,a,b} t2[i,k,a,b]·(-(ja|kb) + 0.5·(jb|ka))  (symmetrized)
 *   Σ_vv[a,b] = Σ_{i,j,c} t2[i,j,a,c]·(-(ib|jc) + 0.5·(jb|ic))  (symmetrized)
 *
 * M12[KE, IJCD] = δ_{I,K}·[2·(EC|JD) - (DE|JC)]
 *               + δ_{C,E}·[(JK|ID) - 2·(IK|JD)]
 *   where (EC|JD) = eri_vvov[E,C,J,D], (IK|JD) = eri_ooov[I,K,J,D]
 *
 * M21[IJCD, KE] = δ_{K,I}·(EC|JD) + δ_{K,J}·(ED|IC)
 *               - δ_{E,C}·(IK|JD) - δ_{E,D}·(JK|IC)
 *
 * D2[IJCD] = eps_C + eps_D - eps_I - eps_J
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "adc2_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// ========================================================================
//  CUDA kernels — ERI block extraction
// ========================================================================

/**
 * @brief Extract (ia|jb) from full MO ERI tensor
 * eri_ovov[i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]
 */
__global__ void adc2_extract_eri_ovov_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t* __restrict__ d_eri_ovov,
    int nocc, int nvir, int nao)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nocc * nvir * nocc * nvir;
    if (idx >= total) return;

    int i = idx / (nvir * nocc * nvir);
    int rem = idx % (nvir * nocc * nvir);
    int a = rem / (nocc * nvir);
    rem = rem % (nocc * nvir);
    int j = rem / nvir;
    int b = rem % nvir;

    size_t nao2 = (size_t)nao * nao;
    int a_abs = a + nocc;
    int b_abs = b + nocc;
    d_eri_ovov[idx] = d_eri_mo[((size_t)i * nao + a_abs) * nao2 + (size_t)j * nao + b_abs];
}

/**
 * @brief Extract (ab|ic) from full MO ERI tensor
 * eri_vvov[a*nvir*nocc*nvir + b*nocc*nvir + i*nvir + c]
 */
__global__ void adc2_extract_eri_vvov_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t* __restrict__ d_eri_vvov,
    int nocc, int nvir, int nao)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nvir * nvir * nocc * nvir;
    if (idx >= total) return;

    int a = idx / (nvir * nocc * nvir);
    int rem = idx % (nvir * nocc * nvir);
    int b = rem / (nocc * nvir);
    rem = rem % (nocc * nvir);
    int i = rem / nvir;
    int c = rem % nvir;

    size_t nao2 = (size_t)nao * nao;
    int a_abs = a + nocc;
    int b_abs = b + nocc;
    int c_abs = c + nocc;
    d_eri_vvov[idx] = d_eri_mo[((size_t)a_abs * nao + b_abs) * nao2 + (size_t)i * nao + c_abs];
}

/**
 * @brief Extract (ji|kb) from full MO ERI tensor
 * eri_ooov[j*nocc*nocc*nvir + i*nocc*nvir + k*nvir + b]
 */
__global__ void adc2_extract_eri_ooov_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t* __restrict__ d_eri_ooov,
    int nocc, int nvir, int nao)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nocc * nocc * nocc * nvir;
    if (idx >= total) return;

    int j = idx / (nocc * nocc * nvir);
    int rem = idx % (nocc * nocc * nvir);
    int i = rem / (nocc * nvir);
    rem = rem % (nocc * nvir);
    int k = rem / nvir;
    int b = rem % nvir;

    size_t nao2 = (size_t)nao * nao;
    int b_abs = b + nocc;
    d_eri_ooov[idx] = d_eri_mo[((size_t)j * nao + i) * nao2 + (size_t)k * nao + b_abs];
}

// ========================================================================
//  CUDA kernels — T2 amplitudes and denominators
// ========================================================================

/**
 * @brief Compute MP1 T2 amplitudes and D2 denominators
 * t2[i,j,a,b] = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)
 * D2[i,j,a,b] = eps_a + eps_b - eps_i - eps_j
 */
__global__ void adc2_compute_mp1_t2_and_D2_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_t2,
    real_t* __restrict__ d_D2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nocc * nocc * nvir * nvir;
    if (idx >= total) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem = rem % (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    real_t eps_i = d_orbital_energies[i];
    real_t eps_j = d_orbital_energies[j];
    real_t eps_a = d_orbital_energies[a + nocc];
    real_t eps_b = d_orbital_energies[b + nocc];

    real_t denom = eps_i + eps_j - eps_a - eps_b;
    real_t D2_val = eps_a + eps_b - eps_i - eps_j;

    // (ia|jb) from eri_ovov
    real_t ia_jb = d_eri_ovov[(size_t)i * nvir * nocc * nvir +
                              (size_t)a * nocc * nvir +
                              (size_t)j * nvir + b];

    d_t2[idx] = ia_jb / denom;
    d_D2[idx] = D2_val;
}

/**
 * @brief Compute D1: D1[i*nvir+a] = eps_a - eps_i
 */
__global__ void adc2_compute_D1_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_D1,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    d_D1[idx] = d_orbital_energies[a + nocc] - d_orbital_energies[i];
}

// ========================================================================
//  CUDA kernels — M11 construction
// ========================================================================

/**
 * @brief Build CIS A-matrix (column-major)
 * Singlet: A[ia,jb] = δ_ij·δ_ab·(eps_a - eps_i) + 2(ia|jb) - (ij|ab)
 * Triplet: A[ia,jb] = δ_ij·δ_ab·(eps_a - eps_i) - (ij|ab)
 */
__global__ void adc2_build_cis_matrix_kernel(
    const real_t* __restrict__ d_eri_mo,
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_cis,
    int nocc, int nvir, int nao,
    bool is_triplet)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    // Column-major: idx = row + col * ov
    int ia_idx = idx % ov;
    int jb_idx = idx / ov;

    int i = ia_idx / nvir;
    int a = ia_idx % nvir;
    int j = jb_idx / nvir;
    int b = jb_idx % nvir;

    int a_abs = a + nocc;
    int b_abs = b + nocc;
    size_t nao2 = (size_t)nao * nao;

    real_t val = 0.0;

    // Diagonal: δ_ij·δ_ab·(eps_a - eps_i)
    if (i == j && a == b) {
        val += d_orbital_energies[a_abs] - d_orbital_energies[i];
    }

    // -(ij|ab)
    real_t ij_ab = d_eri_mo[((size_t)i * nao + j) * nao2 + (size_t)a_abs * nao + b_abs];
    val -= ij_ab;

    if (!is_triplet) {
        // Singlet: +2(ia|jb)
        real_t ia_jb = d_eri_mo[((size_t)i * nao + a_abs) * nao2 + (size_t)j * nao + b_abs];
        val += 2.0 * ia_jb;
    }

    d_cis[idx] = val;
}

/**
 * @brief Build ISR-ADC(2) full-block correction (column-major)
 *
 * Singlet ISR: (2P1 - P2 - P5 + 0.5P6) + transpose
 * Triplet ISR: (0P1 + 0P2 + 0P5 + 0.5P6) + transpose = 0.5*(P6 + P6^T)
 *
 * Derived by spin-tracing the ADC(2) ISR coupling term (Term C):
 *   s1[i,a] += <jk||bc> * t2[ik,ac] * r[j,b]
 * For singlet (r_β = +r_α): both αα and ββ channels contribute with same sign.
 * For triplet (r_β = -r_α): ββ channel has opposite sign, canceling P1/P2/P5.
 */
__global__ void adc2_build_M11_ISR_correction_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_ISR_corr,
    int nocc, int nvir,
    bool is_triplet)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    // Column-major: idx = row + col * ov
    int ia_idx = idx % ov;
    int jb_idx = idx / ov;

    int i = ia_idx / nvir;
    int a = ia_idx % nvir;
    int j = jb_idx / nvir;
    int b = jb_idx % nvir;

    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t val = 0.0;

    for (int k = 0; k < nocc; k++) {
        for (int c = 0; c < nvir; c++) {
            // ovov indices: ovov[p,q,r,s] at p*nvir*nocc*nvir + q*nocc*nvir + r*nvir + s
            real_t t2_kiac = d_t2[(size_t)k * nocc * nvir * nvir +
                                  (size_t)i * nvir * nvir +
                                  (size_t)a * nvir + c];
            real_t ovov_kbjc = d_eri_ovov[(size_t)k * ovov_stride +
                                          (size_t)b * nocc * nvir +
                                          (size_t)j * nvir + c];

            // P6 term (present in both singlet and triplet)
            val += 0.5 * t2_kiac * ovov_kbjc;    // +0.5*P6

            if (!is_triplet) {
                // Additional singlet terms: +2*P1 - P2 - P5
                real_t t2_ikac = d_t2[(size_t)i * nocc * nvir * nvir +
                                      (size_t)k * nvir * nvir +
                                      (size_t)a * nvir + c];
                real_t ovov_jbkc = d_eri_ovov[(size_t)j * ovov_stride +
                                              (size_t)b * nocc * nvir +
                                              (size_t)k * nvir + c];

                val += 2.0 * t2_ikac * ovov_jbkc;   // +2*P1
                val -=       t2_ikac * ovov_kbjc;    // -P2
                val -=       t2_kiac * ovov_jbkc;    // -P5
            }

            // Transpose: same patterns with (i,a) ↔ (j,b)
            real_t t2_kjbc = d_t2[(size_t)k * nocc * nvir * nvir +
                                  (size_t)j * nvir * nvir +
                                  (size_t)b * nvir + c];
            real_t ovov_kaic = d_eri_ovov[(size_t)k * ovov_stride +
                                          (size_t)a * nocc * nvir +
                                          (size_t)i * nvir + c];

            val += 0.5 * t2_kjbc * ovov_kaic;    // +0.5*P6^T

            if (!is_triplet) {
                real_t t2_jkbc = d_t2[(size_t)j * nocc * nvir * nvir +
                                      (size_t)k * nvir * nvir +
                                      (size_t)b * nvir + c];
                real_t ovov_iakc = d_eri_ovov[(size_t)i * ovov_stride +
                                              (size_t)a * nocc * nvir +
                                              (size_t)k * nvir + c];

                val += 2.0 * t2_jkbc * ovov_iakc;   // +2*P1^T
                val -=       t2_jkbc * ovov_kaic;    // -P2^T
                val -=       t2_kjbc * ovov_iakc;    // -P5^T
            }
        }
    }

    d_ISR_corr[idx] = val;
}

/**
 * @brief Compute ISR-ADC(2) self-energy matrices Σ_oo and Σ_vv (symmetrized)
 *
 * ISR Σ_oo[i,j] = Σ_{k,a,b} t2[i,k,a,b] × (ovov[j,a,k,b] - 0.5*ovov[j,b,k,a])
 * ISR Σ_vv[a,b] = Σ_{i,j,c} t2[i,j,a,c] × (-ovov[i,b,j,c] + 0.5*ovov[j,b,i,c])
 *
 * Output is symmetrized: Σ = (Σ + Σ^T) / 2
 *
 * Sign convention: M11 += -δ_ab*Σ_oo + δ_ij*Σ_vv
 * Σ_oo stores positive values so that -Σ_oo gives the correct ISR correction.
 */
__global__ void adc2_compute_sigma_oo_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_oo,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc) return;

    int i = idx / nocc;
    int j = idx % nocc;

    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t val_ij = 0.0;
    real_t val_ji = 0.0;

    for (int k = 0; k < nocc; k++) {
        for (int a = 0; a < nvir; a++) {
            for (int b = 0; b < nvir; b++) {
                real_t t2_ikab = d_t2[(size_t)i * nocc * nvir * nvir +
                                      (size_t)k * nvir * nvir +
                                      (size_t)a * nvir + b];
                // ISR: ovov[j,a,k,b] - 0.5*ovov[j,b,k,a]
                real_t ja_kb = d_eri_ovov[(size_t)j * ovov_stride +
                                          (size_t)a * nocc * nvir +
                                          (size_t)k * nvir + b];
                real_t jb_ka = d_eri_ovov[(size_t)j * ovov_stride +
                                          (size_t)b * nocc * nvir +
                                          (size_t)k * nvir + a];
                val_ij += t2_ikab * (ja_kb - 0.5 * jb_ka);

                // For symmetrization: compute Σ[j,i] too
                real_t t2_jkab = d_t2[(size_t)j * nocc * nvir * nvir +
                                      (size_t)k * nvir * nvir +
                                      (size_t)a * nvir + b];
                real_t ia_kb = d_eri_ovov[(size_t)i * ovov_stride +
                                          (size_t)a * nocc * nvir +
                                          (size_t)k * nvir + b];
                real_t ib_ka = d_eri_ovov[(size_t)i * ovov_stride +
                                          (size_t)b * nocc * nvir +
                                          (size_t)k * nvir + a];
                val_ji += t2_jkab * (ia_kb - 0.5 * ib_ka);
            }
        }
    }

    // Symmetrize: Σ[i,j] + Σ[j,i] (no 0.5 factor — ISR adds raw + transpose)
    d_sigma_oo[idx] = val_ij + val_ji;
}

__global__ void adc2_compute_sigma_vv_kernel(
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_vv,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir) return;

    int a = idx / nvir;
    int b = idx % nvir;

    size_t ovov_stride = (size_t)nvir * nocc * nvir;
    real_t val_ab = 0.0;
    real_t val_ba = 0.0;

    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            for (int c = 0; c < nvir; c++) {
                real_t t2_ijac = d_t2[(size_t)i * nocc * nvir * nvir +
                                      (size_t)j * nvir * nvir +
                                      (size_t)a * nvir + c];
                // ISR: -ovov[i,b,j,c] + 0.5*ovov[j,b,i,c]
                real_t ib_jc = d_eri_ovov[(size_t)i * ovov_stride +
                                          (size_t)b * nocc * nvir +
                                          (size_t)j * nvir + c];
                real_t jb_ic = d_eri_ovov[(size_t)j * ovov_stride +
                                          (size_t)b * nocc * nvir +
                                          (size_t)i * nvir + c];
                val_ab += t2_ijac * (-ib_jc + 0.5 * jb_ic);

                // For symmetrization: compute Σ_vv[b,a] too
                real_t t2_ijbc = d_t2[(size_t)i * nocc * nvir * nvir +
                                      (size_t)j * nvir * nvir +
                                      (size_t)b * nvir + c];
                real_t ia_jc = d_eri_ovov[(size_t)i * ovov_stride +
                                          (size_t)a * nocc * nvir +
                                          (size_t)j * nvir + c];
                real_t ja_ic = d_eri_ovov[(size_t)j * ovov_stride +
                                          (size_t)a * nocc * nvir +
                                          (size_t)i * nvir + c];
                val_ba += t2_ijbc * (-ia_jc + 0.5 * ja_ic);
            }
        }
    }

    // Symmetrize: Σ[a,b] + Σ[b,a] (no 0.5 factor — ISR adds raw + transpose)
    d_sigma_vv[idx] = val_ab + val_ba;
}

/**
 * @brief Add self-energy corrections to M11 (column-major)
 *
 * M11[(i,a),(j,b)] += -δ_ab × Σ_oo[i,j] + δ_ij × Σ_vv[a,b]
 */
__global__ void adc2_add_self_energy_to_M11_kernel(
    const real_t* __restrict__ d_sigma_oo,
    const real_t* __restrict__ d_sigma_vv,
    real_t* __restrict__ d_M11,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    // Column-major: idx = row + col * ov
    int ia_idx = idx % ov;
    int jb_idx = idx / ov;

    int i = ia_idx / nvir;
    int a = ia_idx % nvir;
    int j = jb_idx / nvir;
    int b = jb_idx % nvir;

    real_t correction = 0.0;

    // -δ_ab × Σ_oo[i,j]
    if (a == b) {
        correction -= d_sigma_oo[i * nocc + j];
    }

    // +δ_ij × Σ_vv[a,b]
    if (i == j) {
        correction += d_sigma_vv[a * nvir + b];
    }

    d_M11[idx] += correction;
}

// ========================================================================
//  CUDA kernels — M12 and M21 dense matrix construction
// ========================================================================

/**
 * @brief Build M12 dense matrix (column-major [ov × dd])
 *
 * M12[KE, IJCD] = δ_{I,K}·[2·vvov[E,C,J,D] - vvov[D,E,J,C]]
 *               + δ_{C,E}·[ooov[J,K,I,D] - 2·ooov[I,K,J,D]]
 */
__global__ void adc2_build_M12_dense_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    real_t* __restrict__ d_M12,
    int nocc, int nvir)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    size_t dd = (size_t)nocc * nocc * nvir * nvir;
    if (idx >= (size_t)ov * dd) return;

    // Column-major [ov × dd]: idx = ov_idx + dd_idx * ov
    int ov_idx = (int)(idx % ov);
    size_t dd_idx = idx / ov;

    int K = ov_idx / nvir;
    int E = ov_idx % nvir;

    int I = (int)(dd_idx / ((size_t)nocc * nvir * nvir));
    int rem = (int)(dd_idx % ((size_t)nocc * nvir * nvir));
    int J = rem / (nvir * nvir);
    rem = rem % (nvir * nvir);
    int C = rem / nvir;
    int D = rem % nvir;

    size_t vvov_stride = (size_t)nvir * nocc * nvir;
    size_t ooov_stride = (size_t)nocc * nocc * nvir;

    real_t val = 0.0;

    // Term 1: δ_{I,K}·[2·vvov[E,C,J,D] - vvov[D,E,J,C]]
    if (I == K) {
        // vvov[E,C,J,D] = (EC|JD)
        real_t EC_JD = d_eri_vvov[(size_t)E * vvov_stride +
                                  (size_t)C * nocc * nvir +
                                  (size_t)J * nvir + D];
        // vvov[D,E,J,C] = (DE|JC)
        real_t DE_JC = d_eri_vvov[(size_t)D * vvov_stride +
                                  (size_t)E * nocc * nvir +
                                  (size_t)J * nvir + C];
        val += 2.0 * EC_JD - DE_JC;
    }

    // Term 2: δ_{C,E}·[ooov[J,K,I,D] - 2·ooov[I,K,J,D]]
    if (C == E) {
        // ooov[J,K,I,D] = (JK|ID)
        real_t JK_ID = d_eri_ooov[(size_t)J * ooov_stride +
                                  (size_t)K * nocc * nvir +
                                  (size_t)I * nvir + D];
        // ooov[I,K,J,D] = (IK|JD)
        real_t IK_JD = d_eri_ooov[(size_t)I * ooov_stride +
                                  (size_t)K * nocc * nvir +
                                  (size_t)J * nvir + D];
        val += JK_ID - 2.0 * IK_JD;
    }

    d_M12[idx] = val;
}

/**
 * @brief Build M21 dense matrix (column-major [dd × ov])
 *
 * M21[IJCD, KE] = δ_{K,I}·vvov[E,C,J,D]
 *               + δ_{K,J}·vvov[E,D,I,C]
 *               - δ_{E,C}·ooov[I,K,J,D]
 *               - δ_{E,D}·ooov[J,K,I,C]
 */
__global__ void adc2_build_M21_dense_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    real_t* __restrict__ d_M21,
    int nocc, int nvir)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    size_t dd = (size_t)nocc * nocc * nvir * nvir;
    if (idx >= dd * ov) return;

    // Column-major [dd × ov]: idx = dd_idx + ov_idx * dd
    size_t dd_idx = idx % dd;
    int ov_idx = (int)(idx / dd);

    int K = ov_idx / nvir;
    int E = ov_idx % nvir;

    int I = (int)(dd_idx / ((size_t)nocc * nvir * nvir));
    int rem = (int)(dd_idx % ((size_t)nocc * nvir * nvir));
    int J = rem / (nvir * nvir);
    rem = rem % (nvir * nvir);
    int C = rem / nvir;
    int D = rem % nvir;

    size_t vvov_stride = (size_t)nvir * nocc * nvir;
    size_t ooov_stride = (size_t)nocc * nocc * nvir;

    real_t val = 0.0;

    // Term 1: δ_{K,I}·vvov[E,C,J,D] = (EC|JD)
    if (K == I) {
        val += d_eri_vvov[(size_t)E * vvov_stride +
                          (size_t)C * nocc * nvir +
                          (size_t)J * nvir + D];
    }

    // Term 2: δ_{K,J}·vvov[E,D,I,C] = (ED|IC)
    if (K == J) {
        val += d_eri_vvov[(size_t)E * vvov_stride +
                          (size_t)D * nocc * nvir +
                          (size_t)I * nvir + C];
    }

    // Term 3: -δ_{E,C}·ooov[I,K,J,D] = -(IK|JD)
    if (E == C) {
        val -= d_eri_ooov[(size_t)I * ooov_stride +
                          (size_t)K * nocc * nvir +
                          (size_t)J * nvir + D];
    }

    // Term 4: -δ_{E,D}·ooov[J,K,I,C] = -(JK|IC)
    if (E == D) {
        val -= d_eri_ooov[(size_t)J * ooov_stride +
                          (size_t)K * nocc * nvir +
                          (size_t)I * nvir + C];
    }

    d_M21[idx] = val;
}

// ========================================================================
//  CUDA kernel — M_eff diagonal (for preconditioner when M12/M21 not stored)
// ========================================================================

/**
 * @brief Compute diagonal of M_eff(ω) = M11_diag + [M12·diag(1/(ω-D2))·M21]_diag
 *
 * Specialization of on-the-fly kernel for row==col (k==j', e==b').
 * ov threads, each O(ov² + o²v) work. One-time computation for preconditioner.
 */
__global__ void adc2_compute_M_eff_diagonal_kernel(
    const real_t* __restrict__ d_M11,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_diagonal,
    real_t omega,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;

    int k = idx / nvir;
    int e = idx % nvir;

    size_t vs1 = (size_t)nvir * nocc * nvir;
    size_t vs2 = (size_t)nocc * nvir;
    size_t os1 = (size_t)nocc * nocc * nvir;
    size_t os2 = (size_t)nocc * nvir;
    size_t ds1 = (size_t)nocc * nvir * nvir;
    size_t ds2 = (size_t)nvir * nvir;

    real_t schur = 0.0;

    // === GROUP 1: δ_{I,k} from M12, with j'=k, b'=e ===

    // G1a: M21 δ_{K,I}→I=k, always active (j'=k). Sum JCD
    for (int J = 0; J < nocc; J++)
        for (int C = 0; C < nvir; C++)
            for (int D = 0; D < nvir; D++) {
                real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + J*nvir + D]
                                 - d_eri_vvov[D*vs1 + e*vs2 + J*nvir + C];
                real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + C*nvir + D]);
                schur += m12 * w * d_eri_vvov[e*vs1 + C*vs2 + J*nvir + D];
            }

    // G1b: M21 δ_{K,J}→J=k. Sum CD
    for (int C = 0; C < nvir; C++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + k*nvir + D]
                             - d_eri_vvov[D*vs1 + e*vs2 + k*nvir + C];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + k*ds2 + C*nvir + D]);
            schur += m12 * w * d_eri_vvov[e*vs1 + D*vs2 + k*nvir + C];
        }

    // G1c: M21 -δ_{E,C}→C=e. Sum JD
    for (int J = 0; J < nocc; J++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + e*vs2 + J*nvir + D]
                             - d_eri_vvov[D*vs1 + e*vs2 + J*nvir + e];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + e*nvir + D]);
            schur -= m12 * w * d_eri_ooov[k*os1 + k*os2 + J*nvir + D];
        }

    // G1d: M21 -δ_{E,D}→D=e. Sum JC
    for (int J = 0; J < nocc; J++)
        for (int C = 0; C < nvir; C++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + J*nvir + e]
                             - d_eri_vvov[e*vs1 + e*vs2 + J*nvir + C];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + C*nvir + e]);
            schur -= m12 * w * d_eri_ooov[J*os1 + k*os2 + k*nvir + C];
        }

    // === GROUP 2: δ_{C,e} from M12, with j'=k, b'=e ===

    // G2a: M21 δ_{K,I}→I=k. Sum JD
    for (int J = 0; J < nocc; J++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = d_eri_ooov[J*os1 + k*os2 + k*nvir + D]
                   - 2.0*d_eri_ooov[k*os1 + k*os2 + J*nvir + D];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + e*nvir + D]);
            schur += m12 * w * d_eri_vvov[e*vs1 + e*vs2 + J*nvir + D];
        }

    // G2b: M21 δ_{K,J}→J=k. Sum ID
    for (int I = 0; I < nocc; I++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = d_eri_ooov[k*os1 + k*os2 + I*nvir + D]
                   - 2.0*d_eri_ooov[I*os1 + k*os2 + k*nvir + D];
            real_t w = 1.0 / (omega - d_D2[I*ds1 + k*ds2 + e*nvir + D]);
            schur += m12 * w * d_eri_vvov[e*vs1 + D*vs2 + I*nvir + e];
        }

    // G2c: M21 -δ_{E,C}→C=e, always active (b'=e). Sum IJD
    for (int I = 0; I < nocc; I++)
        for (int J = 0; J < nocc; J++)
            for (int D = 0; D < nvir; D++) {
                real_t m12 = d_eri_ooov[J*os1 + k*os2 + I*nvir + D]
                       - 2.0*d_eri_ooov[I*os1 + k*os2 + J*nvir + D];
                real_t w = 1.0 / (omega - d_D2[I*ds1 + J*ds2 + e*nvir + D]);
                schur -= m12 * w * d_eri_ooov[I*os1 + k*os2 + J*nvir + D];
            }

    // G2d: M21 -δ_{E,D}→D=e. Sum IJ
    for (int I = 0; I < nocc; I++)
        for (int J = 0; J < nocc; J++) {
            real_t m12 = d_eri_ooov[J*os1 + k*os2 + I*nvir + e]
                   - 2.0*d_eri_ooov[I*os1 + k*os2 + J*nvir + e];
            real_t w = 1.0 / (omega - d_D2[I*ds1 + J*ds2 + e*nvir + e]);
            schur -= m12 * w * d_eri_ooov[J*os1 + k*os2 + I*nvir + e];
        }

    // M11 diagonal: stride = ov + 1
    d_diagonal[idx] = d_M11[(size_t)idx * ov + idx] + schur;
}

// ========================================================================
//  CUDA kernel — on-the-fly M_eff(ω) construction (no dense M12/M21)
// ========================================================================

/**
 * @brief Build M_eff(ω) = M11 + M12·diag(1/(ω-D2))·M21 without storing M12/M21
 *
 * Exploits δ structure: M12 has 2 δ-terms, M21 has 4 → 8 cross-terms,
 * each with reduced summation (O(v²), O(ov), or O(o²) per element).
 */
__global__ void adc2_build_M_eff_onthefly_kernel(
    const real_t* __restrict__ d_M11,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_M_eff,
    real_t omega,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * ov) return;

    int row = idx % ov;
    int col = idx / ov;
    int k  = row / nvir;
    int e  = row % nvir;
    int jp = col / nvir;
    int bp = col % nvir;

    // vvov[a][b][i][c] = (ab|ic), ooov[j][i][k][b] = (ji|kb)
    size_t vs1 = (size_t)nvir * nocc * nvir;
    size_t vs2 = (size_t)nocc * nvir;
    size_t os1 = (size_t)nocc * nocc * nvir;
    size_t os2 = (size_t)nocc * nvir;
    size_t ds1 = (size_t)nocc * nvir * nvir;
    size_t ds2 = (size_t)nvir * nvir;

    real_t schur = 0.0;

    // === GROUP 1: δ_{I,k} from M12 ===
    // M12_val = 2·(eC|JD) - (De|JC),  w = 1/(ω - D2[k,J,C,D])

    // G1a: M21 δ_{j'I} → j'=k, sum JCD
    if (jp == k) {
        for (int J = 0; J < nocc; J++)
            for (int C = 0; C < nvir; C++)
                for (int D = 0; D < nvir; D++) {
                    real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + J*nvir + D]
                                     - d_eri_vvov[D*vs1 + e*vs2 + J*nvir + C];
                    real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + C*nvir + D]);
                    schur += m12 * w * d_eri_vvov[bp*vs1 + C*vs2 + J*nvir + D];
                }
    }

    // G1b: M21 δ_{j'J} → J=j', sum CD
    for (int C = 0; C < nvir; C++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + jp*nvir + D]
                             - d_eri_vvov[D*vs1 + e*vs2 + jp*nvir + C];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + jp*ds2 + C*nvir + D]);
            schur += m12 * w * d_eri_vvov[bp*vs1 + D*vs2 + k*nvir + C];
        }

    // G1c: M21 -δ_{b'C} → C=b', sum JD
    for (int J = 0; J < nocc; J++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + bp*vs2 + J*nvir + D]
                             - d_eri_vvov[D*vs1 + e*vs2 + J*nvir + bp];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + bp*nvir + D]);
            schur -= m12 * w * d_eri_ooov[k*os1 + jp*os2 + J*nvir + D];
        }

    // G1d: M21 -δ_{b'D} → D=b', sum JC
    for (int J = 0; J < nocc; J++)
        for (int C = 0; C < nvir; C++) {
            real_t m12 = 2.0 * d_eri_vvov[e*vs1 + C*vs2 + J*nvir + bp]
                             - d_eri_vvov[bp*vs1 + e*vs2 + J*nvir + C];
            real_t w = 1.0 / (omega - d_D2[k*ds1 + J*ds2 + C*nvir + bp]);
            schur -= m12 * w * d_eri_ooov[J*os1 + jp*os2 + k*nvir + C];
        }

    // === GROUP 2: δ_{C,e} from M12 ===
    // M12_val = (Jk|ID) - 2(Ik|JD),  w = 1/(ω - D2[I,J,e,D])

    // G2a: M21 δ_{j'I} → I=j', sum JD
    for (int J = 0; J < nocc; J++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = d_eri_ooov[J*os1 + k*os2 + jp*nvir + D]
                   - 2.0*d_eri_ooov[jp*os1 + k*os2 + J*nvir + D];
            real_t w = 1.0 / (omega - d_D2[jp*ds1 + J*ds2 + e*nvir + D]);
            schur += m12 * w * d_eri_vvov[bp*vs1 + e*vs2 + J*nvir + D];
        }

    // G2b: M21 δ_{j'J} → J=j', sum ID
    for (int I = 0; I < nocc; I++)
        for (int D = 0; D < nvir; D++) {
            real_t m12 = d_eri_ooov[jp*os1 + k*os2 + I*nvir + D]
                   - 2.0*d_eri_ooov[I*os1 + k*os2 + jp*nvir + D];
            real_t w = 1.0 / (omega - d_D2[I*ds1 + jp*ds2 + e*nvir + D]);
            schur += m12 * w * d_eri_vvov[bp*vs1 + D*vs2 + I*nvir + e];
        }

    // G2c: M21 -δ_{b'e} → b'=e, sum IJD
    if (bp == e) {
        for (int I = 0; I < nocc; I++)
            for (int J = 0; J < nocc; J++)
                for (int D = 0; D < nvir; D++) {
                    real_t m12 = d_eri_ooov[J*os1 + k*os2 + I*nvir + D]
                           - 2.0*d_eri_ooov[I*os1 + k*os2 + J*nvir + D];
                    real_t w = 1.0 / (omega - d_D2[I*ds1 + J*ds2 + e*nvir + D]);
                    schur -= m12 * w * d_eri_ooov[I*os1 + jp*os2 + J*nvir + D];
                }
    }

    // G2d: M21 -δ_{b'D} → D=b', sum IJ
    for (int I = 0; I < nocc; I++)
        for (int J = 0; J < nocc; J++) {
            real_t m12 = d_eri_ooov[J*os1 + k*os2 + I*nvir + bp]
                   - 2.0*d_eri_ooov[I*os1 + k*os2 + J*nvir + bp];
            real_t w = 1.0 / (omega - d_D2[I*ds1 + J*ds2 + e*nvir + bp]);
            schur -= m12 * w * d_eri_ooov[J*os1 + jp*os2 + I*nvir + e];
        }

    d_M_eff[idx] = d_M11[idx] + schur;
}

// ========================================================================
//  CUDA kernels — ω-dependent operations
// ========================================================================

/**
 * @brief Scale M21 by 1/(ω - D2): scaled_M21[dd_idx + ov_idx*dd] = M21[...] / (ω - D2[dd_idx])
 * Input/output: column-major [dd × ov]
 */
__global__ void adc2_scale_M21_by_omega_D2_kernel(
    const real_t* __restrict__ d_M21,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_scaled_M21,
    real_t omega,
    int dd, int ov)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)dd * ov;
    if (idx >= total) return;

    size_t dd_idx = idx % dd;
    real_t scale = 1.0 / (omega - d_D2[dd_idx]);
    d_scaled_M21[idx] = d_M21[idx] * scale;
}

/**
 * @brief Apply (ω - D2)^{-1} element-wise to a vector
 */
__global__ void adc2_apply_omega_D2_inv_kernel(
    const real_t* __restrict__ d_input,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_output,
    real_t omega, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    d_output[idx] = d_input[idx] / (omega - d_D2[idx]);
}

/**
 * @brief Preconditioner: output[i] = input[i] / diagonal[i]
 */
__global__ void adc2_preconditioner_kernel(
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

/**
 * @brief Kernel-based M21·x1: σ2[I,J,C,D] = M21[IJCD,KE]·x1[KE]
 * Exploits δ-structure: 4 terms with sums over E(nvir) or K(nocc)
 * o²v² threads, each O(o+v) work — excellent GPU parallelism
 */
__global__ void adc2_apply_M21_x1_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_x1,
    real_t* __restrict__ d_sigma2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nocc * nocc * nvir * nvir;
    if (idx >= total) return;

    int I = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int J = rem / (nvir * nvir);
    rem = rem % (nvir * nvir);
    int C = rem / nvir;
    int D = rem % nvir;

    int ov = nocc * nvir;
    int vov = nvir * ov;

    real_t val = 0.0;

    // Term 1: Σ_E (EC|JD)·x1[I,E]  where (EC|JD) = eri_vvov[E,C,J,D]
    for (int E = 0; E < nvir; E++) {
        val += d_eri_vvov[E * vov + C * ov + J * nvir + D] * d_x1[I * nvir + E];
    }

    // Term 2: Σ_E (ED|IC)·x1[J,E]  where (ED|IC) = eri_vvov[E,D,I,C]
    for (int E = 0; E < nvir; E++) {
        val += d_eri_vvov[E * vov + D * ov + I * nvir + C] * d_x1[J * nvir + E];
    }

    // Term 3: -Σ_K (IK|JD)·x1[K,C]  where (IK|JD) = eri_ooov[I,K,J,D]
    for (int K = 0; K < nocc; K++) {
        val -= d_eri_ooov[I * nocc * ov + K * ov + J * nvir + D] * d_x1[K * nvir + C];
    }

    // Term 4: -Σ_K (JK|IC)·x1[K,D]  where (JK|IC) = eri_ooov[J,K,I,C]
    for (int K = 0; K < nocc; K++) {
        val -= d_eri_ooov[J * nocc * ov + K * ov + I * nvir + C] * d_x1[K * nvir + D];
    }

    d_sigma2[idx] = val;
}

/**
 * @brief Kernel-based M12·x2: σ1[K,E] += M12[KE,IJCD]·x2[IJCD]
 * Exploits δ-structure: 2 terms with sums over (J,C,D) or (I,J,D)
 * ov threads, each O(ov²+o²v) work
 */
__global__ void adc2_apply_M12_x2_kernel(
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_x2,
    real_t* __restrict__ d_sigma1,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;

    int K = idx / nvir;
    int E = idx % nvir;

    int vov = nvir * ov;
    int vv = nvir * nvir;

    real_t val = 0.0;

    // Term A: Σ_{J,C,D} [2·(EC|JD) - (DE|JC)] · x2[K,J,C,D]
    for (int J = 0; J < nocc; J++) {
        for (int C = 0; C < nvir; C++) {
            for (int D = 0; D < nvir; D++) {
                real_t eri1 = d_eri_vvov[E * vov + C * ov + J * nvir + D];
                real_t eri2 = d_eri_vvov[D * vov + E * ov + J * nvir + C];
                real_t x2v = d_x2[K * nocc * vv + J * vv + C * nvir + D];
                val += (2.0 * eri1 - eri2) * x2v;
            }
        }
    }

    // Term B: Σ_{I,J,D} [(JK|ID) - 2·(IK|JD)] · x2[I,J,E,D]
    for (int I = 0; I < nocc; I++) {
        for (int J = 0; J < nocc; J++) {
            for (int D = 0; D < nvir; D++) {
                real_t eri1 = d_eri_ooov[J * nocc * ov + K * ov + I * nvir + D];
                real_t eri2 = d_eri_ooov[I * nocc * ov + K * ov + J * nvir + D];
                real_t x2v = d_x2[I * nocc * vv + J * vv + E * nvir + D];
                val += (eri1 - 2.0 * eri2) * x2v;
            }
        }
    }

    d_sigma1[idx] += val;  // accumulate (M11·x already written)
}

// ========================================================================
//  ADC2Operator Implementation
// ========================================================================

ADC2Operator::ADC2Operator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int nao,
    bool is_triplet)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      singles_dim_(nocc * nvir),
      doubles_dim_(nocc * nocc * nvir * nvir),
      omega_(0.0),
      is_triplet_(is_triplet),
      d_eri_ovov_(nullptr), d_eri_vvov_(nullptr), d_eri_ooov_(nullptr),
      d_t2_(nullptr),
      d_M11_(nullptr), d_M12_(nullptr), d_M21_(nullptr),
      d_D2_(nullptr), d_D1_(nullptr),
      d_scaled_M21_(nullptr), d_temp_doubles_(nullptr),
      d_diagonal_(nullptr)
{
    size_t ov = singles_dim_;
    size_t dd = doubles_dim_;

    // Allocate ERI blocks
    tracked_cudaMalloc(&d_eri_ovov_, (size_t)nocc * nvir * nocc * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_vvov_, (size_t)nvir * nvir * nocc * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ooov_, (size_t)nocc * nocc * nocc * nvir * sizeof(real_t));

    // Allocate T2 and denominators
    tracked_cudaMalloc(&d_t2_, dd * sizeof(real_t));
    tracked_cudaMalloc(&d_D2_, dd * sizeof(real_t));
    tracked_cudaMalloc(&d_D1_, ov * sizeof(real_t));

    // Allocate M11 (always needed)
    tracked_cudaMalloc(&d_M11_, ov * ov * sizeof(real_t));

    // Check if dense M12/M21 fit in GPU memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t m12_bytes = 3ULL * ov * dd * sizeof(real_t);  // M12 + M21 + scaled_M21
    use_dense_M12_ = (m12_bytes < free_mem / 2);

    // Always allocate temp_doubles (needed for both dense and kernel-based apply)
    tracked_cudaMalloc(&d_temp_doubles_, dd * sizeof(real_t));

    if (use_dense_M12_) {
        tracked_cudaMalloc(&d_M12_, ov * dd * sizeof(real_t));
        tracked_cudaMalloc(&d_M21_, dd * ov * sizeof(real_t));
        tracked_cudaMalloc(&d_scaled_M21_, dd * ov * sizeof(real_t));
    } else {
        printf("  ADC(2): Kernel-based M_eff·x mode (dense M12 would need %.1f GB)\n",
               m12_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    // Allocate diagonal
    tracked_cudaMalloc(&d_diagonal_, ov * sizeof(real_t));

    // Build everything
    extract_eri_blocks(d_eri_mo);
    compute_mp1_t2_and_D2(d_orbital_energies);
    compute_D1(d_orbital_energies);
    build_M11(d_eri_mo, d_orbital_energies);
    if (use_dense_M12_) {
        build_M12_M21();
    }
    compute_diagonal();
}

ADC2Operator::~ADC2Operator() {
    if (d_eri_ovov_) tracked_cudaFree(d_eri_ovov_);
    if (d_eri_vvov_) tracked_cudaFree(d_eri_vvov_);
    if (d_eri_ooov_) tracked_cudaFree(d_eri_ooov_);
    if (d_t2_) tracked_cudaFree(d_t2_);
    if (d_D2_) tracked_cudaFree(d_D2_);
    if (d_D1_) tracked_cudaFree(d_D1_);
    if (d_M11_) tracked_cudaFree(d_M11_);
    if (d_M12_) tracked_cudaFree(d_M12_);
    if (d_M21_) tracked_cudaFree(d_M21_);
    if (d_scaled_M21_) tracked_cudaFree(d_scaled_M21_);
    if (d_temp_doubles_) tracked_cudaFree(d_temp_doubles_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void ADC2Operator::extract_eri_blocks(const real_t* d_eri_mo) {
    int threads = 256;

    int n_ovov = nocc_ * nvir_ * nocc_ * nvir_;
    adc2_extract_eri_ovov_kernel<<<(n_ovov + threads - 1) / threads, threads>>>(
        d_eri_mo, d_eri_ovov_, nocc_, nvir_, nao_);

    int n_vvov = nvir_ * nvir_ * nocc_ * nvir_;
    adc2_extract_eri_vvov_kernel<<<(n_vvov + threads - 1) / threads, threads>>>(
        d_eri_mo, d_eri_vvov_, nocc_, nvir_, nao_);

    int n_ooov = nocc_ * nocc_ * nocc_ * nvir_;
    adc2_extract_eri_ooov_kernel<<<(n_ooov + threads - 1) / threads, threads>>>(
        d_eri_mo, d_eri_ooov_, nocc_, nvir_, nao_);

    cudaDeviceSynchronize();
}

void ADC2Operator::compute_mp1_t2_and_D2(const real_t* d_orbital_energies) {
    int threads = 256;
    int blocks = (doubles_dim_ + threads - 1) / threads;
    adc2_compute_mp1_t2_and_D2_kernel<<<blocks, threads>>>(
        d_eri_ovov_, d_orbital_energies, d_t2_, d_D2_, nocc_, nvir_);
    cudaDeviceSynchronize();
}

void ADC2Operator::compute_D1(const real_t* d_orbital_energies) {
    int threads = 256;
    int blocks = (singles_dim_ + threads - 1) / threads;
    adc2_compute_D1_kernel<<<blocks, threads>>>(
        d_orbital_energies, d_D1_, nocc_, nvir_);
    cudaDeviceSynchronize();
}

void ADC2Operator::build_M11(const real_t* d_eri_mo, const real_t* d_orbital_energies) {
    int threads = 256;
    size_t matrix_size = (size_t)singles_dim_ * singles_dim_;
    int blocks = (matrix_size + threads - 1) / threads;

    // Step 1: Build CIS A-matrix into d_M11_
    adc2_build_cis_matrix_kernel<<<blocks, threads>>>(
        d_eri_mo, d_orbital_energies, d_M11_,
        nocc_, nvir_, nao_, is_triplet_);
    cudaDeviceSynchronize();

    // Step 2: Compute ISR full-block correction into temp
    real_t* d_ISR_corr = nullptr;
    tracked_cudaMalloc(&d_ISR_corr, matrix_size * sizeof(real_t));
    adc2_build_M11_ISR_correction_kernel<<<blocks, threads>>>(
        d_t2_, d_eri_ovov_, d_ISR_corr, nocc_, nvir_, is_triplet_);
    cudaDeviceSynchronize();

    // Step 3: M11 = CIS + ISR_correction (add via cublasDaxpy)
    const real_t one = 1.0;
    cublasDaxpy(gpu::GPUHandle::cublas(), matrix_size,
                &one, d_ISR_corr, 1, d_M11_, 1);
    tracked_cudaFree(d_ISR_corr);

    // Step 4: Compute symmetrized occupied self-energy Σ_oo [nocc × nocc]
    real_t* d_sigma_oo = nullptr;
    tracked_cudaMalloc(&d_sigma_oo, (size_t)nocc_ * nocc_ * sizeof(real_t));
    {
        int n_oo = nocc_ * nocc_;
        int blk = (n_oo + threads - 1) / threads;
        adc2_compute_sigma_oo_kernel<<<blk, threads>>>(
            d_t2_, d_eri_ovov_, d_sigma_oo, nocc_, nvir_);
        cudaDeviceSynchronize();
    }

    // Step 5: Compute symmetrized virtual self-energy Σ_vv [nvir × nvir]
    real_t* d_sigma_vv = nullptr;
    tracked_cudaMalloc(&d_sigma_vv, (size_t)nvir_ * nvir_ * sizeof(real_t));
    {
        int n_vv = nvir_ * nvir_;
        int blk = (n_vv + threads - 1) / threads;
        adc2_compute_sigma_vv_kernel<<<blk, threads>>>(
            d_t2_, d_eri_ovov_, d_sigma_vv, nocc_, nvir_);
        cudaDeviceSynchronize();
    }

    // Step 6: Add self-energy corrections to M11
    //   M11[(i,a),(j,b)] += -δ_ab × Σ_oo[i,j] + δ_ij × Σ_vv[a,b]
    adc2_add_self_energy_to_M11_kernel<<<blocks, threads>>>(
        d_sigma_oo, d_sigma_vv, d_M11_, nocc_, nvir_);
    cudaDeviceSynchronize();

    tracked_cudaFree(d_sigma_oo);
    tracked_cudaFree(d_sigma_vv);
}

void ADC2Operator::build_M12_M21() {
    int threads = 256;
    size_t ov = singles_dim_;
    size_t dd = doubles_dim_;

    // Build M12 [ov × dd] column-major
    {
        size_t total = ov * dd;
        size_t blocks = (total + threads - 1) / threads;
        adc2_build_M12_dense_kernel<<<blocks, threads>>>(
            d_eri_vvov_, d_eri_ooov_, d_M12_, nocc_, nvir_);
    }

    // Build M21 [dd × ov] column-major
    {
        size_t total = dd * ov;
        size_t blocks = (total + threads - 1) / threads;
        adc2_build_M21_dense_kernel<<<blocks, threads>>>(
            d_eri_vvov_, d_eri_ooov_, d_M21_, nocc_, nvir_);
    }

    cudaDeviceSynchronize();
}

void ADC2Operator::compute_diagonal() {
    // Compute M_eff(ω) diagonal including Schur complement contribution
    // Critical for Davidson preconditioner quality
    int threads = 256;
    int blocks = (singles_dim_ + threads - 1) / threads;
    adc2_compute_M_eff_diagonal_kernel<<<blocks, threads>>>(
        d_M11_, d_eri_vvov_, d_eri_ooov_, d_D2_,
        d_diagonal_, omega_, nocc_, nvir_);
    cudaDeviceSynchronize();
}

void ADC2Operator::update_diagonal() {
    compute_diagonal();
}

// ========================================================================
//  Validation: compare on-the-fly kernel vs dense DGEMM
// ========================================================================

void ADC2Operator::validate_onthefly_vs_dense(real_t omega) const {
    if (!use_dense_M12_) {
        printf("  [validate] Cannot validate: dense M12/M21 not available.\n");
        return;
    }

    int ov = singles_dim_;
    size_t matrix_size = (size_t)ov * ov;
    int threads = 256;

    // Allocate two temp buffers
    real_t* d_M_eff_dense = nullptr;
    real_t* d_M_eff_onthefly = nullptr;
    tracked_cudaMalloc(&d_M_eff_dense, matrix_size * sizeof(real_t));
    tracked_cudaMalloc(&d_M_eff_onthefly, matrix_size * sizeof(real_t));

    // Compute via dense DGEMM path
    {
        size_t dd = doubles_dim_;
        size_t total = dd * ov;
        size_t blocks = (total + threads - 1) / threads;
        adc2_scale_M21_by_omega_D2_kernel<<<blocks, threads>>>(
            d_M21_, d_D2_, d_scaled_M21_, omega, (int)dd, ov);
        cudaDeviceSynchronize();

        cudaMemcpy(d_M_eff_dense, d_M11_, matrix_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

        const real_t alpha = 1.0;
        const real_t beta = 1.0;
        cublasDgemm(gpu::GPUHandle::cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
                    ov, ov, (int)dd,
                    &alpha,
                    d_M12_, ov,
                    d_scaled_M21_, (int)dd,
                    &beta,
                    d_M_eff_dense, ov);
    }

    // Compute via on-the-fly kernel
    {
        size_t blocks = (matrix_size + threads - 1) / threads;
        adc2_build_M_eff_onthefly_kernel<<<blocks, threads>>>(
            d_M11_, d_eri_vvov_, d_eri_ooov_, d_D2_,
            d_M_eff_onthefly, omega, nocc_, nvir_);
        cudaDeviceSynchronize();
    }

    // Copy both to host and compare
    std::vector<real_t> h_dense(matrix_size);
    std::vector<real_t> h_onthefly(matrix_size);
    cudaMemcpy(h_dense.data(), d_M_eff_dense, matrix_size * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_onthefly.data(), d_M_eff_onthefly, matrix_size * sizeof(real_t), cudaMemcpyDeviceToHost);

    double max_abs_diff = 0.0;
    int max_row = 0, max_col = 0;
    double max_rel_diff = 0.0;
    int count_large = 0;

    for (size_t idx = 0; idx < matrix_size; idx++) {
        double diff = std::abs(h_dense[idx] - h_onthefly[idx]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_row = (int)(idx % ov);
            max_col = (int)(idx / ov);
        }
        double scale = std::max(std::abs(h_dense[idx]), 1e-15);
        double rel = diff / scale;
        if (rel > max_rel_diff) max_rel_diff = rel;
        if (diff > 1e-10) count_large++;
    }

    int k_max = max_row / nvir_, e_max = max_row % nvir_;
    int jp_max = max_col / nvir_, bp_max = max_col % nvir_;

    printf("  [validate] omega=%.6f  max_abs_diff=%.6e at row=%d(k=%d,e=%d) col=%d(j'=%d,b'=%d)\n",
           omega, max_abs_diff, max_row, k_max, e_max, max_col, jp_max, bp_max);
    printf("  [validate] max_rel_diff=%.6e  elements_with_diff>1e-10: %d / %zu\n",
           max_rel_diff, count_large, matrix_size);

    if (max_abs_diff > 1e-8) {
        // Print a few worst elements for debugging
        printf("  [validate] Dense vs on-the-fly at worst location:\n");
        printf("    dense=%.15e  onthefly=%.15e  diff=%.6e\n",
               h_dense[max_row + (size_t)max_col * ov],
               h_onthefly[max_row + (size_t)max_col * ov],
               max_abs_diff);

        // Also print the diagonal element for reference
        for (int i = 0; i < std::min(5, ov); i++) {
            size_t diag_idx = (size_t)i * ov + i;
            printf("    diag[%d]: dense=%.12f  onthefly=%.12f  diff=%.6e\n",
                   i, h_dense[diag_idx], h_onthefly[diag_idx],
                   std::abs(h_dense[diag_idx] - h_onthefly[diag_idx]));
        }
    }

    tracked_cudaFree(d_M_eff_dense);
    tracked_cudaFree(d_M_eff_onthefly);
}

// ========================================================================
//  build_M_eff_matrix: M_eff(ω) = M11 + M12 · diag(1/(ω-D2)) · M21
// ========================================================================

void ADC2Operator::build_M_eff_matrix(real_t omega, real_t* d_M_eff) const {
    int ov = singles_dim_;
    size_t dd = doubles_dim_;
    int threads = 256;

    if (!use_dense_M12_) {
        // On-the-fly: compute M_eff directly using δ-structure of M12/M21
        size_t total = (size_t)ov * ov;
        size_t blocks = (total + threads - 1) / threads;
        adc2_build_M_eff_onthefly_kernel<<<blocks, threads>>>(
            d_M11_, d_eri_vvov_, d_eri_ooov_, d_D2_,
            d_M_eff, omega, nocc_, nvir_);
        cudaDeviceSynchronize();
        return;
    }

    // Dense path: M_eff = M11 + M12 · diag(1/(ω-D2)) · M21
    // Step 1: scaled_M21 = diag(1/(ω-D2)) · M21
    {
        size_t total = dd * ov;
        size_t blocks = (total + threads - 1) / threads;
        adc2_scale_M21_by_omega_D2_kernel<<<blocks, threads>>>(
            d_M21_, d_D2_, d_scaled_M21_, omega, (int)dd, ov);
        cudaDeviceSynchronize();
    }

    // Step 2: Copy M11 to M_eff
    cudaMemcpy(d_M_eff, d_M11_, (size_t)ov * ov * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Step 3: M_eff += M12 · scaled_M21 (DGEMM)
    const real_t alpha = 1.0;
    const real_t beta = 1.0;
    cublasDgemm(gpu::GPUHandle::cublas(), CUBLAS_OP_N, CUBLAS_OP_N,
                ov, ov, (int)dd,
                &alpha,
                d_M12_, ov,
                d_scaled_M21_, (int)dd,
                &beta,
                d_M_eff, ov);
}

// ========================================================================
//  apply: sigma = M_eff(ω) · R1  (for Davidson solver path)
// ========================================================================

void ADC2Operator::apply(const real_t* d_input, real_t* d_output) const {
    int ov = singles_dim_;
    size_t dd = doubles_dim_;
    int threads = 256;

    // Step 1: sigma = M11 · x (DGEMV) — common to both paths
    {
        const real_t alpha = 1.0;
        const real_t beta = 0.0;
        cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                    ov, ov, &alpha,
                    d_M11_, ov,
                    d_input, 1,
                    &beta, d_output, 1);
    }

    if (use_dense_M12_) {
        // Dense path: DGEMV with stored M12/M21

        // Step 2: temp_doubles = M21 · x
        {
            const real_t alpha = 1.0;
            const real_t beta = 0.0;
            cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                        (int)dd, ov, &alpha,
                        d_M21_, (int)dd,
                        d_input, 1,
                        &beta, d_temp_doubles_, 1);
        }

        // Step 3: temp_doubles /= (ω - D2)
        {
            int blocks = ((int)dd + threads - 1) / threads;
            adc2_apply_omega_D2_inv_kernel<<<blocks, threads>>>(
                d_temp_doubles_, d_D2_, d_temp_doubles_, omega_, (int)dd);
            cudaDeviceSynchronize();
        }

        // Step 4: sigma += M12 · temp_doubles
        {
            const real_t alpha = 1.0;
            const real_t beta = 1.0;
            cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                        ov, (int)dd, &alpha,
                        d_M12_, ov,
                        d_temp_doubles_, 1,
                        &beta, d_output, 1);
        }
    } else {
        // Kernel-based path: exploit δ-structure without storing M12/M21

        // Step 2: temp_doubles = M21 · x (via δ-structure kernel)
        {
            int blocks = ((int)dd + threads - 1) / threads;
            adc2_apply_M21_x1_kernel<<<blocks, threads>>>(
                d_eri_vvov_, d_eri_ooov_, d_input, d_temp_doubles_,
                nocc_, nvir_);
            cudaDeviceSynchronize();
        }

        // Step 3: temp_doubles /= (ω - D2)
        {
            int blocks = ((int)dd + threads - 1) / threads;
            adc2_apply_omega_D2_inv_kernel<<<blocks, threads>>>(
                d_temp_doubles_, d_D2_, d_temp_doubles_, omega_, (int)dd);
            cudaDeviceSynchronize();
        }

        // Step 4: sigma += M12 · temp_doubles (via δ-structure kernel, accumulates)
        {
            int blocks = (ov + threads - 1) / threads;
            adc2_apply_M12_x2_kernel<<<blocks, threads>>>(
                d_eri_vvov_, d_eri_ooov_, d_temp_doubles_, d_output,
                nocc_, nvir_);
            cudaDeviceSynchronize();
        }
    }
}

void ADC2Operator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256;
    int blocks = (singles_dim_ + threads - 1) / threads;
    adc2_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, singles_dim_);
}

} // namespace gansu
