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
 * @file eom_cc2_operator.cu
 * @brief GPU implementation of EOM-CC2 operator in full singles+doubles space
 *
 * All coefficients use ÷2 convention (spin degeneracy factor divided out).
 *
 * σ1 (EOM_CC2_RHF.md, ÷2, 11 grouped terms):
 *   T1-T2:  Same as EOM-MP2 (Fock, CIS-like ERI, T2×R1, R2 coupling)
 *   T3-T4:  T1-dependent terms (f_ov×T1×R1, ERI×T1×R1)
 *   T11:    T1×R2 coupling via OVOV
 *
 * σ2 (EOM_CC2_RHF.md, ÷2, 8 grouped terms):
 *   S1-S2:  Fock diagonal on R2 (M22 is EXACTLY diagonal)
 *   S3-S8:  R1-dependent terms (ERI×R1, ERI×T1×R1, ERI×T1×T1×R1)
 *           NO T2×R2 or ERI×R2 terms (unlike EOM-MP2)
 */

#include <cstdio>
#include <cmath>
#include <vector>

#include "eom_cc2_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// Reuse EOM-MP2 ERI extraction kernels (declared extern)
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_compute_D1_kernel(
    const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_extract_fock_kernel(
    const real_t*, real_t*, real_t*, int, int);
extern __global__ void eom_mp2_build_diagonal_kernel(
    const real_t*, const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_preconditioner_kernel(
    const real_t*, const real_t*, real_t*, int);

// M22 diagonal computation: 3*(ε_b - ε_j) in ÷2 convention
// Used by Schur complement for exact M22⁻¹ inversion
__global__ void eom_cc2_compute_D2_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_D2,
    int nocc, int nvir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;
    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;
    // M22 diagonal in ÷2 convention: 3*(ε_b - ε_j)
    // From σ2 Fock terms: -3*f_jj*R2 + 3*f_bb*R2 (S1+S2)
    d_D2[idx] = 3.0 * (d_orbital_energies[b + nocc] - d_orbital_energies[j]);
}

// ========================================================================
//  ERI access macros
// ========================================================================

#define OVOV(i,a,j,b) d_eri_ovov[(size_t)(i)*nvir*nocc*nvir + (size_t)(a)*nocc*nvir + (size_t)(j)*nvir + (b)]
#define VVOV(a,b,i,c) d_eri_vvov[(size_t)(a)*nvir*nocc*nvir + (size_t)(b)*nocc*nvir + (size_t)(i)*nvir + (c)]
#define OOOV(j,i,k,b) d_eri_ooov[(size_t)(j)*nocc*nocc*nvir + (size_t)(i)*nocc*nvir + (size_t)(k)*nvir + (b)]
#define OOOO(i,j,k,l) d_eri_oooo[(size_t)(i)*nocc*nocc*nocc + (size_t)(j)*nocc*nocc + (size_t)(k)*nocc + (l)]
#define VVVV(a,b,c,d) d_eri_vvvv[(size_t)(a)*nvir*nvir*nvir + (size_t)(b)*nvir*nvir + (size_t)(c)*nvir + (d)]
#define OOVV(i,j,a,b) d_eri_oovv[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define OVVO(i,a,b,j) d_eri_ovvo[(size_t)(i)*nvir*nvir*nocc + (size_t)(a)*nvir*nocc + (size_t)(b)*nocc + (j)]
#define T1(i,a) d_t1[(i)*nvir + (a)]
#define T2(i,j,a,b) d_t2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define R1(i,a) d_r1[(i)*nvir + (a)]
#define R2(i,j,a,b) d_r2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// ========================================================================
//  σ1 kernel — EOM-CC2 singles sigma vector
// ========================================================================

/**
 * EOM-CC2 σ1 (÷2, from EOM_CC2_RHF.md factored form):
 *
 * Terms T1-T2 (Fock diagonal, same as EOM-MP2):
 *   -1 f_mi r^a_m + 1 f_ae r^e_i
 *
 * Term T3 (CIS-like + T1 correction):
 *   Σ_{m,e} r^e_m [2(ai|me) - (ae|mi)]                     (same as EOM-MP2)
 *   + Σ_{m,e} f_me [-t^a_m r^e_i + t^e_i r^a_m]           (T1-dependent)
 *   + Σ_{m,n,e} t^e_n r^a_m [-2(mi|ne) + (me|ni)]         (T1×R1)
 *   + Σ_{m,e,f} t^f_m r^e_i [2(ae|mf) - (af|me)]          (T1×R1)
 *
 * Terms T4-T5 (T2×R1, same structure as EOM-MP2):
 *   Σ_{m,n,e,f} t^{ef}_{in} r^a_m [-(me|nf) + (mf|ne)]
 *   Σ_{m,n,e,f} t^{af}_{mn} r^e_i [-(me|nf) + (mf|ne)]
 *
 * Terms T6-T7 (R2 coupling, same as EOM-MP2):
 *   Σ_{m,n,e} r^{ae}_{mn} [-(mi|ne) + (me|ni)]
 *   2 Σ_{m,e,f} r^{ef}_{im} [(ae|mf) - (af|me)]
 *
 * Term T11 (T1×R2, NOT in EOM-MP2):
 *   2 Σ_{m,n,e,f} t^f_n r^{ae}_{im} [-2(me|nf) + (mf|ne)]  (T1×R2 coupling)
 */
__global__ void eom_cc2_sigma1_kernel(
    const real_t* __restrict__ d_f_oo,
    const real_t* __restrict__ d_f_vv,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_r1,
    const real_t* __restrict__ d_r2,
    real_t* __restrict__ d_sigma1,
    int nocc, int nvir)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= nocc * nvir) return;
    int i = ia / nvir;
    int a = ia % nvir;

    real_t sigma = 0.0;

    // T1: -1 f_mi r^a_m (Fock diagonal)
    sigma += -1.0 * d_f_oo[i] * R1(i, a);

    // T2: +1 f_ae r^e_i (Fock diagonal)
    sigma += 1.0 * d_f_vv[a] * R1(i, a);

    // T3: CIS-like: Σ_{m,e} r^e_m [2(ai|me) - (ae|mi)]
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            real_t ai_me = OVOV(i, a, m, e);
            real_t ae_mi = OOVV(m, i, a, e);
            sigma += R1(m, e) * (2.0 * ai_me - ae_mi);
        }
    }

    // T1-dependent: Σ_{m,n,e} t^e_n r^a_m [-2(mi|ne) + (me|ni)]
    for (int m = 0; m < nocc; m++) {
        real_t r1_am = R1(m, a);
        real_t inner = 0.0;
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                inner += T1(n, e) * (-2.0 * OOOV(m, i, n, e) + OOOV(n, i, m, e));
            }
        }
        sigma += r1_am * inner;
    }

    // T1-dependent: Σ_{m,e,f} t^f_m r^e_i [2(ae|mf) - (af|me)]
    for (int e = 0; e < nvir; e++) {
        real_t r1_ei = R1(i, e);
        real_t inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int f = 0; f < nvir; f++) {
                inner += T1(m, f) * (2.0 * VVOV(a, e, m, f) - VVOV(a, f, m, e));
            }
        }
        sigma += r1_ei * inner;
    }

    // T4: Σ_{m,n,e,f} t^{ef}_{in} r^a_m [-(me|nf) + (mf|ne)]
    for (int m = 0; m < nocc; m++) {
        real_t r1_am = R1(m, a);
        real_t inner = 0.0;
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -OVOV(m, e, n, f) + OVOV(m, f, n, e);
                    inner += T2(i, n, e, f) * K;
                }
            }
        }
        sigma += r1_am * inner;
    }

    // T5: Σ_{m,n,e,f} t^{af}_{mn} r^e_i [-(me|nf) + (mf|ne)]
    for (int e = 0; e < nvir; e++) {
        real_t r1_ei = R1(i, e);
        real_t inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -OVOV(m, e, n, f) + OVOV(m, f, n, e);
                    inner += T2(m, n, a, f) * K;
                }
            }
        }
        sigma += r1_ei * inner;
    }

    // T6: Σ_{m,n,e} r^{ae}_{mn} [-(mi|ne) + (me|ni)]
    for (int m = 0; m < nocc; m++) {
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                real_t K = -OOOV(m, i, n, e) + OOOV(n, i, m, e);
                sigma += R2(m, n, a, e) * K;
            }
        }
    }

    // T7: 2 Σ_{m,e,f} r^{ef}_{im} [(ae|mf) - (af|me)]
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            for (int f = 0; f < nvir; f++) {
                real_t K = VVOV(a, e, m, f) - VVOV(a, f, m, e);
                sigma += 2.0 * R2(i, m, e, f) * K;
            }
        }
    }

    // T11 (EOM-CC2 specific): 2 Σ_{m,n,e,f} t^f_n r^{ae}_{im} [-2(me|nf) + (mf|ne)]
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            real_t r2_ae_im = R2(i, m, a, e);
            real_t inner = 0.0;
            for (int n = 0; n < nocc; n++) {
                for (int f = 0; f < nvir; f++) {
                    inner += T1(n, f) * (-2.0 * OVOV(m, e, n, f) + OVOV(m, f, n, e));
                }
            }
            sigma += 2.0 * r2_ae_im * inner;
        }
    }

    d_sigma1[ia] = sigma;
}


// ========================================================================
//  σ2 kernel — EOM-CC2 doubles sigma vector
// ========================================================================

/**
 * EOM-CC2 σ2 (÷2, from EOM_CC2_RHF.md):
 *
 * S1-S2: Fock diagonal on R2 (M22 is EXACTLY diagonal):
 *   -3 f_mj r^{ab}_{im} + 3 f_be r^{ae}_{ij}
 *
 * S3-S8: R1-dependent terms ONLY (no R2 coupling beyond Fock):
 *   S3: 2 Σ_m r^a_m [-(mi|bj) + (mj|bi)]                 (ERI×R1)
 *   S4: 2 Σ_e r^e_j [(ai|be) - (ae|bi)]                   (ERI×R1)
 *   S5: 2 Σ_{m,e} t^e_j r^a_m [-(mi|be) + (me|bi)]       (T1×R1)
 *   S6: Σ_{e} t^e_j r^b_i [(ai|be) - 2(ae|bi)]           (T1×R1, broadcast)
 *   S7: 2 Σ_{m,k} t^a_m r^b_k [-(mi|kj) + (mj|ki)]      (T1×R1)
 *   S8: 2 Σ_{c,e} t^c_i r^e_j [(ac|be) - (ae|bc)]        (T1×R1)
 */
__global__ void eom_cc2_sigma2_kernel(
    const real_t* __restrict__ d_f_oo,
    const real_t* __restrict__ d_f_vv,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oooo,
    const real_t* __restrict__ d_eri_vvvv,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_eri_ovvo,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_r1,
    const real_t* __restrict__ d_r2,
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

    // S1: -3 f_mj r^{ab}_{im} → canonical: -3 eps_j r^{ab}_{ij}
    sigma += -3.0 * d_f_oo[j] * R2(i, j, a, b);

    // S2: +3 f_be r^{ae}_{ij} → canonical: +3 eps_b r^{ab}_{ij}
    sigma += 3.0 * d_f_vv[b] * R2(i, j, a, b);

    // S3: 2 Σ_m r^a_m [-(mi|bj) + (mj|bi)]
    for (int m = 0; m < nocc; m++) {
        real_t mi_bj = OOOV(i, m, j, b);
        real_t mj_bi = OOOV(j, m, i, b);
        sigma += 2.0 * R1(m, a) * (-mi_bj + mj_bi);
    }

    // S4: 2 Σ_e r^e_j [(ai|be) - (ae|bi)]
    for (int e = 0; e < nvir; e++) {
        real_t ai_be = VVOV(b, e, i, a);  // (be|ia) = (ai|be) by bra-ket swap
        real_t ae_bi = VVOV(a, e, i, b);
        sigma += 2.0 * R1(j, e) * (ai_be - ae_bi);
    }

    // S5: 2 Σ_{m,e} t^e_j r^a_m [-(mi|be) + (me|bi)]
    for (int m = 0; m < nocc; m++) {
        real_t r1_am = R1(m, a);
        real_t inner = 0.0;
        for (int e = 0; e < nvir; e++) {
            real_t mi_be = OOVV(m, i, b, e);
            real_t me_bi = OVVO(m, e, b, i);
            inner += T1(j, e) * (-mi_be + me_bi);
        }
        sigma += 2.0 * r1_am * inner;
    }

    // S6: Σ_{e} t^e_j r^b_i [(ai|be) - 2(ae|bi)]
    // (ai|be) = VVOV(b,e,i,a) [bra-ket swap: (be|ia) = (ai|be)]
    // (ae|bi) = VVOV(a,e,i,b)
    {
        real_t sum6 = 0.0;
        for (int e = 0; e < nvir; e++) {
            real_t ai_be = VVOV(b, e, i, a);
            real_t ae_bi = VVOV(a, e, i, b);
            sum6 += T1(j, e) * (ai_be - 2.0 * ae_bi);
        }
        sigma += R1(i, b) * sum6;
    }

    // S7: 2 Σ_{m,k} t^a_m r^b_k [-(mi|kj) + (mj|ki)]
    for (int m = 0; m < nocc; m++) {
        for (int k = 0; k < nocc; k++) {
            real_t mi_kj = OOOO(m, i, k, j);
            real_t mj_ki = OOOO(m, j, k, i);
            sigma += 2.0 * T1(m, a) * R1(k, b) * (-mi_kj + mj_ki);
        }
    }

    // S8: 2 Σ_{c,e} t^c_i r^e_j [(ac|be) - (ae|bc)]
    for (int c = 0; c < nvir; c++) {
        for (int e = 0; e < nvir; e++) {
            real_t ac_be = VVVV(a, c, b, e);
            real_t ae_bc = VVVV(a, e, b, c);
            sigma += 2.0 * T1(i, c) * R1(j, e) * (ac_be - ae_bc);
        }
    }

    d_sigma2[idx] = sigma;
}

#undef OVOV
#undef VVOV
#undef OOOV
#undef OOOO
#undef VVVV
#undef OOVV
#undef OVVO
#undef T1
#undef T2
#undef R1
#undef R2

// ========================================================================
//  EOMCC2Operator Implementation
// ========================================================================

EOMCC2Operator::EOMCC2Operator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    int nocc, int nvir, int nao)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      singles_dim_(nocc * nvir),
      doubles_dim_(nocc * nocc * nvir * nvir),
      total_dim_(nocc * nvir + nocc * nocc * nvir * nvir),
      d_t1_(d_t1), d_t2_(d_t2),
      d_eri_ovov_(nullptr), d_eri_vvov_(nullptr), d_eri_ooov_(nullptr),
      d_eri_oooo_(nullptr), d_eri_vvvv_(nullptr), d_eri_oovv_(nullptr),
      d_eri_ovvo_(nullptr),
      d_D1_(nullptr), d_D2_(nullptr),
      d_f_oo_(nullptr), d_f_vv_(nullptr),
      d_diagonal_(nullptr)
{
    extract_eri_blocks(d_eri_mo);
    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
}

EOMCC2Operator::~EOMCC2Operator() {
    if (d_t1_) tracked_cudaFree(d_t1_);
    if (d_t2_) tracked_cudaFree(d_t2_);
    if (d_eri_ovov_) tracked_cudaFree(d_eri_ovov_);
    if (d_eri_vvov_) tracked_cudaFree(d_eri_vvov_);
    if (d_eri_ooov_) tracked_cudaFree(d_eri_ooov_);
    if (d_eri_oooo_) tracked_cudaFree(d_eri_oooo_);
    if (d_eri_vvvv_) tracked_cudaFree(d_eri_vvvv_);
    if (d_eri_oovv_) tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovvo_) tracked_cudaFree(d_eri_ovvo_);
    if (d_D1_) tracked_cudaFree(d_D1_);
    if (d_D2_) tracked_cudaFree(d_D2_);
    if (d_f_oo_) tracked_cudaFree(d_f_oo_);
    if (d_f_vv_) tracked_cudaFree(d_f_vv_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void EOMCC2Operator::extract_eri_blocks(const real_t* d_eri_mo) {
    int threads = 256;
    int blocks;

    size_t ovov_size = (size_t)nocc_ * nvir_ * nocc_ * nvir_;
    tracked_cudaMalloc(&d_eri_ovov_, ovov_size * sizeof(real_t));
    blocks = (ovov_size + threads - 1) / threads;
    eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_, nvir_, nao_);

    size_t vvov_size = (size_t)nvir_ * nvir_ * nocc_ * nvir_;
    tracked_cudaMalloc(&d_eri_vvov_, vvov_size * sizeof(real_t));
    blocks = (vvov_size + threads - 1) / threads;
    eom_mp2_extract_eri_vvov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvov_, nocc_, nvir_, nao_);

    size_t ooov_size = (size_t)nocc_ * nocc_ * nocc_ * nvir_;
    tracked_cudaMalloc(&d_eri_ooov_, ooov_size * sizeof(real_t));
    blocks = (ooov_size + threads - 1) / threads;
    eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_, nvir_, nao_);

    size_t oooo_size = (size_t)nocc_ * nocc_ * nocc_ * nocc_;
    tracked_cudaMalloc(&d_eri_oooo_, oooo_size * sizeof(real_t));
    blocks = (oooo_size + threads - 1) / threads;
    eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_, nao_);

    size_t vvvv_size = (size_t)nvir_ * nvir_ * nvir_ * nvir_;
    tracked_cudaMalloc(&d_eri_vvvv_, vvvv_size * sizeof(real_t));
    blocks = (vvvv_size + threads - 1) / threads;
    eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_, nvir_, nao_);

    size_t oovv_size = (size_t)nocc_ * nocc_ * nvir_ * nvir_;
    tracked_cudaMalloc(&d_eri_oovv_, oovv_size * sizeof(real_t));
    blocks = (oovv_size + threads - 1) / threads;
    eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_, nvir_, nao_);

    size_t ovvo_size = (size_t)nocc_ * nvir_ * nvir_ * nocc_;
    tracked_cudaMalloc(&d_eri_ovvo_, ovvo_size * sizeof(real_t));
    blocks = (ovvo_size + threads - 1) / threads;
    eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_, nvir_, nao_);

    cudaDeviceSynchronize();
}

void EOMCC2Operator::compute_denominators_and_fock(const real_t* d_orbital_energies) {
    int threads = 256;

    tracked_cudaMalloc(&d_D2_, (size_t)doubles_dim_ * sizeof(real_t));
    int blocks = (doubles_dim_ + threads - 1) / threads;
    eom_cc2_compute_D2_kernel<<<blocks, threads>>>(
        d_orbital_energies, d_D2_, nocc_, nvir_);

    tracked_cudaMalloc(&d_D1_, (size_t)singles_dim_ * sizeof(real_t));
    blocks = (singles_dim_ + threads - 1) / threads;
    eom_mp2_compute_D1_kernel<<<blocks, threads>>>(
        d_orbital_energies, d_D1_, nocc_, nvir_);

    tracked_cudaMalloc(&d_f_oo_, (size_t)nocc_ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv_, (size_t)nvir_ * sizeof(real_t));
    int nao = nocc_ + nvir_;
    blocks = (nao + threads - 1) / threads;
    eom_mp2_extract_fock_kernel<<<blocks, threads>>>(
        d_orbital_energies, d_f_oo_, d_f_vv_, nocc_, nvir_);

    cudaDeviceSynchronize();
}

void EOMCC2Operator::build_diagonal() {
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));
    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    eom_mp2_build_diagonal_kernel<<<blocks, threads>>>(
        d_D1_, d_D2_, d_diagonal_, singles_dim_, doubles_dim_);
    cudaDeviceSynchronize();
}

void EOMCC2Operator::apply(const real_t* d_input, real_t* d_output) const {
    int threads = 256;

    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + singles_dim_;
    real_t* d_sigma1 = d_output;
    real_t* d_sigma2 = d_output + singles_dim_;

    int blocks1 = (singles_dim_ + threads - 1) / threads;
    eom_cc2_sigma1_kernel<<<blocks1, threads>>>(
        d_f_oo_, d_f_vv_,
        d_eri_ovov_, d_eri_vvov_, d_eri_ooov_, d_eri_oovv_,
        d_t1_, d_t2_, d_r1, d_r2, d_sigma1,
        nocc_, nvir_);

    int blocks2 = (doubles_dim_ + threads - 1) / threads;
    eom_cc2_sigma2_kernel<<<blocks2, threads>>>(
        d_f_oo_, d_f_vv_,
        d_eri_ovov_, d_eri_vvov_, d_eri_ooov_,
        d_eri_oooo_, d_eri_vvvv_, d_eri_oovv_, d_eri_ovvo_,
        d_t1_, d_r1, d_r2, d_sigma2,
        nocc_, nvir_);

    cudaDeviceSynchronize();
}

void EOMCC2Operator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256;
    int blocks = (total_dim_ + threads - 1) / threads;
    eom_mp2_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, total_dim_);
}

} // namespace gansu
