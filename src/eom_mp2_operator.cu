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
 * @file eom_mp2_operator.cu
 * @brief GPU implementation of EOM-MP2 operator in full singles+doubles space
 *
 * Implements σ = H_EOM-MP2 × R for RHF reference.
 * R = [R1(ov) | R2(oovv)], σ = [σ1(ov) | σ2(oovv)]
 *
 * σ1 (RHF factored, EOM_MP2_RHF.md lines 90-97):
 *   7 terms using f_oo, f_vv, eri_ovov, eri_oovv, eri_vvov, eri_ooov, t2
 *
 * σ2 (RHF factored, EOM_MP2_RHF.md lines 180-193):
 *   13 terms using f_oo, f_vv, eri_oooo, eri_vvvv, eri_oovv, eri_vvov,
 *   eri_ooov, eri_ovov, eri_ovvo, t2
 *
 * ALL indices except free (a,b for σ; i,j for σ) are summed.
 * In canonical MOs, Fock is diagonal: f_oo[i,j] = eps_i δ_ij, f_vv[a,b] = eps_a δ_ab.
 *
 * ERI block index layouts:
 *   eri_ovov[(i)*nvir*nocc*nvir + (a)*nocc*nvir + (j)*nvir + (b)] = (ia|jb)
 *   eri_vvov[(a)*nvir*nocc*nvir + (b)*nocc*nvir + (i)*nvir + (c)] = (ab|ic)
 *   eri_ooov[(j)*nocc*nocc*nvir + (i)*nocc*nvir + (k)*nvir + (b)] = (ji|kb)
 *   eri_oooo[(i)*nocc*nocc*nocc + (j)*nocc*nocc + (k)*nocc + (l)] = (ij|kl)
 *   eri_vvvv[(a)*nvir*nvir*nvir + (b)*nvir*nvir + (c)*nvir + (d)] = (ab|cd)
 *   eri_oovv[(i)*nocc*nvir*nvir + (j)*nvir*nvir + (a)*nvir + (b)] = (ij|ab)
 *   eri_ovvo[(i)*nvir*nvir*nocc + (a)*nvir*nocc + (b)*nocc + (j)] = (ia|bj)
 *
 * Symmetries used (real orbitals): (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq)
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "eom_mp2_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// ========================================================================
//  ERI block extraction kernels
// ========================================================================

__global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nocc * nvir * nocc * nvir;
    if (idx >= total) return;
    int i = idx / (nvir * nocc * nvir);
    int rem = idx % (nvir * nocc * nvir);
    int a = rem / (nocc * nvir);
    rem %= (nocc * nvir);
    int j = rem / nvir;
    int b = rem % nvir;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)i * nao + a + nocc) * nao2 + (size_t)j * nao + b + nocc];
}

__global__ void eom_mp2_extract_eri_vvov_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir * nocc * nvir) return;
    int a = idx / (nvir * nocc * nvir);
    int rem = idx % (nvir * nocc * nvir);
    int b = rem / (nocc * nvir);
    rem %= (nocc * nvir);
    int i = rem / nvir;
    int c = rem % nvir;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)(a+nocc) * nao + b+nocc) * nao2 + (size_t)i * nao + c+nocc];
}

__global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nocc * nvir) return;
    int j = idx / (nocc * nocc * nvir);
    int rem = idx % (nocc * nocc * nvir);
    int i = rem / (nocc * nvir);
    rem %= (nocc * nvir);
    int k = rem / nvir;
    int b = rem % nvir;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)j * nao + i) * nao2 + (size_t)k * nao + b+nocc];
}

__global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nocc * nocc) return;
    int i = idx / (nocc * nocc * nocc);
    int rem = idx % (nocc * nocc * nocc);
    int j = rem / (nocc * nocc);
    rem %= (nocc * nocc);
    int k = rem / nocc;
    int l = rem % nocc;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)i * nao + j) * nao2 + (size_t)k * nao + l];
}

__global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir * nvir * nvir) return;
    int a = idx / (nvir * nvir * nvir);
    int rem = idx % (nvir * nvir * nvir);
    int b = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int c = rem / nvir;
    int d = rem % nvir;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)(a+nocc)*nao + b+nocc)*nao2 + (size_t)(c+nocc)*nao + d+nocc];
}

__global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;
    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
}

__global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t* __restrict__ d_eri_mo, real_t* __restrict__ d_out,
    int nocc, int nvir, int nao) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir * nvir * nocc) return;
    int i = idx / (nvir * nvir * nocc);
    int rem = idx % (nvir * nvir * nocc);
    int a = rem / (nvir * nocc);
    rem %= (nvir * nocc);
    int b = rem / nocc;
    int j = rem % nocc;
    size_t nao2 = (size_t)nao * nao;
    d_out[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
}

// ========================================================================
//  T2, D1, D2, Fock kernels
// ========================================================================

__global__ void eom_mp2_compute_t2_D2_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_t2, real_t* __restrict__ d_D2,
    int nocc, int nvir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;
    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;
    real_t eps_i = d_orbital_energies[i];
    real_t eps_j = d_orbital_energies[j];
    real_t eps_a = d_orbital_energies[a + nocc];
    real_t eps_b = d_orbital_energies[b + nocc];
    real_t denom = eps_i + eps_j - eps_a - eps_b;
    real_t ia_jb = d_eri_ovov[(size_t)i * nvir * nocc * nvir +
                              (size_t)a * nocc * nvir + (size_t)j * nvir + b];
    d_t2[idx] = ia_jb / denom;
    d_D2[idx] = eps_a + eps_b - eps_i - eps_j;
}

__global__ void eom_mp2_compute_D1_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_D1, int nocc, int nvir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    d_D1[idx] = d_orbital_energies[idx % nvir + nocc] - d_orbital_energies[idx / nvir];
}

__global__ void eom_mp2_extract_fock_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_f_oo, real_t* __restrict__ d_f_vv,
    int nocc, int nvir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nocc) d_f_oo[idx] = d_orbital_energies[idx];
    if (idx < nvir) d_f_vv[idx] = d_orbital_energies[idx + nocc];
}

// ========================================================================
//  Diagonal and preconditioner kernels
// ========================================================================

__global__ void eom_mp2_build_diagonal_kernel(
    const real_t* __restrict__ d_D1, const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_diagonal, int singles_dim, int doubles_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= singles_dim + doubles_dim) return;
    d_diagonal[idx] = (idx < singles_dim) ? d_D1[idx] : d_D2[idx - singles_dim];
}

__global__ void eom_mp2_preconditioner_kernel(
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t diag = d_diagonal[idx];
    d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
}

// ========================================================================
//  Helper macros for ERI block access
// ========================================================================

// eri_ovov[i,a,j,b] = (ia|jb)
#define OVOV(i,a,j,b) d_eri_ovov[(size_t)(i)*nvir*nocc*nvir + (size_t)(a)*nocc*nvir + (size_t)(j)*nvir + (b)]

// eri_vvov[a,b,i,c] = (ab|ic)
#define VVOV(a,b,i,c) d_eri_vvov[(size_t)(a)*nvir*nocc*nvir + (size_t)(b)*nocc*nvir + (size_t)(i)*nvir + (c)]

// eri_ooov[j,i,k,b] = (ji|kb)
#define OOOV(j,i,k,b) d_eri_ooov[(size_t)(j)*nocc*nocc*nvir + (size_t)(i)*nocc*nvir + (size_t)(k)*nvir + (b)]

// eri_oooo[i,j,k,l] = (ij|kl)
#define OOOO(i,j,k,l) d_eri_oooo[(size_t)(i)*nocc*nocc*nocc + (size_t)(j)*nocc*nocc + (size_t)(k)*nocc + (l)]

// eri_vvvv[a,b,c,d] = (ab|cd)
#define VVVV(a,b,c,d) d_eri_vvvv[(size_t)(a)*nvir*nvir*nvir + (size_t)(b)*nvir*nvir + (size_t)(c)*nvir + (d)]

// eri_oovv[i,j,a,b] = (ij|ab)
#define OOVV(i,j,a,b) d_eri_oovv[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// eri_ovvo[i,a,b,j] = (ia|bj)
#define OVVO(i,a,b,j) d_eri_ovvo[(size_t)(i)*nvir*nvir*nocc + (size_t)(a)*nvir*nocc + (size_t)(b)*nocc + (j)]

// t2[i,j,a,b] = t2^{ab}_{ij}
#define T2(i,j,a,b) d_t2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// r1[i,a] = r1^a_i  (singles vector, layout: i*nvir + a)
#define R1(i,a) d_r1[(i)*nvir + (a)]

// r2[i,j,a,b] = r2^{ab}_{ij}  (doubles vector)
#define R2(i,j,a,b) d_r2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// ========================================================================
//  σ1 kernel — EOM-MP2 singles sigma vector
// ========================================================================

/**
 * σ1 (RHF factored form, EOM_MP2_RHF.md lines 90-97):
 *
 *   σ^a_i = -1 Σ_m f_mi r^a_m                                        (T1)
 *          + 1 Σ_e f_ae r^e_i                                         (T2)
 *          + 1 Σ_{m,e} r^e_m [2(ai|me) - (ae|mi)]                    (T3)
 *          + 1 Σ_{m,n,e,f} t^{ef}_{in} r^a_m [-(me|nf) + (mf|ne)]   (T4)
 *          + 1 Σ_{m,n,e,f} t^{af}_{mn} r^e_i [-(me|nf) + (mf|ne)]   (T5)
 *          + 1 Σ_{m,n,e} r^{ae}_{mn} [-(mi|ne) + (me|ni)]           (T6)
 *          + 2 Σ_{m,e,f} r^{ef}_{im} [(ae|mf) - (af|me)]            (T7)
 *
 * In canonical MOs: T1+T2 = 2(eps_a - eps_i) r^a_i
 *
 * ERI symmetry: (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq) for real orbitals.
 * Use stored blocks and symmetry to access needed integrals.
 */
__global__ void eom_mp2_sigma1_kernel(
    const real_t* __restrict__ d_f_oo,
    const real_t* __restrict__ d_f_vv,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oovv,
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

    // T1: -1 f_mi r^a_m → canonical: -eps_i r^a_i
    sigma += -1.0 * d_f_oo[i] * R1(i, a);

    // T2: +1 f_ae r^e_i → canonical: +eps_a r^a_i
    sigma += 1.0 * d_f_vv[a] * R1(i, a);

    // T3: +1 Σ_{m,e} r^e_m [2(ai|me) - (ae|mi)]
    //   (ai|me): a∈vir,i∈occ,m∈occ,e∈vir → (ai|me) = (ia|me) [swap bra] = OVOV(i,a,m,e) ✓
    //   (ae|mi): a∈vir,e∈vir,m∈occ,i∈occ → (ae|mi) = (mi|ae) [bra-ket] = OOVV(m,i,a,e) ✓
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            real_t ai_me = OVOV(i, a, m, e);
            real_t ae_mi = OOVV(m, i, a, e);
            sigma += 1.0 * R1(m, e) * (2.0 * ai_me - ae_mi);
        }
    }

    // T4: +2 Σ_{m,n,e,f} t^{ef}_{in} r^a_m [-(me|nf) + (mf|ne)]
    //   (me|nf) = OVOV(m,e,n,f) ✓
    //   (mf|ne) = OVOV(m,f,n,e) ✓
    for (int m = 0; m < nocc; m++) {
        real_t r1_am = R1(m, a);
        real_t sum_inner = 0.0;
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                    sum_inner += T2(i,n,e,f) * K;
                }
            }
        }
        sigma += 1.0 * r1_am * sum_inner;
    }

    // T5: +1 Σ_{m,n,e,f} t^{af}_{mn} r^e_i [-(me|nf) + (mf|ne)]
    for (int e = 0; e < nvir; e++) {
        real_t r1_ei = R1(i, e);
        real_t sum_inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                    sum_inner += T2(m,n,a,f) * K;
                }
            }
        }
        sigma += 1.0 * r1_ei * sum_inner;
    }

    // T6: +1 Σ_{m,n,e} r^{ae}_{mn} [-(mi|ne) + (me|ni)]
    //   (mi|ne): m∈occ,i∈occ,n∈occ,e∈vir → OOOV(m,i,n,e) ✓
    //   (me|ni): m∈occ,e∈vir,n∈occ,i∈occ → (me|ni) = (ni|me) [bra-ket] = OOOV(n,i,m,e) ✓
    for (int m = 0; m < nocc; m++) {
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                real_t mi_ne = OOOV(m, i, n, e);
                real_t me_ni = OOOV(n, i, m, e);
                sigma += 1.0 * R2(m, n, a, e) * (-mi_ne + me_ni);
            }
        }
    }

    // T7: +2 Σ_{m,e,f} r^{ef}_{im} [(ae|mf) - (af|me)]
    //   (ae|mf): a∈vir,e∈vir,m∈occ,f∈vir → VVOV(a,e,m,f) ✓
    //   (af|me): a∈vir,f∈vir,m∈occ,e∈vir → VVOV(a,f,m,e) ✓
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            for (int f = 0; f < nvir; f++) {
                real_t ae_mf = VVOV(a, e, m, f);
                real_t af_me = VVOV(a, f, m, e);
                sigma += 2.0 * R2(i, m, e, f) * (ae_mf - af_me);
            }
        }
    }

    d_sigma1[ia] = sigma;
}


// ========================================================================
//  σ2 kernel — EOM-MP2 doubles sigma vector
// ========================================================================

/**
 * σ2 (RHF factored form, EOM_MP2_RHF.md lines 180-193):
 *
 *   σ^{ab}_{ij} =
 *     (S1)  -3 Σ_m f_mj r^{ab}_{im}
 *     (S2)  +3 Σ_e f_be r^{ae}_{ij}
 *     (S3)  +2 Σ_m r^a_m [-(mi|bj) + (mj|bi)]
 *     (S4)  +2 Σ_e r^e_j [(ai|be) - (ae|bi)]
 *     (S5)  +1.5 Σ_{m,n,e,f} t^{ef}_{ij} r^a_m [-(me|nf) + (mf|ne)]
 *     (S6)  +1.5 Σ_{m,n,e,f} t^{ab}_{mn} r^e_j [-(me|nf) + (mf|ne)]
 *     (S7)  +2 Σ_{m,n,e,f} t^{ae}_{im} r^f_n [2(me|nf) - (mf|ne)]
 *     (S8)  +1.5 Σ_{m,n} r^{ab}_{mn} [(mi|nj) - (mj|ni)]
 *     (S9)  +1 Σ_{m,e} r^{ae}_{im} [-3(mj|be) + 4(me|bj)]
 *     (S10) +1.5 Σ_{e,f} r^{ef}_{ij} [(ae|bf) - (af|be)]
 *     (S11) +3 Σ_{m,n,e,f} t^{ef}_{in} r^{ab}_{mj} [-(me|nf) + (mf|ne)]
 *     (S12) +1 Σ_{m,n,e,f} t^{ae}_{im} r^{bf}_{jn} [8(me|nf) - 5(mf|ne)]
 *     (S13) +2.5 Σ_{m,n,e,f} t^{ab}_{mn} r^{ef}_{ij} [-(me|nf) + (mf|ne)]
 *
 * In canonical MOs: S1 → -3 eps_j r^{ab}_{ij}, S2 → +3 eps_b r^{ab}_{ij}
 *
 * Note: All coefficients are ÷2 from raw spin2spatial output (spin degeneracy factor)
 */
__global__ void eom_mp2_sigma2_kernel(
    const real_t* __restrict__ d_f_oo,
    const real_t* __restrict__ d_f_vv,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oooo,
    const real_t* __restrict__ d_eri_vvvv,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_eri_ovvo,
    const real_t* __restrict__ d_t2,
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

    // S3: +2 Σ_m r^a_m [-(mi|bj) + (mj|bi)]
    //   (mi|bj): m,i∈occ, b∈vir, j∈occ → (mi|bj) = (im|jb) [swap each pair] = OVOV(i,?,j,?)
    //     Wait: (mi|bj) = (im|bj) [swap bra]. (im|bj): i,m∈occ, b∈vir, j∈occ → (oo|vo).
    //     = (bj|im) [bra-ket] = ... use OOOV: (im|jb) = (pq|rs) with p=i,q=m,r=j,s=b+nocc
    //     Actually: (mi|bj) = (im|bj) = (im|jb) [swap ket] = OOOV: ji|kb format
    //     OOOV stores (ji|kb) → to get (im|jb): need j'=i, i'=m, k=j, b'=b → OOOV(i,m,j,b) ✓
    //   (mj|bi): similarly = (jm|ib) → OOOV(j,m,i,b) ✓
    for (int m = 0; m < nocc; m++) {
        real_t mi_bj = OOOV(i, m, j, b);
        real_t mj_bi = OOOV(j, m, i, b);
        sigma += 2.0 * R1(m, a) * (-mi_bj + mj_bi);
    }

    // S4: +2 Σ_e r^e_j [(ai|be) - (ae|bi)]
    //   (ai|be): a∈vir,i∈occ,b∈vir,e∈vir → (ai|be) = (be|ai) [bra-ket]
    //     = (be|ia) [swap ket] = VVOV(b,e,i,a) ✓
    //   (ae|bi): a∈vir,e∈vir,b∈vir,i∈occ → (ae|bi) = (ae|ib) [swap ket] = VVOV(a,e,i,b) ✓
    for (int e = 0; e < nvir; e++) {
        real_t ai_be = VVOV(b, e, i, a);
        real_t ae_bi = VVOV(a, e, i, b);
        sigma += 2.0 * R1(j, e) * (ai_be - ae_bi);
    }

    // S5: +1.5 Σ_{m,n,e,f} t^{ef}_{ij} r^a_m [-(me|nf) + (mf|ne)]
    //   Depends on a,i,j only (constant in b).
    //   Precompute: K5 = Σ_{n,e,f} t^{ef}_{ij} Σ_m r^a_m [-(me|nf) + (mf|ne)]
    //   Actually, restructure: Σ_m r^a_m × Σ_{n,e,f} t^{ef}_{ij} × [-(me|nf) + (mf|ne)]
    {
        real_t sum5 = 0.0;
        for (int m = 0; m < nocc; m++) {
            real_t r1_am = R1(m, a);
            real_t inner = 0.0;
            for (int n = 0; n < nocc; n++) {
                for (int e = 0; e < nvir; e++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                        inner += T2(i,j,e,f) * K;
                    }
                }
            }
            sum5 += r1_am * inner;
        }
        sigma += 1.5 * sum5;
    }

    // S6: +1.5 Σ_{m,n,e,f} t^{ab}_{mn} r^e_j [-(me|nf) + (mf|ne)]
    //   Depends on a,b,j only (constant in i).
    {
        real_t sum6 = 0.0;
        for (int e = 0; e < nvir; e++) {
            real_t r1_ej = R1(j, e);
            real_t inner = 0.0;
            for (int m = 0; m < nocc; m++) {
                for (int n = 0; n < nocc; n++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                        inner += T2(m,n,a,b) * K;
                    }
                }
            }
            sum6 += r1_ej * inner;
        }
        sigma += 1.5 * sum6;
    }

    // S7: +2 Σ_{m,n,e,f} t^{ae}_{im} r^f_n [2(me|nf) - (mf|ne)]
    //   Depends on a,i only (constant in b,j).
    {
        real_t sum7 = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int e = 0; e < nvir; e++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = 2.0 * OVOV(m,e,n,f) - OVOV(m,f,n,e);
                        sum7 += T2(i,m,a,e) * R1(n, f) * K;
                    }
                }
            }
        }
        sigma += 2.0 * sum7;
    }

    // S8: +1.5 Σ_{m,n} r^{ab}_{mn} [(mi|nj) - (mj|ni)]
    //   (mi|nj) = OOOO(m,i,n,j) ✓
    //   (mj|ni) = OOOO(m,j,n,i) ✓
    for (int m = 0; m < nocc; m++) {
        for (int n = 0; n < nocc; n++) {
            sigma += 1.5 * R2(m, n, a, b) * (OOOO(m,i,n,j) - OOOO(m,j,n,i));
        }
    }

    // S9: +1 Σ_{m,e} r^{ae}_{im} [-3(mj|be) + 4(me|bj)]
    //   (mj|be): m,j∈occ, b,e∈vir → (mj|be) = (mj|eb) [swap ket] = OOVV(m,j,b,e)?
    //     Wait: OOVV stores (ij|ab). (mj|be) has m,j∈occ and b,e∈vir → OOVV(m,j,b,e) ✓
    //   (me|bj): m∈occ,e∈vir,b∈vir,j∈occ → OVVO(m,e,b,j) ✓
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            real_t mj_be = OOVV(m, j, b, e);
            real_t me_bj = OVVO(m, e, b, j);
            sigma += 1.0 * R2(i, m, a, e) * (-3.0 * mj_be + 4.0 * me_bj);
        }
    }

    // S10: +1.5 Σ_{e,f} r^{ef}_{ij} [(ae|bf) - (af|be)]
    //   (ae|bf) = VVVV(a,e,b,f) ✓
    //   (af|be) = VVVV(a,f,b,e) ✓
    for (int e = 0; e < nvir; e++) {
        for (int f = 0; f < nvir; f++) {
            sigma += 1.5 * R2(i, j, e, f) * (VVVV(a,e,b,f) - VVVV(a,f,b,e));
        }
    }

    // S11: +3 Σ_{m,n,e,f} t^{ef}_{in} r^{ab}_{mj} [-(me|nf) + (mf|ne)]
    {
        real_t sum11 = 0.0;
        for (int m = 0; m < nocc; m++) {
            real_t r2_ab_mj = R2(m, j, a, b);
            for (int n = 0; n < nocc; n++) {
                for (int e = 0; e < nvir; e++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                        sum11 += T2(i,n,e,f) * r2_ab_mj * K;
                    }
                }
            }
        }
        sigma += 3.0 * sum11;
    }

    // S12: +1 Σ_{m,n,e,f} t^{ae}_{im} r^{bf}_{jn} [8(me|nf) - 5(mf|ne)]
    {
        real_t sum12 = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int e = 0; e < nvir; e++) {
                    real_t t2_ae_im = T2(i,m,a,e);
                    for (int f = 0; f < nvir; f++) {
                        real_t K = 8.0 * OVOV(m,e,n,f) - 5.0 * OVOV(m,f,n,e);
                        sum12 += t2_ae_im * R2(j,n,b,f) * K;
                    }
                }
            }
        }
        sigma += 1.0 * sum12;
    }

    // S13: +2.5 Σ_{m,n,e,f} t^{ab}_{mn} r^{ef}_{ij} [-(me|nf) + (mf|ne)]
    {
        real_t sum13 = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                real_t t2_ab_mn = T2(m,n,a,b);
                for (int e = 0; e < nvir; e++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = -OVOV(m,e,n,f) + OVOV(m,f,n,e);
                        sum13 += t2_ab_mn * R2(i,j,e,f) * K;
                    }
                }
            }
        }
        sigma += 2.5 * sum13;
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
#undef T2
#undef R1
#undef R2

// ========================================================================
//  EOMMP2Operator Implementation
// ========================================================================

EOMMP2Operator::EOMMP2Operator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int nao)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      singles_dim_(nocc * nvir),
      doubles_dim_(nocc * nocc * nvir * nvir),
      total_dim_(nocc * nvir + nocc * nocc * nvir * nvir),
      d_eri_ovov_(nullptr), d_eri_vvov_(nullptr), d_eri_ooov_(nullptr),
      d_eri_oooo_(nullptr), d_eri_vvvv_(nullptr), d_eri_oovv_(nullptr),
      d_eri_ovvo_(nullptr),
      d_t2_(nullptr), d_D1_(nullptr), d_D2_(nullptr),
      d_f_oo_(nullptr), d_f_vv_(nullptr),
      d_diagonal_(nullptr), d_work1_(nullptr), d_work2_(nullptr)
{
    extract_eri_blocks(d_eri_mo);
    compute_t2_and_denominators(d_orbital_energies);
    build_diagonal();

    tracked_cudaMalloc(&d_work1_, (size_t)singles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_work2_, (size_t)doubles_dim_ * sizeof(real_t));
}

EOMMP2Operator::~EOMMP2Operator() {
    if (d_eri_ovov_) tracked_cudaFree(d_eri_ovov_);
    if (d_eri_vvov_) tracked_cudaFree(d_eri_vvov_);
    if (d_eri_ooov_) tracked_cudaFree(d_eri_ooov_);
    if (d_eri_oooo_) tracked_cudaFree(d_eri_oooo_);
    if (d_eri_vvvv_) tracked_cudaFree(d_eri_vvvv_);
    if (d_eri_oovv_) tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovvo_) tracked_cudaFree(d_eri_ovvo_);
    if (d_t2_) tracked_cudaFree(d_t2_);
    if (d_D1_) tracked_cudaFree(d_D1_);
    if (d_D2_) tracked_cudaFree(d_D2_);
    if (d_f_oo_) tracked_cudaFree(d_f_oo_);
    if (d_f_vv_) tracked_cudaFree(d_f_vv_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
    if (d_work1_) tracked_cudaFree(d_work1_);
    if (d_work2_) tracked_cudaFree(d_work2_);
}

void EOMMP2Operator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_, nvir = nvir_, nao = nao_;
    size_t nao2 = (size_t)nao * nao;

    size_t ovov_size = (size_t)nocc * nvir * nocc * nvir;
    tracked_cudaMalloc(&d_eri_ovov_, ovov_size * sizeof(real_t));

    size_t vvov_size = (size_t)nvir * nvir * nocc * nvir;
    tracked_cudaMalloc(&d_eri_vvov_, vvov_size * sizeof(real_t));

    size_t ooov_size = (size_t)nocc * nocc * nocc * nvir;
    tracked_cudaMalloc(&d_eri_ooov_, ooov_size * sizeof(real_t));

    size_t oooo_size = (size_t)nocc * nocc * nocc * nocc;
    tracked_cudaMalloc(&d_eri_oooo_, oooo_size * sizeof(real_t));

    size_t vvvv_size = (size_t)nvir * nvir * nvir * nvir;
    tracked_cudaMalloc(&d_eri_vvvv_, vvvv_size * sizeof(real_t));

    size_t oovv_size = (size_t)nocc * nocc * nvir * nvir;
    tracked_cudaMalloc(&d_eri_oovv_, oovv_size * sizeof(real_t));

    size_t ovvo_size = (size_t)nocc * nvir * nvir * nocc;
    tracked_cudaMalloc(&d_eri_ovvo_, ovvo_size * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback: extract ERI sub-blocks directly
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovov_size; idx++) {
            int i = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int a = rem / (nocc * nvir);
            rem %= (nocc * nvir);
            int j = rem / nvir;
            int b = rem % nvir;
            d_eri_ovov_[idx] = d_eri_mo[((size_t)i * nao + a + nocc) * nao2 + (size_t)j * nao + b + nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)vvov_size; idx++) {
            int a = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int b = rem / (nocc * nvir);
            rem %= (nocc * nvir);
            int i = rem / nvir;
            int c = rem % nvir;
            d_eri_vvov_[idx] = d_eri_mo[((size_t)(a+nocc) * nao + b+nocc) * nao2 + (size_t)i * nao + c+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ooov_size; idx++) {
            int j = idx / (nocc * nocc * nvir);
            int rem = idx % (nocc * nocc * nvir);
            int i = rem / (nocc * nvir);
            rem %= (nocc * nvir);
            int k = rem / nvir;
            int b = rem % nvir;
            d_eri_ooov_[idx] = d_eri_mo[((size_t)j * nao + i) * nao2 + (size_t)k * nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oooo_size; idx++) {
            int i = idx / (nocc * nocc * nocc);
            int rem = idx % (nocc * nocc * nocc);
            int j = rem / (nocc * nocc);
            rem %= (nocc * nocc);
            int k = rem / nocc;
            int l = rem % nocc;
            d_eri_oooo_[idx] = d_eri_mo[((size_t)i * nao + j) * nao2 + (size_t)k * nao + l];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)vvvv_size; idx++) {
            int a = idx / (nvir * nvir * nvir);
            int rem = idx % (nvir * nvir * nvir);
            int b = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int c = rem / nvir;
            int d = rem % nvir;
            d_eri_vvvv_[idx] = d_eri_mo[((size_t)(a+nocc)*nao + b+nocc)*nao2 + (size_t)(c+nocc)*nao + d+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oovv_size; idx++) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int a = rem / nvir;
            int b = rem % nvir;
            d_eri_oovv_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovvo_size; idx++) {
            int i = idx / (nvir * nvir * nocc);
            int rem = idx % (nvir * nvir * nocc);
            int a = rem / (nvir * nocc);
            rem %= (nvir * nocc);
            int b = rem / nocc;
            int j = rem % nocc;
            d_eri_ovvo_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
        }
    } else {
        int threads = 256;
        int blocks;

        blocks = (ovov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_, nvir_, nao_);

        blocks = (vvov_size + threads - 1) / threads;
        eom_mp2_extract_eri_vvov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvov_, nocc_, nvir_, nao_);

        blocks = (ooov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_, nvir_, nao_);

        blocks = (oooo_size + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_, nao_);

        blocks = (vvvv_size + threads - 1) / threads;
        eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_, nvir_, nao_);

        blocks = (oovv_size + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_, nvir_, nao_);

        blocks = (ovvo_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_, nvir_, nao_);

        cudaDeviceSynchronize();
    }
}

void EOMMP2Operator::compute_t2_and_denominators(const real_t* d_orbital_energies) {
    int nocc = nocc_, nvir = nvir_;

    tracked_cudaMalloc(&d_t2_, (size_t)doubles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_D2_, (size_t)doubles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_D1_, (size_t)singles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_oo_, (size_t)nocc * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv_, (size_t)nvir * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback: compute T2 amplitudes and D2 denominators
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int a = rem / nvir;
            int b = rem % nvir;
            real_t eps_i = d_orbital_energies[i];
            real_t eps_j = d_orbital_energies[j];
            real_t eps_a = d_orbital_energies[a + nocc];
            real_t eps_b = d_orbital_energies[b + nocc];
            real_t denom = eps_i + eps_j - eps_a - eps_b;
            real_t ia_jb = d_eri_ovov_[(size_t)i * nvir * nocc * nvir +
                                       (size_t)a * nocc * nvir + (size_t)j * nvir + b];
            d_t2_[idx] = ia_jb / denom;
            d_D2_[idx] = eps_a + eps_b - eps_i - eps_j;
        }
        // CPU fallback: compute D1
        #pragma omp parallel for
        for (int idx = 0; idx < singles_dim_; idx++) {
            d_D1_[idx] = d_orbital_energies[idx % nvir + nocc] - d_orbital_energies[idx / nvir];
        }
        // CPU fallback: extract Fock diagonal
        for (int idx = 0; idx < nocc; idx++)
            d_f_oo_[idx] = d_orbital_energies[idx];
        for (int idx = 0; idx < nvir; idx++)
            d_f_vv_[idx] = d_orbital_energies[idx + nocc];
    } else {
        int threads = 256;

        int blocks = (doubles_dim_ + threads - 1) / threads;
        eom_mp2_compute_t2_D2_kernel<<<blocks, threads>>>(
            d_eri_ovov_, d_orbital_energies, d_t2_, d_D2_, nocc_, nvir_);

        blocks = (singles_dim_ + threads - 1) / threads;
        eom_mp2_compute_D1_kernel<<<blocks, threads>>>(
            d_orbital_energies, d_D1_, nocc_, nvir_);

        int nao = nocc + nvir;
        blocks = (nao + threads - 1) / threads;
        eom_mp2_extract_fock_kernel<<<blocks, threads>>>(
            d_orbital_energies, d_f_oo_, d_f_vv_, nocc_, nvir_);

        cudaDeviceSynchronize();
    }
}

void EOMMP2Operator::build_diagonal() {
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; idx++) {
            d_diagonal_[idx] = (idx < singles_dim_) ? d_D1_[idx] : d_D2_[idx - singles_dim_];
        }
    } else {
        int threads = 256;
        int blocks = (total_dim_ + threads - 1) / threads;
        eom_mp2_build_diagonal_kernel<<<blocks, threads>>>(
            d_D1_, d_D2_, d_diagonal_, singles_dim_, doubles_dim_);
        cudaDeviceSynchronize();
    }
}

void EOMMP2Operator::apply(const real_t* d_input, real_t* d_output) const {
    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + singles_dim_;
    real_t* d_sigma1 = d_output;
    real_t* d_sigma2 = d_output + singles_dim_;

    if (!gpu::gpu_available()) {
        int nocc = nocc_, nvir = nvir_;
        // Macros for CPU access (same layout as GPU kernels)
        #define CPU_OVOV(i,a,j,b) d_eri_ovov_[(size_t)(i)*nvir*nocc*nvir + (size_t)(a)*nocc*nvir + (size_t)(j)*nvir + (b)]
        #define CPU_VVOV(a,b,i,c) d_eri_vvov_[(size_t)(a)*nvir*nocc*nvir + (size_t)(b)*nocc*nvir + (size_t)(i)*nvir + (c)]
        #define CPU_OOOV(j,i,k,b) d_eri_ooov_[(size_t)(j)*nocc*nocc*nvir + (size_t)(i)*nocc*nvir + (size_t)(k)*nvir + (b)]
        #define CPU_OOOO(i,j,k,l) d_eri_oooo_[(size_t)(i)*nocc*nocc*nocc + (size_t)(j)*nocc*nocc + (size_t)(k)*nocc + (l)]
        #define CPU_VVVV(a,b,c,d) d_eri_vvvv_[(size_t)(a)*nvir*nvir*nvir + (size_t)(b)*nvir*nvir + (size_t)(c)*nvir + (d)]
        #define CPU_OOVV(i,j,a,b) d_eri_oovv_[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
        #define CPU_OVVO(i,a,b,j) d_eri_ovvo_[(size_t)(i)*nvir*nvir*nocc + (size_t)(a)*nvir*nocc + (size_t)(b)*nocc + (j)]
        #define CPU_T2(i,j,a,b) d_t2_[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
        #define CPU_R1(i,a) d_r1[(i)*nvir + (a)]
        #define CPU_R2(i,j,a,b) d_r2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

        // CPU sigma1
        #pragma omp parallel for
        for (int ia = 0; ia < singles_dim_; ia++) {
            int i = ia / nvir;
            int a = ia % nvir;
            real_t sigma = 0.0;
            sigma += -1.0 * d_f_oo_[i] * CPU_R1(i, a);
            sigma += 1.0 * d_f_vv_[a] * CPU_R1(i, a);
            for (int m = 0; m < nocc; m++)
                for (int e = 0; e < nvir; e++)
                    sigma += 1.0 * CPU_R1(m, e) * (2.0 * CPU_OVOV(i, a, m, e) - CPU_OOVV(m, i, a, e));
            for (int m = 0; m < nocc; m++) {
                real_t r1_am = CPU_R1(m, a);
                real_t sum_inner = 0.0;
                for (int n = 0; n < nocc; n++)
                    for (int e = 0; e < nvir; e++)
                        for (int f = 0; f < nvir; f++)
                            sum_inner += CPU_T2(i,n,e,f) * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                sigma += 1.0 * r1_am * sum_inner;
            }
            for (int e = 0; e < nvir; e++) {
                real_t r1_ei = CPU_R1(i, e);
                real_t sum_inner = 0.0;
                for (int m = 0; m < nocc; m++)
                    for (int n = 0; n < nocc; n++)
                        for (int f = 0; f < nvir; f++)
                            sum_inner += CPU_T2(m,n,a,f) * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                sigma += 1.0 * r1_ei * sum_inner;
            }
            for (int m = 0; m < nocc; m++)
                for (int n = 0; n < nocc; n++)
                    for (int e = 0; e < nvir; e++)
                        sigma += 1.0 * CPU_R2(m, n, a, e) * (-CPU_OOOV(m, i, n, e) + CPU_OOOV(n, i, m, e));
            for (int m = 0; m < nocc; m++)
                for (int e = 0; e < nvir; e++)
                    for (int f = 0; f < nvir; f++)
                        sigma += 2.0 * CPU_R2(i, m, e, f) * (CPU_VVOV(a, e, m, f) - CPU_VVOV(a, f, m, e));
            d_sigma1[ia] = sigma;
        }

        // CPU sigma2
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int a = rem / nvir;
            int b = rem % nvir;
            real_t sigma = 0.0;
            sigma += -3.0 * d_f_oo_[j] * CPU_R2(i, j, a, b);
            sigma += 3.0 * d_f_vv_[b] * CPU_R2(i, j, a, b);
            for (int m = 0; m < nocc; m++)
                sigma += 2.0 * CPU_R1(m, a) * (-CPU_OOOV(i, m, j, b) + CPU_OOOV(j, m, i, b));
            for (int e = 0; e < nvir; e++)
                sigma += 2.0 * CPU_R1(j, e) * (CPU_VVOV(b, e, i, a) - CPU_VVOV(a, e, i, b));
            { // S5
                real_t sum5 = 0.0;
                for (int m = 0; m < nocc; m++) {
                    real_t r1_am = CPU_R1(m, a);
                    real_t inner = 0.0;
                    for (int n = 0; n < nocc; n++)
                        for (int e = 0; e < nvir; e++)
                            for (int f = 0; f < nvir; f++)
                                inner += CPU_T2(i,j,e,f) * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                    sum5 += r1_am * inner;
                }
                sigma += 1.5 * sum5;
            }
            { // S6
                real_t sum6 = 0.0;
                for (int e = 0; e < nvir; e++) {
                    real_t r1_ej = CPU_R1(j, e);
                    real_t inner = 0.0;
                    for (int m = 0; m < nocc; m++)
                        for (int n = 0; n < nocc; n++)
                            for (int f = 0; f < nvir; f++)
                                inner += CPU_T2(m,n,a,b) * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                    sum6 += r1_ej * inner;
                }
                sigma += 1.5 * sum6;
            }
            { // S7
                real_t sum7 = 0.0;
                for (int m = 0; m < nocc; m++)
                    for (int n = 0; n < nocc; n++)
                        for (int e = 0; e < nvir; e++)
                            for (int f = 0; f < nvir; f++)
                                sum7 += CPU_T2(i,m,a,e) * CPU_R1(n, f) * (2.0 * CPU_OVOV(m,e,n,f) - CPU_OVOV(m,f,n,e));
                sigma += 2.0 * sum7;
            }
            for (int m = 0; m < nocc; m++)
                for (int n = 0; n < nocc; n++)
                    sigma += 1.5 * CPU_R2(m, n, a, b) * (CPU_OOOO(m,i,n,j) - CPU_OOOO(m,j,n,i));
            for (int m = 0; m < nocc; m++)
                for (int e = 0; e < nvir; e++)
                    sigma += 1.0 * CPU_R2(i, m, a, e) * (-3.0 * CPU_OOVV(m, j, b, e) + 4.0 * CPU_OVVO(m, e, b, j));
            for (int e = 0; e < nvir; e++)
                for (int f = 0; f < nvir; f++)
                    sigma += 1.5 * CPU_R2(i, j, e, f) * (CPU_VVVV(a,e,b,f) - CPU_VVVV(a,f,b,e));
            { // S11
                real_t sum11 = 0.0;
                for (int m = 0; m < nocc; m++) {
                    real_t r2_ab_mj = CPU_R2(m, j, a, b);
                    for (int n = 0; n < nocc; n++)
                        for (int e = 0; e < nvir; e++)
                            for (int f = 0; f < nvir; f++)
                                sum11 += CPU_T2(i,n,e,f) * r2_ab_mj * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                }
                sigma += 3.0 * sum11;
            }
            { // S12
                real_t sum12 = 0.0;
                for (int m = 0; m < nocc; m++)
                    for (int n = 0; n < nocc; n++)
                        for (int e = 0; e < nvir; e++) {
                            real_t t2_ae_im = CPU_T2(i,m,a,e);
                            for (int f = 0; f < nvir; f++)
                                sum12 += t2_ae_im * CPU_R2(j,n,b,f) * (8.0 * CPU_OVOV(m,e,n,f) - 5.0 * CPU_OVOV(m,f,n,e));
                        }
                sigma += 1.0 * sum12;
            }
            { // S13
                real_t sum13 = 0.0;
                for (int m = 0; m < nocc; m++)
                    for (int n = 0; n < nocc; n++) {
                        real_t t2_ab_mn = CPU_T2(m,n,a,b);
                        for (int e = 0; e < nvir; e++)
                            for (int f = 0; f < nvir; f++)
                                sum13 += t2_ab_mn * CPU_R2(i,j,e,f) * (-CPU_OVOV(m,e,n,f) + CPU_OVOV(m,f,n,e));
                    }
                sigma += 2.5 * sum13;
            }
            d_sigma2[idx] = sigma;
        }

        #undef CPU_OVOV
        #undef CPU_VVOV
        #undef CPU_OOOV
        #undef CPU_OOOO
        #undef CPU_VVVV
        #undef CPU_OOVV
        #undef CPU_OVVO
        #undef CPU_T2
        #undef CPU_R1
        #undef CPU_R2
    } else {
        int threads = 256;

        int blocks1 = (singles_dim_ + threads - 1) / threads;
        eom_mp2_sigma1_kernel<<<blocks1, threads>>>(
            d_f_oo_, d_f_vv_,
            d_eri_ovov_, d_eri_vvov_, d_eri_ooov_, d_eri_oovv_,
            d_t2_, d_r1, d_r2, d_sigma1,
            nocc_, nvir_);

        int blocks2 = (doubles_dim_ + threads - 1) / threads;
        eom_mp2_sigma2_kernel<<<blocks2, threads>>>(
            d_f_oo_, d_f_vv_,
            d_eri_ovov_, d_eri_vvov_, d_eri_ooov_,
            d_eri_oooo_, d_eri_vvvv_, d_eri_oovv_, d_eri_ovvo_,
            d_t2_, d_r1, d_r2, d_sigma2,
            nocc_, nvir_);

        cudaDeviceSynchronize();
    }
}

void EOMMP2Operator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; idx++) {
            real_t diag = d_diagonal_[idx];
            d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
        }
    } else {
        int threads = 256;
        int blocks = (total_dim_ + threads - 1) / threads;
        eom_mp2_preconditioner_kernel<<<blocks, threads>>>(
            d_diagonal_, d_input, d_output, total_dim_);
    }
}

} // namespace gansu
