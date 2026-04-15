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
 * @file eom_ccsd_operator.cu
 * @brief GPU implementation of EOM-CCSD operator in full singles+doubles space
 *
 * All coefficients use ÷2 convention (spin degeneracy factor divided out).
 *
 * σ1 (8 terms using dressed intermediates, matches PySCF eeccsd_matvec_singlet):
 *   T1-T2:  dressed Fock (Fvv, Foo) × r1
 *   T3-T4:  Fov × r2
 *   T5:     ovvv × θ_r2
 *   T6:     (0.5*WoVVo + WoVvO) × r1
 *   T7:     woOoV × θ_r2
 *   T8:     t1 × (ovov × θ_r2)
 *
 * σ2 (EOM_CCSD_RHF.md, ÷2, 21 grouped terms):
 *   S1-S2:   Fock diagonal on R2
 *   S3-S4:   ERI × R1 (OOOO-like, VVVV-like)
 *   S5-S6:   T1 × R1 × ERI
 *   S7:      T1² × R1 × ERI
 *   S8-S10:  T2 × R1 × ERI
 *   S11-S13: R2 × ERI (OOOO, OOVV/OVVO, VVVV)
 *   S14-S17: T1 × R2 × ERI
 *   S18:     T1² × R2 × ERI
 *   S19-S21: T2 × R2 × ERI
 *
 * ERI block mapping (chemist notation symmetries used):
 *   OOOV(m,i,j,b) = (mi|jb),  with (mi|bj)=(mi|jb) by (pq|sr)=(pq|rs)
 *   OVOV(i,a,j,b) = (ia|jb)
 *   VVOV(a,b,i,c) = (ab|ic)
 *   OOVV(i,j,a,b) = (ij|ab)
 *   OVVO(i,a,b,j) = (ia|bj)
 *   OOOO(i,j,k,l) = (ij|kl)
 *   VVVV(a,b,c,d) = (ab|cd)
 *   OVVV(i,a,b,c) = (ia|bc)  — NEW for EOM-CCSD
 *
 *   Derived blocks via symmetry:
 *   VOVV(a,i,b,e) = (ai|be) = (ia|be) = OVVV(i,a,b,e)
 *   VVVO(a,e,b,i) = (ae|bi) = (ib|ae) = OVVV(i,b,a,e)
 *   OOVO(m,i,b,j) = (mi|bj) = (mi|jb) = OOOV(m,i,j,b)
 *   OVOO(m,e,n,i) = (me|ni) = (ni|me) = OOOV(n,i,m,e) [bra-ket]
 */

#include <cstdio>
#include <cmath>
#include <vector>

#include "eom_ccsd_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// Reuse EOM-MP2/CC2 ERI extraction kernels (declared extern)
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


// ========================================================================
//  OVVV extraction kernel — NEW for EOM-CCSD
// ========================================================================

/**
 * Extract OVVV block: eri_ovvv[i,a,b,c] = eri_mo[(i)*N³ + (a+nocc)*N² + (b+nocc)*N + (c+nocc)]
 * = (ia|bc) in chemist notation, where i ∈ occ, a,b,c ∈ vir
 */
__global__ void eom_ccsd_extract_eri_ovvv_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t* __restrict__ d_eri_ovvv,
    int nocc, int nvir, int nao)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nocc * nvir * nvir * nvir;
    if (idx >= total) return;

    int i = (int)(idx / ((size_t)nvir * nvir * nvir));
    size_t rem = idx % ((size_t)nvir * nvir * nvir);
    int a = (int)(rem / ((size_t)nvir * nvir));
    rem %= ((size_t)nvir * nvir);
    int b = (int)(rem / nvir);
    int c = (int)(rem % nvir);

    size_t N = (size_t)nao;
    size_t mo_idx = (size_t)i * N * N * N
                  + (size_t)(a + nocc) * N * N
                  + (size_t)(b + nocc) * N
                  + (size_t)(c + nocc);

    d_eri_ovvv[idx] = d_eri_mo[mo_idx];
}

// ========================================================================
//  D2 computation for EOM-CCSD: standard orbital energy denominator
// ========================================================================

/**
 * D2[ijab] = ε_a + ε_b - ε_i - ε_j  (standard orbital energy denominator)
 * Used for preconditioner in Davidson solver.
 */
__global__ void eom_ccsd_compute_D2_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_D2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    d_D2[idx] = d_orbital_energies[a + nocc] + d_orbital_energies[b + nocc]
              - d_orbital_energies[i] - d_orbital_energies[j];
}

// ========================================================================
//  Response tau kernel: tau2[ijab] = r2[ijab] + r1[ia]*t1[jb] + t1[ia]*r1[jb]
// ========================================================================
__global__ void eom_ccsd_response_tau_kernel(
    const real_t* __restrict__ d_r1,
    const real_t* __restrict__ d_r2,
    const real_t* __restrict__ d_t1,
    real_t* __restrict__ d_tau2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    d_tau2[idx] = d_r2[idx]
                + d_r1[i * nvir + a] * d_t1[j * nvir + b]
                + d_t1[i * nvir + a] * d_r1[j * nvir + b];
}

// ========================================================================
//  Symmetrize kernel: sigma2[ijab] = half[ijab] + half[jiba]
// ========================================================================
__global__ void eom_ccsd_symmetrize_kernel(
    const real_t* __restrict__ d_half,
    real_t* __restrict__ d_sigma2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    // jiba index
    size_t jiba = (size_t)j * nocc * nvir * nvir
                + (size_t)i * nvir * nvir
                + (size_t)b * nvir + a;

    d_sigma2[idx] = d_half[idx] + d_half[jiba];
}

// Symmetrize input r2 for singlet: r2_sym[ijab] = 0.5*(r2[ijab] + r2[jiba])
__global__ void eom_ccsd_symmetrize_r2_kernel(
    const real_t* __restrict__ d_r2,
    real_t* __restrict__ d_r2_sym,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    size_t jiba = (size_t)j * nocc * nvir * nvir
                + (size_t)i * nvir * nvir
                + (size_t)b * nvir + a;

    d_r2_sym[idx] = 0.5 * (d_r2[idx] + d_r2[jiba]);
}

// ========================================================================
//  ERI/amplitude access macros for sigma2 kernel
// ========================================================================

#define OVOV(i,a,j,b) d_eri_ovov[(size_t)(i)*nvir*nocc*nvir + (size_t)(a)*nocc*nvir + (size_t)(j)*nvir + (b)]
#define VVVV(a,b,c,d_) d_eri_vvvv[(size_t)(a)*nvir*nvir*nvir + (size_t)(b)*nvir*nvir + (size_t)(c)*nvir + (d_)]
#define OVVV(i,a,b,c) d_eri_ovvv[(size_t)(i)*nvir*nvir*nvir + (size_t)(a)*nvir*nvir + (size_t)(b)*nvir + (c)]
#define T1(i,a) d_t1[(i)*nvir + (a)]
#define T2(i,j,a,b) d_t2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define R1(i,a) d_r1[(i)*nvir + (a)]
#define R2(i,j,a,b) d_r2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define TAU2(i,j,a,b) d_tau2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// Dressed intermediate access
#define FOO(m,i) d_Foo[(m)*nocc + (i)]
#define FVV(a,e) d_Fvv[(a)*nvir + (e)]
#define WOOOO(m,n,i,j) d_Woooo[(size_t)(m)*nocc*nocc*nocc + (size_t)(n)*nocc*nocc + (size_t)(i)*nocc + (j)]
#define WoVVo(m,b,e,j) d_WoVVo[(size_t)(m)*nvir*nvir*nocc + (size_t)(b)*nvir*nocc + (size_t)(e)*nocc + (j)]
#define WoVvO(m,b,e,j) d_WoVvO[(size_t)(m)*nvir*nvir*nocc + (size_t)(b)*nvir*nocc + (size_t)(e)*nocc + (j)]
#define woVoO(m,b,i,j) d_woVoO[(size_t)(m)*nvir*nocc*nocc + (size_t)(b)*nocc*nocc + (size_t)(i)*nocc + (j)]
#define wvOvV(e,j,a,b) d_wvOvV[(size_t)(e)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define woOoV(m,n,i,e) d_woOoV[(size_t)(m)*nocc*nocc*nvir + (size_t)(n)*nocc*nvir + (size_t)(i)*nvir + (e)]

// Precomputed R-dependent small intermediates
#define AF_TMP(a,f) d_af_tmp[(a)*nvir + (f)]
#define NI_TMP(n,i) d_ni_tmp[(n)*nocc + (i)]
#define EB_TMP(e,b) d_eb_tmp[(e)*nvir + (b)]
#define NI_TMP2(n,i) d_ni_tmp2[(n)*nocc + (i)]
#define MNIJ_TMP(m,n,i,j) d_mnij_tmp[(size_t)(m)*nocc*nocc*nocc + (size_t)(n)*nocc*nocc + (size_t)(i)*nocc + (j)]
#define TAU_HALF(m,n,a,b) d_tau_half[(size_t)(m)*nocc*nvir*nvir + (size_t)(n)*nvir*nvir + (size_t)(a)*nvir + (b)]

// ========================================================================
//  R-dependent small intermediates (af_tmp, ni_tmp, eb_tmp, ni_tmp2,
//  mnij_tmp, tau_half). Each kernel parallelizes over output indices and
//  reduces internally — direct port of the host loops.
// ========================================================================

__global__ void eom_ccsd_af_tmp_kernel(
    const real_t* __restrict__ d_eri_ovvv,
    const real_t* __restrict__ d_r1,
    real_t* __restrict__ d_af_tmp,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir) return;
    int a = idx / nvir;
    int f = idx % nvir;
    real_t val = 0.0;
    for (int m = 0; m < nocc; m++)
        for (int e = 0; e < nvir; e++) {
            real_t r = R1(m, e);
            val += (2.0 * OVVV(m, e, a, f) - OVVV(m, f, a, e)) * r;
        }
    d_af_tmp[idx] = val;
}

__global__ void eom_ccsd_ni_tmp_kernel(
    const real_t* __restrict__ d_woOoV,
    const real_t* __restrict__ d_r1,
    real_t* __restrict__ d_ni_tmp,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc) return;
    int n = idx / nocc;
    int i = idx % nocc;
    real_t val = 0.0;
    for (int m = 0; m < nocc; m++)
        for (int e = 0; e < nvir; e++) {
            real_t r = R1(m, e);
            val += (2.0 * woOoV(n, m, i, e) - woOoV(m, n, i, e)) * r;
        }
    d_ni_tmp[idx] = val;
}

// eb_tmp[e,b] = Σ_n en_tmp[e,n]*t1[n,b]
//             + Σ_{m,n,f} ovov[m,e,n,f] * (2*r2[m,n,b,f] - r2[m,n,f,b])
//   where en_tmp[e,n] = Σ_{m,f} (2*ovov[m,f,n,e] - ovov[m,e,n,f]) * r1[m,f]
__global__ void eom_ccsd_eb_tmp_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_r1,
    const real_t* __restrict__ d_r2,
    real_t* __restrict__ d_eb_tmp,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nvir * nvir) return;
    int e = idx / nvir;
    int b = idx % nvir;
    real_t val = 0.0;
    // Part A: Σ_n en_tmp[e,n]*t1[n,b]
    for (int n = 0; n < nocc; n++) {
        real_t en_tmp = 0.0;
        for (int m = 0; m < nocc; m++)
            for (int f = 0; f < nvir; f++) {
                real_t r = R1(m, f);
                en_tmp += (2.0 * OVOV(m, f, n, e) - OVOV(m, e, n, f)) * r;
            }
        val += en_tmp * T1(n, b);
    }
    // Part B: Σ_{m,n,f} ovov[m,e,n,f] * theta_r[m,n,b,f]
    for (int m = 0; m < nocc; m++)
        for (int n = 0; n < nocc; n++)
            for (int f = 0; f < nvir; f++) {
                real_t theta = 2.0 * R2(m, n, b, f) - R2(m, n, f, b);
                val += OVOV(m, e, n, f) * theta;
            }
    d_eb_tmp[idx] = val;
}

__global__ void eom_ccsd_ni_tmp2_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_r2,
    real_t* __restrict__ d_ni_tmp2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc) return;
    int n = idx / nocc;
    int i = idx % nocc;
    real_t val = 0.0;
    for (int e = 0; e < nvir; e++)
        for (int m = 0; m < nocc; m++)
            for (int f = 0; f < nvir; f++) {
                real_t theta = 2.0 * R2(i, m, e, f) - R2(i, m, f, e);
                val += OVOV(n, e, m, f) * theta;
            }
    d_ni_tmp2[idx] = val;
}

__global__ void eom_ccsd_mnij_tmp_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_tau2,
    real_t* __restrict__ d_mnij_tmp,
    int nocc, int nvir)
{
    size_t total = (size_t)nocc * nocc * nocc * nocc;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = idx / (nocc * nocc * nocc);
    int rem = idx % (nocc * nocc * nocc);
    int n = rem / (nocc * nocc);
    rem %= (nocc * nocc);
    int i = rem / nocc;
    int j = rem % nocc;
    real_t val = 0.0;
    for (int e = 0; e < nvir; e++)
        for (int f = 0; f < nvir; f++)
            val += OVOV(m, e, n, f) * TAU2(i, j, e, f);
    d_mnij_tmp[idx] = val;
}

// tau_half[m,n,a,b] = 0.5*t2[m,n,a,b] + 0.25*(t1[m,a]*t1[n,b] + t1[n,a]*t1[m,b])
__global__ void eom_ccsd_tau_half_kernel(
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_t2,
    real_t* __restrict__ d_tau_half,
    int nocc, int nvir)
{
    size_t total = (size_t)nocc * nocc * nvir * nvir;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int m = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int n = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;
    d_tau_half[idx] = 0.5 * T2(m, n, a, b)
                    + 0.25 * (T1(m, a) * T1(n, b) + T1(n, a) * T1(m, b));
}

// ========================================================================
//  Half σ2 kernel — PySCF eeccsd_matvec_singlet algorithm
// ========================================================================

/**
 * Computes "half" σ2 following PySCF's algorithm.
 * Final σ2 is obtained by symmetrizing: σ2[ijab] = half[ijab] + half[jiba].
 *
 * Uses 8 precomputed dressed intermediates (Foo, Fvv, Woooo, WoVVo, WoVvO,
 * woOoV, woVoO, wvOvV) and 6 R-dependent small intermediates
 * (af_tmp, ni_tmp, eb_tmp, ni_tmp2, mnij_tmp, tau_half).
 */
__global__ void eom_ccsd_half_sigma2_kernel(
    // Raw ERI blocks
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvvv,
    const real_t* __restrict__ d_eri_ovvv,
    // Amplitudes
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_t2,
    // R vectors
    const real_t* __restrict__ d_r1,
    const real_t* __restrict__ d_r2,
    // Response tau (precomputed)
    const real_t* __restrict__ d_tau2,
    // Dressed intermediates (precomputed in constructor)
    const real_t* __restrict__ d_Foo,
    const real_t* __restrict__ d_Fvv,
    const real_t* __restrict__ d_Woooo,
    const real_t* __restrict__ d_WoVVo,
    const real_t* __restrict__ d_WoVvO,
    const real_t* __restrict__ d_woOoV,
    const real_t* __restrict__ d_woVoO,
    const real_t* __restrict__ d_wvOvV,
    // R-dependent small intermediates (precomputed per apply)
    const real_t* __restrict__ d_af_tmp,
    const real_t* __restrict__ d_ni_tmp,
    const real_t* __restrict__ d_eb_tmp,
    const real_t* __restrict__ d_ni_tmp2,
    const real_t* __restrict__ d_mnij_tmp,
    const real_t* __restrict__ d_tau_half,
    // Output
    real_t* __restrict__ d_half_sigma2,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    real_t hr2 = 0.0;

    // === Term 1: 0.5 * VVVV * tau2 ===
    // Hr2[i,j,a,b] += 0.5 * Σ_{e,f} tau2[i,j,e,f] * vvvv[a,e,b,f]
    {
        real_t sum = 0.0;
        for (int e = 0; e < nvir; e++)
            for (int f = 0; f < nvir; f++)
                sum += TAU2(i, j, e, f) * VVVV(a, e, b, f);
        hr2 += 0.5 * sum;
    }

    // === Term 2: 0.5 * Woooo * r2 ===
    // Hr2[i,j,a,b] += 0.5 * Σ_{m,n} Woooo[m,n,i,j] * r2[m,n,a,b]
    {
        real_t sum = 0.0;
        for (int m = 0; m < nocc; m++)
            for (int n = 0; n < nocc; n++)
                sum += WOOOO(m, n, i, j) * R2(m, n, a, b);
        hr2 += 0.5 * sum;
    }

    // === Term 3: Fvv * r2 ===
    // Hr2[i,j,a,b] += Σ_e Fvv[b,e] * r2[i,j,a,e]
    for (int e = 0; e < nvir; e++)
        hr2 += FVV(b, e) * R2(i, j, a, e);

    // === Term 4: -Foo * r2 ===
    // Hr2[i,j,a,b] -= Σ_m Foo[m,j] * r2[i,m,a,b]
    for (int m = 0; m < nocc; m++)
        hr2 -= FOO(m, j) * R2(i, m, a, b);

    // === Term 5: -t1 * (ovvv * tau2) ===
    // Hr2[i,j,a,b] -= Σ_m t1[m,a] * Σ_{e,f} ovvv[m,e,b,f] * tau2[i,j,e,f]
    for (int m = 0; m < nocc; m++) {
        real_t inner = 0.0;
        for (int e = 0; e < nvir; e++)
            for (int f = 0; f < nvir; f++)
                inner += OVVV(m, e, b, f) * TAU2(i, j, e, f);
        hr2 -= T1(m, a) * inner;
    }

    // === Term 6: af_tmp * t2 ===
    // Hr2[i,j,a,b] += Σ_f af_tmp[a,f] * t2[i,j,f,b]
    for (int f = 0; f < nvir; f++)
        hr2 += AF_TMP(a, f) * T2(i, j, f, b);

    // === Term 7: -woVoO * r1 ===
    // Hr2[i,j,a,b] -= Σ_m woVoO[m,b,i,j] * r1[m,a]
    for (int m = 0; m < nocc; m++)
        hr2 -= woVoO(m, b, i, j) * R1(m, a);

    // === Term 8: +wvOvV * r1 ===
    // Hr2[i,j,a,b] += Σ_e wvOvV[e,j,a,b] * r1[i,e]
    for (int e = 0; e < nvir; e++)
        hr2 += wvOvV(e, j, a, b) * R1(i, e);

    // === Term 9: WoVVo * r2 (with 0.5*transpose) ===
    // tmp[j,i,a,b] = Σ_{m,e} WoVVo[m,b,e,i] * r2[j,m,e,a]
    // Hr2 += tmp + 0.5 * tmp.T(0,1,3,2)
    // For element [i,j,a,b]: Hr2 += Σ_{m,e} WoVVo[m,b,e,i]*r2[j,m,e,a]
    //                             + 0.5*Σ_{m,e} WoVVo[m,a,e,i]*r2[j,m,e,b]
    {
        real_t sum_ab = 0.0, sum_ba = 0.0;
        for (int m = 0; m < nocc; m++)
            for (int e = 0; e < nvir; e++) {
                sum_ab += WoVVo(m, b, e, i) * R2(j, m, e, a);
                sum_ba += WoVVo(m, a, e, i) * R2(j, m, e, b);
            }
        hr2 += sum_ab + 0.5 * sum_ba;
    }

    // === Term 10: (0.5*WoVVo + WoVvO) * theta ===
    // theta[i,m,a,e] = 2*r2[i,m,a,e] - r2[i,m,e,a]
    // Hr2[i,j,a,b] += Σ_{m,e} combined[m,b,e,j] * theta[i,m,a,e]
    for (int m = 0; m < nocc; m++)
        for (int e = 0; e < nvir; e++) {
            real_t combined = 0.5 * WoVVo(m, b, e, j) + WoVvO(m, b, e, j);
            real_t theta_imae = 2.0 * R2(i, m, a, e) - R2(i, m, e, a);
            hr2 += combined * theta_imae;
        }

    // === Term 11: -ni_tmp * t2 ===
    // Hr2[i,j,a,b] -= Σ_n ni_tmp[n,i] * t2[n,j,a,b]
    for (int n = 0; n < nocc; n++)
        hr2 -= NI_TMP(n, i) * T2(n, j, a, b);

    // === Term 12: -eb_tmp * t2 ===
    // Hr2[i,j,a,b] -= Σ_e eb_tmp[e,b] * t2[j,i,e,a]
    for (int e = 0; e < nvir; e++)
        hr2 -= EB_TMP(e, b) * T2(j, i, e, a);

    // === Term 13: -ni_tmp2 * t2 (→ ijba output) ===
    // Hr2[i,j,a,b] -= Σ_m ni_tmp2[m,j] * t2[m,i,b,a]
    for (int m = 0; m < nocc; m++)
        hr2 -= NI_TMP2(m, j) * T2(m, i, b, a);

    // === Term 14: mnij_tmp * tau_half ===
    // Hr2[i,j,a,b] += Σ_{m,n} mnij_tmp[m,n,i,j] * tau_half[m,n,a,b]
    for (int m = 0; m < nocc; m++)
        for (int n = 0; n < nocc; n++)
            hr2 += MNIJ_TMP(m, n, i, j) * TAU_HALF(m, n, a, b);

    d_half_sigma2[idx] = hr2;
}

#undef OVOV
#undef VVVV
#undef OVVV
#undef T1
#undef T2
#undef R1
#undef R2
#undef TAU2
#undef FOO
#undef FVV
#undef WOOOO
#undef WoVVo
#undef WoVvO
#undef woVoO
#undef wvOvV
#undef woOoV
#undef AF_TMP
#undef NI_TMP
#undef EB_TMP
#undef NI_TMP2
#undef MNIJ_TMP
#undef TAU_HALF

// ========================================================================
//  EOMCCSDOperator Implementation
// ========================================================================

EOMCCSDOperator::EOMCCSDOperator(
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
      d_eri_ovvo_(nullptr), d_eri_ovvv_(nullptr),
      d_Foo_(nullptr), d_Fvv_(nullptr), d_Fov_(nullptr), d_Woooo_(nullptr),
      d_WoVVo_(nullptr), d_WoVvO_(nullptr), d_woOoV_(nullptr),
      d_woVoO_(nullptr), d_wvOvV_(nullptr),
      d_half_sigma2_(nullptr),
      d_D1_(nullptr), d_D2_(nullptr),
      d_f_oo_(nullptr), d_f_vv_(nullptr),
      d_diagonal_(nullptr)
{
    extract_eri_blocks(d_eri_mo);
    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
    build_dressed_intermediates();
}

EOMCCSDOperator::~EOMCCSDOperator() {
    if (d_t1_) tracked_cudaFree(d_t1_);
    if (d_t2_) tracked_cudaFree(d_t2_);
    if (d_eri_ovov_) tracked_cudaFree(d_eri_ovov_);
    if (d_eri_vvov_) tracked_cudaFree(d_eri_vvov_);
    if (d_eri_ooov_) tracked_cudaFree(d_eri_ooov_);
    if (d_eri_oooo_) tracked_cudaFree(d_eri_oooo_);
    if (d_eri_vvvv_) tracked_cudaFree(d_eri_vvvv_);
    if (d_eri_oovv_) tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovvo_) tracked_cudaFree(d_eri_ovvo_);
    if (d_eri_ovvv_) tracked_cudaFree(d_eri_ovvv_);
    if (d_Foo_) tracked_cudaFree(d_Foo_);
    if (d_Fvv_) tracked_cudaFree(d_Fvv_);
    if (d_Fov_) tracked_cudaFree(d_Fov_);
    if (d_Woooo_) tracked_cudaFree(d_Woooo_);
    if (d_WoVVo_) tracked_cudaFree(d_WoVVo_);
    if (d_WoVvO_) tracked_cudaFree(d_WoVvO_);
    if (d_woOoV_) tracked_cudaFree(d_woOoV_);
    if (d_woVoO_) tracked_cudaFree(d_woVoO_);
    if (d_wvOvV_) tracked_cudaFree(d_wvOvV_);
    if (d_half_sigma2_) tracked_cudaFree(d_half_sigma2_);
    if (d_D1_) tracked_cudaFree(d_D1_);
    if (d_D2_) tracked_cudaFree(d_D2_);
    if (d_f_oo_) tracked_cudaFree(d_f_oo_);
    if (d_f_vv_) tracked_cudaFree(d_f_vv_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void EOMCCSDOperator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_, nvir = nvir_, nao = nao_;
    size_t nao2 = (size_t)nao * nao;
    size_t N = (size_t)nao;

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

    size_t ovvv_size = (size_t)nocc * nvir * nvir * nvir;
    tracked_cudaMalloc(&d_eri_ovvv_, ovvv_size * sizeof(real_t));

    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovov_size; idx++) {
            int i = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int a = rem / (nocc * nvir); rem %= (nocc * nvir);
            int j = rem / nvir; int b = rem % nvir;
            d_eri_ovov_[idx] = d_eri_mo[((size_t)i * nao + a + nocc) * nao2 + (size_t)j * nao + b + nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)vvov_size; idx++) {
            int a = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int b = rem / (nocc * nvir); rem %= (nocc * nvir);
            int i = rem / nvir; int c = rem % nvir;
            d_eri_vvov_[idx] = d_eri_mo[((size_t)(a+nocc) * nao + b+nocc) * nao2 + (size_t)i * nao + c+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ooov_size; idx++) {
            int j = idx / (nocc * nocc * nvir);
            int rem = idx % (nocc * nocc * nvir);
            int i = rem / (nocc * nvir); rem %= (nocc * nvir);
            int k = rem / nvir; int b = rem % nvir;
            d_eri_ooov_[idx] = d_eri_mo[((size_t)j * nao + i) * nao2 + (size_t)k * nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oooo_size; idx++) {
            int i = idx / (nocc * nocc * nocc);
            int rem = idx % (nocc * nocc * nocc);
            int j = rem / (nocc * nocc); rem %= (nocc * nocc);
            int k = rem / nocc; int l = rem % nocc;
            d_eri_oooo_[idx] = d_eri_mo[((size_t)i * nao + j) * nao2 + (size_t)k * nao + l];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)vvvv_size; idx++) {
            int a = idx / (nvir * nvir * nvir);
            int rem = idx % (nvir * nvir * nvir);
            int b = rem / (nvir * nvir); rem %= (nvir * nvir);
            int c = rem / nvir; int d = rem % nvir;
            d_eri_vvvv_[idx] = d_eri_mo[((size_t)(a+nocc)*nao + b+nocc)*nao2 + (size_t)(c+nocc)*nao + d+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oovv_size; idx++) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir); rem %= (nvir * nvir);
            int a = rem / nvir; int b = rem % nvir;
            d_eri_oovv_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovvo_size; idx++) {
            int i = idx / (nvir * nvir * nocc);
            int rem = idx % (nvir * nvir * nocc);
            int a = rem / (nvir * nocc); rem %= (nvir * nocc);
            int b = rem / nocc; int j = rem % nocc;
            d_eri_ovvo_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
        }
        // OVVV: eri_ovvv[i,a,b,c] = eri_mo[(i)*N^3 + (a+nocc)*N^2 + (b+nocc)*N + (c+nocc)]
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovvv_size; idx++) {
            int i = (int)(idx / ((size_t)nvir * nvir * nvir));
            size_t rem2 = idx % ((size_t)nvir * nvir * nvir);
            int a = (int)(rem2 / ((size_t)nvir * nvir));
            rem2 %= ((size_t)nvir * nvir);
            int b = (int)(rem2 / nvir);
            int c = (int)(rem2 % nvir);
            size_t mo_idx = (size_t)i * N * N * N + (size_t)(a + nocc) * N * N + (size_t)(b + nocc) * N + (size_t)(c + nocc);
            d_eri_ovvv_[idx] = d_eri_mo[mo_idx];
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

        blocks = (ovvv_size + threads - 1) / threads;
        eom_ccsd_extract_eri_ovvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvv_, nocc_, nvir_, nao_);

        cudaDeviceSynchronize();
    }
}

void EOMCCSDOperator::compute_denominators_and_fock(const real_t* d_orbital_energies) {
    int nocc = nocc_, nvir = nvir_;

    tracked_cudaMalloc(&d_D2_, (size_t)doubles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_D1_, (size_t)singles_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_oo_, (size_t)nocc * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv_, (size_t)nvir * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback: D2 = eps_a + eps_b - eps_i - eps_j
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir);
            rem %= (nvir * nvir);
            int a = rem / nvir;
            int b = rem % nvir;
            d_D2_[idx] = d_orbital_energies[a + nocc] + d_orbital_energies[b + nocc]
                       - d_orbital_energies[i] - d_orbital_energies[j];
        }
        // CPU fallback: D1
        #pragma omp parallel for
        for (int idx = 0; idx < singles_dim_; idx++) {
            d_D1_[idx] = d_orbital_energies[idx % nvir + nocc] - d_orbital_energies[idx / nvir];
        }
        // CPU fallback: Fock diagonal
        for (int idx = 0; idx < nocc; idx++)
            d_f_oo_[idx] = d_orbital_energies[idx];
        for (int idx = 0; idx < nvir; idx++)
            d_f_vv_[idx] = d_orbital_energies[idx + nocc];
    } else {
        int threads = 256;

        int blocks = (doubles_dim_ + threads - 1) / threads;
        eom_ccsd_compute_D2_kernel<<<blocks, threads>>>(
            d_orbital_energies, d_D2_, nocc_, nvir_);

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

void EOMCCSDOperator::build_diagonal() {
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

// ========================================================================
//  Build dressed intermediates (host-side, called once in constructor)
// ========================================================================

// Host-side index macros (NO=nocc, NV=nvir)
#define H_OVOV(p,a,q,b) h_ovov[(size_t)(p)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(q)*NV + (b)]
#define H_OOOV(p,q,r,a) h_ooov[(size_t)(p)*NO*NO*NV + (size_t)(q)*NO*NV + (size_t)(r)*NV + (a)]
#define H_OOVV(p,q,a,b) h_oovv[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
#define H_OVVO(p,a,b,q) h_ovvo[(size_t)(p)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(b)*NO + (q)]
#define H_OVVV(p,a,b,c) h_ovvv[(size_t)(p)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
#define H_OOOO(p,q,r,s) h_oooo[(size_t)(p)*NO*NO*NO + (size_t)(q)*NO*NO + (size_t)(r)*NO + (s)]
#define H_T1(p,a) h_t1[(p)*NV + (a)]
#define H_T2(p,q,a,b) h_t2[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
// ovoo[i,a,j,k] = (ia|jk) = (jk|ia) [bra-ket] = ooov[j,k,i,a]
#define H_OVOO(p,a,q,r) H_OOOV(q,r,p,a)

void EOMCCSDOperator::build_dressed_intermediates() {
    const int NO = nocc_;
    const int NV = nvir_;
    const size_t t1_sz = (size_t)NO * NV;
    const size_t t2_sz = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t oovv_sz = (size_t)NO * NO * NV * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;

    // Download all arrays to host
    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz), h_oooo(oooo_sz);

    cudaMemcpy(h_t1.data(), d_t1_, t1_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(), d_t2_, t2_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_, ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_, ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_, oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_, ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_, ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oooo.data(), d_eri_oooo_, oooo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Download fock diagonal
    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    // === Helper: ovov_2m1[i,a,j,b] = 2*ovov[i,a,j,b] - ovov[i,b,j,a] ===
    std::vector<real_t> ovov_2m1(ovov_sz);
    for (size_t idx = 0; idx < ovov_sz; idx++) {
        int i = (int)(idx / ((size_t)NV * NO * NV));
        size_t r = idx % ((size_t)NV * NO * NV);
        int a = (int)(r / ((size_t)NO * NV));
        r %= ((size_t)NO * NV);
        int j = (int)(r / NV);
        int b = (int)(r % NV);
        ovov_2m1[idx] = 2.0 * H_OVOV(i, a, j, b) - H_OVOV(i, b, j, a);
    }
    // Access macro for ovov_2m1
    #define OVOV2M1(i,a,j,b) ovov_2m1[(size_t)(i)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(j)*NV + (b)]

    // === Helper: tilab[i,j,a,b] = 0.5*t1[i,a]*t1[j,b] + t2[i,j,a,b] ===
    std::vector<real_t> tilab(t2_sz);
    for (int i = 0; i < NO; i++)
        for (int j = 0; j < NO; j++)
            for (int a = 0; a < NV; a++)
                for (int b = 0; b < NV; b++)
                    tilab[(size_t)i*NO*NV*NV + j*NV*NV + a*NV + b] =
                        0.5 * H_T1(i,a) * H_T1(j,b) + H_T2(i,j,a,b);
    #define TILAB(i,j,a,b) tilab[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]

    // === Helper: tau[i,j,a,b] = t2 + 0.5*(t1⊗t1 + transpose) ===
    std::vector<real_t> tau(t2_sz);
    for (int i = 0; i < NO; i++)
        for (int j = 0; j < NO; j++)
            for (int a = 0; a < NV; a++)
                for (int b = 0; b < NV; b++) {
                    real_t t1t1 = H_T1(i,a) * H_T1(j,b);
                    real_t t1t1_T = H_T1(j,a) * H_T1(i,b);
                    tau[(size_t)i*NO*NV*NV + j*NV*NV + a*NV + b] =
                        H_T2(i,j,a,b) + 0.5 * (t1t1 + t1t1_T);
                }
    #define TAU(i,j,a,b) tau[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]

    // === Helper: theta_t[i,j,a,b] = 2*t2 - t2.T(0,1,3,2) ===
    std::vector<real_t> theta_t(t2_sz);
    for (int i = 0; i < NO; i++)
        for (int j = 0; j < NO; j++)
            for (int a = 0; a < NV; a++)
                for (int b = 0; b < NV; b++)
                    theta_t[(size_t)i*NO*NV*NV + j*NV*NV + a*NV + b] =
                        2.0 * H_T2(i,j,a,b) - H_T2(i,j,b,a);
    #define THETA_T(i,j,a,b) theta_t[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]

    // === Fov ===
    // fov = 0 for canonical MOs, but compute anyway
    std::vector<real_t> h_Fov(t1_sz, 0.0);
    for (int m = 0; m < NO; m++)
        for (int e = 0; e < NV; e++) {
            real_t val = 0.0; // fov=0 for canonical
            for (int n = 0; n < NO; n++)
                for (int f = 0; f < NV; f++)
                    val += H_T1(n,f) * OVOV2M1(m,e,n,f);
            h_Fov[m*NV + e] = val;
        }
    #define H_FOV(m,e) h_Fov[(m)*NV + (e)]

    // ================================================================
    //  1. Foo [nocc × nocc]
    // ================================================================
    std::vector<real_t> h_Foo(NO * NO, 0.0);
    // Foo[n,i] = Σ_{m,e,f} tilab[m,i,e,f] * ovov_2m1[m,e,n,f]
    for (int n = 0; n < NO; n++)
        for (int i = 0; i < NO; i++) {
            real_t val = 0.0;
            for (int m = 0; m < NO; m++)
                for (int e = 0; e < NV; e++)
                    for (int f = 0; f < NV; f++)
                        val += TILAB(m,i,e,f) * OVOV2M1(m,e,n,f);
            h_Foo[n*NO + i] = val;
        }
    // + einsum('ne,nemi->mi', t1, ovoo_2m1)
    // ovoo_2m1[n,e,m,i] = 2*ovoo[n,e,m,i] - ovoo[m,e,n,i]
    // ovoo[n,e,m,i] = H_OVOO(n,e,m,i) = H_OOOV(m,i,n,e)
    for (int m = 0; m < NO; m++)
        for (int i = 0; i < NO; i++) {
            real_t val = 0.0;
            for (int n = 0; n < NO; n++)
                for (int e = 0; e < NV; e++) {
                    real_t ovoo_nemi = H_OOOV(m,i,n,e); // = (mi|ne) = (ne|mi) = ovoo[n,e,m,i]
                    real_t ovoo_meni = H_OOOV(n,i,m,e); // = (ni|me) = (me|ni) = ovoo[m,e,n,i]
                    val += H_T1(n,e) * (2.0 * ovoo_nemi - ovoo_meni);
                }
            h_Foo[m*NO + i] += val;
        }
    // + foo + 0.5 * einsum('me,ie->mi', Fov+fov, t1)
    for (int m = 0; m < NO; m++)
        for (int i = 0; i < NO; i++) {
            h_Foo[m*NO + i] += (m == i ? h_f_oo[m] : 0.0);
            real_t val = 0.0;
            for (int e = 0; e < NV; e++)
                val += H_FOV(m,e) * H_T1(i,e); // fov=0, so just Fov
            h_Foo[m*NO + i] += 0.5 * val;
        }

    // ================================================================
    //  2. Fvv [nvir × nvir]
    // ================================================================
    std::vector<real_t> h_Fvv(NV * NV, 0.0);
    // Fvv[a,e] = fvv[a,e] - Σ_{m,n,f} tilab[m,n,a,f]*ovov_2m1[m,e,n,f]
    //          + 2*Σ_{m,f} t1[m,f]*ovvv[m,f,a,e] - Σ_{m,f} t1[m,f]*ovvv[m,e,a,f]
    //          - 0.5*Σ_e (Fov[m,e]+fov[m,e])*t1[m,a]
    for (int a = 0; a < NV; a++)
        for (int e = 0; e < NV; e++) {
            real_t val = (a == e ? h_f_vv[a] : 0.0);
            // tilab contraction
            for (int m = 0; m < NO; m++)
                for (int n = 0; n < NO; n++)
                    for (int f = 0; f < NV; f++)
                        val -= TILAB(m,n,a,f) * OVOV2M1(m,e,n,f);
            // ovvv terms
            for (int m = 0; m < NO; m++)
                for (int f = 0; f < NV; f++)
                    val += 2.0 * H_T1(m,f) * H_OVVV(m,f,a,e)
                         - H_T1(m,f) * H_OVVV(m,e,a,f);
            // Fov correction
            real_t fov_corr = 0.0;
            for (int m = 0; m < NO; m++)
                fov_corr += H_FOV(m,e) * H_T1(m,a); // fov=0
            val -= 0.5 * fov_corr;
            h_Fvv[a*NV + e] = val;
        }

    // ================================================================
    //  3. Woooo [nocc^4] (make_ee version)
    // ================================================================
    std::vector<real_t> h_Woooo(oooo_sz, 0.0);
    for (int m = 0; m < NO; m++)
        for (int n = 0; n < NO; n++)
            for (int i = 0; i < NO; i++)
                for (int j = 0; j < NO; j++) {
                    // bare: (mi|nj)
                    real_t val = H_OOOO(m,i,n,j);
                    // tmp + tmp.T(1,0,3,2): ovoo contractions
                    real_t tmp1 = 0.0, tmp2 = 0.0;
                    for (int e = 0; e < NV; e++) {
                        tmp1 += H_T1(j,e) * H_OOOV(m,i,n,e); // ovoo[n,e,m,i]=(ne|mi)=ooov[m,i,n,e]
                        tmp2 += H_T1(i,e) * H_OOOV(n,j,m,e); // ovoo[m,e,n,j]=(me|nj)=ooov[n,j,m,e]
                    }
                    val += tmp1 + tmp2;
                    // ovov*tau
                    for (int e = 0; e < NV; e++)
                        for (int f = 0; f < NV; f++)
                            val += H_OVOV(m,e,n,f) * TAU(i,j,e,f);
                    h_Woooo[(size_t)m*NO*NO*NO + n*NO*NO + i*NO + j] = val;
                }

    // ================================================================
    //  4. woOoV [nocc^3 × nvir]
    // ================================================================
    std::vector<real_t> h_woOoV(ooov_sz, 0.0);
    // woOoV[m,n,i,e] = Σ_f t1[i,f]*ovov[m,f,n,e] + ooov[m,i,n,e]
    //   (the ooov part comes from ovoo.T(2,0,3,1) = ovoo[i,e,m,n] = ooov[m,n,i,e])
    //   Wait: ovoo.T(2,0,3,1)[m,n,i,e] = ovoo[i,e,m,n] = ooov[m,n,i,e]
    //   Actually: ovoo[i,e,m,n] = (ie|mn) = (mn|ie) = ooov[m,n,i,e]. BUT
    //   H_OOOV(m,n,i,e) = (mn|ie) where i∈occ, e∈vir. Yes: ooov[m,n,i,e]=(mn|ie) ✓
    for (int m = 0; m < NO; m++)
        for (int n = 0; n < NO; n++)
            for (int i = 0; i < NO; i++)
                for (int e = 0; e < NV; e++) {
                    real_t val = H_OOOV(m,n,i,e); // bare: H_OOOV(m,n,i,e) = (mn|ie)
                    // Hmm wait: PySCF's woOoV is ovoo.T(2,0,3,1). Let me re-derive.
                    // ovoo[i,a,j,k] shape (NO,NV,NO,NO).
                    // T(2,0,3,1): new[α,β,γ,δ] = old[γ,β,α,δ]... no.
                    // T(2,0,3,1) means: new axis order is (old_axis2, old_axis0, old_axis3, old_axis1)
                    // So new[a,b,c,d] = old[b,d,a,c]
                    // new[m,n,i,e] = old[n,e,m,i] = ovoo[n,e,m,i] = (ne|mi) = (mi|ne) = H_OOOV(m,i,n,e)
                    // So the bare term is H_OOOV(m,i,n,e), NOT H_OOOV(m,n,i,e)!
                    val = H_OOOV(m,i,n,e);
                    for (int f = 0; f < NV; f++)
                        val += H_T1(i,f) * H_OVOV(m,f,n,e);
                    h_woOoV[(size_t)m*NO*NO*NV + n*NO*NV + i*NV + e] = val;
                }
    #define H_WOOOV(m,n,i,e) h_woOoV[(size_t)(m)*NO*NO*NV + (size_t)(n)*NO*NV + (size_t)(i)*NV + (e)]

    // ================================================================
    //  5. WoVVo [nocc × nvir × nvir × nocc]
    // ================================================================
    std::vector<real_t> h_WoVVo(ovvo_sz, 0.0);
    // WoVVo[m,b,e,j] = -(mj|be) - Σ_f t1[j,f]*ovvv[m,f,b,e]
    //                  + Σ_{n,f} t2[n,j,b,f]*ovov[m,f,n,e]
    //                  + Σ_{n,f} t1[n,b]*ovov[m,f,n,e]*t1[j,f]
    //                  + Σ_n t1[n,b]*ooov[m,j,n,e]  (ovoo[n,e,m,j]=(ne|mj)=ooov[m,j,n,e])
    for (int m = 0; m < NO; m++)
        for (int b = 0; b < NV; b++)
            for (int e = 0; e < NV; e++)
                for (int j = 0; j < NO; j++) {
                    real_t val = -H_OOVV(m,j,b,e); // -(mj|be)
                    // -t1*ovvv
                    for (int f = 0; f < NV; f++)
                        val -= H_T1(j,f) * H_OVVV(m,f,b,e);
                    // t2*ovov
                    for (int n = 0; n < NO; n++)
                        for (int f = 0; f < NV; f++)
                            val += H_T2(n,j,b,f) * H_OVOV(m,f,n,e);
                    // t1*t1*ovov (via tmp2 = ovov*t1[j,f], then t1[n,b]*tmp2)
                    for (int n = 0; n < NO; n++) {
                        real_t tmp2 = 0.0;
                        for (int f = 0; f < NV; f++)
                            tmp2 += H_OVOV(m,f,n,e) * H_T1(j,f);
                        val += H_T1(n,b) * tmp2;
                    }
                    // t1*ovoo: ovoo[n,e,m,j] = (ne|mj) = ooov[m,j,n,e]
                    for (int n = 0; n < NO; n++)
                        val += H_T1(n,b) * H_OOOV(m,j,n,e);
                    h_WoVVo[(size_t)m*NV*NV*NO + b*NV*NO + e*NO + j] = val;
                }
    #define H_WoVVo(m,b,e,j) h_WoVVo[(size_t)(m)*NV*NV*NO + (size_t)(b)*NV*NO + (size_t)(e)*NO + (j)]

    // ================================================================
    //  6. WoVvO [nocc × nvir × nvir × nocc]
    // ================================================================
    std::vector<real_t> h_WoVvO(ovvo_sz, 0.0);
    // WoVvO[m,b,e,j] = (me|bj) + Σ_f t1[j,f]*ovvv[m,e,b,f]
    //                  - 0.5*Σ_{n,f} t2[n,j,b,f]*ovov[m,f,n,e]
    //                  - Σ_{n,f} t1[n,b]*ovov[m,e,n,f]*t1[j,f]
    //                  - Σ_n t1[n,b]*ovoo[m,e,n,j]  (ovoo[m,e,n,j]=(me|nj)=ooov[n,j,m,e])
    //                  + 0.5*Σ_{n,f} theta_t[n,j,f,b]*ovov_sym[m,e,n,f]
    //   where ovov_sym = 2*ovov - ovov.T(0,3,2,1) = ovov_2m1
    for (int m = 0; m < NO; m++)
        for (int b = 0; b < NV; b++)
            for (int e = 0; e < NV; e++)
                for (int j = 0; j < NO; j++) {
                    real_t val = H_OVVO(m,e,b,j); // (me|bj)
                    // +t1*ovvv
                    for (int f = 0; f < NV; f++)
                        val += H_T1(j,f) * H_OVVV(m,e,b,f);
                    // -0.5*t2*ovov (reuse same contraction as WoVVo but -0.5)
                    for (int n = 0; n < NO; n++)
                        for (int f = 0; f < NV; f++)
                            val -= 0.5 * H_T2(n,j,b,f) * H_OVOV(m,f,n,e);
                    // -t1*t1*ovov (note: ovov[m,e,n,f] not [m,f,n,e])
                    for (int n = 0; n < NO; n++) {
                        real_t tmp3 = 0.0;
                        for (int f = 0; f < NV; f++)
                            tmp3 += H_OVOV(m,e,n,f) * H_T1(j,f);
                        val -= H_T1(n,b) * tmp3;
                    }
                    // -t1*ovoo: ovoo[m,e,n,j] = (me|nj) = ooov[n,j,m,e]
                    for (int n = 0; n < NO; n++)
                        val -= H_T1(n,b) * H_OOOV(n,j,m,e);
                    // +0.5*theta_t*ovov_2m1
                    for (int n = 0; n < NO; n++)
                        for (int f = 0; f < NV; f++)
                            val += 0.5 * THETA_T(n,j,f,b) * OVOV2M1(m,e,n,f);
                    h_WoVvO[(size_t)m*NV*NV*NO + b*NV*NO + e*NO + j] = val;
                }

    // ================================================================
    //  7. wvOvV [nvir × nocc × nvir × nvir]  (PySCF Wvovv)
    // ================================================================
    // wvOvV[a,l,c,d] = -Σ_k t1[k,a]*ovov[k,c,l,d] + ovvv[l,d,a,c]
    size_t wvOvV_sz = (size_t)NV * NO * NV * NV;
    std::vector<real_t> h_wvOvV(wvOvV_sz, 0.0);
    for (int a = 0; a < NV; a++)
        for (int l = 0; l < NO; l++)
            for (int c = 0; c < NV; c++)
                for (int d = 0; d < NV; d++) {
                    real_t val = H_OVVV(l,d,a,c); // ovvv[l,d,a,c] = (ld|ac)
                    for (int k = 0; k < NO; k++)
                        val -= H_T1(k,a) * H_OVOV(k,c,l,d);
                    h_wvOvV[(size_t)a*NO*NV*NV + l*NV*NV + c*NV + d] = val;
                }

    // ================================================================
    //  8. woVoO [nocc × nvir × nocc × nocc]  (PySCF Wovoo from rintermediates)
    // ================================================================
    // This is the most complex intermediate. Following PySCF's rintermediates.py:
    // Wkbij = W1ovov*t1 - Woooo_ri*t1 + W1ovvo*t1
    //       + ovoo/ovvv/Fov contractions with t2 + bare ERI

    // --- Helper: W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d]*t2[i,l,c,b] ---
    size_t w1ovov_sz = ovov_sz; // nocc*nvir*nocc*nvir
    std::vector<real_t> W1ovov(w1ovov_sz, 0.0);
    for (int k = 0; k < NO; k++)
        for (int b = 0; b < NV; b++)
            for (int i = 0; i < NO; i++)
                for (int d = 0; d < NV; d++) {
                    real_t val = H_OOVV(k,i,b,d); // (ki|bd)
                    for (int c = 0; c < NV; c++)
                        for (int l = 0; l < NO; l++)
                            val -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                    W1ovov[(size_t)k*NV*NO*NV + b*NO*NV + i*NV + d] = val;
                }
    #define H_W1OVOV(k,b,i,d) W1ovov[(size_t)(k)*NV*NO*NV + (size_t)(b)*NO*NV + (size_t)(i)*NV + (d)]

    // --- Helper: Woooo_ri[k,l,i,j] (rintermediates version) ---
    std::vector<real_t> Woooo_ri(oooo_sz, 0.0);
    for (int k = 0; k < NO; k++)
        for (int l = 0; l < NO; l++)
            for (int i = 0; i < NO; i++)
                for (int j = 0; j < NO; j++) {
                    real_t val = H_OOOO(k,i,l,j); // (ki|lj)
                    for (int c = 0; c < NV; c++)
                        for (int d = 0; d < NV; d++)
                            val += H_OVOV(k,c,l,d) * (H_T2(i,j,c,d) + H_T1(i,c)*H_T1(j,d));
                    // ovoo terms: ovoo[l,d,k,i]*t1[j,d] + ovoo[k,c,l,j]*t1[i,c]
                    for (int d = 0; d < NV; d++)
                        val += H_OOOV(k,i,l,d) * H_T1(j,d); // ovoo[l,d,k,i]=(ld|ki)=ooov[k,i,l,d]
                    for (int c = 0; c < NV; c++)
                        val += H_OOOV(l,j,k,c) * H_T1(i,c); // ovoo[k,c,l,j]=(kc|lj)=ooov[l,j,k,c]
                    Woooo_ri[(size_t)k*NO*NO*NO + l*NO*NO + i*NO + j] = val;
                }
    #define H_WOOOO_RI(k,l,i,j) Woooo_ri[(size_t)(k)*NO*NO*NO + (size_t)(l)*NO*NO + (size_t)(i)*NO + (j)]

    // --- Helper: W1ovvo[k,a,c,i] ---
    // = ovvo[k,c,a,i] + 2*Σ_{l,d} ovov[k,c,l,d]*t2[i,l,a,d]
    //   - Σ_{l,d} ovov[k,c,l,d]*t2[l,i,a,d] - Σ_{l,d} ovov[k,d,l,c]*t2[i,l,a,d]
    std::vector<real_t> W1ovvo(ovvo_sz, 0.0);
    for (int k = 0; k < NO; k++)
        for (int a = 0; a < NV; a++)
            for (int c = 0; c < NV; c++)
                for (int i = 0; i < NO; i++) {
                    real_t val = H_OVVO(k,c,a,i); // (kc|ai)
                    for (int l = 0; l < NO; l++)
                        for (int d = 0; d < NV; d++) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            val += 2.0 * kcld * H_T2(i,l,a,d);
                            val -= kcld * H_T2(l,i,a,d);
                            val -= H_OVOV(k,d,l,c) * H_T2(i,l,a,d);
                        }
                    W1ovvo[(size_t)k*NV*NV*NO + a*NV*NO + c*NO + i] = val;
                }
    // W1ovvo[k,b,c,j] is accessed by relabeling a→b, i→j
    #define H_W1OVVO(k,a,c,i) W1ovvo[(size_t)(k)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(c)*NO + (i)]

    // --- cc_Fov for woVoO ---
    // cc_Fov[k,c] = fov[k,c] + 2*Σ_{l,d} ovov[k,c,l,d]*t1[l,d] - Σ_{l,d} ovov[k,d,l,c]*t1[l,d]
    std::vector<real_t> cc_Fov(t1_sz, 0.0);
    for (int k = 0; k < NO; k++)
        for (int c = 0; c < NV; c++) {
            real_t val = 0.0; // fov=0 for canonical
            for (int l = 0; l < NO; l++)
                for (int d = 0; d < NV; d++)
                    val += (2.0 * H_OVOV(k,c,l,d) - H_OVOV(k,d,l,c)) * H_T1(l,d);
            cc_Fov[k*NV + c] = val;
        }

    // --- Assemble woVoO[k,b,i,j] ---
    size_t woVoO_sz = (size_t)NO * NV * NO * NO;
    std::vector<real_t> h_woVoO(woVoO_sz, 0.0);
    for (int k = 0; k < NO; k++)
        for (int b = 0; b < NV; b++)
            for (int i = 0; i < NO; i++)
                for (int j = 0; j < NO; j++) {
                    real_t val = 0.0;

                    // Bare ERI: ovoo.T(3,1,2,0)[k,b,i,j] = ovoo[j,b,i,k] = (jb|ik) = ooov[i,k,j,b]
                    val += H_OOOV(i,k,j,b);

                    // W1ovov * t1: Σ_d W1ovov[k,b,i,d]*t1[j,d]
                    for (int d = 0; d < NV; d++)
                        val += H_W1OVOV(k,b,i,d) * H_T1(j,d);

                    // -Woooo_ri * t1: -Σ_l Woooo_ri[k,l,i,j]*t1[l,b]
                    for (int l = 0; l < NO; l++)
                        val -= H_WOOOO_RI(k,l,i,j) * H_T1(l,b);

                    // W1ovvo * t1: Σ_c W1ovvo[k,b,c,j]*t1[i,c]
                    for (int c = 0; c < NV; c++)
                        val += H_W1OVVO(k,b,c,j) * H_T1(i,c);

                    // 2*ovoo*t2: 2*Σ_{l,d} ovoo[l,d,k,i]*t2[l,j,d,b]
                    // ovoo[l,d,k,i] = (ld|ki) = ooov[k,i,l,d]
                    for (int l = 0; l < NO; l++)
                        for (int d = 0; d < NV; d++)
                            val += 2.0 * H_OOOV(k,i,l,d) * H_T2(l,j,d,b);

                    // -ovoo*t2: -Σ_{l,d} ovoo[l,d,k,i]*t2[j,l,d,b]
                    for (int l = 0; l < NO; l++)
                        for (int d = 0; d < NV; d++)
                            val -= H_OOOV(k,i,l,d) * H_T2(j,l,d,b);

                    // -ovoo*t2: -Σ_{l,d} ovoo[k,d,l,i]*t2[l,j,d,b]
                    // ovoo[k,d,l,i] = (kd|li) = ooov[l,i,k,d]
                    for (int l = 0; l < NO; l++)
                        for (int d = 0; d < NV; d++)
                            val -= H_OOOV(l,i,k,d) * H_T2(l,j,d,b);

                    // ovvv*t2: Σ_{c,d} ovvv[k,c,b,d]*t2[j,i,d,c]
                    for (int c = 0; c < NV; c++)
                        for (int d = 0; d < NV; d++)
                            val += H_OVVV(k,c,b,d) * H_T2(j,i,d,c);

                    // ovvv*t1*t1: Σ_{c,d} ovvv[k,c,b,d]*t1[j,d]*t1[i,c]
                    for (int c = 0; c < NV; c++)
                        for (int d = 0; d < NV; d++)
                            val += H_OVVV(k,c,b,d) * H_T1(j,d) * H_T1(i,c);

                    // -ovoo*t2: -Σ_{c,l} ovoo[k,c,l,j]*t2[l,i,b,c]
                    // ovoo[k,c,l,j] = (kc|lj) = ooov[l,j,k,c]
                    for (int c = 0; c < NV; c++)
                        for (int l = 0; l < NO; l++)
                            val -= H_OOOV(l,j,k,c) * H_T2(l,i,b,c);

                    // Fov*t2: Σ_c cc_Fov[k,c]*t2[i,j,c,b]
                    for (int c = 0; c < NV; c++)
                        val += cc_Fov[k*NV + c] * H_T2(i,j,c,b);

                    h_woVoO[(size_t)k*NV*NO*NO + b*NO*NO + i*NO + j] = val;
                }

    // ================================================================
    //  Keep full Foo, Fvv (do NOT subtract bare Fock diagonal).
    //  PySCF subtracts it because eeccsd_matvec_singlet adds bare Fock
    //  via (ea+eb-ei-ej)*r2 explicitly. Our half+symmetrize kernel
    //  already includes Foo/Fvv contractions that supply the Fock
    //  contribution, so we keep the diagonal intact.
    // ================================================================

    // ================================================================
    //  Upload all 8 intermediates to device
    // ================================================================
    size_t foo_sz = (size_t)NO * NO;
    size_t fvv_sz = (size_t)NV * NV;

    tracked_cudaMalloc(&d_Foo_, foo_sz * sizeof(real_t));
    cudaMemcpy(d_Foo_, h_Foo.data(), foo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_Fvv_, fvv_sz * sizeof(real_t));
    cudaMemcpy(d_Fvv_, h_Fvv.data(), fvv_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    size_t fov_sz = (size_t)NO * NV;
    tracked_cudaMalloc(&d_Fov_, fov_sz * sizeof(real_t));
    cudaMemcpy(d_Fov_, h_Fov.data(), fov_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_Woooo_, oooo_sz * sizeof(real_t));
    cudaMemcpy(d_Woooo_, h_Woooo.data(), oooo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_WoVVo_, ovvo_sz * sizeof(real_t));
    cudaMemcpy(d_WoVVo_, h_WoVVo.data(), ovvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_WoVvO_, ovvo_sz * sizeof(real_t));
    cudaMemcpy(d_WoVvO_, h_WoVvO.data(), ovvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_woOoV_, ooov_sz * sizeof(real_t));
    cudaMemcpy(d_woOoV_, h_woOoV.data(), ooov_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_woVoO_, woVoO_sz * sizeof(real_t));
    cudaMemcpy(d_woVoO_, h_woVoO.data(), woVoO_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    tracked_cudaMalloc(&d_wvOvV_, wvOvV_sz * sizeof(real_t));
    cudaMemcpy(d_wvOvV_, h_wvOvV.data(), wvOvV_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    // Allocate workspace for half_sigma2
    tracked_cudaMalloc(&d_half_sigma2_, (size_t)doubles_dim_ * sizeof(real_t));

    std::cout << "  EOM-CCSD dressed intermediates built." << std::endl;

    // Cleanup local macros
    #undef OVOV2M1
    #undef TILAB
    #undef TAU
    #undef THETA_T
    #undef H_FOV
    #undef H_WOOOV
    #undef H_WoVVo
    #undef H_W1OVOV
    #undef H_WOOOO_RI
    #undef H_W1OVVO
}

#undef H_OVOV
#undef H_OOOV
#undef H_OOVV
#undef H_OVVO
#undef H_OVVV
#undef H_OOOO
#undef H_T1
#undef H_T2
#undef H_OVOO

// ========================================================================
//  EOM-CCSD σ1 kernel — 8 terms using dressed intermediates
// ========================================================================
/**
 * Verified formula (matches PySCF eeccsd_matvec_singlet exactly):
 *
 * σ1[ia]  = Σ_e Fvv[a,e] r1[i,e]                                   (T1)
 *         - Σ_m Foo[m,i] r1[m,a]                                   (T2)
 *         + 2 Σ_{m,e} Fov[m,e] r2[i,m,a,e]                        (T3)
 *         -   Σ_{m,e} Fov[m,e] r2[i,m,e,a]                        (T4)
 *         + Σ_{m,f,e} ovvv[m,f,a,e] θ_r2[m,i,f,e]                 (T5)
 *         + 2 Σ_{m,e} (0.5*WoVVo[m,a,e,i]+WoVvO[m,a,e,i]) r1[m,e] (T6)
 *         - Σ_{m,n,e} woOoV[m,n,i,e] θ_r2[m,n,a,e]                (T7)
 *         - Σ_n t1[n,a] Σ_{e,m,f} ovov[n,e,m,f] θ_r2[i,m,e,f]    (T8)
 *
 * where θ_r2[i,j,a,b] = 2*r2[i,j,a,b] - r2[i,j,b,a]
 */
__global__ void eom_ccsd_sigma1_kernel(
    const real_t* __restrict__ d_Foo,
    const real_t* __restrict__ d_Fvv,
    const real_t* __restrict__ d_Fov,
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_ovvv,
    const real_t* __restrict__ d_WoVVo,
    const real_t* __restrict__ d_WoVvO,
    const real_t* __restrict__ d_woOoV,
    const real_t* __restrict__ d_t1,
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

    // T1: einsum('ae,ie->ia', Fvv, r1)
    for (int e = 0; e < nvir; e++)
        sigma += d_Fvv[a * nvir + e] * d_r1[i * nvir + e];

    // T2: -einsum('mi,ma->ia', Foo, r1)
    for (int m = 0; m < nocc; m++)
        sigma -= d_Foo[m * nocc + i] * d_r1[m * nvir + a];

    // T3+T4: Fov × r2
    for (int m = 0; m < nocc; m++)
        for (int e = 0; e < nvir; e++) {
            real_t fov_me = d_Fov[m * nvir + e];
            // T3: +2*Fov[m,e]*r2[i,m,a,e]
            sigma += 2.0 * fov_me
                   * d_r2[(size_t)i * nocc * nvir * nvir + (size_t)m * nvir * nvir + (size_t)a * nvir + e];
            // T4: -Fov[m,e]*r2[i,m,e,a]
            sigma -= fov_me
                   * d_r2[(size_t)i * nocc * nvir * nvir + (size_t)m * nvir * nvir + (size_t)e * nvir + a];
        }

    // T5: einsum('mfae,mife->ia', ovvv, theta_r2)
    // ovvv[m,f,a,e], theta_r2[m,i,f,e] = 2*r2[m,i,f,e] - r2[m,i,e,f]
    for (int m = 0; m < nocc; m++)
        for (int f = 0; f < nvir; f++)
            for (int e = 0; e < nvir; e++) {
                real_t ovvv_mfae = d_eri_ovvv[(size_t)m * nvir * nvir * nvir
                                            + (size_t)f * nvir * nvir
                                            + (size_t)a * nvir + e];
                size_t r2_base = (size_t)m * nocc * nvir * nvir + (size_t)i * nvir * nvir;
                real_t theta = 2.0 * d_r2[r2_base + (size_t)f * nvir + e]
                             -       d_r2[r2_base + (size_t)e * nvir + f];
                sigma += ovvv_mfae * theta;
            }

    // T6: 2*einsum('maei,me->ia', combined, r1)
    // combined[m,a,e,i] = 0.5*WoVVo[m,a,e,i] + WoVvO[m,a,e,i]
    for (int m = 0; m < nocc; m++)
        for (int e = 0; e < nvir; e++) {
            size_t idx = (size_t)m * nvir * nvir * nocc
                       + (size_t)a * nvir * nocc
                       + (size_t)e * nocc + i;
            real_t combined = 0.5 * d_WoVVo[idx] + d_WoVvO[idx];
            sigma += 2.0 * combined * d_r1[m * nvir + e];
        }

    // T7: -einsum('mnie,mnae->ia', woOoV, theta_r2)
    // woOoV[m,n,i,e], theta_r2[m,n,a,e] = 2*r2[m,n,a,e] - r2[m,n,e,a]
    for (int m = 0; m < nocc; m++)
        for (int n = 0; n < nocc; n++)
            for (int e = 0; e < nvir; e++) {
                real_t w_mnie = d_woOoV[(size_t)m * nocc * nocc * nvir
                                       + (size_t)n * nocc * nvir
                                       + (size_t)i * nvir + e];
                size_t r2_base = (size_t)m * nocc * nvir * nvir + (size_t)n * nvir * nvir;
                real_t theta = 2.0 * d_r2[r2_base + (size_t)a * nvir + e]
                             -       d_r2[r2_base + (size_t)e * nvir + a];
                sigma -= w_mnie * theta;
            }

    // T8: -einsum('na,ni->ia', t1, tmp) where tmp[n,i] = einsum('nemf,imef->ni', ovov, theta_r2)
    for (int n = 0; n < nocc; n++) {
        real_t tmp_ni = 0.0;
        for (int e = 0; e < nvir; e++)
            for (int m = 0; m < nocc; m++)
                for (int f = 0; f < nvir; f++) {
                    real_t ovov_nemf = d_eri_ovov[(size_t)n * nvir * nocc * nvir
                                                + (size_t)e * nocc * nvir
                                                + (size_t)m * nvir + f];
                    size_t r2_base = (size_t)i * nocc * nvir * nvir + (size_t)m * nvir * nvir;
                    real_t theta = 2.0 * d_r2[r2_base + (size_t)e * nvir + f]
                                 -       d_r2[r2_base + (size_t)f * nvir + e];
                    tmp_ni += ovov_nemf * theta;
                }
        sigma -= d_t1[n * nvir + a] * tmp_ni;
    }

    d_sigma1[ia] = sigma;
}

// Add penalty for anti-singlet r2 components:
// σ2[idx] += penalty * (r2[idx] - r2_sym[idx])
// This pushes anti-singlet eigenvalues to +penalty, far above physical states.
__global__ void eom_ccsd_singlet_penalty_kernel(
    const real_t* __restrict__ d_r2,
    const real_t* __restrict__ d_r2_sym,
    real_t* __restrict__ d_sigma2,
    real_t penalty, int doubles_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= doubles_dim) return;
    d_sigma2[idx] += penalty * (d_r2[idx] - d_r2_sym[idx]);
}

void EOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    const int NO = nocc_;
    const int NV = nvir_;

    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + singles_dim_;
    real_t* d_sigma1 = d_output;
    real_t* d_sigma2 = d_output + singles_dim_;

    // Step 0: Symmetrize input r2 for singlet: r2_sym[ijab] = 0.5*(r2[ijab] + r2[jiba])
    real_t* d_r2_sym = nullptr;
    tracked_cudaMalloc(&d_r2_sym, (size_t)doubles_dim_ * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU: symmetrize r2
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (NO * NV * NV);
            int rem = idx % (NO * NV * NV);
            int j = rem / (NV * NV);
            rem %= (NV * NV);
            int a = rem / NV;
            int b = rem % NV;
            size_t jiba = (size_t)j * NO * NV * NV + (size_t)i * NV * NV + (size_t)b * NV + a;
            d_r2_sym[idx] = 0.5 * (d_r2[idx] + d_r2[jiba]);
        }
    } else {
        int threads = 256;
        int blocks2 = (doubles_dim_ + threads - 1) / threads;
        eom_ccsd_symmetrize_r2_kernel<<<blocks2, threads>>>(
            d_r2, d_r2_sym, NO, NV);
    }

    // sigma1 kernel
    if (!gpu::gpu_available()) {
        // CPU fallback for sigma1
        #pragma omp parallel for
        for (int ia = 0; ia < singles_dim_; ia++) {
            int i = ia / NV;
            int a = ia % NV;
            real_t sigma = 0.0;
            // T1: Fvv
            for (int e = 0; e < NV; e++)
                sigma += d_Fvv_[a * NV + e] * d_r1[i * NV + e];
            // T2: -Foo
            for (int m = 0; m < NO; m++)
                sigma -= d_Foo_[m * NO + i] * d_r1[m * NV + a];
            // T3+T4: Fov * r2
            for (int m = 0; m < NO; m++)
                for (int e = 0; e < NV; e++) {
                    real_t fov_me = d_Fov_[m * NV + e];
                    sigma += 2.0 * fov_me
                           * d_r2_sym[(size_t)i * NO * NV * NV + (size_t)m * NV * NV + (size_t)a * NV + e];
                    sigma -= fov_me
                           * d_r2_sym[(size_t)i * NO * NV * NV + (size_t)m * NV * NV + (size_t)e * NV + a];
                }
            // T5: ovvv * theta_r2
            for (int m = 0; m < NO; m++)
                for (int f = 0; f < NV; f++)
                    for (int e = 0; e < NV; e++) {
                        real_t ovvv_mfae = d_eri_ovvv_[(size_t)m * NV * NV * NV + (size_t)f * NV * NV + (size_t)a * NV + e];
                        size_t r2_base = (size_t)m * NO * NV * NV + (size_t)i * NV * NV;
                        real_t theta = 2.0 * d_r2_sym[r2_base + (size_t)f * NV + e]
                                     -       d_r2_sym[r2_base + (size_t)e * NV + f];
                        sigma += ovvv_mfae * theta;
                    }
            // T6: combined * r1
            for (int m = 0; m < NO; m++)
                for (int e = 0; e < NV; e++) {
                    size_t widx = (size_t)m * NV * NV * NO + (size_t)a * NV * NO + (size_t)e * NO + i;
                    real_t combined = 0.5 * d_WoVVo_[widx] + d_WoVvO_[widx];
                    sigma += 2.0 * combined * d_r1[m * NV + e];
                }
            // T7: -woOoV * theta_r2
            for (int m = 0; m < NO; m++)
                for (int n = 0; n < NO; n++)
                    for (int e = 0; e < NV; e++) {
                        real_t w_mnie = d_woOoV_[(size_t)m * NO * NO * NV + (size_t)n * NO * NV + (size_t)i * NV + e];
                        size_t r2_base = (size_t)m * NO * NV * NV + (size_t)n * NV * NV;
                        real_t theta = 2.0 * d_r2_sym[r2_base + (size_t)a * NV + e]
                                     -       d_r2_sym[r2_base + (size_t)e * NV + a];
                        sigma -= w_mnie * theta;
                    }
            // T8: -t1 * (ovov * theta_r2)
            for (int n = 0; n < NO; n++) {
                real_t tmp_ni = 0.0;
                for (int e = 0; e < NV; e++)
                    for (int m = 0; m < NO; m++)
                        for (int f = 0; f < NV; f++) {
                            real_t ovov_nemf = d_eri_ovov_[(size_t)n * NV * NO * NV + (size_t)e * NO * NV + (size_t)m * NV + f];
                            size_t r2_base = (size_t)i * NO * NV * NV + (size_t)m * NV * NV;
                            real_t theta = 2.0 * d_r2_sym[r2_base + (size_t)e * NV + f]
                                         -       d_r2_sym[r2_base + (size_t)f * NV + e];
                            tmp_ni += ovov_nemf * theta;
                        }
                sigma -= d_t1_[n * NV + a] * tmp_ni;
            }
            d_sigma1[ia] = sigma;
        }
    } else {
        int threads = 256;
        int blocks1 = (singles_dim_ + threads - 1) / threads;
        eom_ccsd_sigma1_kernel<<<blocks1, threads>>>(
            d_Foo_, d_Fvv_, d_Fov_,
            d_eri_ovov_, d_eri_ovvv_,
            d_WoVVo_, d_WoVvO_, d_woOoV_,
            d_t1_, d_r1, d_r2_sym, d_sigma1,
            nocc_, nvir_);
    }

    // Step 1: Compute response tau (using symmetrized r2)
    real_t* d_tau2 = nullptr;
    tracked_cudaMalloc(&d_tau2, (size_t)doubles_dim_ * sizeof(real_t));

    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (NO * NV * NV);
            int rem = idx % (NO * NV * NV);
            int j = rem / (NV * NV);
            rem %= (NV * NV);
            int a = rem / NV;
            int b = rem % NV;
            d_tau2[idx] = d_r2_sym[idx]
                        + d_r1[i * NV + a] * d_t1_[j * NV + b]
                        + d_t1_[i * NV + a] * d_r1[j * NV + b];
        }
    } else {
        int threads = 256;
        int blocks2 = (doubles_dim_ + threads - 1) / threads;
        eom_ccsd_response_tau_kernel<<<blocks2, threads>>>(
            d_r1, d_r2_sym, d_t1_, d_tau2, NO, NV);
        cudaDeviceSynchronize();
    }

    // Step 2: Compute small R-dependent intermediates
    size_t t1_sz = (size_t)NO * NV;
    size_t t2_sz = (size_t)NO * NO * NV * NV;
    size_t ovov_sz = (size_t)NO * NV * NO * NV;
    size_t ooov_sz = (size_t)NO * NO * NO * NV;
    size_t oooo_sz = (size_t)NO * NO * NO * NO;

    // Allocate device intermediates (used by both paths)
    real_t *d_af_tmp, *d_ni_tmp, *d_eb_tmp, *d_ni_tmp2, *d_mnij_tmp, *d_tau_half;
    tracked_cudaMalloc(&d_af_tmp, NV * NV * sizeof(real_t));
    tracked_cudaMalloc(&d_ni_tmp, NO * NO * sizeof(real_t));
    tracked_cudaMalloc(&d_eb_tmp, NV * NV * sizeof(real_t));
    tracked_cudaMalloc(&d_ni_tmp2, NO * NO * sizeof(real_t));
    tracked_cudaMalloc(&d_mnij_tmp, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_tau_half, t2_sz * sizeof(real_t));

    if (gpu::gpu_available()) {
        // GPU path: launch kernels directly on device buffers (no host roundtrip)
        int threads = 256;
        eom_ccsd_af_tmp_kernel<<<(NV*NV + threads-1)/threads, threads>>>(
            d_eri_ovvv_, d_r1, d_af_tmp, NO, NV);
        eom_ccsd_ni_tmp_kernel<<<(NO*NO + threads-1)/threads, threads>>>(
            d_woOoV_, d_r1, d_ni_tmp, NO, NV);
        eom_ccsd_eb_tmp_kernel<<<(NV*NV + threads-1)/threads, threads>>>(
            d_eri_ovov_, d_t1_, d_r1, d_r2_sym, d_eb_tmp, NO, NV);
        eom_ccsd_ni_tmp2_kernel<<<(NO*NO + threads-1)/threads, threads>>>(
            d_eri_ovov_, d_r2_sym, d_ni_tmp2, NO, NV);
        size_t mnij_blocks = (oooo_sz + threads - 1) / threads;
        eom_ccsd_mnij_tmp_kernel<<<mnij_blocks, threads>>>(
            d_eri_ovov_, d_tau2, d_mnij_tmp, NO, NV);
        size_t tau_blocks = (t2_sz + threads - 1) / threads;
        eom_ccsd_tau_half_kernel<<<tau_blocks, threads>>>(
            d_t1_, d_t2_, d_tau_half, NO, NV);
        cudaDeviceSynchronize();
    } else {
    // ----- CPU host fallback (kept verbatim) -----
    std::vector<real_t> h_r1(t1_sz), h_r2(t2_sz), h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ovvv((size_t)NO*NV*NV*NV);
    std::vector<real_t> h_tau2(t2_sz);
    std::vector<real_t> h_woOoV(ooov_sz);

    cudaMemcpy(h_r1.data(), d_r1, t1_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_r2.data(), d_r2_sym, t2_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t1.data(), d_t1_, t1_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(), d_t2_, t2_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_, ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_, (size_t)NO*NV*NV*NV * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tau2.data(), d_tau2, t2_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_woOoV.data(), d_woOoV_, ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Index helpers
    auto ovov_idx = [&](int i, int a, int j, int b) -> size_t {
        return (size_t)i*NV*NO*NV + a*NO*NV + j*NV + b; };
    auto ovvv_idx = [&](int i, int a, int b, int c) -> size_t {
        return (size_t)i*NV*NV*NV + a*NV*NV + b*NV + c; };
    auto t2_idx = [&](int i, int j, int a, int b) -> size_t {
        return (size_t)i*NO*NV*NV + j*NV*NV + a*NV + b; };
    auto ooov_idx = [&](int i, int j, int k, int a) -> size_t {
        return (size_t)i*NO*NO*NV + j*NO*NV + k*NV + a; };

    // af_tmp[a,f] = 2*Σ_{m,e} ovvv[m,e,a,f]*r1[m,e] - Σ_{m,e} ovvv[m,f,a,e]*r1[m,e]
    std::vector<real_t> h_af_tmp(NV * NV, 0.0);
    for (int a = 0; a < NV; a++)
        for (int f = 0; f < NV; f++) {
            real_t val = 0.0;
            for (int m = 0; m < NO; m++)
                for (int e = 0; e < NV; e++)
                    val += (2.0 * h_ovvv[ovvv_idx(m,e,a,f)] - h_ovvv[ovvv_idx(m,f,a,e)])
                         * h_r1[m*NV + e];
            h_af_tmp[a*NV + f] = val;
        }

    // ni_tmp[n,i] = 2*Σ_{m,e} woOoV[n,m,i,e]*r1[m,e] - Σ_{m,e} woOoV[m,n,i,e]*r1[m,e]
    std::vector<real_t> h_ni_tmp(NO * NO, 0.0);
    for (int n = 0; n < NO; n++)
        for (int i = 0; i < NO; i++) {
            real_t val = 0.0;
            for (int m = 0; m < NO; m++)
                for (int e = 0; e < NV; e++)
                    val += (2.0 * h_woOoV[ooov_idx(n,m,i,e)] - h_woOoV[ooov_idx(m,n,i,e)])
                         * h_r1[m*NV + e];
            h_ni_tmp[n*NO + i] = val;
        }

    // eb_tmp[e,b]:
    //   en_tmp[e,n] = 2*Σ_{m,f} ovov[m,f,n,e]*r1[m,f] - Σ_{m,f} ovov[m,e,n,f]*r1[m,f]
    //   eb_tmp[e,b] = Σ_n en_tmp[e,n]*t1[n,b]
    //              + Σ_{m,n,f} ovov[m,e,n,f] * theta_r[m,n,b,f]
    //   where theta_r = 2*r2 - r2.T(0,1,3,2)
    std::vector<real_t> h_eb_tmp(NV * NV, 0.0);
    {
        // First compute en_tmp
        std::vector<real_t> en_tmp(NV * NO, 0.0);
        for (int e = 0; e < NV; e++)
            for (int n = 0; n < NO; n++) {
                real_t val = 0.0;
                for (int m = 0; m < NO; m++)
                    for (int f = 0; f < NV; f++)
                        val += (2.0 * h_ovov[ovov_idx(m,f,n,e)] - h_ovov[ovov_idx(m,e,n,f)])
                             * h_r1[m*NV + f];
                en_tmp[e*NO + n] = val;
            }
        // eb_tmp = en_tmp * t1 + ovov * theta_r
        for (int e = 0; e < NV; e++)
            for (int b = 0; b < NV; b++) {
                real_t val = 0.0;
                for (int n = 0; n < NO; n++)
                    val += en_tmp[e*NO + n] * h_t1[n*NV + b];
                for (int m = 0; m < NO; m++)
                    for (int n = 0; n < NO; n++)
                        for (int f = 0; f < NV; f++)
                            val += h_ovov[ovov_idx(m,e,n,f)]
                                 * (2.0 * h_r2[t2_idx(m,n,b,f)] - h_r2[t2_idx(m,n,f,b)]);
                h_eb_tmp[e*NV + b] = val;
            }
    }

    // ni_tmp2[n,i] = Σ_{e,m,f} ovov[n,e,m,f] * theta_r[i,m,e,f]
    std::vector<real_t> h_ni_tmp2(NO * NO, 0.0);
    for (int n = 0; n < NO; n++)
        for (int i = 0; i < NO; i++) {
            real_t val = 0.0;
            for (int e = 0; e < NV; e++)
                for (int m = 0; m < NO; m++)
                    for (int f = 0; f < NV; f++)
                        val += h_ovov[ovov_idx(n,e,m,f)]
                             * (2.0 * h_r2[t2_idx(i,m,e,f)] - h_r2[t2_idx(i,m,f,e)]);
            h_ni_tmp2[n*NO + i] = val;
        }

    // mnij_tmp[m,n,i,j] = Σ_{e,f} ovov[m,e,n,f] * tau2[i,j,e,f]
    std::vector<real_t> h_mnij_tmp(oooo_sz, 0.0);
    for (int m = 0; m < NO; m++)
        for (int n = 0; n < NO; n++)
            for (int i = 0; i < NO; i++)
                for (int j = 0; j < NO; j++) {
                    real_t val = 0.0;
                    for (int e = 0; e < NV; e++)
                        for (int f = 0; f < NV; f++)
                            val += h_ovov[ovov_idx(m,e,n,f)] * h_tau2[t2_idx(i,j,e,f)];
                    h_mnij_tmp[(size_t)m*NO*NO*NO + n*NO*NO + i*NO + j] = val;
                }

    // tau_half[m,n,a,b] = 0.5*tau[m,n,a,b]
    //   = 0.5*t2[m,n,a,b] + 0.25*(t1[m,a]*t1[n,b] + t1[n,a]*t1[m,b])
    std::vector<real_t> h_tau_half(t2_sz, 0.0);
    for (int m = 0; m < NO; m++)
        for (int n = 0; n < NO; n++)
            for (int a = 0; a < NV; a++)
                for (int b = 0; b < NV; b++)
                    h_tau_half[t2_idx(m,n,a,b)] =
                        0.5 * h_t2[t2_idx(m,n,a,b)]
                        + 0.25 * (h_t1[m*NV+a]*h_t1[n*NV+b] + h_t1[n*NV+a]*h_t1[m*NV+b]);

    // Upload host-computed intermediates to (already-allocated) device buffers
    cudaMemcpy(d_af_tmp, h_af_tmp.data(), NV*NV*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ni_tmp, h_ni_tmp.data(), NO*NO*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eb_tmp, h_eb_tmp.data(), NV*NV*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ni_tmp2, h_ni_tmp2.data(), NO*NO*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mnij_tmp, h_mnij_tmp.data(), oooo_sz*sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tau_half, h_tau_half.data(), t2_sz*sizeof(real_t), cudaMemcpyHostToDevice);
    }  // end CPU host-fallback block

    // Step 4: Launch half_sigma2 kernel
    if (!gpu::gpu_available()) {
        // CPU fallback: half_sigma2 — replicate kernel logic
        // Access macros for dressed intermediates and raw ERI
        #define HS2_OVOV(i,a,j,b) d_eri_ovov_[(size_t)(i)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(j)*NV + (b)]
        #define HS2_VVVV(a,b,c,d_) d_eri_vvvv_[(size_t)(a)*NV*NV*NV + (size_t)(b)*NV*NV + (size_t)(c)*NV + (d_)]
        #define HS2_OVVV(i,a,b,c) d_eri_ovvv_[(size_t)(i)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
        #define HS2_T1(i,a) d_t1_[(i)*NV + (a)]
        #define HS2_T2(i,j,a,b) d_t2_[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]
        #define HS2_R1(i,a) d_r1[(i)*NV + (a)]
        #define HS2_R2(i,j,a,b) d_r2_sym[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]
        #define HS2_TAU2(i,j,a,b) d_tau2[(size_t)(i)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]
        #define HS2_FOO(m,i) d_Foo_[(m)*NO + (i)]
        #define HS2_FVV(a,e) d_Fvv_[(a)*NV + (e)]
        #define HS2_WOOOO(m,n,i,j) d_Woooo_[(size_t)(m)*NO*NO*NO + (size_t)(n)*NO*NO + (size_t)(i)*NO + (j)]
        #define HS2_WoVVo(m,b,e,j) d_WoVVo_[(size_t)(m)*NV*NV*NO + (size_t)(b)*NV*NO + (size_t)(e)*NO + (j)]
        #define HS2_WoVvO(m,b,e,j) d_WoVvO_[(size_t)(m)*NV*NV*NO + (size_t)(b)*NV*NO + (size_t)(e)*NO + (j)]
        #define HS2_woVoO(m,b,i,j) d_woVoO_[(size_t)(m)*NV*NO*NO + (size_t)(b)*NO*NO + (size_t)(i)*NO + (j)]
        #define HS2_wvOvV(e,j,a,b) d_wvOvV_[(size_t)(e)*NO*NV*NV + (size_t)(j)*NV*NV + (size_t)(a)*NV + (b)]
        #define HS2_AF_TMP(a,f) d_af_tmp[(a)*NV + (f)]
        #define HS2_NI_TMP(n,i) d_ni_tmp[(n)*NO + (i)]
        #define HS2_EB_TMP(e,b) d_eb_tmp[(e)*NV + (b)]
        #define HS2_NI_TMP2(n,i) d_ni_tmp2[(n)*NO + (i)]
        #define HS2_MNIJ_TMP(m,n,i,j) d_mnij_tmp[(size_t)(m)*NO*NO*NO + (size_t)(n)*NO*NO + (size_t)(i)*NO + (j)]
        #define HS2_TAU_HALF(m,n,a,b) d_tau_half[(size_t)(m)*NO*NV*NV + (size_t)(n)*NV*NV + (size_t)(a)*NV + (b)]

        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (NO * NV * NV);
            int rem = idx % (NO * NV * NV);
            int j = rem / (NV * NV);
            rem %= (NV * NV);
            int a = rem / NV;
            int b = rem % NV;
            real_t hr2 = 0.0;
            // Term 1: 0.5 * VVVV * tau2
            { real_t sum = 0.0;
              for (int e = 0; e < NV; e++) for (int f = 0; f < NV; f++)
                  sum += HS2_TAU2(i, j, e, f) * HS2_VVVV(a, e, b, f);
              hr2 += 0.5 * sum; }
            // Term 2: 0.5 * Woooo * r2
            { real_t sum = 0.0;
              for (int m = 0; m < NO; m++) for (int n = 0; n < NO; n++)
                  sum += HS2_WOOOO(m, n, i, j) * HS2_R2(m, n, a, b);
              hr2 += 0.5 * sum; }
            // Term 3: Fvv * r2
            for (int e = 0; e < NV; e++) hr2 += HS2_FVV(b, e) * HS2_R2(i, j, a, e);
            // Term 4: -Foo * r2
            for (int m = 0; m < NO; m++) hr2 -= HS2_FOO(m, j) * HS2_R2(i, m, a, b);
            // Term 5: -t1 * (ovvv * tau2)
            for (int m = 0; m < NO; m++) {
                real_t inner = 0.0;
                for (int e = 0; e < NV; e++) for (int f = 0; f < NV; f++)
                    inner += HS2_OVVV(m, e, b, f) * HS2_TAU2(i, j, e, f);
                hr2 -= HS2_T1(m, a) * inner;
            }
            // Term 6: af_tmp * t2
            for (int f = 0; f < NV; f++) hr2 += HS2_AF_TMP(a, f) * HS2_T2(i, j, f, b);
            // Term 7: -woVoO * r1
            for (int m = 0; m < NO; m++) hr2 -= HS2_woVoO(m, b, i, j) * HS2_R1(m, a);
            // Term 8: +wvOvV * r1
            for (int e = 0; e < NV; e++) hr2 += HS2_wvOvV(e, j, a, b) * HS2_R1(i, e);
            // Term 9: WoVVo * r2 (with 0.5*transpose)
            { real_t sum_ab = 0.0, sum_ba = 0.0;
              for (int m = 0; m < NO; m++) for (int e = 0; e < NV; e++) {
                  sum_ab += HS2_WoVVo(m, b, e, i) * HS2_R2(j, m, e, a);
                  sum_ba += HS2_WoVVo(m, a, e, i) * HS2_R2(j, m, e, b);
              }
              hr2 += sum_ab + 0.5 * sum_ba; }
            // Term 10: (0.5*WoVVo + WoVvO) * theta
            for (int m = 0; m < NO; m++) for (int e = 0; e < NV; e++) {
                real_t combined = 0.5 * HS2_WoVVo(m, b, e, j) + HS2_WoVvO(m, b, e, j);
                real_t theta_imae = 2.0 * HS2_R2(i, m, a, e) - HS2_R2(i, m, e, a);
                hr2 += combined * theta_imae;
            }
            // Term 11: -ni_tmp * t2
            for (int n = 0; n < NO; n++) hr2 -= HS2_NI_TMP(n, i) * HS2_T2(n, j, a, b);
            // Term 12: -eb_tmp * t2
            for (int e = 0; e < NV; e++) hr2 -= HS2_EB_TMP(e, b) * HS2_T2(j, i, e, a);
            // Term 13: -ni_tmp2 * t2
            for (int m = 0; m < NO; m++) hr2 -= HS2_NI_TMP2(m, j) * HS2_T2(m, i, b, a);
            // Term 14: mnij_tmp * tau_half
            for (int m = 0; m < NO; m++) for (int n = 0; n < NO; n++)
                hr2 += HS2_MNIJ_TMP(m, n, i, j) * HS2_TAU_HALF(m, n, a, b);
            d_half_sigma2_[idx] = hr2;
        }

        #undef HS2_OVOV
        #undef HS2_VVVV
        #undef HS2_OVVV
        #undef HS2_T1
        #undef HS2_T2
        #undef HS2_R1
        #undef HS2_R2
        #undef HS2_TAU2
        #undef HS2_FOO
        #undef HS2_FVV
        #undef HS2_WOOOO
        #undef HS2_WoVVo
        #undef HS2_WoVvO
        #undef HS2_woVoO
        #undef HS2_wvOvV
        #undef HS2_AF_TMP
        #undef HS2_NI_TMP
        #undef HS2_EB_TMP
        #undef HS2_NI_TMP2
        #undef HS2_MNIJ_TMP
        #undef HS2_TAU_HALF

        // CPU: Symmetrize
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            int i = idx / (NO * NV * NV);
            int rem = idx % (NO * NV * NV);
            int j = rem / (NV * NV);
            rem %= (NV * NV);
            int a = rem / NV;
            int b = rem % NV;
            size_t jiba = (size_t)j * NO * NV * NV + (size_t)i * NV * NV + (size_t)b * NV + a;
            d_sigma2[idx] = d_half_sigma2_[idx] + d_half_sigma2_[jiba];
        }

        // CPU: Singlet penalty
        #pragma omp parallel for
        for (int idx = 0; idx < doubles_dim_; idx++) {
            d_sigma2[idx] += 1000.0 * (d_r2[idx] - d_r2_sym[idx]);
        }
    } else {
        int threads = 256;
        int blocks2 = (doubles_dim_ + threads - 1) / threads;

        eom_ccsd_half_sigma2_kernel<<<blocks2, threads>>>(
            d_eri_ovov_, d_eri_vvvv_, d_eri_ovvv_,
            d_t1_, d_t2_, d_r1, d_r2_sym, d_tau2,
            d_Foo_, d_Fvv_, d_Woooo_, d_WoVVo_, d_WoVvO_,
            d_woOoV_, d_woVoO_, d_wvOvV_,
            d_af_tmp, d_ni_tmp, d_eb_tmp, d_ni_tmp2, d_mnij_tmp, d_tau_half,
            d_half_sigma2_, NO, NV);

        // Step 5: Symmetrize
        eom_ccsd_symmetrize_kernel<<<blocks2, threads>>>(
            d_half_sigma2_, d_sigma2, NO, NV);

        // Step 6: Penalty for anti-singlet r2 components
        eom_ccsd_singlet_penalty_kernel<<<blocks2, threads>>>(
            d_r2, d_r2_sym, d_sigma2, 1000.0, doubles_dim_);

        cudaDeviceSynchronize();
    }

    // Cleanup temporary device allocations
    tracked_cudaFree(d_r2_sym);
    tracked_cudaFree(d_tau2);
    tracked_cudaFree(d_af_tmp);
    tracked_cudaFree(d_ni_tmp);
    tracked_cudaFree(d_eb_tmp);
    tracked_cudaFree(d_ni_tmp2);
    tracked_cudaFree(d_mnij_tmp);
    tracked_cudaFree(d_tau_half);
}

void EOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
