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
 * @file ea_eom_ccsd_operator.cu
 * @brief EA-EOM-CCSD operator — sub-phases 2.0-2.6 complete.
 *
 *  2.0+2.1  LinearOperator scaffolding + diagonal-only stub apply()
 *  2.2      PySCF-equivalent dressed intermediates (Loo, Lvv, Fov, Wovov,
 *           Wovvo, Wvovv, Wvvvv, Wvvvo) + extract vvvv ERI block
 *  2.3-2.6  Full PySCF eaccsd_matvec — σ1 (4 terms, 1p-1p + 1p-2p1h coupling)
 *           and σ2 (9 terms incl. Wvvvv loop + tmp[k]·t2). CPU OMP + GPU
 *           kernel paths, dispatching on whether intermediates were built
 *           (unit tests pass d_eri_mo == nullptr and exercise only the stub).
 *
 * EA-specific PySCF intermediates (cc/rintermediates.py L131-210):
 *   Wvovv  [a,l,c,d]  = -Σ_k t1[k,a] ovov[k,c,l,d] + ovvv[l,d,a,c]
 *   Wvvvv  [a,b,c,d]  = (ac|bd) + ovov·t2 + ovov·t1·t1 - ovvv·t1 (×2)
 *   Wvvvo  [a,b,c,j]  = 11-term assembly (incl. Wvvvv·t1 because t1 ≠ 0)
 *
 * PySCF eaccsd_matvec (eom_rccsd.py L649-693), closed-shell partition:
 *   σ1[a] = Lvv·r1 + 2 Fov·r2 - Fov·r2 + (2 Wvovv - Wvovv^T)·r2
 *   σ2[j,a,b] = Wvvvo·r1 + Lvv·r2 (×2 slots) - Loo·r2
 *             + (2 Wovvo - Wovov^T)·r2 - Wovov·r2 - Wovvo·r2
 *             + Wvvvv·r2 - tmp[k]·t2[k,j,a,b]
 *   with tmp[k] = (2 Woovv - Woovv^T) · r2,  Woovv[k,l,c,d] = ovov[k,c,l,d]
 */

#include "ea_eom_ccsd_operator.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "eri.hpp"   // Phase 0: ERI_RI::mo_eri_block_into (on-the-fly MO-ERI blocks)
#include "multi_gpu_manager.hpp"   // Stage EA-5: multi-GPU σ (MultiGpuManager/DeviceGuard)

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

// MO ERI block extraction kernels are shared with EE-EOM / IP-EOM.
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_ccsd_extract_eri_ovvv_kernel(
    const real_t*, real_t*, int, int, int);

#ifndef GANSU_CPU_ONLY
// Build the [p_dim | p2h_dim] diagonal in a single launch.
//   1p sector:    D[a]       = +eps[a + nocc]
//   2p1h sector:  D[jab]     = -eps[j] + eps[a + nocc] + eps[b + nocc]
__global__ void ea_eom_build_diagonal_kernel(
    const real_t* __restrict__ eps,
    real_t* __restrict__ D,
    int nocc, int nvir)
{
    const int p_dim   = nvir;
    const int p2h_dim = nocc * nvir * nvir;
    const int total   = p_dim + p2h_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (idx < p_dim) {
        D[idx] = eps[idx + nocc];
    } else {
        int t  = idx - p_dim;
        int b  = t % nvir;
        int t2 = t / nvir;
        int a  = t2 % nvir;
        int j  = t2 / nvir;
        D[idx] = -eps[j] + eps[a + nocc] + eps[b + nocc];
    }
}

__global__ void ea_eom_diag_matvec_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = D[idx] * x[idx];
}

__global__ void ea_eom_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}

// ==================================================================
//  EA-EOM-CCSD sigma kernels (full PySCF eaccsd_matvec)
// ==================================================================

// σ1[a] block (1p-1p + 1p-2p1h coupling)
//   σ1[a] = Σ_c Lvv[a,c] r1[c]
//         + 2 Σ_{l,d} Fov[l,d] r2[l,a,d]
//         -   Σ_{l,d} Fov[l,d] r2[l,d,a]
//         + Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2[l,c,d]
// r2 layout: idx = (l*nvir + c)*nvir + d
__global__ void ea_eom_sigma1_full_kernel(
    const real_t* __restrict__ Lvv,    // [nvir²]
    const real_t* __restrict__ Fov,    // [nocc · nvir]
    const real_t* __restrict__ Wvovv,  // [nvir · nocc · nvir · nvir]
    const real_t* __restrict__ r1,     // [nvir]
    const real_t* __restrict__ r2,     // [nocc · nvir²]
    real_t* __restrict__ sigma1,       // [nvir]
    int nocc, int nvir)
{
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nvir) return;

    real_t s = 0.0;
    // + Σ_c Lvv[a,c] r1[c]
    for (int c = 0; c < nvir; ++c) s += Lvv[a * nvir + c] * r1[c];

    // ± Σ Fov · r2
    for (int l = 0; l < nocc; ++l)
        for (int d = 0; d < nvir; ++d) {
            real_t fov_ld = Fov[l * nvir + d];
            // +2 Fov[l,d] r2[l,a,d]
            s += 2.0 * fov_ld * r2[((size_t)l * nvir + a) * nvir + d];
            // -Fov[l,d] r2[l,d,a]
            s -=       fov_ld * r2[((size_t)l * nvir + d) * nvir + a];
        }

    // + Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2[l,c,d]
    for (int l = 0; l < nocc; ++l)
        for (int c = 0; c < nvir; ++c)
            for (int d = 0; d < nvir; ++d) {
                real_t w1 = Wvovv[(((size_t)a * nocc + l) * nvir + c) * nvir + d];
                real_t w2 = Wvovv[(((size_t)a * nocc + l) * nvir + d) * nvir + c];
                s += (2.0 * w1 - w2) * r2[((size_t)l * nvir + c) * nvir + d];
            }
    sigma1[a] = s;
}


// Pre-stage kernel: tmp[k] = Σ_{l,c,d} (2 Woovv[k,l,c,d] - Woovv[k,l,d,c]) r2[l,c,d]
//   where Woovv[k,l,c,d] = ovov[k,c,l,d] (= H_OVOV in host code, = d_eri_oovv NOT here;
//   PySCF Woovv comes from eris.ovov.transpose(0,2,1,3) so it equals ovov[k,c,l,d])
__global__ void ea_eom_sigma2_tmp_k_kernel(
    const real_t* __restrict__ ovov,   // [nocc·nvir·nocc·nvir]  ovov[k,c,l,d] = (kc|ld)
    const real_t* __restrict__ r2,     // [nocc · nvir²]
    real_t* __restrict__ tmp_k,        // [nocc]
    int nocc, int nvir)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nocc) return;
    real_t s = 0.0;
    for (int l = 0; l < nocc; ++l)
        for (int c = 0; c < nvir; ++c)
            for (int d = 0; d < nvir; ++d) {
                real_t w_kcld = ovov[(((size_t)k * nvir + c) * nocc + l) * nvir + d];
                real_t w_kdlc = ovov[(((size_t)k * nvir + d) * nocc + l) * nvir + c];
                s += (2.0 * w_kcld - w_kdlc) * r2[((size_t)l * nvir + c) * nvir + d];
            }
    tmp_k[k] = s;
}


// σ2[j,a,b] block (2p1h-1p + 2p1h-2p1h)
//   σ2[j,a,b] = +Σ_c Wvvvo[a,b,c,j] r1[c]                      (1p→2p1h coupling)
//             + Σ_c Lvv[a,c] r2[j,c,b]                          (1-body Lvv slot a)
//             + Σ_d Lvv[b,d] r2[j,a,d]                          (1-body Lvv slot b)
//             - Σ_l Loo[l,j] r2[l,a,b]                          (1-body Loo)
//             + Σ_{l,d} (2 Wovvo[l,b,d,j] - Wovov[l,b,j,d]) r2[l,a,d]
//             - Σ_{l,c} Wovov[l,a,j,c] r2[l,c,b]
//             - Σ_{l,c} Wovvo[l,b,c,j] r2[l,c,a]
//             + Σ_{c,d} Wvvvv[a,b,c,d] r2[j,c,d]                (★ EA-only)
//             - Σ_k tmp[k] · t2[k,j,a,b]
// r2 layout: idx = (l*nvir + a)*nvir + b
// t2 layout: idx = (k*nocc + j)*nvir² + a*nvir + b
__global__ void ea_eom_sigma2_full_kernel(
    const real_t* __restrict__ Lvv,        // [nvir²]
    const real_t* __restrict__ Loo,        // [nocc²]
    const real_t* __restrict__ Wovov,      // [nocc · nvir · nocc · nvir]
    const real_t* __restrict__ Wovvo,      // [nocc · nvir · nvir · nocc]
    const real_t* __restrict__ Wvvvv,      // [nvir^4]
    const real_t* __restrict__ Wvvvo,      // [nvir · nvir · nvir · nocc]
    const real_t* __restrict__ d_tmp_k,    // [nocc] precomputed
    const real_t* __restrict__ t2,         // [nocc² · nvir²]
    const real_t* __restrict__ r1,         // [nvir]
    const real_t* __restrict__ r2,         // [nocc · nvir²]
    real_t* __restrict__ sigma2,           // [nocc · nvir²]
    int nocc, int nvir, int include_wvvvv, int include_ring,
    int j_begin, int j_end)                // Stage EA-5c: compute only this output-occ slab
{
    int lidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int slabtot = (j_end - j_begin) * nvir * nvir;
    if (lidx >= slabtot) return;
    int b = lidx % nvir;
    int t = lidx / nvir;
    int a = t % nvir;
    int j = j_begin + t / nvir;            // global output-occ index

    real_t s = 0.0;

    // +Σ_c Wvvvo[a,b,c,j] r1[c]   layout: ((a*nvir+b)*nvir+c)*nocc + j
    for (int c = 0; c < nvir; ++c) {
        s += Wvvvo[(((size_t)a * nvir + b) * nvir + c) * nocc + j] * r1[c];
    }

    // +Σ_c Lvv[a,c] r2[j,c,b]
    for (int c = 0; c < nvir; ++c) {
        s += Lvv[a * nvir + c] * r2[((size_t)j * nvir + c) * nvir + b];
    }
    // +Σ_d Lvv[b,d] r2[j,a,d]
    for (int d = 0; d < nvir; ++d) {
        s += Lvv[b * nvir + d] * r2[((size_t)j * nvir + a) * nvir + d];
    }

    // -Σ_l Loo[l,j] r2[l,a,b]
    for (int l = 0; l < nocc; ++l) {
        s -= Loo[l * nocc + j] * r2[((size_t)l * nvir + a) * nvir + b];
    }

    // Ring terms (skipped here when include_ring==0; handled by 3 cuBLAS GEMMs
    // over precomputed reorganized M matrices in apply()).
    if (include_ring) {
        // +Σ_{l,d} (2 Wovvo[l,b,d,j] - Wovov[l,b,j,d]) r2[l,a,d]     (R_A)
        //   Wovvo[l,b,d,j] layout: ((l*nvir+b)*nvir+d)*nocc + j
        //   Wovov[l,b,j,d] layout: ((l*nvir+b)*nocc+j)*nvir + d
        for (int l = 0; l < nocc; ++l)
            for (int d = 0; d < nvir; ++d) {
                real_t wovvo = Wovvo[(((size_t)l * nvir + b) * nvir + d) * nocc + j];
                real_t wovov = Wovov[(((size_t)l * nvir + b) * nocc + j) * nvir + d];
                s += (2.0 * wovvo - wovov) * r2[((size_t)l * nvir + a) * nvir + d];
            }

        // -Σ_{l,c} Wovov[l,a,j,c] r2[l,c,b]                          (R_B)
        for (int l = 0; l < nocc; ++l)
            for (int c = 0; c < nvir; ++c) {
                real_t w = Wovov[(((size_t)l * nvir + a) * nocc + j) * nvir + c];
                s -= w * r2[((size_t)l * nvir + c) * nvir + b];
            }

        // -Σ_{l,c} Wovvo[l,b,c,j] r2[l,c,a]                          (R_C)
        for (int l = 0; l < nocc; ++l)
            for (int c = 0; c < nvir; ++c) {
                real_t w = Wovvo[(((size_t)l * nvir + b) * nvir + c) * nocc + j];
                s -= w * r2[((size_t)l * nvir + c) * nvir + a];
            }
    }

    // +Σ_{c,d} Wvvvv[a,b,c,d] r2[j,c,d]    layout: ((a*nvir+b)*nvir+c)*nvir + d
    // (skipped here when include_wvvvv==0; handled by a single cuBLAS GEMM in apply())
    if (include_wvvvv)
        for (int c = 0; c < nvir; ++c)
            for (int d = 0; d < nvir; ++d) {
                s += Wvvvv[(((size_t)a * nvir + b) * nvir + c) * nvir + d]
                     * r2[((size_t)j * nvir + c) * nvir + d];
            }

    // -Σ_k tmp[k] · t2[k,j,a,b]    t2 layout: ((k*nocc+j)*nvir + a)*nvir + b
    for (int k = 0; k < nocc; ++k) {
        s -= d_tmp_k[k] * t2[(((size_t)k * nocc + j) * nvir + a) * nvir + b];
    }

    sigma2[((size_t)j * nvir + a) * nvir + b] = s;   // global index (slab writes its rows)
}

// ------------------------------------------------------------------
//  Ring-term GEMM acceleration helpers
// ------------------------------------------------------------------
// Build the 3 reorganized W matrices used by the σ2 ring GEMMs.  All three
// share the output linear index ((i1·nvir+v1)·nocc+i2)·nvir+v2 with
// out=(i1,v1), con=(i2,v2):
//   M_A[(j,b),(l,d)] = 2 Wovvo[l,b,d,j] - Wovov[l,b,j,d]
//   M_B[(j,a),(l,c)] = Wovov[l,a,j,c]
//   M_C[(j,b),(l,c)] = Wovvo[l,b,c,j]
__global__ void ea_build_ring_M_kernel(
    const real_t* __restrict__ Wovov,   // ((l*nvir+b)*nocc+j)*nvir + d
    const real_t* __restrict__ Wovvo,   // ((l*nvir+b)*nvir+d)*nocc + j
    real_t* __restrict__ M_A,
    real_t* __restrict__ M_B,
    real_t* __restrict__ M_C,
    int nocc, int nvir)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    if (idx >= total) return;
    int v2 = idx % nvir;
    size_t t = idx / nvir;
    int i2 = t % nocc;
    t /= nocc;
    int v1 = t % nvir;
    int i1 = t / nvir;
    real_t wovvo = Wovvo[(((size_t)i2 * nvir + v1) * nvir + v2) * nocc + i1];
    real_t wovov = Wovov[(((size_t)i2 * nvir + v1) * nocc + i1) * nvir + v2];
    M_A[idx] = 2.0 * wovvo - wovov;
    M_B[idx] = wovov;
    M_C[idx] = wovvo;
}

// r2T[(l,d),a] = r2[l,a,d]  (swap the two virtual indices, per occupied l)
__global__ void ea_r2_swap_vir_kernel(
    const real_t* __restrict__ r2, real_t* __restrict__ r2T, int nocc, int nvir)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // r2 layout (l,a,d)
    const size_t total = (size_t)nocc * nvir * nvir;
    if (idx >= total) return;
    int d = idx % nvir;
    size_t t = idx / nvir;
    int a = t % nvir;
    int l = t / nvir;
    r2T[((size_t)l * nvir + d) * nvir + a] = r2[idx];
}

// σ2[j,a,b] += tmp[(j,b),a]   (scatter the [(jb),a]-shaped R_A+R_C accumulator)
// Stage EA-5c: only the output-occ slab [j_begin, j_end) is scattered.
__global__ void ea_ring_scatter_kernel(
    const real_t* __restrict__ tmp, real_t* __restrict__ sigma2, int nocc, int nvir,
    int j_begin, int j_end)
{
    size_t lidx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // slab-local
    const size_t slabtot = (size_t)(j_end - j_begin) * nvir * nvir;
    if (lidx >= slabtot) return;
    int b = lidx % nvir;
    size_t t = lidx / nvir;
    int a = t % nvir;
    int j = j_begin + (int)(t / nvir);
    const size_t g = ((size_t)j * nvir + a) * nvir + b;
    sigma2[g] += tmp[((size_t)j * nvir + b) * nvir + a];
}

// ---- Wvvvo term3+4 device-resident repack/scatter (build_dressed hotspot) ----
// Replace the host repack of 7 NV³·NO / NO²·NV² arrays + 3 D2H + 3 host scatter
// with on-device gather kernels reading d_eri_ovvv_ ([o][v][v][v]) and d_t2_
// ([o][o][v][v]) straight off the device, a single accumulating scatter kernel,
// and one final D2H of the assembled wvvvo_big.  Grid-stride for 100-atom size_t
// safety.  Layouts mirror the validated host repacks (GANSU_STEOM_BUILD_VALIDATE).
//
// A-matrix [(x,c),(y,d)] gather off ovvv:
//   mode 0 (hAcomb): 2·ovvv[y,d,x,c] − ovvv[y,c,x,d]
//   mode 1 (hA)    :   ovvv[y,d,x,c]                  (x=a, y=l)
//   mode 2 (hA4)   :   ovvv[y,c,x,d]                  (x=b, y=k)
__global__ void ea_wvvvo_repack_A_kernel(
    const real_t* __restrict__ ovvv, real_t* __restrict__ dst, int NO, int NV, int mode)
{
    const size_t KV  = (size_t)NO * NV;
    const size_t tot = (size_t)NV * NV * KV;          // NV²·KV
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const size_t row = idx / KV, col = idx % KV;
        const int x = (int)(row / NV), c = (int)(row % NV);
        const int y = (int)(col / NV), d = (int)(col % NV);
        real_t v;
        if (mode == 0)
            v = 2.0 * ovvv[(((size_t)(y*NV+d))*NV + x)*NV + c]
                    -   ovvv[(((size_t)(y*NV+c))*NV + x)*NV + d];
        else if (mode == 1)
            v =         ovvv[(((size_t)(y*NV+d))*NV + x)*NV + c];
        else
            v =         ovvv[(((size_t)(y*NV+c))*NV + x)*NV + d];
        dst[idx] = v;
    }
}

// B-matrix [(p,d),(q,e)] gather off t2:
//   mode 0 (hB) : t2[p,q,d,e]   (p=l, q=j, e=b)
//   mode 1 (hBp): t2[p,q,e,d]   (p=l, q=j, e=b)
//   mode 2 (hB4): t2[q,p,d,e]   (p=k, q=j, e=a)
__global__ void ea_wvvvo_repack_B_kernel(
    const real_t* __restrict__ t2, real_t* __restrict__ dst, int NO, int NV, int mode)
{
    const size_t KV  = (size_t)NO * NV;
    const size_t tot = KV * KV;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const size_t row = idx / KV, col = idx % KV;
        const int p = (int)(row / NV), d = (int)(row % NV);
        const int q = (int)(col / NV), e = (int)(col % NV);
        real_t v;
        if (mode == 0)      v = t2[(((size_t)(p*NO+q))*NV + d)*NV + e];
        else if (mode == 1) v = t2[(((size_t)(p*NO+q))*NV + e)*NV + d];
        else                v = t2[(((size_t)(q*NO+p))*NV + d)*NV + e];
        dst[idx] = v;
    }
}

// Accumulate coeff·M into wvvvo_big[a,b,c,j] (= ((a*NV+b)*NV+c)*NO+j):
//   free_bc=0:  M[(a,c),(j,b)] = M[(a*NV+c)*KV + (j*NV+b)]
//   free_bc=1:  M[(b,c),(j,a)] = M[(b*NV+c)*KV + (j*NV+a)]
__global__ void ea_wvvvo_scatter_kernel(
    const real_t* __restrict__ M, real_t* __restrict__ big,
    real_t coeff, int free_bc, int NO, int NV)
{
    const size_t KV  = (size_t)NO * NV;
    const size_t tot = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const int j = (int)(idx % NO);
        const int c = (int)((idx / NO) % NV);
        const int b = (int)((idx / ((size_t)NO*NV)) % NV);
        const int a = (int)(idx / ((size_t)NO*NV*NV));
        const size_t mo = free_bc ? ((size_t)(b*NV+c)*KV + (size_t)(j*NV+a))
                                  : ((size_t)(a*NV+c)*KV + (size_t)(j*NV+b));
        big[idx] += coeff * M[mo];
    }
}

// ---- (RI Term A) Gather the virtual-virtual block of the half-transformed
//      B_mo (naux × nmo_full², row-major) into a contiguous [naux × nvir²]
//      buffer:  Bvv[P, a, b] = B_mo[P, voff+a, voff+b],  voff = nocc + frozen_off.
//      Lets the Wvvvo·t1 dressing contract B directly instead of materialising
//      the nvir⁴ (ab|cd) tensor. Grid-stride, size_t safe.
__global__ void ea_gather_bvv_kernel(const real_t* __restrict__ B_mo,
                                     real_t* __restrict__ Bvv,
                                     int naux, int nmo_full, int voff, int nvir)
{
    const size_t total  = (size_t)naux * nvir * nvir;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int b = (int)(idx % nvir);
        const size_t t = idx / nvir;
        const int a = (int)(t % nvir);
        const int P = (int)(t / nvir);
        Bvv[idx] = B_mo[(size_t)P * nmo_full * nmo_full
                      + (size_t)(voff + a) * nmo_full + (size_t)(voff + b)];
    }
}

// ---- ct_ovov / ct_ovvo GEMM-input builders (mirror of the validated IP-EOM
//      increment 5) — fill dA/dB/dA1/dB1/dA2/dB2 on device from the resident
//      d_eri_ovov_ / d_t2_, eliminating the 6×(NO·NV)² host arrays + ~1.35 GB H2D
//      (host omp-fill + memcpy was the build_dressed hotspot). EA ct_ovvo uses
//      (j,b) in the (Y,X) slot ⇒ byte-identical to IP. Grid-stride, size_t safe.
//   ovov[k,c,l,d] = ovov[((k*NV+c)*NO+l)*NV+d];  t2[i,j,c,d] = t2[((i*NO+j)*NV+c)*NV+d]
__global__ void ea_ctovov_buildA(real_t* __restrict__ dA,
                                 const real_t* __restrict__ ovov, int NO, int NV) {
    // dA[(k,d),(c,l)] = ovov[k,c,l,d];  enumerate (k,d,c,l)
    const size_t tot = (size_t)NO*NV*NV*NO, KOcl = (size_t)NV*NO;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int l = x % NO; size_t t = x / NO;
        const int c = t % NV; t /= NV;
        const int d = t % NV; const int k = (int)(t / NV);
        dA[((size_t)(k*NV+d))*KOcl + (c*NO+l)] = ovov[(((size_t)k*NV+c)*NO+l)*NV+d];
    }
}
__global__ void ea_ctovov_buildB(real_t* __restrict__ dB,
                                 const real_t* __restrict__ t2, int NO, int NV) {
    // dB[(c,l),(i,b)] = t2[i,l,c,b];  enumerate (c,l,i,b)
    const size_t tot = (size_t)NV*NO*NO*NV, NOib = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int b = x % NV; size_t t = x / NV;
        const int i = t % NO; t /= NO;
        const int l = t % NO; const int c = (int)(t / NO);
        dB[((size_t)(c*NO+l))*NOib + (i*NV+b)] = t2[(((size_t)i*NO+l)*NV+c)*NV+b];
    }
}
__global__ void ea_ctovvo_buildA12(real_t* __restrict__ dA1, real_t* __restrict__ dA2,
                                   const real_t* __restrict__ ovov, int NO, int NV) {
    // dA1[(k,c),(l,d)] = 2·ovov[k,c,l,d] − ovov[k,d,l,c];  dA2 = ovov[k,c,l,d];  enumerate (k,c,l,d)
    const size_t tot = (size_t)NO*NV*NO*NV, Nld = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = x % NV; size_t t = x / NV;
        const int l = t % NO; t /= NO;
        const int c = t % NV; const int k = (int)(t / NV);
        const real_t kcld = ovov[(((size_t)k*NV+c)*NO+l)*NV+d];
        const real_t kdlc = ovov[(((size_t)k*NV+d)*NO+l)*NV+c];
        const size_t o = ((size_t)(k*NV+c))*Nld + (l*NV+d);
        dA1[o] = 2.0*kcld - kdlc;  dA2[o] = kcld;
    }
}
__global__ void ea_ctovvo_buildB12(real_t* __restrict__ dB1, real_t* __restrict__ dB2,
                                   const real_t* __restrict__ t2, int NO, int NV) {
    // dB1[(l,d),(j,b)] = t2[j,l,b,d];  dB2 = t2[l,j,b,d];  enumerate (l,d,j,b)
    const size_t tot = (size_t)NO*NV*NO*NV, Njb = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int b = x % NV; size_t t = x / NV;
        const int j = t % NO; t /= NO;
        const int d = t % NV; const int l = (int)(t / NV);
        const size_t o = ((size_t)(l*NV+d))*Njb + (j*NV+b);
        dB1[o] = t2[(((size_t)j*NO+l)*NV+b)*NV+d];
        dB2[o] = t2[(((size_t)l*NO+j)*NV+b)*NV+d];
    }
}
// [EA build device-build] cc_Foo / cc_Fvv GEMM-input builders — fill hA/hB on
// device from d_eri_ovov_ / d_t2_ / d_t1_, eliminating the 4×(NO·NV)² host arrays
// (~900 MB) + H2D (EA build "cc_Foo + cc_Fvv" 0.835 s). Each flat thread index
// equals the output offset (layouts match the host repacks exactly).
//   ovov[k,c,l,d]=ovov[((k*NV+c)*NO+l)*NV+d];  t2[i,j,c,d]=t2[((i*NO+j)*NV+c)*NV+d]
__global__ void ea_ccfoo_buildA(real_t* __restrict__ dA,
                                const real_t* __restrict__ ovov, int NO, int NV) {
    // dA[k*Kf + (l*NV+c)*NV+d] = 2 ovov[k,c,l,d] − ovov[k,d,l,c];  enumerate (k,l,c,d)
    const size_t tot = (size_t)NO*NO*NV*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = (int)(x % NV); size_t t = x / NV;
        const int c = (int)(t % NV); t /= NV;
        const int l = (int)(t % NO); const int k = (int)(t / NO);
        const real_t kcld = ovov[(((size_t)k*NV+c)*NO+l)*NV+d];
        const real_t kdlc = ovov[(((size_t)k*NV+d)*NO+l)*NV+c];
        dA[x] = 2.0*kcld - kdlc;
    }
}
__global__ void ea_ccfoo_buildB(real_t* __restrict__ dB, const real_t* __restrict__ t2,
                                const real_t* __restrict__ t1, int NO, int NV) {
    // dB[((l*NV+c)*NV+d)*NO + i] = t2[i,l,c,d] + t1[i,c]·t1[l,d];  enumerate (l,c,d,i)
    const size_t tot = (size_t)NO*NV*NV*NO;
    for (size_t y = (size_t)blockIdx.x*blockDim.x + threadIdx.x; y < tot;
         y += (size_t)gridDim.x*blockDim.x) {
        const int i = (int)(y % NO); size_t t = y / NO;
        const int d = (int)(t % NV); t /= NV;
        const int c = (int)(t % NV); const int l = (int)(t / NV);
        dB[y] = t2[(((size_t)i*NO+l)*NV+c)*NV+d] + t1[(size_t)i*NV+c]*t1[(size_t)l*NV+d];
    }
}
__global__ void ea_ccfvv_buildA(real_t* __restrict__ dA, const real_t* __restrict__ t2,
                                const real_t* __restrict__ t1, int NO, int NV) {
    // dA[a*Kv + (k*NO+l)*NV+d] = t2[k,l,a,d] + t1[k,a]·t1[l,d];  enumerate (a,k,l,d)
    const size_t tot = (size_t)NV*NO*NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = (int)(x % NV); size_t t = x / NV;
        const int l = (int)(t % NO); t /= NO;
        const int k = (int)(t % NO); const int a = (int)(t / NO);
        dA[x] = t2[(((size_t)k*NO+l)*NV+a)*NV+d] + t1[(size_t)k*NV+a]*t1[(size_t)l*NV+d];
    }
}
__global__ void ea_ccfvv_buildB(real_t* __restrict__ dB,
                                const real_t* __restrict__ ovov, int NO, int NV) {
    // dB[((k*NO+l)*NV+d)*NV + c] = 2 ovov[k,c,l,d] − ovov[k,d,l,c];  enumerate (k,l,d,c)
    const size_t tot = (size_t)NO*NO*NV*NV;
    for (size_t y = (size_t)blockIdx.x*blockDim.x + threadIdx.x; y < tot;
         y += (size_t)gridDim.x*blockDim.x) {
        const int c = (int)(y % NV); size_t t = y / NV;
        const int d = (int)(t % NV); t /= NV;
        const int l = (int)(t % NO); const int k = (int)(t / NO);
        const real_t kcld = ovov[(((size_t)k*NV+c)*NO+l)*NV+d];
        const real_t kdlc = ovov[(((size_t)k*NV+d)*NO+l)*NV+c];
        dB[y] = 2.0*kcld - kdlc;
    }
}

// ---- canonical-skip wvvvo_w_t1 device-resident scatter (Term A/B/C/D) ----
// Accumulate coeff·src into wt1[a,b,c,j] (= ((a*NV+b)*NV+c)*NO+j), reading the
// per-term GEMM result still resident on the device.  Eliminates the 4× D2H of
// the NV³·NO results + 4 host scatter loops (SUBPROF "Wvvvv / wvvvo_w_t1").
//   mode 0 (Term D, identity): src[a,b,c,j]
//   mode 1 (Term A/B, b↔c)   : src[a,c,b,j]
//   mode 2 (Term C, a↔b)     : src[b,a,c,j]
__global__ void ea_wvvvo_wt1_scatter_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ wt1,
    real_t coeff, int mode, int NO, int NV)
{
    const size_t tot = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const int j = (int)(idx % NO);
        const int c = (int)((idx / NO) % NV);
        const int b = (int)((idx / ((size_t)NO*NV)) % NV);
        const int a = (int)(idx / ((size_t)NO*NV*NV));
        size_t s;
        if (mode == 0)      s = (((size_t)(a*NV+b)*NV + c)*NO) + j;
        else if (mode == 1) s = (((size_t)(a*NV+c)*NV + b)*NO) + j;
        else                s = (((size_t)(b*NV+a)*NV + c)*NO) + j;
        wt1[idx] += coeff * src[s];
    }
}

// ---- Wvovv device build (broader refactor stage 1) ----
// d_Wvovv[a,l,c,d] (= ((a*NO+l)*NV+c)*NV+d) = ovvv[l,d,a,c] − ct[a,(c,l,d)]
// reads d_eri_ovvv_ (bare term) + the resident GEMM result ct (Σ_k t1·ovov),
// writing d_Wvovv_ straight on the device — no host assemble, no H2D upload.
__global__ void ea_wvovv_assemble_kernel(
    const real_t* __restrict__ ovvv, const real_t* __restrict__ ct,
    real_t* __restrict__ wvovv, int NO, int NV)
{
    const size_t tot = (size_t)NV * NO * NV * NV;
    const size_t Nv  = (size_t)NV * NO * NV;   // ct column dim = (c,l,d)
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const int d = (int)(idx % NV);
        const int c = (int)((idx / NV) % NV);
        const int l = (int)((idx / ((size_t)NV*NV)) % NO);
        const int a = (int)(idx / ((size_t)NO*NV*NV));
        const real_t bare = ovvv[(((size_t)(l*NV+d))*NV + a)*NV + c];   // ovvv[l,d,a,c]
        const real_t cv   = ct[(size_t)a*Nv + ((size_t)c*NO*NV + l*NV + d)];
        wvovv[idx] = bare - cv;
    }
}

// ---- Wvvvo device assembly (broader refactor stage 3, canonical-skip path) ----
// d_Wvvvo[a,b,c,j] (= ((a*NV+b)*NV+c)*NO+j) assembled from the per-term device
// buffers kept resident (no D2H, no host assemble, no H2D upload).  Layouts
// mirror the validated host assembly:
//   + ovvv[j,b,c,a]                                     (bare)
//   − ct1[b·N_w1 + (a·NO·NV + j·NV + c)]                N_w1 = NV·NO·NV
//   − ct2[a·N_w2 + (b·NV·NO + c·NO + j)]                N_w2 = NV·NV·NO
//   + big[a,b,c,j]                                      (term3+4)
//   + t5[(j·NV+c)·NV² + (b·NV+a)]                       (term5)
//   − ct5[c·N_w5 + (j·NV² + a·NV + b)]                  N_w5 = NO·NV·NV
//   + wt1[a,b,c,j]                                      (Σ_d Wvvvv·t1, canonical-skip)
__global__ void ea_wvvvo_assemble_kernel(
    const real_t* __restrict__ ovvv, const real_t* __restrict__ ct1,
    const real_t* __restrict__ ct2,  const real_t* __restrict__ big,
    const real_t* __restrict__ t5,   const real_t* __restrict__ ct5,
    const real_t* __restrict__ wt1,  real_t* __restrict__ wvvvo, int NO, int NV)
{
    const size_t tot   = (size_t)NV * NV * NV * NO;
    const size_t NONV  = (size_t)NO * NV;
    const size_t NV2   = (size_t)NV * NV;
    const size_t N_w1  = (size_t)NV * NO * NV;
    const size_t N_w2  = (size_t)NV * NV * NO;
    const size_t N_w5  = (size_t)NO * NV * NV;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot; idx += stride) {
        const int j = (int)(idx % NO);
        const int c = (int)((idx / NO) % NV);
        const int b = (int)((idx / NONV) % NV);
        const int a = (int)(idx / ((size_t)NO*NV*NV));
        real_t v = ovvv[(((size_t)(j*NV+b))*NV + c)*NV + a];                       // bare
        v -= ct1[(size_t)b*N_w1 + ((size_t)a*NONV + (size_t)j*NV + c)];
        v -= ct2[(size_t)a*N_w2 + ((size_t)b*NONV + (size_t)c*NO + j)];
        v += big[idx];
        v += t5[((size_t)(j*NV+c))*NV2 + ((size_t)b*NV + a)];
        v -= ct5[(size_t)c*N_w5 + ((size_t)j*NV2 + (size_t)a*NV + b)];
        v += wt1[idx];
        wvvvo[idx] = v;
    }
}
#endif  // !GANSU_CPU_ONLY


// ==================================================================
//  Constructor / destructor
// ==================================================================

EAEOMCCSDOperator::EAEOMCCSDOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    int nocc, int nvir, int nao,
    const ERI_RI* eri_block_src,
    const real_t* d_B_mo_blocks,
    int nmo_full,
    int num_gpus,
    std::vector<real_t*>* d_eri_vvvv_slabs_input,
    SteomBarHCache* barh_cache,
    int frozen_off)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      p_dim_(nvir),
      p2h_dim_(nocc * nvir * nvir),
      total_dim_(nvir + nocc * nvir * nvir),
      d_t1_(d_t1), d_t2_(d_t2),
      eri_block_src_(eri_block_src), d_B_mo_blocks_(d_B_mo_blocks), nmo_full_(nmo_full),
      frozen_off_(frozen_off),
      barh_cache_(barh_cache)
{
    // Ship 12 — take ownership of per-device d_eri_vvvv slabs allocated +
    // extracted by the driver (compute_ea_eom_ccsd_impl).  Slab boundaries
    // are uniform along the outermost a-axis; the consumer GEMM
    // (canonical-skip Term A) runs independently on each device against its
    // own slab + a broadcast t1.
    if (d_eri_vvvv_slabs_input && (int)d_eri_vvvv_slabs_input->size() >= 2) {
        const int N = (int)d_eri_vvvv_slabs_input->size();
        eri_vvvv_nslab_ = N;
        d_eri_vvvv_slabs_ = std::move(*d_eri_vvvv_slabs_input);
        a_starts_.assign(N, 0);
        a_ends_.assign(N, 0);
        for (int d = 0; d < N; ++d) {
            a_starts_[d] = (int)((int64_t)d * nvir / N);
            a_ends_[d]   = (int)((int64_t)(d + 1) * nvir / N);
        }
        std::cout << "  [EA-EOM Ship 12] d_eri_vvvv slab mode ON: N="
                  << N << " devices, a-axis split [";
        for (int d = 0; d < N; ++d)
            std::cout << a_starts_[d] << "-" << a_ends_[d]
                      << (d + 1 < N ? "," : "]");
        const double per_dev_gb = (double)(a_ends_[0] - a_starts_[0])
                                  * nvir * nvir * nvir * sizeof(real_t)
                                  / (1024.0 * 1024.0 * 1024.0);
        std::cout << ", slab[0] ≈ " << std::fixed << std::setprecision(2)
                  << per_dev_gb << " GB per device" << std::defaultfloat
                  << std::endl;
    }
    if (nocc <= 0 || nvir <= 0 || nao != nocc + nvir) {
        throw std::invalid_argument(
            "EAEOMCCSDOperator: invalid (nocc, nvir, nao) — require nao == nocc + nvir, both positive");
    }

    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
    // P5 canonical-skip: active only when the native per-pair σ takes over both σ2
    // and the dressed-W reads. Master-switch (2026-06-03): NATIVE_EOM is the explicit
    // gate (default OFF here — this canonical operator is also built on the reference
    // path, where skip must stay off for the bit-exact full build); NATIVE_BARE and
    // CANONICAL_SKIP default ON under it (so NATIVE_EOM=1 alone enables skip; "=0" on
    // either opts out).
    {
        auto on = [](const char* n, bool d) {
            const char* e = std::getenv(n);
            return (!e || !e[0]) ? d : (e[0] != '0');
        };
        canonical_skip_wvvvv_ = on("GANSU_DLPNO_NATIVE_EOM", false) &&
                                on("GANSU_DLPNO_NATIVE_BARE", true) &&
                                on("GANSU_DLPNO_CANONICAL_SKIP", true);
        if (canonical_skip_wvvvv_)
            std::cout << "  [EA-EOM canonical-skip] dressed Wvvvv build SKIPPED "
                         "(nvir⁴ host+device elided; Wvvvo·t1 refactored)" << std::endl;
        // (RI Term A) Evaluate the Wvvvo·t1 dressing from RI B-factors so the
        // nvir⁴ (ab|cd) block is never materialised. Requires the canonical-skip
        // path (Wvvvo·t1 is then the sole vvvv consumer) and an RI block source.
        ri_vvvv_term_a_ = canonical_skip_wvvvv_ && (eri_block_src_ != nullptr)
                          && on("GANSU_DLPNO_EA_VVVV_RI", true);
        if (ri_vvvv_term_a_)
            std::cout << "  [EA-EOM RI-Term-A] Wvvvo·t1 via RI B-factors "
                         "(d_eri_vvvv nvir⁴ not materialised)" << std::endl;
    }
    // Ship 12: vvvv slab mode requires canonical-skip ON.  The legacy
    // non-skip code path D2H's d_eri_vvvv_ (nullptr in slab mode) and reads
    // h_vvvv (empty in slab mode); both would crash without skip.  Force the
    // flag and warn if the env disagrees so the user notices.
    if (eri_vvvv_nslab_ > 1 && !canonical_skip_wvvvv_) {
        canonical_skip_wvvvv_ = true;
        std::cout << "  [EA-EOM Ship 12] forcing canonical_skip_wvvvv_=true "
                     "(slab mode requires skip path; set GANSU_DLPNO_CANONICAL_SKIP=1 "
                     "to silence this notice)" << std::endl;
    }
    // Sub-phase 2.2: when d_eri_mo is provided, extract the EA-needed MO ERI
    // blocks and build PySCF-equivalent dressed intermediates.
    // (Unit tests pass d_eri_mo == nullptr and exercise only the diagonal.)
    if (d_eri_mo != nullptr || eri_block_src_ != nullptr) {  // Phase 0: block source
        // Per-phase build profiling.  Default ON so each phase prints START
        // and END markers — the canonical EA-EOM operator build is a long
        // silent stretch otherwise (anthracene ~160 s, tetracene ~236 s, no
        // Davidson activity to fall back on).  Set GANSU_EOM_BUILD_PROF=0
        // to silence both markers.
        const char* env_prof = std::getenv("GANSU_EOM_BUILD_PROF");
        const bool prof = !env_prof || env_prof[0] != '0';
        auto tphase = [&](const char* name, auto&& fn) {
            if (!prof) { fn(); return; }
            std::cout << "  [EA-EOM build-PROF] " << name << " ..." << std::endl;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
            const auto t0 = std::chrono::high_resolution_clock::now();
            fn();
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
            const double s = std::chrono::duration<double>(
                                 std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  [EA-EOM build-PROF] " << name << " = " << std::fixed
                      << std::setprecision(3) << s << " s" << std::defaultfloat << std::endl;
        };
        tphase("extract_eri_blocks",          [&]{ extract_eri_blocks(d_eri_mo); });
        tphase("build_dressed_intermediates", [&]{ build_dressed_intermediates(); });
        // (A) shared bar-H: publish the 3 EA-unique intermediates (IP already
        // published the 8 shared ones). Wvvvv may be nullptr under canonical-skip
        // (consistent with STEOM, which also skips it). EA's Loo/Lvv/Fov/Wovov/
        // Wovvo are separate copies left owned by EA (cache holds IP's).
        if (barh_cache_ != nullptr) {
            barh_cache_->d_Wvovv = d_Wvovv_;
            barh_cache_->d_Wvvvv = d_Wvvvv_;   // nullptr under canonical-skip
            barh_cache_->d_Wvvvo = d_Wvvvo_;
            barh_cache_->canonical_skip_wvvvv = canonical_skip_wvvvv_;
            if (barh_cache_->nocc == 0) { barh_cache_->nocc = nocc_; barh_cache_->nvir = nvir_; }
            barh_cache_->has_ea  = true;
            barh_published_      = true;
            std::cout << "  [STEOM share-barH] EA published 3 bar-H intermediates "
                         "(Wvovv/Wvvvv/Wvvvo; canonical_skip_wvvvv="
                      << (canonical_skip_wvvvv_ ? "1" : "0") << ")." << std::endl;
        }
        num_gpus_ = (num_gpus > 1 ? num_gpus : 1);
        setup_multi_gpu();   // Stage EA-5a: per-device replicas (no-op when num_gpus_==1)
    }
}

EAEOMCCSDOperator::~EAEOMCCSDOperator() {
    if (d_t1_)        tracked_cudaFree(d_t1_);
    if (d_t2_)        tracked_cudaFree(d_t2_);
    if (d_D_p_)       tracked_cudaFree(d_D_p_);
    if (d_D_p2h_)     tracked_cudaFree(d_D_p2h_);
    if (d_f_oo_)      tracked_cudaFree(d_f_oo_);
    if (d_f_vv_)      tracked_cudaFree(d_f_vv_);
    if (d_diagonal_)  tracked_cudaFree(d_diagonal_);
    if (d_eri_oooo_)  tracked_cudaFree(d_eri_oooo_);
    if (d_eri_ooov_)  tracked_cudaFree(d_eri_ooov_);
    if (d_eri_oovv_)  tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovov_)  tracked_cudaFree(d_eri_ovov_);
    if (d_eri_ovvo_)  tracked_cudaFree(d_eri_ovvo_);
    if (d_eri_ovvv_)  tracked_cudaFree(d_eri_ovvv_);
    if (d_eri_vvvv_)  tracked_cudaFree(d_eri_vvvv_);
    // Ship 12: free per-device vvvv slabs (current device may not match the
    // alloc device, so wrap each free in cudaSetDevice).
#ifndef GANSU_CPU_ONLY
    if (!d_eri_vvvv_slabs_.empty()) {
        int saved = 0;
        cudaGetDevice(&saved);
        for (size_t d = 0; d < d_eri_vvvv_slabs_.size(); ++d) {
            if (d_eri_vvvv_slabs_[d]) {
                cudaSetDevice((int)d);
                tracked_cudaFree(d_eri_vvvv_slabs_[d]);
                d_eri_vvvv_slabs_[d] = nullptr;
            }
        }
        cudaSetDevice(saved);
    }
#endif
    // EA's Loo/Lvv/Fov/Wovov/Wovvo are always EA-owned (the shared cache holds
    // IP's bit-identical copies), so free them unconditionally.
    if (d_Loo_)       tracked_cudaFree(d_Loo_);
    if (d_Lvv_)       tracked_cudaFree(d_Lvv_);
    if (d_Fov_)       tracked_cudaFree(d_Fov_);
    if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
    if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
    // (A) shared bar-H: Wvovv/Wvvvv/Wvvvo published to the cache (owned by the
    // STEOM driver now) — skip to avoid double-free / use-after-free.
    if (!barh_published_) {
        if (d_Wvovv_)     tracked_cudaFree(d_Wvovv_);
        if (d_Wvvvv_)     tracked_cudaFree(d_Wvvvv_);
        if (d_Wvvvo_)     tracked_cudaFree(d_Wvvvo_);
    }
    if (d_M_ringA_)   tracked_cudaFree(d_M_ringA_);
    if (d_M_ringB_)   tracked_cudaFree(d_M_ringB_);
    if (d_M_ringC_)   tracked_cudaFree(d_M_ringC_);
#ifndef GANSU_CPU_ONLY
    // Stage EA-5: free per-device replicas (skip ws_[0], which aliases the
    // device-0 members freed above).
    for (size_t d = 1; d < ws_.size(); ++d) {
        DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(w.device);
        for (real_t* p : {w.d_input, w.d_s1, w.d_s2, w.d_tmp_k, w.d_r2T, w.d_ring_tmp,
                          w.d_Lvv, w.d_Loo, w.d_Fov, w.d_Wovov, w.d_Wovvo, w.d_Wvovv,
                          w.d_Wvvvo, w.d_Wvvvv, w.d_t2, w.d_eri_ovov,
                          w.d_M_ringA, w.d_M_ringB, w.d_M_ringC})
            if (p) tracked_cudaFree(p);
        if (w.cublas) cublasDestroy((cublasHandle_t)w.cublas);
    }
    // ws_[0] only owns its σ scratch (other pointers alias the device-0 members
    // freed above); free just the scratch on device 0.
    if (!ws_.empty()) {
        MultiGpuManager::DeviceGuard guard(0);
        if (ws_[0].d_tmp_k)    tracked_cudaFree(ws_[0].d_tmp_k);
        if (ws_[0].d_r2T)      tracked_cudaFree(ws_[0].d_r2T);
        if (ws_[0].d_ring_tmp) tracked_cudaFree(ws_[0].d_ring_tmp);
    }
#endif
}

// ==================================================================
//  Stage EA-5a: multi-GPU scaffolding — per-device replicas of all
//  σ-build intermediates (peer-copied from device 0).  Gated by
//  GANSU_STEOM_EOM_GPUS=N>1 (decoupled from RI/CIS-NTO --num_gpus).
//  Pure setup: apply() is unchanged this stage (slab σ2 + gather land
//  in EA-5b/5c).  num_gpus_==1 → no-op (byte-identical legacy path).
// ==================================================================
void EAEOMCCSDOperator::setup_multi_gpu() {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || num_gpus_ <= 1) return;
    const char* e = std::getenv("GANSU_STEOM_EOM_GPUS");
    const int env_gpus = (e && e[0]) ? std::atoi(e) : 0;
    if (env_gpus <= 1) return;   // decoupled opt-in; default = legacy single-GPU

    // Decouple the EA σ device count from the RI/CIS-NTO MultiGpuManager singleton,
    // which is already initialized to 1 device by the --num_gpus 1 HF/RI path
    // (initialize() is a no-op after the first call).  Use the physical device count
    // directly and create our own per-device cuBLAS handles (the singleton only holds
    // a device-0 handle).  MultiGpuManager::DeviceGuard is a pure cudaSetDevice RAII
    // (no singleton state) and is safe to reuse.
    int phys = 0;
    cudaGetDeviceCount(&phys);
    const int nuse = std::min({num_gpus_, env_gpus, phys});
    if (nuse <= 1) return;
    use_gpu_multi_ = true;

    const size_t NO = nocc_, NV = nvir_;
    const size_t loo_sz = NO*NO, lvv_sz = NV*NV, fov_sz = NO*NV;
    const size_t ovov_sz = NO*NV*NO*NV;          // == ovvo, M_ring, eri_ovov size
    const size_t wvovv_sz = NV*NO*NV*NV;
    const size_t vvvv_sz = NV*NV*NV*NV;
    const size_t wvvvo_sz = NV*NV*NV*NO;
    const size_t t2_sz = NO*NO*NV*NV;
    const size_t p2h = (size_t)p2h_dim_, total = (size_t)total_dim_;

    ws_.resize(nuse);
    // ws_[0] aliases the device-0 members (exact restore; no alloc/copy).
    {
        DeviceWorkspace& w = ws_[0];
        w.device = 0; w.cublas = nullptr;
        w.d_Lvv=d_Lvv_; w.d_Loo=d_Loo_; w.d_Fov=d_Fov_; w.d_Wovov=d_Wovov_;
        w.d_Wovvo=d_Wovvo_; w.d_Wvovv=d_Wvovv_; w.d_Wvvvo=d_Wvvvo_; w.d_Wvvvv=d_Wvvvv_;
        w.d_t2=d_t2_; w.d_eri_ovov=d_eri_ovov_;
        w.d_M_ringA=d_M_ringA_; w.d_M_ringB=d_M_ringB_; w.d_M_ringC=d_M_ringC_;
        auto p0 = aux_partition(NO, nuse, 0);
        w.j_begin = (int)p0.first; w.j_end = (int)p0.second;
        // EA-5d: device-0 persistent σ scratch (avoid per-matvec cudaMalloc).
        tracked_cudaMalloc(&w.d_tmp_k,    NO * sizeof(real_t));
        tracked_cudaMalloc(&w.d_r2T,      p2h * sizeof(real_t));
        tracked_cudaMalloc(&w.d_ring_tmp, p2h * sizeof(real_t));
    }
    size_t bytes_per_dev = 0;
    for (int d = 1; d < nuse; ++d) {
        DeviceWorkspace& w = ws_[d];
        w.device = d;
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t h = nullptr;
        cublasCreate(&h);          // our own per-device handle (destroyed in dtor)
        w.cublas = (void*)h;
        auto pd = aux_partition(NO, nuse, d);
        w.j_begin = (int)pd.first; w.j_end = (int)pd.second;
        // alloc a device-d buffer, peer-copy from the device-0 source.
        auto repl = [&](real_t** dst, const real_t* src0, size_t n) {
            tracked_cudaMalloc(dst, n * sizeof(real_t));
            cudaMemcpyPeer(*dst, d, src0, 0, n * sizeof(real_t));
            bytes_per_dev += n * sizeof(real_t);
        };
        repl(&w.d_Lvv,   d_Lvv_,   lvv_sz);
        repl(&w.d_Loo,   d_Loo_,   loo_sz);
        repl(&w.d_Fov,   d_Fov_,   fov_sz);
        repl(&w.d_Wovov, d_Wovov_, ovov_sz);
        repl(&w.d_Wovvo, d_Wovvo_, ovov_sz);
        repl(&w.d_Wvovv, d_Wvovv_, wvovv_sz);
        repl(&w.d_Wvvvo, d_Wvvvo_, wvvvo_sz);
        if (d_Wvvvv_)
            repl(&w.d_Wvvvv, d_Wvvvv_, vvvv_sz);
        // else: canonical-skip → canonical σ2 not invoked, replica unused
        repl(&w.d_t2,    d_t2_,    t2_sz);
        repl(&w.d_eri_ovov, d_eri_ovov_, ovov_sz);
        repl(&w.d_M_ringA, d_M_ringA_, ovov_sz);
        repl(&w.d_M_ringB, d_M_ringB_, ovov_sz);
        repl(&w.d_M_ringC, d_M_ringC_, ovov_sz);
        // per-device scratch (uninitialized; used by EA-5b/5c apply_multi).
        tracked_cudaMalloc(&w.d_input,    total * sizeof(real_t));
        tracked_cudaMalloc(&w.d_s1,       (size_t)p_dim_ * sizeof(real_t));
        tracked_cudaMalloc(&w.d_s2,       p2h * sizeof(real_t));
        tracked_cudaMalloc(&w.d_tmp_k,    NO * sizeof(real_t));
        tracked_cudaMalloc(&w.d_r2T,      p2h * sizeof(real_t));
        tracked_cudaMalloc(&w.d_ring_tmp, p2h * sizeof(real_t));
        bytes_per_dev += (total + p_dim_ + 3*p2h + NO) * sizeof(real_t);
    }
    MultiGpuManager::DeviceGuard guard0(0);  // restore device 0
    const double gb_per_dev = (nuse > 1)
        ? (double)bytes_per_dev / (double)(nuse - 1) / (1024.0*1024.0*1024.0) : 0.0;
    std::cout << "[EA-EOM Stage 5c] multi-GPU σ: nuse=" << nuse
              << "  j-slab over nocc=" << NO
              << "  per-device replicas ≈ " << std::fixed << std::setprecision(2)
              << gb_per_dev << " GB"
              << std::defaultfloat << " (each device computes its σ2 j-slab → disjoint gather)" << std::endl;
#endif
}

void EAEOMCCSDOperator::compute_denominators_and_fock(const real_t* d_orbital_energies) {
    tracked_cudaMalloc(&d_D_p_,   (size_t)p_dim_   * sizeof(real_t));
    tracked_cudaMalloc(&d_D_p2h_, (size_t)p2h_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_oo_,  (size_t)nocc_    * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv_,  (size_t)nvir_    * sizeof(real_t));

    if (!gpu::gpu_available()) {
        for (int a = 0; a < nvir_; ++a) d_D_p_[a] = d_orbital_energies[a + nocc_];
        #pragma omp parallel for
        for (int idx = 0; idx < p2h_dim_; ++idx) {
            int b  = idx % nvir_;
            int t  = idx / nvir_;
            int a  = t % nvir_;
            int j  = t / nvir_;
            d_D_p2h_[idx] = -d_orbital_energies[j]
                          +  d_orbital_energies[a + nocc_]
                          +  d_orbital_energies[b + nocc_];
        }
        for (int i = 0; i < nocc_; ++i) d_f_oo_[i] = d_orbital_energies[i];
        for (int a = 0; a < nvir_; ++a) d_f_vv_[a] = d_orbital_energies[a + nocc_];
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        real_t* d_packed = nullptr;
        tracked_cudaMalloc(&d_packed, (size_t)total_dim_ * sizeof(real_t));
        ea_eom_build_diagonal_kernel<<<blocks, threads>>>(
            d_orbital_energies, d_packed, nocc_, nvir_);
        cudaDeviceSynchronize();
        cudaMemcpy(d_D_p_,   d_packed,           (size_t)p_dim_   * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_D_p2h_, d_packed + p_dim_,  (size_t)p2h_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        tracked_cudaFree(d_packed);
        cudaMemcpy(d_f_oo_, d_orbital_energies,         (size_t)nocc_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_f_vv_, d_orbital_energies + nocc_, (size_t)nvir_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
    }
}

void EAEOMCCSDOperator::build_diagonal() {
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (int i = 0; i < p_dim_;   ++i) d_diagonal_[i]          = d_D_p_[i];
        for (int i = 0; i < p2h_dim_; ++i) d_diagonal_[p_dim_ + i] = d_D_p2h_[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_diagonal_,           d_D_p_,   (size_t)p_dim_   * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_diagonal_ + p_dim_,  d_D_p2h_, (size_t)p2h_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
    }
}

void EAEOMCCSDOperator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_, nvir = nvir_, nao = nao_;
    size_t nao2 = (size_t)nao * nao;
    size_t N    = (size_t)nao;

    const size_t ovov_sz = (size_t)nocc * nvir * nocc * nvir;
    const size_t ooov_sz = (size_t)nocc * nocc * nocc * nvir;
    const size_t oooo_sz = (size_t)nocc * nocc * nocc * nocc;
    const size_t oovv_sz = (size_t)nocc * nocc * nvir * nvir;
    const size_t ovvo_sz = (size_t)nocc * nvir * nvir * nocc;
    const size_t ovvv_sz = (size_t)nocc * nvir * nvir * nvir;
    const size_t vvvv_sz = (size_t)nvir * nvir * nvir * nvir;

    tracked_cudaMalloc(&d_eri_oooo_, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ooov_, ooov_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovov_, ovov_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_oovv_, oovv_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovvo_, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovvv_, ovvv_sz * sizeof(real_t));
    // Ship 12: in slab mode d_eri_vvvv_ stays null; each device owns its
    // slab via d_eri_vvvv_slabs_[d] allocated below in the P4b branch.
    // (RI Term A) skip the nvir⁴ alloc entirely when Wvvvo·t1 is evaluated from
    // the B-factors; keep it under BUILD_VALIDATE so the host self-check
    // reference (h_vvvv, D2H'd from d_eri_vvvv_) still survives.
    const bool keep_dense_vvvv = !ri_vvvv_term_a_
                                 || std::getenv("GANSU_STEOM_BUILD_VALIDATE") != nullptr;
    if (eri_vvvv_nslab_ <= 1 && keep_dense_vvvv) {
        tracked_cudaMalloc(&d_eri_vvvv_, vvvv_sz * sizeof(real_t));
    }

#ifndef GANSU_CPU_ONLY
    // Phase 0: build the 7 blocks on the fly from B_mo (naux×nmo²), never the
    // full nmo⁴. o=[0,nocc), v=[nocc,nmo). Layouts match the gather kernels below.
    if (eri_block_src_ != nullptr) {
        const int M = nmo_full_;
        // Frozen core: B_mo spans the full C (M MOs); shift every range start by
        // frozen_off_ so the active occ [O,O+nocc) / vir [O+nocc,O+nocc+nvir)
        // window is read. O = 0 ⇒ non-frozen (byte-identical).
        const int O = frozen_off_;
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,O,nocc,           d_eri_oooo_); // (ij|kl)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,nocc+O,nvir,      d_eri_ooov_); // (ji|kb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    O,nocc,nocc+O,nvir,      d_eri_ovov_); // (ia|jb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         nocc+O,nvir,nocc+O,nvir, d_eri_oovv_); // (ij|ab)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,O,nocc,      d_eri_ovvo_); // (ia|bj)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,nocc+O,nvir, d_eri_ovvv_); // (ia|bc)

        // Ship 12: in slab mode the d_eri_vvvv slabs are already allocated
        // and populated by the driver (compute_ea_eom_ccsd_impl) before this
        // ctor was reached — operator just owns them. Skip the legacy
        // single-device vvvv extract here.
        if (eri_vvvv_nslab_ <= 1 && keep_dense_vvvv) {
            eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, nocc+O,nvir,nocc+O,nvir,nocc+O,nvir,nocc+O,nvir,d_eri_vvvv_); // (ab|cd)
        }
        cudaDeviceSynchronize();
        return;
    }
#endif

    if (!gpu::gpu_available()) {
        // CPU fallback — explicit nested loops, mirroring src/ip_eom_ccsd_operator.cu.
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovov_sz; ++idx) {
            int i = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int a = rem / (nocc * nvir); rem %= (nocc * nvir);
            int j = rem / nvir; int b = rem % nvir;
            d_eri_ovov_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)j*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ooov_sz; ++idx) {
            int j = idx / (nocc * nocc * nvir);
            int rem = idx % (nocc * nocc * nvir);
            int i = rem / (nocc * nvir); rem %= (nocc * nvir);
            int k = rem / nvir; int b = rem % nvir;
            d_eri_ooov_[idx] = d_eri_mo[((size_t)j*nao + i)*nao2 + (size_t)k*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oooo_sz; ++idx) {
            int i = idx / (nocc * nocc * nocc);
            int rem = idx % (nocc * nocc * nocc);
            int j = rem / (nocc * nocc); rem %= (nocc * nocc);
            int k = rem / nocc; int l = rem % nocc;
            d_eri_oooo_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)k*nao + l];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oovv_sz; ++idx) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir); rem %= (nvir * nvir);
            int a = rem / nvir; int b = rem % nvir;
            d_eri_oovv_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovvo_sz; ++idx) {
            int i = idx / (nvir * nvir * nocc);
            int rem = idx % (nvir * nvir * nocc);
            int a = rem / (nvir * nocc); rem %= (nvir * nocc);
            int b = rem / nocc; int j = rem % nocc;
            d_eri_ovvo_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
        }
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovvv_sz; ++idx) {
            int i = (int)(idx / ((size_t)nvir * nvir * nvir));
            size_t rem2 = idx % ((size_t)nvir * nvir * nvir);
            int a = (int)(rem2 / ((size_t)nvir * nvir));
            rem2 %= ((size_t)nvir * nvir);
            int b = (int)(rem2 / nvir);
            int c = (int)(rem2 % nvir);
            size_t mo_idx = (size_t)i * N * N * N + (size_t)(a + nocc) * N * N
                          + (size_t)(b + nocc) * N + (size_t)(c + nocc);
            d_eri_ovvv_[idx] = d_eri_mo[mo_idx];
        }
        #pragma omp parallel for
        for (size_t idx = 0; idx < vvvv_sz; ++idx) {
            int a = (int)(idx / ((size_t)nvir * nvir * nvir));
            size_t rem2 = idx % ((size_t)nvir * nvir * nvir);
            int b = (int)(rem2 / ((size_t)nvir * nvir));
            rem2 %= ((size_t)nvir * nvir);
            int c = (int)(rem2 / nvir);
            int d = (int)(rem2 % nvir);
            size_t mo_idx = (size_t)(a + nocc) * N * N * N + (size_t)(b + nocc) * N * N
                          + (size_t)(c + nocc) * N + (size_t)(d + nocc);
            d_eri_vvvv_[idx] = d_eri_mo[mo_idx];
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        int blocks;
        blocks = (ovov_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_, nvir_, nao_);
        blocks = (ooov_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_, nvir_, nao_);
        blocks = (oooo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_, nao_, 0);
        blocks = (oovv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_, nvir_, nao_, 0, -1);
        blocks = (ovvo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_, nvir_, nao_);
        blocks = (ovvv_sz + threads - 1) / threads;
        eom_ccsd_extract_eri_ovvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvv_, nocc_, nvir_, nao_);
        blocks = (vvvv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_, nvir_, nao_, 0, -1);
        cudaDeviceSynchronize();
#endif
    }
}


// ==================================================================
//  build_dressed_intermediates — PySCF EA-EOM-CCSD versions
// ==================================================================
//  References (PySCF cc/rintermediates.py):
//    cc_Foo / cc_Fvv / cc_Fov   (lines 30-59)  — shared helpers
//    Loo / Lvv                  (lines 63-79)  — shared with IP
//    W1ovov / W2ovov / Wovov    (lines 155-168) — shared with IP
//    W1ovvo / W2ovvo / Wovvo    (lines 137-153) — shared with IP
//    Wvovv                      (lines 131-135) — EA-specific
//    Wvvvv                      (lines 180-188) — EA-specific
//    Wvvvo                      (lines 190-210) — EA-specific
// Host-side index macros (NO = nocc, NV = nvir).
#define H_OVOV(p,a,q,b) h_ovov[(size_t)(p)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(q)*NV + (b)]
#define H_OOOV(p,q,r,a) h_ooov[(size_t)(p)*NO*NO*NV + (size_t)(q)*NO*NV + (size_t)(r)*NV + (a)]
#define H_OOVV(p,q,a,b) h_oovv[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
#define H_OVVO(p,a,b,q) h_ovvo[(size_t)(p)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(b)*NO + (q)]
#define H_OVVV(p,a,b,c) h_ovvv[(size_t)(p)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
#define H_VVVV(a,b,c,d) h_vvvv[(size_t)(a)*NV*NV*NV + (size_t)(b)*NV*NV + (size_t)(c)*NV + (d)]
#define H_T1(p,a)       h_t1[(p)*NV + (a)]
#define H_T2(p,q,a,b)   h_t2[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]

void EAEOMCCSDOperator::build_dressed_intermediates() {
    const int NO = nocc_;
    const int NV = nvir_;
    const size_t t1_sz   = (size_t)NO * NV;
    const size_t t2_sz   = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t oovv_sz = (size_t)NO * NO * NV * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t vvvv_sz = (size_t)NV * NV * NV * NV;

    // EA build_dressed sub-phase profiler (env GANSU_EA_BUILD_SUBPROF, default
    // off, math-inert). Mirrors STEOM's GANSU_STEOM_BUILD_PROF: each subprof()
    // call prints wall time since the previous checkpoint so the host-scatter
    // hotspots inside the 32 s EA build are localized before GPU-porting them.
    // A cudaDeviceSynchronize() is issued so GEMM-port phases time their device
    // work, not just the async launch.
    const char* _eap = std::getenv("GANSU_PROGRESS");   // progress default-on; GANSU_PROGRESS=0 to quiet
    const bool ea_subprof = (std::getenv("GANSU_EA_BUILD_SUBPROF") != nullptr)
                          || !_eap || _eap[0] != '0';
    auto _spclk = std::chrono::high_resolution_clock::now();
    auto subprof = [&](const char* nm) {
        if (!ea_subprof) return;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
        const auto now = std::chrono::high_resolution_clock::now();
        std::cout << "    [EA build-SUBPROF] " << nm << " = " << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double>(now - _spclk).count() << " s"
                  << std::defaultfloat << std::endl;
        _spclk = now;
    };

    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz);
    // Ship 12: in slab mode the bare vvvv tensor lives in per-device device
    // buffers (d_eri_vvvv_slabs_); the only consumer that needs h_vvvv (host)
    // is the canonical-skip OFF Wvvvv build + validate sample + CPU fallback.
    // Slab mode enforces canonical_skip_wvvvv_=true and uses the Term A slab
    // GEMM directly off device, so h_vvvv stays empty.  Saves NV⁴·8B host RAM
    // (Pentacene: 91 GB host alloc avoided).
    // h_vvvv (host NV⁴ = 4.7 GB at naphthalene) is read ONLY by: the canonical-skip
    // OFF Wvvvv build (line ~1807, gated by !canonical_skip_wvvvv_), the VALIDATE
    // self-check (line ~1952), and the CPU fallback. Under canonical-skip ON + GPU
    // + non-validate the wvvvo_w_t1 Term A reads d_eri_vvvv_ straight off the device
    // (line ~1927), so h_vvvv is never touched — skip its 4.7 GB host alloc + D2H
    // entirely (EA build_dressed "host alloc + D2H inputs" 4.6→~0.6 s at naphthalene).
    const bool need_h_vvvv = (eri_vvvv_nslab_ <= 1) &&
        (!canonical_skip_wvvvv_
         || std::getenv("GANSU_STEOM_BUILD_VALIDATE") != nullptr
         || !gpu::gpu_available());
    std::vector<real_t> h_vvvv;
    if (need_h_vvvv) h_vvvv.assign(vvvv_sz, 0.0);

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_,  ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (need_h_vvvv)
        cudaMemcpy(h_vvvv.data(), d_eri_vvvv_,  vvvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    subprof("host alloc + D2H inputs");
    // ============================================================
    //  cc_Fov[k,c] = fov + 2 ovov[k,c,l,d] t1[l,d] - ovov[k,d,l,c] t1[l,d]
    // ============================================================
    std::vector<real_t> h_Fov(t1_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int c = 0; c < NV; ++c) {
            real_t v = 0.0;
            for (int l = 0; l < NO; ++l)
                for (int d = 0; d < NV; ++d) {
                    v += 2.0 * H_OVOV(k,c,l,d) * H_T1(l,d);
                    v -=       H_OVOV(k,d,l,c) * H_T1(l,d);
                }
            h_Fov[k*NV + c] = v;
        }

    subprof("cc_Fov");
    // ============================================================
    //  cc_Foo / cc_Fvv (internal helpers for Loo / Lvv)
    // ============================================================
    // GPU GEMM port of the two cc_Foo / cc_Fvv host quad-loops (EA build_dressed
    // profiler hotspot = 6.8 s at naphthalene, cc_Fvv alone O(NO²·NV³)). Both
    // reduce to a single GEMM after folding τ = t2 + t1·t1:
    //   cc_Foo[k,i] = δ_ki·f_oo + Σ_{l,c,d} (2 ovov(k,c,l,d) − ovov(k,d,l,c))·τ(i,l,c,d)
    //              = A_foo[k,(lcd)] · τ_oo[i,(lcd)]ᵀ           (M=N=NO, K=NO·NV²)
    //   cc_Fvv[a,c] = δ_ac·f_vv − Σ_{k,l,d} (2 ovov(k,c,l,d) − ovov(k,d,l,c))·σ(a,k,l,d)
    //              = −σ_vv[a,(kld)] · B_vv[c,(kld)]ᵀ           (M=N=NV, K=NO²·NV)
    // Host repack (NO²·NV² each) + 1 GEMM + tiny D2H, then the f_oo/f_vv diagonal
    // is added on read.  Mirrors the validated ct_ovov / Wvvvv-term3 ports.
    std::vector<real_t> h_ccFoo(NO * NO, 0.0);
    std::vector<real_t> h_ccFvv(NV * NV, 0.0);
    bool ccF_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        {   // ---- cc_Foo: row-major C[k,i] = Σ_q A_foo[k,q]·τ_oo[i,q], q=(l,c,d) ----
            const int Mf = NO, Nf = NO, Kf = NO * NV * NV;
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)Mf*Kf*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)Kf*Nf*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)Mf*Nf*sizeof(real_t));
            { const size_t n=(size_t)NO*NO*NV*NV; const unsigned NB=(unsigned)((n+255)/256);
              ea_ccfoo_buildA<<<NB,256>>>(dA, d_eri_ovov_, NO, NV);
              ea_ccfoo_buildB<<<NB,256>>>(dB, d_t2_, d_t1_, NO, NV); }
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,Nf,Mf,Kf,&one,dB,Nf,dA,Kf,&zero,dC,Nf);
            std::vector<real_t> hC((size_t)Mf*Nf);
            cudaMemcpy(hC.data(),dC,(size_t)Mf*Nf*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
            #pragma omp parallel for
            for (int k = 0; k < NO; ++k)
                for (int i = 0; i < NO; ++i)
                    h_ccFoo[k*NO + i] = (k == i ? h_f_oo[k] : 0.0) + hC[(size_t)k*NO + i];
        }
        {   // ---- cc_Fvv: row-major C[a,c] = Σ_q σ_vv[a,q]·B_vv[c,q], q=(k,l,d) ----
            const int Mv = NV, Nv = NV, Kv = NO * NO * NV;
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)Mv*Kv*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)Kv*Nv*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)Mv*Nv*sizeof(real_t));
            { const size_t n=(size_t)NV*NO*NO*NV; const unsigned NB=(unsigned)((n+255)/256);
              ea_ccfvv_buildA<<<NB,256>>>(dA, d_t2_, d_t1_, NO, NV);
              ea_ccfvv_buildB<<<NB,256>>>(dB, d_eri_ovov_, NO, NV); }
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,Nv,Mv,Kv,&one,dB,Nv,dA,Kv,&zero,dC,Nv);
            std::vector<real_t> hC((size_t)Mv*Nv);
            cudaMemcpy(hC.data(),dC,(size_t)Mv*Nv*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
            #pragma omp parallel for
            for (int a = 0; a < NV; ++a)
                for (int c = 0; c < NV; ++c)
                    h_ccFvv[a*NV + c] = (a == c ? h_f_vv[a] : 0.0) - hC[(size_t)a*NV + c];
        }
        ccF_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1 = 0.0, d2 = 0.0;
            for (int k = 0; k < NO; k += (NO/2>0?NO/2:1))
                for (int i = 0; i < NO; ++i) {
                    real_t v = (k == i ? h_f_oo[k] : 0.0);
                    for (int l = 0; l < NO; ++l) for (int c = 0; c < NV; ++c) for (int d = 0; d < NV; ++d) {
                        real_t kcld = H_OVOV(k,c,l,d), kdlc = H_OVOV(k,d,l,c);
                        v += 2.0 * kcld * H_T2(i,l,c,d) - kdlc * H_T2(i,l,c,d);
                        v += (2.0 * kcld - kdlc) * H_T1(i,c) * H_T1(l,d);
                    }
                    d1 = std::max(d1, std::fabs(v - h_ccFoo[k*NO + i]));
                }
            for (int a = 0; a < NV; a += (NV/2>0?NV/2:1))
                for (int c = 0; c < NV; c += (NV/2>0?NV/2:1)) {
                    real_t v = (a == c ? h_f_vv[a] : 0.0);
                    for (int k = 0; k < NO; ++k) for (int l = 0; l < NO; ++l) for (int d = 0; d < NV; ++d) {
                        real_t kcld = H_OVOV(k,c,l,d), kdlc = H_OVOV(k,d,l,c);
                        v -= 2.0 * kcld * H_T2(k,l,a,d) - kdlc * H_T2(k,l,a,d);
                        v -= (2.0 * kcld - kdlc) * H_T1(k,a) * H_T1(l,d);
                    }
                    d2 = std::max(d2, std::fabs(v - h_ccFvv[a*NV + c]));
                }
            std::cout << "[EA build self-check] cc_Foo GEMM vs host: max|Δ| = " << std::scientific << d1
                      << ", cc_Fvv max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    if (!ccF_gpu) {
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int i = 0; i < NO; ++i) {
                real_t v = (k == i ? h_f_oo[k] : 0.0);
                for (int l = 0; l < NO; ++l)
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v += 2.0 * kcld * H_T2(i,l,c,d) - kdlc * H_T2(i,l,c,d);
                            v += (2.0 * kcld - kdlc) * H_T1(i,c) * H_T1(l,d);
                        }
                h_ccFoo[k*NO + i] = v;
            }

        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c) {
                real_t v = (a == c ? h_f_vv[a] : 0.0);
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v -= 2.0 * kcld * H_T2(k,l,a,d) - kdlc * H_T2(k,l,a,d);
                            v -= (2.0 * kcld - kdlc) * H_T1(k,a) * H_T1(l,d);
                        }
                h_ccFvv[a*NV + c] = v;
            }
    }

    subprof("cc_Foo + cc_Fvv");
    // ============================================================
    //  Loo[k,i] = cc_Foo + Fov·t1 + 2 ooov·t1 - ooov·t1 (PySCF Loo)
    //  Lvv[a,c] = cc_Fvv - Fov·t1 + 2 ovvv·t1 - ovvv·t1 (PySCF Lvv)
    // ============================================================
    std::vector<real_t> h_Loo(NO * NO, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int i = 0; i < NO; ++i) {
            real_t v = h_ccFoo[k*NO + i];
            for (int c = 0; c < NV; ++c)
                v += h_Fov[k*NV + c] * H_T1(i,c);
            for (int l = 0; l < NO; ++l)
                for (int c = 0; c < NV; ++c) {
                    v += 2.0 * H_OOOV(k,i,l,c) * H_T1(l,c);
                    v -=       H_OOOV(l,i,k,c) * H_T1(l,c);
                }
            h_Loo[k*NO + i] = v;
        }

    std::vector<real_t> h_Lvv(NV * NV, 0.0);
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int c = 0; c < NV; ++c) {
            real_t v = h_ccFvv[a*NV + c];
            for (int k = 0; k < NO; ++k)
                v -= h_Fov[k*NV + c] * H_T1(k,a);
            for (int k = 0; k < NO; ++k)
                for (int d = 0; d < NV; ++d) {
                    v += 2.0 * H_OVVV(k,d,a,c) * H_T1(k,d);
                    v -=       H_OVVV(k,c,a,d) * H_T1(k,d);
                }
            h_Lvv[a*NV + c] = v;
        }

    subprof("Loo + Lvv");
    // ============================================================
    //  W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d] t2[i,l,c,b]
    //  W2ovov[k,b,i,d] = -Σ_l Wooov[k,l,i,d] t1[l,b]
    //                  +  Σ_c ovvv[k,c,b,d] t1[i,c]
    //  Wovov = W1ovov + W2ovov                    (PySCF Wovov)
    // Where Wooov[k,l,i,d] = ooov[k,i,l,d] + Σ_c t1[i,c] ovov[k,c,l,d]
    // ============================================================
    // Stage 2: GPU port of the Σ_c contraction in Wooov (O(NO³·NV²) host loop).
    //   ct_wooov[k,i,(l,d)] = Σ_c t1[i,c]·ovov[k,c,l,d]  via a per-k strided-
    //   batched GEMM reading d_eri_ovov_ straight off the device (each k slab is
    //   a contiguous [c,(l,d)] block, no repack) and d_t1_ (shared across k).
    //   Then h_Wooov[k,l,i,d] = ooov[k,i,l,d] + ct[k,i,l,d] (host, NO³·NV reorder).
    std::vector<real_t> h_Wooov(ooov_sz, 0.0);
    bool wooov_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t ct_sz = (size_t)NO * NO * NO * NV;   // [k,i,l,d]
        real_t* dC = nullptr;
        tracked_cudaMalloc(&dC, ct_sz * sizeof(real_t));
        const real_t one = 1.0, zero = 0.0;
        cublasDgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            NO*NV, NO, NV,
            &one,
            d_eri_ovov_, NO*NV, (long long)NV*NO*NV,
            d_t1_,       NV,    0LL,
            &zero,
            dC,          NO*NV, (long long)NO*NO*NV,
            NO);
        std::vector<real_t> ct(ct_sz);
        cudaMemcpy(ct.data(), dC, ct_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dC);
        wooov_gpu = true;
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int l = 0; l < NO; ++l)
                for (int i = 0; i < NO; ++i)
                    for (int d = 0; d < NV; ++d)
                        h_Wooov[(((size_t)k * NO + l) * NO + i) * NV + d] =
                            H_OOOV(k,i,l,d) + ct[(((size_t)k*NO+i)*NO+l)*NV+d];
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1))
                for (int l=0;l<NO;l+=(NO/2>0?NO/2:1))
                    for (int i=0;i<NO;++i)
                        for (int d=0;d<NV;d+=(NV/2>0?NV/2:1)) {
                            real_t v = H_OOOV(k,i,l,d);
                            for (int c=0;c<NV;++c) v += H_T1(i,c)*H_OVOV(k,c,l,d);
                            const real_t got = h_Wooov[(((size_t)k*NO+l)*NO+i)*NV+d];
                            dmax = std::max(dmax, std::fabs(v - got));
                        }
            std::cout << "[EA build self-check] Wooov (t1·ovov) batched GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    if (!wooov_gpu) {
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int l = 0; l < NO; ++l)
                for (int i = 0; i < NO; ++i)
                    for (int d = 0; d < NV; ++d) {
                        real_t v = H_OOOV(k,i,l,d);
                        for (int c = 0; c < NV; ++c)
                            v += H_T1(i,c) * H_OVOV(k,c,l,d);
                        h_Wooov[(((size_t)k * NO + l) * NO + i) * NV + d] = v;
                    }
    }
    #define H_WOOOV(k,l,i,d) h_Wooov[(((size_t)(k) * NO + (l)) * NO + (i)) * NV + (d)]

    // GPU GEMM port of the two O(NO³·NV³) contraction hotspots (mirrors the
    // validated IP/STEOM build_dressed ports). Each is computed ONCE on the GPU
    // and reused by the W1ovov / W1ovvo standalone loops below.
    //   ct_ovov[(k,d),(i,b)] = Σ_{c,l} ovov[k,c,l,d]·t2[i,l,c,b]
    //   ct_ovvo[(k,c),(j,b)] = Σ_{l,d}[(2·ovov[k,c,l,d]−ovov[k,d,l,c])·t2[j,l,b,d]
    //                                  − ovov[k,c,l,d]·t2[l,j,b,d]]
    const int EA_MO_kd = NO*NV, EA_NO_ib = NO*NV, EA_KO_cl = NV*NO;   // ct_ovov
    const int EA_M_kc = NO*NV, EA_N_jb = NO*NV, EA_K_ld = NO*NV;      // ct_ovvo
    std::vector<real_t> ct_ovov, ct_ovvo;
    bool ea_ct_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, negone = -1.0, zero = 0.0;
        {   // ct_ovov: C[(k,d),(i,b)] = A[(k,d),(c,l)]·B[(c,l),(i,b)] — inputs device-built
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)EA_MO_kd*EA_KO_cl*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)EA_KO_cl*EA_NO_ib*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)EA_MO_kd*EA_NO_ib*sizeof(real_t));
            const size_t nel=(size_t)NO*NV*NV*NO; const unsigned NB=(unsigned)((nel+255)/256);
            ea_ctovov_buildA<<<NB,256>>>(dA, d_eri_ovov_, NO, NV);
            ea_ctovov_buildB<<<NB,256>>>(dB, d_t2_, NO, NV);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,EA_NO_ib,EA_MO_kd,EA_KO_cl,&one,
                        dB,EA_NO_ib,dA,EA_KO_cl,&zero,dC,EA_NO_ib);
            ct_ovov.assign((size_t)EA_MO_kd*EA_NO_ib,0.0);
            cudaMemcpy(ct_ovov.data(),dC,(size_t)EA_MO_kd*EA_NO_ib*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        {   // ct_ovvo: C[(k,c),(j,b)] = A1·B1 − A2·B2 — inputs device-built
            real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA1,(size_t)EA_M_kc*EA_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB1,(size_t)EA_K_ld*EA_N_jb*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)EA_M_kc*EA_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)EA_K_ld*EA_N_jb*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)EA_M_kc*EA_N_jb*sizeof(real_t));
            const size_t nel=(size_t)NO*NV*NO*NV; const unsigned NB=(unsigned)((nel+255)/256);
            ea_ctovvo_buildA12<<<NB,256>>>(dA1, dA2, d_eri_ovov_, NO, NV);
            ea_ctovvo_buildB12<<<NB,256>>>(dB1, dB2, d_t2_, NO, NV);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,EA_N_jb,EA_M_kc,EA_K_ld,&one,
                        dB1,EA_N_jb,dA1,EA_K_ld,&zero,dC,EA_N_jb);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,EA_N_jb,EA_M_kc,EA_K_ld,&negone,
                        dB2,EA_N_jb,dA2,EA_K_ld,&one,dC,EA_N_jb);
            ct_ovvo.assign((size_t)EA_M_kc*EA_N_jb,0.0);
            cudaMemcpy(ct_ovvo.data(),dC,(size_t)EA_M_kc*EA_N_jb*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA1);tracked_cudaFree(dB1);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        }
        ea_ct_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1=0.0,d2=0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int d=0;d<NV;d+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int b=0;b<NV;++b) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int l=0;l<NO;++l) v+=H_OVOV(k,c,l,d)*H_T2(i,l,c,b);
                    d1=std::max(d1,std::fabs(v-ct_ovov[(size_t)(k*NV+d)*EA_NO_ib+(i*NV+b)]));
                }
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int c=0;c<NV;++c) for (int j=0;j<NO;++j) {
                    real_t v=0.0; for (int l=0;l<NO;++l) for (int dd=0;dd<NV;++dd){
                        real_t kcld=H_OVOV(k,c,l,dd);
                        v+=2.0*kcld*H_T2(j,l,b,dd)-kcld*H_T2(l,j,b,dd)-H_OVOV(k,dd,l,c)*H_T2(j,l,b,dd);}
                    d2=std::max(d2,std::fabs(v-ct_ovvo[(size_t)(k*NV+c)*EA_N_jb+(j*NV+b)]));
                }
            std::cout << "[EA-EOM build self-check] ct_ovov max|Δ| = " << std::scientific << d1
                      << ", ct_ovvo max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    subprof("Wooov + ct_ovov/ct_ovvo GEMM");
    std::vector<real_t> h_W1ovov(ovov_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);
                    if (ea_ct_gpu) {
                        v -= ct_ovov[(size_t)(k*NV+d)*EA_NO_ib + (i*NV+b)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                    }
                    h_W1ovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }
    #define H_W1OVOV(k,b,i,d) h_W1ovov[(((size_t)(k) * NV + (b)) * NO + (i)) * NV + (d)]

    // Stage 4: GPU port of the Wovov W2 +Σ_c ovvv·t1 contraction (the dominant
    // O(NO²·NV³) host loop). ct_wovov[k,i,(b,d)] = Σ_c t1[i,c]·ovvv[k,c,b,d] via
    // per-k strided-batched GEMM (B = d_eri_ovvv_ slab contiguous [c,(b,d)]).
    std::vector<real_t> ct_wovov;
    bool wovov_w2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t cw_sz = (size_t)NO*NO*NV*NV;   // [k,i,(b,d)]
        real_t* dC=nullptr; tracked_cudaMalloc(&dC, cw_sz*sizeof(real_t));
        const real_t one=1.0, zero=0.0;
        cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            NV*NV, NO, NV, &one,
            d_eri_ovvv_, NV*NV, (long long)NV*NV*NV,
            d_t1_,       NV,    0LL,
            &zero, dC, NV*NV, (long long)NO*NV*NV, NO);
        ct_wovov.assign(cw_sz, 0.0);
        cudaMemcpy(ct_wovov.data(), dC, cw_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dC);
        wovov_w2_gpu = true;
    }
#endif
    std::vector<real_t> h_Wovov(ovov_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_W1OVOV(k,b,i,d);                       // W1
                    for (int l = 0; l < NO; ++l)
                        v -= H_WOOOV(k,l,i,d) * H_T1(l,b);               // W2: -Wooov·t1
                    if (wovov_w2_gpu) {
                        v += ct_wovov[(((size_t)k*NO+i)*NV+b)*NV+d];     // W2: +ovvv·t1 (GEMM)
                    } else {
                        for (int c = 0; c < NV; ++c)
                            v += H_OVVV(k,c,b,d) * H_T1(i,c);
                    }
                    h_Wovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }
    if (wovov_w2_gpu && std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
        real_t dmax = 0.0;
        for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int i=0;i<NO;i+=(NO/2>0?NO/2:1))
            for (int b=0;b<NV;b+=(NV/2>0?NV/2:1)) for (int d=0;d<NV;d+=(NV/2>0?NV/2:1)) {
                real_t t=0.0; for (int c=0;c<NV;++c) t += H_OVVV(k,c,b,d)*H_T1(i,c);
                dmax = std::max(dmax, std::fabs(t - ct_wovov[(((size_t)k*NO+i)*NV+b)*NV+d]));
            }
        std::cout << "[EA build self-check] Wovov W2 (ovvv·t1) batched GEMM vs host: max|Δ| = "
                  << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
    }

    subprof("W1ovov + Wovov");
    // ============================================================
    //  W1ovvo[k,b,c,j] = ovvo[k,c,b,j]
    //                  + 2 Σ_{l,d} ovov[k,c,l,d] t2[j,l,b,d]
    //                  -   Σ_{l,d} ovov[k,c,l,d] t2[l,j,b,d]
    //                  -   Σ_{l,d} ovov[k,d,l,c] t2[j,l,b,d]
    //  W2ovvo[k,b,c,j] = -Σ_l t1[l,b] Wooov[l,k,j,c]
    //                  + Σ_d ovvv[k,c,b,d] t1[j,d]
    //  Wovvo = W1ovvo + W2ovvo                      (PySCF Wovvo)
    // ============================================================
    std::vector<real_t> h_W1ovvo(ovvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVO(k,c,b,j);
                    if (ea_ct_gpu) {
                        v += ct_ovvo[(size_t)(k*NV+c)*EA_N_jb + (j*NV+b)];
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += 2.0 * kcld * H_T2(j,l,b,d);
                                v -=       kcld * H_T2(l,j,b,d);
                                v -= H_OVOV(k,d,l,c) * H_T2(j,l,b,d);
                            }
                    }
                    h_W1ovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }
    #define H_W1OVVO(k,b,c,j) h_W1ovvo[(((size_t)(k) * NV + (b)) * NV + (c)) * NO + (j)]

    // Stage 4: GPU port of the Wovvo W2 +Σ_d ovvv·t1 contraction (dominant
    // O(NO·NV³·NO) host loop). ct_wovvo[(k,c,b),j] = Σ_d ovvv[k,c,b,d]·t1[j,d]
    // = ovvv[(k,c,b),d]·t1T[d,j] (single GEMM, d is the trailing ovvv axis).
    std::vector<real_t> ct_wovvo;
    bool wovvo_w2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t cw_sz = (size_t)NO*NV*NV*NO;   // [(k,c,b),j]
        std::vector<real_t> hT((size_t)NV*NO);
        #pragma omp parallel for
        for (int d=0;d<NV;++d) for (int j=0;j<NO;++j) hT[(size_t)d*NO+j] = h_t1[j*NV+d];
        real_t *dT=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dT,(size_t)NV*NO*sizeof(real_t));
        tracked_cudaMalloc(&dC, cw_sz*sizeof(real_t));
        cudaMemcpy(dT,hT.data(),(size_t)NV*NO*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0, zero=0.0;
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            NO, NO*NV*NV, NV, &one,
            dT, NO, d_eri_ovvv_, NV, &zero, dC, NO);
        ct_wovvo.assign(cw_sz, 0.0);
        cudaMemcpy(ct_wovvo.data(), dC, cw_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dT); tracked_cudaFree(dC);
        wovvo_w2_gpu = true;
    }
#endif
    std::vector<real_t> h_Wovvo(ovvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_W1OVVO(k,b,c,j);
                    for (int l = 0; l < NO; ++l)
                        v -= H_T1(l,b) * H_WOOOV(l,k,j,c);
                    if (wovvo_w2_gpu) {
                        v += ct_wovvo[(((size_t)k*NV+c)*NV+b)*NO+j];     // W2: +ovvv·t1 (GEMM)
                    } else {
                        for (int d = 0; d < NV; ++d)
                            v += H_OVVV(k,c,b,d) * H_T1(j,d);
                    }
                    h_Wovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }
    if (wovvo_w2_gpu && std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
        real_t dmax = 0.0;
        for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
            for (int b=0;b<NV;b+=(NV/2>0?NV/2:1)) for (int j=0;j<NO;++j) {
                real_t t=0.0; for (int d=0;d<NV;++d) t += H_OVVV(k,c,b,d)*H_T1(j,d);
                dmax = std::max(dmax, std::fabs(t - ct_wovvo[(((size_t)k*NV+c)*NV+b)*NO+j]));
            }
        std::cout << "[EA build self-check] Wovvo W2 (ovvv·t1) GEMM vs host: max|Δ| = "
                  << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
    }

    subprof("W1ovvo + Wovvo");
    // ============================================================
    //  Wvovv[a,l,c,d] = -Σ_k t1[k,a] ovov[k,c,l,d] + ovvv[l,d,a,c]
    //  (PySCF: 'ka,kcld->alcd' with -t1, then + ovvv.transpose(2,0,3,1).)
    //  layout: [a, l, c, d]  index = ((a*NO + l)*NV + c)*NV + d
    // ============================================================
    // Broader-refactor stage 1: build d_Wvovv_ entirely on the device.
    // ct[a,(c,l,d)] = Σ_k t1[k,a]·ovov[k,c,l,d] via one GEMM (B = d_eri_ovov_
    // resident in natural [k,c,l,d] layout, ldb = NV·NO·NV; A = t1 transposed),
    // then ea_wvovv_assemble_kernel writes d_Wvovv_[a,l,c,d] = ovvv[l,d,a,c] − ct
    // straight on device (reads d_eri_ovvv_ for the bare term).  No host assemble,
    // no H2D upload, and Wvovv no longer reads h_ovvv/h_ovov.
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    std::vector<real_t> h_Wvovv;          // CPU fallback only; device path builds d_Wvovv_ directly
    bool wvovv_gpu = false, wvovv_on_device = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const int Mv = NV, Nv = NV*NO*NV, Kv = NO;
        std::vector<real_t> hA((size_t)Mv*Kv);
        #pragma omp parallel for
        for (int a = 0; a < NV; ++a)
            for (int k = 0; k < NO; ++k) hA[(size_t)a*Kv + k] = h_t1[k*NV + a];
        real_t *dA=nullptr, *dC=nullptr;
        tracked_cudaMalloc(&dA, (size_t)Mv*Kv*sizeof(real_t));
        tracked_cudaMalloc(&dC, (size_t)Mv*Nv*sizeof(real_t));
        cudaMemcpy(dA, hA.data(), (size_t)Mv*Kv*sizeof(real_t), cudaMemcpyHostToDevice);
        const real_t one=1.0, zero=0.0;
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, Nv, Mv, Kv, &one,
                    d_eri_ovov_, Nv, dA, Kv, &zero, dC, Nv);
        tracked_cudaFree(dA);
        wvovv_gpu = true;
        // Build d_Wvovv_ on device (member, persists for σ1).  No host assemble.
        tracked_cudaMalloc(&d_Wvovv_, wvovv_sz * sizeof(real_t));
        {
            const int thr = 256;
            const int blk = (int)std::min<size_t>((wvovv_sz + thr - 1) / thr, 65535);
            ea_wvovv_assemble_kernel<<<blk, thr>>>(d_eri_ovvv_, dC, d_Wvovv_, NO, NV);
        }
        wvovv_on_device = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            std::vector<real_t> h_chk(wvovv_sz);
            cudaMemcpy(h_chk.data(), d_Wvovv_, wvovv_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
            real_t dmax = 0.0;
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1))
                for (int l=0;l<NO;l+=(NO/2>0?NO/2:1))
                    for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
                        for (int d=0;d<NV;d+=(NV/2>0?NV/2:1)) {
                            real_t v = H_OVVV(l,d,a,c);
                            for (int k=0;k<NO;++k) v -= H_T1(k,a)*H_OVOV(k,c,l,d);
                            const real_t got = h_chk[(((size_t)a*NO+l)*NV+c)*NV+d];
                            dmax = std::max(dmax, std::fabs(v - got));
                        }
            std::cout << "[EA build self-check] Wvovv device build vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
        tracked_cudaFree(dC);
    }
#endif
    if (!wvovv_gpu) {
        h_Wvovv.assign(wvovv_sz, 0.0);
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int l = 0; l < NO; ++l)
                for (int c = 0; c < NV; ++c)
                    for (int d = 0; d < NV; ++d) {
                        real_t v = H_OVVV(l,d,a,c);  // bare ovvv chemist (ld|ac)
                        for (int k = 0; k < NO; ++k)
                            v -= H_T1(k,a) * H_OVOV(k,c,l,d);
                        h_Wvovv[(((size_t)a * NO + l) * NV + c) * NV + d] = v;
                    }
    }

    subprof("Wvovv");
    // ============================================================
    //  Wvvvv[a,b,c,d] = + Σ_{k,l} ovov[k,c,l,d] t2[k,l,a,b]
    //                 + Σ_{k,l} ovov[k,c,l,d] t1[k,a] t1[l,b]
    //                 + (ac|bd)                       (chemist→physicist of vvvv)
    //                 - Σ_l ovvv[l,d,a,c] t1[l,b]
    //                 - Σ_k ovvv[k,c,b,d] t1[k,a]
    //  layout: [a, b, c, d]  index = ((a*NV + b)*NV + c)*NV + d
    // ============================================================
    // Wvvvv dominant term3 (Σ_kl ovov[k,c,l,d]·τ[k,l,a,b], τ=t2+t1t1) via a single GEMM —
    // the EA-EOM build hotspot (mirrors the STEOM build). result[(ab),(cd)] = Σ_(kl)
    // τ[(kl),(ab)]·ovov2[(kl),(cd)] with ovov2 = ovov repacked to [k,l,c,d]; cublasDgemm(N,T)
    // → C col-major [cd×ab] = result row-major [ab][cd] = Wvvvv layout. Host fallback keeps
    // the k,l sum. Self-check (GANSU_STEOM_BUILD_VALIDATE) below.
    const int NV2 = NV * NV, NO2 = NO * NO;
    // Stage 3 (Wvvvo device assembly, canonical-skip path): keep each term's
    // device buffer resident so the final Wvvvo is assembled by a kernel — no
    // D2H of the NV³·NO terms, no host assemble, no H2D upload.  Host arrays and
    // per-term self-checks are still produced under GANSU_STEOM_BUILD_VALIDATE.
    real_t *d_wvt1_keep=nullptr, *d_big_keep=nullptr, *d_t5_keep=nullptr,
           *d_ct1_keep=nullptr, *d_ct2_keep=nullptr, *d_ct5_keep=nullptr;
    bool wvvvo_dev_asm = false;
    const bool wvvvo_keep_validate = (std::getenv("GANSU_STEOM_BUILD_VALIDATE") != nullptr);
#ifndef GANSU_CPU_ONLY
    wvvvo_dev_asm = gpu::gpu_available() && canonical_skip_wvvvv_ && eri_vvvv_nslab_ <= 1;
#endif
    std::vector<real_t> wvvvv_t3;
    bool wvvvv_t3_gpu = false;
    std::vector<real_t> h_Wvvvv;
    if (!canonical_skip_wvvvv_) {
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
            std::vector<real_t> h_tau(t2_sz), h_ovov2(ovov_sz);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < NO; ++k)
                for (int l = 0; l < NO; ++l)
                    for (int a = 0; a < NV; ++a)
                        for (int b = 0; b < NV; ++b)
                            h_tau[(((size_t)k*NO+l)*NV+a)*NV+b] = H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < NO; ++k)
                for (int c = 0; c < NV; ++c)
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d)
                            h_ovov2[(((size_t)k*NO+l)*NV+c)*NV+d] = H_OVOV(k,c,l,d);
            real_t *d_tau = nullptr, *d_ovov2 = nullptr, *d_t3 = nullptr;
            tracked_cudaMalloc(&d_tau,   t2_sz   * sizeof(real_t));
            tracked_cudaMalloc(&d_ovov2, ovov_sz * sizeof(real_t));
            tracked_cudaMalloc(&d_t3,    (size_t)NV2 * NV2 * sizeof(real_t));
            cudaMemcpy(d_tau,   h_tau.data(),   t2_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ovov2, h_ovov2.data(), ovov_sz * sizeof(real_t), cudaMemcpyHostToDevice);
            const real_t one = 1.0, zero = 0.0;
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, NV2, NV2, NO2, &one,
                        d_ovov2, NV2, d_tau, NV2, &zero, d_t3, NV2);
            wvvvv_t3.assign((size_t)NV2 * NV2, 0.0);
            cudaMemcpy(wvvvv_t3.data(), d_t3, (size_t)NV2 * NV2 * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(d_tau); tracked_cudaFree(d_ovov2); tracked_cudaFree(d_t3);
            wvvvv_t3_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                const int asamp[2] = {0, NV - 1};
                for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                    for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                        for (int c = 0; c < NV; ++c) for (int d = 0; d < NV; ++d) {
                            real_t v = 0.0;
                            for (int k = 0; k < NO; ++k) for (int l = 0; l < NO; ++l)
                                v += H_OVOV(k,c,l,d) * (H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b));
                            dmax = std::max(dmax, std::fabs(v - wvvvv_t3[((size_t)(a*NV+b))*NV2 + (c*NV+d)]));
                        }
                }
                std::cout << "[EA-EOM build self-check] Wvvvv term3 GEMM vs host: max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)"
                          << std::defaultfloat << std::endl;
            }
        }
#endif

        h_Wvvvv.assign(vvvv_sz, 0.0);
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int b = 0; b < NV; ++b)
                for (int c = 0; c < NV; ++c)
                    for (int d = 0; d < NV; ++d) {
                        real_t v = H_VVVV(a,c,b,d);  // (ac|bd) chemist → Wabcd physicist
                        for (int k = 0; k < NO; ++k)
                            v -= H_OVVV(k,c,b,d) * H_T1(k,a);
                        for (int l = 0; l < NO; ++l)
                            v -= H_OVVV(l,d,a,c) * H_T1(l,b);
                        if (wvvvv_t3_gpu) {
                            v += wvvvv_t3[((size_t)(a * NV + b)) * NV2 + (c * NV + d)];
                        } else {
                            for (int k = 0; k < NO; ++k)
                                for (int l = 0; l < NO; ++l) {
                                    real_t kcld = H_OVOV(k,c,l,d);
                                    v += kcld * (H_T2(k,l,a,b) + H_T1(k,a) * H_T1(l,b));
                                }
                        }
                        h_Wvvvv[(((size_t)a * NV + b) * NV + c) * NV + d] = v;
                    }
    }  // close canonical_skip guard
    #define H_WVVVV(a,b,c,d) h_Wvvvv[(((size_t)(a) * NV + (b)) * NV + (c)) * NV + (d)]

    // P5 canonical-skip: when h_Wvvvv is not materialized, the only consumer
    // (Σ_d Wvvvv[a,b,c,d]·t1[j,d] in the Wvvvo build below) is recomputed via
    // 4 fused contractions without ever forming a nvir⁴ intermediate. Each term
    // uses an O(nocc³·nvir) or O(nocc·nvir²) scratch and runs at O(nvir⁵·nocc)
    // or less — same complexity class as the original d-loop, no extra work.
    std::vector<real_t> wvvvo_w_t1;
    if (canonical_skip_wvvvv_) {
        const size_t wvvvo_sz_local = (size_t)NV * NV * NV * NO;
        wvvvo_w_t1.assign(wvvvo_sz_local, 0.0);

        // ship 4: device-resident accumulator.  Each Term A/B/C/D GEMM result
        // is scattered into d_wt1 on the device (no D2H, no host scatter); one
        // final D2H copies d_wt1 → wvvvo_w_t1.  GPU path only; CPU keeps the
        // per-term host fallbacks writing wvvvo_w_t1 directly.
        real_t* d_wt1 = nullptr;
        bool wt1_dev = false;
        const int wt1_thr = 256;
        const int wt1_blk = (int)std::min<size_t>((wvvvo_sz_local + wt1_thr - 1) / wt1_thr, 65535);
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            tracked_cudaMalloc(&d_wt1, wvvvo_sz_local * sizeof(real_t));
            cudaMemset(d_wt1, 0, wvvvo_sz_local * sizeof(real_t));
            wt1_dev = true;
        }
#endif

        // Term A: + Σ_d (ac|bd)·t1[j,d] = Σ_d H_VVVV(a,c,b,d)·t1[j,d]
        // GPU: 1 cuBLAS GEMM uses d_eri_vvvv_ already on device (no upload).
        //   intermediate[a,b,c,j] = Σ_d h_vvvv[a,b,c,d]·t1[j,d] (natural layout)
        //   then wvvvo_w_t1[a,b,c,j] += intermediate[a,c,b,j] (b↔c swap on host).
        // STEOM mirror.  anthracene -25 to -30 s contribution to EA build.
        {
            bool termA_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                const size_t inter_sz = (size_t)NV * NV * NV * NO;
                real_t* d_interA = nullptr;
                tracked_cudaMalloc(&d_interA, inter_sz * sizeof(real_t));
                std::vector<real_t> h_inter;   // assembled for slab path or validate D2H
                if (ri_vvvv_term_a_) {
                    // RI Term A: (ab|cd) = Σ_P B_vv[P,ab]·B_vv[P,cd]  ⇒
                    //   d_interA[a,b,c,j] = Σ_P B_vv[P,ab]·(Σ_d B_vv[P,cd]·t1[j,d]).
                    // Two GEMMs over the contiguous [naux×nvir²] B_vv block; the
                    // nvir⁴ (ab|cd) tensor is never formed. d_interA layout is
                    // identical to the dense GEMM below ⇒ scatter is unchanged.
                    cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                    const int naux = eri_block_src_->get_num_auxiliary_basis();
                    const int voff = NO + frozen_off_;
                    real_t *d_Bvv = nullptr, *d_Y = nullptr, *d_t1_dev = nullptr;
                    tracked_cudaMalloc(&d_Bvv,    (size_t)naux * NV * NV * sizeof(real_t));
                    tracked_cudaMalloc(&d_Y,      (size_t)naux * NV * NO * sizeof(real_t));
                    tracked_cudaMalloc(&d_t1_dev, (size_t)NO   * NV      * sizeof(real_t));
                    cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t),
                               cudaMemcpyHostToDevice);
                    { const int thr = 256;
                      const size_t tot = (size_t)naux * NV * NV;
                      int blk = (int)((tot + thr - 1) / thr); if (blk > 65535) blk = 65535;
                      ea_gather_bvv_kernel<<<blk, thr>>>(d_B_mo_blocks_, d_Bvv,
                                                         naux, nmo_full_, voff, NV); }
                    const real_t one = 1.0, zero = 0.0;
                    // GEMM1: Y[(P,c),j] = Σ_d B_vv[(P,c),d]·t1[j,d]  (mirrors dense Term A shape)
                    cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                NO, naux * NV, NV, &one,
                                d_t1_dev, NV, d_Bvv, NV, &zero, d_Y, NO);
                    // GEMM2: d_interA[(a,b),(c,j)] = Σ_P B_vv[P,(a,b)]·Y[P,(c,j)]
                    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                NV * NO, NV * NV, naux, &one,
                                d_Y, NV * NO, d_Bvv, NV * NV, &zero, d_interA, NV * NO);
                    tracked_cudaFree(d_Bvv); tracked_cudaFree(d_Y); tracked_cudaFree(d_t1_dev);
                    termA_gpu = true;
                } else if (eri_vvvv_nslab_ > 1) {
                    h_inter.assign(inter_sz, 0.0);
                    // Ship 12: per-device slab GEMM.  Each slab's GEMM output
                    // [a_local, b, c, j] (col k = a_local*NV² + b*NV + c) lines
                    // up directly with the global [a, b, c, j] layout, so each
                    // D2H writes contiguously into h_inter at byte offset
                    // a_starts_[d_dev] * NV²·NO·8.  The host swap loop below
                    // is unchanged (b↔c on read, see line ~1345).
                    const real_t one = 1.0, zero = 0.0;
                    const int N = eri_vvvv_nslab_;
                    std::vector<cublasHandle_t> hdl(N, nullptr);
                    std::vector<real_t*>        d_t1_per(N, nullptr), d_inter_per(N, nullptr);
                    int saved = 0; cudaGetDevice(&saved);
                    for (int d_dev = 0; d_dev < N; ++d_dev) {
                        const int an = a_ends_[d_dev] - a_starts_[d_dev];
                        if (an <= 0) continue;
                        cudaSetDevice(d_dev);
                        cublasCreate(&hdl[d_dev]);
                        cudaMalloc(&d_t1_per[d_dev], (size_t)NO * NV * sizeof(real_t));
                        cudaMemcpyAsync(d_t1_per[d_dev], h_t1.data(),
                                        (size_t)NO * NV * sizeof(real_t),
                                        cudaMemcpyHostToDevice, 0);
                        const size_t slab_inter_sz = (size_t)an * NV * NV * NO;
                        cudaMalloc(&d_inter_per[d_dev], slab_inter_sz * sizeof(real_t));
                        cublasDgemm(hdl[d_dev], CUBLAS_OP_T, CUBLAS_OP_N,
                                    NO, an * NV * NV, NV, &one,
                                    d_t1_per[d_dev],          NV,
                                    d_eri_vvvv_slabs_[d_dev], NV,
                                    &zero,
                                    d_inter_per[d_dev],       NO);
                        // D2H into the slab's contiguous portion of h_inter.
                        cudaMemcpyAsync(h_inter.data()
                                            + (size_t)a_starts_[d_dev] * NV * NV * NO,
                                        d_inter_per[d_dev],
                                        slab_inter_sz * sizeof(real_t),
                                        cudaMemcpyDeviceToHost, 0);
                    }
                    // Sync + free per device.
                    for (int d_dev = 0; d_dev < N; ++d_dev) {
                        const int an = a_ends_[d_dev] - a_starts_[d_dev];
                        if (an <= 0) continue;
                        cudaSetDevice(d_dev);
                        cudaDeviceSynchronize();
                        cudaFree(d_t1_per[d_dev]);
                        cudaFree(d_inter_per[d_dev]);
                        cublasDestroy(hdl[d_dev]);
                    }
                    cudaSetDevice(saved);
                    cudaMemcpy(d_interA, h_inter.data(), inter_sz * sizeof(real_t), cudaMemcpyHostToDevice);
                    termA_gpu = true;
                } else {
                    cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                    real_t *d_t1_dev = nullptr;
                    tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                    cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                    const real_t one = 1.0, zero = 0.0;
                    cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                NO, NV*NV*NV, NV, &one,
                                d_t1_dev,    NV,
                                d_eri_vvvv_, NV,
                                &zero,
                                d_interA,    NO);
                    tracked_cudaFree(d_t1_dev);
                    termA_gpu = true;
                }
                // Ship 12: self-check reads H_VVVV (= h_vvvv), unavailable in
                // slab mode → silently skip. Single-GPU validate path keeps
                // its existing GANSU_STEOM_BUILD_VALIDATE gate.
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE") && eri_vvvv_nslab_ <= 1) {
                    if (h_inter.empty()) {
                        h_inter.assign(inter_sz, 0.0);
                        cudaMemcpy(h_inter.data(), d_interA, inter_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                    }
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int d = 0; d < NV; ++d)
                                        v += H_VVVV(a,c,b,d) * H_T1(j,d);
                                    const real_t got = h_inter[(((size_t)a*NV+c)*NV+b)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[EA build self-check] Term A GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // ship 4: device scatter wvvvo_w_t1[a,b,c,j] += d_interA[a,c,b,j] (b↔c)
                ea_wvvvo_wt1_scatter_kernel<<<wt1_blk, wt1_thr>>>(d_interA, d_wt1, 1.0, 1, NO, NV);
                tracked_cudaFree(d_interA);
            }
#endif
            if (!termA_gpu) {
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_VVVV(a,c,b,d) * H_T1(j,d);
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] += s;
                            }
            }
        }

        // Term B: - Σ_dk ovvv[k,c,b,d]·t1[k,a]·t1[j,d]
        //   Stage 1: inter1[(k,c,b),j] = Σ_d ovvv[(k,c,b),d]·t1[j,d]  (natural ovvv layout)
        //   Stage 2: result[a,(c,b,j)] = Σ_k t1[k,a]·inter1[k,(c,b,j)]
        //   wvvvo_w_t1[a,b,c,j] -= result[a,c,b,j]  (b↔c swap on host)
        // STEOM mirror; uses d_eri_ovvv_ on device.
        {
            const size_t inter1_sz = (size_t)NO * NV * NV * NO;
            const size_t result_sz = (size_t)NV * NV * NV * NO;
            bool termB_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_t1_dev = nullptr, *d_inter1 = nullptr, *d_result = nullptr;
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_inter1, inter1_sz   * sizeof(real_t));
                tracked_cudaMalloc(&d_result, result_sz   * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                // Stage 1
                cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            NO, NO*NV*NV, NV, &one,
                            d_t1_dev,    NV,
                            d_eri_ovvv_, NV,
                            &zero,
                            d_inter1,    NO);
                // Stage 2
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                            NV*NV*NO, NV, NO, &one,
                            d_inter1,  NV*NV*NO,
                            d_t1_dev,  NV,
                            &zero,
                            d_result,  NV*NV*NO);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_inter1);
                termB_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    std::vector<real_t> h_result(result_sz, 0.0);
                    cudaMemcpy(h_result.data(), d_result, result_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int k = 0; k < NO; ++k) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVVV(k,c,b,d) * H_T1(j,d);
                                        v += H_T1(k,a) * s;
                                    }
                                    const real_t got = h_result[(((size_t)a*NV+c)*NV+b)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[EA build self-check] Term B 2-GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // ship 4: device scatter wvvvo_w_t1[a,b,c,j] -= d_result[a,c,b,j] (b↔c)
                ea_wvvvo_wt1_scatter_kernel<<<wt1_blk, wt1_thr>>>(d_result, d_wt1, -1.0, 1, NO, NV);
                tracked_cudaFree(d_result);
            }
#endif
            if (!termB_gpu) {
                std::vector<real_t> hkbcj((size_t)NO*NV*NV*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVVV(k,c,b,d) * H_T1(j,d);
                                hkbcj[(((size_t)k*NV+b)*NV+c)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int k = 0; k < NO; ++k)
                                    s += H_T1(k,a) * hkbcj[(((size_t)k*NV+b)*NV+c)*NO+j];
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] -= s;
                            }
            }
        }

        // Term C: - Σ_dl ovvv[l,d,a,c]·t1[l,b]·t1[j,d]
        //   Stage 1 batched: u_C[l,(a,c),j] = Σ_d slab_l[(a,c),d]·t1[j,d]  (per-l batch)
        //                     contracts h_ovvv axis_1 = d (non-trailing)
        //   Stage 2: result_C[b,(a,c,j)] = Σ_l t1[l,b]·u_C[l,(a,c,j)]
        //   wvvvo_w_t1[a,b,c,j] -= result_C[b,a,c,j]  (a↔b swap on host)
        // STEOM mirror.
        {
            const size_t u_C_sz    = (size_t)NO * NV * NV * NO;
            const size_t result_sz = (size_t)NV * NV * NV * NO;
            bool termC_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_t1_dev = nullptr, *d_u_C = nullptr, *d_result = nullptr;
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_u_C,    u_C_sz       * sizeof(real_t));
                tracked_cudaMalloc(&d_result, result_sz    * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                cublasDgemmStridedBatched(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_T,
                    NO, NV*NV, NV,
                    &one,
                    d_t1_dev, NV, /*strideA=*/0,
                    d_eri_ovvv_, NV*NV, /*strideB=*/(long long)NV*NV*NV,
                    &zero,
                    d_u_C, NO, /*strideC=*/(long long)NV*NV*NO,
                    NO);
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                            NV*NV*NO, NV, NO, &one,
                            d_u_C,    NV*NV*NO,
                            d_t1_dev, NV,
                            &zero,
                            d_result, NV*NV*NO);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_u_C);
                termC_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    std::vector<real_t> h_result(result_sz, 0.0);
                    cudaMemcpy(h_result.data(), d_result, result_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int l = 0; l < NO; ++l) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVVV(l,d,a,c) * H_T1(j,d);
                                        v += H_T1(l,b) * s;
                                    }
                                    const real_t got = h_result[(((size_t)b*NV+a)*NV+c)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[EA build self-check] Term C 2-GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // ship 4: device scatter wvvvo_w_t1[a,b,c,j] -= d_result[b,a,c,j] (a↔b)
                ea_wvvvo_wt1_scatter_kernel<<<wt1_blk, wt1_thr>>>(d_result, d_wt1, -1.0, 2, NO, NV);
                tracked_cudaFree(d_result);
            }
#endif
            if (!termC_gpu) {
                std::vector<real_t> hlacj((size_t)NO*NV*NV*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int l = 0; l < NO; ++l)
                    for (int a = 0; a < NV; ++a)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVVV(l,d,a,c) * H_T1(j,d);
                                hlacj[(((size_t)l*NV+a)*NV+c)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int l = 0; l < NO; ++l)
                                    s += H_T1(l,b) * hlacj[(((size_t)l*NV+a)*NV+c)*NO+j];
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] -= s;
                            }
            }
        }

        // Term D: + Σ_dkl ovov[k,c,l,d]·τ[k,l,a,b]·t1[j,d]   (τ = t2 + t1·t1)
        // GPU path: single GEMM of shape (m=NV·NO, n=NV², k=NO²) with
        //   row-major C[(a,b),(c,j)] = Σ_{(k,l)} τ[(k,l),(a,b)] · u'[(k,l),(c,j)]
        // mirrors the STEOM build_dressed_intermediates Term D ship; output
        // h_termD layout = wvvvo_w_t1 (no scatter), omp += accumulate.
        // anthracene canonical-skip ON regression = +300 s here (host loop
        // O(NV³·NO³) cache-unfriendly H_T2(k,l,...) stride).  Single-GPU path
        // only — multi_gpu_gemm_nslab helper is in steom_ccsd_operator.cu
        // translation unit and not visible here; A100 single-device GEMM
        // = ~0.1 s flop + ~0.5 s PCIe at anthracene.
        {
            const size_t u_klcj_sz   = (size_t)NO*NO*NV*NO;
            const size_t tau_klab_sz = (size_t)NO*NO*NV*NV;
            const size_t termD_sz    = (size_t)NV*NV*NV*NO;
            bool termD_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                std::vector<real_t> h_u_klcj(u_klcj_sz, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                h_u_klcj[(((size_t)k*NO+l)*NV+c)*NO+j] = s;
                            }
                std::vector<real_t> h_tau_klab(tau_klab_sz, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int a = 0; a < NV; ++a)
                            for (int b = 0; b < NV; ++b)
                                h_tau_klab[(((size_t)k*NO+l)*NV+a)*NV+b] =
                                    H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_u = nullptr, *d_tau = nullptr, *d_C = nullptr;
                tracked_cudaMalloc(&d_u,   u_klcj_sz   * sizeof(real_t));
                tracked_cudaMalloc(&d_tau, tau_klab_sz * sizeof(real_t));
                tracked_cudaMalloc(&d_C,   termD_sz    * sizeof(real_t));
                cudaMemcpy(d_u,   h_u_klcj.data(),   u_klcj_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_tau, h_tau_klab.data(), tau_klab_sz * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                            NV*NO, NV*NV, NO*NO, &one,
                            d_u, NV*NO, d_tau, NV*NV, &zero, d_C, NV*NO);
                tracked_cudaFree(d_u); tracked_cudaFree(d_tau);
                termD_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    std::vector<real_t> h_termD(termD_sz, 0.0);
                    cudaMemcpy(h_termD.data(), d_C, termD_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int k = 0; k < NO; ++k) for (int l = 0; l < NO; ++l) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                        v += s * (H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b));
                                    }
                                    const real_t got = h_termD[(((size_t)a*NV+b)*NV+c)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[EA build self-check] Term D GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // ship 4: device scatter wvvvo_w_t1[i] += d_C[i] (identity)
                ea_wvvvo_wt1_scatter_kernel<<<wt1_blk, wt1_thr>>>(d_C, d_wt1, 1.0, 0, NO, NV);
                tracked_cudaFree(d_C);
            }
#endif
            if (!termD_gpu) {
                std::vector<real_t> u((size_t)NO*NV*NO*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int c = 0; c < NV; ++c)
                        for (int l = 0; l < NO; ++l)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                u[(((size_t)k*NV+c)*NO+l)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int k = 0; k < NO; ++k)
                                    for (int l = 0; l < NO; ++l) {
                                        const real_t tau_klab = H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
                                        s += u[(((size_t)k*NV+c)*NO+l)*NO+j] * tau_klab;
                                    }
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] += s;
                            }
            }
        }

        // ship 4: copy the device-accumulated 4-term result back to the host
        // wvvvo_w_t1 (consumed by the Wvvvo assembly loop below).
#ifndef GANSU_CPU_ONLY
        if (wt1_dev) {
            if (wvvvo_dev_asm) d_wvt1_keep = d_wt1;   // stage 3: keep for device assembly
            if (!wvvvo_dev_asm || wvvvo_keep_validate)
                cudaMemcpy(wvvvo_w_t1.data(), d_wt1, wvvvo_sz_local * sizeof(real_t), cudaMemcpyDeviceToHost);
            if (!wvvvo_dev_asm) tracked_cudaFree(d_wt1);
        }
#endif
    }

    subprof("Wvvvv build / wvvvo_w_t1 (canonical-skip 4-term GEMM)");
    // ============================================================
    //  Wvvvo[a,b,c,j]  (PySCF Wvvvo, 11 terms)
    //   = -Σ_l W1ovov[l,a,j,c] t1[l,b]                # 'alcj,lb' on W1ovov.T(1,0,3,2)
    //   - Σ_k W1ovvo[k,b,c,j] t1[k,a]
    //   + 2 Σ_{l,d} ovvv[l,d,a,c] t2[l,j,d,b]
    //   - Σ_{l,d} ovvv[l,d,a,c] t2[l,j,b,d]
    //   - Σ_{l,d} ovvv[l,c,a,d] t2[l,j,d,b]
    //   - Σ_{k,d} ovvv[k,c,b,d] t2[j,k,d,a]
    //   + Σ_{k,l} ooov[l,j,k,c] t2[l,k,b,a]            # using ovoo[k,c,l,j] = ooov[l,j,k,c]
    //   + Σ_{k,l} ooov[l,j,k,c] t1[l,b] t1[k,a]
    //   - Σ_k Fov[k,c] t2[k,j,a,b]
    //   + ovvv[j,b,c,a]                                # bare term, ovvv.transpose(3,1,2,0)
    //   + Σ_d Wvvvv[a,b,c,d] t1[j,d]                   # only if t1 != 0
    //  layout: [a, b, c, j]  index = ((a*NV + b)*NV + c)*NO + j
    // ============================================================
    // Wvvvo big terms (term3 + term4) via 3 GEMMs (mirrors the STEOM build):
    //   term3 = Σ_ld [2 ovvv[l,d,a,c] t2[l,j,d,b] − ovvv[l,d,a,c] t2[l,j,b,d] − ovvv[l,c,a,d] t2[l,j,d,b]]
    //   term4 = −Σ_kd ovvv[k,c,b,d] t2[j,k,d,a].
    // Each M[NV²×KV] = A[NV²×KV]·B[KV×KV] (KV=NO·NV) over repacked ovvv/t2; scattered into
    // Wvvvo[a,b,c,j]. 3a+3c fused (shared B=t2[l,j,d,b]). Cheap terms stay in the host loop.
    const int KV = NO * NV;
    std::vector<real_t> wvvvo_big;
    bool wvvvo_big_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t Asz = (size_t)NV2 * KV, Bsz = (size_t)KV * KV, Msz = (size_t)NV2 * KV;
        const size_t bigsz = (size_t)NV * NV * NV * NO;
        // Device-resident repack/scatter (ship 3): gather A off d_eri_ovvv_ and
        // B off d_t2_ straight on the device, GEMM, scatter-accumulate into dBig,
        // one final D2H.  Eliminates the 7 host repack arrays (~4.9 GB host RAM
        // at naphthalene), the 3× dA/dB H2D, the 3× dM D2H, and the 3 host
        // scatter loops — the SUBPROF "Wvvvo term3+4" 6.7 s hotspot.  Layouts
        // mirror the validated host repacks; GEMM args unchanged.
        real_t *dA = nullptr, *dB = nullptr, *dM = nullptr, *dBig = nullptr;
        tracked_cudaMalloc(&dA,   Asz   * sizeof(real_t));
        tracked_cudaMalloc(&dB,   Bsz   * sizeof(real_t));
        tracked_cudaMalloc(&dM,   Msz   * sizeof(real_t));
        tracked_cudaMalloc(&dBig, bigsz * sizeof(real_t));
        cudaMemset(dBig, 0, bigsz * sizeof(real_t));
        const real_t one = 1.0, zero = 0.0;
        const int thr = 256;
        const int blkA = (int)std::min<size_t>((Asz   + thr - 1) / thr, 65535);
        const int blkB = (int)std::min<size_t>((Bsz   + thr - 1) / thr, 65535);
        const int blkS = (int)std::min<size_t>((bigsz + thr - 1) / thr, 65535);
        // All work on the legacy default stream → repack → GEMM → scatter are
        // implicitly ordered; consecutive scatters serialize their dBig RMW.
        auto gemm_scatter_dev = [&](int Amode, int Bmode, real_t coeff, int free_bc) {
            ea_wvvvo_repack_A_kernel<<<blkA, thr>>>(d_eri_ovvv_, dA, NO, NV, Amode);
            ea_wvvvo_repack_B_kernel<<<blkB, thr>>>(d_t2_,       dB, NO, NV, Bmode);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, KV, NV2, KV, &one,
                        dB, KV, dA, KV, &zero, dM, KV);
            ea_wvvvo_scatter_kernel<<<blkS, thr>>>(dM, dBig, coeff, free_bc, NO, NV);
        };
        gemm_scatter_dev(0, 0,  1.0, 0);   // hAcomb · hB   (term3 parts 1+3)
        gemm_scatter_dev(1, 1, -1.0, 0);   // hA     · hBp  (term3 part 2)
        gemm_scatter_dev(2, 2, -1.0, 1);   // hA4    · hB4  (term4)
        tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dM);
        if (wvvvo_dev_asm) d_big_keep = dBig;   // stage 3: keep for device assembly
        if (!wvvvo_dev_asm || wvvvo_keep_validate) {
            wvvvo_big.assign(bigsz, 0.0);
            cudaMemcpy(wvvvo_big.data(), dBig, bigsz * sizeof(real_t), cudaMemcpyDeviceToHost);
        }
        if (!wvvvo_dev_asm) tracked_cudaFree(dBig);
        wvvvo_big_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            const int as[2] = {0, NV - 1};
            for (int ai = 0; ai < 2; ++ai) { const int a = as[ai];
                for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                    for (int c = 0; c < NV; ++c) for (int j = 0; j < NO; ++j) {
                        real_t v = 0.0;
                        for (int l = 0; l < NO; ++l) for (int d = 0; d < NV; ++d) {
                            v += 2.0 * H_OVVV(l,d,a,c) * H_T2(l,j,d,b);
                            v -=       H_OVVV(l,d,a,c) * H_T2(l,j,b,d);
                            v -=       H_OVVV(l,c,a,d) * H_T2(l,j,d,b);
                        }
                        for (int k = 0; k < NO; ++k) for (int d = 0; d < NV; ++d)
                            v -= H_OVVV(k,c,b,d) * H_T2(j,k,d,a);
                        dmax = std::max(dmax, std::fabs(v - wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j]));
                    }
            }
            std::cout << "[EA-EOM build self-check] Wvvvo term3+4 (device-resident) vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif

    subprof("Wvvvo term3+4 GEMM (big)");
    // GPU GEMM port of Wvvvo term5 (the EA build_dressed hotspot, O(NV³·NO³)
    // strided/memory-bound — mirrors the validated STEOM port):
    //   ct[a,b,c,j] = Σ_{k,l} ooov(l,j,k,c)·(t2(l,k,b,a) + t1(l,b)·t1(k,a))
    // C[(j,c),(b,a)] = A·B, A[(j,c),(k,l)]=ooov(l,j,k,c), B[(k,l),(b,a)]=tau2(l,k,b,a).
    const int JC_M5 = NO*NV, BA_N5 = NV*NV, KL_K5 = NO*NO;
    std::vector<real_t> wvvvo_t5;
    bool wvvvo_t5_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)JC_M5*KL_K5), hB((size_t)KL_K5*BA_N5);
        #pragma omp parallel for collapse(2)
        for (int j=0;j<NO;++j) for (int c=0;c<NV;++c)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hA[(size_t)(j*NV+c)*KL_K5+(k*NO+l)] = H_OOOV(l,j,k,c);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
            for (int b=0;b<NV;++b) for (int a=0;a<NV;++a)
                hB[(size_t)(k*NO+l)*BA_N5+(b*NV+a)] = H_T2(l,k,b,a) + H_T1(l,b)*H_T1(k,a);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)JC_M5*KL_K5*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)KL_K5*BA_N5*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)JC_M5*BA_N5*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)JC_M5*KL_K5*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)KL_K5*BA_N5*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,BA_N5,JC_M5,KL_K5,&one,dB,BA_N5,dA,KL_K5,&zero,dC,BA_N5);
        tracked_cudaFree(dA);tracked_cudaFree(dB);
        if (wvvvo_dev_asm) d_t5_keep = dC;   // stage 3: keep for device assembly
        if (!wvvvo_dev_asm || wvvvo_keep_validate) {
            wvvvo_t5.assign((size_t)JC_M5*BA_N5,0.0);
            cudaMemcpy(wvvvo_t5.data(),dC,(size_t)JC_M5*BA_N5*sizeof(real_t),cudaMemcpyDeviceToHost);
        }
        if (!wvvvo_dev_asm) tracked_cudaFree(dC);
        wvvvo_t5_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            const int as[2]={0,NV-1};
            for (int ai=0;ai<2;++ai){ const int a=as[ai];
                for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                    for (int c=0;c<NV;c+=(NV/2>0?NV/2:1)) for (int j=0;j<NO;++j) {
                        real_t v=0.0;
                        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l) {
                            real_t klc=H_OOOV(l,j,k,c);
                            v += klc*H_T2(l,k,b,a) + klc*H_T1(l,b)*H_T1(k,a);
                        }
                        dmax=std::max(dmax,std::fabs(v-wvvvo_t5[(size_t)(j*NV+c)*BA_N5+(b*NV+a)]));
                    }
            }
            std::cout << "[EA-EOM build self-check] Wvvvo term5 (ooov·tau2) GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    subprof("Wvvvo term5 GEMM");
    // GPU GEMM port of three remaining Wvvvo inner-sum hotspots (canonical_skip_wvvvv_
    // on, anthracene profile = each ~21 G ops/scatter, STEOM build mirror).
    //   Item 1: ct1[b, (a,j,c)] = Σ_l T1(l,b) · W1OVOV(l, a, j, c)
    //   Item 2: ct2[a, (b,c,j)] = Σ_k T1(k,a) · W1OVVO(k, b, c, j)
    //   Item 5: ct5[c, (j,a,b)] = Σ_k Fov(k,c) · T2(k, j, a, b)
    // B-side direct (no repack: h_W1ovov / h_W1ovvo / h_t2 leading dim = N),
    // A-side small repack (NO×NV doubles). Sign applied at scatter site
    // (`v -= ct_wvvvo_N[...]`).
    const int M_w1 = NV, N_w1 = NV*NO*NV, K_w1 = NO;
    const int M_w2 = NV, N_w2 = NV*NV*NO, K_w2 = NO;
    const int M_w5 = NV, N_w5 = NO*NV*NV, K_w5 = NO;
    std::vector<real_t> ct_wvvvo_1, ct_wvvvo_2, ct_wvvvo_5;
    bool wvvvo_1_gpu = false, wvvvo_2_gpu = false, wvvvo_5_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        // --- Item 1: A[b,l] = T1(l,b); B = h_W1ovov direct (stride l = N_w1) ---
        {
            std::vector<real_t> hA((size_t)M_w1*K_w1);
            #pragma omp parallel for
            for (int b = 0; b < NV; ++b)
                for (int l = 0; l < NO; ++l) hA[(size_t)b*K_w1 + l] = h_t1[l*NV + b];
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w1*K_w1*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w1*N_w1*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w1*N_w1*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w1*K_w1*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_W1ovov.data(),(size_t)K_w1*N_w1*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w1,M_w1,K_w1,&one,dB,N_w1,dA,K_w1,&zero,dC,N_w1);
            tracked_cudaFree(dA);tracked_cudaFree(dB);
            if (wvvvo_dev_asm) d_ct1_keep = dC;   // stage 3: keep for device assembly
            if (!wvvvo_dev_asm || wvvvo_keep_validate) {
                ct_wvvvo_1.assign((size_t)M_w1*N_w1, 0.0);
                cudaMemcpy(ct_wvvvo_1.data(),dC,(size_t)M_w1*N_w1*sizeof(real_t),cudaMemcpyDeviceToHost);
            }
            if (!wvvvo_dev_asm) tracked_cudaFree(dC);
            wvvvo_1_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int l=0;l<NO;++l) t += H_W1OVOV(l,a,j,c)*H_T1(l,b);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)]));
                            }
                std::cout << "[EA-EOM build self-check] Wvvvo item 1 (W1ovov·t1): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // --- Item 2: A[a,k] = T1(k,a); B = h_W1ovvo direct (stride k = N_w2) ---
        {
            std::vector<real_t> hA((size_t)M_w2*K_w2);
            #pragma omp parallel for
            for (int a = 0; a < NV; ++a)
                for (int k = 0; k < NO; ++k) hA[(size_t)a*K_w2 + k] = h_t1[k*NV + a];
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w2*K_w2*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w2*N_w2*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w2*N_w2*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w2*K_w2*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_W1ovvo.data(),(size_t)K_w2*N_w2*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w2,M_w2,K_w2,&one,dB,N_w2,dA,K_w2,&zero,dC,N_w2);
            tracked_cudaFree(dA);tracked_cudaFree(dB);
            if (wvvvo_dev_asm) d_ct2_keep = dC;   // stage 3: keep for device assembly
            if (!wvvvo_dev_asm || wvvvo_keep_validate) {
                ct_wvvvo_2.assign((size_t)M_w2*N_w2, 0.0);
                cudaMemcpy(ct_wvvvo_2.data(),dC,(size_t)M_w2*N_w2*sizeof(real_t),cudaMemcpyDeviceToHost);
            }
            if (!wvvvo_dev_asm) tracked_cudaFree(dC);
            wvvvo_2_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int k=0;k<NO;++k) t += H_W1OVVO(k,b,c,j)*H_T1(k,a);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)]));
                            }
                std::cout << "[EA-EOM build self-check] Wvvvo item 2 (W1ovvo·t1): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // --- Item 5: A[c,k] = Fov(k,c); B = h_t2 direct (stride k = N_w5) ---
        {
            std::vector<real_t> hA((size_t)M_w5*K_w5);
            #pragma omp parallel for
            for (int c = 0; c < NV; ++c)
                for (int k = 0; k < NO; ++k) hA[(size_t)c*K_w5 + k] = h_Fov[k*NV + c];
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w5*K_w5*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w5*N_w5*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w5*N_w5*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w5*K_w5*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_t2.data(),(size_t)K_w5*N_w5*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w5,M_w5,K_w5,&one,dB,N_w5,dA,K_w5,&zero,dC,N_w5);
            tracked_cudaFree(dA);tracked_cudaFree(dB);
            if (wvvvo_dev_asm) d_ct5_keep = dC;   // stage 3: keep for device assembly
            if (!wvvvo_dev_asm || wvvvo_keep_validate) {
                ct_wvvvo_5.assign((size_t)M_w5*N_w5, 0.0);
                cudaMemcpy(ct_wvvvo_5.data(),dC,(size_t)M_w5*N_w5*sizeof(real_t),cudaMemcpyDeviceToHost);
            }
            if (!wvvvo_dev_asm) tracked_cudaFree(dC);
            wvvvo_5_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int k=0;k<NO;++k) t += h_Fov[k*NV + c]*H_T2(k,j,a,b);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)]));
                            }
                std::cout << "[EA-EOM build self-check] Wvvvo item 5 (Fov·t2): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
    }
#endif

    subprof("Wvvvo items 1/2/5 GEMM");
    const size_t wvvvo_sz = (size_t)NV * NV * NV * NO;
    std::vector<real_t> h_Wvvvo;   // host path only; device path builds d_Wvvvo_ directly
#ifndef GANSU_CPU_ONLY
    if (wvvvo_dev_asm) {
        // Stage 3: assemble d_Wvvvo_ on the device from the resident per-term
        // buffers (no D2H of the NV³·NO terms, no host assemble, no H2D upload).
        tracked_cudaMalloc(&d_Wvvvo_, wvvvo_sz * sizeof(real_t));
        const int thr = 256;
        const int blk = (int)std::min<size_t>((wvvvo_sz + thr - 1) / thr, 65535);
        ea_wvvvo_assemble_kernel<<<blk, thr>>>(d_eri_ovvv_, d_ct1_keep, d_ct2_keep,
            d_big_keep, d_t5_keep, d_ct5_keep, d_wvt1_keep, d_Wvvvo_, NO, NV);
        if (wvvvo_keep_validate) {
            std::vector<real_t> h_chk(wvvvo_sz);
            cudaMemcpy(h_chk.data(), d_Wvvvo_, wvvvo_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
            real_t dmax = 0.0;
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1))
              for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
                  for (int j=0;j<NO;++j) {
                    real_t v = H_OVVV(j,b,c,a);
                    v -= ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)];
                    v -= ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)];
                    v += wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j];
                    v += wvvvo_t5[(size_t)(j*NV+c)*BA_N5 + (b*NV+a)];
                    v -= ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)];
                    v += wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j];
                    dmax = std::max(dmax, std::fabs(v - h_chk[(((size_t)a*NV+b)*NV+c)*NO+j]));
                  }
            std::cout << "[EA-EOM build self-check] Wvvvo device assembly vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
        tracked_cudaFree(d_ct1_keep); tracked_cudaFree(d_ct2_keep); tracked_cudaFree(d_big_keep);
        tracked_cudaFree(d_t5_keep);  tracked_cudaFree(d_ct5_keep); tracked_cudaFree(d_wvt1_keep);
    }
#endif
    if (!wvvvo_dev_asm) {
    h_Wvvvo.assign(wvvvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVV(j,b,c,a);  // bare: ovvv[j,b,c,a] = (jb|ca) chemist
                    if (wvvvo_1_gpu) {
                        v -= ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)];
                    } else {
                        for (int l = 0; l < NO; ++l)
                            v -= H_W1OVOV(l,a,j,c) * H_T1(l,b);
                    }
                    if (wvvvo_2_gpu) {
                        v -= ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)];
                    } else {
                        for (int k = 0; k < NO; ++k)
                            v -= H_W1OVVO(k,b,c,j) * H_T1(k,a);
                    }
                    if (wvvvo_big_gpu) {
                        v += wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j];   // term3 + term4 (GEMM)
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                v += 2.0 * H_OVVV(l,d,a,c) * H_T2(l,j,d,b);
                                v -=       H_OVVV(l,d,a,c) * H_T2(l,j,b,d);
                                v -=       H_OVVV(l,c,a,d) * H_T2(l,j,d,b);
                            }
                        for (int k = 0; k < NO; ++k)
                            for (int d = 0; d < NV; ++d)
                                v -= H_OVVV(k,c,b,d) * H_T2(j,k,d,a);
                    }
                    if (wvvvo_t5_gpu) {
                        v += wvvvo_t5[(size_t)(j*NV+c)*BA_N5 + (b*NV+a)];   // term5 (GEMM)
                    } else {
                        for (int k = 0; k < NO; ++k)
                            for (int l = 0; l < NO; ++l) {
                                real_t klc_lj = H_OOOV(l,j,k,c);  // ovoo[k,c,l,j] = ooov[l,j,k,c]
                                v += klc_lj * H_T2(l,k,b,a);
                                v += klc_lj * H_T1(l,b) * H_T1(k,a);
                            }
                    }
                    if (wvvvo_5_gpu) {
                        v -= ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)];
                    } else {
                        for (int k = 0; k < NO; ++k)
                            v -= h_Fov[k*NV + c] * H_T2(k,j,a,b);
                    }
                    if (canonical_skip_wvvvv_) {
                        // P5 canonical-skip: Σ_d Wvvvv[a,b,c,d]·t1[j,d] precomputed
                        // term-by-term above without materializing nvir⁴ Wvvvv.
                        v += wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j];
                    } else {
                        for (int d = 0; d < NV; ++d)
                            v += H_WVVVV(a,b,c,d) * H_T1(j,d);
                    }
                    h_Wvvvo[(((size_t)a * NV + b) * NV + c) * NO + j] = v;
                }
    }  // close if (!wvvvo_dev_asm)

    subprof("Wvvvo final host assembly (scatter)");
    // ==========================================================
    //  Upload all intermediates to device
    // ==========================================================
    const size_t loo_sz = (size_t)NO * NO;
    const size_t lvv_sz = (size_t)NV * NV;
    const size_t fov_sz = (size_t)NO * NV;

    tracked_cudaMalloc(&d_Loo_,   loo_sz   * sizeof(real_t));
    cudaMemcpy(d_Loo_,   h_Loo.data(),   loo_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Lvv_,   lvv_sz   * sizeof(real_t));
    cudaMemcpy(d_Lvv_,   h_Lvv.data(),   lvv_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Fov_,   fov_sz   * sizeof(real_t));
    cudaMemcpy(d_Fov_,   h_Fov.data(),   fov_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovov_, ovov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovov_, h_Wovov.data(), ovov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovvo_, ovvo_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovvo_, h_Wovvo.data(), ovvo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    if (!wvovv_on_device) {   // stage 1: device path already built d_Wvovv_
        tracked_cudaMalloc(&d_Wvovv_, wvovv_sz * sizeof(real_t));
        cudaMemcpy(d_Wvovv_, h_Wvovv.data(), wvovv_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    if (!canonical_skip_wvvvv_) {
        tracked_cudaMalloc(&d_Wvvvv_, vvvv_sz  * sizeof(real_t));
        cudaMemcpy(d_Wvvvv_, h_Wvvvv.data(), vvvv_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    }  // canonical_skip: d_Wvvvv_ stays nullptr (native operator handles σ2 via per-pair PNO)
    if (!wvvvo_dev_asm) {   // stage 3: device path already built d_Wvvvo_
        tracked_cudaMalloc(&d_Wvvvo_, wvvvo_sz * sizeof(real_t));
        cudaMemcpy(d_Wvvvo_, h_Wvvvo.data(), wvvvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    }

#ifndef GANSU_CPU_ONLY
    // Precompute the 3 reorganized ring matrices (M_A/M_B/M_C, each [NO·NV × NO·NV])
    // used to GEMM-ify the σ2 ring contractions in apply().  GPU only; on the CPU
    // path the per-thread loops in apply() handle the ring terms directly.
    if (gpu::gpu_available()) {
        const size_t M_sz = (size_t)NO * NV * NO * NV;
        tracked_cudaMalloc(&d_M_ringA_, M_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_M_ringB_, M_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_M_ringC_, M_sz * sizeof(real_t));
        const int thr = 256;
        const int blk = (int)((M_sz + thr - 1) / thr);
        ea_build_ring_M_kernel<<<blk, thr>>>(d_Wovov_, d_Wovvo_,
                                             d_M_ringA_, d_M_ringB_, d_M_ringC_, NO, NV);
        cudaDeviceSynchronize();
    }
#endif

    subprof("H2D uploads + ring-M kernel");
    std::cout << "  EA-EOM-CCSD dressed intermediates built (PySCF EA definitions)." << std::endl;

    #undef H_WOOOV
    #undef H_W1OVOV
    #undef H_W1OVVO
    #undef H_WVVVV
}

#undef H_OVOV
#undef H_OOOV
#undef H_OOVV
#undef H_OVVO
#undef H_OVVV
#undef H_VVVV
#undef H_T1
#undef H_T2


// ==================================================================
//  apply — sub-phase 2.3-2.6 full PySCF EA-EOM-CCSD matvec
// ==================================================================
//  The full matvec runs only when intermediates have been built
//  (d_Lvv_ != nullptr). When d_eri_mo was passed as nullptr (unit-test path)
//  the operator falls back to a diagonal-only matvec.
void EAEOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    // P5 canonical-skip contract: when the dressed Wvvvv was elided (native per-pair
    // σ takes over), canonical apply() must not be called — the σ2 kernels would
    // segfault on nullptr d_Wvvvv_. The Davidson driver wraps this in dlpno_op when
    // native EOM is active, so this is unreachable in practice; defensive assert.
    if (canonical_skip_wvvvv_ && d_Wvvvv_ == nullptr) {
        throw std::runtime_error(
            "EAEOMCCSDOperator::apply called with canonical-skip Wvvvv elided — "
            "this operator's σ should be wrapped by the native DLPNO-EA-EOM operator.");
    }
    // (a) Stub path: intermediates not built → diagonal-only matvec.
    if (d_Lvv_ == nullptr) {
        if (!gpu::gpu_available()) {
            #pragma omp parallel for
            for (int idx = 0; idx < total_dim_; ++idx) {
                d_output[idx] = d_diagonal_[idx] * d_input[idx];
            }
        } else {
#ifndef GANSU_CPU_ONLY
            const int threads = 256;
            const int blocks  = (total_dim_ + threads - 1) / threads;
            ea_eom_diag_matvec_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
        }
        return;
    }

    // (b) Full PySCF EA-EOM-CCSD matvec.
    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + p_dim_;
    real_t*       d_s1 = d_output;
    real_t*       d_s2 = d_output + p_dim_;

    if (!gpu::gpu_available()) {
        // -------- CPU path --------
        // σ1
        #pragma omp parallel for
        for (int a = 0; a < nvir_; ++a) {
            real_t s = 0.0;
            for (int c = 0; c < nvir_; ++c) s += d_Lvv_[a * nvir_ + c] * d_r1[c];
            for (int l = 0; l < nocc_; ++l)
                for (int d = 0; d < nvir_; ++d) {
                    real_t fov_ld = d_Fov_[l * nvir_ + d];
                    s += 2.0 * fov_ld * d_r2[((size_t)l * nvir_ + a) * nvir_ + d];
                    s -=       fov_ld * d_r2[((size_t)l * nvir_ + d) * nvir_ + a];
                }
            for (int l = 0; l < nocc_; ++l)
                for (int c = 0; c < nvir_; ++c)
                    for (int dd = 0; dd < nvir_; ++dd) {
                        real_t w1 = d_Wvovv_[(((size_t)a * nocc_ + l) * nvir_ + c) * nvir_ + dd];
                        real_t w2 = d_Wvovv_[(((size_t)a * nocc_ + l) * nvir_ + dd) * nvir_ + c];
                        s += (2.0 * w1 - w2) * d_r2[((size_t)l * nvir_ + c) * nvir_ + dd];
                    }
            d_s1[a] = s;
        }

        // tmp[k] pre-stage
        std::vector<real_t> tmp_k(nocc_, 0.0);
        for (int k = 0; k < nocc_; ++k) {
            real_t s = 0.0;
            for (int l = 0; l < nocc_; ++l)
                for (int c = 0; c < nvir_; ++c)
                    for (int dd = 0; dd < nvir_; ++dd) {
                        real_t w_kcld = d_eri_ovov_[(((size_t)k * nvir_ + c) * nocc_ + l) * nvir_ + dd];
                        real_t w_kdlc = d_eri_ovov_[(((size_t)k * nvir_ + dd) * nocc_ + l) * nvir_ + c];
                        s += (2.0 * w_kcld - w_kdlc) * d_r2[((size_t)l * nvir_ + c) * nvir_ + dd];
                    }
            tmp_k[k] = s;
        }

        // σ2
        #pragma omp parallel for
        for (int idx = 0; idx < p2h_dim_; ++idx) {
            int b = idx % nvir_;
            int t = idx / nvir_;
            int a = t % nvir_;
            int j = t / nvir_;

            real_t s = 0.0;
            for (int c = 0; c < nvir_; ++c)
                s += d_Wvvvo_[(((size_t)a * nvir_ + b) * nvir_ + c) * nocc_ + j] * d_r1[c];
            for (int c = 0; c < nvir_; ++c)
                s += d_Lvv_[a * nvir_ + c] * d_r2[((size_t)j * nvir_ + c) * nvir_ + b];
            for (int dd = 0; dd < nvir_; ++dd)
                s += d_Lvv_[b * nvir_ + dd] * d_r2[((size_t)j * nvir_ + a) * nvir_ + dd];
            for (int l = 0; l < nocc_; ++l)
                s -= d_Loo_[l * nocc_ + j] * d_r2[((size_t)l * nvir_ + a) * nvir_ + b];
            for (int l = 0; l < nocc_; ++l)
                for (int dd = 0; dd < nvir_; ++dd) {
                    real_t wovvo = d_Wovvo_[(((size_t)l * nvir_ + b) * nvir_ + dd) * nocc_ + j];
                    real_t wovov = d_Wovov_[(((size_t)l * nvir_ + b) * nocc_ + j) * nvir_ + dd];
                    s += (2.0 * wovvo - wovov) * d_r2[((size_t)l * nvir_ + a) * nvir_ + dd];
                }
            for (int l = 0; l < nocc_; ++l)
                for (int c = 0; c < nvir_; ++c) {
                    real_t w = d_Wovov_[(((size_t)l * nvir_ + a) * nocc_ + j) * nvir_ + c];
                    s -= w * d_r2[((size_t)l * nvir_ + c) * nvir_ + b];
                }
            for (int l = 0; l < nocc_; ++l)
                for (int c = 0; c < nvir_; ++c) {
                    real_t w = d_Wovvo_[(((size_t)l * nvir_ + b) * nvir_ + c) * nocc_ + j];
                    s -= w * d_r2[((size_t)l * nvir_ + c) * nvir_ + a];
                }
            for (int c = 0; c < nvir_; ++c)
                for (int dd = 0; dd < nvir_; ++dd) {
                    s += d_Wvvvv_[(((size_t)a * nvir_ + b) * nvir_ + c) * nvir_ + dd]
                         * d_r2[((size_t)j * nvir_ + c) * nvir_ + dd];
                }
            for (int k = 0; k < nocc_; ++k)
                s -= tmp_k[k] * d_t2_[(((size_t)k * nocc_ + j) * nvir_ + a) * nvir_ + b];
            d_s2[idx] = s;
        }
    } else {
#ifndef GANSU_CPU_ONLY
        if (use_gpu_multi_) {
            apply_multi(d_input, d_output);   // Stage EA-5b/5c
        } else {
            apply_sigma_gpu(d_r1, d_r2, d_s1, d_s2,
                            d_Lvv_, d_Loo_, d_Fov_, d_Wovov_, d_Wovvo_, d_Wvovv_,
                            d_Wvvvo_, d_Wvvvv_, d_t2_, d_eri_ovov_,
                            d_M_ringA_, d_M_ringB_, d_M_ringC_,
                            (void*)gansu::gpu::GPUHandle::cublas(),
                            /*j_begin=*/0, /*j_end=*/nocc_, /*do_sigma1=*/true);
        }
#endif
    }
}

#ifndef GANSU_CPU_ONLY
// ==================================================================
//  Device-parametric GPU σ (Stage EA-5b).  Identical numerics to the
//  legacy single-GPU body, but every intermediate + the cuBLAS handle
//  are explicit arguments, so it runs unchanged on any device after a
//  cudaSetDevice (DeviceGuard) by the caller.  Scratch is allocated on
//  the current device.
// ==================================================================
void EAEOMCCSDOperator::apply_sigma_gpu(
    const real_t* d_r1, const real_t* d_r2, real_t* d_s1, real_t* d_s2,
    const real_t* Lvv, const real_t* Loo, const real_t* Fov,
    const real_t* Wovov, const real_t* Wovvo, const real_t* Wvovv,
    const real_t* Wvvvo, const real_t* Wvvvv, const real_t* t2,
    const real_t* eri_ovov, const real_t* M_ringA, const real_t* M_ringB,
    const real_t* M_ringC, void* cublas_v,
    int j_begin, int j_end, bool do_sigma1,
    real_t* scr_tmp_k, real_t* scr_r2T, real_t* scr_tmp) const
{
    cublasHandle_t cublas = (cublasHandle_t)cublas_v;
    const int threads = 256;
    const int jslab = j_end - j_begin;
    if (jslab <= 0 && !do_sigma1) return;

    // σ1 (full p sector; only on the owning device — device 0)
    if (do_sigma1) {
        const int blocks_1 = (p_dim_ + threads - 1) / threads;
        ea_eom_sigma1_full_kernel<<<blocks_1, threads>>>(
            Lvv, Fov, Wvovv, d_r1, d_r2, d_s1, nocc_, nvir_);
    }
    if (jslab <= 0) return;

    // Pre-stage: tmp[k] (full reduction; needed by the σ2 tmp·t2 term for every k).
    // EA-5d: use persistent scratch when provided (multi path) to avoid per-matvec malloc.
    real_t* d_tmp_k = scr_tmp_k;
    if (!d_tmp_k) tracked_cudaMalloc(&d_tmp_k, (size_t)nocc_ * sizeof(real_t));
    const int blocks_tmp = (nocc_ + threads - 1) / threads;
    ea_eom_sigma2_tmp_k_kernel<<<blocks_tmp, threads>>>(
        eri_ovov, d_r2, d_tmp_k, nocc_, nvir_);

    // Wvvvv·r2 and ring terms via GEMM (env-gated, same as legacy).
    static const bool wvvvv_gemm = [] {
        const char* e = std::getenv("GANSU_EA_WVVVV_GEMM");
        return !(e && e[0] == '0');
    }();
    static const bool ring_gemm_env = [] {
        const char* e = std::getenv("GANSU_EA_RING_GEMM");
        return !(e && e[0] == '0');
    }();
    const bool ring_gemm = ring_gemm_env && (M_ringA != nullptr);

    const int nvir2 = nvir_ * nvir_;
    const size_t joff = (size_t)j_begin * nvir2;   // σ2/r2 row offset for this slab

    // σ2 kernel over the slab [j_begin, j_end); writes its rows of the full d_s2.
    const int blocks_2 = (jslab * nvir2 + threads - 1) / threads;
    ea_eom_sigma2_full_kernel<<<blocks_2, threads>>>(
        Lvv, Loo, Wovov, Wovvo, Wvvvv, Wvvvo,
        d_tmp_k, t2, d_r1, d_r2, d_s2, nocc_, nvir_,
        wvvvv_gemm ? 0 : 1, ring_gemm ? 0 : 1, j_begin, j_end);

    const real_t one = 1.0;
    if (wvvvv_gemm) {
        // slab columns only: σ2[j∈slab,(ab)] += Wvvvv·r2[j∈slab,(cd)]  (N = jslab)
        cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    nvir2, jslab, nvir2, &one,
                    Wvvvv, nvir2, d_r2 + joff, nvir2, &one, d_s2 + joff, nvir2);
    }
    if (ring_gemm) {
        const real_t zero = 0.0, neg = -1.0;
        const int NK = nocc_ * nvir_;
        const int mrows = jslab * nvir_;                 // slab output rows (jb)/(ja)
        const size_t moff = (size_t)j_begin * nvir_ * NK; // M_ring row offset
        // R_B: σ2[(ja),b] -= M_B[slab rows]·r2[(lc),b]
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    nvir_, mrows, NK, &neg, d_r2, nvir_, M_ringB + moff, NK,
                    &one, d_s2 + joff, nvir_);
        // R_A + R_C → tmp[(jb),a] slab → scatter
        real_t* d_r2T = scr_r2T;
        real_t* d_tmp = scr_tmp;
        if (!d_r2T) tracked_cudaMalloc(&d_r2T, (size_t)p2h_dim_ * sizeof(real_t));
        if (!d_tmp) tracked_cudaMalloc(&d_tmp, (size_t)p2h_dim_ * sizeof(real_t));
        const int blocks_r2 = (p2h_dim_ + threads - 1) / threads;
        ea_r2_swap_vir_kernel<<<blocks_r2, threads>>>(d_r2, d_r2T, nocc_, nvir_);  // full (input)
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    nvir_, mrows, NK, &one, d_r2T, nvir_, M_ringA + moff, NK,
                    &zero, d_tmp + joff, nvir_);
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    nvir_, mrows, NK, &neg, d_r2, nvir_, M_ringC + moff, NK,
                    &one, d_tmp + joff, nvir_);
        const int blocks_sc = (jslab * nvir2 + threads - 1) / threads;
        ea_ring_scatter_kernel<<<blocks_sc, threads>>>(d_tmp, d_s2, nocc_, nvir_, j_begin, j_end);
        if (!scr_r2T) tracked_cudaFree(d_r2T);   // free only if locally allocated
        if (!scr_tmp) tracked_cudaFree(d_tmp);
    }
    if (!scr_tmp_k) tracked_cudaFree(d_tmp_k);
}

// ==================================================================
//  Stage EA-5c: j-slab multi-GPU σ.  σ1 (full) + device 0's σ2 slab are
//  computed on device 0 into d_output; each device d>0 computes ONLY its
//  output-occ σ2 slab into its replica workspace, then the slabs are
//  disjoint-gathered (cudaMemcpyPeer) into d_output.  No redundant work →
//  real speedup.  Result is identical to single-GPU (disjoint slabs cover
//  [0,nocc); GANSU_STEOM_EOM_MULTI_VALIDATE=1 cross-checks the gathered σ2
//  against a full device-0 reference on the first matvecs).
// ==================================================================
void EAEOMCCSDOperator::apply_multi(const real_t* d_input, real_t* d_output) const {
    const int nuse = (int)ws_.size();
    const int nvir2 = nvir_ * nvir_;

    // EA-5d: broadcast the input to every d>0 FIRST (device 0's stream is still idle),
    // so the peer copies do not serialize behind device 0's σ kernels.  Then launch all
    // devices' σ asynchronously (real overlap), then sync + gather.
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        cudaMemcpyPeerAsync(w.d_input, d, d_input, 0,
                            (size_t)total_dim_ * sizeof(real_t), 0);
    }

    // device 0: σ1 (full) + its own σ2 slab [0, j_end0) into d_output (async).
    apply_sigma_gpu(d_input, d_input + p_dim_, d_output, d_output + p_dim_,
                    d_Lvv_, d_Loo_, d_Fov_, d_Wovov_, d_Wovvo_, d_Wvovv_,
                    d_Wvvvo_, d_Wvvvv_, d_t2_, d_eri_ovov_,
                    d_M_ringA_, d_M_ringB_, d_M_ringC_,
                    (void*)gansu::gpu::GPUHandle::cublas(),
                    ws_[0].j_begin, ws_[0].j_end, /*do_sigma1=*/true,
                    ws_[0].d_tmp_k, ws_[0].d_r2T, ws_[0].d_ring_tmp);

    // d>0: compute σ2 slab into ws_[d].d_s2 (async on device d, persistent scratch).
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        apply_sigma_gpu(w.d_input, w.d_input + p_dim_, w.d_s1, w.d_s2,
                        w.d_Lvv, w.d_Loo, w.d_Fov, w.d_Wovov, w.d_Wovvo, w.d_Wvovv,
                        w.d_Wvvvo, w.d_Wvvvv, w.d_t2, w.d_eri_ovov,
                        w.d_M_ringA, w.d_M_ringB, w.d_M_ringC, w.cublas,
                        w.j_begin, w.j_end, /*do_sigma1=*/false,
                        w.d_tmp_k, w.d_r2T, w.d_ring_tmp);
    }
    // gather: sync each d>0 and copy its σ2 slab rows into d_output (disjoint).
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        cudaDeviceSynchronize();
        const size_t off = (size_t)w.j_begin * nvir2;
        const size_t cnt = (size_t)(w.j_end - w.j_begin) * nvir2;
        if (cnt > 0)
            cudaMemcpyPeer(d_output + p_dim_ + off, 0, w.d_s2 + off, d, cnt * sizeof(real_t));
    }
    { MultiGpuManager::DeviceGuard g0(0); cudaDeviceSynchronize(); }  // device 0 slab + σ1 done

    // Optional capped cross-check: full device-0 σ2 reference vs the gathered result.
    static const bool do_validate = [] {
        const char* e = std::getenv("GANSU_STEOM_EOM_MULTI_VALIDATE");
        return e && e[0] == '1';
    }();
    if (do_validate && multi_check_count_ < 3) {
        real_t* d_ref = nullptr;
        tracked_cudaMalloc(&d_ref, (size_t)p2h_dim_ * sizeof(real_t));
        real_t* d_s1tmp = nullptr;
        tracked_cudaMalloc(&d_s1tmp, (size_t)p_dim_ * sizeof(real_t));
        apply_sigma_gpu(d_input, d_input + p_dim_, d_s1tmp, d_ref,
                        d_Lvv_, d_Loo_, d_Fov_, d_Wovov_, d_Wovvo_, d_Wvovv_,
                        d_Wvvvo_, d_Wvvvv_, d_t2_, d_eri_ovov_,
                        d_M_ringA_, d_M_ringB_, d_M_ringC_,
                        (void*)gansu::gpu::GPUHandle::cublas(),
                        0, nocc_, /*do_sigma1=*/false,
                        ws_[0].d_tmp_k, ws_[0].d_r2T, ws_[0].d_ring_tmp);
        cudaDeviceSynchronize();
        std::vector<real_t> h_ref(p2h_dim_), h_out(p2h_dim_);
        cudaMemcpy(h_ref.data(), d_ref, (size_t)p2h_dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out.data(), d_output + p_dim_, (size_t)p2h_dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);
        real_t dmax = 0.0;
        for (int i = 0; i < p2h_dim_; ++i) dmax = std::max(dmax, std::fabs(h_ref[i] - h_out[i]));
        std::cout << "[EA-EOM Stage 5c self-check] gathered σ2 vs full device-0 ref: max|Δ| = "
                  << std::scientific << dmax << std::defaultfloat << " (expect ≤1e-11)" << std::endl;
        tracked_cudaFree(d_ref); tracked_cudaFree(d_s1tmp);
        ++multi_check_count_;
    }
}
#endif

void EAEOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; ++idx) {
            real_t d = d_diagonal_[idx];
            d_output[idx] = (std::fabs(d) > 1e-12) ? (d_input[idx] / d) : real_t(0.0);
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        ea_eom_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}


// ==================================================================
//  print_intermediate_norms
// ==================================================================
namespace {
real_t frobenius_norm_device(const real_t* d_ptr, size_t n) {
    if (d_ptr == nullptr || n == 0) return 0.0;
    std::vector<real_t> h_buf(n);
    cudaMemcpy(h_buf.data(), d_ptr, n * sizeof(real_t), cudaMemcpyDeviceToHost);
    real_t s = 0.0;
    for (size_t i = 0; i < n; ++i) s += h_buf[i] * h_buf[i];
    return std::sqrt(s);
}
} // namespace

void EAEOMCCSDOperator::print_intermediate_norms(std::ostream& os) const {
    const int NO = nocc_, NV = nvir_;
    const size_t loo_sz   = (size_t)NO * NO;
    const size_t lvv_sz   = (size_t)NV * NV;
    const size_t fov_sz   = (size_t)NO * NV;
    const size_t ovov_sz  = (size_t)NO * NV * NO * NV;
    const size_t ovvo_sz  = (size_t)NO * NV * NV * NO;
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    const size_t wvvvv_sz = (size_t)NV * NV * NV * NV;
    const size_t wvvvo_sz = (size_t)NV * NV * NV * NO;

    os << "  [EA-EOM-CCSD intermediate Frobenius norms]\n"
       << "    dims:  nocc=" << NO << "  nvir=" << NV
       <<           "  p_dim=" << p_dim_ << "  p2h_dim=" << p2h_dim_ << "\n";
    if (d_Loo_ == nullptr) {
        os << "    (intermediates not built — diagonal-only stub)\n";
        return;
    }
    os << std::fixed << std::setprecision(8)
       << "    ‖Loo‖       = " << frobenius_norm_device(d_Loo_,   loo_sz)   << "\n"
       << "    ‖Lvv‖       = " << frobenius_norm_device(d_Lvv_,   lvv_sz)   << "\n"
       << "    ‖Fov‖       = " << frobenius_norm_device(d_Fov_,   fov_sz)   << "\n"
       << "    ‖Wovov‖     = " << frobenius_norm_device(d_Wovov_, ovov_sz)  << "\n"
       << "    ‖Wovvo‖     = " << frobenius_norm_device(d_Wovvo_, ovvo_sz)  << "\n"
       << "    ‖Wvovv‖     = " << frobenius_norm_device(d_Wvovv_, wvovv_sz) << "\n"
       << "    ‖Wvvvv‖     = "
       << (d_Wvvvv_ ? frobenius_norm_device(d_Wvvvv_, wvvvv_sz) : 0.0)
       << (d_Wvvvv_ ? "" : "  (canonical-skip)") << "\n"
       << "    ‖Wvvvo‖     = " << frobenius_norm_device(d_Wvvvo_, wvvvo_sz) << "\n";
}

} // namespace gansu
