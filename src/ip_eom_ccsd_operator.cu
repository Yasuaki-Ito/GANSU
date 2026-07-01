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
 * @file ip_eom_ccsd_operator.cu
 * @brief IP-EOM-CCSD operator implementation (full sigma, sub-phases 1.5+1.6).
 *
 * The IP-EOM-CCSD bar-H intermediates are NOT the same as the EE-EOM ones —
 * EE uses symmetrized Foo/Fvv/Woooo/WoVVo/WoVvO (with 0.5 prefactors and
 * 0.5*Fov·t1 corrections), while IP-EOM uses the un-symmetrized PySCF
 * versions (Loo, Lvv, Wooov, Wovov, Wovvo, Woooo, Wovoo) defined in
 * pyscf/cc/rintermediates.py. We build the IP versions from scratch
 * following PySCF, Eqs (8)-(9) of Nooijen & Snijders 1995.
 *
 * The matvec implemented here is the closed-shell spin-adapted IP-EOM-CCSD
 * sigma from PySCF eom_rccsd.py `ipccsd_matvec` (CCSD partition, the
 * non-MBPT branch).
 */

#include "ip_eom_ccsd_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "eri.hpp"   // Phase 0: ERI_RI::mo_eri_block_into (on-the-fly MO-ERI blocks)
#include "multi_gpu_manager.hpp"   // Stage IP-5: multi-GPU σ (MultiGpuManager/DeviceGuard)

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace gansu {

// MO ERI block extraction kernels are shared with EE-EOM-CCSD
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_ccsd_extract_eri_ovvv_kernel(
    const real_t*, real_t*, int, int, int);

#ifndef GANSU_CPU_ONLY
// Diagonal builder for the Davidson preconditioner.
__global__ void ip_eom_build_diagonal_kernel(
    const real_t* __restrict__ eps,
    real_t* __restrict__ D,
    int nocc, int nvir)
{
    const int h_dim   = nocc;
    const int h2p_dim = nocc * nocc * nvir;
    const int total   = h_dim + h2p_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (idx < h_dim) {
        D[idx] = -eps[idx];
    } else {
        int t  = idx - h_dim;
        int a  = t % nvir;
        int t2 = t / nvir;
        int j  = t2 % nocc;
        int i  = t2 / nocc;
        D[idx] = -eps[i] - eps[j] + eps[a + nocc];
    }
}

__global__ void ip_eom_diag_matvec_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = D[idx] * x[idx];
}

__global__ void ip_eom_precondition_kernel(
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
//  IP-EOM-CCSD sigma kernels (full PySCF ipccsd_matvec)
// ==================================================================

// σ1[i] block (1h-1h + 1h-2h1p)
//   σ1[i] = -Σ_k Loo[k,i] r1[k]
//         + 2 Σ_{l,d} Fov[l,d] r2[i,l,d]
//         -   Σ_{k,d} Fov[k,d] r2[k,i,d]
//         - 2 Σ_{k,l,d} Wooov[k,l,i,d] r2[k,l,d]
//         +   Σ_{k,l,d} Wooov[l,k,i,d] r2[k,l,d]
// r2 layout: idx = (k*nocc + l)*nvir + d
__global__ void ip_eom_sigma1_full_kernel(
    const real_t* __restrict__ Loo,    // [nocc²]
    const real_t* __restrict__ Fov,    // [nocc · nvir]
    const real_t* __restrict__ Wooov,  // [nocc² · nocc · nvir]
    const real_t* __restrict__ r1,     // [nocc]
    const real_t* __restrict__ r2,     // [nocc² · nvir]
    real_t* __restrict__ sigma1,       // [nocc]
    int nocc, int nvir)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nocc) return;

    real_t s = 0.0;
    // -Σ_k Loo[k,i] r1[k]
    for (int k = 0; k < nocc; ++k) s -= Loo[k * nocc + i] * r1[k];

    // ± Σ Fov · r2
    for (int l = 0; l < nocc; ++l)
        for (int d = 0; d < nvir; ++d) {
            real_t fov_ld = Fov[l * nvir + d];
            // +2 Fov[l,d] r2[i,l,d]
            s += 2.0 * fov_ld * r2[((size_t)i * nocc + l) * nvir + d];
            // -Fov[l,d] r2[l,i,d]  (using l instead of k to make notation consistent)
            s -=       fov_ld * r2[((size_t)l * nocc + i) * nvir + d];
        }

    // -2 Wooov[k,l,i,d] r2[k,l,d] + Wooov[l,k,i,d] r2[k,l,d]
    for (int k = 0; k < nocc; ++k)
        for (int l = 0; l < nocc; ++l)
            for (int d = 0; d < nvir; ++d) {
                real_t w1 = Wooov[((size_t)k * nocc + l) * nocc * nvir + (size_t)i * nvir + d];
                real_t w2 = Wooov[((size_t)l * nocc + k) * nocc * nvir + (size_t)i * nvir + d];
                real_t r2_kld = r2[((size_t)k * nocc + l) * nvir + d];
                s += (-2.0 * w1 + w2) * r2_kld;
            }
    sigma1[i] = s;
}


// σ2[i,j,a] block (2h1p-1h + 2h1p-2h1p)
//   σ2[i,j,a] = -Σ_k Wovoo[k,a,i,j] r1[k]                       (1h→2h1p coupling)
//             +  Σ_d Lvv[a,d] r2[i,j,d]
//             -  Σ_k Loo[k,i] r2[k,j,a]
//             -  Σ_l Loo[l,j] r2[i,l,a]
//             +  Σ_{k,l} Woooo[k,l,i,j] r2[k,l,a]
//             +2 Σ_{l,d} Wovvo[l,a,d,j] r2[i,l,d]
//             -  Σ_{k,d} Wovvo[k,a,d,j] r2[k,i,d]
//             -  Σ_{l,d} Wovov[l,a,j,d] r2[i,l,d]
//             -  Σ_{k,d} Wovov[k,a,i,d] r2[k,j,d]
//             -  Σ_c [Σ_{kld}(2 Woovv[l,k,d,c] - Woovv[k,l,d,c]) r2[k,l,d]] * t2[i,j,c,a]
//   Woovv[i,j,a,b] = (ij|ab) = eri_oovv[i,j,a,b]
//
// To avoid recomputing the tmp[c] = Σ ... inside every (i,j,a) sub-call,
// we pre-compute tmp on the host side and pass it as a small [nvir] buffer.
// (At this scale tmp is cheap to compute on-device too; we expose it via a
//  separate kernel and a pre-stage that fills d_tmp_c before the main kernel.)
__global__ void ip_eom_sigma2_full_kernel(
    const real_t* __restrict__ Loo,        // [nocc²]
    const real_t* __restrict__ Lvv,        // [nvir²]
    const real_t* __restrict__ Woooo,      // [nocc^4]
    const real_t* __restrict__ Wovov,      // [nocc · nvir · nocc · nvir]
    const real_t* __restrict__ Wovvo,      // [nocc · nvir · nvir · nocc]
    const real_t* __restrict__ Wovoo,      // [nocc · nvir · nocc²]
    const real_t* __restrict__ d_tmp_c,    // [nvir] precomputed: tmp[c] = (2 Woovv - Woovv^T) · r2
    const real_t* __restrict__ t2,         // [nocc² · nvir²]
    const real_t* __restrict__ r1,         // [nocc]
    const real_t* __restrict__ r2,         // [nocc² · nvir]
    real_t* __restrict__ sigma2,           // [nocc² · nvir]
    int nocc, int nvir,
    int i_begin, int i_end,                // Stage IP-5: compute only this outer-occ slab
    bool add_big)                          // false ⇒ Woooo/Wovvo/Wovov terms done by cuBLAS GEMMs
{
    int lidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int slabtot = (i_end - i_begin) * nocc * nvir;
    if (lidx >= slabtot) return;
    int a = lidx % nvir;
    int t = lidx / nvir;
    int j = t % nocc;
    int i = i_begin + t / nocc;            // global outer-occ index

    real_t s = 0.0;

    // -Σ_k Wovoo[k,a,i,j] r1[k]                  layout: (k*nvir + a)*nocc² + (i*nocc + j)
    for (int k = 0; k < nocc; ++k) {
        s -= Wovoo[((size_t)k * nvir + a) * nocc * nocc + (size_t)i * nocc + j] * r1[k];
    }

    // +Lvv[a,d] r2[i,j,d]
    for (int d = 0; d < nvir; ++d) {
        s += Lvv[a * nvir + d] * r2[((size_t)i * nocc + j) * nvir + d];
    }

    // -Loo[k,i] r2[k,j,a]
    for (int k = 0; k < nocc; ++k) {
        s -= Loo[k * nocc + i] * r2[((size_t)k * nocc + j) * nvir + a];
    }
    // -Loo[l,j] r2[i,l,a]
    for (int l = 0; l < nocc; ++l) {
        s -= Loo[l * nocc + j] * r2[((size_t)i * nocc + l) * nvir + a];
    }

    // The three big terms below (Woooo·r2, Wovvo·r2, Wovov·r2) are O(nocc²)/O(nocc·nvir)
    // per output element and read the nocc⁴ / nocc²·nvir² intermediates with poor reuse
    // (Wovvo/Wovov are UNCOALESCED: their index depends on a, so a warp's consecutive-a
    // threads touch elements nvir·nocc apart). This is the memory-bound bulk of the kernel.
    // When add_big is false they are all computed instead as dense cuBLAS GEMMs.
    if (add_big) {
    // +Σ_{k,l} Woooo[k,l,i,j] r2[k,l,a]      layout: ((k*nocc+l)*nocc + i)*nocc + j
    for (int k = 0; k < nocc; ++k)
        for (int l = 0; l < nocc; ++l) {
            s += Woooo[(((size_t)k * nocc + l) * nocc + i) * nocc + j]
                 * r2[((size_t)k * nocc + l) * nvir + a];
        }

    // +2 Σ_{l,d} Wovvo[l,a,d,j] r2[i,l,d]     layout: ((l*nvir+a)*nvir+d)*nocc + j
    // -  Σ_{k,d} Wovvo[k,a,d,j] r2[k,i,d]     (same Wovvo, different r2 slice)
    for (int k_or_l = 0; k_or_l < nocc; ++k_or_l)
        for (int d = 0; d < nvir; ++d) {
            real_t wovvo = Wovvo[(((size_t)k_or_l * nvir + a) * nvir + d) * nocc + j];
            // +2 from l-summed
            s += 2.0 * wovvo * r2[((size_t)i * nocc + k_or_l) * nvir + d];
            // -1 from k-summed
            s -=       wovvo * r2[((size_t)k_or_l * nocc + i) * nvir + d];
        }

    // -Σ_{l,d} Wovov[l,a,j,d] r2[i,l,d]       layout: ((l*nvir+a)*nocc + j)*nvir + d
    // -Σ_{k,d} Wovov[k,a,i,d] r2[k,j,d]       (different j position)
    for (int k_or_l = 0; k_or_l < nocc; ++k_or_l)
        for (int d = 0; d < nvir; ++d) {
            // Wovov[l,a,j,d] r2[i,l,d]
            real_t w1 = Wovov[(((size_t)k_or_l * nvir + a) * nocc + j) * nvir + d];
            s -= w1 * r2[((size_t)i * nocc + k_or_l) * nvir + d];
            // Wovov[k,a,i,d] r2[k,j,d]
            real_t w2 = Wovov[(((size_t)k_or_l * nvir + a) * nocc + i) * nvir + d];
            s -= w2 * r2[((size_t)k_or_l * nocc + j) * nvir + d];
        }
    }  // add_big

    // -Σ_c d_tmp_c[c] * t2[i,j,c,a]
    for (int c = 0; c < nvir; ++c) {
        s -= d_tmp_c[c] * t2[(((size_t)i * nocc + j) * nvir + c) * nvir + a];
    }

    sigma2[((size_t)i * nocc + j) * nvir + a] = s;   // global index (slab writes its rows)
}

// ------------------------------------------------------------------------------
// (perf) GEMM-path helpers for the σ2 Wovvo/Wovov terms (the uncoalesced, wall-time-
// dominant part of the matvec).  We recast them as dense cuBLAS GEMMs.  One-time
// index repacks turn the intermediates into contraction-major matrices [(l,d),(a,j)];
// per-matvec we build reshaped r2 views, GEMM, and transpose-accumulate into σ2.
// ------------------------------------------------------------------------------

// One-time: WA[(l,d),(a,j)] = Wovvo(l,a,d,j).  Output linear = ((l*nvir+d)*nvir+a)*nocc+j.
__global__ void ip_repack_wovvo_kernel(const real_t* __restrict__ Wovvo,
                                       real_t* __restrict__ WA, int nocc, int nvir) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tot = (size_t)nocc * nvir * nvir * nocc;
    if (idx >= tot) return;
    int j = idx % nocc; size_t t = idx / nocc;
    int a = t % nvir;   t /= nvir;
    int d = t % nvir;   t /= nvir;
    int l = (int)t;
    WA[idx] = Wovvo[(((size_t)l * nvir + a) * nvir + d) * nocc + j];
}

// One-time: WVA[(l,d),(a,j)] = Wovov(l,a,j,d).  Same output layout as WA.
__global__ void ip_repack_wovov_kernel(const real_t* __restrict__ Wovov,
                                       real_t* __restrict__ WVA, int nocc, int nvir) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tot = (size_t)nocc * nvir * nvir * nocc;
    if (idx >= tot) return;
    int j = idx % nocc; size_t t = idx / nocc;
    int a = t % nvir;   t /= nvir;
    int d = t % nvir;   t /= nvir;
    int l = (int)t;
    WVA[idx] = Wovov[(((size_t)l * nvir + a) * nocc + j) * nvir + d];
}

// Per-matvec: build R2comb[x,(l,d)] = 2·r2(x,l,d) − r2(l,x,d) and R2sw[x,(l,d)] = r2(l,x,d).
// (R2b = r2 directly.)  Linear index over [x,(l,d)] = (x*nocc+l)*nvir+d = r2 layout.
__global__ void ip_build_r2_views_kernel(const real_t* __restrict__ r2,
                                         real_t* __restrict__ R2comb,
                                         real_t* __restrict__ R2sw, int nocc, int nvir) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tot = (size_t)nocc * nocc * nvir;
    if (idx >= tot) return;
    int d = idx % nvir; size_t t = idx / nvir;
    int l = t % nocc;   int x = (int)(t / nocc);
    const real_t r_xl = r2[((size_t)x * nocc + l) * nvir + d];   // r2(x,l,d)
    const real_t r_lx = r2[((size_t)l * nocc + x) * nvir + d];   // r2(l,x,d)
    R2sw[idx]   = r_lx;
    R2comb[idx] = 2.0 * r_xl - r_lx;
}

// Per-matvec: σ2[(i,j),a] += out1[i,(a,j)] + out2[j,(a,i)].
//   out1 linear = (i*nvir+a)*nocc+j ;  out2 linear = (j*nvir+a)*nocc+i.
__global__ void ip_sigma2_transpose_add_kernel(real_t* __restrict__ sigma2,
                                               const real_t* __restrict__ out1,
                                               const real_t* __restrict__ out2,
                                               int nocc, int nvir) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tot = (size_t)nocc * nocc * nvir;
    if (idx >= tot) return;
    int a = idx % nvir; size_t t = idx / nvir;
    int j = t % nocc;   int i = (int)(t / nocc);
    sigma2[idx] += out1[((size_t)i * nvir + a) * nocc + j]
                 + out2[((size_t)j * nvir + a) * nocc + i];
}


// Pre-stage kernel: compute tmp[c] = Σ_{k,l,d} (2 Woovv[l,k,d,c] - Woovv[k,l,d,c]) · r2[k,l,d]
__global__ void ip_eom_sigma2_tmp_c_kernel(
    const real_t* __restrict__ Woovv,   // [nocc²·nvir²]
    const real_t* __restrict__ r2,      // [nocc²·nvir]
    real_t* __restrict__ tmp_c,         // [nvir]
    int nocc, int nvir)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nvir) return;
    real_t s = 0.0;
    for (int k = 0; k < nocc; ++k)
        for (int l = 0; l < nocc; ++l)
            for (int d = 0; d < nvir; ++d) {
                real_t w_lkdc = Woovv[(((size_t)l * nocc + k) * nvir + d) * nvir + c];
                real_t w_kldc = Woovv[(((size_t)k * nocc + l) * nvir + d) * nvir + c];
                s += (2.0 * w_lkdc - w_kldc) * r2[((size_t)k * nocc + l) * nvir + d];
            }
    tmp_c[c] = s;
}
#endif  // !GANSU_CPU_ONLY


// ==================================================================
//  Constructor / destructor
// ==================================================================

IPEOMCCSDOperator::IPEOMCCSDOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    int nocc, int nvir, int nao,
    const ERI_RI* eri_block_src,
    const real_t* d_B_mo_blocks,
    int nmo_full,
    int num_gpus,
    SteomBarHCache* barh_cache,
    int frozen_off)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      h_dim_(nocc),
      h2p_dim_(nocc * nocc * nvir),
      total_dim_(nocc + nocc * nocc * nvir),
      d_t1_(d_t1), d_t2_(d_t2),
      eri_block_src_(eri_block_src), d_B_mo_blocks_(d_B_mo_blocks), nmo_full_(nmo_full),
      frozen_off_(frozen_off),
      barh_cache_(barh_cache)
{
    if (nocc <= 0 || nvir <= 0 || nao != nocc + nvir) {
        throw std::invalid_argument(
            "IPEOMCCSDOperator: invalid (nocc, nvir, nao) — require nao == nocc + nvir, both positive");
    }

    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
    if (d_eri_mo != nullptr || eri_block_src_ != nullptr) {  // Phase 0: block source
        // Per-phase build profiling.  Default ON so each phase prints START
        // and END markers — the canonical IP-EOM operator build is a long
        // silent stretch otherwise (anthracene ~60 s, tetracene ~106 s, no
        // Davidson activity to fall back on).  Set GANSU_EOM_BUILD_PROF=0
        // to silence both markers.
        const char* env_prof = std::getenv("GANSU_EOM_BUILD_PROF");
        const bool prof = !env_prof || env_prof[0] != '0';
        auto tphase = [&](const char* name, auto&& fn) {
            if (!prof) { fn(); return; }
            std::cout << "  [IP-EOM build-PROF] " << name << " ..." << std::endl;
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
            std::cout << "  [IP-EOM build-PROF] " << name << " = " << std::fixed
                      << std::setprecision(3) << s << " s" << std::defaultfloat << std::endl;
        };
        tphase("extract_eri_blocks",          [&]{ extract_eri_blocks(d_eri_mo); });
        tphase("build_dressed_intermediates", [&]{ build_dressed_intermediates(); });
        // (A) shared bar-H: publish the 8 IP-side intermediates and relinquish
        // ownership (dtor will skip freeing them). EA later adds Wvovv/Wvvvv/
        // Wvvvo; STEOM borrows the union. All bit-identical across operators.
        if (barh_cache_ != nullptr) {
            barh_cache_->d_Loo   = d_Loo_;
            barh_cache_->d_Lvv   = d_Lvv_;
            barh_cache_->d_Fov   = d_Fov_;
            barh_cache_->d_Woooo = d_Woooo_;
            barh_cache_->d_Wooov = d_Wooov_;
            barh_cache_->d_Wovov = d_Wovov_;
            barh_cache_->d_Wovvo = d_Wovvo_;
            barh_cache_->d_Wovoo = d_Wovoo_;
            barh_cache_->nocc    = nocc_;
            barh_cache_->nvir    = nvir_;
            barh_cache_->has_ip  = true;
            barh_published_      = true;
            std::cout << "  [STEOM share-barH] IP published 8 bar-H intermediates "
                         "(Loo/Lvv/Fov/Woooo/Wooov/Wovov/Wovvo/Wovoo)." << std::endl;
        }
        num_gpus_ = (num_gpus > 1 ? num_gpus : 1);
        setup_multi_gpu();   // Stage IP-5: per-device replicas (no-op when num_gpus_==1)
    }
    // (perf) Opt-in: cast the memory-bound big σ2 terms (Woooo/Wovvo/Wovov) as
    // device-0 cuBLAS GEMMs instead of the uncoalesced one-thread-per-output kernel.
    sigma_gemm_ = (std::getenv("GANSU_IP_SIGMA_GEMM") != nullptr) && gpu::gpu_available();
}

IPEOMCCSDOperator::~IPEOMCCSDOperator() {
    // (perf) GEMM σ2 path scratch (device 0).
    if (d_WA_)     tracked_cudaFree(d_WA_);
    if (d_WVA_)    tracked_cudaFree(d_WVA_);
    if (d_R2comb_) tracked_cudaFree(d_R2comb_);
    if (d_R2sw_)   tracked_cudaFree(d_R2sw_);
    if (d_out1_)   tracked_cudaFree(d_out1_);
    if (d_out2_)   tracked_cudaFree(d_out2_);
    if (d_t1_)        tracked_cudaFree(d_t1_);
    if (d_t2_)        tracked_cudaFree(d_t2_);
    if (d_D_h_)       tracked_cudaFree(d_D_h_);
    if (d_D_h2p_)     tracked_cudaFree(d_D_h2p_);
    if (d_f_oo_)      tracked_cudaFree(d_f_oo_);
    if (d_f_vv_)      tracked_cudaFree(d_f_vv_);
    if (d_diagonal_)  tracked_cudaFree(d_diagonal_);
    if (d_eri_oooo_)  tracked_cudaFree(d_eri_oooo_);
    if (d_eri_oovv_)  tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovov_)  tracked_cudaFree(d_eri_ovov_);
    if (d_eri_ovvo_)  tracked_cudaFree(d_eri_ovvo_);
    if (d_eri_ooov_)  tracked_cudaFree(d_eri_ooov_);
    if (d_eri_ovvv_)  tracked_cudaFree(d_eri_ovvv_);
    // (A) shared bar-H: when published, the cache owns these 8 device buffers
    // (freed by the STEOM driver after STEOM borrows them). Skip here to avoid
    // a double-free / use-after-free. The per-device replicas (d≥1) below are
    // SEPARATE allocations and are always freed.
    if (!barh_published_) {
        if (d_Loo_)       tracked_cudaFree(d_Loo_);
        if (d_Lvv_)       tracked_cudaFree(d_Lvv_);
        if (d_Fov_)       tracked_cudaFree(d_Fov_);
        if (d_Woooo_)     tracked_cudaFree(d_Woooo_);
        if (d_Wooov_)     tracked_cudaFree(d_Wooov_);
        if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
        if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
        if (d_Wovoo_)     tracked_cudaFree(d_Wovoo_);
    }
#ifndef GANSU_CPU_ONLY
    // Stage IP-5: free per-device replicas + scratch (ws_[0] only owns scratch;
    // its replica pointers alias the device-0 members freed above).
    for (size_t d = 1; d < ws_.size(); ++d) {
        DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(w.device);
        for (real_t* p : {w.d_input, w.d_s1, w.d_s2, w.d_tmp_c,
                          w.d_Loo, w.d_Lvv, w.d_Fov, w.d_Woooo, w.d_Wooov,
                          w.d_Wovov, w.d_Wovvo, w.d_Wovoo, w.d_eri_oovv, w.d_t2})
            if (p) tracked_cudaFree(p);
    }
    if (!ws_.empty()) {
        MultiGpuManager::DeviceGuard guard(0);
        if (ws_[0].d_input)  tracked_cudaFree(ws_[0].d_input);
        if (ws_[0].d_s1)     tracked_cudaFree(ws_[0].d_s1);
        if (ws_[0].d_s2)     tracked_cudaFree(ws_[0].d_s2);
        if (ws_[0].d_tmp_c)  tracked_cudaFree(ws_[0].d_tmp_c);
    }
#endif
}

// ==================================================================
//  Stage IP-5: multi-GPU σ (mirrors validated EA-EOM 5a-5d).  i-slab of the
//  σ2 sector across N physical GPUs (decoupled from RI num_gpus via
//  GANSU_STEOM_EOM_GPUS).  IP apply() has no GEMM → no per-device cuBLAS.
// ==================================================================
void IPEOMCCSDOperator::setup_multi_gpu() {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || num_gpus_ <= 1) return;
    const char* e = std::getenv("GANSU_STEOM_EOM_GPUS");
    const int env_gpus = (e && e[0]) ? std::atoi(e) : 0;
    if (env_gpus <= 1) return;
    int phys = 0;
    cudaGetDeviceCount(&phys);
    const int nuse = std::min({num_gpus_, env_gpus, phys});
    if (nuse <= 1) return;
    use_gpu_multi_ = true;

    const size_t NO = nocc_, NV = nvir_;
    const size_t loo_sz = NO*NO, lvv_sz = NV*NV, fov_sz = NO*NV;
    const size_t oooo_sz = NO*NO*NO*NO, ooov_sz = NO*NO*NO*NV;
    const size_t ovov_sz = NO*NV*NO*NV, ovvo_sz = NO*NV*NV*NO;
    const size_t wovoo_sz = NO*NV*NO*NO, oovv_sz = NO*NO*NV*NV, t2_sz = NO*NO*NV*NV;
    const size_t h2p = (size_t)h2p_dim_, total = (size_t)total_dim_;

    ws_.resize(nuse);
    {   DeviceWorkspace& w = ws_[0];
        w.device = 0;
        w.d_Loo=d_Loo_; w.d_Lvv=d_Lvv_; w.d_Fov=d_Fov_; w.d_Woooo=d_Woooo_;
        w.d_Wooov=d_Wooov_; w.d_Wovov=d_Wovov_; w.d_Wovvo=d_Wovvo_; w.d_Wovoo=d_Wovoo_;
        w.d_eri_oovv=d_eri_oovv_; w.d_t2=d_t2_;
        auto p0 = aux_partition(NO, nuse, 0);
        w.i_begin=(int)p0.first; w.i_end=(int)p0.second;
        tracked_cudaMalloc(&w.d_tmp_c, NV * sizeof(real_t));   // device-0 σ scratch
    }
    size_t bytes_per_dev = 0;
    for (int d = 1; d < nuse; ++d) {
        DeviceWorkspace& w = ws_[d];
        w.device = d;
        MultiGpuManager::DeviceGuard guard(d);
        auto pd = aux_partition(NO, nuse, d);
        w.i_begin=(int)pd.first; w.i_end=(int)pd.second;
        auto repl = [&](real_t** dst, const real_t* src0, size_t n) {
            tracked_cudaMalloc(dst, n * sizeof(real_t));
            cudaMemcpyPeer(*dst, d, src0, 0, n * sizeof(real_t));
            bytes_per_dev += n * sizeof(real_t);
        };
        repl(&w.d_Loo,   d_Loo_,   loo_sz);
        repl(&w.d_Lvv,   d_Lvv_,   lvv_sz);
        repl(&w.d_Fov,   d_Fov_,   fov_sz);
        repl(&w.d_Woooo, d_Woooo_, oooo_sz);
        repl(&w.d_Wooov, d_Wooov_, ooov_sz);
        if (d_Wovov_) repl(&w.d_Wovov, d_Wovov_, ovov_sz); else w.d_Wovov = nullptr;
        if (d_Wovvo_) repl(&w.d_Wovvo, d_Wovvo_, ovvo_sz); else w.d_Wovvo = nullptr;
        repl(&w.d_Wovoo, d_Wovoo_, wovoo_sz);
        repl(&w.d_eri_oovv, d_eri_oovv_, oovv_sz);
        repl(&w.d_t2,    d_t2_,    t2_sz);
        tracked_cudaMalloc(&w.d_input, total * sizeof(real_t));
        tracked_cudaMalloc(&w.d_s1,    (size_t)h_dim_ * sizeof(real_t));
        tracked_cudaMalloc(&w.d_s2,    h2p * sizeof(real_t));
        tracked_cudaMalloc(&w.d_tmp_c, NV * sizeof(real_t));
        bytes_per_dev += (total + h_dim_ + h2p + NV) * sizeof(real_t);
    }
    MultiGpuManager::DeviceGuard guard0(0);
    const double gb = (nuse > 1) ? (double)bytes_per_dev/(double)(nuse-1)/(1024.0*1024.0*1024.0) : 0.0;
    std::cout << "[IP-EOM Stage IP-5] multi-GPU σ: nuse=" << nuse
              << "  i-slab over nocc=" << NO
              << "  per-device replicas ≈ " << std::fixed << std::setprecision(2) << gb
              << " GB" << std::defaultfloat
              << " (each device computes its σ2 i-slab → disjoint gather)" << std::endl;
#endif
}

void IPEOMCCSDOperator::compute_denominators_and_fock(const real_t* d_orbital_energies) {
    tracked_cudaMalloc(&d_D_h_,   (size_t)h_dim_   * sizeof(real_t));
    tracked_cudaMalloc(&d_D_h2p_, (size_t)h2p_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_oo_,  (size_t)nocc_    * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv_,  (size_t)nvir_    * sizeof(real_t));

    if (!gpu::gpu_available()) {
        for (int i = 0; i < nocc_; ++i) d_D_h_[i] = -d_orbital_energies[i];
        #pragma omp parallel for
        for (int idx = 0; idx < h2p_dim_; ++idx) {
            int a = idx % nvir_;
            int t = idx / nvir_;
            int j = t % nocc_;
            int i = t / nocc_;
            d_D_h2p_[idx] = -d_orbital_energies[i] - d_orbital_energies[j]
                          +  d_orbital_energies[a + nocc_];
        }
        for (int i = 0; i < nocc_; ++i) d_f_oo_[i] = d_orbital_energies[i];
        for (int a = 0; a < nvir_; ++a) d_f_vv_[a] = d_orbital_energies[a + nocc_];
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        real_t* d_packed = nullptr;
        tracked_cudaMalloc(&d_packed, (size_t)total_dim_ * sizeof(real_t));
        ip_eom_build_diagonal_kernel<<<blocks, threads>>>(
            d_orbital_energies, d_packed, nocc_, nvir_);
        cudaDeviceSynchronize();
        cudaMemcpy(d_D_h_,   d_packed,           (size_t)h_dim_   * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_D_h2p_, d_packed + h_dim_,  (size_t)h2p_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        tracked_cudaFree(d_packed);
        cudaMemcpy(d_f_oo_, d_orbital_energies,         (size_t)nocc_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_f_vv_, d_orbital_energies + nocc_, (size_t)nvir_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
    }
}

void IPEOMCCSDOperator::build_diagonal() {
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (int i = 0; i < h_dim_;   ++i) d_diagonal_[i]          = d_D_h_[i];
        for (int i = 0; i < h2p_dim_; ++i) d_diagonal_[h_dim_ + i] = d_D_h2p_[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_diagonal_,           d_D_h_,   (size_t)h_dim_   * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_diagonal_ + h_dim_,  d_D_h2p_, (size_t)h2p_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
    }
}


void IPEOMCCSDOperator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_, nvir = nvir_, nao = nao_;
    size_t nao2 = (size_t)nao * nao;
    size_t N    = (size_t)nao;

    const size_t ovov_size = (size_t)nocc * nvir * nocc * nvir;
    const size_t ooov_size = (size_t)nocc * nocc * nocc * nvir;
    const size_t oooo_size = (size_t)nocc * nocc * nocc * nocc;
    const size_t oovv_size = (size_t)nocc * nocc * nvir * nvir;
    const size_t ovvo_size = (size_t)nocc * nvir * nvir * nocc;
    const size_t ovvv_size = (size_t)nocc * nvir * nvir * nvir;

    tracked_cudaMalloc(&d_eri_ovov_, ovov_size * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ooov_, ooov_size * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_oooo_, oooo_size * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_oovv_, oovv_size * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovvo_, ovvo_size * sizeof(real_t));
    // Inc4(b): in block mode every ovvv consumer builds per-k slices from B (the
    // tiled Wovov/Wovvo/Woooo/Lvv paths), so the dense ovvv block (nocc·nvir³ =
    // 46 GB at nvir=334) is never read — skip it unless the host self-check needs
    // it. The other blocks (ovov/oovv/ovvo, ~22 GB) stay dense (not yet tiled).
    const bool need_ovvv = (eri_block_src_ == nullptr)
                           || (std::getenv("GANSU_STEOM_BUILD_VALIDATE") != nullptr);
    if (need_ovvv) tracked_cudaMalloc(&d_eri_ovvv_, ovvv_size * sizeof(real_t));

#ifndef GANSU_CPU_ONLY
    // Phase 0: build the 6 blocks on the fly from B_mo (naux×nmo²), never the
    // full nmo⁴. o=[0,nocc), v=[nocc,nmo). Layouts match the gather kernels below.
    if (eri_block_src_ != nullptr) {
        const int M = nmo_full_;
        // Frozen core: shift every range start by frozen_off_ to read the active
        // window from the full-C B_mo. O = 0 ⇒ non-frozen (byte-identical).
        const int O = frozen_off_;
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,O,nocc,           d_eri_oooo_); // (ij|kl)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,nocc+O,nvir,      d_eri_ooov_); // (ji|kb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    O,nocc,nocc+O,nvir,      d_eri_ovov_); // (ia|jb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         nocc+O,nvir,nocc+O,nvir, d_eri_oovv_); // (ij|ab)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,O,nocc,      d_eri_ovvo_); // (ia|bj)
        if (need_ovvv)
            eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,nocc+O,nvir, d_eri_ovvv_); // (ia|bc)
        cudaDeviceSynchronize();
        return;
    }
#endif

    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovov_size; ++idx) {
            int i = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int a = rem / (nocc * nvir); rem %= (nocc * nvir);
            int j = rem / nvir; int b = rem % nvir;
            d_eri_ovov_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)j*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ooov_size; ++idx) {
            int j = idx / (nocc * nocc * nvir);
            int rem = idx % (nocc * nocc * nvir);
            int i = rem / (nocc * nvir); rem %= (nocc * nvir);
            int k = rem / nvir; int b = rem % nvir;
            d_eri_ooov_[idx] = d_eri_mo[((size_t)j*nao + i)*nao2 + (size_t)k*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oooo_size; ++idx) {
            int i = idx / (nocc * nocc * nocc);
            int rem = idx % (nocc * nocc * nocc);
            int j = rem / (nocc * nocc); rem %= (nocc * nocc);
            int k = rem / nocc; int l = rem % nocc;
            d_eri_oooo_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)k*nao + l];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oovv_size; ++idx) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir); rem %= (nvir * nvir);
            int a = rem / nvir; int b = rem % nvir;
            d_eri_oovv_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovvo_size; ++idx) {
            int i = idx / (nvir * nvir * nocc);
            int rem = idx % (nvir * nvir * nocc);
            int a = rem / (nvir * nocc); rem %= (nvir * nocc);
            int b = rem / nocc; int j = rem % nocc;
            d_eri_ovvo_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
        }
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovvv_size; ++idx) {
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
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        int blocks;
        blocks = (ovov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_, nvir_, nao_);
        blocks = (ooov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_, nvir_, nao_);
        blocks = (oooo_size + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_, nao_, 0);
        blocks = (oovv_size + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_, nvir_, nao_, 0, -1);
        blocks = (ovvo_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_, nvir_, nao_);
        blocks = (ovvv_size + threads - 1) / threads;
        eom_ccsd_extract_eri_ovvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvv_, nocc_, nvir_, nao_);
        cudaDeviceSynchronize();
#endif
    }
}


#ifndef GANSU_CPU_ONLY
// [IP build device-build] Build the Woooo/Wovoo Σcd GEMM inputs directly on
// device from the resident ERI/T2/T1 blocks, eliminating the host hAo/hAv/hB/
// hB2 reorder (~1.1 s) + their H2D. Output layouts match the host build
// exactly, so the downstream cublasDgemm + the Σcd self-check are unchanged.
//   dAo[(k,l),(c,d)] = ovov[k,c,l,d]                       (OO_kl × VV_cd)
//   dAv[(k,b),(c,d)] = ovvv[k,c,b,d]                       (OV_kb × VV_cd)
//   dB [(c,d),(i,j)] = t2[i,j,c,d] + t1[i,c]·t1[j,d]       (VV_cd × OO_ij)
//   dB2[(c,d),(i,j)] = t2[j,i,d,c] + t1[j,d]·t1[i,c]
__global__ void ip_woooo_build_Ao(real_t* __restrict__ dAo,
                                  const real_t* __restrict__ ovov, int NO, int NV) {
    const size_t tot = (size_t)NO*NO*NV*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = x % NV; size_t t = x / NV;
        const int c = t % NV; t /= NV;
        const int l = t % NO; const int k = (int)(t / NO);
        dAo[x] = ovov[(((size_t)k*NV+c)*NO+l)*NV+d];
    }
}
__global__ void ip_woooo_build_Av(real_t* __restrict__ dAv,
                                  const real_t* __restrict__ ovvv, int NO, int NV) {
    const size_t tot = (size_t)NO*NV*NV*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = x % NV; size_t t = x / NV;
        const int c = t % NV; t /= NV;
        const int b = t % NV; const int k = (int)(t / NV);
        dAv[x] = ovvv[(((size_t)k*NV+c)*NV+b)*NV+d];
    }
}
// Inc4(b): combine the two ovvv·t1 partials of bar-Lvv. y1[a,c]=2Σ ovvv[k,d,a,c]t1,
// z[c,a]=Σ ovvv[k,c,a,d]t1 (both accumulated over k by GEMV); Lvv_ovvv[a,c]=y1[a,c]-z[c,a].
__global__ void ip_lvv_ovvv_combine(real_t* __restrict__ lvv,
                                    const real_t* __restrict__ y1,
                                    const real_t* __restrict__ z, int NV) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    if (x >= NV*NV) return;
    int a = x / NV, c = x % NV;
    lvv[(size_t)a*NV+c] = y1[(size_t)a*NV+c] - z[(size_t)c*NV+a];
}
__global__ void ip_woooo_build_B(real_t* __restrict__ dB,
                                 const real_t* __restrict__ t2,
                                 const real_t* __restrict__ t1,
                                 int NO, int NV, bool swap) {
    const size_t tot = (size_t)NV*NV*NO*NO;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int j = x % NO; size_t t = x / NO;
        const int i = t % NO; t /= NO;
        const int d = t % NV; const int c = (int)(t / NV);
        if (!swap)  // dB : t2[i,j,c,d] + t1[i,c]·t1[j,d]
            dB[x] = t2[(((size_t)i*NO+j)*NV+c)*NV+d] + t1[i*NV+c]*t1[j*NV+d];
        else        // dB2: t2[j,i,d,c] + t1[j,d]·t1[i,c]
            dB[x] = t2[(((size_t)j*NO+i)*NV+d)*NV+c] + t1[j*NV+d]*t1[i*NV+c];
    }
}
// [IP build device-assembly] Assemble d_Woooo_[k,l,i,j] directly on device from
// the resident oooo + the device-resident Σcd (ct) and ooov·t1 (ct1) buffers,
// eliminating the NO⁴ host scatter loop + the final H2D upload.
//   Woooo[k,l,i,j] = oooo[k,i,l,j] + ct1[(k,i,l),j] + ct1[(l,j,k),i] + ct[(k,l),(i,j)]
// ct layout = [(k,l),(i,j)] = the Woooo flat index x; ct1 = [(p,q,r),f].
__global__ void ip_woooo_assemble(real_t* __restrict__ dW,
                                  const real_t* __restrict__ oooo,
                                  const real_t* __restrict__ ct1,
                                  const real_t* __restrict__ ct, int NO) {
    const size_t tot = (size_t)NO*NO*NO*NO;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int j = x % NO; size_t t = x / NO;
        const int i = t % NO; t /= NO;
        const int l = t % NO; const int k = (int)(t / NO);
        dW[x] = oooo[(((size_t)k*NO+i)*NO+l)*NO+j]
              + ct1[(((size_t)k*NO+i)*NO+l)*NO+j]
              + ct1[(((size_t)l*NO+j)*NO+k)*NO+i]
              + ct[x];
    }
}
// [IP build device-build] GEMM-input builders for the two O(NO³·NV³) ct_ovov /
// ct_ovvo contractions — fill dA/dB/dA1/dB1/dA2/dB2 directly on device from the
// resident d_eri_ovov_ / d_t2_, eliminating the 6×(NO·NV)² host arrays + the
// ~1.35 GB H2D upload (host omp-fill + memcpy was the build_dressed hotspot).
// ovov[k,c,l,d] = ovov[((k*NV+c)*NO+l)*NV+d];  t2[i,j,c,d] = t2[((i*NO+j)*NV+c)*NV+d]
__global__ void ip_ctovov_buildA(real_t* __restrict__ dA,
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
__global__ void ip_ctovov_buildB(real_t* __restrict__ dB,
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
__global__ void ip_ctovvo_buildA12(real_t* __restrict__ dA1, real_t* __restrict__ dA2,
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
__global__ void ip_ctovvo_buildB12(real_t* __restrict__ dB1, real_t* __restrict__ dB2,
                                   const real_t* __restrict__ t2, int NO, int NV) {
    // dB1[(l,d),(Y,X)] = t2[Y,l,X,d];  dB2 = t2[l,Y,X,d];  enumerate (l,d,Y,X)
    const size_t tot = (size_t)NO*NV*NO*NV, Nyx = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int X = x % NV; size_t t = x / NV;
        const int Y = t % NO; t /= NO;
        const int d = t % NV; const int l = (int)(t / NV);
        const size_t o = ((size_t)(l*NV+d))*Nyx + (Y*NV+X);
        dB1[o] = t2[(((size_t)Y*NO+l)*NV+X)*NV+d];
        dB2[o] = t2[(((size_t)l*NO+Y)*NV+X)*NV+d];
    }
}
// [IP build device-build] Wovoo Σ_ld ooov·t2 GEMM-input builders — fill the 4+2
// host arrays (hA13/hA2/hBlj/hB2 for ct_t2, hA/hB for ct_6, ~800 MB total +
// H2D) on device from the resident d_eri_ooov_ / d_t2_.
//   ooov[k,i,l,d] = ooov[((k*NO+i)*NO+l)*NV+d];  t2[i,j,c,d] = t2[((i*NO+j)*NV+c)*NV+d]
__global__ void ip_wovoo_buildA13_2(real_t* __restrict__ dA13, real_t* __restrict__ dA2,
                                    const real_t* __restrict__ ooov, int NO, int NV) {
    // dA13[(k,i),(l,d)] = 2·ooov[k,i,l,d] − ooov[l,i,k,d];  dA2 = ooov[k,i,l,d];  enumerate (k,i,l,d)
    const size_t tot = (size_t)NO*NO*NO*NV, LDk = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int d = x % NV; size_t t = x / NV;
        const int l = t % NO; t /= NO;
        const int i = t % NO; const int k = (int)(t / NO);
        const real_t kild = ooov[(((size_t)k*NO+i)*NO+l)*NV+d];
        const real_t likd = ooov[(((size_t)l*NO+i)*NO+k)*NV+d];
        const size_t o = ((size_t)(k*NO+i))*LDk + (l*NV+d);
        dA13[o] = 2.0*kild - likd;  dA2[o] = kild;
    }
}
__global__ void ip_wovoo_buildB_2(real_t* __restrict__ dBlj, real_t* __restrict__ dB2,
                                  const real_t* __restrict__ t2, int NO, int NV) {
    // dBlj[(l,d),(j,b)] = t2[l,j,d,b];  dB2 = t2[j,l,d,b];  enumerate (l,d,j,b)
    const size_t tot = (size_t)NO*NV*NO*NV, JBn = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int b = x % NV; size_t t = x / NV;
        const int j = t % NO; t /= NO;
        const int d = t % NV; const int l = (int)(t / NV);
        const size_t o = ((size_t)(l*NV+d))*JBn + (j*NV+b);
        dBlj[o] = t2[(((size_t)l*NO+j)*NV+d)*NV+b];
        dB2[o]  = t2[(((size_t)j*NO+l)*NV+d)*NV+b];
    }
}
__global__ void ip_wovoo6_buildA(real_t* __restrict__ dA,
                                 const real_t* __restrict__ ooov, int NO, int NV) {
    // dA[(k,j),(l,c)] = ooov[l,j,k,c];  enumerate (k,j,l,c)
    const size_t tot = (size_t)NO*NO*NO*NV, Kw6 = (size_t)NO*NV;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int c = x % NV; size_t t = x / NV;
        const int l = t % NO; t /= NO;
        const int j = t % NO; const int k = (int)(t / NO);
        dA[((size_t)(k*NO+j))*Kw6 + (l*NV+c)] = ooov[(((size_t)l*NO+j)*NO+k)*NV+c];
    }
}
__global__ void ip_wovoo6_buildB(real_t* __restrict__ dB,
                                 const real_t* __restrict__ t2, int NO, int NV) {
    // dB[(l,c),(b,i)] = t2[l,i,b,c];  enumerate (l,c,b,i)
    const size_t tot = (size_t)NO*NV*NV*NO, Nw6 = (size_t)NV*NO;
    for (size_t x = (size_t)blockIdx.x*blockDim.x + threadIdx.x; x < tot;
         x += (size_t)gridDim.x*blockDim.x) {
        const int i = x % NO; size_t t = x / NO;
        const int b = t % NV; t /= NV;
        const int c = t % NV; const int l = (int)(t / NV);
        dB[((size_t)(l*NV+c))*Nw6 + (b*NO+i)] = t2[(((size_t)l*NO+i)*NV+b)*NV+c];
    }
}
// [IP build device-build] cc_Foo / cc_Fvv GEMM-input builders — fill hAfoo/hBfoo
// (Foo) and hAfvv/hBfvv (Fvv) on device from the resident d_eri_ovov_ / d_t2_ /
// d_t1_, eliminating the 4×(NO·NV)² host arrays (~900 MB) + H2D (F+L hotspot).
//   A[x,(l,c,d)] = 2·ovov[x,c,l,d] − ovov[x,d,l,c];  B[x,(l,c,d)] = t2[x,l,c,d] + t1[x,c]·t1[l,d]   (Foo, x=k/i ∈ NO)
//   A[x,(k,l,d)] = 2·ovov[k,x,l,d] − ovov[k,d,l,x];  B[x,(k,l,d)] = t2[k,l,x,d] + t1[k,x]·t1[l,d]   (Fvv, x=c/a ∈ NV)
__global__ void ip_foo_buildAB(real_t* __restrict__ dA, real_t* __restrict__ dB,
                               const real_t* __restrict__ ovov, const real_t* __restrict__ t2,
                               const real_t* __restrict__ t1, int NO, int NV) {
    const size_t Mfoo = (size_t)NO*NV*NV, tot = (size_t)NO*Mfoo;
    for (size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x; idx < tot;
         idx += (size_t)gridDim.x*blockDim.x) {
        const int x = (int)(idx / Mfoo); const size_t m = idx % Mfoo;
        const int d = (int)(m % NV); size_t mm = m / NV;
        const int c = (int)(mm % NV); const int l = (int)(mm / NV);
        const real_t xcld = ovov[(((size_t)x*NV+c)*NO+l)*NV+d];
        const real_t xdlc = ovov[(((size_t)x*NV+d)*NO+l)*NV+c];
        dA[idx] = 2.0*xcld - xdlc;
        dB[idx] = t2[(((size_t)x*NO+l)*NV+c)*NV+d] + t1[(size_t)x*NV+c]*t1[(size_t)l*NV+d];
    }
}
__global__ void ip_fvv_buildAB(real_t* __restrict__ dA, real_t* __restrict__ dB,
                               const real_t* __restrict__ ovov, const real_t* __restrict__ t2,
                               const real_t* __restrict__ t1, int NO, int NV) {
    const size_t Mfvv = (size_t)NO*NO*NV, tot = (size_t)NV*Mfvv;
    for (size_t idx = (size_t)blockIdx.x*blockDim.x + threadIdx.x; idx < tot;
         idx += (size_t)gridDim.x*blockDim.x) {
        const int x = (int)(idx / Mfvv); const size_t m = idx % Mfvv;
        const int d = (int)(m % NV); size_t mm = m / NV;
        const int l = (int)(mm % NO); const int k = (int)(mm / NO);
        const real_t kxld = ovov[(((size_t)k*NV+x)*NO+l)*NV+d];
        const real_t kdlx = ovov[(((size_t)k*NV+d)*NO+l)*NV+x];
        dA[idx] = 2.0*kxld - kdlx;
        dB[idx] = t2[(((size_t)k*NO+l)*NV+x)*NV+d] + t1[(size_t)k*NV+x]*t1[(size_t)l*NV+d];
    }
}
#endif // !GANSU_CPU_ONLY

// ==================================================================
//  build_dressed_intermediates — PySCF IP-EOM-CCSD versions
// ==================================================================
// Host-side index macros. (NO=nocc, NV=nvir)
#define H_OVOV(p,a,q,b) h_ovov[(size_t)(p)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(q)*NV + (b)]
#define H_OOOV(p,q,r,a) h_ooov[(size_t)(p)*NO*NO*NV + (size_t)(q)*NO*NV + (size_t)(r)*NV + (a)]
#define H_OOVV(p,q,a,b) h_oovv[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
#define H_OVVO(p,a,b,q) h_ovvo[(size_t)(p)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(b)*NO + (q)]
#define H_OVVV(p,a,b,c) h_ovvv[(size_t)(p)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
#define H_OOOO(p,q,r,s) h_oooo[(size_t)(p)*NO*NO*NO + (size_t)(q)*NO*NO + (size_t)(r)*NO + (s)]
#define H_T1(p,a)       h_t1[(p)*NV + (a)]
#define H_T2(p,q,a,b)   h_t2[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]

void IPEOMCCSDOperator::build_dressed_intermediates() {
    const int NO = nocc_;
    const int NV = nvir_;
    const size_t t1_sz   = (size_t)NO * NV;
    const size_t t2_sz   = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t oovv_sz = (size_t)NO * NO * NV * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;

    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_oooo(oooo_sz);
    // Inc4(b): h_ovvv (nocc·nvir³, 46 GB host) is only materialised when the dense
    // device ovvv exists — i.e. the dense path or the opt-in build self-check. In
    // block mode the ovvv consumers are GPU-tiled, so it stays empty.
    std::vector<real_t> h_ovvv;

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (eri_block_src_ == nullptr || std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
        h_ovvv.resize(ovvv_sz);
        cudaMemcpy(h_ovvv.data(), d_eri_ovvv_, ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(h_oooo.data(), d_eri_oooo_,  oooo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Per-intermediate profiler (env GANSU_EOM_BUILD_PROF2=1) — pinpoints the host-loop
    // hotspot inside build_dressed (GANSU_EOM_BUILD_PROF times only the whole phase). Each
    // blap() prints the wall time since the previous call. Off → zero overhead.
    // Per-sub-step progress through the (minutes-long on large clusters)
    // build_dressed phase, so it shows movement instead of freezing silently.
    // Default ON; quiet the whole progress stream with GANSU_PROGRESS=0.
    const bool bprof2 = [](){ const char* e = std::getenv("GANSU_EOM_BUILD_PROF2");
        const char* p = std::getenv("GANSU_PROGRESS");
        return (e && e[0] == '1') || !p || p[0] != '0'; }();
    auto bp_t0 = std::chrono::high_resolution_clock::now();
    auto blap = [&](const char* name) {
        if (!bprof2) return;
        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "    [IP build-PROF2] " << name << " = " << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(t1 - bp_t0).count() << " s" << std::endl;
        bp_t0 = t1;
    };
    blap("D2H inputs");

    // ============================================================
    //  cc_Fov[k,c] = fov + 2 ovov[k,c,l,d] t1[l,d] - ovov[k,d,l,c] t1[l,d]
    //  (Canonical: fov = 0.)
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

    // GPU GEMM port of cc_Foo Σlcd (NO³·NV²) + cc_Fvv Σkld (NO²·NV³) — the build_dressed
    // hotspot (~45% / 6 s at naphthalene per build-PROF2). Each is a single [N×N] = P·Qᵀ
    // contraction over a 3-index (Foo: m=(l,c,d); Fvv: m=(k,l,d)). The A factor is the
    // (2 ovov - ovovᵀ) combination; the B factor is (t2 + t1·t1). Mirrors the validated
    // Woooo Σcd port below. Host fallback (CPU-only) = the per-element cc_Foo/cc_Fvv loops.
    std::vector<real_t> ct_foo, ct_fvv;
    bool foo_fvv_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        const size_t M_foo = (size_t)NO * NV * NV;   // (l,c,d)
        const size_t M_fvv = (size_t)NO * NO * NV;   // (k,l,d)
        real_t *dAfoo=nullptr,*dBfoo=nullptr,*dCfoo=nullptr,*dAfvv=nullptr,*dBfvv=nullptr,*dCfvv=nullptr;
        tracked_cudaMalloc(&dAfoo,(size_t)NO*M_foo*sizeof(real_t));
        tracked_cudaMalloc(&dBfoo,(size_t)NO*M_foo*sizeof(real_t));
        tracked_cudaMalloc(&dCfoo,(size_t)NO*NO*sizeof(real_t));
        tracked_cudaMalloc(&dAfvv,(size_t)NV*M_fvv*sizeof(real_t));
        tracked_cudaMalloc(&dBfvv,(size_t)NV*M_fvv*sizeof(real_t));
        tracked_cudaMalloc(&dCfvv,(size_t)NV*NV*sizeof(real_t));
        // A[x,(l,c,d)] = 2 ovov[x,c,l,d] − ovov[x,d,l,c];  B[x,(l,c,d)] = t2[x,l,c,d] + t1[x,c] t1[l,d]  (Foo)
        // A[x,(k,l,d)] = 2 ovov[k,x,l,d] − ovov[k,d,l,x];  B[x,(k,l,d)] = t2[k,l,x,d] + t1[k,x] t1[l,d]  (Fvv)
        { const size_t n=(size_t)NO*M_foo; const unsigned NB=(unsigned)((n+255)/256);
          ip_foo_buildAB<<<NB,256>>>(dAfoo, dBfoo, d_eri_ovov_, d_t2_, d_t1_, NO, NV); }
        { const size_t n=(size_t)NV*M_fvv; const unsigned NB=(unsigned)((n+255)/256);
          ip_fvv_buildAB<<<NB,256>>>(dAfvv, dBfvv, d_eri_ovov_, d_t2_, d_t1_, NO, NV); }
        // ct_foo[k,i] = Σ_m A_foo[k,m] B_foo[i,m]  (row-major [NO×NO], elem k*NO+i).
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,NO,NO,(int)M_foo,&one,dBfoo,(int)M_foo,dAfoo,(int)M_foo,&zero,dCfoo,NO);
        // ct_fvv[a,c] = Σ_m B_fvv[a,m] A_fvv[c,m]  (row-major [NV×NV], elem a*NV+c).
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,NV,NV,(int)M_fvv,&one,dAfvv,(int)M_fvv,dBfvv,(int)M_fvv,&zero,dCfvv,NV);
        ct_foo.assign((size_t)NO*NO,0.0); ct_fvv.assign((size_t)NV*NV,0.0);
        cudaMemcpy(ct_foo.data(),dCfoo,(size_t)NO*NO*sizeof(real_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(ct_fvv.data(),dCfvv,(size_t)NV*NV*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dAfoo);tracked_cudaFree(dBfoo);tracked_cudaFree(dCfoo);
        tracked_cudaFree(dAfvv);tracked_cudaFree(dBfvv);tracked_cudaFree(dCfvv);
        foo_fvv_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1=0.0,d2=0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int i=0;i<NO;++i) {
                real_t v=0.0; for (int l=0;l<NO;++l) for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                    v += (2.0*H_OVOV(k,c,l,d)-H_OVOV(k,d,l,c))*(H_T2(i,l,c,d)+H_T1(i,c)*H_T1(l,d));
                d1=std::max(d1,std::fabs(v-ct_foo[(size_t)k*NO+i]));
            }
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1)) {
                real_t v=0.0; for (int k=0;k<NO;++k) for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
                    v += (2.0*H_OVOV(k,c,l,d)-H_OVOV(k,d,l,c))*(H_T2(k,l,a,d)+H_T1(k,a)*H_T1(l,d));
                d2=std::max(d2,std::fabs(v-ct_fvv[(size_t)a*NV+c]));
            }
            std::cout << "[IP-EOM build self-check] cc_Foo Σlcd max|Δ| = " << std::scientific << d1
                      << ", cc_Fvv Σkld max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    // ============================================================
    //  cc_Foo[k,i] = foo[k,i]
    //              + 2 ovov[k,c,l,d] t2[i,l,c,d]  -  ovov[k,d,l,c] t2[i,l,c,d]
    //              + 2 ovov[k,c,l,d] t1[i,c] t1[l,d]  -  ovov[k,d,l,c] t1[i,c] t1[l,d]
    // ============================================================
    std::vector<real_t> h_ccFoo(NO * NO, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int i = 0; i < NO; ++i) {
            real_t v = (k == i ? h_f_oo[k] : 0.0);
            if (foo_fvv_gpu) {
                v += ct_foo[(size_t)k*NO + i];                 // GPU GEMM Σlcd
            } else {
                for (int l = 0; l < NO; ++l)
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v += 2.0 * kcld * H_T2(i,l,c,d) - kdlc * H_T2(i,l,c,d);
                            v += (2.0 * kcld - kdlc) * H_T1(i,c) * H_T1(l,d);
                        }
            }
            h_ccFoo[k*NO + i] = v;
        }

    // ============================================================
    //  cc_Fvv[a,c] = fvv[a,c]
    //              - 2 ovov[k,c,l,d] t2[k,l,a,d]  +  ovov[k,d,l,c] t2[k,l,a,d]
    //              - 2 ovov[k,c,l,d] t1[k,a] t1[l,d]  +  ovov[k,d,l,c] t1[k,a] t1[l,d]
    // ============================================================
    std::vector<real_t> h_ccFvv(NV * NV, 0.0);
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int c = 0; c < NV; ++c) {
            real_t v = (a == c ? h_f_vv[a] : 0.0);
            if (foo_fvv_gpu) {
                v -= ct_fvv[(size_t)a*NV + c];                 // GPU GEMM Σkld
            } else {
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v -= 2.0 * kcld * H_T2(k,l,a,d) - kdlc * H_T2(k,l,a,d);
                            v -= (2.0 * kcld - kdlc) * H_T1(k,a) * H_T1(l,d);
                        }
            }
            h_ccFvv[a*NV + c] = v;
        }

    // ============================================================
    //  Loo[k,i] = cc_Foo[k,i] + Σ_c Fov[k,c] t1[i,c]
    //           + 2 Σ_{l,c} ovoo[l,c,k,i] t1[l,c]    (= 2 Σ ooov[k,i,l,c] t1[l,c])
    //           -   Σ_{l,c} ovoo[k,c,l,i] t1[l,c]    (= Σ ooov[l,i,k,c] t1[l,c])
    //  (fov = 0 for canonical; the Fov[k,c]·t1[i,c] piece is in.)
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

    // ============================================================
    //  Lvv[a,c] = cc_Fvv[a,c] - Σ_k Fov[k,c] t1[k,a]
    //           + 2 Σ_{k,d} ovvv[k,d,a,c] t1[k,d]
    //           -   Σ_{k,d} ovvv[k,c,a,d] t1[k,d]
    // ============================================================
    std::vector<real_t> h_Lvv(NV * NV, 0.0);
    // Inc4(b): the ovvv·t1 contribution to Lvv is computed storage-free on GPU
    // (per-k ovvv slice from B + two accumulating GEMVs + transpose-combine),
    // avoiding the dense ovvv host loop. Falls back to the host loop when no
    // block source (dense path) or no GPU.
    std::vector<real_t> h_lvv_ovvv;
#ifndef GANSU_CPU_ONLY
    if (eri_block_src_ != nullptr && gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const int M = nmo_full_, O = frozen_off_;
        const real_t two=2.0, one=1.0;
        real_t *d_ovvv_k=nullptr,*dY1=nullptr,*dZ=nullptr,*dLvvO=nullptr;
        tracked_cudaMalloc(&d_ovvv_k,(size_t)NV*NV*NV*sizeof(real_t));
        tracked_cudaMalloc(&dY1,(size_t)NV*NV*sizeof(real_t));
        tracked_cudaMalloc(&dZ, (size_t)NV*NV*sizeof(real_t));
        tracked_cudaMalloc(&dLvvO,(size_t)NV*NV*sizeof(real_t));
        cudaMemset(dY1,0,(size_t)NV*NV*sizeof(real_t));
        cudaMemset(dZ, 0,(size_t)NV*NV*sizeof(real_t));
        for (int k=0;k<NO;++k) {
            eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M,
                O+k,1, NO+O,NV, NO+O,NV, NO+O,NV, d_ovvv_k);
            const real_t* t1k = d_t1_ + (size_t)k*NV;
            // y1[a,c] += 2 Σ_d ovvv[k,d,a,c]·t1[k,d]   (ovvv_k col-major [NV²(ac),NV(d)])
            cublasDgemv(cublas, CUBLAS_OP_N, NV*NV, NV, &two, d_ovvv_k, NV*NV, t1k, 1, &one, dY1, 1);
            // z[c,a]  +=   Σ_d ovvv[k,c,a,d]·t1[k,d]   (ovvv_k col-major [NV(d),NV²(ca)], op=T)
            cublasDgemv(cublas, CUBLAS_OP_T, NV, NV*NV, &one, d_ovvv_k, NV, t1k, 1, &one, dZ, 1);
        }
        const int TPB=256; const unsigned nb=(unsigned)(((size_t)NV*NV+TPB-1)/TPB);
        ip_lvv_ovvv_combine<<<nb,TPB>>>(dLvvO, dY1, dZ, NV);
        cudaDeviceSynchronize();
        h_lvv_ovvv.assign((size_t)NV*NV,0.0);
        cudaMemcpy(h_lvv_ovvv.data(), dLvvO, (size_t)NV*NV*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_ovvv_k);tracked_cudaFree(dY1);tracked_cudaFree(dZ);tracked_cudaFree(dLvvO);
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int c = 0; c < NV; ++c) {
            real_t v = h_ccFvv[a*NV + c];
            for (int k = 0; k < NO; ++k)
                v -= h_Fov[k*NV + c] * H_T1(k,a);
            if (!h_lvv_ovvv.empty()) {
                v += h_lvv_ovvv[a*NV + c];
            } else {
                for (int k = 0; k < NO; ++k)
                    for (int d = 0; d < NV; ++d) {
                        v += 2.0 * H_OVVV(k,d,a,c) * H_T1(k,d);
                        v -=       H_OVVV(k,c,a,d) * H_T1(k,d);
                    }
            }
            h_Lvv[a*NV + c] = v;
        }

    blap("F+L (Fov/Foo/Fvv/Loo/Lvv)");

    // ============================================================
    //  Wooov[k,l,i,d] = Σ_c t1[i,c] ovov[k,c,l,d] + ooov[k,i,l,d]
    //  (PySCF Wooov)
    // ============================================================
    std::vector<real_t> h_Wooov(ooov_sz, 0.0);
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
    #define H_WOOOV(k,l,i,d) h_Wooov[(((size_t)(k) * NO + (l)) * NO + (i)) * NV + (d)]
    blap("Wooov");

    // ============================================================
    //  Woooo[k,l,i,j] (IP/PySCF version — NO t1·t1 symmetrization)
    //    = oooo[k,i,l,j]
    //      + Σ_d ovoo[l,d,k,i] t1[j,d]   = Σ_d ooov[k,i,l,d] t1[j,d]
    //      + Σ_c ovoo[k,c,l,j] t1[i,c]   = Σ_c ooov[l,j,k,c] t1[i,c]
    //      + Σ_{c,d} ovov[k,c,l,d] (t2[i,j,c,d] + t1[i,c] t1[j,d])
    // ============================================================
    // GPU GEMM port of Woooo Σcd (NO⁴·NV²) + Wovoo Σcd ovvv·tau (NO³·NV³),
    // sharing tau B (Woooo: tau[i,j,c,d]; Wovoo: tau[j,i,d,c] — DLPNO bt-T2 only
    // ~1e-6 symmetric, so build both explicitly). Mirrors the validated STEOM port.
    const int OO_N = NO*NO, VV_K = NV*NV, OOkl_M = NO*NO, OVkb_M = NO*NV;
    std::vector<real_t> ct_woooo, ct_wovoo;
    bool oooo_wovoo_gpu = false;
    // [device-assembly] When the GPU Σcd path runs, keep the Σcd output (dCo)
    // and the ooov·t1 output (ct_woooo_t1's dC) ALIVE on device so Woooo can be
    // assembled by a kernel into d_Woooo_ (no NO⁴ host scatter / final H2D).
    [[maybe_unused]] real_t* d_ct_woooo    = nullptr;
    [[maybe_unused]] real_t* d_ct_woooo_t1 = nullptr;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        real_t *dB=nullptr,*dB2=nullptr,*dAo=nullptr,*dAv=nullptr,*dCo=nullptr,*dCv=nullptr;
        // Inc4(b): in block mode the dAv reorder (OVkb_M·VV_K = NO·NV³ = 46 GB, a
        // copy of ovvv) is built per-k slice (NV³) instead of materialised whole —
        // see the Wovoo GEMM below. Dense mode keeps the single full reorder.
        const bool tile_av = (eri_block_src_ != nullptr);
        tracked_cudaMalloc(&dB,(size_t)VV_K*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dB2,(size_t)VV_K*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dAo,(size_t)OOkl_M*VV_K*sizeof(real_t));
        if (!tile_av) tracked_cudaMalloc(&dAv,(size_t)OVkb_M*VV_K*sizeof(real_t));
        tracked_cudaMalloc(&dCo,(size_t)OOkl_M*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dCv,(size_t)OVkb_M*OO_N*sizeof(real_t));
        blap("  Woooo Σcd cudaMalloc");
        // Device-build the GEMM inputs directly from the resident ERI/T2/T1
        // blocks (eliminates the host hAo/hAv/hB/hB2 reorder + their H2D).
        {
            const int TPB = 256;
            auto NB = [&](size_t n){ return (unsigned)((n + TPB - 1) / TPB); };
            ip_woooo_build_Ao<<<NB((size_t)OOkl_M*VV_K), TPB>>>(dAo, d_eri_ovov_, NO, NV);
            if (!tile_av)
                ip_woooo_build_Av<<<NB((size_t)OVkb_M*VV_K), TPB>>>(dAv, d_eri_ovvv_, NO, NV);
            ip_woooo_build_B <<<NB((size_t)VV_K*OO_N),   TPB>>>(dB,  d_t2_, d_t1_, NO, NV, false);
            ip_woooo_build_B <<<NB((size_t)VV_K*OO_N),   TPB>>>(dB2, d_t2_, d_t1_, NO, NV, true);
            cudaDeviceSynchronize();
        }
        blap("  Woooo Σcd device-build inputs (Ao/Av/B/B2)");
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OOkl_M,VV_K,&one,dB, OO_N,dAo,VV_K,&zero,dCo,OO_N);
        if (!tile_av) {
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OVkb_M,VV_K,&one,dB2,OO_N,dAv,VV_K,&zero,dCv,OO_N);
        } else {
            // Inc4(b) per-k Wovoo: build ovvv[k] (NV³) from B, reorder to dAv_k
            // (build_Av with NO=1 + per-k pointer), then dCv[:,(k,·)] = dB2·dAv_k.
            // k is a spectator (output column block); slice layout matches the
            // dense path so the result is byte-identical.
            const int M = nmo_full_, O = frozen_off_, TPB = 256;
            const unsigned nb = (unsigned)(((size_t)NV*NV*NV + TPB - 1)/TPB);
            real_t *d_ovvv_k=nullptr,*dAv_k=nullptr;
            tracked_cudaMalloc(&d_ovvv_k,(size_t)NV*NV*NV*sizeof(real_t));
            tracked_cudaMalloc(&dAv_k,   (size_t)NV*NV*NV*sizeof(real_t));
            for (int k=0;k<NO;++k) {
                eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M,
                    O+k,1, NO+O,NV, NO+O,NV, NO+O,NV, d_ovvv_k);
                ip_woooo_build_Av<<<nb,TPB>>>(dAv_k, d_ovvv_k, 1, NV);
                cudaDeviceSynchronize();
                cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
                    OO_N,NV,VV_K,&one, dB2,OO_N, dAv_k,VV_K, &zero,
                    dCv + (size_t)k*NV*OO_N, OO_N);
            }
            tracked_cudaFree(d_ovvv_k); tracked_cudaFree(dAv_k);
        }
        ct_woooo.assign((size_t)OOkl_M*OO_N,0.0); ct_wovoo.assign((size_t)OVkb_M*OO_N,0.0);
        cudaMemcpy(ct_woooo.data(),dCo,(size_t)OOkl_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(ct_wovoo.data(),dCv,(size_t)OVkb_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dB);tracked_cudaFree(dB2);tracked_cudaFree(dAo);if(!tile_av)tracked_cudaFree(dAv);tracked_cudaFree(dCv);
        d_ct_woooo = dCo;   // keep Σcd output on device for ip_woooo_assemble
        blap("  Woooo Σcd GEMM+D2H+malloc/free");
        oooo_wovoo_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1=0.0,d2=0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int l=0;l<NO;++l)
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                        v += H_OVOV(k,c,l,d)*(H_T2(i,j,c,d)+H_T1(i,c)*H_T1(j,d));
                    d1=std::max(d1,std::fabs(v-ct_woooo[(size_t)(k*NO+l)*OO_N+(i*NO+j)]));
                }
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                        v += H_OVVV(k,c,b,d)*(H_T2(j,i,d,c)+H_T1(j,d)*H_T1(i,c));
                    d2=std::max(d2,std::fabs(v-ct_wovoo[(size_t)(k*NV+b)*OO_N+(i*NO+j)]));
                }
            std::cout << "[IP-EOM build self-check] Woooo Σcd max|Δ| = " << std::scientific << d1
                      << ", Wovoo Σcd max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // GPU port of the Woooo +Σ_d ooov·t1 contractions (the two NO⁴·NV host
    // terms). A SINGLE GEMM ct[(p,q,r),f] = Σ_d ooov[p,q,r,d]·t1[f,d] serves
    // BOTH terms (d is the trailing ooov axis ⇒ direct GEMM with d_eri_ooov_):
    //   term1  Σ_d ooov[k,i,l,d]·t1[j,d]  read at (p,q,r,f)=(k,i,l,j)
    //   term2  Σ_c ooov[l,j,k,c]·t1[i,c]  read at (p,q,r,f)=(l,j,k,i)
    // Mirrors the EA-EOM Stage 4 ooov/ovvv·t1 idiom.
    std::vector<real_t> ct_woooo_t1;
    bool woooo_t1_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t c_sz = (size_t)NO*NO*NO*NO;   // [(p,q,r),f]
        std::vector<real_t> hT((size_t)NV*NO);
        #pragma omp parallel for
        for (int d=0;d<NV;++d) for (int f=0;f<NO;++f) hT[(size_t)d*NO+f] = h_t1[f*NV+d];
        real_t *dT=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dT,(size_t)NV*NO*sizeof(real_t));
        tracked_cudaMalloc(&dC, c_sz*sizeof(real_t));
        cudaMemcpy(dT,hT.data(),(size_t)NV*NO*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0, zero=0.0;
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            NO, NO*NO*NO, NV, &one,
            dT, NO, d_eri_ooov_, NV, &zero, dC, NO);
        ct_woooo_t1.assign(c_sz, 0.0);
        cudaMemcpy(ct_woooo_t1.data(), dC, c_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dT);
        d_ct_woooo_t1 = dC;   // keep ooov·t1 output on device for ip_woooo_assemble
        woooo_t1_gpu = true;
    }
#endif
    std::vector<real_t> h_Woooo(oooo_sz, 0.0);
    bool woooo_dev_assembled = false;
#ifndef GANSU_CPU_ONLY
    if (d_ct_woooo && d_ct_woooo_t1) {
        // Device-assemble d_Woooo_ directly from oooo + the device-resident Σcd
        // (d_ct_woooo) and ooov·t1 (d_ct_woooo_t1) buffers — eliminates the NO⁴
        // host scatter loop + the final H2D upload. A host copy is D2H'd back
        // because the Wovoo build below still reads h_Woooo (Σ_l Woooo·t1).
        tracked_cudaMalloc(&d_Woooo_, oooo_sz * sizeof(real_t));
        const int TPB = 256;
        const unsigned NBk = (unsigned)((oooo_sz + TPB - 1) / TPB);
        ip_woooo_assemble<<<NBk, TPB>>>(d_Woooo_, d_eri_oooo_,
                                        d_ct_woooo_t1, d_ct_woooo, NO);
        cudaMemcpy(h_Woooo.data(), d_Woooo_, oooo_sz * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_ct_woooo);    d_ct_woooo = nullptr;
        tracked_cudaFree(d_ct_woooo_t1); d_ct_woooo_t1 = nullptr;
        woooo_dev_assembled = true;
    }
#endif
    blap("  Woooo ct_t1 GEMM + dev-asm");
    if (!woooo_dev_assembled) {
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int l = 0; l < NO; ++l)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOO(k,i,l,j);
                    if (woooo_t1_gpu) {
                        v += ct_woooo_t1[(((size_t)k*NO+i)*NO+l)*NO + j];  // Σ_d ooov[k,i,l,d]·t1[j,d]
                        v += ct_woooo_t1[(((size_t)l*NO+j)*NO+k)*NO + i];  // Σ_c ooov[l,j,k,c]·t1[i,c]
                    } else {
                        for (int d = 0; d < NV; ++d)
                            v += H_OOOV(k,i,l,d) * H_T1(j,d);
                        for (int c = 0; c < NV; ++c)
                            v += H_OOOV(l,j,k,c) * H_T1(i,c);
                    }
                    if (oooo_wovoo_gpu) {
                        v += ct_woooo[(size_t)(k*NO+l)*OO_N + (i*NO+j)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int d = 0; d < NV; ++d) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += kcld * (H_T2(i,j,c,d) + H_T1(i,c) * H_T1(j,d));
                            }
                    }
                    h_Woooo[(((size_t)k * NO + l) * NO + i) * NO + j] = v;
                }
    }  // !woooo_dev_assembled
    if (woooo_t1_gpu && std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
        real_t dmax = 0.0;
        for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int i=0;i<NO;++i)
            for (int l=0;l<NO;l+=(NO/2>0?NO/2:1)) for (int j=0;j<NO;++j) {
                real_t t=0.0; for (int d=0;d<NV;++d) t += H_OOOV(k,i,l,d)*H_T1(j,d);
                dmax = std::max(dmax, std::fabs(t - ct_woooo_t1[(((size_t)k*NO+i)*NO+l)*NO+j]));
            }
        std::cout << "[IP-EOM build self-check] Woooo Σ_d ooov·t1 GEMM vs host: max|Δ| = "
                  << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
    }

    // ============================================================
    // GPU GEMM port of the two O(NO³·NV³) contraction hotspots. Each is reused
    // in TWO consumers below (Wovov + standalone W1ovov; Wovvo + standalone
    // W1ovvo), so the IP build computes them ONCE on the GPU instead of 4×
    // host nested loops. Pattern mirrors the validated STEOM build_dressed port.
    //   ct_ovov[(k,d),(i,b)] = Σ_{c,l} ovov[k,c,l,d]·t2[i,l,c,b]
    //   ct_ovvo[(k,c),(Y,X)] = Σ_{l,d}[(2·ovov[k,c,l,d]−ovov[k,d,l,c])·t2[Y,l,X,d]
    //                                  − ovov[k,c,l,d]·t2[l,Y,X,d]]
    //     consumed as (Y,X)=(i,a) by Wovvo[k,a,c,i] and (Y,X)=(j,b) by W1ovvo[k,b,c,j].
    // ============================================================
    const int IP_MO_kd = NO*NV, IP_NO_ib = NO*NV, IP_KO_cl = NV*NO;   // ct_ovov
    const int IP_M_kc = NO*NV, IP_N_yx = NO*NV, IP_K_ld = NO*NV;      // ct_ovvo
    std::vector<real_t> ct_ovov, ct_ovvo;
    bool ip_ct_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, negone = -1.0, zero = 0.0;
        {   // ct_ovov: C[(k,d),(i,b)] = A[(k,d),(c,l)]·B[(c,l),(i,b)] — inputs device-built
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)IP_MO_kd*IP_KO_cl*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)IP_KO_cl*IP_NO_ib*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)IP_MO_kd*IP_NO_ib*sizeof(real_t));
            const size_t nel=(size_t)NO*NV*NV*NO; const unsigned NB=(unsigned)((nel+255)/256);
            ip_ctovov_buildA<<<NB,256>>>(dA, d_eri_ovov_, NO, NV);
            ip_ctovov_buildB<<<NB,256>>>(dB, d_t2_, NO, NV);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,IP_NO_ib,IP_MO_kd,IP_KO_cl,&one,
                        dB,IP_NO_ib,dA,IP_KO_cl,&zero,dC,IP_NO_ib);
            ct_ovov.assign((size_t)IP_MO_kd*IP_NO_ib,0.0);
            cudaMemcpy(ct_ovov.data(),dC,(size_t)IP_MO_kd*IP_NO_ib*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        {   // ct_ovvo: C[(k,c),(Y,X)] = A1·B1 − A2·B2 — inputs device-built
            real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA1,(size_t)IP_M_kc*IP_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB1,(size_t)IP_K_ld*IP_N_yx*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)IP_M_kc*IP_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)IP_K_ld*IP_N_yx*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)IP_M_kc*IP_N_yx*sizeof(real_t));
            const size_t nel=(size_t)NO*NV*NO*NV; const unsigned NB=(unsigned)((nel+255)/256);
            ip_ctovvo_buildA12<<<NB,256>>>(dA1, dA2, d_eri_ovov_, NO, NV);
            ip_ctovvo_buildB12<<<NB,256>>>(dB1, dB2, d_t2_, NO, NV);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,IP_N_yx,IP_M_kc,IP_K_ld,&one,
                        dB1,IP_N_yx,dA1,IP_K_ld,&zero,dC,IP_N_yx);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,IP_N_yx,IP_M_kc,IP_K_ld,&negone,
                        dB2,IP_N_yx,dA2,IP_K_ld,&one,dC,IP_N_yx);
            ct_ovvo.assign((size_t)IP_M_kc*IP_N_yx,0.0);
            cudaMemcpy(ct_ovvo.data(),dC,(size_t)IP_M_kc*IP_N_yx*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA1);tracked_cudaFree(dB1);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        }
        ip_ct_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1 = 0.0, d2 = 0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int d=0;d<NV;d+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int b=0;b<NV;++b) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int l=0;l<NO;++l) v+=H_OVOV(k,c,l,d)*H_T2(i,l,c,b);
                    d1=std::max(d1,std::fabs(v-ct_ovov[(size_t)(k*NV+d)*IP_NO_ib+(i*NV+b)]));
                }
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int X=0;X<NV;X+=(NV/2>0?NV/2:1))
                for (int c=0;c<NV;++c) for (int Y=0;Y<NO;++Y) {
                    real_t v=0.0; for (int l=0;l<NO;++l) for (int dd=0;dd<NV;++dd){
                        real_t kcld=H_OVOV(k,c,l,dd);
                        v+=2.0*kcld*H_T2(Y,l,X,dd)-kcld*H_T2(l,Y,X,dd)-H_OVOV(k,dd,l,c)*H_T2(Y,l,X,dd);}
                    d2=std::max(d2,std::fabs(v-ct_ovvo[(size_t)(k*NV+c)*IP_N_yx+(Y*NV+X)]));
                }
            std::cout << "[IP-EOM build self-check] ct_ovov max|Δ| = " << std::scientific << d1
                      << ", ct_ovvo max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    blap("Woooo + ct_ovov/ovvo Σcd GEMM");

    // (scaling) Skip the dense canonical Wovov/Wovvo build (the ~15 s O(NO²·NV³)
    // sub-phase that dominates build_dressed at 100-atom scale) when the
    // native-bare operator is active AND share-barH is off:
    //   • the native bare IP operator never borrows the dense nocc²·nvir² Wovov/
    //     Wovvo (see the `if (!use_native_bare_)` guard in
    //     dlpno_ip_eom_native_operator.cu — it rebuilds them per-pair from
    //     Phase24 bare + native ring), and
    //   • share-barH off (barh_cache_ == nullptr) means STEOM builds its own copy
    //     rather than borrowing ours,
    // so building them here is pure dead work.  Mirrors the EA h_Wvvvv skip.
    // Non-native / reference / share-on paths keep the full build (byte-unchanged:
    // d_Wovov_/d_Wovvo_ are still produced, published, and consumed as before).
    const bool skip_dense_wovov_wovvo = [&]{
        auto on = [](const char* n, bool d){ const char* e = std::getenv(n);
            return (!e || !e[0]) ? d : (e[0] != '0'); };
        return on("GANSU_DLPNO_NATIVE_EOM", false) &&
               on("GANSU_DLPNO_NATIVE_BARE", true) &&
               on("GANSU_DLPNO_CANONICAL_SKIP", true) &&
               (barh_cache_ == nullptr);
    }();
    if (skip_dense_wovov_wovvo)
        std::cout << "  [IP-EOM canonical-skip] dense Wovov/Wovvo build SKIPPED "
                     "(native-bare + share-barH off; STEOM builds its own)\n";
    std::vector<real_t> h_Wovov, h_Wovvo;   // declared out here for the D2H upload
  if (!skip_dense_wovov_wovvo) {

    // ============================================================
    //  W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d] t2[i,l,c,b]
    //  W2ovov[k,b,i,d] = -Σ_l Wooov[k,l,i,d] t1[l,b]
    //                   + Σ_c ovvv[k,c,b,d] t1[i,c]
    //  Wovov = W1ovov + W2ovov
    // ============================================================
    // GPU port of the Wovov W2 +Σ_c ovvv·t1 contraction (dominant O(NO²·NV³)
    // host loop). ct_wovov[k,i,(b,d)] = Σ_c t1[i,c]·ovvv[k,c,b,d] via per-k
    // strided-batched GEMM (B = d_eri_ovvv_ slab contiguous [c,(b,d)]). Mirrors
    // the validated EA-EOM Stage 4 port.
    std::vector<real_t> ct_wovov;
    bool wovov_w2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t cw_sz = (size_t)NO*NO*NV*NV;   // [k,i,(b,d)]
        real_t* dC=nullptr; tracked_cudaMalloc(&dC, cw_sz*sizeof(real_t));
        const real_t one=1.0, zero=0.0;
        if (eri_block_src_ != nullptr) {
            // Inc4(b) ovvv storage-free: build each occupied slice ovvv[k] (nvir³,
            // 0.3 GB at nvir=334 vs the dense nocc·nvir³ = 46 GB) on the fly from
            // the cluster B, then one GEMM per k. k is a spectator in this Wovov-W2
            // contraction. The slice layout (mo_eri_block_into over occ range
            // [O+k,O+k+1)) is byte-identical to d_eri_ovvv_[k·nvir³], so the result
            // matches the dense batched GEMM exactly.
            const int M = nmo_full_, O = frozen_off_;
            real_t* d_ovvv_k = nullptr;
            tracked_cudaMalloc(&d_ovvv_k, (size_t)NV*NV*NV*sizeof(real_t));
            for (int k = 0; k < NO; ++k) {
                eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M,
                    O+k, 1, NO+O, NV, NO+O, NV, NO+O, NV, d_ovvv_k);
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    NV*NV, NO, NV, &one,
                    d_ovvv_k, NV*NV,
                    d_t1_,    NV,
                    &zero, dC + (size_t)k*NO*NV*NV, NV*NV);
            }
            tracked_cudaFree(d_ovvv_k);
        } else {
            cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                NV*NV, NO, NV, &one,
                d_eri_ovvv_, NV*NV, (long long)NV*NV*NV,
                d_t1_,       NV,    0LL,
                &zero, dC, NV*NV, (long long)NO*NV*NV, NO);
        }
        ct_wovov.assign(cw_sz, 0.0);
        cudaMemcpy(ct_wovov.data(), dC, cw_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dC);
        wovov_w2_gpu = true;
    }
#endif
    h_Wovov.assign(ovov_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);                       // W1: oovv
                    if (ip_ct_gpu) {
                        v -= ct_ovov[(size_t)(k*NV+d)*IP_NO_ib + (i*NV+b)];  // W1: -ovov·t2
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b); // W1: -ovov·t2
                    }
                    for (int l = 0; l < NO; ++l)
                        v -= H_WOOOV(k,l,i,d) * H_T1(l,b);             // W2: -Wooov·t1
                    if (wovov_w2_gpu) {
                        v += ct_wovov[(((size_t)k*NO+i)*NV+b)*NV+d];  // W2: +ovvv·t1 (GEMM)
                    } else {
                        for (int c = 0; c < NV; ++c)
                            v += H_OVVV(k,c,b,d) * H_T1(i,c);          // W2: +ovvv·t1
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
        std::cout << "[IP-EOM build self-check] Wovov W2 (ovvv·t1) batched GEMM vs host: max|Δ| = "
                  << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
    }

    // ============================================================
    //  W1ovvo[k,a,c,i] = ovvo[k,c,a,i]
    //                  + 2 Σ_{l,d} ovov[k,c,l,d] t2[i,l,a,d]
    //                  -   Σ_{l,d} ovov[k,c,l,d] t2[l,i,a,d]
    //                  -   Σ_{l,d} ovov[k,d,l,c] t2[i,l,a,d]
    //  W2ovvo[k,a,c,i] = -Σ_l t1[l,a] Wooov[l,k,i,c]
    //                  + Σ_d ovvv[k,c,a,d] t1[i,d]
    //  Wovvo = W1ovvo + W2ovvo
    // ============================================================
    // GPU port of the Wovvo W2 +Σ_d ovvv·t1 contraction (dominant O(NO·NV³·NO)
    // host loop). ct_wovvo[(k,c,a),i] = Σ_d ovvv[k,c,a,d]·t1[i,d]
    // = ovvv[(k,c,a),d]·t1T[d,i] (single GEMM, d is the trailing ovvv axis).
    // Mirrors the validated EA-EOM Stage 4 port.
    std::vector<real_t> ct_wovvo;
    bool wovvo_w2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t cw_sz = (size_t)NO*NV*NV*NO;   // [(k,c,a),i]
        std::vector<real_t> hT((size_t)NV*NO);
        #pragma omp parallel for
        for (int d=0;d<NV;++d) for (int i=0;i<NO;++i) hT[(size_t)d*NO+i] = h_t1[i*NV+d];
        real_t *dT=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dT,(size_t)NV*NO*sizeof(real_t));
        tracked_cudaMalloc(&dC, cw_sz*sizeof(real_t));
        cudaMemcpy(dT,hT.data(),(size_t)NV*NO*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0, zero=0.0;
        if (eri_block_src_ != nullptr) {
            // Inc4(b) ovvv storage-free: per-k slice ovvv[k] (nvir³) from B, one
            // GEMM per k (k spectator). Contracts the trailing vir axis; the slice
            // layout matches d_eri_ovvv_[k·nvir³] so the result is byte-identical.
            const int M = nmo_full_, O = frozen_off_;
            real_t* d_ovvv_k = nullptr;
            tracked_cudaMalloc(&d_ovvv_k, (size_t)NV*NV*NV*sizeof(real_t));
            for (int k = 0; k < NO; ++k) {
                eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M,
                    O+k, 1, NO+O, NV, NO+O, NV, NO+O, NV, d_ovvv_k);
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    NO, NV*NV, NV, &one,
                    dT, NO, d_ovvv_k, NV, &zero, dC + (size_t)k*NO*NV*NV, NO);
            }
            tracked_cudaFree(d_ovvv_k);
        } else {
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                NO, NO*NV*NV, NV, &one,
                dT, NO, d_eri_ovvv_, NV, &zero, dC, NO);
        }
        ct_wovvo.assign(cw_sz, 0.0);
        cudaMemcpy(ct_wovvo.data(), dC, cw_sz*sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(dT); tracked_cudaFree(dC);
        wovvo_w2_gpu = true;
    }
#endif
    h_Wovvo.assign(ovvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c)
                for (int i = 0; i < NO; ++i) {
                    real_t v = H_OVVO(k,c,a,i);
                    if (ip_ct_gpu) {
                        v += ct_ovvo[(size_t)(k*NV+c)*IP_N_yx + (i*NV+a)];  // (Y,X)=(i,a)
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += 2.0 * kcld * H_T2(i,l,a,d);
                                v -=       kcld * H_T2(l,i,a,d);
                                v -= H_OVOV(k,d,l,c) * H_T2(i,l,a,d);
                            }
                    }
                    for (int l = 0; l < NO; ++l)
                        v -= H_T1(l,a) * H_WOOOV(l,k,i,c);
                    if (wovvo_w2_gpu) {
                        v += ct_wovvo[(((size_t)k*NV+c)*NV+a)*NO+i];  // W2: +ovvv·t1 (GEMM)
                    } else {
                        for (int d = 0; d < NV; ++d)
                            v += H_OVVV(k,c,a,d) * H_T1(i,d);
                    }
                    h_Wovvo[(((size_t)k * NV + a) * NV + c) * NO + i] = v;
                }
    if (wovvo_w2_gpu && std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
        real_t dmax = 0.0;
        for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) for (int i=0;i<NO;++i) {
                real_t t=0.0; for (int d=0;d<NV;++d) t += H_OVVV(k,c,a,d)*H_T1(i,d);
                dmax = std::max(dmax, std::fabs(t - ct_wovvo[(((size_t)k*NV+c)*NV+a)*NO+i]));
            }
        std::cout << "[IP-EOM build self-check] Wovvo W2 (ovvv·t1) GEMM vs host: max|Δ| = "
                  << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
    }
  }  // end if (!skip_dense_wovov_wovvo) — dense Wovov/Wovvo dead in native-bare path
    blap("Wovov + Wovvo (W2: ovvv·t1, Wooov·t1)");

    // ============================================================
    //  Wovoo[k,b,i,j] — used by the σ2 ← r1 coupling. Same form as
    //  the EE-EOM woVoO build, which already follows PySCF Wovoo
    //  exactly (verified line-by-line). We recompute it inline here
    //  rather than reuse the EE-EOM helper to keep the IP-EOM module
    //  self-contained.
    //
    //  Wkbij = ooov[i,k,j,b]                                          (bare)
    //        + Σ_d W1ovov[k,b,i,d] t1[j,d]
    //        - Σ_l Woooo[k,l,i,j] t1[l,b]
    //        + Σ_c W1ovvo[k,b,c,j] t1[i,c]
    //        + 2 Σ_{l,d} ooov[k,i,l,d] t2[l,j,d,b]
    //        -   Σ_{l,d} ooov[k,i,l,d] t2[j,l,d,b]
    //        -   Σ_{l,d} ooov[l,i,k,d] t2[l,j,d,b]
    //        +   Σ_{c,d} ovvv[k,c,b,d] t2[j,i,d,c]
    //        +   Σ_{c,d} ovvv[k,c,b,d] t1[j,d] t1[i,c]
    //        -   Σ_{c,l} ooov[l,j,k,c] t2[l,i,b,c]
    //        +   Σ_c Fov[k,c] t2[i,j,c,b]
    // ============================================================
    // First build the internal helpers (W1ovov is already available, but we
    // need W1ovvo here too — recompute the W1 versions explicitly to keep
    // the assembly transparent).
    std::vector<real_t> h_W1ovov(ovov_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);
                    if (ip_ct_gpu) {
                        v -= ct_ovov[(size_t)(k*NV+d)*IP_NO_ib + (i*NV+b)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                    }
                    h_W1ovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }
    #define H_W1OVOV(k,b,i,d) h_W1ovov[(((size_t)(k) * NV + (b)) * NO + (i)) * NV + (d)]

    std::vector<real_t> h_W1ovvo(ovvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVO(k,c,b,j);
                    if (ip_ct_gpu) {
                        v += ct_ovvo[(size_t)(k*NV+c)*IP_N_yx + (j*NV+b)];  // (Y,X)=(j,b)
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
    blap("  Wovoo W1ovov+W1ovvo rebuild");

    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;
    // GPU GEMM port of Wovoo Σ_ld ooov·t2 (NO⁴·NV²), mirrors STEOM:
    //   ct[k,i,j,b] = Σ_{l,d}[(2·ooov(k,i,l,d)−ooov(l,i,k,d))·t2(l,j,d,b) − ooov(k,i,l,d)·t2(j,l,d,b)]
    const int KI_M = NO*NO, JB_N = NO*NV, LD_K2 = NO*NV;
    std::vector<real_t> ct_wovoo_t2;
    bool wovoo_t2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        real_t *dA13=nullptr,*dBlj=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA13,(size_t)KI_M*LD_K2*sizeof(real_t));
        tracked_cudaMalloc(&dBlj,(size_t)LD_K2*JB_N*sizeof(real_t));
        tracked_cudaMalloc(&dA2,(size_t)KI_M*LD_K2*sizeof(real_t));
        tracked_cudaMalloc(&dB2,(size_t)LD_K2*JB_N*sizeof(real_t));
        tracked_cudaMalloc(&dC, (size_t)KI_M*JB_N*sizeof(real_t));
        { const size_t nA=(size_t)NO*NO*NO*NV; const unsigned NB=(unsigned)((nA+255)/256);
          ip_wovoo_buildA13_2<<<NB,256>>>(dA13, dA2, d_eri_ooov_, NO, NV); }
        { const size_t nB=(size_t)NO*NV*NO*NV; const unsigned NB=(unsigned)((nB+255)/256);
          ip_wovoo_buildB_2<<<NB,256>>>(dBlj, dB2, d_t2_, NO, NV); }
        const real_t one=1.0,negone=-1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,JB_N,KI_M,LD_K2,&one,   dBlj,JB_N,dA13,LD_K2,&zero,dC,JB_N);
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,JB_N,KI_M,LD_K2,&negone,dB2, JB_N,dA2, LD_K2,&one, dC,JB_N);
        ct_wovoo_t2.assign((size_t)KI_M*JB_N,0.0);
        cudaMemcpy(ct_wovoo_t2.data(),dC,(size_t)KI_M*JB_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA13);tracked_cudaFree(dBlj);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        wovoo_t2_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int k=0;k<NO;++k) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
                        t += 2.0*H_OOOV(k,i,l,d)*H_T2(l,j,d,b) - H_OOOV(k,i,l,d)*H_T2(j,l,d,b)
                             - H_OOOV(l,i,k,d)*H_T2(l,j,d,b);
                    dmax=std::max(dmax,std::fabs(t-ct_wovoo_t2[(size_t)(k*NO+i)*JB_N+(j*NV+b)]));
                }
            std::cout << "[IP-EOM build self-check] Wovoo ooov·t2 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    blap("  Wovoo ct_t2 ooov·t2 (prep+GEMM)");
    // GPU GEMM port of Wovoo item 6 — the last big inner-loop scatter term
    // (was outer-loop O(NO⁴·NV²) ~ 220 G ops at anthracene = Wovoo phase
    // bottleneck per build_dressed-PROF, STEOM build mirror). C[(k,j),(b,i)] =
    // Σ_{l,c} OOOV(l,j,k,c)·T2(l,i,b,c). Free (k,j) on M / (b,i) on N /
    // contract (l,c) on K. Sign applied at scatter (`v -= ct6[...]`).
    const int M_w6 = NO*NO, N_w6 = NV*NO, K_w6 = NO*NV;
    std::vector<real_t> ct_wovoo_6;
    bool wovoo_6_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        ct_wovoo_6.assign((size_t)M_w6*N_w6, 0.0);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)M_w6*K_w6*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)K_w6*N_w6*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)M_w6*N_w6*sizeof(real_t));
        { const size_t nA=(size_t)NO*NO*NO*NV; const unsigned NB=(unsigned)((nA+255)/256);
          ip_wovoo6_buildA<<<NB,256>>>(dA, d_eri_ooov_, NO, NV); }
        { const size_t nB=(size_t)NO*NV*NV*NO; const unsigned NB=(unsigned)((nB+255)/256);
          ip_wovoo6_buildB<<<NB,256>>>(dB, d_t2_, NO, NV); }
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w6,M_w6,K_w6,&one,dB,N_w6,dA,K_w6,&zero,dC,N_w6);
        cudaMemcpy(ct_wovoo_6.data(),dC,(size_t)M_w6*N_w6*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        wovoo_6_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int k=0;k<NO;++k) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
                        t += H_OOOV(l,j,k,c)*H_T2(l,i,b,c);
                    dmax=std::max(dmax,std::fabs(t-ct_wovoo_6[(size_t)(k*NO+j)*N_w6+(b*NO+i)]));
                }
            std::cout << "[IP-EOM build self-check] Wovoo item 6 (ooov·t2 contract l,c): max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    blap("  Wovoo ct_6 ooov·t2 (prep+GEMM)");
    std::vector<real_t> h_Wovoo(wovoo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOV(i,k,j,b);  // bare
                    for (int d = 0; d < NV; ++d)
                        v += H_W1OVOV(k,b,i,d) * H_T1(j,d);
                    for (int l = 0; l < NO; ++l)
                        v -= h_Woooo[(((size_t)k * NO + l) * NO + i) * NO + j] * H_T1(l,b);
                    for (int c = 0; c < NV; ++c)
                        v += H_W1OVVO(k,b,c,j) * H_T1(i,c);
                    if (wovoo_t2_gpu) {
                        v += ct_wovoo_t2[(size_t)(k*NO+i)*JB_N + (j*NV+b)];
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                v += 2.0 * H_OOOV(k,i,l,d) * H_T2(l,j,d,b);
                                v -=       H_OOOV(k,i,l,d) * H_T2(j,l,d,b);
                                v -=       H_OOOV(l,i,k,d) * H_T2(l,j,d,b);
                            }
                    }
                    if (oooo_wovoo_gpu) {
                        v += ct_wovoo[(size_t)(k*NV+b)*OO_N + (i*NO+j)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int d = 0; d < NV; ++d) {
                                v += H_OVVV(k,c,b,d) * H_T2(j,i,d,c);
                                v += H_OVVV(k,c,b,d) * H_T1(j,d) * H_T1(i,c);
                            }
                    }
                    if (wovoo_6_gpu) {
                        v -= ct_wovoo_6[(size_t)(k*NO+j)*N_w6 + (b*NO+i)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OOOV(l,j,k,c) * H_T2(l,i,b,c);
                    }
                    for (int c = 0; c < NV; ++c)
                        v += h_Fov[k*NV + c] * H_T2(i,j,c,b);
                    h_Wovoo[(((size_t)k * NV + b) * NO + i) * NO + j] = v;
                }
    blap("Wovoo (+ W1ovov/W1ovvo)");

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
    if (!woooo_dev_assembled) {   // already malloc'd + filled by ip_woooo_assemble
        tracked_cudaMalloc(&d_Woooo_, oooo_sz  * sizeof(real_t));
        cudaMemcpy(d_Woooo_, h_Woooo.data(), oooo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    tracked_cudaMalloc(&d_Wooov_, ooov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wooov_, h_Wooov.data(), ooov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    if (!skip_dense_wovov_wovvo) {   // else d_Wovov_/d_Wovvo_ stay nullptr (native bare never reads them)
        tracked_cudaMalloc(&d_Wovov_, ovov_sz  * sizeof(real_t));
        cudaMemcpy(d_Wovov_, h_Wovov.data(), ovov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
        tracked_cudaMalloc(&d_Wovvo_, ovvo_sz  * sizeof(real_t));
        cudaMemcpy(d_Wovvo_, h_Wovvo.data(), ovvo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    tracked_cudaMalloc(&d_Wovoo_, wovoo_sz * sizeof(real_t));
    cudaMemcpy(d_Wovoo_, h_Wovoo.data(), wovoo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    blap("D2H upload");
    std::cout << "  IP-EOM-CCSD dressed intermediates built (PySCF IP definitions)." << std::endl;

    #undef H_WOOOV
    #undef H_W1OVOV
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


// ==================================================================
//  apply — full PySCF IP-EOM-CCSD matvec
// ==================================================================

void IPEOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    // (a) Stub path: intermediates not built → diagonal-only matvec
    //     (unit-test path with d_eri_mo == nullptr).
    if (d_Loo_ == nullptr) {
        if (!gpu::gpu_available()) {
            #pragma omp parallel for
            for (int idx = 0; idx < total_dim_; ++idx) {
                d_output[idx] = d_diagonal_[idx] * d_input[idx];
            }
        } else {
#ifndef GANSU_CPU_ONLY
            const int threads = 256;
            const int blocks  = (total_dim_ + threads - 1) / threads;
            ip_eom_diag_matvec_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
        }
        return;
    }

    // (b) Full IP-EOM-CCSD matvec (PySCF ipccsd_matvec).
    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + h_dim_;
    real_t*       d_s1 = d_output;
    real_t*       d_s2 = d_output + h_dim_;

    if (!gpu::gpu_available()) {
        // -------- CPU path --------
        // σ1
        #pragma omp parallel for
        for (int i = 0; i < nocc_; ++i) {
            real_t s = 0.0;
            for (int k = 0; k < nocc_; ++k) s -= d_Loo_[k * nocc_ + i] * d_r1[k];
            for (int l = 0; l < nocc_; ++l)
                for (int d = 0; d < nvir_; ++d) {
                    real_t fov_ld = d_Fov_[l * nvir_ + d];
                    s += 2.0 * fov_ld * d_r2[((size_t)i * nocc_ + l) * nvir_ + d];
                    s -=       fov_ld * d_r2[((size_t)l * nocc_ + i) * nvir_ + d];
                }
            for (int k = 0; k < nocc_; ++k)
                for (int l = 0; l < nocc_; ++l)
                    for (int dd = 0; dd < nvir_; ++dd) {
                        real_t w1 = d_Wooov_[(((size_t)k * nocc_ + l) * nocc_ + i) * nvir_ + dd];
                        real_t w2 = d_Wooov_[(((size_t)l * nocc_ + k) * nocc_ + i) * nvir_ + dd];
                        real_t r2_kld = d_r2[((size_t)k * nocc_ + l) * nvir_ + dd];
                        s += (-2.0 * w1 + w2) * r2_kld;
                    }
            d_s1[i] = s;
        }

        // tmp[c] precompute
        std::vector<real_t> tmp_c(nvir_, 0.0);
        for (int c = 0; c < nvir_; ++c) {
            real_t s = 0.0;
            for (int k = 0; k < nocc_; ++k)
                for (int l = 0; l < nocc_; ++l)
                    for (int dd = 0; dd < nvir_; ++dd) {
                        real_t w_lkdc = d_eri_oovv_[(((size_t)l * nocc_ + k) * nvir_ + dd) * nvir_ + c];
                        real_t w_kldc = d_eri_oovv_[(((size_t)k * nocc_ + l) * nvir_ + dd) * nvir_ + c];
                        s += (2.0 * w_lkdc - w_kldc) * d_r2[((size_t)k * nocc_ + l) * nvir_ + dd];
                    }
            tmp_c[c] = s;
        }

        // σ2
        #pragma omp parallel for
        for (int idx = 0; idx < h2p_dim_; ++idx) {
            int a = idx % nvir_;
            int t = idx / nvir_;
            int j = t % nocc_;
            int i = t / nocc_;
            real_t s = 0.0;
            for (int k = 0; k < nocc_; ++k)
                s -= d_Wovoo_[((size_t)k * nvir_ + a) * nocc_ * nocc_ + (size_t)i * nocc_ + j] * d_r1[k];
            for (int dd = 0; dd < nvir_; ++dd)
                s += d_Lvv_[a * nvir_ + dd] * d_r2[((size_t)i * nocc_ + j) * nvir_ + dd];
            for (int k = 0; k < nocc_; ++k)
                s -= d_Loo_[k * nocc_ + i] * d_r2[((size_t)k * nocc_ + j) * nvir_ + a];
            for (int l = 0; l < nocc_; ++l)
                s -= d_Loo_[l * nocc_ + j] * d_r2[((size_t)i * nocc_ + l) * nvir_ + a];
            for (int k = 0; k < nocc_; ++k)
                for (int l = 0; l < nocc_; ++l)
                    s += d_Woooo_[(((size_t)k * nocc_ + l) * nocc_ + i) * nocc_ + j]
                       * d_r2[((size_t)k * nocc_ + l) * nvir_ + a];
            for (int kl = 0; kl < nocc_; ++kl)
                for (int dd = 0; dd < nvir_; ++dd) {
                    real_t wovvo = d_Wovvo_[(((size_t)kl * nvir_ + a) * nvir_ + dd) * nocc_ + j];
                    s += 2.0 * wovvo * d_r2[((size_t)i * nocc_ + kl) * nvir_ + dd];
                    s -=       wovvo * d_r2[((size_t)kl * nocc_ + i) * nvir_ + dd];
                }
            for (int kl = 0; kl < nocc_; ++kl)
                for (int dd = 0; dd < nvir_; ++dd) {
                    real_t w1 = d_Wovov_[(((size_t)kl * nvir_ + a) * nocc_ + j) * nvir_ + dd];
                    s -= w1 * d_r2[((size_t)i * nocc_ + kl) * nvir_ + dd];
                    real_t w2 = d_Wovov_[(((size_t)kl * nvir_ + a) * nocc_ + i) * nvir_ + dd];
                    s -= w2 * d_r2[((size_t)kl * nocc_ + j) * nvir_ + dd];
                }
            for (int c = 0; c < nvir_; ++c)
                s -= tmp_c[c] * d_t2_[(((size_t)i * nocc_ + j) * nvir_ + c) * nvir_ + a];
            d_s2[idx] = s;
        }
    } else {
#ifndef GANSU_CPU_ONLY
        if (use_gpu_multi_) {
            apply_multi(d_input, d_output);   // Stage IP-5
        } else {
            apply_sigma_gpu(d_r1, d_r2, d_s1, d_s2,
                            d_Loo_, d_Lvv_, d_Fov_, d_Woooo_, d_Wooov_, d_Wovov_,
                            d_Wovvo_, d_Wovoo_, d_eri_oovv_, d_t2_,
                            /*i_begin=*/0, /*i_end=*/nocc_, /*do_sigma1=*/true);
            if (sigma_gemm_) {
                add_big_terms_gemm(d_r2, d_s2);            // Woooo/Wovvo/Wovov via GEMM
                sigma_gemm_selfcheck(d_input, d_s2);
            }
        }
#endif
    }
}

#ifndef GANSU_CPU_ONLY
// Device-parametric IP σ (Stage IP-5).  Identical numerics to the legacy
// single-GPU body; all intermediates are explicit args + an i-slab on σ2, so it
// runs unchanged on any device after a cudaSetDevice (DeviceGuard) by the caller.
void IPEOMCCSDOperator::apply_sigma_gpu(
    const real_t* d_r1, const real_t* d_r2, real_t* d_s1, real_t* d_s2,
    const real_t* Loo, const real_t* Lvv, const real_t* Fov,
    const real_t* Woooo, const real_t* Wooov, const real_t* Wovov,
    const real_t* Wovvo, const real_t* Wovoo, const real_t* eri_oovv,
    const real_t* t2, int i_begin, int i_end, bool do_sigma1, real_t* scr_tmp_c) const
{
    const int threads = 256;
    const int islab = i_end - i_begin;
    if (islab <= 0 && !do_sigma1) return;

    if (do_sigma1) {   // σ1 (full 1h sector; only on the owning device — device 0)
        const int blocks_1 = (h_dim_ + threads - 1) / threads;
        ip_eom_sigma1_full_kernel<<<blocks_1, threads>>>(
            Loo, Fov, Wooov, d_r1, d_r2, d_s1, nocc_, nvir_);
    }
    if (islab <= 0) return;

    // tmp[c] (full reduction; needed by the σ2 tmp·t2 term for every c)
    real_t* d_tmp_c = scr_tmp_c;
    if (!d_tmp_c) tracked_cudaMalloc(&d_tmp_c, (size_t)nvir_ * sizeof(real_t));
    const int blocks_tmp = (nvir_ + threads - 1) / threads;
    ip_eom_sigma2_tmp_c_kernel<<<blocks_tmp, threads>>>(eri_oovv, d_r2, d_tmp_c, nocc_, nvir_);

    // σ2 over the i-slab [i_begin, i_end); writes its rows of the full d_s2.
    const int blocks_2 = (islab * nocc_ * nvir_ + threads - 1) / threads;
    ip_eom_sigma2_full_kernel<<<blocks_2, threads>>>(
        Loo, Lvv, Woooo, Wovov, Wovvo, Wovoo,
        d_tmp_c, t2, d_r1, d_r2, d_s2, nocc_, nvir_, i_begin, i_end,
        /*add_big=*/!sigma_gemm_);

    if (!scr_tmp_c) tracked_cudaFree(d_tmp_c);
}

#ifndef GANSU_CPU_ONLY
// Lazy device-0 setup for the GEMM σ2 path: allocate scratch + build the one-time
// contraction-major repacks WA (Wovvo) and WVA (Wovov).  Idempotent.
void IPEOMCCSDOperator::ensure_sigma_gemm_scratch() const {
    if (d_WA_ != nullptr) return;   // already built
    const size_t Wsz   = (size_t)nocc_ * nvir_ * nvir_ * nocc_;   // WA / WVA
    const size_t h2p   = (size_t)h2p_dim_;
    tracked_cudaMalloc(&d_WA_,     Wsz * sizeof(real_t));
    tracked_cudaMalloc(&d_WVA_,    Wsz * sizeof(real_t));
    tracked_cudaMalloc(&d_R2comb_, h2p * sizeof(real_t));
    tracked_cudaMalloc(&d_R2sw_,   h2p * sizeof(real_t));
    tracked_cudaMalloc(&d_out1_,   h2p * sizeof(real_t));
    tracked_cudaMalloc(&d_out2_,   h2p * sizeof(real_t));
    const int threads = 256;
    const size_t wblocks = (Wsz + threads - 1) / threads;
    ip_repack_wovvo_kernel<<<wblocks, threads>>>(d_Wovvo_, d_WA_,  nocc_, nvir_);
    ip_repack_wovov_kernel<<<wblocks, threads>>>(d_Wovov_, d_WVA_, nocc_, nvir_);
}

// (perf) Device-0 cuBLAS GEMMs for the three big σ2 terms (skipped in the kernel when
// sigma_gemm_ is set), accumulated into d_s2:
//   Woooo : σ2[(ij),a] += Σ_kl Woooo[(kl),(ij)]·r2[(kl),a]
//           col-major Scm[a,ij]=Rcm·Wcmᵀ ⇒ dgemm(N,T,nvir,P,P, r2, Woooo, s2), β=1.
//   Wovvo/Wovov (uncoalesced in the kernel → the real bottleneck):
//     out1[i,(a,j)] = Σ_(l,d) R2comb[i,(l,d)]·WA[(l,d),(a,j)]  − R2b·WVA
//     out2[j,(a,i)] = Σ_(l,d) (−R2sw[j,(l,d)])·WVA[(l,d),(a,i)]
//     σ2[(i,j),a] += out1[i,(a,j)] + out2[j,(a,i)]
//   (R2comb = 2·r2(x,l,d)−r2(l,x,d), R2sw = r2(l,x,d), R2b = r2 direct.)
// Caller guarantees the current device is 0 (GPUHandle::cublas() is device-0 bound).
void IPEOMCCSDOperator::add_big_terms_gemm(const real_t* d_r2, real_t* d_s2) const {
    if (!gpu::gpu_available() || d_Woooo_ == nullptr) return;
    ensure_sigma_gemm_scratch();
    const int P = nocc_ * nocc_;
    const int MO = nocc_, KLD = nocc_ * nvir_, AJ = nvir_ * nocc_;
    cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
    const real_t one = 1.0, negone = -1.0, zero = 0.0;

    // Woooo term (accumulate directly into σ2).
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, nvir_, P, P,
                &one, d_r2, nvir_, d_Woooo_, P, &one, d_s2, nvir_);

    // Reshaped r2 views for the Wovvo/Wovov GEMMs.
    const int threads = 256;
    const size_t blocks = ((size_t)h2p_dim_ + threads - 1) / threads;
    ip_build_r2_views_kernel<<<blocks, threads>>>(d_r2, d_R2comb_, d_R2sw_, nocc_, nvir_);

    // out1 = R2comb·WA − R2b·WVA  (rows i=MO, cols (a,j)=AJ, contract (l,d)=KLD).
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, AJ, MO, KLD,
                &one,    d_WA_,  AJ, d_R2comb_, KLD, &zero, d_out1_, AJ);
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, AJ, MO, KLD,
                &negone, d_WVA_, AJ, d_r2,      KLD, &one,  d_out1_, AJ);
    // out2 = −R2sw·WVA  (rows j=MO, cols (a,i)=AJ).
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, AJ, MO, KLD,
                &negone, d_WVA_, AJ, d_R2sw_,   KLD, &zero, d_out2_, AJ);

    // σ2[(i,j),a] += out1[i,(a,j)] + out2[j,(a,i)].
    ip_sigma2_transpose_add_kernel<<<blocks, threads>>>(d_s2, d_out1_, d_out2_, nocc_, nvir_);
}

// (validation) Compare the assembled GEMM-path σ2 (in d_output_s2) against the legacy
// full kernel (all terms, add_big=true) on device 0.  Gated by GANSU_IP_SIGMA_GEMM_VALIDATE.
void IPEOMCCSDOperator::sigma_gemm_selfcheck(const real_t* d_input,
                                             const real_t* d_output_s2) const {
    if (!(std::getenv("GANSU_IP_SIGMA_GEMM_VALIDATE") != nullptr) || sigma_gemm_check_count_ >= 2)
        return;
    const real_t* d_r1 = d_input;
    const real_t* d_r2 = d_input + h_dim_;
    real_t* d_ref  = nullptr;  real_t* d_tmpc = nullptr;
    tracked_cudaMalloc(&d_ref,  (size_t)h2p_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_tmpc, (size_t)nvir_    * sizeof(real_t));
    const int threads = 256;
    ip_eom_sigma2_tmp_c_kernel<<<(nvir_+threads-1)/threads, threads>>>(
        d_eri_oovv_, d_r2, d_tmpc, nocc_, nvir_);
    ip_eom_sigma2_full_kernel<<<(h2p_dim_+threads-1)/threads, threads>>>(
        d_Loo_, d_Lvv_, d_Woooo_, d_Wovov_, d_Wovvo_, d_Wovoo_,
        d_tmpc, d_t2_, d_r1, d_r2, d_ref, nocc_, nvir_, 0, nocc_, /*add_big=*/true);
    cudaDeviceSynchronize();
    std::vector<real_t> hr(h2p_dim_), ho(h2p_dim_);
    cudaMemcpy(hr.data(), d_ref,        (size_t)h2p_dim_*sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(ho.data(), d_output_s2,  (size_t)h2p_dim_*sizeof(real_t), cudaMemcpyDeviceToHost);
    real_t dmax = 0.0, ref = 0.0;
    for (int i = 0; i < h2p_dim_; ++i) {
        dmax = std::max(dmax, std::fabs(hr[i] - ho[i]));
        ref  = std::max(ref,  std::fabs(hr[i]));
    }
    std::cout << "[IP-EOM σ2 GEMM self-check] max|Δ| = " << std::scientific << dmax
              << "  (max|σ2_ref| = " << ref << ", expect Δ ≤ 1e-9)" << std::defaultfloat << std::endl;
    tracked_cudaFree(d_ref); tracked_cudaFree(d_tmpc);
    ++sigma_gemm_check_count_;
}
#endif

// Stage IP-5: i-slab multi-GPU σ.  σ1 (full) + device 0's σ2 slab into d_output;
// each d>0 computes ONLY its σ2 i-slab into its replica workspace, then the slabs
// are disjoint-gathered into d_output.  Broadcast-first → real device overlap.
void IPEOMCCSDOperator::apply_multi(const real_t* d_input, real_t* d_output) const {
    const int nuse = (int)ws_.size();
    const int no_nv = nocc_ * nvir_;   // per-i row stride of σ2 (j,a)

    // broadcast input to all d>0 first (device 0 idle), then launch all (overlap).
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        cudaMemcpyPeerAsync(w.d_input, d, d_input, 0, (size_t)total_dim_ * sizeof(real_t), 0);
    }
    // device 0: σ1 (full) + its σ2 slab into d_output.
    apply_sigma_gpu(d_input, d_input + h_dim_, d_output, d_output + h_dim_,
                    d_Loo_, d_Lvv_, d_Fov_, d_Woooo_, d_Wooov_, d_Wovov_,
                    d_Wovvo_, d_Wovoo_, d_eri_oovv_, d_t2_,
                    ws_[0].i_begin, ws_[0].i_end, /*do_sigma1=*/true, ws_[0].d_tmp_c);
    // d>0: σ2 slab into ws_[d].d_s2 (async on device d).
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        apply_sigma_gpu(w.d_input, w.d_input + h_dim_, w.d_s1, w.d_s2,
                        w.d_Loo, w.d_Lvv, w.d_Fov, w.d_Woooo, w.d_Wooov, w.d_Wovov,
                        w.d_Wovvo, w.d_Wovoo, w.d_eri_oovv, w.d_t2,
                        w.i_begin, w.i_end, /*do_sigma1=*/false, w.d_tmp_c);
    }
    // gather: sync each d>0 and copy its σ2 i-slab rows into d_output (disjoint).
    for (int d = 1; d < nuse; ++d) {
        const DeviceWorkspace& w = ws_[d];
        MultiGpuManager::DeviceGuard guard(d);
        cudaDeviceSynchronize();
        const size_t off = (size_t)w.i_begin * no_nv;
        const size_t cnt = (size_t)(w.i_end - w.i_begin) * no_nv;
        if (cnt > 0)
            cudaMemcpyPeer(d_output + h_dim_ + off, 0, w.d_s2 + off, d, cnt * sizeof(real_t));
    }
    { MultiGpuManager::DeviceGuard g0(0); cudaDeviceSynchronize(); }

    // (perf) Big σ2 terms (Woooo/Wovvo/Wovov): device-0 GEMMs over the full range,
    // added onto the gathered σ2 (the per-device kernels skipped them when sigma_gemm_).
    if (sigma_gemm_) {
        MultiGpuManager::DeviceGuard g0(0);
        add_big_terms_gemm(d_input + h_dim_, d_output + h_dim_);
        sigma_gemm_selfcheck(d_input, d_output + h_dim_);
    }

    static const bool do_validate = [] {
        const char* e = std::getenv("GANSU_STEOM_EOM_MULTI_VALIDATE");
        return e && e[0] == '1';
    }();
    if (do_validate && multi_check_count_ < 3) {
        real_t* d_ref = nullptr;  real_t* d_s1tmp = nullptr;
        tracked_cudaMalloc(&d_ref,   (size_t)h2p_dim_ * sizeof(real_t));
        tracked_cudaMalloc(&d_s1tmp, (size_t)h_dim_   * sizeof(real_t));
        apply_sigma_gpu(d_input, d_input + h_dim_, d_s1tmp, d_ref,
                        d_Loo_, d_Lvv_, d_Fov_, d_Woooo_, d_Wooov_, d_Wovov_,
                        d_Wovvo_, d_Wovoo_, d_eri_oovv_, d_t2_,
                        0, nocc_, /*do_sigma1=*/false, ws_[0].d_tmp_c);
        // Match the gathered path: it also adds the big-term GEMMs when sigma_gemm_.
        if (sigma_gemm_) add_big_terms_gemm(d_input + h_dim_, d_ref);
        cudaDeviceSynchronize();
        std::vector<real_t> h_ref(h2p_dim_), h_out(h2p_dim_);
        cudaMemcpy(h_ref.data(), d_ref, (size_t)h2p_dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_out.data(), d_output + h_dim_, (size_t)h2p_dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);
        real_t dmax = 0.0;
        for (int i = 0; i < h2p_dim_; ++i) dmax = std::max(dmax, std::fabs(h_ref[i] - h_out[i]));
        std::cout << "[IP-EOM Stage IP-5 self-check] gathered σ2 vs full device-0 ref: max|Δ| = "
                  << std::scientific << dmax << std::defaultfloat << " (expect ≤1e-11)" << std::endl;
        tracked_cudaFree(d_ref); tracked_cudaFree(d_s1tmp);
        ++multi_check_count_;
    }
}
#endif

void IPEOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        ip_eom_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
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

void IPEOMCCSDOperator::print_intermediate_norms(std::ostream& os) const {
    const int NO = nocc_, NV = nvir_;
    const size_t loo_sz = (size_t)NO * NO;
    const size_t lvv_sz = (size_t)NV * NV;
    const size_t fov_sz = (size_t)NO * NV;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;

    os << "  [IP-EOM-CCSD intermediate Frobenius norms]\n"
       << "    dims:  nocc=" << NO << "  nvir=" << NV
       <<           "  h_dim=" << h_dim_ << "  h2p_dim=" << h2p_dim_ << "\n";
    if (d_Loo_ == nullptr) {
        os << "    (intermediates not built — diagonal-only stub)\n";
        return;
    }
    os << std::fixed << std::setprecision(8)
       << "    ‖Loo‖       = " << frobenius_norm_device(d_Loo_,   loo_sz)   << "\n"
       << "    ‖Lvv‖       = " << frobenius_norm_device(d_Lvv_,   lvv_sz)   << "\n"
       << "    ‖Fov‖       = " << frobenius_norm_device(d_Fov_,   fov_sz)   << "\n"
       << "    ‖Woooo‖     = " << frobenius_norm_device(d_Woooo_, oooo_sz)  << "\n"
       << "    ‖Wooov‖     = " << frobenius_norm_device(d_Wooov_, ooov_sz)  << "\n"
       << "    ‖Wovov‖     = " << (d_Wovov_ ? frobenius_norm_device(d_Wovov_, ovov_sz) : 0.0)  << "\n"
       << "    ‖Wovvo‖     = " << (d_Wovvo_ ? frobenius_norm_device(d_Wovvo_, ovvo_sz) : 0.0)  << "\n"
       << "    ‖Wovoo‖     = " << frobenius_norm_device(d_Wovoo_, wovoo_sz) << "\n";
}

} // namespace gansu
