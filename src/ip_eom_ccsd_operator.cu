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
    int i_begin, int i_end)                // Stage IP-5: compute only this outer-occ slab
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

    // -Σ_c d_tmp_c[c] * t2[i,j,c,a]
    for (int c = 0; c < nvir; ++c) {
        s -= d_tmp_c[c] * t2[(((size_t)i * nocc + j) * nvir + c) * nvir + a];
    }

    sigma2[((size_t)i * nocc + j) * nvir + a] = s;   // global index (slab writes its rows)
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
    int num_gpus)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      h_dim_(nocc),
      h2p_dim_(nocc * nocc * nvir),
      total_dim_(nocc + nocc * nocc * nvir),
      d_t1_(d_t1), d_t2_(d_t2),
      eri_block_src_(eri_block_src), d_B_mo_blocks_(d_B_mo_blocks), nmo_full_(nmo_full)
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
        num_gpus_ = (num_gpus > 1 ? num_gpus : 1);
        setup_multi_gpu();   // Stage IP-5: per-device replicas (no-op when num_gpus_==1)
    }
}

IPEOMCCSDOperator::~IPEOMCCSDOperator() {
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
    if (d_Loo_)       tracked_cudaFree(d_Loo_);
    if (d_Lvv_)       tracked_cudaFree(d_Lvv_);
    if (d_Fov_)       tracked_cudaFree(d_Fov_);
    if (d_Woooo_)     tracked_cudaFree(d_Woooo_);
    if (d_Wooov_)     tracked_cudaFree(d_Wooov_);
    if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
    if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
    if (d_Wovoo_)     tracked_cudaFree(d_Wovoo_);
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
        repl(&w.d_Wovov, d_Wovov_, ovov_sz);
        repl(&w.d_Wovvo, d_Wovvo_, ovvo_sz);
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
    tracked_cudaMalloc(&d_eri_ovvv_, ovvv_size * sizeof(real_t));

#ifndef GANSU_CPU_ONLY
    // Phase 0: build the 6 blocks on the fly from B_mo (naux×nmo²), never the
    // full nmo⁴. o=[0,nocc), v=[nocc,nmo). Layouts match the gather kernels below.
    if (eri_block_src_ != nullptr) {
        const int M = nmo_full_;
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,0,nocc,      0,nocc,0,nocc,      d_eri_oooo_); // (ij|kl)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,0,nocc,      0,nocc,nocc,nvir,   d_eri_ooov_); // (ji|kb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,nocc,nvir,   0,nocc,nocc,nvir,   d_eri_ovov_); // (ia|jb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,0,nocc,      nocc,nvir,nocc,nvir,d_eri_oovv_); // (ij|ab)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,nocc,nvir,   nocc,nvir,0,nocc,   d_eri_ovvo_); // (ia|bj)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, 0,nocc,nocc,nvir,   nocc,nvir,nocc,nvir,d_eri_ovvv_); // (ia|bc)
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
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz), h_oooo(oooo_sz);

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_,  ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oooo.data(), d_eri_oooo_,  oooo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

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
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        std::vector<real_t> hB((size_t)VV_K*OO_N), hB2((size_t)VV_K*OO_N);
        #pragma omp parallel for collapse(2)
        for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
            for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                const size_t o=(size_t)(c*NV+d)*OO_N+(i*NO+j);
                hB[o]  = H_T2(i,j,c,d) + H_T1(i,c)*H_T1(j,d);   // Woooo
                hB2[o] = H_T2(j,i,d,c) + H_T1(j,d)*H_T1(i,c);   // Wovoo
            }
        std::vector<real_t> hAo((size_t)OOkl_M*VV_K), hAv((size_t)OVkb_M*VV_K);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
            for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                hAo[(size_t)(k*NO+l)*VV_K+(c*NV+d)] = H_OVOV(k,c,l,d);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int b=0;b<NV;++b)
            for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                hAv[(size_t)(k*NV+b)*VV_K+(c*NV+d)] = H_OVVV(k,c,b,d);
        real_t *dB=nullptr,*dB2=nullptr,*dAo=nullptr,*dAv=nullptr,*dCo=nullptr,*dCv=nullptr;
        tracked_cudaMalloc(&dB,(size_t)VV_K*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dB2,(size_t)VV_K*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dAo,(size_t)OOkl_M*VV_K*sizeof(real_t));
        tracked_cudaMalloc(&dAv,(size_t)OVkb_M*VV_K*sizeof(real_t));
        tracked_cudaMalloc(&dCo,(size_t)OOkl_M*OO_N*sizeof(real_t));
        tracked_cudaMalloc(&dCv,(size_t)OVkb_M*OO_N*sizeof(real_t));
        cudaMemcpy(dB,hB.data(),(size_t)VV_K*OO_N*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB2,hB2.data(),(size_t)VV_K*OO_N*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dAo,hAo.data(),(size_t)OOkl_M*VV_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dAv,hAv.data(),(size_t)OVkb_M*VV_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OOkl_M,VV_K,&one,dB, OO_N,dAo,VV_K,&zero,dCo,OO_N);
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OVkb_M,VV_K,&one,dB2,OO_N,dAv,VV_K,&zero,dCv,OO_N);
        ct_woooo.assign((size_t)OOkl_M*OO_N,0.0); ct_wovoo.assign((size_t)OVkb_M*OO_N,0.0);
        cudaMemcpy(ct_woooo.data(),dCo,(size_t)OOkl_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        cudaMemcpy(ct_wovoo.data(),dCv,(size_t)OVkb_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dB);tracked_cudaFree(dB2);tracked_cudaFree(dAo);tracked_cudaFree(dAv);tracked_cudaFree(dCo);tracked_cudaFree(dCv);
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
    std::vector<real_t> h_Woooo(oooo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int l = 0; l < NO; ++l)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOO(k,i,l,j);
                    for (int d = 0; d < NV; ++d)
                        v += H_OOOV(k,i,l,d) * H_T1(j,d);
                    for (int c = 0; c < NV; ++c)
                        v += H_OOOV(l,j,k,c) * H_T1(i,c);
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
        {   // ct_ovov: single GEMM C[(k,d),(i,b)] = A[(k,d),(c,l)]·B[(c,l),(i,b)]
            std::vector<real_t> hA((size_t)IP_MO_kd*IP_KO_cl), hB((size_t)IP_KO_cl*IP_NO_ib);
            #pragma omp parallel for collapse(2)
            for (int k=0;k<NO;++k) for (int d=0;d<NV;++d)
                for (int c=0;c<NV;++c) for (int l=0;l<NO;++l)
                    hA[(size_t)(k*NV+d)*IP_KO_cl+(c*NO+l)] = H_OVOV(k,c,l,d);
            #pragma omp parallel for collapse(2)
            for (int c=0;c<NV;++c) for (int l=0;l<NO;++l)
                for (int i=0;i<NO;++i) for (int b=0;b<NV;++b)
                    hB[(size_t)(c*NO+l)*IP_NO_ib+(i*NV+b)] = H_T2(i,l,c,b);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)IP_MO_kd*IP_KO_cl*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)IP_KO_cl*IP_NO_ib*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)IP_MO_kd*IP_NO_ib*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)IP_MO_kd*IP_KO_cl*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hB.data(),(size_t)IP_KO_cl*IP_NO_ib*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,IP_NO_ib,IP_MO_kd,IP_KO_cl,&one,
                        dB,IP_NO_ib,dA,IP_KO_cl,&zero,dC,IP_NO_ib);
            ct_ovov.assign((size_t)IP_MO_kd*IP_NO_ib,0.0);
            cudaMemcpy(ct_ovov.data(),dC,(size_t)IP_MO_kd*IP_NO_ib*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        {   // ct_ovvo: C[(k,c),(Y,X)] = A1·B1 − A2·B2 (accumulated)
            std::vector<real_t> hA1((size_t)IP_M_kc*IP_K_ld), hB1((size_t)IP_K_ld*IP_N_yx),
                                hA2((size_t)IP_M_kc*IP_K_ld), hB2((size_t)IP_K_ld*IP_N_yx);
            #pragma omp parallel for collapse(2)
            for (int k=0;k<NO;++k) for (int c=0;c<NV;++c)
                for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                    const size_t o=(size_t)(k*NV+c)*IP_K_ld+(l*NV+d);
                    hA1[o]=2.0*H_OVOV(k,c,l,d)-H_OVOV(k,d,l,c); hA2[o]=H_OVOV(k,c,l,d);
                }
            #pragma omp parallel for collapse(2)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
                for (int Y=0;Y<NO;++Y) for (int X=0;X<NV;++X) {
                    const size_t o=(size_t)(l*NV+d)*IP_N_yx+(Y*NV+X);
                    hB1[o]=H_T2(Y,l,X,d); hB2[o]=H_T2(l,Y,X,d);
                }
            real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA1,(size_t)IP_M_kc*IP_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB1,(size_t)IP_K_ld*IP_N_yx*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)IP_M_kc*IP_K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)IP_K_ld*IP_N_yx*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)IP_M_kc*IP_N_yx*sizeof(real_t));
            cudaMemcpy(dA1,hA1.data(),(size_t)IP_M_kc*IP_K_ld*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB1,hB1.data(),(size_t)IP_K_ld*IP_N_yx*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dA2,hA2.data(),(size_t)IP_M_kc*IP_K_ld*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB2,hB2.data(),(size_t)IP_K_ld*IP_N_yx*sizeof(real_t),cudaMemcpyHostToDevice);
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

    // ============================================================
    //  W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d] t2[i,l,c,b]
    //  W2ovov[k,b,i,d] = -Σ_l Wooov[k,l,i,d] t1[l,b]
    //                   + Σ_c ovvv[k,c,b,d] t1[i,c]
    //  Wovov = W1ovov + W2ovov
    // ============================================================
    std::vector<real_t> h_Wovov(ovov_sz, 0.0);
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
                    for (int c = 0; c < NV; ++c)
                        v += H_OVVV(k,c,b,d) * H_T1(i,c);              // W2: +ovvv·t1
                    h_Wovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
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
    std::vector<real_t> h_Wovvo(ovvo_sz, 0.0);
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
                    for (int d = 0; d < NV; ++d)
                        v += H_OVVV(k,c,a,d) * H_T1(i,d);
                    h_Wovvo[(((size_t)k * NV + a) * NV + c) * NO + i] = v;
                }

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

    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;
    // GPU GEMM port of Wovoo Σ_ld ooov·t2 (NO⁴·NV²), mirrors STEOM:
    //   ct[k,i,j,b] = Σ_{l,d}[(2·ooov(k,i,l,d)−ooov(l,i,k,d))·t2(l,j,d,b) − ooov(k,i,l,d)·t2(j,l,d,b)]
    const int KI_M = NO*NO, JB_N = NO*NV, LD_K2 = NO*NV;
    std::vector<real_t> ct_wovoo_t2;
    bool wovoo_t2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA13((size_t)KI_M*LD_K2), hBlj((size_t)LD_K2*JB_N),
                            hA2((size_t)KI_M*LD_K2), hB2((size_t)LD_K2*JB_N);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int i=0;i<NO;++i)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(k*NO+i)*LD_K2+(l*NV+d);
                hA13[o]=2.0*H_OOOV(k,i,l,d)-H_OOOV(l,i,k,d); hA2[o]=H_OOOV(k,i,l,d);
            }
        #pragma omp parallel for collapse(2)
        for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
            for (int j=0;j<NO;++j) for (int b=0;b<NV;++b) {
                const size_t o=(size_t)(l*NV+d)*JB_N+(j*NV+b);
                hBlj[o]=H_T2(l,j,d,b); hB2[o]=H_T2(j,l,d,b);
            }
        real_t *dA13=nullptr,*dBlj=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA13,(size_t)KI_M*LD_K2*sizeof(real_t));
        tracked_cudaMalloc(&dBlj,(size_t)LD_K2*JB_N*sizeof(real_t));
        tracked_cudaMalloc(&dA2,(size_t)KI_M*LD_K2*sizeof(real_t));
        tracked_cudaMalloc(&dB2,(size_t)LD_K2*JB_N*sizeof(real_t));
        tracked_cudaMalloc(&dC, (size_t)KI_M*JB_N*sizeof(real_t));
        cudaMemcpy(dA13,hA13.data(),(size_t)KI_M*LD_K2*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dBlj,hBlj.data(),(size_t)LD_K2*JB_N*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dA2,hA2.data(),(size_t)KI_M*LD_K2*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB2,hB2.data(),(size_t)LD_K2*JB_N*sizeof(real_t),cudaMemcpyHostToDevice);
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
        std::vector<real_t> hA((size_t)M_w6*K_w6), hB((size_t)K_w6*N_w6);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int j=0;j<NO;++j)
            for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
                hA[(size_t)(k*NO+j)*K_w6 + (l*NV+c)] = H_OOOV(l,j,k,c);
        #pragma omp parallel for collapse(2)
        for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
            for (int b=0;b<NV;++b) for (int i=0;i<NO;++i)
                hB[(size_t)(l*NV+c)*N_w6 + (b*NO+i)] = H_T2(l,i,b,c);
        ct_wovoo_6.assign((size_t)M_w6*N_w6, 0.0);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)M_w6*K_w6*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)K_w6*N_w6*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)M_w6*N_w6*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)M_w6*K_w6*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)K_w6*N_w6*sizeof(real_t),cudaMemcpyHostToDevice);
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
    tracked_cudaMalloc(&d_Woooo_, oooo_sz  * sizeof(real_t));
    cudaMemcpy(d_Woooo_, h_Woooo.data(), oooo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wooov_, ooov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wooov_, h_Wooov.data(), ooov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovov_, ovov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovov_, h_Wovov.data(), ovov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovvo_, ovvo_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovvo_, h_Wovvo.data(), ovvo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovoo_, wovoo_sz * sizeof(real_t));
    cudaMemcpy(d_Wovoo_, h_Wovoo.data(), wovoo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

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
        d_tmp_c, t2, d_r1, d_r2, d_s2, nocc_, nvir_, i_begin, i_end);

    if (!scr_tmp_c) tracked_cudaFree(d_tmp_c);
}

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
       << "    ‖Wovov‖     = " << frobenius_norm_device(d_Wovov_, ovov_sz)  << "\n"
       << "    ‖Wovvo‖     = " << frobenius_norm_device(d_Wovvo_, ovvo_sz)  << "\n"
       << "    ‖Wovoo‖     = " << frobenius_norm_device(d_Wovoo_, wovoo_sz) << "\n";
}

} // namespace gansu
