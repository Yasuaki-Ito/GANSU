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

#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
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
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nocc * nvir * nvir;
    if (idx >= total) return;
    int b = idx % nvir;
    int t = idx / nvir;
    int a = t % nvir;
    int j = t / nvir;

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

    // +Σ_{l,d} (2 Wovvo[l,b,d,j] - Wovov[l,b,j,d]) r2[l,a,d]
    //   Wovvo[l,b,d,j] layout: ((l*nvir+b)*nvir+d)*nocc + j
    //   Wovov[l,b,j,d] layout: ((l*nvir+b)*nocc+j)*nvir + d
    for (int l = 0; l < nocc; ++l)
        for (int d = 0; d < nvir; ++d) {
            real_t wovvo = Wovvo[(((size_t)l * nvir + b) * nvir + d) * nocc + j];
            real_t wovov = Wovov[(((size_t)l * nvir + b) * nocc + j) * nvir + d];
            s += (2.0 * wovvo - wovov) * r2[((size_t)l * nvir + a) * nvir + d];
        }

    // -Σ_{l,c} Wovov[l,a,j,c] r2[l,c,b]
    for (int l = 0; l < nocc; ++l)
        for (int c = 0; c < nvir; ++c) {
            real_t w = Wovov[(((size_t)l * nvir + a) * nocc + j) * nvir + c];
            s -= w * r2[((size_t)l * nvir + c) * nvir + b];
        }

    // -Σ_{l,c} Wovvo[l,b,c,j] r2[l,c,a]
    for (int l = 0; l < nocc; ++l)
        for (int c = 0; c < nvir; ++c) {
            real_t w = Wovvo[(((size_t)l * nvir + b) * nvir + c) * nocc + j];
            s -= w * r2[((size_t)l * nvir + c) * nvir + a];
        }

    // +Σ_{c,d} Wvvvv[a,b,c,d] r2[j,c,d]    layout: ((a*nvir+b)*nvir+c)*nvir + d
    for (int c = 0; c < nvir; ++c)
        for (int d = 0; d < nvir; ++d) {
            s += Wvvvv[(((size_t)a * nvir + b) * nvir + c) * nvir + d]
                 * r2[((size_t)j * nvir + c) * nvir + d];
        }

    // -Σ_k tmp[k] · t2[k,j,a,b]    t2 layout: ((k*nocc+j)*nvir + a)*nvir + b
    for (int k = 0; k < nocc; ++k) {
        s -= d_tmp_k[k] * t2[(((size_t)k * nocc + j) * nvir + a) * nvir + b];
    }

    sigma2[idx] = s;
}
#endif  // !GANSU_CPU_ONLY


// ==================================================================
//  Constructor / destructor
// ==================================================================

EAEOMCCSDOperator::EAEOMCCSDOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    int nocc, int nvir, int nao)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      p_dim_(nvir),
      p2h_dim_(nocc * nvir * nvir),
      total_dim_(nvir + nocc * nvir * nvir),
      d_t1_(d_t1), d_t2_(d_t2)
{
    if (nocc <= 0 || nvir <= 0 || nao != nocc + nvir) {
        throw std::invalid_argument(
            "EAEOMCCSDOperator: invalid (nocc, nvir, nao) — require nao == nocc + nvir, both positive");
    }

    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
    // Sub-phase 2.2: when d_eri_mo is provided, extract the EA-needed MO ERI
    // blocks and build PySCF-equivalent dressed intermediates.
    // (Unit tests pass d_eri_mo == nullptr and exercise only the diagonal.)
    if (d_eri_mo != nullptr) {
        extract_eri_blocks(d_eri_mo);
        build_dressed_intermediates();
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
    if (d_Loo_)       tracked_cudaFree(d_Loo_);
    if (d_Lvv_)       tracked_cudaFree(d_Lvv_);
    if (d_Fov_)       tracked_cudaFree(d_Fov_);
    if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
    if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
    if (d_Wvovv_)     tracked_cudaFree(d_Wvovv_);
    if (d_Wvvvv_)     tracked_cudaFree(d_Wvvvv_);
    if (d_Wvvvo_)     tracked_cudaFree(d_Wvvvo_);
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
    tracked_cudaMalloc(&d_eri_vvvv_, vvvv_sz * sizeof(real_t));

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

    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz), h_vvvv(vvvv_sz);

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_,  ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vvvv.data(), d_eri_vvvv_,  vvvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    // ============================================================
    //  cc_Fov[k,c] = fov + 2 ovov[k,c,l,d] t1[l,d] - ovov[k,d,l,c] t1[l,d]
    // ============================================================
    std::vector<real_t> h_Fov(t1_sz, 0.0);
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
    //  cc_Foo / cc_Fvv (internal helpers for Loo / Lvv)
    // ============================================================
    std::vector<real_t> h_ccFoo(NO * NO, 0.0);
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

    std::vector<real_t> h_ccFvv(NV * NV, 0.0);
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
    //  Loo[k,i] = cc_Foo + Fov·t1 + 2 ooov·t1 - ooov·t1 (PySCF Loo)
    //  Lvv[a,c] = cc_Fvv - Fov·t1 + 2 ovvv·t1 - ovvv·t1 (PySCF Lvv)
    // ============================================================
    std::vector<real_t> h_Loo(NO * NO, 0.0);
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
    //  W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d] t2[i,l,c,b]
    //  W2ovov[k,b,i,d] = -Σ_l Wooov[k,l,i,d] t1[l,b]
    //                  +  Σ_c ovvv[k,c,b,d] t1[i,c]
    //  Wovov = W1ovov + W2ovov                    (PySCF Wovov)
    // Where Wooov[k,l,i,d] = ooov[k,i,l,d] + Σ_c t1[i,c] ovov[k,c,l,d]
    // ============================================================
    std::vector<real_t> h_Wooov(ooov_sz, 0.0);
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

    std::vector<real_t> h_W1ovov(ovov_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);
                    for (int c = 0; c < NV; ++c)
                        for (int l = 0; l < NO; ++l)
                            v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                    h_W1ovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }
    #define H_W1OVOV(k,b,i,d) h_W1ovov[(((size_t)(k) * NV + (b)) * NO + (i)) * NV + (d)]

    std::vector<real_t> h_Wovov(ovov_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_W1OVOV(k,b,i,d);                       // W1
                    for (int l = 0; l < NO; ++l)
                        v -= H_WOOOV(k,l,i,d) * H_T1(l,b);               // W2: -Wooov·t1
                    for (int c = 0; c < NV; ++c)
                        v += H_OVVV(k,c,b,d) * H_T1(i,c);                // W2: +ovvv·t1
                    h_Wovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }

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
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVO(k,c,b,j);
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            v += 2.0 * kcld * H_T2(j,l,b,d);
                            v -=       kcld * H_T2(l,j,b,d);
                            v -= H_OVOV(k,d,l,c) * H_T2(j,l,b,d);
                        }
                    h_W1ovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }
    #define H_W1OVVO(k,b,c,j) h_W1ovvo[(((size_t)(k) * NV + (b)) * NV + (c)) * NO + (j)]

    std::vector<real_t> h_Wovvo(ovvo_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_W1OVVO(k,b,c,j);
                    for (int l = 0; l < NO; ++l)
                        v -= H_T1(l,b) * H_WOOOV(l,k,j,c);
                    for (int d = 0; d < NV; ++d)
                        v += H_OVVV(k,c,b,d) * H_T1(j,d);
                    h_Wovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }

    // ============================================================
    //  Wvovv[a,l,c,d] = -Σ_k t1[k,a] ovov[k,c,l,d] + ovvv[l,d,a,c]
    //  (PySCF: 'ka,kcld->alcd' with -t1, then + ovvv.transpose(2,0,3,1).)
    //  layout: [a, l, c, d]  index = ((a*NO + l)*NV + c)*NV + d
    // ============================================================
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    std::vector<real_t> h_Wvovv(wvovv_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int l = 0; l < NO; ++l)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OVVV(l,d,a,c);  // bare ovvv chemist (ld|ac)
                    for (int k = 0; k < NO; ++k)
                        v -= H_T1(k,a) * H_OVOV(k,c,l,d);
                    h_Wvovv[(((size_t)a * NO + l) * NV + c) * NV + d] = v;
                }

    // ============================================================
    //  Wvvvv[a,b,c,d] = + Σ_{k,l} ovov[k,c,l,d] t2[k,l,a,b]
    //                 + Σ_{k,l} ovov[k,c,l,d] t1[k,a] t1[l,b]
    //                 + (ac|bd)                       (chemist→physicist of vvvv)
    //                 - Σ_l ovvv[l,d,a,c] t1[l,b]
    //                 - Σ_k ovvv[k,c,b,d] t1[k,a]
    //  layout: [a, b, c, d]  index = ((a*NV + b)*NV + c)*NV + d
    // ============================================================
    std::vector<real_t> h_Wvvvv(vvvv_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_VVVV(a,c,b,d);  // (ac|bd) chemist → Wabcd physicist
                    for (int k = 0; k < NO; ++k)
                        v -= H_OVVV(k,c,b,d) * H_T1(k,a);
                    for (int l = 0; l < NO; ++l)
                        v -= H_OVVV(l,d,a,c) * H_T1(l,b);
                    for (int k = 0; k < NO; ++k)
                        for (int l = 0; l < NO; ++l) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            v += kcld * (H_T2(k,l,a,b) + H_T1(k,a) * H_T1(l,b));
                        }
                    h_Wvvvv[(((size_t)a * NV + b) * NV + c) * NV + d] = v;
                }
    #define H_WVVVV(a,b,c,d) h_Wvvvv[(((size_t)(a) * NV + (b)) * NV + (c)) * NV + (d)]

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
    const size_t wvvvo_sz = (size_t)NV * NV * NV * NO;
    std::vector<real_t> h_Wvvvo(wvvvo_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVV(j,b,c,a);  // bare: ovvv[j,b,c,a] = (jb|ca) chemist
                    for (int l = 0; l < NO; ++l)
                        v -= H_W1OVOV(l,a,j,c) * H_T1(l,b);
                    for (int k = 0; k < NO; ++k)
                        v -= H_W1OVVO(k,b,c,j) * H_T1(k,a);
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            v += 2.0 * H_OVVV(l,d,a,c) * H_T2(l,j,d,b);
                            v -=       H_OVVV(l,d,a,c) * H_T2(l,j,b,d);
                            v -=       H_OVVV(l,c,a,d) * H_T2(l,j,d,b);
                        }
                    for (int k = 0; k < NO; ++k)
                        for (int d = 0; d < NV; ++d)
                            v -= H_OVVV(k,c,b,d) * H_T2(j,k,d,a);
                    for (int k = 0; k < NO; ++k)
                        for (int l = 0; l < NO; ++l) {
                            real_t klc_lj = H_OOOV(l,j,k,c);  // ovoo[k,c,l,j] = ooov[l,j,k,c]
                            v += klc_lj * H_T2(l,k,b,a);
                            v += klc_lj * H_T1(l,b) * H_T1(k,a);
                        }
                    for (int k = 0; k < NO; ++k)
                        v -= h_Fov[k*NV + c] * H_T2(k,j,a,b);
                    for (int d = 0; d < NV; ++d)
                        v += H_WVVVV(a,b,c,d) * H_T1(j,d);
                    h_Wvvvo[(((size_t)a * NV + b) * NV + c) * NO + j] = v;
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
    tracked_cudaMalloc(&d_Wovov_, ovov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovov_, h_Wovov.data(), ovov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovvo_, ovvo_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovvo_, h_Wovvo.data(), ovvo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvovv_, wvovv_sz * sizeof(real_t));
    cudaMemcpy(d_Wvovv_, h_Wvovv.data(), wvovv_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvvvv_, vvvv_sz  * sizeof(real_t));
    cudaMemcpy(d_Wvvvv_, h_Wvvvv.data(), vvvv_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvvvo_, wvvvo_sz * sizeof(real_t));
    cudaMemcpy(d_Wvvvo_, h_Wvvvo.data(), wvvvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

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
        const int threads = 256;

        // σ1
        const int blocks_1 = (p_dim_ + threads - 1) / threads;
        ea_eom_sigma1_full_kernel<<<blocks_1, threads>>>(
            d_Lvv_, d_Fov_, d_Wvovv_, d_r1, d_r2, d_s1, nocc_, nvir_);

        // Pre-stage: tmp[k]
        real_t* d_tmp_k = nullptr;
        tracked_cudaMalloc(&d_tmp_k, (size_t)nocc_ * sizeof(real_t));
        const int blocks_tmp = (nocc_ + threads - 1) / threads;
        ea_eom_sigma2_tmp_k_kernel<<<blocks_tmp, threads>>>(
            d_eri_ovov_, d_r2, d_tmp_k, nocc_, nvir_);

        // σ2
        const int blocks_2 = (p2h_dim_ + threads - 1) / threads;
        ea_eom_sigma2_full_kernel<<<blocks_2, threads>>>(
            d_Lvv_, d_Loo_, d_Wovov_, d_Wovvo_, d_Wvvvv_, d_Wvvvo_,
            d_tmp_k, d_t2_, d_r1, d_r2, d_s2, nocc_, nvir_);

        tracked_cudaFree(d_tmp_k);
#endif
    }
}

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
       << "    ‖Wvvvv‖     = " << frobenius_norm_device(d_Wvvvv_, wvvvv_sz) << "\n"
       << "    ‖Wvvvo‖     = " << frobenius_norm_device(d_Wvvvo_, wvvvo_sz) << "\n";
}

} // namespace gansu
