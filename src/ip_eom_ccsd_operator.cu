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
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nocc * nocc * nvir;
    if (idx >= total) return;
    int a = idx % nvir;
    int t = idx / nvir;
    int j = t % nocc;
    int i = t / nocc;

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

    sigma2[idx] = s;
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
    int nocc, int nvir, int nao)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      h_dim_(nocc),
      h2p_dim_(nocc * nocc * nvir),
      total_dim_(nocc + nocc * nocc * nvir),
      d_t1_(d_t1), d_t2_(d_t2)
{
    if (nocc <= 0 || nvir <= 0 || nao != nocc + nvir) {
        throw std::invalid_argument(
            "IPEOMCCSDOperator: invalid (nocc, nvir, nao) — require nao == nocc + nvir, both positive");
    }

    compute_denominators_and_fock(d_orbital_energies);
    build_diagonal();
    if (d_eri_mo != nullptr) {
        extract_eri_blocks(d_eri_mo);
        build_dressed_intermediates();
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
    std::vector<real_t> h_Woooo(oooo_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int l = 0; l < NO; ++l)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOO(k,i,l,j);
                    for (int d = 0; d < NV; ++d)
                        v += H_OOOV(k,i,l,d) * H_T1(j,d);
                    for (int c = 0; c < NV; ++c)
                        v += H_OOOV(l,j,k,c) * H_T1(i,c);
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            v += kcld * (H_T2(i,j,c,d) + H_T1(i,c) * H_T1(j,d));
                        }
                    h_Woooo[(((size_t)k * NO + l) * NO + i) * NO + j] = v;
                }

    // ============================================================
    //  W1ovov[k,b,i,d] = oovv[k,i,b,d] - Σ_{c,l} ovov[k,c,l,d] t2[i,l,c,b]
    //  W2ovov[k,b,i,d] = -Σ_l Wooov[k,l,i,d] t1[l,b]
    //                   + Σ_c ovvv[k,c,b,d] t1[i,c]
    //  Wovov = W1ovov + W2ovov
    // ============================================================
    std::vector<real_t> h_Wovov(ovov_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);                       // W1: oovv
                    for (int c = 0; c < NV; ++c)
                        for (int l = 0; l < NO; ++l)
                            v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);     // W1: -ovov·t2
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
    for (int k = 0; k < NO; ++k)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c)
                for (int i = 0; i < NO; ++i) {
                    real_t v = H_OVVO(k,c,a,i);
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            v += 2.0 * kcld * H_T2(i,l,a,d);
                            v -=       kcld * H_T2(l,i,a,d);
                            v -= H_OVOV(k,d,l,c) * H_T2(i,l,a,d);
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

    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;
    std::vector<real_t> h_Wovoo(wovoo_sz, 0.0);
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
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            v += 2.0 * H_OOOV(k,i,l,d) * H_T2(l,j,d,b);
                            v -=       H_OOOV(k,i,l,d) * H_T2(j,l,d,b);
                            v -=       H_OOOV(l,i,k,d) * H_T2(l,j,d,b);
                        }
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d) {
                            v += H_OVVV(k,c,b,d) * H_T2(j,i,d,c);
                            v += H_OVVV(k,c,b,d) * H_T1(j,d) * H_T1(i,c);
                        }
                    for (int c = 0; c < NV; ++c)
                        for (int l = 0; l < NO; ++l)
                            v -= H_OOOV(l,j,k,c) * H_T2(l,i,b,c);
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
        const int threads = 256;

        // σ1
        const int blocks_1 = (h_dim_ + threads - 1) / threads;
        ip_eom_sigma1_full_kernel<<<blocks_1, threads>>>(
            d_Loo_, d_Fov_, d_Wooov_, d_r1, d_r2, d_s1, nocc_, nvir_);

        // Pre-stage: tmp[c]
        real_t* d_tmp_c = nullptr;
        tracked_cudaMalloc(&d_tmp_c, (size_t)nvir_ * sizeof(real_t));
        const int blocks_tmp = (nvir_ + threads - 1) / threads;
        ip_eom_sigma2_tmp_c_kernel<<<blocks_tmp, threads>>>(
            d_eri_oovv_, d_r2, d_tmp_c, nocc_, nvir_);

        // σ2
        const int blocks_2 = (h2p_dim_ + threads - 1) / threads;
        ip_eom_sigma2_full_kernel<<<blocks_2, threads>>>(
            d_Loo_, d_Lvv_, d_Woooo_, d_Wovov_, d_Wovvo_, d_Wovoo_,
            d_tmp_c, d_t2_, d_r1, d_r2, d_s2, nocc_, nvir_);

        tracked_cudaFree(d_tmp_c);
#endif
    }
}

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
