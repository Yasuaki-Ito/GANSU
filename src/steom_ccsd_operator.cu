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
 * @file steom_ccsd_operator.cu
 * @brief STEOM-CCSD operator — sub-phase 3.0-3.4.
 *
 *   3.0+3.1   LinearOperator scaffolding + diagonal-only apply() stub.
 *   3.2       bar-H rebuild: 7 MO ERI blocks + 11 PySCF rintermediates-equivalent
 *             dressed intermediates (Loo, Lvv, Fov, Woooo, Wooov, Wovov,
 *             Wovvo, Wovoo, Wvovv, Wvvvv, Wvvvo).
 *   3.3a     Ŝ amplitude single-divide normalization (R2/(U·R1)). Superseded
 *             by sub-phase 3.4 X-matrix (matrix inverse).
 *   3.4       F^eff_oo dressing per CFOUR `gmi_steom_rhf`:
 *               • build X(MI) = inverse of active R1 matrix (n_act × n_act)
 *               • build U(M,I) = +2 Fov·R2 − Fov·R2 − 2 Wooov·R2 + Wooov·R2
 *                 (PySCF σ1-style spin-adapted from IP-EOM matvec)
 *               • F^eff_oo[M_idx, I] = Loo[M_idx, I] − Σ_N U(N,I) · X(N,M)
 *               • inactive rows of F^eff_oo = bar Loo (no dressing)
 *             Reference: megansimons/steom_ccsd-ct (GitHub) — CFOUR-style
 *             closed-shell RHF STEOM, `steom_intermediates.cxx` lines 7-81 +
 *             `steom.cxx` lines 23-49 (`renormalize`).
 *
 * Sub-phase 3.5+:  W^eff dressing + G(EM)/G(Mn,Ie) intermediates.
 * Sub-phase 3.6:   Cross IP×EA dressing.
 * Sub-phase 3.7:   Full G^{1h1p} matvec assembly (★ H2O sto-3g 1 mHa gate).
 */

#include "steom_ccsd_operator.hpp"

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

// MO ERI block extraction kernels are shared with EE/IP/EA-EOM modules.
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
__global__ void steom_build_diagonal_kernel(
    const real_t* __restrict__ eps_occ, const real_t* __restrict__ eps_vir,
    real_t* __restrict__ D, int nocc_active, int nvir)
{
    const int total = nocc_active * nvir;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int a = idx % nvir;
    int i = idx / nvir;
    D[idx] = eps_vir[a] - eps_occ[i];
}

__global__ void steom_diag_matvec_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = D[idx] * x[idx];
}

__global__ void steom_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}

// Dense (non-symmetric) matvec y = G x, G row-major [n×n] (row*n + col).
__global__ void steom_dense_matvec_kernel(
    const real_t* __restrict__ G, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        real_t s = 0.0;
        const real_t* Grow = G + (size_t)row * n;
        for (int c = 0; c < n; ++c) s += Grow[c] * x[c];
        y[row] = s;
    }
}
#endif  // !GANSU_CPU_ONLY


// ==================================================================
//  Constructor / destructor
// ==================================================================

STEOMCCSDOperator::STEOMCCSDOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    const real_t* h_R2_IP_amplitudes,
    const real_t* h_R2_EA_amplitudes,
    const real_t* h_R1_IP_amplitudes,
    const real_t* h_R1_EA_amplitudes,
    const int* h_active_occ_idx,
    const int* h_active_vir_idx,
    int nocc_active, int nvir, int nao_active,
    int n_act_occ, int n_act_vir)
    : nocc_active_(nocc_active), nvir_(nvir), nao_active_(nao_active),
      n_act_occ_(n_act_occ), n_act_vir_(n_act_vir),
      total_dim_(nocc_active * nvir),
      d_t1_(d_t1), d_t2_(d_t2)
{
    if (nocc_active <= 0 || nvir <= 0 || nao_active != nocc_active + nvir) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: invalid (nocc_active, nvir, nao_active) — require "
            "nao_active == nocc_active + nvir, both positive");
    }
    if (n_act_occ <= 0 || n_act_vir <= 0) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: n_act_occ and n_act_vir must both be positive — "
            "did P1 IP-EOM and P2 EA-EOM run their active-mode routing?");
    }
    if (h_R2_IP_amplitudes == nullptr || h_R2_EA_amplitudes == nullptr) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: R2 amplitude inputs must be non-null host pointers");
    }

    // Copy raw R2 amplitudes from host to device.
    const size_t r2_ip_sz = (size_t)n_act_occ * nocc_active * nocc_active * nvir;
    const size_t r2_ea_sz = (size_t)n_act_vir * nocc_active * nvir * nvir;
    tracked_cudaMalloc(&d_R2_IP_, r2_ip_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_R2_EA_, r2_ea_sz * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < r2_ip_sz; ++i) d_R2_IP_[i] = h_R2_IP_amplitudes[i];
        for (size_t i = 0; i < r2_ea_sz; ++i) d_R2_EA_[i] = h_R2_EA_amplitudes[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_R2_IP_, h_R2_IP_amplitudes, r2_ip_sz * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R2_EA_, h_R2_EA_amplitudes, r2_ea_sz * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    build_diagonal(d_orbital_energies);

    // Sub-phase 3.4: full pipeline — bar-H + X(MI) + F^eff_oo.
    if (d_eri_mo != nullptr) {
        if (d_t1 == nullptr || d_t2 == nullptr) {
            throw std::invalid_argument(
                "STEOMCCSDOperator: d_eri_mo non-null requires both d_t1 and d_t2 non-null");
        }
        if (h_R1_IP_amplitudes == nullptr || h_R1_EA_amplitudes == nullptr ||
            h_active_occ_idx == nullptr || h_active_vir_idx == nullptr) {
            throw std::invalid_argument(
                "STEOMCCSDOperator: sub-phase 3.4 F^eff_oo build requires non-null "
                "h_R1_IP_amplitudes, h_R1_EA_amplitudes, h_active_occ_idx, h_active_vir_idx");
        }
        // Store active NTO ↔ MO index maps (host-side, used by build_F_eff_oo).
        active_occ_idx_.assign(h_active_occ_idx, h_active_occ_idx + n_act_occ);
        active_vir_idx_.assign(h_active_vir_idx, h_active_vir_idx + n_act_vir);
        for (int m = 0; m < n_act_occ; ++m) {
            if (active_occ_idx_[m] < 0 || active_occ_idx_[m] >= nocc_active) {
                throw std::invalid_argument(
                    "STEOMCCSDOperator: active_occ_idx out of range");
            }
        }
        for (int e = 0; e < n_act_vir; ++e) {
            if (active_vir_idx_[e] < 0 || active_vir_idx_[e] >= nvir) {
                throw std::invalid_argument(
                    "STEOMCCSDOperator: active_vir_idx out of range");
            }
        }
        // Upload R1 amplitudes to device.
        const size_t r1_ip_sz = (size_t)n_act_occ * nocc_active;
        const size_t r1_ea_sz = (size_t)n_act_vir * nvir;
        tracked_cudaMalloc(&d_R1_IP_, r1_ip_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_R1_EA_, r1_ea_sz * sizeof(real_t));
        if (!gpu::gpu_available()) {
            for (size_t i = 0; i < r1_ip_sz; ++i) d_R1_IP_[i] = h_R1_IP_amplitudes[i];
            for (size_t i = 0; i < r1_ea_sz; ++i) d_R1_EA_[i] = h_R1_EA_amplitudes[i];
        } else {
#ifndef GANSU_CPU_ONLY
            cudaMemcpy(d_R1_IP_, h_R1_IP_amplitudes, r1_ip_sz * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_R1_EA_, h_R1_EA_amplitudes, r1_ea_sz * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
        }

        // Need Fock diagonals from orbital energies (mirrors P1/P2 pattern).
        tracked_cudaMalloc(&d_f_oo_, (size_t)nocc_active_ * sizeof(real_t));
        tracked_cudaMalloc(&d_f_vv_, (size_t)nvir_        * sizeof(real_t));
        if (!gpu::gpu_available()) {
            for (int i = 0; i < nocc_active_; ++i) d_f_oo_[i] = d_orbital_energies[i];
            for (int a = 0; a < nvir_; ++a)         d_f_vv_[a] = d_orbital_energies[a + nocc_active_];
        } else {
#ifndef GANSU_CPU_ONLY
            cudaMemcpy(d_f_oo_, d_orbital_energies,                 (size_t)nocc_active_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_f_vv_, d_orbital_energies + nocc_active_,  (size_t)nvir_        * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
        }
        extract_eri_blocks(d_eri_mo);
        build_dressed_intermediates();

        // Sub-phase 3.4: build X(MI), X(EA) then F^eff_oo + F^eff_vv per
        // CFOUR `gmi_steom_rhf` / `gea_steom_rhf`.
        build_x_matrices(h_R1_IP_amplitudes, h_R1_EA_amplitudes);
        build_F_eff_oo();
        build_F_eff_vv();
        // Sub-phase 3.5-3.7: full W^eff dressing + dense G^{1h1p}.
        build_W_eff_and_G();
    }
}

STEOMCCSDOperator::~STEOMCCSDOperator() {
    if (d_G_)         tracked_cudaFree(d_G_);
    if (d_R2_IP_)     tracked_cudaFree(d_R2_IP_);
    if (d_R2_EA_)     tracked_cudaFree(d_R2_EA_);
    if (d_R1_IP_)     tracked_cudaFree(d_R1_IP_);
    if (d_R1_EA_)     tracked_cudaFree(d_R1_EA_);
    if (d_X_IP_)      tracked_cudaFree(d_X_IP_);
    if (d_X_EA_)      tracked_cudaFree(d_X_EA_);
    if (d_F_eff_oo_)  tracked_cudaFree(d_F_eff_oo_);
    if (d_U_MI_)      tracked_cudaFree(d_U_MI_);
    if (d_F_eff_vv_)  tracked_cudaFree(d_F_eff_vv_);
    if (d_U_EA_)      tracked_cudaFree(d_U_EA_);
    if (d_t1_)        tracked_cudaFree(d_t1_);
    if (d_t2_)        tracked_cudaFree(d_t2_);
    if (d_diagonal_)  tracked_cudaFree(d_diagonal_);
    if (d_eps_occ_)   tracked_cudaFree(d_eps_occ_);
    if (d_eps_vir_)   tracked_cudaFree(d_eps_vir_);
    if (d_f_oo_)      tracked_cudaFree(d_f_oo_);
    if (d_f_vv_)      tracked_cudaFree(d_f_vv_);
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
    if (d_Woooo_)     tracked_cudaFree(d_Woooo_);
    if (d_Wooov_)     tracked_cudaFree(d_Wooov_);
    if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
    if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
    if (d_Wovoo_)     tracked_cudaFree(d_Wovoo_);
    if (d_Wvovv_)     tracked_cudaFree(d_Wvovv_);
    if (d_Wvvvv_)     tracked_cudaFree(d_Wvvvv_);
    if (d_Wvvvo_)     tracked_cudaFree(d_Wvvvo_);
}

void STEOMCCSDOperator::build_diagonal(const real_t* d_orbital_energies) {
    tracked_cudaMalloc(&d_eps_occ_,  (size_t)nocc_active_ * sizeof(real_t));
    tracked_cudaMalloc(&d_eps_vir_,  (size_t)nvir_        * sizeof(real_t));
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_   * sizeof(real_t));

    if (!gpu::gpu_available()) {
        for (int i = 0; i < nocc_active_; ++i) d_eps_occ_[i] = d_orbital_energies[i];
        for (int a = 0; a < nvir_; ++a)         d_eps_vir_[a] = d_orbital_energies[a + nocc_active_];
        for (int i = 0; i < nocc_active_; ++i)
            for (int a = 0; a < nvir_; ++a)
                d_diagonal_[i * nvir_ + a] = d_eps_vir_[a] - d_eps_occ_[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_eps_occ_, d_orbital_energies,                (size_t)nocc_active_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_eps_vir_, d_orbital_energies + nocc_active_, (size_t)nvir_        * sizeof(real_t), cudaMemcpyDeviceToDevice);
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        steom_build_diagonal_kernel<<<blocks, threads>>>(d_eps_occ_, d_eps_vir_, d_diagonal_, nocc_active_, nvir_);
        cudaDeviceSynchronize();
#endif
    }
}


// ==================================================================
//  apply / apply_preconditioner — diagonal-only stubs (sub-phase 3.0+3.1)
// ==================================================================
void STEOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    // Sub-phase 3.5-3.7: dense G^{1h1p} matvec when built; else diagonal stub.
    if (d_G_ != nullptr) {
        if (!gpu::gpu_available()) {
            #pragma omp parallel for
            for (int row = 0; row < total_dim_; ++row) {
                real_t s = 0.0;
                const real_t* Grow = d_G_ + (size_t)row * total_dim_;
                for (int c = 0; c < total_dim_; ++c) s += Grow[c] * d_input[c];
                d_output[row] = s;
            }
        } else {
#ifndef GANSU_CPU_ONLY
            const int threads = 256;
            const int blocks  = (total_dim_ + threads - 1) / threads;
            steom_dense_matvec_kernel<<<blocks, threads>>>(d_G_, d_input, d_output, total_dim_);
#endif
        }
        return;
    }
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; ++idx) {
            d_output[idx] = d_diagonal_[idx] * d_input[idx];
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        steom_diag_matvec_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

void STEOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        steom_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}


// ==================================================================
//  extract_eri_blocks — union of IP + EA needs (7 blocks)
//  Literal copy of P2's extract_eri_blocks (which already includes the
//  IP-required blocks plus vvvv).
// ==================================================================
void STEOMCCSDOperator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_active_, nvir = nvir_, nao = nao_active_;
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
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_active_, nvir_, nao_active_);
        blocks = (ooov_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_active_, nvir_, nao_active_);
        blocks = (oooo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_active_, nao_active_, 0);
        blocks = (oovv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_active_, nvir_, nao_active_, 0, -1);
        blocks = (ovvo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_active_, nvir_, nao_active_);
        blocks = (ovvv_sz + threads - 1) / threads;
        eom_ccsd_extract_eri_ovvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvv_, nocc_active_, nvir_, nao_active_);
        blocks = (vvvv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_active_, nvir_, nao_active_, 0, -1);
        cudaDeviceSynchronize();
#endif
    }
}


// ==================================================================
//  build_dressed_intermediates — union of IP + EA versions
//  (literal copy from src/ip_eom_ccsd_operator.cu + src/ea_eom_ccsd_operator.cu;
//  the 11 intermediates Loo, Lvv, Fov, Woooo, Wooov, Wovov, Wovvo, Wovoo,
//  Wvovv, Wvvvv, Wvvvo are all required by the STEOM G^{1h1p} matvec —
//  sub-phases 3.4-3.7).
// ==================================================================
#define H_OVOV(p,a,q,b) h_ovov[(size_t)(p)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(q)*NV + (b)]
#define H_OOOV(p,q,r,a) h_ooov[(size_t)(p)*NO*NO*NV + (size_t)(q)*NO*NV + (size_t)(r)*NV + (a)]
#define H_OOVV(p,q,a,b) h_oovv[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
#define H_OVVO(p,a,b,q) h_ovvo[(size_t)(p)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(b)*NO + (q)]
#define H_OVVV(p,a,b,c) h_ovvv[(size_t)(p)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
#define H_OOOO(p,q,r,s) h_oooo[(size_t)(p)*NO*NO*NO + (size_t)(q)*NO*NO + (size_t)(r)*NO + (s)]
#define H_VVVV(a,b,c,d) h_vvvv[(size_t)(a)*NV*NV*NV + (size_t)(b)*NV*NV + (size_t)(c)*NV + (d)]
#define H_T1(p,a)       h_t1[(p)*NV + (a)]
#define H_T2(p,q,a,b)   h_t2[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]

void STEOMCCSDOperator::build_dressed_intermediates() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t t1_sz   = (size_t)NO * NV;
    const size_t t2_sz   = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t oovv_sz = (size_t)NO * NO * NV * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;
    const size_t vvvv_sz = (size_t)NV * NV * NV * NV;

    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz), h_oooo(oooo_sz);
    std::vector<real_t> h_vvvv(vvvv_sz);

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_,  ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oooo.data(), d_eri_oooo_,  oooo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vvvv.data(), d_eri_vvvv_,  vvvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    // cc_Fov, cc_Foo, cc_Fvv — helpers for Loo/Lvv
    std::vector<real_t> h_Fov(t1_sz, 0.0), h_ccFoo(NO * NO, 0.0), h_ccFvv(NV * NV, 0.0);
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

    // Loo / Lvv
    std::vector<real_t> h_Loo(NO * NO, 0.0), h_Lvv(NV * NV, 0.0);
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

    // Wooov[k,l,i,d] (IP version, also used in Woooo / Wovoo / Wovvo / Wovov below)
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

    // Woooo (IP version)
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

    // W1ovov (helper) + Wovov = W1 + W2
    std::vector<real_t> h_W1ovov(ovov_sz, 0.0), h_Wovov(ovov_sz, 0.0);
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
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_W1OVOV(k,b,i,d);
                    for (int l = 0; l < NO; ++l)
                        v -= H_WOOOV(k,l,i,d) * H_T1(l,b);
                    for (int c = 0; c < NV; ++c)
                        v += H_OVVV(k,c,b,d) * H_T1(i,c);
                    h_Wovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }

    // W1ovvo (helper) + Wovvo = W1 + W2
    std::vector<real_t> h_W1ovvo(ovvo_sz, 0.0), h_Wovvo(ovvo_sz, 0.0);
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

    // Wovoo (IP-side, full 11-term PySCF Wovoo build — needed in STEOM σ2 coupling)
    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;
    std::vector<real_t> h_Wovoo(wovoo_sz, 0.0);
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOV(i,k,j,b);
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

    // Wvovv (EA-side)
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    std::vector<real_t> h_Wvovv(wvovv_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int l = 0; l < NO; ++l)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OVVV(l,d,a,c);
                    for (int k = 0; k < NO; ++k)
                        v -= H_T1(k,a) * H_OVOV(k,c,l,d);
                    h_Wvovv[(((size_t)a * NO + l) * NV + c) * NV + d] = v;
                }

    // Wvvvv (EA-side)
    std::vector<real_t> h_Wvvvv(vvvv_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_VVVV(a,c,b,d);
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

    // Wvvvo (EA-side, 11-term)
    const size_t wvvvo_sz = (size_t)NV * NV * NV * NO;
    std::vector<real_t> h_Wvvvo(wvvvo_sz, 0.0);
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVV(j,b,c,a);
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
                            real_t klc_lj = H_OOOV(l,j,k,c);
                            v += klc_lj * H_T2(l,k,b,a);
                            v += klc_lj * H_T1(l,b) * H_T1(k,a);
                        }
                    for (int k = 0; k < NO; ++k)
                        v -= h_Fov[k*NV + c] * H_T2(k,j,a,b);
                    for (int d = 0; d < NV; ++d)
                        v += H_WVVVV(a,b,c,d) * H_T1(j,d);
                    h_Wvvvo[(((size_t)a * NV + b) * NV + c) * NO + j] = v;
                }

    // Upload all 11 intermediates to device
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
    tracked_cudaMalloc(&d_Wvovv_, wvovv_sz * sizeof(real_t));
    cudaMemcpy(d_Wvovv_, h_Wvovv.data(), wvovv_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvvvv_, vvvv_sz  * sizeof(real_t));
    cudaMemcpy(d_Wvvvv_, h_Wvvvv.data(), vvvv_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvvvo_, wvvvo_sz * sizeof(real_t));
    cudaMemcpy(d_Wvvvo_, h_Wvvvo.data(), wvvvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    std::cout << "  STEOM-CCSD bar-H intermediates built (PySCF union of IP+EA: 11 quantities)." << std::endl;

    #undef H_WOOOV
    #undef H_W1OVOV
    #undef H_W1OVVO
    #undef H_WVVVV
}


// ==================================================================
//  build_x_matrices — sub-phase 3.4: X(MI) and X(EA) = active R1 inverse.
//
//  CFOUR `renormalize` (steom.cxx lines 23-49):
//    R1_active[m_NTO, n_root] = R1^(n_root)[active_*_idx[m_NTO]]
//    X = inverse of R1_active (n_act × n_act matrix)
//
//  For canonical STEOM (CIS-NTO trans = identity over active subset), this
//  is what GANSU needs. The matrix is tiny (n_act ≤ ~20 typical, ≤ ~10 for
//  H2O/N2 test systems), so a hand-rolled Gauss-Jordan elimination is more
//  than fast enough and avoids pulling in LAPACK for this tiny inversion.
// ==================================================================
namespace {

void invert_small_matrix_inplace(real_t* A, int n) {
    // Gauss-Jordan elimination with partial pivoting on the augmented
    // [A | I] system. Operates in row-major layout. n is expected ≤ ~32.
    std::vector<real_t> aug((size_t)n * 2 * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[(size_t)i * 2 * n + j] = A[(size_t)i * n + j];
        aug[(size_t)i * 2 * n + (n + i)] = real_t(1.0);
    }
    for (int col = 0; col < n; ++col) {
        // Pivot: find row with max |aug[row, col]| at row ≥ col.
        int piv = col;
        real_t best = std::fabs(aug[(size_t)col * 2 * n + col]);
        for (int row = col + 1; row < n; ++row) {
            real_t v = std::fabs(aug[(size_t)row * 2 * n + col]);
            if (v > best) { best = v; piv = row; }
        }
        if (best < 1e-14) {
            throw std::runtime_error(
                "STEOMCCSDOperator::build_x_matrices: active R1 matrix is "
                "singular (likely degenerate roots or pathological NTO mixing)");
        }
        if (piv != col) {
            for (int j = 0; j < 2 * n; ++j)
                std::swap(aug[(size_t)col * 2 * n + j], aug[(size_t)piv * 2 * n + j]);
        }
        // Normalize pivot row.
        real_t inv_pivot = real_t(1.0) / aug[(size_t)col * 2 * n + col];
        for (int j = 0; j < 2 * n; ++j) aug[(size_t)col * 2 * n + j] *= inv_pivot;
        // Eliminate other rows.
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            real_t factor = aug[(size_t)row * 2 * n + col];
            if (factor == 0.0) continue;
            for (int j = 0; j < 2 * n; ++j)
                aug[(size_t)row * 2 * n + j] -= factor * aug[(size_t)col * 2 * n + j];
        }
    }
    // Copy inverse out of augmented [A | A^{-1}].
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = aug[(size_t)i * 2 * n + (n + j)];
}

}  // namespace

void STEOMCCSDOperator::build_x_matrices(const real_t* h_R1_IP_amplitudes,
                                          const real_t* h_R1_EA_amplitudes) {
    // ----- X(MI) = inverse of R1_active_IP (n_act_occ × n_act_occ) -----
    std::vector<real_t> h_X_IP((size_t)n_act_occ_ * n_act_occ_, 0.0);
    for (int m_NTO = 0; m_NTO < n_act_occ_; ++m_NTO) {
        const int m_idx = active_occ_idx_[m_NTO];
        for (int n_root = 0; n_root < n_act_occ_; ++n_root) {
            // R1^(n_root) is row n_root of h_R1_IP_amplitudes with stride nocc_active_.
            const real_t* r1 = h_R1_IP_amplitudes + (size_t)n_root * nocc_active_;
            h_X_IP[(size_t)m_NTO * n_act_occ_ + n_root] = r1[m_idx];
        }
    }
    invert_small_matrix_inplace(h_X_IP.data(), n_act_occ_);
    tracked_cudaMalloc(&d_X_IP_, (size_t)n_act_occ_ * n_act_occ_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < h_X_IP.size(); ++i) d_X_IP_[i] = h_X_IP[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_X_IP_, h_X_IP.data(),
                   h_X_IP.size() * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    // ----- X(EA) = inverse of R1_active_EA (n_act_vir × n_act_vir) -----
    std::vector<real_t> h_X_EA((size_t)n_act_vir_ * n_act_vir_, 0.0);
    for (int e_NTO = 0; e_NTO < n_act_vir_; ++e_NTO) {
        const int a_idx = active_vir_idx_[e_NTO];
        for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
            const real_t* r1 = h_R1_EA_amplitudes + (size_t)e_root * nvir_;
            h_X_EA[(size_t)e_NTO * n_act_vir_ + e_root] = r1[a_idx];
        }
    }
    invert_small_matrix_inplace(h_X_EA.data(), n_act_vir_);
    tracked_cudaMalloc(&d_X_EA_, (size_t)n_act_vir_ * n_act_vir_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < h_X_EA.size(); ++i) d_X_EA_[i] = h_X_EA[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_X_EA_, h_X_EA.data(),
                   h_X_EA.size() * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    std::cout << "  STEOM-CCSD X(MI)/X(EA) matrices built (CFOUR `renormalize`, "
              << "active R1 inverse, " << n_act_occ_ << "×" << n_act_occ_
              << " and " << n_act_vir_ << "×" << n_act_vir_ << ")." << std::endl;
}


// ==================================================================
//  build_F_eff_oo — sub-phase 3.4: U(M,I) + F^eff_oo per CFOUR
//    `gmi_steom_rhf` (steom_intermediates.cxx lines 7-81).
//
//  U(M,I) is built per active root m by the same contractions the PySCF
//  IP-EOM σ1[i] matvec performs on r2 (spin-adapted closed-shell), with
//  r2 replaced by R2_IP^{(m)}:
//    U(M,I) = +2 Σ Fov[l,d] R2_IP^{(m)}[I,l,d]
//             -   Σ Fov[k,d] R2_IP^{(m)}[k,I,d]
//             - 2 Σ Wooov[k,l,I,d] R2_IP^{(m)}[k,l,d]
//             +   Σ Wooov[l,k,I,d] R2_IP^{(m)}[k,l,d]
//
//  Then F^eff_oo[M_idx, I] = Loo[M_idx, I] - Σ_N U(N,I) · X(N,M)
//  (active rows only — inactive rows are copied from bar Loo).
//
//  Host implementation: F^eff_oo is small (nocc² ≤ ~10⁴ doubles for valence
//  basis sets), no GPU kernel needed for sub-phase 3.4 — F^eff_oo is consumed
//  by the σ matvec in sub-phase 3.7, where the actual hot loop lives.
// ==================================================================
void STEOMCCSDOperator::build_F_eff_oo() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t fov_sz   = (size_t)NO * NV;
    const size_t ooov_sz  = (size_t)NO * NO * NO * NV;
    const size_t r2_per_m = (size_t)NO * NO * NV;

    // Pull bar-H + R2 + Loo back to host (small for sub-phase 3.4).
    std::vector<real_t> h_Fov(fov_sz);
    std::vector<real_t> h_Wooov(ooov_sz);
    std::vector<real_t> h_Loo((size_t)NO * NO);
    std::vector<real_t> h_R2_IP((size_t)n_act_occ_ * r2_per_m);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_Fov.data(),   d_Fov_,   fov_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Wooov.data(), d_Wooov_, ooov_sz  * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Loo.data(),   d_Loo_,   (size_t)NO*NO * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_R2_IP.data(), d_R2_IP_,
                   h_R2_IP.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < fov_sz;   ++i) h_Fov[i]   = d_Fov_[i];
        for (size_t i = 0; i < ooov_sz;  ++i) h_Wooov[i] = d_Wooov_[i];
        for (size_t i = 0; i < (size_t)NO*NO; ++i) h_Loo[i] = d_Loo_[i];
        for (size_t i = 0; i < h_R2_IP.size(); ++i) h_R2_IP[i] = d_R2_IP_[i];
    }

    // X(MI) pulled from device (small).
    std::vector<real_t> h_X_IP((size_t)n_act_occ_ * n_act_occ_);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_X_IP.data(), d_X_IP_,
                   h_X_IP.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < h_X_IP.size(); ++i) h_X_IP[i] = d_X_IP_[i];
    }

    // ----- Build U(M,I) = U[m_NTO_or_root, i_full_occ] (n_act_occ × nocc) -
    // Note: in CFOUR notation M is the "root index" (the n_root in our X_IP
    // 2nd index), not the m_NTO index. The U dimension is (n_act_occ × NO).
    std::vector<real_t> h_U_MI((size_t)n_act_occ_ * NO, 0.0);

    auto R2 = [&](int m_root, int i, int j, int a) -> real_t {
        return h_R2_IP[((size_t)m_root * NO + i) * NO * NV
                       + (size_t)j * NV + a];
    };
    auto W = [&](int k, int l, int i, int d) -> real_t {
        return h_Wooov[((size_t)k * NO + l) * NO * NV + (size_t)i * NV + d];
    };

    #pragma omp parallel for collapse(2) if (n_act_occ_ * NO > 16)
    for (int m_root = 0; m_root < n_act_occ_; ++m_root) {
        for (int I = 0; I < NO; ++I) {
            real_t u = 0.0;
            // +2 Σ_{l,d} Fov[l,d] R2[I,l,d]   -   Σ_{k,d} Fov[k,d] R2[k,I,d]
            for (int x = 0; x < NO; ++x) {
                for (int d = 0; d < NV; ++d) {
                    const real_t fov = h_Fov[(size_t)x * NV + d];
                    u += 2.0 * fov * R2(m_root, I, x, d);
                    u -=       fov * R2(m_root, x, I, d);
                }
            }
            // -2 Σ Wooov[k,l,I,d] R2[k,l,d] + Σ Wooov[l,k,I,d] R2[k,l,d]
            for (int k = 0; k < NO; ++k) {
                for (int l = 0; l < NO; ++l) {
                    for (int d = 0; d < NV; ++d) {
                        const real_t r2_kld = R2(m_root, k, l, d);
                        u -= 2.0 * W(k, l, I, d) * r2_kld;
                        u +=       W(l, k, I, d) * r2_kld;
                    }
                }
            }
            h_U_MI[(size_t)m_root * NO + I] = u;
        }
    }

    // ----- Build F^eff_oo: F^eff[M_idx, I] = Loo[M_idx, I] - Σ_N U(N,I) X(N,M)
    // For inactive rows of F^eff (rows not in active_occ_idx_), keep bar Loo.
    std::vector<real_t> h_F_eff_oo = h_Loo;  // copy bar Loo

    for (int m_NTO = 0; m_NTO < n_act_occ_; ++m_NTO) {
        const int m_idx = active_occ_idx_[m_NTO];
        for (int I = 0; I < NO; ++I) {
            real_t s = h_Loo[(size_t)m_idx * NO + I];
            for (int n_root = 0; n_root < n_act_occ_; ++n_root) {
                s -= h_U_MI[(size_t)n_root * NO + I]
                     * h_X_IP[(size_t)n_root * n_act_occ_ + m_NTO];
            }
            h_F_eff_oo[(size_t)m_idx * NO + I] = s;
        }
    }

    // Upload to device.
    tracked_cudaMalloc(&d_F_eff_oo_, (size_t)NO * NO * sizeof(real_t));
    tracked_cudaMalloc(&d_U_MI_,     (size_t)n_act_occ_ * NO * sizeof(real_t));
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(d_F_eff_oo_, h_F_eff_oo.data(),
                   h_F_eff_oo.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U_MI_, h_U_MI.data(),
                   h_U_MI.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i = 0; i < h_F_eff_oo.size(); ++i) d_F_eff_oo_[i] = h_F_eff_oo[i];
        for (size_t i = 0; i < h_U_MI.size();     ++i) d_U_MI_[i]     = h_U_MI[i];
    }

    std::cout << "  STEOM-CCSD F^eff_oo built (CFOUR `gmi_steom_rhf` formula; "
              << "active rows dressed, inactive rows = bar Loo)." << std::endl;
}


// ==================================================================
//  build_F_eff_vv — sub-phase 3.4 (extended): U(E,A) + F^eff_vv per
//    CFOUR `gea_steom_rhf` (steom_intermediates.cxx lines 83-157).
//
//  U(E, A) per active EA root e — PySCF EA-EOM σ1[a] r2 contributions:
//    U(e, A) = +2 Σ Fov[l,d] R2_EA^{(e)}[l, A, d]
//              -   Σ Fov[l,d] R2_EA^{(e)}[l, d, A]
//              + Σ_{l,c,d} (2 Wvovv[A,l,c,d] − Wvovv[A,l,d,c]) R2_EA^{(e)}[l,c,d]
//
//  Then F^eff_vv[A_idx, B] = Lvv[A_idx, B] + Σ_E U(E, B) · X(E, A_NTO)
//  (active rows dressed; inactive rows = bar Lvv).
//
//  Sign is +, opposite of F^eff_oo (which had −) — matches the asymmetry
//  between hole / particle absorption in the second similarity transform.
// ==================================================================
void STEOMCCSDOperator::build_F_eff_vv() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t fov_sz   = (size_t)NO * NV;
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    const size_t r2_per_e = (size_t)NO * NV * NV;

    // Pull bar-H + R2_EA + Lvv back to host.
    std::vector<real_t> h_Fov(fov_sz);
    std::vector<real_t> h_Wvovv(wvovv_sz);
    std::vector<real_t> h_Lvv((size_t)NV * NV);
    std::vector<real_t> h_R2_EA((size_t)n_act_vir_ * r2_per_e);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_Fov.data(),   d_Fov_,   fov_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Wvovv.data(), d_Wvovv_, wvovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Lvv.data(),   d_Lvv_,   (size_t)NV*NV * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_R2_EA.data(), d_R2_EA_,
                   h_R2_EA.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < fov_sz;   ++i) h_Fov[i]   = d_Fov_[i];
        for (size_t i = 0; i < wvovv_sz; ++i) h_Wvovv[i] = d_Wvovv_[i];
        for (size_t i = 0; i < (size_t)NV*NV; ++i) h_Lvv[i] = d_Lvv_[i];
        for (size_t i = 0; i < h_R2_EA.size(); ++i) h_R2_EA[i] = d_R2_EA_[i];
    }

    // X(EA) pulled from device (small).
    std::vector<real_t> h_X_EA((size_t)n_act_vir_ * n_act_vir_);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_X_EA.data(), d_X_EA_,
                   h_X_EA.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < h_X_EA.size(); ++i) h_X_EA[i] = d_X_EA_[i];
    }

    // ----- Build U(E,A) = U[e_root, A_full_vir] (n_act_vir × nvir) -----
    std::vector<real_t> h_U_EA((size_t)n_act_vir_ * NV, 0.0);

    auto R2 = [&](int e_root, int l, int a, int b) -> real_t {
        return h_R2_EA[((size_t)e_root * NO + l) * NV * NV
                       + (size_t)a * NV + b];
    };
    auto W = [&](int a, int l, int c, int d) -> real_t {
        return h_Wvovv[((size_t)a * NO + l) * NV * NV + (size_t)c * NV + d];
    };

    #pragma omp parallel for collapse(2) if (n_act_vir_ * NV > 16)
    for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
        for (int A = 0; A < NV; ++A) {
            real_t u = 0.0;
            // +2 Σ_{l,d} Fov[l,d] R2[l,A,d]  -  Σ_{l,d} Fov[l,d] R2[l,d,A]
            for (int l = 0; l < NO; ++l) {
                for (int d = 0; d < NV; ++d) {
                    const real_t fov = h_Fov[(size_t)l * NV + d];
                    u += 2.0 * fov * R2(e_root, l, A, d);
                    u -=       fov * R2(e_root, l, d, A);
                }
            }
            // +2 Σ Wvovv[A,l,c,d] R2[l,c,d]  -  Σ Wvovv[A,l,d,c] R2[l,c,d]
            for (int l = 0; l < NO; ++l) {
                for (int c = 0; c < NV; ++c) {
                    for (int d = 0; d < NV; ++d) {
                        const real_t r2_lcd = R2(e_root, l, c, d);
                        u += 2.0 * W(A, l, c, d) * r2_lcd;
                        u -=       W(A, l, d, c) * r2_lcd;
                    }
                }
            }
            h_U_EA[(size_t)e_root * NV + A] = u;
        }
    }

    // ----- Build F^eff_vv: F^eff[A_idx, B] = Lvv[A_idx, B] + Σ_E U(E,B) X(E,A_NTO)
    std::vector<real_t> h_F_eff_vv = h_Lvv;  // copy bar Lvv (inactive rows untouched)

    for (int a_NTO = 0; a_NTO < n_act_vir_; ++a_NTO) {
        const int a_idx = active_vir_idx_[a_NTO];
        for (int B = 0; B < NV; ++B) {
            real_t s = h_Lvv[(size_t)a_idx * NV + B];
            for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
                s += h_U_EA[(size_t)e_root * NV + B]
                     * h_X_EA[(size_t)e_root * n_act_vir_ + a_NTO];
            }
            h_F_eff_vv[(size_t)a_idx * NV + B] = s;
        }
    }

    // Upload to device.
    tracked_cudaMalloc(&d_F_eff_vv_, (size_t)NV * NV * sizeof(real_t));
    tracked_cudaMalloc(&d_U_EA_,     (size_t)n_act_vir_ * NV * sizeof(real_t));
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(d_F_eff_vv_, h_F_eff_vv.data(),
                   h_F_eff_vv.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U_EA_, h_U_EA.data(),
                   h_U_EA.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i = 0; i < h_F_eff_vv.size(); ++i) d_F_eff_vv_[i] = h_F_eff_vv[i];
        for (size_t i = 0; i < h_U_EA.size();     ++i) d_U_EA_[i]     = h_U_EA[i];
    }

    std::cout << "  STEOM-CCSD F^eff_vv built (CFOUR `gea_steom_rhf` formula; "
              << "active rows dressed, inactive rows = bar Lvv)." << std::endl;
}


// ==================================================================
//  build_W_eff_and_G — sub-phase 3.5-3.7: full W^eff dressing + dense
//  G^{1h1p}. Direct host port of Python build_g_canonical_full
//  (script/pyscf_steom_feff_reference.py, Nooijen-Bartlett 1997 Eq.34-63).
//
//  All tensors are tiny for valence basis sets; host loops mirror the
//  Python einsums one-to-one. The hot loop (Davidson) reuses the dense
//  d_G_ matvec, so no GPU kernel for the build itself.
//
//  Validation gate (H2O sto-3g, (3,2) active): lowest singlet eigenvalues
//  of d_G_ == 0.392886 / 0.449061 (Python reference, accepted STEOM-level).
// ==================================================================
void STEOMCCSDOperator::build_W_eff_and_G() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const int NMo = n_act_occ_;
    const int NMv = n_act_vir_;

    // ---- pull bar-H + R2 + X to host ----
    auto pull = [&](real_t* dptr, size_t sz) {
        std::vector<real_t> h(sz);
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available())
            cudaMemcpy(h.data(), dptr, sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        else
#endif
            for (size_t i = 0; i < sz; ++i) h[i] = dptr[i];
        return h;
    };
    std::vector<real_t> Fov   = pull(d_Fov_,   (size_t)NO*NV);
    std::vector<real_t> Loo   = pull(d_Loo_,   (size_t)NO*NO);
    std::vector<real_t> Lvv   = pull(d_Lvv_,   (size_t)NV*NV);
    std::vector<real_t> Wooov = pull(d_Wooov_, (size_t)NO*NO*NO*NV);
    std::vector<real_t> Wvovv = pull(d_Wvovv_, (size_t)NV*NO*NV*NV);
    std::vector<real_t> Wovoo = pull(d_Wovoo_, (size_t)NO*NV*NO*NO);
    std::vector<real_t> Wovov = pull(d_Wovov_, (size_t)NO*NV*NO*NV);
    std::vector<real_t> Wovvo = pull(d_Wovvo_, (size_t)NO*NV*NV*NO);
    std::vector<real_t> ERIov = pull(d_eri_ovov_, (size_t)NO*NV*NO*NV);
    std::vector<real_t> R2IP  = pull(d_R2_IP_, (size_t)NMo*NO*NO*NV);
    std::vector<real_t> R2EA  = pull(d_R2_EA_, (size_t)NMv*NO*NV*NV);
    std::vector<real_t> XIP   = pull(d_X_IP_,  (size_t)NMo*NMo);
    std::vector<real_t> XEA   = pull(d_X_EA_,  (size_t)NMv*NMv);

    // ---- bar-H accessors (row-major natural order; see build_dressed_intermediates) ----
    auto fov  = [&](int k,int c){ return Fov[(size_t)k*NV+c]; };
    auto loo  = [&](int i,int j){ return Loo[(size_t)i*NO+j]; };
    auto lvv  = [&](int a,int b){ return Lvv[(size_t)a*NV+b]; };
    auto wooov= [&](int k,int l,int i,int d){ return Wooov[(((size_t)k*NO+l)*NO+i)*NV+d]; };
    auto wvovv= [&](int a,int l,int c,int d){ return Wvovv[(((size_t)a*NO+l)*NV+c)*NV+d]; };
    auto wovoo= [&](int k,int c,int l,int i){ return Wovoo[(((size_t)k*NV+c)*NO+l)*NO+i]; };
    auto wovov= [&](int k,int b,int i,int d){ return Wovov[(((size_t)k*NV+b)*NO+i)*NV+d]; };
    auto wovvo= [&](int k,int b,int c,int j){ return Wovvo[(((size_t)k*NV+b)*NV+c)*NO+j]; };
    auto eriov= [&](int k,int b,int i,int d){ return ERIov[(((size_t)k*NV+b)*NO+i)*NV+d]; };
    auto r2ip = [&](int m,int i,int j,int a){ return R2IP[(((size_t)m*NO+i)*NO+j)*NV+a]; };
    auto r2ea = [&](int e,int i,int a,int b){ return R2EA[(((size_t)e*NO+i)*NV+a)*NV+b]; };

    // ---- s_IP[m][i,j,a] = -Σ_λ R2_IP[λ][i,j,a]·X_IP[m,λ] ; s_EA[e][i,a,b] = +Σ_λ R2_EA[λ]·X_EA[e,λ]
    //  (X_IP[m_NTO,λ] = rinv_IP[λ,m_NTO] from build_x_matrices, matching build_normalized_s.)
    std::vector<real_t> sIP((size_t)NMo*NO*NO*NV, 0.0);
    std::vector<real_t> sEA((size_t)NMv*NO*NV*NV, 0.0);
    auto siP = [&](int m,int i,int j,int a)->real_t&{ return sIP[(((size_t)m*NO+i)*NO+j)*NV+a]; };
    auto seA = [&](int e,int i,int a,int b)->real_t&{ return sEA[(((size_t)e*NO+i)*NV+a)*NV+b]; };
    for (int m=0;m<NMo;++m)
        for (int lam=0;lam<NMo;++lam){
            const real_t x = XIP[(size_t)m*NMo+lam];
            for (int i=0;i<NO;++i) for(int j=0;j<NO;++j) for(int a=0;a<NV;++a)
                siP(m,i,j,a) -= r2ip(lam,i,j,a)*x;
        }
    for (int e=0;e<NMv;++e)
        for (int lam=0;lam<NMv;++lam){
            const real_t x = XEA[(size_t)e*NMv+lam];
            for (int i=0;i<NO;++i) for(int a=0;a<NV;++a) for(int b=0;b<NV;++b)
                seA(e,i,a,b) += r2ea(lam,i,a,b)*x;
        }

    // ---- F_eff_oo (Eq.34-35) + F_eff_vv (Eq.36-37) rebuilt with normalized s ----
    std::vector<real_t> Foo = Loo;   // [NO×NO]
    for (int m=0;m<NMo;++m){
        const int mrow = active_occ_idx_[m];
        for (int i=0;i<NO;++i){
            real_t u=0.0;
            for (int k=0;k<NO;++k) for(int c=0;c<NV;++c){
                const real_t st = 2.0*siP(m,i,k,c)-siP(m,k,i,c);
                u += fov(k,c)*st;
            }
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*siP(m,k,l,d)-siP(m,l,k,d);
                u -= wooov(k,l,i,d)*st;
            }
            Foo[(size_t)mrow*NO+i] += u;
        }
    }
    std::vector<real_t> Fvv = Lvv;   // [NV×NV]
    for (int e=0;e<NMv;++e){
        const int arow = active_vir_idx_[e];
        for (int a=0;a<NV;++a){
            real_t u=0.0;
            for (int k=0;k<NO;++k) for(int c=0;c<NV;++c){
                const real_t st = 2.0*seA(e,k,a,c)-seA(e,k,c,a);
                u += fov(k,c)*st;
            }
            for (int l=0;l<NO;++l) for(int c=0;c<NV;++c) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,c,d)-seA(e,l,d,c);
                u += wvovv(a,l,c,d)*st;
            }
            Fvv[(size_t)arow*NV+a] += u;
        }
    }

    // ---- hp (Eq.38-39) ----
    std::vector<real_t> u_ma((size_t)NMo*NV, 0.0);  // [m][a]
    std::vector<real_t> u_ie((size_t)NO*NMv, 0.0);  // [i][e]
    for (int m=0;m<NMo;++m)
        for (int a=0;a<NV;++a){
            real_t v=0.0;
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*siP(m,k,l,d)-siP(m,l,k,d);
                v -= wovvo(k,d,a,l)*st;
            }
            u_ma[(size_t)m*NV+a]=v;
        }
    for (int e=0;e<NMv;++e)
        for (int i=0;i<NO;++i){
            real_t v=0.0;
            for (int l=0;l<NO;++l) for(int c=0;c<NV;++c) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,c,d)-seA(e,l,d,c);
                v += wovvo(i,d,c,l)*st;
            }
            u_ie[(size_t)i*NMv+e]=v;
        }

    // ---- hhhp (Eq.42-44, bare v = eri_ovov) ----
    std::vector<real_t> u_mlid((size_t)NMo*NO*NO*NV, 0.0); // [m][l,i,d]
    std::vector<real_t> u_kmid((size_t)NO*NMo*NO*NV, 0.0); // [k][m][i,d]
    std::vector<real_t> u_klie((size_t)NO*NO*NO*NMv, 0.0); // [k,l,i][e]
    auto UMLID=[&](int m,int l,int i,int d)->real_t&{ return u_mlid[(((size_t)m*NO+l)*NO+i)*NV+d]; };
    auto UKMID=[&](int k,int m,int i,int d)->real_t&{ return u_kmid[(((size_t)k*NMo+m)*NO+i)*NV+d]; };
    auto UKLIE=[&](int k,int l,int i,int e)->real_t&{ return u_klie[(((size_t)k*NO+l)*NO+i)*NMv+e]; };
    for (int m=0;m<NMo;++m)
        for (int l=0;l<NO;++l) for(int i=0;i<NO;++i) for(int d=0;d<NV;++d){
            real_t v=0.0;
            for (int j=0;j<NO;++j) for(int b=0;b<NV;++b){
                const real_t st = 2.0*siP(m,i,j,b)-siP(m,j,i,b);
                v += eriov(j,b,l,d)*st - eriov(l,b,j,d)*siP(m,i,j,b);
            }
            UMLID(m,l,i,d)=v;
        }
    for (int m=0;m<NMo;++m)
        for (int k=0;k<NO;++k) for(int i=0;i<NO;++i) for(int d=0;d<NV;++d){
            real_t v=0.0;
            for (int j=0;j<NO;++j) for(int b=0;b<NV;++b)
                v -= eriov(j,d,k,b)*siP(m,j,i,b);
            UKMID(k,m,i,d)=v;
        }
    for (int e=0;e<NMv;++e)
        for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int i=0;i<NO;++i){
            real_t v=0.0;
            for (int a=0;a<NV;++a) for(int b=0;b<NV;++b)
                v += eriov(k,a,l,b)*seA(e,i,a,b);
            UKLIE(k,l,i,e)=v;
        }

    // ---- phph (Eq.56-58) ----
    std::vector<real_t> u_amci((size_t)NV*NMo*NV*NO, 0.0); // [a][m][c][i]
    std::vector<real_t> u_akei((size_t)NV*NO*NMv*NO, 0.0); // [a][k][e][i]
    std::vector<real_t> u_amei((size_t)NV*NMo*NMv*NO, 0.0);// [a][m][e][i]
    auto UAMCI=[&](int a,int m,int c,int i)->real_t&{ return u_amci[(((size_t)a*NMo+m)*NV+c)*NO+i]; };
    auto UAKEI=[&](int a,int k,int e,int i)->real_t&{ return u_akei[(((size_t)a*NO+k)*NMv+e)*NO+i]; };
    auto UAMEI=[&](int a,int m,int e,int i)->real_t&{ return u_amei[(((size_t)a*NMo+m)*NMv+e)*NO+i]; };
    for (int m=0;m<NMo;++m)
        for (int a=0;a<NV;++a) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i){
            real_t t=0.0;
            for (int k=0;k<NO;++k) t -= fov(k,c)*siP(m,i,k,a);                    // T1
            for (int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*siP(m,i,l,d)-siP(m,l,i,d);
                t += wvovv(a,l,c,d)*st;                                            // T2
                t -= wvovv(a,l,d,c)*siP(m,i,l,d);                                  // T3
            }
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                t += wovoo(k,c,l,i)*siP(m,l,k,a);                                  // T4
            UAMCI(a,m,c,i)=t;
        }
    for (int e=0;e<NMv;++e)
        for (int a=0;a<NV;++a) for(int k=0;k<NO;++k) for(int i=0;i<NO;++i){
            real_t t=0.0;
            for (int c=0;c<NV;++c) t -= fov(k,c)*seA(e,i,a,c);                     // T1
            for (int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,a,d)-seA(e,l,d,a);
                t += wovoo(l,d,k,i)*st;                                            // T2
                t -= wooov(l,k,i,d)*seA(e,l,a,d);                                  // T3
            }
            for (int c=0;c<NV;++c) for(int d=0;d<NV;++d)
                t += wvovv(a,k,c,d)*seA(e,i,c,d);                                  // T4
            UAKEI(a,k,e,i)=t;
        }
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e)
        for (int a=0;a<NV;++a) for(int i=0;i<NO;++i){
            real_t t=0.0;
            for (int c=0;c<NV;++c) t += u_ma[(size_t)m*NV+c]*seA(e,i,a,c);         // T1
            for (int k=0;k<NO;++k) t -= u_ie[(size_t)k*NMv+e]*siP(m,i,k,a);        // T2
            for (int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,a,d)-seA(e,l,d,a);
                t += UMLID(m,l,i,d)*st;                                            // T3
                t -= UKMID(l,m,i,d)*seA(e,l,a,d);                                  // T4
            }
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                t += UKLIE(k,l,i,e)*siP(m,l,k,a);                                  // T5
            UAMEI(a,m,e,i)=t;
        }

    // ---- phhp (Eq.60-62) ----
    std::vector<real_t> u_bmjc((size_t)NV*NMo*NO*NV, 0.0); // [b][m][j][c]
    std::vector<real_t> u_bkje((size_t)NV*NO*NO*NMv, 0.0); // [b][k][j][e]
    std::vector<real_t> u_bmje((size_t)NV*NMo*NO*NMv, 0.0);// [b][m][j][e]
    auto UBMJC=[&](int b,int m,int j,int c)->real_t&{ return u_bmjc[(((size_t)b*NMo+m)*NO+j)*NV+c]; };
    auto UBKJE=[&](int b,int k,int j,int e)->real_t&{ return u_bkje[(((size_t)b*NO+k)*NO+j)*NMv+e]; };
    auto UBMJE=[&](int b,int m,int j,int e)->real_t&{ return u_bmje[(((size_t)b*NMo+m)*NO+j)*NMv+e]; };
    for (int m=0;m<NMo;++m)
        for (int b=0;b<NV;++b) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c){
            real_t t=0.0;
            for (int k=0;k<NO;++k) t -= fov(k,c)*siP(m,k,j,b);                     // T1
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                t += wovoo(k,c,l,j)*siP(m,k,l,b);                                  // T2
            for (int k=0;k<NO;++k) for(int d=0;d<NV;++d)
                t -= wvovv(b,k,d,c)*siP(m,k,j,d);                                  // T3
            UBMJC(b,m,j,c)=t;
        }
    for (int e=0;e<NMv;++e)
        for (int b=0;b<NV;++b) for(int k=0;k<NO;++k) for(int j=0;j<NO;++j){
            real_t t=0.0;
            for (int d=0;d<NV;++d) t += fov(k,d)*seA(e,j,d,b);                     // T1
            for (int d=0;d<NV;++d) for(int c=0;c<NV;++c)
                t += wvovv(b,k,d,c)*seA(e,j,c,d);                                  // T2
            for (int l=0;l<NO;++l) for(int d=0;d<NV;++d)
                t -= wooov(l,k,j,d)*seA(e,l,d,b);                                  // T3
            UBKJE(b,k,j,e)=t;
        }
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e)
        for (int b=0;b<NV;++b) for(int j=0;j<NO;++j){
            real_t t=0.0;
            for (int d=0;d<NV;++d) t += u_ma[(size_t)m*NV+d]*seA(e,j,d,b);         // T1
            for (int k=0;k<NO;++k) t -= u_ie[(size_t)k*NMv+e]*siP(m,k,j,b);        // T2
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                t += UKLIE(k,l,j,e)*siP(m,k,l,b);                                  // T3
            for (int l=0;l<NO;++l) for(int d=0;d<NV;++d)
                t -= UMLID(m,l,j,d)*seA(e,l,d,b);                                  // T4
            UBMJE(b,m,j,e)=t;
        }

    // ---- assemble g_phph[a,k,c,i] (Eq.59) and g_phhp[b,k,j,c] (Eq.63) ----
    std::vector<real_t> g_phph((size_t)NV*NO*NV*NO, 0.0);
    std::vector<real_t> g_phhp((size_t)NV*NO*NO*NV, 0.0);
    auto GPHPH=[&](int a,int k,int c,int i)->real_t&{ return g_phph[(((size_t)a*NO+k)*NV+c)*NO+i]; };
    auto GPHHP=[&](int b,int k,int j,int c)->real_t&{ return g_phhp[(((size_t)b*NO+k)*NO+j)*NV+c]; };
    for (int a=0;a<NV;++a) for(int k=0;k<NO;++k) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i)
        GPHPH(a,k,c,i)=wovov(k,a,i,c);
    for (int m=0;m<NMo;++m){ const int kf=active_occ_idx_[m];
        for(int a=0;a<NV;++a) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i)
            GPHPH(a,kf,c,i)+=UAMCI(a,m,c,i); }
    for (int e=0;e<NMv;++e){ const int cf=active_vir_idx_[e];
        for(int a=0;a<NV;++a) for(int k=0;k<NO;++k) for(int i=0;i<NO;++i)
            GPHPH(a,k,cf,i)+=UAKEI(a,k,e,i); }
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e){ const int kf=active_occ_idx_[m], cf=active_vir_idx_[e];
        for(int a=0;a<NV;++a) for(int i=0;i<NO;++i)
            GPHPH(a,kf,cf,i)+=UAMEI(a,m,e,i); }
    for (int b=0;b<NV;++b) for(int k=0;k<NO;++k) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c)
        GPHHP(b,k,j,c)=wovvo(k,b,c,j);
    for (int m=0;m<NMo;++m){ const int kf=active_occ_idx_[m];
        for(int b=0;b<NV;++b) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c)
            GPHHP(b,kf,j,c)+=UBMJC(b,m,j,c); }
    for (int e=0;e<NMv;++e){ const int cf=active_vir_idx_[e];
        for(int b=0;b<NV;++b) for(int k=0;k<NO;++k) for(int j=0;j<NO;++j)
            GPHHP(b,k,j,cf)+=UBKJE(b,k,j,e); }
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e){ const int kf=active_occ_idx_[m], cf=active_vir_idx_[e];
        for(int b=0;b<NV;++b) for(int j=0;j<NO;++j)
            GPHHP(b,kf,j,cf)+=UBMJE(b,m,j,e); }

    // ---- G^{1h1p} singlet: row=i*NV+a, col=j*NV+b ----
    //  G = F_eff_vv δ_ij − F_eff_oo δ_ab + 2 g_phhp[b,j,i,a] − g_phph[a,j,b,i]
    std::vector<real_t> Gmat((size_t)total_dim_*total_dim_, 0.0);
    for (int i=0;i<NO;++i) for(int a=0;a<NV;++a){
        const int row=i*NV+a;
        for (int j=0;j<NO;++j) for(int b=0;b<NV;++b){
            const int col=j*NV+b;
            real_t v = 2.0*GPHHP(b,j,i,a) - GPHPH(a,j,b,i);
            if (i==j) v += Fvv[(size_t)a*NV+b];
            if (a==b) v -= Foo[(size_t)i*NO+j];
            Gmat[(size_t)row*total_dim_+col]=v;
        }
    }

    // ---- upload dense G + refresh diagonal ----
    tracked_cudaMalloc(&d_G_, (size_t)total_dim_*total_dim_*sizeof(real_t));
    std::vector<real_t> h_diag(total_dim_);
    for (int r=0;r<total_dim_;++r) h_diag[r]=Gmat[(size_t)r*total_dim_+r];
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()){
        cudaMemcpy(d_G_, Gmat.data(), Gmat.size()*sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_diagonal_, h_diag.data(), h_diag.size()*sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i=0;i<Gmat.size();++i) d_G_[i]=Gmat[i];
        for (int r=0;r<total_dim_;++r) d_diagonal_[r]=h_diag[r];
    }
    // Verification aid: HOMO->LUMO diagonal element (= eigenvalue for the
    // pure HOMO->LUMO state). H2O sto-3g (3,2) reference: 0.392886.
    {
        const int hl = (NO - 1) * NV + 0;
        std::cout << "  STEOM-CCSD G^{1h1p} built (W^eff dressing Eq.34-63, dense "
                  << total_dim_ << "×" << total_dim_ << "); G[HOMO->LUMO diag] = "
                  << Gmat[(size_t)hl * total_dim_ + hl]
                  << "  (H2O sto-3g ref 0.392886)." << std::endl;
    }
}

#undef H_OVOV
#undef H_OOOV
#undef H_OOVV
#undef H_OVVO
#undef H_OVVV
#undef H_OOOO
#undef H_VVVV
#undef H_T1
#undef H_T2


// ==================================================================
//  print_amplitude_norms — Ŝ Frobenius for sub-phase 3.0+3.1 smoke
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

void STEOMCCSDOperator::print_amplitude_norms(std::ostream& os) const {
    const int NO = nocc_active_, NV = nvir_;
    const size_t r2_ip_sz  = (size_t)n_act_occ_ * NO * NO * NV;
    const size_t r2_ea_sz  = (size_t)n_act_vir_ * NO * NV * NV;
    const size_t r1_ip_sz  = (size_t)n_act_occ_ * NO;
    const size_t r1_ea_sz  = (size_t)n_act_vir_ * NV;
    const size_t loo_sz    = (size_t)NO * NO;
    const size_t lvv_sz    = (size_t)NV * NV;
    const size_t fov_sz    = (size_t)NO * NV;
    const size_t oooo_sz   = (size_t)NO * NO * NO * NO;
    const size_t ooov_sz   = (size_t)NO * NO * NO * NV;
    const size_t ovov_sz   = (size_t)NO * NV * NO * NV;
    const size_t ovvo_sz   = (size_t)NO * NV * NV * NO;
    const size_t wovoo_sz  = (size_t)NO * NV * NO * NO;
    const size_t wvovv_sz  = (size_t)NV * NO * NV * NV;
    const size_t wvvvv_sz  = (size_t)NV * NV * NV * NV;
    const size_t wvvvo_sz  = (size_t)NV * NV * NV * NO;
    const size_t x_ip_sz   = (size_t)n_act_occ_ * n_act_occ_;
    const size_t x_ea_sz   = (size_t)n_act_vir_ * n_act_vir_;
    const size_t u_mi_sz   = (size_t)n_act_occ_ * NO;

    os << "  [STEOM-CCSD amplitude + bar-H + F^eff_oo Frobenius norms]\n"
       << "    dims:  nocc_active=" << NO << "  nvir=" << NV
       <<           "  n_act_occ=" << n_act_occ_ << "  n_act_vir=" << n_act_vir_
       <<           "  total_dim=" << total_dim_ << "\n"
       << std::fixed << std::setprecision(8)
       << "    ‖R2^IP‖     = " << frobenius_norm_device(d_R2_IP_, r2_ip_sz) << "\n"
       << "    ‖R2^EA‖     = " << frobenius_norm_device(d_R2_EA_, r2_ea_sz) << "\n";
    if (d_R1_IP_ != nullptr)
        os << "    ‖R1^IP‖     = " << frobenius_norm_device(d_R1_IP_, r1_ip_sz) << "\n";
    if (d_R1_EA_ != nullptr)
        os << "    ‖R1^EA‖     = " << frobenius_norm_device(d_R1_EA_, r1_ea_sz) << "\n";
    if (d_Loo_ != nullptr) {
        os << "    ‖Loo‖       = " << frobenius_norm_device(d_Loo_,   loo_sz)   << "\n"
           << "    ‖Lvv‖       = " << frobenius_norm_device(d_Lvv_,   lvv_sz)   << "\n"
           << "    ‖Fov‖       = " << frobenius_norm_device(d_Fov_,   fov_sz)   << "\n"
           << "    ‖Woooo‖     = " << frobenius_norm_device(d_Woooo_, oooo_sz)  << "\n"
           << "    ‖Wooov‖     = " << frobenius_norm_device(d_Wooov_, ooov_sz)  << "\n"
           << "    ‖Wovov‖     = " << frobenius_norm_device(d_Wovov_, ovov_sz)  << "\n"
           << "    ‖Wovvo‖     = " << frobenius_norm_device(d_Wovvo_, ovvo_sz)  << "\n"
           << "    ‖Wovoo‖     = " << frobenius_norm_device(d_Wovoo_, wovoo_sz) << "\n"
           << "    ‖Wvovv‖     = " << frobenius_norm_device(d_Wvovv_, wvovv_sz) << "\n"
           << "    ‖Wvvvv‖     = " << frobenius_norm_device(d_Wvvvv_, wvvvv_sz) << "\n"
           << "    ‖Wvvvo‖     = " << frobenius_norm_device(d_Wvvvo_, wvvvo_sz) << "\n";
    } else {
        os << "    (bar-H intermediates not built — d_eri_mo was nullptr; stub mode)\n";
    }
    if (d_X_IP_ != nullptr) {
        const size_t u_ea_sz = (size_t)n_act_vir_ * NV;
        os << "    ‖X(MI)‖     = " << frobenius_norm_device(d_X_IP_, x_ip_sz) << "\n"
           << "    ‖X(EA)‖     = " << frobenius_norm_device(d_X_EA_, x_ea_sz) << "\n"
           << "    ‖U(M,I)‖    = " << frobenius_norm_device(d_U_MI_, u_mi_sz) << "\n"
           << "    ‖U(E,A)‖    = " << frobenius_norm_device(d_U_EA_, u_ea_sz) << "\n"
           << "    ‖F^eff_oo‖  = " << frobenius_norm_device(d_F_eff_oo_, loo_sz) << "\n"
           << "    ‖F^eff_vv‖  = " << frobenius_norm_device(d_F_eff_vv_, lvv_sz) << "\n";
        // Frobenius of (F^eff_oo - Loo) and (F^eff_vv - Lvv) are the Ŝ-induced
        // dressing magnitudes (more interpretable than the absolute norms).
        auto pull_pair = [&](real_t* d_a, real_t* d_b, size_t n,
                             std::vector<real_t>& h_a, std::vector<real_t>& h_b) {
            h_a.resize(n); h_b.resize(n);
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cudaMemcpy(h_a.data(), d_a, n * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b.data(), d_b, n * sizeof(real_t), cudaMemcpyDeviceToHost);
            } else
#endif
            {
                for (size_t i = 0; i < n; ++i) { h_a[i] = d_a[i]; h_b[i] = d_b[i]; }
            }
        };
        std::vector<real_t> h_F_eff, h_bar;
        pull_pair(d_F_eff_oo_, d_Loo_, loo_sz, h_F_eff, h_bar);
        real_t s_oo = 0.0;
        for (size_t i = 0; i < loo_sz; ++i) {
            real_t d = h_F_eff[i] - h_bar[i];
            s_oo += d * d;
        }
        os << "    ‖F^eff_oo−Loo‖ = " << std::sqrt(s_oo) << "  (Ŝ^IP-induced dressing magnitude)\n";
        pull_pair(d_F_eff_vv_, d_Lvv_, lvv_sz, h_F_eff, h_bar);
        real_t s_vv = 0.0;
        for (size_t i = 0; i < lvv_sz; ++i) {
            real_t d = h_F_eff[i] - h_bar[i];
            s_vv += d * d;
        }
        os << "    ‖F^eff_vv−Lvv‖ = " << std::sqrt(s_vv) << "  (Ŝ^EA-induced dressing magnitude)\n";
    }
    os << "    (Sub-phase 3.4: F^eff_oo / F^eff_vv dressed per CFOUR\n"
       << "     `gmi_steom_rhf` / `gea_steom_rhf`; X(MI)/X(EA) are active R1\n"
       << "     inverses per CFOUR `renormalize`.)\n";
}

} // namespace gansu
