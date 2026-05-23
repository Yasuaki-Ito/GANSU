/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file dlpno_ip_eom_operator.cu
 * @brief DLPNO-IP-EOM-CCSD operator (stage B Phase B0.2, diagonal-only).
 *        See dlpno_ip_eom_operator.hpp for the layout/contract.
 */

#include "dlpno_ip_eom_operator.hpp"

#include <cmath>

#include "dlpno_pair_data.hpp"     // PairSetup, PairData
#include "device_host_memory.hpp"  // tracked_cudaMalloc / tracked_cudaFree
#include "gpu_manager.hpp"         // gpu::gpu_available

namespace gansu {

namespace {
#ifndef GANSU_CPU_ONLY
__global__ void dlpno_ip_diag_matvec_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = D[idx] * x[idx];
}

__global__ void dlpno_ip_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}
#endif
}  // namespace

DLPNOIPEOMCCSDOperator::DLPNOIPEOMCCSDOperator(
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& packing,
    const std::vector<real_t>& eps_o)
{
    total_dim_ = packing.total_dim;
    const int nocc = packing.nocc;
    h_diagonal_.assign(static_cast<size_t>(total_dim_), 0.0);

    // 1h sector: Koopmans diagonal D[i] = -ε_i.
    for (int i = 0; i < nocc; ++i)
        h_diagonal_[i] = -eps_o[i];

    // 2h1p sector: D[ij,a] = -F_ii - F_jj + Λ_a  (≈ -ε_i - ε_j + ε_a).
    // Λ_a is the PNO-canonical virtual energy of pair (i,j). Both orientations
    // of an off-diagonal pair share the same energies (F_ii+F_jj symmetric).
    const int n_pairs = static_cast<int>(res.pairs.size());
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = packing.n_pno[idx];
        if (n == 0) continue;
        const real_t Fii = res.setups[idx].F_ii;
        const real_t Fjj = res.setups[idx].F_jj;
        const std::vector<real_t>& Lam = res.pairs[idx].Lambda;
        const int o_ij = packing.off_ij[idx];
        for (int a = 0; a < n; ++a)
            h_diagonal_[o_ij + a] = -Fii - Fjj + Lam[a];
        if (!packing.diagonal(idx)) {
            const int o_ji = packing.off_ji[idx];
            for (int a = 0; a < n; ++a)
                h_diagonal_[o_ji + a] = -Fjj - Fii + Lam[a];
        }
    }

    // Upload (tracked_cudaMalloc/cudaMemcpy resolve to host ops in CPU builds).
    tracked_cudaMalloc(&d_diagonal_, static_cast<size_t>(total_dim_) * sizeof(real_t));
    cudaMemcpy(d_diagonal_, h_diagonal_.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

DLPNOIPEOMCCSDOperator::~DLPNOIPEOMCCSDOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void DLPNOIPEOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    // Diagonal-only (B0.2): σ = D · x.
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; ++idx)
            d_output[idx] = d_diagonal_[idx] * d_input[idx];
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        dlpno_ip_diag_matvec_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

void DLPNOIPEOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        dlpno_ip_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

} // namespace gansu
