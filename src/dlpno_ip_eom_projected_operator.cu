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
 * @file dlpno_ip_eom_projected_operator.cu
 * @brief DLPNO-IP-EOM Galerkin-projected operator (stage B Phase B1b).
 *        See dlpno_ip_eom_projected_operator.hpp for the contract.
 */

#include "dlpno_ip_eom_projected_operator.hpp"

#include <cmath>
#include <cstring>

#include "dlpno_pair_data.hpp"        // PairSetup, PairData
#include "dlpno_ip_eom_transform.hpp" // ip_packed_r2_to_canonical / inverse
#include "device_host_memory.hpp"     // tracked_cudaMalloc / tracked_cudaFree
#include "gpu_manager.hpp"            // gpu::gpu_available

namespace gansu {

namespace {
#ifndef GANSU_CPU_ONLY
__global__ void dlpno_ip_proj_precondition_kernel(
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

DLPNOIPEOMProjectedOperator::DLPNOIPEOMProjectedOperator(
    const LinearOperator& canonical,
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& packing,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& eps_o)
    : canonical_(canonical), res_(res), packing_(packing),
      U_loc_(U_loc), C_vir_(C_vir),
      h_S_(h_S, h_S + static_cast<size_t>(nao) * nao),
      nao_(nao), nvir_(nvir), nocc_(packing.nocc),
      total_dim_(packing.total_dim),
      canonical_dim_(packing.nocc + packing.nocc * packing.nocc * nvir)
{
    // Preconditioner diagonal (Koopmans 1h, -F_ii-F_jj+Λ_a 2h1p) — as in B0.2.
    h_diagonal_.assign(static_cast<size_t>(total_dim_), 0.0);
    for (int i = 0; i < nocc_; ++i) h_diagonal_[i] = -eps_o[i];
    const int n_pairs = static_cast<int>(res_.pairs.size());
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = packing_.n_pno[idx];
        if (n == 0) continue;
        const real_t Fii = res_.setups[idx].F_ii;
        const real_t Fjj = res_.setups[idx].F_jj;
        const std::vector<real_t>& Lam = res_.pairs[idx].Lambda;
        const int o_ij = packing_.off_ij[idx];
        for (int a = 0; a < n; ++a) h_diagonal_[o_ij + a] = -Fii - Fjj + Lam[a];
        if (!packing_.diagonal(idx)) {
            const int o_ji = packing_.off_ji[idx];
            for (int a = 0; a < n; ++a) h_diagonal_[o_ji + a] = -Fjj - Fii + Lam[a];
        }
    }
    tracked_cudaMalloc(&d_diagonal_, static_cast<size_t>(total_dim_) * sizeof(real_t));
    cudaMemcpy(d_diagonal_, h_diagonal_.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

DLPNOIPEOMProjectedOperator::~DLPNOIPEOMProjectedOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void DLPNOIPEOMProjectedOperator::apply(const real_t* d_input, real_t* d_output) const {
    // 1. D2H packed input → unpack [R1 | packed_r2].
    std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
    cudaMemcpy(h_in.data(), d_input,
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    std::vector<real_t> packed_r2(h_in.begin() + nocc_, h_in.end());

    // 2. Lift R2 to canonical (P): R2_canon[(I*nocc+J)*nvir+a].
    std::vector<real_t> R2_canon = ip_packed_r2_to_canonical(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, nvir_, packed_r2);

    // 3. Assemble canonical input [R1 | R2_canon], H2D.
    std::vector<real_t> h_cin(static_cast<size_t>(canonical_dim_), 0.0);
    std::memcpy(h_cin.data(), h_in.data(), static_cast<size_t>(nocc_) * sizeof(real_t));
    std::memcpy(h_cin.data() + nocc_, R2_canon.data(), R2_canon.size() * sizeof(real_t));

    real_t* d_cin = nullptr;
    real_t* d_cout = nullptr;
    tracked_cudaMalloc(&d_cin,  static_cast<size_t>(canonical_dim_) * sizeof(real_t));
    tracked_cudaMalloc(&d_cout, static_cast<size_t>(canonical_dim_) * sizeof(real_t));
    cudaMemcpy(d_cin, h_cin.data(),
               static_cast<size_t>(canonical_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);

    // 4. Inner canonical σ (validated P1 operator) on device.
    canonical_.apply(d_cin, d_cout);

    std::vector<real_t> h_cout(static_cast<size_t>(canonical_dim_));
    cudaMemcpy(h_cout.data(), d_cout,
               static_cast<size_t>(canonical_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_cin);
    tracked_cudaFree(d_cout);

    // 5. Project σ2 down to the PNO blocks (P); σ1 passes through.
    std::vector<real_t> sig2_canon(h_cout.begin() + nocc_, h_cout.end());
    std::vector<real_t> packed_sig_r2 = ip_canonical_r2_to_packed(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, nvir_, sig2_canon);

    // 6. Assemble packed σ [σ1 | packed_sig_r2], H2D → output.
    std::vector<real_t> h_out(static_cast<size_t>(total_dim_), 0.0);
    std::memcpy(h_out.data(), h_cout.data(), static_cast<size_t>(nocc_) * sizeof(real_t));
    std::memcpy(h_out.data() + nocc_, packed_sig_r2.data(),
                packed_sig_r2.size() * sizeof(real_t));
    cudaMemcpy(d_output, h_out.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

void DLPNOIPEOMProjectedOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        dlpno_ip_proj_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

} // namespace gansu
