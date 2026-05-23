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
 * @file dlpno_ea_eom_projected_operator.cu
 * @brief DLPNO-EA-EOM Galerkin-projected operator (stage B). EA analog of
 *        dlpno_ip_eom_projected_operator.cu. See the header for the contract.
 */

#include "dlpno_ea_eom_projected_operator.hpp"

#include <cmath>
#include <cstring>

#include "dlpno_pair_data.hpp"        // PairSetup, PairData
#include "dlpno_ea_eom_transform.hpp" // ea_packed_r2_to_canonical / inverse
#include "device_host_memory.hpp"     // tracked_cudaMalloc / tracked_cudaFree
#include "gpu_manager.hpp"            // gpu::gpu_available

namespace gansu {

namespace {
#ifndef GANSU_CPU_ONLY
__global__ void dlpno_ea_proj_precondition_kernel(
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

DLPNOEAEOMProjectedOperator::DLPNOEAEOMProjectedOperator(
    const LinearOperator& canonical,
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& packing,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao,
    const std::vector<real_t>& eps_v)
    : canonical_(canonical), res_(res), packing_(packing),
      U_loc_(U_loc), C_vir_(C_vir),
      h_S_(h_S, h_S + static_cast<size_t>(nao) * nao),
      nao_(nao), nvir_(packing.nvir), nocc_(packing.nocc),
      total_dim_(packing.total_dim),
      canonical_dim_(packing.nvir + packing.nocc * packing.nvir * packing.nvir)
{
    // Preconditioner diagonal: 1p +ε_a; 2p1h -F_ii + Λ_a' + Λ_b' (per-i PNO).
    h_diagonal_.assign(static_cast<size_t>(total_dim_), 0.0);
    for (int a = 0; a < nvir_; ++a) h_diagonal_[a] = eps_v[a];
    for (int i = 0; i < nocc_; ++i) {
        const int n = packing_.n_pno_ii[i];
        if (n == 0) continue;
        const int idx = res_.pair_lookup[static_cast<size_t>(i) * nocc_ + i];
        const real_t Fii = res_.setups[idx].F_ii;
        const std::vector<real_t>& Lam = res_.pairs[idx].Lambda;
        real_t* blk = h_diagonal_.data() + packing_.off_i[i];
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                blk[a * n + b] = -Fii + Lam[a] + Lam[b];
    }
    tracked_cudaMalloc(&d_diagonal_, static_cast<size_t>(total_dim_) * sizeof(real_t));
    cudaMemcpy(d_diagonal_, h_diagonal_.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

DLPNOEAEOMProjectedOperator::~DLPNOEAEOMProjectedOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void DLPNOEAEOMProjectedOperator::apply(const real_t* d_input, real_t* d_output) const {
    std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
    cudaMemcpy(h_in.data(), d_input,
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    std::vector<real_t> packed_r2(h_in.begin() + nvir_, h_in.end());

    std::vector<real_t> R2_canon = ea_packed_r2_to_canonical(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, packed_r2);

    std::vector<real_t> h_cin(static_cast<size_t>(canonical_dim_), 0.0);
    std::memcpy(h_cin.data(), h_in.data(), static_cast<size_t>(nvir_) * sizeof(real_t));  // R1
    std::memcpy(h_cin.data() + nvir_, R2_canon.data(), R2_canon.size() * sizeof(real_t));

    real_t* d_cin = nullptr;
    real_t* d_cout = nullptr;
    tracked_cudaMalloc(&d_cin,  static_cast<size_t>(canonical_dim_) * sizeof(real_t));
    tracked_cudaMalloc(&d_cout, static_cast<size_t>(canonical_dim_) * sizeof(real_t));
    cudaMemcpy(d_cin, h_cin.data(),
               static_cast<size_t>(canonical_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);

    canonical_.apply(d_cin, d_cout);

    std::vector<real_t> h_cout(static_cast<size_t>(canonical_dim_));
    cudaMemcpy(h_cout.data(), d_cout,
               static_cast<size_t>(canonical_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_cin);
    tracked_cudaFree(d_cout);

    std::vector<real_t> sig2_canon(h_cout.begin() + nvir_, h_cout.end());
    std::vector<real_t> packed_sig_r2 = ea_canonical_r2_to_packed(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, sig2_canon);

    std::vector<real_t> h_out(static_cast<size_t>(total_dim_), 0.0);
    std::memcpy(h_out.data(), h_cout.data(), static_cast<size_t>(nvir_) * sizeof(real_t));  // σ1
    std::memcpy(h_out.data() + nvir_, packed_sig_r2.data(),
                packed_sig_r2.size() * sizeof(real_t));
    cudaMemcpy(d_output, h_out.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

void DLPNOEAEOMProjectedOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        dlpno_ea_proj_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

} // namespace gansu
