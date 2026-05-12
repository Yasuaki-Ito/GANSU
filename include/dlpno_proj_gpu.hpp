/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <array>
#include <vector>

#include "dlpno_pair_data.hpp"
#include "types.hpp"

namespace gansu {

/**
 * @brief GPU helper for the per-triple T2 amplitude projections from pair
 *        PNO basis to TNO basis (the "T_il_ext / T_jl_ext / T_kl_ext / T_part"
 *        steps inside the DLPNO-(T) outer loop).
 *
 * Each per-triple iteration of compute_dlpno_ccsd_t needs ~ 3 · nocc + 6
 * projections of the form
 *
 *     T_tno = (Q_tno^T · S · bar_Q_pair) · Y_pair_oriented · (Q_tno^T · S · bar_Q_pair)^T
 *
 * For TEOS-class molecules (nocc ~ 60) this is ~190 small DGEMM chains per
 * triple → tens of thousands of CPU dispatches → measurable wall time.
 *
 * TripleProjGpu uploads, once at construction:
 *   - All pairs' bar_Q (padded to max_n_pno) on device
 *   - All pairs' S · bar_Q (precomputed via DGEMM with S_AO)
 *   - Y oriented for every ordered (p, q) LMO pair (so per-triple lookups
 *     never need a runtime transpose flag)
 *
 * Per-triple compute uploads only the Q_tno (small) and runs three batched
 * DGEMMs (one for R = Q_tno^T · S·bar_Q, one for RY = R · Y, one for
 * T = RY · R^T) over all 3·nocc + 6 requested pairs.
 */
class TripleProjGpu {
public:
    TripleProjGpu(const std::vector<PairData>&   pairs,
                  const std::vector<PairSetup>&  setups,
                  const std::vector<int>&        pair_lookup,
                  const real_t*                  S_AO_host,
                  int nao, int nocc, int max_n_tno);

    ~TripleProjGpu();
    TripleProjGpu(const TripleProjGpu&)            = delete;
    TripleProjGpu& operator=(const TripleProjGpu&) = delete;

    bool active() const { return active_; }

    /// Project all 3·nocc + 6 T2 amplitudes for one triple.
    ///   T_il_ext_out[l]: t_{lmo_i, l}^{ab}  in TNO basis (n × n).  Empty if
    ///                    pair (i, l) is empty.
    ///   T_jl_ext_out, T_kl_ext_out  — similarly for j, k.
    ///   T_part_out[sp*3 + sq] for sp≠sq: t_{lmo_{sp}, lmo_{sq}}^{ab}.
    bool project_for_triple(
        const real_t* Q_tno_host,
        int n_tno,
        const int triple_lmos[3],
        std::vector<std::vector<real_t>>& T_il_ext_out,
        std::vector<std::vector<real_t>>& T_jl_ext_out,
        std::vector<std::vector<real_t>>& T_kl_ext_out,
        std::array<std::vector<real_t>, 9>& T_part_out);

private:
    bool  active_ = false;
    int   nao_    = 0;
    int   nocc_   = 0;
    int   max_n_  = 0;          // max n_tno
    int   max_pno_= 0;          // max n_pno across all pairs (padded shape)
    int   n_pairs_= 0;

    void* stream_ = nullptr;    // cudaStream_t
    void* cublas_ = nullptr;    // cublasHandle_t

    // Once-uploaded device buffers:
    real_t* d_S_AO_         = nullptr;    // nao × nao
    real_t* d_S_bar_Q_      = nullptr;    // n_pairs × nao × max_pno
    real_t* d_Y_oriented_   = nullptr;    // nocc² × max_pno × max_pno
    int*    d_pair_lookup_  = nullptr;    // nocc × nocc
    int*    d_n_pno_pair_   = nullptr;    // n_pairs

    // Per-call scratch (sized to max batch = 3·nocc + 6):
    real_t* d_Q_tno_        = nullptr;    // nao × max_n
    real_t* d_R_batch_      = nullptr;    // batch × max_n × max_pno
    real_t* d_RY_batch_     = nullptr;    // batch × max_n × max_pno
    real_t* d_T_batch_      = nullptr;    // batch × max_n × max_n

    // cuBLAS batched-DGEMM pointer arrays. Step 1/2/3 use SEPARATE host
    // pinned buffers to avoid the CPU↔DMA race: cudaMemcpyAsync from pinned
    // host to device returns immediately, but the DMA may still be in flight
    // when the CPU loop overwrites the host buffer for the next step.
    // Device-side reuse is safe because the stream serialises read-then-
    // overwrite within itself. cuBLAS expects `const T *const Aarray[]`
    // (= `const T *const *`); we declare matching types so no const_cast
    // is needed at call sites.
    const real_t** d_A_array_ = nullptr;
    const real_t** d_B_array_ = nullptr;
    real_t**       d_C_array_ = nullptr;

    // Pinned host for downloads + pointer array staging:
    real_t* h_pinned_T_     = nullptr;
    const real_t** h_A_array_[3] = {nullptr, nullptr, nullptr};
    const real_t** h_B_array_[3] = {nullptr, nullptr, nullptr};
    real_t**       h_C_array_[3] = {nullptr, nullptr, nullptr};

    // Host copy of pair_lookup for fast indexing in project_for_triple.
    std::vector<int> pair_lookup_host_;
};

} // namespace gansu
