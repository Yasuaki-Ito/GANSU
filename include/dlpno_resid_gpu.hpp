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

#include <vector>
#include <Eigen/Dense>

#include "types.hpp"
#include "dlpno_pair_data.hpp"
#include "dlpno_picache_gpu.hpp"

namespace gansu {

/**
 * @brief Step 6.2 — GPU port for the per-iter ph-ladder contributions in
 *        iterate_dlpno_ccsd_t2's residual loop.
 *
 * The original CPU resid loop has, per pair (idx) and per iter, ~12 medium
 * DGEMMs (n × nocc·n) for the i-side and j-side particle-hole ladder, plus
 * stack assembly via build_stack_for_I / build_outer_stack helpers. With
 * 465 pairs × 60 iter × 12 calls and avg n=5.9 the cost was ~4.4 s/run
 * (CPU dispatch overhead dominated).
 *
 * This class:
 *   - Borrows pi_T_stack + per-pair metadata from a PiCacheGpu instance
 *     (Step 6.1). pi_T_stack carries every oriented projection π_{k,l} we
 *     need, so ph-ladder stacks (pi_N_i, pi_T_i, PI_kj_*, PI_ki_*) are
 *     extracted via custom kernels rather than rebuilt from pi_cache.
 *   - Owns iter-invariant V_meta_T, V_meta_TT, T_meta, W_ovov_i/j,
 *     W_ovvo_i/j on device (padded to (max_n·nocc) × (max_n·nocc) and
 *     max_n × nocc·max_n respectively). One-time upload at constructor.
 *   - Per-iter, runs a sequence of cuBLAS strided batched DGEMMs to
 *     produce R_ph_pad (max_n × max_n per pair) and copies the unpadded
 *     contribution back to host.
 *
 * Falls back to active()=false (caller takes the existing CPU path) when
 * - GPU is unavailable
 * - PiCacheGpu wasn't constructed in stacked mode (no d_pi_T_stack)
 * - device memory is insufficient
 * - GANSU_CPU_ONLY is defined.
 */
class ResidGpu {
public:
    using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    /**
     * @brief One-time setup. Borrows from `pgpu` and uploads iter-invariant
     *        ph-ladder integrals.
     *
     * Requires phase24 to be non-null and fully populated (V_ovov_pair,
     * T_pair, W_ovov_i/j, W_ovvo_i/j). `pgpu.stacked()` must be true; we
     * read the same pair_lookup/setup_i/n_pno that pgpu uploaded.
     *
     * Step 6.5: also takes `F_LMO_host` (nocc × nocc, row-major) which is
     * uploaded once and reused across iters by the inter-pair Fock kernels.
     *
     * Caller is responsible for the lifetime of `pgpu`: ResidGpu only
     * stores const pointers into it; the PiCacheGpu instance must outlive
     * the ResidGpu.
     */
    ResidGpu(const PiCacheGpu&             pgpu,
             const std::vector<PairSetup>& setups,
             const std::vector<PairData>&  pairs,
             const Phase24Integrals&       phase24,
             const std::vector<real_t>&    F_LMO_host,
             int nocc, int max_n);

    ~ResidGpu();

    ResidGpu(const ResidGpu&) = delete;
    ResidGpu& operator=(const ResidGpu&) = delete;

    /**
     * @brief Whether the GPU path is live. False = caller should run the
     * CPU ph-ladder code as before.
     */
    bool active() const noexcept { return active_; }

    /**
     * @brief Per-iter compute: produce R_ph[idx] for every pair, where
     *        R_ph[idx] is the (n_ij × n_ij) sum of all ph-ladder i+j
     *        contractions (R += 2·W_block·PI - W_block2·PI - … on i-side
     *        plus the j-side mirror).
     *
     * Pre-condition: the most recent `pgpu.rebuild_with_stack(...)` call
     * has refreshed pi_T_stack on device (we read it directly).
     *
     * R_ph_out is resized to `pairs.size()`. Empty pairs (n_ij == 0) get
     * a 0×0 matrix. Caller adds R_ph_out[idx] to its own R inside the
     * per-pair OMP residual loop.
     */
    void compute(std::vector<RowMatXd>& R_ph_out);

    /**
     * @brief Step 6.4 — async-pair API for CPU/GPU overlap.
     *
     * `compute_async()` queues all GPU work (slice kernels + 14 batched
     * DGEMMs + async D2H of R_ph_pad) on the default stream and records
     * a completion event, then returns without blocking the host. The
     * caller can then run unrelated CPU work (e.g. dF_ki and DFpair
     * dressing) while the GPU pipeline runs.
     *
     * `compute_finalize(R_ph_out)` synchronises on the recorded event,
     * resizes R_ph_out, and unpacks the pinned-host R_ph_pad into the
     * per-pair (n_ij × n_ij) matrices.
     *
     * If `active()` is false both methods are no-ops and the caller's
     * R_ph_out remains empty (CPU ph-ladder fallback path kicks in).
     */
    void compute_async();
    void compute_finalize(std::vector<RowMatXd>& R_ph_out);

    /**
     * @brief Step 6.5 — full async variant including inter-pair Fock i+j.
     *
     * `dF_ki_host` is the (nocc × nocc) row-major dressing matrix
     * dF_ki[k * nocc + i] = ΔF_{ki}, refreshed by the caller before
     * each iter. Combined with the iter-invariant F_LMO uploaded at
     * construction time, the kernel forms F_eff[i, k] = F_LMO[i, k]
     * + dF_ki[k, i] (and the j-side mirror) and accumulates the
     * inter-pair Fock contributions into the same R_pad buffer that
     * receives the ph-ladder result. R_ph_out finalised by
     * compute_finalize then contains R += R_phladder + R_interpair.
     */
    void compute_async(const std::vector<real_t>& dF_ki_host);

private:
    // Step 6.5 — internal helper: queue the ph-ladder pipeline kernels +
    // contractions onto the default stream WITHOUT recording the
    // completion event or doing the D2H. Lets the public compute_async
    // overloads chain inter-pair Fock kernels after the ph-ladder before
    // the trailing D2H + event.
    void compute_async_phladder_only_();

    // Step 6.5 — record the trailing async D2H of R_ph_pad and the
    // completion event. Called once at the end of each compute_async
    // entry point.
    void compute_async_finalize_pipeline_();

    struct Impl;
    Impl* p_ = nullptr;
    bool active_ = false;

    // Cached const pointers into the borrowed PiCacheGpu (read-only).
    const PiCacheGpu* pgpu_ = nullptr;

    // Capture the per-pair n_pno + per-pair (sij.i, sij.j) on host so we
    // can size the host output and (for sij.i / sij.j) initialise device
    // arrays at construction.
    std::vector<int> n_pno_;
    std::vector<int> setup_i_per_pair_;  // = setups[idx].i
    std::vector<int> setup_j_per_pair_;  // = setups[idx].j
    int N_pair_ = 0;
    int max_n_  = 0;
    int nocc_   = 0;
};

} // namespace gansu
