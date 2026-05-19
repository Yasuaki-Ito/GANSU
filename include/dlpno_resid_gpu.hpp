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
 * @brief Per-stage GPU timing accumulators for compute_async().
 *
 * Populated by `compute_finalize()` via cudaEventElapsedTime between
 * stage-boundary events recorded on the default stream. Times are in
 * seconds (converted from cudaEventElapsedTime's ms output) and accumulate
 * across all compute_async/compute_finalize round-trips since the last
 * `reset_stage_times()` call. n_iter is the number of finalize calls
 * contributing to the totals.
 *
 * Stages (in execution order on the default stream):
 *   slice      : 4 pi-stack slicing kernels
 *   w_block    : 8 strided-batched DGEMM (W_block_{i,i2,j,j2} build)
 *   r_contract : 8 strided-batched DGEMM (R += W·PI contractions)
 *   inter_fock : 2 inter-pair Fock kernels (i + j)
 *   oooo       : oooo_lad_kernel
 *   d2h        : async D2H of R_ph_pad slab
 */
struct ResidStageTimes {
    double slice      = 0.0;
    double w_block    = 0.0;
    double r_contract = 0.0;
    double inter_fock = 0.0;
    double oooo       = 0.0;
    double d2h        = 0.0;
    int    n_iter     = 0;
};

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
     * @brief Stage-level timing accumulators for the GPU pipeline.
     *
     * Use to attribute the cholesterol-class 187 s residual budget to its
     * 6 internal stages. `get_stage_times()` reads the current totals
     * without resetting them; `reset_stage_times()` zeros all counters
     * (intended to be called once after the per-round dump). When
     * `active()` is false both return zeros / are no-ops.
     *
     * Timing is implemented via cudaEvent (default flags) recorded on
     * the default stream between stages. Elapsed time is read inside
     * compute_finalize() after the existing completion-event sync, so
     * by the time we sample the events all stages are guaranteed
     * complete. Overhead is ~6 × cudaEventRecord per iter (1-2 μs each
     * on host launch, no device stall), negligible vs the per-iter
     * pipeline cost.
     */
    ResidStageTimes get_stage_times() const;
    void            reset_stage_times();

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

    // Step S10 scaffolding (2026-05-17) — infrastructure for the future
    // active-only compact storage refactor that will allow ResidGpu to
    // activate on cholesterol-class systems. The kernels in this version
    // still index by orig pair idx (so the active list / position map are
    // NOT used by any execution path yet); they exist only so the next
    // session can flip the buffer allocation + kernel indexing in one
    // coherent change without re-deriving the slab-active set.
    //
    //   active_pair_list_[a] = orig pair idx of the a-th active pair in
    //                          THIS device's slab [pair_begin_, pair_end_).
    //                          Built only for pairs with n_pno > 0.
    //   active_pos_[orig_idx] = position `a` in active_pair_list_, or -1
    //                          if orig_idx has n_pno == 0 OR lies outside
    //                          the slab. Sized to N_pair_.
    //   n_active_in_slab_    = active_pair_list_.size().
    //
    // pair_begin_ / pair_end_ are sourced from the borrowed PiCacheGpu so
    // they match the slab assignment used by the kernels. The current
    // budget calculation (full N_pair × per_pair, see ResidGpu ctor) is
    // additionally logged alongside a projected `active × per_pair / 8`
    // figure so we can confirm the refactor will close the memory gap
    // before committing to the full reindex.
    std::vector<int> active_pair_list_;
    std::vector<int> active_pos_;
    int              n_active_in_slab_ = 0;

    // Step S11 Phase 1 scaffolding (2026-05-17 night-3) — bucket-by-n_ij
    // grouping for the upcoming packed V_meta/T_meta refactor. Built in
    // the constructor; consumed by `pack_V_meta_kernel` (Phase 1 — writes
    // both the legacy padded AND the new packed buffers, gated on the
    // packed alloc succeeding) and by Phase 2's bucket cuBLAS loops
    // (W_block_build + r_contract; not yet wired).
    //
    // Buckets group THIS device's active slab pairs by their n_pno value
    // (= n_ij). Within a bucket all pairs share one per-slot stride
    // (nn_b² for meta, n_b·nn_b for block etc.), letting Phase 2 use
    // `cublasDgemmStridedBatched` per bucket without pointer arrays.
    //
    //   bucket_n_ij_[b]             = the n_ij value defining bucket b
    //   bucket_count_[b]            = pairs in bucket b
    //   bucket_first_[b]            = start index into
    //                                 bucket_active_pair_list_ for bucket b
    //                                 (bucket_first_[b+1] - bucket_first_[b]
    //                                  == bucket_count_[b]).
    //   bucket_active_pair_list_    = active slab pairs, sorted by n_ij
    //                                 (= concatenation of buckets in
    //                                 ascending n_ij order). Same set as
    //                                 `active_pair_list_` but in bucket
    //                                 order, not orig pair-idx order.
    //   per_pair_meta_off_[orig]    = element offset into the packed
    //                                 V_meta_*_packed buffer for the pair
    //                                 at orig pair idx (or 0 if the pair
    //                                 is inactive; callers must guard via
    //                                 active_pos_[orig] >= 0).
    //   packed_meta_total_          = total packed element count across
    //                                 all buckets (= sum of nn_ij² over
    //                                 active slab pairs).
    std::vector<int>    bucket_n_ij_;
    std::vector<int>    bucket_count_;
    std::vector<int>    bucket_first_;
    std::vector<int>    bucket_active_pair_list_;
    std::vector<size_t> per_pair_meta_off_;
    size_t              packed_meta_total_ = 0;

    // Step S11 Phase 2a (2026-05-18) — additional per-pair offset tables and
    // totals for the remaining 14 packed cuBLAS buffers (W_bare/W_block/pi_N/
    // pi_T/PI_stack/PI_TT/R). Built alongside per_pair_meta_off_ in the same
    // bucket pass. Phase 2a allocates the packed buffers but the cuBLAS GEMMs
    // still read the legacy padded buffers, so bit-exactness is preserved.
    //
    // Element offsets into the corresponding packed device buffer for the
    // pair at orig pair idx:
    //   per_pair_block_off_[orig] : stride n_b · nn_b (row-major n_b × nn_b)
    //   per_pair_stack_off_[orig] : stride nn_b · n_b (row-major nn_b × n_b;
    //                              numerically same total as block but kept
    //                              separate because Phase 2c's cuBLAS lda
    //                              differs: nn_b vs n_b)
    //   per_pair_R_off_   [orig] : stride n_b²
    // All values 0 for inactive pairs; callers must guard via active_pos_.
    std::vector<size_t> per_pair_block_off_;
    std::vector<size_t> per_pair_stack_off_;
    std::vector<size_t> per_pair_R_off_;
    size_t              packed_block_total_ = 0;
    size_t              packed_stack_total_ = 0;
    size_t              packed_R_total_     = 0;

    // Per-bucket base offsets (host-only, length n_buckets + 1 as prefix
    // sums). Phase 2c's cuBLAS bucket loops use bucket_base_*_[b] to derive
    // the base device pointer for the bucket's strided-batched GEMM:
    //   d_X_packed + bucket_base_*_[b]   (with batch = bucket_count_[b],
    //                                     stride = per-pair element count).
    // bucket_base_*_[n_buckets] equals the corresponding packed_*_total_.
    std::vector<size_t> bucket_base_meta_;
    std::vector<size_t> bucket_base_block_;
    std::vector<size_t> bucket_base_stack_;
    std::vector<size_t> bucket_base_R_;
};

} // namespace gansu
