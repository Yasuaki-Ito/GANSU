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

namespace gansu {

/**
 * @brief GPU-accelerated builder for the per-iter pi_cache
 *        pi[i_ij][i_kl] = barS[i_ij][i_kl] · Y_old[i_kl] · barS[i_ij][i_kl]^T
 *
 * Replaces the CPU OMP-over-pairs Eigen loop in iterate_lmp2 /
 * iterate_dlpno_ccsd_t2 (`src/dlpno_pair_data.cu`). The OpenBLAS
 * dispatch overhead (~10 µs/call × N_pair²) was the dominant cost at
 * hexamer scale (24.3 s out of 73 s post_process_after_scf, see
 * memory/project_dlpno_step6_gpu_port.md).
 *
 * Strategy: pad every (n × m) sub-block to (max_n × max_n), upload
 * `barS_cache` once, and process per-iter Y_old → pi_cache using two
 * `cublasDgemmStridedBatched` calls per outer i_ij. Result is downloaded
 * back into a host-side `std::vector<std::vector<RowMatXd>>` matching
 * the CPU layout exactly (caller code unchanged downstream).
 *
 * Falls back to the equivalent CPU Eigen kernel when GANSU_CPU_ONLY is
 * defined or when `gpu::gpu_available()` returns false.
 *
 * Lifetime: intended to be constructed once per iter loop, just before
 * the iter `for`; reused across iters; destroyed when leaving the
 * surrounding scope.
 */
class PiCacheGpu {
public:
    using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>;

    /**
     * @brief One-time setup: pad and upload barS_cache.
     * @param barS_cache  [N_pair][N_pair] each (n_ij × n_kl), 0×0 if either pair empty
     * @param n_pno_per_pair  size N_pair, n_pno_[i] = pairs[i].n_pno (0 = empty)
     * @param max_n  max over n_pno_per_pair (caller computed)
     * @param pair_lookup  optional [nocc·nocc] map (k,l) → canonical pair idx.
     *                     When non-null, enables Step 6.1 stacked mode.
     * @param setup_i_per_pair  optional [N_pair] list of setups[idx].i — needed
     *                          to decide pi_canon^T vs pi_canon orientation.
     * @param nocc  occupied count (only used in stacked mode).
     * @param pair_begin / pair_end  output-row slab `[pair_begin, pair_end)`.
     *                  For multi-GPU pair partitioning. When `pair_end < 0`,
     *                  defaults to `[0, N_pair)` (single-GPU behavior).
     *                  All input data (barS for ALL i_kl columns) is still
     *                  uploaded — only the OUTPUT pi rows / pi_T_stack rows
     *                  for i_ij in this slab are computed and downloaded.
     * @param device_id  Logical CUDA device id (used with MultiGpuManager when
     *                  num_gpus > 1). Defaults to 0 (use current device's
     *                  thread-local cublas handle).
     *
     * Throws std::runtime_error on cudaMalloc / cuBLAS failure. In that
     * case the caller should fall back to the CPU Eigen kernel.
     */
    PiCacheGpu(const std::vector<std::vector<RowMatXd>>& barS_cache,
               const std::vector<int>& n_pno_per_pair,
               int max_n,
               const std::vector<int>* pair_lookup = nullptr,
               const std::vector<int>* setup_i_per_pair = nullptr,
               int nocc = 0,
               int pair_begin = 0,
               int pair_end = -1,
               int device_id = 0,
               // Phase 2 (CCSD sparse barS): setups[idx].j per pair, needed by
               // the sparse pi_T_stack scatter to place each coupling block at
               // both (p·nocc+q) and (q·nocc+p) slots. Optional; only used when
               // GANSU_DLPNO_CCSD_BARS_SPARSE is on for a stacked instance.
               const std::vector<int>* setup_j_per_pair = nullptr);

    ~PiCacheGpu();

    PiCacheGpu(const PiCacheGpu&) = delete;
    PiCacheGpu& operator=(const PiCacheGpu&) = delete;

    /**
     * @brief Per-iter rebuild: pi_cache[i_ij][i_kl] from current Y_old.
     *
     * @param Y_old  size N_pair, each is the n_kl² flat row-major Y matrix
     *               for pair i_kl (0-length if pair empty).
     * @param pi_cache_out  pre-resized to [N_pair][N_pair]; this method
     *               assigns each `pi_cache_out[i_ij][i_kl]` to either the
     *               (n_ij × n_ij) result or 0×0 if either pair is empty.
     *
     * Caller must invoke this from a single thread (no surrounding OMP
     * region). Internal pad/unpad is OMP-parallel.
     */
    void rebuild(const std::vector<std::vector<real_t>>& Y_old,
                 std::vector<std::vector<RowMatXd>>& pi_cache_out);

    /**
     * @brief LMP2 needed-column variant of rebuild().
     *
     * Identical to rebuild() (pad Y → H2D → 2× batched DGEMM filling the
     * compact d_pi_pad on device) EXCEPT that, instead of D2H'ing + host
     * scattering every active (i_ij, i_kl) column, it transfers ONLY the
     * columns the LMP2 inter-pair Fock residual actually reads — namely the
     * (k,j) and (i,l) pairs supplied per i_ij in `needed_ikl_per_pair`.
     *
     * A device gather kernel compacts those columns of d_pi_pad into a small
     * d_pi_needed buffer (≈ 2·nocc of N_pair columns), which is D2H'd (~6×
     * smaller) and scattered into pi_cache_out[i_ij][i_kl] only for the needed
     * i_kl. Non-needed entries are left as the caller initialised them (the
     * residual never reads them). Bit-exact w.r.t. rebuild() for every entry
     * the residual consumes.
     *
     * @param needed_ikl_per_pair  size N_pair; needed_ikl_per_pair[i_ij] is the
     *        (deduped) list of original pair indices i_kl whose projection
     *        pi[i_ij][i_kl] the residual reads for output pair i_ij. Built once
     *        by the caller (iter-invariant). The device-side needed map is
     *        constructed lazily on the first call and reused across iters.
     *
     * Falls back to rebuild() when GANSU_CPU_ONLY / !active().
     */
    void rebuild_needed(const std::vector<std::vector<real_t>>& Y_old,
                        std::vector<std::vector<RowMatXd>>& pi_cache_out,
                        const std::vector<std::vector<int>>& needed_ikl_per_pair);

    /**
     * @brief Step 6.1 — also produce pi_T_stack on the same call.
     *
     * pi_T_stack_out[i_ij](a, (k·nocc + l)·n_ij + d) = π_{k, l}^{oriented}[a, d]
     * where oriented = pi_canon (skl.i == k) or pi_canon^T (skl.i != k),
     * and pi_canon = barS · Y_old · barS^T at (i_ij, idx_kl=pair_lookup[k·nocc+l]).
     *
     * Empty (i_ij or i_kl) ⇒ corresponding output region is zero.
     * pi_T_stack_out[i_ij] is resized to (n_ij × nocc²·n_ij) on output.
     *
     * Requires the constructor to have been called with non-null
     * pair_lookup / setup_i_per_pair. Otherwise falls back to the CPU
     * pi_cache build + a CPU pi_T_stack assembly that the caller can
     * skip if it has its own stack-build code.
     *
     * @param skip_pi_cache_host  When true AND the GPU path is active AND
     *        stacked mode is live, skip the d_pi_pad → host D2H plus the
     *        host-side unpadding into pi_cache_out. The on-device d_pi_pad
     *        is still produced (the pack kernel needs it for pi_T_stack),
     *        and pi_T_stack_out is still filled as usual. Use this when
     *        the caller's downstream consumers read pi_cache only via
     *        the device buffers (ResidGpu) or never read it at all in the
     *        current iter — see iterate_dlpno_ccsd_t2 with any_rgpu_active.
     *        On the CPU fallback / non-stacked path the flag is ignored
     *        and pi_cache_out is populated as before.
     */
    void rebuild_with_stack(const std::vector<std::vector<real_t>>& Y_old,
                            std::vector<std::vector<RowMatXd>>& pi_cache_out,
                            std::vector<RowMatXd>& pi_T_stack_out,
                            bool skip_pi_cache_host = false,
                            const std::vector<std::vector<int>>*
                                coupling_ikl_per_pair = nullptr);

    /**
     * @brief DFpair GPU port — upload the iter-invariant T_meta_dpair to this
     *        instance's device (slab range only) ONCE per iterate() call.
     *
     * T_meta_dpair[idx] is row-major [(nocc²·n_ij) × n_ij] (built host-side,
     * iter-invariant). Each element block has the SAME per-pair element count
     * (nocc²·n_ij²) as pi_T_stack[idx], so the device segments reuse the
     * pi_T_stack offset cumulant. Only the slab [pair_begin_, pair_end_) is
     * uploaded; the device buffer is slab-sized. Returns false (caller keeps
     * the CPU DFpair loop) when stacked-mode is off or a cudaMalloc fails.
     */
    bool upload_T_meta_dpair(const std::vector<RowMatXd>& T_meta_dpair);

    /**
     * @brief Stage D (D1b): pre-build the sparse kl-slot machinery (kl-slot list,
     *        slot maps, sparse d_pi_T_stack) BEFORE the iter loop. No-op unless
     *        pitstack_sparse_ is on. Needed because upload_T_meta_dpair() (a
     *        one-time pre-loop call) reads the kl-slot list to pack T_meta_dpair
     *        in sparse layout, but the list is otherwise first built in
     *        rebuild_with_stack() (first iter, AFTER the upload). Idempotent
     *        (setup_sparse_stacked_ is guarded); rebuild's later call early-returns.
     */
    void ensure_sparse_stacked(const std::vector<std::vector<int>>& coupling);

    /**
     * @brief DFpair GPU port — compute DF_per_pair[idx] = −(pi_T_stack[idx] ·
     *        T_meta_dpair[idx]) for this instance's slab [pair_begin_,pair_end_)
     *        as a per-pair cublasDgemm, D2H each [n_ij × n_ij] block into
     *        DF_per_pair_out[idx]. MUST be called AFTER rebuild_with_stack so
     *        d_pi_T_stack reflects the current iter's Y_old. No-op if
     *        upload_T_meta_dpair was not (successfully) called first.
     *        DF_per_pair_out must be pre-sized to N_pair (disjoint slab writes).
     */
    void compute_dfpair(std::vector<RowMatXd>& DF_per_pair_out);

    /**
     * @brief Whether the GPU path is active. False = CPU fallback is in use.
     * Returns true when the constructor successfully built device buffers.
     */
    bool active() const noexcept { return active_; }

    /**
     * @brief Whether Step 6.1 stacked-mode buffers are live. False = stacked
     * outputs go via the CPU fallback path inside `rebuild_with_stack`.
     */
    bool stacked() const noexcept { return stacked_; }

    /**
     * @brief Step 6.2 — read-only device buffer getters for downstream
     * GPU users (e.g. ResidGpu) that want to operate on the same in-memory
     * pi_T_stack and per-pair metadata.
     *
     * The pointers reflect the latest state after the most recent
     * `rebuild_with_stack` call. They are nullptr when stacked() is false.
     *
     * Layout reference:
     *   - device_pi_T_stack: per-pair flat (n_ij × nocc²·n_ij) row-major,
     *     contiguous segments at `device_idx_offset_pi_T()[i_ij]`.
     *   - device_pair_lookup: nocc² ints, pair_lookup[k·nocc + l] = canonical idx.
     *   - device_setup_i / device_n_pno: N_pair ints, setups[idx].i / pairs[idx].n_pno.
     *   - device_idx_offset_pi_T: (N_pair+1) size_t cumulative offsets into pi_T_stack.
     */
    const real_t*  device_pi_T_stack()      const noexcept;
    const real_t*  device_Y_pad()           const noexcept;  // padded Y_old after last rebuild
    const real_t*  device_Y_pad_T()         const noexcept;  // Step 6.7: per-pair transpose for oooo_lad coalescing
    const int*     device_pair_lookup()     const noexcept;
    const int*     device_setup_i()         const noexcept;
    const int*     device_n_pno()           const noexcept;
    const size_t*  device_idx_offset_pi_T() const noexcept;
    int            device_max_n()           const noexcept { return max_n_; }
    int            device_N_pair()          const noexcept { return N_pair_; }
    int            device_nocc()            const noexcept { return nocc_; }

    // Stage D — sparse pi_T_stack kl-slot list + slot maps (orig-idx indexed).
    // nullptr until setup_sparse_stacked_ has run (sparse stacked mode). Used by
    // the D1b ResidGpu consumers to read the coupling-list pi_T_stack layout.
    const int*     device_n_slots()          const noexcept;   // [N_pair]
    const size_t*  device_slot_offset()      const noexcept;   // [N_pair+1] (kl_slot CSR)
    const int*     device_kl_slot()          const noexcept;   // [Σ n_slots] kl per slot
    const size_t*  device_idx_offset_sparse()const noexcept;   // [N_pair+1] pi_T sparse offsets
    const int*     device_slot_jcol()        const noexcept;   // [N_pair·nocc] (k,j)
    const int*     device_slot_irow()        const noexcept;   // [N_pair·nocc] (i,l)
    const int*     device_slot_icol()        const noexcept;   // [N_pair·nocc] (k,i)
    const int*     device_slot_jrow()        const noexcept;   // [N_pair·nocc] (j,l)
    bool           pitstack_sparse_ready()   const noexcept;   // kl_slot_built
    bool           pitstack_sparse()         const noexcept { return pitstack_sparse_; } // sparse storage active
    /// Host n_slots per ORIG pair idx (size N_pair) or nullptr if kl-slot list
    /// not built. Used by ResidGpu (D3a) to size the sparse V_stacked_oooo by
    /// the per-pair coupling-slot count instead of nocc².
    const int*     host_n_slots()            const noexcept;   // [N_pair] or nullptr

    /// Slab info (output-row range and CUDA device).
    int            pair_begin()              const noexcept { return pair_begin_; }
    int            pair_end()                const noexcept { return pair_end_; }
    int            device_id()               const noexcept { return device_id_; }

private:
    struct Impl;
    Impl* p_ = nullptr;
    bool active_ = false;
    bool stacked_ = false;

    // Phase 1 (sparse barS) — LMP2-only sparse storage mode. When true, the
    // dense d_barS_pad / d_pi_pad device buffers are NOT allocated; instead a
    // CSR-style d_barS_csr (only the needed_ikl coupling columns per row) is
    // built lazily on the first rebuild_needed(), cutting device barS memory
    // by ~N_act_kl/max_needed (≈6.5×, fixes the Decacene OOM). Set in the
    // constructor when GANSU_DLPNO_LMP2_BARS_SPARSE=1 AND this is the
    // non-stacked LMP2 instance. Implies the ragged GEMM path. Bit-exact.
    bool sparse_lmp2_ = false;

    // Phase 2 (CCSD sparse barS) — stacked (CCSD T2) analogue of sparse_lmp2_.
    // When true, the dense d_barS_pad / d_pi_pad are NOT allocated; the CSR
    // barS + ragged pi (coupling columns only) are built lazily on the first
    // rebuild_with_stack() that supplies a coupling list, and a scatter kernel
    // writes the ragged pi into the DENSE d_pi_T_stack at the (k·nocc+l) slots
    // (non-coupling slots stay zero). ResidGpu reads d_pi_T_stack unchanged.
    // Set in the constructor when GANSU_DLPNO_CCSD_BARS_SPARSE=1 AND this is a
    // stacked instance. Screening (norm/distance) ⇒ NOT bit-exact (ΔE<1e-4).
    bool sparse_stacked_ = false;

    // Stage D (D1b) — sparse pi_T_stack layout active. When true, d_pi_T_stack
    // is stored in the coupling-list (kl-slot) layout (per pair n_ij²·n_slots,
    // offsets d_idx_offset_sparse) instead of the dense n_ij²·nocc². The scatter
    // writes slot positions and ResidGpu consumers read via the slot maps.
    // Set = GANSU_DLPNO_CCSD_PITSTACK_SPARSE && sparse_stacked_. The dense
    // d_pi_T_stack / h_pi_T_stack are then NOT allocated (pi_T 48.8→~5GB Decacene).
    bool pitstack_sparse_ = false;

    // CPU fallback for !active() — keeps the public API stable when GPU
    // is unavailable. Same pi_cache_out layout.
    void rebuild_cpu_(const std::vector<std::vector<real_t>>& Y_old,
                      std::vector<std::vector<RowMatXd>>& pi_cache_out);

    // CPU fallback for !stacked() — produces pi_T_stack from a host pi_cache.
    void build_stack_cpu_(const std::vector<std::vector<RowMatXd>>& pi_cache,
                          std::vector<RowMatXd>& pi_T_stack_out);

    // GPU-only kernels for the per-iter rebuild (Y pad → H2D → 2× batched
    // DGEMM filling d_pi_pad on device). No D2H. Used when skip_pi_cache_host
    // is requested via rebuild_with_stack so the host pi_cache_out can be
    // skipped while still feeding d_pi_pad to the pack kernel below.
    // Step Z: throws if d_pi_pad is tiled (tile_size_ < N_act_ij_) — use
    // rebuild_with_stack in that case, which orchestrates the tile loop.
    void rebuild_gpu_kernels_(const std::vector<std::vector<real_t>>& Y_old);

    // Step Z helpers — split rebuild_gpu_kernels_ into per-iter Y prep
    // and per-tile DGEMM compute, so the tile loop in rebuild_with_stack
    // can interleave compute with pack into d_pi_T_stack and reuse the
    // tile buffer for the next tile.
    void pack_Y_and_transpose_(
        const std::vector<std::vector<real_t>>& Y_old);
    void compute_pi_tile_(int tile_start, int tile_end);

    // Phase 1 (sparse barS) — ragged coupling-list variant of compute_pi_tile_.
    // Instead of the dense per-row strided-batched GEMM over ALL N_act_kl
    // columns followed by gather_needed_kernel, this computes pi ONLY for the
    // ~2·nocc coupling columns the LMP2 residual reads (the needed_ikl set),
    // writing the result directly into d_pi_needed via a per-row pointer-array
    // cublasDgemmBatched. Bit-exact w.r.t. compute_pi_tile_ + gather for every
    // consumed (i_ij, i_kl) block. Requires the ragged pointer arrays to have
    // been built (one-time, iter-invariant) and a single tile
    // (tile_size_ >= N_act_ij_). Gated by GANSU_DLPNO_LMP2_BARS_RAGGED_GEMM.
    void compute_pi_needed_ragged_(int tile_start, int tile_end);

    // Phase 2 (CCSD sparse barS) — one-time setup of the stacked sparse path
    // from a coupling list: builds needed_ikl_host / d_needed_* / d_coupling_ikl,
    // the CSR d_barS_csr (coupling blocks only), d_pi_needed, and the ragged
    // GEMM pointer arrays (barS → CSR). Idempotent (guarded). Mirrors the
    // one-time blocks of rebuild_needed but sourced from the CCSD coupling list.
    void setup_sparse_stacked_(
        const std::vector<std::vector<int>>& coupling_ikl_per_pair);

    // D2H d_pi_pad → h_pi_pad (slab range) and unpad into pi_cache_out.
    // Only called when the host pi_cache is actually needed downstream.
    void download_pi_cache_(std::vector<std::vector<RowMatXd>>& pi_cache_out);

    // For the CPU fallback only: keep a const view into the input barS_cache.
    const std::vector<std::vector<RowMatXd>>* barS_cache_ref_ = nullptr;
    std::vector<int> n_pno_;
    int N_pair_ = 0;
    int max_n_ = 0;

    // CPU-path acceleration: precomputed list of indices with n_pno > 0.
    // - active_i_kl_: full N_pair range (used as inner loop in rebuild_cpu_)
    // - active_i_ij_: restricted to the slab [pair_begin_, pair_end_)
    // Built once in the constructor; iterating only these in rebuild_cpu_
    // turns the 34M outer×inner sweep on cholesterol into ~3.8M active calls.
    std::vector<int> active_i_kl_;
    std::vector<int> active_i_ij_;

    // Step Z — compact storage support for the GPU path.
    //
    // The full N_pair × N_pair × max_n² padded layout for d_barS_pad and
    // d_pi_pad is structurally infeasible at cholesterol scale (5886² ·
    // 26² · 8 = 175 GB per buffer). Step Z stores only the active
    // (i_ij, i_kl) sub-grid (n_active² × max_n² ≈ 20 GB at cholesterol)
    // and additionally tiles d_pi_pad in the active_i_ij dimension so it
    // fits in ~5 GB at any moment.
    //
    //   active_kl_pos_[i_kl] = position of i_kl in active_i_kl_, or -1
    //                          if i_kl has n_pno = 0.
    //   active_ij_pos_[i_ij] = position of i_ij in active_i_ij_, or -1
    //                          if i_ij is outside the slab or has n_pno = 0.
    //   tile_size_           = number of active i_ij rows that d_pi_pad
    //                          can hold at once (chosen at construction
    //                          from free GPU memory).
    std::vector<int> active_kl_pos_;
    std::vector<int> active_ij_pos_;
    int N_act_ij_ = 0;
    int N_act_kl_ = 0;
    int tile_size_ = 0;

    // Step 6.1 metadata captured at construction (used by both GPU and
    // CPU fallback paths of build_stack_cpu_).
    std::vector<int> pair_lookup_;        // size nocc² (or empty)
    std::vector<int> setup_i_per_pair_;   // size N_pair (or empty)
    std::vector<int> setup_j_per_pair_;   // size N_pair (or empty); Phase 2 scatter
    int nocc_ = 0;

    // Multi-GPU slab info: this instance owns output rows for
    // i_ij ∈ [pair_begin_, pair_end_) on CUDA device device_id_.
    // For single-GPU operation pair_begin_=0, pair_end_=N_pair_, device_id_=0.
    int pair_begin_ = 0;
    int pair_end_   = 0;
    int device_id_  = 0;
};

} // namespace gansu
