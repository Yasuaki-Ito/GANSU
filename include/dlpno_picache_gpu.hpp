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
     *
     * Throws std::runtime_error on cudaMalloc / cuBLAS failure. In that
     * case the caller should fall back to the CPU Eigen kernel.
     */
    PiCacheGpu(const std::vector<std::vector<RowMatXd>>& barS_cache,
               const std::vector<int>& n_pno_per_pair,
               int max_n);

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
     * @brief Whether the GPU path is active. False = CPU fallback is in use.
     * Returns true when the constructor successfully built device buffers.
     */
    bool active() const noexcept { return active_; }

private:
    struct Impl;
    Impl* p_ = nullptr;
    bool active_ = false;

    // CPU fallback for !active() — keeps the public API stable when GPU
    // is unavailable. Same pi_cache_out layout.
    void rebuild_cpu_(const std::vector<std::vector<real_t>>& Y_old,
                      std::vector<std::vector<RowMatXd>>& pi_cache_out);

    // For the CPU fallback only: keep a const view into the input barS_cache.
    const std::vector<std::vector<RowMatXd>>* barS_cache_ref_ = nullptr;
    std::vector<int> n_pno_;
    int N_pair_ = 0;
    int max_n_ = 0;
};

} // namespace gansu
