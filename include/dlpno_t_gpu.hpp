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

#include "dlpno_tno.hpp"
#include "types.hpp"

namespace gansu {

/**
 * @brief Phase 3.2.6 GPU port — closed-shell DLPNO-(T) per-triple energy.
 *
 * Mirrors `compute_triple_t_energy_pyscf` (CPU) on the GPU. One instance is
 * created per OMP-thread (= per GPU); device buffers are allocated to the
 * maximum (max_n_tno, nocc) sizes once at construction and reused across
 * every triple processed by that thread.
 *
 * Per-triple kernels:
 *   1. Build 6 simple W tensors (cuBLAS DGEMM for the particle term + custom
 *      kernel for the hole rank-1 outer-product accumulation).
 *   2. Apply r3 on (a,b,c) for each W to form the 6 z tensors.
 *   3. 36-contraction kernel that sums W_σw under permutation π_{σz,σw}
 *      against z_σz and the d3-scaled inverse denominator, reducing to a
 *      single scalar.
 *   4. Multiply by 2 (closed-shell spin sum).
 *
 * If GPU is unavailable, `active()` returns false and the caller falls back
 * to the CPU path.
 */
class TripleTGpu {
public:
    /**
     * @param max_n_tno    upper bound on TNO size across all triples this
     *                     instance will process.
     * @param nocc         number of occupied LMOs.
     * @param max_batch    upper bound on the number of triples that may be
     *                     queued via add_to_batch() before a flush.
     */
    TripleTGpu(int max_n_tno, int nocc, int max_batch);
    ~TripleTGpu();

    TripleTGpu(const TripleTGpu&)            = delete;
    TripleTGpu& operator=(const TripleTGpu&) = delete;

    bool active() const { return active_; }

    /// Reset the batch state (call before queuing triples).
    void begin_batch();

    /// Append one triple's data to the batch staging buffers. Caller must
    /// ensure tno.n_tno ≤ max_n_tno passed at construction. Returns false if
    /// the batch is full (caller should flush_batch() and retry).
    bool add_to_batch(
        int i, int j, int k,
        real_t eps_i, real_t eps_j, real_t eps_k,
        const TNOData& tno,
        const real_t* K_iadc,
        const std::array<std::vector<real_t>, 9>& M,
        const std::array<std::vector<real_t>, 9>& T_part_oriented,
        const std::vector<std::vector<real_t>>& T_il_ext,
        const std::vector<std::vector<real_t>>& T_jl_ext,
        const std::vector<std::vector<real_t>>& T_kl_ext,
        int nocc);

    /// Compute the (T) energy contribution from all queued triples in one
    /// batched kernel launch. Returns the sum of contributions.
    real_t flush_batch();

    /// Single-triple legacy entry point (kept for backward compatibility and
    /// fallback testing). Internally just begin_batch() + add_to_batch() +
    /// flush_batch() with one element.
    real_t compute_triple(
        int i, int j, int k,
        real_t eps_i, real_t eps_j, real_t eps_k,
        const TNOData& tno,
        const real_t* K_iadc,
        const std::array<std::vector<real_t>, 9>& M,
        const std::array<std::vector<real_t>, 9>& T_part_oriented,
        const std::vector<std::vector<real_t>>& T_il_ext,
        const std::vector<std::vector<real_t>>& T_jl_ext,
        const std::vector<std::vector<real_t>>& T_kl_ext,
        int nocc);

private:
    bool active_     = false;
    int  max_n_      = 0;
    int  nocc_       = 0;
    int  max_batch_  = 0;
    int  batch_n_    = 0;        // number of triples currently queued
    int  batch_max_n_ = 0;       // max n_tno actually seen in current batch (≤ max_n_)

    // Per-triple slot byte-stride within the batched input buffer.
    size_t per_triple_words_ = 0;

    // Opaque pointers to avoid leaking CUDA types into the header.
    void* stream_ = nullptr;          // cudaStream_t
    void* cublas_ = nullptr;          // cublasHandle_t

    // Device buffer for the batched input. Layout per triple slot:
    //   [K (3·max_n³)] [M (9·nocc·max_n)] [T_part (9·max_n²)]
    //   [T_ext (3·nocc·max_n²)] [eps_tno (max_n)]
    // Total of `per_triple_words_` doubles per slot, `max_batch_` slots.
    real_t* d_input_     = nullptr;
    size_t  off_K_       = 0;
    size_t  off_M_       = 0;
    size_t  off_T_part_  = 0;
    size_t  off_T_ext_   = 0;
    size_t  off_eps_     = 0;

    real_t* d_W_         = nullptr;   // batch × 6 × max_n³
    real_t* d_R3W_       = nullptr;   // batch × 6 × max_n³
    real_t* d_D_inv_     = nullptr;   // batch × max_n³
    real_t* d_partial_   = nullptr;   // batch × 36 partial sums
    int*    d_n_tno_     = nullptr;   // batch
    int*    d_d3_factor_ = nullptr;   // batch
    real_t* d_eps_sum_   = nullptr;   // batch  (eps_i + eps_j + eps_k)

    // Pinned host staging buffer (cudaHostAlloc'd) — allows true async
    // host→device memcpy with the staging data outliving the kernel launch.
    real_t* h_pinned_input_   = nullptr;   // max_batch × per_triple_words
    real_t* h_pinned_partial_ = nullptr;   // max_batch × 36
    int*    h_pinned_n_tno_   = nullptr;   // max_batch
    int*    h_pinned_d3_      = nullptr;   // max_batch
    real_t* h_pinned_eps_sum_ = nullptr;   // max_batch
};

} // namespace gansu
