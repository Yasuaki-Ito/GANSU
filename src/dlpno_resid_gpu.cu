/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_resid_gpu.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_manager.hpp"
#endif

namespace gansu {

#ifndef GANSU_CPU_ONLY

namespace {

inline void check_cuda_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("ResidGpu CUDA error in ")
                                 + what + ": " + cudaGetErrorString(e));
    }
}

inline void check_cublas_(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ResidGpu cuBLAS error in ")
                                 + what + " status="
                                 + std::to_string(static_cast<int>(s)));
    }
}

} // anonymous namespace

// ===========================================================================
//  Kernels
// ===========================================================================

// slice_pi_N_T_for_I_kernel — extract per-pair (idx, l) sub-blocks of
// pi_T_stack for k=I_lmo and produce two padded buffers:
//   pi_N_pad[idx][a, l·max_n + d] = oriented_{I, l}(a, d)   (max_n × max_nn)
//   pi_T_pad[idx][l·max_n + d, a] = oriented_{I, l}(d, a)   (max_nn × max_n)
//
// pi_T_stack[idx](row a, col (I·nocc + l)·n_ij + d) is read.
// For pi_T_pad, we read the SAME pi_T_stack but at (row d, col (I·nocc + l)·n_ij + a)
// (i.e., transposed inner indices).
//
// Block: (max_n × max_n) — threadIdx.x = inner-col, threadIdx.y = inner-row.
// Grid:  (N_pair, nocc).
__global__ void slice_pi_N_T_for_I_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_per_pair_I,
    const int*    __restrict__ d_pair_lookup,
    real_t*       __restrict__ d_pi_N_pad,
    real_t*       __restrict__ d_pi_T_pad,
    int N_pair, int nocc, int max_n)
{
    const int idx = blockIdx.x;
    const int l   = blockIdx.y;
    if (idx >= N_pair || l >= nocc) return;

    const int n_ij = d_n_pno[idx];
    const int max_nn = nocc * max_n;
    const size_t pi_N_pair_stride = static_cast<size_t>(max_n)
                                  * static_cast<size_t>(max_nn);
    const size_t pi_T_pair_stride = static_cast<size_t>(max_nn)
                                  * static_cast<size_t>(max_n);

    // Step 6.6 fix: strided per-thread loop so block (TILE × TILE) covers
    // the full max_n × max_n grid even when max_n > sqrt(1024) = 32.
    for (int a = threadIdx.y; a < max_n; a += blockDim.y) {
        for (int d = threadIdx.x; d < max_n; d += blockDim.x) {
            real_t v_N = real_t(0);  // oriented(a, d)  → at (a, l*max_n + d)
            real_t v_T = real_t(0);  // oriented(d, a)  → at (l*max_n + d, a)

            if (n_ij > 0 && a < n_ij && d < n_ij) {
                const int I_lmo = d_per_pair_I[idx];
                const int idx_il = d_pair_lookup[I_lmo * nocc + l];
                const int n_il   = d_n_pno[idx_il];
                if (n_il > 0) {
                    const size_t base = d_idx_offset_pi_T[idx]
                        + (static_cast<size_t>(I_lmo) * nocc + l)
                        * static_cast<size_t>(n_ij);
                    const size_t row_stride = static_cast<size_t>(nocc) * nocc * n_ij;
                    v_N = d_pi_T_stack[base + static_cast<size_t>(a) * row_stride + d];
                    v_T = d_pi_T_stack[base + static_cast<size_t>(d) * row_stride + a];
                }
            }

            d_pi_N_pad[static_cast<size_t>(idx) * pi_N_pair_stride
                     + static_cast<size_t>(a) * max_nn
                     + static_cast<size_t>(l) * max_n + d] = v_N;
            d_pi_T_pad[static_cast<size_t>(idx) * pi_T_pair_stride
                     + (static_cast<size_t>(l) * max_n + d) * max_n + a] = v_T;
        }
    }
}

// Step 6.5 — Inter-pair Fock i-coupling kernel.
//
// For each pair idx with non-empty n_pno = n_ij and (a, d) ∈ [0, n_ij)²:
//   R[idx][a, d] -= Σ_{k != I_i} F_eff[I_i, k] · π_{k, J_j}^{oriented}[a, d]
// where F_eff[I_i, k] = F_LMO[I_i, k] + dF_ki[k, I_i].
//
// pi_T_stack[idx](a, (k·nocc + l)·n_ij + d) = π_{k, l}^{oriented}(a, d).
// We slice at l = J_j[idx] and reduce over k. The threshold filter in the
// CPU code is preserved (kFLMOThresh = 1e-14) but in practice this filters
// only exactly-zero entries; sparsity isn't really exploited.
//
// Block: (max_n × max_n). Grid: (N_pair). Per thread (a, d) does up to nocc
// FMAs against a strided slice of pi_T_stack.
__global__ void inter_pair_fock_i_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_I_i,
    const int*    __restrict__ d_I_j,
    const real_t* __restrict__ d_F_LMO,
    const real_t* __restrict__ d_dF_ki,
    real_t*       __restrict__ d_R_pad,
    int N_pair, int nocc, int max_n,
    real_t threshold)
{
    const int idx = blockIdx.x;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    const int I_i = d_I_i[idx];
    const int J_j = d_I_j[idx];
    const size_t k_stride = static_cast<size_t>(nocc) * n_ij;

    // Step 6.6 fix: strided per-thread loop so max_n > sqrt(1024) = 32 works.
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            const size_t base = d_idx_offset[idx]
                              + static_cast<size_t>(a) * nocc * nocc * n_ij
                              + static_cast<size_t>(J_j) * n_ij + d;
            real_t sum = real_t(0);
            for (int k = 0; k < nocc; ++k) {
                if (k == I_i) continue;
                const real_t F_LMO_ik  = d_F_LMO[I_i * nocc + k];
                const real_t dF_ki_val = d_dF_ki[k * nocc + I_i];
                const real_t F_ik = F_LMO_ik + dF_ki_val;
                if (F_ik > -threshold && F_ik < threshold) continue;

                const int idx_kj = d_pair_lookup[k * nocc + J_j];
                if (d_n_pno[idx_kj] == 0) continue;

                const real_t pi_val = d_pi_T_stack[base
                                                + static_cast<size_t>(k) * k_stride];
                sum -= F_ik * pi_val;
            }
            d_R_pad[static_cast<size_t>(idx) * max_n * max_n
                  + static_cast<size_t>(a) * max_n + d] += sum;
        }
    }
}

// Step 6.5 — Inter-pair Fock j-coupling kernel.
//
// For each pair idx with non-empty n_pno = n_ij and (a, d) ∈ [0, n_ij)²:
//   R[idx][a, d] -= Σ_{l != J_j} F_eff[l, J_j] · π_{I_i, l}^{oriented}[a, d]
// where F_eff[l, J_j] = F_LMO[l, J_j] + dF_ki[l, J_j].
//
// Slice at k = I_i[idx]; reduce over l. Stride per l is n_ij (much tighter
// than the i-kernel's nocc·n_ij stride), so reads are more cache-friendly.
__global__ void inter_pair_fock_j_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_I_i,
    const int*    __restrict__ d_I_j,
    const real_t* __restrict__ d_F_LMO,
    const real_t* __restrict__ d_dF_ki,
    real_t*       __restrict__ d_R_pad,
    int N_pair, int nocc, int max_n,
    real_t threshold)
{
    const int idx = blockIdx.x;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    const int I_i = d_I_i[idx];
    const int J_j = d_I_j[idx];
    const size_t l_stride = static_cast<size_t>(n_ij);

    // Step 6.6 fix: strided per-thread loop for max_n > sqrt(1024).
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            const size_t base = d_idx_offset[idx]
                              + static_cast<size_t>(a) * nocc * nocc * n_ij
                              + static_cast<size_t>(I_i) * nocc * n_ij + d;
            real_t sum = real_t(0);
            for (int l = 0; l < nocc; ++l) {
                if (l == J_j) continue;
                const real_t F_LMO_lj = d_F_LMO[l * nocc + J_j];
                const real_t dF_lj    = d_dF_ki[l * nocc + J_j];
                const real_t F_lj     = F_LMO_lj + dF_lj;
                if (F_lj > -threshold && F_lj < threshold) continue;

                const int idx_il = d_pair_lookup[I_i * nocc + l];
                if (d_n_pno[idx_il] == 0) continue;

                const real_t pi_val = d_pi_T_stack[base
                                                + static_cast<size_t>(l) * l_stride];
                sum -= F_lj * pi_val;
            }
            d_R_pad[static_cast<size_t>(idx) * max_n * max_n
                  + static_cast<size_t>(a) * max_n + d] += sum;
        }
    }
}

// Step 6.6 — fused oooo ladder kernel.
//
// Per pair idx (with non-empty n_pno = n_ij) and (a, b) ∈ [0, n_ij)²:
//   R[idx][a, b] += Σ_{kl} W_eff[idx][kl] · π_{kl}^{oriented}[a, b]
// where  W_eff[idx][kl] = W_oooo[idx][kl] + W_dress[idx][kl]
//        W_dress[idx][kl] = Σ_{d, c} V_lk(d, c) · Y_old(c, d)
//                         = Σ_{d, c} V_stacked[idx][kl, d·max_n + c] · Y_pad[idx][c·max_n + d]
//
// Each thread runs a doubly-nested loop: outer kl ∈ [0, nocc²), inner
// (d, c) ∈ [0, n_ij)². V_lk and Y_pad reads are warp-broadcast (all 144
// threads in a block share the same V/Y per inner iteration, so L1 cache
// handles the redundancy efficiently). pi_T_stack reads are coalesced
// across b (innermost threadIdx). R_pad writes are coalesced.
//
// (Step 6.6b experimented with splitting this into compute-W_eff +
//  apply-W_eff kernels — no measurable speedup since the cache-broadcast
//  pattern already amortised the W_dress redundancy in the fused form.)
__global__ void oooo_lad_kernel(
    const real_t* __restrict__ d_V_stacked_oooo_pad,
    const real_t* __restrict__ d_W_oooo,
    const real_t* __restrict__ d_pi_T_stack,
    const real_t* __restrict__ d_Y_pad,
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    real_t*       __restrict__ d_R_pad,
    int N_pair, int nocc, int max_n)
{
    const int idx = blockIdx.x;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    const int    nocc2     = nocc * nocc;
    const int    max_nn    = max_n * max_n;
    const size_t v_pair_off = static_cast<size_t>(idx) * nocc2 * max_nn;
    const size_t w_pair_off = static_cast<size_t>(idx) * nocc2;
    const size_t y_pair_off = static_cast<size_t>(idx) * max_nn;
    const size_t pi_kl_stride = static_cast<size_t>(n_ij);

    // Step 6.6 fix: strided per-thread loop for max_n > sqrt(1024).
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int b = threadIdx.x; b < n_ij; b += blockDim.x) {
            const size_t pi_pair_off = d_idx_offset_pi_T[idx]
                                     + static_cast<size_t>(a) * nocc * nocc * n_ij + b;
            real_t acc = real_t(0);
            for (int kl = 0; kl < nocc2; ++kl) {
                const real_t* V_lk = d_V_stacked_oooo_pad
                                   + v_pair_off
                                   + static_cast<size_t>(kl) * max_nn;

                // W_dress = Σ_{d, c < n_ij} V_lk(d, c) · Y_old(c, d)
                real_t W_dress = real_t(0);
                for (int dd = 0; dd < n_ij; ++dd) {
                    const real_t* V_row = V_lk + static_cast<size_t>(dd) * max_n;
                    const real_t* Y_col = d_Y_pad + y_pair_off + dd;
                    for (int cc = 0; cc < n_ij; ++cc) {
                        W_dress += V_row[cc] * Y_col[static_cast<size_t>(cc) * max_n];
                    }
                }

                const real_t W_eff = d_W_oooo[w_pair_off + kl] + W_dress;
                const real_t pi_val = d_pi_T_stack[pi_pair_off
                                                 + static_cast<size_t>(kl) * pi_kl_stride];
                acc += W_eff * pi_val;
            }
            d_R_pad[static_cast<size_t>(idx) * max_n * max_n
                  + static_cast<size_t>(a) * max_n + b] += acc;
        }
    }
}

// slice_PI_outer_for_J_kernel — extract per-pair (idx, k) blocks at
// l=J_lmo. Produces:
//   PI_stack_pad[idx][k·max_n + r, c] = oriented_{k, J}(r, c)  (max_nn × max_n)
//   PI_TT_pad   [idx][c, k·max_n + r] = oriented_{k, J}(c, r)  (max_n × max_nn)
__global__ void slice_PI_outer_for_J_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_per_pair_J,
    const int*    __restrict__ d_pair_lookup,
    real_t*       __restrict__ d_PI_stack_pad,
    real_t*       __restrict__ d_PI_TT_pad,
    int N_pair, int nocc, int max_n)
{
    const int idx = blockIdx.x;
    const int k   = blockIdx.y;
    if (idx >= N_pair || k >= nocc) return;

    const int n_ij = d_n_pno[idx];
    const int max_nn = nocc * max_n;
    const size_t stack_stride = static_cast<size_t>(max_nn) * max_n;
    const size_t tt_stride    = static_cast<size_t>(max_n) * max_nn;

    // Step 6.6 fix: strided per-thread loop for max_n > sqrt(1024).
    for (int r = threadIdx.y; r < max_n; r += blockDim.y) {
        for (int c = threadIdx.x; c < max_n; c += blockDim.x) {
            real_t v_stack = real_t(0);  // oriented(r, c)
            real_t v_TT    = real_t(0);  // oriented(c, r)

            if (n_ij > 0 && r < n_ij && c < n_ij) {
                const int J_lmo = d_per_pair_J[idx];
                const int idx_kJ = d_pair_lookup[k * nocc + J_lmo];
                const int n_kJ   = d_n_pno[idx_kJ];
                if (n_kJ > 0) {
                    const size_t base = d_idx_offset_pi_T[idx]
                        + (static_cast<size_t>(k) * nocc + J_lmo)
                        * static_cast<size_t>(n_ij);
                    const size_t row_stride = static_cast<size_t>(nocc) * nocc * n_ij;
                    v_stack = d_pi_T_stack[base + static_cast<size_t>(r) * row_stride + c];
                    v_TT    = d_pi_T_stack[base + static_cast<size_t>(c) * row_stride + r];
                }
            }

            d_PI_stack_pad[static_cast<size_t>(idx) * stack_stride
                         + (static_cast<size_t>(k) * max_n + r) * max_n + c] = v_stack;
            d_PI_TT_pad[static_cast<size_t>(idx) * tt_stride
                      + static_cast<size_t>(c) * max_nn
                      + static_cast<size_t>(k) * max_n + r] = v_TT;
        }
    }
}

// ===========================================================================
//  Impl
// ===========================================================================

struct ResidGpu::Impl {
    int N_pair = 0;
    int max_n  = 0;
    int max_nn = 0;
    int nocc   = 0;

    // Per-pair I_lmo / J_lmo (= sij.i / sij.j). Iter-invariant.
    int* d_I_i = nullptr;   // = sij.i per pair
    int* d_I_j = nullptr;   // = sij.j per pair

    // Iter-invariant integral buffers.
    real_t* d_V_meta_T_pad     = nullptr;  // (N_pair × max_nn × max_nn) row-major
    real_t* d_V_meta_TT_pad    = nullptr;
    real_t* d_T_meta_pad       = nullptr;
    real_t* d_W_bare_ovov_i_pad = nullptr; // (N_pair × max_n × max_nn) row-major
    real_t* d_W_bare_ovov_j_pad = nullptr;
    real_t* d_W_bare_ovvo_i_pad = nullptr;
    real_t* d_W_bare_ovvo_j_pad = nullptr;

    // Per-iter scratch (allocated once, reused).
    real_t* d_pi_N_i_pad = nullptr;        // (max_n × max_nn) per pair
    real_t* d_pi_N_j_pad = nullptr;
    real_t* d_pi_T_i_pad = nullptr;        // (max_nn × max_n) per pair
    real_t* d_pi_T_j_pad = nullptr;
    real_t* d_PI_kj_stack_pad = nullptr;   // (max_nn × max_n) per pair
    real_t* d_PI_ki_stack_pad = nullptr;
    real_t* d_PI_kj_TT_pad = nullptr;      // (max_n × max_nn) per pair
    real_t* d_PI_ki_TT_pad = nullptr;
    real_t* d_W_block_i_pad  = nullptr;    // (max_n × max_nn) per pair
    real_t* d_W_block_i2_pad = nullptr;
    real_t* d_W_block_j_pad  = nullptr;
    real_t* d_W_block_j2_pad = nullptr;
    real_t* d_R_ph_pad   = nullptr;        // (max_n × max_n) per pair
    real_t* h_R_ph_pad   = nullptr;        // pinned host mirror

    // Step 6.5 — inter-pair Fock i+j device buffers.
    real_t* d_F_LMO  = nullptr;            // (nocc × nocc) iter-invariant
    real_t* d_dF_ki  = nullptr;            // (nocc × nocc) refreshed each iter

    // Step 6.6 — oooo ladder iter-invariant buffers.
    // V_stacked_oooo_pad[idx][kl, d*max_n + c] = V_lk(d, c) for d, c < n_ij else 0
    // (kl varies in [0, nocc²), per-pair flat layout).
    real_t* d_V_stacked_oooo_pad = nullptr;   // (N × nocc² × max_n²)
    real_t* d_W_oooo             = nullptr;   // (N × nocc²)
    // Note: Step 6.6b experimented with a 2-kernel split (precomputed W_eff
    // table) to remove the 144× redundant W_dress computation in the fused
    // kernel below — it was no faster (cache broadcast already amortised
    // the redundancy on warp-shared V/Y reads), so we kept the simpler
    // fused kernel.

    // Borrowed cuBLAS handle.
    cublasHandle_t cublas = nullptr;

    // Step 6.4 — completion event used to gate compute_finalize on the
    // async D2H of R_ph_pad (recorded in compute_async, waited on in
    // compute_finalize). Lets the caller overlap CPU work like dF_ki /
    // DFpair with the rgpu pipeline.
    cudaEvent_t completion_event = nullptr;
    bool        async_in_flight  = false;

    void free_all() {
        if (completion_event) {
            cudaEventDestroy(completion_event);
            completion_event = nullptr;
        }
        auto free_d = [](real_t** p) {
            if (*p) { cudaFree(*p); *p = nullptr; }
        };
        auto free_di = [](int** p) {
            if (*p) { cudaFree(*p); *p = nullptr; }
        };
        free_di(&d_I_i);
        free_di(&d_I_j);
        free_d(&d_V_meta_T_pad);
        free_d(&d_V_meta_TT_pad);
        free_d(&d_T_meta_pad);
        free_d(&d_W_bare_ovov_i_pad);
        free_d(&d_W_bare_ovov_j_pad);
        free_d(&d_W_bare_ovvo_i_pad);
        free_d(&d_W_bare_ovvo_j_pad);
        free_d(&d_pi_N_i_pad);
        free_d(&d_pi_N_j_pad);
        free_d(&d_pi_T_i_pad);
        free_d(&d_pi_T_j_pad);
        free_d(&d_PI_kj_stack_pad);
        free_d(&d_PI_ki_stack_pad);
        free_d(&d_PI_kj_TT_pad);
        free_d(&d_PI_ki_TT_pad);
        free_d(&d_W_block_i_pad);
        free_d(&d_W_block_i2_pad);
        free_d(&d_W_block_j_pad);
        free_d(&d_W_block_j2_pad);
        free_d(&d_R_ph_pad);
        if (h_R_ph_pad) { cudaFreeHost(h_R_ph_pad); h_R_ph_pad = nullptr; }
        free_d(&d_F_LMO);
        free_d(&d_dF_ki);
        free_d(&d_V_stacked_oooo_pad);
        free_d(&d_W_oooo);
    }
};

#else // GANSU_CPU_ONLY

struct ResidGpu::Impl {};

#endif // GANSU_CPU_ONLY

// ===========================================================================
//  Constructor — allocate + upload iter-invariant data
// ===========================================================================
ResidGpu::ResidGpu(const PiCacheGpu&             pgpu,
                   const std::vector<PairSetup>& setups,
                   const std::vector<PairData>&  pairs,
                   const Phase24Integrals&       phase24,
                   const std::vector<real_t>&    F_LMO_host,
                   int nocc, int max_n)
    : pgpu_(&pgpu),
      N_pair_(static_cast<int>(pairs.size())),
      max_n_(max_n),
      nocc_(nocc)
{
    // Capture per-pair host metadata.
    n_pno_.assign(N_pair_, 0);
    setup_i_per_pair_.assign(N_pair_, 0);
    setup_j_per_pair_.assign(N_pair_, 0);
    for (int i = 0; i < N_pair_; ++i) {
        n_pno_[i]            = pairs[i].n_pno;
        setup_i_per_pair_[i] = setups[i].i;
        setup_j_per_pair_[i] = setups[i].j;
    }

#ifndef GANSU_CPU_ONLY
    // GPU activation gates.
    if (!pgpu.stacked() || !gpu::gpu_available()
        || N_pair_ == 0 || max_n_ <= 0 || nocc_ <= 0)
    {
        active_ = false;
        return;
    }
    if (phase24.nocc != nocc_
        || phase24.V_ovov_pair.size() != static_cast<size_t>(N_pair_)
        || phase24.T_pair.size()      != static_cast<size_t>(N_pair_)
        || phase24.W_ovov_i.size()    != static_cast<size_t>(N_pair_)
        || phase24.W_ovov_j.size()    != static_cast<size_t>(N_pair_)
        || phase24.W_ovvo_i.size()    != static_cast<size_t>(N_pair_)
        || phase24.W_ovvo_j.size()    != static_cast<size_t>(N_pair_))
    {
        active_ = false;
        return;
    }

    p_ = new Impl();
    Impl& s = *p_;
    s.N_pair = N_pair_;
    s.max_n  = max_n_;
    s.nocc   = nocc_;
    s.max_nn = nocc_ * max_n_;
    const int max_nn = s.max_nn;

    // Memory budget.
    const size_t per_pair_meta   = static_cast<size_t>(max_nn) * max_nn;          // V_meta_T etc
    const size_t per_pair_block  = static_cast<size_t>(max_n) * max_nn;           // W_block / W_bare / pi_N / PI_TT
    const size_t per_pair_stack  = static_cast<size_t>(max_nn) * max_n;           // pi_T / PI_stack
    const size_t per_pair_R      = static_cast<size_t>(max_n) * max_n;            // R_ph
    const size_t bytes_meta_full   = static_cast<size_t>(N_pair_) * per_pair_meta  * sizeof(real_t);
    const size_t bytes_block_full  = static_cast<size_t>(N_pair_) * per_pair_block * sizeof(real_t);
    const size_t bytes_stack_full  = static_cast<size_t>(N_pair_) * per_pair_stack * sizeof(real_t);
    const size_t bytes_R_full      = static_cast<size_t>(N_pair_) * per_pair_R     * sizeof(real_t);

    // Total estimate:
    //   3 × meta (V_meta_T/TT, T_meta) + 4 × block (W_bare_ovov/ovvo i/j)
    //   + per-iter scratch: 2 pi_N + 2 pi_T + 2 PI_stack + 2 PI_TT + 4 W_block + 1 R
    //                     = 4 stack + 6 block + 1 R
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        const size_t bytes_v_oooo_full = static_cast<size_t>(N_pair_)
                                       * static_cast<size_t>(nocc_) * nocc_
                                       * static_cast<size_t>(max_n) * max_n
                                       * sizeof(real_t);
        const size_t need = 3 * bytes_meta_full
                          + 4 * bytes_block_full
                          + 4 * bytes_stack_full       // pi_T_i/j + PI_xstack i/j
                          + 6 * bytes_block_full       // pi_N i/j + PI_xTT i/j + W_block i/i2/j/j2
                          + 1 * bytes_R_full
                          + 1 * bytes_R_full           // pinned host mirror
                          + bytes_v_oooo_full          // Step 6.6 V_stacked_oooo_pad
                          + (size_t)128 * 1024 * 1024;
        if (need > free_b) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
    }

    try {
        // Iter-invariant.
        check_cuda_(cudaMalloc(&s.d_V_meta_T_pad,   bytes_meta_full),  "alloc V_meta_T_pad");
        check_cuda_(cudaMalloc(&s.d_V_meta_TT_pad,  bytes_meta_full),  "alloc V_meta_TT_pad");
        check_cuda_(cudaMalloc(&s.d_T_meta_pad,     bytes_meta_full),  "alloc T_meta_pad");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovov_i_pad, bytes_block_full), "alloc W_bare_ovov_i");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovov_j_pad, bytes_block_full), "alloc W_bare_ovov_j");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovvo_i_pad, bytes_block_full), "alloc W_bare_ovvo_i");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovvo_j_pad, bytes_block_full), "alloc W_bare_ovvo_j");
        check_cuda_(cudaMalloc(&s.d_I_i, static_cast<size_t>(N_pair_) * sizeof(int)), "alloc d_I_i");
        check_cuda_(cudaMalloc(&s.d_I_j, static_cast<size_t>(N_pair_) * sizeof(int)), "alloc d_I_j");

        // Per-iter scratch.
        check_cuda_(cudaMalloc(&s.d_pi_N_i_pad, bytes_block_full), "alloc pi_N_i");
        check_cuda_(cudaMalloc(&s.d_pi_N_j_pad, bytes_block_full), "alloc pi_N_j");
        check_cuda_(cudaMalloc(&s.d_pi_T_i_pad, bytes_stack_full), "alloc pi_T_i");
        check_cuda_(cudaMalloc(&s.d_pi_T_j_pad, bytes_stack_full), "alloc pi_T_j");
        check_cuda_(cudaMalloc(&s.d_PI_kj_stack_pad, bytes_stack_full), "alloc PI_kj_stack");
        check_cuda_(cudaMalloc(&s.d_PI_ki_stack_pad, bytes_stack_full), "alloc PI_ki_stack");
        check_cuda_(cudaMalloc(&s.d_PI_kj_TT_pad,    bytes_block_full), "alloc PI_kj_TT");
        check_cuda_(cudaMalloc(&s.d_PI_ki_TT_pad,    bytes_block_full), "alloc PI_ki_TT");
        check_cuda_(cudaMalloc(&s.d_W_block_i_pad,   bytes_block_full), "alloc W_block_i");
        check_cuda_(cudaMalloc(&s.d_W_block_i2_pad,  bytes_block_full), "alloc W_block_i2");
        check_cuda_(cudaMalloc(&s.d_W_block_j_pad,   bytes_block_full), "alloc W_block_j");
        check_cuda_(cudaMalloc(&s.d_W_block_j2_pad,  bytes_block_full), "alloc W_block_j2");
        check_cuda_(cudaMalloc(&s.d_R_ph_pad,        bytes_R_full),     "alloc R_ph");
        check_cuda_(cudaMallocHost(&s.h_R_ph_pad,    bytes_R_full),     "alloc h_R_ph pinned");
        const size_t bytes_F = static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t);
        check_cuda_(cudaMalloc(&s.d_F_LMO,           bytes_F),          "alloc d_F_LMO");
        check_cuda_(cudaMalloc(&s.d_dF_ki,           bytes_F),          "alloc d_dF_ki");

        // Step 6.6 — oooo ladder iter-invariant buffers.
        const size_t per_pair_v_oooo = static_cast<size_t>(nocc_) * nocc_
                                     * static_cast<size_t>(max_n) * max_n;
        const size_t bytes_v_oooo = static_cast<size_t>(N_pair_) * per_pair_v_oooo
                                  * sizeof(real_t);
        const size_t bytes_w_oooo = static_cast<size_t>(N_pair_) * nocc_ * nocc_
                                  * sizeof(real_t);
        check_cuda_(cudaMalloc(&s.d_V_stacked_oooo_pad, bytes_v_oooo),
                    "alloc d_V_stacked_oooo_pad");
        check_cuda_(cudaMalloc(&s.d_W_oooo,             bytes_w_oooo),
                    "alloc d_W_oooo");
    } catch (const std::exception&) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    s.cublas = gpu::GPUHandle::cublas();
    if (!s.cublas) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    // Step 6.4: completion event (default flags — disable timing for lower
    // overhead; we only use it for stream-side synchronisation).
    if (cudaEventCreateWithFlags(&s.completion_event, cudaEventDisableTiming)
            != cudaSuccess) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    // ---- Upload F_LMO (Step 6.5, iter-invariant) and zero-init dF_ki ----
    if (static_cast<int>(F_LMO_host.size()) >= nocc_ * nocc_) {
        check_cuda_(cudaMemcpy(s.d_F_LMO, F_LMO_host.data(),
                               static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_F_LMO");
    }
    check_cuda_(cudaMemset(s.d_dF_ki, 0,
                           static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t)),
                "memset d_dF_ki");

    // ---- Upload per-pair I_i / I_j ----
    check_cuda_(cudaMemcpy(s.d_I_i, setup_i_per_pair_.data(),
                           static_cast<size_t>(N_pair_) * sizeof(int),
                           cudaMemcpyHostToDevice), "H2D d_I_i");
    check_cuda_(cudaMemcpy(s.d_I_j, setup_j_per_pair_.data(),
                           static_cast<size_t>(N_pair_) * sizeof(int),
                           cudaMemcpyHostToDevice), "H2D d_I_j");

    // ---- Pack + upload V_meta_T / V_meta_TT / T_meta ----
    // Source layouts (host): phase24->V_ovov_pair[idx][(l*nocc + k)*n*n + d*n + c] = V_lk(d, c)
    //                       phase24->T_pair[idx]    [(k*nocc + l)*n*n + c*n + d] = T_kl(c, d)
    // Padded layouts (device, row-major (max_nn × max_nn) per pair):
    //   V_meta_T_pad [l*max_n + d, k*max_n + c] = V_lk(d, c)
    //   V_meta_TT_pad[l*max_n + d, k*max_n + c] = V_lk(c, d)
    //   T_meta_pad   [l*max_n + d, k*max_n + c] = T_kl(c, d)
    {
        std::vector<real_t> h_V_T (static_cast<size_t>(N_pair_) * per_pair_meta, real_t{0});
        std::vector<real_t> h_V_TT(static_cast<size_t>(N_pair_) * per_pair_meta, real_t{0});
        std::vector<real_t> h_T   (static_cast<size_t>(N_pair_) * per_pair_meta, real_t{0});
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < N_pair_; ++idx) {
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (phase24.V_ovov_pair[idx].empty()) continue;
            if (phase24.T_pair[idx].empty())     continue;
            const real_t* V = phase24.V_ovov_pair[idx].data();
            const real_t* T = phase24.T_pair[idx].data();
            const size_t pair_off = static_cast<size_t>(idx) * per_pair_meta;
            for (int l = 0; l < nocc; ++l) {
                for (int k = 0; k < nocc; ++k) {
                    const real_t* V_lk = V + (static_cast<size_t>(l) * nocc + k)
                                         * static_cast<size_t>(n) * n;
                    const real_t* T_kl = T + (static_cast<size_t>(k) * nocc + l)
                                         * static_cast<size_t>(n) * n;
                    for (int d = 0; d < n; ++d) {
                        const size_t row_off = pair_off
                            + (static_cast<size_t>(l) * max_n + d)
                            * static_cast<size_t>(max_nn);
                        real_t* row_VT  = h_V_T.data()  + row_off;
                        real_t* row_VTT = h_V_TT.data() + row_off;
                        real_t* row_T   = h_T.data()    + row_off;
                        for (int c = 0; c < n; ++c) {
                            const size_t col_off = static_cast<size_t>(k) * max_n + c;
                            row_VT [col_off] = V_lk[d * n + c];
                            row_VTT[col_off] = V_lk[c * n + d];
                            row_T  [col_off] = T_kl[c * n + d];
                        }
                    }
                }
            }
        }
        check_cuda_(cudaMemcpy(s.d_V_meta_T_pad,  h_V_T.data(),  bytes_meta_full,
                               cudaMemcpyHostToDevice), "H2D V_meta_T");
        check_cuda_(cudaMemcpy(s.d_V_meta_TT_pad, h_V_TT.data(), bytes_meta_full,
                               cudaMemcpyHostToDevice), "H2D V_meta_TT");
        check_cuda_(cudaMemcpy(s.d_T_meta_pad,    h_T.data(),    bytes_meta_full,
                               cudaMemcpyHostToDevice), "H2D T_meta");
    }

    // ---- Pack + upload W_bare_ovov_i/j and W_bare_ovvo_i/j into block layout ----
    // Source (host): phase24->W_ovov_i[idx][(a*nocc + k)*n + c]  = W(a, k, c) for a, c < n
    // Padded (device, row-major (max_n × max_nn)):
    //   W_pad[a, k*max_n + c] = W(a, k, c) for a, c < n else 0
    auto pack_W_bare = [&](const std::vector<std::vector<real_t>>& src,
                           real_t* d_dst,
                           const char* label)
    {
        std::vector<real_t> h_W(static_cast<size_t>(N_pair_) * per_pair_block, real_t{0});
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < N_pair_; ++idx) {
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (idx >= static_cast<long long>(src.size())) continue;
            if (src[idx].empty()) continue;
            const real_t* W = src[idx].data();
            const size_t pair_off = static_cast<size_t>(idx) * per_pair_block;
            for (int a = 0; a < n; ++a) {
                for (int k = 0; k < nocc; ++k) {
                    const real_t* row_in = W
                        + (static_cast<size_t>(a) * nocc + k) * n;
                    real_t* row_out = h_W.data() + pair_off
                                    + static_cast<size_t>(a) * max_nn
                                    + static_cast<size_t>(k) * max_n;
                    for (int c = 0; c < n; ++c) {
                        row_out[c] = row_in[c];
                    }
                }
            }
        }
        check_cuda_(cudaMemcpy(d_dst, h_W.data(), bytes_block_full,
                               cudaMemcpyHostToDevice), label);
    };
    pack_W_bare(phase24.W_ovov_i, s.d_W_bare_ovov_i_pad, "H2D W_bare_ovov_i");
    pack_W_bare(phase24.W_ovov_j, s.d_W_bare_ovov_j_pad, "H2D W_bare_ovov_j");
    pack_W_bare(phase24.W_ovvo_i, s.d_W_bare_ovvo_i_pad, "H2D W_bare_ovvo_i");
    pack_W_bare(phase24.W_ovvo_j, s.d_W_bare_ovvo_j_pad, "H2D W_bare_ovvo_j");

    // ---- Step 6.6: Pack + upload V_stacked_oooo (padded) and W_oooo. ----
    // Source: phase24.V_ovov_pair[idx]: (l*nocc + k, d, c) flat → V_lk(d, c).
    // Padded layout: V_stacked_oooo_pad[idx][kl, d*max_n + c] = V_lk(d, c)
    //                where kl = k*nocc + l (note CPU code uses (k*nocc + l)
    //                indexing on the row axis of V_stacked_oooo).
    {
        const size_t per_pair_v_oooo = static_cast<size_t>(nocc_) * nocc_
                                     * static_cast<size_t>(max_n) * max_n;
        const size_t bytes_v_oooo = static_cast<size_t>(N_pair_) * per_pair_v_oooo
                                  * sizeof(real_t);
        std::vector<real_t> h_V_oooo(static_cast<size_t>(N_pair_) * per_pair_v_oooo,
                                     real_t{0});
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < N_pair_; ++idx) {
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (idx >= static_cast<long long>(phase24.V_ovov_pair.size())) continue;
            if (phase24.V_ovov_pair[idx].empty()) continue;
            const real_t* V = phase24.V_ovov_pair[idx].data();
            const size_t pair_off = static_cast<size_t>(idx) * per_pair_v_oooo;
            for (int k = 0; k < nocc_; ++k) {
                for (int l = 0; l < nocc_; ++l) {
                    // CPU layout: V_stacked_oooo[idx] row index = (k*nocc + l)
                    //             with V_lk row-major flat in the columns.
                    const real_t* V_lk = V + (static_cast<size_t>(l) * nocc_ + k)
                                         * static_cast<size_t>(n) * n;
                    real_t* row_dst = h_V_oooo.data() + pair_off
                                    + (static_cast<size_t>(k) * nocc_ + l)
                                      * static_cast<size_t>(max_n) * max_n;
                    for (int d = 0; d < n; ++d) {
                        for (int c = 0; c < n; ++c) {
                            row_dst[static_cast<size_t>(d) * max_n + c]
                                = V_lk[d * n + c];
                        }
                    }
                }
            }
        }
        check_cuda_(cudaMemcpy(s.d_V_stacked_oooo_pad, h_V_oooo.data(),
                               bytes_v_oooo, cudaMemcpyHostToDevice),
                    "H2D V_stacked_oooo_pad");
    }
    // W_oooo[idx] is already (nocc²) flat per pair → direct upload (concat).
    {
        const size_t per_pair_w = static_cast<size_t>(nocc_) * nocc_;
        const size_t bytes_w = static_cast<size_t>(N_pair_) * per_pair_w * sizeof(real_t);
        std::vector<real_t> h_W(static_cast<size_t>(N_pair_) * per_pair_w, real_t{0});
        for (int idx = 0; idx < N_pair_; ++idx) {
            if (idx >= static_cast<int>(phase24.W_oooo.size())) continue;
            if (phase24.W_oooo[idx].size() == per_pair_w) {
                std::memcpy(h_W.data() + static_cast<size_t>(idx) * per_pair_w,
                            phase24.W_oooo[idx].data(),
                            per_pair_w * sizeof(real_t));
            }
        }
        check_cuda_(cudaMemcpy(s.d_W_oooo, h_W.data(),
                               bytes_w, cudaMemcpyHostToDevice),
                    "H2D W_oooo");
    }

    active_ = true;
#else
    (void)pgpu;
    (void)setups;
    (void)pairs;
    (void)phase24;
    active_ = false;
#endif // !GANSU_CPU_ONLY
}

// ===========================================================================
//  Destructor
// ===========================================================================
ResidGpu::~ResidGpu() {
#ifndef GANSU_CPU_ONLY
    if (p_) {
        p_->free_all();
        delete p_;
        p_ = nullptr;
    }
#endif
}

// ===========================================================================
//  compute — per-iter ph-ladder R contributions
// ===========================================================================
void ResidGpu::compute(std::vector<RowMatXd>& R_ph_out)
{
    // Step 6.4: existing synchronous API kept for callers that don't need
    // CPU/GPU overlap. Internally just back-to-back async + finalize.
    compute_async();
    compute_finalize(R_ph_out);
}

void ResidGpu::compute_async()
{
    compute_async_phladder_only_();
    compute_async_finalize_pipeline_();
}

void ResidGpu::compute_async_phladder_only_()
{
    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    Impl& s = *p_;
    const int N      = N_pair_;
    const int nocc   = nocc_;
    const int max_n  = max_n_;
    const int max_nn = s.max_nn;

    // Sanity: pgpu_ must still be in stacked mode with current pi_T_stack.
    const real_t* d_pi_T_stack    = pgpu_->device_pi_T_stack();
    const size_t* d_idx_offset    = pgpu_->device_idx_offset_pi_T();
    const int*    d_pair_lookup   = pgpu_->device_pair_lookup();
    const int*    d_n_pno         = pgpu_->device_n_pno();
    if (!d_pi_T_stack || !d_idx_offset || !d_pair_lookup || !d_n_pno) return;

    // ---- Stage 1: slice pi_T_stack into per-pair pad blocks. ----
    // Two kernel launches per side: one for I (= sij.i for i-side, sij.j for j-side)
    // and one for the outer J (= sij.j for i-side, sij.i for j-side).
    //
    // Step 6.6 fix: cap block at TILE=16 (256 threads) to keep within
    // CUDA's 1024 threads/block limit when max_n > 32 (e.g. Benzene cc-pVDZ).
    // Kernels iterate strided over the (a, d) range internally.
    {
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(N),
                  static_cast<unsigned>(nocc), 1);

        // i-side: I = sij.i, J = sij.j
        slice_pi_N_T_for_I_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, s.d_I_i, d_pair_lookup,
            s.d_pi_N_i_pad, s.d_pi_T_i_pad,
            N, nocc, max_n);
        slice_PI_outer_for_J_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, s.d_I_j, d_pair_lookup,
            s.d_PI_kj_stack_pad, s.d_PI_kj_TT_pad,
            N, nocc, max_n);

        // j-side: I = sij.j, J = sij.i
        slice_pi_N_T_for_I_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, s.d_I_j, d_pair_lookup,
            s.d_pi_N_j_pad, s.d_pi_T_j_pad,
            N, nocc, max_n);
        slice_PI_outer_for_J_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, s.d_I_i, d_pair_lookup,
            s.d_PI_ki_stack_pad, s.d_PI_ki_TT_pad,
            N, nocc, max_n);

        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("ResidGpu slice kernels failed: ")
                                     + cudaGetErrorString(e));
        }
    }

    const real_t neg_half = -0.5;
    const real_t plus_half =  0.5;
    const real_t one      =  1.0;
    const real_t neg_one  = -1.0;
    const real_t two      =  2.0;
    const real_t zero     =  0.0;

    const long long stride_meta  = static_cast<long long>(max_nn) * max_nn;
    const long long stride_block = static_cast<long long>(max_n) * max_nn;
    const long long stride_stack = static_cast<long long>(max_nn) * max_n;
    const long long stride_R     = static_cast<long long>(max_n)  * max_n;
    const size_t bytes_block_full = static_cast<size_t>(N) * stride_block * sizeof(real_t);

    // ---- Stage 2: build W_block_i, W_block_i2, W_block_j, W_block_j2 ----
    //
    // Each W_block starts as a copy of the corresponding bare W (already in
    // the (a, k*max_n + c) layout). Two cuBLAS DGEMM accumulations follow.
    //
    // Math (in row-major):
    //   W_block_i  = -0.5 · pi_T_i^T · V_meta_T  + 0.5 · pi_N_i · T_meta + W_bare_ovov_i
    //   W_block_i2 = -0.5 · pi_T_i^T · V_meta_TT + W_bare_ovvo_i
    //   (j-side mirrors with pi_T_j / pi_N_j and W_bare_ov{ov,vo}_j)
    //
    // The cuBLAS view of a row-major (R × C) matrix is column-major (C × R)
    // with leading dim R. For result_row = X_row · Y_row (in row-major):
    //   result_col = Y_col · X_col   (cuBLAS direct call).
    //
    // For W_block_i term -0.5 · pi_T_i^T · V_meta_T (n × nn):
    //   result_col(max_nn × max_n) = V_meta_T_col · pi_T_i_col^T
    //   → m = max_nn, n = max_n, k = max_nn, TransA=N, TransB=T
    //   A = V_meta_T (lda = max_nn), B = pi_T_i (ldb = max_n)
    //   C = W_block_i (ldc = max_nn)

    auto run_W_block_build = [&](real_t* d_W_block,
                                 real_t* d_W_bare,
                                 real_t* d_pi_N,
                                 real_t* d_pi_T,
                                 real_t* d_V_buf,   // V_meta_T (W_block_i/j) or V_meta_TT (W_block_i2/j2)
                                 bool   add_T_term)  // include +0.5·pi_N·T_meta?
    {
        // Init from bare W_block (already in (a, k·max_n + c) padded layout).
        check_cuda_(cudaMemcpyAsync(d_W_block, d_W_bare, bytes_block_full,
                                    cudaMemcpyDeviceToDevice, /*stream=*/0),
                    "D2D W_bare → W_block");

        // W_block += -0.5 · pi_T^T · V (one batched DGEMM, β=1).
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            /*m=*/ max_nn, /*n=*/ max_n, /*k=*/ max_nn,
            &neg_half,
            d_V_buf, max_nn, stride_meta,
            d_pi_T,  max_n,  stride_stack,
            &one,
            d_W_block, max_nn, stride_block,
            N), "W_block += -0.5·pi_T^T·V");

        if (add_T_term) {
            // W_block += +0.5 · pi_N · T_meta.
            check_cublas_(cublasDgemmStridedBatched(
                s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                /*m=*/ max_nn, /*n=*/ max_n, /*k=*/ max_nn,
                &plus_half,
                s.d_T_meta_pad, max_nn, stride_meta,
                d_pi_N, max_nn, stride_block,
                &one,
                d_W_block, max_nn, stride_block,
                N), "W_block += +0.5·pi_N·T_meta");
        }
    };

    // i-side: W_block_i uses V_meta_T  (and adds T term)
    //         W_block_i2 uses V_meta_TT (no T term)
    run_W_block_build(s.d_W_block_i_pad,  s.d_W_bare_ovov_i_pad,
                      s.d_pi_N_i_pad,     s.d_pi_T_i_pad,
                      s.d_V_meta_T_pad,   /*add_T_term=*/true);
    run_W_block_build(s.d_W_block_i2_pad, s.d_W_bare_ovvo_i_pad,
                      s.d_pi_N_i_pad,     s.d_pi_T_i_pad,
                      s.d_V_meta_TT_pad,  /*add_T_term=*/false);
    // j-side mirror.
    run_W_block_build(s.d_W_block_j_pad,  s.d_W_bare_ovov_j_pad,
                      s.d_pi_N_j_pad,     s.d_pi_T_j_pad,
                      s.d_V_meta_T_pad,   /*add_T_term=*/true);
    run_W_block_build(s.d_W_block_j2_pad, s.d_W_bare_ovvo_j_pad,
                      s.d_pi_N_j_pad,     s.d_pi_T_j_pad,
                      s.d_V_meta_TT_pad,  /*add_T_term=*/false);

    // ---- Stage 3: contract into R_ph_pad ----
    // R_ph[idx] (row-major n × n) starts at zero.
    check_cuda_(cudaMemsetAsync(s.d_R_ph_pad, 0,
                                static_cast<size_t>(N) * stride_R * sizeof(real_t),
                                /*stream=*/0),
                "memset R_ph");

    // Helper: result_row[a, b] += α · A_row · B_row, where A_row is (m_row × k_row)
    //         and B_row is (k_row × n_row). In col-major: C_col += α · B_col · A_col.
    //         We always pass with TransA = TransB = N here unless the math has X^T.
    //         For X^T patterns, set Trans flag and adjust dim/ld accordingly.
    //
    // Op 1: R_ph += 2 · W_block_i · PI_kj_stack       (n × nn × n)
    //   row-major:  C_row = α · A_row · B_row  with A=W_block_i(max_n×max_nn),
    //               B=PI_kj_stack(max_nn×max_n). Result (max_n×max_n).
    //   col-major:  C_col(max_n × max_n) = PI_kj_stack_col(max_n × max_nn)
    //                                    · W_block_i_col(max_nn × max_n)
    //   m=max_n, n=max_n, k=max_nn, TransA=N, TransB=N
    //   A=PI_kj_stack lda=max_n, B=W_block_i ldb=max_nn, C=R_ph ldc=max_n.
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &two,
        s.d_PI_kj_stack_pad, max_n, stride_stack,
        s.d_W_block_i_pad,   max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R += 2·W_i·PI_kj");

    // Op 2: R_ph -= W_block_i2 · PI_kj_stack
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_PI_kj_stack_pad, max_n, stride_stack,
        s.d_W_block_i2_pad,  max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= W_i2·PI_kj");

    // Op 3: R_ph -= W_block_i · PI_kj_TT^T
    //   row-major: C = W · K^T  (W=W_block_i, K=PI_kj_TT)
    //   col-major: C_col = K_col^T · W_col
    //   m=max_n, n=max_n, k=max_nn, TransA=T, TransB=N
    //   A=PI_kj_TT lda=max_nn, B=W_block_i ldb=max_nn, C=R_ph ldc=max_n
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_PI_kj_TT_pad,    max_nn, stride_block,
        s.d_W_block_i_pad,   max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= W_i·PI_kj_TT^T");

    // Op 4: R_ph -= PI_kj_TT · W_block_i2^T
    //   row-major: C = K · W^T   (K=PI_kj_TT, W=W_block_i2)
    //   col-major: C_col = W_col^T · K_col
    //   m=max_n, n=max_n, k=max_nn, TransA=T, TransB=N
    //   A=W_block_i2 lda=max_nn, B=PI_kj_TT ldb=max_nn (it's (max_n × max_nn) row-major
    //     → col-major (max_nn × max_n) ld=max_nn). With TransB=N, op(B)=B_col=(max_nn × max_n) ✓
    //   Wait: C_col = W_col^T · K_col. W_col is (max_nn × max_n), so W_col^T is (max_n × max_nn).
    //         For C_col(m×n) = op(A)·op(B): m=max_n, k=max_nn, n=max_n.
    //         op(A) = W_col^T = (max_n × max_nn) ⇒ A=W_block_i2_col with TransA=T, lda=max_nn.
    //         op(B) = K_col   = (max_nn × max_n) ⇒ B=PI_kj_TT_col with TransB=N, ldb=max_nn.
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_W_block_i2_pad, max_nn, stride_block,
        s.d_PI_kj_TT_pad,   max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= PI_kj_TT·W_i2^T");

    // Op 5 (j-side): R_ph += 2 · PI_ki_stack^T · W_block_j^T
    //   row-major: C = PI^T · W^T   = (W · PI)^T. Result is (n × n) so transpose-equality.
    //   col-major: C_col = (PI^T · W^T)_col = ((W·PI)^T)_col = (W·PI)_row = W_row · PI_row
    //                    = W_col^T · PI_col^T
    //   m=max_n, n=max_n, k=max_nn, TransA=T, TransB=T
    //   A=W_block_j_col (max_nn × max_n) ld=max_nn → with TransA=T op(A)=(max_n × max_nn).
    //   B=PI_ki_stack_col (max_n × max_nn) ld=max_n → with TransB=T op(B)=(max_nn × max_n).
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_T, CUBLAS_OP_T,
        max_n, max_n, max_nn,
        &two,
        s.d_W_block_j_pad,   max_nn, stride_block,
        s.d_PI_ki_stack_pad, max_n,  stride_stack,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R += 2·PI_ki^T·W_j^T");

    // Op 6: R_ph -= PI_ki_stack^T · W_block_j2^T
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_T, CUBLAS_OP_T,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_W_block_j2_pad,  max_nn, stride_block,
        s.d_PI_ki_stack_pad, max_n,  stride_stack,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= PI_ki^T·W_j2^T");

    // Op 7: R_ph -= PI_ki_TT · W_block_j^T
    //   row-major: C = K · W^T  (K=PI_ki_TT (n×nn), W=W_block_j (n×nn) → W^T (nn×n))
    //   col-major: C_col = W_col^T · K_col
    //   m=max_n, n=max_n, k=max_nn, TransA=T, TransB=N
    //   A=W_block_j_col (max_nn × max_n), TransA=T → (max_n × max_nn), lda=max_nn.
    //   B=PI_ki_TT_col (max_nn × max_n), TransB=N → (max_nn × max_n), ldb=max_nn.
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_W_block_j_pad, max_nn, stride_block,
        s.d_PI_ki_TT_pad,  max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= PI_ki_TT·W_j^T");

    // Op 8: R_ph -= W_block_j2 · PI_ki_stack
    //   row-major: C = W · PI  (W=W_block_j2 (n×nn), PI=PI_ki_stack (nn×n))
    //   col-major: C_col = PI_col · W_col
    //   m=max_n, n=max_n, k=max_nn, TransA=N, TransB=N
    //   A=PI_ki_stack_col (max_n × max_nn), lda=max_n.
    //   B=W_block_j2_col (max_nn × max_n), ldb=max_nn.
    check_cublas_(cublasDgemmStridedBatched(
        s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        max_n, max_n, max_nn,
        &neg_one,
        s.d_PI_ki_stack_pad, max_n,  stride_stack,
        s.d_W_block_j2_pad,  max_nn, stride_block,
        &one,
        s.d_R_ph_pad, max_n, stride_R,
        N), "R -= W_j2·PI_ki");

#endif // !GANSU_CPU_ONLY
}

void ResidGpu::compute_async_finalize_pipeline_()
{
    if (!active_) return;
#ifndef GANSU_CPU_ONLY
    Impl& s = *p_;
    const int N     = N_pair_;
    const int max_n = max_n_;
    const long long stride_R = static_cast<long long>(max_n) * max_n;
    const size_t bytes_R = static_cast<size_t>(N) * stride_R * sizeof(real_t);
    check_cuda_(cudaMemcpyAsync(s.h_R_ph_pad, s.d_R_ph_pad,
                                bytes_R, cudaMemcpyDeviceToHost,
                                /*stream=*/0),
                "async D2H R_ph_pad");
    check_cuda_(cudaEventRecord(s.completion_event, /*stream=*/0),
                "record completion event");
    s.async_in_flight = true;
#endif
}

void ResidGpu::compute_async(const std::vector<real_t>& dF_ki_host)
{
    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    Impl& s = *p_;
    const int N     = N_pair_;
    const int nocc  = nocc_;
    const int max_n = max_n_;

    // 1. Upload dF_ki (async on default stream) for this iter.
    if (static_cast<int>(dF_ki_host.size()) >= nocc * nocc) {
        check_cuda_(cudaMemcpyAsync(s.d_dF_ki, dF_ki_host.data(),
                                    static_cast<size_t>(nocc) * nocc * sizeof(real_t),
                                    cudaMemcpyHostToDevice, /*stream=*/0),
                    "H2D dF_ki");
    }

    // 2. Run the ph-ladder pipeline (slice + W_block + 8 contractions,
    //    accumulating into d_R_ph_pad), but NOT the trailing D2H yet.
    compute_async_phladder_only_();

    // 3. Inter-pair Fock i+j kernels accumulate into the SAME d_R_ph_pad
    //    in row-major (max_n × max_n) layout per pair. They queue on the
    //    default stream, so they implicitly wait for ph-ladder kernels to
    //    finish before launching.
    const real_t threshold = 1e-14;  // matches kFLMOThresh in dlpno_pair_data.cu
    {
        // Step 6.6 fix: cap block at TILE=16 (see Stage 1 comment).
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(N), 1, 1);

        const real_t* d_pi_T_stack    = pgpu_->device_pi_T_stack();
        const size_t* d_idx_offset    = pgpu_->device_idx_offset_pi_T();
        const int*    d_pair_lookup   = pgpu_->device_pair_lookup();
        const int*    d_n_pno         = pgpu_->device_n_pno();

        inter_pair_fock_i_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, d_pair_lookup,
            s.d_I_i, s.d_I_j, s.d_F_LMO, s.d_dF_ki,
            s.d_R_ph_pad, N, nocc, max_n, threshold);
        inter_pair_fock_j_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_offset, d_n_pno, d_pair_lookup,
            s.d_I_i, s.d_I_j, s.d_F_LMO, s.d_dF_ki,
            s.d_R_ph_pad, N, nocc, max_n, threshold);

        // Step 6.6: fused oooo ladder kernel — borrows d_Y_pad from pgpu
        // (set by the most recent rebuild_with_stack call). Accumulates
        // into d_R_ph_pad alongside ph-ladder + inter-pair Fock.
        const real_t* d_Y_pad = pgpu_->device_Y_pad();
        if (d_Y_pad) {
            oooo_lad_kernel<<<grid, block>>>(
                s.d_V_stacked_oooo_pad, s.d_W_oooo,
                d_pi_T_stack, d_Y_pad, d_idx_offset, d_n_pno,
                s.d_R_ph_pad, N, nocc, max_n);
        }

        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("ResidGpu inter-pair Fock + oooo kernels failed: ")
                + cudaGetErrorString(e));
        }
    }

    // 4. Final async D2H + event record after ph-ladder + Fock + oooo
    //    contributions have been accumulated.
    compute_async_finalize_pipeline_();
#endif // !GANSU_CPU_ONLY
}

void ResidGpu::compute_finalize(std::vector<RowMatXd>& R_ph_out)
{
    R_ph_out.assign(static_cast<size_t>(N_pair_), RowMatXd());

    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    Impl& s = *p_;
    const int N      = N_pair_;
    const int max_n  = max_n_;
    const long long stride_R = static_cast<long long>(max_n) * max_n;

    if (!s.async_in_flight) {
        // Defensive: caller invoked finalize without a matching compute_async.
        // Run the synchronous compute path to keep the contract intact.
        compute_async();
    }

    // ---- Wait on the async D2H to complete. ----
    check_cuda_(cudaEventSynchronize(s.completion_event),
                "wait completion event");
    s.async_in_flight = false;

    // ---- Unpad to host vec<RowMatXd>. ----
    #pragma omp parallel for schedule(static)
    for (long long idx = 0; idx < N; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) {
            R_ph_out[idx].resize(0, 0);
            continue;
        }
        R_ph_out[idx].resize(n, n);
        const real_t* src = s.h_R_ph_pad
                          + static_cast<size_t>(idx) * stride_R;
        real_t* dst = R_ph_out[idx].data();
        for (int r = 0; r < n; ++r) {
            std::memcpy(dst + static_cast<size_t>(r) * n,
                        src + static_cast<size_t>(r) * max_n,
                        static_cast<size_t>(n) * sizeof(real_t));
        }
    }
#endif // !GANSU_CPU_ONLY
}

} // namespace gansu
