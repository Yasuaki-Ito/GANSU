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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
#include "multi_gpu_manager.hpp"  // MultiGpuManager, DeviceGuard
#endif

namespace gansu {

#ifndef GANSU_CPU_ONLY

namespace {

// Step 6.8c: RAII pinned host buffer for H2D staging. Replaces
// std::vector<real_t>(N, 0) at ResidGpu construction time. Benefits:
//   - cudaMallocHost on Linux uses pinned (page-locked) pages directly,
//     avoiding the cold mmap page-fault that hits regular std::vector
//     when first-touch happens on its 642 MB of zeros.
//   - cudaMemcpy from pinned memory uses driver DMA without internal
//     staging-buffer copies, ~3-4× faster than from pageable host memory.
// Used as: PinnedHost h_V_T(N * per_pair_meta); h_V_T.zero(); ...; pack;
//          cudaMemcpy(d_dst, h_V_T.get(), bytes, ...);  ~~RAII frees on scope exit.
class PinnedHost {
public:
    PinnedHost() = default;
    // pinned=true → cudaMallocHost (page-locked, fast repeated DMA but the
    // allocation itself takes the kernel mmap semaphore). pinned=false →
    // plain malloc (pageable): the H2D is a touch slower but there is NO
    // mmap-lock contention. For the ONE-TIME ResidGpu ctor staging buffers,
    // 7 pinned allocs × N_gpu construct concurrently and serialise on that
    // lock (Pentacene 8×A100: ~8-11 s, ~independent of data size). Those use
    // pinned=false; per-iter buffers (repeated DMA) keep pinned=true.
    explicit PinnedHost(size_t n_doubles, bool pinned = true) : pinned_(pinned) {
        if (n_doubles > 0) {
            if (pinned_) {
                const cudaError_t e = cudaMallocHost(&p_, n_doubles * sizeof(real_t));
                if (e != cudaSuccess) p_ = nullptr;
            } else {
                p_ = static_cast<real_t*>(std::malloc(n_doubles * sizeof(real_t)));
            }
        }
        n_ = (p_ ? n_doubles : 0);
    }
    PinnedHost(const PinnedHost&) = delete;
    PinnedHost& operator=(const PinnedHost&) = delete;
    PinnedHost(PinnedHost&& other) noexcept
        : p_(other.p_), n_(other.n_), pinned_(other.pinned_) {
        other.p_ = nullptr; other.n_ = 0;
    }
    PinnedHost& operator=(PinnedHost&& other) noexcept {
        if (this != &other) { free_(); p_ = other.p_; n_ = other.n_;
                              pinned_ = other.pinned_;
                              other.p_ = nullptr; other.n_ = 0; }
        return *this;
    }
    ~PinnedHost() { free_(); }
    real_t*       data()       noexcept { return p_; }
    const real_t* data() const noexcept { return p_; }
    size_t        size() const noexcept { return n_; }
    bool          valid() const noexcept { return p_ != nullptr; }
    void zero() noexcept {
        if (p_) std::memset(p_, 0, n_ * sizeof(real_t));
    }
private:
    void free_() noexcept {
        if (p_) {
            if (pinned_) cudaFreeHost(p_);
            else         std::free(p_);
            p_ = nullptr; n_ = 0;
        }
    }
    real_t* p_ = nullptr;
    size_t  n_ = 0;
    bool    pinned_ = true;
};

// Step 6.9 — GPU pack kernel for V_meta_T / V_meta_TT / T_meta / V_stacked_oooo.
//
// S11 Phase 2d (2026-05-19) — padded sinks removed. Writes go exclusively
// to the per-pair packed buffers built during S10b / S11 Phase 1:
//   V_meta_T_packed   [l·n + d, k·n + c] = V_lk(d, c)         (active pairs)
//   V_meta_TT_packed  [l·n + d, k·n + c] = V_lk(c, d)         (active pairs)
//   T_meta_packed     [l·n + d, k·n + c] = T_kl(c, d)         (active pairs)
//   V_stacked_oooo_pad[(k·nocc + l)·n² + d·n + c] = V_lk(d, c) (active pairs)
// Inactive pairs (d_active_pos[idx] < 0) own no slot and skip all writes.
//
// Source layout (d_V_flat / d_T_flat, per pair contiguous):
//   V_ovov_pair[idx][((l·nocc + k)·n + d)·n + c] = V_lk(d, c)
//   T_pair    [idx][((k·nocc + l)·n + c)·n + d] = T_kl(c, d)
// Pair offsets in d_pair_src_off (cumulative size_t array of length
// N_pair + 1, computed host-side).
__global__ void pack_V_meta_kernel(
    const real_t* __restrict__ d_V_flat,
    const real_t* __restrict__ d_T_flat,
    const int*    __restrict__ d_n_pno,
    const size_t* __restrict__ d_pair_src_off,
    const int*    __restrict__ d_active_pos,        // S10b: orig→a (or -1)
    const size_t* __restrict__ d_v_oooo_off_packed, // S10b: prefix sum
    const size_t* __restrict__ d_per_pair_meta_off, // S11 Phase 1: orig→packed offset (elem)
    real_t*       __restrict__ d_V_meta_T_packed,
    real_t*       __restrict__ d_V_meta_TT_packed,
    real_t*       __restrict__ d_T_meta_packed,
    real_t*       __restrict__ d_V_stacked_oooo_pad,
    int nocc,
    // Stage D (D3a): when voooo_sparse, d_V_stacked_oooo_pad holds only the
    // coupling (k,l) blocks (slot layout). The (l,k) loop below skips the dense
    // V_stacked write; a separate slot loop writes V_stacked[slot] for the
    // n_slots[idx] coupling (k,l) from the kl-slot list. V_meta stays dense.
    int voooo_sparse,
    const int*    __restrict__ d_kl_slot,      // [Σ n_slots] kl(=k·nocc+l) per slot
    const size_t* __restrict__ d_slot_offset,  // [N_pair+1] kl-slot CSR offsets
    const int*    __restrict__ d_n_slots)      // [N_pair] slots per pair
{
    const int idx = blockIdx.x;
    const int n   = d_n_pno[idx];
    if (n == 0) return;

    // Active-only gate — inactive pairs (a < 0) own no packed slot.
    const int a = d_active_pos[idx];
    if (a < 0) return;

    const size_t per_pair_kl_oooo = static_cast<size_t>(n) * n;
    const size_t pair_oooo_off    = d_v_oooo_off_packed[a];
    const size_t pair_meta_pack_off = d_per_pair_meta_off[idx];
    const size_t nn_ij            = static_cast<size_t>(n) * nocc;
    const size_t pair_src_off     = d_pair_src_off[idx];

    // Outer (l, k) loop, inner (d, c) strided by thread. Writes V_meta (dense)
    // always; writes the dense V_stacked_oooo only when !voooo_sparse.
    for (int l = 0; l < nocc; ++l) {
        for (int k = 0; k < nocc; ++k) {
            const size_t lk_off = static_cast<size_t>(l * nocc + k)
                                * static_cast<size_t>(n) * n;
            const size_t kl_off = static_cast<size_t>(k * nocc + l)
                                * static_cast<size_t>(n) * n;
            const real_t* V_lk = d_V_flat + pair_src_off + lk_off;
            const real_t* T_kl = d_T_flat + pair_src_off + kl_off;
            // V_stacked_oooo row index = (k·nocc + l) (axes swapped vs V_meta).
            const size_t oooo_block_off = pair_oooo_off
                                        + (static_cast<size_t>(k) * nocc + l)
                                          * per_pair_kl_oooo;
            // Packed V_meta row base for the (l, k) block.
            // Packed layout: [l·n + d, k·n + c] = V_lk(d, c) with row
            // stride nn_ij and inner col stride 1.
            const size_t row_base_pack = pair_meta_pack_off
                                       + static_cast<size_t>(l) * n * nn_ij
                                       + static_cast<size_t>(k) * n;

            for (int d = threadIdx.y; d < n; d += blockDim.y) {
                const size_t oooo_off     = oooo_block_off
                                          + static_cast<size_t>(d) * n;
                const size_t row_off_pack = row_base_pack
                                          + static_cast<size_t>(d) * nn_ij;
                for (int c = threadIdx.x; c < n; c += blockDim.x) {
                    const real_t v_dc = V_lk[d * n + c];
                    if (!voooo_sparse)
                        d_V_stacked_oooo_pad [oooo_off + c]  = v_dc;
                    d_V_meta_T_packed    [row_off_pack + c] = v_dc;
                    d_V_meta_TT_packed   [row_off_pack + c] = V_lk[c * n + d];
                    d_T_meta_packed      [row_off_pack + c] = T_kl[c * n + d];
                }
            }
        }
    }

    // Stage D (D3a): sparse V_stacked_oooo — write only the coupling (k,l)
    // blocks at their slot positions. slot s ↔ kl = d_kl_slot[slot_base + s];
    // source V_lk lives at (l·nocc + k) in d_V_flat. The oooo ladder reads
    // V_stacked at the same slot index (Phase 1 sparse).
    if (voooo_sparse) {
        const size_t slot_base = d_slot_offset[idx];
        const int    ns        = d_n_slots[idx];
        for (int s = 0; s < ns; ++s) {
            const int kl = d_kl_slot[slot_base + s];
            const int k  = kl / nocc;
            const int l  = kl % nocc;
            const size_t lk_off = static_cast<size_t>(l * nocc + k)
                                * static_cast<size_t>(n) * n;
            const real_t* V_lk = d_V_flat + pair_src_off + lk_off;
            const size_t slot_off = pair_oooo_off
                                  + static_cast<size_t>(s) * per_pair_kl_oooo;
            for (int d = threadIdx.y; d < n; d += blockDim.y)
                for (int c = threadIdx.x; c < n; c += blockDim.x)
                    d_V_stacked_oooo_pad[slot_off + static_cast<size_t>(d) * n + c]
                        = V_lk[d * n + c];
        }
    }
}

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
// S11 Phase 2d (2026-05-19) — padded sinks removed. Writes go exclusively
// to the per-pair packed buffers (pi_N_packed n_ij × nn_ij, pi_T_packed
// nn_ij × n_ij). Inactive pairs (d_active_pos[idx] < 0) own no slot and
// the thread block returns early. Grid still walks max_n × max_n inner
// because TILE = 16 covers the max possible n_ij; inner threads with
// (a >= n_ij || d >= n_ij) become no-ops.
__global__ void slice_pi_N_T_for_I_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_per_pair_I,
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_active_pos,
    const size_t* __restrict__ d_per_pair_block_off,
    const size_t* __restrict__ d_per_pair_stack_off,
    real_t*       __restrict__ d_pi_N_packed,
    real_t*       __restrict__ d_pi_T_packed,
    int N_pair, int nocc, int max_n,
    int pair_begin,
    // Stage D sparse: (I_lmo,l) slot via d_slot_irow, row stride n_slots·n_ij,
    // d_idx_offset_pi_T = sparse offsets.
    int pit_sparse,
    const int* __restrict__ d_n_slots,
    const int* __restrict__ d_slot_irow)
{
    const int idx = blockIdx.x + pair_begin;
    const int l   = blockIdx.y;
    if (idx >= N_pair || l >= nocc) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;
    const int a_pos = d_active_pos[idx];
    if (a_pos < 0) return;

    const size_t pi_N_pack_off = d_per_pair_block_off[idx];
    const size_t pi_T_pack_off = d_per_pair_stack_off[idx];
    const size_t nn_ij_szt = static_cast<size_t>(n_ij)
                           * static_cast<size_t>(nocc);
    const size_t row_stride = pit_sparse
        ? static_cast<size_t>(d_n_slots[idx]) * n_ij
        : static_cast<size_t>(nocc) * nocc * n_ij;

    // Step 6.6 fix: strided per-thread loop so block (TILE × TILE) covers
    // the full n_ij × n_ij grid even when n_ij > sqrt(1024) = 32. max_n is
    // the upper bound used to size the launch.
    for (int a = threadIdx.y; a < max_n; a += blockDim.y) {
        if (a >= n_ij) continue;
        for (int d = threadIdx.x; d < max_n; d += blockDim.x) {
            if (d >= n_ij) continue;
            real_t v_N = real_t(0);  // oriented(a, d)
            real_t v_T = real_t(0);  // oriented(d, a)

            const int I_lmo = d_per_pair_I[idx];
            const int idx_il = d_pair_lookup[I_lmo * nocc + l];
            const int n_il   = d_n_pno[idx_il];
            int KL = pit_sparse ? -1 : (I_lmo * nocc + l);
            if (pit_sparse && n_il > 0)
                KL = d_slot_irow[static_cast<size_t>(idx) * nocc + l];
            if (n_il > 0 && KL >= 0) {
                const size_t base = d_idx_offset_pi_T[idx]
                    + static_cast<size_t>(KL) * static_cast<size_t>(n_ij);
                v_N = d_pi_T_stack[base + static_cast<size_t>(a) * row_stride + d];
                v_T = d_pi_T_stack[base + static_cast<size_t>(d) * row_stride + a];
            }

            // pi_N_packed row-major (n_ij × nn_ij); pi_T_packed row-major
            // (nn_ij × n_ij). Both restricted to a, d ∈ [0, n_ij).
            d_pi_N_packed[pi_N_pack_off
                        + static_cast<size_t>(a) * nn_ij_szt
                        + static_cast<size_t>(l) * n_ij + d] = v_N;
            d_pi_T_packed[pi_T_pack_off
                        + (static_cast<size_t>(l) * n_ij + d)
                          * static_cast<size_t>(n_ij)
                        + a] = v_T;
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
    // S11 Phase 2c — R write switched to per-pair packed (n_ij × n_ij)
    // at element offset d_per_pair_R_off[idx]. The padded buffer is no
    // longer touched here (Phase 2d will drop the alloc).
    const size_t* __restrict__ d_per_pair_R_off,
    real_t*       __restrict__ d_R_packed,
    int N_pair, int nocc, int /*max_n*/,
    real_t threshold,
    int pair_begin,
    // Stage D sparse pi_T_stack: when pit_sparse, d_idx_offset is the sparse
    // (kl-slot) offset array, row stride is n_slots[idx]·n_ij, and the (k,J_j)
    // column is found via d_slot_jcol[idx·nocc+k] (−1 = not coupled ⇒ pi=0).
    int pit_sparse,
    const int* __restrict__ d_n_slots,
    const int* __restrict__ d_slot_jcol)
{
    const int idx = blockIdx.x + pair_begin;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    const int I_i = d_I_i[idx];
    const int J_j = d_I_j[idx];
    const size_t R_pair_off = d_per_pair_R_off[idx];
    const size_t row_stride = pit_sparse
        ? static_cast<size_t>(d_n_slots[idx]) * n_ij
        : static_cast<size_t>(nocc) * nocc * n_ij;

    // Step 6.6 fix: strided per-thread loop so max_n > sqrt(1024) = 32 works.
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            const size_t abase = d_idx_offset[idx]
                               + static_cast<size_t>(a) * row_stride + d;
            real_t sum = real_t(0);
            for (int k = 0; k < nocc; ++k) {
                if (k == I_i) continue;
                const real_t F_LMO_ik  = d_F_LMO[I_i * nocc + k];
                const real_t dF_ki_val = d_dF_ki[k * nocc + I_i];
                const real_t F_ik = F_LMO_ik + dF_ki_val;
                if (F_ik > -threshold && F_ik < threshold) continue;

                const int idx_kj = d_pair_lookup[k * nocc + J_j];
                if (d_n_pno[idx_kj] == 0) continue;

                int KL;
                if (pit_sparse) {
                    KL = d_slot_jcol[static_cast<size_t>(idx) * nocc + k];
                    if (KL < 0) continue;          // (k,J_j) not in coupling ⇒ 0
                } else {
                    KL = k * nocc + J_j;
                }
                const real_t pi_val = d_pi_T_stack[abase
                                                + static_cast<size_t>(KL) * n_ij];
                sum -= F_ik * pi_val;
            }
            d_R_packed[R_pair_off
                     + static_cast<size_t>(a) * n_ij + d] += sum;
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
    // S11 Phase 2c — R write switched to packed (same semantics as
    // inter_pair_fock_i_kernel).
    const size_t* __restrict__ d_per_pair_R_off,
    real_t*       __restrict__ d_R_packed,
    int N_pair, int nocc, int /*max_n*/,
    real_t threshold,
    int pair_begin,
    // Stage D sparse: d_idx_offset = sparse offsets, row stride n_slots·n_ij,
    // (I_i,l) column via d_slot_irow[idx·nocc+l] (−1 = not coupled).
    int pit_sparse,
    const int* __restrict__ d_n_slots,
    const int* __restrict__ d_slot_irow)
{
    const int idx = blockIdx.x + pair_begin;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    const int I_i = d_I_i[idx];
    const int J_j = d_I_j[idx];
    const size_t R_pair_off = d_per_pair_R_off[idx];
    const size_t row_stride = pit_sparse
        ? static_cast<size_t>(d_n_slots[idx]) * n_ij
        : static_cast<size_t>(nocc) * nocc * n_ij;

    // Step 6.6 fix: strided per-thread loop for max_n > sqrt(1024).
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            const size_t abase = d_idx_offset[idx]
                               + static_cast<size_t>(a) * row_stride + d;
            real_t sum = real_t(0);
            for (int l = 0; l < nocc; ++l) {
                if (l == J_j) continue;
                const real_t F_LMO_lj = d_F_LMO[l * nocc + J_j];
                const real_t dF_lj    = d_dF_ki[l * nocc + J_j];
                const real_t F_lj     = F_LMO_lj + dF_lj;
                if (F_lj > -threshold && F_lj < threshold) continue;

                const int idx_il = d_pair_lookup[I_i * nocc + l];
                if (d_n_pno[idx_il] == 0) continue;

                int KL;
                if (pit_sparse) {
                    KL = d_slot_irow[static_cast<size_t>(idx) * nocc + l];
                    if (KL < 0) continue;
                } else {
                    KL = I_i * nocc + l;
                }
                const real_t pi_val = d_pi_T_stack[abase
                                                + static_cast<size_t>(KL) * n_ij];
                sum -= F_lj * pi_val;
            }
            d_R_packed[R_pair_off
                     + static_cast<size_t>(a) * n_ij + d] += sum;
        }
    }
}

// Step 6.6 — fused oooo ladder kernel (Phase 2.2 refactor with shared-mem W_eff).
//
// Per pair idx (with non-empty n_pno = n_ij) and (a, b) ∈ [0, n_ij)²:
//   R[idx][a, b] += Σ_{kl} W_eff[idx][kl] · π_{kl}^{oriented}[a, b]
// where  W_eff[idx][kl] = W_oooo[idx][kl] + W_dress[idx][kl]
//        W_dress[idx][kl] = Σ_{d, c} V_lk(d, c) · Y_old(c, d)
//                         = Σ_{d, c} V_stacked[idx][kl, d·max_n + c] · Y_pad_T[idx][d·max_n + c]
//
// Optimisation history:
//   - Step 6.6a fused (until 2026-05-13): each thread re-computed
//     W_dress[kl] independently for its (a, b). With n_ij up to 8,
//     this meant 64× redundant inner-loop work per kl. Strided Y reads
//     blew the cache, nsys showed kernel dominating 91.5% of GPU time.
//   - Step 6.6 shmem refactor: each block cooperatively computes the
//     nocc² values of W_eff[kl] ONCE into shared memory, then loops over
//     (a, b) to accumulate. Eliminates 64× redundancy. Shared mem use
//     = nocc² × 8 B (7-30 KB, fits 48 KB per-block limit). 2× kernel speedup.
//   - Step 6.7 (this version): switch Phase 1's strided Y_pad access to
//     pre-transposed `d_Y_pad_T` for fully sequential row-wise reads of
//     the (dd, cc) inner loop. The inner sum is now element-wise product
//     of two contiguous rows V_row[cc] · Y_pad_T_row[cc], hitting the
//     same cache line for the whole cc loop instead of n_ij separate
//     cache lines as before. Cache traffic ↓ by min(n_ij, max_n/8)×
//     (typically 4-50× for production n_pno), with biggest gains at
//     TEOS-class n_pno=50+. d_Y_pad_T is built once per outer T iter
//     by `transpose_Y_pad_kernel` in PiCacheGpu::rebuild_pi, negligible
//     extra cost (~ms-scale).
__global__ void oooo_lad_kernel(
    const real_t* __restrict__ d_V_stacked_oooo_pad,
    const real_t* __restrict__ d_W_oooo,
    const real_t* __restrict__ d_pi_T_stack,
    const real_t* __restrict__ d_Y_pad_T,   // Step 6.7: pre-transposed Y
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_active_pos,        // S10b: orig→a
    const size_t* __restrict__ d_v_oooo_off_packed, // S10b: prefix sum
    // S11 Phase 2c — R write switched to per-pair packed (n_ij × n_ij)
    // at d_per_pair_R_off[idx]. Active guard via d_active_pos already
    // present; inactive pairs early-return before any R touch. max_n is
    // still needed for the padded d_Y_pad_T addressing (built by
    // PiCacheGpu::rebuild_pi at full max_n × max_n).
    const size_t* __restrict__ d_per_pair_R_off,
    real_t*       __restrict__ d_R_packed,
    int N_pair, int nocc, int max_n,
    int pair_begin,
    // Stage D sparse pi_T_stack: Phase 2 reads pi over the kl-slot list instead
    // of all nocc². W_oooo + V_stacked_oooo (Phase 1) stay dense (W_oooo is the
    // O(N·nocc²) blocker; V_stacked sparsify is D3). d_idx_offset_pi_T is the
    // sparse offset array when pit_sparse.
    int pit_sparse,
    const int*    __restrict__ d_n_slots,
    const int*    __restrict__ d_kl_slot,
    const size_t* __restrict__ d_slot_offset,
    // Stage D (D3a): when voooo_sparse, d_V_stacked_oooo_pad holds only the
    // coupling (k,l) blocks (slot layout); Phase 1 iterates the kl-slots and
    // fills s_W_eff[kl] only for coupling (k,l). Requires pit_sparse (Phase 2
    // reads s_W_eff only at coupling kl), guaranteed by the ctor gate.
    int voooo_sparse)
{
    extern __shared__ real_t s_W_eff[];  // size nocc² doubles

    const int idx = blockIdx.x + pair_begin;
    if (idx >= N_pair) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;

    // Step S10b — V_stacked_oooo is packed: per active pair a, layout is
    // (nocc² rows) × (n_ij × n_ij) contiguous starting at the offset
    // d_v_oooo_off_packed[a]. The slab kernel hits idx ∈ [pair_begin,
    // pair_end), and any such idx with n_pno>0 has d_active_pos[idx] >= 0
    // (set in the constructor on every device). If for any reason a == -1
    // we'd crash on the offset table; guard defensively.
    const int a = d_active_pos[idx];
    if (a < 0) return;

    const int    nocc2     = nocc * nocc;
    const int    max_nn    = max_n * max_n;
    const size_t v_pair_off = d_v_oooo_off_packed[a];        // S10b: packed base
    const size_t v_kl_stride = static_cast<size_t>(n_ij) * n_ij;  // S10b: per-(k,l) block size
    const size_t w_pair_off = static_cast<size_t>(idx) * nocc2;
    const size_t y_pair_off = static_cast<size_t>(idx) * max_nn;
    const size_t pi_kl_stride = static_cast<size_t>(n_ij);

    const int tid       = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads  = blockDim.x * blockDim.y;

    // ---- Phase 1: cooperatively compute s_W_eff[kl] = W_oooo[kl] + W_dress. ----
    // Threads stride through kl (dense) or over the kl-slots (D3a sparse). Per
    // kl: read W_oooo + compute W_dress (n_ij² mul-adds on V and Y_pad_T, both
    // accessed sequentially row-wise for L1-friendly cache-line reuse).
    if (voooo_sparse) {
        // V_stacked_oooo holds only coupling blocks (slot s ↔ kl). s_W_eff is
        // filled only at coupling kl; Phase 2 (pit_sparse) reads only those.
        const int    ns_o      = d_n_slots[idx];
        const size_t slot_base_o = d_slot_offset[idx];
        for (int s = tid; s < ns_o; s += nthreads) {
            const int kl = d_kl_slot[slot_base_o + s];
            const real_t* V_lk = d_V_stacked_oooo_pad
                               + v_pair_off
                               + static_cast<size_t>(s) * v_kl_stride;
            real_t W_dress = real_t(0);
            for (int dd = 0; dd < n_ij; ++dd) {
                const real_t* V_row = V_lk + static_cast<size_t>(dd) * n_ij;
                const real_t* Y_row = d_Y_pad_T + y_pair_off
                                    + static_cast<size_t>(dd) * max_n;
                for (int cc = 0; cc < n_ij; ++cc)
                    W_dress += V_row[cc] * Y_row[cc];
            }
            s_W_eff[kl] = d_W_oooo[w_pair_off + kl] + W_dress;
        }
    } else {
        for (int kl = tid; kl < nocc2; kl += nthreads) {
            // Step S10b — V_lk now uses per-(k,l) packed stride n_ij² and
            // per-row stride n_ij (was max_n² block and max_n row stride).
            const real_t* V_lk = d_V_stacked_oooo_pad
                               + v_pair_off
                               + static_cast<size_t>(kl) * v_kl_stride;
            real_t W_dress = real_t(0);
            for (int dd = 0; dd < n_ij; ++dd) {
                const real_t* V_row  = V_lk + static_cast<size_t>(dd) * n_ij;
                const real_t* Y_row  = d_Y_pad_T + y_pair_off
                                     + static_cast<size_t>(dd) * max_n;
                // Y_pad_T[idx, dd, cc] = Y[cc, dd] (per-pair transpose),
                // so the cc-loop is now sequential in both V_row and Y_row.
                for (int cc = 0; cc < n_ij; ++cc) {
                    W_dress += V_row[cc] * Y_row[cc];
                }
            }
            s_W_eff[kl] = d_W_oooo[w_pair_off + kl] + W_dress;
        }
    }
    __syncthreads();

    // ---- Phase 2: each thread handles its (a, b), Σ_kl W_eff[kl] · pi(a, kl, b). ----
    // pi_T_stack layout: per-pair contiguous segment at d_idx_offset_pi_T[idx],
    // pi[a, (k·nocc + l)·n_ij + d]. With kl = k·nocc + l, "b" plays the
    // role of "d" here (we read pi at (a, kl·n_ij + b)). pi reads vary by
    // n_ij stride across kl (strided), but threads within a warp read
    // adjacent b indices for fixed kl (coalesced across the warp).
    const size_t R_pair_off = d_per_pair_R_off[idx];
    const int    n_slots = pit_sparse ? d_n_slots[idx] : 0;
    const size_t row_stride = pit_sparse
        ? static_cast<size_t>(n_slots) * n_ij
        : static_cast<size_t>(nocc) * nocc * n_ij;
    const size_t slot_base = pit_sparse
        ? d_slot_offset[idx] : static_cast<size_t>(0);
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int b = threadIdx.x; b < n_ij; b += blockDim.x) {
            const size_t pi_pair_off = d_idx_offset_pi_T[idx]
                                     + static_cast<size_t>(a) * row_stride + b;
            real_t acc = real_t(0);
            if (pit_sparse) {
                // Iterate only the kl-slots; W_eff still indexed by the real kl.
                for (int s = 0; s < n_slots; ++s) {
                    const int kl = d_kl_slot[slot_base + s];
                    const real_t pi_val = d_pi_T_stack[pi_pair_off
                                             + static_cast<size_t>(s) * pi_kl_stride];
                    acc += s_W_eff[kl] * pi_val;
                }
            } else {
                for (int kl = 0; kl < nocc2; ++kl) {
                    const real_t W_eff  = s_W_eff[kl];
                    const real_t pi_val = d_pi_T_stack[pi_pair_off
                                             + static_cast<size_t>(kl) * pi_kl_stride];
                    acc += W_eff * pi_val;
                }
            }
            // S11 Phase 2c — packed R write (n_ij × n_ij row-major).
            d_R_packed[R_pair_off
                     + static_cast<size_t>(a) * n_ij + b] += acc;
        }
    }
}

// slice_PI_outer_for_J_kernel — extract per-pair (idx, k) blocks at
// l=J_lmo. S11 Phase 2d (2026-05-19) — packed-only writes:
//   PI_stack_packed[off_stack + (k·n_ij + r)·n_ij + c] = oriented_{k,J}(r,c)
//   PI_TT_packed   [off_block + c·nn_ij + k·n_ij + r]  = oriented_{k,J}(c,r)
// Inactive pairs (d_active_pos[idx] < 0) early-return.
__global__ void slice_PI_outer_for_J_kernel(
    const real_t* __restrict__ d_pi_T_stack,
    const size_t* __restrict__ d_idx_offset_pi_T,
    const int*    __restrict__ d_n_pno,
    const int*    __restrict__ d_per_pair_J,
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_active_pos,
    const size_t* __restrict__ d_per_pair_block_off,   // for PI_TT
    const size_t* __restrict__ d_per_pair_stack_off,   // for PI_stack
    real_t*       __restrict__ d_PI_stack_packed,
    real_t*       __restrict__ d_PI_TT_packed,
    int N_pair, int nocc, int max_n,
    int pair_begin,
    // Stage D sparse: (k,J_lmo) slot via d_slot_jcol, row stride n_slots·n_ij,
    // d_idx_offset_pi_T = sparse offsets.
    int pit_sparse,
    const int* __restrict__ d_n_slots,
    const int* __restrict__ d_slot_jcol)
{
    const int idx = blockIdx.x + pair_begin;
    const int k   = blockIdx.y;
    if (idx >= N_pair || k >= nocc) return;

    const int n_ij = d_n_pno[idx];
    if (n_ij == 0) return;
    const int a_pos = d_active_pos[idx];
    if (a_pos < 0) return;

    const size_t PI_stack_pack_off = d_per_pair_stack_off[idx];
    const size_t PI_TT_pack_off    = d_per_pair_block_off[idx];
    const size_t nn_ij_szt = static_cast<size_t>(n_ij)
                           * static_cast<size_t>(nocc);
    const size_t row_stride = pit_sparse
        ? static_cast<size_t>(d_n_slots[idx]) * n_ij
        : static_cast<size_t>(nocc) * nocc * n_ij;

    // Step 6.6 fix: strided per-thread loop for n_ij > sqrt(1024).
    for (int r = threadIdx.y; r < max_n; r += blockDim.y) {
        if (r >= n_ij) continue;
        for (int c = threadIdx.x; c < max_n; c += blockDim.x) {
            if (c >= n_ij) continue;
            real_t v_stack = real_t(0);  // oriented(r, c)
            real_t v_TT    = real_t(0);  // oriented(c, r)

            const int J_lmo = d_per_pair_J[idx];
            const int idx_kJ = d_pair_lookup[k * nocc + J_lmo];
            const int n_kJ   = d_n_pno[idx_kJ];
            int KL = pit_sparse ? -1 : (k * nocc + J_lmo);
            if (pit_sparse && n_kJ > 0)
                KL = d_slot_jcol[static_cast<size_t>(idx) * nocc + k];
            if (n_kJ > 0 && KL >= 0) {
                const size_t base = d_idx_offset_pi_T[idx]
                    + static_cast<size_t>(KL) * static_cast<size_t>(n_ij);
                v_stack = d_pi_T_stack[base + static_cast<size_t>(r) * row_stride + c];
                v_TT    = d_pi_T_stack[base + static_cast<size_t>(c) * row_stride + r];
            }

            d_PI_stack_packed[PI_stack_pack_off
                            + (static_cast<size_t>(k) * n_ij + r)
                              * static_cast<size_t>(n_ij)
                            + c] = v_stack;
            d_PI_TT_packed[PI_TT_pack_off
                         + static_cast<size_t>(c) * nn_ij_szt
                         + static_cast<size_t>(k) * n_ij + r] = v_TT;
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

    // S11 Phase 2d (2026-05-19) — padded buffers removed. All ResidGpu
    // reads/writes now go through the per-pair packed buffers declared
    // below; the legacy N_pair × max-padded layout was dropped because it
    // blocked cholesterol-class activation (1+ TB/dev). See
    // [[dlpno-s11-phase2-detailed-plan]] for the migration history.

    // Step 6.5 — inter-pair Fock i+j device buffers.
    real_t* d_F_LMO  = nullptr;            // (nocc × nocc) iter-invariant
    real_t* d_dF_ki  = nullptr;            // (nocc × nocc) refreshed each iter

    // Step 6.6 — oooo ladder iter-invariant buffers.
    // V_stacked_oooo_pad[idx][kl, d*max_n + c] = V_lk(d, c) for d, c < n_ij else 0
    // (kl varies in [0, nocc²), per-pair flat layout).
    //
    // Step S10b POC (2026-05-17) — V_stacked_oooo_pad is the FIRST buffer to
    // be migrated to per-pair packed storage. Layout:
    //   active a ∈ [0, n_active_in_slab_):
    //     d_V_stacked_oooo_pad[ d_v_oooo_off_packed[a]
    //                         + kl · n_ij² + d·n_ij + c ]
    //                                 = V_lk(d, c)   for d, c < n_ij
    // where kl ∈ [0, nocc²) and n_ij = n_pno_[active_pair_list_[a]]. The
    // packed total = Σ_a (nocc² · n_ij²). Inactive pairs (active_pos[idx]=-1)
    // own no slot. cholesterol: 354 GB → 17.5 GB total (slab÷8 = 2.2 GB/dev).
    //
    // Other buffers (3 meta + 10 block + 4 stack + R) remain full N_pair ×
    // max-padded for now (separate refactor passes).
    real_t* d_V_stacked_oooo_pad = nullptr;   // S10b: packed total = v_oooo_packed_total
    real_t* d_W_oooo             = nullptr;   // (N × nocc²) — unchanged
    int*    d_active_pos         = nullptr;   // (N_pair) orig pair idx → a in [0, n_active_in_slab_) or -1
    size_t* d_v_oooo_off_packed  = nullptr;   // (n_active_in_slab_ + 1) prefix sum of nocc² · n_ij²
    size_t  v_oooo_packed_total  = 0;         // total packed element count (= host offset_packed.back())
    // Stage D (D3a): when true, d_V_stacked_oooo_pad is sized/written/read by
    // the per-pair coupling-slot count (n_slots · n_ij²) instead of nocc²·n_ij².
    // The oooo ladder's Phase-2 sparse pi path only reads W_eff[kl] for coupling
    // (k,l), so the dropped non-coupling W_dress was never used ⇒ bit-exact.
    // Gated on env GANSU_DLPNO_CCSD_VOOOO_SPARSE + pgpu kl-slot list ready.
    bool    voooo_sparse         = false;

    // Step S11 Phase 1 (2026-05-17 night-3) — packed V_meta/T_meta buffers
    // and per-pair offset table. Allocated alongside the legacy padded
    // d_V_meta_*_pad and written by pack_V_meta_kernel via a packed-also
    // path. NOT consumed by any cuBLAS call in Phase 1 — the existing
    // strided batched DGEMMs in run_W_block_build still read the padded
    // buffers, preserving bit-exactness. Phase 2 will switch the cuBLAS
    // calls to bucket loops reading these packed buffers and then drop
    // the padded allocs.
    //
    // Each buffer is a single contiguous device alloc of size
    // packed_meta_total elements (= Σ_active n_ij² · nocc²). The element
    // offset of pair `orig` lives in d_per_pair_meta_off[orig]; inactive
    // pairs (active_pos[orig] == -1) have offset 0 but must NOT be
    // dereferenced — guard on active_pos.
    real_t* d_V_meta_T_packed   = nullptr;
    real_t* d_V_meta_TT_packed  = nullptr;
    real_t* d_T_meta_packed     = nullptr;
    size_t* d_per_pair_meta_off = nullptr;   // (N_pair) elements
    size_t  packed_meta_total   = 0;

    // Step S11 Phase 2a (2026-05-18) — packed device buffers for the
    // remaining 14 cuBLAS-touched ResidGpu buffers + 3 packed offset
    // tables. Allocated alongside the still-live padded scratch; written
    // by Phase 2b's slice / pack_W_bare paths (next commit) and consumed
    // by Phase 2c's cuBLAS bucket loops (later commit). Phase 2a keeps
    // the cuBLAS calls reading the padded buffers, so bit-exactness is
    // preserved.
    //
    // Buffer layout: each is a single contiguous device alloc with size
    // equal to its packed_*_total elements. Per-pair element offsets live
    // in d_per_pair_block/stack/R_off (uploaded from the host vectors of
    // the same name on ResidGpu).
    //   block layout (n_b × nn_b row-major), per-slot elem = n_b · nn_b:
    //     W_bare_ovov_i/j, W_bare_ovvo_i/j, W_block_i/i2/j/j2,
    //     pi_N_i/j, PI_kj_TT, PI_ki_TT                 (12 buffers)
    //   stack layout (nn_b × n_b row-major), per-slot elem = nn_b · n_b
    //   (numerically same as block, separate buffer because Phase 2c
    //   GEMMs use a different leading dim):
    //     pi_T_i/j, PI_kj_stack, PI_ki_stack            (4 buffers)
    //   R layout (n_b × n_b row-major), per-slot elem = n_b²:
    //     R_ph (device) + h_R_ph_packed (pinned host)   (1 device + 1 host)
    real_t* d_W_bare_ovov_i_packed = nullptr;
    real_t* d_W_bare_ovov_j_packed = nullptr;
    real_t* d_W_bare_ovvo_i_packed = nullptr;
    real_t* d_W_bare_ovvo_j_packed = nullptr;
    real_t* d_W_block_i_packed     = nullptr;
    real_t* d_W_block_i2_packed    = nullptr;
    real_t* d_W_block_j_packed     = nullptr;
    real_t* d_W_block_j2_packed    = nullptr;
    real_t* d_pi_N_i_packed        = nullptr;
    real_t* d_pi_N_j_packed        = nullptr;
    real_t* d_PI_kj_TT_packed      = nullptr;
    real_t* d_PI_ki_TT_packed      = nullptr;
    real_t* d_pi_T_i_packed        = nullptr;
    real_t* d_pi_T_j_packed        = nullptr;
    real_t* d_PI_kj_stack_packed   = nullptr;
    real_t* d_PI_ki_stack_packed   = nullptr;
    real_t* d_R_ph_packed          = nullptr;
    real_t* h_R_ph_packed          = nullptr;   // pinned host mirror
    size_t* d_per_pair_block_off   = nullptr;   // (N_pair) elements
    size_t* d_per_pair_stack_off   = nullptr;   // (N_pair) elements
    size_t* d_per_pair_R_off       = nullptr;   // (N_pair) elements
    size_t  packed_block_total     = 0;
    size_t  packed_stack_total     = 0;
    size_t  packed_R_total         = 0;
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

    // Stage-level timing events (default flags = timing enabled). Recorded
    // on the default stream between pipeline stages; read inside
    // compute_finalize after completion_event sync, so all 7 events have
    // been hit by the GPU when we sample them.
    cudaEvent_t e_start       = nullptr;  // top of phladder pipeline
    cudaEvent_t e_after_slice = nullptr;
    cudaEvent_t e_after_W     = nullptr;
    cudaEvent_t e_after_R     = nullptr;
    cudaEvent_t e_after_F     = nullptr;  // after inter-pair Fock kernels
    cudaEvent_t e_after_O     = nullptr;  // after oooo_lad_kernel
    cudaEvent_t e_after_D2H   = nullptr;  // after async D2H of R_ph_pad
    bool        has_stage_record = false;

    // Stage time accumulators (seconds, summed across iter).
    double t_slice      = 0.0;
    double t_W_block    = 0.0;
    double t_R_contract = 0.0;
    double t_inter_fock = 0.0;
    double t_oooo       = 0.0;
    double t_d2h        = 0.0;
    int    n_stage_iter = 0;

    void free_all() {
        if (completion_event) {
            cudaEventDestroy(completion_event);
            completion_event = nullptr;
        }
        auto free_ev = [](cudaEvent_t* e) {
            if (*e) { cudaEventDestroy(*e); *e = nullptr; }
        };
        free_ev(&e_start);
        free_ev(&e_after_slice);
        free_ev(&e_after_W);
        free_ev(&e_after_R);
        free_ev(&e_after_F);
        free_ev(&e_after_O);
        free_ev(&e_after_D2H);
        auto free_d = [](real_t** p) {
            if (*p) { cudaFree(*p); *p = nullptr; }
        };
        auto free_di = [](int** p) {
            if (*p) { cudaFree(*p); *p = nullptr; }
        };
        free_di(&d_I_i);
        free_di(&d_I_j);
        // S11 Phase 2d — legacy padded buffers removed.
        free_d(&d_F_LMO);
        free_d(&d_dF_ki);
        free_d(&d_V_stacked_oooo_pad);
        free_d(&d_W_oooo);
        free_di(&d_active_pos);
        if (d_v_oooo_off_packed) {
            cudaFree(d_v_oooo_off_packed);
            d_v_oooo_off_packed = nullptr;
        }
        // Step S11 Phase 1 — packed V_meta/T_meta + per-pair offset.
        free_d(&d_V_meta_T_packed);
        free_d(&d_V_meta_TT_packed);
        free_d(&d_T_meta_packed);
        if (d_per_pair_meta_off) {
            cudaFree(d_per_pair_meta_off);
            d_per_pair_meta_off = nullptr;
        }
        // Step S11 Phase 2a — packed W_bare/W_block/pi_N/pi_T/PI_*/R + 3
        // per-pair offset tables (block/stack/R). Pinned host R mirror
        // released via cudaFreeHost; device buffers via cudaFree.
        free_d(&d_W_bare_ovov_i_packed);
        free_d(&d_W_bare_ovov_j_packed);
        free_d(&d_W_bare_ovvo_i_packed);
        free_d(&d_W_bare_ovvo_j_packed);
        free_d(&d_W_block_i_packed);
        free_d(&d_W_block_i2_packed);
        free_d(&d_W_block_j_packed);
        free_d(&d_W_block_j2_packed);
        free_d(&d_pi_N_i_packed);
        free_d(&d_pi_N_j_packed);
        free_d(&d_PI_kj_TT_packed);
        free_d(&d_PI_ki_TT_packed);
        free_d(&d_pi_T_i_packed);
        free_d(&d_pi_T_j_packed);
        free_d(&d_PI_kj_stack_packed);
        free_d(&d_PI_ki_stack_packed);
        free_d(&d_R_ph_packed);
        if (h_R_ph_packed) {
            cudaFreeHost(h_R_ph_packed);
            h_R_ph_packed = nullptr;
        }
        if (d_per_pair_block_off) {
            cudaFree(d_per_pair_block_off);
            d_per_pair_block_off = nullptr;
        }
        if (d_per_pair_stack_off) {
            cudaFree(d_per_pair_stack_off);
            d_per_pair_stack_off = nullptr;
        }
        if (d_per_pair_R_off) {
            cudaFree(d_per_pair_R_off);
            d_per_pair_R_off = nullptr;
        }
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

    // Step S10 scaffolding — build the slab-active list and orig→compact
    // position map. The kernels in THIS version still index by orig pair
    // idx, so this data is only used to (a) log the projected memory
    // requirement under an active-only buffer layout, and (b) be ready for
    // the next session's reindex step. n_pno_ is captured above and is
    // const for this PiCacheGpu lifetime (PNOs are re-truncated by the
    // SC-PNO refresh, which reconstructs the entire ResidGpu).
    {
        const int ib_for_active = pgpu.pair_begin();
        const int ie_for_active = pgpu.pair_end();
        active_pair_list_.clear();
        active_pair_list_.reserve(
            static_cast<size_t>(std::max(0, ie_for_active - ib_for_active)));
        active_pos_.assign(static_cast<size_t>(N_pair_), -1);
        for (int idx = ib_for_active; idx < ie_for_active; ++idx) {
            if (n_pno_[idx] > 0) {
                active_pos_[idx] =
                    static_cast<int>(active_pair_list_.size());
                active_pair_list_.push_back(idx);
            }
        }
        n_active_in_slab_ = static_cast<int>(active_pair_list_.size());
    }

    // Step S11 Phase 1 scaffolding (2026-05-17 night-3) — bucket-by-n_ij
    // grouping. Sort the slab's active pairs by n_pno, partition into
    // contiguous buckets that share one n_ij value, and record per-pair
    // packed offsets into the (future) packed V_meta buffer. The data
    // structures are built unconditionally so the budget log can show
    // projected packed savings even when ResidGpu ends up inactive on
    // this device; only the device-side allocations (below) are gated on
    // the GPU path.
    //
    // Sort is stable so that within a bucket the per-pair order matches
    // ascending orig pair idx — irrelevant for bit-exactness but easier
    // to reason about when debugging packed offsets.
    {
        bucket_active_pair_list_ = active_pair_list_;
        std::stable_sort(
            bucket_active_pair_list_.begin(),
            bucket_active_pair_list_.end(),
            [this](int a, int b) {
                return n_pno_[a] < n_pno_[b];
            });

        bucket_n_ij_.clear();
        bucket_count_.clear();
        bucket_first_.clear();
        bucket_first_.push_back(0);
        for (size_t i = 0; i < bucket_active_pair_list_.size(); ) {
            const int n_ij = n_pno_[bucket_active_pair_list_[i]];
            size_t j = i;
            while (j < bucket_active_pair_list_.size()
                   && n_pno_[bucket_active_pair_list_[j]] == n_ij) {
                ++j;
            }
            bucket_n_ij_.push_back(n_ij);
            bucket_count_.push_back(static_cast<int>(j - i));
            bucket_first_.push_back(static_cast<int>(j));
            i = j;
        }

        // Per-pair packed offsets (element offset into each packed buffer).
        // Inactive pairs (active_pos_[orig] == -1) keep 0 — but callers
        // must guard via active_pos_ before dereferencing.
        //
        // Step S11 Phase 2a (2026-05-18) — also build 3 additional
        // prefix-sum tables for the block / stack / R packed buffers, in
        // the same bucket pass. block and stack have the same per-slot
        // element count (n_b · nn_b) but are split into separate buffers
        // because Phase 2c's cuBLAS will use a different leading dim
        // (nn_b for stack, n_b for block).
        per_pair_meta_off_.assign(static_cast<size_t>(N_pair_), 0);
        per_pair_block_off_.assign(static_cast<size_t>(N_pair_), 0);
        per_pair_stack_off_.assign(static_cast<size_t>(N_pair_), 0);
        per_pair_R_off_.assign(static_cast<size_t>(N_pair_), 0);
        const size_t n_buckets = bucket_n_ij_.size();
        bucket_base_meta_.assign(n_buckets + 1, 0);
        bucket_base_block_.assign(n_buckets + 1, 0);
        bucket_base_stack_.assign(n_buckets + 1, 0);
        bucket_base_R_.assign(n_buckets + 1, 0);
        size_t off_meta  = 0;
        size_t off_block = 0;
        size_t off_stack = 0;
        size_t off_R     = 0;
        for (size_t b = 0; b < n_buckets; ++b) {
            bucket_base_meta_[b]  = off_meta;
            bucket_base_block_[b] = off_block;
            bucket_base_stack_[b] = off_stack;
            bucket_base_R_[b]     = off_R;
            const long long n_b  = bucket_n_ij_[b];
            const long long nn_b = n_b * static_cast<long long>(nocc_);
            const size_t per_slot_meta  = static_cast<size_t>(nn_b)
                                        * static_cast<size_t>(nn_b);
            const size_t per_slot_block = static_cast<size_t>(n_b)
                                        * static_cast<size_t>(nn_b);
            const size_t per_slot_stack = static_cast<size_t>(nn_b)
                                        * static_cast<size_t>(n_b);
            const size_t per_slot_R     = static_cast<size_t>(n_b)
                                        * static_cast<size_t>(n_b);
            const int bf  = bucket_first_[b];
            const int cnt = bucket_count_[b];
            for (int p = 0; p < cnt; ++p) {
                const int orig = bucket_active_pair_list_[bf + p];
                per_pair_meta_off_[orig]  = off_meta
                                          + static_cast<size_t>(p) * per_slot_meta;
                per_pair_block_off_[orig] = off_block
                                          + static_cast<size_t>(p) * per_slot_block;
                per_pair_stack_off_[orig] = off_stack
                                          + static_cast<size_t>(p) * per_slot_stack;
                per_pair_R_off_[orig]     = off_R
                                          + static_cast<size_t>(p) * per_slot_R;
            }
            off_meta  += static_cast<size_t>(cnt) * per_slot_meta;
            off_block += static_cast<size_t>(cnt) * per_slot_block;
            off_stack += static_cast<size_t>(cnt) * per_slot_stack;
            off_R     += static_cast<size_t>(cnt) * per_slot_R;
        }
        bucket_base_meta_[n_buckets]  = off_meta;
        bucket_base_block_[n_buckets] = off_block;
        bucket_base_stack_[n_buckets] = off_stack;
        bucket_base_R_[n_buckets]     = off_R;
        packed_meta_total_  = off_meta;
        packed_block_total_ = off_block;
        packed_stack_total_ = off_stack;
        packed_R_total_     = off_R;
    }

#ifndef GANSU_CPU_ONLY
    // Multi-GPU: bind to pgpu's device. Allocations and uploads land on the
    // requested device thanks to the DeviceGuard.
    MultiGpuManager::DeviceGuard _guard(pgpu.device_id());

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

    // Step S10b POC — V_stacked_oooo per-pair packed offset table.
    // Per active pair a: occupies (nocc · nocc · n_ij²) elements, contiguous.
    // The (n_active_in_slab_ + 1)-long prefix-sum array is uploaded to
    // device for kernel-side indexing. Both kernels that read/write
    // d_V_stacked_oooo_pad (pack_V_meta_kernel writer, oooo_lad_kernel
    // reader) consult this table via d_v_oooo_off_packed[a].
    // Stage D (D3a): decide whether V_stacked_oooo is sparse (per-pair
    // coupling-slot sized) — requires the pgpu kl-slot list to be built and the
    // opt-in env flag. host n_slots is indexed by ORIG pair idx.
    const int* h_n_slots_oooo = pgpu.host_n_slots();
    {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_VOOOO_SPARSE");
        // Require pitstack_sparse(): the oooo Phase-1 sparse path fills s_W_eff
        // only for coupling (k,l), so Phase-2 MUST iterate the kl-slots (the
        // pit_sparse branch), else non-coupling s_W_eff garbage would be read.
        s.voooo_sparse = (e && e[0] == '1') && (h_n_slots_oooo != nullptr)
                       && pgpu.pitstack_sparse();
    }
    std::vector<size_t> v_oooo_off_packed_host(
        static_cast<size_t>(n_active_in_slab_) + 1, 0);
    for (int a = 0; a < n_active_in_slab_; ++a) {
        const int    idx  = active_pair_list_[a];
        const int    n_ij = n_pno_[idx];
        // Dense: nocc²·n²; sparse (D3a): n_slots·n² (n_slots ≈ 2·coupling).
        const size_t kl_count = s.voooo_sparse
            ? static_cast<size_t>(h_n_slots_oooo[idx])
            : static_cast<size_t>(nocc_) * nocc_;
        const size_t per  = kl_count * static_cast<size_t>(n_ij) * n_ij;
        v_oooo_off_packed_host[a + 1] = v_oooo_off_packed_host[a] + per;
    }
    s.v_oooo_packed_total = v_oooo_off_packed_host[n_active_in_slab_];

    // Memory budget.
    const size_t per_pair_meta   = static_cast<size_t>(max_nn) * max_nn;          // V_meta_T etc
    const size_t per_pair_block  = static_cast<size_t>(max_n) * max_nn;           // W_block / W_bare / pi_N / PI_TT
    const size_t per_pair_stack  = static_cast<size_t>(max_nn) * max_n;           // pi_T / PI_stack
    const size_t per_pair_R      = static_cast<size_t>(max_n) * max_n;            // R_ph
    const size_t bytes_meta_full   = static_cast<size_t>(N_pair_) * per_pair_meta  * sizeof(real_t);
    const size_t bytes_block_full  = static_cast<size_t>(N_pair_) * per_pair_block * sizeof(real_t);
    const size_t bytes_stack_full  = static_cast<size_t>(N_pair_) * per_pair_stack * sizeof(real_t);
    const size_t bytes_R_full      = static_cast<size_t>(N_pair_) * per_pair_R     * sizeof(real_t);
    // Step S10b — V_stacked_oooo is now packed (per active pair, no padding).
    const size_t bytes_v_oooo_packed = s.v_oooo_packed_total * sizeof(real_t);

    // Step S11 Phase 1 — packed V_meta byte count (one of the three meta
    // buffers; total alloc = 3 × bytes_meta_packed). This is the projected
    // per-meta footprint once the cuBLAS bucket loops in Phase 2 let us
    // drop the padded `bytes_meta_full` allocs. Phase 1 keeps BOTH the
    // padded and packed allocs live — the +3 × bytes_meta_packed is
    // included in `need` so the budget check fails cleanly if the device
    // can't host both.
    const size_t bytes_meta_packed = packed_meta_total_ * sizeof(real_t);

    // Step S11 Phase 1 — projected need under FULL packed layout (all 17
    // buffers, including R_ph and the per-iter scratch). Used only in the
    // budget log to show how close cholesterol-class ResidGpu activation
    // gets once Phase 2/3 are complete; not consumed by the gating
    // decision in Phase 1.
    size_t bytes_block_packed_elem = 0;  // sum of n_ij · nn_ij
    size_t bytes_stack_packed_elem = 0;  // sum of nn_ij · n_ij (same total)
    size_t bytes_R_packed_elem     = 0;  // sum of n_ij²
    size_t bytes_src_pack_elem     = 0;  // sum of n_ij² · nocc² (V_flat / T_flat per active pair)
    for (int orig : active_pair_list_) {
        const long long n_ij  = n_pno_[orig];
        const long long nn_ij = n_ij * static_cast<long long>(nocc_);
        bytes_block_packed_elem += static_cast<size_t>(n_ij * nn_ij);
        bytes_stack_packed_elem += static_cast<size_t>(nn_ij * n_ij);
        bytes_R_packed_elem     += static_cast<size_t>(n_ij * n_ij);
        bytes_src_pack_elem     += static_cast<size_t>(n_ij * n_ij)
                                 * static_cast<size_t>(nocc_) * nocc_;
    }
    const size_t bytes_block_packed = bytes_block_packed_elem * sizeof(real_t);
    const size_t bytes_stack_packed = bytes_stack_packed_elem * sizeof(real_t);
    const size_t bytes_R_packed     = bytes_R_packed_elem     * sizeof(real_t);
    // S11 Phase 2e (2026-05-19) — transient V_flat / T_flat staging
    // alloc'd during the pack_V_meta_kernel setup. Sized to slab-active
    // pairs only (Phase 2e restrict); peaks at 2 × bytes_src_pack
    // (host pinned + device, freed before iter loop starts). Included
    // in `need` so the gate fails cleanly before cudaMalloc OOMs.
    const size_t bytes_src_pack     = bytes_src_pack_elem     * sizeof(real_t);

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
        // Step S10b — V_stacked_oooo is now packed (slab × per-pair n_ij²).
        // The budget reflects this; bytes_v_oooo_full is retained only as a
        // reference for the log.
        // S11 Phase 2d/2e (2026-05-19) — padded allocs removed; transient
        // V_flat / T_flat staging (Phase 2e slab-active restrict)
        // included in `need` so the budget gate fails cleanly before
        // cudaMalloc OOM. Composition:
        //   3 × meta_packed (V_meta_T/TT + T_meta)
        //   12 × block_packed (W_bare ×4 + W_block ×4 + pi_N ×2 + PI_TT ×2)
        //   4 × stack_packed (pi_T ×2 + PI_stack ×2)
        //   2 × R_packed (device + pinned host mirror)
        //   + bytes_v_oooo_packed
        //   + 2 × bytes_src_pack (transient V_flat + T_flat at ctor)
        //   + 128 MiB overhead
        const size_t need = 3 * bytes_meta_packed
                          + 12 * bytes_block_packed
                          + 4  * bytes_stack_packed
                          + 2  * bytes_R_packed
                          + bytes_v_oooo_packed
                          + 2  * bytes_src_pack
                          + (size_t)128 * 1024 * 1024;
        // need_packed retained as alias of `need` for log-line backwards
        // compatibility (the two were distinct during the Phase 2a-2c
        // transition while padded + packed coexisted).
        const size_t need_packed = need;
        // Step A0 diagnostics — always log the budget so we can see why the
        // GPU path is gated on a given system. Single line per device per
        // construction; ms wall is negligible. Slab range [pair_begin_,
        // pair_end_) is shown so we can correlate with per-pgpu slab sizes.
        // Bytes-to-MiB conversion uses 1024² to match cudaMemGetInfo idioms.
        constexpr double kMiB = 1024.0 * 1024.0;
        const double mib_free   = static_cast<double>(free_b) / kMiB;
        const double mib_need   = static_cast<double>(need)   / kMiB;
        const double mib_meta   = static_cast<double>(bytes_meta_full)  / kMiB;
        const double mib_block  = static_cast<double>(bytes_block_full) / kMiB;
        const double mib_stack  = static_cast<double>(bytes_stack_full) / kMiB;
        const double mib_R      = static_cast<double>(bytes_R_full)     / kMiB;
        const double mib_voooo_full   = static_cast<double>(bytes_v_oooo_full)  / kMiB;
        const double mib_voooo_packed = static_cast<double>(bytes_v_oooo_packed) / kMiB;
        // Step S10 — projected need under active-only-slab layout (kernels
        // not yet adapted, see scaffolding members `active_pair_list_` /
        // `active_pos_`). Scale-by-active-only assumes the same per_pair
        // strides but only n_active_in_slab_ pairs are stored on this
        // device. v_oooo scales the same way; small (block/stack/R) buffers
        // included for completeness. Compare proj_need to mib_free to see
        // whether the active-slab refactor would close the budget gap.
        const double scale_active =
            (N_pair_ > 0)
            ? static_cast<double>(n_active_in_slab_)
              / static_cast<double>(N_pair_)
            : 0.0;
        const double mib_proj_need = mib_need * scale_active;
        // S11 Phase 1 — packed projections.
        const double mib_meta_packed  = static_cast<double>(bytes_meta_packed)  / kMiB;
        const double mib_block_packed = static_cast<double>(bytes_block_packed) / kMiB;
        const double mib_stack_packed = static_cast<double>(bytes_stack_packed) / kMiB;
        const double mib_R_packed     = static_cast<double>(bytes_R_packed)     / kMiB;
        const double mib_need_packed  = static_cast<double>(need_packed)        / kMiB;
        std::printf(
            "[ResidGpu-budget dev=%d slab=[%d,%d) n_active=%d N_pair=%d nocc=%d max_n=%d max_nn=%d]"
            " free=%.0f need=%.0f proj_active=%.0f need_packed=%.0f MiB"
            "  meta(x3)=%.0f block(x10)=%.0f stack(x4)=%.0f R(x2)=%.0f"
            "  packed meta(x3)=%.0f block(x10)=%.0f stack(x4)=%.0f R(x2)=%.0f"
            "  v_oooo_full=%.0f v_oooo_packed=%.0f MiB  n_buckets=%zu active=%s\n",
            pgpu.device_id(), pgpu.pair_begin(), pgpu.pair_end(),
            n_active_in_slab_, N_pair_, nocc_, max_n_, max_nn,
            mib_free, mib_need, mib_proj_need, mib_need_packed,
            3 * mib_meta, 10 * mib_block, 4 * mib_stack, 2 * mib_R,
            3 * mib_meta_packed, 10 * mib_block_packed,
            4 * mib_stack_packed, 2 * mib_R_packed,
            mib_voooo_full, mib_voooo_packed,
            bucket_n_ij_.size(),
            (need > free_b) ? "NO" : "YES");
        std::fflush(stdout);
        if (need > free_b) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
    }

    try {
        // S11 Phase 2d — legacy padded allocs removed. All ResidGpu
        // device storage is now per-pair packed (see the d_*_packed
        // block below + S10b's d_V_stacked_oooo_pad).
        check_cuda_(cudaMalloc(&s.d_I_i, static_cast<size_t>(N_pair_) * sizeof(int)), "alloc d_I_i");
        check_cuda_(cudaMalloc(&s.d_I_j, static_cast<size_t>(N_pair_) * sizeof(int)), "alloc d_I_j");

        const size_t bytes_F = static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t);
        check_cuda_(cudaMalloc(&s.d_F_LMO,           bytes_F),          "alloc d_F_LMO");
        check_cuda_(cudaMalloc(&s.d_dF_ki,           bytes_F),          "alloc d_dF_ki");

        // Step 6.6 — oooo ladder iter-invariant buffers.
        // Step S10b: V_stacked_oooo is per-pair packed (no per-pair max
        // padding, only active slab pairs). Allocation falls back to a
        // 1-byte stub when n_active_in_slab_=0 so the device pointer is
        // never null for the kernel's branch-free check.
        const size_t bytes_w_oooo = static_cast<size_t>(N_pair_) * nocc_ * nocc_
                                  * sizeof(real_t);
        check_cuda_(cudaMalloc(&s.d_V_stacked_oooo_pad,
                               bytes_v_oooo_packed > 0 ? bytes_v_oooo_packed : 1),
                    "alloc d_V_stacked_oooo_pad (packed)");
        check_cuda_(cudaMalloc(&s.d_W_oooo,             bytes_w_oooo),
                    "alloc d_W_oooo");
        // Step S10b — d_active_pos: orig pair idx → a in [0, n_active_in_slab_) or -1.
        check_cuda_(cudaMalloc(&s.d_active_pos,
                               static_cast<size_t>(N_pair_) * sizeof(int)),
                    "alloc d_active_pos");
        check_cuda_(cudaMemcpy(s.d_active_pos, active_pos_.data(),
                               static_cast<size_t>(N_pair_) * sizeof(int),
                               cudaMemcpyHostToDevice),
                    "H2D d_active_pos");
        // Step S10b — d_v_oooo_off_packed: prefix sum of nocc² · n_ij² across
        // active slab pairs. Length n_active_in_slab_ + 1 (final entry = total).
        check_cuda_(cudaMalloc(&s.d_v_oooo_off_packed,
                               static_cast<size_t>(n_active_in_slab_ + 1)
                                   * sizeof(size_t)),
                    "alloc d_v_oooo_off_packed");
        check_cuda_(cudaMemcpy(s.d_v_oooo_off_packed,
                               v_oooo_off_packed_host.data(),
                               static_cast<size_t>(n_active_in_slab_ + 1)
                                   * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_v_oooo_off_packed");

        // Step S11 Phase 1 (2026-05-17 night-3) — packed V_meta/T_meta
        // buffers and per-pair offset table on device. Allocated alongside
        // the still-live padded d_V_meta_*_pad; written by the packed-also
        // path of pack_V_meta_kernel; NOT yet consumed by cuBLAS (Phase 2
        // will switch the bucket-loop GEMMs to read from these).
        //
        // Falls back to a 1-byte stub when packed_meta_total_ == 0 so the
        // device pointers are never null.
        const size_t bytes_meta_packed_alloc =
            bytes_meta_packed > 0 ? bytes_meta_packed : 1;
        check_cuda_(cudaMalloc(&s.d_V_meta_T_packed,  bytes_meta_packed_alloc),
                    "alloc d_V_meta_T_packed");
        check_cuda_(cudaMalloc(&s.d_V_meta_TT_packed, bytes_meta_packed_alloc),
                    "alloc d_V_meta_TT_packed");
        check_cuda_(cudaMalloc(&s.d_T_meta_packed,    bytes_meta_packed_alloc),
                    "alloc d_T_meta_packed");
        s.packed_meta_total = packed_meta_total_;
        check_cuda_(cudaMalloc(&s.d_per_pair_meta_off,
                               static_cast<size_t>(N_pair_) * sizeof(size_t)),
                    "alloc d_per_pair_meta_off");
        check_cuda_(cudaMemcpy(s.d_per_pair_meta_off,
                               per_pair_meta_off_.data(),
                               static_cast<size_t>(N_pair_) * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_per_pair_meta_off");

        // Step S11 Phase 2a (2026-05-18) — packed W_bare/W_block/pi_N/pi_T/
        // PI_*/R buffers + per-pair block/stack/R offset tables on device.
        // Allocated alongside the still-live padded scratch; Phase 2b will
        // write to these via slice / pack_W_bare paths, Phase 2c will read
        // them via cuBLAS bucket loops. Phase 2a is alloc-only — bit-exact
        // is preserved because the cuBLAS calls still read the padded
        // buffers and the packed buffers are never consumed.
        //
        // 1-byte stub for empty slabs so device pointers are never null.
        const size_t bytes_block_packed_alloc =
            bytes_block_packed > 0 ? bytes_block_packed : 1;
        const size_t bytes_stack_packed_alloc =
            bytes_stack_packed > 0 ? bytes_stack_packed : 1;
        const size_t bytes_R_packed_alloc =
            bytes_R_packed > 0 ? bytes_R_packed : 1;
        check_cuda_(cudaMalloc(&s.d_W_bare_ovov_i_packed, bytes_block_packed_alloc),
                    "alloc d_W_bare_ovov_i_packed");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovov_j_packed, bytes_block_packed_alloc),
                    "alloc d_W_bare_ovov_j_packed");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovvo_i_packed, bytes_block_packed_alloc),
                    "alloc d_W_bare_ovvo_i_packed");
        check_cuda_(cudaMalloc(&s.d_W_bare_ovvo_j_packed, bytes_block_packed_alloc),
                    "alloc d_W_bare_ovvo_j_packed");
        check_cuda_(cudaMalloc(&s.d_W_block_i_packed,     bytes_block_packed_alloc),
                    "alloc d_W_block_i_packed");
        check_cuda_(cudaMalloc(&s.d_W_block_i2_packed,    bytes_block_packed_alloc),
                    "alloc d_W_block_i2_packed");
        check_cuda_(cudaMalloc(&s.d_W_block_j_packed,     bytes_block_packed_alloc),
                    "alloc d_W_block_j_packed");
        check_cuda_(cudaMalloc(&s.d_W_block_j2_packed,    bytes_block_packed_alloc),
                    "alloc d_W_block_j2_packed");
        check_cuda_(cudaMalloc(&s.d_pi_N_i_packed,        bytes_block_packed_alloc),
                    "alloc d_pi_N_i_packed");
        check_cuda_(cudaMalloc(&s.d_pi_N_j_packed,        bytes_block_packed_alloc),
                    "alloc d_pi_N_j_packed");
        check_cuda_(cudaMalloc(&s.d_PI_kj_TT_packed,      bytes_block_packed_alloc),
                    "alloc d_PI_kj_TT_packed");
        check_cuda_(cudaMalloc(&s.d_PI_ki_TT_packed,      bytes_block_packed_alloc),
                    "alloc d_PI_ki_TT_packed");
        check_cuda_(cudaMalloc(&s.d_pi_T_i_packed,        bytes_stack_packed_alloc),
                    "alloc d_pi_T_i_packed");
        check_cuda_(cudaMalloc(&s.d_pi_T_j_packed,        bytes_stack_packed_alloc),
                    "alloc d_pi_T_j_packed");
        check_cuda_(cudaMalloc(&s.d_PI_kj_stack_packed,   bytes_stack_packed_alloc),
                    "alloc d_PI_kj_stack_packed");
        check_cuda_(cudaMalloc(&s.d_PI_ki_stack_packed,   bytes_stack_packed_alloc),
                    "alloc d_PI_ki_stack_packed");
        check_cuda_(cudaMalloc(&s.d_R_ph_packed,          bytes_R_packed_alloc),
                    "alloc d_R_ph_packed");
        check_cuda_(cudaMallocHost(&s.h_R_ph_packed,      bytes_R_packed_alloc),
                    "alloc h_R_ph_packed pinned");
        s.packed_block_total = packed_block_total_;
        s.packed_stack_total = packed_stack_total_;
        s.packed_R_total     = packed_R_total_;
        check_cuda_(cudaMalloc(&s.d_per_pair_block_off,
                               static_cast<size_t>(N_pair_) * sizeof(size_t)),
                    "alloc d_per_pair_block_off");
        check_cuda_(cudaMalloc(&s.d_per_pair_stack_off,
                               static_cast<size_t>(N_pair_) * sizeof(size_t)),
                    "alloc d_per_pair_stack_off");
        check_cuda_(cudaMalloc(&s.d_per_pair_R_off,
                               static_cast<size_t>(N_pair_) * sizeof(size_t)),
                    "alloc d_per_pair_R_off");
        check_cuda_(cudaMemcpy(s.d_per_pair_block_off,
                               per_pair_block_off_.data(),
                               static_cast<size_t>(N_pair_) * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_per_pair_block_off");
        check_cuda_(cudaMemcpy(s.d_per_pair_stack_off,
                               per_pair_stack_off_.data(),
                               static_cast<size_t>(N_pair_) * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_per_pair_stack_off");
        check_cuda_(cudaMemcpy(s.d_per_pair_R_off,
                               per_pair_R_off_.data(),
                               static_cast<size_t>(N_pair_) * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D d_per_pair_R_off");
    } catch (const std::exception&) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    // Per-device cuBLAS handle. Fallback to thread-local single-GPU handle.
    s.cublas = nullptr;
    {
        auto& mgm = MultiGpuManager::instance();
        if (mgm.num_devices() > pgpu.device_id()) {
            s.cublas = mgm.cublas(pgpu.device_id());
        }
        if (!s.cublas) s.cublas = gpu::GPUHandle::cublas();
    }
    if (!s.cublas) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    // S11 Phase 2e (2026-05-19) — opt in to dynamic shared memory for
    // oooo_lad_kernel when nocc² · sizeof(real_t) exceeds the 48 KB
    // default cap. At nocc=108 (cholesterol class) the kernel needs
    // 91 KB shared mem, which exceeds the default and triggers launch
    // failure unless we set this attribute. Hopper (compute 9.0) supports
    // up to ~228 KB; Ampere up to 100 KB. cudaDeviceProp gives the exact
    // ceiling; we clamp to that. Failure to set is non-fatal (kernel
    // launch will surface a clear error message instead).
    {
        const size_t shmem_needed =
            static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t);
        if (shmem_needed > 48u * 1024u) {
            cudaDeviceProp prop{};
            if (cudaGetDeviceProperties(&prop, pgpu.device_id())
                    == cudaSuccess) {
                const size_t max_smem = prop.sharedMemPerBlockOptin
                    ? prop.sharedMemPerBlockOptin
                    : prop.sharedMemPerBlock;
                const size_t target =
                    shmem_needed < max_smem ? shmem_needed : max_smem;
                cudaFuncSetAttribute(
                    oooo_lad_kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(target));
            }
        }
    }

    // Step 6.4: completion event (default flags — disable timing for lower
    // overhead; we only use it for stream-side synchronisation).
    if (cudaEventCreateWithFlags(&s.completion_event, cudaEventDisableTiming)
            != cudaSuccess) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    // Stage-level timing events (default flags = timing enabled). If
    // creation fails on any event the per-stage breakdown is silently
    // disabled — pipeline still runs identically.
    {
        cudaEvent_t* evs[] = {
            &s.e_start, &s.e_after_slice, &s.e_after_W, &s.e_after_R,
            &s.e_after_F, &s.e_after_O, &s.e_after_D2H,
        };
        bool ok = true;
        for (cudaEvent_t* ev : evs) {
            if (cudaEventCreate(ev) != cudaSuccess) { ok = false; break; }
        }
        if (!ok) {
            // Stage timing unavailable; release any successfully created
            // events and continue without per-stage instrumentation.
            for (cudaEvent_t* ev : evs) {
                if (*ev) { cudaEventDestroy(*ev); *ev = nullptr; }
            }
        }
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
    // Packed layouts (device, row-major (n_ij × nn_ij) per active pair) — written
    // by pack_V_meta_kernel.
    //
    // S11 Phase 2e (2026-05-19) — slab+active restrict: V_flat / T_flat
    // staging buffers (host pinned + device) include ONLY pairs that
    // belong to this device's slab AND have n_pno > 0. Non-slab pairs
    // contribute zero source bytes; the kernel guards against them via
    // d_active_pos[idx] < 0 early-return so their pair_src_off slot is
    // never dereferenced. This shrinks the transient V_flat/T_flat from
    // O(Σ_all n_i² · nocc²) to O(Σ_slab_active n_i² · nocc²), critical
    // for cholesterol-class activation (was OOMing on dev 0 at ~19 GB).
    {
        std::vector<size_t> pair_src_off_host(N_pair_ + 1, 0);
        for (int idx = 0; idx < N_pair_; ++idx) {
            const int n = n_pno_[idx];
            const bool slab_active = (active_pos_[idx] >= 0);
            const size_t pair_sz = slab_active
                ? static_cast<size_t>(n) * n
                    * static_cast<size_t>(nocc) * nocc
                : static_cast<size_t>(0);
            pair_src_off_host[idx + 1] = pair_src_off_host[idx] + pair_sz;
        }
        const size_t total_src_sz = pair_src_off_host.back();

        // Host pinned source buffers (slab-active sized; 1-byte stub for
        // empty slabs so the pointer is non-null).
        const size_t total_src_sz_alloc = total_src_sz > 0 ? total_src_sz : 1;
        // Pageable (pinned=false): one-time staging; avoids the mmap-lock
        // contention of concurrent cudaMallocHost across the N_gpu ctors.
        PinnedHost h_V_flat(total_src_sz_alloc, /*pinned=*/false);
        PinnedHost h_T_flat(total_src_sz_alloc, /*pinned=*/false);
        if (!h_V_flat.valid() || !h_T_flat.valid()) {
            delete p_; p_ = nullptr; active_ = false; return;
        }

        // Gather: only slab-active pairs contribute. Non-slab and n=0
        // pairs are skipped; their pair_src_off slot is zero-length so
        // the kernel's active_pos guard prevents any out-of-range read.
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < N_pair_; ++idx) {
            if (active_pos_[idx] < 0) continue;
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (phase24.V_ovov_pair[idx].empty()) continue;
            if (phase24.T_pair[idx].empty())     continue;
            const size_t off = pair_src_off_host[idx];
            const size_t pair_sz =
                pair_src_off_host[idx + 1] - off;
            std::memcpy(h_V_flat.data() + off,
                        phase24.V_ovov_pair[idx].data(),
                        pair_sz * sizeof(real_t));
            std::memcpy(h_T_flat.data() + off,
                        phase24.T_pair[idx].data(),
                        pair_sz * sizeof(real_t));
        }

        // Device-side source + offsets. 1-byte stub on empty slab so
        // the device pointers remain valid (kernel still guards via
        // active_pos).
        real_t* d_V_flat = nullptr;
        real_t* d_T_flat = nullptr;
        size_t* d_pair_src_off = nullptr;
        const size_t total_src_bytes =
            total_src_sz > 0 ? total_src_sz * sizeof(real_t) : 1;
        check_cuda_(cudaMalloc(&d_V_flat,    total_src_bytes), "cudaMalloc d_V_flat");
        check_cuda_(cudaMalloc(&d_T_flat,    total_src_bytes), "cudaMalloc d_T_flat");
        check_cuda_(cudaMalloc(&d_pair_src_off,
                               (N_pair_ + 1) * sizeof(size_t)),
                    "cudaMalloc d_pair_src_off");

        if (total_src_sz > 0) {
            check_cuda_(cudaMemcpy(d_V_flat, h_V_flat.data(),
                                   total_src_sz * sizeof(real_t),
                                   cudaMemcpyHostToDevice), "H2D V_flat");
            check_cuda_(cudaMemcpy(d_T_flat, h_T_flat.data(),
                                   total_src_sz * sizeof(real_t),
                                   cudaMemcpyHostToDevice), "H2D T_flat");
        }
        check_cuda_(cudaMemcpy(d_pair_src_off, pair_src_off_host.data(),
                               (N_pair_ + 1) * sizeof(size_t),
                               cudaMemcpyHostToDevice),
                    "H2D pair_src_off");

        // S11 Phase 2d — padded V_meta memsets removed. The packed
        // V_meta/T_meta buffers are fully written by pack_V_meta_kernel
        // (no inter-block padding inside the per-pair slot). V_stacked_oooo
        // is still per-pair packed and zeroed below to clear the small
        // inter-(k,l)-block region.
        if (bytes_v_oooo_packed > 0) {
            check_cuda_(cudaMemset(s.d_V_stacked_oooo_pad, 0, bytes_v_oooo_packed),
                        "memset V_stacked_oooo (packed)");
        }

        // Launch GPU pack. d_n_pno is borrowed from pgpu (uploaded once).
        const int* d_n_pno = pgpu.device_n_pno();
        if (d_n_pno == nullptr) {
            cudaFree(d_V_flat); cudaFree(d_T_flat); cudaFree(d_pair_src_off);
            delete p_; p_ = nullptr; active_ = false; return;
        }
        const dim3 block(16, 16);
        const dim3 grid(static_cast<unsigned>(N_pair_));
        // S11 Phase 2d — pack_V_meta_kernel writes only to packed sinks.
        // Inactive pairs (d_active_pos[idx] < 0) early-return inside the
        // kernel.
        // Stage D (D3a): sparse V_stacked_oooo needs the pgpu kl-slot list.
        const int*    d_kl_slot_pack   = s.voooo_sparse ? pgpu.device_kl_slot()     : nullptr;
        const size_t* d_slot_off_pack  = s.voooo_sparse ? pgpu.device_slot_offset() : nullptr;
        const int*    d_n_slots_pack   = s.voooo_sparse ? pgpu.device_n_slots()     : nullptr;
        const int     voooo_sparse_i   =
            (s.voooo_sparse && d_kl_slot_pack && d_slot_off_pack && d_n_slots_pack)
            ? 1 : 0;
        pack_V_meta_kernel<<<grid, block>>>(
            d_V_flat, d_T_flat, d_n_pno, d_pair_src_off,
            s.d_active_pos, s.d_v_oooo_off_packed,
            s.d_per_pair_meta_off,
            s.d_V_meta_T_packed, s.d_V_meta_TT_packed, s.d_T_meta_packed,
            s.d_V_stacked_oooo_pad,
            nocc_,
            voooo_sparse_i, d_kl_slot_pack, d_slot_off_pack, d_n_slots_pack);
        check_cuda_(cudaGetLastError(),
                    "pack_V_meta_kernel launch");

        // Free temporary device buffers (pack done).
        cudaFree(d_V_flat);
        cudaFree(d_T_flat);
        cudaFree(d_pair_src_off);
    }

    // ---- Pack + upload W_bare_ovov_i/j and W_bare_ovvo_i/j (packed only) ----
    // Source (host): phase24->W_ovov_i[idx][(a*nocc + k)*n + c]  = W(a, k, c) for a, c < n
    // Packed (device, row-major (n_ij × nn_ij), active pairs only):
    //   W_packed[off + a · nn_ij + k · n + c] = W(a, k, c)
    //
    // S11 Phase 2d (2026-05-19) — padded staging removed. Single pinned
    // host buffer sized packed_block_total_; OMP-parallel gather over
    // active pairs only.
    auto pack_W_bare = [&](const std::vector<std::vector<real_t>>& src,
                           real_t* d_dst_packed,
                           const char* label_packed)
    {
        if (packed_block_total_ == 0) return;
        PinnedHost h_W_packed(packed_block_total_, /*pinned=*/false);
        if (!h_W_packed.valid()) {
            check_cuda_(cudaErrorMemoryAllocation,
                        "cudaMallocHost h_W_packed for pack_W_bare");
            return;
        }
        #pragma omp parallel for schedule(static)
        for (long long idx = 0; idx < N_pair_; ++idx) {
            if (active_pos_[idx] < 0) continue;
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (idx >= static_cast<long long>(src.size())) continue;
            if (src[idx].empty()) continue;
            const real_t* W = src[idx].data();
            const size_t pair_off_packed = per_pair_block_off_[idx];
            const size_t nn_ij_szt = static_cast<size_t>(n)
                                   * static_cast<size_t>(nocc);
            for (int a = 0; a < n; ++a) {
                for (int k = 0; k < nocc; ++k) {
                    const real_t* row_in = W
                        + (static_cast<size_t>(a) * nocc + k) * n;
                    real_t* row_out_packed = h_W_packed.data()
                                           + pair_off_packed
                                           + static_cast<size_t>(a) * nn_ij_szt
                                           + static_cast<size_t>(k) * n;
                    std::memcpy(row_out_packed, row_in,
                                static_cast<size_t>(n) * sizeof(real_t));
                }
            }
        }
        check_cuda_(cudaMemcpy(d_dst_packed, h_W_packed.data(),
                               packed_block_total_ * sizeof(real_t),
                               cudaMemcpyHostToDevice),
                    label_packed);
    };
    pack_W_bare(phase24.W_ovov_i, s.d_W_bare_ovov_i_packed,
                "H2D W_bare_ovov_i packed");
    pack_W_bare(phase24.W_ovov_j, s.d_W_bare_ovov_j_packed,
                "H2D W_bare_ovov_j packed");
    pack_W_bare(phase24.W_ovvo_i, s.d_W_bare_ovvo_i_packed,
                "H2D W_bare_ovvo_i packed");
    pack_W_bare(phase24.W_ovvo_j, s.d_W_bare_ovvo_j_packed,
                "H2D W_bare_ovvo_j packed");

    // Step 6.9: V_stacked_oooo_pad is now written by pack_V_meta_kernel
    // above (fused with V_meta_T/V_meta_TT/T_meta packing). The separate
    // CPU pack + H2D scope that lived here is removed.
    // W_oooo[idx] is already (nocc²) flat per pair → direct upload (concat).
    {
        const size_t per_pair_w = static_cast<size_t>(nocc_) * nocc_;
        const size_t bytes_w = static_cast<size_t>(N_pair_) * per_pair_w * sizeof(real_t);
        // Pageable staging (one-time; avoids cudaMallocHost mmap-lock contention).
        PinnedHost h_W(static_cast<size_t>(N_pair_) * per_pair_w, /*pinned=*/false);
        if (!h_W.valid()) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        h_W.zero();
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
        // Free buffers on the same device they were allocated on.
        int dev = pgpu_ ? pgpu_->device_id() : 0;
        MultiGpuManager::DeviceGuard _guard(dev);
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
    if (!active_) return;
#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(pgpu_->device_id());
    Impl& s = *p_;
    // Open the stage timeline at the top so all 7 events are recorded
    // in order on every iter (inter_fock/oooo collapse to 0 ms here
    // since this path skips them).
    if (s.e_start) cudaEventRecord(s.e_start, /*stream=*/0);
    compute_async_phladder_only_();
    // No inter-Fock / oooo work on this path — collapse those buckets to
    // zero by recording the boundary events at the same point.
    if (s.e_after_F) cudaEventRecord(s.e_after_F, /*stream=*/0);
    if (s.e_after_O) cudaEventRecord(s.e_after_O, /*stream=*/0);
    compute_async_finalize_pipeline_();
#endif
}

void ResidGpu::compute_async_phladder_only_()
{
    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(pgpu_->device_id());
    Impl& s = *p_;
    const int N      = N_pair_;
    const int nocc   = nocc_;
    const int max_n  = max_n_;
    const int max_nn = s.max_nn;
    // Slab partition (from pgpu). For single-GPU operation slab = [0, N).
    const int ib     = pgpu_->pair_begin();
    const int ie     = pgpu_->pair_end();
    const int slab_n = ie - ib;
    // Helper: record the 3 internal phladder boundary events at the
    // current stream position. Used on every early-return path so the
    // per-iter event timeline stays well-defined (cudaEventElapsedTime
    // would otherwise pair a fresh e_start against a stale e_after_slice
    // from a previous iter).
    auto record_phladder_stubs = [&]() {
        if (s.e_after_slice) cudaEventRecord(s.e_after_slice, /*stream=*/0);
        if (s.e_after_W)     cudaEventRecord(s.e_after_W,     /*stream=*/0);
        if (s.e_after_R)     cudaEventRecord(s.e_after_R,     /*stream=*/0);
    };
    if (slab_n <= 0) { record_phladder_stubs(); return; }

    // Sanity: pgpu_ must still be in stacked mode with current pi_T_stack.
    const real_t* d_pi_T_stack    = pgpu_->device_pi_T_stack();
    const size_t* d_idx_offset    = pgpu_->device_idx_offset_pi_T();
    const int*    d_pair_lookup   = pgpu_->device_pair_lookup();
    const int*    d_n_pno         = pgpu_->device_n_pno();
    if (!d_pi_T_stack || !d_pair_lookup || !d_n_pno) {
        record_phladder_stubs();
        return;
    }
    // Stage D: sparse pi_T_stack reads — swap to sparse offsets + slot maps.
    const bool    pit_sparse  = pgpu_->pitstack_sparse();
    const size_t* d_idx_pit   = pit_sparse ? pgpu_->device_idx_offset_sparse()
                                           : d_idx_offset;
    const int*    d_n_slots   = pgpu_->device_n_slots();
    const int*    d_slot_jcol = pgpu_->device_slot_jcol();  // (k,j)
    const int*    d_slot_irow = pgpu_->device_slot_irow();  // (i,l)
    const int*    d_slot_icol = pgpu_->device_slot_icol();  // (k,i) j-side
    const int*    d_slot_jrow = pgpu_->device_slot_jrow();  // (j,l) j-side
    if (!d_idx_pit) { record_phladder_stubs(); return; }

    // ---- Stage 1: slice pi_T_stack into per-pair pad blocks. ----
    // Two kernel launches per side: one for I (= sij.i for i-side, sij.j for j-side)
    // and one for the outer J (= sij.j for i-side, sij.i for j-side).
    //
    // Step 6.6 fix: cap block at TILE=16 (256 threads) to keep within
    // CUDA's 1024 threads/block limit when max_n > 32 (e.g. Benzene cc-pVDZ).
    // Kernels iterate strided over the (a, d) range internally.
    //
    // Multi-GPU: grid_x = slab_n (kernel adds pair_begin to recover global idx).
    {
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(slab_n),
                  static_cast<unsigned>(nocc), 1);

        // S11 Phase 2d — slice kernels write only to packed buffers.
        // i-side: I = sij.i, J = sij.j
        slice_pi_N_T_for_I_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, s.d_I_i, d_pair_lookup,
            s.d_active_pos, s.d_per_pair_block_off, s.d_per_pair_stack_off,
            s.d_pi_N_i_packed, s.d_pi_T_i_packed,
            N, nocc, max_n, ib, pit_sparse ? 1 : 0, d_n_slots, d_slot_irow);
        slice_PI_outer_for_J_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, s.d_I_j, d_pair_lookup,
            s.d_active_pos, s.d_per_pair_block_off, s.d_per_pair_stack_off,
            s.d_PI_kj_stack_packed, s.d_PI_kj_TT_packed,
            N, nocc, max_n, ib, pit_sparse ? 1 : 0, d_n_slots, d_slot_jcol);

        // j-side: I = sij.j, J = sij.i — reads (j,l) row and (k,i) column, so
        // uses slot_jrow / slot_icol (NOT the i-side maps).
        slice_pi_N_T_for_I_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, s.d_I_j, d_pair_lookup,
            s.d_active_pos, s.d_per_pair_block_off, s.d_per_pair_stack_off,
            s.d_pi_N_j_packed, s.d_pi_T_j_packed,
            N, nocc, max_n, ib, pit_sparse ? 1 : 0, d_n_slots, d_slot_jrow);
        slice_PI_outer_for_J_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, s.d_I_i, d_pair_lookup,
            s.d_active_pos, s.d_per_pair_block_off, s.d_per_pair_stack_off,
            s.d_PI_ki_stack_packed, s.d_PI_ki_TT_packed,
            N, nocc, max_n, ib, pit_sparse ? 1 : 0, d_n_slots, d_slot_icol);

        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string("ResidGpu slice kernels failed: ")
                                     + cudaGetErrorString(e));
        }
    }
    if (s.e_after_slice) cudaEventRecord(s.e_after_slice, /*stream=*/0);

    const real_t neg_half = -0.5;
    const real_t plus_half =  0.5;
    const real_t one      =  1.0;
    const real_t neg_one  = -1.0;
    const real_t two      =  2.0;

    // S11 Phase 2d — outer stride / bytes_block_full constants removed.
    // Per-bucket strides are computed inside the bucket loops below.

    // ---- Stage 2: build W_block_i, W_block_i2, W_block_j, W_block_j2 ----
    //
    // S11 Phase 2c (2026-05-18) — bucket loop on packed buffers. Each
    // bucket b shares one n_ij = n_b value, so cublasDgemmStridedBatched
    // runs once per (math op, bucket) with batch = bucket_count_[b],
    // m/n/k = n_b/nn_b derived from n_b, and strides = per-pair packed
    // element counts (nn_b² / n_b·nn_b / nn_b·n_b / n_b² for meta /
    // block / stack / R). The host-side bucket tables (bucket_n_ij_ /
    // bucket_count_ / bucket_base_*_) are built once at construction
    // and reused across iters.
    //
    // Math (in row-major):
    //   W_block_i  = -0.5 · pi_T_i^T · V_meta_T  + 0.5 · pi_N_i · T_meta + W_bare_ovov_i
    //   W_block_i2 = -0.5 · pi_T_i^T · V_meta_TT + W_bare_ovvo_i
    //   (j-side mirrors with pi_T_j / pi_N_j and W_bare_ov{ov,vo}_j)
    //
    // The cuBLAS view of a row-major (R × C) matrix is column-major (C × R)
    // with leading dim R. For W_block_i += -0.5 · pi_T_i^T · V_meta_T (packed):
    //   row-major W_block(n_b × nn_b) += -0.5 · pi_T(nn_b × n_b)^T · V(nn_b × nn_b)
    //   col-major C_col(nn_b × n_b) = V_col(nn_b × nn_b) · pi_T_col^T(nn_b × n_b)
    //   m=nn_b, n=n_b, k=nn_b, TransA=N, TransB=T
    //   A=V_packed (lda=nn_b), B=pi_T_packed (ldb=n_b), C=W_block_packed (ldc=nn_b)
    //   Strides: A nn_b², B nn_b·n_b, C n_b·nn_b.

    const size_t nb_total = bucket_n_ij_.size();

    auto run_W_block_build = [&](real_t* d_W_block_packed,
                                 real_t* d_W_bare_packed,
                                 real_t* d_pi_N_packed,
                                 real_t* d_pi_T_packed,
                                 real_t* d_V_packed,    // V_meta_T or V_meta_TT
                                 bool   add_T_term)
    {
        // Init from bare W_block (entire packed range — slab-local so no
        // pair_begin offset). For zero-active devices this is a no-op.
        if (packed_block_total_ > 0) {
            check_cuda_(cudaMemcpyAsync(
                d_W_block_packed, d_W_bare_packed,
                packed_block_total_ * sizeof(real_t),
                cudaMemcpyDeviceToDevice, /*stream=*/0),
                "D2D W_bare_packed → W_block_packed");
        }
        for (size_t b = 0; b < nb_total; ++b) {
            const int cnt_b = bucket_count_[b];
            if (cnt_b == 0) continue;
            const int n_b  = bucket_n_ij_[b];
            const int nn_b = n_b * nocc;
            const long long stride_meta_b  = static_cast<long long>(nn_b) * nn_b;
            const long long stride_block_b = static_cast<long long>(n_b)  * nn_b;
            const long long stride_stack_b = static_cast<long long>(nn_b) * n_b;
            const size_t base_meta  = bucket_base_meta_[b];
            const size_t base_block = bucket_base_block_[b];
            const size_t base_stack = bucket_base_stack_[b];

            // W_block += -0.5 · pi_T^T · V (packed bucket batched).
            check_cublas_(cublasDgemmStridedBatched(
                s.cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                /*m=*/ nn_b, /*n=*/ n_b, /*k=*/ nn_b,
                &neg_half,
                d_V_packed    + base_meta,  nn_b, stride_meta_b,
                d_pi_T_packed + base_stack, n_b,  stride_stack_b,
                &one,
                d_W_block_packed + base_block, nn_b, stride_block_b,
                cnt_b), "W_block += -0.5·pi_T^T·V (bucket)");

            if (add_T_term) {
                // W_block += +0.5 · pi_N · T_meta (packed bucket).
                // row-major: W(n_b × nn_b) += +0.5 · pi_N(n_b × nn_b) · T(nn_b × nn_b)
                // col-major C_col(nn_b × n_b) = T_col(nn_b × nn_b) · pi_N_col(nn_b × n_b)
                // m=nn_b, n=n_b, k=nn_b, TransA=N, TransB=N.
                check_cublas_(cublasDgemmStridedBatched(
                    s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    /*m=*/ nn_b, /*n=*/ n_b, /*k=*/ nn_b,
                    &plus_half,
                    s.d_T_meta_packed + base_meta,  nn_b, stride_meta_b,
                    d_pi_N_packed     + base_block, nn_b, stride_block_b,
                    &one,
                    d_W_block_packed + base_block, nn_b, stride_block_b,
                    cnt_b), "W_block += +0.5·pi_N·T_meta (bucket)");
            }
        }
    };

    // i-side: W_block_i uses V_meta_T  (and adds T term)
    //         W_block_i2 uses V_meta_TT (no T term)
    run_W_block_build(s.d_W_block_i_packed,  s.d_W_bare_ovov_i_packed,
                      s.d_pi_N_i_packed,     s.d_pi_T_i_packed,
                      s.d_V_meta_T_packed,   /*add_T_term=*/true);
    run_W_block_build(s.d_W_block_i2_packed, s.d_W_bare_ovvo_i_packed,
                      s.d_pi_N_i_packed,     s.d_pi_T_i_packed,
                      s.d_V_meta_TT_packed,  /*add_T_term=*/false);
    // j-side mirror.
    run_W_block_build(s.d_W_block_j_packed,  s.d_W_bare_ovov_j_packed,
                      s.d_pi_N_j_packed,     s.d_pi_T_j_packed,
                      s.d_V_meta_T_packed,   /*add_T_term=*/true);
    run_W_block_build(s.d_W_block_j2_packed, s.d_W_bare_ovvo_j_packed,
                      s.d_pi_N_j_packed,     s.d_pi_T_j_packed,
                      s.d_V_meta_TT_packed,  /*add_T_term=*/false);
    if (s.e_after_W) cudaEventRecord(s.e_after_W, /*stream=*/0);

    // ---- Stage 3: contract into R_ph_packed ----
    // Zero the entire packed R buffer (slab-local; per_pair_R_off_ only
    // has slots for active slab pairs).
    if (packed_R_total_ > 0) {
        check_cuda_(cudaMemsetAsync(
            s.d_R_ph_packed, 0,
            packed_R_total_ * sizeof(real_t),
            /*stream=*/0),
            "memset R_ph_packed");
    }

    // S11 Phase 2c — 8 r_contract GEMMs in a bucket loop. Per bucket
    // all 8 ops share m=n_b, n=n_b, k=nn_b but vary in TransA/TransB +
    // operand orientation. Strides: stack n_b·nn_b (numerically same as
    // block but for col-major lda derivation we still pass lda = n_b
    // for stack and lda = nn_b for block, matching the row-major shape
    // of the packed buffers).
    for (size_t b = 0; b < nb_total; ++b) {
        const int cnt_b = bucket_count_[b];
        if (cnt_b == 0) continue;
        const int n_b  = bucket_n_ij_[b];
        const int nn_b = n_b * nocc;
        const long long stride_block_b = static_cast<long long>(n_b)  * nn_b;
        const long long stride_stack_b = static_cast<long long>(nn_b) * n_b;
        const long long stride_R_b     = static_cast<long long>(n_b)  * n_b;
        const size_t base_block = bucket_base_block_[b];
        const size_t base_stack = bucket_base_stack_[b];
        const size_t base_R     = bucket_base_R_[b];

        // Op 1: R_packed += 2 · W_block_i · PI_kj_stack
        //   row-major: C(n_b × n_b) = α · W(n_b × nn_b) · PI_stack(nn_b × n_b)
        //   col-major: C_col(n_b × n_b) = PI_stack_col(n_b × nn_b)
        //                               · W_col(nn_b × n_b)
        //   m=n_b, n=n_b, k=nn_b, TransA=N, TransB=N
        //   A=PI_kj_stack_packed (lda=n_b), B=W_block_i_packed (ldb=nn_b),
        //   C=R_packed (ldc=n_b).
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &two,
            s.d_PI_kj_stack_packed + base_stack, n_b,  stride_stack_b,
            s.d_W_block_i_packed   + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R += 2·W_i·PI_kj (bucket)");

        // Op 2: R_packed -= W_block_i2 · PI_kj_stack
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_PI_kj_stack_packed + base_stack, n_b,  stride_stack_b,
            s.d_W_block_i2_packed  + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= W_i2·PI_kj (bucket)");

        // Op 3: R_packed -= W_block_i · PI_kj_TT^T
        //   row-major C = W · K^T (W=W_block_i n_b×nn_b, K=PI_kj_TT n_b×nn_b)
        //   col-major C_col = K_col^T · W_col
        //   m=n_b, n=n_b, k=nn_b, TransA=T, TransB=N
        //   A=PI_kj_TT_packed (lda=nn_b), B=W_block_i_packed (ldb=nn_b),
        //   C=R_packed (ldc=n_b).
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_PI_kj_TT_packed   + base_block, nn_b, stride_block_b,
            s.d_W_block_i_packed  + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= W_i·PI_kj_TT^T (bucket)");

        // Op 4: R_packed -= PI_kj_TT · W_block_i2^T
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_W_block_i2_packed + base_block, nn_b, stride_block_b,
            s.d_PI_kj_TT_packed   + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= PI_kj_TT·W_i2^T (bucket)");

        // Op 5 (j-side): R_packed += 2 · PI_ki_stack^T · W_block_j^T
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_T,
            n_b, n_b, nn_b,
            &two,
            s.d_W_block_j_packed   + base_block, nn_b, stride_block_b,
            s.d_PI_ki_stack_packed + base_stack, n_b,  stride_stack_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R += 2·PI_ki^T·W_j^T (bucket)");

        // Op 6: R_packed -= PI_ki_stack^T · W_block_j2^T
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_T,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_W_block_j2_packed  + base_block, nn_b, stride_block_b,
            s.d_PI_ki_stack_packed + base_stack, n_b,  stride_stack_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= PI_ki^T·W_j2^T (bucket)");

        // Op 7: R_packed -= PI_ki_TT · W_block_j^T
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_W_block_j_packed + base_block, nn_b, stride_block_b,
            s.d_PI_ki_TT_packed  + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= PI_ki_TT·W_j^T (bucket)");

        // Op 8: R_packed -= W_block_j2 · PI_ki_stack
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n_b, n_b, nn_b,
            &neg_one,
            s.d_PI_ki_stack_packed + base_stack, n_b,  stride_stack_b,
            s.d_W_block_j2_packed  + base_block, nn_b, stride_block_b,
            &one,
            s.d_R_ph_packed + base_R, n_b, stride_R_b,
            cnt_b), "R -= W_j2·PI_ki (bucket)");
    }
    if (s.e_after_R) cudaEventRecord(s.e_after_R, /*stream=*/0);

#endif // !GANSU_CPU_ONLY
}

void ResidGpu::compute_async_finalize_pipeline_()
{
    if (!active_) return;
#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(pgpu_->device_id());
    Impl& s = *p_;
    // S11 Phase 2c — D2H from packed R buffer. packed_R_total_ counts
    // only active slab pairs on this device, so a single contiguous
    // copy covers the slab. Inactive pairs (n_pno = 0 or outside slab)
    // have no slot and are left as 0×0 matrices in finalize.
    if (packed_R_total_ > 0) {
        check_cuda_(cudaMemcpyAsync(
                        s.h_R_ph_packed, s.d_R_ph_packed,
                        packed_R_total_ * sizeof(real_t),
                        cudaMemcpyDeviceToHost,
                        /*stream=*/0),
                    "async D2H R_ph_packed (slab)");
    }
    if (s.e_after_D2H) cudaEventRecord(s.e_after_D2H, /*stream=*/0);
    check_cuda_(cudaEventRecord(s.completion_event, /*stream=*/0),
                "record completion event");
    s.async_in_flight = true;
    s.has_stage_record = (s.e_start && s.e_after_slice && s.e_after_W
                          && s.e_after_R && s.e_after_F && s.e_after_O
                          && s.e_after_D2H);
#endif
}

void ResidGpu::compute_async(const std::vector<real_t>& dF_ki_host)
{
    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(pgpu_->device_id());
    Impl& s = *p_;
    const int N     = N_pair_;
    const int nocc  = nocc_;
    const int max_n = max_n_;
    const int ib     = pgpu_->pair_begin();
    const int slab_n = pgpu_->pair_end() - ib;

    // Stage-timing marker: top of the per-iter pipeline. The tiny H2D
    // dF_ki + slice kernels both fall into the "slice" bucket below.
    if (s.e_start) cudaEventRecord(s.e_start, /*stream=*/0);

    // 1. Upload dF_ki (async on default stream) for this iter.
    //    NOTE: dF_ki is GLOBAL (nocc × nocc) — uploaded full on every device.
    if (static_cast<int>(dF_ki_host.size()) >= nocc * nocc) {
        check_cuda_(cudaMemcpyAsync(s.d_dF_ki, dF_ki_host.data(),
                                    static_cast<size_t>(nocc) * nocc * sizeof(real_t),
                                    cudaMemcpyHostToDevice, /*stream=*/0),
                    "H2D dF_ki");
    }

    // 2. Run the ph-ladder pipeline (slice + W_block + 8 contractions,
    //    accumulating into d_R_ph_packed), but NOT the trailing D2H yet.
    compute_async_phladder_only_();

    // 3. Inter-pair Fock i+j kernels accumulate into the SAME d_R_ph_packed
    //    at per-pair (n_ij × n_ij) packed offsets. They queue on the
    //    default stream, so they implicitly wait for ph-ladder kernels to
    //    finish before launching. Slab-restricted via pair_begin arg.
    if (slab_n <= 0) {
        // Close the inter_fock / oooo stage windows with zero-length spans
        // so the per-iter accumulator pattern stays well-defined.
        if (s.e_after_F) cudaEventRecord(s.e_after_F, /*stream=*/0);
        if (s.e_after_O) cudaEventRecord(s.e_after_O, /*stream=*/0);
        compute_async_finalize_pipeline_();
        return;
    }
    const real_t threshold = 1e-14;  // matches kFLMOThresh in dlpno_pair_data.cu
    {
        // Step 6.6 fix: cap block at TILE=16 (see Stage 1 comment).
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(slab_n), 1, 1);

        const real_t* d_pi_T_stack    = pgpu_->device_pi_T_stack();
        const size_t* d_idx_offset    = pgpu_->device_idx_offset_pi_T();
        const int*    d_pair_lookup   = pgpu_->device_pair_lookup();
        const int*    d_n_pno         = pgpu_->device_n_pno();
        // Stage D sparse pi_T_stack reads.
        const bool    pit_sparse  = pgpu_->pitstack_sparse();
        const size_t* d_idx_pit   = pit_sparse ? pgpu_->device_idx_offset_sparse()
                                               : d_idx_offset;
        const int*    d_n_slots   = pgpu_->device_n_slots();
        const int*    d_slot_jcol = pgpu_->device_slot_jcol();
        const int*    d_slot_irow = pgpu_->device_slot_irow();
        const int*    d_kl_slot   = pgpu_->device_kl_slot();
        const size_t* d_slot_off  = pgpu_->device_slot_offset();

        // S11 Phase 2c — both inter-pair Fock kernels write to packed R
        // at d_per_pair_R_off[idx]; padded R buffer is dead from here on.
        inter_pair_fock_i_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, d_pair_lookup,
            s.d_I_i, s.d_I_j, s.d_F_LMO, s.d_dF_ki,
            s.d_per_pair_R_off, s.d_R_ph_packed,
            N, nocc, max_n, threshold, ib,
            pit_sparse ? 1 : 0, d_n_slots, d_slot_jcol);
        inter_pair_fock_j_kernel<<<grid, block>>>(
            d_pi_T_stack, d_idx_pit, d_n_pno, d_pair_lookup,
            s.d_I_i, s.d_I_j, s.d_F_LMO, s.d_dF_ki,
            s.d_per_pair_R_off, s.d_R_ph_packed,
            N, nocc, max_n, threshold, ib,
            pit_sparse ? 1 : 0, d_n_slots, d_slot_irow);
        if (s.e_after_F) cudaEventRecord(s.e_after_F, /*stream=*/0);

        // Step 6.6 + 6.7: fused oooo ladder kernel — borrows pre-transposed
        // d_Y_pad_T from pgpu (set by the most recent rebuild_with_stack call).
        // Accumulates into d_R_ph_packed alongside ph-ladder + inter-pair Fock.
        //
        // Step 6.6 refactor: kernel uses dynamic shared memory of size
        // nocc² × sizeof(real_t) for the per-block W_eff[kl] cache. For
        // nocc=30 (hexamer) this is 7.2 KB, for nocc=60 (cholesterol class)
        // 28.8 KB — both well under the 48 KB default per-block limit.
        // Step 6.7: pre-transposed Y_pad_T eliminates the strided Y access
        // in Phase 1 (cache-line waste was ~max(1, max_n / cache_line/8) ×).
        const real_t* d_Y_pad_T = pgpu_->device_Y_pad_T();
        if (d_Y_pad_T) {
            const size_t shmem_bytes =
                static_cast<size_t>(nocc) * nocc * sizeof(real_t);
            // S11 Phase 2c — oooo ladder R write also switched to packed.
            oooo_lad_kernel<<<grid, block, shmem_bytes>>>(
                s.d_V_stacked_oooo_pad, s.d_W_oooo,
                d_pi_T_stack, d_Y_pad_T, d_idx_pit, d_n_pno,
                s.d_active_pos, s.d_v_oooo_off_packed,   // S10b
                s.d_per_pair_R_off, s.d_R_ph_packed,
                N, nocc, max_n, ib,
                pit_sparse ? 1 : 0, d_n_slots, d_kl_slot, d_slot_off,
                s.voooo_sparse ? 1 : 0);   // Stage D (D3a): sparse V_stacked_oooo
        }
        if (s.e_after_O) cudaEventRecord(s.e_after_O, /*stream=*/0);

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
    // Multi-GPU: caller pre-sizes R_ph_out to N_pair on the *first* slab call
    // each iter; subsequent slab calls must NOT clobber peer slabs' results.
    // We resize only if the vector hasn't been sized yet.
    if (R_ph_out.size() != static_cast<size_t>(N_pair_)) {
        R_ph_out.assign(static_cast<size_t>(N_pair_), RowMatXd());
    }

    if (!active_) return;

#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(pgpu_->device_id());
    Impl& s = *p_;
    const int ib  = pgpu_->pair_begin();
    const int ie  = pgpu_->pair_end();

    if (!s.async_in_flight) {
        // Defensive: caller invoked finalize without a matching compute_async.
        // Run the synchronous compute path to keep the contract intact.
        compute_async();
    }

    // ---- Wait on the async D2H to complete. ----
    check_cuda_(cudaEventSynchronize(s.completion_event),
                "wait completion event");
    s.async_in_flight = false;

    // ---- Sample per-stage timings (events were all recorded before the
    //      completion_event, so we can read elapsed times safely now). ----
    if (s.has_stage_record) {
        auto add_ms = [](double& acc, cudaEvent_t a, cudaEvent_t b) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, a, b) == cudaSuccess) {
                acc += static_cast<double>(ms) * 1e-3;  // ms → s
            }
        };
        add_ms(s.t_slice,      s.e_start,       s.e_after_slice);
        add_ms(s.t_W_block,    s.e_after_slice, s.e_after_W);
        add_ms(s.t_R_contract, s.e_after_W,     s.e_after_R);
        add_ms(s.t_inter_fock, s.e_after_R,     s.e_after_F);
        add_ms(s.t_oooo,       s.e_after_F,     s.e_after_O);
        add_ms(s.t_d2h,        s.e_after_O,     s.e_after_D2H);
        s.n_stage_iter += 1;
        s.has_stage_record = false;
    }

    // ---- Unpack packed h_R_ph_packed → host vec<RowMatXd>. Only fill
    //      the slab; other slabs are populated by peer ResidGpu
    //      instances on other devices.
    //
    //      S11 Phase 2c — packed source: per-pair n × n contiguous at
    //      element offset per_pair_R_off_[idx]. Inactive pairs (n_pno=0
    //      or outside this slab) get 0×0 matrices. Each active pair's
    //      n² block is already row-major n × n, so the unpack is a
    //      single memcpy per pair (no per-row striding).
    #pragma omp parallel for schedule(static)
    for (long long idx = ib; idx < ie; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) {
            R_ph_out[idx].resize(0, 0);
            continue;
        }
        R_ph_out[idx].resize(n, n);
        const real_t* src = s.h_R_ph_packed + per_pair_R_off_[idx];
        std::memcpy(R_ph_out[idx].data(), src,
                    static_cast<size_t>(n) * n * sizeof(real_t));
    }
#endif // !GANSU_CPU_ONLY
}

ResidStageTimes ResidGpu::get_stage_times() const
{
    ResidStageTimes out;
#ifndef GANSU_CPU_ONLY
    if (!active_ || !p_) return out;
    const Impl& s = *p_;
    out.slice      = s.t_slice;
    out.w_block    = s.t_W_block;
    out.r_contract = s.t_R_contract;
    out.inter_fock = s.t_inter_fock;
    out.oooo       = s.t_oooo;
    out.d2h        = s.t_d2h;
    out.n_iter     = s.n_stage_iter;
#endif
    return out;
}

void ResidGpu::reset_stage_times()
{
#ifndef GANSU_CPU_ONLY
    if (!active_ || !p_) return;
    Impl& s = *p_;
    s.t_slice      = 0.0;
    s.t_W_block    = 0.0;
    s.t_R_contract = 0.0;
    s.t_inter_fock = 0.0;
    s.t_oooo       = 0.0;
    s.t_d2h        = 0.0;
    s.n_stage_iter = 0;
#endif
}

} // namespace gansu
