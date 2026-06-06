/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_picache_gpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_manager.hpp"   // gpu::GPUHandle, gpu::gpu_available
#include "multi_gpu_manager.hpp"  // MultiGpuManager, DeviceGuard (multi-GPU dispatch)
#endif

namespace gansu {

#ifndef GANSU_CPU_ONLY

namespace {

inline void check_cuda_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("PiCacheGpu CUDA error in ")
                                 + what + ": " + cudaGetErrorString(e));
    }
}

inline void check_cublas_(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("PiCacheGpu cuBLAS error in ")
                                 + what + " status="
                                 + std::to_string(static_cast<int>(s)));
    }
}

// Step S1 + Step Z — GPU scatter kernel for compact barS_pad.
//
// Source layout (`d_barS_flat`, contiguous per active (i_ij, i_kl) pair):
//   d_barS_flat[d_block_src_off[bi] + r·n_kl + c] = barS[i_ij][i_kl](r, c)
// where bi = ai · N_act_kl + ak, with active_i_ij_[ai] = i_ij,
// active_i_kl_[ak] = i_kl. Only active pairs (n_pno > 0) are included.
//
// Destination (`d_barS_pad`, COMPACT [N_act_ij · N_act_kl · max_n²]):
//   d_barS_pad[(ai · N_act_kl + ak) · max_n² + r·max_n + c] = barS(r, c)
//
// Padding (rows/cols past n_ij / n_kl) is pre-zeroed by cudaMemset before
// the kernel.
__global__ void scatter_barS_kernel(
    const real_t* __restrict__ d_barS_flat,
    const size_t* __restrict__ d_block_src_off,
    const int*    __restrict__ d_n_pno_active_ij,  // n_pno indexed by ai
    const int*    __restrict__ d_n_pno_active_kl,  // n_pno indexed by ak
    real_t*       __restrict__ d_barS_pad,
    int N_act_kl, int max_n)
{
    const int ai = blockIdx.x;
    const int ak = blockIdx.y;

    const int n_ij = d_n_pno_active_ij[ai];
    const int n_kl = d_n_pno_active_kl[ak];
    if (n_ij == 0 || n_kl == 0) return;

    const size_t bi      = static_cast<size_t>(ai) * N_act_kl + ak;
    const size_t src_off = d_block_src_off[bi];
    const size_t dst_off = bi
                         * static_cast<size_t>(max_n)
                         * static_cast<size_t>(max_n);

    for (int r = threadIdx.y; r < n_ij; r += blockDim.y) {
        const size_t dst_row = dst_off + static_cast<size_t>(r) * max_n;
        const size_t src_row = src_off + static_cast<size_t>(r) * n_kl;
        for (int c = threadIdx.x; c < n_kl; c += blockDim.x) {
            d_barS_pad[dst_row + c] = d_barS_flat[src_row + c];
        }
    }
}

} // anonymous namespace

struct PiCacheGpu::Impl {
    int      N_pair    = 0;
    int      max_n     = 0;
    int      nocc      = 0;
    long long stride_pair = 0;        // max_n²
    long long stride_outer = 0;       // N_pair · max_n²

    // Persistent device buffers (Step 6.0)
    real_t* d_barS_pad = nullptr;     // [N_pair · N_pair · max_n²]
    real_t* d_Y_pad    = nullptr;     // [N_pair · max_n²]
    real_t* d_half_pad = nullptr;     // [N_pair · max_n²] scratch
    real_t* d_pi_pad   = nullptr;     // [N_pair · N_pair · max_n²]

    // Step 6.7: per-pair transpose of d_Y_pad. Layout:
    //   d_Y_pad_T[idx · max_n² + d · max_n + c] = Y[c, d]
    //                                           = d_Y_pad[idx · max_n² + c · max_n + d]
    // Used by oooo_lad_kernel Phase 1 to replace strided column-wise access
    // with sequential row-wise access (~3-50× cache traffic reduction for
    // the inner (dd, cc) double-loop, depending on max_n / n_pno).
    real_t* d_Y_pad_T  = nullptr;     // [N_pair · max_n²]

    // Pinned host buffers (Step 6.0)
    real_t* h_Y_pad  = nullptr;
    real_t* h_pi_pad = nullptr;

    // Step 6.1 stacked-mode device buffers (allocated only when stacked).
    int*     d_pair_lookup = nullptr;     // [nocc²] pair_lookup
    int*     d_setup_i     = nullptr;     // [N_pair] setups[idx].i
    int*     d_n_pno       = nullptr;     // [N_pair] pairs[idx].n_pno
    size_t*  d_idx_offset  = nullptr;     // [N_pair+1] cumulative pi_T_stack[idx] offset
    real_t*  d_pi_T_stack  = nullptr;     // [Σ n_pno²·nocc²] unpadded (FULL N_pair on device)
    real_t*  h_pi_T_stack  = nullptr;     // pinned mirror — Step S9 (2026-05-17):
                                          // sized to the slab [pair_begin_, pair_end_)
                                          // only, not the full N_pair sum. Saves
                                          // ~18 GB/device on cholesterol with
                                          // num_gpus=8 and lets concurrent
                                          // cudaMallocHost on N_gpus PiCacheGpu
                                          // instances avoid the kernel's
                                          // process-wide mmap_lock contention.
    size_t   pi_T_stack_total = 0;        // sum of n_pno²·nocc² (full N_pair, retained for legacy callers)
    // Step S9 — slab-only pinned mirror metadata. Both are in element counts
    // (not bytes). `h_pi_T_stack` is allocated to pi_T_slab_total elements.
    // To read pair i_ij ∈ [pair_begin_, pair_end_), index with
    // `h_pi_T_stack + (idx_offset_host[i_ij] - pi_T_slab_base_offset)`.
    size_t   pi_T_slab_base_offset = 0;   // Σ_{i<pair_begin_} n_pno_[i]²·nocc²
    size_t   pi_T_slab_total       = 0;   // Σ_{pair_begin_≤i<pair_end_} n_pno_[i]²·nocc²

    // DFpair GPU port — iter-invariant T_meta_dpair uploaded ONCE per iterate(),
    // slab-sized (Σ_{slab} nocc²·n²). Read at (idx_offset[idx] - slab_base) while
    // d_pi_T_stack is read at the FULL idx_offset[idx]. d_DF_scratch is a single
    // reused max_n² block (GEMM target before the per-pair D2H).
    real_t*  d_T_meta_dpair = nullptr;
    real_t*  d_DF_scratch   = nullptr;

    // Step Z stacked-mode buffers — orig idx → compact active position.
    int*     d_active_ij_pos = nullptr;   // [N_pair]
    int*     d_active_kl_pos = nullptr;   // [N_pair]

    // Step Z — ak → orig pair idx (for the transpose kernel writing to
    // d_Y_pad_T at the original pair offset). Allocated in the always-on
    // GPU init block (not stacked-mode-specific).
    int*     d_active_i_kl   = nullptr;   // [N_act_kl_]

    // Borrowed cuBLAS handle (thread_local from GPUHandle).
    cublasHandle_t cublas = nullptr;

    // [picache-PROF] per-iter rebuild() sub-phase accumulators (seconds),
    // dumped per-device in the dtor when GANSU_DLPNO_PICACHE_PROF=1. Splits
    // the MP2 rebuild() cost into pack_Y(+H2D) / tile GEMM / D2H / host
    // scatter so the next optimization targets the real dominant sub-phase.
    double t_packY = 0.0, t_gemm = 0.0, t_d2h = 0.0, t_scatter = 0.0;
    int    n_rebuild = 0;

    // [picache-gather] rebuild_needed() device map + compact buffers. Built
    // lazily on the first rebuild_needed() call (iter-invariant), reused
    // across iters. Indexed by ACTIVE i_ij position ai (0..N_act_ij-1):
    //   d_needed_ak[ai*max_needed + n] = active-kl position of the n-th needed
    //       column for active-ij ai (valid for n < needed_count[ai]).
    //   d_needed_count[ai]             = number of needed columns for ai.
    //   needed_ikl_host[ai][n]         = original i_kl (host, for the scatter).
    // d_pi_needed/h_pi_needed hold the gathered (tile_rows × max_needed)
    // max_n² blocks per tile.
    bool     needed_built  = false;
    int      max_needed    = 0;
    int*     d_needed_ak    = nullptr;   // [N_act_ij · max_needed]
    int*     d_needed_count = nullptr;   // [N_act_ij]
    real_t*  d_pi_needed    = nullptr;   // [tile_size · max_needed · max_n²]
    real_t*  h_pi_needed    = nullptr;   // pinned mirror of d_pi_needed
    std::vector<std::vector<int>> needed_ikl_host;  // [N_act_ij][count]

    void free_all() {
        if (d_barS_pad)    cudaFree(d_barS_pad);
        if (d_Y_pad)       cudaFree(d_Y_pad);
        if (d_Y_pad_T)     cudaFree(d_Y_pad_T);
        if (d_half_pad)    cudaFree(d_half_pad);
        if (d_pi_pad)      cudaFree(d_pi_pad);
        if (h_Y_pad)       cudaFreeHost(h_Y_pad);
        if (h_pi_pad)      cudaFreeHost(h_pi_pad);
        if (d_pair_lookup) cudaFree(d_pair_lookup);
        if (d_setup_i)     cudaFree(d_setup_i);
        if (d_n_pno)       cudaFree(d_n_pno);
        if (d_idx_offset)  cudaFree(d_idx_offset);
        if (d_pi_T_stack)  cudaFree(d_pi_T_stack);
        if (h_pi_T_stack)  cudaFreeHost(h_pi_T_stack);
        if (d_active_ij_pos) cudaFree(d_active_ij_pos);
        if (d_active_kl_pos) cudaFree(d_active_kl_pos);
        if (d_active_i_kl)   cudaFree(d_active_i_kl);
        if (d_needed_ak)     cudaFree(d_needed_ak);
        if (d_needed_count)  cudaFree(d_needed_count);
        if (d_pi_needed)     cudaFree(d_pi_needed);
        if (h_pi_needed)     cudaFreeHost(h_pi_needed);
        if (d_T_meta_dpair)  cudaFree(d_T_meta_dpair);
        if (d_DF_scratch)    cudaFree(d_DF_scratch);
        d_T_meta_dpair = nullptr;
        d_DF_scratch   = nullptr;
        d_barS_pad = d_Y_pad = d_Y_pad_T = d_half_pad = d_pi_pad = nullptr;
        h_Y_pad = h_pi_pad = nullptr;
        d_pair_lookup = d_setup_i = d_n_pno = nullptr;
        d_idx_offset = nullptr;
        d_pi_T_stack = nullptr;
        h_pi_T_stack = nullptr;
        d_active_ij_pos = d_active_kl_pos = nullptr;
        d_active_i_kl = nullptr;
        d_needed_ak = d_needed_count = nullptr;
        d_pi_needed = nullptr;
        h_pi_needed = nullptr;
    }
};

// ---------------------------------------------------------------------------
//  Step 6.1 — pack pi_pad (per-pair max_n × max_n padded canonical projection)
//  into pi_T_stack_unpadded[i_ij](a, (k·nocc + l)·n_ij + d) = π_{k,l}^{oriented}.
//
//  Layout invariants (kernel side):
//   pi_pad: row-major (max_n × max_n) per (i_ij, i_kl) block, contiguous.
//   pi_T_stack: per-i_ij block at offset d_idx_offset[i_ij], shape
//               (n_ij × nocc²·n_ij) row-major.
//
//  Block dims: (max_n_d, max_n_a). Each thread writes one element. Threads
//  with (a, d) outside the n_ij × n_ij valid sub-block return early.
//  Empty pairs (n_ij=0 or n_kl=0) write zero (zero-fill via cudaMemsetAsync
//  beforehand handles n_ij=0; n_kl=0 we explicitly write zero here).
// ---------------------------------------------------------------------------
// Step 6.7 — per-pair transpose kernel.
//   d_Y_pad_T[idx, r, c] = d_Y_pad[idx, c, r]    (matrix transpose per pair)
// Launched as <<<N_pair, dim3(16, 16)>>> with strided thread loops for
// max_n > 16. Negligible cost: O(N_pair · max_n²) bytes, ~1 ms even at
// TEOS scale. The point is to convert the strided column-wise reads of
// d_Y_pad in `oooo_lad_kernel` Phase 1 into sequential row-wise reads of
// d_Y_pad_T, restoring L1 cache-line efficiency.
//
// Step Z: d_Y_pad is now COMPACT (indexed by active position ak), but
// d_Y_pad_T must remain FULL (N_pair × max_n²) because oooo_lad_kernel
// in ResidGpu indexes it by orig pair idx (i_ij). This kernel reads
// d_Y_pad at the compact position d_active_i_kl[ak], and writes the
// transpose to d_Y_pad_T at the ORIGINAL pair idx. Inactive pairs are
// left at zero (pre-set by cudaMemset in the constructor).
__global__ void transpose_Y_pad_kernel(
    const real_t* __restrict__ d_Y_pad,           // compact: indexed by ak
    real_t*       __restrict__ d_Y_pad_T,         // FULL: indexed by orig
    const int*    __restrict__ d_active_i_kl,     // ak → orig
    int N_act_kl, int max_n)
{
    const int ak = blockIdx.x;
    if (ak >= N_act_kl) return;
    const int orig_idx = d_active_i_kl[ak];
    const size_t src_off = static_cast<size_t>(ak)
                         * static_cast<size_t>(max_n)
                         * static_cast<size_t>(max_n);
    const size_t dst_off = static_cast<size_t>(orig_idx)
                         * static_cast<size_t>(max_n)
                         * static_cast<size_t>(max_n);
    for (int r = threadIdx.y; r < max_n; r += blockDim.y) {
        for (int c = threadIdx.x; c < max_n; c += blockDim.x) {
            d_Y_pad_T[dst_off + static_cast<size_t>(r) * max_n + c] =
                d_Y_pad[src_off + static_cast<size_t>(c) * max_n + r];
        }
    }
}

// Step Z — d_pi_pad is now COMPACT + TILED:
//   d_pi_pad[(ai_in_tile · N_act_kl + ak) · max_n² + r·max_n + c]
// with ai_in_tile = active_ij_pos_[i_ij] - tile_start and
// ak = active_kl_pos_[idx_kl]. Empty (i_ij, idx_kl) pairs (active_*_pos = -1)
// contribute zero to d_pi_T_stack. The kernel reads d_active_ij_pos /
// d_active_kl_pos to translate orig indices into compact-tile coordinates.
__global__ void pack_pi_T_stack_kernel(
    const real_t* __restrict__ d_pi_pad,            // tile compact
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_setup_i,
    const int*    __restrict__ d_n_pno,
    const size_t* __restrict__ d_idx_offset,
    const int*    __restrict__ d_active_ij_pos,     // orig i_ij → active ai (-1)
    const int*    __restrict__ d_active_kl_pos,     // orig i_kl → active ak (-1)
    real_t*       __restrict__ d_pi_T_stack,
    int N_pair, int nocc, int max_n,
    int N_act_kl, int tile_start, int tile_end)
{
    const int i_ij = blockIdx.x;
    const int kl   = blockIdx.y;          // = k * nocc + l
    if (i_ij >= N_pair || kl >= nocc * nocc) return;

    const int n_ij = d_n_pno[i_ij];
    if (n_ij == 0) return;

    // Tile range check: skip pairs not in this tile.
    const int ai = d_active_ij_pos[i_ij];
    if (ai < tile_start || ai >= tile_end) return;
    const int ai_in_tile = ai - tile_start;

    const int idx_kl = d_pair_lookup[kl];
    const int n_kl   = d_n_pno[idx_kl];
    const int k      = kl / nocc;
    const int s_i_kl = d_setup_i[idx_kl];
    const int ak     = d_active_kl_pos[idx_kl];  // -1 if n_kl == 0

    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            real_t v = real_t(0);
            if (n_kl > 0 && ak >= 0) {
                const real_t* src = d_pi_pad
                    + (static_cast<size_t>(ai_in_tile)
                       * static_cast<size_t>(N_act_kl)
                       + static_cast<size_t>(ak))
                    * static_cast<size_t>(max_n) * static_cast<size_t>(max_n);
                if (s_i_kl != k) {
                    v = src[static_cast<size_t>(d) * max_n + static_cast<size_t>(a)];
                } else {
                    v = src[static_cast<size_t>(a) * max_n + static_cast<size_t>(d)];
                }
            }

            real_t* dst = d_pi_T_stack + d_idx_offset[i_ij]
                        + static_cast<size_t>(a)
                        * static_cast<size_t>(nocc) * static_cast<size_t>(nocc)
                        * static_cast<size_t>(n_ij)
                        + static_cast<size_t>(kl) * static_cast<size_t>(n_ij)
                        + static_cast<size_t>(d);
            *dst = v;
        }
    }
}

// [picache-gather] Compact the needed columns of the tile's d_pi_pad into
// d_pi_needed so only ~2·nocc columns/row are D2H'd (vs N_act_kl). Each block
// copies one (ai_in_tile, n) max_n² padded projection block verbatim — the
// orientation/transpose stays in the host residual, so this is a pure byte
// copy and is bit-exact w.r.t. the full rebuild() scatter.
//   src: d_pi_pad   [(ai_in_tile·N_act_kl + ak)         · max_n²]
//   dst: d_pi_needed[(ai_in_tile·max_needed + n)        · max_n²]
// where ak = d_needed_ak[ai·max_needed + n], ai = ai_in_tile + tile_start.
// Threads with n >= d_needed_count[ai] do nothing (those dst slots are never
// read by the host scatter).
__global__ void gather_needed_kernel(
    const real_t* __restrict__ d_pi_pad,
    real_t*       __restrict__ d_pi_needed,
    const int*    __restrict__ d_needed_ak,
    const int*    __restrict__ d_needed_count,
    int tile_start, int tile_rows, int max_needed,
    long long stride_pair /* = max_n² */, int N_act_kl)
{
    const int slot = blockIdx.x;                 // = ai_in_tile · max_needed + n
    const int ai_in_tile = slot / max_needed;
    const int n          = slot % max_needed;
    if (ai_in_tile >= tile_rows) return;
    const int ai = ai_in_tile + tile_start;
    if (n >= d_needed_count[ai]) return;
    const int ak = d_needed_ak[static_cast<size_t>(ai) * max_needed + n];
    if (ak < 0) return;

    const real_t* src = d_pi_pad
        + (static_cast<size_t>(ai_in_tile) * N_act_kl + ak) * stride_pair;
    real_t* dst = d_pi_needed + static_cast<size_t>(slot) * stride_pair;
    for (long long e = threadIdx.x; e < stride_pair; e += blockDim.x)
        dst[e] = src[e];
}

#else  // GANSU_CPU_ONLY

struct PiCacheGpu::Impl {
    // Empty stub — CPU fallback path uses members on the outer class.
};

#endif // GANSU_CPU_ONLY

// ---------------------------------------------------------------------------
//  Constructor
// ---------------------------------------------------------------------------
PiCacheGpu::PiCacheGpu(const std::vector<std::vector<RowMatXd>>& barS_cache,
                       const std::vector<int>& n_pno_per_pair,
                       int max_n,
                       const std::vector<int>* pair_lookup,
                       const std::vector<int>* setup_i_per_pair,
                       int nocc,
                       int pair_begin,
                       int pair_end,
                       int device_id)
    : n_pno_(n_pno_per_pair),
      N_pair_(static_cast<int>(n_pno_per_pair.size())),
      max_n_(max_n),
      nocc_(nocc)
{
    // Step S8 profiling: timestamp the constructor stages so we can attribute
    // the cholesterol-class ~9 s/device cost to its components. Only the
    // success path is timed; early-return paths skip the dump. Stages:
    //   cpu_prep    : active list build + position maps + barS-cache refs
    //   mem_probe   : cudaMemGetInfo + tile_size_ decision
    //   alloc_base  : Step Z base buffer cudaMalloc + d_Y_pad_T cudaMemset
    //   alloc_stack : stacked-mode cudaMalloc + slab-range cudaMallocHost
    //                 h_pi_T_stack (Step S9: was full N_pair = ~21 GB, now
    //                 slab-only ~2.6 GB at cholesterol/num_gpus=8) + 4 small
    //                 H2D copies (pair_lookup / setup_i / n_pno / idx_offset
    //                 / active_*_pos)
    //   cublas      : MultiGpuManager cuBLAS handle resolution
    //   barS_h2d    : host barS pack + pinned alloc + ~2.5 GB H2D + scatter
    //                 kernel + cudaDeviceSynchronize
    // [[maybe_unused]] in CPU-only builds where the printf is compiled out.
    [[maybe_unused]] const auto t_ctor_0 = std::chrono::steady_clock::now();

    barS_cache_ref_ = &barS_cache;
    if (pair_lookup)        pair_lookup_       = *pair_lookup;
    if (setup_i_per_pair)   setup_i_per_pair_  = *setup_i_per_pair;

    // Slab info (output rows). pair_end<0 means use full range.
    pair_begin_ = (pair_begin < 0) ? 0 : pair_begin;
    pair_end_   = (pair_end   < 0) ? N_pair_ : pair_end;
    if (pair_begin_ > N_pair_) pair_begin_ = N_pair_;
    if (pair_end_   > N_pair_) pair_end_   = N_pair_;
    if (pair_end_   < pair_begin_) pair_end_ = pair_begin_;
    device_id_ = device_id;

    // CPU-path acceleration: snapshot the indices with n_pno > 0 so
    // rebuild_cpu_ can iterate only the active sub-grid (1952² instead of
    // 5886² on cholesterol). n_pno_ is fixed for the lifetime of this
    // instance (the iter loop in iterate_dlpno_ccsd_t2 / iterate_lmp2 does
    // not re-truncate PNOs between iters — re-truncation is a separate
    // SC-PNO round that constructs a new PiCacheGpu).
    active_i_kl_.reserve(static_cast<size_t>(N_pair_));
    for (int k = 0; k < N_pair_; ++k) {
        if (n_pno_[k] > 0) active_i_kl_.push_back(k);
    }
    active_i_ij_.reserve(static_cast<size_t>(pair_end_ - pair_begin_));
    for (int k = pair_begin_; k < pair_end_; ++k) {
        if (n_pno_[k] > 0) active_i_ij_.push_back(k);
    }

    // Step Z — compact storage maps. active_*_pos_[orig_idx] gives the
    // position of orig_idx in active_i_*_, or -1 if the pair is empty /
    // outside the slab.
    N_act_ij_ = static_cast<int>(active_i_ij_.size());
    N_act_kl_ = static_cast<int>(active_i_kl_.size());
    active_kl_pos_.assign(static_cast<size_t>(N_pair_), -1);
    for (int ak = 0; ak < N_act_kl_; ++ak) {
        active_kl_pos_[active_i_kl_[ak]] = ak;
    }
    active_ij_pos_.assign(static_cast<size_t>(N_pair_), -1);
    for (int ai = 0; ai < N_act_ij_; ++ai) {
        active_ij_pos_[active_i_ij_[ai]] = ai;
    }

    [[maybe_unused]] const auto t_cpu_prep = std::chrono::steady_clock::now();

#ifndef GANSU_CPU_ONLY
    // Decide whether to take the GPU path.
    if (!gpu::gpu_available() || N_pair_ == 0 || max_n_ == 0) {
        active_ = false;
        return;
    }

    // For multi-GPU pair partitioning, route allocations to the requested
    // device (caller is expected to have set device or pass device_id; we
    // ensure correctness with DeviceGuard here so destructor and rebuild()
    // calls land on the same device).
    MultiGpuManager::DeviceGuard _guard(device_id_);

    p_ = new Impl();
    Impl& s = *p_;
    s.N_pair      = N_pair_;
    s.max_n       = max_n_;
    s.nocc        = nocc_;
    s.stride_pair = static_cast<long long>(max_n_) * max_n_;
    // Note: stride_outer no longer used by Step Z compact path but kept
    // populated for any legacy reader. Compact strides are below.
    s.stride_outer = s.stride_pair * static_cast<long long>(N_act_kl_);

    // ----------------------------------------------------------------------
    // Step Z — compact + tiled buffer sizing.
    //
    //   d_barS_pad : compact [N_act_ij × N_act_kl × max_n²] (~20 GB cholesterol)
    //   d_Y_pad    : compact [N_act_kl × max_n²]  (~tiny, internal use only)
    //   d_Y_pad_T  : FULL    [N_pair    × max_n²]  (~32 MB cholesterol; oooo_lad
    //                in ResidGpu reads it by orig pair idx, so must be full)
    //   d_half_pad : per-output-row scratch [N_act_kl × max_n²]
    //   d_pi_pad   : tile [tile_size × N_act_kl × max_n²] (~few GB)
    //   d_pi_T_stack: full sparse [Σ n_pno²·nocc²]  (unchanged)
    //
    // The full N_pair²·max_n² padded layout (~175 GB at cholesterol) is
    // structurally infeasible; even compact (~20 GB each) two of them +
    // pi_T (~21 GB) exceeds 55 GB free. Tiling d_pi_pad in the active i_ij
    // dimension is what brings total need below the device budget.
    // ----------------------------------------------------------------------
    const size_t bytes_barS_compact =
        static_cast<size_t>(N_act_ij_) * static_cast<size_t>(N_act_kl_)
        * static_cast<size_t>(max_n_) * static_cast<size_t>(max_n_)
        * sizeof(real_t);
    const size_t bytes_Y_compact =
        static_cast<size_t>(N_act_kl_) * static_cast<size_t>(max_n_)
        * static_cast<size_t>(max_n_) * sizeof(real_t);
    const size_t bytes_Y_T_full =
        static_cast<size_t>(N_pair_) * static_cast<size_t>(max_n_)
        * static_cast<size_t>(max_n_) * sizeof(real_t);
    const size_t per_pi_row_bytes = bytes_Y_compact;  // = N_act_kl × max_n² × 8

    // Step 6.1 stacked-mode budget (only when pair_lookup + setup_i + nocc>0).
    const bool want_stacked = pair_lookup && setup_i_per_pair && nocc_ > 0
                              && !pair_lookup_.empty()
                              && !setup_i_per_pair_.empty()
                              && static_cast<int>(pair_lookup_.size()) == nocc_ * nocc_
                              && static_cast<int>(setup_i_per_pair_.size()) == N_pair_;
    size_t pi_T_total = 0;
    // Step S9 — also accumulate slab-only counts. `pi_T_slab_base_offset` is
    // the element offset of pair `pair_begin_` within the full pi_T_stack
    // (i.e. the size of the prefix before this device's slab). `pi_T_slab_total`
    // is the slab's own elem count. These let us size `h_pi_T_stack` to the
    // slab only — the device side `d_pi_T_stack` keeps the full-N_pair layout
    // so all the kernels that index via `d_idx_offset` work unchanged.
    size_t pi_T_slab_base_off = 0;
    size_t pi_T_slab_count    = 0;
    if (want_stacked) {
        for (int i = 0; i < N_pair_; ++i) {
            const size_t n = static_cast<size_t>(n_pno_[i]);
            const size_t add = n * n
                             * static_cast<size_t>(nocc_) * static_cast<size_t>(nocc_);
            if (i < pair_begin_)      pi_T_slab_base_off += add;
            else if (i < pair_end_)   pi_T_slab_count    += add;
            pi_T_total += add;
        }
    }
    const size_t bytes_pi_T      = pi_T_total      * sizeof(real_t);
    const size_t bytes_pi_T_slab = pi_T_slab_count * sizeof(real_t);

    // Probe free memory and choose tile_size_ so d_pi_pad fits.
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        // Fixed needs (everything except d_pi_pad):
        //   d_barS_pad + d_Y_pad + d_Y_pad_T (full) + d_half_pad + d_pi_T_stack + margin
        const size_t margin = (size_t)256 * 1024 * 1024;  // 256 MB
        const size_t fixed_need = bytes_barS_compact
                                + bytes_Y_compact      // d_Y_pad
                                + bytes_Y_T_full       // d_Y_pad_T (FULL)
                                + bytes_Y_compact      // d_half_pad
                                + bytes_pi_T
                                + margin;
        if (fixed_need > free_b || N_act_ij_ == 0) {
            // Even without d_pi_pad we don't fit, or no active pairs to compute.
            delete p_; p_ = nullptr; active_ = false; return;
        }
        const size_t avail_for_pi = free_b - fixed_need;
        if (per_pi_row_bytes == 0) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        // Tile up to N_act_ij_ rows; cap at avail_for_pi.
        size_t max_tile = avail_for_pi / per_pi_row_bytes;
        if (max_tile == 0) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        if (max_tile > static_cast<size_t>(N_act_ij_)) {
            max_tile = static_cast<size_t>(N_act_ij_);
        }
        // Cap tile size at a reasonable upper bound to avoid huge transient
        // buffers (~8 GB) — beyond that, returns from larger tile diminish.
        const size_t cap_tile_bytes = (size_t)8 * 1024 * 1024 * 1024;  // 8 GB
        if (max_tile * per_pi_row_bytes > cap_tile_bytes) {
            max_tile = cap_tile_bytes / per_pi_row_bytes;
            if (max_tile == 0) max_tile = 1;
        }
        tile_size_ = static_cast<int>(max_tile);
    }
    const size_t bytes_pi_tile =
        static_cast<size_t>(tile_size_) * per_pi_row_bytes;

    const auto t_mem_probe = std::chrono::steady_clock::now();

    try {
        check_cuda_(cudaMalloc(&s.d_barS_pad, bytes_barS_compact),
                    "cudaMalloc d_barS_pad (compact)");
        check_cuda_(cudaMalloc(&s.d_pi_pad,   bytes_pi_tile),
                    "cudaMalloc d_pi_pad (tile)");
        check_cuda_(cudaMalloc(&s.d_Y_pad,    bytes_Y_compact),
                    "cudaMalloc d_Y_pad (compact)");
        check_cuda_(cudaMalloc(&s.d_Y_pad_T,  bytes_Y_T_full),
                    "cudaMalloc d_Y_pad_T (full N_pair)");
        // Step Z: zero-init d_Y_pad_T (full); transpose kernel only writes
        // active positions, so inactive pair slots stay at zero.
        check_cuda_(cudaMemset(s.d_Y_pad_T, 0, bytes_Y_T_full),
                    "cudaMemset d_Y_pad_T");
        check_cuda_(cudaMalloc(&s.d_half_pad, bytes_Y_compact),
                    "cudaMalloc d_half_pad (compact)");

        check_cuda_(cudaMallocHost(&s.h_Y_pad,  bytes_Y_compact),
                    "cudaMallocHost h_Y_pad (compact)");

        // Step Z — persistent active_i_kl map for the transpose kernel.
        if (N_act_kl_ > 0) {
            check_cuda_(cudaMalloc(&s.d_active_i_kl,
                            static_cast<size_t>(N_act_kl_) * sizeof(int)),
                        "cudaMalloc d_active_i_kl");
            check_cuda_(cudaMemcpy(s.d_active_i_kl, active_i_kl_.data(),
                            static_cast<size_t>(N_act_kl_) * sizeof(int),
                            cudaMemcpyHostToDevice),
                        "H2D d_active_i_kl");
        }
        // Step S1: defer the 692 MB (hexamer) / 60 GB (cholesterol) pinned
        // allocation of h_pi_pad until the first call to download_pi_cache_,
        // which is bypassed entirely by Step 6's stacked-mode +
        // skip_pi_cache_host=true path. cudaMallocHost of large buffers can
        // dominate setup_pgpu (~200-500 ms for 692 MB pinning on Linux), and
        // is pure waste when the CCSD T2 iter never asks for the host pi
        // cache. download_pi_cache_ allocates on demand.
    } catch (const std::exception&) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    const auto t_alloc_base = std::chrono::steady_clock::now();

    // Step 6.1 stacked-mode allocations + uploads.
    if (want_stacked) {
        try {
            const size_t n_lookup = static_cast<size_t>(nocc_)
                                  * static_cast<size_t>(nocc_);
            check_cuda_(cudaMalloc(&s.d_pair_lookup,
                                   n_lookup * sizeof(int)),
                        "cudaMalloc d_pair_lookup");
            check_cuda_(cudaMalloc(&s.d_setup_i,
                                   static_cast<size_t>(N_pair_) * sizeof(int)),
                        "cudaMalloc d_setup_i");
            check_cuda_(cudaMalloc(&s.d_n_pno,
                                   static_cast<size_t>(N_pair_) * sizeof(int)),
                        "cudaMalloc d_n_pno");
            check_cuda_(cudaMalloc(&s.d_idx_offset,
                                   static_cast<size_t>(N_pair_ + 1) * sizeof(size_t)),
                        "cudaMalloc d_idx_offset");
            check_cuda_(cudaMalloc(&s.d_pi_T_stack, bytes_pi_T),
                        "cudaMalloc d_pi_T_stack");
            // Step S9 — size the pinned host mirror to the slab range only.
            // For a single-GPU run this is the same as bytes_pi_T; for
            // num_gpus=8 it is ~1/8 the size, which cuts the per-device
            // pinning time from ~8 s to ~1 s and (more importantly) lets
            // 8 concurrent cudaMallocHost calls fight over the kernel
            // mmap_lock for less wall time. Without the slab-only sizing,
            // cholesterol CCSD T2 setup_pgpu was 63 s wall with all 8
            // devices stuck in alloc_stack at the same time (per-stage
            // profile, 2026-05-17).
            if (bytes_pi_T_slab > 0) {
                check_cuda_(cudaMallocHost(&s.h_pi_T_stack, bytes_pi_T_slab),
                            "cudaMallocHost h_pi_T_stack (slab)");
            }
            s.pi_T_slab_base_offset = pi_T_slab_base_off;
            s.pi_T_slab_total       = pi_T_slab_count;

            check_cuda_(cudaMemcpy(s.d_pair_lookup, pair_lookup_.data(),
                                   n_lookup * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D pair_lookup");
            check_cuda_(cudaMemcpy(s.d_setup_i, setup_i_per_pair_.data(),
                                   static_cast<size_t>(N_pair_) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D setup_i");
            check_cuda_(cudaMemcpy(s.d_n_pno, n_pno_.data(),
                                   static_cast<size_t>(N_pair_) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D n_pno");

            std::vector<size_t> idx_offset_host(
                static_cast<size_t>(N_pair_ + 1), 0);
            for (int i = 0; i < N_pair_; ++i) {
                const size_t n = static_cast<size_t>(n_pno_[i]);
                idx_offset_host[i + 1] = idx_offset_host[i]
                    + n * n
                    * static_cast<size_t>(nocc_)
                    * static_cast<size_t>(nocc_);
            }
            check_cuda_(cudaMemcpy(s.d_idx_offset, idx_offset_host.data(),
                                   static_cast<size_t>(N_pair_ + 1) * sizeof(size_t),
                                   cudaMemcpyHostToDevice),
                        "H2D idx_offset");

            // Step Z — active position maps on device for the pack kernel.
            check_cuda_(cudaMalloc(&s.d_active_ij_pos,
                                   static_cast<size_t>(N_pair_) * sizeof(int)),
                        "cudaMalloc d_active_ij_pos");
            check_cuda_(cudaMalloc(&s.d_active_kl_pos,
                                   static_cast<size_t>(N_pair_) * sizeof(int)),
                        "cudaMalloc d_active_kl_pos");
            check_cuda_(cudaMemcpy(s.d_active_ij_pos, active_ij_pos_.data(),
                                   static_cast<size_t>(N_pair_) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D d_active_ij_pos");
            check_cuda_(cudaMemcpy(s.d_active_kl_pos, active_kl_pos_.data(),
                                   static_cast<size_t>(N_pair_) * sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "H2D d_active_kl_pos");

            s.pi_T_stack_total = pi_T_total;
            stacked_ = true;
        } catch (const std::exception&) {
            // Free stacked-mode buffers; keep Step 6.0 buffers — caller can
            // still use rebuild() and rebuild_with_stack() falls back to CPU
            // pi_T_stack assembly internally.
            if (s.d_pair_lookup)    { cudaFree(s.d_pair_lookup);    s.d_pair_lookup    = nullptr; }
            if (s.d_setup_i)        { cudaFree(s.d_setup_i);        s.d_setup_i        = nullptr; }
            if (s.d_n_pno)          { cudaFree(s.d_n_pno);          s.d_n_pno          = nullptr; }
            if (s.d_idx_offset)     { cudaFree(s.d_idx_offset);     s.d_idx_offset     = nullptr; }
            if (s.d_pi_T_stack)     { cudaFree(s.d_pi_T_stack);     s.d_pi_T_stack     = nullptr; }
            if (s.h_pi_T_stack)     { cudaFreeHost(s.h_pi_T_stack); s.h_pi_T_stack     = nullptr; }
            if (s.d_active_ij_pos)  { cudaFree(s.d_active_ij_pos);  s.d_active_ij_pos  = nullptr; }
            if (s.d_active_kl_pos)  { cudaFree(s.d_active_kl_pos);  s.d_active_kl_pos  = nullptr; }
            stacked_ = false;
        }
    }

    const auto t_alloc_stack = std::chrono::steady_clock::now();

    // Multi-GPU: use per-device cuBLAS handle from MultiGpuManager. For the
    // single-GPU path (device_id_=0, MGM not initialised) fall back to the
    // thread-local handle from gpu::GPUHandle.
    s.cublas = nullptr;
    {
        auto& mgm = MultiGpuManager::instance();
        if (mgm.num_devices() > device_id_) {
            s.cublas = mgm.cublas(device_id_);
        }
        if (s.cublas == nullptr) s.cublas = gpu::GPUHandle::cublas();
    }
    if (s.cublas == nullptr) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    const auto t_cublas = std::chrono::steady_clock::now();

    // -------- Step S1: GPU scatter from compact host pack.
    //
    // Replaces the 692 MB host-side padded buffer (`h_barS_pad`) + the
    // N_pair² × memcpy loop with a flat pack of active blocks only (~30
    // MB at hexamer, ~25× smaller) + pinned H2D + a single device
    // scatter kernel. Savings sources:
    //   - 692 MB std::vector zero-init on host (~14 ms cold)        gone
    //   - N_pair² small-block memcpy loop (~200-300 ms cache-cold)  gone
    //   - 692 MB pageable H2D (~170 ms at PCIe4)                    → 30 MB pinned (~3 ms)
    //   - Padding region (where blocks don't reach max_n) filled by cudaMemset on
    //     device (~ms at H200 bandwidth) instead of host zero-init.
    //
    // Layout: d_barS_pad[(i_ij·N_pair + i_kl)·max_n² + r·max_n + c]
    //         row-major within each (max_n × max_n) padded block.
    {
        const int N_act_ij = static_cast<int>(active_i_ij_.size());
        const int N_act_kl = static_cast<int>(active_i_kl_.size());

        if (N_act_ij > 0 && N_act_kl > 0) {
            const size_t n_blocks =
                static_cast<size_t>(N_act_ij) * static_cast<size_t>(N_act_kl);

            // Cumulative source offsets for each active (i_ij, i_kl) block.
            std::vector<size_t> block_src_off(n_blocks + 1, 0);
            for (int ai = 0; ai < N_act_ij; ++ai) {
                const int n_ij = n_pno_[active_i_ij_[ai]];
                for (int ak = 0; ak < N_act_kl; ++ak) {
                    const int n_kl = n_pno_[active_i_kl_[ak]];
                    const size_t bi =
                        static_cast<size_t>(ai) * N_act_kl + ak;
                    block_src_off[bi + 1] = block_src_off[bi]
                        + static_cast<size_t>(n_ij) * n_kl;
                }
            }
            const size_t flat_size = block_src_off.back();

            // Pinned host flat buffer (fast DMA). Zero-init so any
            // defensively-empty barS_cache[i_ij][i_kl] block (where
            // n_pno_[i_ij] > 0 and n_pno_[i_kl] > 0 but bs is 0×0)
            // contributes zeros instead of leaking uninitialised pinned
            // memory into d_barS_pad.
            real_t* h_barS_flat = nullptr;
            check_cuda_(cudaMallocHost(&h_barS_flat,
                            flat_size * sizeof(real_t)),
                        "cudaMallocHost h_barS_flat");
            std::memset(h_barS_flat, 0, flat_size * sizeof(real_t));

            #pragma omp parallel for collapse(2) schedule(static)
            for (int ai = 0; ai < N_act_ij; ++ai) {
                for (int ak = 0; ak < N_act_kl; ++ak) {
                    const int i_ij = active_i_ij_[ai];
                    const int i_kl = active_i_kl_[ak];
                    const int n_ij = n_pno_[i_ij];
                    const int n_kl = n_pno_[i_kl];
                    const RowMatXd& bs = barS_cache[i_ij][i_kl];
                    if (bs.rows() == 0 || bs.cols() == 0) continue;
                    const size_t bi =
                        static_cast<size_t>(ai) * N_act_kl + ak;
                    std::memcpy(h_barS_flat + block_src_off[bi],
                                bs.data(),
                                static_cast<size_t>(n_ij) * n_kl
                                    * sizeof(real_t));
                }
            }

            // Build n_pno arrays indexed by active position (small).
            std::vector<int> n_pno_active_ij(static_cast<size_t>(N_act_ij));
            std::vector<int> n_pno_active_kl(static_cast<size_t>(N_act_kl));
            for (int ai = 0; ai < N_act_ij; ++ai)
                n_pno_active_ij[ai] = n_pno_[active_i_ij_[ai]];
            for (int ak = 0; ak < N_act_kl; ++ak)
                n_pno_active_kl[ak] = n_pno_[active_i_kl_[ak]];

            // Device temporaries (kernel input only; freed after launch).
            int*    d_n_pno_active_ij = nullptr;
            int*    d_n_pno_active_kl = nullptr;
            size_t* d_block_src_off   = nullptr;
            real_t* d_barS_flat       = nullptr;

            check_cuda_(cudaMalloc(&d_n_pno_active_ij,
                            N_act_ij * sizeof(int)),
                        "cudaMalloc d_n_pno_active_ij");
            check_cuda_(cudaMalloc(&d_n_pno_active_kl,
                            N_act_kl * sizeof(int)),
                        "cudaMalloc d_n_pno_active_kl");
            check_cuda_(cudaMalloc(&d_block_src_off,
                            (n_blocks + 1) * sizeof(size_t)),
                        "cudaMalloc d_block_src_off");
            check_cuda_(cudaMalloc(&d_barS_flat,
                            flat_size * sizeof(real_t)),
                        "cudaMalloc d_barS_flat");

            check_cuda_(cudaMemcpy(d_n_pno_active_ij, n_pno_active_ij.data(),
                            N_act_ij * sizeof(int),
                            cudaMemcpyHostToDevice),
                        "H2D d_n_pno_active_ij");
            check_cuda_(cudaMemcpy(d_n_pno_active_kl, n_pno_active_kl.data(),
                            N_act_kl * sizeof(int),
                            cudaMemcpyHostToDevice),
                        "H2D d_n_pno_active_kl");
            check_cuda_(cudaMemcpy(d_block_src_off, block_src_off.data(),
                            (n_blocks + 1) * sizeof(size_t),
                            cudaMemcpyHostToDevice),
                        "H2D d_block_src_off");
            check_cuda_(cudaMemcpy(d_barS_flat, h_barS_flat,
                            flat_size * sizeof(real_t),
                            cudaMemcpyHostToDevice),
                        "H2D d_barS_flat");

            // Pre-zero the COMPACT padding region.
            check_cuda_(cudaMemset(s.d_barS_pad, 0, bytes_barS_compact),
                        "cudaMemset d_barS_pad (compact)");

            // Scatter kernel: one block per active (ai, ak) pair → compact
            // destination [ai · N_act_kl + ak].
            dim3 grid(static_cast<unsigned>(N_act_ij),
                      static_cast<unsigned>(N_act_kl));
            dim3 block(8, 8);
            scatter_barS_kernel<<<grid, block>>>(
                d_barS_flat, d_block_src_off,
                d_n_pno_active_ij, d_n_pno_active_kl,
                s.d_barS_pad, N_act_kl, max_n_);
            check_cuda_(cudaGetLastError(), "scatter_barS_kernel launch");
            check_cuda_(cudaDeviceSynchronize(),
                        "scatter_barS_kernel sync");

            cudaFree(d_barS_flat);
            cudaFree(d_block_src_off);
            cudaFree(d_n_pno_active_kl);
            cudaFree(d_n_pno_active_ij);
            cudaFreeHost(h_barS_flat);
        } else {
            // No active pairs: zero everything so downstream
            // pi = barS · Y · barS^T trivially produces zero.
            check_cuda_(cudaMemset(s.d_barS_pad, 0, bytes_barS_compact),
                        "cudaMemset d_barS_pad (no active)");
        }
    }

    // Step S8 profiling — dump per-stage timings as a single atomic line
    // per device (printf is line-atomic on glibc up to PIPE_BUF). The
    // double-precision values below are in milliseconds. With multi-GPU
    // OMP construction this will produce n_gpus lines, each tagged with
    // its device_id_ for attribution. Bytes annotated so we can compute
    // per-stage throughput (MB / ms = GB / s).
    {
        const auto t_barS_h2d = std::chrono::steady_clock::now();
        using msd = std::chrono::duration<double, std::milli>;
        const double ms_cpu      = msd(t_cpu_prep   - t_ctor_0).count();
        const double ms_probe    = msd(t_mem_probe  - t_cpu_prep).count();
        const double ms_alloc_b  = msd(t_alloc_base - t_mem_probe).count();
        const double ms_alloc_s  = msd(t_alloc_stack- t_alloc_base).count();
        const double ms_cublas   = msd(t_cublas     - t_alloc_stack).count();
        const double ms_barS     = msd(t_barS_h2d   - t_cublas).count();
        const double ms_total    = msd(t_barS_h2d   - t_ctor_0).count();
        // pi_T sizes: `full` is the full-N_pair count (legacy reference,
        // matches d_pi_T_stack on device); `slab` is the slab-only count
        // (matches h_pi_T_stack pinned host alloc after Step S9).
        const double mb_pi_T_full = static_cast<double>(bytes_pi_T)
                                    / (1024.0 * 1024.0);
        const double mb_pi_T_slab = static_cast<double>(bytes_pi_T_slab)
                                    / (1024.0 * 1024.0);
        const double mb_barS      = static_cast<double>(bytes_barS_compact)
                                    / (1024.0 * 1024.0);
        std::printf(
            "[picache-ctor dev=%d slab=[%d,%d) N_act_ij=%d N_act_kl=%d max_n=%d]"
            " cpu=%.1f probe=%.1f alloc_base=%.1f alloc_stack=%.1f"
            " cublas=%.1f barS_h2d=%.1f total=%.1f ms"
            " | pi_T_full=%.0f MB pi_T_slab=%.0f MB barS=%.0f MB stacked=%d\n",
            device_id_, pair_begin_, pair_end_,
            N_act_ij_, N_act_kl_, max_n_,
            ms_cpu, ms_probe, ms_alloc_b, ms_alloc_s,
            ms_cublas, ms_barS, ms_total,
            mb_pi_T_full, mb_pi_T_slab, mb_barS, stacked_ ? 1 : 0);
        std::fflush(stdout);
    }

    active_ = true;
#else
    (void)max_n;
    active_ = false;  // CPU-only build: always fallback path
#endif
}

// ---------------------------------------------------------------------------
//  Destructor
// ---------------------------------------------------------------------------
PiCacheGpu::~PiCacheGpu() {
#ifndef GANSU_CPU_ONLY
    if (p_) {
        // [picache-PROF] dump the per-iter rebuild() sub-phase split (env
        // GANSU_DLPNO_PICACHE_PROF=1). Per-device line; sum over the slab's
        // n_rebuild calls. Tells us whether the MP2 picache cost is pack_Y,
        // tile GEMM, D2H, or host scatter before targeting an optimization.
        if (p_->n_rebuild > 0) {
            const char* e = std::getenv("GANSU_DLPNO_PICACHE_PROF");
            if (e && e[0] == '1') {
                std::printf("[picache-PROF dev=%d] rebuild calls=%d  "
                            "pack_Y=%.3fs  tile_GEMM=%.3fs  D2H=%.3fs  "
                            "scatter=%.3fs  (slab [%d,%d) N_act_ij=%d "
                            "N_act_kl=%d)\n",
                            device_id_, p_->n_rebuild, p_->t_packY,
                            p_->t_gemm, p_->t_d2h, p_->t_scatter,
                            pair_begin_, pair_end_, N_act_ij_, N_act_kl_);
                std::fflush(stdout);
            }
        }
        // Free buffers on the device that allocated them.
        MultiGpuManager::DeviceGuard _guard(device_id_);
        p_->free_all();
        delete p_;
        p_ = nullptr;
    }
#endif
}

// ---------------------------------------------------------------------------
//  CPU fallback — equivalent to the original Eigen kernel in
//  iterate_lmp2 / iterate_dlpno_ccsd_t2 (kept here so the caller can use a
//  single API path regardless of GPU availability).
//
//  Optimizations on top of the original per-pair Eigen loop:
//   (1) active_i_ij_ / active_i_kl_ — precomputed indices with n_pno > 0
//       turn the 5886² outer×inner sweep on cholesterol into a 1952² active
//       sub-grid (≈10× fewer iterations). Inactive cells in pi_cache_out
//       stay default-constructed 0×0 — no `resize(0, 0)` needed each iter.
//       Both lists are built once in the constructor; n_pno_ is fixed
//       across the entire iter loop (PNO re-truncation lives in the SC-PNO
//       refresh, which constructs a fresh PiCacheGpu).
//   (2) thread-local scratch `half_buf` — eliminates the per-pair heap
//       allocation that `const RowMatXd half = barS * Y_canon` triggers
//       (~50 ns × 1952² × 53 iter ≈ 10s on cholesterol). We pre-allocate
//       max_n × max_n doubles per thread and map a Map<RowMatXd> view of
//       shape (n_ij × n_kl) on top of it each pair-pair — no realloc.
//   (3) schedule(static) over active_i_ij_ — even thread load (vs the
//       original sweep that gave threads a mix of active+inactive outers).
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild_cpu_(
    const std::vector<std::vector<real_t>>& Y_old,
    std::vector<std::vector<RowMatXd>>& pi_cache_out)
{
    const auto& barS_cache = *barS_cache_ref_;
    const int max_n = max_n_;
    const long long n_active_ij = static_cast<long long>(active_i_ij_.size());
    const int* const active_kl_ptr = active_i_kl_.data();
    const int n_active_kl = static_cast<int>(active_i_kl_.size());

    #pragma omp parallel
    {
        // Thread-local scratch for the intermediate `half = barS · Y_canon`.
        // Sized once to max_n × max_n; per pair-pair a fresh Map<RowMatXd>
        // view of shape (n_ij × n_kl) re-uses the same memory. The buffer
        // is overwritten every call, so the unused trailing entries don't
        // contaminate downstream consumers.
        std::vector<real_t> half_buf(
            static_cast<size_t>(max_n) * static_cast<size_t>(max_n));

        #pragma omp for schedule(static)
        for (long long idx = 0; idx < n_active_ij; ++idx) {
            const int i_ij = active_i_ij_[idx];
            const int n_ij = n_pno_[i_ij];
            // n_ij > 0 guaranteed by the active_i_ij_ list construction.

            const auto& barS_row = barS_cache[i_ij];
            auto&       pi_row   = pi_cache_out[i_ij];

            for (int j = 0; j < n_active_kl; ++j) {
                const int i_kl = active_kl_ptr[j];
                const int n_kl = n_pno_[i_kl];

                const RowMatXd& barS = barS_row[i_kl];
                Eigen::Map<const RowMatXd> Y_canon(
                    Y_old[i_kl].data(), n_kl, n_kl);
                Eigen::Map<RowMatXd> half(half_buf.data(), n_ij, n_kl);
                half.noalias() = barS * Y_canon;
                pi_row[i_kl].noalias() = half * barS.transpose();
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  rebuild — main per-iter entry point.
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild(const std::vector<std::vector<real_t>>& Y_old,
                         std::vector<std::vector<RowMatXd>>& pi_cache_out)
{
    if (!active_) {
        rebuild_cpu_(Y_old, pi_cache_out);
        return;
    }

#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(device_id_);

    // Step Z — drive the tile loop with per-tile D2H + host scatter so
    // pi_cache_out[i_ij][i_kl] is populated from the compact d_pi_pad
    // (rows indexed by active ai, cols by active ak) without relying on
    // the legacy N_pair × N_pair layout that download_pi_cache_ assumed.
    Impl& s = *p_;
    const int max_n = max_n_;
    const long long stride_pair = s.stride_pair;
    const size_t bytes_pi_tile_full =
        static_cast<size_t>(tile_size_) * N_act_kl_
        * static_cast<size_t>(max_n) * max_n * sizeof(real_t);

    // Pinned host tile mirror — allocated on first call (sized to one tile,
    // reused across iters). Replaces the old N_pair²-sized h_pi_pad which
    // is no longer compatible with the compact d_pi_pad layout.
    if (bytes_pi_tile_full > 0 && s.h_pi_pad == nullptr) {
        check_cuda_(cudaMallocHost(&s.h_pi_pad, bytes_pi_tile_full),
                    "cudaMallocHost h_pi_pad (tile, Step Z)");
    }

    // [picache-PROF] sub-phase split (env GANSU_DLPNO_PICACHE_PROF=1). When on,
    // each timed phase is bracketed by a cudaDeviceSynchronize so async GPU
    // work is attributed to the right phase; off ⇒ zero cost (no syncs added).
    static const bool picache_prof = []() {
        const char* e = std::getenv("GANSU_DLPNO_PICACHE_PROF");
        return e && e[0] == '1';
    }();
    std::chrono::steady_clock::time_point pc_t0;
    if (picache_prof) ++s.n_rebuild;

    if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
    pack_Y_and_transpose_(Y_old);
    if (picache_prof) {
        cudaDeviceSynchronize();
        s.t_packY += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();
    }

    // Pre-resize pi_cache_out for inactive pairs (zero-size, so downstream
    // consumers handle "no contribution"). Only ROWS in our slab; peer
    // PiCacheGpus on other devices populate other slabs.
    for (long long i_ij = pair_begin_; i_ij < pair_end_; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) {
            for (int i_kl = 0; i_kl < N_pair_; ++i_kl) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
            }
        }
    }

    for (int tile_start = 0; tile_start < N_act_ij_;
             tile_start += tile_size_) {
        const int tile_end = std::min(tile_start + tile_size_, N_act_ij_);
        const int tile_rows = tile_end - tile_start;
        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        compute_pi_tile_(tile_start, tile_end);
        if (picache_prof) {
            cudaDeviceSynchronize();
            s.t_gemm += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pc_t0).count();
        }

        // D2H only this tile's worth.
        const size_t bytes_this_tile =
            static_cast<size_t>(tile_rows) * N_act_kl_
            * static_cast<size_t>(max_n) * max_n * sizeof(real_t);
        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        check_cuda_(cudaMemcpy(s.h_pi_pad, s.d_pi_pad,
                               bytes_this_tile, cudaMemcpyDeviceToHost),
                    "D2H d_pi_pad (tile, Step Z)");
        if (picache_prof) s.t_d2h += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();

        // Host scatter: per active (ai_in_tile, ak) → pi_cache_out[i_ij][i_kl].
        // pi block shape is (n_ij × n_ij) — n_kl only appears in the
        // intermediate compute, not the output.
        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(static)
        for (int ai_in_tile = 0; ai_in_tile < tile_rows; ++ai_in_tile) {
            const int ai = ai_in_tile + tile_start;
            const int i_ij = active_i_ij_[ai];
            const int n_ij = n_pno_[i_ij];
            // Zero out non-active i_kl columns first (inactive pairs).
            // Active pair list is dense, so it's faster to loop all
            // and only resize on inactive than to check per-pair.
            for (int i_kl = 0; i_kl < N_pair_; ++i_kl) {
                if (active_kl_pos_[i_kl] < 0) {
                    pi_cache_out[i_ij][i_kl].resize(0, 0);
                }
            }
            for (int ak = 0; ak < N_act_kl_; ++ak) {
                const int i_kl = active_i_kl_[ak];
                pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
                const real_t* src = s.h_pi_pad
                    + (static_cast<size_t>(ai_in_tile) * N_act_kl_
                       + static_cast<size_t>(ak))
                    * static_cast<size_t>(stride_pair);
                real_t* dst = pi_cache_out[i_ij][i_kl].data();
                for (int r = 0; r < n_ij; ++r) {
                    std::memcpy(dst + static_cast<ptrdiff_t>(r) * n_ij,
                                src + static_cast<size_t>(r) * max_n,
                                static_cast<size_t>(n_ij) * sizeof(real_t));
                }
            }
        }
        if (picache_prof) s.t_scatter += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();
    }
#endif // !GANSU_CPU_ONLY
}

// ---------------------------------------------------------------------------
//  rebuild_needed — LMP2 needed-column variant of rebuild(). Same GEMM into
//  the tile's d_pi_pad, but a device gather kernel compacts only the columns
//  the residual reads ((k,j)/(i,l) per i_ij) before a (~6× smaller) D2H and a
//  needed-only host scatter. Bit-exact w.r.t. rebuild() for consumed entries.
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild_needed(
    const std::vector<std::vector<real_t>>& Y_old,
    std::vector<std::vector<RowMatXd>>& pi_cache_out,
    const std::vector<std::vector<int>>& needed_ikl_per_pair)
{
    // CPU fallback / inactive GPU: full rebuild (correct; no D2H saving).
    if (!active_) {
        rebuild_cpu_(Y_old, pi_cache_out);
        return;
    }
#ifdef GANSU_CPU_ONLY
    rebuild_cpu_(Y_old, pi_cache_out);
#else
    MultiGpuManager::DeviceGuard _guard(device_id_);
    Impl& s = *p_;
    const int max_n = max_n_;
    const long long stride_pair = s.stride_pair;   // max_n²

    // --- one-time build of the device needed-column map (iter-invariant) ---
    if (!s.needed_built) {
        s.needed_ikl_host.assign(N_act_ij_, {});
        int max_needed = 0;
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const int i_ij = active_i_ij_[ai];
            std::vector<int>& dst = s.needed_ikl_host[ai];
            if (i_ij >= 0 && i_ij < static_cast<int>(needed_ikl_per_pair.size())) {
                for (int i_kl : needed_ikl_per_pair[i_ij]) {
                    if (i_kl >= 0 && i_kl < N_pair_ && active_kl_pos_[i_kl] >= 0)
                        dst.push_back(i_kl);
                }
            }
            if (static_cast<int>(dst.size()) > max_needed)
                max_needed = static_cast<int>(dst.size());
        }
        s.max_needed = std::max(1, max_needed);

        std::vector<int> h_ak(static_cast<size_t>(N_act_ij_) * s.max_needed, -1);
        std::vector<int> h_cnt(std::max(1, N_act_ij_), 0);
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const std::vector<int>& dst = s.needed_ikl_host[ai];
            h_cnt[ai] = static_cast<int>(dst.size());
            for (int n = 0; n < static_cast<int>(dst.size()); ++n)
                h_ak[static_cast<size_t>(ai) * s.max_needed + n] =
                    active_kl_pos_[dst[n]];
        }
        check_cuda_(cudaMalloc(&s.d_needed_ak,
                               sizeof(int) * std::max<size_t>(1, h_ak.size())),
                    "cudaMalloc d_needed_ak");
        check_cuda_(cudaMalloc(&s.d_needed_count,
                               sizeof(int) * std::max(1, N_act_ij_)),
                    "cudaMalloc d_needed_count");
        check_cuda_(cudaMemcpy(s.d_needed_ak, h_ak.data(),
                               sizeof(int) * h_ak.size(),
                               cudaMemcpyHostToDevice), "H2D d_needed_ak");
        check_cuda_(cudaMemcpy(s.d_needed_count, h_cnt.data(),
                               sizeof(int) * h_cnt.size(),
                               cudaMemcpyHostToDevice), "H2D d_needed_count");
        const size_t bytes_needed =
            static_cast<size_t>(tile_size_) * s.max_needed
            * static_cast<size_t>(stride_pair) * sizeof(real_t);
        if (bytes_needed > 0) {
            check_cuda_(cudaMalloc(&s.d_pi_needed, bytes_needed),
                        "cudaMalloc d_pi_needed");
            check_cuda_(cudaMallocHost(&s.h_pi_needed, bytes_needed),
                        "cudaMallocHost h_pi_needed");
        }
        s.needed_built = true;
    }

    static const bool picache_prof = []() {
        const char* e = std::getenv("GANSU_DLPNO_PICACHE_PROF");
        return e && e[0] == '1';
    }();
    std::chrono::steady_clock::time_point pc_t0;
    if (picache_prof) ++s.n_rebuild;

    if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
    pack_Y_and_transpose_(Y_old);
    if (picache_prof) {
        cudaDeviceSynchronize();
        s.t_packY += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();
    }

    // Inactive-pair rows → 0×0 (those i_ij are never written below).
    for (long long i_ij = pair_begin_; i_ij < pair_end_; ++i_ij) {
        if (n_pno_[i_ij] == 0) {
            for (int i_kl = 0; i_kl < N_pair_; ++i_kl)
                pi_cache_out[i_ij][i_kl].resize(0, 0);
        }
    }

    for (int tile_start = 0; tile_start < N_act_ij_;
             tile_start += tile_size_) {
        const int tile_end  = std::min(tile_start + tile_size_, N_act_ij_);
        const int tile_rows = tile_end - tile_start;

        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        compute_pi_tile_(tile_start, tile_end);
        if (picache_prof) {
            cudaDeviceSynchronize();
            s.t_gemm += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pc_t0).count();
        }

        // Gather the needed columns of this tile into the compact buffer.
        const int n_slots = tile_rows * s.max_needed;
        if (n_slots > 0) {
            gather_needed_kernel<<<n_slots, 256>>>(
                s.d_pi_pad, s.d_pi_needed, s.d_needed_ak, s.d_needed_count,
                tile_start, tile_rows, s.max_needed, stride_pair, N_act_kl_);
            check_cuda_(cudaGetLastError(), "gather_needed_kernel launch");
        }

        // D2H only the compact (tile_rows × max_needed) needed blocks.
        const size_t bytes_this =
            static_cast<size_t>(tile_rows) * s.max_needed
            * static_cast<size_t>(stride_pair) * sizeof(real_t);
        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        if (bytes_this > 0)
            check_cuda_(cudaMemcpy(s.h_pi_needed, s.d_pi_needed, bytes_this,
                                   cudaMemcpyDeviceToHost),
                        "D2H d_pi_needed");
        if (picache_prof) s.t_d2h += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();

        // Host scatter: only the needed (i_ij, i_kl) entries.
        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(static)
        for (int ai_in_tile = 0; ai_in_tile < tile_rows; ++ai_in_tile) {
            const int ai   = ai_in_tile + tile_start;
            const int i_ij = active_i_ij_[ai];
            const int n_ij = n_pno_[i_ij];
            const std::vector<int>& need = s.needed_ikl_host[ai];
            for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                const int i_kl = need[n];
                pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
                const real_t* srcb = s.h_pi_needed
                    + (static_cast<size_t>(ai_in_tile) * s.max_needed
                       + static_cast<size_t>(n))
                      * static_cast<size_t>(stride_pair);
                real_t* dstb = pi_cache_out[i_ij][i_kl].data();
                for (int r = 0; r < n_ij; ++r) {
                    std::memcpy(dstb + static_cast<ptrdiff_t>(r) * n_ij,
                                srcb + static_cast<size_t>(r) * max_n,
                                static_cast<size_t>(n_ij) * sizeof(real_t));
                }
            }
        }
        if (picache_prof) s.t_scatter += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pc_t0).count();
    }
#endif // GANSU_CPU_ONLY
}

#ifndef GANSU_CPU_ONLY
// ---------------------------------------------------------------------------
//  rebuild_gpu_kernels_ — GPU-side work only: pad Y → H2D, two batched DGEMMs
//  filling d_pi_pad on device. No D2H of d_pi_pad. Caller must hold a
//  DeviceGuard on device_id_. Used both by rebuild() (which then downloads
//  via download_pi_cache_) and by rebuild_with_stack() when the caller has
//  asked to skip the host pi_cache (skip_pi_cache_host=true).
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild_gpu_kernels_(
    const std::vector<std::vector<real_t>>& Y_old)
{
    // Step Z: NOTE — d_pi_pad is now a TILE buffer (tile_size_ × N_act_kl ×
    // max_n²). This routine fills d_pi_pad ONE TILE AT A TIME; callers
    // that need d_pi_pad on device for downstream consumption must do so
    // tile-by-tile too. The canonical pipeline runs through
    // rebuild_with_stack, which packs each tile into d_pi_T_stack before
    // moving on to the next tile.
    //
    // For backward compatibility with the rebuild()/download_pi_cache_
    // path (host pi_cache_out), the caller invokes
    // rebuild_one_tile_(tile_start, tile_end) explicitly per tile. This
    // top-level function is now only valid when d_pi_pad has capacity for
    // all N_act_ij_ rows (tile_size_ == N_act_ij_), so a single tile fits.
    //
    // The common case where tile_size_ < N_act_ij_ (cholesterol) MUST use
    // rebuild_with_stack instead — this routine throws if invoked then.
    Impl& s = *p_;
    if (tile_size_ < N_act_ij_) {
        throw std::runtime_error(
            "PiCacheGpu::rebuild_gpu_kernels_: d_pi_pad is tiled "
            "(tile_size_ < N_act_ij_) — use rebuild_with_stack");
    }
    // Pack Y, transpose, then run a single full "tile" covering all rows.
    pack_Y_and_transpose_(Y_old);
    compute_pi_tile_(/*tile_start=*/0, /*tile_end=*/N_act_ij_);
    (void)s;
}

// ---------------------------------------------------------------------------
// Step Z helper — pack Y_old into compact h_Y_pad/d_Y_pad and build the
// transpose. Called once per rebuild (before the tile loop).
// ---------------------------------------------------------------------------
void PiCacheGpu::pack_Y_and_transpose_(
    const std::vector<std::vector<real_t>>& Y_old)
{
    Impl& s = *p_;
    const int max_n = max_n_;
    const long long stride_pair = s.stride_pair;

    const size_t bytes_Y_compact =
        static_cast<size_t>(N_act_kl_) * static_cast<size_t>(max_n)
        * static_cast<size_t>(max_n) * sizeof(real_t);

    // Pack Y_old → h_Y_pad, indexed by active position ak (not orig i_kl).
    std::memset(s.h_Y_pad, 0, bytes_Y_compact);
    #pragma omp parallel for schedule(static)
    for (int ak = 0; ak < N_act_kl_; ++ak) {
        const int i_kl = active_i_kl_[ak];
        const int n_kl = n_pno_[i_kl];
        const real_t* src = Y_old[i_kl].data();
        real_t* dst = s.h_Y_pad + static_cast<size_t>(ak) * stride_pair;
        for (int r = 0; r < n_kl; ++r) {
            std::memcpy(dst + static_cast<size_t>(r) * max_n,
                        src + static_cast<ptrdiff_t>(r) * n_kl,
                        static_cast<size_t>(n_kl) * sizeof(real_t));
        }
    }
    check_cuda_(cudaMemcpy(s.d_Y_pad, s.h_Y_pad,
                           bytes_Y_compact, cudaMemcpyHostToDevice),
                "rebuild H2D Y_pad (compact)");

    // Transpose grid over active_i_kl positions; each block reads compact
    // d_Y_pad[ak] and writes transposed to d_Y_pad_T[orig_i_kl] so that
    // oooo_lad_kernel in ResidGpu sees the legacy full-N_pair layout.
    if (N_act_kl_ > 0) {
        const int block_dim = (max_n <= 16) ? max_n : 16;
        const dim3 t_block(block_dim, block_dim);
        const dim3 t_grid(static_cast<unsigned>(N_act_kl_));
        transpose_Y_pad_kernel<<<t_grid, t_block>>>(
            s.d_Y_pad, s.d_Y_pad_T,
            s.d_active_i_kl, N_act_kl_, max_n);
        check_cuda_(cudaGetLastError(), "transpose_Y_pad_kernel launch");
    }
}

// ---------------------------------------------------------------------------
// Step Z helper — compute one tile of d_pi_pad (rows in compact active i_ij
// space [tile_start, tile_end)). d_pi_pad indexing inside the tile is
// (ai - tile_start) × N_act_kl × max_n². batched DGEMM over the active i_kl
// columns (N_act_kl batches), with both barS and Y read from compact buffers.
// ---------------------------------------------------------------------------
void PiCacheGpu::compute_pi_tile_(int tile_start, int tile_end)
{
    Impl& s = *p_;
    const int max_n = max_n_;
    const long long stride_pair = s.stride_pair;
    const int N_kl = N_act_kl_;
    const real_t one  = 1.0;
    const real_t zero = 0.0;

    // Conventions: compact d_barS_pad rows are indexed by ai (full N_act_kl
    // columns per row). cuBLAS sees the same column-major view as before.
    for (int ai = tile_start; ai < tile_end; ++ai) {
        const int i_ij = active_i_ij_[ai];
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) continue;  // active list filtered this, but be safe

        real_t* dA_row = s.d_barS_pad
                      + static_cast<size_t>(ai) * N_kl * stride_pair;
        real_t* dC_row = s.d_pi_pad
                      + static_cast<size_t>(ai - tile_start) * N_kl
                        * stride_pair;

        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/ max_n, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            s.d_Y_pad,    /*lda=*/ max_n, /*strideA=*/ stride_pair,
            dA_row,       /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            s.d_half_pad, /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            N_kl), "stage1 strided batched (tile)");

        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            /*m=*/ n_ij, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            dA_row,       /*lda=*/ max_n, /*strideA=*/ stride_pair,
            s.d_half_pad, /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            dC_row,       /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            N_kl), "stage2 strided batched (tile)");
    }
}

// ---------------------------------------------------------------------------
//  download_pi_cache_ — D2H d_pi_pad → h_pi_pad (slab range only) and unpad
//  into pi_cache_out. Caller must hold DeviceGuard. Run after
//  rebuild_gpu_kernels_ when the host pi_cache is needed.
// ---------------------------------------------------------------------------
void PiCacheGpu::download_pi_cache_(
    std::vector<std::vector<RowMatXd>>& pi_cache_out)
{
    // Step Z — d_pi_pad is now compact + tiled (rows indexed by active ai,
    // not orig i_ij). The legacy N_pair × N_pair × max_n² slab D2H below
    // no longer reads the right bytes. Callers that need pi_cache_out
    // populated should use rebuild_with_stack with skip_pi_cache_host=true
    // followed by their own consumer of pi_T_stack, or run on a smaller
    // system where the path is tile_size_ == N_act_ij_ and we can implement
    // single-tile compact scatter.
    //
    // For now, throw if anyone reaches here — Step 6 era always uses the
    // skip path, so this should not fire in practice. If it does, the
    // intended fix is a compact-aware scatter that walks active_i_ij_ /
    // active_i_kl_ instead of the orig (i_ij, i_kl) sweep.
    throw std::runtime_error(
        "PiCacheGpu::download_pi_cache_: incompatible with Step Z compact "
        "d_pi_pad layout. Use rebuild_with_stack with skip_pi_cache_host=true.");

    Impl& s = *p_;
    const int N      = N_pair_;
    const int max_n  = max_n_;
    const int ib = pair_begin_;
    const int ie = pair_end_;
    const int slab_n = ie - ib;

    // Slab D2H size: only the slab's output rows of pi_pad.
    const size_t bytes_slab  = static_cast<size_t>(std::max(0, slab_n))
                             * static_cast<size_t>(N)
                             * static_cast<size_t>(max_n)
                             * static_cast<size_t>(max_n) * sizeof(real_t);

    // Step S1: lazy allocation of h_pi_pad. Step 6's stacked-mode +
    // skip_pi_cache_host=true path bypasses download_pi_cache_ entirely,
    // so this large pinned buffer is allocated only when actually needed.
    if (slab_n > 0 && s.h_pi_pad == nullptr) {
        const size_t bytes_full = static_cast<size_t>(N)
                                * static_cast<size_t>(N)
                                * static_cast<size_t>(max_n)
                                * static_cast<size_t>(max_n) * sizeof(real_t);
        check_cuda_(cudaMallocHost(&s.h_pi_pad, bytes_full),
                    "cudaMallocHost h_pi_pad (lazy)");
    }

    // D2H — only the slab range (per-pair N × max_n² block starting at
    // pair_begin_, length slab_n contiguous in i_ij).
    if (slab_n > 0) {
        const size_t row_bytes = static_cast<size_t>(N)
                               * static_cast<size_t>(max_n)
                               * static_cast<size_t>(max_n) * sizeof(real_t);
        const size_t off_bytes = static_cast<size_t>(ib) * row_bytes;
        check_cuda_(cudaMemcpy(
            reinterpret_cast<char*>(s.h_pi_pad) + off_bytes,
            reinterpret_cast<const char*>(s.d_pi_pad) + off_bytes,
            bytes_slab, cudaMemcpyDeviceToHost),
            "rebuild D2H pi_pad (slab)");
    }

    // -------- Unpad h_pi_pad → pi_cache_out (host).
    //          Threads outside the slab leave their pi_cache_out rows
    //          untouched (other PiCacheGpu instances on peer GPUs populate
    //          them in the multi-GPU dispatch). On single-GPU operation
    //          (ib=0, ie=N) the entire vector is populated as before.
    #pragma omp parallel for schedule(static)
    for (long long i_ij = ib; i_ij < ie; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) {
            for (int i_kl = 0; i_kl < N; ++i_kl) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
            }
            continue;
        }
        for (int i_kl = 0; i_kl < N; ++i_kl) {
            const int n_kl = n_pno_[i_kl];
            if (n_kl == 0) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
                continue;
            }
            pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
            const real_t* src = s.h_pi_pad
                              + (static_cast<size_t>(i_ij)
                                 * static_cast<size_t>(N)
                                 + static_cast<size_t>(i_kl))
                                * static_cast<size_t>(max_n)
                                * static_cast<size_t>(max_n);
            real_t* dst = pi_cache_out[i_ij][i_kl].data();
            for (int r = 0; r < n_ij; ++r) {
                std::memcpy(dst + static_cast<ptrdiff_t>(r) * n_ij,
                            src + static_cast<size_t>(r) * max_n,
                            static_cast<size_t>(n_ij) * sizeof(real_t));
            }
        }
    }
}
#endif // !GANSU_CPU_ONLY

// ---------------------------------------------------------------------------
//  Step 6.2 — read-only device buffer getters for ResidGpu cooperation.
// ---------------------------------------------------------------------------
const real_t* PiCacheGpu::device_pi_T_stack() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && stacked_) ? p_->d_pi_T_stack : nullptr;
#else
    return nullptr;
#endif
}
const real_t* PiCacheGpu::device_Y_pad() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && active_) ? p_->d_Y_pad : nullptr;
#else
    return nullptr;
#endif
}
const real_t* PiCacheGpu::device_Y_pad_T() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && active_) ? p_->d_Y_pad_T : nullptr;
#else
    return nullptr;
#endif
}
const int* PiCacheGpu::device_pair_lookup() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && stacked_) ? p_->d_pair_lookup : nullptr;
#else
    return nullptr;
#endif
}
const int* PiCacheGpu::device_setup_i() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && stacked_) ? p_->d_setup_i : nullptr;
#else
    return nullptr;
#endif
}
const int* PiCacheGpu::device_n_pno() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && stacked_) ? p_->d_n_pno : nullptr;
#else
    return nullptr;
#endif
}
const size_t* PiCacheGpu::device_idx_offset_pi_T() const noexcept {
#ifndef GANSU_CPU_ONLY
    return (p_ && stacked_) ? p_->d_idx_offset : nullptr;
#else
    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
//  CPU fallback — assemble pi_T_stack from a host-resident pi_cache.
//  Equivalent to the original middleCols loop in iterate_dlpno_ccsd_t2.
// ---------------------------------------------------------------------------
void PiCacheGpu::build_stack_cpu_(
    const std::vector<std::vector<RowMatXd>>& pi_cache,
    std::vector<RowMatXd>& pi_T_stack_out)
{
    const int nocc  = nocc_;
    const auto& pl  = pair_lookup_;
    const auto& si  = setup_i_per_pair_;
    const int ib   = pair_begin_;
    const int ie   = pair_end_;

    #pragma omp parallel for schedule(static)
    for (long long i_ij = ib; i_ij < ie; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) {
            pi_T_stack_out[i_ij].resize(0, 0);
            continue;
        }
        pi_T_stack_out[i_ij].setZero(
            n_ij, static_cast<size_t>(nocc) * nocc * n_ij);
        for (int k = 0; k < nocc; ++k) {
            for (int l = 0; l < nocc; ++l) {
                const int idx_kl = pl[k * nocc + l];
                if (n_pno_[idx_kl] == 0) continue;
                const RowMatXd& pi_canon = pi_cache[i_ij][idx_kl];
                const size_t col_off =
                    (static_cast<size_t>(k) * nocc + l) * n_ij;
                if (si[idx_kl] != k) {
                    pi_T_stack_out[i_ij].middleCols(col_off, n_ij) =
                        pi_canon.transpose();
                } else {
                    pi_T_stack_out[i_ij].middleCols(col_off, n_ij) =
                        pi_canon;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  rebuild_with_stack — Step 6.0 pi_cache + Step 6.1 pi_T_stack in one call.
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild_with_stack(
    const std::vector<std::vector<real_t>>& Y_old,
    std::vector<std::vector<RowMatXd>>& pi_cache_out,
    std::vector<RowMatXd>& pi_T_stack_out,
    bool skip_pi_cache_host)
{
    // Fast path: when the GPU + stacked-mode buffers are both live AND the
    // caller has told us pi_cache_out is dead this iter (any_rgpu_active),
    // run only the GPU kernels for d_pi_pad and skip the D2H + host unpad.
    // The pack kernel below still reads d_pi_pad on device, so pi_T_stack
    // is unaffected. pi_T_stack_out is still filled (DFpair needs it).
    const bool skip_host_pi = skip_pi_cache_host && active_ && stacked_;

    if (!stacked_) {
        // CPU fallback for pi_T_stack — same algorithm as the original
        // middleCols loop, just relocated here so the caller sees one API.
        rebuild(Y_old, pi_cache_out);
        build_stack_cpu_(pi_cache_out, pi_T_stack_out);
        return;
    }

#ifndef GANSU_CPU_ONLY
    MultiGpuManager::DeviceGuard _guard(device_id_);
    Impl& s = *p_;
    const int N      = N_pair_;
    const int max_n  = max_n_;
    const int nocc   = nocc_;
    const int ib    = pair_begin_;
    const int ie    = pair_end_;
    const int slab_n = ie - ib;
    const long long stride_pair = s.stride_pair;

    // Per-pair offsets into d_pi_T_stack — needed to compute the slab range.
    std::vector<size_t> idx_offset_host(static_cast<size_t>(N + 1), 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        idx_offset_host[i + 1] = idx_offset_host[i]
            + n * n
            * static_cast<size_t>(nocc)
            * static_cast<size_t>(nocc);
    }

    // Zero the slab range of d_pi_T_stack. Empty (i_ij, i_kl) cells skip
    // their writes; we need them zero on entry so they don't carry stale data.
    const size_t bytes_pi_T_slab = (idx_offset_host[ie] - idx_offset_host[ib])
                                 * sizeof(real_t);
    if (bytes_pi_T_slab > 0) {
        check_cuda_(cudaMemsetAsync(
            s.d_pi_T_stack + idx_offset_host[ib],
            0, bytes_pi_T_slab, /*stream=*/0),
            "memset d_pi_T_stack (slab)");
    }

    // Step Z — unified tile loop: pack Y once, then per-tile compute +
    // (optional) D2H scatter to pi_cache_out + pack into pi_T_stack.
    // skip_host_pi path skips the D2H + scatter (Step 6 fast path).
    pack_Y_and_transpose_(Y_old);

    // Pinned tile mirror for D2H scatter (non-skip only). Sized one tile.
    const size_t bytes_pi_tile_full =
        static_cast<size_t>(tile_size_) * N_act_kl_
        * static_cast<size_t>(max_n) * max_n * sizeof(real_t);
    if (!skip_host_pi && bytes_pi_tile_full > 0 && s.h_pi_pad == nullptr) {
        check_cuda_(cudaMallocHost(&s.h_pi_pad, bytes_pi_tile_full),
                    "cudaMallocHost h_pi_pad (tile)");
    }

    // Init pi_cache_out: zero-size blocks for all pairs in our slab.
    // We'll fill active blocks per-tile below.
    if (!skip_host_pi) {
        for (long long i_ij = ib; i_ij < ie; ++i_ij) {
            for (int i_kl = 0; i_kl < N; ++i_kl) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
            }
        }
    }

    if (slab_n > 0 && nocc > 0 && max_n > 0 && N_act_ij_ > 0 && tile_size_ > 0) {
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(N),
                  static_cast<unsigned>(nocc * nocc), 1);

        for (int tile_start = 0; tile_start < N_act_ij_;
                 tile_start += tile_size_) {
            const int tile_end = std::min(tile_start + tile_size_,
                                          N_act_ij_);
            const int tile_rows = tile_end - tile_start;
            compute_pi_tile_(tile_start, tile_end);

            // Non-skip path: D2H tile + host scatter to pi_cache_out.
            if (!skip_host_pi) {
                const size_t bytes_this_tile =
                    static_cast<size_t>(tile_rows) * N_act_kl_
                    * static_cast<size_t>(max_n) * max_n * sizeof(real_t);
                check_cuda_(cudaMemcpy(s.h_pi_pad, s.d_pi_pad,
                                       bytes_this_tile, cudaMemcpyDeviceToHost),
                            "D2H d_pi_pad (tile, rebuild_with_stack)");
                #pragma omp parallel for schedule(static)
                for (int ai_in_tile = 0; ai_in_tile < tile_rows; ++ai_in_tile) {
                    const int ai = ai_in_tile + tile_start;
                    const int i_ij = active_i_ij_[ai];
                    const int n_ij = n_pno_[i_ij];
                    for (int ak = 0; ak < N_act_kl_; ++ak) {
                        const int i_kl = active_i_kl_[ak];
                        pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
                        const real_t* src = s.h_pi_pad
                            + (static_cast<size_t>(ai_in_tile) * N_act_kl_
                               + static_cast<size_t>(ak))
                            * static_cast<size_t>(stride_pair);
                        real_t* dst = pi_cache_out[i_ij][i_kl].data();
                        for (int r = 0; r < n_ij; ++r) {
                            std::memcpy(dst + static_cast<ptrdiff_t>(r) * n_ij,
                                        src + static_cast<size_t>(r) * max_n,
                                        static_cast<size_t>(n_ij)
                                            * sizeof(real_t));
                        }
                    }
                }
            }

            pack_pi_T_stack_kernel<<<grid, block>>>(
                s.d_pi_pad,
                s.d_pair_lookup,
                s.d_setup_i,
                s.d_n_pno,
                s.d_idx_offset,
                s.d_active_ij_pos,
                s.d_active_kl_pos,
                s.d_pi_T_stack,
                N, nocc, max_n,
                N_act_kl_, tile_start, tile_end);
            const cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) {
                throw std::runtime_error(
                    std::string("PiCacheGpu pack kernel launch failed: ")
                    + cudaGetErrorString(e));
            }
        }
    }

    // D2H — only the slab range of pi_T_stack.
    // Step S9: h_pi_T_stack is sized to the slab only, so the destination is
    // its base (offset 0). The source on device keeps the full-N_pair layout
    // (used by kernels that index via d_idx_offset), so we still read from
    // d_pi_T_stack + idx_offset_host[ib].
    if (bytes_pi_T_slab > 0) {
        check_cuda_(cudaMemcpy(s.h_pi_T_stack,
                               s.d_pi_T_stack + idx_offset_host[ib],
                               bytes_pi_T_slab, cudaMemcpyDeviceToHost),
                    "D2H pi_T_stack (slab)");
    }

    // Host scatter: per-pair contiguous segment → pi_T_stack_out[idx] (n_ij × nocc²·n_ij).
    // The device layout already matches the unpadded host layout, so this
    // is a single memcpy per pair. Only fills the slab range; other slabs
    // are populated by peer PiCacheGpu instances on other GPUs.
    // Step S9: h_pi_T_stack is slab-base (0 = pair_begin_). For pair i_ij ∈
    // [pair_begin_, pair_end_), the corresponding source offset within the
    // slab buffer is idx_offset_host[i_ij] - idx_offset_host[ib], i.e. the
    // absolute offset minus the slab's leading prefix. This shift is a
    // compile-time-zero cost when num_gpus=1 (ib=0).
    const size_t slab_base = idx_offset_host[ib];
    #pragma omp parallel for schedule(static)
    for (long long i_ij = ib; i_ij < ie; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) {
            pi_T_stack_out[i_ij].resize(0, 0);
            continue;
        }
        pi_T_stack_out[i_ij].resize(
            n_ij, static_cast<size_t>(nocc) * nocc * n_ij);
        const size_t bytes = static_cast<size_t>(n_ij)
                           * static_cast<size_t>(n_ij)
                           * static_cast<size_t>(nocc)
                           * static_cast<size_t>(nocc) * sizeof(real_t);
        std::memcpy(pi_T_stack_out[i_ij].data(),
                    s.h_pi_T_stack + (idx_offset_host[i_ij] - slab_base),
                    bytes);
    }
#endif // !GANSU_CPU_ONLY
}

// ---------------------------------------------------------------------------
//  DFpair GPU port — upload_T_meta_dpair + compute_dfpair.
//
//  DF_per_pair[idx] = -(pi_T_stack[idx] · T_meta_dpair[idx]), one cublasDgemm
//  per pair (variable K = nocc²·n_ij), replacing the per-pair host Eigen GEMM
//  at dlpno_pair_data.cu DFpair loop (the largest single CCSD-T2 cost).
//
//  Layouts (row-major):
//    pi_T_stack[idx]   [n × K]   at d_pi_T_stack   + off[idx]        (FULL off)
//    T_meta_dpair[idx] [K × n]   at d_T_meta_dpair + off[idx]-base   (slab off)
//    DF[idx]           [n × n]   → d_DF_scratch → D2H to DF_out[idx]
//  Both pi_T and T_meta per-pair blocks have nocc²·n² elements, so the pi_T
//  offset cumulant indexes both; d_T_meta_dpair is slab-sized.
// ---------------------------------------------------------------------------
bool PiCacheGpu::upload_T_meta_dpair(const std::vector<RowMatXd>& T_meta_dpair)
{
#ifdef GANSU_CPU_ONLY
    (void)T_meta_dpair;
    return false;
#else
    if (!active_ || !stacked_) return false;
    MultiGpuManager::DeviceGuard _guard(device_id_);
    Impl& s = *p_;
    const int N    = N_pair_;
    const int nocc = nocc_;
    if (N <= 0 || nocc <= 0) return false;

    // Per-pair offsets (cumulative nocc²·n²) — identical formula to d_idx_offset.
    std::vector<size_t> off(static_cast<size_t>(N) + 1, 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        off[i + 1] = off[i] + n * n * static_cast<size_t>(nocc)
                                    * static_cast<size_t>(nocc);
    }
    const int    ib         = pair_begin_, ie = pair_end_;
    const size_t slab_base  = off[ib];
    const size_t slab_total = off[ie] - off[ib];
    if (slab_total == 0) return false;

    if (s.d_T_meta_dpair == nullptr) {
        if (cudaMalloc(&s.d_T_meta_dpair, slab_total * sizeof(real_t))
            != cudaSuccess) {
            s.d_T_meta_dpair = nullptr;
            return false;   // OOM → caller keeps the CPU DFpair loop
        }
    }
    if (s.d_DF_scratch == nullptr) {
        const size_t df_bytes =
            static_cast<size_t>(max_n_) * max_n_ * sizeof(real_t);
        if (df_bytes > 0
            && cudaMalloc(&s.d_DF_scratch, df_bytes) != cudaSuccess) {
            s.d_DF_scratch = nullptr;
            return false;
        }
    }
    // Zero so empty-host-T_meta pairs (and padding) contribute exactly 0.
    if (cudaMemset(s.d_T_meta_dpair, 0, slab_total * sizeof(real_t))
        != cudaSuccess) {
        return false;
    }
    // Upload this device's slab pairs only.
    for (int idx = ib; idx < ie; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) continue;
        if (idx >= static_cast<int>(T_meta_dpair.size())
            || T_meta_dpair[idx].size() == 0) continue;
        const size_t cnt = off[idx + 1] - off[idx];   // = nocc²·n²
        if (cudaMemcpy(s.d_T_meta_dpair + (off[idx] - slab_base),
                       T_meta_dpair[idx].data(),
                       cnt * sizeof(real_t), cudaMemcpyHostToDevice)
            != cudaSuccess) {
            return false;
        }
    }
    return true;
#endif
}

void PiCacheGpu::compute_dfpair(std::vector<RowMatXd>& DF_per_pair_out)
{
#ifndef GANSU_CPU_ONLY
    if (!active_ || !stacked_ || p_->d_T_meta_dpair == nullptr
        || p_->d_DF_scratch == nullptr) {
        return;   // not set up → caller's CPU path already filled DF_per_pair
    }
    MultiGpuManager::DeviceGuard _guard(device_id_);
    Impl& s = *p_;
    const int N    = N_pair_;
    const int nocc = nocc_;

    std::vector<size_t> off(static_cast<size_t>(N) + 1, 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        off[i + 1] = off[i] + n * n * static_cast<size_t>(nocc)
                                    * static_cast<size_t>(nocc);
    }
    const int    ib        = pair_begin_, ie = pair_end_;
    const size_t slab_base = off[ib];

    const real_t alpha = -1.0, beta = 0.0;
    for (int idx = ib; idx < ie; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) { DF_per_pair_out[idx].resize(0, 0); continue; }
        DF_per_pair_out[idx].setZero(n, n);
        const int K = static_cast<int>(
            static_cast<size_t>(nocc) * nocc * n);
        // row-major DF[n×n] = pi_T[n×K] · T_meta[K×n]; cuBLAS col-major idiom
        // (= resid_gpu Op-1): first ptr = right factor T_meta (lda=n), second
        // ptr = left factor pi_T (ldb=K), C = DF (ldc=n), alpha=-1 folds the −=.
        const real_t* A_piT   = s.d_pi_T_stack   + off[idx];                // full
        const real_t* B_tmeta = s.d_T_meta_dpair + (off[idx] - slab_base);  // slab
        const cublasStatus_t st = cublasDgemm(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            n, n, K,
            &alpha,
            B_tmeta, n,
            A_piT,   K,
            &beta,
            s.d_DF_scratch, n);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                "PiCacheGpu::compute_dfpair: cublasDgemm failed (status "
                + std::to_string(static_cast<int>(st)) + ")");
        }
        check_cuda_(cudaMemcpy(DF_per_pair_out[idx].data(), s.d_DF_scratch,
                               static_cast<size_t>(n) * n * sizeof(real_t),
                               cudaMemcpyDeviceToHost),
                    "D2H DF_per_pair (compute_dfpair)");
    }
#else
    (void)DF_per_pair_out;
#endif
}

} // namespace gansu
