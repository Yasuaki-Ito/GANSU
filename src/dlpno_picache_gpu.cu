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
    int*     d_setup_j     = nullptr;     // [N_pair] setups[idx].j (Phase 2 scatter)
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
    // Inactive (empty-pair) pi_cache_out rows are 0×0 from construction and are
    // never written; the per-iter loop that re-resizes them to 0×0 is a no-op
    // but its 2.3M-iteration host sweep (Decacene heavy slab) dominated the LMP2
    // picache wall. Zero them ONCE per instance instead.
    bool     inactive_zeroed = false;
    int      max_needed    = 0;
    int*     d_needed_ak    = nullptr;   // [N_act_ij · max_needed]
    int*     d_needed_count = nullptr;   // [N_act_ij]
    real_t*  d_pi_needed    = nullptr;   // [tile_size · max_needed · max_n²]
    real_t*  h_pi_needed    = nullptr;   // pinned mirror of d_pi_needed
    std::vector<std::vector<int>> needed_ikl_host;  // [N_act_ij][count]

    // picache D2H compaction (env GANSU_DLPNO_PICACHE_COMPACT_D2H, LMP2 ragged
    // path). d_pi_needed stores each pi block padded to max_n² but the real
    // data is only n_ij² (avg n_pno ≪ max_n ⇒ ~max_n²/n_ij² ≈ 28× waste). The
    // D2H of the LMP2 picache (to materialise host pi for the host residual) is
    // the dominant per-iter cost. Packing the n_ij² blocks contiguously on
    // device before the D2H cuts the transfer ~28×. Compact slot n (ai's n-th
    // coupling) lives at compact_offset[ai] + n·n_ij². Pure layout ⇒ bit-exact.
    // Built once (offsets depend only on n_pno/count, iter-invariant).
    bool     compact_built    = false;
    real_t*  d_pi_compact     = nullptr; // [Σ count[ai]·n_ij²]
    real_t*  h_pi_compact     = nullptr; // pinned mirror
    size_t*  d_compact_offset = nullptr; // [N_act_ij] cumulative count·n_ij²
    int*     d_ai_n_ij        = nullptr; // [N_act_ij] n_pno of output pair ai
    std::vector<size_t> compact_offset;  // [N_act_ij] host
    size_t   compact_total    = 0;

    // Stage D (offset-CSR) — when non-empty, the ragged coupling buffers
    // (d_needed_ak / d_coupling_ikl / d_coupling_slot_base / d_pi_needed /
    // d_barS_csr) are sized by Σ count[ai] (= total_needed, AVG) instead of
    // N_act_ij·max_needed (MAX, rectangular padding). Each row ai's slots start
    // at needed_offset[ai] instead of ai·max_needed. This is what makes the
    // distance screen actually shrink memory (padding kept buffers at max-
    // coupling size regardless of avg). Populated ONLY in setup_sparse_stacked_
    // (CCSD); LMP2's setup leaves it empty ⇒ the shared compute_pi_needed_ragged_
    // falls back to padded indexing. Pure layout change ⇒ bit-exact.
    std::vector<size_t> needed_offset;   // [N_act_ij+1] cumulative count (host)
    size_t              total_needed = 0;// Σ count[ai] (= needed_offset[N_act_ij])
    size_t*             d_needed_offset = nullptr; // [N_act_ij+1] device mirror

    // Phase 1 (sparse barS) — ragged coupling-list batched-GEMM pointer arrays.
    // Built once (iter-invariant: the ak mapping + base buffer pointers are
    // fixed after allocation, and Y is repacked IN PLACE into d_Y_pad each
    // iter so the pointer values stay valid). Indexed by slot = ai·max_needed+n
    // for n < needed_count[ai]; padded slots are never dereferenced (per-row
    // cublasDgemmBatched uses batchCount = needed_count[ai]).
    //   d_pY_ptrs[slot]    = d_Y_pad    + ak·stride_pair          (stage1 A)
    //   d_pBarS_ptrs[slot] = d_barS_pad + (ai·N_act_kl+ak)·stride (stage1 B / stage2 A)
    //   d_pHalf_ptrs[slot] = d_half_pad + n·stride_pair           (stage1 C / stage2 B)
    //   d_pPi_ptrs[slot]   = d_pi_needed+ slot·stride_pair        (stage2 C)
    bool     coupling_ptrs_built = false;
    real_t** d_pY_ptrs    = nullptr;
    real_t** d_pBarS_ptrs = nullptr;
    real_t** d_pHalf_ptrs = nullptr;
    real_t** d_pPi_ptrs   = nullptr;

    // Phase 1 (sparse barS / Stage 2a) — CSR barS storage built lazily on the
    // first rebuild_needed when sparse_lmp2_. Layout:
    //   d_barS_csr[(ai·max_needed + n)·max_n² + r·max_n + c] = barS[i_ij][i_kl](r,c)
    // for the n-th needed column (i_kl = needed_ikl_host[ai][n]). Replaces the
    // dense d_barS_pad [N_act_ij·N_act_kl·max_n²] which is NOT allocated in
    // sparse mode. Identical padded bytes to the dense block ⇒ bit-exact.
    bool     barS_csr_built = false;
    real_t*  d_barS_csr     = nullptr;

    // Phase 2 speed — coupling-contiguous Y gather so the ragged pi compute
    // uses per-row cublasDgemmStridedBatched (dense speed) instead of the
    // slower pointer-array cublasDgemmBatched. d_Y_coupling[(ai·max_needed+
    // slot)·max_n²] = d_Y_pad[ak] (ak = active-kl pos of the slot's coupling
    // pair). Re-gathered each iter (Y changes); same size as d_pi_needed.
    real_t*  d_Y_coupling   = nullptr;

    // Phase 2 (CCSD sparse) — extra device maps for the pi_T_stack scatter.
    //   d_active_i_ij[ai]                 = orig pair idx of active row ai
    //   d_coupling_ikl[ai·max_needed+n]   = orig pair idx of the n-th coupling
    // (needed_ikl_host stores the same on host; max_needed is the coupling cap).
    int*     d_active_i_ij  = nullptr;
    int*     d_coupling_ikl = nullptr;

    // Stage D (D1a) — sparse pi_T_stack kl-slot list + slot maps. These replace
    // the dense nocc²-(k,l) pi_T_stack layout with a per-pair coupling-list
    // (kl-slot) layout: each coupling pair idx_kl=(p,q) expands to the ordered
    // (k,l) slots (p·nocc+q) and (q·nocc+p) (one if p==q). All indexed by ORIG
    // pair idx (small ints over N_pair) so ResidGpu consumers (which index by
    // orig idx) can read them directly. Built in setup_sparse_stacked_; unused
    // until D1b switches the storage + consumers. None are large (ints).
    bool     kl_slot_built  = false;
    int      max_slots      = 0;        // max n_slots over pairs (pad/cap)
    int*     d_n_slots      = nullptr;  // [N_pair] slot count per pair (0 = inactive)
    size_t*  d_slot_offset  = nullptr;  // [N_pair+1] cumulative n_slots (kl_slot CSR)
    int*     d_kl_slot      = nullptr;  // [Σ n_slots] kl(=k·nocc+l) per slot
    size_t*  d_idx_offset_sparse = nullptr; // [N_pair+1] cumulative n_ij²·n_slots (pi_T sparse)
    int*     d_slot_jcol    = nullptr;  // [N_pair·nocc] slot of (k,j) or -1
    int*     d_slot_irow    = nullptr;  // [N_pair·nocc] slot of (i,l) or -1
    int*     d_slot_icol    = nullptr;  // [N_pair·nocc] slot of (k,i) or -1 (j-side slice)
    int*     d_slot_jrow    = nullptr;  // [N_pair·nocc] slot of (j,l) or -1 (j-side slice)
    int*     d_coupling_slot_base = nullptr; // [N_act_ij·max_needed] local slot of
                                             // coupling n's canonical (p,q); +1 = transpose
    size_t   pi_T_sparse_total = 0;     // Σ n_ij²·n_slots (full N_pair)
    size_t   pi_T_sparse_slab_base  = 0; // Σ_{i<pair_begin} n_ij²·n_slots
    size_t   pi_T_sparse_slab_total = 0; // slab's own elem count (sparse)
    // Host copies of the kl-slot list (for the sparse T_meta_dpair gather in
    // upload_T_meta_dpair, which re-orders the dense host T_meta into kl-slot
    // device layout). Small ints.
    std::vector<int>    h_kl_slot_host;     // [Σ n_slots] kl per slot
    std::vector<size_t> h_slot_offset_host; // [N_pair+1] cumulative n_slots
    std::vector<int>    h_n_slots_host;     // [N_pair]

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
        if (d_setup_j)     cudaFree(d_setup_j);
        if (d_n_pno)       cudaFree(d_n_pno);
        if (d_idx_offset)  cudaFree(d_idx_offset);
        if (d_pi_T_stack)  cudaFree(d_pi_T_stack);
        if (h_pi_T_stack)  cudaFreeHost(h_pi_T_stack);
        if (d_active_ij_pos) cudaFree(d_active_ij_pos);
        if (d_active_kl_pos) cudaFree(d_active_kl_pos);
        if (d_active_i_kl)   cudaFree(d_active_i_kl);
        if (d_needed_ak)     cudaFree(d_needed_ak);
        if (d_needed_count)  cudaFree(d_needed_count);
        if (d_needed_offset) cudaFree(d_needed_offset);
        if (d_pi_needed)     cudaFree(d_pi_needed);
        if (h_pi_needed)     cudaFreeHost(h_pi_needed);
        if (d_pi_compact)    cudaFree(d_pi_compact);
        if (h_pi_compact)    cudaFreeHost(h_pi_compact);
        if (d_compact_offset) cudaFree(d_compact_offset);
        if (d_ai_n_ij)       cudaFree(d_ai_n_ij);
        if (d_pY_ptrs)       cudaFree(d_pY_ptrs);
        if (d_pBarS_ptrs)    cudaFree(d_pBarS_ptrs);
        if (d_pHalf_ptrs)    cudaFree(d_pHalf_ptrs);
        if (d_pPi_ptrs)      cudaFree(d_pPi_ptrs);
        if (d_barS_csr)      cudaFree(d_barS_csr);
        if (d_Y_coupling)    cudaFree(d_Y_coupling);
        if (d_active_i_ij)   cudaFree(d_active_i_ij);
        if (d_coupling_ikl)  cudaFree(d_coupling_ikl);
        if (d_n_slots)       cudaFree(d_n_slots);
        if (d_slot_offset)   cudaFree(d_slot_offset);
        if (d_kl_slot)       cudaFree(d_kl_slot);
        if (d_idx_offset_sparse) cudaFree(d_idx_offset_sparse);
        if (d_slot_jcol)     cudaFree(d_slot_jcol);
        if (d_slot_irow)     cudaFree(d_slot_irow);
        if (d_slot_icol)     cudaFree(d_slot_icol);
        if (d_slot_jrow)     cudaFree(d_slot_jrow);
        if (d_coupling_slot_base) cudaFree(d_coupling_slot_base);
        if (d_T_meta_dpair)  cudaFree(d_T_meta_dpair);
        if (d_DF_scratch)    cudaFree(d_DF_scratch);
        d_T_meta_dpair = nullptr;
        d_DF_scratch   = nullptr;
        d_barS_pad = d_Y_pad = d_Y_pad_T = d_half_pad = d_pi_pad = nullptr;
        h_Y_pad = h_pi_pad = nullptr;
        d_pair_lookup = d_setup_i = d_setup_j = d_n_pno = nullptr;
        d_active_i_ij = d_coupling_ikl = nullptr;
        d_n_slots = d_kl_slot = d_slot_jcol = d_slot_irow = nullptr;
        d_slot_icol = d_slot_jrow = nullptr;
        d_slot_offset = d_idx_offset_sparse = nullptr;
        d_coupling_slot_base = nullptr;
        d_idx_offset = nullptr;
        d_pi_T_stack = nullptr;
        h_pi_T_stack = nullptr;
        d_active_ij_pos = d_active_kl_pos = nullptr;
        d_active_i_kl = nullptr;
        d_needed_ak = d_needed_count = nullptr;
        d_needed_offset = nullptr;
        d_pi_needed = nullptr;
        h_pi_needed = nullptr;
        d_pi_compact = nullptr;
        h_pi_compact = nullptr;
        d_compact_offset = nullptr;
        d_ai_n_ij = nullptr;
        d_pY_ptrs = d_pBarS_ptrs = d_pHalf_ptrs = d_pPi_ptrs = nullptr;
        d_barS_csr = nullptr;
        d_Y_coupling = nullptr;
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

// Phase 2 (CCSD sparse barS) — scatter the ragged pi (coupling slots in
// d_pi_needed) into the DENSE d_pi_T_stack at the (k·nocc+l) positions.
// Replaces pack_pi_T_stack_kernel for the sparse stacked path: instead of a
// grid over (i_ij, nocc²) reading a dense d_pi_pad column, this grid is
// (ai, slot) — for each coupling block (output pair i_ij, canonical pair
// idx_kl=(p,q)) it writes BOTH oriented slots kl1=p·nocc+q (canon) and
// kl2=q·nocc+p (transpose). Non-coupling slots are left at the value set by
// the prior cudaMemset (zero). The src byte layout of d_pi_needed[slot] is
// IDENTICAL to what dense d_pi_pad[(ai,ak)] would hold (Phase 1 verified), so
// the src indexing (src[a·max_n+d] canon / src[d·max_n+a] transpose) matches
// pack_pi_T_stack_kernel exactly ⇒ bit-exact for every coupling block.
__global__ void scatter_pi_T_coupling_kernel(
    const real_t* __restrict__ d_pi_needed,     // ragged: (ai·max_coupling+slot)·max_n²
    const int*    __restrict__ d_active_i_ij,   // ai → orig i_ij
    const int*    __restrict__ d_coupling_ikl,  // (ai·max_coupling+slot) → orig idx_kl
    const int*    __restrict__ d_coupling_count,// [N_act_ij]
    const int*    __restrict__ d_setup_i,       // idx → setups[idx].i (= p)
    const int*    __restrict__ d_setup_j,       // idx → setups[idx].j (= q)
    const int*    __restrict__ d_n_pno,
    const size_t* __restrict__ d_idx_offset,    // dense pi_T offsets (n_ij²·nocc²)
    real_t*       __restrict__ d_pi_T_stack,
    int nocc, int max_n, int max_coupling, int N_act_ij,
    // Stage D sparse-output extras (used when pitstack_sparse != 0):
    int pitstack_sparse,
    const size_t* __restrict__ d_idx_offset_sparse, // sparse pi_T offsets
    const int*    __restrict__ d_n_slots,           // [N_pair] slots per pair
    const int*    __restrict__ d_coupling_slot_base,// (ai·max_coupling+slot) → canon local slot
    // Stage D offset-CSR: when non-null, the ragged companions (d_pi_needed,
    // d_coupling_ikl, d_coupling_slot_base) are indexed by needed_offset[ai]+slot
    // instead of ai·max_coupling+slot. d_coupling_count still gives the per-row
    // valid slot count; the grid.y span (max_coupling) is the padded max.
    const size_t* __restrict__ d_needed_offset)
{
    const int ai   = blockIdx.x;
    const int slot = blockIdx.y;
    if (ai >= N_act_ij) return;
    if (slot >= d_coupling_count[ai]) return;

    const int i_ij = d_active_i_ij[ai];
    const int n_ij = d_n_pno[i_ij];
    if (n_ij == 0) return;

    const size_t cslot = d_needed_offset
        ? (d_needed_offset[ai] + slot)
        : (static_cast<size_t>(ai) * max_coupling + slot);
    const int idx_kl = d_coupling_ikl[cslot];
    const int p = d_setup_i[idx_kl];
    const int q = d_setup_j[idx_kl];

    const real_t* src = d_pi_needed
        + cslot * static_cast<size_t>(max_n) * static_cast<size_t>(max_n);

    // Output position: dense (k·nocc+l) layout, or sparse (kl-slot) layout.
    size_t out_base, out_row_stride;
    int pos1, pos2;  // canon, transpose column positions (× n_ij)
    if (pitstack_sparse) {
        out_base       = d_idx_offset_sparse[i_ij];
        out_row_stride = static_cast<size_t>(d_n_slots[i_ij]) * n_ij;
        pos1 = d_coupling_slot_base[cslot];   // canonical local slot
        pos2 = pos1 + 1;                       // transpose slot (if p!=q)
    } else {
        out_base       = d_idx_offset[i_ij];
        out_row_stride = static_cast<size_t>(nocc) * nocc * n_ij;
        pos1 = p * nocc + q;                   // canon kl
        pos2 = q * nocc + p;                   // transpose kl
    }

    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            const real_t v_canon = src[static_cast<size_t>(a) * max_n + d];
            d_pi_T_stack[out_base + static_cast<size_t>(a) * out_row_stride
                         + static_cast<size_t>(pos1) * n_ij + d] = v_canon;
            if (p != q) {
                const real_t v_t = src[static_cast<size_t>(d) * max_n + a];
                d_pi_T_stack[out_base + static_cast<size_t>(a) * out_row_stride
                             + static_cast<size_t>(pos2) * n_ij + d] = v_t;
            }
        }
    }
}

// Phase 2 speed — gather ONE output row's coupling Y into a contiguous,
// reused buffer so the ragged pi compute uses per-row cublasDgemmStridedBatched
// (uniform stride, dense speed) instead of the slower pointer-array batched.
// d_Y_row[slot] = d_Y_pad[d_needed_ak_row[slot]] for slot < cnt. The buffer is
// just max_needed × max_n² (one row), not the full N_act_ij grid — keeps the
// memory win intact.
__global__ void gather_Y_row_kernel(
    const real_t* __restrict__ d_Y_pad,
    const int*    __restrict__ d_needed_ak_row,
    int cnt,
    real_t*       __restrict__ d_Y_row,
    long long stride_pair)
{
    const int slot = blockIdx.x;
    if (slot >= cnt) return;
    const int ak = d_needed_ak_row[slot];
    if (ak < 0) return;
    const real_t* src = d_Y_pad + static_cast<size_t>(ak) * stride_pair;
    real_t* dst = d_Y_row + static_cast<size_t>(slot) * stride_pair;
    for (long long e = threadIdx.x; e < stride_pair; e += blockDim.x)
        dst[e] = src[e];
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

// picache D2H compaction — strip the max_n padding: pack the real n_ij² sub-
// block out of each max_n²-padded slot of d_pi_needed into a contiguous compact
// buffer (so the D2H transfers n_ij² instead of max_n² per block, ~28× less).
// Grid (ai, n) — one block per coupling slot. Compact slot n of row ai lives at
// d_compact_offset[ai] + n·n_ij². d_pi_needed uses LMP2 padded indexing
// (ai·max_needed + n)·max_n². n_ij = n_pno of the output pair ai. bit-exact.
__global__ void pack_pi_compact_kernel(
    const real_t* __restrict__ d_pi_needed,
    real_t*       __restrict__ d_pi_compact,
    const size_t* __restrict__ d_compact_offset,
    const int*    __restrict__ d_ai_n_ij,
    const int*    __restrict__ d_needed_count,
    int max_needed, int max_n, long long stride_pair, int N_act_ij)
{
    const int ai = blockIdx.x;
    const int n  = blockIdx.y;          // coupling slot within row ai
    if (ai >= N_act_ij) return;
    const int n_ij = d_ai_n_ij[ai];
    if (n_ij == 0) return;
    if (n >= d_needed_count[ai]) return;
    const real_t* src = d_pi_needed
        + (static_cast<size_t>(ai) * max_needed + n) * stride_pair;
    real_t* dst = d_pi_compact
        + d_compact_offset[ai]
        + static_cast<size_t>(n) * n_ij * n_ij;
    const int nn = n_ij * n_ij;
    for (int idx = threadIdx.x; idx < nn; idx += blockDim.x) {
        const int r = idx / n_ij;
        const int c = idx - r * n_ij;
        dst[idx] = src[static_cast<size_t>(r) * max_n + c];
    }
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
                       int device_id,
                       const std::vector<int>* setup_j_per_pair)
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
    if (setup_j_per_pair)   setup_j_per_pair_  = *setup_j_per_pair;

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

    // Phase 1 (sparse barS / Stage 2a) — enable the LMP2 sparse storage path
    // for the non-stacked (LMP2) instance when the env flag is set. The dense
    // d_barS_pad / d_pi_pad are skipped below; a CSR barS is built lazily on
    // the first rebuild_needed(). want_stacked instances (CCSD) keep the dense
    // path (their coupling is the full (k,l) superset, handled in a later
    // phase). Requires active rows.
    {
        // Default-on (bit-exact: coupling = needed_ikl is the exact set the
        // LMP2 residual reads). Must track the host barS_cache build gate in
        // dlpno_pair_data.cu (same env) so the device CSR and the host cache
        // agree on which blocks exist. Opt-out: GANSU_DLPNO_LMP2_BARS_SPARSE=0.
        const char* e = std::getenv("GANSU_DLPNO_LMP2_BARS_SPARSE");
        sparse_lmp2_ = !(e && e[0] == '0') && !want_stacked && N_act_ij_ > 0;
        // Phase 2: CCSD (stacked) sparse barS. Requires setup_j (for the
        // pi_T_stack scatter). Coupling list is supplied per rebuild_with_stack.
        const char* ec = std::getenv("GANSU_DLPNO_CCSD_BARS_SPARSE");
        sparse_stacked_ = (ec && ec[0] == '1') && want_stacked
                          && !setup_j_per_pair_.empty() && N_act_ij_ > 0;
        // Stage D: sparse pi_T_stack storage (implies sparse_stacked_).
        const char* ep = std::getenv("GANSU_DLPNO_CCSD_PITSTACK_SPARSE");
        pitstack_sparse_ = (ep && ep[0] == '1') && sparse_stacked_;
    }
    // Unified gate: skip the dense d_barS_pad / d_pi_pad in either sparse mode.
    const bool sparse_barS = sparse_lmp2_ || sparse_stacked_;

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
        // Phase 1 (sparse barS): the dense d_barS_pad + d_pi_pad are NOT
        // allocated; a single tile is forced (tile_size_ = N_act_ij_) and the
        // CSR barS / d_pi_needed (both ≈ N_act_kl/max_needed × smaller) are
        // built lazily in rebuild_needed. So the dense fixed-need check (which
        // would false-OOM on Decacene) is bypassed; only the small fixed
        // buffers (d_Y_pad ×2 + d_Y_pad_T) are required here.
        const size_t margin = (size_t)256 * 1024 * 1024;  // 256 MB
        if (sparse_barS) {
            // Sparse: dense d_barS_pad + d_pi_pad are skipped; single tile is
            // forced. The CSR barS / d_pi_needed (≈ N_act_kl/max_needed ×
            // smaller) are built lazily. For the stacked (CCSD) case the dense
            // d_pi_T_stack (+ slab pinned mirror) IS still allocated, so count
            // bytes_pi_T here. Bypasses the dense fixed-need check that would
            // false-OOM on Decacene.
            // Stage D: when pitstack_sparse_, the dense d_pi_T_stack is NOT
            // allocated (sparse one built lazily in setup, much smaller), so
            // exclude bytes_pi_T from the ctor budget.
            const size_t fixed_need_sparse = bytes_Y_compact   // d_Y_pad
                                           + bytes_Y_T_full     // d_Y_pad_T
                                           + bytes_Y_compact    // d_half_pad
                                           + (pitstack_sparse_ ? 0 : bytes_pi_T)
                                           + margin;
            if (fixed_need_sparse > free_b || N_act_ij_ == 0) {
                delete p_; p_ = nullptr; active_ = false; return;
            }
            tile_size_ = N_act_ij_;   // single tile; d_pi_pad not allocated
        } else {
        // Fixed needs (everything except d_pi_pad):
        //   d_barS_pad + d_Y_pad + d_Y_pad_T (full) + d_half_pad + d_pi_T_stack + margin
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
        }  // end !sparse_lmp2_ tiling branch
    }
    const size_t bytes_pi_tile =
        static_cast<size_t>(tile_size_) * per_pi_row_bytes;

    const auto t_mem_probe = std::chrono::steady_clock::now();

    try {
        // Phase 1/2 (sparse barS): skip the dense d_barS_pad [N_act_ij·N_act_kl·
        // max_n²] and the d_pi_pad tile — both replaced by the smaller CSR
        // d_barS_csr + d_pi_needed built lazily (rebuild_needed / rebuild_with_stack).
        if (!sparse_barS) {
            check_cuda_(cudaMalloc(&s.d_barS_pad, bytes_barS_compact),
                        "cudaMalloc d_barS_pad (compact)");
            check_cuda_(cudaMalloc(&s.d_pi_pad,   bytes_pi_tile),
                        "cudaMalloc d_pi_pad (tile)");
        }
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
            // Stage D: when pitstack_sparse_, d_pi_T_stack + h_pi_T_stack use
            // the sparse (kl-slot) sizing and are allocated lazily in
            // setup_sparse_stacked_ (pi_T_sparse_total unknown until then).
            if (!pitstack_sparse_) {
                check_cuda_(cudaMalloc(&s.d_pi_T_stack, bytes_pi_T),
                            "cudaMalloc d_pi_T_stack");
                // Step S9 / lazy: the pinned host mirror h_pi_T_stack is the
                // destination of the per-iter pi_T_stack D2H. That D2H is now
                // SKIPPED whenever the GPU consumers (oooo + DFpair) read the
                // device buffer directly (skip_pi_T_stack_host) — i.e. for all
                // all-GPU-active runs it is never used. Allocating its ~slab-size
                // pinned buffer here cost a cudaMallocHost that serialised on the
                // kernel mmap lock across the N_gpu ctors for nothing. Defer it
                // to the first rebuild_with_stack that actually performs the D2H
                // (partial activation); all-active never allocates it.
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
            // Phase 2 (CCSD sparse): setup_j for the pi_T_stack scatter.
            if (sparse_stacked_ && !setup_j_per_pair_.empty()) {
                check_cuda_(cudaMalloc(&s.d_setup_j,
                                       static_cast<size_t>(N_pair_) * sizeof(int)),
                            "cudaMalloc d_setup_j");
                check_cuda_(cudaMemcpy(s.d_setup_j, setup_j_per_pair_.data(),
                                       static_cast<size_t>(N_pair_) * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "H2D setup_j");
                check_cuda_(cudaMalloc(&s.d_active_i_ij,
                                       static_cast<size_t>(std::max(1, N_act_ij_))
                                           * sizeof(int)),
                            "cudaMalloc d_active_i_ij");
                check_cuda_(cudaMemcpy(s.d_active_i_ij, active_i_ij_.data(),
                                       static_cast<size_t>(N_act_ij_) * sizeof(int),
                                       cudaMemcpyHostToDevice),
                            "H2D active_i_ij");
            }
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
    // Phase 1/2 (sparse barS): skipped entirely — d_barS_pad is not allocated;
    // the CSR barS is built lazily (rebuild_needed / rebuild_with_stack) from
    // barS_cache_ref_.
    if (!sparse_barS) {
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
            // Pageable staging (was pinned cudaMallocHost): one-time dense
            // barS upload. The pinned alloc serialised on the kernel mmap lock
            // across the N_gpu picache ctors — the barS_h2d cost is mostly that
            // contention, not the transfer. Pageable removes it; the one-time
            // H2D is marginally slower but negligible. Value-init supplies the
            // zeros the defensively-empty barS_cache blocks rely on, and RAII
            // frees it (also exception-safe vs the old cudaFreeHost, which the
            // check_cuda_ throw paths below would have leaked).
            std::vector<real_t> h_barS_flat_vec(flat_size, real_t(0));
            real_t* h_barS_flat = h_barS_flat_vec.data();

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
            // h_barS_flat is now a pageable std::vector (RAII-freed).
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
                // [Stage 0] coupling sparsity: dense barS grid is
                // N_act_ij·N_act_kl blocks; the LMP2 residual only needs
                // max_needed (~2·nocc) columns per row. The byte figures are
                // the dense d_barS_pad footprint vs a sparse (CSR) barS holding
                // only the needed columns (the Stage 2 target storage).
                const double blk_mb = static_cast<double>(max_n_)
                    * max_n_ * sizeof(real_t) / (1024.0 * 1024.0);
                const double dense_barS_mb = static_cast<double>(N_act_ij_)
                    * N_act_kl_ * blk_mb;
                const double sparse_barS_mb = static_cast<double>(N_act_ij_)
                    * p_->max_needed * blk_mb;
                std::printf("[picache-PROF dev=%d] rebuild calls=%d  "
                            "pack_Y=%.3fs  tile_GEMM=%.3fs  D2H=%.3fs  "
                            "scatter=%.3fs  (slab [%d,%d) N_act_ij=%d "
                            "N_act_kl=%d max_needed=%d  barS dense=%.0f MB "
                            "sparse=%.0f MB (%.1fx))\n",
                            device_id_, p_->n_rebuild, p_->t_packY,
                            p_->t_gemm, p_->t_d2h, p_->t_scatter,
                            pair_begin_, pair_end_, N_act_ij_, N_act_kl_,
                            p_->max_needed, dense_barS_mb, sparse_barS_mb,
                            sparse_barS_mb > 0.0
                                ? dense_barS_mb / sparse_barS_mb : 0.0);
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
                // Sparse barS_cache (LMP2 default-on / CCSD opt-in): non-coupling
                // (i_ij, i_kl) blocks are left 0×0 by the host build. The LMP2
                // residual never reads those pi_cache entries, so leaving them
                // empty here is bit-exact — and it keeps the CPU fallback path
                // (this routine) from dereferencing an unbuilt block.
                if (barS.rows() == 0 || barS.cols() == 0) {
                    pi_row[i_kl].resize(0, 0);
                    continue;
                }
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
    // Phase 1 (Stage 2a): sparse mode does not allocate the dense d_barS_pad
    // that this full-column rebuild needs. It only ever runs via rebuild_needed
    // (picache_gather, default on). If a caller disables gather while sparse is
    // on, fall back to the CPU path (uses the host barS_cache, correct).
    if (sparse_lmp2_) {
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

    // Phase 1 (sparse barS) — ragged coupling-list batched GEMM. Computes pi
    // ONLY for the needed_ikl columns directly into d_pi_needed (no dense
    // N_act_kl GEMM + gather). Bit-exact for every consumed block. Requires a
    // single tile (tile_size_ >= N_act_ij_); falls back to the dense path
    // otherwise. Opt-in, default off = the dense path (bit-identical).
    static const bool coupling_ragged_env = []() {
        const char* e = std::getenv("GANSU_DLPNO_LMP2_BARS_RAGGED_GEMM");
        return e && e[0] == '1';
    }();
    // sparse_lmp2_ (Stage 2a) implies the ragged path (the dense d_barS_pad /
    // d_pi_pad are not allocated, so the dense compute_pi_tile_ cannot run).
    // sparse_lmp2_ forces tile_size_ = N_act_ij_, so the single-tile gate holds.
    const bool use_ragged =
        (coupling_ragged_env || sparse_lmp2_) && (tile_size_ >= N_act_ij_);

    // picache D2H compaction (default-on; opt out GANSU_DLPNO_PICACHE_COMPACT_D2H
    // =0). Packs the real n_ij² blocks out of the max_n²-padded d_pi_needed
    // before the D2H (~max_n²/n_ij² ≈ 28× less host traffic). LMP2 ragged path
    // only. Bit-exact (pure layout). When active, the padded host mirror
    // h_pi_needed is never read ⇒ its (large) pinned alloc is skipped below.
    static const bool compact_d2h = []() {
        const char* e = std::getenv("GANSU_DLPNO_PICACHE_COMPACT_D2H");
        return !(e && e[0] == '0');
    }();
    const bool compact_active = compact_d2h && use_ragged;

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
            // h_pi_needed (padded host mirror) is only read by the non-compact
            // D2H path; skip its large pinned alloc when compaction is active.
            if (!compact_active)
                check_cuda_(cudaMallocHost(&s.h_pi_needed, bytes_needed),
                            "cudaMallocHost h_pi_needed");
        }

        // Phase 1 (Stage 2a) — lazy CSR barS build. Pack ONLY the needed
        // coupling blocks (max_needed per row) into d_barS_csr, with the exact
        // same max_n-padded layout the dense scatter_barS_kernel produces, so
        // the ragged GEMM is bit-exact w.r.t. the dense path. barS_cache_ref_
        // is the host barS_cache passed to the constructor (alive for the iter
        // loop scope). Built whenever the ragged compute is used (it reads
        // d_barS_csr), i.e. RAGGED_GEMM or SPARSE; with SPARSE the dense
        // d_barS_pad is additionally skipped (ctor) for the memory win.
        if (use_ragged && !s.barS_csr_built) {
            const auto& barS_cache = *barS_cache_ref_;
            const size_t csr_elems =
                static_cast<size_t>(N_act_ij_) * s.max_needed
                * static_cast<size_t>(stride_pair);
            real_t* h_csr = nullptr;
            check_cuda_(cudaMallocHost(&h_csr, csr_elems * sizeof(real_t)),
                        "cudaMallocHost h_barS_csr");
            std::memset(h_csr, 0, csr_elems * sizeof(real_t));
            #pragma omp parallel for schedule(static)
            for (int ai = 0; ai < N_act_ij_; ++ai) {
                const int i_ij = active_i_ij_[ai];
                const int n_ij = n_pno_[i_ij];
                const std::vector<int>& need = s.needed_ikl_host[ai];
                for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                    const int i_kl = need[n];
                    const int n_kl = n_pno_[i_kl];
                    const RowMatXd& bs = barS_cache[i_ij][i_kl];
                    if (bs.rows() == 0 || bs.cols() == 0) continue;
                    real_t* dst = h_csr
                        + (static_cast<size_t>(ai) * s.max_needed + n)
                          * static_cast<size_t>(stride_pair);
                    const real_t* src = bs.data();
                    for (int r = 0; r < n_ij; ++r)
                        std::memcpy(dst + static_cast<size_t>(r) * max_n,
                                    src + static_cast<size_t>(r) * n_kl,
                                    static_cast<size_t>(n_kl) * sizeof(real_t));
                }
            }
            check_cuda_(cudaMalloc(&s.d_barS_csr, csr_elems * sizeof(real_t)),
                        "cudaMalloc d_barS_csr");
            check_cuda_(cudaMemcpy(s.d_barS_csr, h_csr,
                                   csr_elems * sizeof(real_t),
                                   cudaMemcpyHostToDevice), "H2D d_barS_csr");
            cudaFreeHost(h_csr);
            s.barS_csr_built = true;
        }

        s.needed_built = true;
    }

    // Phase 1 — one-time build of the ragged GEMM pointer arrays. Depends on
    // the needed map (d_pi_needed alloc, max_needed) built above, and on the
    // fixed device base buffers (d_Y_pad / d_barS_pad / d_half_pad). Y is
    // repacked in place each iter, so these pointers stay valid across iters.
    if (use_ragged && !s.coupling_ptrs_built && N_act_ij_ > 0) {
        const size_t ns = static_cast<size_t>(N_act_ij_) * s.max_needed;
        std::vector<real_t*> hY(ns, nullptr), hB(ns, nullptr),
                             hH(ns, nullptr), hP(ns, nullptr);
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const std::vector<int>& need = s.needed_ikl_host[ai];
            for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                const int ak = active_kl_pos_[need[n]];
                const size_t slot = static_cast<size_t>(ai) * s.max_needed + n;
                hY[slot] = s.d_Y_pad
                         + static_cast<size_t>(ak) * stride_pair;
                // barS source: CSR slot (sparse mode) or dense (ai,ak) block.
                hB[slot] = sparse_lmp2_
                         ? (s.d_barS_csr + slot * stride_pair)
                         : (s.d_barS_pad
                            + (static_cast<size_t>(ai) * N_act_kl_
                               + static_cast<size_t>(ak)) * stride_pair);
                hH[slot] = s.d_half_pad
                         + static_cast<size_t>(n) * stride_pair;
                hP[slot] = s.d_pi_needed + slot * stride_pair;
            }
        }
        const size_t bytes_ptrs = std::max<size_t>(1, ns) * sizeof(real_t*);
        check_cuda_(cudaMalloc(&s.d_pY_ptrs,    bytes_ptrs), "cudaMalloc d_pY_ptrs");
        check_cuda_(cudaMalloc(&s.d_pBarS_ptrs, bytes_ptrs), "cudaMalloc d_pBarS_ptrs");
        check_cuda_(cudaMalloc(&s.d_pHalf_ptrs, bytes_ptrs), "cudaMalloc d_pHalf_ptrs");
        check_cuda_(cudaMalloc(&s.d_pPi_ptrs,   bytes_ptrs), "cudaMalloc d_pPi_ptrs");
        check_cuda_(cudaMemcpy(s.d_pY_ptrs, hY.data(), ns * sizeof(real_t*),
                               cudaMemcpyHostToDevice), "H2D d_pY_ptrs");
        check_cuda_(cudaMemcpy(s.d_pBarS_ptrs, hB.data(), ns * sizeof(real_t*),
                               cudaMemcpyHostToDevice), "H2D d_pBarS_ptrs");
        check_cuda_(cudaMemcpy(s.d_pHalf_ptrs, hH.data(), ns * sizeof(real_t*),
                               cudaMemcpyHostToDevice), "H2D d_pHalf_ptrs");
        check_cuda_(cudaMemcpy(s.d_pPi_ptrs, hP.data(), ns * sizeof(real_t*),
                               cudaMemcpyHostToDevice), "H2D d_pPi_ptrs");
        s.coupling_ptrs_built = true;
    }

    // picache D2H compaction (LMP2 ragged path). One-time: per-row compact
    // offset (Σ count·n_ij²) + n_ij table + compact device/pinned buffers.
    if (compact_active && !s.compact_built && N_act_ij_ > 0) {
        s.compact_offset.assign(static_cast<size_t>(N_act_ij_), 0);
        std::vector<int> h_ai_n_ij(static_cast<size_t>(N_act_ij_), 0);
        size_t off = 0;
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const int i_ij = active_i_ij_[ai];
            const int n_ij = n_pno_[i_ij];
            const int cnt  = static_cast<int>(s.needed_ikl_host[ai].size());
            h_ai_n_ij[ai] = n_ij;
            s.compact_offset[ai] = off;
            off += static_cast<size_t>(cnt) * n_ij * n_ij;
        }
        s.compact_total = off;
        check_cuda_(cudaMalloc(&s.d_compact_offset,
                               sizeof(size_t) * N_act_ij_),
                    "cudaMalloc d_compact_offset");
        check_cuda_(cudaMemcpy(s.d_compact_offset, s.compact_offset.data(),
                               sizeof(size_t) * N_act_ij_,
                               cudaMemcpyHostToDevice), "H2D d_compact_offset");
        check_cuda_(cudaMalloc(&s.d_ai_n_ij, sizeof(int) * N_act_ij_),
                    "cudaMalloc d_ai_n_ij");
        check_cuda_(cudaMemcpy(s.d_ai_n_ij, h_ai_n_ij.data(),
                               sizeof(int) * N_act_ij_,
                               cudaMemcpyHostToDevice), "H2D d_ai_n_ij");
        if (s.compact_total > 0) {
            check_cuda_(cudaMalloc(&s.d_pi_compact,
                                   s.compact_total * sizeof(real_t)),
                        "cudaMalloc d_pi_compact");
            check_cuda_(cudaMallocHost(&s.h_pi_compact,
                                       s.compact_total * sizeof(real_t)),
                        "cudaMallocHost h_pi_compact");
        }
        s.compact_built = true;
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

    // Inactive-pair rows → 0×0 (those i_ij are never written below). One-time:
    // the blocks are 0×0 from construction and stay 0×0 across iters, so this
    // O(empty·N_pair) host sweep only needs to run once per instance (it was
    // ~90% of the Decacene LMP2 picache wall when re-run every iter).
    if (!s.inactive_zeroed) {
        for (long long i_ij = pair_begin_; i_ij < pair_end_; ++i_ij) {
            if (n_pno_[i_ij] == 0) {
                for (int i_kl = 0; i_kl < N_pair_; ++i_kl)
                    pi_cache_out[i_ij][i_kl].resize(0, 0);
            }
        }
        s.inactive_zeroed = true;
    }

    for (int tile_start = 0; tile_start < N_act_ij_;
             tile_start += tile_size_) {
        const int tile_end  = std::min(tile_start + tile_size_, N_act_ij_);
        const int tile_rows = tile_end - tile_start;

        if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
        if (use_ragged) {
            // Ragged path: per-row pointer-array batched GEMM writes pi for the
            // needed columns straight into d_pi_needed — no dense GEMM, no
            // gather kernel. (Single tile guaranteed by use_ragged gate.)
            compute_pi_needed_ragged_(tile_start, tile_end);
        } else {
            compute_pi_tile_(tile_start, tile_end);
        }
        if (picache_prof) {
            cudaDeviceSynchronize();
            s.t_gemm += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pc_t0).count();
        }

        // Gather the needed columns of this tile into the compact buffer.
        // (Skipped in the ragged path: d_pi_needed is already filled.)
        const int n_slots = tile_rows * s.max_needed;
        if (!use_ragged && n_slots > 0) {
            gather_needed_kernel<<<n_slots, 256>>>(
                s.d_pi_pad, s.d_pi_needed, s.d_needed_ak, s.d_needed_count,
                tile_start, tile_rows, s.max_needed, stride_pair, N_act_kl_);
            check_cuda_(cudaGetLastError(), "gather_needed_kernel launch");
        }

        if (compact_active) {
            // Compact path (single tile): pack the real n_ij² sub-blocks out of
            // the max_n²-padded d_pi_needed on device, then D2H the compact
            // buffer (~max_n²/n_ij² ≈ 28× less than the padded transfer). On the
            // null stream so it is ordered after the cuBLAS GEMM and before D2H.
            dim3 grid(static_cast<unsigned>(N_act_ij_),
                      static_cast<unsigned>(s.max_needed));
            pack_pi_compact_kernel<<<grid, 128>>>(
                s.d_pi_needed, s.d_pi_compact, s.d_compact_offset,
                s.d_ai_n_ij, s.d_needed_count,
                s.max_needed, max_n, stride_pair, N_act_ij_);
            check_cuda_(cudaGetLastError(), "pack_pi_compact_kernel launch");

            if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
            if (s.compact_total > 0)
                check_cuda_(cudaMemcpy(s.h_pi_compact, s.d_pi_compact,
                                       s.compact_total * sizeof(real_t),
                                       cudaMemcpyDeviceToHost),
                            "D2H d_pi_compact");
            if (picache_prof) s.t_d2h += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pc_t0).count();

            // Host scatter: compact blocks are already n_ij×n_ij contiguous, so
            // each is a single memcpy (no per-row max_n stride).
            if (picache_prof) pc_t0 = std::chrono::steady_clock::now();
            #pragma omp parallel for schedule(static)
            for (int ai = 0; ai < N_act_ij_; ++ai) {
                const int i_ij = active_i_ij_[ai];
                const int n_ij = n_pno_[i_ij];
                if (n_ij == 0) continue;
                const std::vector<int>& need = s.needed_ikl_host[ai];
                const size_t base = s.compact_offset[ai];
                const size_t blk  = static_cast<size_t>(n_ij) * n_ij;
                for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                    const int i_kl = need[n];
                    pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
                    std::memcpy(pi_cache_out[i_ij][i_kl].data(),
                                s.h_pi_compact + base
                                    + static_cast<size_t>(n) * blk,
                                blk * sizeof(real_t));
                }
            }
            if (picache_prof) s.t_scatter += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pc_t0).count();
        } else {
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
        }  // end else (padded D2H + scatter)
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
// Phase 1 (sparse barS) — ragged coupling-list compute. Same two GEMMs as
// compute_pi_tile_ but batched over the needed_ikl coupling columns per row
// (count = needed_count[ai]) via pointer arrays, writing pi straight into
// d_pi_needed at slot (ai·max_needed + n). Each (ai, n) batch element is
// arithmetically identical to compute_pi_tile_'s (ai, ak) element followed by
// gather_needed_kernel's copy of column ak into slot n → bit-exact. Requires
// the single-tile gate (ai_in_tile == ai) and the pointer arrays built.
// ---------------------------------------------------------------------------
void PiCacheGpu::compute_pi_needed_ragged_(int tile_start, int tile_end)
{
    Impl& s = *p_;
    const int max_n = max_n_;
    const long long stride_pair = s.stride_pair;
    const real_t one  = 1.0;
    const real_t zero = 0.0;

    // Phase 2 speed: one-row reused Y gather buffer (max_needed × max_n²) so
    // both GEMM factors are contiguous with uniform stride → per-row
    // cublasDgemmStridedBatched (dense speed). Tiny vs the full grid ⇒ no
    // memory penalty.
    if (!s.d_Y_coupling) {
        const size_t bytes = static_cast<size_t>(s.max_needed)
                           * static_cast<size_t>(stride_pair) * sizeof(real_t);
        if (bytes > 0)
            check_cuda_(cudaMalloc(&s.d_Y_coupling, bytes),
                        "cudaMalloc d_Y_coupling (one row)");
    }
    // Run the gather on the SAME stream as the cuBLAS GEMMs so the per-row
    // gather → stage1 → stage2 chain is correctly ordered (no race on the
    // reused d_Y_coupling / d_half_pad buffers).
    cudaStream_t gemm_stream = nullptr;
    cublasGetStream(s.cublas, &gemm_stream);

    for (int ai = tile_start; ai < tile_end; ++ai) {
        const int i_ij = active_i_ij_[ai];
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) continue;
        const int cnt = static_cast<int>(s.needed_ikl_host[ai].size());
        if (cnt == 0) continue;
        // Offset-CSR (Stage D, CCSD): row ai's slots start at needed_offset[ai];
        // padded fallback (LMP2, needed_offset empty): ai·max_needed.
        const size_t row_slot = s.needed_offset.empty()
            ? static_cast<size_t>(ai) * s.max_needed
            : s.needed_offset[ai];
        const size_t base = row_slot * static_cast<size_t>(stride_pair);

        // Gather this row's coupling Y into the contiguous one-row buffer
        // (on the cuBLAS stream so it precedes stage1 reading d_Y_coupling).
        gather_Y_row_kernel<<<static_cast<unsigned>(cnt), 256, 0, gemm_stream>>>(
            s.d_Y_pad,
            s.d_needed_ak + row_slot,
            cnt, s.d_Y_coupling, stride_pair);
        check_cuda_(cudaGetLastError(), "gather_Y_row_kernel launch");

        // Stage 1: half = Y · barS  (per-row strided-batched over cnt coupling
        // slots; matches compute_pi_tile_ stage1, contiguous ⇒ bit-exact).
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/ max_n, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            s.d_Y_coupling,        /*lda=*/ max_n, /*strideA=*/ stride_pair,
            s.d_barS_csr   + base, /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            s.d_half_pad,          /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            cnt), "ragged stage1 strided");

        // Stage 2: pi = barS^T · half  (matches compute_pi_tile_ stage2).
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            /*m=*/ n_ij, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            s.d_barS_csr   + base, /*lda=*/ max_n, /*strideA=*/ stride_pair,
            s.d_half_pad,          /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            s.d_pi_needed  + base, /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            cnt), "ragged stage2 strided");
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

// Stage D — sparse pi_T_stack kl-slot list + slot map getters.
#ifndef GANSU_CPU_ONLY
const int*    PiCacheGpu::device_n_slots()          const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_n_slots : nullptr; }
const size_t* PiCacheGpu::device_slot_offset()      const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_slot_offset : nullptr; }
const int*    PiCacheGpu::device_kl_slot()          const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_kl_slot : nullptr; }
const size_t* PiCacheGpu::device_idx_offset_sparse()const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_idx_offset_sparse : nullptr; }
const int*    PiCacheGpu::device_slot_jcol()        const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_slot_jcol : nullptr; }
const int*    PiCacheGpu::device_slot_irow()        const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_slot_irow : nullptr; }
const int*    PiCacheGpu::device_slot_icol()        const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_slot_icol : nullptr; }
const int*    PiCacheGpu::device_slot_jrow()        const noexcept { return (p_ && p_->kl_slot_built) ? p_->d_slot_jrow : nullptr; }
bool          PiCacheGpu::pitstack_sparse_ready()   const noexcept { return p_ && p_->kl_slot_built; }
const int*    PiCacheGpu::host_n_slots()            const noexcept { return (p_ && p_->kl_slot_built && !p_->h_n_slots_host.empty()) ? p_->h_n_slots_host.data() : nullptr; }
#else
const int*    PiCacheGpu::device_n_slots()          const noexcept { return nullptr; }
const size_t* PiCacheGpu::device_slot_offset()      const noexcept { return nullptr; }
const int*    PiCacheGpu::device_kl_slot()          const noexcept { return nullptr; }
const size_t* PiCacheGpu::device_idx_offset_sparse()const noexcept { return nullptr; }
const int*    PiCacheGpu::device_slot_jcol()        const noexcept { return nullptr; }
const int*    PiCacheGpu::device_slot_irow()        const noexcept { return nullptr; }
const int*    PiCacheGpu::device_slot_icol()        const noexcept { return nullptr; }
const int*    PiCacheGpu::device_slot_jrow()        const noexcept { return nullptr; }
bool          PiCacheGpu::pitstack_sparse_ready()   const noexcept { return false; }
const int*    PiCacheGpu::host_n_slots()            const noexcept { return nullptr; }
#endif

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
                // Sparse pi_cache (coupling-only): non-coupling (k,l) blocks are
                // 0×0 ⇒ leave their pi_T_stack column block zero (setZero above).
                if (pi_canon.rows() == 0 || pi_canon.cols() == 0) continue;
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
#ifndef GANSU_CPU_ONLY
// ---------------------------------------------------------------------------
// Phase 2 (CCSD sparse barS) — one-time stacked sparse setup. See header.
// ---------------------------------------------------------------------------
void PiCacheGpu::setup_sparse_stacked_(
    const std::vector<std::vector<int>>& coupling)
{
    Impl& s = *p_;
    if (s.coupling_ptrs_built) return;              // idempotent (one-time)
    // Sparse-path diagnostics (coupling/memory footprint per device). Off by
    // default; enable with GANSU_DLPNO_CCSD_SPARSE_DIAG=1.
    static const bool diag_on = []() {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_SPARSE_DIAG");
        return e && e[0] == '1';
    }();
    const long long stride_pair = s.stride_pair;    // max_n²
    const int max_n = max_n_;

    // 1) needed_ikl_host from the coupling list (active, deduped) + max_needed.
    s.needed_ikl_host.assign(std::max(1, N_act_ij_), {});
    int max_needed = 0;
    for (int ai = 0; ai < N_act_ij_; ++ai) {
        const int i_ij = active_i_ij_[ai];
        std::vector<int>& dst = s.needed_ikl_host[ai];
        if (i_ij >= 0 && i_ij < static_cast<int>(coupling.size())) {
            for (int i_kl : coupling[i_ij])
                if (i_kl >= 0 && i_kl < N_pair_ && active_kl_pos_[i_kl] >= 0)
                    dst.push_back(i_kl);
        }
        std::sort(dst.begin(), dst.end());
        dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
        if (static_cast<int>(dst.size()) > max_needed)
            max_needed = static_cast<int>(dst.size());
    }
    s.max_needed = std::max(1, max_needed);

    // Stage D (offset-CSR): cumulative per-row offsets so the big ragged buffers
    // are sized by Σ count (AVG) not N_act_ij·max_needed (MAX padding). Row ai's
    // slots live at [needed_offset[ai], needed_offset[ai]+count[ai]). This is
    // what lets the distance screen actually reduce memory.
    s.needed_offset.assign(static_cast<size_t>(N_act_ij_) + 1, 0);
    for (int ai = 0; ai < N_act_ij_; ++ai)
        s.needed_offset[ai + 1] =
            s.needed_offset[ai] + s.needed_ikl_host[ai].size();
    s.total_needed = s.needed_offset[N_act_ij_];
    const size_t ns = std::max<size_t>(1, s.total_needed);  // offset-CSR total

    // Stage D — print the PLANNED ragged footprint BEFORE the big allocs so an
    // OOM here still reports avg/max coupling + buffer sizes (the post-build
    // diagnostic at the end never prints when an alloc throws first).
    if (diag_on) {
        const double blk_mb = static_cast<double>(stride_pair)
                            * sizeof(real_t) / (1024.0 * 1024.0);
        const double avg = N_act_ij_ > 0
            ? static_cast<double>(s.total_needed) / N_act_ij_ : 0.0;
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const double ragged_mb = static_cast<double>(s.total_needed) * blk_mb;
        std::printf("[CCSD-sparse-plan dev=%d slab=[%d,%d)] N_act_ij=%d "
                    "max_coupling=%d avg_coupling=%.1f total_needed=%zu "
                    "→ d_pi_needed=%.0f MB + d_barS_csr=%.0f MB "
                    "(padded would be %.0f MB each); GPU free=%.0f MB / "
                    "total=%.0f MB\n",
                    device_id_, pair_begin_, pair_end_, N_act_ij_,
                    s.max_needed, avg, s.total_needed,
                    ragged_mb, ragged_mb,
                    static_cast<double>(N_act_ij_) * s.max_needed * blk_mb,
                    free_b / (1024.0 * 1024.0), total_b / (1024.0 * 1024.0));
        std::fflush(stdout);
    }

    // 2) device maps: d_needed_ak (active-kl pos), d_needed_count, d_coupling_ikl.
    //    Offset-CSR layout: slot = needed_offset[ai] + n (NOT ai·max_needed).
    std::vector<int> h_ak(ns, -1);
    std::vector<int> h_cnt(std::max(1, N_act_ij_), 0);
    std::vector<int> h_cik(ns, -1);
    for (int ai = 0; ai < N_act_ij_; ++ai) {
        const std::vector<int>& dst = s.needed_ikl_host[ai];
        h_cnt[ai] = static_cast<int>(dst.size());
        for (int n = 0; n < static_cast<int>(dst.size()); ++n) {
            const size_t slot = s.needed_offset[ai] + n;
            h_ak[slot]  = active_kl_pos_[dst[n]];
            h_cik[slot] = dst[n];
        }
    }
    check_cuda_(cudaMalloc(&s.d_needed_ak, sizeof(int) * h_ak.size()),
                "cudaMalloc d_needed_ak (stacked)");
    check_cuda_(cudaMalloc(&s.d_needed_count, sizeof(int) * h_cnt.size()),
                "cudaMalloc d_needed_count (stacked)");
    check_cuda_(cudaMalloc(&s.d_coupling_ikl, sizeof(int) * h_cik.size()),
                "cudaMalloc d_coupling_ikl");
    check_cuda_(cudaMalloc(&s.d_needed_offset,
                           sizeof(size_t) * s.needed_offset.size()),
                "cudaMalloc d_needed_offset");
    check_cuda_(cudaMemcpy(s.d_needed_ak, h_ak.data(), sizeof(int) * h_ak.size(),
                           cudaMemcpyHostToDevice), "H2D d_needed_ak (stacked)");
    check_cuda_(cudaMemcpy(s.d_needed_count, h_cnt.data(), sizeof(int) * h_cnt.size(),
                           cudaMemcpyHostToDevice), "H2D d_needed_count (stacked)");
    check_cuda_(cudaMemcpy(s.d_coupling_ikl, h_cik.data(), sizeof(int) * h_cik.size(),
                           cudaMemcpyHostToDevice), "H2D d_coupling_ikl");
    check_cuda_(cudaMemcpy(s.d_needed_offset, s.needed_offset.data(),
                           sizeof(size_t) * s.needed_offset.size(),
                           cudaMemcpyHostToDevice), "H2D d_needed_offset");

    // 3) d_pi_needed (device). h_pi_needed (pinned mirror) is NOT allocated
    //    here — it is only needed by the CPU-resid host-pi_cache path
    //    (!skip_host_pi), which is rare for CCSD (rgpu active). Allocating it
    //    eagerly pinned a second N_act_ij·max_coupling·max_n² buffer (21 GB on
    //    PTCDA) whose cudaMallocHost dominated the one-time setup (~tens of s
    //    under 4-device mmap_lock contention). Lazy-allocate it on first use.
    const size_t bytes_needed = ns * static_cast<size_t>(stride_pair) * sizeof(real_t);
    if (bytes_needed > 0) {
        check_cuda_(cudaMalloc(&s.d_pi_needed, bytes_needed),
                    "cudaMalloc d_pi_needed (stacked)");
    }
    s.needed_built = true;

    // 4) CSR barS from barS_cache_ref_ (coupling blocks; max_n-padded, identical
    //    bytes to the dense scatter ⇒ bit-exact per block).
    {
        const auto& barS_cache = *barS_cache_ref_;
        const size_t csr_elems = ns * static_cast<size_t>(stride_pair);
        // Pageable host buffer (NOT cudaMallocHost): this is a one-time build,
        // so the instant pageable allocation + slightly slower H2D beats pinning
        // a 21 GB buffer (4-device mmap_lock contention dominated setup).
        std::vector<real_t> h_csr(csr_elems, real_t(0));
        #pragma omp parallel for schedule(static)
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const int i_ij = active_i_ij_[ai];
            const int n_ij = n_pno_[i_ij];
            const std::vector<int>& need = s.needed_ikl_host[ai];
            for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                const int i_kl = need[n];
                const int n_kl = n_pno_[i_kl];
                const RowMatXd& bs = barS_cache[i_ij][i_kl];
                if (bs.rows() == 0 || bs.cols() == 0) continue;
                real_t* dstb = h_csr.data()
                    + (s.needed_offset[ai] + n)
                      * static_cast<size_t>(stride_pair);
                const real_t* srcb = bs.data();
                for (int r = 0; r < n_ij; ++r)
                    std::memcpy(dstb + static_cast<size_t>(r) * max_n,
                                srcb + static_cast<size_t>(r) * n_kl,
                                static_cast<size_t>(n_kl) * sizeof(real_t));
            }
        }
        check_cuda_(cudaMalloc(&s.d_barS_csr, csr_elems * sizeof(real_t)),
                    "cudaMalloc d_barS_csr (stacked)");
        check_cuda_(cudaMemcpy(s.d_barS_csr, h_csr.data(),
                               csr_elems * sizeof(real_t),
                               cudaMemcpyHostToDevice), "H2D d_barS_csr (stacked)");
        s.barS_csr_built = true;
    }

    // (Pointer-array GEMM build removed — compute_pi_needed_ragged_ now uses
    //  per-row cublasDgemmStridedBatched on the contiguous CSR + one-row Y
    //  gather, so the old d_p*_ptrs arrays are unused.)
    s.coupling_ptrs_built = true;

    // ── Stage D (D1a) — sparse pi_T_stack kl-slot list + slot maps ──────────
    // Build the per-pair ordered (k,l)-slot list (coupling expanded to both
    // orientations) + the slot maps the ResidGpu consumers need. Indexed by
    // ORIG pair idx (small ints). Unused until D1b switches storage/consumers.
    if (!s.kl_slot_built) {
        const int nocc = nocc_;
        std::vector<int>    h_n_slots(static_cast<size_t>(N_pair_), 0);
        std::vector<size_t> h_slot_off(static_cast<size_t>(N_pair_) + 1, 0);
        std::vector<size_t> h_idx_off_sp(static_cast<size_t>(N_pair_) + 1, 0);
        // First pass: count slots per pair (orig idx) over this slab's active rows.
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const int i_ij = active_i_ij_[ai];
            int cnt = 0;
            for (int idx_kl : s.needed_ikl_host[ai]) {
                const int p = setup_i_per_pair_[idx_kl];
                const int q = setup_j_per_pair_[idx_kl];
                cnt += (p == q) ? 1 : 2;
            }
            h_n_slots[i_ij] = cnt;
        }
        for (int idx = 0; idx < N_pair_; ++idx) {
            h_slot_off[idx + 1] = h_slot_off[idx] + h_n_slots[idx];
            const size_t n = static_cast<size_t>(n_pno_[idx]);
            h_idx_off_sp[idx + 1] = h_idx_off_sp[idx]
                + n * n * static_cast<size_t>(h_n_slots[idx]);
        }
        const size_t kl_total = h_slot_off[N_pair_];
        std::vector<int> h_kl_slot(std::max<size_t>(1, kl_total), -1);
        std::vector<int> h_jcol(static_cast<size_t>(N_pair_) * nocc, -1);
        std::vector<int> h_irow(static_cast<size_t>(N_pair_) * nocc, -1);
        std::vector<int> h_icol(static_cast<size_t>(N_pair_) * nocc, -1); // (k,i)
        std::vector<int> h_jrow(static_cast<size_t>(N_pair_) * nocc, -1); // (j,l)
        // coupling n → its canonical local slot (transpose = base+1 if p!=q).
        // Offset-CSR: indexed by needed_offset[ai]+n (same as the other ragged
        // companions), sized total_needed.
        std::vector<int> h_csb(std::max<size_t>(1, s.total_needed), -1);
        int max_sl = 0;
        // Second pass: fill kl-slot list (local slot index) + slot maps.
        for (int ai = 0; ai < N_act_ij_; ++ai) {
            const int i_ij = active_i_ij_[ai];
            const int i = setup_i_per_pair_[i_ij];
            const int j = setup_j_per_pair_[i_ij];
            const size_t base = h_slot_off[i_ij];
            int slot = 0;
            auto emit = [&](int k, int l) {
                h_kl_slot[base + slot] = k * nocc + l;
                if (l == j) h_jcol[static_cast<size_t>(i_ij) * nocc + k] = slot; // (k,j)
                if (k == i) h_irow[static_cast<size_t>(i_ij) * nocc + l] = slot; // (i,l)
                if (l == i) h_icol[static_cast<size_t>(i_ij) * nocc + k] = slot; // (k,i)
                if (k == j) h_jrow[static_cast<size_t>(i_ij) * nocc + l] = slot; // (j,l)
                ++slot;
            };
            const std::vector<int>& need = s.needed_ikl_host[ai];
            for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                const int idx_kl = need[n];
                const int p = setup_i_per_pair_[idx_kl];
                const int q = setup_j_per_pair_[idx_kl];
                h_csb[s.needed_offset[ai] + n] = slot;
                emit(p, q);                 // canonical (k,l)=(p,q) at slot base
                if (p != q) emit(q, p);     // transpose (q,p) at base+1
            }
            if (slot > max_sl) max_sl = slot;
        }
        s.max_slots = max_sl;
        s.pi_T_sparse_total = h_idx_off_sp[N_pair_];
        // Stage D (multi-GPU) — slab the sparse d_pi_T_stack: each device stores
        // ONLY its slab pairs [pair_begin_, pair_end_), not the full N_pair
        // replica (ResidGpu + compute_dfpair only read pi_T_stack[idx] for idx
        // in this slab). pi_T_sparse_slab_{base,total} are the full-cumulative
        // base + slab elem count. d_idx_offset_sparse is uploaded SLAB-RELATIVE
        // (− slab_base) so all kernels indexing d_idx_offset_sparse[idx] hit the
        // slab buffer at the right offset with NO kernel change. Out-of-slab idx
        // entries become garbage but are never read. Single-GPU: slab_base=0 ⇒
        // identical to the old full layout (bit-exact).
        s.pi_T_sparse_slab_base  = h_idx_off_sp[pair_begin_];
        s.pi_T_sparse_slab_total = h_idx_off_sp[pair_end_]
                                 - h_idx_off_sp[pair_begin_];
        // Keep host copies for the sparse T_meta_dpair gather (upload).
        s.h_kl_slot_host     = h_kl_slot;
        s.h_slot_offset_host = h_slot_off;
        s.h_n_slots_host     = h_n_slots;
        check_cuda_(cudaMalloc(&s.d_n_slots, sizeof(int) * N_pair_), "d_n_slots");
        check_cuda_(cudaMalloc(&s.d_slot_offset, sizeof(size_t) * (N_pair_ + 1)), "d_slot_offset");
        check_cuda_(cudaMalloc(&s.d_kl_slot, sizeof(int) * std::max<size_t>(1, kl_total)), "d_kl_slot");
        check_cuda_(cudaMalloc(&s.d_idx_offset_sparse, sizeof(size_t) * (N_pair_ + 1)), "d_idx_offset_sparse");
        check_cuda_(cudaMalloc(&s.d_slot_jcol, sizeof(int) * h_jcol.size()), "d_slot_jcol");
        check_cuda_(cudaMalloc(&s.d_slot_irow, sizeof(int) * h_irow.size()), "d_slot_irow");
        check_cuda_(cudaMemcpy(s.d_n_slots, h_n_slots.data(), sizeof(int) * N_pair_, cudaMemcpyHostToDevice), "H2D n_slots");
        check_cuda_(cudaMemcpy(s.d_slot_offset, h_slot_off.data(), sizeof(size_t) * (N_pair_ + 1), cudaMemcpyHostToDevice), "H2D slot_offset");
        check_cuda_(cudaMemcpy(s.d_kl_slot, h_kl_slot.data(), sizeof(int) * std::max<size_t>(1, kl_total), cudaMemcpyHostToDevice), "H2D kl_slot");
        // Slab-relative offsets for the sparse (slabbed) d_pi_T_stack. Subtract
        // the slab base so kernels reading d_idx_offset_sparse[idx] (idx in slab)
        // address the slab-local buffer. !pitstack_sparse_: keep full (unused).
        if (pitstack_sparse_) {
            std::vector<size_t> h_idx_off_rel(h_idx_off_sp.size());
            const size_t base = s.pi_T_sparse_slab_base;
            for (size_t t = 0; t < h_idx_off_sp.size(); ++t)
                h_idx_off_rel[t] = h_idx_off_sp[t] - base;  // wraps for idx<begin (unused)
            check_cuda_(cudaMemcpy(s.d_idx_offset_sparse, h_idx_off_rel.data(), sizeof(size_t) * (N_pair_ + 1), cudaMemcpyHostToDevice), "H2D idx_offset_sparse (slab-rel)");
        } else {
            check_cuda_(cudaMemcpy(s.d_idx_offset_sparse, h_idx_off_sp.data(), sizeof(size_t) * (N_pair_ + 1), cudaMemcpyHostToDevice), "H2D idx_offset_sparse");
        }
        check_cuda_(cudaMemcpy(s.d_slot_jcol, h_jcol.data(), sizeof(int) * h_jcol.size(), cudaMemcpyHostToDevice), "H2D slot_jcol");
        check_cuda_(cudaMemcpy(s.d_slot_irow, h_irow.data(), sizeof(int) * h_irow.size(), cudaMemcpyHostToDevice), "H2D slot_irow");
        check_cuda_(cudaMalloc(&s.d_slot_icol, sizeof(int) * h_icol.size()), "d_slot_icol");
        check_cuda_(cudaMalloc(&s.d_slot_jrow, sizeof(int) * h_jrow.size()), "d_slot_jrow");
        check_cuda_(cudaMemcpy(s.d_slot_icol, h_icol.data(), sizeof(int) * h_icol.size(), cudaMemcpyHostToDevice), "H2D slot_icol");
        check_cuda_(cudaMemcpy(s.d_slot_jrow, h_jrow.data(), sizeof(int) * h_jrow.size(), cudaMemcpyHostToDevice), "H2D slot_jrow");
        check_cuda_(cudaMalloc(&s.d_coupling_slot_base, sizeof(int) * std::max<size_t>(1, h_csb.size())), "d_coupling_slot_base");
        check_cuda_(cudaMemcpy(s.d_coupling_slot_base, h_csb.data(), sizeof(int) * h_csb.size(), cudaMemcpyHostToDevice), "H2D coupling_slot_base");

        if (pitstack_sparse_ && diag_on) {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            std::printf("[CCSD-sparse-piT dev=%d] d_pi_T_stack slab=%.0f MB "
                        "(full-replica would be %.0f MB); GPU free=%.0f MB\n",
                        device_id_,
                        static_cast<double>(s.pi_T_sparse_slab_total)
                            * sizeof(real_t) / (1024.0 * 1024.0),
                        static_cast<double>(s.pi_T_sparse_total)
                            * sizeof(real_t) / (1024.0 * 1024.0),
                        free_b / (1024.0 * 1024.0));
            std::fflush(stdout);
        }

        // Stage D (multi-GPU): allocate the SPARSE d_pi_T_stack SLAB-SIZED (only
        // this device's [pair_begin_, pair_end_) pairs), not the full-N_pair
        // replica. Device buffer base corresponds to pair_begin_; the slab-
        // relative d_idx_offset_sparse uploaded above makes kernels address it
        // correctly. Plus the slab pinned mirror.
        if (pitstack_sparse_) {
            check_cuda_(cudaMalloc(&s.d_pi_T_stack,
                            std::max<size_t>(1, s.pi_T_sparse_slab_total) * sizeof(real_t)),
                        "cudaMalloc d_pi_T_stack (sparse slab)");
            if (s.pi_T_sparse_slab_total > 0) {
                check_cuda_(cudaMallocHost(&s.h_pi_T_stack,
                                s.pi_T_sparse_slab_total * sizeof(real_t)),
                            "cudaMallocHost h_pi_T_stack (sparse slab)");
            }
        }
        s.kl_slot_built = true;
    }

    // Phase 2 diagnostic — coupling sparsity for this slab. Shows how much the
    // norm/distance screen shrank the per-row coupling vs the dense N_act_kl
    // grid, plus the CSR barS + d_pi_needed footprint. One line per device.
    if (diag_on) {
        size_t total_coupling = 0;
        for (int ai = 0; ai < N_act_ij_; ++ai)
            total_coupling += s.needed_ikl_host[ai].size();
        const double avg_coupling = N_act_ij_ > 0
            ? static_cast<double>(total_coupling) / N_act_ij_ : 0.0;
        const double blk_mb = static_cast<double>(stride_pair)
                            * sizeof(real_t) / (1024.0 * 1024.0);
        // Offset-CSR: the big buffers are sized total_needed (= Σ count = avg×
        // N_act_ij), NOT N_act_ij·max_needed (padded). pad_mb shows what the old
        // rectangular layout would have cost, so the screen's memory win is visible.
        const double csr_mb = static_cast<double>(s.total_needed) * blk_mb;
        const double pad_mb = static_cast<double>(N_act_ij_)
                            * s.max_needed * blk_mb;
        std::printf("[CCSD-sparse dev=%d slab=[%d,%d)] N_act_ij=%d N_act_kl=%d "
                    "max_coupling=%d avg_coupling=%.1f (dense N_act_kl=%d) "
                    "CSR=%.0f MB d_pi_needed=%.0f MB (padded would be %.0f MB)\n",
                    device_id_, pair_begin_, pair_end_, N_act_ij_, N_act_kl_,
                    s.max_needed, avg_coupling, N_act_kl_, csr_mb, csr_mb, pad_mb);
        std::fflush(stdout);
    }
}
#endif // !GANSU_CPU_ONLY

// Stage D (D1b): pre-build the sparse kl-slot machinery before the iter loop so
// the one-time pre-loop upload_T_meta_dpair() can pack T_meta in sparse layout.
// No-op unless pitstack_sparse_; idempotent (setup_sparse_stacked_ guarded).
void PiCacheGpu::ensure_sparse_stacked(
    const std::vector<std::vector<int>>& coupling) {
#ifndef GANSU_CPU_ONLY
    if (!pitstack_sparse_) return;
    MultiGpuManager::DeviceGuard _guard(device_id_);
    setup_sparse_stacked_(coupling);
#else
    (void)coupling;
#endif
}

void PiCacheGpu::rebuild_with_stack(
    const std::vector<std::vector<real_t>>& Y_old,
    std::vector<std::vector<RowMatXd>>& pi_cache_out,
    std::vector<RowMatXd>& pi_T_stack_out,
    bool skip_pi_cache_host,
    const std::vector<std::vector<int>>* coupling_ikl_per_pair,
    bool skip_pi_T_stack_host)
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

    // Phase 2 (CCSD sparse barS): sparse stacked path active when the instance
    // was built sparse AND a coupling list is supplied. Single tile is forced
    // by the ctor (tile_size_ = N_act_ij_).
    const bool use_sparse = sparse_stacked_ && (coupling_ikl_per_pair != nullptr)
                            && N_act_ij_ > 0;

    // Per-pair DENSE offsets into d_pi_T_stack (slab range for the dense path;
    // also used by the dense memset/D2H + host pi_T_stack_out scatter).
    std::vector<size_t> idx_offset_host(static_cast<size_t>(N + 1), 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        idx_offset_host[i + 1] = idx_offset_host[i]
            + n * n
            * static_cast<size_t>(nocc)
            * static_cast<size_t>(nocc);
    }
    const size_t bytes_pi_T_slab = (idx_offset_host[ie] - idx_offset_host[ib])
                                 * sizeof(real_t);
    // NOTE: the slab memset is deferred to AFTER setup_sparse_stacked_ below,
    // because in the sparse pi_T_stack mode (pitstack_sparse_) the device
    // d_pi_T_stack buffer itself is allocated there (its size = pi_T_sparse_total
    // is unknown until the kl-slot list is built).

    // Step Z — unified tile loop: pack Y once, then per-tile compute +
    // (optional) D2H scatter to pi_cache_out + pack into pi_T_stack.
    // skip_host_pi path skips the D2H + scatter (Step 6 fast path).
    pack_Y_and_transpose_(Y_old);

    // Phase 2 (CCSD sparse): one-time build of the coupling machinery (CSR barS,
    // ragged pointers, d_coupling_ikl, d_pi_needed). Idempotent. In
    // pitstack_sparse_ mode this also allocates the sparse d_pi_T_stack.
    if (use_sparse) setup_sparse_stacked_(*coupling_ikl_per_pair);

    // Zero the slab range of d_pi_T_stack (sparse or dense layout). Scattered
    // writes only fill coupling slots; the rest must be zero on entry.
    // Sparse (pitstack) buffer is SLAB-SIZED ⇒ base 0; dense buffer is full ⇒
    // slab base idx_offset_host[ib].
    {
        const size_t zero_base = pitstack_sparse_ ? size_t(0)
                                                   : idx_offset_host[ib];
        const size_t zero_bytes = (pitstack_sparse_ ? s.pi_T_sparse_slab_total
                                                     : (idx_offset_host[ie]
                                                        - idx_offset_host[ib]))
                                  * sizeof(real_t);
        if (zero_bytes > 0) {
            check_cuda_(cudaMemsetAsync(s.d_pi_T_stack + zero_base, 0,
                                        zero_bytes, /*stream=*/0),
                        "memset d_pi_T_stack (slab)");
        }
    }

    // Pinned tile mirror for D2H scatter (non-skip only). Sized one tile.
    // (sparse path uses h_pi_needed instead, allocated in setup_sparse_stacked_.)
    const size_t bytes_pi_tile_full =
        static_cast<size_t>(tile_size_) * N_act_kl_
        * static_cast<size_t>(max_n) * max_n * sizeof(real_t);
    if (!use_sparse && !skip_host_pi && bytes_pi_tile_full > 0
        && s.h_pi_pad == nullptr) {
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

    static const bool sp_prof = []() {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_SPARSE_PROF");
        return e && e[0] == '1';
    }();
    auto sp_now = [&]() {
        cudaDeviceSynchronize();
        return std::chrono::steady_clock::now();
    };
    if (use_sparse && slab_n > 0 && nocc > 0 && max_n > 0 && N_act_ij_ > 0) {
        // ---- Phase 2 sparse path: ragged pi (coupling only) → dense pi_T_stack.
        // Single tile (tile_size_ == N_act_ij_). compute_pi_needed_ragged_ fills
        // d_pi_needed[coupling slots]; the scatter writes them into the dense
        // d_pi_T_stack at the (k·nocc+l) positions (non-coupling stays zero from
        // the memset above). ResidGpu reads d_pi_T_stack unchanged. DFpair reads
        // it via compute_dfpair (unchanged).
        const auto sp_t0 = sp_prof ? sp_now()
                                   : std::chrono::steady_clock::now();
        compute_pi_needed_ragged_(0, N_act_ij_);
        const auto sp_t1 = sp_prof ? sp_now() : sp_t0;

        const int bx = (max_n < 16) ? max_n : 16;
        dim3 sblock(static_cast<unsigned>(bx), static_cast<unsigned>(bx), 1);
        dim3 sgrid(static_cast<unsigned>(N_act_ij_),
                   static_cast<unsigned>(s.max_needed), 1);
        scatter_pi_T_coupling_kernel<<<sgrid, sblock>>>(
            s.d_pi_needed, s.d_active_i_ij, s.d_coupling_ikl, s.d_needed_count,
            s.d_setup_i, s.d_setup_j, s.d_n_pno, s.d_idx_offset,
            s.d_pi_T_stack, nocc, max_n, s.max_needed, N_act_ij_,
            pitstack_sparse_ ? 1 : 0,
            s.d_idx_offset_sparse, s.d_n_slots, s.d_coupling_slot_base,
            s.d_needed_offset);  // offset-CSR (Stage D); null in padded fallback
        if (sp_prof) {
            const auto sp_t2 = sp_now();
            std::printf("[CCSD-sparse-PROF dev=%d] ragged_compute=%.3fs "
                        "scatter=%.3fs (max_coupling=%d)\n", device_id_,
                        std::chrono::duration<double>(sp_t1 - sp_t0).count(),
                        std::chrono::duration<double>(sp_t2 - sp_t1).count(),
                        s.max_needed);
            std::fflush(stdout);
        }
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("PiCacheGpu scatter_pi_T kernel launch failed: ")
                + cudaGetErrorString(e));
        }

        // Optional host pi_cache (CPU resid path): D2H ragged d_pi_needed and
        // scatter the coupling blocks into pi_cache_out (others stay 0×0). The
        // CPU residual only reads {(k,j),(i,l)} which are in the coupling.
        if (!skip_host_pi) {
            // Offset-CSR: mirror is sized total_needed (= Σ count), indexed by
            // needed_offset[ai]+n.
            const size_t bytes_needed =
                std::max<size_t>(1, s.total_needed)
                * static_cast<size_t>(stride_pair) * sizeof(real_t);
            // Lazy pinned mirror (only the CPU-resid path needs it).
            if (s.h_pi_needed == nullptr && bytes_needed > 0)
                check_cuda_(cudaMallocHost(&s.h_pi_needed, bytes_needed),
                            "cudaMallocHost h_pi_needed (lazy)");
            check_cuda_(cudaMemcpy(s.h_pi_needed, s.d_pi_needed, bytes_needed,
                                   cudaMemcpyDeviceToHost),
                        "D2H d_pi_needed (stacked sparse)");
            #pragma omp parallel for schedule(static)
            for (int ai = 0; ai < N_act_ij_; ++ai) {
                const int i_ij = active_i_ij_[ai];
                const int n_ij = n_pno_[i_ij];
                const std::vector<int>& need = s.needed_ikl_host[ai];
                for (int n = 0; n < static_cast<int>(need.size()); ++n) {
                    const int i_kl = need[n];
                    pi_cache_out[i_ij][i_kl].resize(n_ij, n_ij);
                    const real_t* src = s.h_pi_needed
                        + (s.needed_offset[ai] + n)
                          * static_cast<size_t>(stride_pair);
                    real_t* dst = pi_cache_out[i_ij][i_kl].data();
                    for (int r = 0; r < n_ij; ++r)
                        std::memcpy(dst + static_cast<ptrdiff_t>(r) * n_ij,
                                    src + static_cast<size_t>(r) * max_n,
                                    static_cast<size_t>(n_ij) * sizeof(real_t));
                }
            }
        }
    } else if (slab_n > 0 && nocc > 0 && max_n > 0 && N_act_ij_ > 0 && tile_size_ > 0) {
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

    // Stage D: in pitstack_sparse_ mode the host pi_T_stack_out (dense layout)
    // is normally NOT produced — d_pi_T_stack is sparse, and the GPU consumers
    // (ResidGpu + compute_dfpair) read the device buffer. BUT when a device's
    // ResidGpu failed to fit (not all_rgpu_active ⇒ skip_host_pi=false), that
    // slab's residual falls back to the CPU oooo ladder, which reads the host
    // pi_T_stack[idx]. Build it here from the (just-produced, coupling-only)
    // pi_cache_out so the fallback is CORRECT (non-coupling (k,l) stay zero ⇒
    // bit-consistent with the GPU sparse oooo). When skip_host_pi (all GPU
    // active), no fallback runs ⇒ leave 0×0.
    if (pitstack_sparse_) {
        if (!skip_host_pi) {
            build_stack_cpu_(pi_cache_out, pi_T_stack_out);  // CPU oooo fallback
        } else {
            for (long long i_ij = ib; i_ij < ie; ++i_ij)
                pi_T_stack_out[i_ij].resize(0, 0);
        }
        return;
    }

    // Step 6 fast path (dense): when this slab's host pi_T_stack_out is never
    // read on the CPU this iter — the device's ResidGpu is active (GPU oooo)
    // AND DFpair runs on the GPU (compute_dfpair reads the device d_pi_T_stack
    // directly) — the per-iter D2H + host scatter below is pure waste
    // (~pi_T_slab bytes/device/iter). Skip it and leave pi_T_stack_out 0×0 for
    // the slab; the consumers guard on pi_T_stack[idx].size() > 0. Mirrors the
    // skip_pi_cache_host fast path for pi_cache above.
    if (skip_pi_T_stack_host) {
        for (long long i_ij = ib; i_ij < ie; ++i_ij)
            pi_T_stack_out[i_ij].resize(0, 0);
        return;
    }

    // D2H — only the slab range of pi_T_stack.
    // Step S9: h_pi_T_stack is sized to the slab only, so the destination is
    // its base (offset 0). The source on device keeps the full-N_pair layout
    // (used by kernels that index via d_idx_offset), so we still read from
    // d_pi_T_stack + idx_offset_host[ib].
    // Lazy alloc: the pinned mirror is created on first use here (partial
    // activation), not in the ctor — all-active runs return above and never
    // pay the pinned cudaMallocHost (mmap-lock contention across the ctors).
    if (bytes_pi_T_slab > 0 && s.h_pi_T_stack == nullptr) {
        check_cuda_(cudaMallocHost(&s.h_pi_T_stack, bytes_pi_T_slab),
                    "cudaMallocHost h_pi_T_stack (lazy)");
    }
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
#ifndef GANSU_CPU_ONLY
// Device reshape: raw phase24 T_pair → T_meta_dpair layout, per pair.
//   src (T_pair[idx], contiguous): [kl·n² + c·n + d]  with kl ∈ [0,nocc²)
//   dst (T_meta_dpair[idx]):       [(kl·n + d)·n + c]
// One block per slab pair; threads grid-stride the pair's nocc²·n² elements.
// Both src and dst use the dense slab-relative offset d_idx_offset[idx]-slab_base.
__global__ void reshape_tpair_to_tmeta_kernel(
    const real_t* __restrict__ d_tpair,
    real_t*       __restrict__ d_tmeta,
    const int*    __restrict__ d_n_pno,
    const size_t* __restrict__ d_idx_offset,
    size_t slab_base_elems,
    int nocc, int ib)
{
    const int idx = ib + static_cast<int>(blockIdx.x);
    const int n = d_n_pno[idx];
    if (n == 0) return;
    const size_t base = d_idx_offset[idx] - slab_base_elems;
    const size_t total =
        static_cast<size_t>(nocc) * nocc * n * n;
    for (size_t t = threadIdx.x; t < total; t += blockDim.x) {
        const int d = static_cast<int>(t % n);
        const size_t t1 = t / n;
        const int c = static_cast<int>(t1 % n);
        const size_t kl = t1 / n;
        const size_t dst = base
            + (kl * static_cast<size_t>(n) + d) * static_cast<size_t>(n) + c;
        d_tmeta[dst] = d_tpair[base + t];
    }
}
#endif

// Device-build variant of upload_T_meta_dpair: builds d_T_meta_dpair directly
// from the raw (host, contiguous) phase24 T_pair via a per-pair H2D + a device
// reshape kernel — skipping the host strided T_meta_dpair build (the `vmeta`
// cost). Dense path only; returns false for the sparse (pitstack) layout so the
// caller falls back to the host gather in upload_T_meta_dpair(). Bit-exact: the
// reshape is a pure index permutation of the same values.
bool PiCacheGpu::build_T_meta_dpair_dev(
    const std::vector<std::vector<real_t>>& T_pair)
{
#ifdef GANSU_CPU_ONLY
    (void)T_pair;
    return false;
#else
    if (!active_ || !stacked_ || pitstack_sparse_) return false;
    MultiGpuManager::DeviceGuard _guard(device_id_);
    Impl& s = *p_;
    const int N = N_pair_, nocc = nocc_;
    if (N <= 0 || nocc <= 0) return false;

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
            != cudaSuccess) { s.d_T_meta_dpair = nullptr; return false; }
    }
    if (s.d_DF_scratch == nullptr) {
        const size_t df_bytes =
            static_cast<size_t>(max_n_) * max_n_ * sizeof(real_t);
        if (df_bytes > 0
            && cudaMalloc(&s.d_DF_scratch, df_bytes) != cudaSuccess) {
            s.d_DF_scratch = nullptr; return false;
        }
    }
    // Temp device buffer for the raw T_pair slab (contiguous per pair).
    real_t* d_tpair = nullptr;
    if (cudaMalloc(&d_tpair, slab_total * sizeof(real_t)) != cudaSuccess)
        return false;
    // Zero the temp src so any pair skipped below (empty/absent T_pair) reshapes
    // to 0, matching the host build's setZero. (dst is fully written by the
    // reshape for every n>0 pair, so it need not be pre-zeroed, but the src
    // does — skipped pairs leave their src region untouched.)
    if (cudaMemset(d_tpair, 0, slab_total * sizeof(real_t)) != cudaSuccess) {
        cudaFree(d_tpair); return false;
    }
    bool ok = true;
    for (int idx = ib; idx < ie && ok; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) continue;
        if (idx >= static_cast<int>(T_pair.size()) || T_pair[idx].empty())
            continue;
        const size_t cnt = off[idx + 1] - off[idx];   // = nocc²·n²
        if (cudaMemcpy(d_tpair + (off[idx] - slab_base), T_pair[idx].data(),
                       cnt * sizeof(real_t), cudaMemcpyHostToDevice)
            != cudaSuccess) ok = false;
    }
    if (ok) {
        const int n_blocks = ie - ib;
        if (n_blocks > 0) {
            reshape_tpair_to_tmeta_kernel<<<n_blocks, 256>>>(
                d_tpair, s.d_T_meta_dpair, s.d_n_pno, s.d_idx_offset,
                slab_base, nocc, ib);
            if (cudaGetLastError() != cudaSuccess) ok = false;
            if (ok && cudaDeviceSynchronize() != cudaSuccess) ok = false;
        }
    }
    cudaFree(d_tpair);
    return ok;
#endif
}

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

    // Stage D: sparse T_meta_dpair — gather the dense host T_meta (nocc²·n × n)
    // into the kl-slot (n_slots·n × n) device layout that matches the sparse
    // pi_T_stack, so compute_dfpair contracts over n_slots instead of nocc².
    // Per-pair sparse size = n_slots·n² (= n²·n_slots = same as sparse pi_T).
    if (pitstack_sparse_ && s.kl_slot_built) {
        std::vector<size_t> sp_off(static_cast<size_t>(N) + 1, 0);
        for (int i = 0; i < N; ++i) {
            const size_t n = static_cast<size_t>(n_pno_[i]);
            sp_off[i + 1] = sp_off[i]
                + n * n * static_cast<size_t>(s.h_n_slots_host[i]);
        }
        const size_t sp_base  = sp_off[ib];
        const size_t sp_slab  = sp_off[ie] - sp_off[ib];
        if (sp_slab == 0) return false;
        if (s.d_T_meta_dpair == nullptr) {
            if (cudaMalloc(&s.d_T_meta_dpair, sp_slab * sizeof(real_t))
                != cudaSuccess) { s.d_T_meta_dpair = nullptr; return false; }
        }
        if (s.d_DF_scratch == nullptr) {
            const size_t df_bytes =
                static_cast<size_t>(max_n_) * max_n_ * sizeof(real_t);
            if (df_bytes > 0
                && cudaMalloc(&s.d_DF_scratch, df_bytes) != cudaSuccess) {
                s.d_DF_scratch = nullptr; return false;
            }
        }
        std::vector<real_t> stage(sp_slab, real_t(0));
        for (int idx = ib; idx < ie; ++idx) {
            const int n = n_pno_[idx];
            if (n == 0) continue;
            if (idx >= static_cast<int>(T_meta_dpair.size())
                || T_meta_dpair[idx].size() == 0) continue;
            const real_t* src = T_meta_dpair[idx].data();    // (nocc²·n × n)
            real_t* dstp = stage.data() + (sp_off[idx] - sp_base);
            const size_t blk  = static_cast<size_t>(n) * n;  // per-(kl) n×n block
            const size_t soff = s.h_slot_offset_host[idx];
            const int    ns   = s.h_n_slots_host[idx];
            for (int sslot = 0; sslot < ns; ++sslot) {
                const int kl = s.h_kl_slot_host[soff + sslot];
                std::memcpy(dstp + static_cast<size_t>(sslot) * blk,
                            src + static_cast<size_t>(kl) * blk,
                            blk * sizeof(real_t));
            }
        }
        if (cudaMemcpy(s.d_T_meta_dpair, stage.data(),
                       sp_slab * sizeof(real_t), cudaMemcpyHostToDevice)
            != cudaSuccess) return false;
        return true;
    }

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

    // Per-pair offsets + contracted dim K. Stage D sparse: K = n_slots·n and the
    // pi_T / T_meta offsets use the kl-slot (n²·n_slots) cumulative; else dense
    // (K = nocc²·n, offsets nocc²·n²). Both pi_T and T_meta share the same sparse
    // per-pair size (n²·n_slots) ⇒ one offset array.
    const bool sp = pitstack_sparse_ && s.kl_slot_built;
    std::vector<size_t> off(static_cast<size_t>(N) + 1, 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        const size_t per = sp ? (n * n * static_cast<size_t>(s.h_n_slots_host[i]))
                              : (n * n * static_cast<size_t>(nocc) * nocc);
        off[i + 1] = off[i] + per;
    }
    const int    ib        = pair_begin_, ie = pair_end_;
    const size_t slab_base = off[ib];

    const real_t alpha = -1.0, beta = 0.0;
    for (int idx = ib; idx < ie; ++idx) {
        const int n = n_pno_[idx];
        if (n == 0) { DF_per_pair_out[idx].resize(0, 0); continue; }
        DF_per_pair_out[idx].setZero(n, n);
        const int K = sp
            ? static_cast<int>(static_cast<size_t>(s.h_n_slots_host[idx]) * n)
            : static_cast<int>(static_cast<size_t>(nocc) * nocc * n);
        if (K == 0) continue;  // pair with no coupling slots ⇒ DF stays 0
        // row-major DF[n×n] = pi_T[n×K] · T_meta[K×n]; cuBLAS col-major idiom
        // (= resid_gpu Op-1): first ptr = right factor T_meta (lda=n), second
        // ptr = left factor pi_T (ldb=K), C = DF (ldc=n), alpha=-1 folds the −=.
        // Sparse d_pi_T_stack is SLAB-SIZED (Stage D multi-GPU) ⇒ slab-relative
        // offset; dense is full ⇒ absolute. T_meta is always slab-sized.
        const real_t* A_piT   = s.d_pi_T_stack
                              + (sp ? (off[idx] - slab_base) : off[idx]);
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
