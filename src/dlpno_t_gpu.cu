/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_t_gpu.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

#ifndef GANSU_CPU_ONLY

namespace {

inline void check_cuda_(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("TripleTGpu CUDA error in ")
                                 + what + ": " + cudaGetErrorString(e));
    }
}

// Slot mapping for the 6 perms of (i,j,k):
//   perm 0 = (i,j,k)  → slots (0,1,2)
//   perm 1 = (i,k,j)  → slots (0,2,1)
//   perm 2 = (j,i,k)  → slots (1,0,2)
//   perm 3 = (j,k,i)  → slots (1,2,0)
//   perm 4 = (k,i,j)  → slots (2,0,1)
//   perm 5 = (k,j,i)  → slots (2,1,0)
__constant__ int c_perm_table[6 * 3] = {
    0, 1, 2,
    0, 2, 1,
    1, 0, 2,
    1, 2, 0,
    2, 0, 1,
    2, 1, 0,
};

// 36 perm codes: π_{σz, σw} that gets applied to W_σw before pairing with z_σz.
// Verified bit-exact against PySCF in Phase 3.2.6 verify_perm_table.py.
__constant__ int c_perm_codes[36] = {
    0, 1, 2, 3, 4, 5,    // sz=0
    1, 0, 4, 5, 2, 3,    // sz=1
    2, 3, 0, 1, 5, 4,    // sz=2
    4, 5, 1, 0, 3, 2,    // sz=3
    3, 2, 5, 4, 0, 1,    // sz=4
    5, 4, 3, 2, 1, 0,    // sz=5
};

// ===========================================================================
//  Batched kernels — each kernel handles a batch of triples in one launch.
//  Layout (max_n = max_n_tno across the batch):
//    K_batch[t, s0, a, b, d]    : t * 3*max_n³ + ...
//    M_batch[t, slot, l, a]     : t * 9*nocc*max_n + ...
//    T_part_batch[t, slot, c, d]: t * 9*max_n² + ...
//    T_ext_batch[t, s2, l, a, b]: t * 3*nocc*max_n² + ...
//    eps_tno_batch[t, a]        : t * max_n + a
// Per-triple buffers d_W, d_R3W, d_D_inv similarly slotted.
// ===========================================================================

/// Build all 6 W tensors for all triples in one launch.
///   blockIdx.z = t * 6 + p     (triple-index × perm-index)
///   blockIdx.x = a, blockIdx.y = b, threadIdx.x = c
/// Threads with a/b/c ≥ d_n_tno[t] return early (don't write).
///
/// Shared-memory optimisation: K[s0, a, b, *] (n elements) and
/// M[s0*3+s1, *, a] (nocc elements) are constant across threadIdx.x but were
/// re-read from global on every thread's inner-loop iteration. Cache them
/// once per block in shared memory; threads then read from L1-fast shmem.
/// Total shmem per block: (max_n + nocc) doubles ≤ ~1 KB for typical sizes.
__global__ void build_w_batched_kernel(
    real_t*       __restrict__ d_W,        // batch × 6 × max_n³
    const real_t* __restrict__ d_input,    // batch × per_triple_words
    const int*    __restrict__ d_n_tno,    // batch
    size_t per_triple_words,
    size_t off_K, size_t off_M, size_t off_T_part, size_t off_T_ext,
    int max_n, int nocc)
{
    extern __shared__ real_t shmem[];
    real_t* sh_K = shmem;                     // [blockDim.x] (≤ max_n)
    real_t* sh_M = shmem + blockDim.x;        // [nocc]

    const int t = blockIdx.z / 6;
    const int p = blockIdx.z % 6;
    const int a = blockIdx.x;
    const int b = blockIdx.y;
    const int c = threadIdx.x;
    const int n = d_n_tno[t];

    const int s0 = c_perm_table[p * 3 + 0];
    const int s1 = c_perm_table[p * 3 + 1];
    const int s2 = c_perm_table[p * 3 + 2];

    const size_t mn  = max_n;
    const size_t mn2 = mn * mn;
    const size_t mn3 = mn2 * mn;
    const size_t no  = nocc;

    // Pointers into this triple's input slot.
    const real_t* slot_base = d_input + static_cast<size_t>(t) * per_triple_words;
    const real_t* dK        = slot_base + off_K;
    const real_t* dM        = slot_base + off_M;
    const real_t* dT_part   = slot_base + off_T_part;
    const real_t* dT_ext    = slot_base + off_T_ext;

    // Cooperative shared-memory load. Use grid-stride loops so the loads
    // remain correct regardless of the relation between blockDim.x, n, and
    // nocc (avoids assuming blockDim.x ≥ nocc).
    if (a < n && b < n) {
        const real_t* K_ab_global = dK + static_cast<size_t>(s0) * mn3
                                       + (static_cast<size_t>(a) * mn + b) * mn;
        for (int d_idx = threadIdx.x; d_idx < n; d_idx += blockDim.x) {
            sh_K[d_idx] = K_ab_global[d_idx];
        }
    }
    {
        const real_t* M_pq_global = dM + static_cast<size_t>(s0 * 3 + s1) * no * mn;
        if (a < n) {
            for (int l_idx = threadIdx.x; l_idx < nocc; l_idx += blockDim.x) {
                sh_M[l_idx] = M_pq_global[static_cast<size_t>(l_idx) * mn + a];
            }
        }
    }
    __syncthreads();

    if (a >= n || b >= n || c >= n) return;

    // Particle: w[a,b,c] = Σ_d K[s0, a, b, d] · T_part[s2*3+s1, d, c]
    // T_part is packed in transposed layout [slot, d, c] so that adjacent
    // threads (c, c+1) read adjacent memory at fixed d → coalesced.
    const real_t* T_part_slot = dT_part + static_cast<size_t>(s2 * 3 + s1) * mn2;
    real_t part = real_t(0);
    #pragma unroll 4
    for (int d = 0; d < n; ++d) {
        part += sh_K[d] * T_part_slot[static_cast<size_t>(d) * mn + c];
    }

    // Hole: -Σ_l M[s0*3+s1, l, a] · T_ext[s2, l, b, c]
    // T_ext is packed in transposed layout [s2, l, b, c] so that adjacent
    // threads (c, c+1) read adjacent memory at fixed (l, b) → coalesced.
    const real_t* T_ext_slot = dT_ext + static_cast<size_t>(s2) * no * mn2;
    real_t hole = real_t(0);
    for (int l = 0; l < nocc; ++l) {
        const real_t mla = sh_M[l];
        if (mla == real_t(0)) continue;
        const real_t tlbc = T_ext_slot[(static_cast<size_t>(l) * mn + b) * mn + c];
        hole += mla * tlbc;
    }

    const size_t w_idx = (static_cast<size_t>(t) * 6 + p) * mn3
                       + (static_cast<size_t>(a) * mn + b) * mn + c;
    d_W[w_idx] = part - hole;
}

/// Apply r3 over (a,b,c) for all 6 perms × all triples in one launch.
__global__ void apply_r3_batched_kernel(
    const real_t* __restrict__ d_W,
    real_t*       __restrict__ d_R3W,
    const int*    __restrict__ d_n_tno,
    int max_n)
{
    const int t = blockIdx.z / 6;
    const int p = blockIdx.z % 6;
    const int a = blockIdx.x;
    const int b = blockIdx.y;
    const int c = threadIdx.x;
    const int n = d_n_tno[t];
    if (a >= n || b >= n || c >= n) return;

    const size_t mn  = max_n;
    const size_t mn2 = mn * mn;
    const size_t mn3 = mn2 * mn;
    const size_t base = (static_cast<size_t>(t) * 6 + p) * mn3;

    auto IDX = [mn, base](int a, int b, int c) {
        return base + (static_cast<size_t>(a) * mn + b) * mn + c;
    };
    const real_t Wabc = d_W[IDX(a, b, c)];
    const real_t Wcab = d_W[IDX(c, a, b)];
    const real_t Wbca = d_W[IDX(b, c, a)];
    const real_t Wcba = d_W[IDX(c, b, a)];
    const real_t Wacb = d_W[IDX(a, c, b)];
    const real_t Wbac = d_W[IDX(b, a, c)];
    d_R3W[IDX(a, b, c)] = real_t(4) * Wabc + Wcab + Wbca
                         - real_t(2) * Wcba - real_t(2) * Wacb - real_t(2) * Wbac;
}

/// Build inverse denominator for all triples in batch.
__global__ void build_d_inv_batched_kernel(
    real_t*       __restrict__ d_D_inv,    // batch × max_n³
    const real_t* __restrict__ d_input,    // for eps_tno
    const real_t* __restrict__ d_eps_sum,  // batch
    const int*    __restrict__ d_n_tno,
    const int*    __restrict__ d_d3_factor,
    size_t per_triple_words, size_t off_eps,
    int max_n)
{
    const int t = blockIdx.z;
    const int a = blockIdx.x;
    const int b = blockIdx.y;
    const int c = threadIdx.x;
    const int n = d_n_tno[t];
    if (a >= n || b >= n || c >= n) {
        // For unused entries, set D_inv = 0 so contract36 contributions vanish.
        if (a < max_n && b < max_n && c < max_n) {
            const size_t idx = (static_cast<size_t>(t) * max_n + a) * max_n * max_n
                             + static_cast<size_t>(b) * max_n + c;
            d_D_inv[idx] = real_t(0);
        }
        return;
    }
    const real_t* eps = d_input + static_cast<size_t>(t) * per_triple_words + off_eps;
    const real_t e_occ_sum = d_eps_sum[t];
    const int d3 = d_d3_factor[t];
    const real_t D = e_occ_sum - eps[a] - eps[b] - eps[c];

    const size_t idx = static_cast<size_t>(t) * max_n * max_n * max_n
                     + (static_cast<size_t>(a) * max_n + b) * max_n + c;
    d_D_inv[idx] = real_t(1) / (D * d3);
}

/// 36-contraction reduction for all triples × all (sz, sw) pairs.
/// Grid: (6, 6, batch); block: threads (e.g. 256). Shared mem reduction.
__global__ void contract36_batched_kernel(
    const real_t* __restrict__ d_W,
    const real_t* __restrict__ d_R3W,
    const real_t* __restrict__ d_D_inv,
    real_t*       __restrict__ d_partial,   // batch × 36
    const int*    __restrict__ d_n_tno,
    int max_n)
{
    extern __shared__ real_t shmem[];
    const int sz = blockIdx.x;
    const int sw = blockIdx.y;
    const int t  = blockIdx.z;
    const int code = c_perm_codes[sz * 6 + sw];
    const int n = d_n_tno[t];

    const size_t mn  = max_n;
    const size_t mn3 = mn * mn * mn;
    const real_t* W_sw = d_W   + (static_cast<size_t>(t) * 6 + sw) * mn3;
    const real_t* z_sz = d_R3W + (static_cast<size_t>(t) * 6 + sz) * mn3;
    const real_t* D_inv = d_D_inv + static_cast<size_t>(t) * mn3;

    real_t acc = real_t(0);
    const int n3 = n * n * n;
    for (int idx = threadIdx.x; idx < n3; idx += blockDim.x) {
        const int a = idx / (n * n);
        const int b = (idx / n) % n;
        const int c = idx % n;
        const size_t ijk = (static_cast<size_t>(a) * mn + b) * mn + c;

        size_t w_off;
        switch (code) {
            case 0: w_off = (static_cast<size_t>(a) * mn + b) * mn + c; break;
            case 1: w_off = (static_cast<size_t>(a) * mn + c) * mn + b; break;
            case 2: w_off = (static_cast<size_t>(b) * mn + a) * mn + c; break;
            case 3: w_off = (static_cast<size_t>(b) * mn + c) * mn + a; break;
            case 4: w_off = (static_cast<size_t>(c) * mn + a) * mn + b; break;
            case 5: w_off = (static_cast<size_t>(c) * mn + b) * mn + a; break;
            default: w_off = ijk;
        }
        acc += W_sw[w_off] * z_sz[ijk] * D_inv[ijk];
    }

    shmem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shmem[threadIdx.x] += shmem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_partial[static_cast<size_t>(t) * 36 + sz * 6 + sw] = shmem[0];
    }
}

} // anonymous namespace

// ===========================================================================
//  Device-pack kernels (GANSU_DLPNO_T_DEVICE_PACK)
//  Copy GPU-resident K/M/T into one d_input_ slot with the SAME padded/
//  transposed layout that host add_to_batch produces (so flush_batch's
//  energy kernels are unchanged). Each kernel is a 1D grid-stride over the
//  valid [0,n) element count; the slot was zeroed (cudaMemsetAsync) first so
//  padding (n→max_n) and empty (b<0) entries stay zero.
// ===========================================================================
__global__ void pack_K_dev_kernel(real_t* __restrict__ dst,        // H + off_K_
                                  const real_t* __restrict__ d_K,  // 3·n³ (s,a,b,d)
                                  int n, int mn) {
    const size_t total = static_cast<size_t>(3) * n * n * n;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(idx % n);
        size_t r = idx / n;
        const int b = static_cast<int>(r % n); r /= n;
        const int a = static_cast<int>(r % n); r /= n;
        const int s = static_cast<int>(r);          // 0..2
        const size_t dst_i =
            (static_cast<size_t>(s) * mn * mn + (static_cast<size_t>(a) * mn + b)) * mn + d;
        dst[dst_i] = d_K[idx];
    }
}

__global__ void pack_M_dev_kernel(real_t* __restrict__ dst,        // H + off_M_
                                  const real_t* __restrict__ d_M,  // 9·nocc·n (slot,l,a)
                                  int n, int nocc, int mn) {
    const size_t total = static_cast<size_t>(9) * nocc * n;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int a = static_cast<int>(idx % n);
        size_t r = idx / n;
        const int l = static_cast<int>(r % nocc); r /= nocc;
        const int slot = static_cast<int>(r);       // 0..8
        if (slot / 3 == slot % 3) continue;         // diagonal slots unused
        const size_t dst_i = (static_cast<size_t>(slot) * nocc + l) * mn + a;
        const size_t src_i = (static_cast<size_t>(slot) * nocc + l) * n  + a;
        dst[dst_i] = d_M[src_i];
    }
}

__global__ void pack_Tpart_dev_kernel(real_t* __restrict__ dst,    // H + off_T_part_
                                      const real_t* __restrict__ d_T_batch,
                                      const int* __restrict__ d_b_part,
                                      int n, int mn) {
    // element (slot, c, d); transpose [c,d] → [d,c]
    const size_t total = static_cast<size_t>(9) * n * n;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int d = static_cast<int>(idx % n);
        size_t r = idx / n;
        const int c = static_cast<int>(r % n); r /= n;
        const int slot = static_cast<int>(r);       // 0..8
        if (slot / 3 == slot % 3) continue;
        const int b = d_b_part[slot];
        if (b < 0) continue;
        dst[static_cast<size_t>(slot) * mn * mn + static_cast<size_t>(d) * mn + c] =
            d_T_batch[static_cast<size_t>(b) * n * n + static_cast<size_t>(c) * n + d];
    }
}

__global__ void pack_Text_dev_kernel(real_t* __restrict__ dst,     // H + off_T_ext_
                                     const real_t* __restrict__ d_T_batch,
                                     const int* __restrict__ d_b_il,
                                     const int* __restrict__ d_b_jl,
                                     const int* __restrict__ d_b_kl,
                                     int n, int nocc, int mn) {
    // element (which∈{0,1,2}, l, a, b); transpose [a,b] → [b,a]
    const size_t total = static_cast<size_t>(3) * nocc * n * n;
    const size_t mn2    = static_cast<size_t>(mn) * mn;
    const size_t no_mn2 = static_cast<size_t>(nocc) * mn2;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int bb = static_cast<int>(idx % n);   // the 'b' index
        size_t r = idx / n;
        const int a = static_cast<int>(r % n); r /= n;
        const int l = static_cast<int>(r % nocc); r /= nocc;
        const int which = static_cast<int>(r);      // 0,1,2
        const int* map = (which == 0) ? d_b_il : (which == 1) ? d_b_jl : d_b_kl;
        const int bidx = map[l];
        if (bidx < 0) continue;
        dst[static_cast<size_t>(which) * no_mn2 + static_cast<size_t>(l) * mn2
            + static_cast<size_t>(bb) * mn + a] =
            d_T_batch[static_cast<size_t>(bidx) * n * n + static_cast<size_t>(a) * n + bb];
    }
}

// ===========================================================================
//  Class implementation
// ===========================================================================

TripleTGpu::TripleTGpu(int max_n_tno, int nocc, int max_batch)
    : max_n_(max_n_tno), nocc_(nocc), max_batch_(max_batch)
{
    if (max_n_tno <= 0 || nocc <= 0 || max_batch <= 0) {
        active_ = false;
        return;
    }

    cudaError_t err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_));
    if (err != cudaSuccess) { active_ = false; return; }

    cublasHandle_t cublas;
    cublasStatus_t cs = cublasCreate(&cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
        active_ = false;
        return;
    }
    cublasSetStream(cublas, reinterpret_cast<cudaStream_t>(stream_));
    cublas_ = cublas;

    const size_t mn  = max_n_tno;
    const size_t mn2 = mn * mn;
    const size_t mn3 = mn2 * mn;
    const size_t no  = nocc;

    // Per-triple input layout (offsets in doubles within one slot)
    off_K_      = 0;
    off_M_      = 3 * mn3;
    off_T_part_ = off_M_ + 9 * no * mn;
    off_T_ext_  = off_T_part_ + 9 * mn2;
    off_eps_    = off_T_ext_ + 3 * no * mn2;
    per_triple_words_ = off_eps_ + mn;

    // Memory-aware batch sizing. The caller asks for up to max_batch slots,
    // but on TEOS-class systems (max_n_tno ~ 100+) the W, R3W, D_inv buffers
    // alone reach ~200 MB / triple, so a 1000-triple batch can need ~200 GB.
    // Cap to a budget computed from free GPU memory at construction time so
    // the alloc never fails and the caller chunked-flushes instead.
    const size_t per_triple_bytes = sizeof(real_t) * (
            per_triple_words_      // d_input slot
          + 6 * mn3                // d_W slot
          + 6 * mn3                // d_R3W slot
          + mn3                    // d_D_inv slot
          + 36                     // d_partial slot
          + 1                      // d_eps_sum slot
        ) + 2 * sizeof(int);       // d_n_tno + d_d3_factor slots

    size_t budget_bytes = 4ULL * 1024 * 1024 * 1024;   // hard cap 4 GB / instance
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        // Use at most 1/3 of currently-free memory so other helpers
        // (EriBuildGpu, TripleProjGpu) and per-triple scratch still fit.
        const size_t third = free_b / 3;
        budget_bytes = budget_bytes < third ? budget_bytes : third;
    }
    int max_batch_safe = std::max(1,
        static_cast<int>(budget_bytes / std::max<size_t>(per_triple_bytes, 1)));
    if (max_batch_safe < max_batch) max_batch_ = max_batch_safe;
    const size_t B = max_batch_;

    auto try_alloc_d = [](real_t** ptr, size_t bytes) -> bool {
        return cudaMalloc(ptr, bytes) == cudaSuccess;
    };
    auto try_alloc_di = [](int** ptr, size_t bytes) -> bool {
        return cudaMalloc(ptr, bytes) == cudaSuccess;
    };

    bool ok = true;
    ok &= try_alloc_d(&d_input_,     sizeof(real_t) * B * per_triple_words_);
    ok &= try_alloc_d(&d_W_,         sizeof(real_t) * B * 6 * mn3);
    ok &= try_alloc_d(&d_R3W_,       sizeof(real_t) * B * 6 * mn3);
    ok &= try_alloc_d(&d_D_inv_,     sizeof(real_t) * B * mn3);
    ok &= try_alloc_d(&d_partial_,   sizeof(real_t) * B * 36);
    ok &= try_alloc_d(&d_eps_sum_,   sizeof(real_t) * B);
    ok &= try_alloc_di(&d_n_tno_,    sizeof(int) * B);
    ok &= try_alloc_di(&d_d3_factor_, sizeof(int) * B);
    ok &= try_alloc_di(&d_b_il_,     sizeof(int) * nocc);
    ok &= try_alloc_di(&d_b_jl_,     sizeof(int) * nocc);
    ok &= try_alloc_di(&d_b_kl_,     sizeof(int) * nocc);
    ok &= try_alloc_di(&d_b_part_,   sizeof(int) * 9);

    if (ok) {
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_input_),
                             sizeof(real_t) * B * per_triple_words_,
                             cudaHostAllocDefault) == cudaSuccess);
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_partial_),
                             sizeof(real_t) * B * 36,
                             cudaHostAllocDefault) == cudaSuccess);
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_n_tno_),
                             sizeof(int) * B,
                             cudaHostAllocDefault) == cudaSuccess);
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_d3_),
                             sizeof(int) * B,
                             cudaHostAllocDefault) == cudaSuccess);
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_eps_sum_),
                             sizeof(real_t) * B,
                             cudaHostAllocDefault) == cudaSuccess);
    }

    if (!ok) {
        if (cublas_)            { cublasDestroy(cublas); cublas_ = nullptr; }
        if (stream_)            { cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
                                  stream_ = nullptr; }
        if (d_input_)           cudaFree(d_input_);
        if (d_W_)               cudaFree(d_W_);
        if (d_R3W_)             cudaFree(d_R3W_);
        if (d_D_inv_)           cudaFree(d_D_inv_);
        if (d_partial_)         cudaFree(d_partial_);
        if (d_eps_sum_)         cudaFree(d_eps_sum_);
        if (d_n_tno_)           cudaFree(d_n_tno_);
        if (d_d3_factor_)       cudaFree(d_d3_factor_);
        if (h_pinned_input_)    cudaFreeHost(h_pinned_input_);
        if (h_pinned_partial_)  cudaFreeHost(h_pinned_partial_);
        if (h_pinned_n_tno_)    cudaFreeHost(h_pinned_n_tno_);
        if (h_pinned_d3_)       cudaFreeHost(h_pinned_d3_);
        if (h_pinned_eps_sum_)  cudaFreeHost(h_pinned_eps_sum_);
        if (d_b_il_)            cudaFree(d_b_il_);
        if (d_b_jl_)            cudaFree(d_b_jl_);
        if (d_b_kl_)            cudaFree(d_b_kl_);
        if (d_b_part_)          cudaFree(d_b_part_);
        d_input_ = d_W_ = d_R3W_ = d_D_inv_ = d_partial_ = d_eps_sum_ = nullptr;
        d_n_tno_ = d_d3_factor_ = nullptr;
        d_b_il_ = d_b_jl_ = d_b_kl_ = d_b_part_ = nullptr;
        h_pinned_input_ = h_pinned_partial_ = h_pinned_eps_sum_ = nullptr;
        h_pinned_n_tno_ = h_pinned_d3_ = nullptr;
        active_ = false;
        return;
    }

    active_ = true;
    batch_n_ = 0;
}

TripleTGpu::~TripleTGpu() {
    if (cublas_)            cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
    if (stream_)            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    if (d_input_)           cudaFree(d_input_);
    if (d_W_)               cudaFree(d_W_);
    if (d_R3W_)             cudaFree(d_R3W_);
    if (d_D_inv_)           cudaFree(d_D_inv_);
    if (d_partial_)         cudaFree(d_partial_);
    if (d_eps_sum_)         cudaFree(d_eps_sum_);
    if (d_n_tno_)           cudaFree(d_n_tno_);
    if (d_d3_factor_)       cudaFree(d_d3_factor_);
    if (h_pinned_input_)    cudaFreeHost(h_pinned_input_);
    if (h_pinned_partial_)  cudaFreeHost(h_pinned_partial_);
    if (h_pinned_n_tno_)    cudaFreeHost(h_pinned_n_tno_);
    if (h_pinned_d3_)       cudaFreeHost(h_pinned_d3_);
    if (h_pinned_eps_sum_)  cudaFreeHost(h_pinned_eps_sum_);
    if (d_b_il_)            cudaFree(d_b_il_);
    if (d_b_jl_)            cudaFree(d_b_jl_);
    if (d_b_kl_)            cudaFree(d_b_kl_);
    if (d_b_part_)          cudaFree(d_b_part_);
}

void TripleTGpu::begin_batch() {
    batch_n_ = 0;
    batch_max_n_ = 0;
}

bool TripleTGpu::add_to_batch(
    int i, int j, int k,
    real_t eps_i, real_t eps_j, real_t eps_k,
    const TNOData& tno,
    const real_t* K_iadc,
    const std::array<std::vector<real_t>, 9>& M,
    const std::array<std::vector<real_t>, 9>& T_part_oriented,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc)
{
    if (!active_) return false;
    if (batch_n_ >= max_batch_) return false;
    const int n = tno.n_tno;
    if (n == 0) return true;     // no-op slot
    if (n > max_n_ || nocc != nocc_) return false;

    const size_t mn  = max_n_;
    const size_t mn2 = mn * mn;
    const size_t no  = nocc;

    // Pack into the batch slot. Pad unused regions with zero so kernels with
    // n < max_n don't read garbage.
    real_t* H = h_pinned_input_ + static_cast<size_t>(batch_n_) * per_triple_words_;
    std::memset(H, 0, sizeof(real_t) * per_triple_words_);

    // K: 3 × n³ → unpack into 3 × max_n³ (a, b, d slabs)
    real_t* dst_K = H + off_K_;
    for (int s = 0; s < 3; ++s) {
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                std::memcpy(dst_K + static_cast<size_t>(s) * mn * mn * mn
                                  + (static_cast<size_t>(a) * mn + b) * mn,
                            K_iadc + (static_cast<size_t>(s) * n * n
                                    + static_cast<size_t>(a) * n + b) * n,
                            sizeof(real_t) * n);
            }
        }
    }

    // M: 9 × no × n → 9 × no × max_n
    real_t* dst_M = H + off_M_;
    for (int slot = 0; slot < 9; ++slot) {
        if (M[slot].empty()) continue;
        for (int l = 0; l < nocc; ++l) {
            std::memcpy(dst_M + (static_cast<size_t>(slot) * no + l) * mn,
                        M[slot].data() + static_cast<size_t>(l) * n,
                        sizeof(real_t) * n);
        }
    }

    // T_part: pack in TRANSPOSED layout [slot, d, c] so GPU coalesces threads
    // along the c axis. Source is row-major (c, d); we transpose during copy.
    real_t* dst_Tp = H + off_T_part_;
    for (int slot = 0; slot < 9; ++slot) {
        if (T_part_oriented[slot].empty()) continue;
        const real_t* src = T_part_oriented[slot].data();
        real_t* dst_slot = dst_Tp + static_cast<size_t>(slot) * mn2;
        for (int c = 0; c < n; ++c) {
            for (int d = 0; d < n; ++d) {
                dst_slot[static_cast<size_t>(d) * mn + c] =
                    src[static_cast<size_t>(c) * n + d];
            }
        }
    }

    // T_ext: pack in TRANSPOSED layout [s2, l, b, c] so GPU coalesces threads
    // along the c axis. Source is row-major (a, b) where (a, b) ↔ (c, b) by
    // the t2 antisymmetry that the kernel exploits — we transpose during copy
    // to land in the kernel's expected coalesced layout.
    real_t* dst_Te = H + off_T_ext_;
    auto pack_ext = [&](const std::vector<std::vector<real_t>>& src,
                        size_t base_off) {
        for (int l = 0; l < nocc; ++l) {
            if (src[l].empty()) continue;
            const real_t* src_lp = src[l].data();
            real_t* dst_lp = dst_Te + base_off
                                    + static_cast<size_t>(l) * mn2;
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    // src_lp[a, b] (row-major a*n+b)
                    //   → dst[b, a] (row-major b*mn+a) so that, after the
                    //     kernel's t2-symmetry swap, [b, c] is innermost-c.
                    dst_lp[static_cast<size_t>(b) * mn + a] =
                        src_lp[static_cast<size_t>(a) * n + b];
                }
            }
        }
    };
    pack_ext(T_il_ext, 0 * no * mn2);
    pack_ext(T_jl_ext, 1 * no * mn2);
    pack_ext(T_kl_ext, 2 * no * mn2);

    // eps_tno
    std::memcpy(H + off_eps_, tno.eps_tno.data(), sizeof(real_t) * n);

    // Scalar metadata
    h_pinned_n_tno_[batch_n_]   = n;
    if (n > batch_max_n_) batch_max_n_ = n;
    int d3 = 1;
    if (i == j && j == k)        d3 = 6;
    else if (i == j || j == k)   d3 = 2;
    h_pinned_d3_[batch_n_]      = d3;
    h_pinned_eps_sum_[batch_n_] = eps_i + eps_j + eps_k;

    ++batch_n_;
    return true;
}

bool TripleTGpu::add_to_batch_device(
    int i, int j, int k,
    real_t eps_i, real_t eps_j, real_t eps_k,
    const TNOData& tno,
    real_t* d_K, real_t* d_M, real_t* d_T_batch,
    const int* b_il, const int* b_jl, const int* b_kl, const int* b_part,
    void* ev_eri, void* ev_proj, int nocc)
{
    if (!active_) return false;
    if (batch_n_ >= max_batch_) return false;
    const int n = tno.n_tno;
    if (n == 0) { device_packed_ = true; return true; }   // no-op slot
    if (n > max_n_ || nocc != nocc_) return false;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    // Cross-stream ordering: the pack reads K/M (eri stream) and T (proj
    // stream); wait for both producers' completion events before packing.
    if (ev_eri)  cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(ev_eri),  0);
    if (ev_proj) cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(ev_proj), 0);

    real_t* H = d_input_ + static_cast<size_t>(batch_n_) * per_triple_words_;
    // Zero the padded slot on device; pack kernels write only the [0,n) regions.
    cudaMemsetAsync(H, 0, sizeof(real_t) * per_triple_words_, stream);

    // Upload the per-triple proj inverse batch map (tiny; reused buffer — the
    // end-of-call stream sync guarantees the pack kernels read it before the
    // next triple overwrites it).
    cudaMemcpyAsync(d_b_il_,   b_il,   sizeof(int) * nocc_, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_jl_,   b_jl,   sizeof(int) * nocc_, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_kl_,   b_kl,   sizeof(int) * nocc_, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b_part_, b_part, sizeof(int) * 9,     cudaMemcpyHostToDevice, stream);

    const int TPB = 128;
    auto nblk = [](size_t total) -> int {
        size_t b = (total + TPB - 1) / TPB;
        if (b < 1) b = 1; if (b > 65535) b = 65535;
        return static_cast<int>(b);
    };
    pack_K_dev_kernel<<<nblk(static_cast<size_t>(3) * n * n * n), TPB, 0, stream>>>(
        H + off_K_, d_K, n, max_n_);
    pack_M_dev_kernel<<<nblk(static_cast<size_t>(9) * nocc_ * n), TPB, 0, stream>>>(
        H + off_M_, d_M, n, nocc_, max_n_);
    pack_Tpart_dev_kernel<<<nblk(static_cast<size_t>(9) * n * n), TPB, 0, stream>>>(
        H + off_T_part_, d_T_batch, d_b_part_, n, max_n_);
    pack_Text_dev_kernel<<<nblk(static_cast<size_t>(3) * nocc_ * n * n), TPB, 0, stream>>>(
        H + off_T_ext_, d_T_batch, d_b_il_, d_b_jl_, d_b_kl_, n, nocc_, max_n_);

    // eps_tno is host-resident (from TNOData) → small H2D into the slot.
    cudaMemcpyAsync(H + off_eps_, tno.eps_tno.data(),
                    sizeof(real_t) * n, cudaMemcpyHostToDevice, stream);

    // Serialize: the pack must finish reading the producers' scratch
    // (d_K/d_M/d_T_batch, overwritten on the next triple) and d_b_* before the
    // caller issues the next eri/proj. The pack is a device-bandwidth memset +
    // 4 small kernels (~µs), so this per-triple sync is cheap vs the eliminated
    // host memset/pack.
    cudaStreamSynchronize(stream);

    // Host scalar metadata (identical to the host add_to_batch path).
    h_pinned_n_tno_[batch_n_] = n;
    if (n > batch_max_n_) batch_max_n_ = n;
    int d3 = 1;
    if (i == j && j == k)        d3 = 6;
    else if (i == j || j == k)   d3 = 2;
    h_pinned_d3_[batch_n_]      = d3;
    h_pinned_eps_sum_[batch_n_] = eps_i + eps_j + eps_k;
    ++batch_n_;
    device_packed_ = true;
    return true;
}

real_t TripleTGpu::flush_batch() {
    if (!active_ || batch_n_ == 0) {
        batch_n_ = 0;
        return real_t(0);
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    const int N = batch_n_;
    const int mn = max_n_;          // slot stride (constant across batches)
    const int gn = batch_max_n_;    // tight grid bound: actual max n in this batch

    // -- Single big H→D upload of the batched input + scalars. In device-pack
    //    mode the per-triple slots were written directly into d_input_ by the
    //    pack kernels, so skip the input H2D (scalars stay host-staged).
    if (!device_packed_) {
        check_cuda_(cudaMemcpyAsync(d_input_, h_pinned_input_,
                                    sizeof(real_t) * N * per_triple_words_,
                                    cudaMemcpyHostToDevice, stream),
                    "memcpy batched input");
    }
    check_cuda_(cudaMemcpyAsync(d_n_tno_, h_pinned_n_tno_,
                                sizeof(int) * N,
                                cudaMemcpyHostToDevice, stream),
                "memcpy n_tno");
    check_cuda_(cudaMemcpyAsync(d_d3_factor_, h_pinned_d3_,
                                sizeof(int) * N,
                                cudaMemcpyHostToDevice, stream),
                "memcpy d3");
    check_cuda_(cudaMemcpyAsync(d_eps_sum_, h_pinned_eps_sum_,
                                sizeof(real_t) * N,
                                cudaMemcpyHostToDevice, stream),
                "memcpy eps_sum");

    // -- Build all W tensors in one launch.
    // Shared memory: max_n (for K row) + nocc (for M column) doubles per block.
    // Grid is sized to batch_max_n_ (gn) — the actual max n_tno in this batch
    // — rather than max_n_ (mn = nvir). This avoids launching huge numbers of
    // early-return threads when actual TNOs are much smaller than nvir
    // (typical for water clusters: avg n=24 vs nvir=120 → 125× waste).
    // Slot strides remain mn for layout consistency.
    {
        dim3 grid(gn, gn, 6 * N);
        dim3 block(gn);
        const size_t shmem_bytes = sizeof(real_t) * (gn + nocc_);
        build_w_batched_kernel<<<grid, block, shmem_bytes, stream>>>(
            d_W_, d_input_, d_n_tno_,
            per_triple_words_, off_K_, off_M_, off_T_part_, off_T_ext_,
            mn, nocc_);
    }

    // -- r3 for all W's
    {
        dim3 grid(gn, gn, 6 * N);
        dim3 block(gn);
        apply_r3_batched_kernel<<<grid, block, 0, stream>>>(
            d_W_, d_R3W_, d_n_tno_, mn);
    }

    // -- D_inv (one slab per triple)
    {
        dim3 grid(gn, gn, N);
        dim3 block(gn);
        build_d_inv_batched_kernel<<<grid, block, 0, stream>>>(
            d_D_inv_, d_input_, d_eps_sum_, d_n_tno_, d_d3_factor_,
            per_triple_words_, off_eps_, mn);
    }

    // -- 36 contractions × N triples
    {
        const int threads = 256;
        dim3 grid(6, 6, N);
        size_t shmem_bytes = sizeof(real_t) * threads;
        contract36_batched_kernel<<<grid, threads, shmem_bytes, stream>>>(
            d_W_, d_R3W_, d_D_inv_, d_partial_, d_n_tno_, mn);
    }

    // -- D→H copy of N × 36 partial sums
    check_cuda_(cudaMemcpyAsync(h_pinned_partial_, d_partial_,
                                sizeof(real_t) * N * 36,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy partial");
    check_cuda_(cudaStreamSynchronize(stream), "sync flush");

    real_t total = real_t(0);
    for (int t = 0; t < N; ++t) {
        real_t et = real_t(0);
        for (int q = 0; q < 36; ++q) et += h_pinned_partial_[t * 36 + q];
        total += real_t(2) * et;
    }
    batch_n_ = 0;
    return total;
}

real_t TripleTGpu::compute_triple(
    int i, int j, int k,
    real_t eps_i, real_t eps_j, real_t eps_k,
    const TNOData& tno,
    const real_t* K_iadc,
    const std::array<std::vector<real_t>, 9>& M,
    const std::array<std::vector<real_t>, 9>& T_part_oriented,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc)
{
    begin_batch();
    if (!add_to_batch(i, j, k, eps_i, eps_j, eps_k, tno, K_iadc, M,
                      T_part_oriented, T_il_ext, T_jl_ext, T_kl_ext, nocc)) {
        return real_t(0);
    }
    return flush_batch();
}

#else // GANSU_CPU_ONLY: stub-out

TripleTGpu::TripleTGpu(int, int, int)  : active_(false) {}
TripleTGpu::~TripleTGpu()              = default;
void   TripleTGpu::begin_batch()       {}
bool   TripleTGpu::add_to_batch(int,int,int,real_t,real_t,real_t,
                                const TNOData&,const real_t*,
                                const std::array<std::vector<real_t>, 9>&,
                                const std::array<std::vector<real_t>, 9>&,
                                const std::vector<std::vector<real_t>>&,
                                const std::vector<std::vector<real_t>>&,
                                const std::vector<std::vector<real_t>>&,
                                int) { return false; }
bool   TripleTGpu::add_to_batch_device(int,int,int,real_t,real_t,real_t,
                                const TNOData&,real_t*,real_t*,real_t*,
                                const int*,const int*,const int*,const int*,
                                void*,void*,int) { return false; }
real_t TripleTGpu::flush_batch()       { return real_t(0); }
real_t TripleTGpu::compute_triple(int, int, int,
    real_t, real_t, real_t,
    const TNOData&,
    const real_t*,
    const std::array<std::vector<real_t>, 9>&,
    const std::array<std::vector<real_t>, 9>&,
    const std::vector<std::vector<real_t>>&,
    const std::vector<std::vector<real_t>>&,
    const std::vector<std::vector<real_t>>&,
    int)
{
    return real_t(0);
}

#endif

} // namespace gansu
