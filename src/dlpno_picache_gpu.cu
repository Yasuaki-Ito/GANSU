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
    real_t*  d_pi_T_stack  = nullptr;     // [Σ n_pno²·nocc²] unpadded
    real_t*  h_pi_T_stack  = nullptr;     // pinned mirror
    size_t   pi_T_stack_total = 0;        // sum of n_pno²·nocc²

    // Borrowed cuBLAS handle (thread_local from GPUHandle).
    cublasHandle_t cublas = nullptr;

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
        d_barS_pad = d_Y_pad = d_Y_pad_T = d_half_pad = d_pi_pad = nullptr;
        h_Y_pad = h_pi_pad = nullptr;
        d_pair_lookup = d_setup_i = d_n_pno = nullptr;
        d_idx_offset = nullptr;
        d_pi_T_stack = nullptr;
        h_pi_T_stack = nullptr;
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
__global__ void transpose_Y_pad_kernel(
    const real_t* __restrict__ d_Y_pad,
    real_t*       __restrict__ d_Y_pad_T,
    int N_pair, int max_n)
{
    const int idx = blockIdx.x;
    if (idx >= N_pair) return;
    const size_t pair_off = static_cast<size_t>(idx)
                          * static_cast<size_t>(max_n)
                          * static_cast<size_t>(max_n);
    for (int r = threadIdx.y; r < max_n; r += blockDim.y) {
        for (int c = threadIdx.x; c < max_n; c += blockDim.x) {
            d_Y_pad_T[pair_off + static_cast<size_t>(r) * max_n + c] =
                d_Y_pad[pair_off + static_cast<size_t>(c) * max_n + r];
        }
    }
}

__global__ void pack_pi_T_stack_kernel(
    const real_t* __restrict__ d_pi_pad,
    const int*    __restrict__ d_pair_lookup,
    const int*    __restrict__ d_setup_i,
    const int*    __restrict__ d_n_pno,
    const size_t* __restrict__ d_idx_offset,
    real_t*       __restrict__ d_pi_T_stack,
    int N_pair, int nocc, int max_n)
{
    const int i_ij = blockIdx.x;
    const int kl   = blockIdx.y;          // = k * nocc + l
    if (i_ij >= N_pair || kl >= nocc * nocc) return;

    const int n_ij = d_n_pno[i_ij];
    if (n_ij == 0) return;

    const int idx_kl = d_pair_lookup[kl];
    const int n_kl   = d_n_pno[idx_kl];
    const int k      = kl / nocc;
    const int s_i_kl = d_setup_i[idx_kl];

    // Step 6.6 fix — block (TILE × TILE) with strided per-thread loop so
    // we never exceed CUDA's 1024 threads/block when max_n > 32 (small
    // dense molecules like Benzene cc-pVDZ have max_n ~ 30-50).
    for (int a = threadIdx.y; a < n_ij; a += blockDim.y) {
        for (int d = threadIdx.x; d < n_ij; d += blockDim.x) {
            real_t v = real_t(0);
            if (n_kl > 0) {
                const real_t* src = d_pi_pad
                    + (static_cast<size_t>(i_ij) * static_cast<size_t>(N_pair)
                       + static_cast<size_t>(idx_kl))
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
    s.stride_outer = s.stride_pair * static_cast<long long>(N_pair_);

    const size_t n_pair_pair = static_cast<size_t>(N_pair_)
                             * static_cast<size_t>(N_pair_);
    const size_t bytes_full  = n_pair_pair * static_cast<size_t>(max_n_)
                             * static_cast<size_t>(max_n_) * sizeof(real_t);
    const size_t bytes_outer = static_cast<size_t>(N_pair_)
                             * static_cast<size_t>(max_n_)
                             * static_cast<size_t>(max_n_) * sizeof(real_t);

    // Step 6.1 stacked-mode budget (only when pair_lookup + setup_i + nocc>0).
    const bool want_stacked = pair_lookup && setup_i_per_pair && nocc_ > 0
                              && !pair_lookup_.empty()
                              && !setup_i_per_pair_.empty()
                              && static_cast<int>(pair_lookup_.size()) == nocc_ * nocc_
                              && static_cast<int>(setup_i_per_pair_.size()) == N_pair_;
    size_t pi_T_total = 0;
    if (want_stacked) {
        for (int i = 0; i < N_pair_; ++i) {
            const size_t n = static_cast<size_t>(n_pno_[i]);
            pi_T_total += n * n
                        * static_cast<size_t>(nocc_) * static_cast<size_t>(nocc_);
        }
    }
    const size_t bytes_pi_T = pi_T_total * sizeof(real_t);

    // Probe free memory; we need 2 × bytes_full + 2 × bytes_outer + bytes_pi_T
    // plus a safety margin. Fall back if not enough.
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        // Step 6.7: +1 outer (d_Y_pad_T) for oooo_lad coalescing.
        const size_t need = 2 * bytes_full + 3 * bytes_outer + bytes_pi_T
                          + (size_t)64 * 1024 * 1024;  // 64 MB margin
        if (need > free_b) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
    }

    try {
        check_cuda_(cudaMalloc(&s.d_barS_pad, bytes_full),  "cudaMalloc d_barS_pad");
        check_cuda_(cudaMalloc(&s.d_pi_pad,   bytes_full),  "cudaMalloc d_pi_pad");
        check_cuda_(cudaMalloc(&s.d_Y_pad,    bytes_outer), "cudaMalloc d_Y_pad");
        check_cuda_(cudaMalloc(&s.d_Y_pad_T,  bytes_outer), "cudaMalloc d_Y_pad_T");
        check_cuda_(cudaMalloc(&s.d_half_pad, bytes_outer), "cudaMalloc d_half_pad");

        check_cuda_(cudaMallocHost(&s.h_Y_pad,  bytes_outer), "cudaMallocHost h_Y_pad");
        check_cuda_(cudaMallocHost(&s.h_pi_pad, bytes_full),  "cudaMallocHost h_pi_pad");
    } catch (const std::exception&) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

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
            check_cuda_(cudaMallocHost(&s.h_pi_T_stack, bytes_pi_T),
                        "cudaMallocHost h_pi_T_stack");

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

            s.pi_T_stack_total = pi_T_total;
            stacked_ = true;
        } catch (const std::exception&) {
            // Free stacked-mode buffers; keep Step 6.0 buffers — caller can
            // still use rebuild() and rebuild_with_stack() falls back to CPU
            // pi_T_stack assembly internally.
            if (s.d_pair_lookup) { cudaFree(s.d_pair_lookup); s.d_pair_lookup = nullptr; }
            if (s.d_setup_i)     { cudaFree(s.d_setup_i);     s.d_setup_i     = nullptr; }
            if (s.d_n_pno)       { cudaFree(s.d_n_pno);       s.d_n_pno       = nullptr; }
            if (s.d_idx_offset)  { cudaFree(s.d_idx_offset);  s.d_idx_offset  = nullptr; }
            if (s.d_pi_T_stack)  { cudaFree(s.d_pi_T_stack);  s.d_pi_T_stack  = nullptr; }
            if (s.h_pi_T_stack)  { cudaFreeHost(s.h_pi_T_stack); s.h_pi_T_stack = nullptr; }
            stacked_ = false;
        }
    }

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

    // -------- Pad host barS into a contiguous [N_pair·N_pair·max_n²] buffer.
    // Layout: idx = (i_ij·N_pair + i_kl)·max_n² + r·max_n + c
    //         row-major within each (max_n × max_n) padded block.
    std::vector<real_t> h_barS_pad(n_pair_pair
                                   * static_cast<size_t>(max_n_)
                                   * static_cast<size_t>(max_n_),
                                   real_t{0});

    #pragma omp parallel for schedule(static)
    for (long long i_ij = 0; i_ij < static_cast<long long>(N_pair_); ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) continue;
        for (int i_kl = 0; i_kl < N_pair_; ++i_kl) {
            const int n_kl = n_pno_[i_kl];
            if (n_kl == 0) continue;
            const RowMatXd& bs = barS_cache[i_ij][i_kl];
            // Skip empties defensively (a 0×0 with size 0).
            if (bs.rows() == 0 || bs.cols() == 0) continue;
            const size_t base = (static_cast<size_t>(i_ij)
                               * static_cast<size_t>(N_pair_)
                               + static_cast<size_t>(i_kl))
                              * static_cast<size_t>(max_n_)
                              * static_cast<size_t>(max_n_);
            for (int r = 0; r < n_ij; ++r) {
                std::memcpy(&h_barS_pad[base + static_cast<size_t>(r) * max_n_],
                            bs.data() + static_cast<ptrdiff_t>(r) * n_kl,
                            static_cast<size_t>(n_kl) * sizeof(real_t));
            }
        }
    }

    // One-time H→D upload of barS_pad (synchronous; we don't reuse stream).
    check_cuda_(cudaMemcpy(s.d_barS_pad, h_barS_pad.data(),
                           bytes_full, cudaMemcpyHostToDevice),
                "cudaMemcpy barS H2D");

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
    rebuild_gpu_kernels_(Y_old);
    download_pi_cache_(pi_cache_out);
#endif // !GANSU_CPU_ONLY
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
    Impl& s = *p_;
    const int N      = N_pair_;
    const int max_n  = max_n_;
    const long long stride_pair = s.stride_pair;          // max_n²
    const int ib = pair_begin_;
    const int ie = pair_end_;

    const size_t bytes_outer = static_cast<size_t>(N)
                             * static_cast<size_t>(max_n)
                             * static_cast<size_t>(max_n) * sizeof(real_t);

    // -------- Pad Y_old into pinned host buffer h_Y_pad.
    // Layout: h_Y_pad[i_kl · max_n² + r · max_n + c]
    std::memset(s.h_Y_pad, 0, bytes_outer);
    #pragma omp parallel for schedule(static)
    for (long long i_kl = 0; i_kl < static_cast<long long>(N); ++i_kl) {
        const int n_kl = n_pno_[i_kl];
        if (n_kl == 0) continue;
        const real_t* src = Y_old[i_kl].data();
        real_t* dst = s.h_Y_pad + static_cast<size_t>(i_kl) * stride_pair;
        for (int r = 0; r < n_kl; ++r) {
            std::memcpy(dst + static_cast<size_t>(r) * max_n,
                        src + static_cast<ptrdiff_t>(r) * n_kl,
                        static_cast<size_t>(n_kl) * sizeof(real_t));
        }
    }

    // H2D Y_pad (full N × max_n²; all i_kl columns of Y are needed regardless
    // of slab restriction on output rows).
    check_cuda_(cudaMemcpy(s.d_Y_pad, s.h_Y_pad,
                           bytes_outer, cudaMemcpyHostToDevice),
                "rebuild H2D Y_pad");

    // Step 6.7: Build d_Y_pad_T (per-pair transpose) for oooo_lad coalescing.
    // Negligible cost compared to the H2D above; same lifetime as d_Y_pad.
    {
        const int block_dim = (max_n <= 16) ? max_n : 16;
        const dim3 t_block(block_dim, block_dim);
        const dim3 t_grid(static_cast<unsigned>(N));
        transpose_Y_pad_kernel<<<t_grid, t_block>>>(
            s.d_Y_pad, s.d_Y_pad_T, N, max_n);
        check_cuda_(cudaGetLastError(), "transpose_Y_pad_kernel launch");
    }

    // -------- Per-i_ij outer loop, two strided batched DGEMMs over i_kl.
    //          Only computes the slab range [pair_begin_, pair_end_).
    const real_t one  = 1.0;
    const real_t zero = 0.0;

    // Conventions:
    //   - All buffers in memory are row-major padded (max_n × max_n) per block.
    //   - cuBLAS reads them as column-major X_col = X_row^T (ld=max_n).
    //   - Goal: half_row = barS_row · Y_row,   pi_row = half_row · barS_row^T
    //     ⇔ half_col = Y_col · barS_col,      pi_col  = barS_col^T · half_col.
    //   - Padding: barS_row valid (n_ij × n_kl) upper-left, Y_row valid
    //     (n_kl × n_kl) upper-left, both zero outside. Stage 1 writes only
    //     the first n_ij columns of half_col (= first n_ij rows of half_row).
    //     Half_row's (n_kl..max_n-1) columns are zero by construction since
    //     they sum over zeroed rows of Y_row.
    for (int i_ij = ib; i_ij < ie; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) continue;

        real_t* dA_row = s.d_barS_pad
                      + static_cast<size_t>(i_ij)
                      * static_cast<size_t>(N) * stride_pair;
        real_t* dC_row = s.d_pi_pad
                      + static_cast<size_t>(i_ij)
                      * static_cast<size_t>(N) * stride_pair;

        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/ max_n, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            s.d_Y_pad,    /*lda=*/ max_n, /*strideA=*/ stride_pair,
            dA_row,       /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            s.d_half_pad, /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            N), "stage1 strided batched");

        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            /*m=*/ n_ij, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            dA_row,       /*lda=*/ max_n, /*strideA=*/ stride_pair,
            s.d_half_pad, /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            dC_row,       /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            N), "stage2 strided batched");
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

    if (skip_host_pi) {
#ifndef GANSU_CPU_ONLY
        MultiGpuManager::DeviceGuard _guard(device_id_);
        rebuild_gpu_kernels_(Y_old);
#endif
    } else {
        // Default path: produce pi_cache (GPU when active_, CPU otherwise).
        rebuild(Y_old, pi_cache_out);
    }

    if (!stacked_) {
        // CPU fallback for pi_T_stack — same algorithm as the original
        // middleCols loop, just relocated here so the caller sees one API.
        // (Only reachable when skip_host_pi was false above, since skip_host_pi
        //  implies stacked_.)
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

    // Per-pair offsets into d_pi_T_stack — needed to compute the slab range.
    std::vector<size_t> idx_offset_host(static_cast<size_t>(N + 1), 0);
    for (int i = 0; i < N; ++i) {
        const size_t n = static_cast<size_t>(n_pno_[i]);
        idx_offset_host[i + 1] = idx_offset_host[i]
            + n * n
            * static_cast<size_t>(nocc)
            * static_cast<size_t>(nocc);
    }

    // d_pi_pad now holds the latest pi_cache values (set by the rebuild()
    // call above — those buffers are members of *p_ and persist).
    //
    // Zero d_pi_T_stack so empty (i_ij or i_kl) cells stay clean. The kernel
    // also short-circuits empty inputs but we cover the n_kl=0 ⇒ skipped
    // dst region just in case. Slab-aware: only zero the slab portion.
    const size_t bytes_pi_T_slab = (idx_offset_host[ie] - idx_offset_host[ib])
                                 * sizeof(real_t);
    if (bytes_pi_T_slab > 0) {
        check_cuda_(cudaMemsetAsync(
            s.d_pi_T_stack + idx_offset_host[ib],
            0, bytes_pi_T_slab, /*stream=*/0),
            "memset d_pi_T_stack (slab)");
    }

    // Launch the pack kernel. We use the FULL grid (N × nocc²) — running
    // a wider grid is cheap (~10 µs) and avoids adding slab-aware indexing
    // to the kernel. Rows outside the slab read from stale/uninitialised
    // d_pi_pad (rebuild() above wrote only slab rows), so d_pi_T_stack
    // outside the slab will be garbage on this device. That's fine because
    // the D2H below only copies the slab portion; peer GPUs populate the
    // other slabs into the shared host h_pi_T_stack buffers / output vectors.
    if (slab_n > 0 && nocc > 0 && max_n > 0) {
        constexpr int TILE = 16;
        const int tile_x = (max_n < TILE) ? max_n : TILE;
        const int tile_y = (max_n < TILE) ? max_n : TILE;
        dim3 block(static_cast<unsigned>(tile_x),
                   static_cast<unsigned>(tile_y), 1);
        dim3 grid(static_cast<unsigned>(N),
                  static_cast<unsigned>(nocc * nocc), 1);
        pack_pi_T_stack_kernel<<<grid, block>>>(
            s.d_pi_pad,
            s.d_pair_lookup,
            s.d_setup_i,
            s.d_n_pno,
            s.d_idx_offset,
            s.d_pi_T_stack,
            N, nocc, max_n);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            throw std::runtime_error(
                std::string("PiCacheGpu pack kernel launch failed: ")
                + cudaGetErrorString(e));
        }
    }

    // D2H — only the slab range of pi_T_stack.
    if (bytes_pi_T_slab > 0) {
        check_cuda_(cudaMemcpy(s.h_pi_T_stack + idx_offset_host[ib],
                               s.d_pi_T_stack + idx_offset_host[ib],
                               bytes_pi_T_slab, cudaMemcpyDeviceToHost),
                    "D2H pi_T_stack (slab)");
    }

    // Host scatter: per-pair contiguous segment → pi_T_stack_out[idx] (n_ij × nocc²·n_ij).
    // The device layout already matches the unpadded host layout, so this
    // is a single memcpy per pair. Only fills the slab range; other slabs
    // are populated by peer PiCacheGpu instances on other GPUs.
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
                    s.h_pi_T_stack + idx_offset_host[i_ij],
                    bytes);
    }
#endif // !GANSU_CPU_ONLY
}

} // namespace gansu
