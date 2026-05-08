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
    long long stride_pair = 0;        // max_n²
    long long stride_outer = 0;       // N_pair · max_n²

    // Persistent device buffers
    real_t* d_barS_pad = nullptr;     // [N_pair · N_pair · max_n²]
    real_t* d_Y_pad    = nullptr;     // [N_pair · max_n²]
    real_t* d_half_pad = nullptr;     // [N_pair · max_n²] scratch
    real_t* d_pi_pad   = nullptr;     // [N_pair · N_pair · max_n²]

    // Pinned host buffers for fast h<->d
    real_t* h_Y_pad  = nullptr;
    real_t* h_pi_pad = nullptr;

    // Borrowed cuBLAS handle (thread_local from GPUHandle).
    cublasHandle_t cublas = nullptr;

    void free_all() {
        if (d_barS_pad) cudaFree(d_barS_pad);
        if (d_Y_pad)    cudaFree(d_Y_pad);
        if (d_half_pad) cudaFree(d_half_pad);
        if (d_pi_pad)   cudaFree(d_pi_pad);
        if (h_Y_pad)    cudaFreeHost(h_Y_pad);
        if (h_pi_pad)   cudaFreeHost(h_pi_pad);
        d_barS_pad = d_Y_pad = d_half_pad = d_pi_pad = nullptr;
        h_Y_pad = h_pi_pad = nullptr;
    }
};

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
                       int max_n)
    : n_pno_(n_pno_per_pair),
      N_pair_(static_cast<int>(n_pno_per_pair.size())),
      max_n_(max_n)
{
    barS_cache_ref_ = &barS_cache;

#ifndef GANSU_CPU_ONLY
    // Decide whether to take the GPU path.
    if (!gpu::gpu_available() || N_pair_ == 0 || max_n_ == 0) {
        active_ = false;
        return;
    }

    p_ = new Impl();
    Impl& s = *p_;
    s.N_pair      = N_pair_;
    s.max_n       = max_n_;
    s.stride_pair = static_cast<long long>(max_n_) * max_n_;
    s.stride_outer = s.stride_pair * static_cast<long long>(N_pair_);

    const size_t n_pair_pair = static_cast<size_t>(N_pair_)
                             * static_cast<size_t>(N_pair_);
    const size_t bytes_full  = n_pair_pair * static_cast<size_t>(max_n_)
                             * static_cast<size_t>(max_n_) * sizeof(real_t);
    const size_t bytes_outer = static_cast<size_t>(N_pair_)
                             * static_cast<size_t>(max_n_)
                             * static_cast<size_t>(max_n_) * sizeof(real_t);

    // Probe free memory; we need 2 × bytes_full + 2 × bytes_outer plus a
    // safety margin. Fall back to CPU if not enough.
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
        const size_t need = 2 * bytes_full + 2 * bytes_outer
                          + (size_t)64 * 1024 * 1024;  // 64 MB margin
        if (need > free_b) {
            delete p_; p_ = nullptr; active_ = false; return;
        }
    }

    try {
        check_cuda_(cudaMalloc(&s.d_barS_pad, bytes_full),  "cudaMalloc d_barS_pad");
        check_cuda_(cudaMalloc(&s.d_pi_pad,   bytes_full),  "cudaMalloc d_pi_pad");
        check_cuda_(cudaMalloc(&s.d_Y_pad,    bytes_outer), "cudaMalloc d_Y_pad");
        check_cuda_(cudaMalloc(&s.d_half_pad, bytes_outer), "cudaMalloc d_half_pad");

        check_cuda_(cudaMallocHost(&s.h_Y_pad,  bytes_outer), "cudaMallocHost h_Y_pad");
        check_cuda_(cudaMallocHost(&s.h_pi_pad, bytes_full),  "cudaMallocHost h_pi_pad");
    } catch (const std::exception&) {
        s.free_all();
        delete p_; p_ = nullptr; active_ = false;
        return;
    }

    s.cublas = gpu::GPUHandle::cublas();
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
// ---------------------------------------------------------------------------
void PiCacheGpu::rebuild_cpu_(
    const std::vector<std::vector<real_t>>& Y_old,
    std::vector<std::vector<RowMatXd>>& pi_cache_out)
{
    const auto& barS_cache = *barS_cache_ref_;

    #pragma omp parallel for schedule(static)
    for (long long i_ij = 0; i_ij < static_cast<long long>(N_pair_); ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) {
            for (int i_kl = 0; i_kl < N_pair_; ++i_kl) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
            }
            continue;
        }
        for (int i_kl = 0; i_kl < N_pair_; ++i_kl) {
            const int n_kl = n_pno_[i_kl];
            if (n_kl == 0) {
                pi_cache_out[i_ij][i_kl].resize(0, 0);
                continue;
            }
            const RowMatXd& barS = barS_cache[i_ij][i_kl];
            Eigen::Map<const RowMatXd> Y_canon(
                Y_old[i_kl].data(), n_kl, n_kl);
            const RowMatXd half = barS * Y_canon;            // n_ij × n_kl
            pi_cache_out[i_ij][i_kl].noalias() =
                half * barS.transpose();                     // n_ij × n_ij
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
    Impl& s = *p_;
    const int N      = N_pair_;
    const int max_n  = max_n_;
    const long long stride_pair = s.stride_pair;          // max_n²

    const size_t bytes_outer = static_cast<size_t>(N)
                             * static_cast<size_t>(max_n)
                             * static_cast<size_t>(max_n) * sizeof(real_t);
    const size_t bytes_full  = static_cast<size_t>(N)
                             * static_cast<size_t>(N)
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

    // H2D
    check_cuda_(cudaMemcpy(s.d_Y_pad, s.h_Y_pad,
                           bytes_outer, cudaMemcpyHostToDevice),
                "rebuild H2D Y_pad");

    // -------- Per-i_ij outer loop, two strided batched DGEMMs over i_kl.
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
    for (int i_ij = 0; i_ij < N; ++i_ij) {
        const int n_ij = n_pno_[i_ij];
        if (n_ij == 0) continue;

        real_t* dA_row = s.d_barS_pad
                      + static_cast<size_t>(i_ij)
                      * static_cast<size_t>(N) * stride_pair;
        real_t* dC_row = s.d_pi_pad
                      + static_cast<size_t>(i_ij)
                      * static_cast<size_t>(N) * stride_pair;

        // Stage 1: half_col = Y_col · barS_col   (TransA=N, TransB=N)
        //   m=max_n (full rows of half_col, since rows ≥ n_kl trivially zero),
        //   n=n_ij  (clip to valid cols of half_col = valid rows of half_row),
        //   k=max_n.
        check_cublas_(cublasDgemmStridedBatched(
            s.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/ max_n, /*n=*/ n_ij, /*k=*/ max_n,
            &one,
            s.d_Y_pad,    /*lda=*/ max_n, /*strideA=*/ stride_pair,
            dA_row,       /*ldb=*/ max_n, /*strideB=*/ stride_pair,
            &zero,
            s.d_half_pad, /*ldc=*/ max_n, /*strideC=*/ stride_pair,
            N), "stage1 strided batched");

        // Stage 2: pi_col = barS_col^T · half_col   (TransA=T, TransB=N)
        //   m=n_ij, n=n_ij, k=max_n.
        //   k=max_n is safe because half_col rows ≥ n_kl are zero (see above).
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

    // D2H
    check_cuda_(cudaMemcpy(s.h_pi_pad, s.d_pi_pad,
                           bytes_full, cudaMemcpyDeviceToHost),
                "rebuild D2H pi_pad");

    // -------- Unpad h_pi_pad → pi_cache_out (host).
    #pragma omp parallel for schedule(static)
    for (long long i_ij = 0; i_ij < static_cast<long long>(N); ++i_ij) {
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
#endif // !GANSU_CPU_ONLY
}

} // namespace gansu
