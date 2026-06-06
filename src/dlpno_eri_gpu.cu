/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_eri_gpu.hpp"

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
        throw std::runtime_error(std::string("EriBuildGpu CUDA error in ")
                                 + what + ": " + cudaGetErrorString(e));
    }
}

inline void check_cublas_(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("EriBuildGpu cuBLAS error in ")
                                 + what + " status="
                                 + std::to_string(static_cast<int>(s)));
    }
}

/// output[Q, mu, nu] ← input[mu, nu, Q]   (row-major both sides)
__global__ void transpose_Bao_munuQ_to_Qmunu(
    real_t*       __restrict__ output,
    const real_t* __restrict__ input,
    int nao, int naux)
{
    const int Q  = blockIdx.x;
    const int mu = blockIdx.y;
    const int nu = threadIdx.x;
    if (Q >= naux || mu >= nao || nu >= nao) return;
    output[(static_cast<size_t>(Q) * nao + mu) * nao + nu] =
        input[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
}

/// output[a, b, Q] ← input[Q, a, b]   (row-major both sides)
__global__ void transpose_Qab_to_abQ(
    real_t*       __restrict__ output,
    const real_t* __restrict__ input,
    int n, int naux)
{
    const int Q = blockIdx.x;
    const int a = blockIdx.y;
    const int b = threadIdx.x;
    if (Q >= naux || a >= n || b >= n) return;
    output[(static_cast<size_t>(a) * n + b) * naux + Q] =
        input[(static_cast<size_t>(Q) * n + a) * n + b];
}

} // anonymous namespace

EriBuildGpu::EriBuildGpu(const real_t* B_ao_ao_host,
                         const real_t* B_lmo_ao_host,
                         const real_t* B_lmo_lmo_host,
                         int nao, int nocc, int naux, int max_n_tno)
    : nao_(nao), nocc_(nocc), naux_(naux), max_n_(max_n_tno)
{
    if (nao <= 0 || nocc <= 0 || naux <= 0 || max_n_tno <= 0
        || B_ao_ao_host == nullptr
        || B_lmo_ao_host == nullptr
        || B_lmo_lmo_host == nullptr) {
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

    const size_t bao_words   = static_cast<size_t>(naux) * nao * nao;
    const size_t blmo_ao_words  = static_cast<size_t>(nocc) * nao * naux;
    const size_t blmo_lmo_words = static_cast<size_t>(nocc) * nocc * naux;
    const size_t mqt_words   = static_cast<size_t>(naux) * nao * max_n_tno;
    const size_t bttq_words  = static_cast<size_t>(naux) * max_n_tno * max_n_tno;
    const size_t qtno_words  = static_cast<size_t>(nao) * max_n_tno;
    const size_t blTQ_words  = static_cast<size_t>(nocc) * max_n_tno * naux;
    const size_t kiadc_words = static_cast<size_t>(3) * max_n_tno * max_n_tno * max_n_tno;
    const size_t m_words     = static_cast<size_t>(9) * nocc * max_n_tno;

    auto try_alloc = [](real_t** ptr, size_t bytes) -> bool {
        return cudaMalloc(ptr, bytes) == cudaSuccess;
    };

    bool ok = true;
    ok &= try_alloc(&d_B_ao_ao_Qmunu_, sizeof(real_t) * bao_words);
    ok &= try_alloc(&d_B_lmo_ao_,      sizeof(real_t) * blmo_ao_words);
    ok &= try_alloc(&d_B_lmo_lmo_,     sizeof(real_t) * blmo_lmo_words);
    ok &= try_alloc(&d_Q_tno_,         sizeof(real_t) * qtno_words);
    ok &= try_alloc(&d_MQ_T_,          sizeof(real_t) * mqt_words);
    ok &= try_alloc(&d_B_TTQ_,         sizeof(real_t) * bttq_words);
    ok &= try_alloc(&d_B_TTQ_abQ_,     sizeof(real_t) * bttq_words);
    ok &= try_alloc(&d_B_lTQ_,         sizeof(real_t) * blTQ_words);
    ok &= try_alloc(&d_K_iadc_,        sizeof(real_t) * kiadc_words);
    ok &= try_alloc(&d_M_,             sizeof(real_t) * m_words);

    if (ok) {
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_K_),
                             sizeof(real_t) * kiadc_words,
                             cudaHostAllocDefault) == cudaSuccess);
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_M_),
                             sizeof(real_t) * m_words,
                             cudaHostAllocDefault) == cudaSuccess);
    }

    if (!ok) {
        if (d_B_ao_ao_Qmunu_) cudaFree(d_B_ao_ao_Qmunu_);
        if (d_B_lmo_ao_)      cudaFree(d_B_lmo_ao_);
        if (d_B_lmo_lmo_)     cudaFree(d_B_lmo_lmo_);
        if (d_Q_tno_)         cudaFree(d_Q_tno_);
        if (d_MQ_T_)          cudaFree(d_MQ_T_);
        if (d_B_TTQ_)         cudaFree(d_B_TTQ_);
        if (d_B_TTQ_abQ_)     cudaFree(d_B_TTQ_abQ_);
        if (d_B_lTQ_)         cudaFree(d_B_lTQ_);
        if (d_K_iadc_)        cudaFree(d_K_iadc_);
        if (d_M_)             cudaFree(d_M_);
        if (h_pinned_K_)      cudaFreeHost(h_pinned_K_);
        if (h_pinned_M_)      cudaFreeHost(h_pinned_M_);
        cublasDestroy(cublas);
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        d_B_ao_ao_Qmunu_ = d_B_lmo_ao_ = d_B_lmo_lmo_ = d_Q_tno_ = d_MQ_T_
            = d_B_TTQ_ = d_B_TTQ_abQ_ = d_B_lTQ_ = d_K_iadc_ = d_M_ = nullptr;
        h_pinned_K_ = h_pinned_M_ = nullptr;
        cublas_ = nullptr;
        stream_ = nullptr;
        active_ = false;
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    // Upload B_lmo_ao and B_lmo_lmo as-is (native (l, ν, Q) / (l, m, Q) layout).
    cudaMemcpyAsync(d_B_lmo_ao_,  B_lmo_ao_host,
                    sizeof(real_t) * blmo_ao_words,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_lmo_lmo_, B_lmo_lmo_host,
                    sizeof(real_t) * blmo_lmo_words,
                    cudaMemcpyHostToDevice, stream);

    // Upload B_ao_ao via a scratch buffer in (μ, ν, Q) layout, then transpose
    // to Q-major (Q, μ, ν) for cuBLAS strided-batched DGEMM.
    real_t* d_B_ao_munuQ = nullptr;
    if (cudaMalloc(&d_B_ao_munuQ, sizeof(real_t) * bao_words) != cudaSuccess) {
        active_ = false;
        return;
    }
    cudaMemcpyAsync(d_B_ao_munuQ, B_ao_ao_host,
                    sizeof(real_t) * bao_words,
                    cudaMemcpyHostToDevice, stream);
    {
        dim3 grid(naux, nao);
        dim3 block(nao);
        transpose_Bao_munuQ_to_Qmunu<<<grid, block, 0, stream>>>(
            d_B_ao_ao_Qmunu_, d_B_ao_munuQ, nao, naux);
    }
    cudaStreamSynchronize(stream);
    cudaFree(d_B_ao_munuQ);

    active_ = true;
}

EriBuildGpu::~EriBuildGpu() {
    if (cublas_)            cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
    if (stream_)            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    if (d_B_ao_ao_Qmunu_)   cudaFree(d_B_ao_ao_Qmunu_);
    if (d_B_lmo_ao_)        cudaFree(d_B_lmo_ao_);
    if (d_B_lmo_lmo_)       cudaFree(d_B_lmo_lmo_);
    if (d_Q_tno_)           cudaFree(d_Q_tno_);
    if (d_MQ_T_)            cudaFree(d_MQ_T_);
    if (d_B_TTQ_)           cudaFree(d_B_TTQ_);
    if (d_B_TTQ_abQ_)       cudaFree(d_B_TTQ_abQ_);
    if (d_B_lTQ_)           cudaFree(d_B_lTQ_);
    if (d_K_iadc_)          cudaFree(d_K_iadc_);
    if (d_M_)               cudaFree(d_M_);
    if (h_pinned_K_)        cudaFreeHost(h_pinned_K_);
    if (h_pinned_M_)        cudaFreeHost(h_pinned_M_);
}

bool EriBuildGpu::build_eri_and_m(const real_t* Q_tno_host,
                                   int n_tno,
                                   const int triple_lmos[3],
                                   std::vector<real_t>& K_iadc_out,
                                   std::array<std::vector<real_t>, 9>& M_out,
                                   bool download)
{
    if (!active_) return false;
    if (n_tno <= 0 || n_tno > max_n_) return false;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const int n = n_tno;
    const real_t one = 1.0, zero = 0.0;

    // Upload Q_tno (nao × n).
    check_cuda_(cudaMemcpyAsync(d_Q_tno_, Q_tno_host,
                                sizeof(real_t) * nao_ * n,
                                cudaMemcpyHostToDevice, stream),
                "memcpy Q_tno");

    // -----------------------------------------------------------------
    // Step A: per-Q  MQ_T[Q, μ, a] = Σ_ν B_ao[Q, μ, ν] · Q_tno[ν, a]
    //   col-major view: MQ_T_cm = Q_tno_cm · B_ao_cm  (per Q)
    // -----------------------------------------------------------------
    check_cublas_(cublasDgemmStridedBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      n, nao_, nao_,
                      &one,
                      d_Q_tno_,           n,    0,
                      d_B_ao_ao_Qmunu_,   nao_, static_cast<long long>(nao_) * nao_,
                      &zero,
                      d_MQ_T_,            n,    static_cast<long long>(nao_) * n,
                      naux_),
                  "DGEMM step A");

    // -----------------------------------------------------------------
    // Step B: per-Q  TT[Q, a, b] = Σ_μ Q_tno[μ, a] · MQ_T[Q, μ, b]
    //   row-major math: TT = Q_tno^T · MQ_T per Q
    //   col-major: C = MQ_T_cm · Q_tno_cm^T per Q  (op_a=N, op_b=T)
    // -----------------------------------------------------------------
    check_cublas_(cublasDgemmStridedBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_T,
                      n, n, nao_,
                      &one,
                      d_MQ_T_,   n, static_cast<long long>(nao_) * n,
                      d_Q_tno_,  n, 0,
                      &zero,
                      d_B_TTQ_,  n, static_cast<long long>(n) * n,
                      naux_),
                  "DGEMM step B");

    // Transpose (Q, a, b) → (a, b, Q) so K builds via DGEMM over Q (contiguous).
    {
        dim3 grid(naux_, n);
        dim3 block(n);
        transpose_Qab_to_abQ<<<grid, block, 0, stream>>>(
            d_B_TTQ_abQ_, d_B_TTQ_, n, naux_);
    }

    // -----------------------------------------------------------------
    // Step C: build B_lTQ_full[l, a, Q] = Σ_ν Q_tno[ν, a] · B_lmo_ao[l, ν, Q]
    //   per l: B_lTQ_l = Q_tno^T · B_lmo_ao_l   (n × naux)
    //   col-major (per l): C = B_lmo_ao_cm · Q_tno_cm^T → (naux × n) col-major
    //                       = (n × naux) row-major  ✓
    //   Use cublasDgemmStridedBatched batched over l.
    //   For each l: m=naux, n=n, k=nao;
    //     A = B_lmo_ao  (col-major naux × nao), lda=naux, strideA=nao*naux
    //     B = Q_tno     (col-major n × nao),    ldb=n,    strideB=0
    //     C = B_lTQ_l   (col-major naux × n),   ldc=naux, strideC=n*naux
    //   op_a = N, op_b = T (transpose Q_tno's col-major (n × nao) to (nao × n))
    //
    // The col-major view of B_lmo_ao[l, ν, Q] (row-major) is (Q × ν) of shape
    // (naux × nao)? Let me re-check: B_lmo_ao row-major has B[l, ν, Q] at
    // offset (l*nao + ν)*naux + Q. Per-l block is (nao × naux) row-major,
    // i.e., col-major (naux × nao). For col-major, lda = naux. ✓
    // -----------------------------------------------------------------
    check_cublas_(cublasDgemmStridedBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_T,
                      naux_, n, nao_,
                      &one,
                      d_B_lmo_ao_, naux_, static_cast<long long>(nao_) * naux_,
                      d_Q_tno_,    n,     0,
                      &zero,
                      d_B_lTQ_,    naux_, static_cast<long long>(n) * naux_,
                      nocc_),
                  "DGEMM step C (B_lTQ full)");

    // After Step C, d_B_lTQ_ holds (per l) col-major (naux × n) =
    // row-major (n × naux). Total layout: l-major, then per-l (n × naux)
    // row-major: B_lTQ_full[l * n * naux + a * naux + Q].

    // -----------------------------------------------------------------
    // Step D: K_iadc[i_loc, a, d, c] = Σ_Q B_lTQ[l_lmo, a, Q] · B_TTQ_abQ[d, c, Q]
    //   For each i_loc ∈ {0, 1, 2}, l = triple_lmos[i_loc]:
    //     K_iadc[i_loc, a, dc] = B_lTQ[l, a, :] · B_TTQ_abQ_flat[dc, :]^T
    //
    //   Row-major: C (n × n²) = A (n × naux) · B^T (naux × n²)
    //   Col-major: C^T (n² × n) = B (n² × naux) · A^T (naux × n)
    //   cublasDgemm: op_a=N, op_b=N, m=n², n=n, k=naux;
    //     A = B_TTQ_abQ (col-major n² × naux), lda=n², strideA=0 (constant)
    //     B = B_lTQ_l   (col-major naux × n),  ldb=naux
    //     C = K_iadc[i_loc] (col-major n² × n), ldc=n², strideC=...
    //
    // Easier: launch 3 separate DGEMMs (one per i_loc).
    // -----------------------------------------------------------------
    for (int i_loc = 0; i_loc < 3; ++i_loc) {
        const int l = triple_lmos[i_loc];
        const real_t* B_lTQ_l = d_B_lTQ_ + static_cast<size_t>(l) * n * naux_;
        real_t* K_loc = d_K_iadc_ + static_cast<size_t>(i_loc) * n * n * n;
        // K_rm[a, dc] = Σ_Q B_lTQ_l_rm[a, Q] · B_TTQ_flat_rm[dc, Q]
        // d_B_TTQ_abQ_ holds row-major (n² × naux); col-major view (naux × n²).
        // d_B_lTQ_ holds row-major (n × naux);       col-major view (naux × n).
        // K_cm (n² × n) = B_TTQ^T (n² × naux) · B_lTQ_l (naux × n)
        //   → op_a=T on B_TTQ (col-major naux × n², lda=naux),
        //     op_b=N on B_lTQ_l (col-major naux × n,  ldb=naux),
        //     C = K_loc, ldc=n².
        // Output K_loc memory = col-major (n² × n) ≡ row-major (n × n²) ≡
        //   row-major (a, d, c) with index a*n² + d*n + c.
        check_cublas_(cublasDgemm(cublas,
                                  CUBLAS_OP_T, CUBLAS_OP_N,
                                  n * n, n, naux_,
                                  &one,
                                  d_B_TTQ_abQ_, naux_,
                                  B_lTQ_l,      naux_,
                                  &zero,
                                  K_loc,        n * n),
                      "DGEMM step D (K_iadc)");
    }

    // -----------------------------------------------------------------
    // Step E: M[slot, l, a] = Σ_Q B_lTQ[lmo_sp, a, Q] · B_lmo_lmo[l, lmo_sq, Q]
    //   for slot = sp*3 + sq, (sp, sq) ∈ {(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)}.
    //
    // Layouts (row-major host convention, col-major cuBLAS view in parens):
    //   B_lTQ_p_rm  (n × naux)    at d_B_lTQ_ + lmo_p · n · naux
    //                              (col-major (naux × n), lda = naux)
    //   B_lmo_lmo_q "slice"        the lmo_q column of the (l, m, Q) layout.
    //   B_lmo_lmo_rm[l, m, Q] is at base offset (l · nocc + m) · naux + Q.
    //   For fixed m = lmo_q, the slice is laid out:
    //     row 0 (l=0) starts at lmo_q · naux
    //     row 1 (l=1) starts at lmo_q · naux + nocc · naux
    //     etc.  → stride between rows = nocc · naux.
    //   Pointer-offset to lmo_q's row 0 = d_B_lmo_lmo_ + lmo_q · naux.
    //   Then column-major view (naux × nocc) with ldb = nocc · naux gives
    //   cm[Q, l] at offset (lmo_q · naux) + Q + l · (nocc · naux) — exactly
    //   the row-major B_lmo_lmo_rm[l, lmo_q, Q]. ✓
    //
    // Row-major math:  M_rm[l, a] = Σ_Q B_sq_rm[l, Q] · B_lTQ_p_rm[a, Q]
    //   → M_cm[a, l]   = (B_lTQ_p_cm^T · B_sq_cm) [a, l]    (op_a=T, op_b=N)
    //   m = n, n_dim = nocc, k = naux.
    // -----------------------------------------------------------------
    for (int sp = 0; sp < 3; ++sp) {
        for (int sq = 0; sq < 3; ++sq) {
            if (sp == sq) continue;
            const int lmo_p = triple_lmos[sp];
            const int lmo_q = triple_lmos[sq];
            const real_t* B_lTQ_p   = d_B_lTQ_ + static_cast<size_t>(lmo_p) * n * naux_;
            const real_t* B_q_slice = d_B_lmo_lmo_ + static_cast<size_t>(lmo_q) * naux_;
            real_t* M_slot = d_M_ + static_cast<size_t>(sp * 3 + sq) * nocc_ * n;

            check_cublas_(cublasDgemm(cublas,
                                      CUBLAS_OP_T, CUBLAS_OP_N,
                                      n, nocc_, naux_,
                                      &one,
                                      B_lTQ_p,   naux_,
                                      B_q_slice, nocc_ * naux_,   // non-standard ldb
                                      &zero,
                                      M_slot,    n),
                          "DGEMM step E (M tensor)");
        }
    }

    // Device-pack path: leave K/M on device (d_K_iadc_/d_M_), skip the D2H.
    // The DGEMMs above are queued async on `stream`; the caller records an
    // event on `stream` (build_eri_and_m_device) for cross-stream ordering.
    if (!download) return true;

    // -----------------------------------------------------------------
    // Download K_iadc and M tensors to host pinned buffers.
    // -----------------------------------------------------------------
    const size_t kiadc_words = static_cast<size_t>(3) * n * n * n;
    const size_t m_words     = static_cast<size_t>(9) * nocc_ * n;
    check_cuda_(cudaMemcpyAsync(h_pinned_K_, d_K_iadc_,
                                sizeof(real_t) * kiadc_words,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy K_iadc to host");
    check_cuda_(cudaMemcpyAsync(h_pinned_M_, d_M_,
                                sizeof(real_t) * m_words,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy M to host");
    check_cuda_(cudaStreamSynchronize(stream), "sync after build_eri_and_m");

    K_iadc_out.resize(kiadc_words);
    std::memcpy(K_iadc_out.data(), h_pinned_K_,
                sizeof(real_t) * kiadc_words);

    // Unpack 9-slot M into M_out (only 6 off-diagonal slots populated).
    for (int slot = 0; slot < 9; ++slot) {
        const int sp = slot / 3;
        const int sq = slot % 3;
        if (sp == sq) { M_out[slot].clear(); continue; }
        const size_t slot_words = static_cast<size_t>(nocc_) * n;
        M_out[slot].resize(slot_words);
        std::memcpy(M_out[slot].data(),
                    h_pinned_M_ + static_cast<size_t>(slot) * nocc_ * n,
                    sizeof(real_t) * slot_words);
    }
    return true;
}

bool EriBuildGpu::build_eri_and_m_device(const real_t* Q_tno_host,
                                          int n_tno,
                                          const int triple_lmos[3],
                                          void* ev)
{
    std::vector<real_t> dummy_K;
    std::array<std::vector<real_t>, 9> dummy_M;
    if (!build_eri_and_m(Q_tno_host, n_tno, triple_lmos, dummy_K, dummy_M,
                         /*download=*/false))
        return false;
    cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev),
                    reinterpret_cast<cudaStream_t>(stream_));
    return true;
}

#else // GANSU_CPU_ONLY: stub-out

EriBuildGpu::EriBuildGpu(const real_t*, const real_t*, const real_t*,
                         int, int, int, int) : active_(false) {}
EriBuildGpu::~EriBuildGpu() = default;
bool EriBuildGpu::build_eri_and_m(
    const real_t*, int, const int*,
    std::vector<real_t>&,
    std::array<std::vector<real_t>, 9>&, bool) { return false; }

bool EriBuildGpu::build_eri_and_m_device(
    const real_t*, int, const int*, void*) { return false; }

#endif

} // namespace gansu
