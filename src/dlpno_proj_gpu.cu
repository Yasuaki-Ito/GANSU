/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_proj_gpu.hpp"

#include <algorithm>
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
        throw std::runtime_error(std::string("TripleProjGpu CUDA error in ")
                                 + what + ": " + cudaGetErrorString(e));
    }
}
inline void check_cublas_(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("TripleProjGpu cuBLAS error in ")
                                 + what + " status="
                                 + std::to_string(static_cast<int>(s)));
    }
}

} // anonymous namespace

TripleProjGpu::TripleProjGpu(const std::vector<PairData>&   pairs,
                              const std::vector<PairSetup>&  setups,
                              const std::vector<int>&        pair_lookup,
                              const real_t*                  S_AO_host,
                              int nao, int nocc, int max_n_tno)
    : nao_(nao), nocc_(nocc), max_n_(max_n_tno),
      n_pairs_(static_cast<int>(pairs.size()))
{
    if (nao <= 0 || nocc <= 0 || max_n_tno <= 0 || n_pairs_ <= 0
        || S_AO_host == nullptr) {
        active_ = false;
        return;
    }

    max_pno_ = 0;
    for (const auto& p : pairs) max_pno_ = std::max(max_pno_, p.n_pno);
    if (max_pno_ == 0) { active_ = false; return; }

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

    const size_t s_words      = static_cast<size_t>(nao) * nao;
    const size_t sbarQ_words  = static_cast<size_t>(n_pairs_) * nao * max_pno_;
    const size_t Y_words      = static_cast<size_t>(nocc) * nocc * max_pno_ * max_pno_;
    const size_t plookup_words = static_cast<size_t>(nocc) * nocc;
    const int max_batch = 3 * nocc + 6;
    const size_t qtno_words   = static_cast<size_t>(nao) * max_n_tno;
    const size_t R_words      = static_cast<size_t>(max_batch) * max_n_tno * max_pno_;
    const size_t T_words      = static_cast<size_t>(max_batch) * max_n_tno * max_n_tno;

    auto try_alloc = [](void** ptr, size_t bytes) -> bool {
        return cudaMalloc(ptr, bytes) == cudaSuccess;
    };

    bool ok = true;
    ok &= try_alloc(reinterpret_cast<void**>(&d_S_AO_),        sizeof(real_t) * s_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_S_bar_Q_),     sizeof(real_t) * sbarQ_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_Y_oriented_),  sizeof(real_t) * Y_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_pair_lookup_), sizeof(int) * plookup_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_n_pno_pair_),  sizeof(int) * n_pairs_);
    ok &= try_alloc(reinterpret_cast<void**>(&d_Q_tno_),       sizeof(real_t) * qtno_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_R_batch_),     sizeof(real_t) * R_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_RY_batch_),    sizeof(real_t) * R_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_T_batch_),     sizeof(real_t) * T_words);
    ok &= try_alloc(reinterpret_cast<void**>(&d_A_array_),     sizeof(const real_t*) * max_batch);
    ok &= try_alloc(reinterpret_cast<void**>(&d_B_array_),     sizeof(const real_t*) * max_batch);
    ok &= try_alloc(reinterpret_cast<void**>(&d_C_array_),     sizeof(real_t*) * max_batch);

    if (ok) {
        ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_pinned_T_),
                             sizeof(real_t) * T_words, cudaHostAllocDefault) == cudaSuccess);
        for (int s = 0; s < 3 && ok; ++s) {
            ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_A_array_[s]),
                                 sizeof(const real_t*) * max_batch,
                                 cudaHostAllocDefault) == cudaSuccess);
            ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_B_array_[s]),
                                 sizeof(const real_t*) * max_batch,
                                 cudaHostAllocDefault) == cudaSuccess);
            ok &= (cudaHostAlloc(reinterpret_cast<void**>(&h_C_array_[s]),
                                 sizeof(real_t*) * max_batch,
                                 cudaHostAllocDefault) == cudaSuccess);
        }
    }

    if (!ok) {
        if (d_S_AO_)         cudaFree(d_S_AO_);
        if (d_S_bar_Q_)      cudaFree(d_S_bar_Q_);
        if (d_Y_oriented_)   cudaFree(d_Y_oriented_);
        if (d_pair_lookup_)  cudaFree(d_pair_lookup_);
        if (d_n_pno_pair_)   cudaFree(d_n_pno_pair_);
        if (d_Q_tno_)        cudaFree(d_Q_tno_);
        if (d_R_batch_)      cudaFree(d_R_batch_);
        if (d_RY_batch_)     cudaFree(d_RY_batch_);
        if (d_T_batch_)      cudaFree(d_T_batch_);
        if (d_A_array_)      cudaFree(d_A_array_);
        if (d_B_array_)      cudaFree(d_B_array_);
        if (d_C_array_)      cudaFree(d_C_array_);
        if (h_pinned_T_)     cudaFreeHost(h_pinned_T_);
        for (int s = 0; s < 3; ++s) {
            if (h_A_array_[s]) cudaFreeHost(h_A_array_[s]);
            if (h_B_array_[s]) cudaFreeHost(h_B_array_[s]);
            if (h_C_array_[s]) cudaFreeHost(h_C_array_[s]);
            h_A_array_[s] = h_B_array_[s] = nullptr;
            h_C_array_[s] = nullptr;
        }
        cublasDestroy(cublas);
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
        d_S_AO_ = d_S_bar_Q_ = d_Y_oriented_ = d_Q_tno_
            = d_R_batch_ = d_RY_batch_ = d_T_batch_ = nullptr;
        d_pair_lookup_ = d_n_pno_pair_ = nullptr;
        d_A_array_ = d_B_array_ = nullptr;
        d_C_array_ = nullptr;
        h_pinned_T_ = nullptr;
        cublas_ = nullptr; stream_ = nullptr;
        active_ = false;
        return;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);

    // Upload S_AO and pair_lookup.
    cudaMemcpyAsync(d_S_AO_, S_AO_host, sizeof(real_t) * s_words,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pair_lookup_, pair_lookup.data(),
                    sizeof(int) * plookup_words, cudaMemcpyHostToDevice, stream);

    // Stash a host copy of pair_lookup for fast lookup in project_for_triple.
    pair_lookup_host_ = pair_lookup;

    // Pack bar_Q padded → upload → compute S · bar_Q for all pairs.
    real_t* d_bar_Q_padded = nullptr;
    cudaMalloc(&d_bar_Q_padded, sizeof(real_t) * sbarQ_words);

    {
        std::vector<real_t> host_pack(sbarQ_words, 0.0);
        std::vector<int>    host_npno(n_pairs_, 0);
        for (int idx = 0; idx < n_pairs_; ++idx) {
            const int np = pairs[idx].n_pno;
            host_npno[idx] = np;
            if (np == 0) continue;
            const real_t* src = pairs[idx].bar_Q.data();
            real_t* dst = host_pack.data() + static_cast<size_t>(idx) * nao * max_pno_;
            for (int mu = 0; mu < nao; ++mu) {
                std::memcpy(dst + static_cast<size_t>(mu) * max_pno_,
                            src + static_cast<size_t>(mu) * np,
                            sizeof(real_t) * np);
            }
        }
        cudaMemcpyAsync(d_bar_Q_padded, host_pack.data(),
                        sizeof(real_t) * sbarQ_words,
                        cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_n_pno_pair_, host_npno.data(),
                        sizeof(int) * n_pairs_,
                        cudaMemcpyHostToDevice, stream);
        check_cuda_(cudaStreamSynchronize(stream), "sync after bar_Q upload");
    }

    // Strided batched DGEMM: S · bar_Q for all n_pairs_ pairs.
    // Per pair (row-major math): C = S · bar_Q (nao × max_pno)
    // Col-major: C_cm = bar_Q_cm (max_pno × nao) · S_cm (nao × nao)
    //   op_a = N on bar_Q, op_b = N on S.
    //   m = max_pno, n_dim = nao, k = nao.
    //   A = bar_Q_padded (lda = max_pno, strideA = nao · max_pno)
    //   B = S_AO         (ldb = nao, strideB = 0)
    //   C = S_bar_Q      (ldc = max_pno, strideC = nao · max_pno)
    const real_t one = 1.0, zero = 0.0;
    check_cublas_(cublasDgemmStridedBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      max_pno_, nao, nao,
                      &one,
                      d_bar_Q_padded, max_pno_, static_cast<long long>(nao) * max_pno_,
                      d_S_AO_,        nao,      0,
                      &zero,
                      d_S_bar_Q_,     max_pno_, static_cast<long long>(nao) * max_pno_,
                      n_pairs_),
                  "DGEMM precompute S·bar_Q");

    // Build Y_oriented[p, q] for all ordered (p, q), INCLUDING diagonal
    // (p == q). The CPU project_pair_t2_oriented_to_tno path handles diagonal
    // pairs by projecting with Y_ii as-is (or its transpose, which is
    // identical because diagonal Y is (ab)↔(ba)-symmetric for closed-shell
    // amplitudes). Skipping the diagonal here leaves zeros, which would zero
    // out T_ii projections and break (i, i, *)-flavoured triples.
    {
        std::vector<real_t> host_Y(Y_words, 0.0);
        for (int p = 0; p < nocc; ++p) {
            for (int q = 0; q < nocc; ++q) {
                const int idx = pair_lookup[p * nocc + q];
                if (idx < 0) continue;
                const PairData& pair = pairs[idx];
                if (pair.n_pno == 0) continue;
                const PairSetup& setup = setups[idx];
                const int np = pair.n_pno;
                const bool need_transpose = (setup.i == q && setup.j == p);

                real_t* dst = host_Y.data()
                            + (static_cast<size_t>(p) * nocc + q) * max_pno_ * max_pno_;
                const real_t* src = pair.Y.data();
                if (!need_transpose) {
                    for (int a = 0; a < np; ++a)
                        std::memcpy(dst + static_cast<size_t>(a) * max_pno_,
                                    src + static_cast<size_t>(a) * np,
                                    sizeof(real_t) * np);
                } else {
                    for (int a = 0; a < np; ++a)
                        for (int b = 0; b < np; ++b)
                            dst[static_cast<size_t>(a) * max_pno_ + b] =
                                src[static_cast<size_t>(b) * np + a];
                }
            }
        }
        cudaMemcpyAsync(d_Y_oriented_, host_Y.data(),
                        sizeof(real_t) * Y_words,
                        cudaMemcpyHostToDevice, stream);
        check_cuda_(cudaStreamSynchronize(stream), "sync after Y upload");
    }

    cudaFree(d_bar_Q_padded);
    active_ = true;
}

TripleProjGpu::~TripleProjGpu() {
    if (cublas_)         cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
    if (stream_)         cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
    if (d_S_AO_)         cudaFree(d_S_AO_);
    if (d_S_bar_Q_)      cudaFree(d_S_bar_Q_);
    if (d_Y_oriented_)   cudaFree(d_Y_oriented_);
    if (d_pair_lookup_)  cudaFree(d_pair_lookup_);
    if (d_n_pno_pair_)   cudaFree(d_n_pno_pair_);
    if (d_Q_tno_)        cudaFree(d_Q_tno_);
    if (d_R_batch_)      cudaFree(d_R_batch_);
    if (d_RY_batch_)     cudaFree(d_RY_batch_);
    if (d_T_batch_)      cudaFree(d_T_batch_);
    if (d_A_array_)      cudaFree(d_A_array_);
    if (d_B_array_)      cudaFree(d_B_array_);
    if (d_C_array_)      cudaFree(d_C_array_);
    if (h_pinned_T_)     cudaFreeHost(h_pinned_T_);
    for (int s = 0; s < 3; ++s) {
        if (h_A_array_[s]) cudaFreeHost(h_A_array_[s]);
        if (h_B_array_[s]) cudaFreeHost(h_B_array_[s]);
        if (h_C_array_[s]) cudaFreeHost(h_C_array_[s]);
    }
}

bool TripleProjGpu::project_for_triple(
    const real_t* Q_tno_host,
    int n_tno,
    const int triple_lmos[3],
    std::vector<std::vector<real_t>>& T_il_ext_out,
    std::vector<std::vector<real_t>>& T_jl_ext_out,
    std::vector<std::vector<real_t>>& T_kl_ext_out,
    std::array<std::vector<real_t>, 9>& T_part_out,
    bool download,
    int* b_il, int* b_jl, int* b_kl, int* b_part, void* ev)
{
    if (!active_) return false;
    if (n_tno <= 0 || n_tno > max_n_) return false;
    // Device-pack: pre-clear the inverse batch map (logical → -1).
    if (!download) {
        if (b_il)   for (int l = 0; l < nocc_; ++l) b_il[l] = -1;
        if (b_jl)   for (int l = 0; l < nocc_; ++l) b_jl[l] = -1;
        if (b_kl)   for (int l = 0; l < nocc_; ++l) b_kl[l] = -1;
        if (b_part) for (int s = 0; s < 9; ++s)     b_part[s] = -1;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_);
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const int n = n_tno;
    const real_t one = 1.0, zero = 0.0;

    // Upload Q_tno.
    check_cuda_(cudaMemcpyAsync(d_Q_tno_, Q_tno_host,
                                sizeof(real_t) * nao_ * n,
                                cudaMemcpyHostToDevice, stream),
                "memcpy Q_tno");

    // Build batch slot list. Output layout:
    //   slot 0..nocc-1            : (i, l)
    //   slot nocc..2*nocc-1       : (j, l)
    //   slot 2*nocc..3*nocc-1     : (k, l)
    //   slot 3*nocc + sp*3 + sq   : T_part[sp*3+sq]  for off-diag (sp, sq)
    const int slot_il_base   = 0;
    const int slot_jl_base   = nocc_;
    const int slot_kl_base   = 2 * nocc_;
    const int slot_part_base = 3 * nocc_;
    const int max_batch = 3 * nocc_ + 9;   // 3 part slots wasted (diag), kept for indexing

    // Build batch entries that have non-empty pairs.
    std::vector<int> slot_idx;      // slot index in output layout
    std::vector<int> pair_p;        // lmo_p
    std::vector<int> pair_q;        // lmo_q
    std::vector<int> pair_idx_arr;  // index into PairData
    slot_idx.reserve(max_batch);
    pair_p.reserve(max_batch);
    pair_q.reserve(max_batch);
    pair_idx_arr.reserve(max_batch);

    auto enqueue = [&](int slot, int lmo_p, int lmo_q) {
        if (lmo_p < 0 || lmo_p >= nocc_ || lmo_q < 0 || lmo_q >= nocc_) return;
        const int idx = pair_lookup_host_[lmo_p * nocc_ + lmo_q];
        if (idx < 0) return;
        // Empty pair: handled via zero S_bar_Q / Y (still safe to compute,
        // result will be zero). We skip to save GPU work.
        // For simplicity, include anyway — the cost is small.
        slot_idx.push_back(slot);
        pair_p.push_back(lmo_p);
        pair_q.push_back(lmo_q);
        pair_idx_arr.push_back(idx);
    };

    for (int l = 0; l < nocc_; ++l) enqueue(slot_il_base + l, triple_lmos[0], l);
    for (int l = 0; l < nocc_; ++l) enqueue(slot_jl_base + l, triple_lmos[1], l);
    for (int l = 0; l < nocc_; ++l) enqueue(slot_kl_base + l, triple_lmos[2], l);
    for (int sp = 0; sp < 3; ++sp) {
        for (int sq = 0; sq < 3; ++sq) {
            if (sp == sq) continue;
            const int slot = slot_part_base + sp * 3 + sq;
            enqueue(slot, triple_lmos[sp], triple_lmos[sq]);
        }
    }

    const int batch_n = static_cast<int>(slot_idx.size());
    if (batch_n == 0) {
        if (!download) {
            if (ev) cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev), stream);
            return true;
        }
        T_il_ext_out.assign(nocc_, {});
        T_jl_ext_out.assign(nocc_, {});
        T_kl_ext_out.assign(nocc_, {});
        for (auto& s : T_part_out) s.clear();
        return true;
    }

    // ----------------------------------------------------------------
    // Step 1 (batched): R_b = Q_tno^T · (S·bar_Q)_pair
    // Per slot output memory: row-major (n × max_pno).
    // Col-major equiv: C_cm = S_bar_Q_cm · Q_tno_cm^T  (op_a=N, op_b=T)
    //   m = max_pno, n_dim = n, k = nao
    //   A = S_bar_Q (lda = max_pno)
    //   B = Q_tno   (ldb = n, transposed)
    //   C = R       (ldc = max_pno)
    // ----------------------------------------------------------------
    for (int b = 0; b < batch_n; ++b) {
        h_A_array_[0][b] = d_S_bar_Q_
                         + static_cast<size_t>(pair_idx_arr[b]) * nao_ * max_pno_;
        h_B_array_[0][b] = d_Q_tno_;
        h_C_array_[0][b] = d_R_batch_ + static_cast<size_t>(b) * n * max_pno_;
    }
    cudaMemcpyAsync(d_A_array_, h_A_array_[0], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_array_, h_B_array_[0], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_array_, h_C_array_[0], sizeof(real_t*)       * batch_n,
                    cudaMemcpyHostToDevice, stream);
    check_cublas_(cublasDgemmBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_T,
                      max_pno_, n, nao_,
                      &one,
                      d_A_array_, max_pno_,
                      d_B_array_, n,
                      &zero,
                      d_C_array_, max_pno_,
                      batch_n),
                  "DGEMM step 1 (R = Q^T · S·bar_Q)");

    // ----------------------------------------------------------------
    // Step 2 (batched): RY_b = R_b · Y_oriented_pq
    // Row-major math: RY (n × max_pno) = R (n × max_pno) · Y (max_pno × max_pno)
    // Col-major:
    //   RY_cm (max_pno × n) = Y_cm (max_pno × max_pno) · R_cm (max_pno × n)
    //   op_a = N, op_b = N.
    //   m = max_pno, n_dim = n, k = max_pno.
    //   A = Y (lda = max_pno), B = R (ldb = max_pno), C = RY (ldc = max_pno).
    // ----------------------------------------------------------------
    for (int b = 0; b < batch_n; ++b) {
        const int p = pair_p[b];
        const int q = pair_q[b];
        h_A_array_[1][b] = d_Y_oriented_
                         + (static_cast<size_t>(p) * nocc_ + q) * max_pno_ * max_pno_;
        h_B_array_[1][b] = d_R_batch_ + static_cast<size_t>(b) * n * max_pno_;
        h_C_array_[1][b] = d_RY_batch_ + static_cast<size_t>(b) * n * max_pno_;
    }
    cudaMemcpyAsync(d_A_array_, h_A_array_[1], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_array_, h_B_array_[1], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_array_, h_C_array_[1], sizeof(real_t*)       * batch_n,
                    cudaMemcpyHostToDevice, stream);
    check_cublas_(cublasDgemmBatched(
                      cublas,
                      CUBLAS_OP_N, CUBLAS_OP_N,
                      max_pno_, n, max_pno_,
                      &one,
                      d_A_array_, max_pno_,
                      d_B_array_, max_pno_,
                      &zero,
                      d_C_array_, max_pno_,
                      batch_n),
                  "DGEMM step 2 (RY = R · Y)");

    // ----------------------------------------------------------------
    // Step 3 (batched): T_b = RY_b · R_b^T
    // Row-major math: T (n × n) = RY (n × max_pno) · R^T (max_pno × n)
    // Col-major:
    //   T_cm (n × n) = R_cm^T (n × max_pno) · RY_cm (max_pno × n)
    //   op_a = T (transpose R from col-major (max_pno × n) → (n × max_pno))
    //   op_b = N.
    //   m = n, n_dim = n, k = max_pno.
    //   A = R (lda = max_pno), B = RY (ldb = max_pno), C = T (ldc = n).
    // ----------------------------------------------------------------
    for (int b = 0; b < batch_n; ++b) {
        h_A_array_[2][b] = d_R_batch_  + static_cast<size_t>(b) * n * max_pno_;
        h_B_array_[2][b] = d_RY_batch_ + static_cast<size_t>(b) * n * max_pno_;
        h_C_array_[2][b] = d_T_batch_  + static_cast<size_t>(b) * n * n;
    }
    cudaMemcpyAsync(d_A_array_, h_A_array_[2], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B_array_, h_B_array_[2], sizeof(const real_t*) * batch_n,
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C_array_, h_C_array_[2], sizeof(real_t*)       * batch_n,
                    cudaMemcpyHostToDevice, stream);
    check_cublas_(cublasDgemmBatched(
                      cublas,
                      CUBLAS_OP_T, CUBLAS_OP_N,
                      n, n, max_pno_,
                      &one,
                      d_A_array_, max_pno_,
                      d_B_array_, max_pno_,
                      &zero,
                      d_C_array_, n,
                      batch_n),
                  "DGEMM step 3 (T = R^T · RY)");

    // Device-pack path: leave T results in d_T_batch_ (slot b at b·n², row-major
    // (c,d)); expose the inverse batch map and record the event. No D2H.
    if (!download) {
        for (int b = 0; b < batch_n; ++b) {
            const int slot = slot_idx[b];
            if      (slot < slot_jl_base)   { if (b_il)   b_il[slot - slot_il_base]   = b; }
            else if (slot < slot_kl_base)   { if (b_jl)   b_jl[slot - slot_jl_base]   = b; }
            else if (slot < slot_part_base) { if (b_kl)   b_kl[slot - slot_kl_base]   = b; }
            else                            { if (b_part) b_part[slot - slot_part_base] = b; }
        }
        if (ev) cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev), stream);
        return true;
    }

    // Download T_batch to pinned host buffer (only first batch_n slots used).
    const size_t download_words = static_cast<size_t>(batch_n) * n * n;
    check_cuda_(cudaMemcpyAsync(h_pinned_T_, d_T_batch_,
                                sizeof(real_t) * download_words,
                                cudaMemcpyDeviceToHost, stream),
                "memcpy T_batch to host");
    check_cuda_(cudaStreamSynchronize(stream), "sync project_for_triple");

    // Unpack into output containers.
    T_il_ext_out.assign(nocc_, {});
    T_jl_ext_out.assign(nocc_, {});
    T_kl_ext_out.assign(nocc_, {});
    for (auto& s : T_part_out) s.clear();
    const size_t slot_words = static_cast<size_t>(n) * n;

    for (int b = 0; b < batch_n; ++b) {
        const int slot = slot_idx[b];
        const real_t* src = h_pinned_T_ + static_cast<size_t>(b) * n * n;
        std::vector<real_t>* dst = nullptr;
        if (slot < slot_jl_base)      dst = &T_il_ext_out[slot - slot_il_base];
        else if (slot < slot_kl_base) dst = &T_jl_ext_out[slot - slot_jl_base];
        else if (slot < slot_part_base) dst = &T_kl_ext_out[slot - slot_kl_base];
        else {
            const int part_slot = slot - slot_part_base;   // sp*3 + sq
            dst = &T_part_out[part_slot];
        }
        dst->resize(slot_words);
        std::memcpy(dst->data(), src, sizeof(real_t) * slot_words);
    }
    return true;
}

#else // GANSU_CPU_ONLY

TripleProjGpu::TripleProjGpu(const std::vector<PairData>&,
                              const std::vector<PairSetup>&,
                              const std::vector<int>&,
                              const real_t*,
                              int, int, int) : active_(false) {}
TripleProjGpu::~TripleProjGpu() = default;
bool TripleProjGpu::project_for_triple(
    const real_t*, int, const int*,
    std::vector<std::vector<real_t>>&,
    std::vector<std::vector<real_t>>&,
    std::vector<std::vector<real_t>>&,
    std::array<std::vector<real_t>, 9>&,
    bool, int*, int*, int*, int*, void*) { return false; }

#endif

} // namespace gansu
