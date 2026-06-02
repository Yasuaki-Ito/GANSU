/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file dlpno_eom_dressed_pno.cu
 * @brief Per-pair PNO-basis intermediates for the native DLPNO-EOM σ operators.
 *        B-a.6c (a): Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij). See the .hpp for the contract.
 */

#include "dlpno_eom_dressed_pno.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
// OpenBLAS allocates per-caller-thread state and was compiled with a hard
// cap of 128 threads.  When this file's OMP parallel-fors (IP ring builder,
// EA Wvvvv ring builder) launch from a machine with >128 cores (H200 NVL,
// EPYC Zen 4 192-core), Eigen GEMM inside each OMP worker reaches OpenBLAS
// which trips the per-thread metadata array and aborts the process.
// Cap each outer parallel-for to 64 threads — same heuristic CLAUDE.md
// documents for `--dlpno_cpu_threads` (DLPNO ground-state OMP cap).
// Mirrored here because the EOM ring builders are a separate code path
// outside dlpno_ccsd.cu's cap.
inline int dlpno_eom_ring_omp_cap() {
#ifdef _OPENMP
    const int hw = omp_get_max_threads();
    return hw > 64 ? 64 : hw;
#else
    return 1;
#endif
}
}  // anonymous namespace

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#endif

#include <Eigen/Dense>

#include "dlpno_mp2.hpp"        // DLPNOLMP2Result (setups / pairs / pair_lookup / phase24)
#include "dlpno_pair_data.hpp" // PairSetup / PairData / Phase24Integrals
#include "device_host_memory.hpp" // tracked_cudaMalloc / tracked_cudaFree (matches IP/EA op pattern)

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#ifndef GANSU_CPU_ONLY
// B-a.6f Stage F2b: single-kernel V·T contraction for one (idx, I) of the IP
// ring builder. 1 thread per (m, ap, dp) output element; the kernel sums over
// (l, e) internally and writes the 4 terms (A,B,C,D) into nat_vvo / nat_vov.
// Bypasses cuBLAS launch overhead — the per-(idx, I) work is < n³·nocc² flops,
// far below the 10 µs cuBLAS launch budget per call. At 100 atoms 1 kernel
// launch replaces ~4·nocc² = 90k host-Eigen GEMMs.
//
// Layout (matches host F2a):
//   d_V[m,l,dp,e] = V[((m·nocc + l)·n + dp)·n + e]
//   d_T[l,ap,e]   = T_l[ap, e]   (rows = ap, cols = e per l-slab)
//   nat_vvo[m,ap,dp], nat_vov[m,ap,dp] indexed ((m·n + ap)·n + dp)
__global__ void ip_ring_vt_contract_kernel(
    const real_t* __restrict__ d_V,        // [nocc · nocc · n · n]
    const real_t* __restrict__ d_T,        // [nocc · n · n]
    real_t* __restrict__ d_nat_vvo,        // [nocc · n · n]  (output)
    real_t* __restrict__ d_nat_vov,        // [nocc · n · n]  (output)
    int nocc, int n)
{
    const int m  = blockIdx.x;
    const int ap = blockIdx.y;
    const int dp = threadIdx.x;
    if (dp >= n) return;
    const size_t nn = static_cast<size_t>(n) * n;
    real_t s_vvo = 0.0;
    real_t s_vov = 0.0;
    for (int l = 0; l < nocc; ++l) {
        const real_t* Tl  = d_T + static_cast<size_t>(l) * nn;
        const real_t* Mml = d_V + (static_cast<size_t>(m) * nocc + l) * nn;
        for (int e = 0; e < n; ++e) {
            const real_t V1   = Mml[static_cast<size_t>(dp) * n + e];   // M[dp, e]
            const real_t V3   = Mml[static_cast<size_t>(e)  * n + dp];  // M[e, dp]
            const real_t Tape = Tl [static_cast<size_t>(ap) * n + e];   // T[ap, e]
            const real_t Teap = Tl [static_cast<size_t>(e)  * n + ap];  // T[e, ap]
            s_vvo += 2.0 * V1 * Tape - V1 * Teap - V3 * Tape;
            s_vov += -V3 * Teap;
        }
    }
    const size_t off = (static_cast<size_t>(m) * n + ap) * n + dp;
    d_nat_vvo[off] = s_vvo;
    d_nat_vov[off] = s_vov;
}

// Per-(idx, I) wrapper around the V·T kernel. Allocates scratch buffers,
// uploads V[idx] and the t2Il stack, launches the kernel, and copies the
// nat_vvo / nat_vov results back to host. Returns false on any CUDA error
// (the caller can fall back to the host F2a path).
//
// Note: this allocates and frees device buffers per (idx, I). At a thousand
// pairs that adds 1000+ malloc/free pairs; if profiling shows this overhead
// is significant we will hoist the allocation to a per-thread scratch pool.
bool ip_ring_vt_contract_gpu(const std::vector<real_t>& V_host,         // [nocc² · n²]
                             const std::vector<RowMatXd>& t2Il,         // size nocc, each [n × n]
                             int nocc, int n,
                             std::vector<real_t>& nat_vvo_host,         // [nocc · n²] out
                             std::vector<real_t>& nat_vov_host)         // [nocc · n²] out
{
    const size_t v_sz = static_cast<size_t>(nocc) * nocc * n * n;
    const size_t t_sz = static_cast<size_t>(nocc) * n * n;
    const size_t out_sz = static_cast<size_t>(nocc) * n * n;
    nat_vvo_host.assign(out_sz, 0.0);
    nat_vov_host.assign(out_sz, 0.0);

    real_t* d_V = nullptr;
    real_t* d_T = nullptr;
    real_t* d_nat_vvo = nullptr;
    real_t* d_nat_vov = nullptr;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc(&d_V, v_sz * sizeof(real_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_T, t_sz * sizeof(real_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_nat_vvo, out_sz * sizeof(real_t));
    if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc(&d_nat_vov, out_sz * sizeof(real_t));
    if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_V, V_host.data(), v_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    {
        // Stage t2Il (vector of Eigen matrices) into a contiguous host buffer
        // before H2D — a single async memcpy per t2Il slab would also work but
        // adds complexity for marginal benefit at this scale.
        std::vector<real_t> T_packed(t_sz, 0.0);
        for (int l = 0; l < nocc; ++l) {
            const size_t off = static_cast<size_t>(l) * n * n;
            // t2Il[l] is RowMatXd row-major n × n.
            std::memcpy(T_packed.data() + off, t2Il[l].data(),
                        static_cast<size_t>(n) * n * sizeof(real_t));
        }
        err = cudaMemcpy(d_T, T_packed.data(), t_sz * sizeof(real_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup;
    }

    {
        dim3 grid(static_cast<unsigned>(nocc), static_cast<unsigned>(n), 1);
        dim3 block(static_cast<unsigned>(n), 1, 1);
        ip_ring_vt_contract_kernel<<<grid, block>>>(d_V, d_T, d_nat_vvo, d_nat_vov, nocc, n);
        err = cudaGetLastError();
        if (err != cudaSuccess) goto cleanup;
    }

    err = cudaMemcpy(nat_vvo_host.data(), d_nat_vvo,
                     out_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(nat_vov_host.data(), d_nat_vov,
                     out_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

cleanup:
    if (d_V)        cudaFree(d_V);
    if (d_T)        cudaFree(d_T);
    if (d_nat_vvo)  cudaFree(d_nat_vvo);
    if (d_nat_vov)  cudaFree(d_nat_vov);
    return err == cudaSuccess;
}
#endif  // !GANSU_CPU_ONLY
}  // namespace

DressedPnoIP build_dressed_pno_ip(const std::vector<real_t>& h_Lvv,
                                  const std::vector<std::vector<real_t>>& Uall,
                                  const std::vector<int>& n_pno,
                                  int nvir)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    DressedPnoIP out;
    out.Lvv_pno.resize(n_pairs);

    Eigen::Map<const RowMatXd> Lvv(h_Lvv.data(), nvir, nvir);   // Lvv[a,d]

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        // U^(ij): [nvir × n_pno], flat row-major.
        Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
        // Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij)   [n × n].
        const RowMatXd Lpno = U.transpose() * Lvv * U;
        out.Lvv_pno[idx].assign(static_cast<size_t>(n) * n, 0.0);
        for (int a = 0; a < n; ++a)
            for (int d = 0; d < n; ++d)
                out.Lvv_pno[idx][static_cast<size_t>(a) * n + d] = Lpno(a, d);
    }
    return out;
}

void build_dressed_pno_ip_phladder(DressedPnoIP& io,
                                   const std::vector<real_t>& h_Wovvo_lmo,
                                   const std::vector<real_t>& h_Wovov_lmo,
                                   const std::vector<std::vector<real_t>>& Uall,
                                   const std::vector<int>& pair_i,
                                   const std::vector<int>& pair_j,
                                   const std::vector<int>& n_pno,
                                   int nocc, int nvir)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    io.Wovvo_pno_occi.assign(n_pairs, {});
    io.Wovvo_pno_occj.assign(n_pairs, {});
    io.Wovov_pno_occi.assign(n_pairs, {});
    io.Wovov_pno_occj.assign(n_pairs, {});

    const size_t nv = static_cast<size_t>(nvir);

    // For one fixed occupied I and one LMO m, gather the dense [nvir × nvir]
    // virtual block, congruence it U^(ij)ᵀ M U^(ij) → [n × n], scatter to
    // out[(m·n + a')·n + d']. `ovvo` selects the layout (Wovvo vs Wovov).
    auto build_one = [&](std::vector<std::vector<real_t>>& dst, int idx, int n,
                         const Eigen::Map<const RowMatXd>& U, int I, bool ovvo) {
        dst[idx].assign(static_cast<size_t>(nocc) * n * n, 0.0);
        RowMatXd M(nvir, nvir);
        for (int m = 0; m < nocc; ++m) {
            for (int a = 0; a < nvir; ++a)
                for (int d = 0; d < nvir; ++d) {
                    // Wovvo layout ((m·nvir+a)·nvir+d)·nocc + I
                    // Wovov layout ((m·nvir+a)·nocc+I)·nvir + d
                    const size_t off = ovvo
                        ? (((static_cast<size_t>(m) * nv + a) * nv + d) * nocc + I)
                        : (((static_cast<size_t>(m) * nv + a) * nocc + I) * nv + d);
                    M(a, d) = (ovvo ? h_Wovvo_lmo : h_Wovov_lmo)[off];
                }
            const RowMatXd C = U.transpose() * M * U;        // [n × n]
            const size_t base = static_cast<size_t>(m) * n * n;
            for (int ap = 0; ap < n; ++ap)
                for (int dp = 0; dp < n; ++dp)
                    dst[idx][base + static_cast<size_t>(ap) * n + dp] = C(ap, dp);
        }
    };

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
        const int i = pair_i[idx];
        const int j = pair_j[idx];
        build_one(io.Wovvo_pno_occi, idx, n, U, i, /*ovvo=*/true);
        build_one(io.Wovvo_pno_occj, idx, n, U, j, /*ovvo=*/true);
        build_one(io.Wovov_pno_occi, idx, n, U, i, /*ovvo=*/false);
        build_one(io.Wovov_pno_occj, idx, n, U, j, /*ovvo=*/false);
    }
}

// ---------------------------------------------------------------------------
//  B-a.6c IP dense-free bare seed (Phase24 W_ovvo_bare / W_oovv_bare → the per-
//  pair PNO Wovvo_pno/Wovov_pno bare, NO dense nocc²·nvir²). EA B-EA.6e analog.
// ---------------------------------------------------------------------------
void build_dressed_pno_ip_bare(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<int>& n_pno,
                               int nocc)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    const Phase24Integrals& ph = res.phase24;
    io.Wovvo_pno_occi.assign(n_pairs, {});
    io.Wovvo_pno_occj.assign(n_pairs, {});
    io.Wovov_pno_occi.assign(n_pairs, {});
    io.Wovov_pno_occj.assign(n_pairs, {});

    // Copy one Phase24 bare block (layout (m·n+a')·n+d', size nocc·n²) into the
    // matching native Wovvo_pno/Wovov_pno slot, or zero it if Phase24 lacks it.
    auto seed = [&](std::vector<std::vector<real_t>>& dst, int idx, size_t need,
                    const std::vector<std::vector<real_t>>& src) {
        if (idx < static_cast<int>(src.size()) && src[idx].size() == need)
            dst[idx] = src[idx];                 // direct copy (same layout)
        else
            dst[idx].assign(need, 0.0);          // missing block → bare = 0 (ring still added)
    };

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        const size_t need = static_cast<size_t>(nocc) * n * n;
        seed(io.Wovvo_pno_occi, idx, need, ph.W_ovvo_bare_i);
        seed(io.Wovvo_pno_occj, idx, need, ph.W_ovvo_bare_j);
        seed(io.Wovov_pno_occi, idx, need, ph.W_oovv_bare_i);
        seed(io.Wovov_pno_occj, idx, need, ph.W_oovv_bare_j);
    }
}

// ---------------------------------------------------------------------------
//  B-a.6c(b2): native ring build (Phase24 + two-sided barS) − dense-ring
//  congruence subtraction. See the .hpp for the contract and index derivation.
// ---------------------------------------------------------------------------
void build_dressed_pno_ip_ring(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<real_t>& h_ovov_lmo,
                               const std::vector<real_t>& h_t2_lmo,
                               const std::vector<std::vector<real_t>>& Uall,
                               const std::vector<real_t>& h_S,
                               const std::vector<int>& n_pno,
                               int nao, int nocc, int nvir,
                               real_t& max_delta, real_t& max_ring,
                               bool subtract_dense)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    const size_t NV = static_cast<size_t>(nvir);
    const size_t NO = static_cast<size_t>(nocc);
    const Phase24Integrals& ph = res.phase24;
    max_delta = 0.0;
    max_ring  = 0.0;

    // ovov(p,a,q,b) = (pa|qb), layout ((p·nvir+a)·nocc+q)·nvir+b.
    auto OVOV = [&](int p, int a, int q, int b) -> real_t {
        return h_ovov_lmo[(((static_cast<size_t>(p) * NV + a) * NO + q) * NV + b)];
    };
    // t2(p,q,a,b), layout ((p·nocc+q)·nvir+a)·nvir+b.
    auto T2 = [&](int p, int q, int a, int b) -> real_t {
        return h_t2_lmo[(((static_cast<size_t>(p) * NO + q) * NV + a) * NV + b)];
    };

    // --- Dense ring per fixed occupied I (canonical virtual basis), occ→LMO. ---
    // DRvvo[I][m,a,c] = Σ_{l,e}[ 2 ovov(m,c,l,e) t2(I,l,a,e)
    //                          -   ovov(m,c,l,e) t2(l,I,a,e)
    //                          -   ovov(m,e,l,c) t2(I,l,a,e) ]   (Wovvo ring; a=res, c=contr)
    // DRvov[I][m,b,e] = -Σ_{c,l} ovov(m,c,l,e) t2(I,l,c,b)        (Wovov ring; b=res, e=contr)
    // Only for the B-a.6c(b2) validate gate (subtract_dense); the IP dense-free
    // path (subtract_dense=false) NEVER materialises these nocc²·nvir² buffers.
    const size_t dr_per_I = NO * NV * NV;                     // [m,a,c]
    std::vector<std::vector<real_t>> DRvvo(subtract_dense ? nocc : 0),
                                     DRvov(subtract_dense ? nocc : 0);
    if (subtract_dense)
    for (int I = 0; I < nocc; ++I) {
        DRvvo[I].assign(dr_per_I, 0.0);
        DRvov[I].assign(dr_per_I, 0.0);
        for (int m = 0; m < nocc; ++m)
            for (int a = 0; a < nvir; ++a)
                for (int c = 0; c < nvir; ++c) {
                    real_t svvo = 0.0, svov = 0.0;
                    for (int l = 0; l < nocc; ++l)
                        for (int e = 0; e < nvir; ++e) {
                            const real_t mcle = OVOV(m, c, l, e);   // (mc|le)
                            const real_t mecl = OVOV(m, e, l, c);   // (me|lc)
                            svvo += 2.0 * mcle * T2(I, l, a, e)
                                  -       mcle * T2(l, I, a, e)
                                  -       mecl * T2(I, l, a, e);
                            // Wovov ring DR[I][m,a(result),c(contracted)]
                            //   = -Σ_{e,l} ovov(m,e,l,c)·t2(I,l,e,a)   (e = internal vir)
                            svov += -mecl * T2(I, l, e, a);
                        }
                    DRvvo[I][(static_cast<size_t>(m) * NV + a) * NV + c] = svvo;
                    DRvov[I][(static_cast<size_t>(m) * NV + a) * NV + c] = svov;
                }
    }

    Eigen::Map<const RowMatXd> S(h_S.data(), nao, nao);

    // B-a.6f Stage F0: per-phase ctor-build chrono. Split between (i) the
    // per-(idx,I,l) t2_proj = barS·Y·barSᵀ build [t2_proj_ms] and (ii) the
    // inner (m, ap, dp, e) V·T contraction [vt_ms]. At 100 atoms one of
    // these is the multi-hour host hotspot — printed at the end when
    // subtract_dense=false (the BARE / dense-free production path).
    // Under OMP the per-thread sums are reduced; the printed split is the
    // sum of useful work across threads (≈ wall × num_threads for an
    // OMP-bound section).
    double t2_proj_ms = 0.0, vt_ms = 0.0;

    // B-a.6f Stage F2b: env-gated CUDA kernel for the V·T contraction. Only
    // active for the production (subtract_dense=false) BARE path. When ON,
    // each (idx, I) launches a single kernel covering the entire (m, ap, dp)
    // output, summing over (l, e) internally — sidesteps the cuBLAS launch
    // overhead that dominates per-(m, l) GEMMs for small n_pno. F2a Eigen
    // remains the fallback (and is the validation reference).
    const char* env_gpu_ring  = std::getenv("GANSU_DLPNO_NATIVE_BARE_GPU_RING");
    const bool  use_gpu_ring  = (!subtract_dense) &&
                                (env_gpu_ring && env_gpu_ring[0] == '1');
    const char* env_gpu_ringv = std::getenv("GANSU_DLPNO_NATIVE_BARE_GPU_RING_VALIDATE");
    const bool  gpu_ring_validate = use_gpu_ring &&
                                    (env_gpu_ringv && env_gpu_ringv[0] == '1');
    // First-pair self-check counter; only the first pair-role to take the
    // GPU path runs both paths and prints max|nat(GPU) - nat(F2a)|.
    // Atomic flag → at most one self-check print across OMP threads.
    int selfcheck_done = 0;

    // B-a.6f Stage F2a: per (idx, I) builder. Refactored from the scalar
    // (m, ap, dp, e) quad-loop into four Eigen GEMMs per (l, m) plus an
    // outer OMP-over-idx parallelisation. Per-call local max/time outputs
    // so the lambda is thread-safe (the outer loop uses OMP reductions to
    // fold them into max_delta / max_ring / t2_proj_ms / vt_ms).
    //
    // Math (M := V[m,l,:,:] of shape n×n in (rows=dp, cols=e) layout;
    //       T_l := t2_proj[I,l] of shape n×n in (rows=ap, cols=e)):
    //   Term A  nat_vvo += +2 · T_l · M^T            (Wovvo, V1 · T(ap,e))
    //   Term B  nat_vvo += -1 · T_l^T · M^T          (Wovvo, V1 · T(e,ap))
    //   Term C  nat_vvo += -1 · T_l · M              (Wovvo, V3 · T(ap,e))
    //   Term D  nat_vov += -1 · T_l^T · M            (Wovov, V3 · T(e,ap))
    auto apply_role = [&](int idx, int I, int n,
                          const Eigen::Map<const RowMatXd>& U,
                          const RowMatXd& Bij,            // bar_Q_ijᵀ · S  [n × nao]
                          std::vector<std::vector<real_t>>& dst_vvo,
                          std::vector<std::vector<real_t>>& dst_vov,
                          real_t& out_max_delta, real_t& out_max_ring,
                          double& out_t2_ms, double& out_vt_ms) {
        // V[A,B,P,Q]=(AP|BQ), layout ((A·nocc+B)·n+P)·n+Q. No Phase24 ovov for this
        // pair → leave the b1 congruence seed (incl. its dense ring) intact rather
        // than dropping the ring entirely.
        const bool have_V = (static_cast<int>(idx) < static_cast<int>(ph.V_ovov_pair.size())) &&
                            (static_cast<int>(ph.V_ovov_pair[idx].size()) ==
                             static_cast<int>(NO * NO * static_cast<size_t>(n) * n));
        if (!have_V) return;
        const std::vector<real_t>& V = ph.V_ovov_pair[idx];

        // t2_proj[I,l] in target PNO [n×n] for every l (t2_proj[l,I] = transpose).
        const auto t0_t2 = std::chrono::high_resolution_clock::now();
        std::vector<RowMatXd> t2Il(nocc, RowMatXd::Zero(n, n));
        for (int l = 0; l < nocc; ++l) {
            const int src = res.pair_lookup[static_cast<size_t>(I) * nocc + l];
            if (src < 0) continue;
            const int ns = n_pno[src];
            if (ns == 0) continue;
            if (static_cast<int>(res.pairs[src].Y.size()) != ns * ns) continue;
            Eigen::Map<const RowMatXd> barQ_src(res.pairs[src].bar_Q.data(), nao, ns);
            const RowMatXd barS = Bij * barQ_src;                      // [n × ns]
            Eigen::Map<const RowMatXd> Ysrc(res.pairs[src].Y.data(), ns, ns);
            const RowMatXd M = barS * Ysrc * barS.transpose();          // [n × n], ordering (src.i,src.j)
            if (res.setups[src].i == I) t2Il[l] = M;                     // occ order (I,l) == storage
            else                        t2Il[l] = M.transpose();          // (l,I) → transpose
        }
        out_t2_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0_t2).count();

        const auto t0_vt = std::chrono::high_resolution_clock::now();

        // B-a.6f Stage F2b: GPU kernel path for the production (dense-free)
        // branch. One kernel launch per (idx, I) computes nat_vvo / nat_vov
        // for all (m, ap, dp); the host-Eigen F2a loop below remains the
        // fallback. Self-check (gpu_ring_validate) recomputes the F2a result
        // for the first pair-role and prints max|GPU − F2a|.
#ifndef GANSU_CPU_ONLY
        if (use_gpu_ring && !subtract_dense) {
            std::vector<real_t> nat_vvo_gpu_flat, nat_vov_gpu_flat;
            const bool gpu_ok = ip_ring_vt_contract_gpu(V, t2Il, nocc, n,
                                                       nat_vvo_gpu_flat,
                                                       nat_vov_gpu_flat);
            if (gpu_ok) {
                if (gpu_ring_validate) {
                    // Optional self-check: run F2a Eigen for the same (idx, I)
                    // and compare. One-shot — first thread to reach this here
                    // takes the print, the rest skip.
                    bool take_check = false;
#ifdef _OPENMP
                    #pragma omp critical (ip_ring_f2b_selfcheck)
                    {
                        if (selfcheck_done == 0) { selfcheck_done = 1; take_check = true; }
                    }
#else
                    if (selfcheck_done == 0) { selfcheck_done = 1; take_check = true; }
#endif
                    if (take_check) {
                        RowMatXd nat_vvo_ref(n, n), nat_vov_ref(n, n);
                        real_t dmax = 0.0, refmax = 0.0;
                        for (int m = 0; m < nocc; ++m) {
                            nat_vvo_ref.setZero();
                            nat_vov_ref.setZero();
                            const size_t mV = static_cast<size_t>(m) * NO;
                            for (int l = 0; l < nocc; ++l) {
                                const RowMatXd& T = t2Il[l];
                                const size_t vbase = (mV + l) * static_cast<size_t>(n) * n;
                                Eigen::Map<const RowMatXd> M(V.data() + vbase, n, n);
                                nat_vvo_ref.noalias() += 2.0 * T * M.transpose();
                                nat_vvo_ref.noalias() -= T.transpose() * M.transpose();
                                nat_vvo_ref.noalias() -= T * M;
                                nat_vov_ref.noalias() -= T.transpose() * M;
                            }
                            const size_t base = static_cast<size_t>(m) * n * n;
                            for (int ap = 0; ap < n; ++ap)
                                for (int dp = 0; dp < n; ++dp) {
                                    const size_t off = static_cast<size_t>(ap) * n + dp;
                                    const real_t gvvo = nat_vvo_gpu_flat[base + off];
                                    const real_t gvov = nat_vov_gpu_flat[base + off];
                                    const real_t rvvo = nat_vvo_ref(ap, dp);
                                    const real_t rvov = nat_vov_ref(ap, dp);
                                    dmax  = std::max(dmax,  std::max(std::fabs(gvvo - rvvo),
                                                                     std::fabs(gvov - rvov)));
                                    refmax = std::max(refmax, std::max(std::fabs(rvvo),
                                                                        std::fabs(rvov)));
                                }
                        }
                        std::cout << "[bt-PNO B-a.6f F2b self-check] idx=" << idx
                                  << " I=" << I << " n=" << n
                                  << "  max|nat(GPU) - nat(F2a)| = "
                                  << std::scientific << dmax
                                  << "   max|nat(F2a)| = " << refmax
                                  << "   (expect ≤1e-12 = kernel math matches Eigen)"
                                  << std::defaultfloat << std::endl;
                    }
                }
                // Write GPU result into dst_vvo / dst_vov + update max_ring.
                real_t local_max_ring = 0.0;
                real_t* dvvo_base = dst_vvo[idx].data();
                real_t* dvov_base = dst_vov[idx].data();
                for (size_t k = 0; k < nat_vvo_gpu_flat.size(); ++k) {
                    const real_t vvo = nat_vvo_gpu_flat[k];
                    const real_t vov = nat_vov_gpu_flat[k];
                    dvvo_base[k] += vvo;
                    dvov_base[k] += vov;
                    local_max_ring = std::max(local_max_ring,
                                              std::max(std::fabs(vvo), std::fabs(vov)));
                }
                out_max_ring = std::max(out_max_ring, local_max_ring);
                out_vt_ms += std::chrono::duration<double, std::milli>(
                                 std::chrono::high_resolution_clock::now() - t0_vt).count();
                return;
            }
            // GPU failed (rare; usually OOM at extreme scale). Fall through
            // silently to the host F2a path so the build still completes.
        }
#endif

        RowMatXd nat_vvo(n, n), nat_vov(n, n);
        for (int m = 0; m < nocc; ++m) {
            // native ring (PNO space) for this (idx,I,m): four Eigen noalias GEMMs
            // per l, accumulated. Each GEMM is n × n × n.
            nat_vvo.setZero();
            nat_vov.setZero();
            const size_t mV = static_cast<size_t>(m) * NO;             // (m·nocc + l) row base
            for (int l = 0; l < nocc; ++l) {
                const RowMatXd& T = t2Il[l];                            // [n × n] rows=ap, cols=e
                const size_t vbase = (mV + l) * static_cast<size_t>(n) * n;  // V[m,l,·,·] base
                Eigen::Map<const RowMatXd> M(V.data() + vbase, n, n);   // rows=dp, cols=e
                // Term A: +2 · T · M^T  (Wovvo Wick-pair coefficient).
                nat_vvo.noalias() += 2.0 * T * M.transpose();
                // Term B: -1 · T^T · M^T  (Wovvo exchanged-occ).
                nat_vvo.noalias() -= T.transpose() * M.transpose();
                // Term C: -1 · T · M      (Wovvo cross V3 coefficient).
                nat_vvo.noalias() -= T * M;
                // Term D: -1 · T^T · M    (Wovov; one term only).
                nat_vov.noalias() -= T.transpose() * M;
            }
            const size_t base = static_cast<size_t>(m) * n * n;
            if (!subtract_dense) {
                // IP dense-free: add native ring alone on top of the Phase24 bare
                // seed (no dense DR built / no congruence subtraction).
                real_t* dvvo_ptr = dst_vvo[idx].data() + base;
                real_t* dvov_ptr = dst_vov[idx].data() + base;
                real_t local_max_ring = 0.0;
                for (int ap = 0; ap < n; ++ap)
                    for (int dp = 0; dp < n; ++dp) {
                        const size_t off = static_cast<size_t>(ap) * n + dp;
                        const real_t vvo = nat_vvo(ap, dp), vov = nat_vov(ap, dp);
                        dvvo_ptr[off] += vvo;
                        dvov_ptr[off] += vov;
                        local_max_ring = std::max(local_max_ring,
                                                  std::max(std::fabs(vvo), std::fabs(vov)));
                    }
                out_max_ring = std::max(out_max_ring, local_max_ring);
                continue;
            }
            // B-a.6c(b2) validate: Wovvo_pno += native_ring − congruence(dense_ring).
            // congruence of the dense ring: U^(ij)ᵀ · DR[I][m] · U^(ij)  [n×n].
            Eigen::Map<const RowMatXd> DRvm(DRvvo[I].data() + static_cast<size_t>(m) * NV * NV, nvir, nvir);
            Eigen::Map<const RowMatXd> DRom(DRvov[I].data() + static_cast<size_t>(m) * NV * NV, nvir, nvir);
            const RowMatXd cong_vvo = U.transpose() * DRvm * U;        // [n×n]
            const RowMatXd cong_vov = U.transpose() * DRom * U;

            real_t local_max_delta = 0.0, local_max_ring = 0.0;
            for (int ap = 0; ap < n; ++ap)
                for (int dp = 0; dp < n; ++dp) {
                    const real_t dvvo = nat_vvo(ap, dp) - cong_vvo(ap, dp);
                    const real_t dvov = nat_vov(ap, dp) - cong_vov(ap, dp);
                    dst_vvo[idx][base + static_cast<size_t>(ap) * n + dp] += dvvo;
                    dst_vov[idx][base + static_cast<size_t>(ap) * n + dp] += dvov;
                    local_max_delta = std::max(local_max_delta,
                                               std::max(std::fabs(dvvo), std::fabs(dvov)));
                    local_max_ring  = std::max(local_max_ring,
                                               std::max(std::fabs(cong_vvo(ap, dp)),
                                                        std::fabs(cong_vov(ap, dp))));
                }
            out_max_delta = std::max(out_max_delta, local_max_delta);
            out_max_ring  = std::max(out_max_ring,  local_max_ring);
        }
        out_vt_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::high_resolution_clock::now() - t0_vt).count();
    };

    // Eigen does internally launch threads via OpenMP / EIGEN_DONT_PARALLELIZE
    // can mix poorly with our outer #pragma omp parallel for. Set Eigen to a
    // single thread per OMP worker; the outer loop provides the parallelism.
#ifdef _OPENMP
    Eigen::setNbThreads(1);
#endif
    const auto t0_total = std::chrono::high_resolution_clock::now();
    // Periodic progress signal — anthracene/tetracene-scale ring builder is
    // a 2-6 min host-Eigen silent stretch with no Davidson activity to fall
    // back on (operator is still under construction).  Print at ~10% deciles
    // using an atomic pair counter so the OMP-parallel pairs land in order
    // of completion, not idx.  Default ON (~10 lines / ring builder run);
    // set GANSU_DLPNO_NATIVE_PROF=0 to silence both this and the F0 phase
    // split summary.
    const char* env_prog_phase = std::getenv("GANSU_DLPNO_NATIVE_PROF");
    const bool  print_phase_progress = !env_prog_phase || env_prog_phase[0] != '0';
    std::atomic<int> done_pairs{0};
    std::atomic<int> next_decile{1};
#ifdef _OPENMP
    // When the GPU kernel path is on, all OMP threads would serialise on the
    // same CUDA default-stream → keep the OMP parallel construct (for the
    // F2a fallback case the reductions are correct sequentially too) but
    // gate parallelism via the `if` clause.
    #pragma omp declare reduction(maxr : real_t : omp_out = std::max(omp_out, omp_in)) initializer(omp_priv = 0.0)
    const int _ip_ring_threads = dlpno_eom_ring_omp_cap();
    #pragma omp parallel for schedule(dynamic) if (!use_gpu_ring) \
            num_threads(_ip_ring_threads) \
            reduction(maxr : max_delta) reduction(maxr : max_ring) \
            reduction(+   : t2_proj_ms) reduction(+   : vt_ms)
#endif
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n != 0) {
            Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
            Eigen::Map<const RowMatXd> barQ_ij(res.pairs[idx].bar_Q.data(), nao, n);
            const RowMatXd Bij = barQ_ij.transpose() * S;                 // [n × nao]
            const int i = res.setups[idx].i;
            const int j = res.setups[idx].j;
            apply_role(idx, i, n, U, Bij, io.Wovvo_pno_occi, io.Wovov_pno_occi,
                       max_delta, max_ring, t2_proj_ms, vt_ms);
            apply_role(idx, j, n, U, Bij, io.Wovvo_pno_occj, io.Wovov_pno_occj,
                       max_delta, max_ring, t2_proj_ms, vt_ms);
        }
        if (print_phase_progress && !subtract_dense) {
            const int done = done_pairs.fetch_add(1) + 1;
            int expected = next_decile.load();
            while (expected <= 10 && done * 10 >= n_pairs * expected) {
                if (next_decile.compare_exchange_strong(expected, expected + 1)) {
                    const double elapsed_s = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - t0_total).count();
                    const double eta_s = (expected > 0)
                        ? elapsed_s * (10.0 - expected) / expected
                        : 0.0;
                    #pragma omp critical(eom_ring_progress)
                    std::cout << "  [bt-PNO IP ring builder] " << (expected * 10)
                              << "% (" << done << "/" << n_pairs << " pairs, "
                              << std::fixed << std::setprecision(1) << elapsed_s
                              << "s elapsed, ETA " << eta_s << "s)"
                              << std::defaultfloat << std::endl;
                    break;
                }
                // CAS failed → expected was updated to current next_decile;
                // re-check whether we still need to print this decile.
            }
        }
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::high_resolution_clock::now() - t0_total).count();
    // B-a.6f Stage F0 phase split (BARE / dense-free path only — i.e., the
    // path that runs at production scale; the b2 validate path is gate-only).
    // F2a: under OMP the t2_proj/vt sums are across threads (≈ wall × #threads);
    // total_ms is wall. Speed-up ≈ (t2_proj + vt) / total.
    // F2b: when the GPU kernel path is on, OMP is disabled (sequential pair
    // iteration) → t2_proj/vt sums ≈ wall. Same fields, different ratio.
    if (!subtract_dense) {
#ifdef _OPENMP
        const int nthr = use_gpu_ring ? 1 : omp_get_max_threads();
#else
        const int nthr = 1;
#endif
        std::cout << "[bt-PNO B-a.6f IP ring split]  wall = " << std::fixed
                  << std::setprecision(1) << total_ms << " ms"
                  << "   t2_proj (barS·Y·barSᵀ work) = " << t2_proj_ms << " ms"
                  << "   V·T contraction (work) = " << vt_ms << " ms"
                  << "   nthreads = " << nthr
                  << "   path = " << (use_gpu_ring ? "GPU (F2b)" : "host F2a (Eigen+OMP)")
                  << std::defaultfloat << std::endl;
    }
}

// ===========================================================================
//  EA mirror (B-EA.6d): per-pair PNO Wvvvv.
// ===========================================================================

// 4-index congruence W_pno[a'b'c'd'] = Σ_{abcd} U[a,a']U[b,b']U[c,c']U[d,d'] W[abcd],
// as four sequential single-index transforms. Uflat is row-major [nv × n];
// W and out are row-major (((·)*…)). Host reference (the GPU port lives in the EA
// native operator; this is the gpu_selfcheck_ comparison target).
void congruence4(const std::vector<real_t>& W,
                 const std::vector<real_t>& Uflat, int nv, int n,
                 std::vector<real_t>& out) {
    auto Uat = [&](int a, int ap) -> real_t { return Uflat[static_cast<size_t>(a) * n + ap]; };
    const size_t NV = static_cast<size_t>(nv);
    // step 1: a→a'   A[a',b,c,d]   (dims n·nv·nv·nv)
    std::vector<real_t> A(static_cast<size_t>(n) * NV * NV * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int b = 0; b < nv; ++b)
            for (int c = 0; c < nv; ++c)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int a = 0; a < nv; ++a)
                        s += Uat(a, ap) * W[(((static_cast<size_t>(a) * NV + b) * NV + c) * NV + d)];
                    A[(((static_cast<size_t>(ap) * NV + b) * NV + c) * NV + d)] = s;
                }
    // step 2: b→b'   B[a',b',c,d]   (dims n·n·nv·nv)
    std::vector<real_t> B(static_cast<size_t>(n) * n * NV * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int c = 0; c < nv; ++c)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int b = 0; b < nv; ++b)
                        s += Uat(b, bp) * A[(((static_cast<size_t>(ap) * NV + b) * NV + c) * NV + d)];
                    B[(((static_cast<size_t>(ap) * n + bp) * NV + c) * NV + d)] = s;
                }
    // step 3: c→c'   C[a',b',c',d]  (dims n·n·n·nv)
    std::vector<real_t> C(static_cast<size_t>(n) * n * n * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int cp = 0; cp < n; ++cp)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int c = 0; c < nv; ++c)
                        s += Uat(c, cp) * B[(((static_cast<size_t>(ap) * n + bp) * NV + c) * NV + d)];
                    C[(((static_cast<size_t>(ap) * n + bp) * n + cp) * NV + d)] = s;
                }
    // step 4: d→d'   out[a',b',c',d']  (dims n⁴)
    out.assign(static_cast<size_t>(n) * n * n * n, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int cp = 0; cp < n; ++cp)
                for (int dp = 0; dp < n; ++dp) {
                    real_t s = 0.0;
                    for (int d = 0; d < nv; ++d)
                        s += Uat(d, dp) * C[(((static_cast<size_t>(ap) * n + bp) * n + cp) * NV + d)];
                    out[(((static_cast<size_t>(ap) * n + bp) * n + cp) * n + dp)] = s;
                }
}

DressedPnoEA build_dressed_pno_ea_vvvv(const std::vector<real_t>& h_Wvvvv,
                                       const std::vector<std::vector<real_t>>& Uocc,
                                       const std::vector<int>& n_pno_ii,
                                       int nvir)
{
    const int nocc = static_cast<int>(n_pno_ii.size());
    DressedPnoEA out;
    out.Wvvvv_pno.assign(nocc, {});
    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        congruence4(h_Wvvvv, Uocc[j], nvir, n, out.Wvvvv_pno[j]);
    }
    return out;
}

DressedPnoEA build_dressed_pno_ea_vvvv_bare(const DLPNOLMP2Result& res,
                                            const std::vector<int>& n_pno_ii)
{
    const int nocc = static_cast<int>(n_pno_ii.size());
    const Phase24Integrals& ph = res.phase24;
    DressedPnoEA out;
    out.Wvvvv_pno.assign(nocc, {});
    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        const int idx_jj = res.pair_lookup[static_cast<size_t>(j) * nocc + j];
        if (idx_jj < 0) continue;
        const size_t n4 = static_cast<size_t>(n) * n * n * n;
        // W_pair^(jj)[a,b,c,d] = (ac|bd) in PNO(jj), layout ((a·n+b)·n+c)·n+d —
        // the bare Wvvvv contribution, with NO dense nvir⁴ materialisation.
        if (idx_jj < static_cast<int>(ph.W_pair.size()) &&
            ph.W_pair[idx_jj].size() == n4) {
            out.Wvvvv_pno[j] = ph.W_pair[idx_jj];   // direct copy (same [a,b,c,d] layout)
        } else {
            out.Wvvvv_pno[j].assign(n4, 0.0);       // missing block → bare = 0 (ring still added)
        }
    }
    return out;
}

void build_dressed_pno_ea_vvvv_ring(DressedPnoEA& io,
                                    const DLPNOLMP2Result& res,
                                    const std::vector<real_t>& h_ovov_lmo,
                                    const std::vector<real_t>& h_t2_lmo,
                                    const std::vector<std::vector<real_t>>& Uocc,
                                    const std::vector<real_t>& h_S,
                                    const std::vector<int>& n_pno_ii,
                                    int nao, int nocc, int nvir,
                                    real_t& max_delta, real_t& max_ring,
                                    bool subtract_dense)
{
    const size_t NV = static_cast<size_t>(nvir);
    const size_t NO = static_cast<size_t>(nocc);
    const Phase24Integrals& ph = res.phase24;
    max_delta = 0.0;
    max_ring  = 0.0;

    // ovov(k,c,l,d) = (kc|ld), layout ((k·nvir+c)·nocc+l)·nvir+d.
    auto OVOV = [&](int k, int c, int l, int d) -> real_t {
        return h_ovov_lmo[(((static_cast<size_t>(k) * NV + c) * NO + l) * NV + d)];
    };
    // t2(k,l,a,b), layout ((k·nocc+l)·nvir+a)·nvir+b.
    auto T2 = [&](int k, int l, int a, int b) -> real_t {
        return h_t2_lmo[(((static_cast<size_t>(k) * NO + l) * NV + a) * NV + b)];
    };

    // --- Dense Wvvvv ring (canonical virtual basis), built once ---
    //   DR[a,b,c,d] = Σ_{k,l} (kc|ld) t2(k,l,a,b).
    // Only for the B-EA.6d validate gate (subtract_dense); the true-scaling
    // path (subtract_dense=false) NEVER materialises this nvir⁴ buffer.
    std::vector<real_t> DR;
    if (subtract_dense) {
        DR.assign(NV * NV * NV * NV, 0.0);
        #pragma omp parallel for
        for (int a = 0; a < nvir; ++a)
            for (int b = 0; b < nvir; ++b)
                for (int c = 0; c < nvir; ++c)
                    for (int d = 0; d < nvir; ++d) {
                        real_t s = 0.0;
                        for (int k = 0; k < nocc; ++k)
                            for (int l = 0; l < nocc; ++l)
                                s += OVOV(k, c, l, d) * T2(k, l, a, b);
                        DR[(((static_cast<size_t>(a) * NV + b) * NV + c) * NV + d)] = s;
                    }
    }

    Eigen::Map<const RowMatXd> S(h_S.data(), nao, nao);

    // B-a.6f Stage F0: per-phase ctor-build chrono (EA mirror of IP). Split
    // between (i) the t2kl = barS·Y·barSᵀ build over all (k,l) and (ii) the
    // V·T 4-index contraction. Printed at end when subtract_dense=false.
    double t2_proj_ms = 0.0, vt_ms = 0.0;
    const auto t0_total = std::chrono::high_resolution_clock::now();
    // Periodic progress signal — anthracene-scale EA Wvvvv ring builder is
    // a 10-30 s host-Eigen silent stretch (Tetracene scale: 2-5 min) with
    // no Davidson activity to fall back on.  Outer loop runs serially over
    // active occupied indices j (nocc iterations); print at deciles.
    const char* env_prog_phase = std::getenv("GANSU_DLPNO_NATIVE_PROF");
    const bool  print_phase_progress = !env_prog_phase || env_prog_phase[0] != '0';
    int ea_done_j = 0, ea_next_decile = 1;

    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        const int idx_jj = res.pair_lookup[static_cast<size_t>(j) * nocc + j];
        if (idx_jj < 0) continue;
        // V_ovov_pair^(jj)[k,l,c',d'] = (k c'|l d'), layout ((k·nocc+l)·n+c')·n+d'.
        // No Phase24 V for this pair → leave the b1 congruence seed intact.
        const bool have_V = (idx_jj < static_cast<int>(ph.V_ovov_pair.size())) &&
                            (static_cast<int>(ph.V_ovov_pair[idx_jj].size()) ==
                             static_cast<int>(NO * NO * static_cast<size_t>(n) * n));
        if (!have_V) continue;
        const std::vector<real_t>& V = ph.V_ovov_pair[idx_jj];

        Eigen::Map<const RowMatXd> barQ_jj(res.pairs[idx_jj].bar_Q.data(), nao, n);
        const RowMatXd Bjj = barQ_jj.transpose() * S;        // [n × nao]

        // t2_proj^(jj←kl)[a',b'] = oriented(barS Y_src barSᵀ) [n×n] for every (k,l).
        const auto t0_t2 = std::chrono::high_resolution_clock::now();
        std::vector<RowMatXd> t2kl(static_cast<size_t>(nocc) * nocc, RowMatXd::Zero(n, n));
        for (int k = 0; k < nocc; ++k)
            for (int l = 0; l < nocc; ++l) {
                const int src = res.pair_lookup[static_cast<size_t>(k) * nocc + l];
                if (src < 0) continue;
                const int ns = res.pairs[src].n_pno;
                if (ns == 0) continue;
                if (static_cast<int>(res.pairs[src].Y.size()) != ns * ns) continue;
                Eigen::Map<const RowMatXd> barQ_src(res.pairs[src].bar_Q.data(), nao, ns);
                const RowMatXd barS = Bjj * barQ_src;                       // [n × ns]
                Eigen::Map<const RowMatXd> Ysrc(res.pairs[src].Y.data(), ns, ns);
                const RowMatXd M = barS * Ysrc * barS.transpose();           // [n × n], order (src.i,src.j)
                t2kl[static_cast<size_t>(k) * nocc + l] =
                    (res.setups[src].i == k) ? M : RowMatXd(M.transpose());   // (k,l) order
            }
        t2_proj_ms += std::chrono::duration<double, std::milli>(
                          std::chrono::high_resolution_clock::now() - t0_t2).count();

        // native_ring[a',b',c',d'] = Σ_{k,l} V[k,l,c',d'] · t2_proj[k,l][a',b'].
        const auto t0_vt = std::chrono::high_resolution_clock::now();
        std::vector<real_t> nat(static_cast<size_t>(n) * n * n * n, 0.0);
        for (int k = 0; k < nocc; ++k)
            for (int l = 0; l < nocc; ++l) {
                const RowMatXd& T = t2kl[static_cast<size_t>(k) * nocc + l];
                const size_t vbase = (static_cast<size_t>(k) * nocc + l) * n * n;  // V[k,l,·,·]
                #pragma omp parallel for num_threads(dlpno_eom_ring_omp_cap())
                for (int ap = 0; ap < n; ++ap)
                    for (int bp = 0; bp < n; ++bp) {
                        const real_t Tab = T(ap, bp);
                        if (Tab == 0.0) continue;
                        real_t* row = nat.data() + (((static_cast<size_t>(ap) * n + bp) * n) * n);
                        for (int cp = 0; cp < n; ++cp)
                            for (int dp = 0; dp < n; ++dp)
                                row[static_cast<size_t>(cp) * n + dp] +=
                                    Tab * V[vbase + static_cast<size_t>(cp) * n + dp];
                    }
            }
        vt_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::high_resolution_clock::now() - t0_vt).count();

        std::vector<real_t>& W = io.Wvvvv_pno[j];
        const size_t n4 = static_cast<size_t>(n) * n * n * n;
        if (subtract_dense) {
            // B-EA.6d validate: Wvvvv_pno += native_ring − congruence(dense_ring).
            std::vector<real_t> cong;
            congruence4(DR, Uocc[j], nvir, n, cong);
            for (size_t e = 0; e < n4; ++e) {
                const real_t delta = nat[e] - cong[e];
                W[e] += delta;
                max_delta = std::max(max_delta, std::fabs(delta));
                max_ring  = std::max(max_ring,  std::fabs(cong[e]));
            }
        } else {
            // B-EA.6e true scaling: Wvvvv_pno += native_ring (no dense reference).
            for (size_t e = 0; e < n4; ++e) {
                W[e] += nat[e];
                max_ring = std::max(max_ring, std::fabs(nat[e]));
            }
        }
        // Decile progress (single-threaded outer loop → no atomics needed).
        if (print_phase_progress && !subtract_dense) {
            ++ea_done_j;
            while (ea_next_decile <= 10 && ea_done_j * 10 >= nocc * ea_next_decile) {
                const double elapsed_s = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - t0_total).count();
                const double eta_s = (ea_next_decile > 0)
                    ? elapsed_s * (10.0 - ea_next_decile) / ea_next_decile
                    : 0.0;
                std::cout << "  [bt-PNO EA Wvvvv ring builder] " << (ea_next_decile * 10)
                          << "% (" << ea_done_j << "/" << nocc << " occ, "
                          << std::fixed << std::setprecision(1) << elapsed_s
                          << "s elapsed, ETA " << eta_s << "s)"
                          << std::defaultfloat << std::endl;
                ++ea_next_decile;
            }
        }
    }

    // B-a.6f Stage F0 phase split (BARE / dense-free path only).
    if (!subtract_dense) {
        const double total_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::high_resolution_clock::now() - t0_total).count();
        std::cout << "[bt-PNO B-a.6f EA ring split]  total = " << std::fixed
                  << std::setprecision(1) << total_ms << " ms"
                  << "   t2kl (barS·Y·barSᵀ) = " << t2_proj_ms << " ms"
                  << "   V·T contraction = " << vt_ms << " ms"
                  << std::defaultfloat << std::endl;
    }
}

} // namespace gansu
