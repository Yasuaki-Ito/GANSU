/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_pair_data.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "diis.hpp"
#include "dlpno_pno.hpp"
#include "dlpno_picache_gpu.hpp"
#include "dlpno_resid_gpu.hpp"
#ifndef GANSU_CPU_ONLY
#include "multi_gpu_manager.hpp"
#endif

namespace gansu {

namespace {

// Multi-GPU work threshold (per-GPU) for LMP2 / CCSD T2 dispatch. Below
// this, callers fall back to n_gpus=1. The default 1e11 was picked from
// the 2026-05-13 A100×8 benchmark where 8-GPU iter was 2-3× slower than
// 1-GPU due to picache D2H pi_pad (~40 GB/iter) serialising on PCIe.
//
// Step Z (2026-05-16) made pi_pad GPU-resident (no large per-iter D2H),
// so the assumption underlying the fallback may no longer hold. Override
// at runtime with the env var `GANSU_DLPNO_MIN_WORK_PER_GPU`:
//   GANSU_DLPNO_MIN_WORK_PER_GPU=0           # force respect num_gpus (no fallback)
//   GANSU_DLPNO_MIN_WORK_PER_GPU=10000000    # custom small value
// Read once on first call (env vars don't change mid-run).
long long get_dlpno_min_work_per_gpu()
{
    static const long long val = []() -> long long {
        const char* env = std::getenv("GANSU_DLPNO_MIN_WORK_PER_GPU");
        if (env && *env) {
            char* end = nullptr;
            const long long v = std::strtoll(env, &end, 10);
            if (end != env && v >= 0) return v;
        }
        return 100000000000LL;  // legacy default
    }();
    return val;
}

}  // namespace

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
} // anonymous namespace

// ---------------------------------------------------------------------------
//  build_pair_data — see header for full documentation.
// ---------------------------------------------------------------------------
void build_pair_data(const PairSetup& s,
                     const std::vector<real_t>& T_pao,
                     real_t t_cut_pno,
                     bool   os_only,
                     int nao,
                     PairData& out)
{
    const int n_pao = s.n_pao;
    const bool diag = (s.i == s.j);
    out = PairData{};
    out.n_pno = 0;
    if (n_pao == 0) return;

    auto pno = build_pno_from_T(T_pao.data(), diag, n_pao, t_cut_pno, os_only);
    const int n_pno = pno.n_kept;
    out.n_pno = n_pno;
    if (n_pno == 0) return;

    Eigen::Map<const RowMatXd> Vmat(s.V.data(), n_pao, n_pao);
    Eigen::Map<const RowMatXd> Dpno(pno.d_pno.data(), n_pao, n_pno);

    const RowMatXd Vd    = Vmat * Dpno;             // (n_pao × n_pno)
    const RowMatXd K_pno = Dpno.transpose() * Vd;   // (n_pno × n_pno)

    // F projected onto the PNO subspace.
    RowMatXd F_pno(n_pno, n_pno);
    for (int aa = 0; aa < n_pno; ++aa)
        for (int bb = 0; bb < n_pno; ++bb) {
            real_t sum = 0.0;
            for (int a = 0; a < n_pao; ++a)
                sum += pno.d_pno[a * n_pno + aa]
                     * s.eps_a[a]
                     * pno.d_pno[a * n_pno + bb];
            F_pno(aa, bb) = sum;
        }
    F_pno = 0.5 * (F_pno + F_pno.transpose());

    Eigen::SelfAdjointEigenSolver<RowMatXd> es(F_pno);
    if (es.info() != Eigen::Success)
        throw std::runtime_error(
            "DLPNO pair data: F_pno eigendecomposition failed");
    const Eigen::VectorXd lam = es.eigenvalues();
    const RowMatXd        W   = es.eigenvectors();

    const RowMatXd L_W   = W.transpose() * K_pno * W;
    Eigen::Map<const RowMatXd> Cpao(s.C_can_pair.data(), nao, n_pao);
    const RowMatXd DW    = Dpno * W;                 // (n_pao × n_pno)
    const RowMatXd barQ  = Cpao * DW;                // (nao   × n_pno)

    out.bar_Q.assign(static_cast<size_t>(nao) * n_pno, 0.0);
    Eigen::Map<RowMatXd>(out.bar_Q.data(), nao, n_pno) = barQ;
    out.M.assign(static_cast<size_t>(n_pao) * n_pno, 0.0);
    Eigen::Map<RowMatXd>(out.M.data(), n_pao, n_pno) = DW;
    out.Lambda.resize(n_pno);
    for (int aa = 0; aa < n_pno; ++aa) out.Lambda[aa] = lam(aa);
    out.L.assign(static_cast<size_t>(n_pno) * n_pno, 0.0);
    Eigen::Map<RowMatXd>(out.L.data(), n_pno, n_pno) = L_W;

    // Initial Y from per-pair Sylvester (semi-canonical zeroth order).
    out.Y.assign(static_cast<size_t>(n_pno) * n_pno, 0.0);
    const real_t shift = s.F_ii + s.F_jj;
    for (int aa = 0; aa < n_pno; ++aa)
        for (int bb = 0; bb < n_pno; ++bb) {
            const real_t denom = lam(aa) + lam(bb) - shift;
            out.Y[aa * n_pno + bb] = -L_W(aa, bb) / denom;
        }
}

// ---------------------------------------------------------------------------
//  reconstruct_T_pao — see header.
// ---------------------------------------------------------------------------
void reconstruct_T_pao(const PairSetup& s,
                       const PairData&  pd,
                       std::vector<real_t>& T_pao_out)
{
    const int n_pao = s.n_pao;
    T_pao_out.assign(static_cast<size_t>(n_pao) * n_pao, 0.0);
    if (pd.n_pno == 0 || n_pao == 0) return;
    Eigen::Map<const RowMatXd> M(pd.M.data(), n_pao, pd.n_pno);
    Eigen::Map<const RowMatXd> Y(pd.Y.data(), pd.n_pno, pd.n_pno);
    const RowMatXd T = M * Y * M.transpose();
    Eigen::Map<RowMatXd>(T_pao_out.data(), n_pao, n_pao) = T;
}

// ---------------------------------------------------------------------------
//  iterate_lmp2 — see header.
// ---------------------------------------------------------------------------
LMP2Status iterate_lmp2(
    const std::vector<PairSetup>& setups,
    std::vector<PairData>&        pairs,
    const std::vector<int>&       pair_lookup,
    const std::vector<real_t>&    F_LMO,
    const real_t*                 h_S,
    int nocc, int nao,
    int max_iter, real_t conv_tol,
    int verbose, const char* round_tag,
    int num_gpus,
    bool user_explicit_n_gpus)
{
    LMP2Status st;
    Eigen::Map<const RowMatXd> Smat(h_S, nao, nao);
    const real_t kFLMOThresh = 1e-14;

    // ---- Profile probes (Step 4 follow-up; same naming as the CCSD iter) ----
    using prof_clock = std::chrono::steady_clock;
    double dt_barS    = 0.0;  // one-time
    double dt_picache = 0.0;
    double dt_resid   = 0.0;
    const auto t_iter_total_0 = prof_clock::now();

    std::vector<std::vector<real_t>> Y_old(pairs.size());

    // Phase 1 (sparse barS) — per-output-pair coupling list: the LMP2 residual
    // reads pi_cache[idx][idx_kl] only for idx_kl ∈ {(k,j): k≠i} ∪ {(i,l): l≠j}
    // (~2·nocc of N_pair columns), and pi[idx][idx_kl] depends on a SINGLE barS
    // block barS[idx][idx_kl]. So this list is the EXACT set of barS blocks the
    // LMP2 path ever needs — restricting the build to it is bit-exact, not an
    // approximation. Built here (early) so the sparse barS build below can use
    // it; also consumed by the picache_gather rebuild_needed path further down
    // and by the PiCacheGpu CSR build (rebuild_needed). Cheap: O(N_pair·nocc).
    const bool bars_sparse = []() {
        const char* e = std::getenv("GANSU_DLPNO_LMP2_BARS_SPARSE");
        return e && e[0] == '1';
    }();
    std::vector<std::vector<int>> needed_ikl_per_pair(pairs.size());
    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        if (pairs[idx].n_pno == 0) continue;
        const int i = setups[idx].i, j = setups[idx].j;
        std::vector<int>& nb = needed_ikl_per_pair[idx];
        for (int k = 0; k < nocc; ++k) {           // i-coupling: (k, j)
            if (k == i) continue;
            const int idx_kj = pair_lookup[static_cast<size_t>(k) * nocc + j];
            if (idx_kj >= 0 && pairs[idx_kj].n_pno > 0) nb.push_back(idx_kj);
        }
        for (int l = 0; l < nocc; ++l) {           // j-coupling: (i, l)
            if (l == j) continue;
            const int idx_il = pair_lookup[static_cast<size_t>(i) * nocc + l];
            if (idx_il >= 0 && pairs[idx_il].n_pno > 0) nb.push_back(idx_il);
        }
        std::sort(nb.begin(), nb.end());
        nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
    }

    // ---- Pre-computed bar_S^{(ij,kl)} cache (one-time). ----
    // Identical role to the CCSD iter's barS_cache: the inter-pair Fock
    // coupling needs Σ projection · Y · projection^T for every (idx, idx_kl),
    // and the projection bar_Q^{(ij)T} · S · bar_Q^{(kl)} only depends on
    // pair structures, so we build it once outside the iter loop.
    std::vector<std::vector<RowMatXd>> barS_cache(pairs.size());
    {
        const auto t0 = prof_clock::now();
        for (size_t i_ij = 0; i_ij < pairs.size(); ++i_ij) {
            barS_cache[i_ij].resize(pairs.size());
        }
        // dynamic(4): per-i_ij work is O(N_pair × n_ij² × nao); empty pairs
        // (n_ij=0) `continue` fast. Static partitioning gives ~3× imbalance
        // due to clumped empties; dynamic levels it.
        #pragma omp parallel for schedule(dynamic, 4)
        for (long long i_ij = 0; i_ij < static_cast<long long>(pairs.size());
             ++i_ij)
        {
            const int n_ij = pairs[i_ij].n_pno;
            if (n_ij == 0) continue;
            Eigen::Map<const RowMatXd> bQ_ij(
                pairs[i_ij].bar_Q.data(), nao, n_ij);
            const RowMatXd bQ_ij_T_S = bQ_ij.transpose() * Smat;  // n_ij × nao
            if (bars_sparse) {
                // Phase 1 (Stage 2b): build only the coupling columns the
                // residual reads → O(N_pair·nocc) host work instead of
                // O(N_pair²). Bit-exact (same blocks, others never consumed).
                for (int i_kl : needed_ikl_per_pair[i_ij]) {
                    const int n_kl = pairs[i_kl].n_pno;
                    if (n_kl == 0) continue;
                    Eigen::Map<const RowMatXd> bQ_kl(
                        pairs[i_kl].bar_Q.data(), nao, n_kl);
                    barS_cache[i_ij][i_kl].noalias() = bQ_ij_T_S * bQ_kl;
                }
            } else {
                for (size_t i_kl = 0; i_kl < pairs.size(); ++i_kl) {
                    const int n_kl = pairs[i_kl].n_pno;
                    if (n_kl == 0) continue;
                    Eigen::Map<const RowMatXd> bQ_kl(
                        pairs[i_kl].bar_Q.data(), nao, n_kl);
                    barS_cache[i_ij][i_kl].noalias() = bQ_ij_T_S * bQ_kl;
                }
            }
        }
        dt_barS += std::chrono::duration<double>(prof_clock::now() - t0).count();
    }

    // pi_cache built per iter (depends on Y_old).
    std::vector<std::vector<RowMatXd>> pi_cache(pairs.size());
    for (size_t i_ij = 0; i_ij < pairs.size(); ++i_ij) {
        pi_cache[i_ij].resize(pairs.size());
    }

    // Step 6.0 — GPU port for pi_cache build (CPU fallback when GPU is
    // unavailable or alloc fails). Padded barS uploaded once here.
    std::vector<int> n_pno_per_pair(pairs.size(), 0);
    int max_n = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        n_pno_per_pair[i] = pairs[i].n_pno;
        if (pairs[i].n_pno > max_n) max_n = pairs[i].n_pno;
    }

    // ---- Multi-GPU pair-slab partition (n_pno²-weighted greedy). ----
    // For single-GPU (num_gpus<=1) we build one PiCacheGpu instance over
    // the full [0, N_pair) range, preserving the previous behavior bit-exactly.
    //
    // Auto-fallback: same threshold logic as iterate_dlpno_ccsd_t2. For
    // LMP2 the per-iter work is lighter (no oooo lad / ph-ladder), so the
    // crossover N_pair is higher than for CCSD T2.
#ifdef GANSU_CPU_ONLY
    const int n_gpus = 1;
#else
    int n_gpus = (num_gpus < 1) ? 1 : num_gpus;
    if (n_gpus > 1) {
        long long work_estimate = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            const long long n = pairs[i].n_pno;
            work_estimate += n * n;
        }
        work_estimate *= static_cast<long long>(nocc) * nocc;
        // Auto-fallback heuristic: skip when the user explicitly specified
        // `--num_gpus N > 0` (Option C, 2026-05-19). The historical 1e11
        // threshold predates Step Z (GPU-resident pi_pad) + S11 Phase 2
        // (packed cuBLAS + ResidGpu activate); for cholesterol-class systems
        // multi-GPU is now strictly required to activate ResidGpu, so
        // honouring user intent is the right default. The env var
        // GANSU_DLPNO_MIN_WORK_PER_GPU still lets users force the legacy
        // heuristic on `--num_gpus -1` (auto) runs.
        const long long kMinWorkPerGpu = get_dlpno_min_work_per_gpu();
        const long long per_gpu = work_estimate / n_gpus;
        const bool fall_back =
            !user_explicit_n_gpus && (per_gpu < kMinWorkPerGpu);
        if (fall_back) {
            if (verbose >= 1) {
                std::cout << "[DLPNO-LMP2-PROF] " << round_tag
                          << " multi-GPU auto-fallback to n_gpus=1"
                          << " (work=" << work_estimate
                          << " per-GPU=" << per_gpu
                          << " < threshold=" << kMinWorkPerGpu << ")"
                          << std::endl;
            }
            n_gpus = 1;
        } else {
            MultiGpuManager::instance().initialize(n_gpus);
        }
    }
#endif
    const int N_pair = static_cast<int>(pairs.size());
    std::vector<int> slab_starts(n_gpus + 1, 0);
    {
        long long total_w = 0;
        std::vector<long long> w(N_pair, 0);
        for (int i = 0; i < N_pair; ++i) {
            const long long n = pairs[i].n_pno;
            w[i] = n * n;  // dominant per-pair compute scales as n_pno²
            total_w += w[i];
        }
        long long target = (total_w + n_gpus - 1) / n_gpus;
        long long acc = 0;
        int g = 0;
        for (int i = 0; i < N_pair && g < n_gpus - 1; ++i) {
            acc += w[i];
            if (acc >= target * (g + 1)) {
                slab_starts[g + 1] = i + 1;
                ++g;
            }
        }
        slab_starts[n_gpus] = N_pair;
        // Defensive: clamp any unset entries.
        for (int d = 1; d < n_gpus; ++d) {
            if (slab_starts[d] < slab_starts[d - 1])
                slab_starts[d] = slab_starts[d - 1];
        }
    }

    std::vector<std::unique_ptr<PiCacheGpu>> pgpus(n_gpus);
    if (n_gpus == 1) {
        pgpus[0] = std::make_unique<PiCacheGpu>(
            barS_cache, n_pno_per_pair, max_n, nullptr, nullptr, 0,
            0, N_pair, 0);
    } else {
#ifndef GANSU_CPU_ONLY
        // Step S8 (2026-05-17): parallel construction across devices.
        // Each constructor reads shared input (barS_cache, n_pno_per_pair)
        // read-only and writes only its own device memory + per-instance
        // host members, so concurrent execution on distinct devices is
        // race-free. Serialising it was previously hiding ~9 s/device of
        // barS H2D + buffer alloc cost; at LMP2 scale (max_n=3) it is
        // already cheap, but kept symmetric with the CCSD T2 path.
        #pragma omp parallel num_threads(n_gpus)
        {
#ifdef _OPENMP
            const int d = omp_get_thread_num();
#else
            const int d = 0;
#endif
            MultiGpuManager::DeviceGuard guard(d);
            pgpus[d] = std::make_unique<PiCacheGpu>(
                barS_cache, n_pno_per_pair, max_n,
                nullptr, nullptr, 0,
                slab_starts[d], slab_starts[d + 1], d);
        }
#else
        pgpus[0] = std::make_unique<PiCacheGpu>(
            barS_cache, n_pno_per_pair, max_n, nullptr, nullptr, 0,
            0, N_pair, 0);
#endif
    }

    // [picache-gather] Build the per-pair list of pi_cache columns the LMP2
    // residual actually reads: kl ∈ {(k,j): k≠i} ∪ {(i,l): l≠j}. That is only
    // ~2·nocc of the N_pair columns, so D2H'ing/scattering ONLY these (vs the
    // full column sweep) cuts the rebuild's dominant D2H (~76% of picache on
    // the critical-path device) by ~6×. Iter-invariant — depends only on pair
    // structure, not Y_old. Default ON (opt out with
    // GANSU_DLPNO_LMP2_PICACHE_GATHER=0); bit-exact (same projection blocks,
    // fewer transferred; host residual unchanged) — VALIDATED on naphthalene
    // A100×8: dev0 picache D2H 1.279→0.162 s (7.9×), MP2 iterate 8.3→3.1 s,
    // corr/IP0/EA0 bit-identical.
    const bool picache_gather = []() {
        const char* e = std::getenv("GANSU_DLPNO_LMP2_PICACHE_GATHER");
        return !e || e[0] != '0';
    }();
    // needed_ikl_per_pair is now built once early (above), so the sparse barS
    // build and the gather rebuild_needed call share the same coupling list.

    for (int iter = 0; iter < max_iter; ++iter) {
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            Y_old[idx] = pairs[idx].Y;
        }

        // ---- Build pi_cache from Y_old (GPU strided batched, see
        //      include/dlpno_picache_gpu.hpp for layout/falls back to CPU).
        {
            const auto t_pi0 = prof_clock::now();
            if (n_gpus > 1) {
#ifndef GANSU_CPU_ONLY
                #pragma omp parallel num_threads(n_gpus)
                {
#ifdef _OPENMP
                    const int d = omp_get_thread_num();
#else
                    const int d = 0;
#endif
                    MultiGpuManager::DeviceGuard guard(d);
                    if (picache_gather)
                        pgpus[d]->rebuild_needed(Y_old, pi_cache,
                                                 needed_ikl_per_pair);
                    else
                        pgpus[d]->rebuild(Y_old, pi_cache);
                }
#endif
            } else {
                if (picache_gather)
                    pgpus[0]->rebuild_needed(Y_old, pi_cache,
                                             needed_ikl_per_pair);
                else
                    pgpus[0]->rebuild(Y_old, pi_cache);
            }
            dt_picache += std::chrono::duration<double>(
                prof_clock::now() - t_pi0).count();
        }

        // ---- Per-pair residual + Jacobi update (parallel over pairs). ----
        // 2026-05-17 schedule change: matches the iterate_dlpno_ccsd_t2
        // observation — at cholesterol scale 3919/5886 pairs are empty
        // (n_pno=0, `continue` fast path) and the rest have n_pno spread
        // 1..26 (avg 3.1 for LMP2). Static schedule clumps empties on
        // some threads and strong pairs on others → ~3× imbalance. Switch
        // to dynamic(4) so threads steal balance. Bit-exact: scheduling
        // policy only, no computation reordering.
        const auto t_resid_0 = prof_clock::now();
        real_t r_max = 0.0;
        #pragma omp parallel for schedule(dynamic, 4) reduction(max:r_max)
        for (long long idx = 0; idx < static_cast<long long>(pairs.size()); ++idx) {
            PairData&        pij = pairs[idx];
            const PairSetup& sij = setups[idx];
            const int n = pij.n_pno;
            if (n == 0) continue;

            Eigen::Map<RowMatXd>       Y_ij(pij.Y.data(),         n, n);
            Eigen::Map<const RowMatXd> L_ij(pij.L.data(),         n, n);

            const real_t shift = sij.F_ii + sij.F_jj;
            RowMatXd R(n, n);
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    R(a, b) = L_ij(a, b)
                            + (pij.Lambda[a] + pij.Lambda[b] - shift)
                              * Y_old[idx][a * n + b];

            // Coupling on i:  − Σ_{k≠i} F_LMO[i,k] · pi_{kj}^{oriented}
            for (int k = 0; k < nocc; ++k) {
                if (k == sij.i) continue;
                const real_t F_ik = F_LMO[sij.i * nocc + k];
                if (std::fabs(F_ik) < kFLMOThresh) continue;

                const int idx_kj = pair_lookup[k * nocc + sij.j];
                const PairData&  pkj = pairs[idx_kj];
                const PairSetup& skj = setups[idx_kj];
                if (pkj.n_pno == 0) continue;

                const RowMatXd& pi_kj = pi_cache[idx][idx_kj];
                if (skj.i != k) {
                    R.noalias() -= F_ik * pi_kj.transpose();
                } else {
                    R.noalias() -= F_ik * pi_kj;
                }
            }

            // Coupling on j:  − Σ_{l≠j} F_LMO[l,j] · pi_{il}^{oriented}
            for (int l = 0; l < nocc; ++l) {
                if (l == sij.j) continue;
                const real_t F_lj = F_LMO[l * nocc + sij.j];
                if (std::fabs(F_lj) < kFLMOThresh) continue;

                const int idx_il = pair_lookup[sij.i * nocc + l];
                const PairData&  pil = pairs[idx_il];
                const PairSetup& sil = setups[idx_il];
                if (pil.n_pno == 0) continue;

                const RowMatXd& pi_il = pi_cache[idx][idx_il];
                if (sil.i != sij.i) {
                    R.noalias() -= F_lj * pi_il.transpose();
                } else {
                    R.noalias() -= F_lj * pi_il;
                }
            }

            real_t r_max_pair = 0.0;
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b) {
                    const real_t denom =
                        pij.Lambda[a] + pij.Lambda[b] - shift;
                    const real_t r = R(a, b);
                    Y_ij(a, b) -= r / denom;
                    r_max_pair = std::max(r_max_pair, std::fabs(r));
                }
            r_max = std::max(r_max, r_max_pair);
        }
        dt_resid += std::chrono::duration<double>(prof_clock::now() - t_resid_0).count();

        st.max_R = r_max;
        if (verbose >= 2) {
            std::cout << "[DLPNO] " << round_tag
                      << " LMP2 iter " << std::setw(3) << (iter + 1)
                      << "  max|R|=" << std::scientific
                      << std::setprecision(3) << r_max << std::endl;
        }
        if (r_max < conv_tol) {
            st.iters     = iter + 1;
            st.converged = true;
            break;
        }
        st.iters = iter + 1;
    }
    {
        const double dt_total =
            std::chrono::duration<double>(prof_clock::now() - t_iter_total_0).count();
        const double dt_acct = dt_barS + dt_picache + dt_resid;
        std::cout << "[DLPNO-LMP2-PROF] " << round_tag
                  << "  iters="    << st.iters
                  << "  total="    << std::fixed << std::setprecision(3) << dt_total << "s"
                  << "  barS="     << dt_barS
                  << "  picache="  << dt_picache
                  << "  resid="    << dt_resid
                  << "  other="    << (dt_total - dt_acct)
                  << std::endl;
    }
    return st;
}

// ---------------------------------------------------------------------------
//  iterate_dlpno_ccsd_t2 — Phase 2.3.1 wiring + 2.3.2/3 + 2.4.1/2/3 + 2.7 DIIS.
//
//  Phase 2.3.1 (dressing off): delegates verbatim to iterate_lmp2. Used as a
//  sanity hook on top of converged LMP2 amplitudes — must produce max_R ≈ 0.
//
//  Phase 2.3.2 (particle dressing on): adds the intra-pair F_ac dressing
//
//      ΔF^{(ij)}_{ac} = -Σ_d \tilde L^{(ij)}_{cd} · Y^{(ij)}_{ad}
//      \tilde L_{cd}  = 2 L_{cd} - L_{dc}
//
//  contracted into the residual as
//
//      ΔR^{(ij)}_{ab} = Σ_c ΔF^{(ij)}_{ac} Y^{(ij)}_{cb}
//                     + Σ_c ΔF^{(ij)}_{bc} Y^{(ij)}_{ac}
//                     = (ΔF · Y_old + Y_old · ΔF^T)_{ab}.
//
//  Phase 2.3.3 (hole dressing on diagonal pairs): adds
//
//      ΔF_{ii} = Σ_{cd} \tilde L^{(ii)}_{cd} Y^{(ii)}_{cd}     (scalar/LMO)
//      ΔR^{(ij)}_{ab} += -(ΔF_{ii} + ΔF_{jj}) · Y^{(ij),old}_{ab}.
//
//  (l=k=i restriction of canonical F_ki dressing; PySCF-verified to match
//  pair (i,i) MP2 correlation energy at machine precision.)
//
//  Equivalent canonical CCSD term mapping (T1 = 0, Stanton-Bartlett 1991):
//      F_{ac} = -Σ_{kld} w_{kcld} t_{kl}^{ad}    →   restricted to (k,l)=(i,j)
//      val += Lac[a,c] · t2[i,j,c,b]            (eri_stored.cu line 7129)
//
//  PySCF reference (H2O/cc-pVDZ Cartesian, T1=0, T2=MP2) verifies a
//  machine-precision match between the (k=l=0) restriction of canonical
//  F_ac and the DLPNO formula above
//  (`c:/Users/yasuaki/Dropbox/AQUA/DLPNO_phase23_formulas.md` §6.2).
//
//  Phase 2.4.1 / 2.4.3 (hole F_eff[k,i] dressing, full l sum):
//  when `phase24` is non-null,
//      ΔF_{ki} = Σ_{l} Σ_{cd} T_pair^{(il)}[k,c,l,d] · Y_{il}^{cd}
//  is computed each iteration. The inter-pair Fock coupling then uses
//  F_eff[i,k] = F_LMO[i,k] + ΔF_{ki} instead of bare F_LMO. l=i is the
//  Phase 2.4.1 piece (uses diagonal pair (i,i)); l≠i is Phase 2.4.3
//  (uses cross-pair amplitudes; Y_{il} transposed when stored as (l,i)).
//  The k=i diagonal entry of ΔF_{ki} matches the Phase 2.3.3 ΔF_{ii} only
//  when l=i alone is summed (with the full l sum the diagonal acquires
//  additional cross-pair contributions, still absorbed into the −(ΔF_{ii}
//  + ΔF_{jj}) Y shift).
//
//  Phase 2.4.2 (full (k,l) sum particle dressing): when `phase24` is non-null,
//      ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
//      t_{kl,proj}^{ad} = (\bar S^{(ij,kl)} · Y_{kl} · \bar S^{(ij,kl),T})_{ad}
//  is pre-computed per pair before the per-pair residual loop. The full
//  sum subsumes Phase 2.3.2's (k,l)=(i,j) restriction. When `phase24` is
//  null, the loop falls back to the intra-pair (k,l)=(i,j) shortcut.
//
//  Remaining cross-pair contributions (l ≠ i, hole side) land in
//  Phase 2.4.3+. The 4-vir ladder W_abcd is Phase 2.5.
// ---------------------------------------------------------------------------
LMP2Status iterate_dlpno_ccsd_t2(
    const std::vector<PairSetup>& setups,
    std::vector<PairData>&        pairs,
    const std::vector<int>&       pair_lookup,
    const std::vector<real_t>&    F_LMO,
    const real_t*                 h_S,
    int nocc, int nao,
    int max_iter, real_t conv_tol,
    bool enable_dressing,
    int verbose, const char* round_tag,
    const Phase24Integrals* phase24,
    int num_gpus,
    bool user_explicit_n_gpus,
    const real_t* lmo_centroids)
{
    if (!enable_dressing) {
        return iterate_lmp2(
            setups, pairs, pair_lookup, F_LMO, h_S,
            nocc, nao, max_iter, conv_tol, verbose, round_tag, num_gpus,
            user_explicit_n_gpus);
    }

    LMP2Status st;
    Eigen::Map<const RowMatXd> Smat(h_S, nao, nao);
    const real_t kFLMOThresh = 1e-14;

    // ---- Fine-grained CPU profile (Phase D follow-up) ----
    // Accumulators across all iterations; reported at the end of iterate.
    using prof_clock = std::chrono::steady_clock;
    double dt_barS    = 0.0;  // one-time
    double dt_vmeta   = 0.0;  // one-time
    double dt_picache = 0.0;
    double dt_dFki    = 0.0;
    double dt_DFpair  = 0.0;
    double dt_resid   = 0.0;
    double dt_diis    = 0.0;
    // Step 6.8 diagnostics — break "other" into named sub-buckets.
    double dt_setup_pgpu     = 0.0;  // one-time PiCacheGpu ctor (barS H2D)
    double dt_setup_rgpu     = 0.0;  // one-time ResidGpu ctor (V_meta + V_oooo H2D)
    double dt_yold_copy      = 0.0;  // per-iter, pairs.Y -> Y_old snapshot
    double dt_compute_async  = 0.0;  // per-iter, rgpu->compute_async() launch
    double dt_iter_misc      = 0.0;  // per-iter, post-DIIS bookkeeping
    const auto t_iter_total_0 = prof_clock::now();

    std::vector<std::vector<real_t>> Y_old(pairs.size());

    // ---- Pre-computed bar_S^{(ij,kl)} cache ----
    //
    // bar_S^{(ij,kl)} = bar_Q^{(ij)T} · S_AO · bar_Q^{(kl)} appears in every
    // inter-pair coupling, oooo ladder, ph-ladder and particle-dressing
    // contraction. It depends only on the (iter-invariant) bar_Q matrices,
    // so we pre-compute all combinations once before the iter loop. Storage
    // is N_pair² × n_pno² doubles (~50 MB for H2O hexamer); scales as N⁴
    // with system size — refactor to a sparse / distance-cutoff structure
    // if memory becomes an issue.
    std::vector<std::vector<RowMatXd>> barS_cache(pairs.size());
    {
        const auto t0 = prof_clock::now();
        for (size_t i_ij = 0; i_ij < pairs.size(); ++i_ij) {
            barS_cache[i_ij].resize(pairs.size());
        }
        // dynamic(4): per-i_ij work is O(N_pair × n_ij² × nao); empty pairs
        // (n_ij=0) `continue` fast. Static partitioning gives ~3× imbalance
        // due to clumped empties; dynamic levels it.
        #pragma omp parallel for schedule(dynamic, 4)
        for (long long i_ij = 0; i_ij < static_cast<long long>(pairs.size());
             ++i_ij)
        {
            const int n_ij = pairs[i_ij].n_pno;
            if (n_ij == 0) continue;
            Eigen::Map<const RowMatXd> bQ_ij(
                pairs[i_ij].bar_Q.data(), nao, n_ij);
            const RowMatXd bQ_ij_T_S = bQ_ij.transpose() * Smat;  // n_ij × nao
            for (size_t i_kl = 0; i_kl < pairs.size(); ++i_kl) {
                const int n_kl = pairs[i_kl].n_pno;
                if (n_kl == 0) continue;
                Eigen::Map<const RowMatXd> bQ_kl(
                    pairs[i_kl].bar_Q.data(), nao, n_kl);
                barS_cache[i_ij][i_kl].noalias() = bQ_ij_T_S * bQ_kl;
            }
        }
        dt_barS += std::chrono::duration<double>(prof_clock::now() - t0).count();
    }

    // ---- F_eff dressing buffers (Phase 2.3.x + 2.4.x) ----
    //
    // Hole dressing (nocc × nocc matrix, k row × i column):
    //   ΔF_{ki}^{l=i} = Σ_{cd} T_pair^{(ii)}[k, c, i, d] · Y_{ii}^{cd}
    //   diagonal k=i  → Phase 2.3.3 shift  −(ΔF_{ii}+ΔF_{jj}) Y^{(ij)}
    //   off-diag k≠i  → Phase 2.4.1 dressed F_LMO[i,k] in inter-pair coupling.
    //
    // Particle dressing (per pair (i,j), n_pno × n_pno matrix):
    //   ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
    //   t_{kl,proj}^{ad} = (\bar S^{(ij,kl)} · Y_{kl} · \bar S^{(ij,kl),T})_{ad}
    //   Phase 2.3.2  = (k,l)=(i,j) restriction of this sum.
    //   Phase 2.4.2  = full (k,l) sum.
    //
    // When `phase24` is null both fall back to Phase 2.3.x only (no cross
    // pair contributions); the diagonal hole dressing is then computed
    // from local L^{(ii)} as a special case.
    std::vector<real_t> dF_ki(static_cast<size_t>(nocc) * nocc, 0.0);
    std::vector<RowMatXd> DF_per_pair(pairs.size());

    // ---- DIIS (Phase 2.7) — flat amplitude / error vectors ----
    // Layout: each pair contributes its n_pno² Y entries, padded by the per-
    // pair offset table. Error vector is the per-pair Jacobi update
    // ΔY_{ij} = -R/D collected per iteration.
    //
    // max_hist=6 (was 8 in Phase 2.7): on large systems where the residual
    // norm shrinks across many orders of magnitude, the oldest 2 entries in
    // an 8-deep window become near-orthogonal to the latest update direction
    // and contribute mostly to B's null space. A shorter window keeps DIIS
    // working on a more locally-coherent subspace.
    DIIS diis(6, 2);
    // Phase 2.7b: track prev_r_max to detect convergence pathology. On large
    // saturated systems (e.g. cholesterol cc-pVDZ, 5886 pairs) DIIS can stall
    // around 1e-7 with near-parallel error vectors; the B matrix becomes
    // numerically singular, extrapolation either throws or produces a
    // non-improving step. Two reset triggers:
    //   (1) rebound: r_max increases by >5% — a clear sign of trouble.
    //   (2) stagnation: 4 consecutive iters with factor > 0.92 — DIIS has
    //       degraded to plain-Jacobi rate and history is poisoning the
    //       subspace; refresh fixes the asymptotic creep.
    real_t prev_r_max = std::numeric_limits<real_t>::max();
    int n_stalled = 0;
    constexpr real_t kStallFactor = 0.92;
    constexpr int    kStallCount  = 4;
    std::vector<size_t> y_offset(pairs.size() + 1, 0);
    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const int n_p = pairs[idx].n_pno;
        y_offset[idx + 1] = y_offset[idx] + static_cast<size_t>(n_p) * n_p;
    }
    const size_t flat_size = y_offset.back();
    std::vector<double> y_flat(flat_size, 0.0);
    std::vector<double> e_flat(flat_size, 0.0);

    // ---- pi_cache: per-iter projections bar_S · Y_old · bar_S^T ----
    // pi_cache[i_ij][i_kl] = barS_cache[i_ij][i_kl] · Y_canon · barS_cache[..]^T
    //   where Y_canon = pairs[i_kl].Y (stored in canonical (skl.i, skl.j) order).
    // Result shape: n_ij × n_ij.
    //
    // The projection is reused MANY times per iter (DFpair, inter-pair Fock,
    // oooo ladder, ph-ladder via build_pi), each time with a 'transp' flag
    // depending on which LMO plays the "first" role at the call site.
    // Algebraic fact: barS · Y_canon^T · barS^T = (barS · Y_canon · barS^T)^T,
    // so callers that need the swapped orientation simply transpose this cache
    // entry — no second build needed.
    std::vector<std::vector<RowMatXd>> pi_cache(pairs.size());
    {
        for (size_t i_ij = 0; i_ij < pairs.size(); ++i_ij) {
            pi_cache[i_ij].resize(pairs.size());
        }
    }

    // Step 6.0 + 6.1 — GPU port for pi_cache + pi_T_stack build (CPU
    // fallback when GPU memory is insufficient). barS, pair_lookup,
    // setups[idx].i, and per-pair n_pno_per_pair are uploaded once on
    // construction; per-iter rebuild_with_stack() does H2D Y → strided
    // batched DGEMM (Step 6.0) → custom pack kernel (Step 6.1) → D2H both.
    std::vector<int> n_pno_per_pair(pairs.size(), 0);
    int max_n_pno = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        n_pno_per_pair[i] = pairs[i].n_pno;
        if (pairs[i].n_pno > max_n_pno) max_n_pno = pairs[i].n_pno;
    }
    std::vector<int> setup_i_per_pair(pairs.size(), 0);
    for (size_t i = 0; i < pairs.size(); ++i) {
        setup_i_per_pair[i] = setups[i].i;
    }
    // Phase 2 (CCSD sparse barS) — setups[].j (for the pi_T_stack scatter) and
    // the per-output-pair coupling list. Every CCSD term contracts
    // pi_T_stack[i_ij][kl] = barS[i_ij][idx_kl]·Y·barS^T, so dropping a small
    // barS block removes its contribution from ALL terms (inter-Fock, oooo,
    // DFpair). Coupling = inter-Fock seed {(k,j),(i,l)} (ResidGpu reads these
    // via pair_lookup, must always be present) ∪ {idx_kl whose ‖barS‖_F > thr}.
    // thr=0 keeps every active (overlapping) pair ⇒ bit-exact; thr>0 screens
    // (ΔE<1e-4 calibrated). Opt-in via GANSU_DLPNO_CCSD_BARS_SPARSE.
    std::vector<int> setup_j_per_pair(pairs.size(), 0);
    for (size_t i = 0; i < pairs.size(); ++i) setup_j_per_pair[i] = setups[i].j;

    const bool ccsd_bars_sparse = []() {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_BARS_SPARSE");
        return e && e[0] == '1';
    }();
    // Distance-based coupling screen (Bohr). barS-norm proved a poor
    // discriminator (PTCDA: ΔE~1e-14 yet no coupling shrink at thr≤1e-4 — the
    // surviving active pairs' inter-pair barS is dense in norm). Pair-pair
    // CENTROID distance is the sharper, ORCA-like criterion: cut (i_ij,i_kl)
    // when the two pairs are spatially far. cutoff<=0 (or no centroids) ⇒ keep
    // all active ⇒ bit-exact (validation). Energy-contribution decay is
    // captured by the cutoff (calibrated to ΔE<1e-4).
    const real_t ccsd_bars_dist = []() {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_BARS_DIST");
        return e ? std::atof(e) : 0.0;
    }();
    const bool ccsd_dist_screen =
        ccsd_bars_sparse && lmo_centroids != nullptr && ccsd_bars_dist > 0.0;
    const real_t dist2_cut = ccsd_bars_dist * ccsd_bars_dist;
    auto pair_centroid = [&](int idx, real_t& cx, real_t& cy, real_t& cz) {
        const int i = setups[idx].i, j = setups[idx].j;
        cx = 0.5 * (lmo_centroids[3*i+0] + lmo_centroids[3*j+0]);
        cy = 0.5 * (lmo_centroids[3*i+1] + lmo_centroids[3*j+1]);
        cz = 0.5 * (lmo_centroids[3*i+2] + lmo_centroids[3*j+2]);
    };
    std::vector<std::vector<int>> coupling_ccsd_per_pair;
    if (ccsd_bars_sparse) {
        coupling_ccsd_per_pair.resize(pairs.size());
        #pragma omp parallel for schedule(dynamic, 4)
        for (long long idx = 0; idx < static_cast<long long>(pairs.size());
             ++idx)
        {
            if (pairs[idx].n_pno == 0) continue;
            const int i = setups[idx].i, j = setups[idx].j;
            std::vector<int>& nb = coupling_ccsd_per_pair[idx];
            for (int k = 0; k < nocc; ++k) {           // inter-Fock (k, j)
                if (k == i) continue;
                const int idx_kj = pair_lookup[static_cast<size_t>(k) * nocc + j];
                if (idx_kj >= 0 && pairs[idx_kj].n_pno > 0) nb.push_back(idx_kj);
            }
            for (int l = 0; l < nocc; ++l) {           // inter-Fock (i, l)
                if (l == j) continue;
                const int idx_il = pair_lookup[static_cast<size_t>(i) * nocc + l];
                if (idx_il >= 0 && pairs[idx_il].n_pno > 0) nb.push_back(idx_il);
            }
            // Stage D: the j-side ResidGpu slices read (j,l) [slot_jrow] and
            // (k,i) [slot_icol]. These are structural (every l / every k), the
            // j-side mirror of the i-side (k,j)/(i,l) above. They must be
            // force-seeded so the distance screen below cannot drop them —
            // otherwise the j-side ph-ladder would be screened while the i-side
            // is not (asymmetric, wrong). No effect when screen is off (all
            // active pairs already included). All l / all k (no self-skip) so
            // every slot_jrow[l]/slot_icol[k] the j-side slices index exists.
            for (int l = 0; l < nocc; ++l) {           // j-side slice (j, l)
                const int idx_jl = pair_lookup[static_cast<size_t>(j) * nocc + l];
                if (idx_jl >= 0 && pairs[idx_jl].n_pno > 0) nb.push_back(idx_jl);
            }
            for (int k = 0; k < nocc; ++k) {           // j-side slice (k, i)
                const int idx_ki = pair_lookup[static_cast<size_t>(k) * nocc + i];
                if (idx_ki >= 0 && pairs[idx_ki].n_pno > 0) nb.push_back(idx_ki);
            }
            // distance screen over all active pairs (or keep-all when off).
            real_t cx, cy, cz;
            if (ccsd_dist_screen) pair_centroid(static_cast<int>(idx), cx, cy, cz);
            for (size_t i_kl = 0; i_kl < pairs.size(); ++i_kl) {
                if (pairs[i_kl].n_pno == 0) continue;
                const RowMatXd& bs = barS_cache[idx][i_kl];
                if (bs.rows() == 0 || bs.cols() == 0) continue;  // no overlap ⇒ 0 anyway
                if (ccsd_dist_screen) {
                    real_t kx, ky, kz;
                    pair_centroid(static_cast<int>(i_kl), kx, ky, kz);
                    const real_t dx = cx - kx, dy = cy - ky, dz = cz - kz;
                    if (dx*dx + dy*dy + dz*dz > dist2_cut) continue;  // too far ⇒ drop
                }
                nb.push_back(static_cast<int>(i_kl));
            }
            std::sort(nb.begin(), nb.end());
            nb.erase(std::unique(nb.begin(), nb.end()), nb.end());
        }
    }

    // ---- Multi-GPU pair-slab partition (n_pno²-weighted greedy). ----
    // Each device handles its own slab of pair indices. Single-GPU
    // (num_gpus<=1) creates one instance covering the full range,
    // which preserves the previous behavior bit-exactly.
    //
    // Auto-fallback: multi-GPU dispatch overhead (sync cudaMemcpy +
    // cudaEventSynchronize + OMP team coordination, ~400 ms/iter for 8
    // GPUs on A100/H200) only pays off when per-iter GPU compute is
    // large enough to amortise it. Empirically (hexamer cc-pVDZ, 92 iter
    // benchmark + nsys 2026-05-13) the post-oooo-shared-mem-refactor
    // kernel work scales as N_pair × n_pno² × nocc² flops, and the
    // crossover happens around ~2×10⁸ flops per GPU per iter. Below that,
    // 1-GPU outperforms 8-GPU by 2-3×.
#ifdef GANSU_CPU_ONLY
    const int n_gpus = 1;
#else
    int n_gpus = (num_gpus < 1) ? 1 : num_gpus;
    if (n_gpus > 1) {
        long long work_estimate = 0;
        for (size_t i = 0; i < pairs.size(); ++i) {
            const long long n = pairs[i].n_pno;
            work_estimate += n * n;
        }
        work_estimate *= static_cast<long long>(nocc) * nocc;
        // Auto-fallback heuristic: skip when the user explicitly specified
        // `--num_gpus N > 0` (Option C, 2026-05-19). The historical 1e11
        // threshold predates Step Z (GPU-resident pi_pad) + S11 Phase 2
        // (packed cuBLAS + ResidGpu activate); for cholesterol-class systems
        // multi-GPU is now strictly required to activate ResidGpu, so
        // honouring user intent is the right default. The env var
        // GANSU_DLPNO_MIN_WORK_PER_GPU still lets users force the legacy
        // heuristic on `--num_gpus -1` (auto) runs.
        const long long kMinWorkPerGpu = get_dlpno_min_work_per_gpu();
        const long long per_gpu = work_estimate / n_gpus;
        const bool fall_back =
            !user_explicit_n_gpus && (per_gpu < kMinWorkPerGpu);
        if (fall_back) {
            if (verbose >= 1) {
                std::cout << "[DLPNO-ITER-PROF] " << round_tag
                          << " multi-GPU auto-fallback to n_gpus=1"
                          << " (work=" << work_estimate
                          << " per-GPU=" << per_gpu
                          << " < threshold=" << kMinWorkPerGpu << ")"
                          << std::endl;
            }
            n_gpus = 1;
        } else {
            MultiGpuManager::instance().initialize(n_gpus);
        }
    }
#endif
    const int N_pair = static_cast<int>(pairs.size());
    std::vector<int> slab_starts(n_gpus + 1, 0);
    {
        long long total_w = 0;
        std::vector<long long> w(N_pair, 0);
        // Partition weight. Default = n_pno² (residual flop balance). For the
        // CCSD sparse path the picache MEMORY (d_pi_needed + d_barS_csr =
        // 2·coupling_count·max_n², plus the sparse d_pi_T_stack ≈ n²·2·count)
        // is what OOMs on the heaviest slab, and a pure flop balance piles the
        // high-coupling pairs onto dev 0. So when the coupling list is known,
        // weight by coupling_count·(max_n² + n²) to balance picache memory
        // across the GPUs (correctness is unaffected — only which device owns
        // which pair changes). Falls back to n_pno² when no coupling list.
        // Picache ragged memory (d_pi_needed + d_barS_csr ≈ 2·coupling_count·
        // max_n²) drives the per-device OOM in the sparse path; weight by the
        // coupling count so it balances across GPUs. IMPORTANT: this also tends
        // to place the LOW-coupling (= weak, small-n_pno) pairs on the LAST
        // device, which matters because if a device's ResidGpu can't fit (large
        // systems) its slab's ph-ladder falls back to the CPU path — and that
        // fallback drops the ph-ladder contribution in sparse mode, so it must
        // land on the least-important (weak) pairs. (A weight that includes
        // n²·nocc² balances ResidGpu instead but concentrates STRONG pairs on
        // the last device → large error if it falls back — see project notes.)
        // Falls back to n_pno² when no coupling list.
        const bool mem_balance =
            ccsd_bars_sparse
            && coupling_ccsd_per_pair.size() == static_cast<size_t>(N_pair);
        const long long max_n2 =
            static_cast<long long>(max_n_pno) * static_cast<long long>(max_n_pno);
        for (int i = 0; i < N_pair; ++i) {
            const long long n = pairs[i].n_pno;
            if (mem_balance) {
                const long long cnt =
                    static_cast<long long>(coupling_ccsd_per_pair[i].size());
                w[i] = cnt * (max_n2 + n * n);
            } else {
                w[i] = n * n;
            }
            total_w += w[i];
        }
        long long target = (total_w + n_gpus - 1) / n_gpus;
        long long acc = 0;
        int g = 0;
        for (int i = 0; i < N_pair && g < n_gpus - 1; ++i) {
            acc += w[i];
            if (acc >= target * (g + 1)) {
                slab_starts[g + 1] = i + 1;
                ++g;
            }
        }
        slab_starts[n_gpus] = N_pair;
        for (int d = 1; d < n_gpus; ++d) {
            if (slab_starts[d] < slab_starts[d - 1])
                slab_starts[d] = slab_starts[d - 1];
        }
    }

    // Stage D debug — show how the picache pairs got partitioned across GPUs.
    {
        std::printf("[DLPNO-slab] num_gpus(param)=%d n_gpus(effective)=%d "
                    "N_pair=%d slab_starts=[", num_gpus, n_gpus, N_pair);
        for (int d = 0; d <= n_gpus; ++d)
            std::printf("%d%s", slab_starts[d], d < n_gpus ? "," : "");
        std::printf("]\n");
        std::fflush(stdout);
    }

    // Construct N_gpus PiCacheGpu instances (one per device).
    const auto t_setup_pgpu_0 = prof_clock::now();
    std::vector<std::unique_ptr<PiCacheGpu>> pgpus(n_gpus);
    if (n_gpus == 1) {
        pgpus[0] = std::make_unique<PiCacheGpu>(
            barS_cache, n_pno_per_pair, max_n_pno,
            &pair_lookup, &setup_i_per_pair, nocc,
            0, N_pair, 0, &setup_j_per_pair);
    } else {
#ifndef GANSU_CPU_ONLY
        // Step S8 (2026-05-17): parallel construction across devices.
        // Cholesterol cc-pVDZ benchmark (env GANSU_DLPNO_MIN_WORK_PER_GPU=0)
        // measured setup_pgpu = 75.7 s for 8 sequential constructors,
        // dwarfing the ~91 s picache win from multi-GPU dispatch. Each
        // constructor only touches its own device (DeviceGuard) and writes
        // per-instance members; barS_cache, n_pno_per_pair, pair_lookup,
        // and setup_i_per_pair are read-only inputs. Expected: 75 → ~10 s
        // (concurrent barS H2D + buffer alloc across 8 H200).
        #pragma omp parallel num_threads(n_gpus)
        {
#ifdef _OPENMP
            const int d = omp_get_thread_num();
#else
            const int d = 0;
#endif
            MultiGpuManager::DeviceGuard guard(d);
            pgpus[d] = std::make_unique<PiCacheGpu>(
                barS_cache, n_pno_per_pair, max_n_pno,
                &pair_lookup, &setup_i_per_pair, nocc,
                slab_starts[d], slab_starts[d + 1], d, &setup_j_per_pair);
        }
#else
        pgpus[0] = std::make_unique<PiCacheGpu>(
            barS_cache, n_pno_per_pair, max_n_pno,
            &pair_lookup, &setup_i_per_pair, nocc,
            0, N_pair, 0, &setup_j_per_pair);
#endif
    }
    dt_setup_pgpu += std::chrono::duration<double>(
        prof_clock::now() - t_setup_pgpu_0).count();

    // (D3a's pre-build of the sparse kl-slot machinery before the ResidGpu
    // constructors was reverted: it allocated the ragged d_pi_needed/d_barS_csr
    // early, shrinking the ResidGpu memory budget and forcing MORE devices into
    // the CPU fallback. The kl-slot list is built at its original point (before
    // the DFpair upload). D3a's V_stacked_oooo sparsify — which relied on
    // host_n_slots() being ready at ResidGpu-ctor time — is therefore dormant
    // unless that pre-build is restored; option (a) keeps the ph-ladder exact
    // via the corrected CPU fallback instead.)

    // Step 6.2 — GPU port for ph-ladder R contributions. One ResidGpu per
    // device, borrowing the per-device pgpu and uploading iter-invariant
    // V_meta_T/TT, T_meta, and W_bare_ov{ov,vo}_{i,j} to that device.
    // Falls back to active()=false per device when memory is insufficient.
    const auto t_setup_rgpu_0 = prof_clock::now();
    std::vector<std::unique_ptr<ResidGpu>> rgpus(n_gpus);
    if (phase24 != nullptr && phase24->nocc == nocc) {
#ifndef GANSU_CPU_ONLY
        for (int d = 0; d < n_gpus; ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            if (pgpus[d] && pgpus[d]->stacked()) {
                rgpus[d] = std::make_unique<ResidGpu>(
                    *pgpus[d], setups, pairs, *phase24, F_LMO, nocc, max_n_pno);
            }
        }
#else
        if (pgpus[0] && pgpus[0]->stacked()) {
            rgpus[0] = std::make_unique<ResidGpu>(
                *pgpus[0], setups, pairs, *phase24, F_LMO, nocc, max_n_pno);
        }
#endif
    }
    bool any_rgpu_active = false;
    for (int d = 0; d < n_gpus; ++d) {
        if (rgpus[d] && rgpus[d]->active()) { any_rgpu_active = true; break; }
    }
    std::vector<RowMatXd> R_ph_acc;
    if (any_rgpu_active) {
        R_ph_acc.assign(pairs.size(), RowMatXd());
    }
    dt_setup_rgpu += std::chrono::duration<double>(
        prof_clock::now() - t_setup_rgpu_0).count();

    // ---- pi_T_stack (Step 3): per-iter (k, l, d)-stacked projection ----
    // pi_T_stack[idx](a, (k·nocc + l)·n + d) = π_{k, l}^{oriented}[a, d]
    // where oriented means k plays the "first" role
    // (transpose pi_cache when setups[idx_kl].i != k).
    //
    // Used by:
    //   - DFpair:  ΔF[idx]  =  -(pi_T_stack[idx] · T_meta_dpair[idx])
    //                            single (n × nocc²·n × n) DGEMM/pair.
    //   - oooo lad: R[idx] +=  pi_T_stack[idx] · W_repeat (with W_repeat
    //                            block-diagonal in d, scaled by W_kl_eff[k, l]).
    std::vector<RowMatXd> pi_T_stack(pairs.size());

    // ---- V_meta / T_meta caches (Phase D Step 2) ----
    // Iter-invariant block-stacked layouts of phase24->V_ovov_pair[idx] and
    // phase24->T_pair[idx], used to batch the ph-ladder build_W_dressed
    // contractions into 3 medium DGEMMs per pair (instead of 30 × 3 = 90 small
    // DGEMMs in the original l-loop).
    //
    // Layout (each shape: nocc·n × nocc·n, row-major):
    //   V_meta_T [idx](l·n + d, k·n + c) = V_lk[d, c] = (ld|kc)
    //   V_meta_TT[idx](l·n + d, k·n + c) = V_lk[c, d]                   (block-T)
    //   T_meta   [idx](l·n + d, k·n + c) = T_kl[c, d]                   (T_pair, block-T)
    //
    // Built once (per iterate call) since V_ovov_pair and T_pair don't change
    // during the CCSD iter.
    std::vector<RowMatXd> V_meta_T(pairs.size());
    std::vector<RowMatXd> V_meta_TT(pairs.size());
    std::vector<RowMatXd> T_meta(pairs.size());
    // Step 3 caches (oooo ladder + DFpair batching):
    //   V_stacked_oooo[idx](k·nocc + l, a·n + b) = V_lk[a, b]
    //     (one row per (k, l), columns flatten the n × n V block in row-major)
    //   T_meta_dpair  [idx]((k·nocc + l)·n + d, c) = T_kl[c, d]
    //     (rows index (k, l, d), columns the c output index)
    std::vector<RowMatXd> V_stacked_oooo(pairs.size());
    std::vector<RowMatXd> T_meta_dpair(pairs.size());
    if (phase24 != nullptr && phase24->nocc == nocc) {
        const auto t_vm0 = prof_clock::now();
        // dynamic(4): per-pair V_meta / T_meta construction scales with
        // n_pno²; empty pairs skip. Same imbalance pattern as resid loop.
        #pragma omp parallel for schedule(dynamic, 4)
        for (long long idx = 0; idx < static_cast<long long>(pairs.size());
             ++idx)
        {
            const int n = pairs[idx].n_pno;
            if (n == 0) continue;
            if (idx >= static_cast<long long>(phase24->V_ovov_pair.size())
                || phase24->V_ovov_pair[idx].empty()
                || phase24->T_pair[idx].empty()) continue;
            const int nn = nocc * n;
            V_meta_T [idx].setZero(nn, nn);
            V_meta_TT[idx].setZero(nn, nn);
            T_meta   [idx].setZero(nn, nn);
            V_stacked_oooo[idx].setZero(
                static_cast<size_t>(nocc) * nocc, static_cast<size_t>(n) * n);
            T_meta_dpair  [idx].setZero(
                static_cast<size_t>(nocc) * nocc * n, n);
            const real_t* V_ov = phase24->V_ovov_pair[idx].data();
            const real_t* T_p  = phase24->T_pair[idx].data();
            for (int l = 0; l < nocc; ++l) {
                for (int k = 0; k < nocc; ++k) {
                    const real_t* V_lk =
                        V_ov + (static_cast<size_t>(l) * nocc + k)
                             * static_cast<size_t>(n) * n;
                    const real_t* T_kl =
                        T_p  + (static_cast<size_t>(k) * nocc + l)
                             * static_cast<size_t>(n) * n;
                    for (int d = 0; d < n; ++d) {
                        for (int c = 0; c < n; ++c) {
                            V_meta_T [idx](l * n + d, k * n + c) =
                                V_lk[d * n + c];
                            V_meta_TT[idx](l * n + d, k * n + c) =
                                V_lk[c * n + d];
                            T_meta   [idx](l * n + d, k * n + c) =
                                T_kl[c * n + d];
                            // T_meta_dpair: ((k, l), d) outer × c inner
                            T_meta_dpair[idx](
                                (static_cast<size_t>(k) * nocc + l) * n + d, c) =
                                T_kl[c * n + d];
                        }
                    }
                    // V_stacked_oooo row (k*nocc + l) = V_lk row-major flat
                    std::memcpy(
                        V_stacked_oooo[idx].data()
                          + (static_cast<size_t>(k) * nocc + l)
                          * static_cast<size_t>(n) * n,
                        V_lk,
                        static_cast<size_t>(n) * n * sizeof(real_t));
                }
            }
        }
        dt_vmeta += std::chrono::duration<double>(
            prof_clock::now() - t_vm0).count();
    }

    // ---- Per-thread CPU resid stage timers ----
    // When ResidGpu's full memory budget doesn't fit (cholesterol-class:
    // ~509 GB > 141 GB on H200), constructor falls back to active_=false
    // and the per-pair residual loop runs the *full* CPU path: build_stack
    // + 4 W_block CPU DGEMM + 8 R contraction DGEMM + inter_Fock + oooo
    // + 4-virt W·Y + DF dressing. To attribute the ~190 s/cholesterol resid
    // budget to its CPU sub-stages without touching the OMP-parallel inner
    // loop, each thread accumulates into its own slot below; we sum + max
    // at the end of the iter loop.
    struct ResidCpuStage {
        double t_r_init        = 0.0;  // R = L + diag·Y_old + dF_sum·Y_old
        double t_inter_fock    = 0.0;  // Phase 2.4.1 i + j sweeps (CPU path)
        double t_df_dressing   = 0.0;  // DF · Y_old + Y_old · DF^T
        double t_w4virt        = 0.0;  // 4-virt W_flat·Y_flat DGEMV
        double t_oooo          = 0.0;  // V_stacked_oooo·y + pi_T_stack·W_repeat
        double t_phladder_cpu  = 0.0;  // build_stack + 4 W_block + 8 R contract
        double t_gpu_add       = 0.0;  // R += R_ph_acc[idx] (active path only)
        double t_jacobi_diis   = 0.0;  // Jacobi update + DIIS y_flat/e_flat pack
    };
#ifdef _OPENMP
    const int kMaxThreads_resid_prof = omp_get_max_threads();
#else
    const int kMaxThreads_resid_prof = 1;
#endif
    std::vector<ResidCpuStage> resid_prof(kMaxThreads_resid_prof);

    // ---- DFpair GPU port: upload iter-invariant T_meta_dpair once. ----
    // DFpair (per-pair pi_T_stack·T_meta_dpair) is the largest single CCSD-T2
    // cost (~33% of wall). pi_T_stack is already device-resident (rebuilt each
    // iter); T_meta_dpair is iter-invariant, so we upload its slab to each
    // device once here and replace the per-iter host GEMM with a per-pair
    // cublasDgemm in compute_dfpair below. Each pgpus[d] handles its slab; the
    // small [n×n] DF blocks are D2H'd back into DF_per_pair (consumed on CPU).
    // Default OFF (env opt-in); VALIDATE runs both paths and compares.
    bool dfpair_gpu_on = false;
    // Default ON (validated bit-exact on PTCDA: max|CPU-GPU|=4.9e-17, 99× faster;
    // upload OOM → automatic CPU fallback). Opt out with GANSU_DLPNO_DFPAIR_GPU=0.
    const char* dfpair_gpu_env_str = std::getenv("GANSU_DLPNO_DFPAIR_GPU");
    const bool dfpair_gpu_env =
        (dfpair_gpu_env_str == nullptr)
        || (std::string(dfpair_gpu_env_str) != "0");
    const bool dfpair_gpu_validate =
        std::getenv("GANSU_DLPNO_DFPAIR_GPU_VALIDATE") != nullptr;
#ifndef GANSU_CPU_ONLY
    // Stage D (D1b): when pi_T_stack is sparse, the upload below packs
    // T_meta_dpair into the kl-slot layout, which requires the kl-slot list to
    // already exist. That list is otherwise first built inside
    // rebuild_with_stack() (first iter, AFTER this one-time pre-loop upload), so
    // pre-build it here per device. No-op unless pitstack_sparse_; idempotent.
    if (ccsd_bars_sparse && n_gpus > 0) {
        // Serial over devices: setup_sparse_stacked_ is itself omp-parallel inside.
        for (int d = 0; d < n_gpus; ++d) {
            if (pgpus[d]) pgpus[d]->ensure_sparse_stacked(coupling_ccsd_per_pair);
        }
    }
    if (dfpair_gpu_env && phase24 != nullptr && phase24->nocc == nocc
        && n_gpus > 0) {
        bool all_ok = true;
        if (n_gpus > 1) {
            std::vector<char> ok(n_gpus, 0);
            #pragma omp parallel num_threads(n_gpus)
            {
#ifdef _OPENMP
                const int d = omp_get_thread_num();
#else
                const int d = 0;
#endif
                MultiGpuManager::DeviceGuard guard(d);
                ok[d] = (pgpus[d] && pgpus[d]->stacked()
                         && pgpus[d]->upload_T_meta_dpair(T_meta_dpair))
                        ? 1 : 0;
            }
            for (int d = 0; d < n_gpus; ++d) if (!ok[d]) all_ok = false;
        } else {
            all_ok = (pgpus[0] && pgpus[0]->stacked()
                      && pgpus[0]->upload_T_meta_dpair(T_meta_dpair));
        }
        dfpair_gpu_on = all_ok;
        std::cout << "  [DFpair-GPU] "
                  << (dfpair_gpu_on ? "ON" : "OFF (upload failed → CPU)")
                  << (dfpair_gpu_validate ? "  (VALIDATE: both paths)" : "")
                  << std::endl;
    }
#endif

    for (int iter = 0; iter < max_iter; ++iter) {
        {
            const auto t_yold_0 = prof_clock::now();
            for (size_t idx = 0; idx < pairs.size(); ++idx) {
                Y_old[idx] = pairs[idx].Y;
            }
            dt_yold_copy += std::chrono::duration<double>(
                prof_clock::now() - t_yold_0).count();
        }

        // ---- Build pi_cache + pi_T_stack from Y_old (re-built each iter).
        // Step 6.1: pgpu.rebuild_with_stack does both — GPU strided-batched
        // pi_cache build (Step 6.0), then a custom pack kernel that reads
        // pi_pad on-device, applies the (k, l)-oriented transpose flag, and
        // writes pi_T_stack[i_ij] in unpadded ragged layout. Falls back
        // internally to the CPU OMP middleCols loop if stacked-mode
        // buffers couldn't be allocated.
        //
        // Multi-GPU: each pgpus[d] writes its slab range of pi_cache rows
        // and pi_T_stack entries; per-device omp threads run in parallel.
        {
            const auto t_pi0 = prof_clock::now();
            // When ResidGpu is active on EVERY slab (all_rgpu_active), every
            // host-side pi_cache consumer in the resid loop below is gated
            // out: the inter-pair Fock sweeps, build_stack_for_I path, and
            // the oooo CPU sweep are all replaced by R_ph_acc[idx] which
            // every rgpus[d] guarantees to fill (compute_finalize covers its
            // entire slab, slabs partition [0, N_pair) exhaustively). The
            // d_pi_pad → host D2H (4-5 s/iter at cholesterol scale) is then
            // pure waste — pass skip_pi_cache_host=true to skip it. Note we
            // require ALL rgpus active rather than ANY, because a single
            // failed-allocation slab would leave R_ph_acc[idx] empty and the
            // per-idx CPU fallback (lines 1113/1137/else-branch below) would
            // read uninitialized pi_cache. pi_T_stack host is still produced
            // because DFpair always consumes it on CPU.
            // Per-device skip: a device whose ResidGpu is ACTIVE fills
            // R_ph_acc for its slab, so its host pi_cache / pi_T_stack are never
            // read (the CPU resid loop is gated out for those pairs) ⇒ skip the
            // expensive D2H + host build. A device whose ResidGpu is INACTIVE
            // (failed to fit) needs its slab's host pi_cache (CPU inter-Fock +
            // ph-ladder) and host pi_T_stack (CPU oooo) produced for the
            // per-idx CPU fallback. (Previously this was a single all_rgpu_active
            // flag; per-device is both more efficient AND correct — the inactive
            // slab's owner always produces its own pi_cache.)
            if (n_gpus > 1) {
#ifndef GANSU_CPU_ONLY
                #pragma omp parallel num_threads(n_gpus)
                {
#ifdef _OPENMP
                    const int d = omp_get_thread_num();
#else
                    const int d = 0;
#endif
                    MultiGpuManager::DeviceGuard guard(d);
                    const bool skip_d = (rgpus[d] && rgpus[d]->active());
                    pgpus[d]->rebuild_with_stack(
                        Y_old, pi_cache, pi_T_stack, skip_d,
                        ccsd_bars_sparse ? &coupling_ccsd_per_pair : nullptr);
                }
#endif
            } else {
                const bool skip_0 = (rgpus[0] && rgpus[0]->active());
                pgpus[0]->rebuild_with_stack(
                    Y_old, pi_cache, pi_T_stack, skip_0,
                    ccsd_bars_sparse ? &coupling_ccsd_per_pair : nullptr);
            }
            dt_picache += std::chrono::duration<double>(
                prof_clock::now() - t_pi0).count();
        }

        // ---- Refresh ΔF_{ki} (hole dressing). ----
        // Step 6.5: dF_ki must be ready BEFORE rgpu's inter-pair Fock kernels
        // launch (they read d_dF_ki). Compute it on CPU first, then hand
        // off to rgpu->compute_async(dF_ki).
        const auto t_dFki_0 = prof_clock::now();
        std::fill(dF_ki.begin(), dF_ki.end(), 0.0);
        if (phase24 != nullptr && phase24->nocc == nocc) {
            // Phase 2.4.1 (l=i) + Phase 2.4.3 (l≠i): full l sum,
            //   ΔF_{ki} = Σ_l Σ_{cd} T_pair^{(il)}[k,c,l,d] · Y_{il}^{cd}.
            // Phase A — parallelise over (k, i) outer pair (each output
            // dF_ki[k, i] is independent; Y_old, T_pair, pair_lookup are
            // read-only in this region).
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k = 0; k < nocc; ++k) {
                for (int i = 0; i < nocc; ++i) {
                    real_t s = 0.0;
                    for (int l = 0; l < nocc; ++l) {
                        const int idx_il = pair_lookup[i * nocc + l];
                        const PairSetup& sil = setups[idx_il];
                        const PairData&  pil = pairs[idx_il];
                        const int n_il = pil.n_pno;
                        if (n_il == 0) continue;

                        Eigen::Map<const RowMatXd> Y_stored(
                            Y_old[idx_il].data(), n_il, n_il);
                        const bool transp = (sil.i != i);

                        const real_t* T_il = phase24->T_pair[idx_il].data();
                        const size_t stride_kl =
                            static_cast<size_t>(n_il) * n_il;
                        Eigen::Map<const RowMatXd> T_kl(
                            T_il + static_cast<size_t>(k * nocc + l)
                                 * stride_kl,
                            n_il, n_il);
                        // Σ_{cd} T_kl(c, d) · Y_use(c, d) is the trace-like
                        // element-wise contraction. cwiseProduct().sum()
                        // dispatches to a vectorised inner-product.
                        if (transp) {
                            s += (T_kl.array() *
                                  Y_stored.transpose().array()).sum();
                        } else {
                            s += (T_kl.array() * Y_stored.array()).sum();
                        }
                    }
                    dF_ki[static_cast<size_t>(k) * nocc + i] = s;
                }
            }
        } else {
            // Phase 2.3.3 fallback: diagonal entries from local L^{(ii)}.
            for (int i = 0; i < nocc; ++i) {
                const int idx_ii = pair_lookup[i * nocc + i];
                const PairData&  p = pairs[idx_ii];
                const int n_ii = p.n_pno;
                if (n_ii == 0) continue;
                Eigen::Map<const RowMatXd> L(p.L.data(),          n_ii, n_ii);
                Eigen::Map<const RowMatXd> Y(Y_old[idx_ii].data(), n_ii, n_ii);
                real_t s = 0.0;
                for (int c = 0; c < n_ii; ++c)
                    for (int d = 0; d < n_ii; ++d)
                        s += (2.0 * L(c, d) - L(d, c)) * Y(c, d);
                dF_ki[static_cast<size_t>(i) * nocc + i] = s;
            }
        }
        dt_dFki += std::chrono::duration<double>(prof_clock::now() - t_dFki_0).count();

        // ---- Step 6.2/6.4/6.5: per-iter R contributions on GPU. ----
        // R_ph_acc[idx] (n × n) is added inside the resid OMP loop below
        // in lieu of the per-pair CPU build_stack/W_block code AND (with
        // Step 6.5) the per-pair inter-pair Fock i+j sweeps.
        //
        // The async variant launches all GPU work (slice + W_block + 8
        // contractions + 2 Fock kernels) on the default stream so DFpair
        // (the next CPU block) can run concurrently. compute_finalize()
        // syncs on the recorded event before the resid loop reads R_ph_acc.
        //
        // Multi-GPU: each rgpus[d] handles its slab; dF_ki is global and
        // is uploaded full on every device.
        {
            const auto t_ca_0 = prof_clock::now();
            if (any_rgpu_active) {
                if (n_gpus > 1) {
#ifndef GANSU_CPU_ONLY
                    #pragma omp parallel num_threads(n_gpus)
                    {
#ifdef _OPENMP
                        const int d = omp_get_thread_num();
#else
                        const int d = 0;
#endif
                        MultiGpuManager::DeviceGuard guard(d);
                        if (rgpus[d] && rgpus[d]->active()) {
                            rgpus[d]->compute_async(dF_ki);
                        }
                    }
#endif
                } else if (rgpus[0] && rgpus[0]->active()) {
                    rgpus[0]->compute_async(dF_ki);
                }
            }
            dt_compute_async += std::chrono::duration<double>(
                prof_clock::now() - t_ca_0).count();
        }

        // ---- Refresh ΔF^{(ij)} per-pair particle dressing matrix. ----
        // Step 3 batched form: the previous nocc² inner loop is replaced by a
        // single (n × nocc²·n × n) DGEMM per pair using pi_T_stack and
        // T_meta_dpair (both built upstream).
        //
        //   ΔF[idx][a, c] = -Σ_{(k, l), d} π_{k, l}^{oriented}[a, d] · T_kl[c, d]
        //                 = -(pi_T_stack[idx] · T_meta_dpair[idx])[a, c]
        const auto t_DFpair_0 = prof_clock::now();
        if (phase24 != nullptr && phase24->nocc == nocc) {
            // Host reference loop (single (n × nocc²·n × n) Eigen GEMM/pair).
            auto dfpair_cpu = [&](std::vector<RowMatXd>& DFout) {
                // dynamic(4): same workload-skew rationale as the residual loop.
                #pragma omp parallel for schedule(dynamic, 4)
                for (long long idx = 0;
                     idx < static_cast<long long>(pairs.size()); ++idx) {
                    const PairData&  pij = pairs[idx];
                    const int n_ij = pij.n_pno;
                    if (n_ij == 0) { DFout[idx].resize(0, 0); continue; }
                    if (idx >= static_cast<long long>(T_meta_dpair.size())
                        || T_meta_dpair[idx].size() == 0
                        || pi_T_stack[idx].size() == 0)
                    {
                        DFout[idx].setZero(n_ij, n_ij);
                        continue;
                    }
                    DFout[idx].setZero(n_ij, n_ij);
                    DFout[idx].noalias() -=
                        pi_T_stack[idx] * T_meta_dpair[idx];
                }
            };
            // GPU path: per-pair cublasDgemm in each device's slab.
            auto dfpair_gpu = [&](std::vector<RowMatXd>& DFout) {
#ifndef GANSU_CPU_ONLY
                if (n_gpus > 1) {
                    #pragma omp parallel num_threads(n_gpus)
                    {
#ifdef _OPENMP
                        const int d = omp_get_thread_num();
#else
                        const int d = 0;
#endif
                        MultiGpuManager::DeviceGuard guard(d);
                        pgpus[d]->compute_dfpair(DFout);
                    }
                } else {
                    pgpus[0]->compute_dfpair(DFout);
                }
#else
                (void)DFout;
#endif
            };

            if (dfpair_gpu_on && dfpair_gpu_validate) {
                dfpair_cpu(DF_per_pair);          // reference — used downstream
                std::vector<RowMatXd> DF_gpu(pairs.size());
                dfpair_gpu(DF_gpu);
                double maxdiff = 0.0; long long argmax = -1;
                for (long long idx = 0;
                     idx < static_cast<long long>(pairs.size()); ++idx) {
                    if (DF_per_pair[idx].size() == 0) continue;
                    if (DF_gpu[idx].rows() != DF_per_pair[idx].rows()
                        || DF_gpu[idx].cols() != DF_per_pair[idx].cols()) {
                        maxdiff = 1e99; argmax = idx; break;
                    }
                    const double diff =
                        (DF_per_pair[idx] - DF_gpu[idx]).cwiseAbs().maxCoeff();
                    if (diff > maxdiff) { maxdiff = diff; argmax = idx; }
                }
                if (iter == 0 || maxdiff > 1e-9)
                    std::cout << "  [DFpair-GPU-VALIDATE] iter=" << iter
                              << " max|CPU-GPU|=" << std::scientific << maxdiff
                              << std::defaultfloat << "  (pair " << argmax << ")"
                              << std::endl;
            } else if (dfpair_gpu_on) {
                dfpair_gpu(DF_per_pair);
            } else {
                dfpair_cpu(DF_per_pair);
            }
        }
        dt_DFpair += std::chrono::duration<double>(prof_clock::now() - t_DFpair_0).count();

        // Step 6.4: sync on rgpu's async D2H + unpack to R_ph_acc. After
        // this point R_ph_acc[idx] is ready for the resid OMP loop.
        //
        // Multi-GPU: each rgpus[d] fills its slab portion of R_ph_acc.
        if (any_rgpu_active) {
            if (n_gpus > 1) {
#ifndef GANSU_CPU_ONLY
                // Pre-size R_ph_acc to N_pair so peer-slab writes don't
                // accidentally resize during the parallel finalize.
                if (R_ph_acc.size() != pairs.size()) {
                    R_ph_acc.assign(pairs.size(), RowMatXd());
                }
                #pragma omp parallel num_threads(n_gpus)
                {
#ifdef _OPENMP
                    const int d = omp_get_thread_num();
#else
                    const int d = 0;
#endif
                    MultiGpuManager::DeviceGuard guard(d);
                    if (rgpus[d] && rgpus[d]->active()) {
                        rgpus[d]->compute_finalize(R_ph_acc);
                    }
                }
#endif
            } else if (rgpus[0] && rgpus[0]->active()) {
                rgpus[0]->compute_finalize(R_ph_acc);
            }
        }

        // Phase A — parallelise the per-pair residual + Jacobi update.
        // Each thread handles its own slab of pair indices: writes to
        // pairs[idx].Y and the y_flat / e_flat slabs at offset y_offset[idx]
        // are exclusive; Y_old, dF_ki, DF_per_pair, F_LMO, h_S are read-only.
        const auto t_resid_0 = prof_clock::now();
        real_t r_max = 0.0;
        // 2026-05-17 schedule change: cholesterol cc-pVDZ profiling at v0
        // (resid=188s, ResidGpu inactive due to ~509 GB > 141 GB H200 budget)
        // showed CPU utilisation = 32% (per-thread-max=97s vs thread-sum=3796s
        // ÷ 64 threads = 60s ideal). The per-pair workload scales as n_pno²,
        // ranging from n=0 (empty, skip via `continue`) up to n=26 (peak)
        // with avg=14.9 over strong pairs — a 40× workload spread on top of
        // the empty-pair-clustering produced by upstream sort order. Static
        // schedule pins thread 0 onto a clump of strong pairs while others
        // race through empties. `dynamic(4)` lets free threads grab work
        // (4-pair chunks: low atomic overhead vs imbalance risk), expected
        // to flatten the 3× max/avg per-stage thread time → ~2.85× resid
        // wall speedup at cholesterol with no algorithmic change.
        // Bit-exact: scheduling policy only; each thread writes its own
        // y_flat / e_flat slab at y_offset[idx] (no overlap), and
        // reduction(max:r_max) is order-invariant.
        #pragma omp parallel for schedule(dynamic, 4) reduction(max:r_max)
        for (long long idx = 0; idx < static_cast<long long>(pairs.size()); ++idx) {
            PairData&        pij = pairs[idx];
            const PairSetup& sij = setups[idx];
            const int n = pij.n_pno;
            if (n == 0) continue;

#ifdef _OPENMP
            const int tid_rp = omp_get_thread_num();
#else
            const int tid_rp = 0;
#endif
            ResidCpuStage& prof_slot = resid_prof[tid_rp];
            const auto t_rinit_0 = prof_clock::now();

            Eigen::Map<RowMatXd>      Y_ij(pij.Y.data(), n, n);
            Eigen::Map<const RowMatXd> L_ij(pij.L.data(), n, n);
            Eigen::Map<const RowMatXd> bQ_ij(pij.bar_Q.data(), nao, n);
            Eigen::Map<const RowMatXd> Y_old_ij(Y_old[idx].data(), n, n);

            const real_t shift = sij.F_ii + sij.F_jj;   // bare, used for denominator
            RowMatXd R(n, n);
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    R(a, b) = L_ij(a, b)
                            + (pij.Lambda[a] + pij.Lambda[b] - shift)
                              * Y_old_ij(a, b);

            // ---- Phase 2.3.3 intra-pair hole dressing on diagonal F_eff ----
            // Add -(ΔF_ii + ΔF_jj) · Y_old to R (separate from the bare
            // shift, so the Jacobi denominator below is still ε_a − ε_i).
            // ΔF_ii is the (k=i) diagonal of the dF_ki matrix.
            const real_t dF_sum =
                dF_ki[static_cast<size_t>(sij.i) * nocc + sij.i]
              + dF_ki[static_cast<size_t>(sij.j) * nocc + sij.j];
            if (dF_sum != 0.0) {
                R.noalias() -= dF_sum * RowMatXd(Y_old_ij);
            }
            // ---------------------------------------------------------
            prof_slot.t_r_init += std::chrono::duration<double>(
                prof_clock::now() - t_rinit_0).count();
            const auto t_inter_fock_0 = prof_clock::now();

            // Step 6.5: when ResidGpu is active, both inter-pair Fock i+j
            // sweeps live in R_ph_acc[idx] (added below alongside the
            // ph-ladder result). Skip the per-pair CPU sweeps.
            const bool skip_cpu_inter_pair_fock =
                (any_rgpu_active
                 && idx < static_cast<long long>(R_ph_acc.size())
                 && R_ph_acc[idx].rows() == n
                 && R_ph_acc[idx].cols() == n);

            // Inter-pair Fock coupling on i.
            // Phase 2.4.1: F_eff[i,k] = F_LMO[i,k] + ΔF[k,i] where ΔF uses
            // pair (i,i)'s amplitudes (l=i restriction of canonical F_ki).
            // dF_ki[k*nocc + i] = 0 when phase24 is null (Phase 2.3.x only).
            if (!skip_cpu_inter_pair_fock)
            for (int k = 0; k < nocc; ++k) {
                if (k == sij.i) continue;
                const real_t F_ik =
                    F_LMO[sij.i * nocc + k]
                  + dF_ki[static_cast<size_t>(k) * nocc + sij.i];
                if (std::fabs(F_ik) < kFLMOThresh) continue;

                const int idx_kj = pair_lookup[k * nocc + sij.j];
                const PairData&  pkj = pairs[idx_kj];
                const PairSetup& skj = setups[idx_kj];
                if (pkj.n_pno == 0) continue;

                const RowMatXd& pi_kj = pi_cache[idx][idx_kj];
                if (skj.i != k) {
                    R.noalias() -= F_ik * pi_kj.transpose();
                } else {
                    R.noalias() -= F_ik * pi_kj;
                }
            }

            // Inter-pair Fock coupling on j.
            // Phase 2.4.1: F_eff[l,j] = F_LMO[l,j] + ΔF[l,j] using pair
            // (j,j)'s amplitudes (l=j restriction).
            if (!skip_cpu_inter_pair_fock)
            for (int l = 0; l < nocc; ++l) {
                if (l == sij.j) continue;
                const real_t F_lj =
                    F_LMO[l * nocc + sij.j]
                  + dF_ki[static_cast<size_t>(l) * nocc + sij.j];
                if (std::fabs(F_lj) < kFLMOThresh) continue;

                const int idx_il = pair_lookup[sij.i * nocc + l];
                const PairData&  pil = pairs[idx_il];
                const PairSetup& sil = setups[idx_il];
                if (pil.n_pno == 0) continue;

                const RowMatXd& pi_il = pi_cache[idx][idx_il];
                if (sil.i != sij.i) {
                    R.noalias() -= F_lj * pi_il.transpose();
                } else {
                    R.noalias() -= F_lj * pi_il;
                }
            }

            prof_slot.t_inter_fock += std::chrono::duration<double>(
                prof_clock::now() - t_inter_fock_0).count();
            const auto t_df_dressing_0 = prof_clock::now();

            // ---- Particle F_eff dressing (Phase 2.3.2 / 2.4.2) ----
            // R += ΔF^{(ij)} · Y_old + Y_old · ΔF^{(ij),T}.
            //
            // When phase24 is provided, ΔF^{(ij)} is the full (k,l) sum
            // pre-computed above. Otherwise fall back to Phase 2.3.2's
            // (k,l)=(i,j) intra-pair shortcut using L^{(ij)} directly.
            if (phase24 != nullptr && phase24->nocc == nocc) {
                const RowMatXd& DF = DF_per_pair[idx];
                if (DF.size() == n * n) {
                    R.noalias() += DF * RowMatXd(Y_old_ij);
                    R.noalias() += RowMatXd(Y_old_ij) * DF.transpose();
                }
            } else {
                // Phase 2.3.2 fallback: intra-pair only.
                const RowMatXd tilde_L = 2.0 * RowMatXd(L_ij)
                                       - RowMatXd(L_ij.transpose());
                const RowMatXd DF =
                    -(RowMatXd(Y_old_ij) * tilde_L.transpose());
                R.noalias() += DF * RowMatXd(Y_old_ij);
                R.noalias() += RowMatXd(Y_old_ij) * DF.transpose();
            }
            // ---------------------------------------------------------

            // ---- Phase 2.5 / 2.6 — ladder pieces (BARE Ws + DIIS stabilisation) ----
            //
            //   BARE W_abcd alone diverges; Phase 2.7 DIIS extrapolates the
            //   per-iteration Y vectors (residual-norm minimisation in an
            //   8-vector subspace) to absorb the imbalance left by the
            //   missing T2 dressing of W_akic. The full DLPNO-CCSD
            //   formulation has W_akic ⊃ ½ Σ_{ld}[…] T2 ring-diagram terms
            //   that mathematically balance W_abcd; DIIS substitutes
            //   numerical stability for that algebraic balance.
            //
            //   Re-enables 4-vir + oooo + particle-hole ladders as derived
            //   in the combined Phase 2.5+2.6 block below. Numerical
            //   accuracy versus canonical CCSD is approximate (BARE Ws
            //   miss the ring-diagram contribution); a follow-up commit
            //   will fit the W T2 dressing for canonical agreement.
            prof_slot.t_df_dressing += std::chrono::duration<double>(
                prof_clock::now() - t_df_dressing_0).count();

            if (phase24 != nullptr && phase24->nocc == nocc
                && idx < phase24->W_pair.size()
                && phase24->W_pair[idx].size() ==
                       static_cast<size_t>(n) * n * n * n)
            {
                const auto t_w4virt_0 = prof_clock::now();
                // (a) 4-virtual ladder  R[a,b] += Σ_cd W_abcd Y_old[c,d].
                //   Reshape W (n×n×n×n row-major) as a (n²)×(n²) matrix and
                //   Y_old, R as length-n² vectors, then a single BLAS DGEMV
                //   replaces the 4-loop.
                {
                    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic,
                        Eigen::Dynamic, Eigen::RowMajor>>
                        W_flat(phase24->W_pair[idx].data(),
                               n * n, n * n);
                    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
                        Y_flat(Y_old_ij.data(), n * n);
                    Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
                        R_flat(R.data(), n * n);
                    R_flat.noalias() += W_flat * Y_flat;
                }
                prof_slot.t_w4virt += std::chrono::duration<double>(
                    prof_clock::now() - t_w4virt_0).count();
                const auto t_oooo_0 = prof_clock::now();

                // (b) oooo ladder, batched (Step 3).
                //   W_klij^{eff} = (ki|lj) + Σ_{cd} (kc|ld) · Y_{ij,old}^{cd}.
                //   The trace dressing Σ_{c,d} V_lk[d, c] Y_old[c, d] for ALL
                //   (k, l) is one DGEMV  W_dress = V_stacked_oooo · y_flat
                //   where y_flat[a·n + b] = Y_old[b, a] (Y_old transposed flat).
                //
                //   Then  R += Σ_{kl} W_kl_eff · π_{kl}^{oriented}
                //          = pi_T_stack · W_repeat
                //   with W_repeat[(kl)·n + d, b] = W_kl_eff[kl] · δ_{db}
                //   (block-diagonal in d, scaled by W_kl_eff per (k, l) row block).
                //
                // Step 6.6: when ResidGpu is active, the oooo lad result is
                // already accumulated into R_ph_acc[idx] by the GPU pipeline.
                // Skip the per-pair CPU sweep.
                const bool skip_cpu_oooo = skip_cpu_inter_pair_fock;
                if (!skip_cpu_oooo
                    && idx < static_cast<long long>(V_stacked_oooo.size())
                    && V_stacked_oooo[idx].size() > 0
                    && pi_T_stack[idx].size() > 0)
                {
                    const real_t* W_oooo = phase24->W_oooo[idx].data();

                    // y_flat[a·n + b] = Y_old_ij[b, a]
                    Eigen::VectorXd y_flat(static_cast<size_t>(n) * n);
                    for (int a = 0; a < n; ++a)
                        for (int b = 0; b < n; ++b)
                            y_flat[a * n + b] = Y_old_ij(b, a);

                    // W_dress_flat = V_stacked_oooo · y_flat  (shape nocc²)
                    Eigen::VectorXd W_dress_flat =
                        V_stacked_oooo[idx] * y_flat;

                    // W_repeat: (nocc²·n × n) sparse block-diagonal.
                    RowMatXd W_repeat = RowMatXd::Zero(
                        static_cast<size_t>(nocc) * nocc * n, n);
                    for (int kl = 0; kl < nocc * nocc; ++kl) {
                        const real_t w =
                            W_oooo[kl] + W_dress_flat[kl];
                        if (std::fabs(w) < kFLMOThresh) continue;
                        for (int d = 0; d < n; ++d) {
                            W_repeat(static_cast<size_t>(kl) * n + d, d) = w;
                        }
                    }
                    R.noalias() += pi_T_stack[idx] * W_repeat;
                }

                // ---- (c) + (d) particle-hole ladders, batched form ----
                //
                // Original per-pair work was nocc² × O(small DGEMMs) of size
                // n × n. Replaced with O(medium) DGEMMs of size n × nocc·n by
                // stacking the per-l π projections and per-k pi/Y projections.
                //
                // Algebra (i-side; j-side is the (i↔j, a↔b)-canonical partner):
                //   W_block_i [a, k·n + c] = W_ovov_i[a, k, c]
                //       − ½ Σ_l (π_{i,l}^T · V_lk)[a, c]
                //       + ½ Σ_l (π_{i,l} · T_kl^T)[a, c]
                //   W_block_i2[a, k·n + c] = W_ovvo_i[a, k, c]
                //       − ½ Σ_l (π_{i,l}^T · V_lk^T)[a, c]
                //
                // Define
                //   pi_stack_T_i [l·n + d, a]     = π_{i,l}[d, a]
                //   pi_stack_N_i [a, l·n + d]     = π_{i,l}[a, d]
                // (vertical / horizontal stack of the same data).
                //
                // Then
                //   Σ_l π_{i,l}^T · V_lk  = (pi_stack_T_i^T · V_meta_T )_block_k
                //   Σ_l π_{i,l}^T · V_lk^T= (pi_stack_T_i^T · V_meta_TT)_block_k
                //   Σ_l π_{i,l}   · T_kl^T= (pi_stack_N_i   · T_meta   )_block_k
                //
                // and the outer ph-ladder reduces to 4 medium DGEMMs per side
                // by stacking pi_kj (i-side) or pi_ki (j-side) similarly:
                //   PI_kj_stack [k·n + c, b] = π_{k,j}[c, b]
                //   PI_kj_TT    [a, k·n + c] = π_{k,j}[a, c]   (= horizontal)
                //
                //   Σ_k W_k_i · π_kj      = W_block_i  · PI_kj_stack
                //   Σ_k W_k_i2· π_kj      = W_block_i2 · PI_kj_stack
                //   Σ_k W_k_i · π_kj^T    = W_block_i  · PI_kj_TT^T
                //   Σ_k π_kj  · W_k_i2^T  = PI_kj_TT   · W_block_i2^T
                //
                // (j-side mirrors with PI_ki and W_block_j[2] ← I = sij.j.)
                prof_slot.t_oooo += std::chrono::duration<double>(
                    prof_clock::now() - t_oooo_0).count();

                // Step 6.2: when ResidGpu is active, R_ph_acc[idx] already
                // holds the i-side + j-side ph-ladder contributions. Skip
                // the CPU build_stack/W_block path entirely.
                const auto t_phl_0 = prof_clock::now();
                if (any_rgpu_active
                    && idx < static_cast<long long>(R_ph_acc.size())
                    && R_ph_acc[idx].rows() == n
                    && R_ph_acc[idx].cols() == n)
                {
                    R.noalias() += R_ph_acc[idx];
                    prof_slot.t_gpu_add += std::chrono::duration<double>(
                        prof_clock::now() - t_phl_0).count();
                }
                else if (idx < static_cast<long long>(V_meta_T.size())
                    && V_meta_T[idx].size() > 0)
                {
                    const int nn = nocc * n;

                    // Build pi-stacks for I = sij.i (i-side dressing source)
                    // and I = sij.j (j-side dressing source).
                    auto build_stack_for_I = [&](int I_lmo,
                                                 RowMatXd& stack_T,
                                                 RowMatXd& stack_N)
                    {
                        stack_T.setZero(nn, n);
                        stack_N.setZero(n, nn);
                        for (int l = 0; l < nocc; ++l) {
                            const int idx_il =
                                pair_lookup[I_lmo * nocc + l];
                            const PairData&  pil = pairs[idx_il];
                            if (pil.n_pno == 0) continue;
                            const PairSetup& sil = setups[idx_il];
                            const RowMatXd& pi_canon = pi_cache[idx][idx_il];
                            if (sil.i != I_lmo) {
                                stack_T.middleRows(l * n, n) =
                                    pi_canon.transpose();
                                stack_N.middleCols(l * n, n) =
                                    pi_canon.transpose();
                            } else {
                                stack_T.middleRows(l * n, n) = pi_canon;
                                stack_N.middleCols(l * n, n) = pi_canon;
                            }
                        }
                    };

                    // Build outer pi-stacks for ph-ladder term contractions
                    // (PI_xy_stack indexed by k → π_{k, sij.j} or π_{k, sij.i}).
                    auto build_outer_stack = [&](int J_lmo,
                                                 RowMatXd& stack,
                                                 RowMatXd& stack_TT)
                    {
                        stack.setZero(nn, n);
                        stack_TT.setZero(n, nn);
                        for (int k = 0; k < nocc; ++k) {
                            const int idx_kJ =
                                pair_lookup[k * nocc + J_lmo];
                            const PairData&  pkJ = pairs[idx_kJ];
                            if (pkJ.n_pno == 0) continue;
                            const PairSetup& skJ = setups[idx_kJ];
                            const RowMatXd& pi_canon = pi_cache[idx][idx_kJ];
                            if (skJ.i != k) {
                                stack.middleRows(k * n, n) =
                                    pi_canon.transpose();
                                stack_TT.middleCols(k * n, n) =
                                    pi_canon.transpose();
                            } else {
                                stack.middleRows(k * n, n) = pi_canon;
                                stack_TT.middleCols(k * n, n) = pi_canon;
                            }
                        }
                    };

                    auto add_bare_to_block = [&](const real_t* W_bare,
                                                 RowMatXd& W_block)
                    {
                        // W_bare layout: [a, k, c] flattened, with the same
                        // (a, k, c) → (a · nocc + k) · n + c indexing used in
                        // build_mo_eri / phase24 ovov/ovvo stores.
                        for (int k = 0; k < nocc; ++k) {
                            for (int a = 0; a < n; ++a) {
                                const real_t* row =
                                    W_bare
                                  + (static_cast<size_t>(a) * nocc + k) * n;
                                for (int c = 0; c < n; ++c) {
                                    W_block(a, k * n + c) += row[c];
                                }
                            }
                        }
                    };

                    // ----- i-side -----
                    {
                        RowMatXd pi_T_i, pi_N_i;
                        build_stack_for_I(sij.i, pi_T_i, pi_N_i);

                        RowMatXd W_block_i (n, nn);
                        RowMatXd W_block_i2(n, nn);
                        // Note (2026-05-18): the formula below is canonical
                        // RCCSD W_akic / W_akci T2 dressing in disguise, NOT
                        // BARE. Algebra:
                        //   pi_T_i^T[a, l·n+d]  = t_lI^{(ij)}[a, d]
                        //   pi_N_i  [a, l·n+d]  = t_Il^{(ij)}[a, d]
                        //   T_meta  [l·n+d, k·n+c]
                        //     = T_pair^{(ij)}[k,c,l,d]
                        //     = 2(ld|kc) − (lc|kd)
                        //   V_meta_T[l·n+d, k·n+c] = (ld|kc)
                        //   V_meta_TT[l·n+d, k·n+c] = (lc|kd)
                        // ⇒ −½ pi_T_i^T·V_meta_T  +  ½ pi_N_i·T_meta
                        //     = ½ (ld|kc) τ_Il − ½ (lc|kd) t_Il    (W_akic)
                        //   −½ pi_T_i^T·V_meta_TT
                        //     = −½ (lc|kd) t_lI                    (W_akci)
                        // matches PySCF cc_Wvoov / cc_Wvovo at T1=0.
                        // See project_dlpno_bare_w_algebra.md (2026-05-18
                        // audit correction): the legacy ad-hoc-looking form
                        // IS canonical; the gap-claim comment block above is
                        // wrong. The dressed_w flag below is retained as
                        // future-ready infrastructure but is currently a
                        // no-op (will be reused for any genuinely-new
                        // algebraic variant identified later).
                        W_block_i .noalias()  =
                            -0.5 * pi_T_i.transpose() * V_meta_T [idx];
                        W_block_i .noalias() +=
                             0.5 * pi_N_i             * T_meta   [idx];
                        W_block_i2.noalias()  =
                            -0.5 * pi_T_i.transpose() * V_meta_TT[idx];
                        add_bare_to_block(phase24->W_ovov_i[idx].data(),
                                          W_block_i);
                        add_bare_to_block(phase24->W_ovvo_i[idx].data(),
                                          W_block_i2);

                        RowMatXd PI_kj_stack, PI_kj_TT;
                        build_outer_stack(sij.j, PI_kj_stack, PI_kj_TT);

                        R.noalias() += 2.0 * (W_block_i  * PI_kj_stack);
                        R.noalias() -=        W_block_i2 * PI_kj_stack;
                        R.noalias() -=        W_block_i  * PI_kj_TT.transpose();
                        R.noalias() -=        PI_kj_TT   * W_block_i2.transpose();
                    }

                    // ----- j-side -----
                    {
                        RowMatXd pi_T_j, pi_N_j;
                        build_stack_for_I(sij.j, pi_T_j, pi_N_j);

                        RowMatXd W_block_j (n, nn);
                        RowMatXd W_block_j2(n, nn);
                        // See i-side note: formula is canonical W_akic /
                        // W_akci with I_lmo = sij.j (the i↔j symmetrisation
                        // partner of the i-side block).
                        W_block_j .noalias()  =
                            -0.5 * pi_T_j.transpose() * V_meta_T [idx];
                        W_block_j .noalias() +=
                             0.5 * pi_N_j             * T_meta   [idx];
                        W_block_j2.noalias()  =
                            -0.5 * pi_T_j.transpose() * V_meta_TT[idx];
                        add_bare_to_block(phase24->W_ovov_j[idx].data(),
                                          W_block_j);
                        add_bare_to_block(phase24->W_ovvo_j[idx].data(),
                                          W_block_j2);

                        RowMatXd PI_ki_stack, PI_ki_TT;
                        build_outer_stack(sij.i, PI_ki_stack, PI_ki_TT);

                        R.noalias() += 2.0 *
                            (PI_ki_stack.transpose() * W_block_j .transpose());
                        R.noalias() -=
                             PI_ki_stack.transpose() * W_block_j2.transpose();
                        R.noalias() -=
                             PI_ki_TT                * W_block_j .transpose();
                        R.noalias() -=
                             W_block_j2              * PI_ki_stack;
                    }
                    prof_slot.t_phladder_cpu += std::chrono::duration<double>(
                        prof_clock::now() - t_phl_0).count();
                }
            }
            // ---------------------------------------------------------

            const auto t_jacobi_0 = prof_clock::now();
            real_t r_max_pair = 0.0;
            const size_t off = y_offset[idx];
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b) {
                    const real_t denom =
                        pij.Lambda[a] + pij.Lambda[b] - shift;
                    const real_t r = R(a, b);
                    const real_t dY = -r / denom;
                    Y_ij(a, b) += dY;
                    r_max_pair = std::max(r_max_pair, std::fabs(r));
                    // DIIS: pack Y_new (= y) and ΔY (= e, error vector).
                    const size_t k = off + static_cast<size_t>(a) * n + b;
                    y_flat[k] = static_cast<double>(Y_ij(a, b));
                    e_flat[k] = static_cast<double>(dY);
                }
            r_max = std::max(r_max, r_max_pair);
            prof_slot.t_jacobi_diis += std::chrono::duration<double>(
                prof_clock::now() - t_jacobi_0).count();
        }
        dt_resid += std::chrono::duration<double>(prof_clock::now() - t_resid_0).count();

        // ---- DIIS extrapolation (Phase 2.7) ----
        // Use ΔY = -R/D as the DIIS error. Once enough history is built up,
        // the extrapolated Y is a residual-norm-minimising linear combination
        // of the recent iterates — robust against the non-monotonic damping
        // patterns produced by aggressive ladder pieces.
        //
        // Phase 2.7b: reset DIIS history on three triggers — (1) rebound
        // (r_max increases by >5%), (2) stagnation (factor > 0.92 for 4
        // consecutive iters → DIIS is stuck at plain-Jacobi rate), and
        // (3) singular B at extrapolation. Without these, on large saturated
        // systems the subspace becomes near-rank-deficient at ~1e-7 and the
        // remaining iterations creep at the un-accelerated rate (~0.92/iter).
        const auto t_diis_0 = prof_clock::now();
        if (flat_size > 0) {
            const bool rebound = (iter > 0) && (r_max > prev_r_max * 1.05);
            if (iter > 0 && r_max > prev_r_max * kStallFactor) {
                ++n_stalled;
            } else {
                n_stalled = 0;
            }
            const bool stagnation = (n_stalled >= kStallCount);
            if (rebound || stagnation) {
                if (verbose >= 2 && diis.history_size() > 0) {
                    const char* why = rebound ? "rebound" : "stagnation";
                    std::cout << "[DLPNO] DIIS reset on " << why
                              << " (max|R|: "
                              << std::scientific << std::setprecision(3)
                              << prev_r_max << " -> " << r_max
                              << ", history=" << diis.history_size();
                    if (stagnation) std::cout << ", stalled=" << n_stalled;
                    std::cout << ")" << std::endl;
                }
                diis.clear();
                n_stalled = 0;
            }
            diis.push(y_flat, e_flat);
            if (diis.can_extrapolate()) {
                try {
                    const auto y_ext = diis.extrapolate();
                    for (size_t idx = 0; idx < pairs.size(); ++idx) {
                        const int n_p = pairs[idx].n_pno;
                        if (n_p == 0) continue;
                        const size_t off = y_offset[idx];
                        for (int a = 0; a < n_p; ++a)
                            for (int b = 0; b < n_p; ++b) {
                                const size_t k =
                                    off + static_cast<size_t>(a) * n_p + b;
                                pairs[idx].Y[a * n_p + b] =
                                    static_cast<real_t>(y_ext[k]);
                            }
                    }
                } catch (const std::runtime_error&) {
                    // singular DIIS B matrix → reset history so the next
                    // iteration starts from a clean subspace instead of
                    // accumulating more near-parallel error vectors.
                    if (verbose >= 2) {
                        std::cout << "[DLPNO] DIIS reset on singular B"
                                  << " (history=" << diis.history_size()
                                  << ")" << std::endl;
                    }
                    diis.clear();
                    n_stalled = 0;
                }
            }
        }
        dt_diis += std::chrono::duration<double>(prof_clock::now() - t_diis_0).count();

        const auto t_misc_0 = prof_clock::now();
        prev_r_max = r_max;

        st.max_R = r_max;
        if (verbose >= 2) {
            std::cout << "[DLPNO] " << round_tag
                      << " CCSD iter " << std::setw(3) << (iter + 1)
                      << "  max|R|=" << std::scientific
                      << std::setprecision(3) << r_max << std::endl;
        }
        dt_iter_misc += std::chrono::duration<double>(
            prof_clock::now() - t_misc_0).count();
        if (r_max < conv_tol) {
            st.iters     = iter + 1;
            st.converged = true;
            break;
        }
        st.iters = iter + 1;
    }
    {
        const double dt_total =
            std::chrono::duration<double>(prof_clock::now() - t_iter_total_0).count();
        const double dt_acct =
            dt_barS + dt_vmeta + dt_picache
          + dt_dFki + dt_DFpair + dt_resid + dt_diis;
        const double dt_other_named =
            dt_setup_pgpu + dt_setup_rgpu
          + dt_yold_copy + dt_compute_async + dt_iter_misc;
        const double dt_other_residual =
            (dt_total - dt_acct) - dt_other_named;
        std::cout << "[DLPNO-ITER-PROF] " << round_tag
                  << "  iters="    << st.iters
                  << "  total="    << std::fixed << std::setprecision(3) << dt_total << "s"
                  << "  barS="     << dt_barS
                  << "  vmeta="    << dt_vmeta
                  << "  picache="  << dt_picache
                  << "  dFki="     << dt_dFki
                  << "  DFpair="   << dt_DFpair
                  << "  resid="    << dt_resid
                  << "  diis="     << dt_diis
                  << "  other="    << (dt_total - dt_acct)
                  << std::endl;
        // Step 6.8: named breakdown of "other" — separate one-time GPU
        // construction from per-iter overhead so we can target the dominant
        // sub-bucket for further optimisation.
        std::cout << "[DLPNO-ITER-PROF] " << round_tag
                  << "  other-breakdown:"
                  << "  setup_pgpu="    << dt_setup_pgpu
                  << "  setup_rgpu="    << dt_setup_rgpu
                  << "  yold_copy="     << dt_yold_copy
                  << "  compute_async=" << dt_compute_async
                  << "  iter_misc="     << dt_iter_misc
                  << "  unaccounted="   << dt_other_residual
                  << std::endl;
        // Per-stage CPU breakdown of the resid bucket — fires regardless
        // of ResidGpu state. Sum across threads = total CPU work; max
        // across threads ≈ wall time per stage assuming balanced load.
        // The active GPU path collapses everything except t_r_init /
        // t_w4virt / t_gpu_add / t_jacobi_diis to ~0 since those CPU
        // sub-stages are skipped via the skip_cpu_* flags; the inactive
        // path runs everything on CPU and t_phladder_cpu typically
        // dominates.
        {
            ResidCpuStage sum;
            ResidCpuStage mx;
            int n_threads_seen = 0;
            for (const ResidCpuStage& t : resid_prof) {
                const double s = t.t_r_init + t.t_inter_fock + t.t_df_dressing
                               + t.t_w4virt + t.t_oooo + t.t_phladder_cpu
                               + t.t_gpu_add + t.t_jacobi_diis;
                if (s <= 0.0) continue;
                sum.t_r_init       += t.t_r_init;
                sum.t_inter_fock   += t.t_inter_fock;
                sum.t_df_dressing  += t.t_df_dressing;
                sum.t_w4virt       += t.t_w4virt;
                sum.t_oooo         += t.t_oooo;
                sum.t_phladder_cpu += t.t_phladder_cpu;
                sum.t_gpu_add      += t.t_gpu_add;
                sum.t_jacobi_diis  += t.t_jacobi_diis;
                mx.t_r_init       = std::max(mx.t_r_init,       t.t_r_init);
                mx.t_inter_fock   = std::max(mx.t_inter_fock,   t.t_inter_fock);
                mx.t_df_dressing  = std::max(mx.t_df_dressing,  t.t_df_dressing);
                mx.t_w4virt       = std::max(mx.t_w4virt,       t.t_w4virt);
                mx.t_oooo         = std::max(mx.t_oooo,         t.t_oooo);
                mx.t_phladder_cpu = std::max(mx.t_phladder_cpu, t.t_phladder_cpu);
                mx.t_gpu_add      = std::max(mx.t_gpu_add,      t.t_gpu_add);
                mx.t_jacobi_diis  = std::max(mx.t_jacobi_diis,  t.t_jacobi_diis);
                ++n_threads_seen;
            }
            if (n_threads_seen > 0) {
                const double mx_sum = mx.t_r_init + mx.t_inter_fock
                                    + mx.t_df_dressing + mx.t_w4virt + mx.t_oooo
                                    + mx.t_phladder_cpu + mx.t_gpu_add
                                    + mx.t_jacobi_diis;
                std::cout << "[DLPNO-RESID-CPU-PROF] " << round_tag
                          << "  threads=" << n_threads_seen
                          << "  per-thread-max (≈ wall, s)"
                          << "  r_init="       << std::fixed << std::setprecision(3) << mx.t_r_init
                          << "  inter_Fock="   << mx.t_inter_fock
                          << "  df_dressing="  << mx.t_df_dressing
                          << "  w4virt="       << mx.t_w4virt
                          << "  oooo="         << mx.t_oooo
                          << "  phladder_cpu=" << mx.t_phladder_cpu
                          << "  gpu_add="      << mx.t_gpu_add
                          << "  jacobi_diis="  << mx.t_jacobi_diis
                          << "  stage_sum="    << mx_sum
                          << std::endl;
                if (verbose >= 2) {
                    std::cout << "[DLPNO-RESID-CPU-PROF] " << round_tag
                              << "  thread-sum (total CPU work, s)"
                              << "  r_init="       << sum.t_r_init
                              << "  inter_Fock="   << sum.t_inter_fock
                              << "  df_dressing="  << sum.t_df_dressing
                              << "  w4virt="       << sum.t_w4virt
                              << "  oooo="         << sum.t_oooo
                              << "  phladder_cpu=" << sum.t_phladder_cpu
                              << "  gpu_add="      << sum.t_gpu_add
                              << "  jacobi_diis="  << sum.t_jacobi_diis
                              << std::endl;
                }
            }
        }
        // Per-stage GPU pipeline breakdown of the resid bucket. Sum
        // across active devices; max gives the wall-clock-relevant per-
        // iter pipeline length (cudaEventSync at finalize waits on the
        // slowest device). The pipeline sum dominates dt_resid above by
        // construction.
        if (any_rgpu_active) {
            ResidStageTimes sum;
            ResidStageTimes mx;
            int n_active = 0;
            for (int d = 0; d < n_gpus; ++d) {
                if (!rgpus[d] || !rgpus[d]->active()) continue;
                const ResidStageTimes t = rgpus[d]->get_stage_times();
                sum.slice      += t.slice;
                sum.w_block    += t.w_block;
                sum.r_contract += t.r_contract;
                sum.inter_fock += t.inter_fock;
                sum.oooo       += t.oooo;
                sum.d2h        += t.d2h;
                sum.n_iter      = std::max(sum.n_iter, t.n_iter);
                mx.slice      = std::max(mx.slice,      t.slice);
                mx.w_block    = std::max(mx.w_block,    t.w_block);
                mx.r_contract = std::max(mx.r_contract, t.r_contract);
                mx.inter_fock = std::max(mx.inter_fock, t.inter_fock);
                mx.oooo       = std::max(mx.oooo,       t.oooo);
                mx.d2h        = std::max(mx.d2h,        t.d2h);
                mx.n_iter     = std::max(mx.n_iter,     t.n_iter);
                ++n_active;
                if (verbose >= 2) {
                    const double t_total = t.slice + t.w_block + t.r_contract
                                         + t.inter_fock + t.oooo + t.d2h;
                    std::cout << "[DLPNO-RESID-GPU-PROF] " << round_tag
                              << "  device=" << d
                              << "  iters=" << t.n_iter
                              << "  slice="      << std::fixed << std::setprecision(3) << t.slice
                              << "  W_block="    << t.w_block
                              << "  R_contract=" << t.r_contract
                              << "  inter_Fock=" << t.inter_fock
                              << "  oooo="       << t.oooo
                              << "  D2H="        << t.d2h
                              << "  pipe_sum="   << t_total
                              << std::endl;
                }
                rgpus[d]->reset_stage_times();
            }
            if (n_active > 0) {
                const double pipe_sum_mx = mx.slice + mx.w_block + mx.r_contract
                                         + mx.inter_fock + mx.oooo + mx.d2h;
                std::cout << "[DLPNO-RESID-GPU-PROF] " << round_tag
                          << "  per-iter-max-across-GPUs"
                          << "  iters=" << mx.n_iter
                          << "  slice="      << std::fixed << std::setprecision(3) << mx.slice
                          << "  W_block="    << mx.w_block
                          << "  R_contract=" << mx.r_contract
                          << "  inter_Fock=" << mx.inter_fock
                          << "  oooo="       << mx.oooo
                          << "  D2H="        << mx.d2h
                          << "  pipe_sum="   << pipe_sum_mx
                          << std::endl;
            }
        }
    }
    return st;
}

} // namespace gansu
