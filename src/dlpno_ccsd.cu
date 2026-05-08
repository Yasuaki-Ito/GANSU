/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_ccsd.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "device_host_memory.hpp"
#include "dlpno_mp2.hpp"          // solve_dlpno_lmp2 + DLPNOLMP2Result
#include "dlpno_pair_data.hpp"    // PairSetup, PairData
#include "eri.hpp"
#include "rhf.hpp"

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
constexpr real_t kFLMOThresh = 1e-14;

/// Build the LMO–PAO Fock block f_{ia}^{(i)} for a given LMO i in pair
/// (i,i)'s semi-canonical PAO basis:
///     f_i^a = (C_LMO)^T_i · F_AO · C_can_pair^{(ii)}_a
/// Brillouin's theorem makes this vanish when C_can_pair lives in the
/// PAO subspace (true by construction); we still compute it explicitly so
/// that the iteration code is correct in non-trivial extensions.
inline std::vector<real_t> build_f_ia(
    const real_t* h_F, int nao,
    const real_t* h_C_LMO_col_i,           // [nao] (column i of C_LMO)
    const real_t* C_can_pair_ii, int n_pao_ii)
{
    std::vector<real_t> f(n_pao_ii, 0.0);
    if (n_pao_ii == 0) return f;
    Eigen::Map<const RowMatXd> Fm(h_F, nao, nao);
    Eigen::Map<const Eigen::VectorXd> Ci(h_C_LMO_col_i, nao);
    Eigen::Map<const RowMatXd> Cpair(C_can_pair_ii, nao, n_pao_ii);
    const Eigen::VectorXd row = Cpair.transpose() * (Fm * Ci);  // [n_pao_ii]
    for (int a = 0; a < n_pao_ii; ++a) f[a] = row(a);
    return f;
}

/// Project T1 amplitudes from pair (k,k)'s PAO basis to pair (i,i)'s PAO
/// basis through the AO overlap:
///     proj^a = Σ_{a'} (C_can_pair^{(ii)})^T S_AO (C_can_pair^{(kk)})_{a,a'} t_k^{a'}
inline std::vector<real_t> project_t1(
    const real_t* C_can_pair_ii, int n_pao_ii,
    const real_t* C_can_pair_kk, int n_pao_kk,
    const std::vector<real_t>& t_k,
    const real_t* h_S, int nao)
{
    std::vector<real_t> out(n_pao_ii, 0.0);
    if (n_pao_ii == 0 || n_pao_kk == 0) return out;
    Eigen::Map<const RowMatXd> Smat(h_S, nao, nao);
    Eigen::Map<const RowMatXd> Cii(C_can_pair_ii, nao, n_pao_ii);
    Eigen::Map<const RowMatXd> Ckk(C_can_pair_kk, nao, n_pao_kk);
    const RowMatXd S_ii_kk = Cii.transpose() * Smat * Ckk;       // [n_pao_ii × n_pao_kk]
    Eigen::Map<const Eigen::VectorXd> tk(t_k.data(), n_pao_kk);
    Eigen::Map<Eigen::VectorXd>(out.data(), n_pao_ii) = S_ii_kk * tk;
    return out;
}

/// Phase 2.4 — pre-compute F_eff dressing integrals.
///
/// For each pair (i,j) with n_pno > 0, build the 4-tensor
///     T_pair^{(ij)}[k, c, l, d] = 2 (kc|ld) − (kd|lc)
/// in pair (i,j)'s PNO basis. Achieved with one per-pair build_mo_eri call
/// using C_ext = [C_LMO (nao × nocc), bar_Q^{(ij)} (nao × n_pno_ij)], then
/// extracting the (k, n_lmo+c, l, n_lmo+d) slices of the chemists'-notation
/// MO ERI (Mulliken layout `eri_mo[p,q,r,s] = (pq|rs)`, row-major).
///
/// Used downstream for both the particle dressing
///   ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
/// and (via the diagonal pair (i,i) tensor) the hole dressing
///   ΔF_{ki}^{l=i} = Σ_{cd} T_pair^{(ii)}[k,c,i,d] · Y_{ii}^{cd}.
///
/// Cost: per pair, n_emb = nocc + n_pno_ij, build_mo_eri produces n_emb^4
/// doubles. For H2O cc-pVDZ (nocc=5, n_pno~10): n_emb=15, n_emb^4 ≈ 50 K
/// per pair × O(N_occ²) pairs. Larger systems will want a direct RI
/// 3-index path; deferred to Phase 2.4.x optimisation.
inline Phase24Integrals precompute_phase24_integrals(
    const ERI& eri,
    const DLPNOLMP2Result& res,
    int nao)
{
    Phase24Integrals out;
    out.nocc = res.nocc;
    const size_t n_pairs = res.setups.size();
    out.n_pno_per_pair.assign(n_pairs, 0);
    out.T_pair.assign(n_pairs, {});
    out.W_pair.assign(n_pairs, {});
    out.W_oooo.assign(n_pairs, {});
    out.W_ovov_i.assign(n_pairs, {});
    out.W_ovov_j.assign(n_pairs, {});
    out.W_ovvo_i.assign(n_pairs, {});
    out.W_ovvo_j.assign(n_pairs, {});
    out.V_ovov_pair.assign(n_pairs, {});

    // Phase B — multi-GPU pair-parallel integral build.
    //   Detect distributed RI back-end with replicated B; if available,
    //   fan out the per-pair build_mo_eri calls across GPUs via OpenMP.
    //   Each thread owns its GPU (cudaSetDevice(tid)) and runs an
    //   independent slab of pairs. Falls back to single-GPU serial when
    //   the back-end is not distributed (or has a single GPU).
    int num_gpus = 1;
#ifdef GANSU_MULTI_GPU
    if (auto* eri_dist = dynamic_cast<const ERI_RI_Distributed_RHF*>(&eri)) {
        if (eri_dist->num_gpus() > 1) {
            const bool ok = const_cast<ERI_RI_Distributed_RHF*>(eri_dist)
                                ->replicate_B_to_all_gpus();
            if (ok) num_gpus = eri_dist->num_gpus();
        }
    }
#endif // GANSU_MULTI_GPU
#ifdef _OPENMP
    if (num_gpus > 1) {
        omp_set_dynamic(0);
        omp_set_num_threads(num_gpus);
    }
#endif

    #pragma omp parallel num_threads(num_gpus)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (num_gpus > 1) cudaSetDevice(tid);

    #pragma omp for schedule(static, 1)
    for (long long idx = 0; idx < static_cast<long long>(n_pairs); ++idx) {
        const PairSetup& s = res.setups[idx];
        const PairData&  p = res.pairs[idx];
        if (p.n_pno == 0) continue;
        const int n_pno = p.n_pno;
        const int n_lmo = res.nocc;
        const int n_emb = n_lmo + n_pno;
        out.n_pno_per_pair[idx] = n_pno;

        // Pack C_ext = [C_LMO_all, bar_Q^{(ij)}] (nao × n_emb).
        std::vector<real_t> C_ext(static_cast<size_t>(nao) * n_emb, 0.0);
        for (int mu = 0; mu < nao; ++mu) {
            for (int j = 0; j < n_lmo; ++j)
                C_ext[mu * n_emb + j] =
                    res.C_LMO[mu * res.nocc + j];
            for (int a = 0; a < n_pno; ++a)
                C_ext[mu * n_emb + n_lmo + a] =
                    p.bar_Q[mu * n_pno + a];
        }

        real_t* d_C_ext = nullptr;
        tracked_cudaMalloc(&d_C_ext,
            static_cast<size_t>(nao) * n_emb * sizeof(real_t));
        cudaMemcpy(d_C_ext, C_ext.data(),
            static_cast<size_t>(nao) * n_emb * sizeof(real_t),
            cudaMemcpyHostToDevice);
        real_t* d_eri_mo = eri.build_mo_eri(d_C_ext, n_emb);

        const size_t n_emb2 = static_cast<size_t>(n_emb) * n_emb;
        const size_t n_emb3 = n_emb2 * n_emb;
        const size_t n_emb4 = n_emb3 * n_emb;

        std::vector<real_t> h_eri_mo(n_emb4, 0.0);
        cudaMemcpy(h_eri_mo.data(), d_eri_mo,
            n_emb4 * sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_C_ext);
        tracked_cudaFree(d_eri_mo);

        // Extract T_pair^{(ij)}[k, c, l, d]
        //   = 2 ERI[k, n_lmo+c, l, n_lmo+d] - ERI[k, n_lmo+d, l, n_lmo+c].
        // Layout: ((k * nocc + l) * n_pno + c) * n_pno + d.
        out.T_pair[idx].assign(
            static_cast<size_t>(n_lmo) * n_lmo * n_pno * n_pno, 0.0);
        for (int k = 0; k < n_lmo; ++k)
            for (int l = 0; l < n_lmo; ++l)
                for (int c = 0; c < n_pno; ++c)
                    for (int d = 0; d < n_pno; ++d) {
                        const size_t idx_kcld =
                            static_cast<size_t>(k) * n_emb3 +
                            static_cast<size_t>(n_lmo + c) * n_emb2 +
                            static_cast<size_t>(l) * n_emb +
                            static_cast<size_t>(n_lmo + d);
                        const size_t idx_kdlc =
                            static_cast<size_t>(k) * n_emb3 +
                            static_cast<size_t>(n_lmo + d) * n_emb2 +
                            static_cast<size_t>(l) * n_emb +
                            static_cast<size_t>(n_lmo + c);
                        const size_t out_idx =
                            ((static_cast<size_t>(k) * n_lmo + l) * n_pno + c) * n_pno + d;
                        out.T_pair[idx][out_idx] =
                            2.0 * h_eri_mo[idx_kcld] - h_eri_mo[idx_kdlc];
                    }

        // Extract W_pair^{(ij)}[a, b, c, d] = canonical W_abcd at T1=0
        //   = v(A, B, C, D) = (ac|bd) chemist (= <ab|cd> physicist).
        // Mulliken layout: eri_mo[A, C, B, D] = (AC|BD).
        out.W_pair[idx].assign(
            static_cast<size_t>(n_pno) * n_pno * n_pno * n_pno, 0.0);
        for (int a = 0; a < n_pno; ++a)
            for (int b = 0; b < n_pno; ++b)
                for (int c = 0; c < n_pno; ++c)
                    for (int d = 0; d < n_pno; ++d) {
                        const size_t idx_acbd =
                            static_cast<size_t>(n_lmo + a) * n_emb3 +
                            static_cast<size_t>(n_lmo + c) * n_emb2 +
                            static_cast<size_t>(n_lmo + b) * n_emb +
                            static_cast<size_t>(n_lmo + d);
                        const size_t out_idx =
                            ((static_cast<size_t>(a) * n_pno + b) * n_pno + c) * n_pno + d;
                        out.W_pair[idx][out_idx] = h_eri_mo[idx_acbd];
                    }

        // Extract W_oooo^{(ij)}[k, l] = canonical W_klij at T1=0
        //   = v(k, l, i, j) = (ki|lj) chemist (= <kl|ij> physicist).
        // Mulliken layout: eri_mo[k, i, l, j] = (ki|lj).
        out.W_oooo[idx].assign(static_cast<size_t>(n_lmo) * n_lmo, 0.0);
        for (int k = 0; k < n_lmo; ++k)
            for (int l = 0; l < n_lmo; ++l) {
                const size_t idx_kilj =
                    static_cast<size_t>(k) * n_emb3 +
                    static_cast<size_t>(s.i) * n_emb2 +
                    static_cast<size_t>(l) * n_emb +
                    static_cast<size_t>(s.j);
                out.W_oooo[idx][k * n_lmo + l] = h_eri_mo[idx_kilj];
            }

        // Extract ovov / ovvo blocks: (a k | I c) and (a k | c I) for I=i,j.
        // Layout: (a * nocc + k) * n_pno + c.
        const size_t n_pno2 = static_cast<size_t>(n_pno) * n_lmo * n_pno;
        out.W_ovov_i[idx].assign(n_pno2, 0.0);
        out.W_ovov_j[idx].assign(n_pno2, 0.0);
        out.W_ovvo_i[idx].assign(n_pno2, 0.0);
        out.W_ovvo_j[idx].assign(n_pno2, 0.0);
        for (int a = 0; a < n_pno; ++a)
            for (int k = 0; k < n_lmo; ++k)
                for (int c = 0; c < n_pno; ++c) {
                    const size_t out_idx =
                        (static_cast<size_t>(a) * n_lmo + k) * n_pno + c;
                    // Canonical W_akic at T1=0 uses v(A,k,I,C) = (AI|kC) =
                    // (a I | k c) chemist. Storage layout reuses index order
                    // (a, k, c) but the integral itself is (a I | k c).
                    //   chemist (pq|rs) = eri_mo[p,q,r,s], so:
                    //   (a I | k c) = eri_mo[n_lmo+a, I, k, n_lmo+c]
                    {
                        const size_t e =
                            static_cast<size_t>(n_lmo + a) * n_emb3 +
                            static_cast<size_t>(s.i) * n_emb2 +
                            static_cast<size_t>(k) * n_emb +
                            static_cast<size_t>(n_lmo + c);
                        out.W_ovov_i[idx][out_idx] = h_eri_mo[e];
                    }
                    {
                        const size_t e =
                            static_cast<size_t>(n_lmo + a) * n_emb3 +
                            static_cast<size_t>(s.j) * n_emb2 +
                            static_cast<size_t>(k) * n_emb +
                            static_cast<size_t>(n_lmo + c);
                        out.W_ovov_j[idx][out_idx] = h_eri_mo[e];
                    }
                    // Canonical W_akci at T1=0 uses v(A,k,C,I) = (AC|kI) =
                    // (a c | k I) chemist. (VV|OO arrangement.)
                    //   = eri_mo[n_lmo+a, n_lmo+c, k, I]
                    {
                        const size_t e =
                            static_cast<size_t>(n_lmo + a) * n_emb3 +
                            static_cast<size_t>(n_lmo + c) * n_emb2 +
                            static_cast<size_t>(k) * n_emb +
                            static_cast<size_t>(s.i);
                        out.W_ovvo_i[idx][out_idx] = h_eri_mo[e];
                    }
                    {
                        const size_t e =
                            static_cast<size_t>(n_lmo + a) * n_emb3 +
                            static_cast<size_t>(n_lmo + c) * n_emb2 +
                            static_cast<size_t>(k) * n_emb +
                            static_cast<size_t>(s.j);
                        out.W_ovvo_j[idx][out_idx] = h_eri_mo[e];
                    }
                }

        // Extract V_ovov_pair^{(ij)}[l, k, d, c] = (ld|kc) — OV|OV slice.
        out.V_ovov_pair[idx].assign(
            static_cast<size_t>(n_lmo) * n_lmo * n_pno * n_pno, 0.0);
        for (int l = 0; l < n_lmo; ++l)
            for (int k = 0; k < n_lmo; ++k)
                for (int d = 0; d < n_pno; ++d)
                    for (int c = 0; c < n_pno; ++c) {
                        const size_t e =
                            static_cast<size_t>(l) * n_emb3 +
                            static_cast<size_t>(n_lmo + d) * n_emb2 +
                            static_cast<size_t>(k) * n_emb +
                            static_cast<size_t>(n_lmo + c);
                        const size_t out_idx =
                            ((static_cast<size_t>(l) * n_lmo + k) * n_pno + d) * n_pno + c;
                        out.V_ovov_pair[idx][out_idx] = h_eri_mo[e];
                    }

        (void)s;  // s used above; suppress -Wunused if conditional.
    }
    }  // end omp parallel
    return out;
}

} // anonymous namespace

DLPNOCCSD::DLPNOCCSD(RHF& rhf, const ERI& eri, DLPNOParams params)
    : rhf_(rhf),
      eri_(eri),
      params_(std::move(params)),
      nao_(rhf.get_num_basis()),
      nocc_(rhf.get_num_electrons() / 2)
{}

real_t DLPNOCCSD::compute_energy()
{
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD] start  nao=" << nao_
                  << "  nocc=" << nocc_
                  << "  preset=" << params_.preset
                  << "  TCutPairs=" << std::scientific
                  << std::setprecision(2) << params_.t_cut_pairs
                  << "  (Phase 2.2 — pair classification + T1 scaffolding)"
                  << std::endl;
    }

    // -----------------------------------------------------------------------
    // Phase 2.1: full Phase-1 setup → converged LMP2 amplitudes per pair.
    // -----------------------------------------------------------------------
    using prof_clock = std::chrono::steady_clock;
    const auto t_lmp2_0 = prof_clock::now();
    auto res = solve_dlpno_lmp2(rhf_, eri_, params_);
    const double dt_lmp2 = std::chrono::duration<double>(
        prof_clock::now() - t_lmp2_0).count();
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD-PROF] solve_dlpno_lmp2 = "
                  << std::fixed << std::setprecision(3)
                  << dt_lmp2 << " s" << std::endl;
    }
    if (res.nocc <= 0) return 0.0;

    int n_strong = 0, n_weak = 0, n_empty = 0;
    real_t E_strong = 0.0, E_weak = 0.0;

    for (size_t idx = 0; idx < res.setups.size(); ++idx) {
        const PairSetup& s = res.setups[idx];
        const PairData&  p = res.pairs[idx];
        if (p.n_pno == 0) { ++n_empty; continue; }
        real_t e_intrinsic = 0.0;
        for (int a = 0; a < p.n_pno; ++a)
            for (int b = 0; b < p.n_pno; ++b)
                e_intrinsic += p.L[a * p.n_pno + b]
                             * (2.0 * p.Y[a * p.n_pno + b]
                                - p.Y[b * p.n_pno + a]);
        const real_t e_pair = s.pair_factor * e_intrinsic;
        if (std::fabs(e_intrinsic) < params_.t_cut_pairs) {
            ++n_weak;  E_weak  += e_pair;
        } else {
            ++n_strong; E_strong += e_pair;
        }
    }

    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD] Phase 2.1 pair classification: "
                  << "strong=" << n_strong
                  << "  weak=" << n_weak
                  << (n_empty > 0
                      ? "  (empty=" + std::to_string(n_empty) + ")" : "")
                  << std::endl;
    }

    // -----------------------------------------------------------------------
    // Phase 2.2: T1 amplitude scaffolding.
    //
    //   T1 lives per LMO i in pair (i,i)'s semi-canonical PAO basis. The
    //   full CCSD residual reads:
    //     R_i^a = f_ia + Σ_c F_ac^{(ii)} t_i^c − Σ_k F_LMO[i,k] · π^{(i)}(t_k)
    //             + (T2 dressing terms — sub-phase 2.5+)
    //
    //   F_ac^{(ii)} is diagonal in the semi-canonical PAO basis (= eps_a^{(ii)}).
    //   The inter-LMO term projects t_k from pair (k,k)'s PAO basis through
    //   the AO overlap into pair (i,i)'s basis.
    //
    //   In Phase 2.2 (no T2 dressing) the source f_ia vanishes by Brillouin
    //   (PAOs are S-orthogonal to the occupied space) and T1 stays at zero
    //   throughout the iteration — the structural validation here is that
    //   the residual evaluator runs without crashing and that ‖T1‖ remains
    //   below numerical noise.
    // -----------------------------------------------------------------------
    rhf_.get_fock_matrix().toHost();
    rhf_.get_overlap_matrix().toHost();
    const real_t* h_F = rhf_.get_fock_matrix().host_ptr();
    const real_t* h_S = rhf_.get_overlap_matrix().host_ptr();
    const real_t* h_C_LMO = res.C_LMO.data();   // [nao × nocc]

    std::vector<std::vector<real_t>> T1(nocc_);
    std::vector<std::vector<real_t>> f_ia(nocc_);

    for (int i = 0; i < nocc_; ++i) {
        const int idx_ii = res.pair_lookup[i * nocc_ + i];
        const PairSetup& s_ii = res.setups[idx_ii];
        T1[i].assign(s_ii.n_pao, 0.0);

        if (s_ii.n_pao == 0) {
            f_ia[i].clear();
            continue;
        }
        // Column i of C_LMO (row-major nao × nocc → column i is stride nocc).
        std::vector<real_t> Ci(nao_, 0.0);
        for (int mu = 0; mu < nao_; ++mu) Ci[mu] = h_C_LMO[mu * nocc_ + i];
        f_ia[i] = build_f_ia(
            h_F, nao_, Ci.data(),
            s_ii.C_can_pair.data(), s_ii.n_pao);
    }

    real_t f_ia_max = 0.0;
    for (int i = 0; i < nocc_; ++i)
        for (real_t v : f_ia[i]) f_ia_max = std::max(f_ia_max, std::fabs(v));
    if (params_.verbose >= 2) {
        std::cout << "[DLPNO-CCSD] max |f_ia| = "
                  << std::scientific << std::setprecision(3) << f_ia_max
                  << " (Brillouin source; should be ≪ 1e-10)" << std::endl;
    }

    const int max_iter = std::max(1, params_.lmp2_max_iter);
    const real_t conv_tol = params_.lmp2_conv;

    int t1_iter = 0;
    real_t r_max_last = 0.0;
    bool t1_converged = false;

    // Snapshot buffer for Jacobi sweep.
    std::vector<std::vector<real_t>> T1_old(nocc_);

    for (int iter = 0; iter < max_iter; ++iter) {
        T1_old = T1;
        real_t r_max = 0.0;

        for (int i = 0; i < nocc_; ++i) {
            const int idx_ii = res.pair_lookup[i * nocc_ + i];
            const PairSetup& s_ii = res.setups[idx_ii];
            const int n_pao_i = s_ii.n_pao;
            if (n_pao_i == 0) continue;

            const real_t F_ii = res.F_LMO[i * nocc_ + i];

            // R_i = f_ia + (eps_a^{(ii)} − F_ii) · t_i^a − Σ_{k≠i} F_LMO[i,k] · π^{(i)}(t_k)
            std::vector<real_t> R = f_ia[i];
            for (int a = 0; a < n_pao_i; ++a)
                R[a] += (s_ii.eps_a[a] - F_ii) * T1_old[i][a];

            for (int k = 0; k < nocc_; ++k) {
                if (k == i) continue;
                const real_t F_ik = res.F_LMO[i * nocc_ + k];
                if (std::fabs(F_ik) < kFLMOThresh) continue;

                const int idx_kk = res.pair_lookup[k * nocc_ + k];
                const PairSetup& s_kk = res.setups[idx_kk];
                if (s_kk.n_pao == 0) continue;

                const auto proj = project_t1(
                    s_ii.C_can_pair.data(), n_pao_i,
                    s_kk.C_can_pair.data(), s_kk.n_pao,
                    T1_old[k], h_S, nao_);
                for (int a = 0; a < n_pao_i; ++a)
                    R[a] -= F_ik * proj[a];
            }

            for (int a = 0; a < n_pao_i; ++a) {
                const real_t denom = s_ii.eps_a[a] - F_ii;
                T1[i][a] -= R[a] / denom;
                r_max = std::max(r_max, std::fabs(R[a]));
            }
        }

        r_max_last = r_max;
        ++t1_iter;
        if (params_.verbose >= 2) {
            std::cout << "[DLPNO-CCSD] T1 iter " << std::setw(3) << t1_iter
                      << "  max|R|=" << std::scientific
                      << std::setprecision(3) << r_max << std::endl;
        }
        if (r_max < conv_tol) { t1_converged = true; break; }
    }

    real_t t1_norm_max = 0.0;
    for (int i = 0; i < nocc_; ++i)
        for (real_t v : T1[i]) t1_norm_max = std::max(t1_norm_max, std::fabs(v));

    // Phase 2.2 energy contribution from T1 (= 2 Σ f_ia t_i^a).
    // With Brillouin + linear-only residual, T1 stays at machine zero so this
    // contributes essentially nothing. Sub-phase 2.5+ will add the T1↔T2
    // dressing terms that drive T1 away from zero.
    real_t E_T1 = 0.0;
    for (int i = 0; i < nocc_; ++i)
        for (size_t a = 0; a < T1[i].size(); ++a)
            E_T1 += 2.0 * f_ia[i][a] * T1[i][a];

    // -----------------------------------------------------------------------
    // Phase 2.3.2 + 2.3.3 + 2.4.1 + 2.4.2: T2 residual with full F_eff
    // dressing on the particle side and l=i hole dressing.
    //
    //   particle (2.3.2 + 2.4.2):
    //     ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
    //     t_{kl,proj} = \bar S^{(ij,kl)} · Y_{kl} · \bar S^{(ij,kl),T}
    //     contracted as (ΔF · Y_old + Y_old · ΔF^T)_{ab}.
    //   hole, diag k=i (2.3.3):
    //     ΔF_{ii} = Σ_{cd} T_pair^{(ii)}[i,c,i,d] · Y_{ii}^{cd}  (scalar)
    //     R^{(ij)}_{ab} += -(ΔF_{ii} + ΔF_{jj}) · Y^{(ij),old}_{ab}.
    //   hole, off-diag k≠i (2.4.1 + 2.4.3, full l sum):
    //     ΔF_{ki} = Σ_l Σ_{cd} T_pair^{(il)}[k,c,l,d] · Y_{il}^{cd}
    //     inter-pair coupling uses F_eff[i,k] = F_LMO[i,k] + ΔF_{ki}.
    //
    //   Phase 2.5/2.6/2.6b/2.7 — full ladder + T2-dressed W + DIIS.
    //     4-vir (2.5):  R^{(ij)}_{ab} += Σ_{cd} W_{abcd}^{(ij)} Y^{(ij)}_{cd}
    //     oooo  (2.6):  R^{(ij)}_{ab} += Σ_{kl} W_{klij}^{(ij)} π^{(ij,kl)}(Y_{kl})_{ab}
    //     ph-ladder (2.6, i+j side) with T2-dressed W_akic / W_akci (2.6b):
    //       ΔW_akic^{(ij)} = -½ Σ_{ld} (lk|dc) π_{il}[d,a]
    //                       + ½ T_pair^{(ij)}[k,c,l,d] π_{il}[a,d]
    //       ΔW_akci^{(ij)} = -½ Σ_{ld} (lk|cd) π_{il}[d,a]
    //   DIIS (2.7) accelerates / stabilises the iteration.
    // -----------------------------------------------------------------------
    const auto t_pre_0 = prof_clock::now();
    Phase24Integrals phase24 =
        precompute_phase24_integrals(eri_, res, nao_);
    const double dt_pre = std::chrono::duration<double>(
        prof_clock::now() - t_pre_0).count();
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD-PROF] precompute_phase24_integrals = "
                  << std::fixed << std::setprecision(3)
                  << dt_pre << " s" << std::endl;
    }

    LMP2Status t2_status{};
    const auto t_iter_0 = prof_clock::now();
    {
        const int  t2_max_iter = std::max(1, params_.lmp2_max_iter);
        const real_t t2_conv   = params_.lmp2_conv;
        t2_status = iterate_dlpno_ccsd_t2(
            res.setups, res.pairs, res.pair_lookup, res.F_LMO, h_S,
            nocc_, nao_, t2_max_iter, t2_conv,
            /*enable_dressing=*/true,
            params_.verbose, "CCSD T2 (Phase 2.3-2.7 + 2.6b)",
            &phase24);
    }
    const double dt_iter = std::chrono::duration<double>(
        prof_clock::now() - t_iter_0).count();
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD-PROF] iterate_dlpno_ccsd_t2 = "
                  << std::fixed << std::setprecision(3)
                  << dt_iter << " s ("
                  << t2_status.iters << " iter)" << std::endl;
    }

    // Recompute strong/weak energies from the post-CCSD Y. With dressing on
    // these are NOT equal to the LMP2 values — the difference is the
    // accumulated dressing effect.
    real_t E_strong_post = 0.0, E_weak_post = 0.0;
    for (size_t idx = 0; idx < res.setups.size(); ++idx) {
        const PairSetup& s = res.setups[idx];
        const PairData&  p = res.pairs[idx];
        if (p.n_pno == 0) continue;
        real_t e_intrinsic = 0.0;
        for (int a = 0; a < p.n_pno; ++a)
            for (int b = 0; b < p.n_pno; ++b)
                e_intrinsic += p.L[a * p.n_pno + b]
                             * (2.0 * p.Y[a * p.n_pno + b]
                                - p.Y[b * p.n_pno + a]);
        const real_t e_pair = s.pair_factor * e_intrinsic;
        if (std::fabs(e_intrinsic) < params_.t_cut_pairs)
            E_weak_post  += e_pair;
        else
            E_strong_post += e_pair;
    }
    const real_t E_ccsd_minus_mp2 =
        (E_strong_post + E_weak_post) - (E_strong + E_weak);

    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD] T2 "
                  << (t2_status.converged ? "converged" : "MAX_ITER")
                  << " in " << t2_status.iters << " iter, max|R|="
                  << std::scientific << std::setprecision(3)
                  << t2_status.max_R
                  << ", ΔE(CCSD-MP2 intra-pair)=" << E_ccsd_minus_mp2
                  << std::endl;
    }

    // Adopt the post-CCSD energies as the new strong/weak split.
    E_strong = E_strong_post;
    E_weak   = E_weak_post;

    const real_t E_total = E_strong + E_weak + E_T1;

    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD] T1 "
                  << (t1_converged ? "converged" : "MAX_ITER")
                  << " in " << t1_iter << " iter, max|R|="
                  << std::scientific << std::setprecision(3) << r_max_last
                  << ", max|T1|=" << t1_norm_max
                  << "\n[DLPNO-CCSD]   E(strong, Phase 2.3-2.7) = "
                  << std::scientific << std::setprecision(10) << E_strong
                  << "\n[DLPNO-CCSD]   E(weak, MP2)              = " << E_weak
                  << "\n[DLPNO-CCSD]   E(T1, ≈ 0 in Phase 2.2)   = " << E_T1
                  << "\n[DLPNO-CCSD]   E(total, Phase 2.3-2.7)   = " << E_total
                  << std::endl;
    }

    return E_total;
}

// ---------------------------------------------------------------------------
//  ERI wiring  —  RI is the only supported back-end for DLPNO.
// ---------------------------------------------------------------------------
real_t ERI_RI_RHF::compute_dlpno_ccsd() {
    DLPNOParams p = resolve_dlpno_params(
        rhf_.get_dlpno_preset(),
        rhf_.get_dlpno_localizer(),
        rhf_.get_dlpno_t_cut_pno(),
        rhf_.get_dlpno_t_cut_do(),
        rhf_.get_dlpno_t_cut_pairs(),
        rhf_.get_dlpno_t_cut_mkn(),
        rhf_.get_dlpno_t_cut_triples(),
        rhf_.get_dlpno_t_cut_tno(),
        rhf_.get_dlpno_pair_distance_cutoff(),
        rhf_.get_dlpno_max_iter(),
        rhf_.get_dlpno_diis_size(),
        rhf_.get_dlpno_localizer_max_sweep(),
        rhf_.get_dlpno_localizer_conv(),
        rhf_.get_dlpno_lmp2_max_iter(),
        rhf_.get_dlpno_lmp2_conv(),
        rhf_.get_dlpno_sc_pno_iter(),
        rhf_.get_dlpno_pno_os_only(),
        rhf_.get_dlpno_verbose());
    DLPNOCCSD driver(rhf_, *this, std::move(p));
    return driver.compute_energy();
}

// ---------------------------------------------------------------------------
//  Phase 3 — DLPNO-CCSD(T).
//
//    3.0  skeleton: CCSD energy + (T) = 0                            [done]
//    3.1  triple iteration + TNO size statistics (pair PNO union)    [done]
//    3.2  (T) energy via canonical-style contractions in TNO basis   [TODO]
//    3.3  multi-GPU triple parallel framework (per-GPU OpenMP slabs) [this commit]
//    3.4  validation against canonical RI-CCSD(T) under strict mode  [TODO]
// ---------------------------------------------------------------------------

namespace {

/// Active-triple descriptor used by the multi-GPU framework. Phase 3.2 fills
/// in the per-triple (T) contribution; for now each entry contributes 0.
struct TripleEntry {
    int i, j, k;
    int idx_ij, idx_ik, idx_jk;
    int n_pno_ij, n_pno_ik, n_pno_jk;
};

/// Phase 3.1 + 3.3 — build the active-triple list, iterate via per-GPU
/// OpenMP threads, and report PNO-union size statistics + per-thread
/// progression. Each thread processes its own slab via static schedule.
/// Phase 3.2 will replace the per-triple work (currently 0) with the
/// canonical-style (T) contraction in the TNO basis.
real_t run_phase33_triple_loop(const DLPNOLMP2Result& res,
                               int num_gpus,
                               int verbose) {
    const int nocc = res.nocc;

    // Build flat list of active triples (i ≤ j ≤ k, all three pairs non-empty).
    std::vector<TripleEntry> triples;
    long long total = 0;
    long long skipped_empty = 0;
    long long sum_union_size = 0;
    int max_union_size = 0;
    for (int i = 0; i < nocc; ++i)
        for (int j = i; j < nocc; ++j)
            for (int k = j; k < nocc; ++k) {
                ++total;
                const int idx_ij = res.pair_lookup[i * nocc + j];
                const int idx_ik = res.pair_lookup[i * nocc + k];
                const int idx_jk = res.pair_lookup[j * nocc + k];
                const int n_ij = res.pairs[idx_ij].n_pno;
                const int n_ik = res.pairs[idx_ik].n_pno;
                const int n_jk = res.pairs[idx_jk].n_pno;
                if (n_ij == 0 || n_ik == 0 || n_jk == 0) {
                    ++skipped_empty;
                    continue;
                }
                const int u = n_ij + n_ik + n_jk;
                sum_union_size += u;
                max_union_size = std::max(max_union_size, u);
                triples.push_back({i, j, k, idx_ij, idx_ik, idx_jk,
                                   n_ij, n_ik, n_jk});
            }

    // Per-thread accumulators (one entry per GPU = OpenMP thread).
    const int n_threads = std::max(1, num_gpus);
    std::vector<real_t> local_e(n_threads, 0.0);
    std::vector<long long> local_count(n_threads, 0);

#ifdef _OPENMP
    if (n_threads > 1) {
        omp_set_dynamic(0);
        omp_set_num_threads(n_threads);
    }
#endif

    #pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (n_threads > 1) {
            cudaSetDevice(tid);
        }
        #pragma omp for schedule(static, 1)
        for (int t_idx = 0; t_idx < static_cast<int>(triples.size()); ++t_idx) {
            const TripleEntry& tr = triples[t_idx];
            // Phase 3.2 TODO: compute per-triple (T) contribution
            // using the orthogonalised TNO basis built from the union of
            // bar_Q^{(ij)}, bar_Q^{(ik)}, bar_Q^{(jk)} and the converged
            // CCSD T1 / T2 amplitudes. For now contribute 0 per triple.
            (void)tr;
            local_e[tid] += 0.0;
            ++local_count[tid];
        }
    }

    real_t e_triples = 0.0;
    long long total_counted = 0;
    for (int t = 0; t < n_threads; ++t) {
        e_triples += local_e[t];
        total_counted += local_count[t];
    }

    if (verbose >= 1) {
        const long long active = total - skipped_empty;
        const real_t avg_union = active > 0
            ? static_cast<real_t>(sum_union_size) / active : 0.0;
        std::cout << "[DLPNO-(T)] Phase 3.1+3.3 triple statistics:\n"
                  << "  num GPUs (OpenMP threads)  : " << n_threads << "\n"
                  << "  total triples (i≤j≤k)      : " << total << "\n"
                  << "  skipped (empty pair PNO)   : " << skipped_empty << "\n"
                  << "  active triples             : " << active
                  << "  (counted across threads: " << total_counted << ")\n"
                  << "  avg PNO-union size (upper) : "
                  << std::fixed << std::setprecision(1) << avg_union << "\n"
                  << "  max PNO-union size (upper) : " << max_union_size
                  << "\n  per-thread counts          : ";
        for (int t = 0; t < n_threads; ++t)
            std::cout << local_count[t] << (t + 1 == n_threads ? "" : " ");
        std::cout << std::endl;
    }
    return e_triples;
}

} // anonymous namespace

real_t ERI_RI_RHF::compute_dlpno_ccsd_t() {
    if (rhf_.get_dlpno_verbose() >= 1) {
        std::cout << "[DLPNO-CCSD(T)] Phase 3.0/3.1/3.3: "
                     "CCSD energy + (T) = 0 (multi-GPU framework only)"
                  << std::endl;
    }
    const real_t e_ccsd = compute_dlpno_ccsd();

    // Phase 3.1: rebuild the LMP2 pair state for TNO statistics.
    DLPNOParams p_tno = resolve_dlpno_params(
        rhf_.get_dlpno_preset(),
        rhf_.get_dlpno_localizer(),
        rhf_.get_dlpno_t_cut_pno(),
        rhf_.get_dlpno_t_cut_do(),
        rhf_.get_dlpno_t_cut_pairs(),
        rhf_.get_dlpno_t_cut_mkn(),
        rhf_.get_dlpno_t_cut_triples(),
        rhf_.get_dlpno_t_cut_tno(),
        rhf_.get_dlpno_pair_distance_cutoff(),
        rhf_.get_dlpno_max_iter(),
        rhf_.get_dlpno_diis_size(),
        rhf_.get_dlpno_localizer_max_sweep(),
        rhf_.get_dlpno_localizer_conv(),
        rhf_.get_dlpno_lmp2_max_iter(),
        rhf_.get_dlpno_lmp2_conv(),
        rhf_.get_dlpno_sc_pno_iter(),
        rhf_.get_dlpno_pno_os_only(),
        /*verbose=*/0);
    auto res = solve_dlpno_lmp2(rhf_, *this, p_tno);

    // Phase 3.3 multi-GPU framework: detect distributed B at this level so
    // the per-GPU OpenMP triple loop can run on `num_gpus` threads. The
    // dynamic_cast checks whether *this is actually the distributed
    // RI back-end; if not, fall back to single-GPU.
    int num_gpus = 1;
#ifdef GANSU_MULTI_GPU
    if (auto* eri_dist =
            dynamic_cast<ERI_RI_Distributed_RHF*>(this)) {
        if (eri_dist->num_gpus() > 1) {
            num_gpus = eri_dist->num_gpus();
        }
    }
#endif // GANSU_MULTI_GPU
    const real_t e_triples =
        run_phase33_triple_loop(res, num_gpus, rhf_.get_dlpno_verbose());

    if (rhf_.get_dlpno_verbose() >= 1) {
        std::cout << "[DLPNO-CCSD(T)] E(CCSD)            = "
                  << std::scientific << std::setprecision(10) << e_ccsd
                  << "\n[DLPNO-CCSD(T)] E((T) placeholder) = " << e_triples
                  << "\n[DLPNO-CCSD(T)] E(total)           = "
                  << e_ccsd + e_triples << std::endl;
    }
    return e_ccsd + e_triples;
}

} // namespace gansu
