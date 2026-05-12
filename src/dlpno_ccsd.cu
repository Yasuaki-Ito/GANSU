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
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "device_host_memory.hpp"
#include "dlpno_mp2.hpp"          // solve_dlpno_lmp2 + DLPNOLMP2Result
#include "dlpno_pair_data.hpp"    // PairSetup, PairData
#include "dlpno_eri_gpu.hpp"      // EriBuildGpu (Phase 3.2.7 ERI GPU port)
#include "dlpno_proj_gpu.hpp"     // TripleProjGpu (Phase 3.2.8 T2 proj GPU port)
#include "dlpno_t_gpu.hpp"        // TripleTGpu (Phase 3.2.6 GPU port)
#include "dlpno_tno.hpp"          // TNOBuilder, TNOData (Phase 3.2.0)
#include "eri.hpp"
#include "rhf.hpp"

#include <cstring> // memcpy

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
        std::cout << "[DLPNO-CCSD] pair classification: "
                  << "strong=" << n_strong
                  << "  weak=" << n_weak
                  << (n_empty > 0
                      ? "  (empty=" + std::to_string(n_empty) + ")" : "")
                  << std::endl;
    }

    // Register pair stats so the post-HF summary can report them.
    rhf_.set_last_dlpno_pairs(n_strong, n_weak, n_empty);

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
            params_.verbose, "CCSD T2 iteration",
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
                  << "\n[DLPNO-CCSD]   E(strong-pair CCSD) = "
                  << std::scientific << std::setprecision(10) << E_strong
                  << "\n[DLPNO-CCSD]   E(weak-pair MP2)    = " << E_weak
                  << "\n[DLPNO-CCSD]   E(T1 contribution)  = " << E_T1
                  << "\n[DLPNO-CCSD]   E(total CCSD corr)  = " << E_total
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

/// Phase 3.2.2b — extract the global AO 3-index B intermediate to a host
/// buffer with layout (nao² × naux) row-major (μν is slow, Q is fast).
/// Both single-GPU (ERI_RI_RHF::intermediate_matrix_B_) and multi-GPU
/// (ERI_RI_Distributed_RHF::d_B_full_per_gpu_[0]) cases are handled. For
/// the distributed case, replicate_B_to_all_gpus() is invoked here if not
/// already replicated by an earlier solve_dlpno_lmp2 call.
std::vector<real_t> extract_global_B_to_host_munu_Q(ERI_RI_RHF& eri,
                                                    int nao,
                                                    int naux) {
    const size_t nao2 = static_cast<size_t>(nao) * nao;
    const size_t total = nao2 * naux;
    std::vector<real_t> B_in(total);   // (naux × nao²) layout from ERI

#ifdef GANSU_MULTI_GPU
    if (auto* dist = dynamic_cast<ERI_RI_Distributed_RHF*>(&eri)) {
        if (!dist->b_is_replicated()) {
            if (!dist->replicate_B_to_all_gpus()) {
                throw std::runtime_error(
                    "[DLPNO-(T)] B replication failed; per-triple ERI "
                    "construction needs the full AO B on a single device");
            }
        }
        const real_t* d_B = dist->get_replicated_B_device(0);
        if (d_B == nullptr) {
            throw std::runtime_error(
                "[DLPNO-(T)] get_replicated_B_device(0) returned null");
        }
        cudaSetDevice(0);
        cudaMemcpy(B_in.data(), d_B, total * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
    } else
#endif
    {
        eri.get_intermediate_matrix_B().toHost();
        std::memcpy(B_in.data(),
                    eri.get_intermediate_matrix_B().host_ptr(),
                    total * sizeof(real_t));
    }

    // Transpose layout (naux, nao²) → (nao², naux), both row-major.
    std::vector<real_t> B_out(total);
    #pragma omp parallel for schedule(static)
    for (long long mn = 0; mn < static_cast<long long>(nao2); ++mn) {
        const real_t* src_col = B_in.data() + mn;       // stride nao²
        real_t* dst_row = B_out.data() + mn * naux;
        for (int Q = 0; Q < naux; ++Q)
            dst_row[Q] = src_col[static_cast<size_t>(Q) * nao2];
    }
    return B_out;
}

/// Phase 3.2.2b — precompute B_lmo_ao = Σ_μ C_LMO[μ,l] B_ao[μν|Q].
/// Input  B_ao_ao layout: (nao² × naux) row-major, B[(μ*nao+ν)*naux+Q].
/// Output layout:        (nocc × nao × naux) row-major, B[(l*nao+ν)*naux+Q].
std::vector<real_t> contract_C_LMO_into_B_ao(const real_t* B_ao_ao,
                                             const real_t* C_LMO,
                                             int nao, int nocc, int naux) {
    // C_LMO is (nao × nocc) row-major. We want C_LMO^T (nocc × nao) on the
    // left-hand side of a single big DGEMM:
    //    M (nocc × nao*naux) = C_LMO^T (nocc × nao) · B_ao (nao × nao*naux)
    // where B_ao is reinterpreted as (nao × nao*naux) row-major. Output
    // M[l, ν*naux + Q] equals B_lmo_ao[l, ν, Q] with the desired layout.
    Eigen::Map<const RowMatXd> Bm(B_ao_ao, nao, nao * naux);
    Eigen::Map<const RowMatXd> Clmo(C_LMO, nao, nocc);
    const RowMatXd M = Clmo.transpose() * Bm;
    std::vector<real_t> out(static_cast<size_t>(nocc) * nao * naux);
    Eigen::Map<RowMatXd>(out.data(), nocc, nao * naux) = M;
    return out;
}

/// Phase 3.2.2b — precompute B_lmo_lmo = Σ_ν C_LMO[ν,m] B_lmo_ao[l,ν|Q].
/// Output layout: (nocc × nocc × naux) row-major, B[(l*nocc+m)*naux+Q].
std::vector<real_t> contract_C_LMO_into_B_lmo_ao(const real_t* B_lmo_ao,
                                                 const real_t* C_LMO,
                                                 int nao, int nocc, int naux) {
    // Per l: B_lmo_ao_l (nao × naux) reduced by C_LMO^T to (nocc × naux).
    std::vector<real_t> out(static_cast<size_t>(nocc) * nocc * naux);
    Eigen::Map<const RowMatXd> Clmo(C_LMO, nao, nocc);
    #pragma omp parallel for schedule(static)
    for (int l = 0; l < nocc; ++l) {
        Eigen::Map<const RowMatXd> Bl(
            B_lmo_ao + static_cast<size_t>(l) * nao * naux, nao, naux);
        const RowMatXd Bl_lmo = Clmo.transpose() * Bl; // (nocc × naux)
        Eigen::Map<RowMatXd>(
            out.data() + static_cast<size_t>(l) * nocc * naux, nocc, naux)
            = Bl_lmo;
    }
    return out;
}

/// Phase 3.1 + 3.3 — build the active-triple list, iterate via per-GPU
/// OpenMP threads, and report PNO-union size statistics + per-thread
/// progression. Each thread processes its own slab via static schedule.
/// Phase 3.2.0 (TNO basis) and 3.2.1 (T2 projection) wired in. Phase 3.2.2b
/// adds per-triple ERI tensors K_iadc and L_lcmn. The (T) contraction
/// itself remains 0 until Phase 3.2.3.
real_t run_phase33_triple_loop(const DLPNOLMP2Result& res,
                               const real_t* h_F,
                               const real_t* h_S,
                               const real_t* B_ao_ao,
                               const real_t* B_lmo_ao,
                               const real_t* B_lmo_lmo,
                               int naux,
                               int num_gpus,
                               int verbose,
                               long long* out_total_triples = nullptr,
                               long long* out_active_triples = nullptr) {
    const int nocc = res.nocc;
    const int nao  = res.nao;

    // Phase 3.2.6 (2026-05-09): use i ≤ j ≤ k loop with the PySCF-equivalent
    // 6-W formula (compute_triple_t_energy_pyscf). The d3_ijk degeneracy factor
    // (1, 2, 6 for distinct / one-equal / all-equal) is applied INSIDE the new
    // energy function, so we no longer need strict i<j<k filtering.
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

    // Phase 3.2.0: TNO builder is shared across threads (read-only).
    TNOBuilder tno_builder(res.pairs, h_F, h_S, nao);

    // Per-thread accumulators (one entry per GPU = OpenMP thread).
    const int n_threads = std::max(1, num_gpus);
    std::vector<real_t> local_e(n_threads, 0.0);
    std::vector<long long> local_count(n_threads, 0);
    // Per-thread TNO size diagnostics (Phase 3.2.0).
    std::vector<long long> local_tno_sum(n_threads, 0);
    std::vector<int>       local_tno_max(n_threads, 0);
    std::vector<long long> local_dropped_sum(n_threads, 0);
    // Per-thread T2 projection diagnostics (Phase 3.2.1) — accumulate Σ |T_tno|_F²
    // averaged over the 3 pairs of each triple, to confirm projection is
    // actually firing on real amplitudes.
    std::vector<real_t>    local_t2_norm_sum(n_threads, 0.0);
    // Per-thread ERI norm diagnostics.
    std::vector<real_t>    local_K_norm_sum(n_threads, 0.0);

#ifdef _OPENMP
    if (n_threads > 1) {
        omp_set_dynamic(0);
        omp_set_num_threads(n_threads);
    }
#endif

    // Per-thread GPU helper for compute_triple_t_energy. Constructed inside
    // the OMP region so that cudaSetDevice has already pinned the right GPU.
    // Falls back to active()=false if GPU unavailable; caller dispatches to
    // CPU compute_triple_t_energy_pyscf in that case.
    //
    // Phase 3.2.6 GPU optimisation: per-thread *batched* compute. Each
    // thread queues all its triples via add_to_batch() inside the OMP for,
    // then a single flush_batch() at the end performs ONE large upload, ONE
    // batched kernel sequence, and ONE download.
    //
    std::vector<long long> local_gpu_count(n_threads, 0);
    // Per-thread per-section wall-time (seconds). Indices:
    //   0=tno_build, 1=eri_build, 2=t2_proj, 3=energy_queue, 4=flush
    std::vector<std::array<double, 5>> local_sec(n_threads, {0.0, 0.0, 0.0, 0.0, 0.0});
    const int n_triples = static_cast<int>(triples.size());
    // Tight bound on TNO size: post-orthogonalisation can only drop dimensions,
    // never grow them, so the per-triple PNO-union size (n_pno_ij + n_pno_ik +
    // n_pno_jk) is a valid upper bound on n_tno. The loose bound nvir = nao -
    // nocc would over-pad each GPU scratch buffer by O((nvir/max_union)²-³),
    // costing seconds of cudaMalloc per thread.
    const int nvir_bound = std::min(nao - nocc,
                                     std::max(1, max_union_size));
    // Upper bound on triples per thread (static-1 schedule round-robins).
    const int max_per_thread = (n_triples + n_threads - 1) / n_threads + 4;

    // Per-thread setup wall time (outside the per-triple sections).
    std::vector<double> local_setup(n_threads, 0.0);

    #pragma omp parallel num_threads(n_threads)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        using Clk_setup = std::chrono::steady_clock;
        const auto t_setup0 = Clk_setup::now();
        if (n_threads > 1) {
            cudaSetDevice(tid);
        }
        // Allocate GPU scratch sized to the true upper bound on n_tno (=nvir).
        TripleTGpu tgpu(nvir_bound, nocc, max_per_thread);
        if (tgpu.active()) tgpu.begin_batch();

        // GPU helper for the dominant per-triple ERI build. Uploads
        // B_ao_ao / B_lmo_ao / B_lmo_lmo once at construction, then per-triple
        // builds B_TTQ + K_iadc + M on GPU via cuBLAS strided batched DGEMM.
        // Falls back to CPU path inside build_eri_in_tno if active()=false.
        EriBuildGpu eri_gpu(B_ao_ao, B_lmo_ao, B_lmo_lmo,
                            nao, nocc, naux, nvir_bound);

        // GPU helper for the per-triple T2 amplitude projections (99 small
        // projections per triple). Uploads all pairs' bar_Q, precomputed
        // S·bar_Q, and Y_oriented once; per-triple compute is 3 batched
        // cuBLAS DGEMM calls. Falls back to CPU project_pair_t2_oriented
        // when inactive.
        TripleProjGpu proj_gpu(res.pairs, res.setups, res.pair_lookup,
                                h_S, nao, nocc, nvir_bound);
        local_setup[tid] = std::chrono::duration<double>(
                              Clk_setup::now() - t_setup0).count();
        real_t local_e_gpu = 0.0;   // accumulated by this thread's flush

        #pragma omp for schedule(static, 1)
        for (int t_idx = 0; t_idx < static_cast<int>(triples.size()); ++t_idx) {
            using Clk = std::chrono::steady_clock;
            auto t_sec0 = Clk::now();
            const TripleEntry& tr = triples[t_idx];
            // Phase 3.2.0: build orthogonalised TNO basis for this triple.
            // Used only for size statistics here; subsequent sub-phases
            // (3.2.1+) will project amplitudes/integrals into Q_tno and
            // compute the (T) contribution.
            const TNOData tno = tno_builder.build_for_triple(
                tr.idx_ij, tr.idx_ik, tr.idx_jk);
            local_tno_sum[tid] += tno.n_tno;
            local_tno_max[tid] = std::max(local_tno_max[tid], tno.n_tno);
            local_dropped_sum[tid] += tno.n_dropped_overlap;

            // Phase 3.2.1: project the converged T2 amplitudes for the three
            // pairs into the TNO basis. The result is consumed by Phase 3.2.3
            // (T3 amplitude assembly); for now we only accumulate the average
            // Frobenius norm for diagnostics.
            const T2InTNO t2 = project_triple_t2_to_tno(
                tno,
                res.pairs[tr.idx_ij],
                res.pairs[tr.idx_ik],
                res.pairs[tr.idx_jk],
                h_S, nao);
            real_t f2 = 0.0;
            for (real_t v : t2.T_ij) f2 += v * v;
            for (real_t v : t2.T_ik) f2 += v * v;
            for (real_t v : t2.T_jk) f2 += v * v;
            local_t2_norm_sum[tid] += std::sqrt(f2 / 3.0);
            auto t_sec1 = Clk::now();
            local_sec[tid][0] +=
                std::chrono::duration<double>(t_sec1 - t_sec0).count();

            // Build per-triple K_iadc and M tensors. When EriBuildGpu is
            // active, both runs on GPU via cuBLAS strided batched DGEMM.
            // Fallback: full CPU build_eri_in_tno + build_hole_m_tensors path.
            const int triple_lmos[3] = {tr.i, tr.j, tr.k};
            ERIInTNO eri_t;
            std::array<std::vector<real_t>, 9> M_gpu;
            bool eri_done_on_gpu = false;
            if (eri_gpu.active() && tno.n_tno > 0) {
                std::vector<real_t> K_buf;
                if (eri_gpu.build_eri_and_m(tno.Q_tno.data(), tno.n_tno,
                                             triple_lmos, K_buf, M_gpu)) {
                    eri_t.n_tno = tno.n_tno;
                    eri_t.nocc  = nocc;
                    eri_t.K_iadc = std::move(K_buf);
                    eri_done_on_gpu = true;
                }
            }
            if (!eri_done_on_gpu) {
                eri_t = build_eri_in_tno(
                    tno, triple_lmos,
                    B_lmo_ao, B_ao_ao, B_lmo_lmo,
                    nao, nocc, naux, /*build_L=*/false);
            }
            real_t kn = 0.0;
            for (real_t v : eri_t.K_iadc) kn += v * v;
            local_K_norm_sum[tid] += std::sqrt(kn);
            auto t_sec2 = Clk::now();
            local_sec[tid][1] +=
                std::chrono::duration<double>(t_sec2 - t_sec1).count();

            // Extend T2 projection to all (i_in_triple, l) pairs in the TNO
            // basis, plus the 6 off-diagonal T_part orientations. When
            // TripleProjGpu is active, all 3·nocc + 6 projections run in a
            // single batched GPU pipeline; otherwise fall back to per-call
            // CPU projection.
            std::vector<std::vector<real_t>> T_il_ext;
            std::vector<std::vector<real_t>> T_jl_ext;
            std::vector<std::vector<real_t>> T_kl_ext;
            std::array<std::vector<real_t>, 9> T_part_gpu;
            bool proj_done_on_gpu = false;
            if (proj_gpu.active() && tno.n_tno > 0) {
                if (proj_gpu.project_for_triple(tno.Q_tno.data(), tno.n_tno,
                                                 triple_lmos,
                                                 T_il_ext, T_jl_ext, T_kl_ext,
                                                 T_part_gpu)) {
                    proj_done_on_gpu = true;
                }
            }
            if (!proj_done_on_gpu) {
                T_il_ext.assign(nocc, {});
                T_jl_ext.assign(nocc, {});
                T_kl_ext.assign(nocc, {});
                for (int l = 0; l < nocc; ++l) {
                    T_il_ext[l] = project_pair_t2_oriented_to_tno(
                        tno, res.pairs, res.setups, res.pair_lookup,
                        tr.i, l, h_S, nao, nocc);
                    T_jl_ext[l] = project_pair_t2_oriented_to_tno(
                        tno, res.pairs, res.setups, res.pair_lookup,
                        tr.j, l, h_S, nao, nocc);
                    T_kl_ext[l] = project_pair_t2_oriented_to_tno(
                        tno, res.pairs, res.setups, res.pair_lookup,
                        tr.k, l, h_S, nao, nocc);
                }
            }
            auto t_sec3 = Clk::now();
            local_sec[tid][2] +=
                std::chrono::duration<double>(t_sec3 - t_sec2).count();

            // Phase 3.2.6: build hole-side M tensors for the 6 perm pairs
            // (lmo_p, lmo_q) ∈ S(3,2). Reuse GPU-built M when EriBuildGpu was
            // active; fall back to CPU build_hole_m_tensors otherwise.
            HoleMTensors hole_m;
            if (eri_done_on_gpu) {
                hole_m.n_tno = tno.n_tno;
                hole_m.nocc  = nocc;
                hole_m.M     = std::move(M_gpu);
            } else {
                hole_m = build_hole_m_tensors(
                    tno, triple_lmos, B_lmo_ao, B_lmo_lmo, nao, nocc, naux);
            }

            // 6 ordered-pair t2 amplitudes for the particle term. Reuse GPU-
            // built T_part when TripleProjGpu was active; otherwise CPU path.
            std::array<std::vector<real_t>, 9> T_part;
            if (proj_done_on_gpu) {
                T_part = std::move(T_part_gpu);
            } else {
                for (int sp = 0; sp < 3; ++sp) {
                    for (int sq = 0; sq < 3; ++sq) {
                        if (sp == sq) continue;
                        T_part[sp * 3 + sq] = project_pair_t2_oriented_to_tno(
                            tno, res.pairs, res.setups, res.pair_lookup,
                            triple_lmos[sp], triple_lmos[sq], h_S, nao, nocc);
                    }
                }
            }

            // Phase 3.2.6: closed-shell (T) energy via the PySCF-equivalent
            // 6-W formula. d3_ijk degeneracy factor is applied internally.
            const real_t eps_i = res.F_LMO[tr.i * nocc + tr.i];
            const real_t eps_j = res.F_LMO[tr.j * nocc + tr.j];
            const real_t eps_k = res.F_LMO[tr.k * nocc + tr.k];
            real_t e_ijk = 0.0;
            if (tgpu.active()) {
                // Queue this triple for batched GPU compute. The actual
                // energy contribution is summed in flush_batch(). On TEOS-
                // class systems (max_n_tno ~ 100+) the constructor caps
                // max_batch below the per-thread triple count to keep buffer
                // memory ≤ a few GB; we chunked-flush when the slot fills.
                bool queued = tgpu.add_to_batch(
                    tr.i, tr.j, tr.k, eps_i, eps_j, eps_k, tno,
                    eri_t.K_iadc.data(), hole_m.M, T_part,
                    T_il_ext, T_jl_ext, T_kl_ext, nocc);
                if (!queued) {
                    // Slot exhausted — flush the batch in place and retry.
                    const auto t_flush0 = std::chrono::steady_clock::now();
                    local_e[tid] += tgpu.flush_batch();
                    const auto t_flush1 = std::chrono::steady_clock::now();
                    local_sec[tid][4] += std::chrono::duration<double>(
                                            t_flush1 - t_flush0).count();
                    tgpu.begin_batch();
                    queued = tgpu.add_to_batch(
                        tr.i, tr.j, tr.k, eps_i, eps_j, eps_k, tno,
                        eri_t.K_iadc.data(), hole_m.M, T_part,
                        T_il_ext, T_jl_ext, T_kl_ext, nocc);
                }
                if (queued) {
                    ++local_gpu_count[tid];
                } else {
                    // Single triple doesn't fit (shouldn't happen with the
                    // constructor's memory budgeting). Fall back to CPU for
                    // this triple only.
                    e_ijk = compute_triple_t_energy_pyscf(
                        tr.i, tr.j, tr.k, eps_i, eps_j, eps_k, tno,
                        eri_t.K_iadc.data(), hole_m.M, T_part,
                        T_il_ext, T_jl_ext, T_kl_ext, nocc);
                }
            } else {
                e_ijk = compute_triple_t_energy_pyscf(
                    tr.i, tr.j, tr.k, eps_i, eps_j, eps_k, tno,
                    eri_t.K_iadc.data(), hole_m.M, T_part,
                    T_il_ext, T_jl_ext, T_kl_ext, nocc);
            }
            local_e[tid] += e_ijk;
            ++local_count[tid];
            auto t_sec4 = Clk::now();
            local_sec[tid][3] +=
                std::chrono::duration<double>(t_sec4 - t_sec3).count();
        }

        // Phase 3.2.6 GPU optimisation: flush all queued triples in one
        // batched kernel sequence. local_e[tid] picks up the summed
        // contribution from this thread's GPU-queued triples.
        if (tgpu.active()) {
            const auto t_flush0 = std::chrono::steady_clock::now();
            local_e[tid] += tgpu.flush_batch();
            const auto t_flush1 = std::chrono::steady_clock::now();
            local_sec[tid][4] +=
                std::chrono::duration<double>(t_flush1 - t_flush0).count();
        }
    }

    real_t e_triples = 0.0;
    long long total_counted = 0;
    for (int t = 0; t < n_threads; ++t) {
        e_triples += local_e[t];
        total_counted += local_count[t];
    }

    const long long active_total = total - skipped_empty;
    if (out_total_triples)  *out_total_triples  = total;
    if (out_active_triples) *out_active_triples = active_total;

    if (verbose >= 1) {
        const long long active = total - skipped_empty;
        const real_t avg_union = active > 0
            ? static_cast<real_t>(sum_union_size) / active : 0.0;
        long long tno_sum_total = 0;
        long long tno_dropped_total = 0;
        int       tno_max_total = 0;
        real_t    t2_norm_sum_total = 0.0;
        real_t    K_norm_sum_total  = 0.0;
        for (int t = 0; t < n_threads; ++t) {
            tno_sum_total     += local_tno_sum[t];
            tno_dropped_total += local_dropped_sum[t];
            tno_max_total      = std::max(tno_max_total, local_tno_max[t]);
            t2_norm_sum_total += local_t2_norm_sum[t];
            K_norm_sum_total  += local_K_norm_sum[t];
        }
        const real_t avg_tno = active > 0
            ? static_cast<real_t>(tno_sum_total) / active : 0.0;
        const real_t avg_drop = active > 0
            ? static_cast<real_t>(tno_dropped_total) / active : 0.0;
        const real_t avg_t2_norm = active > 0
            ? t2_norm_sum_total / active : 0.0;
        const real_t avg_K_norm = active > 0 ? K_norm_sum_total / active : 0.0;
        std::cout << "[DLPNO-(T)] triple statistics:\n"
                  << "  num GPUs (OpenMP threads)  : " << n_threads << "\n"
                  << "  total triples (i≤j≤k)      : " << total << "\n"
                  << "  skipped (empty pair PNO)   : " << skipped_empty << "\n"
                  << "  active triples             : " << active
                  << "  (counted across threads: " << total_counted << ")\n"
                  << "  avg PNO-union size (upper) : "
                  << std::fixed << std::setprecision(1) << avg_union << "\n"
                  << "  max PNO-union size (upper) : " << max_union_size << "\n"
                  << "  avg TNO size (post-ortho)  : "
                  << std::fixed << std::setprecision(1) << avg_tno << "\n"
                  << "  max TNO size (post-ortho)  : " << tno_max_total << "\n"
                  << "  avg lin-dep dropped        : "
                  << std::fixed << std::setprecision(2) << avg_drop << "\n"
                  << "  avg |T_pair|_F (TNO basis) : "
                  << std::scientific << std::setprecision(3) << avg_t2_norm << "\n"
                  << "  avg |K_iadc|_F             : "
                  << std::scientific << std::setprecision(3) << avg_K_norm << "\n"
                  << "  GPU-accelerated triples    : "
                  << std::accumulate(local_gpu_count.begin(),
                                      local_gpu_count.end(), 0LL) << "\n"
                  << "  per-thread counts          : ";
        for (int t = 0; t < n_threads; ++t)
            std::cout << local_count[t] << (t + 1 == n_threads ? "" : " ");
        std::cout << std::endl;
    }

    // Per-section profile is debug-only (verbose ≥ 2): max per thread approxi-
    // mates the wall the parallel region waited on.
    if (verbose >= 2) {
        const char* sec_names[5] = {
            "tno_build+t2_proj_pno", "eri_build+m   ",
            "t2_proj_ext   ", "energy_queue  ", "flush_batch   "};
        std::cout << "[DLPNO-(T)-PROF] per-section wall (max thread, s):\n";
        for (int s = 0; s < 5; ++s) {
            double mx = 0.0;
            double sm = 0.0;
            for (int t = 0; t < n_threads; ++t) {
                mx = std::max(mx, local_sec[t][s]);
                sm += local_sec[t][s];
            }
            std::cout << "    [" << s << "] " << sec_names[s]
                      << " max=" << std::fixed << std::setprecision(3) << mx
                      << "  sum=" << sm << "\n";
        }
        double setup_mx = 0.0, setup_sm = 0.0;
        for (int t = 0; t < n_threads; ++t) {
            setup_mx = std::max(setup_mx, local_setup[t]);
            setup_sm += local_setup[t];
        }
        std::cout << "    [setup] per-thread GPU helper ctor: max="
                  << std::fixed << std::setprecision(3) << setup_mx
                  << "  sum=" << setup_sm << std::endl;
    }
    return e_triples;
}

} // anonymous namespace

real_t ERI_RI_RHF::compute_dlpno_ccsd_t() {
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
    // Phase 3.2.0: TNO basis builder needs AO-basis Fock and overlap.
    rhf_.get_fock_matrix().toHost();
    rhf_.get_overlap_matrix().toHost();
    const real_t* h_F = rhf_.get_fock_matrix().host_ptr();
    const real_t* h_S = rhf_.get_overlap_matrix().host_ptr();

    // Phase 3.2.2b: extract global RI tensors needed by build_eri_in_tno.
    // Layout convention used downstream: (nao² × naux) for B_ao_ao,
    // (nocc × nao × naux) for B_lmo_ao, (nocc² × naux) for B_lmo_lmo.
    const int naux = static_cast<int>(get_num_auxiliary_basis());
    const int nao  = res.nao;
    const int nocc = res.nocc;
    using Clk = std::chrono::steady_clock;
    const auto t_b0 = Clk::now();
    const auto B_ao_ao    = extract_global_B_to_host_munu_Q(*this, nao, naux);
    const auto t_b1 = Clk::now();
    const auto B_lmo_ao   = contract_C_LMO_into_B_ao(
        B_ao_ao.data(), res.C_LMO.data(), nao, nocc, naux);
    const auto t_b2 = Clk::now();
    const auto B_lmo_lmo  = contract_C_LMO_into_B_lmo_ao(
        B_lmo_ao.data(), res.C_LMO.data(), nao, nocc, naux);
    const auto t_b3 = Clk::now();
    const int dlpno_verbose = rhf_.get_dlpno_verbose();
    if (dlpno_verbose >= 1) {
        const size_t B_mb = (B_ao_ao.size() * sizeof(real_t)) >> 20;
        const size_t Lmo_ao_mb = (B_lmo_ao.size() * sizeof(real_t)) >> 20;
        const size_t Lmo_lmo_mb = (B_lmo_lmo.size() * sizeof(real_t)) >> 20;
        std::cout << "[DLPNO-(T)] global RI host buffers: "
                  << "B_ao_ao=" << B_mb << " MB, "
                  << "B_lmo_ao=" << Lmo_ao_mb << " MB, "
                  << "B_lmo_lmo=" << Lmo_lmo_mb << " MB" << std::endl;
    }
    if (dlpno_verbose >= 2) {
        std::cout << "[DLPNO-(T)-PROF] B prep: extract_B="
                  << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(t_b1 - t_b0).count()
                  << "s  contract_lmo_ao="
                  << std::chrono::duration<double>(t_b2 - t_b1).count()
                  << "s  contract_lmo_lmo="
                  << std::chrono::duration<double>(t_b3 - t_b2).count()
                  << "s" << std::endl;
    }

    long long n_total_triples = 0;
    long long n_active_triples = 0;
    const auto t_loop0 = Clk::now();
    const real_t e_triples =
        run_phase33_triple_loop(res, h_F, h_S,
                                B_ao_ao.data(), B_lmo_ao.data(),
                                B_lmo_lmo.data(), naux,
                                num_gpus, dlpno_verbose,
                                &n_total_triples, &n_active_triples);
    const auto t_loop1 = Clk::now();
    if (dlpno_verbose >= 2) {
        std::cout << "[DLPNO-(T)-PROF] run_phase33_triple_loop="
                  << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double>(t_loop1 - t_loop0).count()
                  << "s" << std::endl;
    }
    // Register triple stats so the post-HF summary can report them.
    rhf_.set_last_dlpno_triples(static_cast<int>(n_total_triples),
                                 static_cast<int>(n_active_triples));

    if (rhf_.get_dlpno_verbose() >= 1) {
        std::cout << "[DLPNO-CCSD(T)] E(CCSD)            = "
                  << std::scientific << std::setprecision(10) << e_ccsd
                  << "\n[DLPNO-CCSD(T)] E((T))             = " << e_triples
                  << "\n[DLPNO-CCSD(T)] E(total)           = "
                  << e_ccsd + e_triples << std::endl;
    }
    return e_ccsd + e_triples;
}

} // namespace gansu
