/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_mp2.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "device_host_memory.hpp"
#include "dlpno_domain.hpp"
#include "dlpno_localizer.hpp"
#include "dlpno_pair_data.hpp"
#include "dlpno_pao.hpp"
#include "dlpno_pno.hpp"
#include "eri.hpp"
#include "rhf.hpp"

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/// Symmetric eigendecomposition of an n × n row-major matrix. Used for
/// the F_PAO eigendecomposition during per-pair setup.
struct EigDecomp {
    std::vector<real_t> eigvals;
    std::vector<real_t> eigvecs;  // columns are eigenvectors (row-major n×n)
};

EigDecomp eig_sym(const real_t* M, int n)
{
    Eigen::Map<const RowMatXd> A(M, n, n);
    Eigen::SelfAdjointEigenSolver<RowMatXd> es(A);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("DLPNO-MP2: eigendecomposition failed");
    EigDecomp out;
    out.eigvals.resize(n);
    out.eigvecs.assign(static_cast<size_t>(n) * n, 0.0);
    for (int k = 0; k < n; ++k) out.eigvals[k] = es.eigenvalues()(k);
    Eigen::Map<RowMatXd>(out.eigvecs.data(), n, n) = es.eigenvectors();
    return out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  solve_dlpno_lmp2 — full Phase-1 setup + iterative LMP2 + SC-PNO.
//
//  This is the workhorse used by both DLPNOMP2 (returns the energy) and
//  DLPNOCCSD (uses the converged pair data as the starting point for the
//  CCSD residual loop).
// ---------------------------------------------------------------------------
DLPNOLMP2Result solve_dlpno_lmp2(
    RHF& rhf, const ERI& eri, const DLPNOParams& params)
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    // ---- Sub-phase profile probes (Step 5 follow-up). ----
    double dt_localize    = 0.0;  // Phase 2
    double dt_pao_global  = 0.0;  // Phase 3 (PAO global + LMO domains)
    double dt_pair_setup  = 0.0;  // Phase 4 (per-pair PAO setup, GPU build_mo_eri)
    double dt_round0_pno  = 0.0;  // Phase 5 (initial PairData / first PNO eigendecomp)
    double dt_scpno_refresh = 0.0;  // Phase 6 minus iterate_lmp2 (SC-PNO PNO rebuild)
    double dt_iterate     = 0.0;  // iterate_lmp2 (already sub-profiled by LMP2-PROF)

    DLPNOLMP2Result result;
    result.nao  = rhf.get_num_basis();
    result.nocc = rhf.get_num_electrons() / 2;
    if (result.nocc <= 0) return result;

    const int nao_  = result.nao;
    const int nocc_ = result.nocc;
    const int natoms = static_cast<int>(rhf.get_atom_to_basis_range().size());

    // -----------------------------------------------------------------------
    // 1. Pull HF data to host.
    // -----------------------------------------------------------------------
    rhf.get_coefficient_matrix().toHost();
    rhf.get_overlap_matrix().toHost();
    rhf.get_fock_matrix().toHost();
    rhf.get_orbital_energies().toHost();

    const real_t* h_C   = rhf.get_coefficient_matrix().host_ptr();
    const real_t* h_S   = rhf.get_overlap_matrix().host_ptr();
    const real_t* h_F   = rhf.get_fock_matrix().host_ptr();
    const real_t* h_eps = rhf.get_orbital_energies().host_ptr();

    std::vector<real_t> C_occ(static_cast<size_t>(nao_) * nocc_, 0.0);
    for (int mu = 0; mu < nao_; ++mu)
        for (int i = 0; i < nocc_; ++i)
            C_occ[mu * nocc_ + i] = h_C[mu * nao_ + i];

    std::vector<std::pair<int,int>> atom_ranges;
    atom_ranges.reserve(natoms);
    for (const auto& r : rhf.get_atom_to_basis_range()) {
        atom_ranges.emplace_back(static_cast<int>(r.start_index),
                                 static_cast<int>(r.end_index));
    }

    if (params.verbose >= 1) {
        std::cout << "[DLPNO-MP2] start  nao=" << nao_
                  << "  nocc=" << nocc_
                  << "  preset=" << params.preset
                  << "  TCutPNO=" << std::scientific << std::setprecision(2)
                  << params.t_cut_pno
                  << "  TCutMKN=" << params.t_cut_mkn
                  << "  TCutDO="  << params.t_cut_do << std::endl;
    }

    // -----------------------------------------------------------------------
    // 2. Localise occupied MOs.
    // -----------------------------------------------------------------------
    const auto t_loc0 = clock::now();
    auto loc = localize_occupied(
        params.localizer,
        C_occ.data(), h_S, /*Dx*/nullptr, /*Dy*/nullptr, /*Dz*/nullptr,
        nao_, nocc_, atom_ranges,
        params.localizer_max_sweep, params.localizer_conv,
        params.verbose);
    dt_localize += std::chrono::duration<double>(
        clock::now() - t_loc0).count();

    const real_t* h_C_LMO = loc.C_LMO.data();
    const real_t* h_U_loc = loc.U.data();
    // Stash a copy of C_LMO so downstream solvers (CCSD T1 residual) can
    // build occ–vir Fock blocks without re-running the localiser.
    result.C_LMO = loc.C_LMO;

    // F_LMO = U_loc^T · diag(eps_occ) · U_loc.
    std::vector<real_t>& F_LMO = result.F_LMO;
    F_LMO.assign(static_cast<size_t>(nocc_) * nocc_, 0.0);
    {
        Eigen::Map<const RowMatXd> U(h_U_loc, nocc_, nocc_);
        Eigen::VectorXd eps_o(nocc_);
        for (int i = 0; i < nocc_; ++i) eps_o(i) = h_eps[i];
        RowMatXd UD = U;
        for (int k = 0; k < nocc_; ++k) UD.row(k) *= eps_o(k);
        RowMatXd FL = U.transpose() * UD;
        Eigen::Map<RowMatXd>(F_LMO.data(), nocc_, nocc_) = FL;
    }

    real_t F_LMO_off_max = 0.0;
    for (int i = 0; i < nocc_; ++i)
        for (int j = 0; j < nocc_; ++j)
            if (i != j) F_LMO_off_max = std::max(
                F_LMO_off_max, std::fabs(F_LMO[i * nocc_ + j]));

    if (params.verbose >= 2) {
        std::cout << "[DLPNO-MP2] max |F_LMO[i!=j]| = "
                  << std::scientific << std::setprecision(3) << F_LMO_off_max
                  << " (off-diagonal LMO Fock)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 3. Global PAO matrix and per-LMO domains.
    // -----------------------------------------------------------------------
    const auto t_pao0 = clock::now();
    auto C_pao_global = build_pao_global(h_C_LMO, h_S, nao_, nocc_);

    auto lmo_domains = build_lmo_domains(
        h_C_LMO, h_S, nao_, nocc_, atom_ranges,
        params.t_cut_mkn, params.verbose);
    dt_pao_global += std::chrono::duration<double>(
        clock::now() - t_pao0).count();

    // -----------------------------------------------------------------------
    // 4. Build per-pair PAO setup (PNO-independent invariants).
    // -----------------------------------------------------------------------
    const long long n_pairs_total =
        static_cast<long long>(nocc_) * (nocc_ + 1) / 2;

    std::vector<PairSetup>& setups       = result.setups;
    std::vector<int>&       pair_lookup  = result.pair_lookup;
    setups.reserve(static_cast<size_t>(n_pairs_total));
    pair_lookup.assign(static_cast<size_t>(nocc_) * nocc_, -1);

    long long n_pao_kept_sum = 0;
    real_t E_pao_total = 0.0;

    Eigen::Map<const RowMatXd> Fmat(h_F, nao_, nao_);

    const auto t_pair_setup_0 = clock::now();
    for (int i = 0; i < nocc_; ++i) {
        for (int j = i; j < nocc_; ++j) {
            const bool diag = (i == j);
            const real_t pair_factor = diag ? 1.0 : 2.0;

            const auto pair_aos = merge_ao_index_sets(
                lmo_domains[i].ao_indices, lmo_domains[j].ao_indices);
            auto dom = orthogonalize_pao_domain(
                C_pao_global.data(), h_S, pair_aos, nao_, params.t_cut_do);
            const int n_pao = dom.n_kept;

            PairSetup s;
            s.i = i; s.j = j;
            s.n_pao = n_pao;
            s.F_ii = F_LMO[i * nocc_ + i];
            s.F_jj = F_LMO[j * nocc_ + j];
            s.pair_factor = pair_factor;

            if (n_pao == 0) {
                pair_lookup[i * nocc_ + j] =
                    pair_lookup[j * nocc_ + i] =
                        static_cast<int>(setups.size());
                setups.push_back(std::move(s));
                continue;
            }
            n_pao_kept_sum += n_pao;

            Eigen::Map<const RowMatXd> Cpao(
                dom.C_pao_orth.data(), nao_, n_pao);

            // F_PAO eigendecomposition → semi-canonical PAOs.
            const RowMatXd FCpao = Fmat * Cpao;
            RowMatXd Fpao_pair = Cpao.transpose() * FCpao;
            Fpao_pair = 0.5 * (Fpao_pair + Fpao_pair.transpose());
            auto fdec = eig_sym(Fpao_pair.data(), n_pao);
            const std::vector<real_t>& eps_a = fdec.eigvals;
            Eigen::Map<const RowMatXd> Upao(
                fdec.eigvecs.data(), n_pao, n_pao);
            const RowMatXd C_can_pair = Cpao * Upao;

            // Pair MO ERI block via build_mo_eri.
            const int n_lmo = diag ? 1 : 2;
            const int n_emb = n_lmo + n_pao;
            std::vector<real_t> C_pair(
                static_cast<size_t>(nao_) * n_emb, 0.0);
            for (int mu = 0; mu < nao_; ++mu) {
                C_pair[mu * n_emb + 0] = h_C_LMO[mu * nocc_ + i];
                if (!diag) C_pair[mu * n_emb + 1] = h_C_LMO[mu * nocc_ + j];
                for (int a = 0; a < n_pao; ++a)
                    C_pair[mu * n_emb + n_lmo + a] = C_can_pair(mu, a);
            }
            real_t* d_C_pair = nullptr;
            tracked_cudaMalloc(&d_C_pair,
                static_cast<size_t>(nao_) * n_emb * sizeof(real_t));
            cudaMemcpy(d_C_pair, C_pair.data(),
                static_cast<size_t>(nao_) * n_emb * sizeof(real_t),
                cudaMemcpyHostToDevice);
            real_t* d_eri_mo = eri.build_mo_eri(d_C_pair, n_emb);
            const size_t n_emb4 = static_cast<size_t>(n_emb) * n_emb
                                * n_emb * n_emb;
            std::vector<real_t> h_eri_mo(n_emb4, 0.0);
            cudaMemcpy(h_eri_mo.data(), d_eri_mo,
                       n_emb4 * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(d_C_pair);
            tracked_cudaFree(d_eri_mo);

            const int j_col = diag ? 0 : 1;
            std::vector<real_t> V(
                static_cast<size_t>(n_pao) * n_pao, 0.0);
            for (int a = 0; a < n_pao; ++a)
                for (int b = 0; b < n_pao; ++b) {
                    const size_t idx =
                          static_cast<size_t>(0) * n_emb * n_emb * n_emb
                        + static_cast<size_t>(n_lmo + a) * n_emb * n_emb
                        + static_cast<size_t>(j_col) * n_emb
                        + static_cast<size_t>(n_lmo + b);
                    V[a * n_pao + b] = h_eri_mo[idx];
                }

            // Diagnostic pre-PNO MP2 in semi-canonical PAO basis.
            real_t E_pair_pao = 0.0;
            for (int a = 0; a < n_pao; ++a)
                for (int b = 0; b < n_pao; ++b) {
                    const real_t Vab = V[a * n_pao + b];
                    const real_t Tab = -Vab
                        / (eps_a[a] + eps_a[b] - s.F_ii - s.F_jj);
                    const real_t Tba = -V[b * n_pao + a]
                        / (eps_a[b] + eps_a[a] - s.F_ii - s.F_jj);
                    E_pair_pao += Vab * (2.0 * Tab - Tba);
                }
            E_pao_total += pair_factor * E_pair_pao;

            // Cache invariants for SC-PNO rounds.
            s.C_can_pair.assign(
                static_cast<size_t>(nao_) * n_pao, 0.0);
            Eigen::Map<RowMatXd>(
                s.C_can_pair.data(), nao_, n_pao) = C_can_pair;
            s.eps_a = std::move(fdec.eigvals);
            s.V     = std::move(V);

            pair_lookup[i * nocc_ + j] =
                pair_lookup[j * nocc_ + i] =
                    static_cast<int>(setups.size());
            setups.push_back(std::move(s));
        }
    }
    dt_pair_setup += std::chrono::duration<double>(
        clock::now() - t_pair_setup_0).count();

    // -----------------------------------------------------------------------
    // 5. Initial PairData (round 0): semi-canonical T_pao seeds the PNO
    //    selection; per-pair Sylvester gives the initial Y in W basis.
    // -----------------------------------------------------------------------
    const auto t_round0_0 = clock::now();
    std::vector<PairData>& pairs = result.pairs;
    pairs.assign(setups.size(), PairData{});
    long long n_pno_kept_sum = 0;
    // Per-pair PNO eigendecomp via build_pair_data is independent across pairs
    // — parallelise. T_pao is per-iter local so it's already thread-private.
    #pragma omp parallel for schedule(static) reduction(+:n_pno_kept_sum)
    for (long long idx = 0; idx < static_cast<long long>(setups.size()); ++idx) {
        const PairSetup& s = setups[idx];
        if (s.n_pao == 0) continue;
        std::vector<real_t> T_pao(
            static_cast<size_t>(s.n_pao) * s.n_pao, 0.0);
        for (int a = 0; a < s.n_pao; ++a)
            for (int b = 0; b < s.n_pao; ++b)
                T_pao[a * s.n_pao + b] =
                    -s.V[a * s.n_pao + b]
                    / (s.eps_a[a] + s.eps_a[b] - s.F_ii - s.F_jj);
        build_pair_data(s, T_pao, params.t_cut_pno, params.pno_os_only,
                        nao_, pairs[idx]);
        n_pno_kept_sum += pairs[idx].n_pno;
    }
    dt_round0_pno += std::chrono::duration<double>(
        clock::now() - t_round0_0).count();

    if (params.verbose >= 1) {
        const real_t avg_pao = n_pairs_total > 0
            ? static_cast<real_t>(n_pao_kept_sum) / n_pairs_total : 0.0;
        const real_t avg_pno = n_pairs_total > 0
            ? static_cast<real_t>(n_pno_kept_sum) / n_pairs_total : 0.0;
        std::cout << "[DLPNO-MP2] setup  pairs=" << n_pairs_total
                  << "  avg n_pao=" << std::fixed << std::setprecision(1)
                  << avg_pao
                  << "  avg n_pno=" << avg_pno
                  << "  E(PAO)=" << std::scientific << std::setprecision(8)
                  << E_pao_total << std::endl;
    }

    // -----------------------------------------------------------------------
    // 6. SC-PNO + iterative LMP2.
    // -----------------------------------------------------------------------
    const real_t kFLMOThresh = 1e-14;
    const bool needs_iter = (F_LMO_off_max > kFLMOThresh);
    const int  max_iter   = std::max(1, params.lmp2_max_iter);
    const real_t conv_tol = params.lmp2_conv;
    const int sc_pno_iter = std::max(0, params.sc_pno_iter);
    const int total_rounds = needs_iter ? (sc_pno_iter + 1) : 1;

    LMP2Status last_status;
    if (needs_iter) {
        for (int round = 0; round < total_rounds; ++round) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "MP2 SC-PNO[%d]", round);
            const auto t_iter0 = clock::now();
            last_status = iterate_lmp2(
                setups, pairs, pair_lookup, F_LMO, h_S,
                nocc_, nao_, max_iter, conv_tol,
                params.verbose, tag);
            dt_iterate += std::chrono::duration<double>(
                clock::now() - t_iter0).count();

            if (round + 1 < total_rounds) {
                const auto t_refresh0 = clock::now();
                long long new_n_pno = 0;
                // T_pao is declared inside the loop body so it's per-iter
                // private under OMP; reconstruct_T_pao + build_pair_data are
                // independent per pair.
                #pragma omp parallel for schedule(static) reduction(+:new_n_pno)
                for (long long idx = 0; idx < static_cast<long long>(setups.size()); ++idx) {
                    const PairSetup& s = setups[idx];
                    if (s.n_pao == 0) continue;
                    std::vector<real_t> T_pao;
                    reconstruct_T_pao(s, pairs[idx], T_pao);
                    build_pair_data(
                        s, T_pao, params.t_cut_pno, params.pno_os_only,
                        nao_, pairs[idx]);
                    new_n_pno += pairs[idx].n_pno;
                }
                dt_scpno_refresh += std::chrono::duration<double>(
                    clock::now() - t_refresh0).count();
                if (params.verbose >= 1) {
                    const real_t avg = n_pairs_total > 0
                        ? static_cast<real_t>(new_n_pno) / n_pairs_total
                        : 0.0;
                    std::cout << "[DLPNO-MP2] SC-PNO round " << (round + 1)
                              << " avg n_pno="
                              << std::fixed << std::setprecision(1) << avg
                              << std::endl;
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. Energy summation.
    // -----------------------------------------------------------------------
    real_t E_total = 0.0;
    for (size_t idx = 0; idx < setups.size(); ++idx) {
        const PairSetup& s = setups[idx];
        const PairData&  p = pairs[idx];
        if (p.n_pno == 0) continue;
        Eigen::Map<const RowMatXd> L(p.L.data(), p.n_pno, p.n_pno);
        Eigen::Map<const RowMatXd> Y(p.Y.data(), p.n_pno, p.n_pno);
        real_t e = 0.0;
        for (int a = 0; a < p.n_pno; ++a)
            for (int b = 0; b < p.n_pno; ++b)
                e += L(a, b) * (2.0 * Y(a, b) - Y(b, a));
        E_total += s.pair_factor * e;
    }

    result.E_pao_total   = E_pao_total;
    result.E_pno_total   = E_total;
    result.needs_iter    = needs_iter;
    result.sc_pno_rounds = needs_iter ? total_rounds : 0;

    if (params.verbose >= 1) {
        const double dt = std::chrono::duration<double>(
            clock::now() - t0).count();
        if (needs_iter) {
            std::cout << "[DLPNO-MP2] LMP2 "
                      << (last_status.converged ? "converged" : "MAX_ITER")
                      << " in " << last_status.iters
                      << " iter (last round of " << total_rounds
                      << "), max|R|="
                      << std::scientific << std::setprecision(3)
                      << last_status.max_R << std::endl;
        } else {
            std::cout << "[DLPNO-MP2] LMP2 skipped "
                      << "(max |F_LMO[i!=j]| = "
                      << std::scientific << std::setprecision(2)
                      << F_LMO_off_max << " ≤ tol)" << std::endl;
        }
        std::cout << "[DLPNO-MP2] done   E(PAO,pre-PNO)="
                  << std::scientific << std::setprecision(10) << E_pao_total
                  << "  E(PNO,final)=" << E_total
                  << "  ΔPNO=" << (E_total - E_pao_total)
                  << "  t=" << std::fixed << std::setprecision(2) << dt
                  << "s" << std::endl;
        const double dt_acct = dt_localize + dt_pao_global + dt_pair_setup
                             + dt_round0_pno + dt_scpno_refresh + dt_iterate;
        std::cout << "[DLPNO-MP2-PROF]  total=" << std::fixed
                  << std::setprecision(3) << dt << "s"
                  << "  localize="  << dt_localize
                  << "  pao_dom="   << dt_pao_global
                  << "  pair_setup=" << dt_pair_setup
                  << "  round0_pno=" << dt_round0_pno
                  << "  scpno_refresh=" << dt_scpno_refresh
                  << "  iterate="   << dt_iterate
                  << "  other="     << (dt - dt_acct)
                  << std::endl;
    }

    return result;
}

// ---------------------------------------------------------------------------
//  DLPNOMP2 class — thin wrapper around solve_dlpno_lmp2.
// ---------------------------------------------------------------------------
DLPNOMP2::DLPNOMP2(RHF& rhf, const ERI& eri, DLPNOParams params)
    : rhf_(rhf), eri_(eri), params_(std::move(params)),
      nao_(rhf.get_num_basis()),
      nocc_(rhf.get_num_electrons() / 2),
      natoms_(static_cast<int>(rhf.get_atom_to_basis_range().size()))
{}

real_t DLPNOMP2::compute_energy()
{
    auto res = solve_dlpno_lmp2(rhf_, eri_, params_);
    E_pao_ = res.E_pao_total;
    E_pno_ = res.E_pno_total;
    return res.E_pno_total;
}

// ---------------------------------------------------------------------------
//  ERI wiring — RI is the only supported back-end for DLPNO.
// ---------------------------------------------------------------------------
real_t ERI_RI_RHF::compute_dlpno_mp2() {
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
    DLPNOMP2 driver(rhf_, *this, std::move(p));
    return driver.compute_energy();
}

} // namespace gansu
