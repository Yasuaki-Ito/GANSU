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
#include <memory>

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>

#include "ccsd_lambda.hpp"
#include "device_host_memory.hpp"
#include "dlpno_density.hpp"
#include "dlpno_domain.hpp"
#include "dlpno_lambda.hpp"
#include "dlpno_localizer.hpp"
#include "dlpno_pair_data.hpp"
#include "dlpno_pao.hpp"
#include "dlpno_pno.hpp"
#include "eri.hpp"
#include "oscillator_strength.hpp"
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

// Step S2a — GPU kernel to extract the V block from the full pair-MO ERI.
//
// Layout: d_eri stores the n_emb⁴ ERI in (i, k, j, l) row-major order.
// The V block needed by LMP2 is the (LMO_i, PAO_a, LMO_j, PAO_b) slice:
//     V[a, b] = d_eri[0·n_emb³ + (n_lmo + a)·n_emb² + j_col·n_emb
//                     + (n_lmo + b)]
// with n_lmo = 1 (diag pair) or 2 (off-diag), j_col = 0 / 1 correspondingly.
//
// Replaces the per-pair D2H of n_emb⁴ doubles (~16 MB for n_emb=63 at
// hexamer) followed by a host strided-copy with: per-pair extraction on
// device + D2H of only n_pao² doubles (~30 KB). Eliminates the entire
// n_emb⁴ host buffer and the ~3 s of cumulative D2H bandwidth across
// the 465 pair_setup iterations.
__global__ void extract_V_block_kernel(
    const real_t* __restrict__ d_eri,
    real_t*       __restrict__ d_V,
    int n_emb, int n_lmo, int n_pao, int j_col)
{
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= n_pao || b >= n_pao) return;

    const size_t idx = static_cast<size_t>(n_lmo + a)
                       * static_cast<size_t>(n_emb)
                       * static_cast<size_t>(n_emb)
                     + static_cast<size_t>(j_col)
                       * static_cast<size_t>(n_emb)
                     + static_cast<size_t>(n_lmo + b);
    d_V[a * n_pao + b] = d_eri[idx];
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
    // Frozen-core aware: nocc = active occupied (= total - num_frozen_core_).
    // For --frozen_core none, num_frozen_core_ = 0 → identical to legacy.
    result.nocc = rhf.get_num_active_occ();
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

    // Frozen-core: skip the first num_fc columns of the (nao × nao) MO
    // coefficient matrix and the matching orbital-energy entries. With
    // --frozen_core none, num_fc = 0 and behaviour is identical to legacy.
    //
    // We need TWO occupied buffers:
    //   - C_occ (nao × nocc_active): input to the localizer and downstream
    //     pair (i,j) loops. Frozen orbitals are excluded.
    //   - C_occ_full (nao × nocc_full): input to build_pao_global, which
    //     constructs the PAO basis as (I − D·S) with D = C_occ·C_occ^T.
    //     The PAO projector must remove ALL occupied (frozen + active),
    //     otherwise the frozen orbital leaks into the PAO virtual space
    //     and CCSD picks up spurious frozen↔active T2 excitations.
    const int num_fc = rhf.get_num_frozen_core();
    const int nocc_full = nocc_ + num_fc;

    std::vector<real_t> C_occ_full(
        static_cast<size_t>(nao_) * nocc_full, 0.0);
    std::vector<real_t> C_occ(static_cast<size_t>(nao_) * nocc_, 0.0);
    for (int mu = 0; mu < nao_; ++mu) {
        for (int i = 0; i < nocc_full; ++i)
            C_occ_full[mu * nocc_full + i] = h_C[mu * nao_ + i];
        for (int i = 0; i < nocc_; ++i)
            C_occ[mu * nocc_ + i] = h_C[mu * nao_ + (num_fc + i)];
    }

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
        for (int i = 0; i < nocc_; ++i) eps_o(i) = h_eps[num_fc + i];
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
    // PAO projector must use ALL occupied (frozen + active). Pass
    // C_occ_full instead of h_C_LMO so frozen orbitals are projected out
    // of the PAO virtual space (otherwise CCSD T2 sees spurious
    // frozen↔active excitations and diverges).
    auto C_pao_global = build_pao_global(
        C_occ_full.data(), h_S, nao_, nocc_full);

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
    pair_lookup.assign(static_cast<size_t>(nocc_) * nocc_, -1);

    long long n_pao_kept_sum = 0;
    real_t E_pao_total = 0.0;

    Eigen::Map<const RowMatXd> Fmat(h_F, nao_, nao_);

    // Step 6.3 — Force build_mo_eri to use the replicated single-GPU path
    // for the per-pair pair_setup loop (otherwise the multi-GPU distributed
    // path adds ~5–10 ms NCCL/peer/sync overhead per pair × 465 pairs).
    // Idempotent: a no-op if already replicated; precompute_phase24 / DMET
    // call the same routine later and reuse the same buffers.
    //
    // Step S2b — Additionally detect num_gpus for OpenMP pair-parallel
    // dispatch below: when the RI back-end is distributed and B is
    // replicated, fan out the per-pair build_mo_eri calls across GPUs
    // (same pattern as precompute_phase24_integrals).
    int num_gpus = 1;
#ifdef GANSU_MULTI_GPU
    if (auto* eri_dist = dynamic_cast<const ERI_RI_Distributed_RHF*>(&eri)) {
        if (eri_dist->num_gpus() > 1) {
            const bool ok = const_cast<ERI_RI_Distributed_RHF*>(eri_dist)
                                ->replicate_B_to_all_gpus();
            if (ok) num_gpus = eri_dist->num_gpus();
        }
    }
#endif

    // Step S2b — pre-build (i, j) pair index + pair_lookup so the loop
    // body can write into setups[idx] by index (no order-dependent
    // push_back, which would race under OpenMP).
    std::vector<std::pair<int,int>> pair_ij;
    pair_ij.reserve(static_cast<size_t>(n_pairs_total));
    for (int i = 0; i < nocc_; ++i) {
        for (int j = i; j < nocc_; ++j) {
            const long long idx = static_cast<long long>(pair_ij.size());
            pair_ij.emplace_back(i, j);
            pair_lookup[i * nocc_ + j] = static_cast<int>(idx);
            pair_lookup[j * nocc_ + i] = static_cast<int>(idx);
        }
    }
    setups.assign(static_cast<size_t>(n_pairs_total), PairSetup{});

    // Reductions across the parallel region.
    long long n_pao_kept_sum_par = 0;
    real_t    E_pao_total_par    = 0.0;

    const auto t_pair_setup_0 = clock::now();

    #pragma omp parallel num_threads(num_gpus) \
        reduction(+:n_pao_kept_sum_par, E_pao_total_par)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (num_gpus > 1) cudaSetDevice(tid);

        // Step 6.3c / Step S2a — per-thread workspaces. Grow once per
        // thread to the running max n_emb / n_pao seen by that thread;
        // freed at end of parallel region.
        //
        //   d_C_pair_ws / ws_C_pair_capacity : per-pair C_pair buffer
        //   d_eri_ws    / ws_eri_capacity    : full n_emb⁴ ERI output
        //   d_V_ws      / ws_V_capacity      : extracted V block (n_pao²)
        //
        // Per-pair D2H drops from n_emb⁴ ≈ 16 MB to n_pao² ≈ 30 KB via
        // extract_V_block_kernel.
        real_t* d_C_pair_ws        = nullptr;
        real_t* d_eri_ws           = nullptr;
        real_t* d_V_ws             = nullptr;
        size_t  ws_C_pair_capacity = 0;
        size_t  ws_eri_capacity    = 0;
        size_t  ws_V_capacity      = 0;

        // schedule(dynamic, 1) — per-pair cost varies with n_pao (12-60
        // at hexamer, even wider on cholesterol), so dynamic outperforms
        // static for load balance across GPUs.
        #pragma omp for schedule(dynamic, 1)
        for (long long idx = 0; idx < n_pairs_total; ++idx) {
            const int i = pair_ij[static_cast<size_t>(idx)].first;
            const int j = pair_ij[static_cast<size_t>(idx)].second;
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
                setups[static_cast<size_t>(idx)] = std::move(s);
                continue;
            }
            n_pao_kept_sum_par += n_pao;

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

            // Step S2a — grow per-thread d_C_pair workspace.
            const size_t n_C = static_cast<size_t>(nao_) * n_emb;
            if (n_C > ws_C_pair_capacity) {
                if (d_C_pair_ws) tracked_cudaFree(d_C_pair_ws);
                tracked_cudaMalloc(&d_C_pair_ws, n_C * sizeof(real_t));
                ws_C_pair_capacity = n_C;
            }
            cudaMemcpy(d_C_pair_ws, C_pair.data(),
                n_C * sizeof(real_t), cudaMemcpyHostToDevice);

            // Step 6.3c — grow per-thread d_eri workspace.
            const size_t n_emb4 = static_cast<size_t>(n_emb) * n_emb
                                * n_emb * n_emb;
            if (n_emb4 > ws_eri_capacity) {
                if (d_eri_ws) tracked_cudaFree(d_eri_ws);
                tracked_cudaMalloc(&d_eri_ws, n_emb4 * sizeof(real_t));
                ws_eri_capacity = n_emb4;
            }

            // Step S2a — grow per-thread d_V workspace.
            const size_t n_V = static_cast<size_t>(n_pao) * n_pao;
            if (n_V > ws_V_capacity) {
                if (d_V_ws) tracked_cudaFree(d_V_ws);
                tracked_cudaMalloc(&d_V_ws, n_V * sizeof(real_t));
                ws_V_capacity = n_V;
            }

            eri.build_mo_eri_into(d_C_pair_ws, n_emb, d_eri_ws);

            // Step S2a — extract V block on device, then D2H only n_pao²
            // doubles instead of the full n_emb⁴ ERI tensor.
            const int j_col = diag ? 0 : 1;
            {
                const dim3 block(16, 16);
                const dim3 grid(
                    static_cast<unsigned>((n_pao + 15) / 16),
                    static_cast<unsigned>((n_pao + 15) / 16));
                extract_V_block_kernel<<<grid, block>>>(
                    d_eri_ws, d_V_ws, n_emb, n_lmo, n_pao, j_col);
            }

            std::vector<real_t> V(n_V, 0.0);
            cudaMemcpy(V.data(), d_V_ws,
                       n_V * sizeof(real_t), cudaMemcpyDeviceToHost);

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
            E_pao_total_par += pair_factor * E_pair_pao;

            // Cache invariants for SC-PNO rounds.
            s.C_can_pair.assign(
                static_cast<size_t>(nao_) * n_pao, 0.0);
            Eigen::Map<RowMatXd>(
                s.C_can_pair.data(), nao_, n_pao) = C_can_pair;
            s.eps_a = std::move(fdec.eigvals);
            s.V     = std::move(V);

            setups[static_cast<size_t>(idx)] = std::move(s);
        }

        // Step S2a — release per-thread workspaces (one cudaMalloc per
        // buffer per thread, reused across the thread's pair slice).
        if (d_C_pair_ws) tracked_cudaFree(d_C_pair_ws);
        if (d_eri_ws)    tracked_cudaFree(d_eri_ws);
        if (d_V_ws)      tracked_cudaFree(d_V_ws);
    }

    n_pao_kept_sum = n_pao_kept_sum_par;
    E_pao_total    = E_pao_total_par;

    // Restore device 0 as the active device (parallel region may have
    // left thread 0 on a non-zero device on entry, but threads other
    // than 0 set their own device above — the main thread's device is
    // unchanged for num_gpus > 1 since the master thread is tid 0).
    if (num_gpus > 1) cudaSetDevice(0);
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

        // Per-stage T2 amplitude count reduction + per-pair (i,j) workload
        // distribution. Useful for sizing parallel slabs and estimating
        // load balance. Counts include the (i,j)↔(j,i) symmetry expansion
        // via pair_factor so they are comparable to canonical = nocc²·nvir².
        const int    nvir = nao_ - nocc_;
        const size_t canonical_T2 = static_cast<size_t>(nocc_) * nocc_
                                  * static_cast<size_t>(nvir) * nvir;
        size_t pao_T2 = 0, pno_T2 = 0;
        int n_pao_min = std::numeric_limits<int>::max();
        int n_pao_max = 0;
        int n_pno_min = std::numeric_limits<int>::max();
        int n_pno_max = 0;
        int n_pao_empty = 0, n_pno_empty = 0;
        std::vector<int> n_pao_vec, n_pno_vec_nz;
        n_pao_vec.reserve(setups.size());
        n_pno_vec_nz.reserve(setups.size());
        for (size_t idx = 0; idx < setups.size(); ++idx) {
            const int npao = setups[idx].n_pao;
            const int npno = pairs[idx].n_pno;
            const auto pf  = static_cast<size_t>(setups[idx].pair_factor);
            pao_T2 += pf * static_cast<size_t>(npao) * npao;
            pno_T2 += pf * static_cast<size_t>(npno) * npno;
            n_pao_vec.push_back(npao);
            if (npao == 0) ++n_pao_empty;
            else {
                if (npao < n_pao_min) n_pao_min = npao;
                if (npao > n_pao_max) n_pao_max = npao;
            }
            if (npno == 0) ++n_pno_empty;
            else {
                if (npno < n_pno_min) n_pno_min = npno;
                if (npno > n_pno_max) n_pno_max = npno;
                n_pno_vec_nz.push_back(npno);
            }
        }
        if (n_pao_empty == static_cast<int>(setups.size())) n_pao_min = 0;
        if (n_pno_vec_nz.empty()) n_pno_min = 0;
        auto median_of = [](std::vector<int>& v) -> int {
            if (v.empty()) return 0;
            std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
            return v[v.size()/2];
        };
        const int n_pao_med = median_of(n_pao_vec);
        const int n_pno_med = median_of(n_pno_vec_nz);
        const double r_pao = canonical_T2 ? double(pao_T2) / canonical_T2 : 0.0;
        const double r_pno = canonical_T2 ? double(pno_T2) / canonical_T2 : 0.0;
        const double r_pno_pao = pao_T2 ? double(pno_T2) / double(pao_T2) : 0.0;
        std::cout << "[DLPNO-MP2] T2 amplitude count reduction"
                  << " (full (i,j) count, vs canonical nocc^2*nvir^2):\n"
                  << "  canonical  = " << canonical_T2
                  << "  (nocc=" << nocc_ << ", nvir=" << nvir << ")\n"
                  << "  PAO domain = " << pao_T2
                  << "  ratio=" << std::scientific << std::setprecision(3) << r_pao
                  << "  (1/" << std::fixed << std::setprecision(1)
                  << (r_pao > 0.0 ? 1.0 / r_pao : 0.0) << "x)\n"
                  << "  PNO        = " << pno_T2
                  << "  ratio=" << std::scientific << std::setprecision(3) << r_pno
                  << "  (1/" << std::fixed << std::setprecision(1)
                  << (r_pno > 0.0 ? 1.0 / r_pno : 0.0) << "x of canonical, 1/"
                  << (r_pno_pao > 0.0 ? 1.0 / r_pno_pao : 0.0) << "x of PAO)"
                  << std::endl;
        std::cout << "[DLPNO-MP2] per-pair (i,j) variable count stats"
                  << "  (parallelization reference):\n"
                  << "  n_pao  min=" << n_pao_min
                  << " max=" << n_pao_max
                  << " avg=" << std::fixed << std::setprecision(1) << avg_pao
                  << " med=" << n_pao_med
                  << " empty=" << n_pao_empty << "/" << setups.size() << "\n"
                  << "  n_pno  min=" << n_pno_min
                  << " max=" << n_pno_max
                  << " avg=" << std::fixed << std::setprecision(1) << avg_pno
                  << " med=" << n_pno_med
                  << " empty=" << n_pno_empty << "/" << setups.size()
                  << "  (max workload n_pno^2=" << (n_pno_max * n_pno_max)
                  << ", peak/avg^2="
                  << std::fixed << std::setprecision(1)
                  << (avg_pno > 0.0
                      ? double(n_pno_max * n_pno_max) / (avg_pno * avg_pno)
                      : 0.0)
                  << "x)"
                  << std::endl;
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

    // Multi-GPU pair partition for LMP2 iteration: replicate B (already
    // forced above), then drive iterate_lmp2 with num_gpus instances of
    // PiCacheGpu — each handles a slab of pairs.
    //
    // Option C (2026-05-19): user_explicit_n_gpus tracks whether the user
    // passed `--num_gpus N > 0` explicitly (vs. the default `-1` auto).
    // When explicit, the auto-fallback threshold in iterate_lmp2 is bypassed
    // so the user intent is honoured (required for ResidGpu activation on
    // cholesterol-class systems after S11 Phase 2).
    int lmp2_num_gpus = 1;
    const bool user_explicit_n_gpus = (rhf.get_num_gpus() != -1);
#ifdef GANSU_MULTI_GPU
    if (auto* eri_dist = dynamic_cast<const ERI_RI_Distributed_RHF*>(&eri)) {
        if (eri_dist->num_gpus() > 1) {
            lmp2_num_gpus = eri_dist->num_gpus();
        }
    }
#endif

    LMP2Status last_status;
    if (needs_iter) {
        for (int round = 0; round < total_rounds; ++round) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "MP2 SC-PNO[%d]", round);
            const auto t_iter0 = clock::now();
            last_status = iterate_lmp2(
                setups, pairs, pair_lookup, F_LMO, h_S,
                nocc_, nao_, max_iter, conv_tol,
                params.verbose, tag, lmp2_num_gpus,
                user_explicit_n_gpus);
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
      // Frozen-core aware (= total - num_frozen_core_; 0 if --frozen_core none).
      nocc_(rhf.get_num_active_occ()),
      natoms_(static_cast<int>(rhf.get_atom_to_basis_range().size()))
{}

real_t DLPNOMP2::compute_energy()
{
    auto res = solve_dlpno_lmp2(rhf_, eri_, params_);
    E_pao_ = res.E_pao_total;
    E_pno_ = res.E_pno_total;

    // ----------------------------------------------------------------
    // Sub-phase 1 hook: build DLPNO-MP2 Λ + 1-RDM and report sanity.
    // (DLPNO_Lambda.md §4.2-4.3, Sub-step 1.1-1.5)
    //
    // Production gate: only runs when --dlpno_compute_density true.
    // Print gated by verbose >= 1 (additionally).
    // ----------------------------------------------------------------
    if (rhf_.get_dlpno_compute_density()) {
        compute_dlpno_mp2_lambda(res.pairs);

        // Pull the canonical MO coefficients and AO overlap from the RHF
        // reference (host side; populated after SCF).
        rhf_.get_coefficient_matrix().toHost();
        rhf_.get_overlap_matrix().toHost();
        const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
        const real_t* S_AO   = rhf_.get_overlap_matrix().host_ptr();
        const int nao        = nao_;
        const int n_lmo      = nocc_;
        const int n_can_vir  = nao - nocc_;
        const int nmo        = n_lmo + n_can_vir;

        // Extract canonical virtual block (column-major slice from C).
        // C is row-major [nao × nao] with column = MO index. Virtual columns
        // are nocc_ .. nao-1.
        std::vector<real_t> C_can_vir(
            static_cast<size_t>(nao) * n_can_vir, 0.0);
        for (int mu = 0; mu < nao; ++mu)
            for (int a = 0; a < n_can_vir; ++a)
                C_can_vir[static_cast<size_t>(mu) * n_can_vir + a]
                    = C_full[static_cast<size_t>(mu) * nao + (nocc_ + a)];

        std::vector<real_t> D_mo(
            static_cast<size_t>(nmo) * nmo, 0.0);
        build_dlpno_mp2_1rdm_mo(
            res.setups, res.pairs, res.pair_lookup,
            n_lmo, n_can_vir, S_AO, C_can_vir.data(), nao, D_mo.data());

        if (params_.verbose >= 1) {
            // Sanity diagnostics.
            real_t tr = 0.0;
            for (int p = 0; p < nmo; ++p)
                tr += D_mo[static_cast<size_t>(p) * nmo + p];

            real_t off_diag_max = 0.0;
            for (int p = 0; p < nmo; ++p)
                for (int q = 0; q < nmo; ++q)
                    if (p != q)
                        off_diag_max = std::max(
                            off_diag_max,
                            std::abs(D_mo[static_cast<size_t>(p) * nmo + q]));

            real_t occ_diag_dev = 0.0;
            for (int i = 0; i < n_lmo; ++i)
                occ_diag_dev = std::max(
                    occ_diag_dev,
                    std::abs(D_mo[static_cast<size_t>(i) * nmo + i] - 2.0));

            real_t vir_diag_max = 0.0;
            for (int a = 0; a < n_can_vir; ++a)
                vir_diag_max = std::max(
                    vir_diag_max,
                    std::abs(D_mo[static_cast<size_t>(n_lmo + a) * nmo
                                  + (n_lmo + a)]));

            // Frobenius norms of each block (for PySCF comparison vs
            // reference_mp2.json's dm1_oo_norm / dm1_vv_norm / dm1_ov_norm).
            real_t oo_norm_sq = 0.0, vv_norm_sq = 0.0, ov_norm_sq = 0.0;
            for (int p = 0; p < n_lmo; ++p)
                for (int q = 0; q < n_lmo; ++q) {
                    real_t v = D_mo[static_cast<size_t>(p) * nmo + q];
                    oo_norm_sq += v * v;
                }
            for (int p = 0; p < n_can_vir; ++p)
                for (int q = 0; q < n_can_vir; ++q) {
                    real_t v = D_mo[static_cast<size_t>(n_lmo + p) * nmo
                                    + (n_lmo + q)];
                    vv_norm_sq += v * v;
                }
            for (int p = 0; p < n_lmo; ++p)
                for (int q = 0; q < n_can_vir; ++q) {
                    real_t v = D_mo[static_cast<size_t>(p) * nmo
                                    + (n_lmo + q)];
                    ov_norm_sq += v * v;
                }

            std::cout << "[DLPNO-MP2-LAMBDA] Sub-phase 1 sanity:" << std::endl;
            std::cout << "  tr(D_mo)            = " << std::fixed
                      << std::setprecision(8) << tr
                      << "  (expect " << (2 * n_lmo) << ")" << std::endl;
            std::cout << "  max |D_mo(occ_diag - 2)| = " << std::scientific
                      << std::setprecision(3) << occ_diag_dev << std::endl;
            std::cout << "  max |D_mo(vir_diag)|     = " << std::scientific
                      << std::setprecision(3) << vir_diag_max << std::endl;
            std::cout << "  max |D_mo(off-diag)|     = " << std::scientific
                      << std::setprecision(3) << off_diag_max << std::endl;
            std::cout << "  ||D_mo[oo]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(oo_norm_sq)
                      << std::endl;
            std::cout << "  ||D_mo[vv]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(vv_norm_sq)
                      << std::endl;
            std::cout << "  ||D_mo[ov]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(ov_norm_sq)
                      << "   (Level A MP2: expect 0)" << std::endl;
            std::cout << "  D_mo diagonal:           ";
            for (int p = 0; p < std::min(nmo, 12); ++p) {
                std::cout << " " << std::fixed << std::setprecision(5)
                          << D_mo[static_cast<size_t>(p) * nmo + p];
            }
            if (nmo > 12) std::cout << " ...";
            std::cout << std::endl;

            // ----------------------------------------------------------------
            // Sub-step 1.6b: AO transform + dipole moment.
            //
            // dipole = -Σ_μν D_AO[μ,ν] · ⟨μ|r|ν⟩ + Σ_atom Z_atom · R_atom
            // (electron part negative because of e⁻ charge sign;
            //  origin = (0,0,0), units a.u. (e·Bohr) → convert to Debye)
            // ----------------------------------------------------------------
            std::vector<real_t> D_ao(
                static_cast<size_t>(nao) * nao, 0.0);
            transform_density_mo_to_ao_cpu(nao, C_full, D_mo.data(),
                                           D_ao.data());

            std::vector<real_t> dip_x_ao, dip_y_ao, dip_z_ao;
            const auto& shells = rhf_.get_primitive_shells();
            const auto& cgto_norm = rhf_.get_cgto_normalization_factors();
            const auto& shell_infos = rhf_.get_shell_type_infos();
            compute_ao_dipole_integrals(
                shells.host_ptr(), shells.size(),
                cgto_norm.host_ptr(), nao, shell_infos,
                dip_x_ao, dip_y_ao, dip_z_ao);

            real_t mu_e_x = 0.0, mu_e_y = 0.0, mu_e_z = 0.0;
            for (int mu = 0; mu < nao; ++mu) {
                for (int nu = 0; nu < nao; ++nu) {
                    const real_t Dmn = D_ao[static_cast<size_t>(mu) * nao + nu];
                    const size_t k = static_cast<size_t>(mu) * nao + nu;
                    mu_e_x += Dmn * dip_x_ao[k];
                    mu_e_y += Dmn * dip_y_ao[k];
                    mu_e_z += Dmn * dip_z_ao[k];
                }
            }
            // Electron dipole is negative (electron has charge -e).
            mu_e_x = -mu_e_x;  mu_e_y = -mu_e_y;  mu_e_z = -mu_e_z;

            // Nuclear contribution: + Σ Z_a · R_a (R already in Bohr).
            real_t mu_n_x = 0.0, mu_n_y = 0.0, mu_n_z = 0.0;
            const auto& atoms = rhf_.get_atoms();
            for (size_t a = 0; a < atoms.size(); ++a) {
                const auto& at = atoms.host_ptr()[a];
                const real_t Z = static_cast<real_t>(at.effective_charge);
                mu_n_x += Z * at.coordinate.x;
                mu_n_y += Z * at.coordinate.y;
                mu_n_z += Z * at.coordinate.z;
            }

            // Total in atomic units (e·Bohr), convert to Debye.
            const real_t kAUtoDebye = 2.5417464157;
            real_t dx = (mu_e_x + mu_n_x) * kAUtoDebye;
            real_t dy = (mu_e_y + mu_n_y) * kAUtoDebye;
            real_t dz = (mu_e_z + mu_n_z) * kAUtoDebye;
            real_t dmag = std::sqrt(dx * dx + dy * dy + dz * dz);

            std::cout << "  dipole [Debye]: x=" << std::fixed
                      << std::setprecision(4) << dx
                      << "  y=" << dy << "  z=" << dz
                      << "  |D|=" << dmag << std::endl;
        }
    }

    return res.E_pno_total;
}

// ---------------------------------------------------------------------------
//  ERI wiring — RI is the only supported back-end for DLPNO.
// ---------------------------------------------------------------------------
real_t ERI_RI_RHF::compute_dlpno_mp2() {
    // Cap OpenMP threads for the per-pair CPU loops so the Eigen->OpenBLAS
    // calls inside them stay under OpenBLAS's 128 per-caller-thread buffer
    // limit on many-core machines (see OmpThreadCapGuard). Restored on return.
    OmpThreadCapGuard omp_cap(rhf_.get_dlpno_cpu_threads());
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
