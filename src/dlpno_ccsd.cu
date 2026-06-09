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
#include <random>
#include <vector>

#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bt_pno_backtransform.hpp"  // bt-PNO-STEOM P5a: PNO→canonical back-transform + validation
#include "ip_eom_ccsd_operator.hpp"   // bt-PNO-STEOM stage B B1b: canonical IP-EOM operator (gate)
#include "davidson_solver.hpp"        // bt-PNO-STEOM stage B B1b: Davidson (gate)
#include "dlpno_ip_packing.hpp"       // bt-PNO-STEOM stage B: packed-vector layout
#include "dlpno_ip_eom_projected_operator.hpp"  // bt-PNO-STEOM stage B B1b: projected operator
#include "dlpno_ip_eom_native_operator.hpp"      // bt-PNO-STEOM stage B (a) B-a.1: native σ operator
#include "dlpno_ip_eom_transform.hpp"            // bt-PNO-STEOM stage B (a): in-gate σ2 reference (lift/project)
#include "ea_eom_ccsd_operator.hpp"              // bt-PNO-STEOM stage B (a): canonical EA-EOM operator (gate)
#include "dlpno_ea_packing.hpp"                  // bt-PNO-STEOM stage B (a): EA packed-vector layout
#include "dlpno_ea_eom_projected_operator.hpp"   // bt-PNO-STEOM stage B (a): projected EA operator (gate ref)
#include "dlpno_ea_eom_native_operator.hpp"      // bt-PNO-STEOM stage B (a): native EA σ operator
#include "dlpno_ea_eom_transform.hpp"            // bt-PNO-STEOM stage B (a): EA packed↔canonical transform
#include "ccsd_lambda.hpp"        // transform_density_mo_to_ao_cpu (Sub-phase 3B properties)
#include "device_host_memory.hpp"
#include "dlpno_density.hpp"      // build_dlpno_ccsd_1rdm_mo_closedform (Sub-phase 2)
#include "dlpno_lambda.hpp"       // compute_dlpno_ccsd_lambda_closedform (Sub-phase 2)
#include "dlpno_mp2.hpp"          // solve_dlpno_lmp2 + DLPNOLMP2Result
#include "dlpno_pair_data.hpp"    // PairSetup, PairData
#include "dlpno_phase24_extract.hpp"  // launch_phase24_extract (S7b)
#include "dlpno_eri_gpu.hpp"      // EriBuildGpu (Phase 3.2.7 ERI GPU port)
#include "dlpno_proj_gpu.hpp"     // TripleProjGpu (Phase 3.2.8 T2 proj GPU port)
#include "oscillator_strength.hpp"  // compute_ao_dipole_integrals (Sub-phase 3B properties)
#include "dlpno_t_gpu.hpp"        // TripleTGpu (Phase 3.2.6 GPU port)
#include "dlpno_tno.hpp"          // TNOBuilder, TNOData (Phase 3.2.0)
#include "eri.hpp"
#include "rhf.hpp"

#include <cstring> // memcpy
#include <cstdlib> // std::getenv (bt-PNO P5a.2 validation gate)

namespace gansu {

// Canonical spatial-orbital CCSD (defined in eri_stored.cu) — used by the
// bt-PNO-STEOM P5a.2 back-transform validation gate as the reference T1/T2.
real_t ccsd_spatial_orbital(const real_t* d_eri_ao,
                            const real_t* d_coefficient_matrix,
                            const real_t* d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed,
                            int num_frozen,
                            const real_t* h_fov_active);

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
    // Sub-step 2X.3.1: OVVV for diagonal pairs only — sized [nocc] of
    // [n_pno_ii³]. Filled below inside the per-pair build whenever
    // setup.i == setup.j. Pairs not encountered (e.g. zero-PNO pairs)
    // leave the entry empty.
    out.W_ovvv_diag.assign(res.nocc, {});
    // Sub-step 2X.3.6b: OVVO / OOVV per strong pair (i,j) for the L1·OVVO
    // and L1·OOVV cross-pair source terms of Λ_1 (term 6 of the canonical
    // catalogue). Filled below per pair alongside the existing W_ovvo_i/j.
    // OVVO needs two orientations (i-role and j-role) since (s.i a | b s.j)
    // and (s.j a | b s.i) are genuinely different ERIs.
    out.W_ovvo_lambda.assign(n_pairs, {});
    out.W_ovvo_lambda_alt.assign(n_pairs, {});
    out.W_oovv_lambda.assign(n_pairs, {});
    // Sub-step 2X.3.7a: OVOO per strong pair (i,j) for the OVOO·moo1
    // T2-source term of Λ_1 (term 3). Also shared with term 8 in 2X.3.7c.
    // Two orientations needed.
    out.W_ovoo_lambda.assign(n_pairs, {});
    out.W_ovoo_lambda_alt.assign(n_pairs, {});
    // B-a.6c IP dense-free bare ph-ladder blocks (ovvo/oovv, i/j roles) for the
    // native DLPNO-IP-EOM σ. Filled below per pair alongside the ovov/ovvo W.
    out.W_ovvo_bare_i.assign(n_pairs, {});
    out.W_ovvo_bare_j.assign(n_pairs, {});
    out.W_oovv_bare_i.assign(n_pairs, {});
    out.W_oovv_bare_j.assign(n_pairs, {});

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
    // Save the current OMP thread count so we can restore it after this
    // parallel region. We use `num_threads(num_gpus)` on the pragma itself
    // (which forces exactly num_gpus threads regardless of global state),
    // but `omp_set_num_threads` was being called globally here previously,
    // which clamped subsequent OMP regions (vmeta / dFki / resid in
    // iterate_dlpno_ccsd_t2) to only num_gpus threads — a 5-9× slowdown on
    // multi-CPU hosts. The num_threads clause alone is sufficient; do NOT
    // mutate the process-wide OMP state here.

    #pragma omp parallel num_threads(num_gpus)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        if (num_gpus > 1) cudaSetDevice(tid);

        // Step S7a + S7b — per-thread workspace reuse + pinned host staging.
        //   d_C_ext_ws    : packed (nao × n_emb) C_ext, capacity-grow
        //   d_eri_ws      : full n_emb⁴ MO ERI output buffer, capacity-grow
        //   d_packed_ws   : (S7b) packed device buffer holding the 14 W/T/V
        //                   extracts back-to-back, capacity-grow. The full
        //                   n_emb⁴ tensor never leaves the GPU after S7b.
        //   h_packed_ws   : (S7b) pinned host mirror of d_packed_ws. Size is
        //                   ~17× smaller than the previous h_eri_ws (which
        //                   shadowed the full n_emb⁴ buffer) for cholesterol-
        //                   class molecules, slashing per-pair D2H traffic.
        // All four are freed at the end of the parallel region.
        real_t* d_C_ext_ws         = nullptr;
        real_t* d_eri_ws           = nullptr;
        real_t* d_packed_ws        = nullptr;
        real_t* h_packed_ws        = nullptr;  // pinned
        size_t  ws_C_ext_capacity  = 0;
        size_t  ws_eri_capacity    = 0;
        size_t  ws_packed_capacity = 0;

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

        // Step S7a — grow d_C_ext workspace if this pair is bigger.
        const size_t n_C_ext = static_cast<size_t>(nao) * n_emb;
        if (n_C_ext > ws_C_ext_capacity) {
            if (d_C_ext_ws) tracked_cudaFree(d_C_ext_ws);
            tracked_cudaMalloc(&d_C_ext_ws, n_C_ext * sizeof(real_t));
            ws_C_ext_capacity = n_C_ext;
        }
        cudaMemcpy(d_C_ext_ws, C_ext.data(),
            n_C_ext * sizeof(real_t), cudaMemcpyHostToDevice);

        const size_t n_emb4 =
            static_cast<size_t>(n_emb) * n_emb * n_emb * n_emb;

        // Step S7a — grow d_eri workspace (still n_emb⁴ since build_mo_eri_into
        // writes the full tensor; only the host-side mirror was eliminated by
        // S7b in favour of a much smaller packed extract buffer).
        if (n_emb4 > ws_eri_capacity) {
            if (d_eri_ws) tracked_cudaFree(d_eri_ws);
            tracked_cudaMalloc(&d_eri_ws, n_emb4 * sizeof(real_t));
            ws_eri_capacity = n_emb4;
        }

        // Build MO ERI directly into the per-thread device workspace
        // (avoids the legacy build_mo_eri's per-call cudaMalloc).
        eri.build_mo_eri_into(d_C_ext_ws, n_emb, d_eri_ws);

        // Step S7b — GPU-side extraction of the 14 W/T/V blocks into one
        // packed device buffer, then a single (much smaller) D2H to pinned
        // host. For cholesterol cc-pVDZ this replaces a ~2.9 GB pinned D2H
        // and 13 strided host extract loops with ~170 MB D2H + 14 sequential
        // host memcpys — ~17× smaller transfer and dramatically more
        // cache-friendly unpacking.
        const bool is_diag = (s.i == s.j);
        const Phase24ExtractLayout layout =
            compute_phase24_extract_layout(n_lmo, n_pno, is_diag);

        if (layout.total > ws_packed_capacity) {
            if (d_packed_ws) tracked_cudaFree(d_packed_ws);
            tracked_cudaMalloc(&d_packed_ws, layout.total * sizeof(real_t));
            if (h_packed_ws) cudaFreeHost(h_packed_ws);
            cudaMallocHost(&h_packed_ws, layout.total * sizeof(real_t));
            ws_packed_capacity = layout.total;
        }

        launch_phase24_extract(
            d_eri_ws, d_packed_ws, layout,
            n_emb, n_lmo, n_pno, s.i, s.j, is_diag,
            /*stream=*/0);

        // Synchronous D2H — implicit wait for the kernels above (default
        // stream). After this, h_packed_ws holds all 14 blocks back-to-back
        // at the offsets recorded in `layout`.
        cudaMemcpy(h_packed_ws, d_packed_ws,
            layout.total * sizeof(real_t), cudaMemcpyDeviceToHost);

        // Unpack pinned packed buffer → per-block std::vector<real_t>
        // destinations. memcpy is sequential and cache-friendly compared
        // with the previous strided host gather over the full n_emb⁴ tensor.
        auto copy_block = [&](std::vector<real_t>& dst,
                              size_t off, size_t sz) {
            dst.assign(sz, 0.0);
            if (sz > 0) {
                std::memcpy(dst.data(), h_packed_ws + off,
                            sz * sizeof(real_t));
            }
        };

        copy_block(out.T_pair[idx],       layout.off_T_pair,   layout.sz_T_pair);
        copy_block(out.W_pair[idx],       layout.off_W_pair,   layout.sz_W_pair);
        copy_block(out.W_oooo[idx],       layout.off_W_oooo,   layout.sz_W_oooo);
        copy_block(out.W_ovov_i[idx],     layout.off_W_ovov_i, layout.sz_W_ovov);
        copy_block(out.W_ovov_j[idx],     layout.off_W_ovov_j, layout.sz_W_ovov);
        copy_block(out.W_ovvo_i[idx],     layout.off_W_ovvo_i, layout.sz_W_ovov);
        copy_block(out.W_ovvo_j[idx],     layout.off_W_ovvo_j, layout.sz_W_ovov);
        copy_block(out.V_ovov_pair[idx],  layout.off_V_ovov,   layout.sz_V_ovov);
        if (is_diag) {
            copy_block(out.W_ovvv_diag[s.i],
                       layout.off_W_ovvv_diag, layout.sz_W_ovvv_diag);
        }
        copy_block(out.W_ovvo_lambda[idx],     layout.off_W_ovvo_lambda,     layout.sz_pno2);
        copy_block(out.W_ovvo_lambda_alt[idx], layout.off_W_ovvo_lambda_alt, layout.sz_pno2);
        copy_block(out.W_oovv_lambda[idx],     layout.off_W_oovv_lambda,     layout.sz_pno2);
        copy_block(out.W_ovoo_lambda[idx],     layout.off_W_ovoo_lambda,     layout.sz_ovoo);
        copy_block(out.W_ovoo_lambda_alt[idx], layout.off_W_ovoo_lambda_alt, layout.sz_ovoo);
        // B-a.6c IP dense-free bare ph-ladder (size n_lmo·n_pno² = sz_W_ovov).
        copy_block(out.W_ovvo_bare_i[idx], layout.off_W_ovvo_bare_i, layout.sz_W_ovov);
        copy_block(out.W_ovvo_bare_j[idx], layout.off_W_ovvo_bare_j, layout.sz_W_ovov);
        copy_block(out.W_oovv_bare_i[idx], layout.off_W_oovv_bare_i, layout.sz_W_ovov);
        copy_block(out.W_oovv_bare_j[idx], layout.off_W_oovv_bare_j, layout.sz_W_ovov);

        (void)s;  // s used above; suppress -Wunused if conditional.
    }

    // Step S7a + S7b — release per-thread workspaces (one alloc per buffer
    // per thread, reused across the thread's pair slice).
    if (d_C_ext_ws)  tracked_cudaFree(d_C_ext_ws);
    if (d_eri_ws)    tracked_cudaFree(d_eri_ws);
    if (d_packed_ws) tracked_cudaFree(d_packed_ws);
    if (h_packed_ws) cudaFreeHost(h_packed_ws);
    }  // end omp parallel
    return out;
}

} // anonymous namespace

DLPNOCCSD::DLPNOCCSD(RHF& rhf, const ERI& eri, DLPNOParams params)
    : rhf_(rhf),
      eri_(eri),
      params_(std::move(params)),
      nao_(rhf.get_num_basis()),
      // Frozen-core aware (= total - num_frozen_core_; 0 if --frozen_core none).
      nocc_(rhf.get_num_active_occ())
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

    // [gap-PROF] classification + f_ia + T1 iteration (host) — previously
    // unaccounted between solve_dlpno_lmp2 and precompute_phase24_integrals.
    const auto t_classt1_0 = prof_clock::now();

    int n_strong = 0, n_weak = 0, n_empty = 0;
    real_t E_strong = 0.0, E_weak = 0.0;
    size_t T2_strong = 0, T2_weak = 0;
    std::vector<int> n_pno_strong, n_pno_weak;
    n_pno_strong.reserve(res.setups.size());
    n_pno_weak.reserve(res.setups.size());

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
        const auto   pf     = static_cast<size_t>(s.pair_factor);
        const size_t n2     = pf * static_cast<size_t>(p.n_pno) * p.n_pno;
        if (std::fabs(e_intrinsic) < params_.t_cut_pairs) {
            ++n_weak;  E_weak  += e_pair; T2_weak  += n2;
            n_pno_weak.push_back(p.n_pno);
        } else {
            ++n_strong; E_strong += e_pair; T2_strong += n2;
            n_pno_strong.push_back(p.n_pno);
        }
    }

    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD] pair classification: "
                  << "strong=" << n_strong
                  << "  weak=" << n_weak
                  << (n_empty > 0
                      ? "  (empty=" + std::to_string(n_empty) + ")" : "")
                  << std::endl;

        // Per-group T2 amplitude counts (full (i,j) count via pair_factor)
        // + per-pair n_pno stats. Strong pairs go through full CCSD T2
        // iteration; weak pairs are frozen at MP2 amplitudes.
        auto stat_line = [](const char* tag, int n_pairs,
                            size_t t2, const std::vector<int>& v) {
            int vmin = 0, vmax = 0, vmed = 0;
            double vavg = 0.0;
            if (!v.empty()) {
                auto vv = v;
                std::nth_element(vv.begin(), vv.begin() + vv.size()/2, vv.end());
                vmed = vv[vv.size()/2];
                long long sum = 0;
                vmin = vv[0]; vmax = vv[0];
                for (int x : vv) {
                    sum += x;
                    if (x < vmin) vmin = x;
                    if (x > vmax) vmax = x;
                }
                vavg = double(sum) / v.size();
            }
            std::cout << "  " << tag
                      << "  pairs=" << n_pairs
                      << "  T2 elem=" << t2
                      << "  n_pno: min=" << vmin
                      << " max=" << vmax
                      << " avg=" << std::fixed << std::setprecision(1) << vavg
                      << " med=" << vmed;
            if (n_pairs > 0) {
                std::cout << "  workload max/avg^2="
                          << std::fixed << std::setprecision(1)
                          << (vavg > 0.0 ? double(vmax * vmax) / (vavg * vavg)
                                         : 0.0)
                          << "x";
            }
            std::cout << std::endl;
        };
        stat_line("strong (full CCSD T2):", n_strong, T2_strong, n_pno_strong);
        if (n_weak > 0)
            stat_line("weak   (MP2 only)   :", n_weak,   T2_weak,   n_pno_weak);
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

    // Independent per LMO i (disjoint T1[i]/f_ia[i] writes, build_f_ia pure).
    #pragma omp parallel for schedule(dynamic)
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

        // Each i reads the T1_old snapshot and writes its own T1[i] (disjoint),
        // and project_t1 is pure ⇒ the i-loop is independent. Parallelise it
        // (was serial host = the bulk of classify+f_ia+T1 on large systems).
        #pragma omp parallel for schedule(dynamic) reduction(max:r_max)
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
    const double dt_classt1 = std::chrono::duration<double>(
        prof_clock::now() - t_classt1_0).count();
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD-PROF] classify+f_ia+T1 = "
                  << std::fixed << std::setprecision(3)
                  << dt_classt1 << " s" << std::endl;
    }

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

    // Multi-GPU pair partition for CCSD T2 iteration. Picked up from the
    // distributed RI back-end if it has replicated B (each device can build
    // pi_cache / R_ph independently from its own B copy).
    //
    // Option C (2026-05-19): user_explicit_n_gpus tracks whether the user
    // passed `--num_gpus N > 0` explicitly (vs. the default `-1` auto).
    // When explicit, the auto-fallback threshold in iterate_dlpno_ccsd_t2
    // is bypassed so the user intent is honoured (required for ResidGpu
    // activation on cholesterol-class systems after S11 Phase 2).
    int t2_num_gpus = 1;
    const bool user_explicit_n_gpus = (rhf_.get_num_gpus() != -1);
#ifdef GANSU_MULTI_GPU
    if (auto* eri_dist = dynamic_cast<const ERI_RI_Distributed_RHF*>(&eri_)) {
        if (eri_dist->num_gpus() > 1) {
            const bool ok = const_cast<ERI_RI_Distributed_RHF*>(eri_dist)
                                ->replicate_B_to_all_gpus();
            if (ok) t2_num_gpus = eri_dist->num_gpus();
        }
    }
#endif
    if (params_.verbose >= 1) {
        std::cout << "[DLPNO-CCSD-PROF] T2 iter num_gpus=" << t2_num_gpus
                  << std::endl;
    }

    // (T) re-use: snapshot the converged LMP2 amplitudes BEFORE the CCSD
    // dressing below mutates res.pairs[].Y in place, so the (T) driver can
    // reuse this LMP2 pair state instead of re-solving LMP2 from scratch.
    std::vector<std::vector<real_t>> lmp2_Y_snap;
    if (capture_lmp2_) {
        lmp2_Y_snap.resize(res.pairs.size());
        for (size_t idx = 0; idx < res.pairs.size(); ++idx)
            lmp2_Y_snap[idx] = res.pairs[idx].Y;
    }

    // Phase 2 (CCSD sparse barS) — per-LMO Mulliken centroids for the
    // distance-based coupling screen. Only built when the sparse path is on
    // (cheap: O(nocc·nao²)). centroid(i) = Σ_A R_A · q_A^i with Mulliken pop
    // q_A^i = Σ_{μ∈A} C_μi (S·C_i)_μ. Passed to iterate_dlpno_ccsd_t2; the
    // distance cutoff is read there from GANSU_DLPNO_CCSD_BARS_DIST.
    std::vector<real_t> lmo_centroids;
    {
        const char* e = std::getenv("GANSU_DLPNO_CCSD_BARS_SPARSE");
        if (e && e[0] == '1' && static_cast<int>(res.C_LMO.size())
                                  == nao_ * nocc_) {
            lmo_centroids.assign(static_cast<size_t>(nocc_) * 3, 0.0);
            const auto& atoms = rhf_.get_atoms();
            const auto& abr   = rhf_.get_atom_to_basis_range();
            const int natoms  = static_cast<int>(abr.size());
            const real_t* Clmo = res.C_LMO.data();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nocc_; ++i) {
                std::vector<real_t> SC(nao_, 0.0);
                for (int mu = 0; mu < nao_; ++mu) {
                    const real_t* Srow = h_S + static_cast<size_t>(mu) * nao_;
                    real_t s = 0.0;
                    for (int nu = 0; nu < nao_; ++nu)
                        s += Srow[nu] * Clmo[static_cast<size_t>(nu) * nocc_ + i];
                    SC[mu] = s;
                }
                real_t cx = 0, cy = 0, cz = 0, qtot = 0;
                for (int a = 0; a < natoms; ++a) {
                    const int b0 = static_cast<int>(abr[a].start_index);
                    const int b1 = static_cast<int>(abr[a].end_index);
                    real_t qA = 0.0;
                    for (int mu = b0; mu < b1; ++mu)
                        qA += Clmo[static_cast<size_t>(mu) * nocc_ + i] * SC[mu];
                    const auto& R = atoms[a].coordinate;
                    cx += qA * R.x; cy += qA * R.y; cz += qA * R.z; qtot += qA;
                }
                if (qtot != 0.0) { cx /= qtot; cy /= qtot; cz /= qtot; }
                lmo_centroids[3*i+0] = cx;
                lmo_centroids[3*i+1] = cy;
                lmo_centroids[3*i+2] = cz;
            }
        }
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
            &phase24, t2_num_gpus, user_explicit_n_gpus,
            lmo_centroids.empty() ? nullptr : lmo_centroids.data());
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
                  << std::scientific << std::setprecision(15) << E_strong
                  << "\n[DLPNO-CCSD]   E(weak-pair MP2)    = " << E_weak
                  << "\n[DLPNO-CCSD]   E(T1 contribution)  = " << E_T1
                  << "\n[DLPNO-CCSD]   E(total CCSD corr)  = " << E_total
                  << std::endl;
    }

    // ----------------------------------------------------------------
    // bt-PNO-STEOM P5a.2 validation gate (env GANSU_DLPNO_BT_VALIDATE=1).
    // Back-transform the converged DLPNO-CCSD T2/T1 (per-pair PNO basis) to
    // canonical MO basis and compare element-wise to a canonical CCSD run on
    // the same RHF reference. At no truncation (full domains, t_cut_pno→0,
    // t_cut_pairs→0) DLPNO is just a rotation of the canonical space, so the
    // back-transformed amplitudes must match canonical CCSD to ~1e-6. Off by
    // default → zero production cost. See STEOM.md §21.4.
    // ----------------------------------------------------------------
    const bool bt_validate = [](){ const char* e = std::getenv("GANSU_DLPNO_BT_VALIDATE");
                                    return e && e[0] == '1'; }();
    if (bt_validate || rhf_.collect_dlpno_bt()) {
        // [gap-PROF] PNO→canonical back-transform (host reconstruct of the
        // nocc²·nvir² canonical T2/T1 for the STEOM hand-off). Largest single
        // item in the previously-unaccounted stage-1 gap.
        const auto t_bt_0 = prof_clock::now();
        double t_bt_fn_s = -1.0;   // back-transform function time (split from collect)
        {
            const int num_fc        = rhf_.get_num_frozen_core();
            const int num_occ_total = nocc_ + num_fc;
            const int nvir          = nao_ - num_occ_total;

            // Canonical virtual block from the full C [nao × nao] (columns
            // [num_occ_total, nao) — accounts for frozen core).
            rhf_.get_coefficient_matrix().toHost();
            const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
            std::vector<real_t> C_vir(static_cast<size_t>(nao_) * nvir, 0.0);
            for (int mu = 0; mu < nao_; ++mu)
                for (int a = 0; a < nvir; ++a)
                    C_vir[static_cast<size_t>(mu) * nvir + a] =
                        C_full[static_cast<size_t>(mu) * nao_ + (num_occ_total + a)];

            // PNO → canonical back-transform (T1 from the per-LMO PAO vector).
            BTAmplitudes bt = bt_pno_to_canonical(
                res, res.U_loc, C_vir, h_S, nao_, nvir, T1, /*include_t1=*/true);
            t_bt_fn_s = std::chrono::duration<double>(
                prof_clock::now() - t_bt_0).count();

            // Hybrid DLPNO-STEOM (P5b): hand the canonical amplitudes to the
            // STEOM driver via the RHF object. (Sets use_dlpno_amplitudes_.)
            if (rhf_.collect_dlpno_bt()) {
                // bt is re-read below only inside the bt_validate canonical-CCSD
                // comparison; set_dlpno_bt_amplitudes takes its argument by value
                // (copying the ~225 MB canonical T2), so hand ownership over with
                // a move when no validation needs the local copy.
                if (bt_validate) rhf_.set_dlpno_bt_amplitudes(bt);
                else             rhf_.set_dlpno_bt_amplitudes(std::move(bt));
                // stage B (a): the native per-pair σ reads the converged per-pair
                // PNO integral blocks. phase24 is still needed below (line ~929,
                // Λ dressing when compute_density is on), so copy rather than
                // move. set_dlpno_res takes its argument by value then moves it
                // into dlpno_res_, so passing the lvalue `res` copies the whole
                // DLPNOLMP2Result (all per-pair PNO blocks + barS_cache, ~GBs) —
                // this copy dominated the stage-1 collect cost. `res` is only
                // read again below inside the env-gated *_VALIDATE blocks and the
                // (flag-gated) compute_density block; when NONE of those are
                // active (production STEOM) `res` is dead after this point, so we
                // hand ownership over with a move and skip the copy entirely.
                // phase24's local copy is re-read below only by the
                // compute_density block (line ~1444); otherwise move it into res.
                if (rhf_.get_dlpno_compute_density()) res.phase24 = phase24;
                else                                  res.phase24 = std::move(phase24);
                const bool need_res_after = [&]() {
                    auto on = [](const char* n) {
                        const char* e = std::getenv(n);
                        return e && e[0] == '1';
                    };
                    return on("GANSU_DLPNO_IP_VALIDATE")
                        || on("GANSU_DLPNO_IP_NATIVE_VALIDATE")
                        || on("GANSU_DLPNO_EA_NATIVE_VALIDATE")
                        || rhf_.get_dlpno_compute_density();
                }();
                if (need_res_after)
                    rhf_.set_dlpno_res(res);             // copy (res reused below)
                else
                    rhf_.set_dlpno_res(std::move(res));  // move (no GB-scale copy)
            }

            // Canonical-CCSD element-wise comparison (validation only; the
            // collect path above does not need it). Runs a full canonical CCSD,
            // so it is skipped unless GANSU_DLPNO_BT_VALIDATE=1.
            if (bt_validate) {
            // Canonical CCSD reference on the same RHF (MO ERI from RI B).
            const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
            const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
            real_t* d_mo_eri = eri_.build_mo_eri(d_C, nao_);
            real_t* d_t1c = nullptr;
            real_t* d_t2c = nullptr;
            const real_t E_can = ccsd_spatial_orbital(
                /*d_eri_ao=*/nullptr, d_C, d_eps, nao_, num_occ_total,
                /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
                &d_t1c, &d_t2c, d_mo_eri, num_fc, /*h_fov_active=*/nullptr);

            const size_t t2n = static_cast<size_t>(nocc_) * nocc_ * nvir * nvir;
            const size_t t1n = static_cast<size_t>(nocc_) * nvir;
            std::vector<real_t> t2c(t2n), t1c(t1n);
            cudaMemcpy(t2c.data(), d_t2c, t2n * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(t1c.data(), d_t1c, t1n * sizeof(real_t), cudaMemcpyDeviceToHost);

            real_t max_dt2 = 0.0, nrm_d = 0.0, nrm_ref = 0.0, max_dt1 = 0.0;
            for (size_t k = 0; k < t2n; ++k) {
                const real_t d = bt.T2[k] - t2c[k];
                max_dt2  = std::max(max_dt2, std::fabs(d));
                nrm_d   += d * d;
                nrm_ref += t2c[k] * t2c[k];
            }
            for (size_t k = 0; k < t1n; ++k)
                max_dt1 = std::max(max_dt1, std::fabs(bt.T1[k] - t1c[k]));
            const real_t rel_f =
                (nrm_ref > 0.0) ? std::sqrt(nrm_d / nrm_ref) : std::sqrt(nrm_d);

            std::cout << "[bt-PNO P5a.2] back-transform vs canonical CCSD  (nocc="
                      << nocc_ << ", nvir=" << nvir << ", frozen=" << num_fc << ")\n"
                      << "  E(DLPNO corr)     = " << std::fixed << std::setprecision(10)
                      << E_total << " Ha\n"
                      << "  E(canonical CCSD) = " << E_can << " Ha   (ΔE = "
                      << std::scientific << std::setprecision(3) << (E_total - E_can) << ")\n"
                      << "  max|ΔT2| = " << max_dt2
                      << "   ||ΔT2||_F/||T2||_F = " << rel_f
                      << "   max|ΔT1| = " << max_dt1 << std::endl;

            tracked_cudaFree(d_t1c);
            tracked_cudaFree(d_t2c);
            tracked_cudaFree(d_mo_eri);
            }  // if (bt_validate)
        }
        const double dt_bt = std::chrono::duration<double>(
            prof_clock::now() - t_bt_0).count();
        if (params_.verbose >= 1) {
            std::cout << "[DLPNO-CCSD-PROF] bt_pno_to_canonical+collect = "
                      << std::fixed << std::setprecision(3)
                      << dt_bt << " s  (back-transform=" << t_bt_fn_s
                      << "  collect=" << (dt_bt - t_bt_fn_s) << ")" << std::endl;
        }
    }

    // ----------------------------------------------------------------
    // bt-PNO-STEOM stage B B1b validation gate (env GANSU_DLPNO_IP_VALIDATE=1).
    // Build the canonical IP-EOM operator from DLPNO back-transformed T1/T2, run
    // (i) the canonical IP-EOM Davidson (= P1 reference) and (ii) the Galerkin-
    // projected DLPNO-IP-EOM Davidson (DLPNOIPEOMProjectedOperator wrapping the
    // same canonical σ). At no truncation (n_pno=nvir) the projection is a
    // bijection → the lowest IP roots must match the canonical reference.
    // Requires --frozen_core none. Off by default → zero production cost.
    // ----------------------------------------------------------------
    {
        const char* ip_env = std::getenv("GANSU_DLPNO_IP_VALIDATE");
        const bool ip_validate = (ip_env && ip_env[0] == '1');
        const int num_fc = rhf_.get_num_frozen_core();
        if (ip_validate && num_fc != 0) {
            std::cout << "[bt-PNO B1b] IP validate gate requires --frozen_core none; skipping."
                      << std::endl;
        } else if (ip_validate) {
            const int nvir = nao_ - nocc_;

            // Canonical virtual block + back-transformed canonical T1/T2.
            rhf_.get_coefficient_matrix().toHost();
            const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
            std::vector<real_t> C_vir(static_cast<size_t>(nao_) * nvir, 0.0);
            for (int mu = 0; mu < nao_; ++mu)
                for (int a = 0; a < nvir; ++a)
                    C_vir[static_cast<size_t>(mu) * nvir + a] =
                        C_full[static_cast<size_t>(mu) * nao_ + (nocc_ + a)];
            BTAmplitudes bt = bt_pno_to_canonical(
                res, res.U_loc, C_vir, h_S, nao_, nvir, T1, /*include_t1=*/true);

            const size_t t1n = static_cast<size_t>(nocc_) * nvir;
            const size_t t2n = static_cast<size_t>(nocc_) * nocc_ * nvir * nvir;
            real_t* d_t1 = nullptr;
            real_t* d_t2 = nullptr;
            tracked_cudaMalloc(&d_t1, t1n * sizeof(real_t));
            tracked_cudaMalloc(&d_t2, t2n * sizeof(real_t));
            cudaMemcpy(d_t1, bt.T1.data(), t1n * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_t2, bt.T2.data(), t2n * sizeof(real_t), cudaMemcpyHostToDevice);

            const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
            const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
            real_t* d_mo_eri = eri_.build_mo_eri(d_C, nao_);

            // Canonical IP-EOM operator (takes ownership of d_t1/d_t2).
            IPEOMCCSDOperator ip_op(d_mo_eri, d_eps, d_t1, d_t2, nocc_, nvir, nao_);
            tracked_cudaFree(d_mo_eri);

            const int k = std::min(3, ip_op.dimension());
            DavidsonConfig cfg;
            cfg.num_eigenvalues       = k;
            cfg.convergence_threshold = 1e-7;
            cfg.max_iterations        = 200;
            cfg.use_preconditioner    = true;
            cfg.symmetric             = false;
            cfg.min_eigenvalue        = 0.0;
            cfg.verbose               = 0;

            // (i) canonical reference roots.
            cfg.max_subspace_size = std::min(ip_op.dimension(), std::max(80, 20 * k));
            DavidsonSolver s_can(ip_op, cfg);
            s_can.solve();
            std::vector<real_t> eig_can = s_can.get_eigenvalues();

            // (ii) projected DLPNO-IP-EOM roots.
            rhf_.get_orbital_energies().toHost();
            const real_t* h_eps = rhf_.get_orbital_energies().host_ptr();
            std::vector<real_t> eps_o(nocc_);
            for (int i = 0; i < nocc_; ++i) eps_o[i] = h_eps[i];
            const DLPNOIPPacking pack = build_ip_packing(res);
            DLPNOIPEOMProjectedOperator proj_op(
                ip_op, res, pack, res.U_loc, C_vir, h_S, nao_, nvir, eps_o);
            DavidsonConfig cfg2 = cfg;
            cfg2.max_subspace_size = std::min(proj_op.dimension(), std::max(80, 20 * k));
            DavidsonSolver s_proj(proj_op, cfg2);
            s_proj.solve();
            std::vector<real_t> eig_proj = s_proj.get_eigenvalues();

            std::cout << "[bt-PNO B1b] DLPNO-IP-EOM (projected) vs canonical IP-EOM  (nocc="
                      << nocc_ << ", nvir=" << nvir
                      << ", packed_dim=" << proj_op.dimension()
                      << ", canon_dim=" << ip_op.dimension() << ")\n";
            real_t max_d = 0.0;
            for (int r = 0; r < k; ++r) {
                const real_t d = eig_proj[r] - eig_can[r];
                max_d = std::max(max_d, std::fabs(d));
                std::cout << "   root " << r << "   canon=" << std::fixed << std::setprecision(8)
                          << eig_can[r] << "   dlpno=" << eig_proj[r]
                          << "   Δ=" << std::scientific << std::setprecision(3) << d << "\n";
            }
            std::cout << "   max|Δω| = " << std::scientific << std::setprecision(3) << max_d
                      << "  (no truncation → expect ~Davidson tol; truncation → DLPNO error)"
                      << std::endl;
            // ip_op destructor frees d_t1/d_t2.
        }
    }

    // ----------------------------------------------------------------
    // bt-PNO-STEOM stage B (a) B-a.1 native-σ validation gate
    // (env GANSU_DLPNO_IP_NATIVE_VALIDATE=1). Compares the σ1 (1h) rows of the
    // native per-pair operator (DLPNOIPEOMNativeOperator) against the validated
    // project-up reference (DLPNOIPEOMProjectedOperator) on random packed
    // vectors. At no truncation the two must agree to ~1e-10 on the σ1 block.
    // (σ2 is diagonal-only in the native operator at B-a.1, so σ2 rows are NOT
    // compared here — the native σ2 terms and their gates land in B-a.2+.)
    // Self-contained (rebuilds ip_op/pack); leaves the IP_VALIDATE block above
    // untouched. Requires --frozen_core none. Off by default → zero prod cost.
    // ----------------------------------------------------------------
    {
        const char* nat_env = std::getenv("GANSU_DLPNO_IP_NATIVE_VALIDATE");
        const bool nat_validate = (nat_env && nat_env[0] == '1');
        const int num_fc = rhf_.get_num_frozen_core();
        if (nat_validate && num_fc != 0) {
            std::cout << "[bt-PNO B-a.1] native-σ gate requires --frozen_core none; skipping."
                      << std::endl;
        } else if (nat_validate) {
            const int nvir = nao_ - nocc_;

            rhf_.get_coefficient_matrix().toHost();
            const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
            std::vector<real_t> C_vir(static_cast<size_t>(nao_) * nvir, 0.0);
            for (int mu = 0; mu < nao_; ++mu)
                for (int a = 0; a < nvir; ++a)
                    C_vir[static_cast<size_t>(mu) * nvir + a] =
                        C_full[static_cast<size_t>(mu) * nao_ + (nocc_ + a)];
            BTAmplitudes bt = bt_pno_to_canonical(
                res, res.U_loc, C_vir, h_S, nao_, nvir, T1, /*include_t1=*/true);

            const size_t t1n = static_cast<size_t>(nocc_) * nvir;
            const size_t t2n = static_cast<size_t>(nocc_) * nocc_ * nvir * nvir;
            real_t* d_t1 = nullptr;
            real_t* d_t2 = nullptr;
            tracked_cudaMalloc(&d_t1, t1n * sizeof(real_t));
            tracked_cudaMalloc(&d_t2, t2n * sizeof(real_t));
            cudaMemcpy(d_t1, bt.T1.data(), t1n * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_t2, bt.T2.data(), t2n * sizeof(real_t), cudaMemcpyHostToDevice);

            const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
            const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
            real_t* d_mo_eri = eri_.build_mo_eri(d_C, nao_);
            IPEOMCCSDOperator ip_op(d_mo_eri, d_eps, d_t1, d_t2, nocc_, nvir, nao_);
            tracked_cudaFree(d_mo_eri);

            // Borrow Lvv (T2) + Wovoo (T1) + Loo (T3/T4) + Woooo (T5) + eri_oovv/t2 (T8)
            // for the in-gate σ2 reference.
            std::vector<real_t> h_Lvv(static_cast<size_t>(nvir) * nvir);
            std::vector<real_t> h_Wovoo(static_cast<size_t>(nocc_) * nvir * nocc_ * nocc_);
            std::vector<real_t> h_Loo(static_cast<size_t>(nocc_) * nocc_);
            std::vector<real_t> h_Woooo(static_cast<size_t>(nocc_) * nocc_ * nocc_ * nocc_);
            std::vector<real_t> h_oovv(static_cast<size_t>(nocc_) * nocc_ * nvir * nvir);
            std::vector<real_t> h_t2(static_cast<size_t>(nocc_) * nocc_ * nvir * nvir);
            std::vector<real_t> h_Wovvo(static_cast<size_t>(nocc_) * nvir * nvir * nocc_);
            std::vector<real_t> h_Wovov(static_cast<size_t>(nocc_) * nvir * nocc_ * nvir);
            cudaMemcpy(h_Lvv.data(),   ip_op.get_Lvv_device(),   h_Lvv.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wovoo.data(), ip_op.get_Wovoo_device(), h_Wovoo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Loo.data(),   ip_op.get_Loo_device(),   h_Loo.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Woooo.data(), ip_op.get_Woooo_device(), h_Woooo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_oovv.data(),  ip_op.get_eri_oovv_device(), h_oovv.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_t2.data(),    ip_op.get_t2_device(),       h_t2.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wovvo.data(), ip_op.get_Wovvo_device(), h_Wovvo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wovov.data(), ip_op.get_Wovov_device(), h_Wovov.size() * sizeof(real_t), cudaMemcpyDeviceToHost);

            rhf_.get_orbital_energies().toHost();
            const real_t* h_eps = rhf_.get_orbital_energies().host_ptr();
            std::vector<real_t> eps_o(nocc_);
            for (int i = 0; i < nocc_; ++i) eps_o[i] = h_eps[i];

            const DLPNOIPPacking pack = build_ip_packing(res);
            DLPNOIPEOMProjectedOperator proj_op(
                ip_op, res, pack, res.U_loc, C_vir, h_S, nao_, nvir, eps_o);
            // Stage 5 multi-GPU: thread the device count into the validation-gate
            // operator too (the production site eri_stored_ip_eom_ccsd.cu wires it
            // for stage 2, but this in-process B-a.4 gate is where the multi-GPU
            // path is actually exercised — stage 2 STEOM needs --num_gpus 1 for
            // CIS-NTO). Default 1 unless the GPU residency + MULTI env are on.
            DLPNOIPEOMNativeOperator native_op(
                ip_op, res, pack, res.U_loc, C_vir, h_S, nao_, nvir, eps_o,
                rhf_.get_num_gpus());

            const int pdim = pack.total_dim;
            real_t* d_x  = nullptr;
            real_t* d_yn = nullptr;
            real_t* d_yp = nullptr;
            tracked_cudaMalloc(&d_x,  static_cast<size_t>(pdim) * sizeof(real_t));
            tracked_cudaMalloc(&d_yn, static_cast<size_t>(pdim) * sizeof(real_t));
            tracked_cudaMalloc(&d_yp, static_cast<size_t>(pdim) * sizeof(real_t));

            std::mt19937_64 rng(20260522ULL);
            std::normal_distribution<real_t> gauss(0.0, 1.0);
            const int n_probe = 3;
            const size_t vstride = static_cast<size_t>(nvir);
            real_t max_d1 = 0.0, max_d2 = 0.0, max_dfull = 0.0;
            std::vector<real_t> x(pdim), yn(pdim), yp(pdim);

            // B-a.6c dressed-path reference: the native dressed ph-ladder works in
            // PNO space, which inserts the projector P_ij = U^(ij) U^(ij)ᵀ on the
            // T6/T7 SOURCE amplitude's virtual index (one-sided barS). Full PNO
            // (P=I) is unreachable — PNOs truncate to the pair-density rank, so U
            // is never square — so "dressed == (b) at full PNO" is a dead gate.
            // Instead, when GANSU_DLPNO_NATIVE_DRESSED=1, project the in-gate σ2
            // reference's T6/T7 sources by P_ij too → native == reference bit-exact
            // at ANY truncation (validates the PNO restructure + one-sided barS +
            // occ-role mapping). T2 (own-orientation, exact) and T1/T3/T4/T5/T8
            // (still dense in the native) need no projection. Note: vfull (native
            // vs (b) projected) will NOT be ~0 in the dressed case — that is the
            // genuine truncation difference; the σ2-ref Δ (v2) is the gate.
            const char* dr_env = std::getenv("GANSU_DLPNO_NATIVE_DRESSED");
            const bool dressed_ref = (dr_env && dr_env[0] == '1');
            std::vector<std::vector<real_t>> Ppair;  // [n_pairs] of [nvir²] (P_ij), empty if screened
            if (dressed_ref) {
                Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao_, nvir);
                Eigen::Map<const RowMatXd> Sm(h_S, nao_, nao_);
                const RowMatXd CvtS = Cv.transpose() * Sm;        // [nvir × nao]
                const int n_pairs = static_cast<int>(res.pairs.size());
                Ppair.assign(n_pairs, {});
                for (int idx = 0; idx < n_pairs; ++idx) {
                    const int n = pack.n_pno[idx];
                    if (n == 0) continue;
                    Eigen::Map<const RowMatXd> barQ(res.pairs[idx].bar_Q.data(), nao_, n);
                    const RowMatXd U = CvtS * barQ;               // [nvir × n]
                    const RowMatXd P = U * U.transpose();         // [nvir × nvir]
                    Ppair[idx].assign(static_cast<size_t>(nvir) * nvir, 0.0);
                    for (int a = 0; a < nvir; ++a)
                        for (int b = 0; b < nvir; ++b)
                            Ppair[idx][static_cast<size_t>(a) * nvir + b] = P(a, b);
                }
            }

            for (int t = 0; t < n_probe; ++t) {
                for (int p = 0; p < pdim; ++p) x[p] = gauss(rng);
                cudaMemcpy(d_x, x.data(), static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyHostToDevice);
                native_op.apply(d_x, d_yn);
                proj_op.apply(d_x, d_yp);
                cudaMemcpy(yn.data(), d_yn, static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(yp.data(), d_yp, static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyDeviceToHost);

                // σ1 (1h): native vs (b) reference (both = canonical σ1 on lifted r2).
                real_t v1 = 0.0;
                for (int i = 0; i < nocc_; ++i) v1 = std::max(v1, std::fabs(yn[i] - yp[i]));
                max_d1 = std::max(max_d1, v1);

                // FULL native σ vs (b) projected reference (all rows). Now that
                // every σ2 term is implemented, the two operators must agree on
                // the entire packed vector — the decisive end-to-end gate.
                real_t vfull = 0.0;
                for (int p = 0; p < pdim; ++p) vfull = std::max(vfull, std::fabs(yn[p] - yp[p]));
                max_dfull = std::max(max_dfull, vfull);

                // σ2 terms T1+T2: native vs in-gate reference = project_down of the
                // canonical (T1+T2) on the lifted r2. Both sides use the same
                // U^(ij)/Lvv/Wovoo so this is bit-exact even with residual PNO
                // truncation (validates the native per-pair machinery, the occ
                // U_loc rotation of Wovoo, offsets, and both orientations).
                //   T2:  sig2c[I,J,a] += Σ_d Lvv[a,d] r2c[I,J,d]
                //   T1:  sig2c[I,J,a] += -Σ_k Wovoo[k,a,I,J] r1[k]
                //   T3:  sig2c[I,J,a] += -Σ_K Loo[K,I] r2c[K,J,a]
                //   T4:  sig2c[I,J,a] += -Σ_L Loo[L,J] r2c[I,L,a]
                //   T5:  sig2c[I,J,a] += +Σ_KL Woooo[K,L,I,J] r2c[K,L,a]
                //   T8:  tmp_c[c]      =  Σ_KLD (2 oovv[L,K,D,c]-oovv[K,L,D,c]) r2c[K,L,D]
                //        sig2c[I,J,a] += -Σ_c tmp_c[c] t2[I,J,c,a]
                const real_t* r1 = x.data();   // canonical 1h block
                std::vector<real_t> packed_r2(x.begin() + nocc_, x.end());
                std::vector<real_t> r2c = ip_packed_r2_to_canonical(
                    res, pack, res.U_loc, C_vir, h_S, nao_, nvir, packed_r2);
                std::vector<real_t> tmp_c(nvir, 0.0);
                for (int K = 0; K < nocc_; ++K)
                    for (int L = 0; L < nocc_; ++L)
                        for (int D = 0; D < nvir; ++D) {
                            const real_t r = r2c[(static_cast<size_t>(K) * nocc_ + L) * vstride + D];
                            for (int c = 0; c < nvir; ++c)
                                tmp_c[c] += (2.0 * h_oovv[((static_cast<size_t>(L) * nocc_ + K) * nvir + D) * nvir + c]
                                                 - h_oovv[((static_cast<size_t>(K) * nocc_ + L) * nvir + D) * nvir + c]) * r;
                        }
                std::vector<real_t> sig2c(r2c.size(), 0.0);
                #pragma omp parallel for
                for (int IJ = 0; IJ < nocc_ * nocc_; ++IJ) {
                    const int I = IJ / nocc_, J = IJ % nocc_;
                    // Dressed: project the ph-ladder T6/T7 sources by P_ij (see
                    // the Ppair note above) so the reference matches the native
                    // PNO-space ph-ladder bit-exact at any truncation.
                    const int idx_t = dressed_ref
                        ? res.pair_lookup[static_cast<size_t>(I) * nocc_ + J] : -1;
                    const bool proj = (idx_t >= 0 && !Ppair[idx_t].empty());
                    std::vector<real_t> Pr_Im, Pr_mI, Pr_mJ;  // [nocc·nvir], filled iff proj
                    if (proj) {
                        const real_t* P = Ppair[idx_t].data();
                        Pr_Im.assign(static_cast<size_t>(nocc_) * nvir, 0.0);
                        Pr_mI.assign(static_cast<size_t>(nocc_) * nvir, 0.0);
                        Pr_mJ.assign(static_cast<size_t>(nocc_) * nvir, 0.0);
                        for (int m = 0; m < nocc_; ++m)
                            for (int d = 0; d < nvir; ++d) {
                                real_t sIm = 0.0, smI = 0.0, smJ = 0.0;
                                for (int e = 0; e < nvir; ++e) {
                                    const real_t p = P[static_cast<size_t>(d) * nvir + e];
                                    sIm += p * r2c[(static_cast<size_t>(I) * nocc_ + m) * vstride + e];
                                    smI += p * r2c[(static_cast<size_t>(m) * nocc_ + I) * vstride + e];
                                    smJ += p * r2c[(static_cast<size_t>(m) * nocc_ + J) * vstride + e];
                                }
                                Pr_Im[static_cast<size_t>(m) * nvir + d] = sIm;
                                Pr_mI[static_cast<size_t>(m) * nvir + d] = smI;
                                Pr_mJ[static_cast<size_t>(m) * nvir + d] = smJ;
                            }
                    }
                    for (int a = 0; a < nvir; ++a) {
                        real_t s = 0.0;
                        for (int d = 0; d < nvir; ++d)            // T2
                            s += h_Lvv[static_cast<size_t>(a) * nvir + d]
                                 * r2c[static_cast<size_t>(IJ) * vstride + d];
                        for (int k = 0; k < nocc_; ++k)           // T1
                            s -= h_Wovoo[(static_cast<size_t>(k) * nvir + a) * nocc_ * nocc_
                                         + static_cast<size_t>(I) * nocc_ + J] * r1[k];
                        for (int K = 0; K < nocc_; ++K)           // T3
                            s -= h_Loo[static_cast<size_t>(K) * nocc_ + I]
                                 * r2c[(static_cast<size_t>(K) * nocc_ + J) * vstride + a];
                        for (int L = 0; L < nocc_; ++L)           // T4
                            s -= h_Loo[static_cast<size_t>(L) * nocc_ + J]
                                 * r2c[(static_cast<size_t>(I) * nocc_ + L) * vstride + a];
                        for (int K = 0; K < nocc_; ++K)           // T5
                            for (int L = 0; L < nocc_; ++L)
                                s += h_Woooo[((static_cast<size_t>(K) * nocc_ + L) * nocc_
                                             + I) * nocc_ + J]
                                     * r2c[(static_cast<size_t>(K) * nocc_ + L) * vstride + a];
                        for (int c = 0; c < nvir; ++c)            // T8b
                            s -= tmp_c[c] * h_t2[((static_cast<size_t>(I) * nocc_ + J) * nvir + c) * nvir + a];
                        for (int m = 0; m < nocc_; ++m)           // T6/T7 (ph-ladder)
                            for (int d = 0; d < nvir; ++d) {
                                const real_t r_Im = proj ? Pr_Im[static_cast<size_t>(m) * nvir + d]
                                    : r2c[(static_cast<size_t>(I) * nocc_ + m) * vstride + d];
                                const real_t r_mI = proj ? Pr_mI[static_cast<size_t>(m) * nvir + d]
                                    : r2c[(static_cast<size_t>(m) * nocc_ + I) * vstride + d];
                                const real_t r_mJ = proj ? Pr_mJ[static_cast<size_t>(m) * nvir + d]
                                    : r2c[(static_cast<size_t>(m) * nocc_ + J) * vstride + d];
                                s += h_Wovvo[((static_cast<size_t>(m) * nvir + a) * nvir + d) * nocc_ + J]
                                     * (2.0 * r_Im - r_mI);                         // T6
                                s -= h_Wovov[((static_cast<size_t>(m) * nvir + a) * nocc_ + J) * nvir + d] * r_Im;  // T7 first
                                s -= h_Wovov[((static_cast<size_t>(m) * nvir + a) * nocc_ + I) * nvir + d] * r_mJ;  // T7 second
                            }
                        sig2c[static_cast<size_t>(IJ) * vstride + a] = s;
                    }
                }
                std::vector<real_t> ref_packed = ip_canonical_r2_to_packed(
                    res, pack, res.U_loc, C_vir, h_S, nao_, nvir, sig2c);
                real_t v2 = 0.0;
                for (int p = 0; p < pdim - nocc_; ++p)
                    v2 = std::max(v2, std::fabs(yn[nocc_ + p] - ref_packed[p]));
                max_d2 = std::max(max_d2, v2);

                std::cout << "[bt-PNO B-a.4]   probe " << t
                          << "  max|Δσ_full(native vs projected)| = "
                          << std::scientific << std::setprecision(3) << vfull
                          << "  (σ2 in-gate ref Δ = " << v2 << ")\n";
            }
            std::cout << "[bt-PNO B-a.4] FULL native-σ vs (b) projected reference  (nocc=" << nocc_
                      << ", nvir=" << nvir << ", packed_dim=" << pdim
                      << ")  max|Δσ_full| = " << std::scientific << std::setprecision(3) << max_dfull
                      << "  (σ1 Δ=" << max_d1 << ", σ2-ref Δ=" << max_d2 << ")" << std::endl;
            if (dressed_ref)
                std::cout << "[bt-PNO B-a.6c] DRESSED gate (localizer none): the bit-exact metric is "
                             "σ2-ref Δ = " << std::scientific << std::setprecision(3) << max_d2
                          << " (in-gate ref P-projects the T6/T7 sources by P_ij; expect ~1e-13). "
                             "max|Δσ_full| vs (b) = " << max_dfull << " is the GENUINE PNO truncation "
                             "difference (P≠I), NOT a gate." << std::endl;
            else
                std::cout << "  (all σ1+σ2 terms; expect ~1e-12)" << std::endl;

            tracked_cudaFree(d_x);
            tracked_cudaFree(d_yn);
            tracked_cudaFree(d_yp);
            // ip_op destructor frees d_t1/d_t2.
        }
    }

    // ----------------------------------------------------------------
    // bt-PNO-STEOM stage B (a) B-EA.1 native-σ validation gate
    // (env GANSU_DLPNO_EA_NATIVE_VALIDATE=1). Compares the σ1 (1p) rows of the
    // native per-pair EA operator (DLPNOEAEOMNativeOperator) against the
    // projected reference on random packed vectors. (σ2 is diagonal-only at
    // B-EA.1, not compared.) Self-contained; requires --frozen_core none.
    // ----------------------------------------------------------------
    {
        const char* nat_env = std::getenv("GANSU_DLPNO_EA_NATIVE_VALIDATE");
        const bool nat_validate = (nat_env && nat_env[0] == '1');
        const int num_fc = rhf_.get_num_frozen_core();
        if (nat_validate && num_fc != 0) {
            std::cout << "[bt-PNO B-EA.1] native-σ gate requires --frozen_core none; skipping."
                      << std::endl;
        } else if (nat_validate) {
            const int nvir = nao_ - nocc_;

            rhf_.get_coefficient_matrix().toHost();
            const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
            std::vector<real_t> C_vir(static_cast<size_t>(nao_) * nvir, 0.0);
            for (int mu = 0; mu < nao_; ++mu)
                for (int a = 0; a < nvir; ++a)
                    C_vir[static_cast<size_t>(mu) * nvir + a] =
                        C_full[static_cast<size_t>(mu) * nao_ + (nocc_ + a)];
            BTAmplitudes bt = bt_pno_to_canonical(
                res, res.U_loc, C_vir, h_S, nao_, nvir, T1, /*include_t1=*/true);

            const size_t t1n = static_cast<size_t>(nocc_) * nvir;
            const size_t t2n = static_cast<size_t>(nocc_) * nocc_ * nvir * nvir;
            real_t* d_t1 = nullptr;
            real_t* d_t2 = nullptr;
            tracked_cudaMalloc(&d_t1, t1n * sizeof(real_t));
            tracked_cudaMalloc(&d_t2, t2n * sizeof(real_t));
            cudaMemcpy(d_t1, bt.T1.data(), t1n * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_t2, bt.T2.data(), t2n * sizeof(real_t), cudaMemcpyHostToDevice);

            const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
            const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
            real_t* d_mo_eri = eri_.build_mo_eri(d_C, nao_);
            EAEOMCCSDOperator ea_op(d_mo_eri, d_eps, d_t1, d_t2, nocc_, nvir, nao_);
            tracked_cudaFree(d_mo_eri);

            // Borrow Lvv (T_Lvv) + Wvvvo (T_r1) + Loo/Wovvo/Wovov (cross-pair) for
            // the in-gate σ2 reference.
            std::vector<real_t> h_Lvv(static_cast<size_t>(nvir) * nvir);
            std::vector<real_t> h_Wvvvo(static_cast<size_t>(nvir) * nvir * nvir * nocc_);
            std::vector<real_t> h_Loo(static_cast<size_t>(nocc_) * nocc_);
            std::vector<real_t> h_Wovvo(static_cast<size_t>(nocc_) * nvir * nvir * nocc_);
            std::vector<real_t> h_Wovov(static_cast<size_t>(nocc_) * nvir * nocc_ * nvir);
            cudaMemcpy(h_Lvv.data(),   ea_op.get_Lvv_device(),   h_Lvv.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wvvvo.data(), ea_op.get_Wvvvo_device(), h_Wvvvo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            std::vector<real_t> h_ovov(static_cast<size_t>(nocc_) * nvir * nocc_ * nvir);
            std::vector<real_t> h_t2(static_cast<size_t>(nocc_) * nocc_ * nvir * nvir);
            std::vector<real_t> h_Wvvvv(static_cast<size_t>(nvir) * nvir * nvir * nvir);
            cudaMemcpy(h_Loo.data(),   ea_op.get_Loo_device(),   h_Loo.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wovvo.data(), ea_op.get_Wovvo_device(), h_Wovvo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wovov.data(), ea_op.get_Wovov_device(), h_Wovov.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ovov.data(),  ea_op.get_eri_ovov_device(), h_ovov.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_t2.data(),    ea_op.get_t2_device(),       h_t2.size()   * sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_Wvvvv.data(), ea_op.get_Wvvvv_device(),    h_Wvvvv.size() * sizeof(real_t), cudaMemcpyDeviceToHost);

            rhf_.get_orbital_energies().toHost();
            const real_t* h_eps = rhf_.get_orbital_energies().host_ptr();
            std::vector<real_t> eps_v(nvir);
            for (int a = 0; a < nvir; ++a) eps_v[a] = h_eps[nocc_ + a];

            const DLPNOEAPacking pack = build_ea_packing(res, nvir);
            DLPNOEAEOMProjectedOperator proj_op(
                ea_op, res, pack, res.U_loc, C_vir, h_S, nao_, eps_v);
            // Stage 5 multi-GPU: thread the device count into the validation-gate
            // operator too (mirror of the IP B-a.4 gate above; the production site
            // eri_stored_ea_eom_ccsd.cu wires stage 2, but this in-process B-EA.4
            // gate is where the multi-GPU path is actually exercised).
            DLPNOEAEOMNativeOperator native_op(
                ea_op, res, pack, res.U_loc, C_vir, h_S, nao_, nvir, eps_v,
                rhf_.get_num_gpus());

            const int pdim = pack.total_dim;
            real_t* d_x  = nullptr;
            real_t* d_yn = nullptr;
            real_t* d_yp = nullptr;
            tracked_cudaMalloc(&d_x,  static_cast<size_t>(pdim) * sizeof(real_t));
            tracked_cudaMalloc(&d_yn, static_cast<size_t>(pdim) * sizeof(real_t));
            tracked_cudaMalloc(&d_yp, static_cast<size_t>(pdim) * sizeof(real_t));

            std::mt19937_64 rng(20260522ULL);
            std::normal_distribution<real_t> gauss(0.0, 1.0);
            const int n_probe = 3;
            const size_t vstride = static_cast<size_t>(nvir);
            real_t max_d1 = 0.0, max_d2 = 0.0, max_dfull = 0.0;
            std::vector<real_t> x(pdim), yn(pdim), yp(pdim);
            for (int t = 0; t < n_probe; ++t) {
                for (int p = 0; p < pdim; ++p) x[p] = gauss(rng);
                cudaMemcpy(d_x, x.data(), static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyHostToDevice);
                native_op.apply(d_x, d_yn);
                proj_op.apply(d_x, d_yp);
                cudaMemcpy(yn.data(), d_yn, static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(yp.data(), d_yp, static_cast<size_t>(pdim) * sizeof(real_t), cudaMemcpyDeviceToHost);

                // σ1 (1p): native vs (b) reference.
                real_t v1 = 0.0;
                for (int a = 0; a < nvir; ++a) v1 = std::max(v1, std::fabs(yn[a] - yp[a]));
                max_d1 = std::max(max_d1, v1);

                // FULL native σ vs (b) projected reference (all rows) — native σ
                // is now complete (σ1 + all σ2 terms), so the two must agree.
                real_t vfull = 0.0;
                for (int p = 0; p < pdim; ++p) vfull = std::max(vfull, std::fabs(yn[p] - yp[p]));
                max_dfull = std::max(max_dfull, vfull);

                // σ2 terms T_Lvv + T_r1: native vs in-gate reference = project_down of
                // the canonical (T_Lvv + T_r1) on the lifted r2. Same U^(ii)/Lvv/Wvvvo
                // ⇒ bit-exact even with PNO truncation. r2c layout (J*nvir+a)*nvir+b.
                //   T_Lvv_a: sig2c[J,a,b] += Σ_c Lvv[a,c] r2c[J,c,b]
                //   T_Lvv_b: sig2c[J,a,b] += Σ_d Lvv[b,d] r2c[J,a,d]
                //   T_r1   : sig2c[J,a,b] += Σ_c Wvvvo[a,b,c,J] r1[c]
                const real_t* r1 = x.data();   // canonical 1p block
                std::vector<real_t> packed_r2(x.begin() + nvir, x.end());
                std::vector<real_t> r2c = ea_packed_r2_to_canonical(
                    res, pack, res.U_loc, C_vir, h_S, nao_, packed_r2);
                // T_tmp pre-stage: tmp[K] = Σ_{l,C,D}(2 ovov[K,C,l,D]-ovov[K,D,l,C]) r2c[l,C,D]
                std::vector<real_t> tmp(nocc_, 0.0);
                for (int K = 0; K < nocc_; ++K) {
                    real_t s = 0.0;
                    for (int l = 0; l < nocc_; ++l)
                        for (int c = 0; c < nvir; ++c)
                            for (int d = 0; d < nvir; ++d)
                                s += (2.0 * h_ovov[((static_cast<size_t>(K) * nvir + c) * nocc_ + l) * nvir + d]
                                          - h_ovov[((static_cast<size_t>(K) * nvir + d) * nocc_ + l) * nvir + c])
                                     * r2c[(static_cast<size_t>(l) * nvir + c) * vstride + d];
                    tmp[K] = s;
                }
                std::vector<real_t> sig2c(r2c.size(), 0.0);
                #pragma omp parallel for
                for (int J = 0; J < nocc_; ++J)
                    for (int a = 0; a < nvir; ++a)
                        for (int b = 0; b < nvir; ++b) {
                            real_t s = 0.0;
                            for (int c = 0; c < nvir; ++c)            // T_Lvv_a + T_Lvv_b
                                s += h_Lvv[static_cast<size_t>(a) * nvir + c]
                                     * r2c[(static_cast<size_t>(J) * nvir + c) * vstride + b]
                                   + h_Lvv[static_cast<size_t>(b) * nvir + c]
                                     * r2c[(static_cast<size_t>(J) * nvir + a) * vstride + c];
                            for (int c = 0; c < nvir; ++c)            // T_r1
                                s += h_Wvvvo[((static_cast<size_t>(a) * nvir + b) * nvir + c) * nocc_ + J] * r1[c];
                            for (int l = 0; l < nocc_; ++l) {        // cross-pair (Loo + ph-ladder)
                                s -= h_Loo[static_cast<size_t>(l) * nocc_ + J]
                                     * r2c[(static_cast<size_t>(l) * nvir + a) * vstride + b];   // T_Loo
                                for (int d = 0; d < nvir; ++d)        // T_ph1
                                    s += (2.0 * h_Wovvo[((static_cast<size_t>(l) * nvir + b) * nvir + d) * nocc_ + J]
                                              - h_Wovov[((static_cast<size_t>(l) * nvir + b) * nocc_ + J) * nvir + d])
                                         * r2c[(static_cast<size_t>(l) * nvir + a) * vstride + d];
                                for (int c = 0; c < nvir; ++c)        // T_ph2
                                    s -= h_Wovov[((static_cast<size_t>(l) * nvir + a) * nocc_ + J) * nvir + c]
                                         * r2c[(static_cast<size_t>(l) * nvir + c) * vstride + b];
                                for (int c = 0; c < nvir; ++c)        // T_ph3
                                    s -= h_Wovvo[((static_cast<size_t>(l) * nvir + b) * nvir + c) * nocc_ + J]
                                         * r2c[(static_cast<size_t>(l) * nvir + c) * vstride + a];
                            }
                            for (int K = 0; K < nocc_; ++K)          // T_tmp
                                s -= tmp[K] * h_t2[((static_cast<size_t>(K) * nocc_ + J) * nvir + a) * nvir + b];
                            for (int c = 0; c < nvir; ++c)           // T_vvvv
                                for (int d = 0; d < nvir; ++d)
                                    s += h_Wvvvv[((static_cast<size_t>(a) * nvir + b) * nvir + c) * nvir + d]
                                         * r2c[(static_cast<size_t>(J) * nvir + c) * vstride + d];
                            sig2c[(static_cast<size_t>(J) * nvir + a) * vstride + b] = s;
                        }
                std::vector<real_t> ref_packed = ea_canonical_r2_to_packed(
                    res, pack, res.U_loc, C_vir, h_S, nao_, sig2c);
                real_t v2 = 0.0;
                for (int p = 0; p < pdim - nvir; ++p)
                    v2 = std::max(v2, std::fabs(yn[nvir + p] - ref_packed[p]));
                max_d2 = std::max(max_d2, v2);

                std::cout << "[bt-PNO B-EA.4]   probe " << t
                          << "  max|Δσ_full(native vs projected)| = "
                          << std::scientific << std::setprecision(3) << vfull
                          << "  (σ2 in-gate ref Δ = " << v2 << ")\n";
            }
            std::cout << "[bt-PNO B-EA.4] FULL native-σ vs (b) projected reference  (nocc=" << nocc_
                      << ", nvir=" << nvir << ", packed_dim=" << pdim
                      << ")  max|Δσ_full| = " << std::scientific << std::setprecision(3) << max_dfull
                      << "  (σ1 Δ=" << max_d1 << ", σ2-ref Δ=" << max_d2
                      << ")  (all σ1+σ2 terms; expect ~1e-12)" << std::endl;

            tracked_cudaFree(d_x);
            tracked_cudaFree(d_yn);
            tracked_cudaFree(d_yp);
            // ea_op destructor frees d_t1/d_t2.
        }
    }

    // ----------------------------------------------------------------
    // Sub-phase 2 strategy (a) hook: build DLPNO-CCSD Λ + 1-RDM via the
    // closed-form approximation (Y_lambda = 2 Y - Y^T applied to CCSD T2,
    // Λ_1 = 0, plus explicit T1 contribution to ov/vo blocks).
    //
    // Production gate: dlpno_compute_density. Print gated by verbose >= 1.
    // See c:\Users\yasuaki\Dropbox\AQUA\DLPNO_Lambda.md §12 for the
    // strategy choice rationale and the route to upgrading to the exact
    // Datta 2016 Λ residual (strategy X).
    // ----------------------------------------------------------------
    if (rhf_.get_dlpno_compute_density()) {
        // Sub-step 2X.1 (default, LMP2 limit): closed-form Λ_2 = 2 Y - Y^T
        // converges in 1 sweep at strict mode and agrees with canonical
        // CCSD oo/vv blocks to ~1e-5 (PM-localised baseline). The closed-
        // form dipole however sits 6.3% off canonical because Λ_1 = 0 and
        // the Λ_2 F-eff dressing is missing.
        //
        // Sub-step 2X.2c (opt-in via --dlpno_lambda_full_dressing 1): turn
        // on the full Path A dressing (phase24-based dF_ki + per-pair DF).
        // Aims to bring the dipole error well below 1%. The historical 2X.2a
        // intra-pair-only attempt degraded oo/vv to ||D[oo]||_F = 8.7e-3
        // (vs 2.5e-5 for closed-form); the full Phase24 path restores
        // balance by mirroring the T-iteration dressing exactly. Validated
        // here in the strict mode sentinel test (Phase2X_2c_*).
        const bool use_full_dressing = rhf_.get_dlpno_lambda_full_dressing();
        std::vector<std::vector<real_t>> Lambda1;
        DLPNOLambdaStatus lam_status = iterate_dlpno_ccsd_lambda(
            res.setups, res.pairs, Lambda1, T1, res.pair_lookup,
            res.F_LMO, h_S, nocc_, nao_,
            params_.lmp2_max_iter, params_.lmp2_conv,
            /*enable_dressing=*/use_full_dressing,
            params_.verbose, "DLPNO-Λ",
            /*phase24=*/use_full_dressing ? &phase24 : nullptr,
            /*num_gpus=*/1);

        rhf_.get_coefficient_matrix().toHost();
        const real_t* C_full = rhf_.get_coefficient_matrix().host_ptr();
        const int n_lmo     = nocc_;
        const int n_can_vir = nao_ - nocc_;
        const int nmo       = n_lmo + n_can_vir;

        // Extract canonical virtual block from full C [nao × nao].
        std::vector<real_t> C_can_vir(
            static_cast<size_t>(nao_) * n_can_vir, 0.0);
        for (int mu = 0; mu < nao_; ++mu)
            for (int a = 0; a < n_can_vir; ++a)
                C_can_vir[static_cast<size_t>(mu) * n_can_vir + a]
                    = C_full[static_cast<size_t>(mu) * nao_ + (nocc_ + a)];

        std::vector<real_t> D_mo(
            static_cast<size_t>(nmo) * nmo, 0.0);
        // Sub-step 2X.3.4: pass Λ_1 amplitudes to the 1-RDM builder so the
        // D[ov]/D[vo] block picks up the L1 contribution. Without the flag
        // Lambda1 remains size-0 inner vectors (Sub-step 2X.3.0 scaffold)
        // and the call collapses to the closed-form Λ_1 = 0 path.
        build_dlpno_ccsd_1rdm_mo_closedform(
            res.setups, res.pairs, res.pair_lookup, T1,
            n_lmo, n_can_vir, h_S, C_can_vir.data(), nao_, D_mo.data(),
            Lambda1);

        if (params_.verbose >= 1) {
            // Sanity: trace + block norms (mirror DLPNO-MP2 sanity layout).
            real_t tr = 0.0;
            for (int p = 0; p < nmo; ++p)
                tr += D_mo[static_cast<size_t>(p) * nmo + p];

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

            std::cout << "[DLPNO-CCSD-LAMBDA] Sub-phase 2 (closed-form approx):"
                      << std::endl;
            std::cout << "  tr(D_mo)            = " << std::fixed
                      << std::setprecision(8) << tr
                      << "  (expect " << (2 * n_lmo) << ")" << std::endl;
            std::cout << "  ||D_mo[oo]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(oo_norm_sq)
                      << std::endl;
            std::cout << "  ||D_mo[vv]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(vv_norm_sq)
                      << std::endl;
            std::cout << "  ||D_mo[ov]||_F           = " << std::fixed
                      << std::setprecision(6) << std::sqrt(ov_norm_sq)
                      << "   (closed-form: T1 contribution)" << std::endl;
            std::cout << "  D_mo diagonal:           ";
            for (int p = 0; p < std::min(nmo, 12); ++p) {
                std::cout << " " << std::fixed << std::setprecision(5)
                          << D_mo[static_cast<size_t>(p) * nmo + p];
            }
            if (nmo > 12) std::cout << " ...";
            std::cout << std::endl;

            // ----------------------------------------------------------------
            // Sub-phase 3 (B): properties wire-in for whole-molecule
            // DLPNO-CCSD. Mirror of the DLPNO-MP2 dipole + Mulliken path
            // (DLPNOMP2::compute_energy in src/dlpno_mp2.cu).
            //
            // dipole = -Σ_μν D_AO[μ,ν] · ⟨μ|r|ν⟩ + Σ_atom Z_atom · R_atom
            // Mulliken q_A = Z_A - Σ_{μ ∈ A, ν} D_AO[μ,ν] · S_AO[μ,ν]
            // ----------------------------------------------------------------
            std::vector<real_t> D_ao(
                static_cast<size_t>(nao_) * nao_, 0.0);
            transform_density_mo_to_ao_cpu(nao_, C_full, D_mo.data(),
                                           D_ao.data());

            std::vector<real_t> dip_x_ao, dip_y_ao, dip_z_ao;
            const auto& shells   = rhf_.get_primitive_shells();
            const auto& cgto_norm = rhf_.get_cgto_normalization_factors();
            const auto& shell_infos = rhf_.get_shell_type_infos();
            compute_ao_dipole_integrals(
                shells.host_ptr(), shells.size(),
                cgto_norm.host_ptr(), nao_, shell_infos,
                dip_x_ao, dip_y_ao, dip_z_ao);

            real_t mu_e_x = 0.0, mu_e_y = 0.0, mu_e_z = 0.0;
            for (int mu = 0; mu < nao_; ++mu) {
                for (int nu = 0; nu < nao_; ++nu) {
                    const real_t Dmn =
                        D_ao[static_cast<size_t>(mu) * nao_ + nu];
                    const size_t k =
                        static_cast<size_t>(mu) * nao_ + nu;
                    mu_e_x += Dmn * dip_x_ao[k];
                    mu_e_y += Dmn * dip_y_ao[k];
                    mu_e_z += Dmn * dip_z_ao[k];
                }
            }
            mu_e_x = -mu_e_x; mu_e_y = -mu_e_y; mu_e_z = -mu_e_z;

            real_t mu_n_x = 0.0, mu_n_y = 0.0, mu_n_z = 0.0;
            const auto& atoms = rhf_.get_atoms();
            for (size_t a = 0; a < atoms.size(); ++a) {
                const auto& at = atoms.host_ptr()[a];
                const real_t Z = static_cast<real_t>(at.effective_charge);
                mu_n_x += Z * at.coordinate.x;
                mu_n_y += Z * at.coordinate.y;
                mu_n_z += Z * at.coordinate.z;
            }
            const real_t kAUtoDebye = 2.5417464157;
            real_t dx = (mu_e_x + mu_n_x) * kAUtoDebye;
            real_t dy = (mu_e_y + mu_n_y) * kAUtoDebye;
            real_t dz = (mu_e_z + mu_n_z) * kAUtoDebye;
            real_t dmag = std::sqrt(dx * dx + dy * dy + dz * dz);

            std::cout << "  dipole [Debye]: x=" << std::fixed
                      << std::setprecision(4) << dx
                      << "  y=" << dy << "  z=" << dz
                      << "  |D|=" << dmag << std::endl;

            // Mulliken population analysis: q_A = Z_A - Σ_{μ ∈ A, ν} D[μ,ν] · S[μ,ν]
            // (closed-shell convention: D includes occupation factor 2).
            rhf_.get_overlap_matrix().toHost();
            const real_t* S_AO_h = rhf_.get_overlap_matrix().host_ptr();
            const auto& a2b = rhf_.get_atom_to_basis_range();
            std::cout << "  Mulliken charges:";
            for (size_t a = 0; a < atoms.size(); ++a) {
                const auto& at = atoms.host_ptr()[a];
                const real_t Z = static_cast<real_t>(at.effective_charge);
                real_t pop = 0.0;
                const int mu_lo = static_cast<int>(a2b[a].start_index);
                const int mu_hi = static_cast<int>(a2b[a].end_index);
                for (int mu = mu_lo; mu < mu_hi; ++mu) {
                    for (int nu = 0; nu < nao_; ++nu) {
                        pop += D_ao[static_cast<size_t>(mu) * nao_ + nu]
                             * S_AO_h[static_cast<size_t>(mu) * nao_ + nu];
                    }
                }
                const real_t q = Z - pop;
                std::cout << "  " << at.atomic_number << "(" << a << "):"
                          << std::showpos << std::fixed
                          << std::setprecision(4) << q << std::noshowpos;
            }
            std::cout << std::endl;
        }
    }

    // (T) re-use: restore the LMP2 amplitudes (undo the in-place CCSD dressing)
    // and hand the LMP2 pair state to the (T) driver via lmp2_snapshot_. The
    // CCSD correlation energy (E_total) was already computed from the dressed
    // amplitudes above, so restoring Y here does not change it.
    if (capture_lmp2_) {
        for (size_t idx = 0; idx < res.pairs.size(); ++idx)
            res.pairs[idx].Y = lmp2_Y_snap[idx];
        lmp2_snapshot_ = std::move(res);
    }

    return E_total;
}

// ---------------------------------------------------------------------------
//  ERI wiring  —  RI is the only supported back-end for DLPNO.
// ---------------------------------------------------------------------------
real_t ERI_RI_RHF::compute_dlpno_ccsd() {
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
    DLPNOCCSD driver(rhf_, *this, std::move(p));
    return driver.compute_energy();
}

real_t ERI_RI_RHF::compute_dlpno_ccsd_capture(DLPNOLMP2Result& lmp2_out) {
    // Same as compute_dlpno_ccsd() but captures the converged LMP2 pair state
    // (pre-CCSD-dressing) into lmp2_out so the (T) driver can reuse it instead
    // of re-solving LMP2 — bit-exact, ~LMP2-time saved.
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
    DLPNOCCSD driver(rhf_, *this, std::move(p));
    driver.capture_lmp2_ = true;
    const real_t e = driver.compute_energy();
    lmp2_out = std::move(driver.lmp2_snapshot_);
    return e;
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

    // Phase 3.2.7 — triples-energy prescreen (ORCA-style). When
    // GANSU_DLPNO_T_PRESCREEN=<thresh> (>0) is set, skip a triple whose cheap
    // amplitude-product estimate is below <thresh>; 0/unset = off = compute all
    // active triples (today's behavior, bit-identical). The estimate
    //   e_ijk^est = |Y_ij|·|Y_ik| + |Y_ij|·|Y_jk| + |Y_ik|·|Y_jk|
    // (the three pair channels feeding a triple) is a near-free proxy for the
    // (T) contribution magnitude (|Y|_F = Frobenius norm of the LMP2 amplitude).
    // Calibrate <thresh> by sweeping vs the unscreened E((T)) — the (T) energy
    // must stay within the intended truncation error. This is the GANSU
    // analogue of ORCA's TCutTriplesPreScreen that makes DLPNO-(T) affordable.
    const char* presc_env = std::getenv("GANSU_DLPNO_T_PRESCREEN");
    const real_t t_prescreen = (presc_env != nullptr) ? std::atof(presc_env) : 0.0;
    long long skipped_prescreen = 0;
    std::vector<real_t> s_norm;
    if (t_prescreen > 0.0) {
        s_norm.assign(res.pairs.size(), 0.0);
        for (size_t p = 0; p < res.pairs.size(); ++p) {
            real_t s = 0.0;
            for (real_t y : res.pairs[p].Y) s += y * y;
            s_norm[p] = std::sqrt(s);
        }
    }

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
                if (t_prescreen > 0.0) {
                    const real_t e_est = s_norm[idx_ij] * s_norm[idx_ik]
                                       + s_norm[idx_ij] * s_norm[idx_jk]
                                       + s_norm[idx_ik] * s_norm[idx_jk];
                    if (e_est < t_prescreen) { ++skipped_prescreen; continue; }
                }
                const int u = n_ij + n_ik + n_jk;
                sum_union_size += u;
                max_union_size = std::max(max_union_size, u);
                triples.push_back({i, j, k, idx_ij, idx_ik, idx_jk,
                                   n_ij, n_ik, n_jk});
            }

    // Phase 3.2.0: TNO builder is shared across threads (read-only).
    // TNO occupation-number truncation (env GANSU_DLPNO_T_CUT_TNO=<thresh>, 0/unset
    // = off = full union span = today's behavior). Per-triple cost ~ n_tno³, so a
    // nonzero threshold (e.g. 1e-8) shrinks the avg TNO (≈44 → ~20) for a large
    // speedup; calibrate vs the unscreened E((T)) like the prescreen threshold.
    const char* tcut_tno_env = std::getenv("GANSU_DLPNO_T_CUT_TNO");
    const real_t t_cut_tno = (tcut_tno_env != nullptr) ? std::atof(tcut_tno_env) : 0.0;
    TNOBuilder tno_builder(res.pairs, h_F, h_S, nao, /*tol_lin_dep=*/1e-7, t_cut_tno);

    // Device-resident batch pack (eliminates the per-triple host memset + pack
    // round-trip; eri/proj K/M/T stay on device and are written straight into
    // the energy batch slot by device kernels). Default ON (validated bit-exact
    // on Anthracene + PTCDA; falls back to the host path when the GPU helpers
    // are inactive). Opt out with GANSU_DLPNO_T_DEVICE_PACK=0.
    const char* devpack_env = std::getenv("GANSU_DLPNO_T_DEVICE_PACK");
    const bool kDevPack = (devpack_env == nullptr)
                          || (std::string(devpack_env) != "0");

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

    // Note: do NOT call omp_set_num_threads(n_threads) globally here — the
    // `num_threads(n_threads)` clause on the pragma below correctly forces
    // n_threads for the (T) triple loop, and mutating global OMP state
    // clamps any subsequent CPU OMP region to n_threads (5-9× slowdown
    // observed in iterate_dlpno_ccsd_t2 on multi-CPU hosts).

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

    // ===================================================================
    // Phase 1 (ALL CPU cores): build every triple's TNO basis up front.
    // build_for_triple is pure CPU (union-overlap + density + Fock
    // eigensolves) and was the dominant (T) cost, but inside the GPU energy
    // loop below it ran on only num_gpus threads. Triples are independent and
    // write disjoint tnos[t_idx], so this is a plain all-cores parallel-for
    // (same split that fixed DLPNO-MP2 pair_setup). The GPU loop then consumes
    // tnos[t_idx] on num_gpus threads. Memory: Σ nao·n_tno doubles (~0.4 GB at
    // anthracene, a few GB at PTCDA — within the (T)-feasible size range).
    // ===================================================================
    std::vector<TNOData> tnos(static_cast<size_t>(n_triples));
    {
        const auto t_p1 = std::chrono::steady_clock::now();
        #pragma omp parallel for schedule(dynamic, 8)
        for (int t_idx = 0; t_idx < n_triples; ++t_idx) {
            const TripleEntry& tr = triples[t_idx];
            tnos[t_idx] = tno_builder.build_for_triple(
                tr.idx_ij, tr.idx_ik, tr.idx_jk);
        }
        if (verbose >= 2)
            std::cout << "[DLPNO-(T)-PROF] phase1 tno_build (all cores)="
                      << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - t_p1).count()
                      << "s" << std::endl;
    }

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
        // Device-pack producer-completion events (per thread).
        cudaEvent_t ev_eri = nullptr, ev_proj = nullptr;
        if (kDevPack) { cudaEventCreate(&ev_eri); cudaEventCreate(&ev_proj); }
        local_setup[tid] = std::chrono::duration<double>(
                              Clk_setup::now() - t_setup0).count();
        real_t local_e_gpu = 0.0;   // accumulated by this thread's flush

        #pragma omp for schedule(static, 1)
        for (int t_idx = 0; t_idx < static_cast<int>(triples.size()); ++t_idx) {
            using Clk = std::chrono::steady_clock;
            auto t_sec0 = Clk::now();
            const TripleEntry& tr = triples[t_idx];
            // Phase 3.2.0: TNO basis was built in the all-cores Phase-1 pre-pass
            // above (tnos[t_idx]); the GPU energy loop only consumes it here.
            const TNOData& tno = tnos[t_idx];
            local_tno_sum[tid] += tno.n_tno;
            local_tno_max[tid] = std::max(local_tno_max[tid], tno.n_tno);
            local_dropped_sum[tid] += tno.n_dropped_overlap;

            // ---- Device-pack fast path: K/M/T computed + packed entirely on
            // device (no host download / memset / transpose). Skips the host
            // eri_t / hole_m / T_*_ext / T_part construction below.
            if (kDevPack && eri_gpu.active() && proj_gpu.active()
                && tgpu.active() && tno.n_tno > 0) {
                const int triple_lmos_d[3] = {tr.i, tr.j, tr.k};
                const real_t eps_i_d = res.F_LMO[tr.i * nocc + tr.i];
                const real_t eps_j_d = res.F_LMO[tr.j * nocc + tr.j];
                const real_t eps_k_d = res.F_LMO[tr.k * nocc + tr.k];
                eri_gpu.build_eri_and_m_device(
                    tno.Q_tno.data(), tno.n_tno, triple_lmos_d, ev_eri);
                std::vector<int> bil(nocc, -1), bjl(nocc, -1),
                                 bkl(nocc, -1), bpart(9, -1);
                std::vector<std::vector<real_t>> du_il, du_jl, du_kl;
                std::array<std::vector<real_t>, 9> du_part;
                proj_gpu.project_for_triple(
                    tno.Q_tno.data(), tno.n_tno, triple_lmos_d,
                    du_il, du_jl, du_kl, du_part, /*download=*/false,
                    bil.data(), bjl.data(), bkl.data(), bpart.data(), ev_proj);
                bool queued = tgpu.add_to_batch_device(
                    tr.i, tr.j, tr.k, eps_i_d, eps_j_d, eps_k_d, tno,
                    eri_gpu.device_K(), eri_gpu.device_M(),
                    proj_gpu.device_T_batch(),
                    bil.data(), bjl.data(), bkl.data(), bpart.data(),
                    ev_eri, ev_proj, nocc);
                if (!queued) {
                    local_e[tid] += tgpu.flush_batch();
                    tgpu.begin_batch();
                    // d_K/d_M/d_T_batch still hold this triple (untouched by the
                    // energy kernels); just re-pack into the fresh batch.
                    queued = tgpu.add_to_batch_device(
                        tr.i, tr.j, tr.k, eps_i_d, eps_j_d, eps_k_d, tno,
                        eri_gpu.device_K(), eri_gpu.device_M(),
                        proj_gpu.device_T_batch(),
                        bil.data(), bjl.data(), bkl.data(), bpart.data(),
                        ev_eri, ev_proj, nocc);
                }
                if (queued) ++local_gpu_count[tid];
                ++local_count[tid];
                continue;
            }

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
        if (ev_eri)  cudaEventDestroy(ev_eri);
        if (ev_proj) cudaEventDestroy(ev_proj);
    }

    real_t e_triples = 0.0;
    long long total_counted = 0;
    for (int t = 0; t < n_threads; ++t) {
        e_triples += local_e[t];
        total_counted += local_count[t];
    }

    const long long active_total = total - skipped_empty - skipped_prescreen;
    if (out_total_triples)  *out_total_triples  = total;
    if (out_active_triples) *out_active_triples = active_total;

    if (verbose >= 1) {
        const long long active = total - skipped_empty - skipped_prescreen;
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
                  << "  skipped (prescreen)        : " << skipped_prescreen
                  << "  (thresh=" << std::scientific << std::setprecision(2)
                  << t_prescreen << std::defaultfloat << ")\n"
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
    // Cap OpenMP threads across the whole (T) driver (the inner
    // compute_dlpno_ccsd re-applies the same cap harmlessly). Keeps the
    // per-triple TNOBuilder eigensolves' Eigen->OpenBLAS calls under the 128
    // per-caller-thread buffer limit — this is exactly where the crash hit.
    OmpThreadCapGuard omp_cap(rhf_.get_dlpno_cpu_threads());
    // (T) reuses the ground CCSD's converged LMP2 pair state (snapshotted
    // pre-dressing inside compute_dlpno_ccsd_capture) — removes the redundant
    // second solve_dlpno_lmp2 that previously cost ~one full LMP2 solve.
    // Bit-exact: the old re-solve used the same thresholds (only verbose=0).
    DLPNOLMP2Result res;
    const real_t e_ccsd = compute_dlpno_ccsd_capture(res);

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
