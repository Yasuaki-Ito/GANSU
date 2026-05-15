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
 * @file dlpno_mp2_lambda.cu
 * @brief DLPNO-MP2 Λ_2 amplitudes and 1-RDM (Sub-phase 1, CPU first).
 *
 * Closed-form spin-traced MP2 lambda relation per pair (no iteration):
 *   Y_lambda[a,b] = 2 · Y[a,b] − Y[b,a]
 *
 * 1-RDM (Datta, Kossmann, Neese 2016 Level A, orbital-unrelaxed) is built as:
 *   D^oo[i,j]: cross-pair sum with barS overlap projection
 *   D^vv[a,b]: per-pair PNO contraction + canonical back-transform
 *   D^ov = D^vo = 0  (closed-shell MP2, T1=Λ1=0)
 *   D_HF: 2·I on occupied diagonal
 *
 * Sub-step 1.2 in this commit implements the oo block; vv block and
 * back-transform follow in Sub-step 1.3-1.4.
 *
 * NOTE: prefactors and sign conventions in oo and vv blocks are recorded
 * here in their TEMPLATE form. The strict-mode validation against PySCF
 * MP2 (mp.MP2(mf).make_rdm1(), orbital-unrelaxed) confirms / fixes them
 * during Sub-step 1.6.
 */

#include "dlpno_lambda.hpp"
#include "dlpno_density.hpp"
#include "dlpno_pair_data.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

namespace gansu {

// File-wide row-major dynamic Eigen matrix used by both the MP2 1-RDM
// helpers (Sub-phase 1) and the iterative DLPNO-CCSD Λ residual code
// (Sub-phase 2X.1). Hoisted to file scope so the helpers in the
// anonymous namespace below can also reuse the alias.
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>;

// ============================================================================
// Sub-step 1.1: closed-form MP2 Λ_2
// ============================================================================

void compute_dlpno_mp2_lambda(std::vector<PairData>& pairs) {
    for (auto& p : pairs) {
        if (p.n_pno == 0) {
            p.Y_lambda.clear();
            continue;
        }
        const int n = p.n_pno;
        p.Y_lambda.assign(static_cast<size_t>(n) * n, 0.0);

        // Closed form: Y_lambda[a,b] = 2 * Y[a,b] - Y[b,a]
        //  ( = spin-traced MP2 lambda in pair-PNO basis, equivalent to the
        //    canonical 2 t_ij^{ab} - t_ij^{ba} relation with Y playing the
        //    role of T2 amplitudes in W basis. )
        for (int a = 0; a < n; ++a) {
            for (int b = 0; b < n; ++b) {
                const size_t ab = static_cast<size_t>(a) * n + b;
                const size_t ba = static_cast<size_t>(b) * n + a;
                p.Y_lambda[ab] = 2.0 * p.Y[ab] - p.Y[ba];
            }
        }
    }
}

// Sub-phase 2 strategy (a): closed-form approximation for DLPNO-CCSD Λ_2.
// Mathematically identical to compute_dlpno_mp2_lambda — Y is now the
// converged CCSD T2 amplitude rather than the MP2 T2. The function name
// makes the calling intent explicit (and the approximation visible in
// stack traces / grep).
void compute_dlpno_ccsd_lambda_closedform(std::vector<PairData>& pairs) {
    compute_dlpno_mp2_lambda(pairs);
}

// ============================================================================
// Sub-step 2X.1: Λ_2 Jacobi iteration in the LMP2 limit (no F-eff dressing,
// T1 / Λ_1 deferred to Sub-step 2X.3).
//
// Residual form (cf. iterate_lmp2 in dlpno_pair_data.cu, with Λ replacing Y
// and source K replaced by the antisymmetrized 2K - K^T):
//
//   R^Λ_{ij}[a,b] = (2 L[a,b] - L[b,a])                            (source)
//                 + (Λ_a + Λ_b - F_ii - F_jj) · Λ_old[a,b]         (Δ shift)
//                 - Σ_{k≠i} F_LMO[i,k] · barS_kj·Λ_old(kj)·barS_kj^T
//                 - Σ_{l≠j} F_LMO[l,j] · barS_il·Λ_old(il)·barS_il^T
//
// Update: Λ[a,b] -= R[a,b] / (Λ_a + Λ_b - F_ii - F_jj).
//
// Verified at convergence in strict mode: Λ_2 = 2 Y - Y^T (closed-form).
// ============================================================================

// Build barS^(ij,kl) overlap [n_pno_ij × n_pno_kl] from per-pair PNOs.
// (Local helper, identical to the one used in build_dlpno_mp2_dm_oo_cpu;
//  duplicated to avoid forward declaration churn for now.)
namespace {
inline void compute_barS_lambda(
    const real_t* bar_Q_ij, int n_pno_ij,
    const real_t* bar_Q_kl, int n_pno_kl,
    const real_t* S_AO,     int nao,
    RowMatXd&     out)
{
    out.setZero(n_pno_ij, n_pno_kl);
    if (n_pno_ij == 0 || n_pno_kl == 0) return;
    Eigen::Map<const RowMatXd> Qij(bar_Q_ij, nao, n_pno_ij);
    Eigen::Map<const RowMatXd> Qkl(bar_Q_kl, nao, n_pno_kl);
    Eigen::Map<const RowMatXd> S(S_AO, nao, nao);
    out = Qij.transpose() * (S * Qkl);
}
} // namespace

DLPNOLambdaStatus iterate_dlpno_ccsd_lambda(
    const std::vector<PairSetup>&            setups,
    std::vector<PairData>&                   pairs,
    std::vector<std::vector<real_t>>&        Lambda1,
    const std::vector<std::vector<real_t>>&  /*T1*/,         // 2X.3
    const std::vector<int>&                  pair_lookup,
    const std::vector<real_t>&               F_LMO,
    const real_t*                            h_S,
    int nocc, int nao,
    int max_iter, real_t conv_tol,
    bool enable_dressing,                                    // 2X.2a +
    int verbose, const char* round_tag,
    const struct Phase24Integrals*           /*phase24*/,    // 2X.2c (cross-pair)
    int /*num_gpus*/)                                         // 2X later
{
    // Λ_1 = 0 in this sub-step (Sub-step 2X.3 will iterate it).
    Lambda1.assign(nocc, {});

    // Initial guess: Λ_2 = 2 Y - Y^T (closed-form). For converged Y in
    // strict mode + LMP2 limit this is also the exact solution, so the
    // iteration should report iters=0 / r_max ≈ 0 immediately. With non-
    // strict truncation or finite F_ik off-diagonal, the iteration will
    // refine Λ_2 to the LMP2-limit fixed point.
    compute_dlpno_mp2_lambda(pairs);

    constexpr real_t kFLMOThresh = 1e-14;

    // ----------------------------------------------------------------
    // Sub-step 2X.2a: intra-pair diagonal hole dressing (ΔF_ii) only.
    //
    // For canonical CCSD Λ the diagonal F_eff dressing reads
    //   ΔF_{ii} = Σ_l Σ_{cd} T_il^{cd} · (2 L_il^{cd} - L_il^{dc})
    // which is the (k=i) diagonal of the Phase 2.4.x dF_ki matrix. Here we
    // only implement the Phase 2.3.3 fallback (l=i restriction), reading
    // from each pair's local L (= K integrals in W basis):
    //   ΔF_{ii} ≈ Σ_{cd} (2 L_ii^{cd} - L_ii^{dc}) · Y_ii^{cd}
    // The Λ_2 residual then receives  R -= (ΔF_ii + ΔF_jj) · Λ_old.
    //
    // Computed ONCE before the iteration (T2 is fixed at this point);
    // gated by enable_dressing so the LMP2-limit (2X.1) path is preserved
    // when dressing=false.
    //
    // Sub-step 2X.2c will swap this fallback for the full Phase24Integrals-
    // based Σ_l T_pair^{(il)} · Y_il sum (matches T-iteration dressing).
    // ----------------------------------------------------------------
    std::vector<real_t> dF_diag(nocc, 0.0);   // ΔF_ii (per LMO)
    if (enable_dressing) {
        for (int i = 0; i < nocc; ++i) {
            const int idx_ii = pair_lookup[i * nocc + i];
            const PairData&  p = pairs[idx_ii];
            const int n_ii = p.n_pno;
            if (n_ii == 0) continue;
            Eigen::Map<const RowMatXd> L(p.L.data(), n_ii, n_ii);
            Eigen::Map<const RowMatXd> Y(p.Y.data(), n_ii, n_ii);
            real_t s = 0.0;
            for (int c = 0; c < n_ii; ++c)
                for (int d = 0; d < n_ii; ++d)
                    s += (2.0 * L(c, d) - L(d, c)) * Y(c, d);
            dF_diag[i] = s;
        }
    }

    DLPNOLambdaStatus s;
    s.iters     = 0;
    s.max_R     = 0.0;
    s.converged = false;

    // Old-Λ snapshot for Jacobi sweep.
    std::vector<std::vector<real_t>> L2_old(pairs.size());

    // Per-thread scratch (kept outside the per-pair loop in the simple
    // single-thread reference; OMP / GPU parallelism added in Sub-step 2X
    // later, mirroring iterate_lmp2's PiCacheGpu/ResidGpu path).
    RowMatXd barS_buf;
    RowMatXd pi_buf;
    RowMatXd R_buf;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Snapshot Y_lambda before sweep.
        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            L2_old[idx] = pairs[idx].Y_lambda;
        }

        real_t r_max = 0.0;

        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            PairData&        pij = pairs[idx];
            const PairSetup& sij = setups[idx];
            const int n = pij.n_pno;
            if (n == 0) continue;

            Eigen::Map<RowMatXd>       L2_ij  (pij.Y_lambda.data(), n, n);
            Eigen::Map<const RowMatXd> L_ij   (pij.L.data(),        n, n);
            Eigen::Map<const RowMatXd> L2_old_ij(L2_old[idx].data(), n, n);

            const real_t shift = sij.F_ii + sij.F_jj;

            // R = (2 L - L^T)  +  (Λ_a+Λ_b-shift) * Λ_old
            R_buf.noalias() = 2.0 * L_ij - L_ij.transpose();
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    R_buf(a, b) +=
                        (pij.Lambda[a] + pij.Lambda[b] - shift)
                        * L2_old_ij(a, b);

            // Sub-step 2X.2a: intra-pair diagonal hole dressing
            //   R -= (ΔF_ii + ΔF_jj) · Λ_old
            // (Mirrors Phase 2.3.3 of iterate_dlpno_ccsd_t2 line 1110-1119.
            //  Same prefactor / sign convention because the Λ_2 equation
            //  inherits the F_eff structure from the Lagrangian via
            //  L_CCSD = ⟨0|(1+Λ) e^{-T} ̄H e^T |0⟩.)
            if (enable_dressing) {
                const real_t dF_sum = dF_diag[sij.i] + dF_diag[sij.j];
                if (dF_sum != 0.0) {
                    R_buf.noalias() -= dF_sum * RowMatXd(L2_old_ij);
                }
            }

            // Cross-pair coupling on i (k != sij.i) using Λ from pair (k,j).
            for (int k = 0; k < nocc; ++k) {
                if (k == sij.i) continue;
                const real_t F_ik = F_LMO[sij.i * nocc + k];
                if (std::fabs(F_ik) < kFLMOThresh) continue;

                const int idx_kj = pair_lookup[k * nocc + sij.j];
                const PairData&  pkj = pairs[idx_kj];
                const PairSetup& skj = setups[idx_kj];
                if (pkj.n_pno == 0) continue;

                // barS^(ij,kj) = bar_Q_ij^T · S · bar_Q_kj  [n × n_kj]
                compute_barS_lambda(pij.bar_Q.data(), n,
                                    pkj.bar_Q.data(), pkj.n_pno,
                                    h_S, nao, barS_buf);

                Eigen::Map<const RowMatXd> L2_kj(L2_old[idx_kj].data(),
                                                 pkj.n_pno, pkj.n_pno);

                // π_kj^(ij) = barS · Λ_kj · barS^T   [n × n]
                // Orientation handling: if stored pair is (j,k) instead of
                // (k,j), Λ_kj  = (Λ_jk)^T.
                if (skj.i != k) {
                    pi_buf.noalias() = barS_buf
                                     * L2_kj.transpose()
                                     * barS_buf.transpose();
                } else {
                    pi_buf.noalias() = barS_buf
                                     * L2_kj
                                     * barS_buf.transpose();
                }
                R_buf.noalias() -= F_ik * pi_buf;
            }

            // Cross-pair coupling on j (l != sij.j) using Λ from pair (i,l).
            for (int l = 0; l < nocc; ++l) {
                if (l == sij.j) continue;
                const real_t F_lj = F_LMO[l * nocc + sij.j];
                if (std::fabs(F_lj) < kFLMOThresh) continue;

                const int idx_il = pair_lookup[sij.i * nocc + l];
                const PairData&  pil = pairs[idx_il];
                const PairSetup& sil = setups[idx_il];
                if (pil.n_pno == 0) continue;

                compute_barS_lambda(pij.bar_Q.data(), n,
                                    pil.bar_Q.data(), pil.n_pno,
                                    h_S, nao, barS_buf);

                Eigen::Map<const RowMatXd> L2_il(L2_old[idx_il].data(),
                                                 pil.n_pno, pil.n_pno);

                if (sil.i != sij.i) {
                    pi_buf.noalias() = barS_buf
                                     * L2_il.transpose()
                                     * barS_buf.transpose();
                } else {
                    pi_buf.noalias() = barS_buf
                                     * L2_il
                                     * barS_buf.transpose();
                }
                R_buf.noalias() -= F_lj * pi_buf;
            }

            // Jacobi update.
            real_t r_max_pair = 0.0;
            for (int a = 0; a < n; ++a) {
                for (int b = 0; b < n; ++b) {
                    const real_t denom =
                        pij.Lambda[a] + pij.Lambda[b] - shift;
                    const real_t r = R_buf(a, b);
                    L2_ij(a, b) -= r / denom;
                    r_max_pair = std::max(r_max_pair, std::fabs(r));
                }
            }
            r_max = std::max(r_max, r_max_pair);
        }

        s.iters = iter + 1;
        s.max_R = r_max;

        if (verbose >= 2) {
            std::cout << "[" << (round_tag ? round_tag : "DLPNO-Λ")
                      << "] iter " << s.iters
                      << "  max|R|=" << std::scientific
                      << std::setprecision(3) << r_max << std::endl;
        }

        if (r_max < conv_tol) {
            s.converged = true;
            break;
        }
    }

    if (verbose >= 1) {
        std::cout << "[" << (round_tag ? round_tag : "DLPNO-Λ") << "] "
                  << (s.converged ? "converged" : "MAX_ITER")
                  << " in " << s.iters
                  << " iter, max|R|=" << std::scientific
                  << std::setprecision(3) << s.max_R << std::endl;
    }
    return s;
}

// ============================================================================
// Helper: build per-pair PNO ↔ PNO overlap (barS) on-the-fly via AO basis.
//
//   barS^(ij,kl)[a,c] = (bar_Q^(ij))^T_a · S_AO · bar_Q^(kl)_c
//
// Computed here as a small dense matrix for the (ij)/(kl) pair only —
// designed for clarity. Production GPU path will use the existing
// PiCacheGpu pattern (Sub-step 1.10).
// ============================================================================

namespace {

// (RowMatXd hoisted to file scope above — see top of namespace gansu.)

// Compute the barS^(ij,kl) overlap matrix [n_pno_ij × n_pno_kl] given the
// two bar_Q matrices (each [nao × n_pno_*]) and the AO overlap S [nao × nao].
//
//   barS = bar_Q_ij^T · S_AO · bar_Q_kl
//
// All matrices are row-major.
void compute_barS(const real_t* bar_Q_ij, int n_pno_ij,
                  const real_t* bar_Q_kl, int n_pno_kl,
                  const real_t* S_AO,     int nao,
                  std::vector<real_t>& out)
{
    out.assign(static_cast<size_t>(n_pno_ij) * n_pno_kl, 0.0);
    if (n_pno_ij == 0 || n_pno_kl == 0) return;

    Eigen::Map<const RowMatXd> Qij(bar_Q_ij, nao, n_pno_ij);
    Eigen::Map<const RowMatXd> Qkl(bar_Q_kl, nao, n_pno_kl);
    Eigen::Map<const RowMatXd> S(S_AO,        nao, nao);

    RowMatXd tmp = S * Qkl;                 // [nao × n_pno_kl]
    RowMatXd bS  = Qij.transpose() * tmp;   // [n_pno_ij × n_pno_kl]

    Eigen::Map<RowMatXd> out_map(out.data(), n_pno_ij, n_pno_kl);
    out_map = bS;
}

} // namespace

// ============================================================================
// Sub-step 1.2: D^oo block in LMO basis (CPU, cross-pair barS projection)
//
// Template form (Datta 2016 Eq. 38, MP2 spin-traced limit, prefactor TBD
// in strict-mode validation):
//
//   D^oo[i,j] = -prefactor *
//        Σ_k  Σ_{a_ik b_ik}  Σ_{a_jk b_jk}
//          Y^(ik)[a_ik, b_ik] · Y_lambda^(jk)[a_jk, b_jk]
//          · barS^(ik,jk)[a_ik, a_jk]
//          · barS^(ik,jk)[b_ik, b_jk]
//
// Notes:
//   - Pair lookup respects symmetry: storage holds i≤j; pair_lookup[i*nocc+j]
//     resolves both (i,k) and (k,i) to the same index. Closed-shell symmetry
//     of Y is Y_jk = Y_kj^T (handled by transposing when fetching swapped
//     pair). The same applies to Y_lambda.
//   - Prefactor and overall sign are recorded as canonical-MP2 limit;
//     strict-mode test (Sub-step 1.6) will lock the value against PySCF.
//   - For Sub-step 1.2 we implement the OO block only; VV and back-transform
//     are added in Sub-step 1.3-1.4.
// ============================================================================

namespace {

// Fetch Y (or Y_lambda) for pair (i,j) in the SAME orientation as
// pair_lookup[i*nocc+j]. Returns a pointer + flag indicating whether
// transpose is needed because the stored pair is (j,i) instead of (i,j).
//
// Returns nullptr if n_pno == 0 for that pair.
struct PairAmpView {
    const real_t* data = nullptr;
    int           n    = 0;       ///< n_pno of the underlying stored pair
    bool          transpose = false;
};

PairAmpView fetch_amp(const std::vector<PairSetup>& setups,
                      const std::vector<PairData>&  pairs,
                      const std::vector<int>&       pair_lookup,
                      int nocc, int i, int j,
                      bool want_lambda)
{
    const int idx = pair_lookup[i * nocc + j];
    if (idx < 0 || idx >= static_cast<int>(pairs.size())) return {};
    const auto& p = pairs[idx];
    if (p.n_pno == 0) return {};
    // Determine orientation: setup stores i'≤j' for canonical pair.
    const auto& s = setups[idx];
    const bool stored_swapped = (s.i != i);   // means stored is (j,i)
    PairAmpView v;
    v.data      = want_lambda ? p.Y_lambda.data() : p.Y.data();
    v.n         = p.n_pno;
    v.transpose = stored_swapped;
    return v;
}

inline real_t get_amp(const PairAmpView& v, int a, int b) {
    if (v.data == nullptr) return 0.0;
    if (v.transpose) std::swap(a, b);
    return v.data[static_cast<size_t>(a) * v.n + b];
}

} // namespace

// CPU reference implementation of the oo block. This is the slow but
// transparent version used to validate the closed-form lambda and the
// barS projection against PySCF before any GPU optimization.
//
// Writes to the upper-left n_lmo × n_lmo block of D_mo_out (row-major).
static void build_dlpno_mp2_dm_oo_cpu(
    const std::vector<PairSetup>& setups,
    const std::vector<PairData>&  pairs,
    const std::vector<int>&       pair_lookup,
    int                           n_lmo,
    const real_t*                 S_AO,
    int                           nao,
    real_t*                       D_oo)
{
    // Zero out the oo block.
    for (int i = 0; i < n_lmo; ++i)
        for (int j = 0; j < n_lmo; ++j)
            D_oo[static_cast<size_t>(i) * n_lmo + j] = 0.0;

    // Convention check (PySCF mp2.py):
    //   doo = -einsum('ikab,jkab->ij', t2, ll2)         (asymmetric form)
    //   dm1[:nocc,:nocc] = doo + doo.T                  (symmetrize)
    //                    = -2·einsum (since doo is symmetric)
    // → effective prefactor on the einsum = +2 inside the leading minus sign.
    // Spin-trace factor matches the +2 on the dvv side (see prefactor_vv below)
    // so that tr(D^oo) + tr(D^vv) = 0 (trace conservation: correlation moves
    // electron density occupied → virtual without changing N_elec).
    const real_t prefactor = 2.0;  // TBD-locked by Sub-step 1.6 strict-mode test

    // Cross-pair scratch for barS — sized to the largest n_pno encountered.
    std::vector<real_t> barS_scratch;

    for (int i = 0; i < n_lmo; ++i) {
        for (int j = 0; j < n_lmo; ++j) {
            real_t Dij = 0.0;
            for (int k = 0; k < n_lmo; ++k) {

                // Fetch Y for (i,k) and Y_lambda for (j,k).
                PairAmpView Yik = fetch_amp(setups, pairs, pair_lookup,
                                            n_lmo, i, k, /*lambda=*/false);
                PairAmpView Ljk = fetch_amp(setups, pairs, pair_lookup,
                                            n_lmo, j, k, /*lambda=*/true);
                if (!Yik.data || !Ljk.data) continue;

                // barS^(ik,jk) is the overlap between pair (i,k)'s PNOs and
                // pair (j,k)'s PNOs.  Look up the underlying stored pair
                // indices (after orientation resolution) to obtain bar_Q.
                const int idx_ik = pair_lookup[i * n_lmo + k];
                const int idx_jk = pair_lookup[j * n_lmo + k];
                const auto& p_ik = pairs[idx_ik];
                const auto& p_jk = pairs[idx_jk];

                // Compute barS^(ik,jk) [n_pno(ik) × n_pno(jk)].
                compute_barS(p_ik.bar_Q.data(), p_ik.n_pno,
                             p_jk.bar_Q.data(), p_jk.n_pno,
                             S_AO, nao, barS_scratch);

                // Sum over a_ik, b_ik, a_jk, b_jk:
                //   D += Y(a_ik,b_ik) · Λ(a_jk,b_jk) · S(a_ik,a_jk) · S(b_ik,b_jk)
                //
                // Stride-aware indexing on the views:
                //   get_amp(Yik, a, b) is Y_{i,k}[a,b] in the pair's PNO basis,
                //   transposed automatically if storage is (k,i).
                const int nik = p_ik.n_pno;
                const int njk = p_jk.n_pno;
                for (int a_ik = 0; a_ik < nik; ++a_ik) {
                    for (int b_ik = 0; b_ik < nik; ++b_ik) {
                        const real_t y_ab = get_amp(Yik, a_ik, b_ik);
                        if (y_ab == 0.0) continue;
                        for (int a_jk = 0; a_jk < njk; ++a_jk) {
                            const real_t s_a = barS_scratch[
                                static_cast<size_t>(a_ik) * njk + a_jk];
                            if (s_a == 0.0) continue;
                            for (int b_jk = 0; b_jk < njk; ++b_jk) {
                                const real_t s_b = barS_scratch[
                                    static_cast<size_t>(b_ik) * njk + b_jk];
                                const real_t l_ab = get_amp(Ljk, a_jk, b_jk);
                                Dij += y_ab * l_ab * s_a * s_b;
                            }
                        }
                    }
                }
            }
            D_oo[static_cast<size_t>(i) * n_lmo + j] = -prefactor * Dij;
        }
    }
}

// ============================================================================
// Sub-step 1.3: D^vv block per-pair contraction
// Sub-step 1.4: back-transform per-pair PNO → canonical virtual basis
//
// Per-pair contribution (Datta 2016 Eq. 39 in MP2 limit, prefactor TBD):
//   Stored pair (i,j) with i ≤ j. Spin-traced spatial-orbital MP2 1-RDM
//   accumulates contributions from BOTH (i,j) and (j,i):
//     D^vv,(ij)[a,b] += Σ_c Y[a,c] · Y_lambda[b,c]      ((i,j) part)
//     D^vv,(ij)[a,b] += Σ_c Y[c,a] · Y_lambda[c,b]      ((j,i) part, only if i<j)
//
//   = (Y · Y_lambda^T) + (Y^T · Y_lambda)               (i<j, two DGEMMs)
//   = (Y · Y_lambda^T)                                  (i=j, one DGEMM; here Y is symmetric)
//
// Back-transform to canonical virtual basis [n_can_vir × n_can_vir]:
//   d^(ij)[a, a_ij] = (C_can_vir^T · S_AO · bar_Q^(ij))[a, a_ij]
//   D^vv_canonical += prefactor_vv · d^(ij) · D^vv,(ij) · d^(ij)^T
//
// prefactor_vv: TBD by strict mode validation (Sub-step 1.6) against PySCF
//   pyscf.mp.MP2.make_rdm1() conventions. Best-guess: +2.0 (spin-trace),
//   sign positive (virtual block is +).
// ============================================================================

namespace {

// Compute d_ij = C_can_vir^T · S_AO · bar_Q^(ij)  [n_can_vir × n_pno].
// All matrices row-major.
//
// C_can_vir is [nao × n_can_vir] (columns are canonical virtual MOs in AO).
// bar_Q is     [nao × n_pno]      (columns are PNOs in AO).
// S_AO is      [nao × nao].
void compute_d_ij(const real_t* C_can_vir, int n_can_vir,
                  const real_t* S_AO,
                  const real_t* bar_Q_ij,  int n_pno,
                  int nao,
                  std::vector<real_t>& out)
{
    out.assign(static_cast<size_t>(n_can_vir) * n_pno, 0.0);
    if (n_pno == 0 || n_can_vir == 0) return;

    Eigen::Map<const RowMatXd> C(C_can_vir, nao, n_can_vir);
    Eigen::Map<const RowMatXd> S(S_AO,      nao, nao);
    Eigen::Map<const RowMatXd> Q(bar_Q_ij,  nao, n_pno);

    RowMatXd SQ = S * Q;                  // [nao × n_pno]
    RowMatXd d  = C.transpose() * SQ;     // [n_can_vir × n_pno]

    Eigen::Map<RowMatXd> out_map(out.data(), n_can_vir, n_pno);
    out_map = d;
}

} // namespace

// CPU reference implementation of the vv block accumulator.
// Writes additive contributions into the bottom-right [n_can_vir × n_can_vir]
// block of D_mo_out (row-major, full nmo × nmo storage, vv offset = n_lmo).
static void build_dlpno_mp2_dm_vv_cpu(
    const std::vector<PairSetup>& setups,
    const std::vector<PairData>&  pairs,
    int                           n_lmo,
    int                           n_can_vir,
    const real_t*                 S_AO,
    const real_t*                 C_can_vir,
    int                           nao,
    real_t*                       D_vv)
{
    // Zero D_vv block.
    const size_t nvv = static_cast<size_t>(n_can_vir) * n_can_vir;
    for (size_t k = 0; k < nvv; ++k) D_vv[k] = 0.0;

    // Best-guess prefactor (TBD by Sub-step 1.6 strict mode test):
    //   PySCF: dvv = 2 * einsum('ijca,ijcb->ab', t2, ll2)
    //   → prefactor_vv = +2.0
    const real_t prefactor_vv = 2.0;

    Eigen::Map<RowMatXd> Dvv(D_vv, n_can_vir, n_can_vir);

    std::vector<real_t> d_buf;             // d_ij [n_can_vir × n_pno]
    std::vector<real_t> Dpair_buf;         // D^vv,(ij) [n_pno × n_pno]

    for (size_t idx = 0; idx < pairs.size(); ++idx) {
        const auto& s = setups[idx];
        const auto& p = pairs[idx];
        if (p.n_pno == 0) continue;
        const int n = p.n_pno;

        // ----------------------------------------------------------------
        // Per-pair D^vv,(ij) contraction in W basis.
        //   part1 = Y · Y_lambda^T   (= Σ_c Y[a,c] Y_lambda[b,c])
        //   if i < j (off-diagonal pair): add part2 = Y^T · Y_lambda
        //     (this contributes the (j,i) channel — see header comment)
        // ----------------------------------------------------------------
        Eigen::Map<const RowMatXd> Y (p.Y.data(),         n, n);
        Eigen::Map<const RowMatXd> YL(p.Y_lambda.data(),  n, n);

        RowMatXd Dpair = Y * YL.transpose();             // (Y · Y_lambda^T)
        if (s.i != s.j) {
            Dpair.noalias() += Y.transpose() * YL;       // + (Y^T · Y_lambda)
        }

        // ----------------------------------------------------------------
        // Back-transform: D^vv,canonical += prefactor · d_ij · Dpair · d_ij^T
        // ----------------------------------------------------------------
        compute_d_ij(C_can_vir, n_can_vir, S_AO,
                     p.bar_Q.data(), n, nao, d_buf);
        Eigen::Map<const RowMatXd> d(d_buf.data(), n_can_vir, n);

        Dvv.noalias() += prefactor_vv * (d * Dpair * d.transpose());
    }
}

// ============================================================================
// build_dlpno_mp2_1rdm_mo — driver (Sub-step 1.2 + 1.3 + 1.4 + 1.5)
//
// Layout of D_mo_out [nmo × nmo, nmo = n_lmo + n_can_vir]:
//   D_mo_out[0:n_lmo, 0:n_lmo]                 = HF (2·I) + D^oo correction
//   D_mo_out[n_lmo:, n_lmo:]                   = D^vv (correlation)
//   D_mo_out[0:n_lmo, n_lmo:] and v↔o          = 0 at Level A MP2 (T1=Λ1=0)
//
// trace(D_mo_out) = 2·n_lmo + tr(D^oo) + tr(D^vv) ≈ N_elec
// (the per-LMO correlation reduces occupied diagonal and increases virtual
// diagonal by equal amounts; net trace = 2·n_lmo = N_elec.)
// ============================================================================

void build_dlpno_mp2_1rdm_mo(
    const std::vector<PairSetup>& setups,
    const std::vector<PairData>&  pairs,
    const std::vector<int>&       pair_lookup,
    int                           n_lmo,
    int                           n_can_vir,
    const real_t*                 S_AO,
    const real_t*                 C_can_vir,
    int                           nao,
    real_t*                       D_mo_out)
{
    const int nmo = n_lmo + n_can_vir;

    // Zero out the full MO density.
    for (int p = 0; p < nmo; ++p)
        for (int q = 0; q < nmo; ++q)
            D_mo_out[static_cast<size_t>(p) * nmo + q] = 0.0;

    // ρ.4 (partial): HF reference, 2·I on occupied diagonal.
    for (int i = 0; i < n_lmo; ++i) {
        D_mo_out[static_cast<size_t>(i) * nmo + i] = 2.0;
    }

    // ρ.1: oo block correction (in LMO basis = upper-left n_lmo × n_lmo).
    {
        std::vector<real_t> D_oo(
            static_cast<size_t>(n_lmo) * n_lmo, 0.0);
        build_dlpno_mp2_dm_oo_cpu(setups, pairs, pair_lookup,
                                  n_lmo, S_AO, nao, D_oo.data());
        for (int i = 0; i < n_lmo; ++i)
            for (int j = 0; j < n_lmo; ++j)
                D_mo_out[static_cast<size_t>(i) * nmo + j] +=
                    D_oo[static_cast<size_t>(i) * n_lmo + j];
    }

    // ρ.2-ρ.3: vv block (per-pair contraction + back-transform).
    {
        std::vector<real_t> D_vv(
            static_cast<size_t>(n_can_vir) * n_can_vir, 0.0);
        build_dlpno_mp2_dm_vv_cpu(setups, pairs,
                                  n_lmo, n_can_vir,
                                  S_AO, C_can_vir, nao, D_vv.data());
        for (int a = 0; a < n_can_vir; ++a)
            for (int b = 0; b < n_can_vir; ++b)
                D_mo_out[static_cast<size_t>(n_lmo + a) * nmo + (n_lmo + b)] +=
                    D_vv[static_cast<size_t>(a) * n_can_vir + b];
    }

    // ρ.4 (full): D^ov / D^vo remain zero at Level A MP2 (T1=Λ1=0).
    // ρ.5 (Sub-step 1.8): weak-pair perturbative correction — TODO.
    //
    // trace check left to caller (Sub-step 1.6 sanity test).
}

// ============================================================================
// Sub-phase 2 strategy (a): DLPNO-CCSD 1-RDM closed-form approximation.
//
// Calls build_dlpno_mp2_1rdm_mo() to fill oo + vv blocks (using CCSD T2 +
// closed-form Y_lambda = 2T - T^T), then adds the explicit T1 contribution
// to the ov/vo blocks via the per-i PAO → canonical virtual back-transform.
//
// The PySCF closed-shell CCSD make_rdm1 in the Λ_1 = 0 limit gives:
//   D[ov][i, a] = 0          (Λ_1 = 0)
//   D[vo][a, i] = T1[i, a]   + small T1·T2 corrections
//
// After PySCF symmetrization (dm1[ov] = dvo.T + dov), this becomes:
//   D[ov][i, a] = T1[i, a]
//   D[vo][a, i] = T1[i, a]
//
// We implement just the leading T1 term here; the dropped corrections are
// O(T1·T2) and small for PM-localized DLPNO-CCSD.
// ============================================================================

void build_dlpno_ccsd_1rdm_mo_closedform(
    const std::vector<PairSetup>&            setups,
    const std::vector<PairData>&             pairs,
    const std::vector<int>&                  pair_lookup,
    const std::vector<std::vector<real_t>>&  T1,
    int                                      n_lmo,
    int                                      n_can_vir,
    const real_t*                            S_AO,
    const real_t*                            C_can_vir,
    int                                      nao,
    real_t*                                  D_mo_out)
{
    // 1. oo + vv + HF blocks (reuse MP2 path; ov/vo left at zero).
    build_dlpno_mp2_1rdm_mo(setups, pairs, pair_lookup,
                            n_lmo, n_can_vir, S_AO, C_can_vir, nao,
                            D_mo_out);

    // 2. Skip T1 contribution if not provided.
    if (T1.empty() || static_cast<int>(T1.size()) != n_lmo) return;

    const int nmo = n_lmo + n_can_vir;
    Eigen::Map<const RowMatXd> S(S_AO,        nao, nao);
    Eigen::Map<const RowMatXd> C(C_can_vir,   nao, n_can_vir);

    // For each LMO i, back-transform T1[i] (in pair (i,i) PAO basis) to
    // canonical virtual basis and add to D[ov][i, a] = D[vo][a, i].
    std::vector<real_t> d_buf;  // d_ii [n_can_vir × n_pao]

    for (int i = 0; i < n_lmo; ++i) {
        if (T1[i].empty()) continue;

        const int idx_ii = pair_lookup[i * n_lmo + i];
        if (idx_ii < 0 || idx_ii >= static_cast<int>(setups.size())) continue;
        const auto& s = setups[idx_ii];
        const int n_pao = s.n_pao;
        if (n_pao == 0 || static_cast<int>(T1[i].size()) != n_pao) continue;

        // d_ii[a, a_pao] = (C_can_vir^T · S_AO · C_can_pair_ii)[a, a_pao]
        d_buf.assign(static_cast<size_t>(n_can_vir) * n_pao, 0.0);
        Eigen::Map<const RowMatXd> Cii(s.C_can_pair.data(), nao, n_pao);
        RowMatXd SCii = S * Cii;                  // [nao × n_pao]
        RowMatXd d    = C.transpose() * SCii;     // [n_can_vir × n_pao]

        // T1_can[a] = Σ_a_pao d[a, a_pao] · T1[i][a_pao]
        Eigen::VectorXd T1_pao(n_pao);
        for (int a = 0; a < n_pao; ++a) T1_pao(a) = T1[i][a];
        Eigen::VectorXd T1_can = d * T1_pao;       // [n_can_vir]

        // Place T1_can in D[ov] and D[vo] (Hermitian).
        for (int a = 0; a < n_can_vir; ++a) {
            const real_t v = T1_can(a);
            D_mo_out[static_cast<size_t>(i) * nmo + (n_lmo + a)] += v;
            D_mo_out[static_cast<size_t>(n_lmo + a) * nmo + i]   += v;
        }
    }
}

} // namespace gansu
