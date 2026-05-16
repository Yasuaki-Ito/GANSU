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
#ifdef _OPENMP
#include <omp.h>
#endif

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
    bool enable_dressing,
    int verbose, const char* round_tag,
    const struct Phase24Integrals*           phase24,        // 2X.2c
    int /*num_gpus*/)                                         // 2X.5 later
{
    // -----------------------------------------------------------------
    // Sub-step 2X.3.0: Λ_1 storage allocation (proper sizes per LMO).
    //
    // Lambda1[i] sits in pair (i,i)'s semi-canonical PAO basis, length
    // setups[pair_lookup[i*nocc+i]].n_pao  — same shape and basis as T1
    // amplitudes stored in DLPNO-CCSD (see iterate_dlpno_ccsd_t2 Phase
    // 2.6c TODO). For Sub-step 2X.3.0 (this commit) we just allocate
    // zero-filled storage. Sub-step 2X.3.1 adds the T2-driven OVVV·mvv1
    // source via new W_ovvv fields on Phase24Integrals; Sub-step 2X.3.2
    // wires the Jacobi update for Λ_1 alongside Λ_2.
    //
    // The closed-shell semi-canonical PAO basis satisfies the Brillouin
    // condition f_{i,a_ii} = 0 within each pair (the PAO is the
    // eigenbasis of F restricted to the pair domain). Cross-pair F
    // elements f_{i, a_jj} for i ≠ j are non-zero but the leading Λ_1
    // source comes from T2-OVVV contractions, NOT from f_{ia}.
    Lambda1.assign(nocc, {});
    for (int i = 0; i < nocc; ++i) {
        const int idx_ii = pair_lookup[i * nocc + i];
        if (idx_ii < 0 || idx_ii >= static_cast<int>(setups.size())) continue;
        const int n_pao_ii = setups[idx_ii].n_pao;
        if (n_pao_ii <= 0) continue;
        Lambda1[i].assign(static_cast<size_t>(n_pao_ii), 0.0);
    }

    // Initial guess: Λ_2 = 2 Y - Y^T (closed-form). For converged Y in
    // strict mode + LMP2 limit this is also the exact solution, so the
    // iteration should report iters=0 / r_max ≈ 0 immediately. With non-
    // strict truncation or finite F_ik off-diagonal, the iteration will
    // refine Λ_2 to the LMP2-limit fixed point.
    compute_dlpno_mp2_lambda(pairs);

    constexpr real_t kFLMOThresh = 1e-14;
    const bool use_phase24 =
        enable_dressing && phase24 != nullptr && phase24->nocc == nocc;

    // ================================================================
    // Sub-step 2X.2c: F-eff dressing intermediates computed from T2.
    //
    // dF_ki[k, i]  = ΔF_{ki} = Σ_l Σ_{cd} T_pair^{(il)}[k,c,l,d] · Y_il^{cd}
    //                (full l sum; mirrors iterate_dlpno_ccsd_t2 line 933-990)
    // DF_per_pair[idx]  = ΔF^{(ij)}_{ac}, signed per-pair particle dressing
    //                = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · Y_kl^{ad}_proj
    //                (mirrors iterate_dlpno_ccsd_t2 line 1033-1052 once
    //                 collapsed; here we use the explicit (k,l,c,d) sum
    //                 since the pi_T_stack cache is not built up.)
    //
    // Both depend only on T2 (= pairs[idx].Y, fixed during Λ iter), so
    // computed ONCE before the iter loop.
    //
    // When `use_phase24` is false (no Phase24Integrals provided OR dressing
    // disabled) we fall back to Sub-step 2X.2a: only ΔF_ii on the diagonal
    // from local L^{(ii)}; full cross-pair dressing is off.
    // ================================================================
    std::vector<real_t> dF_ki(static_cast<size_t>(nocc) * nocc, 0.0);
    std::vector<RowMatXd> DF_per_pair(pairs.size());

    if (use_phase24) {
        // Full hole dressing: full l sum.
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
                        pil.Y.data(), n_il, n_il);
                    const bool transp = (sil.i != i);

                    const real_t* T_il =
                        phase24->T_pair[idx_il].data();
                    const size_t stride_kl =
                        static_cast<size_t>(n_il) * n_il;
                    Eigen::Map<const RowMatXd> T_kl(
                        T_il + static_cast<size_t>(k * nocc + l) * stride_kl,
                        n_il, n_il);
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

        // Full particle dressing per pair (i,j):
        //   DF[idx][a, c] = -Σ_{(k,l), d} π_{kl}^{ij,oriented}[a, d] · T_kl[c, d]
        //                 ≈ -Σ_{(k,l), d} (barS · Y_kl · barS^T)[a, d] · T_kl[c, d]
        // We use the simple per-(k,l) loop here (CPU reference). The
        // production GPU path will hook into pi_T_stack / T_meta_dpair
        // caches once Sub-step 2X.5 lands.
        std::vector<real_t> bQ_ij_T_S_scratch_dummy;
        (void)bQ_ij_T_S_scratch_dummy;

        #pragma omp parallel for schedule(static)
        for (long long idx_ll = 0;
             idx_ll < static_cast<long long>(pairs.size()); ++idx_ll) {
            const size_t idx = static_cast<size_t>(idx_ll);
            const PairData&  pij = pairs[idx];
            const int n_ij = pij.n_pno;
            if (n_ij == 0) { DF_per_pair[idx].resize(0, 0); continue; }
            DF_per_pair[idx].setZero(n_ij, n_ij);

            RowMatXd barS_kl;       // [n_ij × n_kl]
            RowMatXd pi_kl;         // [n_ij × n_kl]
            for (int k = 0; k < nocc; ++k) {
                for (int l = 0; l < nocc; ++l) {
                    const int idx_kl = pair_lookup[k * nocc + l];
                    const PairSetup& skl = setups[idx_kl];
                    const PairData&  pkl = pairs[idx_kl];
                    const int n_kl = pkl.n_pno;
                    if (n_kl == 0) continue;

                    // barS^(ij,kl) projection
                    compute_barS_lambda(
                        pij.bar_Q.data(), n_ij,
                        pkl.bar_Q.data(), n_kl,
                        h_S, nao, barS_kl);

                    Eigen::Map<const RowMatXd> Y_kl(
                        pkl.Y.data(), n_kl, n_kl);

                    // π_{kl}^{ij}[a, d] = barS · Y_kl^{oriented} · barS^T
                    //   oriented so (k, l) plays the "first" role
                    //   (transpose Y when stored pair is (l, k))
                    if (skl.i != k) {
                        pi_kl.noalias() =
                            barS_kl * Y_kl.transpose() * barS_kl.transpose();
                    } else {
                        pi_kl.noalias() =
                            barS_kl * Y_kl * barS_kl.transpose();
                    }
                    // pi_kl is here [n_ij × n_ij] after projection.

                    // T_pair^{(ij)}[k, c, l, d] is in phase24->T_pair[idx]
                    // indexed at ((k*nocc + l) * n_ij + c) * n_ij + d.
                    const real_t* T_ij =
                        phase24->T_pair[idx].data();
                    const size_t stride_kl_ij =
                        static_cast<size_t>(n_ij) * n_ij;
                    Eigen::Map<const RowMatXd> T_kl_ij(
                        T_ij + static_cast<size_t>(k * nocc + l) * stride_kl_ij,
                        n_ij, n_ij);
                    // DF[a, c] -= Σ_d π[a, d] · T_kl[c, d]
                    //          = -(π · T_kl^T)[a, c]
                    DF_per_pair[idx].noalias() -=
                        pi_kl * T_kl_ij.transpose();
                }
            }
        }
    } else if (enable_dressing) {
        // Sub-step 2X.2a fallback: only diagonal ΔF_ii from local L^{(ii)}.
        // Off-diagonal dF_ki and per-pair DF stay at zero.
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
            dF_ki[static_cast<size_t>(i) * nocc + i] = s;
        }
    }

    // ================================================================
    // Sub-step 2X.3.2 + 2X.3.6a: Λ_1 residual = T2-driven source (const)
    //                          + L1-driven self-dressing (terms 4 + 5).
    //
    // At T1 = 0 the surviving canonical Λ_1 source terms (verified
    // against ccsd_lambda.cu update_lambda_full, lines 813-1041) split
    // into:
    //
    //   R0[i, α]  (constant in Λ iter — depends only on T2):
    //     term 1 :  + 2·Σ_bc OVVV(i, a, c, b)·mvv1(b, c)
    //               -   Σ_bc OVVV(i, b, c, a)·mvv1(b, c)
    //               (Sub-step 2X.3.2; OVVV from W_ovvv_diag[i],
    //                mvv1 = DF_per_pair[idx_ii] in pair (i,i) PNO)
    //
    //   R_self[i, α]  (re-evaluated each iter — Λ_1-driven):
    //     term 4 :  - Σ_β L1_old[i, β] · Mvv1_pao[i][β, α]
    //               (vv self-dressing, off-diag of canonical L1·v1;
    //                Mvv1_pao = M^{(ii)} · DF_per_pair[idx_ii] · M^{(ii)T}
    //                placed in pair (i,i) PAO basis to match L1 storage)
    //     term 5 :  - Σ_j L1_old[j, α_in_ii_pao] · moo[i, j]
    //               (oo self-dressing, off-diag of canonical -L1·v2;
    //                moo[i, j] = dF_ki[j·nocc+i], reuses Sub-step 2X.2c
    //                Phase24 cross-pair F-eff dressing; for j ≠ i a
    //                cross-pair barS transform brings L1[j] from pair
    //                (j,j) PAO basis into pair (i,i) PAO basis.)
    //
    // Jacobi update: Λ_1[i, α] -= (R0 + R_self)[i, α] / (eps_a[α] - F_ii).
    // L1 self-iter is interleaved with the Λ_2 sweep inside the main
    // iter loop below. Convergence is reached when max(|R_L2|, |R_L1|)
    // < conv_tol.
    //
    // Higher-order terms (θ·L1 cross — term 2; L1·OVVO/OOVV — term 6;
    // OVOO·moo1 — term 3; L2-coupled sources — terms 7, 8) are deferred
    // to Sub-steps 2X.3.6b/c, 2X.3.7a-c.
    // ================================================================

    std::vector<std::vector<real_t>> R0_pao(nocc);
    std::vector<RowMatXd>            Mvv1_pao(nocc);

    if (use_phase24) {
        for (int i = 0; i < nocc; ++i) {
            const int idx_ii = pair_lookup[i * nocc + i];
            if (idx_ii < 0 || idx_ii >= static_cast<int>(setups.size())) continue;
            const PairSetup& s_ii = setups[idx_ii];
            const PairData&  p_ii = pairs[idx_ii];
            const int n_pno_ii = p_ii.n_pno;
            const int n_pao_ii = s_ii.n_pao;
            if (n_pno_ii == 0 || n_pao_ii == 0) continue;
            if (DF_per_pair[idx_ii].rows() != n_pno_ii ||
                DF_per_pair[idx_ii].cols() != n_pno_ii) continue;

            const RowMatXd& mvv1_pno = DF_per_pair[idx_ii];
            Eigen::Map<const RowMatXd> M(p_ii.M.data(), n_pao_ii, n_pno_ii);

            // --- term 1 source (constant in Λ iter) ---
            // R0_pno[a] = 2·Σ_{b,c} W(a,c,b)·mvv1(b,c) - Σ_{b,c} W(b,c,a)·mvv1(b,c)
            if (i < static_cast<int>(phase24->W_ovvv_diag.size())
                && !phase24->W_ovvv_diag[i].empty()) {
                const real_t* W = phase24->W_ovvv_diag[i].data();
                Eigen::VectorXd R_pno = Eigen::VectorXd::Zero(n_pno_ii);
                for (int a = 0; a < n_pno_ii; ++a) {
                    real_t r = 0.0;
                    for (int c = 0; c < n_pno_ii; ++c) {
                        for (int b = 0; b < n_pno_ii; ++b) {
                            const size_t idx_acb =
                                (static_cast<size_t>(a) * n_pno_ii + c) * n_pno_ii + b;
                            const size_t idx_bca =
                                (static_cast<size_t>(b) * n_pno_ii + c) * n_pno_ii + a;
                            const real_t m_bc = mvv1_pno(b, c);
                            r += 2.0 * W[idx_acb] * m_bc
                               -       W[idx_bca] * m_bc;
                        }
                    }
                    R_pno(a) = r;
                }
                const Eigen::VectorXd R_pao_vec = M * R_pno;
                R0_pao[i].assign(R_pao_vec.data(),
                                 R_pao_vec.data() + n_pao_ii);
            } else {
                R0_pao[i].assign(n_pao_ii, 0.0);
            }

            // --- Mvv1_pao[i] = M · mvv1_pno · M^T (n_pao_ii × n_pao_ii) ---
            // Used by term 4 self-dressing inside the iter loop.
            Mvv1_pao[i].noalias() = M * mvv1_pno * M.transpose();
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

            // ---- Sub-step 2X.2a/c: intra-pair diagonal hole dressing ----
            // R -= (ΔF_ii + ΔF_jj) · Λ_old
            // (Mirrors Phase 2.3.3 of iterate_dlpno_ccsd_t2 line 1110-1119.
            //  Both 2X.2a (fallback, diagonal only) and 2X.2c (full Phase24
            //  path) store ΔF_ii in dF_ki[i*nocc+i].)
            if (enable_dressing) {
                const real_t dF_sum =
                    dF_ki[static_cast<size_t>(sij.i) * nocc + sij.i]
                  + dF_ki[static_cast<size_t>(sij.j) * nocc + sij.j];
                if (dF_sum != 0.0) {
                    R_buf.noalias() -= dF_sum * RowMatXd(L2_old_ij);
                }
            }

            // ---- Cross-pair Fock coupling on i (k != sij.i) ----
            // F_eff[i, k] = F_LMO[i, k] + ΔF_{ki}    (full off-diagonal F_eff
            //                                         is built only when
            //                                         use_phase24 == true;
            //                                         otherwise dF_ki off-
            //                                         diagonal stays zero).
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

                // barS^(ij,kj) = bar_Q_ij^T · S · bar_Q_kj  [n × n_kj]
                compute_barS_lambda(pij.bar_Q.data(), n,
                                    pkj.bar_Q.data(), pkj.n_pno,
                                    h_S, nao, barS_buf);

                Eigen::Map<const RowMatXd> L2_kj(L2_old[idx_kj].data(),
                                                 pkj.n_pno, pkj.n_pno);

                // π_kj^(ij) = barS · Λ_kj · barS^T   [n × n]
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

            // ---- Cross-pair Fock coupling on j (l != sij.j) ----
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

            // ---- Sub-step 2X.2c: particle F_eff dressing per pair (ij) ----
            //
            // Canonical CCSD Λ_2 picks up two terms from ∂L/∂t_2:
            //   R += Σ_c Λ_2[ij,a,c] · F̃_v[c,b]     (b-side)
            //   R += Σ_c F̃_v[c,a] · Λ_2[ij,c,b]     (a-side, via P-symmetry)
            // i.e., R += Λ · F̃_v + F̃_v^T · Λ
            //   where F̃_v == DF_per_pair[idx] in our storage convention.
            //
            // The T_2 iter applies the *transposed* dressing (R += F̃ · Y +
            // Y · F̃^T) because T_2 sits on the other side of the Lagrangian
            // derivative. In semi-canonical DLPNO PNO basis F̃ is NOT
            // symmetric, so the transpose matters.
            //
            // Strict-mode validation (Sub-step 2X.2c sentinel test) locks
            // the sign and transpose convention against canonical CCSD Λ_2.
            if (use_phase24) {
                const RowMatXd& DF = DF_per_pair[idx];
                if (DF.size() == static_cast<Eigen::Index>(n) * n) {
                    R_buf.noalias() += RowMatXd(L2_old_ij) * DF;
                    R_buf.noalias() += DF.transpose() * RowMatXd(L2_old_ij);
                }
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

        // ============================================================
        // Sub-step 2X.3.6a: Λ_1 self-iter sweep (terms 4 + 5).
        //
        // Direct Jacobi mirroring the Λ_2 sweep convention (line 558-573):
        // the eigenvalue diagonal (ε_α - F_ii)·Λ_1_old enters R alongside
        // the off-diagonal source terms, so `Λ_1 -= R/denom` overwrites Λ_1
        // each iter (`L1_old · diag / denom = L1_old` cancels Λ_1_old).
        //
        // At T1 = 0 the residual is
        //   R[i, α] = R0[i, α]                                        (term 1, const)
        //           + (ε_α - F_ii) · Λ_1_old[i, α]                    (eigenvalue diag)
        //           - (Mvv1_pao[i]^T · Λ_1_old[i])[α]                 (term 4, vv)
        //           - Σ_j Λ_1_old[j → ii_pao basis](α) · moo[i, j]    (term 5, oo)
        // At convergence R = 0 ⇒ Λ_1[i, α] = (R_offdiag without diag)/denom,
        // matching canonical Stanton-Bartlett Λ_1 fixed point.
        //
        // Cross-basis transform for term 5 off-diagonal j ≠ i:
        //   L1_j_in_ii_pao = M^{(ii)} · barS^{(ii,jj)} · M^{(jj)T} · L1[j]
        // mirrors the cross-pair Fock coupling pattern used in the Λ_2
        // sweep above (line 469-538).
        // ============================================================
        if (use_phase24) {
            std::vector<std::vector<real_t>> L1_old(Lambda1);

            RowMatXd barS_ii_jj;
            for (int i = 0; i < nocc; ++i) {
                const int idx_ii = pair_lookup[i * nocc + i];
                if (idx_ii < 0 || idx_ii >= static_cast<int>(setups.size())) continue;
                const PairSetup& s_ii = setups[idx_ii];
                const PairData&  p_ii = pairs[idx_ii];
                const int n_pao_ii = s_ii.n_pao;
                const int n_pno_ii = p_ii.n_pno;
                if (n_pao_ii == 0 || n_pno_ii == 0) continue;
                if (static_cast<int>(R0_pao[i].size()) != n_pao_ii) continue;
                if (Mvv1_pao[i].rows() != n_pao_ii ||
                    Mvv1_pao[i].cols() != n_pao_ii) continue;

                const real_t F_ii_local =
                    F_LMO[static_cast<size_t>(i) * nocc + i];

                // Start from constant T2 source.
                Eigen::VectorXd R = Eigen::Map<const Eigen::VectorXd>(
                    R0_pao[i].data(), n_pao_ii);

                // term 4: R -= Mvv1_pao[i]^T · L1_old[i]
                const bool have_L1_i =
                    (static_cast<int>(L1_old[i].size()) == n_pao_ii);
                if (have_L1_i) {
                    Eigen::Map<const Eigen::VectorXd> L1_i(
                        L1_old[i].data(), n_pao_ii);
                    R.noalias() -= Mvv1_pao[i].transpose() * L1_i;
                }

                // term 5: R -= Σ_j L1_old[j → ii_pao basis] · moo[i, j]
                // Build convention: dF_ki[a·nocc+b] = canonical moo[a, b]
                // (first index a, second b). Canonical term 5 wants moo[i, j],
                // so use dF_ki[i·nocc+j] (not the transpose).
                Eigen::Map<const RowMatXd> M_ii(
                    p_ii.M.data(), n_pao_ii, n_pno_ii);
                for (int j = 0; j < nocc; ++j) {
                    const real_t moo_ij =
                        dF_ki[static_cast<size_t>(i) * nocc + j];
                    if (std::fabs(moo_ij) < kFLMOThresh) continue;
                    if (static_cast<int>(L1_old[j].size()) == 0) continue;

                    if (j == i) {
                        // Same basis — direct scaling.
                        Eigen::Map<const Eigen::VectorXd> L1_j(
                            L1_old[j].data(), n_pao_ii);
                        R.noalias() -= moo_ij * L1_j;
                    } else {
                        const int idx_jj = pair_lookup[j * nocc + j];
                        if (idx_jj < 0
                            || idx_jj >= static_cast<int>(setups.size()))
                            continue;
                        const PairSetup& s_jj = setups[idx_jj];
                        const PairData&  p_jj = pairs[idx_jj];
                        const int n_pao_jj = s_jj.n_pao;
                        const int n_pno_jj = p_jj.n_pno;
                        if (n_pao_jj == 0 || n_pno_jj == 0) continue;
                        if (static_cast<int>(L1_old[j].size()) != n_pao_jj)
                            continue;

                        // L1[j] in pair (j,j) PAO → pair (j,j) PNO
                        Eigen::Map<const RowMatXd> M_jj(
                            p_jj.M.data(), n_pao_jj, n_pno_jj);
                        Eigen::Map<const Eigen::VectorXd> L1_j(
                            L1_old[j].data(), n_pao_jj);
                        const Eigen::VectorXd L1_j_pno_jj = M_jj.transpose() * L1_j;

                        // pair (j,j) PNO → pair (i,i) PNO via barS^{(ii,jj)}
                        compute_barS_lambda(p_ii.bar_Q.data(), n_pno_ii,
                                            p_jj.bar_Q.data(), n_pno_jj,
                                            h_S, nao, barS_ii_jj);
                        const Eigen::VectorXd L1_j_pno_ii =
                            barS_ii_jj * L1_j_pno_jj;

                        // pair (i,i) PNO → pair (i,i) PAO via M_ii
                        const Eigen::VectorXd L1_j_pao_ii =
                            M_ii * L1_j_pno_ii;

                        R.noalias() -= moo_ij * L1_j_pao_ii;
                    }
                }

                // ================================================
                // Sub-step 2X.3.6b: term 6 — L1·OVVO and L1·OOVV
                //   R[i, α] += 2 · Σ_{jb} L1[j, b]·OVVO[i, α, b, j]
                //              -   Σ_{jb} L1[j, b]·OOVV[i, j, b, α]
                //
                // Pair (i,j) is stored at idx_ij with s.i ≤ s.j. We pick
                // the appropriate OVVO orientation:
                //   - if i == s.i → W_ovvo_lambda (i-role: (s.i a | b s.j))
                //   - if i == s.j → W_ovvo_lambda_alt (j-role: (s.j a | b s.i))
                // OOVV is symmetric in the LMO pair indices so a single
                // W_oovv_lambda storage covers both orientations.
                //
                // Steps:
                //   1. Transform L1[j_pao_jj] → L1_pno_ij via M^(jj)^T then
                //      barS^(ij, jj).
                //   2. Contract:
                //        R_pno_ij[a] = 2·W_ovvo_oriented[a,b]·L1_j[b]
                //                    -   W_oovv[b,a]·L1_j[b]
                //   3. Transform R_pno_ij → R_pao_ii via barS^(ii, ij), then
                //      M^(ii), and accumulate into R.
                // ================================================
                RowMatXd barS_ij_jj_b, barS_ii_ij_b;
                for (int j = 0; j < nocc; ++j) {
                    const int idx_ij = pair_lookup[i * nocc + j];
                    if (idx_ij < 0
                        || idx_ij >= static_cast<int>(setups.size()))
                        continue;
                    if (idx_ij >= static_cast<int>(
                            phase24->W_ovvo_lambda.size()))
                        continue;
                    if (phase24->W_ovvo_lambda[idx_ij].empty()) continue;

                    const PairSetup& s_ij = setups[idx_ij];
                    const PairData&  p_ij = pairs[idx_ij];
                    const int n_pno_ij = p_ij.n_pno;
                    if (n_pno_ij == 0) continue;

                    const int idx_jj = pair_lookup[j * nocc + j];
                    if (idx_jj < 0
                        || idx_jj >= static_cast<int>(setups.size()))
                        continue;
                    const PairSetup& s_jj = setups[idx_jj];
                    const PairData&  p_jj = pairs[idx_jj];
                    const int n_pao_jj = s_jj.n_pao;
                    const int n_pno_jj = p_jj.n_pno;
                    if (n_pao_jj == 0 || n_pno_jj == 0) continue;
                    if (static_cast<int>(L1_old[j].size()) != n_pao_jj)
                        continue;

                    // Select OVVO orientation based on which LMO of the
                    // stored pair matches the iter's `i` index.
                    const real_t* W_ovvo_data = nullptr;
                    if (s_ij.i == i) {
                        W_ovvo_data = phase24->W_ovvo_lambda[idx_ij].data();
                    } else if (s_ij.j == i) {
                        if (idx_ij >= static_cast<int>(
                                phase24->W_ovvo_lambda_alt.size()))
                            continue;
                        if (phase24->W_ovvo_lambda_alt[idx_ij].empty())
                            continue;
                        W_ovvo_data =
                            phase24->W_ovvo_lambda_alt[idx_ij].data();
                    } else {
                        continue;  // pair (i,j) lookup mismatch — shouldn't happen
                    }

                    // Step 1: L1[j] in pair (j,j) PAO → pair (i,j) PNO
                    Eigen::Map<const RowMatXd> M_jj(
                        p_jj.M.data(), n_pao_jj, n_pno_jj);
                    Eigen::Map<const Eigen::VectorXd> L1_j(
                        L1_old[j].data(), n_pao_jj);
                    const Eigen::VectorXd L1_j_pno_jj =
                        M_jj.transpose() * L1_j;
                    compute_barS_lambda(p_ij.bar_Q.data(), n_pno_ij,
                                        p_jj.bar_Q.data(), n_pno_jj,
                                        h_S, nao, barS_ij_jj_b);
                    const Eigen::VectorXd L1_j_pno_ij =
                        barS_ij_jj_b * L1_j_pno_jj;

                    // Step 2: contract with W_ovvo (oriented) / W_oovv
                    Eigen::Map<const RowMatXd> W_ovvo(
                        W_ovvo_data, n_pno_ij, n_pno_ij);
                    Eigen::Map<const RowMatXd> W_oovv(
                        phase24->W_oovv_lambda[idx_ij].data(),
                        n_pno_ij, n_pno_ij);
                    Eigen::VectorXd R_pno_ij =
                        2.0 * (W_ovvo * L1_j_pno_ij)
                        - (W_oovv.transpose() * L1_j_pno_ij);

                    // Step 3: pair (i,j) PNO → pair (i,i) PNO via barS^(ii,ij),
                    //         then pair (i,i) PNO → pair (i,i) PAO via M_ii.
                    compute_barS_lambda(p_ii.bar_Q.data(), n_pno_ii,
                                        p_ij.bar_Q.data(), n_pno_ij,
                                        h_S, nao, barS_ii_ij_b);
                    const Eigen::VectorXd R_pno_ii =
                        barS_ii_ij_b * R_pno_ij;
                    const Eigen::VectorXd R_contrib =
                        M_ii * R_pno_ii;

                    R.noalias() += R_contrib;
                }

                // ================================================
                // Sub-step 2X.3.7a: term 3 — OVOO·moo1 T2-source
                //   R[i, α] += -2 · Σ_{j,k} OVOO[i, α, j, k]·moo1[k, j]
                //              +   Σ_{j,k} OVOO[j, α, i, k]·moo1[k, j]
                //
                // moo1[k, j] = dF_ki[j·nocc + k] (existing DLPNO usage
                // convention, consistent with term 5).
                //
                // For each pair (i, j) [j summed]:
                //   1st term uses W_ovoo (orientation A:
                //     s.i == i picks W_ovoo_lambda; s.j == i picks alt)
                //   2nd term uses W_ovoo with OPPOSITE orientation
                //     (selects the j-LMO-first storage)
                //
                // Steps:
                //   R_pno_ij[a] = -2·W_first[a,k]·moo1[k,j] + W_second[a,k]·moo1[k,j]
                //               summed over k.
                //   Transform R_pno_ij → R_pao_ii via barS^(ii, ij), then M^(ii).
                // ================================================
                RowMatXd barS_ii_ij_3;
                Eigen::VectorXd moo1_col_j(nocc);
                for (int j = 0; j < nocc; ++j) {
                    const int idx_ij = pair_lookup[i * nocc + j];
                    if (idx_ij < 0
                        || idx_ij >= static_cast<int>(setups.size()))
                        continue;
                    if (idx_ij >= static_cast<int>(
                            phase24->W_ovoo_lambda.size()))
                        continue;
                    if (phase24->W_ovoo_lambda[idx_ij].empty()) continue;
                    if (phase24->W_ovoo_lambda_alt[idx_ij].empty()) continue;

                    const PairSetup& s_ij = setups[idx_ij];
                    const PairData&  p_ij = pairs[idx_ij];
                    const int n_pno_ij = p_ij.n_pno;
                    if (n_pno_ij == 0) continue;

                    // Select orientation. For 1st term OVOO[i, α, j, k]:
                    //   first LMO == i_loop → choose storage by s.i
                    // For 2nd term OVOO[j, α, i, k]:
                    //   opposite orientation
                    const real_t* W_first  = nullptr;
                    const real_t* W_second = nullptr;
                    if (s_ij.i == i) {
                        W_first  = phase24->W_ovoo_lambda[idx_ij].data();
                        W_second = phase24->W_ovoo_lambda_alt[idx_ij].data();
                    } else if (s_ij.j == i) {
                        W_first  = phase24->W_ovoo_lambda_alt[idx_ij].data();
                        W_second = phase24->W_ovoo_lambda[idx_ij].data();
                    } else {
                        continue;  // shouldn't happen
                    }

                    // Build moo1_col_j[k] = moo1[k, j] = dF_ki[k·nocc + j]
                    // (build convention: dF_ki[a·nocc+b] = canonical moo[a, b]).
                    for (int k = 0; k < nocc; ++k) {
                        moo1_col_j(k) = dF_ki[
                            static_cast<size_t>(k) * nocc + j];
                    }

                    // R_pno_ij[a] = -2 · Σ_k W_first[a, k]·moo1_col_j[k]
                    //              +   Σ_k W_second[a, k]·moo1_col_j[k]
                    Eigen::Map<const RowMatXd> W_first_mat(
                        W_first, n_pno_ij, nocc);
                    Eigen::Map<const RowMatXd> W_second_mat(
                        W_second, n_pno_ij, nocc);
                    Eigen::VectorXd R_pno_ij =
                        -2.0 * (W_first_mat  * moo1_col_j)
                        +       (W_second_mat * moo1_col_j);

                    // Transform R_pno_ij → R_pao_ii.
                    compute_barS_lambda(p_ii.bar_Q.data(), n_pno_ii,
                                        p_ij.bar_Q.data(), n_pno_ij,
                                        h_S, nao, barS_ii_ij_3);
                    const Eigen::VectorXd R_pno_ii =
                        barS_ii_ij_3 * R_pno_ij;
                    const Eigen::VectorXd R_contrib_3 =
                        M_ii * R_pno_ii;

                    R.noalias() += R_contrib_3;
                }

                // ================================================
                // Sub-step 2X.3.7c (DEFERRED): term 8 — L2·OVOO
                //
                // Attempted leading-order-only implementation (WOOVO ≈
                // OVOO1[k, c, j, i], i.e. line 440 of canonical WOOVO build)
                // gave Λ_1 norm overshoot (2.08e-2 → 4.81e-2) and dipole
                // undershoot (1.5921 → 1.4707 vs canonical 1.5775).
                //
                // Root cause: canonical WOOVO at T1=0 includes T2-dressed
                // corrections (canonical ccsd_lambda.cu lines 441-444):
                //   line 441: woovo[i,k,b,j] += ovoo1·theta
                //   line 442: woovo[i,j,b,k] -= ovoo1·t2
                //   line 443: woovo[i,j,b,k] -= ovoo1·t2 (different contraction)
                //   line 444: woovo[i,j,c,k] += ovvv·tau    ← requires per-pair
                //                                            OVVV (heavy, TEOS
                //                                            issue per term 7)
                //
                // Without these corrections, term 8 alone over-contributes.
                // Proper term 8 implementation needs to be deferred and
                // bundled with term 7 (L2·OVVV) + full WOOVO dressings.
                // ================================================

                // Add eigenvalue diagonal (ε_α - F_ii)·Λ_1_old[i, α] so
                // `Λ_1 -= R/denom` overwrites Λ_1[i, α] in one Jacobi step
                // (direct convention, same as the Λ_2 sweep above).
                if (have_L1_i) {
                    for (int alpha = 0; alpha < n_pao_ii; ++alpha) {
                        R(alpha) += (s_ii.eps_a[alpha] - F_ii_local)
                                  * L1_old[i][alpha];
                    }
                }

                // Jacobi update + convergence tracking.
                real_t r_max_l1_i = 0.0;
                for (int alpha = 0; alpha < n_pao_ii; ++alpha) {
                    const real_t denom = s_ii.eps_a[alpha] - F_ii_local;
                    if (std::fabs(denom) < 1.0e-14) continue;
                    Lambda1[i][alpha] -= R(alpha) / denom;
                    r_max_l1_i =
                        std::max(r_max_l1_i, std::fabs(R(alpha)));
                }
                r_max = std::max(r_max, r_max_l1_i);
            }
        }

        s.iters = iter + 1;
        s.max_R = r_max;

        if (verbose >= 2) {
            real_t lam1_norm_sq = 0.0;
            for (int i = 0; i < nocc; ++i) {
                for (real_t v : Lambda1[i]) lam1_norm_sq += v * v;
            }
            std::cout << "[" << (round_tag ? round_tag : "DLPNO-Λ")
                      << "] iter " << s.iters
                      << "  max|R|=" << std::scientific
                      << std::setprecision(3) << r_max
                      << "  |Λ_1|=" << std::scientific
                      << std::setprecision(8) << std::sqrt(lam1_norm_sq)
                      << std::endl;
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

        if (use_phase24) {
            real_t lam1_norm_sq = 0.0;
            for (int i = 0; i < nocc; ++i) {
                for (real_t v : Lambda1[i]) lam1_norm_sq += v * v;
            }
            std::cout << "[" << (round_tag ? round_tag : "DLPNO-Λ")
                      << "] Λ_1 norm (after T2 source) = "
                      << std::scientific << std::setprecision(8)
                      << std::sqrt(lam1_norm_sq)
                      << "  (self-iter ran " << s.iters << " iter)"
                      << std::endl;
        }
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
    real_t*                                  D_mo_out,
    const std::vector<std::vector<real_t>>&  Lambda1)
{
    // 1. oo + vv + HF blocks (reuse MP2 path; ov/vo left at zero).
    build_dlpno_mp2_1rdm_mo(setups, pairs, pair_lookup,
                            n_lmo, n_can_vir, S_AO, C_can_vir, nao,
                            D_mo_out);

    const int nmo = n_lmo + n_can_vir;
    Eigen::Map<const RowMatXd> S(S_AO,        nao, nao);
    Eigen::Map<const RowMatXd> C(C_can_vir,   nao, n_can_vir);

    // For each LMO i, back-transform T1[i] and Λ_1[i] (both in pair (i,i)'s
    // PAO basis) to the canonical virtual basis. Both contributions sum
    // into the D[ov]/D[vo] block following the canonical CCSD 1-RDM
    // formula D[ov][i,a] = dov[i,a] + dvo[a,i] = L1[i,a] + T1[i,a] + …
    // (cross-terms O(T1·L1) deferred to Sub-step 2X.3.5 — only fire when
    // T1 is non-zero, which requires Phase 2.6c to land first).
    std::vector<real_t> d_buf;  // d_ii [n_can_vir × n_pao]

    const bool have_T1 =
        !T1.empty() && static_cast<int>(T1.size()) == n_lmo;
    const bool have_L1 =
        !Lambda1.empty() && static_cast<int>(Lambda1.size()) == n_lmo;
    if (!have_T1 && !have_L1) return;

    for (int i = 0; i < n_lmo; ++i) {
        const int idx_ii = pair_lookup[i * n_lmo + i];
        if (idx_ii < 0 || idx_ii >= static_cast<int>(setups.size())) continue;
        const auto& s = setups[idx_ii];
        const int n_pao = s.n_pao;
        if (n_pao == 0) continue;

        const bool i_has_T1 =
            have_T1 && static_cast<int>(T1[i].size()) == n_pao;
        const bool i_has_L1 =
            have_L1 && static_cast<int>(Lambda1[i].size()) == n_pao;
        if (!i_has_T1 && !i_has_L1) continue;

        // d_ii[a, a_pao] = (C_can_vir^T · S_AO · C_can_pair_ii)[a, a_pao]
        d_buf.assign(static_cast<size_t>(n_can_vir) * n_pao, 0.0);
        Eigen::Map<const RowMatXd> Cii(s.C_can_pair.data(), nao, n_pao);
        RowMatXd SCii = S * Cii;                  // [nao × n_pao]
        RowMatXd d    = C.transpose() * SCii;     // [n_can_vir × n_pao]

        // Combined T1 + Λ_1 vector in pair (i,i)'s PAO basis.
        Eigen::VectorXd amp_pao = Eigen::VectorXd::Zero(n_pao);
        if (i_has_T1) {
            for (int a = 0; a < n_pao; ++a)
                amp_pao(a) += T1[i][a];
        }
        if (i_has_L1) {
            for (int a = 0; a < n_pao; ++a)
                amp_pao(a) += Lambda1[i][a];
        }
        Eigen::VectorXd amp_can = d * amp_pao;     // [n_can_vir]

        // Place into D[ov] and D[vo] (Hermitian).
        for (int a = 0; a < n_can_vir; ++a) {
            const real_t v = amp_can(a);
            D_mo_out[static_cast<size_t>(i) * nmo + (n_lmo + a)] += v;
            D_mo_out[static_cast<size_t>(n_lmo + a) * nmo + i]   += v;
        }
    }
}

} // namespace gansu
