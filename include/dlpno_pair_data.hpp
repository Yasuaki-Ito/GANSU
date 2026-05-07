/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <vector>
#include "types.hpp"

namespace gansu {

/**
 * @brief Per-pair invariants (independent of PNO selection).
 *
 * Built once per pair from the AO-basis data and reused across all
 * SC-PNO refinement rounds and the LMP2/CCSD solvers.
 */
struct PairSetup {
    int i = 0;
    int j = 0;
    int n_pao = 0;
    real_t F_ii = 0.0;
    real_t F_jj = 0.0;
    real_t pair_factor = 1.0;             ///< 1 if i==j else 2

    std::vector<real_t> C_can_pair;       ///< [nao × n_pao] semi-canonical PAOs in AO basis
    std::vector<real_t> eps_a;            ///< [n_pao] semi-canonical PAO orbital energies
    std::vector<real_t> V;                ///< [n_pao × n_pao] (ia|jb) in semi-canonical PAO basis
};

/**
 * @brief PNO-dependent state for a single pair.
 *
 * Rebuilt at each SC-PNO refinement round from updated amplitudes.
 */
struct PairData {
    int n_pno = 0;
    /// PNOs in AO basis after F_pno-eigenvector rotation:
    ///   bar_Q = C_can_pair · D_pno · W   (nao × n_pno).
    /// Columns are S_AO-orthonormal: bar_Q^T · S · bar_Q = I.
    std::vector<real_t> bar_Q;
    /// M = D · W                          (n_pao × n_pno).
    /// Used to rotate amplitudes from W basis back to PAO basis:
    /// T_pao = M · Y · M^T.
    std::vector<real_t> M;
    /// F_pno eigenvalues in W basis (PNO-canonical virtual energies).
    std::vector<real_t> Lambda;           ///< [n_pno]
    /// L = W^T · D^T · V · D · W          (n_pno × n_pno).
    std::vector<real_t> L;
    /// Amplitudes in W basis. Updated in place during the Jacobi sweep.
    std::vector<real_t> Y;                ///< [n_pno × n_pno]
};

/**
 * @brief Build a PairData from supplied PAO-basis amplitudes T_pao.
 *
 * Steps performed:
 *   1. Diagonalise the pair density derived from T_pao to obtain PNOs.
 *   2. Truncate by occupation cutoff t_cut_pno.
 *   3. Project F into the PNO subspace and diagonalise → Lambda, W.
 *   4. Rotate K integrals into the W basis → L.
 *   5. Compute initial Y = -L / (Λ_a + Λ_b - F_ii - F_jj) (per-pair Sylvester).
 *
 * `os_only` controls the PNO density form:
 *   - false (default): D = T̃^T T + T̃ T^T, T̃ = 2T - T^T  (Riplinger 2013, full LMP2)
 *   - true: D = T^T T + T T^T  (Pinski 2015, OS-only — best with SOS-MP2)
 */
void build_pair_data(const PairSetup& s,
                     const std::vector<real_t>& T_pao,
                     real_t t_cut_pno,
                     bool   os_only,
                     int nao,
                     PairData& out);

/// Reconstruct T_pao = M · Y · M^T for the next SC-PNO round.
void reconstruct_T_pao(const PairSetup& s,
                       const PairData&  pd,
                       std::vector<real_t>& T_pao_out);

/// Result of a single Jacobi LMP2 sweep loop.
struct LMP2Status {
    int  iters     = 0;
    real_t max_R   = 0.0;
    bool converged = false;
};

/**
 * @brief Pre-computed integrals for Phase 2.4 F_eff dressing.
 *
 * For each pair (i,j) (storage index `idx = pair_lookup[i*nocc+j]`) with
 * n_pno > 0, holds the 4-tensor in pair (i,j)'s PNO basis:
 *
 *     T_pair^{(ij)}[k, c, l, d] = 2 (k c | l d) − (k d | l c)      (chem. notation)
 *
 * with k, l LMO indices (0..nocc-1) and c, d ∈ pair (i,j)'s PNO basis.
 * Indexing: `T_pair[idx][((k * nocc + l) * n_pno + c) * n_pno + d]`.
 *
 * Used to evaluate two pieces of the canonical CCSD dressing:
 *
 *   1. **Particle dressing** (Phase 2.3.2 + 2.4.2, full (k,l) sum):
 *        ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
 *        t_{kl,proj}^{ad} = (\bar S^{(ij,kl)} · Y_{kl} · \bar S^{(ij,kl)\,T})_{ad}
 *
 *   2. **Hole dressing** for pair (i,i) (Phase 2.3.3 + 2.4.1, l=i restr.):
 *        ΔF_{ki}^{l=i} = Σ_{cd} T_pair^{(ii)}[k,c,i,d] · Y_{ii}^{cd}
 *
 * Built once per CCSD computation via `precompute_phase24_integrals()`
 * (in dlpno_ccsd.cu — needs ERI access). Storage scales as
 * N_pair · N_occ² · n_pno² (manageable for ~50 atoms; later optimisation
 * via direct RI 3-index contraction is deferred).
 */
struct Phase24Integrals {
    int nocc = 0;
    std::vector<int> n_pno_per_pair;                 ///< [n_pairs]
    std::vector<std::vector<real_t>> T_pair;         ///< [n_pairs] of [nocc² · n_pno²]
    /// 4-virtual integral block W_pair^{(ij)}[a, b, c, d] = (ab|cd) in pair
    /// (i,j)'s PNO basis. Used by Phase 2.5 for the ladder contribution
    ///   ΔR^{(ij)}_{ab} += Σ_{cd} W_pair^{(ij)}[a, b, c, d] · Y_{ij,old}^{cd}.
    /// Indexing: ((a · n_pno + b) · n_pno + c) · n_pno + d.
    std::vector<std::vector<real_t>> W_pair;         ///< [n_pairs] of [n_pno⁴]

    /// Phase 2.6 (oooo): W_oooo^{(ij)}[k, l] = (kl|ij) for fixed pair (i,j).
    /// Indexing: k * nocc + l.
    std::vector<std::vector<real_t>> W_oooo;         ///< [n_pairs] of [nocc²]

    /// Phase 2.6 (ovov, particle-hole ladder, i-side):
    ///   W_aIic^{(ij)}[a, k, c] = (a k | I c) with I = pair's i, a,c in
    ///   pair (i,j)'s PNO, k an LMO. The j-side variant uses I = pair's j.
    /// Indexing: (a * nocc + k) * n_pno + c.
    std::vector<std::vector<real_t>> W_ovov_i;       ///< [n_pairs] of [n_pno · nocc · n_pno]
    std::vector<std::vector<real_t>> W_ovov_j;
    /// Phase 2.6 (ovvo): W_aIci^{(ij)}[a, k, c] = (a k | c I), I = pair's i/j.
    std::vector<std::vector<real_t>> W_ovvo_i;       ///< same shape
    std::vector<std::vector<real_t>> W_ovvo_j;

    /// Phase 2.6b (ovov, raw ERI): V_ovov_pair^{(ij)}[l, k, d, c] = (ld|kc).
    /// Indexing: ((l * nocc + k) * n_pno + d) * n_pno + c.
    /// Used in W_akic / W_akci ring-diagram dressing.
    std::vector<std::vector<real_t>> V_ovov_pair;    ///< [n_pairs] of [nocc² · n_pno²]

};

/**
 * @brief Iterative LMP2 amplitude solver in per-pair PNO+W basis.
 *
 * Solves the residual equation (Riplinger 2013, Pinski 2015):
 *   R^(ij) = L^(ij) + (Λ_a + Λ_b − F_ii − F_jj) Y^(ij)
 *            − Σ_{k≠i} F_LMO[i,k] · barS^(ij,kj) Y^(kj) barS^(ij,kj)^T
 *            − Σ_{l≠j} F_LMO[l,j] · barS^(ij,il) Y^(il) barS^(ij,il)^T
 * with barS^(ij,kj) = bar_Q_ij^T · S_AO · bar_Q_kj.
 *
 * Pairs are stored only for i ≤ j; closed-shell symmetry Y_kj = Y_jk^T is
 * applied when fetching swapped pairs (handled internally).
 *
 * `pair_lookup[i*nocc + j]` should map both (i,j) and (j,i) to the same
 * storage index.
 */
LMP2Status iterate_lmp2(
    const std::vector<PairSetup>& setups,
    std::vector<PairData>&        pairs,
    const std::vector<int>&       pair_lookup,
    const std::vector<real_t>&    F_LMO,
    const real_t*                 h_S,
    int nocc, int nao,
    int max_iter, real_t conv_tol,
    int verbose, const char* round_tag);

/**
 * @brief CCSD T2 amplitude iterator in per-pair PNO+W basis (Phase 2.3+).
 *
 * Extends `iterate_lmp2` to support optional CCSD-specific F_eff dressing.
 * The base residual structure (Λ shift, inter-pair F_LMO coupling via
 * bar_S projection) is shared with LMP2; the dressing flag toggles
 * incremental CCSD contributions in successive sub-phases:
 *
 *   - Phase 2.3.1 [done]: `enable_dressing=false` reproduces the LMP2
 *     residual exactly. Used as a sanity hook on top of the converged
 *     LMP2 amplitudes — must report `max_R ≈ 0`.
 *   - Phase 2.3.2 [done] / 2.4.2 [this commit]: particle F_ac dressing
 *         ΔF^{(ij)}_{ac} = -Σ_{kl,d} T_pair^{(ij)}[k,c,l,d] · t_{kl,proj}^{ad}
 *         t_{kl,proj} = \bar S^{(ij,kl)} · Y_{kl} · \bar S^{(ij,kl),T}
 *     contracted into R via (ΔF · Y_old + Y_old · ΔF^T)_{ab}.
 *     Phase 2.3.2 = (k,l)=(i,j) restriction; Phase 2.4.2 = full (k,l) sum
 *     (active when `phase24` is non-null).
 *   - Phase 2.3.3 [done] / 2.4.1 [done] / 2.4.3 [done]: hole F_eff[k,i]
 *     dressing, full l sum
 *         ΔF_{ki} = Σ_l Σ_{cd} T_pair^{(il)}[k,c,l,d] · Y_{il}^{cd}
 *         diagonal k=i  → −(ΔF_{ii}+ΔF_{jj}) Y^{(ij),old} shift (2.3.3),
 *         off-diag k≠i  → F_eff[i,k] = F_LMO[i,k] + ΔF_{ki} (2.4.1/3).
 *   - Phase 2.5 + 2.6 [INTEGRAL READY, residual gated off in iterate body]:
 *     4-virtual / oooo / particle-hole ladders. BARE form diverges;
 *     re-enable once DIIS or T2-dressed W's land.
 *   - Phase 2.7 [done]: DIIS extrapolation (8-vector subspace) over the
 *     concatenated per-pair Y vectors with ΔY = -R/D as error vector.
 *
 * Reference: see `c:/Users/yasuaki/Dropbox/AQUA/DLPNO_phase23_formulas.md`
 * for the term-by-term derivation and the PySCF reference verification.
 */
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
    const Phase24Integrals* phase24 = nullptr);

} // namespace gansu
