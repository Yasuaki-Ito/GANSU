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
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "types.hpp"

namespace gansu {

/**
 * @brief Scoped cap on the OpenMP default thread count for the DLPNO CPU path.
 *
 * The per-pair `omp parallel for` loops in iterate_lmp2 /
 * iterate_dlpno_ccsd_t2 / PNO selection have no num_threads() clause, so they
 * default to one thread per logical core. On machines with > ~128 cores each
 * such thread calls Eigen→OpenBLAS, and OpenBLAS allocates one buffer per
 * distinct caller thread with a hard precompiled limit (128); exceeding it
 * corrupts state and segfaults (observed in DLPNO-CCSD(T) (T) phase). Capping
 * the OpenMP default keeps the OpenBLAS caller count bounded. The explicit
 * `num_threads(num_gpus)` GPU-dispatch regions are unaffected (an explicit
 * clause overrides the default). The previous value is restored on scope
 * exit, so no process-wide thread-count state leaks beyond the DLPNO driver.
 *
 * @param requested user value (dlpno_cpu_threads); <= 0 means auto =
 *        min(available cores, 64).
 */
struct OmpThreadCapGuard {
#ifdef _OPENMP
    int saved_;
    explicit OmpThreadCapGuard(int requested) : saved_(omp_get_max_threads()) {
        int cap = requested > 0 ? requested : std::min(saved_, 64);
        if (cap < 1) cap = 1;
        omp_set_num_threads(cap);
    }
    ~OmpThreadCapGuard() { omp_set_num_threads(saved_); }
#else
    explicit OmpThreadCapGuard(int /*requested*/) {}
#endif
    OmpThreadCapGuard(const OmpThreadCapGuard&) = delete;
    OmpThreadCapGuard& operator=(const OmpThreadCapGuard&) = delete;
};

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
    /// Λ_2 (lambda) amplitudes in W basis, same shape as Y.
    /// For DLPNO-MP2 (Sub-phase 1 of the DLPNO-CCSD-Λ project): closed-form
    ///   Y_lambda[a,b] = 2 Y[a,b] - Y[b,a]
    /// For DLPNO-CCSD (Sub-phase 2): updated iteratively via DLPNO Λ residual.
    /// Empty (size 0) if Lambda solver has not been run.
    /// Reference: Datta, Kossmann, Neese, J. Chem. Phys. 145, 114101 (2016)
    /// and c:\Users\yasuaki\Dropbox\AQUA\DLPNO_Lambda.md.
    std::vector<real_t> Y_lambda;         ///< [n_pno × n_pno]
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

    /// Sub-step 2X.3.1: OVVV integrals for diagonal pairs (i,i), used by
    /// the leading T2-driven Λ_1 source term
    ///   R_Λ_1[i,a_ii] ⊃ 2·Σ_bc W_ovvv_diag[i](a,c,b)·mvv1[b,c]
    ///                 -   Σ_bc W_ovvv_diag[i](b,c,a)·mvv1[b,c]
    /// where mvv1 in pair (i,i)'s PNO basis is recoverable from
    /// DF_per_pair[idx_ii] computed by the T iteration.
    ///
    /// Storage: one entry per LMO i (NOT per pair). Each entry holds
    /// (i, a, b, c) ≡ eri_mo[i, n_lmo+a, n_lmo+b, n_lmo+c] for pair (i,i),
    /// where a, b, c index the pair's PNO basis (n_pno_ii values each).
    /// Layout: ((a * n_pno_ii + b) * n_pno_ii + c).
    ///
    /// Scaling: nocc · n_pno_ii³, e.g. cholesterol (nocc=75, n_pno≈30) is
    /// ~16 MB total — fits even at TEOS scale.
    std::vector<std::vector<real_t>> W_ovvv_diag;    ///< [nocc] of [n_pno_ii³]

    /// Sub-step 2X.3.6b: OVVO and OOVV integrals per strong pair (i,j),
    /// used by the L1·OVVO / L1·OOVV cross-pair source terms in Λ_1:
    ///   R_Λ_1[i,α_ii] ⊃ +2 · Σ_{j,b} L_1[j,b]·OVVO[i,α,b,j]
    ///                  -   Σ_{j,b} L_1[j,b]·OOVV[i,j,b,α]
    /// (term 6 of the canonical Λ_1 catalogue, T1=0 limit).
    ///
    /// Pair storage: pairs are stored with s.i ≤ s.j. The Λ_1 iter
    /// computes R for both orderings (loop_i = s.i, loop_j = s.j) and
    /// (loop_i = s.j, loop_j = s.i), so two OVVO orientations are needed:
    ///   W_ovvo_lambda[idx][a, b]     = (s.i a | b s.j)  — "i-role"
    ///   W_ovvo_lambda_alt[idx][a, b] = (s.j a | b s.i)  — "j-role"
    /// (these are genuinely different ERIs; no 8-fold symmetry relates them.)
    ///
    /// OOVV is symmetric in the LMO pair indices via (pq|rs) = (qp|rs):
    ///   (s.i s.j | b a) = (s.j s.i | b a)
    /// so a single OOVV storage suffices for both orientations.
    ///
    /// Layout per pair: indices a, b are in pair (i,j) PNO basis (n_pno_ij²
    /// each). The L1 in the iter loop is back-rotated through barS chains:
    ///   L1_j_pno_ij = barS^(ij,jj) · M^(jj)^T · L1[j_pao_jj]
    /// and the contracted residual is forward-rotated to pair (i,i) PAO:
    ///   R_pao_ii += M^(ii) · barS^(ii,ij) · R_pno_ij
    /// matching the cross-pair Fock coupling pattern of the Λ_2 sweep.
    ///
    /// Layout (all row-major):
    ///   W_ovvo_lambda[idx][a·n_pno + b]     = eri_mo[s.i, n_lmo+a, n_lmo+b, s.j]
    ///   W_ovvo_lambda_alt[idx][a·n_pno + b] = eri_mo[s.j, n_lmo+a, n_lmo+b, s.i]
    ///   W_oovv_lambda[idx][b·n_pno + a]     = eri_mo[s.i, s.j, n_lmo+b, n_lmo+a]
    ///
    /// Scaling: n_pair_strong · n_pno_ij² · 24 B (3 × 8 B); TEOS-class
    /// (n_pair_strong ≈ 8e3, n_pno ≈ 50) sits at ~300 MB. Cholesterol-class
    /// (n_pair_strong ≈ 4e3, n_pno ≈ 30) is ~45 MB.
    std::vector<std::vector<real_t>> W_ovvo_lambda;     ///< [n_pairs] of [n_pno_ij²]
    std::vector<std::vector<real_t>> W_ovvo_lambda_alt; ///< [n_pairs] of [n_pno_ij²]
    std::vector<std::vector<real_t>> W_oovv_lambda;     ///< [n_pairs] of [n_pno_ij²]

    /// Sub-step 2X.3.7a: OVOO integrals per strong pair (i,j), used by the
    /// OVOO·moo1 T2-source term of Λ_1:
    ///   R_Λ_1[i,α_ii] ⊃ -2 · Σ_{j,k} OVOO[i, α, j, k]·moo1[k, j]
    ///                  +   Σ_{j,k} OVOO[j, α, i, k]·moo1[k, j]
    /// (term 3 of the canonical Λ_1 catalogue, T1=0 limit).
    /// Also shared with term 8 (L2·OVOO) in Sub-step 2X.3.7c.
    ///
    /// Pair storage with s.i ≤ s.j: the two OVOO orientations
    ///   W_ovoo_lambda[idx][a, k]     = (s.i a | s.j k)  — "i-role"
    ///   W_ovoo_lambda_alt[idx][a, k] = (s.j a | s.i k)  — "j-role"
    /// are stored together. Indices: a in pair (i,j) PNO basis, k LMO.
    /// Layout: a·nocc + k (row-major).
    ///
    /// Scaling: n_pair_strong · n_pno · nocc · 16 B (2 × 8 B). TEOS-class
    /// (n_pair_strong ≈ 8e3, n_pno ≈ 50, nocc ≈ 200) is ~1.3 GB.
    /// Cholesterol-class (n_pair_strong ≈ 4e3, n_pno ≈ 30, nocc ≈ 75)
    /// is ~150 MB.
    std::vector<std::vector<real_t>> W_ovoo_lambda;     ///< [n_pairs] of [n_pno·nocc]
    std::vector<std::vector<real_t>> W_ovoo_lambda_alt; ///< [n_pairs] of [n_pno·nocc]

    /// B-a.6c IP dense-free bare ph-ladder blocks, per pair (i,j), occ-role I=i/j.
    /// These are the dense-free seeds of the native DLPNO-IP-EOM per-pair PNO
    /// Wovvo_pno/Wovov_pno (ip_eom_ccsd_operator.cu:688/:663 bare terms), so the
    /// dense nocc²·nvir² Wovvo/Wovov is never materialised at 100 atoms:
    ///   W_ovvo_bare_i[idx][m,a',d'] = (m d'|a' s.i) = eri[m,n_lmo+d',n_lmo+a',s.i]
    ///   W_oovv_bare_i[idx][m,a',d'] = (m s.i|a' d') = eri[m,s.i,n_lmo+a',n_lmo+d']
    /// (the _j variants use s.j). Layout (m·n_pno + a')·n_pno + d', m an LMO,
    /// a',d' in pair (i,j)'s PNO. Size n_pno · nocc · n_pno (= W_ovov shape).
    std::vector<std::vector<real_t>> W_ovvo_bare_i;     ///< [n_pairs] of [nocc · n_pno²]
    std::vector<std::vector<real_t>> W_ovvo_bare_j;
    std::vector<std::vector<real_t>> W_oovv_bare_i;     ///< [n_pairs] of [nocc · n_pno²]
    std::vector<std::vector<real_t>> W_oovv_bare_j;

    /// Increment 2 / S1: VVOV + VOOO blocks per strong pair (i,j) for the
    /// linear T1→T2 back-coupling in the CCSD T2 residual (the spin-adapted,
    /// symmetrised reduction of the canonical raw(i,a,j,b) terms
    ///   + Σ_c (ab|ic) t1(j,c)   − Σ_k (ak|ij) t1(k,b)).
    ///
    /// Per pair (i,j) with PNO indices a,b,c and LMO index k:
    ///   W_vvov_i[idx][a,b,c] = (n_lmo+a, n_lmo+b | s.i,    n_lmo+c) = (ab|ic)
    ///   W_vvov_j[idx][a,b,c] = (n_lmo+a, n_lmo+b | s.j,    n_lmo+c) = (ab|jc)
    ///   W_vooo_i[idx][a,k]   = (n_lmo+a, k        | s.i,    s.j)    = (ak|ij)
    /// (no 8-fold symmetry relates the i/j orientations of vvov; vooo needs
    /// only the (i,j) ket since the j-role contraction reuses the same block
    /// with the virtual roles swapped — see the residual sweep).
    ///
    /// Layout (row-major): vvov ((a·n_pno + b)·n_pno + c); vooo (a·n_lmo + k).
    /// Scaling: vvov is n_pair_strong · n_pno³ · 16 B (two blocks); vooo is
    /// n_pair_strong · n_pno · n_lmo · 8 B (both ≲ the existing W_pair n_pno⁴).
    /// Only built/consumed when CCSD singles (T1_pao) are active.
    std::vector<std::vector<real_t>> W_vvov_i;   ///< [n_pairs] of [n_pno³]
    std::vector<std::vector<real_t>> W_vvov_j;   ///< [n_pairs] of [n_pno³]
    std::vector<std::vector<real_t>> W_vooo_i;   ///< [n_pairs] of [n_pno · n_lmo]
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
    int verbose, const char* round_tag,
    int num_gpus = 1,
    bool user_explicit_n_gpus = false);

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
    const Phase24Integrals* phase24 = nullptr,
    int num_gpus = 1,
    bool user_explicit_n_gpus = false,
    // Phase 2 (CCSD sparse barS) — per-LMO Mulliken centroids [nocc × 3] (xyz,
    // Bohr) for the distance-based coupling screen. nullptr = no distance
    // screen (keep all active = bit-exact). Used only when
    // GANSU_DLPNO_CCSD_BARS_SPARSE is on with GANSU_DLPNO_CCSD_BARS_DIST > 0.
    const real_t* lmo_centroids = nullptr,
    // Increment 2 (full singles, T1↔T2 coupling). T1_pao[i] = singles amplitude
    // of LMO i in pair (i,i)'s PAO basis (length n_pao_ii). nullptr (default) =
    // pure CCD, byte-identical to the legacy path. When non-null, the T2 residual
    // picks up the T1→T2 back-coupling (currently: the 4-virtual tau=t2+t1·t1
    // ladder piece; more terms added incrementally). The caller drives the
    // coupled macro-iteration (T2 sweep with current T1, then T1 sweep).
    const std::vector<std::vector<real_t>>* T1_pao = nullptr);

} // namespace gansu
