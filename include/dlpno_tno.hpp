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

#include "dlpno_pair_data.hpp"

#include <array>
#include <vector>

namespace gansu {

/**
 * @brief Triple Natural Orbital (TNO) data for a single (i, j, k) triple.
 *
 * Built from the union of pair PNOs bar_Q^(ij), bar_Q^(ik), bar_Q^(jk).
 * Q_tno columns are S_AO-orthonormal and are eigenvectors of the AO Fock
 * matrix restricted to the union span (semi-canonical TNOs).
 */
struct TNOData {
    int n_tno = 0;
    /// TNOs in AO basis: (nao × n_tno), row-major.
    /// Q_tno^T · S_AO · Q_tno = I_n_tno.
    std::vector<real_t> Q_tno;
    /// Semi-canonical orbital energies: (n_tno).
    /// F_AO acts on Q_tno columns as F_AO · Q_tno = S_AO · Q_tno · diag(eps_tno).
    std::vector<real_t> eps_tno;
    /// Number of redundant directions removed in the union ortho step.
    int n_dropped_overlap = 0;
};

/**
 * @brief Build TNO bases on demand from a converged DLPNO-LMP2 pair state.
 *
 * The builder caches read-only references to AO-basis matrices needed by
 * each per-triple build. It does NOT cache TNO bases themselves; each call
 * to build_for_triple() runs the full union → orthogonalisation → diagonal
 * pipeline. Cost per triple is O(m³) with m ≈ n_pno_ij + n_pno_ik + n_pno_jk
 * (typically 10-30), small compared to the (T) energy contraction itself.
 *
 * Phase 3.2.0 of DLPNO-CCSD(T): see project_dlpno_ccsd_t.md for context.
 */
class TNOBuilder {
public:
    /**
     * @param pairs       per-pair PNO state (uses bar_Q only).
     * @param F_AO        AO-basis Fock matrix (nao × nao, row-major).
     * @param S_AO        AO-basis overlap (nao × nao, row-major).
     * @param nao         number of AO basis functions.
     * @param tol_lin_dep eigenvalue threshold for the union overlap to drop
     *                    redundant linear-dependent directions (default 1e-7).
     */
    TNOBuilder(const std::vector<PairData>& pairs,
               const real_t* F_AO,
               const real_t* S_AO,
               int nao,
               real_t tol_lin_dep = 1e-7);

    /**
     * @brief Build the semi-canonical TNO basis for triple (i, j, k).
     * @param idx_ij,idx_ik,idx_jk  pair indices via pair_lookup.
     *
     * Returns an empty TNOData (n_tno = 0) if any pair has zero PNOs (caller
     * is expected to skip such triples earlier; this is a safety net).
     */
    TNOData build_for_triple(int idx_ij, int idx_ik, int idx_jk) const;

private:
    const std::vector<PairData>& pairs_;
    const real_t* F_AO_;
    const real_t* S_AO_;
    int           nao_;
    real_t        tol_lin_dep_;
};

/**
 * @brief Phase 3.2.1 — project a pair-PNO amplitude into TNO basis.
 *
 * The converged DLPNO-CCSD T2 amplitude for pair (p) lives in its W basis
 * (PNO-canonical, n_pno × n_pno). To use it inside a (T) contraction over
 * the union basis Q_tno, transform via the rectangular rotation
 *   R = Q_tno^T · S_AO · bar_Q^(p)         (n_tno × n_pno)
 * giving
 *   T_tno = R · Y · R^T                    (n_tno × n_tno).
 *
 * R^T · R = I_{n_pno} when the pair's PNO span lies entirely inside the
 * TNO span (which holds by construction for any of the three pairs of the
 * triple this TNO was built for), so trace(T_tno) = trace(Y) and the
 * round-trip R^T · T_tno · R recovers Y exactly.
 *
 * @param tno    TNOData built for the host triple.
 * @param pair   pair contributing the amplitude (one of (i,j), (i,k), (j,k)).
 * @param S_AO   AO overlap matrix (nao × nao, row-major).
 * @param nao    number of AO basis functions.
 * @return       T (n_tno × n_tno, row-major). Empty if either basis is empty.
 */
std::vector<real_t> project_pno_to_tno(const TNOData& tno,
                                       const PairData& pair,
                                       const real_t* S_AO,
                                       int nao);

/**
 * @brief Phase 3.2.1 — project all three pair T2 amplitudes for a triple.
 *
 * Convenience wrapper around project_pno_to_tno() that returns the three
 * T-matrices in the TNO basis as a struct.
 */
struct T2InTNO {
    std::vector<real_t> T_ij;   ///< n_tno × n_tno (from pair (i, j)).
    std::vector<real_t> T_ik;
    std::vector<real_t> T_jk;
};

T2InTNO project_triple_t2_to_tno(const TNOData& tno,
                                 const PairData& p_ij,
                                 const PairData& p_ik,
                                 const PairData& p_jk,
                                 const real_t* S_AO,
                                 int nao);

/**
 * @brief Phase 3.2.3a — project T2 for an arbitrary LMO pair (lmo_p, lmo_q)
 *        into the TNO basis with caller-specified orientation.
 *
 * The PairData/PairSetup structures store amplitudes only once per unordered
 * pair, with PairSetup.i ≤ PairSetup.j. When the caller asks for
 *   t_{lmo_p, lmo_q}^{ab}
 * we must check whether the canonical storage is (p, q) or (q, p) and
 * transpose the projected matrix in the latter case (using
 * t_{p,q}^{ab} = t_{q,p}^{ba}).
 *
 * Returns an (n_tno × n_tno) matrix in row-major layout. Empty if the pair
 * has zero PNOs.
 */
std::vector<real_t> project_pair_t2_oriented_to_tno(
    const TNOData& tno,
    const std::vector<PairData>& pairs,
    const std::vector<PairSetup>& setups,
    const std::vector<int>& pair_lookup,
    int lmo_p, int lmo_q,
    const real_t* S_AO,
    int nao, int nocc);

// ---------------------------------------------------------------------------
//  Phase 3.2.2 — ERI in TNO basis (per triple).
// ---------------------------------------------------------------------------

/**
 * @brief Two-body integral tensors for one triple (i, j, k) in TNO basis.
 *
 * The (T) energy contraction needs two tensors:
 *
 *   K[i_loc, a, d, c]  =  (i_LMO  a_TNO | d_TNO  c_TNO)
 *     i_loc ∈ {0, 1, 2} maps to {triple_lmos[i_loc]} ∈ {i, j, k}.
 *     Layout: 3 × n_tno × n_tno × n_tno, row-major flat.
 *
 *   L[l, c, m_loc, n_loc] =  (l_LMO  c_TNO | m_LMO  n_LMO)
 *     l ∈ [0, nocc), m_loc, n_loc ∈ {0, 1, 2} map to triple LMOs.
 *     Layout: nocc × n_tno × 9 (m_loc * 3 + n_loc).
 *
 * Both tensors are assembled via RI from precomputed 3-index B blocks.
 */
struct ERIInTNO {
    int n_tno = 0;
    int nocc  = 0;
    std::vector<real_t> K_iadc;   ///< 3 × n_tno × n_tno × n_tno
    std::vector<real_t> L_lcmn;   ///< nocc × n_tno × 9
};

/**
 * @brief Build per-triple ERI tensors in the TNO basis from precomputed RI
 *        3-index integrals.
 *
 * Inputs (caller-provided, shared across all triples):
 * @param tno          TNO basis for this triple (Q_tno, n_tno).
 * @param triple_lmos  3-element array mapping i_loc → LMO index.
 * @param B_lmo_ao     B_l_ν^Q  =  Σ_μ C_LMO[μ, l] · B_AO[μ ν | Q]
 *                     shape (nocc × nao × naux), layout row-major
 *                     B_lmo_ao[(l * nao + ν) * naux + Q].
 * @param B_ao_ao      B_AO[μ ν | Q] in row-major slab layout
 *                     B_ao_ao[(μ * nao + ν) * naux + Q].
 * @param B_lmo_lmo    Σ_μν C_LMO[μ,l] C_LMO[ν,m] B_AO[μ ν | Q]
 *                     shape (nocc × nocc × naux), layout
 *                     B_lmo_lmo[(l * nocc + m) * naux + Q].
 * @param nao,nocc,naux  dimensions.
 * @return       ERIInTNO with K_iadc and L_lcmn populated. Empty when
 *               n_tno == 0.
 */
ERIInTNO build_eri_in_tno(const TNOData& tno,
                          const int triple_lmos[3],
                          const real_t* B_lmo_ao,
                          const real_t* B_ao_ao,
                          const real_t* B_lmo_lmo,
                          int nao,
                          int nocc,
                          int naux,
                          bool build_L = true,
                          const real_t* B_TTQ_precomputed = nullptr);
// build_L = false:        skip the L_lcmn build (unused by Phase 3.2.6 path).
// B_TTQ_precomputed != nullptr:  caller has built B_TTQ on GPU (see
//   EriBuildGpu::build_b_ttq) — Step 2 is skipped and B_TTQ is reused
//   directly. Layout: (n_tno × n_tno × naux) row-major.

// ---------------------------------------------------------------------------
//  Phase 3.2.3 — T3 amplitude precursor + closed-shell (T) energy.
// ---------------------------------------------------------------------------

/**
 * @brief Phase 3.2.3b — build the per-triple T3-precursor tensor W[a,b,c].
 *
 * For triple (i, j, k) (LMO indices given via triple_lmos), W has the
 * symmetric structure required by the closed-shell (T) energy expression:
 *
 *   W[a,b,c] = Σ_d (6 particle perms of t·K)
 *            − Σ_l (6 hole     perms of t·L)
 *
 * The 6 hole permutations contract over an arbitrary occupied l, so we
 * need the extended T2 projection for pairs (i, l), (j, l), (k, l) for
 * every l ∈ [0, nocc). T_il_ext[l] = oriented projection of t_{i, l}^{ab}.
 *
 * Returns flat (n_tno × n_tno × n_tno) tensor. Empty if n_tno == 0.
 */
std::vector<real_t> build_W_tensor_for_triple(
    const TNOData& tno,
    const ERIInTNO& eri,
    const T2InTNO& t2_triple,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc);

/**
 * @brief Phase 3.2.3c — closed-shell (T) energy contribution for one triple.
 *
 * Standard restricted CCSD(T) energy formula iterated with i ≤ j ≤ k:
 *
 *   E_ijk = factor(i,j,k) · Σ_{a,b,c} W[a,b,c] · ω[a,b,c] / D[a,b,c]
 *
 * where
 *   ω[a,b,c] = 4·W[a,b,c] − 2·W[a,c,b] − 2·W[b,a,c] + W[b,c,a] + W[c,a,b]
 *   D[a,b,c] = ε_i + ε_j + ε_k − eps_tno[a] − eps_tno[b] − eps_tno[c]
 *   factor   = 1            (i < j < k)
 *            = 1/2          (i = j < k  or  i < j = k)
 *            = 1/6          (i = j = k)
 *
 * Phase 2.2 of DLPNO-CCSD has T1 ≈ 0 by Brillouin condition, so the
 * V-tensor (with t1 dressing) reduces to W. The implementation here uses
 * V = W exclusively; t1 dressing is deferred to a future sub-phase.
 *
 * @param eps_i,eps_j,eps_k  LMO Fock diagonals = F_LMO[i,i] etc.
 */
real_t compute_triple_t_energy(int i, int j, int k,
                               real_t eps_i, real_t eps_j, real_t eps_k,
                               const TNOData& tno,
                               const std::vector<real_t>& W);

// ---------------------------------------------------------------------------
//  Phase 3.2.6 — PySCF-equivalent closed-shell (T) energy formula.
//
//  Replaces the 12-term-W + ω(W) approach (which gave basis-dependent
//  over-counts) with the canonical PySCF ccsd_t_slow formulation, reorganised
//  to outer (i ≤ j ≤ k) loop. Verified bit-exact against PySCF on H2O sto-3g
//  and cc-pVDZ (ratio = 1.0).
//
//  Per (i ≤ j ≤ k), we build six "simple" W tensors (one per S_3 permutation
//  σ of (i,j,k)) — each is a 1-particle + 1-hole contribution:
//
//    w_σ(a,b,c) = Σ_d (lmo_{σ[0]}, a | b, d) · t_{lmo_{σ[2]} lmo_{σ[1]}}^{cd}
//               − Σ_l (lmo_{σ[0]}, a | l, lmo_{σ[1]}) · t_{l, lmo_{σ[2]}}^{bc}
//
//  The 6 z-tensors z_σ = r3(w_σ) / d3_ijk (r3 acts on (a,b,c)). Then 36
//  einsum-style contractions between the 6 w's and 6 z's, summed and × 2.
//
//  d3_ijk degeneracy factor on the outer occupied triple:
//    distinct (i<j<k):       1
//    one pair equal:         2
//    all equal (i=j=k):      6
// ---------------------------------------------------------------------------

/**
 * @brief Phase 3.2.6 hole-side integrals for one triple in TNO basis.
 *
 * For each ordered pair (slot_p, slot_q) with slot_p, slot_q ∈ {0, 1, 2}
 * (mapping to {i, j, k} of the outer triple) and slot_p ≠ slot_q, we need:
 *
 *   M_{p,q}[l, a] = (lmo_{slot_p}, a | l, lmo_{slot_q})
 *                 = Σ_Q B_lTQ[lmo_p, a, Q] · B_lmo_lmo[l, lmo_q, Q]
 *
 * Six tensors total (3 × 2 ordered pairs). Layout: nocc × n_tno per pair,
 * stored flat in a 9-slot array indexed by (slot_p * 3 + slot_q). Diagonal
 * entries (p == q) are unused.
 */
struct HoleMTensors {
    int n_tno = 0;
    int nocc  = 0;
    /// 9 × (nocc × n_tno), flat; only off-diagonal slots populated.
    std::array<std::vector<real_t>, 9> M;
};

HoleMTensors build_hole_m_tensors(const TNOData& tno,
                                  const int triple_lmos[3],
                                  const real_t* B_lmo_ao,
                                  const real_t* B_lmo_lmo,
                                  int nao,
                                  int nocc,
                                  int naux);

/**
 * @brief Phase 3.2.6 — compute the closed-shell (T) energy contribution
 *        for one triple via the PySCF-equivalent 6-W formula.
 *
 * Replaces compute_triple_t_energy (which used 12-term W + ω with strict
 * i<j<k). This function uses the i ≤ j ≤ k loop with d3_ijk degeneracy.
 *
 * Inputs:
 * @param i,j,k                LMO indices, with i ≤ j ≤ k.
 * @param eps_i,eps_j,eps_k    LMO Fock diagonals.
 * @param tno                  TNO basis (eps_tno used).
 * @param K_iadc               (3 × n_tno³) — particle integrals
 *                              K[loc, a, b, d] = (lmo_loc, a | b, d).
 * @param M                    Six hole-integral M tensors (nocc × n_tno each)
 *                              M[loc_p * 3 + loc_q][l, a] = (lmo_p, a | l, lmo_q).
 * @param T_part_oriented      6 × (n_tno × n_tno) ordered-pair t2 amplitudes
 *                              T_part[loc_p * 3 + loc_q][c, d] = t_{lmo_p, lmo_q}^{cd}.
 *                              Only off-diagonal (p ≠ q) slots are used.
 * @param T_il_ext,T_jl_ext,T_kl_ext   per-l hole-side t2 amplitudes
 *                              T_il_ext[l][b, c] = t_{i, l}^{bc} (in TNO basis).
 * @param nocc                 number of occupied LMOs.
 * @return    contribution of this triple to the (T) energy (already including
 *            the 1/d3_ijk degeneracy factor and the closed-shell × 2 multiplier).
 */
real_t compute_triple_t_energy_pyscf(
    int i, int j, int k,
    real_t eps_i, real_t eps_j, real_t eps_k,
    const TNOData& tno,
    const real_t* K_iadc,
    const std::array<std::vector<real_t>, 9>& M,
    const std::array<std::vector<real_t>, 9>& T_part_oriented,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc);

} // namespace gansu
