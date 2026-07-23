/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <vector>
#include "types.hpp"

namespace gansu {

class RHF;
class ERI;

/// Result of CIS-guided automatic fragment extraction for DMET-STEOM.
///
/// The excitation-driven, real-space atomic fragment is derived from the
/// full-system state-averaged CIS-NTO per-atom weights (hole + particle). This
/// is the novel selection step: existing DMET auto-fragmentation (e.g. GAF) cuts
/// from ground-state connectivity, and excited-state DMET variants pick the
/// fragment manually or from a CASSCF active space — none use NTO per-atom
/// weights to choose the real-space fragment.
struct DMETAutoFragmentResult {
    std::vector<int> atoms;        ///< Selected chromophore atom indices (sorted asc)
    int    n_cis_used   = 0;       ///< CIS state count used for the extraction. The
                                   ///< state-averaged CIS-NTO for this count is left
                                   ///< cached on rhf.get_cis_nto_result() for reuse
                                   ///< by the downstream NTO-bath augmentation.
    double coverage     = 0.0;     ///< Achieved cumulative per-atom NTO coverage ∈ [0,1]
    bool   budget_hit   = false;   ///< Selection capped by the cluster orbital budget
    bool   delocalized  = false;   ///< Above-floor atoms could not reach the coverage
                                   ///< target → excitation is delocalized/diffuse
    int    n_components = 1;       ///< Connected-component count of the selected atoms
};

/// Extract the chromophore fragment from full-system CIS-NTO per-atom weights.
///
/// Runs `eri.compute_cis_nto(n_cis)` internally (result cached on `rhf`), scores
/// every atom by its occupation-weighted hole+particle NTO Löwdin population,
/// then greedily selects atoms above the per-atom floor until the coverage target
/// is met or the cluster orbital budget is hit. Prints a self-contained
/// diagnostic (per-atom table, selected set, coverage, sensitivity/budget/
/// delocalization/multi-region warnings). Never throws: on an empty or failed
/// CIS-NTO it returns all atoms (whole molecule) so the caller degenerates to
/// plain STEOM.
///
/// @param rhf        SCF object (source of C/S, geometry, CIS-NTO cache, and the
///                   dmet_steom_auto_* parameters).
/// @param eri        Integral engine (drives compute_cis_nto).
/// @param n_states   Requested number of excited states (sets the auto n_cis).
/// @param num_atoms  Atom count.
/// @param nao        Number of AO basis functions.
/// @param nocc       Number of occupied MOs (full system).
/// Default cluster orbital budget (est. n_emb ceiling) when dmet_steom_auto_budget
/// is 0. dlpno cluster solver → 700, canonical → 460 (measured ceilings). Honors
/// both the --dmet_cluster_solver parameter and the GANSU_DMET_STEOM_DLPNO=2 env
/// override (the production scripts select the dlpno solver via the env).
int dmet_steom_default_budget(const RHF& rhf);

DMETAutoFragmentResult dmet_steom_auto_extract_fragment(
    RHF& rhf, ERI& eri, int n_states, int num_atoms, int nao, int nocc);

/// Bath-sufficiency gauge for a given embedding, with per-atom attribution of the
/// uncaptured excitation character (drives Phase B fragment expansion).
///
/// Mirrors the §4.3 gauge in DMET::compute_steom (occupation-weighted uncaptured =
/// 1 − cluster capture of each full-system active NTO, hole/particle weighted
/// separately, verdict from the worse side) so the loop's stop/continue decision
/// uses the same ruler as the diagnostic printed in the solve. Additionally
/// projects each active NTO's uncaptured residual onto the environment AOs and
/// attributes |residual|² to atoms → the highest-scoring environment atom is the
/// expansion candidate. Consumes the CIS-NTO already cached on `rhf`.
struct DMETBathGaugeResult {
    double wunc     = 0.0;                  ///< occupation-weighted uncaptured (η-aligned, worse side)
    double wunc_vir = 0.0;
    double wunc_occ = 0.0;
    const char* verdict = "SUFFICIENT";     ///< "SUFFICIENT" (<0.02) / "MARGINAL" (<0.10) / "INSUFFICIENT"
    std::vector<double> atom_uncaptured;    ///< [num_atoms] environment residual attribution
    // Virtual-space sufficiency (extended particle coverage). The gauge above
    // weights only the leading ACTIVE particle NTOs; this extends the scan down
    // to occupation 1e-4 so the correlating particle TAIL is included. A tail
    // uncaptured markedly above the active one flags a cluster that spans the
    // leading excitation but truncates the virtual (particle) space it needs —
    // a genuine virtual-space deficiency. (It does NOT catch the mean-field
    // embedding error that blueshifts fully-delocalized excitations even when
    // both active and tail coverage are high — that is outside DMET's domain.)
    double wunc_vir_ext = 0.0;              ///< occupation-weighted uncaptured over the extended virtual-NTO tail
    int    n_vir_ext    = 0;                ///< number of virtual NTOs in the extended set (occupation > 1e-4)
};

DMETBathGaugeResult dmet_steom_bath_gauge(
    RHF& rhf, const real_t* S_half, const real_t* h_C,
    int nao, int nocc, int num_atoms,
    const real_t* C_emb, int n_emb, const std::vector<char>& is_frag_ao);

} // namespace gansu
