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
#include "dlpno_params.hpp"
#include "dlpno_pair_data.hpp"

namespace gansu {

class RHF;
class ERI;

/**
 * @brief Bundle returned by `solve_dlpno_lmp2()`.
 *
 * Carries the per-pair PNO state at LMP2 convergence so that downstream
 * solvers (DLPNO-CCSD, properties) can reuse all of the Phase-1 setup
 * (LMO, PAO, BP domain, PNO, bar_Q, L, Y_converged) without rebuilding.
 */
struct DLPNOLMP2Result {
    std::vector<PairSetup> setups;        ///< per-pair invariants
    std::vector<PairData>  pairs;         ///< per-pair PNO state + converged Y
    std::vector<int>       pair_lookup;   ///< pair_lookup[i*nocc+j] = pair index (i,j ↔ j,i)
    std::vector<real_t>    F_LMO;         ///< [nocc × nocc] LMO Fock matrix
    std::vector<real_t>    C_LMO;         ///< [nao  × nocc] LMO coefficients in AO basis (used by CCSD T1 residual)
    std::vector<real_t>    U_loc;         ///< [nocc × nocc] localization rotation, C_LMO = C_occ · U_loc, element (I_can,i_lmo) at U_loc[I*nocc+i] (bt-PNO-STEOM P5a back-transform). Identity when localizer == "none".
    /// Per-pair PNO two-electron integral blocks (W_ovov/W_ovvo/W_oooo/T_pair/…)
    /// required by the native per-pair DLPNO-IP/EA-EOM σ (bt-PNO-STEOM stage B
    /// (a), see STEOM.md §21.11). Built once during DLPNO-CCSD by
    /// precompute_phase24_integrals() and normally discarded; copied here only
    /// on the DLPNO-bt-STEOM collect path (rhf.collect_dlpno_bt()), so plain
    /// DLPNO-CCSD / non-DLPNO runs leave it default-empty at zero cost.
    Phase24Integrals       phase24;
    real_t E_pao_total = 0.0;             ///< pre-PNO MP2 (diagnostic)
    real_t E_pno_total = 0.0;             ///< post-PNO converged LMP2
    int    nao = 0;
    int    nocc = 0;
    bool   needs_iter = false;
    int    sc_pno_rounds = 0;
};

/**
 * @brief Optional embedding-cluster orbital space for solve_dlpno_lmp2
 *        (DMET×DLPNO Phase 1a — rectangular C).
 *
 * The cluster MOs are genuine LCAO over the full-molecule AO basis
 * (rectangular C, nao × nmo with nmo < nao); the AO overlap S and the
 * atom→AO ranges still come from the RHF. Two reformulated inputs make the
 * square-C limit EXACT (they reduce algebraically to the molecular
 * quantities when C is the complete square MO set):
 *   - Fock: supplied in AO-covariant effective form
 *       F_eff = S·C·diag(eps)·C^T·S      (== F when C C^T = S⁻¹),
 *   - PAO:  built from the cluster-VIRTUAL projector image of the AOs
 *       C̃^PAO = (C_vir·C_vir^T)·S       (== I − D_occ·S when D_occ+D_vir=S⁻¹),
 *     restricting the PAO span to the cluster virtual subspace
 *     (rank nmo − nocc_full); the per-domain Löwdin (t_cut_do) removes the
 *     redundancy as usual.
 * Occupied localization / domains are unchanged (cluster occ MOs are real-AO
 * LCAO, so per-atom Mulliken populations stay well defined — validated by the
 * GANSU_DMET_STEOM_LOC_TEST probe, rectangular PM converged).
 */
struct DLPNOClusterSpace {
    const real_t* h_C     = nullptr; ///< [nao × nmo] row-major cluster MO coefficients (host)
    const real_t* h_eps   = nullptr; ///< [nmo] cluster orbital energies (host)
    const real_t* h_F_eff = nullptr; ///< [nao × nao] S·C·diag(eps)·C^T·S (host, caller-built)
    int nmo     = 0;                 ///< cluster MO count (columns of C, = n_emb)
    int nocc    = 0;                 ///< cluster ACTIVE occupied count (frozen excluded)
    int nfrozen = 0;                 ///< cluster frozen-core count (frozen = first columns)
};

/// Run all of Phase-1 (PM + PAO + BP-domain + PNO + iterate_lmp2 + SC-PNO)
/// and return the converged pair state. Used by `DLPNOMP2::compute_energy`
/// and by the Phase-2 DLPNO-CCSD driver as its initial pair-data setup.
/// `cluster` non-null runs the same machinery on an embedding-cluster orbital
/// space (rectangular C) — nullptr is the legacy full-molecule path
/// (byte-identical).
DLPNOLMP2Result solve_dlpno_lmp2(
    RHF& rhf,
    const ERI& eri,
    const DLPNOParams& params,
    const DLPNOClusterSpace* cluster = nullptr);

/**
 * @brief DLPNO-MP2 driver (Phase 1).
 *
 * Implements MP2 in a domain-based local PNO framework:
 *   1. Pipek-Mezey localisation of the occupied space.
 *   2. PAO construction (C̃ = I − D_occ S) and per-LMO Boughton-Pulay
 *      domains (Mulliken cumulative).
 *   3. For each pair (i, j) with i ≤ j:
 *        - pair domain = LMO i domain ∪ LMO j domain
 *        - per-pair PAO Löwdin orthogonalisation + redundancy removal
 *        - semi-canonical PAOs by diagonalising F restricted to the
 *          orthogonalised PAO subspace
 *        - first-order MP2 amplitudes T_{ij}^{ab} (semi-canonical)
 *        - PNO construction + occupation-based truncation
 *        - re-evaluate amplitudes / energy in the truncated PNO basis.
 *   4. Total correlation energy
 *        E = Σ_i E_{ii} + 2 Σ_{i<j} E_{ij}.
 *
 * The driver currently uses `ERI::build_mo_eri` for the per-pair
 * integral block, which is correct but quadratic in pair count. A direct
 * RI 3-index half-transform is planned as an optimisation step.
 */
class DLPNOMP2 {
public:
    DLPNOMP2(RHF& rhf, const ERI& eri, DLPNOParams params);

    /// Run the full DLPNO-MP2 calculation. Returns the correlation energy
    /// (after PNO truncation).
    real_t compute_energy();

    /// Untruncated energy (full PAO subspace per pair). Useful for
    /// validation: with full domains it matches RI-MP2 exactly.
    real_t energy_untruncated() const { return E_pao_; }

private:
    RHF& rhf_;
    const ERI& eri_;
    DLPNOParams params_;

    int nao_  = 0;
    int nocc_ = 0;
    int natoms_ = 0;

    real_t E_pao_ = 0.0;  // Pre-truncation energy (PAO/semi-canonical basis)
    real_t E_pno_ = 0.0;  // Post-truncation energy
};

} // namespace gansu
