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
 * @file bt_pno_backtransform.hpp
 * @brief PNO → canonical-MO back-transform of DLPNO-CCSD amplitudes
 *        (bt-PNO-STEOM Phase P5a — the shared foundation for Steps 2/3/3.5).
 *
 * Given a converged DLPNO-CCSD result (per-pair amplitudes Y in the PNO "W"
 * basis), reconstruct the canonical-MO-basis CCSD T2 / T1 in the SAME
 * chemist-notation layout that ccsd_spatial_orbital produces and the
 * canonical IP/EA/STEOM operators consume:
 *
 *   T2_can[((I*nocc + J)*nvir + a)*nvir + b]   size nocc²·nvir², row-major,
 *       with the symmetry T2_can[I,J,a,b] == T2_can[J,I,b,a].
 *   T1_can[I*nvir + a]                          size nocc·nvir.
 *
 * Math (Dutta-Neese-Izsák 2018, Step 3.5 applied to ground-state amplitudes):
 *   Step 1 (per pair):  U^(ij) = C_vir^T · S_AO · bar_Q_ij      [nvir × n_pno]
 *   Step 2 (per pair):  T2_lmo_{ij} = U^(ij) · Y^(ij) · U^(ij)^T [nvir × nvir]
 *   Step 3 (occ rot):   T2_can[I,J] = Σ_{ij} U_loc[I,i] U_loc[J,j] T2_lmo_{ij}
 *
 * Pairs are stored only for i ≤ j (Y_ji = Y_ij^T), so the off-diagonal
 * orientation is filled by the (a,b)-transpose of T2_lmo_{ij}. The PairSetup
 * pair_factor (2 for i≠j) is an ENERGY weight and is NOT applied here.
 *
 * Validation: at the no-truncation limit (full domains, t_cut_pno→0) DLPNO is
 * just a rotation of the canonical space, so the back-transformed T2 must
 * converge to the canonical CCSD T2 from ccsd_spatial_orbital.
 */

#pragma once

#include <vector>

#include "types.hpp"
#include "dlpno_mp2.hpp"   // DLPNOLMP2Result

namespace gansu {

/// Canonical-MO-basis amplitudes produced by the bt-PNO back-transform.
/// Layouts match ccsd_spatial_orbital exactly (see file header).
struct BTAmplitudes {
    std::vector<real_t> T2;   ///< [nocc²·nvir²] canonical chemist layout
    std::vector<real_t> T1;   ///< [nocc·nvir]   (zero unless include_t1)
    int nocc = 0;             ///< active occupied
    int nvir = 0;             ///< canonical virtual (= nao - num_occ_total)
};

/**
 * @brief Back-transform converged DLPNO-CCSD per-pair PNO amplitudes
 *        (res.pairs[*].Y) to canonical-MO-basis T2 (and optionally T1).
 *
 * @param res       Converged DLPNO-CCSD result. Y must already hold the CCSD
 *                  (not just LMP2) amplitudes in the PNO W basis.
 * @param U_loc     [nocc × nocc] localization rotation, C_LMO = C_occ · U_loc,
 *                  element (I_can, i_lmo) at U_loc[I*nocc + i]. If empty or of
 *                  the wrong size it is treated as the identity (= no
 *                  localization, the P5a.0 fast path).
 * @param C_vir     [nao × nvir] canonical virtual MO coefficients in the AO
 *                  basis (columns = full C columns [num_occ_total, nao)).
 * @param h_S       [nao × nao] AO overlap matrix (row-major, host).
 * @param nao       Basis-function count.
 * @param nvir      Canonical virtual count.
 * @param T1_pao    Per-LMO PAO-basis T1 (T1_pao[i] has size setups[(i,i)].n_pao).
 *                  Only consulted when include_t1 is true; may be empty.
 * @param include_t1  When false (P5a.0 gate) T1 is left zero.
 * @return BTAmplitudes with canonical T2 (and T1).
 */
BTAmplitudes bt_pno_to_canonical(
    const DLPNOLMP2Result& res,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<std::vector<real_t>>& T1_pao,
    bool include_t1);

} // namespace gansu
