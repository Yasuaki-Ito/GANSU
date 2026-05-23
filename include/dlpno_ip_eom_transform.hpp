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
 * @file dlpno_ip_eom_transform.hpp
 * @brief Single-index R2 transforms between the DLPNO-IP-EOM packed-PNO basis
 *        and the canonical IP-EOM r2 layout (bt-PNO-STEOM stage B, Phase B1).
 *
 * The IP 2h1p amplitude r^a_{ij} has ONE virtual index (a ∈ PNO(ij)), so the
 * PNO↔canonical map is the single-sided analog of the bt_pno T2 transform:
 *
 *   packed → canonical:  R2_canon[I,J,a] = Σ_{ij} U_loc[I,i] U_loc[J,j]
 *                                          Σ_{a'} U^(ij)_{a,a'} r2_pno^{(ij)}_{a'}
 *   canonical → packed:  r2_pno^{(ij)}_{a'} = Σ_a U^(ij)_{a,a'}
 *                                             Σ_{IJ} U_loc[I,i] U_loc[J,j] R2_canon[I,J,a]
 *
 * with U^(ij) = C_vir^T · S_AO · bar_Q_ij  [nvir × n_pno] (same isometry as the
 * bt_pno back-transform). Off-diagonal pairs carry two independent orientations
 * (i,j) and (j,i) — see dlpno_ip_packing.hpp.
 *
 * Canonical r2 layout matches IPEOMCCSDOperator: R2_canon[(I*nocc+J)*nvir + a]
 * (FULL (I,J), both orderings). The packed R2 region is the Davidson vector
 * minus its leading nocc R1 entries (length pack.total_dim - pack.nocc).
 *
 * These converters are the bridge for every B1/B2 no-truncation gate: they let
 * the DLPNO-IP σ (built in packed-PNO space) be compared element-wise to the
 * validated canonical P1 σ. At no truncation (n_pno = nvir, U^(ij) square
 * orthogonal) the round-trip packed→canonical→packed is the identity.
 */

#pragma once

#include <vector>

#include "types.hpp"
#include "dlpno_mp2.hpp"        // DLPNOLMP2Result
#include "dlpno_ip_packing.hpp" // DLPNOIPPacking

namespace gansu {

/// packed-PNO R2 region → canonical R2 [nocc²·nvir].
/// @param packed_r2 length pack.total_dim - pack.nocc (the per-pair blocks).
/// @return canonical R2, layout [(I*nocc+J)*nvir + a].
std::vector<real_t> ip_packed_r2_to_canonical(
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& packed_r2);

/// canonical R2 [nocc²·nvir] → packed-PNO R2 region (inverse of the above).
/// @return packed R2 region, length pack.total_dim - pack.nocc.
std::vector<real_t> ip_canonical_r2_to_packed(
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& R2_canon);

} // namespace gansu
