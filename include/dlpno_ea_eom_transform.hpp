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
 * @file dlpno_ea_eom_transform.hpp
 * @brief Two-virtual R2 transforms between the DLPNO-EA-EOM packed-PNO basis
 *        and the canonical EA-EOM r2 layout (bt-PNO-STEOM stage B).
 *
 * EA 2p1h r^{ab}_i has TWO virtual indices a,b ∈ PNO(i,i) and ONE occupied i:
 *
 *   packed → canonical:  R2_canon[I,a,b] = Σ_i U_loc[I,i]
 *                            Σ_{a'b'} U^(ii)_{a,a'} U^(ii)_{b,b'} r2_pno^{(i)}_{a'b'}
 *   canonical → packed:  r2_pno^{(i)}_{a'b'} = Σ_{ab} U^(ii)_{a,a'} U^(ii)_{b,b'}
 *                            Σ_I U_loc[I,i] R2_canon[I,a,b]
 *
 * with U^(ii) = C_vir^T · S_AO · bar_Q_(ii)  [nvir × n_pno(ii)] (same isometry
 * as bt_pno). The per-i virtual map is the two-sided U·Y·U^T (as in the T2
 * back-transform); the occupied map is the single-index U_loc rotation (EA has
 * one occupied index, unlike IP/T2 which have two).
 *
 * Canonical r2 layout matches EAEOMCCSDOperator: R2_canon[(I*nvir+a)*nvir + b].
 * The packed R2 region is the Davidson vector minus its leading nvir R1
 * entries (length pack.total_dim - pack.nvir).
 *
 * At no truncation (n_pno = nvir, U^(ii) square orthogonal) the round-trip
 * packed→canonical→packed is the identity.
 */

#pragma once

#include <vector>

#include "types.hpp"
#include "dlpno_mp2.hpp"        // DLPNOLMP2Result
#include "dlpno_ea_packing.hpp" // DLPNOEAPacking

namespace gansu {

/// packed-PNO R2 region → canonical R2 [nocc·nvir²].
/// @param packed_r2 length pack.total_dim - pack.nvir.
/// @return canonical R2, layout [(I*nvir+a)*nvir + b].
std::vector<real_t> ea_packed_r2_to_canonical(
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao,
    const std::vector<real_t>& packed_r2);

/// canonical R2 [nocc·nvir²] → packed-PNO R2 region (inverse of the above).
std::vector<real_t> ea_canonical_r2_to_packed(
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao,
    const std::vector<real_t>& R2_canon);

} // namespace gansu
