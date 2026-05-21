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

/**
 * @file ea_eom_result.hpp
 * @brief Per-active-virtual EA-EOM-CCSD root storage for the bt-PNO-STEOM-CCSD
 *        pipeline (Phase P2 → P3 hand-off).
 *
 * After EA-EOM-CCSD is solved, one Davidson root is mapped to each active
 * virtual NTO ã (from `CISNTOResult.U_vir`, first `n_act_vir` columns) using a
 * FollowCIS overlap + %singles filter (STEOM.md §14.2 mirrored for EA). The
 * EA eigenvalue ω, the 1p amplitudes R1 [nvir], the 2p1h amplitudes R2
 * [nocc · nvir²] (full layout), and diagnostic tags are stored here so Phase
 * P3 STEOM can assemble Ŝ^EA without re-running Davidson.
 */

#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace gansu {

struct EAEOMResult {
    int nocc_active = 0;
    int nvir        = 0;
    int num_frozen  = 0;
    int n_active    = 0;   ///< Number of populated active roots (= cis_nto_result.n_act_vir in active mode)

    struct PerRoot {
        real_t omega = 0.0;                 ///< EA value in Ha (positive)
        std::vector<real_t> R1;             ///< 1p amplitudes [nvir]
        std::vector<real_t> R2;             ///< 2p1h amplitudes [nocc · nvir²] (full layout, b<a anti-symmetric)
        real_t percent_singles  = 0.0;      ///< ‖R1‖² / (‖R1‖² + ‖R2‖²), in [0,1]
        real_t followcis_overlap = 0.0;     ///< |<U_vir^(ã) | R1>|² in [0,1]
        int    canonical_vir_label = -1;    ///< Active NTO index ã (0..n_active-1) this root maps to (-1 = unassigned)
    };

    /// Roots assigned to active virtual NTOs, indexed 0..n_active-1.
    std::vector<PerRoot> per_active;

    /// Roots Davidson found but that did not survive %singles or FollowCIS
    /// (kept for diagnostics, NOT consumed by STEOM second transform).
    std::vector<PerRoot> auxiliary;

    /// Human-readable summary table (printed in final summary + retained for memory dump).
    std::string report;
};

// Forward declaration — CISNTOResult lives in cis_nto_active_space.hpp.
struct CISNTOResult;

/**
 * @brief Sub-phase 2.9 + 2.10 + 2.13 — pure CPU routing decision for EA-EOM-CCSD.
 *
 * Mirrors `select_active_ip_roots` (see `ip_eom_result.hpp`) but reads the
 * virtual NTO basis (CISNTOResult.U_vir, first n_act_vir columns) instead of
 * the occupied side. Algorithm is identical (greedy bipartite match by
 * descending overlap, subject to a %singles floor).
 *
 * The function is side-effect-free apart from the returned struct; it does
 * NOT mutate `all_roots` (the driver writes follow-up tags after the call).
 *
 * NOTE — sub-phase 2.0+2.1 ships the declaration only; the implementation
 * lands in sub-phase 2.9 (file `src/ea_eom_routing.cpp`).
 */
struct EAEOMRoutingDecision {
    std::vector<int>    assigned_root_for_a;  ///< size n_act_vir, -1 = unassigned
    std::vector<real_t> overlap_for_a;        ///< size n_act_vir, FollowCIS overlap of the assigned root (0 if -1)
    std::vector<bool>   root_taken;           ///< size n_roots, true if assigned to some ã
};

EAEOMRoutingDecision select_active_ea_roots(
    const CISNTOResult& cis_nto,
    const std::vector<EAEOMResult::PerRoot>& all_roots,
    int nvir,
    real_t ea_thresh);

} // namespace gansu
