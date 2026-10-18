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
 * @file ip_eom_result.hpp
 * @brief Per-active-occupied IP-EOM-CCSD root storage for the
 *        bt-PNO-STEOM-CCSD pipeline (Phase P1 → P3 hand-off).
 *
 * After IP-EOM-CCSD is solved, one Davidson root is mapped to each active
 * occupied NTO m̃ (from `CISNTOResult`) using FollowCIS overlap + %singles
 * filter (STEOM.md §14.2). The IP eigenvalue ω, the 1h amplitudes R1, the
 * 2h1p amplitudes R2, and diagnostic tags are stored in this struct so
 * Phase P3 STEOM can assemble Ŝ^IP without re-running Davidson.
 */

#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace gansu {

struct IPEOMResult {
    int nocc_active = 0;   ///< Active-occupied dimension (full_occ - num_frozen)
    int nvir        = 0;
    int num_frozen  = 0;
    int n_active    = 0;   ///< Number of populated active roots (= cis_nto_result.n_act_occ in active mode)

    struct PerRoot {
        real_t omega = 0.0;                ///< IP value in Ha (positive)
        std::vector<real_t> R1;            ///< 1h amplitudes [nocc_active]
        std::vector<real_t> R2;            ///< 2h1p amplitudes [nocc_active * nocc_active * nvir] (full layout, j<i anti-symmetric)
        real_t percent_singles  = 0.0;     ///< ‖R1‖² / (‖R1‖² + ‖R2‖²), in [0,1]
        real_t followcis_overlap = 0.0;    ///< |<U_occ^(m̃) | R1>|² in [0,1]
        int    canonical_occ_label = -1;   ///< Active NTO index m̃ (0..n_active-1) this root maps to (-1 = unassigned)
    };

    /// Roots assigned to active NTOs, indexed 0..n_active-1.
    std::vector<PerRoot> per_active;

    /// Roots Davidson found but that did not survive %singles or FollowCIS
    /// (kept for diagnostics, NOT consumed by STEOM second transform).
    std::vector<PerRoot> auxiliary;

    /// Human-readable summary table (printed in final summary + retained for memory dump).
    std::string report;
};

// Forward declaration — CISNTOResult lives in cis_nto_active_space.hpp; we
// only need the type to take by const&, so we don't include the header here.
struct CISNTOResult;

/**
 * @brief Sub-phase 1.9 + 1.10 + 1.13 — pure CPU routing decision.
 *
 * Given a batch of Davidson IP-EOM-CCSD candidate roots (each carrying its
 * R1 / percent_singles already populated) and a CIS-NTO active-occupied
 * basis, decide which Davidson root, if any, maps to each active NTO m̃.
 *
 * Algorithm (matches STEOM.md §14.2):
 *   1. Compute overlap[k][m̃] = ( Σ_i U_occ[i, m̃] · R1_k[i] )² / ‖R1_k‖².
 *   2. Greedy bipartite match: repeatedly pick the unassigned (m̃, k) pair
 *      with the largest overlap that satisfies
 *          all_roots[k].percent_singles ≥ ip_thresh,
 *      assign root k to NTO m̃, mark both as taken, repeat until either pool
 *      is exhausted or no surviving candidate exists for any remaining m̃.
 *   3. NTOs with no surviving candidate keep `assigned_root_for_m[m̃] = -1`.
 *
 * The function is side-effect-free apart from the returned struct; it does
 * NOT mutate `all_roots` (the driver writes follow-up tags after the call).
 */
struct IPEOMRoutingDecision {
    std::vector<int>    assigned_root_for_m;  ///< size n_act_occ, -1 = unassigned
    std::vector<real_t> overlap_for_m;        ///< size n_act_occ, FollowCIS overlap of the assigned root (0 if -1)
    std::vector<bool>   root_taken;           ///< size n_roots, true if assigned to some m̃

    // ---- overlap-floor diagnostics (see steom_followcis_overlap_floor) ----
    /// true for each m̃ left unassigned because every remaining candidate's
    /// overlap was below the floor — as opposed to the %singles filter.
    std::vector<char>   unassigned_by_floor;
    real_t overlap_floor      = 0.0;  ///< floor actually applied
    real_t best_below_floor   = 0.0;  ///< largest overlap the floor rejected (0 if none)
    int    n_below_floor      = 0;    ///< how many m̃ that left unassigned
};

/**
 * @brief Minimum FollowCIS overlap for a Davidson root to represent an NTO.
 *
 * Below this, a root is NOT a representative of that NTO and assigning it
 * anyway is a modelling error, not just a tie-break detail: the greedy match
 * then picks among candidates whose overlaps are all ~0, so the winner is
 * decided by ~1e-9 noise from the arbitrary rotation of near-degenerate
 * Davidson eigenvectors. Measured consequence before this floor existed: on
 * MgO 2x1x1 + ghost the SAME input gave STEOM roots differing by 20.7 meV
 * across runs, while the Davidson roots themselves agreed to every printed
 * digit — only the routing flipped (RFC 86.3b).
 *
 * Default 1e-4. Calibrated against every routing table in the campaign logs:
 * genuine assignments bottom out at 4.2e-3, while the noise-driven ones print
 * as 0.0000 (< 5e-5) — the floor sits in that 100x gap. An NTO left
 * unassigned contributes zero to S^IP / S^EA, which is what "no corresponding
 * root was found" means; raise `ip_eom_safety_margin` to compute more roots if
 * a real partner is expected. GANSU_STEOM_OVERLAP_FLOOR overrides (0 disables,
 * restoring the pre-fix behaviour for A/B).
 */
real_t steom_followcis_overlap_floor();

IPEOMRoutingDecision select_active_ip_roots(
    const CISNTOResult& cis_nto,
    const std::vector<IPEOMResult::PerRoot>& all_roots,
    int nocc_active,
    real_t ip_thresh);

} // namespace gansu
