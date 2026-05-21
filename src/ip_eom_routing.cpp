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
 * @file ip_eom_routing.cpp
 * @brief P1 sub-phase 1.9 + 1.10 + 1.13 — pure CPU FollowCIS overlap selector
 *        for IP-EOM-CCSD root → active NTO routing.
 *
 * The interesting code is `select_active_ip_roots`, used by
 * `compute_ip_eom_ccsd_impl` (production driver) and by `test_ip_eom_ccsd.cu`
 * (unit tests of the routing logic in isolation).
 */

#include "ip_eom_result.hpp"

#include <algorithm>
#include <cstddef>

#include "cis_nto_active_space.hpp"

namespace gansu {

IPEOMRoutingDecision select_active_ip_roots(
    const CISNTOResult& cis_nto,
    const std::vector<IPEOMResult::PerRoot>& all_roots,
    int nocc_active,
    real_t ip_thresh)
{
    const int n_act_occ = cis_nto.n_act_occ;
    const int n_roots   = static_cast<int>(all_roots.size());

    IPEOMRoutingDecision out;
    out.assigned_root_for_m.assign(n_act_occ, -1);
    out.overlap_for_m.assign(n_act_occ, real_t(0.0));
    out.root_taken.assign(n_roots, false);

    if (n_act_occ <= 0 || n_roots <= 0 || nocc_active <= 0) return out;

    // ‖R1_k‖² per root (used to normalise overlaps).
    std::vector<real_t> r1_norm2(n_roots, 0.0);
    for (int k = 0; k < n_roots; ++k) {
        const auto& R1 = all_roots[k].R1;
        for (real_t v : R1) r1_norm2[k] += v * v;
    }

    // overlap[k * n_act_occ + m̃] = (<U_occ.col(m̃), R1_k>)² / ‖R1_k‖².
    std::vector<real_t> overlap(static_cast<size_t>(n_roots) * n_act_occ, 0.0);
    for (int k = 0; k < n_roots; ++k) {
        if (r1_norm2[k] <= 0.0) continue;
        const auto& R1 = all_roots[k].R1;
        if (static_cast<int>(R1.size()) != nocc_active) continue;
        for (int m = 0; m < n_act_occ; ++m) {
            real_t dot = 0.0;
            for (int i = 0; i < nocc_active; ++i) {
                dot += cis_nto.U_occ[static_cast<size_t>(i) * nocc_active + m] * R1[i];
            }
            overlap[static_cast<size_t>(k) * n_act_occ + m] = (dot * dot) / r1_norm2[k];
        }
    }

    // Greedy bipartite matching by descending overlap, subject to the
    // %singles filter (Sub-phase 1.10).
    std::vector<bool> m_taken(n_act_occ, false);
    for (int step = 0; step < std::min(n_act_occ, n_roots); ++step) {
        real_t best_ov = -1.0;
        int    best_k  = -1;
        int    best_m  = -1;
        for (int k = 0; k < n_roots; ++k) {
            if (out.root_taken[k]) continue;
            if (all_roots[k].percent_singles < ip_thresh) continue;
            for (int m = 0; m < n_act_occ; ++m) {
                if (m_taken[m]) continue;
                real_t ov = overlap[static_cast<size_t>(k) * n_act_occ + m];
                if (ov > best_ov) { best_ov = ov; best_k = k; best_m = m; }
            }
        }
        if (best_k < 0) break;
        out.root_taken[best_k]              = true;
        m_taken[best_m]                     = true;
        out.assigned_root_for_m[best_m]     = best_k;
        out.overlap_for_m[best_m]           = best_ov;
    }

    return out;
}

} // namespace gansu
