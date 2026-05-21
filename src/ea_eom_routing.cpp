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
 * @file ea_eom_routing.cpp
 * @brief P2 sub-phase 2.9 + 2.10 + 2.13 — pure CPU FollowCIS overlap selector
 *        for EA-EOM-CCSD root → active virtual NTO routing.
 *
 * Mirrors `select_active_ip_roots` (see src/ip_eom_routing.cpp); the only
 * differences are
 *   • U_occ → U_vir   (read CISNTOResult.U_vir / n_act_vir)
 *   • R1 size = nvir   (instead of nocc_active)
 * The greedy bipartite matching + %singles filter is identical so the two
 * routines could be templated in a future cleanup pass (P3).
 */

#include "ea_eom_result.hpp"

#include <algorithm>
#include <cstddef>

#include "cis_nto_active_space.hpp"

namespace gansu {

EAEOMRoutingDecision select_active_ea_roots(
    const CISNTOResult& cis_nto,
    const std::vector<EAEOMResult::PerRoot>& all_roots,
    int nvir,
    real_t ea_thresh)
{
    const int n_act_vir = cis_nto.n_act_vir;
    const int n_roots   = static_cast<int>(all_roots.size());

    EAEOMRoutingDecision out;
    out.assigned_root_for_a.assign(n_act_vir, -1);
    out.overlap_for_a.assign(n_act_vir, real_t(0.0));
    out.root_taken.assign(n_roots, false);

    if (n_act_vir <= 0 || n_roots <= 0 || nvir <= 0) return out;

    // ‖R1_k‖² per root (used to normalise overlaps).
    std::vector<real_t> r1_norm2(n_roots, 0.0);
    for (int k = 0; k < n_roots; ++k) {
        const auto& R1 = all_roots[k].R1;
        for (real_t v : R1) r1_norm2[k] += v * v;
    }

    // overlap[k * n_act_vir + a] = (<U_vir.col(a), R1_k>)² / ‖R1_k‖².
    std::vector<real_t> overlap(static_cast<size_t>(n_roots) * n_act_vir, 0.0);
    for (int k = 0; k < n_roots; ++k) {
        if (r1_norm2[k] <= 0.0) continue;
        const auto& R1 = all_roots[k].R1;
        if (static_cast<int>(R1.size()) != nvir) continue;
        for (int m = 0; m < n_act_vir; ++m) {
            real_t dot = 0.0;
            for (int a = 0; a < nvir; ++a) {
                dot += cis_nto.U_vir[static_cast<size_t>(a) * nvir + m] * R1[a];
            }
            overlap[static_cast<size_t>(k) * n_act_vir + m] = (dot * dot) / r1_norm2[k];
        }
    }

    // Greedy bipartite matching by descending overlap, subject to %singles filter.
    std::vector<bool> m_taken(n_act_vir, false);
    for (int step = 0; step < std::min(n_act_vir, n_roots); ++step) {
        real_t best_ov = -1.0;
        int    best_k  = -1;
        int    best_m  = -1;
        for (int k = 0; k < n_roots; ++k) {
            if (out.root_taken[k]) continue;
            if (all_roots[k].percent_singles < ea_thresh) continue;
            for (int m = 0; m < n_act_vir; ++m) {
                if (m_taken[m]) continue;
                real_t ov = overlap[static_cast<size_t>(k) * n_act_vir + m];
                if (ov > best_ov) { best_ov = ov; best_k = k; best_m = m; }
            }
        }
        if (best_k < 0) break;
        out.root_taken[best_k]              = true;
        m_taken[best_m]                     = true;
        out.assigned_root_for_a[best_m]     = best_k;
        out.overlap_for_a[best_m]           = best_ov;
    }

    return out;
}

} // namespace gansu
