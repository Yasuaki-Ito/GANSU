/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_domain.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace gansu {

namespace {

// SC = S · C, row-major (nao × nocc)
void compute_SC(const real_t* S, const real_t* C, int nao, int nocc,
                std::vector<real_t>& SC)
{
    SC.assign(static_cast<size_t>(nao) * nocc, 0.0);
    for (int mu = 0; mu < nao; mu++) {
        for (int i = 0; i < nocc; i++) {
            real_t v = 0.0;
            for (int nu = 0; nu < nao; nu++) {
                v += S[mu * nao + nu] * C[nu * nocc + i];
            }
            SC[mu * nocc + i] = v;
        }
    }
}

} // namespace

std::vector<DLPNODomain> build_lmo_domains(
    const real_t* C_LMO,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    real_t t_cut_mkn,
    int verbose)
{
    std::vector<DLPNODomain> domains(nocc);
    if (nocc == 0) return domains;

    std::vector<real_t> SC;
    compute_SC(S, C_LMO, nao, nocc, SC);

    const int n_atoms = static_cast<int>(atom_ao_ranges.size());
    const real_t target_completeness = 1.0 - t_cut_mkn;

    // Domain construction levers (mode-2 fix, opt-in; default = legacy Mulliken-greedy).
    //   GANSU_DLPNO_FULL_DOMAIN=1     : every LMO domain = all atoms (= canonical;
    //                                   diagnostic to confirm the domain is the limiter).
    //   GANSU_DLPNO_DOMAIN_OVERLAP_EXT=<thr> : after the Mulliken-greedy domain, extend by
    //                                   spatial connectivity — add any atom whose AOs overlap
    //                                   (max|S| ≥ thr) the domain AOs. Captures the correlation
    //                                   tail (ORCA-style extended domain); typical thr 1e-3..1e-2.
    const bool  full_domain   = (std::getenv("GANSU_DLPNO_FULL_DOMAIN") != nullptr);
    const char* ext_env       = std::getenv("GANSU_DLPNO_DOMAIN_OVERLAP_EXT");
    const real_t overlap_ext  = (ext_env != nullptr) ? std::atof(ext_env) : 0.0;

    for (int i = 0; i < nocc; i++) {
        // Mulliken population per atom for LMO i.
        std::vector<real_t> q(n_atoms, 0.0);
        for (int a = 0; a < n_atoms; a++) {
            const int mu0 = atom_ao_ranges[a].first;
            const int mu1 = atom_ao_ranges[a].second;
            real_t s = 0.0;
            for (int mu = mu0; mu < mu1; mu++) {
                s += SC[mu * nocc + i] * C_LMO[mu * nocc + i];
            }
            q[a] = s;
        }

        // Sort atom indices by |q|, descending.
        std::vector<int> order(n_atoms);
        for (int a = 0; a < n_atoms; a++) order[a] = a;
        std::sort(order.begin(), order.end(),
                  [&](int x, int y) { return std::fabs(q[x]) > std::fabs(q[y]); });

        // Build the LMO domain atom set.
        DLPNODomain& dom = domains[i];
        dom.lmo_index = i;
        real_t cum = 0.0;
        if (full_domain) {
            for (int a = 0; a < n_atoms; a++) { dom.atom_indices.push_back(a); cum += q[a]; }
        } else {
            // Greedy accumulate atoms by |Mulliken population| until the cumulative
            // population reaches 1 − t_cut_mkn (≥ at least one atom).
            for (int idx = 0; idx < n_atoms; idx++) {
                const int a = order[idx];
                dom.atom_indices.push_back(a);
                cum += q[a];
                if (cum >= target_completeness) break;
            }
            // Spatial-connectivity extension (opt-in): add any atom whose AOs overlap the
            // current domain AOs above `overlap_ext`. Recovers the diffuse correlation tail
            // the charge-based domain drops (ORCA-style extended domain). One connectivity
            // shell; scan is O(n_dom_ao · n_b_ao) per added atom, cheap for the domain sizes.
            if (overlap_ext > 0.0) {
                std::vector<char> in_dom(n_atoms, 0);
                for (int a : dom.atom_indices) in_dom[a] = 1;
                std::vector<int> added;
                for (int b = 0; b < n_atoms; b++) {
                    if (in_dom[b]) continue;
                    real_t maxs = 0.0;
                    for (int a : dom.atom_indices)
                        for (int mu = atom_ao_ranges[a].first; mu < atom_ao_ranges[a].second; mu++)
                            for (int nu = atom_ao_ranges[b].first; nu < atom_ao_ranges[b].second; nu++)
                                maxs = std::max(maxs, std::fabs(S[(size_t)mu * nao + nu]));
                    if (maxs >= overlap_ext) added.push_back(b);
                }
                for (int b : added) dom.atom_indices.push_back(b);
            }
        }
        dom.mulliken_completeness = cum;

        // Build sorted AO list.
        for (int a : dom.atom_indices) {
            for (int mu = atom_ao_ranges[a].first;
                 mu < atom_ao_ranges[a].second; mu++) {
                dom.ao_indices.push_back(mu);
            }
        }
        std::sort(dom.ao_indices.begin(), dom.ao_indices.end());

        if (verbose >= 2) {
            std::cout << "[DLPNO domain] LMO " << std::setw(3) << i
                      << " atoms=" << dom.atom_indices.size()
                      << " AOs=" << dom.ao_indices.size()
                      << " Σq=" << std::fixed << std::setprecision(6)
                      << dom.mulliken_completeness << std::endl;
        }
    }

    if (verbose >= 1) {
        size_t total_aos = 0, max_aos = 0, min_aos = nao;
        for (const auto& d : domains) {
            total_aos += d.ao_indices.size();
            max_aos = std::max(max_aos, d.ao_indices.size());
            if (!d.ao_indices.empty())
                min_aos = std::min(min_aos, d.ao_indices.size());
        }
        const real_t avg = nocc > 0
            ? static_cast<real_t>(total_aos) / static_cast<real_t>(nocc)
            : 0.0;
        std::cout << "[DLPNO domain] " << nocc << " LMOs, AO/domain "
                  << "min=" << min_aos << " avg=" << std::fixed
                  << std::setprecision(1) << avg
                  << " max=" << max_aos
                  << " (target completeness ≥ "
                  << std::scientific << std::setprecision(2)
                  << target_completeness << ")" << std::endl;
    }

    return domains;
}

std::vector<int> merge_ao_index_sets(
    const std::vector<int>& a,
    const std::vector<int>& b)
{
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(out));
    return out;
}

} // namespace gansu
