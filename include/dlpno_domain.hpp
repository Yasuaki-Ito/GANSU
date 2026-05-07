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
#include <utility>
#include "types.hpp"

namespace gansu {

/**
 * @brief Domain associated with a single localised occupied MO.
 */
struct DLPNODomain {
    int lmo_index = -1;                ///< Which LMO this domain belongs to
    std::vector<int> atom_indices;     ///< Selected atoms (Mulliken-cumulative order)
    std::vector<int> ao_indices;       ///< Sorted ascending AO indices
    real_t mulliken_completeness = 0.0;///< Σ_{A ∈ atoms} q^A_i achieved
};

/**
 * @brief Boughton-Pulay domain selection (J. Comp. Chem. 14, 736 (1993)).
 *
 * For each LMO `i`, compute the Mulliken population
 *
 *     q^A_i = Σ_{μ ∈ A} (S C^{LMO})_{μ i} · C^{LMO}_{μ i},
 *
 * sort atoms by `|q^A_i|` descending, and greedily add atoms until
 * the cumulative population reaches `1 − t_cut_mkn`. Returns one domain
 * per LMO with the selected AO indices in ascending order.
 *
 * @param C_LMO            [nao × nocc] row-major localised occupied MOs.
 * @param S                [nao × nao]  AO overlap (row-major, symmetric).
 * @param atom_ao_ranges   per-atom (start, end_exclusive) AO ranges.
 * @param t_cut_mkn        completeness deficit threshold (e.g. 1e-3).
 * @param verbose          0 silent, 1 summary, ≥2 per-LMO log.
 */
std::vector<DLPNODomain> build_lmo_domains(
    const real_t* C_LMO,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    real_t t_cut_mkn,
    int verbose);

/// Sorted union of two AO-index lists (used to build pair domains).
std::vector<int> merge_ao_index_sets(
    const std::vector<int>& a,
    const std::vector<int>& b);

} // namespace gansu
