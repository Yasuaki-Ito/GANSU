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
 * @file cis_nto_active_space.hpp
 * @brief State-averaged CIS Natural Transition Orbital (NTO) active-space selector.
 *
 * Phase P0 of the bt-PNO-STEOM-CCSD project (see AQUA/STEOM.md §7). The module
 * consumes a batch of CIS amplitudes C^(n)_{ia} and produces the two NTO
 * occupation spectra plus the transformation matrices U_occ, U_vir that map
 * canonical (active-occupied / virtual) MOs onto the NTO basis. Subsequent
 * phases (P1 IP-EOM-CCSD, P2 EA-EOM-CCSD) consume the resulting active set.
 *
 * The state-averaged densities are
 *
 *   ρ_occ[i,j] = Σ_n w_n Σ_a C^(n)_{ia} C^(n)_{ja}
 *   ρ_vir[a,b] = Σ_n w_n Σ_i C^(n)_{ia} C^(n)_{ib}
 *
 * with default uniform weights w_n = 1/N_states. NTO occupations above
 * o_thresh / v_thresh (ORCA default 1e-3) flag active orbitals.
 */

#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace gansu {

/**
 * @brief Result of state-averaged CIS NTO analysis.
 *
 * U_occ / U_vir are row-major with column k holding the k-th NTO eigenvector in
 * the canonical (active-occupied or virtual) basis. Columns are sorted by NTO
 * occupation in descending order; the first n_act_occ / n_act_vir columns are
 * the active set. The remaining columns are retained for diagnostics so the
 * threshold can be re-evaluated without re-running the diagonalization.
 */
struct CISNTOResult {
    int nocc_active = 0;   ///< nocc - num_frozen_core (input dimension)
    int nvir        = 0;
    int num_frozen  = 0;
    int n_act_occ   = 0;   ///< number of NTO occupations > o_thresh
    int n_act_vir   = 0;   ///< number of NTO occupations > v_thresh

    std::vector<real_t> nto_occ_occupations;  ///< size nocc_active, sorted desc
    std::vector<real_t> nto_vir_occupations;  ///< size nvir, sorted desc

    std::vector<real_t> U_occ;  ///< [nocc_active * nocc_active], row-major
    std::vector<real_t> U_vir;  ///< [nvir * nvir], row-major

    std::vector<int> active_occ_indices;  ///< 0..n_act_occ-1 after the sort
    std::vector<int> active_vir_indices;

    real_t trace_occ   = 0.0;
    real_t trace_vir   = 0.0;
    real_t weight_sum  = 0.0;
    real_t o_thresh    = 1e-3;
    real_t v_thresh    = 1e-3;

    std::string report;
};

/**
 * @brief State-averaged CIS NTO active-space selector.
 *
 * Stateless functional API: compute() takes the CIS amplitudes already on the
 * host and returns the analysis result by value. There is one device session
 * inside compute() (build ρ_occ/ρ_vir + two cusolver eigendecompositions); no
 * iteration, no persistent state.
 */
class CISNTOActiveSpace {
public:
    struct Params {
        real_t o_thresh = 1e-3;             ///< Active occupied NTO occupation cutoff (ORCA OThresh)
        real_t v_thresh = 1e-3;             ///< Active virtual NTO occupation cutoff (ORCA VThresh)
        std::vector<real_t> weights;        ///< Per-state weights w_n. Empty → uniform 1/N.
        int verbose = 1;                    ///< 0 silent, 1 summary, 2 full spectrum.
    };

    /**
     * @brief Build the state-averaged NTO active space from CIS amplitudes.
     *
     * CIS amplitude layout (matches src/eri_stored_cis.cu):
     *   h_eigenvectors[ state * (nocc_active * nvir) + i_active * nvir + a ]
     * where i_active = i_canonical - num_frozen, a ∈ [0, nvir).
     *
     * @param h_eigenvectors  Host pointer to CIS amplitudes [n_states * nocc_active * nvir].
     * @param n_states        Number of CIS roots used to build the state-averaged density.
     * @param nocc_active     Active-occupied dimension (full_occ - num_frozen).
     * @param nvir            Virtual dimension.
     * @param num_frozen      Frozen-core count (passed through; needed by make_canonical_projectors).
     * @param params          Thresholds, weights, verbosity.
     */
    static CISNTOResult compute(
        const real_t* h_eigenvectors,
        int n_states,
        int nocc_active,
        int nvir,
        int num_frozen,
        const Params& params);

    /**
     * @brief Expand the active NTOs into the full canonical-MO basis.
     *
     * P_occ_can is nocc_canonical × n_act_occ row-major; the first num_frozen
     * rows are zero, the remaining rows hold the active-NTO columns of U_occ.
     * P_vir_can is nvir × n_act_vir row-major and equals U_vir's active columns
     * verbatim (no frozen offset on the virtual side). P1 IP-EOM uses these
     * projectors for FollowCIS overlap selection.
     */
    static void make_canonical_projectors(
        const CISNTOResult& result,
        int nocc_canonical,
        std::vector<real_t>& P_occ_can,
        std::vector<real_t>& P_vir_can);
};

} // namespace gansu
