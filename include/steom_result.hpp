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
 * @file steom_result.hpp
 * @brief Excited-state STEOM-CCSD result POD (bt-PNO-STEOM Phase P3).
 *
 * The canonical STEOM-CCSD effective Hamiltonian G^{1h1p} is diagonalized over
 * the singles excitation manifold (dim = nocc_active × nvir) using the
 * IP-EOM-CCSD R2 / EA-EOM-CCSD R2 amplitudes (from `IPEOMResult.per_active` /
 * `EAEOMResult.per_active`) as the second similarity transform Ŝ (sub-phase
 * 3.3 in normalised form). Excited-state vectors come out CIS-sized but
 * carry near-EOM-CCSD-quality energies on the active subspace.
 */

#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace gansu {

struct STEOMResult {
    int nocc_active = 0;
    int nvir        = 0;
    int num_frozen  = 0;
    int n_states    = 0;    ///< Number of excited states actually produced (≤ n_excited_states)
    real_t ground_corr_energy = 0.0;  ///< cluster CCSD ground correlation energy (DMET-STEOM; reported as the post-HF correction)

    struct PerRoot {
        real_t omega          = 0.0;   ///< Excitation energy in Ha (positive)
        std::vector<real_t> R1;        ///< Singles amplitudes [nocc_active * nvir]
        real_t eta            = -1.0;  ///< % active character; populated by sub-phase 3.10 (sentinel -1.0 = not yet computed)
        real_t percent_active_occ = -1.0; ///< Σ_ã,a |⟨U_occ.col(ã) | R1.row(a)⟩|² (sub-phase 3.10)
        real_t percent_active_vir = -1.0; ///< Σ_ẽ,i |⟨U_vir.col(ẽ) | R1.col(i)⟩|² (sub-phase 3.10)
    };

    std::vector<PerRoot> per_root;     ///< Sorted by ascending omega.

    /// Human-readable summary for the final summary print + memory dump.
    std::string report;
};

} // namespace gansu
