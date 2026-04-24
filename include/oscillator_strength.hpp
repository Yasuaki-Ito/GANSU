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
 * @file oscillator_strength.hpp
 * @brief Oscillator strength computation for excited states
 *
 * Provides:
 *   1. AO dipole integrals ⟨μ|r|ν⟩ via McMurchie-Davidson on CPU
 *   2. MO transformation of dipole integrals
 *   3. Transition dipole moments from R1 eigenvectors
 *   4. Oscillator strengths f = (2/3) ω |⟨0|μ|n⟩|²
 */

#pragma once

#include <string>
#include <vector>
#include "types.hpp"

namespace gansu {

/**
 * @brief Compute AO dipole integrals on CPU
 *
 * Computes ⟨μ|r_d|ν⟩ for d = x, y, z using McMurchie-Davidson.
 * Gauge origin is the coordinate origin (0, 0, 0).
 */
void compute_ao_dipole_integrals(
    const PrimitiveShell* shells, size_t num_shells,
    const real_t* cgto_norms,
    int nao,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    std::vector<real_t>& dipole_x,
    std::vector<real_t>& dipole_y,
    std::vector<real_t>& dipole_z);

/**
 * @brief Transform AO dipole integrals to MO basis (ov block only)
 * @param C MO coefficient matrix [nao×nao], row-major: C[mu*nao+p]
 */
std::vector<real_t> transform_dipole_ao_to_mo_ov(
    const std::vector<real_t>& dipole_ao,
    const real_t* C, int nao, int nocc, int nvir);

/**
 * @brief Compute oscillator strengths from R1 eigenvectors and MO dipole integrals
 *
 * f_n = (2/3) ω_n |⟨0|μ|n⟩|²
 * ⟨0|μ_d|n⟩ = √2 Σ_{ia} R1^n_{ia} μ^d_{ia}  (RHF factor √2 for singlet)
 */
std::vector<real_t> compute_oscillator_strengths(
    const real_t* h_eigenvectors,
    const std::vector<real_t>& excitation_energies,
    const std::vector<real_t>& dipole_mo_ov_x,
    const std::vector<real_t>& dipole_mo_ov_y,
    const std::vector<real_t>& dipole_mo_ov_z,
    int n_states, int nocc, int nvir);

/**
 * @brief Result of excited state analysis
 */
struct ExcitedStateResult {
    std::vector<real_t> oscillator_strengths;
    std::string report;  // Formatted table for final summary
};

/**
 * @brief Compute oscillator strengths and generate excited state report
 *
 * Common function for all excited state methods (CIS, ADC(2), EOM-*).
 * Computes AO dipole integrals, transforms to MO, and generates a
 * formatted report string (not printed to stdout).
 *
 * @return ExcitedStateResult with oscillator strengths and report string
 */
ExcitedStateResult compute_excited_state_properties(
    const std::string& method_name,
    const PrimitiveShell* shells, size_t num_shells,
    const real_t* cgto_norms,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const real_t* C_host,
    const std::vector<real_t>& excitation_energies,
    const real_t* h_eigenvectors,
    int n_states, int nao, int nocc, int nvir,
    int occ_offset = 0, int vir_start = -1);

} // namespace gansu
