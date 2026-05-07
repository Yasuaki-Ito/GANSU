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
#include <string>
#include "types.hpp"

namespace gansu {

/**
 * @brief Result of an occupied-MO localization run.
 *
 * Storage convention: row-major. `C_LMO` has shape (nao × nocc) — element
 * (μ, i) at index `μ * nocc + i`. `U` is the rotation that maps canonical
 * occupied MOs to LMOs: `C_LMO = C_occ · U`, with shape (nocc × nocc) and
 * element (j, i) at `j * nocc + i`.
 */
struct DLPNOLocalizationResult {
    std::vector<real_t> C_LMO;
    std::vector<real_t> U;
    int n_sweeps = 0;
    real_t functional_initial = 0.0;
    real_t functional_final = 0.0;
    bool converged = false;
};

/// Pipek-Mezey localization (Mulliken population) of occupied MOs.
///
/// Maximises  L = Σ_i Σ_A (P^A_{ii})^2  with
///   P^A_{kl} = (1/2) Σ_{μ ∈ A} [C_{μk}(SC)_{μl} + C_{μl}(SC)_{μk}]
/// via 2×2 Jacobi sweeps (Pipek & Mezey, JCP 90, 4916 (1989)).
///
/// @param C_occ            [nao × nocc] row-major occupied MO coefficients in AO basis.
/// @param S                [nao × nao]  row-major AO overlap matrix.
/// @param atom_ao_ranges   per-atom (start, end_exclusive) AO ranges.
/// @param max_sweep        max Jacobi sweep count.
/// @param conv_tol         convergence on per-sweep functional gain.
/// @param verbose          0 silent, 1 summary, ≥2 per-sweep log.
DLPNOLocalizationResult localize_pipek_mezey(
    const real_t* C_occ,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    int max_sweep,
    real_t conv_tol,
    int verbose);

/// Foster-Boys localization (minimise Σ_i [⟨φ_i|r²|φ_i⟩ − ⟨φ_i|r|φ_i⟩²]).
///
/// `D_x`, `D_y`, `D_z` are dipole-moment integrals in AO basis (row-major,
/// nao × nao, symmetric). Implementation TODO — declared here for the Phase 0
/// API surface so callers can switch via dlpno_localizer.
DLPNOLocalizationResult localize_foster_boys(
    const real_t* C_occ,
    const real_t* D_x,
    const real_t* D_y,
    const real_t* D_z,
    int nao, int nocc,
    int max_sweep,
    real_t conv_tol,
    int verbose);

/// Convenience driver dispatching on the localizer name ("pm" or "boys").
/// Boys requires dipole integrals; pass nullptrs and use "pm" if unavailable.
DLPNOLocalizationResult localize_occupied(
    const std::string& method,
    const real_t* C_occ,
    const real_t* S,
    const real_t* D_x, const real_t* D_y, const real_t* D_z,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    int max_sweep,
    real_t conv_tol,
    int verbose);

/// Compute the PM functional value L = Σ_i Σ_A (P^A_{ii})^2 for the supplied
/// occupied coefficients. Used by tests and diagnostics.
real_t pipek_mezey_functional(
    const real_t* C_occ,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges);

} // namespace gansu
