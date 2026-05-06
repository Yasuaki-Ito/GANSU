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
 * @file thc_grid.hpp
 * @brief Molecular numerical-integration grid for Tensor Hypercontraction (THC).
 *
 * Provides atom-centred grids built from
 *   - Treutler-Ahlrichs M3 logarithmic radial quadrature
 *   - Lebedev angular quadrature (110 / 194 / 302 points)
 *   - Becke fuzzy-cell partitioning (atomic ownership weights)
 *
 * The resulting MolecularGrid holds N_g grid points {r_P, w_P} with the Becke
 * partition weight already folded into w_P.  This is the substrate from which
 * the THC collocation matrix X^P_mu = phi_mu(r_P) is evaluated.
 *
 * Phase 2.0a scope: CPU construction of the grid (single allocation, then
 * mirrored to device).  Per-grid-point GPU kernels (collocation, LS-THC) live
 * in separate translation units.
 */

#pragma once

#include <vector>
#include <cstddef>

#include "types.hpp"
#include "molecular.hpp"

namespace gansu {

/**
 * @brief Lebedev angular grid resolution.
 *
 * Numbers are angular-point counts; ordering is from coarse (cheap) to fine
 * (accurate).  L194 is a sensible default for organic molecules; L110 is for
 * speed prototyping; L302 is for tight precision tests.
 */
enum class LebedevOrder : int {
    L110 = 110,
    L194 = 194,
    L302 = 302
};

/**
 * @brief Radial quadrature scheme.
 *
 * Phase 2.0a ships TREUTLER_AHLRICHS_M3 only (xi from element, mapping
 * x in [-1,1] -> r in [0,inf) via r = (xi/ln 2) * (1+x)^0.6 * ln(2/(1-x))).
 */
enum class RadialScheme : int {
    TREUTLER_AHLRICHS_M3 = 0
};

/**
 * @brief Becke fuzzy-cell partition stiffness (number of polynomial sweeps in
 * the smooth step function s(mu)).  k=3 is the classic Becke 1988 choice.
 */
struct BeckeOptions {
    int k = 3;
};

/**
 * @brief Knob bag for the grid builder.
 */
struct ThcGridOptions {
    LebedevOrder lebedev = LebedevOrder::L194;
    int n_radial = 50;                 ///< radial points per atom
    RadialScheme radial = RadialScheme::TREUTLER_AHLRICHS_M3;
    BeckeOptions becke{};
    /**
     * Drop grid points whose final (Becke * radial * angular) weight is
     * smaller than this threshold.  0 disables pruning.
     */
    real_t weight_eps = 1.0e-14;
};

/**
 * @brief Single 3D quadrature point with Becke-folded weight.
 *
 * Fields are public for cache-friendly array-of-struct iteration on the host.
 * The device side stores SoA vectors via MolecularGrid::host_to_device().
 */
struct GridPoint {
    real_t x;
    real_t y;
    real_t z;
    real_t w;        ///< Becke * radial * angular, ready for sum_P f(r_P)*w_P
    int atom_index;  ///< owning atom under the Becke partition (informational)
};

/**
 * @brief Molecular numerical-integration grid (host + device).
 *
 * The host vector @ref points is the canonical representation.  After it is
 * filled by build_molecular_grid(), call @ref host_to_device() once to populate
 * the SoA device buffers used by the collocation kernel.
 */
struct MolecularGrid {
    /// Host-side points (ordered by atom, then radial, then angular).
    std::vector<GridPoint> points;

    /// Number of points (== points.size() once populated).
    std::size_t num_points = 0;

    /// Per-atom (start, count) into @ref points; useful for sparsity exploitation.
    struct AtomRange { std::size_t start; std::size_t count; };
    std::vector<AtomRange> atom_ranges;

    /**
     * Device SoA buffers (column-major, length num_points each).  Owned by
     * MolecularGrid; freed in the destructor.  Null until host_to_device().
     */
    real_t* d_x = nullptr;
    real_t* d_y = nullptr;
    real_t* d_z = nullptr;
    real_t* d_w = nullptr;

    /// Mirror @ref points onto the device.  Idempotent.
    void host_to_device();

    /// Free device buffers (called automatically by destructor).
    void release_device();

    MolecularGrid() = default;
    ~MolecularGrid();
    MolecularGrid(const MolecularGrid&) = delete;
    MolecularGrid& operator=(const MolecularGrid&) = delete;
    MolecularGrid(MolecularGrid&&) noexcept;
    MolecularGrid& operator=(MolecularGrid&&) noexcept;
};

/**
 * @brief Build the molecular grid from a Molecular instance.
 *
 * Steps:
 *   1. For each atom, generate (r_i, w_i^rad) from the radial scheme.
 *   2. For each radial shell, attach the Lebedev angular sub-grid scaled by
 *      r_i^2 (the Jacobian of spherical -> Cartesian).
 *   3. Compute Becke fuzzy-cell weight w_A(r) for each point and fold it into
 *      the per-point weight.
 *   4. Drop points whose final weight is below options.weight_eps.
 *
 * The result honours sum_P w_P * f(r_P) ~ integral f(r) d^3r with relative
 * accuracy 1e-6 ... 1e-8 (Becke + Treutler M3 + Lebedev 194 typical).
 */
MolecularGrid build_molecular_grid(
    const std::vector<Atom>& atoms,
    const ThcGridOptions& options = ThcGridOptions{});

// Convenience overload that pulls atoms from a Molecular instance.
inline MolecularGrid build_molecular_grid(
    const Molecular& molecular,
    const ThcGridOptions& options = ThcGridOptions{})
{
    return build_molecular_grid(molecular.get_atoms(), options);
}

/**
 * @brief Lebedev tables (publicly exposed for unit tests).
 *
 * Returns @ref order points {x,y,z,w} on the unit sphere with weights summing
 * to 4*pi.
 */
const std::vector<GridPoint>& get_lebedev_grid(LebedevOrder order);

/**
 * @brief Treutler-Ahlrichs M3 radial node, atom-dependent xi.
 *
 * Bragg-Slater radii table is indexed by atomic number (1 == H).
 * Returns r and w_radial for the given quadrature node index k = 1..n_radial.
 */
void treutler_ahlrichs_m3(
    int atomic_number,
    int n_radial,
    std::vector<real_t>& r_out,
    std::vector<real_t>& w_out);

} // namespace gansu
