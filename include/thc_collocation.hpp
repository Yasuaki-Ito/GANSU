/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_collocation.hpp
 * @brief THC collocation matrix X^P_mu = phi_mu(r_P).
 *
 * The Cartesian-Gaussian basis used by GANSU is
 *   phi_mu(r) = N_mu * sum_p c_p * (x-Ax)^lx (y-Ay)^ly (z-Az)^lz
 *                                * exp(-alpha_p * |r - A|^2)
 * where N_mu = cgto_normalization_factors[mu] (one per Cartesian component),
 * (lx, ly, lz) follow the GANSU convention
 *   for lx = L .. 0:
 *     for ly = L-lx .. 0:
 *       lz = L - lx - ly
 * and primitives sharing the same contracted basis function are summed.
 *
 * Storage convention: X is laid out as [N_bas x N_g] column-major,
 *   X[mu + P * N_bas].
 * This makes filling the column for one grid point P contiguous (each thread
 * writes consecutive mu indices) and lets the downstream Gram matrix
 *   G_PR = sum_mu X^P_mu * X^R_mu
 * fall out of a single cuBLAS DGEMM with op_T x op_N.
 */

#pragma once

#include <vector>
#include <cstddef>
#include <memory>

#include "types.hpp"
#include "molecular.hpp"
#include "thc_grid.hpp"
#ifndef GANSU_CPU_ONLY
#include "device_host_memory.hpp"
#endif

namespace gansu {

/**
 * @brief CPU reference implementation of the THC collocation matrix.
 *
 * Single-threaded, used as the ground truth for the GPU kernel.  Allocates
 * and returns the matrix on the host.
 *
 * @param molecular  Molecule (provides primitive_shells, cgto_normalisation).
 * @param grid       Pre-built molecular integration grid.
 * @return std::vector<real_t> of size N_bas * N_g, column-major,
 *         X[mu + P * N_bas].
 */
std::vector<real_t> compute_X_ao_cpu(int N_bas,
                                     const std::vector<PrimitiveShell>& prims,
                                     const std::vector<real_t>& cgto_norms,
                                     const MolecularGrid& grid);

// Convenience overload that pulls primitives + norms from a Molecular instance.
inline std::vector<real_t> compute_X_ao_cpu(const Molecular& mol,
                                            const MolecularGrid& grid)
{
    return compute_X_ao_cpu(static_cast<int>(mol.get_num_basis()),
                             mol.get_primitive_shells(),
                             mol.get_cgto_normalization_factors(),
                             grid);
}

/**
 * @brief Build the grid-quadrature overlap matrix.
 *
 *   S_THC_{mu,nu} = sum_P w_P * X^P_mu * X^P_nu.
 *
 * Used to validate the collocation against GANSU's analytic overlap matrix.
 *
 * @param X_ao  Collocation matrix [N_bas x N_g] column-major.
 * @param grid  Grid (weights are pulled from grid.points[*].w).
 * @param N_bas Number of basis functions.
 * @return Symmetric S_THC of size N_bas * N_bas, column-major.
 */
std::vector<real_t> build_overlap_thc_cpu(const std::vector<real_t>& X_ao,
                                          const MolecularGrid& grid,
                                          int N_bas);

#ifndef GANSU_CPU_ONLY

/**
 * @brief GPU collocation kernel.
 *
 * Computes X^P_mu = phi_mu(r_P) using the GANSU canonical Cartesian-Gaussian
 * convention (per-component primitive normalisation + cgto normalisation, same
 * as compute_X_ao_cpu).  Output is written to @ref d_X_out, [N_bas x N_g]
 * column-major (X[mu + P*N_bas]).
 *
 * Requires d_X_out to be zero-initialised (the kernel only accumulates).
 *
 * @param N_bas, N_g, N_prim
 * @param d_grid_x/y/z   Grid coordinates [N_g] each (from MolecularGrid).
 * @param d_prims        Primitive shells [N_prim] (struct copied to device).
 * @param d_cgto_norms   Per-basis-function CGTO normalisation factors [N_bas].
 * @param d_X_out        Pre-allocated device buffer [N_bas * N_g] doubles.
 */
void compute_X_ao_gpu_impl(int N_bas, int N_g, int N_prim,
                            const real_t* d_grid_x,
                            const real_t* d_grid_y,
                            const real_t* d_grid_z,
                            const PrimitiveShell* d_prims,
                            const real_t* d_cgto_norms,
                            real_t* d_X_out);

/**
 * @brief High-level wrapper that allocates device buffers, copies primitive
 * shells / cgto-norms to device, ensures @p grid has been mirrored to device,
 * and runs the kernel.
 *
 * @param mol  Host molecule (provides primitive shells + norms).
 * @param grid Molecular grid; must already have host_to_device() called or
 *             this function will call it.
 * @return Owning unique_ptr to a DeviceHostMatrix [N_bas x N_g] column-major.
 *         Call .toHost() to pull back to host memory.  unique_ptr is used
 *         because DeviceHostMatrix is non-movable.
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
compute_X_ao_gpu(int N_bas,
                 const std::vector<PrimitiveShell>& prims,
                 const std::vector<real_t>& cgto_norms,
                 MolecularGrid& grid);

// Convenience overload that pulls primitives + norms from a Molecular instance.
inline std::unique_ptr<DeviceHostMatrix<real_t>>
compute_X_ao_gpu(const Molecular& mol, MolecularGrid& grid)
{
    return compute_X_ao_gpu(static_cast<int>(mol.get_num_basis()),
                             mol.get_primitive_shells(),
                             mol.get_cgto_normalization_factors(),
                             grid);
}

/**
 * @brief Density-based grid pruning (Phase 2.3 (B), JCTC 2025 §9.1).
 *
 * Drops grid points where the electron density
 *   ρ(r_P) = Σ_{μν} P_{μν} · φ_μ(r_P) · φ_ν(r_P)
 * falls below @p threshold.  Returns a fresh DeviceHostMatrix containing the
 * compacted X (only the surviving columns), and writes the new grid size to
 * @p N_g_kept_out.
 *
 * Implementation:
 *   - Y = P × X (N_bas × N_g)             (DGEMM)
 *   - ρ[P] = Σ_μ X[μ, P] × Y[μ, P]        (column-dot kernel)
 *   - mask[P] = (ρ[P] > threshold)
 *   - compact X[μ, P_kept] = X[μ, P_orig_kept]   (gather kernel)
 *
 * @param d_X_ao        [N_bas × N_g] col-major collocation on device.
 * @param d_density     [N_bas × N_bas] col-major density matrix on device
 *                      (closed-shell convention: P = 2 C_occ C_occ^T).
 * @param N_bas, N_g    Original dimensions.
 * @param threshold     Density cutoff.  Points with ρ ≤ threshold are dropped.
 *                      A threshold of 0 disables pruning (returns a copy of X).
 * @param N_g_kept_out  Output: new grid size after pruning.
 * @return DeviceHostMatrix [N_bas × N_g_kept] column-major holding the
 *         compacted collocation.
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
prune_X_by_density_gpu(const real_t* d_X_ao,
                       const real_t* d_density,
                       int N_bas, int N_g,
                       real_t threshold,
                       int* N_g_kept_out);

#endif // GANSU_CPU_ONLY

} // namespace gansu
