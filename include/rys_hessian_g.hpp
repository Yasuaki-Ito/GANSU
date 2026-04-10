/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_HESSIAN_G_HPP
#define RYS_HESSIAN_G_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature ERI Hessian kernel (RHF)
// Computes d²(pq|rs)/dA dB contracted with density matrix
// Output: hessian matrix [3*num_atoms × 3*num_atoms]
__global__ void Rys_compute_hessian_two_electron(
    double* g_hessian,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const int num_atoms,
    const double* g_boys_grid);

// CPU host-callable mirror.
void Rys_compute_hessian_two_electron_cpu(
    double* g_hessian,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const int num_atoms,
    const double* g_boys_grid);

} // namespace gansu::gpu

#endif // RYS_HESSIAN_G_HPP
