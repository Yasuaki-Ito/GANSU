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

#ifndef RYS_ERI_HPP
#define RYS_ERI_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature ERI kernel — replaces MD_1T1SP for D+ shells
__global__ void RysERI(
    double* g_int2e,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const double* g_upper_bound_factors,
    const int num_basis,
    const double* g_boys_grid,
    const size_t head_bra, const size_t head_ket);

// Hash version: writes to hash table instead of dense array
__global__ void RysERI_Hash(
    unsigned long long* hash_keys,
    double* hash_values,
    size_t hash_capacity_mask,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const double* g_upper_bound_factors,
    const int num_basis,
    const double* g_boys_grid,
    const size_t head_bra, const size_t head_ket,
    size_t capacity_mask_unused = 0);

} // namespace gansu::gpu

#endif // RYS_ERI_HPP
