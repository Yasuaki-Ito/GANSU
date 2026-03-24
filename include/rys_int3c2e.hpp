/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_INT3C2E_HPP
#define RYS_INT3C2E_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature 3-center ERI kernel: (pq|A)
// Replaces MD_int3c2e_1T1SP for D+ basis shells
__global__ void Rys_int3c2e(
    real_t* g_result,
    const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux,
    const real_t* d_cgto_normalization_factors,
    const real_t* d_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,
    int64_t num_tasks,
    int num_basis,
    const size_t2* d_primitive_shell_pair_indices,
    const double* g_upper_bound_factors,
    const double* g_auxiliary_upper_bound_factors,
    const double schwarz_screening_threshold,
    int num_auxiliary_basis,
    const double* g_boys_grid);

} // namespace gansu::gpu

#endif // RYS_INT3C2E_HPP
