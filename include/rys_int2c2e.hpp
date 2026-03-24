/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_INT2C2E_HPP
#define RYS_INT2C2E_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature 2-center ERI kernel: (A|B)
// Replaces MD_int2c2e_1T1SP for D+ auxiliary shells
__global__ void Rys_int2c2e(
    real_t* g_result,
    const PrimitiveShell* g_pshell_aux,
    const real_t* d_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    int num_shell_pairs,
    const double* g_upper_bound_factors,
    const double schwarz_screening_threshold,
    int num_auxiliary_basis,
    const double* g_boys_grid);

} // namespace gansu::gpu

#endif // RYS_INT2C2E_HPP
