/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_ERI_DIRECT_HPP
#define RYS_ERI_DIRECT_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature Direct SCF kernel — replaces MD_direct_SCF_1T1SP
__global__ void RysERI_direct(
    real_t* g_fock,
    const real_t* g_dens,
    const PrimitiveShell* g_shell,
    const int num_fock_replicas,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const real_t* g_upper_bound_factors,
    const int2* d_primitive_shell_pair_indices,
    const int num_basis,
    const real_t* g_boys_grid,
    const size_t head_bra,
    const size_t head_ket);

} // namespace gansu::gpu

#endif // RYS_ERI_DIRECT_HPP
