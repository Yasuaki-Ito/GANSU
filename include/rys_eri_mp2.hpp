/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_ERI_MP2_HPP
#define RYS_ERI_MP2_HPP

#include "types.hpp"

namespace gansu::gpu {

/// Rys quadrature kernel for Direct MP2 half-transformation.
/// Computes H(mu,nu,la,i) = sum_sigma (mu nu | la sigma) * C(sigma, i)
/// for i in [i_start, i_start + block_occ).
///
/// Output buffer d_half has shape (nao, nao, nao, block_occ) and is indexed as:
///   d_half[((mu*nao + nu)*nao + la)*block_occ + (i - i_start)]
///
/// NOTE: Uses Rys quadrature for all angular momenta.
///       Students can create specialized s/p kernels for optimization.
__global__ void RysERI_half_transform(
    real_t* d_half,
    const real_t* d_C,
    const int i_start,
    const int block_occ,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const real_t* g_upper_bound_factors,
    const int num_basis,
    const real_t* g_boys_grid,
    const size_t head_bra,
    const size_t head_ket);

} // namespace gansu::gpu

#endif // RYS_ERI_MP2_HPP
