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

#ifndef GANSU_GRAD_2C_HPP
#define GANSU_GRAD_2C_HPP

#include "types.hpp"

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif

namespace gansu::gpu {

// ============================================================================
// 2c2e (P|Q) derivative kernel for RI gradient
// ============================================================================
//
// Computes
//   dE / dR_A += Σ_{P,Q} Γ^(2)_{PQ} * ∂(P|Q)/∂R_A
// for the auxiliary 2-center two-electron integrals (P|Q).
//
// Math:
//   $(P|Q)$ depends on |R_P - R_Q|^2 via Boys(ξ |R_P-R_Q|^2), ξ = αβ/(α+β).
//   In McMurchie-Davidson form, ∂(P|Q)/∂R_{P,x} = (same sum, R index t→t+1).
//   Translational invariance: ∂(P|Q)/∂R_{Q,x} = -∂(P|Q)/∂R_{P,x} (exact for 2c).
//
// I/O contract:
//   - g_gamma2_aux: row-major [naux × naux], normalized basis (cgto_norm absorbed)
//   - g_gradients : [3*num_atoms], atomicAdd accumulation
//   - cgto_norm is applied inside the kernel (same convention as the 2c2e
//     integral kernel in src/int2c2e.cu — keeps the two consistent)
//
// Symmetry:
//   - When shell_sP.start_index == shell_sQ.start_index (same shell-type bucket),
//     primitive pairs satisfy prim_P ≤ prim_Q (triangular iteration).
//     Off-diagonal (prim_P ≠ prim_Q) contribute twice (×2 symmetry factor),
//     since Γ^(2) is symmetric and we only iterate over upper triangle.
//   - When P, Q are on the same atom, the gradient contribution is exactly zero
//     by translational invariance — kernel skips those pairs.
// ============================================================================

// `no_pair_symmetry`: when true, sym_factor is forced to 1.0 (each ordered pair
// counted once). Used by distributed launchers that iterate all (s_P, s_Q)
// shell-type combinations rather than the upper triangle — see the design note
// in src/eri_ri_gradient.cu.
__global__ void compute_gradients_2c2e(
    double* g_gradients,
    const real_t* g_gamma2_aux,
    const PrimitiveShell* g_pshell_aux,
    const real_t* g_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_sP, ShellTypeInfo shell_sQ,
    const size_t num_threads,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid,
    const bool no_pair_symmetry = false);

void compute_gradients_2c2e_cpu(
    double* g_gradients,
    const real_t* g_gamma2_aux,
    const PrimitiveShell* g_pshell_aux,
    const real_t* g_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_sP, ShellTypeInfo shell_sQ,
    const size_t num_threads,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid,
    const bool no_pair_symmetry = false);

} // namespace gansu::gpu

#endif // GANSU_GRAD_2C_HPP
