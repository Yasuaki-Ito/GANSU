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

#ifndef GANSU_GRAD_3C_HPP
#define GANSU_GRAD_3C_HPP

#include "types.hpp"

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif

namespace gansu::gpu {

// ============================================================================
// 3c2e (μν|P) derivative kernel for RI gradient
// ============================================================================
//
// Computes
//   dE / dR_A += Σ_{P,μ,ν} Γ^(3)_{P,μν} * ∂(μν|P)/∂R_A
// where (μν|P) is the auxiliary 3-center two-electron integral.
//
// Math (McMurchie-Davidson):
//   (μν|P) = c_μ c_ν c_P · cgto_norm · prim_norm · exp(-αβ|R_{μν}|^2/p) · 2π^{5/2}/(pγ√(p+γ))
//          · Σ_{tuvτνφ} E_t^{lμ,lν}(p, R_{μν}, R_P-R_μ) · E_u · E_v
//                      · E_τ^{lP,0}(γ, 0, 0) · E_ν · E_φ · (-1)^{τ+ν+φ}
//                      · R^{(0)}_{t+τ, u+ν, v+φ}(ξ_3, R_P_prod - R_aux)
//   with p = α + β, γ = α_aux, ξ_3 = pγ/(p+γ), R_P_prod = (αA_μ + βA_ν)/p.
//
// All 3 atom centers are differentiated DIRECTLY (no translational invariance
// fallback — see §2.6 of RI_Gradient.md for the reasoning):
//   - ∂/∂R_μ: Et derivative on μ side (compute_grad_A*) + standard aux-side Et
//   - ∂/∂R_ν: Et derivative on ν side (compute_grad_B*) + standard aux-side Et
//   - ∂/∂R_P_aux: R index bump (compute_grad_C*) + standard aux-side Et
//                 (sign flip absorbed in the +R^{(t+1)} convention vs ∂/∂R_C)
//
// I/O contract:
//   - g_gamma3 : row-major [naux × num_basis × num_basis], normalized basis
//                (cgto_norm absorbed). Access as
//                g_gamma3[P * num_basis*num_basis + μ * num_basis + ν].
//   - g_gradients : [3*num_atoms], atomicAdd accumulation.
//   - cgto_norm is applied inside the kernel (same convention as the 3c2e
//     integral kernel in src/int3c2e.cu).
//
// Iteration:
//   - All (prim_μ, prim_ν, prim_P) triples for the given (shell_s_μ, shell_s_ν,
//     shell_s_P) buckets are enumerated. No (μ ↔ ν) symmetry is used in the
//     primitive enumeration — Γ^(3) is symmetric in (μ, ν), so iterating all
//     pairs reproduces the full sum without extra factors. The caller therefore
//     also passes all (shell_s_μ, shell_s_ν) shell-type combinations, not just
//     upper-triangle pairs.
// ============================================================================

__global__ void compute_gradients_3c2e(
    double* g_gradients,
    const real_t* g_gamma3,
    const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux,
    const real_t* g_cgto_normalization_factors,
    const real_t* g_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s_mu, ShellTypeInfo shell_s_nu, ShellTypeInfo shell_s_P,
    const size_t num_threads,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid);

void compute_gradients_3c2e_cpu(
    double* g_gradients,
    const real_t* g_gamma3,
    const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux,
    const real_t* g_cgto_normalization_factors,
    const real_t* g_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s_mu, ShellTypeInfo shell_s_nu, ShellTypeInfo shell_s_P,
    const size_t num_threads,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid);

} // namespace gansu::gpu

#endif // GANSU_GRAD_3C_HPP
