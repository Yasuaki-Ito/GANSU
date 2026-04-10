/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_GRAD_G_HPP
#define RYS_GRAD_G_HPP

#include "types.hpp"

namespace gansu::gpu {

// Rys quadrature ERI gradient kernel (RHF)
__global__ void Rys_compute_gradients_two_electron(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid);

// Rys quadrature ERI gradient kernel (RHF + non-separable 2-PDM correction).
__global__ void Rys_compute_gradients_two_electron_2pdm(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid,
    const double* g_gamma_4idx);

// Rys quadrature ERI gradient kernel (UHF)
__global__ void Rys_compute_gradients_two_electron_uhf(
    double* g_gradients,
    const real_t* g_density_matrix_alpha,
    const real_t* g_density_matrix_beta,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid);

// CPU host-callable mirrors of the three Rys 2-electron gradient kernels.
// Used by the analytic CPU gradient path in gpu_manager.cu.
void Rys_compute_gradients_two_electron_cpu(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid);

void Rys_compute_gradients_two_electron_2pdm_cpu(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid,
    const double* g_gamma_4idx);

void Rys_compute_gradients_two_electron_uhf_cpu(
    double* g_gradients,
    const real_t* g_density_matrix_alpha,
    const real_t* g_density_matrix_beta,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid);

} // namespace gansu::gpu

#endif // RYS_GRAD_G_HPP
