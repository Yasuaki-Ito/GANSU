/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef INT2E_DIRECT_CUH
#define INT2E_DIRECT_CUH

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu{


__global__ void composeFockMatrix(
    real_t* g_fock_matrix, real_t* g_fock_matrix_replicas, const real_t* g_int1e, const int num_basis, const int num_fock_replicas);

__global__ void ssss2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);
__global__ void sssp2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);
__global__ void sspp2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);
__global__ void spsp2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);
__global__ void sppp2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);
__global__ void pppp2e_direct(real_t* g_fock_matrix, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_braket, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, const size_t head_bra, const size_t head_ket, const int num_fock_replicas);

__global__ void ssss2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);
__global__ void sssp2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);
__global__ void sspp2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);
__global__ void spsp2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);
__global__ void sppp2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);
__global__ void pppp2e_dynamic(real_t* g_fock_matrix_replicas, const PrimitiveShell* g_primitive_shells, const int2* g_primitive_shell_pair_indices, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const real_t schwarz_screening_threshold, const real_t* g_schwarz_upper_bound_factors, const int num_basis, const real_t* g_boys_grid, const real_t* g_density_matrix, int* g_counter, int* g_min_skipped_column, const size_t head_bra, const size_t head_ket, const size_t num_bra, const size_t num_ket, const int num_fock_replicas);

__global__ void MD_direct_SCF_1T1SP(real_t* g_fock, const real_t* g_dens, const PrimitiveShell* g_shell, const int num_fock_replicas, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t swartz_screening_threshold, const real_t* g_upper_bound_factors, const int2* d_primitive_shell_pair_indices, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);

using eri_kernel_direct_t = void (*)(real_t*, const PrimitiveShell*, const int2*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const real_t, const real_t*, const int, const real_t*, const real_t*, const size_t, const size_t, const int);
using eri_kernel_dynamic_t = void (*)(real_t*, const PrimitiveShell*, const int2*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const real_t, const real_t*, const int, const real_t*, const real_t*, int*, int*, const size_t, const size_t, const size_t, const size_t, const int);

inline eri_kernel_direct_t get_eri_kernel_direct(int a, int b, int c, int d) {
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }

    if      (a == 0 && b == 0 && c == 0 && d == 0) return ssss2e_direct;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return sssp2e_direct;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return sspp2e_direct;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return spsp2e_direct;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return sppp2e_direct;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return pppp2e_direct;
    else throw std::runtime_error("Invalid shell type");
}

inline eri_kernel_dynamic_t get_eri_kernel_dynamic(int a, int b, int c, int d) {
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }

    if      (a == 0 && b == 0 && c == 0 && d == 0) return ssss2e_dynamic;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return sssp2e_dynamic;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return sspp2e_dynamic;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return spsp2e_dynamic;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return sppp2e_dynamic;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return pppp2e_dynamic;
    else throw std::runtime_error("Invalid shell type");
}









} // namespace gansu::gpu

#endif