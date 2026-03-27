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



#pragma once

#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

// constant values
const int WARP_SIZE = 32;
const unsigned int FULL_MASK = 0xffffffff;


// prototype declarations of CUDA kernels
__global__ void inverseSqrt_kernel(double* d_eigenvalues, const size_t size, const double threshold);
__global__ void sqrt_kernel(double* d_eigenvalues, const size_t size, const double threshold=1e-8);
__global__ void computeDensityMatrix_RHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_electron, const size_t num_basis);
__global__ void computeDensityMatrix_UHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_spin, const size_t num_basis);
__global__ void computeDensityMatrix_ROHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix_closed, double* d_density_matrix_open, double* d_density_matrix, const int num_closed, const int num_open, const size_t num_basis);
__global__ void transposeMatrixInPlace_kernel(double* d_matrix, int size);
__global__ void computeFockMatrix_RHF_kernel(const double* d_density_matrix, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix, int num_basis);
__global__ void computeFockMatrix_UHF_kernel(const double* d_density_matrix_a, const double* d_density_matrix_b, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix_a, double* d_fock_matrix_b, int num_basis);
__global__ void computeFockMatrix_ROHF_kernel(const double* d_density_matrix_closed, const double* d_density_matrix_open, const double* d_core_hamiltonian_matrix, const double* d_eri, double* d_fock_matrix_closed, double* d_fock_matrix_open, int num_basis);
__global__ void computeUnifiedFockMatrix_ROHF_kernel(const double* d_fock_mo_closed_matrix, const double* d_fock_mo_open_matrix, const ROHF_ParameterSet rohf_params, double* d_unified_fock_matrix, const int num_closed, const int num_open, const size_t num_basis);
__global__ void getMatrixTrace(const double* d_matrix, double* d_trace, const int num_basis);
__global__ void computeInitialFockMatrix_GWH_kernel(const double* d_core_hamiltonian_matrix, const double* d_overlap_matrix, double* d_fock_matrix, const int num_basis, const double c_x);
__global__ void computeRIIntermediateMatrixB_kernel(const double* d_three_center_eri, const double* d_matrix_L, double* d_matrix_B, const int num_basis, const int num_auxiliary_basis);
__global__ void weighted_sum_matrices_kernel(double* d_J, const double* d_B, const double* d_W, const int M, const int N, const bool accumulated=false);
__global__ void sum_matrices_kernel(double* d_K, const double* d_B, const int M, const int N, const bool accumulated=false);
__global__ void computeFockMatrix_RI_RHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis);
__global__ void computeFockMatrix_RI_UHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis);
__global__ void computeFockMatrix_RI_ROHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix_closed, const double* d_K_matrix_open, double* d_Fock_matrix_closed, double* d_Fock_matrix_open, const int num_basis);
__global__ void setZeroUpperTriangle(double* d_A, const int N);
__global__ void compute_diagonal_of_product(const double* A, const double* B, double* diag, const int N);
__global__ void compute_diagonal_of_product_sum(const double* A, const double* B, const double* C, double* diag, const int N);

__global__ void denseToHash_kernel(const real_t* d_dense_eri, unsigned long long* d_hash_keys, real_t* d_hash_values, size_t hash_capacity_mask, const int num_basis);
__global__ void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const unsigned long long* d_hash_keys, const real_t* d_hash_values, size_t hash_capacity_mask, real_t* d_fock_matrix, const int num_basis);
__global__ void computeFockMatrix_Hash_Push_RHF_kernel(const real_t* d_density_matrix, const unsigned long long* d_hash_keys, const real_t* d_hash_values, size_t hash_table_capacity, real_t* d_fock_matrix, const int num_basis, const int num_fock_replicas);
__global__ void computeFockMatrix_Hash_Push_Indexed_RHF_kernel(const real_t* d_density_matrix, const unsigned long long* d_hash_keys, const real_t* d_hash_values, const size_t* d_nonzero_indices, size_t num_nonzero, real_t* d_fock_matrix, const int num_basis, const int num_fock_replicas);
__global__ void computeFockMatrix_COO_Push_RHF_kernel(const real_t* d_density_matrix, const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries, real_t* d_fock_matrix, const int num_basis, const int num_fock_replicas);
__global__ void addCoreHamiltonian_kernel(const real_t* d_core_hamiltonian_matrix, real_t* d_fock_matrix, const int size);
__global__ void cleanupHashTable_kernel(unsigned long long* d_hash_keys, real_t* d_hash_values, size_t hash_table_capacity, real_t threshold);
__global__ void countNonEmptySlots_kernel(const unsigned long long* d_hash_keys, size_t hash_table_capacity, size_t* d_count);
__global__ void collectNonEmptyIndices_kernel(const unsigned long long* d_hash_keys, size_t hash_table_capacity, size_t* d_indices, size_t* d_write_pos);

} // namespace gansu::gpu