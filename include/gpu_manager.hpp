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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <stdexcept>

#include "types.hpp"
#include "profiler.hpp"
#include "gpu_kernels.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu::gpu{


// prototype declarations
void invertSqrtElements(real_t* d_vectors, const size_t size, const double threshold=1e-6);
void sqrtElements(real_t* d_vectors, const size_t size);
void transposeMatrixInPlace(real_t* d_matrix, const int size);
void makeDiagonalMatrix(const real_t* d_vector, real_t* d_matrix, const int size);
real_t computeMatrixTrace(const real_t* d_matrix, const int size);
int eigenDecomposition(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size);
int eigenDecompositionNonSymmetric(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size);
int partialEigenDecomposition(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size, const int num_eigenvalues);
void matrixMatrixProduct(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size, const bool transpose_A = false, const bool transpose_B = false, const bool accumulate=false);
void matrixMatrixProductRect(const double* d_A, const double* d_B, double* d_C, const int M, const int N, const int K, const bool transpose_A = false, const bool transpose_B = false, const bool accumulate = false, const double alpha = 1.0);
void matrixMatrixProductBatched(const double* d_A, const double* d_B, double* d_C, const int M, const int N, const int K, const long long strideA, const long long strideB, const long long strideC, const int batchCount, const bool transpose_A = false, const bool transpose_B = false, const bool accumulate = false, const double alpha = 1.0);
void weightedMatrixSum(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const double weight_A, const double weight_B, const int size);
void matrixAddition(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size);
void matrixSubtraction(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size);
double innerProduct(const double* d_vector_A, const double* d_vector_B, double* result, const int size);
void invertMatrix(double* d_A, const int N);
void choleskyDecomposition(double* d_A, const int N);

void computeCoreHamiltonianMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, Atom* d_atoms, PrimitiveShell* d_primitive_shells, real_t* d_boys_grid, real_t* d_cgto_normalization_factors, real_t* d_overlap_matrix, real_t* d_core_hamiltonian_matrix, const int num_atoms, const int num_basis, const std::string int1e_method, const bool verbose=false);
void computeCoefficientMatrix(const real_t* d_fock_matrix, const real_t* d_transform_matrix, real_t* d_coefficient_matrix, const int num_basis, real_t* d_orbital_energies=nullptr);

//void computeERIMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_eri_matrix, const real_t schwarz_screening_threshold,  const int num_basis, const bool verbose=false);
void computeERIMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,  real_t* d_eri_matrix, const real_t* d_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const bool verbose) ;

/// Direct MP2 half-transformation: H(mu,nu,la,i) = sum_sigma (mu nu|la sigma) C(sigma,i)
/// d_half must be pre-zeroed, shape (nao, nao, nao, block_occ).
void computeHalfTransformedERI(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_half, const real_t* d_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const real_t* d_C, int i_start, int block_occ);

void computeTwoCenterERIs(const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors, real_t* d_two_center_eri, const int num_auxiliary_basis, const real_t* d_boys_grid, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const bool verbose=false);
void computeThreeCenterERIs(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors, real_t* d_three_center_eri, const size_t2* d_primitive_shell_pair_indices, const int num_basis, const int num_auxiliary_basis, const real_t* d_boys_grid, const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const bool verbose=false);


void computeDensityMatrix_RHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis);
void computeDensityMatrix_UHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const int num_basis);
void computeDensityMatrix_ROHF(const real_t* d_coefficient_matrix, real_t* d_density_matrix_closed, real_t* d_density_matrix_open, real_t* d_density_matrix, const int num_closed, const int num_open, const int num_basis);

void computeFockMatrix_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, const int num_basis);
void computeFockMatrix_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis);
void computeFockMatrix_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_coefficient_matrix, const real_t* d_overlap_matrix, const real_t* d_eri, const ROHF_ParameterSet ROH_parameters, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, real_t* d_fock_matrix, const int num_closed, const int num_open, const int num_basis);

real_t computeEnergy_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix, const int num_basis);
real_t computeEnergy_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_a, const real_t* d_fock_matrix_b, const int num_basis);
real_t computeEnergy_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_fock_matrix_closed, const real_t* d_fock_matrix_open, const int num_basis);

real_t computeOptimalDampingFactor_RHF(const real_t* d_fock_matrix, const real_t* d_prev_fock_matrix, const real_t* d_density_matrix, const real_t* d_prev_density_matrix, const int num_basis);
real_t computeOptimalDampingFactor_ROHF(const real_t* d_fock_matrix, const real_t* d_prev_fock_matrix, const real_t* d_density_matrix, const real_t* d_prev_density_matrix, const int num_basis);
void damping(real_t* d_matrix_new, real_t* d_matrix_old, const real_t alpha, int num_basis);

void computeDIISErrorMatrix(const real_t* d_overlap_matrix, const real_t* d_transform_matrix, const real_t* d_fock_matrix, const real_t* d_density_matrix, real_t* d_diis_error_matrix, const int num_basis, const bool is_include_transform = false);
void computeFockMatrixDIIS(real_t* d_error_matrices, real_t* d_fock_matrices, real_t* d_new_fock_matrix, const int num_prev, const int num_basis);

void computeInitialCoefficientMatrix_GWH(const real_t* d_core_hamiltonian_matrix, const real_t* d_overlap_matrix, const real_t* d_transform_matrix, real_t* d_coefficient_matrix, const int num_basis);

void compute_RI_IntermediateMatrixB(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors, real_t* d_intermediate_matrix_B, const size_t2* d_primitive_shell_pair_indices, const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const int num_auxiliary_basis, const real_t* d_boys_grid, const bool verbose);
void computeIntermediateMatrixB(const real_t* d_three_center_eri, const real_t* d_two_center_eri, real_t* d_intermediate_matrix_B, const int num_basis, const int num_auxiliary_basis);
//void computeFockMatrix_RI_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix, const int num_basis, const int num_auxiliary_basis, real_t* d_J, real_t* d_K, real_t* d_W, real_t* d_T, real_t* d_V);
void computeFockMatrix_RI_RHF_with_density_matrix(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix, const int num_basis, const int num_auxiliary_basis, real_t* d_J, real_t* d_K, real_t* d_W, real_t* d_T, real_t* d_V);
void computeFockMatrix_RI_RHF_with_coefficient_matrix(const real_t* d_coefficient_matrix, const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix, const int num_basis, const int num_auxiliary_basis, const int num_occ, real_t* d_J, real_t* d_K, real_t* d_W, real_t* d_X, real_t* d_X_packed);
void computeFockMatrix_RI_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis, const int num_auxiliary_basis);
void computeFockMatrix_RI_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_coefficient_matrix, const real_t* d_overlap_matrix, const real_t* d_intermediate_matrix_B, const ROHF_ParameterSet ROH_parameters, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, real_t* d_fock_matrix, const int num_closed, const int num_open, const int num_basis, const int num_auxiliary_basis);
//void computeFockMatrix_Direct_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const int2* d_primitive_shell_pair_indices, const real_t* d_cgto_normalization_factors, const real_t* d_boys_grid, const real_t* d_schwarz_upper_bound_factors, const real_t  schwarz_screening_threshold, real_t* d_fock_matrix, const int num_basis, std::vector<int*>& d_global_counters, std::vector<int*>& d_min_skipped_columns, real_t* d_fock_matrix_replicas, const int num_fock_replicas, const int verbose);
void computeFockMatrix_Direct_RHF(const real_t* d_density_matrix, real_t* d_density_matrix_diff, real_t* d_density_matrix_diff_shell, const real_t* d_core_hamiltonian_matrix, const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const int2* d_primitive_shell_pair_indices, const real_t* d_cgto_nomalization_factors, const real_t* d_boys_grid, const real_t* d_schwarz_upper_bound_factors, const real_t  schwarz_screening_threshold, real_t* d_fock_matrix, real_t* d_fock_matrix_prev, const int num_basis, std::vector<int*>& d_global_counters, std::vector<int*>& d_min_skipped_columns, real_t* d_fock_matrix_replicas, const int num_fock_replicas, const int verbose, bool& is_first_call);


std::vector<double> computeSkeletonHessian_RHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms, const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const int num_atoms, const int num_basis, const int num_electron, const bool verbose);

std::vector<double> computeEnergyGradient_RHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms, const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const int num_atoms, const int num_basis, const int num_electron, const bool verbose);
std::vector<double> computeEnergyGradient_UHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms, const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_coefficient_matrix_a, const real_t* d_coefficient_matrix_b, const real_t* d_orbital_energies_a, const real_t* d_orbital_energies_b, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const int num_atoms, const int num_basis, const int num_alpha, const int num_beta, const bool verbose);
// void computeEnergyGradient_ROHF(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const Atom* d_atoms, const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const int num_atoms, const int num_basis, const int num_electron, const bool verbose);


size_t makeShellPairTypeInfo(const std::vector<ShellTypeInfo>& shell_type_infos, std::vector<ShellPairTypeInfo>& shell_pair_type_infos);
void computeSchwarzUpperBounds(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_upper_bound_factors, const bool verbose);
void computeAuxiliarySchwarzUpperBounds(const std::vector<ShellTypeInfo>& shell_aux_type_infos, const PrimitiveShell* d_primitive_shells_aux, const real_t* d_boys_grid, const real_t* d_cgto_aux_normalization_factors, real_t* d_upper_bound_factors_aux, const bool verbose);


void computeMullikenPopulation_RHF(const real_t* d_density_matrix, const real_t* d_overlap_matrix, real_t* mulliken_population_basis, const int num_basis);
void computeMullikenPopulation_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* overlap_matrix, real_t* mulliken_population_basis, const int num_basis);

void computeDensityOverlapMatrix(const real_t* d_density_matrix, const real_t* overlap_matrix, real_t* result_matrix, const int num_basis); // for Mayer bond order

void computeSqrtOverlapDensitySqrtOverlapMatrix(const real_t* d_density_matrix, const real_t* overlap_matrix, real_t* result_matrix, const int num_basis); // for Wiberg bond order (RHF, ROHF)


void constructERIHash(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, const real_t* d_schwarz_upper_bound_factors, real_t schwarz_screening_threshold, unsigned long long* d_hash_keys, real_t* d_hash_values, size_t hash_capacity_mask, const int num_basis, const bool verbose);
void computeFockMatrix_Hash_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries, real_t* d_fock_matrix, const int num_basis, const int verbose);
void computeFockMatrix_Hash_Indexed_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const unsigned long long* d_hash_keys, const real_t* d_hash_values, const size_t* d_nonzero_indices, size_t num_nonzero, real_t* d_fock_matrix, const int num_basis, const int verbose);
void computeFockMatrix_Hash_FullScan_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const unsigned long long* d_hash_keys, const real_t* d_hash_values, size_t hash_capacity, real_t* d_fock_matrix, const int num_basis, const int verbose);


void computeFockMatrix_RI_Direct_RHF(const real_t* d_density_matrix, const real_t* d_coefficient_matrix, const real_t* d_L_inv,  real_t* d_decomposed_two_center_eris, const real_t* d_core_hamiltonian_matrix,  real_t* d_fock_matrix,  real_t* d_coefficient_matrix_prev, real_t* h_Z_tensor_prev, const std::vector<ShellTypeInfo>& shell_type_infos,  const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,  const PrimitiveShell* h_primitive_shells,  const PrimitiveShell* d_primitive_shells,  const real_t* d_cgto_normalization_factors,  const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,  const PrimitiveShell* d_auxiliary_primitive_shells,  const real_t* d_auxiliary_cgto_normalization_factors,  const size_t2* d_primitive_shell_pair_indices, const int num_basis, const int num_auxiliary_basis, const int num_electrons, const int num_primitive_shells, const real_t* d_boys_grid, const double schwarz_screening_threshold,  const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors,const bool verbose);
void computeInverseByDtrsm(real_t* two_center_eris, real_t* two_center_eris_inverse, int num_auxiliary_basis);

/// Semi-Direct RI Fock build: computes 3-center ERIs ONCE per iteration into temporary B, then BLAS-only J/K.
void computeFockMatrix_RI_Direct_v2(
    const real_t* d_density_matrix, const real_t* d_coefficient_matrix,
    const real_t* d_two_center_eris_cholesky,  // L (Cholesky factor, row-major)
    const real_t* d_L_inv,                     // L^{-1} (precomputed, row-major)
    const real_t* d_core_hamiltonian_matrix, real_t* d_fock_matrix,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors,
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors,
    const size_t2* d_primitive_shell_pair_indices,
    int num_basis, int num_auxiliary_basis, int num_occ,
    const real_t* d_boys_grid,
    double schwarz_screening_threshold,
    const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors,
    bool verbose);


void computeInitialFockMatrix_RI_Direct_RHF(const real_t* d_density_matrix, const real_t* d_C, const real_t* d_L_inv,  const real_t* d_core_hamiltonian_matrix,  real_t* d_fock_matrix,  const std::vector<ShellTypeInfo>& shell_type_infos,  const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,  const PrimitiveShell* h_primitive_shells,  const PrimitiveShell* d_primitive_shells,  const real_t* d_cgto_normalization_factors,  const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,  const PrimitiveShell* d_auxiliary_primitive_shells,  const real_t* d_auxiliary_cgto_normalization_factors,  const size_t2* d_primitive_shell_pair_indices, size_t2* h_primitive_shell_pair_indices_for_SAD_K_computation, const size_t2* d_primitive_shell_pair_indices_for_SAD_K_computation, const int num_basis, const int num_auxiliary_basis, const int num_electrons, const int num_primitive_shells, const real_t* d_boys_grid, const double schwarz_screening_threshold,  const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors, const bool verbose, real_t* d_decomposed_two_center_eris);

void computeSchwarzUpperBounds_for_SAD_K_computation(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_upper_bound_factors, ShellPairSorter* d_upper_bound_factors_for_SAD_K_computation, const size_t num_primitive_shells, const bool verbose);




/**
 * @brief class for managing cuSOLVER.
 * @details This class provides methods for eigenvalue decomposition using cuSOLVER.
 * @details This class has a handle a parameter for cuSOLVER that are created in the constructor and destroyed in the destructor.
 */
class cusolverManager {
public:
    const cusolverDnHandle_t cusolverHandle; ///< cuSOLVER handle for eigenvalue decomposition
    const cusolverDnParams_t cusolverParams; ///< cuSOLVER parameters

    cusolverManager()
        : cusolverHandle(createCusolverHandle()),
          cusolverParams(createCusolverDnParams()){}

    ~cusolverManager() {
        cusolverDnDestroy(cusolverHandle);
        cusolverDnDestroyParams(cusolverParams);
    }
    
private:
    /**
     * @brief Helper function to create and initialize a cuSOLVER handle.
     * @return A valid cuSOLVER handle.
     */
     static cusolverDnHandle_t createCusolverHandle() {
//        PROFILE_FUNCTION(Initialize_cuSOLVER);

        cusolverDnHandle_t handle;
        if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS) {
            THROW_EXCEPTION("Failed to initialize cuSOLVER handle");
        }
        return handle;
    }

    /**
     * @brief Helper function to create and initialize a cuSOLVER handle.
     * @return A valid cuSOLVER handle.
     */
    static cusolverDnParams_t createCusolverDnParams() {
//        PROFILE_FUNCTION(Initialize_cuSOLVER);

        cusolverDnParams_t params;
        if (cusolverDnCreateParams(&params) != CUSOLVER_STATUS_SUCCESS) {
            THROW_EXCEPTION("Failed to initialize cuSOLVER parameters");
        }
        return params;
    }
};

/**
 * @brief Class for cuBLAS.
 * @details This class provides methods for matrix operations using cuBLAS.
 * @details This class has a handle that is created in the constructor and destroyed in the destructor.
 */
class cublasManager {
public:
    const cublasHandle_t cublasHandle; ///< cuBLAS handle for matrix operations


    cublasManager(): cublasHandle(createcublasHandle()) {}

    ~cublasManager() {
        cublasDestroy(cublasHandle);
    }

private:
    /**
     * @brief Helper function to create and initialize a cuBLAS handle.
     * @return A valid cuBLAS handle.
     */
     static cublasHandle_t createcublasHandle() {
//        PROFILE_FUNCTION(Initialize_cuBLAS);

        cublasHandle_t handle;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            THROW_EXCEPTION("Failed to initialize cuBLAS handle");
        }
        return handle;
    }
};



/**
 * @brief Per-thread GPU handle manager for cuBLAS and cuSOLVER.
 * @details Each thread has its own cuBLAS handle, cuSOLVER handle, and cuSOLVER advanced parameters.
 * @note Thread-safe (thread-local). Handles are destroyed when the thread exits.
 */
class GPUHandle {
public:
    /**
     * @brief Get thread-local cuBLAS handle.
     */
    static cublasHandle_t cublas() {
        return instance().cublas_;
    }

    /**
     * @brief Get thread-local cuSOLVER handle.
     */
    static cusolverDnHandle_t cusolver() {
        return instance().cusolver_;
    }

    /**
     * @brief Get thread-local cuSOLVER advanced parameters.
     */
    static cusolverDnParams_t cusolverParams() {
        return instance().cusolver_params_;
    }

private:
    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    cusolverDnParams_t cusolver_params_ = nullptr;

    GPUHandle() {
        // Create cuBLAS handle
        if (cublasCreate(&cublas_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }

        // Create cuSOLVER handle
        if (cusolverDnCreate(&cusolver_) != CUSOLVER_STATUS_SUCCESS) {
            cublasDestroy(cublas_);
            throw std::runtime_error("Failed to create cuSOLVER handle");
        }

        // Create cuSOLVER params
        if (cusolverDnCreateParams(&cusolver_params_) != CUSOLVER_STATUS_SUCCESS) {
            cusolverDnDestroy(cusolver_);
            cublasDestroy(cublas_);
            throw std::runtime_error("Failed to create cuSOLVER params");
        }

        // Link params to handle
    //    if (cusolverDnSetAdvOptions(cusolver_, cusolver_params_) != CUSOLVER_STATUS_SUCCESS) {
    //        cusolverDnDestroyParams(cusolver_params_);
    //        cusolverDnDestroy(cusolver_);
    //        cublasDestroy(cublas_);
    //        throw std::runtime_error("Failed to set cuSOLVER advanced options");
    //    }
    }

    ~GPUHandle() {
        if (cusolver_params_) cusolverDnDestroyParams(cusolver_params_);
        if (cusolver_) cusolverDnDestroy(cusolver_);
        if (cublas_) cublasDestroy(cublas_);
    }

    // Disable copy and assignment
    GPUHandle(const GPUHandle&) = delete;
    GPUHandle& operator=(const GPUHandle&) = delete;

    // Get thread-local instance
    static GPUHandle& instance() {
        thread_local GPUHandle instance;
        return instance;
    }
};


} // namespace gansu::gpu
