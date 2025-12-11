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
void transposeMatrixInPlace(real_t* d_matrix, const int size);
void makeDiagonalMatrix(const real_t* d_vector, real_t* d_matrix, const int size);
real_t computeMatrixTrace(const real_t* d_matrix, const int size);
int eigenDecomposition(const real_t* d_matrix, real_t* d_eigenvalues, real_t* d_eigenvectors, const int size);
void matrixMatrixProduct(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size, const bool transpose_A = false, const bool transpose_B = false, const bool accumulate=false);
void weightedMatrixSum(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const double weight_A, const double weight_B, const int size);
void matrixAddition(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size);
void matrixSubtraction(const double* d_matrix_A, const double* d_matrix_B, double* d_matrix_C, const int size);
double innerProduct(const double* d_vector_A, const double* d_vector_B, double* result, const int size);
void invertMatrix(double* d_A, const int N);
void choleskyDecomposition(double* d_A, const int N);

void computeCoreHamiltonianMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, Atom* d_atoms, PrimitiveShell* d_primitive_shells, real_t* d_boys_grid, real_t* d_cgto_normalization_factors, real_t* d_overlap_matrix, real_t* d_core_hamiltonian_matrix, const int num_atoms, const int num_basis, const bool verbose=false);
void computeCoefficientMatrix(const real_t* d_fock_matrix, const real_t* d_transform_matrix, real_t* d_coefficient_matrix, const int num_basis, real_t* d_orbital_energies=nullptr);

//void computeERIMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_eri_matrix, const real_t schwarz_screening_threshold,  const int num_basis, const bool verbose=false);
void computeERIMatrix(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors,  real_t* d_eri_matrix, const real_t* d_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const bool verbose) ;
void computeTwoCenterERIs(const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_nomalization_factors, real_t* d_two_center_eri, const int num_auxiliary_basis, const real_t* d_boys_grid, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const bool verbose=false);
void computeThreeCenterERIs(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_nomalization_factors, const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_nomalization_factors, real_t* d_three_center_eri, const size_t2* d_primitive_shell_pair_indices, const int num_basis, const int num_auxiliary_basis, const real_t* d_boys_grid, const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const bool verbose=false);


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

void compute_RI_IntermediateMatrixB(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors, const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos, const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_nomalization_factors, real_t* d_intermediate_matrix_B, const size_t2* d_primitive_shell_pair_indices, const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors, const real_t schwarz_screening_threshold, const int num_basis, const int num_auxiliary_basis, const real_t* d_boys_grid, const bool verbose);
void computeIntermediateMatrixB(const real_t* d_three_center_eri, const real_t* d_two_center_eri, real_t* d_intermediate_matrix_B, const int num_basis, const int num_auxiliary_basis);
void computeFockMatrix_RI_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix, const int num_basis, const int num_auxiliary_basis);
void computeFockMatrix_RI_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_intermediate_matrix_B, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, const int num_basis, const int num_auxiliary_basis);
void computeFockMatrix_RI_ROHF(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_coefficient_matrix, const real_t* d_overlap_matrix, const real_t* d_intermediate_matrix_B, const ROHF_ParameterSet ROH_parameters, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, real_t* d_fock_matrix, const int num_closed, const int num_open, const int num_basis, const int num_auxiliary_basis);
void computeFockMatrix_Direct_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const std::vector<ShellTypeInfo>& shell_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_nomalization_factors, const real_t* d_boys_grid, const real_t* d_schwarz_upper_bound_factors, const real_t  schwarz_screening_threshold, real_t* d_fock_matrix, const int num_basis, const int verbose);


size_t makeShellPairTypeInfo(const std::vector<ShellTypeInfo>& shell_type_infos, std::vector<ShellPairTypeInfo>& shell_pair_type_infos);
void computeSchwarzUpperBounds(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, real_t* d_upper_bound_factors, const bool verbose);
void computeAuxiliarySchwarzUpperBounds(const std::vector<ShellTypeInfo>& shell_aux_type_infos, const PrimitiveShell* d_primitive_shells_aux, const real_t* d_boys_grid, const real_t* d_cgto_aux_normalization_factors, real_t* d_upper_bound_factors_aux, const bool verbose);


void computeMullikenPopulation_RHF(const real_t* d_density_matrix, const real_t* d_overlap_matrix, real_t* mulliken_population_basis, const int num_basis);
void computeMullikenPopulation_UHF(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* overlap_matrix, real_t* mulliken_population_basis, const int num_basis);

void computeDensityOverlapMatrix(const real_t* d_density_matrix, const real_t* overlap_matrix, real_t* result_matrix, const int num_basis); // for Mayer bond order

void constructERIHash(const std::vector<ShellTypeInfo>& shell_type_infos, const std::vector<ShellPairTypeInfo>& shell_pair_type_infos, const PrimitiveShell* d_primitive_shells, const real_t* d_boys_grid, const real_t* d_cgto_normalization_factors, /* Hash memoryへのポインタ, */ const bool verbose);
void computeFockMatrix_Hash_RHF(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, /* Hash memoryへのポインタ, */ real_t* d_fock_matrix, const int num_basis, const int verbose);


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