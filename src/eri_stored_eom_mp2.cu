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

/**
 * @file eri_stored_eom_mp2.cu
 * @brief EOM-MP2 excited state calculation using stored ERIs
 *
 * Supports four solver modes (controlled by --eom_mp2_solver parameter):
 *
 *   auto (default): Automatically selects full or schur_omega based on
 *     available GPU memory (80% threshold).
 *
 *   full: Full Davidson in singles+doubles space, exact but may fail
 *     to converge due to ω=0 null space of the EOM-MP2 matrix.
 *
 *   schur_static: Approximate Schur complement in singles space with ω=0.
 *     M22 off-diagonal (t2×r2 coupling) is ignored, M22 ≈ D2 (diagonal).
 *     M_eff = M11 - M12 · D2⁻¹ · M21. Approximate but avoids null space.
 *
 *   schur_omega: ω-dependent Schur complement with self-consistent iteration.
 *     M_eff(ω) = M11 + M12 · (ωI - D2)⁻¹ · M21. Per-root ω iteration
 *     until convergence. More accurate than schur_static.
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rhf.hpp"
#include "eom_mp2_operator.hpp"
#include "eom_mp2_schur_operator.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include "profiler.hpp"
#include "oscillator_strength.hpp"

namespace gansu {

// Forward declaration from eri_stored.cu
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);


// ========================================================================
//  Solver mode: full — Davidson in full singles+doubles space
// ========================================================================

static void solve_eom_mp2_full(
    EOMMP2Operator& eom_mp2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_eigenvectors)
{
    int total_dim = eom_mp2_op.dimension();

    std::cout << "  Solving with full Davidson (dim=" << total_dim << ")..." << std::endl;

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.convergence_threshold = 1e-6;
    config.max_subspace_size = std::min(total_dim, std::max(200, 20 * n_states));
    config.max_iterations = 500;
    config.use_preconditioner = true;
    config.symmetric = false;
    config.min_eigenvalue = 0.01;
    config.verbose = 2;

    DavidsonSolver solver(eom_mp2_op, config);
    bool converged = solver.solve();

    if (!converged) {
        std::cout << "  Warning: Davidson did not fully converge" << std::endl;
    }

    const auto& eigenvalues = solver.get_eigenvalues();
    excitation_energies.resize(n_states);
    h_eigenvectors.resize((size_t)n_states * singles_dim);

    std::vector<real_t> h_full_evecs((size_t)n_states * total_dim);
    solver.copy_eigenvectors_to_host(h_full_evecs.data());

    for (int k = 0; k < n_states; k++) {
        excitation_energies[k] = eigenvalues[k];
        std::copy(&h_full_evecs[k * total_dim],
                  &h_full_evecs[k * total_dim + singles_dim],
                  &h_eigenvectors[k * singles_dim]);
    }
}


// ========================================================================
//  Solver mode: schur_static — approximate Schur complement (ω=0)
// ========================================================================

static void solve_eom_mp2_schur_static(
    EOMMP2Operator& eom_mp2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_eigenvectors)
{
    std::cout << "  Solving with approximate Schur complement (dim=" << singles_dim << ")..." << std::endl;
    std::cout << "  (M22 off-diagonal ignored, omega=0 approximation)" << std::endl;

    EOMMP2SchurOperator schur_op(eom_mp2_op);

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.convergence_threshold = 1e-6;
    config.max_subspace_size = std::min(singles_dim, std::max(100, 10 * n_states));
    config.max_iterations = 500;
    config.use_preconditioner = true;
    config.symmetric = false;  // EOM-MP2 Schur complement is not guaranteed symmetric
    config.verbose = 2;

    DavidsonSolver solver(schur_op, config);
    bool converged = solver.solve();

    if (!converged) {
        std::cout << "  Warning: Davidson did not fully converge" << std::endl;
    }

    const auto& eigenvalues = solver.get_eigenvalues();
    excitation_energies.resize(n_states);
    h_eigenvectors.resize((size_t)n_states * singles_dim);

    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    for (int k = 0; k < n_states; k++) {
        excitation_energies[k] = eigenvalues[k];
    }
}


// ========================================================================
//  Helper: build dense matrix from LinearOperator (column-by-column)
// ========================================================================

static void build_dense_from_operator(LinearOperator& op, real_t* d_dense, int dim) {
    real_t* d_unit = nullptr;
    real_t* d_col = nullptr;
    tracked_cudaMalloc(&d_unit, (size_t)dim * sizeof(real_t));
    tracked_cudaMalloc(&d_col, (size_t)dim * sizeof(real_t));

    for (int j = 0; j < dim; j++) {
        cudaMemset(d_unit, 0, (size_t)dim * sizeof(real_t));
        real_t one = 1.0;
        cudaMemcpy(d_unit + j, &one, sizeof(real_t), cudaMemcpyHostToDevice);

        op.apply(d_unit, d_col);

        // Column j of dense matrix (column-major)
        cudaMemcpy(d_dense + (size_t)j * dim, d_col,
                   (size_t)dim * sizeof(real_t), cudaMemcpyDeviceToDevice);
    }

    tracked_cudaFree(d_unit);
    tracked_cudaFree(d_col);
}


// ========================================================================
//  Solver mode: schur_omega — ω-dependent Schur complement iteration
// ========================================================================

static void solve_eom_mp2_schur_omega(
    EOMMP2Operator& eom_mp2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_eigenvectors)
{
    const double omega_threshold = 1e-8;
    const int max_omega_iter = 30;

    EOMMP2SchurOperator schur_op(eom_mp2_op);

    real_t* d_M_eff = nullptr;
    real_t* d_eigenvalues = nullptr;
    real_t* d_eigenvectors = nullptr;
    tracked_cudaMalloc(&d_M_eff, (size_t)singles_dim * singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvalues, (size_t)singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvectors, (size_t)singles_dim * singles_dim * sizeof(real_t));

    // Phase 1: Initial solve with ω=0
    std::cout << "  Solving with omega-dependent Schur complement (dim=" << singles_dim << ")..." << std::endl;
    schur_op.set_omega(0.0);
    build_dense_from_operator(schur_op, d_M_eff, singles_dim);
    gpu::eigenDecompositionNonSymmetric(d_M_eff, d_eigenvalues, d_eigenvectors, singles_dim);

    std::vector<real_t> h_eigenvalues(singles_dim);
    cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
               singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_all_eigenvectors((size_t)singles_dim * singles_dim);

    excitation_energies.resize(n_states);
    h_eigenvectors.resize((size_t)n_states * singles_dim);

    // Phase 2: Per-root ω iteration
    for (int k = 0; k < n_states; k++) {
        real_t omega = 0.0;
        bool converged = false;

        for (int iter = 0; iter < max_omega_iter; iter++) {
            schur_op.set_omega(omega);
            build_dense_from_operator(schur_op, d_M_eff, singles_dim);
            gpu::eigenDecompositionNonSymmetric(d_M_eff, d_eigenvalues, d_eigenvectors, singles_dim);

            cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
                       singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

            real_t omega_new = h_eigenvalues[k];
            real_t delta = std::abs(omega_new - omega);

            std::cout << "  Root " << k + 1 << " iter " << std::setw(2) << iter + 1
                      << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                      << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                      << std::defaultfloat << std::endl;

            if (delta < omega_threshold) {
                excitation_energies[k] = omega_new;
                cudaMemcpy(h_all_eigenvectors.data(), d_eigenvectors,
                           (size_t)singles_dim * singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);
                std::copy(&h_all_eigenvectors[k * singles_dim],
                          &h_all_eigenvectors[(k + 1) * singles_dim],
                          &h_eigenvectors[k * singles_dim]);
                std::cout << "  Root " << k + 1 << ": converged in " << iter + 1
                          << " iterations" << std::endl;
                converged = true;
                break;
            }

            omega = omega_new;
        }

        if (!converged) {
            excitation_energies[k] = h_eigenvalues[k];
            std::cout << "  Root " << k + 1 << ": NOT converged after " << max_omega_iter
                      << " iterations" << std::endl;
        }
    }

    tracked_cudaFree(d_M_eff);
    tracked_cudaFree(d_eigenvalues);
    tracked_cudaFree(d_eigenvectors);
}


// ========================================================================
//  Main entry point: compute_eom_mp2
// ========================================================================

void ERI_Stored_RHF::compute_eom_mp2(int n_states) {
    PROFILE_FUNCTION();

    const int num_basis = rhf_.get_num_basis();
    const int num_occ = rhf_.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const int singles_dim = num_occ * num_vir;
    const int doubles_dim = num_occ * num_occ * num_vir * num_vir;

    std::string solver_mode = rhf_.get_eom_mp2_solver();

    // "schur" is an alias for "schur_static" (backward compatibility)
    if (solver_mode == "schur") {
        solver_mode = "schur_static";
    }

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eri_ao = eri_matrix_.device_ptr();

    // Auto solver selection: use full Davidson if GPU memory is sufficient, else schur_omega
    if (solver_mode == "auto") {
        int total_dim = singles_dim + doubles_dim;
        int max_sub = std::min(total_dim, std::max(100, 10 * n_states));
        size_t davidson_bytes = (
            static_cast<size_t>(total_dim) * max_sub * 2 +   // subspace + sigma vectors
            static_cast<size_t>(max_sub) * max_sub * 2 +     // subspace matrix + eigvecs
            static_cast<size_t>(total_dim) * n_states * 2 +  // residuals + eigenvectors
            max_sub
        ) * sizeof(real_t);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        if (davidson_bytes < free_mem * 0.8) {  // 80% threshold for safety margin
            solver_mode = "full";
            std::cout << "  Auto solver: full Davidson ("
                      << CudaMemoryManager<real_t>::format_bytes(davidson_bytes)
                      << " needed, "
                      << CudaMemoryManager<real_t>::format_bytes(free_mem)
                      << " available)" << std::endl;
        } else {
            solver_mode = "schur_omega";
            std::cout << "  Auto solver: schur_omega (full Davidson would need "
                      << CudaMemoryManager<real_t>::format_bytes(davidson_bytes)
                      << ", only "
                      << CudaMemoryManager<real_t>::format_bytes(free_mem)
                      << " available)" << std::endl;
        }
    }

    std::cout << "\n---- EOM-MP2 excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", solver=" << solver_mode
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Requested " << n_states << " states but singles dimension is "
                  << singles_dim << ". Reducing to " << singles_dim << "." << std::endl;
        n_states = singles_dim;
    }

    // Step 1: Transform AO ERIs to MO ERIs
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_mo,
                       (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
    transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);

    // Step 2: Build EOM-MP2 operator
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    EOMMP2Operator eom_mp2_op(d_eri_mo, d_orbital_energies, num_occ, num_vir, num_basis);

    // Free full MO ERIs — blocks are already extracted
    tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    // Step 3: Solve (dispatch based on solver mode)
    Timer solve_timer;

    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_eigenvectors;

    if (solver_mode == "schur_static") {
        solve_eom_mp2_schur_static(eom_mp2_op, n_states, singles_dim,
                                   excitation_energies, h_eigenvectors);
    } else if (solver_mode == "full") {
        solve_eom_mp2_full(eom_mp2_op, n_states, singles_dim,
                           excitation_energies, h_eigenvectors);
    } else {
        // Default: schur_omega
        if (solver_mode != "schur_omega") {
            std::cout << "  Warning: Unknown eom_mp2_solver '" << solver_mode
                      << "', using schur_omega" << std::endl;
        }
        solve_eom_mp2_schur_omega(eom_mp2_op, n_states, singles_dim,
                                  excitation_energies, h_eigenvectors);
    }

    std::cout << "  EOM-MP2 time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

    rhf_.set_excitation_energies(excitation_energies);

    // Step 4: Print results with oscillator strengths
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "EOM-MP2",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        excitation_energies, h_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);
}

} // namespace gansu
