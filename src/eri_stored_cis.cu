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
 * @file eri_stored_cis.cu
 * @brief CIS (Configuration Interaction Singles) excited state calculation
 *
 * Computes singlet or triplet excited states using the CIS method with RHF reference.
 * Uses stored AO ERIs transformed to MO basis, builds the CIS A-matrix,
 * and solves the eigenvalue problem with Davidson solver.
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>

#include "rhf.hpp"
#include "cis_operator.hpp"
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


static void compute_cis_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_basis = rhf.get_num_basis();
    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const int cis_dim = num_occ * num_vir;

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();

    bool is_triplet = rhf.is_triplet();
    std::string spin_label = is_triplet ? "triplet" : "singlet";

    std::cout << "\n---- CIS " << spin_label << " excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", dim=" << cis_dim
              << ", nstates=" << n_states << std::endl;

    if (n_states > cis_dim) {
        std::cout << "Warning: Requested " << n_states << " states but CIS dimension is "
                  << cis_dim << ". Reducing to " << cis_dim << "." << std::endl;
        n_states = cis_dim;
    }

    // ------------------------------------------------------------------
    // Step 1: Transform AO ERIs to MO ERIs
    // ------------------------------------------------------------------
    real_t* d_eri_mo;
    bool free_eri_mo;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        tracked_cudaMalloc(&d_eri_mo,
                           (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
        free_eri_mo = true;
    }

    // ------------------------------------------------------------------
    // Step 2: Get orbital energies
    // ------------------------------------------------------------------
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    // ------------------------------------------------------------------
    // Step 3: Build CIS operator and solve
    // ------------------------------------------------------------------
    CISOperator cis_op(d_eri_mo, d_orbital_energies, num_occ, num_vir, num_basis, is_triplet);

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.max_subspace_size = std::min(cis_dim, std::max(30, 4 * n_states));
    config.convergence_threshold = 1e-6;
    config.max_iterations = 100;
    config.use_preconditioner = true;
    config.verbose = 2;

    DavidsonSolver solver(cis_op, config);
    bool converged = solver.solve();

    if (!converged) {
        std::cout << "Warning: Davidson solver did not converge for all states." << std::endl;
    }

    const auto& eigenvalues = solver.get_eigenvalues();

    // Store excitation energies for external access (e.g., tests)
    rhf.set_excitation_energies(eigenvalues);

    // ------------------------------------------------------------------
    // Step 4: Analyze and print results with oscillator strengths
    // ------------------------------------------------------------------
    std::vector<real_t> h_eigenvectors(cis_dim * n_states);
    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    // Get host data for oscillator strength computation
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "CIS (triplet)" : "CIS";
    auto es_result = compute_excited_state_properties(
        method_name,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);

    // Cleanup
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
}

void ERI_Stored_RHF::compute_cis(int n_states) {
    compute_cis_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

void ERI_RI_RHF::compute_cis(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_cis_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

} // namespace gansu
