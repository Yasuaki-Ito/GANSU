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
 * @file eri_stored_adc2x.cu
 * @brief ADC(2)-x excited state calculation using stored ERIs
 *
 * ADC(2)-x = ADC(2)-s with first-order off-diagonal M22 terms.
 * Reuses ADC2Operator for M11/M12/M21, adds oooo/vvvv/voov M22 corrections.
 * Uses full Davidson solver in the singles+doubles space.
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rhf.hpp"
#include "adc2_operator.hpp"
#include "adc2x_operator.hpp"
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


static void compute_adc2x_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_frozen = rhf.get_num_frozen_core();
    const int num_basis = rhf.get_num_basis();
    const int full_occ = rhf.get_num_electrons() / 2;
    const int num_occ = full_occ - num_frozen;
    const int num_vir = num_basis - full_occ;
    const int singles_dim = num_occ * num_vir;
    const int doubles_dim = num_occ * num_occ * num_vir * num_vir;
    const int total_dim = singles_dim + doubles_dim;

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();

    bool is_triplet = rhf.is_triplet();
    std::string spin_label = is_triplet ? "triplet" : "singlet";

    std::cout << "\n---- ADC(2)-x " << spin_label << " excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", total_dim=" << total_dim
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Requested " << n_states << " states but singles dimension is "
                  << singles_dim << ". Reducing to " << singles_dim << "." << std::endl;
        n_states = singles_dim;
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
    // Step 2: Build ADC(2)-s operator (for M11, M12, M21, D2)
    //         Then build ADC(2)-x operator (adds M22 first-order terms)
    // ------------------------------------------------------------------
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    ADC2Operator adc2_op(d_eri_mo, d_orbital_energies, num_occ, num_vir, num_basis, is_triplet, num_frozen, full_occ);

    // Build ADC(2)-x operator (extracts oooo/vvvv/oovv blocks, needs d_eri_mo)
    ADC2XOperator adc2x_op(adc2_op, d_eri_mo, num_basis);

    // Free full MO ERIs — all blocks are extracted
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    // ------------------------------------------------------------------
    // Step 3: Davidson solver
    // ------------------------------------------------------------------
    Timer adc2x_timer;

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.convergence_threshold = 1e-6;
    config.max_subspace_size = std::min(total_dim, std::max(100, 10 * n_states));
    config.max_iterations = 500;
    config.use_preconditioner = true;
    config.symmetric = false;  // ADC full matrix is non-symmetric in spatial orbitals
    config.verbose = 2;

    std::cout << "  Solving with full Davidson (dim=" << total_dim << ")..." << std::endl;

    DavidsonSolver solver(adc2x_op, config);
    bool converged = solver.solve();

    if (!converged) {
        std::cout << "  Warning: Davidson did not fully converge" << std::endl;
    }

    std::cout << "  ADC(2)-x time: " << std::fixed << std::setprecision(3)
              << adc2x_timer.elapsed_seconds() << " s" << std::endl;

    // ------------------------------------------------------------------
    // Step 4: Extract results
    // ------------------------------------------------------------------
    const auto& eigenvalues = solver.get_eigenvalues();
    std::vector<real_t> h_full_evecs((size_t)n_states * total_dim);
    solver.copy_eigenvectors_to_host(h_full_evecs.data());

    // Filter out spurious near-zero eigenvalues
    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_final_eigenvectors;

    for (int k = 0; k < n_states; k++) {
        if (eigenvalues[k] < 0.01) continue;
        excitation_energies.push_back(eigenvalues[k]);
        h_final_eigenvectors.insert(h_final_eigenvectors.end(),
                                    &h_full_evecs[k * total_dim],
                                    &h_full_evecs[k * total_dim + singles_dim]);
    }
    n_states = static_cast<int>(excitation_energies.size());

    // Store excitation energies
    rhf.set_excitation_energies(excitation_energies);

    // ------------------------------------------------------------------
    // Step 5: Print results with oscillator strengths
    // ------------------------------------------------------------------
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "ADC(2)-x (triplet)" : "ADC(2)-x";
    auto es_result = compute_excited_state_properties(
        method_name,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        excitation_energies, h_final_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir, num_frozen, full_occ);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);
}

void ERI_Stored_RHF::compute_adc2x(int n_states) {
    compute_adc2x_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

void ERI_RI_RHF::compute_adc2x(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_adc2x_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

void ERI_Direct_RHF::compute_adc2x(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_adc2x_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

void ERI_Hash_RHF::compute_adc2x(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_adc2x_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

} // namespace gansu
