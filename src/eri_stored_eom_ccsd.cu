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
 * @file eri_stored_eom_ccsd.cu
 * @brief EOM-CCSD excited state calculation using stored ERIs
 *
 * Workflow:
 *   1. Solve CCSD ground state (reuse existing ccsd_spatial_orbital)
 *      - Modified to return T1/T2 amplitudes on device
 *   2. Transform AO ERIs to MO ERIs
 *   3. Build EOM-CCSD operator using CCSD amplitudes
 *   4. Solve eigenvalue problem using full Davidson solver
 *      (M22 is NOT diagonal → no Schur complement)
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rhf.hpp"
#include "eom_ccsd_operator.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include "profiler.hpp"
#include "oscillator_strength.hpp"

namespace gansu {

// Forward declarations
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);

real_t ccsd_spatial_orbital(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed = nullptr);


static void compute_eom_ccsd_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_basis = rhf.get_num_basis();
    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const int singles_dim = num_occ * num_vir;
    const int doubles_dim = num_occ * num_occ * num_vir * num_vir;
    const int total_dim = singles_dim + doubles_dim;

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    std::cout << "\n---- EOM-CCSD excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", total=" << total_dim
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Requested " << n_states << " states but singles dimension is "
                  << singles_dim << ". Reducing to " << singles_dim << "." << std::endl;
        n_states = singles_dim;
    }

    // Step 1: Solve CCSD ground state and extract T1/T2
    Timer ccsd_timer;
    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    real_t E_CCSD = ccsd_spatial_orbital(d_eri_ao, d_C, d_eps, num_basis, num_occ,
                                          false, nullptr, &d_t1, &d_t2, d_eri_mo_precomputed);

    std::cout << "  CCSD correlation energy: " << std::fixed << std::setprecision(10)
              << E_CCSD << " Ha" << std::endl;
    std::cout << "  CCSD solver time: " << std::fixed << std::setprecision(3)
              << ccsd_timer.elapsed_seconds() << " s" << std::endl;

    // Store CCSD correlation energy
    rhf.set_post_hf_energy(E_CCSD);

    // Step 2: Get MO ERIs (use precomputed if available, otherwise transform)
    Timer mo_timer;
    real_t* d_eri_mo;
    bool free_eri_mo;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        d_eri_mo = nullptr;
        tracked_cudaMalloc(&d_eri_mo,
                           (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
        free_eri_mo = true;
    }

    std::cout << "  MO transform time: " << std::fixed << std::setprecision(3)
              << mo_timer.elapsed_seconds() << " s" << std::endl;

    // Step 3: Build EOM-CCSD operator (takes ownership of T1, T2)
    Timer build_timer;
    EOMCCSDOperator eom_ccsd_op(d_eri_mo, d_eps,
                                 d_t1, d_t2,
                                 num_occ, num_vir, num_basis);

    // Free full MO ERIs (blocks have been extracted)
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s" << std::endl;

    // Step 4: Solve eigenvalue problem using full Davidson solver
    Timer solve_timer;
    std::cout << "  Solving EOM-CCSD with full Davidson (dim=" << total_dim << ")..." << std::endl;

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.convergence_threshold = 1e-6;
    // Use large subspace to avoid restart instability in non-symmetric dgeev.
    // For non-Hermitian EOM problems, restarts can cause eigenvalue collapse.
    config.max_subspace_size = std::min(total_dim, std::max(500, 40 * n_states));
    config.max_iterations = 500;
    config.use_preconditioner = true;
    config.symmetric = false;
    config.min_eigenvalue = 0.01;  // Filter out spurious near-zero eigenvalues
    config.verbose = 2;

    DavidsonSolver solver(eom_ccsd_op, config);
    solver.solve();

    const auto& eigenvalues = solver.get_eigenvalues();

    // Filter out spurious near-zero eigenvalues (ground state in EOM)
    std::vector<real_t> h_full_eigenvectors((size_t)n_states * total_dim);
    solver.copy_eigenvectors_to_host(h_full_eigenvectors.data());

    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_eigenvectors;
    for (int k = 0; k < n_states; k++) {
        if (eigenvalues[k] < 0.01) continue;
        excitation_energies.push_back(eigenvalues[k]);
        h_eigenvectors.insert(h_eigenvectors.end(),
                              &h_full_eigenvectors[k * total_dim],
                              &h_full_eigenvectors[k * total_dim + singles_dim]);
    }
    n_states = static_cast<int>(excitation_energies.size());

    std::cout << "  EOM-CCSD solve time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

    rhf.set_excitation_energies(excitation_energies);

    // Step 5: Print results with oscillator strengths
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "EOM-CCSD",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        excitation_energies, h_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);
}

void ERI_Stored_RHF::compute_eom_ccsd(int n_states) {
    compute_eom_ccsd_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

void ERI_RI_RHF::compute_eom_ccsd(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_eom_ccsd_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

void ERI_Direct_RHF::compute_eom_ccsd(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_eom_ccsd_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

void ERI_Hash_RHF::compute_eom_ccsd(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_eom_ccsd_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

} // namespace gansu
