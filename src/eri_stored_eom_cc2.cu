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
 * @file eri_stored_eom_cc2.cu
 * @brief EOM-CC2 excited state calculation using stored ERIs
 *
 * Workflow:
 *   1. Transform AO ERIs to MO ERIs
 *   2. Extract ERI blocks needed by CC2 solver
 *   3. Solve CC2 ground state amplitudes (T1, T2) iteratively
 *   4. Build EOM-CC2 operator using CC2 amplitudes
 *   5. Solve eigenvalue problem (Schur complement in singles space)
 *
 * EOM-CC2 has diagonal M22 (unlike EOM-MP2), so:
 *   - Full Davidson works (no null space)
 *   - Schur complement is EXACT (not approximate)
 *   - Default solver: Schur (operates in smaller singles space)
 */

#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "rhf.hpp"
#include "cc2_solver.hpp"
#include "eom_cc2_operator.hpp"
#include "eom_mp2_schur_operator.hpp"
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

// Reuse EOM-MP2 kernels for ERI block extraction and D1/D2 computation
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int);  // (d_eri_mo, d_out, nocc, nao)
extern __global__ void eom_mp2_compute_D1_kernel(
    const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_extract_fock_kernel(
    const real_t*, real_t*, real_t*, int, int);
extern __global__ void eom_cc2_compute_D2_kernel(
    const real_t*, real_t*, int, int);

// Standard D2 = ε_a + ε_b - ε_i - ε_j for CC2 solver (T2 initialization/update)
// Distinct from M22 diagonal = 3*(ε_b - ε_j) used by Schur complement
__global__ void cc2_standard_D2_kernel(
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_D2,
    int nocc, int nvir) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nocc * nvir * nvir) return;
    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;
    d_D2[idx] = d_orbital_energies[a + nocc] + d_orbital_energies[b + nocc]
              - d_orbital_energies[i] - d_orbital_energies[j];
}


static void compute_eom_cc2_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_basis = rhf.get_num_basis();
    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const int singles_dim = num_occ * num_vir;
    const int doubles_dim = num_occ * num_occ * num_vir * num_vir;

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();

    std::cout << "\n---- EOM-CC2 excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Requested " << n_states << " states but singles dimension is "
                  << singles_dim << ". Reducing to " << singles_dim << "." << std::endl;
        n_states = singles_dim;
    }

    // Step 1: Transform AO ERIs to MO ERIs
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

    // Step 2: Extract ERI blocks for CC2 solver
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    int threads = 256;
    int blocks;

    // OVOV
    size_t ovov_size = (size_t)num_occ * num_vir * num_occ * num_vir;
    real_t* d_eri_ovov = nullptr;
    tracked_cudaMalloc(&d_eri_ovov, ovov_size * sizeof(real_t));
    blocks = (ovov_size + threads - 1) / threads;
    eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov, num_occ, num_vir, num_basis);

    // VVOV
    size_t vvov_size = (size_t)num_vir * num_vir * num_occ * num_vir;
    real_t* d_eri_vvov = nullptr;
    tracked_cudaMalloc(&d_eri_vvov, vvov_size * sizeof(real_t));
    blocks = (vvov_size + threads - 1) / threads;
    eom_mp2_extract_eri_vvov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvov, num_occ, num_vir, num_basis);

    // OOOV
    size_t ooov_size = (size_t)num_occ * num_occ * num_occ * num_vir;
    real_t* d_eri_ooov = nullptr;
    tracked_cudaMalloc(&d_eri_ooov, ooov_size * sizeof(real_t));
    blocks = (ooov_size + threads - 1) / threads;
    eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov, num_occ, num_vir, num_basis);

    // OOVV
    size_t oovv_size = (size_t)num_occ * num_occ * num_vir * num_vir;
    real_t* d_eri_oovv = nullptr;
    tracked_cudaMalloc(&d_eri_oovv, oovv_size * sizeof(real_t));
    blocks = (oovv_size + threads - 1) / threads;
    eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv, num_occ, num_vir, num_basis);

    // OVVO
    size_t ovvo_size = (size_t)num_occ * num_vir * num_vir * num_occ;
    real_t* d_eri_ovvo = nullptr;
    tracked_cudaMalloc(&d_eri_ovvo, ovvo_size * sizeof(real_t));
    blocks = (ovvo_size + threads - 1) / threads;
    eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo, num_occ, num_vir, num_basis);

    // VVVV (for CC2 T2 dressed integrals)
    size_t vvvv_size = (size_t)num_vir * num_vir * num_vir * num_vir;
    real_t* d_eri_vvvv = nullptr;
    tracked_cudaMalloc(&d_eri_vvvv, vvvv_size * sizeof(real_t));
    blocks = (vvvv_size + threads - 1) / threads;
    eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv, num_occ, num_vir, num_basis);

    // OOOO (for CC2 T2 dressed integrals)
    size_t oooo_size = (size_t)num_occ * num_occ * num_occ * num_occ;
    real_t* d_eri_oooo = nullptr;
    tracked_cudaMalloc(&d_eri_oooo, oooo_size * sizeof(real_t));
    blocks = (oooo_size + threads - 1) / threads;
    eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo, num_occ, num_basis);

    // D1, D2, Fock
    real_t* d_D1 = nullptr;
    tracked_cudaMalloc(&d_D1, (size_t)singles_dim * sizeof(real_t));
    blocks = (singles_dim + threads - 1) / threads;
    eom_mp2_compute_D1_kernel<<<blocks, threads>>>(d_orbital_energies, d_D1, num_occ, num_vir);

    real_t* d_D2 = nullptr;
    tracked_cudaMalloc(&d_D2, (size_t)doubles_dim * sizeof(real_t));
    blocks = (doubles_dim + threads - 1) / threads;
    // Standard D2 for CC2 solver (NOT the M22 diagonal used by Schur complement)
    cc2_standard_D2_kernel<<<blocks, threads>>>(d_orbital_energies, d_D2, num_occ, num_vir);

    real_t* d_f_oo = nullptr;
    real_t* d_f_vv = nullptr;
    tracked_cudaMalloc(&d_f_oo, (size_t)num_occ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv, (size_t)num_vir * sizeof(real_t));
    blocks = (num_basis + threads - 1) / threads;
    eom_mp2_extract_fock_kernel<<<blocks, threads>>>(d_orbital_energies, d_f_oo, d_f_vv, num_occ, num_vir);

    cudaDeviceSynchronize();

    // Step 3: Solve CC2 ground state amplitudes
    Timer cc2_timer;
    CC2Result cc2 = solve_cc2(
        d_eri_ovov, d_eri_vvov, d_eri_ooov, d_eri_oovv, d_eri_ovvo,
        d_eri_vvvv, d_eri_oooo,
        d_f_oo, d_f_vv, d_D1, d_D2,
        num_occ, num_vir);

    std::cout << "  CC2 solver time: " << std::fixed << std::setprecision(3)
              << cc2_timer.elapsed_seconds() << " s" << std::endl;

    // Store CC2 correlation energy
    rhf.set_post_hf_energy(cc2.cc2_energy);

    // Free CC2 solver ERI blocks (EOM-CC2 operator will extract its own)
    tracked_cudaFree(d_eri_ovov);
    tracked_cudaFree(d_eri_vvov);
    tracked_cudaFree(d_eri_ooov);
    tracked_cudaFree(d_eri_oovv);
    tracked_cudaFree(d_eri_ovvo);
    tracked_cudaFree(d_eri_vvvv);
    tracked_cudaFree(d_eri_oooo);
    tracked_cudaFree(d_D1);
    tracked_cudaFree(d_D2);
    tracked_cudaFree(d_f_oo);
    tracked_cudaFree(d_f_vv);

    // Step 4: Build EOM-CC2 operator (takes ownership of T1, T2)
    EOMCC2Operator eom_cc2_op(d_eri_mo, d_orbital_energies,
                              cc2.d_t1, cc2.d_t2,
                              num_occ, num_vir, num_basis);

    // Free full MO ERIs
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    // Step 5: Solve EOM-CC2 eigenvalue problem
    // M22 is exactly diagonal → Schur complement is EXACT at each ω.
    // M_eff(ω) = M11 + M12 · (ωI - M22)⁻¹ · M21
    Timer solve_timer;

    std::string solver_mode = rhf.get_eom_cc2_solver();

    // Auto solver selection: use schur_omega (exact & efficient) unless full is feasible and small
    if (solver_mode == "auto") {
        int total_dim = singles_dim + doubles_dim;
        int max_sub = std::min(total_dim, std::max(200, 20 * n_states));
        size_t davidson_bytes = (
            static_cast<size_t>(total_dim) * max_sub * 2 +
            static_cast<size_t>(max_sub) * max_sub * 2 +
            static_cast<size_t>(total_dim) * n_states * 2 +
            max_sub
        ) * sizeof(real_t);

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        if (davidson_bytes < free_mem * 0.8) {
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

    std::cout << "\n  EOM-CC2 solver=" << solver_mode
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", nstates=" << n_states << std::endl;

    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_eigenvectors;

    if (solver_mode == "full") {
        // ---- Full Davidson in singles+doubles space ----
        // M22 is diagonal so no null space issues, but uses more memory
        int total_dim = eom_cc2_op.dimension();
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

        DavidsonSolver solver(eom_cc2_op, config);
        bool converged = solver.solve();

        if (!converged) {
            std::cout << "  Warning: Davidson did not fully converge" << std::endl;
        }

        const auto& eigenvalues = solver.get_eigenvalues();

        // Filter out spurious near-zero eigenvalues (ground state in EOM)
        std::vector<real_t> h_full_evecs((size_t)n_states * total_dim);
        solver.copy_eigenvectors_to_host(h_full_evecs.data());

        for (int k = 0; k < n_states; k++) {
            if (eigenvalues[k] < 0.01) continue;
            excitation_energies.push_back(eigenvalues[k]);
            h_eigenvectors.insert(h_eigenvectors.end(),
                                  &h_full_evecs[k * total_dim],
                                  &h_full_evecs[k * total_dim + singles_dim]);
        }
        n_states = static_cast<int>(excitation_energies.size());

    } else if (solver_mode == "schur_static") {
        // ---- Schur complement with ω=0 (approximate but fast) ----
        // M22 is exactly diagonal → only ω=0 approximation, no M22 diagonal approximation
        std::cout << "  Solving with Schur complement (ω=0, dim=" << singles_dim << ")..." << std::endl;

        EOMMP2SchurOperator schur_op(eom_cc2_op);
        schur_op.set_omega(0.0);

        DavidsonConfig config;
        config.num_eigenvalues = n_states;
        config.convergence_threshold = 1e-6;
        config.max_subspace_size = std::min(singles_dim, std::max(100, 10 * n_states));
        config.max_iterations = 500;
        config.use_preconditioner = true;
        config.symmetric = false;
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

    } else {
        // ---- schur_omega (default): ω-dependent Schur complement iteration ----
        // M22 is exactly diagonal → Schur complement is EXACT at each ω.
        // Self-consistent iteration: solve with ω=0, then iterate ω = eigenvalue until convergence.
        if (solver_mode != "schur_omega") {
            std::cout << "  Warning: Unknown eom_cc2_solver '" << solver_mode
                      << "', using schur_omega" << std::endl;
        }

        std::cout << "  Solving with frequency-dependent Schur complement (dim="
                  << singles_dim << ")..." << std::endl;

        EOMMP2SchurOperator schur_op(eom_cc2_op);

        DavidsonConfig config;
        config.num_eigenvalues = n_states;
        config.convergence_threshold = 1e-6;
        config.max_subspace_size = std::min(singles_dim, std::max(100, 10 * n_states));
        config.max_iterations = 500;
        config.use_preconditioner = true;
        config.symmetric = false;
        config.verbose = 2;

        const int max_omega_iter = 20;
        const real_t omega_tol = 1e-6;

        excitation_energies.resize(n_states, 0.0);
        h_eigenvectors.resize((size_t)n_states * singles_dim);

        // Iteration 0: ω = 0
        schur_op.set_omega(0.0);
        {
            DavidsonSolver solver(schur_op, config);
            solver.solve();
            const auto& eigenvalues = solver.get_eigenvalues();
            solver.copy_eigenvectors_to_host(h_eigenvectors.data());
            for (int k = 0; k < n_states; k++) {
                excitation_energies[k] = eigenvalues[k];
            }
        }

        std::cout << "  ω-iteration  0: ω=0.000000";
        for (int k = 0; k < std::min(n_states, 5); k++) {
            std::cout << std::fixed << std::setprecision(6)
                      << "  E" << k+1 << "=" << excitation_energies[k];
        }
        std::cout << std::endl;

        // Subsequent iterations: use lowest eigenvalue as ω
        for (int omega_iter = 1; omega_iter <= max_omega_iter; omega_iter++) {
            std::vector<real_t> prev_energies = excitation_energies;

            real_t omega = prev_energies[0];
            schur_op.set_omega(omega);

            DavidsonConfig iter_config = config;
            iter_config.verbose = 0;

            DavidsonSolver solver(schur_op, iter_config);
            solver.solve();
            const auto& eigenvalues = solver.get_eigenvalues();
            solver.copy_eigenvectors_to_host(h_eigenvectors.data());
            for (int k = 0; k < n_states; k++) {
                excitation_energies[k] = eigenvalues[k];
            }

            real_t max_change = 0.0;
            for (int k = 0; k < n_states; k++) {
                max_change = std::max(max_change,
                                      std::abs(excitation_energies[k] - prev_energies[k]));
            }

            std::cout << "  ω-iteration " << std::setw(2) << omega_iter
                      << ": ω=" << std::fixed << std::setprecision(6) << omega;
            for (int k = 0; k < std::min(n_states, 5); k++) {
                std::cout << "  E" << k+1 << "=" << excitation_energies[k];
            }
            std::cout << "  Δmax=" << std::scientific << std::setprecision(2)
                      << max_change << std::endl;

            if (max_change < omega_tol) {
                std::cout << "  ω-iteration converged after " << omega_iter
                          << " iterations" << std::endl;
                break;
            }

            if (omega_iter == max_omega_iter) {
                std::cout << "  Warning: ω-iteration did not converge (Δmax="
                          << std::scientific << max_change << ")" << std::endl;
            }
        }
    }

    std::cout << "  EOM-CC2 time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

    rhf.set_excitation_energies(excitation_energies);

    // Step 6: Print results with oscillator strengths
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "EOM-CC2",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        excitation_energies, h_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);
}

void ERI_Stored_RHF::compute_eom_cc2(int n_states) {
    compute_eom_cc2_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

void ERI_RI_RHF::compute_eom_cc2(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_eom_cc2_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

void ERI_Direct_RHF::compute_eom_cc2(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_eom_cc2_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

} // namespace gansu
