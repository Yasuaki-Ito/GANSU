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
#include "cis_operator_ri.hpp"
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
    PROFILE_FUNCTION();

    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int naux = num_auxiliary_basis_;
    const int cis_dim = nocc * nvir;
    const bool is_triplet = rhf_.is_triplet();

    std::cout << "\n---- CIS (RI, B-matrix based) "
              << (is_triplet ? "triplet" : "singlet") << " ----"
              << " nocc=" << nocc << ", nvir=" << nvir << ", naux=" << naux
              << ", dim=" << cis_dim << ", nstates=" << n_states << std::endl;

    if (n_states > cis_dim) {
        std::cout << "Warning: Reducing to " << cis_dim << " states." << std::endl;
        n_states = cis_dim;
    }

    // Step 1: Build B_ov(Q,ia), B_oo(Q,ij), B_vv(Q,ab) from intermediate_matrix_B_
    const real_t* d_B = intermediate_matrix_B_.device_ptr();  // (naux, nao*nao)
    const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();

    // Transform B(Q,μν) → MO blocks via DGEMM
    // B_full_mo(Q,pq) would be naux×nao², but we only need ov, oo, vv blocks.
    // Strategy: transform both indices to get B_mo(Q,p,q), then extract blocks.
    // For memory efficiency, transform to B_mo(Q,nao,nao) temporarily.
    real_t* d_B_mo = nullptr;   // (naux, nao, nao)
    real_t* d_B_tmp = nullptr;  // workspace
    tracked_cudaMalloc(&d_B_mo, (size_t)naux * nao * nao * sizeof(real_t));
    tracked_cudaMalloc(&d_B_tmp, (size_t)naux * nao * nao * sizeof(real_t));

    // B_tmp(Q,μ,q) = Σ_ν B(Q,μ,ν) C(ν,q)
    // B(Q,μ,ν) stored as (naux, nao*nao), each Q-slice is (nao, nao) row-major
    // For all Q at once: reshape as (naux*nao, nao) × C(nao, nao) → (naux*nao, nao)
    {
        const real_t alpha = 1.0, beta = 0.0;
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        // cuBLAS col-major: B_cm(nao, naux*nao) × C_cm(nao, nao)
        // = (naux*nao, nao)_rm × C_rm(nao, nao) → result (naux*nao, nao)_rm
        // cublasDgemm(N, N, nao, naux*nao, nao, ...)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    nao, naux * nao, nao, &alpha,
                    d_C, nao,
                    d_B, nao,
                    &beta, d_B_tmp, nao);
        // B_tmp: cuBLAS col-major (nao, naux*nao) = row-major (naux*nao, nao)
        // B_tmp[Q*nao+μ, q] = Σ_ν C(q,ν) B(Q,μ,ν) — but this is C^T × B, not B × C
        // Actually in col-major: C_cm × B_cm where B_cm is col-major (nao, naux*nao).
        // Row-major B(naux*nao, nao) viewed as col-major = (nao, naux*nao).
        // C_cm(nao, nao) = C_rm^T.
        // Result(nao, naux*nao) = C_rm^T × (naux*nao slices of B_rm^T each (nao, nao))
        // This gives: result_cm(q, Q*nao+μ) = Σ_ν C_rm(ν,q) × B_rm(Q*nao+μ, ν) = Σ_ν C(ν,q) B(Q,μ,ν)
        // row-major result(Q*nao+μ, q) = Σ_ν C(ν,q) B(Q,μ,ν) = B_tmp(Q,μ,q) ✓

        // B_mo(Q,p,q) = Σ_μ C(μ,p) × B_tmp(Q,μ,q)
        // Reshape: C^T(p,μ) × B_tmp(Q-blocks of (μ,q))
        // All Q: C^T(nao, nao) × B_tmp(naux*nao, nao) — but C^T operates on μ dimension.
        // Need to sum over μ: B_mo(Q,p,q) = Σ_μ C(μ,p) B_tmp(Q,μ,q)
        // B_tmp viewed as (naux, nao, nao) — for each Q, (nao, nao) matrix.
        // B_mo(Q,p,q) = C^T × B_tmp_Q  for each Q.
        // Batched: B_tmp is (naux*nao, nao) row-major. We need to multiply C^T on the left
        // for each Q-slice independently.
        // Alternative: transpose B_tmp to (naux, q, μ), then multiply.
        // Simplest: B_tmp_reshaped(naux*nao, nao) — stride nao in Q*nao+μ.
        // We want: for each (Q,q), B_mo(Q,:,q) = C^T × B_tmp(Q,:,q)
        // If we group by q: B_mo(:,q) = C^T × B_tmp(:,q) where sizes (naux*nao, 1) per q.
        // Not efficient. Use batched DGEMM or a different approach.

        // Actually: all-at-once DGEMM.
        // B_tmp is (naux, nao, nao) row-major = for each Q: M_Q(μ, q) shape (nao, nao).
        // B_mo_Q(p, q) = Σ_μ C(μ,p) M_Q(μ,q) = C^T × M_Q
        // For all Q at once: stack M_Q as (naux*nao, nao). B_mo stacked as (naux*nao, nao).
        // This is NOT a single DGEMM because C^T operates on the μ index within each Q-block.

        // Use cublasDgemmStridedBatched:
        // A = C^T (nao, nao), stride=0 (same for all Q)
        // B = B_tmp_Q (nao, nao) per Q, stride=nao*nao
        // C = B_mo_Q (nao, nao) per Q, stride=nao*nao
        // cublasDgemmStridedBatched(handle, T, N, nao, nao, nao, alpha,
        //   C, nao, 0, B_tmp, nao, nao*nao, beta, B_mo, nao, nao*nao, naux)
        // In cuBLAS col-major: C_cm = C_rm^T. CUBLAS_OP_T of C_cm = C_rm.
        // B_tmp_cm(nao, nao) per Q = B_tmp_rm^T. CUBLAS_OP_N.
        // Result_cm = C_rm × B_tmp_rm^T → Result_rm = B_tmp_rm × C_rm^T = B_tmp_rm × C^T_rm ≠ what we want.
        // We want B_mo = C^T × B_tmp per Q.
        // col-major: B_mo_cm = B_tmp_cm × C_cm^T = B_tmp_rm^T × C_rm = (C_rm^T × B_tmp_rm)^T
        // This gives B_mo_cm(q,p) = (C^T × B_tmp)^T = B_tmp^T × C.
        // So B_mo_rm(p,q) = B_mo_cm(q,p) = (C^T B_tmp)_pq. ✓
        // cublasDgemmStridedBatched(handle, N, T, nao, nao, nao, alpha,
        //   B_tmp, nao, nao*nao, C, nao, 0, beta, B_mo, nao, nao*nao, naux)
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            nao, nao, nao, &alpha,
            d_B_tmp, nao, (long long)nao * nao,
            d_C, nao, 0,
            &beta,
            d_B_mo, nao, (long long)nao * nao,
            naux);
    }
    tracked_cudaFree(d_B_tmp);

    // Extract B_ov, B_oo, B_vv from B_mo(Q, p, q)
    // B_mo is (naux, nao, nao) row-major. B_mo[Q*nao*nao + p*nao + q]
    // B_ov(Q, ia) where ia = i*nvir + a_rel: B_mo[Q*nao² + i*nao + (a_rel+nocc)]
    // B_oo(Q, ij): B_mo[Q*nao² + i*nao + j]
    // B_vv(Q, ab): B_mo[Q*nao² + (a+nocc)*nao + (b+nocc)]
    // Need to copy into contiguous arrays.
    real_t *d_B_ov = nullptr, *d_B_oo = nullptr, *d_B_vv = nullptr;
    tracked_cudaMalloc(&d_B_ov, (size_t)naux * nocc * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_B_oo, (size_t)naux * nocc * nocc * sizeof(real_t));
    tracked_cudaMalloc(&d_B_vv, (size_t)naux * nvir * nvir * sizeof(real_t));

    // Extract on host (simple, nao is small for RI systems)
    {
        std::vector<real_t> h_B_mo((size_t)naux * nao * nao);
        cudaMemcpy(h_B_mo.data(), d_B_mo, h_B_mo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);

        std::vector<real_t> h_B_ov(naux * nocc * nvir);
        std::vector<real_t> h_B_oo(naux * nocc * nocc);
        std::vector<real_t> h_B_vv(naux * nvir * nvir);

        for (int Q = 0; Q < naux; Q++) {
            for (int i = 0; i < nocc; i++)
                for (int a = 0; a < nvir; a++)
                    h_B_ov[Q * nocc * nvir + i * nvir + a] = h_B_mo[Q * nao * nao + i * nao + (a + nocc)];
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    h_B_oo[Q * nocc * nocc + i * nocc + j] = h_B_mo[Q * nao * nao + i * nao + j];
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++)
                    h_B_vv[Q * nvir * nvir + a * nvir + b] = h_B_mo[Q * nao * nao + (a + nocc) * nao + (b + nocc)];
        }

        cudaMemcpy(d_B_ov, h_B_ov.data(), h_B_ov.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_oo, h_B_oo.data(), h_B_oo.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_vv, h_B_vv.data(), h_B_vv.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    tracked_cudaFree(d_B_mo);

    // Step 2: Build RI CIS operator and solve
    const real_t* d_orbital_energies = rhf_.get_orbital_energies().device_ptr();
    CISOperator_RI cis_op(d_B_ov, d_B_oo, d_B_vv, d_orbital_energies, nocc, nvir, naux, is_triplet);

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.max_subspace_size = std::min(cis_dim, std::max(30, 4 * n_states));
    config.convergence_threshold = 1e-6;
    config.max_iterations = 100;
    config.use_preconditioner = true;
    config.verbose = 2;

    DavidsonSolver solver(cis_op, config);
    bool converged = solver.solve();
    if (!converged)
        std::cout << "Warning: Davidson solver did not converge." << std::endl;

    const auto& eigenvalues = solver.get_eigenvalues();
    rhf_.set_excitation_energies(eigenvalues);

    // Step 3: Oscillator strengths
    std::vector<real_t> h_eigenvectors(cis_dim * n_states);
    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    rhf_.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "CIS-RI (triplet)" : "CIS-RI";
    auto es_result = compute_excited_state_properties(
        method_name,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        rhf_.get_coefficient_matrix().host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, nao, nocc, nvir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);

    // Cleanup
    tracked_cudaFree(d_B_ov);
    tracked_cudaFree(d_B_oo);
    tracked_cudaFree(d_B_vv);
}

void ERI_Direct_RHF::compute_cis(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_cis_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

} // namespace gansu
