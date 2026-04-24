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

#include <Eigen/Dense>
#include "rhf.hpp"
#include "cis_operator.hpp"
#include "cis_operator_ri.hpp"
#include "cis_operator_jk.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "progress.hpp"
#include "utils.hpp"
#include "profiler.hpp"
#include "oscillator_strength.hpp"

namespace gansu {

// Forward declaration from eri_stored.cu
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);


static void compute_cis_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_frozen = rhf.get_num_frozen_core();
    const int num_basis = rhf.get_num_basis();
    const int full_occ = rhf.get_num_electrons() / 2;
    const int num_occ = full_occ - num_frozen;  // active occupied
    const int num_vir = num_basis - full_occ;    // virtual (unchanged)
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
    CISOperator cis_op(d_eri_mo, d_orbital_energies, num_occ, num_vir, num_basis, is_triplet, num_frozen, full_occ);

    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_eigenvectors;

    // Try direct diagonalization if the dense CIS matrix fits in GPU memory
    size_t dense_bytes = (size_t)cis_dim * cis_dim * sizeof(real_t);
    size_t eigen_bytes = (size_t)cis_dim * sizeof(real_t);
    size_t total_needed = dense_bytes * 2 + eigen_bytes;  // matrix + eigenvectors + eigenvalues
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (false && total_needed < free_mem * 0.8) {  // TODO: fix direct diag path
        // Direct diagonalization (symmetric eigendecomposition)
        std::cout << "  Solver: direct diagonalization (dense "
                  << CudaMemoryManager<real_t>::format_bytes(dense_bytes) << ")" << std::endl;

        report_progress("excited", 2, 0, nullptr);  // solver start

        real_t* d_dense = nullptr;
        real_t* d_eigenvalues = nullptr;
        real_t* d_eigenvectors_all = nullptr;
        tracked_cudaMalloc(&d_dense, dense_bytes);
        tracked_cudaMalloc(&d_eigenvalues, eigen_bytes);
        tracked_cudaMalloc(&d_eigenvectors_all, dense_bytes);

        // Build dense CIS matrix column by column
        real_t* d_unit = nullptr;
        real_t* d_col = nullptr;
        tracked_cudaMalloc(&d_unit, (size_t)cis_dim * sizeof(real_t));
        tracked_cudaMalloc(&d_col, (size_t)cis_dim * sizeof(real_t));
        for (int j = 0; j < cis_dim; j++) {
            cudaMemset(d_unit, 0, (size_t)cis_dim * sizeof(real_t));
            real_t one = 1.0;
            cudaMemcpy(d_unit + j, &one, sizeof(real_t), cudaMemcpyHostToDevice);
            cis_op.apply(d_unit, d_col);
            cudaMemcpy(d_dense + (size_t)j * cis_dim, d_col,
                       (size_t)cis_dim * sizeof(real_t), cudaMemcpyDeviceToDevice);
        }
        tracked_cudaFree(d_unit);
        tracked_cudaFree(d_col);

        // Symmetric eigendecomposition
        gpu::eigenDecomposition(d_dense, d_eigenvalues, d_eigenvectors_all, cis_dim);

        // Copy results to host
        std::vector<real_t> h_all_eigenvalues(cis_dim);
        cudaMemcpy(h_all_eigenvalues.data(), d_eigenvalues,
                   cis_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

        std::vector<real_t> h_all_eigvecs((size_t)cis_dim * cis_dim);
        cudaMemcpy(h_all_eigvecs.data(), d_eigenvectors_all,
                   (size_t)cis_dim * cis_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

        excitation_energies.resize(n_states);
        h_eigenvectors.resize((size_t)n_states * cis_dim);
        for (int k = 0; k < n_states; k++) {
            excitation_energies[k] = h_all_eigenvalues[k];
            std::copy(&h_all_eigvecs[k * cis_dim],
                      &h_all_eigvecs[(k + 1) * cis_dim],
                      &h_eigenvectors[k * cis_dim]);
        }

        tracked_cudaFree(d_dense);
        tracked_cudaFree(d_eigenvalues);
        tracked_cudaFree(d_eigenvectors_all);
    } else {
        // Fall back to Davidson for large systems
        std::cout << "  Solver: Davidson (dense matrix too large: "
                  << CudaMemoryManager<real_t>::format_bytes(total_needed)
                  << " needed, " << CudaMemoryManager<real_t>::format_bytes(free_mem)
                  << " available)" << std::endl;

        report_progress("excited", 2, 0, nullptr);

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
        excitation_energies.assign(eigenvalues.begin(), eigenvalues.end());
        h_eigenvectors.resize((size_t)n_states * cis_dim);
        solver.copy_eigenvectors_to_host(h_eigenvectors.data());
    }

    // Store excitation energies for external access
    rhf.set_excitation_energies(excitation_energies);

    // ------------------------------------------------------------------
    // Step 4: Analyze and print results with oscillator strengths
    // ------------------------------------------------------------------
    // Get host data for oscillator strength computation
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "CIS (triplet)" : "CIS";
    try {
        auto es_result = compute_excited_state_properties(
            method_name,
            prim_shells.host_ptr(), prim_shells.size(),
            cgto_norms.host_ptr(),
            rhf.get_shell_type_infos(),
            coefficient_matrix.host_ptr(),
            excitation_energies, h_eigenvectors.data(),
            n_states, num_basis, num_occ, num_vir,
            num_frozen, full_occ);
        rhf.set_oscillator_strengths(es_result.oscillator_strengths);
        rhf.set_excited_state_report(es_result.report);
    } catch (const std::exception& e) {
        std::cerr << "[CIS] compute_excited_state_properties FAILED: " << e.what() << std::endl;
        // Still set excitation energies even if oscillator strengths fail
    }

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

    // CPU fallback: CISOperator_RI uses cuBLAS-only kernels with no CPU
    // backend, so on CPU we route through the base CISOperator path by
    // building the full MO ERI tensor via build_mo_eri (which on CPU
    // reconstructs AO ERI from B and applies the O(N^5) quarter transform).
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        compute_cis_impl(rhf_, nullptr, n_states, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return;
    }

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
    if (!gpu::gpu_available()) {
        // CPU fallback: two-step MO transformation using Eigen
        using Eigen::Map;
        using Eigen::MatrixXd;
        const real_t alpha = 1.0, beta = 0.0;

        // Step 1: B_tmp(Q*nao+μ, q) = Σ_ν B(Q*nao+μ, ν) × C(ν, q)
        // = (naux*nao, nao) × (nao, nao)
        Map<const MatrixXd> B_mat(d_B, nao, naux * nao);   // col-major view
        Map<const MatrixXd> C_mat(d_C, nao, nao);           // col-major view
        Map<MatrixXd> Tmp_mat(d_B_tmp, nao, naux * nao);
        // col-major: result(nao, naux*nao) = C_cm × B_cm = C_rm^T × B_rm^T (as col-major)
        Tmp_mat.noalias() = C_mat * B_mat;

        // Step 2: B_mo_Q(p,q) = C^T × B_tmp_Q for each Q
        // In col-major: B_mo_cm = B_tmp_cm × C_cm^T  per Q-slice
        for (int Q = 0; Q < naux; Q++) {
            Map<const MatrixXd> Bq(d_B_tmp + (size_t)Q * nao * nao, nao, nao);
            Map<MatrixXd> Mq(d_B_mo + (size_t)Q * nao * nao, nao, nao);
            Mq.noalias() = Bq * C_mat.transpose();
        }
    } else {
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

// Forward declarations (defined in half_transform_mp3.cu)
void half_transform_steps234(
    const real_t* d_half, const real_t* d_C, real_t* d_result,
    real_t* d_Ki, real_t* d_Li,
    int nao, int bs, int p_start, int n_p, int q_start, int n_q, int r_start, int n_r);
__global__ void scatter_s_block_kernel(
    const double* __restrict__ src, double* __restrict__ dst,
    int n_p, int n_q, int n_r, int n_s, int s_blk, int bs);

/**
 * Build OVOV and OOVV sub-blocks via Direct (on-the-fly) half-transform.
 */
static void build_ovov_oovv_direct(
    RHF& rhf, const HF& hf,
    real_t* d_ovov, real_t* d_oovv, int block_s)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const size_t nao3 = (size_t)nao * nao * nao;
    const int max_dim = std::max(nocc, nvir);

    // Schwarz factors
    DeviceHostMemory<real_t> schwarz_unsorted(hf.get_num_primitive_shell_pairs());
    gpu::computeSchwarzUpperBounds(
        hf.get_shell_type_infos(), hf.get_shell_pair_type_infos(),
        hf.get_primitive_shells().device_ptr(),
        hf.get_boys_grid().device_ptr(),
        hf.get_cgto_normalization_factors().device_ptr(),
        schwarz_unsorted.device_ptr(), false);

    const auto& shell_type_infos = hf.get_shell_type_infos();
    const auto& shell_pair_type_infos = hf.get_shell_pair_type_infos();
    const auto* d_primitive_shells = hf.get_primitive_shells().device_ptr();
    const auto* d_boys_grid = hf.get_boys_grid().device_ptr();
    const auto* d_cgto_norm = hf.get_cgto_normalization_factors().device_ptr();
    const auto* d_schwarz = schwarz_unsorted.device_ptr();
    const real_t schwarz_thresh = hf.get_schwarz_screening_threshold();

    // Workspace
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;
    real_t* d_tmp_block = nullptr;
    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)max_dim * nao * block_s * sizeof(real_t));
    const size_t max_block = std::max((size_t)nocc * nvir * nocc, (size_t)nocc * nocc * nvir) * block_s;
    tracked_cudaMalloc(&d_tmp_block, max_block * sizeof(real_t));

    constexpr int scatter_threads = 256;

    auto build_and_scatter = [&](real_t* d_dst, int n_p, int n_q, int n_r, int n_s,
                                  int s_blk, int bs,
                                  int p_start, int q_start, int r_start) {
        half_transform_steps234(d_half, d_C, d_tmp_block,
            d_Ki, d_Li, nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);
        const size_t block_total = (size_t)n_p * n_q * n_r * bs;
        scatter_s_block_kernel<<<(block_total + scatter_threads - 1) / scatter_threads, scatter_threads>>>(
            d_tmp_block, d_dst, n_p, n_q, n_r, n_s, s_blk, bs);
    };

    for (int s_blk = 0; s_blk < nvir; s_blk += block_s) {
        int bs = std::min(block_s, nvir - s_blk);

        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        gpu::computeHalfTransformedERI(
            shell_type_infos, shell_pair_type_infos,
            d_primitive_shells, d_boys_grid, d_cgto_norm,
            d_half, d_schwarz, schwarz_thresh,
            nao, d_C, nocc + s_blk, bs);

        build_and_scatter(d_ovov, nocc, nvir, nocc, nvir, s_blk, bs, 0, nocc, 0);
        build_and_scatter(d_oovv, nocc, nocc, nvir, nvir, s_blk, bs, 0, 0, nocc);
    }

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_tmp_block);
}


/**
 * Helper: run CIS Davidson with OVOV/OOVV sub-blocks.
 * Shared by Direct and Hash CIS.
 */
static void run_cis_with_subblocks(
    RHF& rhf, const real_t* d_ovov, const real_t* d_oovv,
    int n_states, const std::string& method_label)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int cis_dim = nocc * nvir;
    const bool is_triplet = rhf.is_triplet();
    const real_t* d_eps = rhf.get_orbital_energies().device_ptr();

    CISOperator_HalfTransform cis_op(d_ovov, d_oovv, d_eps, nocc, nvir, is_triplet);

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
    rhf.set_excitation_energies(eigenvalues);

    std::vector<real_t> h_eigenvectors(cis_dim * n_states);
    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    rhf.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        method_label,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        rhf.get_coefficient_matrix().host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, nao, nocc, nvir);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);
}


void ERI_Direct_RHF::compute_cis(int n_states) {
    PROFILE_FUNCTION();

    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int cis_dim = nocc * nvir;
    const bool is_triplet = rhf_.is_triplet();
    const size_t num_ovov = (size_t)nocc * nvir * nocc * nvir;
    const size_t num_oovv = (size_t)nocc * nocc * nvir * nvir;

    // CPU fallback: half-transform path (computeHalfTransformedERI) is GPU-only,
    // so reconstruct the AO ERI on CPU and route through the stored-CIS impl.
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        compute_cis_impl(rhf_, nullptr, n_states, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return;
    }

    std::string spin_label = is_triplet ? "triplet" : "singlet";
    std::cout << "\n---- CIS (half-transform, Direct) " << spin_label << " ----"
              << " nocc=" << nocc << ", nvir=" << nvir
              << ", dim=" << cis_dim << ", nstates=" << n_states << std::endl;

    if (n_states > cis_dim) {
        std::cout << "Warning: Reducing to " << cis_dim << " states." << std::endl;
        n_states = cis_dim;
    }

    real_t* d_ovov = nullptr;
    real_t* d_oovv = nullptr;
    tracked_cudaMalloc(&d_ovov, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_oovv, num_oovv * sizeof(real_t));
    cudaMemset(d_ovov, 0, num_ovov * sizeof(real_t));
    cudaMemset(d_oovv, 0, num_oovv * sizeof(real_t));

    const size_t nao3 = (size_t)nao * nao * nao;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
    block_s = std::min(block_s, nao);
    if (block_s > 8) block_s = 8;

    build_ovov_oovv_direct(rhf_, hf_, d_ovov, d_oovv, block_s);

    std::string label = is_triplet ? "CIS-HT-Direct (triplet)" : "CIS-HT-Direct";
    run_cis_with_subblocks(rhf_, d_ovov, d_oovv, n_states, label);

    tracked_cudaFree(d_ovov);
    tracked_cudaFree(d_oovv);
}

// Forward declaration: build OVOV + OOVV via Hash half-transform
// (defined in half_transform_mp3.cu)
real_t mp3_half_transform_hash(
    RHF& rhf,
    const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    const size_t* d_nonzero_indices, size_t num_nonzero,
    size_t hash_capacity_mask, HashFockMethod method, int block_s);

// Forward declarations for half-transform kernels (from half_transform_mp3.cu)
__global__ void hash_half_transform_compact_kernel(
    const unsigned long long* g_coo_keys, const double* g_coo_values, size_t num_entries,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);
__global__ void hash_half_transform_indexed_kernel(
    const unsigned long long* g_hash_keys, const double* g_hash_values,
    const size_t* g_nonzero_indices, size_t num_nonzero,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);
__global__ void hash_half_transform_fullscan_kernel(
    const unsigned long long* g_hash_keys, const double* g_hash_values, size_t hash_capacity,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);



/**
 * Build OVOV and OOVV sub-blocks via Hash half-transform.
 * Used for CIS (and potentially other methods).
 */
static void build_ovov_oovv_hash(
    RHF& rhf,
    const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    const size_t* d_nonzero_indices, size_t num_nonzero,
    size_t hash_capacity_mask, HashFockMethod method,
    real_t* d_ovov, real_t* d_oovv, int block_s)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const size_t nao3 = (size_t)nao * nao * nao;
    const int max_dim = std::max(nocc, nvir);

    // Workspace
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;
    real_t* d_tmp_block = nullptr;
    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)max_dim * nao * block_s * sizeof(real_t));
    const size_t max_block = std::max((size_t)nocc * nvir * nocc, (size_t)nocc * nocc * nvir) * block_s;
    tracked_cudaMalloc(&d_tmp_block, max_block * sizeof(real_t));

    constexpr int scatter_threads = 256;

    auto build_and_scatter = [&](real_t* d_dst, int n_p, int n_q, int n_r, int n_s,
                                  int s_blk, int bs,
                                  int p_start, int q_start, int r_start) {
        half_transform_steps234(d_half, d_C, d_tmp_block,
            d_Ki, d_Li, nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);
        const size_t block_total = (size_t)n_p * n_q * n_r * bs;
        scatter_s_block_kernel<<<(block_total + scatter_threads - 1) / scatter_threads, scatter_threads>>>(
            d_tmp_block, d_dst, n_p, n_q, n_r, n_s, s_blk, bs);
    };

    // Build OVOV and OOVV (same as streaming MP3 Pass 0)
    for (int s_blk = 0; s_blk < nvir; s_blk += block_s) {
        int bs = std::min(block_s, nvir - s_blk);

        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        {
            const int threads = 256;
            if (method == HashFockMethod::Compact) {
                const int blocks = ((int)num_entries + threads - 1) / threads;
                hash_half_transform_compact_kernel<<<blocks, threads>>>(
                    d_coo_keys, d_coo_values, num_entries,
                    d_C, d_half, nao, nocc + s_blk, bs);
            } else if (method == HashFockMethod::Indexed) {
                const int blocks = ((int)num_nonzero + threads - 1) / threads;
                hash_half_transform_indexed_kernel<<<blocks, threads>>>(
                    d_hash_keys, d_hash_values,
                    d_nonzero_indices, num_nonzero,
                    d_C, d_half, nao, nocc + s_blk, bs);
            } else {
                const size_t capacity = hash_capacity_mask + 1;
                const int blocks = ((int)capacity + threads - 1) / threads;
                hash_half_transform_fullscan_kernel<<<blocks, threads>>>(
                    d_hash_keys, d_hash_values, capacity,
                    d_C, d_half, nao, nocc + s_blk, bs);
            }
            cudaDeviceSynchronize();
        }

        build_and_scatter(d_ovov, nocc, nvir, nocc, nvir, s_blk, bs, 0, nocc, 0);
        build_and_scatter(d_oovv, nocc, nocc, nvir, nvir, s_blk, bs, 0, 0, nocc);
    }

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_tmp_block);
}


void ERI_Hash_RHF::compute_cis(int n_states) {
    PROFILE_FUNCTION();

    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int cis_dim = nocc * nvir;
    const bool is_triplet = rhf_.is_triplet();
    const size_t num_ovov = (size_t)nocc * nvir * nocc * nvir;
    const size_t num_oovv = (size_t)nocc * nocc * nvir * nvir;

    // CPU fallback: half-transform kernels are GPU-only.  Use cached
    // hash AO ERI tensor (set up by ERI_Hash::precomputation on CPU)
    // via build_mo_eri + stored-CIS impl.
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        compute_cis_impl(rhf_, nullptr, n_states, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return;
    }

    std::string spin_label = is_triplet ? "triplet" : "singlet";
    std::cout << "\n---- CIS (half-transform, Hash) " << spin_label << " ----"
              << " nocc=" << nocc << ", nvir=" << nvir
              << ", dim=" << cis_dim << ", nstates=" << n_states << std::endl;

    if (n_states > cis_dim) {
        std::cout << "Warning: Reducing to " << cis_dim << " states." << std::endl;
        n_states = cis_dim;
    }

    // Build OVOV and OOVV via half-transform
    real_t* d_ovov = nullptr;
    real_t* d_oovv = nullptr;
    tracked_cudaMalloc(&d_ovov, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_oovv, num_oovv * sizeof(real_t));
    cudaMemset(d_ovov, 0, num_ovov * sizeof(real_t));
    cudaMemset(d_oovv, 0, num_oovv * sizeof(real_t));

    // block_s heuristic
    const size_t nao3 = (size_t)nao * nao * nao;
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
    block_s = std::min(block_s, nao);
    if (block_s > 8) block_s = 8;

    build_ovov_oovv_hash(rhf_,
        d_coo_keys_, d_coo_values_, num_entries_,
        d_hash_keys_, d_hash_values_,
        d_nonzero_indices_, num_nonzero_,
        hash_capacity_mask_, hash_fock_method_,
        d_ovov, d_oovv, block_s);

    std::string label = is_triplet ? "CIS-HT-Hash (triplet)" : "CIS-HT-Hash";
    run_cis_with_subblocks(rhf_, d_ovov, d_oovv, n_states, label);

    tracked_cudaFree(d_ovov);
    tracked_cudaFree(d_oovv);
}

} // namespace gansu
