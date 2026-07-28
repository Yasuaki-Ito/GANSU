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
 * @file eri_stored_adc2.cu
 * @brief ADC(2) excited state calculation using stored ERIs
 *
 * Supports three solver modes (controlled by --adc2_solver parameter):
 *
 *   schur_static: Static Schur complement (ω=0), approximate (~0.005-0.02 Ha error)
 *     M_eff(0) = M11 + M12 · diag(1/(0-D2)) · M21
 *     Single diagonalization, fastest but least accurate.
 *
 *   schur_omega: ω-dependent Schur complement (default), exact
 *     M_eff(ω) = M11 + M12 · diag(1/(ω-D2)) · M21
 *     Per-root ω iteration until convergence (5-8 iterations each).
 *
 *   full: Full Davidson in singles+doubles space, exact
 *     Direct iterative diagonalization of the full ADC(2) matrix.
 *     No ω iteration needed. Uses DavidsonSolver + ADC2FullOperator.
 */

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>   // std::getenv (GANSU_ADC2_OMEGA_REFINE)

#include "rhf.hpp"
#include "eri.hpp"
#include "adc2_operator.hpp"
#include "adc2_full_operator.hpp"
#include "davidson_solver.hpp"
#include "progress.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include "profiler.hpp"
#include "oscillator_strength.hpp"
#include "eom_chain_context.hpp"   // STEOMResult + adc2_spatial_orbital declaration

namespace gansu {

// Forward declaration from eri_stored.cu
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);


// ========================================================================
//  Solver mode: schur_static — single M_eff(ω=0) diagonalization
// ========================================================================

static void solve_schur_static(
    ADC2Operator& adc2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_final_eigenvectors)
{
    std::cout << "  Solving with static Schur complement (omega=0)..." << std::endl;

    real_t* d_M_eff = nullptr;
    real_t* d_eigenvalues = nullptr;
    real_t* d_eigenvectors = nullptr;
    tracked_cudaMalloc(&d_M_eff, (size_t)singles_dim * singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvalues, (size_t)singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvectors, (size_t)singles_dim * singles_dim * sizeof(real_t));

    { double v[] = {0.0}; report_progress("schur", 0, 1, v); }  // diagonalization
    adc2_op.build_M_eff_matrix(0.0, d_M_eff);
    // Only the n_states lowest roots are needed — a full syevd of the singles×singles
    // M_eff computes ALL eigenpairs (O(singles³): ~an hour at singles≈30k). The partial
    // solver (syevdx, lowest n_states) is O(singles²·n_states) → minutes.
    gpu::partialEigenDecomposition(d_M_eff, d_eigenvalues, d_eigenvectors, singles_dim, n_states);

    std::vector<real_t> h_eigenvalues(singles_dim);
    cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
               singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_all_eigenvectors((size_t)singles_dim * singles_dim);
    cudaMemcpy(h_all_eigenvectors.data(), d_eigenvectors,
               (size_t)singles_dim * singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

    excitation_energies.resize(n_states);
    h_final_eigenvectors.resize((size_t)n_states * singles_dim);
    for (int k = 0; k < n_states; k++) {
        excitation_energies[k] = h_eigenvalues[k];
        std::copy(&h_all_eigenvectors[k * singles_dim],
                  &h_all_eigenvectors[(k + 1) * singles_dim],
                  &h_final_eigenvectors[k * singles_dim]);
    }
    { double v[] = {(double)n_states}; report_progress("schur", 1, 1, v); }  // done

    tracked_cudaFree(d_M_eff);
    tracked_cudaFree(d_eigenvalues);
    tracked_cudaFree(d_eigenvectors);

    std::cout << "  (approximate: ~0.005-0.02 Ha error vs. exact methods)" << std::endl;
}


// ========================================================================
//  Solver mode: schur_omega — per-root ω iteration
// ========================================================================

static void solve_schur_omega(
    ADC2Operator& adc2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_final_eigenvectors)
{
    const double omega_threshold = 1e-8;
    const int max_omega_iter = 30;

    std::cout << "  Solving with omega-dependent Schur complement..." << std::endl;

    real_t* d_M_eff = nullptr;
    real_t* d_eigenvalues = nullptr;
    real_t* d_eigenvectors = nullptr;
    tracked_cudaMalloc(&d_M_eff, (size_t)singles_dim * singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvalues, (size_t)singles_dim * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvectors, (size_t)singles_dim * singles_dim * sizeof(real_t));

    // Phase 1: Initial solve with ω=0 (= schur_static) for initial guesses
    // Partial eigendecomposition (cusolverDnXsyevdx) — only need n_states lowest.
    adc2_op.build_M_eff_matrix(0.0, d_M_eff);
    gpu::partialEigenDecomposition(d_M_eff, d_eigenvalues, d_eigenvectors, singles_dim, n_states);

    std::vector<real_t> h_eigenvalues(singles_dim);
    cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
               singles_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_all_eigenvectors((size_t)singles_dim * singles_dim);

    excitation_energies.resize(n_states);
    h_final_eigenvectors.resize((size_t)n_states * singles_dim);

    // Phase 2: Per-root ω iteration, starting from schur_static initial guess
    for (int k = 0; k < n_states; k++) {
        real_t omega = h_eigenvalues[k];
        bool converged = false;

        for (int iter = 0; iter < max_omega_iter; iter++) {
            adc2_op.build_M_eff_matrix(omega, d_M_eff);
            gpu::partialEigenDecomposition(d_M_eff, d_eigenvalues, d_eigenvectors, singles_dim, n_states);

            cudaMemcpy(h_eigenvalues.data(), d_eigenvalues,
                       n_states * sizeof(real_t), cudaMemcpyDeviceToHost);

            real_t omega_new = h_eigenvalues[k];
            real_t delta = std::abs(omega_new - omega);

            { double v[] = {(double)(k+1), omega_new, delta};
              report_progress("schur_omega", k * max_omega_iter + iter, 3, v); }

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
                          &h_final_eigenvectors[k * singles_dim]);
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
//  Solver mode: full — Davidson in full singles+doubles space
// ========================================================================

static void solve_full_davidson(
    ADC2Operator& adc2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_final_eigenvectors)
{
    int doubles_dim = adc2_op.get_doubles_dim();
    int total_dim = singles_dim + doubles_dim;

    std::cout << "  Solving with full Davidson (dim=" << total_dim << ")..." << std::endl;

    ADC2FullOperator full_op(adc2_op);

    DavidsonConfig config;
    config.num_eigenvalues = n_states;
    config.convergence_threshold = 1e-6;
    // Full-space dimension can be very large (e.g., 99540 for benzene/STO-3G).
    // Need a large subspace to capture doubles contributions.
    config.max_subspace_size = std::min(total_dim, std::max(100, 10 * n_states));
    config.max_iterations = 500;
    config.use_preconditioner = true;
    config.symmetric = false;  // ADC(2) full matrix is non-symmetric in spatial orbitals (M12 ≠ M21^T)
    config.verbose = 2;

    DavidsonSolver solver(full_op, config);
    bool converged = solver.solve();

    if (!converged) {
        std::cout << "  Warning: Davidson did not fully converge" << std::endl;
    }

    const auto& eigenvalues = solver.get_eigenvalues();
    excitation_energies.resize(n_states);
    h_final_eigenvectors.resize((size_t)n_states * singles_dim);

    // Copy eigenvalues and extract singles part of eigenvectors
    std::vector<real_t> h_full_evecs((size_t)n_states * total_dim);
    solver.copy_eigenvectors_to_host(h_full_evecs.data());

    for (int k = 0; k < n_states; k++) {
        excitation_energies[k] = eigenvalues[k];
        // Extract only the singles part [0..singles_dim) of each eigenvector
        std::copy(&h_full_evecs[k * total_dim],
                  &h_full_evecs[k * total_dim + singles_dim],
                  &h_final_eigenvectors[k * singles_dim]);
    }

}


// ========================================================================
//  Solver mode: schur_davidson — matrix-free Davidson on the folded M_eff(ω)
//  For large clusters (singles ≳ 10⁴) where a dense diagonalisation of the
//  singles×singles M_eff is prohibitive (O(singles³) tridiagonalisation ≈ hours
//  at singles≈30k, even for a partial eigensolver — syevdx only trims the
//  eigenvector back-transform, not the tridiagonalisation). M_eff(ω) is symmetric
//  and ADC2Operator::apply() already applies it matrix-free (M11·x kernel-folded
//  through the diagonal doubles block), so a symmetric Davidson needs only a
//  handful of O(singles) vectors and fast matvecs — minutes instead of hours.
//  Solves at ω=0 (schur_static quality, ~0.005–0.02 Ha); an optional ω-refinement
//  loop (warm-started, cheap because ω barely moves) tightens it toward exact.
// ========================================================================
static void solve_schur_davidson(
    ADC2Operator& adc2_op, int n_states, int singles_dim,
    std::vector<real_t>& excitation_energies,
    std::vector<real_t>& h_final_eigenvectors,
    bool omega_refine)
{
    std::cout << "  Solving with matrix-free Davidson on M_eff(ω)"
              << (omega_refine ? " + ω-refinement" : " (ω=0, schur_static quality)")
              << " — no dense eig..." << std::endl;

    // Solve for a BUFFER of extra roots and keep the lowest n_states. Near-degenerate
    // dark/bright manifolds (e.g. naphthalene ~7 eV) let a random-init Davidson skip a
    // root and converge to {1,2,4}; the buffer makes the lowest n_states robust.
    const int nev = std::min(singles_dim, n_states + std::max(3, n_states));

    DavidsonConfig config;
    config.num_eigenvalues       = nev;
    config.symmetric             = true;   // the folded M_eff(ω) is symmetric
    config.use_preconditioner    = true;
    config.convergence_threshold = 1e-6;
    config.max_iterations        = 300;
    config.max_subspace_size     = std::max(30, 4 * nev);
    config.min_eigenvalue        = 1e-4;   // skip any spurious ~0 root
    config.verbose               = 1;

    // Phase 1: ω = 0 (matrix-free equivalent of schur_static).
    adc2_op.set_omega(0.0);
    adc2_op.update_diagonal();             // preconditioner diagonal for M_eff(0)
    DavidsonSolver solver(adc2_op, config);
    bool converged = solver.solve();
    if (!converged)
        std::cout << "  Warning: Davidson (ω=0) did not fully converge" << std::endl;

    // Keep the lowest n_states (eigenvalues ascending; vectors contiguous per root).
    excitation_energies.assign(solver.get_eigenvalues().begin(),
                               solver.get_eigenvalues().begin() + n_states);
    h_final_eigenvectors.resize((size_t)n_states * singles_dim);
    {
        std::vector<real_t> h_all((size_t)nev * singles_dim);
        solver.copy_eigenvectors_to_host(h_all.data());
        std::copy(h_all.begin(), h_all.begin() + (size_t)n_states * singles_dim,
                  h_final_eigenvectors.begin());
    }

    if (!omega_refine) {
        std::cout << "  (approximate: ~0.005-0.02 Ha ω=0 Schur error vs. exact ADC(2))"
                  << std::endl;
        return;
    }

    // Phase 2: per-root ω self-consistency. M_eff depends on ω through the diagonal
    // doubles denominators; ω barely moves between iterations, so warm-starting each
    // Davidson from the previous Ritz vectors converges in a few matvecs.
    const double omega_threshold = 1e-7;
    const int    max_omega_iter  = 12;
    std::vector<real_t> h_warm((size_t)nev * singles_dim);   // dav returns nev vectors
    for (int k = 0; k < n_states; k++) {
        real_t omega = excitation_energies[k];
        for (int iter = 0; iter < max_omega_iter; iter++) {
            adc2_op.set_omega(omega);
            adc2_op.update_diagonal();
            DavidsonSolver dav(adc2_op, config);
            dav.solve(solver.get_eigenvectors_device());   // warm start from ω=0 Ritz vectors
            real_t omega_new = dav.get_eigenvalues()[k];
            real_t delta = std::abs(omega_new - omega);
            std::cout << "  Root " << k + 1 << " ω-iter " << std::setw(2) << iter + 1
                      << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                      << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                      << std::defaultfloat << std::endl;
            omega = omega_new;
            if (delta < omega_threshold) {
                dav.copy_eigenvectors_to_host(h_warm.data());
                std::copy(&h_warm[(size_t)k * singles_dim],
                          &h_warm[(size_t)(k + 1) * singles_dim],
                          &h_final_eigenvectors[(size_t)k * singles_dim]);
                break;
            }
        }
        excitation_energies[k] = omega;
    }
}


// ========================================================================
//  Main entry point: compute_adc2
// ========================================================================

static void compute_adc2_impl(RHF& rhf, const real_t* d_eri_ao, int n_states, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_frozen = rhf.get_num_frozen_core();
    const int num_basis = rhf.get_num_basis();
    const int full_occ = rhf.get_num_electrons() / 2;
    const int num_occ = full_occ - num_frozen;
    const int num_vir = num_basis - full_occ;
    const int singles_dim = num_occ * num_vir;
    std::string solver_mode = rhf.get_adc2_solver();

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();

    int doubles_dim = num_occ * num_occ * num_vir * num_vir;

    // Auto solver selection: schur_static is fastest but inaccurate with frozen core.
    // Frozen core: use full Davidson (Schur complement approximation breaks down).
    if (solver_mode == "auto") {
        if (singles_dim > 10000) {
            // Dense M_eff diag is prohibitive at this size — matrix-free Davidson.
            solver_mode = "schur_davidson";
            std::cout << "  Auto solver: schur_davidson (large singles=" << singles_dim << ")" << std::endl;
        } else if (num_frozen > 0) {
            solver_mode = "full";
            std::cout << "  Auto solver: full (frozen core)" << std::endl;
        } else {
            solver_mode = "schur_static";
            std::cout << "  Auto solver: schur_static" << std::endl;
        }
    }

    bool is_triplet = rhf.is_triplet();
    std::string spin_label = is_triplet ? "triplet" : "singlet";

    std::cout << "\n---- ADC(2) " << spin_label << " excited states ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", solver=" << solver_mode
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Requested " << n_states << " states but singles dimension is "
                  << singles_dim << ". Reducing to " << singles_dim << "." << std::endl;
        n_states = singles_dim;
    }

    // ------------------------------------------------------------------
    // Step 1: Transform AO ERIs to MO ERIs
    // ------------------------------------------------------------------
    report_progress("excited", 0, 0, nullptr);  // MO transform start
    Timer mo_timer;
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
    std::cout << "  MO transform time: " << std::fixed << std::setprecision(3)
              << mo_timer.elapsed_seconds() << " s" << std::endl;

    // ------------------------------------------------------------------
    // Step 2: Get orbital energies and build ADC(2) operator
    // ------------------------------------------------------------------
    report_progress("excited", 1, 0, nullptr);  // Operator build
    Timer build_timer;
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    ADC2Operator adc2_op(d_eri_mo, d_orbital_energies, num_occ, num_vir, num_basis, is_triplet, num_frozen, full_occ);

    // Free full MO ERIs — blocks are already extracted
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;
    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s" << std::endl;

    // ------------------------------------------------------------------
    // Step 3: Solve (dispatch based on solver mode)
    // ------------------------------------------------------------------
    report_progress("excited", 2, 0, nullptr);  // Solver start
    Timer adc2_timer;

    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_final_eigenvectors;

    if (solver_mode == "schur_static") {
        solve_schur_static(adc2_op, n_states, singles_dim,
                           excitation_energies, h_final_eigenvectors);
    } else if (solver_mode == "schur_davidson") {
        solve_schur_davidson(adc2_op, n_states, singles_dim,
                             excitation_energies, h_final_eigenvectors,
                             /*omega_refine=*/std::getenv("GANSU_ADC2_OMEGA_REFINE") != nullptr);
    } else if (solver_mode == "full") {
        solve_full_davidson(adc2_op, n_states, singles_dim,
                            excitation_energies, h_final_eigenvectors);
    } else {
        // Default: schur_omega
        if (solver_mode != "schur_omega") {
            std::cout << "  Warning: Unknown adc2_solver '" << solver_mode
                      << "', using schur_omega" << std::endl;
        }
        solve_schur_omega(adc2_op, n_states, singles_dim,
                          excitation_energies, h_final_eigenvectors);
    }

    std::cout << "  ADC(2) time: " << std::fixed << std::setprecision(3)
              << adc2_timer.elapsed_seconds() << " s" << std::endl;

    // Store excitation energies
    rhf.set_excitation_energies(excitation_energies);

    // ------------------------------------------------------------------
    // Step 4: Print results with oscillator strengths
    // ------------------------------------------------------------------
    coefficient_matrix.toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "ADC(2) (triplet)" : "ADC(2)";
    auto es_result = compute_excited_state_properties(
        method_name,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        coefficient_matrix.host_ptr(),
        excitation_energies, h_final_eigenvectors.data(),
        n_states, num_basis, num_occ, num_vir,
        num_frozen, full_occ);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);

    // Cleanup: ADC2Operator handles its own memory via RAII
}

// ============================================================================
//  DMET-cluster ADC(2) entry — ADC(2) analogue of steom_spatial_orbital.
//  Runs ADC(2) on the embedded cluster MO space; the ADC2Operator math is
//  already RHF-free, so this is the compute_adc2_impl body with the RHF-sourced
//  orbital space (num_basis/full_occ/frozen/C/eps) replaced by cluster params.
//  ⚠ d_eps MUST be the PHYSICAL (un-shifted) cluster ε — ADC(2) denominators
//  (ε_a+ε_b−ε_i−ε_j) require the true energies, unlike the STEOM chain which
//  carries the level shift and un-shifts internally.
// ============================================================================
STEOMResult adc2_spatial_orbital(RHF& cfg, ERI& eri_method,
                                 const real_t* d_C_can, const real_t* d_eps,
                                 real_t* d_eri_mo,
                                 int nao, int n_emb, int n_emb_occ,
                                 int n_states, int n_frozen)
{
    (void)eri_method; (void)d_C_can; (void)nao;   // integrals already in MO basis (d_eri_mo)
    const int num_occ = n_emb_occ - n_frozen;     // active occupied
    const int num_vir = n_emb - n_emb_occ;
    const int singles_dim = num_occ * num_vir;
    const bool is_triplet = cfg.is_triplet();

    std::string solver_mode = cfg.get_adc2_solver();
    if (solver_mode == "auto") {
        // Full Davidson stores the O(singles²) doubles subspace explicitly, so for a
        // large DMET cluster it blows past GPU memory (n_emb=427 → ~1.3 TB). The
        // ADC(2) doubles–doubles block is diagonal, so schur_omega folds it EXACTLY
        // (same roots as full Davidson) at only O(singles²) for the M_eff matrix.
        // Prefer full only when its footprint comfortably fits; otherwise schur_omega.
        const size_t total_dim = (size_t)singles_dim + (size_t)singles_dim * singles_dim;
        const size_t subspace  = std::min(total_dim, (size_t)std::max(100, 10 * n_states));
        const size_t est_full  = (size_t)(2.5 * total_dim * subspace * sizeof(real_t));
        size_t free_b = 0, tot_b = 0;
        if (gpu::gpu_available()) cudaMemGetInfo(&free_b, &tot_b);
        const bool full_fits = tot_b > 0 && est_full < (size_t)(0.5 * tot_b);
        // When full won't fit, fold the (diagonal) doubles into a singles×singles
        // M_eff. Both Schur solvers dense-diagonalise M_eff (O(singles³) each); the
        // ω-iterative variant repeats it per root, so it is only affordable while
        // that diag is cheap. Past ~1e4 singles a single 30k×30k eig is already
        // ~25 min, so fall back to the one-shot ω=0 static Schur (approximate but
        // tractable). schur_omega/full remain available via explicit --adc2_solver.
        // Past ~10⁴ singles a dense M_eff diag (even partial syevdx) is prohibitive
        // (O(singles³) tridiagonalisation), so use the matrix-free Davidson on the
        // folded M_eff — minutes instead of hours, same ω=0 Schur accuracy.
        if (full_fits)                 solver_mode = "full";
        else if (singles_dim <= 10000) solver_mode = "schur_omega";
        else                           solver_mode = "schur_davidson";
        std::cout << "  [ADC(2) auto-solver] full-Davidson footprint ~"
                  << (est_full >> 30) << " GB vs GPU " << (tot_b >> 30)
                  << " GB, singles=" << singles_dim << " → " << solver_mode << std::endl;
    }

    if (n_states > singles_dim) n_states = singles_dim;

    // Prefer the RI-block path: build the cluster B_mo (naux·n_emb²) and pull the
    // four ADC(2) sub-blocks (ovov/vvov/ooov/oovv) from it via mo_eri_block_into,
    // never materialising the dense n_emb⁴ MO-ERI (247 GB for n_emb=427). Only
    // when the caller passed a dense d_eri_mo (non-RI / forced) do we use it.
    ERI_RI* eri_ri = dynamic_cast<ERI_RI*>(&eri_method);
    const bool ri_block = (d_eri_mo == nullptr) && (eri_ri != nullptr) && gpu::gpu_available();

    std::cout << "\n---- DMET-cluster ADC(2) " << (is_triplet ? "triplet" : "singlet")
              << " ---- n_emb=" << n_emb << " nocc_act=" << num_occ << " nvir=" << num_vir
              << " n_frozen=" << n_frozen << " singles=" << singles_dim
              << " solver=" << solver_mode << " ERI=" << (ri_block ? "RI-block" : "dense")
              << " nstates=" << n_states << std::endl;

    real_t *d_ovov = nullptr, *d_vvov = nullptr, *d_ooov = nullptr, *d_oovv = nullptr;
    ADC2Operator* op = nullptr;
    if (ri_block) {
        const int O = n_frozen, V = n_emb_occ;   // active-occ start, virtual start
        tracked_cudaMalloc(&d_ovov, (size_t)num_occ*num_vir*num_occ*num_vir*sizeof(real_t));
        tracked_cudaMalloc(&d_vvov, (size_t)num_vir*num_vir*num_occ*num_vir*sizeof(real_t));
        tracked_cudaMalloc(&d_ooov, (size_t)num_occ*num_occ*num_occ*num_vir*sizeof(real_t));
        tracked_cudaMalloc(&d_oovv, (size_t)num_occ*num_occ*num_vir*num_vir*sizeof(real_t));
        // (pq|rs) blocks — ranges match build_adc2_blocks' scatter layout exactly.
        // Rebuild B_mo right before each consume: the B_mo workspace is transient and
        // mo_eri_block_into consumes it (matches the STEOM RI-block pattern; calling
        // build_B_mo once and consuming it four times corrupts the 2nd–4th blocks).
        eri_ri->mo_eri_block_into(eri_ri->build_B_mo(d_C_can, n_emb), n_emb, O, num_occ, V, num_vir, O, num_occ, V, num_vir, d_ovov);
        eri_ri->mo_eri_block_into(eri_ri->build_B_mo(d_C_can, n_emb), n_emb, V, num_vir, V, num_vir, O, num_occ, V, num_vir, d_vvov);
        eri_ri->mo_eri_block_into(eri_ri->build_B_mo(d_C_can, n_emb), n_emb, O, num_occ, O, num_occ, O, num_occ, V, num_vir, d_ooov);
        eri_ri->mo_eri_block_into(eri_ri->build_B_mo(d_C_can, n_emb), n_emb, O, num_occ, O, num_occ, V, num_vir, V, num_vir, d_oovv);
        // Block ctor takes COMPACTED ε (active occ then vir, no offsets) = d_eps + n_frozen.
        // The ctor copies ovov/vvov/ooov and consumes oovv into M11, so the source
        // blocks can be freed immediately — keeps the solve working set off the peak.
        op = new ADC2Operator(d_ovov, d_vvov, d_ooov, d_oovv, d_eps + n_frozen,
                              num_occ, num_vir, num_occ + num_vir, is_triplet);
        tracked_cudaFree(d_ovov); d_ovov = nullptr;
        tracked_cudaFree(d_vvov); d_vvov = nullptr;
        tracked_cudaFree(d_ooov); d_ooov = nullptr;
        tracked_cudaFree(d_oovv); d_oovv = nullptr;
    } else {
        // Dense path (occ_offset = n_frozen, vir_start = n_emb_occ into the full ε).
        op = new ADC2Operator(d_eri_mo, d_eps, num_occ, num_vir, n_emb,
                              is_triplet, /*occ_offset=*/n_frozen, /*vir_start=*/n_emb_occ);
    }

    // Sanity diagnostics (cheap; skippable via GANSU_ADC2_NO_DIAG). Two failure
    // modes to catch before an opaque "subspace eig failed": (a) NaN in a block /
    // M11 (indexing bug at this scale), (b) wrong frozen-core ε convention →
    // zeros in D2 → 1/(ω−D2) = inf on the very first apply.
    if (!std::getenv("GANSU_ADC2_NO_DIAG")) {
        auto finite_probe = [](const real_t* d, size_t n, const char* nm) {
            // chunked asum: cublasDasum takes int n, and vvov (2.4e9) exceeds it
            double total = 0.0; bool ok = true;
            const size_t chunk = 1u << 30;
            for (size_t off = 0; off < n; off += chunk) {
                int len = (int)std::min(chunk, n - off);
                real_t s = 0.0;
                cublasDasum(gpu::GPUHandle::cublas(), len, d + off, 1, &s);
                if (!std::isfinite(s)) ok = false;
                total += s;
            }
            std::cout << "  [ADC2 diag] " << nm << " asum=" << std::scientific
                      << total << (ok ? "" : "  *** NON-FINITE ***")
                      << std::defaultfloat << std::endl;
        };
        finite_probe(op->get_M11(), (size_t)singles_dim * singles_dim, "M11");
        finite_probe(op->get_D2(), (size_t)op->get_doubles_dim(), "D2");
        finite_probe(op->get_eri_vvov(), (size_t)num_vir * num_vir * num_occ * num_vir, "vvov");
        finite_probe(op->get_diagonal_device(), (size_t)singles_dim, "diagonal");
        // ε convention check: with the shifted pointer the active HOMO is entry
        // num_occ-1 and the LUMO is entry num_occ; their gap must equal the
        // cluster HOMO-LUMO gap printed above (0.34 Ha for dox), NOT ~0.
        std::vector<real_t> h_eps2(2);
        cudaMemcpy(h_eps2.data(), d_eps + n_frozen + num_occ - 1, 2 * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        std::cout << "  [ADC2 diag] active HOMO=" << h_eps2[0] << " LUMO=" << h_eps2[1]
                  << " gap=" << (h_eps2[1] - h_eps2[0]) << " Ha" << std::endl;
    }

    std::vector<real_t> exc, evec;
    if (solver_mode == "schur_static")
        solve_schur_static(*op, n_states, singles_dim, exc, evec);
    else if (solver_mode == "schur_davidson")
        solve_schur_davidson(*op, n_states, singles_dim, exc, evec,
                             /*omega_refine=*/std::getenv("GANSU_ADC2_OMEGA_REFINE") != nullptr);
    else if (solver_mode == "full")
        solve_full_davidson(*op, n_states, singles_dim, exc, evec);
    else
        solve_schur_omega(*op, n_states, singles_dim, exc, evec);

    delete op;
    if (d_ovov) tracked_cudaFree(d_ovov);
    if (d_vvov) tracked_cudaFree(d_vvov);
    if (d_ooov) tracked_cudaFree(d_ooov);
    if (d_oovv) tracked_cudaFree(d_oovv);

    // Package into a STEOMResult (energies only; η/R1/oscillator not defined for
    // the cluster ADC(2) path in this v1).
    STEOMResult r;
    r.nocc_active = num_occ;
    r.nvir       = num_vir;
    r.num_frozen = n_frozen;
    r.n_states   = (int)exc.size();
    std::ostringstream rep;
    rep << "  DMET-cluster ADC(2) excited-state energies:\n"
        << "   k   omega (Ha)        omega (eV)\n";
    for (int k = 0; k < (int)exc.size(); ++k) {
        STEOMResult::PerRoot pr;
        pr.omega = exc[k];
        r.per_root.push_back(pr);
        rep << "   " << std::setw(2) << k << "   "
            << std::fixed << std::setprecision(8) << std::setw(12) << exc[k] << "   "
            << std::setprecision(4) << std::setw(9) << exc[k] * 27.211386245988 << "\n";
    }
    r.report = rep.str();
    std::cout << r.report;
    return r;
}

void ERI_Stored_RHF::compute_adc2(int n_states) {
    compute_adc2_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

void ERI_RI_RHF::compute_adc2(int n_states) {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    compute_adc2_impl(rhf_, nullptr, n_states, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
}

// ========================================================================
//  Forward declarations (from half_transform_mp3.cu)
// ========================================================================
void half_transform_steps234(
    const real_t* d_half, const real_t* d_C, real_t* d_result,
    real_t* d_Ki, real_t* d_Li,
    int nao, int bs, int p_start, int n_p, int q_start, int n_q, int r_start, int n_r);
__global__ void scatter_s_block_kernel(
    const double* __restrict__ src, double* __restrict__ dst,
    int n_p, int n_q, int n_r, int n_s, int s_blk, int bs);
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


// ========================================================================
//  Build 4 sub-blocks (ovov, vvov, ooov, oovv) via half-transform
// ========================================================================
template <typename Step1Func>
static void build_adc2_blocks(
    RHF& rhf, Step1Func step1_func, int block_s,
    real_t* d_ovov, real_t* d_vvov, real_t* d_ooov, real_t* d_oovv)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const size_t nao3 = (size_t)nao * nao * nao;
    const int max_dim = std::max(nocc, nvir);

    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;
    real_t* d_tmp_block = nullptr;
    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)max_dim * nao * block_s * sizeof(real_t));
    const size_t max_block = std::max({
        (size_t)nocc * nvir * nocc * block_s,
        (size_t)nvir * nvir * nocc * block_s,
        (size_t)nocc * nocc * nocc * block_s,
        (size_t)nocc * nocc * nvir * block_s
    });
    tracked_cudaMalloc(&d_tmp_block, max_block * sizeof(real_t));

    constexpr int scatter_threads = 256;
    auto build_and_scatter = [&](real_t* d_dst, int n_p, int n_q, int n_r, int n_s,
                                  int s_blk, int bs,
                                  int p_start, int q_start, int r_start) {
        half_transform_steps234(d_half, d_C, d_tmp_block,
            d_Ki, d_Li, nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);
        const size_t block_total = (size_t)n_p * n_q * n_r * bs;
        if (!gpu::gpu_available()) {
            // CPU fallback for scatter_s_block_kernel
            #pragma omp parallel for
            for (size_t idx = 0; idx < block_total; idx++) {
                int s_local = (int)(idx % bs);
                size_t rem = idx / bs;
                int r_local = (int)(rem % n_r); rem /= n_r;
                int q_local = (int)(rem % n_q); rem /= n_q;
                int p_local = (int)rem;
                int s_global = s_blk + s_local;
                if (s_global < n_s) {
                    size_t dst_idx = (size_t)p_local * n_q * n_r * n_s
                                   + (size_t)q_local * n_r * n_s
                                   + (size_t)r_local * n_s
                                   + s_global;
                    d_dst[dst_idx] = d_tmp_block[idx];
                }
            }
        } else {
            scatter_s_block_kernel<<<(block_total + scatter_threads - 1) / scatter_threads, scatter_threads>>>(
                d_tmp_block, d_dst, n_p, n_q, n_r, n_s, s_blk, bs);
        }
    };

    std::cout << "  Building ADC(2) sub-blocks via half-transform..." << std::flush;
    for (int s_blk = 0; s_blk < nvir; s_blk += block_s) {
        int bs = std::min(block_s, nvir - s_blk);
        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        step1_func(d_half, nao, d_C, nocc + s_blk, bs);
        build_and_scatter(d_ovov, nocc, nvir, nocc, nvir, s_blk, bs, 0, nocc, 0);
        build_and_scatter(d_vvov, nvir, nvir, nocc, nvir, s_blk, bs, nocc, nocc, 0);
        build_and_scatter(d_ooov, nocc, nocc, nocc, nvir, s_blk, bs, 0, 0, 0);
        build_and_scatter(d_oovv, nocc, nocc, nvir, nvir, s_blk, bs, 0, 0, nocc);
    }
    std::cout << " done" << std::endl;

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_tmp_block);
}


// ========================================================================
//  ADC(2) solver from pre-built sub-blocks
// ========================================================================
static void compute_adc2_from_blocks(
    RHF& rhf,
    const real_t* d_ovov, const real_t* d_vvov, const real_t* d_ooov, const real_t* d_oovv,
    int n_states)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int singles_dim = nocc * nvir;
    const int doubles_dim = nocc * nocc * nvir * nvir;
    std::string solver_mode = rhf.get_adc2_solver();
    const bool is_triplet = rhf.is_triplet();

    if (solver_mode == "auto") {
        solver_mode = (singles_dim > 10000) ? "schur_davidson" : "schur_static";
        std::cout << "  Auto solver: " << solver_mode
                  << (singles_dim > 10000 ? " (large singles)" : "") << std::endl;
    }

    std::string spin_label = is_triplet ? "triplet" : "singlet";
    std::cout << "\n---- ADC(2) " << spin_label << " (half-transform) ---- "
              << "nocc=" << nocc << ", nvir=" << nvir
              << ", singles=" << singles_dim << ", doubles=" << doubles_dim
              << ", solver=" << solver_mode
              << ", nstates=" << n_states << std::endl;

    if (n_states > singles_dim) {
        std::cout << "Warning: Reducing to " << singles_dim << " states." << std::endl;
        n_states = singles_dim;
    }

    const real_t* d_eps = rhf.get_orbital_energies().device_ptr();
    ADC2Operator adc2_op(d_ovov, d_vvov, d_ooov, d_oovv, d_eps,
                         nocc, nvir, nao, is_triplet);

    Timer adc2_timer;
    std::vector<real_t> excitation_energies;
    std::vector<real_t> h_final_eigenvectors;

    if (solver_mode == "schur_static") {
        solve_schur_static(adc2_op, n_states, singles_dim,
                           excitation_energies, h_final_eigenvectors);
    } else if (solver_mode == "schur_davidson") {
        solve_schur_davidson(adc2_op, n_states, singles_dim,
                             excitation_energies, h_final_eigenvectors,
                             /*omega_refine=*/std::getenv("GANSU_ADC2_OMEGA_REFINE") != nullptr);
    } else if (solver_mode == "full") {
        solve_full_davidson(adc2_op, n_states, singles_dim,
                            excitation_energies, h_final_eigenvectors);
    } else {
        solve_schur_omega(adc2_op, n_states, singles_dim,
                          excitation_energies, h_final_eigenvectors);
    }

    std::cout << "  ADC(2) time: " << std::fixed << std::setprecision(3)
              << adc2_timer.elapsed_seconds() << " s" << std::endl;

    rhf.set_excitation_energies(excitation_energies);

    rhf.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf.get_primitive_shells();
    const auto& cgto_norms = rhf.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    std::string method_name = is_triplet ? "ADC(2) (triplet)" : "ADC(2)";
    auto es_result = compute_excited_state_properties(
        method_name,
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf.get_shell_type_infos(),
        rhf.get_coefficient_matrix().host_ptr(),
        excitation_energies, h_final_eigenvectors.data(),
        n_states, nao, nocc, nvir);
    rhf.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf.set_excited_state_report(es_result.report);
}


// ========================================================================
//  Auto-select: build_mo_eri (fast) if nao⁴ fits, else half-transform
// ========================================================================

void ERI_Direct_RHF::compute_adc2(int n_states) {
    PROFILE_FUNCTION();
    const int nao = rhf_.get_num_basis();
    const size_t nao4_bytes = (size_t)nao * nao * nao * nao * sizeof(real_t);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (nao4_bytes < free_mem * 6 / 10) {
        // Fast path: build full MO ERI
        std::cout << "  [Direct ADC(2)] Using build_mo_eri (nao^4 = "
                  << nao4_bytes / (1024*1024) << " MB fits in GPU)" << std::endl;
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        compute_adc2_impl(rhf_, nullptr, n_states, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
    } else {
        // Memory-efficient path: half-transform sub-blocks
        std::cout << "  [Direct ADC(2)] Using half-transform (nao^4 = "
                  << nao4_bytes / (1024*1024) << " MB exceeds GPU memory)" << std::endl;
        const int nocc = rhf_.get_num_electrons() / 2;
        const int nvir = nao - nocc;

        real_t *d_ovov = nullptr, *d_vvov = nullptr, *d_ooov = nullptr, *d_oovv = nullptr;
        tracked_cudaMalloc(&d_ovov, (size_t)nocc * nvir * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_vvov, (size_t)nvir * nvir * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_ooov, (size_t)nocc * nocc * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_oovv, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
        cudaMemset(d_ovov, 0, (size_t)nocc * nvir * nocc * nvir * sizeof(real_t));
        cudaMemset(d_vvov, 0, (size_t)nvir * nvir * nocc * nvir * sizeof(real_t));
        cudaMemset(d_ooov, 0, (size_t)nocc * nocc * nocc * nvir * sizeof(real_t));
        cudaMemset(d_oovv, 0, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));

        const size_t nao3 = (size_t)nao * nao * nao;
        int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
        block_s = std::min(block_s, nao);
        if (block_s > 8) block_s = 8;

        DeviceHostMemory<real_t> schwarz_unsorted(hf_.get_num_primitive_shell_pairs());
        gpu::computeSchwarzUpperBounds(
            hf_.get_shell_type_infos(), hf_.get_shell_pair_type_infos(),
            hf_.get_primitive_shells().device_ptr(),
            hf_.get_boys_grid().device_ptr(),
            hf_.get_cgto_normalization_factors().device_ptr(),
            schwarz_unsorted.device_ptr(), false);

        const auto& sti = hf_.get_shell_type_infos();
        const auto& spti = hf_.get_shell_pair_type_infos();
        const auto* d_ps = hf_.get_primitive_shells().device_ptr();
        const auto* d_bg = hf_.get_boys_grid().device_ptr();
        const auto* d_cn = hf_.get_cgto_normalization_factors().device_ptr();
        const auto* d_sw = schwarz_unsorted.device_ptr();
        const real_t sw_th = hf_.get_schwarz_screening_threshold();

        auto step1 = [&](real_t* d_half, int nao_arg, const real_t* d_C, int s_abs, int bs) {
            gpu::computeHalfTransformedERI(sti, spti, d_ps, d_bg, d_cn,
                d_half, d_sw, sw_th, nao_arg, d_C, s_abs, bs);
        };

        build_adc2_blocks(rhf_, step1, block_s, d_ovov, d_vvov, d_ooov, d_oovv);
        compute_adc2_from_blocks(rhf_, d_ovov, d_vvov, d_ooov, d_oovv, n_states);

        tracked_cudaFree(d_ovov);
        tracked_cudaFree(d_vvov);
        tracked_cudaFree(d_ooov);
        tracked_cudaFree(d_oovv);
    }
}

void ERI_Hash_RHF::compute_adc2(int n_states) {
    PROFILE_FUNCTION();
    const int nao = rhf_.get_num_basis();
    const size_t nao4_bytes = (size_t)nao * nao * nao * nao * sizeof(real_t);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    if (nao4_bytes < free_mem * 6 / 10) {
        std::cout << "  [Hash ADC(2)] Using build_mo_eri (nao^4 = "
                  << nao4_bytes / (1024*1024) << " MB fits in GPU)" << std::endl;
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        compute_adc2_impl(rhf_, nullptr, n_states, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
    } else {
        std::cout << "  [Hash ADC(2)] Using half-transform (nao^4 = "
                  << nao4_bytes / (1024*1024) << " MB exceeds GPU memory)" << std::endl;
        const int nocc = rhf_.get_num_electrons() / 2;
        const int nvir = nao - nocc;

        real_t *d_ovov = nullptr, *d_vvov = nullptr, *d_ooov = nullptr, *d_oovv = nullptr;
        tracked_cudaMalloc(&d_ovov, (size_t)nocc * nvir * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_vvov, (size_t)nvir * nvir * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_ooov, (size_t)nocc * nocc * nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_oovv, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
        cudaMemset(d_ovov, 0, (size_t)nocc * nvir * nocc * nvir * sizeof(real_t));
        cudaMemset(d_vvov, 0, (size_t)nvir * nvir * nocc * nvir * sizeof(real_t));
        cudaMemset(d_ooov, 0, (size_t)nocc * nocc * nocc * nvir * sizeof(real_t));
        cudaMemset(d_oovv, 0, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));

        const size_t nao3 = (size_t)nao * nao * nao;
        int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
        block_s = std::min(block_s, nao);
        if (block_s > 8) block_s = 8;

        auto step1 = [&](real_t* d_half, int nao_arg, const real_t* d_C, int s_abs, int bs) {
            if (!gpu::gpu_available()) {
                // CPU fallback for hash half-transform
                // half[mu * nao * nao * bs + nu * nao * bs + la * bs + s_local]
                //   = sum_si C[si, s_abs + s_local] * (mu nu | la si)
                // Use compact COO format on CPU (same data accessible via host)
                if (hash_fock_method_ == HashFockMethod::Compact) {
                    for (size_t e = 0; e < num_entries_; e++) {
                        unsigned long long key = d_coo_keys_[e];
                        double val = d_coo_values_[e];
                        int si = (int)(key & 0xFFFF);
                        int la = (int)((key >> 16) & 0xFFFF);
                        int nu = (int)((key >> 32) & 0xFFFF);
                        int mu = (int)((key >> 48) & 0xFFFF);
                        for (int s_local = 0; s_local < bs; s_local++) {
                            double c_val = d_C[(s_abs + s_local) * nao_arg + si];
                            size_t half_idx = (size_t)mu * nao_arg * nao_arg * bs
                                            + (size_t)nu * nao_arg * bs
                                            + (size_t)la * bs + s_local;
                            d_half[half_idx] += val * c_val;
                        }
                    }
                } else if (hash_fock_method_ == HashFockMethod::Indexed) {
                    for (size_t n = 0; n < num_nonzero_; n++) {
                        size_t slot = d_nonzero_indices_[n];
                        unsigned long long key = d_hash_keys_[slot];
                        if (key == 0xFFFFFFFFFFFFFFFFULL) continue;
                        double val = d_hash_values_[slot];
                        int si = (int)(key & 0xFFFF);
                        int la = (int)((key >> 16) & 0xFFFF);
                        int nu = (int)((key >> 32) & 0xFFFF);
                        int mu = (int)((key >> 48) & 0xFFFF);
                        for (int s_local = 0; s_local < bs; s_local++) {
                            double c_val = d_C[(s_abs + s_local) * nao_arg + si];
                            size_t half_idx = (size_t)mu * nao_arg * nao_arg * bs
                                            + (size_t)nu * nao_arg * bs
                                            + (size_t)la * bs + s_local;
                            d_half[half_idx] += val * c_val;
                        }
                    }
                } else {
                    const size_t capacity = hash_capacity_mask_ + 1;
                    for (size_t slot = 0; slot < capacity; slot++) {
                        unsigned long long key = d_hash_keys_[slot];
                        if (key == 0xFFFFFFFFFFFFFFFFULL) continue;
                        double val = d_hash_values_[slot];
                        int si = (int)(key & 0xFFFF);
                        int la = (int)((key >> 16) & 0xFFFF);
                        int nu = (int)((key >> 32) & 0xFFFF);
                        int mu = (int)((key >> 48) & 0xFFFF);
                        for (int s_local = 0; s_local < bs; s_local++) {
                            double c_val = d_C[(s_abs + s_local) * nao_arg + si];
                            size_t half_idx = (size_t)mu * nao_arg * nao_arg * bs
                                            + (size_t)nu * nao_arg * bs
                                            + (size_t)la * bs + s_local;
                            d_half[half_idx] += val * c_val;
                        }
                    }
                }
            } else {
                const int threads = 256;
                if (hash_fock_method_ == HashFockMethod::Compact) {
                    const int blocks = ((int)num_entries_ + threads - 1) / threads;
                    hash_half_transform_compact_kernel<<<blocks, threads>>>(
                        d_coo_keys_, d_coo_values_, num_entries_, d_C, d_half, nao_arg, s_abs, bs);
                } else if (hash_fock_method_ == HashFockMethod::Indexed) {
                    const int blocks = ((int)num_nonzero_ + threads - 1) / threads;
                    hash_half_transform_indexed_kernel<<<blocks, threads>>>(
                        d_hash_keys_, d_hash_values_, d_nonzero_indices_, num_nonzero_,
                        d_C, d_half, nao_arg, s_abs, bs);
                } else {
                    const size_t capacity = hash_capacity_mask_ + 1;
                    const int blocks = ((int)capacity + threads - 1) / threads;
                    hash_half_transform_fullscan_kernel<<<blocks, threads>>>(
                        d_hash_keys_, d_hash_values_, capacity, d_C, d_half, nao_arg, s_abs, bs);
                }
                cudaDeviceSynchronize();
            }
        };

        build_adc2_blocks(rhf_, step1, block_s, d_ovov, d_vvov, d_ooov, d_oovv);
        compute_adc2_from_blocks(rhf_, d_ovov, d_vvov, d_ooov, d_oovv, n_states);

        tracked_cudaFree(d_ovov);
        tracked_cudaFree(d_vvov);
        tracked_cudaFree(d_ooov);
        tracked_cudaFree(d_oovv);
    }
}

} // namespace gansu
