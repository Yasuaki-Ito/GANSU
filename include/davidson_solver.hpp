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
 * @file davidson_solver.hpp
 * @brief GPU-accelerated Davidson iterative eigenvalue solver
 *
 * This file provides a generic implementation of the Davidson method for
 * computing the lowest eigenvalues and eigenvectors of large symmetric matrices.
 * The solver uses a matrix-free approach through the LinearOperator interface,
 * making it suitable for Full-CI, EOM-CC, and other large-scale quantum chemistry
 * eigenvalue problems.
 */

#pragma once

#include "linear_operator.hpp"
#include <vector>
#include <string>

namespace gansu {

/**
 * @brief Configuration parameters for Davidson solver
 *
 * Controls the behavior and convergence criteria of the Davidson eigenvalue solver.
 */
struct DavidsonConfig {
    /**
     * @brief Number of lowest eigenvalues to compute
     *
     * Default: 1 (ground state only)
     */
    int num_eigenvalues = 1;

    /**
     * @brief Maximum dimension of the subspace before restart
     *
     * Larger values allow more iterations before restart but use more memory.
     * Typical values: 20-50 for small problems, up to 100 for large problems.
     *
     * Default: 30
     */
    int max_subspace_size = 30;

    /**
     * @brief Initial subspace dimension
     *
     * Number of initial guess vectors. If 0, defaults to 2*num_eigenvalues.
     * Should be >= num_eigenvalues.
     *
     * Default: 0 (auto)
     */
    int initial_subspace_size = 0;

    /**
     * @brief Convergence threshold for residual norms
     *
     * Solver converges when ||r_i|| < convergence_threshold for all i.
     *
     * Default: 1e-6
     */
    double convergence_threshold = 1e-6;

    /**
     * @brief Maximum number of Davidson iterations
     *
     * Default: 100
     */
    int max_iterations = 100;

    /**
     * @brief Enable preconditioning
     *
     * If true, applies operator's preconditioner to correction vectors.
     * Can significantly accelerate convergence.
     *
     * Default: true
     */
    bool use_preconditioner = true;

    /**
     * @brief Verbosity level
     *
     * - 0: Silent (only errors)
     * - 1: Summary (final results)
     * - 2: Detailed (per-iteration progress)
     *
     * Default: 0
     */
    int verbose = 0;
};

/**
 * @brief GPU-accelerated Davidson iterative eigenvalue solver
 *
 * Computes the lowest N eigenvalues and corresponding eigenvectors of a
 * large symmetric matrix using the Davidson iterative subspace method.
 * The operator is accessed through the LinearOperator interface, allowing
 * matrix-free implementations.
 *
 * Algorithm overview:
 * 1. Initialize subspace with random or provided guess vectors
 * 2. Orthogonalize subspace vectors (Modified Gram-Schmidt)
 * 3. Apply operator to basis vectors
 * 4. Build and solve small subspace eigenvalue problem
 * 5. Compute residuals for Ritz vectors
 * 6. Check convergence
 * 7. Add preconditioned correction vectors to subspace
 * 8. Restart if subspace becomes too large
 *
 * Example usage:
 * @code
 * // Create operator (diagonal matrix for testing)
 * std::vector<real_t> diag = {1.0, 2.0, 3.0, 4.0, 5.0};
 * DiagonalOperator op(diag);
 *
 * // Configure solver
 * DavidsonConfig config;
 * config.num_eigenvalues = 3;
 * config.convergence_threshold = 1e-8;
 * config.verbose = 2;
 *
 * // Solve
 * DavidsonSolver solver(op, config);
 * bool converged = solver.solve();
 *
 * // Extract results
 * const auto& eigenvalues = solver.get_eigenvalues();
 * std::cout << "Lowest eigenvalue: " << eigenvalues[0] << std::endl;
 * @endcode
 *
 * Memory requirements:
 * - Subspace vectors: O(dim × max_subspace_size)
 * - Operator-applied vectors: O(dim × max_subspace_size)
 * - Subspace matrix: O(max_subspace_size^2)
 * - Total: ~2 × dim × max_subspace_size real numbers
 *
 * @note All linear algebra operations are performed on the GPU
 * @note Thread-safe: Multiple solvers can run concurrently with different operators
 */
class DavidsonSolver {
public:
    /**
     * @brief Construct a Davidson solver
     *
     * @param linear_op Reference to the linear operator to diagonalize
     *                  (must outlive the solver instance)
     * @param config Solver configuration parameters
     *
     * @throws std::runtime_error if configuration is invalid
     * @throws std::runtime_error if GPU memory allocation fails
     */
    DavidsonSolver(const LinearOperator& linear_op,
                   const DavidsonConfig& config = DavidsonConfig());

    /**
     * @brief Destructor - frees all GPU memory
     */
    ~DavidsonSolver();

    // Disable copying (GPU memory management)
    DavidsonSolver(const DavidsonSolver&) = delete;
    DavidsonSolver& operator=(const DavidsonSolver&) = delete;

    /**
     * @brief Solve for eigenvalues and eigenvectors
     *
     * Performs Davidson iterations until convergence or maximum iterations reached.
     *
     * @param d_initial_guess Optional device pointer to initial guess vectors
     *                        (dim × num_eigenvalues in column-major order)
     *                        If nullptr, random vectors are used
     *
     * @return true if converged, false if maximum iterations reached
     *
     * @post get_eigenvalues() returns computed eigenvalues (sorted ascending)
     * @post get_eigenvectors_device() returns computed eigenvectors
     * @post get_residual_norms() returns final residual norms
     *
     * @note Can be called multiple times with different initial guesses
     * @note Previous results are overwritten
     */
    bool solve(const real_t* d_initial_guess = nullptr);

    /**
     * @brief Get computed eigenvalues (host memory)
     *
     * @return Vector of eigenvalues sorted in ascending order
     *
     * @pre solve() must have been called
     */
    const std::vector<real_t>& get_eigenvalues() const { return h_eigenvalues_; }

    /**
     * @brief Get computed eigenvectors (device memory pointer)
     *
     * Returns device pointer to eigenvectors stored in column-major order:
     * eigenvectors[i * dim + j] = j-th component of i-th eigenvector
     *
     * @return Device pointer to eigenvectors (dim × num_eigenvalues)
     *
     * @pre solve() must have been called
     * @note Pointer is valid until destructor or next solve() call
     * @note Do not free this pointer - managed internally
     */
    const real_t* get_eigenvectors_device() const { return d_eigenvectors_; }

    /**
     * @brief Copy eigenvectors to host memory
     *
     * @param h_output Host memory buffer (must be pre-allocated: dim × num_eigenvalues)
     *
     * @pre solve() must have been called
     * @pre h_output must point to valid host memory of sufficient size
     */
    void copy_eigenvectors_to_host(real_t* h_output) const;

    /**
     * @brief Get residual norms for each computed eigenvalue
     *
     * Returns ||r_i|| where r_i = (H - λ_i)ψ_i for each eigenpair.
     *
     * @return Vector of residual norms (size: num_eigenvalues)
     *
     * @pre solve() must have been called
     */
    const std::vector<real_t>& get_residual_norms() const { return residual_norms_; }

    /**
     * @brief Get number of iterations performed in last solve()
     *
     * @return Number of Davidson iterations (0 if solve() not called)
     */
    int get_num_iterations() const { return num_iterations_; }

private:
    // ========== Member Variables ==========

    const LinearOperator& linear_op_;  ///< Reference to the linear operator
    DavidsonConfig config_;             ///< Solver configuration

    const int dim_;                     ///< Dimension of the operator
    int subspace_dim_;                  ///< Current subspace dimension
    int num_iterations_;                ///< Iteration counter

    // Device memory pointers
    real_t* d_subspace_vectors_;        ///< Subspace basis vectors (dim × max_subspace_size)
    real_t* d_sigma_vectors_;           ///< Operator-applied vectors (dim × max_subspace_size)
    real_t* d_subspace_matrix_;         ///< Subspace matrix H_ij (max_subspace × max_subspace)
    real_t* d_subspace_eigenvalues_;    ///< Subspace eigenvalues (max_subspace)
    real_t* d_subspace_eigenvectors_;   ///< Subspace eigenvectors (max_subspace × max_subspace)
    real_t* d_residuals_;               ///< Residual vectors (dim × num_eigenvalues)
    real_t* d_eigenvectors_;            ///< Final Ritz vectors (dim × num_eigenvalues)

    // Host memory
    std::vector<real_t> h_eigenvalues_;    ///< Computed eigenvalues
    std::vector<real_t> residual_norms_;   ///< Residual norms

    // ========== Private Methods ==========

    /**
     * @brief Initialize subspace with random or provided vectors
     *
     * @param d_initial_guess Optional initial guess vectors
     */
    void initialize_subspace(const real_t* d_initial_guess);

    /**
     * @brief Orthogonalize vectors using Modified Gram-Schmidt
     *
     * Orthonormalizes vectors[start_index : start_index + num_vectors - 1]
     * against all previous vectors and each other.
     *
     * @param start_index Index of first vector to orthogonalize
     * @param num_vectors Number of consecutive vectors to orthogonalize
     *
     * @note Uses cuBLAS for dot products, axpy, and scaling
     * @note Handles linear dependence by replacing with random vectors
     */
    void orthogonalize_vectors(int start_index, int num_vectors);

    /**
     * @brief Build subspace matrix H_ij = <v_i | Op(v_j)>
     *
     * Computes all inner products between subspace vectors and
     * operator-applied vectors.
     *
     * @note Uses cuBLAS for dot products
     * @note Matrix is symmetric, but both triangles are filled for cuSOLVER
     */
    void build_subspace_matrix();

    /**
     * @brief Solve dense eigenvalue problem in subspace
     *
     * Diagonalizes the subspace matrix to find Ritz values and Ritz vectors.
     *
     * @note Uses existing gpu::eigenDecomposition() from gpu_manager
     * @note Updates h_eigenvalues_ with current Ritz values
     */
    void solve_subspace_eigenproblem();

    /**
     * @brief Compute Ritz vectors and residuals
     *
     * For each desired eigenvalue:
     * - Ritz vector: ψ_i = Σ_j c_ji v_j
     * - Residual: r_i = (H - λ_i)ψ_i = Σ_j c_ji σ_j - λ_i ψ_i
     *
     * @note Uses cuBLAS for matrix-vector products
     * @note Updates residual_norms_
     */
    void compute_ritz_vectors_and_residuals();

    /**
     * @brief Check convergence based on residual norms
     *
     * @return true if all ||r_i|| < threshold, false otherwise
     */
    bool check_convergence();

    /**
     * @brief Add correction vectors to subspace
     *
     * For each unconverged eigenvalue, computes preconditioned correction:
     * δ_i = -M^(-1) r_i
     *
     * Adds normalized correction vectors to subspace and orthogonalizes.
     *
     * @note Only adds corrections for unconverged eigenpairs
     * @note Updates subspace_dim_
     */
    void add_correction_vectors();

    /**
     * @brief Restart subspace by keeping only converged vectors
     *
     * Keeps the num_eigenvalues Ritz vectors with lowest eigenvalues
     * and discards the rest. Recomputes operator application for kept vectors.
     *
     * @note Called when subspace_dim_ >= max_subspace_size
     * @note Resets subspace_dim_ to num_eigenvalues
     */
    void restart_subspace();

    /**
     * @brief Allocate GPU memory for solver
     *
     * @throws std::runtime_error if allocation fails
     */
    void allocate_memory();

    /**
     * @brief Free all GPU memory
     *
     * @note Safe to call multiple times
     */
    void free_memory();
};

} // namespace gansu
