/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_decomposition.hpp
 * @brief Least-Squares Tensor Hypercontraction (LS-THC).
 *
 * Given a collocation matrix X^P_mu = phi_mu(r_P) and the analytic 4-index
 * ERI tensor V[mu nu, lambda sigma] = (mu nu | lambda sigma), the LS-THC
 * factorisation finds a core matrix Z [N_g x N_g] such that
 *
 *    (mu nu | lambda sigma)
 *      ~~ sum_{P,Q} X^P_mu X^P_nu Z_{P,Q} X^Q_lambda X^Q_sigma.
 *
 * It minimises  || V - M Z M^T ||_F^2  with the Khatri-Rao columns
 *   M[(mu,nu), P] := X^P_mu * X^P_nu,
 * yielding the closed-form solution
 *
 *    Z = S^+ E S^+,
 *    S_{P,R} = (sum_mu X^P_mu X^R_mu)^2 = (Gram)^2  (Hadamard square),
 *    E_{P,Q} = sum_{mu nu lam sig} (mu nu | lam sig) X^P_mu X^P_nu X^Q_lam X^Q_sig.
 *
 * The metric S is generally rank-deficient when the grid over-resolves the
 * basis-pair space; we form its pseudo-inverse via eigendecomposition with a
 * relative cutoff.
 *
 * Phase 2.0a uses CPU + Eigen throughout.  GPU port comes after end-to-end
 * THC-MP2 validation passes.
 */

#pragma once

#include <vector>
#include <memory>
#include "types.hpp"
#ifndef GANSU_CPU_ONLY
#include "device_host_memory.hpp"
#endif

namespace gansu {

/**
 * @brief Build the Gram matrix G_{P,R} = sum_mu X[mu, P] * X[mu, R].
 *
 * @param X     Collocation matrix [N_bas x N_g] column-major (X[mu + P*N_bas]).
 * @param N_bas Number of basis functions.
 * @param N_g   Number of grid points.
 * @return G    Symmetric matrix [N_g x N_g] column-major.
 */
std::vector<real_t> build_gram_cpu(const std::vector<real_t>& X,
                                   int N_bas, int N_g);

/**
 * @brief Hadamard square A_{ij} := A_{ij} * A_{ij}.  Used to turn the Gram
 *        matrix into the LS-THC metric S.
 */
std::vector<real_t> hadamard_square(const std::vector<real_t>& A);

/**
 * @brief Build the LS-THC source matrix
 *        E_{P,Q} = sum_{mu nu lam sig} (mu nu | lam sig)
 *                                     * X^P_mu X^P_nu X^Q_lam X^Q_sig.
 *
 * Implementation:
 *   M[(mu,nu), P] = X^P_mu * X^P_nu     (N_bas^2 x N_g)
 *   T = V_flat * M                      (N_bas^2 x N_g)
 *   E = M^T * T                         (N_g    x N_g)
 *
 * @param X       Collocation [N_bas x N_g] column-major.
 * @param eri_4d  Full ERI tensor, length N_bas^4, indexed
 *                eri_4d[a + N_bas*(b + N_bas*(c + N_bas*d))] = (ab|cd).
 * @param N_bas, N_g  Dimensions.
 * @return        E [N_g x N_g] column-major.
 */
std::vector<real_t> build_E_from_eri_cpu(const std::vector<real_t>& X,
                                          const std::vector<real_t>& eri_4d,
                                          int N_bas, int N_g);

/**
 * @brief Compute Z = S^+ E S^+ via symmetric eigendecomposition of S.
 *
 *   S = U diag(sigma) U^T
 *   S^+ = sum_{sigma_i > cutoff * sigma_max} U_i sigma_i^{-1} U_i^T
 *
 * @param S         Symmetric PSD matrix [N_g x N_g] column-major.
 * @param E         Source matrix [N_g x N_g] column-major.
 * @param N_g       Dimension.
 * @param rel_cutoff Relative singular-value cutoff (default 1e-7).
 * @return Z        [N_g x N_g] column-major.
 */
std::vector<real_t> solve_Z_pinv_cpu(const std::vector<real_t>& S,
                                     const std::vector<real_t>& E,
                                     int N_g,
                                     double rel_cutoff = 1.0e-7,
                                     int* rank_out = nullptr,
                                     real_t* sigma_max_out = nullptr,
                                     real_t* sigma_min_kept_out = nullptr);

/**
 * @brief Compute Z = M^+ V_eri (M^+)^T directly via thin SVD of M.
 *
 * Uses
 *   M[(ab), P] = X^P_a * X^P_b      (N_bas^2 x N_g, "fat")
 *   M = U Sigma V^T                 (economy SVD; rank r <= N_bas^2)
 *   M^+ = V Sigma^{-1} U^T          (pseudo-inverse with cutoff)
 *   Z   = V Sigma^{-1} U^T V_eri U Sigma^{-1} V^T   (N_g x N_g)
 *
 * Far cheaper than the S = M^T M based path when N_g >> N_bas^2 because the
 * SVD is dominated by the smaller dimension (N_bas^2), giving an O(N_bas^4
 * N_g) cost instead of O(N_g^3).
 *
 * @param X         Collocation [N_bas x N_g] column-major.
 * @param eri_4d    Length N_bas^4, indexed eri[a + N_bas*(b + N_bas*(c + N_bas*d))].
 * @param N_bas, N_g
 * @param rel_cutoff Relative singular-value cutoff (default 1e-7).
 * @param rank_out, sigma_max_out, sigma_min_kept_out
 *                  Optional diagnostics (numerical rank, max sigma,
 *                  smallest sigma kept above cutoff).
 * @return Z       [N_g x N_g] column-major.
 */
std::vector<real_t> compute_Z_via_M_svd_cpu(const std::vector<real_t>& X,
                                            const std::vector<real_t>& eri_4d,
                                            int N_bas, int N_g,
                                            double rel_cutoff = 1.0e-7,
                                            int* rank_out = nullptr,
                                            real_t* sigma_max_out = nullptr,
                                            real_t* sigma_min_kept_out = nullptr);

/**
 * @brief Reconstruct the full 4-index ERI tensor from the THC factorisation:
 *        (mu nu | lam sig)_THC = sum_{PQ} X^P_mu X^P_nu Z_{PQ} X^Q_lam X^Q_sig.
 *
 * Done as M Z M^T where M[(mu,nu), P] = X^P_mu X^P_nu.
 *
 * @param X     Collocation [N_bas x N_g] column-major.
 * @param Z     Core matrix [N_g x N_g] column-major.
 * @param N_bas, N_g
 * @return      eri_thc, length N_bas^4, same indexing as build_E_from_eri_cpu's
 *              eri_4d argument.
 */
std::vector<real_t> reconstruct_eri_thc_cpu(const std::vector<real_t>& X,
                                            const std::vector<real_t>& Z,
                                            int N_bas, int N_g);

#ifndef GANSU_CPU_ONLY

/**
 * @brief GPU LS-THC pipeline (matches compute_Z_via_M_svd_cpu).
 *
 * Build M = X (Khatri-Rao), eigendecompose M M^T (small N_bas^2 x N_bas^2),
 * truncate by relative cutoff, assemble Z = V Sigma^{-1} U^T V_eri U Sigma^{-1} V^T.
 *
 * @param d_X        [N_bas x N_g] column-major collocation on device.
 * @param d_eri_4d   length N_bas^4 (column-major chemist notation) on device.
 * @param N_bas, N_g, rel_cutoff
 * @param rank_out, sigma_max_out, sigma_min_kept_out  Optional diagnostics.
 * @return unique_ptr to DeviceHostMatrix [N_g x N_g] (Z).
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_M_svd_gpu(const real_t* d_X,
                         const real_t* d_eri_4d,
                         int N_bas, int N_g,
                         double rel_cutoff = 1.0e-7,
                         int* rank_out = nullptr,
                         real_t* sigma_max_out = nullptr,
                         real_t* sigma_min_kept_out = nullptr);

/**
 * @brief GPU LS-THC pipeline using the RI 3-index AO tensor instead of the
 *        full analytic 4-index ERI.  Phase 2.3 (memory-light path).
 *
 * Replaces V_{(μν)(λσ)} with V_RI = B B^T, where B is the AO 3-index tensor
 *   B[(μν), R] = (μν|R) [V^{-1/2}]_{R,...}
 * already built by ERI_RI's intermediate_matrix_B_ (col-major (N_bas² × naux)
 * lda=N_bas²).
 *
 * Mathematically:
 *   Z = M^+ V_RI (M^+)^T = (M^+ B) (M^+ B)^T
 *
 * Avoids materialising the O(N_bas^4) ERI tensor; cost is O(N_bas² × N_g × naux).
 *
 * @param d_X        [N_bas × N_g] column-major collocation on device.
 * @param d_B_ao     [N_bas² × naux] column-major (lda=N_bas²) AO 3-index on device.
 * @param N_bas, N_g, naux
 * @param rel_cutoff, rank_out, sigma_max_out, sigma_min_kept_out
 *                   Same diagnostics as compute_Z_via_M_svd_gpu.
 * @return DeviceHostMatrix [N_g × N_g] (Z).
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_M_svd_ri_gpu(const real_t* d_X,
                            const real_t* d_B_ao,
                            int N_bas, int N_g, int naux,
                            double rel_cutoff = 1.0e-7,
                            int* rank_out = nullptr,
                            real_t* sigma_max_out = nullptr,
                            real_t* sigma_min_kept_out = nullptr);

/**
 * @brief Randomized-SVD variant of compute_Z_via_M_svd_ri_gpu (Phase 2.3 large-system).
 *
 * Replaces the full M·M^T eigendecomposition (which costs O(N_bas^4) memory) with
 * Halko–Martinsson–Tropp randomized SVD that finds the top @p max_rank singular
 * pairs of M directly, with peak memory O(N_bas^2 · max_rank).
 *
 * Algorithm (q = power-iteration steps, p = oversampling):
 *   1. Build M[(μν), P] = X^P_μ X^P_ν                     (N_bas² × N_g)
 *   2. Ω = N_g × (max_rank + p) random Gaussian
 *   3. Y = M Ω;  for q iters: Y = M (M^T Y)               (N_bas² × ko)
 *   4. Q = thin QR(Y)                                     (N_bas² × ko, orthonormal)
 *   5. B = Q^T M                                          (ko × N_g)
 *   6. C = B B^T   (small ko × ko symmetric)
 *   7. Eigendecompose C → top singular values σ² and U_C
 *   8. U_THC = Q U_C  (truncated by rel_cutoff)           (N_bas² × rank)
 *   9. RI-Z post-processing: T = Σ⁻¹ U_THC^T B_AO; K = T T^T; V = M^T U_THC Σ⁻¹;
 *                            Z = V K V^T
 *
 * Memory peak: ~N_bas² × max_rank doubles (for Q and U_THC).  For
 * N_bas² = 90000 and max_rank = 10000, that is 7.2 GB — orders of magnitude less
 * than the 65 GB the dense path would need.
 *
 * @param max_rank   Target maximum rank (must be ≤ min(N_bas², N_g)).  Choose
 *                   close to the actual LS-THC rank, padded by ~10–20%.
 * @param n_power_iter Number of power iterations for accuracy (q≥1 recommended;
 *                   default 2).  Each adds 2× M-matvec cost but improves the
 *                   spectral approximation quality.
 *
 * Other parameters and outputs match compute_Z_via_M_svd_ri_gpu().
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_rand_svd_ri_gpu(const real_t* d_X,
                               const real_t* d_B_ao,
                               int N_bas, int N_g, int naux,
                               int max_rank,
                               int n_power_iter = 2,
                               double rel_cutoff = 1.0e-7,
                               int* rank_out = nullptr,
                               real_t* sigma_max_out = nullptr,
                               real_t* sigma_min_kept_out = nullptr);

/**
 * @brief GPU reconstruction of the 4-index THC ERI tensor:
 *        (mu nu | lam sig)_THC = sum_{P,Q} X^P_mu X^P_nu Z_{P,Q} X^Q_lam X^Q_sig.
 *
 * Done as M Z M^T where M[(ab), P] = X^P_a * X^P_b (Khatri-Rao).
 *
 * @param d_X      [N_bas x N_g] column-major collocation on device.  Works in
 *                 either AO or MO basis (the function does not care; pass the
 *                 collocation matched to the chemist convention you want).
 * @param d_Z      [N_g x N_g] column-major core matrix on device.
 * @param N_bas, N_g
 * @return DeviceHostMatrix viewed as [N_bas^2 x N_bas^2] column-major (or
 *         length N_bas^4 flat) holding (ab|cd)_THC.
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
reconstruct_eri_thc_gpu(const real_t* d_X, const real_t* d_Z,
                         int N_bas, int N_g);

#endif // GANSU_CPU_ONLY

} // namespace gansu
