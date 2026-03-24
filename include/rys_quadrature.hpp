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

#ifndef RYS_QUADRATURE_HPP
#define RYS_QUADRATURE_HPP

#include "boys.hpp"

namespace gansu::gpu {

// ============================================================
//  Rys Quadrature: Compute roots (t_n^2) and weights (w_n)
//  from Boys function moments using the modified Chebyshev
//  algorithm + tridiagonal eigenvalue decomposition.
//
//  Reference: Dupuis, Rys, King, J. Chem. Phys. 65, 111 (1976)
// ============================================================

// --- N = 1: Analytical ---
inline __device__
void rysRoots1(double T, const double* g_boys_grid, double* roots, double* weights) {
    double F[2];
    getIncrementalBoys(1, T, g_boys_grid, F);
    // t^2 = F1(T) / F0(T), w = F0(T)
    roots[0] = (F[0] > 1e-300) ? F[1] / F[0] : 0.0;
    weights[0] = F[0];
}

// --- N = 2: Analytical 2x2 eigenvalue ---
inline __device__
void rysRoots2(double T, const double* g_boys_grid, double* roots, double* weights) {
    double F[4];
    getIncrementalBoys(3, T, g_boys_grid, F);

    // Modified Chebyshev algorithm for N=2
    // Moments: mu_k = F_k(T), k = 0..3
    double mu0 = F[0], mu1 = F[1], mu2 = F[2], mu3 = F[3];

    // alpha_0 = mu1/mu0
    double a0 = mu1 / mu0;

    // sigma_{1,l} = sigma_{0,l+1} - alpha_0 * sigma_{0,l}
    // sigma_{0,l} = mu_l
    // sigma_{1,1} = mu2 - a0*mu1
    // sigma_{1,2} = mu3 - a0*mu2
    double s11 = mu2 - a0 * mu1;
    double s12 = mu3 - a0 * mu2;

    // beta_1 = sigma_{1,1} / sigma_{0,0} = s11 / mu0
    double b1 = s11 / mu0;

    // alpha_1 = sigma_{1,2}/sigma_{1,1} - sigma_{0,1}/sigma_{0,0}
    //         = s12/s11 - mu1/mu0
    double a1 = s12 / s11 - a0;

    // Jacobi matrix: [[a0, sqrt(b1)], [sqrt(b1), a1]]
    // Eigenvalues: (a0+a1)/2 +/- sqrt(((a0-a1)/2)^2 + b1)
    double sum = a0 + a1;
    double diff = a0 - a1;
    double disc = sqrt(diff * diff * 0.25 + b1);
    roots[0] = 0.5 * sum - disc;
    roots[1] = 0.5 * sum + disc;

    // Weights from eigenvectors: w_n = mu0 * (v_n[0])^2
    // For 2x2: v1 = [cos(theta), sin(theta)], v2 = [-sin(theta), cos(theta)]
    // where tan(2*theta) = 2*sqrt(b1) / (a0-a1)
    if (fabs(diff) < 1e-15 && fabs(b1) < 1e-30) {
        weights[0] = 0.5 * mu0;
        weights[1] = 0.5 * mu0;
    } else {
        // v1[0]^2 = (a1 - roots[0]) / (roots[1] - roots[0])
        double denom = roots[1] - roots[0];
        if (fabs(denom) < 1e-300) {
            weights[0] = 0.5 * mu0;
            weights[1] = 0.5 * mu0;
        } else {
            double v1_0_sq = (a1 - roots[0]) / denom;
            double v2_0_sq = 1.0 - v1_0_sq;
            weights[0] = mu0 * v1_0_sq;
            weights[1] = mu0 * v2_0_sq;
        }
    }

    // Clamp roots to valid range [0, 1]
    if (roots[0] < 0.0) roots[0] = 0.0;
    if (roots[1] < 0.0) roots[1] = 0.0;
}

// --- N = 3..9: General tridiagonal eigenvalue solver ---
// Modified Chebyshev algorithm + implicit QR iteration
inline __device__
void rysRootsGeneral(int N, double T, const double* g_boys_grid, double* roots, double* weights) {
    // Compute Boys function values F_0..F_{2N-1}
    const int max_moments = 18; // 2*9 for N=9
    double F[18];
    getIncrementalBoys(2 * N - 1, T, g_boys_grid, F);

    // Modified Chebyshev algorithm to compute alpha_k, beta_k
    // Using two rows of sigma (cyclic buffer) + one previous row
    const int max_N = 9;
    double alpha[9], beta[9];

    // sigma storage: sigma[k%3][l], we need sigma for k-2, k-1, k
    // Maximum l index = 2*N-1
    double sigma_prev2[18]; // sigma_{k-2, l}
    double sigma_prev1[18]; // sigma_{k-1, l}
    double sigma_curr[18];  // sigma_{k, l}

    // Initialize: sigma_{0,l} = mu_l = F_l
    for (int l = 0; l < 2 * N; l++) {
        sigma_prev1[l] = F[l];
    }
    // sigma_{-1,l} = 0
    for (int l = 0; l < 2 * N; l++) {
        sigma_prev2[l] = 0.0;
    }

    alpha[0] = F[1] / F[0];
    beta[0] = F[0]; // mu_0, used for weight computation

    for (int k = 1; k < N; k++) {
        // sigma_{k,l} = sigma_{k-1,l+1} - alpha_{k-1}*sigma_{k-1,l} - beta_{k-1}*sigma_{k-2,l}
        // Note: beta[0] is mu_0 (special), beta[k] for k>=1 is the actual recurrence coefficient
        double beta_km1 = (k == 1) ? (sigma_prev1[1] - alpha[0] * sigma_prev1[0]) / F[0]
                                    : beta[k-1]; // already computed in previous iteration...
        // Actually need to compute sigma first, then extract alpha, beta

        for (int l = k; l < 2 * N - k; l++) {
            sigma_curr[l] = sigma_prev1[l + 1] - alpha[k - 1] * sigma_prev1[l];
            if (k >= 2) {
                sigma_curr[l] -= beta[k - 1] * sigma_prev2[l];
            }
        }

        alpha[k] = sigma_curr[k + 1] / sigma_curr[k] - sigma_prev1[k] / sigma_prev1[k - 1];
        beta[k] = sigma_curr[k] / sigma_prev1[k - 1];

        // Shift buffers
        for (int l = 0; l < 2 * N; l++) {
            sigma_prev2[l] = sigma_prev1[l];
            sigma_prev1[l] = sigma_curr[l];
        }
    }

    // Now we have the Jacobi matrix:
    //   diagonal: alpha[0..N-1]
    //   off-diagonal: sqrt(beta[1..N-1])
    // Solve for eigenvalues using implicit QR on tridiagonal matrix

    // Initialize eigenvalue arrays (diagonal and off-diagonal of Jacobi matrix)
    double diag[9], offdiag[9];
    for (int i = 0; i < N; i++) diag[i] = alpha[i];
    for (int i = 1; i < N; i++) {
        if (beta[i] < 0.0) beta[i] = 0.0; // numerical safety
        offdiag[i] = sqrt(beta[i]);
    }
    offdiag[0] = 0.0;

    // Track first component of each eigenvector for weight computation
    // Start with identity matrix, only track first row
    double evec_first[9];
    for (int i = 0; i < N; i++) evec_first[i] = (i == 0) ? 1.0 : 0.0;

    // Implicit QR iteration (Wilkinson shift)
    const int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        // Find the largest unreduced submatrix
        int m = N - 1;
        while (m > 0 && fabs(offdiag[m]) < 1e-15 * (fabs(diag[m - 1]) + fabs(diag[m]))) {
            m--;
        }
        if (m == 0) break; // all converged

        int l = m - 1;
        while (l > 0 && fabs(offdiag[l]) > 1e-15 * (fabs(diag[l - 1]) + fabs(diag[l]))) {
            l--;
        }

        // Wilkinson shift
        double d = (diag[m - 1] - diag[m]) * 0.5;
        double e2 = offdiag[m] * offdiag[m];
        double shift = diag[m] - e2 / (d + copysign(sqrt(d * d + e2), d));

        // Implicit QR step (Givens rotations)
        double g = diag[l] - shift;
        double s_rot = 1.0, c_rot = 1.0;
        double p_val = 0.0;

        for (int i = l; i < m; i++) {
            double f = s_rot * offdiag[i + 1];
            double b_val = c_rot * offdiag[i + 1];

            // Givens rotation
            if (fabs(f) >= fabs(g)) {
                c_rot = g / f;
                double r = sqrt(c_rot * c_rot + 1.0);
                offdiag[i + 1] = f * r; // was offdiag[i], but we track carefully
                s_rot = 1.0 / r;
                c_rot *= s_rot;
            } else {
                s_rot = f / g;
                double r = sqrt(s_rot * s_rot + 1.0);
                offdiag[i + 1] = g * r;
                c_rot = 1.0 / r;
                s_rot *= c_rot;
            }

            // We actually need proper implicit QR. Let me use the standard
            // tql2-style algorithm instead.
            // Break and use simpler approach below.
            break;
        }

        // Fall back to simpler QL iteration with implicit shift
        // This is the standard EISPACK tql1/tql2 algorithm
        break; // exit the QR loop, use QL below
    }

    // QL algorithm with implicit shifts (standard, robust)
    // Re-initialize from Jacobi matrix
    double d_ql[9], e_ql[9];
    for (int i = 0; i < N; i++) d_ql[i] = alpha[i];
    for (int i = 0; i < N - 1; i++) {
        if (beta[i + 1] < 0.0) beta[i + 1] = 0.0;
        e_ql[i] = sqrt(beta[i + 1]);
    }
    e_ql[N - 1] = 0.0;

    // Eigenvector first components (identity initially)
    double z[9];
    for (int i = 0; i < N; i++) z[i] = (i == 0) ? 1.0 : 0.0;

    for (int l_idx = 0; l_idx < N; l_idx++) {
        int iter_count = 0;
        int m_idx;
        do {
            // Find small off-diagonal element
            for (m_idx = l_idx; m_idx < N - 1; m_idx++) {
                double dd = fabs(d_ql[m_idx]) + fabs(d_ql[m_idx + 1]);
                if (fabs(e_ql[m_idx]) <= 1e-15 * dd) break;
            }
            if (m_idx != l_idx) {
                if (iter_count++ >= 60) break; // prevent infinite loop

                // Form shift
                double g_ql = (d_ql[l_idx + 1] - d_ql[l_idx]) / (2.0 * e_ql[l_idx]);
                double r_ql;
                if (fabs(g_ql) > 1.0)
                    r_ql = fabs(g_ql) * sqrt(1.0 + 1.0 / (g_ql * g_ql));
                else
                    r_ql = sqrt(1.0 + g_ql * g_ql);
                g_ql = d_ql[m_idx] - d_ql[l_idx] + e_ql[l_idx] / (g_ql + copysign(r_ql, g_ql));

                double s_ql = 1.0, c_ql = 1.0, p_ql = 0.0;
                for (int i = m_idx - 1; i >= l_idx; i--) {
                    double f_ql = s_ql * e_ql[i];
                    double b_ql = c_ql * e_ql[i];
                    if (fabs(f_ql) >= fabs(g_ql)) {
                        c_ql = g_ql / f_ql;
                        r_ql = sqrt(c_ql * c_ql + 1.0);
                        e_ql[i + 1] = f_ql * r_ql;
                        s_ql = 1.0 / r_ql;
                        c_ql *= s_ql;
                    } else {
                        s_ql = f_ql / g_ql;
                        r_ql = sqrt(s_ql * s_ql + 1.0);
                        e_ql[i + 1] = g_ql * r_ql;
                        c_ql = 1.0 / r_ql;
                        s_ql *= c_ql;
                    }
                    g_ql = d_ql[i + 1] - p_ql;
                    r_ql = (d_ql[i] - g_ql) * s_ql + 2.0 * c_ql * b_ql;
                    p_ql = s_ql * r_ql;
                    d_ql[i + 1] = g_ql + p_ql;
                    g_ql = c_ql * r_ql - b_ql;

                    // Track first-row eigenvector components
                    double f_z = z[i + 1];
                    z[i + 1] = s_ql * z[i] + c_ql * f_z;
                    z[i] = c_ql * z[i] - s_ql * f_z;
                }
                d_ql[l_idx] -= p_ql;
                e_ql[l_idx] = g_ql;
                e_ql[m_idx] = 0.0;
            }
        } while (m_idx != l_idx);
    }

    // Sort eigenvalues (roots) in ascending order, keeping z synchronized
    for (int i = 0; i < N - 1; i++) {
        int k_min = i;
        double d_min = d_ql[i];
        for (int j = i + 1; j < N; j++) {
            if (d_ql[j] < d_min) { k_min = j; d_min = d_ql[j]; }
        }
        if (k_min != i) {
            d_ql[k_min] = d_ql[i];
            d_ql[i] = d_min;
            double tmp = z[i]; z[i] = z[k_min]; z[k_min] = tmp;
        }
    }

    // Output
    for (int i = 0; i < N; i++) {
        roots[i] = (d_ql[i] > 0.0) ? d_ql[i] : 0.0;
        weights[i] = F[0] * z[i] * z[i]; // mu_0 * v_{n,0}^2
    }
}

// --- Dispatcher ---
inline __device__
void computeRysRootsAndWeights(int N, double T, const double* g_boys_grid,
                                double* roots, double* weights) {
    switch (N) {
        case 1: rysRoots1(T, g_boys_grid, roots, weights); break;
        case 2: rysRoots2(T, g_boys_grid, roots, weights); break;
        default: rysRootsGeneral(N, T, g_boys_grid, roots, weights); break;
    }
}

} // namespace gansu::gpu

#endif // RYS_QUADRATURE_HPP
