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
 * @file cpu_integrals.hpp
 * @brief CPU-only integral computation engine (GPU-free fallback)
 *
 * Implements 1-electron (overlap, kinetic, nuclear attraction) and
 * 2-electron (ERI) integrals using McMurchie-Davidson / Obara-Saika
 * recursion on the CPU. Supports angular momentum s through f (L=0..3).
 *
 * Normalization convention matches the GPU kernels:
 *   - Each primitive carries a contraction coefficient (shell.coefficient)
 *   - Per-component CGTO normalization is stored in cgto_norms[basis_index + component]
 *   - Primitive Gaussian normalization N(alpha,l,m,n) is computed inline
 *   - Final integral = sum_primitives coeff_a * coeff_b * N_a * N_b * cgto_a * cgto_b * raw_integral
 */

#pragma once

#include "types.hpp"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gansu::cpu {

// ============================================================
//  Utility: double-factorial, distance, primitive norm
// ============================================================

/// Double factorial n!! with convention (-1)!! = 1, 0!! = 1
inline double double_factorial(int n) {
    if (n <= 1) return 1.0;
    double result = 1.0;
    for (int i = n; i >= 2; i -= 2) result *= i;
    return result;
}

/// Squared distance |A-B|^2
inline double dist2(const Coordinate& A, const Coordinate& B) {
    double dx = A.x - B.x, dy = A.y - B.y, dz = A.z - B.z;
    return dx * dx + dy * dy + dz * dz;
}

/**
 * Primitive Gaussian normalization factor N(alpha, l, m, n).
 * Matches the GPU calcNorm:
 *   N = 2^(l+m+n) / sqrt( (2l-1)!! (2m-1)!! (2n-1)!! ) * (pi)^{-3/4} * (2*alpha)^{(2L+3)/4}
 * Rewritten:
 *   N = (2/pi)^{3/4} * (4*alpha)^{(l+m+n)/2} * alpha^{3/4} / sqrt((2l-1)!!(2m-1)!!(2n-1)!!)
 */
inline double primitive_norm(double alpha, int l, int m, int n) {
    int L = l + m + n;
    return std::pow(2.0 / M_PI, 0.75)
         * std::pow(4.0 * alpha, L / 2.0)
         * std::pow(alpha, 0.75)
         / std::sqrt(double_factorial(2 * l - 1) * double_factorial(2 * m - 1) * double_factorial(2 * n - 1));
}

// ============================================================
//  Boys function  F_n(x) = int_0^1 t^{2n} exp(-x t^2) dt
// ============================================================

/**
 * Compute Boys function F_n(x) for n = 0, 1, ..., nmax.
 * Uses downward recursion from a high-order series for stability,
 * with the asymptotic formula for large x.
 */
inline void boys_function_array(int nmax, double x, double* F) {
    const double THRESHOLD = 30.0;

    if (x < 1.0e-15) {
        // F_n(0) = 1/(2n+1)
        for (int n = 0; n <= nmax; n++)
            F[n] = 1.0 / (2.0 * n + 1.0);
        return;
    }

    if (x > THRESHOLD) {
        // Asymptotic: F_0(x) ~ sqrt(pi/4x), F_n = (2n-1)/(2x) * F_{n-1}
        F[0] = 0.5 * std::sqrt(M_PI / x);
        for (int n = 1; n <= nmax; n++)
            F[n] = F[n - 1] * (2.0 * n - 1.0) / (2.0 * x);
        return;
    }

    // Series expansion for F_{nmax+20}, then downward recursion.
    // F_N(x) = exp(-x) * sum_{k=0}^{inf} (2x)^k / (2N+2k+1)!!
    // which converges for moderate x. We start from a high N for stability.
    int extra = 25;
    int Nstart = nmax + extra;
    double Fhigh = 0.0;
    {
        // Series for F_{Nstart}(x)
        double term = 1.0 / (2.0 * Nstart + 1.0);
        double sum = term;
        for (int k = 1; k <= 200; k++) {
            term *= x / (Nstart + k + 0.5);
            sum += term;
            if (std::abs(term) < 1.0e-16 * std::abs(sum)) break;
        }
        Fhigh = sum * std::exp(-x);
    }

    // Downward recursion: F_{n-1}(x) = (2x F_n(x) + exp(-x)) / (2n-1)
    double expx = std::exp(-x);
    double Fn = Fhigh;
    for (int n = Nstart; n >= 1; n--) {
        double Fn_minus_1 = (2.0 * x * Fn + expx) / (2.0 * n - 1.0);
        if (n - 1 <= nmax) F[n - 1] = Fn_minus_1;
        if (n <= nmax) { /* F[n] already set or will be set */ }
        Fn = Fn_minus_1;
    }
    // Fill remaining using upward recursion from F[0] for better accuracy at low n
    // Actually, downward recursion already filled F[0..nmax]. But we stored them
    // in the wrong direction above. Let's redo properly.

    // Redo: store Fn for all n from Nstart down to 0
    Fn = Fhigh;
    std::vector<double> Fbuf(Nstart + 1);
    Fbuf[Nstart] = Fhigh;
    for (int n = Nstart; n >= 1; n--) {
        Fbuf[n - 1] = (2.0 * x * Fbuf[n] + expx) / (2.0 * n - 1.0);
    }
    for (int n = 0; n <= nmax; n++) F[n] = Fbuf[n];
}

/// Single Boys function value
inline double boys_function(int n, double x) {
    std::vector<double> F(n + 1);
    boys_function_array(n, x, F.data());
    return F[n];
}

// ============================================================
//  McMurchie-Davidson E^t_{i,j} expansion coefficients
// ============================================================

/**
 * Compute E^t_{i,j}(a, b, XAB) for all t from 0 to i+j.
 * Uses the recursion:
 *   E^0_{0,0} = 1
 *   E^t_{i+1,j} = (1/(2p)) E^{t-1}_{i,j} + X_PA E^t_{i,j} + (t+1) E^{t+1}_{i,j}
 *   E^t_{i,j+1} = (1/(2p)) E^{t-1}_{i,j} + X_PB E^t_{i,j} + (t+1) E^{t+1}_{i,j}
 *
 * where p = a + b, P = (a*A + b*B)/p, X_PA = Px - Ax, X_PB = Px - Bx.
 *
 * Returns the full array E[i][j][t] flattened. Access via E_idx(i,j,t).
 * Dimensions: (imax+1) x (jmax+1) x (imax+jmax+1)
 */

// Maximum angular momentum supported (f = 3)
static constexpr int MAX_AM = 3;
// Maximum combined angular momentum for one coordinate (needed for kinetic: l+2)
static constexpr int MAX_AM_PLUS = MAX_AM + 2; // 5
static constexpr int E_DIM1 = MAX_AM_PLUS + 1;  // 6
static constexpr int E_DIM2 = MAX_AM_PLUS + 1;  // 6
static constexpr int E_DIM3 = 2 * MAX_AM_PLUS + 1; // 11

inline int E_idx(int i, int j, int t) {
    return i * E_DIM2 * E_DIM3 + j * E_DIM3 + t;
}

/**
 * Fill the E coefficient table for one Cartesian direction.
 * a, b: exponents; XAB = Ax - Bx (center difference)
 * imax, jmax: maximum angular momentum in this direction
 * E: output buffer of size E_DIM1 * E_DIM2 * E_DIM3
 */
inline void compute_E_coefficients(double a, double b, double XAB,
                                   int imax, int jmax, double* E) {
    double p = a + b;
    double mu = a * b / p;
    double XPA = -b * XAB / p;  // P - A = b*(B-A)/p = -b*XAB/p
    double XPB = a * XAB / p;   // P - B = a*(A-B)/p = a*XAB/p
    double one_over_2p = 0.5 / p;

    int tmax = imax + jmax;
    int total = E_DIM1 * E_DIM2 * E_DIM3;
    std::memset(E, 0, total * sizeof(double));

    // Base case
    E[E_idx(0, 0, 0)] = 1.0;

    // First build up i with j=0 using the X_PA recursion:
    //   E^t_{i+1, 0} = (1/(2p)) E^{t-1}_{i,0} + X_PA E^t_{i,0} + (t+1) E^{t+1}_{i,0}
    for (int i = 0; i < imax; i++) {
        for (int t = 0; t <= i + 1; t++) {
            double val = XPA * E[E_idx(i, 0, t)];
            if (t >= 1) val += one_over_2p * E[E_idx(i, 0, t - 1)];
            if (t + 1 <= i) val += (t + 1) * E[E_idx(i, 0, t + 1)];
            E[E_idx(i + 1, 0, t)] += val;
        }
    }

    // Then for each i, build up j using the X_PB recursion:
    //   E^t_{i, j+1} = (1/(2p)) E^{t-1}_{i,j} + X_PB E^t_{i,j} + (t+1) E^{t+1}_{i,j}
    for (int i = 0; i <= imax; i++) {
        for (int j = 0; j < jmax; j++) {
            int tmax_ij = i + j;
            for (int t = 0; t <= tmax_ij + 1; t++) {
                double val = XPB * E[E_idx(i, j, t)];
                if (t >= 1) val += one_over_2p * E[E_idx(i, j, t - 1)];
                if (t + 1 <= tmax_ij) val += (t + 1) * E[E_idx(i, j, t + 1)];
                E[E_idx(i, j + 1, t)] += val;
            }
        }
    }
}

/// Get E^t_{i,j} from precomputed table
inline double get_E(const double* E, int i, int j, int t) {
    return E[E_idx(i, j, t)];
}

// ============================================================
//  Hermite Coulomb integrals R^m_{t,u,v}
// ============================================================

/**
 * Compute R^m_{t,u,v}(p, RPC) for nuclear attraction integrals.
 *
 * R^m_{0,0,0} = (-2p)^m F_m(p |PC|^2)
 *
 * Recursion (McMurchie-Davidson):
 *   R^m_{t+1,u,v} = t R^{m+1}_{t-1,u,v} + X_PC R^{m+1}_{t,u,v}
 *   R^m_{t,u+1,v} = u R^{m+1}_{t,u-1,v} + Y_PC R^{m+1}_{t,u,v}
 *   R^m_{t,u,v+1} = v R^{m+1}_{t,u,v-1} + Z_PC R^{m+1}_{t,u,v}
 *
 * We need R^0_{t,u,v} for t=0..tmax, u=0..umax, v=0..vmax.
 * Total Boys order needed: m_max = tmax + umax + vmax.
 */

/// Index into R buffer: R[m][t][u][v]
/// Dimensions: (mmax+1) x (tmax+1) x (umax+1) x (vmax+1)
struct R_indexer {
    int tmax, umax, vmax;
    int stride_m, stride_t, stride_u;

    R_indexer(int tmax_, int umax_, int vmax_)
        : tmax(tmax_), umax(umax_), vmax(vmax_) {
        stride_u = vmax + 1;
        stride_t = (umax + 1) * stride_u;
        stride_m = (tmax + 1) * stride_t;
    }

    int operator()(int m, int t, int u, int v) const {
        return m * stride_m + t * stride_t + u * stride_u + v;
    }

    int total(int mmax) const { return (mmax + 1) * stride_m; }
};

/**
 * Compute all R^m_{t,u,v} needed for a nuclear attraction or ERI integral.
 * p: total exponent (a+b for nuclear, or composite for ERI)
 * RPC: vector P - C (P = Gaussian product center, C = nucleus or second pair center)
 * tmax, umax, vmax: maximum Hermite indices
 *
 * Output: R array indexed by R_indexer.
 */
inline void compute_R_integrals(double p, double XPC, double YPC, double ZPC,
                                int tmax, int umax, int vmax, double* R) {
    int mmax = tmax + umax + vmax;
    R_indexer idx(tmax, umax, vmax);

    double T = p * (XPC * XPC + YPC * YPC + ZPC * ZPC);
    std::vector<double> Fm(mmax + 1);
    boys_function_array(mmax, T, Fm.data());

    // Initialize R^m_{0,0,0} = (-2p)^m * F_m(T)
    double neg2p_power = 1.0;
    for (int m = 0; m <= mmax; m++) {
        R[idx(m, 0, 0, 0)] = neg2p_power * Fm[m];
        neg2p_power *= -2.0 * p;
    }

    // Build R^m_{t,u,v} by increasing t+u+v one step at a time.
    // Each recursion step increases one of (t,u,v) by 1 while also
    // increasing m by 1, so we must process in order of decreasing m
    // AND we must ensure that R^{m+1}_{...} values needed are already available.
    //
    // Strategy: process in order of total Hermite order N = t+u+v,
    // building N=1, then N=2, etc. For each target (t,u,v) with t+u+v = N,
    // we need R^{m+1} at level N-1, which was built in the previous iteration.
    // Within each N level, we process m from high to low.
    for (int N = 1; N <= tmax + umax + vmax; N++) {
        for (int t = std::min(N, tmax); t >= 0; t--) {
            for (int u = std::min(N - t, umax); u >= 0; u--) {
                int v = N - t - u;
                if (v < 0 || v > vmax) continue;

                // We need to compute R^m_{t,u,v} for m = 0 .. mmax - N
                // using R^{m+1} at level N-1
                for (int m = mmax - N; m >= 0; m--) {
                    double val;
                    if (t > 0) {
                        // Use t-recursion: R^m_{t,u,v} = (t-1) R^{m+1}_{t-2,u,v} + XPC R^{m+1}_{t-1,u,v}
                        val = XPC * R[idx(m + 1, t - 1, u, v)];
                        if (t >= 2) val += (t - 1) * R[idx(m + 1, t - 2, u, v)];
                    } else if (u > 0) {
                        // Use u-recursion: R^m_{0,u,v} = (u-1) R^{m+1}_{0,u-2,v} + YPC R^{m+1}_{0,u-1,v}
                        val = YPC * R[idx(m + 1, 0, u - 1, v)];
                        if (u >= 2) val += (u - 1) * R[idx(m + 1, 0, u - 2, v)];
                    } else {
                        // Use v-recursion: R^m_{0,0,v} = (v-1) R^{m+1}_{0,0,v-2} + ZPC R^{m+1}_{0,0,v-1}
                        val = ZPC * R[idx(m + 1, 0, 0, v - 1)];
                        if (v >= 2) val += (v - 1) * R[idx(m + 1, 0, 0, v - 2)];
                    }
                    R[idx(m, t, u, v)] = val;
                }
            }
        }
    }
}


// ============================================================
//  Primitive integral routines
// ============================================================

/**
 * Compute overlap integral between two primitive Gaussians.
 * a, b: exponents
 * A, B: centers
 * la,ma,na / lb,mb,nb: angular momentum components
 * Returns the raw (unnormalized) integral.
 */
inline double overlap_primitive(double a, double b,
                                const Coordinate& A, const Coordinate& B,
                                int la, int ma, int na,
                                int lb, int mb, int nb) {
    double p = a + b;
    double mu = a * b / p;
    double AB2 = dist2(A, B);
    double prefactor = std::pow(M_PI / p, 1.5) * std::exp(-mu * AB2);

    // Compute E coefficients for each direction
    double Ex[E_DIM1 * E_DIM2 * E_DIM3];
    double Ey[E_DIM1 * E_DIM2 * E_DIM3];
    double Ez[E_DIM1 * E_DIM2 * E_DIM3];

    compute_E_coefficients(a, b, A.x - B.x, la, lb, Ex);
    compute_E_coefficients(a, b, A.y - B.y, ma, mb, Ey);
    compute_E_coefficients(a, b, A.z - B.z, na, nb, Ez);

    return prefactor * get_E(Ex, la, lb, 0) * get_E(Ey, ma, mb, 0) * get_E(Ez, na, nb, 0);
}

/**
 * Compute kinetic energy integral between two primitive Gaussians.
 * Uses the relation:
 *   T = b(2(lx2+ly2+lz2)+3) S(a,b)
 *       - 2b^2 [S(l2+2,x) S(y) S(z) + S(x) S(l2+2,y) S(z) + S(x) S(y) S(l2+2,z)]
 *       + (1/2) [l2(l2-1) S(l2-2,x) S(y) S(z) + ...]
 *
 * More precisely (McMurchie-Davidson form):
 *   T_x = -2 b^2 E^0_{la, lb+2} + b(2lb+1) E^0_{la, lb} - lb(lb-1)/2 E^0_{la, lb-2}
 * and T = prefactor * (T_x * Ey0 * Ez0 + Ex0 * T_y * Ez0 + Ex0 * Ey0 * T_z)
 */
inline double kinetic_primitive(double a, double b,
                                const Coordinate& A, const Coordinate& B,
                                int la, int ma, int na,
                                int lb, int mb, int nb) {
    double p = a + b;
    double mu = a * b / p;
    double AB2 = dist2(A, B);
    double prefactor = std::pow(M_PI / p, 1.5) * std::exp(-mu * AB2);

    // We need E coefficients up to lb+2 (or mb+2, nb+2) for kinetic
    double Ex[E_DIM1 * E_DIM2 * E_DIM3];
    double Ey[E_DIM1 * E_DIM2 * E_DIM3];
    double Ez[E_DIM1 * E_DIM2 * E_DIM3];

    compute_E_coefficients(a, b, A.x - B.x, la, lb + 2, Ex);
    compute_E_coefficients(a, b, A.y - B.y, ma, mb + 2, Ey);
    compute_E_coefficients(a, b, A.z - B.z, na, nb + 2, Ez);

    double Ex0 = get_E(Ex, la, lb, 0);
    double Ey0 = get_E(Ey, ma, mb, 0);
    double Ez0 = get_E(Ez, na, nb, 0);

    // T_x component
    double Tx = -2.0 * b * b * get_E(Ex, la, lb + 2, 0)
                + b * (2 * lb + 1) * Ex0;
    if (lb >= 2) Tx -= 0.5 * lb * (lb - 1) * get_E(Ex, la, lb - 2, 0);

    // T_y component
    double Ty = -2.0 * b * b * get_E(Ey, ma, mb + 2, 0)
                + b * (2 * mb + 1) * Ey0;
    if (mb >= 2) Ty -= 0.5 * mb * (mb - 1) * get_E(Ey, ma, mb - 2, 0);

    // T_z component
    double Tz = -2.0 * b * b * get_E(Ez, na, nb + 2, 0)
                + b * (2 * nb + 1) * Ez0;
    if (nb >= 2) Tz -= 0.5 * nb * (nb - 1) * get_E(Ez, na, nb - 2, 0);

    return prefactor * (Tx * Ey0 * Ez0 + Ex0 * Ty * Ez0 + Ex0 * Ey0 * Tz);
}

/**
 * Compute nuclear attraction integral for one nucleus.
 * V = -Z * (2 pi / p) * exp(-mu |AB|^2) * sum_{t,u,v} E^t_x E^u_y E^v_z R^0_{t,u,v}
 */
inline double nuclear_attraction_primitive(double a, double b,
                                           const Coordinate& A, const Coordinate& B,
                                           int la, int ma, int na,
                                           int lb, int mb, int nb,
                                           const Coordinate& C, int Z) {
    double p = a + b;
    double mu = a * b / p;
    double AB2 = dist2(A, B);
    double prefactor = (2.0 * M_PI / p) * std::exp(-mu * AB2);

    // Gaussian product center P
    double Px = (a * A.x + b * B.x) / p;
    double Py = (a * A.y + b * B.y) / p;
    double Pz = (a * A.z + b * B.z) / p;

    // E coefficients
    double Ex[E_DIM1 * E_DIM2 * E_DIM3];
    double Ey[E_DIM1 * E_DIM2 * E_DIM3];
    double Ez[E_DIM1 * E_DIM2 * E_DIM3];

    compute_E_coefficients(a, b, A.x - B.x, la, lb, Ex);
    compute_E_coefficients(a, b, A.y - B.y, ma, mb, Ey);
    compute_E_coefficients(a, b, A.z - B.z, na, nb, Ez);

    // R integrals
    int tmax = la + lb;
    int umax = ma + mb;
    int vmax = na + nb;
    int mmax = tmax + umax + vmax;
    R_indexer ridx(tmax, umax, vmax);
    std::vector<double> R(ridx.total(mmax), 0.0);

    compute_R_integrals(p, Px - C.x, Py - C.y, Pz - C.z, tmax, umax, vmax, R.data());

    // Contract E * R
    double result = 0.0;
    for (int t = 0; t <= tmax; t++) {
        double Et = get_E(Ex, la, lb, t);
        for (int u = 0; u <= umax; u++) {
            double Eu = get_E(Ey, ma, mb, u);
            for (int v = 0; v <= vmax; v++) {
                double Ev = get_E(Ez, na, nb, v);
                result += Et * Eu * Ev * R[ridx(0, t, u, v)];
            }
        }
    }

    return -Z * prefactor * result;
}

/**
 * Compute ERI (mu nu | la si) for four primitive Gaussians using McMurchie-Davidson.
 *
 * (ab|cd) = (2 pi^{5/2}) / (p q sqrt(p+q)) * exp(-mu_ab |AB|^2 - mu_cd |CD|^2)
 *           * sum_{t,u,v} E^t_ab E^u_ab E^v_ab * sum_{t',u',v'} E^{t'}_cd E^{u'}_cd E^{v'}_cd
 *             * (-1)^{t'+u'+v'} R^0_{t+t', u+u', v+v'}
 *
 * where p = a+b, q = c+d, alpha = pq/(p+q)
 */
inline double eri_primitive(double a, double b, double c, double d,
                            const Coordinate& A, const Coordinate& B,
                            const Coordinate& C, const Coordinate& D,
                            int la, int ma, int na,
                            int lb, int mb, int nb,
                            int lc, int mc, int nc,
                            int ld, int md, int nd) {
    double p = a + b;
    double q = c + d;
    double alpha = p * q / (p + q);

    double mu_ab = a * b / p;
    double mu_cd = c * d / q;
    double AB2 = dist2(A, B);
    double CD2 = dist2(C, D);

    double prefactor = 2.0 * std::pow(M_PI, 2.5) / (p * q * std::sqrt(p + q))
                     * std::exp(-mu_ab * AB2 - mu_cd * CD2);

    // Product centers
    double Px = (a * A.x + b * B.x) / p;
    double Py = (a * A.y + b * B.y) / p;
    double Pz = (a * A.z + b * B.z) / p;
    double Qx = (c * C.x + d * D.x) / q;
    double Qy = (c * C.y + d * D.y) / q;
    double Qz = (c * C.z + d * D.z) / q;

    // E coefficients for bra (ab) and ket (cd)
    int imax_ab = la + lb, jmax_ab = ma + mb, kmax_ab = na + nb;
    int imax_cd = lc + ld, jmax_cd = mc + md, kmax_cd = nc + nd;

    double Eab_x[E_DIM1 * E_DIM2 * E_DIM3];
    double Eab_y[E_DIM1 * E_DIM2 * E_DIM3];
    double Eab_z[E_DIM1 * E_DIM2 * E_DIM3];
    double Ecd_x[E_DIM1 * E_DIM2 * E_DIM3];
    double Ecd_y[E_DIM1 * E_DIM2 * E_DIM3];
    double Ecd_z[E_DIM1 * E_DIM2 * E_DIM3];

    compute_E_coefficients(a, b, A.x - B.x, la, lb, Eab_x);
    compute_E_coefficients(a, b, A.y - B.y, ma, mb, Eab_y);
    compute_E_coefficients(a, b, A.z - B.z, na, nb, Eab_z);
    compute_E_coefficients(c, d, C.x - D.x, lc, ld, Ecd_x);
    compute_E_coefficients(c, d, C.y - D.y, mc, md, Ecd_y);
    compute_E_coefficients(c, d, C.z - D.z, nc, nd, Ecd_z);

    // R integrals over P-Q with exponent alpha = pq/(p+q)
    int tmax = imax_ab + imax_cd;
    int umax = jmax_ab + jmax_cd;
    int vmax = kmax_ab + kmax_cd;
    R_indexer ridx(tmax, umax, vmax);
    int mmax = tmax + umax + vmax;
    std::vector<double> R(ridx.total(mmax), 0.0);

    compute_R_integrals(alpha, Px - Qx, Py - Qy, Pz - Qz, tmax, umax, vmax, R.data());

    // Contract
    double result = 0.0;
    for (int t1 = 0; t1 <= imax_ab; t1++) {
        double Et1 = get_E(Eab_x, la, lb, t1);
        for (int u1 = 0; u1 <= jmax_ab; u1++) {
            double Eu1 = get_E(Eab_y, ma, mb, u1);
            for (int v1 = 0; v1 <= kmax_ab; v1++) {
                double Ev1 = get_E(Eab_z, na, nb, v1);
                double bra = Et1 * Eu1 * Ev1;

                for (int t2 = 0; t2 <= imax_cd; t2++) {
                    double Et2 = get_E(Ecd_x, lc, ld, t2);
                    for (int u2 = 0; u2 <= jmax_cd; u2++) {
                        double Eu2 = get_E(Ecd_y, mc, md, u2);
                        for (int v2 = 0; v2 <= kmax_cd; v2++) {
                            double Ev2 = get_E(Ecd_z, nc, nd, v2);
                            double sign = ((t2 + u2 + v2) % 2 == 0) ? 1.0 : -1.0;
                            result += bra * sign * Et2 * Eu2 * Ev2
                                    * R[ridx(0, t1 + t2, u1 + u2, v1 + v2)];
                        }
                    }
                }
            }
        }
    }

    return prefactor * result;
}


// ============================================================
//  Contracted integral routines
// ============================================================

/**
 * Compute contracted overlap integral S_{mu,nu} between two contracted shells.
 * Sums over all primitive pairs sharing the same basis_index.
 *
 * @param shells     Full primitive shell array (sorted by shell_type)
 * @param cgto_norms Per-basis normalization factors
 * @param prims_a    Indices into shells[] for the primitives of shell A
 * @param prims_b    Indices into shells[] for the primitives of shell B
 * @param la,ma,na   Angular momentum of component for basis function mu
 * @param lb,mb,nb   Angular momentum of component for basis function nu
 */
inline double overlap_contracted(const PrimitiveShell* shells, const real_t* cgto_norms,
                                 const std::vector<size_t>& prims_a,
                                 const std::vector<size_t>& prims_b,
                                 int la, int ma, int na,
                                 int lb, int mb, int nb) {
    double result = 0.0;
    for (size_t ia : prims_a) {
        const auto& sa = shells[ia];
        double Na = primitive_norm(sa.exponent, la, ma, na);
        for (size_t ib : prims_b) {
            const auto& sb = shells[ib];
            double Nb = primitive_norm(sb.exponent, lb, mb, nb);
            result += sa.coefficient * sb.coefficient * Na * Nb
                    * overlap_primitive(sa.exponent, sb.exponent,
                                        sa.coordinate, sb.coordinate,
                                        la, ma, na, lb, mb, nb);
        }
    }
    return result;
}

/// Contracted kinetic energy integral
inline double kinetic_contracted(const PrimitiveShell* shells, const real_t* cgto_norms,
                                 const std::vector<size_t>& prims_a,
                                 const std::vector<size_t>& prims_b,
                                 int la, int ma, int na,
                                 int lb, int mb, int nb) {
    double result = 0.0;
    for (size_t ia : prims_a) {
        const auto& sa = shells[ia];
        double Na = primitive_norm(sa.exponent, la, ma, na);
        for (size_t ib : prims_b) {
            const auto& sb = shells[ib];
            double Nb = primitive_norm(sb.exponent, lb, mb, nb);
            result += sa.coefficient * sb.coefficient * Na * Nb
                    * kinetic_primitive(sa.exponent, sb.exponent,
                                        sa.coordinate, sb.coordinate,
                                        la, ma, na, lb, mb, nb);
        }
    }
    return result;
}

/// Contracted nuclear attraction integral (sum over all atoms)
inline double nuclear_contracted(const PrimitiveShell* shells, const real_t* cgto_norms,
                                 const std::vector<size_t>& prims_a,
                                 const std::vector<size_t>& prims_b,
                                 int la, int ma, int na,
                                 int lb, int mb, int nb,
                                 const Atom* atoms, int num_atoms) {
    double result = 0.0;
    for (size_t ia : prims_a) {
        const auto& sa = shells[ia];
        double Na = primitive_norm(sa.exponent, la, ma, na);
        for (size_t ib : prims_b) {
            const auto& sb = shells[ib];
            double Nb = primitive_norm(sb.exponent, lb, mb, nb);
            double prim_result = 0.0;
            for (int at = 0; at < num_atoms; at++) {
                prim_result += nuclear_attraction_primitive(
                    sa.exponent, sb.exponent,
                    sa.coordinate, sb.coordinate,
                    la, ma, na, lb, mb, nb,
                    atoms[at].coordinate, atoms[at].atomic_number);
            }
            result += sa.coefficient * sb.coefficient * Na * Nb * prim_result;
        }
    }
    return result;
}

/// Contracted ERI (mu nu | la si)
inline double eri_contracted(const PrimitiveShell* shells,
                             const std::vector<size_t>& prims_a,
                             const std::vector<size_t>& prims_b,
                             const std::vector<size_t>& prims_c,
                             const std::vector<size_t>& prims_d,
                             int la, int ma, int na,
                             int lb, int mb, int nb,
                             int lc, int mc, int nc,
                             int ld, int md, int nd) {
    double result = 0.0;
    for (size_t ia : prims_a) {
        const auto& sa = shells[ia];
        double Na = primitive_norm(sa.exponent, la, ma, na);
        for (size_t ib : prims_b) {
            const auto& sb = shells[ib];
            double Nb = primitive_norm(sb.exponent, lb, mb, nb);
            for (size_t ic : prims_c) {
                const auto& sc = shells[ic];
                double Nc = primitive_norm(sc.exponent, lc, mc, nc);
                for (size_t id : prims_d) {
                    const auto& sd = shells[id];
                    double Nd = primitive_norm(sd.exponent, ld, md, nd);
                    result += sa.coefficient * sb.coefficient
                            * sc.coefficient * sd.coefficient
                            * Na * Nb * Nc * Nd
                            * eri_primitive(sa.exponent, sb.exponent,
                                            sc.exponent, sd.exponent,
                                            sa.coordinate, sb.coordinate,
                                            sc.coordinate, sd.coordinate,
                                            la, ma, na, lb, mb, nb,
                                            lc, mc, nc, ld, md, nd);
                }
            }
        }
    }
    return result;
}


// ============================================================
//  Shell grouping helper
// ============================================================

/**
 * Group primitive shells by (basis_index, shell_type) to identify contracted shells.
 * Returns a vector of {basis_index, shell_type, [primitive indices]}.
 */
struct ContractedShell {
    size_t basis_index;
    int shell_type;
    std::vector<size_t> primitive_indices;
};

inline std::vector<ContractedShell> group_contracted_shells(
    const PrimitiveShell* shells, int num_shells) {

    std::vector<ContractedShell> result;

    for (int i = 0; i < num_shells; i++) {
        // Check if this primitive belongs to an existing contracted shell
        bool found = false;
        for (auto& cs : result) {
            if (cs.basis_index == shells[i].basis_index && cs.shell_type == shells[i].shell_type) {
                cs.primitive_indices.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            ContractedShell cs;
            cs.basis_index = shells[i].basis_index;
            cs.shell_type = shells[i].shell_type;
            cs.primitive_indices.push_back(i);
            result.push_back(cs);
        }
    }
    return result;
}

/// Total number of primitive shells across all shell types
inline int total_num_shells(const std::vector<ShellTypeInfo>& shell_type_infos) {
    int total = 0;
    for (const auto& sti : shell_type_infos)
        total += sti.count;
    return total;
}


// ============================================================
//  8-fold symmetry ERI indexing (CPU version of get_1d_index)
// ============================================================

/// Triangular index: map (i, j) with i <= j to linear index in upper triangle of size n
inline size_t tri_index(size_t i, size_t j, size_t n) {
    // j - i*(i - 2*n + 1) / 2, matching GPU get_index_2to1
    return j - static_cast<size_t>(i * (i - 2 * static_cast<long long>(n) + 1) / 2);
}

/// Sort ERI indices for 8-fold symmetry: (ij|kl) with i<=j, k<=l, ij<=kl
inline void sort_eri_indices(size_t& a, size_t& b, size_t& c, size_t& d) {
    if (a > b) std::swap(a, b);
    if (c > d) std::swap(c, d);
    size_t ab = tri_index(a, b, 0); // dummy n, just for comparison
    size_t cd = tri_index(c, d, 0);
    // Actually need to compare bra vs ket with real n, but we can compare (a,b) vs (c,d)
    // The GPU code uses: if(a > c || (a == c && b > d)) swap bra/ket
    if (a > c || (a == c && b > d)) {
        std::swap(a, c);
        std::swap(b, d);
    }
}

/// Compute 1D index for ERI with 8-fold symmetry, matching GPU get_1d_index
inline size_t get_1d_index(size_t i, size_t j, size_t k, size_t l, size_t num_basis) {
    // Sort indices
    if (i > j) std::swap(i, j);
    if (k > l) std::swap(k, l);
    if (i > k || (i == k && j > l)) {
        std::swap(i, k);
        std::swap(j, l);
    }
    size_t bra = tri_index(i, j, num_basis);
    size_t ket = tri_index(k, l, num_basis);
    size_t npairs = static_cast<size_t>(num_basis) * (num_basis + 1) / 2;
    return tri_index(bra, ket, npairs);
}


// ============================================================
//  Driver: Core Hamiltonian (S, T, V)
// ============================================================

/**
 * Compute the overlap matrix S and core Hamiltonian H = T + V.
 * Matrices are stored in row-major order, size num_basis x num_basis.
 * Upper triangle is computed first, then symmetrized.
 *
 * The normalization matches the GPU code:
 *   integral_value = cgto_norms[mu] * cgto_norms[nu]
 *                  * sum_{prims} coeff_a * coeff_b * N_a * N_b * raw_integral
 */
inline void computeCoreHamiltonianMatrix(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const Atom* atoms, const PrimitiveShell* shells,
    const real_t* cgto_norms,
    real_t* overlap_matrix, real_t* core_hamiltonian_matrix,
    int num_atoms, int num_basis) {

    // Zero out matrices
    std::memset(overlap_matrix, 0, (size_t)num_basis * num_basis * sizeof(real_t));
    std::memset(core_hamiltonian_matrix, 0, (size_t)num_basis * num_basis * sizeof(real_t));

    // Total number of primitives
    int num_prims = total_num_shells(shell_type_infos);

    // Group primitives into contracted shells
    auto contracted = group_contracted_shells(shells, num_prims);
    int num_contracted = (int)contracted.size();

    // For each pair of contracted shells, compute all component integrals
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int A = 0; A < num_contracted; A++) {
        for (int B = A; B < num_contracted; B++) {
            const auto& csA = contracted[A];
            const auto& csB = contracted[B];

            const auto& ang_a_list = AngularMomentums[csA.shell_type];
            const auto& ang_b_list = AngularMomentums[csB.shell_type];

            for (int ca = 0; ca < (int)ang_a_list.size(); ca++) {
                int la = ang_a_list[ca][0], ma = ang_a_list[ca][1], na = ang_a_list[ca][2];
                size_t mu = csA.basis_index + ca;

                for (int cb = 0; cb < (int)ang_b_list.size(); cb++) {
                    int lb = ang_b_list[cb][0], mb = ang_b_list[cb][1], nb = ang_b_list[cb][2];
                    size_t nu = csB.basis_index + cb;

                    // Skip if mu > nu (will be symmetrized)
                    if (mu > nu) continue;

                    double S = overlap_contracted(shells, cgto_norms,
                                                  csA.primitive_indices, csB.primitive_indices,
                                                  la, ma, na, lb, mb, nb);
                    double T = kinetic_contracted(shells, cgto_norms,
                                                  csA.primitive_indices, csB.primitive_indices,
                                                  la, ma, na, lb, mb, nb);
                    double V = nuclear_contracted(shells, cgto_norms,
                                                  csA.primitive_indices, csB.primitive_indices,
                                                  la, ma, na, lb, mb, nb,
                                                  atoms, num_atoms);

                    double norm = cgto_norms[mu] * cgto_norms[nu];

                    // Store in upper triangle (row-major)
                    overlap_matrix[mu * num_basis + nu] = norm * S;
                    core_hamiltonian_matrix[mu * num_basis + nu] = norm * (T + V);

                    // Symmetrize
                    if (mu != nu) {
                        overlap_matrix[nu * num_basis + mu] = norm * S;
                        core_hamiltonian_matrix[nu * num_basis + mu] = norm * (T + V);
                    }
                }
            }
        }
    }
}


// ============================================================
//  Driver: ERI matrix (stored, 8-fold symmetry)
// ============================================================

/**
 * Compute the full ERI matrix with 8-fold symmetry.
 * The output array eri_matrix has size N*(N+1)/2 * (N*(N+1)/2 + 1) / 2
 * where N = num_basis, indexed by get_1d_index.
 *
 * This is the same compact storage used by the GPU ERI_Stored class.
 */
inline void computeERIMatrix(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const PrimitiveShell* shells, const real_t* cgto_norms,
    real_t* eri_matrix, int num_basis) {

    int num_prims = total_num_shells(shell_type_infos);
    auto contracted = group_contracted_shells(shells, num_prims);
    int num_contracted = (int)contracted.size();

    // Zero out ERI matrix
    size_t npairs = (size_t)num_basis * (num_basis + 1) / 2;
    size_t eri_size = npairs * (npairs + 1) / 2;
    std::memset(eri_matrix, 0, eri_size * sizeof(real_t));

    // Build list of all (contracted_shell, component) pairs for iteration
    struct BasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<BasisFunc> basis_funcs;
    basis_funcs.reserve(num_basis);
    for (int A = 0; A < num_contracted; A++) {
        const auto& cs = contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            BasisFunc bf;
            bf.contracted_idx = A;
            bf.component = c;
            bf.basis_idx = cs.basis_index + c;
            bf.lx = ang_list[c][0];
            bf.ly = ang_list[c][1];
            bf.lz = ang_list[c][2];
            basis_funcs.push_back(bf);
        }
    }

    // Iterate over unique quartets (mu nu | la si) with 8-fold symmetry:
    //   mu <= nu, la <= si, (mu,nu) <= (la,si)
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int mu = 0; mu < num_basis; mu++) {
        for (int nu = mu; nu < num_basis; nu++) {
            for (int la = 0; la < num_basis; la++) {
                for (int si = la; si < num_basis; si++) {
                    // Enforce bra <= ket ordering
                    size_t bra_lin = tri_index(mu, nu, num_basis);
                    size_t ket_lin = tri_index(la, si, num_basis);
                    if (bra_lin > ket_lin) continue;

                    const auto& bf_mu = basis_funcs[mu];
                    const auto& bf_nu = basis_funcs[nu];
                    const auto& bf_la = basis_funcs[la];
                    const auto& bf_si = basis_funcs[si];

                    double val = eri_contracted(
                        shells,
                        contracted[bf_mu.contracted_idx].primitive_indices,
                        contracted[bf_nu.contracted_idx].primitive_indices,
                        contracted[bf_la.contracted_idx].primitive_indices,
                        contracted[bf_si.contracted_idx].primitive_indices,
                        bf_mu.lx, bf_mu.ly, bf_mu.lz,
                        bf_nu.lx, bf_nu.ly, bf_nu.lz,
                        bf_la.lx, bf_la.ly, bf_la.lz,
                        bf_si.lx, bf_si.ly, bf_si.lz);

                    double norm = cgto_norms[mu] * cgto_norms[nu]
                                * cgto_norms[la] * cgto_norms[si];

                    size_t idx = get_1d_index(mu, nu, la, si, num_basis);
                    eri_matrix[idx] = norm * val;
                }
            }
        }
    }
}


// ============================================================
//  Driver: Schwarz upper bounds
// ============================================================

/**
 * Compute Schwarz upper bounds: Q_{mu nu} = sqrt( (mu nu | mu nu) )
 * Stored as a flat array of size num_basis * num_basis (row-major, symmetric).
 *
 * The shell_pair_type_infos parameter is accepted for API compatibility
 * but not used in this implementation since we iterate over basis functions directly.
 */
inline void computeSchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* shells, const real_t* cgto_norms,
    real_t* upper_bounds, int num_basis) {

    int num_prims = total_num_shells(shell_type_infos);
    auto contracted = group_contracted_shells(shells, num_prims);
    int num_contracted = (int)contracted.size();

    // Build basis function list
    struct BasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<BasisFunc> basis_funcs;
    basis_funcs.reserve(num_basis);
    for (int A = 0; A < num_contracted; A++) {
        const auto& cs = contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            BasisFunc bf;
            bf.contracted_idx = A;
            bf.component = c;
            bf.basis_idx = cs.basis_index + c;
            bf.lx = ang_list[c][0];
            bf.ly = ang_list[c][1];
            bf.lz = ang_list[c][2];
            basis_funcs.push_back(bf);
        }
    }

    std::memset(upper_bounds, 0, (size_t)num_basis * num_basis * sizeof(real_t));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int mu = 0; mu < num_basis; mu++) {
        for (int nu = mu; nu < num_basis; nu++) {
            const auto& bf_mu = basis_funcs[mu];
            const auto& bf_nu = basis_funcs[nu];

            double val = eri_contracted(
                shells,
                contracted[bf_mu.contracted_idx].primitive_indices,
                contracted[bf_nu.contracted_idx].primitive_indices,
                contracted[bf_mu.contracted_idx].primitive_indices,
                contracted[bf_nu.contracted_idx].primitive_indices,
                bf_mu.lx, bf_mu.ly, bf_mu.lz,
                bf_nu.lx, bf_nu.ly, bf_nu.lz,
                bf_mu.lx, bf_mu.ly, bf_mu.lz,
                bf_nu.lx, bf_nu.ly, bf_nu.lz);

            double norm = cgto_norms[mu] * cgto_norms[nu]
                        * cgto_norms[mu] * cgto_norms[nu];

            double Q = std::sqrt(std::abs(norm * val));
            upper_bounds[mu * num_basis + nu] = Q;
            upper_bounds[nu * num_basis + mu] = Q;
        }
    }
}

} // namespace gansu::cpu
