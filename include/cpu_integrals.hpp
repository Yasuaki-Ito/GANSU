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
    // N = (2/pi)^{3/4} * (4*alpha)^{L/2} * alpha^{3/4} / sqrt(df(2l-1)*df(2m-1)*df(2n-1))
    // Note: GPU calcNorm uses pi^{-3/4} instead of (2/pi)^{3/4}, but the overlap_primitive
    // function already includes (pi/p)^{3/2} prefactor. The correct normalization for the
    // McMurchie-Davidson convention with (pi/p)^{3/2} prefactor is (2/pi)^{3/4}.
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

// Maximum angular momentum supported (g = 4, needed for RI auxiliary basis)
static constexpr int MAX_AM = 4;
// Maximum combined angular momentum for one coordinate (needed for kinetic: l+2)
static constexpr int MAX_AM_PLUS = MAX_AM + 2; // 6
static constexpr int E_DIM1 = MAX_AM_PLUS + 1;  // 7
static constexpr int E_DIM2 = MAX_AM_PLUS + 1;  // 7
static constexpr int E_DIM3 = 2 * MAX_AM_PLUS + 1; // 13

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
                    atoms[at].coordinate, atoms[at].effective_charge);
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
        for (int B = 0; B < num_contracted; B++) {
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

                    // Skip upper triangle (will be symmetrized)
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
 * GPU ERI_Stored class stores ERI as 4D tensor: eri[mu*N^3 + nu*N^2 + la*N + si]
 * with full 8-fold symmetry expansion (all permutations stored).
 */
inline void computeERIMatrix(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const PrimitiveShell* shells, const real_t* cgto_norms,
    real_t* eri_matrix, int num_basis) {

    int num_prims = total_num_shells(shell_type_infos);
    auto contracted = group_contracted_shells(shells, num_prims);
    int num_contracted = (int)contracted.size();

    const size_t N = num_basis;
    // Zero out ERI matrix (4D tensor: N^4 elements)
    std::memset(eri_matrix, 0, N * N * N * N * sizeof(real_t));

    // Build list of all (contracted_shell, component) pairs for iteration
    struct BasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<BasisFunc> basis_funcs(num_basis);
    for (int A = 0; A < num_contracted; A++) {
        const auto& cs = contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            size_t idx = cs.basis_index + c;
            assert(idx < (size_t)num_basis);
            basis_funcs[idx].contracted_idx = A;
            basis_funcs[idx].component = c;
            basis_funcs[idx].basis_idx = idx;
            basis_funcs[idx].lx = ang_list[c][0];
            basis_funcs[idx].ly = ang_list[c][1];
            basis_funcs[idx].lz = ang_list[c][2];
        }
    }

    // Iterate over unique quartets and expand 8-fold symmetry into 4D tensor
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int mu = 0; mu < num_basis; mu++) {
        for (int nu = mu; nu < num_basis; nu++) {
            for (int la = 0; la < num_basis; la++) {
                for (int si = la; si < num_basis; si++) {
                    // Enforce bra <= ket ordering to avoid double-computing
                    if (mu * N + nu > la * N + si) continue;

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
                    double v = norm * val;

                    // Expand all 8 symmetry permutations into 4D tensor
                    // (mu nu|la si) = (nu mu|la si) = (mu nu|si la) = (nu mu|si la)
                    //                = (la si|mu nu) = (si la|mu nu) = (la si|nu mu) = (si la|nu mu)
                    auto set = [&](int a, int b, int c, int d) {
                        eri_matrix[a*N*N*N + b*N*N + c*N + d] = v;
                    };
                    set(mu, nu, la, si);
                    set(nu, mu, la, si);
                    set(mu, nu, si, la);
                    set(nu, mu, si, la);
                    set(la, si, mu, nu);
                    set(si, la, mu, nu);
                    set(la, si, nu, mu);
                    set(si, la, nu, mu);
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

// ============================================================
//  Numerical gradient helper routines
// ============================================================

/**
 * Compute nuclear repulsion energy for a given set of atoms.
 */
inline double compute_nuclear_repulsion(const Atom* atoms, int num_atoms) {
    double V_nn = 0.0;
    for (int A = 0; A < num_atoms; A++) {
        for (int B = A + 1; B < num_atoms; B++) {
            double dx = atoms[A].coordinate.x - atoms[B].coordinate.x;
            double dy = atoms[A].coordinate.y - atoms[B].coordinate.y;
            double dz = atoms[A].coordinate.z - atoms[B].coordinate.z;
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            V_nn += (double)atoms[A].atomic_number * (double)atoms[B].atomic_number / r;
        }
    }
    return V_nn;
}

/**
 * Compute the total RHF energy given fixed density matrix D and integrals.
 *
 * E = V_nn + sum_{mu,nu} D(mu,nu) * H(mu,nu)
 *   + 0.5 * sum_{mu,nu,la,si} D(mu,nu) * D(la,si) * [2*(mu nu|la si) - (mu la|nu si)]
 *
 * where H is the core Hamiltonian, and the ERI tensor is stored as eri[mu*N^3+nu*N^2+la*N+si].
 * The density matrix D is the converged RHF density (fixed).
 */
inline double compute_rhf_energy(
    const real_t* density_matrix,
    const real_t* core_hamiltonian,
    const real_t* eri_matrix,
    double V_nn,
    int num_basis)
{
    const size_t N = num_basis;

    // E = 0.5 * Tr(D*(H+F)) + V_nn
    // F(mu,nu) = H(mu,nu) + sum D(la,si) * [(mu nu|la si) - 0.5*(mu la|nu si)]
    // So: E = 0.5 * sum D * (H + H + G) = 0.5 * sum D * (2H + G)
    //       = sum D*H + 0.5 * sum D*G
    // where G(mu,nu) = sum D(la,si) * [(mn|ls) - 0.5*(ml|ns)]
    // Note: D already contains factor 2 (D=2*CC), and G uses (J-0.5K) matching GPU Fock.
    double E = 0.0;
    for (int mu = 0; mu < num_basis; mu++) {
        for (int nu = 0; nu < num_basis; nu++) {
            double F_mn = core_hamiltonian[mu * N + nu];
            for (int la = 0; la < num_basis; la++) {
                for (int si = 0; si < num_basis; si++) {
                    double D_ls = density_matrix[la * N + si];
                    double J = eri_matrix[mu * N * N * N + nu * N * N + la * N + si];
                    double K = eri_matrix[mu * N * N * N + la * N * N + nu * N + si];
                    F_mn += D_ls * (J - 0.5 * K);
                }
            }
            E += density_matrix[mu * N + nu] * (core_hamiltonian[mu * N + nu] + F_mn);
        }
    }
    E *= 0.5;

    return V_nn + E;
}

// ============================================================
//  Driver: CPU Energy Gradient for RHF (Numerical Differentiation)
// ============================================================

/**
 * Compute the RHF energy gradient on CPU using numerical central differences.
 *
 * For each atom A and direction d (x, y, z):
 *   1. Displace atom A by +h in direction d, rebuild all integrals
 *   2. Compute E(+h) = V_nn(+h) + Tr(D*H(+h)) + 0.5*Tr(D*G(+h)) with FIXED density D
 *   3. Displace by -h, compute E(-h) similarly
 *   4. gradient[3*A+d] = (E(+h) - E(-h)) / (2*h)
 *
 * The density matrix D is kept FIXED from the converged SCF at the original geometry.
 * This is O(N^4) per displacement but guaranteed correct for small molecules.
 *
 * Note: W_matrix is accepted for API compatibility but not used in numerical differentiation.
 */
inline void computeEnergyGradient_RHF(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const Atom* atoms, const PrimitiveShell* shells,
    const real_t* cgto_norms,
    const real_t* density_matrix,
    const real_t* W_matrix,
    int num_atoms, int num_basis,
    std::vector<double>& gradient)
{
    const int n_grad = 3 * num_atoms;
    gradient.assign(n_grad, 0.0);

    const double h = 1.0e-4; // step size in bohr (central difference)
    const size_t N = num_basis;
    const size_t N4 = N * N * N * N;
    const size_t N2 = N * N;

    // Count total primitive shells
    int num_prims = total_num_shells(shell_type_infos);

    // For each atom and direction, compute E(+h) and E(-h)
    for (int atom = 0; atom < num_atoms; atom++) {
        for (int dir = 0; dir < 3; dir++) {
            double E_plus = 0.0, E_minus = 0.0;

            // Do +h and -h displacements
            for (int sign = 0; sign < 2; sign++) {
                double delta = (sign == 0) ? +h : -h;

                // Create displaced atom and shell copies
                std::vector<Atom> disp_atoms(atoms, atoms + num_atoms);
                std::vector<PrimitiveShell> disp_shells(shells, shells + num_prims);

                // Displace the target atom
                if (dir == 0) disp_atoms[atom].coordinate.x += delta;
                else if (dir == 1) disp_atoms[atom].coordinate.y += delta;
                else disp_atoms[atom].coordinate.z += delta;

                // Displace all primitive shells belonging to this atom
                for (int p = 0; p < num_prims; p++) {
                    if (disp_shells[p].atom_index == atom) {
                        if (dir == 0) disp_shells[p].coordinate.x += delta;
                        else if (dir == 1) disp_shells[p].coordinate.y += delta;
                        else disp_shells[p].coordinate.z += delta;
                    }
                }

                // Compute nuclear repulsion with displaced geometry
                double V_nn = compute_nuclear_repulsion(disp_atoms.data(), num_atoms);

                // Compute overlap and core Hamiltonian with displaced geometry
                std::vector<real_t> S_disp(N2, 0.0);
                std::vector<real_t> H_disp(N2, 0.0);
                computeCoreHamiltonianMatrix(
                    shell_type_infos,
                    disp_atoms.data(), disp_shells.data(), cgto_norms,
                    S_disp.data(), H_disp.data(),
                    num_atoms, num_basis);

                // Compute ERI matrix with displaced geometry
                std::vector<real_t> eri_disp(N4, 0.0);
                computeERIMatrix(
                    shell_type_infos,
                    disp_shells.data(), cgto_norms,
                    eri_disp.data(), num_basis);

                // Compute energy with FIXED density matrix but displaced integrals
                double E = compute_rhf_energy(
                    density_matrix, H_disp.data(), eri_disp.data(),
                    V_nn, num_basis);

                if (sign == 0) E_plus = E;
                else E_minus = E;
            }

            gradient[3 * atom + dir] = (E_plus - E_minus) / (2.0 * h);
        }
    }
}

// ============================================================
//  2-center ERI primitive: (P|Q) between two single Gaussians
// ============================================================

/**
 * Compute the 2-center electron repulsion integral between two primitive Gaussians:
 *   (P|Q) = integral phi_P(r1) * (1/|r1-r2|) * phi_Q(r2) dr1 dr2
 *
 * Using McMurchie-Davidson, this is equivalent to a 4-center ERI (P s | Q s)
 * where each "bra" and "ket" consists of a single Gaussian (no pair partner).
 *
 * For a single Gaussian P with exponent alpha at center A, angular momentum (l,m,n):
 *   The Gaussian product center is just A (p = alpha, XPA = 0).
 *   The E coefficients reduce to: E^t_{i,0}(alpha, 0, 0) computed with p=alpha, b=0, XPA=0, XPB=0.
 *   Since b=0 leads to degenerate behavior in the standard recursion,
 *   we treat each side as E^t_{i,0} with the recursion:
 *     E^0_{0,0} = 1
 *     E^t_{i+1,0} = (1/(2p)) E^{t-1}_{i,0} + 0 + (t+1) E^{t+1}_{i,0}
 *   (XPA = 0 because the product center P of (alpha, 0) at center A is just A itself.)
 *
 * Prefactor = 2 pi^{5/2} / (p * q * sqrt(p+q))  where p=alpha_P, q=alpha_Q
 * R integrals are computed with composite exponent pq/(p+q) over P-Q = A-B.
 */
inline double eri2_primitive(double alpha, double beta,
                             const Coordinate& A, const Coordinate& B,
                             int la, int ma, int na,
                             int lb, int mb, int nb) {
    // Implement the 2-center ERI (P|Q) as a 4-center (Ps|Qs) with
    // phantom Gaussians (b=d=EPS, angular momentum 0) placed at the same
    // centers.  Using a small finite EPS avoids the degeneracy in
    // compute_E_coefficients (b=0) whose behavior is untested.
    // With lb=mb=nb=0 on the phantom side, the E coefficients do not
    // depend on the (lb,mb,nb) path and the only effect of EPS is a tiny
    // shift in the product centre, which is negligible for EPS << alpha,beta.
    const double EPS = 0.0;
    // Path: reuse the full 4-center primitive with phantom exponents = 0.
    // With b=d=0, the prefactor has mu_ab = a*b/(a+b) = 0, exp(-mu*AB²)=1,
    // and the product centers P=A, Q=B. The 4-center kernel reduces exactly
    // to the desired 2-center integral.
    return eri_primitive(
        alpha, EPS, beta, EPS,
        A, A, B, B,
        la, ma, na, 0, 0, 0,
        lb, mb, nb, 0, 0, 0);
}


// ============================================================
//  3-center ERI primitive: (P|mu nu) with one auxiliary, two AO
// ============================================================

/**
 * Compute the 3-center ERI between one auxiliary primitive P and an AO pair (mu,nu):
 *   (P|mu nu) = integral phi_P(r1) * (1/|r1-r2|) * phi_mu(r2) phi_nu(r2) dr1 dr2
 *
 * This is a 4-center ERI (P s | mu nu) where the bra is a single Gaussian.
 * Bra: exponent p = alpha_P, center A (single Gaussian, j=0)
 * Ket: exponent q = alpha_mu + alpha_nu, center Q = weighted average
 *
 * Prefactor = 2 pi^{5/2} / (p * q * sqrt(p+q)) * exp(-mu_cd * |CD|^2)
 */
inline double eri3_primitive(double alpha_P,
                             double alpha_mu, double alpha_nu,
                             const Coordinate& A,
                             const Coordinate& C, const Coordinate& D,
                             int lP, int mP, int nP,
                             int lc, int mc, int nc,
                             int ld, int md, int nd) {
    // Delegate to the 4-center primitive with a phantom s-Gaussian at A.
    // With b=0, the bra pair (alpha_P, 0) at (A, A) collapses to the single
    // Gaussian at A with exponent alpha_P and angular momentum (lP,mP,nP).
    return eri_primitive(
        alpha_P, 0.0, alpha_mu, alpha_nu,
        A, A, C, D,
        lP, mP, nP, 0, 0, 0,
        lc, mc, nc, ld, md, nd);
}


// ============================================================
//  Contracted 2-center ERI: (P|Q)
// ============================================================

/// Contracted 2-center ERI between two auxiliary shells
inline double eri2_contracted(const PrimitiveShell* shells,
                              const std::vector<size_t>& prims_P,
                              const std::vector<size_t>& prims_Q,
                              int lP, int mP, int nP,
                              int lQ, int mQ, int nQ) {
    double result = 0.0;
    for (size_t iP : prims_P) {
        const auto& sP = shells[iP];
        double NP = primitive_norm(sP.exponent, lP, mP, nP);
        for (size_t iQ : prims_Q) {
            const auto& sQ = shells[iQ];
            double NQ = primitive_norm(sQ.exponent, lQ, mQ, nQ);
            result += sP.coefficient * sQ.coefficient * NP * NQ
                    * eri2_primitive(sP.exponent, sQ.exponent,
                                    sP.coordinate, sQ.coordinate,
                                    lP, mP, nP, lQ, mQ, nQ);
        }
    }
    return result;
}


// ============================================================
//  Contracted 3-center ERI: (P|mu nu)
// ============================================================

/// Contracted 3-center ERI: auxiliary P with AO pair (mu, nu)
inline double eri3_contracted(const PrimitiveShell* aux_shells,
                              const PrimitiveShell* ao_shells,
                              const std::vector<size_t>& prims_P,
                              const std::vector<size_t>& prims_mu,
                              const std::vector<size_t>& prims_nu,
                              int lP, int mP, int nP,
                              int lmu, int mmu, int nmu,
                              int lnu, int mnu, int nnu) {
    double result = 0.0;
    for (size_t iP : prims_P) {
        const auto& sP = aux_shells[iP];
        double NP = primitive_norm(sP.exponent, lP, mP, nP);
        for (size_t imu : prims_mu) {
            const auto& smu = ao_shells[imu];
            double Nmu = primitive_norm(smu.exponent, lmu, mmu, nmu);
            for (size_t inu : prims_nu) {
                const auto& snu = ao_shells[inu];
                double Nnu = primitive_norm(snu.exponent, lnu, mnu, nnu);
                result += sP.coefficient * smu.coefficient * snu.coefficient
                        * NP * Nmu * Nnu
                        * eri3_primitive(sP.exponent,
                                        smu.exponent, snu.exponent,
                                        sP.coordinate,
                                        smu.coordinate, snu.coordinate,
                                        lP, mP, nP,
                                        lmu, mmu, nmu,
                                        lnu, mnu, nnu);
            }
        }
    }
    return result;
}


// ============================================================
//  Driver: 2-center ERI matrix (P|Q) for RI
// ============================================================

/**
 * Compute the 2-center ERI matrix (P|Q) for RI approximation.
 * Output: two_center_eri[P * num_aux + Q] stored row-major, size num_aux x num_aux.
 * The matrix is symmetric: (P|Q) = (Q|P).
 */
inline void computeTwoCenterERIs(
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* aux_shells,
    const real_t* aux_cgto_norms,
    real_t* two_center_eri,
    int num_auxiliary_basis) {

    int num_aux_prims = total_num_shells(auxiliary_shell_type_infos);
    auto aux_contracted = group_contracted_shells(aux_shells, num_aux_prims);
    int num_aux_contracted = (int)aux_contracted.size();

    // Build auxiliary basis function list indexed by basis_index
    // (contracted-shell iteration order != basis_index order, so we
    //  must index by basis_idx to match the GPU output layout)
    struct AuxBasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<AuxBasisFunc> aux_funcs(num_auxiliary_basis);
    for (int A = 0; A < num_aux_contracted; A++) {
        const auto& cs = aux_contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            size_t idx = cs.basis_index + c;
            assert(idx < (size_t)num_auxiliary_basis);
            aux_funcs[idx].contracted_idx = A;
            aux_funcs[idx].component = c;
            aux_funcs[idx].basis_idx = idx;
            aux_funcs[idx].lx = ang_list[c][0];
            aux_funcs[idx].ly = ang_list[c][1];
            aux_funcs[idx].lz = ang_list[c][2];
        }
    }

    std::memset(two_center_eri, 0, (size_t)num_auxiliary_basis * num_auxiliary_basis * sizeof(real_t));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int P = 0; P < num_auxiliary_basis; P++) {
        for (int Q = P; Q < num_auxiliary_basis; Q++) {
            const auto& bfP = aux_funcs[P];
            const auto& bfQ = aux_funcs[Q];

            double val = eri2_contracted(
                aux_shells,
                aux_contracted[bfP.contracted_idx].primitive_indices,
                aux_contracted[bfQ.contracted_idx].primitive_indices,
                bfP.lx, bfP.ly, bfP.lz,
                bfQ.lx, bfQ.ly, bfQ.lz);

            double norm = aux_cgto_norms[P] * aux_cgto_norms[Q];
            double v = norm * val;

            two_center_eri[(size_t)P * num_auxiliary_basis + Q] = v;
            if (P != Q) {
                two_center_eri[(size_t)Q * num_auxiliary_basis + P] = v;
            }
        }
    }
}


// ============================================================
//  Driver: 3-center ERI matrix (P|mu nu) for RI
// ============================================================

/**
 * Compute the 3-center ERI matrix (P|mu nu) for RI approximation.
 * Output: three_center_eri[P * nao^2 + mu * nao + nu] stored as (N_aux x N^2).
 * Symmetric in (mu, nu): (P|mu nu) = (P|nu mu).
 */
inline void computeThreeCenterERIs(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const PrimitiveShell* ao_shells,
    const real_t* ao_cgto_norms,
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* aux_shells,
    const real_t* aux_cgto_norms,
    real_t* three_center_eri,
    int num_basis,
    int num_auxiliary_basis) {

    int num_ao_prims = total_num_shells(shell_type_infos);
    auto ao_contracted = group_contracted_shells(ao_shells, num_ao_prims);
    int num_ao_contracted = (int)ao_contracted.size();

    int num_aux_prims = total_num_shells(auxiliary_shell_type_infos);
    auto aux_contracted = group_contracted_shells(aux_shells, num_aux_prims);
    int num_aux_contracted = (int)aux_contracted.size();

    // Build AO basis function list indexed by basis_index
    struct BasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<BasisFunc> ao_funcs(num_basis);
    for (int A = 0; A < num_ao_contracted; A++) {
        const auto& cs = ao_contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            size_t idx = cs.basis_index + c;
            assert(idx < (size_t)num_basis);
            ao_funcs[idx].contracted_idx = A;
            ao_funcs[idx].component = c;
            ao_funcs[idx].basis_idx = idx;
            ao_funcs[idx].lx = ang_list[c][0];
            ao_funcs[idx].ly = ang_list[c][1];
            ao_funcs[idx].lz = ang_list[c][2];
        }
    }

    // Build auxiliary basis function list indexed by basis_index
    std::vector<BasisFunc> aux_funcs(num_auxiliary_basis);
    for (int A = 0; A < num_aux_contracted; A++) {
        const auto& cs = aux_contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            size_t idx = cs.basis_index + c;
            assert(idx < (size_t)num_auxiliary_basis);
            aux_funcs[idx].contracted_idx = A;
            aux_funcs[idx].component = c;
            aux_funcs[idx].basis_idx = idx;
            aux_funcs[idx].lx = ang_list[c][0];
            aux_funcs[idx].ly = ang_list[c][1];
            aux_funcs[idx].lz = ang_list[c][2];
        }
    }

    const size_t nao2 = (size_t)num_basis * num_basis;
    std::memset(three_center_eri, 0, (size_t)num_auxiliary_basis * nao2 * sizeof(real_t));

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int P = 0; P < num_auxiliary_basis; P++) {
        const auto& bfP = aux_funcs[P];
        for (int mu = 0; mu < num_basis; mu++) {
            for (int nu = mu; nu < num_basis; nu++) {
                const auto& bfmu = ao_funcs[mu];
                const auto& bfnu = ao_funcs[nu];

                double val = eri3_contracted(
                    aux_shells, ao_shells,
                    aux_contracted[bfP.contracted_idx].primitive_indices,
                    ao_contracted[bfmu.contracted_idx].primitive_indices,
                    ao_contracted[bfnu.contracted_idx].primitive_indices,
                    bfP.lx, bfP.ly, bfP.lz,
                    bfmu.lx, bfmu.ly, bfmu.lz,
                    bfnu.lx, bfnu.ly, bfnu.lz);

                double norm = aux_cgto_norms[P] * ao_cgto_norms[mu] * ao_cgto_norms[nu];
                double v = norm * val;

                three_center_eri[(size_t)P * nao2 + (size_t)mu * num_basis + nu] = v;
                if (mu != nu) {
                    three_center_eri[(size_t)P * nao2 + (size_t)nu * num_basis + mu] = v;
                }
            }
        }
    }
}


// ============================================================
//  Driver: Auxiliary Schwarz upper bounds for RI
// ============================================================

/**
 * Compute Schwarz upper bounds for auxiliary basis shells:
 *   Q_P = sqrt( |(P|P)| )
 * Stored as a flat array of size num_auxiliary_basis.
 */
inline void computeAuxiliarySchwarzUpperBounds(
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* aux_shells,
    const real_t* aux_cgto_norms,
    real_t* upper_bounds,
    int num_auxiliary_basis) {

    int num_aux_prims = total_num_shells(auxiliary_shell_type_infos);
    auto aux_contracted = group_contracted_shells(aux_shells, num_aux_prims);
    int num_aux_contracted = (int)aux_contracted.size();

    struct AuxBasisFunc {
        int contracted_idx;
        int component;
        size_t basis_idx;
        int lx, ly, lz;
    };

    std::vector<AuxBasisFunc> aux_funcs(num_auxiliary_basis);
    for (int A = 0; A < num_aux_contracted; A++) {
        const auto& cs = aux_contracted[A];
        const auto& ang_list = AngularMomentums[cs.shell_type];
        for (int c = 0; c < (int)ang_list.size(); c++) {
            size_t idx = cs.basis_index + c;
            assert(idx < (size_t)num_auxiliary_basis);
            aux_funcs[idx].contracted_idx = A;
            aux_funcs[idx].component = c;
            aux_funcs[idx].basis_idx = idx;
            aux_funcs[idx].lx = ang_list[c][0];
            aux_funcs[idx].ly = ang_list[c][1];
            aux_funcs[idx].lz = ang_list[c][2];
        }
    }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int P = 0; P < num_auxiliary_basis; P++) {
        const auto& bfP = aux_funcs[P];

        double val = eri2_contracted(
            aux_shells,
            aux_contracted[bfP.contracted_idx].primitive_indices,
            aux_contracted[bfP.contracted_idx].primitive_indices,
            bfP.lx, bfP.ly, bfP.lz,
            bfP.lx, bfP.ly, bfP.lz);

        double norm = aux_cgto_norms[P] * aux_cgto_norms[P];
        upper_bounds[P] = std::sqrt(std::abs(norm * val));
    }
}


// ============================================================
//  RI-Fock CPU helper kernels (matching GPU kernel logic)
// ============================================================

/**
 * CPU equivalent of computeRIIntermediateMatrixB_kernel:
 *   B[p][mu][nu] = sum_q three_center[q][mu][nu] * L[q][p]
 */
inline void computeRIIntermediateMatrixB_cpu(
    const real_t* three_center_eri,
    const real_t* matrix_L,
    real_t* matrix_B,
    int num_basis,
    int num_auxiliary_basis) {

    const size_t nao2 = (size_t)num_basis * num_basis;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int p = 0; p < num_auxiliary_basis; p++) {
        for (int mu = 0; mu < num_basis; mu++) {
            for (int nu = 0; nu < num_basis; nu++) {
                real_t sum = 0.0;
                for (int q = 0; q < num_auxiliary_basis; q++) {
                    sum += three_center_eri[(size_t)q * nao2 + (size_t)mu * num_basis + nu]
                         * matrix_L[(size_t)q * num_auxiliary_basis + p];
                }
                matrix_B[(size_t)p * nao2 + (size_t)mu * num_basis + nu] = sum;
            }
        }
    }
}

/**
 * CPU equivalent of weighted_sum_matrices_kernel:
 *   J[id] = sum_j W[j] * B[j * M*M + id]
 */
inline void weighted_sum_matrices_cpu(
    real_t* J, const real_t* B, const real_t* W,
    int M, int N, bool accumulated = false) {

    const size_t M2 = (size_t)M * M;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t id = 0; id < M2; id++) {
        real_t sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += W[j] * B[(size_t)j * M2 + id];
        }
        if (accumulated) {
            J[id] += sum;
        } else {
            J[id] = sum;
        }
    }
}

/**
 * CPU equivalent of sum_matrices_kernel:
 *   K[id] = sum_p B[p * M*M + id]
 */
inline void sum_matrices_cpu(
    real_t* K, const real_t* B,
    int M, int N, bool accumulated = false) {

    const size_t M2 = (size_t)M * M;
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t id = 0; id < M2; id++) {
        real_t sum = 0.0;
        for (int p = 0; p < N; p++) {
            sum += B[(size_t)p * M2 + id];
        }
        if (accumulated) {
            K[id] += sum;
        } else {
            K[id] = sum;
        }
    }
}

/**
 * CPU equivalent of computeFockMatrix_RI_RHF_kernel:
 *   F[id] = H[id] + J[id] - 0.5 * K[id]
 */
inline void computeFockMatrix_RI_RHF_cpu(
    const real_t* core_hamiltonian,
    const real_t* J, const real_t* K,
    real_t* fock, int num_basis) {

    const size_t N2 = (size_t)num_basis * num_basis;
    for (size_t id = 0; id < N2; id++) {
        fock[id] = core_hamiltonian[id] + J[id] - 0.5 * K[id];
    }
}

/**
 * CPU equivalent of computeFockMatrix_RI_UHF_kernel:
 *   F[id] = H[id] + J[id] - K[id]
 */
inline void computeFockMatrix_RI_UHF_cpu(
    const real_t* core_hamiltonian,
    const real_t* J, const real_t* K,
    real_t* fock, int num_basis) {

    const size_t N2 = (size_t)num_basis * num_basis;
    for (size_t id = 0; id < N2; id++) {
        fock[id] = core_hamiltonian[id] + J[id] - K[id];
    }
}

/**
 * CPU equivalent of computeFockMatrix_RI_ROHF_kernel:
 *   F_closed[id] = H[id] + J[id] - 0.5 * K_closed[id]
 *   F_open[id]   = 0.5 * (H[id] + J[id] - K_open[id])
 */
inline void computeFockMatrix_RI_ROHF_cpu(
    const real_t* core_hamiltonian,
    const real_t* J,
    const real_t* K_closed, const real_t* K_open,
    real_t* fock_closed, real_t* fock_open,
    int num_basis) {

    const size_t N2 = (size_t)num_basis * num_basis;
    for (size_t id = 0; id < N2; id++) {
        fock_closed[id] = core_hamiltonian[id] + J[id] - 0.5 * K_closed[id];
        fock_open[id] = 0.5 * (core_hamiltonian[id] + J[id] - K_open[id]);
    }
}

/**
 * CPU equivalent of packThreeDimensionalTensorX:
 *   d_X_out[num_aux*num_occ * mu + num_occ * p + k] = d_X_in[idx]
 * where idx = p * num_basis * num_occ + mu * num_occ + k
 */
inline void packThreeDimensionalTensorX_cpu(
    const real_t* X_in, real_t* X_out,
    int num_basis, int num_auxiliary_basis, int num_occ) {

    for (int p = 0; p < num_auxiliary_basis; p++) {
        for (int mu = 0; mu < num_basis; mu++) {
            for (int k = 0; k < num_occ; k++) {
                size_t in_idx = (size_t)p * num_basis * num_occ + (size_t)mu * num_occ + k;
                size_t out_idx = (size_t)num_auxiliary_basis * num_occ * mu + (size_t)num_occ * p + k;
                X_out[out_idx] = X_in[in_idx];
            }
        }
    }
}

} // namespace gansu::cpu
