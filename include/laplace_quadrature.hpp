/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file laplace_quadrature.hpp
 * @brief Laplace quadrature for 1/x ≈ Σ_k w_k exp(-t_k x)
 *
 * Used in Laplace-transformed MP2 (LT-MP2) to factorize
 * the orbital energy denominator:
 *   1/(ε_a + ε_b - ε_i - ε_j) ≈ Σ_k w_k exp(-t_k (ε_a + ε_b - ε_i - ε_j))
 *
 * Uses logarithmic-grid quadrature optimized for the range [ε_min, ε_max].
 */

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

namespace gansu {

struct LaplaceQuadrature {
    std::vector<double> points;   // t_k
    std::vector<double> weights;  // w_k
    int num_points;
};

/**
 * @brief Generate Laplace quadrature points for 1/x on [x_min, x_max]
 * @param x_min  Minimum denominator value (HOMO-LUMO gap)
 * @param x_max  Maximum denominator value
 * @param n_points Number of quadrature points (default: 10)
 * @return LaplaceQuadrature with points and weights
 *
 * Uses logarithmic Gauss-Legendre quadrature:
 *   1/x = ∫_0^∞ exp(-xt) dt
 * mapped to finite interval via t = exp(u), u ∈ [log(1/x_max), log(1/x_min)]
 * with Gauss-Legendre nodes on the u interval.
 *
 * The quadrature satisfies: 1/x ≈ Σ_k w_k exp(-t_k x) for x ∈ [x_min, x_max]
 */
inline LaplaceQuadrature generate_laplace_quadrature(double x_min, double x_max, int n_points = 30)
{
    // Double exponential (DE) quadrature for 1/x = ∫_0^∞ exp(-xt) dt
    //   Substitution: t = exp(π/2 * sinh(u)), u ∈ [u_lo, u_hi]
    //   dt = π/2 * cosh(u) * t * du
    //   Trapezoidal rule on uniform u grid converges exponentially.
    //
    // Range-adaptive: u bounds from x range.
    //   Peak of integrand exp(-xt)*t is at t = 1/x.
    //   For x ∈ [x_min, x_max], we need t ∈ [~1/x_max, ~1/x_min].
    //   Map to u: u = asinh(2/π × ln(t)).

    double u_lo, u_hi;
    if (x_min > 0.0 && x_max > x_min) {
        u_lo = std::asinh(2.0 / M_PI * std::log(1.0 / x_max));
        u_hi = std::asinh(2.0 / M_PI * std::log(1.0 / x_min));
        // Padding: broader for fewer points to capture tails
        double pad = (n_points <= 10) ? 1.0 : 0.5;
        u_lo -= pad;
        u_hi += pad;
        // Practical bounds
        u_lo = std::max(u_lo, -5.0);
        u_hi = std::min(u_hi,  5.0);
    } else {
        // Fallback: fixed range (same as original)
        u_lo = -3.5;
        u_hi =  3.5;
    }

    double h = (u_hi - u_lo) / std::max(n_points - 1, 1);

    LaplaceQuadrature quad;
    quad.num_points = n_points;
    quad.points.resize(n_points);
    quad.weights.resize(n_points);

    for (int k = 0; k < n_points; k++) {
        double u = u_lo + k * h;
        double sinh_u = std::sinh(u);
        double cosh_u = std::cosh(u);
        double t = std::exp(M_PI / 2.0 * sinh_u);
        double w = h * M_PI / 2.0 * cosh_u * t;
        quad.points[k] = t;
        quad.weights[k] = w;
    }

    return quad;
}

} // namespace gansu
