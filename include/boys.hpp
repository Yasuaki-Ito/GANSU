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
 * @file boys.hpp This file contains the functions for computing the Boys function.
 * @details This implementation uses the GPU algorithms proposed in @cite Tsuji2023, @cite Tsuji2025, @cite GitHubBoys.
 */


#ifndef BOYS_CUH
#define BOYS_CUH

#include "parameters.h"
#include <cuda.h>

namespace gansu::gpu{



inline __device__
double sMaclaurinExpansion(int n, double x)
{
    double numerator = 1.0;
    double factorial = 1.0;
    double F_x = __drcp_rn(2 * n + 1);    // k = 0;
    const int k_max = __double2uint_rd(AA * x * x * x - BB * x * x + CC * x + DD);

    for (int k = 1; k <= k_max; ++k) {
        numerator *= -x;
        factorial *= k;
        F_x += numerator / (factorial * (2 * n + 2 * k + 1));
    }
    return F_x;
}

inline __device__
double sTaylorInterpolation(int n, double x, const double* g_F_xi)
{
    double numerator = 1.0;
    int factorial = 1;
    const int x_idx = __double2int_rd(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    double F_x = g_F_xi[LUT_NUM_XI * n + x_idx];    // (n, x)

    for (int k = 1; k <= LUT_K_MAX; ++k) {
        numerator *= -(delta_x);
        factorial *= k;
        F_x += (g_F_xi[LUT_NUM_XI * (n + k) + x_idx] * numerator) / factorial;
    }
    return F_x;
}

inline __device__
double sRecurrenceSemiInfinite(int n, double x)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = __drcp_rn(2 * x);
    double F_x = 0.5 * __dsqrt_rn(M_PI / x);    // j = 0;

    if (x < A_RS * n + B_RS) {
        exp_neg_x = exp(-x);
        F_x *= erf(__dsqrt_rn(x));
    }
    for (int j = 1; j <= n; ++j) {
        F_x = __fma_rn((2 * j - 1), F_x, -exp_neg_x) * reciprocal_double_x;
    }
    return F_x;
}


inline __device__
void iMaclaurinExpansion(int n_prime, int n, double x, double* boys)
{
    double F_x;
    double numerator;
    double factorial;
    const int k_max = __double2uint_rd(AA * x * x * x - BB * x * x + CC * x + DD);

    for (int j = n_prime; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1.0;
        F_x = __drcp_rn(2 * j + 1);    // k = 0  
        for (int k = 1; k <= k_max; ++k) {
            numerator *= -x;
            factorial *= k;
            F_x += numerator / (factorial * (2 * j + 2 * k + 1));
        }
        boys[j] = F_x;
    }
}

inline __device__
void iTaylorInterpolation(int n, double x, const double* g_F_xi, double* boys)
{
    double F_x;
    double numerator;
    int factorial;
    const int x_idx = __double2uint_rd(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    
    for (int j = 0; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1;
        F_x = g_F_xi[LUT_NUM_XI * j + x_idx];

        for (int k = 1; k <= LUT_K_MAX; ++k) {
            numerator *= -(delta_x);
            factorial *= k;
            F_x += (g_F_xi[LUT_NUM_XI * (j + k) + x_idx] * numerator) / factorial;
        }
        boys[j] = F_x;
    }
}

inline __device__
void iRecurrenceSemiInfinite(int n, double x, double* boys)
{
    double exp_neg_x = 0.0;
    const double reciprocal_double_x = __drcp_rn(2 * x);
    double F_x = 0.5 * __dsqrt_rn(M_PI / x);    // j = 0

    // Recurrence relation method
    if (x < A_RS * n + B_RS) {
        exp_neg_x = exp(-x);
        F_x *= erf(__dsqrt_rn(x));
    }

    boys[0] = F_x;
    for (int j = 1; j <= n; ++j) {
        F_x = __fma_rn((2 * j - 1), F_x, -exp_neg_x) * reciprocal_double_x;
        boys[j] = F_x;
    }
}


inline __device__
double getSingleBoys(int n, double x, const double* g_F_xi)
{
    if (x == 0.0) {
        return __drcp_rn(2 * n + 1);
    } 
    else if (x < (A_TR * n + B_TR)) {
        //return sMaclaurinExpansion(n, x);
        return sTaylorInterpolation(n, x, g_F_xi);
    } 
    else {
        return sRecurrenceSemiInfinite(n, x);
    }
}

inline __device__
void getIncrementalBoys(int n, double x, const double* g_F_xi, double* boys)
{
    if (x == 0.0) {
        for (int j = 0; j <= n; ++j) {
            boys[j] = __drcp_rn(2 * j + 1);
        }
    }
    else if (x < (A_TR * n + B_TR)) {
        /*
        const int n_prime = __double2int_ru((x - B_TR) / A_TR);
        iRecurrenceSemiInfinite(n_prime - 1, x, boys);
        iMaclaurinExpansion(n_prime, n, x, boys);
        */
        iTaylorInterpolation(n, x, g_F_xi, boys);
    }
    else {
        iRecurrenceSemiInfinite(n, x, boys);
    }
}

} // namespace gansu::gpu

#endif