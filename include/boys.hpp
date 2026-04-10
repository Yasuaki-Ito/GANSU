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
#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif
#include <cmath>

namespace gansu::gpu{

// ============================================================
//  CUDA intrinsics wrappers
//
//  gansu_drcp_rn, gansu_dsqrt_rn, gansu_fma_rn, gansu_double2uint_rd, gansu_double2int_rd are
//  __device__-only PTX builtins.  In any `__host__ __device__` Boys helper
//  we must dispatch them at compile time so the host pass compiles cleanly.
// ============================================================
__host__ __device__ inline double gansu_drcp_rn(double x) {
#ifdef __CUDA_ARCH__
    return __drcp_rn(x);
#else
    return 1.0 / x;
#endif
}

__host__ __device__ inline double gansu_dsqrt_rn(double x) {
#ifdef __CUDA_ARCH__
    return __dsqrt_rn(x);
#else
    return std::sqrt(x);
#endif
}

__host__ __device__ inline double gansu_fma_rn(double a, double b, double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}

__host__ __device__ inline unsigned int gansu_double2uint_rd(double x) {
#ifdef __CUDA_ARCH__
    return __double2uint_rd(x);
#else
    return static_cast<unsigned int>(std::floor(x));
#endif
}

__host__ __device__ inline int gansu_double2int_rd(double x) {
#ifdef __CUDA_ARCH__
    return __double2int_rd(x);
#else
    return static_cast<int>(std::floor(x));
#endif
}



inline __host__ __device__
double sMaclaurinExpansion(int n, double x)
{
    double numerator = 1.0;
    double factorial = 1.0;
    double F_x = gansu_drcp_rn(2 * n + 1);    // k = 0;
    const int k_max = gansu_double2uint_rd(AA * x * x * x - BB * x * x + CC * x + DD);

    for (int k = 1; k <= k_max; ++k) {
        numerator *= -x;
        factorial *= k;
        F_x += numerator / (factorial * (2 * n + 2 * k + 1));
    }
    return F_x;
}

inline __host__ __device__
double sTaylorInterpolation(int n, double x, const double* g_F_xi)
{
    double numerator = 1.0;
    int factorial = 1;
    const int x_idx = gansu_double2int_rd(x / LUT_XI_INTERVAL + 0.5);
    const double delta_x = x - (LUT_XI_INTERVAL * x_idx);
    double F_x = g_F_xi[LUT_NUM_XI * n + x_idx];    // (n, x)

    for (int k = 1; k <= LUT_K_MAX; ++k) {
        numerator *= -(delta_x);
        factorial *= k;
        F_x += (g_F_xi[LUT_NUM_XI * (n + k) + x_idx] * numerator) / factorial;
    }
    return F_x;
}

inline __host__ __device__
double sRecurrenceSemiInfinite(int n, double x)
{
    double exp_neg_x = 0.0; 
    const double reciprocal_double_x = gansu_drcp_rn(2 * x);
    double F_x = 0.5 * gansu_dsqrt_rn(M_PI / x);    // j = 0;

    if (x < A_RS * n + B_RS) {
        exp_neg_x = exp(-x);
        F_x *= erf(gansu_dsqrt_rn(x));
    }
    for (int j = 1; j <= n; ++j) {
        F_x = gansu_fma_rn((2 * j - 1), F_x, -exp_neg_x) * reciprocal_double_x;
    }
    return F_x;
}


inline __host__ __device__
void iMaclaurinExpansion(int n_prime, int n, double x, double* boys)
{
    double F_x;
    double numerator;
    double factorial;
    const int k_max = gansu_double2uint_rd(AA * x * x * x - BB * x * x + CC * x + DD);

    for (int j = n_prime; j <= n; ++j) {
        numerator = 1.0;
        factorial = 1.0;
        F_x = gansu_drcp_rn(2 * j + 1);    // k = 0  
        for (int k = 1; k <= k_max; ++k) {
            numerator *= -x;
            factorial *= k;
            F_x += numerator / (factorial * (2 * j + 2 * k + 1));
        }
        boys[j] = F_x;
    }
}

inline __host__ __device__
void iTaylorInterpolation(int n, double x, const double* g_F_xi, double* boys)
{
    double F_x;
    double numerator;
    int factorial;
    const int x_idx = gansu_double2uint_rd(x / LUT_XI_INTERVAL + 0.5);
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

inline __host__ __device__
void iRecurrenceSemiInfinite(int n, double x, double* boys)
{
    double exp_neg_x = 0.0;
    const double reciprocal_double_x = gansu_drcp_rn(2 * x);
    double F_x = 0.5 * gansu_dsqrt_rn(M_PI / x);    // j = 0

    // Recurrence relation method
    if (x < A_RS * n + B_RS) {
        exp_neg_x = exp(-x);
        F_x *= erf(gansu_dsqrt_rn(x));
    }

    boys[0] = F_x;
    for (int j = 1; j <= n; ++j) {
        F_x = gansu_fma_rn((2 * j - 1), F_x, -exp_neg_x) * reciprocal_double_x;
        boys[j] = F_x;
    }
}


inline __host__ __device__
double getSingleBoys(int n, double x, const double* g_F_xi)
{
    if (x == 0.0) {
        return gansu_drcp_rn(2 * n + 1);
    } 
    else if (x < (A_TR * n + B_TR)) {
        //return sMaclaurinExpansion(n, x);
        return sTaylorInterpolation(n, x, g_F_xi);
    } 
    else {
        return sRecurrenceSemiInfinite(n, x);
    }
}

inline __host__ __device__
void getIncrementalBoys(int n, double x, const double* g_F_xi, double* boys)
{
    if (x == 0.0) {
        for (int j = 0; j <= n; ++j) {
            boys[j] = gansu_drcp_rn(2 * j + 1);
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