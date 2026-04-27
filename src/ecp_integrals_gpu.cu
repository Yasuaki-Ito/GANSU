/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ecp_integrals_gpu.cu
 * @brief GPU kernel for off-center ECP Type 2 semi-local integrals
 *
 * Parallelizes the angular quadrature loop across GPU threads.
 * Each block handles one (radial_point) and reduces angular contributions.
 */

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include "ecp_integrals.hpp"

namespace gansu {
namespace ecp_integral {

// Device: real solid harmonics Y_lm at (x,y,z) on unit sphere
// Returns value for a specific (l, m_idx) pair
__device__ double solid_harmonic_dev(int l, int m_idx, double x, double y, double z) {
    constexpr double PI = 3.14159265358979323846;
    if (l == 0) return 1.0 / sqrt(4.0*PI);
    if (l == 1) {
        const double f = sqrt(3.0/(4.0*PI));
        if (m_idx == 0) return f*y;
        if (m_idx == 1) return f*z;
        return f*x;
    }
    if (l == 2) {
        const double f = sqrt(15.0/(4.0*PI));
        const double f0 = sqrt(5.0/(16.0*PI));
        switch(m_idx) {
            case 0: return f*x*y;
            case 1: return f*y*z;
            case 2: return f0*(3.0*z*z-1.0);
            case 3: return f*x*z;
            default: return 0.5*f*(x*x-y*y);
        }
    }
    if (l == 3) {
        switch(m_idx) {
            case 0: return sqrt(35.0/(32.0*PI))*y*(3.0*x*x-y*y);
            case 1: return sqrt(105.0/(4.0*PI))*x*y*z;
            case 2: return sqrt(21.0/(32.0*PI))*y*(5.0*z*z-1.0);
            case 3: return sqrt(7.0/(16.0*PI))*z*(5.0*z*z-3.0);
            case 4: return sqrt(21.0/(32.0*PI))*x*(5.0*z*z-1.0);
            case 5: return sqrt(105.0/(16.0*PI))*z*(x*x-y*y);
            default: return sqrt(35.0/(32.0*PI))*x*(x*x-3.0*y*y);
        }
    }
    return 0.0;
}

// Device: primitive Gaussian normalization
__device__ double primitive_norm_dev(double alpha, int l, int m, int n) {
    constexpr double PI = 3.14159265358979323846;
    int L = l + m + n;
    // double factorial
    auto df = [](int nn) -> double {
        double r = 1.0;
        for (int i = nn; i > 1; i -= 2) r *= i;
        return r;
    };
    return pow(2.0/PI, 0.75) * pow(4.0*alpha, L/2.0) * pow(alpha, 0.75)
         / sqrt(df(2*l-1) * df(2*m-1) * df(2*n-1));
}

// Use AngPtGPU from ecp_integrals.hpp

// Kernel: compute ang_sum(r) for one (ca,cb,ir) combination
// Each thread handles a subset of angular grid points, then reduces
__global__ void ecp_type2_offcenter_kernel(
    // Angular grid
    const AngPtGPU* __restrict__ ang_grid, int n_ang,
    // Basis shell A info
    const double* __restrict__ prim_exp_a, const double* __restrict__ prim_coef_a, int n_prim_a,
    int la, int ma_c, int na_c,
    double dAx, double dAy, double dAz,
    // Basis shell B info
    const double* __restrict__ prim_exp_b, const double* __restrict__ prim_coef_b, int n_prim_b,
    int lb, int mb_c, int nb_c,
    double dBx, double dBy, double dBz,
    // ECP primitives for this semi-local component
    const double* __restrict__ ecp_exp, const double* __restrict__ ecp_coef, const int* __restrict__ ecp_pow, int n_ecp_prim,
    // Radial grid
    const double* __restrict__ rad_nodes, const double* __restrict__ rad_weights, int n_rad,
    double R_max,
    // Projector info
    int l_proj,
    // Output: prim_val[n_ecp_prim] for this (ca,cb) pair
    double* __restrict__ output_prim_val)
{
    int n_m = 2 * l_proj + 1;

    // Each block handles one radial point
    int ir = blockIdx.x;
    if (ir >= n_rad) return;

    double t = (rad_nodes[ir] + 1.0) / 2.0;
    double r = R_max * t;
    double wr = rad_weights[ir] * R_max / 2.0;
    if (r < 1e-15) return;

    // Shared memory for angular reduction of A_lm products
    extern __shared__ double shmem[];
    // Layout: ang_sum_partial[n_ecp_prim * blockDim.x]
    // But simpler: first reduce A_lm_mu and A_lm_nu, then compute ang_sum

    // Each thread accumulates partial A_lm
    double Alm_mu_local[7] = {};
    double Alm_nu_local[7] = {};

    // Parallel over angular grid points
    for (int ia = threadIdx.x; ia < n_ang; ia += blockDim.x) {
        double ox = ang_grid[ia].x, oy = ang_grid[ia].y, oz = ang_grid[ia].z;
        double w_ang = ang_grid[ia].w;

        double rx = r*ox-dAx, ry = r*oy-dAy, rz = r*oz-dAz;
        double r2A = rx*rx + ry*ry + rz*rz;
        double sx = r*ox-dBx, sy = r*oy-dBy, sz = r*oz-dBz;
        double r2B = sx*sx + sy*sy + sz*sz;

        double pow_mu = 1.0;
        for (int k=0;k<la;k++) pow_mu*=rx;
        for (int k=0;k<ma_c;k++) pow_mu*=ry;
        for (int k=0;k<na_c;k++) pow_mu*=rz;
        double pow_nu = 1.0;
        for (int k=0;k<lb;k++) pow_nu*=sx;
        for (int k=0;k<mb_c;k++) pow_nu*=sy;
        for (int k=0;k<nb_c;k++) pow_nu*=sz;

        double mu_v = 0.0;
        for (int ip = 0; ip < n_prim_a; ip++)
            mu_v += prim_coef_a[ip] * primitive_norm_dev(prim_exp_a[ip], la, ma_c, na_c)
                    * pow_mu * exp(-prim_exp_a[ip] * r2A);
        double nu_v = 0.0;
        for (int jp = 0; jp < n_prim_b; jp++)
            nu_v += prim_coef_b[jp] * primitive_norm_dev(prim_exp_b[jp], lb, mb_c, nb_c)
                    * pow_nu * exp(-prim_exp_b[jp] * r2B);

        for (int m = 0; m < n_m; m++) {
            double Ylm = solid_harmonic_dev(l_proj, m, ox, oy, oz);
            Alm_mu_local[m] += w_ang * mu_v * Ylm;
            Alm_nu_local[m] += w_ang * nu_v * Ylm;
        }
    }

    // Warp-level reduction for each A_lm component
    // Use shared memory: shmem[m * blockDim.x + threadIdx.x]
    for (int m = 0; m < n_m; m++) {
        shmem[m * blockDim.x + threadIdx.x] = Alm_mu_local[m];
        shmem[(n_m + m) * blockDim.x + threadIdx.x] = Alm_nu_local[m];
    }
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int m = 0; m < n_m; m++) {
                shmem[m * blockDim.x + threadIdx.x] += shmem[m * blockDim.x + threadIdx.x + stride];
                shmem[(n_m+m) * blockDim.x + threadIdx.x] += shmem[(n_m+m) * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0: compute ang_sum and accumulate for each ECP primitive
    if (threadIdx.x == 0) {
        double ang_sum = 0.0;
        for (int m = 0; m < n_m; m++)
            ang_sum += shmem[m * blockDim.x] * shmem[(n_m+m) * blockDim.x];

        for (int k = 0; k < n_ecp_prim; k++) {
            double ecp_rad = pow(r, (double)ecp_pow[k]) * exp(-ecp_exp[k] * r * r);
            atomicAdd(&output_prim_val[k], wr * ecp_rad * ang_sum);
        }
    }
}

// Host function: launch GPU kernel for off-center Type 2 integral of one (ca,cb) pair
void launch_ecp_type2_offcenter_gpu(
    const AngPtGPU* d_ang_grid, int n_ang,
    const double* d_prim_exp_a, const double* d_prim_coef_a, int n_prim_a,
    int la, int ma_c, int na_c, double dAx, double dAy, double dAz,
    const double* d_prim_exp_b, const double* d_prim_coef_b, int n_prim_b,
    int lb, int mb_c, int nb_c, double dBx, double dBy, double dBz,
    const double* d_ecp_exp, const double* d_ecp_coef, const int* d_ecp_pow, int n_ecp_prim,
    const double* d_rad_nodes, const double* d_rad_weights, int n_rad,
    double R_max, int l_proj,
    double* d_output_prim_val)
{
    int n_m = 2 * l_proj + 1;
    int threads = 128;
    int blocks = n_rad;
    size_t shmem_size = 2 * n_m * threads * sizeof(double);

    // Zero output
    cudaMemset(d_output_prim_val, 0, n_ecp_prim * sizeof(double));

    ecp_type2_offcenter_kernel<<<blocks, threads, shmem_size>>>(
        d_ang_grid, n_ang,
        d_prim_exp_a, d_prim_coef_a, n_prim_a, la, ma_c, na_c, dAx, dAy, dAz,
        d_prim_exp_b, d_prim_coef_b, n_prim_b, lb, mb_c, nb_c, dBx, dBy, dBz,
        d_ecp_exp, d_ecp_coef, d_ecp_pow, n_ecp_prim,
        d_rad_nodes, d_rad_weights, n_rad, R_max, l_proj,
        d_output_prim_val);
}

} // namespace ecp_integral
} // namespace gansu

#endif // !GANSU_CPU_ONLY
