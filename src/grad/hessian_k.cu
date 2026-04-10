/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Kinetic energy integral Hessian: d²T/dA dB
//
// T_μν = -½ <μ|∇²|ν> is computed using the relation:
//   T(l1,l2) = -2β²S(l1,l2+2) + β(2l2+1)S(l1,l2) - l2(l2-1)/2 S(l1,l2-2)
//   for each Cartesian direction, summed over x,y,z contributions.
//
// The derivative uses the same angular momentum shift formula as overlap:
//   d/dA_x T = 2α·T(l1+1,...) - l1·T(l1-1,...)
//
// Translational invariance: d/dB = -d/dA

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif
#include <cmath>
#include "gradients.hpp"

namespace gansu::gpu {

// Overlap integral for one direction (same as hessian_s.cu)
inline __host__ __device__
double ovlp_1d(int l1, int l2, double alpha, double beta, double AB) {
    return MD_Et_NonRecursion(l1, l2, 0, alpha, beta, AB);
}

// Kinetic energy integral for one direction:
// T_sigma = -2β²·S(l1,l2+2) + β(2l2+1)·S(l1,l2) - l2(l2-1)/2·S(l1,l2-2)
inline __host__ __device__
double kinetic_1d(int l1, int l2, double alpha, double beta, double AB) {
    double result = -2.0 * beta * beta * ovlp_1d(l1, l2 + 2, alpha, beta, AB)
                  + beta * (2 * l2 + 1) * ovlp_1d(l1, l2, alpha, beta, AB);
    if (l2 >= 2)
        result -= 0.5 * l2 * (l2 - 1) * ovlp_1d(l1, l2 - 2, alpha, beta, AB);
    return result;
}

// Full kinetic energy integral: T_μν = T_x·S_y·S_z + S_x·T_y·S_z + S_x·S_y·T_z
inline __host__ __device__
double kinetic_full(int l1, int m1, int n1, int l2, int m2, int n2,
                    double alpha, double beta,
                    double ABx, double ABy, double ABz) {
    double Sx = ovlp_1d(l1, l2, alpha, beta, ABx);
    double Sy = ovlp_1d(m1, m2, alpha, beta, ABy);
    double Sz = ovlp_1d(n1, n2, alpha, beta, ABz);
    double Tx = kinetic_1d(l1, l2, alpha, beta, ABx);
    double Ty = kinetic_1d(m1, m2, alpha, beta, ABy);
    double Tz = kinetic_1d(n1, n2, alpha, beta, ABz);
    return Tx * Sy * Sz + Sx * Ty * Sz + Sx * Sy * Tz;
}

__global__
void compute_hessian_kinetic(
    double* g_hessian,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const int num_basis,
    const int num_atoms,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    const size_t num_threads)
{
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_threads) return;

    const int ndim = 3 * num_atoms;

    size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t pidx_a = ab.y + shell_s0.start_index;
    const size_t pidx_b = ab.x + shell_s1.start_index;
    const PrimitiveShell a = g_shell[pidx_a];
    const PrimitiveShell b = g_shell[pidx_b];

    const size_t bi = a.basis_index, bj = b.basis_index;
    const double alpha = a.exponent, beta = b.exponent;
    const double ABx = a.coordinate.x - b.coordinate.x;
    const double ABy = a.coordinate.y - b.coordinate.y;
    const double ABz = a.coordinate.z - b.coordinate.z;

    double CoefBase = a.coefficient * b.coefficient * M_PI / (alpha + beta) * sqrt(M_PI / (alpha + beta));
    double factor = (pidx_a != pidx_b) ? 2.0 : 1.0;

    double hess_AA[9] = {0.0};

    for (int lmn_a = 0; lmn_a < comb_max(a.shell_type); lmn_a++) {
        int l1 = loop_to_ang[a.shell_type][lmn_a][0];
        int m1 = loop_to_ang[a.shell_type][lmn_a][1];
        int n1 = loop_to_ang[a.shell_type][lmn_a][2];
        double NA = calcNorm(alpha, l1, m1, n1);

        for (int lmn_b = 0; lmn_b < comb_max(b.shell_type); lmn_b++) {
            int l2 = loop_to_ang[b.shell_type][lmn_b][0];
            int m2 = loop_to_ang[b.shell_type][lmn_b][1];
            int n2 = loop_to_ang[b.shell_type][lmn_b][2];
            double NB = calcNorm(beta, l2, m2, n2);

            double D_comp = g_density_matrix[(bi + lmn_a) * num_basis + (bj + lmn_b)];
            if (fabs(D_comp) < 1e-18) continue;

            double w = CoefBase * g_cgto_normalization_factors[bi + lmn_a]
                     * g_cgto_normalization_factors[bj + lmn_b]
                     * factor * D_comp * NA * NB;

            // Second derivatives of T_μν using shift formula on A
            // d²T/dA_x dA_x: use T(l1+2) - 2α(2l1+1)T(l1) + l1(l1-1)T(l1-2)
            // (applied to the FULL kinetic integral T_μν)

            // d²T/dA_σ² = 4α² T(lσ1+2,...) - 2α(2lσ1+1) T(lσ1,...) + lσ1(lσ1-1) T(lσ1-2,...)
            double T_base = kinetic_full(l1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);

            double d2T_xx = 4*alpha*alpha * kinetic_full(l1+2, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*l1+1) * T_base
                          + (l1>1 ? l1*(l1-1) * kinetic_full(l1-2, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);
            double d2T_yy = 4*alpha*alpha * kinetic_full(l1, m1+2, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*m1+1) * T_base
                          + (m1>1 ? m1*(m1-1) * kinetic_full(l1, m1-2, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);
            double d2T_zz = 4*alpha*alpha * kinetic_full(l1, m1, n1+2, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*n1+1) * T_base
                          + (n1>1 ? n1*(n1-1) * kinetic_full(l1, m1, n1-2, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);

            // Cross: d²T/dA_x dA_y = dT/dA_x(shifted) * dT/dA_y(shifted)
            // = [2α T(l1+1,...) - l1 T(l1-1,...)] applied to x, then y
            double dT_x_p = kinetic_full(l1+1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_x_m = (l1>0) ? kinetic_full(l1-1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_x = 2*alpha * dT_x_p - l1 * dT_x_m;

            double dT_y_p = kinetic_full(l1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_y_m = (m1>0) ? kinetic_full(l1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_y = 2*alpha * dT_y_p - m1 * dT_y_m;

            double dT_z_p = kinetic_full(l1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_z_m = (n1>0) ? kinetic_full(l1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_z = 2*alpha * dT_z_p - n1 * dT_z_m;

            // d²T/dA_x dA_y: apply y-shift to x-shifted T
            double d2T_xy = 2*alpha * (2*alpha * kinetic_full(l1+1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (m1>0 ? m1 * kinetic_full(l1+1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (l1>0 ? l1 * (2*alpha * kinetic_full(l1-1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (m1>0 ? m1 * kinetic_full(l1-1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            double d2T_xz = 2*alpha * (2*alpha * kinetic_full(l1+1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1+1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (l1>0 ? l1 * (2*alpha * kinetic_full(l1-1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1-1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            double d2T_yz = 2*alpha * (2*alpha * kinetic_full(l1, m1+1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1, m1+1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (m1>0 ? m1 * (2*alpha * kinetic_full(l1, m1-1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1, m1-1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            hess_AA[0] += w * d2T_xx;
            hess_AA[4] += w * d2T_yy;
            hess_AA[8] += w * d2T_zz;
            hess_AA[1] += w * d2T_xy;
            hess_AA[2] += w * d2T_xz;
            hess_AA[5] += w * d2T_yz;
        }
    }

    hess_AA[3] = hess_AA[1];
    hess_AA[6] = hess_AA[2];
    hess_AA[7] = hess_AA[5];

    // Translational invariance: same as overlap
    int iA = a.atom_index, iB = b.atom_index;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = d1; d2 < 3; d2++) {
            double val = hess_AA[d1*3+d2];
            if (val == 0.0) continue;
            atomicAdd(&g_hessian[(3*iA+d1)*ndim + (3*iA+d2)], val);
            if (d1!=d2) atomicAdd(&g_hessian[(3*iA+d2)*ndim + (3*iA+d1)], val);
            atomicAdd(&g_hessian[(3*iB+d1)*ndim + (3*iB+d2)], val);
            if (d1!=d2) atomicAdd(&g_hessian[(3*iB+d2)*ndim + (3*iB+d1)], val);
            atomicAdd(&g_hessian[(3*iA+d1)*ndim + (3*iB+d2)], -val);
            if (d1!=d2) atomicAdd(&g_hessian[(3*iA+d2)*ndim + (3*iB+d1)], -val);
            atomicAdd(&g_hessian[(3*iB+d1)*ndim + (3*iA+d2)], -val);
            if (d1!=d2) atomicAdd(&g_hessian[(3*iB+d2)*ndim + (3*iA+d1)], -val);
        }
    }
}



// CPU host-callable mirror of compute_hessian_kinetic.  Auto-generated.
void compute_hessian_kinetic_cpu(double* g_hessian, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const int num_basis, const int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads)
{
    #pragma omp parallel for schedule(dynamic)
    for (long long _id_ll = 0; _id_ll < (long long)num_threads; _id_ll++) {
        const size_t id = (size_t)_id_ll;

    // (id loop injected by CPU launcher)

    const int ndim = 3 * num_atoms;

    size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t pidx_a = ab.y + shell_s0.start_index;
    const size_t pidx_b = ab.x + shell_s1.start_index;
    const PrimitiveShell a = g_shell[pidx_a];
    const PrimitiveShell b = g_shell[pidx_b];

    const size_t bi = a.basis_index, bj = b.basis_index;
    const double alpha = a.exponent, beta = b.exponent;
    const double ABx = a.coordinate.x - b.coordinate.x;
    const double ABy = a.coordinate.y - b.coordinate.y;
    const double ABz = a.coordinate.z - b.coordinate.z;

    double CoefBase = a.coefficient * b.coefficient * M_PI / (alpha + beta) * sqrt(M_PI / (alpha + beta));
    double factor = (pidx_a != pidx_b) ? 2.0 : 1.0;

    double hess_AA[9] = {0.0};

    for (int lmn_a = 0; lmn_a < comb_max(a.shell_type); lmn_a++) {
        int l1 = loop_to_ang_host[a.shell_type][lmn_a][0];
        int m1 = loop_to_ang_host[a.shell_type][lmn_a][1];
        int n1 = loop_to_ang_host[a.shell_type][lmn_a][2];
        double NA = calcNorm(alpha, l1, m1, n1);

        for (int lmn_b = 0; lmn_b < comb_max(b.shell_type); lmn_b++) {
            int l2 = loop_to_ang_host[b.shell_type][lmn_b][0];
            int m2 = loop_to_ang_host[b.shell_type][lmn_b][1];
            int n2 = loop_to_ang_host[b.shell_type][lmn_b][2];
            double NB = calcNorm(beta, l2, m2, n2);

            double D_comp = g_density_matrix[(bi + lmn_a) * num_basis + (bj + lmn_b)];
            if (fabs(D_comp) < 1e-18) continue;

            double w = CoefBase * g_cgto_normalization_factors[bi + lmn_a]
                     * g_cgto_normalization_factors[bj + lmn_b]
                     * factor * D_comp * NA * NB;

            // Second derivatives of T_μν using shift formula on A
            // d²T/dA_x dA_x: use T(l1+2) - 2α(2l1+1)T(l1) + l1(l1-1)T(l1-2)
            // (applied to the FULL kinetic integral T_μν)

            // d²T/dA_σ² = 4α² T(lσ1+2,...) - 2α(2lσ1+1) T(lσ1,...) + lσ1(lσ1-1) T(lσ1-2,...)
            double T_base = kinetic_full(l1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);

            double d2T_xx = 4*alpha*alpha * kinetic_full(l1+2, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*l1+1) * T_base
                          + (l1>1 ? l1*(l1-1) * kinetic_full(l1-2, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);
            double d2T_yy = 4*alpha*alpha * kinetic_full(l1, m1+2, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*m1+1) * T_base
                          + (m1>1 ? m1*(m1-1) * kinetic_full(l1, m1-2, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);
            double d2T_zz = 4*alpha*alpha * kinetic_full(l1, m1, n1+2, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - 2*alpha*(2*n1+1) * T_base
                          + (n1>1 ? n1*(n1-1) * kinetic_full(l1, m1, n1-2, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0);

            // Cross: d²T/dA_x dA_y = dT/dA_x(shifted) * dT/dA_y(shifted)
            // = [2α T(l1+1,...) - l1 T(l1-1,...)] applied to x, then y
            double dT_x_p = kinetic_full(l1+1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_x_m = (l1>0) ? kinetic_full(l1-1, m1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_x = 2*alpha * dT_x_p - l1 * dT_x_m;

            double dT_y_p = kinetic_full(l1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_y_m = (m1>0) ? kinetic_full(l1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_y = 2*alpha * dT_y_p - m1 * dT_y_m;

            double dT_z_p = kinetic_full(l1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz);
            double dT_z_m = (n1>0) ? kinetic_full(l1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0;
            double dT_z = 2*alpha * dT_z_p - n1 * dT_z_m;

            // d²T/dA_x dA_y: apply y-shift to x-shifted T
            double d2T_xy = 2*alpha * (2*alpha * kinetic_full(l1+1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (m1>0 ? m1 * kinetic_full(l1+1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (l1>0 ? l1 * (2*alpha * kinetic_full(l1-1, m1+1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (m1>0 ? m1 * kinetic_full(l1-1, m1-1, n1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            double d2T_xz = 2*alpha * (2*alpha * kinetic_full(l1+1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1+1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (l1>0 ? l1 * (2*alpha * kinetic_full(l1-1, m1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1-1, m1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            double d2T_yz = 2*alpha * (2*alpha * kinetic_full(l1, m1+1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1, m1+1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0))
                          - (m1>0 ? m1 * (2*alpha * kinetic_full(l1, m1-1, n1+1, l2, m2, n2, alpha, beta, ABx, ABy, ABz)
                          - (n1>0 ? n1 * kinetic_full(l1, m1-1, n1-1, l2, m2, n2, alpha, beta, ABx, ABy, ABz) : 0)) : 0);

            hess_AA[0] += w * d2T_xx;
            hess_AA[4] += w * d2T_yy;
            hess_AA[8] += w * d2T_zz;
            hess_AA[1] += w * d2T_xy;
            hess_AA[2] += w * d2T_xz;
            hess_AA[5] += w * d2T_yz;
        }
    }

    hess_AA[3] = hess_AA[1];
    hess_AA[6] = hess_AA[2];
    hess_AA[7] = hess_AA[5];

    // Translational invariance: same as overlap
    int iA = a.atom_index, iB = b.atom_index;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = d1; d2 < 3; d2++) {
            double val = hess_AA[d1*3+d2];
            if (val == 0.0) continue;
            gansu_atomic_add(&g_hessian[(3*iA+d1)*ndim + (3*iA+d2)], val);
            if (d1!=d2) gansu_atomic_add(&g_hessian[(3*iA+d2)*ndim + (3*iA+d1)], val);
            gansu_atomic_add(&g_hessian[(3*iB+d1)*ndim + (3*iB+d2)], val);
            if (d1!=d2) gansu_atomic_add(&g_hessian[(3*iB+d2)*ndim + (3*iB+d1)], val);
            gansu_atomic_add(&g_hessian[(3*iA+d1)*ndim + (3*iB+d2)], -val);
            if (d1!=d2) gansu_atomic_add(&g_hessian[(3*iA+d2)*ndim + (3*iB+d1)], -val);
            gansu_atomic_add(&g_hessian[(3*iB+d1)*ndim + (3*iA+d2)], -val);
            if (d1!=d2) gansu_atomic_add(&g_hessian[(3*iB+d2)*ndim + (3*iA+d1)], -val);
        }
    }
    }
}

} // namespace gansu::gpu
