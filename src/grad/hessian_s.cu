/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Overlap integral Hessian: d²S/dA dB
//
// Uses the angular momentum shift formula on S_μν:
//   S_μν = Πσ Et_0(lσ1, lσ2, α, β, ABσ)
//
// d/dA_x S = 2α·S(l₁+1,...) - l₁·S(l₁-1,...)
// d²/dA_x dA_y = [shift_x] × [shift_y]  (product rule, different directions)
// d²/dA_x² = second shift formula
//
// Translational invariance: d/dB = -d/dA, so:
//   d²/dA dB = -d²/dA dA (AB block)
//   d²/dB dB = d²/dA dA  (BB = AA)

#include <cuda.h>
#include <cmath>
#include "gradients.hpp"

namespace gansu::gpu {

// Compute overlap integral for one direction with shifted angular momentum
inline __device__
double overlap_1d(int l1, int l2, double alpha, double beta, double AB) {
    return MD_Et_NonRecursion(l1, l2, 0, alpha, beta, AB);
}

__global__
void compute_hessian_overlap(
    double* g_hessian,
    const real_t* g_W_matrix,
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

    // Accumulate hessian for this primitive pair: [AA] block only
    // Other blocks derived from translational invariance
    double hess_AA[9] = {0.0}; // [3×3] for (Ax,Ay,Az) × (Ax,Ay,Az)

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

            double W_comp = -g_W_matrix[(bi + lmn_a) * num_basis + (bj + lmn_b)];
            if (fabs(W_comp) < 1e-18) continue;

            double w = CoefBase * g_cgto_normalization_factors[bi + lmn_a]
                     * g_cgto_normalization_factors[bj + lmn_b]
                     * factor * W_comp * NA * NB;

            // Base overlap in each direction
            double Sx = overlap_1d(l1, l2, alpha, beta, ABx);
            double Sy = overlap_1d(m1, m2, alpha, beta, ABy);
            double Sz = overlap_1d(n1, n2, alpha, beta, ABz);

            // First derivatives dS/dA (shift on center A)
            double dSx = 2 * alpha * overlap_1d(l1 + 1, l2, alpha, beta, ABx)
                       - (l1 > 0 ? l1 * overlap_1d(l1 - 1, l2, alpha, beta, ABx) : 0.0);
            double dSy = 2 * alpha * overlap_1d(m1 + 1, m2, alpha, beta, ABy)
                       - (m1 > 0 ? m1 * overlap_1d(m1 - 1, m2, alpha, beta, ABy) : 0.0);
            double dSz = 2 * alpha * overlap_1d(n1 + 1, n2, alpha, beta, ABz)
                       - (n1 > 0 ? n1 * overlap_1d(n1 - 1, n2, alpha, beta, ABz) : 0.0);

            // Second derivatives d²S/dA² (same direction)
            double d2Sx = 4 * alpha * alpha * overlap_1d(l1 + 2, l2, alpha, beta, ABx)
                        - 2 * alpha * (2 * l1 + 1) * Sx
                        + (l1 > 1 ? l1 * (l1 - 1) * overlap_1d(l1 - 2, l2, alpha, beta, ABx) : 0.0);
            double d2Sy = 4 * alpha * alpha * overlap_1d(m1 + 2, m2, alpha, beta, ABy)
                        - 2 * alpha * (2 * m1 + 1) * Sy
                        + (m1 > 1 ? m1 * (m1 - 1) * overlap_1d(m1 - 2, m2, alpha, beta, ABy) : 0.0);
            double d2Sz = 4 * alpha * alpha * overlap_1d(n1 + 2, n2, alpha, beta, ABz)
                        - 2 * alpha * (2 * n1 + 1) * Sz
                        + (n1 > 1 ? n1 * (n1 - 1) * overlap_1d(n1 - 2, n2, alpha, beta, ABz) : 0.0);

            // d²S/dA_x dA_x = d2Sx * Sy * Sz
            hess_AA[0] += w * d2Sx * Sy * Sz;  // xx
            hess_AA[4] += w * Sx * d2Sy * Sz;  // yy
            hess_AA[8] += w * Sx * Sy * d2Sz;  // zz

            // d²S/dA_x dA_y = dSx * dSy * Sz
            hess_AA[1] += w * dSx * dSy * Sz;  // xy
            hess_AA[2] += w * dSx * Sy * dSz;  // xz
            hess_AA[5] += w * Sx * dSy * dSz;  // yz
        }
    }

    // Symmetrize AA block
    hess_AA[3] = hess_AA[1]; // yx = xy
    hess_AA[6] = hess_AA[2]; // zx = xz
    hess_AA[7] = hess_AA[5]; // zy = yz

    // Write to global Hessian using translational invariance:
    // H[AA] = d²/dA dA, H[AB] = -d²/dA dA, H[BB] = d²/dA dA
    int iA = a.atom_index, iB = b.atom_index;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = d1; d2 < 3; d2++) {
            double val = hess_AA[d1 * 3 + d2];
            if (val == 0.0) continue;

            int gA1 = 3 * iA + d1, gA2 = 3 * iA + d2;
            int gB1 = 3 * iB + d1, gB2 = 3 * iB + d2;

            // AA block
            atomicAdd(&g_hessian[gA1 * ndim + gA2], val);
            if (d1 != d2) atomicAdd(&g_hessian[gA2 * ndim + gA1], val);

            // BB block (same as AA)
            atomicAdd(&g_hessian[gB1 * ndim + gB2], val);
            if (d1 != d2) atomicAdd(&g_hessian[gB2 * ndim + gB1], val);

            // AB block (negative of AA)
            atomicAdd(&g_hessian[gA1 * ndim + gB2], -val);
            if (d1 != d2) atomicAdd(&g_hessian[gA2 * ndim + gB1], -val);

            // BA block (negative of AA)
            atomicAdd(&g_hessian[gB1 * ndim + gA2], -val);
            if (d1 != d2) atomicAdd(&g_hessian[gB2 * ndim + gA1], -val);
        }
    }
}

} // namespace gansu::gpu
