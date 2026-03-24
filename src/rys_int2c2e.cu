/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "rys_int2c2e.hpp"
#include "rys_quadrature.hpp"
#include "int2c2e.hpp"     // addToResult_2center
#include "int2e.hpp"       // comb_max, loop_to_ang, calcNorm, M_PI_2_5
#include "utils_cuda.hpp"  // index1to2

namespace gansu::gpu {

// --- VRR for 2-center: I(a) only, no ket ---
// For 2-center (A|B), both are single functions.
// VRR builds I(a_total) where a_total = la + lb.
// Then TRR splits into I(la, lb).
//
// But actually, (A|B) has the same structure as a 4-center ERI
// with the bra being function A and ket being function B.
// rho = alpha*beta/(alpha+beta), T = rho * |A-B|^2
// VRR builds I(a, c) where a = 0..la, c = 0..lb
inline __device__
void vrr_1d_2c(int a_max, int c_max,
               double C00, double D00,
               double B10, double B01, double B00,
               double* __restrict__ I) {
    const int cs = c_max + 1;
    I[0] = 1.0;
    if (a_max > 0) {
        I[cs] = C00;
        for (int a = 1; a < a_max; a++)
            I[(a+1)*cs] = C00*I[a*cs] + a*B10*I[(a-1)*cs];
    }
    for (int c = 0; c < c_max; c++) {
        double cB01 = c * B01;
        I[c+1] = D00*I[c] + ((c > 0) ? cB01*I[c-1] : 0.0);
        for (int a = 1; a <= a_max; a++)
            I[a*cs+c+1] = D00*I[a*cs+c]
                        + ((c > 0) ? cB01*I[a*cs+c-1] : 0.0)
                        + a*B00*I[(a-1)*cs+c];
    }
}

// ============================================================
//  Rys 2-center ERI kernel: (A|B) over auxiliary basis
//
//  Structure mirrors MD_int2c2e_1T1SP:
//  - Each function is a single Gaussian (no bra/ket pairing)
//  - A at alpha, center A_pos; B at beta, center B_pos
//  - rho = alpha*beta/(alpha+beta), T = rho*|A-B|^2
//  - No TRR needed: VRR directly builds I(la, lb) using
//    "bra" = A, "ket" = B
// ============================================================
__global__
void Rys_int2c2e(
    real_t* g_result,
    const PrimitiveShell* g_pshell_aux,
    const real_t* d_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    int num_shell_pairs,
    const double* g_upper_bound_factors,
    const double schwarz_screening_threshold,
    int num_auxiliary_basis,
    const double* g_boys_grid)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_shell_pairs) return;

    // Thread mapping (same as MD_int2c2e_1T1SP)
    size_t2 ab = index1to2(id, (shell_s0.start_index == shell_s1.start_index), shell_s1.count);
    const size_t pidx_a = ab.x + shell_s0.start_index;
    const size_t pidx_b = ab.y + shell_s1.start_index;

    // Screening
    if (g_upper_bound_factors[pidx_a] * g_upper_bound_factors[pidx_b]
        < schwarz_screening_threshold) return;

    const PrimitiveShell sa = g_pshell_aux[pidx_a];
    const PrimitiveShell sb = g_pshell_aux[pidx_b];

    const size_t base_a = sa.basis_index, base_b = sb.basis_index;
    const bool is_prim_id_not_equal = (pidx_a != pidx_b);

    const double alpha = sa.exponent, beta = sb.exponent;
    const double rho = alpha * beta / (alpha + beta);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double AB2 = ABx*ABx + ABy*ABy + ABz*ABz;
    const double T = rho * AB2;

    const int la = sa.shell_type, lb = sb.shell_type;
    const int L = la + lb;

    // Prefactor: 2*pi^(5/2) / (alpha*beta*sqrt(alpha+beta))
    // No K_AB/K_CD pairing exponential: both functions are single Gaussians
    // The overlap factor is implicit in the Rys quadrature via VRR
    const double prefactor = 2.0 * M_PI_2_5 / (alpha * beta * sqrt(alpha + beta))
                           * sa.coefficient * sb.coefficient;

    // Rys roots/weights
    const int N = L / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // VRR dimensions: a = 0..la, c = 0..lb (no TRR needed)
    const int cs = lb + 1;
    double Ix[81], Iy[81], Iz[81]; // max (4+1)*(4+1) = 25 for g-shell

    const int na = comb_max(la), nb = comb_max(lb);

    // Component loop
    for (int ia = 0; ia < na; ia++) {
        const int ax = loop_to_ang[la][ia][0], ay = loop_to_ang[la][ia][1], az = loop_to_ang[la][ia][2];
        const double Norm_A = calcNorm(alpha, ax, ay, az);

        for (int ib = 0; ib < nb; ib++) {
            const int bx = loop_to_ang[lb][ib][0], by = loop_to_ang[lb][ib][1], bz = loop_to_ang[lb][ib][2];
            const double Norm_B = calcNorm(beta, bx, by, bz);

            double eri_value = 0.0;

            for (int n = 0; n < N; n++) {
                const double t2 = rys_roots[n];
                const double wn = rys_weights[n];
                const double u = rho * t2;
                const double u_over_a = u / alpha;
                const double u_over_b = u / beta;

                const double B00 = t2 / (2.0*(alpha + beta));
                const double B10 = (1.0 - u_over_a) / (2.0*alpha);
                const double B01 = (1.0 - u_over_b) / (2.0*beta);

                // For 2-center: "P" is a dummy (single function A)
                // C00 = u_over_a * (B - A)  [shift from A toward B]
                // D00 = u_over_b * (A - B)  [shift from B toward A]
                const double C00x = u_over_a * (Bx - Ax);
                const double C00y = u_over_a * (By - Ay);
                const double C00z = u_over_a * (Bz - Az);
                const double D00x = u_over_b * (Ax - Bx);
                const double D00y = u_over_b * (Ay - By);
                const double D00z = u_over_b * (Az - Bz);

                vrr_1d_2c(la, lb, C00x, D00x, B10, B01, B00, Ix);
                vrr_1d_2c(la, lb, C00y, D00y, B10, B01, B00, Iy);
                vrr_1d_2c(la, lb, C00z, D00z, B10, B01, B00, Iz);

                // Direct access: I(ax, bx) from VRR
                double Ix_val = Ix[ax * cs + bx];
                double Iy_val = Iy[ay * cs + by];
                double Iz_val = Iz[az * cs + bz];

                eri_value += wn * Ix_val * Iy_val * Iz_val;
            }

            addToResult_2center(
                Norm_A * Norm_B * prefactor * eri_value,
                g_result,
                base_a + ia, base_b + ib,
                num_auxiliary_basis,
                is_prim_id_not_equal,
                d_auxiliary_cgto_normalization_factors);
        }
    }
}

} // namespace gansu::gpu
