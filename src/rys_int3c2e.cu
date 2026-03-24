/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "rys_int3c2e.hpp"
#include "rys_quadrature.hpp"
#include "int3c2e.hpp"     // addToResult_3center, calcNormsWOFact2_3center
#include "int2e.hpp"       // comb_max, loop_to_ang, calcNorm, M_PI_2_5
#include "utils_cuda.hpp"  // index1to2

namespace gansu::gpu {

// --- VRR (same as rys_eri.cu) ---
inline __device__
void vrr_1d_3c(int a_max, int c_max,
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

// --- Ket TRR (iterative) ---
inline __device__
double trr_ket_3c(int c, int d, double CD, const double* __restrict__ vals) {
    double buf[9];
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d-dd-1; i++)
            buf[i] = buf[i+1] + CD*buf[i];
    return buf[0];
}

// ============================================================
//  Rys 3-center ERI kernel: (pq|A)
//
//  Structure mirrors MD_int3c2e_1T1SP:
//  - Thread mapping: d_primitive_shell_pair_indices for (p,q), linear for A
//  - Schwarz screening
//  - addToResult_3center for output
//
//  Rys quadrature replaces McMurchie-Davidson:
//  - Bra: Gaussian product of (alpha, A) and (beta, B) → p, P
//  - Ket: single auxiliary function (gamma, C) → Q = C
//  - No ket pairing, no K_CD (single function)
// ============================================================
__global__
void Rys_int3c2e(
    real_t* g_result,
    const PrimitiveShell* g_pshell,
    const PrimitiveShell* g_pshell_aux,
    const real_t* d_cgto_normalization_factors,
    const real_t* d_auxiliary_cgto_normalization_factors,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2,
    int64_t num_tasks,
    int num_basis,
    const size_t2* d_primitive_shell_pair_indices,
    const double* g_upper_bound_factors,
    const double* g_auxiliary_upper_bound_factors,
    const double schwarz_screening_threshold,
    int num_auxiliary_basis,
    const double* g_boys_grid)
{
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_tasks) return;

    // --- Thread mapping (same as MD_int3c2e_1T1SP) ---
    const size_t2 abc = index1to2(id, false, shell_s2.count);

    // Bra pair from pre-indexed array
    const size_t pidx_a = d_primitive_shell_pair_indices[abc.x].x + shell_s0.start_index;
    const size_t pidx_b = d_primitive_shell_pair_indices[abc.x].y + shell_s1.start_index;
    const size_t pidx_c = abc.y + shell_s2.start_index;

    // Screening
    if (g_upper_bound_factors[abc.x] * g_auxiliary_upper_bound_factors[pidx_c]
        < schwarz_screening_threshold) return;

    const PrimitiveShell sa = g_pshell[pidx_a];
    const PrimitiveShell sb = g_pshell[pidx_b];
    const PrimitiveShell sc = g_pshell_aux[pidx_c];

    const size_t base_a = sa.basis_index, base_b = sb.basis_index, base_c = sc.basis_index;
    const bool is_prim_id_not_equal = (pidx_a != pidx_b);

    // --- Gaussian Product Theorem ---
    const double alpha = sa.exponent, beta = sb.exponent, gamma_exp = sc.exponent;
    const double p = alpha + beta;
    const double rho = p * gamma_exp / (p + gamma_exp);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;

    const double Px = (alpha*Ax + beta*Bx)/p;
    const double Py = (alpha*Ay + beta*By)/p;
    const double Pz = (alpha*Az + beta*Bz)/p;

    const double PCx = Px-Cx, PCy = Py-Cy, PCz = Pz-Cz;
    const double T = rho * (PCx*PCx + PCy*PCy + PCz*PCz);

    const int la = sa.shell_type, lb = sb.shell_type, lc = sc.shell_type;
    const int L = la + lb + lc;

    // --- Prefactor ---
    const double AB2 = (Ax-Bx)*(Ax-Bx) + (Ay-By)*(Ay-By) + (Az-Bz)*(Az-Bz);
    const double K_AB = exp(-alpha*beta/p * AB2);
    // No K_CD for 3-center (single auxiliary function)
    const double prefactor = 2.0 * M_PI_2_5 / (p * gamma_exp * sqrt(p + gamma_exp))
                           * K_AB
                           * sa.coefficient * sb.coefficient * sc.coefficient;

    // --- Rys roots/weights ---
    const int N = L / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // --- Dimensions ---
    const int a_max = la + lb;  // VRR bra range
    const int c_max_val = lc;   // VRR ket range (auxiliary, no pairing)
    const int cs = c_max_val + 1;
    const int na = comb_max(la), nb = comb_max(lb), nc = comb_max(lc);

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;

    // VRR workspace: max (8+1)*(4+1) = 45 per direction for d+f basis + g aux
    double Ix[81], Iy[81], Iz[81];

    // --- Component loop ---
    for (int ia = 0; ia < na; ia++) {
        const int ax = loop_to_ang[la][ia][0], ay = loop_to_ang[la][ia][1], az = loop_to_ang[la][ia][2];
        const double Norm_A = calcNorm(alpha, ax, ay, az);

        for (int ib = 0; ib < nb; ib++) {
            const int bx = loop_to_ang[lb][ib][0], by = loop_to_ang[lb][ib][1], bz = loop_to_ang[lb][ib][2];
            const double Norm_B = calcNorm(beta, bx, by, bz);

            for (int ic = 0; ic < nc; ic++) {
                const int cx = loop_to_ang[lc][ic][0], cy = loop_to_ang[lc][ic][1], cz = loop_to_ang[lc][ic][2];
                const double Norm_C = calcNorm(gamma_exp, cx, cy, cz);

                double eri_value = 0.0;

                // Sum over Rys roots
                for (int n = 0; n < N; n++) {
                    const double t2 = rys_roots[n];
                    const double wn = rys_weights[n];
                    const double u = rho * t2;
                    const double u_over_p = u / p;
                    const double u_over_q = u / gamma_exp;

                    const double B00 = t2 / (2.0*(p + gamma_exp));
                    const double B10 = (1.0 - u_over_p) / (2.0*p);
                    const double B01 = (1.0 - u_over_q) / (2.0*gamma_exp);

                    const double C00x = (Px-Ax) + u_over_p*(Cx-Px);
                    const double C00y = (Py-Ay) + u_over_p*(Cy-Py);
                    const double C00z = (Pz-Az) + u_over_p*(Cz-Pz);
                    // D00: ket center shift. Q=C, so D00 = u_over_q * (P-C)
                    const double D00x = u_over_q*(Px-Cx);
                    const double D00y = u_over_q*(Py-Cy);
                    const double D00z = u_over_q*(Pz-Cz);

                    vrr_1d_3c(a_max, c_max_val, C00x, D00x, B10, B01, B00, Ix);
                    vrr_1d_3c(a_max, c_max_val, C00y, D00y, B10, B01, B00, Iy);
                    vrr_1d_3c(a_max, c_max_val, C00z, D00z, B10, B01, B00, Iz);

                    // Bra TRR (binomial expansion) for each direction
                    // I(ax, bx, cx): bra transfers a→(a,b), ket has no transfer (ld=0)
                    double Ix_bra = 0.0, Iy_bra = 0.0, Iz_bra = 0.0;
                    {
                        int binom=1; double apow=1.0;
                        for (int k=0; k<=bx; k++) {
                            Ix_bra += binom*apow*Ix[(ax+bx-k)*cs + cx];
                            if(k<bx){apow*=ABx; binom=binom*(bx-k)/(k+1);}
                        }
                    }
                    {
                        int binom=1; double apow=1.0;
                        for (int k=0; k<=by; k++) {
                            Iy_bra += binom*apow*Iy[(ay+by-k)*cs + cy];
                            if(k<by){apow*=ABy; binom=binom*(by-k)/(k+1);}
                        }
                    }
                    {
                        int binom=1; double apow=1.0;
                        for (int k=0; k<=bz; k++) {
                            Iz_bra += binom*apow*Iz[(az+bz-k)*cs + cz];
                            if(k<bz){apow*=ABz; binom=binom*(bz-k)/(k+1);}
                        }
                    }

                    eri_value += wn * Ix_bra * Iy_bra * Iz_bra;
                }

                // Write result
                addToResult_3center(
                    Norm_A * Norm_B * Norm_C * prefactor * eri_value,
                    g_result,
                    base_a + ia, base_b + ib, base_c + ic,
                    num_basis, num_auxiliary_basis,
                    is_prim_id_not_equal,
                    d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors);
            }
        }
    }
}

} // namespace gansu::gpu
