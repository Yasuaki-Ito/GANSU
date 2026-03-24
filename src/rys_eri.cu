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

#include "rys_eri.hpp"
#include "rys_quadrature.hpp"
#include "int2e.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

// ============================================================
//  VRR: Build 2D integrals I(a, c) for one Cartesian direction
//  Layout: I[a * cs + c], cs = c_max + 1
// ============================================================
inline __device__
void vrr_1d(int a_max, int c_max,
            double C00, double D00,
            double B10, double B01, double B00,
            double* __restrict__ I) {
    const int cs = c_max + 1;
    I[0] = 1.0;

    // Build I(a, 0)
    if (a_max > 0) {
        I[cs] = C00;
        for (int a = 1; a < a_max; a++) {
            I[(a + 1) * cs] = C00 * I[a * cs] + a * B10 * I[(a - 1) * cs];
        }
    }

    // Build I(a, c+1) from I(a, c)
    for (int c = 0; c < c_max; c++) {
        double cB01 = c * B01;
        I[c + 1] = D00 * I[c] + ((c > 0) ? cB01 * I[c - 1] : 0.0);
        for (int a = 1; a <= a_max; a++) {
            I[a * cs + c + 1] = D00 * I[a * cs + c]
                              + ((c > 0) ? cB01 * I[a * cs + c - 1] : 0.0)
                              + a * B00 * I[(a - 1) * cs + c];
        }
    }
}

// ============================================================
//  Iterative bra TRR: expand I_vrr(a_total, c_total) to
//  I_bra(a, b, c_total) for a=0..la, b=0..lb.
//
//  I(a, 0, ct) = I_vrr(a, ct)
//  I(a, b+1, ct) = I(a+1, b, ct) + AB * I(a, b, ct)
//
//  Layout: bra[(b * (la+1) + a) * ket_stride + ct]
//  Requires: VRR has a_total = la+lb entries in a direction.
// ============================================================
inline __device__
void trr_bra_expand(int la, int lb, int ket_stride,
                    double AB, const double* __restrict__ vrr,
                    double* __restrict__ bra) {
    const int a_stride = ket_stride;
    const int b_stride = (la + 1) * a_stride;

    // b = 0: copy from VRR (a = 0..la+lb, but we only need 0..la for b=0
    //        and up to la+lb for recursion)
    // Actually, for b=0, I(a, 0, ct) = vrr[a * vrr_cs + ct]
    // We need I(a, 0, ct) for a = 0..la (for output) and a = 0..la+lb (for recursion)
    // Store a = 0..la in bra, use vrr directly for a > la in recursion.

    // Initialize b=0 level
    for (int a = 0; a <= la; a++) {
        for (int ct = 0; ct < ket_stride; ct++) {
            bra[a * a_stride + ct] = vrr[a * ket_stride + ct];
        }
    }

    // Build b = 1..lb
    for (int b = 0; b < lb; b++) {
        for (int a = 0; a <= la; a++) {
            for (int ct = 0; ct < ket_stride; ct++) {
                // I(a, b+1, ct) = I(a+1, b, ct) + AB * I(a, b, ct)
                double I_ab = bra[b * b_stride + a * a_stride + ct];
                double I_a1b;
                if (b == 0) {
                    // I(a+1, 0, ct) from VRR directly
                    I_a1b = vrr[(a + 1) * ket_stride + ct];
                } else {
                    // I(a+1, b, ct) — need a+1 <= la for current b level
                    // But a+1 can exceed la. We need extended storage.
                    // Workaround: for b >= 1, we stored only a=0..la.
                    // Need a+1, so a must be <= la-1 for b >= 1... but a goes to la.
                    // Fix: store a = 0..la+lb-b for each b level.
                    // Simpler fix: use a separate workspace that extends a range.
                    // For now, access bra with extended a (store la+lb-b+1 entries per b).
                    I_a1b = bra[b * b_stride + (a + 1) * a_stride + ct];
                }
                bra[(b + 1) * b_stride + a * a_stride + ct] = I_a1b + AB * I_ab;
            }
        }
        // Also compute extended a values for next recursion level
        // I(a, b+1) for a = la+1..la+lb-b-1 (needed for next b level)
        if (b + 1 < lb) {
            for (int a = la + 1; a <= la + lb - b - 1; a++) {
                for (int ct = 0; ct < ket_stride; ct++) {
                    double I_ab, I_a1b;
                    if (b == 0) {
                        I_ab = vrr[a * ket_stride + ct];
                        I_a1b = vrr[(a + 1) * ket_stride + ct];
                    } else {
                        I_ab = bra[b * b_stride + a * a_stride + ct];
                        I_a1b = bra[b * b_stride + (a + 1) * a_stride + ct];
                    }
                    bra[(b + 1) * b_stride + a * a_stride + ct] = I_a1b + AB * I_ab;
                }
            }
        }
    }
}

// ============================================================
//  Iterative ket TRR: expand I(c_total) to I(c, d)
//  I(ct, 0) = input[ct - c_base]
//  I(ct, d+1) = I(ct+1, d) + CD * I(ct, d)
//
//  For a single (a,b) pair, computes I(c, d) from the bra-expanded values.
// ============================================================
inline __device__
double trr_ket_eval(int c, int d, double CD, const double* __restrict__ vals) {
    // vals[0..d] = I(c, 0), I(c+1, 0), ..., I(c+d, 0)
    // Build I(c, d) iteratively
    // Use a small local buffer
    double buf[9]; // max d+1 = 5 for g-shell
    for (int i = 0; i <= d; i++) buf[i] = vals[i];

    for (int dd = 0; dd < d; dd++) {
        // Reduce: buf[i] = buf[i+1] + CD * buf[i], for i = 0..d-dd-1
        for (int i = 0; i <= d - dd - 1; i++) {
            buf[i] = buf[i + 1] + CD * buf[i];
        }
    }
    return buf[0];
}

// ============================================================
//  RysERI kernel — Optimized version
//
//  Key optimizations over initial implementation:
//  1. Root loop outside component loop: VRR computed once per root
//  2. Iterative TRR (no recursive branching)
//  3. Results accumulated in buffer, written once per component
// ============================================================
__global__
void RysERI(
    double* g_int2e,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const double* g_upper_bound_factors,
    const int num_basis,
    const double* g_boys_grid,
    const size_t head_bra, const size_t head_ket)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    // --- Thread → shell quartet mapping ---
    int ket_size;
    if (shell_s2.start_index == shell_s3.start_index) {
        ket_size = (shell_s2.count * (shell_s2.count + 1)) / 2;
    } else {
        ket_size = shell_s2.count * shell_s3.count;
    }
    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // --- Schwarz screening ---
    if (g_upper_bound_factors[head_bra + abcd.x] *
        g_upper_bound_factors[head_ket + abcd.y] < schwarz_screening_threshold) {
        return;
    }

    // --- Fetch primitive shells ---
    const size_t pidx_a = ab.x + shell_s0.start_index;
    const size_t pidx_b = ab.y + shell_s1.start_index;
    const size_t pidx_c = cd.x + shell_s2.start_index;
    const size_t pidx_d = cd.y + shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a];
    const PrimitiveShell sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c];
    const PrimitiveShell sd = g_shell[pidx_d];

    const size_t base_a = sa.basis_index;
    const size_t base_b = sb.basis_index;
    const size_t base_c = sc.basis_index;
    const size_t base_d = sd.basis_index;

    const bool sym_bra    = (pidx_a == pidx_b);
    const bool sym_ket    = (pidx_c == pidx_d);
    const bool sym_braket = (pidx_a == pidx_c && pidx_b == pidx_d);

    // --- Exponents, coefficients ---
    const double alpha = sa.exponent, beta = sb.exponent;
    const double gamma = sc.exponent, delta_exp = sd.exponent;
    const double p = alpha + beta;
    const double q = gamma + delta_exp;
    const double rho = p * q / (p + q);

    // --- Coordinates ---
    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;
    const double Dx = sd.coordinate.x, Dy = sd.coordinate.y, Dz = sd.coordinate.z;

    const double Px = (alpha * Ax + beta * Bx) / p;
    const double Py = (alpha * Ay + beta * By) / p;
    const double Pz = (alpha * Az + beta * Bz) / p;
    const double Qx = (gamma * Cx + delta_exp * Dx) / q;
    const double Qy = (gamma * Cy + delta_exp * Dy) / q;
    const double Qz = (gamma * Cz + delta_exp * Dz) / q;

    const double PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
    const double T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);

    const int la = sa.shell_type, lb = sb.shell_type;
    const int lc = sc.shell_type, ld = sd.shell_type;
    const int L = la + lb + lc + ld;

    // --- Prefactor ---
    const double AB2 = (Ax-Bx)*(Ax-Bx) + (Ay-By)*(Ay-By) + (Az-Bz)*(Az-Bz);
    const double CD2 = (Cx-Dx)*(Cx-Dx) + (Cy-Dy)*(Cy-Dy) + (Cz-Dz)*(Cz-Dz);
    const double prefactor = 2.0 * M_PI_2_5 / (p * q * sqrt(p + q))
                           * exp(-alpha * beta / p * AB2 - gamma * delta_exp / q * CD2)
                           * sa.coefficient * sb.coefficient * sc.coefficient * sd.coefficient;

    // --- Rys roots and weights ---
    const int N = L / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // --- Dimensions ---
    const int a_max = la + lb;
    const int c_max_val = lc + ld;
    const int cs = c_max_val + 1;
    const int na = comb_max(la), nb = comb_max(lb);
    const int nc = comb_max(lc), nd = comb_max(ld);
    const int n_components = na * nb * nc * nd;

    // TRR geometry (root-independent)
    const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
    const double CDx = Cx - Dx, CDy = Cy - Dy, CDz = Cz - Dz;

    // --- Accumulation buffer: eri[ia * nb*nc*nd + ib * nc*nd + ic * nd + id] ---
    // Max: dd|dd = 1296, ff|ff = 10000, limited by local memory
    double eri_buf[1296]; // sufficient up to dd|dd; for higher L, use direct write
    const bool use_buffer = (n_components <= 1296);

    if (use_buffer) {
        for (int i = 0; i < n_components; i++) eri_buf[i] = 0.0;
    }

    // --- VRR workspace (reused per root) ---
    double Ix_vrr[81], Iy_vrr[81], Iz_vrr[81]; // (a_max+1)*(c_max+1), max 9*9=81

    // --- Main loop: iterate over Rys roots ---
    for (int n = 0; n < N; n++) {
        const double t2 = rys_roots[n];
        const double wn = rys_weights[n];
        const double u = rho * t2;
        const double u_over_p = u / p;
        const double u_over_q = u / q;

        const double B00 = t2 / (2.0 * (p + q));
        const double B10 = (1.0 - u_over_p) / (2.0 * p);
        const double B01 = (1.0 - u_over_q) / (2.0 * q);

        const double C00x = (Px - Ax) + u_over_p * (Qx - Px);
        const double C00y = (Py - Ay) + u_over_p * (Qy - Py);
        const double C00z = (Pz - Az) + u_over_p * (Qz - Pz);
        const double D00x = (Qx - Cx) + u_over_q * (Px - Qx);
        const double D00y = (Qy - Cy) + u_over_q * (Py - Qy);
        const double D00z = (Qz - Cz) + u_over_q * (Pz - Qz);

        // VRR: compute once per root
        vrr_1d(a_max, c_max_val, C00x, D00x, B10, B01, B00, Ix_vrr);
        vrr_1d(a_max, c_max_val, C00y, D00y, B10, B01, B00, Iy_vrr);
        vrr_1d(a_max, c_max_val, C00z, D00z, B10, B01, B00, Iz_vrr);

        // Accumulate all component integrals for this root
        for (int ia = 0; ia < na; ia++) {
            const int ax = loop_to_ang[la][ia][0];
            const int ay = loop_to_ang[la][ia][1];
            const int az = loop_to_ang[la][ia][2];

            for (int ib = 0; ib < nb; ib++) {
                const int bx = loop_to_ang[lb][ib][0];
                const int by = loop_to_ang[lb][ib][1];
                const int bz = loop_to_ang[lb][ib][2];

                // Bra TRR for x: I(ax, bx, ct) for ct = 0..c_max_val
                // Compute iteratively using a small workspace
                double bra_x[9]; // max lc+ld+1 = 9
                for (int ct = 0; ct <= c_max_val; ct++) {
                    // Iterative bra TRR: I(a, b) = I(a+1, b-1) + AB * I(a, b-1)
                    // Use Newton's binomial: I(a,b,ct) = sum_{k=0}^{b} C(b,k) * AB^(b-k) * I(a+k, ct)
                    double val = 0.0;
                    double ab_power = 1.0; // AB^(b-k) for k starting from b down to 0
                    // Binomial expansion: I(a,b) = sum_{k=0}^{b} C(b,k) * AB^k * I_vrr(a+b-k, ct)
                    // where C(b,k) = b! / (k! * (b-k)!)
                    int binom = 1; // C(bx, 0) = 1
                    double AB_pow = 1.0;
                    for (int k = 0; k <= bx; k++) {
                        val += binom * AB_pow * Ix_vrr[(ax + bx - k) * cs + ct];
                        // Update binomial coefficient and AB power
                        if (k < bx) {
                            AB_pow *= ABx;
                            binom = binom * (bx - k) / (k + 1);
                        }
                    }
                    bra_x[ct] = val;
                }

                double bra_y[9];
                for (int ct = 0; ct <= c_max_val; ct++) {
                    double val = 0.0;
                    int binom = 1;
                    double AB_pow = 1.0;
                    for (int k = 0; k <= by; k++) {
                        val += binom * AB_pow * Iy_vrr[(ay + by - k) * cs + ct];
                        if (k < by) { AB_pow *= ABy; binom = binom * (by - k) / (k + 1); }
                    }
                    bra_y[ct] = val;
                }

                double bra_z[9];
                for (int ct = 0; ct <= c_max_val; ct++) {
                    double val = 0.0;
                    int binom = 1;
                    double AB_pow = 1.0;
                    for (int k = 0; k <= bz; k++) {
                        val += binom * AB_pow * Iz_vrr[(az + bz - k) * cs + ct];
                        if (k < bz) { AB_pow *= ABz; binom = binom * (bz - k) / (k + 1); }
                    }
                    bra_z[ct] = val;
                }

                for (int ic = 0; ic < nc; ic++) {
                    const int cx_c = loop_to_ang[lc][ic][0];
                    const int cy_c = loop_to_ang[lc][ic][1];
                    const int cz_c = loop_to_ang[lc][ic][2];

                    for (int id_c = 0; id_c < nd; id_c++) {
                        const int dx_d = loop_to_ang[ld][id_c][0];
                        const int dy_d = loop_to_ang[ld][id_c][1];
                        const int dz_d = loop_to_ang[ld][id_c][2];

                        // Ket TRR: I(cx, dx) from bra_x[cx..cx+dx]
                        double Ix_val = trr_ket_eval(cx_c, dx_d, CDx, &bra_x[cx_c]);
                        double Iy_val = trr_ket_eval(cy_c, dy_d, CDy, &bra_y[cy_c]);
                        double Iz_val = trr_ket_eval(cz_c, dz_d, CDz, &bra_z[cz_c]);

                        double contrib = wn * Ix_val * Iy_val * Iz_val;

                        if (use_buffer) {
                            eri_buf[((ia * nb + ib) * nc + ic) * nd + id_c] += contrib;
                        } else {
                            // Direct write for large component counts
                            double Norm = calcNorm(alpha, ax, ay, az)
                                        * calcNorm(beta, bx, by, bz)
                                        * calcNorm(gamma, cx_c, cy_c, cz_c)
                                        * calcNorm(delta_exp, dx_d, dy_d, dz_d);
                            addToResult_case1(
                                Norm * prefactor * contrib, g_int2e,
                                base_a + ia, base_b + ib, base_c + ic, base_d + id_c,
                                num_basis, sym_bra, sym_ket, sym_braket,
                                g_cgto_normalization_factors);
                        }
                    }
                }
            }
        }
    }

    // --- Write buffered results to global memory ---
    if (use_buffer) {
        for (int ia = 0; ia < na; ia++) {
            const int ax = loop_to_ang[la][ia][0];
            const int ay = loop_to_ang[la][ia][1];
            const int az = loop_to_ang[la][ia][2];
            const double Norm_A = calcNorm(alpha, ax, ay, az);

            for (int ib = 0; ib < nb; ib++) {
                const int bx = loop_to_ang[lb][ib][0];
                const int by = loop_to_ang[lb][ib][1];
                const int bz = loop_to_ang[lb][ib][2];
                const double Norm_B = calcNorm(beta, bx, by, bz);

                for (int ic = 0; ic < nc; ic++) {
                    const int cx_c = loop_to_ang[lc][ic][0];
                    const int cy_c = loop_to_ang[lc][ic][1];
                    const int cz_c = loop_to_ang[lc][ic][2];
                    const double Norm_C = calcNorm(gamma, cx_c, cy_c, cz_c);

                    for (int id_c = 0; id_c < nd; id_c++) {
                        const int dx_d = loop_to_ang[ld][id_c][0];
                        const int dy_d = loop_to_ang[ld][id_c][1];
                        const int dz_d = loop_to_ang[ld][id_c][2];
                        const double Norm_D = calcNorm(delta_exp, dx_d, dy_d, dz_d);

                        double val = eri_buf[((ia * nb + ib) * nc + ic) * nd + id_c];
                        if (val != 0.0) {
                            addToResult_case1(
                                Norm_A * Norm_B * Norm_C * Norm_D * prefactor * val,
                                g_int2e,
                                base_a + ia, base_b + ib, base_c + ic, base_d + id_c,
                                num_basis, sym_bra, sym_ket, sym_braket,
                                g_cgto_normalization_factors);
                        }
                    }
                }
            }
        }
    }
}

} // namespace gansu::gpu
