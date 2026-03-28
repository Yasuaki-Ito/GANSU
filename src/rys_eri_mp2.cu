/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Direct MP2 half-transformation kernel.
// Computes H(mu,nu,la,i) = sum_sigma (mu nu | la sigma) * C(sigma, i)
// by iterating shell quartets via Rys quadrature and contracting the 4th
// AO index with MO coefficients on-the-fly.
//
// Based on RysERI (stored ERI kernel) with addToResult_case1 replaced by
// add2half which contracts the 4th index and writes to a (nao,nao,nao,block_occ) buffer.
//
// TODO: GPU最適化 — s/p特化カーネルを作成し、高角運動量のみRysを使用

#include "rys_eri_mp2.hpp"
#include "rys_quadrature.hpp"
#include "int2e.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

// ============================================================
//  add2half: accumulate half-transformed integral
//
//  For each symmetry-equivalent (p,q,r,s) of the canonical quartet,
//  accumulate: H(p,q,r,ii) += val * C(s, i_start+ii) for ii=0..block_occ-1
//
//  The 8-fold permutation symmetry of (pq|rs):
//    (p,q,r,s), (q,p,r,s), (p,q,s,r), (q,p,s,r),
//    (r,s,p,q), (r,s,q,p), (s,r,p,q), (s,r,q,p)
//  Each permutation contracts a DIFFERENT AO index with C.
// ============================================================
inline __device__
void add2half(double val, double* d_half,
              int p, int q, int r, int s,
              int nao, int block_occ, int i_start,
              const double* d_C,
              bool sym_bra, bool sym_ket, bool sym_braket,
              const double* g_cgto_normalization_factors)
{
    val *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q]
         * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];

    // Helper macro: H[a][b][c][ii] = d_half[((a*nao+b)*nao+c)*block_occ + ii]
    #define HALF_ADD(a, b, c, coeff_idx) \
        for (int ii = 0; ii < block_occ; ii++) \
            atomicAdd(&d_half[((size_t)(a)*nao+(b))*nao*block_occ + (c)*block_occ + ii], \
                      val * d_C[(coeff_idx)*nao + i_start + ii])

    // --- Enumerate all unique symmetry permutations ---
    // 1. (p,q,r,s): contract s
    HALF_ADD(p, q, r, s);

    if (!sym_ket) {
        // 2. (p,q,s,r): contract r
        HALF_ADD(p, q, s, r);
    }

    if (!sym_bra) {
        // 3. (q,p,r,s): contract s
        HALF_ADD(q, p, r, s);

        if (!sym_ket) {
            // 4. (q,p,s,r): contract r
            HALF_ADD(q, p, s, r);
        }
    }

    if (!sym_braket) {
        // 5. (r,s,p,q): contract q
        HALF_ADD(r, s, p, q);

        if (!sym_bra) {
            // 6. (r,s,q,p): contract p
            HALF_ADD(r, s, q, p);
        }

        if (!sym_ket) {
            // 7. (s,r,p,q): contract q
            HALF_ADD(s, r, p, q);

            if (!sym_bra) {
                // 8. (s,r,q,p): contract p
                HALF_ADD(s, r, q, p);
            }
        }
    }

    #undef HALF_ADD
}


// ============================================================
//  VRR (identical to rys_eri.cu)
// ============================================================
inline __device__
void vrr_1d_mp2(int a_max, int c_max,
                double C00, double D00,
                double B10, double B01, double B00,
                double* __restrict__ I) {
    const int cs = c_max + 1;
    I[0] = 1.0;
    if (a_max > 0) {
        I[cs] = C00;
        for (int a = 1; a < a_max; a++)
            I[(a + 1) * cs] = C00 * I[a * cs] + a * B10 * I[(a - 1) * cs];
    }
    for (int c = 0; c < c_max; c++) {
        double cB01 = c * B01;
        I[c + 1] = D00 * I[c] + ((c > 0) ? cB01 * I[c - 1] : 0.0);
        for (int a = 1; a <= a_max; a++)
            I[a * cs + c + 1] = D00 * I[a * cs + c]
                              + ((c > 0) ? cB01 * I[a * cs + c - 1] : 0.0)
                              + a * B00 * I[(a - 1) * cs + c];
    }
}

// ============================================================
//  Ket TRR (identical to rys_eri.cu)
// ============================================================
inline __device__
double trr_ket_eval_mp2(int c, int d, double CD, const double* __restrict__ vals) {
    if (d == 0) return vals[0];
    double buf[9];
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d - dd - 1; i++)
            buf[i] = buf[i + 1] + CD * buf[i];
    return buf[0];
}

// ============================================================
//  RysERI_half_transform kernel
//  Structure identical to RysERI (stored), but writes to
//  half-transformed buffer via add2half instead of addToResult_case1.
// ============================================================
__global__
void RysERI_half_transform(
    real_t* d_half,
    const real_t* d_C,
    const int i_start,
    const int block_occ,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const real_t* g_upper_bound_factors,
    const int num_basis,
    const real_t* g_boys_grid,
    const size_t head_bra,
    const size_t head_ket)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    // --- Thread → shell quartet mapping (identical to RysERI) ---
    int ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count * (shell_s2.count + 1)) / 2;
    else
        ket_size = shell_s2.count * shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // --- Schwarz screening ---
    if (g_upper_bound_factors[head_bra + abcd.x] *
        g_upper_bound_factors[head_ket + abcd.y] < schwarz_screening_threshold)
        return;

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

    const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
    const double CDx = Cx - Dx, CDy = Cy - Dy, CDz = Cz - Dz;

    // --- Accumulation buffer ---
    double eri_buf[1296];
    const bool use_buffer = (n_components <= 1296);
    if (use_buffer) {
        for (int i = 0; i < n_components; i++) eri_buf[i] = 0.0;
    }

    double Ix_vrr[81], Iy_vrr[81], Iz_vrr[81];

    // --- Main loop: Rys roots ---
    for (int n = 0; n < N; n++) {
        const double t2 = rys_roots[n];
        const double wn = rys_weights[n];
        const double u = rho * t2;
        const double u_over_p = u / p, u_over_q = u / q;

        const double B00 = t2 / (2.0 * (p + q));
        const double B10 = (1.0 - u_over_p) / (2.0 * p);
        const double B01 = (1.0 - u_over_q) / (2.0 * q);

        const double C00x = (Px - Ax) + u_over_p * (Qx - Px);
        const double C00y = (Py - Ay) + u_over_p * (Qy - Py);
        const double C00z = (Pz - Az) + u_over_p * (Qz - Pz);
        const double D00x = (Qx - Cx) + u_over_q * (Px - Qx);
        const double D00y = (Qy - Cy) + u_over_q * (Py - Qy);
        const double D00z = (Qz - Cz) + u_over_q * (Pz - Qz);

        vrr_1d_mp2(a_max, c_max_val, C00x, D00x, B10, B01, B00, Ix_vrr);
        vrr_1d_mp2(a_max, c_max_val, C00y, D00y, B10, B01, B00, Iy_vrr);
        vrr_1d_mp2(a_max, c_max_val, C00z, D00z, B10, B01, B00, Iz_vrr);

        for (int ia = 0; ia < na; ia++) {
            const int ax = loop_to_ang[la][ia][0];
            const int ay = loop_to_ang[la][ia][1];
            const int az = loop_to_ang[la][ia][2];

            for (int ib = 0; ib < nb; ib++) {
                const int bx = loop_to_ang[lb][ib][0];
                const int by = loop_to_ang[lb][ib][1];
                const int bz = loop_to_ang[lb][ib][2];

                double bra_x[9];
                for (int ct = 0; ct <= c_max_val; ct++) {
                    double val = 0.0;
                    int binom = 1; double AB_pow = 1.0;
                    for (int k = 0; k <= bx; k++) {
                        val += binom * AB_pow * Ix_vrr[(ax + bx - k) * cs + ct];
                        if (k < bx) { AB_pow *= ABx; binom = binom * (bx - k) / (k + 1); }
                    }
                    bra_x[ct] = val;
                }

                double bra_y[9];
                for (int ct = 0; ct <= c_max_val; ct++) {
                    double val = 0.0;
                    int binom = 1; double AB_pow = 1.0;
                    for (int k = 0; k <= by; k++) {
                        val += binom * AB_pow * Iy_vrr[(ay + by - k) * cs + ct];
                        if (k < by) { AB_pow *= ABy; binom = binom * (by - k) / (k + 1); }
                    }
                    bra_y[ct] = val;
                }

                double bra_z[9];
                for (int ct = 0; ct <= c_max_val; ct++) {
                    double val = 0.0;
                    int binom = 1; double AB_pow = 1.0;
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

                        double Ix_val = trr_ket_eval_mp2(cx_c, dx_d, CDx, &bra_x[cx_c]);
                        double Iy_val = trr_ket_eval_mp2(cy_c, dy_d, CDy, &bra_y[cy_c]);
                        double Iz_val = trr_ket_eval_mp2(cz_c, dz_d, CDz, &bra_z[cz_c]);

                        double contrib = wn * Ix_val * Iy_val * Iz_val;

                        if (use_buffer) {
                            eri_buf[((ia * nb + ib) * nc + ic) * nd + id_c] += contrib;
                        } else {
                            double Norm = calcNorm(alpha, ax, ay, az)
                                        * calcNorm(beta, bx, by, bz)
                                        * calcNorm(gamma, cx_c, cy_c, cz_c)
                                        * calcNorm(delta_exp, dx_d, dy_d, dz_d);
                            add2half(Norm * prefactor * contrib, d_half,
                                     base_a + ia, base_b + ib, base_c + ic, base_d + id_c,
                                     num_basis, block_occ, i_start, d_C,
                                     sym_bra, sym_ket, sym_braket,
                                     g_cgto_normalization_factors);
                        }
                    }
                }
            }
        }
    }

    // --- Write buffered results ---
    if (use_buffer) {
        for (int ia = 0; ia < na; ia++) {
            const double Norm_A = calcNorm(alpha,
                loop_to_ang[la][ia][0], loop_to_ang[la][ia][1], loop_to_ang[la][ia][2]);
            for (int ib = 0; ib < nb; ib++) {
                const double Norm_B = calcNorm(beta,
                    loop_to_ang[lb][ib][0], loop_to_ang[lb][ib][1], loop_to_ang[lb][ib][2]);
                for (int ic = 0; ic < nc; ic++) {
                    const double Norm_C = calcNorm(gamma,
                        loop_to_ang[lc][ic][0], loop_to_ang[lc][ic][1], loop_to_ang[lc][ic][2]);
                    for (int id_c = 0; id_c < nd; id_c++) {
                        double val = eri_buf[((ia * nb + ib) * nc + ic) * nd + id_c];
                        if (val != 0.0) {
                            const double Norm_D = calcNorm(delta_exp,
                                loop_to_ang[ld][id_c][0], loop_to_ang[ld][id_c][1], loop_to_ang[ld][id_c][2]);
                            add2half(Norm_A * Norm_B * Norm_C * Norm_D * prefactor * val,
                                     d_half,
                                     base_a + ia, base_b + ib, base_c + ic, base_d + id_c,
                                     num_basis, block_occ, i_start, d_C,
                                     sym_bra, sym_ket, sym_braket,
                                     g_cgto_normalization_factors);
                        }
                    }
                }
            }
        }
    }
}

} // namespace gansu::gpu
