/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "rys_eri_direct.hpp"
#include "rys_quadrature.hpp"
#include "int2e.hpp"       // comb_max, loop_to_ang, calcNorm, M_PI_2_5
#include "int2fock.cuh"    // utm_id, twoDim2oneDim, swap_indices
#include "utils_cuda.hpp"  // index1to2

namespace gansu::gpu {

// --- VRR ---
inline __device__
void vrr_1d_direct(int a_max, int c_max,
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

// --- Ket TRR (iterative) ---
inline __device__
double trr_ket_eval_direct(int c, int d, double CD, const double* __restrict__ vals) {
    double buf[9];
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d - dd - 1; i++)
            buf[i] = buf[i + 1] + CD * buf[i];
    return buf[0];
}

// --- add2fock (replicated from int2e_direct.cu) ---
inline __device__
void add2fock_rys(double val, double* g_fock,
                  int mu, int nu, int la, int si, int num_basis, const double* g_dens) {
    if (mu > nu) { swap_indices(mu, nu); }
    if (la > si) { swap_indices(la, si); }
    if (mu > la || (mu == la && nu > si)) { swap_indices(mu, la); swap_indices(nu, si); }

    bool is_sym_bra = (mu == nu);
    bool is_sym_ket = (la == si);
    bool is_sym_braket = (mu == la && nu == si);

    if (is_sym_bra && is_sym_ket && is_sym_braket) {
        atomicAdd(g_fock + num_basis * mu + nu, 0.5 * g_dens[num_basis * la + si] * val);
    } else if (is_sym_bra && is_sym_ket) {
        atomicAdd(g_fock + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_fock + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_fock + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
    } else if (is_sym_bra) {
        atomicAdd(g_fock + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_fock + num_basis * la + si, 1.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_fock + num_basis * mu + la, ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_fock + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), (-0.5) * g_dens[num_basis * mu + la] * val);
    } else if (is_sym_ket) {
        atomicAdd(g_fock + num_basis * mu + nu, 1.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_fock + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_fock + num_basis * mu + la, (-0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_fock + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
    } else if (is_sym_braket) {
        atomicAdd(g_fock + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_fock + ((mu <= la) ? num_basis * mu + la : num_basis * la + mu), ((mu == la) ? -0.5 : -0.25) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_fock + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -0.5 : -0.25) * g_dens[num_basis * mu + la] * val);
        atomicAdd(g_fock + ((mu <= si) ? num_basis * mu + si : num_basis * si + mu), ((mu == si) ? -0.5 : -0.25) * g_dens[num_basis * nu + la] * val);
        atomicAdd(g_fock + ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -0.5 : -0.25) * g_dens[num_basis * mu + si] * val);
    } else {
        atomicAdd(g_fock + num_basis * mu + nu, 2.0 * g_dens[num_basis * la + si] * val);
        atomicAdd(g_fock + num_basis * la + si, 2.0 * g_dens[num_basis * mu + nu] * val);
        atomicAdd(g_fock + ((mu <= la) ? num_basis * mu + la : num_basis * la + mu), ((mu == la) ? -1.0 : -0.5) * g_dens[num_basis * nu + si] * val);
        atomicAdd(g_fock + ((nu <= si) ? num_basis * nu + si : num_basis * si + nu), ((nu == si) ? -1.0 : -0.5) * g_dens[num_basis * mu + la] * val);
        atomicAdd(g_fock + ((mu <= si) ? num_basis * mu + si : num_basis * si + mu), ((mu == si) ? -1.0 : -0.5) * g_dens[num_basis * nu + la] * val);
        atomicAdd(g_fock + ((nu <= la) ? num_basis * nu + la : num_basis * la + nu), ((nu == la) ? -1.0 : -0.5) * g_dens[num_basis * mu + si] * val);
    }
}

// ============================================================
//  RysERI_direct kernel
//  Structure mirrors MD_direct_SCF_1T1SP exactly:
//  - Same thread mapping (d_primitive_shell_pair_indices)
//  - Same symmetry multipliers
//  - Same add2fock pattern
//  Only the integral computation uses Rys instead of MD.
// ============================================================
__global__
void RysERI_direct(
    real_t* g_fock,
    const real_t* g_dens,
    const PrimitiveShell* g_shell,
    const int num_fock_replicas,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const real_t* g_upper_bound_factors,
    const int2* d_primitive_shell_pair_indices,
    const int num_basis,
    const real_t* g_boys_grid,
    const size_t head_bra,
    const size_t head_ket)
{
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    // --- Thread → shell quartet (identical to MD_direct_SCF_1T1SP) ---
    int ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count * (shell_s2.count + 1)) / 2;
    else
        ket_size = shell_s2.count * shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const int2 ab = d_primitive_shell_pair_indices[head_bra + abcd.x];
    const int2 cd = d_primitive_shell_pair_indices[head_ket + abcd.y];

    if (g_upper_bound_factors[head_bra + abcd.x] *
        g_upper_bound_factors[head_ket + abcd.y] < schwarz_screening_threshold)
        return;

    const size_t pidx_a = ab.x, pidx_b = ab.y;
    const size_t pidx_c = cd.x, pidx_d = cd.y;

    const PrimitiveShell sa = g_shell[pidx_a];
    const PrimitiveShell sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c];
    const PrimitiveShell sd = g_shell[pidx_d];

    const size_t size_a = sa.basis_index, size_b = sb.basis_index;
    const size_t size_c = sc.basis_index, size_d = sd.basis_index;

    const bool is_bra_symmetric = (pidx_a == pidx_b);
    const bool is_ket_symmetric = (pidx_c == pidx_d);
    const bool is_braket_symmetric = (utm_id(pidx_a, pidx_b) == utm_id(pidx_c, pidx_d));

    // --- Gaussian Product Theorem ---
    const double alpha = sa.exponent, beta = sb.exponent;
    const double gamma = sc.exponent, delta_exp = sd.exponent;
    const double p = alpha + beta, q = gamma + delta_exp;
    const double rho = p * q / (p + q);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;
    const double Dx = sd.coordinate.x, Dy = sd.coordinate.y, Dz = sd.coordinate.z;

    const double Px = (alpha*Ax + beta*Bx)/p, Py = (alpha*Ay + beta*By)/p, Pz = (alpha*Az + beta*Bz)/p;
    const double Qx = (gamma*Cx + delta_exp*Dx)/q, Qy = (gamma*Cy + delta_exp*Dy)/q, Qz = (gamma*Cz + delta_exp*Dz)/q;

    const double PQx = Px-Qx, PQy = Py-Qy, PQz = Pz-Qz;
    const double T = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    const int orbital_A = sa.shell_type, orbital_B = sb.shell_type;
    const int orbital_C = sc.shell_type, orbital_D = sd.shell_type;
    const int L = orbital_A + orbital_B + orbital_C + orbital_D;

    const double AB2 = (Ax-Bx)*(Ax-Bx)+(Ay-By)*(Ay-By)+(Az-Bz)*(Az-Bz);
    const double CD2 = (Cx-Dx)*(Cx-Dx)+(Cy-Dy)*(Cy-Dy)+(Cz-Dz)*(Cz-Dz);
    const double prefactor = 2.0 * M_PI_2_5 / (p*q*sqrt(p+q))
        * exp(-alpha*beta/p*AB2 - gamma*delta_exp/q*CD2)
        * sa.coefficient * sb.coefficient * sc.coefficient * sd.coefficient;

    // --- Rys roots/weights ---
    const int N = L / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // --- Dimensions ---
    const int a_max = orbital_A + orbital_B;
    const int c_max_val = orbital_C + orbital_D;
    const int cs = c_max_val + 1;

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double CDx = Cx-Dx, CDy = Cy-Dy, CDz = Cz-Dz;

    double Ix_vrr[81], Iy_vrr[81], Iz_vrr[81];

    // --- Component loop (same structure as MD_direct_SCF_1T1SP) ---
    for (int lmn_a = 0; lmn_a < comb_max(orbital_A); lmn_a++) {
        int l1 = loop_to_ang[orbital_A][lmn_a][0];
        int m1 = loop_to_ang[orbital_A][lmn_a][1];
        int n1 = loop_to_ang[orbital_A][lmn_a][2];
        double Norm_A = calcNorm(alpha, l1, m1, n1);

        for (int lmn_b = 0; lmn_b < comb_max(orbital_B); lmn_b++) {
            int l2 = loop_to_ang[orbital_B][lmn_b][0];
            int m2 = loop_to_ang[orbital_B][lmn_b][1];
            int n2 = loop_to_ang[orbital_B][lmn_b][2];
            double Norm_B = calcNorm(beta, l2, m2, n2);

            for (int lmn_c = 0; lmn_c < comb_max(orbital_C); lmn_c++) {
                int l3 = loop_to_ang[orbital_C][lmn_c][0];
                int m3 = loop_to_ang[orbital_C][lmn_c][1];
                int n3 = loop_to_ang[orbital_C][lmn_c][2];
                double Norm_C = calcNorm(gamma, l3, m3, n3);

                for (int lmn_d = 0; lmn_d < comb_max(orbital_D); lmn_d++) {
                    // Skip lower triangle when bra/ket shells share same CGTO
                    if (size_c == size_d && lmn_c > lmn_d) continue;
                    if (size_a == size_b && lmn_a > lmn_b) continue;

                    int l4 = loop_to_ang[orbital_D][lmn_d][0];
                    int m4 = loop_to_ang[orbital_D][lmn_d][1];
                    int n4 = loop_to_ang[orbital_D][lmn_d][2];
                    double Norm_D = calcNorm(delta_exp, l4, m4, n4);

                    double thread_val = 0.0;

                    // Sum over Rys roots
                    for (int n = 0; n < N; n++) {
                        const double t2 = rys_roots[n];
                        const double wn = rys_weights[n];
                        const double u = rho * t2;
                        const double u_over_p = u / p, u_over_q = u / q;

                        const double B00 = t2 / (2.0*(p+q));
                        const double B10 = (1.0 - u_over_p) / (2.0*p);
                        const double B01 = (1.0 - u_over_q) / (2.0*q);

                        const double C00x = (Px-Ax) + u_over_p*(Qx-Px);
                        const double C00y = (Py-Ay) + u_over_p*(Qy-Py);
                        const double C00z = (Pz-Az) + u_over_p*(Qz-Pz);
                        const double D00x = (Qx-Cx) + u_over_q*(Px-Qx);
                        const double D00y = (Qy-Cy) + u_over_q*(Py-Qy);
                        const double D00z = (Qz-Cz) + u_over_q*(Pz-Qz);

                        vrr_1d_direct(a_max, c_max_val, C00x, D00x, B10, B01, B00, Ix_vrr);
                        vrr_1d_direct(a_max, c_max_val, C00y, D00y, B10, B01, B00, Iy_vrr);
                        vrr_1d_direct(a_max, c_max_val, C00z, D00z, B10, B01, B00, Iz_vrr);

                        // Bra TRR (binomial expansion) + Ket TRR
                        double bra_x[9], bra_y[9], bra_z[9];
                        for (int ct = 0; ct <= c_max_val; ct++) {
                            double vx=0, vy=0, vz=0;
                            int bx=1, by2=1, bz2=1;
                            double apx=1, apy=1, apz=1;
                            for (int k=0; k<=l2; k++) {
                                vx += bx*apx*Ix_vrr[(l1+l2-k)*cs+ct];
                                if(k<l2){apx*=ABx; bx=bx*(l2-k)/(k+1);}
                            }
                            for (int k=0; k<=m2; k++) {
                                vy += by2*apy*Iy_vrr[(m1+m2-k)*cs+ct];
                                if(k<m2){apy*=ABy; by2=by2*(m2-k)/(k+1);}
                            }
                            for (int k=0; k<=n2; k++) {
                                vz += bz2*apz*Iz_vrr[(n1+n2-k)*cs+ct];
                                if(k<n2){apz*=ABz; bz2=bz2*(n2-k)/(k+1);}
                            }
                            bra_x[ct]=vx; bra_y[ct]=vy; bra_z[ct]=vz;
                        }

                        double Ix_val = trr_ket_eval_direct(l3, l4, CDx, &bra_x[l3]);
                        double Iy_val = trr_ket_eval_direct(m3, m4, CDy, &bra_y[m3]);
                        double Iz_val = trr_ket_eval_direct(n3, n4, CDz, &bra_z[n3]);

                        thread_val += wn * Ix_val * Iy_val * Iz_val;
                    }

                    // Apply prefactor and normalization
                    // (Same pattern as MD_direct_SCF_1T1SP lines 1355-1384)
                    thread_val *= prefactor;
                    double Norm = Norm_A * Norm_B * Norm_C * Norm_D;

                    thread_val *= Norm * g_cgto_normalization_factors[size_a + lmn_a]
                                      * g_cgto_normalization_factors[size_b + lmn_b]
                                      * g_cgto_normalization_factors[size_c + lmn_c]
                                      * g_cgto_normalization_factors[size_d + lmn_d];

                    if (!is_bra_symmetric && size_a == size_b) thread_val *= 2.0;
                    if (!is_ket_symmetric && size_c == size_d) thread_val *= 2.0;

                    if (utm_id(size_a, size_b) == utm_id(size_c, size_d)) {
                        if (!is_braket_symmetric) thread_val *= 2.0;
                        if (twoDim2oneDim(size_a+lmn_a, size_b+lmn_b, num_basis) !=
                            twoDim2oneDim(size_c+lmn_c, size_d+lmn_d, num_basis))
                            thread_val *= 0.5;
                    }

                    add2fock_rys(
                        thread_val,
                        g_fock + num_basis * num_basis * (threadIdx.x % num_fock_replicas),
                        size_a+lmn_a, size_b+lmn_b, size_c+lmn_c, size_d+lmn_d,
                        num_basis, g_dens);
                }
            }
        }
    }
}

} // namespace gansu::gpu
