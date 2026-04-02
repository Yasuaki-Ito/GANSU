/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Rys quadrature ERI gradient kernel.
//
// Uses the angular momentum shift formula:
//   d/dA_x (pq|rs) = 2α · (p+1_x, q | rs) - l₁ · (p-1_x, q | rs)
//
// This avoids the McMurchie-Davidson Et_grad functions entirely.
// VRR is extended by 1 in each direction to provide the shifted integrals.

#include "rys_grad_g.hpp"
#include "rys_quadrature.hpp"
#include "int2e.hpp"       // comb_max, loop_to_ang, calcNorm, M_PI_2_5
#include "utils_cuda.hpp"  // index1to2

namespace gansu::gpu {

// --- VRR (same as rys_eri.cu) ---
inline __device__
void vrr_1d_grad(int a_max, int c_max,
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
double trr_ket_grad(int c, int d, double CD, const double* __restrict__ vals) {
    double buf[9];
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d-dd-1; i++)
            buf[i] = buf[i+1] + CD*buf[i];
    return buf[0];
}

// --- Compute full integral I(ax, bx, cx, dx) from VRR using bra+ket TRR ---
inline __device__
double compute_integral_1d(int ax, int bx, int cx, int dx,
                           double AB, double CD, int cs,
                           const double* __restrict__ vrr) {
    // Bra TRR (binomial expansion)
    double bra[9]; // max c_max+1
    int c_max = cx + dx;
    for (int ct = cx; ct <= c_max; ct++) {
        double val = 0.0;
        int binom = 1; double apow = 1.0;
        for (int k = 0; k <= bx; k++) {
            val += binom * apow * vrr[(ax+bx-k)*cs + ct];
            if (k < bx) { apow *= AB; binom = binom*(bx-k)/(k+1); }
        }
        bra[ct - cx] = val;
    }
    // Ket TRR
    return trr_ket_grad(0, dx, CD, bra);
}

// ============================================================
//  Rys ERI gradient kernel (RHF)
//
//  For each primitive quartet, computes 12 gradient components
//  (4 atoms × 3 directions) using the shift formula:
//    d/dA_x = 2α·I(l1+1,...) - l1·I(l1-1,...)
//    d/dB_x = 2β·I(...,l2+1,...) - l2·I(...,l2-1,...)
//    d/dC_x = 2γ·I(...,l3+1,...) - l3·I(...,l3-1,...)
//    d/dD_x = 2δ·I(...,...,l4+1) - l4·I(...,...,l4-1)
// ============================================================
__global__
void Rys_compute_gradients_two_electron(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    // --- Thread → shell quartet mapping (same as grad_g.cu) ---
    size_t ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    else
        ket_size = shell_s2.count * shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    const size_t pidx_a = ab.x + shell_s0.start_index;
    const size_t pidx_b = ab.y + shell_s1.start_index;
    const size_t pidx_c = cd.x + shell_s2.start_index;
    const size_t pidx_d = cd.y + shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a];
    const PrimitiveShell sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c];
    const PrimitiveShell sd = g_shell[pidx_d];

    const size_t base_a = sa.basis_index, base_b = sb.basis_index;
    const size_t base_c = sc.basis_index, base_d = sd.basis_index;

    const bool sym_bra = (pidx_a == pidx_b);
    const bool sym_ket = (pidx_c == pidx_d);
    const bool sym_braket = (pidx_a == pidx_c && pidx_b == pidx_d);

    const double alpha = sa.exponent, beta = sb.exponent;
    const double gamma_e = sc.exponent, delta_e = sd.exponent;
    const double p = alpha + beta, q = gamma_e + delta_e;
    const double rho = p * q / (p + q);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;
    const double Dx = sd.coordinate.x, Dy = sd.coordinate.y, Dz = sd.coordinate.z;

    const double Px = (alpha*Ax+beta*Bx)/p, Py = (alpha*Ay+beta*By)/p, Pz = (alpha*Az+beta*Bz)/p;
    const double Qx = (gamma_e*Cx+delta_e*Dx)/q, Qy = (gamma_e*Cy+delta_e*Dy)/q, Qz = (gamma_e*Cz+delta_e*Dz)/q;

    const double PQx = Px-Qx, PQy = Py-Qy, PQz = Pz-Qz;
    const double T = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    const int la = sa.shell_type, lb = sb.shell_type;
    const int lc = sc.shell_type, ld = sd.shell_type;
    const int L = la + lb + lc + ld;

    // Prefactor (same as ERI, without K_AB*K_CD which is in VRR implicitly via shift formula)
    const double AB2 = (Ax-Bx)*(Ax-Bx)+(Ay-By)*(Ay-By)+(Az-Bz)*(Az-Bz);
    const double CD2 = (Cx-Dx)*(Cx-Dx)+(Cy-Dy)*(Cy-Dy)+(Cz-Dz)*(Cz-Dz);
    const double CoefBase = 2.0 * M_PI_2_5 / (p*q*sqrt(p+q))
                          * exp(-alpha*beta/p*AB2 - gamma_e*delta_e/q*CD2)
                          * sa.coefficient * sb.coefficient * sc.coefficient * sd.coefficient;

    // Symmetry factor
    int sym_f = 1 + (!sym_bra ? 1 : 0) + (!sym_ket ? 1 : 0)
              + (!sym_bra && !sym_ket ? 1 : 0)
              + (!sym_braket ? 1 : 0) * (1 + (!sym_bra ? 1 : 0) + (!sym_ket ? 1 : 0)
              + (!sym_bra && !sym_ket ? 1 : 0));

    // Rys roots — need N' = (L+1)/2 + 1 for the gradient-shifted integrals
    const int N = (L + 1) / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // VRR dimensions — extended by 1 for gradient shifts
    const int a_max = la + lb + 1;  // +1 for gradient
    const int c_max = lc + ld + 1;  // +1 for gradient
    const int cs = c_max + 1;

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double CDx = Cx-Dx, CDy = Cy-Dy, CDz = Cz-Dz;

    const int na = comb_max(la), nb = comb_max(lb);
    const int nc = comb_max(lc), nd = comb_max(ld);

    // VRR workspace (extended): max (9+1)*(9+1) = 100 per direction
    double Ix[100], Iy[100], Iz[100];

    // Per-atom gradient accumulator
    double grad_atom[12] = {0.0};

    // --- Component loop ---
    for (int ia = 0; ia < na; ia++) {
        int l1 = loop_to_ang[la][ia][0], m1 = loop_to_ang[la][ia][1], n1 = loop_to_ang[la][ia][2];
        double NA = calcNorm(alpha, l1, m1, n1);

        for (int ib = 0; ib < nb; ib++) {
            int l2 = loop_to_ang[lb][ib][0], m2 = loop_to_ang[lb][ib][1], n2 = loop_to_ang[lb][ib][2];
            double NB = calcNorm(beta, l2, m2, n2);

            for (int ic = 0; ic < nc; ic++) {
                int l3 = loop_to_ang[lc][ic][0], m3 = loop_to_ang[lc][ic][1], n3 = loop_to_ang[lc][ic][2];
                double NC = calcNorm(gamma_e, l3, m3, n3);

                for (int id_c = 0; id_c < nd; id_c++) {
                    int l4 = loop_to_ang[ld][id_c][0], m4 = loop_to_ang[ld][id_c][1], n4 = loop_to_ang[ld][id_c][2];
                    double ND = calcNorm(delta_e, l4, m4, n4);

                    // Density matrix elements
                    double D_ab = g_density_matrix[(base_a+ia)*num_basis + (base_b+ib)];
                    double D_cd = g_density_matrix[(base_c+ic)*num_basis + (base_d+id_c)];
                    double D_ac = g_density_matrix[(base_a+ia)*num_basis + (base_c+ic)];
                    double D_bd = g_density_matrix[(base_b+ib)*num_basis + (base_d+id_c)];
                    double D_ad = g_density_matrix[(base_a+ia)*num_basis + (base_d+id_c)];
                    double D_bc = g_density_matrix[(base_b+ib)*num_basis + (base_c+ic)];

                    double density_w = 0.5*D_ab*D_cd - 0.125*(D_ac*D_bd + D_ad*D_bc);
                    if (fabs(density_w) < 1.0e-18) continue;

                    double w = (double)sym_f * CoefBase
                        * g_cgto_normalization_factors[base_a+ia]
                        * g_cgto_normalization_factors[base_b+ib]
                        * g_cgto_normalization_factors[base_c+ic]
                        * g_cgto_normalization_factors[base_d+id_c]
                        * NA * NB * NC * ND * density_w;

                    double part[12] = {0.0};

                    // Sum over Rys roots
                    for (int n = 0; n < N; n++) {
                        const double t2 = rys_roots[n];
                        const double wn = rys_weights[n];
                        const double u = rho * t2;
                        const double u_over_p = u / p, u_over_q = u / q;

                        const double B00 = t2 / (2.0*(p+q));
                        const double B10 = (1.0-u_over_p) / (2.0*p);
                        const double B01 = (1.0-u_over_q) / (2.0*q);

                        const double C00x = (Px-Ax)+u_over_p*(Qx-Px);
                        const double C00y = (Py-Ay)+u_over_p*(Qy-Py);
                        const double C00z = (Pz-Az)+u_over_p*(Qz-Pz);
                        const double D00x = (Qx-Cx)+u_over_q*(Px-Qx);
                        const double D00y = (Qy-Cy)+u_over_q*(Py-Qy);
                        const double D00z = (Qz-Cz)+u_over_q*(Pz-Qz);

                        // VRR with extended range (a_max+1, c_max+1 for gradient)
                        vrr_1d_grad(a_max, c_max, C00x, D00x, B10, B01, B00, Ix);
                        vrr_1d_grad(a_max, c_max, C00y, D00y, B10, B01, B00, Iy);
                        vrr_1d_grad(a_max, c_max, C00z, D00z, B10, B01, B00, Iz);

                        // Helper: compute I(ax,bx,cx,dx) from VRR for one direction
                        #define I1D(vrr, ax, bx, cx, dx, AB, CD) \
                            compute_integral_1d(ax, bx, cx, dx, AB, CD, cs, vrr)

                        // Base integrals for y,z (reused across x-derivatives)
                        double Iy_base = I1D(Iy, m1, m2, m3, m4, ABy, CDy);
                        double Iz_base = I1D(Iz, n1, n2, n3, n4, ABz, CDz);
                        double Ix_base = I1D(Ix, l1, l2, l3, l4, ABx, CDx);

                        // --- Atom A derivatives: d/dA_x = 2α·I(l1+1,...) - l1·I(l1-1,...) ---
                        double dA_x = 2.0*alpha * I1D(Ix, l1+1, l2, l3, l4, ABx, CDx)
                                    - (l1 > 0 ? l1 * I1D(Ix, l1-1, l2, l3, l4, ABx, CDx) : 0.0);
                        double dA_y = 2.0*alpha * I1D(Iy, m1+1, m2, m3, m4, ABy, CDy)
                                    - (m1 > 0 ? m1 * I1D(Iy, m1-1, m2, m3, m4, ABy, CDy) : 0.0);
                        double dA_z = 2.0*alpha * I1D(Iz, n1+1, n2, n3, n4, ABz, CDz)
                                    - (n1 > 0 ? n1 * I1D(Iz, n1-1, n2, n3, n4, ABz, CDz) : 0.0);

                        part[0] += wn * dA_x * Iy_base * Iz_base;
                        part[1] += wn * Ix_base * dA_y * Iz_base;
                        part[2] += wn * Ix_base * Iy_base * dA_z;

                        // --- Atom B derivatives: d/dB_x = 2β·I(l1, l2+1,...) - l2·I(l1, l2-1,...) ---
                        double dB_x = 2.0*beta * I1D(Ix, l1, l2+1, l3, l4, ABx, CDx)
                                    - (l2 > 0 ? l2 * I1D(Ix, l1, l2-1, l3, l4, ABx, CDx) : 0.0);
                        double dB_y = 2.0*beta * I1D(Iy, m1, m2+1, m3, m4, ABy, CDy)
                                    - (m2 > 0 ? m2 * I1D(Iy, m1, m2-1, m3, m4, ABy, CDy) : 0.0);
                        double dB_z = 2.0*beta * I1D(Iz, n1, n2+1, n3, n4, ABz, CDz)
                                    - (n2 > 0 ? n2 * I1D(Iz, n1, n2-1, n3, n4, ABz, CDz) : 0.0);

                        part[3] += wn * dB_x * Iy_base * Iz_base;
                        part[4] += wn * Ix_base * dB_y * Iz_base;
                        part[5] += wn * Ix_base * Iy_base * dB_z;

                        // --- Atom C derivatives: d/dC_x = 2γ·I(...,l3+1,...) - l3·I(...,l3-1,...) ---
                        double dC_x = 2.0*gamma_e * I1D(Ix, l1, l2, l3+1, l4, ABx, CDx)
                                    - (l3 > 0 ? l3 * I1D(Ix, l1, l2, l3-1, l4, ABx, CDx) : 0.0);
                        double dC_y = 2.0*gamma_e * I1D(Iy, m1, m2, m3+1, m4, ABy, CDy)
                                    - (m3 > 0 ? m3 * I1D(Iy, m1, m2, m3-1, m4, ABy, CDy) : 0.0);
                        double dC_z = 2.0*gamma_e * I1D(Iz, n1, n2, n3+1, n4, ABz, CDz)
                                    - (n3 > 0 ? n3 * I1D(Iz, n1, n2, n3-1, n4, ABz, CDz) : 0.0);

                        part[6] += wn * dC_x * Iy_base * Iz_base;
                        part[7] += wn * Ix_base * dC_y * Iz_base;
                        part[8] += wn * Ix_base * Iy_base * dC_z;

                        // --- Atom D derivatives: d/dD_x = 2δ·I(...,l4+1) - l4·I(...,l4-1) ---
                        double dD_x = 2.0*delta_e * I1D(Ix, l1, l2, l3, l4+1, ABx, CDx)
                                    - (l4 > 0 ? l4 * I1D(Ix, l1, l2, l3, l4-1, ABx, CDx) : 0.0);
                        double dD_y = 2.0*delta_e * I1D(Iy, m1, m2, m3, m4+1, ABy, CDy)
                                    - (m4 > 0 ? m4 * I1D(Iy, m1, m2, m3, m4-1, ABy, CDy) : 0.0);
                        double dD_z = 2.0*delta_e * I1D(Iz, n1, n2, n3, n4+1, ABz, CDz)
                                    - (n4 > 0 ? n4 * I1D(Iz, n1, n2, n3, n4-1, ABz, CDz) : 0.0);

                        part[9]  += wn * dD_x * Iy_base * Iz_base;
                        part[10] += wn * Ix_base * dD_y * Iz_base;
                        part[11] += wn * Ix_base * Iy_base * dD_z;

                        #undef I1D
                    }

                    // Accumulate weighted gradient
                    for (int dir = 0; dir < 3; dir++) {
                        grad_atom[0+dir] += w * part[0+dir];
                        grad_atom[3+dir] += w * part[3+dir];
                        grad_atom[6+dir] += w * part[6+dir];
                        grad_atom[9+dir] += w * part[9+dir];
                    }
                }
            }
        }
    }

    // Write to global gradient
    for (int dir = 0; dir < 3; dir++) {
        atomicAdd(&g_gradients[3*sa.atom_index + dir], grad_atom[0+dir]);
        atomicAdd(&g_gradients[3*sb.atom_index + dir], grad_atom[3+dir]);
        atomicAdd(&g_gradients[3*sc.atom_index + dir], grad_atom[6+dir]);
        atomicAdd(&g_gradients[3*sd.atom_index + dir], grad_atom[9+dir]);
    }
}

// ============================================================
//  2-PDM version: adds 4-index Gamma correction to density_w
//  Used for MP2 gradient non-separable 2-PDM contribution
// ============================================================
__global__
void Rys_compute_gradients_two_electron_2pdm(
    double* g_gradients,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid,
    const double* g_gamma_4idx)   // nao^4 symmetrized 2-PDM correction
{
    // Identical to Rys_compute_gradients_two_electron except density_w includes Gamma
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    size_t ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count * (shell_s2.count + 1)) / 2;
    else
        ket_size = shell_s2.count * shell_s3.count;
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    const size_t pidx_a = ab.x + shell_s0.start_index;
    const size_t pidx_b = ab.y + shell_s1.start_index;
    const size_t pidx_c = cd.x + shell_s2.start_index;
    const size_t pidx_d = cd.y + shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a];
    const PrimitiveShell sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c];
    const PrimitiveShell sd = g_shell[pidx_d];

    const size_t base_a = sa.basis_index, base_b = sb.basis_index;
    const size_t base_c = sc.basis_index, base_d = sd.basis_index;

    const bool sym_bra = (pidx_a == pidx_b);
    const bool sym_ket = (pidx_c == pidx_d);
    const bool sym_braket = (pidx_a == pidx_c && pidx_b == pidx_d);

    const double alpha = sa.exponent, beta = sb.exponent;
    const double gamma_e = sc.exponent, delta_e = sd.exponent;
    const double p = alpha + beta, q = gamma_e + delta_e;
    const double rho = p * q / (p + q);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;
    const double Dx = sd.coordinate.x, Dy = sd.coordinate.y, Dz = sd.coordinate.z;

    const double Px = (alpha*Ax+beta*Bx)/p, Py = (alpha*Ay+beta*By)/p, Pz = (alpha*Az+beta*Bz)/p;
    const double Qx = (gamma_e*Cx+delta_e*Dx)/q, Qy = (gamma_e*Cy+delta_e*Dy)/q, Qz = (gamma_e*Cz+delta_e*Dz)/q;

    const double PQx = Px-Qx, PQy = Py-Qy, PQz = Pz-Qz;
    const double T = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    const int la = sa.shell_type, lb = sb.shell_type;
    const int lc = sc.shell_type, ld = sd.shell_type;
    const int L = la + lb + lc + ld;

    const double AB2 = (Ax-Bx)*(Ax-Bx)+(Ay-By)*(Ay-By)+(Az-Bz)*(Az-Bz);
    const double CD2 = (Cx-Dx)*(Cx-Dx)+(Cy-Dy)*(Cy-Dy)+(Cz-Dz)*(Cz-Dz);
    const double CoefBase = 2.0 * M_PI_2_5 / (p*q*sqrt(p+q))
                          * exp(-alpha*beta/p*AB2 - gamma_e*delta_e/q*CD2)
                          * sa.coefficient * sb.coefficient * sc.coefficient * sd.coefficient;

    int sym_f = 1 + (!sym_bra ? 1 : 0) + (!sym_ket ? 1 : 0)
              + (!sym_bra && !sym_ket ? 1 : 0)
              + (!sym_braket ? 1 : 0) * (1 + (!sym_bra ? 1 : 0) + (!sym_ket ? 1 : 0)
              + (!sym_bra && !sym_ket ? 1 : 0));

    const int N = (L + 1) / 2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    const int a_max = la + lb + 1;
    const int c_max = lc + ld + 1;
    const int cs = c_max + 1;

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double CDx = Cx-Dx, CDy = Cy-Dy, CDz = Cz-Dz;

    const int na = comb_max(la), nb = comb_max(lb);
    const int nc = comb_max(lc), nd = comb_max(ld);

    double Ix[100], Iy[100], Iz[100];

    double grad_atom[12] = {0.0};
    const size_t N4 = (size_t)num_basis;

    for (int ia = 0; ia < na; ia++) {
        int l1 = loop_to_ang[la][ia][0], m1 = loop_to_ang[la][ia][1], n1 = loop_to_ang[la][ia][2];
        double NA = calcNorm(alpha, l1, m1, n1);
        for (int ib = 0; ib < nb; ib++) {
            int l2 = loop_to_ang[lb][ib][0], m2 = loop_to_ang[lb][ib][1], n2 = loop_to_ang[lb][ib][2];
            double NB = calcNorm(beta, l2, m2, n2);
            for (int ic = 0; ic < nc; ic++) {
                int l3 = loop_to_ang[lc][ic][0], m3 = loop_to_ang[lc][ic][1], n3 = loop_to_ang[lc][ic][2];
                double NC = calcNorm(gamma_e, l3, m3, n3);
                for (int id_c = 0; id_c < nd; id_c++) {
                    int l4 = loop_to_ang[ld][id_c][0], m4 = loop_to_ang[ld][id_c][1], n4 = loop_to_ang[ld][id_c][2];
                    double ND = calcNorm(delta_e, l4, m4, n4);

                    const size_t a_idx = base_a + ia, b_idx = base_b + ib;
                    const size_t c_idx = base_c + ic, d_idx = base_d + id_c;

                    double D_ab = g_density_matrix[a_idx*num_basis + b_idx];
                    double D_cd = g_density_matrix[c_idx*num_basis + d_idx];
                    double D_ac = g_density_matrix[a_idx*num_basis + c_idx];
                    double D_bd = g_density_matrix[b_idx*num_basis + d_idx];
                    double D_ad = g_density_matrix[a_idx*num_basis + d_idx];
                    double D_bc = g_density_matrix[b_idx*num_basis + c_idx];

                    // Standard HF 2-PDM + non-separable MP2 correction
                    double density_w = 0.5*D_ab*D_cd - 0.125*(D_ac*D_bd + D_ad*D_bc)
                                     + g_gamma_4idx[((a_idx*N4 + b_idx)*N4 + c_idx)*N4 + d_idx];

                    if (fabs(density_w) < 1.0e-18) continue;

                    double w = (double)sym_f * CoefBase
                        * g_cgto_normalization_factors[a_idx]
                        * g_cgto_normalization_factors[b_idx]
                        * g_cgto_normalization_factors[c_idx]
                        * g_cgto_normalization_factors[d_idx]
                        * NA * NB * NC * ND * density_w;

                    double part[12] = {0.0};

                    for (int n = 0; n < N; n++) {
                        const double t2 = rys_roots[n];
                        const double wn = rys_weights[n];
                        const double u = rho * t2;
                        const double u_over_p = u / p, u_over_q = u / q;

                        const double B00 = t2 / (2.0*(p+q));
                        const double B10 = (1.0-u_over_p) / (2.0*p);
                        const double B01 = (1.0-u_over_q) / (2.0*q);

                        const double C00x = (Px-Ax)+u_over_p*(Qx-Px);
                        const double C00y = (Py-Ay)+u_over_p*(Qy-Py);
                        const double C00z = (Pz-Az)+u_over_p*(Qz-Pz);
                        const double D00x = (Qx-Cx)+u_over_q*(Px-Qx);
                        const double D00y = (Qy-Cy)+u_over_q*(Py-Qy);
                        const double D00z = (Qz-Cz)+u_over_q*(Pz-Qz);

                        vrr_1d_grad(a_max, c_max, C00x, D00x, B10, B01, B00, Ix);
                        vrr_1d_grad(a_max, c_max, C00y, D00y, B10, B01, B00, Iy);
                        vrr_1d_grad(a_max, c_max, C00z, D00z, B10, B01, B00, Iz);

                        #define I1D(vrr, ax, bx, cx, dx, AB, CD) \
                            compute_integral_1d(ax, bx, cx, dx, AB, CD, cs, vrr)

                        double Ix_base = I1D(Ix, l1, l2, l3, l4, ABx, CDx);
                        double Iy_base = I1D(Iy, m1, m2, m3, m4, ABy, CDy);
                        double Iz_base = I1D(Iz, n1, n2, n3, n4, ABz, CDz);

                        double dA_x = 2.0*alpha * I1D(Ix, l1+1, l2, l3, l4, ABx, CDx)
                                    - (l1 > 0 ? l1 * I1D(Ix, l1-1, l2, l3, l4, ABx, CDx) : 0.0);
                        double dA_y = 2.0*alpha * I1D(Iy, m1+1, m2, m3, m4, ABy, CDy)
                                    - (m1 > 0 ? m1 * I1D(Iy, m1-1, m2, m3, m4, ABy, CDy) : 0.0);
                        double dA_z = 2.0*alpha * I1D(Iz, n1+1, n2, n3, n4, ABz, CDz)
                                    - (n1 > 0 ? n1 * I1D(Iz, n1-1, n2, n3, n4, ABz, CDz) : 0.0);

                        part[0] += wn * dA_x * Iy_base * Iz_base;
                        part[1] += wn * Ix_base * dA_y * Iz_base;
                        part[2] += wn * Ix_base * Iy_base * dA_z;

                        double dB_x = 2.0*beta * I1D(Ix, l1, l2+1, l3, l4, ABx, CDx)
                                    - (l2 > 0 ? l2 * I1D(Ix, l1, l2-1, l3, l4, ABx, CDx) : 0.0);
                        double dB_y = 2.0*beta * I1D(Iy, m1, m2+1, m3, m4, ABy, CDy)
                                    - (m2 > 0 ? m2 * I1D(Iy, m1, m2-1, m3, m4, ABy, CDy) : 0.0);
                        double dB_z = 2.0*beta * I1D(Iz, n1, n2+1, n3, n4, ABz, CDz)
                                    - (n2 > 0 ? n2 * I1D(Iz, n1, n2-1, n3, n4, ABz, CDz) : 0.0);

                        part[3] += wn * dB_x * Iy_base * Iz_base;
                        part[4] += wn * Ix_base * dB_y * Iz_base;
                        part[5] += wn * Ix_base * Iy_base * dB_z;

                        double dC_x = 2.0*gamma_e * I1D(Ix, l1, l2, l3+1, l4, ABx, CDx)
                                    - (l3 > 0 ? l3 * I1D(Ix, l1, l2, l3-1, l4, ABx, CDx) : 0.0);
                        double dC_y = 2.0*gamma_e * I1D(Iy, m1, m2, m3+1, m4, ABy, CDy)
                                    - (m3 > 0 ? m3 * I1D(Iy, m1, m2, m3-1, m4, ABy, CDy) : 0.0);
                        double dC_z = 2.0*gamma_e * I1D(Iz, n1, n2, n3+1, n4, ABz, CDz)
                                    - (n3 > 0 ? n3 * I1D(Iz, n1, n2, n3-1, n4, ABz, CDz) : 0.0);

                        part[6] += wn * dC_x * Iy_base * Iz_base;
                        part[7] += wn * Ix_base * dC_y * Iz_base;
                        part[8] += wn * Ix_base * Iy_base * dC_z;

                        double dD_x = 2.0*delta_e * I1D(Ix, l1, l2, l3, l4+1, ABx, CDx)
                                    - (l4 > 0 ? l4 * I1D(Ix, l1, l2, l3, l4-1, ABx, CDx) : 0.0);
                        double dD_y = 2.0*delta_e * I1D(Iy, m1, m2, m3, m4+1, ABy, CDy)
                                    - (m4 > 0 ? m4 * I1D(Iy, m1, m2, m3, m4-1, ABy, CDy) : 0.0);
                        double dD_z = 2.0*delta_e * I1D(Iz, n1, n2, n3, n4+1, ABz, CDz)
                                    - (n4 > 0 ? n4 * I1D(Iz, n1, n2, n3, n4-1, ABz, CDz) : 0.0);

                        part[9]  += wn * dD_x * Iy_base * Iz_base;
                        part[10] += wn * Ix_base * dD_y * Iz_base;
                        part[11] += wn * Ix_base * Iy_base * dD_z;

                        #undef I1D
                    }

                    for (int dir = 0; dir < 3; dir++) {
                        grad_atom[0+dir] += w * part[0+dir];
                        grad_atom[3+dir] += w * part[3+dir];
                        grad_atom[6+dir] += w * part[6+dir];
                        grad_atom[9+dir] += w * part[9+dir];
                    }
                }
            }
        }
    }

    for (int dir = 0; dir < 3; dir++) {
        atomicAdd(&g_gradients[3*sa.atom_index + dir], grad_atom[0+dir]);
        atomicAdd(&g_gradients[3*sb.atom_index + dir], grad_atom[3+dir]);
        atomicAdd(&g_gradients[3*sc.atom_index + dir], grad_atom[6+dir]);
        atomicAdd(&g_gradients[3*sd.atom_index + dir], grad_atom[9+dir]);
    }
}

// ============================================================
//  UHF version
// ============================================================
__global__
void Rys_compute_gradients_two_electron_uhf(
    double* g_gradients,
    const real_t* g_density_matrix_alpha,
    const real_t* g_density_matrix_beta,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const double* g_boys_grid)
{
    // UHF gradient: same structure as RHF but with separate alpha/beta density
    // Coulomb: D_total = Da + Db
    // Exchange: Da and Db separately
    // density_w = 0.5*Dt_ab*Dt_cd - 0.25*(Da_ac*Da_bd + Da_ad*Da_bc + Db_ac*Db_bd + Db_ad*Db_bc)

    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    size_t ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count*(shell_s2.count+1))/2;
    else
        ket_size = shell_s2.count*shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    const size_t pidx_a = ab.x+shell_s0.start_index, pidx_b = ab.y+shell_s1.start_index;
    const size_t pidx_c = cd.x+shell_s2.start_index, pidx_d = cd.y+shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a], sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c], sd = g_shell[pidx_d];

    const size_t base_a=sa.basis_index, base_b=sb.basis_index, base_c=sc.basis_index, base_d=sd.basis_index;

    const bool sym_bra=(pidx_a==pidx_b), sym_ket=(pidx_c==pidx_d);
    const bool sym_braket=(pidx_a==pidx_c && pidx_b==pidx_d);

    const double alpha=sa.exponent, beta=sb.exponent, gamma_e=sc.exponent, delta_e=sd.exponent;
    const double p=alpha+beta, q=gamma_e+delta_e, rho=p*q/(p+q);

    const double Ax=sa.coordinate.x,Ay=sa.coordinate.y,Az=sa.coordinate.z;
    const double Bx=sb.coordinate.x,By=sb.coordinate.y,Bz=sb.coordinate.z;
    const double Cx=sc.coordinate.x,Cy=sc.coordinate.y,Cz=sc.coordinate.z;
    const double Dx=sd.coordinate.x,Dy=sd.coordinate.y,Dz=sd.coordinate.z;

    const double Px=(alpha*Ax+beta*Bx)/p, Py=(alpha*Ay+beta*By)/p, Pz=(alpha*Az+beta*Bz)/p;
    const double Qx=(gamma_e*Cx+delta_e*Dx)/q, Qy=(gamma_e*Cy+delta_e*Dy)/q, Qz=(gamma_e*Cz+delta_e*Dz)/q;

    const double T = rho*((Px-Qx)*(Px-Qx)+(Py-Qy)*(Py-Qy)+(Pz-Qz)*(Pz-Qz));

    const int la=sa.shell_type,lb=sb.shell_type,lc=sc.shell_type,ld=sd.shell_type;
    const int L=la+lb+lc+ld;

    const double AB2=(Ax-Bx)*(Ax-Bx)+(Ay-By)*(Ay-By)+(Az-Bz)*(Az-Bz);
    const double CD2=(Cx-Dx)*(Cx-Dx)+(Cy-Dy)*(Cy-Dy)+(Cz-Dz)*(Cz-Dz);
    const double CoefBase = 2.0*M_PI_2_5/(p*q*sqrt(p+q))
                          * exp(-alpha*beta/p*AB2 - gamma_e*delta_e/q*CD2)
                          * sa.coefficient*sb.coefficient*sc.coefficient*sd.coefficient;

    int sym_f = 1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0)
              +(!sym_braket?1:0)*(1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0));

    const int N = (L+1)/2+1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    const int a_max=la+lb+1, c_max_v=lc+ld+1, cs=c_max_v+1;
    const double ABx=Ax-Bx,ABy=Ay-By,ABz=Az-Bz;
    const double CDx=Cx-Dx,CDy=Cy-Dy,CDz=Cz-Dz;
    const int na=comb_max(la),nb=comb_max(lb),nc=comb_max(lc),nd=comb_max(ld);

    double Ix[100],Iy[100],Iz[100];
    double grad_atom[12]={0.0};

    for(int ia_c=0;ia_c<na;ia_c++){
        int l1=loop_to_ang[la][ia_c][0],m1=loop_to_ang[la][ia_c][1],n1=loop_to_ang[la][ia_c][2];
        double NA=calcNorm(alpha,l1,m1,n1);
        for(int ib_c=0;ib_c<nb;ib_c++){
            int l2=loop_to_ang[lb][ib_c][0],m2=loop_to_ang[lb][ib_c][1],n2=loop_to_ang[lb][ib_c][2];
            double NB=calcNorm(beta,l2,m2,n2);
            for(int ic_c=0;ic_c<nc;ic_c++){
                int l3=loop_to_ang[lc][ic_c][0],m3=loop_to_ang[lc][ic_c][1],n3=loop_to_ang[lc][ic_c][2];
                double NC=calcNorm(gamma_e,l3,m3,n3);
                for(int id_cc=0;id_cc<nd;id_cc++){
                    int l4=loop_to_ang[ld][id_cc][0],m4=loop_to_ang[ld][id_cc][1],n4=loop_to_ang[ld][id_cc][2];
                    double ND=calcNorm(delta_e,l4,m4,n4);

                    int p_a=base_a+ia_c, p_b=base_b+ib_c, p_c=base_c+ic_c, p_d=base_d+id_cc;

                    // UHF density
                    double Da_ab=g_density_matrix_alpha[p_a*num_basis+p_b];
                    double Db_ab=g_density_matrix_beta[p_a*num_basis+p_b];
                    double Da_cd=g_density_matrix_alpha[p_c*num_basis+p_d];
                    double Db_cd=g_density_matrix_beta[p_c*num_basis+p_d];
                    double Dt_ab=Da_ab+Db_ab, Dt_cd=Da_cd+Db_cd;

                    double Da_ac=g_density_matrix_alpha[p_a*num_basis+p_c];
                    double Da_bd=g_density_matrix_alpha[p_b*num_basis+p_d];
                    double Da_ad=g_density_matrix_alpha[p_a*num_basis+p_d];
                    double Da_bc=g_density_matrix_alpha[p_b*num_basis+p_c];
                    double Db_ac=g_density_matrix_beta[p_a*num_basis+p_c];
                    double Db_bd=g_density_matrix_beta[p_b*num_basis+p_d];
                    double Db_ad=g_density_matrix_beta[p_a*num_basis+p_d];
                    double Db_bc=g_density_matrix_beta[p_b*num_basis+p_c];

                    double density_w = 0.5*Dt_ab*Dt_cd
                        - 0.25*(Da_ac*Da_bd + Da_ad*Da_bc + Db_ac*Db_bd + Db_ad*Db_bc);
                    if(fabs(density_w)<1e-18) continue;

                    double w = (double)sym_f*CoefBase
                        *g_cgto_normalization_factors[p_a]*g_cgto_normalization_factors[p_b]
                        *g_cgto_normalization_factors[p_c]*g_cgto_normalization_factors[p_d]
                        *NA*NB*NC*ND*density_w;

                    double part[12]={0.0};
                    for(int n=0;n<N;n++){
                        double t2=rys_roots[n],wn=rys_weights[n],u=rho*t2;
                        double B00=t2/(2.0*(p+q)), B10=(1.0-u/p)/(2.0*p), B01=(1.0-u/q)/(2.0*q);
                        double u_p=u/p, u_q=u/q;
                        vrr_1d_grad(a_max,c_max_v,(Px-Ax)+u_p*(Qx-Px),(Qx-Cx)+u_q*(Px-Qx),B10,B01,B00,Ix);
                        vrr_1d_grad(a_max,c_max_v,(Py-Ay)+u_p*(Qy-Py),(Qy-Cy)+u_q*(Py-Qy),B10,B01,B00,Iy);
                        vrr_1d_grad(a_max,c_max_v,(Pz-Az)+u_p*(Qz-Pz),(Qz-Cz)+u_q*(Pz-Qz),B10,B01,B00,Iz);

                        #define I1D(V,a,b,c,d,AB,CD) compute_integral_1d(a,b,c,d,AB,CD,cs,V)
                        double Iy_b=I1D(Iy,m1,m2,m3,m4,ABy,CDy);
                        double Iz_b=I1D(Iz,n1,n2,n3,n4,ABz,CDz);
                        double Ix_b=I1D(Ix,l1,l2,l3,l4,ABx,CDx);

                        // Same shift formula as RHF
                        double dAx=2*alpha*I1D(Ix,l1+1,l2,l3,l4,ABx,CDx)-(l1>0?l1*I1D(Ix,l1-1,l2,l3,l4,ABx,CDx):0);
                        double dAy=2*alpha*I1D(Iy,m1+1,m2,m3,m4,ABy,CDy)-(m1>0?m1*I1D(Iy,m1-1,m2,m3,m4,ABy,CDy):0);
                        double dAz=2*alpha*I1D(Iz,n1+1,n2,n3,n4,ABz,CDz)-(n1>0?n1*I1D(Iz,n1-1,n2,n3,n4,ABz,CDz):0);
                        part[0]+=wn*dAx*Iy_b*Iz_b; part[1]+=wn*Ix_b*dAy*Iz_b; part[2]+=wn*Ix_b*Iy_b*dAz;

                        double dBx=2*beta*I1D(Ix,l1,l2+1,l3,l4,ABx,CDx)-(l2>0?l2*I1D(Ix,l1,l2-1,l3,l4,ABx,CDx):0);
                        double dBy=2*beta*I1D(Iy,m1,m2+1,m3,m4,ABy,CDy)-(m2>0?m2*I1D(Iy,m1,m2-1,m3,m4,ABy,CDy):0);
                        double dBz=2*beta*I1D(Iz,n1,n2+1,n3,n4,ABz,CDz)-(n2>0?n2*I1D(Iz,n1,n2-1,n3,n4,ABz,CDz):0);
                        part[3]+=wn*dBx*Iy_b*Iz_b; part[4]+=wn*Ix_b*dBy*Iz_b; part[5]+=wn*Ix_b*Iy_b*dBz;

                        double dCx=2*gamma_e*I1D(Ix,l1,l2,l3+1,l4,ABx,CDx)-(l3>0?l3*I1D(Ix,l1,l2,l3-1,l4,ABx,CDx):0);
                        double dCy=2*gamma_e*I1D(Iy,m1,m2,m3+1,m4,ABy,CDy)-(m3>0?m3*I1D(Iy,m1,m2,m3-1,m4,ABy,CDy):0);
                        double dCz=2*gamma_e*I1D(Iz,n1,n2,n3+1,n4,ABz,CDz)-(n3>0?n3*I1D(Iz,n1,n2,n3-1,n4,ABz,CDz):0);
                        part[6]+=wn*dCx*Iy_b*Iz_b; part[7]+=wn*Ix_b*dCy*Iz_b; part[8]+=wn*Ix_b*Iy_b*dCz;

                        double dDx=2*delta_e*I1D(Ix,l1,l2,l3,l4+1,ABx,CDx)-(l4>0?l4*I1D(Ix,l1,l2,l3,l4-1,ABx,CDx):0);
                        double dDy=2*delta_e*I1D(Iy,m1,m2,m3,m4+1,ABy,CDy)-(m4>0?m4*I1D(Iy,m1,m2,m3,m4-1,ABy,CDy):0);
                        double dDz=2*delta_e*I1D(Iz,n1,n2,n3,n4+1,ABz,CDz)-(n4>0?n4*I1D(Iz,n1,n2,n3,n4-1,ABz,CDz):0);
                        part[9]+=wn*dDx*Iy_b*Iz_b; part[10]+=wn*Ix_b*dDy*Iz_b; part[11]+=wn*Ix_b*Iy_b*dDz;
                        #undef I1D
                    }
                    for(int dir=0;dir<3;dir++){
                        grad_atom[0+dir]+=w*part[0+dir]; grad_atom[3+dir]+=w*part[3+dir];
                        grad_atom[6+dir]+=w*part[6+dir]; grad_atom[9+dir]+=w*part[9+dir];
                    }
                }
            }
        }
    }
    for(int dir=0;dir<3;dir++){
        atomicAdd(&g_gradients[3*sa.atom_index+dir],grad_atom[0+dir]);
        atomicAdd(&g_gradients[3*sb.atom_index+dir],grad_atom[3+dir]);
        atomicAdd(&g_gradients[3*sc.atom_index+dir],grad_atom[6+dir]);
        atomicAdd(&g_gradients[3*sd.atom_index+dir],grad_atom[9+dir]);
    }
}

} // namespace gansu::gpu
