/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Rys quadrature ERI Hessian kernel.
//
// Computes second derivatives of two-electron integrals using
// the angular momentum shift formula applied twice:
//
//   d²/dA_x² = 4α²·I(l₁+2,...) - 2α(2l₁+1)·I(l₁,...) + l₁(l₁-1)·I(l₁-2,...)
//   d²/dA_x dB_y = [2α·I(l₁+1,...) - l₁·I(l₁-1,...)]
//                 × [2β·I(...,m₂+1,...) - m₂·I(...,m₂-1,...)]
//
// VRR extended by +2 in each direction.

#include "rys_hessian_g.hpp"
#include "rys_quadrature.hpp"
#include "int2e.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

// --- VRR ---
inline __host__ __device__
void vrr_1d_hess(int a_max, int c_max,
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

// --- Ket TRR ---
inline __host__ __device__
double trr_ket_hess(int c, int d, double CD, const double* __restrict__ vals) {
    double buf[11]; // max d+1 = 5+2 = 7 for g-shell + 2
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d-dd-1; i++)
            buf[i] = buf[i+1] + CD*buf[i];
    return buf[0];
}

// --- Compute I(ax, bx, cx, dx) from VRR ---
inline __host__ __device__
double compute_1d_hess(int ax, int bx, int cx, int dx,
                       double AB, double CD, int cs,
                       const double* __restrict__ vrr) {
    // Bra TRR (binomial expansion)
    double bra[11]; // max cx+dx = 4+4+2 = 10
    int c_total = cx + dx;
    for (int ct = cx; ct <= c_total; ct++) {
        double val = 0.0;
        int binom = 1; double apow = 1.0;
        for (int k = 0; k <= bx; k++) {
            val += binom * apow * vrr[(ax+bx-k)*cs + ct];
            if (k < bx) { apow *= AB; binom = binom*(bx-k)/(k+1); }
        }
        bra[ct - cx] = val;
    }
    return trr_ket_hess(0, dx, CD, bra);
}

// ============================================================
//  Helper: compute d²I_sigma/d(center1)d(center2) for same direction
//
//  Using the shift formula twice:
//  d/d(center) with exponent exp shifts angular momentum:
//    d/dC = 2*exp * I(l+1) - l * I(l-1)
//
//  d²/dC1 dC2 = apply shift for C2 to the shifted integrals of C1
// ============================================================
inline __host__ __device__
double compute_second_deriv_1d(
    int l1, int l2, int l3, int l4,
    int center1, int center2,
    double alpha, double beta, double gamma_e, double delta_e,
    double AB, double CD, int cs, const double* vrr)
{
    int ll[4] = {l1, l2, l3, l4};
    double exps[4] = {alpha, beta, gamma_e, delta_e};

    double e1 = exps[center1];
    int l_orig1 = ll[center1];
    double e2 = exps[center2];

    double result = 0.0;

    // Term 1: 2*e1 * d/dC2 I(..., l_c1+1, ...)
    ll[center1] = l_orig1 + 1;
    {
        int l2_val = ll[center2];
        ll[center2] = l2_val + 1;
        double I_pp = compute_1d_hess(ll[0], ll[1], ll[2], ll[3], AB, CD, cs, vrr);
        ll[center2] = l2_val - 1;
        double I_pm = (l2_val > 0) ? compute_1d_hess(ll[0], ll[1], ll[2], ll[3], AB, CD, cs, vrr) : 0.0;
        ll[center2] = l2_val;
        result += 2*e1 * (2*e2 * I_pp - l2_val * I_pm);
    }
    ll[center1] = l_orig1;

    // Term 2: -l_orig1 * d/dC2 I(..., l_c1-1, ...)
    if (l_orig1 > 0) {
        ll[center1] = l_orig1 - 1;
        int l2_val = ll[center2];
        ll[center2] = l2_val + 1;
        double I_mp = compute_1d_hess(ll[0], ll[1], ll[2], ll[3], AB, CD, cs, vrr);
        ll[center2] = l2_val - 1;
        double I_mm = (l2_val > 0) ? compute_1d_hess(ll[0], ll[1], ll[2], ll[3], AB, CD, cs, vrr) : 0.0;
        ll[center2] = l2_val;
        ll[center1] = l_orig1;
        result -= l_orig1 * (2*e2 * I_mp - l2_val * I_mm);
    }

    return result;
}

// ============================================================
//  Rys ERI Hessian kernel
//
//  For each primitive quartet, computes the upper triangle of
//  the 12×12 second derivative matrix (4 atoms × 3 directions)
//  using the shift formula applied twice.
//
//  Cross derivatives: d²/dA_x dB_y factors into products of
//  single-shift integrals in different directions:
//    [2α·I(l₁+1,...) - l₁·I(l₁-1,...)] is the x-shift on atom A
//    applied to the x-component Ix only, while Iy and Iz are unshifted.
//  For d²/dA_x dA_y, both shifts apply to the SAME atom A but
//  different Cartesian directions (x affects Ix, y affects Iy).
// ============================================================
__global__
void Rys_compute_hessian_two_electron(
    double* g_hessian,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const int num_basis,
    const int num_atoms,
    const double* g_boys_grid)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    const int ndim = 3 * num_atoms;

    // --- Thread → shell quartet mapping ---
    size_t ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count*(shell_s2.count+1))/2;
    else
        ket_size = shell_s2.count*shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    const size_t pidx_a = ab.x+shell_s0.start_index, pidx_b = ab.y+shell_s1.start_index;
    const size_t pidx_c = cd.x+shell_s2.start_index, pidx_d = cd.y+shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a], sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c], sd = g_shell[pidx_d];

    const size_t base_a=sa.basis_index, base_b=sb.basis_index;
    const size_t base_c=sc.basis_index, base_d=sd.basis_index;

    const bool sym_bra=(pidx_a==pidx_b), sym_ket=(pidx_c==pidx_d);
    const bool sym_braket=(pidx_a==pidx_c && pidx_b==pidx_d);

    const double alpha=sa.exponent, beta=sb.exponent;
    const double gamma_e=sc.exponent, delta_e=sd.exponent;
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

    // Symmetry factor
    int sym_f = 1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0)
              +(!sym_braket?1:0)*(1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0));

    // Rys roots: N = (L+2)/2 + 1 for second derivatives
    const int N = (L+2)/2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // VRR extended by +2
    const int a_max = la+lb+2, c_max_v = lc+ld+2;
    const int cs = c_max_v + 1;

    const double ABx=Ax-Bx, ABy=Ay-By, ABz=Az-Bz;
    const double CDx=Cx-Dx, CDy=Cy-Dy, CDz=Cz-Dz;

    const int na=comb_max(la),nb=comb_max(lb),nc=comb_max(lc),nd=comb_max(ld);

    // VRR workspace: max (8+2+1)*(8+2+1) = 121 per direction
    double Ix[121], Iy[121], Iz[121];

    // Atom indices for the 4 centers
    const int atom_A = sa.atom_index, atom_B = sb.atom_index;
    const int atom_C = sc.atom_index, atom_D = sd.atom_index;

    // Exponents for shift formula: [A, B, C, D]
    const double exps[4] = {alpha, beta, gamma_e, delta_e};
    const int atoms[4] = {atom_A, atom_B, atom_C, atom_D};

    // Per-atom-pair Hessian accumulator
    // We accumulate into a local 12×12 array then write to global
    // 12 = 4 atoms × 3 directions: [Ax,Ay,Az, Bx,By,Bz, Cx,Cy,Cz, Dx,Dy,Dz]
    double hess_local[144] = {0.0}; // 12×12

    // --- Component loop ---
    for (int ia_c=0; ia_c<na; ia_c++) {
        int l1=loop_to_ang[la][ia_c][0], m1=loop_to_ang[la][ia_c][1], n1=loop_to_ang[la][ia_c][2];
        double NA=calcNorm(alpha,l1,m1,n1);
        for (int ib_c=0; ib_c<nb; ib_c++) {
            int l2=loop_to_ang[lb][ib_c][0], m2=loop_to_ang[lb][ib_c][1], n2=loop_to_ang[lb][ib_c][2];
            double NB=calcNorm(beta,l2,m2,n2);
            for (int ic_c=0; ic_c<nc; ic_c++) {
                int l3=loop_to_ang[lc][ic_c][0], m3=loop_to_ang[lc][ic_c][1], n3=loop_to_ang[lc][ic_c][2];
                double NC=calcNorm(gamma_e,l3,m3,n3);
                for (int id_c=0; id_c<nd; id_c++) {
                    int l4=loop_to_ang[ld][id_c][0], m4=loop_to_ang[ld][id_c][1], n4=loop_to_ang[ld][id_c][2];
                    double ND=calcNorm(delta_e,l4,m4,n4);

                    // Density weighting
                    int pa=base_a+ia_c, pb=base_b+ib_c, pc=base_c+ic_c, pd=base_d+id_c;
                    double D_ab=g_density_matrix[pa*num_basis+pb];
                    double D_cd=g_density_matrix[pc*num_basis+pd];
                    double D_ac=g_density_matrix[pa*num_basis+pc];
                    double D_bd=g_density_matrix[pb*num_basis+pd];
                    double D_ad=g_density_matrix[pa*num_basis+pd];
                    double D_bc=g_density_matrix[pb*num_basis+pc];

                    double density_w = 0.5*D_ab*D_cd - 0.125*(D_ac*D_bd + D_ad*D_bc);
                    if (fabs(density_w) < 1e-18) continue;

                    double w = (double)sym_f * CoefBase
                        * g_cgto_normalization_factors[pa]*g_cgto_normalization_factors[pb]
                        * g_cgto_normalization_factors[pc]*g_cgto_normalization_factors[pd]
                        * NA*NB*NC*ND * density_w;

                    // Angular momentum arrays: [A, B, C, D] for each direction
                    const int lx[4] = {l1, l2, l3, l4};
                    const int ly[4] = {m1, m2, m3, m4};
                    const int lz[4] = {n1, n2, n3, n4};

                    // Accumulate over Rys roots
                    double part[144] = {0.0};

                    for (int n=0; n<N; n++) {
                        double t2=rys_roots[n], wn=rys_weights[n];
                        double u=rho*t2, u_p=u/p, u_q=u/q;
                        double B00=t2/(2.0*(p+q));
                        double B10=(1.0-u_p)/(2.0*p), B01=(1.0-u_q)/(2.0*q);

                        vrr_1d_hess(a_max, c_max_v,
                            (Px-Ax)+u_p*(Qx-Px), (Qx-Cx)+u_q*(Px-Qx), B10,B01,B00, Ix);
                        vrr_1d_hess(a_max, c_max_v,
                            (Py-Ay)+u_p*(Qy-Py), (Qy-Cy)+u_q*(Py-Qy), B10,B01,B00, Iy);
                        vrr_1d_hess(a_max, c_max_v,
                            (Pz-Az)+u_p*(Qz-Pz), (Qz-Cz)+u_q*(Pz-Qz), B10,B01,B00, Iz);

                        // Helper macro for shifted integrals
                        #define I1D(V, a, b, c, d) compute_1d_hess(a, b, c, d, \
                            (V==Ix?ABx:(V==Iy?ABy:ABz)), (V==Ix?CDx:(V==Iy?CDy:CDz)), cs, V)

                        // Precompute base and first-derivative integrals for each direction
                        // For each center C_idx (0=A,1=B,2=C,3=D) and direction (x,y,z):
                        // grad1[center][dir] = 2*exp * I(l+1) - l * I(l-1)

                        // Base integrals
                        double Ix_base = I1D(Ix, l1, l2, l3, l4);
                        double Iy_base = I1D(Iy, m1, m2, m3, m4);
                        double Iz_base = I1D(Iz, n1, n2, n3, n4);

                        // First derivatives for each center×direction (12 total)
                        // Format: dI[center_idx][dir_idx]  where dir 0=x uses Ix, 1=y uses Iy, 2=z uses Iz
                        // center 0=A shifts lx[0], center 1=B shifts lx[1], etc.
                        // The shift applies to the SPECIFIC component of the SPECIFIC center

                        // For center A (idx 0): shift l1 in x, m1 in y, n1 in z
                        double dIx_A = 2*alpha * I1D(Ix, l1+1, l2, l3, l4) - (l1>0 ? l1*I1D(Ix, l1-1, l2, l3, l4) : 0);
                        double dIy_A = 2*alpha * I1D(Iy, m1+1, m2, m3, m4) - (m1>0 ? m1*I1D(Iy, m1-1, m2, m3, m4) : 0);
                        double dIz_A = 2*alpha * I1D(Iz, n1+1, n2, n3, n4) - (n1>0 ? n1*I1D(Iz, n1-1, n2, n3, n4) : 0);

                        // For center B (idx 1): shift l2 in x, m2 in y, n2 in z
                        double dIx_B = 2*beta * I1D(Ix, l1, l2+1, l3, l4) - (l2>0 ? l2*I1D(Ix, l1, l2-1, l3, l4) : 0);
                        double dIy_B = 2*beta * I1D(Iy, m1, m2+1, m3, m4) - (m2>0 ? m2*I1D(Iy, m1, m2-1, m3, m4) : 0);
                        double dIz_B = 2*beta * I1D(Iz, n1, n2+1, n3, n4) - (n2>0 ? n2*I1D(Iz, n1, n2-1, n3, n4) : 0);

                        // For center C (idx 2): shift l3, m3, n3
                        double dIx_C = 2*gamma_e * I1D(Ix, l1, l2, l3+1, l4) - (l3>0 ? l3*I1D(Ix, l1, l2, l3-1, l4) : 0);
                        double dIy_C = 2*gamma_e * I1D(Iy, m1, m2, m3+1, m4) - (m3>0 ? m3*I1D(Iy, m1, m2, m3-1, m4) : 0);
                        double dIz_C = 2*gamma_e * I1D(Iz, n1, n2, n3+1, n4) - (n3>0 ? n3*I1D(Iz, n1, n2, n3-1, n4) : 0);

                        // For center D (idx 3): shift l4, m4, n4
                        double dIx_D = 2*delta_e * I1D(Ix, l1, l2, l3, l4+1) - (l4>0 ? l4*I1D(Ix, l1, l2, l3, l4-1) : 0);
                        double dIy_D = 2*delta_e * I1D(Iy, m1, m2, m3, m4+1) - (m4>0 ? m4*I1D(Iy, m1, m2, m3, m4-1) : 0);
                        double dIz_D = 2*delta_e * I1D(Iz, n1, n2, n3, n4+1) - (n4>0 ? n4*I1D(Iz, n1, n2, n3, n4-1) : 0);

                        // Pack first derivatives: grad[center*3+dir]
                        // Each is dI_sigma / d(center_dir) for the sigma that matches dir
                        // The full derivative is: d/d(center_dir) (Ix*Iy*Iz) = dI_dir * (product of other two)
                        double dI_x[4] = {dIx_A, dIx_B, dIx_C, dIx_D};
                        double dI_y[4] = {dIy_A, dIy_B, dIy_C, dIy_D};
                        double dI_z[4] = {dIz_A, dIz_B, dIz_C, dIz_D};
                        double I_base[3] = {Ix_base, Iy_base, Iz_base};

                        // Second derivatives: d²/d(center1_dir1) d(center2_dir2)
                        // = product rule on Ix*Iy*Iz
                        //
                        // Case 1: same direction (e.g., both x)
                        //   d²/d(c1_x) d(c2_x) (Ix*Iy*Iz) = d²Ix/d(c1_x)d(c2_x) * Iy * Iz
                        //
                        // Case 2: different directions (e.g., x and y)
                        //   d²/d(c1_x) d(c2_y) (Ix*Iy*Iz) = dIx/d(c1_x) * dIy/d(c2_y) * Iz

                        for (int c1 = 0; c1 < 4; c1++) {
                            for (int d1 = 0; d1 < 3; d1++) {
                                int idx1 = c1*3 + d1;
                                for (int c2 = c1; c2 < 4; c2++) {
                                    int d2_start = (c2 == c1) ? d1 : 0;
                                    for (int d2 = d2_start; d2 < 3; d2++) {
                                        int idx2 = c2*3 + d2;

                                        double val;
                                        if (d1 == d2) {
                                            // Same Cartesian direction: need d²I_sigma/d(c1)d(c2)
                                            // and multiply by the other two base integrals
                                            double d2I;
                                            if (d1 == 0) { // x
                                                d2I = compute_second_deriv_1d(
                                                    l1,l2,l3,l4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABx,CDx, cs, Ix);
                                                val = wn * d2I * Iy_base * Iz_base;
                                            } else if (d1 == 1) { // y
                                                d2I = compute_second_deriv_1d(
                                                    m1,m2,m3,m4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABy,CDy, cs, Iy);
                                                val = wn * d2I * Ix_base * Iz_base;
                                            } else { // z
                                                d2I = compute_second_deriv_1d(
                                                    n1,n2,n3,n4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABz,CDz, cs, Iz);
                                                val = wn * d2I * Ix_base * Iy_base;
                                            }
                                        } else {
                                            // Different directions: product of first derivatives
                                            // d/d(c1_d1) acts on I_{d1}, d/d(c2_d2) acts on I_{d2}
                                            // Third direction is unperturbed
                                            double dI1, dI2, I_third;
                                            if (d1 == 0) dI1 = dI_x[c1];
                                            else if (d1 == 1) dI1 = dI_y[c1];
                                            else dI1 = dI_z[c1];

                                            if (d2 == 0) dI2 = dI_x[c2];
                                            else if (d2 == 1) dI2 = dI_y[c2];
                                            else dI2 = dI_z[c2];

                                            // Third = the direction that is neither d1 nor d2
                                            int d3 = 3 - d1 - d2; // 0+1+2=3
                                            I_third = I_base[d3];

                                            val = wn * dI1 * dI2 * I_third;
                                        }

                                        part[idx1*12 + idx2] += val;
                                    }
                                }
                            }
                        }

                        #undef I1D
                    }

                    // Accumulate weighted Hessian
                    for (int i = 0; i < 12; i++)
                        for (int j = i; j < 12; j++)
                            hess_local[i*12+j] += w * part[i*12+j];
                }
            }
        }
    }

    // Write to global Hessian matrix [ndim × ndim]
    // Map local 12-index (4 atoms × 3 dirs) to global (ndim)
    //
    // The local 12×12 stores the upper triangle: entries where
    // (c1,d1) <= (c2,d2) lexicographically. For the physical Hessian:
    //   H[gi,gj] = Σ hess_local[c1d1, c2d2]  (all c1d1→gi, c2d2→gj)
    // including BOTH (c1d1,c2d2) and (c2d2,c1d1) from the full matrix.
    // When gi != gj, the "mirror" atomicAdd handles this.
    // When gi == gj but li != lj (different centers on same atom),
    // we need factor 2 to count both the upper and lower triangle.
    for (int c1 = 0; c1 < 4; c1++) {
        for (int d1 = 0; d1 < 3; d1++) {
            int gi = 3*atoms[c1] + d1;
            int li = c1*3 + d1;
            for (int c2 = c1; c2 < 4; c2++) {
                int d2_start = (c2 == c1) ? d1 : 0;
                for (int d2 = d2_start; d2 < 3; d2++) {
                    int gj = 3*atoms[c2] + d2;
                    int lj = c2*3 + d2;
                    double val = hess_local[li*12 + lj];
                    if (val != 0.0) {
                        if (gi != gj) {
                            atomicAdd(&g_hessian[gi*ndim + gj], val);
                            atomicAdd(&g_hessian[gj*ndim + gi], val);
                        } else if (li == lj) {
                            // Diagonal of both local and global: count once
                            atomicAdd(&g_hessian[gi*ndim + gj], val);
                        } else {
                            // Off-diagonal local but diagonal global
                            // (different centers on same atom, same direction)
                            // Need factor 2 for the symmetric pair
                            atomicAdd(&g_hessian[gi*ndim + gj], 2.0 * val);
                        }
                    }
                }
            }
        }
    }
}



// CPU host-callable mirror of Rys_compute_hessian_two_electron.  Auto-generated.
void Rys_compute_hessian_two_electron_cpu(double* g_hessian, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const int num_basis, const int num_atoms, const double* g_boys_grid)
{
    #pragma omp parallel for schedule(dynamic)
    for (long long _id_ll = 0; _id_ll < (long long)num_threads; _id_ll++) {
        const size_t id = (size_t)_id_ll;

    // (id loop injected by CPU launcher)

    const int ndim = 3 * num_atoms;

    // --- Thread → shell quartet mapping ---
    size_t ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count*(shell_s2.count+1))/2;
    else
        ket_size = shell_s2.count*shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    const size_t pidx_a = ab.x+shell_s0.start_index, pidx_b = ab.y+shell_s1.start_index;
    const size_t pidx_c = cd.x+shell_s2.start_index, pidx_d = cd.y+shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a], sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c], sd = g_shell[pidx_d];

    const size_t base_a=sa.basis_index, base_b=sb.basis_index;
    const size_t base_c=sc.basis_index, base_d=sd.basis_index;

    const bool sym_bra=(pidx_a==pidx_b), sym_ket=(pidx_c==pidx_d);
    const bool sym_braket=(pidx_a==pidx_c && pidx_b==pidx_d);

    const double alpha=sa.exponent, beta=sb.exponent;
    const double gamma_e=sc.exponent, delta_e=sd.exponent;
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

    // Symmetry factor
    int sym_f = 1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0)
              +(!sym_braket?1:0)*(1+(!sym_bra?1:0)+(!sym_ket?1:0)+(!sym_bra&&!sym_ket?1:0));

    // Rys roots: N = (L+2)/2 + 1 for second derivatives
    const int N = (L+2)/2 + 1;
    double rys_roots[9], rys_weights[9];
    computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

    // VRR extended by +2
    const int a_max = la+lb+2, c_max_v = lc+ld+2;
    const int cs = c_max_v + 1;

    const double ABx=Ax-Bx, ABy=Ay-By, ABz=Az-Bz;
    const double CDx=Cx-Dx, CDy=Cy-Dy, CDz=Cz-Dz;

    const int na=comb_max(la),nb=comb_max(lb),nc=comb_max(lc),nd=comb_max(ld);

    // VRR workspace: max (8+2+1)*(8+2+1) = 121 per direction
    double Ix[121], Iy[121], Iz[121];

    // Atom indices for the 4 centers
    const int atom_A = sa.atom_index, atom_B = sb.atom_index;
    const int atom_C = sc.atom_index, atom_D = sd.atom_index;

    // Exponents for shift formula: [A, B, C, D]
    const double exps[4] = {alpha, beta, gamma_e, delta_e};
    const int atoms[4] = {atom_A, atom_B, atom_C, atom_D};

    // Per-atom-pair Hessian accumulator
    // We accumulate into a local 12×12 array then write to global
    // 12 = 4 atoms × 3 directions: [Ax,Ay,Az, Bx,By,Bz, Cx,Cy,Cz, Dx,Dy,Dz]
    double hess_local[144] = {0.0}; // 12×12

    // --- Component loop ---
    for (int ia_c=0; ia_c<na; ia_c++) {
        int l1=loop_to_ang_host[la][ia_c][0], m1=loop_to_ang_host[la][ia_c][1], n1=loop_to_ang_host[la][ia_c][2];
        double NA=calcNorm(alpha,l1,m1,n1);
        for (int ib_c=0; ib_c<nb; ib_c++) {
            int l2=loop_to_ang_host[lb][ib_c][0], m2=loop_to_ang_host[lb][ib_c][1], n2=loop_to_ang_host[lb][ib_c][2];
            double NB=calcNorm(beta,l2,m2,n2);
            for (int ic_c=0; ic_c<nc; ic_c++) {
                int l3=loop_to_ang_host[lc][ic_c][0], m3=loop_to_ang_host[lc][ic_c][1], n3=loop_to_ang_host[lc][ic_c][2];
                double NC=calcNorm(gamma_e,l3,m3,n3);
                for (int id_c=0; id_c<nd; id_c++) {
                    int l4=loop_to_ang_host[ld][id_c][0], m4=loop_to_ang_host[ld][id_c][1], n4=loop_to_ang_host[ld][id_c][2];
                    double ND=calcNorm(delta_e,l4,m4,n4);

                    // Density weighting
                    int pa=base_a+ia_c, pb=base_b+ib_c, pc=base_c+ic_c, pd=base_d+id_c;
                    double D_ab=g_density_matrix[pa*num_basis+pb];
                    double D_cd=g_density_matrix[pc*num_basis+pd];
                    double D_ac=g_density_matrix[pa*num_basis+pc];
                    double D_bd=g_density_matrix[pb*num_basis+pd];
                    double D_ad=g_density_matrix[pa*num_basis+pd];
                    double D_bc=g_density_matrix[pb*num_basis+pc];

                    double density_w = 0.5*D_ab*D_cd - 0.125*(D_ac*D_bd + D_ad*D_bc);
                    if (fabs(density_w) < 1e-18) continue;

                    double w = (double)sym_f * CoefBase
                        * g_cgto_normalization_factors[pa]*g_cgto_normalization_factors[pb]
                        * g_cgto_normalization_factors[pc]*g_cgto_normalization_factors[pd]
                        * NA*NB*NC*ND * density_w;

                    // Angular momentum arrays: [A, B, C, D] for each direction
                    const int lx[4] = {l1, l2, l3, l4};
                    const int ly[4] = {m1, m2, m3, m4};
                    const int lz[4] = {n1, n2, n3, n4};

                    // Accumulate over Rys roots
                    double part[144] = {0.0};

                    for (int n=0; n<N; n++) {
                        double t2=rys_roots[n], wn=rys_weights[n];
                        double u=rho*t2, u_p=u/p, u_q=u/q;
                        double B00=t2/(2.0*(p+q));
                        double B10=(1.0-u_p)/(2.0*p), B01=(1.0-u_q)/(2.0*q);

                        vrr_1d_hess(a_max, c_max_v,
                            (Px-Ax)+u_p*(Qx-Px), (Qx-Cx)+u_q*(Px-Qx), B10,B01,B00, Ix);
                        vrr_1d_hess(a_max, c_max_v,
                            (Py-Ay)+u_p*(Qy-Py), (Qy-Cy)+u_q*(Py-Qy), B10,B01,B00, Iy);
                        vrr_1d_hess(a_max, c_max_v,
                            (Pz-Az)+u_p*(Qz-Pz), (Qz-Cz)+u_q*(Pz-Qz), B10,B01,B00, Iz);

                        // Helper macro for shifted integrals
                        #define I1D(V, a, b, c, d) compute_1d_hess(a, b, c, d, \
                            (V==Ix?ABx:(V==Iy?ABy:ABz)), (V==Ix?CDx:(V==Iy?CDy:CDz)), cs, V)

                        // Precompute base and first-derivative integrals for each direction
                        // For each center C_idx (0=A,1=B,2=C,3=D) and direction (x,y,z):
                        // grad1[center][dir] = 2*exp * I(l+1) - l * I(l-1)

                        // Base integrals
                        double Ix_base = I1D(Ix, l1, l2, l3, l4);
                        double Iy_base = I1D(Iy, m1, m2, m3, m4);
                        double Iz_base = I1D(Iz, n1, n2, n3, n4);

                        // First derivatives for each center×direction (12 total)
                        // Format: dI[center_idx][dir_idx]  where dir 0=x uses Ix, 1=y uses Iy, 2=z uses Iz
                        // center 0=A shifts lx[0], center 1=B shifts lx[1], etc.
                        // The shift applies to the SPECIFIC component of the SPECIFIC center

                        // For center A (idx 0): shift l1 in x, m1 in y, n1 in z
                        double dIx_A = 2*alpha * I1D(Ix, l1+1, l2, l3, l4) - (l1>0 ? l1*I1D(Ix, l1-1, l2, l3, l4) : 0);
                        double dIy_A = 2*alpha * I1D(Iy, m1+1, m2, m3, m4) - (m1>0 ? m1*I1D(Iy, m1-1, m2, m3, m4) : 0);
                        double dIz_A = 2*alpha * I1D(Iz, n1+1, n2, n3, n4) - (n1>0 ? n1*I1D(Iz, n1-1, n2, n3, n4) : 0);

                        // For center B (idx 1): shift l2 in x, m2 in y, n2 in z
                        double dIx_B = 2*beta * I1D(Ix, l1, l2+1, l3, l4) - (l2>0 ? l2*I1D(Ix, l1, l2-1, l3, l4) : 0);
                        double dIy_B = 2*beta * I1D(Iy, m1, m2+1, m3, m4) - (m2>0 ? m2*I1D(Iy, m1, m2-1, m3, m4) : 0);
                        double dIz_B = 2*beta * I1D(Iz, n1, n2+1, n3, n4) - (n2>0 ? n2*I1D(Iz, n1, n2-1, n3, n4) : 0);

                        // For center C (idx 2): shift l3, m3, n3
                        double dIx_C = 2*gamma_e * I1D(Ix, l1, l2, l3+1, l4) - (l3>0 ? l3*I1D(Ix, l1, l2, l3-1, l4) : 0);
                        double dIy_C = 2*gamma_e * I1D(Iy, m1, m2, m3+1, m4) - (m3>0 ? m3*I1D(Iy, m1, m2, m3-1, m4) : 0);
                        double dIz_C = 2*gamma_e * I1D(Iz, n1, n2, n3+1, n4) - (n3>0 ? n3*I1D(Iz, n1, n2, n3-1, n4) : 0);

                        // For center D (idx 3): shift l4, m4, n4
                        double dIx_D = 2*delta_e * I1D(Ix, l1, l2, l3, l4+1) - (l4>0 ? l4*I1D(Ix, l1, l2, l3, l4-1) : 0);
                        double dIy_D = 2*delta_e * I1D(Iy, m1, m2, m3, m4+1) - (m4>0 ? m4*I1D(Iy, m1, m2, m3, m4-1) : 0);
                        double dIz_D = 2*delta_e * I1D(Iz, n1, n2, n3, n4+1) - (n4>0 ? n4*I1D(Iz, n1, n2, n3, n4-1) : 0);

                        // Pack first derivatives: grad[center*3+dir]
                        // Each is dI_sigma / d(center_dir) for the sigma that matches dir
                        // The full derivative is: d/d(center_dir) (Ix*Iy*Iz) = dI_dir * (product of other two)
                        double dI_x[4] = {dIx_A, dIx_B, dIx_C, dIx_D};
                        double dI_y[4] = {dIy_A, dIy_B, dIy_C, dIy_D};
                        double dI_z[4] = {dIz_A, dIz_B, dIz_C, dIz_D};
                        double I_base[3] = {Ix_base, Iy_base, Iz_base};

                        // Second derivatives: d²/d(center1_dir1) d(center2_dir2)
                        // = product rule on Ix*Iy*Iz
                        //
                        // Case 1: same direction (e.g., both x)
                        //   d²/d(c1_x) d(c2_x) (Ix*Iy*Iz) = d²Ix/d(c1_x)d(c2_x) * Iy * Iz
                        //
                        // Case 2: different directions (e.g., x and y)
                        //   d²/d(c1_x) d(c2_y) (Ix*Iy*Iz) = dIx/d(c1_x) * dIy/d(c2_y) * Iz

                        for (int c1 = 0; c1 < 4; c1++) {
                            for (int d1 = 0; d1 < 3; d1++) {
                                int idx1 = c1*3 + d1;
                                for (int c2 = c1; c2 < 4; c2++) {
                                    int d2_start = (c2 == c1) ? d1 : 0;
                                    for (int d2 = d2_start; d2 < 3; d2++) {
                                        int idx2 = c2*3 + d2;

                                        double val;
                                        if (d1 == d2) {
                                            // Same Cartesian direction: need d²I_sigma/d(c1)d(c2)
                                            // and multiply by the other two base integrals
                                            double d2I;
                                            if (d1 == 0) { // x
                                                d2I = compute_second_deriv_1d(
                                                    l1,l2,l3,l4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABx,CDx, cs, Ix);
                                                val = wn * d2I * Iy_base * Iz_base;
                                            } else if (d1 == 1) { // y
                                                d2I = compute_second_deriv_1d(
                                                    m1,m2,m3,m4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABy,CDy, cs, Iy);
                                                val = wn * d2I * Ix_base * Iz_base;
                                            } else { // z
                                                d2I = compute_second_deriv_1d(
                                                    n1,n2,n3,n4, c1,c2, alpha,beta,gamma_e,delta_e,
                                                    ABz,CDz, cs, Iz);
                                                val = wn * d2I * Ix_base * Iy_base;
                                            }
                                        } else {
                                            // Different directions: product of first derivatives
                                            // d/d(c1_d1) acts on I_{d1}, d/d(c2_d2) acts on I_{d2}
                                            // Third direction is unperturbed
                                            double dI1, dI2, I_third;
                                            if (d1 == 0) dI1 = dI_x[c1];
                                            else if (d1 == 1) dI1 = dI_y[c1];
                                            else dI1 = dI_z[c1];

                                            if (d2 == 0) dI2 = dI_x[c2];
                                            else if (d2 == 1) dI2 = dI_y[c2];
                                            else dI2 = dI_z[c2];

                                            // Third = the direction that is neither d1 nor d2
                                            int d3 = 3 - d1 - d2; // 0+1+2=3
                                            I_third = I_base[d3];

                                            val = wn * dI1 * dI2 * I_third;
                                        }

                                        part[idx1*12 + idx2] += val;
                                    }
                                }
                            }
                        }

                        #undef I1D
                    }

                    // Accumulate weighted Hessian
                    for (int i = 0; i < 12; i++)
                        for (int j = i; j < 12; j++)
                            hess_local[i*12+j] += w * part[i*12+j];
                }
            }
        }
    }

    // Write to global Hessian matrix [ndim × ndim]
    // Map local 12-index (4 atoms × 3 dirs) to global (ndim)
    //
    // The local 12×12 stores the upper triangle: entries where
    // (c1,d1) <= (c2,d2) lexicographically. For the physical Hessian:
    //   H[gi,gj] = Σ hess_local[c1d1, c2d2]  (all c1d1→gi, c2d2→gj)
    // including BOTH (c1d1,c2d2) and (c2d2,c1d1) from the full matrix.
    // When gi != gj, the "mirror" atomicAdd handles this.
    // When gi == gj but li != lj (different centers on same atom),
    // we need factor 2 to count both the upper and lower triangle.
    for (int c1 = 0; c1 < 4; c1++) {
        for (int d1 = 0; d1 < 3; d1++) {
            int gi = 3*atoms[c1] + d1;
            int li = c1*3 + d1;
            for (int c2 = c1; c2 < 4; c2++) {
                int d2_start = (c2 == c1) ? d1 : 0;
                for (int d2 = d2_start; d2 < 3; d2++) {
                    int gj = 3*atoms[c2] + d2;
                    int lj = c2*3 + d2;
                    double val = hess_local[li*12 + lj];
                    if (val != 0.0) {
                        if (gi != gj) {
                            gansu_atomic_add(&g_hessian[gi*ndim + gj], val);
                            gansu_atomic_add(&g_hessian[gj*ndim + gi], val);
                        } else if (li == lj) {
                            // Diagonal of both local and global: count once
                            gansu_atomic_add(&g_hessian[gi*ndim + gj], val);
                        } else {
                            // Off-diagonal local but diagonal global
                            // (different centers on same atom, same direction)
                            // Need factor 2 for the symmetric pair
                            gansu_atomic_add(&g_hessian[gi*ndim + gj], 2.0 * val);
                        }
                    }
                }
            }
        }
    }
    }
}

} // namespace gansu::gpu
