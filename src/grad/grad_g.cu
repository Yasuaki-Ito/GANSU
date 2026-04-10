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


#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif
#include <cmath>

#include "gradients.hpp"


namespace gansu::gpu{


// ---- compute_gradients_two_electron ----
// Rewritten to fix:
//   1. Exchange formula bug (was using + instead of *)
//   2. Per-component density matrix indexing (was shell-level)
//   3. Per-component cgto_norm indexing (was shell-level)
//   4. R computed once per primitive quartet (was per lmn combination)
//   5. Single-pass 6-loop for all 12 gradient components (was 12 separate functions)
__global__
void compute_gradients_two_electron(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors,
                                    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                                    const size_t num_threads, const int num_basis, const double* g_boys_grid)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_threads) return;

    // Compute 4D index from thread id
    size_t ket_size;
    if(shell_s2.start_index == shell_s3.start_index){
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    }else{
        ket_size = shell_s2.count*shell_s3.count;
    }
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // Obtain primitive shells [ab|cd]
    const size_t primitive_index_a = ab.x+shell_s0.start_index;
    const size_t primitive_index_b = ab.y+shell_s1.start_index;
    const size_t primitive_index_c = cd.x+shell_s2.start_index;
    const size_t primitive_index_d = cd.y+shell_s3.start_index;

    const PrimitiveShell a = g_shell[primitive_index_a];
    const PrimitiveShell b = g_shell[primitive_index_b];
    const PrimitiveShell c = g_shell[primitive_index_c];
    const PrimitiveShell d = g_shell[primitive_index_d];

    // Obtain basis index (starting index for angular momentum components)
    const size_t ia = a.basis_index;
    const size_t ib = b.basis_index;
    const size_t ic = c.basis_index;
    const size_t id_idx = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = (primitive_index_a == primitive_index_c && primitive_index_b == primitive_index_d);

    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double delta = d.exponent;
    const double p = alpha + beta;
    const double q = gamma + delta;

    const double3 P = make_double3((alpha*a.coordinate.x + beta*b.coordinate.x)/p, (alpha*a.coordinate.y + beta*b.coordinate.y)/p, (alpha*a.coordinate.z + beta*b.coordinate.z)/p);
    const double3 Q = make_double3((gamma*c.coordinate.x + delta*d.coordinate.x)/q, (gamma*c.coordinate.y + delta*d.coordinate.y)/q, (gamma*c.coordinate.z + delta*d.coordinate.z)/q);

    const double AB_Dx = a.coordinate.x - b.coordinate.x;
    const double AB_Dy = a.coordinate.y - b.coordinate.y;
    const double AB_Dz = a.coordinate.z - b.coordinate.z;

    const double CD_Dx = c.coordinate.x - d.coordinate.x;
    const double CD_Dy = c.coordinate.y - d.coordinate.y;
    const double CD_Dz = c.coordinate.z - d.coordinate.z;

    const int K = a.shell_type + b.shell_type + c.shell_type + d.shell_type + 1;

    double CoefBase = a.coefficient*b.coefficient*c.coefficient*d.coefficient * 2*M_PI_2_5 /(p*q*sqrt((p+q)));

    // Symmetry factor (8-fold ERI symmetry)
    int sym_f = 1 + static_cast<int>(!is_bra_symmetric) + static_cast<int>(!is_ket_symmetric)
              + static_cast<int>(!is_bra_symmetric && !is_ket_symmetric)
              + static_cast<int>(!is_braket_symmetric) * (1 + static_cast<int>(!is_bra_symmetric)
              + static_cast<int>(!is_ket_symmetric) + static_cast<int>(!is_bra_symmetric && !is_ket_symmetric));

    // R and Boys computation — ONCE per primitive quartet (moved out of lmn loop)
    double R_mid[3*size_Rmid];
    double R[size_R];
    double Boys[boys_size];

    getIncrementalBoys(K, (p*q/(p+q)) * calc_dist_GPU(P, Q), g_boys_grid, Boys);
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2 * (p*q/(p+q))), i));
    }

    // Compute R integrals with maximum extents
    compute_R_TripleBuffer(R, R_mid, Boys, P, Q, K, K, K, K);

    // Accumulate gradient contributions over angular momentum components
    // with per-component density matrix indexing
    double grad_atom[12] = {0.0};  // [Ax,Ay,Az, Bx,By,Bz, Cx,Cy,Cz, Dx,Dy,Dz]

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int l1 = loop_to_ang[a.shell_type][lmn_a][0];
        int m1 = loop_to_ang[a.shell_type][lmn_a][1];
        int n1 = loop_to_ang[a.shell_type][lmn_a][2];
        double NA = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){
            int l2 = loop_to_ang[b.shell_type][lmn_b][0];
            int m2 = loop_to_ang[b.shell_type][lmn_b][1];
            int n2 = loop_to_ang[b.shell_type][lmn_b][2];
            double NB = calcNorm(beta, l2, m2, n2);

            for(int lmn_c=0; lmn_c<comb_max(c.shell_type); lmn_c++){
                int l3 = loop_to_ang[c.shell_type][lmn_c][0];
                int m3 = loop_to_ang[c.shell_type][lmn_c][1];
                int n3 = loop_to_ang[c.shell_type][lmn_c][2];
                double NC = calcNorm(gamma, l3, m3, n3);

                for(int lmn_d=0; lmn_d<comb_max(d.shell_type); lmn_d++){
                    int l4 = loop_to_ang[d.shell_type][lmn_d][0];
                    int m4 = loop_to_ang[d.shell_type][lmn_d][1];
                    int n4 = loop_to_ang[d.shell_type][lmn_d][2];
                    double ND = calcNorm(delta, l4, m4, n4);

                    // Per-component density matrix elements
                    double D_ab = g_density_matrix[(ia + lmn_a) * num_basis + (ib + lmn_b)];
                    double D_cd = g_density_matrix[(ic + lmn_c) * num_basis + (id_idx + lmn_d)];
                    double D_ac = g_density_matrix[(ia + lmn_a) * num_basis + (ic + lmn_c)];
                    double D_bd = g_density_matrix[(ib + lmn_b) * num_basis + (id_idx + lmn_d)];
                    double D_ad = g_density_matrix[(ia + lmn_a) * num_basis + (id_idx + lmn_d)];
                    double D_bc = g_density_matrix[(ib + lmn_b) * num_basis + (ic + lmn_c)];

                    // Coulomb: 0.5 * D_ab * D_cd
                    // Exchange: -(1/8) * (D_ac*D_bd + D_ad*D_bc)
                    double density_w = 0.5 * D_ab * D_cd - 0.125 * (D_ac * D_bd + D_ad * D_bc);
                    if (fabs(density_w) < 1.0e-18) continue;

                    // Weight: symmetry * coefficients * cgto_norm (per-component) * primitive_norm * density
                    double w = static_cast<double>(sym_f) * CoefBase
                        * g_cgto_normalization_factors[ia + lmn_a] * g_cgto_normalization_factors[ib + lmn_b]
                        * g_cgto_normalization_factors[ic + lmn_c] * g_cgto_normalization_factors[id_idx + lmn_d]
                        * NA * NB * NC * ND * density_w;

                    // Single-pass 6-loop computing all 12 gradient components
                    int lim_bra_x = l1 + l2 + 1;
                    int lim_bra_y = m1 + m2 + 1;
                    int lim_bra_z = n1 + n2 + 1;
                    int lim_ket_x = l3 + l4 + 1;
                    int lim_ket_y = m3 + m4 + 1;
                    int lim_ket_z = n3 + n4 + 1;

                    double part[12] = {0.0};

                    for (int t = 0; t < lim_bra_x + 1; t++) {
                        double Et_N = MD_Et_NonRecursion(l1, l2, t, alpha, beta, AB_Dx);
                        double Et_G = Et_grad_NonRecursion(l1, l2, t, alpha, beta, AB_Dx);
                        double Et_tm1 = MD_Et_NonRecursion(l1, l2, t - 1, alpha, beta, AB_Dx);
                        double Et_dA_x = (alpha / p) * Et_tm1 + Et_G;
                        double Et_dB_x = (beta / p) * Et_tm1 - Et_G;

                        for (int u = 0; u < lim_bra_y + 1; u++) {
                            double Eu_N = MD_Et_NonRecursion(m1, m2, u, alpha, beta, AB_Dy);
                            double Eu_G = Et_grad_NonRecursion(m1, m2, u, alpha, beta, AB_Dy);
                            double Eu_um1 = MD_Et_NonRecursion(m1, m2, u - 1, alpha, beta, AB_Dy);
                            double Eu_dA_y = (alpha / p) * Eu_um1 + Eu_G;
                            double Eu_dB_y = (beta / p) * Eu_um1 - Eu_G;

                            for (int v = 0; v < lim_bra_z + 1; v++) {
                                double Ev_N = MD_Et_NonRecursion(n1, n2, v, alpha, beta, AB_Dz);
                                double Ev_G = Et_grad_NonRecursion(n1, n2, v, alpha, beta, AB_Dz);
                                double Ev_vm1 = MD_Et_NonRecursion(n1, n2, v - 1, alpha, beta, AB_Dz);
                                double Ev_dA_z = (alpha / p) * Ev_vm1 + Ev_G;
                                double Ev_dB_z = (beta / p) * Ev_vm1 - Ev_G;

                                for (int tau = 0; tau < lim_ket_x + 1; tau++) {
                                    double Etau_N = MD_Et_NonRecursion(l3, l4, tau, gamma, delta, CD_Dx);
                                    double Etau_G = Et_grad_NonRecursion(l3, l4, tau, gamma, delta, CD_Dx);
                                    double Etau_tm1 = MD_Et_NonRecursion(l3, l4, tau - 1, gamma, delta, CD_Dx);
                                    double Etau_dC_x = (gamma / q) * Etau_tm1 + Etau_G;
                                    double Etau_dD_x = (delta / q) * Etau_tm1 - Etau_G;

                                    for (int nu = 0; nu < lim_ket_y + 1; nu++) {
                                        double Enu_N = MD_Et_NonRecursion(m3, m4, nu, gamma, delta, CD_Dy);
                                        double Enu_G = Et_grad_NonRecursion(m3, m4, nu, gamma, delta, CD_Dy);
                                        double Enu_nm1 = MD_Et_NonRecursion(m3, m4, nu - 1, gamma, delta, CD_Dy);
                                        double Enu_dC_y = (gamma / q) * Enu_nm1 + Enu_G;
                                        double Enu_dD_y = (delta / q) * Enu_nm1 - Enu_G;

                                        for (int phi = 0; phi < lim_ket_z + 1; phi++) {
                                            double Ephi_N = MD_Et_NonRecursion(n3, n4, phi, gamma, delta, CD_Dz);
                                            double Ephi_G = Et_grad_NonRecursion(n3, n4, phi, gamma, delta, CD_Dz);
                                            double Ephi_pm1 = MD_Et_NonRecursion(n3, n4, phi - 1, gamma, delta, CD_Dz);
                                            double Ephi_dC_z = (gamma / q) * Ephi_pm1 + Ephi_G;
                                            double Ephi_dD_z = (delta / q) * Ephi_pm1 - Ephi_G;

                                            int k = t + u + v + tau + nu + phi;
                                            double sign = (1 - 2 * ((tau + nu + phi) & 1));

                                            // R value lookup
                                            int u_nu = u + nu;
                                            int v_phi = v + phi;
                                            double R_val = R[k*(k+1)*(k+2)/6 + v_phi*(k+2) - v_phi*(v_phi+1)/2 + u_nu];
                                            double common = sign * R_val;

                                            // Range flags: each derivative extends one direction by +1
                                            bool t_in = (t < lim_bra_x);
                                            bool u_in = (u < lim_bra_y);
                                            bool v_in = (v < lim_bra_z);
                                            bool tau_in = (tau < lim_ket_x);
                                            bool nu_in = (nu < lim_ket_y);
                                            bool phi_in = (phi < lim_ket_z);

                                            double bra_yz = Eu_N * Ev_N;
                                            double ket_base = Etau_N * Enu_N * Ephi_N;

                                            // Ax: derivative on bra-x
                                            if (u_in && v_in && tau_in && nu_in && phi_in)
                                                part[0] += Et_dA_x * bra_yz * ket_base * common;
                                            // Ay: derivative on bra-y
                                            if (t_in && v_in && tau_in && nu_in && phi_in)
                                                part[1] += Et_N * Eu_dA_y * Ev_N * ket_base * common;
                                            // Az: derivative on bra-z
                                            if (t_in && u_in && tau_in && nu_in && phi_in)
                                                part[2] += Et_N * Eu_N * Ev_dA_z * ket_base * common;
                                            // Bx
                                            if (u_in && v_in && tau_in && nu_in && phi_in)
                                                part[3] += Et_dB_x * bra_yz * ket_base * common;
                                            // By
                                            if (t_in && v_in && tau_in && nu_in && phi_in)
                                                part[4] += Et_N * Eu_dB_y * Ev_N * ket_base * common;
                                            // Bz
                                            if (t_in && u_in && tau_in && nu_in && phi_in)
                                                part[5] += Et_N * Eu_N * Ev_dB_z * ket_base * common;
                                            // Cx
                                            if (t_in && u_in && v_in && nu_in && phi_in)
                                                part[6] += Et_N * bra_yz * Etau_dC_x * Enu_N * Ephi_N * common;
                                            // Cy
                                            if (t_in && u_in && v_in && tau_in && phi_in)
                                                part[7] += Et_N * bra_yz * Etau_N * Enu_dC_y * Ephi_N * common;
                                            // Cz
                                            if (t_in && u_in && v_in && tau_in && nu_in)
                                                part[8] += Et_N * bra_yz * Etau_N * Enu_N * Ephi_dC_z * common;
                                            // Dx
                                            if (t_in && u_in && v_in && nu_in && phi_in)
                                                part[9] += Et_N * bra_yz * Etau_dD_x * Enu_N * Ephi_N * common;
                                            // Dy
                                            if (t_in && u_in && v_in && tau_in && phi_in)
                                                part[10] += Et_N * bra_yz * Etau_N * Enu_dD_y * Ephi_N * common;
                                            // Dz
                                            if (t_in && u_in && v_in && tau_in && nu_in)
                                                part[11] += Et_N * bra_yz * Etau_N * Enu_N * Ephi_dD_z * common;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Accumulate weighted ERI gradient for this component combination
                    for (int dir = 0; dir < 3; dir++) {
                        grad_atom[0 + dir] += w * part[0 + dir];   // atom A
                        grad_atom[3 + dir] += w * part[3 + dir];   // atom B
                        grad_atom[6 + dir] += w * part[6 + dir];   // atom C
                        grad_atom[9 + dir] += w * part[9 + dir];   // atom D
                    }
                }
            }
        }
    }

    // Write results to global gradient array via atomicAdd
    for (int dir = 0; dir < 3; dir++) {
        atomicAdd(&g_gradients[3*a.atom_index + dir], grad_atom[0 + dir]);
        atomicAdd(&g_gradients[3*b.atom_index + dir], grad_atom[3 + dir]);
        atomicAdd(&g_gradients[3*c.atom_index + dir], grad_atom[6 + dir]);
        atomicAdd(&g_gradients[3*d.atom_index + dir], grad_atom[9 + dir]);
    }
}



// ---- compute_gradients_two_electron_uhf ----
// UHF version: takes alpha and beta density matrices separately.
// Coulomb uses D_total = Da + Db, Exchange uses Da and Db independently.
// density_w = 0.5 * Dt_ab * Dt_cd - 0.25 * (Da_ac*Da_bd + Da_ad*Da_bc + Db_ac*Db_bd + Db_ad*Db_bc)
__global__
void compute_gradients_two_electron_uhf(double* g_gradients, const real_t* g_density_matrix_a, const real_t* g_density_matrix_b,
                                        const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors,
                                        const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
                                        const size_t num_threads, const int num_basis, const double* g_boys_grid)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= num_threads) return;

    // Compute 4D index from thread id
    size_t ket_size;
    if(shell_s2.start_index == shell_s3.start_index){
        ket_size = (shell_s2.count * (shell_s2.count+1)) / 2;
    }else{
        ket_size = shell_s2.count*shell_s3.count;
    }
    const size_t2 abcd = index1to2(id, (shell_s0.start_index == shell_s2.start_index && shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y, shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    // Obtain primitive shells [ab|cd]
    const size_t primitive_index_a = ab.x+shell_s0.start_index;
    const size_t primitive_index_b = ab.y+shell_s1.start_index;
    const size_t primitive_index_c = cd.x+shell_s2.start_index;
    const size_t primitive_index_d = cd.y+shell_s3.start_index;

    const PrimitiveShell a = g_shell[primitive_index_a];
    const PrimitiveShell b = g_shell[primitive_index_b];
    const PrimitiveShell c = g_shell[primitive_index_c];
    const PrimitiveShell d = g_shell[primitive_index_d];

    // Obtain basis index (starting index for angular momentum components)
    const size_t ia = a.basis_index;
    const size_t ib = b.basis_index;
    const size_t ic = c.basis_index;
    const size_t id_idx = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = (primitive_index_a == primitive_index_c && primitive_index_b == primitive_index_d);

    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double delta = d.exponent;
    const double p = alpha + beta;
    const double q = gamma + delta;

    const double3 P = make_double3((alpha*a.coordinate.x + beta*b.coordinate.x)/p, (alpha*a.coordinate.y + beta*b.coordinate.y)/p, (alpha*a.coordinate.z + beta*b.coordinate.z)/p);
    const double3 Q = make_double3((gamma*c.coordinate.x + delta*d.coordinate.x)/q, (gamma*c.coordinate.y + delta*d.coordinate.y)/q, (gamma*c.coordinate.z + delta*d.coordinate.z)/q);

    const double AB_Dx = a.coordinate.x - b.coordinate.x;
    const double AB_Dy = a.coordinate.y - b.coordinate.y;
    const double AB_Dz = a.coordinate.z - b.coordinate.z;

    const double CD_Dx = c.coordinate.x - d.coordinate.x;
    const double CD_Dy = c.coordinate.y - d.coordinate.y;
    const double CD_Dz = c.coordinate.z - d.coordinate.z;

    const int K = a.shell_type + b.shell_type + c.shell_type + d.shell_type + 1;

    double CoefBase = a.coefficient*b.coefficient*c.coefficient*d.coefficient * 2*M_PI_2_5 /(p*q*sqrt((p+q)));

    // Symmetry factor (8-fold ERI symmetry)
    int sym_f = 1 + static_cast<int>(!is_bra_symmetric) + static_cast<int>(!is_ket_symmetric)
              + static_cast<int>(!is_bra_symmetric && !is_ket_symmetric)
              + static_cast<int>(!is_braket_symmetric) * (1 + static_cast<int>(!is_bra_symmetric)
              + static_cast<int>(!is_ket_symmetric) + static_cast<int>(!is_bra_symmetric && !is_ket_symmetric));

    // R and Boys computation — ONCE per primitive quartet
    double R_mid[3*size_Rmid];
    double R[size_R];
    double Boys[boys_size];

    getIncrementalBoys(K, (p*q/(p+q)) * calc_dist_GPU(P, Q), g_boys_grid, Boys);
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2 * (p*q/(p+q))), i));
    }

    // Compute R integrals with maximum extents
    compute_R_TripleBuffer(R, R_mid, Boys, P, Q, K, K, K, K);

    // Accumulate gradient contributions over angular momentum components
    double grad_atom[12] = {0.0};  // [Ax,Ay,Az, Bx,By,Bz, Cx,Cy,Cz, Dx,Dy,Dz]

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int l1 = loop_to_ang[a.shell_type][lmn_a][0];
        int m1 = loop_to_ang[a.shell_type][lmn_a][1];
        int n1 = loop_to_ang[a.shell_type][lmn_a][2];
        double NA = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){
            int l2 = loop_to_ang[b.shell_type][lmn_b][0];
            int m2 = loop_to_ang[b.shell_type][lmn_b][1];
            int n2 = loop_to_ang[b.shell_type][lmn_b][2];
            double NB = calcNorm(beta, l2, m2, n2);

            for(int lmn_c=0; lmn_c<comb_max(c.shell_type); lmn_c++){
                int l3 = loop_to_ang[c.shell_type][lmn_c][0];
                int m3 = loop_to_ang[c.shell_type][lmn_c][1];
                int n3 = loop_to_ang[c.shell_type][lmn_c][2];
                double NC = calcNorm(gamma, l3, m3, n3);

                for(int lmn_d=0; lmn_d<comb_max(d.shell_type); lmn_d++){
                    int l4 = loop_to_ang[d.shell_type][lmn_d][0];
                    int m4 = loop_to_ang[d.shell_type][lmn_d][1];
                    int n4 = loop_to_ang[d.shell_type][lmn_d][2];
                    double ND = calcNorm(delta, l4, m4, n4);

                    // Per-component density matrix elements (alpha)
                    double Da_ab = g_density_matrix_a[(ia + lmn_a) * num_basis + (ib + lmn_b)];
                    double Da_cd = g_density_matrix_a[(ic + lmn_c) * num_basis + (id_idx + lmn_d)];
                    double Da_ac = g_density_matrix_a[(ia + lmn_a) * num_basis + (ic + lmn_c)];
                    double Da_bd = g_density_matrix_a[(ib + lmn_b) * num_basis + (id_idx + lmn_d)];
                    double Da_ad = g_density_matrix_a[(ia + lmn_a) * num_basis + (id_idx + lmn_d)];
                    double Da_bc = g_density_matrix_a[(ib + lmn_b) * num_basis + (ic + lmn_c)];

                    // Per-component density matrix elements (beta)
                    double Db_ab = g_density_matrix_b[(ia + lmn_a) * num_basis + (ib + lmn_b)];
                    double Db_cd = g_density_matrix_b[(ic + lmn_c) * num_basis + (id_idx + lmn_d)];
                    double Db_ac = g_density_matrix_b[(ia + lmn_a) * num_basis + (ic + lmn_c)];
                    double Db_bd = g_density_matrix_b[(ib + lmn_b) * num_basis + (id_idx + lmn_d)];
                    double Db_ad = g_density_matrix_b[(ia + lmn_a) * num_basis + (id_idx + lmn_d)];
                    double Db_bc = g_density_matrix_b[(ib + lmn_b) * num_basis + (ic + lmn_c)];

                    // UHF: Coulomb uses D_total, Exchange uses Da and Db separately
                    double Dt_ab = Da_ab + Db_ab;
                    double Dt_cd = Da_cd + Db_cd;
                    double density_w = 0.5 * Dt_ab * Dt_cd
                                     - 0.25 * (Da_ac * Da_bd + Da_ad * Da_bc
                                              + Db_ac * Db_bd + Db_ad * Db_bc);
                    if (fabs(density_w) < 1.0e-18) continue;

                    // Weight: symmetry * coefficients * cgto_norm (per-component) * primitive_norm * density
                    double w = static_cast<double>(sym_f) * CoefBase
                        * g_cgto_normalization_factors[ia + lmn_a] * g_cgto_normalization_factors[ib + lmn_b]
                        * g_cgto_normalization_factors[ic + lmn_c] * g_cgto_normalization_factors[id_idx + lmn_d]
                        * NA * NB * NC * ND * density_w;

                    // Single-pass 6-loop computing all 12 gradient components
                    int lim_bra_x = l1 + l2 + 1;
                    int lim_bra_y = m1 + m2 + 1;
                    int lim_bra_z = n1 + n2 + 1;
                    int lim_ket_x = l3 + l4 + 1;
                    int lim_ket_y = m3 + m4 + 1;
                    int lim_ket_z = n3 + n4 + 1;

                    double part[12] = {0.0};

                    for (int t = 0; t < lim_bra_x + 1; t++) {
                        double Et_N = MD_Et_NonRecursion(l1, l2, t, alpha, beta, AB_Dx);
                        double Et_G = Et_grad_NonRecursion(l1, l2, t, alpha, beta, AB_Dx);
                        double Et_tm1 = MD_Et_NonRecursion(l1, l2, t - 1, alpha, beta, AB_Dx);
                        double Et_dA_x = (alpha / p) * Et_tm1 + Et_G;
                        double Et_dB_x = (beta / p) * Et_tm1 - Et_G;

                        for (int u = 0; u < lim_bra_y + 1; u++) {
                            double Eu_N = MD_Et_NonRecursion(m1, m2, u, alpha, beta, AB_Dy);
                            double Eu_G = Et_grad_NonRecursion(m1, m2, u, alpha, beta, AB_Dy);
                            double Eu_um1 = MD_Et_NonRecursion(m1, m2, u - 1, alpha, beta, AB_Dy);
                            double Eu_dA_y = (alpha / p) * Eu_um1 + Eu_G;
                            double Eu_dB_y = (beta / p) * Eu_um1 - Eu_G;

                            for (int v = 0; v < lim_bra_z + 1; v++) {
                                double Ev_N = MD_Et_NonRecursion(n1, n2, v, alpha, beta, AB_Dz);
                                double Ev_G = Et_grad_NonRecursion(n1, n2, v, alpha, beta, AB_Dz);
                                double Ev_vm1 = MD_Et_NonRecursion(n1, n2, v - 1, alpha, beta, AB_Dz);
                                double Ev_dA_z = (alpha / p) * Ev_vm1 + Ev_G;
                                double Ev_dB_z = (beta / p) * Ev_vm1 - Ev_G;

                                for (int tau = 0; tau < lim_ket_x + 1; tau++) {
                                    double Etau_N = MD_Et_NonRecursion(l3, l4, tau, gamma, delta, CD_Dx);
                                    double Etau_G = Et_grad_NonRecursion(l3, l4, tau, gamma, delta, CD_Dx);
                                    double Etau_tm1 = MD_Et_NonRecursion(l3, l4, tau - 1, gamma, delta, CD_Dx);
                                    double Etau_dC_x = (gamma / q) * Etau_tm1 + Etau_G;
                                    double Etau_dD_x = (delta / q) * Etau_tm1 - Etau_G;

                                    for (int nu = 0; nu < lim_ket_y + 1; nu++) {
                                        double Enu_N = MD_Et_NonRecursion(m3, m4, nu, gamma, delta, CD_Dy);
                                        double Enu_G = Et_grad_NonRecursion(m3, m4, nu, gamma, delta, CD_Dy);
                                        double Enu_nm1 = MD_Et_NonRecursion(m3, m4, nu - 1, gamma, delta, CD_Dy);
                                        double Enu_dC_y = (gamma / q) * Enu_nm1 + Enu_G;
                                        double Enu_dD_y = (delta / q) * Enu_nm1 - Enu_G;

                                        for (int phi = 0; phi < lim_ket_z + 1; phi++) {
                                            double Ephi_N = MD_Et_NonRecursion(n3, n4, phi, gamma, delta, CD_Dz);
                                            double Ephi_G = Et_grad_NonRecursion(n3, n4, phi, gamma, delta, CD_Dz);
                                            double Ephi_pm1 = MD_Et_NonRecursion(n3, n4, phi - 1, gamma, delta, CD_Dz);
                                            double Ephi_dC_z = (gamma / q) * Ephi_pm1 + Ephi_G;
                                            double Ephi_dD_z = (delta / q) * Ephi_pm1 - Ephi_G;

                                            int k = t + u + v + tau + nu + phi;
                                            double sign = (1 - 2 * ((tau + nu + phi) & 1));

                                            // R value lookup
                                            int u_nu = u + nu;
                                            int v_phi = v + phi;
                                            double R_val = R[k*(k+1)*(k+2)/6 + v_phi*(k+2) - v_phi*(v_phi+1)/2 + u_nu];
                                            double common = sign * R_val;

                                            // Range flags
                                            bool t_in = (t < lim_bra_x);
                                            bool u_in = (u < lim_bra_y);
                                            bool v_in = (v < lim_bra_z);
                                            bool tau_in = (tau < lim_ket_x);
                                            bool nu_in = (nu < lim_ket_y);
                                            bool phi_in = (phi < lim_ket_z);

                                            double bra_yz = Eu_N * Ev_N;
                                            double ket_base = Etau_N * Enu_N * Ephi_N;

                                            if (u_in && v_in && tau_in && nu_in && phi_in)
                                                part[0] += Et_dA_x * bra_yz * ket_base * common;
                                            if (t_in && v_in && tau_in && nu_in && phi_in)
                                                part[1] += Et_N * Eu_dA_y * Ev_N * ket_base * common;
                                            if (t_in && u_in && tau_in && nu_in && phi_in)
                                                part[2] += Et_N * Eu_N * Ev_dA_z * ket_base * common;
                                            if (u_in && v_in && tau_in && nu_in && phi_in)
                                                part[3] += Et_dB_x * bra_yz * ket_base * common;
                                            if (t_in && v_in && tau_in && nu_in && phi_in)
                                                part[4] += Et_N * Eu_dB_y * Ev_N * ket_base * common;
                                            if (t_in && u_in && tau_in && nu_in && phi_in)
                                                part[5] += Et_N * Eu_N * Ev_dB_z * ket_base * common;
                                            if (t_in && u_in && v_in && nu_in && phi_in)
                                                part[6] += Et_N * bra_yz * Etau_dC_x * Enu_N * Ephi_N * common;
                                            if (t_in && u_in && v_in && tau_in && phi_in)
                                                part[7] += Et_N * bra_yz * Etau_N * Enu_dC_y * Ephi_N * common;
                                            if (t_in && u_in && v_in && tau_in && nu_in)
                                                part[8] += Et_N * bra_yz * Etau_N * Enu_N * Ephi_dC_z * common;
                                            if (t_in && u_in && v_in && nu_in && phi_in)
                                                part[9] += Et_N * bra_yz * Etau_dD_x * Enu_N * Ephi_N * common;
                                            if (t_in && u_in && v_in && tau_in && phi_in)
                                                part[10] += Et_N * bra_yz * Etau_N * Enu_dD_y * Ephi_N * common;
                                            if (t_in && u_in && v_in && tau_in && nu_in)
                                                part[11] += Et_N * bra_yz * Etau_N * Enu_N * Ephi_dD_z * common;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Accumulate weighted ERI gradient for this component combination
                    for (int dir = 0; dir < 3; dir++) {
                        grad_atom[0 + dir] += w * part[0 + dir];
                        grad_atom[3 + dir] += w * part[3 + dir];
                        grad_atom[6 + dir] += w * part[6 + dir];
                        grad_atom[9 + dir] += w * part[9 + dir];
                    }
                }
            }
        }
    }

    // Write results to global gradient array via atomicAdd
    for (int dir = 0; dir < 3; dir++) {
        atomicAdd(&g_gradients[3*a.atom_index + dir], grad_atom[0 + dir]);
        atomicAdd(&g_gradients[3*b.atom_index + dir], grad_atom[3 + dir]);
        atomicAdd(&g_gradients[3*c.atom_index + dir], grad_atom[6 + dir]);
        atomicAdd(&g_gradients[3*d.atom_index + dir], grad_atom[9 + dir]);
    }
}


} // namespace gansu::gpu
