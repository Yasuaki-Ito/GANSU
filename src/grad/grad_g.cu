/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include <cuda.h>
#include <cmath>

#include "gradients.hpp"


namespace gansu::gpu{


#include "device_function_grad_g.txt"

// ---- compute_gradients_two_electron ----
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

    // Obtain basis index (ij|kl)
    const size_t Cindex_a = a.basis_index;
    const size_t Cindex_b = b.basis_index;
    const size_t Cindex_c = c.basis_index;
    const size_t Cindex_d = d.basis_index;

    bool is_bra_symmetric = (primitive_index_a == primitive_index_b);
    bool is_ket_symmetric = (primitive_index_c == primitive_index_d);
    bool is_braket_symmetric = (primitive_index_a == primitive_index_c && primitive_index_b == primitive_index_d);

    // quick density-based early exit (optional; uncomment if desired)
    if ((g_density_matrix[Cindex_a * num_basis + Cindex_b] < 1.0e-15 ||
         g_density_matrix[Cindex_c * num_basis + Cindex_d] < 1.0e-15) &&
        (g_density_matrix[Cindex_a * num_basis + Cindex_c] < 1.0e-15 ||
         g_density_matrix[Cindex_b * num_basis + Cindex_d] < 1.0e-15)) return;

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

    double Norm_A, Norm_B, Norm_C, Norm_D;

    double CoefandNorm = a.coefficient*b.coefficient*c.coefficient*d.coefficient * 2*M_PI_2_5 /(p*q*sqrt((p+q))) * g_cgto_normalization_factors[Cindex_a]*g_cgto_normalization_factors[Cindex_b]*g_cgto_normalization_factors[Cindex_c]*g_cgto_normalization_factors[Cindex_d];

    double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;
    double result_Bx = 0.0, result_By = 0.0, result_Bz = 0.0;
    double result_Cx = 0.0, result_Cy = 0.0, result_Cz = 0.0;
    double result_Dx = 0.0, result_Dy = 0.0, result_Dz = 0.0;


    double R_mid[3*size_Rmid];
    double R[size_R];

    // Boys関数の計算
    double Boys[boys_size];
    getIncrementalBoys(K, (p*q/(p+q)) * calc_dist_GPU(P, Q), g_boys_grid, Boys);
    for(int i=0; i <= K; i++){
        Boys[i] *= (right2left_binary_woif((-2 * (p*q/(p+q))), i));
    }

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int3 A_ang = make_int3(loop_to_ang[a.shell_type][lmn_a][0], loop_to_ang[a.shell_type][lmn_a][1], loop_to_ang[a.shell_type][lmn_a][2]);
        Norm_A = calcNorm(alpha, A_ang.x, A_ang.y, A_ang.z);

        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){
            int3 B_ang = make_int3(loop_to_ang[b.shell_type][lmn_b][0], loop_to_ang[b.shell_type][lmn_b][1], loop_to_ang[b.shell_type][lmn_b][2]);
            Norm_B = calcNorm(beta, B_ang.x, B_ang.y, B_ang.z);

            for(int lmn_c=0; lmn_c<comb_max(c.shell_type); lmn_c++){
                int3 C_ang = make_int3(loop_to_ang[c.shell_type][lmn_c][0], loop_to_ang[c.shell_type][lmn_c][1], loop_to_ang[c.shell_type][lmn_c][2]);
                Norm_C = calcNorm(gamma, C_ang.x, C_ang.y, C_ang.z);

                for(int lmn_d=0; lmn_d<comb_max(d.shell_type); lmn_d++){
                    int3 D_ang = make_int3(loop_to_ang[d.shell_type][lmn_d][0], loop_to_ang[d.shell_type][lmn_d][1], loop_to_ang[d.shell_type][lmn_d][2]);
                    Norm_D = calcNorm(delta, D_ang.x, D_ang.y, D_ang.z);

                    // トリプルバッファリングで使用(MD法)
                    compute_R_TripleBuffer(R, R_mid, Boys, P, Q, K, A_ang.x+B_ang.x+C_ang.x+D_ang.x+1, A_ang.y+B_ang.y+C_ang.y+D_ang.y+1, A_ang.z+B_ang.z+C_ang.z+D_ang.z+1);  // Rの計算

                    // ERI計算部
                    result_Ax += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Ax(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Ay += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Ay(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Az += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Az(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Bx += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Bx(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_By += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_By(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Bz += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Bz(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Cx += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Cx(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Cy += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Cy(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Cz += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Cz(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Dx += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Dx(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Dy += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Dy(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);
                    result_Dz += Norm_A * Norm_B * Norm_C * Norm_D * compute_two_grad_Dz(A_ang, B_ang, C_ang, D_ang, alpha, beta, gamma, delta, AB_Dx, AB_Dy, AB_Dz, CD_Dx, CD_Dy, CD_Dz, p, q, P, Q, R);

                }
            }
        }
    }

    // --- Coulomb contribution (added to atoms a and b) ---
    const double D_abcd = 0.5 * CoefandNorm * g_density_matrix[Cindex_a * num_basis + Cindex_b] * g_density_matrix[Cindex_c * num_basis + Cindex_d];

    AddToResult_TEI(g_gradients, 3*a.atom_index+0, D_abcd * result_Ax, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*a.atom_index+1, D_abcd * result_Ay, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*a.atom_index+2, D_abcd * result_Az, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+0, D_abcd * result_Bx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+1, D_abcd * result_By, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+2, D_abcd * result_Bz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+0, D_abcd * result_Cx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+1, D_abcd * result_Cy, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+2, D_abcd * result_Cz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+0, D_abcd * result_Dx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+1, D_abcd * result_Dy, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+2, D_abcd * result_Dz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);

    // --- Exchange contribution (added to atoms a and b, with -1/2 prefactor) ---
    const double D_acbd = 0.5 * CoefandNorm * g_density_matrix[Cindex_a * num_basis + Cindex_c] + g_density_matrix[Cindex_b * num_basis + Cindex_d];
    const double exch_pref = -0.5;

    AddToResult_TEI(g_gradients, 3*a.atom_index+0, exch_pref * D_acbd * result_Ax, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*a.atom_index+1, exch_pref * D_acbd * result_Ay, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*a.atom_index+2, exch_pref * D_acbd * result_Az, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+0, exch_pref * D_acbd * result_Bx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+1, exch_pref * D_acbd * result_By, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*b.atom_index+2, exch_pref * D_acbd * result_Bz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+0, exch_pref * D_acbd * result_Cx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+1, exch_pref * D_acbd * result_Cy, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*c.atom_index+2, exch_pref * D_acbd * result_Cz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+0, exch_pref * D_acbd * result_Dx, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+1, exch_pref * D_acbd * result_Dy, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
    AddToResult_TEI(g_gradients, 3*d.atom_index+2, exch_pref * D_acbd * result_Dz, is_bra_symmetric, is_ket_symmetric, is_braket_symmetric);
}

} // namespace gansu::gpu