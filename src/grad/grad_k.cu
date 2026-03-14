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


#include <cuda.h>
#include <cmath>

#include "gradients.hpp"


namespace gansu::gpu{

// ----- compute_gradients_kinetic -----
// Fixed: per-component density matrix and cgto_norm indexing (was shell-level)
__global__
void compute_gradients_kinetic(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors,
                                const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads)
{
	const size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= num_threads) return;

	size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count);

	const size_t primitive_index_a = ab.y+shell_s0.start_index;
	const size_t primitive_index_b = ab.x+shell_s1.start_index;
	const PrimitiveShell a = g_shell[primitive_index_a];
	const PrimitiveShell b = g_shell[primitive_index_b];

	size_t i = a.basis_index;
	size_t j = b.basis_index;

    const double alpha = a.exponent;
    const double beta = b.exponent;

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

    double N_Ex_m2, N_Ex_0, N_Ex_p2, N_Ey_m2, N_Ey_0, N_Ey_p2, N_Ez_m2, N_Ez_0, N_Ez_p2;
    double G_Ex_m2, G_Ex_0, G_Ex_p2, G_Ey_m2, G_Ey_0, G_Ey_p2, G_Ez_m2, G_Ez_0, G_Ez_p2;

    // Coefficient base WITHOUT cgto_norm (applied per-component inside loop)
    double CoefBase = a.coefficient * b.coefficient * M_PI/(alpha + beta) * sqrt(M_PI/(alpha + beta));

    // Symmetry factor
    double factor = (primitive_index_a != primitive_index_b && i != j) ? 2.0 : 1.0;

    double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int l1=loop_to_ang[a.shell_type][lmn_a][0];
        int m1=loop_to_ang[a.shell_type][lmn_a][1];
        int n1=loop_to_ang[a.shell_type][lmn_a][2];
        double Norm_A = calcNorm(alpha, l1, m1, n1);

        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){
            int l2=loop_to_ang[b.shell_type][lmn_b][0];
            int m2=loop_to_ang[b.shell_type][lmn_b][1];
            int n2=loop_to_ang[b.shell_type][lmn_b][2];
            double Norm_B = calcNorm(beta, l2, m2, n2);

            // Per-component density and cgto_norm
            double D_comp = g_density_matrix[(i + lmn_a)*num_basis + (j + lmn_b)];
            if(fabs(D_comp) < 1.0e-15) continue;

            double cgto_prod = g_cgto_normalization_factors[i + lmn_a] * g_cgto_normalization_factors[j + lmn_b];
            double w = CoefBase * cgto_prod * factor * D_comp * Norm_A * Norm_B;

            N_Ex_m2 = MD_Et_NonRecursion(l1, l2-2, 0, alpha, beta, Dx);
            N_Ey_m2 = MD_Et_NonRecursion(m1, m2-2, 0, alpha, beta, Dy);
            N_Ez_m2 = MD_Et_NonRecursion(n1, n2-2, 0, alpha, beta, Dz);
            N_Ex_0  = MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx);
            N_Ey_0  = MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy);
            N_Ez_0  = MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            N_Ex_p2 = MD_Et_NonRecursion(l1, l2+2, 0, alpha, beta, Dx);
            N_Ey_p2 = MD_Et_NonRecursion(m1, m2+2, 0, alpha, beta, Dy);
            N_Ez_p2 = MD_Et_NonRecursion(n1, n2+2, 0, alpha, beta, Dz);

            G_Ex_m2 = Et_grad_NonRecursion(l1, l2-2, 0, alpha, beta, Dx);
            G_Ey_m2 = Et_grad_NonRecursion(m1, m2-2, 0, alpha, beta, Dy);
            G_Ez_m2 = Et_grad_NonRecursion(n1, n2-2, 0, alpha, beta, Dz);
            G_Ex_0  = Et_grad_NonRecursion(l1, l2, 0, alpha, beta, Dx);
            G_Ey_0  = Et_grad_NonRecursion(m1, m2, 0, alpha, beta, Dy);
            G_Ez_0  = Et_grad_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            G_Ex_p2 = Et_grad_NonRecursion(l1, l2+2, 0, alpha, beta, Dx);
            G_Ey_p2 = Et_grad_NonRecursion(m1, m2+2, 0, alpha, beta, Dy);
            G_Ez_p2 = Et_grad_NonRecursion(n1, n2+2, 0, alpha, beta, Dz);

            result_Ax += w * (
                (-2*beta*beta*G_Ex_p2 + (2*l2+1)*beta*G_Ex_0 - (l2*(l2-1)/2)*G_Ex_m2) * N_Ey_0 * N_Ez_0
                + G_Ex_0 * (-2*beta*beta*N_Ey_p2 + (2*m2+1)*beta*N_Ey_0 - (m2*(m2-1)/2)*N_Ey_m2) * N_Ez_0
                + G_Ex_0 * N_Ey_0 * (-2*beta*beta*N_Ez_p2 + (2*n2+1)*beta*N_Ez_0 - (n2*(n2-1)/2)*N_Ez_m2)
            );

            result_Ay += w * (
                (-2*beta*beta*N_Ex_p2 + (2*l2+1)*beta*N_Ex_0 - (l2*(l2-1)/2)*N_Ex_m2) * G_Ey_0 * N_Ez_0
                + N_Ex_0 * (-2*beta*beta*G_Ey_p2 + (2*m2+1)*beta*G_Ey_0 - (m2*(m2-1)/2)*G_Ey_m2) * N_Ez_0
                + N_Ex_0 * G_Ey_0 * (-2*beta*beta*N_Ez_p2 + (2*n2+1)*beta*N_Ez_0 - (n2*(n2-1)/2)*N_Ez_m2)
            );

            result_Az += w * (
                (-2*beta*beta*N_Ex_p2 + (2*l2+1)*beta*N_Ex_0 - (l2*(l2-1)/2)*N_Ex_m2) * N_Ey_0 * G_Ez_0
                + N_Ex_0 * (-2*beta*beta*N_Ey_p2 + (2*m2+1)*beta*N_Ey_0 - (m2*(m2-1)/2)*N_Ey_m2) * G_Ez_0
                + N_Ex_0 * N_Ey_0 * (-2*beta*beta*G_Ez_p2 + (2*n2+1)*beta*G_Ez_0 - (n2*(n2-1)/2)*G_Ez_m2)
            );
        }
    }

    // Translational invariance: dT/dR_B = -dT/dR_A
    atomicAdd(&g_gradients[3*a.atom_index+0], result_Ax);
    atomicAdd(&g_gradients[3*a.atom_index+1], result_Ay);
    atomicAdd(&g_gradients[3*a.atom_index+2], result_Az);

    atomicAdd(&g_gradients[3*b.atom_index+0], -result_Ax);
    atomicAdd(&g_gradients[3*b.atom_index+1], -result_Ay);
    atomicAdd(&g_gradients[3*b.atom_index+2], -result_Az);
}


} // namespace gansu::gpu
