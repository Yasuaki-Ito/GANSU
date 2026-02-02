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


// ----- compute_gradients_overlap -----
__global__
void compute_gradients_overlap(double* g_gradients, const real_t* g_W_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads)
{
	const size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= num_threads) return;

	size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count); // Convert 1D index to 2D index a,b of [a|b]

	const size_t primitive_index_a = ab.y+shell_s0.start_index;
	const size_t primitive_index_b = ab.x+shell_s1.start_index;
	const PrimitiveShell a = g_shell[primitive_index_a];
	const PrimitiveShell b = g_shell[primitive_index_b];

	size_t i = a.basis_index;
	size_t j = b.basis_index;

    if(fabs(g_W_matrix[i*num_basis + j]) < 1.0e-15) return;

    const double alpha = a.exponent;
    const double beta = b.exponent;

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

    double result_Ax = 0.0;
    double result_Ay = 0.0;
    double result_Az = 0.0;
   
    double Norm_A, Norm_B;
    double CoefandNorm = a.coefficient * b.coefficient * M_PI/(alpha + beta) * sqrt(M_PI/(alpha + beta)) * g_cgto_normalization_factors[i]*g_cgto_normalization_factors[j];

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int l1=loop_to_ang[a.shell_type][lmn_a][0];
        int m1=loop_to_ang[a.shell_type][lmn_a][1]; 
        int n1=loop_to_ang[a.shell_type][lmn_a][2];
        Norm_A = calcNorm(alpha, l1, m1, n1);
        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){                  
            int l2=loop_to_ang[b.shell_type][lmn_b][0]; 
            int m2=loop_to_ang[b.shell_type][lmn_b][1]; 
            int n2=loop_to_ang[b.shell_type][lmn_b][2];
            Norm_B = calcNorm(beta, l2, m2, n2);

            result_Ax += Norm_A * Norm_B * Et_grad_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            result_Ay += Norm_A * Norm_B * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * Et_grad_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            result_Az += Norm_A * Norm_B * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * Et_grad_NonRecursion(n1, n2, 0, alpha, beta, Dz);

        }
    }
    result_Ax *= CoefandNorm * (-g_W_matrix[i*num_basis + j]);
    result_Ay *= CoefandNorm * (-g_W_matrix[i*num_basis + j]);
    result_Az *= CoefandNorm * (-g_W_matrix[i*num_basis + j]);

    AddToResult(g_gradients, 3*a.atom_index+0, result_Ax, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*a.atom_index+1, result_Ay, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*a.atom_index+2, result_Az, primitive_index_a != primitive_index_b && i != j);

    AddToResult(g_gradients, 3*b.atom_index+0, -result_Ax, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*b.atom_index+1, -result_Ay, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*b.atom_index+2, -result_Az, primitive_index_a != primitive_index_b && i != j);
}


} // namespace gansu::gpu