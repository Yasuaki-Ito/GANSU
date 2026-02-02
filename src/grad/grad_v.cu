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

#include "device_function_grad_v.txt"


// ----- compute_gradients_nuclear -----
__global__
void compute_gradients_nuclear(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                const Atom* g_atom, const int num_atoms, const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, 
                                const size_t num_threads, const real_t* g_boys_grid)
{
	const size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= num_threads) return;

	size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count); // Convert 1D index to 2D index a,b of [a|b]

	const size_t primitive_index_a = ab.y+shell_s0.start_index;
	const size_t primitive_index_b = ab.x+shell_s1.start_index;
	const PrimitiveShell a = g_shell[primitive_index_a];
	const PrimitiveShell b = g_shell[primitive_index_b];

	size_t i = a.basis_index; // Obtain basis index (i|j)
	size_t j = b.basis_index;

    if(fabs(g_density_matrix[i*num_basis + j]) < 1.0e-15) return;

    const double alpha = a.exponent;
    const double beta = b.exponent;
	const double p = alpha + beta;

	const double3 P = make_double3((alpha*a.coordinate.x + beta*b.coordinate.x)/p, (alpha*a.coordinate.y + beta*b.coordinate.y)/p, (alpha*a.coordinate.z + beta*b.coordinate.z)/p);

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

	const int K = a.shell_type + b.shell_type + 1;
	double Boys[boys_one_size];

    double iteration_Ax, iteration_Ay, iteration_Az;
    double iteration_Bx, iteration_By, iteration_Bz;
    double iteration_Cx, iteration_Cy, iteration_Cz;
    double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;
    double result_Bx = 0.0, result_By = 0.0, result_Bz = 0.0;

    double Norm_A, Norm_B;
    double CoefandNorm = a.coefficient * b.coefficient * ((2*M_PI)/p) * g_cgto_normalization_factors[i]*g_cgto_normalization_factors[j];

    double R_mid[3*size_one_Rmid];
    double R[size_one_R];

    for(int atom_index=0; atom_index<num_atoms; atom_index++){
        iteration_Ax = iteration_Ay = iteration_Az = 0.0;
        iteration_Bx = iteration_By = iteration_Bz = 0.0;
        iteration_Cx = iteration_Cy = iteration_Cz = 0.0;
        getIncrementalBoys(K, p*calc_dist_GPU(P, g_atom[atom_index].coordinate), g_boys_grid, Boys);
        for(int x=0; x <= K; x++){
            Boys[x] *= (right2left_binary_woif((-2*p), x));
        }

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

                compute_R_TripleBuffer(R, R_mid, Boys, P, g_atom[atom_index].coordinate, K, l1+l2+1, m1+m2+1, n1+n2+1);

                iteration_Ax += Norm_A * Norm_B * compute_grad_Ax(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Ay += Norm_A * Norm_B * compute_grad_Ay(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Az += Norm_A * Norm_B * compute_grad_Az(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Bx += Norm_A * Norm_B * compute_grad_Bx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_By += Norm_A * Norm_B * compute_grad_By(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Bz += Norm_A * Norm_B * compute_grad_Bz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Cx += Norm_A * Norm_B * compute_grad_Cx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Cy += Norm_A * Norm_B * compute_grad_Cy(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iteration_Cz += Norm_A * Norm_B * compute_grad_Cz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
            }
        }

        // A includes the effects of C
        iteration_Ax -= iteration_Cx;
        iteration_Ay -= iteration_Cy;
        iteration_Az -= iteration_Cz;
        // iteration_Bx -= iteration_Cx;
        // iteration_By -= iteration_Cy;
        // iteration_Bz -= iteration_Cz;

        result_Ax += g_atom[atom_index].atomic_number * iteration_Ax;
        result_Ay += g_atom[atom_index].atomic_number * iteration_Ay;
        result_Az += g_atom[atom_index].atomic_number * iteration_Az;

        result_Bx += g_atom[atom_index].atomic_number * iteration_Bx;
        result_By += g_atom[atom_index].atomic_number * iteration_By;
        result_Bz += g_atom[atom_index].atomic_number * iteration_Bz;
    }

    result_Ax *= CoefandNorm * g_density_matrix[i*num_basis + j];
    result_Ay *= CoefandNorm * g_density_matrix[i*num_basis + j];
    result_Az *= CoefandNorm * g_density_matrix[i*num_basis + j];

    result_Bx *= CoefandNorm * g_density_matrix[i*num_basis + j];
    result_By *= CoefandNorm * g_density_matrix[i*num_basis + j];
    result_Bz *= CoefandNorm * g_density_matrix[i*num_basis + j];

    // atomicAdd
    AddToResult(g_gradients, 3*a.atom_index+0, result_Ax, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*a.atom_index+1, result_Ay, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*a.atom_index+2, result_Az, primitive_index_a != primitive_index_b && i != j);

    AddToResult(g_gradients, 3*b.atom_index+0, result_Bx, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*b.atom_index+1, result_By, primitive_index_a != primitive_index_b && i != j);
    AddToResult(g_gradients, 3*b.atom_index+2, result_Bz, primitive_index_a != primitive_index_b && i != j);

}

} // namespace gansu::gpu