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

#include "device_function_grad_v.txt"


// ----- compute_gradients_nuclear -----
// Fixed: per-component density/cgto_norm indexing (was shell-level)
// Fixed: compute_R_TripleBuffer moved outside lmn loop (was per lmn combination)
__global__
void compute_gradients_nuclear(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors,
                                const Atom* g_atom, const int num_atoms, const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                                const size_t num_threads, const real_t* g_boys_grid)
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
	const double p = alpha + beta;

	const double3 P = make_double3((alpha*a.coordinate.x + beta*b.coordinate.x)/p, (alpha*a.coordinate.y + beta*b.coordinate.y)/p, (alpha*a.coordinate.z + beta*b.coordinate.z)/p);

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

	const int K = a.shell_type + b.shell_type + 1;
	double Boys[boys_one_size];

    // Coefficient base WITHOUT cgto_norm (applied per-component inside loop)
    double CoefBase = a.coefficient * b.coefficient * ((2*M_PI)/p);

    // Symmetry factor
    double factor = (primitive_index_a != primitive_index_b) ? 2.0 : 1.0;

    double R_mid[3*size_one_Rmid];
    double R[size_one_R];

    double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;
    double result_Bx = 0.0, result_By = 0.0, result_Bz = 0.0;

    for(int atom_index=0; atom_index<num_atoms; atom_index++){
        getIncrementalBoys(K, p*calc_dist_GPU(P, g_atom[atom_index].coordinate), g_boys_grid, Boys);
        for(int x=0; x <= K; x++){
            Boys[x] *= (right2left_binary_woif((-2*p), x));
        }

        // Compute R once per atom with maximum extents (moved outside lmn loop)
        compute_R_TripleBuffer(R, R_mid, Boys, P, g_atom[atom_index].coordinate, K, K, K, K);

        double iter_Ax = 0.0, iter_Ay = 0.0, iter_Az = 0.0;
        double iter_Bx = 0.0, iter_By = 0.0, iter_Bz = 0.0;
        double iter_Cx = 0.0, iter_Cy = 0.0, iter_Cz = 0.0;

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

                double w = CoefBase * g_cgto_normalization_factors[i + lmn_a] * g_cgto_normalization_factors[j + lmn_b]
                         * factor * D_comp * Norm_A * Norm_B;

                iter_Ax += w * compute_grad_Ax(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Ay += w * compute_grad_Ay(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Az += w * compute_grad_Az(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Bx += w * compute_grad_Bx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_By += w * compute_grad_By(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Bz += w * compute_grad_Bz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Cx += w * compute_grad_Cx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Cy += w * compute_grad_Cy(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                iter_Cz += w * compute_grad_Cz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
            }
        }

        double Z = g_atom[atom_index].atomic_number;

        // Pulay A: V_ab = -Z_C * <a|1/rC|b>, sign is -Z_C
        result_Ax -= Z * iter_Ax;
        result_Ay -= Z * iter_Ay;
        result_Az -= Z * iter_Az;

        // Pulay B: -Z_C * gB
        result_Bx -= Z * iter_Bx;
        result_By -= Z * iter_By;
        result_Bz -= Z * iter_Bz;

        // Hellmann-Feynman C: (-Z_C)*(-grad_C) = +Z_C*grad_C → to nucleus atom_index
        atomicAdd(&g_gradients[3*atom_index+0], Z * iter_Cx);
        atomicAdd(&g_gradients[3*atom_index+1], Z * iter_Cy);
        atomicAdd(&g_gradients[3*atom_index+2], Z * iter_Cz);
    }

    atomicAdd(&g_gradients[3*a.atom_index+0], result_Ax);
    atomicAdd(&g_gradients[3*a.atom_index+1], result_Ay);
    atomicAdd(&g_gradients[3*a.atom_index+2], result_Az);

    atomicAdd(&g_gradients[3*b.atom_index+0], result_Bx);
    atomicAdd(&g_gradients[3*b.atom_index+1], result_By);
    atomicAdd(&g_gradients[3*b.atom_index+2], result_Bz);
}


// CPU host-callable mirror of compute_gradients_nuclear.  Mirrors the GPU
// kernel exactly: computes the Pulay (basis-on-A, basis-on-B) and
// Hellmann-Feynman (nucleus-on-C) contributions to the nuclear-attraction
// gradient for one shell-pair element per OpenMP iteration.
void compute_gradients_nuclear_cpu(double* g_gradients, const real_t* g_density_matrix,
                                   const PrimitiveShell* g_shell,
                                   const real_t* g_cgto_normalization_factors,
                                   const Atom* g_atom, const int num_atoms,
                                   const int num_basis,
                                   ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                                   const size_t num_threads,
                                   const real_t* g_boys_grid)
{
    #pragma omp parallel for schedule(static)
    for (long long id_ll = 0; id_ll < (long long)num_threads; id_ll++) {
        size_t id = (size_t)id_ll;

        size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count);

        const size_t primitive_index_a = ab.y+shell_s0.start_index;
        const size_t primitive_index_b = ab.x+shell_s1.start_index;
        const PrimitiveShell a = g_shell[primitive_index_a];
        const PrimitiveShell b = g_shell[primitive_index_b];

        size_t i = a.basis_index;
        size_t j = b.basis_index;

        const double alpha = a.exponent;
        const double beta = b.exponent;
        const double p = alpha + beta;

        const double3 P = make_double3((alpha*a.coordinate.x + beta*b.coordinate.x)/p,
                                       (alpha*a.coordinate.y + beta*b.coordinate.y)/p,
                                       (alpha*a.coordinate.z + beta*b.coordinate.z)/p);

        const double Dx = a.coordinate.x - b.coordinate.x;
        const double Dy = a.coordinate.y - b.coordinate.y;
        const double Dz = a.coordinate.z - b.coordinate.z;

        const int K = a.shell_type + b.shell_type + 1;
        double Boys[boys_one_size];

        double CoefBase = a.coefficient * b.coefficient * ((2*M_PI)/p);
        double factor = (primitive_index_a != primitive_index_b) ? 2.0 : 1.0;

        double R_mid[3*size_one_Rmid];
        double R[size_one_R];

        double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;
        double result_Bx = 0.0, result_By = 0.0, result_Bz = 0.0;

        for(int atom_index=0; atom_index<num_atoms; atom_index++){
            getIncrementalBoys(K, p*calc_dist_GPU(P, g_atom[atom_index].coordinate), g_boys_grid, Boys);
            for(int x=0; x <= K; x++){
                Boys[x] *= (right2left_binary_woif((-2*p), x));
            }

            compute_R_TripleBuffer(R, R_mid, Boys, P, g_atom[atom_index].coordinate, K, K, K, K);

            double iter_Ax = 0.0, iter_Ay = 0.0, iter_Az = 0.0;
            double iter_Bx = 0.0, iter_By = 0.0, iter_Bz = 0.0;
            double iter_Cx = 0.0, iter_Cy = 0.0, iter_Cz = 0.0;

            for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
                int l1=loop_to_ang_host[a.shell_type][lmn_a][0];
                int m1=loop_to_ang_host[a.shell_type][lmn_a][1];
                int n1=loop_to_ang_host[a.shell_type][lmn_a][2];
                double Norm_A = calcNorm(alpha, l1, m1, n1);

                for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){
                    int l2=loop_to_ang_host[b.shell_type][lmn_b][0];
                    int m2=loop_to_ang_host[b.shell_type][lmn_b][1];
                    int n2=loop_to_ang_host[b.shell_type][lmn_b][2];
                    double Norm_B = calcNorm(beta, l2, m2, n2);

                    double D_comp = g_density_matrix[(i + lmn_a)*num_basis + (j + lmn_b)];
                    if(std::fabs(D_comp) < 1.0e-15) continue;

                    double w = CoefBase * g_cgto_normalization_factors[i + lmn_a] * g_cgto_normalization_factors[j + lmn_b]
                             * factor * D_comp * Norm_A * Norm_B;

                    iter_Ax += w * compute_grad_Ax(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Ay += w * compute_grad_Ay(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Az += w * compute_grad_Az(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Bx += w * compute_grad_Bx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_By += w * compute_grad_By(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Bz += w * compute_grad_Bz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Cx += w * compute_grad_Cx(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Cy += w * compute_grad_Cy(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                    iter_Cz += w * compute_grad_Cz(l1, m1, n1, l2, m2, n2, alpha, beta, Dx, Dy, Dz, P, g_atom[atom_index].coordinate, R);
                }
            }

            double Z = g_atom[atom_index].atomic_number;

            result_Ax -= Z * iter_Ax;
            result_Ay -= Z * iter_Ay;
            result_Az -= Z * iter_Az;
            result_Bx -= Z * iter_Bx;
            result_By -= Z * iter_By;
            result_Bz -= Z * iter_Bz;

            gansu_atomic_add(&g_gradients[3*atom_index+0], Z * iter_Cx);
            gansu_atomic_add(&g_gradients[3*atom_index+1], Z * iter_Cy);
            gansu_atomic_add(&g_gradients[3*atom_index+2], Z * iter_Cz);
        }

        gansu_atomic_add(&g_gradients[3*a.atom_index+0], result_Ax);
        gansu_atomic_add(&g_gradients[3*a.atom_index+1], result_Ay);
        gansu_atomic_add(&g_gradients[3*a.atom_index+2], result_Az);

        gansu_atomic_add(&g_gradients[3*b.atom_index+0], result_Bx);
        gansu_atomic_add(&g_gradients[3*b.atom_index+1], result_By);
        gansu_atomic_add(&g_gradients[3*b.atom_index+2], result_Bz);
    }
}

} // namespace gansu::gpu
