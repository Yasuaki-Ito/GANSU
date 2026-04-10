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


// ----- compute_gradients_overlap -----
// Fixed: per-component W matrix and cgto_norm indexing (was shell-level)
__global__
void compute_gradients_overlap(double* g_gradients, const real_t* g_W_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors,
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

    // Coefficient base WITHOUT cgto_norm (applied per-component inside loop)
    double CoefBase = a.coefficient * b.coefficient * M_PI/(alpha + beta) * sqrt(M_PI/(alpha + beta));

    // Symmetry factor: double for off-diagonal primitive pairs with different basis indices
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

            // Per-component W matrix and cgto_norm
            double W_comp = -g_W_matrix[(i + lmn_a)*num_basis + (j + lmn_b)];
            if(fabs(W_comp) < 1.0e-15) continue;

            double w = CoefBase * g_cgto_normalization_factors[i + lmn_a] * g_cgto_normalization_factors[j + lmn_b]
                     * factor * W_comp * Norm_A * Norm_B;

            result_Ax += w * Et_grad_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            result_Ay += w * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * Et_grad_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            result_Az += w * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * Et_grad_NonRecursion(n1, n2, 0, alpha, beta, Dz);
        }
    }

    // Translational invariance: dS/dR_B = -dS/dR_A
    atomicAdd(&g_gradients[3*a.atom_index+0], result_Ax);
    atomicAdd(&g_gradients[3*a.atom_index+1], result_Ay);
    atomicAdd(&g_gradients[3*a.atom_index+2], result_Az);

    atomicAdd(&g_gradients[3*b.atom_index+0], -result_Ax);
    atomicAdd(&g_gradients[3*b.atom_index+1], -result_Ay);
    atomicAdd(&g_gradients[3*b.atom_index+2], -result_Az);
}


// CPU host-callable version of compute_gradients_overlap.  Mirrors the GPU
// kernel exactly: each "thread id" handles one shell-pair element of the
// per-shell-type-pair triangular block.  Atomic accumulation into
// g_gradients[3*atom+dir] is needed because multiple shell pairs share the
// same atom slot, so we use the gansu_atomic_add wrapper which compiles to
// `#pragma omp atomic` on host.
void compute_gradients_overlap_cpu(double* g_gradients, const real_t* g_W_matrix,
                                   const PrimitiveShell* g_shell,
                                   const real_t* g_cgto_normalization_factors,
                                   const int num_basis,
                                   ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
                                   const size_t num_threads)
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

        const double Dx = a.coordinate.x - b.coordinate.x;
        const double Dy = a.coordinate.y - b.coordinate.y;
        const double Dz = a.coordinate.z - b.coordinate.z;

        double CoefBase = a.coefficient * b.coefficient * M_PI/(alpha + beta) * std::sqrt(M_PI/(alpha + beta));

        double factor = (primitive_index_a != primitive_index_b && i != j) ? 2.0 : 1.0;

        double result_Ax = 0.0, result_Ay = 0.0, result_Az = 0.0;

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

                double W_comp = -g_W_matrix[(i + lmn_a)*num_basis + (j + lmn_b)];
                if(std::fabs(W_comp) < 1.0e-15) continue;

                double w = CoefBase * g_cgto_normalization_factors[i + lmn_a] * g_cgto_normalization_factors[j + lmn_b]
                         * factor * W_comp * Norm_A * Norm_B;

                result_Ax += w * Et_grad_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
                result_Ay += w * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * Et_grad_NonRecursion(m1, m2, 0, alpha, beta, Dy) * MD_Et_NonRecursion(n1, n2, 0, alpha, beta, Dz);
                result_Az += w * MD_Et_NonRecursion(l1, l2, 0, alpha, beta, Dx) * MD_Et_NonRecursion(m1, m2, 0, alpha, beta, Dy) * Et_grad_NonRecursion(n1, n2, 0, alpha, beta, Dz);
            }
        }

        gansu_atomic_add(&g_gradients[3*a.atom_index+0],  result_Ax);
        gansu_atomic_add(&g_gradients[3*a.atom_index+1],  result_Ay);
        gansu_atomic_add(&g_gradients[3*a.atom_index+2],  result_Az);
        gansu_atomic_add(&g_gradients[3*b.atom_index+0], -result_Ax);
        gansu_atomic_add(&g_gradients[3*b.atom_index+1], -result_Ay);
        gansu_atomic_add(&g_gradients[3*b.atom_index+2], -result_Az);
    }
}


} // namespace gansu::gpu
