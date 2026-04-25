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


#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif
#include <cmath>

#include "gradients.hpp"


namespace gansu::gpu{


// 重なり部分の微分の係数
__global__ void compute_W_Matrix_kernel(real_t* d_W_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_electron, const int num_basis){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_basis * num_basis) return;

    size_t i = idx / num_basis;
    size_t j = idx % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_electron / 2; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k] * d_orbital_energies[k];
    }
    d_W_matrix[idx] = 2.0 * sum;
}

// CPU host-callable equivalent of compute_W_Matrix_kernel.  Mirrors the GPU
// kernel exactly so the analytic CPU gradient path can call it directly
// without going through cuda kernel launch syntax.
void compute_W_Matrix_cpu(real_t* d_W_matrix, const real_t* d_coefficient_matrix,
                         const real_t* d_orbital_energies,
                         const int num_electron, const int num_basis)
{
    const size_t total = (size_t)num_basis * num_basis;
    #pragma omp parallel for schedule(static)
    for (long long idx_ll = 0; idx_ll < (long long)total; idx_ll++) {
        size_t idx = (size_t)idx_ll;
        size_t i = idx / num_basis;
        size_t j = idx % num_basis;
        real_t sum = 0.0;
        for (size_t k = 0; k < (size_t)(num_electron / 2); k++) {
            sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k] * d_orbital_energies[k];
        }
        d_W_matrix[idx] = 2.0 * sum;
    }
}



// 核反発部分の微分を計算
__global__ void compute_nuclear_repulsion_gradient_kernel(double* g_grad, const Atom* g_atom, const int num_atoms){
    size_t A = blockIdx.x * blockDim.x + threadIdx.x;
    if (A >= num_atoms) return;

    double grad_x = 0.0;
    double grad_y = 0.0;
    double grad_z = 0.0;

    for (size_t B = 0; B < num_atoms; B++) {
        if (B == A) continue;

        double dx = g_atom[B].coordinate.x - g_atom[A].coordinate.x;
        double dy = g_atom[B].coordinate.y - g_atom[A].coordinate.y;
        double dz = g_atom[B].coordinate.z - g_atom[A].coordinate.z;
        double R_AB2 = dx * dx + dy * dy + dz * dz;
        double R_AB3 = R_AB2 * sqrt(R_AB2);

        grad_x += g_atom[B].effective_charge * dx / R_AB3;
        grad_y += g_atom[B].effective_charge * dy / R_AB3;
        grad_z += g_atom[B].effective_charge * dz / R_AB3;
    }

    atomicAdd(&g_grad[3*A+0], g_atom[A].effective_charge * grad_x);
    atomicAdd(&g_grad[3*A+1], g_atom[A].effective_charge * grad_y);
    atomicAdd(&g_grad[3*A+2], g_atom[A].effective_charge * grad_z);
}

// CPU host-callable nuclear repulsion gradient.  Mirrors the GPU kernel.
// Each atom A is independent so a plain serial loop is enough; no
// atomicAdd needed because A indexes the gradient slot uniquely.
void compute_nuclear_repulsion_gradient_cpu(double* g_grad, const Atom* g_atom,
                                            const int num_atoms)
{
    #pragma omp parallel for schedule(static)
    for (int A = 0; A < num_atoms; A++) {
        double grad_x = 0.0;
        double grad_y = 0.0;
        double grad_z = 0.0;
        for (int B = 0; B < num_atoms; B++) {
            if (B == A) continue;
            double dx = g_atom[B].coordinate.x - g_atom[A].coordinate.x;
            double dy = g_atom[B].coordinate.y - g_atom[A].coordinate.y;
            double dz = g_atom[B].coordinate.z - g_atom[A].coordinate.z;
            double R_AB2 = dx * dx + dy * dy + dz * dz;
            double R_AB3 = R_AB2 * std::sqrt(R_AB2);
            grad_x += g_atom[B].effective_charge * dx / R_AB3;
            grad_y += g_atom[B].effective_charge * dy / R_AB3;
            grad_z += g_atom[B].effective_charge * dz / R_AB3;
        }
        g_grad[3*A+0] += g_atom[A].effective_charge * grad_x;
        g_grad[3*A+1] += g_atom[A].effective_charge * grad_y;
        g_grad[3*A+2] += g_atom[A].effective_charge * grad_z;
    }
}



// 計算結果を出力するカーネル
__global__ void printGradientMatrix_Kernel(const double* g_grad, int num_atoms){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx != 0) return;

    printf("=== gradients_on_GPU_after_atomicAdd ===\n");
    printf("[\n");
    for (int i = 0; i < num_atoms; ++i) {
        printf(" [%10.6f, %10.6f, %10.6f]", g_grad[3*i+0], g_grad[3*i+1], g_grad[3*i+2]);
        if (i != num_atoms - 1) printf(",");
        printf("\n");
    }
    printf("]\n\n");
}

} // namespace gansu::gpu