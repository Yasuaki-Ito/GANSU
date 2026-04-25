/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Nuclear repulsion Hessian: d²V_nn/dR_A dR_B
// Purely analytical (no integrals needed)

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#endif
#include <cmath>
#include "types.hpp"
#include "int2e.hpp"  // gansu_atomic_add

namespace gansu::gpu {

// V_nn = Σ_{A<B} Z_A Z_B / |R_A - R_B|
//
// d²V_nn/dA_x dA_x = Σ_{C≠A} Z_A Z_C (3(A_x-C_x)² - |A-C|²) / |A-C|⁵
// d²V_nn/dA_x dA_y = Σ_{C≠A} Z_A Z_C 3(A_x-C_x)(A_y-C_y) / |A-C|⁵
// d²V_nn/dA_x dB_x = -Z_A Z_B (3(A_x-B_x)² - |A-B|²) / |A-B|⁵  (A≠B)
// d²V_nn/dA_x dB_y = -Z_A Z_B 3(A_x-B_x)(A_y-B_y) / |A-B|⁵     (A≠B)

__global__
void compute_hessian_nuclear_repulsion(
    double* g_hessian,
    const Atom* g_atoms,
    const int num_atoms)
{
    // One thread per atom pair (A, B) where A <= B
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pairs = num_atoms * (num_atoms + 1) / 2;
    if (id >= num_pairs) return;

    const int ndim = 3 * num_atoms;

    // Map 1D index to (A, B) with A <= B
    int A = 0, B = 0;
    int acc = 0;
    for (int a = 0; a < num_atoms; a++) {
        if (id < acc + (num_atoms - a)) {
            A = a;
            B = a + (id - acc);
            break;
        }
        acc += (num_atoms - a);
    }

    if (A == B) {
        // Diagonal block: d²V_nn/dA dA = Σ_{C≠A} contribution
        double Z_A = (double)g_atoms[A].effective_charge;
        double hess[9] = {0.0};

        for (int C = 0; C < num_atoms; C++) {
            if (C == A) continue;
            double Z_C = (double)g_atoms[C].effective_charge;
            double dx = g_atoms[A].coordinate.x - g_atoms[C].coordinate.x;
            double dy = g_atoms[A].coordinate.y - g_atoms[C].coordinate.y;
            double dz = g_atoms[A].coordinate.z - g_atoms[C].coordinate.z;
            double r2 = dx * dx + dy * dy + dz * dz;
            double r = sqrt(r2);
            double r5 = r2 * r2 * r;
            double ZZ_r5 = Z_A * Z_C / r5;

            hess[0] += ZZ_r5 * (3 * dx * dx - r2); // xx
            hess[4] += ZZ_r5 * (3 * dy * dy - r2); // yy
            hess[8] += ZZ_r5 * (3 * dz * dz - r2); // zz
            hess[1] += ZZ_r5 * 3 * dx * dy;         // xy
            hess[2] += ZZ_r5 * 3 * dx * dz;         // xz
            hess[5] += ZZ_r5 * 3 * dy * dz;         // yz
        }

        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = d1; d2 < 3; d2++) {
                int idx = d1 * 3 + d2;
                double val = hess[idx];
                if (val != 0.0) {
                    atomicAdd(&g_hessian[(3*A+d1)*ndim + (3*A+d2)], val);
                    if (d1 != d2)
                        atomicAdd(&g_hessian[(3*A+d2)*ndim + (3*A+d1)], val);
                }
            }
        }
    } else {
        // Off-diagonal block: d²V_nn/dA dB = -Z_A Z_B * f(R_AB)
        double Z_A = (double)g_atoms[A].effective_charge, Z_B = (double)g_atoms[B].effective_charge;
        double dx = g_atoms[A].coordinate.x - g_atoms[B].coordinate.x;
        double dy = g_atoms[A].coordinate.y - g_atoms[B].coordinate.y;
        double dz = g_atoms[A].coordinate.z - g_atoms[B].coordinate.z;
        double r2 = dx * dx + dy * dy + dz * dz;
        double r = sqrt(r2);
        double r5 = r2 * r2 * r;
        double ZZ_r5 = -Z_A * Z_B / r5;

        double hess[9];
        hess[0] = ZZ_r5 * (3 * dx * dx - r2); // xx
        hess[4] = ZZ_r5 * (3 * dy * dy - r2); // yy
        hess[8] = ZZ_r5 * (3 * dz * dz - r2); // zz
        hess[1] = ZZ_r5 * 3 * dx * dy;         // xy
        hess[2] = ZZ_r5 * 3 * dx * dz;         // xz
        hess[5] = ZZ_r5 * 3 * dy * dz;         // yz

        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = d1; d2 < 3; d2++) {
                int idx = d1 * 3 + d2;
                double val = hess[idx];
                if (val == 0.0) continue;

                // AB block
                atomicAdd(&g_hessian[(3*A+d1)*ndim + (3*B+d2)], val);
                if (d1 != d2)
                    atomicAdd(&g_hessian[(3*A+d2)*ndim + (3*B+d1)], val);

                // BA block (symmetric)
                atomicAdd(&g_hessian[(3*B+d1)*ndim + (3*A+d2)], val);
                if (d1 != d2)
                    atomicAdd(&g_hessian[(3*B+d2)*ndim + (3*A+d1)], val);
            }
        }
    }
}



// CPU host-callable mirror of compute_hessian_nuclear_repulsion.
void compute_hessian_nuclear_repulsion_cpu(double* g_hessian, const Atom* g_atoms, const int num_atoms)
{
    const int num_pairs = num_atoms * (num_atoms + 1) / 2;
    const int ndim = 3 * num_atoms;

    #pragma omp parallel for schedule(dynamic)
    for (int id = 0; id < num_pairs; id++) {
        // Map 1D index to (A, B) with A <= B
        int A = 0, B = 0;
        int acc = 0;
        for (int a = 0; a < num_atoms; a++) {
            if (id < acc + (num_atoms - a)) {
                A = a;
                B = a + (id - acc);
                break;
            }
            acc += (num_atoms - a);
        }

        if (A == B) {
            double Z_A = (double)g_atoms[A].effective_charge;
            double hess[9] = {0.0};
            for (int C = 0; C < num_atoms; C++) {
                if (C == A) continue;
                double Z_C = (double)g_atoms[C].effective_charge;
                double dx = g_atoms[A].coordinate.x - g_atoms[C].coordinate.x;
                double dy = g_atoms[A].coordinate.y - g_atoms[C].coordinate.y;
                double dz = g_atoms[A].coordinate.z - g_atoms[C].coordinate.z;
                double r2 = dx*dx + dy*dy + dz*dz;
                double r = std::sqrt(r2);
                double r5 = r2 * r2 * r;
                double ZZ_r5 = Z_A * Z_C / r5;
                hess[0] += ZZ_r5 * (3*dx*dx - r2);
                hess[4] += ZZ_r5 * (3*dy*dy - r2);
                hess[8] += ZZ_r5 * (3*dz*dz - r2);
                hess[1] += ZZ_r5 * 3*dx*dy;
                hess[2] += ZZ_r5 * 3*dx*dz;
                hess[5] += ZZ_r5 * 3*dy*dz;
            }
            for (int d1 = 0; d1 < 3; d1++)
                for (int d2 = d1; d2 < 3; d2++) {
                    double val = hess[d1*3+d2];
                    if (val != 0.0) {
                        gansu_atomic_add(&g_hessian[(3*A+d1)*ndim + (3*A+d2)], val);
                        if (d1 != d2)
                            gansu_atomic_add(&g_hessian[(3*A+d2)*ndim + (3*A+d1)], val);
                    }
                }
        } else {
            double Z_A = (double)g_atoms[A].effective_charge, Z_B = (double)g_atoms[B].effective_charge;
            double dx = g_atoms[A].coordinate.x - g_atoms[B].coordinate.x;
            double dy = g_atoms[A].coordinate.y - g_atoms[B].coordinate.y;
            double dz = g_atoms[A].coordinate.z - g_atoms[B].coordinate.z;
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = std::sqrt(r2);
            double r5 = r2 * r2 * r;
            double ZZ_r5 = -Z_A * Z_B / r5;
            double hess[9];
            hess[0] = ZZ_r5 * (3*dx*dx - r2);
            hess[4] = ZZ_r5 * (3*dy*dy - r2);
            hess[8] = ZZ_r5 * (3*dz*dz - r2);
            hess[1] = ZZ_r5 * 3*dx*dy;
            hess[2] = ZZ_r5 * 3*dx*dz;
            hess[5] = ZZ_r5 * 3*dy*dz;
            for (int d1 = 0; d1 < 3; d1++)
                for (int d2 = d1; d2 < 3; d2++) {
                    double val = hess[d1*3+d2];
                    if (val == 0.0) continue;
                    gansu_atomic_add(&g_hessian[(3*A+d1)*ndim + (3*B+d2)], val);
                    if (d1 != d2)
                        gansu_atomic_add(&g_hessian[(3*A+d2)*ndim + (3*B+d1)], val);
                    gansu_atomic_add(&g_hessian[(3*B+d1)*ndim + (3*A+d2)], val);
                    if (d1 != d2)
                        gansu_atomic_add(&g_hessian[(3*B+d2)*ndim + (3*A+d1)], val);
                }
        }
    }
}

} // namespace gansu::gpu
