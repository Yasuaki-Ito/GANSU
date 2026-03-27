/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Nuclear attraction integral Hessian: d²V/dR dR
//
// V_μν = -Σ_C Z_C ∫ φ_μ(r) (1/|r-C|) φ_ν(r) dr
//
// This is a 3-center 1-electron integral (2 basis centers + 1 nucleus).
// Uses Rys quadrature with angular momentum shift formula.
//
// Derivatives:
//   d/dA_x V = 2α V(l1+1,...) - l1 V(l1-1,...)       (Pulay on basis A)
//   d/dC_x V = explicit nuclear derivative (Hellmann-Feynman)
//
// For the Hessian, there are 3 types of atom pairs:
//   (A, A), (A, B), (A, C), (B, B), (B, C), (C, C)
//   where A,B are basis centers and C is nucleus center.

#include <cuda.h>
#include <cmath>
#include "gradients.hpp"
#include "rys_quadrature.hpp"

namespace gansu::gpu {

// VRR for nuclear attraction (same structure as 2-center ERI)
inline __device__
void vrr_1d_nai(int a_max, double PA, double PC, double B10, double B00,
                double* __restrict__ I) {
    // 1D VRR for nuclear attraction: only "bra" direction
    // I(0) = 1
    // I(a+1) = PA * I(a) + a * B10 * I(a-1) + 0 * B00 * I(...)
    // Actually for NAI, the "ket" is a point charge, so c_max = 0
    // VRR simplifies to:
    // I(a+1) = PA' * I(a) + a * B10' * I(a-1)
    // where PA' = C00 (depends on Rys root), B10' as in 2-center
    I[0] = 1.0;
    if (a_max > 0) {
        I[1] = PA;
        for (int a = 1; a < a_max; a++)
            I[a+1] = PA * I[a] + a * B10 * I[a-1];
    }
}

__global__
void compute_hessian_nuclear_attraction(
    double* g_hessian,
    const real_t* g_density_matrix,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const Atom* g_atoms,
    const int num_basis,
    const int num_atoms,
    ShellTypeInfo shell_s0, ShellTypeInfo shell_s1,
    const size_t num_threads,
    const double* g_boys_grid)
{
    const size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_threads) return;

    const int ndim = 3 * num_atoms;

    size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t pidx_a = ab.y + shell_s0.start_index;
    const size_t pidx_b = ab.x + shell_s1.start_index;
    const PrimitiveShell a = g_shell[pidx_a];
    const PrimitiveShell b = g_shell[pidx_b];

    const size_t bi_a = a.basis_index, bi_b = b.basis_index;
    const double alpha = a.exponent, beta = b.exponent;
    const double p = alpha + beta;

    const double Ax = a.coordinate.x, Ay = a.coordinate.y, Az = a.coordinate.z;
    const double Bx = b.coordinate.x, By = b.coordinate.y, Bz = b.coordinate.z;
    const double Px = (alpha*Ax + beta*Bx)/p;
    const double Py = (alpha*Ay + beta*By)/p;
    const double Pz = (alpha*Az + beta*Bz)/p;

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double AB2 = ABx*ABx + ABy*ABy + ABz*ABz;
    const double K_AB = exp(-alpha*beta/p * AB2);

    double CoefBase = a.coefficient * b.coefficient * 2.0 * M_PI / p * K_AB;
    double factor = (pidx_a != pidx_b) ? 2.0 : 1.0;

    const int iA = a.atom_index, iB = b.atom_index;

    // Loop over all nuclei C
    for (int iC = 0; iC < num_atoms; iC++) {
        double Z_C = (double)g_atoms[iC].atomic_number;
        if (Z_C == 0.0) continue;

        double Cx = g_atoms[iC].coordinate.x, Cy = g_atoms[iC].coordinate.y, Cz = g_atoms[iC].coordinate.z;
        double PCx = Px-Cx, PCy = Py-Cy, PCz = Pz-Cz;
        double PC2 = PCx*PCx + PCy*PCy + PCz*PCz;
        double T = p * PC2;

        // Component loop
        for (int lmn_a = 0; lmn_a < comb_max(a.shell_type); lmn_a++) {
            int l1 = loop_to_ang[a.shell_type][lmn_a][0];
            int m1 = loop_to_ang[a.shell_type][lmn_a][1];
            int n1 = loop_to_ang[a.shell_type][lmn_a][2];
            double NA = calcNorm(alpha, l1, m1, n1);

            for (int lmn_b = 0; lmn_b < comb_max(b.shell_type); lmn_b++) {
                int l2 = loop_to_ang[b.shell_type][lmn_b][0];
                int m2 = loop_to_ang[b.shell_type][lmn_b][1];
                int n2 = loop_to_ang[b.shell_type][lmn_b][2];
                double NB = calcNorm(beta, l2, m2, n2);

                double D_comp = g_density_matrix[(bi_a+lmn_a)*num_basis + (bi_b+lmn_b)];
                if (fabs(D_comp) < 1e-18) continue;

                double w = -Z_C * CoefBase * g_cgto_normalization_factors[bi_a+lmn_a]
                         * g_cgto_normalization_factors[bi_b+lmn_b]
                         * factor * D_comp * NA * NB;

                // NAI Hessian: 3 centers (A=basis_μ, B=basis_ν, C=nucleus)
                // Need AA, BB, AB blocks independently; C blocks from translational invariance
                // d/dA acts on φ_μ (shift α, l1/m1/n1)
                // d/dB acts on φ_ν (shift β, l2/m2/n2)
                // d/dC = -(d/dA + d/dB) from translational invariance

                int L = l1+l2+m1+m2+n1+n2;
                int N = (L+2)/2 + 1;
                double rys_roots[9], rys_weights[9];
                computeRysRootsAndWeights(N, T, g_boys_grid, rys_roots, rys_weights);

                // hess_AA[3×3], hess_BB[3×3], hess_AB[3×3]
                double hAA[9]={0}, hBB[9]={0}, hAB[9]={0};

                for (int n = 0; n < N; n++) {
                    double t2 = rys_roots[n], wn = rys_weights[n];
                    double B10_r = (1.0 - t2) / (2.0 * p);

                    double C00x = (Px-Ax) + t2*(Cx-Px);
                    double C00y = (Py-Ay) + t2*(Cy-Py);
                    double C00z = (Pz-Az) + t2*(Cz-Pz);

                    int ax_max = l1+l2+2, ay_max = m1+m2+2, az_max = n1+n2+2;
                    double Ix[13], Iy[13], Iz[13];

                    vrr_1d_nai(ax_max, C00x, 0, B10_r, 0, Ix);
                    vrr_1d_nai(ay_max, C00y, 0, B10_r, 0, Iy);
                    vrr_1d_nai(az_max, C00z, 0, B10_r, 0, Iz);

                    // TRR: I(a,b) from VRR values I_vrr[0..a+b]
                    // I(a,b) = Σ_k C(b,k) AB^k I_vrr[a+b-k]
                    #define NAI_1D(vrr, a, b, AB_val) ({ \
                        double _v = 0.0; int _bn = 1; double _ap = 1.0; \
                        for(int _k=0;_k<=(b);_k++){ \
                            _v += _bn * _ap * (vrr)[(a)+(b)-_k]; \
                            if(_k<(b)){_ap*=(AB_val); _bn=_bn*((b)-_k)/(_k+1);} \
                        } _v; })

                    // Base integrals V_x(l1,l2), V_y(m1,m2), V_z(n1,n2)
                    double Vx = NAI_1D(Ix, l1, l2, ABx);
                    double Vy = NAI_1D(Iy, m1, m2, ABy);
                    double Vz = NAI_1D(Iz, n1, n2, ABz);

                    // --- First derivatives on A (shift exponent α, AM l1/m1/n1) ---
                    double dVx_A = 2*alpha * NAI_1D(Ix, l1+1, l2, ABx)
                                 - (l1>0 ? l1*NAI_1D(Ix, l1-1, l2, ABx) : 0);
                    double dVy_A = 2*alpha * NAI_1D(Iy, m1+1, m2, ABy)
                                 - (m1>0 ? m1*NAI_1D(Iy, m1-1, m2, ABy) : 0);
                    double dVz_A = 2*alpha * NAI_1D(Iz, n1+1, n2, ABz)
                                 - (n1>0 ? n1*NAI_1D(Iz, n1-1, n2, ABz) : 0);

                    // --- First derivatives on B (shift exponent β, AM l2/m2/n2) ---
                    double dVx_B = 2*beta * NAI_1D(Ix, l1, l2+1, ABx)
                                 - (l2>0 ? l2*NAI_1D(Ix, l1, l2-1, ABx) : 0);
                    double dVy_B = 2*beta * NAI_1D(Iy, m1, m2+1, ABy)
                                 - (m2>0 ? m2*NAI_1D(Iy, m1, m2-1, ABy) : 0);
                    double dVz_B = 2*beta * NAI_1D(Iz, n1, n2+1, ABz)
                                 - (n2>0 ? n2*NAI_1D(Iz, n1, n2-1, ABz) : 0);

                    // --- Second derivatives on A: d²/dA_σ² ---
                    double d2Vx_AA = 4*alpha*alpha * NAI_1D(Ix, l1+2, l2, ABx)
                                   - 2*alpha*(2*l1+1) * Vx
                                   + (l1>1 ? l1*(l1-1)*NAI_1D(Ix, l1-2, l2, ABx) : 0);
                    double d2Vy_AA = 4*alpha*alpha * NAI_1D(Iy, m1+2, m2, ABy)
                                   - 2*alpha*(2*m1+1) * Vy
                                   + (m1>1 ? m1*(m1-1)*NAI_1D(Iy, m1-2, m2, ABy) : 0);
                    double d2Vz_AA = 4*alpha*alpha * NAI_1D(Iz, n1+2, n2, ABz)
                                   - 2*alpha*(2*n1+1) * Vz
                                   + (n1>1 ? n1*(n1-1)*NAI_1D(Iz, n1-2, n2, ABz) : 0);

                    // --- Second derivatives on B: d²/dB_σ² ---
                    double d2Vx_BB = 4*beta*beta * NAI_1D(Ix, l1, l2+2, ABx)
                                   - 2*beta*(2*l2+1) * Vx
                                   + (l2>1 ? l2*(l2-1)*NAI_1D(Ix, l1, l2-2, ABx) : 0);
                    double d2Vy_BB = 4*beta*beta * NAI_1D(Iy, m1, m2+2, ABy)
                                   - 2*beta*(2*m2+1) * Vy
                                   + (m2>1 ? m2*(m2-1)*NAI_1D(Iy, m1, m2-2, ABy) : 0);
                    double d2Vz_BB = 4*beta*beta * NAI_1D(Iz, n1, n2+2, ABz)
                                   - 2*beta*(2*n2+1) * Vz
                                   + (n2>1 ? n2*(n2-1)*NAI_1D(Iz, n1, n2-2, ABz) : 0);

                    // --- Mixed second derivatives on A,B: d²V/(dA_σ dB_τ) ---
                    // dA acts on (l1/m1/n1 with α), dB acts on (l2/m2/n2 with β)
                    double d2Vx_AB = (2*alpha * NAI_1D(Ix, l1+1, l2+1, ABx)
                                    - (l1>0 ? l1*NAI_1D(Ix, l1-1, l2+1, ABx) : 0)) * 2*beta
                                   -(2*alpha * NAI_1D(Ix, l1+1, l2-1, ABx)
                                    - (l1>0 ? l1*NAI_1D(Ix, l1-1, l2-1, ABx) : 0)) * (l2>0 ? l2 : 0);
                    double d2Vy_AB = (2*alpha * NAI_1D(Iy, m1+1, m2+1, ABy)
                                    - (m1>0 ? m1*NAI_1D(Iy, m1-1, m2+1, ABy) : 0)) * 2*beta
                                   -(2*alpha * NAI_1D(Iy, m1+1, m2-1, ABy)
                                    - (m1>0 ? m1*NAI_1D(Iy, m1-1, m2-1, ABy) : 0)) * (m2>0 ? m2 : 0);
                    double d2Vz_AB = (2*alpha * NAI_1D(Iz, n1+1, n2+1, ABz)
                                    - (n1>0 ? n1*NAI_1D(Iz, n1-1, n2+1, ABz) : 0)) * 2*beta
                                   -(2*alpha * NAI_1D(Iz, n1+1, n2-1, ABz)
                                    - (n1>0 ? n1*NAI_1D(Iz, n1-1, n2-1, ABz) : 0)) * (n2>0 ? n2 : 0);

                    // === AA block: d²V/(dA_σ dA_τ) ===
                    // σ=τ: d²Vσ_AA * Πτ≠σ Vτ
                    hAA[0*3+0] += wn * d2Vx_AA * Vy * Vz;
                    hAA[1*3+1] += wn * Vx * d2Vy_AA * Vz;
                    hAA[2*3+2] += wn * Vx * Vy * d2Vz_AA;
                    // σ≠τ: dVσ_A * dVτ_A * Πρ≠σ,τ Vρ
                    hAA[0*3+1] += wn * dVx_A * dVy_A * Vz;
                    hAA[0*3+2] += wn * dVx_A * Vy * dVz_A;
                    hAA[1*3+2] += wn * Vx * dVy_A * dVz_A;

                    // === BB block: d²V/(dB_σ dB_τ) ===
                    hBB[0*3+0] += wn * d2Vx_BB * Vy * Vz;
                    hBB[1*3+1] += wn * Vx * d2Vy_BB * Vz;
                    hBB[2*3+2] += wn * Vx * Vy * d2Vz_BB;
                    hBB[0*3+1] += wn * dVx_B * dVy_B * Vz;
                    hBB[0*3+2] += wn * dVx_B * Vy * dVz_B;
                    hBB[1*3+2] += wn * Vx * dVy_B * dVz_B;

                    // === AB block: d²V/(dA_σ dB_τ) ===
                    // σ=τ: d²Vσ_AB * Πρ≠σ Vρ
                    hAB[0*3+0] += wn * d2Vx_AB * Vy * Vz;
                    hAB[1*3+1] += wn * Vx * d2Vy_AB * Vz;
                    hAB[2*3+2] += wn * Vx * Vy * d2Vz_AB;
                    // σ≠τ: dVσ_A * dVτ_B * Πρ≠σ,τ Vρ
                    hAB[0*3+1] += wn * dVx_A * dVy_B * Vz;
                    hAB[0*3+2] += wn * dVx_A * Vy * dVz_B;
                    hAB[1*3+0] += wn * dVy_A * dVx_B * Vz;
                    hAB[1*3+2] += wn * Vx * dVy_A * dVz_B;
                    hAB[2*3+0] += wn * dVz_A * dVx_B * Vy;
                    hAB[2*3+1] += wn * Vx * dVz_A * dVy_B;

                    #undef NAI_1D
                }

                // Symmetrize AA and BB (only upper triangle was accumulated)
                hAA[1*3+0] = hAA[0*3+1]; hAA[2*3+0] = hAA[0*3+2]; hAA[2*3+1] = hAA[1*3+2];
                hBB[1*3+0] = hBB[0*3+1]; hBB[2*3+0] = hBB[0*3+2]; hBB[2*3+1] = hBB[1*3+2];
                // AB is NOT symmetric (d²V/dA_x dB_y ≠ d²V/dA_y dB_x in general)

                // Write AA, BB, AB blocks to global Hessian
                // Then derive C-related blocks from translational invariance:
                //   d/dC = -(d/dA + d/dB)
                //   d²V/(dA dC) = -(d²V/dA dA + d²V/dA dB)
                //   d²V/(dB dC) = -(d²V/dB dA + d²V/dB dB)
                //   d²V/(dC dC) = d²V/dA dA + d²V/dA dB + d²V/dB dA + d²V/dB dB
                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        double vAA = w * hAA[d1*3+d2];
                        double vBB = w * hBB[d1*3+d2];
                        double vAB = w * hAB[d1*3+d2];
                        double vBA = w * hAB[d2*3+d1]; // d²V/(dB_d1 dA_d2)

                        // AA block
                        atomicAdd(&g_hessian[(3*iA+d1)*ndim+(3*iA+d2)], vAA);
                        // BB block
                        atomicAdd(&g_hessian[(3*iB+d1)*ndim+(3*iB+d2)], vBB);
                        // AB block
                        atomicAdd(&g_hessian[(3*iA+d1)*ndim+(3*iB+d2)], vAB);
                        // BA block = transpose of AB
                        atomicAdd(&g_hessian[(3*iB+d1)*ndim+(3*iA+d2)], vBA);

                        // AC block: d²V/(dA_d1 dC_d2) = -(d²V/dA_d1 dA_d2 + d²V/dA_d1 dB_d2)
                        double vAC = -(vAA + vAB);
                        atomicAdd(&g_hessian[(3*iA+d1)*ndim+(3*iC+d2)], vAC);
                        // CA block = transpose
                        atomicAdd(&g_hessian[(3*iC+d2)*ndim+(3*iA+d1)], vAC);

                        // BC block: d²V/(dB_d1 dC_d2) = -(d²V/dB_d1 dA_d2 + d²V/dB_d1 dB_d2)
                        double vBC = -(vBA + vBB);
                        atomicAdd(&g_hessian[(3*iB+d1)*ndim+(3*iC+d2)], vBC);
                        // CB block = transpose
                        atomicAdd(&g_hessian[(3*iC+d2)*ndim+(3*iB+d1)], vBC);

                        // CC block: d²V/(dC_d1 dC_d2) = AA + AB + BA + BB
                        double vCC = vAA + vAB + vBA + vBB;
                        atomicAdd(&g_hessian[(3*iC+d1)*ndim+(3*iC+d2)], vCC);
                    }
                }
            }
        }
    }
}

} // namespace gansu::gpu
