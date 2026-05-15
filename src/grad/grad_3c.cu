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
#include "grad_3c.hpp"

namespace gansu::gpu {

// ============================================================================
// Single (μν|P) primitive triple — Γ^(3)-weighted gradient contribution.
//
// This mirrors the structure of MD_int3c2e_1T1SP (src/int3c2e.cu) but accumulates
// 9 derivative components (∂R_μ, ∂R_ν, ∂R_P_aux) × (x, y, z) per call.
//
// The 1-electron compute_grad_A/B/Cx helpers in grad_v.cu give the analytic
// recipe for the μν-side Et derivatives:
//   A-side: Et^{A_x}_t = (α/p) E_{t-1}^{l_1,l_2} + Ẽ_t^{l_1,l_2}
//   B-side: Et^{B_x}_t = (β/p) E_{t-1}^{l_1,l_2} − Ẽ_t^{l_1,l_2}
//   C-side (aux): R-buffer bump (t → t+1 etc.) — analogous to point-nucleus
// with Ẽ_t = Et_grad_NonRecursion(l_1, l_2, t, α, β, D). The aux-side Et's
// E_τ^{l_3,0}(γ, 0, 0) are coordinate-independent, identical to the 2c2e case.
//
// Translational invariance (∂A + ∂B + ∂C = 0) is NOT used to generate any of
// the 9 components — every direction is computed directly. This avoids the
// 2c-vs-3c-P-center cancellation pathology that broke the previous attempt
// (see §2.6, §7 of RI_Gradient.md).
// ============================================================================
__host__ __device__ static inline
void grad_3c2e_triple_body(
    double* g_gradients,
    const real_t* g_gamma3,
    const PrimitiveShell& a,    // μ AO primitive
    const PrimitiveShell& b,    // ν AO primitive
    const PrimitiveShell& c,    // P aux primitive
    const real_t* g_cgto_normalization_factors,
    const real_t* g_auxiliary_cgto_normalization_factors,
    const int num_basis,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid)
{
    const size_t base_mu  = a.basis_index;
    const size_t base_nu  = b.basis_index;
    const size_t base_aux = c.basis_index;

    const double alpha = a.exponent;
    const double beta  = b.exponent;
    const double gamma = c.exponent;
    const double p     = alpha + beta;
    const double xi    = p * gamma / (p + gamma);

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

    const double3 Pprod = make_double3(
        (alpha * a.coordinate.x + beta * b.coordinate.x) / p,
        (alpha * a.coordinate.y + beta * b.coordinate.y) / p,
        (alpha * a.coordinate.z + beta * b.coordinate.z) / p);

    const double RPCx = Pprod.x - c.coordinate.x;
    const double RPCy = Pprod.y - c.coordinate.y;
    const double RPCz = Pprod.z - c.coordinate.z;
    const double dist2 = RPCx*RPCx + RPCy*RPCy + RPCz*RPCz;

    // Total order for R buffer. Integral needs L_μ+L_ν+L_P; derivative bumps by one.
    const int K = a.shell_type + b.shell_type + c.shell_type + 1;

    double Boys[boys_size];
    getIncrementalBoys(K, xi * dist2, g_boys_grid, Boys);
    for (int x = 0; x <= K; x++) {
        Boys[x] *= right2left_binary_woif(-2.0 * xi, x);
    }

    double R[size_R];
    double R_mid[3 * size_Rmid];
    compute_R_TripleBuffer(R, R_mid, Boys, Pprod, c.coordinate, K, K, K, K);

    // Pre-multiplied prefactor (Et / Norm / cgto are applied per-component below)
    const double pref = a.coefficient * b.coefficient * c.coefficient *
                        TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                        (p * gamma * sqrt(p + gamma));

    double grad_Ax_acc = 0.0, grad_Ay_acc = 0.0, grad_Az_acc = 0.0;
    double grad_Bx_acc = 0.0, grad_By_acc = 0.0, grad_Bz_acc = 0.0;
    double grad_Cx_acc = 0.0, grad_Cy_acc = 0.0, grad_Cz_acc = 0.0;

    const int L_a = a.shell_type;
    const int L_b = b.shell_type;
    const int L_c = c.shell_type;

    for (int lmn_a = 0; lmn_a < comb_max(L_a); lmn_a++) {
#ifdef __CUDA_ARCH__
        const int l1 = loop_to_ang[L_a][lmn_a][0];
        const int m1 = loop_to_ang[L_a][lmn_a][1];
        const int n1 = loop_to_ang[L_a][lmn_a][2];
#else
        const int l1 = loop_to_ang_host[L_a][lmn_a][0];
        const int m1 = loop_to_ang_host[L_a][lmn_a][1];
        const int n1 = loop_to_ang_host[L_a][lmn_a][2];
#endif
        const double Norm_A = calcNorm(alpha, l1, m1, n1);

        for (int lmn_b = 0; lmn_b < comb_max(L_b); lmn_b++) {
#ifdef __CUDA_ARCH__
            const int l2 = loop_to_ang[L_b][lmn_b][0];
            const int m2 = loop_to_ang[L_b][lmn_b][1];
            const int n2 = loop_to_ang[L_b][lmn_b][2];
#else
            const int l2 = loop_to_ang_host[L_b][lmn_b][0];
            const int m2 = loop_to_ang_host[L_b][lmn_b][1];
            const int n2 = loop_to_ang_host[L_b][lmn_b][2];
#endif
            const double Norm_B = calcNorm(beta, l2, m2, n2);

            const double cgto_munu = g_cgto_normalization_factors[base_mu + lmn_a] *
                                     g_cgto_normalization_factors[base_nu + lmn_b];

            for (int lmn_c = 0; lmn_c < comb_max(L_c); lmn_c++) {
#ifdef __CUDA_ARCH__
                const int l3 = loop_to_ang[L_c][lmn_c][0];
                const int m3 = loop_to_ang[L_c][lmn_c][1];
                const int n3 = loop_to_ang[L_c][lmn_c][2];
#else
                const int l3 = loop_to_ang_host[L_c][lmn_c][0];
                const int m3 = loop_to_ang_host[L_c][lmn_c][1];
                const int n3 = loop_to_ang_host[L_c][lmn_c][2];
#endif
                const double Norm_C = calcNorm(gamma, l3, m3, n3);

                // Γ^(3) row-major (P_aux, μ, ν)
                const size_t gamma3_idx = ((size_t)(base_aux + lmn_c) * (size_t)num_basis +
                                           (size_t)(base_mu  + lmn_a)) * (size_t)num_basis +
                                          (size_t)(base_nu  + lmn_b);
                const double gamma3 = g_gamma3[gamma3_idx];
                if (fabs(gamma3) < 1.0e-15) continue;

                const double w = pref * Norm_A * Norm_B * Norm_C * cgto_munu *
                                 g_auxiliary_cgto_normalization_factors[base_aux + lmn_c] *
                                 gamma3;

                // -------------------- 9 derivative components --------------------
                //
                // For ∂R_A,x (= ∂R_μ,x): replace Et^{l1,l2}(t, Dx) with
                //   Et_A_x = (α/p)·E_{t-1}^{l1,l2}(Dx) + Ẽ_t^{l1,l2}(Dx),
                //   loop t up to l1+l2+1 instead of l1+l2.
                // Similarly Et_A_y on u, Et_A_z on v. Same for B-side with (β/p) and -Ẽ.
                //
                // For ∂R_C,x (= ∂R_aux,x): keep all Et's normal but bump R index by 1
                //   in x (k=t+τ+1, u=u+ν, v=v+φ). And analogously for y, z.
                //
                // Aux-side Et's E_τ^{l3,0}(γ,0,0) are coordinate-independent.
                // -----------------------------------------------------------------

                // Sum over 6-fold (t, u, v, τ, ν, φ)
                double dAx = 0.0, dAy = 0.0, dAz = 0.0;
                double dBx = 0.0, dBy = 0.0, dBz = 0.0;
                double dCx = 0.0, dCy = 0.0, dCz = 0.0;

                // For ∂A_x / ∂B_x the t index runs to l1+l2+1; for ∂A_y / ∂B_y u runs
                // to m1+m2+1, etc. The other derivative axes use the integral's range.
                // To keep one loop, we use the MAX range and zero out terms when
                // an Et out of bounds returns 0 (MD_Et_NonRecursion / Et_grad return
                // 0 for t<0 or t > l1+l2 in their respective conventions).
                //
                // Practical bound: t ≤ l1+l2+1, u ≤ m1+m2+1, v ≤ n1+n2+1 (all bumped).

                for (int t = 0; t < l1 + l2 + 2; t++) {
                    const double Et_int  = MD_Et_NonRecursion(l1, l2, t,     alpha, beta, Dx);
                    const double Et_prev = MD_Et_NonRecursion(l1, l2, t - 1, alpha, beta, Dx);
                    const double Et_grd  = Et_grad_NonRecursion(l1, l2, t,   alpha, beta, Dx);
                    // Et for axis-x derivative of A and B (l1+l2+1 range)
                    const double Et_Ax = (alpha / p) * Et_prev + Et_grd;
                    const double Et_Bx = (beta  / p) * Et_prev - Et_grd;

                    for (int u = 0; u < m1 + m2 + 2; u++) {
                        const double Eu_int  = MD_Et_NonRecursion(m1, m2, u,     alpha, beta, Dy);
                        const double Eu_prev = MD_Et_NonRecursion(m1, m2, u - 1, alpha, beta, Dy);
                        const double Eu_grd  = Et_grad_NonRecursion(m1, m2, u,   alpha, beta, Dy);
                        const double Eu_Ay = (alpha / p) * Eu_prev + Eu_grd;
                        const double Eu_By = (beta  / p) * Eu_prev - Eu_grd;

                        for (int v = 0; v < n1 + n2 + 2; v++) {
                            const double Ev_int  = MD_Et_NonRecursion(n1, n2, v,     alpha, beta, Dz);
                            const double Ev_prev = MD_Et_NonRecursion(n1, n2, v - 1, alpha, beta, Dz);
                            const double Ev_grd  = Et_grad_NonRecursion(n1, n2, v,   alpha, beta, Dz);
                            const double Ev_Az = (alpha / p) * Ev_prev + Ev_grd;
                            const double Ev_Bz = (beta  / p) * Ev_prev - Ev_grd;

                            // Aux-side τ, ν, φ loops (3 nested)
                            for (int tau = 0; tau < l3 + 1; tau++) {
                                const double Etau = MD_Et_NonRecursion(l3, 0, tau, gamma, 0.0, 0.0);
                                for (int nu = 0; nu < m3 + 1; nu++) {
                                    const double Enu = MD_Et_NonRecursion(m3, 0, nu, gamma, 0.0, 0.0);
                                    for (int phi = 0; phi < n3 + 1; phi++) {
                                        const double Ephi = MD_Et_NonRecursion(n3, 0, phi, gamma, 0.0, 0.0);
                                        const double aux_factor = Etau * Enu * Ephi *
                                                                  (1 - 2 * ((tau + nu + phi) & 1));

                                        const int ti = t + tau;
                                        const int ui = u + nu;
                                        const int vi = v + phi;

                                        // ---- ∂A_x ----  t-axis A-side Et, normal Et on u/v
                                        if (t < l1 + l2 + 2 && u < m1 + m2 + 1 && v < n1 + n2 + 1) {
                                            const int k = ti + ui + vi;
                                            const double rval = R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui, vi, 0, 0, 0)];
                                            dAx += Et_Ax * Eu_int * Ev_int * aux_factor * rval;
                                            dBx += Et_Bx * Eu_int * Ev_int * aux_factor * rval;
                                        }

                                        // ---- ∂A_y ----  normal Et on t/v, u-axis A-side Et
                                        if (t < l1 + l2 + 1 && u < m1 + m2 + 2 && v < n1 + n2 + 1) {
                                            const int k = ti + ui + vi;
                                            const double rval = R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui, vi, 0, 0, 0)];
                                            dAy += Et_int * Eu_Ay * Ev_int * aux_factor * rval;
                                            dBy += Et_int * Eu_By * Ev_int * aux_factor * rval;
                                        }

                                        // ---- ∂A_z ----  normal Et on t/u, v-axis A-side Et
                                        if (t < l1 + l2 + 1 && u < m1 + m2 + 1 && v < n1 + n2 + 2) {
                                            const int k = ti + ui + vi;
                                            const double rval = R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui, vi, 0, 0, 0)];
                                            dAz += Et_int * Eu_int * Ev_Az * aux_factor * rval;
                                            dBz += Et_int * Eu_int * Ev_Bz * aux_factor * rval;
                                        }

                                        // ---- ∂C_x ----  normal Et all axes, R index ti+1
                                        if (t < l1 + l2 + 1 && u < m1 + m2 + 1 && v < n1 + n2 + 1) {
                                            const int kc = (ti + 1) + ui + vi;
                                            const double rv_x = R[kc*(kc+1)*(kc+2)/6 + calc_Idx_Rmid(kc, ui, vi, 0, 0, 0)];
                                            dCx += Et_int * Eu_int * Ev_int * aux_factor * rv_x;

                                            // ---- ∂C_y ----  R index ui+1
                                            const int ky = ti + (ui + 1) + vi;
                                            const double rv_y = R[ky*(ky+1)*(ky+2)/6 + calc_Idx_Rmid(ky, ui+1, vi, 0, 0, 0)];
                                            dCy += Et_int * Eu_int * Ev_int * aux_factor * rv_y;

                                            // ---- ∂C_z ----  R index vi+1
                                            const int kz = ti + ui + (vi + 1);
                                            const double rv_z = R[kz*(kz+1)*(kz+2)/6 + calc_Idx_Rmid(kz, ui, vi+1, 0, 0, 0)];
                                            dCz += Et_int * Eu_int * Ev_int * aux_factor * rv_z;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                grad_Ax_acc += w * dAx;
                grad_Ay_acc += w * dAy;
                grad_Az_acc += w * dAz;
                grad_Bx_acc += w * dBx;
                grad_By_acc += w * dBy;
                grad_Bz_acc += w * dBz;
                grad_Cx_acc += w * dCx;
                grad_Cy_acc += w * dCy;
                grad_Cz_acc += w * dCz;
            }
        }
    }

    // Accumulate to gradient. Sign convention:
    //   - A/B helpers return +∂(μν|P)/∂R_μ (and ν), so accumulate with +.
    //   - C-helper R-bump returns "+R^{(t+1)}" which corresponds to
    //     −∂(μν|P)/∂R_aux (chain rule of (R_P_prod − R_aux) on the buffer);
    //     so accumulate the aux atom with a flipped sign.
    const int aA = a.atom_index;
    const int aB = b.atom_index;
    const int aC = c.atom_index;
    gansu_atomic_add(&g_gradients[3*aA + 0],  grad_Ax_acc);
    gansu_atomic_add(&g_gradients[3*aA + 1],  grad_Ay_acc);
    gansu_atomic_add(&g_gradients[3*aA + 2],  grad_Az_acc);
    gansu_atomic_add(&g_gradients[3*aB + 0],  grad_Bx_acc);
    gansu_atomic_add(&g_gradients[3*aB + 1],  grad_By_acc);
    gansu_atomic_add(&g_gradients[3*aB + 2],  grad_Bz_acc);
    gansu_atomic_add(&g_gradients[3*aC + 0], -grad_Cx_acc);
    gansu_atomic_add(&g_gradients[3*aC + 1], -grad_Cy_acc);
    gansu_atomic_add(&g_gradients[3*aC + 2], -grad_Cz_acc);
}

// ----------------------------------------------------------------------------
// GPU kernel
// ----------------------------------------------------------------------------
__global__
void compute_gradients_3c2e(double* g_gradients, const real_t* g_gamma3,
                            const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux,
                            const real_t* g_cgto_normalization_factors,
                            const real_t* g_auxiliary_cgto_normalization_factors,
                            ShellTypeInfo shell_s_mu, ShellTypeInfo shell_s_nu, ShellTypeInfo shell_s_P,
                            const size_t num_threads, const int num_basis,
                            const int num_auxiliary_basis, const real_t* g_boys_grid)
{
    const size_t id = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (id >= num_threads) return;

    // Decompose id into (lmn_μ, lmn_ν, lmn_P) primitive indices.
    // Layout: P (innermost) × ν × μ (outermost). num_threads = n_μ × n_ν × n_P.
    const size_t n_P  = shell_s_P.count;
    const size_t n_nu = shell_s_nu.count;

    const size_t p_loc  = id % n_P;
    const size_t nu_loc = (id / n_P) % n_nu;
    const size_t mu_loc = (id / (n_P * n_nu));

    const size_t prim_mu  = mu_loc + shell_s_mu.start_index;
    const size_t prim_nu  = nu_loc + shell_s_nu.start_index;
    const size_t prim_P   = p_loc  + shell_s_P.start_index;

    const PrimitiveShell a = g_pshell[prim_mu];
    const PrimitiveShell b = g_pshell[prim_nu];
    const PrimitiveShell c = g_pshell_aux[prim_P];

    grad_3c2e_triple_body(g_gradients, g_gamma3, a, b, c,
                          g_cgto_normalization_factors,
                          g_auxiliary_cgto_normalization_factors,
                          num_basis, num_auxiliary_basis, g_boys_grid);
}

// ----------------------------------------------------------------------------
// CPU mirror (OpenMP)
// ----------------------------------------------------------------------------
void compute_gradients_3c2e_cpu(double* g_gradients, const real_t* g_gamma3,
                                 const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux,
                                 const real_t* g_cgto_normalization_factors,
                                 const real_t* g_auxiliary_cgto_normalization_factors,
                                 ShellTypeInfo shell_s_mu, ShellTypeInfo shell_s_nu, ShellTypeInfo shell_s_P,
                                 const size_t num_threads, const int num_basis,
                                 const int num_auxiliary_basis, const real_t* g_boys_grid)
{
    #pragma omp parallel for schedule(static)
    for (long long id_ll = 0; id_ll < (long long)num_threads; id_ll++) {
        const size_t id = (size_t)id_ll;
        const size_t n_P  = shell_s_P.count;
        const size_t n_nu = shell_s_nu.count;

        const size_t p_loc  = id % n_P;
        const size_t nu_loc = (id / n_P) % n_nu;
        const size_t mu_loc = (id / (n_P * n_nu));

        const size_t prim_mu  = mu_loc + shell_s_mu.start_index;
        const size_t prim_nu  = nu_loc + shell_s_nu.start_index;
        const size_t prim_P   = p_loc  + shell_s_P.start_index;

        const PrimitiveShell a = g_pshell[prim_mu];
        const PrimitiveShell b = g_pshell[prim_nu];
        const PrimitiveShell c = g_pshell_aux[prim_P];

        grad_3c2e_triple_body(g_gradients, g_gamma3, a, b, c,
                              g_cgto_normalization_factors,
                              g_auxiliary_cgto_normalization_factors,
                              num_basis, num_auxiliary_basis, g_boys_grid);
    }
}

} // namespace gansu::gpu
