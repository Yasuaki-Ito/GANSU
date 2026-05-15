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
#include "grad_2c.hpp"

namespace gansu::gpu {

// ----------------------------------------------------------------------------
// 2c2e (P|Q) derivative — single primitive shell pair body
//
// The McMurchie-Davidson form of (P|Q) is
//   (P|Q) = c_P c_Q · 2π^{5/2} / (αβ √(α+β))
//          · Σ_{tuvτνφ}  E_t^{l1,0}(α) E_u^{m1,0}(α) E_v^{n1,0}(α)
//                       · E_τ^{l2,0}(β) E_ν^{m2,0}(β) E_φ^{n2,0}(β)
//                       · (−1)^{τ+ν+φ}
//                       · R^{(0)}_{t+τ, u+ν, v+φ}(ξ, R_P−R_Q)
//
// Because E_t coefficients for a single-center expansion (D=0) are independent
// of nuclear coordinates, the derivative wrt R_{P,x} is obtained by bumping
// the t-index of R by one:
//   ∂(P|Q)/∂R_{P,x} = (same sum, with R^{(0)}_{(t+τ+1), u+ν, v+φ})
// and analogously for y, z. For 2c, translational invariance gives the exact
// (well-conditioned) identity ∂(P|Q)/∂R_Q = −∂(P|Q)/∂R_P.
// ----------------------------------------------------------------------------
__host__ __device__
static inline void grad_2c2e_pair_body(
    double* g_gradients,
    const real_t* g_gamma2_aux,
    const PrimitiveShell& P,
    const PrimitiveShell& Q,
    const size_t prim_P_idx,
    const size_t prim_Q_idx,
    const real_t* g_auxiliary_cgto_normalization_factors,
    const int num_auxiliary_basis,
    const real_t* g_boys_grid,
    const bool no_pair_symmetry)
{
    // Same-atom pairs contribute exactly zero (translational invariance for 2c)
    if (P.atom_index == Q.atom_index) return;

    const size_t base_P = P.basis_index;
    const size_t base_Q = Q.basis_index;

    const double alpha = P.exponent;
    const double beta  = Q.exponent;
    const double xi    = alpha * beta / (alpha + beta);

    const double3 cP = make_double3(P.coordinate.x, P.coordinate.y, P.coordinate.z);
    // The 1-electron compute_R_TripleBuffer overload takes a Coordinate for the
    // second center; PrimitiveShell::coordinate is already Coordinate.

    const double dRx = P.coordinate.x - Q.coordinate.x;
    const double dRy = P.coordinate.y - Q.coordinate.y;
    const double dRz = P.coordinate.z - Q.coordinate.z;
    const double dist2 = dRx*dRx + dRy*dRy + dRz*dRz;

    // K = L_P + L_Q + 1 because we need R^{(0)} up to total order one higher
    // than the integral itself (for the derivative bump).
    const int K = P.shell_type + Q.shell_type + 1;

    double Boys[boys_one_size];
    getIncrementalBoys(K, xi * dist2, g_boys_grid, Boys);
    for (int x = 0; x <= K; x++) {
        Boys[x] *= right2left_binary_woif(-2.0 * xi, x);
    }

    double R[size_one_R];
    double R_mid[3 * size_one_Rmid];
    compute_R_TripleBuffer(R, R_mid, Boys, cP, Q.coordinate, K, K, K, K);

    // Common prefactor (Et, per-component norm, cgto_norm are applied below)
    const double pref = P.coefficient * Q.coefficient *
                        TWO_TIMES_PI_TO_THE_2_POINT_5_TH_POWER /
                        (alpha * beta * sqrt(alpha + beta));

    // Off-diagonal primitive pairs cover both (P,Q) and (Q,P) entries of Γ^(2)
    // because we iterate the upper triangle when shell_sP == shell_sQ.
    // When `no_pair_symmetry` is set (distributed all-pair iteration), each
    // ordered pair is visited exactly once across all callers; we use factor 1.
    const double sym_factor = no_pair_symmetry
        ? 1.0
        : ((prim_P_idx != prim_Q_idx) ? 2.0 : 1.0);

    double grad_Px = 0.0, grad_Py = 0.0, grad_Pz = 0.0;

    const int LP = P.shell_type;
    const int LQ = Q.shell_type;

    for (int lmn_P = 0; lmn_P < comb_max(LP); lmn_P++) {
#ifdef __CUDA_ARCH__
        const int l1 = loop_to_ang[LP][lmn_P][0];
        const int m1 = loop_to_ang[LP][lmn_P][1];
        const int n1 = loop_to_ang[LP][lmn_P][2];
#else
        const int l1 = loop_to_ang_host[LP][lmn_P][0];
        const int m1 = loop_to_ang_host[LP][lmn_P][1];
        const int n1 = loop_to_ang_host[LP][lmn_P][2];
#endif
        const double NormP = calcNorm(alpha, l1, m1, n1);

        for (int lmn_Q = 0; lmn_Q < comb_max(LQ); lmn_Q++) {
#ifdef __CUDA_ARCH__
            const int l2 = loop_to_ang[LQ][lmn_Q][0];
            const int m2 = loop_to_ang[LQ][lmn_Q][1];
            const int n2 = loop_to_ang[LQ][lmn_Q][2];
#else
            const int l2 = loop_to_ang_host[LQ][lmn_Q][0];
            const int m2 = loop_to_ang_host[LQ][lmn_Q][1];
            const int n2 = loop_to_ang_host[LQ][lmn_Q][2];
#endif
            const double NormQ = calcNorm(beta, l2, m2, n2);

            // Γ^(2) is row-major and symmetric. We read (base_P+lmn_P, base_Q+lmn_Q);
            // the (Q,P) symmetry partner is folded in via sym_factor for prim_P≠prim_Q.
            const double gamma2 = g_gamma2_aux[(base_P + lmn_P) * num_auxiliary_basis + (base_Q + lmn_Q)];
            if (fabs(gamma2) < 1.0e-15) continue;

            const double w = pref * NormP * NormQ * sym_factor *
                             g_auxiliary_cgto_normalization_factors[base_P + lmn_P] *
                             g_auxiliary_cgto_normalization_factors[base_Q + lmn_Q] *
                             gamma2;

            // ∂(P|Q)/∂R_{P,axis} contracted with Et coefficients.
            // axis ∈ {x, y, z} corresponds to bumping the (t+τ), (u+ν), (v+φ)
            // index of R^{(0)} by 1 respectively.
            double dPx = 0.0, dPy = 0.0, dPz = 0.0;
            for (int t = 0; t <= l1; t++) {
                const double Et = MD_Et_NonRecursion(l1, 0, t, alpha, 0.0, 0.0);
                for (int u = 0; u <= m1; u++) {
                    const double Eu = MD_Et_NonRecursion(m1, 0, u, alpha, 0.0, 0.0);
                    for (int v = 0; v <= n1; v++) {
                        const double Ev = MD_Et_NonRecursion(n1, 0, v, alpha, 0.0, 0.0);
                        for (int tau = 0; tau <= l2; tau++) {
                            const double Etau = MD_Et_NonRecursion(l2, 0, tau, beta, 0.0, 0.0);
                            for (int nu = 0; nu <= m2; nu++) {
                                const double Enu = MD_Et_NonRecursion(m2, 0, nu, beta, 0.0, 0.0);
                                for (int phi = 0; phi <= n2; phi++) {
                                    const double Ephi = MD_Et_NonRecursion(n2, 0, phi, beta, 0.0, 0.0);
                                    const double sign = (1 - 2 * ((tau + nu + phi) & 1));
                                    const double EE = Et * Eu * Ev * Etau * Enu * Ephi * sign;

                                    const int ti = t + tau;
                                    const int ui = u + nu;
                                    const int vi = v + phi;

                                    // ∂/∂R_{P,x}: R^{(0)}_{ti+1, ui, vi}
                                    {
                                        const int k = (ti + 1) + ui + vi;
                                        const int idx = k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui, vi, 0, 0, 0);
                                        dPx += EE * R[idx];
                                    }
                                    // ∂/∂R_{P,y}: R^{(0)}_{ti, ui+1, vi}
                                    {
                                        const int k = ti + (ui + 1) + vi;
                                        const int idx = k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui+1, vi, 0, 0, 0);
                                        dPy += EE * R[idx];
                                    }
                                    // ∂/∂R_{P,z}: R^{(0)}_{ti, ui, vi+1}
                                    {
                                        const int k = ti + ui + (vi + 1);
                                        const int idx = k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k, ui, vi+1, 0, 0, 0);
                                        dPz += EE * R[idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }

            grad_Px += w * dPx;
            grad_Py += w * dPy;
            grad_Pz += w * dPz;
        }
    }

    // Translational invariance for 2c: ∂(P|Q)/∂R_Q = −∂(P|Q)/∂R_P (exact for 2c).
    const int aP = P.atom_index;
    const int aQ = Q.atom_index;
    gansu_atomic_add(&g_gradients[3*aP + 0],  grad_Px);
    gansu_atomic_add(&g_gradients[3*aP + 1],  grad_Py);
    gansu_atomic_add(&g_gradients[3*aP + 2],  grad_Pz);
    gansu_atomic_add(&g_gradients[3*aQ + 0], -grad_Px);
    gansu_atomic_add(&g_gradients[3*aQ + 1], -grad_Py);
    gansu_atomic_add(&g_gradients[3*aQ + 2], -grad_Pz);
}

// ----------------------------------------------------------------------------
// GPU kernel
// ----------------------------------------------------------------------------
__global__
void compute_gradients_2c2e(double* g_gradients, const real_t* g_gamma2_aux,
                            const PrimitiveShell* g_pshell_aux,
                            const real_t* g_auxiliary_cgto_normalization_factors,
                            ShellTypeInfo shell_sP, ShellTypeInfo shell_sQ,
                            const size_t num_threads, const int num_auxiliary_basis,
                            const real_t* g_boys_grid,
                            const bool no_pair_symmetry)
{
    const size_t id = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (id >= num_threads) return;

    const bool same_shell = (!no_pair_symmetry) &&
                            (shell_sP.start_index == shell_sQ.start_index);
    const size_t2 ab = same_shell ? index1to2(id, true)
                                  : index1to2(id, false, shell_sQ.count);
    const size_t prim_P_idx = ab.x + shell_sP.start_index;
    const size_t prim_Q_idx = ab.y + shell_sQ.start_index;

    const PrimitiveShell P = g_pshell_aux[prim_P_idx];
    const PrimitiveShell Q = g_pshell_aux[prim_Q_idx];

    grad_2c2e_pair_body(g_gradients, g_gamma2_aux, P, Q,
                        prim_P_idx, prim_Q_idx,
                        g_auxiliary_cgto_normalization_factors,
                        num_auxiliary_basis, g_boys_grid,
                        no_pair_symmetry);
}

// ----------------------------------------------------------------------------
// CPU mirror (OpenMP)
// ----------------------------------------------------------------------------
void compute_gradients_2c2e_cpu(double* g_gradients, const real_t* g_gamma2_aux,
                                 const PrimitiveShell* g_pshell_aux,
                                 const real_t* g_auxiliary_cgto_normalization_factors,
                                 ShellTypeInfo shell_sP, ShellTypeInfo shell_sQ,
                                 const size_t num_threads, const int num_auxiliary_basis,
                                 const real_t* g_boys_grid,
                                 const bool no_pair_symmetry)
{
    #pragma omp parallel for schedule(static)
    for (long long id_ll = 0; id_ll < (long long)num_threads; id_ll++) {
        const size_t id = (size_t)id_ll;

        const bool same_shell = (!no_pair_symmetry) &&
                                (shell_sP.start_index == shell_sQ.start_index);
        const size_t2 ab = same_shell ? index1to2(id, true)
                                      : index1to2(id, false, shell_sQ.count);
        const size_t prim_P_idx = ab.x + shell_sP.start_index;
        const size_t prim_Q_idx = ab.y + shell_sQ.start_index;

        const PrimitiveShell P = g_pshell_aux[prim_P_idx];
        const PrimitiveShell Q = g_pshell_aux[prim_Q_idx];

        grad_2c2e_pair_body(g_gradients, g_gamma2_aux, P, Q,
                            prim_P_idx, prim_Q_idx,
                            g_auxiliary_cgto_normalization_factors,
                            num_auxiliary_basis, g_boys_grid,
                            no_pair_symmetry);
    }
}

} // namespace gansu::gpu
