/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RYS_ERI_SPECIALIZED_HPP
#define RYS_ERI_SPECIALIZED_HPP

// ============================================================
//  Template-specialized Rys ERI kernels
//
//  Template parameters LA, LB, LC, LD are shell angular momenta
//  (0=s, 1=p, 2=d, 3=f, 4=g). All loop bounds become compile-time
//  constants, enabling full unrolling and optimal register allocation.
// ============================================================

// NOTE: This header must be included from int2e.hpp AFTER all function definitions.
// It uses: comb_max, loop_to_ang, calcNorm, addToResult_case1, M_PI_2_5, eri_kernel_t
// from int2e.hpp, and computeRysRootsAndWeights from rys_quadrature.hpp.
#include "rys_quadrature.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu {

// Compile-time comb_max (macro to avoid constexpr __host__ restriction in __global__)
#define COMB_MAX_CT(l) (((l)+1)*((l)+2)/2)

// ============================================================
//  VRR with compile-time dimensions: fully unrollable
// ============================================================
template<int A_MAX, int C_MAX>
inline __device__
void vrr_1d_ct(double C00, double D00, double B10, double B01, double B00,
               double (&I)[(A_MAX+1)*(C_MAX+1)]) {
    constexpr int CS = C_MAX + 1;
    I[0] = 1.0;

    // Build I(a, 0)
    if constexpr (A_MAX > 0) {
        I[CS] = C00;
        #pragma unroll
        for (int a = 1; a < A_MAX; a++)
            I[(a+1)*CS] = C00 * I[a*CS] + a * B10 * I[(a-1)*CS];
    }

    // Build I(a, c+1)
    #pragma unroll
    for (int c = 0; c < C_MAX; c++) {
        const double cB01 = c * B01;
        I[c+1] = D00 * I[c] + ((c > 0) ? cB01 * I[c-1] : 0.0);
        #pragma unroll
        for (int a = 1; a <= A_MAX; a++)
            I[a*CS + c+1] = D00 * I[a*CS + c]
                          + ((c > 0) ? cB01 * I[a*CS + c-1] : 0.0)
                          + a * B00 * I[(a-1)*CS + c];
    }
}

// ============================================================
//  Bra TRR with compile-time b: binomial expansion, fully unrolled
// ============================================================
template<int C_MAX>
inline __device__
double bra_trr_ct(int a, int b, int ct, double AB, const double* vrr, int cs) {
    // Binomial expansion: I(a,b,ct) = Σ C(b,k) * AB^k * I_vrr(a+b-k, ct)
    // For small b (0..4), this unrolls to a few FMAs
    double val = 0.0;
    int binom = 1;
    double AB_pow = 1.0;
    for (int k = 0; k <= b; k++) {
        val += binom * AB_pow * vrr[(a + b - k) * cs + ct];
        if (k < b) { AB_pow *= AB; binom = binom * (b - k) / (k + 1); }
    }
    return val;
}

// ============================================================
//  Ket TRR with compile-time d: iterative reduction
// ============================================================
template<int D_MAX>
inline __device__
double ket_trr_ct(int d, double CD, const double* vals) {
    double buf[D_MAX + 1];
    #pragma unroll
    for (int i = 0; i <= d; i++) buf[i] = vals[i];
    for (int dd = 0; dd < d; dd++)
        for (int i = 0; i <= d - dd - 1; i++)
            buf[i] = buf[i+1] + CD * buf[i];
    return buf[0];
}

// ============================================================
//  Template-specialized RysERI kernel
// ============================================================
template<int LA, int LB, int LC, int LD>
__global__
void RysERI_T(
    double* g_int2e,
    const PrimitiveShell* g_shell,
    const real_t* g_cgto_normalization_factors,
    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
    const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3,
    const size_t num_threads,
    const real_t schwarz_screening_threshold,
    const double* g_upper_bound_factors,
    const int num_basis,
    const double* g_boys_grid,
    const size_t head_bra, const size_t head_ket)
{
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_threads) return;

    // Compile-time constants
    constexpr int L = LA + LB + LC + LD;
    constexpr int N_ROOTS = L / 2 + 1;
    constexpr int A_MAX = LA + LB;
    constexpr int C_MAX = LC + LD;
    constexpr int CS = C_MAX + 1;
    constexpr int NA = COMB_MAX_CT(LA), NB = COMB_MAX_CT(LB);
    constexpr int NC = COMB_MAX_CT(LC), ND = COMB_MAX_CT(LD);
    constexpr int N_COMP = NA * NB * NC * ND;

    // --- Thread → shell quartet mapping ---
    int ket_size;
    if (shell_s2.start_index == shell_s3.start_index)
        ket_size = (shell_s2.count * (shell_s2.count + 1)) / 2;
    else
        ket_size = shell_s2.count * shell_s3.count;

    const size_t2 abcd = index1to2(id,
        (shell_s0.start_index == shell_s2.start_index &&
         shell_s1.start_index == shell_s3.start_index), ket_size);
    const size_t2 ab = index1to2(abcd.x,
        shell_s0.start_index == shell_s1.start_index, shell_s1.count);
    const size_t2 cd = index1to2(abcd.y,
        shell_s2.start_index == shell_s3.start_index, shell_s3.count);

    if (g_upper_bound_factors[head_bra + abcd.x] *
        g_upper_bound_factors[head_ket + abcd.y] < schwarz_screening_threshold)
        return;

    const size_t pidx_a = ab.x + shell_s0.start_index;
    const size_t pidx_b = ab.y + shell_s1.start_index;
    const size_t pidx_c = cd.x + shell_s2.start_index;
    const size_t pidx_d = cd.y + shell_s3.start_index;

    const PrimitiveShell sa = g_shell[pidx_a];
    const PrimitiveShell sb = g_shell[pidx_b];
    const PrimitiveShell sc = g_shell[pidx_c];
    const PrimitiveShell sd = g_shell[pidx_d];

    const size_t base_a = sa.basis_index, base_b = sb.basis_index;
    const size_t base_c = sc.basis_index, base_d = sd.basis_index;

    const bool sym_bra    = (pidx_a == pidx_b);
    const bool sym_ket    = (pidx_c == pidx_d);
    const bool sym_braket = (pidx_a == pidx_c && pidx_b == pidx_d);

    const double alpha = sa.exponent, beta = sb.exponent;
    const double gamma = sc.exponent, delta_exp = sd.exponent;
    const double p = alpha + beta, q = gamma + delta_exp;
    const double rho = p * q / (p + q);

    const double Ax = sa.coordinate.x, Ay = sa.coordinate.y, Az = sa.coordinate.z;
    const double Bx = sb.coordinate.x, By = sb.coordinate.y, Bz = sb.coordinate.z;
    const double Cx = sc.coordinate.x, Cy = sc.coordinate.y, Cz = sc.coordinate.z;
    const double Dx = sd.coordinate.x, Dy = sd.coordinate.y, Dz = sd.coordinate.z;

    const double Px = (alpha*Ax + beta*Bx)/p, Py = (alpha*Ay + beta*By)/p, Pz = (alpha*Az + beta*Bz)/p;
    const double Qx = (gamma*Cx + delta_exp*Dx)/q, Qy = (gamma*Cy + delta_exp*Dy)/q, Qz = (gamma*Cz + delta_exp*Dz)/q;

    const double PQx = Px-Qx, PQy = Py-Qy, PQz = Pz-Qz;
    const double T = rho * (PQx*PQx + PQy*PQy + PQz*PQz);

    const double AB2 = (Ax-Bx)*(Ax-Bx) + (Ay-By)*(Ay-By) + (Az-Bz)*(Az-Bz);
    const double CD2 = (Cx-Dx)*(Cx-Dx) + (Cy-Dy)*(Cy-Dy) + (Cz-Dz)*(Cz-Dz);
    const double prefactor = 2.0 * M_PI_2_5 / (p*q*sqrt(p+q))
                           * exp(-alpha*beta/p*AB2 - gamma*delta_exp/q*CD2)
                           * sa.coefficient * sb.coefficient * sc.coefficient * sd.coefficient;

    // Rys roots/weights with compile-time N
    double rys_roots[N_ROOTS], rys_weights[N_ROOTS];
    computeRysRootsAndWeights(N_ROOTS, T, g_boys_grid, rys_roots, rys_weights);

    const double ABx = Ax-Bx, ABy = Ay-By, ABz = Az-Bz;
    const double CDx = Cx-Dx, CDy = Cy-Dy, CDz = Cz-Dz;

    // Compile-time sized VRR workspace
    double Ix[(A_MAX+1)*(C_MAX+1)], Iy[(A_MAX+1)*(C_MAX+1)], Iz[(A_MAX+1)*(C_MAX+1)];

    // Accumulation buffer (compile-time size)
    double eri_buf[N_COMP];
    #pragma unroll
    for (int i = 0; i < N_COMP; i++) eri_buf[i] = 0.0;

    // --- Root loop ---
    #pragma unroll
    for (int n = 0; n < N_ROOTS; n++) {
        const double t2 = rys_roots[n];
        const double wn = rys_weights[n];
        const double u = rho * t2;
        const double u_over_p = u / p, u_over_q = u / q;

        const double B00 = t2 / (2.0*(p+q));
        const double B10 = (1.0 - u_over_p) / (2.0*p);
        const double B01 = (1.0 - u_over_q) / (2.0*q);

        // VRR with compile-time dimensions
        vrr_1d_ct<A_MAX, C_MAX>(
            (Px-Ax) + u_over_p*(Qx-Px), (Qx-Cx) + u_over_q*(Px-Qx),
            B10, B01, B00, Ix);
        vrr_1d_ct<A_MAX, C_MAX>(
            (Py-Ay) + u_over_p*(Qy-Py), (Qy-Cy) + u_over_q*(Py-Qy),
            B10, B01, B00, Iy);
        vrr_1d_ct<A_MAX, C_MAX>(
            (Pz-Az) + u_over_p*(Qz-Pz), (Qz-Cz) + u_over_q*(Pz-Qz),
            B10, B01, B00, Iz);

        // Component accumulation
        #pragma unroll
        for (int ia = 0; ia < NA; ia++) {
            const int ax = loop_to_ang[LA][ia][0];
            const int ay = loop_to_ang[LA][ia][1];
            const int az = loop_to_ang[LA][ia][2];
            #pragma unroll
            for (int ib = 0; ib < NB; ib++) {
                const int bx = loop_to_ang[LB][ib][0];
                const int by = loop_to_ang[LB][ib][1];
                const int bz = loop_to_ang[LB][ib][2];

                // Bra TRR
                double bra_x[CS], bra_y[CS], bra_z[CS];
                #pragma unroll
                for (int ct = 0; ct <= C_MAX; ct++) {
                    bra_x[ct] = bra_trr_ct<C_MAX>(ax, bx, ct, ABx, Ix, CS);
                    bra_y[ct] = bra_trr_ct<C_MAX>(ay, by, ct, ABy, Iy, CS);
                    bra_z[ct] = bra_trr_ct<C_MAX>(az, bz, ct, ABz, Iz, CS);
                }

                #pragma unroll
                for (int ic = 0; ic < NC; ic++) {
                    const int cx = loop_to_ang[LC][ic][0];
                    const int cy = loop_to_ang[LC][ic][1];
                    const int cz = loop_to_ang[LC][ic][2];
                    #pragma unroll
                    for (int id_c = 0; id_c < ND; id_c++) {
                        const int dx = loop_to_ang[LD][id_c][0];
                        const int dy = loop_to_ang[LD][id_c][1];
                        const int dz = loop_to_ang[LD][id_c][2];

                        double Ix_val = ket_trr_ct<(LD > 0 ? LD : 1)>(dx, CDx, &bra_x[cx]);
                        double Iy_val = ket_trr_ct<(LD > 0 ? LD : 1)>(dy, CDy, &bra_y[cy]);
                        double Iz_val = ket_trr_ct<(LD > 0 ? LD : 1)>(dz, CDz, &bra_z[cz]);

                        eri_buf[((ia*NB + ib)*NC + ic)*ND + id_c] += wn * Ix_val * Iy_val * Iz_val;
                    }
                }
            }
        }
    }

    // --- Write results ---
    #pragma unroll
    for (int ia = 0; ia < NA; ia++) {
        const double Norm_A = calcNorm(alpha, loop_to_ang[LA][ia][0], loop_to_ang[LA][ia][1], loop_to_ang[LA][ia][2]);
        #pragma unroll
        for (int ib = 0; ib < NB; ib++) {
            const double Norm_B = calcNorm(beta, loop_to_ang[LB][ib][0], loop_to_ang[LB][ib][1], loop_to_ang[LB][ib][2]);
            #pragma unroll
            for (int ic = 0; ic < NC; ic++) {
                const double Norm_C = calcNorm(gamma, loop_to_ang[LC][ic][0], loop_to_ang[LC][ic][1], loop_to_ang[LC][ic][2]);
                #pragma unroll
                for (int id_c = 0; id_c < ND; id_c++) {
                    const double Norm_D = calcNorm(delta_exp, loop_to_ang[LD][id_c][0], loop_to_ang[LD][id_c][1], loop_to_ang[LD][id_c][2]);
                    double val = eri_buf[((ia*NB + ib)*NC + ic)*ND + id_c];
                    if (val != 0.0) {
                        addToResult_case1(
                            Norm_A * Norm_B * Norm_C * Norm_D * prefactor * val,
                            g_int2e,
                            base_a + ia, base_b + ib, base_c + ic, base_d + id_c,
                            num_basis, sym_bra, sym_ket, sym_braket,
                            g_cgto_normalization_factors);
                    }
                }
            }
        }
    }
}

// ============================================================
//  Dispatch: select specialized kernel by shell types
// ============================================================
inline eri_kernel_t get_rys_kernel_specialized(int a, int b, int c, int d) {
    // Canonical ordering: a<=b, c<=d, (a,b)<=(c,d)
    // Already done by get_eri_kernel before calling this

    // D-shell combinations (cc-pVDZ hot paths)
    #define CASE(A,B,C,D) if(a==A&&b==B&&c==C&&d==D) return RysERI_T<A,B,C,D>
    CASE(0,0,0,2);  // sssd  N=2
    CASE(0,0,1,2);  // sspd  N=2
    CASE(0,0,2,2);  // ssdd  N=3
    CASE(0,1,0,2);  // spsd  N=2
    CASE(0,1,1,2);  // sppd  N=2
    CASE(0,1,2,2);  // spdd  N=3
    CASE(0,2,0,2);  // sdsd  N=3
    CASE(0,2,1,2);  // sdpd  N=3
    CASE(0,2,2,2);  // sddd  N=4
    CASE(1,1,0,2);  // ppsd  N=2
    CASE(1,1,1,2);  // pppd  N=3
    CASE(1,1,2,2);  // ppdd  N=3
    CASE(1,2,1,2);  // pdpd  N=3
    CASE(1,2,2,2);  // pddd  N=4
    CASE(2,2,2,2);  // dddd  N=5

    // F-shell combinations (cc-pVTZ)
    CASE(0,0,0,3);  CASE(0,0,1,3);  CASE(0,0,2,3);  CASE(0,0,3,3);
    CASE(0,1,0,3);  CASE(0,1,1,3);  CASE(0,1,2,3);  CASE(0,1,3,3);
    CASE(0,2,0,3);  CASE(0,2,1,3);  CASE(0,2,2,3);  CASE(0,2,3,3);
    CASE(0,3,0,3);  CASE(0,3,1,3);  CASE(0,3,2,3);  CASE(0,3,3,3);
    CASE(1,1,0,3);  CASE(1,1,1,3);  CASE(1,1,2,3);  CASE(1,1,3,3);
    CASE(1,2,1,3);  CASE(1,2,2,3);  CASE(1,2,3,3);
    CASE(1,3,1,3);  CASE(1,3,2,3);  CASE(1,3,3,3);
    CASE(2,2,0,3);  CASE(2,2,1,3);  CASE(2,2,2,3);  CASE(2,2,3,3);
    CASE(2,3,2,3);  CASE(2,3,3,3);
    CASE(3,3,3,3);
    #undef CASE

    return nullptr; // fallback to generic RysERI
}

} // namespace gansu::gpu

#endif // RYS_ERI_SPECIALIZED_HPP
