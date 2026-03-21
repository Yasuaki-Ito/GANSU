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

/**
 * @file cc2_solver.cu
 * @brief CC2 ground state amplitude solver (EOM convention T2)
 *
 * T1 equation: Same as CCSD T1 (full spin-summed spatial equation).
 *   Uses D1 = ε_a - ε_i as preconditioner: t1_new = -numerator / D1.
 *   All antisymmetrized integrals use w = 2*Coulomb - Exchange (both spin channels).
 *
 * T2 equation: Uses dressed integral formulation from the similarity
 *   transformation exp(-T1) H exp(T1). The CC2 T2 equation is:
 *     D2 * t2(i,j,a,b) + (ĩã|j̃b̃) = 0
 *   where (ĩã|j̃b̃) is the fully T1-dressed chemist integral.
 *
 *   Dressed orbital coefficients (exact for T1, BCH truncates at 1st order):
 *     ĩ = |i⟩ + Σ_c t1(i,c) |c⟩        [occupied dressed with virtual]
 *     ã = ⟨a| + Σ_k t1(k,a) ⟨k|        [virtual dressed with occupied]
 *
 *   The expansion has 2^4 = 16 terms (orders 0-4 in T1).
 *   - Pair symmetry: t2(i,j,a,b) = t2(j,i,b,a)
 *   - Denominator: D2 = ε_a + ε_b - ε_i - ε_j (same as CCSD/EOM-MP2)
 *   - MP1 initial: t2(i,j,a,b) = (ia|jb) / (ε_i+ε_j-ε_a-ε_b) = -(ia|jb)/D2
 *
 * DIIS acceleration after initial damping iterations.
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

#include "cc2_solver.hpp"
#include "diis.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "profiler.hpp"

namespace gansu {

// ========================================================================
//  ERI access macros (same convention as eom_mp2_operator.cu)
// ========================================================================

#define OVOV(i,a,j,b) d_eri_ovov[(size_t)(i)*nvir*nocc*nvir + (size_t)(a)*nocc*nvir + (size_t)(j)*nvir + (b)]
#define VVOV(a,b,i,c) d_eri_vvov[(size_t)(a)*nvir*nocc*nvir + (size_t)(b)*nocc*nvir + (size_t)(i)*nvir + (c)]
#define OOOV(j,i,k,b) d_eri_ooov[(size_t)(j)*nocc*nocc*nvir + (size_t)(i)*nocc*nvir + (size_t)(k)*nvir + (b)]
#define OOVV(i,j,a,b) d_eri_oovv[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]
#define OVVO(i,a,b,j) d_eri_ovvo[(size_t)(i)*nvir*nvir*nocc + (size_t)(a)*nvir*nocc + (size_t)(b)*nocc + (j)]
#define VVVV(a,b,c,d) d_eri_vvvv[(size_t)(a)*nvir*nvir*nvir + (size_t)(b)*nvir*nvir + (size_t)(c)*nvir + (d)]
#define OOOO(i,j,k,l) d_eri_oooo[(size_t)(i)*nocc*nocc*nocc + (size_t)(j)*nocc*nocc + (size_t)(k)*nocc + (l)]
#define T1(i,a) d_t1[(i)*nvir + (a)]
#define T2(i,j,a,b) d_t2[(size_t)(i)*nocc*nvir*nvir + (size_t)(j)*nvir*nvir + (size_t)(a)*nvir + (b)]

// ========================================================================
//  CC2 T2 residual kernel (αβ spin sector, EOM convention)
// ========================================================================

/**
 * CC2 T2 update using the dressed integral formulation.
 *
 * The CC2 T2 equation in the αβ sector is:
 *   D2 * t2(i,j,a,b) + (ĩã|j̃b̃) = 0
 *
 * where (ĩã|j̃b̃) is the T1-similarity-transformed (dressed) ERI:
 *   (ĩã|j̃b̃) = Σ_{PQRS} c^ĩ_P c^ã_Q c^j̃_R c^b̃_S (PQ|RS)
 *
 * Dressed orbital coefficients (from matrix element ⟨Φ^{ab}_{ij}| V̂ |HF⟩, X²=0 → U=I+X):
 *
 * In V = Σ(pq|rs) a†_p a_q a†_r a_s, the contraction with ⟨Φ^{ab}_{ij}| maps:
 *   a†_p (creation α) → particle a:  dressed via [e^{-X}]_{ap} = {a:1, k:-t^a_k}
 *   a_q  (annihil. α) → hole i:      dressed via [e^{+X}]_{qi} = {i:1, c:+t^c_i}
 *   a†_r (creation β) → particle b:  dressed via [e^{-X}]_{br} = {b:1, l:-t^b_l}
 *   a_s  (annihil. β) → hole j:      dressed via [e^{+X}]_{sj} = {j:1, d:+t^d_j}
 *
 * Equivalently (swapping within electron pairs by (pq|rs)=(qp|sr)):
 *   ĩ: c_i = 1, c_c = +t1(i,c)  [hole dressed with virtual, POSITIVE]
 *   ã: c_a = 1, c_k = -t1(k,a)  [particle dressed with occupied, NEGATIVE]
 *   j̃: c_j = 1, c_d = +t1(j,d)
 *   b̃: c_b = 1, c_l = -t1(l,b)
 *
 * Pair symmetry: (ĩã|j̃b̃) = (j̃b̃|ĩã) guarantees t2(i,j,a,b) = t2(j,i,b,a) ✓
 */
__global__ void cc2_t2_update_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_eri_ovvo,
    const real_t* __restrict__ d_eri_vvvv,
    const real_t* __restrict__ d_eri_oooo,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_t2_new,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doubles_dim = nocc * nocc * nvir * nvir;
    if (idx >= doubles_dim) return;

    int i = idx / (nocc * nvir * nvir);
    int rem = idx % (nocc * nvir * nvir);
    int j = rem / (nvir * nvir);
    rem %= (nvir * nvir);
    int a = rem / nvir;
    int b = rem % nvir;

    // val = -(ĩã|j̃b̃), the negated dressed integral

    // === Order 0: -(ia|jb) ===
    real_t val = -OVOV(i, a, j, b);

    // === Order 1: 4 terms (dress one index) ===
    // Dress ĩ→c: hole i dressed with +t1(i,c), integral (ca|jb)
    for (int c = 0; c < nvir; c++) {
        val -= T1(i, c) * VVOV(c, a, j, b);
    }
    // Dress ã→k: particle a dressed with -t1(k,a), integral (ik|jb)
    for (int k = 0; k < nocc; k++) {
        val += T1(k, a) * OOOV(i, k, j, b);
    }
    // Dress j̃→d: hole j dressed with +t1(j,d), integral (ia|db)
    for (int d = 0; d < nvir; d++) {
        val -= T1(j, d) * VVOV(d, b, i, a);
    }
    // Dress b̃→l: particle b dressed with -t1(l,b), integral (ia|jl)
    for (int l = 0; l < nocc; l++) {
        val += T1(l, b) * OOOV(j, l, i, a);
    }

    // === Order 2: 6 terms (dress two indices) ===
    // Dress ĩ,ã: (+t1_ic)(-t1_ka) → val += t1_ic*t1_ka*(ck|jb)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int k = 0; k < nocc; k++) {
            val += t1_ic * T1(k, a) * OVOV(k, c, j, b);
        }
    }
    // Dress ĩ,j̃: (+t1_ic)(+t1_jd) → val -= t1_ic*t1_jd*(ca|db)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int d = 0; d < nvir; d++) {
            val -= t1_ic * T1(j, d) * VVVV(c, a, d, b);
        }
    }
    // Dress ĩ,b̃: (+t1_ic)(-t1_lb) → val += t1_ic*t1_lb*(ca|jl)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int l = 0; l < nocc; l++) {
            val += t1_ic * T1(l, b) * OOVV(j, l, c, a);
        }
    }
    // Dress ã,j̃: (-t1_ka)(+t1_jd) → val += t1_ka*t1_jd*(ik|db)
    for (int k = 0; k < nocc; k++) {
        real_t t1_ka = T1(k, a);
        for (int d = 0; d < nvir; d++) {
            val += t1_ka * T1(j, d) * OOVV(i, k, d, b);
        }
    }
    // Dress ã,b̃: (-t1_ka)(-t1_lb) → val -= t1_ka*t1_lb*(ik|jl)
    for (int k = 0; k < nocc; k++) {
        real_t t1_ka = T1(k, a);
        for (int l = 0; l < nocc; l++) {
            val -= t1_ka * T1(l, b) * OOOO(i, k, j, l);
        }
    }
    // Dress j̃,b̃: (+t1_jd)(-t1_lb) → val += t1_jd*t1_lb*(ia|dl)
    for (int d = 0; d < nvir; d++) {
        real_t t1_jd = T1(j, d);
        for (int l = 0; l < nocc; l++) {
            val += t1_jd * T1(l, b) * OVVO(i, a, d, l);
        }
    }

    // === Order 3: 4 terms (dress three indices) ===
    // Dress ĩ,ã,j̃: (+t_ic)(-t_ka)(+t_jd), (ck|db) = VVOV(d,b,k,c)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int k = 0; k < nocc; k++) {
            real_t t1_ic_ka = t1_ic * T1(k, a);
            for (int d = 0; d < nvir; d++) {
                val += t1_ic_ka * T1(j, d) * VVOV(d, b, k, c);
            }
        }
    }
    // Dress ĩ,ã,b̃: (+t_ic)(-t_ka)(-t_lb), (ck|jl) = OOOV(j,l,k,c)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int k = 0; k < nocc; k++) {
            real_t t1_ic_ka = t1_ic * T1(k, a);
            for (int l = 0; l < nocc; l++) {
                val -= t1_ic_ka * T1(l, b) * OOOV(j, l, k, c);
            }
        }
    }
    // Dress ĩ,j̃,b̃: (+t_ic)(+t_jd)(-t_lb), (ca|dl) = VVOV(c,a,l,d)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int d = 0; d < nvir; d++) {
            real_t t1_ic_jd = t1_ic * T1(j, d);
            for (int l = 0; l < nocc; l++) {
                val += t1_ic_jd * T1(l, b) * VVOV(c, a, l, d);
            }
        }
    }
    // Dress ã,j̃,b̃: (-t_ka)(+t_jd)(-t_lb), (ik|dl) = OOOV(i,k,l,d)
    for (int k = 0; k < nocc; k++) {
        real_t t1_ka = T1(k, a);
        for (int d = 0; d < nvir; d++) {
            real_t t1_ka_jd = t1_ka * T1(j, d);
            for (int l = 0; l < nocc; l++) {
                val -= t1_ka_jd * T1(l, b) * OOOV(i, k, l, d);
            }
        }
    }

    // === Order 4: 1 term (all four dressed) ===
    // Dress all: (+t_ic)(-t_ka)(+t_jd)(-t_lb) = +t*t*t*t → val -= ...
    //   where (ck|dl) = (kc|ld) = OVOV(k,c,l,d)
    for (int c = 0; c < nvir; c++) {
        real_t t1_ic = T1(i, c);
        for (int k = 0; k < nocc; k++) {
            real_t t1_ic_ka = t1_ic * T1(k, a);
            for (int d = 0; d < nvir; d++) {
                real_t t1_ic_ka_jd = t1_ic_ka * T1(j, d);
                for (int l = 0; l < nocc; l++) {
                    val -= t1_ic_ka_jd * T1(l, b) * OVOV(k, c, l, d);
                }
            }
        }
    }

    // Divide by D2 = ε_a + ε_b - ε_i - ε_j (positive)
    real_t d2 = d_D2[idx];
    d_t2_new[idx] = (fabs(d2) > 1e-12) ? val / d2 : 0.0;
}


// ========================================================================
//  CC2 T1 residual kernel (÷2 convention, same as CCSD T1)
// ========================================================================

/**
 * CC2 T1 update: computes numerator for Jacobi update.
 * Full CCSD T1 equation (identical to CCSD).
 *
 * For canonical HF, f_ov = f_vo = 0, so several terms vanish.
 * All terms use spin-summed antisymmetrized integrals w = 2*J - K:
 *
 *   (T3)  Σ_{m,e} t1[m,e] [2*(ia|me) - (mi|ae)]                (CIS-like, w_voov×t1)
 *   (T4)  Σ_m t1[m,a] * Σ_{n,e} t1[n,e] [-2*(mi|ne) + (me|ni)] (-w_ooov×t1², spin-summed)
 *   (T5)  Σ_e t1[i,e] * Σ_{m,f} t1[m,f] [2*(ae|mf) - (af|me)]  (w_ovvv×t1², spin-summed)
 *   (T6)  t1[i,a] * Σ_{m,n,e,f} t1[m,e]*t1[n,f] [-(me|nf) + 0.5*(mf|ne)]  (T1³, approx)
 *   (T7)  Σ_{m,n,e} t2[m,n,a,e] [-2*(mi|ne) + (me|ni)]         (-w_ooov×t2, spin-summed)
 *   (T8)  Σ_{m,e,f} t2[i,m,e,f] [2*(ae|mf) - (af|me)]          (w_ovvv×t2, spin-summed)
 *   (T9)  Σ_m t1[m,a] * Σ_{n,e,f} t2[i,n,e,f] [-2*(me|nf) + (mf|ne)] (-Fki_T2×t1)
 *   (T10) Σ_e t1[i,e] * Σ_{m,n,f} t2[m,n,a,f] [-2*(me|nf) + (mf|ne)] (Fac_T2×t1)
 *   (T11) Σ_{k,c} Fkc * [2*t2(k,i,c,a) - t2(i,k,c,a) + t1(i,c)*t1(k,a)]  (Fkc term)
 *         where Fkc = Σ_{l,d} [2*(kc|ld)-(kd|lc)] * t1(l,d)
 *
 * t1_new[i,a] = numerator / D1[i,a]  where D1 = eps_i - eps_a
 */
__global__ void cc2_t1_update_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_eri_vvov,
    const real_t* __restrict__ d_eri_ooov,
    const real_t* __restrict__ d_eri_oovv,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_t2,
    const real_t* __restrict__ d_D1,
    real_t* __restrict__ d_t1_new,
    int nocc, int nvir)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= nocc * nvir) return;
    int i = ia / nvir;
    int a = ia % nvir;

    real_t val = 0.0;

    // ---- CIS-like term (T3): Σ_{m,e} t1[m,e] [2*(ai|me) - (ae|mi)] ----
    // (ai|me) = OVOV(i,a,m,e) [using (pq|rs)=(qp|rs) symmetry]
    // (ae|mi) = OOVV(m,i,a,e) [using (pq|rs)=(rs|pq) symmetry]
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            real_t ai_me = OVOV(i, a, m, e);
            real_t ae_mi = OOVV(m, i, a, e);
            val += T1(m, e) * (2.0 * ai_me - ae_mi);
        }
    }

    // ---- T1×T1 terms (T4): Σ_m t1[m,a] * Σ_{n,e} t1[n,e] [-2*(mi|ne) + (me|ni)] ----
    // Spin-summed: αα gives -(mi|ne)+(me|ni), ββ gives -(mi|ne) → total -2*(mi|ne)+(me|ni)
    // (mi|ne) = OOOV(m,i,n,e), (me|ni) = OOOV(n,i,m,e)
    for (int m = 0; m < nocc; m++) {
        real_t t1_ma = T1(m, a);
        real_t inner = 0.0;
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                inner += T1(n, e) * (-2.0 * OOOV(m, i, n, e) + OOOV(n, i, m, e));
            }
        }
        val += t1_ma * inner;
    }

    // ---- T1×T1 terms (T5): Σ_e t1[i,e] * Σ_{m,f} t1[m,f] [2*(ae|mf) - (af|me)] ----
    // (ae|mf) = VVOV(a,e,m,f) [using (pq|rs)=(qp|rs)]
    // Wait: VVOV stores (ab|ic) = VVOV(a,b,i,c). So (ae|mf): a,e∈vir, m∈occ, f∈vir
    // → VVOV(a,e,m,f) ✓
    // (af|me) = VVOV(a,f,m,e) ✓
    for (int e = 0; e < nvir; e++) {
        real_t t1_ie = T1(i, e);
        real_t inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int f = 0; f < nvir; f++) {
                inner += T1(m, f) * (2.0 * VVOV(a, e, m, f) - VVOV(a, f, m, e));
            }
        }
        val += t1_ie * inner;
    }

    // ---- T1×T1×T1 term (T6): t1[i,a] * Σ_{m,n,e,f} t1[m,e]*t1[n,f]*[-(me|nf)+0.5*(mf|ne)] ----
    // (me|nf) = OVOV(m,e,n,f)
    // (mf|ne) = OVOV(m,f,n,e)
    {
        real_t inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int e = 0; e < nvir; e++) {
                    for (int f = 0; f < nvir; f++) {
                        real_t K = -OVOV(m, e, n, f) + 0.5 * OVOV(m, f, n, e);
                        inner += T1(m, e) * T1(n, f) * K;
                    }
                }
            }
        }
        val += T1(i, a) * inner;
    }

    // ---- ERI×T2 term (T7): Σ_{m,n,e} t2[m,n,a,e] [-2*(mi|ne) + (me|ni)] ----
    // Spin-summed: -w_ooov × t2 where w = 2J - K
    for (int m = 0; m < nocc; m++) {
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                real_t K = -2.0 * OOOV(m, i, n, e) + OOOV(n, i, m, e);
                val += T2(m, n, a, e) * K;
            }
        }
    }

    // ---- ERI×T2 term (T8): Σ_{m,e,f} t2[i,m,e,f] [2*(ae|mf) - (af|me)] ----
    // Spin-summed: w_ovvv × t2 where w = 2J - K
    // (ae|mf) = VVOV(a,e,m,f), (af|me) = VVOV(a,f,m,e)
    for (int m = 0; m < nocc; m++) {
        for (int e = 0; e < nvir; e++) {
            for (int f = 0; f < nvir; f++) {
                val += (2.0 * VVOV(a, e, m, f) - VVOV(a, f, m, e)) * T2(i, m, e, f);
            }
        }
    }

    // ---- T1×T2 terms (T9): Σ_m t1[m,a] * Σ_{n,e,f} t2[i,n,e,f]*[-2*(me|nf)+(mf|ne)] ----
    // Spin-summed: -Fki(T2 part)*t1 where Fki = Σ w_oovv*tau, w = 2J-K
    for (int m = 0; m < nocc; m++) {
        real_t t1_ma = T1(m, a);
        real_t inner = 0.0;
        for (int n = 0; n < nocc; n++) {
            for (int e = 0; e < nvir; e++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -2.0 * OVOV(m, e, n, f) + OVOV(m, f, n, e);
                    inner += T2(i, n, e, f) * K;
                }
            }
        }
        val += t1_ma * inner;
    }

    // ---- T1×T2 terms (T10): Σ_e t1[i,e] * Σ_{m,n,f} t2[m,n,a,f]*[-2*(me|nf)+(mf|ne)] ----
    // Spin-summed: Fac(T2 part)*t1 where Fac = -Σ w_oovv*tau, w = 2J-K
    for (int e = 0; e < nvir; e++) {
        real_t t1_ie = T1(i, e);
        real_t inner = 0.0;
        for (int m = 0; m < nocc; m++) {
            for (int n = 0; n < nocc; n++) {
                for (int f = 0; f < nvir; f++) {
                    real_t K = -2.0 * OVOV(m, e, n, f) + OVOV(m, f, n, e);
                    inner += T2(m, n, a, f) * K;
                }
            }
        }
        val += t1_ie * inner;
    }

    // ---- Fkc term (T11): Σ_{k,c} Fkc * [2*t2(k,i,c,a) - t2(i,k,c,a) + t1(i,c)*t1(k,a)] ----
    // Fkc[k,c] = Σ_{l,d} w_oovv[k,l,c,d] * t1(l,d) = Σ_{l,d} [2*(kc|ld)-(kd|lc)] * t1(l,d)
    // This is the CCSD T1 Fock intermediate × T2 term, missing from original spin2spatial derivation.
    for (int k = 0; k < nocc; k++) {
        for (int c = 0; c < nvir; c++) {
            // Compute Fkc inline
            real_t fkc = 0.0;
            for (int l = 0; l < nocc; l++) {
                for (int d = 0; d < nvir; d++) {
                    fkc += (2.0 * OVOV(k, c, l, d) - OVOV(k, d, l, c)) * T1(l, d);
                }
            }
            val += fkc * (2.0 * T2(k, i, c, a) - T2(i, k, c, a) + T1(i, c) * T1(k, a));
        }
    }

    // Divide by D1 (Jacobi update)
    // D1 stored as eps_a - eps_i (positive). Denominator for update is eps_i - eps_a (negative).
    // So: t1_new = val / (eps_i - eps_a) = val / (-D1) = -val / D1
    real_t d1 = d_D1[ia];
    d_t1_new[ia] = (fabs(d1) > 1e-12) ? -val / d1 : 0.0;
}


// ========================================================================
//  CC2 correlation energy kernel
// ========================================================================

/**
 * CC2 correlation energy = Σ_{i,j,a,b} (2*(ia|jb) - (ib|ja)) * (t2[i,j,a,b] + t1[i,a]*t1[j,b])
 * This is the same formula as MP2/CCSD correlation energy.
 */
__global__ void cc2_correlation_energy_kernel(
    const real_t* __restrict__ d_eri_ovov,
    const real_t* __restrict__ d_t1,
    const real_t* __restrict__ d_t2,
    real_t* __restrict__ d_partial_sums,
    int nocc, int nvir)
{
    extern __shared__ real_t sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doubles_dim = nocc * nocc * nvir * nvir;

    real_t local_sum = 0.0;
    if (idx < doubles_dim) {
        int i = idx / (nocc * nvir * nvir);
        int rem = idx % (nocc * nvir * nvir);
        int j = rem / (nvir * nvir);
        rem %= (nvir * nvir);
        int a = rem / nvir;
        int b = rem % nvir;

        real_t ia_jb = OVOV(i, a, j, b);
        real_t ib_ja = OVOV(i, b, j, a);
        real_t tau = T2(i, j, a, b) + T1(i, a) * T1(j, b);
        local_sum = (2.0 * ia_jb - ib_ja) * tau;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_partial_sums[blockIdx.x] = sdata[0];
    }
}

#undef OVOV
#undef VVOV
#undef OOOV
#undef OOVV
#undef OVVO
#undef VVVV
#undef OOOO
#undef T1
#undef T2

// ========================================================================
//  CC2 solver main function
// ========================================================================

CC2Result solve_cc2(
    const real_t* d_eri_ovov,
    const real_t* d_eri_vvov,
    const real_t* d_eri_ooov,
    const real_t* d_eri_oovv,
    const real_t* d_eri_ovvo,
    const real_t* d_eri_vvvv,
    const real_t* d_eri_oooo,
    const real_t* d_f_oo,
    const real_t* d_f_vv,
    const real_t* d_D1,
    const real_t* d_D2,
    int nocc, int nvir,
    int max_iter,
    real_t conv_thresh)
{
    PROFILE_FUNCTION();

    const int singles_dim = nocc * nvir;
    const int doubles_dim = nocc * nocc * nvir * nvir;
    const size_t t1_bytes = (size_t)singles_dim * sizeof(real_t);
    const size_t t2_bytes = (size_t)doubles_dim * sizeof(real_t);

    std::cout << "  CC2 solver: nocc=" << nocc << " nvir=" << nvir
              << " singles=" << singles_dim << " doubles=" << doubles_dim << std::endl;

    // Allocate T1 and T2 on device
    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    real_t* d_t1_new = nullptr;
    real_t* d_t2_new = nullptr;

    tracked_cudaMalloc(&d_t1, t1_bytes);
    tracked_cudaMalloc(&d_t2, t2_bytes);
    tracked_cudaMalloc(&d_t1_new, t1_bytes);
    tracked_cudaMalloc(&d_t2_new, t2_bytes);

    // Initialize T1 = 0
    cudaMemset(d_t1, 0, t1_bytes);

    // Initialize T2 = MP1: t2[i,j,a,b] = (ia|jb) / D2[i,j,a,b]
    // D2 is stored as eps_a + eps_b - eps_i - eps_j (positive for bound states)
    // Denominator for MP1 is eps_i + eps_j - eps_a - eps_b = -D2
    // So t2 = -(ia|jb) / D2 ... wait, D2 = eps_a+eps_b-eps_i-eps_j, so eps_i+eps_j-eps_a-eps_b = -D2
    // Actually, checking EOM-MP2: d_t2[idx] = ia_jb / denom where denom = eps_i+eps_j-eps_a-eps_b
    // And d_D2[idx] = eps_a+eps_b-eps_i-eps_j. So denom = -D2.
    // t2_mp1 = (ia|jb) / (-D2) = -(ia|jb) / D2
    // Wait, let me double-check. In eom_mp2_compute_t2_D2_kernel:
    //   denom = eps_i + eps_j - eps_a - eps_b; (NEGATIVE for bound states)
    //   d_t2[idx] = ia_jb / denom;  (ia_jb > 0 typically, so t2 < 0)
    //   d_D2[idx] = eps_a + eps_b - eps_i - eps_j;  (POSITIVE, = -denom)
    // So D2 = -denom, and t2 = ia_jb / denom = -ia_jb / D2
    {
        // Simple kernel to compute MP1 T2
        // t2[idx] = -OVOV(i,a,j,b) / D2[idx]  (since D2 = eps_a+eps_b-eps_i-eps_j > 0)
        // Actually t2[idx] = OVOV(i,a,j,b) / (eps_i+eps_j-eps_a-eps_b) = -OVOV / D2
        int threads = 256;
        int blocks = (doubles_dim + threads - 1) / threads;
        // Use T2 update kernel with t1=0 to get MP1 (or do it manually)
        // For simplicity, just copy from the ERI and divide by denominator
        // We can use cc2_t2_update_kernel with t1=0 which gives:
        //   val = 2*(ia|jb) - 2*(ib|ja) + 0 + 0 = 2*[(ia|jb) - (ib|ja)]
        //   t2_new = val / D2
        // But this includes the factor 2 and the exchange term. The correct MP1 is t2 = (ia|jb)/(-D2).
        // The CC2 T2 update gives: t2 = [2*(ia|jb) - 2*(ib|ja)] / D2
        // This is NOT the MP1 T2 we want.
        //
        // The issue: our Jacobi update computes the FULL non-Fock numerator and divides by D2.
        // But D2 = eps_a+eps_b-eps_i-eps_j, and the Fock diagonal in the residual is
        // -3ε_j + 3ε_b (after ÷2). The Jacobi update with D2 as preconditioner doesn't give
        // the exact MP1 solution in one step.
        //
        // So we initialize T2 explicitly as the correct MP1 amplitude.
        // We'll write a small lambda kernel.
    }

    // Initialize T2 to MP1 explicitly
    {
        std::vector<real_t> h_ovov((size_t)nocc * nvir * nocc * nvir);
        std::vector<real_t> h_D2((size_t)doubles_dim);
        cudaMemcpy(h_ovov.data(), d_eri_ovov, h_ovov.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_D2.data(), d_D2, h_D2.size() * sizeof(real_t), cudaMemcpyDeviceToHost);

        std::vector<real_t> h_t2((size_t)doubles_dim);
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++) {
                        size_t idx = (size_t)i * nocc * nvir * nvir + j * nvir * nvir + a * nvir + b;
                        size_t ovov_idx = (size_t)i * nvir * nocc * nvir + a * nocc * nvir + j * nvir + b;
                        real_t denom = -h_D2[idx]; // eps_i + eps_j - eps_a - eps_b
                        h_t2[idx] = (std::abs(denom) > 1e-12) ? h_ovov[ovov_idx] / denom : 0.0;
                    }
        cudaMemcpy(d_t2, h_t2.data(), t2_bytes, cudaMemcpyHostToDevice);
    }

    // Compute initial CC2 energy (= MP2 energy at this point)
    auto compute_cc2_energy = [&](const real_t* d_t1_curr, const real_t* d_t2_curr) -> real_t {
        int threads = 256;
        int blocks = (doubles_dim + threads - 1) / threads;
        real_t* d_partial = nullptr;
        tracked_cudaMalloc(&d_partial, blocks * sizeof(real_t));
        cc2_correlation_energy_kernel<<<blocks, threads, threads * sizeof(real_t)>>>(
            d_eri_ovov, d_t1_curr, d_t2_curr, d_partial, nocc, nvir);
        std::vector<real_t> h_partial(blocks);
        cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_partial);
        real_t sum = 0.0;
        for (int i = 0; i < blocks; i++) sum += h_partial[i];
        return sum;
    };

    real_t E_cc2 = compute_cc2_energy(d_t1, d_t2);
    std::cout << "  Initial (MP2) correlation energy: " << std::fixed << std::setprecision(10)
              << E_cc2 << " Ha" << std::endl;

    // Host buffers for DIIS
    size_t num_amps = (size_t)singles_dim + doubles_dim;
    std::vector<real_t> h_amps_old(num_amps);
    std::vector<real_t> h_amps_new(num_amps);
    std::vector<real_t> h_residual(num_amps);

    DIIS diis(8, 2);
    bool converged = false;

    // Download current amplitudes as "old"
    cudaMemcpy(h_amps_old.data(), d_t1, t1_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_amps_old.data() + singles_dim, d_t2, t2_bytes, cudaMemcpyDeviceToHost);

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute new T1
        {
            int threads = 256;
            int blocks = (singles_dim + threads - 1) / threads;
            cc2_t1_update_kernel<<<blocks, threads>>>(
                d_eri_ovov, d_eri_vvov, d_eri_ooov, d_eri_oovv,
                d_t1, d_t2, d_D1, d_t1_new, nocc, nvir);
        }

        // Compute new T2
        {
            int threads = 256;
            int blocks = (doubles_dim + threads - 1) / threads;
            cc2_t2_update_kernel<<<blocks, threads>>>(
                d_eri_ovov, d_eri_vvov, d_eri_ooov, d_eri_oovv, d_eri_ovvo,
                d_eri_vvvv, d_eri_oooo,
                d_t1, d_D2, d_t2_new, nocc, nvir);
        }

        cudaDeviceSynchronize();

        // Compute CC2 energy with new amplitudes
        real_t E_cc2_new = compute_cc2_energy(d_t1_new, d_t2_new);

        // Download new amplitudes
        cudaMemcpy(h_amps_new.data(), d_t1_new, t1_bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_amps_new.data() + singles_dim, d_t2_new, t2_bytes, cudaMemcpyDeviceToHost);

        // Compute residual = new - old
        real_t rms = 0.0;
        for (size_t k = 0; k < num_amps; k++) {
            h_residual[k] = h_amps_new[k] - h_amps_old[k];
            rms += h_residual[k] * h_residual[k];
        }
        rms = std::sqrt(rms / num_amps);

        real_t dE = std::abs(E_cc2_new - E_cc2);

        std::cout << "  CC2 iter " << std::setw(3) << iter + 1
                  << "  E_corr=" << std::fixed << std::setprecision(10) << E_cc2_new
                  << "  dE=" << std::scientific << std::setprecision(3) << dE
                  << "  RMS=" << rms << std::endl;

        // Check convergence
        if (rms < conv_thresh || dE < conv_thresh * 0.1) {
            converged = true;
            E_cc2 = E_cc2_new;
            // Copy final amplitudes
            cudaMemcpy(d_t1, d_t1_new, t1_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_t2, d_t2_new, t2_bytes, cudaMemcpyDeviceToDevice);
            break;
        }

        E_cc2 = E_cc2_new;

        // DIIS
        diis.push(h_amps_new, h_residual);

        if (iter > 3 && diis.can_extrapolate()) {
            auto h_extrap = diis.extrapolate();
            cudaMemcpy(d_t1, h_extrap.data(), t1_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_t2, h_extrap.data() + singles_dim, t2_bytes, cudaMemcpyHostToDevice);
            std::copy(h_extrap.begin(), h_extrap.end(), h_amps_old.begin());
        } else {
            // Damping for first few iterations
            real_t alpha = 0.5;
            for (size_t k = 0; k < num_amps; k++) {
                h_amps_new[k] = (1.0 - alpha) * h_amps_old[k] + alpha * h_amps_new[k];
            }
            cudaMemcpy(d_t1, h_amps_new.data(), t1_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_t2, h_amps_new.data() + singles_dim, t2_bytes, cudaMemcpyHostToDevice);
            std::copy(h_amps_new.begin(), h_amps_new.end(), h_amps_old.begin());
        }
    }

    if (converged) {
        std::cout << "  CC2 converged! E_corr = " << std::fixed << std::setprecision(10)
                  << E_cc2 << " Ha" << std::endl;
    } else {
        std::cout << "  Warning: CC2 did not converge in " << max_iter << " iterations" << std::endl;
    }

    // Cleanup workspace
    tracked_cudaFree(d_t1_new);
    tracked_cudaFree(d_t2_new);

    return CC2Result{d_t1, d_t2, E_cc2, converged};
}

} // namespace gansu
