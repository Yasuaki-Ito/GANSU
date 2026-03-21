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
 * @file cc2_solver.hpp
 * @brief CC2 ground state amplitude solver for RHF reference
 *
 * Solves the CC2 amplitude equations iteratively:
 *   T1: Full CCSD singles equation (same as CCSD T1)
 *   T2: Simplified doubles equation (Fock×T2 + ERI + ERI×T1 only, no T2×T2)
 *
 * Uses spatial orbital convention consistent with EOM operators:
 *   T1[i*nvir + a] = t^a_i
 *   T2[i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b] = t^{ab}_{ij}
 *   where t^{ab}_{ij} = (ia|jb)/(ε_i+ε_j-ε_a-ε_b) at MP1 level
 *
 * T1 uses ÷2 convention (spin degeneracy factor divided out).
 * T2 uses αβ spin-sector projection (directly gives EOM-convention T2).
 *
 * ERI blocks used:
 *   eri_ovov[i,a,j,b] = (ia|jb)
 *   eri_vvov[a,b,i,c] = (ab|ic)
 *   eri_ooov[j,i,k,b] = (ji|kb)
 *   eri_oovv[i,j,a,b] = (ij|ab)
 *   eri_ovvo[i,a,b,j] = (ia|bj)
 *   eri_vvvv[a,b,c,d] = (ab|cd)  -- for T2 dressed integrals
 *   eri_oooo[i,j,k,l] = (ij|kl)  -- for T2 dressed integrals
 */

#pragma once

#include "types.hpp"

namespace gansu {

struct CC2Result {
    real_t* d_t1;       // [nocc * nvir] device pointer, caller must free
    real_t* d_t2;       // [nocc * nocc * nvir * nvir] device pointer, caller must free
    real_t cc2_energy;  // CC2 correlation energy
    bool converged;
};

/**
 * @brief Solve CC2 ground state amplitudes
 *
 * @param d_eri_ovov Device pointer to (ia|jb) block [nocc*nvir*nocc*nvir]
 * @param d_eri_vvov Device pointer to (ab|ic) block [nvir*nvir*nocc*nvir]
 * @param d_eri_ooov Device pointer to (ji|kb) block [nocc*nocc*nocc*nvir]
 * @param d_eri_oovv Device pointer to (ij|ab) block [nocc*nocc*nvir*nvir]
 * @param d_eri_ovvo Device pointer to (ia|bj) block [nocc*nvir*nvir*nocc]
 * @param d_eri_vvvv Device pointer to (ab|cd) block [nvir^4]
 * @param d_eri_oooo Device pointer to (ij|kl) block [nocc^4]
 * @param d_f_oo     Device pointer to occupied orbital energies [nocc]
 * @param d_f_vv     Device pointer to virtual orbital energies [nvir]
 * @param d_D1       Device pointer to D1[ia] = eps_a - eps_i [nocc*nvir]
 * @param d_D2       Device pointer to D2[ijab] = eps_a + eps_b - eps_i - eps_j [nocc^2*nvir^2]
 * @param nocc       Number of occupied spatial orbitals
 * @param nvir       Number of virtual spatial orbitals
 * @param max_iter   Maximum number of iterations (default 100)
 * @param conv_thresh Convergence threshold for RMS residual (default 1e-8)
 * @return CC2Result with converged T1, T2 amplitudes and CC2 energy
 */
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
    int max_iter = 100,
    real_t conv_thresh = 1e-8);

} // namespace gansu
