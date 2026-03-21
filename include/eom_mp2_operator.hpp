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
 * @file eom_mp2_operator.hpp
 * @brief EOM-MP2 operator in the full singles+doubles space
 *
 * Implements the EOM-MP2 σ-vector equations for RHF reference.
 * Unlike ADC(2), the doubles-doubles block (M22) is NOT diagonal
 * due to t2×r2 coupling terms, so Schur complement is not applicable.
 * Only full Davidson in singles+doubles space is supported.
 *
 * σ1 equations (12 einsum terms from EOM_MP2_RHF.md):
 *   f_oo×r1, f_vv×r1, eri_voov×r1, eri_vvoo×r1,
 *   eri_ovov×t2×r1 (4 terms), eri_ooov×r2, eri_ovoo×r2,
 *   eri_vvov×r2 (2 terms)
 *
 * σ2 equations (24 einsum terms from EOM_MP2_RHF.md):
 *   f_oo×r2, f_vv×r2, eri_oooo×r2, eri_oovo×r1,
 *   eri_oovv×r2, eri_ovov×t2×r1 (6 terms), eri_ovov×t2×r2 (6 terms),
 *   eri_ovvo×r2, eri_vovv×r1, eri_vvvo×r1, eri_vvvv×r2 (2 terms)
 *
 * The full matrix is non-symmetric (EOM equations are not Hermitian).
 */

#pragma once

#include "linear_operator.hpp"

namespace gansu {

class EOMMP2Operator : public LinearOperator {
public:
    /**
     * @brief Construct EOM-MP2 operator
     *
     * Precomputes T2_MP1 amplitudes and extracts all needed ERI blocks.
     *
     * @param d_eri_mo Device pointer to full MO ERI tensor [nao^4], chemist notation (pq|rs)
     * @param d_orbital_energies Device pointer to orbital energies [nao]
     * @param nocc Number of occupied spatial orbitals
     * @param nvir Number of virtual spatial orbitals
     * @param nao Total number of spatial orbitals (nocc + nvir)
     */
    EOMMP2Operator(const real_t* d_eri_mo,
                   const real_t* d_orbital_energies,
                   int nocc, int nvir, int nao);

    ~EOMMP2Operator();

    EOMMP2Operator(const EOMMP2Operator&) = delete;
    EOMMP2Operator& operator=(const EOMMP2Operator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "EOMMP2Operator"; }

    // --- Accessors ---
    int get_nocc() const { return nocc_; }
    int get_nvir() const { return nvir_; }
    int get_singles_dim() const { return singles_dim_; }
    int get_doubles_dim() const { return doubles_dim_; }
    const real_t* get_D1() const { return d_D1_; }
    const real_t* get_D2() const { return d_D2_; }

private:
    int nocc_;
    int nvir_;
    int nao_;
    int singles_dim_;   // nocc * nvir
    int doubles_dim_;   // nocc * nocc * nvir * nvir
    int total_dim_;     // singles_dim + doubles_dim

    // === Pre-extracted MO ERI blocks (device) ===
    real_t* d_eri_ovov_;   // (ia|jb) [nocc * nvir * nocc * nvir]
    real_t* d_eri_vvov_;   // (ab|ic) [nvir * nvir * nocc * nvir]
    real_t* d_eri_ooov_;   // (ji|kb) [nocc * nocc * nocc * nvir]
    real_t* d_eri_oooo_;   // (ij|kl) [nocc * nocc * nocc * nocc]
    real_t* d_eri_vvvv_;   // (ab|cd) [nvir * nvir * nvir * nvir]
    real_t* d_eri_oovv_;   // (ij|ab) [nocc * nocc * nvir * nvir]
    real_t* d_eri_ovvo_;   // (ia|bj) [nocc * nvir * nvir * nocc]

    // === MP1 T2 amplitudes (device) ===
    real_t* d_t2_;         // t2[i][j][a][b] = (ia|jb)/(ei+ej-ea-eb) [doubles_dim]

    // === Orbital energy denominators ===
    real_t* d_D1_;         // D1[ia] = eps_a - eps_i [singles_dim]
    real_t* d_D2_;         // D2[ijab] = eps_a + eps_b - eps_i - eps_j [doubles_dim]

    // === Fock matrix blocks (diagonal in canonical MOs) ===
    real_t* d_f_oo_;       // f_oo[i] = eps_i [nocc]
    real_t* d_f_vv_;       // f_vv[a] = eps_{a+nocc} [nvir]

    // === Diagonal for preconditioner ===
    real_t* d_diagonal_;   // [total_dim] = [D1 | D2]

    // === Workspace for apply() ===
    mutable real_t* d_work1_;  // [singles_dim] workspace
    mutable real_t* d_work2_;  // [doubles_dim] workspace

    // --- Internal build methods ---
    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_t2_and_denominators(const real_t* d_orbital_energies);
    void build_diagonal();
};

} // namespace gansu
