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
 * @file eom_cc2_operator.hpp
 * @brief EOM-CC2 operator in the full singles+doubles space
 *
 * Implements σ = H_EOM-CC2 × R for RHF reference.
 * R = [R1(ov) | R2(oovv)], σ = [σ1(ov) | σ2(oovv)]
 *
 * Key properties:
 *   - Uses CC2 ground state T1 + T2 amplitudes (not just MP1 T2)
 *   - σ1 has 20 einsum terms (÷2 → 11 grouped terms)
 *   - σ2 has 14 einsum terms (÷2 → 8 grouped terms)
 *   - M22 (doubles-doubles block) is EXACTLY diagonal
 *     (only Fock terms, no ERI×R2 or T2×R2 coupling)
 *   - No null space problem (unlike EOM-MP2)
 *   - Schur complement is EXACT (not approximate)
 *
 * The full matrix is non-symmetric (EOM equations are not Hermitian).
 */

#pragma once

#include "linear_operator.hpp"

namespace gansu {

class EOMCC2Operator : public LinearOperator {
public:
    /**
     * @brief Construct EOM-CC2 operator
     *
     * Takes ownership of T1, T2 pointers (will free them in destructor).
     * Extracts needed ERI blocks from full MO ERI tensor.
     *
     * @param d_eri_mo Device pointer to full MO ERI tensor [nao^4], chemist notation (pq|rs)
     * @param d_orbital_energies Device pointer to orbital energies [nao]
     * @param d_t1 Device pointer to CC2 T1 amplitudes [nocc*nvir] (takes ownership)
     * @param d_t2 Device pointer to CC2 T2 amplitudes [nocc^2*nvir^2] (takes ownership)
     * @param nocc Number of occupied spatial orbitals
     * @param nvir Number of virtual spatial orbitals
     * @param nao Total number of spatial orbitals (nocc + nvir)
     */
    EOMCC2Operator(const real_t* d_eri_mo,
                   const real_t* d_orbital_energies,
                   real_t* d_t1, real_t* d_t2,
                   int nocc, int nvir, int nao);

    ~EOMCC2Operator();

    EOMCC2Operator(const EOMCC2Operator&) = delete;
    EOMCC2Operator& operator=(const EOMCC2Operator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "EOMCC2Operator"; }

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

    // === CC2 amplitudes (owned) ===
    real_t* d_t1_;     // [singles_dim]
    real_t* d_t2_;     // [doubles_dim]

    // === Pre-extracted MO ERI blocks (device) ===
    real_t* d_eri_ovov_;   // (ia|jb) [nocc * nvir * nocc * nvir]
    real_t* d_eri_vvov_;   // (ab|ic) [nvir * nvir * nocc * nvir]
    real_t* d_eri_ooov_;   // (ji|kb) [nocc * nocc * nocc * nvir]
    real_t* d_eri_oooo_;   // (ij|kl) [nocc^4]
    real_t* d_eri_vvvv_;   // (ab|cd) [nvir^4]
    real_t* d_eri_oovv_;   // (ij|ab) [nocc^2 * nvir^2]
    real_t* d_eri_ovvo_;   // (ia|bj) [nocc * nvir * nvir * nocc]

    // === Orbital energy denominators ===
    real_t* d_D1_;         // D1[ia] = eps_a - eps_i [singles_dim]
    real_t* d_D2_;         // D2[ijab] = eps_a + eps_b - eps_i - eps_j [doubles_dim]

    // === Fock matrix blocks (diagonal in canonical MOs) ===
    real_t* d_f_oo_;       // f_oo[i] = eps_i [nocc]
    real_t* d_f_vv_;       // f_vv[a] = eps_{a+nocc} [nvir]

    // === Diagonal for preconditioner ===
    real_t* d_diagonal_;   // [total_dim] = [D1 | D2]

    // --- Internal build methods ---
    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_denominators_and_fock(const real_t* d_orbital_energies);
    void build_diagonal();
};

} // namespace gansu
