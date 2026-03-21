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
 * @file eom_ccsd_operator.hpp
 * @brief EOM-CCSD operator in the full singles+doubles space
 *
 * Implements σ = H_EOM-CCSD × R for RHF reference.
 * R = [R1(ov) | R2(oovv)], σ = [σ1(ov) | σ2(oovv)]
 *
 * σ2 follows PySCF's eeccsd_matvec_singlet algorithm:
 *   1. Precompute dressed intermediates (Foo, Fvv, Woooo, WoVVo, WoVvO,
 *      woOoV, woVoO, wvOvV) from T1, T2, and ERI blocks
 *   2. Compute "half" σ2 using these intermediates
 *   3. Symmetrize: σ2[ijab] = half[ijab] + half[jiba]
 *
 * σ1 uses 8 terms with dressed intermediates (Foo, Fvv, Fov, WoVVo,
 *   WoVvO, woOoV) and bare ERIs (ovov, ovvv).
 */

#pragma once

#include "linear_operator.hpp"

namespace gansu {

class EOMCCSDOperator : public LinearOperator {
public:
    EOMCCSDOperator(const real_t* d_eri_mo,
                    const real_t* d_orbital_energies,
                    real_t* d_t1, real_t* d_t2,
                    int nocc, int nvir, int nao);

    ~EOMCCSDOperator();

    EOMCCSDOperator(const EOMCCSDOperator&) = delete;
    EOMCCSDOperator& operator=(const EOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "EOMCCSDOperator"; }

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

    // === CCSD amplitudes (owned) ===
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
    real_t* d_eri_ovvv_;   // (ia|bc) [nocc * nvir^3]

    // === Dressed intermediates for σ1 and σ2 (device, precomputed in constructor) ===
    real_t* d_Foo_;        // [nocc * nocc]
    real_t* d_Fvv_;        // [nvir * nvir]
    real_t* d_Fov_;        // [nocc * nvir]
    real_t* d_Woooo_;      // [nocc^4]
    real_t* d_WoVVo_;      // [nocc * nvir * nvir * nocc]
    real_t* d_WoVvO_;      // [nocc * nvir * nvir * nocc]
    real_t* d_woOoV_;      // [nocc * nocc * nocc * nvir]
    real_t* d_woVoO_;      // [nocc * nvir * nocc * nocc]
    real_t* d_wvOvV_;      // [nvir * nocc * nvir * nvir]

    // === Workspace for half-σ2 (mutable for use in const apply()) ===
    mutable real_t* d_half_sigma2_;  // [doubles_dim]

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
    void build_dressed_intermediates();
};

} // namespace gansu
