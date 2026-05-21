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
 * @file ea_eom_ccsd_operator.hpp
 * @brief EA-EOM-CCSD linear operator for Davidson (bt-PNO-STEOM Phase P2).
 *
 * Excitation manifold (closed-shell RHF reference):
 *   R = Σ_a r_a a†_a + (1/2) Σ_{abi} r^{ab}_i a†_a a†_b a_i
 *
 * Davidson solves: σ = bar H · R with eigenvalues ω = EA value (positive Ha).
 *
 * State layout (host/device row-major):
 *   total_dim = p_dim + p2h_dim
 *   p_dim     = nvir                                  ← 1p sector
 *   p2h_dim   = nocc_active * nvir * nvir             ← 2p1h sector (FULL layout)
 *
 * The 2p1h sector keeps the full r^{ab}_i (both a<b and a>b) and is
 * anti-symmetrized on the fly inside apply(). This matches PySCF's
 * `eom_rccsd.EOMEA` layout and keeps debugging simple at the cost of 2×
 * memory in the 2p1h sector (negligible vs the n²v² EE-EOM doubles sector).
 *
 * Sub-phase 2.0+2.1: apply() implements a diagonal-only matvec. Davidson
 * eigenvalues therefore equal the sorted diagonal entries — for the 1p
 * sector these are Koopmans +ε_a values (the EA Koopmans estimates), which
 * validates the plumbing end-to-end. Full bar-H matvec lands in sub-phases
 * 2.3-2.6.
 */

#pragma once

#include <string>

#include "linear_operator.hpp"
#include "types.hpp"

namespace gansu {

class EAEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the EA-EOM-CCSD operator.
     *
     * @param d_eri_mo       Device pointer to active-space MO ERIs [nao_active^4]
     *                       (chemist notation; trimmed if frozen_core was used).
     * @param d_orbital_energies Device pointer to active-space orbital energies [nao_active].
     * @param d_t1           Device pointer to CCSD T1 [nocc_active * nvir]
     *                       (ownership transferred to operator, freed in destructor).
     * @param d_t2           Device pointer to CCSD T2 [nocc_active² * nvir²]
     *                       (ownership transferred).
     * @param nocc           Active-occupied count (= full_occ - num_frozen).
     * @param nvir           Virtual count.
     * @param nao            Active-space basis count (= nocc + nvir).
     */
    EAEOMCCSDOperator(const real_t* d_eri_mo,
                      const real_t* d_orbital_energies,
                      real_t* d_t1, real_t* d_t2,
                      int nocc, int nvir, int nao);

    ~EAEOMCCSDOperator();

    EAEOMCCSDOperator(const EAEOMCCSDOperator&) = delete;
    EAEOMCCSDOperator& operator=(const EAEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "EAEOMCCSDOperator"; }

    // --- Accessors ---
    int get_nocc()    const { return nocc_; }
    int get_nvir()    const { return nvir_; }
    int get_p_dim()   const { return p_dim_; }    // 1p sector dim   = nvir
    int get_p2h_dim() const { return p2h_dim_; }  // 2p1h sector dim = nocc · nvir² (FULL layout)

    /// Print intermediate Frobenius norms (used by ea_eom_verbose ≥ 2 for
    /// PySCF cross-validation in sub-phases 2.3-2.6). Sub-phase 2.0+2.1
    /// returns the identity-stub annotation only.
    void print_intermediate_norms(std::ostream& os) const;

private:
    int nocc_;
    int nvir_;
    int nao_;
    int p_dim_;
    int p2h_dim_;
    int total_dim_;

    // === CCSD amplitudes (owned — freed in destructor) ===
    real_t* d_t1_ = nullptr;   // [nocc * nvir]
    real_t* d_t2_ = nullptr;   // [nocc² * nvir²]

    // === MO ERI blocks (subset of EE — EA needs vvvv + the standard 2-virtual blocks) ===
    // Allocated in sub-phase 2.2 (currently constructor extracts none —
    // diagonal uses orbital energies only).
    real_t* d_eri_oooo_ = nullptr;
    real_t* d_eri_ooov_ = nullptr;  // used by Wvvvo via (lj|kc) = ovoo identity
    real_t* d_eri_oovv_ = nullptr;
    real_t* d_eri_ovov_ = nullptr;
    real_t* d_eri_ovvo_ = nullptr;
    real_t* d_eri_ovvv_ = nullptr;
    real_t* d_eri_vvvv_ = nullptr;  // ★ EA-only: needed for Wvvvv build

    // === Dressed intermediates (EA-EOM-CCSD versions, PySCF rintermediates.py
    //  definitions). Loo/Lvv/Fov/Wovov/Wovvo are shared with IP-EOM (P1);
    //  Wvovv/Wvvvv/Wvvvo are EA-specific.  Built in build_dressed_intermediates.
    real_t* d_Loo_    = nullptr;  // [nocc²]
    real_t* d_Lvv_    = nullptr;  // [nvir²]
    real_t* d_Fov_    = nullptr;  // [nocc · nvir]
    real_t* d_Wovov_  = nullptr;  // [nocc · nvir · nocc · nvir]
    real_t* d_Wovvo_  = nullptr;  // [nocc · nvir · nvir · nocc]
    real_t* d_Wvovv_  = nullptr;  // [nvir · nocc · nvir · nvir]
    real_t* d_Wvvvv_  = nullptr;  // [nvir^4]
    real_t* d_Wvvvo_  = nullptr;  // [nvir · nvir · nvir · nocc]

    // === Diagonal & denominators ===
    real_t* d_D_p_   = nullptr;  // [nvir]            ≈ +ε_a
    real_t* d_D_p2h_ = nullptr;  // [nocc · nvir²]    ≈ -ε_j + ε_a + ε_b
    real_t* d_diagonal_ = nullptr;  // [total_dim] = [D_p | D_p2h] (used by preconditioner)
    real_t* d_f_oo_  = nullptr;  // [nocc] diagonal Fock-occ
    real_t* d_f_vv_  = nullptr;  // [nvir] diagonal Fock-vir

    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_denominators_and_fock(const real_t* d_orbital_energies);
    void build_diagonal();
    void build_dressed_intermediates();   // PySCF EA intermediates — placeholder in 2.0+2.1, body in 2.2
};

} // namespace gansu
