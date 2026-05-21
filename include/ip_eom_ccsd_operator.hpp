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
 * @file ip_eom_ccsd_operator.hpp
 * @brief IP-EOM-CCSD linear operator for Davidson (bt-PNO-STEOM Phase P1).
 *
 * Excitation manifold (closed-shell RHF reference):
 *   R = Σ_i r_i a_i + (1/2) Σ_{ija} r^a_{ij} a†_a a_j a_i
 *
 * Davidson solves: σ = bar H · R with eigenvalues ω = IP value (positive Ha).
 *
 * State layout (host/device row-major):
 *   total_dim = h_dim + h2p_dim
 *   h_dim     = nocc_active                                  ← 1h sector
 *   h2p_dim   = nocc_active * nocc_active * nvir             ← 2h1p sector (FULL layout per design Q3)
 *
 * The 2h1p sector keeps the full r^a_{ij} (both i<j and i>j) and is
 * anti-symmetrized on the fly inside apply(). This matches PySCF's
 * `eom_rccsd.EOMIP` layout and keeps debugging simple at the cost of 2×
 * memory in the 2h1p sector (negligible vs the n²v² EE-EOM doubles
 * sector).
 *
 * Sub-phase 1.0+1.1: apply() implements a diagonal-only matvec. Davidson
 * eigenvalues therefore equal the sorted diagonal entries — for the 1h
 * sector these are Koopmans -ε_i values, which validates the plumbing
 * end-to-end. Full bar-H matvec lands in sub-phases 1.3–1.6.
 */

#pragma once

#include <string>

#include "linear_operator.hpp"
#include "types.hpp"

namespace gansu {

class IPEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the IP-EOM-CCSD operator.
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
    IPEOMCCSDOperator(const real_t* d_eri_mo,
                      const real_t* d_orbital_energies,
                      real_t* d_t1, real_t* d_t2,
                      int nocc, int nvir, int nao);

    ~IPEOMCCSDOperator();

    IPEOMCCSDOperator(const IPEOMCCSDOperator&) = delete;
    IPEOMCCSDOperator& operator=(const IPEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "IPEOMCCSDOperator"; }

    // --- Accessors ---
    int get_nocc()    const { return nocc_; }
    int get_nvir()    const { return nvir_; }
    int get_h_dim()   const { return h_dim_; }    // 1h sector dim   = nocc
    int get_h2p_dim() const { return h2p_dim_; }  // 2h1p sector dim = nocc² · nvir (FULL layout)

    /// Print intermediate Frobenius norms (used by ip_eom_verbose ≥ 2 for
    /// PySCF cross-validation in sub-phases 1.3-1.6). Sub-phase 1.0+1.1
    /// returns the identity-stub annotation only.
    void print_intermediate_norms(std::ostream& os) const;

private:
    int nocc_;
    int nvir_;
    int nao_;
    int h_dim_;
    int h2p_dim_;
    int total_dim_;

    // === CCSD amplitudes (owned — freed in destructor) ===
    real_t* d_t1_ = nullptr;   // [nocc * nvir]
    real_t* d_t2_ = nullptr;   // [nocc² * nvir²]

    // === MO ERI blocks (subset of EE — vvvv is NOT needed by IP-EOM 2h1p) ===
    // Allocated in sub-phase 1.2 (currently constructor extracts only what
    // build_diagonal needs, which is none — diagonal uses orbital energies only).
    real_t* d_eri_oooo_ = nullptr;
    real_t* d_eri_oovv_ = nullptr;
    real_t* d_eri_ovov_ = nullptr;
    real_t* d_eri_ovvo_ = nullptr;
    real_t* d_eri_ooov_ = nullptr;
    real_t* d_eri_ovvv_ = nullptr;

    // === Dressed intermediates (IP-EOM-CCSD versions, PySCF rintermediates.py
    //  definitions; NOT the EE-EOM versions). Built in build_dressed_intermediates.
    real_t* d_Loo_    = nullptr;  // [nocc²]                Loo = cc_Foo + 2*ovoo·t1 - ovoo·t1   (PySCF Loo)
    real_t* d_Lvv_    = nullptr;  // [nvir²]                Lvv = cc_Fvv + 2*ovvv·t1 - ovvv·t1   (PySCF Lvv)
    real_t* d_Fov_    = nullptr;  // [nocc · nvir]          Fov = cc_Fov                          (PySCF cc_Fov)
    real_t* d_Woooo_  = nullptr;  // [nocc^4]               Woooo (PySCF IP version, no t1·t1 symmetrization)
    real_t* d_Wooov_  = nullptr;  // [nocc² · nocc · nvir]  Wooov = ooov + t1·ovov                (PySCF Wooov)
    real_t* d_Wovov_  = nullptr;  // [nocc · nvir · nocc · nvir]  Wovov = W1ovov + W2ovov         (PySCF Wovov, IP)
    real_t* d_Wovvo_  = nullptr;  // [nocc · nvir · nvir · nocc]  Wovvo = W1ovvo + W2ovvo         (PySCF Wovvo, IP)
    real_t* d_Wovoo_  = nullptr;  // [nocc · nvir · nocc²]  Wovoo (PySCF, used in 1h↔2h1p coupling)

    // === Diagonal & denominators ===
    real_t* d_D_h_   = nullptr;  // [nocc]            ≈ -ε_i
    real_t* d_D_h2p_ = nullptr;  // [nocc² · nvir]    ≈ -ε_i - ε_j + ε_a
    real_t* d_diagonal_ = nullptr;  // [total_dim] = [D_h | D_h2p] (used by preconditioner)
    real_t* d_f_oo_  = nullptr;  // [nocc] diagonal Fock-occ
    real_t* d_f_vv_  = nullptr;  // [nvir] diagonal Fock-vir

    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_denominators_and_fock(const real_t* d_orbital_energies);
    void build_diagonal();
    void build_dressed_intermediates();   // duplicated inline from EE-EOM (design Q2 = duplicate)
};

} // namespace gansu
