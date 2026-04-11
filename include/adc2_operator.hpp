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
 * @file adc2_operator.hpp
 * @brief ADC(2) operator using ω-dependent Schur complement
 *
 * Implements the ADC(2) effective Hamiltonian via ω-dependent Schur complement:
 *   M_eff(ω) = M11 + M12 · diag(1/(ω - D2)) · M21
 *
 * where:
 *   M11 = CIS + ISR_corr - δ_ab×Σ_oo + δ_ij×Σ_vv  (ISR-ADC(2) singles-singles)
 *   M12 = singles-doubles coupling
 *   M21 = doubles-singles coupling
 *   D2  = orbital energy denominators (eps_a + eps_b - eps_i - eps_j), diagonal
 *
 * Key properties (verified against PySCF):
 *   - M_eff(ω) IS symmetric for any ω — standard symmetric eigensolvers apply
 *   - The full ADC(2) matrix is non-symmetric (M12 ≠ M21^T)
 *   - Static Schur complement (ω=0) is approximate (~0.005-0.02 Ha error)
 *   - ω-dependent Schur complement converges in 5-8 iterations to machine precision
 *
 * Two solve paths:
 *   1. Direct: build_M_eff_matrix(ω) + eigvalsh (for stored ERI / small molecules)
 *   2. Davidson: apply() uses ω-dependent operator (for future large-molecule support)
 *
 * Formulas verified numerically against PySCF EE-ADC(2).
 */

#pragma once

#include "linear_operator.hpp"

namespace gansu {

class ADC2Operator : public LinearOperator {
public:
    /**
     * @brief Construct ADC(2) operator
     *
     * Precomputes M11 (dense), M12 (dense), M21 (dense), T2, D2.
     *
     * @param d_eri_mo Device pointer to full MO ERI tensor [nao^4], chemist notation (pq|rs)
     * @param d_orbital_energies Device pointer to orbital energies [nao]
     * @param nocc Number of occupied spatial orbitals
     * @param nvir Number of virtual spatial orbitals
     * @param nao Total number of spatial orbitals (nocc + nvir)
     */
    ADC2Operator(const real_t* d_eri_mo,
                 const real_t* d_orbital_energies,
                 int nocc, int nvir, int nao,
                 bool is_triplet = false);

    /**
     * @brief Construct ADC(2) operator from pre-built MO ERI sub-blocks
     *
     * No full MO ERI (nao⁴) needed. Used when nao⁴ doesn't fit in GPU memory.
     */
    ADC2Operator(const real_t* d_eri_ovov,
                 const real_t* d_eri_vvov,
                 const real_t* d_eri_ooov,
                 const real_t* d_eri_oovv,
                 const real_t* d_orbital_energies,
                 int nocc, int nvir, int nao,
                 bool is_triplet = false);

    ~ADC2Operator();

    ADC2Operator(const ADC2Operator&) = delete;
    ADC2Operator& operator=(const ADC2Operator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return singles_dim_; }
    std::string name() const override { return "ADC2Operator"; }

    // --- ω management ---
    void set_omega(real_t omega) { omega_ = omega; }
    void recompute_diagonal() { compute_diagonal(); }
    real_t get_omega() const { return omega_; }

    /**
     * @brief Recompute diagonal for current omega value
     *
     * Must be called after set_omega() to update the preconditioner diagonal.
     * For kernel path: computes exact M_eff(ω) diagonal including Schur complement.
     * For dense path: computes M_eff(ω) diagonal via M11 + M12·diag(1/(ω-D2))·M21.
     */
    void update_diagonal();

    /**
     * @brief Build M_eff(ω) as a dense matrix for direct diagonalization
     *
     * M_eff(ω) = M11 + M12 · diag(1/(ω - D2)) · M21
     *
     * @param omega The ω value for the Schur complement
     * @param d_M_eff Device pointer to output matrix [singles_dim × singles_dim], column-major
     */
    void build_M_eff_matrix(real_t omega, real_t* d_M_eff) const;

    /**
     * @brief Validate on-the-fly kernel against dense DGEMM path
     *
     * Only works when use_dense_M12_ is true.
     * Computes M_eff via both paths and reports max difference.
     */
    void validate_onthefly_vs_dense(real_t omega) const;

    // --- Accessors ---
    int get_nocc() const { return nocc_; }
    int get_nvir() const { return nvir_; }
    int get_singles_dim() const { return singles_dim_; }
    int get_doubles_dim() const { return doubles_dim_; }
    const real_t* get_D1() const { return d_D1_; }
    const real_t* get_D2() const { return d_D2_; }
    const real_t* get_M11() const { return d_M11_; }
    const real_t* get_eri_ovov() const { return d_eri_ovov_; }
    const real_t* get_eri_vvov() const { return d_eri_vvov_; }
    const real_t* get_eri_ooov() const { return d_eri_ooov_; }
    bool is_dense_M12() const { return use_dense_M12_; }
    const real_t* get_M12() const { return d_M12_; }
    const real_t* get_M21() const { return d_M21_; }

private:
    int nocc_;
    int nvir_;
    int nao_;
    int singles_dim_;   // nocc * nvir
    int doubles_dim_;   // nocc * nocc * nvir * nvir

    real_t omega_ = 0.0;
    bool is_triplet_ = false;    // true for triplet excited states
    bool use_dense_M12_ = true;  // false → on-the-fly M_eff (no M12/M21 stored)

    // === Pre-extracted MO ERI blocks (device) ===
    real_t* d_eri_ovov_;   // (ia|jb) [nocc * nvir * nocc * nvir]
    real_t* d_eri_vvov_;   // (ab|ic) [nvir * nvir * nocc * nvir]
    real_t* d_eri_ooov_;   // (ji|kb) [nocc * nocc * nocc * nvir]

    // === MP1 T2 amplitudes (device) ===
    real_t* d_t2_;         // t2[i][j][a][b] = (ia|jb)/(ei+ej-ea-eb) [doubles_dim]

    // === Dense matrices (device, column-major) ===
    real_t* d_M11_;        // CIS + T2 correction [singles_dim × singles_dim]
    real_t* d_M12_;        // singles-doubles coupling [singles_dim × doubles_dim]
    real_t* d_M21_;        // doubles-singles coupling [doubles_dim × singles_dim]

    // === Orbital energy denominators ===
    real_t* d_D2_;         // D2[ijab] = eps_a + eps_b - eps_i - eps_j [doubles_dim]
    real_t* d_D1_;         // D1[ia] = eps_a - eps_i [singles_dim]

    // === Workspace (mutable for const apply) ===
    mutable real_t* d_scaled_M21_;    // [doubles_dim × singles_dim] for build_M_eff
    mutable real_t* d_temp_doubles_;  // [doubles_dim] for apply()

    // === Diagonal for preconditioner ===
    real_t* d_diagonal_;   // [singles_dim]

    // --- Internal build methods ---
    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_mp1_t2_and_D2(const real_t* d_orbital_energies);
    void compute_D1(const real_t* d_orbital_energies);
    void build_M11(const real_t* d_eri_mo, const real_t* d_orbital_energies);
    void build_M11_from_blocks(const real_t* d_eri_ovov, const real_t* d_eri_oovv,
                               const real_t* d_orbital_energies);
    void build_M12_M21();
    void compute_diagonal();
};

} // namespace gansu
