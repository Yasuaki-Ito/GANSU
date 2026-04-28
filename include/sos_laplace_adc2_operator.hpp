/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file sos_laplace_adc2_operator.hpp
 * @brief Laplace-SOS-ADC(2) operator — O(N⁴) sigma build
 *
 * Uses RI 3-index integrals B^P_{ia} with Laplace-transformed denominators
 * and the SOS (opposite-spin only) approximation to achieve O(N⁴) scaling.
 *
 * Sigma vector (Schur-folded, ω-dependent):
 *   σ_{ia}(ω) = M11_{ia,jb} x_{jb}
 *             + c_os Σ_τ w_τ e^{ωt_τ} Σ_P B̃^P_{ia}(τ) [Σ_Q B̃^Q_{ia}(τ) X^{PQ}(τ)]
 *   where X^{PQ}(τ) = Σ_{jb} B̃^P_{jb}(τ) B̃^Q_{jb}(τ) x_{jb} (OS structure)
 *         B̃^P_{ia}(τ) = B^P_{ia} exp(+ε_i t/2) exp(-ε_a t/2)
 *
 * Key advantages over dense ADC(2):
 *   - Memory: O(N³) instead of O(N⁴) (no M12/M21/doubles vectors)
 *   - Compute: O(N⁴) per sigma instead of O(N⁵)
 *   - GPU: DGEMM-dominated (high GPU utilization)
 */

#pragma once

#include "linear_operator.hpp"
#include <vector>

namespace gansu {

class SOSLaplaceADC2Operator : public LinearOperator {
public:
    /**
     * @brief Construct from RI B_ia^P matrix
     *
     * @param d_B_ia     Device pointer to B^P_{ia} [naux × nov], column-major
     * @param d_orbital_energies Device pointer to orbital energies [nao]
     * @param nocc       Number of occupied orbitals
     * @param nvir       Number of virtual orbitals
     * @param naux       Number of auxiliary basis functions
     * @param c_os       Opposite-spin scaling factor (default: 1.17 for ADC(2))
     * @param n_laplace  Number of Laplace quadrature points (default: 10)
     */
    /**
     * @param d_B_ia     B^P_{ia} [ov × naux] col-major (occ-virt block)
     * @param d_M11_ext  Pre-built M11 matrix [ov × ov] col-major.
     *                   If nullptr, M11 is built internally (CIS Coulomb only).
     */
    SOSLaplaceADC2Operator(
        const real_t* d_B_ia,
        const real_t* d_M11_ext,
        const real_t* d_orbital_energies,
        int nocc, int nvir, int naux,
        double c_os = 1.17,
        int n_laplace = 10);

    ~SOSLaplaceADC2Operator();

    SOSLaplaceADC2Operator(const SOSLaplaceADC2Operator&) = delete;
    SOSLaplaceADC2Operator& operator=(const SOSLaplaceADC2Operator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return singles_dim_; }
    std::string name() const override { return "SOSLaplaceADC2"; }

    // --- ω management ---
    void set_omega(real_t omega);
    real_t get_omega() const { return omega_; }
    void update_laplace_quadrature();

    // --- Accessors ---
    int get_nocc() const { return nocc_; }
    int get_nvir() const { return nvir_; }
    int get_naux() const { return naux_; }

private:
    int nocc_, nvir_, naux_;
    int singles_dim_;   // nocc * nvir

    real_t omega_ = 0.0;
    double c_os_;
    int n_laplace_;

    // Laplace quadrature (ω-dependent)
    std::vector<double> laplace_t_;   // nodes t_τ
    std::vector<double> laplace_w_;   // weights w_τ

    // RI B matrix (device, persistent)
    const real_t* d_B_ia_;    // [ov × naux], NOT owned

    // Orbital energies (host copy for Laplace scaling)
    std::vector<double> eps_occ_;  // ε_i [nocc]
    std::vector<double> eps_vir_;  // ε_a [nvir]

    // M11 matrix (device, owned) — CIS + ISR correction
    real_t* d_M11_ = nullptr;   // [singles_dim × singles_dim]

    // Diagonal for preconditioner
    real_t* d_diagonal_ = nullptr;  // [singles_dim]

    // D1 = ε_a - ε_i for preconditioner
    real_t* d_D1_ = nullptr;       // [singles_dim]

    // Device orbital energies (separated occ/vir for scaling kernel)
    real_t* d_eps_occ_dev_ = nullptr;  // [nocc]
    real_t* d_eps_vir_dev_ = nullptr;  // [nvir]

    // Workspace (mutable for const apply)
    mutable real_t* d_B_scaled_ = nullptr;    // B̃(τ) [naux × nov]
    mutable real_t* d_F_ = nullptr;           // B̃ × x [naux × nov]
    mutable real_t* d_X_PQ_ = nullptr;        // X^{PQ}(τ) [naux × naux]
    mutable real_t* d_temp_ov_aux_ = nullptr; // temp [naux × nov]

    // --- Internal methods ---
    void build_M11();
    void compute_diagonal();
    void compute_D1();
};

} // namespace gansu
