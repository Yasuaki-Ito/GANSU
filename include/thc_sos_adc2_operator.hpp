/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_sos_adc2_operator.hpp
 * @brief THC-SOS-ADC(2) Schur-folded sigma operator (Phase 2.2a, MVP).
 *
 * Replaces the RI 3-index tensor B^P_{ia} of SOSLaplaceADC2Operator with the
 * THC factorisation
 *
 *   (ia|jb) ~= sum_{P,Q} X^P_i X^P_a Z_{PQ} X^Q_j X^Q_b
 *
 * keeping the ω-shifted Laplace + opposite-spin approximation.
 *
 * Scaling per σ build:  O(N_g^3) per Laplace point (same as Phase 2.1 SOS-MP2).
 *
 * Phase 2.2a MVP scope:
 *   - Closed-shell RHF
 *   - Coulomb-only OS Schur correction (drops B3-exchange / A3-Coulomb)
 *   - M11 = CIS Hessian = D1 + 2(ia|jb) - (ij|ab)  (no ISR self-energy yet)
 *
 * Phase 2.2b will add B3 and A3 corrections; Phase 2.2c multi-GPU.
 */

#pragma once

#include <vector>
#include "types.hpp"
#include "linear_operator.hpp"
#ifndef GANSU_CPU_ONLY
#include "device_host_memory.hpp"
#endif

namespace gansu {

class THCSOSADC2Operator : public LinearOperator {
public:
    /**
     * @brief Construct from THC factors in MO basis.
     *
     * @param d_X_mo     [N_orb x N_g] column-major MO collocation on device.
     *                   Owned by caller; lifetime must outlast the operator.
     * @param d_Z        [N_g x N_g] column-major LS-THC core matrix on device.
     * @param d_orbital_energies  Orbital energies [N_orb] on device.
     * @param n_occ, n_vir, N_orb (= n_occ + n_vir), N_g.
     * @param c_os       Opposite-spin scaling factor (1.17 for SOS-ADC(2)
     *                   per Hellweg-Grun-Hattig).
     * @param n_laplace  Number of Laplace quadrature points.
     */
    THCSOSADC2Operator(const real_t* d_X_mo,
                       const real_t* d_Z,
                       const real_t* d_orbital_energies,
                       int n_occ, int n_vir, int N_orb, int N_g,
                       double c_os = 1.17,
                       int n_laplace = 10,
                       int num_gpus = 1,
                       bool enable_b3a3 = true,
                       bool enable_b3 = true,
                       bool enable_a3 = true);

    ~THCSOSADC2Operator();

    THCSOSADC2Operator(const THCSOSADC2Operator&) = delete;
    THCSOSADC2Operator& operator=(const THCSOSADC2Operator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return singles_dim_; }
    std::string name() const override { return "THCSOSADC2"; }

    // --- ω management ---
    void set_omega(real_t omega);
    real_t get_omega() const { return omega_; }
    void update_laplace_quadrature();

    // --- Accessors ---
    int get_nocc() const { return n_occ_; }
    int get_nvir() const { return n_vir_; }
    int get_N_g() const { return N_g_; }

private:
    void build_M11();        // CIS Hessian via THC
    void compute_D1();       // ε_a - ε_i diagonal

    // Dimensions
    int n_occ_, n_vir_, N_orb_, N_g_;
    int singles_dim_;        // n_occ * n_vir

    // Parameters
    real_t omega_ = 0.0;
    double c_os_;
    int n_laplace_;
    bool enable_b3a3_ = true;
    bool enable_b3_ = true;
    bool enable_a3_ = true;

    // Laplace quadrature (ω-dependent, regenerated each set_omega())
    std::vector<double> laplace_t_;
    std::vector<double> laplace_w_;

    // External (not owned) THC factors and orbital energies
    const real_t* d_X_mo_;          // [N_orb x N_g]
    const real_t* d_Z_;             // [N_g x N_g]
    const real_t* d_orbital_energies_;  // [N_orb]

    // Host copies of orbital energies for Laplace range
    std::vector<double> eps_h_;     // [N_orb]

    // Owned device buffers
    real_t* d_M11_ = nullptr;       // [singles_dim x singles_dim]
    real_t* d_D1_ = nullptr;        // [singles_dim], = ε_a - ε_i for each (i,a)
    real_t* d_diagonal_ = nullptr;  // [singles_dim] preconditioner diagonal

    // Multi-GPU resources: per-device replicas + workspaces.
    // For num_gpus_=1 these vectors are length 1 and aliasing the caller's
    // const inputs (no allocation).  For num_gpus_>1, peer GPUs hold their
    // own replicas of X_mo / Z / eps and dedicated workspaces.
    int num_gpus_ = 1;

    std::vector<real_t*> d_X_mo_per_;   // [num_gpus] X_mo replicas
    std::vector<real_t*> d_Z_per_;      // [num_gpus] Z replicas
    std::vector<real_t*> d_eps_per_;    // [num_gpus] eps replicas

    std::vector<real_t*> d_X_occ_t_per_;
    std::vector<real_t*> d_X_vir_t_per_;
    std::vector<real_t*> d_M_per_;
    std::vector<real_t*> d_F_per_;
    std::vector<real_t*> d_Y_PQ_per_;
    std::vector<real_t*> d_ZY_per_;
    std::vector<real_t*> d_T_per_;
    std::vector<real_t*> d_MT_per_;

    // B3/A3 workspaces (allocated only when enable_b3a3_).  All sized per-GPU.
    //
    //   d_Z_occ_per_ : (N_g x N_g)   = X̃_occ^T X̃_occ        (also G̃ for B3 Hadamard)
    //   d_W_per_     : (N_g x N_g)   = Z̃_occ ⊙ Z̃_vir, then reused as z_A3 → V
    //   d_U_per_     : (N_g x N_g)   = Z W Z^T, then reused as N_B3
    //   d_yB3_per_   : (N_g x n_vir) = X_occ_unscaled^T x       (shared by A3 & B3)
    //   d_X2vir_per_ : (n_vir x N_g) = X_vir × exp(-εC t)       (full-t)
    //   d_X2occ_per_ : (n_occ x N_g) = X_occ × exp(+εK t)       (full-t)
    //   d_tmp1_per_  : (n_vir x N_g) = X_vir_unscaled · V         (A3 step)
    //   d_tmpB3_per_ : (n_occ x N_g) = X_occ_unscaled · N_B3      (B3 step)
    //   d_sig_corr_per_ : (ov)       = scratch for one B3 or A3 contribution
    std::vector<real_t*> d_Z_occ_per_;
    std::vector<real_t*> d_W_per_;
    std::vector<real_t*> d_U_per_;
    std::vector<real_t*> d_yB3_per_;
    std::vector<real_t*> d_X2vir_per_;
    std::vector<real_t*> d_X2occ_per_;
    std::vector<real_t*> d_tmp1_per_;
    std::vector<real_t*> d_tmpB3_per_;
    mutable std::vector<real_t*> d_sig_corr_per_;

    mutable std::vector<real_t*> d_input_per_;          // trial vector replicas (per GPU)
    mutable std::vector<real_t*> d_sigma_partial_per_;  // per-GPU σ contribution
};

} // namespace gansu
