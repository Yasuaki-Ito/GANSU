/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ri_adc2_schur_operator.hpp
 * @brief RI-factored ADC(2) Schur complement operator — exact M12·(ω-D2)⁻¹·M21
 *
 * Computes the full ADC(2) effective Hamiltonian sigma vector:
 *   σ(ω) = M11·x + Σ_{IJCD} M12[·,IJCD] × 1/(ω-D2) × M21[IJCD,·] × x
 *
 * Uses only RI 3-index integrals B_ia^P (OV), B_ab^P (VV), B_ij^P (OO).
 * No nao⁴ MO-ERI storage. Memory: O(N³), Cost: O(N⁵) per sigma.
 *
 * The 8 cross-terms (from M12 groups A,B × M21 terms 1-4) are computed
 * in a batch loop over occupied pairs (I,J), using DGEMM intermediates
 * of size O(naux × nvir) per batch.
 *
 * M12[KE, IJCD]:
 *   δ_{IK}: +2(EC|JD) - (DE|JC)    [Group A, VVOV]
 *   δ_{CE}: +(JK|ID) - 2(IK|JD)    [Group B, OOOV]
 *
 * M21[IJCD, LF]:
 *   δ_{LI}: +(FC|JD)               [Term 1, VVOV]
 *   δ_{LJ}: +(FD|IC)               [Term 2, VVOV]
 *   δ_{FC}: -(IL|JD)               [Term 3, OOOV]
 *   δ_{FD}: -(JL|IC)               [Term 4, OOOV]
 *
 * In RI:
 *   (EC|JD) = Σ_P B_EC^P × B_JD^P   (VV × OV)
 *   (JK|ID) = Σ_P B_JK^P × B_ID^P   (OO × OV)
 */

#pragma once

#include "linear_operator.hpp"
#include <vector>

namespace gansu {

class RIADC2SchurOperator : public LinearOperator {
public:
    /**
     * @param d_B_ia  B_ia^P [ov × naux] col-major, NOT owned
     * @param d_B_ab  B_ab^P [vv × naux] col-major, NOT owned
     * @param d_B_ij  B_ij^P [oo × naux] col-major, NOT owned
     * @param d_M11   Pre-built M11 [ov × ov] col-major, NOT owned (copied internally)
     * @param d_orbital_energies  [nocc+nvir] on device
     */
    RIADC2SchurOperator(
        const real_t* d_B_ia,
        const real_t* d_B_ab,
        const real_t* d_B_ij,
        const real_t* d_M11,
        const real_t* d_orbital_energies,
        int nocc, int nvir, int naux);

    ~RIADC2SchurOperator();

    RIADC2SchurOperator(const RIADC2SchurOperator&) = delete;
    RIADC2SchurOperator& operator=(const RIADC2SchurOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return nocc_ * nvir_; }
    std::string name() const override { return "RIADC2Schur"; }

    // --- ω management ---
    void set_omega(real_t omega) { omega_ = omega; }
    real_t get_omega() const { return omega_; }

private:
    int nocc_, nvir_, naux_;
    int ov_, vv_, oo_;
    real_t omega_ = 0.0;

    // RI 3-index blocks (device, NOT owned)
    const real_t* d_B_ia_;   // [ov × naux]
    const real_t* d_B_ab_;   // [vv × naux]
    const real_t* d_B_ij_;   // [oo × naux]

    // M11 matrix (device, owned copy)
    real_t* d_M11_ = nullptr;   // [ov × ov]

    // Orbital energies (host)
    std::vector<double> eps_occ_;  // [nocc]
    std::vector<double> eps_vir_;  // [nvir]

    // Diagonal / preconditioner
    real_t* d_D1_ = nullptr;       // [ov]
    real_t* d_diagonal_ = nullptr; // [ov]

    // Device orbital energies for kernels
    real_t* d_eps_occ_dev_ = nullptr;   // [nocc]
    real_t* d_eps_vir_dev_ = nullptr;   // [nvir]

    // Precomputed intermediates (built in apply, size [nvir × nocc × naux])
    mutable real_t* d_phi_ = nullptr;   // φ^P_C(I) = Σ_F B_ab^P[C,F]×x[I,F]
    mutable real_t* d_psi_ = nullptr;   // ψ^P_C(I) = Σ_L B_ij^P[I,L]×x[L,C]

    // Per-J workspace (I loop eliminated)
    mutable real_t* d_alpha_all_ = nullptr; // [ov × naux] φ - ψ for all I
    mutable real_t* d_beta_ = nullptr;      // [nvir × naux] φ_J - ψ_J
    mutable real_t* d_R_all_ = nullptr;     // [ov × nvir] R for all I
    mutable real_t* d_W_all_ = nullptr;     // [ov × nvir] weighted R for all I
    mutable real_t* d_eri_vvov_ = nullptr;  // [vv × nvir] (FC|JD) block
    mutable real_t* d_ooov1_all_ = nullptr; // [nocc × ov] (JK|ID) for all I
    mutable real_t* d_ooov2_all_ = nullptr; // [oo × nvir] (IK|JD) for all I

    void compute_D1();
    void compute_diagonal();

public:
    /**
     * @brief Build M11 matrix entirely from RI 3-index integrals (no nao⁴ MO-ERI)
     *
     * M11 = CIS + ISR_corr - δ_ab Σ_oo + δ_ij Σ_vv
     *   CIS[ia,jb] = δ_ij δ_ab (ε_a-ε_i) + 2(ia|jb) - (ij|ab)
     *   (ia|jb) = Σ_P B_ia^P B_jb^P,  (ij|ab) = Σ_P B_ij^P B_ab^P
     *   t2[ijab] = (ia|jb) / (εi+εj-εa-εb)
     *   ISR, Σ_oo, Σ_vv: contractions of t2 × (ia|jb)
     *
     * @param d_M11_out  Output [ov × ov] col-major, device (caller must allocate)
     */
    static void build_M11_from_RI(
        real_t* d_M11_out,
        const real_t* d_B_ia, const real_t* d_B_ab, const real_t* d_B_ij,
        const real_t* d_orbital_energies,
        int nocc, int nvir, int naux);
};

} // namespace gansu
