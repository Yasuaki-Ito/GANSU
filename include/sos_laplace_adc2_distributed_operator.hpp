/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file sos_laplace_adc2_distributed_operator.hpp
 * @brief Multi-GPU distributed SOS-Laplace-ADC(2) operator
 *
 * Distributes the Laplace quadrature loop across GPUs.
 * After AllGather of B_ia to B_ia_full on each GPU, each GPU
 * independently computes sigma contributions for its assigned
 * Laplace points, then AllReduce(sigma_schur).
 *
 * Memory per GPU: ~3 × ov × naux (B_ia_full + B_scaled + F + temp)
 *                 + naux² (X_PQ)
 * Communication:  AllGather B_ia (once), AllReduce sigma (ov doubles/apply)
 * Speedup:        ~linear in num_gpus for the O(N⁴) Laplace loop
 */

#pragma once

#ifdef GANSU_MULTI_GPU

#include "linear_operator.hpp"
#include <vector>

namespace gansu {

class SOSLaplaceADC2DistributedOperator : public LinearOperator {
public:
    /**
     * @param num_gpus        Number of GPUs
     * @param d_B_ia_local    Per-GPU B_ia^P [ov × naux_local[d]], NOT owned
     * @param naux_local      Per-GPU local auxiliary count
     * @param d_M11           M11 [ov × ov] on GPU 0, NOT owned (copied internally)
     * @param d_orbital_energies  [nocc+nvir] on GPU 0
     * @param nocc, nvir      Occupied / virtual orbital counts
     * @param naux_total      Total number of auxiliary basis functions
     * @param c_os            Opposite-spin scaling factor (default: 1.17)
     * @param n_laplace       Number of Laplace quadrature points (default: 10)
     */
    SOSLaplaceADC2DistributedOperator(
        int num_gpus,
        const std::vector<real_t*>& d_B_ia_local,
        const std::vector<int>& naux_local,
        const real_t* d_M11,
        const real_t* d_orbital_energies,
        int nocc, int nvir, int naux_total,
        double c_os = 1.17,
        int n_laplace = 10);

    ~SOSLaplaceADC2DistributedOperator();

    SOSLaplaceADC2DistributedOperator(const SOSLaplaceADC2DistributedOperator&) = delete;
    SOSLaplaceADC2DistributedOperator& operator=(const SOSLaplaceADC2DistributedOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return ov_; }
    std::string name() const override { return "SOSLaplaceADC2Distributed"; }

    // --- ω management ---
    void set_omega(real_t omega);
    real_t get_omega() const { return omega_; }
    void update_laplace_quadrature();

private:
    int num_gpus_;
    int nocc_, nvir_, naux_;
    int ov_;
    real_t omega_ = 0.0;
    double c_os_;
    int n_laplace_;

    // Laplace quadrature (ω-dependent)
    std::vector<double> laplace_t_;
    std::vector<double> laplace_w_;

    // Orbital energies (host)
    std::vector<double> eps_occ_;
    std::vector<double> eps_vir_;

    // Per-GPU workspace (owned)
    struct PerGpuWorkspace {
        real_t* d_B_ia_full  = nullptr;  // AllGathered [ov × naux]
        real_t* d_x          = nullptr;  // replicated input [ov]
        real_t* d_B_scaled   = nullptr;  // B̃(τ) [ov × naux]
        real_t* d_F          = nullptr;  // F = B̃ × x [ov × naux]
        real_t* d_X_PQ       = nullptr;  // X^{PQ}(τ) [naux × naux]
        real_t* d_temp       = nullptr;  // temp [ov × naux]
        real_t* d_sigma_local = nullptr; // partial sigma [ov]
        real_t* d_eps_occ    = nullptr;  // [nocc]
        real_t* d_eps_vir    = nullptr;  // [nvir]
    };
    std::vector<PerGpuWorkspace> ws_;

    // GPU 0 only (owned)
    real_t* d_M11_      = nullptr;
    real_t* d_D1_       = nullptr;
    real_t* d_diagonal_ = nullptr;

    void compute_D1();
    void compute_diagonal();
    void allgather_B_ia(const std::vector<real_t*>& d_B_ia_local,
                        const std::vector<int>& naux_local);
    void allocate_workspace();
    void free_workspace();
};

} // namespace gansu

#endif // GANSU_MULTI_GPU
