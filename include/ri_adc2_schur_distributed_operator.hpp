/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ri_adc2_schur_distributed_operator.hpp
 * @brief Multi-GPU distributed RI-ADC(2) Schur complement operator
 *
 * Distributes the RI 3-index tensors B_ia, B_ab, B_ij across GPUs along
 * the auxiliary (P) axis.  Each GPU holds B_*_local [dim × naux_local].
 *
 * sigma(ω) = M11·x + Σ_J { Group-A + Group-B contractions }
 *
 * In the J loop, DGEMMs that contract over the P axis produce partial sums
 * on each GPU.  These are AllReduced before the nonlinear kernel steps
 * (weight division, Group-A/B accumulation) which run on GPU 0.
 *
 * AllReduce pattern per J iteration:
 *   (1) eri_vvov [vv×nvir], R_all [ov×nvir]        — after VVOV+R DGEMMs
 *   (2) ooov1 [nocc×ov], ooov2 [oo×nvir]           — after OOOV DGEMMs
 * Communication volume is O(N³) per J, negligible on NVLink.
 */

#pragma once

#ifdef GANSU_MULTI_GPU

#include "linear_operator.hpp"
#include <vector>

namespace gansu {

class RIADC2SchurDistributedOperator : public LinearOperator {
public:
    /**
     * @param num_gpus        Number of GPUs
     * @param d_B_ia_local    Per-GPU B_ia^P [ov × naux_local[d]], NOT owned
     * @param d_B_ab_local    Per-GPU B_ab^P [vv × naux_local[d]], NOT owned
     * @param d_B_ij_local    Per-GPU B_ij^P [oo × naux_local[d]], NOT owned
     * @param naux_local      Per-GPU local auxiliary count
     * @param d_M11           M11 [ov × ov] on GPU 0, NOT owned (copied internally)
     * @param d_orbital_energies  [nocc+nvir] on GPU 0
     * @param nocc, nvir      Occupied / virtual orbital counts
     */
    RIADC2SchurDistributedOperator(
        int num_gpus,
        const std::vector<real_t*>& d_B_ia_local,
        const std::vector<real_t*>& d_B_ab_local,
        const std::vector<real_t*>& d_B_ij_local,
        const std::vector<int>& naux_local,
        const real_t* d_M11,
        const real_t* d_orbital_energies,
        int nocc, int nvir);

    ~RIADC2SchurDistributedOperator();

    RIADC2SchurDistributedOperator(const RIADC2SchurDistributedOperator&) = delete;
    RIADC2SchurDistributedOperator& operator=(const RIADC2SchurDistributedOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return ov_; }
    std::string name() const override { return "RIADC2SchurDistributed"; }

    // --- ω management ---
    void set_omega(real_t omega) { omega_ = omega; }
    real_t get_omega() const { return omega_; }

    /**
     * @brief Build M11 from distributed RI 3-index integrals
     *
     * Computes OVOV = Σ_P B_ia^P × B_ia^{P,T} and OOVV = Σ_P B_ij^P × B_ab^{P,T}
     * via per-GPU partial DGEMM + AllReduce, then runs CIS/ISR/Σ kernels on GPU 0.
     *
     * @param d_M11_out  Output [ov × ov] on GPU 0 (caller must allocate)
     */
    static void build_M11_distributed(
        real_t* d_M11_out,
        int num_gpus,
        const std::vector<real_t*>& d_B_ia_local,
        const std::vector<real_t*>& d_B_ab_local,
        const std::vector<real_t*>& d_B_ij_local,
        const std::vector<int>& naux_local,
        const real_t* d_orbital_energies,
        int nocc, int nvir);

private:
    int num_gpus_;
    int nocc_, nvir_;
    int ov_, vv_, oo_;
    real_t omega_ = 0.0;

    std::vector<int> naux_local_;

    // Per-GPU RI blocks (NOT owned)
    std::vector<const real_t*> d_B_ia_local_;  // [ov  × naux_local]
    std::vector<const real_t*> d_B_ab_local_;  // [vv  × naux_local]
    std::vector<const real_t*> d_B_ij_local_;  // [oo  × naux_local]

    // Per-GPU workspace (owned)
    struct PerGpuWorkspace {
        real_t* d_x        = nullptr;  // replicated input [ov]
        real_t* d_phi      = nullptr;  // [nvir*nocc × naux_local]
        real_t* d_psi      = nullptr;  // [nvir*nocc × naux_local]
        real_t* d_alpha    = nullptr;  // [ov × naux_local]
        real_t* d_beta     = nullptr;  // [nvir × naux_local]
        real_t* d_eri_vvov = nullptr;  // [vv × nvir]  (partial → AllReduced)
        real_t* d_R_all    = nullptr;  // [ov × nvir]  (partial → AllReduced)
        real_t* d_ooov1    = nullptr;  // [nocc × ov]  (partial → AllReduced)
        real_t* d_ooov2    = nullptr;  // [oo × nvir]  (partial → AllReduced)
    };
    std::vector<PerGpuWorkspace> ws_;

    // GPU 0 only (owned)
    real_t* d_M11_      = nullptr;  // [ov × ov]
    real_t* d_D1_       = nullptr;  // [ov]
    real_t* d_diagonal_ = nullptr;  // [ov]
    real_t* d_W_all_    = nullptr;  // [ov × nvir]

    // Device orbital energies on GPU 0
    real_t* d_eps_occ_dev_ = nullptr;  // [nocc]
    real_t* d_eps_vir_dev_ = nullptr;  // [nvir]

    // Host orbital energies (for omega+eps_J)
    std::vector<double> eps_occ_;
    std::vector<double> eps_vir_;

    void compute_D1();
    void compute_diagonal();
    void allocate_workspace();
    void free_workspace();
};

} // namespace gansu

#endif // GANSU_MULTI_GPU
