/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file dlpno_ip_eom_projected_operator.hpp
 * @brief DLPNO-IP-EOM operator via Galerkin projection of the canonical
 *        IP-EOM σ onto the per-pair PNO space (bt-PNO-STEOM stage B,
 *        Phase B1b, "project-up reference").
 *
 * Implements the packed-PNO operator  σ_packed = P · σ_canon · P  where:
 *   - P lifts a packed-PNO vector to the canonical r2 layout (ip_packed_r2_to
 *     _canonical) and, on the way back, projects the canonical σ2 down to the
 *     per-pair PNO blocks (ip_canonical_r2_to_packed).
 *   - σ_canon is the VALIDATED canonical IP-EOM operator (P1 IPEOMCCSDOperator),
 *     supplied as the inner LinearOperator.
 *
 * This is correct-by-construction (reuses the validated canonical σ, no
 * formula re-derivation): at no truncation (n_pno = nvir) the projection is a
 * bijection, so the DLPNO-IP roots equal the canonical P1 roots bit-closely;
 * with truncation it is the standard DLPNO (PNO-subspace) approximation. It is
 * NOT a scaling implementation (the full canonical σ runs each matvec) — it is
 * the reference against which the native per-pair σ (stage B (a)) is validated,
 * and it already yields a working end-to-end DLPNO-bt-STEOM on small systems.
 *
 * apply() round-trips through the host (D2H → Eigen transform → H2D → inner
 * device σ → D2H → Eigen transform → H2D); acceptable for the reference path.
 * The preconditioner uses the Koopmans/PNO diagonal (as in B0.2).
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "dlpno_mp2.hpp"        // DLPNOLMP2Result
#include "dlpno_ip_packing.hpp" // DLPNOIPPacking

namespace gansu {

class DLPNOIPEOMProjectedOperator : public LinearOperator {
public:
    /**
     * @param canonical Inner canonical IP-EOM σ (dim = nocc + nocc²·nvir,
     *                  layout [R1 | R2_canon[(I*nocc+J)*nvir+a]]). Must outlive
     *                  this operator. Built from DLPNO back-transformed T1/T2.
     * @param res       Converged DLPNO result (per-pair bar_Q/Lambda/setups). Ref held.
     * @param packing   Packed-vector offset table. Ref held.
     * @param U_loc     [nocc²] localization rotation (copied).
     * @param C_vir     [nao·nvir] canonical virtual coefficients (copied).
     * @param h_S       [nao²] AO overlap (copied).
     * @param nao,nvir  dimensions.
     * @param eps_o     [nocc] active-occupied energies (preconditioner diagonal).
     */
    DLPNOIPEOMProjectedOperator(const LinearOperator& canonical,
                                const DLPNOLMP2Result& res,
                                const DLPNOIPPacking& packing,
                                const std::vector<real_t>& U_loc,
                                const std::vector<real_t>& C_vir,
                                const real_t* h_S,
                                int nao, int nvir,
                                const std::vector<real_t>& eps_o);

    ~DLPNOIPEOMProjectedOperator();

    DLPNOIPEOMProjectedOperator(const DLPNOIPEOMProjectedOperator&) = delete;
    DLPNOIPEOMProjectedOperator& operator=(const DLPNOIPEOMProjectedOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "DLPNOIPEOMProjectedOperator"; }

private:
    const LinearOperator& canonical_;
    const DLPNOLMP2Result& res_;
    const DLPNOIPPacking& packing_;
    std::vector<real_t> U_loc_;
    std::vector<real_t> C_vir_;
    std::vector<real_t> h_S_;
    int nao_ = 0;
    int nvir_ = 0;
    int nocc_ = 0;
    int total_dim_ = 0;       ///< packed dimension
    int canonical_dim_ = 0;   ///< nocc + nocc²·nvir
    real_t* d_diagonal_ = nullptr;
    std::vector<real_t> h_diagonal_;
};

} // namespace gansu
