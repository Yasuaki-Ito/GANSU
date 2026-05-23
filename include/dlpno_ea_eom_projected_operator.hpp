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
 * @file dlpno_ea_eom_projected_operator.hpp
 * @brief DLPNO-EA-EOM operator via Galerkin projection of the canonical
 *        EA-EOM σ onto the per-i PNO(i,i) 2p1h space (bt-PNO-STEOM stage B,
 *        "project-up reference"). EA analog of DLPNOIPEOMProjectedOperator.
 *
 * σ_packed = P · σ_canon · P, where P lifts a packed-PNO EA vector to the
 * canonical r2 layout (ea_packed_r2_to_canonical) and projects the canonical
 * σ2 down to the per-i PNO(i,i) blocks (ea_canonical_r2_to_packed); σ_canon is
 * the validated canonical EA-EOM operator (P2 EAEOMCCSDOperator), the inner
 * LinearOperator. Reference path (full canonical σ each matvec; not scaling).
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "dlpno_mp2.hpp"        // DLPNOLMP2Result
#include "dlpno_ea_packing.hpp" // DLPNOEAPacking

namespace gansu {

class DLPNOEAEOMProjectedOperator : public LinearOperator {
public:
    /**
     * @param canonical Inner canonical EA-EOM σ (dim = nvir + nocc·nvir²,
     *                  layout [R1(nvir) | R2_canon[(I*nvir+a)*nvir+b]]).
     * @param res       Converged DLPNO result. Ref held.
     * @param packing   EA packed-vector offset table. Ref held.
     * @param U_loc     [nocc²] localization rotation (copied).
     * @param C_vir     [nao·nvir] canonical virtual coefficients (copied).
     * @param h_S       [nao²] AO overlap (copied).
     * @param nao       basis-function count.
     * @param eps_v     [nvir] virtual orbital energies (1p Koopmans diagonal).
     */
    DLPNOEAEOMProjectedOperator(const LinearOperator& canonical,
                                const DLPNOLMP2Result& res,
                                const DLPNOEAPacking& packing,
                                const std::vector<real_t>& U_loc,
                                const std::vector<real_t>& C_vir,
                                const real_t* h_S,
                                int nao,
                                const std::vector<real_t>& eps_v);

    ~DLPNOEAEOMProjectedOperator();

    DLPNOEAEOMProjectedOperator(const DLPNOEAEOMProjectedOperator&) = delete;
    DLPNOEAEOMProjectedOperator& operator=(const DLPNOEAEOMProjectedOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "DLPNOEAEOMProjectedOperator"; }

private:
    const LinearOperator& canonical_;
    const DLPNOLMP2Result& res_;
    const DLPNOEAPacking& packing_;
    std::vector<real_t> U_loc_;
    std::vector<real_t> C_vir_;
    std::vector<real_t> h_S_;
    int nao_ = 0;
    int nvir_ = 0;
    int nocc_ = 0;
    int total_dim_ = 0;       ///< packed dimension
    int canonical_dim_ = 0;   ///< nvir + nocc·nvir²
    real_t* d_diagonal_ = nullptr;
    std::vector<real_t> h_diagonal_;
};

} // namespace gansu
