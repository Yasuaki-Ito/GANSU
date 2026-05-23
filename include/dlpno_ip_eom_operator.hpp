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
 * @file dlpno_ip_eom_operator.hpp
 * @brief DLPNO-IP-EOM-CCSD linear operator for Davidson in the per-pair PNO
 *        basis (bt-PNO-STEOM stage B, Dutta-Saitow-Riplinger-Neese-Izsák
 *        2018 JCP 148, 244101).
 *
 * The Davidson vector is packed as [ R1 (nocc) | per-pair R2 blocks ] per
 * DLPNOIPPacking (see dlpno_ip_packing.hpp). Davidson is agnostic to the
 * vector meaning (it only needs dimension() + apply()), so the packed,
 * non-canonical-sized vector works directly.
 *
 * Phase B0.2 (this commit): diagonal-only apply() — σ = D·x with the Koopmans
 * 1h diagonal (−ε_i) and the 2h1p denominator (−F_ii − F_jj + Λ_a, Λ_a the PNO
 * eigenvalue of pair (i,j)). Davidson eigenvalues therefore equal the sorted
 * diagonal, which at no truncation + localizer none reproduces the canonical
 * IP-EOM diagonal. Validates the packed Davidson plumbing end-to-end before
 * the real sigma terms land (B1 pair-local, B2 cross-pair).
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "dlpno_mp2.hpp"        // DLPNOLMP2Result
#include "dlpno_ip_packing.hpp" // DLPNOIPPacking

namespace gansu {

class DLPNOIPEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the (diagonal-only, B0.2) DLPNO-IP-EOM operator.
     * @param res      Converged DLPNO result (per-pair Lambda + setups F_ii/F_jj).
     * @param packing  Packed-vector offset table (build_ip_packing).
     * @param eps_o    [nocc] active-occupied orbital energies (1h Koopmans diagonal).
     */
    DLPNOIPEOMCCSDOperator(const DLPNOLMP2Result& res,
                           const DLPNOIPPacking& packing,
                           const std::vector<real_t>& eps_o);

    ~DLPNOIPEOMCCSDOperator();

    DLPNOIPEOMCCSDOperator(const DLPNOIPEOMCCSDOperator&) = delete;
    DLPNOIPEOMCCSDOperator& operator=(const DLPNOIPEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "DLPNOIPEOMCCSDOperator"; }

    /// Host copy of the packed diagonal (for the B0.2 deterministic gate).
    const std::vector<real_t>& host_diagonal() const { return h_diagonal_; }

private:
    int total_dim_ = 0;
    real_t* d_diagonal_ = nullptr;     ///< [total_dim] packed diagonal (device-owned)
    std::vector<real_t> h_diagonal_;   ///< host mirror (gate / CPU path)
};

} // namespace gansu
