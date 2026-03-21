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
 * @file eom_mp2_schur_operator.hpp
 * @brief Schur complement operator in singles space for EOM methods
 *
 * Effective Hamiltonian:
 *   M_eff = M11 - M12 · D2⁻¹ · M21
 *
 * Works with any EOM operator that has:
 *   - apply() in the full singles+doubles space
 *   - get_D1(), get_D2(), get_singles_dim(), get_doubles_dim()
 *
 * For EOM-MP2: approximate (M22 off-diagonal ignored)
 * For EOM-CC2: EXACT (M22 is purely diagonal)
 */

#pragma once

#include "linear_operator.hpp"
#include "eom_mp2_operator.hpp"
#include "eom_cc2_operator.hpp"

namespace gansu {

class EOMMP2SchurOperator : public LinearOperator {
public:
    /**
     * @brief Construct Schur complement operator wrapping an EOMMP2Operator
     */
    explicit EOMMP2SchurOperator(EOMMP2Operator& full_op);

    /**
     * @brief Construct Schur complement operator wrapping an EOMCC2Operator
     */
    explicit EOMMP2SchurOperator(EOMCC2Operator& full_op);

    ~EOMMP2SchurOperator();

    EOMMP2SchurOperator(const EOMMP2SchurOperator&) = delete;
    EOMMP2SchurOperator& operator=(const EOMMP2SchurOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return singles_dim_; }
    std::string name() const override { return "EOMMP2SchurOperator"; }

    /**
     * @brief Set frequency parameter for frequency-dependent Schur complement
     *
     * M_eff(ω) = M11 + M12 · (ωI - M22)⁻¹ · M21
     * ω = 0 reduces to standard Schur complement: M11 - M12 · D2⁻¹ · M21
     */
    void set_omega(real_t omega) { omega_ = omega; }
    real_t get_omega() const { return omega_; }

private:
    void init(LinearOperator& op, const real_t* d_D1, const real_t* d_D2,
              int singles_dim, int doubles_dim, int total_dim);

    LinearOperator& full_op_;
    const real_t* d_D2_ptr_;   // borrowed pointer to D2 from the full operator
    real_t omega_ = 0.0;       // frequency parameter for ω-dependent Schur complement
    int singles_dim_;
    int doubles_dim_;
    int total_dim_;

    // Workspace for calling full_op_.apply()
    mutable real_t* d_full_input1_;   // [total_dim] for [R1 | 0]
    mutable real_t* d_full_output1_;  // [total_dim] for σ([R1|0])
    mutable real_t* d_full_input2_;   // [total_dim] for [0 | scaled_R2]
    mutable real_t* d_full_output2_;  // [total_dim] for σ([0|scaled_R2])

    // Diagonal of M_eff for preconditioner (approximated as D1)
    real_t* d_diagonal_;              // [singles_dim]
};

} // namespace gansu
