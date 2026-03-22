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
 * @file adc2x_operator.hpp
 * @brief ADC(2)-x (extended) operator in the full singles+doubles space
 *
 * ADC(2)-x extends ADC(2)-s by including first-order off-diagonal terms in M22:
 *   σ1 = M11 · R1 + M12 · R2     (identical to ADC(2)-s)
 *   σ2 = M21 · R1 + (D2 + V) · R2  (V = first-order off-diagonal M22)
 *
 * M11, M12, M21 are IDENTICAL to ADC(2)-s (reused via ADC2Operator).
 * V includes: oooo, vvvv, voov contractions derived from spin integration.
 *
 * After singlet RHF spin integration (verified against PySCF radc_ee.py):
 *   oooo: Σ_{kl} (ik|jl) r^{ab}_{kl}
 *   vvvv: Σ_{cd} (ac|bd) r^{cd}_{ij}
 *   voov (8 terms): see adc2x_operator.cu for details
 */

#pragma once

#include "linear_operator.hpp"
#include "adc2_operator.hpp"

namespace gansu {

class ADC2XOperator : public LinearOperator {
public:
    /**
     * @brief Construct ADC(2)-x operator from an existing ADC2Operator
     *
     * @param adc2_op Reference to ADC2Operator (must outlive this object)
     * @param d_eri_mo Full MO ERI tensor on device [nao^4] for extracting oooo/vvvv/oovv blocks
     * @param nao Total number of basis functions
     */
    ADC2XOperator(const ADC2Operator& adc2_op,
                  const real_t* d_eri_mo, int nao);

    ~ADC2XOperator();

    ADC2XOperator(const ADC2XOperator&) = delete;
    ADC2XOperator& operator=(const ADC2XOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "ADC2XOperator"; }

private:
    const ADC2Operator& adc2_op_;
    int nocc_, nvir_;
    int singles_dim_;
    int doubles_dim_;
    int total_dim_;

    // Additional ERI blocks for M22 first-order terms (device)
    real_t* d_eri_oooo_;  // (ij|kl) [nocc^4]
    real_t* d_eri_vvvv_;  // (ab|cd) [nvir^4]
    real_t* d_eri_oovv_;  // (ij|ab) [nocc^2*nvir^2]

    // Full-space diagonal for preconditioner [total_dim]
    real_t* d_diagonal_;
};

} // namespace gansu
