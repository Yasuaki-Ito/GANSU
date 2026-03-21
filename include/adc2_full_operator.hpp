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
 * @file adc2_full_operator.hpp
 * @brief ADC(2) operator in the full singles+doubles space
 *
 * Full-space ADC(2) matrix-vector product without Schur complement:
 *   σ1 = M11 · R1 + M12 · R2
 *   σ2 = M21 · R1 + D2  · R2
 *
 * The input/output vector layout is [R1(ov) | R2(oovv)].
 * This operator is used with the Davidson solver for direct diagonalization
 * of the full ADC(2) matrix, avoiding the ω-dependent Schur complement iteration.
 *
 * Uses data from an existing ADC2Operator instance (ERI blocks, M11, D2).
 */

#pragma once

#include "linear_operator.hpp"
#include "adc2_operator.hpp"

namespace gansu {

class ADC2FullOperator : public LinearOperator {
public:
    /**
     * @brief Construct full-space ADC(2) operator
     *
     * @param adc2_op Reference to ADC2Operator that holds precomputed M11, ERI blocks, D2, etc.
     *                Must outlive this object.
     */
    explicit ADC2FullOperator(const ADC2Operator& adc2_op);

    ~ADC2FullOperator();

    ADC2FullOperator(const ADC2FullOperator&) = delete;
    ADC2FullOperator& operator=(const ADC2FullOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "ADC2FullOperator"; }

private:
    const ADC2Operator& adc2_op_;
    int singles_dim_;
    int doubles_dim_;
    int total_dim_;

    real_t* d_diagonal_;  // [total_dim] = [D1 | D2] for preconditioner
};

} // namespace gansu
