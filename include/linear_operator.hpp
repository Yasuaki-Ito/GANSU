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
 * @file linear_operator.hpp
 * @brief Abstract interface for linear operators in iterative eigenvalue solvers
 *
 * This file defines the LinearOperator interface for matrix-free operator applications
 * used in Davidson and other iterative eigenvalue solvers.
 */

#pragma once

#include <string>
#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#endif
#include "types.hpp"

namespace gansu {

/**
 * @brief Abstract base class for linear operators
 *
 * LinearOperator provides a matrix-free interface for applying linear operators
 * to vectors on the GPU. This is used in iterative eigenvalue solvers like the
 * Davidson method, where explicit matrix construction would be prohibitively expensive.
 *
 * Derived classes must implement:
 * - apply(): Apply the operator to an input vector
 * - dimension(): Return the dimension of the operator
 * - name(): Return a descriptive name for debugging
 *
 * Optionally, derived classes can override:
 * - apply_preconditioner(): Provide a preconditioner for faster convergence
 *
 * Example usage:
 * @code
 * class MyOperator : public LinearOperator {
 * public:
 *     void apply(const real_t* d_input, real_t* d_output) const override {
 *         // Implement: output = Op * input
 *     }
 *     int dimension() const override { return my_dim_; }
 *     std::string name() const override { return "MyOperator"; }
 * };
 * @endcode
 */
class LinearOperator {
public:
    /**
     * @brief Virtual destructor for proper cleanup of derived classes
     */
    virtual ~LinearOperator() = default;

    /**
     * @brief Apply the linear operator to an input vector
     *
     * Computes output = Op * input, where Op is the linear operator.
     * Both input and output vectors reside in GPU device memory.
     *
     * @param d_input Device pointer to input vector (size: dimension())
     * @param d_output Device pointer to output vector (size: dimension())
     *
     * @pre d_input and d_output must point to valid device memory
     * @pre d_input and d_output must have size >= dimension()
     * @pre d_input and d_output may alias (same pointer) if operator supports in-place operation
     *
     * @note This is a const method - operator state should not be modified
     * @note Implementation must be thread-safe if used in multi-threaded context
     */
    virtual void apply(const real_t* d_input, real_t* d_output) const = 0;

    /**
     * @brief Get the dimension of the operator
     *
     * Returns the size of the vector space on which this operator acts.
     * For an N×N matrix operator, this returns N.
     *
     * @return Dimension of the operator (positive integer)
     */
    virtual int dimension() const = 0;

    /**
     * @brief Get a descriptive name for the operator
     *
     * Returns a human-readable name used for debugging, logging, and profiling.
     *
     * @return Name string (e.g., "FCI_Hamiltonian", "DiagonalTest")
     */
    virtual std::string name() const = 0;

    /**
     * @brief Apply a preconditioner to accelerate convergence
     *
     * Computes output = M^(-1) * input, where M is a preconditioner that approximates
     * the operator. A good preconditioner makes M^(-1)*Op close to the identity,
     * accelerating iterative solver convergence.
     *
     * Default implementation is the identity: output = input (no preconditioning).
     * Derived classes should override this for better performance.
     *
     * Common preconditioners:
     * - Diagonal (Jacobi): M_ii = Op_ii
     * - Incomplete factorization
     * - Approximate inverse
     *
     * @param d_input Device pointer to input vector (size: dimension())
     * @param d_output Device pointer to output vector (size: dimension())
     *
     * @note Default implementation simply copies input to output
     * @note d_input and d_output must not alias unless preconditioner supports in-place
     */
    virtual void apply_preconditioner(const real_t* d_input, real_t* d_output) const {
        // Default: identity preconditioner (no preconditioning)
        cudaMemcpy(d_output, d_input, dimension() * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
    }

    /**
     * @brief Get device pointer to operator diagonal elements for Davidson correction
     *
     * Returns nullptr by default (no diagonal available).
     * If non-null, Davidson solver uses Jacobi correction:
     *   correction[j] = -residual[j] / (diagonal[j] - eigenvalue)
     *
     * @return Device pointer to diagonal elements [dimension()], or nullptr
     */
    virtual const real_t* get_diagonal_device() const { return nullptr; }
};

} // namespace gansu
