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
 * @file test_operators.hpp
 * @brief Test operators for Davidson solver validation
 *
 * This file provides simple linear operators with known eigenvalues
 * for testing and validating the Davidson eigenvalue solver.
 */

#pragma once

#include "linear_operator.hpp"
#include <vector>

namespace gansu {

/**
 * @brief Diagonal matrix operator for testing
 *
 * Represents a diagonal matrix D where D_ii = diagonal[i].
 * The eigenvalues are exactly the diagonal elements, making this
 * ideal for validating Davidson solver accuracy.
 *
 * Example:
 * @code
 * std::vector<real_t> diag = {1.0, 2.0, 3.0, 4.0, 5.0};
 * DiagonalOperator op(diag);
 * // op has eigenvalues [1.0, 2.0, 3.0, 4.0, 5.0]
 * @endcode
 */
class DiagonalOperator : public LinearOperator {
public:
    /**
     * @brief Construct a diagonal operator
     *
     * @param diagonal Vector of diagonal elements (host memory)
     *
     * @note Copies diagonal to device memory
     * @note Diagonal elements should be in ascending order for easier testing
     */
    explicit DiagonalOperator(const std::vector<real_t>& diagonal);

    /**
     * @brief Destructor - frees device memory
     */
    ~DiagonalOperator() override;

    /**
     * @brief Apply diagonal operator: output[i] = diagonal[i] * input[i]
     */
    void apply(const real_t* d_input, real_t* d_output) const override;

    /**
     * @brief Get dimension (size of diagonal)
     */
    int dimension() const override { return dim_; }

    /**
     * @brief Get operator name
     */
    std::string name() const override { return "DiagonalOperator"; }

    /**
     * @brief Apply diagonal preconditioner: output[i] = input[i] / diagonal[i]
     *
     * This is the exact inverse for diagonal matrices.
     */
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;

private:
    int dim_;                  ///< Dimension of the operator
    real_t* d_diagonal_;       ///< Device pointer to diagonal elements
};

/**
 * @brief Small symmetric matrix operator for testing
 *
 * Represents a small symmetric matrix stored explicitly on the GPU.
 * Useful for testing Davidson solver with matrices that have known
 * analytical eigenvalues.
 *
 * Example:
 * @code
 * // 3x3 symmetric matrix:
 * // [4  1  0]
 * // [1  3  1]
 * // [0  1  2]
 * std::vector<real_t> matrix = {
 *     4.0, 1.0, 0.0,
 *     1.0, 3.0, 1.0,
 *     0.0, 1.0, 2.0
 * };
 * SymmetricMatrixOperator op(matrix, 3);
 * @endcode
 */
class SymmetricMatrixOperator : public LinearOperator {
public:
    /**
     * @brief Construct a symmetric matrix operator
     *
     * @param matrix Symmetric matrix in row-major order (host memory)
     * @param n Dimension of the matrix (n×n)
     *
     * @pre matrix.size() must equal n*n
     * @pre Matrix must be symmetric: matrix[i*n+j] == matrix[j*n+i]
     *
     * @note Copies matrix to device memory
     * @note Uses cuBLAS for matrix-vector product
     */
    SymmetricMatrixOperator(const std::vector<real_t>& matrix, int n);

    /**
     * @brief Destructor - frees device memory
     */
    ~SymmetricMatrixOperator() override;

    /**
     * @brief Apply matrix-vector product: output = matrix * input
     *
     * Uses cuBLAS Dgemv for efficient computation.
     */
    void apply(const real_t* d_input, real_t* d_output) const override;

    /**
     * @brief Get dimension
     */
    int dimension() const override { return dim_; }

    /**
     * @brief Get operator name
     */
    std::string name() const override { return "SymmetricMatrixOperator"; }

    /**
     * @brief Apply diagonal preconditioner
     *
     * Uses only diagonal elements: output[i] = input[i] / matrix[i,i]
     */
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;

private:
    int dim_;                  ///< Dimension of the matrix
    real_t* d_matrix_;         ///< Device pointer to matrix (row-major)
    real_t* d_diagonal_;       ///< Device pointer to diagonal elements (for preconditioner)
};

} // namespace gansu
