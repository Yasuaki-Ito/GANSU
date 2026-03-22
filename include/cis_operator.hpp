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
 * @file cis_operator.hpp
 * @brief CIS (Configuration Interaction Singles) linear operator for Davidson solver
 *
 * Implements the CIS matrix-vector product for RHF reference:
 *   A_{ia,jb} = delta_{ij} delta_{ab} (eps_a - eps_i) + 2(ia|jb) - (ij|ab)
 *
 * The operator is pre-built as an explicit matrix and applied via cuBLAS DGEMV.
 */

#pragma once

#include "linear_operator.hpp"

namespace gansu {

/**
 * @brief CIS operator for excited state calculations with RHF reference
 *
 * Constructs the CIS A-matrix from MO ERIs and orbital energies,
 * then applies it to trial vectors using DGEMV.
 *
 * Singlet: A_{ia,jb} = delta_{ij} delta_{ab} (eps_a - eps_i) + 2(ia|jb) - (ij|ab)
 * Triplet: A_{ia,jb} = delta_{ij} delta_{ab} (eps_a - eps_i) - (ij|ab)
 */
class CISOperator : public LinearOperator {
public:
    /**
     * @brief Construct CIS operator
     *
     * @param d_eri_mo Device pointer to full MO ERI tensor [nao^4], chemist notation (pq|rs)
     * @param d_orbital_energies Device pointer to orbital energies [nao]
     * @param nocc Number of occupied spatial orbitals
     * @param nvir Number of virtual spatial orbitals
     * @param nao Total number of spatial orbitals (nocc + nvir)
     * @param is_triplet If true, build triplet CIS matrix (TDA triplet)
     */
    CISOperator(const real_t* d_eri_mo,
                const real_t* d_orbital_energies,
                int nocc, int nvir, int nao,
                bool is_triplet = false);

    ~CISOperator();

    CISOperator(const CISOperator&) = delete;
    CISOperator& operator=(const CISOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return dim_; }
    std::string name() const override { return "CISOperator"; }

private:
    int nocc_;
    int nvir_;
    int nao_;
    int dim_;  // nocc * nvir

    bool is_triplet_;      // true for triplet CIS
    real_t* d_A_matrix_;   // CIS A-matrix [dim x dim], row-major
    real_t* d_diagonal_;   // diagonal elements for preconditioner [dim]

    void build_A_matrix(const real_t* d_eri_mo, const real_t* d_orbital_energies);
};

} // namespace gansu
