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

#pragma once

#include "linear_operator.hpp"
#include <vector>
#include <cstdint>

namespace gansu {

/**
 * @brief FCI Hamiltonian operator for Davidson eigenvalue solver
 *
 * Implements the Full-CI Hamiltonian as a LinearOperator, computing
 * sigma = H * C using Slater-Condon rules on GPU.
 *
 * The FCI basis consists of all Slater determinants formed by distributing
 * N_alpha and N_beta electrons among M spatial orbitals.
 * Total dimension = C(M, N_alpha) * C(M, N_beta).
 *
 * Determinants are represented as uint64_t bit strings (max 64 spatial orbitals).
 */
class FCIHamiltonianOperator : public LinearOperator {
public:
    /**
     * @brief Construct FCI Hamiltonian operator
     *
     * @param d_h1_mo  Device pointer to 1-electron MO integrals [M x M], row-major
     * @param d_eri_mo Device pointer to 2-electron MO integrals [M^4], (pq|rs) chemist notation
     * @param num_orbitals Number of spatial orbitals (M)
     * @param num_alpha Number of alpha electrons
     * @param num_beta  Number of beta electrons
     *
     * @note d_h1_mo and d_eri_mo must remain valid for the lifetime of this object
     */
    FCIHamiltonianOperator(const real_t* d_h1_mo,
                           const real_t* d_eri_mo,
                           int num_orbitals,
                           int num_alpha,
                           int num_beta);

    ~FCIHamiltonianOperator();

    FCIHamiltonianOperator(const FCIHamiltonianOperator&) = delete;
    FCIHamiltonianOperator& operator=(const FCIHamiltonianOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    int dimension() const override { return num_det_; }
    std::string name() const override { return "FCIHamiltonianOperator"; }

    const real_t* get_diagonal_device() const { return d_diagonal_; }

private:
    const real_t* d_h1_mo_;
    const real_t* d_eri_mo_;

    int num_orbitals_;
    int num_alpha_;
    int num_beta_;
    int num_alpha_det_;
    int num_beta_det_;
    int num_det_;

    uint64_t* d_alpha_strings_;
    uint64_t* d_beta_strings_;
    real_t* d_diagonal_;

    void generate_determinants();
    void compute_diagonal();
};

} // namespace gansu
