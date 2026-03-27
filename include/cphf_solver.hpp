/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef CPHF_SOLVER_HPP
#define CPHF_SOLVER_HPP

#include "types.hpp"
#include "linear_operator.hpp"
#include <vector>

namespace gansu {

/**
 * @brief CPHF (Coupled-Perturbed Hartree-Fock) Operator
 *
 * Implements A * U = B where:
 *   A_{ai,bj} = δ_{ab}δ_{ij}(ε_a - ε_i) + 4(ai|bj) - (ab|ij) - (aj|ib)
 *
 * The operator acts on vectors U of size nocc*nvir (occ-vir block).
 */
class CPHFOperator : public LinearOperator {
public:
    /**
     * @param d_eri_mo   Device pointer to full MO ERI tensor [nmo^4]
     * @param d_orbital_energies  Device pointer to orbital energies [nmo]
     * @param nocc       Number of occupied orbitals
     * @param nvir       Number of virtual orbitals
     * @param nmo        Total number of MOs (nocc + nvir)
     */
    CPHFOperator(const real_t* d_eri_mo, const real_t* d_orbital_energies,
                 int nocc, int nvir, int nmo);
    ~CPHFOperator();

    void apply(const real_t* d_input, real_t* d_output) const override;
    int dimension() const override { return nocc_ * nvir_; }
    std::string name() const override { return "CPHF"; }
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }

private:
    const real_t* d_eri_mo_;
    const real_t* d_orbital_energies_;
    int nocc_, nvir_, nmo_;
    real_t* d_diagonal_;  // (ε_a - ε_i) diagonal preconditioner
};

/**
 * @brief Solve CPHF equations using preconditioned conjugate gradient
 *
 * Solves A * U^x = -RHS^x for each perturbation direction x.
 *
 * @param cphf_op    CPHF operator
 * @param d_rhs      Device pointer to RHS vectors [n_pert × nocc*nvir]
 * @param d_U        Device pointer to output U vectors [n_pert × nocc*nvir]
 * @param n_pert     Number of perturbation directions (3*num_atoms)
 * @param tol        Convergence tolerance
 * @param max_iter   Maximum iterations
 */
void solve_cphf(const CPHFOperator& cphf_op,
                const real_t* d_rhs, real_t* d_U,
                int n_pert, double tol = 1e-8, int max_iter = 100);

} // namespace gansu

#endif // CPHF_SOLVER_HPP
