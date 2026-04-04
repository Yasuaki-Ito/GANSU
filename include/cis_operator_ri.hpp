#pragma once

#include "linear_operator.hpp"
#include "gpu_manager.hpp"

namespace gansu {

/**
 * @brief B-matrix based CIS operator (RI approximation)
 *
 * Avoids nmo^4 MO ERI storage by computing sigma vectors directly
 * from the RI intermediate B matrix:
 *   B_ov(Q, ia) = B(Q, i, a+nocc)
 *   B_oo(Q, ij) = B(Q, i, j)
 *   B_vv(Q, ab) = B(Q, a+nocc, b+nocc)
 *
 * Sigma vector:
 *   σ(ia) = (ε_a - ε_i) r(ia)
 *         + 2 Σ_Q B_ov(Q,ia) [Σ_{jb} B_ov(Q,jb) r(jb)]     [Coulomb]
 *         - Σ_Q Σ_j B_oo(Q,ij) [Σ_b B_vv(Q,ab) r(jb)]      [Exchange]
 *
 * Memory: O(naux × nmo²) instead of O(nmo⁴)
 */
class CISOperator_RI : public LinearOperator {
public:
    /**
     * @param d_B_ov  B(Q, ia) in row-major [naux × nocc*nvir]. Caller retains ownership.
     * @param d_B_oo  B(Q, ij) in row-major [naux × nocc*nocc]. Caller retains ownership.
     * @param d_B_vv  B(Q, ab) in row-major [naux × nvir*nvir]. Caller retains ownership.
     * @param d_orbital_energies  [nao]
     * @param nocc, nvir, naux
     * @param is_triplet  If true, skip Coulomb term (triplet CIS)
     */
    CISOperator_RI(const real_t* d_B_ov, const real_t* d_B_oo, const real_t* d_B_vv,
                   const real_t* d_orbital_energies,
                   int nocc, int nvir, int naux,
                   bool is_triplet = false);
    ~CISOperator_RI();

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return dim_; }
    std::string name() const override { return "CISOperator_RI"; }

private:
    int nocc_, nvir_, naux_, dim_;
    bool is_triplet_;

    const real_t* d_B_ov_;   // [naux, nocc*nvir] — not owned
    const real_t* d_B_oo_;   // [naux, nocc*nocc] — not owned
    const real_t* d_B_vv_;   // [naux, nvir*nvir] — not owned

    real_t* d_diagonal_;     // [dim] — owned
    real_t* d_work_;         // workspace for sigma computation — owned

    void build_diagonal(const real_t* d_orbital_energies);
};

} // namespace gansu
