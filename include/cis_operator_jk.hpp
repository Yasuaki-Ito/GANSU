#pragma once

#include "linear_operator.hpp"
#include "eri.hpp"
#include "gpu_manager.hpp"

namespace gansu {

/**
 * @brief Half-transform based CIS operator (no full MO ERI needed)
 *
 * Uses OVOV (ia|jb) and OOVV (ij|ab) sub-blocks built via half-transformation.
 * Sigma vector computed as matrix-free contraction:
 *
 * Singlet: σ(ia) = (ε_a - ε_i) r(ia) + Σ_{jb} [2(ia|jb) - (ij|ab)] r(jb)
 * Triplet: σ(ia) = (ε_a - ε_i) r(ia) - Σ_{jb} (ij|ab) r(jb)
 *
 * Memory: 2 × nocc²×nvir² (OVOV + OOVV blocks). No nao⁴ MO ERI.
 */
class CISOperator_HalfTransform : public LinearOperator {
public:
    /**
     * @param d_ovov  (ia|jb) block [nocc*nvir*nocc*nvir], layout: [i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]
     * @param d_oovv  (ij|ab) block [nocc*nocc*nvir*nvir], layout: [i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b]
     * @param d_orbital_energies  [nao]
     * @param nocc, nvir
     * @param is_triplet  If true, use triplet CIS formula
     */
    CISOperator_HalfTransform(const real_t* d_ovov, const real_t* d_oovv,
                              const real_t* d_orbital_energies,
                              int nocc, int nvir,
                              bool is_triplet = false);
    ~CISOperator_HalfTransform();

    CISOperator_HalfTransform(const CISOperator_HalfTransform&) = delete;
    CISOperator_HalfTransform& operator=(const CISOperator_HalfTransform&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return dim_; }
    std::string name() const override { return "CISOperator_HalfTransform"; }

private:
    const real_t* d_ovov_;  // not owned
    const real_t* d_oovv_;  // not owned
    int nocc_, nvir_, dim_;
    bool is_triplet_;

    real_t* d_diagonal_;    // [dim]
    mutable real_t* d_work_; // [dim] workspace for sigma

    void build_diagonal(const real_t* d_orbital_energies);
};

} // namespace gansu
