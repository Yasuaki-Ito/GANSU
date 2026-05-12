/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_dlpno_tno.cu
 * @brief Phase 3.2.0 unit tests for the TNOBuilder.
 *
 * Synthetic-data tests that isolate TNOBuilder from the DLPNO-LMP2
 * pipeline. We construct a small AO basis (nao = 8) with a non-trivial
 * overlap S and a random symmetric Fock F, build three S-orthonormal pair
 * PNO blocks (with deliberately overlapping spans so the union is
 * linearly dependent), and check:
 *   1. n_tno is positive and ≤ Σ n_pno.
 *   2. Q_tno is S-orthonormal: Q_tno^T · S · Q_tno = I.
 *   3. eps_tno is sorted ascending.
 *   4. F is diagonal in the TNO basis: Q_tno^T · F · Q_tno = diag(eps_tno).
 *   5. When the three pair blocks share columns, n_dropped_overlap > 0.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "dlpno_pair_data.hpp"
#include "dlpno_tno.hpp"

using namespace gansu;

namespace {

using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Build a random symmetric positive definite "overlap" matrix.
std::vector<real_t> make_overlap(int nao, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    RowMatXd A(nao, nao);
    for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j)
            A(i, j) = nd(rng);
    RowMatXd S = A.transpose() * A;
    // Diagonal-dominant for well-conditioned S.
    for (int i = 0; i < nao; ++i) S(i, i) += real_t(nao);
    std::vector<real_t> out(static_cast<size_t>(nao) * nao, 0.0);
    Eigen::Map<RowMatXd>(out.data(), nao, nao) = S;
    return out;
}

// Build a random symmetric Fock matrix.
std::vector<real_t> make_fock(int nao, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    RowMatXd F(nao, nao);
    for (int i = 0; i < nao; ++i)
        for (int j = 0; j <= i; ++j) {
            const real_t v = nd(rng);
            F(i, j) = v;
            F(j, i) = v;
        }
    std::vector<real_t> out(static_cast<size_t>(nao) * nao, 0.0);
    Eigen::Map<RowMatXd>(out.data(), nao, nao) = F;
    return out;
}

// Pick `n_pno` columns from a fixed virtual basis (S-canonical orthonormal),
// optionally sharing a column with another block to force linear dependence.
std::vector<real_t> make_pair_bar_Q(const real_t* C_vir,
                                    int nao, int n_vir,
                                    const std::vector<int>& cols)
{
    std::vector<real_t> out(static_cast<size_t>(nao) * cols.size(), 0.0);
    for (size_t k = 0; k < cols.size(); ++k) {
        const int src = cols[k];
        if (src < 0 || src >= n_vir)
            throw std::out_of_range("col idx");
        for (int mu = 0; mu < nao; ++mu)
            out[mu * cols.size() + k] = C_vir[mu * n_vir + src];
    }
    return out;
}

// Compute Q^T · M · Q. Q row-major (nao × n), M row-major (nao × nao).
std::vector<real_t> QtMQ(const real_t* Q, int n,
                         const real_t* M, int nao)
{
    Eigen::Map<const RowMatXd> Qm(Q, nao, n);
    Eigen::Map<const RowMatXd> Mm(M, nao, nao);
    const RowMatXd MQ = Mm * Qm;
    const RowMatXd R  = Qm.transpose() * MQ;
    std::vector<real_t> out(static_cast<size_t>(n) * n, 0.0);
    Eigen::Map<RowMatXd>(out.data(), n, n) = R;
    return out;
}

// Build a fixed S-orthonormal virtual basis C_vir (nao × n_vir) by
// diagonalising S and dividing by sqrt(eigenvalues), then taking the first
// n_vir columns.
std::vector<real_t> build_S_orthonormal_basis(const real_t* S, int nao, int n_vir)
{
    Eigen::Map<const RowMatXd> Sm(S, nao, nao);
    Eigen::SelfAdjointEigenSolver<RowMatXd> es(Sm);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("S eigendecomp failed");
    Eigen::VectorXd ev = es.eigenvalues();
    RowMatXd V = es.eigenvectors();
    // Symmetric inverse-sqrt-style basis: each column v_k / sqrt(λ_k)
    // satisfies (v_k/√λ_k)^T · S · (v_l/√λ_l) = δ_kl.
    RowMatXd C(nao, nao);
    for (int k = 0; k < nao; ++k)
        C.col(k) = V.col(k) / std::sqrt(ev(k));
    std::vector<real_t> out(static_cast<size_t>(nao) * n_vir, 0.0);
    for (int mu = 0; mu < nao; ++mu)
        for (int b = 0; b < n_vir; ++b)
            out[mu * n_vir + b] = C(mu, b);
    return out;
}

real_t max_abs_off_diag(const std::vector<real_t>& M, int n)
{
    real_t m = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) m = std::max(m, std::fabs(M[i * n + j]));
    return m;
}

real_t max_abs_diag_minus_one(const std::vector<real_t>& M, int n)
{
    real_t m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(M[i * n + i] - 1.0));
    return m;
}

PairData make_pair(int n_pno, std::vector<real_t>&& bar_Q)
{
    PairData p;
    p.n_pno = n_pno;
    p.bar_Q = std::move(bar_Q);
    return p;
}

} // namespace

// =========================================================================
// 1. Disjoint pair blocks: TNO size = total PNO sum, no linear dependence.
// =========================================================================
TEST(DLPNOTno, DisjointBlocks_PreserveAllDirections)
{
    const int nao = 8;
    const int n_vir = nao;
    const auto S = make_overlap(nao, /*seed=*/42);
    const auto F = make_fock(nao, /*seed=*/7);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);

    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {4, 5})));

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);

    EXPECT_EQ(tno.n_tno, 6);
    EXPECT_EQ(tno.n_dropped_overlap, 0);
    EXPECT_EQ(tno.eps_tno.size(), 6u);
    EXPECT_EQ(tno.Q_tno.size(), static_cast<size_t>(nao) * 6);

    // S-orthonormality.
    auto QtSQ = QtMQ(tno.Q_tno.data(), tno.n_tno, S.data(), nao);
    EXPECT_LT(max_abs_off_diag(QtSQ, tno.n_tno), 1e-10);
    EXPECT_LT(max_abs_diag_minus_one(QtSQ, tno.n_tno), 1e-10);

    // F diagonalised in TNO basis.
    auto QtFQ = QtMQ(tno.Q_tno.data(), tno.n_tno, F.data(), nao);
    real_t worst_off = 0.0, worst_diag = 0.0;
    for (int a = 0; a < tno.n_tno; ++a)
        for (int b = 0; b < tno.n_tno; ++b) {
            const real_t expected = (a == b) ? tno.eps_tno[a] : real_t(0);
            const real_t diff = std::fabs(QtFQ[a * tno.n_tno + b] - expected);
            if (a == b) worst_diag = std::max(worst_diag, diff);
            else        worst_off  = std::max(worst_off,  diff);
        }
    EXPECT_LT(worst_off,  1e-10);
    EXPECT_LT(worst_diag, 1e-10);

    // Ascending eps.
    for (int a = 1; a < tno.n_tno; ++a)
        EXPECT_LE(tno.eps_tno[a - 1], tno.eps_tno[a] + 1e-12);
}

// =========================================================================
// 2. Overlapping pair blocks: linear dependence is detected and removed.
// =========================================================================
TEST(DLPNOTno, OverlappingBlocks_DropLinearlyDependent)
{
    const int nao = 8;
    const int n_vir = nao;
    const auto S = make_overlap(nao, /*seed=*/123);
    const auto F = make_fock(nao, /*seed=*/456);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);

    // Pairs share columns: ij = {0,1,2}, ik = {1,2,3}, jk = {2,3,4}.
    // Union spans {0,1,2,3,4} = 5 directions but raw m = 9.
    std::vector<PairData> pairs;
    pairs.push_back(make_pair(3, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1, 2})));
    pairs.push_back(make_pair(3, make_pair_bar_Q(Cvir.data(), nao, n_vir, {1, 2, 3})));
    pairs.push_back(make_pair(3, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3, 4})));

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);

    EXPECT_EQ(tno.n_tno, 5);
    EXPECT_EQ(tno.n_dropped_overlap, 4); // 9 raw - 5 unique

    auto QtSQ = QtMQ(tno.Q_tno.data(), tno.n_tno, S.data(), nao);
    EXPECT_LT(max_abs_off_diag(QtSQ, tno.n_tno), 1e-9);
    EXPECT_LT(max_abs_diag_minus_one(QtSQ, tno.n_tno), 1e-9);

    auto QtFQ = QtMQ(tno.Q_tno.data(), tno.n_tno, F.data(), nao);
    real_t worst_off = 0.0;
    for (int a = 0; a < tno.n_tno; ++a)
        for (int b = 0; b < tno.n_tno; ++b)
            if (a != b) worst_off = std::max(worst_off,
                std::fabs(QtFQ[a * tno.n_tno + b]));
    EXPECT_LT(worst_off, 1e-9);
}

// =========================================================================
// 3. Empty pair: builder returns empty TNOData (defensive path).
// =========================================================================
TEST(DLPNOTno, EmptyPair_ReturnsEmpty)
{
    const int nao = 4;
    const auto S = make_overlap(nao, /*seed=*/1);
    const auto F = make_fock(nao, /*seed=*/2);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, nao);

    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, nao, {0, 1})));
    pairs.push_back(make_pair(0, std::vector<real_t>{}));   // empty
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, nao, {2, 3})));

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);
    EXPECT_EQ(tno.n_tno, 0);
    EXPECT_TRUE(tno.Q_tno.empty());
    EXPECT_TRUE(tno.eps_tno.empty());
}

// =========================================================================
// Phase 3.2.1 helpers + tests
// =========================================================================

namespace {

// Set pair.Y to a known symmetric matrix for testing the projection.
void set_pair_Y_symmetric(PairData& p, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int n = p.n_pno;
    p.Y.assign(static_cast<size_t>(n) * n, 0.0);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b <= a; ++b) {
            const real_t v = nd(rng);
            p.Y[a * n + b] = v;
            p.Y[b * n + a] = v;
        }
}

real_t fro_norm(const std::vector<real_t>& M)
{
    real_t s = 0.0;
    for (real_t v : M) s += v * v;
    return std::sqrt(s);
}

// Compute R^T · T_tno · R given R = Q_tno^T · S · bar_Q. Returns n_p × n_p.
std::vector<real_t> back_project_tno_to_pno(const TNOData& tno,
                                            const PairData& pair,
                                            const real_t* S_AO,
                                            int nao,
                                            const std::vector<real_t>& T_tno)
{
    const int n_tno = tno.n_tno;
    const int n_p   = pair.n_pno;
    Eigen::Map<const RowMatXd> Qtno(tno.Q_tno.data(), nao, n_tno);
    Eigen::Map<const RowMatXd> Smap(S_AO,             nao, nao);
    Eigen::Map<const RowMatXd> bQ  (pair.bar_Q.data(), nao, n_p);
    Eigen::Map<const RowMatXd> Tt  (T_tno.data(),      n_tno, n_tno);

    const RowMatXd S_bQ = Smap * bQ;
    const RowMatXd R    = Qtno.transpose() * S_bQ;
    const RowMatXd back = R.transpose() * Tt * R;

    std::vector<real_t> out(static_cast<size_t>(n_p) * n_p, 0.0);
    Eigen::Map<RowMatXd>(out.data(), n_p, n_p) = back;
    return out;
}

real_t trace_of(const std::vector<real_t>& M, int n)
{
    real_t t = 0.0;
    for (int i = 0; i < n; ++i) t += M[i * n + i];
    return t;
}

} // namespace

// =========================================================================
// 4. Phase 3.2.1 — round-trip recovers Y; trace and Frobenius norm preserved.
// =========================================================================
TEST(DLPNOTno, ProjectPnoToTno_RoundTripIdentityAndTracePreservation)
{
    const int nao = 8;
    const int n_vir = nao;
    const auto S = make_overlap(nao, /*seed=*/9);
    const auto F = make_fock(nao, /*seed=*/11);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);

    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {4, 5})));
    set_pair_Y_symmetric(pairs[0], /*seed=*/100);
    set_pair_Y_symmetric(pairs[1], /*seed=*/200);
    set_pair_Y_symmetric(pairs[2], /*seed=*/300);

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);
    ASSERT_EQ(tno.n_tno, 6);

    for (int p = 0; p < 3; ++p) {
        const PairData& pr = pairs[p];
        const auto T_tno = project_pno_to_tno(tno, pr, S.data(), nao);
        ASSERT_EQ(T_tno.size(), static_cast<size_t>(tno.n_tno) * tno.n_tno);

        // (a) Trace preserved: trace(T_tno) == trace(Y)  (because R^T R = I).
        const real_t tr_Y   = trace_of(pr.Y, pr.n_pno);
        const real_t tr_T   = trace_of(T_tno, tno.n_tno);
        EXPECT_NEAR(tr_T, tr_Y, 1e-10) << "trace not preserved for pair " << p;

        // (b) Frobenius norm preserved.
        EXPECT_NEAR(fro_norm(T_tno), fro_norm(pr.Y), 1e-10)
            << "Frobenius norm not preserved for pair " << p;

        // (c) Round trip:  R^T · T_tno · R == Y.
        const auto Y_back =
            back_project_tno_to_pno(tno, pr, S.data(), nao, T_tno);
        ASSERT_EQ(Y_back.size(), pr.Y.size());
        real_t worst = 0.0;
        for (size_t k = 0; k < Y_back.size(); ++k)
            worst = std::max(worst, std::fabs(Y_back[k] - pr.Y[k]));
        EXPECT_LT(worst, 1e-10) << "round-trip mismatch for pair " << p;
    }
}

// =========================================================================
// 5. Phase 3.2.1 — empty TNO or empty pair returns empty/zero tensor.
// =========================================================================
TEST(DLPNOTno, ProjectPnoToTno_EmptyInputsReturnEmpty)
{
    const int nao = 4;
    const auto S = make_overlap(nao, /*seed=*/1);

    // Empty TNO.
    {
        TNOData empty_tno;
        PairData p;
        p.n_pno = 2;
        p.bar_Q.assign(static_cast<size_t>(nao) * 2, 0.0);
        p.Y.assign(4, 1.0);
        const auto T = project_pno_to_tno(empty_tno, p, S.data(), nao);
        EXPECT_EQ(T.size(), 0u); // n_tno = 0 → 0×0 result
    }

    // Empty pair, non-empty TNO.
    {
        TNOData tno;
        tno.n_tno = 3;
        tno.Q_tno.assign(static_cast<size_t>(nao) * 3, 0.0);
        tno.eps_tno.assign(3, 0.0);
        PairData p;
        p.n_pno = 0;
        const auto T = project_pno_to_tno(tno, p, S.data(), nao);
        EXPECT_EQ(T.size(), 9u);
        for (real_t v : T) EXPECT_EQ(v, 0.0);
    }
}

// =========================================================================
// Phase 3.2.2 helpers + tests
// =========================================================================

namespace {

// Random AO 3-index tensor B[μ, ν, Q] (symmetric in μ, ν per Q).
std::vector<real_t> make_B_ao_ao(int nao, int naux, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<real_t> B(static_cast<size_t>(nao) * nao * naux, 0.0);
    for (int Q = 0; Q < naux; ++Q)
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu <= mu; ++nu) {
                const real_t v = nd(rng);
                B[(static_cast<size_t>(mu) * nao + nu) * naux + Q] = v;
                B[(static_cast<size_t>(nu) * nao + mu) * naux + Q] = v;
            }
    return B;
}

// Random LMO coefficient C_LMO (nao × nocc) — just random columns, no
// orthogonality requirement (Phase 3.2.2 algebra is independent of that).
std::vector<real_t> make_C_LMO(int nao, int nocc, uint32_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<real_t> C(static_cast<size_t>(nao) * nocc, 0.0);
    for (size_t k = 0; k < C.size(); ++k) C[k] = nd(rng);
    return C;
}

// Build B_lmo_ao[l, ν, Q] = Σ_μ C_LMO[μ, l] · B_ao_ao[μ, ν, Q].
std::vector<real_t> contract_lmo_ao(const real_t* C_LMO,
                                    const real_t* B_ao,
                                    int nao, int nocc, int naux)
{
    std::vector<real_t> B_lmo_ao(
        static_cast<size_t>(nocc) * nao * naux, 0.0);
    for (int l = 0; l < nocc; ++l)
        for (int nu = 0; nu < nao; ++nu)
            for (int Q = 0; Q < naux; ++Q) {
                real_t s = 0.0;
                for (int mu = 0; mu < nao; ++mu)
                    s += C_LMO[mu * nocc + l]
                       * B_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
                B_lmo_ao[(static_cast<size_t>(l) * nao + nu) * naux + Q] = s;
            }
    return B_lmo_ao;
}

// Build B_lmo_lmo[l, m, Q] = Σ_ν C_LMO[ν, m] · B_lmo_ao[l, ν, Q].
std::vector<real_t> contract_lmo_lmo(const real_t* C_LMO,
                                     const real_t* B_lmo_ao,
                                     int nao, int nocc, int naux)
{
    std::vector<real_t> B_lmo_lmo(
        static_cast<size_t>(nocc) * nocc * naux, 0.0);
    for (int l = 0; l < nocc; ++l)
        for (int m = 0; m < nocc; ++m)
            for (int Q = 0; Q < naux; ++Q) {
                real_t s = 0.0;
                for (int nu = 0; nu < nao; ++nu)
                    s += C_LMO[nu * nocc + m]
                       * B_lmo_ao[(static_cast<size_t>(l) * nao + nu) * naux + Q];
                B_lmo_lmo[(static_cast<size_t>(l) * nocc + m) * naux + Q] = s;
            }
    return B_lmo_lmo;
}

// Reference K[i_loc, a, d, c] = (i a | d c) computed by direct O(N^6)
// transformation: contract C_LMO and Q_tno against B then apply RI.
real_t ref_K_iadc(const real_t* B_ao,
                  const real_t* C_LMO,
                  const real_t* Q_tno,
                  int nao, int nocc, int naux, int n_tno,
                  int i_lmo, int a, int d, int c)
{
    real_t s = 0.0;
    for (int Q = 0; Q < naux; ++Q) {
        real_t b_iaQ = 0.0;
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                b_iaQ += C_LMO[mu * nocc + i_lmo] * Q_tno[nu * n_tno + a]
                       * B_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
        real_t b_dcQ = 0.0;
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                b_dcQ += Q_tno[mu * n_tno + d] * Q_tno[nu * n_tno + c]
                       * B_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
        s += b_iaQ * b_dcQ;
    }
    return s;
}

real_t ref_L_lcmn(const real_t* B_ao,
                  const real_t* C_LMO,
                  const real_t* Q_tno,
                  int nao, int nocc, int naux, int n_tno,
                  int l, int c, int m_lmo, int n_lmo)
{
    real_t s = 0.0;
    for (int Q = 0; Q < naux; ++Q) {
        real_t b_lcQ = 0.0;
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                b_lcQ += C_LMO[mu * nocc + l] * Q_tno[nu * n_tno + c]
                       * B_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
        real_t b_mnQ = 0.0;
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu)
                b_mnQ += C_LMO[mu * nocc + m_lmo] * C_LMO[nu * nocc + n_lmo]
                       * B_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
        s += b_lcQ * b_mnQ;
    }
    return s;
}

} // namespace

// =========================================================================
// 6. Phase 3.2.1 — triple-pair convenience wrapper.
// =========================================================================
TEST(DLPNOTno, ProjectTripleT2_AllThreePairsIndependent)
{
    const int nao = 8;
    const int n_vir = nao;
    const auto S = make_overlap(nao, /*seed=*/77);
    const auto F = make_fock(nao, /*seed=*/88);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);

    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {4, 5})));
    set_pair_Y_symmetric(pairs[0], /*seed=*/41);
    set_pair_Y_symmetric(pairs[1], /*seed=*/42);
    set_pair_Y_symmetric(pairs[2], /*seed=*/43);

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);

    const T2InTNO T = project_triple_t2_to_tno(
        tno, pairs[0], pairs[1], pairs[2], S.data(), nao);

    const size_t n2 = static_cast<size_t>(tno.n_tno) * tno.n_tno;
    EXPECT_EQ(T.T_ij.size(), n2);
    EXPECT_EQ(T.T_ik.size(), n2);
    EXPECT_EQ(T.T_jk.size(), n2);

    // Each block individually matches a direct project_pno_to_tno call.
    EXPECT_EQ(T.T_ij, project_pno_to_tno(tno, pairs[0], S.data(), nao));
    EXPECT_EQ(T.T_ik, project_pno_to_tno(tno, pairs[1], S.data(), nao));
    EXPECT_EQ(T.T_jk, project_pno_to_tno(tno, pairs[2], S.data(), nao));
}

// =========================================================================
// 7. Phase 3.2.2 — K and L tensors match a direct O(N^6) reference.
// =========================================================================
TEST(DLPNOTno, BuildEriInTno_MatchesDirectReference)
{
    const int nao = 6;
    const int nocc = 4;
    const int naux = 5;
    const int n_vir = nao;

    const auto S    = make_overlap(nao, /*seed=*/13);
    const auto F    = make_fock(nao, /*seed=*/14);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);
    const auto Bao  = make_B_ao_ao(nao, naux, /*seed=*/15);
    const auto Clmo = make_C_LMO(nao, nocc, /*seed=*/16);

    // Three pairs covering {0,1}, {2,3}, {1,2} so that the union span is
    // 4-dim and one direction is shared between two pairs (lin-dep dropped).
    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {1, 2})));

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);
    ASSERT_GE(tno.n_tno, 3);

    // Choose three triple LMOs.
    const int triple_lmos[3] = {0, 1, 2};

    // Precompute the global RI tensors the function expects.
    const auto B_lmo_ao  = contract_lmo_ao(Clmo.data(), Bao.data(),
                                           nao, nocc, naux);
    const auto B_lmo_lmo = contract_lmo_lmo(Clmo.data(), B_lmo_ao.data(),
                                            nao, nocc, naux);

    const ERIInTNO eri =
        build_eri_in_tno(tno, triple_lmos,
                         B_lmo_ao.data(), Bao.data(), B_lmo_lmo.data(),
                         nao, nocc, naux);

    ASSERT_EQ(eri.n_tno, tno.n_tno);
    ASSERT_EQ(eri.K_iadc.size(),
              static_cast<size_t>(3) * tno.n_tno * tno.n_tno * tno.n_tno);
    ASSERT_EQ(eri.L_lcmn.size(),
              static_cast<size_t>(nocc) * tno.n_tno * 9);

    const int n = tno.n_tno;
    real_t worst_K = 0.0;
    for (int i_loc = 0; i_loc < 3; ++i_loc)
        for (int a = 0; a < n; ++a)
            for (int d = 0; d < n; ++d)
                for (int c = 0; c < n; ++c) {
                    const real_t got = eri.K_iadc[
                        ((static_cast<size_t>(i_loc) * n + a) * n + d) * n + c];
                    const real_t exp = ref_K_iadc(
                        Bao.data(), Clmo.data(), tno.Q_tno.data(),
                        nao, nocc, naux, n,
                        triple_lmos[i_loc], a, d, c);
                    worst_K = std::max(worst_K, std::fabs(got - exp));
                }
    EXPECT_LT(worst_K, 1e-9);

    real_t worst_L = 0.0;
    for (int l = 0; l < nocc; ++l)
        for (int c = 0; c < n; ++c)
            for (int m_loc = 0; m_loc < 3; ++m_loc)
                for (int n_loc = 0; n_loc < 3; ++n_loc) {
                    const real_t got = eri.L_lcmn[
                        (static_cast<size_t>(l) * n + c) * 9 + m_loc * 3 + n_loc];
                    const real_t exp = ref_L_lcmn(
                        Bao.data(), Clmo.data(), tno.Q_tno.data(),
                        nao, nocc, naux, n,
                        l, c, triple_lmos[m_loc], triple_lmos[n_loc]);
                    worst_L = std::max(worst_L, std::fabs(got - exp));
                }
    EXPECT_LT(worst_L, 1e-9);
}

// =========================================================================
// 8. Phase 3.2.2 — empty TNO returns empty tensors (defensive path).
// =========================================================================
TEST(DLPNOTno, BuildEriInTno_EmptyTnoReturnsEmpty)
{
    TNOData empty;
    const int triple_lmos[3] = {0, 1, 2};
    std::vector<real_t> B_lmo_ao(1, 0.0), Bao(1, 0.0), B_lmo_lmo(1, 0.0);
    const ERIInTNO eri =
        build_eri_in_tno(empty, triple_lmos,
                         B_lmo_ao.data(), Bao.data(), B_lmo_lmo.data(),
                         /*nao=*/1, /*nocc=*/1, /*naux=*/1);
    EXPECT_EQ(eri.n_tno, 0);
    EXPECT_TRUE(eri.K_iadc.empty());
    EXPECT_TRUE(eri.L_lcmn.empty());
}

// =========================================================================
// 9. Phase 3.2.3a — oriented projection respects canonical pair order:
//    swap i↔j ⇒ transpose of TNO-basis T2.
// =========================================================================
TEST(DLPNOTno, ProjectPairT2Oriented_TransposeWhenSwapped)
{
    const int nao = 8;
    const int n_vir = nao;
    const int nocc = 4;     // synthetic LMO indices: 0..3
    const auto S = make_overlap(nao, /*seed=*/55);
    const auto F = make_fock(nao, /*seed=*/56);
    const auto Cvir = build_S_orthonormal_basis(S.data(), nao, n_vir);

    std::vector<PairData> pairs;
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {0, 1})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {2, 3})));
    pairs.push_back(make_pair(2, make_pair_bar_Q(Cvir.data(), nao, n_vir, {4, 5})));
    set_pair_Y_symmetric(pairs[0], 1001); // Y_canonical for pair 0
    set_pair_Y_symmetric(pairs[1], 1002);
    set_pair_Y_symmetric(pairs[2], 1003);

    // Single PairSetup per pair with canonical ordering (lmo_a, lmo_b)
    // where lmo_a ≤ lmo_b. Use 3 logical pairs:
    //   pair 0 = (0, 1)
    //   pair 1 = (1, 2)   ← we'll test transpose here
    //   pair 2 = (2, 3)
    // pair_lookup maps both (a, b) and (b, a) to the same index.
    std::vector<PairSetup> setups(3);
    setups[0].i = 0; setups[0].j = 1;
    setups[1].i = 1; setups[1].j = 2;
    setups[2].i = 2; setups[2].j = 3;
    std::vector<int> pair_lookup(nocc * nocc, -1);
    pair_lookup[0 * nocc + 1] = 0; pair_lookup[1 * nocc + 0] = 0;
    pair_lookup[1 * nocc + 2] = 1; pair_lookup[2 * nocc + 1] = 1;
    pair_lookup[2 * nocc + 3] = 2; pair_lookup[3 * nocc + 2] = 2;

    TNOBuilder builder(pairs, F.data(), S.data(), nao);
    const TNOData tno = builder.build_for_triple(0, 1, 2);

    // (a) Canonical query (1, 2) returns the direct projection of pair 1.
    const auto T_12 = project_pair_t2_oriented_to_tno(
        tno, pairs, setups, pair_lookup,
        /*lmo_p=*/1, /*lmo_q=*/2,
        S.data(), nao, nocc);
    const auto T_direct = project_pno_to_tno(tno, pairs[1], S.data(), nao);
    EXPECT_EQ(T_12, T_direct);

    // (b) Reversed query (2, 1) returns the transpose.
    const auto T_21 = project_pair_t2_oriented_to_tno(
        tno, pairs, setups, pair_lookup,
        /*lmo_p=*/2, /*lmo_q=*/1,
        S.data(), nao, nocc);
    ASSERT_EQ(T_21.size(), T_direct.size());
    const int n = tno.n_tno;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            EXPECT_NEAR(T_21[a * n + b], T_direct[b * n + a], 1e-14);

    // (c) Missing pair (0, 3) returns zeros.
    const auto T_missing = project_pair_t2_oriented_to_tno(
        tno, pairs, setups, pair_lookup,
        /*lmo_p=*/0, /*lmo_q=*/3,
        S.data(), nao, nocc);
    EXPECT_EQ(T_missing.size(), static_cast<size_t>(n) * n);
    for (real_t v : T_missing) EXPECT_EQ(v, 0.0);
}
