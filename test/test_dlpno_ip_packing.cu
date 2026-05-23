/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_dlpno_ip_packing.cu
 * @brief Deterministic bookkeeping guards for the DLPNO-IP-EOM packed-vector
 *        layout (stage B Phase B0). Pure CPU, no SCF/GPU — builds a synthetic
 *        DLPNOLMP2Result and checks the offset table is contiguous, non-
 *        overlapping, and that total_dim accounts for both orientations of
 *        off-diagonal pairs and a single block for diagonal pairs.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "dlpno_ip_packing.hpp"
#include "dlpno_ip_eom_operator.hpp"
#include "dlpno_ip_eom_transform.hpp"
#include "dlpno_ip_eom_projected_operator.hpp"
#include "linear_operator.hpp"
#include "dlpno_mp2.hpp"
#include "dlpno_pair_data.hpp"

using namespace gansu;

namespace {
// nocc=2 → stored pairs (i≤j): (0,0), (0,1), (1,1).
DLPNOLMP2Result make_res(int n00, int n01, int n11) {
    DLPNOLMP2Result res;
    res.nocc = 2;
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    res.pairs[0].n_pno = n00;
    res.pairs[1].n_pno = n01;
    res.pairs[2].n_pno = n11;
    return res;
}
}  // namespace

TEST(DLPNOIPPacking, OffsetsContiguousAndTotalDim) {
    // n_pno: (0,0)=2, (0,1)=3, (1,1)=2.
    // total = nocc(2) + 2 [diag] + 3+3 [off-diag, both orientations] + 2 [diag] = 12.
    const DLPNOIPPacking p = build_ip_packing(make_res(2, 3, 2));

    EXPECT_EQ(p.nocc, 2);
    EXPECT_EQ(p.total_dim, 12);

    // Diagonal flags.
    EXPECT_TRUE(p.diagonal(0));
    EXPECT_FALSE(p.diagonal(1));
    EXPECT_TRUE(p.diagonal(2));

    // (0,0): single block at [2,4).
    EXPECT_EQ(p.off_ij[0], 2);
    EXPECT_EQ(p.off_ji[0], -1);
    // (0,1): two blocks (ij)=[4,7), (ji)=[7,10).
    EXPECT_EQ(p.off_ij[1], 4);
    EXPECT_EQ(p.off_ji[1], 7);
    // (1,1): single block at [10,12).
    EXPECT_EQ(p.off_ij[2], 10);
    EXPECT_EQ(p.off_ji[2], -1);

    // Contiguity: walk all blocks in packed order, assert they tile [nocc, total_dim)
    // exactly with no gap/overlap.
    std::vector<std::pair<int,int>> blocks;  // (start, len)
    for (int idx = 0; idx < 3; ++idx) {
        blocks.emplace_back(p.off_ij[idx], p.n_pno[idx]);
        if (!p.diagonal(idx)) blocks.emplace_back(p.off_ji[idx], p.n_pno[idx]);
    }
    std::sort(blocks.begin(), blocks.end());
    int cursor = p.nocc;
    for (const auto& b : blocks) {
        EXPECT_EQ(b.first, cursor) << "gap/overlap at packed offset " << cursor;
        cursor += b.second;
    }
    EXPECT_EQ(cursor, p.total_dim);
}

TEST(DLPNOIPPacking, EmptyPnoBlocksHandled) {
    // A pair with n_pno=0 contributes a zero-length block (still gets an offset).
    const DLPNOIPPacking p = build_ip_packing(make_res(0, 4, 0));
    EXPECT_EQ(p.total_dim, 2 + 0 + 4 + 4 + 0);  // = 10
    EXPECT_EQ(p.off_ij[0], 2);
    EXPECT_EQ(p.off_ij[1], 2);   // (0,0) was empty → (0,1) starts at same offset
    EXPECT_EQ(p.off_ji[1], 6);
    EXPECT_EQ(p.off_ij[2], 10);
}

// B0.2: diagonal-only operator. Checks the packed diagonal is built with the
// Koopmans 1h entries (-ε_i) and the 2h1p denominators (-F_ii-F_jj+Λ_a), with
// both orientations of the off-diagonal pair sharing the same energies.
TEST(DLPNOIPEOMOperator, DiagonalConstruction) {
    DLPNOLMP2Result res = make_res(2, 2, 2);  // pairs (0,0),(0,1),(1,1), n_pno=2 each
    res.setups[0].F_ii = -0.5; res.setups[0].F_jj = -0.5;
    res.setups[1].F_ii = -0.5; res.setups[1].F_jj = -0.4;
    res.setups[2].F_ii = -0.4; res.setups[2].F_jj = -0.4;
    res.pairs[0].Lambda = {0.1, 0.2};
    res.pairs[1].Lambda = {0.15, 0.25};
    res.pairs[2].Lambda = {0.12, 0.22};
    const std::vector<real_t> eps_o = {-0.5, -0.4};

    const DLPNOIPPacking pack = build_ip_packing(res);
    DLPNOIPEOMCCSDOperator op(res, pack, eps_o);

    EXPECT_EQ(op.dimension(), 10);
    const std::vector<real_t>& D = op.host_diagonal();
    ASSERT_EQ(static_cast<int>(D.size()), 10);

    EXPECT_NEAR(D[0], 0.5, 1e-12);   // 1h: -ε_0
    EXPECT_NEAR(D[1], 0.4, 1e-12);   // 1h: -ε_1
    EXPECT_NEAR(D[2], 1.1, 1e-12);   // (0,0) a=0: 0.5+0.5+0.1
    EXPECT_NEAR(D[3], 1.2, 1e-12);   // (0,0) a=1
    EXPECT_NEAR(D[4], 1.05, 1e-12);  // (0,1) ij a=0: 0.5+0.4+0.15
    EXPECT_NEAR(D[5], 1.15, 1e-12);  // (0,1) ij a=1
    EXPECT_NEAR(D[6], 1.05, 1e-12);  // (0,1) ji a=0 (same energies)
    EXPECT_NEAR(D[7], 1.15, 1e-12);  // (0,1) ji a=1
    EXPECT_NEAR(D[8], 0.92, 1e-12);  // (1,1) a=0: 0.4+0.4+0.12
    EXPECT_NEAR(D[9], 1.02, 1e-12);  // (1,1) a=1
}

// B1: single-index R2 PNO↔canonical round-trip. With n_pno = nvir and an
// orthogonal U^(ij) (S=I, C_vir=I, bar_Q orthogonal), the PNO space spans the
// full virtual, so canonical→packed→canonical is the identity. Run with both
// U_loc = identity and U_loc = a genuine rotation to exercise the occupied
// rotation + both off-diagonal orientations.
namespace {
void run_roundtrip(const std::vector<real_t>& U_loc) {
    const int nao = 3, nvir = 3, nocc = 2, n = 3;  // n_pno = nvir → full span

    // Orthogonal 3×3 (rows/cols orthonormal) used as bar_Q for every pair.
    const std::vector<real_t> Q = {
        2.0/3, -2.0/3,  1.0/3,
        2.0/3,  1.0/3, -2.0/3,
        1.0/3,  2.0/3,  2.0/3};
    std::vector<real_t> S(nao * nao, 0.0);
    for (int k = 0; k < nao; ++k) S[k * nao + k] = 1.0;  // S = I
    std::vector<real_t> C_vir(nao * nvir, 0.0);
    for (int k = 0; k < nvir; ++k) C_vir[k * nvir + k] = 1.0;  // C_vir = I

    DLPNOLMP2Result res;
    res.nocc = nocc;
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    for (int idx = 0; idx < 3; ++idx) { res.pairs[idx].n_pno = n; res.pairs[idx].bar_Q = Q; }

    const DLPNOIPPacking pack = build_ip_packing(res);

    // Arbitrary canonical R2 [nocc²·nvir]; both (I,J) orderings independent.
    std::vector<real_t> R2(static_cast<size_t>(nocc) * nocc * nvir);
    for (size_t k = 0; k < R2.size(); ++k) R2[k] = 0.1 * (k + 1) - 0.03 * (k % 5);

    const std::vector<real_t> packed =
        ip_canonical_r2_to_packed(res, pack, U_loc, C_vir, S.data(), nao, nvir, R2);
    const std::vector<real_t> R2b =
        ip_packed_r2_to_canonical(res, pack, U_loc, C_vir, S.data(), nao, nvir, packed);

    ASSERT_EQ(R2b.size(), R2.size());
    for (size_t k = 0; k < R2.size(); ++k)
        EXPECT_NEAR(R2b[k], R2[k], 1e-10) << "round-trip mismatch at " << k;
}
}  // namespace

TEST(DLPNOIPEOMTransform, RoundTrip_ULocIdentity) {
    run_roundtrip({1.0, 0.0, 0.0, 1.0});
}

TEST(DLPNOIPEOMTransform, RoundTrip_ULocRotation) {
    run_roundtrip({0.8, -0.6, 0.6, 0.8});  // genuine 2×2 rotation
}

// B1b: project-up operator mechanics. A trivial inner σ_canon = c·x, wrapped by
// the projected operator at full PNO (n_pno=nvir, orthogonal U → P is a
// bijection), must reproduce c·x in packed coords. Validates lift → inner
// device apply → project-down → R1 passthrough + the host↔device round-trips.
namespace {
// Inner LinearOperator that scales by a constant (host round-trip; test only).
class ScaleOperator : public LinearOperator {
public:
    ScaleOperator(int dim, real_t c) : dim_(dim), c_(c) {}
    void apply(const real_t* d_in, real_t* d_out) const override {
        std::vector<real_t> h(dim_);
        cudaMemcpy(h.data(), d_in, dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);
        for (auto& v : h) v *= c_;
        cudaMemcpy(d_out, h.data(), dim_ * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    void apply_preconditioner(const real_t* d_in, real_t* d_out) const override {
        cudaMemcpy(d_out, d_in, dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
    }
    const real_t* get_diagonal_device() const override { return nullptr; }
    int dimension() const override { return dim_; }
    std::string name() const override { return "ScaleOperator"; }
private:
    int dim_;
    real_t c_;
};
}  // namespace

TEST(DLPNOIPEOMProjected, FullPNO_PreservesScalarInner) {
    const int nao = 3, nvir = 3, nocc = 2, n = 3;  // full PNO
    const std::vector<real_t> Q = {
        2.0/3, -2.0/3,  1.0/3,
        2.0/3,  1.0/3, -2.0/3,
        1.0/3,  2.0/3,  2.0/3};
    std::vector<real_t> S(nao * nao, 0.0);
    for (int k = 0; k < nao; ++k) S[k * nao + k] = 1.0;
    std::vector<real_t> C_vir(nao * nvir, 0.0);
    for (int k = 0; k < nvir; ++k) C_vir[k * nvir + k] = 1.0;

    DLPNOLMP2Result res;
    res.nocc = nocc;
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    for (int idx = 0; idx < 3; ++idx) {
        res.pairs[idx].n_pno = n;
        res.pairs[idx].bar_Q = Q;
        res.pairs[idx].Lambda = {0.1, 0.2, 0.3};  // for the (unused here) diagonal
        res.setups[idx].F_ii = -0.5;
        res.setups[idx].F_jj = -0.4;
    }
    const std::vector<real_t> eps_o = {-0.5, -0.4};
    const std::vector<real_t> U_loc = {0.8, -0.6, 0.6, 0.8};  // genuine rotation

    const DLPNOIPPacking pack = build_ip_packing(res);
    EXPECT_EQ(pack.total_dim, nocc + nocc * nocc * nvir);  // full PNO: packed == canonical

    ScaleOperator inner(pack.total_dim, 2.0);
    DLPNOIPEOMProjectedOperator op(inner, res, pack, U_loc, C_vir, S.data(), nao, nvir, eps_o);

    ASSERT_EQ(op.dimension(), pack.total_dim);
    const int dim = pack.total_dim;
    std::vector<real_t> x(dim);
    for (int k = 0; k < dim; ++k) x[k] = 0.07 * (k + 1) - 0.02 * (k % 3);

    real_t* d_in = nullptr;
    real_t* d_out = nullptr;
    cudaMalloc(&d_in, dim * sizeof(real_t));
    cudaMalloc(&d_out, dim * sizeof(real_t));
    cudaMemcpy(d_in, x.data(), dim * sizeof(real_t), cudaMemcpyHostToDevice);
    op.apply(d_in, d_out);
    std::vector<real_t> y(dim);
    cudaMemcpy(y.data(), d_out, dim * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    for (int k = 0; k < dim; ++k)
        EXPECT_NEAR(y[k], 2.0 * x[k], 1e-10) << "projected scalar mismatch at " << k;
}
