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
 * @file test_bt_pno_backtransform.cu
 * @brief Deterministic layout/transpose guards for the bt-PNO-STEOM P5a
 *        PNO → canonical-MO back-transform (Plan "Option B").
 *
 * These tests are pure-CPU: they hand-build a tiny synthetic DLPNOLMP2Result
 * (no SCF, no GPU) and assert that bt_pno_to_canonical produces exactly the
 * analytic U·Y·U^T placement, with the correct chemist layout and the i↔j
 * transpose symmetry T2[I,J,a,b] == T2[J,I,b,a]. They isolate the
 * highest-risk convention points (isometry direction, a-b transpose,
 * off-diagonal orientation) before the physics gate (full no-truncation
 * equivalence to canonical CCSD T2) runs inside the DLPNO driver.
 */

#include <gtest/gtest.h>

#include <vector>

#include "bt_pno_backtransform.hpp"
#include "dlpno_mp2.hpp"
#include "dlpno_pair_data.hpp"

using namespace gansu;

namespace {

// Reference: U[a][a'] = Σ_μν C_vir[μ][a] S[μ][ν] bar_Q[ν][a']
//            T2ij[a][b] = Σ_{a'b'} U[a][a'] Y[a'][b'] U[b][b']
// All matrices row-major (C_vir: nao×nvir, S: nao×nao, bar_Q: nao×n_pno,
// Y: n_pno×n_pno). Returns T2ij as flat nvir×nvir row-major.
std::vector<real_t> ref_pair_block(const std::vector<real_t>& C_vir,
                                   const std::vector<real_t>& S,
                                   const std::vector<real_t>& bar_Q,
                                   const std::vector<real_t>& Y,
                                   int nao, int nvir, int n_pno) {
    // SbarQ[μ][a'] = Σ_ν S[μ][ν] bar_Q[ν][a']
    std::vector<real_t> SbarQ((size_t)nao * n_pno, 0.0);
    for (int mu = 0; mu < nao; ++mu)
        for (int ap = 0; ap < n_pno; ++ap) {
            real_t acc = 0.0;
            for (int nu = 0; nu < nao; ++nu)
                acc += S[(size_t)mu * nao + nu] * bar_Q[(size_t)nu * n_pno + ap];
            SbarQ[(size_t)mu * n_pno + ap] = acc;
        }
    // U[a][a'] = Σ_μ C_vir[μ][a] SbarQ[μ][a']
    std::vector<real_t> U((size_t)nvir * n_pno, 0.0);
    for (int a = 0; a < nvir; ++a)
        for (int ap = 0; ap < n_pno; ++ap) {
            real_t acc = 0.0;
            for (int mu = 0; mu < nao; ++mu)
                acc += C_vir[(size_t)mu * nvir + a] * SbarQ[(size_t)mu * n_pno + ap];
            U[(size_t)a * n_pno + ap] = acc;
        }
    // T2ij[a][b] = Σ_{a'b'} U[a][a'] Y[a'][b'] U[b][b']
    std::vector<real_t> T2ij((size_t)nvir * nvir, 0.0);
    for (int a = 0; a < nvir; ++a)
        for (int b = 0; b < nvir; ++b) {
            real_t acc = 0.0;
            for (int ap = 0; ap < n_pno; ++ap)
                for (int bp = 0; bp < n_pno; ++bp)
                    acc += U[(size_t)a * n_pno + ap] * Y[(size_t)ap * n_pno + bp]
                         * U[(size_t)b * n_pno + bp];
            T2ij[(size_t)a * nvir + b] = acc;
        }
    return T2ij;
}

}  // namespace

// Single diagonal pair (nocc=1): isolates Steps 1+2 (U_loc trivially identity).
// Asserts the canonical block equals the analytic U·Y·U^T with exact layout.
TEST(BTBackTransform, SyntheticDiagonalPair_LayoutExact) {
    const int nao = 4, nvir = 3, n_pno = 2;

    // Non-orthogonal overlap (general S) to exercise the C_vir^T·S·bar_Q path.
    std::vector<real_t> S = {
        1.00, 0.10, 0.00, 0.05,
        0.10, 1.00, 0.20, 0.00,
        0.00, 0.20, 1.00, 0.15,
        0.05, 0.00, 0.15, 1.00};
    std::vector<real_t> C_vir = {  // [nao × nvir]
        0.30, -0.20, 0.10,
        0.50, 0.40, -0.30,
        -0.10, 0.60, 0.20,
        0.20, 0.10, 0.70};
    std::vector<real_t> bar_Q = {  // [nao × n_pno]
        0.40, 0.10,
        -0.20, 0.50,
        0.30, -0.10,
        0.10, 0.60};
    std::vector<real_t> Y = {  // [n_pno × n_pno] (not symmetric on purpose)
        0.12, -0.03,
        0.05, 0.09};

    DLPNOLMP2Result res;
    res.nao = nao;
    res.nocc = 1;
    res.setups.resize(1);
    res.setups[0].i = 0;
    res.setups[0].j = 0;
    res.setups[0].n_pao = 0;
    res.pairs.resize(1);
    res.pairs[0].n_pno = n_pno;
    res.pairs[0].bar_Q = bar_Q;
    res.pairs[0].Y = Y;
    res.pair_lookup = {0};

    const std::vector<real_t> U_loc = {1.0};  // 1×1 identity

    BTAmplitudes out = bt_pno_to_canonical(
        res, U_loc, C_vir, S.data(), nao, nvir, /*T1_pao=*/{}, /*include_t1=*/false);

    ASSERT_EQ(out.T2.size(), (size_t)1 * 1 * nvir * nvir);
    const std::vector<real_t> ref = ref_pair_block(C_vir, S, bar_Q, Y, nao, nvir, n_pno);
    for (int a = 0; a < nvir; ++a)
        for (int b = 0; b < nvir; ++b)
            EXPECT_NEAR(out.T2[(size_t)a * nvir + b], ref[(size_t)a * nvir + b], 1e-12)
                << "mismatch at (a,b)=(" << a << "," << b << ")";
}

// Off-diagonal stored pair (0,1) with nocc=2: asserts the (j,i) orientation is
// the (a,b)-transpose of the (i,j) block and matches the analytic value (no
// pair_factor scaling). U_loc identity isolates Step 3.
TEST(BTBackTransform, SyntheticOffDiagonal_TransposeFill) {
    const int nao = 4, nvir = 3, n_pno = 2;
    const int nocc = 2;

    std::vector<real_t> S = {
        1.00, 0.08, 0.00, 0.03,
        0.08, 1.00, 0.12, 0.00,
        0.00, 0.12, 1.00, 0.09,
        0.03, 0.00, 0.09, 1.00};
    std::vector<real_t> C_vir = {
        0.25, -0.15, 0.20,
        0.45, 0.35, -0.10,
        -0.05, 0.55, 0.30,
        0.15, 0.05, 0.65};
    std::vector<real_t> bar_Q01 = {
        0.35, 0.15,
        -0.25, 0.45,
        0.20, -0.05,
        0.05, 0.55};
    std::vector<real_t> Y01 = {
        0.10, -0.04,
        0.06, 0.07};

    DLPNOLMP2Result res;
    res.nao = nao;
    res.nocc = nocc;
    // Stored pairs (i≤j): (0,0)=idx0, (0,1)=idx1, (1,1)=idx2. Only (0,1) carries
    // amplitudes here; diagonals left with n_pno=0 (skipped by the transform).
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    res.pairs[0].n_pno = 0;
    res.pairs[1].n_pno = n_pno;
    res.pairs[1].bar_Q = bar_Q01;
    res.pairs[1].Y = Y01;
    res.pairs[2].n_pno = 0;
    res.pair_lookup.assign((size_t)nocc * nocc, 0);
    res.pair_lookup[0 * nocc + 0] = 0;
    res.pair_lookup[0 * nocc + 1] = 1;
    res.pair_lookup[1 * nocc + 0] = 1;
    res.pair_lookup[1 * nocc + 1] = 2;

    const std::vector<real_t> U_loc = {1.0, 0.0, 0.0, 1.0};  // 2×2 identity

    BTAmplitudes out = bt_pno_to_canonical(
        res, U_loc, C_vir, S.data(), nao, nvir, /*T1_pao=*/{}, /*include_t1=*/false);

    ASSERT_EQ(out.T2.size(), (size_t)nocc * nocc * nvir * nvir);
    const size_t vv = (size_t)nvir * nvir;
    const std::vector<real_t> ref01 = ref_pair_block(C_vir, S, bar_Q01, Y01, nao, nvir, n_pno);

    const real_t* blk01 = out.T2.data() + ((size_t)0 * nocc + 1) * vv;
    const real_t* blk10 = out.T2.data() + ((size_t)1 * nocc + 0) * vv;
    for (int a = 0; a < nvir; ++a)
        for (int b = 0; b < nvir; ++b) {
            // (0,1) block matches the analytic U·Y·U^T.
            EXPECT_NEAR(blk01[(size_t)a * nvir + b], ref01[(size_t)a * nvir + b], 1e-12)
                << "(0,1) mismatch at (" << a << "," << b << ")";
            // (1,0) block is the (a,b)-transpose of the (0,1) block.
            EXPECT_NEAR(blk10[(size_t)a * nvir + b], blk01[(size_t)b * nvir + a], 1e-12)
                << "transpose-fill mismatch at (" << a << "," << b << ")";
        }
}

// Non-identity occupied rotation (nocc=2): the DECISIVE U_loc test. The
// synthetic identity-U_loc tests above pass even if Step 3 transposed U_loc,
// and the physics gate (DLPNO driver) cannot isolate U_loc because DLPNO
// truncation dominates and is localizer-dependent. Here U_loc is a genuine
// 2×2 rotation (non-symmetric), so a U_loc↔U_loc^T swap in
//   T2_can[I,J,a,b] = Σ_{ij} U_loc[I,i] U_loc[J,j] T2_lmo[i,j,a,b]
// changes the result and is caught. All three stored pairs carry amplitudes
// so the rotation genuinely mixes the LMO grid.
TEST(BTBackTransform, SyntheticOccRotation_ULocApplied) {
    const int nao = 3, nvir = 2, n_pno = 2, nocc = 2;

    std::vector<real_t> S = {
        1.00, 0.07, 0.04,
        0.07, 1.00, 0.11,
        0.04, 0.11, 1.00};
    std::vector<real_t> C_vir = {  // [nao × nvir]
        0.40, -0.20,
        0.30, 0.50,
        -0.10, 0.60};
    std::vector<real_t> barQ00 = {0.50, 0.10, -0.20, 0.40, 0.30, -0.05};
    std::vector<real_t> Y00    = {0.14, -0.02, 0.03, 0.11};
    std::vector<real_t> barQ01 = {0.35, 0.15, -0.25, 0.45, 0.20, -0.05};
    std::vector<real_t> Y01    = {0.10, -0.04, 0.06, 0.07};
    std::vector<real_t> barQ11 = {0.20, 0.55, 0.40, -0.10, -0.05, 0.30};
    std::vector<real_t> Y11    = {0.09, 0.05, -0.03, 0.12};

    DLPNOLMP2Result res;
    res.nao = nao;
    res.nocc = nocc;
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    res.pairs[0].n_pno = n_pno; res.pairs[0].bar_Q = barQ00; res.pairs[0].Y = Y00;
    res.pairs[1].n_pno = n_pno; res.pairs[1].bar_Q = barQ01; res.pairs[1].Y = Y01;
    res.pairs[2].n_pno = n_pno; res.pairs[2].bar_Q = barQ11; res.pairs[2].Y = Y11;
    res.pair_lookup.assign((size_t)nocc * nocc, 0);
    res.pair_lookup[0 * nocc + 0] = 0;
    res.pair_lookup[0 * nocc + 1] = 1;
    res.pair_lookup[1 * nocc + 0] = 1;
    res.pair_lookup[1 * nocc + 1] = 2;

    // Genuine rotation: U_loc[I*nocc+i], non-symmetric (c=0.8, s=0.6).
    const std::vector<real_t> U_loc = {0.8, -0.6, 0.6, 0.8};

    BTAmplitudes out = bt_pno_to_canonical(
        res, U_loc, C_vir, S.data(), nao, nvir, /*T1_pao=*/{}, /*include_t1=*/false);

    // Reference: build the full LMO grid (with (1,0) = (a,b)-transpose of (0,1)),
    // then apply the occupied rotation exactly as specified.
    const size_t vv = (size_t)nvir * nvir;
    const std::vector<real_t> b00 = ref_pair_block(C_vir, S, barQ00, Y00, nao, nvir, n_pno);
    const std::vector<real_t> b01 = ref_pair_block(C_vir, S, barQ01, Y01, nao, nvir, n_pno);
    const std::vector<real_t> b11 = ref_pair_block(C_vir, S, barQ11, Y11, nao, nvir, n_pno);
    auto lmo = [&](int i, int j, int a, int b) -> real_t {
        if (i == 0 && j == 0) return b00[(size_t)a * nvir + b];
        if (i == 0 && j == 1) return b01[(size_t)a * nvir + b];
        if (i == 1 && j == 1) return b11[(size_t)a * nvir + b];
        /* (1,0) = (a,b)-transpose of (0,1) */ return b01[(size_t)b * nvir + a];
    };
    for (int I = 0; I < nocc; ++I)
        for (int J = 0; J < nocc; ++J)
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    real_t expect = 0.0;
                    for (int i = 0; i < nocc; ++i)
                        for (int j = 0; j < nocc; ++j)
                            expect += U_loc[(size_t)I * nocc + i] * U_loc[(size_t)J * nocc + j]
                                    * lmo(i, j, a, b);
                    const real_t got = out.T2[(((size_t)I * nocc + J) * nvir + a) * nvir + b];
                    EXPECT_NEAR(got, expect, 1e-12)
                        << "U_loc rotation mismatch at (I,J,a,b)=("
                        << I << "," << J << "," << a << "," << b << ")";
                }
}
