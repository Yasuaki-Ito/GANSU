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
 * @file test_dlpno_ea_packing.cu
 * @brief Deterministic guards for the DLPNO-EA-EOM packed layout + two-virtual
 *        R2 PNO↔canonical transform (stage B). Pure CPU, no SCF/GPU.
 */

#include <gtest/gtest.h>

#include <vector>

#include "dlpno_ea_packing.hpp"
#include "dlpno_ea_eom_transform.hpp"
#include "dlpno_mp2.hpp"
#include "dlpno_pair_data.hpp"

using namespace gansu;

namespace {
// nocc=2, nao=5 → nvir=3. Diagonal pairs (0,0),(1,1); off-diagonal (0,1) present.
DLPNOLMP2Result make_res(int n00, int n11, const std::vector<real_t>& barQ_diag) {
    DLPNOLMP2Result res;
    res.nocc = 2;
    res.nao = 5;
    res.setups.resize(3);
    res.setups[0].i = 0; res.setups[0].j = 0;
    res.setups[1].i = 0; res.setups[1].j = 1;
    res.setups[2].i = 1; res.setups[2].j = 1;
    res.pairs.resize(3);
    res.pairs[0].n_pno = n00;
    res.pairs[1].n_pno = 0;     // off-diagonal pair unused by EA 2p1h
    res.pairs[2].n_pno = n11;
    res.pairs[0].bar_Q = barQ_diag;
    res.pairs[2].bar_Q = barQ_diag;
    res.pair_lookup.assign(2 * 2, 0);
    res.pair_lookup[0 * 2 + 0] = 0;
    res.pair_lookup[0 * 2 + 1] = 1;
    res.pair_lookup[1 * 2 + 0] = 1;
    res.pair_lookup[1 * 2 + 1] = 2;
    return res;
}
}  // namespace

TEST(DLPNOEAPacking, OffsetsAndTotalDim) {
    // n_pno(00)=2, n_pno(11)=3, nvir=3. total = nvir(3) + 2² + 3² = 3+4+9 = 16.
    const DLPNOLMP2Result res = make_res(2, 3, /*barQ unused here*/ {});
    const DLPNOEAPacking p = build_ea_packing(res);
    EXPECT_EQ(p.nvir, 3);
    EXPECT_EQ(p.nocc, 2);
    EXPECT_EQ(p.total_dim, 16);
    EXPECT_EQ(p.n_pno_ii[0], 2);
    EXPECT_EQ(p.n_pno_ii[1], 3);
    EXPECT_EQ(p.off_i[0], 3);       // after R1 (nvir)
    EXPECT_EQ(p.off_i[1], 3 + 4);   // after i=0's 2×2 block
}

// Two-virtual R2 round-trip. Full PNO (n_pno=nvir=3, orthogonal U^(ii) via
// S=I, C_vir = [I_3;0], bar_Q = [R;0] with R orthogonal) → the per-i virtual
// projection is a bijection, so canonical→packed→canonical is the identity.
namespace {
void run_ea_roundtrip(const std::vector<real_t>& U_loc) {
    const int nao = 5, nvir = 3, nocc = 2, n = 3;
    // R: 3×3 orthogonal.
    const std::vector<real_t> R = {
        2.0/3, -2.0/3,  1.0/3,
        2.0/3,  1.0/3, -2.0/3,
        1.0/3,  2.0/3,  2.0/3};
    // bar_Q [nao × n] = [R; 0] (rows 0-2 = R, rows 3-4 = 0).
    std::vector<real_t> barQ((size_t)nao * n, 0.0);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) barQ[(size_t)r * n + c] = R[(size_t)r * 3 + c];
    // C_vir [nao × nvir] = [I_3; 0].
    std::vector<real_t> C_vir((size_t)nao * nvir, 0.0);
    for (int k = 0; k < 3; ++k) C_vir[(size_t)k * nvir + k] = 1.0;
    std::vector<real_t> S(nao * nao, 0.0);
    for (int k = 0; k < nao; ++k) S[k * nao + k] = 1.0;

    const DLPNOLMP2Result res = make_res(n, n, barQ);
    const DLPNOEAPacking pack = build_ea_packing(res);

    std::vector<real_t> R2(static_cast<size_t>(nocc) * nvir * nvir);
    for (size_t k = 0; k < R2.size(); ++k) R2[k] = 0.05 * (k + 1) - 0.02 * (k % 4);

    const std::vector<real_t> packed =
        ea_canonical_r2_to_packed(res, pack, U_loc, C_vir, S.data(), nao, R2);
    const std::vector<real_t> R2b =
        ea_packed_r2_to_canonical(res, pack, U_loc, C_vir, S.data(), nao, packed);

    ASSERT_EQ(R2b.size(), R2.size());
    for (size_t k = 0; k < R2.size(); ++k)
        EXPECT_NEAR(R2b[k], R2[k], 1e-10) << "EA round-trip mismatch at " << k;
}
}  // namespace

TEST(DLPNOEAEOMTransform, RoundTrip_ULocIdentity) {
    run_ea_roundtrip({1.0, 0.0, 0.0, 1.0});
}

TEST(DLPNOEAEOMTransform, RoundTrip_ULocRotation) {
    run_ea_roundtrip({0.8, -0.6, 0.6, 0.8});
}
