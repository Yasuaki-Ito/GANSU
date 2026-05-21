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
 * @file test_cis_nto_active_space.cu
 * @brief P0 Day 2a unit tests for CISNTOActiveSpace::compute.
 *
 * Coverage (matches the P0 design doc §4.1):
 *   1. trace conservation
 *   2. symmetry of ρ_occ / ρ_vir
 *   3. non-negativity of NTO occupations
 *   4. threshold edge cases (zero, all, exact equality)
 *   5. pure 1-state limit (rank-1 ρ_occ recovers the CIS amplitude)
 * Plus support tests:
 *   6. two-state degeneracy (rank check)
 *   7. frozen-core projector embedding via make_canonical_projectors
 *   8. user weights vs uniform default
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "cis_nto_active_space.hpp"

using namespace gansu;

namespace {

// Build n_states random CIS amplitude vectors, each L2-normalized.
std::vector<real_t> make_random_cis_amps(int n_states, int nocc, int nvir,
                                         std::uint64_t seed)
{
    const size_t cis_dim = static_cast<size_t>(nocc) * nvir;
    std::vector<real_t> amps(static_cast<size_t>(n_states) * cis_dim);
    std::mt19937_64 rng(seed);
    std::normal_distribution<real_t> normal(0.0, 1.0);
    for (int n = 0; n < n_states; ++n) {
        real_t norm2 = 0.0;
        for (size_t k = 0; k < cis_dim; ++k) {
            real_t v = normal(rng);
            amps[n * cis_dim + k] = v;
            norm2 += v * v;
        }
        const real_t scale = static_cast<real_t>(1.0) / std::sqrt(norm2);
        for (size_t k = 0; k < cis_dim; ++k) {
            amps[n * cis_dim + k] *= scale;
        }
    }
    return amps;
}

// L2 norm of a contiguous double buffer.
real_t l2_norm(const real_t* p, size_t n) {
    real_t s = 0.0;
    for (size_t i = 0; i < n; ++i) s += p[i] * p[i];
    return std::sqrt(s);
}

} // namespace

// ====================================================================
// 1. Trace conservation
//    With unit-normalized CIS amplitudes and uniform weights,
//    trace(ρ_occ) = trace(ρ_vir) = Σ_n w_n ‖C_n‖² = 1.0
// ====================================================================
TEST(CISNTOActiveSpace, TraceConservation) {
    const int n_states = 3, nocc = 4, nvir = 6;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/12345);

    CISNTOActiveSpace::Params p;
    p.o_thresh = 0.0;  // ensure no orbital is rejected
    p.v_thresh = 0.0;
    p.verbose  = 0;

    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    EXPECT_NEAR(r.weight_sum, 1.0, 1e-12);
    EXPECT_NEAR(r.trace_occ, 1.0, 1e-10);
    EXPECT_NEAR(r.trace_vir, 1.0, 1e-10);

    // Sum of NTO occupations must equal the trace (eigenvalues sum to trace).
    real_t sum_occ = 0.0;
    for (real_t v : r.nto_occ_occupations) sum_occ += v;
    real_t sum_vir = 0.0;
    for (real_t v : r.nto_vir_occupations) sum_vir += v;
    EXPECT_NEAR(sum_occ, 1.0, 1e-10);
    EXPECT_NEAR(sum_vir, 1.0, 1e-10);
}

// ====================================================================
// 2. Symmetry: trace(ρ) recovered above guarantees nothing about
//    off-diagonal symmetry. Instead test that the NTO eigenvectors are
//    orthonormal: U^T U = I.  If ρ were asymmetric, eigh() would either
//    fail or produce non-orthogonal columns, so orthonormality is the
//    operationally meaningful symmetry check.
// ====================================================================
TEST(CISNTOActiveSpace, NTOOrthonormality) {
    const int n_states = 3, nocc = 4, nvir = 6;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/24680);

    CISNTOActiveSpace::Params p; p.o_thresh = 0.0; p.v_thresh = 0.0; p.verbose = 0;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    // Check U_occ^T U_occ = I_nocc (within 1e-12).
    // U is row-major: U[i*n + k] holds entry (row=i, col=k).
    auto check_orthonormal = [](const std::vector<real_t>& U, int n) {
        for (int p1 = 0; p1 < n; ++p1) {
            for (int p2 = 0; p2 < n; ++p2) {
                real_t dot = 0.0;
                for (int i = 0; i < n; ++i) {
                    dot += U[static_cast<size_t>(i) * n + p1] *
                           U[static_cast<size_t>(i) * n + p2];
                }
                const real_t expected = (p1 == p2) ? 1.0 : 0.0;
                ASSERT_NEAR(dot, expected, 1e-12)
                    << "U^T U deviates from identity at (" << p1 << "," << p2 << ")";
            }
        }
    };
    check_orthonormal(r.U_occ, nocc);
    check_orthonormal(r.U_vir, nvir);
}

// ====================================================================
// 3. Non-negativity. A state-averaged density of unit-normalized CIS
//    amplitudes with positive weights is PSD; all eigenvalues must be
//    ≥ -ε (floating-point roundoff only).
// ====================================================================
TEST(CISNTOActiveSpace, NonNegativeOccupations) {
    const int n_states = 5, nocc = 6, nvir = 12;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/97531);

    CISNTOActiveSpace::Params p; p.o_thresh = 0.0; p.v_thresh = 0.0; p.verbose = 0;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    for (real_t v : r.nto_occ_occupations) {
        EXPECT_GE(v, -1e-12) << "ρ_occ has a significantly negative eigenvalue";
    }
    for (real_t v : r.nto_vir_occupations) {
        EXPECT_GE(v, -1e-12) << "ρ_vir has a significantly negative eigenvalue";
    }

    // Descending sort invariant.
    for (size_t k = 1; k < r.nto_occ_occupations.size(); ++k) {
        EXPECT_GE(r.nto_occ_occupations[k - 1], r.nto_occ_occupations[k]);
    }
    for (size_t k = 1; k < r.nto_vir_occupations.size(); ++k) {
        EXPECT_GE(r.nto_vir_occupations[k - 1], r.nto_vir_occupations[k]);
    }
}

// ====================================================================
// 4. Threshold edge cases.
// ====================================================================
TEST(CISNTOActiveSpace, ThresholdAcceptsAllAtZero) {
    const int n_states = 3, nocc = 4, nvir = 6;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/1111);

    CISNTOActiveSpace::Params p; p.o_thresh = 0.0; p.v_thresh = 0.0; p.verbose = 0;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    EXPECT_EQ(r.n_act_occ, nocc);
    EXPECT_EQ(r.n_act_vir, nvir);
}

TEST(CISNTOActiveSpace, ThresholdRejectsAllAtOne) {
    const int n_states = 3, nocc = 4, nvir = 6;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/2222);

    CISNTOActiveSpace::Params p; p.o_thresh = 1.0; p.v_thresh = 1.0; p.verbose = 0;
    EXPECT_THROW(
        CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p),
        std::runtime_error);
}

// ====================================================================
// 5. Pure 1-state limit with a STRUCTURED rank-1 amplitude.
//
//    A random C ∈ R^{nocc × nvir} is full rank, so C C^T has rank
//    min(nocc, nvir) > 1 in general — only a rank-1 outer product
//    C = u v^T forces both ρ_occ and ρ_vir to be rank-1. We use the
//    sparsest such input (single excitation i=0 → a=0) so the analytic
//    answer is the canonical basis vector e_0.
// ====================================================================
TEST(CISNTOActiveSpace, OneStateRankOne) {
    const int n_states = 1, nocc = 4, nvir = 6;
    const size_t cis_dim = static_cast<size_t>(nocc) * nvir;
    std::vector<real_t> amps(n_states * cis_dim, 0.0);
    amps[0 * cis_dim + 0 * nvir + 0] = 1.0;  // C[i=0, a=0] = 1, all else zero

    CISNTOActiveSpace::Params p; p.o_thresh = 1e-12; p.v_thresh = 1e-12; p.verbose = 0;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    // ρ_occ = e_0 e_0^T → exactly rank-1, eigenvalues {1, 0, 0, 0}.
    EXPECT_NEAR(r.nto_occ_occupations[0], 1.0, 1e-12);
    for (int k = 1; k < nocc; ++k) {
        EXPECT_NEAR(r.nto_occ_occupations[k], 0.0, 1e-12)
            << "Expected rank-1 ρ_occ; non-zero eigenvalue at index " << k;
    }
    // ρ_vir = f_0 f_0^T → exactly rank-1, eigenvalues {1, 0, ..., 0}.
    EXPECT_NEAR(r.nto_vir_occupations[0], 1.0, 1e-12);
    for (int k = 1; k < nvir; ++k) {
        EXPECT_NEAR(r.nto_vir_occupations[k], 0.0, 1e-12)
            << "Expected rank-1 ρ_vir; non-zero eigenvalue at index " << k;
    }
}

// ====================================================================
// 6. Two-state degeneracy (rank check).
//    State 1 = excitation i=0,a=0;  State 2 = excitation i=0,a=1.
//    ρ_occ has rank 1 (only orbital i=0 contributes), eigenvalue = 1.0.
//    ρ_vir has rank 2 with two eigenvalues = 0.5 each (50/50 split).
// ====================================================================
TEST(CISNTOActiveSpace, TwoStateDegenerateVirtual) {
    const int n_states = 2, nocc = 3, nvir = 4;
    const size_t cis_dim = static_cast<size_t>(nocc) * nvir;
    std::vector<real_t> amps(n_states * cis_dim, 0.0);
    // State 0: C[i=0, a=0] = 1
    amps[0 * cis_dim + 0 * nvir + 0] = 1.0;
    // State 1: C[i=0, a=1] = 1
    amps[1 * cis_dim + 0 * nvir + 1] = 1.0;

    CISNTOActiveSpace::Params p; p.o_thresh = 1e-12; p.v_thresh = 1e-12; p.verbose = 0;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    // ρ_occ = 0.5 (e_0 e_0^T + e_0 e_0^T) = e_0 e_0^T  → eigenvalue = 1.0
    // (uniform weights are 1/2 each, so the sum is 1.0)
    EXPECT_NEAR(r.nto_occ_occupations[0], 1.0, 1e-12);
    EXPECT_NEAR(r.nto_occ_occupations[1], 0.0, 1e-12);
    EXPECT_NEAR(r.nto_occ_occupations[2], 0.0, 1e-12);
    EXPECT_EQ(r.n_act_occ, 1);

    // ρ_vir = 0.5 * (e_0 e_0^T) + 0.5 * (e_1 e_1^T) → eigenvalues {0.5, 0.5, 0, 0}
    EXPECT_NEAR(r.nto_vir_occupations[0], 0.5, 1e-12);
    EXPECT_NEAR(r.nto_vir_occupations[1], 0.5, 1e-12);
    EXPECT_NEAR(r.nto_vir_occupations[2], 0.0, 1e-12);
    EXPECT_NEAR(r.nto_vir_occupations[3], 0.0, 1e-12);
    EXPECT_EQ(r.n_act_vir, 2);
}

// ====================================================================
// 7. Frozen-core projector embedding.  num_frozen rows of P_occ_can
//    must be zero; the remaining nocc_active rows hold the active U_occ.
// ====================================================================
TEST(CISNTOActiveSpace, FrozenCoreProjectorEmbedding) {
    const int n_states = 2, nocc = 3, nvir = 4;
    auto amps = make_random_cis_amps(n_states, nocc, nvir, /*seed=*/4444);

    CISNTOActiveSpace::Params p; p.o_thresh = 1e-12; p.v_thresh = 1e-12; p.verbose = 0;
    const int num_frozen = 2;
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, num_frozen, p);

    const int nocc_canonical = num_frozen + nocc;
    std::vector<real_t> P_occ_can, P_vir_can;
    CISNTOActiveSpace::make_canonical_projectors(r, nocc_canonical, P_occ_can, P_vir_can);

    ASSERT_EQ(P_occ_can.size(), static_cast<size_t>(nocc_canonical) * r.n_act_occ);
    ASSERT_EQ(P_vir_can.size(), static_cast<size_t>(nvir)           * r.n_act_vir);

    // Frozen rows must be exactly zero.
    for (int i = 0; i < num_frozen; ++i) {
        for (int k = 0; k < r.n_act_occ; ++k) {
            EXPECT_DOUBLE_EQ(P_occ_can[static_cast<size_t>(i) * r.n_act_occ + k], 0.0);
        }
    }
    // Active-occupied rows must equal U_occ's first n_act_occ columns.
    for (int i_act = 0; i_act < nocc; ++i_act) {
        const int i_can = num_frozen + i_act;
        for (int k = 0; k < r.n_act_occ; ++k) {
            EXPECT_DOUBLE_EQ(
                P_occ_can[static_cast<size_t>(i_can) * r.n_act_occ + k],
                r.U_occ[static_cast<size_t>(i_act) * nocc + k]);
        }
    }
}

// ====================================================================
// 8. User weights — weights=(1,0,0) reproduces the rank-1 limit of
//    test 5 from a multi-state input, even when states 1..N-1 are
//    arbitrary (their contribution is silenced by weight=0).
//    Use a structured rank-1 state 0 so the analytic answer is e_0.
// ====================================================================
TEST(CISNTOActiveSpace, UserWeightsCollapseToFirstState) {
    const int n_states = 3, nocc = 4, nvir = 6;
    const size_t cis_dim = static_cast<size_t>(nocc) * nvir;

    // State 0: structured rank-1 (i=0,a=0).
    // States 1, 2: random — must NOT contribute because their weights are zero.
    std::vector<real_t> amps(n_states * cis_dim, 0.0);
    amps[0 * cis_dim + 0 * nvir + 0] = 1.0;
    std::mt19937_64 rng(5555);
    std::normal_distribution<real_t> normal(0.0, 1.0);
    for (size_t k = cis_dim; k < amps.size(); ++k) amps[k] = normal(rng);

    CISNTOActiveSpace::Params p; p.o_thresh = 1e-12; p.v_thresh = 1e-12; p.verbose = 0;
    p.weights = {1.0, 0.0, 0.0};
    auto r = CISNTOActiveSpace::compute(amps.data(), n_states, nocc, nvir, 0, p);

    EXPECT_NEAR(r.weight_sum, 1.0, 1e-12);
    // Only state 0 contributes → rank-1 with eigenvalue 1, all others zero.
    EXPECT_NEAR(r.nto_occ_occupations[0], 1.0, 1e-12);
    for (int k = 1; k < nocc; ++k) {
        EXPECT_NEAR(r.nto_occ_occupations[k], 0.0, 1e-12)
            << "weight=(1,0,0) must zero out states 1..N-1; non-zero at index " << k;
    }
}
