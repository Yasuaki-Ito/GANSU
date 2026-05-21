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
 * @file test_ip_eom_ccsd.cu
 * @brief P1 sub-phase 1.0+1.1 unit tests for IPEOMCCSDOperator.
 *
 * Coverage:
 *   1. dimension() returns the documented total_dim = nocc + nocc^2 * nvir
 *   2. apply() returns D * x exactly (diagonal-only matvec)
 *   3. apply_preconditioner() returns x / D
 *   4. Constructor refuses (nao != nocc + nvir)
 */

#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "cis_nto_active_space.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "ip_eom_ccsd_operator.hpp"
#include "ip_eom_result.hpp"

using namespace gansu;

namespace {

// Allocate a tiny set of host orbital energies, T1, T2 stand-ins on device.
// The sub-phase 1.0+1.1 operator only reads orbital energies in its diagonal;
// T1/T2 are owned for ownership-transfer correctness but not consumed.
struct DeviceHandles {
    real_t* d_eps = nullptr;
    real_t* d_t1  = nullptr;
    real_t* d_t2  = nullptr;
    int nocc;
    int nvir;
    int nao;
    std::vector<real_t> h_eps;  // CPU mirror for expected-value computation

    DeviceHandles(int nocc_, int nvir_) : nocc(nocc_), nvir(nvir_), nao(nocc_ + nvir_) {
        h_eps.resize(nao);
        // Occupied: negative orbital energies
        for (int i = 0; i < nocc; ++i) h_eps[i] = -0.5 - 0.1 * i;
        // Virtual: positive
        for (int a = 0; a < nvir; ++a) h_eps[nocc + a] = 0.1 + 0.05 * a;
        tracked_cudaMalloc(&d_eps, nao * sizeof(real_t));
        cudaMemcpy(d_eps, h_eps.data(), nao * sizeof(real_t), cudaMemcpyHostToDevice);
        tracked_cudaMalloc(&d_t1, nocc * nvir * sizeof(real_t));
        cudaMemset(d_t1, 0, nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_t2, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
        cudaMemset(d_t2, 0, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
    }
    // ownership of d_t1, d_t2 is transferred to the operator — DON'T free here
};

}  // namespace

TEST(IPEOMCCSDOperator, Dimension) {
    DeviceHandles h(/*nocc=*/3, /*nvir=*/5);
    IPEOMCCSDOperator op(/*d_eri_mo=*/nullptr, h.d_eps, h.d_t1, h.d_t2,
                         h.nocc, h.nvir, h.nao);
    EXPECT_EQ(op.get_nocc(),   3);
    EXPECT_EQ(op.get_nvir(),   5);
    EXPECT_EQ(op.get_h_dim(),  3);                  // nocc
    EXPECT_EQ(op.get_h2p_dim(), 3 * 3 * 5);         // nocc² · nvir
    EXPECT_EQ(op.dimension(),  3 + 3 * 3 * 5);
    tracked_cudaFree(h.d_eps);
}

TEST(IPEOMCCSDOperator, DiagonalMatvecMatchesExpected) {
    const int nocc = 3, nvir = 5;
    DeviceHandles h(nocc, nvir);
    IPEOMCCSDOperator op(nullptr, h.d_eps, h.d_t1, h.d_t2, h.nocc, h.nvir, h.nao);

    const int total = op.dimension();
    std::vector<real_t> h_x(total), h_y_expected(total);
    for (int i = 0; i < total; ++i) h_x[i] = 1.3 + 0.1 * i;

    // Expected D[i] for 1h sector: -eps_i
    for (int i = 0; i < nocc; ++i) h_y_expected[i] = -h.h_eps[i] * h_x[i];
    // Expected D for 2h1p sector: -eps_i - eps_j + eps_a
    for (int idx = 0; idx < op.get_h2p_dim(); ++idx) {
        int a  = idx % nvir;
        int t  = idx / nvir;
        int j  = t % nocc;
        int i  = t / nocc;
        real_t d = -h.h_eps[i] - h.h_eps[j] + h.h_eps[a + nocc];
        h_y_expected[nocc + idx] = d * h_x[nocc + idx];
    }

    real_t* d_x = nullptr; real_t* d_y = nullptr;
    tracked_cudaMalloc(&d_x, total * sizeof(real_t));
    tracked_cudaMalloc(&d_y, total * sizeof(real_t));
    cudaMemcpy(d_x, h_x.data(), total * sizeof(real_t), cudaMemcpyHostToDevice);
    op.apply(d_x, d_y);
    cudaDeviceSynchronize();
    std::vector<real_t> h_y(total);
    cudaMemcpy(h_y.data(), d_y, total * sizeof(real_t), cudaMemcpyDeviceToHost);

    for (int k = 0; k < total; ++k) {
        EXPECT_NEAR(h_y[k], h_y_expected[k], 1e-12)
            << "diagonal matvec mismatch at index " << k;
    }
    tracked_cudaFree(d_x);
    tracked_cudaFree(d_y);
    tracked_cudaFree(h.d_eps);
}

TEST(IPEOMCCSDOperator, PreconditionerDividesByDiagonal) {
    const int nocc = 2, nvir = 4;
    DeviceHandles h(nocc, nvir);
    IPEOMCCSDOperator op(nullptr, h.d_eps, h.d_t1, h.d_t2, h.nocc, h.nvir, h.nao);

    const int total = op.dimension();
    std::vector<real_t> h_x(total, 2.7);
    std::vector<real_t> h_y(total);
    real_t* d_x = nullptr; real_t* d_y = nullptr;
    tracked_cudaMalloc(&d_x, total * sizeof(real_t));
    tracked_cudaMalloc(&d_y, total * sizeof(real_t));
    cudaMemcpy(d_x, h_x.data(), total * sizeof(real_t), cudaMemcpyHostToDevice);
    op.apply_preconditioner(d_x, d_y);
    cudaDeviceSynchronize();
    cudaMemcpy(h_y.data(), d_y, total * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Reconstruct expected diagonal entries
    auto expected_D = [&](int idx) -> real_t {
        if (idx < nocc) return -h.h_eps[idx];
        int t  = idx - nocc;
        int a  = t % nvir;
        int t2 = t / nvir;
        int j  = t2 % nocc;
        int i  = t2 / nocc;
        return -h.h_eps[i] - h.h_eps[j] + h.h_eps[a + nocc];
    };
    for (int k = 0; k < total; ++k) {
        const real_t d = expected_D(k);
        const real_t expect = (std::fabs(d) > 1e-12) ? (h_x[k] / d) : real_t(0.0);
        EXPECT_NEAR(h_y[k], expect, 1e-12)
            << "preconditioner mismatch at index " << k;
    }
    tracked_cudaFree(d_x);
    tracked_cudaFree(d_y);
    tracked_cudaFree(h.d_eps);
}

namespace {

// Build a CISNTOResult whose U_occ is the identity (nocc_active × nocc_active).
// With identity U_occ, the m̃-th NTO is the canonical orbital m̃ itself, so a
// root with R1 = unit vector e_m maps unambiguously to NTO m̃.
CISNTOResult make_identity_cis_nto(int nocc_active, int n_act_occ) {
    CISNTOResult r;
    r.nocc_active = nocc_active;
    r.nvir        = 0;     // unused by the routing function
    r.num_frozen  = 0;
    r.n_act_occ   = n_act_occ;
    r.n_act_vir   = 0;
    r.U_occ.assign(static_cast<size_t>(nocc_active) * nocc_active, 0.0);
    for (int i = 0; i < nocc_active; ++i) {
        r.U_occ[static_cast<size_t>(i) * nocc_active + i] = 1.0;
    }
    return r;
}

// Build a PerRoot with R1 = e_i + tiny noise, R2 = zeros (→ %singles = 1).
IPEOMResult::PerRoot make_singles_root(real_t omega, int nocc_active,
                                        int dominant_orbital,
                                        real_t noise_amplitude = 0.0)
{
    IPEOMResult::PerRoot pr;
    pr.omega = omega;
    pr.R1.assign(nocc_active, 0.0);
    pr.R1[dominant_orbital] = 1.0;
    for (int i = 0; i < nocc_active; ++i) {
        if (i != dominant_orbital) pr.R1[i] = noise_amplitude * (1 + i * 0.1);
    }
    pr.R2.clear();              // empty → %singles = 1 by construction below
    pr.percent_singles = 1.0;
    pr.followcis_overlap   = 0.0;
    pr.canonical_occ_label = -1;
    return pr;
}

} // namespace

TEST(IPEOMRouting, AssignsBestOverlapPerActiveNTO) {
    // 3 active NTOs, 4 Davidson roots. Roots 0,1,2 each align with one NTO;
    // root 3 has overlap split across all NTOs (worse than roots 0..2 for any one).
    const int nocc_active = 4;
    const int n_act_occ   = 3;
    CISNTOResult cis_nto = make_identity_cis_nto(nocc_active, n_act_occ);

    std::vector<IPEOMResult::PerRoot> roots;
    roots.push_back(make_singles_root(/*omega=*/0.30, nocc_active, /*orbital=*/0, 0.05));
    roots.push_back(make_singles_root(/*omega=*/0.40, nocc_active, /*orbital=*/1, 0.05));
    roots.push_back(make_singles_root(/*omega=*/0.50, nocc_active, /*orbital=*/2, 0.05));
    // Root 3: equal mix across all 4 orbitals — large IP, low alignment.
    IPEOMResult::PerRoot mix;
    mix.omega = 0.70;
    mix.R1.assign(nocc_active, 0.5);
    mix.percent_singles = 1.0;
    mix.followcis_overlap = 0.0;
    mix.canonical_occ_label = -1;
    roots.push_back(mix);

    auto decision = select_active_ip_roots(cis_nto, roots, nocc_active, /*ip_thresh=*/0.80);

    EXPECT_EQ(decision.assigned_root_for_m[0], 0);
    EXPECT_EQ(decision.assigned_root_for_m[1], 1);
    EXPECT_EQ(decision.assigned_root_for_m[2], 2);
    EXPECT_GT(decision.overlap_for_m[0], 0.9);  // dominant orbital alignment
    EXPECT_GT(decision.overlap_for_m[1], 0.9);
    EXPECT_GT(decision.overlap_for_m[2], 0.9);
    EXPECT_TRUE(decision.root_taken[0]);
    EXPECT_TRUE(decision.root_taken[1]);
    EXPECT_TRUE(decision.root_taken[2]);
    EXPECT_FALSE(decision.root_taken[3]);  // mix root is left for auxiliary
}

TEST(IPEOMRouting, PercentSinglesFilterRejectsLowSinglesRoots) {
    // Two NTOs. Two roots: root 0 has matching R1 but %singles=0.5 (below
    // threshold); root 1 has weakly matching R1 but %singles=0.95. With
    // ip_thresh=0.80, root 0 is rejected — root 1 gets NTO 0.
    const int nocc_active = 2;
    const int n_act_occ   = 2;
    CISNTOResult cis_nto = make_identity_cis_nto(nocc_active, n_act_occ);

    std::vector<IPEOMResult::PerRoot> roots;
    auto r0 = make_singles_root(0.30, nocc_active, /*orbital=*/0);
    r0.percent_singles = 0.50;  // too low — must be filtered
    roots.push_back(r0);
    auto r1 = make_singles_root(0.40, nocc_active, /*orbital=*/0, 0.30);
    r1.percent_singles = 0.95;
    roots.push_back(r1);

    auto decision = select_active_ip_roots(cis_nto, roots, nocc_active, /*ip_thresh=*/0.80);

    EXPECT_FALSE(decision.root_taken[0]);          // %singles filter
    EXPECT_TRUE (decision.root_taken[1]);
    EXPECT_EQ(decision.assigned_root_for_m[0], 1);
    EXPECT_EQ(decision.assigned_root_for_m[1], -1); // no second qualifying root
}

TEST(IPEOMRouting, ZeroR1NormProducesNoOverlap) {
    // Pathological root whose R1 is entirely zero (would be ‖R1‖²=0). Must not
    // divide by zero; root must be left untouched.
    const int nocc_active = 3;
    const int n_act_occ   = 1;
    CISNTOResult cis_nto = make_identity_cis_nto(nocc_active, n_act_occ);

    std::vector<IPEOMResult::PerRoot> roots;
    IPEOMResult::PerRoot pr;
    pr.omega = 0.5;
    pr.R1.assign(nocc_active, 0.0);   // all zero
    pr.R2.assign(8, 1.0);             // doubles-dominated
    pr.percent_singles = 0.0;
    roots.push_back(pr);

    auto decision = select_active_ip_roots(cis_nto, roots, nocc_active, /*ip_thresh=*/0.80);

    EXPECT_EQ(decision.assigned_root_for_m[0], -1);
    EXPECT_FALSE(decision.root_taken[0]);
}

TEST(IPEOMRouting, EmptyInputsReturnEmptyDecision) {
    const int nocc_active = 4;
    CISNTOResult cis_nto = make_identity_cis_nto(nocc_active, /*n_act_occ=*/0);
    std::vector<IPEOMResult::PerRoot> roots;
    auto decision = select_active_ip_roots(cis_nto, roots, nocc_active, 0.80);
    EXPECT_TRUE(decision.assigned_root_for_m.empty());
    EXPECT_TRUE(decision.overlap_for_m.empty());
    EXPECT_TRUE(decision.root_taken.empty());
}

TEST(IPEOMCCSDOperator, RejectsInconsistentNao) {
    DeviceHandles h(/*nocc=*/3, /*nvir=*/4);
    EXPECT_THROW(
        IPEOMCCSDOperator(nullptr, h.d_eps, h.d_t1, h.d_t2,
                          /*nocc=*/3, /*nvir=*/4, /*nao=*/8),  // wrong (3+4=7, not 8)
        std::invalid_argument);
    // Note: when constructor throws, d_t1 and d_t2 are NOT transferred to the
    // operator and remain leaked. Clean up explicitly to keep the test
    // process tidy under repeated test runs.
    tracked_cudaFree(h.d_t1);
    tracked_cudaFree(h.d_t2);
    tracked_cudaFree(h.d_eps);
}
