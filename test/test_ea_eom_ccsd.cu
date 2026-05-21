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
 * @file test_ea_eom_ccsd.cu
 * @brief P2 sub-phase 2.0+2.1 unit tests for EAEOMCCSDOperator.
 *
 * Coverage:
 *   1. dimension() returns the documented total_dim = nvir + nocc * nvir²
 *   2. apply() returns D * x exactly (diagonal-only matvec)
 *      - 1p sector D[a]     = +eps[a + nocc]   (positive! IP differed in sign)
 *      - 2p1h sector D[jab] = -eps[j] + eps[a + nocc] + eps[b + nocc]
 *   3. apply_preconditioner() returns x / D
 *   4. Constructor refuses (nao != nocc + nvir)
 */

#include <gtest/gtest.h>

#include <vector>
#include <cmath>

#include "cis_nto_active_space.hpp"
#include "device_host_memory.hpp"
#include "ea_eom_ccsd_operator.hpp"
#include "ea_eom_result.hpp"
#include "gpu_manager.hpp"

using namespace gansu;

namespace {

// Allocate orbital energies, T1, T2 stand-ins on device. Sub-phase 2.0+2.1
// operator only reads orbital energies in its diagonal; T1/T2 are owned for
// ownership-transfer correctness but not consumed.
struct DeviceHandles {
    real_t* d_eps = nullptr;
    real_t* d_t1  = nullptr;
    real_t* d_t2  = nullptr;
    int nocc;
    int nvir;
    int nao;
    std::vector<real_t> h_eps;

    DeviceHandles(int nocc_, int nvir_) : nocc(nocc_), nvir(nvir_), nao(nocc_ + nvir_) {
        h_eps.resize(nao);
        for (int i = 0; i < nocc; ++i) h_eps[i] = -0.5 - 0.1 * i;  // occupied: negative
        for (int a = 0; a < nvir; ++a) h_eps[nocc + a] = 0.1 + 0.05 * a;  // virtual: positive
        tracked_cudaMalloc(&d_eps, nao * sizeof(real_t));
        cudaMemcpy(d_eps, h_eps.data(), nao * sizeof(real_t), cudaMemcpyHostToDevice);
        tracked_cudaMalloc(&d_t1, nocc * nvir * sizeof(real_t));
        cudaMemset(d_t1, 0, nocc * nvir * sizeof(real_t));
        tracked_cudaMalloc(&d_t2, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
        cudaMemset(d_t2, 0, (size_t)nocc * nocc * nvir * nvir * sizeof(real_t));
    }
    // d_t1, d_t2 ownership is transferred to the operator — DON'T free here
};

}  // namespace

TEST(EAEOMCCSDOperator, Dimension) {
    DeviceHandles h(/*nocc=*/3, /*nvir=*/5);
    EAEOMCCSDOperator op(/*d_eri_mo=*/nullptr, h.d_eps, h.d_t1, h.d_t2,
                         h.nocc, h.nvir, h.nao);
    EXPECT_EQ(op.get_nocc(),    3);
    EXPECT_EQ(op.get_nvir(),    5);
    EXPECT_EQ(op.get_p_dim(),   5);                  // nvir
    EXPECT_EQ(op.get_p2h_dim(), 3 * 5 * 5);          // nocc · nvir²
    EXPECT_EQ(op.dimension(),   5 + 3 * 5 * 5);
    tracked_cudaFree(h.d_eps);
}

TEST(EAEOMCCSDOperator, DiagonalMatvecMatchesExpected) {
    const int nocc = 3, nvir = 5;
    DeviceHandles h(nocc, nvir);
    EAEOMCCSDOperator op(nullptr, h.d_eps, h.d_t1, h.d_t2, h.nocc, h.nvir, h.nao);

    const int total = op.dimension();
    std::vector<real_t> h_x(total), h_y_expected(total);
    for (int i = 0; i < total; ++i) h_x[i] = 1.3 + 0.1 * i;

    // 1p sector: D[a] = +eps[nocc+a]
    for (int a = 0; a < nvir; ++a) h_y_expected[a] = h.h_eps[nocc + a] * h_x[a];
    // 2p1h sector: D[jab] = -eps[j] + eps[nocc+a] + eps[nocc+b]
    for (int idx = 0; idx < op.get_p2h_dim(); ++idx) {
        int b  = idx % nvir;
        int t  = idx / nvir;
        int a  = t % nvir;
        int j  = t / nvir;
        real_t d = -h.h_eps[j] + h.h_eps[nocc + a] + h.h_eps[nocc + b];
        h_y_expected[nvir + idx] = d * h_x[nvir + idx];
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

TEST(EAEOMCCSDOperator, PreconditionerDividesByDiagonal) {
    const int nocc = 2, nvir = 4;
    DeviceHandles h(nocc, nvir);
    EAEOMCCSDOperator op(nullptr, h.d_eps, h.d_t1, h.d_t2, h.nocc, h.nvir, h.nao);

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

    auto expected_D = [&](int idx) -> real_t {
        if (idx < nvir) return h.h_eps[nocc + idx];
        int t  = idx - nvir;
        int b  = t % nvir;
        int t2 = t / nvir;
        int a  = t2 % nvir;
        int j  = t2 / nvir;
        return -h.h_eps[j] + h.h_eps[nocc + a] + h.h_eps[nocc + b];
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

// Build a CISNTOResult whose U_vir is the identity (nvir × nvir). With
// identity U_vir, the ã-th virtual NTO is canonical virtual orbital ã itself,
// so a root with R1 = unit vector e_a maps unambiguously to NTO ã.
CISNTOResult make_identity_cis_nto_vir(int nvir, int n_act_vir) {
    CISNTOResult r;
    r.nocc_active = 0;
    r.nvir        = nvir;
    r.num_frozen  = 0;
    r.n_act_occ   = 0;
    r.n_act_vir   = n_act_vir;
    r.U_vir.assign(static_cast<size_t>(nvir) * nvir, 0.0);
    for (int a = 0; a < nvir; ++a) {
        r.U_vir[static_cast<size_t>(a) * nvir + a] = 1.0;
    }
    return r;
}

EAEOMResult::PerRoot make_singles_root_ea(real_t omega, int nvir,
                                          int dominant_vir,
                                          real_t noise_amplitude = 0.0)
{
    EAEOMResult::PerRoot pr;
    pr.omega = omega;
    pr.R1.assign(nvir, 0.0);
    pr.R1[dominant_vir] = 1.0;
    for (int a = 0; a < nvir; ++a) {
        if (a != dominant_vir) pr.R1[a] = noise_amplitude * (1 + a * 0.1);
    }
    pr.R2.clear();
    pr.percent_singles = 1.0;
    pr.followcis_overlap   = 0.0;
    pr.canonical_vir_label = -1;
    return pr;
}

} // namespace

TEST(EAEOMRouting, AssignsBestOverlapPerActiveVirNTO) {
    const int nvir = 4;
    const int n_act_vir = 3;
    CISNTOResult cis_nto = make_identity_cis_nto_vir(nvir, n_act_vir);

    std::vector<EAEOMResult::PerRoot> roots;
    roots.push_back(make_singles_root_ea(/*omega=*/0.30, nvir, /*vir=*/0, 0.05));
    roots.push_back(make_singles_root_ea(/*omega=*/0.40, nvir, /*vir=*/1, 0.05));
    roots.push_back(make_singles_root_ea(/*omega=*/0.50, nvir, /*vir=*/2, 0.05));
    EAEOMResult::PerRoot mix;
    mix.omega = 0.70;
    mix.R1.assign(nvir, 0.5);
    mix.percent_singles = 1.0;
    mix.followcis_overlap = 0.0;
    mix.canonical_vir_label = -1;
    roots.push_back(mix);

    auto decision = select_active_ea_roots(cis_nto, roots, nvir, /*ea_thresh=*/0.80);

    EXPECT_EQ(decision.assigned_root_for_a[0], 0);
    EXPECT_EQ(decision.assigned_root_for_a[1], 1);
    EXPECT_EQ(decision.assigned_root_for_a[2], 2);
    EXPECT_GT(decision.overlap_for_a[0], 0.9);
    EXPECT_GT(decision.overlap_for_a[1], 0.9);
    EXPECT_GT(decision.overlap_for_a[2], 0.9);
    EXPECT_TRUE(decision.root_taken[0]);
    EXPECT_TRUE(decision.root_taken[1]);
    EXPECT_TRUE(decision.root_taken[2]);
    EXPECT_FALSE(decision.root_taken[3]);
}

TEST(EAEOMRouting, PercentSinglesFilterRejectsLowSinglesRoots) {
    const int nvir = 2;
    const int n_act_vir = 2;
    CISNTOResult cis_nto = make_identity_cis_nto_vir(nvir, n_act_vir);

    std::vector<EAEOMResult::PerRoot> roots;
    auto r0 = make_singles_root_ea(0.30, nvir, /*vir=*/0);
    r0.percent_singles = 0.50;
    roots.push_back(r0);
    auto r1 = make_singles_root_ea(0.40, nvir, /*vir=*/0, 0.30);
    r1.percent_singles = 0.95;
    roots.push_back(r1);

    auto decision = select_active_ea_roots(cis_nto, roots, nvir, /*ea_thresh=*/0.80);

    EXPECT_FALSE(decision.root_taken[0]);
    EXPECT_TRUE (decision.root_taken[1]);
    EXPECT_EQ(decision.assigned_root_for_a[0], 1);
    EXPECT_EQ(decision.assigned_root_for_a[1], -1);
}

TEST(EAEOMRouting, ZeroR1NormProducesNoOverlap) {
    const int nvir = 3;
    const int n_act_vir = 1;
    CISNTOResult cis_nto = make_identity_cis_nto_vir(nvir, n_act_vir);

    std::vector<EAEOMResult::PerRoot> roots;
    EAEOMResult::PerRoot pr;
    pr.omega = 0.5;
    pr.R1.assign(nvir, 0.0);
    pr.R2.assign(8, 1.0);
    pr.percent_singles = 0.0;
    roots.push_back(pr);

    auto decision = select_active_ea_roots(cis_nto, roots, nvir, /*ea_thresh=*/0.80);

    EXPECT_EQ(decision.assigned_root_for_a[0], -1);
    EXPECT_FALSE(decision.root_taken[0]);
}

TEST(EAEOMRouting, EmptyInputsReturnEmptyDecision) {
    const int nvir = 4;
    CISNTOResult cis_nto = make_identity_cis_nto_vir(nvir, /*n_act_vir=*/0);
    std::vector<EAEOMResult::PerRoot> roots;
    auto decision = select_active_ea_roots(cis_nto, roots, nvir, 0.80);
    EXPECT_TRUE(decision.assigned_root_for_a.empty());
    EXPECT_TRUE(decision.overlap_for_a.empty());
    EXPECT_TRUE(decision.root_taken.empty());
}

TEST(EAEOMCCSDOperator, RejectsInconsistentNao) {
    DeviceHandles h(/*nocc=*/3, /*nvir=*/4);
    EXPECT_THROW(
        EAEOMCCSDOperator(nullptr, h.d_eps, h.d_t1, h.d_t2,
                          /*nocc=*/3, /*nvir=*/4, /*nao=*/8),  // 3+4 != 8
        std::invalid_argument);
    tracked_cudaFree(h.d_t1);
    tracked_cudaFree(h.d_t2);
    tracked_cudaFree(h.d_eps);
}
