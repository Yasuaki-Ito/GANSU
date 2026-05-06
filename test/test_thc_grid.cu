/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_thc_grid.cu
 * @brief Self-tests for the THC molecular-integration grid.
 *
 * These tests must pass before any THC collocation / LS-THC / THC-MP2 code
 * is trusted: an incorrect Lebedev coefficient or Becke partition will silently
 * corrupt every downstream tensor.  Each test is a small (~milliseconds) sanity
 * integral with an analytic answer.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "thc_grid.hpp"
#include "types.hpp"
#include "molecular.hpp"

using namespace gansu;

// =============================================================================
// 1. Lebedev sum-of-weights == 4*pi
//
// Every closed Lebedev formula has weights summing to the surface area of the
// unit sphere, 4*pi.  This catches transcription errors in the orbit weights.
// =============================================================================

namespace {

double lebedev_total_weight(LebedevOrder order)
{
    const auto& pts = get_lebedev_grid(order);
    double s = 0.0;
    for (const auto& p : pts) s += static_cast<double>(p.w);
    return s;
}

} // namespace

TEST(THCGrid_Lebedev, SumOfWeights_L110)
{
    EXPECT_NEAR(lebedev_total_weight(LebedevOrder::L110), 4.0 * M_PI, 1.0e-10);
    EXPECT_EQ(get_lebedev_grid(LebedevOrder::L110).size(), 110u);
}

TEST(THCGrid_Lebedev, SumOfWeights_L194)
{
    EXPECT_NEAR(lebedev_total_weight(LebedevOrder::L194), 4.0 * M_PI, 1.0e-10);
    EXPECT_EQ(get_lebedev_grid(LebedevOrder::L194).size(), 194u);
}

TEST(THCGrid_Lebedev, SumOfWeights_L302)
{
    EXPECT_NEAR(lebedev_total_weight(LebedevOrder::L302), 4.0 * M_PI, 1.0e-10);
    EXPECT_EQ(get_lebedev_grid(LebedevOrder::L302).size(), 302u);
}

// =============================================================================
// 2. Polynomial-exactness sanity integrals
//
// Pure-power monomials over the unit sphere have closed-form values:
//   I(p,q,r) = int_S^2 x^p y^q z^r dOmega
//            = 0   if any of p, q, r is odd,
//            = 4*pi * (p-1)!! (q-1)!! (r-1)!! / (p+q+r+1)!!   otherwise.
//
// For each Lebedev order, the formula is exact for total degree <= D where:
//   L110 -> D = 17
//   L194 -> D = 23
//   L302 -> D = 29
//
// We test at low (D=2, D=4) and at one degree below the order's exactness limit.
// =============================================================================

namespace {

double sphere_monomial_integral(int p, int q, int r)
{
    auto odd = [](int n) { return n % 2 != 0; };
    if (odd(p) || odd(q) || odd(r)) return 0.0;
    auto dfact = [](int n) {
        // (-1)!! == 1 for our purposes (n=0 case).
        if (n <= 0) return 1.0;
        double v = 1.0;
        for (int k = n; k > 0; k -= 2) v *= static_cast<double>(k);
        return v;
    };
    const int s = p + q + r;
    return 4.0 * M_PI * dfact(p - 1) * dfact(q - 1) * dfact(r - 1) / dfact(s + 1);
}

double lebedev_monomial(LebedevOrder order, int p, int q, int r)
{
    const auto& pts = get_lebedev_grid(order);
    double sum = 0.0;
    for (const auto& g : pts) {
        const double v = std::pow(g.x, p) * std::pow(g.y, q) * std::pow(g.z, r);
        sum += static_cast<double>(g.w) * v;
    }
    return sum;
}

} // namespace

TEST(THCGrid_Lebedev, MonomialDegree2)
{
    for (auto ord : {LebedevOrder::L110, LebedevOrder::L194, LebedevOrder::L302}) {
        EXPECT_NEAR(lebedev_monomial(ord, 2, 0, 0), sphere_monomial_integral(2, 0, 0), 1.0e-10);
        EXPECT_NEAR(lebedev_monomial(ord, 0, 2, 0), sphere_monomial_integral(0, 2, 0), 1.0e-10);
        EXPECT_NEAR(lebedev_monomial(ord, 0, 0, 2), sphere_monomial_integral(0, 0, 2), 1.0e-10);
    }
}

TEST(THCGrid_Lebedev, MonomialDegree4)
{
    for (auto ord : {LebedevOrder::L110, LebedevOrder::L194, LebedevOrder::L302}) {
        EXPECT_NEAR(lebedev_monomial(ord, 4, 0, 0), sphere_monomial_integral(4, 0, 0), 1.0e-10);
        EXPECT_NEAR(lebedev_monomial(ord, 2, 2, 0), sphere_monomial_integral(2, 2, 0), 1.0e-10);
    }
}

// L110 is exact through degree 17; degree 16 test discriminates from broken tables.
TEST(THCGrid_Lebedev, MonomialDegree16_L110)
{
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L110, 16, 0, 0),
                sphere_monomial_integral(16, 0, 0), 1.0e-9);
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L110, 8, 8, 0),
                sphere_monomial_integral(8, 8, 0), 1.0e-9);
}

TEST(THCGrid_Lebedev, MonomialDegree22_L194)
{
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L194, 22, 0, 0),
                sphere_monomial_integral(22, 0, 0), 1.0e-9);
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L194, 10, 12, 0),
                sphere_monomial_integral(10, 12, 0), 1.0e-9);
}

TEST(THCGrid_Lebedev, MonomialDegree28_L302)
{
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L302, 28, 0, 0),
                sphere_monomial_integral(28, 0, 0), 1.0e-9);
    EXPECT_NEAR(lebedev_monomial(LebedevOrder::L302, 14, 14, 0),
                sphere_monomial_integral(14, 14, 0), 1.0e-9);
}

// =============================================================================
// 3. Treutler-Ahlrichs M3 radial: gaussian integral on a hydrogen
//
//   int_0^infty exp(-a r^2) r^2 dr = sqrt(pi) / (4 a^{3/2})
//
// for a = 1 this is sqrt(pi)/4 ~ 0.4431134627263.
// =============================================================================

// w_rad already absorbs the r^2 Jacobian, so
//   sum_k w_rad(k) g(r_k)  ~  int_0^infty g(r) r^2 dr.
// For g(r)=exp(-r^2): expected = sqrt(pi)/4 ~ 0.4431...
TEST(THCGrid_Radial, GaussianHydrogen_n50)
{
    std::vector<real_t> r, w;
    treutler_ahlrichs_m3(/*Z=*/1, /*n_radial=*/50, r, w);

    double integral = 0.0;
    for (std::size_t k = 0; k < r.size(); ++k) {
        const double rk = static_cast<double>(r[k]);
        integral += static_cast<double>(w[k]) * std::exp(-rk * rk);
    }
    const double expected = std::sqrt(M_PI) / 4.0;
    EXPECT_NEAR(integral, expected, 1.0e-6);
}

// =============================================================================
// 4. Single-atom radial+angular integrand of a normalised Gaussian density
//
// int (a/pi)^(3/2) exp(-a r^2) d^3r = 1, exact.
// This isolates radial + Lebedev (no Becke partition involved).
// =============================================================================

namespace {

std::vector<Atom> water_atoms()
{
    std::vector<Atom> atoms;
    atoms.push_back(Atom{ 8, 8, Coordinate{ 0.0,  0.0,  0.0}, 0});
    atoms.push_back(Atom{ 1, 1, Coordinate{ 1.43, 1.10, 0.0}, 1});
    atoms.push_back(Atom{ 1, 1, Coordinate{-1.43, 1.10, 0.0}, 2});
    return atoms;
}

} // namespace

TEST(THCGrid_AtomicQuadrature, NormalisedGaussianHydrogen)
{
    std::vector<real_t> r, w;
    treutler_ahlrichs_m3(/*Z=*/1, /*n_radial=*/50, r, w);

    const auto& angular = get_lebedev_grid(LebedevOrder::L194);

    const double a = 1.0;
    const double norm = std::pow(a / M_PI, 1.5);

    double total = 0.0;
    for (std::size_t kr = 0; kr < r.size(); ++kr) {
        const double rk = static_cast<double>(r[kr]);
        const double rho = norm * std::exp(-a * rk * rk);
        for (const auto& ang : angular) {
            total += static_cast<double>(w[kr]) * static_cast<double>(ang.w) * rho;
        }
    }
    EXPECT_NEAR(total, 1.0, 1.0e-6);
}

// =============================================================================
// 5. End-to-end: build_molecular_grid integrating a sum of atomic Gaussians
//
// For an isolated atom at R_A: int (a/pi)^(3/2) exp(-a|r-R_A|^2) d^3r = 1.
// Place such a normalised Gaussian on every atom of H2O and integrate via the
// constructed molecular grid.  Becke partition + radial + angular together
// must integrate to N_atoms = 3.  Tolerance ~1e-3 is easy with default
// (n_radial=50, L194).
// =============================================================================

TEST(THCGrid_Molecular, GaussianSumIntegral_H2O)
{
    // Build a Molecular instance.  The basis content does not affect the grid
    // itself, but the constructor needs something; use the minimal sto-3g.
    //
    // To avoid coupling to filesystem state in tests, we build a Molecular
    // with an in-memory atoms list and any pre-constructed BasisSet.
    //
    // The basis set with no element basis defined will fail in
    // create_basis_set_from_object for missing element.  Instead, use the
    // file-loading constructor with sto-3g if available; otherwise skip.
    BasisSet bs;
    try {
        bs = BasisSet::construct_from_gbs("../basis/sto-3g.gbs");
    } catch (...) {
        GTEST_SKIP() << "sto-3g.gbs not available from CWD; skip end-to-end test";
        return;
    }

    std::vector<Atom> atoms = water_atoms();
    Molecular mol(atoms, bs, /*charge=*/0);

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L194;
    opts.n_radial = 50;
    opts.weight_eps = 0.0; // no pruning for this validation

    MolecularGrid grid = build_molecular_grid(mol, opts);

    EXPECT_GT(grid.num_points, 0u);

    const double a = 1.5;
    const double norm = std::pow(a / M_PI, 1.5);

    double integral = 0.0;
    for (std::size_t i = 0; i < grid.num_points; ++i) {
        const auto& p = grid.points[i];
        double rho = 0.0;
        for (const auto& at : atoms) {
            const double dx = p.x - at.coordinate.x;
            const double dy = p.y - at.coordinate.y;
            const double dz = p.z - at.coordinate.z;
            rho += norm * std::exp(-a * (dx*dx + dy*dy + dz*dz));
        }
        integral += static_cast<double>(p.w) * rho;
    }

    EXPECT_NEAR(integral, static_cast<double>(atoms.size()), 1.0e-3);
}

// =============================================================================
// 6. Device-mirror sanity: host_to_device, then read back via cudaMemcpy
// =============================================================================

#ifndef GANSU_CPU_ONLY

#include <cuda_runtime.h>

TEST(THCGrid_Device, HostToDeviceRoundtrip_L110_Hydrogen)
{
    BasisSet bs;
    try {
        bs = BasisSet::construct_from_gbs("../basis/sto-3g.gbs");
    } catch (...) {
        GTEST_SKIP() << "sto-3g.gbs not available";
        return;
    }
    std::vector<Atom> atoms;
    atoms.push_back(Atom{1, 1, Coordinate{0.0, 0.0, 0.0}, 0});
    Molecular mol(atoms, bs);

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 20;
    opts.weight_eps = 0.0;

    MolecularGrid grid = build_molecular_grid(mol, opts);
    grid.host_to_device();

    ASSERT_NE(grid.d_x, nullptr);
    ASSERT_NE(grid.d_w, nullptr);

    std::vector<real_t> back_w(grid.num_points);
    cudaMemcpy(back_w.data(), grid.d_w,
               grid.num_points * sizeof(real_t),
               cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < grid.num_points; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(back_w[i]),
                         static_cast<double>(grid.points[i].w));
    }
}

#endif
