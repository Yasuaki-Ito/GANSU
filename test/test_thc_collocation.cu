/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_thc_collocation.cu
 * @brief Validate THC collocation by reconstructing the AO overlap matrix.
 *
 * The grid integral
 *   S_THC_{mu,nu} = sum_P w_P * phi_mu(r_P) * phi_nu(r_P)
 * must reproduce GANSU's analytic overlap matrix S_{mu,nu} (built by
 * computeCoreHamiltonianMatrix during HF::solve()) within an accuracy that
 * scales with the grid quality.  For default grids on a small molecule we
 * expect Frobenius error well below 1e-3 with sto-3g and below 1e-2 with
 * cc-pVDZ.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "thc_collocation.hpp"
#include "thc_grid.hpp"
#include "molecular.hpp"
#include "basis_set.hpp"
#include "builder.hpp"
#include "parameter_manager.hpp"
#include "device_host_memory.hpp"
#include "hf.hpp"

using namespace gansu;

namespace {

struct HFFixture {
    std::unique_ptr<HF> hf;
    std::vector<real_t> S_analytic; // [N_bas * N_bas] column-major
    int N_bas = 0;
};

HFFixture run_hf_and_grab_overlap(const std::string& xyz, const std::string& basis)
{
    ParameterManager params;
    params["xyzfilename"] = xyz;
    params["gbsfilename"] = basis;
    params["method"] = "RHF";
    params["convergence_energy_threshold"] = "1e-8";
    params["initial_guess"] = "core";

    // Suppress GANSU verbose output.
    std::streambuf* orig = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());

    HFFixture fx;
    fx.hf = HFBuilder::buildHF(params);
    fx.hf->solve();

    std::cout.rdbuf(orig);

    auto& dev_S = fx.hf->get_overlap_matrix();
    dev_S.toHost();
    fx.N_bas = fx.hf->get_num_basis();
    const std::size_t N = static_cast<std::size_t>(fx.N_bas) * fx.N_bas;
    fx.S_analytic.resize(N);
    const real_t* src = dev_S.host_ptr();
    for (std::size_t i = 0; i < N; ++i) fx.S_analytic[i] = src[i];
    return fx;
}

double frobenius_diff(const std::vector<real_t>& A,
                      const std::vector<real_t>& B)
{
    double s = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        const double d = static_cast<double>(A[i]) - static_cast<double>(B[i]);
        s += d * d;
    }
    return std::sqrt(s);
}

double frobenius_norm(const std::vector<real_t>& A)
{
    double s = 0.0;
    for (auto v : A) s += static_cast<double>(v) * static_cast<double>(v);
    return std::sqrt(s);
}

} // namespace

// =============================================================================
// 1. H2O / sto-3g
//
// Small basis (7 functions, only s/p), default grid (n_radial=50, L194) should
// integrate the AO overlap matrix to ~1e-4 relative Frobenius accuracy.
// =============================================================================

TEST(THCCollocation, OverlapMatrix_H2O_STO3G_DefaultGrid)
{
    const std::string xyz = "../xyz/H2O.xyz";
    const std::string basis = "../basis/sto-3g.gbs";
    HFFixture fx;
    std::unique_ptr<Molecular> mol;
    try {
        fx = run_hf_and_grab_overlap(xyz, basis);
        mol = std::make_unique<Molecular>(xyz, basis);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run HF for H2O/sto-3g: " << e.what();
        return;
    }

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L194;
    opts.n_radial = 50;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(*mol, opts);

    auto X = compute_X_ao_cpu(*mol, grid);
    auto S_thc = build_overlap_thc_cpu(X, grid, fx.N_bas);

    const double diff = frobenius_diff(S_thc, fx.S_analytic);
    const double norm = frobenius_norm(fx.S_analytic);
    const double rel = diff / norm;

    std::cout << "  H2O/sto-3g  N_bas=" << fx.N_bas
              << "  N_g=" << grid.num_points
              << "  ||S_THC - S||_F=" << std::scientific << std::setprecision(3) << diff
              << "  rel=" << rel << std::endl;

    EXPECT_LT(rel, 1.0e-3);
}

// =============================================================================
// 2. H2O / cc-pVDZ
//
// Includes d-functions (degree-4 polynomials), tighter grid needed.  L302 plus
// n_radial=80 should land around 1e-3 relative Frobenius.
// =============================================================================

TEST(THCCollocation, OverlapMatrix_H2O_ccpVDZ_FineGrid)
{
    const std::string xyz = "../xyz/H2O.xyz";
    const std::string basis = "../basis/cc-pvdz.gbs";
    HFFixture fx;
    std::unique_ptr<Molecular> mol;
    try {
        fx = run_hf_and_grab_overlap(xyz, basis);
        mol = std::make_unique<Molecular>(xyz, basis);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run HF for H2O/cc-pVDZ: " << e.what();
        return;
    }

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L302;
    opts.n_radial = 80;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(*mol, opts);

    auto X = compute_X_ao_cpu(*mol, grid);
    auto S_thc = build_overlap_thc_cpu(X, grid, fx.N_bas);

    const double diff = frobenius_diff(S_thc, fx.S_analytic);
    const double norm = frobenius_norm(fx.S_analytic);
    const double rel = diff / norm;

    std::cout << "  H2O/cc-pVDZ  N_bas=" << fx.N_bas
              << "  N_g=" << grid.num_points
              << "  ||S_THC - S||_F=" << std::scientific << std::setprecision(3) << diff
              << "  rel=" << rel << std::endl;

    EXPECT_LT(rel, 5.0e-3);
}

// =============================================================================
// 3. Symmetry: S_THC must be symmetric (sanity, no analytic comparison).
// =============================================================================

TEST(THCCollocation, OverlapTHCIsSymmetric_H2O_STO3G)
{
    const std::string xyz = "../xyz/H2O.xyz";
    const std::string basis = "../basis/sto-3g.gbs";
    std::unique_ptr<Molecular> mol;
    try {
        mol = std::make_unique<Molecular>(xyz, basis);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot construct Molecular: " << e.what();
        return;
    }
    const int N_bas = static_cast<int>(mol->get_num_basis());

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 30;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(*mol, opts);

    auto X = compute_X_ao_cpu(*mol, grid);
    auto S_thc = build_overlap_thc_cpu(X, grid, N_bas);

    double max_asym = 0.0;
    for (int i = 0; i < N_bas; ++i) {
        for (int j = 0; j < N_bas; ++j) {
            const double a = S_thc[i + j * N_bas];
            const double b = S_thc[j + i * N_bas];
            max_asym = std::max(max_asym, std::abs(a - b));
        }
    }
    EXPECT_LT(max_asym, 1.0e-12);
}

#ifndef GANSU_CPU_ONLY

#include <cuda_runtime.h>

// =============================================================================
// GPU vs CPU collocation: identical math, must match to machine precision.
// =============================================================================

namespace {

void run_gpu_vs_cpu_collocation(const std::string& xyz, const std::string& basis,
                                const std::string& tag, double tol)
{
    std::unique_ptr<Molecular> mol;
    try {
        mol = std::make_unique<Molecular>(xyz, basis);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot construct Molecular: " << e.what();
        return;
    }

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 5;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(*mol, opts);

    auto X_cpu     = compute_X_ao_cpu(*mol, grid);
    auto X_gpu_dev = compute_X_ao_gpu(*mol, grid);
    X_gpu_dev->toHost();

    const std::size_t N = X_cpu.size();
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double xc = static_cast<double>(X_cpu[i]);
        const double xg = static_cast<double>(X_gpu_dev->host_ptr()[i]);
        const double d = xc - xg;
        diff += d * d;
        norm += xc * xc;
    }
    const double rel = std::sqrt(diff) / std::sqrt(norm);
    std::cout << "  " << tag << " rel ||X_GPU - X_CPU|| = "
              << std::scientific << std::setprecision(3) << rel << std::endl;
    EXPECT_LT(rel, tol);
}

} // namespace

TEST(THCCollocation_GPU, MatchesCpu_H2O_STO3G)
{
    run_gpu_vs_cpu_collocation("../xyz/H2O.xyz", "../basis/sto-3g.gbs",
                                "H2O/sto-3g ", 1.0e-12);
}

TEST(THCCollocation_GPU, MatchesCpu_H2O_ccpVDZ)
{
    run_gpu_vs_cpu_collocation("../xyz/H2O.xyz", "../basis/cc-pvdz.gbs",
                                "H2O/cc-pVDZ", 1.0e-12);
}

#endif // GANSU_CPU_ONLY
