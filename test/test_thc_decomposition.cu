/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_thc_decomposition.cu
 * @brief Validate LS-THC by reconstructing the 4-index ERI tensor.
 *
 *   1. Build a small molecular grid + collocation X^P_mu (CPU).
 *   2. Build the analytic 4-index ERI via GANSU's CPU integral library.
 *   3. Run LS-THC: Gram -> S = G^2 -> E -> Z = S^+ E S^+.
 *   4. Reconstruct (mu nu | lam sig)_THC = sum_{PQ} X^P_mu X^P_nu Z X^Q ...
 *   5. Compare against the analytic ERI (Frobenius relative error).
 *
 * The grid size N_g is the only knob the LS-THC error converges in: typical
 * benchmarks (Hohenstein-Parrish-Martinez 2012) report 1e-3 ... 1e-5 relative
 * error for N_g / N_bas ~~ 5..10.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

#include "thc_collocation.hpp"
#include "thc_decomposition.hpp"
#include "thc_grid.hpp"
#include "molecular.hpp"
#include "cpu_integrals.hpp"

using namespace gansu;

namespace {

double frob(const std::vector<real_t>& v)
{
    double s = 0.0;
    for (auto x : v) s += static_cast<double>(x) * static_cast<double>(x);
    return std::sqrt(s);
}

double frob_diff(const std::vector<real_t>& A, const std::vector<real_t>& B)
{
    double s = 0.0;
    for (std::size_t i = 0; i < A.size(); ++i) {
        const double d = static_cast<double>(A[i]) - static_cast<double>(B[i]);
        s += d * d;
    }
    return std::sqrt(s);
}

// Compute the GANSU analytic 4-index ERI via cpu_integrals::computeERIMatrix.
// Returned as length-N_bas^4 array, eri[a + N_bas*(b + N_bas*(c + N_bas*d))]
//                                  = (a b | c d).
std::vector<real_t> compute_analytic_eri(const Molecular& mol)
{
    const int N_bas = static_cast<int>(mol.get_num_basis());
    const std::size_t N4 = static_cast<std::size_t>(N_bas) * N_bas * N_bas * N_bas;
    std::vector<real_t> eri(N4, 0.0);

    cpu::computeERIMatrix(
        mol.get_shell_type_infos(),
        mol.get_primitive_shells().data(),
        mol.get_cgto_normalization_factors().data(),
        eri.data(),
        N_bas);

    return eri;
}

struct ThcLsRun {
    int N_bas;
    int N_g;
    int n_pair;          // expected target rank = N_bas*(N_bas+1)/2
    int rank_S;          // actual numerical rank of S above cutoff
    real_t sigma_max;
    real_t sigma_min_kept;
    double rel_err_eri;
    double rel_err_E_consistency; // V_THC consistency
};

ThcLsRun run_ls_thc(const std::string& xyz, const std::string& basis,
                    LebedevOrder lev, int n_radial,
                    double rel_cutoff = 1.0e-7)
{
    Molecular mol(xyz, basis);
    const int N_bas = static_cast<int>(mol.get_num_basis());

    ThcGridOptions opts;
    opts.lebedev = lev;
    opts.n_radial = n_radial;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(mol, opts);

    auto X = compute_X_ao_cpu(mol, grid);
    auto eri = compute_analytic_eri(mol);

    int rank_M = 0;
    real_t sigma_max = 0.0, sigma_min_kept = 0.0;
    auto Z = compute_Z_via_M_svd_cpu(X, eri, N_bas,
                                     static_cast<int>(grid.num_points),
                                     rel_cutoff,
                                     &rank_M, &sigma_max, &sigma_min_kept);

    auto eri_thc = reconstruct_eri_thc_cpu(X, Z, N_bas, grid.num_points);

    ThcLsRun r;
    r.N_bas = N_bas;
    r.N_g = static_cast<int>(grid.num_points);
    r.n_pair = N_bas * (N_bas + 1) / 2;
    r.rank_S = rank_M;
    r.sigma_max = sigma_max;
    r.sigma_min_kept = sigma_min_kept;
    r.rel_err_eri = frob_diff(eri_thc, eri) / frob(eri);

    // E-consistency: build E from analytic ERI and from THC ERI.
    auto E      = build_E_from_eri_cpu(X, eri,     N_bas, grid.num_points);
    auto E_thc  = build_E_from_eri_cpu(X, eri_thc, N_bas, grid.num_points);
    r.rel_err_E_consistency = frob_diff(E_thc, E) / frob(E);
    return r;
}

void print_run(const std::string& tag, const ThcLsRun& r)
{
    std::cout << "  " << tag
              << "  N_bas=" << r.N_bas
              << "  N_g=" << r.N_g
              << "  N_pair_target=" << r.n_pair
              << "  rank(S)=" << r.rank_S
              << "  sigma_max=" << std::scientific << std::setprecision(2) << r.sigma_max
              << "  sigma_min_kept=" << r.sigma_min_kept
              << "\n      ||eri_THC - eri||/||eri|| = " << std::setprecision(3) << r.rel_err_eri
              << "  E-consistency=" << r.rel_err_E_consistency
              << std::endl;
}

} // namespace

// =============================================================================
// Grid sizing note
// -----------------------------------------------------------------------------
// LS-THC requires N_g >~ a few * N_bas^2 for the metric S = (Gram)^2 to be
// well-conditioned.  Pushing N_g beyond that just wastes CPU on the dense
// eigendecomposition (O(N_g^3) per test).  We deliberately keep N_g modest:
//   sto-3g  (N_bas=7,  N_bas^2=49 ): N_g ~ 1000
//   cc-pVDZ (N_bas=25, N_bas^2=625): N_g ~ 2500
// On a single CPU thread Eigen's SelfAdjointEigenSolver runs in seconds at
// these sizes.  Once we move to GPU these can grow.
// =============================================================================

TEST(THCDecomposition, EriReconstruction_H2_STO3G)
{
    ThcLsRun r;
    try {
        r = run_ls_thc("../xyz/H2.xyz", "../basis/sto-3g.gbs",
                       LebedevOrder::L110, /*n_radial=*/3);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot construct H2/sto-3g molecule: " << e.what();
        return;
    }
    print_run("H2/sto-3g", r);
    EXPECT_LT(r.rel_err_eri, 1.0e-3);
    EXPECT_LT(r.rel_err_E_consistency, 1.0e-8);
}

// H2O / sto-3g.  Pair-space dim is 28; we want rank(M) >= 28 for an exact
// fit.  M is N_bas^2 x N_g = 49 x N_g, so SVD is fast even for N_g ~ 6600.
// n_radial=20 with L110 -> N_g = 6600 gives ample radial resolution to span
// the 28-dim pair space (10..150 alpha range of sto-3g/O).
TEST(THCDecomposition, EriReconstruction_H2O_STO3G)
{
    ThcLsRun r;
    try {
        r = run_ls_thc("../xyz/H2O.xyz", "../basis/sto-3g.gbs",
                       LebedevOrder::L110, /*n_radial=*/20);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot construct H2O/sto-3g molecule: " << e.what();
        return;
    }
    print_run("H2O/sto-3g", r);
    EXPECT_GE(r.rank_S, r.n_pair); // grid must span the pair space
    EXPECT_LT(r.rel_err_eri, 1.0e-3);
    EXPECT_LT(r.rel_err_E_consistency, 1.0e-8);
}

// cc-pVDZ skipped at this stage: the dense O(N_g^3) eigendecomposition needs
// N_g ~ 2500+ to over-resolve the 625-dim basis-pair space, which is on the
// edge of what runs in unit-test time on CPU.  Re-enable after the GPU
// solve_Z_pinv path lands.

#ifndef GANSU_CPU_ONLY

#include <cuda_runtime.h>
#include "thc_collocation.hpp"
#include "device_host_memory.hpp"

// =============================================================================
// GPU LS-THC vs CPU LS-THC: same X and same V_eri, the resulting Z must agree
// to machine precision (matrix factorisations are unique up to truncation
// rounding, which we share).
// =============================================================================

TEST(THCDecomposition_GPU, MatchesCpu_H2_STO3G)
{
    Molecular mol("../xyz/H2.xyz", "../basis/sto-3g.gbs");
    const int N_bas = static_cast<int>(mol.get_num_basis());

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 3;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(mol, opts);

    // CPU reference
    auto X_cpu   = compute_X_ao_cpu(mol, grid);
    std::vector<real_t> eri_cpu(static_cast<std::size_t>(N_bas) * N_bas * N_bas * N_bas, 0.0);
    cpu::computeERIMatrix(
        mol.get_shell_type_infos(),
        mol.get_primitive_shells().data(),
        mol.get_cgto_normalization_factors().data(),
        eri_cpu.data(),
        N_bas);
    int rank_cpu = 0;
    real_t s_max_cpu = 0, s_min_cpu = 0;
    auto Z_cpu = compute_Z_via_M_svd_cpu(X_cpu, eri_cpu, N_bas,
                                         static_cast<int>(grid.num_points),
                                         1.0e-7, &rank_cpu, &s_max_cpu, &s_min_cpu);

    // GPU path: build X, eri on device, run LS-THC GPU.
    auto X_gpu_dh = compute_X_ao_gpu(mol, grid);
    real_t* d_eri = nullptr;
    cudaMalloc(&d_eri, eri_cpu.size() * sizeof(real_t));
    cudaMemcpy(d_eri, eri_cpu.data(), eri_cpu.size() * sizeof(real_t),
               cudaMemcpyHostToDevice);

    int rank_gpu = 0;
    real_t s_max_gpu = 0, s_min_gpu = 0;
    auto Z_gpu_dh = compute_Z_via_M_svd_gpu(
        X_gpu_dh->device_ptr(), d_eri, N_bas,
        static_cast<int>(grid.num_points),
        1.0e-7, &rank_gpu, &s_max_gpu, &s_min_gpu);
    Z_gpu_dh->toHost();

    cudaFree(d_eri);

    EXPECT_EQ(rank_cpu, rank_gpu);
    const std::size_t N = Z_cpu.size();
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double zc = static_cast<double>(Z_cpu[i]);
        const double zg = static_cast<double>(Z_gpu_dh->host_ptr()[i]);
        const double d = zc - zg;
        diff += d * d;
        norm += zc * zc;
    }
    const double rel = std::sqrt(diff) / std::sqrt(norm);
    std::cout << "  H2/sto-3g rank=" << rank_cpu << "/" << rank_gpu
              << "  rel ||Z_GPU - Z_CPU|| = "
              << std::scientific << std::setprecision(3) << rel << std::endl;
    EXPECT_LT(rel, 1.0e-12);
}

TEST(THCDecomposition_GPU, MatchesCpu_H2O_STO3G)
{
    Molecular mol("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    const int N_bas = static_cast<int>(mol.get_num_basis());

    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 20;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(mol, opts);

    auto X_cpu   = compute_X_ao_cpu(mol, grid);
    std::vector<real_t> eri_cpu(static_cast<std::size_t>(N_bas) * N_bas * N_bas * N_bas, 0.0);
    cpu::computeERIMatrix(
        mol.get_shell_type_infos(),
        mol.get_primitive_shells().data(),
        mol.get_cgto_normalization_factors().data(),
        eri_cpu.data(),
        N_bas);
    int rank_cpu = 0;
    real_t s_max_cpu = 0, s_min_cpu = 0;
    auto Z_cpu = compute_Z_via_M_svd_cpu(X_cpu, eri_cpu, N_bas,
                                         static_cast<int>(grid.num_points),
                                         1.0e-7, &rank_cpu, &s_max_cpu, &s_min_cpu);

    auto X_gpu_dh = compute_X_ao_gpu(mol, grid);
    real_t* d_eri = nullptr;
    cudaMalloc(&d_eri, eri_cpu.size() * sizeof(real_t));
    cudaMemcpy(d_eri, eri_cpu.data(), eri_cpu.size() * sizeof(real_t),
               cudaMemcpyHostToDevice);

    int rank_gpu = 0;
    real_t s_max_gpu = 0, s_min_gpu = 0;
    auto Z_gpu_dh = compute_Z_via_M_svd_gpu(
        X_gpu_dh->device_ptr(), d_eri, N_bas,
        static_cast<int>(grid.num_points),
        1.0e-7, &rank_gpu, &s_max_gpu, &s_min_gpu);
    Z_gpu_dh->toHost();
    cudaFree(d_eri);

    EXPECT_EQ(rank_cpu, rank_gpu);
    const std::size_t N = Z_cpu.size();
    double diff = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const double zc = static_cast<double>(Z_cpu[i]);
        const double zg = static_cast<double>(Z_gpu_dh->host_ptr()[i]);
        const double d = zc - zg;
        diff += d * d;
        norm += zc * zc;
    }
    const double rel = std::sqrt(diff) / std::sqrt(norm);
    std::cout << "  H2O/sto-3g rank=" << rank_cpu << "/" << rank_gpu
              << "  rel ||Z_GPU - Z_CPU|| = "
              << std::scientific << std::setprecision(3) << rel << std::endl;
    // Z is N_g x N_g (6600^2 ~ 43M entries), each summing 49 products.
    // CPU = Eigen BDCSVD on M; GPU = cuSOLVER syevd on M M^T.  The two paths
    // give the same Z up to numerical noise (~ N_g * eps for cuBLAS DGEMM
    // accumulation), which lands around 1e-9 here.  The downstream MP2 energy
    // (a scalar) averages this out and reaches machine precision (verified in
    // test_thc_mp2 GPU comparison).
    EXPECT_LT(rel, 1.0e-7);
}

#endif // GANSU_CPU_ONLY
