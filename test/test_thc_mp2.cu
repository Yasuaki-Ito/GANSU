/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_thc_mp2.cu
 * @brief End-to-end Phase 2.0a validation: THC-MP2 energy vs canonical MP2.
 *
 *   1. Run RHF on H2O / sto-3g.
 *   2. Build the analytic 4-index AO ERI, the analytic MO ERI tensor and
 *      the canonical MP2 energy from those.
 *   3. Build the THC factor X^P_mu, Z; transform X to the MO basis; rebuild
 *      the MO ERI tensor through the THC formula
 *        (pq|rs)_THC = sum_{PQ} X^P_p X^P_q Z_{PQ} X^Q_r X^Q_s.
 *   4. Compute MP2 from the THC MO ERI and compare against the canonical
 *      MP2 energy.
 *
 * The comparison metric is the absolute energy difference; with rank(M) = N_pair
 * and a high-resolution grid the difference should sit at the LS-THC noise
 * floor (~1e-12 Ha for sto-3g).
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

#include <Eigen/Dense>

#include "thc_grid.hpp"
#include "thc_collocation.hpp"
#include "thc_decomposition.hpp"
#include "thc_mp2.hpp"
#include "molecular.hpp"
#include "builder.hpp"
#include "parameter_manager.hpp"
#include "device_host_memory.hpp"
#include "hf.hpp"
#include "rhf.hpp"
#include "cpu_integrals.hpp"

using namespace gansu;

namespace {

// -----------------------------------------------------------------------------
// Run RHF, return MO coefficients (column-major), orbital energies and the
// analytic AO ERI tensor.
// -----------------------------------------------------------------------------
struct HFData {
    std::unique_ptr<HF> hf;
    std::vector<real_t> C_ao_to_mo;   // [N_bas x N_orb] col-major
    std::vector<real_t> eps;          // [N_orb]
    std::vector<real_t> eri_ao;       // [N_bas^4]
    int N_bas;
    int N_orb;
    int n_occ;
};

HFData run_hf(const std::string& xyz, const std::string& basis)
{
    ParameterManager params;
    params["xyzfilename"] = xyz;
    params["gbsfilename"] = basis;
    params["method"] = "RHF";
    params["convergence_energy_threshold"] = "1e-10";
    params["initial_guess"] = "core";

    std::streambuf* orig = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());

    HFData fx;
    fx.hf = HFBuilder::buildHF(params);
    fx.hf->solve();

    std::cout.rdbuf(orig);

    fx.N_bas = fx.hf->get_num_basis();
    fx.N_orb = fx.N_bas; // closed-shell, no truncation in this Phase
    fx.n_occ = fx.hf->get_num_alpha_spins();

    // get_coefficient_matrix / get_orbital_energies live on the derived RHF
    // class, not on HF.  HFBuilder returns unique_ptr<HF>; downcast.
    auto* rhf = dynamic_cast<RHF*>(fx.hf.get());
    if (!rhf) throw std::runtime_error("run_hf: expected RHF instance");

    auto& dev_C = rhf->get_coefficient_matrix();
    dev_C.toHost();
    fx.C_ao_to_mo.assign(fx.N_bas * fx.N_orb, 0.0);
    for (std::size_t i = 0; i < fx.C_ao_to_mo.size(); ++i)
        fx.C_ao_to_mo[i] = dev_C.host_ptr()[i];

    auto& dev_eps = rhf->get_orbital_energies();
    dev_eps.toHost();
    fx.eps.assign(fx.N_orb, 0.0);
    for (int i = 0; i < fx.N_orb; ++i) fx.eps[i] = dev_eps.host_ptr()[i];

    // Analytic AO ERI (length N_bas^4) via the CPU integral library.
    Molecular mol(xyz, basis);
    fx.eri_ao.assign(static_cast<std::size_t>(fx.N_bas) * fx.N_bas * fx.N_bas * fx.N_bas,
                     0.0);
    cpu::computeERIMatrix(
        mol.get_shell_type_infos(),
        mol.get_primitive_shells().data(),
        mol.get_cgto_normalization_factors().data(),
        fx.eri_ao.data(),
        fx.N_bas);
    return fx;
}

// -----------------------------------------------------------------------------
// 4-index AO->MO ERI transform.
//
//   eri_ao   ROW-major (GANSU computeERIMatrix layout):
//     eri_ao[a*N^3 + b*N^2 + c*N + d]   = (ab|cd)
//   C        ROW-major (GANSU coefficient_matrix layout):
//     C[mu*N + p]                       = C_{mu, p}
//   eri_mo   COL-major (matches reconstruct_eri_thc_cpu output):
//     eri_mo[p + N*(q + N*(r + N*s))]   = (pq|rs)
//
// Performs the standard 4-step transform with explicit nested loops.  Costs
// 4*N^5 flops -- trivial for N <= 25.
// -----------------------------------------------------------------------------
std::vector<real_t> transform_eri_to_mo(const std::vector<real_t>& eri_ao,
                                        const std::vector<real_t>& C, int N)
{
    auto C_at      = [&C, N](int mu, int p)         { return C[mu * N + p]; };
    auto V_ao_at   = [&eri_ao, N](int a, int b, int c, int d) {
        return eri_ao[((a * N + b) * N + c) * N + d];
    };
    auto idx_T     = [N](int i0, int i1, int i2, int i3) {
        return ((i0 * N + i1) * N + i2) * N + i3; // row-major working layout
    };

    const std::size_t N4 = static_cast<std::size_t>(N) * N * N * N;
    std::vector<real_t> T1(N4, 0.0), T2(N4, 0.0), T3(N4, 0.0);
    std::vector<real_t> eri_mo(N4, 0.0);

    // Step 1: T1[p, b, c, d] = sum_a C[a, p] * V[a, b, c, d]
    for (int p = 0; p < N; ++p)
        for (int b = 0; b < N; ++b)
            for (int c = 0; c < N; ++c)
                for (int d = 0; d < N; ++d) {
                    real_t s = 0.0;
                    for (int a = 0; a < N; ++a)
                        s += C_at(a, p) * V_ao_at(a, b, c, d);
                    T1[idx_T(p, b, c, d)] = s;
                }
    // Step 2: T2[p, q, c, d] = sum_b C[b, q] * T1[p, b, c, d]
    for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            for (int c = 0; c < N; ++c)
                for (int d = 0; d < N; ++d) {
                    real_t s = 0.0;
                    for (int b = 0; b < N; ++b)
                        s += C_at(b, q) * T1[idx_T(p, b, c, d)];
                    T2[idx_T(p, q, c, d)] = s;
                }
    // Step 3: T3[p, q, r, d] = sum_c C[c, r] * T2[p, q, c, d]
    for (int p = 0; p < N; ++p)
        for (int q = 0; q < N; ++q)
            for (int r = 0; r < N; ++r)
                for (int d = 0; d < N; ++d) {
                    real_t s = 0.0;
                    for (int c = 0; c < N; ++c)
                        s += C_at(c, r) * T2[idx_T(p, q, c, d)];
                    T3[idx_T(p, q, r, d)] = s;
                }
    // Step 4: eri_mo[p + N*(q + N*(r + N*s))] = sum_d C[d, s] * T3[p, q, r, d]
    for (int s = 0; s < N; ++s)
        for (int r = 0; r < N; ++r)
            for (int q = 0; q < N; ++q)
                for (int p = 0; p < N; ++p) {
                    real_t v = 0.0;
                    for (int d = 0; d < N; ++d)
                        v += C_at(d, s) * T3[idx_T(p, q, r, d)];
                    eri_mo[p + N * (q + N * (r + N * s))] = v;
                }
    return eri_mo;
}

} // namespace

// =============================================================================
// THC-MP2 vs canonical MP2, H2O / sto-3g.
// =============================================================================

TEST(THCMP2, H2O_STO3G)
{
    HFData hf;
    try {
        hf = run_hf("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run RHF: " << e.what();
        return;
    }

    // ---- Canonical MP2 from analytic MO ERI -------------------------------
    auto eri_mo_ana = transform_eri_to_mo(hf.eri_ao, hf.C_ao_to_mo, hf.N_orb);
    const real_t E_mp2_ana =
        compute_mp2_energy_from_mo_eri_cpu(eri_mo_ana, hf.eps, hf.n_occ, hf.N_orb);

    // ---- THC factorisation in AO ------------------------------------------
    Molecular mol("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 20;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(mol, opts);

    auto X_ao = compute_X_ao_cpu(mol, grid);
    int rank_M = 0;
    real_t s_max = 0, s_min_kept = 0;
    auto Z = compute_Z_via_M_svd_cpu(
        X_ao, hf.eri_ao, hf.N_bas, static_cast<int>(grid.num_points),
        1.0e-7, &rank_M, &s_max, &s_min_kept);

    // ---- THC ERI in MO basis ---------------------------------------------
    auto X_mo = transform_X_to_mo_cpu(
        X_ao, hf.C_ao_to_mo, hf.N_bas, hf.N_orb,
        static_cast<int>(grid.num_points));

    auto eri_mo_thc = reconstruct_eri_thc_cpu(
        X_mo, Z, hf.N_orb, static_cast<int>(grid.num_points));

    const real_t E_mp2_thc =
        compute_mp2_energy_from_mo_eri_cpu(eri_mo_thc, hf.eps, hf.n_occ, hf.N_orb);

    const real_t dE = std::abs(E_mp2_thc - E_mp2_ana);

    std::cout << "  H2O/sto-3g  N_bas=" << hf.N_bas
              << "  n_occ=" << hf.n_occ
              << "  N_g=" << grid.num_points
              << "  rank(M)=" << rank_M << "/" << (hf.N_bas * (hf.N_bas + 1) / 2)
              << "\n      E_MP2(analytic) = " << std::scientific << std::setprecision(10) << E_mp2_ana
              << "\n      E_MP2(THC)      = " << E_mp2_thc
              << "\n      |dE|            = " << std::setprecision(3) << dE << " Ha"
              << std::endl;

    EXPECT_LT(dE, 1.0e-8);
}

#ifndef GANSU_CPU_ONLY

#include <cuda_runtime.h>

// =============================================================================
// End-to-end GPU THC-MP2 vs analytic MP2 (CPU reference).
//
// All four THC stages (collocation -> LS-THC -> MO transform -> MP2) on GPU.
// Final scalar energy must match the CPU analytic MP2 within numerical noise.
// =============================================================================

TEST(THCMP2_GPU, EndToEnd_H2O_STO3G)
{
    HFData hf;
    try {
        hf = run_hf("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run RHF: " << e.what();
        return;
    }

    // ---- Reference: canonical analytic MP2 (CPU) -----------------------------
    auto eri_mo_ana = transform_eri_to_mo(hf.eri_ao, hf.C_ao_to_mo, hf.N_orb);
    const real_t E_mp2_ana =
        compute_mp2_energy_from_mo_eri_cpu(eri_mo_ana, hf.eps, hf.n_occ, hf.N_orb);

    // ---- GPU end-to-end ------------------------------------------------------
    Molecular mol("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    ThcGridOptions opts;
    opts.lebedev = LebedevOrder::L110;
    opts.n_radial = 20;
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(mol, opts);

    // Stage 1: GPU collocation
    auto X_ao_dh = compute_X_ao_gpu(mol, grid);

    // Push AO ERI and C and eps to device.
    real_t* d_eri = nullptr;
    cudaMalloc(&d_eri, hf.eri_ao.size() * sizeof(real_t));
    cudaMemcpy(d_eri, hf.eri_ao.data(),
               hf.eri_ao.size() * sizeof(real_t), cudaMemcpyHostToDevice);

    real_t* d_C = nullptr;
    cudaMalloc(&d_C, hf.C_ao_to_mo.size() * sizeof(real_t));
    cudaMemcpy(d_C, hf.C_ao_to_mo.data(),
               hf.C_ao_to_mo.size() * sizeof(real_t), cudaMemcpyHostToDevice);

    real_t* d_eps = nullptr;
    cudaMalloc(&d_eps, hf.eps.size() * sizeof(real_t));
    cudaMemcpy(d_eps, hf.eps.data(),
               hf.eps.size() * sizeof(real_t), cudaMemcpyHostToDevice);

    // Stage 2: GPU LS-THC -> Z
    int rank = 0;
    real_t s_max = 0, s_min_kept = 0;
    auto Z_dh = compute_Z_via_M_svd_gpu(
        X_ao_dh->device_ptr(), d_eri, hf.N_bas,
        static_cast<int>(grid.num_points),
        1.0e-7, &rank, &s_max, &s_min_kept);

    // Stage 3: MO transform of X
    auto X_mo_dh = transform_X_to_mo_gpu(
        X_ao_dh->device_ptr(), d_C, hf.N_bas, hf.N_orb,
        static_cast<int>(grid.num_points));

    // Stage 4: Reconstruct MO ERI via THC on GPU, then MP2 reduction.
    // No CPU roundtrip -- full GPU pipeline.
    auto eri_mo_thc_dh = reconstruct_eri_thc_gpu(
        X_mo_dh->device_ptr(), Z_dh->device_ptr(),
        hf.N_orb, static_cast<int>(grid.num_points));

    const real_t E_mp2_thc_gpu =
        compute_mp2_energy_from_mo_eri_gpu(eri_mo_thc_dh->device_ptr(), d_eps,
                                            hf.n_occ, hf.N_orb);

    cudaFree(d_eps);
    cudaFree(d_C);
    cudaFree(d_eri);

    const real_t dE = std::abs(E_mp2_thc_gpu - E_mp2_ana);
    std::cout << "  H2O/sto-3g GPU end-to-end"
              << "  N_g=" << grid.num_points
              << "  rank=" << rank
              << "\n      E_MP2(analytic CPU) = " << std::scientific
              << std::setprecision(10) << E_mp2_ana
              << "\n      E_MP2(THC-GPU)      = " << E_mp2_thc_gpu
              << "\n      |dE|                = " << std::setprecision(3) << dE << " Ha"
              << std::endl;
    EXPECT_LT(dE, 1.0e-8);
}

#endif // GANSU_CPU_ONLY
