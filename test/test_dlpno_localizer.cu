/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_dlpno_localizer.cu
 * @brief Unit tests for the Pipek-Mezey occupied-MO localiser used by DLPNO.
 *
 * The PM functional L = Σ_i Σ_A (P^A_{ii})^2 monotonically increases under
 * a converged Jacobi sweep; orthonormality of the localised set must be
 * preserved (LMOs span the same occupied subspace). For a small molecule
 * this can be checked end-to-end by running RHF, localising, and verifying
 *   (a) L_final ≥ L_initial
 *   (b) C_LMO^T S C_LMO = I
 *   (c) The occupied projector D = C_LMO C_LMO^T is unchanged from the
 *       canonical-MO version.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "dlpno_localizer.hpp"
#include "builder.hpp"
#include "device_host_memory.hpp"
#include "hf.hpp"
#include "parameter_manager.hpp"
#include "rhf.hpp"

using namespace gansu;

namespace {

struct LocFixture {
    std::unique_ptr<HF> hf;
    std::vector<real_t> C_occ;     // [nao × nocc] row-major
    std::vector<real_t> S;         // [nao × nao] row-major
    std::vector<std::pair<int,int>> atom_ao_ranges;
    int nao = 0;
    int nocc = 0;
};

LocFixture run_rhf_collect_occ(const std::string& xyz, const std::string& basis)
{
    ParameterManager params;
    params["xyzfilename"] = xyz;
    params["gbsfilename"] = basis;
    params["method"] = "RHF";
    params["convergence_energy_threshold"] = "1e-9";
    params["initial_guess"] = "core";

    std::streambuf* orig = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());

    LocFixture fx;
    fx.hf = HFBuilder::buildHF(params);
    fx.hf->solve();
    std::cout.rdbuf(orig);

    auto* rhf = dynamic_cast<RHF*>(fx.hf.get());
    if (!rhf) throw std::runtime_error("Not an RHF object");

    auto& devC = rhf->get_coefficient_matrix();
    auto& devS = rhf->get_overlap_matrix();
    devC.toHost();
    devS.toHost();

    fx.nao  = rhf->get_num_basis();
    fx.nocc = rhf->get_num_electrons() / 2;

    // C_full is row-major [nao × nao]; copy first nocc columns into [nao × nocc].
    fx.C_occ.assign(static_cast<size_t>(fx.nao) * fx.nocc, 0.0);
    const real_t* Cfull = devC.host_ptr();
    for (int mu = 0; mu < fx.nao; ++mu) {
        for (int i = 0; i < fx.nocc; ++i) {
            fx.C_occ[mu * fx.nocc + i] = Cfull[mu * fx.nao + i];
        }
    }

    fx.S.assign(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    const real_t* Sptr = devS.host_ptr();
    for (size_t i = 0; i < fx.S.size(); ++i) fx.S[i] = Sptr[i];

    const auto& a2b = rhf->get_atom_to_basis_range();
    fx.atom_ao_ranges.clear();
    fx.atom_ao_ranges.reserve(a2b.size());
    for (const auto& r : a2b) {
        fx.atom_ao_ranges.emplace_back(static_cast<int>(r.start_index),
                                       static_cast<int>(r.end_index));
    }
    return fx;
}

// max_{ij} |M_{ij}|
real_t max_abs(const std::vector<real_t>& M)
{
    real_t m = 0.0;
    for (auto v : M) m = std::max<real_t>(m, std::fabs(v));
    return m;
}

// Compute C_a^T · S · C_b (row-major, sizes ka, kb).
std::vector<real_t> overlap_block(const real_t* Ca, int ka,
                                  const real_t* Cb, int kb,
                                  const real_t* S, int nao)
{
    std::vector<real_t> SC(static_cast<size_t>(nao) * kb, 0.0);
    for (int mu = 0; mu < nao; ++mu) {
        for (int b = 0; b < kb; ++b) {
            real_t v = 0.0;
            for (int nu = 0; nu < nao; ++nu) {
                v += S[mu * nao + nu] * Cb[nu * kb + b];
            }
            SC[mu * kb + b] = v;
        }
    }
    std::vector<real_t> M(static_cast<size_t>(ka) * kb, 0.0);
    for (int a = 0; a < ka; ++a) {
        for (int b = 0; b < kb; ++b) {
            real_t v = 0.0;
            for (int mu = 0; mu < nao; ++mu) {
                v += Ca[mu * ka + a] * SC[mu * kb + b];
            }
            M[a * kb + b] = v;
        }
    }
    return M;
}

} // namespace

// =========================================================================
// 1. Functional non-decrease + orthonormality + invariance of the projector
// =========================================================================

TEST(DLPNOLocalizer, PipekMezey_H2O_STO3G)
{
    LocFixture fx;
    try {
        fx = run_rhf_collect_occ("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run HF for H2O/sto-3g: " << e.what();
        return;
    }
    ASSERT_GT(fx.nocc, 0);
    ASSERT_GT(fx.nao, 0);

    const real_t L0 = pipek_mezey_functional(
        fx.C_occ.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges);

    auto res = localize_pipek_mezey(
        fx.C_occ.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges,
        /*max_sweep=*/200, /*conv_tol=*/1e-12, /*verbose=*/0);

    EXPECT_TRUE(res.converged);

    // (a) Functional must be non-decreasing.
    EXPECT_GE(res.functional_final, L0 - 1e-12);

    // (b) Orthonormality of the LMO set: C_LMO^T · S · C_LMO = I.
    auto SS = overlap_block(res.C_LMO.data(), fx.nocc,
                            res.C_LMO.data(), fx.nocc,
                            fx.S.data(), fx.nao);
    real_t max_off_diag = 0.0;
    real_t max_diag_err = 0.0;
    for (int i = 0; i < fx.nocc; ++i) {
        for (int j = 0; j < fx.nocc; ++j) {
            const real_t v = SS[i * fx.nocc + j];
            if (i == j) max_diag_err = std::max(max_diag_err, std::fabs(v - 1.0));
            else        max_off_diag = std::max(max_off_diag, std::fabs(v));
        }
    }
    EXPECT_LT(max_off_diag, 1e-10);
    EXPECT_LT(max_diag_err, 1e-10);

    // (c) Same occupied projector: D_canon = C_occ · C_occ^T should equal
    //     D_lmo = C_LMO · C_LMO^T (both [nao × nao]).
    std::vector<real_t> D_canon(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    std::vector<real_t> D_lmo(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    for (int mu = 0; mu < fx.nao; ++mu) {
        for (int nu = 0; nu < fx.nao; ++nu) {
            real_t a = 0.0, b = 0.0;
            for (int i = 0; i < fx.nocc; ++i) {
                a += fx.C_occ[mu * fx.nocc + i] * fx.C_occ[nu * fx.nocc + i];
                b += res.C_LMO[mu * fx.nocc + i] * res.C_LMO[nu * fx.nocc + i];
            }
            D_canon[mu * fx.nao + nu] = a;
            D_lmo[mu * fx.nao + nu]   = b;
        }
    }
    real_t maxd = 0.0;
    for (size_t k = 0; k < D_canon.size(); ++k) {
        maxd = std::max(maxd, std::fabs(D_canon[k] - D_lmo[k]));
    }
    EXPECT_LT(maxd, 1e-10);

    std::cout << "  H2O/sto-3g  L0=" << std::scientific << std::setprecision(6)
              << L0 << "  L_final=" << res.functional_final
              << "  sweeps=" << res.n_sweeps
              << "  ||SS-I||_max=" << std::max(max_off_diag, max_diag_err)
              << "  ||ΔD||_max=" << maxd << std::endl;

    // Localisation actually makes L strictly larger for H2O (5 occupied MOs
    // → 2×OH + 2×lp + 1×1s_O after PM).
    EXPECT_GT(res.functional_final, L0 + 1e-6);
}

// =========================================================================
// 2. Idempotency: rerun PM on already localised orbitals → no change.
// =========================================================================

TEST(DLPNOLocalizer, Idempotent_OnConverged)
{
    LocFixture fx;
    try {
        fx = run_rhf_collect_occ("../xyz/H2O.xyz", "../basis/sto-3g.gbs");
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Cannot run HF: " << e.what();
        return;
    }

    auto res1 = localize_pipek_mezey(
        fx.C_occ.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges, 200, 1e-12, 0);
    EXPECT_TRUE(res1.converged);

    auto res2 = localize_pipek_mezey(
        res1.C_LMO.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges, 200, 1e-12, 0);
    EXPECT_TRUE(res2.converged);

    // Functional must not change.
    EXPECT_NEAR(res1.functional_final, res2.functional_final, 1e-12);
    // The second sweep should detect convergence within a couple of sweeps.
    EXPECT_LE(res2.n_sweeps, 3);

    // C_LMO_2 ≈ C_LMO_1 up to sign permutation. Easier: occupied projectors
    // must coincide.
    std::vector<real_t> D1(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    std::vector<real_t> D2(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    for (int mu = 0; mu < fx.nao; ++mu) {
        for (int nu = 0; nu < fx.nao; ++nu) {
            real_t a = 0.0, b = 0.0;
            for (int i = 0; i < fx.nocc; ++i) {
                a += res1.C_LMO[mu * fx.nocc + i] * res1.C_LMO[nu * fx.nocc + i];
                b += res2.C_LMO[mu * fx.nocc + i] * res2.C_LMO[nu * fx.nocc + i];
            }
            D1[mu * fx.nao + nu] = a;
            D2[mu * fx.nao + nu] = b;
        }
    }
    real_t maxd = 0.0;
    for (size_t k = 0; k < D1.size(); ++k) {
        maxd = std::max(maxd, std::fabs(D1[k] - D2[k]));
    }
    EXPECT_LT(maxd, 1e-12);
}
