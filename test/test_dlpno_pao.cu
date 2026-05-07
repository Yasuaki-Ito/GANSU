/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file test_dlpno_pao.cu
 * @brief Unit tests for the PAO construction and per-domain Löwdin
 *        orthogonalisation used by DLPNO.
 *
 * Verifications:
 *   - Global PAO matrix C̃ = I − D_occ S satisfies C_occ^T S C̃ = 0
 *     (PAOs are S-orthogonal to all occupied MOs).
 *   - Per-domain orthogonalised PAO set has C^T S C = I_{n_kept}.
 *   - Restricting the domain to all AOs yields a PAO subspace of rank
 *     N_vir = nao − nocc.
 *   - Building the LMO domains via Boughton-Pulay on H2O/sto-3g produces
 *     a domain that contains the LMO's own atom (sanity check).
 */

#include <gtest/gtest.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "dlpno_domain.hpp"
#include "dlpno_localizer.hpp"
#include "dlpno_pao.hpp"
#include "builder.hpp"
#include "device_host_memory.hpp"
#include "hf.hpp"
#include "parameter_manager.hpp"
#include "rhf.hpp"

using namespace gansu;

namespace {

struct PaoFixture {
    std::unique_ptr<HF> hf;
    std::vector<real_t> C_occ;     // [nao × nocc]
    std::vector<real_t> S;         // [nao × nao]
    std::vector<std::pair<int,int>> atom_ao_ranges;
    int nao = 0;
    int nocc = 0;
};

PaoFixture run_rhf(const std::string& xyz, const std::string& basis)
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

    PaoFixture fx;
    fx.hf = HFBuilder::buildHF(params);
    fx.hf->solve();
    std::cout.rdbuf(orig);

    auto* rhf = dynamic_cast<RHF*>(fx.hf.get());
    auto& devC = rhf->get_coefficient_matrix();
    auto& devS = rhf->get_overlap_matrix();
    devC.toHost();
    devS.toHost();

    fx.nao  = rhf->get_num_basis();
    fx.nocc = rhf->get_num_electrons() / 2;

    fx.C_occ.assign(static_cast<size_t>(fx.nao) * fx.nocc, 0.0);
    const real_t* Cfull = devC.host_ptr();
    for (int mu = 0; mu < fx.nao; ++mu)
        for (int i = 0; i < fx.nocc; ++i)
            fx.C_occ[mu * fx.nocc + i] = Cfull[mu * fx.nao + i];

    fx.S.assign(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    const real_t* Sptr = devS.host_ptr();
    for (size_t i = 0; i < fx.S.size(); ++i) fx.S[i] = Sptr[i];

    const auto& a2b = rhf->get_atom_to_basis_range();
    for (const auto& r : a2b) {
        fx.atom_ao_ranges.emplace_back(static_cast<int>(r.start_index),
                                       static_cast<int>(r.end_index));
    }
    return fx;
}

real_t max_abs_off_diag(const std::vector<real_t>& M, int n)
{
    real_t m = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (i != j) m = std::max(m, std::fabs(M[i * n + j]));
    return m;
}

real_t max_abs_diag_minus_one(const std::vector<real_t>& M, int n)
{
    real_t m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(M[i * n + i] - 1.0));
    return m;
}

// (n × m) C^T · S · C, where C is row-major (nao × m). Returns row-major (m×m).
std::vector<real_t> CtSC(const real_t* C, int m,
                         const real_t* S, int nao)
{
    std::vector<real_t> SC(static_cast<size_t>(nao) * m, 0.0);
    for (int mu = 0; mu < nao; ++mu)
        for (int b = 0; b < m; ++b) {
            real_t v = 0.0;
            for (int nu = 0; nu < nao; ++nu)
                v += S[mu * nao + nu] * C[nu * m + b];
            SC[mu * m + b] = v;
        }
    std::vector<real_t> M(static_cast<size_t>(m) * m, 0.0);
    for (int a = 0; a < m; ++a)
        for (int b = 0; b < m; ++b) {
            real_t v = 0.0;
            for (int mu = 0; mu < nao; ++mu)
                v += C[mu * m + a] * SC[mu * m + b];
            M[a * m + b] = v;
        }
    return M;
}

} // namespace

// =========================================================================
// 1. C_occ^T · S · C̃ = 0  (PAOs orthogonal to occupied space).
// =========================================================================
TEST(DLPNOPao, GlobalPAO_OrthogonalToOccupied)
{
    PaoFixture fx;
    try { fx = run_rhf("../xyz/H2O.xyz", "../basis/sto-3g.gbs"); }
    catch (const std::exception& e) { GTEST_SKIP() << e.what(); return; }

    auto C_pao = build_pao_global(
        fx.C_occ.data(), fx.S.data(), fx.nao, fx.nocc);
    ASSERT_EQ(C_pao.size(),
              static_cast<size_t>(fx.nao) * fx.nao);

    // Compute C_occ^T · S · C_pao  →  shape (nocc × nao). Should be ≈ 0.
    std::vector<real_t> SC_pao(static_cast<size_t>(fx.nao) * fx.nao, 0.0);
    for (int mu = 0; mu < fx.nao; ++mu)
        for (int b = 0; b < fx.nao; ++b) {
            real_t v = 0.0;
            for (int nu = 0; nu < fx.nao; ++nu)
                v += fx.S[mu * fx.nao + nu] * C_pao[nu * fx.nao + b];
            SC_pao[mu * fx.nao + b] = v;
        }
    real_t max_off = 0.0;
    for (int i = 0; i < fx.nocc; ++i)
        for (int b = 0; b < fx.nao; ++b) {
            real_t v = 0.0;
            for (int mu = 0; mu < fx.nao; ++mu)
                v += fx.C_occ[mu * fx.nocc + i] * SC_pao[mu * fx.nao + b];
            max_off = std::max(max_off, std::fabs(v));
        }

    EXPECT_LT(max_off, 1e-10);
}

// =========================================================================
// 2. Full-AO domain: rank = N_vir, orthonormal after Löwdin.
// =========================================================================
TEST(DLPNOPao, FullDomain_RankEqualsNvir)
{
    PaoFixture fx;
    try { fx = run_rhf("../xyz/H2O.xyz", "../basis/sto-3g.gbs"); }
    catch (const std::exception& e) { GTEST_SKIP() << e.what(); return; }

    auto C_pao = build_pao_global(fx.C_occ.data(), fx.S.data(),
                                  fx.nao, fx.nocc);
    std::vector<int> all_aos(fx.nao);
    for (int mu = 0; mu < fx.nao; ++mu) all_aos[mu] = mu;

    auto dom = orthogonalize_pao_domain(
        C_pao.data(), fx.S.data(), all_aos, fx.nao, /*t_cut_do=*/1e-8);

    EXPECT_EQ(dom.n_kept, fx.nao - fx.nocc);
    EXPECT_EQ(dom.n_redundant_dropped, fx.nocc);

    auto SS = CtSC(dom.C_pao_orth.data(), dom.n_kept,
                   fx.S.data(), fx.nao);
    EXPECT_LT(max_abs_off_diag(SS, dom.n_kept), 1e-10);
    EXPECT_LT(max_abs_diag_minus_one(SS, dom.n_kept), 1e-10);
}

// =========================================================================
// 3. Boughton-Pulay domain selection on PM-localised H2O.
//
//    For H2O the PM-localised orbitals correspond to:
//      O 1s core, two O lone pairs, two O–H bond LMOs.
//    Each LMO must include the oxygen atom (since every LMO is centred on
//    or near O), and the OH bond LMOs must additionally include an H atom.
// =========================================================================
TEST(DLPNOPao, DomainSelection_H2O_STO3G)
{
    PaoFixture fx;
    try { fx = run_rhf("../xyz/H2O.xyz", "../basis/sto-3g.gbs"); }
    catch (const std::exception& e) { GTEST_SKIP() << e.what(); return; }

    auto loc = localize_pipek_mezey(
        fx.C_occ.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges, 200, 1e-12, 0);
    ASSERT_TRUE(loc.converged);

    auto domains = build_lmo_domains(
        loc.C_LMO.data(), fx.S.data(),
        fx.nao, fx.nocc, fx.atom_ao_ranges,
        /*t_cut_mkn=*/1e-3, /*verbose=*/0);

    ASSERT_EQ(static_cast<int>(domains.size()), fx.nocc);

    // Every LMO must have the O atom (atom 0 in the H2O xyz layout) since
    // every occupied MO has substantial weight on oxygen.
    int oxygen_atom_id = -1;
    for (int a = 0; a < static_cast<int>(fx.atom_ao_ranges.size()); ++a) {
        const int n_basis = fx.atom_ao_ranges[a].second
                          - fx.atom_ao_ranges[a].first;
        if (n_basis >= 5) { oxygen_atom_id = a; break; }  // O: 5 basis fns in sto-3g
    }
    ASSERT_GE(oxygen_atom_id, 0);

    int with_oxygen = 0;
    int with_hydrogen = 0;
    for (const auto& d : domains) {
        bool has_O = false, has_H = false;
        for (int a : d.atom_indices) {
            if (a == oxygen_atom_id) has_O = true;
            else has_H = true;
        }
        if (has_O) with_oxygen++;
        if (has_H) with_hydrogen++;
        EXPECT_GE(d.mulliken_completeness, 1.0 - 1e-3 - 1e-12);
    }
    EXPECT_EQ(with_oxygen, fx.nocc); // all LMOs touch O
    EXPECT_GE(with_hydrogen, 2);     // OH bond LMOs touch H
}

// =========================================================================
// 4. set-union of two LMO domains is sorted and unique (pair domain helper).
// =========================================================================
TEST(DLPNOPao, MergeDomains)
{
    std::vector<int> a = {0, 2, 5, 7};
    std::vector<int> b = {1, 2, 6, 7};
    auto m = merge_ao_index_sets(a, b);
    ASSERT_EQ(m.size(), 6u);
    EXPECT_EQ(m[0], 0);
    EXPECT_EQ(m[1], 1);
    EXPECT_EQ(m[2], 2);
    EXPECT_EQ(m[3], 5);
    EXPECT_EQ(m[4], 6);
    EXPECT_EQ(m[5], 7);
}
