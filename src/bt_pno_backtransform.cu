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
 * @file bt_pno_backtransform.cu
 * @brief PNO → canonical-MO back-transform (bt-PNO-STEOM Phase P5a).
 *        Host/Eigen implementation (no CUDA kernels — runs once after CCSD
 *        convergence, O(N_pair·nvir·n_pno²) + O(nvir²·nocc²), negligible).
 *        See bt_pno_backtransform.hpp for the math and layout contract.
 */

#include "bt_pno_backtransform.hpp"

#include <cmath>
#include <utility>

#include <Eigen/Dense>

#include "dlpno_pair_data.hpp"   // PairSetup, PairData

namespace gansu {
namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

BTAmplitudes bt_pno_to_canonical(
    const DLPNOLMP2Result& res,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<std::vector<real_t>>& T1_pao,
    bool include_t1)
{
    const int nocc = res.nocc;
    BTAmplitudes out;
    out.nocc = nocc;
    out.nvir = nvir;
    const size_t vv     = static_cast<size_t>(nvir) * nvir;
    const size_t t2size = static_cast<size_t>(nocc) * nocc * vv;
    out.T2.assign(t2size, 0.0);
    out.T1.assign(static_cast<size_t>(nocc) * nvir, 0.0);
    if (nocc == 0 || nvir == 0) return out;

    Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao, nvir);
    Eigen::Map<const RowMatXd> S(h_S, nao, nao);
    const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]

    // ----------------------------------------------------------------------
    // Steps 1+2: per-pair T2_lmo on the full LMO grid. Stored pairs are i≤j;
    // the (j,i) orientation is the (a,b)-transpose of T2_lmo_{ij}. No
    // pair_factor scaling (that is an energy weight, not an amplitude scale).
    // ----------------------------------------------------------------------
    std::vector<real_t> T2_lmo(t2size, 0.0);
    const int n_pairs = static_cast<int>(res.pairs.size());
    // Each stored pair idx maps to a unique (i,j) with i≤j and writes only its
    // own [i,j] and [j,i] blocks of T2_lmo, so the writes are disjoint across
    // pairs → safe to parallelise. Per-pair Eigen temporaries are thread-local
    // (declared inside the loop body). The GEMMs are small (nvir × n_pno) so
    // Eigen will not spawn nested threads. Matches the rebuild_cpu_ idiom.
    #pragma omp parallel for schedule(dynamic, 4)
    for (int idx = 0; idx < n_pairs; ++idx) {
        const PairData& pd = res.pairs[idx];
        const int n_pno = pd.n_pno;
        if (n_pno == 0) continue;
        const int i = res.setups[idx].i;
        const int j = res.setups[idx].j;

        Eigen::Map<const RowMatXd> barQ(pd.bar_Q.data(), nao, n_pno);
        Eigen::Map<const RowMatXd> Y(pd.Y.data(), n_pno, n_pno);
        const RowMatXd U_ij = CvtS * barQ;                   // [nvir × n_pno]
        const RowMatXd T2ij = U_ij * Y * U_ij.transpose();   // [nvir × nvir]

        real_t* dst_ij = T2_lmo.data() + (static_cast<size_t>(i) * nocc + j) * vv;
        Eigen::Map<RowMatXd>(dst_ij, nvir, nvir) = T2ij;
        if (i != j) {
            real_t* dst_ji = T2_lmo.data() + (static_cast<size_t>(j) * nocc + i) * vv;
            Eigen::Map<RowMatXd>(dst_ji, nvir, nvir) = T2ij.transpose();
        }
    }

    // ----------------------------------------------------------------------
    // Step 3: occupied LMO → canonical rotation
    //   T2_can[I,J,a,b] = Σ_{ij} U_loc[I,i] U_loc[J,j] T2_lmo[i,j,a,b]
    // For localizer == "none", U_loc is the identity → cheap copy fast path
    // (also avoids the matrix-product roundoff that would pollute the
    //  bit-exact P5a.0/P5a.2-case-1 validation gates).
    // ----------------------------------------------------------------------
    const bool have_uloc = (static_cast<int>(U_loc.size()) == nocc * nocc);
    bool uloc_is_identity = !have_uloc;
    if (have_uloc) {
        uloc_is_identity = true;
        for (int I = 0; I < nocc && uloc_is_identity; ++I)
            for (int k = 0; k < nocc; ++k) {
                const real_t expect = (I == k) ? 1.0 : 0.0;
                if (std::fabs(U_loc[static_cast<size_t>(I) * nocc + k] - expect) > 1e-12) {
                    uloc_is_identity = false;
                    break;
                }
            }
    }

    if (uloc_is_identity) {
        out.T2 = std::move(T2_lmo);
    } else {
        // Occupied LMO → canonical rotation, factored as two GEMMs. This is
        // the SAME contraction (and the same left-to-right reduction order)
        // as the per-(a,b)  C_ab = Uloc · M_ab · Ulocᵀ  form, so it is
        // bit-exact — but with contiguous memory access and parallelism
        // instead of the cache-hostile stride-vv gather/scatter that made
        // this the single largest item in the stage-1 timeline (~5.6 s).
        //   Step A:  tmp[I,(j,a,b)] = Σ_i  Uloc[I,i] · T2_lmo[i,(j,a,b)]
        //   Step B:  out[I,J,(a,b)] = Σ_j  Uloc[J,j] · tmp [I,(j,a,b)]   (∀ I)
        Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);  // U[I,i]
        const Eigen::Index ncol = static_cast<Eigen::Index>(nocc) * vv;
        std::vector<real_t> tmp(t2size);
        Eigen::Map<const RowMatXd> T2lmo_mat(T2_lmo.data(), nocc, ncol);
        Eigen::Map<RowMatXd>       tmp_mat(tmp.data(), nocc, ncol);
        tmp_mat.noalias() = Uloc * T2lmo_mat;            // Step A (one GEMM)

        #pragma omp parallel for schedule(static)
        for (int I = 0; I < nocc; ++I) {                 // Step B (per-I GEMM)
            Eigen::Map<const RowMatXd> tmp_I(
                tmp.data() + static_cast<size_t>(I) * nocc * vv,
                nocc, static_cast<Eigen::Index>(vv));
            Eigen::Map<RowMatXd> out_I(
                out.T2.data() + static_cast<size_t>(I) * nocc * vv,
                nocc, static_cast<Eigen::Index>(vv));
            out_I.noalias() = Uloc * tmp_I;
        }
    }

    // ----------------------------------------------------------------------
    // T1 (P5a.1): per-LMO PAO → canonical virtual, then occupied rotation.
    //   U1^(i) = C_vir^T S C_can_pair(i,i)        [nvir × n_pao_ii]
    //   T1_lmo[i,a] = Σ_{a''} U1^(i)[a,a''] t1_pao[i][a'']
    //   T1_can[I,a] = Σ_i U_loc[I,i] T1_lmo[i,a]
    // ----------------------------------------------------------------------
    if (include_t1 && !T1_pao.empty()) {
        std::vector<real_t> T1_lmo(static_cast<size_t>(nocc) * nvir, 0.0);
        for (int i = 0; i < nocc; ++i) {
            if (static_cast<int>(T1_pao.size()) <= i || T1_pao[i].empty()) continue;
            const int idx_ii = res.pair_lookup[static_cast<size_t>(i) * nocc + i];
            const PairSetup& s = res.setups[idx_ii];
            const int n_pao = s.n_pao;
            if (n_pao == 0 || static_cast<int>(T1_pao[i].size()) != n_pao) continue;
            Eigen::Map<const RowMatXd> Cii(s.C_can_pair.data(), nao, n_pao);
            const RowMatXd U1 = CvtS * Cii;                  // [nvir × n_pao]
            Eigen::Map<const Eigen::VectorXd> t1v(T1_pao[i].data(), n_pao);
            const Eigen::VectorXd t1c = U1 * t1v;            // [nvir]
            for (int a = 0; a < nvir; ++a)
                T1_lmo[static_cast<size_t>(i) * nvir + a] = t1c(a);
        }
        if (uloc_is_identity) {
            out.T1 = std::move(T1_lmo);
        } else {
            Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);
            Eigen::Map<const RowMatXd> T1m(T1_lmo.data(), nocc, nvir);
            Eigen::Map<RowMatXd>(out.T1.data(), nocc, nvir) = Uloc * T1m;
        }
    }

    return out;
}

}  // namespace gansu
