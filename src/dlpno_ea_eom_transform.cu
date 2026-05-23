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
 * @file dlpno_ea_eom_transform.cu
 * @brief Two-virtual R2 PNO↔canonical transforms for DLPNO-EA-EOM (stage B).
 *        Host/Eigen. See dlpno_ea_eom_transform.hpp for the math.
 */

#include "dlpno_ea_eom_transform.hpp"

#include <Eigen/Dense>

#include "dlpno_pair_data.hpp"   // PairSetup, PairData

namespace gansu {
namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

bool uloc_is_identity(const std::vector<real_t>& U_loc, int nocc) {
    if (static_cast<int>(U_loc.size()) != nocc * nocc) return true;
    for (int I = 0; I < nocc; ++I)
        for (int k = 0; k < nocc; ++k) {
            const real_t expect = (I == k) ? 1.0 : 0.0;
            if (std::fabs(U_loc[static_cast<size_t>(I) * nocc + k] - expect) > 1e-12)
                return false;
        }
    return true;
}
}  // namespace

std::vector<real_t> ea_packed_r2_to_canonical(
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao,
    const std::vector<real_t>& packed_r2)
{
    const int nocc = pack.nocc;
    const int nvir = pack.nvir;
    const size_t vv = static_cast<size_t>(nvir) * nvir;

    Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao, nvir);
    Eigen::Map<const RowMatXd> S(h_S, nao, nao);
    const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]

    // Per-i: r2_lmo[i] = U^(ii) · r2_pno_i · U^(ii)^T  (nvir × nvir).
    std::vector<real_t> r2_lmo(static_cast<size_t>(nocc) * vv, 0.0);
    for (int i = 0; i < nocc; ++i) {
        const int n = pack.n_pno_ii[i];
        if (n == 0) continue;
        const int idx = res.pair_lookup[static_cast<size_t>(i) * nocc + i];
        Eigen::Map<const RowMatXd> barQ(res.pairs[idx].bar_Q.data(), nao, n);
        const RowMatXd U_ii = CvtS * barQ;                          // [nvir × n]
        Eigen::Map<const RowMatXd> r2p(packed_r2.data() + (pack.off_i[i] - nvir), n, n);
        const RowMatXd r2c = U_ii * r2p * U_ii.transpose();         // [nvir × nvir]
        Eigen::Map<RowMatXd>(r2_lmo.data() + static_cast<size_t>(i) * vv, nvir, nvir) = r2c;
    }

    // Occupied LMO → canonical (single index): viewing r2_lmo as [nocc × nvir²],
    //   R2_canon = U_loc · r2_lmo   (U_loc[I,i]).
    if (uloc_is_identity(U_loc, nocc)) return r2_lmo;
    std::vector<real_t> R2_canon(r2_lmo.size(), 0.0);
    Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);
    Eigen::Map<const RowMatXd> M(r2_lmo.data(), nocc, static_cast<int>(vv));
    Eigen::Map<RowMatXd>(R2_canon.data(), nocc, static_cast<int>(vv)).noalias() = Uloc * M;
    return R2_canon;
}

std::vector<real_t> ea_canonical_r2_to_packed(
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao,
    const std::vector<real_t>& R2_canon)
{
    const int nocc = pack.nocc;
    const int nvir = pack.nvir;
    const size_t vv = static_cast<size_t>(nvir) * nvir;

    // Inverse occupied rotation: r2_lmo = U_loc^T · R2_canon  (as [nocc × nvir²]).
    std::vector<real_t> r2_lmo;
    if (uloc_is_identity(U_loc, nocc)) {
        r2_lmo = R2_canon;
    } else {
        r2_lmo.assign(R2_canon.size(), 0.0);
        Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);
        Eigen::Map<const RowMatXd> M(R2_canon.data(), nocc, static_cast<int>(vv));
        Eigen::Map<RowMatXd>(r2_lmo.data(), nocc, static_cast<int>(vv)).noalias() = Uloc.transpose() * M;
    }

    std::vector<real_t> packed(static_cast<size_t>(pack.total_dim - nvir), 0.0);

    Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao, nvir);
    Eigen::Map<const RowMatXd> S(h_S, nao, nao);
    const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]

    for (int i = 0; i < nocc; ++i) {
        const int n = pack.n_pno_ii[i];
        if (n == 0) continue;
        const int idx = res.pair_lookup[static_cast<size_t>(i) * nocc + i];
        Eigen::Map<const RowMatXd> barQ(res.pairs[idx].bar_Q.data(), nao, n);
        const RowMatXd U_ii = CvtS * barQ;                          // [nvir × n]
        Eigen::Map<const RowMatXd> r2c(r2_lmo.data() + static_cast<size_t>(i) * vv, nvir, nvir);
        // Project both virtuals down to PNO(ii): r2_pno = U^T · r2_canon · U.
        const RowMatXd r2p = U_ii.transpose() * r2c * U_ii;         // [n × n]
        Eigen::Map<RowMatXd>(packed.data() + (pack.off_i[i] - nvir), n, n) = r2p;
    }
    return packed;
}

} // namespace gansu
