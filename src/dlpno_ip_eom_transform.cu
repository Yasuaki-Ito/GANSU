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
 * @file dlpno_ip_eom_transform.cu
 * @brief Single-index R2 PNO↔canonical transforms for DLPNO-IP-EOM (stage B
 *        Phase B1). Host/Eigen. See dlpno_ip_eom_transform.hpp for the math.
 */

#include "dlpno_ip_eom_transform.hpp"

#include <Eigen/Dense>

#include "dlpno_pair_data.hpp"   // PairSetup, PairData

namespace gansu {
namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// U_loc handling identical to bt_pno_backtransform: identity if absent/wrong size.
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

std::vector<real_t> ip_packed_r2_to_canonical(
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& packed_r2)
{
    const int nocc = pack.nocc;
    const size_t vstride = static_cast<size_t>(nvir);
    // LMO-indexed canonical-virtual r2: r2_lmo[(i*nocc+j)*nvir + a].
    std::vector<real_t> r2_lmo(static_cast<size_t>(nocc) * nocc * vstride, 0.0);

    Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao, nvir);
    Eigen::Map<const RowMatXd> S(h_S, nao, nao);
    const RowMatXd CvtS = Cv.transpose() * S;            // [nvir × nao]

    const int n_pairs = static_cast<int>(res.pairs.size());
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = pack.n_pno[idx];
        if (n == 0) continue;
        const int i = res.setups[idx].i;
        const int j = res.setups[idx].j;
        Eigen::Map<const RowMatXd> barQ(res.pairs[idx].bar_Q.data(), nao, n);
        const RowMatXd U_ij = CvtS * barQ;                // [nvir × n_pno]

        // (i,j) orientation block.
        {
            const real_t* blk = packed_r2.data() + (pack.off_ij[idx] - nocc);
            Eigen::Map<const Eigen::VectorXd> r2v(blk, n);
            const Eigen::VectorXd r2c = U_ij * r2v;       // [nvir]
            real_t* dst = r2_lmo.data() + (static_cast<size_t>(i) * nocc + j) * vstride;
            for (int a = 0; a < nvir; ++a) dst[a] = r2c(a);
        }
        // (j,i) orientation block (off-diagonal pairs only — independent amplitude).
        if (!pack.diagonal(idx)) {
            const real_t* blk = packed_r2.data() + (pack.off_ji[idx] - nocc);
            Eigen::Map<const Eigen::VectorXd> r2v(blk, n);
            const Eigen::VectorXd r2c = U_ij * r2v;       // [nvir]
            real_t* dst = r2_lmo.data() + (static_cast<size_t>(j) * nocc + i) * vstride;
            for (int a = 0; a < nvir; ++a) dst[a] = r2c(a);
        }
    }

    // Occupied LMO → canonical rotation, per fixed a:
    //   R2_canon[I,J,a] = Σ_{ij} U_loc[I,i] U_loc[J,j] r2_lmo[i,j,a]
    if (uloc_is_identity(U_loc, nocc)) return r2_lmo;

    std::vector<real_t> R2_canon(r2_lmo.size(), 0.0);
    Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);  // U[I,i]
    RowMatXd M(nocc, nocc), C(nocc, nocc);
    for (int a = 0; a < nvir; ++a) {
        for (int i = 0; i < nocc; ++i)
            for (int jj = 0; jj < nocc; ++jj)
                M(i, jj) = r2_lmo[(static_cast<size_t>(i) * nocc + jj) * vstride + a];
        C.noalias() = Uloc * M * Uloc.transpose();
        for (int I = 0; I < nocc; ++I)
            for (int J = 0; J < nocc; ++J)
                R2_canon[(static_cast<size_t>(I) * nocc + J) * vstride + a] = C(I, J);
    }
    return R2_canon;
}

std::vector<real_t> ip_canonical_r2_to_packed(
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& pack,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& R2_canon)
{
    const int nocc = pack.nocc;
    const size_t vstride = static_cast<size_t>(nvir);

    // Inverse occupied rotation (U_loc orthogonal → R2_lmo = U_loc^T R2_canon U_loc per a):
    //   r2_lmo[i,j,a] = Σ_{IJ} U_loc[I,i] U_loc[J,j] R2_canon[I,J,a]
    std::vector<real_t> r2_lmo;
    if (uloc_is_identity(U_loc, nocc)) {
        r2_lmo = R2_canon;
    } else {
        r2_lmo.assign(R2_canon.size(), 0.0);
        Eigen::Map<const RowMatXd> Uloc(U_loc.data(), nocc, nocc);  // U[I,i]
        RowMatXd M(nocc, nocc), C(nocc, nocc);
        for (int a = 0; a < nvir; ++a) {
            for (int I = 0; I < nocc; ++I)
                for (int J = 0; J < nocc; ++J)
                    M(I, J) = R2_canon[(static_cast<size_t>(I) * nocc + J) * vstride + a];
            // r2_lmo(i,j) = Σ_IJ U[I,i] U[J,j] M[I,J] = (U^T M U)[i,j]
            C.noalias() = Uloc.transpose() * M * Uloc;
            for (int i = 0; i < nocc; ++i)
                for (int jj = 0; jj < nocc; ++jj)
                    r2_lmo[(static_cast<size_t>(i) * nocc + jj) * vstride + a] = C(i, jj);
        }
    }

    std::vector<real_t> packed(static_cast<size_t>(pack.total_dim - nocc), 0.0);

    Eigen::Map<const RowMatXd> Cv(C_vir.data(), nao, nvir);
    Eigen::Map<const RowMatXd> S(h_S, nao, nao);
    const RowMatXd CvtS = Cv.transpose() * S;            // [nvir × nao]

    const int n_pairs = static_cast<int>(res.pairs.size());
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = pack.n_pno[idx];
        if (n == 0) continue;
        const int i = res.setups[idx].i;
        const int j = res.setups[idx].j;
        Eigen::Map<const RowMatXd> barQ(res.pairs[idx].bar_Q.data(), nao, n);
        const RowMatXd U_ij = CvtS * barQ;                // [nvir × n_pno]

        // Project canonical-virtual r2 down to PNO(ij): r2_pno = U_ij^T r2_lmo.
        {
            const real_t* src = r2_lmo.data() + (static_cast<size_t>(i) * nocc + j) * vstride;
            Eigen::Map<const Eigen::VectorXd> r2c(src, nvir);
            const Eigen::VectorXd r2p = U_ij.transpose() * r2c;   // [n_pno]
            real_t* dst = packed.data() + (pack.off_ij[idx] - nocc);
            for (int a = 0; a < n; ++a) dst[a] = r2p(a);
        }
        if (!pack.diagonal(idx)) {
            const real_t* src = r2_lmo.data() + (static_cast<size_t>(j) * nocc + i) * vstride;
            Eigen::Map<const Eigen::VectorXd> r2c(src, nvir);
            const Eigen::VectorXd r2p = U_ij.transpose() * r2c;   // [n_pno]
            real_t* dst = packed.data() + (pack.off_ji[idx] - nocc);
            for (int a = 0; a < n; ++a) dst[a] = r2p(a);
        }
    }
    return packed;
}

} // namespace gansu
