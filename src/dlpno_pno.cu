/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_pno.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

PNOResult build_pno_from_T(
    const real_t* T_amp,
    bool i_eq_j,
    int n_pao,
    real_t t_cut_pno,
    bool os_only)
{
    PNOResult res;
    if (n_pao <= 0) return res;

    Eigen::Map<const RowMatXd> T(T_amp, n_pao, n_pao);

    // Build the pair density. Two forms; see header for the formulae.
    RowMatXd D;
    if (os_only) {
        // OS-only:   D = (T^T T + T T^T) / (1 + δ_ij)
        D = T.transpose() * T + T * T.transpose();
    } else {
        // Full LMP2: D = (T̃^T T + T̃ T^T) / (1 + δ_ij),   T̃ = 2T − T^T
        const RowMatXd Ttilde = 2.0 * T - T.transpose();
        D = Ttilde.transpose() * T + Ttilde * T.transpose();
    }
    if (i_eq_j) D *= 0.5;
    // Symmetrise to remove FP asymmetry before eigendecomposition.
    D = 0.5 * (D + D.transpose());

    Eigen::SelfAdjointEigenSolver<RowMatXd> es(D);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("PNO eigendecomposition failed");

    // Eigen returns ascending eigenvalues. Re-sort descending for our output.
    Eigen::VectorXd ev = es.eigenvalues();
    RowMatXd        V  = es.eigenvectors();

    res.occupations.resize(n_pao);
    for (int k = 0; k < n_pao; ++k) {
        res.occupations[k] = ev(n_pao - 1 - k);
    }

    // Determine retention. Negative eigenvalues (FP noise on tiny components)
    // are also dropped.
    int n_kept = 0;
    for (int k = 0; k < n_pao; ++k) {
        if (res.occupations[k] > t_cut_pno) ++n_kept;
        else                                 break;
    }
    res.n_kept = n_kept;

    if (n_kept == 0) return res;

    // d_pno[a, ã] = V[:, n_pao-1-ã], so columns of d_pno are the kept PNOs
    // in descending-occupation order.
    RowMatXd d(n_pao, n_kept);
    real_t sum_occ = 0.0;
    for (int kk = 0; kk < n_kept; ++kk) {
        const int src = n_pao - 1 - kk;
        d.col(kk) = V.col(src);
        sum_occ += res.occupations[kk];
    }
    res.sum_occupations = sum_occ;

    res.d_pno.assign(static_cast<size_t>(n_pao) * n_kept, 0.0);
    Eigen::Map<RowMatXd>(res.d_pno.data(), n_pao, n_kept) = d;
    return res;
}

} // namespace gansu
