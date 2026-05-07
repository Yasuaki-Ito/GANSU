/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_pao.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}

// ---------------------------------------------------------------------------
//   C̃^{PAO} = I − D_occ · S,   D_occ = C_occ · C_occ^T   (idempotent)
// ---------------------------------------------------------------------------
std::vector<real_t> build_pao_global(
    const real_t* C_occ,
    const real_t* S,
    int nao, int nocc)
{
    if (nao <= 0)
        throw std::runtime_error("build_pao_global: nao must be > 0");
    if (nocc < 0 || nocc > nao)
        throw std::runtime_error("build_pao_global: invalid nocc");

    Eigen::Map<const RowMatXd> Cocc(C_occ, nao, nocc);
    Eigen::Map<const RowMatXd> Smat(S, nao, nao);

    // D_occ = C_occ · C_occ^T (closed-shell idempotent projector form;
    // physical density is 2·D_occ but we want the projector here).
    const RowMatXd D = Cocc * Cocc.transpose();

    // C̃^{PAO} = I − D · S
    RowMatXd Cpao = -(D * Smat);
    for (int mu = 0; mu < nao; mu++) Cpao(mu, mu) += 1.0;

    std::vector<real_t> out(static_cast<size_t>(nao) * nao);
    Eigen::Map<RowMatXd>(out.data(), nao, nao) = Cpao;
    return out;
}

// ---------------------------------------------------------------------------
//   Per-domain Löwdin orthogonalisation with redundancy removal.
// ---------------------------------------------------------------------------
PAODomainResult orthogonalize_pao_domain(
    const real_t* C_pao_global,
    const real_t* S,
    const std::vector<int>& domain_ao_indices,
    int nao,
    real_t t_cut_do)
{
    PAODomainResult res;
    const int d = static_cast<int>(domain_ao_indices.size());
    if (d == 0) return res;

    Eigen::Map<const RowMatXd> Cpao(C_pao_global, nao, nao);
    Eigen::Map<const RowMatXd> Smat(S, nao, nao);

    // Step 1: extract the d domain columns of C̃^{PAO}.
    RowMatXd Cdom(nao, d);
    for (int k = 0; k < d; k++) {
        const int col = domain_ao_indices[k];
        if (col < 0 || col >= nao)
            throw std::runtime_error("PAO domain index out of range");
        Cdom.col(k) = Cpao.col(col);
    }

    // Step 2: domain PAO overlap S_dom = C_dom^T · S · C_dom (d × d).
    const RowMatXd SCdom = Smat * Cdom;       // (nao × d)
    RowMatXd Sdom = Cdom.transpose() * SCdom; // (d × d)
    // Symmetrise to remove any FP asymmetry before eigendecomposition.
    Sdom = 0.5 * (Sdom + Sdom.transpose());

    // Step 3: eigendecomposition (ascending eigenvalues from SelfAdjoint).
    Eigen::SelfAdjointEigenSolver<RowMatXd> es(Sdom);
    if (es.info() != Eigen::Success)
        throw std::runtime_error("PAO domain overlap eigendecomp failed");

    Eigen::VectorXd eigvals = es.eigenvalues();
    RowMatXd        eigvecs = es.eigenvectors();

    // Step 4: drop redundant directions (λ < t_cut_do).
    const int n_total = d;
    int n_kept = 0;
    for (int k = 0; k < n_total; k++) {
        if (eigvals(k) > t_cut_do) n_kept++;
    }
    const int n_dropped = n_total - n_kept;

    res.n_kept = n_kept;
    res.n_redundant_dropped = n_dropped;
    // Store eigenvalues in descending order for diagnostics.
    res.overlap_eigenvalues.resize(n_total);
    for (int k = 0; k < n_total; k++) {
        res.overlap_eigenvalues[k] = eigvals(n_total - 1 - k);
    }

    if (n_kept == 0) {
        res.C_pao_orth.clear();
        return res;
    }

    // Step 5: M = V_kept · diag(1/√λ_kept). Kept = top n_kept eigenvalues
    // (largest), which appear at the *end* of ascending Eigen output.
    RowMatXd M(n_total, n_kept);
    for (int k = 0; k < n_kept; k++) {
        const int src_idx = n_total - n_kept + k;  // ascending → kept tail
        const real_t lam = eigvals(src_idx);
        const real_t inv_sqrt = 1.0 / std::sqrt(lam);
        M.col(k) = eigvecs.col(src_idx) * inv_sqrt;
    }
    // Final orthonormal PAOs in global AO basis: C^{orth} = C_dom · M.
    RowMatXd Corth = Cdom * M;  // (nao × n_kept)

    res.C_pao_orth.assign(static_cast<size_t>(nao) * n_kept, 0.0);
    Eigen::Map<RowMatXd>(res.C_pao_orth.data(), nao, n_kept) = Corth;
    return res;
}

} // namespace gansu
