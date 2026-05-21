/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file cis_nto_active_space.cu
 * @brief P0 Day 2a — host (Eigen) implementation of the state-averaged CIS
 *        NTO active-space selector.
 *
 * Day 2a delivers the full numerical algorithm in a single CPU path using
 * Eigen, exercised by the bit-exact unit tests in test_cis_nto_active_space.cu
 * and validated against PySCF TDA. Day 2b will replace the Eigen routine with
 * the cuBLAS/cusolver device implementation while keeping the host path as
 * the reference and as the CPU-only fallback.
 *
 * Algorithm (STEOM.md §7):
 *   ρ_occ[i,j] = Σ_n w_n Σ_a C^(n)[i,a] C^(n)[j,a]
 *   ρ_vir[a,b] = Σ_n w_n Σ_i C^(n)[i,a] C^(n)[i,b]
 * then diagonalize each, sort eigenvalues descending, fix the eigenvector
 * sign convention (largest-magnitude entry positive), threshold-filter to get
 * the active NTO set, and emit a human-readable report.
 */

#include "cis_nto_active_space.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <Eigen/Dense>

namespace gansu {

namespace {

// Eigen row-major dense view of the CIS amplitudes for a single state n.
// h_eigenvectors layout: [state * (nocc_active*nvir) + i*nvir + a].
using CISStateMap = Eigen::Map<
    const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

using DenseMat = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;

// Diagonalize a symmetric matrix and return (eigenvalues, eigenvectors) sorted
// in DESCENDING eigenvalue order, with each eigenvector's largest-magnitude
// entry sign-flipped to be positive (run-to-run deterministic sign convention).
void diagonalize_descending_signed(const DenseMat& rho,
                                   Eigen::VectorXd& eigvals,
                                   DenseMat& eigvecs)
{
    Eigen::SelfAdjointEigenSolver<DenseMat> es(rho);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("CISNTOActiveSpace: SelfAdjointEigenSolver failed");
    }
    const Eigen::VectorXd asc_vals  = es.eigenvalues();
    const DenseMat        asc_vecs  = es.eigenvectors();
    const int n = static_cast<int>(asc_vals.size());

    eigvals.resize(n);
    eigvecs.resize(n, n);
    for (int k = 0; k < n; ++k) {
        // Reverse: ascending column (n-1-k) becomes our descending column k.
        const int src = n - 1 - k;
        eigvals(k) = asc_vals(src);
        eigvecs.col(k) = asc_vecs.col(src);

        // Sign convention: largest-magnitude row of this column must be positive.
        Eigen::Index row_max;
        eigvecs.col(k).cwiseAbs().maxCoeff(&row_max);
        if (eigvecs(row_max, k) < 0.0) {
            eigvecs.col(k) *= -1.0;
        }
    }
}

// Build state-averaged density ρ_occ via accumulation
//   ρ_occ += w_n · C_n · C_n^T,   C_n shape (nocc_active, nvir).
DenseMat build_rho_occ(const real_t* h_eigenvectors,
                       int n_states, int nocc_active, int nvir,
                       const std::vector<real_t>& weights)
{
    DenseMat rho = DenseMat::Zero(nocc_active, nocc_active);
    const size_t cis_dim = static_cast<size_t>(nocc_active) * nvir;
    for (int n = 0; n < n_states; ++n) {
        CISStateMap C_n(h_eigenvectors + n * cis_dim, nocc_active, nvir);
        rho.noalias() += weights[n] * (C_n * C_n.transpose());
    }
    return rho;
}

// Build state-averaged density ρ_vir via accumulation
//   ρ_vir += w_n · C_n^T · C_n,   shape (nvir, nvir).
DenseMat build_rho_vir(const real_t* h_eigenvectors,
                       int n_states, int nocc_active, int nvir,
                       const std::vector<real_t>& weights)
{
    DenseMat rho = DenseMat::Zero(nvir, nvir);
    const size_t cis_dim = static_cast<size_t>(nocc_active) * nvir;
    for (int n = 0; n < n_states; ++n) {
        CISStateMap C_n(h_eigenvectors + n * cis_dim, nocc_active, nvir);
        rho.noalias() += weights[n] * (C_n.transpose() * C_n);
    }
    return rho;
}

// Copy an Eigen column-major matrix [n x n] into a row-major std::vector with
// the same logical entries (i.e. preserving entry meaning, not memory order):
//   out[i*n + k] = mat(i, k)
void to_row_major(const DenseMat& mat, std::vector<real_t>& out)
{
    const int n = static_cast<int>(mat.rows());
    out.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            out[static_cast<size_t>(i) * n + k] = mat(i, k);
        }
    }
}

std::string format_spectrum_report(
    const std::vector<real_t>& occ_occ, const std::vector<real_t>& vir_occ,
    real_t o_thresh, real_t v_thresh,
    int verbose)
{
    // verbose == 1: print active rows only + 2 lines of slack above threshold
    // verbose >= 2: print the full spectrum
    std::ostringstream os;
    const auto fmt_occ = std::setprecision(6);
    (void)fmt_occ;

    auto emit_block = [&](const std::string& label, const std::vector<real_t>& vals, real_t thresh) {
        os << "  " << label << " NTO occupations  (threshold = " << thresh << ")\n";
        const int n = static_cast<int>(vals.size());
        int n_act = 0;
        for (real_t v : vals) if (v > thresh) ++n_act;
        // Heuristic for verbose=1: dump n_act + 2 trailing rows for context.
        const int print_n = (verbose >= 2) ? n : std::min(n, n_act + 2);
        for (int k = 0; k < print_n; ++k) {
            const bool active = vals[k] > thresh;
            // Degenerate marker: tolerate Davidson convergence noise (~1e-6).
            // Use a hybrid abs/rel tolerance so small eigenvalues (near 0)
            // also fire only on real degeneracy. Tunes for CIS singlet pairs
            // (π_u / π_g* in N2 etc.) which differ by 1e-7 – 1e-8 in practice.
            const real_t kDegAbs = 1e-6;
            const real_t kDegRel = 1e-4;
            const bool degenerate = (k > 0) && [&] {
                const real_t da = std::abs(vals[k] - vals[k-1]);
                const real_t scale = std::max({std::abs(vals[k]), std::abs(vals[k-1]), real_t(1.0)});
                return da < kDegAbs || da / scale < kDegRel;
            }();
            os << "    [" << std::setw(3) << k << "] n = "
               << std::setw(14) << std::setprecision(10) << std::fixed << vals[k]
               << "  " << (active ? "[ACTIVE]" : "        ")
               << (degenerate ? " [DEGENERATE]" : "")
               << "\n";
        }
        if (print_n < n) {
            os << "    ... (" << (n - print_n) << " more orbital"
               << ((n - print_n) > 1 ? "s" : "") << " below threshold)\n";
        }
        os << "    n_active = " << n_act << " / " << n << "\n";
    };
    emit_block("Occupied", occ_occ, o_thresh);
    emit_block("Virtual ", vir_occ, v_thresh);
    return os.str();
}

} // namespace

CISNTOResult CISNTOActiveSpace::compute(
    const real_t* h_eigenvectors,
    int n_states,
    int nocc_active,
    int nvir,
    int num_frozen,
    const Params& params)
{
    if (n_states <= 0 || nocc_active <= 0 || nvir <= 0) {
        throw std::invalid_argument(
            "CISNTOActiveSpace::compute: n_states, nocc_active and nvir must be positive");
    }
    if (num_frozen < 0) {
        throw std::invalid_argument(
            "CISNTOActiveSpace::compute: num_frozen must be non-negative");
    }
    if (!params.weights.empty() && static_cast<int>(params.weights.size()) != n_states) {
        throw std::invalid_argument(
            "CISNTOActiveSpace::compute: weights size must equal n_states (or be empty for uniform)");
    }
    if (h_eigenvectors == nullptr) {
        throw std::invalid_argument("CISNTOActiveSpace::compute: h_eigenvectors is null");
    }

    // Weight resolution. The weights are NOT renormalized to sum=1: keeping the
    // user's raw weights means trace(ρ_occ) = trace(ρ_vir) = Σ_n w_n ‖C_n‖² and
    // any deviation from Σ_n w_n is itself a sanity tag (e.g. unnormalized CIS
    // root). The default uniform weight is 1/N.
    std::vector<real_t> weights;
    if (params.weights.empty()) {
        weights.assign(n_states, static_cast<real_t>(1.0) / static_cast<real_t>(n_states));
    } else {
        weights = params.weights;
    }

    const DenseMat rho_occ_raw = build_rho_occ(h_eigenvectors, n_states, nocc_active, nvir, weights);
    const DenseMat rho_vir_raw = build_rho_vir(h_eigenvectors, n_states, nocc_active, nvir, weights);

    // Symmetrize against accumulation rounding so the SelfAdjointEigenSolver
    // sees a strictly symmetric input.
    const DenseMat rho_occ = 0.5 * (rho_occ_raw + rho_occ_raw.transpose());
    const DenseMat rho_vir = 0.5 * (rho_vir_raw + rho_vir_raw.transpose());

    Eigen::VectorXd occ_vals, vir_vals;
    DenseMat        occ_vecs, vir_vecs;
    diagonalize_descending_signed(rho_occ, occ_vals, occ_vecs);
    diagonalize_descending_signed(rho_vir, vir_vals, vir_vecs);

    CISNTOResult r;
    r.nocc_active = nocc_active;
    r.nvir        = nvir;
    r.num_frozen  = num_frozen;
    r.o_thresh    = params.o_thresh;
    r.v_thresh    = params.v_thresh;

    r.weight_sum = 0.0;
    for (real_t w : weights) r.weight_sum += w;

    r.trace_occ = rho_occ.trace();
    r.trace_vir = rho_vir.trace();

    r.nto_occ_occupations.assign(nocc_active, 0.0);
    r.nto_vir_occupations.assign(nvir, 0.0);
    for (int k = 0; k < nocc_active; ++k) r.nto_occ_occupations[k] = occ_vals(k);
    for (int k = 0; k < nvir;        ++k) r.nto_vir_occupations[k] = vir_vals(k);

    // Numerical hygiene: a slightly negative eigenvalue from floating roundoff
    // (e.g. -1e-15) is acceptable; anything more negative signals an algorithm
    // bug or a malformed input and must surface.
    constexpr real_t kNegTol = 1e-8;
    for (real_t v : r.nto_occ_occupations) {
        if (v < -kNegTol) {
            throw std::runtime_error(
                "CISNTOActiveSpace::compute: ρ_occ has a meaningfully negative eigenvalue ("
                + std::to_string(v) + "); aborting (density is not PSD).");
        }
    }
    for (real_t v : r.nto_vir_occupations) {
        if (v < -kNegTol) {
            throw std::runtime_error(
                "CISNTOActiveSpace::compute: ρ_vir has a meaningfully negative eigenvalue ("
                + std::to_string(v) + "); aborting (density is not PSD).");
        }
    }

    // Threshold filter (strict >, matches ORCA OThresh / VThresh semantics).
    r.n_act_occ = 0;
    for (real_t v : r.nto_occ_occupations) if (v > params.o_thresh) ++r.n_act_occ;
    r.n_act_vir = 0;
    for (real_t v : r.nto_vir_occupations) if (v > params.v_thresh) ++r.n_act_vir;

    if (r.n_act_occ == 0) {
        throw std::runtime_error(
            "CISNTOActiveSpace::compute: zero active occupied NTOs above o_thresh = "
            + std::to_string(params.o_thresh) + "; loosen the threshold or raise n_states.");
    }
    if (r.n_act_vir == 0) {
        throw std::runtime_error(
            "CISNTOActiveSpace::compute: zero active virtual NTOs above v_thresh = "
            + std::to_string(params.v_thresh) + "; loosen the threshold or raise n_states.");
    }

    r.active_occ_indices.resize(r.n_act_occ);
    for (int k = 0; k < r.n_act_occ; ++k) r.active_occ_indices[k] = k;
    r.active_vir_indices.resize(r.n_act_vir);
    for (int k = 0; k < r.n_act_vir; ++k) r.active_vir_indices[k] = k;

    to_row_major(occ_vecs, r.U_occ);
    to_row_major(vir_vecs, r.U_vir);

    if (params.verbose >= 1) {
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os << "[CIS-NTO] state-averaged active space (bt-PNO-STEOM Phase P0)\n"
           << "  n_states_cis    = " << n_states  << "\n"
           << "  nocc_active     = " << nocc_active
           << "   (num_frozen    = " << num_frozen << ")\n"
           << "  nvir            = " << nvir << "\n"
           << "  weight scheme   = " << (params.weights.empty() ? "uniform" : "user")
           << "   (Σ w_n         = " << std::setprecision(8) << r.weight_sum << ")\n"
           << "  trace(ρ_occ)    = " << std::setprecision(8) << r.trace_occ << "\n"
           << "  trace(ρ_vir)    = " << std::setprecision(8) << r.trace_vir << "\n"
           << "  n_act_occ       = " << r.n_act_occ << " / " << nocc_active << "\n"
           << "  n_act_vir       = " << r.n_act_vir << " / " << nvir << "\n";
        os << format_spectrum_report(r.nto_occ_occupations, r.nto_vir_occupations,
                                     params.o_thresh, params.v_thresh, params.verbose);
        r.report = os.str();
    }

    return r;
}

void CISNTOActiveSpace::make_canonical_projectors(
    const CISNTOResult& result,
    int nocc_canonical,
    std::vector<real_t>& P_occ_can,
    std::vector<real_t>& P_vir_can)
{
    if (nocc_canonical < result.num_frozen + result.nocc_active) {
        throw std::invalid_argument(
            "make_canonical_projectors: nocc_canonical < num_frozen + nocc_active");
    }
    const int n_act_occ = result.n_act_occ;
    const int n_act_vir = result.n_act_vir;

    P_occ_can.assign(static_cast<size_t>(nocc_canonical) * n_act_occ, 0.0);
    // U_occ active columns sit in canonical rows [num_frozen, num_frozen+nocc_active).
    for (int i_act = 0; i_act < result.nocc_active; ++i_act) {
        const int i_can = result.num_frozen + i_act;
        for (int k = 0; k < n_act_occ; ++k) {
            P_occ_can[static_cast<size_t>(i_can) * n_act_occ + k] =
                result.U_occ[static_cast<size_t>(i_act) * result.nocc_active + k];
        }
    }

    P_vir_can.assign(static_cast<size_t>(result.nvir) * n_act_vir, 0.0);
    for (int a = 0; a < result.nvir; ++a) {
        for (int k = 0; k < n_act_vir; ++k) {
            P_vir_can[static_cast<size_t>(a) * n_act_vir + k] =
                result.U_vir[static_cast<size_t>(a) * result.nvir + k];
        }
    }
}

} // namespace gansu
