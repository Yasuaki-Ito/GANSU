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
 * @file dlpno_lambda.hpp
 * @brief DLPNO-MP2 / DLPNO-CCSD Λ (lambda) amplitude solvers (Sub-phase 1/2).
 *
 * Implements analytic-derivative Λ amplitudes for DLPNO-CC theory following
 *   Datta, Kossmann, Neese, J. Chem. Phys. 145, 114101 (2016)
 *   (Level A, orbital-unrelaxed).
 *
 * Sub-phase 1 (this header, MP2):
 *   - compute_dlpno_mp2_lambda(): closed-form Y_lambda = 2 Y - Y^T per pair.
 *
 * Sub-phase 2 (planned, CCSD):
 *   - iterate_dlpno_ccsd_lambda(): iterative Λ residual with DIIS.
 *
 * Design notes: c:\Users\yasuaki\Dropbox\AQUA\DLPNO_Lambda.md
 */

#pragma once

#include <vector>
#include "types.hpp"
#include "dlpno_pair_data.hpp"

namespace gansu {

/**
 * @brief Closed-form DLPNO-MP2 Λ_2 amplitudes (Sub-phase 1, Sub-step 1.1).
 *
 * For each pair (i,j) with i ≤ j stored in `pairs`, populates the per-pair
 * Y_lambda field by the closed-form spin-traced MP2 lambda relation:
 *
 *   Y_lambda[a,b] = 2 · Y[a,b] - Y[b,a]
 *
 * where Y is the converged DLPNO-MP2 amplitude in pair-PNO basis. No iteration
 * is required at MP2 level.
 *
 * Pairs with n_pno == 0 (empty PNO domain) are left with empty Y_lambda.
 *
 * @param pairs Per-pair state with converged Y; Y_lambda is allocated and
 *              filled in-place on return.
 */
void compute_dlpno_mp2_lambda(std::vector<PairData>& pairs);

/**
 * @brief Closed-form DLPNO-CCSD Λ_2 — APPROXIMATION (Sub-phase 2 strategy (a)).
 *
 * Reuses the MP2 closed-form Y_lambda = 2 Y - Y^T applied to the converged
 * CCSD T2 amplitudes. Λ_1 is treated as zero in this approximation (the
 * caller is expected to use the explicit T1 amplitudes directly when
 * building the 1-RDM ov/vo block).
 *
 * This is NOT the exact DLPNO-CCSD Λ from Datta et al. 2016 (which would
 * require iterative solution of the full DLPNO Λ residual equations,
 * Sub-phase 2 strategy (X)). It is, however, exact in the T1=0 limit and
 * provides a practical density for DMET integration when T1 is small
 * (typical of PM-localized DLPNO-CCSD).
 *
 * Use this for early DMET / property prototyping. Validate the resulting
 * density against canonical CCSD make_rdm1() to estimate the approximation
 * error before relying on it in production.
 *
 * @param pairs Per-pair state with converged CCSD T2 amplitudes (Y).
 *              Y_lambda is allocated and filled in-place on return.
 */
void compute_dlpno_ccsd_lambda_closedform(std::vector<PairData>& pairs);

// ===========================================================================
// Sub-phase 2X (full Datta Path A): iterative DLPNO-CCSD Λ residual.
//
// Implements the Lagrangian-derived Λ equations from Datta, Kossmann, Neese
// 2016 (J. Chem. Phys. 145, 114101). The residual structure mirrors the
// existing DLPNO-CCSD T2 iteration (iterate_dlpno_ccsd_t2) but the source
// terms and dressing intermediates differ — see DLPNO_Lambda.md §11.5b and
// the canonical Λ template in src/ccsd_lambda.cu (update_lambda_full).
//
// Sub-step 2X.0 (this commit, scaffolding): the function exists and is
// callable, but currently delegates to the closed-form approximation.
// Full iteration is delivered in Sub-step 2X.1 onwards.
// ===========================================================================

/**
 * @brief Result of a DLPNO-CCSD Λ iteration (parallel to LMP2Status).
 *
 * Same shape as iterate_dlpno_ccsd_t2's return so the driver can use one
 * progress-print template for both T and Λ iterations.
 */
struct DLPNOLambdaStatus {
    int    iters     = 0;     ///< number of iterations performed
    real_t max_R     = 0.0;   ///< final max-abs Λ residual
    bool   converged = false;
};

/**
 * @brief Iterate DLPNO-CCSD Λ_1 + Λ_2 amplitudes (Sub-phase 2X scaffolding).
 *
 * Joint iteration mirroring the canonical update_lambda_full pattern but
 * restricted to per-pair PNO basis with cross-pair barS projections.
 *
 *   Λ_1[i] is per-LMO i in pair (i,i)'s semi-canonical PAO basis
 *          (length setups[pair_lookup[i*nocc+i]].n_pao).
 *   Λ_2 is stored in pairs[idx].Y_lambda (per-pair PNO basis), parallel
 *          to the Y storage of iterate_dlpno_ccsd_t2.
 *
 * Sub-step 2X.0 stub behavior: ignores Lambda1, Lambda1_init, T1, F_LMO,
 * h_S, max_iter, conv_tol, dressing flag, and Phase24Integrals; just sets
 * Y_lambda = 2 Y - Y^T (closed-form, MP2-faithful) and clears Lambda1.
 * Returns iters=0, converged=true.
 *
 * Sub-step 2X.1 (next): replace stub with the proper Λ_2 residual
 * iteration, validating in strict mode that the closed-form is the
 * unique solution at T1=0 (LMP2 limit).
 *
 * @param setups        Per-pair invariants
 * @param pairs         Per-pair PNO state with converged Y (CCSD T2);
 *                      Y_lambda is updated in-place
 * @param Lambda1       Per-LMO Λ_1 amplitudes; updated in-place
 * @param T1            Per-LMO CCSD T1 amplitudes (read-only, for source)
 * @param pair_lookup   [nocc²] pair lookup
 * @param F_LMO         LMO Fock matrix
 * @param h_S           AO overlap matrix
 * @param nocc, nao     dimensions
 * @param max_iter      iteration cap
 * @param conv_tol      max-abs residual tolerance
 * @param enable_dressing   F-eff dressing flag (Sub-step 2X.2+)
 * @param verbose, round_tag   logging knobs
 * @param phase24       optional precomputed integrals for dressing
 * @param num_gpus      multi-GPU count
 */
DLPNOLambdaStatus iterate_dlpno_ccsd_lambda(
    const std::vector<PairSetup>&            setups,
    std::vector<PairData>&                   pairs,
    std::vector<std::vector<real_t>>&        Lambda1,
    const std::vector<std::vector<real_t>>&  T1,
    const std::vector<int>&                  pair_lookup,
    const std::vector<real_t>&               F_LMO,
    const real_t*                            h_S,
    int nocc, int nao,
    int max_iter, real_t conv_tol,
    bool enable_dressing,
    int verbose, const char* round_tag,
    const struct Phase24Integrals* phase24 = nullptr,
    int num_gpus = 1);

} // namespace gansu
