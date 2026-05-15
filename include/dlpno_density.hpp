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
 * @file dlpno_density.hpp
 * @brief DLPNO 1-RDM construction in canonical MO/AO basis (Sub-phase 1 — MP2).
 *
 * Builds the spin-traced one-particle density matrix from DLPNO-MP2 amplitudes
 * (Y) and lambda amplitudes (Y_lambda) following Datta, Kossmann, Neese 2016
 * Eq. 38-41 (Level A, orbital-unrelaxed).
 *
 * For Sub-phase 1 (MP2):
 *   D^oo (LMO basis, cross-pair sum via barS overlap)
 *   D^vv (per-pair PNO → back-transform → canonical virtual basis)
 *   D^ov = D^vo = 0  (T1 = Λ1 = 0 at MP2 level)
 *   HF reference 2·I added to occupied diagonal.
 *
 * Output D_mo is then transformed to AO basis via transform_density_mo_to_ao_cpu
 * (already implemented in ccsd_lambda.cu) for dipole / Mulliken / etc.
 *
 * Design notes: c:\Users\yasuaki\Dropbox\AQUA\DLPNO_Lambda.md §4.3, §6.
 */

#pragma once

#include <vector>
#include "types.hpp"
#include "dlpno_pair_data.hpp"

namespace gansu {

/**
 * @brief Build DLPNO-MP2 1-RDM in canonical MO basis (Sub-phase 1, Sub-step 1.2-1.5).
 *
 * Steps performed (see DLPNO_Lambda.md §6):
 *   ρ.1: D^oo block in LMO basis (cross-pair sum + barS projection)
 *   ρ.2: D^vv block per-pair (Y · Y_lambda contraction in PNO basis)
 *   ρ.3: D^vv back-transform per-pair PNO → canonical virtual basis
 *   ρ.4: HF reference (2·I_no on occupied diagonal)
 *
 * Convention:
 *   - D_mo_out is [nmo × nmo] row-major where nmo = n_lmo + n_canonical_vir.
 *   - Occupied block uses LMO indexing (not canonical occupied).
 *   - Virtual block uses canonical virtual indexing (after PNO → canonical
 *     back-transform).
 *   - Trace check: tr(D_mo_out) = N_elec.
 *
 * Caller must subsequently transform to AO basis via:
 *   transform_density_mo_to_ao_cpu(nmo, h_C, D_mo, D_ao_out)
 * using the [nao × nmo] coefficient matrix C with the same column ordering
 * (LMO first, canonical virtual second).
 *
 * Sub-phase 1 limitation: assumes Y_lambda already populated via
 * compute_dlpno_mp2_lambda(). Pairs with n_pno == 0 contribute zero.
 *
 * @param setups      Per-pair invariants ([n_pairs])
 * @param pairs       Per-pair PNO state with Y and Y_lambda populated
 * @param pair_lookup [nocc × nocc] flattened pair index (i,j ↔ j,i)
 * @param n_lmo       Number of LMOs (= occupied count)
 * @param n_can_vir   Number of canonical virtuals
 * @param S_AO        [nao × nao] AO overlap matrix (row-major)
 * @param C_can_vir   [nao × n_can_vir] canonical virtual MO coefficients
 *                    (column-wise, MO index along columns)
 * @param nao         AO basis dimension
 * @param D_mo_out    Output [(n_lmo + n_can_vir)²] row-major, MO basis
 */
void build_dlpno_mp2_1rdm_mo(
    const std::vector<PairSetup>& setups,
    const std::vector<PairData>&  pairs,
    const std::vector<int>&       pair_lookup,
    int                           n_lmo,
    int                           n_can_vir,
    const real_t*                 S_AO,
    const real_t*                 C_can_vir,
    int                           nao,
    real_t*                       D_mo_out);

/**
 * @brief Build DLPNO-CCSD 1-RDM in canonical MO basis — APPROXIMATION
 *        (Sub-phase 2 strategy (a)).
 *
 * Wraps build_dlpno_mp2_1rdm_mo() to populate the oo + vv blocks (using
 * CCSD T2 + closed-form Y_lambda = 2T - T^T), then adds the explicit T1
 * contribution to the ov/vo blocks:
 *
 *   D[ov][i, a] = D[vo][a, i] = (back-transform of T1[i] from pair-(i,i)
 *                               PAO basis to canonical virtual basis)
 *
 * The closed-form approximation for Y_lambda implies Λ_1 = 0; the explicit
 * T1 here is the leading "Hartree-Fock-like" contribution to the ov/vo
 * blocks of the canonical CCSD 1-RDM (ccsd_rdm._gamma1_intermediates dvo
 * line: dvo = t1.T + …small T1·T2 corrections that are dropped here).
 *
 * Trace conservation: holds (the ov/vo additions are pure off-diagonal,
 * traceless contributions to the symmetric MO density).
 *
 * @param T1   Per-LMO T1 amplitudes; T1[i] is in pair (i,i)'s semi-canonical
 *             PAO basis (length setups[pair_lookup[i*nocc+i]].n_pao).
 *             Pass an empty vector or vectors of size 0 to skip the T1
 *             contribution and recover the MP2 result.
 */
void build_dlpno_ccsd_1rdm_mo_closedform(
    const std::vector<PairSetup>&         setups,
    const std::vector<PairData>&          pairs,
    const std::vector<int>&               pair_lookup,
    const std::vector<std::vector<real_t>>& T1,
    int                                   n_lmo,
    int                                   n_can_vir,
    const real_t*                         S_AO,
    const real_t*                         C_can_vir,
    int                                   nao,
    real_t*                               D_mo_out);

} // namespace gansu
