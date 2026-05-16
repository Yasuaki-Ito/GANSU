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
 * @brief Build DLPNO-CCSD 1-RDM in canonical MO basis.
 *
 * Wraps build_dlpno_mp2_1rdm_mo() to populate the oo + vv blocks (using
 * CCSD T2 + closed-form Y_lambda = 2T - T^T), then adds explicit T1 and
 * Λ_1 contributions to the ov/vo blocks.
 *
 * Canonical CCSD 1-RDM (PySCF closed-shell ccsd_rdm._gamma1_intermediates
 * spin-traced, see GANSU [src/ccsd_lambda.cu](src/ccsd_lambda.cu)
 * build_ccsd_1rdm_mo_cpu lines 1389-1518):
 *
 *   dov[i, a] = L1[i, a]
 *   dvo[a, i] = T1[i, a]
 *             + Σ_me θ[i,m,a,e] · L1[m, e]
 *             - Σ_m xt1[m,i] · T1[m, a]
 *             - Σ_e T1[i, e] · xt2[e, a]
 *   D[ov][i, a] = dov[i,a] + dvo[a,i]   (symmetrised)
 *
 *   θ = 2 T2 - T2^T
 *   xt1[m,i] = Σ_nef L2[m,n,e,f] · θ[i,n,e,f]
 *   xt2[e,a] = Σ_mnf L2[m,n,a,f] · θ[m,n,e,f]
 *
 * Sub-phase 2 strategies:
 *   - Λ_1 = 0 (closed-form, default): D[ov][i,a] = T1[i,a]. This is the
 *     Sub-step 2X.1/2X.2c baseline path — ~6.3 % off canonical dipole.
 *   - Λ_1 ≠ 0 (Sub-step 2X.3.2+): D[ov][i,a] = T1[i,a] + Λ_1[i,a]. At T1=0
 *     this is just Λ_1; θ·L1 cross terms are deferred to Sub-step 2X.3.5
 *     (kicks in only when T1 itself is non-zero, i.e. after Phase 2.6c).
 *
 * Trace conservation: the ov/vo additions are pure off-diagonal, so the
 * MO trace is unchanged (still N_elec from the HF + oo + vv blocks).
 *
 * @param T1       Per-LMO T1 amplitudes in pair (i,i)'s semi-canonical PAO
 *                 basis (length setups[pair_lookup[i*nocc+i]].n_pao). Pass
 *                 empty / size 0 vectors to skip the T1 contribution.
 * @param Lambda1  Same shape as T1; per-LMO Λ_1 amplitudes in pair (i,i)'s
 *                 PAO basis. Default empty → Λ_1 = 0 (closed-form Sub-step
 *                 2X.1 path preserved). When supplied with non-empty entries
 *                 the leading L1 contribution is added to D[ov]/D[vo] (Sub-
 *                 step 2X.3.4 — closes most of the closed-form dipole gap).
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
    real_t*                               D_mo_out,
    const std::vector<std::vector<real_t>>& Lambda1 = {});

} // namespace gansu
