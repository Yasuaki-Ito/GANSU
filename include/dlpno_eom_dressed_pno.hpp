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
 * @file dlpno_eom_dressed_pno.hpp
 * @brief Per-pair PNO-basis intermediates for the native DLPNO-IP/EA-EOM σ
 *        operators — bt-PNO-STEOM stage B-a.6 / B-EA.6, the *true-scaling* path
 *        (Dutta-Saitow-Riplinger-Neese-Izsák 2018 JCP 148, 244101).
 *
 * The B-a.0..B-a.5 native operators are bit-exact but, at the gate scale, borrow
 * the *dense* canonical-virtual intermediates (Lvv[nvir²], Wovvo/Wovov[nocc·nvir²
 * ·nocc], …) from the canonical operator and contract them through an [nvir]
 * accumulator each matvec. That dense surface is infeasible at 100 atoms
 * (Wovvo/Wovov ~180 TB). This module replaces those borrows with intermediates
 * living directly in each pair's PNO basis [n_pno × n_pno], so the per-pair σ2
 * never touches the canonical nvir surface.
 *
 * Build-up (each gated; see §gate below):
 *   B-a.6c (a) [this commit]: Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij)  — the simplest term.
 *       Lvv is occ-free and (in the DLPNO T1≈0 limit) carries NO T2-ring
 *       dressing, so this is a pure congruence of the dense (already-dressed)
 *       Lvv onto the pair PNO. It is U_loc-independent. Because the own-pair
 *       lift r2c = U^(ij)·r2_packed is exact at ANY truncation, the dressed T2
 *       (Lvv^(ij)·r2_packed) is *algebraically identical* to the dense-borrow T2
 *       (U^(ij)ᵀ·Lvv·U^(ij)·r2_packed) — it cannot change numerics, which makes
 *       it the safe step that de-risks the module / env-flag / gate plumbing
 *       before the genuinely truncation-sensitive ring-dressed terms.
 *   B-a.6c (b) [next]: Wovvo^(ij) / Wovov^(ij) — the per-pair-PNO ph-ladder with
 *       the ring T2-dressing (canonical formulas ip_eom_ccsd_operator.cu:658-701:
 *       Wovov ⊃ -Σ_{c,l} (kc|ld) t2[i,l,c,b]; Wovvo ⊃ 2(kc|ld)t2[i,l,a,d] - …).
 *       The bare ovvo/ovov blocks come from Phase24Integrals (W_ovvo_i/j,
 *       W_ovov_i/j, V_ovov_pair); t2 in pair basis = barS-projected Y. This is
 *       the delicate part where truncated W^(ij) ≠ dense (it pulls in OTHER
 *       pairs' amplitudes through barS), so it is the first term whose gate
 *       must show monotonic t_cut_pno→0 convergence, not bit-exactness.
 *   B-EA.6: EA mirror — Wvvvv^(ii), Wvovv/Wvvvo per-pair (symmetric).
 *
 * §gate (Dutta 2018 methodology):
 *   Tier 1 (full PNO, t_cut_pno=0): U^(ij) is square+orthogonal so the per-pair
 *       intermediate equals the dense congruence exactly → the dressed-native
 *       full σ must match the (b) project-up reference (and the dense-borrow
 *       native) to ~1e-12. Toggled via GANSU_DLPNO_NATIVE_DRESSED=1 on top of the
 *       existing GANSU_DLPNO_IP_NATIVE_VALIDATE harness (no new gate code).
 *   Tier 2 (truncated preset): IP/EA root MAE ≤ 0.08 eV vs canonical AND
 *       monotonic convergence as t_cut_pno→0 (the decisive sign the ring
 *       dressing is correct — a bug fails to converge to canonical at full PNO).
 */

#pragma once

#include <vector>

#include "types.hpp"

namespace gansu {

struct DLPNOLMP2Result;  // dlpno_mp2.hpp — holds per-pair Y/bar_Q/setups + phase24

/**
 * @brief Per-pair PNO-basis IP-EOM intermediates (true-scaling native σ).
 *
 * Indexed by DLPNO pair index. Each matrix lives in pair (i,j)'s PNO basis
 * [n_pno × n_pno], row-major. Empty (size-0) entries mark screened pairs
 * (n_pno == 0).
 */
struct DressedPnoIP {
    /// Lvv^(ij)[a',d'] = (U^(ij)ᵀ Lvv U^(ij))[a',d']  [n_pno²], row-major.
    /// Occ-free, U_loc-independent; no ring dressing (Lvv carries none).
    std::vector<std::vector<real_t>> Lvv_pno;   ///< [n_pairs] of [n_pno²]

    /// ph-ladder (T6/T7) per-pair PNO intermediates. Two "roles" per pair: the
    /// fixed occupied index I is the pair's i (occi) or j (occj) — the σ2 ph-
    /// ladder of orientation (oi,oj) needs occ=oj (T6 + T7-first) and occ=oi
    /// (T7-second). The contracted/result virtuals a',d' both live in PNO(ij).
    ///   Wovvo_pno[m,a',d'] (occ=I)  layout (m·n_pno + a')·n_pno + d'
    ///   Wovov_pno[m,a',d'] (occ=I)  same layout
    /// m runs over LMOs (n_lmo = nocc). B-a.6c(b1): seeded by congruence of the
    /// dense (already-dressed, ring-included) Wovvo/Wovov from the canonical op,
    ///   Wovvo_pno[m,a',d'] = Σ_{a,d} U^(ij)[a,a'] Wovvo[m,a,d,I] U^(ij)[d,d'],
    /// which at full PNO reproduces the dense ph-ladder bit-for-bit (U·Uᵀ=I).
    /// B-a.6c(b2) replaces the ring part with a native build (Phase24 V_ovov_pair
    /// + two-sided barS-projected Y), so truncated W^(ij) ≠ dense (true scaling).
    std::vector<std::vector<real_t>> Wovvo_pno_occi;  ///< [n_pairs] of [n_lmo·n_pno²]
    std::vector<std::vector<real_t>> Wovvo_pno_occj;
    std::vector<std::vector<real_t>> Wovov_pno_occi;
    std::vector<std::vector<real_t>> Wovov_pno_occj;
};

/**
 * @brief Build the per-pair PNO IP intermediates by congruence of the dense
 *        canonical-virtual intermediates onto each pair's PNO basis.
 *
 * @param h_Lvv  Dense canonical Lvv [nvir²], row-major (already T1/T2-dressed;
 *               borrowed bit-identically from the canonical operator).
 * @param Uall   Per-pair virtual transform U^(ij) = C_virᵀ S bar_Q_ij, each a
 *               flat row-major [nvir × n_pno] vector (empty if n_pno == 0).
 *               Length = n_pairs.
 * @param n_pno  Per-pair PNO counts (length n_pairs).
 * @param nvir   Number of canonical virtuals.
 * @return DressedPnoIP with Lvv_pno populated for every non-screened pair.
 */
DressedPnoIP build_dressed_pno_ip(const std::vector<real_t>& h_Lvv,
                                  const std::vector<std::vector<real_t>>& Uall,
                                  const std::vector<int>& n_pno,
                                  int nvir);

/**
 * @brief B-a.6c(b1): build the ph-ladder per-pair PNO intermediates
 *        (Wovvo_pno_occ{i,j}, Wovov_pno_occ{i,j}) by congruence of the dense,
 *        already-dressed canonical Wovvo/Wovov onto each pair's PNO basis.
 *
 * This validates the PNO-space T6/T7 contraction machinery + one-sided barS r2
 * projection (it is bit-exact vs the dense ph-ladder at full PNO). B-a.6c(b2)
 * will replace the ring part with a native Phase24 + two-sided-barS build.
 *
 * @param io            DressedPnoIP to populate (Lvv_pno already built).
 * @param h_Wovvo_lmo   dense Wovvo, occ indices already U_loc-rotated → LMO,
 *                      layout ((m·nvir+a)·nvir+d)·nocc + I  [nocc·nvir²·nocc].
 * @param h_Wovov_lmo   dense Wovov, occ→LMO, layout ((m·nvir+a)·nocc+I)·nvir+d
 *                      [nocc·nvir·nocc·nvir].
 * @param Uall          per-pair U^(ij) flat row-major [nvir × n_pno] (empty if 0).
 * @param pair_i,pair_j per-pair occupied LMO indices (res.setups[idx].i / .j).
 * @param n_pno         per-pair PNO counts.
 * @param nocc,nvir     dimensions (n_lmo = nocc).
 */
void build_dressed_pno_ip_phladder(DressedPnoIP& io,
                                   const std::vector<real_t>& h_Wovvo_lmo,
                                   const std::vector<real_t>& h_Wovov_lmo,
                                   const std::vector<std::vector<real_t>>& Uall,
                                   const std::vector<int>& pair_i,
                                   const std::vector<int>& pair_j,
                                   const std::vector<int>& n_pno,
                                   int nocc, int nvir);

/**
 * @brief B-a.6c IP dense-free bare seed: set the per-pair PNO Wovvo_pno/Wovov_pno
 *        to the bare ph-ladder ERI blocks straight from Phase24 (occ-role i/j),
 *        WITHOUT ever materialising the dense nocc²·nvir² Wovvo/Wovov.
 *
 * The bare terms of the canonical ph-ladder (ip_eom_ccsd_operator.cu:688/:663)
 *   bare Wovvo_pno[m,a',d'](I) = (m d'|a' I),  bare Wovov_pno[m,a',d'](I) = (m I|a' d')
 * in the pair PNO basis ARE the congruence U^(ij)ᵀ⊗2 of the canonical bare ERIs —
 * and the PNO ERI is exactly that congruence, so Phase24's directly-extracted
 * W_ovvo_bare_{i,j}/W_oovv_bare_{i,j} equal build_dressed_pno_ip_phladder's bare
 * contribution bit-for-bit, at NO dense cost. The T2 ring is added on top by
 * build_dressed_pno_ip_ring with subtract_dense=false (native-only); the (small,
 * T1≈0) W2 ph-ladder T1 terms (Wooov·t1, ovvv·t1) are deferred. This is the EA
 * B-EA.6e analog (build_dressed_pno_ea_vvvv_bare) for the IP ph-ladder.
 *
 * @param io        DressedPnoIP to populate (Lvv_pno already built). The four
 *                  Wovvo_pno/Wovov_pno occ{i,j} vectors are (re)assigned here.
 * @param res       DLPNO result (phase24.W_ovvo_bare_{i,j} / W_oovv_bare_{i,j}).
 * @param n_pno     per-pair PNO counts (length n_pairs).
 * @param nocc      number of occupied LMOs (n_lmo; the m index range / bare block
 *                  leading dimension).
 */
void build_dressed_pno_ip_bare(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<int>& n_pno,
                               int nocc);

/**
 * @brief B-a.6c(b2): replace the T2-ring part of the congruence-seeded ph-ladder
 *        per-pair PNO W (built by build_dressed_pno_ip_phladder) with a NATIVE
 *        build from Phase24 V_ovov_pair + two-sided barS-projected amplitudes —
 *        the true-scaling step that never materialises the dense ring.
 *
 * For each pair (i,j) and fixed occupied I ∈ {i,j}, the canonical ring
 * (ip_eom_ccsd_operator.cu:658-701, the Σ_{l,d} terms) is built directly in the
 * pair PNO basis from
 *   - V_ovov_pair^{(ij)}[A,B,P,Q] = (A P | B Q)  (the bare ovov ERI, PNO basis), and
 *   - t2_proj[p,q] = oriented(barS^{(ij,src)} · Y_src · barS^{(ij,src)ᵀ}), the same
 *     two-sided barS projection the CCSD T2 sweep uses (dlpno_pair_data.cu:411-414).
 * The dense ring is ALSO built (canonical formula, from h_ovov_lmo + h_t2_lmo) and
 * congruenced (U^(ij)ᵀ · ring · U^(ij)); the update is
 *   Wovvo_pno += native_ring − congruence(dense_ring),
 * so the bare+T1 part stays the b1-validated congruence seed and only the ring
 * becomes truncation-sensitive (at full PNO native_ring == congruence(dense_ring)
 * so this reduces to b1 exactly).
 *
 * @param io          DressedPnoIP with Lvv_pno + ph-ladder congruence seed already built.
 * @param res         DLPNO result: per-pair bar_Q / Y / setups / pair_lookup / phase24.
 * @param h_ovov_lmo  raw (ov|ov), occ indices rotated canonical→LMO, layout
 *                    ((p·nvir+a)·nocc+q)·nvir+b for (pa|qb) (copy for localizer none).
 * @param h_t2_lmo    CCSD T2, 2 leading occ rotated→LMO, layout ((p·nocc+q)·nvir+a)·nvir+b.
 * @param Uall        per-pair U^(ij) = C_virᵀ S bar_Q_ij flat [nvir × n_pno] (empty if 0).
 * @param h_S         [nao²] AO overlap (for barS = bar_Q_ijᵀ S bar_Q_src).
 * @param n_pno       per-pair PNO counts.
 * @param nao,nocc,nvir dimensions.
 * @param[out] max_delta   max |native_ring − congruence(dense_ring)| over all entries
 *                         (the truncation correction magnitude; →0 at full PNO).
 *                         When subtract_dense=false, set to 0 (no dense reference).
 * @param[out] max_ring    max |congruence(dense_ring)| (reference scale for max_delta),
 *                         or max|native_ring| when subtract_dense=false.
 * @param subtract_dense   true (default, B-a.6c(b2) validate): build the dense ring
 *                         DR [nocc²·nvir²], congruence it, and apply native_ring −
 *                         cong(DR) on top of the congruence-seed ph-ladder W (gate
 *                         vs the dense ph-ladder). false (IP dense-free true scaling):
 *                         apply native_ring alone on top of a Phase24 bare seed — NO
 *                         dense DR is built, so h_ovov_lmo / h_t2_lmo are unused
 *                         (pass empty). EA B-EA.6e analog.
 */
void build_dressed_pno_ip_ring(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<real_t>& h_ovov_lmo,
                               const std::vector<real_t>& h_t2_lmo,
                               const std::vector<std::vector<real_t>>& Uall,
                               const std::vector<real_t>& h_S,
                               const std::vector<int>& n_pno,
                               int nao, int nocc, int nvir,
                               real_t& max_delta, real_t& max_ring,
                               bool subtract_dense = true);

// ===========================================================================
//  EA mirror (B-EA.6d): per-pair PNO Wvvvv for the native DLPNO-EA-EOM σ.
// ===========================================================================

/**
 * @brief Per-pair PNO-basis EA-EOM intermediate (true-scaling native σ).
 *
 * The EA 2p1h amplitude r^{ab}_j keeps BOTH virtuals in the diagonal-pair PNO
 * basis a,b ∈ PNO(j,j), so the only 4-virtual σ2 term — T_vvvv,
 *   σ2_packed^(jj)[a',b'] += Σ_{c',d'} Wvvvv^(jj)[a',b',c',d'] r2_packed^(jj)[c',d'],
 * needs Wvvvv directly in each output occupied j's diagonal PNO basis. Indexed
 * by occupied LMO j (0..nocc-1); empty (size-0) marks a screened occupied.
 */
struct DressedPnoEA {
    /// Wvvvv^(jj)[a',b',c',d'] [n_pno(jj)⁴], row-major ((a'·n+b')·n+c')·n+d'.
    std::vector<std::vector<real_t>> Wvvvv_pno;   ///< [nocc] of [n_pno(jj)⁴]
};

/**
 * @brief Build the per-occ diagonal-PNO Wvvvv by congruence of the dense
 *        canonical-virtual Wvvvv onto each occupied j's PNO(j,j) basis:
 *        Wvvvv^(jj)[a',b',c',d'] = Σ_{abcd} U^(jj)[a,a']U[b,b']U[c,c']U[d,d'] Wvvvv[a,b,c,d].
 *
 * This is the EA analog of build_dressed_pno_ip (the simplest, numerics-
 * preserving step): because the own-pair lift r2c = U^(jj)·r2_packed is exact at
 * ANY truncation, the PNO-space T_vvvv (Wvvvv^(jj)·r2_packed) reproduces the
 * dense-borrow T_vvvv (U^(jj)ᵀ⊗2 Σ Wvvvv·r2c U^(jj)) bit-for-bit — it de-risks
 * the module/env-flag/σ-wiring before the truncation-sensitive ring build.
 *
 * @param h_Wvvvv  dense canonical Wvvvv [nvir⁴], row-major ((a·nvir+b)·nvir+c)·nvir+d
 *                 (already T1/T2-dressed; borrowed bit-identically from ea_op).
 * @param Uocc     per-occ U^(jj) = C_virᵀ S bar_Q_jj, flat row-major [nvir × n_pno(jj)]
 *                 (empty if n_pno(jj) == 0). Length = nocc.
 * @param n_pno_ii per-occ diagonal PNO counts (length nocc).
 * @param nvir     number of canonical virtuals.
 */
DressedPnoEA build_dressed_pno_ea_vvvv(const std::vector<real_t>& h_Wvvvv,
                                       const std::vector<std::vector<real_t>>& Uocc,
                                       const std::vector<int>& n_pno_ii,
                                       int nvir);

/**
 * @brief Host 4-index congruence W_pno[a'b'c'd'] = Σ_{abcd} U[a,a']U[b,b']U[c,c']U[d,d']
 *        W[abcd] for one occ (four sequential single-index transforms). The reference for
 *        the GPU port (DLPNOEAEOMNativeOperator::build_dressed_vvvv_gpu) — exposed so the
 *        ctor gpu_selfcheck_ gate can compare one occ block.
 * @param W     dense canonical Wvvvv [nvir⁴], row-major ((a·nv+b)·nv+c)·nv+d.
 * @param Uflat U^(jj) flat row-major [nv × n].
 * @param out   filled with [n⁴], row-major ((a'·n+b')·n+c')·n+d'.
 */
void congruence4(const std::vector<real_t>& W, const std::vector<real_t>& Uflat,
                 int nv, int n, std::vector<real_t>& out);

/**
 * @brief B-EA.6e dense-free seed: set Wvvvv^(jj) to the bare PNO 4-virtual ERI
 *        block straight from Phase24 (W_pair^{(jj)}[a,b,c,d] = (ac|bd) in PNO),
 *        WITHOUT ever materialising the dense nvir⁴ Wvvvv.
 *
 * The bare term of the canonical Wvvvv (ea_eom_ccsd_operator.cu:748, (ac|bd))
 * in the diagonal pair's PNO basis IS the congruence U^(jj)ᵀ⊗4 (ac|bd) — but the
 * PNO ERI is exactly that congruence, so Phase24's directly-extracted W_pair
 * equals build_dressed_pno_ea_vvvv's bare contribution bit-for-bit, at NO nvir⁴
 * cost. The T2 ring is added on top by build_dressed_pno_ea_vvvv_ring with
 * subtract_dense=false (native-only); the (small, T1≈0) Wvvvv T1 terms are
 * deferred. This is the true-scaling Wvvvv path for 100-atom EA-EOM.
 *
 * @param res       DLPNO result (phase24.W_pair + pair_lookup).
 * @param n_pno_ii  per-occ diagonal PNO counts (length nocc).
 * @return DressedPnoEA with Wvvvv_pno[j] = W_pair^{(jj)} for every active occ j
 *         (empty where n_pno(jj)==0 or no Phase24 W_pair for the diagonal pair).
 */
DressedPnoEA build_dressed_pno_ea_vvvv_bare(const DLPNOLMP2Result& res,
                                            const std::vector<int>& n_pno_ii);

/**
 * @brief B-EA.6d: replace the T2-ring part of the congruence-seeded Wvvvv^(jj)
 *        (built by build_dressed_pno_ea_vvvv) with a NATIVE build from Phase24
 *        V_ovov_pair + two-sided barS-projected amplitudes — the true-scaling
 *        step that never materialises the dense ring.
 *
 * The canonical Wvvvv ring (ea_eom_ccsd_operator.cu:736) is the single t2 term
 *   ΔWvvvv[a,b,c,d] = Σ_{k,l} (kc|ld) t2[k,l,a,b].
 * In the output pair (j,j)'s PNO basis it is built directly from
 *   - V_ovov_pair^{(jj)}[k,l,c',d'] = (k c' | l d')  (bare ovov ERI, PNO basis), and
 *   - t2_proj^{(jj←kl)}[a',b'] = oriented(barS^{(jj,src)} Y_src barS^{(jj,src)ᵀ}),
 *     src = pair_lookup[k,l], two-sided barS as the CCSD T2 sweep
 *     (dlpno_pair_data.cu:411-414):
 *   native_ring^(jj)[a',b',c',d'] = Σ_{k,l} V^{(jj)}[k,l,c',d'] · t2_proj^{(jj←kl)}[a',b'].
 * The dense ring is ALSO built (canonical formula, from h_ovov_lmo + h_t2_lmo)
 * and congruenced; the update is Wvvvv_pno += native_ring − congruence(dense_ring),
 * so the bare+T1 part stays the validated congruence seed and only the ring
 * becomes truncation-sensitive (at full PNO native_ring == congruence(dense_ring),
 * reducing to the seed exactly).
 *
 * @param io          DressedPnoEA with Wvvvv_pno congruence seed already built.
 * @param res         DLPNO result: per-pair bar_Q / Y / setups / pair_lookup / phase24.
 * @param h_ovov_lmo  raw (ov|ov), BOTH occ indices rotated canonical→LMO, layout
 *                    ((k·nvir+c)·nocc+l)·nvir+d for (kc|ld) (copy for localizer none).
 * @param h_t2_lmo    CCSD T2, both leading occ rotated→LMO, layout ((k·nocc+l)·nvir+a)·nvir+b.
 * @param Uocc        per-occ U^(jj) = C_virᵀ S bar_Q_jj flat [nvir × n_pno(jj)] (empty if 0).
 * @param h_S         [nao²] AO overlap (for barS = bar_Q_jjᵀ S bar_Q_src).
 * @param n_pno_ii    per-occ diagonal PNO counts.
 * @param nao,nocc,nvir dimensions.
 * @param[out] max_delta max |native_ring − congruence(dense_ring)| (→0 at full PNO).
 *                       When subtract_dense=false, set to 0 (no dense reference).
 * @param[out] max_ring  max |congruence(dense_ring)|, or max|native_ring| when
 *                       subtract_dense=false (reference scale for max_delta).
 * @param subtract_dense true (default, B-EA.6d validate): build the dense ring DR
 *                       [nvir⁴], congruence it, and apply native_ring − cong(DR)
 *                       on top of a cong(dense Wvvvv) seed (gate vs the (b)
 *                       reference). false (B-EA.6e true scaling): apply native_ring
 *                       alone on top of a W_pair bare seed — NO dense DR is built,
 *                       so h_ovov_lmo / h_t2_lmo are unused (pass empty).
 */
void build_dressed_pno_ea_vvvv_ring(DressedPnoEA& io,
                                    const DLPNOLMP2Result& res,
                                    const std::vector<real_t>& h_ovov_lmo,
                                    const std::vector<real_t>& h_t2_lmo,
                                    const std::vector<std::vector<real_t>>& Uocc,
                                    const std::vector<real_t>& h_S,
                                    const std::vector<int>& n_pno_ii,
                                    int nao, int nocc, int nvir,
                                    real_t& max_delta, real_t& max_ring,
                                    bool subtract_dense = true);

} // namespace gansu
