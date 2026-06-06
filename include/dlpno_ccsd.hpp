/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "types.hpp"
#include "dlpno_params.hpp"
#include "dlpno_mp2.hpp"   // DLPNOLMP2Result (value member for the (T) re-use)

namespace gansu {

class RHF;
class ERI;

/**
 * @brief DLPNO-CCSD driver (Phase 2 — under construction).
 *
 * Builds on the Phase-1 LMP2 infrastructure (LMO + PAO + Boughton-Pulay
 * domains + per-pair PNO basis) and replaces the LMP2 amplitude solver
 * with a CCSD residual loop carrying both T1 (per-LMO PAO) and T2
 * (per-pair PNO) amplitudes plus their inter-pair couplings via the
 * pair-pair PNO overlap bar_S^{(ij,kj)}.
 *
 * Sub-phase status:
 *   2.0  plumbing skeleton                                         [done]
 *   2.1  strong/weak pair classification (TCutPairs)               [done]
 *   2.2  T1 amplitudes + T1 residual                               [done]
 *   2.3.1  T2 residual machinery (dressing off, sanity loop)       [done]
 *   2.3.2  intra-pair F_eff particle dressing                      [done]
 *   2.3.3  intra-pair hole dressing on diagonal pairs              [done]
 *   2.4.1  hole F_eff[k,i] dressing for k≠i (l=i restriction)      [done]
 *   2.4.2  cross-pair particle F_ac dressing (barS projection)     [done]
 *   2.4.3  hole F_eff[k,i] cross-pair (l≠i) contributions          [done]
 *   2.5  4-virtual ladder W_abcd                                   [done]
 *   2.6  oooo + particle-hole ladder W_akic/W_akci                 [done]
 *   2.6b T2 ring-diagram dressing of W_klij / W_akic / W_akci      [this commit]
 *        ΔW_klij = + Σ_{cd} (kl|cd) t_{ij}^{cd}                      (oooo)
 *        ΔW_akic = -½ Σ_{ld} (ld|kc) π_{il}[d,a] + ½ T_pair[k,c,l,d] π_{il}[a,d]
 *        ΔW_akci = -½ Σ_{ld} (lc|kd) π_{il}[d,a]
 *   2.7  DIIS (Direct Inversion in the Iterative Subspace)         [done]
 *   3.0  DLPNO-CCSD(T) skeleton (CCSD + (T)=0 placeholder)         [done]
 *   3.1  triple iteration + PNO-union TNO size statistics           [done]
 *   3.2  (T) energy via canonical-style contractions in TNO basis   [TODO]
 *   3.3  multi-GPU triple parallel framework (per-GPU OpenMP slabs) [this commit]
 *   3.4  validation against canonical RI-CCSD(T) under strict mode  [TODO]
 *   2.6  remaining residual terms (oooo, ovov, T1 dressing)        [TODO]
 *   2.7  DIIS + energy + validation against canonical RI-CCSD      [TODO]
 *
 * Reference: Riplinger, Neese 2013 (J. Chem. Phys. 138, 034106);
 *            Riplinger, Pinski, Becker, Valeev, Neese 2016 (JCP 144, 024109).
 */
class DLPNOCCSD {
public:
    DLPNOCCSD(RHF& rhf, const ERI& eri, DLPNOParams params);

    /// Run the full DLPNO-CCSD calculation. Returns the correlation energy.
    /// Throws gansu::Exception while the residual is under construction
    /// (sub-phases 2.1 onwards).
    real_t compute_energy();

    /// DLPNO-CCSD(T) re-use hook: when capture_lmp2_ is set before
    /// compute_energy(), the converged LMP2 pair state (amplitudes BEFORE the
    /// CCSD dressing) is snapshotted into lmp2_snapshot_ so the (T) driver can
    /// reuse it instead of re-solving LMP2 from scratch. Bit-exact w.r.t. the
    /// re-solve (same thresholds; only verbose differs, which is numerics-inert).
    bool             capture_lmp2_ = false;
    DLPNOLMP2Result  lmp2_snapshot_;

private:
    RHF& rhf_;
    const ERI& eri_;
    DLPNOParams params_;
    int nao_ = 0;
    int nocc_ = 0;
};

} // namespace gansu
