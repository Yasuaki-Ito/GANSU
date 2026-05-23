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
 * @file dlpno_ip_packing.hpp
 * @brief Packed-vector layout for the DLPNO-IP-EOM-CCSD Davidson
 *        (bt-PNO-STEOM stage B, Phase B0, Dutta-Saitow-Riplinger-Neese-Izsák
 *        2018 JCP 148, 244101).
 *
 * The IP-EOM amplitude is R = Σ_i r_i a_i + ½ Σ_{ija} r^a_{ij} a†_a a_j a_i.
 * In the DLPNO scheme the 2h1p amplitude r^a_{ij} keeps the virtual a in the
 * GS-DLPNO-CCSD PNO basis of pair (i,j) (the GS-PNO span suffices — STEOM.md
 * §13.8). The Davidson vector is therefore packed as
 *
 *     [ R1 (nocc) | per-pair R2 blocks ]
 *
 * Pairs are stored for i ≤ j (DLPNO convention, pair_lookup[i*nocc+j]). For
 * i < j the two orientations r^a_{ij} and r^a_{ji} are INDEPENDENT amplitudes
 * (unlike the CCSD T2 where Y_ji = Y_ij^T), but share the same PNO(ij)=PNO(ji)
 * space, so each gets its own length-n_pno block. Diagonal pairs (i==j) hold a
 * single block. This mirrors the FULL (i,j) layout of the canonical P1
 * IPEOMCCSDOperator so the no-truncation gate is a literal index permutation.
 */

#pragma once

#include <vector>

#include "types.hpp"
#include "dlpno_mp2.hpp"   // DLPNOLMP2Result

namespace gansu {

/// Offset table mapping (pair, orientation) → start index in the packed
/// Davidson vector. The R1 sector occupies [0, nocc); R2 blocks follow.
struct DLPNOIPPacking {
    int nocc = 0;                  ///< 1h sector size (active occupied)
    int total_dim = 0;             ///< nocc + Σ_pairs (1 or 2)·n_pno_idx
    std::vector<int> n_pno;        ///< [n_pairs] PNO count per stored pair idx
    std::vector<int> off_ij;       ///< [n_pairs] packed offset of the (i,j) R2 block (≥ nocc)
    std::vector<int> off_ji;       ///< [n_pairs] packed offset of the (j,i) R2 block; -1 for diagonal pairs
    std::vector<char> is_diagonal; ///< [n_pairs] 1 if i==j (single block) else 0

    /// True if stored pair idx is diagonal (i==j) → only the (i,j) block exists.
    bool diagonal(int idx) const { return is_diagonal[idx] != 0; }
};

/// Build the packed-vector offset table from a converged DLPNO result.
/// Walks res.pairs in storage order; the (i,j) block precedes the (j,i) block
/// for off-diagonal pairs. total_dim is the full Davidson vector length.
DLPNOIPPacking build_ip_packing(const DLPNOLMP2Result& res);

} // namespace gansu
