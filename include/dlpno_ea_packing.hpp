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
 * @file dlpno_ea_packing.hpp
 * @brief Packed-vector layout for the DLPNO-EA-EOM-CCSD Davidson
 *        (bt-PNO-STEOM stage B, Dutta et al. 2018).
 *
 * EA-EOM amplitude: R = Σ_a r_a a†_a + ½ Σ_{abi} r^{ab}_i a†_a a†_b a_i.
 * In the DLPNO scheme the 2p1h amplitude r^{ab}_i keeps BOTH virtuals in the
 * GS-DLPNO-CCSD PNO basis of the diagonal pair (i,i): a,b ∈ PNO(i,i)
 * (STEOM.md §13.4). The Davidson vector is packed as
 *
 *     [ R1 (nvir) | per-i R2 blocks of n_pno(ii)² ]
 *
 * The 1p (singles) sector keeps the full canonical virtual (nvir) — virtual
 * singles are not domain-truncated. Each occupied i contributes an
 * n_pno(ii) × n_pno(ii) block (row-major a',b'). This mirrors the canonical
 * P2 EAEOMCCSDOperator FULL (a,b) layout so the no-truncation gate is a
 * literal index permutation.
 */

#pragma once

#include <vector>

#include "types.hpp"
#include "dlpno_mp2.hpp"   // DLPNOLMP2Result

namespace gansu {

/// Offset table for the packed DLPNO-EA Davidson vector. R1 occupies
/// [0, nvir); per-i 2p1h blocks follow.
struct DLPNOEAPacking {
    int nvir = 0;               ///< 1p sector size (full canonical virtual)
    int nocc = 0;               ///< active occupied
    int total_dim = 0;          ///< nvir + Σ_i n_pno(ii)²
    std::vector<int> n_pno_ii;  ///< [nocc] PNO count of the diagonal pair (i,i)
    std::vector<int> off_i;     ///< [nocc] packed offset of i's R2 block (≥ nvir)
};

/// Build the EA packed-vector offset table from a converged DLPNO result.
/// nvir is taken as res.nao - res.nocc (no frozen core).
DLPNOEAPacking build_ea_packing(const DLPNOLMP2Result& res);

} // namespace gansu
