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
 * @file dlpno_ip_packing.cu
 * @brief Packed-vector layout builder for DLPNO-IP-EOM-CCSD (stage B, Phase B0).
 *        See dlpno_ip_packing.hpp for the layout contract.
 */

#include "dlpno_ip_packing.hpp"

#include "dlpno_pair_data.hpp"   // PairSetup, PairData

namespace gansu {

DLPNOIPPacking build_ip_packing(const DLPNOLMP2Result& res) {
    DLPNOIPPacking p;
    p.nocc = res.nocc;
    const int n_pairs = static_cast<int>(res.pairs.size());
    p.n_pno.assign(n_pairs, 0);
    p.off_ij.assign(n_pairs, -1);
    p.off_ji.assign(n_pairs, -1);
    p.is_diagonal.assign(n_pairs, 0);

    int off = p.nocc;  // R1 sector occupies [0, nocc)
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n     = res.pairs[idx].n_pno;
        const bool diag = (res.setups[idx].i == res.setups[idx].j);
        p.n_pno[idx]       = n;
        p.is_diagonal[idx] = diag ? 1 : 0;
        // (i,j) orientation block (always present, even if n==0 → empty block).
        p.off_ij[idx] = off;
        off += n;
        // (j,i) orientation block (off-diagonal pairs only — independent amplitude).
        if (!diag) {
            p.off_ji[idx] = off;
            off += n;
        }
    }
    p.total_dim = off;
    return p;
}

} // namespace gansu
