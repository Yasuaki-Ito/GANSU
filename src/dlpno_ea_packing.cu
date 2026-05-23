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
 * @file dlpno_ea_packing.cu
 * @brief Packed-vector layout builder for DLPNO-EA-EOM-CCSD (stage B).
 *        See dlpno_ea_packing.hpp for the layout contract.
 */

#include "dlpno_ea_packing.hpp"

#include "dlpno_pair_data.hpp"   // PairSetup, PairData

namespace gansu {

DLPNOEAPacking build_ea_packing(const DLPNOLMP2Result& res) {
    DLPNOEAPacking p;
    p.nocc = res.nocc;
    p.nvir = res.nao - res.nocc;
    p.n_pno_ii.assign(p.nocc, 0);
    p.off_i.assign(p.nocc, -1);

    int off = p.nvir;  // R1 (singles) sector occupies [0, nvir)
    for (int i = 0; i < p.nocc; ++i) {
        const int idx = res.pair_lookup[static_cast<size_t>(i) * p.nocc + i];  // diagonal pair (i,i)
        const int n = res.pairs[idx].n_pno;
        p.n_pno_ii[i] = n;
        p.off_i[i] = off;
        off += n * n;  // per-i 2p1h block is n_pno(ii) × n_pno(ii)
    }
    p.total_dim = off;
    return p;
}

} // namespace gansu
