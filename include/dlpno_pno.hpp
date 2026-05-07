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

#include <vector>
#include "types.hpp"

namespace gansu {

/**
 * @brief Pair Natural Orbital (PNO) construction from MP2 amplitudes.
 *
 * For a pair (i,j) with semi-canonical PAO amplitudes T_{ij}^{ab} the pair
 * density is (Riplinger & Neese 2013, JCP 138, 034106, Eq. 8)
 *
 *   D_{ij} = (1 + δ_{ij})^{-1} ( T̃_{ij} · T_{ij}^T + T̃_{ij}^T · T_{ij} ),
 *   T̃_{ij}^{ab} = 2 T_{ij}^{ab} − T_{ij}^{ba}.
 *
 * D_{ij} is symmetric and positive semi-definite. Diagonalising it yields
 * occupation numbers n_{ã}^{ij} ≥ 0 and the PNO transformation matrix d_{aã}.
 * PNOs with occupation below `t_cut_pno` are dropped.
 *
 * Result members are row-major:
 *   - d_pno          [n_pao × n_kept]  PNO transformation in semi-canonical PAO basis
 *   - occupations    [n_pao]           sorted descending; first `n_kept` entries kept
 *   - n_kept         number of retained PNOs
 *   - sum_occupations Σ_a n_a (over kept PNOs) — diagnostic
 */
struct PNOResult {
    std::vector<real_t> d_pno;
    std::vector<real_t> occupations;
    int n_kept = 0;
    real_t sum_occupations = 0.0;
};

/**
 * @brief Diagonalise the pair density matrix and return the truncated PNO set.
 *
 * Two density forms are supported:
 *
 *   - Full LMP2 density (Riplinger 2013, Eq. 8):
 *       D = (1+δ_{ij})^{-1} (T̃^T T + T T̃^T),  T̃ = 2T - T^T
 *
 *   - OS-only density (Pinski 2015, JCP 144, 094111 / ORCA default):
 *       D^OS = (1+δ_{ij})^{-1} (T^T T + T T^T)
 *
 *   The OS-only form drops the same-spin antisymmetrisation; it leaves PNO
 *   occupation more compactly distributed so the same TCutPNO retains more
 *   correlation. Recommended (and ORCA's default) for DLPNO-MP2.
 *
 * @param T_amp     [n_pao × n_pao] row-major MP2 amplitudes T_{ij}^{ab}
 *                  in the *semi-canonical PAO* basis of the pair.
 * @param i_eq_j    true when the pair is diagonal (i = j) → factor 1/2 applied.
 * @param n_pao     PAO domain size for this pair.
 * @param t_cut_pno PNO occupation cutoff.
 * @param os_only   If true, use OS-only density form for the PNO selection.
 *                  Default false: the full LMP2 density better matches a
 *                  closed-shell MP2 energy evaluation (T̃^T T + T̃ T^T).
 *                  Set true only when pairing with SOS-MP2-style energies.
 */
PNOResult build_pno_from_T(
    const real_t* T_amp,
    bool i_eq_j,
    int n_pao,
    real_t t_cut_pno,
    bool os_only = false);

} // namespace gansu
