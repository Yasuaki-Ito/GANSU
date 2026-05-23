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

#include <cstddef>
#include <cuda_runtime.h>

#include "types.hpp"

namespace gansu {

/**
 * @brief Step S7b — GPU extraction of the 14 W/T/V blocks from the full
 *        per-pair MO ERI tensor inside precompute_phase24_integrals.
 *
 * Previous flow (S7a):
 *   1. build_mo_eri_into  →  d_eri_ws  (n_emb⁴ doubles)
 *   2. cudaMemcpy d_eri_ws → h_eri_ws (pinned, n_emb⁴ doubles)
 *   3. 13 host-side strided extraction loops over h_eri_ws
 *
 * S7b flow:
 *   1. build_mo_eri_into  →  d_eri_ws (unchanged)
 *   2. launch_phase24_extract  →  d_packed_out (single device buffer holding
 *      all 14 blocks back-to-back at known offsets)
 *   3. cudaMemcpy d_packed_out → h_packed (pinned, layout.total doubles)
 *   4. 14 small memcpy calls from h_packed to the Phase24Integrals
 *      std::vector<real_t> destinations.
 *
 * The packed total is roughly Σ(block sizes) ≪ n_emb⁴ for cholesterol-class
 * (n_emb ≈ 138 vs Σ ≈ 21M doubles → ~17× smaller D2H).
 *
 * Bit-exactness: every extract is a pure index gather (`out = eri[src]`)
 * except T_pair, which fuses `2·eri[A] − eri[B]`. T_pair uses strict
 * `__dsub_rn(__dmul_rn(2, A), B)` intrinsics to forbid FMA contraction and
 * match the host `2.0*x - y` rounding sequence bit-for-bit.
 */
struct Phase24ExtractLayout {
    std::size_t total = 0;
    std::size_t off_T_pair             = 0;
    std::size_t off_W_pair             = 0;
    std::size_t off_W_oooo             = 0;
    std::size_t off_W_ovov_i           = 0;
    std::size_t off_W_ovov_j           = 0;
    std::size_t off_W_ovvo_i           = 0;
    std::size_t off_W_ovvo_j           = 0;
    std::size_t off_V_ovov             = 0;
    std::size_t off_W_ovvv_diag        = 0;  ///< 0 when !is_diag (skip)
    std::size_t off_W_ovvo_lambda      = 0;
    std::size_t off_W_ovvo_lambda_alt  = 0;
    std::size_t off_W_oovv_lambda      = 0;
    std::size_t off_W_ovoo_lambda      = 0;
    std::size_t off_W_ovoo_lambda_alt  = 0;
    /// B-a.6c IP dense-free bare ph-ladder blocks (native DLPNO-IP-EOM σ).
    /// ovvo_bare[m,a',d'](I) = (m d'|a' I) = eri[m,n_lmo+d',n_lmo+a',I];
    /// oovv_bare[m,a',d'](I) = (m I|a' d') = eri[m,I,n_lmo+a',n_lmo+d'].
    /// Each is the dense-free bare seed of Wovvo_pno / Wovov_pno (occ-role i/j),
    /// layout (m·n_pno + a')·n_pno + d', size n_lmo · n_pno² (= sz_W_ovov).
    std::size_t off_W_ovvo_bare_i      = 0;
    std::size_t off_W_ovvo_bare_j      = 0;
    std::size_t off_W_oovv_bare_i      = 0;
    std::size_t off_W_oovv_bare_j      = 0;

    std::size_t sz_T_pair       = 0;  ///< n_lmo² · n_pno²
    std::size_t sz_W_pair       = 0;  ///< n_pno⁴
    std::size_t sz_W_oooo       = 0;  ///< n_lmo²
    std::size_t sz_W_ovov       = 0;  ///< n_pno · n_lmo · n_pno  (each of i/j)
    std::size_t sz_V_ovov       = 0;  ///< n_lmo² · n_pno²
    std::size_t sz_W_ovvv_diag  = 0;  ///< n_pno³ (0 when !is_diag)
    std::size_t sz_pno2         = 0;  ///< n_pno² (W_ovvo_lambda etc.)
    std::size_t sz_ovoo         = 0;  ///< n_pno · n_lmo (W_ovoo_lambda etc.)
};

/// Compute byte offsets / sizes for the packed output buffer.
Phase24ExtractLayout compute_phase24_extract_layout(
    int n_lmo, int n_pno, bool is_diag);

/// Launch all 14 extract kernels on `stream`. `d_packed_out` must be at
/// least `layout.total * sizeof(real_t)` bytes. Synchronisation is the
/// caller's responsibility (downstream cudaMemcpy on the same stream is
/// sufficient).
void launch_phase24_extract(
    const real_t*               d_eri_mo,
    real_t*                     d_packed_out,
    const Phase24ExtractLayout& layout,
    int                         n_emb,
    int                         n_lmo,
    int                         n_pno,
    int                         si,
    int                         sj,
    bool                        is_diag,
    cudaStream_t                stream);

} // namespace gansu
