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
    /// Increment 2 / S1: VVOV (ab|ic)/(ab|jc) + VOOO (ak|ij) for the linear
    /// T1→T2 back-coupling. vvov layout (a·n_pno+b)·n_pno+c (size n_pno³);
    /// vooo layout a·n_lmo+k (size n_pno·n_lmo).
    std::size_t off_W_vvov_i           = 0;
    std::size_t off_W_vvov_j           = 0;
    std::size_t off_W_vooo_i           = 0;

    std::size_t sz_T_pair       = 0;  ///< n_lmo² · n_pno²
    std::size_t sz_W_pair       = 0;  ///< n_pno⁴
    std::size_t sz_W_oooo       = 0;  ///< n_lmo²
    std::size_t sz_W_ovov       = 0;  ///< n_pno · n_lmo · n_pno  (each of i/j)
    std::size_t sz_V_ovov       = 0;  ///< n_lmo² · n_pno²
    std::size_t sz_W_ovvv_diag  = 0;  ///< n_pno³ (0 when !is_diag)
    std::size_t sz_pno2         = 0;  ///< n_pno² (W_ovvo_lambda etc.)
    std::size_t sz_ovoo         = 0;  ///< n_pno · n_lmo (W_ovoo_lambda etc.)
    std::size_t sz_vvov         = 0;  ///< n_pno³ (W_vvov_i/j)
    std::size_t sz_vooo         = 0;  ///< n_pno · n_lmo (W_vooo_i; == sz_ovoo)
};

/// Compute byte offsets / sizes for the packed output buffer.
/// `include_singles` (Increment 2 / S1) reserves space for the VVOV/VOOO
/// blocks; when false those sizes are 0 (no allocation / extraction).
Phase24ExtractLayout compute_phase24_extract_layout(
    int n_lmo, int n_pno, bool is_diag, bool include_singles = false);

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

// ---------------------------------------------------------------------------
// Step S7c — block-wise build relayouts.
//
// S7c replaces the full n_emb⁴ build_mo_eri_into + S7b gather with a
// per-pair build_B_mo + a handful of mo_eri_block_into block builds (the
// MP2 V-block pattern). mo_eri_block_into emits a block (p q | r s) in its
// natural "bra-outer, ket-inner" layout out[((p·qn+q)·rn+r)·sn+s]. Most of
// the 14 Phase24 destinations match that natural layout directly, but a few
// require a transpose of the two middle indices, and T_pair/V_ovov are both
// derived from a single (occ,pno|occ,pno) block `A` with A[k,c,l,d]=(kc|ld).
// These two launchers cover those non-direct cases.
// ---------------------------------------------------------------------------

/// Transpose the two middle indices of a row-major 4-index tensor:
///   out(i,j,k,l) = in(i,k,j,l),  in dims (A0,A1,A2,A3), out dims (A0,A2,A1,A3).
/// Used for W_pair ([a,c,b,d]→[a,b,c,d]), V_ovov ([k,c,l,d]→[k,l,c,d] i.e.
/// the A-block congruent V layout), W_ovvo_i/j ([a,c,k]→[a,k,c], A3=1) and the
/// IP bare W_ovvo_bare_i/j ([m,d,a]→[m,a,d], A3=1).
void launch_phase24_transpose_mid(
    const real_t* d_in,
    real_t*       d_out,
    int A0, int A1, int A2, int A3,
    cudaStream_t  stream);

/// T_pair^{(ij)}[k,l,c,d] = 2·A[k,c,l,d] − A[k,d,l,c] from the single
/// (occ,pno|occ,pno) block A (A[k,c,l,d] = (k,c'|l,d'), natural layout
/// ((k·n_pno+c)·n_lmo+l)·n_pno+d). Output layout ((k·n_lmo+l)·n_pno+c)·n_pno+d.
/// Uses __dsub_rn(__dmul_rn(2,x),y) to bit-match the host 2·x−y rounding.
void launch_phase24_fuse_T_from_A(
    const real_t* d_A,
    real_t*       d_out,
    int n_lmo, int n_pno,
    cudaStream_t  stream);

} // namespace gansu
