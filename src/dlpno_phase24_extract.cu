/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_phase24_extract.hpp"

#include <cuda_runtime.h>

namespace gansu {

namespace {

constexpr int kBlock = 256;

inline int grid_for(std::size_t total) {
    return static_cast<int>((total + kBlock - 1) / kBlock);
}

// ---------------------------------------------------------------------------
// Kernels — each writes ONE per-output-element value via a strided gather
// from the full MO ERI buffer (`d_eri_mo`, layout `[p,q,r,s] = (pq|rs)`,
// row-major over n_emb⁴).
//
// Index conventions match precompute_phase24_integrals (dlpno_ccsd.cu).
// ---------------------------------------------------------------------------

/// T_pair^{(ij)}[k, l, c, d] = 2·(k, n_lmo+c | l, n_lmo+d) − (k, n_lmo+d | l, n_lmo+c)
/// Output layout: ((k · n_lmo + l) · n_pno + c) · n_pno + d.
/// Uses __dmul_rn / __dsub_rn to forbid FMA contraction → bit-exact match
/// with host `2.0*x - y` sequence.
__global__ void extract_T_pair_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int d = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int c = static_cast<int>(t1 % n_pno);
    const std::size_t t2 = t1 / n_pno;
    const int l = static_cast<int>(t2 % n_lmo);
    const int k = static_cast<int>(t2 / n_lmo);

    const std::size_t idx_kcld =
        static_cast<std::size_t>(k) * n_emb3 +
        static_cast<std::size_t>(n_lmo + c) * n_emb2 +
        static_cast<std::size_t>(l) * n_emb +
        static_cast<std::size_t>(n_lmo + d);
    const std::size_t idx_kdlc =
        static_cast<std::size_t>(k) * n_emb3 +
        static_cast<std::size_t>(n_lmo + d) * n_emb2 +
        static_cast<std::size_t>(l) * n_emb +
        static_cast<std::size_t>(n_lmo + c);

    const real_t x = d_eri_mo[idx_kcld];
    const real_t y = d_eri_mo[idx_kdlc];
    // Forbid FMA contraction to bit-match host (2.0*x) - y.
    d_out[tid] = __dsub_rn(__dmul_rn(2.0, x), y);
}

/// W_pair^{(ij)}[a, b, c, d] = (n_lmo+a, n_lmo+c | n_lmo+b, n_lmo+d).
/// Output layout: ((a · n_pno + b) · n_pno + c) · n_pno + d.
__global__ void extract_W_pair_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno)
{
    const std::size_t total =
        static_cast<std::size_t>(n_pno) * n_pno * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int d = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int c = static_cast<int>(t1 % n_pno);
    const std::size_t t2 = t1 / n_pno;
    const int b = static_cast<int>(t2 % n_pno);
    const int a = static_cast<int>(t2 / n_pno);

    const std::size_t e =
        static_cast<std::size_t>(n_lmo + a) * n_emb3 +
        static_cast<std::size_t>(n_lmo + c) * n_emb2 +
        static_cast<std::size_t>(n_lmo + b) * n_emb +
        static_cast<std::size_t>(n_lmo + d);
    d_out[tid] = d_eri_mo[e];
}

/// W_oooo^{(ij)}[k, l] = (k, s.i | l, s.j).  Output layout: k · n_lmo + l.
__global__ void extract_W_oooo_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int si, int sj)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int l = blockIdx.y;
    if (k >= n_lmo || l >= n_lmo) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const std::size_t e =
        static_cast<std::size_t>(k) * n_emb3 +
        static_cast<std::size_t>(si) * n_emb2 +
        static_cast<std::size_t>(l) * n_emb +
        static_cast<std::size_t>(sj);
    d_out[static_cast<std::size_t>(k) * n_lmo + l] = d_eri_mo[e];
}

/// W_ovov[a, k, c] = (n_lmo+a, I | k, n_lmo+c)   with I = si or sj.
/// W_ovvo[a, k, c] = (n_lmo+a, n_lmo+c | k, I).
/// Output layout: (a · n_lmo + k) · n_pno + c.
/// The role flag `mode` selects 0=OVOV-i, 1=OVOV-j, 2=OVVO-i, 3=OVVO-j.
__global__ void extract_ovov_ovvo_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno,
    int I,
    int mode)
{
    const std::size_t total =
        static_cast<std::size_t>(n_pno) * n_lmo * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int c = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int k = static_cast<int>(t1 % n_lmo);
    const int a = static_cast<int>(t1 / n_lmo);

    std::size_t e;
    if (mode == 0 || mode == 1) {
        // OVOV: eri_mo[n_lmo+a, I, k, n_lmo+c]
        e = static_cast<std::size_t>(n_lmo + a) * n_emb3 +
            static_cast<std::size_t>(I) * n_emb2 +
            static_cast<std::size_t>(k) * n_emb +
            static_cast<std::size_t>(n_lmo + c);
    } else {
        // OVVO: eri_mo[n_lmo+a, n_lmo+c, k, I]
        e = static_cast<std::size_t>(n_lmo + a) * n_emb3 +
            static_cast<std::size_t>(n_lmo + c) * n_emb2 +
            static_cast<std::size_t>(k) * n_emb +
            static_cast<std::size_t>(I);
    }
    d_out[tid] = d_eri_mo[e];
}

/// B-a.6c IP dense-free bare ph-ladder block, occupied I = si or sj fixed.
///   mode 0 (OVVO): ovvo_bare[m,a,d] = (m d | a I) = eri[m, n_lmo+d, n_lmo+a, I]
///   mode 1 (OOVV): oovv_bare[m,a,d] = (m I | a d) = eri[m, I, n_lmo+a, n_lmo+d]
/// Output layout: (m · n_pno + a) · n_pno + d  (size n_lmo · n_pno²).
/// These are the bare seeds of the per-pair PNO Wovvo_pno/Wovov_pno (occ=I);
/// the PNO ERI equals the canonical congruence U^(ij)ᵀ⊗2 (bare) bit-for-bit,
/// so the dense nocc²·nvir² Wovvo/Wovov is never materialised (true scaling).
__global__ void extract_ip_bare_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno,
    int I,
    int mode)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int d = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int a = static_cast<int>(t1 % n_pno);
    const int m = static_cast<int>(t1 / n_pno);

    std::size_t e;
    if (mode == 0) {
        // OVVO: eri_mo[m, n_lmo+d, n_lmo+a, I]
        e = static_cast<std::size_t>(m) * n_emb3 +
            static_cast<std::size_t>(n_lmo + d) * n_emb2 +
            static_cast<std::size_t>(n_lmo + a) * n_emb +
            static_cast<std::size_t>(I);
    } else {
        // OOVV: eri_mo[m, I, n_lmo+a, n_lmo+d]
        e = static_cast<std::size_t>(m) * n_emb3 +
            static_cast<std::size_t>(I) * n_emb2 +
            static_cast<std::size_t>(n_lmo + a) * n_emb +
            static_cast<std::size_t>(n_lmo + d);
    }
    d_out[tid] = d_eri_mo[e];
}

/// V_ovov_pair^{(ij)}[l, k, d, c] = (l, n_lmo+d | k, n_lmo+c).
/// Output layout: ((l · n_lmo + k) · n_pno + d) · n_pno + c.
__global__ void extract_V_ovov_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int c = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int d = static_cast<int>(t1 % n_pno);
    const std::size_t t2 = t1 / n_pno;
    const int k = static_cast<int>(t2 % n_lmo);
    const int l = static_cast<int>(t2 / n_lmo);

    const std::size_t e =
        static_cast<std::size_t>(l) * n_emb3 +
        static_cast<std::size_t>(n_lmo + d) * n_emb2 +
        static_cast<std::size_t>(k) * n_emb +
        static_cast<std::size_t>(n_lmo + c);
    d_out[tid] = d_eri_mo[e];
}

/// W_ovvv_diag[i_lmo][a, b, c] = (i_lmo, n_lmo+a | n_lmo+b, n_lmo+c).
/// Output layout: (a · n_pno + b) · n_pno + c. Only invoked for diagonal pairs.
__global__ void extract_W_ovvv_diag_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_out,
    int n_emb, int n_lmo, int n_pno, int i_lmo)
{
    const std::size_t total =
        static_cast<std::size_t>(n_pno) * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int c = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int b = static_cast<int>(t1 % n_pno);
    const int a = static_cast<int>(t1 / n_pno);

    const std::size_t e =
        static_cast<std::size_t>(i_lmo) * n_emb3 +
        static_cast<std::size_t>(n_lmo + a) * n_emb2 +
        static_cast<std::size_t>(n_lmo + b) * n_emb +
        static_cast<std::size_t>(n_lmo + c);
    d_out[tid] = d_eri_mo[e];
}

/// Three λ-block extracts in one launch:
///   W_ovvo_lambda[a, b]     = eri[si, n_lmo+a, n_lmo+b, sj]
///   W_ovvo_lambda_alt[a, b] = eri[sj, n_lmo+a, n_lmo+b, si]
///   W_oovv_lambda[b, a]     = eri[si, sj, n_lmo+b, n_lmo+a]
/// Each output is n_pno² doubles, written to three separate destination
/// pointers passed in.
__global__ void extract_lambda_pno2_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_ovvo,
    real_t*       __restrict__ d_ovvo_alt,
    real_t*       __restrict__ d_oovv,
    int n_emb, int n_lmo, int n_pno, int si, int sj)
{
    const std::size_t total =
        static_cast<std::size_t>(n_pno) * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int b = static_cast<int>(tid % n_pno);
    const int a = static_cast<int>(tid / n_pno);

    const std::size_t e_ovvo_i =
        static_cast<std::size_t>(si) * n_emb3 +
        static_cast<std::size_t>(n_lmo + a) * n_emb2 +
        static_cast<std::size_t>(n_lmo + b) * n_emb +
        static_cast<std::size_t>(sj);
    const std::size_t e_ovvo_j =
        static_cast<std::size_t>(sj) * n_emb3 +
        static_cast<std::size_t>(n_lmo + a) * n_emb2 +
        static_cast<std::size_t>(n_lmo + b) * n_emb +
        static_cast<std::size_t>(si);
    const std::size_t e_oovv =
        static_cast<std::size_t>(si) * n_emb3 +
        static_cast<std::size_t>(sj) * n_emb2 +
        static_cast<std::size_t>(n_lmo + b) * n_emb +
        static_cast<std::size_t>(n_lmo + a);

    const std::size_t out_ab = static_cast<std::size_t>(a) * n_pno + b;
    const std::size_t out_ba = static_cast<std::size_t>(b) * n_pno + a;
    d_ovvo[out_ab]     = d_eri_mo[e_ovvo_i];
    d_ovvo_alt[out_ab] = d_eri_mo[e_ovvo_j];
    d_oovv[out_ba]     = d_eri_mo[e_oovv];
}

/// Two OVOO extracts in one launch:
///   W_ovoo_lambda[a, k]     = eri[si, n_lmo+a, sj, k]
///   W_ovoo_lambda_alt[a, k] = eri[sj, n_lmo+a, si, k]
__global__ void extract_lambda_ovoo_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_ovoo,
    real_t*       __restrict__ d_ovoo_alt,
    int n_emb, int n_lmo, int n_pno, int si, int sj)
{
    const std::size_t total =
        static_cast<std::size_t>(n_pno) * n_lmo;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int k = static_cast<int>(tid % n_lmo);
    const int a = static_cast<int>(tid / n_lmo);

    const std::size_t e_i =
        static_cast<std::size_t>(si) * n_emb3 +
        static_cast<std::size_t>(n_lmo + a) * n_emb2 +
        static_cast<std::size_t>(sj) * n_emb +
        static_cast<std::size_t>(k);
    const std::size_t e_j =
        static_cast<std::size_t>(sj) * n_emb3 +
        static_cast<std::size_t>(n_lmo + a) * n_emb2 +
        static_cast<std::size_t>(si) * n_emb +
        static_cast<std::size_t>(k);

    d_ovoo[tid]     = d_eri_mo[e_i];
    d_ovoo_alt[tid] = d_eri_mo[e_j];
}

/// Increment 2 / S2: OOOV + OOVO extracts in one launch (over n_lmo · n_pno).
/// EXCHANGE-paired chemist blocks for the canonical ooov·t2 T1 source
/// (see dlpno_pair_data.hpp pairing note); both laid out i·n_pno + c:
///   W_ooov_pq[i,c] = eri[si, i, sj, n_lmo+c] = (pi|qc)
///   W_oovo_pq[i,c] = eri[sj, i, si, n_lmo+c] = (qi|pc)
__global__ void extract_ooov_oovo_kernel(
    const real_t* __restrict__ d_eri_mo,
    real_t*       __restrict__ d_ooov,
    real_t*       __restrict__ d_oovo,
    int n_emb, int n_lmo, int n_pno, int si, int sj)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const std::size_t n_emb2 = static_cast<std::size_t>(n_emb) * n_emb;
    const std::size_t n_emb3 = n_emb2 * n_emb;

    const int c = static_cast<int>(tid % n_pno);
    const int i = static_cast<int>(tid / n_pno);

    const std::size_t e_ooov =
        static_cast<std::size_t>(si) * n_emb3 +
        static_cast<std::size_t>(i) * n_emb2 +
        static_cast<std::size_t>(sj) * n_emb +
        static_cast<std::size_t>(n_lmo + c);
    const std::size_t e_oovo =
        static_cast<std::size_t>(sj) * n_emb3 +
        static_cast<std::size_t>(i) * n_emb2 +
        static_cast<std::size_t>(si) * n_emb +
        static_cast<std::size_t>(n_lmo + c);
    d_ooov[tid] = d_eri_mo[e_ooov];
    d_oovo[tid] = d_eri_mo[e_oovo];
}

// ---------------------------------------------------------------------------
// S7c relayout kernels (block-wise build path). These operate on the small
// per-block buffers produced by mo_eri_block_into, never the n_emb⁴ tensor.
// ---------------------------------------------------------------------------

/// out(i,j,k,l) = in(i,k,j,l); in dims (A0,A1,A2,A3), out dims (A0,A2,A1,A3).
__global__ void transpose_mid_kernel(
    const real_t* __restrict__ d_in,
    real_t*       __restrict__ d_out,
    int A0, int A1, int A2, int A3)
{
    const std::size_t total =
        static_cast<std::size_t>(A0) * A1 * A2 * A3;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    // Decode output index (i, j, k, l) with output dims (A0, A2, A1, A3).
    const int l = static_cast<int>(tid % A3);
    const std::size_t t1 = tid / A3;
    const int k = static_cast<int>(t1 % A1);   // out dim 2 has extent A1
    const std::size_t t2 = t1 / A1;
    const int j = static_cast<int>(t2 % A2);   // out dim 1 has extent A2
    const int i = static_cast<int>(t2 / A2);

    // in(i, k, j, l) with in dims (A0, A1, A2, A3).
    const std::size_t in_idx =
        ((static_cast<std::size_t>(i) * A1 + k) * A2 + j) * A3 + l;
    d_out[tid] = d_in[in_idx];
}

/// T_pair[k,l,c,d] = 2·A[k,c,l,d] − A[k,d,l,c], A natural layout
/// ((k·n_pno+c)·n_lmo+l)·n_pno+d, output ((k·n_lmo+l)·n_pno+c)·n_pno+d.
__global__ void fuse_T_from_A_kernel(
    const real_t* __restrict__ d_A,
    real_t*       __restrict__ d_out,
    int n_lmo, int n_pno)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    const std::size_t tid =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    const int d = static_cast<int>(tid % n_pno);
    const std::size_t t1 = tid / n_pno;
    const int c = static_cast<int>(t1 % n_pno);
    const std::size_t t2 = t1 / n_pno;
    const int l = static_cast<int>(t2 % n_lmo);
    const int k = static_cast<int>(t2 / n_lmo);

    const std::size_t a_kcld =
        (static_cast<std::size_t>(k) * n_pno + c) * n_lmo * n_pno
        + static_cast<std::size_t>(l) * n_pno + d;
    const std::size_t a_kdlc =
        (static_cast<std::size_t>(k) * n_pno + d) * n_lmo * n_pno
        + static_cast<std::size_t>(l) * n_pno + c;

    const real_t x = d_A[a_kcld];
    const real_t y = d_A[a_kdlc];
    // Forbid FMA contraction to bit-match host (2.0*x) - y.
    d_out[tid] = __dsub_rn(__dmul_rn(2.0, x), y);
}

} // anonymous namespace

Phase24ExtractLayout compute_phase24_extract_layout(
    int n_lmo, int n_pno, bool is_diag, bool include_singles)
{
    Phase24ExtractLayout L;
    L.sz_T_pair      = static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    L.sz_W_pair      = static_cast<std::size_t>(n_pno) * n_pno * n_pno * n_pno;
    L.sz_W_oooo      = static_cast<std::size_t>(n_lmo) * n_lmo;
    L.sz_W_ovov      = static_cast<std::size_t>(n_pno) * n_lmo * n_pno;
    L.sz_V_ovov      = static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    L.sz_W_ovvv_diag = is_diag
                       ? static_cast<std::size_t>(n_pno) * n_pno * n_pno
                       : 0;
    L.sz_pno2        = static_cast<std::size_t>(n_pno) * n_pno;
    L.sz_ovoo        = static_cast<std::size_t>(n_pno) * n_lmo;
    L.sz_ovvv        = include_singles
                       ? static_cast<std::size_t>(n_pno) * n_pno * n_pno : 0;
    L.sz_ooov        = include_singles
                       ? static_cast<std::size_t>(n_lmo) * n_pno : 0;

    std::size_t off = 0;
    L.off_T_pair             = off; off += L.sz_T_pair;
    L.off_W_pair             = off; off += L.sz_W_pair;
    L.off_W_oooo             = off; off += L.sz_W_oooo;
    L.off_W_ovov_i           = off; off += L.sz_W_ovov;
    L.off_W_ovov_j           = off; off += L.sz_W_ovov;
    L.off_W_ovvo_i           = off; off += L.sz_W_ovov;
    L.off_W_ovvo_j           = off; off += L.sz_W_ovov;
    L.off_V_ovov             = off; off += L.sz_V_ovov;
    L.off_W_ovvv_diag        = off; off += L.sz_W_ovvv_diag;     // 0 if !diag
    L.off_W_ovvo_lambda      = off; off += L.sz_pno2;
    L.off_W_ovvo_lambda_alt  = off; off += L.sz_pno2;
    L.off_W_oovv_lambda      = off; off += L.sz_pno2;
    L.off_W_ovoo_lambda      = off; off += L.sz_ovoo;
    L.off_W_ovoo_lambda_alt  = off; off += L.sz_ovoo;
    // B-a.6c IP dense-free bare ph-ladder blocks (size n_lmo·n_pno² = sz_W_ovov).
    L.off_W_ovvo_bare_i      = off; off += L.sz_W_ovov;
    L.off_W_ovvo_bare_j      = off; off += L.sz_W_ovov;
    L.off_W_oovv_bare_i      = off; off += L.sz_W_ovov;
    L.off_W_oovv_bare_j      = off; off += L.sz_W_ovov;
    // Increment 2 / S2: off-diag ovvv (occ i/j) + ooov/oovo.
    L.off_W_ovvv_pi          = off; off += L.sz_ovvv;
    L.off_W_ovvv_pj          = off; off += L.sz_ovvv;
    L.off_W_ooov_pq          = off; off += L.sz_ooov;
    L.off_W_oovo_pq          = off; off += L.sz_ooov;
    L.total = off;
    return L;
}

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
    cudaStream_t                stream)
{
    // T_pair (largest output for cholesterol-class — n_lmo² · n_pno²).
    {
        const std::size_t n = layout.sz_T_pair;
        if (n > 0) {
            extract_T_pair_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_T_pair,
                n_emb, n_lmo, n_pno);
        }
    }
    // W_pair (n_pno⁴ — small for n_pno≲30).
    {
        const std::size_t n = layout.sz_W_pair;
        if (n > 0) {
            extract_W_pair_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_pair,
                n_emb, n_lmo, n_pno);
        }
    }
    // W_oooo (n_lmo² — tiny).
    {
        if (n_lmo > 0) {
            dim3 block(64, 1, 1);
            dim3 grid((n_lmo + block.x - 1) / block.x, n_lmo, 1);
            extract_W_oooo_kernel<<<grid, block, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_oooo,
                n_emb, n_lmo, si, sj);
        }
    }
    // W_ovov_i, W_ovov_j, W_ovvo_i, W_ovvo_j (each n_pno · n_lmo · n_pno).
    {
        const std::size_t n = layout.sz_W_ovov;
        if (n > 0) {
            extract_ovov_ovvo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovov_i,
                n_emb, n_lmo, n_pno, si, /*mode=*/0);
            extract_ovov_ovvo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovov_j,
                n_emb, n_lmo, n_pno, sj, /*mode=*/1);
            extract_ovov_ovvo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvo_i,
                n_emb, n_lmo, n_pno, si, /*mode=*/2);
            extract_ovov_ovvo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvo_j,
                n_emb, n_lmo, n_pno, sj, /*mode=*/3);
        }
    }
    // V_ovov_pair (n_lmo² · n_pno² — same size as T_pair).
    {
        const std::size_t n = layout.sz_V_ovov;
        if (n > 0) {
            extract_V_ovov_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_V_ovov,
                n_emb, n_lmo, n_pno);
        }
    }
    // W_ovvv_diag (only when is_diag).
    if (is_diag) {
        const std::size_t n = layout.sz_W_ovvv_diag;
        if (n > 0) {
            // For diagonal pair (i,i) the LMO index is si (== sj).
            extract_W_ovvv_diag_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvv_diag,
                n_emb, n_lmo, n_pno, si);
        }
    }
    // λ pno²-shaped triple (W_ovvo_lambda, _alt, W_oovv_lambda).
    {
        const std::size_t n = layout.sz_pno2;
        if (n > 0) {
            extract_lambda_pno2_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo,
                d_packed_out + layout.off_W_ovvo_lambda,
                d_packed_out + layout.off_W_ovvo_lambda_alt,
                d_packed_out + layout.off_W_oovv_lambda,
                n_emb, n_lmo, n_pno, si, sj);
        }
    }
    // λ ovoo pair (W_ovoo_lambda, _alt).
    {
        const std::size_t n = layout.sz_ovoo;
        if (n > 0) {
            extract_lambda_ovoo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo,
                d_packed_out + layout.off_W_ovoo_lambda,
                d_packed_out + layout.off_W_ovoo_lambda_alt,
                n_emb, n_lmo, n_pno, si, sj);
        }
    }
    // B-a.6c IP dense-free bare ph-ladder (ovvo/oovv, i/j roles); each
    // n_lmo·n_pno² = sz_W_ovov.
    {
        const std::size_t n = layout.sz_W_ovov;
        if (n > 0) {
            extract_ip_bare_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvo_bare_i,
                n_emb, n_lmo, n_pno, si, /*mode=*/0);
            extract_ip_bare_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvo_bare_j,
                n_emb, n_lmo, n_pno, sj, /*mode=*/0);
            extract_ip_bare_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_oovv_bare_i,
                n_emb, n_lmo, n_pno, si, /*mode=*/1);
            extract_ip_bare_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_oovv_bare_j,
                n_emb, n_lmo, n_pno, sj, /*mode=*/1);
        }
    }
    // Increment 2 / S2: off-diag ovvv (occ si / sj) reuse the ovvv_diag kernel.
    {
        const std::size_t n = layout.sz_ovvv;
        if (n > 0) {
            extract_W_ovvv_diag_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvv_pi,
                n_emb, n_lmo, n_pno, si);
            extract_W_ovvv_diag_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo, d_packed_out + layout.off_W_ovvv_pj,
                n_emb, n_lmo, n_pno, sj);
        }
    }
    // Increment 2 / S2: ooov + oovo.
    {
        const std::size_t n = layout.sz_ooov;
        if (n > 0) {
            extract_ooov_oovo_kernel<<<grid_for(n), kBlock, 0, stream>>>(
                d_eri_mo,
                d_packed_out + layout.off_W_ooov_pq,
                d_packed_out + layout.off_W_oovo_pq,
                n_emb, n_lmo, n_pno, si, sj);
        }
    }
}

void launch_phase24_transpose_mid(
    const real_t* d_in,
    real_t*       d_out,
    int A0, int A1, int A2, int A3,
    cudaStream_t  stream)
{
    const std::size_t total =
        static_cast<std::size_t>(A0) * A1 * A2 * A3;
    if (total == 0) return;
    transpose_mid_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_in, d_out, A0, A1, A2, A3);
}

void launch_phase24_fuse_T_from_A(
    const real_t* d_A,
    real_t*       d_out,
    int n_lmo, int n_pno,
    cudaStream_t  stream)
{
    const std::size_t total =
        static_cast<std::size_t>(n_lmo) * n_lmo * n_pno * n_pno;
    if (total == 0) return;
    fuse_T_from_A_kernel<<<grid_for(total), kBlock, 0, stream>>>(
        d_A, d_out, n_lmo, n_pno);
}

} // namespace gansu
