/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iomanip>
#include <iostream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include "uhf.hpp"
#include "eri_stored.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"

#include <Eigen/Dense>

#include "ao2mo.cuh"

#define FULLMASK 0xffffffff

namespace gansu {
// Forward declaration (defined in eri_stored.cu, has CPU fallback via Kronecker product)
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);
}

// ============================================================
//  CPU-side index helpers (mirrors of __device__ functions in ao2mo.cuh)
// ============================================================
namespace cpu_idx {

static inline size_t q2s(int mu, int nu, int la, int si, int N) {
    return (size_t)N*N*N*mu + (size_t)N*N*nu + (size_t)N*la + si;
}

static inline size_t ovov2seq(int i, int a, int j, int b,
                              int nocc, int nvir) {
    return ((size_t)i*nvir + a)*nocc*nvir + (size_t)j*nvir + b;
}

static inline size_t ovov2seq_aabb(int i, int a, int j, int b,
                                   int nocc_al, int nvir_al,
                                   int nocc_be, int nvir_be) {
    return ((size_t)i*nvir_al + a)*nocc_be*nvir_be + (size_t)j*nvir_be + b;
}

static inline size_t ovov2s(int i, int a, int j, int b, int nocc, int nvir) {
    return (size_t)nocc*nvir*nvir*i + (size_t)nocc*nvir*(a-nocc) + (size_t)nvir*j + (b-nocc);
}
static inline size_t oovv2s(int i, int j, int a, int b, int nocc, int nvir) {
    return (size_t)nvir*nvir*nocc*i + (size_t)nvir*nvir*j + (size_t)nvir*(a-nocc) + (b-nocc);
}
static inline size_t vvoo2s(int c, int d, int i, int j, int nocc, int nvir) {
    return (size_t)nocc*nocc*nvir*(c-nocc) + (size_t)nocc*nocc*(d-nocc) + (size_t)nocc*i + j;
}
static inline size_t ovvo2s(int k, int c, int b, int j, int nocc, int nvir) {
    return (size_t)nvir*nvir*nocc*k + (size_t)nvir*nocc*(c-nocc) + (size_t)nocc*(b-nocc) + j;
}
static inline size_t oooo2s(int i, int j, int k, int l, int nocc) {
    return (size_t)nocc*nocc*nocc*i + (size_t)nocc*nocc*j + (size_t)nocc*k + l;
}
static inline size_t vvvv2s(int a, int b, int c, int d, int nocc, int nvir) {
    return (size_t)nvir*nvir*nvir*(a-nocc) + (size_t)nvir*nvir*(b-nocc) + (size_t)nvir*(c-nocc) + (d-nocc);
}

// aabb index helpers
static inline size_t oooo2s_abab(int i, int j, int k, int l, int nocc_al, int nocc_be) {
    return (size_t)nocc_be*nocc_al*nocc_be*i + (size_t)nocc_al*nocc_be*j + (size_t)nocc_be*k + l;
}
static inline size_t vvvv2s_abab(int a, int b, int c, int d, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    return (size_t)nvir_be*nvir_al*nvir_be*(a-nocc_al) + (size_t)nvir_al*nvir_be*(b-nocc_be) + (size_t)nvir_be*(c-nocc_al) + (d-nocc_be);
}
static inline size_t ovvo2s_bbaa(int k, int c, int b, int j, int nocc_be, int nocc_al, int nvir_be, int nvir_al) {
    return (size_t)nvir_be*nvir_al*nocc_al*k + (size_t)nvir_al*nocc_al*(c-nocc_be) + (size_t)nocc_al*(b-nocc_al) + j;
}
static inline size_t ovvo2s_baab(int k, int c, int b, int j, int nocc_be, int nocc_al, int nvir_al, int nvir_be) {
    return (size_t)nvir_al*nvir_al*nocc_be*k + (size_t)nvir_al*nocc_be*(c-nocc_al) + (size_t)nocc_be*(b-nocc_al) + j;
}
static inline size_t ovov2s_aabb(int i, int a, int j, int b, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    return (size_t)nvir_al*nocc_be*nvir_be*i + (size_t)nocc_be*nvir_be*(a-nocc_al) + (size_t)nvir_be*j + (b-nocc_be);
}
static inline size_t ovvo2s_aabb(int i, int a, int b, int j, int nocc_al, int nvir_al, int nvir_be, int nocc_be) {
    return (size_t)nvir_al*nvir_be*nocc_be*i + (size_t)nvir_be*nocc_be*(a-nocc_al) + (size_t)nocc_be*(b-nocc_be) + j;
}
static inline size_t oovv2s_abab(int i, int j, int a, int b, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    return (size_t)nocc_be*nvir_al*nvir_be*i + (size_t)nvir_al*nvir_be*j + (size_t)nvir_be*(a-nocc_al) + (b-nocc_be);
}
static inline size_t vvoo2s_abab(int a, int b, int i, int j, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    return (size_t)nvir_be*nocc_al*nocc_be*(a-nocc_al) + (size_t)nocc_al*nocc_be*(b-nocc_be) + (size_t)nocc_be*i + j;
}

} // namespace cpu_idx


// CPU-side row-major DGEMM helper: C = opA(A)*opB(B)  (m x n), inner dim k.
// opA_rm/opB_rm: CUBLAS_OP_N or CUBLAS_OP_T (reuse the enum even on CPU path).
// This wraps gpu::matrixMatrixProductRect which handles CPU mode internally.
static inline void cpu_dgemm_row_major(
    int m, int n, int k,
    double alpha_val,
    const double* A, const double* B,
    double* C,
    bool trA, bool trB,
    bool accumulate = false)
{
    // gpu::matrixMatrixProductRect already has row-major semantics:
    //   C(m,n) = opA(A)*opB(B), with A stored row-major (m x k) [!trA]
    //   or (k x m) [trA], B stored row-major (k x n) [!trB] or (n x k) [trB].
    // (The previous implementation passed (B, A, n, m, k, trB, trA), which
    // computes B*A for square shapes and misreads layouts for non-square
    // shapes; that silently corrupted the CPU MP3 contractions.)
    gansu::gpu::matrixMatrixProductRect(A, B, C, m, n, k, trA, trB, accumulate, alpha_val);
}

// CPU mirror of dgemm_device_row_major (ao2mo.cuh): row-major
// C(m x n) = opA(A) * opB(B) with explicit leading dimensions (row strides),
// so the CPU path can issue exactly the same GEMM sequence (including
// column-sliced operands such as C + num_occ with lda = N) as the cuBLAS
// path. Uses Eigen's OpenMP-threaded GEMM.
static inline void cpu_dgemm_ld_row_major(
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const double* A, int lda,
    const double* B, int ldb,
    double* C, int ldc)
{
    const bool trA = (opA == CUBLAS_OP_T);
    const bool trB = (opB == CUBLAS_OP_T);
    // Physical (stored) shapes: A is (trA ? k x m : m x k), B is (trB ? n x k : k x n).
    const int rowsA = trA ? k : m;
    const int colsA = trA ? m : k;
    const int rowsB = trB ? n : k;
    const int colsB = trB ? k : n;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<>> Am(
        A, rowsA, colsA, Eigen::OuterStride<>(lda));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<>> Bm(
        B, rowsB, colsB, Eigen::OuterStride<>(ldb));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>, 0, Eigen::OuterStride<>> Cm(
        C, m, n, Eigen::OuterStride<>(ldc));

    if (trA && trB)      { Cm.noalias() = Am.transpose() * Bm.transpose(); }
    else if (trA)        { Cm.noalias() = Am.transpose() * Bm; }
    else if (trB)        { Cm.noalias() = Am * Bm.transpose(); }
    else                 { Cm.noalias() = Am * Bm; }
}

// CPU mirror of dgemm_strided_batched_device_row_major (ao2mo.cuh): the same
// row-major batched GEMM with per-batch strides (stride 0 broadcasts the
// operand). Batches run sequentially; each GEMM is OpenMP-threaded by Eigen.
static inline void cpu_dgemm_strided_batched_ld_row_major(
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    const double* A, int lda, long long int strideA,
    const double* B, int ldb, long long int strideB,
    double* C, int ldc, long long int strideC,
    int batchCount)
{
    for (int b = 0; b < batchCount; b++) {
        cpu_dgemm_ld_row_major(opA, opB, m, n, k,
            A + (size_t)b * (size_t)strideA, lda,
            B + (size_t)b * (size_t)strideB, ldb,
            C + (size_t)b * (size_t)strideC, ldc);
    }
}

// CPU mirror of dgeam_transpose_row_major (ao2mo.cuh):
// AT_rm (cols x rows) = A_rm^T (rows x cols), tiled for cache efficiency.
static inline void cpu_dgeam_transpose_row_major(
    int rows_rm, int cols_rm, const double* A_rm, double* AT_rm)
{
    constexpr int TILE = 64;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < rows_rm; ii += TILE) {
        for (int jj = 0; jj < cols_rm; jj += TILE) {
            const int imax = std::min(ii + TILE, rows_rm);
            const int jmax = std::min(jj + TILE, cols_rm);
            for (int i = ii; i < imax; i++) {
                for (int j = jj; j < jmax; j++) {
                    AT_rm[(size_t)j * rows_rm + i] = A_rm[(size_t)i * cols_rm + j];
                }
            }
        }
    }
}

// ============================================================
//  CPU implementations of ao2mo.cuh kernels (used from this file only)
// ============================================================
namespace cpu_kernels {
using namespace cpu_idx;

// --- same-spin tensorize kernels ---
static void tensorize_g_aaaa_oooo_cpu(double* out, const double* full, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nocc*nocc*nocc;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nocc*nocc), kl = idx % (nocc*nocc);
        int i = ij/nocc, j = ij%nocc, k = kl/nocc, l = kl%nocc;
        out[oooo2s(i,j,k,l,nocc)] = full[q2s(i,k,j,l,N)];
    }
}
static void tensorize_g_aaaa_vvvv_cpu(double* out, const double* full, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nvir*nvir*nvir*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ab = idx / (nvir*nvir), cd = idx % (nvir*nvir);
        int a = ab/nvir+nocc, b = ab%nvir+nocc, c = cd/nvir+nocc, d = cd%nvir+nocc;
        out[vvvv2s(a,b,c,d,nocc,nvir)] = full[q2s(a,c,b,d,N)];
    }
}
static void tensorize_u_aaaa_ovvo_cpu(double* out, const double* full, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nvir*nvir*nocc;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kc = idx / (nvir*nocc), bj = idx % (nvir*nocc);
        int k = kc/nvir, c = kc%nvir+nocc, b = bj/nocc+nocc, j = bj%nocc;
        out[ovvo2s(k,c,b,j,nocc,nvir)] = full[q2s(k,c,b,j,N)] - full[q2s(k,j,b,c,N)];
    }
}
static void tensorize_x_aaaa_ovov_cpu(double* out, const double* full, const double* eps, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nvir*nocc*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc*nvir), jb = idx % (nocc*nvir);
        int i = ia/nvir, a = ia%nvir+nocc, j = jb/nvir, b = jb%nvir+nocc;
        double d_eps = eps[i]+eps[j]-eps[a]-eps[b];
        out[ovov2s(i,a,j,b,nocc,nvir)] = full[q2s(i,a,j,b,N)] / d_eps;
    }
}
// Plain (non-amplitude) ovov block, CPU mirror of tensorize_g_aaaa_ovov:
// identical layout to tensorize_x_aaaa_ovov_cpu without the epsilon division.
// Used to obtain the UMP2 same-spin energy from the cascade full tensor
static void tensorize_g_aaaa_ovov_cpu(double* out, const double* full, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nvir*nocc*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc*nvir), jb = idx % (nocc*nvir);
        int i = ia/nvir, a = ia%nvir+nocc, j = jb/nvir, b = jb%nvir+nocc;
        out[ovov2s(i,a,j,b,nocc,nvir)] = full[q2s(i,a,j,b,N)];
    }
}
static void tensorize_y_aaaa_ovov_cpu(double* out, const double* full, const double* eps, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nvir*nocc*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc*nvir), jb = idx % (nocc*nvir);
        int i = ia/nvir, a = ia%nvir+nocc, j = jb/nvir, b = jb%nvir+nocc;
        double d_eps = eps[i]+eps[j]-eps[a]-eps[b];
        out[ovov2s(i,a,j,b,nocc,nvir)] = (full[q2s(i,a,j,b,N)] - full[q2s(i,b,j,a,N)]) / d_eps;
    }
}
static void kalb2klab_aaaa_cpu(double* oovv, const double* ovov, int nocc, int nvir) {
    const size_t n = (size_t)nocc*nvir*nocc*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ka = idx / (nocc*nvir), lb = idx % (nocc*nvir);
        int k = ka/nvir, a = ka%nvir+nocc, l = lb/nvir, b = lb%nvir+nocc;
        oovv[oovv2s(k,l,a,b,nocc,nvir)] = ovov[ovov2s(k,a,l,b,nocc,nvir)];
    }
}
static void icjd2cdij_aaaa_cpu(double* vvoo, const double* ovov, int nocc, int nvir) {
    const size_t n = (size_t)nocc*nvir*nocc*nvir;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ic = idx / (nocc*nvir), jd = idx % (nocc*nvir);
        int i = ic/nvir, c = ic%nvir+nocc, j = jd/nvir, d = jd%nvir+nocc;
        vvoo[vvoo2s(c,d,i,j,nocc,nvir)] = ovov[ovov2s(i,c,j,d,nocc,nvir)];
    }
}

// --- contraction kernels (reduction, return sum) ---
static double contract_3h3p_aaaaaa_cpu(const double* y, const double* t, int nocc, int nvir) {
    const size_t n = (size_t)nocc*nocc*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir*nvir), ab = idx % (nvir*nvir);
        int i = ij/nocc, j = ij%nocc, a = ab/nvir+nocc, b = ab%nvir+nocc;
        sum += y[ovov2s(i,a,j,b,nocc,nvir)] * t[ovvo2s(i,a,b,j,nocc,nvir)];
    }
    return sum;
}
static double contract_4h2p_2h4p_aaaaaa_cpu(const double* x, const double* t_oovv, const double* t_vvoo, int nocc, int nvir) {
    const size_t n = (size_t)nocc*nocc*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir*nvir), ab = idx % (nvir*nvir);
        int i = ij/nocc, j = ij%nocc, a = ab/nvir+nocc, b = ab%nvir+nocc;
        double xv = x[ovov2s(i,a,j,b,nocc,nvir)];
        double tv1 = t_oovv[oovv2s(i,j,a,b,nocc,nvir)];
        double tv2 = t_vvoo[vvoo2s(a,b,i,j,nocc,nvir)];
        sum += 0.5 * xv * (tv1 + tv2);
    }
    return sum;
}
static double contract_3h3p_aabaab_abaaba_cpu(const double* y, const double* t, int nocc, int nvir) {
    const size_t n = (size_t)nocc*nocc*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir*nvir), ab = idx % (nvir*nvir);
        int i = ij/nocc, j = ij%nocc, a = ab/nvir+nocc, b = ab%nvir+nocc;
        sum += 2.0 * y[ovov2s(i,a,j,b,nocc,nvir)] * t[ovvo2s(i,a,b,j,nocc,nvir)];
    }
    return sum;
}

// --- mixed-spin tensorize kernels ---
static void tensorize_g_aabb_oooo_cpu(double* out, const double* full, int nocc_al, int nocc_be, int N) {
    const size_t n = (size_t)nocc_al*nocc_be*nocc_al*nocc_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nocc_al*nocc_be), kl = idx % (nocc_al*nocc_be);
        int i = ij/nocc_be, j = ij%nocc_be, k = kl/nocc_be, l = kl%nocc_be;
        out[oooo2s_abab(i,j,k,l,nocc_al,nocc_be)] = full[q2s(i,k,j,l,N)];
    }
}
static void tensorize_g_aabb_vvvv_cpu(double* out, const double* full, int nocc_al, int nocc_be, int nvir_al, int nvir_be, int N) {
    const size_t n = (size_t)nvir_al*nvir_be*nvir_al*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ab = idx / (nvir_al*nvir_be), cd = idx % (nvir_al*nvir_be);
        int a = ab/nvir_be+nocc_al, b = ab%nvir_be+nocc_be, c = cd/nvir_be+nocc_al, d = cd%nvir_be+nocc_be;
        out[vvvv2s_abab(a,b,c,d,nocc_al,nocc_be,nvir_al,nvir_be)] = full[q2s(a,c,b,d,N)];
    }
}
static void tensorize_g_bbaa_ovvo_cpu(double* out, const double* full, int nocc_be, int nocc_al, int nvir_be, int nvir_al, int N) {
    const size_t n = (size_t)nocc_be*nvir_be*nvir_al*nocc_al;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kc = idx / (nvir_al*nocc_al), bj = idx % (nvir_al*nocc_al);
        int k = kc/nvir_be, c = kc%nvir_be+nocc_be, b = bj/nocc_al+nocc_al, j = bj%nocc_al;
        out[ovvo2s_bbaa(k,c,b,j,nocc_be,nocc_al,nvir_be,nvir_al)] = full[q2s(k,c,b,j,N)];
    }
}
static void tensorize_g_bbaa_oovv_cpu(double* out, const double* full, int nocc_be, int nocc_al, int nvir_be, int nvir_al, int N) {
    const size_t n = (size_t)nocc_be*nocc_be*nvir_al*nvir_al;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kj = idx / (nvir_al*nvir_al), bc = idx % (nvir_al*nvir_al);
        int k = kj/nocc_be, j = kj%nocc_be, b = bc/nvir_al+nocc_al, c = bc%nvir_al+nocc_al;
        out[ovvo2s_baab(k,c,b,j,nocc_be,nocc_al,nvir_al,nvir_be)] = full[q2s(k,j,b,c,N)];
    }
}
static void tensorize_u_bbbb_ovvo_cpu(double* out, const double* full, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nvir*nvir*nocc;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kc = idx / (nvir*nocc), bj = idx % (nvir*nocc);
        int k = kc/nvir, c = kc%nvir+nocc, b = bj/nocc+nocc, j = bj%nocc;
        out[ovvo2s(k,c,b,j,nocc,nvir)] = full[q2s(k,c,b,j,N)] - full[q2s(k,j,b,c,N)];
    }
}
static void tensorize_x_aabb_ovov_cpu(double* out, const double* full, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be, int N) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc_be*nvir_be), jb = idx % (nocc_be*nvir_be);
        int i = ia/nvir_al, a = ia%nvir_al+nocc_al, j = jb/nvir_be, b = jb%nvir_be+nocc_be;
        double d_eps = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        out[ovov2s_aabb(i,a,j,b,nocc_al,nvir_al,nocc_be,nvir_be)] = full[q2s(i,a,j,b,N)] / d_eps;
    }
}
// Plain (non-amplitude) aabb ovov block, CPU mirror of tensorize_g_aabb_ovov:
// identical layout to tensorize_x_aabb_ovov_cpu without the epsilon division.
// Used to obtain the UMP2 opposite-spin energy from the cascade full tensor
static void tensorize_g_aabb_ovov_cpu(double* out, const double* full, int nocc_al, int nvir_al, int nocc_be, int nvir_be, int N) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc_be*nvir_be), jb = idx % (nocc_be*nvir_be);
        int i = ia/nvir_al, a = ia%nvir_al+nocc_al, j = jb/nvir_be, b = jb%nvir_be+nocc_be;
        out[ovov2s_aabb(i,a,j,b,nocc_al,nvir_al,nocc_be,nvir_be)] = full[q2s(i,a,j,b,N)];
    }
}

// --- swapped-source variants: pack (bb|aa) compact blocks from the (aa|bb) full tensor ---
// ERI pair symmetry gives full_bbaa[k,c,b,j] = full_aabb[b,j,k,c], so the same
// compact blocks can be tensorized from the aabb tensor with the source index
// pairs swapped. Compact layouts and denominators are identical to the originals.
static void tensorize_g_bbaa_ovvo_swapped_cpu(double* out, const double* full, int nocc_be, int nocc_al, int nvir_be, int nvir_al, int N) {
    const size_t n = (size_t)nocc_be*nvir_be*nvir_al*nocc_al;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kc = idx / (nvir_al*nocc_al), bj = idx % (nvir_al*nocc_al);
        int k = kc/nvir_be, c = kc%nvir_be+nocc_be, b = bj/nocc_al+nocc_al, j = bj%nocc_al;
        out[ovvo2s_bbaa(k,c,b,j,nocc_be,nocc_al,nvir_be,nvir_al)] = full[q2s(b,j,k,c,N)];
    }
}
static void tensorize_g_bbaa_oovv_swapped_cpu(double* out, const double* full, int nocc_be, int nocc_al, int nvir_be, int nvir_al, int N) {
    const size_t n = (size_t)nocc_be*nocc_be*nvir_al*nvir_al;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int kj = idx / (nvir_al*nvir_al), bc = idx % (nvir_al*nvir_al);
        int k = kj/nocc_be, j = kj%nocc_be, b = bc/nvir_al+nocc_al, c = bc%nvir_al+nocc_al;
        out[ovvo2s_baab(k,c,b,j,nocc_be,nocc_al,nvir_al,nvir_be)] = full[q2s(b,c,k,j,N)];
    }
}
static void tensorize_x_aabb_ovov_swapped_cpu(double* out, const double* full, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be, int N) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ia = idx / (nocc_be*nvir_be), jb = idx % (nocc_be*nvir_be);
        int i = ia/nvir_al, a = ia%nvir_al+nocc_al, j = jb/nvir_be, b = jb%nvir_be+nocc_be;
        double d_eps = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        out[ovov2s_aabb(i,a,j,b,nocc_al,nvir_al,nocc_be,nvir_be)] = full[q2s(j,b,i,a,N)] / d_eps;
    }
}

// --- mixed-spin reorder kernels ---
static void aabb_icka2abba_iakc_cpu(double* out, const double* in, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ic = idx / (nocc_be*nvir_be), ka = idx % (nocc_be*nvir_be);
        int i = ic/nvir_al, c = ic%nvir_al+nocc_al, k = ka/nvir_be, a = ka%nvir_be+nocc_be;
        size_t iakc = (size_t)nvir_be*nocc_be*nvir_al*i + (size_t)nocc_be*nvir_al*(a-nocc_be) + (size_t)nvir_al*k + (c-nocc_al);
        out[iakc] = in[ovov2s_aabb(i,c,k,a,nocc_al,nvir_al,nocc_be,nvir_be)];
    }
}
static void aabb_kalb2abab_klab_cpu(double* out, const double* in, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ka = idx / (nocc_be*nvir_be), lb = idx % (nocc_be*nvir_be);
        int k = ka/nvir_al, a = ka%nvir_al+nocc_al, l = lb/nvir_be, b = lb%nvir_be+nocc_be;
        size_t klab = (size_t)nocc_be*nvir_al*nvir_be*k + (size_t)nvir_al*nvir_be*l + (size_t)nvir_be*(a-nocc_al) + (b-nocc_be);
        out[klab] = in[ovov2s_aabb(k,a,l,b,nocc_al,nvir_al,nocc_be,nvir_be)];
    }
}
static void aabb_icjd2abab_cdij_cpu(double* out, const double* in, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const size_t n = (size_t)nocc_al*nvir_al*nocc_be*nvir_be;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ic = idx / (nocc_be*nvir_be), jd = idx % (nocc_be*nvir_be);
        int i = ic/nvir_al, c = ic%nvir_al+nocc_al, j = jd/nvir_be, d = jd%nvir_be+nocc_be;
        size_t cdij = (size_t)nvir_be*nocc_al*nocc_be*(c-nocc_al) + (size_t)nocc_al*nocc_be*(d-nocc_be) + (size_t)nocc_be*i + j;
        out[cdij] = in[ovov2s_aabb(i,c,j,d,nocc_al,nvir_al,nocc_be,nvir_be)];
    }
}

// --- mixed-spin contraction kernels ---
static double contract_3h3p_abbabb_cpu(const double* x, const double* t, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    const size_t n = (size_t)nocc_al*nocc_be*nvir_al*nvir_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir_al*nvir_be), ab = idx % (nvir_al*nvir_be);
        int i = ij/nocc_be, j = ij%nocc_be, a = ab/nvir_be+nocc_al, b = ab%nvir_be+nocc_be;
        double xv = x[ovov2s_aabb(i,a,j,b,nocc_al,nvir_al,nocc_be,nvir_be)];
        double tv = t[ovvo2s_aabb(i,a,b,j,nocc_al,nvir_al,nvir_be,nocc_be)];
        sum += xv * tv;
    }
    return sum;
}
static double contract_3h3p_abbbaa_cpu(const double* x_aabb, const double* t_abab, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    const size_t n = (size_t)nocc_al*nocc_be*nvir_be*nvir_al;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir_be*nvir_al), ab = idx % (nvir_be*nvir_al);
        int i = ij/nocc_be, j = ij%nocc_be, a = ab/nvir_al+nocc_be, b = ab%nvir_al+nocc_al;
        double xv = x_aabb[ovov2s_aabb(i,b,j,a,nocc_al,nvir_al,nocc_be,nvir_be)];
        size_t iabj = (size_t)nvir_be*nvir_al*nocc_be*i + (size_t)nvir_al*nocc_be*(a-nocc_be) + (size_t)nocc_be*(b-nocc_al) + j;
        double tv = t_abab[iabj];
        sum += (-1.0) * xv * tv;
    }
    return sum;
}
static double contract_4h2p_2h4p_ababab_bababa_cpu(const double* x, const double* t_oovv, const double* t_vvoo, int nocc_al, int nocc_be, int nvir_al, int nvir_be) {
    const size_t n = (size_t)nocc_al*nocc_be*nvir_al*nvir_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t idx = 0; idx < n; idx++) {
        int ij = idx / (nvir_al*nvir_be), ab = idx % (nvir_al*nvir_be);
        int i = ij/nocc_be, j = ij%nocc_be, a = ab/nvir_be+nocc_al, b = ab%nvir_be+nocc_be;
        double xv = x[ovov2s_aabb(i,a,j,b,nocc_al,nvir_al,nocc_be,nvir_be)];
        double tv1 = t_oovv[oovv2s_abab(i,j,a,b,nocc_al,nocc_be,nvir_al,nvir_be)];
        double tv2 = t_vvoo[vvoo2s_abab(a,b,i,j,nocc_al,nocc_be,nvir_al,nvir_be)];
        sum += xv * (tv1 + tv2);
    }
    return sum;
}

// --- MP3 6-index kernels (same-spin) ---
static double compute_4h2p_ss_cpu(const double* g, const double* eps, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nocc*nocc*nocc*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijkl = tid / (nvir*nvir);
        int ab = tid % (nvir*nvir);
        int ij = ijkl / (nocc*nocc), kl = ijkl % (nocc*nocc);
        int i = ij/nocc, j = ij%nocc, k = kl/nocc, l = kl%nocc;
        int a = nocc + ab/nvir, b = nocc + ab%nvir;
        double eps_ijab = eps[i]+eps[j]-eps[a]-eps[b];
        double eps_klab = eps[k]+eps[l]-eps[a]-eps[b];
        double num = g[q2s(i,a,j,b,N)] * g[q2s(i,k,j,l,N)] * (g[q2s(k,a,l,b,N)] - g[q2s(k,b,l,a,N)]);
        sum += num / (eps_ijab * eps_klab);
    }
    return sum;
}
static double compute_4h2p_os_cpu(const double* g, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t occa2 = nocc_al*nocc_al, occb2 = nocc_be*nocc_be;
    const size_t virab = nvir_al*nvir_be;
    const size_t n = occa2 * occb2 * virab;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ikjl = tid / virab;
        size_t ab = tid % virab;
        size_t ik = ikjl / occb2, jl = ikjl % occb2;
        int i = ik/nocc_al, k = ik%nocc_al, j = jl/nocc_be, l = jl%nocc_be;
        int a = nocc_al + ab/nvir_be, b = nocc_be + ab%nvir_be;
        double eps_ijab = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        double eps_klab = eps_al[k]+eps_be[l]-eps_al[a]-eps_be[b];
        double num = g[q2s(i,a,j,b,N)] * g[q2s(i,k,j,l,N)] * g[q2s(k,a,l,b,N)];
        sum += num / (eps_ijab * eps_klab);
    }
    return sum;
}
static double compute_2h4p_ss_cpu(const double* g, const double* eps, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nocc*nvir*nvir*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        int ij = tid / ((size_t)nvir*nvir*nvir*nvir);
        size_t abcd = tid % ((size_t)nvir*nvir*nvir*nvir);
        int ab = abcd / (nvir*nvir), cd = abcd % (nvir*nvir);
        int i = ij/nocc, j = ij%nocc;
        int a = nocc + ab/nvir, b = nocc + ab%nvir, c = nocc + cd/nvir, d = nocc + cd%nvir;
        double eps_ijab = eps[i]+eps[j]-eps[a]-eps[b];
        double eps_ijcd = eps[i]+eps[j]-eps[c]-eps[d];
        double num = g[q2s(i,a,j,b,N)] * g[q2s(a,c,b,d,N)] * (g[q2s(i,c,j,d,N)] - g[q2s(i,d,j,c,N)]);
        sum += num / (eps_ijab * eps_ijcd);
    }
    return sum;
}
static double compute_2h4p_os_cpu(const double* g, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t n = (size_t)nocc_al*nocc_be*nvir_al*nvir_al*nvir_be*nvir_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        int ij = tid / ((size_t)nvir_al*nvir_al*nvir_be*nvir_be);
        size_t abcd = tid % ((size_t)nvir_al*nvir_al*nvir_be*nvir_be);
        int ac = abcd / (nvir_be*nvir_be), bd = abcd % (nvir_be*nvir_be);
        int i = ij/nocc_be, j = ij%nocc_be;
        int a = nocc_al + ac/nvir_al, c = nocc_al + ac%nvir_al;
        int b = nocc_be + bd/nvir_be, d = nocc_be + bd%nvir_be;
        double eps_ijab = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        double eps_ijcd = eps_al[i]+eps_be[j]-eps_al[c]-eps_be[d];
        double num = g[q2s(i,a,j,b,N)] * g[q2s(a,c,b,d,N)] * g[q2s(i,c,j,d,N)];
        sum += num / (eps_ijab * eps_ijcd);
    }
    return sum;
}

// --- 3h3p 6-index kernels ---
static double compute_3h3p_aaaaaa_cpu(const double* g, const double* eps, int nocc, int nvir) {
    const int N = nocc + nvir;
    const size_t n = (size_t)nocc*nocc*nocc*nvir*nvir*nvir;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijk = tid / ((size_t)nvir*nvir*nvir);
        size_t abc = tid % ((size_t)nvir*nvir*nvir);
        int ij2 = ijk / nocc, k = ijk % nocc, i = ij2/nocc, j = ij2%nocc;
        int ab2 = abc / nvir, cv = abc % nvir;
        int c = nocc + cv, a = nocc + ab2/nvir, b = nocc + ab2%nvir;
        double eps_ijab = eps[i]+eps[j]-eps[a]-eps[b];
        double eps_ikac = eps[i]+eps[k]-eps[a]-eps[c];
        double num = (g[q2s(i,a,j,b,N)]-g[q2s(i,b,j,a,N)]) * (g[q2s(i,a,k,c,N)]-g[q2s(i,c,k,a,N)]) * (g[q2s(k,c,b,j,N)]-g[q2s(k,j,b,c,N)]);
        sum += num / (eps_ijab * eps_ikac);
    }
    return sum;
}
static double compute_3h3p_aabaab_cpu(const double* g_aaaa, const double* g_aabb, const double* g_bbaa, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t n = (size_t)nocc_al*nocc_al*nocc_be*nvir_al*nvir_al*nvir_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijk = tid / ((size_t)nvir_al*nvir_al*nvir_be);
        size_t abc = tid % ((size_t)nvir_al*nvir_al*nvir_be);
        int ij2 = ijk / nocc_be, k = ijk % nocc_be, i = ij2/nocc_al, j = ij2%nocc_al;
        int ab2 = abc / nvir_be, cv = abc % nvir_be;
        int c = nocc_be + cv, a = nocc_al + ab2/nvir_al, b = nocc_al + ab2%nvir_al;
        double eps_ijab = eps_al[i]+eps_al[j]-eps_al[a]-eps_al[b];
        double eps_ikac = eps_al[i]+eps_be[k]-eps_al[a]-eps_be[c];
        double num = (g_aaaa[q2s(i,a,j,b,N)]-g_aaaa[q2s(i,b,j,a,N)]) * g_aabb[q2s(i,a,k,c,N)] * g_bbaa[q2s(k,c,b,j,N)];
        sum += num / (eps_ijab * eps_ikac);
    }
    return sum;
}
static double compute_3h3p_abaaba_cpu(const double* g_aaaa, const double* g_aabb, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t n = (size_t)nocc_al*nocc_be*nocc_al*nvir_al*nvir_be*nvir_al;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijk = tid / ((size_t)nvir_al*nvir_be*nvir_al);
        size_t abc = tid % ((size_t)nvir_al*nvir_be*nvir_al);
        int ij2 = ijk / nocc_al, k = ijk % nocc_al, i = ij2/nocc_be, j = ij2%nocc_be;
        int ab2 = abc / nvir_al, cv = abc % nvir_al;
        int c = nocc_al + cv, a = nocc_al + ab2/nvir_be, b = nocc_be + ab2%nvir_be;
        double eps_ijab = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        double eps_ikac = eps_al[i]+eps_al[k]-eps_al[a]-eps_al[c];
        double num = g_aabb[q2s(i,a,j,b,N)] * (g_aaaa[q2s(i,a,k,c,N)]-g_aaaa[q2s(i,c,k,a,N)]) * g_aabb[q2s(k,c,b,j,N)];
        sum += num / (eps_ijab * eps_ikac);
    }
    return sum;
}
static double compute_3h3p_abbabb_cpu(const double* g_aabb, const double* g_bbbb, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t n = (size_t)nocc_al*nocc_be*nocc_be*nvir_al*nvir_be*nvir_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijk = tid / ((size_t)nvir_al*nvir_be*nvir_be);
        size_t abc = tid % ((size_t)nvir_al*nvir_be*nvir_be);
        int ij2 = ijk / nocc_be, k = ijk % nocc_be, i = ij2/nocc_be, j = ij2%nocc_be;
        int ab2 = abc / nvir_be, cv = abc % nvir_be;
        int c = nocc_be + cv, a = nocc_al + ab2/nvir_be, b = nocc_be + ab2%nvir_be;
        double eps_ijab = eps_al[i]+eps_be[j]-eps_al[a]-eps_be[b];
        double eps_ikac = eps_al[i]+eps_be[k]-eps_al[a]-eps_be[c];
        double num = g_aabb[q2s(i,a,j,b,N)] * g_aabb[q2s(i,a,k,c,N)] * (g_bbbb[q2s(k,c,b,j,N)]-g_bbbb[q2s(k,j,b,c,N)]);
        sum += num / (eps_ijab * eps_ikac);
    }
    return sum;
}
static double compute_3h3p_abbbaa_cpu(const double* g_aabb, const double* g_bbaa, const double* eps_al, const double* eps_be, int nocc_al, int nvir_al, int nocc_be, int nvir_be) {
    const int N = nocc_al + nvir_al;
    const size_t n = (size_t)nocc_al*nocc_be*nocc_be*nvir_be*nvir_al*nvir_al;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(dynamic)
    for (size_t tid = 0; tid < n; tid++) {
        size_t ijk = tid / ((size_t)nvir_be*nvir_al*nvir_al);
        size_t abc = tid % ((size_t)nvir_be*nvir_al*nvir_al);
        int ij2 = ijk / nocc_be, k = ijk % nocc_be, i = ij2/nocc_be, j = ij2%nocc_be;
        int ab2 = abc / nvir_al, cv = abc % nvir_al;
        int c = nocc_al + cv, a = nocc_be + ab2/nvir_al, b = nocc_al + ab2%nvir_al;
        double eps_ijab = eps_al[i]+eps_be[j]-eps_be[a]-eps_al[b];
        double eps_ikac = eps_al[i]+eps_be[k]-eps_be[a]-eps_al[c];
        double num = g_aabb[q2s(i,b,j,a,N)] * g_aabb[q2s(i,c,k,a,N)] * g_bbaa[q2s(k,j,b,c,N)];
        sum += (-1.0) * num / (eps_ijab * eps_ikac);
    }
    return sum;
}

// ============================================================
//  CPU mirrors of the ao2mo.cuh axis-sequential GEMM transforms.
//  These issue the identical GEMM/transpose sequence as the GPU versions
//  (same operand shapes, strides, and ping-pong buffer schedule), so the
//  CPU and GPU backends execute the same algorithm and only the hardware
//  differs. GEMMs run through Eigen (OpenMP-threaded).
// ============================================================

// Mirror of transform_eri_ao2mo_dgemm_full (ao2mo.cuh): destructive in-place
// 4-axis AO->MO transformation of the full M^4 tensor (same-spin coefficient
// matrix on all four axes). Ping-pongs between d_eri_ao and d_eri_mo.
static void transform_eri_ao2mo_dgemm_full_cpu(
    double* d_eri_ao, double* d_eri_mo,
    const double* d_coefficient_matrix, const int num_basis)
{
    const size_t num_basis_2 = (size_t)num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, (int)num_basis_3, num_basis,
        d_coefficient_matrix, num_basis,
        d_eri_ao, (int)num_basis_3,
        d_eri_mo, (int)num_basis_3);
    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, (int)num_basis_2, num_basis,
        d_coefficient_matrix, num_basis, 0,
        d_eri_mo, (int)num_basis_2, (long long int)num_basis_3,
        d_eri_ao, (int)num_basis_2, (long long int)num_basis_3,
        num_basis);

    cpu_dgeam_transpose_row_major((int)num_basis_2, (int)num_basis_2, d_eri_ao, d_eri_mo);

    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, (int)num_basis_3, num_basis,
        d_coefficient_matrix, num_basis,
        d_eri_mo, (int)num_basis_3,
        d_eri_ao, (int)num_basis_3);
    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, (int)num_basis_2, num_basis,
        d_coefficient_matrix, num_basis, 0,
        d_eri_ao, (int)num_basis_2, (long long int)num_basis_3,
        d_eri_mo, (int)num_basis_2, (long long int)num_basis_3,
        num_basis);
}

// Mirror of transform_eri_ao2mo_dgemm_ovov_keep_ao (ao2mo.cuh): non-destructive
// compact (ia|jb) transformation. Only the first GEMM reads the full M^4 AO
// tensor; the remaining stages ping-pong inside num_occ * N^3 buffers
// (d_scratch, d_eri_mo). Returns d_scratch, which holds the compact ovov
// block in ovov2seq layout.
static double* transform_eri_ao2mo_dgemm_ovov_keep_ao_cpu(
    const double* d_eri_src, double* d_scratch, double* d_eri_mo,
    const double* d_coefficient_matrix,
    const int num_occ, const int num_vir)
{
    const size_t num_basis = (size_t)num_occ + num_vir;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, (int)num_basis_3, num_basis,
        d_coefficient_matrix, num_basis,
        d_eri_src, (int)num_basis_3,
        d_eri_mo, (int)num_basis_3);
    cpu_dgeam_transpose_row_major(num_occ * num_basis, (int)(num_basis * num_basis), d_eri_mo, d_scratch);
    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, (int)(num_basis_2 * num_occ), num_basis,
        d_coefficient_matrix, num_basis,
        d_scratch, (int)(num_basis_2 * num_occ),
        d_eri_mo, (int)(num_basis_2 * num_occ));

    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_basis, num_basis,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_basis, (long long int)(num_basis_2 * num_occ),
        d_scratch, num_occ * num_basis, (long long int)(num_vir * num_occ * num_basis),
        num_occ);
    cpu_dgeam_transpose_row_major(num_occ * num_vir, num_occ * num_basis, d_scratch, d_eri_mo);
    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_vir, num_basis,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_vir, (long long int)(num_basis * num_occ * num_vir),
        d_scratch, num_occ * num_vir, (long long int)(num_vir * num_occ * num_vir),
        num_occ);

    return d_scratch;
}

// Mirror of transform_eri_ao2mo_dgemm_ovov_os_keep_ao (ao2mo.cuh):
// non-destructive compact (i_alpha a_alpha | j_beta b_beta) transformation.
// Returns d_scratch (compact ovov block in ovov2seq_aabb layout).
static double* transform_eri_ao2mo_dgemm_ovov_os_keep_ao_cpu(
    const double* d_eri_src, double* d_scratch, double* d_eri_mo,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t num_basis = (size_t)num_occ_al + num_vir_al;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_al, (int)num_basis_3, num_basis,
        d_coefficient_matrix_al, num_basis,
        d_eri_src, (int)num_basis_3,
        d_eri_mo, (int)num_basis_3);
    cpu_dgeam_transpose_row_major(num_occ_al * num_basis, (int)(num_basis * num_basis), d_eri_mo, d_scratch);
    cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_be, (int)(num_basis_2 * num_occ_al), num_basis,
        d_coefficient_matrix_be, num_basis,
        d_scratch, (int)(num_basis_2 * num_occ_al),
        d_eri_mo, (int)(num_basis_2 * num_occ_al));

    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_be, num_occ_al * num_basis, num_basis,
        d_coefficient_matrix_be + num_occ_be, num_basis, 0,
        d_eri_mo, num_occ_al * num_basis, (long long int)(num_basis_2 * num_occ_al),
        d_scratch, num_occ_al * num_basis, (long long int)(num_vir_be * num_occ_al * num_basis),
        num_occ_be);
    cpu_dgeam_transpose_row_major(num_occ_be * num_vir_be, num_occ_al * num_basis, d_scratch, d_eri_mo);
    cpu_dgemm_strided_batched_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_al, num_occ_be * num_vir_be, num_basis,
        d_coefficient_matrix_al + num_occ_al, num_basis, 0,
        d_eri_mo, num_occ_be * num_vir_be, (long long int)(num_basis * num_occ_be * num_vir_be),
        d_scratch, num_occ_be * num_vir_be, (long long int)(num_vir_al * num_occ_be * num_vir_be),
        num_occ_al);

    return d_scratch;
}

// CPU mirrors of the compute_ump2_energy_contrib_ss / _os CUDA kernels.
// Same summation, OpenMP reduction instead of warp shuffle + atomicAdd.
static void compute_ump2_energy_contrib_ss_cpu(
    double* g_energy_second,
    const double* g_eri_mo, const double* g_eps,
    const int num_occupied, const int num_virtual, const int num_frozen = 0)
{
    const int active_occ = num_occupied - num_frozen;
    const size_t total = (size_t)active_occ * num_virtual * (size_t)active_occ * num_virtual;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t seq = 0; seq < total; seq++) {
        const size_t ia = seq / ((size_t)active_occ * num_virtual);
        const size_t jb = seq % ((size_t)active_occ * num_virtual);
        const int i = (int)(ia / num_virtual) + num_frozen;
        const int a = (int)(ia % num_virtual);
        const int j = (int)(jb / num_virtual) + num_frozen;
        const int b = (int)(jb % num_virtual);

        const double iajb = g_eri_mo[ovov2seq(i, a, j, b, num_occupied, num_virtual)];
        const double jaib = g_eri_mo[ovov2seq(j, a, i, b, num_occupied, num_virtual)];
        sum += iajb * (iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
    }
    *g_energy_second += sum * 0.5;
}

static void compute_ump2_energy_contrib_os_cpu(
    double* g_energy_second, const double* g_eri_mo,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occupied_al, const int num_virtual_al,
    const int num_occupied_be, const int num_virtual_be,
    const int num_frozen = 0)
{
    const int active_occ_al = num_occupied_al - num_frozen;
    const int active_occ_be = num_occupied_be - num_frozen;
    const size_t total = (size_t)active_occ_al * num_virtual_al * (size_t)active_occ_be * num_virtual_be;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (size_t seq = 0; seq < total; seq++) {
        const size_t ia = seq / ((size_t)active_occ_be * num_virtual_be);
        const size_t jb = seq % ((size_t)active_occ_be * num_virtual_be);
        const int i = (int)(ia / num_virtual_al) + num_frozen;
        const int a = (int)(ia % num_virtual_al);
        const int j = (int)(jb / num_virtual_be) + num_frozen;
        const int b = (int)(jb % num_virtual_be);

        const double iajb = g_eri_mo[ovov2seq_aabb(i, a, j, b, num_occupied_al, num_virtual_al, num_occupied_be, num_virtual_be)];
        sum += (iajb * iajb) / (g_eps_al[i] + g_eps_be[j] - g_eps_al[num_occupied_al + a] - g_eps_be[num_occupied_be + b]);
    }
    *g_energy_second += sum;
}

} // namespace cpu_kernels

namespace gansu {

static inline dim3 make_2d_grid_from_1d_blocks(const size_t num_blocks_1d, const cudaDeviceProp& prop)
{
    if (num_blocks_1d == 0) {
        return dim3(1, 1, 1);
    }

    const size_t max_x = static_cast<size_t>(prop.maxGridSize[0]);
    const size_t max_y = static_cast<size_t>(prop.maxGridSize[1]);

    if (num_blocks_1d <= max_x) {
        return dim3(static_cast<unsigned int>(num_blocks_1d), 1, 1);
    }

    const size_t grid_x = max_x;
    const size_t grid_y = (num_blocks_1d + grid_x - 1) / grid_x;
    if (grid_y > max_y) {
        THROW_EXCEPTION("Error: Too many blocks for the 2D grid size.");
    }

    return dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1);
}


void transform_ump3_full_mo_eris(
    double* d_eri_ao,
    double* d_g_aaaa_full,
    double* d_g_aabb_full,
    double* d_g_bbaa_full,
    double* d_g_bbbb_full,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const size_t num_basis_4,
    const int num_basis)
{
    // Reuse d_g_bbbb_full as temporary storage for the original AO ERIs.
    double* d_eri_tmp = d_g_bbbb_full;
    cudaMemcpy(d_eri_tmp, d_eri_ao, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);

    transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_aaaa_full, d_coefficient_matrix_al, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_aabb_full, d_coefficient_matrix_al, d_coefficient_matrix_be, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_bbaa_full, d_coefficient_matrix_be, d_coefficient_matrix_al, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_bbbb_full, d_coefficient_matrix_be, num_basis);

    cudaDeviceSynchronize();
}


void transform_ump3_single_mo_eri(
    double* d_eri_ao,
    double* d_g_full,
    double* d_eri_tmp,
    const double* d_coefficient_matrix_1,
    const double* d_coefficient_matrix_2,
    const size_t num_basis_4,
    const int num_basis,
    const bool same_spin)
{
    cudaMemcpy(d_eri_tmp, d_eri_ao, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);

    if (!gansu::gpu::gpu_available()) {
        // CPU: use the same Kronecker product method as eri_stored.cu's transform_ao_eri_to_mo_eri_full
        // which already has a CPU fallback. This ensures identical index convention.
        if (same_spin) {
            transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix_1, num_basis, d_g_full);
        } else {
            // For different spins, do 4-stage with correct Kronecker convention:
            // D(mu*N+nu, p*N+q) = C1(mu,p)*C2(nu,q)
            // G = D^T * A * D  →  g_full(p*N+q, r*N+s)
            const int N = num_basis;
            const size_t N2 = (size_t)N * N;
            const double* C1 = d_coefficient_matrix_1;
            const double* C2 = d_coefficient_matrix_2;
            const double* eri = d_eri_tmp;

            // Build D = kron(C1, C2): D(mu*N+nu, p*N+q) = C1(mu,p)*C2(nu,q)
            double* D = nullptr;
            gansu::tracked_cudaMalloc(&D, N2 * N2 * sizeof(double));
            #pragma omp parallel for
            for (size_t idx = 0; idx < N2 * N2; idx++) {
                size_t P = idx / N2, R = idx % N2;
                int mu = P / N, nu = P % N;
                int p = R / N, q = R % N;
                D[idx] = C1[mu*N+p] * C2[nu*N+q];
            }

            // G = D^T * A * D  (A and G are N²×N² matrices)
            // T = A * D
            double* T = nullptr;
            gansu::tracked_cudaMalloc(&T, N2 * N2 * sizeof(double));
            gansu::gpu::matrixMatrixProductRect(eri, D, T, (int)N2, (int)N2, (int)N2);
            // G = D^T * T
            gansu::gpu::matrixMatrixProductRect(D, T, d_g_full, (int)N2, (int)N2, (int)N2, true, false);

            gansu::tracked_cudaFree(D);
            gansu::tracked_cudaFree(T);
        }
    } else {
        if (same_spin) {
            transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_full, d_coefficient_matrix_1, num_basis);
        } else {
            transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_full, d_coefficient_matrix_1, d_coefficient_matrix_2, num_basis);
        }
    }

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}






//*
__global__ void compute_ump2_energy_contrib_ss(
    double* g_energy_second,
    const double* g_eri_mo, const double* g_eps,
    const int num_occupied, const int num_virtual, const int num_frozen = 0)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_tmp = 0;
    }
    __syncthreads();

    double tmp = 0.0;
    const int active_occ = num_occupied - num_frozen;
    const size_t seq = (((size_t)blockDim.x * blockDim.y) * blockIdx.x) + blockDim.x * threadIdx.y + threadIdx.x;
    if (seq < (size_t)active_occ * num_virtual * (size_t)active_occ * num_virtual) {
        const int ia = seq / (active_occ * num_virtual);
        const int jb = seq % (active_occ * num_virtual);
        const int i = ia / num_virtual + num_frozen;
        const int a = ia % num_virtual;
        const int j = jb / num_virtual + num_frozen;
        const int b = jb % num_virtual;

        const double iajb = g_eri_mo[ovov2seq(i, a, j, b, num_occupied, num_virtual)];
        const double jaib = g_eri_mo[ovov2seq(j, a, i, b, num_occupied, num_virtual)];
        tmp = iajb * (iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += __shfl_down_sync(FULLMASK, tmp, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_tmp, tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_second, s_tmp * 0.5);
    }
}
/**/


//*
__global__ void compute_ump2_energy_contrib_os(
    double* g_energy_second, const double* g_eri_mo,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occupied_al, const int num_virtual_al,
    const int num_occupied_be, const int num_virtual_be,
    const int num_frozen = 0)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_tmp = 0;
    }
    __syncthreads();

    double tmp = 0.0;
    const int active_occ_al = num_occupied_al - num_frozen;
    const int active_occ_be = num_occupied_be - num_frozen;
    const size_t seq = (((size_t)blockDim.x * blockDim.y) * blockIdx.x) + blockDim.x * threadIdx.y + threadIdx.x;
    if (seq < (size_t)active_occ_al * num_virtual_al * (size_t)active_occ_be * num_virtual_be) {
        const int ia = seq / (active_occ_be * num_virtual_be);
        const int jb = seq % (active_occ_be * num_virtual_be);
        const int i = ia / num_virtual_al + num_frozen;
        const int a = ia % num_virtual_al;
        const int j = jb / num_virtual_be + num_frozen;
        const int b = jb % num_virtual_be;

        const double iajb = g_eri_mo[ovov2seq_aabb(i, a, j, b, num_occupied_al, num_virtual_al, num_occupied_be, num_virtual_be)];
        tmp = (iajb * iajb) / (g_eps_al[i] + g_eps_be[j] - g_eps_al[num_occupied_al + a] - g_eps_be[num_occupied_be + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += __shfl_down_sync(FULLMASK, tmp, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_tmp, tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_second, s_tmp);
    }
}
/**/





double ump2_from_aoeri_via_required_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis,
    const int num_occ_al,
    const int num_occ_be,
    const int num_frozen = 0,
    double* out_E_os = nullptr,
    double* out_E_ss = nullptr)
{
    double* d_eri_scratch = nullptr;
    double* d_eri_tmp2 = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const int max_num_occ = std::max(num_occ_al, num_occ_be);
    // The AO ERIs (d_eri_ao) are only read, never overwritten: only the first
    // GEMM of each spin-pattern transformation touches the full M^4 tensor and
    // the remaining stages ping-pong inside num_occ * N^3 regions. No M^4
    // working copy is therefore needed; the peak memory is one M^4 tensor
    // (the AO ERIs themselves) plus two max_num_occ * N^3 buffers.
    tracked_cudaMalloc(&d_eri_scratch, sizeof(double) * max_num_occ * num_basis_2 * num_basis);
    tracked_cudaMalloc(&d_eri_tmp2, sizeof(double) * max_num_occ * num_basis_2 * num_basis);
    if (!d_eri_scratch) { THROW_EXCEPTION("cudaMalloc failed for d_eri_scratch."); }
    if (!d_eri_tmp2) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp_2."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    double* d_second_energy = nullptr;
    double* d_energy_ss = nullptr;
    double* d_energy_os = nullptr;
    tracked_cudaMalloc(&d_second_energy, sizeof(double));
    tracked_cudaMalloc(&d_energy_ss, sizeof(double));
    tracked_cudaMalloc(&d_energy_os, sizeof(double));
    cudaMemset(d_second_energy, 0, sizeof(double));
    cudaMemset(d_energy_ss, 0, sizeof(double));
    cudaMemset(d_energy_os, 0, sizeof(double));

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;

    if (!gpu::gpu_available()) {
        // ---------- CPU path: same algorithm as the GPU path ----------
        // Runs the identical keep_ao GEMM-chain transformations and energy
        // contractions as the GPU branch below (same buffer schedule:
        // the AO ERIs are only read, two num_occ * N^3 ping-pong buffers),
        // so GPU-vs-CPU comparisons reflect hardware, not algorithm,
        // differences.
        //
        // Historical note: the CPU path previously used a brute-force
        // O(o^2 v^2 N^4) inline evaluation that recomputed every (ia|jb)
        // directly from the AO ERIs. It is preserved below (commented out)
        // for reference.
#if 0
        // ---------- OLD CPU implementation: brute-force inline ----------
        const int N = num_basis;
        const double* Ca = d_coefficient_matrix_al;  // C_alpha(AO, MO) row-major
        const double* Cb = d_coefficient_matrix_be;  // C_beta(AO, MO) row-major
        const double* ea = d_orbital_energies_al;
        const double* eb = d_orbital_energies_be;
        const double* eri = d_eri_ao;  // AO ERI: eri[mu*N³+nu*N²+la*N+si]

        // Helper: compute (ia|jb) with given C matrices for bra/ket
        auto mo_eri = [&](const double* C1, const double* C2, int i, int a, int j, int b) -> double {
            double val = 0.0;
            for (int mu = 0; mu < N; mu++) {
                double c1_mi = C1[mu*N + i];
                for (int nu = 0; nu < N; nu++) {
                    double c1c2 = c1_mi * C1[nu*N + a];
                    for (int la = 0; la < N; la++) {
                        double c1c2c3 = c1c2 * C2[la*N + j];
                        for (int si = 0; si < N; si++) {
                            val += c1c2c3 * C2[si*N + b] * eri[mu*N*N*N + nu*N*N + la*N + si];
                        }
                    }
                }
            }
            return val;
        };

        // alpha-alpha: 0.5 * sum (ia|jb)[(ia|jb)-(ja|ib)] / denom
        {
            double sum_aa = 0.0;
            #pragma omp parallel for reduction(+:sum_aa) schedule(dynamic)
            for (int i = num_frozen; i < num_occ_al; i++)
                for (int a = num_occ_al; a < N; a++)
                    for (int j = num_frozen; j < num_occ_al; j++)
                        for (int b = num_occ_al; b < N; b++) {
                            double iajb = mo_eri(Ca, Ca, i, a, j, b);
                            double jaib = mo_eri(Ca, Ca, j, a, i, b);
                            sum_aa += iajb*(iajb-jaib) / (ea[i]+ea[j]-ea[a]-ea[b]);
                        }
            *d_second_energy += sum_aa * 0.5;
            *d_energy_ss += sum_aa * 0.5;
        }

        // beta-beta: 0.5 * sum (ia|jb)[(ia|jb)-(ja|ib)] / denom
        {
            double sum_bb = 0.0;
            #pragma omp parallel for reduction(+:sum_bb) schedule(dynamic)
            for (int i = num_frozen; i < num_occ_be; i++)
                for (int a = num_occ_be; a < N; a++)
                    for (int j = num_frozen; j < num_occ_be; j++)
                        for (int b = num_occ_be; b < N; b++) {
                            double iajb = mo_eri(Cb, Cb, i, a, j, b);
                            double jaib = mo_eri(Cb, Cb, j, a, i, b);
                            sum_bb += iajb*(iajb-jaib) / (eb[i]+eb[j]-eb[a]-eb[b]);
                        }
            *d_second_energy += sum_bb * 0.5;
            *d_energy_ss += sum_bb * 0.5;
        }

        // alpha-beta: sum (ia_alpha|jb_beta)^2 / denom
        {
            double sum_ab = 0.0;
            #pragma omp parallel for reduction(+:sum_ab) schedule(dynamic)
            for (int i = num_frozen; i < num_occ_al; i++)
                for (int a = num_occ_al; a < N; a++)
                    for (int j = num_frozen; j < num_occ_be; j++)
                        for (int b = num_occ_be; b < N; b++) {
                            double iajb = mo_eri(Ca, Cb, i, a, j, b);
                            sum_ab += (iajb*iajb) / (ea[i]+eb[j]-ea[a]-eb[b]);
                        }
            *d_second_energy += sum_ab;
            *d_energy_os += sum_ab;
        }
#endif // OLD brute-force CPU implementation

        auto cpu_elapsed_ms = [](const std::chrono::steady_clock::time_point& t0) {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
        };

        // Compute alpha-alpha energy contribution
        {
            std::string str = "Computing 1st term... ";
            PROFILE_ELAPSED_TIME(str);
            const auto t0 = std::chrono::steady_clock::now();

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_aa = cpu_kernels::transform_eri_ao2mo_dgemm_ovov_keep_ao_cpu(
                d_eri_ao, d_eri_scratch, d_eri_tmp2,
                d_coefficient_matrix_al, num_occ_al, num_vir_al);

            cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_second_energy, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occ_al, num_vir_al, num_frozen);
            cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_energy_ss, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occ_al, num_vir_al, num_frozen);

            printf("alpha-alpha: %.2f [ms]\n", cpu_elapsed_ms(t0));
        }

        // Compute beta-beta energy contribution
        {
            std::string str = "Computing 2nd term... ";
            PROFILE_ELAPSED_TIME(str);
            const auto t0 = std::chrono::steady_clock::now();

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_bb = cpu_kernels::transform_eri_ao2mo_dgemm_ovov_keep_ao_cpu(
                d_eri_ao, d_eri_scratch, d_eri_tmp2,
                d_coefficient_matrix_be, num_occ_be, num_vir_be);

            cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_second_energy, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occ_be, num_vir_be, num_frozen);
            cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_energy_ss, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occ_be, num_vir_be, num_frozen);

            printf("beta-beta: %.2f [ms]\n", cpu_elapsed_ms(t0));
        }

        // Compute alpha-beta energy contribution
        {
            std::string str = "Computing 3rd term... ";
            PROFILE_ELAPSED_TIME(str);
            const auto t0 = std::chrono::steady_clock::now();

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_ab = cpu_kernels::transform_eri_ao2mo_dgemm_ovov_os_keep_ao_cpu(
                d_eri_ao, d_eri_scratch, d_eri_tmp2,
                d_coefficient_matrix_al, d_coefficient_matrix_be,
                num_occ_al, num_vir_al, num_occ_be, num_vir_be);

            cpu_kernels::compute_ump2_energy_contrib_os_cpu(d_second_energy, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_frozen);
            cpu_kernels::compute_ump2_energy_contrib_os_cpu(d_energy_os, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_frozen);

            printf("alpha-beta: %.2f [ms]\n", cpu_elapsed_ms(t0));
        }
    } else {
        // ---------- GPU path ----------
        float time_aa, time_bb, time_ab;
        cudaEvent_t begin, end;
        cudaEventCreate(&begin);
        cudaEventCreate(&end);

        cudaEventRecord(begin);
        // Compute alpha-alpha energy contribution
        {
            std::string str = "Computing 1st term... ";
            PROFILE_ELAPSED_TIME(str);

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_aa = transform_eri_ao2mo_dgemm_ovov_keep_ao(d_eri_ao, d_eri_scratch, d_eri_tmp2, d_coefficient_matrix_al, num_occ_al, num_vir_al);
            cudaDeviceSynchronize();

            const int active_occ_al = num_occ_al - num_frozen;
            const size_t total = (size_t)active_occ_al * num_vir_al * active_occ_al * num_vir_al;
            const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
            const dim3 blocks(num_blocks);
            const dim3 threads(num_threads_per_warp, num_warps_per_block);

            compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occ_al, num_vir_al, num_frozen);
            compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_energy_ss, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occ_al, num_vir_al, num_frozen);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time_aa, begin, end);
        printf("alpha-alpha: %.2f [ms]\n", time_aa);

        cudaEventRecord(begin);
        // Compute beta-beta energy contribution
        {
            std::string str = "Computing 2nd term... ";
            PROFILE_ELAPSED_TIME(str);

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_bb = transform_eri_ao2mo_dgemm_ovov_keep_ao(d_eri_ao, d_eri_scratch, d_eri_tmp2, d_coefficient_matrix_be, num_occ_be, num_vir_be);
            cudaDeviceSynchronize();

            const int active_occ_be = num_occ_be - num_frozen;
            const size_t total = (size_t)active_occ_be * num_vir_be * active_occ_be * num_vir_be;
            const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
            const dim3 blocks(num_blocks);
            const dim3 threads(num_threads_per_warp, num_warps_per_block);

            compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occ_be, num_vir_be, num_frozen);
            compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_energy_ss, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occ_be, num_vir_be, num_frozen);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time_bb, begin, end);
        printf("beta-beta: %.2f [ms]\n", time_bb);

        cudaEventRecord(begin);
        // Compute alpha-beta energy contribution
        {
            std::string str = "Computing 3rd term... ";
            PROFILE_ELAPSED_TIME(str);

            cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

            double* d_eri_mo_ovov_ab = transform_eri_ao2mo_dgemm_ovov_os_keep_ao(d_eri_ao, d_eri_scratch, d_eri_tmp2, d_coefficient_matrix_al, d_coefficient_matrix_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
            cudaDeviceSynchronize();

            const int active_occ_al2 = num_occ_al - num_frozen;
            const int active_occ_be2 = num_occ_be - num_frozen;
            const size_t total = (size_t)active_occ_al2 * num_vir_al * active_occ_be2 * num_vir_be;
            const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
            const dim3 blocks(num_blocks);
            const dim3 threads(num_threads_per_warp, num_warps_per_block);

            compute_ump2_energy_contrib_os<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_frozen);
            compute_ump2_energy_contrib_os<<<blocks, threads>>>(d_energy_os, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_frozen);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time_ab, begin, end);
        printf("alpha-beta: %.2f [ms]\n", time_ab);

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
    }

    double h_second_energy = 0.0, h_E_ss = 0.0, h_E_os = 0.0;
    cudaMemcpy(&h_second_energy, d_second_energy, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_E_ss, d_energy_ss, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_E_os, d_energy_os, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "UMP2 correlation energy: " << std::setprecision(12) << h_second_energy << std::endl;

    if (out_E_os) *out_E_os = h_E_os;
    if (out_E_ss) *out_E_ss = h_E_ss;

    tracked_cudaFree(d_eri_scratch);
    tracked_cudaFree(d_eri_tmp2);
    tracked_cudaFree(d_second_energy);
    tracked_cudaFree(d_energy_ss);
    tracked_cudaFree(d_energy_os);

    return h_second_energy;
}




real_t ERI_Stored_UHF::compute_mp2_energy()
{
    PROFILE_FUNCTION();

    if (!is_ao_eri_valid()) {
        THROW_EXCEPTION("AO ERIs are no longer available: the UMP3 cascading integral transformation destructively reused the AO ERI buffer as scratch.");
    }

    const int num_basis = uhf_.get_num_basis();
    const int num_occ_al = uhf_.get_num_alpha_spins();
    const int num_occ_be = uhf_.get_num_beta_spins();
    const int num_frozen = uhf_.get_num_frozen_core();

    DeviceHostMatrix<real_t>& coefficient_matrix_al = uhf_.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& coefficient_matrix_be = uhf_.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& orbital_energies_al = uhf_.get_orbital_energies_a();
    DeviceHostMemory<real_t>& orbital_energies_be = uhf_.get_orbital_energies_b();

    const real_t E_UMP2 = ump2_from_aoeri_via_required_moeri(
        eri_matrix_.device_ptr(),
        coefficient_matrix_al.device_ptr(),
        coefficient_matrix_be.device_ptr(),
        orbital_energies_al.device_ptr(),
        orbital_energies_be.device_ptr(),
        num_basis,
        num_occ_al,
        num_occ_be,
        num_frozen
    );

    std::cout << "UMP2 energy test" << std::endl;

    return E_UMP2;
}

// mode: 0 = SCS, 1 = SOS
static real_t compute_scaled_ump2_impl(UHF& uhf, real_t* d_eri, int mode) {
    const int num_basis = uhf.get_num_basis();
    const int num_occ_al = uhf.get_num_alpha_spins();
    const int num_occ_be = uhf.get_num_beta_spins();
    const int num_frozen = uhf.get_num_frozen_core();

    DeviceHostMatrix<real_t>& Ca = uhf.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& Cb = uhf.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& ea = uhf.get_orbital_energies_a();
    DeviceHostMemory<real_t>& eb = uhf.get_orbital_energies_b();

    double E_os = 0.0, E_ss = 0.0;
    ump2_from_aoeri_via_required_moeri(
        d_eri, Ca.device_ptr(), Cb.device_ptr(),
        ea.device_ptr(), eb.device_ptr(),
        num_basis, num_occ_al, num_occ_be, num_frozen,
        &E_os, &E_ss);

    std::cout << "  E_os = " << std::setprecision(12) << E_os
              << "  E_ss = " << E_ss
              << "  E_UMP2 = " << (E_os + E_ss) << std::endl;

    if (mode == 0) {
        real_t E_SCS = (6.0/5.0) * E_os + (1.0/3.0) * E_ss;
        std::cout << "  SCS-UMP2 energy (c_os=6/5, c_ss=1/3): " << E_SCS << std::endl;
        return E_SCS;
    } else {
        real_t E_SOS = 1.3 * E_os;
        std::cout << "  SOS-UMP2 energy (c_os=1.3): " << E_SOS << std::endl;
        return E_SOS;
    }
}

real_t ERI_Stored_UHF::compute_scs_mp2_energy() {
    if (!is_ao_eri_valid()) {
        THROW_EXCEPTION("AO ERIs are no longer available: the UMP3 cascading integral transformation destructively reused the AO ERI buffer as scratch.");
    }
    return compute_scaled_ump2_impl(uhf_, eri_matrix_.device_ptr(), 0);
}

real_t ERI_Stored_UHF::compute_sos_mp2_energy() {
    if (!is_ao_eri_valid()) {
        THROW_EXCEPTION("AO ERIs are no longer available: the UMP3 cascading integral transformation destructively reused the AO ERI buffer as scratch.");
    }
    return compute_scaled_ump2_impl(uhf_, eri_matrix_.device_ptr(), 1);
}










__global__ void compute_4h2p_ss(
    double* g_energy_4h2p, 
    const double* g_int2e_aaaa, const double* g_eps_al, 
    const int num_occ_al, const int num_vir_al) 
{
    __shared__ double s_energy_4h2p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_4h2p = 0;
    }
    __syncthreads();

    const size_t tid_4h2p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t serial_4h2p = num_occ_al_2 * num_occ_al_2 * num_vir_al_2;

    double energy_4h2p = 0.0;
    if (tid_4h2p < serial_4h2p) {
        const size_t ijkl = tid_4h2p / num_vir_al_2;
        const int ab = tid_4h2p % num_vir_al_2;
        const int ij = ijkl / num_occ_al_2;
        const int kl = ijkl % num_occ_al_2;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int k = kl / num_occ_al;
        const int l = kl % num_occ_al;
        const int a = num_occ_al + (ab / num_vir_al);
        const int b = num_occ_al + (ab % num_vir_al);

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_klab = g_eps_al[k] + g_eps_al[l] - g_eps_al[a] - g_eps_al[b];
        const double numerator = 
            g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] * \
            g_int2e_aaaa[q2s(i, k, j, l, num_orbitals)] * \
            (g_int2e_aaaa[q2s(k, a, l, b, num_orbitals)] - g_int2e_aaaa[q2s(k, b, l, a, num_orbitals)]);
        energy_4h2p = numerator / (eps_ijab * eps_klab);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_4h2p += __shfl_down_sync(FULLMASK, energy_4h2p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_4h2p, energy_4h2p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_4h2p, s_energy_4h2p);
    }
}


__global__ void compute_4h2p_os(
    double* g_energy_4h2p,
    const double* g_int2e_aabb, 
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_4h2p;
    if (threadIdx.x == 0) {
        s_energy_4h2p = 0.0;
    }
    __syncthreads();

    //const size_t tid_4h2p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tid_4h2p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;

    const size_t occa2 = num_occ_al * num_occ_al;
    const size_t occb2 = num_occ_be * num_occ_be;
    const size_t virab = num_vir_al * num_vir_be;
    const size_t serial_4h2p_os = occa2 * occb2 * virab;

    double energy_4h2p = 0.0;
    if (tid_4h2p < serial_4h2p_os) {
        const size_t ikjl = tid_4h2p / virab;
        const size_t ab   = tid_4h2p % virab;
        const size_t ik = ikjl / occb2;
        const size_t jl = ikjl % occb2;
        const int i = ik / num_occ_al;
        const int k = ik % num_occ_al;
        const int j = jl / num_occ_be;
        const int l = jl % num_occ_be;
        const int a = num_occ_al + (ab / num_vir_be);
        const int b = num_occ_be + (ab % num_vir_be);

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_klab = g_eps_al[k] + g_eps_be[l] - g_eps_al[a] - g_eps_be[b];
        const double numerator =
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(i, k, j, l, num_orbitals)] *
            g_int2e_aabb[q2s(k, a, l, b, num_orbitals)];

        energy_4h2p = numerator / (eps_ijab * eps_klab);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_4h2p += __shfl_down_sync(FULLMASK, energy_4h2p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_4h2p, energy_4h2p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_4h2p, s_energy_4h2p);
    }
}


__global__ void compute_2h4p_ss(
    double* g_energy_2h4p, 
    const double* g_int2e_aaaa, const double* g_eps_al, 
    const int num_occ_al, const int num_vir_al) 
{
    __shared__ double s_energy_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_2h4p = 0;
    }
    __syncthreads();

    const size_t tid_2h4p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t serial_2h4p = num_occ_al_2 * num_vir_al_2 * num_vir_al_2;

    double energy_2h4p = 0.0;
    if (tid_2h4p < serial_2h4p) {
        const int ij = tid_2h4p / (num_vir_al_2 * num_vir_al_2);
        const size_t abcd = tid_2h4p % (num_vir_al_2 * num_vir_al_2);
        const int ab = abcd / num_vir_al_2;
        const int cd = abcd % num_vir_al_2;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int a = num_occ_al + (ab / num_vir_al);
        const int b = num_occ_al + (ab % num_vir_al);
        const int c = num_occ_al + (cd / num_vir_al);
        const int d = num_occ_al + (cd % num_vir_al);

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ijcd = g_eps_al[i] + g_eps_al[j] - g_eps_al[c] - g_eps_al[d];
        const double numerator = 
            g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] * \
            g_int2e_aaaa[q2s(a, c, b, d, num_orbitals)] * \
            (g_int2e_aaaa[q2s(i, c, j, d, num_orbitals)] - g_int2e_aaaa[q2s(i, d, j, c, num_orbitals)]);

        energy_2h4p = numerator / (eps_ijab * eps_ijcd);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_2h4p += __shfl_down_sync(FULLMASK, energy_2h4p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_2h4p, energy_2h4p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_2h4p, s_energy_2h4p);
    }
}



__global__ void compute_2h4p_os(
    double* g_energy_2h4p,
    const double* g_int2e_aabb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_2h4p = 0;
    }
    __syncthreads();

    const size_t tid_2h4p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;
    const size_t serial_2h4p = num_occ_al * num_occ_be * num_vir_al_2 * num_vir_be_2;

    double energy_2h4p = 0.0;
    if (tid_2h4p < serial_2h4p) {
        const int ij = tid_2h4p / (num_vir_al_2 * num_vir_be_2);
        const size_t abcd = tid_2h4p % (num_vir_al_2 * num_vir_be_2);
        const int ac = abcd / num_vir_be_2;
        const int bd = abcd % num_vir_be_2;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int a = num_occ_al + (ac / num_vir_al);
        const int c = num_occ_al + (ac % num_vir_al);
        const int b = num_occ_be + (bd / num_vir_be);
        const int d = num_occ_be + (bd % num_vir_be);

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ijcd = g_eps_al[i] + g_eps_be[j] - g_eps_al[c] - g_eps_be[d];
        const double numerator =
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(a, c, b, d, num_orbitals)] *
            g_int2e_aabb[q2s(i, c, j, d, num_orbitals)];

        energy_2h4p = numerator / (eps_ijab * eps_ijcd);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_2h4p += __shfl_down_sync(FULLMASK, energy_2h4p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_2h4p, energy_2h4p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_2h4p, s_energy_2h4p);
    }
}











__global__ void compute_3h3p_aaaaaa(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa,
    const double* g_eps_al,
    const int num_occ_al, const int num_vir_al)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_3 = (size_t)num_occ_al * num_occ_al * num_occ_al;
    const size_t num_vir_al_3 = (size_t)num_vir_al * num_vir_al * num_vir_al;
    const size_t serial_3h3p = num_occ_al_3 * num_vir_al_3;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / num_vir_al_3;
        const size_t abc = tid_3h3p % num_vir_al_3;
        //const int i = ijk / (num_occ_al * num_occ_al);
        //const int j = (ijk / num_occ_al) % num_occ_al;
        //const int k = ijk % num_occ_al;
        //const int a = num_occ_al + abc / (num_vir_al * num_vir_al);
        //const int b = num_occ_al + (abc / num_vir_al) % num_vir_al;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_al;
        const int k = ijk % num_occ_al;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_al + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_al[k] - g_eps_al[a] - g_eps_al[c];
        const double numerator = 
            (g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] - g_int2e_aaaa[q2s(i, b, j, a, num_orbitals)]) *
            (g_int2e_aaaa[q2s(i, a, k, c, num_orbitals)] - g_int2e_aaaa[q2s(i, c, k, a, num_orbitals)]) *
            (g_int2e_aaaa[q2s(k, c, b, j, num_orbitals)] - g_int2e_aaaa[q2s(k, j, b, c, num_orbitals)]);
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_aabaab(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa, 
    const double* g_int2e_aabb,
    const double* g_int2e_bbaa,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_al * num_occ_be * num_vir_al * num_vir_al * num_vir_be;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_al * num_vir_be);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_al * num_vir_be);
        //const int i = ijk / (num_occ_al * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_al;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_al + abc / (num_vir_al * num_vir_be);
        //const int b = num_occ_al + (abc / num_vir_be) % num_vir_al;
        //const int c = num_occ_be + abc % num_vir_be;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int ab = abc / num_vir_be;
        const int c = num_occ_be + abc % num_vir_be;
        const int a = num_occ_al + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_al[a] - g_eps_be[c];
        const double numerator = 
            (g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] - g_int2e_aaaa[q2s(i, b, j, a, num_orbitals)]) *
            g_int2e_aabb[q2s(i, a, k, c, num_orbitals)] *
            g_int2e_bbaa[q2s(k, c, b, j, num_orbitals)];
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_abaaba(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa, 
    const double* g_int2e_aabb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_al * num_vir_al * num_vir_be * num_vir_al;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_be * num_vir_al);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_be * num_vir_al);
        //const int i = ijk / (num_occ_be * num_occ_al);
        //const int j = (ijk / num_occ_al) % num_occ_be;
        //const int k = ijk % num_occ_al;
        //const int a = num_occ_al + abc / (num_vir_be * num_vir_al);
        //const int b = num_occ_be + (abc / num_vir_al) % num_vir_be;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_al;
        const int k = ijk % num_occ_al;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_al + ab / num_vir_be;
        const int b = num_occ_be + ab % num_vir_be;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ikac = g_eps_al[i] + g_eps_al[k] - g_eps_al[a] - g_eps_al[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            (g_int2e_aaaa[q2s(i, a, k, c, num_orbitals)] - g_int2e_aaaa[q2s(i, c, k, a, num_orbitals)]) *
            g_int2e_aabb[q2s(k, c, b, j, num_orbitals)];
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}

__global__ void compute_3h3p_abbabb(
    double* g_energy_3h3p,
    const double* g_int2e_aabb,
    const double* g_int2e_bbbb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_be * num_vir_al * num_vir_be * num_vir_be;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_be * num_vir_be);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_be * num_vir_be);
        //const int i = ijk / (num_occ_be * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_be;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_al + abc / (num_vir_be * num_vir_be);
        //const int b = num_occ_be + (abc / num_vir_be) % num_vir_be;
        //const int c = num_occ_be + abc % num_vir_be;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_be;
        const int c = num_occ_be + abc % num_vir_be;
        const int a = num_occ_al + ab / num_vir_be;
        const int b = num_occ_be + ab % num_vir_be;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_al[a] - g_eps_be[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(i, a, k, c, num_orbitals)] *
            (g_int2e_bbbb[q2s(k, c, b, j, num_orbitals)] - g_int2e_bbbb[q2s(k, j, b, c, num_orbitals)]);
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_abbbaa(
    double* g_energy_3h3p,
    const double* g_int2e_aabb,
    const double* g_int2e_bbaa,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_be * num_vir_be * num_vir_al * num_vir_al;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_be * num_vir_al * num_vir_al);
        const size_t abc = tid_3h3p % ((size_t)num_vir_be * num_vir_al * num_vir_al);
        //const int i = ijk / (num_occ_be * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_be;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_be + abc / (num_vir_al * num_vir_al);
        //const int b = num_occ_al + (abc / num_vir_al) % num_vir_al;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_be + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_be[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_be[a] - g_eps_al[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, b, j, a, num_orbitals)] *
            g_int2e_aabb[q2s(i, c, k, a, num_orbitals)] *
            g_int2e_bbaa[q2s(k, j, b, c, num_orbitals)];
        energy_3h3p = (-1) * numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}









double ump3_from_aoeri_via_full_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al, const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al, const double* d_orbital_energies_be,
    const int num_basis, const int num_occ_al, const int num_occ_be)
{
    double* d_g_aaaa_full = nullptr;
    double* d_g_aabb_full = nullptr;
    double* d_g_bbaa_full = nullptr;
    double* d_g_bbbb_full = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_aaaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_aabb_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbbb_full, sizeof(double) * num_basis_4);
    if (!d_g_aaaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_full."); }
    if (!d_g_aabb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_full."); }
    if (!d_g_bbaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_full."); }
    if (!d_g_bbbb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_full."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    transform_ump3_full_mo_eris(
        d_eri_ao,
        d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
        d_coefficient_matrix_al, d_coefficient_matrix_be,
        num_basis_4, num_basis
    );

    if (!gpu::gpu_available()) {
        // ---------- CPU fallback for brute-force UMP3 ----------
        using namespace cpu_kernels;
        double h_4h2p = 0.0, h_2h4p = 0.0, h_3h3p = 0.0;

        // 4h2p: aa + ab + ba + bb, each scaled by 0.5
        h_4h2p += compute_4h2p_ss_cpu(d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
        h_4h2p += compute_4h2p_os_cpu(d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        h_4h2p += compute_4h2p_os_cpu(d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        h_4h2p += compute_4h2p_ss_cpu(d_g_bbbb_full, d_orbital_energies_be, num_occ_be, num_vir_be);
        h_4h2p *= 0.5;

        // 2h4p: aa + ab + ba + bb, each scaled by 0.5
        h_2h4p += compute_2h4p_ss_cpu(d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
        h_2h4p += compute_2h4p_os_cpu(d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        h_2h4p += compute_2h4p_os_cpu(d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        h_2h4p += compute_2h4p_ss_cpu(d_g_bbbb_full, d_orbital_energies_be, num_occ_be, num_vir_be);
        h_2h4p *= 0.5;

        // 3h3p: 10 spin combinations
        h_3h3p += compute_3h3p_aaaaaa_cpu(d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
        h_3h3p += compute_3h3p_aabaab_cpu(d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        h_3h3p += compute_3h3p_abaaba_cpu(d_g_aaaa_full, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        h_3h3p += compute_3h3p_abbabb_cpu(d_g_aabb_full, d_g_bbbb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        h_3h3p += compute_3h3p_abbbaa_cpu(d_g_aabb_full, d_g_bbaa_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        // baaabb = abbbaa with swapped spins
        h_3h3p += compute_3h3p_abbbaa_cpu(d_g_bbaa_full, d_g_aabb_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        // baabaa = abbabb with swapped spins
        h_3h3p += compute_3h3p_abbabb_cpu(d_g_bbaa_full, d_g_aaaa_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        // babbab = abaaba with swapped spins
        h_3h3p += compute_3h3p_abaaba_cpu(d_g_bbbb_full, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        // bbabba = aabaab with swapped spins
        h_3h3p += compute_3h3p_aabaab_cpu(d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al);
        h_3h3p += compute_3h3p_aaaaaa_cpu(d_g_bbbb_full, d_orbital_energies_be, num_occ_be, num_vir_be);

        std::cout << "E_4h2p: " << h_4h2p << " [hartree]" << std::endl;
        std::cout << "E_2h4p: " << h_2h4p << " [hartree]" << std::endl;
        std::cout << "E_3h3p: " << h_3h3p << " [hartree]" << std::endl;

        tracked_cudaFree(d_g_aaaa_full);
        tracked_cudaFree(d_g_aabb_full);
        tracked_cudaFree(d_g_bbaa_full);
        tracked_cudaFree(d_g_bbbb_full);
        return h_4h2p + h_2h4p + h_3h3p;
    }

    // ---------- GPU path ----------
    double* d_energy_4h2p = nullptr;
    double* d_energy_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p, 0, sizeof(double));
    cudaMemset(d_energy_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 0.5;

    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_occ_be_2 = num_occ_be * num_occ_be;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;
    const size_t num_occ_al_3 = num_occ_al_2 * num_occ_al;
    const size_t num_occ_be_3 = num_occ_be_2 * num_occ_be;
    const size_t num_vir_al_3 = num_vir_al_2 * num_vir_al;
    const size_t num_vir_be_3 = num_vir_be_2 * num_vir_be;




    // 4h2p contributions
    double* d_energy_4h2p_aa = nullptr;
    double* d_energy_4h2p_ab = nullptr;
    double* d_energy_4h2p_ba = nullptr;
    double* d_energy_4h2p_bb = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_aa, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_ab, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_ba, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_bb, sizeof(double));
    cudaMemset(d_energy_4h2p_aa, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_ab, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_ba, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_bb, 0, sizeof(double));

    const size_t num_4h2p_aa = num_occ_al_2 * num_occ_al_2 * num_vir_al_2;
    const size_t num_4h2p_bb = num_occ_be_2 * num_occ_be_2 * num_vir_be_2;
    const size_t num_4h2p_ab = num_occ_al_2 * num_occ_be_2 * num_vir_al * num_vir_be;
    const size_t num_4h2p_ba = num_occ_be_2 * num_occ_al_2 * num_vir_be * num_vir_al;
    const size_t num_blocks_4h2p_aa = (num_4h2p_aa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_bb = (num_4h2p_bb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_ab = (num_4h2p_ab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_ba = (num_4h2p_ba + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_4h2p_aa = make_2d_grid_from_1d_blocks(num_blocks_4h2p_aa, prop);
    const dim3 blocks_4h2p_bb = make_2d_grid_from_1d_blocks(num_blocks_4h2p_bb, prop);
    const dim3 blocks_4h2p_ab = make_2d_grid_from_1d_blocks(num_blocks_4h2p_ab, prop);
    const dim3 blocks_4h2p_ba = make_2d_grid_from_1d_blocks(num_blocks_4h2p_ba, prop);
    // 4h2p-aa
    compute_4h2p_ss<<<blocks_4h2p_aa, threads>>>(
        d_energy_4h2p, d_g_aaaa_full, d_orbital_energies_al, 
        //d_energy_4h2p_aa, d_g_aaaa_full, d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 4h2p-ab
    compute_4h2p_os<<<blocks_4h2p_ab, threads>>>(
        d_energy_4h2p, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        //d_energy_4h2p_ab, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 4h2p-ba
    compute_4h2p_os<<<blocks_4h2p_ba, threads>>>(
        d_energy_4h2p, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        //d_energy_4h2p_ba, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 4h2p-bb
    compute_4h2p_ss<<<blocks_4h2p_bb, threads>>>(
        d_energy_4h2p, d_g_bbbb_full, d_orbital_energies_be, 
        //d_energy_4h2p_bb, d_g_bbbb_full, d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );
    cudaDeviceSynchronize();
    // Scale the 4h2p energy by 0.5
    cublasDscal(handle, 1, &alpha, d_energy_4h2p, 1);





    // 2h4p contributions
    double* d_energy_2h4p_aa = nullptr;
    double* d_energy_2h4p_ab = nullptr;
    double* d_energy_2h4p_ba = nullptr;
    double* d_energy_2h4p_bb = nullptr;
    tracked_cudaMalloc(&d_energy_2h4p_aa, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_ab, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_ba, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_bb, sizeof(double));
    cudaMemset(d_energy_2h4p_aa, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_ab, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_ba, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_bb, 0, sizeof(double));


    const size_t num_2h4p_aa = num_occ_al_2 * num_vir_al_2 * num_vir_al_2;
    const size_t num_2h4p_ab = num_occ_al * num_occ_be * num_vir_al_2 * num_vir_be_2;
    const size_t num_2h4p_ba = num_occ_be * num_occ_al * num_vir_be_2 * num_vir_al_2;
    const size_t num_2h4p_bb = num_occ_be_2 * num_vir_be_2 * num_vir_be_2;
    const size_t num_blocks_2h4p_aa = (num_2h4p_aa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_ab = (num_2h4p_ab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_ba = (num_2h4p_ba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_bb = (num_2h4p_bb + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_2h4p_aa = make_2d_grid_from_1d_blocks(num_blocks_2h4p_aa, prop);
    const dim3 blocks_2h4p_ab = make_2d_grid_from_1d_blocks(num_blocks_2h4p_ab, prop);
    const dim3 blocks_2h4p_ba = make_2d_grid_from_1d_blocks(num_blocks_2h4p_ba, prop);
    const dim3 blocks_2h4p_bb = make_2d_grid_from_1d_blocks(num_blocks_2h4p_bb, prop);
    // 2h4p-aa
    compute_2h4p_ss<<<blocks_2h4p_aa, threads>>>(
        d_energy_2h4p, d_g_aaaa_full, d_orbital_energies_al, 
        //d_energy_2h4p_aa, d_g_aaaa_full, d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 2h4p-ab
    compute_2h4p_os<<<blocks_2h4p_ab, threads>>>(
        d_energy_2h4p, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        //d_energy_2h4p_ab, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 2h4p-ba
    compute_2h4p_os<<<blocks_2h4p_ba, threads>>>(
        d_energy_2h4p, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        //d_energy_2h4p_ba, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 2h4p-bb
    compute_2h4p_ss<<<blocks_2h4p_bb, threads>>>(
        d_energy_2h4p, d_g_bbbb_full, d_orbital_energies_be, 
        //d_energy_2h4p_bb, d_g_bbbb_full, d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );
    // Scale the 2h4p energy by 0.5
    cublasDscal(handle, 1, &alpha, d_energy_2h4p, 1);



    // 3h3p contributions
    double* d_energy_aaaaaa = nullptr;
    double* d_energy_aabaab = nullptr;
    double* d_energy_abaaba = nullptr;
    double* d_energy_abbabb = nullptr;
    double* d_energy_abbbaa = nullptr;
    double* d_energy_baaabb = nullptr;
    double* d_energy_baabaa = nullptr;
    double* d_energy_babbab = nullptr;
    double* d_energy_bbabba = nullptr;
    double* d_energy_bbbbbb = nullptr;
    tracked_cudaMalloc(&d_energy_aaaaaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_aabaab, sizeof(double));
    tracked_cudaMalloc(&d_energy_abaaba, sizeof(double));
    tracked_cudaMalloc(&d_energy_abbabb, sizeof(double));
    tracked_cudaMalloc(&d_energy_abbbaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_baaabb, sizeof(double));
    tracked_cudaMalloc(&d_energy_baabaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_babbab, sizeof(double));
    tracked_cudaMalloc(&d_energy_bbabba, sizeof(double));
    tracked_cudaMalloc(&d_energy_bbbbbb, sizeof(double));
    cudaMemset(d_energy_aaaaaa, 0, sizeof(double));
    cudaMemset(d_energy_aabaab, 0, sizeof(double));
    cudaMemset(d_energy_abaaba, 0, sizeof(double));
    cudaMemset(d_energy_abbabb, 0, sizeof(double));
    cudaMemset(d_energy_abbbaa, 0, sizeof(double));
    cudaMemset(d_energy_baaabb, 0, sizeof(double));
    cudaMemset(d_energy_baabaa, 0, sizeof(double));
    cudaMemset(d_energy_babbab, 0, sizeof(double));
    cudaMemset(d_energy_bbabba, 0, sizeof(double));
    cudaMemset(d_energy_bbbbbb, 0, sizeof(double));

    const size_t num_3h3p_aaaaaa = num_occ_al_3                * num_vir_al_3;
    const size_t num_3h3p_aabaab = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_abaaba = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_abbabb = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_abbbaa = num_occ_al   * num_occ_be_2 * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_baaabb = num_occ_al_2 * num_occ_be   * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_baabaa = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_babbab = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_bbabba = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_bbbbbb =                num_occ_be_3                * num_vir_be_3;
    const size_t num_blocks_3h3p_aaaaaa = (num_3h3p_aaaaaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_aabaab = (num_3h3p_aabaab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abaaba = (num_3h3p_abaaba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abbabb = (num_3h3p_abbabb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abbbaa = (num_3h3p_abbbaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_baaabb = (num_3h3p_baaabb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_baabaa = (num_3h3p_baabaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_babbab = (num_3h3p_babbab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_bbabba = (num_3h3p_bbabba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_bbbbbb = (num_3h3p_bbbbbb + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_3h3p_aaaaaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_aaaaaa, prop);
    const dim3 blocks_3h3p_aabaab = make_2d_grid_from_1d_blocks(num_blocks_3h3p_aabaab, prop);
    const dim3 blocks_3h3p_abaaba = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abaaba, prop);
    const dim3 blocks_3h3p_abbabb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abbabb, prop);
    const dim3 blocks_3h3p_abbbaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abbbaa, prop);
    const dim3 blocks_3h3p_baaabb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_baaabb, prop);
    const dim3 blocks_3h3p_baabaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_baabaa, prop);
    const dim3 blocks_3h3p_babbab = make_2d_grid_from_1d_blocks(num_blocks_3h3p_babbab, prop);
    const dim3 blocks_3h3p_bbabba = make_2d_grid_from_1d_blocks(num_blocks_3h3p_bbabba, prop);
    const dim3 blocks_3h3p_bbbbbb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_bbbbbb, prop);
    // 3h3p-aaaaaa
    compute_3h3p_aaaaaa<<<blocks_3h3p_aaaaaa, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, 
        //d_energy_aaaaaa, d_g_aaaa_full, 
        d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 3h3p-aabaab
    compute_3h3p_aabaab<<<blocks_3h3p_aabaab, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, 
        //d_energy_aabaab, d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abaaba
    compute_3h3p_abaaba<<<blocks_3h3p_abaaba, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, d_g_aabb_full, 
        //d_energy_abaaba, d_g_aaaa_full, d_g_aabb_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abbabb
    compute_3h3p_abbabb<<<blocks_3h3p_abbabb, threads>>>(
        d_energy_3h3p, d_g_aabb_full, d_g_bbbb_full, 
        //d_energy_abbabb, d_g_aabb_full, d_g_bbbb_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abbbaa
    compute_3h3p_abbbaa<<<blocks_3h3p_abbbaa, threads>>>(
        d_energy_3h3p, d_g_aabb_full, d_g_bbaa_full, 
        //d_energy_abbbaa, d_g_aabb_full, d_g_bbaa_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-baaabb
    compute_3h3p_abbbaa<<<blocks_3h3p_baaabb, threads>>>(
        d_energy_3h3p, d_g_bbaa_full, d_g_aabb_full, 
        //d_energy_baaabb, d_g_bbaa_full, d_g_aabb_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-baabaa
    compute_3h3p_abbabb<<<blocks_3h3p_baabaa, threads>>>(
        d_energy_3h3p, d_g_bbaa_full, d_g_aaaa_full, 
        //d_energy_baabaa, d_g_bbaa_full, d_g_aaaa_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-babbab
    compute_3h3p_abaaba<<<blocks_3h3p_babbab, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, d_g_bbaa_full, 
        //d_energy_babbab, d_g_bbbb_full, d_g_bbaa_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-bbabba
    compute_3h3p_aabaab<<<blocks_3h3p_bbabba, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, 
        //d_energy_bbabba, d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-bbbbbb
    compute_3h3p_aaaaaa<<<blocks_3h3p_bbbbbb, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, 
        //d_energy_bbbbbb, d_g_bbbb_full, 
        d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );


    cudaDeviceSynchronize();
    double h_energy_4h2p = 0.0;
    double h_energy_2h4p = 0.0;
    double h_energy_3h3p = 0.0;
    double h_energy = 0.0;

    /*
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_aa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_aa: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_ab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_ab: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_ba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_ba: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_bb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_bb: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    std::cout << "Total E_4h2p: " << h_energy << " [hartree]" << std::endl;
    h_energy = 0.0;
    /**/

    /*
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_aa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_aa: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_ab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_ab: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_ba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_ba: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_bb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_bb: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    std::cout << "Total E_2h4p: " << h_energy << " [hartree]" << std::endl;
    h_energy = 0.0;
    /**/

    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_aaaaaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_aaaaaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_aabaab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_aabaab: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abaaba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abaaba: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abbabb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abbabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abbbaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abbbaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_baaabb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_baaabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_baabaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_baabaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_babbab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_babbab: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_bbabba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_bbabba: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_bbbbbb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_bbbbbb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    std::cout << "Total E_3h3p: " << h_energy << " [hartree]" << std::endl;
    /**/

    //*
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p: " << h_energy_4h2p << " [hartree]" << std::endl;
    std::cout << "E_2h4p: " << h_energy_2h4p << " [hartree]" << std::endl;
    std::cout << "E_3h3p: " << h_energy_3h3p << " [hartree]" << std::endl;
    /**/

    tracked_cudaFree(d_g_aaaa_full);
    tracked_cudaFree(d_g_aabb_full);
    tracked_cudaFree(d_g_bbaa_full);
    tracked_cudaFree(d_g_bbbb_full);
    tracked_cudaFree(d_energy_4h2p);
    tracked_cudaFree(d_energy_2h4p);
    tracked_cudaFree(d_energy_3h3p);

    tracked_cudaFree(d_energy_4h2p_aa);
    tracked_cudaFree(d_energy_4h2p_ab);
    tracked_cudaFree(d_energy_4h2p_ba);
    tracked_cudaFree(d_energy_4h2p_bb);
    tracked_cudaFree(d_energy_2h4p_aa);
    tracked_cudaFree(d_energy_2h4p_ab);
    tracked_cudaFree(d_energy_2h4p_ba);
    tracked_cudaFree(d_energy_2h4p_bb);
    tracked_cudaFree(d_energy_aaaaaa);
    tracked_cudaFree(d_energy_aabaab);
    tracked_cudaFree(d_energy_abaaba);
    tracked_cudaFree(d_energy_abbabb);
    tracked_cudaFree(d_energy_abbbaa);
    tracked_cudaFree(d_energy_baaabb);
    tracked_cudaFree(d_energy_baabaa);
    tracked_cudaFree(d_energy_babbab);
    tracked_cudaFree(d_energy_bbabba);
    tracked_cudaFree(d_energy_bbbbbb);

    cublasDestroy(handle);

    //return 0.0;
    return h_energy_4h2p + h_energy_2h4p + h_energy_3h3p;
}













double compute_aaaa_contributions(
    double* d_energy_4h2p_2h4p,
    double* d_energy_3h3p,
    const double* d_g_aaaa_full,
    const double* d_orbital_energies_al,
    const int num_occ_al,
    const int num_vir_al)
{
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_ov_al_2 = num_occ_al * num_vir_al;

    const size_t num_oooo = num_occ_al_2 * num_occ_al_2;
    const size_t num_vvvv = num_vir_al_2 * num_vir_al_2;
    const size_t num_ovov = num_occ_al_2 * num_vir_al_2;
    const size_t num_ovvo = num_occ_al_2 * num_vir_al_2;

    if (!gpu::gpu_available()) {
        // ---------- CPU fallback ----------
        using namespace cpu_kernels;
        double* g_oooo = (double*)malloc(sizeof(double)*num_oooo);
        double* g_vvvv = (double*)malloc(sizeof(double)*num_vvvv);
        double* u_ovvo = (double*)malloc(sizeof(double)*num_ovvo);
        double* x_ovov = (double*)malloc(sizeof(double)*num_ovov);
        double* y_ovov = (double*)malloc(sizeof(double)*num_ovov);
        double* tmp_1  = (double*)malloc(sizeof(double)*num_ovov);
        double* tmp_2  = (double*)malloc(sizeof(double)*num_ovov);
        double* tmp_3  = (double*)malloc(sizeof(double)*num_ovov);

        tensorize_g_aaaa_oooo_cpu(g_oooo, d_g_aaaa_full, num_occ_al, num_vir_al);
        tensorize_g_aaaa_vvvv_cpu(g_vvvv, d_g_aaaa_full, num_occ_al, num_vir_al);
        tensorize_u_aaaa_ovvo_cpu(u_ovvo, d_g_aaaa_full, num_occ_al, num_vir_al);
        tensorize_x_aaaa_ovov_cpu(x_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
        tensorize_y_aaaa_ovov_cpu(y_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

        // Y * u -> t_ovvo (tmp_1)
        cpu_dgemm_row_major(num_ov_al_2, num_ov_al_2, num_ov_al_2, 1.0, y_ovov, u_ovvo, tmp_1, false, false);
        double e3h3p = contract_3h3p_aaaaaa_cpu(y_ovov, tmp_1, num_occ_al, num_vir_al);

        // kalb -> klab (tmp_1)
        kalb2klab_aaaa_cpu(tmp_1, y_ovov, num_occ_al, num_vir_al);
        // g_oooo * Y_klab -> t_oovv (tmp_2)
        cpu_dgemm_row_major(num_occ_al_2, num_vir_al_2, num_occ_al_2, 1.0, g_oooo, tmp_1, tmp_2, false, false);

        // icjd -> cdij (tmp_1)
        icjd2cdij_aaaa_cpu(tmp_1, y_ovov, num_occ_al, num_vir_al);
        // g_vvvv * Y_cdij -> t_vvoo (tmp_3)
        cpu_dgemm_row_major(num_vir_al_2, num_occ_al_2, num_vir_al_2, 1.0, g_vvvv, tmp_1, tmp_3, false, false);

        double e4h2p = contract_4h2p_2h4p_aaaaaa_cpu(x_ovov, tmp_2, tmp_3, num_occ_al, num_vir_al);

        *d_energy_3h3p += e3h3p;
        *d_energy_4h2p_2h4p += e4h2p;

        double h_e3 = *d_energy_3h3p;
        double h_e4 = *d_energy_4h2p_2h4p;

        free(g_oooo); free(g_vvvv); free(u_ovvo);
        free(x_ovov); free(y_ovov); free(tmp_1); free(tmp_2); free(tmp_3);

        return h_e4 + h_e3;
    }

    // ---------- GPU path ----------
    // g: (ik|jl), (ac|bd)
    // u: (kc||bj)
    double* d_g_aaaa_oooo = nullptr;
    double* d_g_aaaa_vvvv = nullptr;
    double* d_u_aaaa_ovvo = nullptr;
    double* d_x_aaaa_ovov = nullptr;
    double* d_y_aaaa_ovov = nullptr;
    double* d_tmp_1 = nullptr;
    double* d_tmp_2 = nullptr;

    tracked_cudaMalloc(&d_g_aaaa_oooo, sizeof(double) * num_oooo);
    tracked_cudaMalloc(&d_g_aaaa_vvvv, sizeof(double) * num_vvvv);
    tracked_cudaMalloc(&d_u_aaaa_ovvo, sizeof(double) * num_ovvo);
    tracked_cudaMalloc(&d_x_aaaa_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_tmp_1, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_tmp_2, sizeof(double) * num_ovov);

    if (!d_g_aaaa_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_oooo."); }
    if (!d_g_aaaa_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_vvvv."); }
    if (!d_u_aaaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_aaaa_ovvo."); }
    if (!d_x_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aaaa_ovov."); }
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_tmp_1) { THROW_EXCEPTION("cudaMalloc failed for d_t_aaaa_ovov."); }
    if (!d_tmp_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_2."); }

    cudaMemset(d_g_aaaa_oooo, 0, sizeof(double) * num_oooo);
    cudaMemset(d_g_aaaa_vvvv, 0, sizeof(double) * num_vvvv);
    cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_ovvo);
    cudaMemset(d_x_aaaa_ovov, 0, sizeof(double) * num_ovov);
    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_ovov);
    cudaMemset(d_tmp_1, 0, sizeof(double) * num_ovov);
    cudaMemset(d_tmp_2, 0, sizeof(double) * num_ovov);

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_blocks_oooo = (num_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_vvvv = (num_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_ovov = (num_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_ovvo = (num_ovvo + num_threads_per_block - 1) / num_threads_per_block;

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    if (num_blocks_oooo > prop.maxGridSize[0] ||
        num_blocks_vvvv > prop.maxGridSize[0] ||
        num_blocks_ovov > prop.maxGridSize[0] ||
        num_blocks_ovvo > prop.maxGridSize[0]) {
        THROW_EXCEPTION("Error: Too many blocks for the grid size.");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    tensorize_g_aaaa_oooo<<<num_blocks_oooo, threads>>>(d_g_aaaa_oooo, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_g_aaaa_vvvv<<<num_blocks_vvvv, threads>>>(d_g_aaaa_vvvv, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_u_aaaa_ovvo<<<num_blocks_ovvo, threads>>>(d_u_aaaa_ovvo, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_x_aaaa_ovov<<<num_blocks_ovov, threads>>>(d_x_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
    tensorize_y_aaaa_ovov<<<num_blocks_ovov, threads>>>(d_y_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

    // Y_iakc * u_kcbj --> t_iabj (d_tmp_1)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_ov_al_2, num_ov_al_2, num_ov_al_2,
        &alpha, d_y_aaaa_ovov, num_ov_al_2,
        d_u_aaaa_ovvo, num_ov_al_2,
        &beta, d_tmp_1, num_ov_al_2
    );
    const double* d_t_aaaa_ovvo = d_tmp_1;
    contract_3h3p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_3h3p, d_y_aaaa_ovov, d_t_aaaa_ovvo, num_occ_al, num_vir_al);

    // Y_klab --> d_tmp_1
    kalb2klab_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_aaaa_ovov, num_occ_al, num_vir_al);
    // g_ijkl * Y_klab --> d_tmp_2
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al_2, num_vir_al_2, num_occ_al_2,
        &alpha, d_g_aaaa_oooo, num_occ_al_2,
        d_tmp_1, num_vir_al_2,
        &beta, d_tmp_2, num_vir_al_2
    );
    const double* d_t_aaaa_oovv = d_tmp_2;

    double* d_tmp_3 = d_u_aaaa_ovvo;
    // Y_cdij --> d_tmp_1
    icjd2cdij_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_aaaa_ovov, num_occ_al, num_vir_al);
    // g_abcd * Y_cdij --> d_tmp_3
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_al_2, num_occ_al_2, num_vir_al_2,
        &alpha, d_g_aaaa_vvvv, num_vir_al_2,
        d_tmp_1, num_occ_al_2,
        &beta, d_tmp_3, num_occ_al_2
    );
    const double* d_t_aaaa_vvoo = d_tmp_3;
    contract_4h2p_2h4p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_aaaa_ovov, d_t_aaaa_oovv, d_t_aaaa_vvoo, num_occ_al, num_vir_al);

    cudaDeviceSynchronize();

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
    //std::cout << "E_3h3p_aaaaaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    //std::cout << "E_4h2p_2h4p_aaaaaa: " << h_energy_4h2p_2h4p << " [hartree]" << std::endl;

    cublasDestroy(cublasH);

    tracked_cudaFree(d_g_aaaa_oooo);
    tracked_cudaFree(d_g_aaaa_vvvv);
    tracked_cudaFree(d_u_aaaa_ovvo);
    tracked_cudaFree(d_x_aaaa_ovov);
    tracked_cudaFree(d_y_aaaa_ovov);
    tracked_cudaFree(d_tmp_1);
    tracked_cudaFree(d_tmp_2);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}





double compute_aabb_contributions(
    double* d_energy_4h2p_2h4p, double* d_energy_3h3p,
    const double* d_g_aaaa_full, const double* d_g_aabb_full,
    const double* d_g_bbaa_full, const double* d_g_bbbb_full,
    const double* d_orbital_energies_al, const double* d_orbital_energies_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be,
    const bool compute_4h2p_2h4p_flag)
{
    const int num_basis = num_occ_al + num_vir_al;

    if (!gpu::gpu_available()) {
        // ---------- CPU fallback ----------
        using namespace cpu_kernels;

        const size_t num_aabb_oooo = (size_t)num_occ_al*num_occ_be*num_occ_al*num_occ_be;
        const size_t num_aabb_vvvv = (size_t)num_vir_al*num_vir_be*num_vir_al*num_vir_be;
        const size_t num_bbaa_ovvo = (size_t)num_occ_be*num_vir_be*num_vir_al*num_occ_al;
        const size_t num_bbaa_oovv = (size_t)num_occ_be*num_occ_be*num_vir_al*num_vir_al;
        const size_t num_bbbb_ovvo = (size_t)num_occ_be*num_vir_be*num_vir_be*num_occ_be;
        const size_t num_aabb_ovov = (size_t)num_occ_al*num_vir_al*num_occ_be*num_vir_be;
        const size_t num_aaaa_ovov = (size_t)num_occ_al*num_vir_al*num_occ_al*num_vir_al;

        double* g_aabb_oooo = (double*)malloc(sizeof(double)*num_aabb_oooo);
        double* g_aabb_vvvv = (double*)malloc(sizeof(double)*num_aabb_vvvv);
        double* g_bbaa_ovvo = (double*)malloc(sizeof(double)*num_bbaa_ovvo);
        double* g_bbaa_oovv = (double*)malloc(sizeof(double)*num_bbaa_oovv);
        double* u_bbbb_ovvo = (double*)malloc(sizeof(double)*num_bbbb_ovvo);
        double* x_aabb_ovov = (double*)malloc(sizeof(double)*num_aabb_ovov);
        double* y_aaaa_ovov = (double*)malloc(sizeof(double)*num_aaaa_ovov);
        double* tmp_1 = (double*)malloc(sizeof(double)*num_aaaa_ovov);
        double* tmp_2 = (double*)malloc(sizeof(double)*num_aabb_ovov);
        double* tmp_3 = (double*)malloc(sizeof(double)*num_aabb_ovov);
        double* tmp_4 = (double*)malloc(sizeof(double)*num_aabb_ovov);

        tensorize_g_aabb_oooo_cpu(g_aabb_oooo, d_g_aabb_full, num_occ_al, num_occ_be, num_basis);
        tensorize_g_aabb_vvvv_cpu(g_aabb_vvvv, d_g_aabb_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);
        tensorize_g_bbaa_ovvo_cpu(g_bbaa_ovvo, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
        tensorize_g_bbaa_oovv_cpu(g_bbaa_oovv, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
        tensorize_u_bbbb_ovvo_cpu(u_bbbb_ovvo, d_g_bbbb_full, num_occ_be, num_vir_be);
        tensorize_x_aabb_ovov_cpu(x_aabb_ovov, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_basis);
        tensorize_y_aaaa_ovov_cpu(y_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

        double h_e3 = 0.0, h_e4 = 0.0;

        // X_aabb * g_bbaa_ovvo -> t_aaaa_ovvo (tmp_1)
        cpu_dgemm_row_major(num_occ_al*num_vir_al, num_vir_al*num_occ_al, num_occ_be*num_vir_be, 1.0, x_aabb_ovov, g_bbaa_ovvo, tmp_1, false, false);
        h_e3 += contract_3h3p_aabaab_abaaba_cpu(y_aaaa_ovov, tmp_1, num_occ_al, num_vir_al);

        // X_aabb * u_bbbb_ovvo -> t_aabb_ovvo (tmp_3)
        cpu_dgemm_row_major(num_occ_al*num_vir_al, num_vir_be*num_occ_be, num_occ_be*num_vir_be, 1.0, x_aabb_ovov, u_bbbb_ovvo, tmp_3, false, false);
        h_e3 += contract_3h3p_abbabb_cpu(x_aabb_ovov, tmp_3, num_occ_al, num_occ_be, num_vir_al, num_vir_be);

        // reorder X_aabb -> X_abba (tmp_3), then X_abba * g_bbaa_oovv -> t_abab (tmp_2)
        aabb_icka2abba_iakc_cpu(tmp_3, x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        cpu_dgemm_row_major(num_occ_al*num_vir_be, num_vir_al*num_occ_be, num_occ_be*num_vir_al, 1.0, tmp_3, g_bbaa_oovv, tmp_2, false, false);
        h_e3 += contract_3h3p_abbbaa_cpu(x_aabb_ovov, tmp_2, num_occ_al, num_occ_be, num_vir_al, num_vir_be);

        if (compute_4h2p_2h4p_flag) {
            // reorder X -> klab (tmp_2), then g_oooo * klab -> t_oovv (tmp_3)
            aabb_kalb2abab_klab_cpu(tmp_2, x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
            cpu_dgemm_row_major(num_occ_al*num_occ_be, num_vir_al*num_vir_be, num_occ_al*num_occ_be, 1.0, g_aabb_oooo, tmp_2, tmp_3, false, false);
            // reorder X -> cdij (tmp_2), then g_vvvv * cdij -> t_vvoo (tmp_4)
            aabb_icjd2abab_cdij_cpu(tmp_2, x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
            cpu_dgemm_row_major(num_vir_al*num_vir_be, num_occ_al*num_occ_be, num_vir_al*num_vir_be, 1.0, g_aabb_vvvv, tmp_2, tmp_4, false, false);
            h_e4 += contract_4h2p_2h4p_ababab_bababa_cpu(x_aabb_ovov, tmp_3, tmp_4, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
        }

        *d_energy_3h3p += h_e3;
        *d_energy_4h2p_2h4p += h_e4;

        double ret_e3 = *d_energy_3h3p;
        double ret_e4 = *d_energy_4h2p_2h4p;

        free(g_aabb_oooo); free(g_aabb_vvvv); free(g_bbaa_ovvo); free(g_bbaa_oovv);
        free(u_bbbb_ovvo); free(x_aabb_ovov); free(y_aaaa_ovov);
        free(tmp_1); free(tmp_2); free(tmp_3); free(tmp_4);

        return ret_e4 + ret_e3;
    }

    // ---------- GPU path ----------
    // g: (ik|jl), (ac|bd), (kc|bj), (kj|bc)
    // u: (kc||bj)
    double* d_g_aabb_oooo = nullptr;
    double* d_g_aabb_vvvv = nullptr;
    double* d_g_bbaa_ovvo = nullptr;
    double* d_g_bbaa_oovv = nullptr;
    double* d_u_bbbb_ovvo = nullptr;
    double* d_x_aabb_ovov = nullptr;
    double* d_y_aaaa_ovov = nullptr;
    double* d_tmp_1 = nullptr;
    double* d_tmp_2 = nullptr;
    double* d_tmp_4 = nullptr;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_occ_be_2 = num_occ_be * num_occ_be;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;

    const size_t num_aabb_oooo = num_occ_al_2 * num_occ_be_2;
    const size_t num_aabb_vvvv = num_vir_al_2 * num_vir_be_2;
    const size_t num_bbaa_ovvo = num_occ_be * num_vir_be * num_vir_al * num_occ_al;
    const size_t num_bbaa_oovv = num_occ_be_2 * num_vir_al_2;
    const size_t num_bbbb_ovvo = num_occ_be * num_vir_be * num_vir_be * num_occ_be;
    const size_t num_aabb_ovov = num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    const size_t num_aaaa_ovov = num_occ_al * num_vir_al * num_occ_al * num_vir_al;

    tracked_cudaMalloc(&d_g_aabb_oooo, sizeof(double) * num_aabb_oooo);
    tracked_cudaMalloc(&d_g_aabb_vvvv, sizeof(double) * num_aabb_vvvv);
    tracked_cudaMalloc(&d_g_bbaa_ovvo, sizeof(double) * num_bbaa_ovvo);
    tracked_cudaMalloc(&d_g_bbaa_oovv, sizeof(double) * num_bbaa_oovv);
    tracked_cudaMalloc(&d_u_bbbb_ovvo, sizeof(double) * num_bbbb_ovvo);
    tracked_cudaMalloc(&d_x_aabb_ovov, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_tmp_1, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_tmp_2, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_4, sizeof(double) * num_aabb_ovov);

    if (!d_g_aabb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oooo."); }
    if (!d_g_aabb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_vvvv."); }
    if (!d_g_bbaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_ovvo."); }
    if (!d_g_bbaa_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_oovv."); }
    if (!d_u_bbbb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_bbbb_ovvo."); }
    if (!d_x_aabb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aabb_ovov."); }
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_tmp_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_1."); }
    if (!d_tmp_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_2."); }
    if (!d_tmp_4) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_4."); }

    cudaMemset(d_g_aabb_oooo, 0, sizeof(double) * num_aabb_oooo);
    cudaMemset(d_g_aabb_vvvv, 0, sizeof(double) * num_aabb_vvvv);
    cudaMemset(d_g_bbaa_ovvo, 0, sizeof(double) * num_bbaa_ovvo);
    cudaMemset(d_g_bbaa_oovv, 0, sizeof(double) * num_bbaa_oovv);
    cudaMemset(d_u_bbbb_ovvo, 0, sizeof(double) * num_bbbb_ovvo);
    cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_tmp_1, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_tmp_2, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_tmp_4, 0, sizeof(double) * num_aabb_ovov);

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_blocks_aabb_oooo = (num_aabb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_vvvv = (num_aabb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_ovvo = (num_bbaa_ovvo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_oovv = (num_bbaa_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_ovvo = (num_bbbb_ovvo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_ovov = (num_aabb_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_ovov = (num_aaaa_ovov + num_threads_per_block - 1) / num_threads_per_block;

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    if (num_blocks_aabb_oooo > prop.maxGridSize[0] ||
        num_blocks_aabb_vvvv > prop.maxGridSize[0] ||
        num_blocks_bbaa_ovvo > prop.maxGridSize[0] ||
        num_blocks_bbaa_oovv > prop.maxGridSize[0] ||
        num_blocks_bbbb_ovvo > prop.maxGridSize[0] ||
        num_blocks_aabb_ovov > prop.maxGridSize[0] ||
        num_blocks_aaaa_ovov > prop.maxGridSize[0]) {
        THROW_EXCEPTION("Error: Too many blocks for the grid size.");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    tensorize_g_aabb_oooo<<<num_blocks_aabb_oooo, threads>>>(d_g_aabb_oooo, d_g_aabb_full, num_occ_al, num_occ_be, num_basis);
    tensorize_g_aabb_vvvv<<<num_blocks_aabb_vvvv, threads>>>(d_g_aabb_vvvv, d_g_aabb_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);
    tensorize_g_bbaa_ovvo<<<num_blocks_bbaa_ovvo, threads>>>(d_g_bbaa_ovvo, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
    tensorize_g_bbaa_oovv<<<num_blocks_bbaa_oovv, threads>>>(d_g_bbaa_oovv, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
    tensorize_u_bbbb_ovvo<<<num_blocks_bbbb_ovvo, threads>>>(d_u_bbbb_ovvo, d_g_bbbb_full, num_occ_be, num_vir_be);
    tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_aabb_ovov, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_basis);
    tensorize_y_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_y_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;

    // X_iakc^aabb * g_kcbj^bbaa --> t_iabj^aaaa (d_tmp_1)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_al, num_vir_al * num_occ_al, num_occ_be * num_vir_be,
        &alpha, d_x_aabb_ovov, num_occ_be * num_vir_be,
        d_g_bbaa_ovvo, num_vir_al * num_occ_al,
        &beta, d_tmp_1, num_vir_al * num_occ_al
    );
    const double* d_t_aaaa_ovvo = d_tmp_1;
    contract_3h3p_aabaab_abaaba<<<num_blocks_aaaa_ovov, threads>>>(d_energy_3h3p, d_y_aaaa_ovov, d_t_aaaa_ovvo, num_occ_al, num_vir_al);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_aabaab: " << (0.5) * h_energy_3h3p << " [hartree]" << std::endl;
    std::cout << "E_3h3p_abaaba: " << (0.5) * h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/


    double* d_tmp_3 = d_g_bbaa_ovvo;
    // X_iakc^aabb * u_kcbj^bbbb --> t_iabj^aabb (d_tmp_3)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_al, num_vir_be * num_occ_be, num_occ_be * num_vir_be,
        &alpha, d_x_aabb_ovov, num_occ_be * num_vir_be,
        d_u_bbbb_ovvo, num_vir_be * num_occ_be,
        &beta, d_tmp_3, num_vir_be * num_occ_be
    );
    const double* d_t_aabb_ovvo = d_tmp_3;
    contract_3h3p_abbabb<<<num_blocks_aabb_ovov, threads>>>(d_energy_3h3p, d_x_aabb_ovov, d_t_aabb_ovvo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_abbabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/

    // X_icka^aabb --> X_iakc^abba (d_tmp_3)
    aabb_icka2abba_iakc<<<num_blocks_aabb_ovov, threads>>>(d_tmp_3, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
    // X_iakc^abba * g_kcbj^baab --> t_iabj^abab (d_tmp_2)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_be, num_vir_al * num_occ_be, num_occ_be * num_vir_al,
        &alpha, d_tmp_3, num_occ_be * num_vir_al,
        d_g_bbaa_oovv, num_vir_al * num_occ_be,
        &beta, d_tmp_2, num_vir_al * num_occ_be
    );
    const double* d_t_abab_ovvo = d_tmp_2;
    contract_3h3p_abbbaa<<<num_blocks_aabb_ovov, threads>>>(d_energy_3h3p, d_x_aabb_ovov, d_t_abab_ovvo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_abbbaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/

    if (compute_4h2p_2h4p_flag) {
        // X_kalb^aabb --> X_klab^abab (d_tmp_2)
        aabb_kalb2abab_klab<<<num_blocks_aabb_ovov, threads>>>(d_tmp_2, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        // g_ijkl^abab * X_klab^abab --> t_ijab^abab (d_tmp_3)
        dgemm_device_row_major(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            num_occ_al * num_occ_be, num_vir_al * num_vir_be, num_occ_al * num_occ_be,
            &alpha, d_g_aabb_oooo, num_occ_al * num_occ_be,
            d_tmp_2, num_vir_al * num_vir_be,
            &beta, d_tmp_3, num_vir_al * num_vir_be
        );
        const double* d_t_abab_oovv = d_tmp_3;
        // X_icjd^aabb --> X_cdij^abab (d_tmp_2)
        aabb_icjd2abab_cdij<<<num_blocks_aabb_ovov, threads>>>(d_tmp_2, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        // g_abcd^abab * X_cdij^abab --> t_abij^abab (d_tmp_4)
        dgemm_device_row_major(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            num_vir_al * num_vir_be, num_occ_al * num_occ_be, num_vir_al * num_vir_be,
            &alpha, d_g_aabb_vvvv, num_vir_al * num_vir_be,
            d_tmp_2, num_occ_al * num_occ_be,
            &beta, d_tmp_4, num_occ_al * num_occ_be
        );
        const double* d_t_abab_vvoo = d_tmp_4;
        contract_4h2p_2h4p_ababab_bababa<<<num_blocks_aabb_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_aabb_ovov, d_t_abab_oovv, d_t_abab_vvoo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
        cudaDeviceSynchronize();
        /*
        cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "E_4h2p_2h4p_ababab: " << h_energy_4h2p_2h4p * 0.5 << " [hartree]" << std::endl;
        std::cout << "E_4h2p_2h4p_bababa: " << h_energy_4h2p_2h4p * 0.5 << " [hartree]" << std::endl;
        /**/
    }

    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(cublasH);

    tracked_cudaFree(d_g_aabb_oooo);
    tracked_cudaFree(d_g_aabb_vvvv);
    tracked_cudaFree(d_g_bbaa_ovvo);
    tracked_cudaFree(d_g_bbaa_oovv);
    tracked_cudaFree(d_u_bbbb_ovvo);
    tracked_cudaFree(d_x_aabb_ovov);
    tracked_cudaFree(d_y_aaaa_ovov);
    tracked_cudaFree(d_tmp_1);
    tracked_cudaFree(d_tmp_2);
    tracked_cudaFree(d_tmp_4);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}




// Naive 5 x M^4 reference baseline: all four spin blocks of the full MO
// ERI tensor (aaaa/aabb/bbaa/bbbb) are transformed and held simultaneously
// alongside the AO ERIs -- no intermediate lifetime management and no
// spin-transform reuse. This is the ablation baseline of the paper (the
// "naive algorithm" up to the matrix-multiplication formulation), retained
// as source only: compute_mp3_energy above always runs the cascading
// integral transformation and never calls this function. To reproduce the
// ablation, call it directly and invalidate_ao_eri() afterwards (the
// transform is destructive to the AO ERIs). Frozen core is unsupported.
double ump3_from_aoeri_via_full_moeri_dgemm(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis, 
    const int num_occ_al,
    const int num_occ_be)
{
    double* d_g_aaaa_full = nullptr;
    double* d_g_aabb_full = nullptr;
    double* d_g_bbaa_full = nullptr;
    double* d_g_bbbb_full = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_aaaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_aabb_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbbb_full, sizeof(double) * num_basis_4);
    if (!d_g_aaaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_full."); }
    if (!d_g_aabb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_full."); }
    if (!d_g_bbaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_full."); }
    if (!d_g_bbbb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_full."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    double* d_energy_4h2p_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    transform_ump3_full_mo_eris(
        d_eri_ao,
        d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
        d_coefficient_matrix_al, d_coefficient_matrix_be,
        num_basis_4, num_basis
    );

    // (alpha, alpha, alpha, alpha)
    const double energy_aaaa = 
        compute_aaaa_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_aaaa_full, d_orbital_energies_al, 
            num_occ_al, num_vir_al
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    //*
    // (alpha, alpha, beta, beta)
    const double energy_aabb = 
        compute_aabb_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
            d_orbital_energies_al, d_orbital_energies_be,
            num_occ_al, num_vir_al, num_occ_be, num_vir_be, 
            true
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));


    // (beta, beta, alpha, alpha)
    const double energy_bbaa = 
        compute_aabb_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, d_g_aaaa_full,
            d_orbital_energies_be, d_orbital_energies_al,
            num_occ_be, num_vir_be, num_occ_al, num_vir_al, 
            false
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));


    // (beta, beta, beta, beta)
    const double energy_bbbb = 
        compute_aaaa_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_bbbb_full, d_orbital_energies_be, 
            num_occ_be, num_vir_be
        );
    /**/

    tracked_cudaFree(d_g_aaaa_full);
    tracked_cudaFree(d_g_aabb_full);
    tracked_cudaFree(d_g_bbaa_full);
    tracked_cudaFree(d_g_bbbb_full);
    tracked_cudaFree(d_energy_4h2p_2h4p);
    tracked_cudaFree(d_energy_3h3p);


    return energy_aaaa + energy_aabb + energy_bbaa + energy_bbbb;
}


static void contract_same_spin_contributions_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_4h2p_2h4p,
    double* d_energy_3h3p,
    const double* d_x_ovov,
    const double* d_y_ovov,
    const double* d_g_oooo,
    const double* d_g_vvvv,
    double* d_u_ovvo,
    double* d_tmp_1,
    double* d_tmp_2,
    double* d_tmp_3,
    const dim3 threads,
    const size_t num_blocks_ovov,
    const int num_occ,
    const int num_vir)
{
    const size_t num_occ_2 = num_occ * num_occ;
    const size_t num_vir_2 = num_vir * num_vir;
    const size_t num_ov_2 = num_occ * num_vir;
    const double alpha = 1.0;
    const double beta = 0.0;

    if (!gpu::gpu_available()) {
        using namespace cpu_kernels;
        cpu_dgemm_row_major(num_ov_2, num_ov_2, num_ov_2, 1.0, d_y_ovov, d_u_ovvo, d_tmp_1, false, false);
        *d_energy_3h3p += contract_3h3p_aaaaaa_cpu(d_y_ovov, d_tmp_1, num_occ, num_vir);

        kalb2klab_aaaa_cpu(d_tmp_1, d_y_ovov, num_occ, num_vir);
        cpu_dgemm_row_major(num_occ_2, num_vir_2, num_occ_2, 1.0, d_g_oooo, d_tmp_1, d_tmp_2, false, false);

        icjd2cdij_aaaa_cpu(d_tmp_1, d_y_ovov, num_occ, num_vir);
        cpu_dgemm_row_major(num_vir_2, num_occ_2, num_vir_2, 1.0, d_g_vvvv, d_tmp_1, d_tmp_3, false, false);

        *d_energy_4h2p_2h4p += contract_4h2p_2h4p_aaaaaa_cpu(d_x_ovov, d_tmp_2, d_tmp_3, num_occ, num_vir);
        return;
    }

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_ov_2, num_ov_2, num_ov_2,
        &alpha, d_y_ovov, num_ov_2,
        d_u_ovvo, num_ov_2,
        &beta, d_tmp_1, num_ov_2
    );
    const double* d_t_ovvo = d_tmp_1;
    contract_3h3p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_3h3p, d_y_ovov, d_t_ovvo, num_occ, num_vir);

    kalb2klab_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_ovov, num_occ, num_vir);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_2, num_vir_2, num_occ_2,
        &alpha, d_g_oooo, num_occ_2,
        d_tmp_1, num_vir_2,
        &beta, d_tmp_2, num_vir_2
    );
    const double* d_t_oovv = d_tmp_2;

    icjd2cdij_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_ovov, num_occ, num_vir);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_2, num_occ_2, num_vir_2,
        &alpha, d_g_vvvv, num_vir_2,
        d_tmp_1, num_occ_2,
        &beta, d_tmp_3, num_occ_2
    );
    const double* d_t_vvoo = d_tmp_3;
    contract_4h2p_2h4p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_ovov, d_t_oovv, d_t_vvoo, num_occ, num_vir);

    cudaDeviceSynchronize();
}


static void contract_mixed_yxg_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_y_same_ovov,
    const double* d_x_mixed_ovov,
    const double* d_g_rev_ovvo,
    double* d_tmp_same_ovov,
    const dim3 threads,
    const size_t num_blocks_same_ovov,
    const int num_occ_same,
    const int num_vir_same,
    const int num_occ_other,
    const int num_vir_other)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    if (!gpu::gpu_available()) {
        using namespace cpu_kernels;
        cpu_dgemm_row_major(num_occ_same*num_vir_same, num_vir_same*num_occ_same, num_occ_other*num_vir_other, 1.0, d_x_mixed_ovov, d_g_rev_ovvo, d_tmp_same_ovov, false, false);
        *d_energy_3h3p += contract_3h3p_aabaab_abaaba_cpu(d_y_same_ovov, d_tmp_same_ovov, num_occ_same, num_vir_same);
        return;
    }

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_same * num_vir_same, num_vir_same * num_occ_same, num_occ_other * num_vir_other,
        &alpha, d_x_mixed_ovov, num_occ_other * num_vir_other,
        d_g_rev_ovvo, num_vir_same * num_occ_same,
        &beta, d_tmp_same_ovov, num_vir_same * num_occ_same
    );
    const double* d_t_same_ovvo = d_tmp_same_ovov;
    contract_3h3p_aabaab_abaaba<<<num_blocks_same_ovov, threads>>>(d_energy_3h3p, d_y_same_ovov, d_t_same_ovvo, num_occ_same, num_vir_same);

    cudaDeviceSynchronize();
}


static void contract_mixed_xu_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_x_mixed_ovov,
    const double* d_u_same_ovvo,
    double* d_tmp_mixed_ovov,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    if (!gpu::gpu_available()) {
        using namespace cpu_kernels;
        cpu_dgemm_row_major(num_occ_1*num_vir_1, num_vir_2*num_occ_2, num_occ_2*num_vir_2, 1.0, d_x_mixed_ovov, d_u_same_ovvo, d_tmp_mixed_ovov, false, false);
        *d_energy_3h3p += contract_3h3p_abbabb_cpu(d_x_mixed_ovov, d_tmp_mixed_ovov, num_occ_1, num_occ_2, num_vir_1, num_vir_2);
        return;
    }

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_vir_1, num_vir_2 * num_occ_2, num_occ_2 * num_vir_2,
        &alpha, d_x_mixed_ovov, num_occ_2 * num_vir_2,
        d_u_same_ovvo, num_vir_2 * num_occ_2,
        &beta, d_tmp_mixed_ovov, num_vir_2 * num_occ_2
    );
    const double* d_t_mixed_ovvo = d_tmp_mixed_ovov;
    contract_3h3p_abbabb<<<num_blocks_mixed_ovov, threads>>>(d_energy_3h3p, d_x_mixed_ovov, d_t_mixed_ovvo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


static void contract_mixed_xg_oovv_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_x_mixed_ovov,
    const double* d_g_rev_oovv,
    double* d_tmp_reordered,
    double* d_tmp_mixed_ovov,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    if (!gpu::gpu_available()) {
        using namespace cpu_kernels;
        aabb_icka2abba_iakc_cpu(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
        cpu_dgemm_row_major(num_occ_1*num_vir_2, num_vir_1*num_occ_2, num_occ_2*num_vir_1, 1.0, d_tmp_reordered, d_g_rev_oovv, d_tmp_mixed_ovov, false, false);
        *d_energy_3h3p += contract_3h3p_abbbaa_cpu(d_x_mixed_ovov, d_tmp_mixed_ovov, num_occ_1, num_occ_2, num_vir_1, num_vir_2);
        return;
    }

    aabb_icka2abba_iakc<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_vir_2, num_vir_1 * num_occ_2, num_occ_2 * num_vir_1,
        &alpha, d_tmp_reordered, num_occ_2 * num_vir_1,
        d_g_rev_oovv, num_vir_1 * num_occ_2,
        &beta, d_tmp_mixed_ovov, num_vir_1 * num_occ_2
    );
    const double* d_t_mixed_ovvo = d_tmp_mixed_ovov;
    contract_3h3p_abbbaa<<<num_blocks_mixed_ovov, threads>>>(d_energy_3h3p, d_x_mixed_ovov, d_t_mixed_ovvo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


static void contract_mixed_4h2p_2h4p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_4h2p_2h4p,
    const double* d_x_mixed_ovov,
    const double* d_g_mixed_oooo,
    const double* d_g_mixed_vvvv,
    double* d_tmp_reordered,
    double* d_tmp_oovv,
    double* d_tmp_vvoo,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    if (!gpu::gpu_available()) {
        using namespace cpu_kernels;
        aabb_kalb2abab_klab_cpu(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
        cpu_dgemm_row_major(num_occ_1*num_occ_2, num_vir_1*num_vir_2, num_occ_1*num_occ_2, 1.0, d_g_mixed_oooo, d_tmp_reordered, d_tmp_oovv, false, false);
        aabb_icjd2abab_cdij_cpu(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
        cpu_dgemm_row_major(num_vir_1*num_vir_2, num_occ_1*num_occ_2, num_vir_1*num_vir_2, 1.0, d_g_mixed_vvvv, d_tmp_reordered, d_tmp_vvoo, false, false);
        *d_energy_4h2p_2h4p += contract_4h2p_2h4p_ababab_bababa_cpu(d_x_mixed_ovov, d_tmp_oovv, d_tmp_vvoo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);
        return;
    }

    aabb_kalb2abab_klab<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_occ_2, num_vir_1 * num_vir_2, num_occ_1 * num_occ_2,
        &alpha, d_g_mixed_oooo, num_occ_1 * num_occ_2,
        d_tmp_reordered, num_vir_1 * num_vir_2,
        &beta, d_tmp_oovv, num_vir_1 * num_vir_2
    );
    const double* d_t_oovv = d_tmp_oovv;

    aabb_icjd2abab_cdij<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_1 * num_vir_2, num_occ_1 * num_occ_2, num_vir_1 * num_vir_2,
        &alpha, d_g_mixed_vvvv, num_vir_1 * num_vir_2,
        d_tmp_reordered, num_occ_1 * num_occ_2,
        &beta, d_tmp_vvoo, num_occ_1 * num_occ_2
    );
    const double* d_t_vvoo = d_tmp_vvoo;

    contract_4h2p_2h4p_ababab_bababa<<<num_blocks_mixed_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_mixed_ovov, d_t_oovv, d_t_vvoo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


// GPU kernel for trimming MO ERI (defined in eri_stored.cu)
__global__ void trim_eri_frozen_core_kernel(const real_t* __restrict__ eri_full,
                                            real_t* __restrict__ eri_trimmed,
                                            int N_full, int na_active, int offset);

// ============================================================
//  Swapped-source tensorize kernels: pack the (bb|aa) compact
//  blocks from the (aa|bb) full MO ERI tensor via ERI pair
//  symmetry, full_bbaa[k,c,b,j] = full_aabb[b,j,k,c].
//  Compact layouts and denominators are identical to the
//  originals in ao2mo.cuh; only the source index read differs.
// ============================================================
static __global__ void tensorize_g_bbaa_ovvo_swapped(
    double* g_g_bbaa_ovvo, const double* g_g_aabb_full,
    const int num_occ_be, const int num_occ_al,
    const int num_vir_be, const int num_vir_al, const int num_basis)
{
    const size_t kcbj = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovvo = (size_t)num_occ_be * num_vir_be * num_vir_al * num_occ_al;
    if (kcbj >= num_ovvo) {
        return;
    }

    const int kc = kcbj / (num_vir_al * num_occ_al);
    const int bj = kcbj % (num_vir_al * num_occ_al);
    const int k = kc / num_vir_be;
    const int c = kc % num_vir_be + num_occ_be;
    const int b = bj / num_occ_al + num_occ_al;
    const int j = bj % num_occ_al;

    g_g_bbaa_ovvo[ovvo2s_bbaa(k, c, b, j, num_occ_be, num_occ_al, num_vir_be, num_vir_al)] = g_g_aabb_full[q2s(b, j, k, c, num_basis)];
}

static __global__ void tensorize_g_bbaa_oovv_swapped(
    double* g_g_bbaa_oovv, const double* g_g_aabb_full,
    const int num_occ_be, const int num_occ_al,
    const int num_vir_be, const int num_vir_al, const int num_basis)
{
    const size_t kjbc = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_be * num_occ_be * num_vir_al * num_vir_al;
    if (kjbc >= num_oovv) {
        return;
    }

    const int kj = kjbc / (num_vir_al * num_vir_al);
    const int bc = kjbc % (num_vir_al * num_vir_al);
    const int k = kj / num_occ_be;
    const int j = kj % num_occ_be;
    const int b = bc / num_vir_al + num_occ_al;
    const int c = bc % num_vir_al + num_occ_al;

    g_g_bbaa_oovv[ovvo2s_baab(k, c, b, j, num_occ_be, num_occ_al, num_vir_al, num_vir_be)] = g_g_aabb_full[q2s(b, c, k, j, num_basis)];
}

static __global__ void tensorize_x_aabb_ovov_swapped(
    double* g_x_aabb_ovov, const double* g_g_aabb_full,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be, const int num_basis)
{
    const size_t iajb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    if (iajb >= num_ovov) {
        return;
    }

    const int ia = iajb / (num_occ_be * num_vir_be);
    const int jb = iajb % (num_occ_be * num_vir_be);
    const int i = ia / num_vir_al;
    const int a = ia % num_vir_al + num_occ_al;
    const int j = jb / num_vir_be;
    const int b = jb % num_vir_be + num_occ_be;

    const double g_iajb = g_g_aabb_full[q2s(j, b, i, a, num_basis)];
    const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];

    g_x_aabb_ovov[ovov2s_aabb(i, a, j, b, num_occ_al, num_vir_al, num_occ_be, num_vir_be)] = g_iajb / eps_ijab;
}

double ump3_from_aoeri_via_full_moeri_dgemm_eff(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis,
    const int num_occ_al,
    const int num_occ_be,
    const int num_frozen = 0)
{
    const int active_occ_al = num_occ_al - num_frozen;
    const int active_occ_be = num_occ_be - num_frozen;
    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;
    const int na_active = num_basis - num_frozen;  // = active_occ + num_vir (same for al/be)

    // Shifted epsilon pointers for frozen core
    const double* d_eps_al = d_orbital_energies_al + num_frozen;
    const double* d_eps_be = d_orbital_energies_be + num_frozen;

    double* d_g_full = nullptr;
    double* d_eri_tmp = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_eri_tmp, sizeof(double) * num_basis_4);
    if (!d_g_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_full."); }
    if (!d_eri_tmp) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp."); }

    double* d_energy_4h2p_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    const dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_aaaa_oooo = (size_t)active_occ_al * active_occ_al * active_occ_al * active_occ_al;
    const size_t num_aaaa_vvvv = (size_t)num_vir_al * num_vir_al * num_vir_al * num_vir_al;
    const size_t num_aaaa_ovov = (size_t)active_occ_al * num_vir_al * active_occ_al * num_vir_al;
    const size_t num_aabb_oooo = (size_t)active_occ_al * active_occ_be * active_occ_al * active_occ_be;
    const size_t num_aabb_vvvv = (size_t)num_vir_al * num_vir_be * num_vir_al * num_vir_be;
    const size_t num_aabb_ovov = (size_t)active_occ_al * num_vir_al * active_occ_be * num_vir_be;
    const size_t num_aabb_oovv = (size_t)active_occ_al * active_occ_al * num_vir_be * num_vir_be;
    const size_t num_bbaa_oovv = (size_t)active_occ_be * active_occ_be * num_vir_al * num_vir_al;
    const size_t num_bbbb_oooo = (size_t)active_occ_be * active_occ_be * active_occ_be * active_occ_be;
    const size_t num_bbbb_vvvv = (size_t)num_vir_be * num_vir_be * num_vir_be * num_vir_be;
    const size_t num_bbbb_ovov = (size_t)active_occ_be * num_vir_be * active_occ_be * num_vir_be;
    const size_t num_same_ovov_max = std::max(num_aaaa_ovov, num_bbbb_ovov);

    const size_t num_blocks_aaaa_oooo = (num_aaaa_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_vvvv = (num_aaaa_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_ovov = (num_aaaa_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oooo = (num_aabb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_vvvv = (num_aabb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_ovov = (num_aabb_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oovv = (num_aabb_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_oovv = (num_bbaa_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_oooo = (num_bbbb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_vvvv = (num_bbbb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_ovov = (num_bbbb_ovov + num_threads_per_block - 1) / num_threads_per_block;

    if (gpu::gpu_available()) {
        cudaDeviceProp prop;
        int device = 0;
        cudaGetDeviceProperties(&prop, device);

        const auto check_num_blocks = [&prop](const size_t num_blocks) {
            if (num_blocks > static_cast<size_t>(prop.maxGridSize[0])) {
                THROW_EXCEPTION("Error: Too many blocks for the grid size.");
            }
        };

        check_num_blocks(num_blocks_aaaa_oooo);
        check_num_blocks(num_blocks_aaaa_vvvv);
        check_num_blocks(num_blocks_aaaa_ovov);
        check_num_blocks(num_blocks_aabb_oooo);
        check_num_blocks(num_blocks_aabb_vvvv);
        check_num_blocks(num_blocks_aabb_ovov);
        check_num_blocks(num_blocks_aabb_oovv);
        check_num_blocks(num_blocks_bbaa_oovv);
        check_num_blocks(num_blocks_bbbb_oooo);
        check_num_blocks(num_blocks_bbbb_vvvv);
        check_num_blocks(num_blocks_bbbb_ovov);
    }

    cublasHandle_t cublasH = NULL;
    if (gpu::gpu_available()) {
        cublasCreate(&cublasH);
    }

    double* d_tmp_same_1 = nullptr;
    double* d_tmp_same_2 = nullptr;
    double* d_tmp_same_3 = nullptr;
    double* d_tmp_mixed_1 = nullptr;
    double* d_tmp_mixed_2 = nullptr;
    double* d_tmp_mixed_3 = nullptr;
    tracked_cudaMalloc(&d_tmp_same_1, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_2, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_3, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_mixed_1, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_2, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_3, sizeof(double) * num_aabb_ovov);
    if (!d_tmp_same_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_1."); }
    if (!d_tmp_same_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_2."); }
    if (!d_tmp_same_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_3."); }
    if (!d_tmp_mixed_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_1."); }
    if (!d_tmp_mixed_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_2."); }
    if (!d_tmp_mixed_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_3."); }

    double* d_y_aaaa_ovov = nullptr;
    double* d_u_aaaa_ovvo = nullptr;
    double* d_x_aabb_ovov = nullptr;
    double* d_g_aabb_oovv = nullptr;
    double* d_g_aabb_ovvo = nullptr;
    double* d_x_bbaa_ovov = nullptr;

    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_u_aaaa_ovvo, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_x_aabb_ovov, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_g_aabb_oovv, sizeof(double) * num_aabb_oovv);
    tracked_cudaMalloc(&d_g_aabb_ovvo, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_x_bbaa_ovov, sizeof(double) * num_aabb_ovov);
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_u_aaaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_aaaa_ovvo."); }
    if (!d_x_aabb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aabb_ovov."); }
    if (!d_g_aabb_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oovv."); }
    if (!d_g_aabb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_ovvo."); }
    if (!d_x_bbaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbaa_ovov."); }

    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
    cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);

    // Helper: trim full MO ERI to active space when frozen core is present
    // After transform, d_g_full is num_basis^4. Trim into d_eri_tmp (reused as temp).
    auto trim_eri = [&](double* d_src, double* d_dst) {
        if (num_frozen == 0) return d_src;  // no trim needed
        const size_t na4 = (size_t)na_active * na_active * na_active * na_active;
        if (!gpu::gpu_available()) {
            const size_t N = num_basis;
            #pragma omp parallel for collapse(2)
            for (int p = 0; p < na_active; p++)
                for (int q = 0; q < na_active; q++)
                    for (int r = 0; r < na_active; r++)
                        for (int s = 0; s < na_active; s++) {
                            size_t src = ((size_t)(num_frozen+p)*N + (num_frozen+q))*N*N
                                       + (size_t)(num_frozen+r)*N + (num_frozen+s);
                            size_t dst_idx = ((size_t)p*na_active*na_active + (size_t)q*na_active + r)*(size_t)na_active + s;
                            d_dst[dst_idx] = d_src[src];
                        }
        } else {
            int thr = 256;
            int blk = (int)((na4 + thr - 1) / thr);
            trim_eri_frozen_core_kernel<<<blk, thr>>>(d_src, d_dst, num_basis, na_active, num_frozen);
            cudaDeviceSynchronize();
        }
        return d_dst;
    };

    // (αα|αα) block
    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_al, d_coefficient_matrix_al,
            num_basis_4, num_basis, true
        );
        double* d_g_src = trim_eri(d_g_full, d_eri_tmp);

        double* d_g_aaaa_oooo = nullptr;
        double* d_g_aaaa_vvvv = nullptr;
        double* d_x_aaaa_ovov = nullptr;
        tracked_cudaMalloc(&d_g_aaaa_oooo, sizeof(double) * num_aaaa_oooo);
        tracked_cudaMalloc(&d_g_aaaa_vvvv, sizeof(double) * num_aaaa_vvvv);
        tracked_cudaMalloc(&d_x_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
        if (!d_g_aaaa_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_oooo."); }
        if (!d_g_aaaa_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_vvvv."); }
        if (!d_x_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aaaa_ovov."); }

        cudaMemset(d_g_aaaa_oooo, 0, sizeof(double) * num_aaaa_oooo);
        cudaMemset(d_g_aaaa_vvvv, 0, sizeof(double) * num_aaaa_vvvv);
        cudaMemset(d_x_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aaaa_oooo_cpu(d_g_aaaa_oooo, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_g_aaaa_vvvv_cpu(d_g_aaaa_vvvv, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_u_aaaa_ovvo_cpu(d_u_aaaa_ovvo, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_x_aaaa_ovov_cpu(d_x_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_y_aaaa_ovov_cpu(d_y_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
        } else {
            tensorize_g_aaaa_oooo<<<num_blocks_aaaa_oooo, threads>>>(d_g_aaaa_oooo, d_g_src, active_occ_al, num_vir_al);
            tensorize_g_aaaa_vvvv<<<num_blocks_aaaa_vvvv, threads>>>(d_g_aaaa_vvvv, d_g_src, active_occ_al, num_vir_al);
            tensorize_u_aaaa_ovvo<<<num_blocks_aaaa_ovov, threads>>>(d_u_aaaa_ovvo, d_g_src, active_occ_al, num_vir_al);
            tensorize_x_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_x_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
            tensorize_y_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_y_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
        }

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_aaaa_ovov, d_y_aaaa_ovov,
            d_g_aaaa_oooo, d_g_aaaa_vvvv,
            d_u_aaaa_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_aaaa_ovov,
            active_occ_al, num_vir_al
        );

        tracked_cudaFree(d_g_aaaa_oooo);
        tracked_cudaFree(d_g_aaaa_vvvv);
        tracked_cudaFree(d_x_aaaa_ovov);
    }

    // (αα|ββ) block
    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_al, d_coefficient_matrix_be,
            num_basis_4, num_basis, false
        );
        double* d_g_src = trim_eri(d_g_full, d_eri_tmp);

        double* d_g_aabb_oooo = nullptr;
        double* d_g_aabb_vvvv = nullptr;
        tracked_cudaMalloc(&d_g_aabb_oooo, sizeof(double) * num_aabb_oooo);
        tracked_cudaMalloc(&d_g_aabb_vvvv, sizeof(double) * num_aabb_vvvv);
        if (!d_g_aabb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oooo."); }
        if (!d_g_aabb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_vvvv."); }

        cudaMemset(d_g_aabb_oooo, 0, sizeof(double) * num_aabb_oooo);
        cudaMemset(d_g_aabb_vvvv, 0, sizeof(double) * num_aabb_vvvv);
        cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
        cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aabb_oooo_cpu(d_g_aabb_oooo, d_g_src, active_occ_al, active_occ_be, na_active);
            cpu_kernels::tensorize_g_aabb_vvvv_cpu(d_g_aabb_vvvv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            cpu_kernels::tensorize_x_aabb_ovov_cpu(d_x_aabb_ovov, d_g_src, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
            cpu_kernels::tensorize_g_bbaa_oovv_cpu(d_g_aabb_oovv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            cpu_kernels::tensorize_g_bbaa_ovvo_cpu(d_g_aabb_ovvo, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
        } else {
            tensorize_g_aabb_oooo<<<num_blocks_aabb_oooo, threads>>>(d_g_aabb_oooo, d_g_src, active_occ_al, active_occ_be, na_active);
            tensorize_g_aabb_vvvv<<<num_blocks_aabb_vvvv, threads>>>(d_g_aabb_vvvv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_aabb_ovov, d_g_src, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
            tensorize_g_bbaa_oovv<<<num_blocks_aabb_oovv, threads>>>(d_g_aabb_oovv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            tensorize_g_bbaa_ovvo<<<num_blocks_aabb_ovov, threads>>>(d_g_aabb_ovvo, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
        }

        contract_mixed_4h2p_2h4p_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p,
            d_x_aabb_ovov,
            d_g_aabb_oooo,
            d_g_aabb_vvvv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            d_tmp_mixed_3,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        tracked_cudaFree(d_g_aabb_oooo);
        tracked_cudaFree(d_g_aabb_vvvv);
    }

    // (ββ|αα) block
    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_be, d_coefficient_matrix_al,
            num_basis_4, num_basis, false
        );
        double* d_g_src = trim_eri(d_g_full, d_eri_tmp);

        double* d_g_bbaa_ovvo = nullptr;
        double* d_g_bbaa_oovv = nullptr;
        tracked_cudaMalloc(&d_g_bbaa_ovvo, sizeof(double) * num_aabb_ovov);
        tracked_cudaMalloc(&d_g_bbaa_oovv, sizeof(double) * num_bbaa_oovv);
        if (!d_g_bbaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_ovvo."); }
        if (!d_g_bbaa_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_oovv."); }

        cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_ovvo, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_oovv, 0, sizeof(double) * num_bbaa_oovv);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_x_aabb_ovov_cpu(d_x_bbaa_ovov, d_g_src, d_eps_be, d_eps_al, active_occ_be, num_vir_be, active_occ_al, num_vir_al, na_active);
            cpu_kernels::tensorize_g_bbaa_ovvo_cpu(d_g_bbaa_ovvo, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
            cpu_kernels::tensorize_g_bbaa_oovv_cpu(d_g_bbaa_oovv, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
        } else {
            tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_bbaa_ovov, d_g_src, d_eps_be, d_eps_al, active_occ_be, num_vir_be, active_occ_al, num_vir_al, na_active);
            tensorize_g_bbaa_ovvo<<<num_blocks_aabb_ovov, threads>>>(d_g_bbaa_ovvo, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
            tensorize_g_bbaa_oovv<<<num_blocks_bbaa_oovv, threads>>>(d_g_bbaa_oovv, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
        }

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_aaaa_ovov,
            d_x_aabb_ovov,
            d_g_bbaa_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_aaaa_ovov,
            active_occ_al, num_vir_al,
            active_occ_be, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_g_bbaa_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_g_aabb_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            active_occ_be, active_occ_al, num_vir_be, num_vir_al
        );

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_u_aaaa_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            active_occ_be, active_occ_al, num_vir_be, num_vir_al
        );

        tracked_cudaFree(d_g_bbaa_ovvo);
        tracked_cudaFree(d_g_bbaa_oovv);
        tracked_cudaFree(d_y_aaaa_ovov);
        tracked_cudaFree(d_u_aaaa_ovvo);
        tracked_cudaFree(d_g_aabb_oovv);
        d_y_aaaa_ovov = nullptr;
        d_u_aaaa_ovvo = nullptr;
        d_g_aabb_oovv = nullptr;
    }

    // (ββ|ββ) block
    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_be, d_coefficient_matrix_be,
            num_basis_4, num_basis, true
        );
        double* d_g_src = trim_eri(d_g_full, d_eri_tmp);

        double* d_g_bbbb_oooo = nullptr;
        double* d_g_bbbb_vvvv = nullptr;
        double* d_u_bbbb_ovvo = nullptr;
        double* d_x_bbbb_ovov = nullptr;
        double* d_y_bbbb_ovov = nullptr;
        tracked_cudaMalloc(&d_g_bbbb_oooo, sizeof(double) * num_bbbb_oooo);
        tracked_cudaMalloc(&d_g_bbbb_vvvv, sizeof(double) * num_bbbb_vvvv);
        tracked_cudaMalloc(&d_u_bbbb_ovvo, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_x_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_y_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        if (!d_g_bbbb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_oooo."); }
        if (!d_g_bbbb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_vvvv."); }
        if (!d_u_bbbb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_bbbb_ovvo."); }
        if (!d_x_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbbb_ovov."); }
        if (!d_y_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_bbbb_ovov."); }

        cudaMemset(d_g_bbbb_oooo, 0, sizeof(double) * num_bbbb_oooo);
        cudaMemset(d_g_bbbb_vvvv, 0, sizeof(double) * num_bbbb_vvvv);
        cudaMemset(d_u_bbbb_ovvo, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_x_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_y_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aaaa_oooo_cpu(d_g_bbbb_oooo, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_g_aaaa_vvvv_cpu(d_g_bbbb_vvvv, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_u_aaaa_ovvo_cpu(d_u_bbbb_ovvo, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_x_aaaa_ovov_cpu(d_x_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_y_aaaa_ovov_cpu(d_y_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
        } else {
            tensorize_g_aaaa_oooo<<<num_blocks_bbbb_oooo, threads>>>(d_g_bbbb_oooo, d_g_src, active_occ_be, num_vir_be);
            tensorize_g_aaaa_vvvv<<<num_blocks_bbbb_vvvv, threads>>>(d_g_bbbb_vvvv, d_g_src, active_occ_be, num_vir_be);
            tensorize_u_aaaa_ovvo<<<num_blocks_bbbb_ovov, threads>>>(d_u_bbbb_ovvo, d_g_src, active_occ_be, num_vir_be);
            tensorize_x_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_x_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
            tensorize_y_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_y_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
        }

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_u_bbbb_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_bbbb_ovov,
            d_x_bbaa_ovov,
            d_g_aabb_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_bbbb_ovov,
            active_occ_be, num_vir_be,
            active_occ_al, num_vir_al
        );

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_bbbb_ovov, d_y_bbbb_ovov,
            d_g_bbbb_oooo, d_g_bbbb_vvvv,
            d_u_bbbb_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_bbbb_ovov,
            active_occ_be, num_vir_be
        );

        tracked_cudaFree(d_g_bbbb_oooo);
        tracked_cudaFree(d_g_bbbb_vvvv);
        tracked_cudaFree(d_u_bbbb_ovvo);
        tracked_cudaFree(d_x_bbbb_ovov);
        tracked_cudaFree(d_y_bbbb_ovov);
    }

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);

    if (gpu::gpu_available()) {
        cublasDestroy(cublasH);
    }

    tracked_cudaFree(d_g_full);
    tracked_cudaFree(d_eri_tmp);
    tracked_cudaFree(d_energy_4h2p_2h4p);
    tracked_cudaFree(d_energy_3h3p);
    tracked_cudaFree(d_tmp_same_1);
    tracked_cudaFree(d_tmp_same_2);
    tracked_cudaFree(d_tmp_same_3);
    tracked_cudaFree(d_tmp_mixed_1);
    tracked_cudaFree(d_tmp_mixed_2);
    tracked_cudaFree(d_tmp_mixed_3);
    tracked_cudaFree(d_x_aabb_ovov);
    tracked_cudaFree(d_g_aabb_ovvo);
    tracked_cudaFree(d_x_bbaa_ovov);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}

// ============================================================
//  UMP3 via spin-transform reuse (peak 2 x M^4 buffers, GPU only)
// ============================================================
// The reference path transforms the AO ERIs four times (aaaa, aabb, bbaa,
// bbbb), peaking at 3 M^4 buffers (AO + d_g_full + d_eri_tmp). This path
// builds the alpha-spin MO ERIs once and obtains the beta-spin blocks by
// rotating axes with the spin transformation matrix
//
//     U = C_alpha^T S C_beta
//
// (C_beta = C_alpha U holds exactly because both C_alpha and C_beta are
// complete S-orthonormal bases of the same num_basis-dimensional space):
//
//     G_aabb[p,q,r,s] = sum_{x,y} G_aaaa[p,q,x,y] U(x,r) U(y,s)  (axes 2,3)
//     G_bbbb[k,c,b,j] = sum_{x,y} G_aabb[x,y,b,j] U(x,k) U(y,c)  (axes 0,1)
//
// The (bb|aa) compact blocks are packed from the aabb tensor via ERI pair
// symmetry (full_bbaa[k,c,b,j] = full_aabb[b,j,k,c]).
//
// Buffer ping-pong (B0 = d_eri_ao, reused as scratch and destroyed;
// B1 = d_g_full):
//   stage A: AO --(full transform, C_alpha)--> aaaa in B1
//   stage B: B1 --rot axis 3--> B0 --transpose--> B1 --rot axis 0--> B0
//            --transpose--> B1 (= aabb)
//   stage C: B1 --rot axis 1 (batched)--> B0 --rot axis 0--> B1 (= bbbb)
//
// The contraction sequence is identical to the reference implementation.
// U is verified on the host before the AO ERIs are touched; if the check
// fails, the reference path runs instead and *out_ao_consumed stays false.

// Tolerance for the spin rotation matrix consistency check.
constexpr double kSpinRotationTolerance = 1e-8;

// Builds U = C_alpha^T S C_beta and returns the maximum deviation of the
// consistency checks max|C_alpha U - C_beta| and max|U^T U - I|.
// (C_alpha and C_beta are complete S-orthonormal bases, so both should hold
// to SCF orthonormality precision, ~1e-14.) The built matrix is handed out
// through *out_d_U (tracked allocation; the caller owns and frees it).
// Used both inside ump3_from_aoeri_via_cascade and by callers that need
// the check result before deciding whether to enter the fused UMP2/UMP3
// workflow, so both see the identical (deterministic) decision.
static double build_spin_rotation_matrix(
    double** out_d_U,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_overlap_matrix,
    const int num_basis)
{
    const size_t num_basis_2 = (size_t)num_basis * num_basis;

    double* d_U = nullptr;
    double* d_U_tmp = nullptr;
    tracked_cudaMalloc(&d_U, sizeof(double) * num_basis_2);
    tracked_cudaMalloc(&d_U_tmp, sizeof(double) * num_basis_2);
    if (!d_U) { THROW_EXCEPTION("cudaMalloc failed for d_U."); }
    if (!d_U_tmp) { THROW_EXCEPTION("cudaMalloc failed for d_U_tmp."); }

    {
        if (gpu::gpu_available()) {
            cublasHandle_t cublasH = NULL;
            cublasCreate(&cublasH);
            const double alpha = 1.0;
            const double beta = 0.0;
            // U_tmp = C_alpha^T S
            dgemm_device_row_major(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                num_basis, num_basis, num_basis, &alpha,
                d_coefficient_matrix_al, num_basis,
                d_overlap_matrix, num_basis, &beta,
                d_U_tmp, num_basis);
            // U = U_tmp C_beta
            dgemm_device_row_major(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                num_basis, num_basis, num_basis, &alpha,
                d_U_tmp, num_basis,
                d_coefficient_matrix_be, num_basis, &beta,
                d_U, num_basis);
            cublasDestroy(cublasH);
        } else {
            // U_tmp = C_alpha^T S
            cpu_dgemm_ld_row_major(CUBLAS_OP_T, CUBLAS_OP_N,
                num_basis, num_basis, num_basis,
                d_coefficient_matrix_al, num_basis,
                d_overlap_matrix, num_basis,
                d_U_tmp, num_basis);
            // U = U_tmp C_beta
            cpu_dgemm_ld_row_major(CUBLAS_OP_N, CUBLAS_OP_N,
                num_basis, num_basis, num_basis,
                d_U_tmp, num_basis,
                d_coefficient_matrix_be, num_basis,
                d_U, num_basis);
        }
        cudaDeviceSynchronize();
    }

    // Host-side verification: max |C_alpha U - C_beta| and max |U^T U - I|.
    double max_dev = 0.0;
    {
        std::vector<double> h_Ca(num_basis_2), h_Cb(num_basis_2), h_U(num_basis_2);
        cudaMemcpy(h_Ca.data(), d_coefficient_matrix_al, sizeof(double) * num_basis_2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Cb.data(), d_coefficient_matrix_be, sizeof(double) * num_basis_2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_U.data(), d_U, sizeof(double) * num_basis_2, cudaMemcpyDeviceToHost);

        for (int mu = 0; mu < num_basis; mu++) {
            for (int q = 0; q < num_basis; q++) {
                double sum = 0.0;
                for (int p = 0; p < num_basis; p++) {
                    sum += h_Ca[(size_t)mu * num_basis + p] * h_U[(size_t)p * num_basis + q];
                }
                max_dev = std::max(max_dev, std::fabs(sum - h_Cb[(size_t)mu * num_basis + q]));
            }
        }
        for (int p = 0; p < num_basis; p++) {
            for (int q = 0; q < num_basis; q++) {
                double sum = 0.0;
                for (int x = 0; x < num_basis; x++) {
                    sum += h_U[(size_t)x * num_basis + p] * h_U[(size_t)x * num_basis + q];
                }
                max_dev = std::max(max_dev, std::fabs(sum - (p == q ? 1.0 : 0.0)));
            }
        }
    }

    tracked_cudaFree(d_U_tmp);
    *out_d_U = d_U;
    return max_dev;
}

double ump3_from_aoeri_via_cascade(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_overlap_matrix,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis,
    const int num_occ_al,
    const int num_occ_be,
    const int num_frozen = 0,
    bool* out_ao_consumed = nullptr,
    double* out_ump2_energy = nullptr,
    bool* out_ump2_from_superset = nullptr)
{
    if (out_ao_consumed) { *out_ao_consumed = false; }
    if (out_ump2_from_superset) { *out_ump2_from_superset = false; }
    // ump3_superset: when out_ump2_energy is non-null, the UMP2 correlation
    // energy is obtained from the cascade intermediates (the cascade UMP3
    // intermediates are a strict superset of the canonical UMP2 inputs),
    // making the standalone UMP2 phase -- a full extra pass over the AO ERIs
    // -- unnecessary.
    const bool superset_ump2 = (out_ump2_energy != nullptr);

    // NOTE: the CPU backend no longer falls back to the reference (eff)
    // implementation. Both backends now execute the identical cascade
    // algorithm — the same GEMM/transform sequence, tensorization kernels,
    // and contraction order — so GPU-vs-CPU comparisons reflect hardware
    // differences, not algorithm differences. The dispatchers below choose
    // cuBLAS (GPU) or Eigen (CPU, OpenMP-threaded) for each GEMM.

    const int active_occ_al = num_occ_al - num_frozen;
    const int active_occ_be = num_occ_be - num_frozen;
    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;
    const int na_active = num_basis - num_frozen;  // = active_occ + num_vir (same for al/be)

    // Shifted epsilon pointers for frozen core
    const double* d_eps_al = d_orbital_energies_al + num_frozen;
    const double* d_eps_be = d_orbital_energies_be + num_frozen;

    double* d_g_full = nullptr;       // B1 (the only newly allocated M^4 buffer)
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_full, sizeof(double) * num_basis_4);
    if (!d_g_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_full."); }

    double* d_energy_4h2p_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    // UMP2 energy accumulator for the fused workflow (ss_aa + os_ab + ss_bb,
    // accumulated by the existing compute_ump2_energy_contrib_* kernels).
    double* d_ump2_energy = nullptr;
    if (superset_ump2) {
        tracked_cudaMalloc(&d_ump2_energy, sizeof(double));
        if (!d_ump2_energy) { THROW_EXCEPTION("cudaMalloc failed for d_ump2_energy."); }
        cudaMemset(d_ump2_energy, 0, sizeof(double));
    }

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    const dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_aaaa_oooo = (size_t)active_occ_al * active_occ_al * active_occ_al * active_occ_al;
    const size_t num_aaaa_vvvv = (size_t)num_vir_al * num_vir_al * num_vir_al * num_vir_al;
    const size_t num_aaaa_ovov = (size_t)active_occ_al * num_vir_al * active_occ_al * num_vir_al;
    const size_t num_aabb_oooo = (size_t)active_occ_al * active_occ_be * active_occ_al * active_occ_be;
    const size_t num_aabb_vvvv = (size_t)num_vir_al * num_vir_be * num_vir_al * num_vir_be;
    const size_t num_aabb_ovov = (size_t)active_occ_al * num_vir_al * active_occ_be * num_vir_be;
    const size_t num_aabb_oovv = (size_t)active_occ_al * active_occ_al * num_vir_be * num_vir_be;
    const size_t num_bbaa_oovv = (size_t)active_occ_be * active_occ_be * num_vir_al * num_vir_al;
    const size_t num_bbbb_oooo = (size_t)active_occ_be * active_occ_be * active_occ_be * active_occ_be;
    const size_t num_bbbb_vvvv = (size_t)num_vir_be * num_vir_be * num_vir_be * num_vir_be;
    const size_t num_bbbb_ovov = (size_t)active_occ_be * num_vir_be * active_occ_be * num_vir_be;
    const size_t num_same_ovov_max = std::max(num_aaaa_ovov, num_bbbb_ovov);

    const size_t num_blocks_aaaa_oooo = (num_aaaa_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_vvvv = (num_aaaa_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_ovov = (num_aaaa_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oooo = (num_aabb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_vvvv = (num_aabb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_ovov = (num_aabb_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oovv = (num_aabb_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_oovv = (num_bbaa_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_oooo = (num_bbbb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_vvvv = (num_bbbb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_ovov = (num_bbbb_ovov + num_threads_per_block - 1) / num_threads_per_block;

    {
        if (gpu::gpu_available()) {
            cudaDeviceProp prop;
            int device = 0;
            cudaGetDeviceProperties(&prop, device);

            const auto check_num_blocks = [&prop](const size_t num_blocks) {
                if (num_blocks > static_cast<size_t>(prop.maxGridSize[0])) {
                    THROW_EXCEPTION("Error: Too many blocks for the grid size.");
                }
            };

            check_num_blocks(num_blocks_aaaa_oooo);
            check_num_blocks(num_blocks_aaaa_vvvv);
            check_num_blocks(num_blocks_aaaa_ovov);
            check_num_blocks(num_blocks_aabb_oooo);
            check_num_blocks(num_blocks_aabb_vvvv);
            check_num_blocks(num_blocks_aabb_ovov);
            check_num_blocks(num_blocks_aabb_oovv);
            check_num_blocks(num_blocks_bbaa_oovv);
            check_num_blocks(num_blocks_bbbb_oooo);
            check_num_blocks(num_blocks_bbbb_vvvv);
            check_num_blocks(num_blocks_bbbb_ovov);
        }
    }

    cublasHandle_t cublasH = NULL;
    if (gpu::gpu_available()) {
        cublasCreate(&cublasH);
    }

    // Backend-agnostic GEMM/transpose dispatchers: on GPU they call the
    // cuBLAS wrappers of ao2mo.cuh; on CPU they run the identical operation
    // through Eigen (OpenMP-threaded), preserving operand shapes, strides,
    // and the ping-pong buffer schedule of the GPU path.
    auto dgemm_rm = [cublasH](cublasOperation_t opA, cublasOperation_t opB,
                              int m, int n, int k,
                              const double* A, int lda,
                              const double* B, int ldb,
                              double* C, int ldc) {
        const double alpha = 1.0;
        const double beta = 0.0;
        if (gpu::gpu_available()) {
            dgemm_device_row_major(cublasH, opA, opB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
        } else {
            cpu_dgemm_ld_row_major(opA, opB, m, n, k, A, lda, B, ldb, C, ldc);
        }
    };
    auto dgeam_transpose_rm = [cublasH](int rows_rm, int cols_rm, const double* A, double* AT) {
        if (gpu::gpu_available()) {
            dgeam_transpose_row_major(cublasH, rows_rm, cols_rm, A, AT);
        } else {
            cpu_dgeam_transpose_row_major(rows_rm, cols_rm, A, AT);
        }
    };
    auto dgemm_batched_rm = [cublasH](cublasOperation_t opA, cublasOperation_t opB,
                                      int m, int n, int k,
                                      const double* A, int lda, long long int strideA,
                                      const double* B, int ldb, long long int strideB,
                                      double* C, int ldc, long long int strideC,
                                      int batchCount) {
        const double alpha = 1.0;
        const double beta = 0.0;
        if (gpu::gpu_available()) {
            dgemm_strided_batched_device_row_major(cublasH, opA, opB, m, n, k, &alpha,
                A, lda, strideA, B, ldb, strideB, &beta, C, ldc, strideC, batchCount);
        } else {
            cpu_dgemm_strided_batched_ld_row_major(opA, opB, m, n, k,
                A, lda, strideA, B, ldb, strideB, C, ldc, strideC, batchCount);
        }
    };

    double* d_tmp_same_1 = nullptr;
    double* d_tmp_same_2 = nullptr;
    double* d_tmp_same_3 = nullptr;
    double* d_tmp_mixed_1 = nullptr;
    double* d_tmp_mixed_2 = nullptr;
    double* d_tmp_mixed_3 = nullptr;
    tracked_cudaMalloc(&d_tmp_same_1, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_2, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_3, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_mixed_1, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_2, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_3, sizeof(double) * num_aabb_ovov);
    if (!d_tmp_same_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_1."); }
    if (!d_tmp_same_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_2."); }
    if (!d_tmp_same_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_3."); }
    if (!d_tmp_mixed_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_1."); }
    if (!d_tmp_mixed_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_2."); }
    if (!d_tmp_mixed_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_3."); }

    double* d_y_aaaa_ovov = nullptr;
    double* d_u_aaaa_ovvo = nullptr;
    double* d_x_aabb_ovov = nullptr;
    double* d_g_aabb_oovv = nullptr;
    double* d_g_aabb_ovvo = nullptr;
    double* d_x_bbaa_ovov = nullptr;
    double* d_g_bbaa_ovvo = nullptr;
    double* d_g_bbaa_oovv = nullptr;

    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_u_aaaa_ovvo, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_x_aabb_ovov, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_g_aabb_oovv, sizeof(double) * num_aabb_oovv);
    tracked_cudaMalloc(&d_g_aabb_ovvo, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_x_bbaa_ovov, sizeof(double) * num_aabb_ovov);
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_u_aaaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_aaaa_ovvo."); }
    if (!d_x_aabb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aabb_ovov."); }
    if (!d_g_aabb_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oovv."); }
    if (!d_g_aabb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_ovvo."); }
    if (!d_x_bbaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbaa_ovov."); }

    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
    cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);

    // Helper: trim full MO ERI to active space when frozen core is present
    // After transform, d_g_full is num_basis^4. Trim into d_eri_ao (reused as temp).
    auto trim_eri = [&](double* d_src, double* d_dst) {
        if (num_frozen == 0) return d_src;  // no trim needed
        const size_t na4 = (size_t)na_active * na_active * na_active * na_active;
        if (!gpu::gpu_available()) {
            const size_t N = num_basis;
            #pragma omp parallel for collapse(2)
            for (int p = 0; p < na_active; p++)
                for (int q = 0; q < na_active; q++)
                    for (int r = 0; r < na_active; r++)
                        for (int s = 0; s < na_active; s++) {
                            size_t src = ((size_t)(num_frozen+p)*N + (num_frozen+q))*N*N
                                       + (size_t)(num_frozen+r)*N + (num_frozen+s);
                            size_t dst_idx = ((size_t)p*na_active*na_active + (size_t)q*na_active + r)*(size_t)na_active + s;
                            d_dst[dst_idx] = d_src[src];
                        }
        } else {
            int thr = 256;
            int blk = (int)((na4 + thr - 1) / thr);
            trim_eri_frozen_core_kernel<<<blk, thr>>>(d_src, d_dst, num_basis, na_active, num_frozen);
            cudaDeviceSynchronize();
        }
        return d_dst;
    };

    // The AO-ERI buffer (d_eri_ao) is dead storage between the transforms/
    // rotations: it only serves as their ping-pong scratch, and after each
    // transform its contents are garbage. The large vvvv intermediates are
    // allocated exactly inside those dead windows, so they can be carved
    // into the AO-ERI buffer instead of allocating fresh memory. This
    // reduces the peak from 2*M^4 + v^4 to 2*M^4.
    // Restriction: with a frozen core, trim_eri() writes the trimmed tensor
    // into d_eri_ao, so the buffer is live during tensorization; in that
    // case keep the plain mallocs (frozen-core cases are small anyway).
    const bool carve_vvvv_into_eri = (num_frozen == 0);

    // ---- Build and verify U = C_alpha^T S C_beta (before the AO ERIs are touched) ----
    double* d_U = nullptr;
    const double u_max_dev = build_spin_rotation_matrix(
        &d_U,
        d_coefficient_matrix_al,
        d_coefficient_matrix_be,
        d_overlap_matrix,
        num_basis);

    if (u_max_dev > kSpinRotationTolerance) {
        std::cout << "[UMP3] cascade: spin transformation matrix check failed (max deviation "
                  << u_max_dev << "); falling back to the default algorithm." << std::endl;
        tracked_cudaFree(d_U);
        if (superset_ump2) { tracked_cudaFree(d_ump2_energy); }
        cublasDestroy(cublasH);
        tracked_cudaFree(d_g_full);
        tracked_cudaFree(d_energy_4h2p_2h4p);
        tracked_cudaFree(d_energy_3h3p);
        tracked_cudaFree(d_tmp_same_1);
        tracked_cudaFree(d_tmp_same_2);
        tracked_cudaFree(d_tmp_same_3);
        tracked_cudaFree(d_tmp_mixed_1);
        tracked_cudaFree(d_tmp_mixed_2);
        tracked_cudaFree(d_tmp_mixed_3);
        tracked_cudaFree(d_y_aaaa_ovov);
        tracked_cudaFree(d_u_aaaa_ovvo);
        tracked_cudaFree(d_x_aabb_ovov);
        tracked_cudaFree(d_g_aabb_oovv);
        tracked_cudaFree(d_g_aabb_ovvo);
        tracked_cudaFree(d_x_bbaa_ovov);
        return ump3_from_aoeri_via_full_moeri_dgemm_eff(
            d_eri_ao,
            d_coefficient_matrix_al,
            d_coefficient_matrix_be,
            d_orbital_energies_al,
            d_orbital_energies_be,
            num_basis,
            num_occ_al,
            num_occ_be,
            num_frozen
        );
    }

    // (αα|αα) block: B0 (= d_eri_ao) is destructively consumed by the transform.
    {
        if (gpu::gpu_available()) {
            transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_full, d_coefficient_matrix_al, num_basis);
        } else {
            cpu_kernels::transform_eri_ao2mo_dgemm_full_cpu(d_eri_ao, d_g_full, d_coefficient_matrix_al, num_basis);
        }
        double* d_g_src = trim_eri(d_g_full, d_eri_ao);

        double* d_g_aaaa_oooo = nullptr;
        double* d_g_aaaa_vvvv = nullptr;
        double* d_x_aaaa_ovov = nullptr;
        tracked_cudaMalloc(&d_g_aaaa_oooo, sizeof(double) * num_aaaa_oooo);
        bool aaaa_vvvv_is_carved = false;
        if (carve_vvvv_into_eri) {
            // vvvv <= M^4 always holds, so the block fits in the dead
            // AO-ERI buffer. The Stage-B rotations that overwrite this
            // buffer are issued after the contraction below (same-stream
            // execution order), so the handoff is race-free.
            d_g_aaaa_vvvv = d_eri_ao;
            aaaa_vvvv_is_carved = true;
        } else {
            tracked_cudaMalloc(&d_g_aaaa_vvvv, sizeof(double) * num_aaaa_vvvv);
        }
        tracked_cudaMalloc(&d_x_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
        if (!d_g_aaaa_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_oooo."); }
        if (!d_g_aaaa_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_vvvv."); }
        if (!d_x_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aaaa_ovov."); }

        cudaMemset(d_g_aaaa_oooo, 0, sizeof(double) * num_aaaa_oooo);
        cudaMemset(d_g_aaaa_vvvv, 0, sizeof(double) * num_aaaa_vvvv);
        cudaMemset(d_x_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aaaa_oooo_cpu(d_g_aaaa_oooo, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_g_aaaa_vvvv_cpu(d_g_aaaa_vvvv, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_u_aaaa_ovvo_cpu(d_u_aaaa_ovvo, d_g_src, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_x_aaaa_ovov_cpu(d_x_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
            cpu_kernels::tensorize_y_aaaa_ovov_cpu(d_y_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
        } else {
            tensorize_g_aaaa_oooo<<<num_blocks_aaaa_oooo, threads>>>(d_g_aaaa_oooo, d_g_src, active_occ_al, num_vir_al);
            tensorize_g_aaaa_vvvv<<<num_blocks_aaaa_vvvv, threads>>>(d_g_aaaa_vvvv, d_g_src, active_occ_al, num_vir_al);
            tensorize_u_aaaa_ovvo<<<num_blocks_aaaa_ovov, threads>>>(d_u_aaaa_ovvo, d_g_src, active_occ_al, num_vir_al);
            tensorize_x_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_x_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
            tensorize_y_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_y_aaaa_ovov, d_g_src, d_eps_al, active_occ_al, num_vir_al);
        }

        if (superset_ump2) {
            // Plain (ia|jb) ovov block -> UMP2 same-spin (alpha-alpha) energy.
            // Reuses d_tmp_same_1 (overwritten by the contraction below). The
            // compact ovov layout (ovov2s) is identical to the layout expected
            // by the UMP2 energy kernel (ovov2seq) when called with the
            // active-space counts, num_frozen = 0, and the shifted epsilon
            // pointer, so the standalone-validated kernel is reused verbatim.
            if (!gpu::gpu_available()) {
                cpu_kernels::tensorize_g_aaaa_ovov_cpu(d_tmp_same_1, d_g_src, active_occ_al, num_vir_al);
                cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_ump2_energy, d_tmp_same_1, d_eps_al, active_occ_al, num_vir_al, 0);
            } else {
                tensorize_g_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_tmp_same_1, d_g_src, active_occ_al, num_vir_al);
                compute_ump2_energy_contrib_ss<<<num_blocks_aaaa_ovov, threads>>>(d_ump2_energy, d_tmp_same_1, d_eps_al, active_occ_al, num_vir_al, 0);
            }
        }

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_aaaa_ovov, d_y_aaaa_ovov,
            d_g_aaaa_oooo, d_g_aaaa_vvvv,
            d_u_aaaa_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_aaaa_ovov,
            active_occ_al, num_vir_al
        );

        tracked_cudaFree(d_g_aaaa_oooo);
        if (!aaaa_vvvv_is_carved) { tracked_cudaFree(d_g_aaaa_vvvv); }  // carved: backed by the AO-ERI buffer, freed by its owner
        tracked_cudaFree(d_x_aaaa_ovov);
    }

    // (αα|ββ) block: rotate axes 2,3 of the aaaa tensor (B1) with U.
    {
        // axis 3: B0[a0,a1,a2,s] = sum_y B1[a0,a1,a2,y] U(y,s)
        dgemm_rm(CUBLAS_OP_N, CUBLAS_OP_N,
            (int)num_basis_3, num_basis, num_basis,
            d_g_full, num_basis,
            d_U, num_basis,
            d_eri_ao, num_basis
        );
        // (N^2 x N^2) transpose: B1[old axis2, s, a0, a1] = B0[a0, a1, old axis2, s]
        dgeam_transpose_rm((int)num_basis_2, (int)num_basis_2, d_eri_ao, d_g_full);
        // axis 0: B0[k, s, a0, a1] = sum_x U(x,k) B1[x, s, a0, a1]
        dgemm_rm(CUBLAS_OP_T, CUBLAS_OP_N,
            num_basis, (int)num_basis_3, num_basis,
            d_U, num_basis,
            d_g_full, (int)num_basis_3,
            d_eri_ao, (int)num_basis_3
        );
        // (N^2 x N^2) transpose back: B1 = aabb [p_alpha, q_alpha, r_beta, s_beta]
        dgeam_transpose_rm((int)num_basis_2, (int)num_basis_2, d_eri_ao, d_g_full);
        cudaDeviceSynchronize();

        double* d_g_src = trim_eri(d_g_full, d_eri_ao);

        double* d_g_aabb_oooo = nullptr;
        double* d_g_aabb_vvvv = nullptr;
        tracked_cudaMalloc(&d_g_aabb_oooo, sizeof(double) * num_aabb_oooo);
        bool aabb_vvvv_is_carved = false;
        if (carve_vvvv_into_eri) {
            // Dead window: the Stage-C rotations that overwrite the AO-ERI
            // buffer are issued after the 4h2p contraction below consumed
            // this block (same-stream execution order).
            d_g_aabb_vvvv = d_eri_ao;
            aabb_vvvv_is_carved = true;
        } else {
            tracked_cudaMalloc(&d_g_aabb_vvvv, sizeof(double) * num_aabb_vvvv);
        }
        if (!d_g_aabb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oooo."); }
        if (!d_g_aabb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_vvvv."); }

        cudaMemset(d_g_aabb_oooo, 0, sizeof(double) * num_aabb_oooo);
        cudaMemset(d_g_aabb_vvvv, 0, sizeof(double) * num_aabb_vvvv);
        cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
        cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aabb_oooo_cpu(d_g_aabb_oooo, d_g_src, active_occ_al, active_occ_be, na_active);
            cpu_kernels::tensorize_g_aabb_vvvv_cpu(d_g_aabb_vvvv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            cpu_kernels::tensorize_x_aabb_ovov_cpu(d_x_aabb_ovov, d_g_src, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
            cpu_kernels::tensorize_g_bbaa_oovv_cpu(d_g_aabb_oovv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            cpu_kernels::tensorize_g_bbaa_ovvo_cpu(d_g_aabb_ovvo, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
        } else {
            tensorize_g_aabb_oooo<<<num_blocks_aabb_oooo, threads>>>(d_g_aabb_oooo, d_g_src, active_occ_al, active_occ_be, na_active);
            tensorize_g_aabb_vvvv<<<num_blocks_aabb_vvvv, threads>>>(d_g_aabb_vvvv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_aabb_ovov, d_g_src, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
            tensorize_g_bbaa_oovv<<<num_blocks_aabb_oovv, threads>>>(d_g_aabb_oovv, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
            tensorize_g_bbaa_ovvo<<<num_blocks_aabb_ovov, threads>>>(d_g_aabb_ovvo, d_g_src, active_occ_al, active_occ_be, num_vir_al, num_vir_be, na_active);
        }

        // Pack the (ββ|αα) compact blocks from the aabb tensor via pair
        // exchange (full_bbaa[k,c,b,j] = full_aabb[b,j,k,c]).
        tracked_cudaMalloc(&d_g_bbaa_ovvo, sizeof(double) * num_aabb_ovov);
        tracked_cudaMalloc(&d_g_bbaa_oovv, sizeof(double) * num_bbaa_oovv);
        if (!d_g_bbaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_ovvo."); }
        if (!d_g_bbaa_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_oovv."); }

        cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_ovvo, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_oovv, 0, sizeof(double) * num_bbaa_oovv);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_x_aabb_ovov_swapped_cpu(d_x_bbaa_ovov, d_g_src, d_eps_be, d_eps_al, active_occ_be, num_vir_be, active_occ_al, num_vir_al, na_active);
            cpu_kernels::tensorize_g_bbaa_ovvo_swapped_cpu(d_g_bbaa_ovvo, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
            cpu_kernels::tensorize_g_bbaa_oovv_swapped_cpu(d_g_bbaa_oovv, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
        } else {
            tensorize_x_aabb_ovov_swapped<<<num_blocks_aabb_ovov, threads>>>(d_x_bbaa_ovov, d_g_src, d_eps_be, d_eps_al, active_occ_be, num_vir_be, active_occ_al, num_vir_al, na_active);
            tensorize_g_bbaa_ovvo_swapped<<<num_blocks_aabb_ovov, threads>>>(d_g_bbaa_ovvo, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
            tensorize_g_bbaa_oovv_swapped<<<num_blocks_bbaa_oovv, threads>>>(d_g_bbaa_oovv, d_g_src, active_occ_be, active_occ_al, num_vir_be, num_vir_al, na_active);
        }

        if (superset_ump2) {
            // Plain (i_al a_al | j_be b_be) ovov block -> UMP2 opposite-spin
            // energy. Reuses d_tmp_mixed_1 (overwritten by the contraction
            // below). Layout note: ovov2s_aabb is identical to the layout
            // expected by compute_ump2_energy_contrib_os (ovov2seq_aabb)
            // with the active-space counts, num_frozen = 0, and the shifted
            // epsilon pointers.
            if (!gpu::gpu_available()) {
                cpu_kernels::tensorize_g_aabb_ovov_cpu(d_tmp_mixed_1, d_g_src, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
                cpu_kernels::compute_ump2_energy_contrib_os_cpu(d_ump2_energy, d_tmp_mixed_1, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, 0);
            } else {
                tensorize_g_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_tmp_mixed_1, d_g_src, active_occ_al, num_vir_al, active_occ_be, num_vir_be, na_active);
                compute_ump2_energy_contrib_os<<<num_blocks_aabb_ovov, threads>>>(d_ump2_energy, d_tmp_mixed_1, d_eps_al, d_eps_be, active_occ_al, num_vir_al, active_occ_be, num_vir_be, 0);
            }
        }

        contract_mixed_4h2p_2h4p_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p,
            d_x_aabb_ovov,
            d_g_aabb_oooo,
            d_g_aabb_vvvv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            d_tmp_mixed_3,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        tracked_cudaFree(d_g_aabb_oooo);
        if (!aabb_vvvv_is_carved) { tracked_cudaFree(d_g_aabb_vvvv); }  // carved: backed by the AO-ERI buffer, freed by its owner

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_aaaa_ovov,
            d_x_aabb_ovov,
            d_g_bbaa_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_aaaa_ovov,
            active_occ_al, num_vir_al,
            active_occ_be, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_g_bbaa_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_g_aabb_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            active_occ_be, active_occ_al, num_vir_be, num_vir_al
        );

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_u_aaaa_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            active_occ_be, active_occ_al, num_vir_be, num_vir_al
        );

        tracked_cudaFree(d_g_bbaa_ovvo);
        tracked_cudaFree(d_g_bbaa_oovv);
        tracked_cudaFree(d_y_aaaa_ovov);
        tracked_cudaFree(d_u_aaaa_ovvo);
        tracked_cudaFree(d_g_aabb_oovv);
        d_y_aaaa_ovov = nullptr;
        d_u_aaaa_ovvo = nullptr;
        d_g_aabb_oovv = nullptr;
    }

    // (ββ|ββ) block: rotate axes 0,1 of the aabb tensor (B1) with U.
    {
        // axis 1 (batched over axis 0): B0[a0, c, b, j] = sum_y U(y,c) B1[a0, y, b, j]
        dgemm_batched_rm(CUBLAS_OP_T, CUBLAS_OP_N,
            num_basis, (int)num_basis_2, num_basis,
            d_U, num_basis, 0,
            d_g_full, (int)num_basis_2, (long long int)num_basis_3,
            d_eri_ao, (int)num_basis_2, (long long int)num_basis_3,
            num_basis
        );
        // axis 0: B1[k, c, b, j] = sum_x U(x,k) B0[x, c, b, j]  (B1 = bbbb)
        dgemm_rm(CUBLAS_OP_T, CUBLAS_OP_N,
            num_basis, (int)num_basis_3, num_basis,
            d_U, num_basis,
            d_eri_ao, (int)num_basis_3,
            d_g_full, (int)num_basis_3
        );
        cudaDeviceSynchronize();

        double* d_g_src = trim_eri(d_g_full, d_eri_ao);

        double* d_g_bbbb_oooo = nullptr;
        double* d_g_bbbb_vvvv = nullptr;
        double* d_u_bbbb_ovvo = nullptr;
        double* d_x_bbbb_ovov = nullptr;
        double* d_y_bbbb_ovov = nullptr;
        tracked_cudaMalloc(&d_g_bbbb_oooo, sizeof(double) * num_bbbb_oooo);
        bool bbbb_vvvv_is_carved = false;
        if (carve_vvvv_into_eri) {
            // Final stage: the AO-ERI buffer is dead for the rest of the
            // function, so the carved block can stay until the end.
            d_g_bbbb_vvvv = d_eri_ao;
            bbbb_vvvv_is_carved = true;
        } else {
            tracked_cudaMalloc(&d_g_bbbb_vvvv, sizeof(double) * num_bbbb_vvvv);
        }
        tracked_cudaMalloc(&d_u_bbbb_ovvo, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_x_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_y_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        if (!d_g_bbbb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_oooo."); }
        if (!d_g_bbbb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_vvvv."); }
        if (!d_u_bbbb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_bbbb_ovvo."); }
        if (!d_x_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbbb_ovov."); }
        if (!d_y_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_bbbb_ovov."); }

        cudaMemset(d_g_bbbb_oooo, 0, sizeof(double) * num_bbbb_oooo);
        cudaMemset(d_g_bbbb_vvvv, 0, sizeof(double) * num_bbbb_vvvv);
        cudaMemset(d_u_bbbb_ovvo, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_x_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_y_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);

        if (!gpu::gpu_available()) {
            cpu_kernels::tensorize_g_aaaa_oooo_cpu(d_g_bbbb_oooo, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_g_aaaa_vvvv_cpu(d_g_bbbb_vvvv, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_u_aaaa_ovvo_cpu(d_u_bbbb_ovvo, d_g_src, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_x_aaaa_ovov_cpu(d_x_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
            cpu_kernels::tensorize_y_aaaa_ovov_cpu(d_y_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
        } else {
            tensorize_g_aaaa_oooo<<<num_blocks_bbbb_oooo, threads>>>(d_g_bbbb_oooo, d_g_src, active_occ_be, num_vir_be);
            tensorize_g_aaaa_vvvv<<<num_blocks_bbbb_vvvv, threads>>>(d_g_bbbb_vvvv, d_g_src, active_occ_be, num_vir_be);
            tensorize_u_aaaa_ovvo<<<num_blocks_bbbb_ovov, threads>>>(d_u_bbbb_ovvo, d_g_src, active_occ_be, num_vir_be);
            tensorize_x_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_x_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
            tensorize_y_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_y_bbbb_ovov, d_g_src, d_eps_be, active_occ_be, num_vir_be);
        }

        if (superset_ump2) {
            // Plain (ia|jb) ovov block -> UMP2 same-spin (beta-beta) energy.
            // Reuses d_tmp_same_1 (overwritten by the contraction below);
            // same layout argument as the Stage-A superset accumulation above.
            if (!gpu::gpu_available()) {
                cpu_kernels::tensorize_g_aaaa_ovov_cpu(d_tmp_same_1, d_g_src, active_occ_be, num_vir_be);
                cpu_kernels::compute_ump2_energy_contrib_ss_cpu(d_ump2_energy, d_tmp_same_1, d_eps_be, active_occ_be, num_vir_be, 0);
            } else {
                tensorize_g_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_tmp_same_1, d_g_src, active_occ_be, num_vir_be);
                compute_ump2_energy_contrib_ss<<<num_blocks_bbbb_ovov, threads>>>(d_ump2_energy, d_tmp_same_1, d_eps_be, active_occ_be, num_vir_be, 0);
            }
        }

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_u_bbbb_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            active_occ_al, active_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_bbbb_ovov,
            d_x_bbaa_ovov,
            d_g_aabb_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_bbbb_ovov,
            active_occ_be, num_vir_be,
            active_occ_al, num_vir_al
        );

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_bbbb_ovov, d_y_bbbb_ovov,
            d_g_bbbb_oooo, d_g_bbbb_vvvv,
            d_u_bbbb_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_bbbb_ovov,
            active_occ_be, num_vir_be
        );

        tracked_cudaFree(d_g_bbbb_oooo);
        if (!bbbb_vvvv_is_carved) { tracked_cudaFree(d_g_bbbb_vvvv); }  // carved: backed by the AO-ERI buffer, freed by its owner
        tracked_cudaFree(d_u_bbbb_ovvo);
        tracked_cudaFree(d_x_bbbb_ovov);
        tracked_cudaFree(d_y_bbbb_ovov);
    }

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);

    if (superset_ump2) {
        double h_ump2_energy = 0.0;
        cudaMemcpy(&h_ump2_energy, d_ump2_energy, sizeof(double), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_ump2_energy);
        if (out_ump2_energy) { *out_ump2_energy = h_ump2_energy; }
        if (out_ump2_from_superset) { *out_ump2_from_superset = true; }
    }

    if (gpu::gpu_available()) {
        cublasDestroy(cublasH);
    }

    tracked_cudaFree(d_g_full);
    tracked_cudaFree(d_energy_4h2p_2h4p);
    tracked_cudaFree(d_energy_3h3p);
    tracked_cudaFree(d_tmp_same_1);
    tracked_cudaFree(d_tmp_same_2);
    tracked_cudaFree(d_tmp_same_3);
    tracked_cudaFree(d_tmp_mixed_1);
    tracked_cudaFree(d_tmp_mixed_2);
    tracked_cudaFree(d_tmp_mixed_3);
    tracked_cudaFree(d_x_aabb_ovov);
    tracked_cudaFree(d_g_aabb_ovvo);
    tracked_cudaFree(d_x_bbaa_ovov);
    tracked_cudaFree(d_U);

    if (out_ao_consumed) { *out_ao_consumed = true; }

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}



real_t ERI_Stored_UHF::compute_mp3_energy()
{
    PROFILE_FUNCTION();

    if (!is_ao_eri_valid()) {
        THROW_EXCEPTION("AO ERIs are no longer available: the UMP3 cascading integral transformation destructively reused the AO ERI buffer as scratch.");
    }

    const int num_basis = uhf_.get_num_basis();
    const int num_occ_al = uhf_.get_num_alpha_spins();
    const int num_occ_be = uhf_.get_num_beta_spins();
    const int num_frozen = uhf_.get_num_frozen_core();

    DeviceHostMatrix<real_t>& coefficient_matrix_al = uhf_.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& coefficient_matrix_be = uhf_.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& orbital_energies_al = uhf_.get_orbital_energies_a();
    DeviceHostMemory<real_t>& orbital_energies_be = uhf_.get_orbital_energies_b();

    real_t E_UMP2 = 0.0;
    real_t E_UMP3 = 0.0;

    // Cascading integral transformation: 2 x M^4 peak buffers; destructively
    // reuses the AO ERI buffer as scratch (invalidated below).
    //
    // Superset evaluation (UMP3 subsuming UMP2): the cascade UMP3
    // intermediates (the plain ovov blocks tensorized during the stages) are
    // a strict superset of the canonical UMP2 inputs, so when the spin
    // rotation matrix passes the consistency check the UMP2 energy is
    // obtained inside the cascade and the standalone UMP2 phase -- a full
    // extra pass of AO->MO transforms over the AO ERIs -- is skipped
    // entirely.
    double* d_U_probe = nullptr;
    const bool fuse_ump2 =
        build_spin_rotation_matrix(
            &d_U_probe,
            coefficient_matrix_al.device_ptr(),
            coefficient_matrix_be.device_ptr(),
            uhf_.get_overlap_matrix().device_ptr(),
            num_basis) <= kSpinRotationTolerance;
    tracked_cudaFree(d_U_probe);

    real_t superset_ump2 = 0.0;
    bool ump2_from_superset = false;
    bool ao_consumed = false;
    E_UMP3 = ump3_from_aoeri_via_cascade(
        eri_matrix_.device_ptr(),
        coefficient_matrix_al.device_ptr(),
        coefficient_matrix_be.device_ptr(),
        uhf_.get_overlap_matrix().device_ptr(),
        orbital_energies_al.device_ptr(),
        orbital_energies_be.device_ptr(),
        num_basis,
        num_occ_al,
        num_occ_be,
        num_frozen,
        &ao_consumed,
        fuse_ump2 ? &superset_ump2 : nullptr,
        fuse_ump2 ? &ump2_from_superset : nullptr
    );
    if (ao_consumed) {
        invalidate_ao_eri();
    }
    if (ump2_from_superset) {
        E_UMP2 = superset_ump2;
        std::cout << "UMP2 correlation energy (superset evaluation within the UMP3 cascade): "
                  << std::setprecision(12) << E_UMP2 << std::endl;
    } else {
        // The superset evaluation did not run: either the pre-check above
        // failed (the cascade then fell back to the reference algorithm
        // internally, which leaves the AO ERIs intact) or the internal
        // check disagreed with the pre-check (defensive; both run the
        // identical deterministic computation).
        E_UMP2 = ump2_from_aoeri_via_required_moeri(
            eri_matrix_.device_ptr(),
            coefficient_matrix_al.device_ptr(),
            coefficient_matrix_be.device_ptr(),
            orbital_energies_al.device_ptr(),
            orbital_energies_be.device_ptr(),
            num_basis,
            num_occ_al,
            num_occ_be,
            num_frozen
        );
    }

    std::cout << "UMP3 energy: " << E_UMP3 << " [hartree]" << std::endl;

    return E_UMP2 + E_UMP3;
}



























}   // namespace gansu
