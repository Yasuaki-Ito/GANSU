/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ccsd_lambda_gpu.cu
 * @brief CCSD Lambda solver — GPU implementation.
 *
 * Mirrors solve_ccsd_lambda_cpu term-for-term with CUDA kernels parallelized
 * over output indices. cuBLAS DGEMM is used for the m3 first term (l2·vvvv).
 *
 * All tensors are stored row-major matching the CPU convention:
 *   t1, l1, v4         [nocc, nvir]
 *   t2, l2, m3, l2new  [nocc, nocc, nvir, nvir]
 *   v1[b,a]            [nvir, nvir]
 *   v2[i,j]            [nocc, nocc]
 *   v5[b,j], w3[c,k]   [nvir, nocc]
 *   ovov[i,a,j,b]      [nocc, nvir, nocc, nvir]
 *   ovoo[i,a,j,k]      [nocc, nvir, nocc, nocc]
 *   ovvv[i,a,b,c]      [nocc, nvir, nvir, nvir]
 *   oovv[i,j,a,b]      [nocc, nocc, nvir, nvir]
 *   ovvo[i,a,b,j]      [nocc, nvir, nvir, nocc]
 *   oooo[i,j,k,l]      [nocc, nocc, nocc, nocc]
 *   vvvv[a,b,c,d]      [nvir, nvir, nvir, nvir]
 *   woooo[i,k,j,l]     [nocc, nocc, nocc, nocc]
 *   wovvo[j,b,c,k]     [nocc, nvir, nvir, nocc]
 *   ...
 */

#include "ccsd_lambda.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

namespace gansu {

// ============================================================================
//  Index helper macros (device-side)
// ============================================================================
//   __device__ inline functions are too verbose; macros let us match the
//   CPU code very closely. NO, NV are passed as kernel args.
// ============================================================================

#define IDX2(i, a)            ((i) * NV + (a))
#define IDX4(i,j,a,b)         ((((i) * NO + (j)) * NV + (a)) * NV + (b))
#define IDX_OVOV(i,a,j,b)     ((((i) * NV + (a)) * NO + (j)) * NV + (b))
#define IDX_OVOO(i,a,j,k)     ((((i) * NV + (a)) * NO + (j)) * NO + (k))
#define IDX_OVVV(i,a,b,c)     ((((i) * NV + (a)) * NV + (b)) * NV + (c))
#define IDX_OOVV(i,j,a,b)     ((((i) * NO + (j)) * NV + (a)) * NV + (b))
#define IDX_OVVO(i,a,b,j)     ((((i) * NV + (a)) * NV + (b)) * NO + (j))
#define IDX_OOOO(i,j,k,l)     ((((i) * NO + (j)) * NO + (k)) * NO + (l))
#define IDX_VVVV(a,b,c,d)     ((((a) * NV + (b)) * NV + (c)) * NV + (d))
#define IDX_OOOV(i,j,k,a)     ((((i) * NO + (j)) * NO + (k)) * NV + (a))
#define IDX_VVOV(a,b,i,c)     ((((a) * NV + (b)) * NO + (i)) * NV + (c))
#define IDX_OVVNO(j,b,c,k)    ((((j) * NV + (b)) * NV + (c)) * NO + (k))
// woooo[i,k,j,l] uses standard 4-index OOOO layout
#define IDX_WOOOO(i,k,j,l)    ((((i) * NO + (k)) * NO + (j)) * NO + (l))

// Forward decls for kernels referenced before their definitions
__global__ void combine_wovvo_k(const real_t* a, const real_t* b, real_t* out, size_t n);

// ============================================================================
//  Sub-block extraction kernels (full eri_mo[na^4] → blocks)
// ============================================================================

__global__ void extract_ovov_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV*NO*NV) return;
    int i = idx / (NV*NO*NV);
    int r = idx % (NV*NO*NV);
    int a = r / (NO*NV);
    r %= (NO*NV);
    int j = r / NV;
    int b = r % NV;
    out[idx] = eri[(((size_t)i * NA + (NO+a)) * NA + j) * NA + (NO+b)];
}

__global__ void extract_oovv_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO*NV*NV) return;
    int i = idx / (NO*NV*NV);
    int r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    out[idx] = eri[(((size_t)i * NA + j) * NA + (NO+a)) * NA + (NO+b)];
}

__global__ void extract_ovvo_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV*NV*NO) return;
    int i = idx / (NV*NV*NO);
    int r = idx % (NV*NV*NO);
    int a = r / (NV*NO);
    r %= (NV*NO);
    int b = r / NO;
    int j = r % NO;
    out[idx] = eri[(((size_t)i * NA + (NO+a)) * NA + (NO+b)) * NA + j];
}

__global__ void extract_oooo_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO*NO*NO) return;
    int i = idx / (NO*NO*NO);
    int r = idx % (NO*NO*NO);
    int j = r / (NO*NO);
    r %= (NO*NO);
    int k = r / NO;
    int l = r % NO;
    out[idx] = eri[(((size_t)i * NA + j) * NA + k) * NA + l];
}

__global__ void extract_ovoo_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV*NO*NO) return;
    int i = idx / (NV*NO*NO);
    int r = idx % (NV*NO*NO);
    int a = r / (NO*NO);
    r %= (NO*NO);
    int j = r / NO;
    int k = r % NO;
    out[idx] = eri[(((size_t)i * NA + (NO+a)) * NA + j) * NA + k];
}

__global__ void extract_ovvv_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV*NV*NV) return;
    int i = idx / (NV*NV*NV);
    int r = idx % (NV*NV*NV);
    int a = r / (NV*NV);
    r %= (NV*NV);
    int b = r / NV;
    int c = r % NV;
    out[idx] = eri[(((size_t)i * NA + (NO+a)) * NA + (NO+b)) * NA + (NO+c)];
}

__global__ void extract_vvvv_k(const real_t* __restrict__ eri,
                               real_t* __restrict__ out, int NA, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NV*NV*NV*NV;
    if (idx >= total) return;
    int a = idx / (NV*NV*NV);
    size_t r = idx % (NV*NV*NV);
    int b = r / (NV*NV);
    r %= (NV*NV);
    int c = r / NV;
    int d = r % NV;
    out[idx] = eri[(((size_t)(NO+a) * NA + (NO+b)) * NA + (NO+c)) * NA + (NO+d)];
}

// ============================================================================
//  tau, theta
// ============================================================================

__global__ void make_tau_k(const real_t* __restrict__ t1, const real_t* __restrict__ t2,
                           real_t* __restrict__ tau, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    int r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    tau[idx] = t2[idx] + t1[IDX2(i,a)] * t1[IDX2(j,b)];
}

__global__ void make_theta_k(const real_t* __restrict__ t2,
                             real_t* __restrict__ theta, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    int r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    theta[idx] = 2.0 * t2[idx] - t2[IDX4(i,j,b,a)];
}

// ============================================================================
//  Intermediates: v1, v2, v4, v5, w3 (small — NV² or NO² output)
// ============================================================================

// v1[b,a] = (a==b ? eps[NO+a] : 0) - sum_{j,k,c} ovov1[j,a,k,c]*tau[j,k,b,c]
//          + sum_{j,c} ovvv1[j,c,b,a]*t1[j,c]
// where ovov1[j,a,k,c] = 2*ovov[j,a,k,c] - ovov[j,c,k,a]
//       ovvv1[j,c,b,a] = 2*ovvv[j,c,b,a] - ovvv[j,a,b,c]
__global__ void compute_v1_k(const real_t* __restrict__ ovov, const real_t* __restrict__ ovvv,
                             const real_t* __restrict__ t1,   const real_t* __restrict__ tau,
                             const real_t* __restrict__ eps,
                             real_t* __restrict__ v1, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NV*NV) return;
    int b = idx / NV;
    int a = idx % NV;
    real_t v = (a == b) ? eps[NO + a] : 0.0;
    for (int j = 0; j < NO; j++)
      for (int k = 0; k < NO; k++)
        for (int c = 0; c < NV; c++)
          v -= (2.0*ovov[IDX_OVOV(j,a,k,c)] - ovov[IDX_OVOV(j,c,k,a)])
                * tau[IDX4(j,k,b,c)];
    for (int j = 0; j < NO; j++)
      for (int c = 0; c < NV; c++)
        v += (2.0*ovvv[IDX_OVVV(j,c,b,a)] - ovvv[IDX_OVVV(j,a,b,c)])
              * t1[IDX2(j,c)];
    v1[idx] = v;
}

// v2[i,j] = (i==j ? eps[i] : 0) + sum_{b,k,c} ovov1[i,b,k,c]*tau[j,k,b,c]
//                                + sum_{k,b} ovoo1[k,b,i,j]*t1[k,b]
// ovoo1[k,b,i,j] = 2*ovoo[k,b,i,j] - ovoo[i,b,k,j]
__global__ void compute_v2_k(const real_t* __restrict__ ovov, const real_t* __restrict__ ovoo,
                             const real_t* __restrict__ t1,   const real_t* __restrict__ tau,
                             const real_t* __restrict__ eps,
                             real_t* __restrict__ v2, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO) return;
    int i = idx / NO;
    int j = idx % NO;
    real_t v = (i == j) ? eps[i] : 0.0;
    for (int b = 0; b < NV; b++)
      for (int k = 0; k < NO; k++)
        for (int c = 0; c < NV; c++)
          v += (2.0*ovov[IDX_OVOV(i,b,k,c)] - ovov[IDX_OVOV(i,c,k,b)])
                * tau[IDX4(j,k,b,c)];
    for (int k = 0; k < NO; k++)
      for (int b = 0; b < NV; b++)
        v += (2.0*ovoo[IDX_OVOO(k,b,i,j)] - ovoo[IDX_OVOO(i,b,k,j)])
              * t1[IDX2(k,b)];
    v2[idx] = v;
}

// v4[j,b] = sum_{k,c} ovov1[j,b,k,c]*t1[k,c]
__global__ void compute_v4_k(const real_t* __restrict__ ovov, const real_t* __restrict__ t1,
                             real_t* __restrict__ v4, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV) return;
    int j = idx / NV;
    int b = idx % NV;
    real_t v = 0.0;
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        v += (2.0*ovov[IDX_OVOV(j,b,k,c)] - ovov[IDX_OVOV(j,c,k,b)]) * t1[IDX2(k,c)];
    v4[idx] = v;
}

// v5[b,j] = sum_{k,c} v4[k,c]*t1[k,b]*t1[j,c]
//        - sum_{l,c,k} ovoo1[l,c,k,j]*t2[k,l,b,c]
//        + sum_{k,c,d} ovvv1[k,d,b,c]*t2[j,k,c,d]
__global__ void compute_v5_k(const real_t* __restrict__ ovoo, const real_t* __restrict__ ovvv,
                             const real_t* __restrict__ t1, const real_t* __restrict__ t2,
                             const real_t* __restrict__ v4,
                             real_t* __restrict__ v5, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NV*NO) return;
    int b = idx / NO;
    int j = idx % NO;
    real_t v = 0.0;
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        v += v4[IDX2(k,c)] * t1[IDX2(k,b)] * t1[IDX2(j,c)];
    for (int l = 0; l < NO; l++)
      for (int c = 0; c < NV; c++)
        for (int k = 0; k < NO; k++)
          v -= (2.0*ovoo[IDX_OVOO(l,c,k,j)] - ovoo[IDX_OVOO(k,c,l,j)]) * t2[IDX4(k,l,b,c)];
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        for (int d = 0; d < NV; d++)
          v += (2.0*ovvv[IDX_OVVV(k,d,b,c)] - ovvv[IDX_OVVV(k,c,b,d)]) * t2[IDX4(j,k,c,d)];
    v5[idx] = v;
}

// ============================================================================
//  v4OVvo, v4oVVo, then wOVvo, woVVo, wovvo (= 2*wOVvo + woVVo)
// ============================================================================

__global__ void compute_v4OVvo_k(const real_t* ovov, const real_t* t2, const real_t* ovvo,
                                 real_t* v4o, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NV*NV*NO;
    if (idx >= total) return;
    int j = idx / (NV*NV*NO);
    size_t r = idx % (NV*NV*NO);
    int b = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;
    real_t v = ovvo[IDX_OVVO(j,b,c,k)];
    for (int l = 0; l < NO; l++)
      for (int d = 0; d < NV; d++) {
        real_t ov1 = 2.0*ovov[IDX_OVOV(l,d,j,b)] - ovov[IDX_OVOV(l,b,j,d)];
        v += ov1 * t2[IDX4(k,l,c,d)];
        v -= ovov[IDX_OVOV(l,d,j,b)] * t2[IDX4(k,l,d,c)];
      }
    v4o[idx] = v;
}

__global__ void compute_v4oVVo_k(const real_t* ovov, const real_t* t2, const real_t* oovv,
                                 real_t* v4v, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NV*NV*NO;
    if (idx >= total) return;
    int j = idx / (NV*NV*NO);
    size_t r = idx % (NV*NV*NO);
    int b = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;
    real_t v = 0.0;
    for (int l = 0; l < NO; l++)
      for (int d = 0; d < NV; d++)
        v += ovov[IDX_OVOV(j,d,l,b)] * t2[IDX4(k,l,d,c)];
    v -= oovv[IDX_OOVV(j,k,c,b)];
    v4v[idx] = v;
}

// wOVvo and woVVo: starting from v4OVvo / v4oVVo, add corrections
__global__ void compute_wOVvo_woVVo_k(const real_t* v4o, const real_t* v4v,
                                      const real_t* ovov, const real_t* ovoo,
                                      const real_t* ovvv, const real_t* t1,
                                      real_t* wOVvo, real_t* woVVo, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NV*NV*NO;
    if (idx >= total) return;
    int j = idx / (NV*NV*NO);
    size_t r = idx % (NV*NV*NO);
    int b = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;
    real_t vO = v4o[idx];
    real_t vV = v4v[idx];
    for (int l = 0; l < NO; l++)
      for (int d = 0; d < NV; d++) {
        vO -= ovov[IDX_OVOV(j,b,l,d)] * t1[IDX2(k,d)] * t1[IDX2(l,c)];
        vV += ovov[IDX_OVOV(j,d,l,b)] * t1[IDX2(k,d)] * t1[IDX2(l,c)];
      }
    for (int l = 0; l < NO; l++) {
      vO -= ovoo[IDX_OVOO(j,b,l,k)] * t1[IDX2(l,c)];
      vV += ovoo[IDX_OVOO(l,b,j,k)] * t1[IDX2(l,c)];
    }
    for (int d = 0; d < NV; d++) {
      vO += ovvv[IDX_OVVV(j,b,c,d)] * t1[IDX2(k,d)];
      vV -= ovvv[IDX_OVVV(j,d,c,b)] * t1[IDX2(k,d)];
    }
    wOVvo[idx] = vO;
    woVVo[idx] = vV;
}

// ============================================================================
//  woooo[i,k,j,l]
// ============================================================================
__global__ void compute_woooo_k(const real_t* oooo, const real_t* ovoo,
                                const real_t* ovov, const real_t* t1,
                                const real_t* tau,
                                real_t* woooo, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO*NO*NO) return;
    int i = idx / (NO*NO*NO);
    int r = idx % (NO*NO*NO);
    int k = r / (NO*NO);
    r %= (NO*NO);
    int j = r / NO;
    int l = r % NO;
    real_t v = oooo[IDX_OOOO(i,k,j,l)];
    for (int c = 0; c < NV; c++)
      v += ovoo[IDX_OVOO(i,c,j,l)] * t1[IDX2(k,c)];
    for (int c = 0; c < NV; c++)
      v += ovoo[IDX_OVOO(j,c,i,k)] * t1[IDX2(l,c)];
    for (int c = 0; c < NV; c++)
      for (int d = 0; d < NV; d++)
        v += ovov[IDX_OVOV(i,c,j,d)] * tau[IDX4(k,l,c,d)];
    woooo[idx] = v;
}

// ============================================================================
//  woovo[i,j,c,k] — 8 contributions (long expression mirrored from CPU)
// ============================================================================
__global__ void compute_woovo_k(const real_t* v4o, const real_t* v4v,
                                const real_t* ovoo, const real_t* ovvv,
                                const real_t* t1, const real_t* t2, const real_t* theta,
                                const real_t* tau,
                                real_t* woovo, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NO*NO*NV*NO;
    if (idx >= total) return;
    int i = idx / (NO*NV*NO);
    int r = idx % (NO*NV*NO);
    int j = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;

    real_t v = 0.0;
    // step1: sum_b (2*v4o + v4v)[i,b,c,k]*t1[j,b]
    for (int b = 0; b < NV; b++) {
        real_t v4 = 2.0 * v4o[IDX_OVVNO(i,b,c,k)] + v4v[IDX_OVVNO(i,b,c,k)];
        v += v4 * t1[IDX2(j,b)];
    }
    // step2: -= the same with k↔j: sum_b (2*v4o + v4v)[i,b,c,j]*t1[k,b]
    for (int b = 0; b < NV; b++) {
        real_t v4 = 2.0 * v4o[IDX_OVVNO(i,b,c,j)] + v4v[IDX_OVVNO(i,b,c,j)];
        v -= v4 * t1[IDX2(k,b)];
    }
    // step3: WOOVO(i,k,c,j) += sum_b (v4o-v4v)[i,b,c,k]*t1[j,b]
    //   Reindex: at (i,j,c,k), this contributes when (j↔k swap) for index pair (k_dst, j_dst).
    //   I.e., for current (i,j,c,k), the step3 contribution is to WOOVO(i, k_dst, c, j_dst)
    //   where output is woovo[i, j', c, k']. Substituting j'=k(orig)→k_dst=j(here), k'=j(orig)→j_dst=k(here):
    //   contribution to WOOVO(i, j', c, k') = sum_b (v4o-v4v)[i,b,c,j']*t1[k',b]
    //   i.e., with (j', k') ↔ our (j, k): += sum_b (v4o-v4v)[i,b,c,j]*t1[k,b]
    for (int b = 0; b < NV; b++) {
        real_t v4d = v4o[IDX_OVVNO(i,b,c,j)] - v4v[IDX_OVVNO(i,b,c,j)];
        v += v4d * t1[IDX2(k,b)];
    }
    // step4: += ovoo1[k,c,j,i]
    v += 2.0 * ovoo[IDX_OVOO(k,c,j,i)] - ovoo[IDX_OVOO(j,c,k,i)];
    // step5: WOOVO(i,k,b,j) += sum_{l,c'} ovoo1[l,c',i,k]*theta[j,l,b,c']
    //   At (i,j,c,k), reindex (k↔k, b→c, j→j_in_orig). Original step is on WOOVO(i,k,b,j); to map:
    //   WOOVO(i, k_dst, b_dst, j_dst) where here (k_dst, b_dst, j_dst) = (j_curr, c_curr, k_curr) — wait.
    //   Let me re-derive. The PySCF einsum is:
    //     woovo[i,k,b,j] += sum_{l,c'} ovoo1[l,c',i,k]*theta[j,l,b,c']
    //   At current dest (i,j,c,k), what does this contribute? We need (i,k_dst,b_dst,j_dst) = (i,j,c,k).
    //   So k_dst=j, b_dst=c, j_dst=k. Then the contribution is sum_{l,c'} ovoo1[l,c',i,k_dst=j]*theta[j_dst=k,l,b_dst=c,c']
    for (int l = 0; l < NO; l++)
      for (int cc = 0; cc < NV; cc++) {
        real_t ovoo1 = 2.0*ovoo[IDX_OVOO(l,cc,i,j)] - ovoo[IDX_OVOO(i,cc,l,j)];
        v += ovoo1 * theta[IDX4(k,l,c,cc)];
      }
    // step6: -= einsum('lcik,jlbc->ijbk', ovoo1, t2) at (i,j,c,k):
    //   sum_{l,c'} ovoo1[l,c',i,k]*t2[j,l,c,c']
    for (int l = 0; l < NO; l++)
      for (int cc = 0; cc < NV; cc++) {
        real_t ovoo1 = 2.0*ovoo[IDX_OVOO(l,cc,i,k)] - ovoo[IDX_OVOO(i,cc,l,k)];
        v -= ovoo1 * t2[IDX4(j,l,c,cc)];
      }
    // step7: -= einsum('iclk,ljbc->ijbk', ovoo1, t2):
    //   sum_{l,c'} ovoo1[i,c',l,k]*t2[l,j,c,c']
    for (int l = 0; l < NO; l++)
      for (int cc = 0; cc < NV; cc++) {
        real_t ovoo1 = 2.0*ovoo[IDX_OVOO(i,cc,l,k)] - ovoo[IDX_OVOO(l,cc,i,k)];
        v -= ovoo1 * t2[IDX4(l,j,c,cc)];
      }
    // step8: += einsum('idcb,jkdb->ijck', ovvv1, tau):
    //   ovvv1[i,d,c,b] = 2*ovvv[i,d,c,b] - ovvv[i,b,c,d]  (PySCF uses reassigned ovvv1 here)
    for (int d = 0; d < NV; d++)
      for (int b = 0; b < NV; b++) {
        real_t ovvv1 = 2.0 * ovvv[IDX_OVVV(i,d,c,b)] - ovvv[IDX_OVVV(i,b,c,d)];
        v += ovvv1 * tau[IDX4(j,k,d,b)];
      }

    woovo[idx] = v;
}

// ============================================================================
//  wvvvo[b,a,c,k] — analogous structure, 7 steps
// ============================================================================
__global__ void compute_wvvvo_k(const real_t* v4o, const real_t* v4v,
                                const real_t* ovoo, const real_t* ovvv,
                                const real_t* t1, const real_t* t2, const real_t* theta,
                                const real_t* tau,
                                real_t* wvvvo, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NV*NV*NV*NO;
    if (idx >= total) return;
    int b = idx / (NV*NV*NO);
    size_t r = idx % (NV*NV*NO);
    int a = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;

    real_t v = 0.0;
    // step1: wvvvo[b,a,c,k] += sum_j (2*v4o + v4v)[j,a,c,k]*t1[j,b]
    for (int j = 0; j < NO; j++) {
        real_t v4 = 2.0 * v4o[IDX_OVVNO(j,a,c,k)] + v4v[IDX_OVVNO(j,a,c,k)];
        v += v4 * t1[IDX2(j,b)];
    }
    // step2: -= same with b↔c: sum_j (2*v4o + v4v)[j,a,b,k]*t1[j,c]
    for (int j = 0; j < NO; j++) {
        real_t v4 = 2.0 * v4o[IDX_OVVNO(j,a,b,k)] + v4v[IDX_OVVNO(j,a,b,k)];
        v -= v4 * t1[IDX2(j,c)];
    }
    // step3: wvvvo[c,a,b,k] += sum_j (v4o-v4v)[j,a,c,k]*t1[j,b]
    //   At current (b,a,c,k), reindex (b↔c): contribution = sum_j (v4o-v4v)[j,a,b,k]*t1[j,c]
    for (int j = 0; j < NO; j++) {
        real_t v4d = v4o[IDX_OVVNO(j,a,b,k)] - v4v[IDX_OVVNO(j,a,b,k)];
        v += v4d * t1[IDX2(j,c)];
    }
    // step4: wvvvo[c,a,b,k] -= sum_{l,j} ovoo1[l,a,j,k]*tau[j,l,b,c]
    //   At (b,a,c,k) (i.e. dest indices match (c,a,b,k) under b↔c swap):
    //   contribution = -sum_{l,j} ovoo1[l,a,j,k]*tau[j,l,c,b]
    for (int l = 0; l < NO; l++)
      for (int j = 0; j < NO; j++) {
        real_t ovoo1 = 2.0*ovoo[IDX_OVOO(l,a,j,k)] - ovoo[IDX_OVOO(j,a,l,k)];
        v -= ovoo1 * tau[IDX4(j,l,c,b)];
      }
    // step5: wvvvo[b,a,c,k] += 1.5 * sum_{k',d} ovvv[k',a,c,d]*t2[k',k,b,d]
    //   (Note: k inside einsum is the dummy index; the dest k matches output k)
    //   PySCF: einsum('kacd,kjbd->bacj', ovvv, t2)*1.5 → sum_{k',d} ovvv[k',a,c,d]*t2[k',j,b,d] at (b,a,c,j)
    //   At (b,a,c,k) here = (b,a,c,j_out): += 1.5 * sum_{k',d} ovvv[k',a,c,d]*t2[k',k,b,d]
    {
        real_t sum = 0.0;
        for (int kp = 0; kp < NO; kp++)
          for (int d = 0; d < NV; d++)
            sum += ovvv[IDX_OVVV(kp,a,c,d)] * t2[IDX4(kp,k,b,d)];
        v += 1.5 * sum;
    }
    // step6: tmp[c,a,b,j] = sum_{k',d} ovvv1[k',d,c,a]*theta[j,k',b,d]
    //         wvvvo -= tmp;  wvvvo += tmp.T(2,1,0,3)*0.5
    //   At (b,a,c,k):
    //     wvvvo -= tmp[b,a,c,k]   = -sum_{k',d} ovvv1[k',d,b,a]*theta[k,k',c,d]
    //     wvvvo[bacj] += 0.5*tmpT[b,a,c,k] = 0.5*tmp[c,a,b,k] = 0.5*sum_{k',d} ovvv1[k',d,c,a]*theta[k,k',b,d]
    {
        real_t sum1 = 0.0, sum2 = 0.0;
        for (int kp = 0; kp < NO; kp++)
          for (int d = 0; d < NV; d++) {
            real_t ovvv1_dba = 2.0*ovvv[IDX_OVVV(kp,d,b,a)] - ovvv[IDX_OVVV(kp,a,b,d)];
            real_t ovvv1_dca = 2.0*ovvv[IDX_OVVV(kp,d,c,a)] - ovvv[IDX_OVVV(kp,a,c,d)];
            sum1 += ovvv1_dba * theta[IDX4(k,kp,c,d)];
            sum2 += ovvv1_dca * theta[IDX4(k,kp,b,d)];
          }
        v -= sum1;
        v += 0.5 * sum2;
    }
    // step7: -= ovvv1[k,c,a,b]
    v -= 2.0*ovvv[IDX_OVVV(k,c,a,b)] - ovvv[IDX_OVVV(k,b,a,c)];

    wvvvo[idx] = v;
}

// w3[c,k] = sum_{j,b} (2*v4o + v4v)[j,b,c,k]*t1[j,b]
//         + v5[c,k] + sum_b v1[c,b]*t1[k,b] - sum_j v2[j,k]*t1[j,c]
__global__ void compute_w3_k(const real_t* v4o, const real_t* v4v,
                             const real_t* v1, const real_t* v2, const real_t* v5,
                             const real_t* t1,
                             real_t* w3, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NV*NO) return;
    int c = idx / NO;
    int k = idx % NO;
    real_t v = v5[idx];
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++) {
        real_t v4 = 2.0 * v4o[IDX_OVVNO(j,b,c,k)] + v4v[IDX_OVVNO(j,b,c,k)];
        v += v4 * t1[IDX2(j,b)];
      }
    for (int b = 0; b < NV; b++) v += v1[c*NV + b] * t1[IDX2(k,b)];
    for (int j = 0; j < NO; j++) v -= v2[j*NO + k] * t1[IDX2(j,c)];
    w3[idx] = v;
}

// ============================================================================
//  mvv, moo, mvv1, moo1
// ============================================================================
__global__ void compute_mvv_moo_k(const real_t* l2, const real_t* theta,
                                  real_t* mvv, real_t* moo, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NV*NV + NO*NO;
    if (idx >= total) return;
    if (idx < NV*NV) {
        int a = idx / NV;
        int b = idx % NV;
        real_t v = 0.0;
        // mvv[a,b] = Σ l2[k,l,c,b]*theta[k,l,c,a]  (PySCF 'klca,klcb->ba')
        for (int k = 0; k < NO; k++)
          for (int l = 0; l < NO; l++)
            for (int c = 0; c < NV; c++)
              v += l2[IDX4(k,l,c,b)] * theta[IDX4(k,l,c,a)];
        mvv[idx] = v;
    } else {
        int oo_idx = idx - NV*NV;
        int i = oo_idx / NO;
        int j = oo_idx % NO;
        real_t v = 0.0;
        for (int k = 0; k < NO; k++)
          for (int c = 0; c < NV; c++)
            for (int d = 0; d < NV; d++)
              v += l2[IDX4(k,i,c,d)] * theta[IDX4(k,j,c,d)];
        moo[oo_idx] = v;
    }
}

__global__ void compute_mvv1_k(const real_t* mvv, const real_t* l1, const real_t* t1,
                               real_t* mvv1, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NV*NV) return;
    int b = idx / NV;
    int c = idx % NV;
    real_t v = mvv[idx];
    for (int j = 0; j < NO; j++) v += l1[IDX2(j,c)] * t1[IDX2(j,b)];
    mvv1[idx] = v;
}

__global__ void compute_moo1_k(const real_t* moo, const real_t* l1, const real_t* t1,
                               real_t* moo1, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO) return;
    int i = idx / NO;
    int k = idx % NO;
    real_t v = moo[idx];
    for (int c = 0; c < NV; c++) v += l1[IDX2(i,c)] * t1[IDX2(k,c)];
    moo1[idx] = v;
}

// ============================================================================
//  m3 second + third terms: m3 = 0.5*(l2.vvvv + l2.woooo) + 0.5*ovov*(l2.tau)
//  Only second + third here; l2.vvvv computed via cuBLAS DGEMM in host code.
// ============================================================================
__global__ void compute_m3_woooo_k(const real_t* l2, const real_t* woooo,
                                   real_t* m3_add, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    real_t v = 0.0;
    for (int k = 0; k < NO; k++)
      for (int l = 0; l < NO; l++)
        v += l2[IDX4(k,l,a,b)] * woooo[IDX_WOOOO(i,k,j,l)];
    m3_add[idx] = v;
}

// l2tau[i,j,k,l] = sum_{c,d} l2[i,j,c,d]*tau[k,l,c,d]
__global__ void compute_l2tau_k(const real_t* l2, const real_t* tau,
                                real_t* l2tau, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NO*NO*NO) return;
    int i = idx / (NO*NO*NO);
    int r = idx % (NO*NO*NO);
    int j = r / (NO*NO);
    r %= (NO*NO);
    int k = r / NO;
    int l = r % NO;
    real_t v = 0.0;
    for (int c = 0; c < NV; c++)
      for (int d = 0; d < NV; d++)
        v += l2[IDX4(i,j,c,d)] * tau[IDX4(k,l,c,d)];
    l2tau[idx] = v;
}

// m3_part3[i,j,a,b] += 0.5 * sum_{k,l} ovov[k,a,l,b]*l2tau[i,j,k,l]
__global__ void compute_m3_ovov_k(const real_t* ovov, const real_t* l2tau,
                                  real_t* m3, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    real_t v = 0.0;
    for (int k = 0; k < NO; k++)
      for (int l = 0; l < NO; l++)
        v += ovov[IDX_OVOV(k,a,l,b)] * l2tau[(((size_t)i * NO + j) * NO + k) * NO + l];
    m3[idx] += 0.5 * v;
}

// l2t1[i,j,c,k] = sum_d l2[i,j,c,d]*t1[k,d]
__global__ void compute_l2t1_k(const real_t* l2, const real_t* t1,
                               real_t* l2t1, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NO;
    if (idx >= total) return;
    int i = idx / (NO*NV*NO);
    size_t r = idx % (NO*NV*NO);
    int j = r / (NV*NO);
    r %= (NV*NO);
    int c = r / NO;
    int k = r % NO;
    real_t v = 0.0;
    for (int d = 0; d < NV; d++)
      v += l2[IDX4(i,j,c,d)] * t1[IDX2(k,d)];
    l2t1[idx] = v;
}

// m3 -= einsum('kbca,ijck->ijab', ovvv, l2t1)
__global__ void m3_sub_ovvv_l2t1_k(const real_t* ovvv, const real_t* l2t1,
                                   real_t* m3, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    real_t v = 0.0;
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        v += ovvv[IDX_OVVV(k,b,c,a)] * l2t1[(((size_t)i * NO + j) * NV + c) * NO + k];
    m3[idx] -= v;
}

// scale by 0.5
__global__ void scale_inplace_k(real_t* x, real_t s, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= s;
}

// ============================================================================
//  l2new and l1new — many contributions
// ============================================================================

// l2new[i,j,a,b] = 0.5*ovov[i,a,j,b]
//                + sum_c l2[i,j,a,c]*v1[c,b]
//                - sum_k l2[i,k,a,b]*v2[j,k]
//                - sum_c mvv1[c,a]*ovov[i,c,j,b]
//                - sum_k moo1[i,k]*ovov[k,a,j,b]
//                + l1[i,a]*v4[j,b]
//                + sum_c l1[i,c]*ovvv[j,b,c,a]   (from .T, but mapped: see CPU code 'jiba')
//                - sum_k l1[k,a]*ovoo[j,b,i,k]
//                + 0.5 * sum_{k,c} (2*l2[i,k,a,c]-l2[i,k,c,a])*wovvo[j,b,c,k]
//                + 0.5 * sum_{k,c} l2[i,k,c,a]*woVVo[j,b,c,k]
//                +       sum_{k,c} l2[j,k,c,a]*woVVo[i,b,c,k]   (from tmp.T(1,0,2,3))
__global__ void compute_l2new_k(const real_t* ovov, const real_t* ovvv, const real_t* ovoo,
                                const real_t* l1,   const real_t* l2,
                                const real_t* v1,   const real_t* v2,   const real_t* v4,
                                const real_t* mvv1, const real_t* moo1,
                                const real_t* wovvo,const real_t* woVVo,
                                real_t* l2new, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;

    real_t v = 0.5 * ovov[IDX_OVOV(i,a,j,b)];
    for (int c = 0; c < NV; c++) v += l2[IDX4(i,j,a,c)] * v1[c*NV + b];
    for (int k = 0; k < NO; k++) v -= l2[IDX4(i,k,a,b)] * v2[j*NO + k];
    for (int c = 0; c < NV; c++) v -= mvv1[c*NV + a] * ovov[IDX_OVOV(i,c,j,b)];
    for (int k = 0; k < NO; k++) v -= moo1[i*NO + k] * ovov[IDX_OVOV(k,a,j,b)];
    v += l1[IDX2(i,a)] * v4[IDX2(j,b)];
    // PySCF: l2new += einsum('ic,jbca->jiba', l1, ovvv)  → at dest (j,i,b,a)... we are at (i,j,a,b).
    // Reindex (i↔j, a↔b): contribution += sum_c l1[j,c]*ovvv[i,a,c,b]
    for (int c = 0; c < NV; c++) v += l1[IDX2(j,c)] * ovvv[IDX_OVVV(i,a,c,b)];
    // PySCF: l2new -= einsum('ka,jbik->ijab', l1, ovoo) — at (i,j,a,b)
    for (int k = 0; k < NO; k++) v -= l1[IDX2(k,a)] * ovoo[IDX_OVOO(j,b,i,k)];
    // wovvo and woVVo terms
    real_t v_w = 0.0, v_wV = 0.0;
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++) {
        v_w  += (2.0*l2[IDX4(i,k,a,c)] - l2[IDX4(i,k,c,a)]) * wovvo[IDX_OVVNO(j,b,c,k)];
        v_wV +=     l2[IDX4(i,k,c,a)]                       * woVVo[IDX_OVVNO(j,b,c,k)];
      }
    // tmp.T(1,0,2,3): at (i,j,a,b), tmp[j,i,a,b] = sum_{k,c} l2[j,k,c,a]*woVVo[i,b,c,k]
    real_t v_wV2 = 0.0;
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        v_wV2 += l2[IDX4(j,k,c,a)] * woVVo[IDX_OVVNO(i,b,c,k)];
    v += 0.5 * v_w + 0.5 * v_wV + v_wV2;

    l2new[idx] = v;
}

// l1new[i,a] = -sum_k moo[i,k]*v4[k,a]
//             -sum_c mvv[c,a]*v4[i,c]
//             + sum_{j,b} ovov1[j,b,i,a]*tmp_jb            (tmp computed separately)
//             + sum_{b,c} (2*ovvv[i,a,c,b] - ovvv[i,b,c,a])*mvv1[b,c]
//             + 2*sum_{j,b} m3[i,j,a,b]*t1[j,b]
//             + 2*sum_{j,b} m3[j,i,b,a]*t1[j,b]
//             -   sum_{j,b} m3[i,j,b,a]*t1[j,b]
//             -   sum_{j,b} m3[j,i,a,b]*t1[j,b]
//             - 2*sum_{j,k} ovoo[i,a,j,k]*moo1[k,j]
//             +   sum_{j,k} ovoo[j,a,i,k]*moo1[k,j]
//             + sum_b l1[i,b]*v1[b,a]
//             - sum_j l1[j,a]*v2[i,j]
//             + 2*sum_{j,b} l1[j,b]*ovvo[i,a,b,j]
//             -   sum_{j,b} l1[j,b]*oovv[i,j,b,a]
//             -   sum_{j,b,c} l2[i,j,b,c]*wvvvo[b,a,c,j]
//             -   sum_{k,j,c} l2[k,j,c,a]*woovo[i,j,c,k]
//             + 2*sum_{j,b} l2[i,j,a,b]*w3[b,j]
//             -   sum_{j,b} l2[i,j,b,a]*w3[b,j]
//
// Helper tmp[j,b] = t1[j,b] + sum_{k,c} l1[k,c]*theta[k,j,c,b]
//                            - sum_d mvv1[b,d]*t1[j,d]
//                            - sum_l moo[l,j]*t1[l,b]   (computed by separate kernel)

__global__ void compute_l1new_tmp_k(const real_t* t1, const real_t* l1, const real_t* theta,
                                    const real_t* mvv1, const real_t* moo,
                                    real_t* tmp, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV) return;
    int j = idx / NV;
    int b = idx % NV;
    real_t v = t1[idx];
    for (int k = 0; k < NO; k++)
      for (int c = 0; c < NV; c++)
        v += l1[IDX2(k,c)] * theta[IDX4(k,j,c,b)];
    for (int d = 0; d < NV; d++) v -= mvv1[b*NV + d] * t1[IDX2(j,d)];
    for (int l = 0; l < NO; l++) v -= moo[l*NO + j] * t1[IDX2(l,b)];
    tmp[idx] = v;
}

__global__ void compute_l1new_k(
    const real_t* ovov, const real_t* ovoo, const real_t* ovvv,
    const real_t* ovvo, const real_t* oovv,
    const real_t* l1, const real_t* l2,
    const real_t* v1, const real_t* v2, const real_t* v4,
    const real_t* mvv, const real_t* mvv1, const real_t* moo, const real_t* moo1,
    const real_t* wovvo_unused, const real_t* woovo, const real_t* wvvvo, const real_t* w3,
    const real_t* m3, const real_t* t1,
    const real_t* tmp_jb,
    real_t* l1new, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV) return;
    int i = idx / NV;
    int a = idx % NV;
    real_t v = 0.0;
    // -moo*v4
    for (int k = 0; k < NO; k++) v -= moo[i*NO + k] * v4[IDX2(k,a)];
    // -mvv*v4
    for (int c = 0; c < NV; c++) v -= mvv[c*NV + a] * v4[IDX2(i,c)];
    // ovov1*tmp
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++) {
        real_t ov1 = 2.0*ovov[IDX_OVOV(j,b,i,a)] - ovov[IDX_OVOV(j,a,i,b)];
        v += ov1 * tmp_jb[IDX2(j,b)];
      }
    // ovvv*mvv1
    for (int b = 0; b < NV; b++)
      for (int c = 0; c < NV; c++) {
        v += 2.0 * ovvv[IDX_OVVV(i,a,c,b)] * mvv1[b*NV + c];
        v -=       ovvv[IDX_OVVV(i,b,c,a)] * mvv1[b*NV + c];
      }
    // m3 contributions
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++) {
        v += 2.0 * m3[IDX4(i,j,a,b)] * t1[IDX2(j,b)];
        v += 2.0 * m3[IDX4(j,i,b,a)] * t1[IDX2(j,b)];
        v -=       m3[IDX4(i,j,b,a)] * t1[IDX2(j,b)];
        v -=       m3[IDX4(j,i,a,b)] * t1[IDX2(j,b)];
      }
    // ovoo*moo1
    for (int j = 0; j < NO; j++)
      for (int k = 0; k < NO; k++) {
        v -= 2.0 * ovoo[IDX_OVOO(i,a,j,k)] * moo1[k*NO + j];
        v +=       ovoo[IDX_OVOO(j,a,i,k)] * moo1[k*NO + j];
      }
    // l1*v1, l1*v2
    for (int b = 0; b < NV; b++) v += l1[IDX2(i,b)] * v1[b*NV + a];
    for (int j = 0; j < NO; j++) v -= l1[IDX2(j,a)] * v2[i*NO + j];
    // l1*ovvo, l1*oovv
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++) {
        v += 2.0 * l1[IDX2(j,b)] * ovvo[IDX_OVVO(i,a,b,j)];
        v -=       l1[IDX2(j,b)] * oovv[IDX_OOVV(i,j,b,a)];
      }
    // l2*wvvvo, l2*woovo
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++)
        for (int c = 0; c < NV; c++)
          v -= l2[IDX4(i,j,b,c)] * wvvvo[(((size_t)b * NV + a) * NV + c) * NO + j];
    for (int k = 0; k < NO; k++)
      for (int j = 0; j < NO; j++)
        for (int c = 0; c < NV; c++)
          v -= l2[IDX4(k,j,c,a)] * woovo[(((size_t)i * NO + j) * NV + c) * NO + k];
    // l2*w3
    for (int j = 0; j < NO; j++)
      for (int b = 0; b < NV; b++) {
        v += 2.0 * l2[IDX4(i,j,a,b)] * w3[b*NO + j];
        v -=       l2[IDX4(i,j,b,a)] * w3[b*NO + j];
      }
    l1new[idx] = v;
}

// Apply denominators: l1new /= eia ; l1new += l1
__global__ void apply_l1_denom_k(real_t* l1new, const real_t* l1, const real_t* eps,
                                 int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NO*NV) return;
    int i = idx / NV;
    int a = idx % NV;
    real_t eia = eps[i] - eps[NO + a];
    l1new[idx] = l1new[idx] / eia + l1[idx];
}

// l2new = l2new + l2new.T(1,0,3,2); l2new /= (eia + ejb); l2new += l2
__global__ void symmetrize_and_apply_l2_denom_k(const real_t* l2new_in, const real_t* l2,
                                                const real_t* eps,
                                                real_t* l2new_out, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    real_t sym = l2new_in[idx] + l2new_in[IDX4(j,i,b,a)];
    real_t denom = (eps[i] - eps[NO + a]) + (eps[j] - eps[NO + b]);
    l2new_out[idx] = sym / denom + l2[idx];
}

// ============================================================================
//  Compute residual norm via reduction
// ============================================================================
__global__ void diff_sq_sum_k(const real_t* a, const real_t* b, real_t* out, size_t n) {
    __shared__ real_t s[256];
    int tid = threadIdx.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? (a[i] - b[i]) * (a[i] - b[i]) : 0.0;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, s[0]);
}

// ============================================================================
//  l2 · vvvv  (the dominant O(no² nv⁴) contraction) via cuBLAS DGEMM.
//   l2[i,j,c,d] * vvvv[a,c,b,d] → result[i,j,a,b]
//   Reshape: l2 as (no², nv²) with rows (i,j) cols (c,d) contiguous.
//            vvvv as (nv², nv²) with rows (a,c) cols (b,d).
//   result[i,j,a,b] = Σ_{c,d} L[(i,j),(c,d)] * V[(a,c),(b,d)]
//   This isn't a clean matmul. Need to permute.
//
//  Alternative: reshape vvvv → V'[c,d,a,b] then result = L × V'^?
//
//  Actually let's use:
//   l2[i,j,c,d] * vvvv[a,c,b,d]
//   = sum_d (sum_c l2[i,j,c,d]*vvvv[a,c,b,d])
//   Define X[a,b,i,j,d] = sum_c vvvv[a,c,b,d]*l2[i,j,c,d]  — too many indices
//
//  Simplest correct approach: a kernel that does it directly.
// ============================================================================
__global__ void m3_l2_vvvv_k(const real_t* l2, const real_t* vvvv,
                             real_t* m3_out, int NO, int NV)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)NO*NO*NV*NV;
    if (idx >= total) return;
    int i = idx / (NO*NV*NV);
    size_t r = idx % (NO*NV*NV);
    int j = r / (NV*NV);
    r %= (NV*NV);
    int a = r / NV;
    int b = r % NV;
    real_t v = 0.0;
    for (int c = 0; c < NV; c++)
      for (int d = 0; d < NV; d++)
        v += l2[IDX4(i,j,c,d)] * vvvv[IDX_VVVV(a,c,b,d)];
    m3_out[idx] = v;
}

// ============================================================================
//  1-RDM (gpu) — direct port from CPU build_ccsd_1rdm_mo_cpu
// ============================================================================
__global__ void build_d1_doo_dvv_kernels_k(
    const real_t* t1, const real_t* t2, const real_t* l1, const real_t* l2,
    real_t* doo, real_t* dvv, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NO*NO + NV*NV;
    if (idx >= total) return;
    if (idx < NO*NO) {
        int i = idx / NO;
        int j = idx % NO;
        real_t v = 0.0;
        for (int a = 0; a < NV; a++) v -= t1[IDX2(j,a)] * l1[IDX2(i,a)];
        for (int k = 0; k < NO; k++)
          for (int a = 0; a < NV; a++)
            for (int b = 0; b < NV; b++) {
              real_t th = 2.0 * t2[IDX4(j,k,a,b)] - t2[IDX4(j,k,b,a)];
              v -= th * l2[IDX4(i,k,a,b)];
            }
        doo[idx] = v;
    } else {
        int vv_idx = idx - NO*NO;
        int a = vv_idx / NV;
        int b = vv_idx % NV;
        real_t v = 0.0;
        for (int i = 0; i < NO; i++) v += t1[IDX2(i,a)] * l1[IDX2(i,b)];
        for (int j = 0; j < NO; j++)
          for (int i = 0; i < NO; i++)
            for (int c = 0; c < NV; c++) {
              real_t th = 2.0 * t2[IDX4(j,i,c,a)] - t2[IDX4(j,i,a,c)];
              v += th * l2[IDX4(j,i,c,b)];
            }
        dvv[vv_idx] = v;
    }
}

__global__ void build_xt1_xt2_k(const real_t* l2, const real_t* t2,
                                real_t* xt1, real_t* xt2, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NO*NO + NV*NV;
    if (idx >= total) return;
    if (idx < NO*NO) {
        int m = idx / NO;
        int i = idx % NO;
        real_t v = 0.0;
        for (int n = 0; n < NO; n++)
          for (int e = 0; e < NV; e++)
            for (int f = 0; f < NV; f++) {
              real_t th = 2.0 * t2[IDX4(i,n,e,f)] - t2[IDX4(i,n,f,e)];
              v += l2[IDX4(m,n,e,f)] * th;
            }
        xt1[idx] = v;
    } else {
        int vv_idx = idx - NO*NO;
        int e = vv_idx / NV;
        int a = vv_idx % NV;
        real_t v = 0.0;
        for (int m = 0; m < NO; m++)
          for (int n = 0; n < NO; n++)
            for (int f = 0; f < NV; f++) {
              real_t th = 2.0 * t2[IDX4(m,n,e,f)] - t2[IDX4(m,n,f,e)];
              v += l2[IDX4(m,n,a,f)] * th;
            }
        xt2[vv_idx] = v;
    }
}

__global__ void build_dvo_k(const real_t* t1, const real_t* t2, const real_t* l1,
                            const real_t* xt1, const real_t* xt2,
                            real_t* dvo, int NO, int NV)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NV*NO) return;
    int a = idx / NO;
    int i = idx % NO;
    real_t v = t1[IDX2(i,a)];
    for (int m = 0; m < NO; m++)
      for (int e = 0; e < NV; e++) {
        real_t th = 2.0 * t2[IDX4(i,m,a,e)] - t2[IDX4(i,m,e,a)];
        v += th * l1[IDX2(m,e)];
      }
    for (int m = 0; m < NO; m++) v -= xt1[m*NO + i] * t1[IDX2(m,a)];
    for (int e = 0; e < NV; e++) v -= t1[IDX2(i,e)] * xt2[e*NV + a];
    dvo[idx] = v;
}

__global__ void assemble_dm_k(const real_t* doo, const real_t* dvv,
                              const real_t* dvo, const real_t* l1,
                              real_t* dm, int NO, int NV, int NA)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NA*NA) return;
    int p = idx / NA;
    int q = idx % NA;
    real_t v = 0.0;
    if (p < NO && q < NO) {
        v = doo[p*NO + q] + doo[q*NO + p];
        if (p == q) v += 2.0;
    } else if (p >= NO && q >= NO) {
        int a = p - NO, b = q - NO;
        v = dvv[a*NV + b] + dvv[b*NV + a];
    } else if (p < NO && q >= NO) {
        int a = q - NO;
        v = l1[IDX2(p,a)] + dvo[a*NO + p];
    } else { // p>=NO, q<NO
        int a = p - NO;
        v = l1[IDX2(q,a)] + dvo[a*NO + q];
    }
    dm[idx] = v;
}

// ============================================================================
//  Main solve_ccsd_lambda_gpu
// ============================================================================

bool solve_ccsd_lambda_gpu(
    int nocc, int nvir,
    const real_t* d_eps,
    const real_t* d_eri_mo,
    const real_t* d_t1,
    const real_t* d_t2,
    real_t* d_lambda1,
    real_t* d_lambda2,
    int max_iter,
    real_t tol,
    int verbose)
{
    const int NO = nocc, NV = nvir, NA = nocc + nvir;
    const size_t l1_sz = (size_t)NO * NV;
    const size_t l2_sz = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = l2_sz;  // [NO,NV,NO,NV] = NO² NV²
    const size_t ovoo_sz = (size_t)NO * NV * NO * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t oovv_sz = l2_sz;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;
    const size_t vvvv_sz = (size_t)NV * NV * NV * NV;

    // Allocate sub-blocks
    real_t *d_ovov=nullptr, *d_ovoo=nullptr, *d_ovvv=nullptr;
    real_t *d_oovv=nullptr, *d_ovvo=nullptr, *d_oooo=nullptr, *d_vvvv=nullptr;
    tracked_cudaMalloc(&d_ovov, ovov_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_ovoo, ovoo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_ovvv, ovvv_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_oovv, oovv_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_ovvo, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_oooo, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_vvvv, vvvv_sz * sizeof(real_t));

    int B = 256;
    auto launch = [&](size_t n) { return std::pair<int,int>{(int)((n + B - 1) / B), B}; };

    {
        auto p = launch(ovov_sz);  extract_ovov_k<<<p.first, p.second>>>(d_eri_mo, d_ovov, NA, NO, NV);
        p = launch(ovoo_sz);       extract_ovoo_k<<<p.first, p.second>>>(d_eri_mo, d_ovoo, NA, NO, NV);
        p = launch(ovvv_sz);       extract_ovvv_k<<<p.first, p.second>>>(d_eri_mo, d_ovvv, NA, NO, NV);
        p = launch(oovv_sz);       extract_oovv_k<<<p.first, p.second>>>(d_eri_mo, d_oovv, NA, NO, NV);
        p = launch(ovvo_sz);       extract_ovvo_k<<<p.first, p.second>>>(d_eri_mo, d_ovvo, NA, NO, NV);
        p = launch(oooo_sz);       extract_oooo_k<<<p.first, p.second>>>(d_eri_mo, d_oooo, NA, NO);
        p = launch(vvvv_sz);       extract_vvvv_k<<<p.first, p.second>>>(d_eri_mo, d_vvvv, NA, NO, NV);
    }
    cudaDeviceSynchronize();

    // Allocate intermediates and lambda buffers
    real_t *d_tau=nullptr, *d_theta=nullptr;
    real_t *d_v1=nullptr, *d_v2=nullptr, *d_v4=nullptr, *d_v5=nullptr, *d_w3=nullptr;
    real_t *d_woooo=nullptr, *d_v4o=nullptr, *d_v4v=nullptr;
    real_t *d_wOVvo=nullptr, *d_woVVo=nullptr, *d_wovvo=nullptr;
    real_t *d_woovo=nullptr, *d_wvvvo=nullptr;
    real_t *d_mvv=nullptr, *d_moo=nullptr, *d_mvv1=nullptr, *d_moo1=nullptr;
    real_t *d_m3=nullptr, *d_l2tau=nullptr, *d_l2t1=nullptr;
    real_t *d_tmp_jb=nullptr;
    real_t *d_l1new=nullptr, *d_l2new_pre=nullptr;

    tracked_cudaMalloc(&d_tau,   l2_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_theta, l2_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_v1, NV*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_v2, NO*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_v4, NO*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_v5, NV*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_w3, NV*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_woooo, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_v4o, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_v4v, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_wOVvo, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_woVVo, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_wovvo, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_woovo, (size_t)NO*NO*NV*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_wvvvo, (size_t)NV*NV*NV*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_mvv, NV*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_moo, NO*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_mvv1, NV*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_moo1, NO*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_m3,   l2_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_l2tau, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_l2t1, (size_t)NO*NO*NV*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_tmp_jb, NO*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_l1new, l1_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_l2new_pre, l2_sz * sizeof(real_t));

    // tau, theta (depend only on T, compute once)
    {
        auto p = launch(l2_sz);
        make_tau_k<<<p.first, p.second>>>(d_t1, d_t2, d_tau, NO, NV);
        make_theta_k<<<p.first, p.second>>>(d_t2, d_theta, NO, NV);
    }

    // Initial guess Λ = 0 (PySCF default)
    cudaMemset(d_lambda1, 0, l1_sz * sizeof(real_t));
    cudaMemset(d_lambda2, 0, l2_sz * sizeof(real_t));

    // Compute T-only intermediates (don't change between iterations)
    {
        auto p = launch(NV*NV); compute_v1_k<<<p.first, p.second>>>(d_ovov, d_ovvv, d_t1, d_tau, d_eps, d_v1, NO, NV);
        p = launch(NO*NO);      compute_v2_k<<<p.first, p.second>>>(d_ovov, d_ovoo, d_t1, d_tau, d_eps, d_v2, NO, NV);
        p = launch(NO*NV);      compute_v4_k<<<p.first, p.second>>>(d_ovov, d_t1, d_v4, NO, NV);
        cudaDeviceSynchronize();
        p = launch(NV*NO);      compute_v5_k<<<p.first, p.second>>>(d_ovoo, d_ovvv, d_t1, d_t2, d_v4, d_v5, NO, NV);
        p = launch(oooo_sz);    compute_woooo_k<<<p.first, p.second>>>(d_oooo, d_ovoo, d_ovov, d_t1, d_tau, d_woooo, NO, NV);
        p = launch(ovvo_sz);    compute_v4OVvo_k<<<p.first, p.second>>>(d_ovov, d_t2, d_ovvo, d_v4o, NO, NV);
        p = launch(ovvo_sz);    compute_v4oVVo_k<<<p.first, p.second>>>(d_ovov, d_t2, d_oovv, d_v4v, NO, NV);
        cudaDeviceSynchronize();
        p = launch(ovvo_sz);    compute_wOVvo_woVVo_k<<<p.first, p.second>>>(d_v4o, d_v4v, d_ovov, d_ovoo, d_ovvv, d_t1, d_wOVvo, d_woVVo, NO, NV);
        p = launch(NV*NO);      compute_w3_k<<<p.first, p.second>>>(d_v4o, d_v4v, d_v1, d_v2, d_v5, d_t1, d_w3, NO, NV);
        cudaDeviceSynchronize();
    }
    // wovvo = 2*wOVvo + woVVo
    {
        auto p = launch(ovvo_sz);
        combine_wovvo_k<<<p.first, p.second>>>(d_wOVvo, d_woVVo, d_wovvo, ovvo_sz);
    }

    // woovo and wvvvo (T-only)
    {
        auto p = launch((size_t)NO*NO*NV*NO);
        compute_woovo_k<<<p.first, p.second>>>(d_v4o, d_v4v, d_ovoo, d_ovvv, d_t1, d_t2, d_theta, d_tau, d_woovo, NO, NV);
        p = launch((size_t)NV*NV*NV*NO);
        compute_wvvvo_k<<<p.first, p.second>>>(d_v4o, d_v4v, d_ovoo, d_ovvv, d_t1, d_t2, d_theta, d_tau, d_wvvvo, NO, NV);
        cudaDeviceSynchronize();
    }

    if (verbose > 0) {
        std::cout << "CCSD Lambda solver (GPU): nocc=" << NO << " nvir=" << NV
                  << " max_iter=" << max_iter << " tol=" << std::scientific << tol
                  << std::defaultfloat << std::endl;
    }

    real_t* d_resid_buf;
    tracked_cudaMalloc(&d_resid_buf, sizeof(real_t));

    bool converged = false;
    for (int iter = 0; iter < max_iter; iter++) {
        // λ-dependent intermediates
        {
            auto p = launch((size_t)NV*NV + NO*NO);
            compute_mvv_moo_k<<<p.first, p.second>>>(d_lambda2, d_theta, d_mvv, d_moo, NO, NV);
            cudaDeviceSynchronize();
            p = launch(NV*NV); compute_mvv1_k<<<p.first, p.second>>>(d_mvv, d_lambda1, d_t1, d_mvv1, NO, NV);
            p = launch(NO*NO); compute_moo1_k<<<p.first, p.second>>>(d_moo, d_lambda1, d_t1, d_moo1, NO, NV);
        }
        // m3: l2.vvvv (kernel) + l2.woooo + 0.5*scale + ovov*l2tau, then -= ovvv*l2t1
        {
            auto p = launch(l2_sz);
            m3_l2_vvvv_k<<<p.first, p.second>>>(d_lambda2, d_vvvv, d_m3, NO, NV);
            // m3 += l2.woooo  via reuse of m3_woooo kernel (which OVERWRITES, so accumulate into temp)
            // Use d_l2new_pre as temp scratch (overwritten later anyway):
            compute_m3_woooo_k<<<p.first, p.second>>>(d_lambda2, d_woooo, d_l2new_pre, NO, NV);
            // m3 += temp
            // Use scale_inplace_k? No, need add. Use cuBLAS daxpy.
            real_t one = 1.0;
            cublasDaxpy(gpu::GPUHandle::cublas(), (int)l2_sz, &one, d_l2new_pre, 1, d_m3, 1);
            // m3 *= 0.5
            real_t half = 0.5;
            cublasDscal(gpu::GPUHandle::cublas(), (int)l2_sz, &half, d_m3, 1);
            // m3 += 0.5 * ovov * l2tau
            auto p2 = launch(oooo_sz);
            compute_l2tau_k<<<p2.first, p2.second>>>(d_lambda2, d_tau, d_l2tau, NO, NV);
            cudaDeviceSynchronize();
            compute_m3_ovov_k<<<p.first, p.second>>>(d_ovov, d_l2tau, d_m3, NO, NV);
            // m3 -= ovvv*l2t1
            auto p3 = launch((size_t)NO*NO*NV*NO);
            compute_l2t1_k<<<p3.first, p3.second>>>(d_lambda2, d_t1, d_l2t1, NO, NV);
            cudaDeviceSynchronize();
            m3_sub_ovvv_l2t1_k<<<p.first, p.second>>>(d_ovvv, d_l2t1, d_m3, NO, NV);
        }
        // l2new (pre-symmetrization, pre-denominator), into d_l2new_pre
        {
            auto p = launch(l2_sz);
            compute_l2new_k<<<p.first, p.second>>>(d_ovov, d_ovvv, d_ovoo,
                                                   d_lambda1, d_lambda2,
                                                   d_v1, d_v2, d_v4, d_mvv1, d_moo1,
                                                   d_wovvo, d_woVVo,
                                                   d_l2new_pre, NO, NV);
            // l2new += m3
            real_t one = 1.0;
            cublasDaxpy(gpu::GPUHandle::cublas(), (int)l2_sz, &one, d_m3, 1, d_l2new_pre, 1);
        }
        // l1new
        {
            auto p = launch(NO*NV);
            compute_l1new_tmp_k<<<p.first, p.second>>>(d_t1, d_lambda1, d_theta, d_mvv1, d_moo, d_tmp_jb, NO, NV);
            cudaDeviceSynchronize();
            compute_l1new_k<<<p.first, p.second>>>(d_ovov, d_ovoo, d_ovvv, d_ovvo, d_oovv,
                                                   d_lambda1, d_lambda2, d_v1, d_v2, d_v4,
                                                   d_mvv, d_mvv1, d_moo, d_moo1,
                                                   d_wovvo, d_woovo, d_wvvvo, d_w3,
                                                   d_m3, d_t1, d_tmp_jb,
                                                   d_l1new, NO, NV);
        }
        // Apply denominators
        {
            auto p = launch(NO*NV);
            apply_l1_denom_k<<<p.first, p.second>>>(d_l1new, d_lambda1, d_eps, NO, NV);
            auto p2 = launch(l2_sz);
            // Use d_m3 as scratch for symmetrization output (then copy)
            symmetrize_and_apply_l2_denom_k<<<p2.first, p2.second>>>(d_l2new_pre, d_lambda2, d_eps, d_m3, NO, NV);
        }

        // Compute residual = sqrt(||l1new - lambda1||² + ||m3 - lambda2||²)
        cudaMemset(d_resid_buf, 0, sizeof(real_t));
        {
            auto p = launch(l1_sz);  diff_sq_sum_k<<<p.first, p.second>>>(d_l1new, d_lambda1, d_resid_buf, l1_sz);
            auto p2 = launch(l2_sz); diff_sq_sum_k<<<p2.first, p2.second>>>(d_m3, d_lambda2, d_resid_buf, l2_sz);
        }
        real_t resid = 0.0;
        cudaMemcpy(&resid, d_resid_buf, sizeof(real_t), cudaMemcpyDeviceToHost);
        resid = std::sqrt(resid);

        // Update lambda
        cudaMemcpy(d_lambda1, d_l1new, l1_sz * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_lambda2, d_m3,    l2_sz * sizeof(real_t), cudaMemcpyDeviceToDevice);

        if (verbose > 0) {
            std::cout << "  Lambda iter " << std::setw(3) << (iter + 1)
                      << ": ||Δλ|| = " << std::scientific << std::setprecision(3) << resid
                      << std::defaultfloat << std::endl;
        }
        if (resid < tol) {
            converged = true;
            if (verbose > 0)
                std::cout << "  Lambda converged in " << (iter + 1) << " iterations" << std::endl;
            break;
        }
    }

    // Cleanup
    tracked_cudaFree(d_ovov); tracked_cudaFree(d_ovoo); tracked_cudaFree(d_ovvv);
    tracked_cudaFree(d_oovv); tracked_cudaFree(d_ovvo); tracked_cudaFree(d_oooo); tracked_cudaFree(d_vvvv);
    tracked_cudaFree(d_tau); tracked_cudaFree(d_theta);
    tracked_cudaFree(d_v1); tracked_cudaFree(d_v2); tracked_cudaFree(d_v4); tracked_cudaFree(d_v5); tracked_cudaFree(d_w3);
    tracked_cudaFree(d_woooo); tracked_cudaFree(d_v4o); tracked_cudaFree(d_v4v);
    tracked_cudaFree(d_wOVvo); tracked_cudaFree(d_woVVo); tracked_cudaFree(d_wovvo);
    tracked_cudaFree(d_woovo); tracked_cudaFree(d_wvvvo);
    tracked_cudaFree(d_mvv); tracked_cudaFree(d_moo); tracked_cudaFree(d_mvv1); tracked_cudaFree(d_moo1);
    tracked_cudaFree(d_m3); tracked_cudaFree(d_l2tau); tracked_cudaFree(d_l2t1);
    tracked_cudaFree(d_tmp_jb); tracked_cudaFree(d_l1new); tracked_cudaFree(d_l2new_pre);
    tracked_cudaFree(d_resid_buf);
    return converged;
}

// Helper kernel referenced via extern decl in solve function
__global__ void combine_wovvo_k(const real_t* a, const real_t* b, real_t* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 2.0 * a[i] + b[i];
}

// ============================================================================
//  Public: build_ccsd_1rdm_mo_gpu
// ============================================================================
void build_ccsd_1rdm_mo_gpu(
    int nocc, int nvir,
    const real_t* d_t1, const real_t* d_t2,
    const real_t* d_l1, const real_t* d_l2,
    real_t* d_D_mo_out)
{
    const int NO = nocc, NV = nvir, NA = nocc + nvir;
    real_t *d_doo, *d_dvv, *d_xt1, *d_xt2, *d_dvo;
    tracked_cudaMalloc(&d_doo, NO*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_dvv, NV*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_xt1, NO*NO * sizeof(real_t));
    tracked_cudaMalloc(&d_xt2, NV*NV * sizeof(real_t));
    tracked_cudaMalloc(&d_dvo, NV*NO * sizeof(real_t));

    int B = 256;
    {
        int n = NO*NO + NV*NV;
        build_d1_doo_dvv_kernels_k<<<(n+B-1)/B, B>>>(d_t1, d_t2, d_l1, d_l2, d_doo, d_dvv, NO, NV);
        build_xt1_xt2_k<<<(n+B-1)/B, B>>>(d_l2, d_t2, d_xt1, d_xt2, NO, NV);
    }
    cudaDeviceSynchronize();
    build_dvo_k<<<((NV*NO)+B-1)/B, B>>>(d_t1, d_t2, d_l1, d_xt1, d_xt2, d_dvo, NO, NV);
    assemble_dm_k<<<((NA*NA)+B-1)/B, B>>>(d_doo, d_dvv, d_dvo, d_l1, d_D_mo_out, NO, NV, NA);
    cudaDeviceSynchronize();

    tracked_cudaFree(d_doo); tracked_cudaFree(d_dvv);
    tracked_cudaFree(d_xt1); tracked_cudaFree(d_xt2);
    tracked_cudaFree(d_dvo);
}

} // namespace gansu
