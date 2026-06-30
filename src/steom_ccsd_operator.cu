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

/**
 * @file steom_ccsd_operator.cu
 * @brief STEOM-CCSD operator — sub-phase 3.0-3.4.
 *
 *   3.0+3.1   LinearOperator scaffolding + diagonal-only apply() stub.
 *   3.2       bar-H rebuild: 7 MO ERI blocks + 11 PySCF rintermediates-equivalent
 *             dressed intermediates (Loo, Lvv, Fov, Woooo, Wooov, Wovov,
 *             Wovvo, Wovoo, Wvovv, Wvvvv, Wvvvo).
 *   3.3a     Ŝ amplitude single-divide normalization (R2/(U·R1)). Superseded
 *             by sub-phase 3.4 X-matrix (matrix inverse).
 *   3.4       F^eff_oo dressing per CFOUR `gmi_steom_rhf`:
 *               • build X(MI) = inverse of active R1 matrix (n_act × n_act)
 *               • build U(M,I) = +2 Fov·R2 − Fov·R2 − 2 Wooov·R2 + Wooov·R2
 *                 (PySCF σ1-style spin-adapted from IP-EOM matvec)
 *               • F^eff_oo[M_idx, I] = Loo[M_idx, I] − Σ_N U(N,I) · X(N,M)
 *               • inactive rows of F^eff_oo = bar Loo (no dressing)
 *             Reference: megansimons/steom_ccsd-ct (GitHub) — CFOUR-style
 *             closed-shell RHF STEOM, `steom_intermediates.cxx` lines 7-81 +
 *             `steom.cxx` lines 23-49 (`renormalize`).
 *
 * Sub-phase 3.5+:  W^eff dressing + G(EM)/G(Mn,Ie) intermediates.
 * Sub-phase 3.6:   Cross IP×EA dressing.
 * Sub-phase 3.7:   Full G^{1h1p} matvec assembly (★ H2O sto-3g 1 mHa gate).
 */

#include "steom_ccsd_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>   // Layer 2: SVD pseudo-inverse + conditioning of active R1 (U)

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "multi_gpu_manager.hpp"  // build-phase multi-GPU slab GEMM (Stage 1+)
#include "eri.hpp"   // Phase 0: ERI_RI::mo_eri_block_into (on-the-fly MO-ERI blocks)

#ifndef GANSU_CPU_ONLY
#include <cublas_v2.h>
#endif

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#endif

namespace gansu {

#ifndef GANSU_CPU_ONLY
namespace {
// ------------------------------------------------------------------
//  Build-phase multi-GPU n-slab GEMM (Stage 1).
//  Computes C[m×n] (column-major, ldc=m) = op(A)·op(B), partitioning the
//  n (column) dimension of C across `nuse` devices. Each device holds the
//  FULL host operands hA/hB (element counts aElems/bElems) and computes its
//  contiguous column slab [lo,hi), writing directly into hC + lo*m.
//
//  Because each output column is a COMPLETE K-contraction performed on a
//  single device (no cross-device reduction), the assembled result is
//  bit-identical to the single-GPU GEMM (modulo cuBLAS kernel selection,
//  caught by GANSU_STEOM_BUILD_VALIDATE). The output column slab is a
//  contiguous byte range in column-major C → no gather/reshuffle.
//
//  Own per-device cuBLAS handles are created here (the MultiGpuManager
//  singleton is pinned to 1 device by the CIS-NTO --num_gpus 1 path, so we
//  cannot rely on its handle vector). DeviceGuard is a pure cudaSetDevice
//  RAII and needs no singleton state.
//
//  Phase split: launch ALL device GEMMs first (async), then D2H, so the
//  per-device GEMMs overlap across devices.
// Pinned-aware H2D/D2H helper: cudaHostRegister + cudaMemcpyAsync make
// transfers truly concurrent across devices (each device's PCIe lane runs
// independently).  Without pinning, cudaMemcpy(HtoD) blocks the host so the
// 4-device H2D becomes 4x slower serial — observed +25 s STEOM build at
// anthracene with the prior sync version (single-GPU baseline was faster).
// On register failure the helper transparently falls back to sync cudaMemcpy
// (= legacy behavior, no regression).
static bool try_host_register(const void* ptr, size_t bytes) {
    if (bytes == 0) return false;
    cudaError_t e = cudaHostRegister(const_cast<void*>(ptr), bytes, cudaHostRegisterDefault);
    if (e == cudaErrorHostMemoryAlreadyRegistered) return false; // someone else owns the pin; sync fallback for this region
    if (e != cudaSuccess) { cudaGetLastError(); return false; }   // clear and fall back
    return true;
}

static void multi_gpu_gemm_nslab(
        int nuse, cublasOperation_t ta, cublasOperation_t tb,
        int m, int n, int k,
        const real_t* hA, size_t aElems, int lda,
        const real_t* hB, size_t bElems, int ldb,
        real_t* hC)
{
    const real_t one = 1.0, zero = 0.0;
    std::vector<real_t*> dA(nuse, nullptr), dB(nuse, nullptr), dC(nuse, nullptr);
    std::vector<cublasHandle_t> hdl(nuse, nullptr);
    std::vector<int> lo(nuse, 0), len(nuse, 0);

    // Page-lock host operands once → cudaMemcpyAsync is truly async, per-device
    // PCIe lanes overlap.  hC is registered for the D2H half too.
    const size_t aBytes = aElems * sizeof(real_t);
    const size_t bBytes = bElems * sizeof(real_t);
    const size_t cBytes = (size_t)m * n * sizeof(real_t);
    const bool aPin = try_host_register(hA, aBytes);
    const bool bPin = try_host_register(hB, bBytes);
    const bool cPin = try_host_register(hC, cBytes);
    const cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
    const cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;

    // Loop 1: per device, async H2D + GEMM on the device's default stream.
    for (int d = 0; d < nuse; ++d) {
        auto rng = aux_partition((size_t)n, nuse, d);
        lo[d]  = (int)rng.first;
        len[d] = (int)(rng.second - rng.first);
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        cublasCreate(&hdl[d]);
        cudaMalloc(&dA[d], aBytes);
        cudaMalloc(&dB[d], bBytes);
        cudaMalloc(&dC[d], (size_t)m * len[d] * sizeof(real_t));
        // Async-if-pinned, else sync (CUDA falls back to sync for non-pinned host).
        if (aPin) cudaMemcpyAsync(dA[d], hA, aBytes, H2D, 0); else cudaMemcpy(dA[d], hA, aBytes, H2D);
        if (bPin) cudaMemcpyAsync(dB[d], hB, bBytes, H2D, 0); else cudaMemcpy(dB[d], hB, bBytes, H2D);
        const real_t* Bslab = (tb == CUBLAS_OP_N) ? dB[d] + (size_t)lo[d] * ldb
                                                   : dB[d] + (size_t)lo[d];
        cublasDgemm(hdl[d], ta, tb, m, len[d], k, &one,
                    dA[d], lda, Bslab, ldb, &zero, dC[d], m);
        // D2H async into hC's contiguous slab; per-device stream serializes vs
        // its own GEMM (so dC is ready), but runs concurrently with other devices.
        if (cPin) cudaMemcpyAsync(hC + (size_t)lo[d] * m, dC[d],
                                  (size_t)m * len[d] * sizeof(real_t), D2H, 0);
    }
    // Sync barrier per device + non-pinned D2H fallback.
    for (int d = 0; d < nuse; ++d) {
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        if (!cPin) cudaMemcpy(hC + (size_t)lo[d] * m, dC[d],
                              (size_t)m * len[d] * sizeof(real_t), D2H);
        else       cudaDeviceSynchronize();
    }
    // Free.
    for (int d = 0; d < nuse; ++d) {
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(dA[d]); cudaFree(dB[d]); cudaFree(dC[d]);
        cublasDestroy(hdl[d]);
    }
    if (aPin) cudaHostUnregister(const_cast<real_t*>(hA));
    if (bPin) cudaHostUnregister(const_cast<real_t*>(hB));
    if (cPin) cudaHostUnregister(hC);
}

// ------------------------------------------------------------------
//  Multi-term variant: C[m×n] (col-major, ldc=m) = Σ_t alpha_t · op(A_t)·op(B_t),
//  partitioning n across `nuse` devices.  Each device replicates EVERY term's
//  A_t and B_t, runs all term GEMMs serially on its own cuBLAS handle (β=0 on
//  term 0, β=1 thereafter), and writes a single contiguous column slab back to
//  hC.  This collapses the W1ovvo/Wovoo accumulation pattern (cublasDgemm
//  ×2 sharing one dC) into one helper call → bit-identical to the per-term
//  single-GPU sequence (same FP order within each term, β=1 accumulation
//  matches the legacy second GEMM).
//
//  Use sites: anchors where two GEMMs build the same C tensor in
//  build_dressed_intermediates / build_W_eff_and_G (W1ovvo, Wovoo ooov·t2,
//  UAMCI T2+T3, UAKEI T2+T3, UMLID, ...).  Single-term sites continue to call
//  multi_gpu_gemm_nslab unchanged.
struct MultiGpuGemmTerm {
    cublasOperation_t ta, tb;
    real_t            alpha;
    const real_t*     hA;
    size_t            aElems;
    int               lda;
    const real_t*     hB;
    size_t            bElems;
    int               ldb;
};

static void multi_gpu_gemm_nslab_terms(
        int nuse, int m, int n, int k,
        const std::vector<MultiGpuGemmTerm>& terms,
        real_t* hC)
{
    if (terms.empty()) return;
    const real_t zero = 0.0, one = 1.0;
    const size_t T = terms.size();
    std::vector<std::vector<real_t*>> dA(T, std::vector<real_t*>(nuse, nullptr));
    std::vector<std::vector<real_t*>> dB(T, std::vector<real_t*>(nuse, nullptr));
    std::vector<real_t*> dC(nuse, nullptr);
    std::vector<cublasHandle_t> hdl(nuse, nullptr);
    std::vector<int> lo(nuse, 0), len(nuse, 0);

    // Pin all term A/B hosts once + output hC (per-device PCIe lanes overlap
    // via cudaMemcpyAsync; falls back to sync cudaMemcpy where pin fails).
    std::vector<bool> aPin(T, false), bPin(T, false);
    for (size_t t = 0; t < T; ++t) {
        aPin[t] = try_host_register(terms[t].hA, terms[t].aElems * sizeof(real_t));
        bPin[t] = try_host_register(terms[t].hB, terms[t].bElems * sizeof(real_t));
    }
    const size_t cBytes = (size_t)m * n * sizeof(real_t);
    const bool cPin = try_host_register(hC, cBytes);
    const cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
    const cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;

    // Loop 1: per device, alloc dC + (alloc dA/dB + async H2D + GEMM) per term + async D2H.
    for (int d = 0; d < nuse; ++d) {
        auto rng = aux_partition((size_t)n, nuse, d);
        lo[d]  = (int)rng.first;
        len[d] = (int)(rng.second - rng.first);
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        cublasCreate(&hdl[d]);
        cudaMalloc(&dC[d], (size_t)m * len[d] * sizeof(real_t));
        for (size_t t = 0; t < T; ++t) {
            const auto& Tt = terms[t];
            const size_t aB = Tt.aElems * sizeof(real_t);
            const size_t bB = Tt.bElems * sizeof(real_t);
            cudaMalloc(&dA[t][d], aB);
            cudaMalloc(&dB[t][d], bB);
            if (aPin[t]) cudaMemcpyAsync(dA[t][d], Tt.hA, aB, H2D, 0);
            else         cudaMemcpy     (dA[t][d], Tt.hA, aB, H2D);
            if (bPin[t]) cudaMemcpyAsync(dB[t][d], Tt.hB, bB, H2D, 0);
            else         cudaMemcpy     (dB[t][d], Tt.hB, bB, H2D);
            const real_t* Bslab = (Tt.tb == CUBLAS_OP_N) ? dB[t][d] + (size_t)lo[d] * Tt.ldb
                                                         : dB[t][d] + (size_t)lo[d];
            const real_t beta = (t == 0) ? zero : one;
            cublasDgemm(hdl[d], Tt.ta, Tt.tb, m, len[d], k, &Tt.alpha,
                        dA[t][d], Tt.lda, Bslab, Tt.ldb, &beta, dC[d], m);
        }
        if (cPin) cudaMemcpyAsync(hC + (size_t)lo[d] * m, dC[d],
                                  (size_t)m * len[d] * sizeof(real_t), D2H, 0);
    }
    // Sync barrier per device + non-pinned D2H fallback.
    for (int d = 0; d < nuse; ++d) {
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        if (!cPin) cudaMemcpy(hC + (size_t)lo[d] * m, dC[d],
                              (size_t)m * len[d] * sizeof(real_t), D2H);
        else       cudaDeviceSynchronize();
    }
    // Free.
    for (int d = 0; d < nuse; ++d) {
        if (len[d] <= 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        for (size_t t = 0; t < T; ++t) {
            cudaFree(dA[t][d]); cudaFree(dB[t][d]);
        }
        cudaFree(dC[d]);
        cublasDestroy(hdl[d]);
    }
    for (size_t t = 0; t < T; ++t) {
        if (aPin[t]) cudaHostUnregister(const_cast<real_t*>(terms[t].hA));
        if (bPin[t]) cudaHostUnregister(const_cast<real_t*>(terms[t].hB));
    }
    if (cPin) cudaHostUnregister(hC);
}
} // anonymous namespace
#endif // GANSU_CPU_ONLY

// MO ERI block extraction kernels are shared with EE/IP/EA-EOM modules.
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_ccsd_extract_eri_ovvv_kernel(
    const real_t*, real_t*, int, int, int);

#ifndef GANSU_CPU_ONLY
__global__ void steom_build_diagonal_kernel(
    const real_t* __restrict__ eps_occ, const real_t* __restrict__ eps_vir,
    real_t* __restrict__ D, int nocc_active, int nvir)
{
    const int total = nocc_active * nvir;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int a = idx % nvir;
    int i = idx / nvir;
    D[idx] = eps_vir[a] - eps_occ[i];
}

__global__ void steom_diag_matvec_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = D[idx] * x[idx];
}

__global__ void steom_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}

// Dense (non-symmetric) matvec y = G x, G row-major [n×n] (row*n + col).
__global__ void steom_dense_matvec_kernel(
    const real_t* __restrict__ G, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        real_t s = 0.0;
        const real_t* Grow = G + (size_t)row * n;
        for (int c = 0; c < n; ++c) s += Grow[c] * x[c];
        y[row] = s;
    }
}

// Fused Wvvvo assembly: replaces the NV³·NO = 450M outer host scatter at anthracene
// (~56 s of build_dressed wall, memory-bound floor across 7 cache-unfriendly array
// reads per output). One thread per output (a,b,c,j). Reads 7 device arrays + writes
// d_Wvvvo. Memory traffic ≈ 8 × NV³·NO·8 bytes ≈ 29 GB; A100 HBM ≈ 1.5 TB/s → ~20 ms
// theoretical, ~200 ms with uncoalesced penalty. Replaces the host floor wholesale.
// Gated on canonical_skip_wvvvv_ ON + all 6 prerequisite GEMMs being device-resident
// (wvvvo_1/2/5_gpu + wvvvo_big_gpu + wvvvo_t5_gpu + wvvvo_w_t1 populated).
__global__ void steom_wvvvo_fused_assembly_kernel(
    const real_t* __restrict__ d_ovvv,    // h_ovvv layout [j,b,c,a] via H_OVVV macro
    const real_t* __restrict__ d_ct1,     // ct_wvvvo_1[b, (a,j,c)] (W1ovov·t1)
    const real_t* __restrict__ d_ct2,     // ct_wvvvo_2[a, (b,c,j)] = output layout
    const real_t* __restrict__ d_ct5,     // ct_wvvvo_5[c, (j,a,b)] (Fov·t2)
    const real_t* __restrict__ d_wbig,    // wvvvo_big[(a,b,c),j] = output layout (term3+4)
    const real_t* __restrict__ d_wt5,     // wvvvo_t5[(j,c),(b,a)] (term5)
    const real_t* __restrict__ d_wt1,     // wvvvo_w_t1[(a,b,c),j] = output layout (Wvvvv·t1, canonical-skip)
    real_t* __restrict__ d_Wvvvo,         // output [(a,b,c),j] = same as d_wbig/d_wt1
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = idx % NO;
        const int c = (idx / NO) % NV;
        const int b = (idx / ((size_t)NO * NV)) % NV;
        const int a = idx / ((size_t)NO * NV * NV);
        // bare OVVV(j,b,c,a) = h_ovvv[j*NV³ + b*NV² + c*NV + a]
        real_t v = d_ovvv[(((size_t)j * NV + b) * NV + c) * NV + a];
        // ct1[b*NV²·NO + a*NO·NV + j·NV + c] = ((b*NV + a)*NO + j)*NV + c
        v -= d_ct1[(((size_t)b * NV + a) * NO + j) * NV + c];
        // ct2 same layout as output → direct idx
        v -= d_ct2[idx];
        // wvvvo_big same layout → direct idx
        v += d_wbig[idx];
        // wt5[(j*NV+c)*NV² + b*NV + a] = ((j*NV + c)*NV + b)*NV + a
        v += d_wt5[(((size_t)j * NV + c) * NV + b) * NV + a];
        // ct5[c*NO·NV² + j·NV² + a·NV + b] = ((c*NO + j)*NV + a)*NV + b
        v -= d_ct5[(((size_t)c * NO + j) * NV + a) * NV + b];
        // wvvvo_w_t1 same layout → direct idx
        v += d_wt1[idx];
        d_Wvvvo[idx] = v;
    }
}

// ---- (RI Term A) Gather the virtual-virtual block of the half-transformed
//      B_mo (naux × nmo_full², row-major) into a contiguous [naux × nvir²]
//      buffer:  Bvv[P, a, b] = B_mo[P, voff+a, voff+b],  voff = nocc + frozen_off.
//      Mirror of ea_gather_bvv_kernel. Grid-stride, size_t safe.
__global__ void steom_gather_bvv_kernel(const real_t* __restrict__ B_mo,
                                        real_t* __restrict__ Bvv,
                                        int naux, int nmo_full, int voff, int nvir)
{
    const size_t total  = (size_t)naux * nvir * nvir;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int b = (int)(idx % nvir);
        const size_t t = idx / nvir;
        const int a = (int)(t % nvir);
        const int P = (int)(t / nvir);
        Bvv[idx] = B_mo[(size_t)P * nmo_full * nmo_full
                      + (size_t)(voff + a) * nmo_full + (size_t)(voff + b)];
    }
}

// ===========================================================================
//  Streaming-accumulate Wvvvo assembly — peak-memory alternative to the
//  fused 7-buffer kernel above.  Each kernel below folds ONE term into
//  d_Wvvvo in-place so the producing GEMM output can be freed immediately
//  before the next GEMM runs.  At tetracene cc-pVDZ (NV=270, NO=60) each
//  Wvvvo intermediate is NV³·NO·8B = 8.8 GiB; holding all 7 simultaneously
//  for the fused kernel needs 7·8.8 = 62 GB on device, on top of the
//  87 GB persistent operator state → 149 GB > H200 141 GB ceiling.
//  Streaming reduces the transient peak to ~28 GB (one output + d_Wvvvo +
//  GEMM operands).  Math is bit-identical to the fused kernel (same sums,
//  same operand order); only the launch granularity changes.
//
//  These kernels are reused by the device-resident path gated on
//  GANSU_STEOM_WVVVO_RESIDENT (default ON when canonical-skip enabled).
// ===========================================================================

// 1. Initialize d_Wvvvo[a,b,c,j] = d_ovvv[(j,b,c,a)] (the bare OVVV term).
//    d_ovvv is the operator's d_eri_ovvv_ persistent buffer — no upload.
__global__ void steom_wvvvo_init_from_ovvv_kernel(
    const real_t* __restrict__ d_ovvv,
    real_t* __restrict__ d_Wvvvo,
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = idx % NO;
        const int c = (idx / NO) % NV;
        const int b = (idx / ((size_t)NO * NV)) % NV;
        const int a = idx / ((size_t)NO * NV * NV);
        d_Wvvvo[idx] = d_ovvv[(((size_t)j * NV + b) * NV + c) * NV + a];
    }
}

// 2. Accumulate same-layout source: d_Wvvvo[idx] += coeff * d_src[idx].
//    Handles ct2 (coeff=-1), wbig (+1), wt1 (+1).
__global__ void steom_wvvvo_accum_same_layout_kernel(
    const real_t* __restrict__ d_src,
    real_t* __restrict__ d_Wvvvo,
    real_t coeff,
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        d_Wvvvo[idx] += coeff * d_src[idx];
    }
}

// 3. Accumulate ct_wvvvo_1[b, (a,j,c)] with coeff=-1 (W1ovov·t1 term).
//    d_Wvvvo[a,b,c,j] -= d_ct1[((b*NV + a)*NO + j)*NV + c].
__global__ void steom_wvvvo_accum_minus_ct1_kernel(
    const real_t* __restrict__ d_ct1,
    real_t* __restrict__ d_Wvvvo,
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = idx % NO;
        const int c = (idx / NO) % NV;
        const int b = (idx / ((size_t)NO * NV)) % NV;
        const int a = idx / ((size_t)NO * NV * NV);
        d_Wvvvo[idx] -= d_ct1[(((size_t)b * NV + a) * NO + j) * NV + c];
    }
}

// 4. Accumulate wvvvo_t5[(j,c), (b,a)] with coeff=+1 (term5 ooov·tau2).
//    d_Wvvvo[a,b,c,j] += d_wt5[((j*NV + c)*NV + b)*NV + a].
__global__ void steom_wvvvo_accum_plus_wt5_kernel(
    const real_t* __restrict__ d_wt5,
    real_t* __restrict__ d_Wvvvo,
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = idx % NO;
        const int c = (idx / NO) % NV;
        const int b = (idx / ((size_t)NO * NV)) % NV;
        const int a = idx / ((size_t)NO * NV * NV);
        d_Wvvvo[idx] += d_wt5[(((size_t)j * NV + c) * NV + b) * NV + a];
    }
}

// 5. Accumulate ct_wvvvo_5[c, (j,a,b)] with coeff=-1 (Fov·t2 term).
//    d_Wvvvo[a,b,c,j] -= d_ct5[((c*NO + j)*NV + a)*NV + b].
__global__ void steom_wvvvo_accum_minus_ct5_kernel(
    const real_t* __restrict__ d_ct5,
    real_t* __restrict__ d_Wvvvo,
    int NO, int NV)
{
    const size_t total = (size_t)NV * NV * NV * NO;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = idx % NO;
        const int c = (idx / NO) % NV;
        const int b = (idx / ((size_t)NO * NV)) % NV;
        const int a = idx / ((size_t)NO * NV * NV);
        d_Wvvvo[idx] -= d_ct5[(((size_t)c * NO + j) * NV + a) * NV + b];
    }
}
#endif  // !GANSU_CPU_ONLY


// ==================================================================
//  Constructor / destructor
// ==================================================================

STEOMCCSDOperator::STEOMCCSDOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    real_t* d_t1, real_t* d_t2,
    const real_t* h_R2_IP_amplitudes,
    const real_t* h_R2_EA_amplitudes,
    const real_t* h_R1_IP_amplitudes,
    const real_t* h_R1_EA_amplitudes,
    const int* h_active_occ_idx,
    const int* h_active_vir_idx,
    int nocc_active, int nvir, int nao_active,
    int n_act_occ, int n_act_vir,
    const ERI_RI* eri_block_src,
    const real_t* d_B_mo_blocks,
    int nmo_full,
    std::vector<real_t*>* d_eri_vvvv_slabs_input,
    SteomBarHCache* barh_cache,
    int frozen_off)
    : nocc_active_(nocc_active), nvir_(nvir), nao_active_(nao_active),
      n_act_occ_(n_act_occ), n_act_vir_(n_act_vir),
      total_dim_(nocc_active * nvir),
      d_t1_(d_t1), d_t2_(d_t2),
      eri_block_src_(eri_block_src), d_B_mo_blocks_(d_B_mo_blocks), nmo_full_(nmo_full),
      frozen_off_(frozen_off),
      barh_cache_(barh_cache)
{
    // Ship 14 — take ownership of per-device d_eri_vvvv slabs allocated +
    // extracted by the driver (compute_steom_ccsd_impl).  Slab boundaries
    // are uniform along the outermost a-axis; the consumer GEMM
    // (canonical-skip Term A) runs independently on each device against its
    // own slab + a broadcast t1.  Mirror of EA Ship 12.
    if (d_eri_vvvv_slabs_input && (int)d_eri_vvvv_slabs_input->size() >= 2) {
        const int N = (int)d_eri_vvvv_slabs_input->size();
        eri_vvvv_nslab_ = N;
        d_eri_vvvv_slabs_ = std::move(*d_eri_vvvv_slabs_input);
        a_starts_.assign(N, 0);
        a_ends_.assign(N, 0);
        for (int d = 0; d < N; ++d) {
            a_starts_[d] = (int)((int64_t)d * nvir / N);
            a_ends_[d]   = (int)((int64_t)(d + 1) * nvir / N);
        }
        std::cout << "  [STEOM Ship 14] d_eri_vvvv slab mode ON: N=" << N
                  << " devices, a-axis split [";
        for (int d = 0; d < N; ++d)
            std::cout << a_starts_[d] << "-" << a_ends_[d]
                      << (d + 1 < N ? "," : "]");
        const double per_dev_gb = (double)(a_ends_[0] - a_starts_[0])
                                  * nvir * nvir * nvir * sizeof(real_t)
                                  / (1024.0 * 1024.0 * 1024.0);
        std::cout << ", slab[0] ≈ " << std::fixed << std::setprecision(2)
                  << per_dev_gb << " GB per device" << std::defaultfloat
                  << std::endl;
    }
    if (nocc_active <= 0 || nvir <= 0 || nao_active != nocc_active + nvir) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: invalid (nocc_active, nvir, nao_active) — require "
            "nao_active == nocc_active + nvir, both positive");
    }
    if (n_act_occ <= 0 || n_act_vir <= 0) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: n_act_occ and n_act_vir must both be positive — "
            "did P1 IP-EOM and P2 EA-EOM run their active-mode routing?");
    }
    if (h_R2_IP_amplitudes == nullptr || h_R2_EA_amplitudes == nullptr) {
        throw std::invalid_argument(
            "STEOMCCSDOperator: R2 amplitude inputs must be non-null host pointers");
    }

    // Copy raw R2 amplitudes from host to device.
    const size_t r2_ip_sz = (size_t)n_act_occ * nocc_active * nocc_active * nvir;
    const size_t r2_ea_sz = (size_t)n_act_vir * nocc_active * nvir * nvir;
    tracked_cudaMalloc(&d_R2_IP_, r2_ip_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_R2_EA_, r2_ea_sz * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < r2_ip_sz; ++i) d_R2_IP_[i] = h_R2_IP_amplitudes[i];
        for (size_t i = 0; i < r2_ea_sz; ++i) d_R2_EA_[i] = h_R2_EA_amplitudes[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_R2_IP_, h_R2_IP_amplitudes, r2_ip_sz * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_R2_EA_, h_R2_EA_amplitudes, r2_ea_sz * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    build_diagonal(d_orbital_energies);

    // Sub-phase 3.4: full pipeline — bar-H + X(MI) + F^eff_oo.
    // Phase 0: eri_block_src_ provides the ERI blocks on the fly (d_eri_mo == nullptr).
    if (d_eri_mo != nullptr || eri_block_src_ != nullptr) {
        if (d_t1 == nullptr || d_t2 == nullptr) {
            throw std::invalid_argument(
                "STEOMCCSDOperator: d_eri_mo non-null requires both d_t1 and d_t2 non-null");
        }
        if (h_R1_IP_amplitudes == nullptr || h_R1_EA_amplitudes == nullptr ||
            h_active_occ_idx == nullptr || h_active_vir_idx == nullptr) {
            throw std::invalid_argument(
                "STEOMCCSDOperator: sub-phase 3.4 F^eff_oo build requires non-null "
                "h_R1_IP_amplitudes, h_R1_EA_amplitudes, h_active_occ_idx, h_active_vir_idx");
        }
        // Store active NTO ↔ MO index maps (host-side, used by build_F_eff_oo).
        active_occ_idx_.assign(h_active_occ_idx, h_active_occ_idx + n_act_occ);
        active_vir_idx_.assign(h_active_vir_idx, h_active_vir_idx + n_act_vir);
        for (int m = 0; m < n_act_occ; ++m) {
            if (active_occ_idx_[m] < 0 || active_occ_idx_[m] >= nocc_active) {
                throw std::invalid_argument(
                    "STEOMCCSDOperator: active_occ_idx out of range");
            }
        }
        for (int e = 0; e < n_act_vir; ++e) {
            if (active_vir_idx_[e] < 0 || active_vir_idx_[e] >= nvir) {
                throw std::invalid_argument(
                    "STEOMCCSDOperator: active_vir_idx out of range");
            }
        }
        // Upload R1 amplitudes to device.
        const size_t r1_ip_sz = (size_t)n_act_occ * nocc_active;
        const size_t r1_ea_sz = (size_t)n_act_vir * nvir;
        tracked_cudaMalloc(&d_R1_IP_, r1_ip_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_R1_EA_, r1_ea_sz * sizeof(real_t));
        if (!gpu::gpu_available()) {
            for (size_t i = 0; i < r1_ip_sz; ++i) d_R1_IP_[i] = h_R1_IP_amplitudes[i];
            for (size_t i = 0; i < r1_ea_sz; ++i) d_R1_EA_[i] = h_R1_EA_amplitudes[i];
        } else {
#ifndef GANSU_CPU_ONLY
            cudaMemcpy(d_R1_IP_, h_R1_IP_amplitudes, r1_ip_sz * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_R1_EA_, h_R1_EA_amplitudes, r1_ea_sz * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
        }

        // Need Fock diagonals from orbital energies (mirrors P1/P2 pattern).
        tracked_cudaMalloc(&d_f_oo_, (size_t)nocc_active_ * sizeof(real_t));
        tracked_cudaMalloc(&d_f_vv_, (size_t)nvir_        * sizeof(real_t));
        if (!gpu::gpu_available()) {
            for (int i = 0; i < nocc_active_; ++i) d_f_oo_[i] = d_orbital_energies[i];
            for (int a = 0; a < nvir_; ++a)         d_f_vv_[a] = d_orbital_energies[a + nocc_active_];
        } else {
#ifndef GANSU_CPU_ONLY
            cudaMemcpy(d_f_oo_, d_orbital_energies,                 (size_t)nocc_active_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_f_vv_, d_orbital_energies + nocc_active_,  (size_t)nvir_        * sizeof(real_t), cudaMemcpyDeviceToDevice);
#endif
        }
        // Per-phase build profiling (env GANSU_STEOM_BUILD_PROF=1). Sync + chrono around
        // each ctor stage to split the "Operator build time" (diagnostic, once).
        const char* _stp = std::getenv("GANSU_PROGRESS");   // progress default-on; GANSU_PROGRESS=0 to quiet
        const bool prof = (std::getenv("GANSU_STEOM_BUILD_PROF") != nullptr)
                        || !_stp || _stp[0] != '0';

        // Stage 1: build-phase multi-GPU device count (env GANSU_STEOM_BUILD_GPUS=N>1).
        // Decoupled from --num_gpus / GANSU_STEOM_EOM_GPUS. Clamp to the physical
        // device count. 1 → legacy single-GPU build (byte-identical).
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            if (const char* bg = std::getenv("GANSU_STEOM_BUILD_GPUS")) {
                int req = std::atoi(bg);
                int phys = 0; cudaGetDeviceCount(&phys);
                build_gpus_ = (req > 1 && phys > 1) ? std::min(req, phys) : 1;
                if (build_gpus_ > 1)
                    std::cout << "[STEOM build] multi-GPU GEMM: build_gpus=" << build_gpus_
                              << " (n-slab across devices, bit-identical)" << std::endl;
            }
        }
#endif

        // P5 canonical-skip Wvvvv (3-AND gate, mirror of the EA canonical-skip
        // at the same env name). Default off → bit-exact, skips ~nvir⁴·8B host
        // (h_Wvvvv) + ~nvir⁴·8B device (d_Wvvvv_) when the native per-pair σ
        // takes over the dressed-W reads (only NATIVE_EOM+NATIVE_BARE supplies
        // the per-pair dressed PNO substitute, so we gate behind that).
        {
            // Master-switch (2026-06-03, mirror of EA): NATIVE_EOM is the explicit
            // gate (default OFF — built on the reference path too); NATIVE_BARE and
            // CANONICAL_SKIP default ON under it. NATIVE_EOM=1 alone enables skip.
            auto on = [](const char* n, bool d) {
                const char* e = std::getenv(n);
                return (!e || !e[0]) ? d : (e[0] != '0');
            };
            canonical_skip_wvvvv_ = on("GANSU_DLPNO_NATIVE_EOM", false) &&
                                    on("GANSU_DLPNO_NATIVE_BARE", true) &&
                                    on("GANSU_DLPNO_CANONICAL_SKIP", true);
            if (canonical_skip_wvvvv_)
                std::cout << "  [STEOM canonical-skip] dressed Wvvvv build SKIPPED "
                             "(nvir⁴ host+device elided; Wvvvo·t1 refactored)" << std::endl;
            // (RI Term A) mirror of EA: evaluate Wvvvo·t1 from RI B-factors so the
            // nvir⁴ (ab|cd) block is never materialised (also keeps share-barH on
            // at scale by removing the auto device-balancing trigger).
            ri_vvvv_term_a_ = canonical_skip_wvvvv_ && (eri_block_src_ != nullptr)
                              && on("GANSU_DLPNO_EA_VVVV_RI", true);
            if (ri_vvvv_term_a_)
                std::cout << "  [STEOM RI-Term-A] Wvvvo·t1 via RI B-factors "
                             "(d_eri_vvvv nvir⁴ not materialised)" << std::endl;
        }
        // Ship 14: vvvv slab mode requires canonical-skip ON (legacy path D2Hs
        // d_eri_vvvv_ which is null in slab mode, and reads h_vvvv which is
        // empty).  Force the flag and warn if env disagrees.
        if (eri_vvvv_nslab_ > 1 && !canonical_skip_wvvvv_) {
            canonical_skip_wvvvv_ = true;
            std::cout << "  [STEOM Ship 14] forcing canonical_skip_wvvvv_=true "
                         "(slab mode requires skip path; set GANSU_DLPNO_CANONICAL_SKIP=1 "
                         "to silence this notice)" << std::endl;
        }
        auto tphase = [&](const char* name, auto&& fn) {
            if (!prof) { fn(); return; }
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
            const auto t0 = std::chrono::high_resolution_clock::now();
            fn();
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
            const double s = std::chrono::duration<double>(
                                 std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << "  [STEOM build-PROF] " << name << " = " << std::fixed
                      << std::setprecision(3) << s << " s" << std::defaultfloat << std::endl;
        };
        tphase("extract_eri_blocks",        [&]{ extract_eri_blocks(d_eri_mo); });
        tphase("build_dressed_intermediates", [&]{ build_dressed_intermediates(); });

        // Sub-phase 3.4: build X(MI), X(EA) then F^eff_oo + F^eff_vv per
        // CFOUR `gmi_steom_rhf` / `gea_steom_rhf`.
        tphase("build_x_matrices",          [&]{ build_x_matrices(h_R1_IP_amplitudes, h_R1_EA_amplitudes); });
        tphase("build_F_eff_oo",            [&]{ build_F_eff_oo(); });
        tphase("build_F_eff_vv",            [&]{ build_F_eff_vv(); });
        // Sub-phase 3.5-3.7: full W^eff dressing + dense G^{1h1p}.
        // Ship 15 — secondary device-balance for build_W_eff_and_G.  The
        // W^eff dressing phase allocates 3 transient GEMM operands of shape
        // NV²·NO·NV·8B ≈ 20 GB each (UAMCI, UAKEI, UMLID — see line 3699+,
        // 3963+, 4068+), plus the persistent d_G_ = total_dim²·8B (~4.5 GB
        // at Pentacene).  By this point the Ship 11 target device holds
        // ~90 GB of bar-H intermediates (d_eri_*, d_W_*, d_M_ring*) and can't
        // afford a single 20 GB transient, never mind 3 of them concurrently.
        // We pick the device with maximum driver-free memory (typically a
        // peer that only holds the d_eri_vvvv slab + DLPNO state ≈ 30 GB)
        // and redirect: all subsequent allocs in build_W_eff_and_G + d_G_
        // land there.  Downstream Davidson/geev solve uses d_G_ on this same
        // device (no second restore — Ship 11's impl-exit restore handles
        // the final device cleanup; tracked_cudaFree is cross-device safe).
        tphase("build_W_eff_and_G", [&]{
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                const char* env_bal = std::getenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
                if (env_bal && env_bal[0] == '1') {
                    int n_dev = 0;
                    cudaGetDeviceCount(&n_dev);
                    if (n_dev > 1) {
                        int saved = 0;
                        cudaGetDevice(&saved);
                        std::vector<size_t> per_dev_free(n_dev, 0);
                        for (int d = 0; d < n_dev; ++d) {
                            cudaSetDevice(d);
                            size_t free_b = 0, total_b = 0;
                            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
                                per_dev_free[d] = free_b;
                        }
                        int best_dev = saved;
                        size_t best_free = per_dev_free[saved];
                        for (int d = 0; d < n_dev; ++d) {
                            if (per_dev_free[d] > best_free) {
                                best_free = per_dev_free[d];
                                best_dev  = d;
                            }
                        }
                        if (best_dev != saved) {
                            cudaSetDevice(best_dev);
                            // Rebind thread_local GPUHandle (cuBLAS) so GEMMs
                            // run on the new device with handle bound there.
                            gpu::GPUHandle::reset();
                            std::cout << "  [STEOM Ship 15 W_eff-balance] redirecting "
                                         "build_W_eff_and_G + d_G_ to GPU " << best_dev
                                      << " (free=" << std::fixed << std::setprecision(2)
                                      << (best_free / (1024.0*1024.0*1024.0)) << " GB"
                                      << ", bar-H state stays on GPU " << saved
                                      << " with free=" << (per_dev_free[saved] / (1024.0*1024.0*1024.0))
                                      << " GB; GPUHandle rebuilt)"
                                      << std::defaultfloat << std::endl;
                        } else {
                            cudaSetDevice(saved);
                        }
                    }
                }
            }
#endif
            build_W_eff_and_G();
            // No restore here — downstream Davidson/geev solve uses d_G_ on
            // the new device.  Ship 11's impl-exit restore handles final
            // device cleanup.
        });
    }
}

STEOMCCSDOperator::~STEOMCCSDOperator() {
    if (d_G_)         tracked_cudaFree(d_G_);
    if (d_R2_IP_)     tracked_cudaFree(d_R2_IP_);
    if (d_R2_EA_)     tracked_cudaFree(d_R2_EA_);
    if (d_R1_IP_)     tracked_cudaFree(d_R1_IP_);
    if (d_R1_EA_)     tracked_cudaFree(d_R1_EA_);
    if (d_X_IP_)      tracked_cudaFree(d_X_IP_);
    if (d_X_EA_)      tracked_cudaFree(d_X_EA_);
    if (d_F_eff_oo_)  tracked_cudaFree(d_F_eff_oo_);
    if (d_U_MI_)      tracked_cudaFree(d_U_MI_);
    if (d_F_eff_vv_)  tracked_cudaFree(d_F_eff_vv_);
    if (d_U_EA_)      tracked_cudaFree(d_U_EA_);
    if (d_t1_)        tracked_cudaFree(d_t1_);
    if (d_t2_)        tracked_cudaFree(d_t2_);
    if (d_diagonal_)  tracked_cudaFree(d_diagonal_);
    if (d_eps_occ_)   tracked_cudaFree(d_eps_occ_);
    if (d_eps_vir_)   tracked_cudaFree(d_eps_vir_);
    if (d_f_oo_)      tracked_cudaFree(d_f_oo_);
    if (d_f_vv_)      tracked_cudaFree(d_f_vv_);
    if (d_eri_oooo_)  tracked_cudaFree(d_eri_oooo_);
    if (d_eri_ooov_)  tracked_cudaFree(d_eri_ooov_);
    if (d_eri_oovv_)  tracked_cudaFree(d_eri_oovv_);
    if (d_eri_ovov_)  tracked_cudaFree(d_eri_ovov_);
    if (d_eri_ovvo_)  tracked_cudaFree(d_eri_ovvo_);
    if (d_eri_ovvv_)  tracked_cudaFree(d_eri_ovvv_);
    if (d_eri_vvvv_)  tracked_cudaFree(d_eri_vvvv_);
    // Ship 14: free per-device d_eri_vvvv slabs (current device may not match
    // the alloc device → wrap each free in cudaSetDevice).
#ifndef GANSU_CPU_ONLY
    if (!d_eri_vvvv_slabs_.empty()) {
        int saved = 0;
        cudaGetDevice(&saved);
        for (size_t d = 0; d < d_eri_vvvv_slabs_.size(); ++d) {
            if (d_eri_vvvv_slabs_[d]) {
                cudaSetDevice((int)d);
                tracked_cudaFree(d_eri_vvvv_slabs_[d]);
                d_eri_vvvv_slabs_[d] = nullptr;
            }
        }
        cudaSetDevice(saved);
    }
#endif
    // (A) shared bar-H: when borrowed, the 11 d_* alias the cache's buffers
    // (owned + freed by the STEOM driver after this operator is destroyed).
    // Skip here to avoid double-free / use-after-free.
    if (!barh_borrowed_) {
        if (d_Loo_)       tracked_cudaFree(d_Loo_);
        if (d_Lvv_)       tracked_cudaFree(d_Lvv_);
        if (d_Fov_)       tracked_cudaFree(d_Fov_);
        if (d_Woooo_)     tracked_cudaFree(d_Woooo_);
        if (d_Wooov_)     tracked_cudaFree(d_Wooov_);
        if (d_Wovov_)     tracked_cudaFree(d_Wovov_);
        if (d_Wovvo_)     tracked_cudaFree(d_Wovvo_);
        if (d_Wovoo_)     tracked_cudaFree(d_Wovoo_);
        if (d_Wvovv_)     tracked_cudaFree(d_Wvovv_);
        if (d_Wvvvv_)     tracked_cudaFree(d_Wvvvv_);
        if (d_Wvvvo_)     tracked_cudaFree(d_Wvvvo_);
    }
}

void STEOMCCSDOperator::build_diagonal(const real_t* d_orbital_energies) {
    tracked_cudaMalloc(&d_eps_occ_,  (size_t)nocc_active_ * sizeof(real_t));
    tracked_cudaMalloc(&d_eps_vir_,  (size_t)nvir_        * sizeof(real_t));
    tracked_cudaMalloc(&d_diagonal_, (size_t)total_dim_   * sizeof(real_t));

    if (!gpu::gpu_available()) {
        for (int i = 0; i < nocc_active_; ++i) d_eps_occ_[i] = d_orbital_energies[i];
        for (int a = 0; a < nvir_; ++a)         d_eps_vir_[a] = d_orbital_energies[a + nocc_active_];
        for (int i = 0; i < nocc_active_; ++i)
            for (int a = 0; a < nvir_; ++a)
                d_diagonal_[i * nvir_ + a] = d_eps_vir_[a] - d_eps_occ_[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_eps_occ_, d_orbital_energies,                (size_t)nocc_active_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_eps_vir_, d_orbital_energies + nocc_active_, (size_t)nvir_        * sizeof(real_t), cudaMemcpyDeviceToDevice);
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        steom_build_diagonal_kernel<<<blocks, threads>>>(d_eps_occ_, d_eps_vir_, d_diagonal_, nocc_active_, nvir_);
        cudaDeviceSynchronize();
#endif
    }
}


// ==================================================================
//  apply / apply_preconditioner — diagonal-only stubs (sub-phase 3.0+3.1)
// ==================================================================
void STEOMCCSDOperator::apply(const real_t* d_input, real_t* d_output) const {
    // Sub-phase 3.5-3.7: dense G^{1h1p} matvec when built; else diagonal stub.
    if (d_G_ != nullptr) {
        if (!gpu::gpu_available()) {
            #pragma omp parallel for
            for (int row = 0; row < total_dim_; ++row) {
                real_t s = 0.0;
                const real_t* Grow = d_G_ + (size_t)row * total_dim_;
                for (int c = 0; c < total_dim_; ++c) s += Grow[c] * d_input[c];
                d_output[row] = s;
            }
        } else {
#ifndef GANSU_CPU_ONLY
            const int threads = 256;
            const int blocks  = (total_dim_ + threads - 1) / threads;
            steom_dense_matvec_kernel<<<blocks, threads>>>(d_G_, d_input, d_output, total_dim_);
#endif
        }
        return;
    }
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; ++idx) {
            d_output[idx] = d_diagonal_[idx] * d_input[idx];
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        steom_diag_matvec_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

void STEOMCCSDOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < total_dim_; ++idx) {
            real_t d = d_diagonal_[idx];
            d_output[idx] = (std::fabs(d) > 1e-12) ? (d_input[idx] / d) : real_t(0.0);
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        const int blocks  = (total_dim_ + threads - 1) / threads;
        steom_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}


// ==================================================================
//  extract_eri_blocks — union of IP + EA needs (7 blocks)
//  Literal copy of P2's extract_eri_blocks (which already includes the
//  IP-required blocks plus vvvv).
// ==================================================================
void STEOMCCSDOperator::extract_eri_blocks(const real_t* d_eri_mo) {
    int nocc = nocc_active_, nvir = nvir_, nao = nao_active_;
    size_t nao2 = (size_t)nao * nao;
    size_t N    = (size_t)nao;

    const size_t ovov_sz = (size_t)nocc * nvir * nocc * nvir;
    const size_t ooov_sz = (size_t)nocc * nocc * nocc * nvir;
    const size_t oooo_sz = (size_t)nocc * nocc * nocc * nocc;
    const size_t oovv_sz = (size_t)nocc * nocc * nvir * nvir;
    const size_t ovvo_sz = (size_t)nocc * nvir * nvir * nocc;
    const size_t ovvv_sz = (size_t)nocc * nvir * nvir * nvir;
    const size_t vvvv_sz = (size_t)nvir * nvir * nvir * nvir;

    tracked_cudaMalloc(&d_eri_oooo_, oooo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ooov_, ooov_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovov_, ovov_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_oovv_, oovv_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovvo_, ovvo_sz * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_ovvv_, ovvv_sz * sizeof(real_t));
    // Ship 14: in slab mode d_eri_vvvv_ stays null; each device owns its
    // slab via d_eri_vvvv_slabs_[d] allocated by the driver before ctor.
    // (RI Term A) skip the nvir⁴ alloc when Wvvvo·t1 is evaluated from B-factors
    // (keep it under BUILD_VALIDATE so the host self-check reference survives).
    const bool keep_dense_vvvv = !ri_vvvv_term_a_
                                 || std::getenv("GANSU_STEOM_BUILD_VALIDATE") != nullptr;
    if (eri_vvvv_nslab_ <= 1 && keep_dense_vvvv) {
        tracked_cudaMalloc(&d_eri_vvvv_, vvvv_sz * sizeof(real_t));
    }

#ifndef GANSU_CPU_ONLY
    // Phase 0: build each block on the fly from B_mo (naux×nmo²) — never the
    // full nmo⁴ tensor. o=[0,nocc), v=[nocc,nmo). mo_eri_block_into writes
    // block[(p,q),(r,s)] row-major = (p q | r s), matching the gather layouts
    // below. Only the single-GPU RI path sets eri_block_src_ (num_frozen==0, so
    // nao==nmo_full_).
    if (eri_block_src_ != nullptr) {
        const int M = nmo_full_;
        // Frozen core: B_mo spans the full C (M MOs); shift every range start by
        // frozen_off_ to read the active occ [O,O+nocc) / vir [O+nocc,O+nocc+nvir)
        // window. O = 0 ⇒ non-frozen (byte-identical).
        const int O = frozen_off_;
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,O,nocc,           d_eri_oooo_); // (ij|kl)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         O,nocc,nocc+O,nvir,      d_eri_ooov_); // (ji|kb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    O,nocc,nocc+O,nvir,      d_eri_ovov_); // (ia|jb)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,O,nocc,         nocc+O,nvir,nocc+O,nvir, d_eri_oovv_); // (ij|ab)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,O,nocc,      d_eri_ovvo_); // (ia|bj)
        eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, O,nocc,nocc+O,nvir,    nocc+O,nvir,nocc+O,nvir, d_eri_ovvv_); // (ia|bc)
        // Ship 14: slab mode skips legacy single-device vvvv extract (driver
        // pre-populated d_eri_vvvv_slabs_).
        if (eri_vvvv_nslab_ <= 1 && keep_dense_vvvv) {
            eri_block_src_->mo_eri_block_into(d_B_mo_blocks_, M, nocc+O,nvir,nocc+O,nvir,nocc+O,nvir,nocc+O,nvir,d_eri_vvvv_); // (ab|cd)
        }
        cudaDeviceSynchronize();
        return;
    }
#endif

    if (!gpu::gpu_available()) {
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovov_sz; ++idx) {
            int i = idx / (nvir * nocc * nvir);
            int rem = idx % (nvir * nocc * nvir);
            int a = rem / (nocc * nvir); rem %= (nocc * nvir);
            int j = rem / nvir; int b = rem % nvir;
            d_eri_ovov_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)j*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ooov_sz; ++idx) {
            int j = idx / (nocc * nocc * nvir);
            int rem = idx % (nocc * nocc * nvir);
            int i = rem / (nocc * nvir); rem %= (nocc * nvir);
            int k = rem / nvir; int b = rem % nvir;
            d_eri_ooov_[idx] = d_eri_mo[((size_t)j*nao + i)*nao2 + (size_t)k*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oooo_sz; ++idx) {
            int i = idx / (nocc * nocc * nocc);
            int rem = idx % (nocc * nocc * nocc);
            int j = rem / (nocc * nocc); rem %= (nocc * nocc);
            int k = rem / nocc; int l = rem % nocc;
            d_eri_oooo_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)k*nao + l];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)oovv_sz; ++idx) {
            int i = idx / (nocc * nvir * nvir);
            int rem = idx % (nocc * nvir * nvir);
            int j = rem / (nvir * nvir); rem %= (nvir * nvir);
            int a = rem / nvir; int b = rem % nvir;
            d_eri_oovv_[idx] = d_eri_mo[((size_t)i*nao + j)*nao2 + (size_t)(a+nocc)*nao + b+nocc];
        }
        #pragma omp parallel for
        for (int idx = 0; idx < (int)ovvo_sz; ++idx) {
            int i = idx / (nvir * nvir * nocc);
            int rem = idx % (nvir * nvir * nocc);
            int a = rem / (nvir * nocc); rem %= (nvir * nocc);
            int b = rem / nocc; int j = rem % nocc;
            d_eri_ovvo_[idx] = d_eri_mo[((size_t)i*nao + a+nocc)*nao2 + (size_t)(b+nocc)*nao + j];
        }
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovvv_sz; ++idx) {
            int i = (int)(idx / ((size_t)nvir * nvir * nvir));
            size_t rem2 = idx % ((size_t)nvir * nvir * nvir);
            int a = (int)(rem2 / ((size_t)nvir * nvir));
            rem2 %= ((size_t)nvir * nvir);
            int b = (int)(rem2 / nvir);
            int c = (int)(rem2 % nvir);
            size_t mo_idx = (size_t)i * N * N * N + (size_t)(a + nocc) * N * N
                          + (size_t)(b + nocc) * N + (size_t)(c + nocc);
            d_eri_ovvv_[idx] = d_eri_mo[mo_idx];
        }
        #pragma omp parallel for
        for (size_t idx = 0; idx < vvvv_sz; ++idx) {
            int a = (int)(idx / ((size_t)nvir * nvir * nvir));
            size_t rem2 = idx % ((size_t)nvir * nvir * nvir);
            int b = (int)(rem2 / ((size_t)nvir * nvir));
            rem2 %= ((size_t)nvir * nvir);
            int c = (int)(rem2 / nvir);
            int d = (int)(rem2 % nvir);
            size_t mo_idx = (size_t)(a + nocc) * N * N * N + (size_t)(b + nocc) * N * N
                          + (size_t)(c + nocc) * N + (size_t)(d + nocc);
            d_eri_vvvv_[idx] = d_eri_mo[mo_idx];
        }
    } else {
#ifndef GANSU_CPU_ONLY
        const int threads = 256;
        int blocks;
        blocks = (ovov_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov_, nocc_active_, nvir_, nao_active_);
        blocks = (ooov_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov_, nocc_active_, nvir_, nao_active_);
        blocks = (oooo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo_, nocc_active_, nao_active_, 0);
        blocks = (oovv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv_, nocc_active_, nvir_, nao_active_, 0, -1);
        blocks = (ovvo_sz + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo_, nocc_active_, nvir_, nao_active_);
        blocks = (ovvv_sz + threads - 1) / threads;
        eom_ccsd_extract_eri_ovvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvv_, nocc_active_, nvir_, nao_active_);
        blocks = (vvvv_sz + threads - 1) / threads;
        eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv_, nocc_active_, nvir_, nao_active_, 0, -1);
        cudaDeviceSynchronize();
#endif
    }
}


// ==================================================================
//  build_dressed_intermediates — union of IP + EA versions
//  (literal copy from src/ip_eom_ccsd_operator.cu + src/ea_eom_ccsd_operator.cu;
//  the 11 intermediates Loo, Lvv, Fov, Woooo, Wooov, Wovov, Wovvo, Wovoo,
//  Wvovv, Wvvvv, Wvvvo are all required by the STEOM G^{1h1p} matvec —
//  sub-phases 3.4-3.7).
// ==================================================================
#define H_OVOV(p,a,q,b) h_ovov[(size_t)(p)*NV*NO*NV + (size_t)(a)*NO*NV + (size_t)(q)*NV + (b)]
#define H_OOOV(p,q,r,a) h_ooov[(size_t)(p)*NO*NO*NV + (size_t)(q)*NO*NV + (size_t)(r)*NV + (a)]
#define H_OOVV(p,q,a,b) h_oovv[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]
#define H_OVVO(p,a,b,q) h_ovvo[(size_t)(p)*NV*NV*NO + (size_t)(a)*NV*NO + (size_t)(b)*NO + (q)]
#define H_OVVV(p,a,b,c) h_ovvv[(size_t)(p)*NV*NV*NV + (size_t)(a)*NV*NV + (size_t)(b)*NV + (c)]
#define H_OOOO(p,q,r,s) h_oooo[(size_t)(p)*NO*NO*NO + (size_t)(q)*NO*NO + (size_t)(r)*NO + (s)]
#define H_VVVV(a,b,c,d) h_vvvv[(size_t)(a)*NV*NV*NV + (size_t)(b)*NV*NV + (size_t)(c)*NV + (d)]
#define H_T1(p,a)       h_t1[(p)*NV + (a)]
#define H_T2(p,q,a,b)   h_t2[(size_t)(p)*NO*NV*NV + (size_t)(q)*NV*NV + (size_t)(a)*NV + (b)]

void STEOMCCSDOperator::build_dressed_intermediates() {
    // (A) shared bar-H: borrow all 11 dressed intermediates from the cache
    // (published by IP's 8 + EA's 3) and SKIP the build entirely. All are
    // bit-identical across IP/EA/STEOM. Fail-safe: any dims / canonical-skip
    // mismatch falls through to a full build. d_f_oo_/d_f_vv_ are already
    // allocated (ctor) and simply go unused; downstream build_F_eff_*/
    // build_W_eff_and_G read the borrowed bar-H, not the raw Fock diagonals.
    if (barh_cache_ && barh_cache_->complete()
        && barh_cache_->nocc == nocc_active_ && barh_cache_->nvir == nvir_
        && barh_cache_->canonical_skip_wvvvv == canonical_skip_wvvvv_) {
        d_Loo_   = barh_cache_->d_Loo;
        d_Lvv_   = barh_cache_->d_Lvv;
        d_Fov_   = barh_cache_->d_Fov;
        d_Woooo_ = barh_cache_->d_Woooo;
        d_Wooov_ = barh_cache_->d_Wooov;
        d_Wovov_ = barh_cache_->d_Wovov;
        d_Wovvo_ = barh_cache_->d_Wovvo;
        d_Wovoo_ = barh_cache_->d_Wovoo;
        d_Wvovv_ = barh_cache_->d_Wvovv;
        d_Wvvvv_ = barh_cache_->d_Wvvvv;   // nullptr under canonical-skip (consistent)
        d_Wvvvo_ = barh_cache_->d_Wvvvo;
        barh_borrowed_ = true;
        std::cout << "  [STEOM share-barH] borrowed all 11 bar-H from IP+EA — "
                     "build_dressed_intermediates SKIPPED." << std::endl;
        return;
    }
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t t1_sz   = (size_t)NO * NV;
    const size_t t2_sz   = (size_t)NO * NO * NV * NV;
    const size_t ovov_sz = (size_t)NO * NV * NO * NV;
    const size_t ooov_sz = (size_t)NO * NO * NO * NV;
    const size_t oovv_sz = (size_t)NO * NO * NV * NV;
    const size_t ovvo_sz = (size_t)NO * NV * NV * NO;
    const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
    const size_t oooo_sz = (size_t)NO * NO * NO * NO;
    const size_t vvvv_sz = (size_t)NV * NV * NV * NV;

    std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz);
    std::vector<real_t> h_ovov(ovov_sz), h_ooov(ooov_sz), h_oovv(oovv_sz);
    std::vector<real_t> h_ovvo(ovvo_sz), h_ovvv(ovvv_sz), h_oooo(oooo_sz);
    // Ship 14: in slab mode the bare vvvv tensor lives in per-device device
    // buffers (d_eri_vvvv_slabs_); host h_vvvv is only needed for the
    // canonical-skip OFF Wvvvv build + self-check + CPU fallback.  Slab mode
    // enforces canonical_skip_wvvvv_=true and runs Term A directly off device,
    // so host h_vvvv stays empty (saves NV⁴·8B host RAM — Pentacene: 91 GB).
    // (RI Term A) when d_eri_vvvv_ is skipped (ri_vvvv_term_a_ && !VALIDATE) the
    // host h_vvvv reference is neither built (canonical-skip ON) nor read, so
    // leave it empty (gate on the actual device pointer, null ⇒ RI path).
    std::vector<real_t> h_vvvv;
    if (eri_vvvv_nslab_ <= 1 && d_eri_vvvv_) h_vvvv.assign(vvvv_sz, 0.0);

    // Internal split-timer (env GANSU_STEOM_BUILD_PROF) to locate the host
    // 足回り hotspots inside build_dressed for the GPU-resident port.
    const char* _bdp = std::getenv("GANSU_PROGRESS");   // progress default-on; GANSU_PROGRESS=0 to quiet
    const bool _bdprof = (std::getenv("GANSU_STEOM_BUILD_PROF") != nullptr)
                       || !_bdp || _bdp[0] != '0';
    std::chrono::high_resolution_clock::time_point _bt = std::chrono::high_resolution_clock::now();
    auto _bsplit = [&](const char* nm){
        if(!_bdprof) return;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
        auto now = std::chrono::high_resolution_clock::now();
        std::cout << "      [build_dressed-PROF] " << nm << " = " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double>(now-_bt).count() << " s" << std::defaultfloat << std::endl;
        _bt = now;
    };

    cudaMemcpy(h_t1.data(),   d_t1_,        t1_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_t2.data(),   d_t2_,        t2_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovov.data(), d_eri_ovov_,  ovov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ooov.data(), d_eri_ooov_,  ooov_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oovv.data(), d_eri_oovv_,  oovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvo.data(), d_eri_ovvo_,  ovvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ovvv.data(), d_eri_ovvv_,  ovvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oooo.data(), d_eri_oooo_,  oooo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (eri_vvvv_nslab_ <= 1 && d_eri_vvvv_)
        cudaMemcpy(h_vvvv.data(), d_eri_vvvv_,  vvvv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_f_oo(NO), h_f_vv(NV);
    cudaMemcpy(h_f_oo.data(), d_f_oo_, NO * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_f_vv.data(), d_f_vv_, NV * sizeof(real_t), cudaMemcpyDeviceToHost);

    _bsplit("pulls D2H");
    // GPU GEMM port of the 3 main O(N⁵) host hotspots in this phase
    // (anthracene profile: ccFvv ~15-20 s + ccFoo ~5-8 s + Wooov ~5 s = ~25 s
    // memory-bound on cache-thrashing OVOV·T2 random access). Each is a single
    // contraction over (l,c,d) or (k,l,d) → free output indices. Single cuBLAS
    // GEMM per intermediate; both operands repacked on host to flatten the
    // contraction index. Memory ~800 MB per repack at anthracene, freed after
    // GEMM. Default ON when GPU available; host fallback retained.
    //   ccFvv: C[a,c] = -Σ_{k,l,d} (2·OVOV(k,c,l,d) - OVOV(k,d,l,c)) · tau(k,l,a,d)
    //          where tau(k,l,a,d) = T2(k,l,a,d) + T1(k,a)·T1(l,d)
    //   ccFoo: C[k,i] = +Σ_{l,c,d} (2·OVOV(k,c,l,d) - OVOV(k,d,l,c)) · tau(i,l,c,d)
    //          where tau(i,l,c,d) = T2(i,l,c,d) + T1(i,c)·T1(l,d)
    //   Wooov: C[k,l,i,d] = Σ_c T1(i,c) · OVOV(k,c,l,d) (added to bare OOOV)
    std::vector<real_t> ct_ccFvv, ct_ccFoo, ct_wooov;
    bool ccF_wooov_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        // ---- ccFvv GEMM ----
        // A_rm[(k,l,d), c] = 2·OVOV(k,c,l,d) - OVOV(k,d,l,c)    shape [K=NO²·NV, N=NV]
        // B_T_rm[a, (k,l,d)] = tau(k,l,a,d)                       shape [M=NV, K=NO²·NV]
        // C_rm[a, c] = B_T · A                                    shape [M=NV, N=NV]
        {
            const int M = NV, N = NV;
            const int Kkld = NO * NO * NV;
            std::vector<real_t> hA((size_t)Kkld * N), hBT((size_t)M * Kkld);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < NO; ++k)
                for (int l = 0; l < NO; ++l)
                    for (int d = 0; d < NV; ++d)
                        for (int c = 0; c < NV; ++c)
                            hA[(((size_t)k*NO + l)*NV + d)*N + c] =
                                2.0 * H_OVOV(k,c,l,d) - H_OVOV(k,d,l,c);
            #pragma omp parallel for collapse(2)
            for (int a = 0; a < NV; ++a)
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d)
                            hBT[(((size_t)a*NO + k)*NO + l)*NV + d] =
                                H_T2(k,l,a,d) + H_T1(k,a) * H_T1(l,d);
            ct_ccFvv.assign((size_t)M * N, 0.0);
            real_t *dA=nullptr, *dB=nullptr, *dC=nullptr;
            tracked_cudaMalloc(&dA, (size_t)Kkld*N*sizeof(real_t));
            tracked_cudaMalloc(&dB, (size_t)M*Kkld*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)M*N*sizeof(real_t));
            cudaMemcpy(dA, hA.data(),  (size_t)Kkld*N*sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hBT.data(), (size_t)M*Kkld*sizeof(real_t), cudaMemcpyHostToDevice);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, Kkld,
                        &one, dA, N, dB, Kkld, &zero, dC, N);
            cudaMemcpy(ct_ccFvv.data(), dC, (size_t)M*N*sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dC);
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0; a<NV; a+=(NV/4>0?NV/4:1))
                    for (int c=0; c<NV; c+=(NV/4>0?NV/4:1)) {
                        real_t ref = 0.0;
                        for (int k=0; k<NO; ++k) for (int l=0; l<NO; ++l) for (int d=0; d<NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            ref += (2.0*kcld - kdlc) * (H_T2(k,l,a,d) + H_T1(k,a)*H_T1(l,d));
                        }
                        dmax = std::max(dmax, std::fabs(ref - ct_ccFvv[(size_t)a*NV + c]));
                    }
                std::cout << "[STEOM build self-check] ccFvv GEMM max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // ---- ccFoo GEMM ----
        // A_rm[k, (l,c,d)] = 2·OVOV(k,c,l,d) - OVOV(k,d,l,c)    shape [M=NO, K=NO·NV²]
        // B_rm[(l,c,d), i] = tau(i,l,c,d)                         shape [K=NO·NV², N=NO]
        // C_rm[k, i] = A·B                                        shape [M=NO, N=NO]
        {
            const int M = NO, N = NO;
            const int Klcd = NO * NV * NV;
            std::vector<real_t> hA((size_t)M * Klcd), hB((size_t)Klcd * N);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < NO; ++k)
                for (int l = 0; l < NO; ++l)
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d)
                            hA[((size_t)k*Klcd) + (((size_t)l*NV + c)*NV + d)] =
                                2.0 * H_OVOV(k,c,l,d) - H_OVOV(k,d,l,c);
            #pragma omp parallel for collapse(2)
            for (int l = 0; l < NO; ++l)
                for (int c = 0; c < NV; ++c)
                    for (int d = 0; d < NV; ++d)
                        for (int i = 0; i < NO; ++i)
                            hB[(((size_t)l*NV + c)*NV + d)*N + i] =
                                H_T2(i,l,c,d) + H_T1(i,c) * H_T1(l,d);
            ct_ccFoo.assign((size_t)M * N, 0.0);
            real_t *dA=nullptr, *dB=nullptr, *dC=nullptr;
            tracked_cudaMalloc(&dA, (size_t)M*Klcd*sizeof(real_t));
            tracked_cudaMalloc(&dB, (size_t)Klcd*N*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)M*N*sizeof(real_t));
            cudaMemcpy(dA, hA.data(), (size_t)M*Klcd*sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), (size_t)Klcd*N*sizeof(real_t), cudaMemcpyHostToDevice);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, Klcd,
                        &one, dB, N, dA, Klcd, &zero, dC, N);
            cudaMemcpy(ct_ccFoo.data(), dC, (size_t)M*N*sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dC);
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int k=0; k<NO; ++k) for (int i=0; i<NO; ++i) {
                    real_t ref = 0.0;
                    for (int l=0; l<NO; ++l) for (int c=0; c<NV; ++c) for (int d=0; d<NV; ++d) {
                        real_t kcld = H_OVOV(k,c,l,d);
                        real_t kdlc = H_OVOV(k,d,l,c);
                        ref += (2.0*kcld - kdlc) * (H_T2(i,l,c,d) + H_T1(i,c)*H_T1(l,d));
                    }
                    dmax = std::max(dmax, std::fabs(ref - ct_ccFoo[(size_t)k*NO + i]));
                }
                std::cout << "[STEOM build self-check] ccFoo GEMM max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // ---- Wooov Σ_c T1·OVOV GEMM ----
        // A_rm[(k,l,d), c] = OVOV(k,c,l,d)                       shape [M=NO²·NV, K=NV]
        // B_rm[c, i] = T1(i,c)                                    shape [K=NV, N=NO]
        // C_rm[(k,l,d), i] = A·B                                  shape [M=NO²·NV, N=NO]
        {
            const int M = NO * NO * NV;
            const int Kc = NV, N = NO;
            std::vector<real_t> hA((size_t)M * Kc), hB((size_t)Kc * N);
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < NO; ++k)
                for (int l = 0; l < NO; ++l)
                    for (int d = 0; d < NV; ++d)
                        for (int c = 0; c < NV; ++c)
                            hA[(((size_t)k*NO + l)*NV + d)*Kc + c] = H_OVOV(k,c,l,d);
            #pragma omp parallel for
            for (int c = 0; c < NV; ++c)
                for (int i = 0; i < NO; ++i)
                    hB[(size_t)c*N + i] = H_T1(i,c);
            ct_wooov.assign((size_t)M * N, 0.0);
            real_t *dA=nullptr, *dB=nullptr, *dC=nullptr;
            tracked_cudaMalloc(&dA, (size_t)M*Kc*sizeof(real_t));
            tracked_cudaMalloc(&dB, (size_t)Kc*N*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)M*N*sizeof(real_t));
            cudaMemcpy(dA, hA.data(), (size_t)M*Kc*sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), (size_t)Kc*N*sizeof(real_t), cudaMemcpyHostToDevice);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, Kc,
                        &one, dB, N, dA, Kc, &zero, dC, N);
            cudaMemcpy(ct_wooov.data(), dC, (size_t)M*N*sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dC);
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int k=0; k<NO; ++k) for (int l=0; l<NO; ++l)
                    for (int i=0; i<NO; ++i) for (int d=0; d<NV; d+=(NV/4>0?NV/4:1)) {
                        real_t ref = 0.0;
                        for (int c=0; c<NV; ++c) ref += H_T1(i,c) * H_OVOV(k,c,l,d);
                        dmax = std::max(dmax, std::fabs(ref -
                            ct_wooov[(((size_t)k*NO + l)*NV + d)*N + i]));
                    }
                std::cout << "[STEOM build self-check] Wooov Σc T1·OVOV GEMM max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        ccF_wooov_gpu = true;
    }
#endif

    // cc_Fov, cc_Foo, cc_Fvv — helpers for Loo/Lvv
    std::vector<real_t> h_Fov(t1_sz, 0.0), h_ccFoo(NO * NO, 0.0), h_ccFvv(NV * NV, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int c = 0; c < NV; ++c) {
            real_t v = 0.0;
            for (int l = 0; l < NO; ++l)
                for (int d = 0; d < NV; ++d) {
                    v += 2.0 * H_OVOV(k,c,l,d) * H_T1(l,d);
                    v -=       H_OVOV(k,d,l,c) * H_T1(l,d);
                }
            h_Fov[k*NV + c] = v;
        }
    if (ccF_wooov_gpu) {
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int i = 0; i < NO; ++i)
                h_ccFoo[k*NO + i] = (k == i ? h_f_oo[k] : 0.0)
                                    + ct_ccFoo[(size_t)k*NO + i];
    } else {
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int i = 0; i < NO; ++i) {
                real_t v = (k == i ? h_f_oo[k] : 0.0);
                for (int l = 0; l < NO; ++l)
                    for (int c = 0; c < NV; ++c)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v += 2.0 * kcld * H_T2(i,l,c,d) - kdlc * H_T2(i,l,c,d);
                            v += (2.0 * kcld - kdlc) * H_T1(i,c) * H_T1(l,d);
                        }
                h_ccFoo[k*NO + i] = v;
            }
    }
    if (ccF_wooov_gpu) {
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c)
                h_ccFvv[a*NV + c] = (a == c ? h_f_vv[a] : 0.0)
                                    - ct_ccFvv[(size_t)a*NV + c];
    } else {
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c) {
                real_t v = (a == c ? h_f_vv[a] : 0.0);
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            real_t kdlc = H_OVOV(k,d,l,c);
                            v -= 2.0 * kcld * H_T2(k,l,a,d) - kdlc * H_T2(k,l,a,d);
                            v -= (2.0 * kcld - kdlc) * H_T1(k,a) * H_T1(l,d);
                        }
                h_ccFvv[a*NV + c] = v;
            }
    }

    // Loo / Lvv
    std::vector<real_t> h_Loo(NO * NO, 0.0), h_Lvv(NV * NV, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int i = 0; i < NO; ++i) {
            real_t v = h_ccFoo[k*NO + i];
            for (int c = 0; c < NV; ++c)
                v += h_Fov[k*NV + c] * H_T1(i,c);
            for (int l = 0; l < NO; ++l)
                for (int c = 0; c < NV; ++c) {
                    v += 2.0 * H_OOOV(k,i,l,c) * H_T1(l,c);
                    v -=       H_OOOV(l,i,k,c) * H_T1(l,c);
                }
            h_Loo[k*NO + i] = v;
        }
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int c = 0; c < NV; ++c) {
            real_t v = h_ccFvv[a*NV + c];
            for (int k = 0; k < NO; ++k)
                v -= h_Fov[k*NV + c] * H_T1(k,a);
            for (int k = 0; k < NO; ++k)
                for (int d = 0; d < NV; ++d) {
                    v += 2.0 * H_OVVV(k,d,a,c) * H_T1(k,d);
                    v -=       H_OVVV(k,c,a,d) * H_T1(k,d);
                }
            h_Lvv[a*NV + c] = v;
        }

    // Wooov[k,l,i,d] (IP version, also used in Woooo / Wovoo / Wovvo / Wovov below)
    std::vector<real_t> h_Wooov(ooov_sz, 0.0);
    if (ccF_wooov_gpu) {
        const int N_w = NO;
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int l = 0; l < NO; ++l)
                for (int i = 0; i < NO; ++i)
                    for (int d = 0; d < NV; ++d)
                        h_Wooov[(((size_t)k * NO + l) * NO + i) * NV + d] =
                            H_OOOV(k,i,l,d)
                            + ct_wooov[(((size_t)k*NO + l)*NV + d)*N_w + i];
    } else {
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int l = 0; l < NO; ++l)
                for (int i = 0; i < NO; ++i)
                    for (int d = 0; d < NV; ++d) {
                        real_t v = H_OOOV(k,i,l,d);
                        for (int c = 0; c < NV; ++c)
                            v += H_T1(i,c) * H_OVOV(k,c,l,d);
                        h_Wooov[(((size_t)k * NO + l) * NO + i) * NV + d] = v;
                    }
    }
    #define H_WOOOV(k,l,i,d) h_Wooov[(((size_t)(k) * NO + (l)) * NO + (i)) * NV + (d)]

    // GPU GEMM port of two more O(N⁶) hotspots that share the SAME right
    // operand tau_B[(c,d),(i,j)] = t2[i,j,c,d] + t1[i,c]·t1[j,d] (by the T2
    // permutation symmetry t2[i,j,c,d]=t2[j,i,d,c], the Wovoo Σ_cd ovvv·tau2
    // term reuses the identical tau). One B, two A's, two GEMMs:
    //   ct_woooo[(k,l),(i,j)] = Σ_cd ovov[k,c,l,d]·tau[i,j,c,d]   (NO⁴·NV²)
    //   ct_wovoo[(k,b),(i,j)] = Σ_cd ovvv[k,c,b,d]·tau[i,j,c,d]   (NO³·NV³)
    const int OO_N = NO*NO, VV_K = NV*NV, OOkl_M = NO*NO, OVkb_M = NO*NV;
    std::vector<real_t> ct_woooo, ct_wovoo;
    bool oooo_wovoo_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        // Two distinct right operands: Woooo needs tau[i,j,c,d]; Wovoo needs
        // tau[j,i,d,c]. These are equal ONLY if T2 has exact permutation
        // symmetry t2[i,j,c,d]=t2[j,i,d,c] — but the DLPNO back-transformed T2
        // is symmetric only to ~1e-6, so we build both explicitly.
        std::vector<real_t> hB((size_t)VV_K*OO_N), hB2((size_t)VV_K*OO_N);
        #pragma omp parallel for collapse(2)
        for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
            for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                const size_t o=(size_t)(c*NV+d)*OO_N+(i*NO+j);
                hB[o]  = H_T2(i,j,c,d) + H_T1(i,c)*H_T1(j,d);   // Woooo
                hB2[o] = H_T2(j,i,d,c) + H_T1(j,d)*H_T1(i,c);   // Wovoo
            }
        std::vector<real_t> hAo((size_t)OOkl_M*VV_K), hAv((size_t)OVkb_M*VV_K);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
            for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                hAo[(size_t)(k*NO+l)*VV_K+(c*NV+d)] = H_OVOV(k,c,l,d);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int b=0;b<NV;++b)
            for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                hAv[(size_t)(k*NV+b)*VV_K+(c*NV+d)] = H_OVVV(k,c,b,d);
        ct_woooo.assign((size_t)OOkl_M*OO_N,0.0); ct_wovoo.assign((size_t)OVkb_M*OO_N,0.0);
        if (build_gpus_ > 1) {
            // Two distinct outputs (OOkl_M vs OVkb_M, distinct K=NV² → distinct B/A pairs)
            // → 2 separate single-term n-slab calls.  Slab the OO_N (col-major n) dim of
            // each.  ct_woooo/ct_wovoo row-major rows = (kl) or (kb) contiguous in OO_N.
            multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 OO_N, OOkl_M, VV_K,
                                 hB.data(), (size_t)VV_K*OO_N, OO_N,
                                 hAo.data(), (size_t)OOkl_M*VV_K, VV_K,
                                 ct_woooo.data());
            multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 OO_N, OVkb_M, VV_K,
                                 hB2.data(), (size_t)VV_K*OO_N, OO_N,
                                 hAv.data(), (size_t)OVkb_M*VV_K, VV_K,
                                 ct_wovoo.data());
        } else {
            real_t *dB=nullptr,*dB2=nullptr,*dAo=nullptr,*dAv=nullptr,*dCo=nullptr,*dCv=nullptr;
            tracked_cudaMalloc(&dB,(size_t)VV_K*OO_N*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)VV_K*OO_N*sizeof(real_t));
            tracked_cudaMalloc(&dAo,(size_t)OOkl_M*VV_K*sizeof(real_t));
            tracked_cudaMalloc(&dAv,(size_t)OVkb_M*VV_K*sizeof(real_t));
            tracked_cudaMalloc(&dCo,(size_t)OOkl_M*OO_N*sizeof(real_t));
            tracked_cudaMalloc(&dCv,(size_t)OVkb_M*OO_N*sizeof(real_t));
            cudaMemcpy(dB,hB.data(),(size_t)VV_K*OO_N*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB2,hB2.data(),(size_t)VV_K*OO_N*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dAo,hAo.data(),(size_t)OOkl_M*VV_K*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dAv,hAv.data(),(size_t)OVkb_M*VV_K*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OOkl_M,VV_K,&one,dB, OO_N,dAo,VV_K,&zero,dCo,OO_N);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,OO_N,OVkb_M,VV_K,&one,dB2,OO_N,dAv,VV_K,&zero,dCv,OO_N);
            cudaMemcpy(ct_woooo.data(),dCo,(size_t)OOkl_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
            cudaMemcpy(ct_wovoo.data(),dCv,(size_t)OVkb_M*OO_N*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dB);tracked_cudaFree(dB2);tracked_cudaFree(dAo);tracked_cudaFree(dAv);tracked_cudaFree(dCo);tracked_cudaFree(dCv);
        }
        oooo_wovoo_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1=0.0,d2=0.0;
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int l=0;l<NO;++l)
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                        v += H_OVOV(k,c,l,d)*(H_T2(i,j,c,d)+H_T1(i,c)*H_T1(j,d));
                    d1=std::max(d1,std::fabs(v-ct_woooo[(size_t)(k*NO+l)*OO_N+(i*NO+j)]));
                }
            for (int k=0;k<NO;k+=(NO/2>0?NO/2:1)) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t v=0.0; for (int c=0;c<NV;++c) for (int d=0;d<NV;++d)
                        v += H_OVVV(k,c,b,d)*(H_T2(j,i,d,c)+H_T1(j,d)*H_T1(i,c));
                    d2=std::max(d2,std::fabs(v-ct_wovoo[(size_t)(k*NV+b)*OO_N+(i*NO+j)]));
                }
            std::cout << "[STEOM build self-check] Woooo Σcd max|Δ| = " << std::scientific << d1
                      << ", Wovoo Σcd max|Δ| = " << d2 << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    _bsplit("ccF/Loo/Lvv/Wooov");
    // Woooo (IP version)
    std::vector<real_t> h_Woooo(oooo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int l = 0; l < NO; ++l)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOO(k,i,l,j);
                    for (int d = 0; d < NV; ++d)
                        v += H_OOOV(k,i,l,d) * H_T1(j,d);
                    for (int c = 0; c < NV; ++c)
                        v += H_OOOV(l,j,k,c) * H_T1(i,c);
                    if (oooo_wovoo_gpu) {
                        v += ct_woooo[(size_t)(k*NO+l)*OO_N + (i*NO+j)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int d = 0; d < NV; ++d) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += kcld * (H_T2(i,j,c,d) + H_T1(i,c) * H_T1(j,d));
                            }
                    }
                    h_Woooo[(((size_t)k * NO + l) * NO + i) * NO + j] = v;
                }

    _bsplit("Woooo");
    // W1ovov (helper) + Wovov = W1 + W2
    // GPU GEMM port of the O(NO³·NV³) hotspot:
    //   ct[k,b,i,d] = Σ_{c,l} ovov[k,c,l,d]·t2[i,l,c,b]
    // As a single GEMM over contract index (c,l): free (k,d) and (i,b).
    //   A[(k,d),(c,l)] = ovov[k,c,l,d]        (M=NO·NV, K=NV·NO)
    //   B[(c,l),(i,b)] = t2[i,l,c,b]          (K=NV·NO, N=NO·NV)
    //   C[(k,d),(i,b)] = Σ A·B  (row-major M×N)  →  W1ovov[k,b,i,d] = oovv − C
    // The result buffer is stored in C-layout [(k,d),(i,b)] and read directly below.
    const int MO_kd = NO * NV, NO_ib = NO * NV, KO_cl = NV * NO;
    std::vector<real_t> w1ovov_ct;     // [(k,d),(i,b)] = ((k*NV+d)*NO·NV + (i*NV+b))
    bool w1ovov_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        std::vector<real_t> hA((size_t)MO_kd * KO_cl), hB((size_t)KO_cl * NO_ib);
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int d = 0; d < NV; ++d)
                for (int c = 0; c < NV; ++c)
                    for (int l = 0; l < NO; ++l)
                        hA[(size_t)(k*NV+d)*KO_cl + (c*NO+l)] = H_OVOV(k,c,l,d);
        #pragma omp parallel for collapse(2)
        for (int c = 0; c < NV; ++c)
            for (int l = 0; l < NO; ++l)
                for (int i = 0; i < NO; ++i)
                    for (int b = 0; b < NV; ++b)
                        hB[(size_t)(c*NO+l)*NO_ib + (i*NV+b)] = H_T2(i,l,c,b);
        w1ovov_ct.assign((size_t)MO_kd * NO_ib, 0.0);
        // C_cm[NO_ib×MO_kd] = dB·dA (col-major). Slabbing MO_kd → (k,d) row range of
        // row-major w1ovov_ct (contiguous); bit-identical to single-GPU.
        if (build_gpus_ > 1) {
            multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 NO_ib, MO_kd, KO_cl,
                                 hB.data(), (size_t)KO_cl*NO_ib, NO_ib,
                                 hA.data(), (size_t)MO_kd*KO_cl, KO_cl,
                                 w1ovov_ct.data());
        } else {
            cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
            real_t *dA = nullptr, *dB = nullptr, *dC = nullptr;
            tracked_cudaMalloc(&dA, (size_t)MO_kd * KO_cl * sizeof(real_t));
            tracked_cudaMalloc(&dB, (size_t)KO_cl * NO_ib * sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)MO_kd * NO_ib * sizeof(real_t));
            cudaMemcpy(dA, hA.data(), (size_t)MO_kd * KO_cl * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hB.data(), (size_t)KO_cl * NO_ib * sizeof(real_t), cudaMemcpyHostToDevice);
            const real_t one = 1.0, zero = 0.0;
            // Row-major C[M×N]=A[M×K]·B[K×N]: cuBLAS(N,N, N,M,K, B(ldb=N), A(lda=K), C(ldc=N)).
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, NO_ib, MO_kd, KO_cl, &one,
                        dB, NO_ib, dA, KO_cl, &zero, dC, NO_ib);
            cudaMemcpy(w1ovov_ct.data(), dC, (size_t)MO_kd * NO_ib * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dC);
        }
        w1ovov_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            for (int k = 0; k < NO; k += (NO/2>0?NO/2:1))
                for (int d = 0; d < NV; d += (NV/2>0?NV/2:1))
                    for (int i = 0; i < NO; ++i) for (int b = 0; b < NV; ++b) {
                        real_t v = 0.0;
                        for (int c = 0; c < NV; ++c) for (int l = 0; l < NO; ++l)
                            v += H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                        dmax = std::max(dmax, std::fabs(v - w1ovov_ct[(size_t)(k*NV+d)*NO_ib + (i*NV+b)]));
                    }
            std::cout << "[STEOM build self-check] W1ovov GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif
    std::vector<real_t> h_W1ovov(ovov_sz, 0.0), h_Wovov(ovov_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OOVV(k,i,b,d);
                    if (w1ovov_gpu) {
                        v -= w1ovov_ct[(size_t)(k*NV+d)*NO_ib + (i*NV+b)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OVOV(k,c,l,d) * H_T2(i,l,c,b);
                    }
                    h_W1ovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }
    #define H_W1OVOV(k,b,i,d) h_W1ovov[(((size_t)(k) * NV + (b)) * NO + (i)) * NV + (d)]
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_W1OVOV(k,b,i,d);
                    for (int l = 0; l < NO; ++l)
                        v -= H_WOOOV(k,l,i,d) * H_T1(l,b);
                    for (int c = 0; c < NV; ++c)
                        v += H_OVVV(k,c,b,d) * H_T1(i,c);
                    h_Wovov[(((size_t)k * NV + b) * NO + i) * NV + d] = v;
                }

    _bsplit("Wovov");
    // W1ovvo (helper) + Wovvo = W1 + W2
    // GPU GEMM port of the O(NO³·NV³) hotspot:
    //   ct[k,b,c,j] = Σ_{l,d} [ (2·ovov[k,c,l,d] − ovov[k,d,l,c])·t2[j,l,b,d]
    //                           − ovov[k,c,l,d]·t2[l,j,b,d] ]
    // Two GEMMs over contract (l,d): free (k,c) and (j,b).
    //   A1[(k,c),(l,d)] = 2·ovov[k,c,l,d] − ovov[k,d,l,c];  B1[(l,d),(j,b)] = t2[j,l,b,d]
    //   A2[(k,c),(l,d)] = ovov[k,c,l,d];                    B2[(l,d),(j,b)] = t2[l,j,b,d]
    //   C[(k,c),(j,b)] = A1·B1 − A2·B2   (accumulated)  →  W1ovvo[k,b,c,j] = ovvo + C
    const int M_kc = NO * NV, N_jb = NO * NV, K_ld = NO * NV;
    std::vector<real_t> w1ovvo_ct;     // [(k,c),(j,b)] = ((k*NV+c)*NO·NV + (j*NV+b))
    bool w1ovvo_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        std::vector<real_t> hA1((size_t)M_kc*K_ld), hB1((size_t)K_ld*N_jb),
                            hA2((size_t)M_kc*K_ld), hB2((size_t)K_ld*N_jb);
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int c = 0; c < NV; ++c)
                for (int l = 0; l < NO; ++l)
                    for (int d = 0; d < NV; ++d) {
                        const size_t o = (size_t)(k*NV+c)*K_ld + (l*NV+d);
                        hA1[o] = 2.0*H_OVOV(k,c,l,d) - H_OVOV(k,d,l,c);
                        hA2[o] =     H_OVOV(k,c,l,d);
                    }
        #pragma omp parallel for collapse(2)
        for (int l = 0; l < NO; ++l)
            for (int d = 0; d < NV; ++d)
                for (int j = 0; j < NO; ++j)
                    for (int b = 0; b < NV; ++b) {
                        const size_t o = (size_t)(l*NV+d)*N_jb + (j*NV+b);
                        hB1[o] = H_T2(j,l,b,d);
                        hB2[o] = H_T2(l,j,b,d);
                    }
        w1ovvo_ct.assign((size_t)M_kc * N_jb, 0.0);
        if (build_gpus_ > 1) {
            // C = (+1)·A1·B1 + (−1)·A2·B2, col-major view [N_jb × M_kc].
            // Slab M_kc across devices; each device's column slab = contiguous
            // (k,c) row range of row-major w1ovvo_ct → bit-identical.
            std::vector<MultiGpuGemmTerm> ts;
            ts.push_back({CUBLAS_OP_N, CUBLAS_OP_N, +1.0,
                          hB1.data(), (size_t)K_ld*N_jb, N_jb,
                          hA1.data(), (size_t)M_kc*K_ld, K_ld});
            ts.push_back({CUBLAS_OP_N, CUBLAS_OP_N, -1.0,
                          hB2.data(), (size_t)K_ld*N_jb, N_jb,
                          hA2.data(), (size_t)M_kc*K_ld, K_ld});
            multi_gpu_gemm_nslab_terms(build_gpus_, N_jb, M_kc, K_ld, ts, w1ovvo_ct.data());
        } else {
            cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
            real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA1,(size_t)M_kc*K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB1,(size_t)K_ld*N_jb*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)M_kc*K_ld*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)K_ld*N_jb*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)M_kc*N_jb*sizeof(real_t));
            cudaMemcpy(dA1,hA1.data(),(size_t)M_kc*K_ld*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB1,hB1.data(),(size_t)K_ld*N_jb*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dA2,hA2.data(),(size_t)M_kc*K_ld*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB2,hB2.data(),(size_t)K_ld*N_jb*sizeof(real_t),cudaMemcpyHostToDevice);
            const real_t one = 1.0, negone = -1.0, zero = 0.0;
            // Row-major C[M×N]=A·B: cuBLAS(N,N, N,M,K, B, N, A, K, beta, C, N).
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N_jb, M_kc, K_ld, &one,
                        dB1, N_jb, dA1, K_ld, &zero, dC, N_jb);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N_jb, M_kc, K_ld, &negone,
                        dB2, N_jb, dA2, K_ld, &one, dC, N_jb);
            cudaMemcpy(w1ovvo_ct.data(), dC, (size_t)M_kc*N_jb*sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA1);tracked_cudaFree(dB1);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        }
        w1ovvo_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            for (int k = 0; k < NO; k += (NO/2>0?NO/2:1))
                for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                    for (int c = 0; c < NV; ++c) for (int j = 0; j < NO; ++j) {
                        real_t v = 0.0;
                        for (int l = 0; l < NO; ++l) for (int d = 0; d < NV; ++d) {
                            real_t kcld = H_OVOV(k,c,l,d);
                            v += 2.0*kcld*H_T2(j,l,b,d) - kcld*H_T2(l,j,b,d)
                                 - H_OVOV(k,d,l,c)*H_T2(j,l,b,d);
                        }
                        dmax = std::max(dmax, std::fabs(v - w1ovvo_ct[(size_t)(k*NV+c)*N_jb + (j*NV+b)]));
                    }
            std::cout << "[STEOM build self-check] W1ovvo GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif
    std::vector<real_t> h_W1ovvo(ovvo_sz, 0.0), h_Wovvo(ovvo_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OVVO(k,c,b,j);
                    if (w1ovvo_gpu) {
                        v += w1ovvo_ct[(size_t)(k*NV+c)*N_jb + (j*NV+b)];
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += 2.0 * kcld * H_T2(j,l,b,d);
                                v -=       kcld * H_T2(l,j,b,d);
                                v -= H_OVOV(k,d,l,c) * H_T2(j,l,b,d);
                            }
                    }
                    h_W1ovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }
    #define H_W1OVVO(k,b,c,j) h_W1ovvo[(((size_t)(k) * NV + (b)) * NV + (c)) * NO + (j)]
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_W1OVVO(k,b,c,j);
                    for (int l = 0; l < NO; ++l)
                        v -= H_T1(l,b) * H_WOOOV(l,k,j,c);
                    for (int d = 0; d < NV; ++d)
                        v += H_OVVV(k,c,b,d) * H_T1(j,d);
                    h_Wovvo[(((size_t)k * NV + b) * NV + c) * NO + j] = v;
                }

    _bsplit("Wovvo");
    // Wovoo (IP-side, full 11-term PySCF Wovoo build — needed in STEOM σ2 coupling)
    const size_t wovoo_sz = (size_t)NO * NV * NO * NO;
    std::vector<real_t> h_Wovoo(wovoo_sz, 0.0);
    // GPU GEMM port of the Wovoo Σ_ld ooov·t2 terms (O(NO⁴·NV²)):
    //   ct[k,i,j,b] = Σ_{l,d}[ (2·ooov(k,i,l,d)−ooov(l,i,k,d))·t2(l,j,d,b)
    //                          − ooov(k,i,l,d)·t2(j,l,d,b) ]
    // 2 GEMMs (T1+T3 share t2(l,j,d,b)): C[(k,i),(j,b)] = A13·B − A2·B2, contract (l,d).
    const int KI_M = NO*NO, JB_N = NO*NV, LD_K2 = NO*NV;
    std::vector<real_t> ct_wovoo_t2;
    bool wovoo_t2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA13((size_t)KI_M*LD_K2), hBlj((size_t)LD_K2*JB_N),
                            hA2((size_t)KI_M*LD_K2), hB2((size_t)LD_K2*JB_N);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int i=0;i<NO;++i)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(k*NO+i)*LD_K2+(l*NV+d);
                hA13[o]=2.0*H_OOOV(k,i,l,d)-H_OOOV(l,i,k,d); hA2[o]=H_OOOV(k,i,l,d);
            }
        #pragma omp parallel for collapse(2)
        for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
            for (int j=0;j<NO;++j) for (int b=0;b<NV;++b) {
                const size_t o=(size_t)(l*NV+d)*JB_N+(j*NV+b);
                hBlj[o]=H_T2(l,j,d,b); hB2[o]=H_T2(j,l,d,b);
            }
        ct_wovoo_t2.assign((size_t)KI_M*JB_N,0.0);
        if (build_gpus_ > 1) {
            // 2-term accumulate, col-major C[JB_N × KI_M], slab KI_M → (k,i) row range
            // of row-major ct_wovoo_t2 (contiguous, no gather), bit-identical to single-GPU.
            std::vector<MultiGpuGemmTerm> ts;
            ts.push_back({CUBLAS_OP_N, CUBLAS_OP_N, +1.0,
                          hBlj.data(), (size_t)LD_K2*JB_N, JB_N,
                          hA13.data(), (size_t)KI_M*LD_K2, LD_K2});
            ts.push_back({CUBLAS_OP_N, CUBLAS_OP_N, -1.0,
                          hB2.data(),  (size_t)LD_K2*JB_N, JB_N,
                          hA2.data(),  (size_t)KI_M*LD_K2, LD_K2});
            multi_gpu_gemm_nslab_terms(build_gpus_, JB_N, KI_M, LD_K2, ts, ct_wovoo_t2.data());
        } else {
            real_t *dA13=nullptr,*dBlj=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA13,(size_t)KI_M*LD_K2*sizeof(real_t));
            tracked_cudaMalloc(&dBlj,(size_t)LD_K2*JB_N*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)KI_M*LD_K2*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)LD_K2*JB_N*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)KI_M*JB_N*sizeof(real_t));
            cudaMemcpy(dA13,hA13.data(),(size_t)KI_M*LD_K2*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dBlj,hBlj.data(),(size_t)LD_K2*JB_N*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dA2,hA2.data(),(size_t)KI_M*LD_K2*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB2,hB2.data(),(size_t)LD_K2*JB_N*sizeof(real_t),cudaMemcpyHostToDevice);
            const real_t one=1.0,negone=-1.0,zero=0.0;
            // C_rm[M×N]=A·B: cuBLAS(N,N, N, M, K, B, N, A, K, C, N).
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,JB_N,KI_M,LD_K2,&one,   dBlj,JB_N,dA13,LD_K2,&zero,dC,JB_N);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,JB_N,KI_M,LD_K2,&negone,dB2, JB_N,dA2, LD_K2,&one, dC,JB_N);
            cudaMemcpy(ct_wovoo_t2.data(),dC,(size_t)KI_M*JB_N*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA13);tracked_cudaFree(dBlj);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        }
        wovoo_t2_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int k=0;k<NO;++k) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
                        t += 2.0*H_OOOV(k,i,l,d)*H_T2(l,j,d,b) - H_OOOV(k,i,l,d)*H_T2(j,l,d,b)
                             - H_OOOV(l,i,k,d)*H_T2(l,j,d,b);
                    dmax=std::max(dmax,std::fabs(t-ct_wovoo_t2[(size_t)(k*NO+i)*JB_N+(j*NV+b)]));
                }
            std::cout << "[STEOM build self-check] Wovoo ooov·t2 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // GPU GEMM port of Wovoo item 6 — the last big inner-loop scatter term
    // (was outer-loop O(NO⁴·NV²) ~ 220 G ops at anthracene = Wovoo phase
    // bottleneck per build_dressed-PROF). C[(k,j),(b,i)] = Σ_{l,c} OOOV(l,j,k,c)·T2(l,i,b,c).
    // Single GEMM, free (k,j) on M side / (b,i) on N side / contract (l,c) on K side.
    // Sign applied at scatter site (`v -= ct6[...]`).
    const int M_w6 = NO*NO, N_w6 = NV*NO, K_w6 = NO*NV;
    std::vector<real_t> ct_wovoo_6;
    bool wovoo_6_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)M_w6*K_w6), hB((size_t)K_w6*N_w6);
        // A[(k,j),(l,c)] = OOOV(l,j,k,c) — h_ooov[l][j][k][c]; must repack (k/l swap).
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int j=0;j<NO;++j)
            for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
                hA[(size_t)(k*NO+j)*K_w6 + (l*NV+c)] = H_OOOV(l,j,k,c);
        // B[(l,c),(b,i)] = T2(l,i,b,c) — h_t2[l][i][b][c]; must repack (c moves outside).
        #pragma omp parallel for collapse(2)
        for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
            for (int b=0;b<NV;++b) for (int i=0;i<NO;++i)
                hB[(size_t)(l*NV+c)*N_w6 + (b*NO+i)] = H_T2(l,i,b,c);
        ct_wovoo_6.assign((size_t)M_w6*N_w6, 0.0);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)M_w6*K_w6*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)K_w6*N_w6*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)M_w6*N_w6*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)M_w6*K_w6*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)K_w6*N_w6*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        // C_rm[M×N] = A·B: cuBLAS(N,N, N, M, K, B, N, A, K, C, N).
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w6,M_w6,K_w6,&one,dB,N_w6,dA,K_w6,&zero,dC,N_w6);
        cudaMemcpy(ct_wovoo_6.data(),dC,(size_t)M_w6*N_w6*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        wovoo_6_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int k=0;k<NO;++k) for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                for (int i=0;i<NO;++i) for (int j=0;j<NO;++j) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int c=0;c<NV;++c)
                        t += H_OOOV(l,j,k,c)*H_T2(l,i,b,c);
                    dmax=std::max(dmax,std::fabs(t-ct_wovoo_6[(size_t)(k*NO+j)*N_w6+(b*NO+i)]));
                }
            std::cout << "[STEOM build self-check] Wovoo item 6 (ooov·t2 contract l,c): max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int k = 0; k < NO; ++k)
        for (int b = 0; b < NV; ++b)
            for (int i = 0; i < NO; ++i)
                for (int j = 0; j < NO; ++j) {
                    real_t v = H_OOOV(i,k,j,b);
                    for (int d = 0; d < NV; ++d)
                        v += H_W1OVOV(k,b,i,d) * H_T1(j,d);
                    for (int l = 0; l < NO; ++l)
                        v -= h_Woooo[(((size_t)k * NO + l) * NO + i) * NO + j] * H_T1(l,b);
                    for (int c = 0; c < NV; ++c)
                        v += H_W1OVVO(k,b,c,j) * H_T1(i,c);
                    if (wovoo_t2_gpu) {
                        v += ct_wovoo_t2[(size_t)(k*NO+i)*JB_N+(j*NV+b)];
                    } else {
                        for (int l = 0; l < NO; ++l)
                            for (int d = 0; d < NV; ++d) {
                                v += 2.0 * H_OOOV(k,i,l,d) * H_T2(l,j,d,b);
                                v -=       H_OOOV(k,i,l,d) * H_T2(j,l,d,b);
                                v -=       H_OOOV(l,i,k,d) * H_T2(l,j,d,b);
                            }
                    }
                    if (oooo_wovoo_gpu) {
                        v += ct_wovoo[(size_t)(k*NV+b)*OO_N + (i*NO+j)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int d = 0; d < NV; ++d) {
                                v += H_OVVV(k,c,b,d) * H_T2(j,i,d,c);
                                v += H_OVVV(k,c,b,d) * H_T1(j,d) * H_T1(i,c);
                            }
                    }
                    if (wovoo_6_gpu) {
                        v -= ct_wovoo_6[(size_t)(k*NO+j)*N_w6 + (b*NO+i)];
                    } else {
                        for (int c = 0; c < NV; ++c)
                            for (int l = 0; l < NO; ++l)
                                v -= H_OOOV(l,j,k,c) * H_T2(l,i,b,c);
                    }
                    for (int c = 0; c < NV; ++c)
                        v += h_Fov[k*NV + c] * H_T2(i,j,c,b);
                    h_Wovoo[(((size_t)k * NV + b) * NO + i) * NO + j] = v;
                }

    _bsplit("Wovoo");
    // Wvovv (EA-side)
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    std::vector<real_t> h_Wvovv(wvovv_sz, 0.0);
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int l = 0; l < NO; ++l)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_OVVV(l,d,a,c);
                    for (int k = 0; k < NO; ++k)
                        v -= H_T1(k,a) * H_OVOV(k,c,l,d);
                    h_Wvovv[(((size_t)a * NO + l) * NV + c) * NV + d] = v;
                }

    _bsplit("Wvovv");

    // ----------------------------------------------------------------
    //  P5 canonical-skip Wvvvo·t1 refactor (4-term Wick, EA mirror).
    //  Precomputes Σ_d Wvvvv[a,b,c,d]·t1[j,d] term-by-term WITHOUT ever
    //  materialising a nvir⁴ intermediate. The 4 terms exhaust the
    //  dressed-W expansion (bare ERI + 2 t1·ovvv contractions + τ·ovov),
    //  matching exactly what would have been read from h_Wvvvv if it
    //  had been built. Each term uses ≤ O(NV³·NO + NV²·NO²) scratch —
    //  same complexity class as a single nvir⁴ slice, no extra work.
    //  Below in the Wvvvo build (line ~1704) we consume wvvvo_w_t1
    //  instead of the H_WVVVV·t1 d-loop.
    // ----------------------------------------------------------------
    std::vector<real_t> wvvvo_w_t1;
    if (canonical_skip_wvvvv_) {
        const size_t wvvvo_sz_local = (size_t)NV * NV * NV * NO;
        wvvvo_w_t1.assign(wvvvo_sz_local, 0.0);

        // Term A: + Σ_d (ac|bd)·t1[j,d] (chemist→physicist: H_VVVV(a,c,b,d))
        // GPU path: single cuBLAS GEMM (m=NO, n=NV³, k=NV) reads d_eri_vvvv_
        // (already on device, no upload!) and t1 (small). Computes
        //   intermediate[a,b,c,j] = Σ_d h_vvvv[a,b,c,d]·t1[j,d]   (natural layout)
        // then a b↔c swap kernel scatters into wvvvo_w_t1[a,b,c,j] += intermediate[a,c,b,j]
        // (Term A's H_VVVV(a,c,b,d) macro = h_vvvv at swapped middle indices).
        // anthracene NV⁴·NO ≈ 9.7e10 host ops → A100 GEMM ~0.5 s + kernel <0.1 s
        // = ~30 s reduction in the build_dressed "Wvvvv" timer.
        bool termA_gpu = false;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            const size_t inter_sz = (size_t)NV * NV * NV * NO;
            std::vector<real_t> h_inter(inter_sz, 0.0);
            if (ri_vvvv_term_a_) {
                // RI Term A (mirror of EA): h_inter[a,b,c,j] = Σ_d (ab|cd)·t1[j,d]
                //   = Σ_P B_vv[P,ab]·(Σ_d B_vv[P,cd]·t1[j,d]).  Two GEMMs over the
                // [naux×nvir²] B_vv block; the nvir⁴ tensor is never formed. Output
                // layout matches the dense GEMM ⇒ host b↔c swap below is unchanged.
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                const int naux = eri_block_src_->get_num_auxiliary_basis();
                const int voff = NO + frozen_off_;
                real_t *d_Bvv=nullptr, *d_Y=nullptr, *d_t1_dev=nullptr, *d_inter=nullptr;
                tracked_cudaMalloc(&d_Bvv,    (size_t)naux * NV * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_Y,      (size_t)naux * NV * NO * sizeof(real_t));
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO   * NV      * sizeof(real_t));
                tracked_cudaMalloc(&d_inter,  inter_sz                * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t),
                           cudaMemcpyHostToDevice);
                { const int thr = 256;
                  const size_t tot = (size_t)naux * NV * NV;
                  int blk = (int)((tot + thr - 1) / thr); if (blk > 65535) blk = 65535;
                  steom_gather_bvv_kernel<<<blk, thr>>>(d_B_mo_blocks_, d_Bvv,
                                                        naux, nmo_full_, voff, NV); }
                const real_t one = 1.0, zero = 0.0;
                cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, NO, naux * NV, NV, &one,
                            d_t1_dev, NV, d_Bvv, NV, &zero, d_Y, NO);
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, NV * NO, NV * NV, naux, &one,
                            d_Y, NV * NO, d_Bvv, NV * NV, &zero, d_inter, NV * NO);
                cudaMemcpy(h_inter.data(), d_inter, inter_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                tracked_cudaFree(d_Bvv); tracked_cudaFree(d_Y);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_inter);
                termA_gpu = true;
            } else if (eri_vvvv_nslab_ > 1) {
                // Ship 14: per-device slab GEMM (mirror of EA Ship 12).  Each
                // slab's GEMM output [a_local, b, c, j] (col k = a_local*NV² +
                // b*NV + c) lines up with the global [a, b, c, j] layout, so
                // each D2H writes contiguously into h_inter at byte offset
                // a_starts_[d_dev]*NV²·NO·8.  The host b↔c swap accumulate
                // below is unchanged.
                const real_t one = 1.0, zero = 0.0;
                const int N = eri_vvvv_nslab_;
                std::vector<cublasHandle_t> hdl(N, nullptr);
                std::vector<real_t*>        d_t1_per(N, nullptr), d_inter_per(N, nullptr);
                int saved = 0; cudaGetDevice(&saved);
                for (int d_dev = 0; d_dev < N; ++d_dev) {
                    const int an = a_ends_[d_dev] - a_starts_[d_dev];
                    if (an <= 0) continue;
                    cudaSetDevice(d_dev);
                    cublasCreate(&hdl[d_dev]);
                    cudaMalloc(&d_t1_per[d_dev], (size_t)NO * NV * sizeof(real_t));
                    cudaMemcpyAsync(d_t1_per[d_dev], h_t1.data(),
                                    (size_t)NO * NV * sizeof(real_t),
                                    cudaMemcpyHostToDevice, 0);
                    const size_t slab_inter_sz = (size_t)an * NV * NV * NO;
                    cudaMalloc(&d_inter_per[d_dev], slab_inter_sz * sizeof(real_t));
                    cublasDgemm(hdl[d_dev], CUBLAS_OP_T, CUBLAS_OP_N,
                                NO, an * NV * NV, NV, &one,
                                d_t1_per[d_dev],          NV,
                                d_eri_vvvv_slabs_[d_dev], NV,
                                &zero,
                                d_inter_per[d_dev],       NO);
                    cudaMemcpyAsync(h_inter.data()
                                        + (size_t)a_starts_[d_dev] * NV * NV * NO,
                                    d_inter_per[d_dev],
                                    slab_inter_sz * sizeof(real_t),
                                    cudaMemcpyDeviceToHost, 0);
                }
                for (int d_dev = 0; d_dev < N; ++d_dev) {
                    const int an = a_ends_[d_dev] - a_starts_[d_dev];
                    if (an <= 0) continue;
                    cudaSetDevice(d_dev);
                    cudaDeviceSynchronize();
                    cudaFree(d_t1_per[d_dev]);
                    cudaFree(d_inter_per[d_dev]);
                    cublasDestroy(hdl[d_dev]);
                }
                cudaSetDevice(saved);
                termA_gpu = true;
            } else {
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_t1_dev = nullptr, *d_inter = nullptr;
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_inter,  inter_sz   * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                // C_rm[(a,b,c), j] = Σ_d h_vvvv[(a,b,c), d] · t1[j, d]
                // col-major: C_col[m=NO, n=NV³] = op_T(t1)·op_N(h_vvvv), k=NV.
                //   A = t1 row-major [j,d] = col-major [d,j] lda=NV, op_T → shape (NO, NV)
                //   B = h_vvvv row-major [(a,b,c),d] = col-major [d,(a,b,c)] ldb=NV, op_N → shape (NV, NV³)
                //   C_col[NO, NV³] ldc=NO = row-major [(a,b,c), j] storage match.
                cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            NO, NV*NV*NV, NV, &one,
                            d_t1_dev,    NV,
                            d_eri_vvvv_, NV,
                            &zero,
                            d_inter,     NO);
                cudaMemcpy(h_inter.data(), d_inter, inter_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_inter);
                termA_gpu = true;
            }
            // Ship 14: self-check reads H_VVVV (= h_vvvv); unavailable in slab
            // mode → silently skip.  Single-GPU path keeps its GANSU_STEOM_BUILD_VALIDATE gate.
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE") && eri_vvvv_nslab_ <= 1) {
                real_t dmax = 0.0;
                const int asamp[2] = {0, NV - 1};
                for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                    for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                        for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                            for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                real_t v = 0.0;
                                for (int d = 0; d < NV; ++d)
                                    v += H_VVVV(a,c,b,d) * H_T1(j,d);
                                // h_inter at [a,b,c,j] = intermediate Σ_d h_vvvv[a,b,c,d]·t1[j,d]
                                // Term A wants Σ_d V[a,c,b,d]·t1[j,d] = intermediate[a,c,b,j]
                                const real_t got = h_inter[(((size_t)a*NV+c)*NV+b)*NO+j];
                                dmax = std::max(dmax, std::fabs(v - got));
                            }
                }
                std::cout << "[STEOM build self-check] Term A GEMM vs host: max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)"
                          << std::defaultfloat << std::endl;
            }
            // wvvvo_w_t1[a,b,c,j] += h_inter[a,c,b,j]  (b↔c swap on read)
            #pragma omp parallel for collapse(2)
            for (int a = 0; a < NV; ++a)
                for (int b = 0; b < NV; ++b)
                    for (int c = 0; c < NV; ++c) {
                        const size_t src_base = ((size_t)a*NV + c)*NV*NO + (size_t)b*NO;
                        const size_t dst_base = ((size_t)a*NV + b)*NV*NO + (size_t)c*NO;
                        for (int j = 0; j < NO; ++j)
                            wvvvo_w_t1[dst_base + j] += h_inter[src_base + j];
                    }
        }
#endif
        if (!termA_gpu) {
            #pragma omp parallel for collapse(2)
            for (int a = 0; a < NV; ++a)
                for (int b = 0; b < NV; ++b)
                    for (int c = 0; c < NV; ++c)
                        for (int j = 0; j < NO; ++j) {
                            real_t s = 0.0;
                            for (int d = 0; d < NV; ++d) s += H_VVVV(a,c,b,d) * H_T1(j,d);
                            wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] += s;
                        }
        }

        // Term B: - Σ_dk ovvv[k,c,b,d]·t1[k,a]·t1[j,d]
        //   factor as: intermediate1[k,c,b,j] = Σ_d ovvv[k,c,b,d]·t1[j,d]; (natural ovvv layout — no swap on Stage 1)
        //              result[a,c,b,j]        = Σ_k t1[k,a]·intermediate1[k,c,b,j];
        //              wvvvo_w_t1[a,b,c,j]   -= result[a,c,b,j]  (b↔c swap on host accumulate)
        // GPU: 2 cuBLAS GEMMs (uses d_eri_ovvv_ already on device + d_t1). Each
        // GEMM ~0.3 s at anthracene. Inter1 ~800 MB / result ~3.6 GB device scratch.
        {
            const size_t inter1_sz = (size_t)NO * NV * NV * NO;     // [k,c,b,j]
            const size_t result_sz = (size_t)NV * NV * NV * NO;     // [a,c,b,j]
            bool termB_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_t1_dev = nullptr, *d_inter1 = nullptr, *d_result = nullptr;
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_inter1, inter1_sz   * sizeof(real_t));
                tracked_cudaMalloc(&d_result, result_sz   * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                // Stage 1: inter1[(k,c,b), j] = Σ_d ovvv[(k,c,b), d] · t1[j, d]
                //   col-major: C_col[m=NO, n=NO·NV²] = op_T(t1)·op_N(ovvv), k=NV.
                //     A = t1 [j,d] row-major = col-major [d,j] lda=NV, op_T → shape (NO, NV)
                //     B = ovvv [k,c,b,d] row-major = col-major [d,(k,c,b)] ldb=NV, op_N → shape (NV, NO·NV²)
                //     C col-major [j, (k,c,b)] ldc=NO = row-major [(k,c,b), j]
                cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            NO, NO*NV*NV, NV, &one,
                            d_t1_dev,    NV,
                            d_eri_ovvv_, NV,
                            &zero,
                            d_inter1,    NO);
                // Stage 2: result[(a,c,b), j] = Σ_k t1[k, a] · inter1[(k,c,b), j]   (i.e. = Σ_k t1·inter1 over k)
                //   col-major: C_col[NV²·NO, NV] = op_N(inter1)·op_T(t1)
                //     We want output rows = (a) and cols = (c,b,j) row-major  → col-major [(c,b,j), a]
                //     A = inter1 [(k,c,b),j] row-major → col-major view: leading dim along inner axis = j
                //         Treat inter1 as row-major [k, (c,b,j)] with stride NV²·NO between k entries
                //         col-major view [(c,b,j), k] lda=NV²·NO, op_N
                //     B = t1 [k,a] row-major → col-major [a,k] ldb=NV, op_T → [k,a]
                //     C col-major [(c,b,j), a] ldc=NV²·NO = row-major [a, (c,b,j)] = [a,c,b,j]
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                            NV*NV*NO, NV, NO, &one,
                            d_inter1,  NV*NV*NO,
                            d_t1_dev,  NV,
                            &zero,
                            d_result,  NV*NV*NO);
                std::vector<real_t> h_result(result_sz, 0.0);
                cudaMemcpy(h_result.data(), d_result, result_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_inter1); tracked_cudaFree(d_result);
                termB_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int k = 0; k < NO; ++k) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVVV(k,c,b,d) * H_T1(j,d);
                                        v += H_T1(k,a) * s;
                                    }
                                    // result at [a, c, b, j]
                                    const real_t got = h_result[(((size_t)a*NV+c)*NV+b)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[STEOM build self-check] Term B 2-GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // wvvvo_w_t1[a,b,c,j] -= h_result[a,c,b,j]  (b↔c swap on read)
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c) {
                            const size_t src_base = ((size_t)a*NV + c)*NV*NO + (size_t)b*NO;
                            const size_t dst_base = ((size_t)a*NV + b)*NV*NO + (size_t)c*NO;
                            for (int j = 0; j < NO; ++j)
                                wvvvo_w_t1[dst_base + j] -= h_result[src_base + j];
                        }
            }
#endif
            if (!termB_gpu) {
                std::vector<real_t> hkbcj((size_t)NO*NV*NV*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVVV(k,c,b,d) * H_T1(j,d);
                                hkbcj[(((size_t)k*NV+b)*NV+c)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int k = 0; k < NO; ++k)
                                    s += H_T1(k,a) * hkbcj[(((size_t)k*NV+b)*NV+c)*NO+j];
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] -= s;
                            }
            }
        }

        // Term C: - Σ_dl ovvv[l,d,a,c]·t1[l,b]·t1[j,d]
        //   factor: u_C[l,a,c,j] = Σ_d ovvv[l,d,a,c]·t1[j,d]
        //   result_C[b,a,c,j] = Σ_l t1[l,b]·u_C[l,a,c,j]
        //   wvvvo_w_t1[a,b,c,j] -= result_C[b,a,c,j]   (a↔b swap on host accumulate)
        // GPU: 2 cuBLAS GEMMs. Stage 1 contracts d but with H_OVVV(l,d,a,c) =
        // h_ovvv at storage (l,d,a,c) — d is the SECOND axis (not last!) which
        // requires a Stage 1 input view that reads h_ovvv[l, d, a, c] axis order.
        // Solution: treat h_ovvv as row-major [l, (d,a,c)] with the 3D inner block
        // ordered (d, a, c).  Storage offset l·NV³ + d·NV² + a·NV + c matches
        // h_ovvv storage where natural axes are (l, axis2, axis3, axis4) = (l,d,a,c).
        {
            const size_t u_C_sz    = (size_t)NO * NV * NV * NO;       // [l,a,c,j]
            const size_t result_sz = (size_t)NV * NV * NV * NO;       // [b,a,c,j]
            bool termC_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                real_t *d_t1_dev = nullptr, *d_u_C = nullptr, *d_result = nullptr;
                tracked_cudaMalloc(&d_t1_dev, (size_t)NO * NV * sizeof(real_t));
                tracked_cudaMalloc(&d_u_C,    u_C_sz       * sizeof(real_t));
                tracked_cudaMalloc(&d_result, result_sz    * sizeof(real_t));
                cudaMemcpy(d_t1_dev, h_t1.data(), (size_t)NO * NV * sizeof(real_t), cudaMemcpyHostToDevice);
                const real_t one = 1.0, zero = 0.0;
                // Stage 1: u_C[l,a,c,j] = Σ_d h_ovvv[l,d,a,c] · t1[j,d]
                //   h_ovvv natural axes (l, axis_1, axis_2, axis_3) row-major. H_OVVV(l,d,a,c)
                //   = h_ovvv[l*NV³ + d*NV² + a*NV + c] so axis_1=d, axis_2=a, axis_3=c.
                //   Contraction is over axis_1 = d (NOT the innermost!). cuBLAS doesn't
                //   directly contract over a non-trailing axis, so use per-l strided
                //   batched DGEMM: each batch is a slab_l (row-major [d, a, c] NV×NV²).
                //   Per-batch: u_C[l,(a,c),j] = Σ_d slab_l[d, (a,c)] · t1[j,d]
                //     col-major C[m=NO, n=NV²] = op_T(t1)·op_T(slab_l), k=NV
                //     A shared: t1 col-major [d,j] lda=NV, op_T → (NO, NV)
                //     B per-l: slab_l col-major [(a,c), d] ldb=NV², op_T → (NV, NV²)
                //                strideB = NV³ (per-l jump in h_ovvv)
                //     C per-l: u_C col-major [j, (a,c)] ldc=NO,
                //                strideC = NV²·NO (per-l jump = inner block size in [l,a,c,j])
                cublasDgemmStridedBatched(
                    cublas, CUBLAS_OP_T, CUBLAS_OP_T,
                    NO, NV*NV, NV,            // per-batch: m=NO, n=NV², k=NV
                    &one,
                    d_t1_dev, NV,    /*strideA=*/0,                      // t1 shared
                    d_eri_ovvv_, NV*NV, /*strideB=*/(long long)NV*NV*NV, // ldb=NV², stride per-l = NV³
                    &zero,
                    d_u_C, NO,       /*strideC=*/(long long)NV*NV*NO,    // per-l stride = NV²·NO
                    NO);                                                  // batch count = NO
                // Stage 2: result_C[b, (a,c,j)] = Σ_l t1[l, b] · u_C[l, (a,c,j)]
                //   col-major: C_col[m=NV²·NO, n=NV, k=NO]
                //     A = u_C [l, (a,c,j)] row-major = col-major [(a,c,j), l] lda=NV²·NO, op_N → (m, k)
                //     B = t1 [l, b] row-major = col-major [b, l] ldb=NV, op_T → (k, n)
                //     C col-major [(a,c,j), b] ldc=NV²·NO = row-major [b, (a,c,j)]
                cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                            NV*NV*NO, NV, NO, &one,
                            d_u_C,    NV*NV*NO,
                            d_t1_dev, NV,
                            &zero,
                            d_result, NV*NV*NO);
                std::vector<real_t> h_result(result_sz, 0.0);
                cudaMemcpy(h_result.data(), d_result, result_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                tracked_cudaFree(d_t1_dev); tracked_cudaFree(d_u_C); tracked_cudaFree(d_result);
                termC_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int l = 0; l < NO; ++l) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVVV(l,d,a,c) * H_T1(j,d);
                                        v += H_T1(l,b) * s;
                                    }
                                    // result_C at [b, a, c, j]
                                    const real_t got = h_result[(((size_t)b*NV+a)*NV+c)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[STEOM build self-check] Term C 2-GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                // wvvvo_w_t1[a,b,c,j] -= h_result[b,a,c,j]  (a↔b swap on read)
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c) {
                            const size_t src_base = ((size_t)b*NV + a)*NV*NO + (size_t)c*NO;
                            const size_t dst_base = ((size_t)a*NV + b)*NV*NO + (size_t)c*NO;
                            for (int j = 0; j < NO; ++j)
                                wvvvo_w_t1[dst_base + j] -= h_result[src_base + j];
                        }
            }
#endif
            if (!termC_gpu) {
                std::vector<real_t> hlacj((size_t)NO*NV*NV*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int l = 0; l < NO; ++l)
                    for (int a = 0; a < NV; ++a)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVVV(l,d,a,c) * H_T1(j,d);
                                hlacj[(((size_t)l*NV+a)*NV+c)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int l = 0; l < NO; ++l)
                                    s += H_T1(l,b) * hlacj[(((size_t)l*NV+a)*NV+c)*NO+j];
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] -= s;
                            }
            }
        }

        // Term D: + Σ_dkl ovov[k,c,l,d]·τ[k,l,a,b]·t1[j,d]   (τ = t2 + t1·t1)
        //   factor: u'[k,l,c,j] = Σ_d ovov[k,c,l,d]·t1[j,d]; then result[a,b,c,j] += Σ_{kl} u'·τ
        // GPU path: single GEMM of shape (m=NV·NO, n=NV², k=NO²) with
        //   row-major C[(a,b),(c,j)] = Σ_{(k,l)} τ[(k,l),(a,b)] · u'[(k,l),(c,j)]
        //   col-major: C_col[(c,j),(a,b)] = u'_col·τ^T_col, op_A=N, op_B=T,
        //   lda=NV·NO (u' (c,j) outer-row), ldb=NV² (τ (a,b) outer-row).
        // canonical_skip ON was a +824 s Term D wall at anthracene (host loop
        // had O(NV³·NO³) cache-unfriendly H_T2(k,l,...) stride = full L1 miss).
        // Output (h_termD = NV³·NO doubles, anthracene 3.6 GB) is omp += into
        // wvvvo_w_t1.  Multi-GPU n-slab over the NV² (a,b) axis is bit-identical
        // to single-GPU (each device gets a contiguous (a,b) column slab).
        {
            const size_t u_klcj_sz   = (size_t)NO*NO*NV*NO;
            const size_t tau_klab_sz = (size_t)NO*NO*NV*NV;
            const size_t termD_sz    = (size_t)NV*NV*NV*NO;
            bool termD_gpu = false;
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                std::vector<real_t> h_u_klcj(u_klcj_sz, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                h_u_klcj[(((size_t)k*NO+l)*NV+c)*NO+j] = s;
                            }
                std::vector<real_t> h_tau_klab(tau_klab_sz, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int l = 0; l < NO; ++l)
                        for (int a = 0; a < NV; ++a)
                            for (int b = 0; b < NV; ++b)
                                h_tau_klab[(((size_t)k*NO+l)*NV+a)*NV+b] =
                                    H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
                std::vector<real_t> h_termD(termD_sz, 0.0);
                if (build_gpus_ > 1) {
                    multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_T,
                                         NV*NO, NV*NV, NO*NO,
                                         h_u_klcj.data(),   u_klcj_sz,   NV*NO,
                                         h_tau_klab.data(), tau_klab_sz, NV*NV,
                                         h_termD.data());
                } else {
                    cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
                    real_t *d_u = nullptr, *d_tau = nullptr, *d_C = nullptr;
                    tracked_cudaMalloc(&d_u,   u_klcj_sz   * sizeof(real_t));
                    tracked_cudaMalloc(&d_tau, tau_klab_sz * sizeof(real_t));
                    tracked_cudaMalloc(&d_C,   termD_sz    * sizeof(real_t));
                    cudaMemcpy(d_u,   h_u_klcj.data(),   u_klcj_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_tau, h_tau_klab.data(), tau_klab_sz * sizeof(real_t), cudaMemcpyHostToDevice);
                    const real_t one = 1.0, zero = 0.0;
                    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                NV*NO, NV*NV, NO*NO, &one,
                                d_u, NV*NO, d_tau, NV*NV, &zero, d_C, NV*NO);
                    cudaMemcpy(h_termD.data(), d_C, termD_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
                    tracked_cudaFree(d_u); tracked_cudaFree(d_tau); tracked_cudaFree(d_C);
                }
                termD_gpu = true;
                if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                    real_t dmax = 0.0;
                    const int asamp[2] = {0, NV - 1};
                    for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                        for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                            for (int c = 0; c < NV; c += (NV/2>0?NV/2:1))
                                for (int j = 0; j < NO; j += (NO/2>0?NO/2:1)) {
                                    real_t v = 0.0;
                                    for (int k = 0; k < NO; ++k) for (int l = 0; l < NO; ++l) {
                                        real_t s = 0.0;
                                        for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                        v += s * (H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b));
                                    }
                                    const real_t got = h_termD[(((size_t)a*NV+b)*NV+c)*NO+j];
                                    dmax = std::max(dmax, std::fabs(v - got));
                                }
                    }
                    std::cout << "[STEOM build self-check] Term D GEMM vs host: max|Δ| = "
                              << std::scientific << dmax << " (expect ≤1e-11)"
                              << std::defaultfloat << std::endl;
                }
                #pragma omp parallel for
                for (size_t i = 0; i < termD_sz; ++i) wvvvo_w_t1[i] += h_termD[i];
            }
#endif
            if (!termD_gpu) {
                std::vector<real_t> u((size_t)NO*NV*NO*NO, 0.0);
                #pragma omp parallel for collapse(2)
                for (int k = 0; k < NO; ++k)
                    for (int c = 0; c < NV; ++c)
                        for (int l = 0; l < NO; ++l)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int d = 0; d < NV; ++d) s += H_OVOV(k,c,l,d) * H_T1(j,d);
                                u[(((size_t)k*NV+c)*NO+l)*NO+j] = s;
                            }
                #pragma omp parallel for collapse(2)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        for (int c = 0; c < NV; ++c)
                            for (int j = 0; j < NO; ++j) {
                                real_t s = 0.0;
                                for (int k = 0; k < NO; ++k)
                                    for (int l = 0; l < NO; ++l) {
                                        const real_t tau_klab = H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
                                        s += u[(((size_t)k*NV+c)*NO+l)*NO+j] * tau_klab;
                                    }
                                wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j] += s;
                            }
            }
        }
    }

    // Wvvvv (EA-side). The dominant term (Σ_kl ovov[k,c,l,d]·τ[k,l,a,b], τ=t2+t1t1;
    // NV⁴ output × NO² contraction, the STEOM build hotspot) is a single GEMM:
    //   result[(ab),(cd)] = Σ_(kl) τ[(kl),(ab)]·ovov2[(kl),(cd)]
    // with τ row-major [NO²×NV²] and ovov2 = ovov repacked to [k,l,c,d] (also [NO²×NV²]).
    // cublasDgemm(N,T) over these → C col-major [cd×ab] = result row-major [ab][cd] =
    // Wvvvv layout. The t1·ovvv + bare-vvvv terms stay in the host loop. Host fallback
    // (no GPU) keeps the original k,l sum. Self-check (GANSU_STEOM_BUILD_VALIDATE) below.
    // P5 canonical-skip: entire dressed-Wvvvv block (term3 GEMM, t1 GEMM, scatter)
    // is gated below — the only consumer (Wvvvo·t1, line ~1704) reads wvvvo_w_t1
    // instead. h_Wvvvv remains an empty std::vector so the H_WVVVV macro stays in
    // scope but is never read under canonical-skip.
    std::vector<real_t> h_Wvvvv;
    // NV2/NO2 must live past the canonical-skip gate — the Wvvvo big-terms block
    // below (line ~1640) reads NV2 regardless.  Declared here at the outer scope.
    const int NV2 = NV * NV, NO2 = NO * NO;
    if (!canonical_skip_wvvvv_) {
    std::vector<real_t> wvvvv_t3;
    bool wvvvv_t3_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        std::vector<real_t> h_tau(t2_sz), h_ovov2(ovov_sz);
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int l = 0; l < NO; ++l)
                for (int a = 0; a < NV; ++a)
                    for (int b = 0; b < NV; ++b)
                        h_tau[(((size_t)k*NO+l)*NV+a)*NV+b] = H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b);
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int c = 0; c < NV; ++c)
                for (int l = 0; l < NO; ++l)
                    for (int d = 0; d < NV; ++d)
                        h_ovov2[(((size_t)k*NO+l)*NV+c)*NV+d] = H_OVOV(k,c,l,d);
        wvvvv_t3.assign((size_t)NV2 * NV2, 0.0);
        if (build_gpus_ > 1) {
            // Stage 1: distribute the ab (n=NV²) columns of C across devices.
            // C[(cd)×(ab)] col-major (ldc=NV²) = ovov2·tauᵀ; a column slab is a
            // contiguous range of wvvvv_t3 (no gather), and each column is a full
            // K=NO² contraction on one device → bit-identical to single-GPU.
            multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_T,
                                 NV2, NV2, NO2,
                                 h_ovov2.data(), ovov_sz, NV2,
                                 h_tau.data(),   t2_sz,   NV2,
                                 wvvvv_t3.data());
        } else {
            cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
            real_t *d_tau = nullptr, *d_ovov2 = nullptr, *d_t3 = nullptr;
            tracked_cudaMalloc(&d_tau,   t2_sz   * sizeof(real_t));
            tracked_cudaMalloc(&d_ovov2, ovov_sz * sizeof(real_t));
            tracked_cudaMalloc(&d_t3,    (size_t)NV2 * NV2 * sizeof(real_t));
            cudaMemcpy(d_tau,   h_tau.data(),   t2_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_ovov2, h_ovov2.data(), ovov_sz * sizeof(real_t), cudaMemcpyHostToDevice);
            const real_t one = 1.0, zero = 0.0;
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, NV2, NV2, NO2, &one,
                        d_ovov2, NV2, d_tau, NV2, &zero, d_t3, NV2);
            cudaMemcpy(wvvvv_t3.data(), d_t3, (size_t)NV2 * NV2 * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(d_tau); tracked_cudaFree(d_ovov2); tracked_cudaFree(d_t3);
        }
        wvvvv_t3_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            const int asamp[2] = {0, NV - 1};
            for (int ai = 0; ai < 2; ++ai) { const int a = asamp[ai];
                for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                    for (int c = 0; c < NV; ++c) for (int d = 0; d < NV; ++d) {
                        real_t v = 0.0;
                        for (int k = 0; k < NO; ++k) for (int l = 0; l < NO; ++l)
                            v += H_OVOV(k,c,l,d) * (H_T2(k,l,a,b) + H_T1(k,a)*H_T1(l,b));
                        dmax = std::max(dmax, std::fabs(v - wvvvv_t3[((size_t)(a*NV+b))*NV2 + (c*NV+d)]));
                    }
            }
            std::cout << "[STEOM build self-check] Wvvvv term3 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif

    h_Wvvvv.assign(vvvv_sz, 0.0);
    // GPU GEMM port of the two Wvvvv host t1 terms (each O(NV⁴·NO)). Both are the
    // SAME contraction C[x,(p,q,r)] = Σ_k t1(k,x)·ovvv(k,p,q,r) read with two
    // different index maps: term_a = C[a,(c,b,d)], term_b = C[b,(d,a,c)].
    const size_t NV3 = (size_t)NV*NV*NV;
    std::vector<real_t> wvvvv_t1;   // [x,(p,q,r)] = x*NV³ + (p*NV²+q*NV+r)
    bool wvvvv_t1_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)NV*NO);                 // A[x,k] = t1(k,x)
        #pragma omp parallel for
        for (int x=0;x<NV;++x) for (int k=0;k<NO;++k) hA[(size_t)x*NO+k] = H_T1(k,x);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)NV*NO*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)NO*NV3*sizeof(real_t));   // h_ovvv [k,(p,q,r)]
        tracked_cudaMalloc(&dC,(size_t)NV*NV3*sizeof(real_t));   // nvir⁴
        cudaMemcpy(dA,hA.data(),(size_t)NV*NO*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,h_ovvv.data(),(size_t)NO*NV3*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        // C_rm[NV×NV³] = A_rm[NV×NO]·B_rm[NO×NV³]: cuBLAS(N,N, NV³, NV, NO, B, NV³, A, NO, C, NV³).
        cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,(int)NV3,NV,NO,&one,dB,(int)NV3,dA,NO,&zero,dC,(int)NV3);
        wvvvv_t1.assign((size_t)NV*NV3,0.0);
        cudaMemcpy(wvvvv_t1.data(),dC,(size_t)NV*NV3*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        wvvvv_t1_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
                for (int b=0;b<NV;b+=(NV/2>0?NV/2:1)) for (int d=0;d<NV;d+=(NV/2>0?NV/2:1)) {
                    real_t t=0.0; for (int k=0;k<NO;++k) t += H_OVVV(k,c,b,d)*H_T1(k,a);
                    dmax=std::max(dmax,std::fabs(t-wvvvv_t1[(size_t)a*NV3+((size_t)c*NV2+b*NV+d)]));
                }
            std::cout << "[STEOM build self-check] Wvvvv t1 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < NV; ++a)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int d = 0; d < NV; ++d) {
                    real_t v = H_VVVV(a,c,b,d);
                    if (wvvvv_t1_gpu) {
                        v -= wvvvv_t1[(size_t)a*NV3+((size_t)c*NV2+b*NV+d)];      // term_a: Σ_k ovvv(k,c,b,d)t1(k,a)
                        v -= wvvvv_t1[(size_t)b*NV3+((size_t)d*NV2+a*NV+c)];      // term_b: Σ_l ovvv(l,d,a,c)t1(l,b)
                    } else {
                        for (int k = 0; k < NO; ++k)
                            v -= H_OVVV(k,c,b,d) * H_T1(k,a);
                        for (int l = 0; l < NO; ++l)
                            v -= H_OVVV(l,d,a,c) * H_T1(l,b);
                    }
                    if (wvvvv_t3_gpu) {
                        v += wvvvv_t3[((size_t)(a * NV + b)) * NV2 + (c * NV + d)];
                    } else {
                        for (int k = 0; k < NO; ++k)
                            for (int l = 0; l < NO; ++l) {
                                real_t kcld = H_OVOV(k,c,l,d);
                                v += kcld * (H_T2(k,l,a,b) + H_T1(k,a) * H_T1(l,b));
                            }
                    }
                    h_Wvvvv[(((size_t)a * NV + b) * NV + c) * NV + d] = v;
                }
    }  // end if (!canonical_skip_wvvvv_)
    #define H_WVVVV(a,b,c,d) h_Wvvvv[(((size_t)(a) * NV + (b)) * NV + (c)) * NV + (d)]

    _bsplit("Wvvvv");
    // ---- Wvvvo big terms (the build hotspot) via 3 GEMMs ----
    // term3 = Σ_ld [2 ovvv[l,d,a,c] t2[l,j,d,b] − ovvv[l,d,a,c] t2[l,j,b,d] − ovvv[l,c,a,d] t2[l,j,d,b]]
    // term4 = −Σ_kd ovvv[k,c,b,d] t2[j,k,d,a].
    // Each is a (contract)×(free) GEMM M[NV²×KV] = A[NV²×KV]·B[KV×KV] (KV=NO·NV), with
    // ovvv/t2 repacked to [free_L × contract] / [contract × free_R]; the result is
    // scattered into Wvvvo[a,b,c,j]. 3a+3c share B=t2[l,j,d,b] → fused as (2A−A')·B.
    // The cheap terms (bare ovvv, W1·t1, ooov, Fov, Wvvvv·t1) stay in the host loop.
    const int KV = NO * NV;
    std::vector<real_t> wvvvo_big;
    bool wvvvo_big_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const size_t Asz = (size_t)NV2 * KV, Bsz = (size_t)KV * KV, Msz = (size_t)NV2 * KV;
        std::vector<real_t> hA(Asz), hAc2(Asz), hAcomb(Asz), hB(Bsz), hBp(Bsz), hA4(Asz), hB4(Bsz);
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int c = 0; c < NV; ++c)
                for (int l = 0; l < NO; ++l)
                    for (int d = 0; d < NV; ++d) {
                        const size_t o = (size_t)(a*NV+c)*KV + (l*NV+d);
                        hA[o]     = H_OVVV(l,d,a,c);                 // ovvv[l,d,a,c]
                        hAc2[o]   = H_OVVV(l,c,a,d);                 // ovvv[l,c,a,d]
                        hAcomb[o] = 2.0*hA[o] - hAc2[o];
                    }
        #pragma omp parallel for collapse(2)
        for (int l = 0; l < NO; ++l)
            for (int d = 0; d < NV; ++d)
                for (int j = 0; j < NO; ++j)
                    for (int b = 0; b < NV; ++b) {
                        const size_t o = (size_t)(l*NV+d)*KV + (j*NV+b);
                        hB[o]  = H_T2(l,j,d,b);                      // t2[l,j,d,b]
                        hBp[o] = H_T2(l,j,b,d);                      // t2[l,j,b,d]
                    }
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < NV; ++b)
            for (int c = 0; c < NV; ++c)
                for (int k = 0; k < NO; ++k)
                    for (int d = 0; d < NV; ++d)
                        hA4[(size_t)(b*NV+c)*KV + (k*NV+d)] = H_OVVV(k,c,b,d);   // ovvv[k,c,b,d]
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < NO; ++k)
            for (int d = 0; d < NV; ++d)
                for (int j = 0; j < NO; ++j)
                    for (int a = 0; a < NV; ++a)
                        hB4[(size_t)(k*NV+d)*KV + (j*NV+a)] = H_T2(j,k,d,a);     // t2[j,k,d,a]
        real_t *dA = nullptr, *dB = nullptr, *dM = nullptr;
        tracked_cudaMalloc(&dA, Asz * sizeof(real_t));
        tracked_cudaMalloc(&dB, Bsz * sizeof(real_t));
        tracked_cudaMalloc(&dM, Msz * sizeof(real_t));
        const real_t one = 1.0, zero = 0.0;
        wvvvo_big.assign((size_t)NV*NV*NV*NO, 0.0);
        // M_rm[NV²(free_L) × KV(free_R)] = A_rm[NV²×KV]·B_rm[KV×KV] (row-major trick).
        // free_bc: free_L=(b,c), free_R=(j,a) [term4]; else free_L=(a,c), free_R=(j,b) [term3].
        auto gemm_scatter = [&](const std::vector<real_t>& hAt, const std::vector<real_t>& hBt,
                                real_t coeff, bool free_bc) {
            cudaMemcpy(dA, hAt.data(), Asz * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dB, hBt.data(), Bsz * sizeof(real_t), cudaMemcpyHostToDevice);
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, KV, NV2, KV, &one,
                        dB, KV, dA, KV, &zero, dM, KV);
            std::vector<real_t> hM(Msz);
            cudaMemcpy(hM.data(), dM, Msz * sizeof(real_t), cudaMemcpyDeviceToHost);
            #pragma omp parallel for collapse(2)
            for (int a = 0; a < NV; ++a)
                for (int b = 0; b < NV; ++b)
                    for (int c = 0; c < NV; ++c)
                        for (int j = 0; j < NO; ++j) {
                            const size_t mo = free_bc
                                ? (size_t)(b*NV+c)*KV + (j*NV+a)
                                : (size_t)(a*NV+c)*KV + (j*NV+b);
                            wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j] += coeff * hM[mo];
                        }
        };
        gemm_scatter(hAcomb, hB,  1.0, false);   // 3a+3c
        gemm_scatter(hA,     hBp, -1.0, false);  // 3b
        gemm_scatter(hA4,    hB4, -1.0, true);   // term4
        tracked_cudaFree(dA); tracked_cudaFree(dB); tracked_cudaFree(dM);
        wvvvo_big_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            const int as[2] = {0, NV - 1};
            for (int ai = 0; ai < 2; ++ai) { const int a = as[ai];
                for (int b = 0; b < NV; b += (NV/2>0?NV/2:1))
                    for (int c = 0; c < NV; ++c) for (int j = 0; j < NO; ++j) {
                        real_t v = 0.0;
                        for (int l = 0; l < NO; ++l) for (int d = 0; d < NV; ++d) {
                            v += 2.0 * H_OVVV(l,d,a,c) * H_T2(l,j,d,b);
                            v -=       H_OVVV(l,d,a,c) * H_T2(l,j,b,d);
                            v -=       H_OVVV(l,c,a,d) * H_T2(l,j,d,b);
                        }
                        for (int k = 0; k < NO; ++k) for (int d = 0; d < NV; ++d)
                            v -= H_OVVV(k,c,b,d) * H_T2(j,k,d,a);
                        dmax = std::max(dmax, std::fabs(v - wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j]));
                    }
            }
            std::cout << "[STEOM build self-check] Wvvvo term3+4 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif

    // GPU GEMM port of Wvvvo term5 (THE build_dressed hotspot — was ~70% of the
    // whole build at naphthalene: O(NV³·NO³) strided/memory-bound host loop):
    //   ct[a,b,c,j] = Σ_{k,l} ooov(l,j,k,c)·(t2(l,k,b,a) + t1(l,b)·t1(k,a))
    // Single GEMM C[(j,c),(b,a)] = A·B, contract (k,l): A[(j,c),(k,l)]=ooov(l,j,k,c),
    // B[(k,l),(b,a)]=tau2(l,k,b,a)=t2(l,k,b,a)+t1(l,b)t1(k,a). Scattered into Wvvvo.
    const int JC_M5 = NO*NV, BA_N5 = NV*NV, KL_K5 = NO*NO;
    std::vector<real_t> wvvvo_t5;
    bool wvvvo_t5_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        std::vector<real_t> hA((size_t)JC_M5*KL_K5), hB((size_t)KL_K5*BA_N5);
        #pragma omp parallel for collapse(2)
        for (int j=0;j<NO;++j) for (int c=0;c<NV;++c)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hA[(size_t)(j*NV+c)*KL_K5+(k*NO+l)] = H_OOOV(l,j,k,c);
        #pragma omp parallel for collapse(2)
        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
            for (int b=0;b<NV;++b) for (int a=0;a<NV;++a)
                hB[(size_t)(k*NO+l)*BA_N5+(b*NV+a)] = H_T2(l,k,b,a) + H_T1(l,b)*H_T1(k,a);
        wvvvo_t5.assign((size_t)JC_M5*BA_N5,0.0);
        // C_cm[BA_N5×JC_M5] = dB·dA (col-major); slabbing the JC_M5 dimension across
        // build_gpus_ devices is a contiguous (j,c)-row range of row-major wvvvo_t5,
        // bit-identical to the single-GPU path (each column = full K=KL_K5 contraction).
        if (build_gpus_ > 1) {
            multi_gpu_gemm_nslab(build_gpus_, CUBLAS_OP_N, CUBLAS_OP_N,
                                 BA_N5, JC_M5, KL_K5,
                                 hB.data(), (size_t)KL_K5*BA_N5, BA_N5,
                                 hA.data(), (size_t)JC_M5*KL_K5, KL_K5,
                                 wvvvo_t5.data());
        } else {
            cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)JC_M5*KL_K5*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)KL_K5*BA_N5*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)JC_M5*BA_N5*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)JC_M5*KL_K5*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hB.data(),(size_t)KL_K5*BA_N5*sizeof(real_t),cudaMemcpyHostToDevice);
            const real_t one=1.0,zero=0.0;
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,BA_N5,JC_M5,KL_K5,&one,dB,BA_N5,dA,KL_K5,&zero,dC,BA_N5);
            cudaMemcpy(wvvvo_t5.data(),dC,(size_t)JC_M5*BA_N5*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        wvvvo_t5_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            const int as[2]={0,NV-1};
            for (int ai=0;ai<2;++ai){ const int a=as[ai];
                for (int b=0;b<NV;b+=(NV/2>0?NV/2:1))
                    for (int c=0;c<NV;c+=(NV/2>0?NV/2:1)) for (int j=0;j<NO;++j) {
                        real_t v=0.0;
                        for (int k=0;k<NO;++k) for (int l=0;l<NO;++l) {
                            real_t klc=H_OOOV(l,j,k,c);
                            v += klc*H_T2(l,k,b,a) + klc*H_T1(l,b)*H_T1(k,a);
                        }
                        dmax=std::max(dmax,std::fabs(v-wvvvo_t5[(size_t)(j*NV+c)*BA_N5+(b*NV+a)]));
                    }
            }
            std::cout << "[STEOM build self-check] Wvvvo term5 (ooov·tau2) GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif

    // GPU GEMM port of three remaining Wvvvo inner-sum hotspots
    // (canonical_skip_wvvvv_ on, anthracene profile = each ~21 G ops/scatter,
    // collectively ~50-60s host wall): items 1, 2, 5 below. Single GEMM each,
    // B-side direct (no repack: h_W1ovov / h_W1ovvo / h_t2 leading dim matches),
    // A-side small repack (NO×NV doubles). Sign applied at scatter site
    // (`v -= ct_wvvvo_N[...]`). Output kept on host as 3.6 GB std::vector each
    // (matches existing wvvvo_big / wvvvo_t5 pattern, ~22 GB host total).
    //
    //   Item 1: ct1[b, (a,j,c)] = Σ_l T1(l,b) · W1OVOV(l, a, j, c)
    //   Item 2: ct2[a, (b,c,j)] = Σ_k T1(k,a) · W1OVVO(k, b, c, j)
    //   Item 5: ct5[c, (j,a,b)] = Σ_k Fov(k,c) · T2(k, j, a, b)
    const int M_w1 = NV, N_w1 = NV*NO*NV, K_w1 = NO;
    const int M_w2 = NV, N_w2 = NV*NV*NO, K_w2 = NO;
    const int M_w5 = NV, N_w5 = NO*NV*NV, K_w5 = NO;
    std::vector<real_t> ct_wvvvo_1, ct_wvvvo_2, ct_wvvvo_5;
    bool wvvvo_1_gpu = false, wvvvo_2_gpu = false, wvvvo_5_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        // --- Item 1: A[b,l] = T1(l,b); B = h_W1ovov direct (stride k = N_w1) ---
        {
            std::vector<real_t> hA((size_t)M_w1*K_w1);
            #pragma omp parallel for
            for (int b = 0; b < NV; ++b)
                for (int l = 0; l < NO; ++l) hA[(size_t)b*K_w1 + l] = h_t1[l*NV + b];
            ct_wvvvo_1.assign((size_t)M_w1*N_w1, 0.0);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w1*K_w1*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w1*N_w1*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w1*N_w1*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w1*K_w1*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_W1ovov.data(),(size_t)K_w1*N_w1*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w1,M_w1,K_w1,&one,dB,N_w1,dA,K_w1,&zero,dC,N_w1);
            cudaMemcpy(ct_wvvvo_1.data(),dC,(size_t)M_w1*N_w1*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
            wvvvo_1_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int l=0;l<NO;++l) t += H_W1OVOV(l,a,j,c)*H_T1(l,b);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)]));
                            }
                std::cout << "[STEOM build self-check] Wvvvo item 1 (W1ovov·t1): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // --- Item 2: A[a,k] = T1(k,a); B = h_W1ovvo direct (stride k = N_w2) ---
        {
            std::vector<real_t> hA((size_t)M_w2*K_w2);
            #pragma omp parallel for
            for (int a = 0; a < NV; ++a)
                for (int k = 0; k < NO; ++k) hA[(size_t)a*K_w2 + k] = h_t1[k*NV + a];
            ct_wvvvo_2.assign((size_t)M_w2*N_w2, 0.0);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w2*K_w2*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w2*N_w2*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w2*N_w2*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w2*K_w2*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_W1ovvo.data(),(size_t)K_w2*N_w2*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w2,M_w2,K_w2,&one,dB,N_w2,dA,K_w2,&zero,dC,N_w2);
            cudaMemcpy(ct_wvvvo_2.data(),dC,(size_t)M_w2*N_w2*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
            wvvvo_2_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int k=0;k<NO;++k) t += H_W1OVVO(k,b,c,j)*H_T1(k,a);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)]));
                            }
                std::cout << "[STEOM build self-check] Wvvvo item 2 (W1ovvo·t1): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // --- Item 5: A[c,k] = Fov(k,c); B = h_t2 direct (stride k = N_w5) ---
        {
            std::vector<real_t> hA((size_t)M_w5*K_w5);
            #pragma omp parallel for
            for (int c = 0; c < NV; ++c)
                for (int k = 0; k < NO; ++k) hA[(size_t)c*K_w5 + k] = h_Fov[k*NV + c];
            ct_wvvvo_5.assign((size_t)M_w5*N_w5, 0.0);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)M_w5*K_w5*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)K_w5*N_w5*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)M_w5*N_w5*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)M_w5*K_w5*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,h_t2.data(),(size_t)K_w5*N_w5*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N_w5,M_w5,K_w5,&one,dB,N_w5,dA,K_w5,&zero,dC,N_w5);
            cudaMemcpy(ct_wvvvo_5.data(),dC,(size_t)M_w5*N_w5*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
            wvvvo_5_gpu = true;
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0;a<NV;a+=(NV/4>0?NV/4:1))
                    for (int b=0;b<NV;b+=(NV/4>0?NV/4:1))
                        for (int c=0;c<NV;c+=(NV/4>0?NV/4:1))
                            for (int j=0;j<NO;++j) {
                                real_t t = 0.0;
                                for (int k=0;k<NO;++k) t += h_Fov[k*NV + c]*H_T2(k,j,a,b);
                                dmax = std::max(dmax, std::fabs(t -
                                    ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)]));
                            }
                std::cout << "[STEOM build self-check] Wvvvo item 5 (Fov·t2): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
    }
#endif

    // Wvvvo (EA-side, 11-term)
    const size_t wvvvo_sz = (size_t)NV * NV * NV * NO;
    std::vector<real_t> h_Wvvvo(wvvvo_sz, 0.0);

    // Fused GPU assembly path — replaces the NV³·NO = 450M outer host scatter
    // (anthracene ~56 s = build_dressed-PROF Wvvvo bottleneck remainder after all 6
    // GEMMs eliminated the inner contractions). Requires all 6 prerequisite GEMMs
    // to have run on GPU. Default ON when conditions met; env
    // GANSU_STEOM_BUILD_WVVVO_HOST=1 forces host fallback (debug). When
    // canonical_skip_wvvvv_ is OFF (production default), wvvvo_w_t1 is not built
    // by the existing term-by-term path → we compute the equivalent
    //   Σ_d WVVVV(a,b,c,d)·T1(j,d) → C[(a,b,c),j]
    // via a single device GEMM (M=NV³, N=NO, K=NV) using the already-built
    // h_Wvvvv (canonical_skip OFF guarantees h_Wvvvv is populated upstream).
    // Kernel reads 7 arrays + writes d_Wvvvo, single launch, ~200 ms est. on A100
    // vs 56 s host (250×).
    const bool wvvvo_gpu_assembly_eligible =
        wvvvo_1_gpu && wvvvo_2_gpu && wvvvo_5_gpu &&
        wvvvo_big_gpu && wvvvo_t5_gpu;
    const char* env_wvvvo_host = std::getenv("GANSU_STEOM_BUILD_WVVVO_HOST");
    const bool use_wvvvo_gpu_assembly = wvvvo_gpu_assembly_eligible
                                        && !(env_wvvvo_host && env_wvvvo_host[0] == '1');
#ifndef GANSU_CPU_ONLY
    if (use_wvvvo_gpu_assembly && gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one = 1.0, zero = 0.0;
        // Step 1: ensure wvvvo_w_t1 (Σ_d WVVVV·T1) is available on host.
        // canonical_skip_wvvvv_ ON: existing term-by-term path already populated wvvvo_w_t1.
        // canonical_skip_wvvvv_ OFF: compute via single GEMM C[(a,b,c),j] = WVVVV[(a,b,c),d]·T1[d,j].
        std::vector<real_t> wvvvo_w_t1_local;  // fallback storage for non-skip path
        const real_t* wvvvo_w_t1_ptr = nullptr;
        size_t wvvvo_w_t1_n = 0;
        if (canonical_skip_wvvvv_) {
            wvvvo_w_t1_ptr = wvvvo_w_t1.data();
            wvvvo_w_t1_n   = wvvvo_w_t1.size();
        } else {
            const size_t vvvv_sz_local = (size_t)NV * NV * NV * NV;
            // Repack T1 from row-major [j,d] to row-major [d,j] so standard
            // C_rm[M,N] = A_rm·B_rm pattern applies (M=NV³, N=NO, K=NV).
            std::vector<real_t> hT1T((size_t)NV * NO);
            #pragma omp parallel for
            for (int d = 0; d < NV; ++d)
                for (int j = 0; j < NO; ++j)
                    hT1T[(size_t)d * NO + j] = h_t1[(size_t)j * NV + d];
            real_t *dWV=nullptr, *dT1=nullptr, *dCwt=nullptr;
            tracked_cudaMalloc(&dWV,  vvvv_sz_local      * sizeof(real_t));
            tracked_cudaMalloc(&dT1,  (size_t)NV * NO    * sizeof(real_t));
            tracked_cudaMalloc(&dCwt, wvvvo_sz           * sizeof(real_t));
            cudaMemcpy(dWV,  h_Wvvvv.data(), vvvv_sz_local   * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dT1,  hT1T.data(),    (size_t)NV * NO * sizeof(real_t), cudaMemcpyHostToDevice);
            const int M_wt = NV * NV * NV, N_wt = NO, K_wt = NV;
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, N_wt, M_wt, K_wt,
                        &one, dT1, N_wt, dWV, K_wt, &zero, dCwt, N_wt);
            wvvvo_w_t1_local.assign(wvvvo_sz, 0.0);
            cudaMemcpy(wvvvo_w_t1_local.data(), dCwt, wvvvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dWV); tracked_cudaFree(dT1); tracked_cudaFree(dCwt);
            wvvvo_w_t1_ptr = wvvvo_w_t1_local.data();
            wvvvo_w_t1_n   = wvvvo_w_t1_local.size();
            if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
                real_t dmax = 0.0;
                for (int a=0; a<NV; a+=(NV/4>0?NV/4:1))
                    for (int b=0; b<NV; b+=(NV/4>0?NV/4:1))
                        for (int c=0; c<NV; c+=(NV/4>0?NV/4:1))
                            for (int j=0; j<NO; ++j) {
                                real_t t = 0.0;
                                for (int d=0; d<NV; ++d) t += H_WVVVV(a,b,c,d) * H_T1(j,d);
                                dmax = std::max(dmax, std::fabs(t -
                                    wvvvo_w_t1_local[(((size_t)a*NV+b)*NV+c)*NO+j]));
                            }
                std::cout << "[STEOM build self-check] Wvvvo·t1 pre-GEMM (non-skip): max|Δ| = "
                          << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
            }
        }
        // Step 2: assembly path.  Two modes:
        //   - streaming (default ON; GANSU_STEOM_WVVVO_RESIDENT=0 disables) — alloc
        //     d_Out, init from d_eri_ovvv_ (operator-resident, no upload), then
        //     for each of the 6 host intermediates: alloc one slab, H2D, accumulator
        //     kernel into d_Out, free.  Peak transient ≈ d_Out + 1 slab = 2·NV³·NO·8B
        //     (≈ 18 GB at tetracene), down from the fused path's 8·NV³·NO·8B (≈ 75 GB).
        //   - fused (legacy) — single 7-input kernel, all slabs simultaneous.  Faster
        //     in anthracene scale but OOMs at tetracene cc-pVDZ on H200 141 GB.
        const char* env_stream = std::getenv("GANSU_STEOM_WVVVO_RESIDENT");
        const bool use_stream = !env_stream || env_stream[0] != '0';
        const int block = 256;
        const int grid = std::min<int>(static_cast<int>((wvvvo_sz + block - 1) / block), 65535);
        if (use_stream) {
            real_t* dOut = nullptr;
            tracked_cudaMalloc(&dOut, wvvvo_sz * sizeof(real_t));
            // Init d_Wvvvo[a,b,c,j] = d_eri_ovvv_[j,b,c,a] (no upload — already resident).
            steom_wvvvo_init_from_ovvv_kernel<<<grid, block>>>(d_eri_ovvv_, dOut, NO, NV);
            auto stream_accum = [&](const real_t* hv_ptr, size_t hv_n, auto kernel_launcher) {
                real_t* d_buf = nullptr;
                tracked_cudaMalloc(&d_buf, hv_n * sizeof(real_t));
                cudaMemcpy(d_buf, hv_ptr, hv_n * sizeof(real_t), cudaMemcpyHostToDevice);
                kernel_launcher(d_buf);
                tracked_cudaFree(d_buf);
            };
            stream_accum(ct_wvvvo_1.data(), ct_wvvvo_1.size(), [&](real_t* p){
                steom_wvvvo_accum_minus_ct1_kernel<<<grid, block>>>(p, dOut, NO, NV);
            });
            stream_accum(ct_wvvvo_2.data(), ct_wvvvo_2.size(), [&](real_t* p){
                steom_wvvvo_accum_same_layout_kernel<<<grid, block>>>(p, dOut, real_t(-1.0), NO, NV);
            });
            stream_accum(wvvvo_big.data(), wvvvo_big.size(), [&](real_t* p){
                steom_wvvvo_accum_same_layout_kernel<<<grid, block>>>(p, dOut, real_t(+1.0), NO, NV);
            });
            stream_accum(wvvvo_t5.data(), wvvvo_t5.size(), [&](real_t* p){
                steom_wvvvo_accum_plus_wt5_kernel<<<grid, block>>>(p, dOut, NO, NV);
            });
            stream_accum(ct_wvvvo_5.data(), ct_wvvvo_5.size(), [&](real_t* p){
                steom_wvvvo_accum_minus_ct5_kernel<<<grid, block>>>(p, dOut, NO, NV);
            });
            stream_accum(wvvvo_w_t1_ptr, wvvvo_w_t1_n, [&](real_t* p){
                steom_wvvvo_accum_same_layout_kernel<<<grid, block>>>(p, dOut, real_t(+1.0), NO, NV);
            });
            cudaDeviceSynchronize();
            cudaMemcpy(h_Wvvvo.data(), dOut, wvvvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dOut);
        } else {
            // Legacy fused path — 8 simultaneous device buffers.
            const size_t ovvv_sz = (size_t)NO * NV * NV * NV;
            real_t *dOvvv=nullptr, *dC1=nullptr, *dC2=nullptr, *dC5=nullptr;
            real_t *dBig=nullptr, *dT5=nullptr, *dWt1=nullptr, *dOut=nullptr;
            tracked_cudaMalloc(&dOvvv, ovvv_sz              * sizeof(real_t));
            tracked_cudaMalloc(&dC1,   ct_wvvvo_1.size()    * sizeof(real_t));
            tracked_cudaMalloc(&dC2,   ct_wvvvo_2.size()    * sizeof(real_t));
            tracked_cudaMalloc(&dC5,   ct_wvvvo_5.size()    * sizeof(real_t));
            tracked_cudaMalloc(&dBig,  wvvvo_big.size()     * sizeof(real_t));
            tracked_cudaMalloc(&dT5,   wvvvo_t5.size()      * sizeof(real_t));
            tracked_cudaMalloc(&dWt1,  wvvvo_w_t1_n        * sizeof(real_t));
            tracked_cudaMalloc(&dOut,  wvvvo_sz             * sizeof(real_t));
            cudaMemcpy(dOvvv, h_ovvv.data(),     ovvv_sz            * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dC1,   ct_wvvvo_1.data(), ct_wvvvo_1.size()  * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dC2,   ct_wvvvo_2.data(), ct_wvvvo_2.size()  * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dC5,   ct_wvvvo_5.data(), ct_wvvvo_5.size()  * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dBig,  wvvvo_big.data(),  wvvvo_big.size()   * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dT5,   wvvvo_t5.data(),   wvvvo_t5.size()    * sizeof(real_t), cudaMemcpyHostToDevice);
            cudaMemcpy(dWt1,  wvvvo_w_t1_ptr,    wvvvo_w_t1_n       * sizeof(real_t), cudaMemcpyHostToDevice);
            steom_wvvvo_fused_assembly_kernel<<<grid, block>>>(
                dOvvv, dC1, dC2, dC5, dBig, dT5, dWt1, dOut, NO, NV);
            cudaDeviceSynchronize();
            cudaMemcpy(h_Wvvvo.data(), dOut, wvvvo_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(dOvvv); tracked_cudaFree(dC1); tracked_cudaFree(dC2);
            tracked_cudaFree(dC5);   tracked_cudaFree(dBig); tracked_cudaFree(dT5);
            tracked_cudaFree(dWt1);  tracked_cudaFree(dOut);
        }
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            for (int a=0; a<NV; a+=(NV/4>0?NV/4:1))
                for (int b=0; b<NV; b+=(NV/4>0?NV/4:1))
                    for (int c=0; c<NV; c+=(NV/4>0?NV/4:1))
                        for (int j=0; j<NO; ++j) {
                            real_t ref = H_OVVV(j,b,c,a);
                            ref -= ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)];
                            ref -= ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)];
                            ref += wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j];
                            ref += wvvvo_t5[(size_t)(j*NV+c)*BA_N5 + (b*NV+a)];
                            ref -= ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)];
                            ref += wvvvo_w_t1_ptr[(((size_t)a*NV+b)*NV+c)*NO+j];
                            const real_t gpu = h_Wvvvo[(((size_t)a*NV+b)*NV+c)*NO+j];
                            dmax = std::max(dmax, std::fabs(gpu - ref));
                        }
            std::cout << "[STEOM build self-check] Wvvvo fused assembly GPU vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-13)" << std::defaultfloat << std::endl;
        }
    } else
#endif
    {
        #pragma omp parallel for collapse(2)
        for (int a = 0; a < NV; ++a)
            for (int b = 0; b < NV; ++b)
                for (int c = 0; c < NV; ++c)
                    for (int j = 0; j < NO; ++j) {
                        real_t v = H_OVVV(j,b,c,a);
                        if (wvvvo_1_gpu) {
                            v -= ct_wvvvo_1[(size_t)b*N_w1 + ((size_t)a*NO*NV + j*NV + c)];
                        } else {
                            for (int l = 0; l < NO; ++l)
                                v -= H_W1OVOV(l,a,j,c) * H_T1(l,b);
                        }
                        if (wvvvo_2_gpu) {
                            v -= ct_wvvvo_2[(size_t)a*N_w2 + ((size_t)b*NV*NO + c*NO + j)];
                        } else {
                            for (int k = 0; k < NO; ++k)
                                v -= H_W1OVVO(k,b,c,j) * H_T1(k,a);
                        }
                        if (wvvvo_big_gpu) {
                            v += wvvvo_big[(((size_t)a*NV+b)*NV+c)*NO+j];   // term3 + term4 (GEMM)
                        } else {
                            for (int l = 0; l < NO; ++l)
                                for (int d = 0; d < NV; ++d) {
                                    v += 2.0 * H_OVVV(l,d,a,c) * H_T2(l,j,d,b);
                                    v -=       H_OVVV(l,d,a,c) * H_T2(l,j,b,d);
                                    v -=       H_OVVV(l,c,a,d) * H_T2(l,j,d,b);
                                }
                            for (int k = 0; k < NO; ++k)
                                for (int d = 0; d < NV; ++d)
                                    v -= H_OVVV(k,c,b,d) * H_T2(j,k,d,a);
                        }
                        if (wvvvo_t5_gpu) {
                            v += wvvvo_t5[(size_t)(j*NV+c)*BA_N5 + (b*NV+a)];   // term5 (GEMM)
                        } else {
                            for (int k = 0; k < NO; ++k)
                                for (int l = 0; l < NO; ++l) {
                                    real_t klc_lj = H_OOOV(l,j,k,c);
                                    v += klc_lj * H_T2(l,k,b,a);
                                    v += klc_lj * H_T1(l,b) * H_T1(k,a);
                                }
                        }
                        if (wvvvo_5_gpu) {
                            v -= ct_wvvvo_5[(size_t)c*N_w5 + ((size_t)j*NV*NV + a*NV + b)];
                        } else {
                            for (int k = 0; k < NO; ++k)
                                v -= h_Fov[k*NV + c] * H_T2(k,j,a,b);
                        }
                        if (canonical_skip_wvvvv_) {
                            // P5 canonical-skip: Σ_d Wvvvv[a,b,c,d]·t1[j,d] precomputed
                            // term-by-term above without materialising nvir⁴ Wvvvv.
                            v += wvvvo_w_t1[(((size_t)a*NV+b)*NV+c)*NO+j];
                        } else {
                            for (int d = 0; d < NV; ++d)
                                v += H_WVVVV(a,b,c,d) * H_T1(j,d);
                        }
                        h_Wvvvo[(((size_t)a * NV + b) * NV + c) * NO + j] = v;
                    }
    }

    _bsplit("Wvvvo");
    // Upload all 11 intermediates to device
    const size_t loo_sz = (size_t)NO * NO;
    const size_t lvv_sz = (size_t)NV * NV;
    const size_t fov_sz = (size_t)NO * NV;

    tracked_cudaMalloc(&d_Loo_,   loo_sz   * sizeof(real_t));
    cudaMemcpy(d_Loo_,   h_Loo.data(),   loo_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Lvv_,   lvv_sz   * sizeof(real_t));
    cudaMemcpy(d_Lvv_,   h_Lvv.data(),   lvv_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Fov_,   fov_sz   * sizeof(real_t));
    cudaMemcpy(d_Fov_,   h_Fov.data(),   fov_sz   * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Woooo_, oooo_sz  * sizeof(real_t));
    cudaMemcpy(d_Woooo_, h_Woooo.data(), oooo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wooov_, ooov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wooov_, h_Wooov.data(), ooov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovov_, ovov_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovov_, h_Wovov.data(), ovov_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovvo_, ovvo_sz  * sizeof(real_t));
    cudaMemcpy(d_Wovvo_, h_Wovvo.data(), ovvo_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wovoo_, wovoo_sz * sizeof(real_t));
    cudaMemcpy(d_Wovoo_, h_Wovoo.data(), wovoo_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    tracked_cudaMalloc(&d_Wvovv_, wvovv_sz * sizeof(real_t));
    cudaMemcpy(d_Wvovv_, h_Wvovv.data(), wvovv_sz * sizeof(real_t), cudaMemcpyHostToDevice);
    if (!canonical_skip_wvvvv_) {
        tracked_cudaMalloc(&d_Wvvvv_, vvvv_sz  * sizeof(real_t));
        cudaMemcpy(d_Wvvvv_, h_Wvvvv.data(), vvvv_sz  * sizeof(real_t), cudaMemcpyHostToDevice);
    }  // canonical-skip: d_Wvvvv_ stays nullptr (verbose norm guards via if (d_Wvvvv_))
    tracked_cudaMalloc(&d_Wvvvo_, wvvvo_sz * sizeof(real_t));
    cudaMemcpy(d_Wvvvo_, h_Wvvvo.data(), wvvvo_sz * sizeof(real_t), cudaMemcpyHostToDevice);

    _bsplit("H2D push");
    std::cout << "  STEOM-CCSD bar-H intermediates built (PySCF union of IP+EA: 11 quantities)." << std::endl;

    #undef H_WOOOV
    #undef H_W1OVOV
    #undef H_W1OVVO
    #undef H_WVVVV
}


// ==================================================================
//  build_x_matrices — sub-phase 3.4: X(MI) and X(EA) = active R1 inverse.
//
//  CFOUR `renormalize` (steom.cxx lines 23-49):
//    R1_active[m_NTO, n_root] = R1^(n_root)[active_*_idx[m_NTO]]
//    X = inverse of R1_active (n_act × n_act matrix)
//
//  For canonical STEOM (CIS-NTO trans = identity over active subset), this
//  is what GANSU needs. The matrix is tiny (n_act ≤ ~20 typical, ≤ ~10 for
//  H2O/N2 test systems), so a hand-rolled Gauss-Jordan elimination is more
//  than fast enough and avoids pulling in LAPACK for this tiny inversion.
// ==================================================================
namespace {

// Layer 2 — Moore-Penrose (pseudo)inverse of a small n×n matrix via SVD, written
// back in place (row-major). Returns the 2-norm condition number and numerical
// rank so the caller can flag an ill-conditioned / rank-deficient active R1 block.
//
// Rationale: the STEOM "signed unit matrix" renormalisation builds U = active R1
// (U[m,μ] = r^(μ)_m) and X = U^{-1}; it is only well posed when the selected IP/EA
// roots SPAN the active space (U full rank, modest condition number). The former
// Gauss-Jordan inverse threw on |pivot|<1e-14 and otherwise returned an exploding
// inverse for near-degenerate (D2h) roots → corrupt Ŝ → spurious low STEOM roots.
// The SVD pinv (a) never throws, (b) truncates genuine null directions instead of
// dividing by ~0, and (c) exposes cond(U)/rank so a deficient active space (the
// real defect, fixed by span-based root selection in Layer 1) is visible, not silent.
// n is small (≤ ~32) so the O(n^3) SVD is negligible.
struct InvReport { double cond; int rank; };
InvReport invert_small_matrix_inplace(real_t* A, int n) {
    Eigen::MatrixXd M(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            M(i, j) = static_cast<double>(A[(size_t)i * n + j]);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const Eigen::VectorXd& s = svd.singularValues();          // descending order
    const double smax = (s.size() > 0) ? s(0)            : 0.0;
    const double smin = (s.size() > 0) ? s(s.size() - 1) : 0.0;
    const double tol  = smax * n * std::numeric_limits<double>::epsilon();

    Eigen::VectorXd sinv(s.size());
    int rank = 0;
    for (int k = 0; k < s.size(); ++k) {
        if (s(k) > tol) { sinv(k) = 1.0 / s(k); ++rank; }     // keep
        else            { sinv(k) = 0.0;        }             // truncate null direction
    }
    const Eigen::MatrixXd pinv =
        svd.matrixV() * sinv.asDiagonal() * svd.matrixU().transpose();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            A[(size_t)i * n + j] = static_cast<real_t>(pinv(i, j));

    InvReport rep;
    rep.cond = (smin > 0.0) ? (smax / smin) : std::numeric_limits<double>::infinity();
    rep.rank = rank;
    return rep;
}

}  // namespace

void STEOMCCSDOperator::build_x_matrices(const real_t* h_R1_IP_amplitudes,
                                          const real_t* h_R1_EA_amplitudes) {
    // ----- X(MI) = inverse of R1_active_IP (n_act_occ × n_act_occ) -----
    std::vector<real_t> h_X_IP((size_t)n_act_occ_ * n_act_occ_, 0.0);
    for (int m_NTO = 0; m_NTO < n_act_occ_; ++m_NTO) {
        const int m_idx = active_occ_idx_[m_NTO];
        for (int n_root = 0; n_root < n_act_occ_; ++n_root) {
            // R1^(n_root) is row n_root of h_R1_IP_amplitudes with stride nocc_active_.
            const real_t* r1 = h_R1_IP_amplitudes + (size_t)n_root * nocc_active_;
            h_X_IP[(size_t)m_NTO * n_act_occ_ + n_root] = r1[m_idx];
        }
    }
    const InvReport rep_ip = invert_small_matrix_inplace(h_X_IP.data(), n_act_occ_);
    tracked_cudaMalloc(&d_X_IP_, (size_t)n_act_occ_ * n_act_occ_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < h_X_IP.size(); ++i) d_X_IP_[i] = h_X_IP[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_X_IP_, h_X_IP.data(),
                   h_X_IP.size() * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    // ----- X(EA) = inverse of R1_active_EA (n_act_vir × n_act_vir) -----
    std::vector<real_t> h_X_EA((size_t)n_act_vir_ * n_act_vir_, 0.0);
    for (int e_NTO = 0; e_NTO < n_act_vir_; ++e_NTO) {
        const int a_idx = active_vir_idx_[e_NTO];
        for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
            const real_t* r1 = h_R1_EA_amplitudes + (size_t)e_root * nvir_;
            h_X_EA[(size_t)e_NTO * n_act_vir_ + e_root] = r1[a_idx];
        }
    }
    const InvReport rep_ea = invert_small_matrix_inplace(h_X_EA.data(), n_act_vir_);
    tracked_cudaMalloc(&d_X_EA_, (size_t)n_act_vir_ * n_act_vir_ * sizeof(real_t));
    if (!gpu::gpu_available()) {
        for (size_t i = 0; i < h_X_EA.size(); ++i) d_X_EA_[i] = h_X_EA[i];
    } else {
#ifndef GANSU_CPU_ONLY
        cudaMemcpy(d_X_EA_, h_X_EA.data(),
                   h_X_EA.size() * sizeof(real_t), cudaMemcpyHostToDevice);
#endif
    }

    std::cout << "  STEOM-CCSD X(MI)/X(EA) matrices built (CFOUR `renormalize`, "
              << "active R1 inverse, " << n_act_occ_ << "×" << n_act_occ_
              << " and " << n_act_vir_ << "×" << n_act_vir_ << ")." << std::endl;
    std::cout << std::scientific << std::setprecision(2)
              << "  [STEOM cond] active R1 (U): IP cond=" << rep_ip.cond
              << " rank=" << rep_ip.rank << "/" << n_act_occ_
              << " | EA cond=" << rep_ea.cond
              << " rank=" << rep_ea.rank << "/" << n_act_vir_
              << std::defaultfloat << std::endl;
    if (rep_ip.rank < n_act_occ_ || rep_ea.rank < n_act_vir_ ||
        rep_ip.cond > 1e8 || rep_ea.cond > 1e8) {
        std::cout << "  [STEOM cond] WARNING: active R1 is "
                  << ((rep_ip.rank < n_act_occ_ || rep_ea.rank < n_act_vir_)
                          ? "RANK-DEFICIENT" : "ill-conditioned")
                  << " — the selected IP/EA roots do not cleanly span the active "
                     "space (near-degenerate D2h roots / dropped active root). "
                     "STEOM roots from this run are unreliable; span-based root "
                     "selection (Layer 1) is required for a correct active space."
                  << std::endl;
    }
}


// ==================================================================
//  build_F_eff_oo — sub-phase 3.4: U(M,I) + F^eff_oo per CFOUR
//    `gmi_steom_rhf` (steom_intermediates.cxx lines 7-81).
//
//  U(M,I) is built per active root m by the same contractions the PySCF
//  IP-EOM σ1[i] matvec performs on r2 (spin-adapted closed-shell), with
//  r2 replaced by R2_IP^{(m)}:
//    U(M,I) = +2 Σ Fov[l,d] R2_IP^{(m)}[I,l,d]
//             -   Σ Fov[k,d] R2_IP^{(m)}[k,I,d]
//             - 2 Σ Wooov[k,l,I,d] R2_IP^{(m)}[k,l,d]
//             +   Σ Wooov[l,k,I,d] R2_IP^{(m)}[k,l,d]
//
//  Then F^eff_oo[M_idx, I] = Loo[M_idx, I] - Σ_N U(N,I) · X(N,M)
//  (active rows only — inactive rows are copied from bar Loo).
//
//  Host implementation: F^eff_oo is small (nocc² ≤ ~10⁴ doubles for valence
//  basis sets), no GPU kernel needed for sub-phase 3.4 — F^eff_oo is consumed
//  by the σ matvec in sub-phase 3.7, where the actual hot loop lives.
// ==================================================================
void STEOMCCSDOperator::build_F_eff_oo() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t fov_sz   = (size_t)NO * NV;
    const size_t ooov_sz  = (size_t)NO * NO * NO * NV;
    const size_t r2_per_m = (size_t)NO * NO * NV;

    // Pull bar-H + R2 + Loo back to host (small for sub-phase 3.4).
    std::vector<real_t> h_Fov(fov_sz);
    std::vector<real_t> h_Wooov(ooov_sz);
    std::vector<real_t> h_Loo((size_t)NO * NO);
    std::vector<real_t> h_R2_IP((size_t)n_act_occ_ * r2_per_m);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_Fov.data(),   d_Fov_,   fov_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Wooov.data(), d_Wooov_, ooov_sz  * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Loo.data(),   d_Loo_,   (size_t)NO*NO * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_R2_IP.data(), d_R2_IP_,
                   h_R2_IP.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < fov_sz;   ++i) h_Fov[i]   = d_Fov_[i];
        for (size_t i = 0; i < ooov_sz;  ++i) h_Wooov[i] = d_Wooov_[i];
        for (size_t i = 0; i < (size_t)NO*NO; ++i) h_Loo[i] = d_Loo_[i];
        for (size_t i = 0; i < h_R2_IP.size(); ++i) h_R2_IP[i] = d_R2_IP_[i];
    }

    // X(MI) pulled from device (small).
    std::vector<real_t> h_X_IP((size_t)n_act_occ_ * n_act_occ_);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_X_IP.data(), d_X_IP_,
                   h_X_IP.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < h_X_IP.size(); ++i) h_X_IP[i] = d_X_IP_[i];
    }

    // ----- Build U(M,I) = U[m_NTO_or_root, i_full_occ] (n_act_occ × nocc) -
    // Note: in CFOUR notation M is the "root index" (the n_root in our X_IP
    // 2nd index), not the m_NTO index. The U dimension is (n_act_occ × NO).
    std::vector<real_t> h_U_MI((size_t)n_act_occ_ * NO, 0.0);

    auto R2 = [&](int m_root, int i, int j, int a) -> real_t {
        return h_R2_IP[((size_t)m_root * NO + i) * NO * NV
                       + (size_t)j * NV + a];
    };
    auto W = [&](int k, int l, int i, int d) -> real_t {
        return h_Wooov[((size_t)k * NO + l) * NO * NV + (size_t)i * NV + d];
    };

    #pragma omp parallel for collapse(2) if (n_act_occ_ * NO > 16)
    for (int m_root = 0; m_root < n_act_occ_; ++m_root) {
        for (int I = 0; I < NO; ++I) {
            real_t u = 0.0;
            // +2 Σ_{l,d} Fov[l,d] R2[I,l,d]   -   Σ_{k,d} Fov[k,d] R2[k,I,d]
            for (int x = 0; x < NO; ++x) {
                for (int d = 0; d < NV; ++d) {
                    const real_t fov = h_Fov[(size_t)x * NV + d];
                    u += 2.0 * fov * R2(m_root, I, x, d);
                    u -=       fov * R2(m_root, x, I, d);
                }
            }
            // -2 Σ Wooov[k,l,I,d] R2[k,l,d] + Σ Wooov[l,k,I,d] R2[k,l,d]
            for (int k = 0; k < NO; ++k) {
                for (int l = 0; l < NO; ++l) {
                    for (int d = 0; d < NV; ++d) {
                        const real_t r2_kld = R2(m_root, k, l, d);
                        u -= 2.0 * W(k, l, I, d) * r2_kld;
                        u +=       W(l, k, I, d) * r2_kld;
                    }
                }
            }
            h_U_MI[(size_t)m_root * NO + I] = u;
        }
    }

    // ----- Build F^eff_oo: F^eff[M_idx, I] = Loo[M_idx, I] - Σ_N U(N,I) X(N,M)
    // For inactive rows of F^eff (rows not in active_occ_idx_), keep bar Loo.
    std::vector<real_t> h_F_eff_oo = h_Loo;  // copy bar Loo

    for (int m_NTO = 0; m_NTO < n_act_occ_; ++m_NTO) {
        const int m_idx = active_occ_idx_[m_NTO];
        for (int I = 0; I < NO; ++I) {
            real_t s = h_Loo[(size_t)m_idx * NO + I];
            for (int n_root = 0; n_root < n_act_occ_; ++n_root) {
                s -= h_U_MI[(size_t)n_root * NO + I]
                     * h_X_IP[(size_t)n_root * n_act_occ_ + m_NTO];
            }
            h_F_eff_oo[(size_t)m_idx * NO + I] = s;
        }
    }

    // Upload to device.
    tracked_cudaMalloc(&d_F_eff_oo_, (size_t)NO * NO * sizeof(real_t));
    tracked_cudaMalloc(&d_U_MI_,     (size_t)n_act_occ_ * NO * sizeof(real_t));
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(d_F_eff_oo_, h_F_eff_oo.data(),
                   h_F_eff_oo.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U_MI_, h_U_MI.data(),
                   h_U_MI.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i = 0; i < h_F_eff_oo.size(); ++i) d_F_eff_oo_[i] = h_F_eff_oo[i];
        for (size_t i = 0; i < h_U_MI.size();     ++i) d_U_MI_[i]     = h_U_MI[i];
    }

    std::cout << "  STEOM-CCSD F^eff_oo built (CFOUR `gmi_steom_rhf` formula; "
              << "active rows dressed, inactive rows = bar Loo)." << std::endl;
}


// ==================================================================
//  build_F_eff_vv — sub-phase 3.4 (extended): U(E,A) + F^eff_vv per
//    CFOUR `gea_steom_rhf` (steom_intermediates.cxx lines 83-157).
//
//  U(E, A) per active EA root e — PySCF EA-EOM σ1[a] r2 contributions:
//    U(e, A) = +2 Σ Fov[l,d] R2_EA^{(e)}[l, A, d]
//              -   Σ Fov[l,d] R2_EA^{(e)}[l, d, A]
//              + Σ_{l,c,d} (2 Wvovv[A,l,c,d] − Wvovv[A,l,d,c]) R2_EA^{(e)}[l,c,d]
//
//  Then F^eff_vv[A_idx, B] = Lvv[A_idx, B] + Σ_E U(E, B) · X(E, A_NTO)
//  (active rows dressed; inactive rows = bar Lvv).
//
//  Sign is +, opposite of F^eff_oo (which had −) — matches the asymmetry
//  between hole / particle absorption in the second similarity transform.
// ==================================================================
void STEOMCCSDOperator::build_F_eff_vv() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const size_t fov_sz   = (size_t)NO * NV;
    const size_t wvovv_sz = (size_t)NV * NO * NV * NV;
    const size_t r2_per_e = (size_t)NO * NV * NV;

    // Pull bar-H + R2_EA + Lvv back to host.
    std::vector<real_t> h_Fov(fov_sz);
    std::vector<real_t> h_Wvovv(wvovv_sz);
    std::vector<real_t> h_Lvv((size_t)NV * NV);
    std::vector<real_t> h_R2_EA((size_t)n_act_vir_ * r2_per_e);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_Fov.data(),   d_Fov_,   fov_sz   * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Wvovv.data(), d_Wvovv_, wvovv_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Lvv.data(),   d_Lvv_,   (size_t)NV*NV * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_R2_EA.data(), d_R2_EA_,
                   h_R2_EA.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < fov_sz;   ++i) h_Fov[i]   = d_Fov_[i];
        for (size_t i = 0; i < wvovv_sz; ++i) h_Wvovv[i] = d_Wvovv_[i];
        for (size_t i = 0; i < (size_t)NV*NV; ++i) h_Lvv[i] = d_Lvv_[i];
        for (size_t i = 0; i < h_R2_EA.size(); ++i) h_R2_EA[i] = d_R2_EA_[i];
    }

    // X(EA) pulled from device (small).
    std::vector<real_t> h_X_EA((size_t)n_act_vir_ * n_act_vir_);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(h_X_EA.data(), d_X_EA_,
                   h_X_EA.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    } else
#endif
    {
        for (size_t i = 0; i < h_X_EA.size(); ++i) h_X_EA[i] = d_X_EA_[i];
    }

    // ----- Build U(E,A) = U[e_root, A_full_vir] (n_act_vir × nvir) -----
    std::vector<real_t> h_U_EA((size_t)n_act_vir_ * NV, 0.0);

    auto R2 = [&](int e_root, int l, int a, int b) -> real_t {
        return h_R2_EA[((size_t)e_root * NO + l) * NV * NV
                       + (size_t)a * NV + b];
    };
    auto W = [&](int a, int l, int c, int d) -> real_t {
        return h_Wvovv[((size_t)a * NO + l) * NV * NV + (size_t)c * NV + d];
    };

    #pragma omp parallel for collapse(2) if (n_act_vir_ * NV > 16)
    for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
        for (int A = 0; A < NV; ++A) {
            real_t u = 0.0;
            // +2 Σ_{l,d} Fov[l,d] R2[l,A,d]  -  Σ_{l,d} Fov[l,d] R2[l,d,A]
            for (int l = 0; l < NO; ++l) {
                for (int d = 0; d < NV; ++d) {
                    const real_t fov = h_Fov[(size_t)l * NV + d];
                    u += 2.0 * fov * R2(e_root, l, A, d);
                    u -=       fov * R2(e_root, l, d, A);
                }
            }
            // +2 Σ Wvovv[A,l,c,d] R2[l,c,d]  -  Σ Wvovv[A,l,d,c] R2[l,c,d]
            for (int l = 0; l < NO; ++l) {
                for (int c = 0; c < NV; ++c) {
                    for (int d = 0; d < NV; ++d) {
                        const real_t r2_lcd = R2(e_root, l, c, d);
                        u += 2.0 * W(A, l, c, d) * r2_lcd;
                        u -=       W(A, l, d, c) * r2_lcd;
                    }
                }
            }
            h_U_EA[(size_t)e_root * NV + A] = u;
        }
    }

    // ----- Build F^eff_vv: F^eff[A_idx, B] = Lvv[A_idx, B] + Σ_E U(E,B) X(E,A_NTO)
    std::vector<real_t> h_F_eff_vv = h_Lvv;  // copy bar Lvv (inactive rows untouched)

    for (int a_NTO = 0; a_NTO < n_act_vir_; ++a_NTO) {
        const int a_idx = active_vir_idx_[a_NTO];
        for (int B = 0; B < NV; ++B) {
            real_t s = h_Lvv[(size_t)a_idx * NV + B];
            for (int e_root = 0; e_root < n_act_vir_; ++e_root) {
                s += h_U_EA[(size_t)e_root * NV + B]
                     * h_X_EA[(size_t)e_root * n_act_vir_ + a_NTO];
            }
            h_F_eff_vv[(size_t)a_idx * NV + B] = s;
        }
    }

    // Upload to device.
    tracked_cudaMalloc(&d_F_eff_vv_, (size_t)NV * NV * sizeof(real_t));
    tracked_cudaMalloc(&d_U_EA_,     (size_t)n_act_vir_ * NV * sizeof(real_t));
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaMemcpy(d_F_eff_vv_, h_F_eff_vv.data(),
                   h_F_eff_vv.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_U_EA_, h_U_EA.data(),
                   h_U_EA.size() * sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i = 0; i < h_F_eff_vv.size(); ++i) d_F_eff_vv_[i] = h_F_eff_vv[i];
        for (size_t i = 0; i < h_U_EA.size();     ++i) d_U_EA_[i]     = h_U_EA[i];
    }

    std::cout << "  STEOM-CCSD F^eff_vv built (CFOUR `gea_steom_rhf` formula; "
              << "active rows dressed, inactive rows = bar Lvv)." << std::endl;
}


// ==================================================================
//  build_W_eff_and_G — sub-phase 3.5-3.7: full W^eff dressing + dense
//  G^{1h1p}. Direct host port of Python build_g_canonical_full
//  (script/pyscf_steom_feff_reference.py, Nooijen-Bartlett 1997 Eq.34-63).
//
//  All tensors are tiny for valence basis sets; host loops mirror the
//  Python einsums one-to-one. The hot loop (Davidson) reuses the dense
//  d_G_ matvec, so no GPU kernel for the build itself.
//
//  Validation gate (H2O sto-3g, (3,2) active): lowest singlet eigenvalues
//  of d_G_ == 0.432663 / 0.496991 (Python reference, W^eff routes fixed 2026-06-20).
// ==================================================================
void STEOMCCSDOperator::build_W_eff_and_G() {
    const int NO = nocc_active_;
    const int NV = nvir_;
    const int NMo = n_act_occ_;
    const int NMv = n_act_vir_;

    // Internal block profiler (env GANSU_STEOM_BUILD_PROF=1). All host loops → chrono only.
    const char* _wgp = std::getenv("GANSU_PROGRESS");   // progress default-on; GANSU_PROGRESS=0 to quiet
    const bool bprof = (std::getenv("GANSU_STEOM_BUILD_PROF") != nullptr)
                     || !_wgp || _wgp[0] != '0';
    auto bclk = std::chrono::high_resolution_clock::now();
    auto bmark = [&](const char* name) {
        if (!bprof) return;
        const auto now = std::chrono::high_resolution_clock::now();
        std::cout << "    [W_eff_and_G-PROF] " << name << " = " << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double>(now - bclk).count() << " s"
                  << std::defaultfloat << std::endl;
        bclk = now;
    };

    // ---- pull bar-H + R2 + X to host ----
    auto pull = [&](real_t* dptr, size_t sz) {
        std::vector<real_t> h(sz);
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available())
            cudaMemcpy(h.data(), dptr, sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        else
#endif
            for (size_t i = 0; i < sz; ++i) h[i] = dptr[i];
        return h;
    };
    std::vector<real_t> Fov   = pull(d_Fov_,   (size_t)NO*NV);
    std::vector<real_t> Loo   = pull(d_Loo_,   (size_t)NO*NO);
    std::vector<real_t> Lvv   = pull(d_Lvv_,   (size_t)NV*NV);
    std::vector<real_t> Wooov = pull(d_Wooov_, (size_t)NO*NO*NO*NV);
    std::vector<real_t> Wvovv = pull(d_Wvovv_, (size_t)NV*NO*NV*NV);
    std::vector<real_t> Wovoo = pull(d_Wovoo_, (size_t)NO*NV*NO*NO);
    std::vector<real_t> Wovov = pull(d_Wovov_, (size_t)NO*NV*NO*NV);
    std::vector<real_t> Wovvo = pull(d_Wovvo_, (size_t)NO*NV*NV*NO);
    std::vector<real_t> ERIov = pull(d_eri_ovov_, (size_t)NO*NV*NO*NV);
    std::vector<real_t> R2IP  = pull(d_R2_IP_, (size_t)NMo*NO*NO*NV);
    std::vector<real_t> R2EA  = pull(d_R2_EA_, (size_t)NMv*NO*NV*NV);
    std::vector<real_t> XIP   = pull(d_X_IP_,  (size_t)NMo*NMo);
    std::vector<real_t> XEA   = pull(d_X_EA_,  (size_t)NMv*NMv);

    // ---- bar-H accessors (row-major natural order; see build_dressed_intermediates) ----
    auto fov  = [&](int k,int c){ return Fov[(size_t)k*NV+c]; };
    auto loo  = [&](int i,int j){ return Loo[(size_t)i*NO+j]; };
    auto lvv  = [&](int a,int b){ return Lvv[(size_t)a*NV+b]; };
    auto wooov= [&](int k,int l,int i,int d){ return Wooov[(((size_t)k*NO+l)*NO+i)*NV+d]; };
    auto wvovv= [&](int a,int l,int c,int d){ return Wvovv[(((size_t)a*NO+l)*NV+c)*NV+d]; };
    auto wovoo= [&](int k,int c,int l,int i){ return Wovoo[(((size_t)k*NV+c)*NO+l)*NO+i]; };
    auto wovov= [&](int k,int b,int i,int d){ return Wovov[(((size_t)k*NV+b)*NO+i)*NV+d]; };
    auto wovvo= [&](int k,int b,int c,int j){ return Wovvo[(((size_t)k*NV+b)*NV+c)*NO+j]; };
    auto eriov= [&](int k,int b,int i,int d){ return ERIov[(((size_t)k*NV+b)*NO+i)*NV+d]; };
    auto r2ip = [&](int m,int i,int j,int a){ return R2IP[(((size_t)m*NO+i)*NO+j)*NV+a]; };
    auto r2ea = [&](int e,int i,int a,int b){ return R2EA[(((size_t)e*NO+i)*NV+a)*NV+b]; };

    // ---- s_IP[m][i,j,a] = -Σ_λ R2_IP[λ][i,j,a]·X_IP[m,λ] ; s_EA[e][i,a,b] = +Σ_λ R2_EA[λ]·X_EA[e,λ]
    //  (X_IP[m_NTO,λ] = rinv_IP[λ,m_NTO] from build_x_matrices, matching build_normalized_s.)
    std::vector<real_t> sIP((size_t)NMo*NO*NO*NV, 0.0);
    std::vector<real_t> sEA((size_t)NMv*NO*NV*NV, 0.0);
    auto siP = [&](int m,int i,int j,int a)->real_t&{ return sIP[(((size_t)m*NO+i)*NO+j)*NV+a]; };
    auto seA = [&](int e,int i,int a,int b)->real_t&{ return sEA[(((size_t)e*NO+i)*NV+a)*NV+b]; };
    #pragma omp parallel for                       // over m (siP(m,·) distinct per m)
    for (int m=0;m<NMo;++m)
        for (int lam=0;lam<NMo;++lam){
            const real_t x = XIP[(size_t)m*NMo+lam];
            for (int i=0;i<NO;++i) for(int j=0;j<NO;++j) for(int a=0;a<NV;++a)
                siP(m,i,j,a) -= r2ip(lam,i,j,a)*x;
        }
    #pragma omp parallel for                       // over e (seA(e,·) distinct per e)
    for (int e=0;e<NMv;++e)
        for (int lam=0;lam<NMv;++lam){
            const real_t x = XEA[(size_t)e*NMv+lam];
            for (int i=0;i<NO;++i) for(int a=0;a<NV;++a) for(int b=0;b<NV;++b)
                seA(e,i,a,b) += r2ea(lam,i,a,b)*x;
        }

    bmark("pull + s_IP/s_EA");
    // ---- F_eff_oo (Eq.34-35) + F_eff_vv (Eq.36-37) rebuilt with normalized s ----
    std::vector<real_t> Foo = Loo;   // [NO×NO]
    #pragma omp parallel for collapse(2)           // (m,i) → Foo[mrow,i] distinct (mrow distinct per m)
    for (int m=0;m<NMo;++m)
        for (int i=0;i<NO;++i){
            const int mrow = active_occ_idx_[m];
            real_t u=0.0;
            for (int k=0;k<NO;++k) for(int c=0;c<NV;++c){
                const real_t st = 2.0*siP(m,i,k,c)-siP(m,k,i,c);
                u += fov(k,c)*st;
            }
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*siP(m,k,l,d)-siP(m,l,k,d);
                u -= wooov(k,l,i,d)*st;
            }
            Foo[(size_t)mrow*NO+i] += u;
        }
    std::vector<real_t> Fvv = Lvv;   // [NV×NV]
    #pragma omp parallel for collapse(2)           // (e,a) → Fvv[arow,a] distinct (arow distinct per e)
    for (int e=0;e<NMv;++e)
        for (int a=0;a<NV;++a){
            const int arow = active_vir_idx_[e];
            real_t u=0.0;
            for (int k=0;k<NO;++k) for(int c=0;c<NV;++c){
                const real_t st = 2.0*seA(e,k,a,c)-seA(e,k,c,a);
                u += fov(k,c)*st;
            }
            for (int l=0;l<NO;++l) for(int c=0;c<NV;++c) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,c,d)-seA(e,l,d,c);
                u += wvovv(a,l,c,d)*st;
            }
            Fvv[(size_t)arow*NV+a] += u;
        }

    bmark("F_eff_oo/vv");
    // ---- hp (Eq.38-39) ----
    std::vector<real_t> u_ma((size_t)NMo*NV, 0.0);  // [m][a]
    std::vector<real_t> u_ie((size_t)NO*NMv, 0.0);  // [i][e]
    #pragma omp parallel for collapse(2)           // (m,a) write distinct u_ma[m,a]
    for (int m=0;m<NMo;++m)
        for (int a=0;a<NV;++a){
            real_t v=0.0;
            for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                const real_t st = 2.0*siP(m,k,l,d)-siP(m,l,k,d);
                v -= wovvo(k,d,a,l)*st;
            }
            u_ma[(size_t)m*NV+a]=v;
        }
    #pragma omp parallel for collapse(2)           // (e,i) write distinct u_ie[i,e]
    for (int e=0;e<NMv;++e)
        for (int i=0;i<NO;++i){
            real_t v=0.0;
            for (int l=0;l<NO;++l) for(int c=0;c<NV;++c) for(int d=0;d<NV;++d){
                const real_t st = 2.0*seA(e,l,c,d)-seA(e,l,d,c);
                v += wovvo(i,d,c,l)*st;
            }
            u_ie[(size_t)i*NMv+e]=v;
        }

    bmark("hp");
    // ---- hhhp (Eq.42-44, bare v = eri_ovov) ----
    std::vector<real_t> u_mlid((size_t)NMo*NO*NO*NV, 0.0); // [m][l,i,d]
    std::vector<real_t> u_kmid((size_t)NO*NMo*NO*NV, 0.0); // [k][m][i,d]
    std::vector<real_t> u_klie((size_t)NO*NO*NO*NMv, 0.0); // [k,l,i][e]
    auto UMLID=[&](int m,int l,int i,int d)->real_t&{ return u_mlid[(((size_t)m*NO+l)*NO+i)*NV+d]; };
    auto UKMID=[&](int k,int m,int i,int d)->real_t&{ return u_kmid[(((size_t)k*NMo+m)*NO+i)*NV+d]; };
    auto UKLIE=[&](int k,int l,int i,int e)->real_t&{ return u_klie[(((size_t)k*NO+l)*NO+i)*NMv+e]; };
    // GPU GEMM port of the 3 hhhp intermediates (O(N⁶) eriov·siP / eriov·seA):
    //   UMLID[m,l,i,d] = Σ_{j,b}[(2·eriov(j,b,l,d)−eriov(l,b,j,d))·siP(m,i,j,b) − eriov(j,b,l,d)·siP(m,j,i,b)]
    //   UKMID[k,m,i,d] = −Σ_{j,b} eriov(j,d,k,b)·siP(m,j,i,b)
    //   UKLIE[k,l,i,e] = Σ_{a,b} eriov(k,a,l,b)·seA(e,i,a,b)
    const int H_LD=NO*NV, H_MI=NMo*NO, H_JB=NO*NV, H_KL=NO*NO, H_EI=NMv*NO, H_VV=NV*NV;
    std::vector<real_t> ct_mlid, ct_kmid, ct_klie;
    bool hhhp_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const real_t one=1.0, negone=-1.0, zero=0.0;
        {   // UMLID: 2 GEMMs C[(l,d),(m,i)] = A1·B1 − A2·B2, contract (j,b)
            std::vector<real_t> hA1((size_t)H_LD*H_JB), hB1((size_t)H_JB*H_MI),
                                hA2((size_t)H_LD*H_JB), hB2((size_t)H_JB*H_MI);
            #pragma omp parallel for collapse(2)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d)
                for (int j=0;j<NO;++j) for (int b=0;b<NV;++b) {
                    const size_t o=(size_t)(l*NV+d)*H_JB+(j*NV+b);
                    hA1[o]=2.0*eriov(j,b,l,d)-eriov(l,b,j,d); hA2[o]=eriov(j,b,l,d);
                }
            #pragma omp parallel for collapse(2)
            for (int j=0;j<NO;++j) for (int b=0;b<NV;++b)
                for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i) {
                    const size_t o=(size_t)(j*NV+b)*H_MI+(m*NO+i);
                    hB1[o]=siP(m,i,j,b); hB2[o]=siP(m,j,i,b);
                }
            real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA1,(size_t)H_LD*H_JB*sizeof(real_t));
            tracked_cudaMalloc(&dB1,(size_t)H_JB*H_MI*sizeof(real_t));
            tracked_cudaMalloc(&dA2,(size_t)H_LD*H_JB*sizeof(real_t));
            tracked_cudaMalloc(&dB2,(size_t)H_JB*H_MI*sizeof(real_t));
            tracked_cudaMalloc(&dC, (size_t)H_LD*H_MI*sizeof(real_t));
            cudaMemcpy(dA1,hA1.data(),(size_t)H_LD*H_JB*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB1,hB1.data(),(size_t)H_JB*H_MI*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dA2,hA2.data(),(size_t)H_LD*H_JB*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB2,hB2.data(),(size_t)H_JB*H_MI*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,H_MI,H_LD,H_JB,&one,   dB1,H_MI,dA1,H_JB,&zero,dC,H_MI);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,H_MI,H_LD,H_JB,&negone,dB2,H_MI,dA2,H_JB,&one, dC,H_MI);
            ct_mlid.assign((size_t)H_LD*H_MI,0.0);
            cudaMemcpy(ct_mlid.data(),dC,(size_t)H_LD*H_MI*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA1);tracked_cudaFree(dB1);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        }
        {   // UKMID: C[(k,d),(m,i)] = A·B, contract (j,b); A[(k,d),(j,b)]=eriov(j,d,k,b)
            std::vector<real_t> hA((size_t)H_LD*H_JB), hB((size_t)H_JB*H_MI);
            #pragma omp parallel for collapse(2)
            for (int k=0;k<NO;++k) for (int d=0;d<NV;++d)
                for (int j=0;j<NO;++j) for (int b=0;b<NV;++b)
                    hA[(size_t)(k*NV+d)*H_JB+(j*NV+b)] = eriov(j,d,k,b);
            #pragma omp parallel for collapse(2)
            for (int j=0;j<NO;++j) for (int b=0;b<NV;++b)
                for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i)
                    hB[(size_t)(j*NV+b)*H_MI+(m*NO+i)] = siP(m,j,i,b);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)H_LD*H_JB*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)H_JB*H_MI*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)H_LD*H_MI*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)H_LD*H_JB*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hB.data(),(size_t)H_JB*H_MI*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,H_MI,H_LD,H_JB,&one,dB,H_MI,dA,H_JB,&zero,dC,H_MI);
            ct_kmid.assign((size_t)H_LD*H_MI,0.0);
            cudaMemcpy(ct_kmid.data(),dC,(size_t)H_LD*H_MI*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        {   // UKLIE: C[(k,l),(e,i)] = A·Bᵀ, contract (a,b); A[(k,l),(a,b)]=eriov(k,a,l,b), B[(e,i),(a,b)]=seA(e,i,a,b)
            std::vector<real_t> hA((size_t)H_KL*H_VV), hB((size_t)H_EI*H_VV);
            #pragma omp parallel for collapse(2)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                for (int a=0;a<NV;++a) for (int b=0;b<NV;++b)
                    hA[(size_t)(k*NO+l)*H_VV+(a*NV+b)] = eriov(k,a,l,b);
            #pragma omp parallel for collapse(2)
            for (int e=0;e<NMv;++e) for (int i=0;i<NO;++i)
                for (int a=0;a<NV;++a) for (int b=0;b<NV;++b)
                    hB[(size_t)(e*NO+i)*H_VV+(a*NV+b)] = seA(e,i,a,b);
            real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
            tracked_cudaMalloc(&dA,(size_t)H_KL*H_VV*sizeof(real_t));
            tracked_cudaMalloc(&dB,(size_t)H_EI*H_VV*sizeof(real_t));
            tracked_cudaMalloc(&dC,(size_t)H_KL*H_EI*sizeof(real_t));
            cudaMemcpy(dA,hA.data(),(size_t)H_KL*H_VV*sizeof(real_t),cudaMemcpyHostToDevice);
            cudaMemcpy(dB,hB.data(),(size_t)H_EI*H_VV*sizeof(real_t),cudaMemcpyHostToDevice);
            cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,H_EI,H_KL,H_VV,&one,dB,H_VV,dA,H_VV,&zero,dC,H_EI);
            ct_klie.assign((size_t)H_KL*H_EI,0.0);
            cudaMemcpy(ct_klie.data(),dC,(size_t)H_KL*H_EI*sizeof(real_t),cudaMemcpyDeviceToHost);
            tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        }
        hhhp_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t d1=0.0,d2=0.0,d3=0.0;
            for (int m=0;m<NMo;m+=(NMo/2>0?NMo/2:1)) for (int l=0;l<NO;l+=(NO/2>0?NO/2:1))
                for (int i=0;i<NO;++i) for (int d=0;d<NV;d+=(NV/2>0?NV/2:1)) {
                    real_t v=0.0; for (int j=0;j<NO;++j) for (int b=0;b<NV;++b){
                        const real_t st=2.0*siP(m,i,j,b)-siP(m,j,i,b);
                        v += eriov(j,b,l,d)*st - eriov(l,b,j,d)*siP(m,i,j,b);}
                    d1=std::max(d1,std::fabs(v-ct_mlid[(size_t)(l*NV+d)*H_MI+(m*NO+i)]));
                    real_t v2=0.0; for (int j=0;j<NO;++j) for (int b=0;b<NV;++b) v2 -= eriov(j,d,l,b)*siP(m,j,i,b);
                    d2=std::max(d2,std::fabs(v2-(-ct_kmid[(size_t)(l*NV+d)*H_MI+(m*NO+i)])));
                }
            for (int e=0;e<NMv;e+=(NMv/2>0?NMv/2:1)) for (int k=0;k<NO;k+=(NO/2>0?NO/2:1))
                for (int l=0;l<NO;++l) for (int i=0;i<NO;++i) {
                    real_t v=0.0; for (int a=0;a<NV;++a) for (int b=0;b<NV;++b) v += eriov(k,a,l,b)*seA(e,i,a,b);
                    d3=std::max(d3,std::fabs(v-ct_klie[(size_t)(k*NO+l)*H_EI+(e*NO+i)]));
                }
            std::cout << "[W_eff_and_G self-check] hhhp UMLID/UKMID/UKLIE GEMM vs host: max|Δ| = "
                      << std::scientific << d1 << " / " << d2 << " / " << d3
                      << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int m=0;m<NMo;++m)
        for (int l=0;l<NO;++l) for(int i=0;i<NO;++i) for(int d=0;d<NV;++d){
            if (hhhp_gpu) { UMLID(m,l,i,d)=ct_mlid[(size_t)(l*NV+d)*H_MI+(m*NO+i)]; continue; }
            real_t v=0.0;
            for (int j=0;j<NO;++j) for(int b=0;b<NV;++b){
                const real_t st = 2.0*siP(m,i,j,b)-siP(m,j,i,b);
                v += eriov(j,b,l,d)*st - eriov(l,b,j,d)*siP(m,i,j,b);
            }
            UMLID(m,l,i,d)=v;
        }
    #pragma omp parallel for collapse(2)
    for (int m=0;m<NMo;++m)
        for (int k=0;k<NO;++k) for(int i=0;i<NO;++i) for(int d=0;d<NV;++d){
            if (hhhp_gpu) { UKMID(k,m,i,d)=-ct_kmid[(size_t)(k*NV+d)*H_MI+(m*NO+i)]; continue; }
            real_t v=0.0;
            for (int j=0;j<NO;++j) for(int b=0;b<NV;++b)
                v -= eriov(j,d,k,b)*siP(m,j,i,b);
            UKMID(k,m,i,d)=v;
        }
    #pragma omp parallel for collapse(2)
    for (int e=0;e<NMv;++e)
        for (int k=0;k<NO;++k) for(int l=0;l<NO;++l) for(int i=0;i<NO;++i){
            if (hhhp_gpu) { UKLIE(k,l,i,e)=ct_klie[(size_t)(k*NO+l)*H_EI+(e*NO+i)]; continue; }
            real_t v=0.0;
            for (int a=0;a<NV;++a) for(int b=0;b<NV;++b)
                v += eriov(k,a,l,b)*seA(e,i,a,b);
            UKLIE(k,l,i,e)=v;
        }

    bmark("hhhp");
    // ---- phph (Eq.56-58) ----
    std::vector<real_t> u_amci((size_t)NV*NMo*NV*NO, 0.0); // [a][m][c][i]
    std::vector<real_t> u_amei((size_t)NV*NMo*NMv*NO, 0.0);// [a][m][e][i]
    // (u_akei[a][k][e][i] removed 2026-06-20: the g_phph EA route is now the simplified CFOUR-ujaie
    //  assembled directly below, reusing the uakei_t4 GEMM. Saves an NV·NO·NMv·NO (~20 GB) alloc.)
    auto UAMCI=[&](int a,int m,int c,int i)->real_t&{ return u_amci[(((size_t)a*NMo+m)*NV+c)*NO+i]; };
    auto UAMEI=[&](int a,int m,int e,int i)->real_t&{ return u_amei[(((size_t)a*NMo+m)*NMv+e)*NO+i]; };
    // GPU GEMM port of UAMCI T2+T3 (the largest phph host term, O(NO³·NV³)):
    //   ct[a,c,m,i] = Σ_{l,d}[ (2·wvovv(a,l,c,d)−wvovv(a,l,d,c))·siP(m,i,l,d)
    //                          − wvovv(a,l,c,d)·siP(m,l,i,d) ]
    // Two GEMMs (different siP index order): C[(a,c),(m,i)] = A1·B1ᵀ − A2·B2ᵀ,
    // contract (l,d). Result scattered into UAMCI(a,m,c,i).
    const int AC_M = NV*NV, MI_N = NMo*NO, LD_K = NO*NV;
    std::vector<real_t> ct_uamci;
    bool uamci_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA1((size_t)AC_M*LD_K), hB1((size_t)MI_N*LD_K),
                            hA2((size_t)AC_M*LD_K), hB2((size_t)MI_N*LD_K);
        #pragma omp parallel for collapse(2)
        for (int a=0;a<NV;++a) for (int c=0;c<NV;++c)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(a*NV+c)*LD_K+(l*NV+d);
                hA1[o]=2.0*wvovv(a,l,c,d)-wvovv(a,l,d,c); hA2[o]=wvovv(a,l,c,d);
            }
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(m*NO+i)*LD_K+(l*NV+d);
                hB1[o]=siP(m,i,l,d); hB2[o]=siP(m,l,i,d);
            }
        real_t *dA1=nullptr,*dB1=nullptr,*dA2=nullptr,*dB2=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA1,(size_t)AC_M*LD_K*sizeof(real_t));
        tracked_cudaMalloc(&dB1,(size_t)MI_N*LD_K*sizeof(real_t));
        tracked_cudaMalloc(&dA2,(size_t)AC_M*LD_K*sizeof(real_t));
        tracked_cudaMalloc(&dB2,(size_t)MI_N*LD_K*sizeof(real_t));
        tracked_cudaMalloc(&dC, (size_t)AC_M*MI_N*sizeof(real_t));
        cudaMemcpy(dA1,hA1.data(),(size_t)AC_M*LD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB1,hB1.data(),(size_t)MI_N*LD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dA2,hA2.data(),(size_t)AC_M*LD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB2,hB2.data(),(size_t)MI_N*LD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,negone=-1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MI_N,AC_M,LD_K,&one,   dB1,LD_K,dA1,LD_K,&zero,dC,MI_N);
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MI_N,AC_M,LD_K,&negone,dB2,LD_K,dA2,LD_K,&one, dC,MI_N);
        ct_uamci.assign((size_t)AC_M*MI_N,0.0);
        cudaMemcpy(ct_uamci.data(),dC,(size_t)AC_M*MI_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA1);tracked_cudaFree(dB1);tracked_cudaFree(dA2);tracked_cudaFree(dB2);tracked_cudaFree(dC);
        uamci_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
                for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                        const real_t st=2.0*siP(m,i,l,d)-siP(m,l,i,d);
                        t += wvovv(a,l,c,d)*st - wvovv(a,l,d,c)*siP(m,i,l,d);
                    }
                    dmax=std::max(dmax,std::fabs(t-ct_uamci[(size_t)(a*NV+c)*MI_N+(m*NO+i)]));
                }
            std::cout << "[W_eff_and_G self-check] UAMCI T2+T3 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // GPU GEMM port of UAMCI T4 (Σ_{k,l} wovoo(k,c,l,i)·siP(m,l,k,a), O(NV²·NMo·NO³)):
    //   C[(c,i),(m,a)] = Σ_{(k,l)} A[(c,i),(k,l)]·B[(m,a),(k,l)]  (= A·Bᵀ, contract (k,l))
    //   FIX 2026-06-20 Wovoo->Wooov: A[(c,i),(k,l)] = wooov(k,l,i,c),  B[(m,a),(k,l)] = siP(m,k,l,a)
    // Result scattered (in the assembly loop) into UAMCI(a,m,c,i).
    const int CI_M4 = NV*NO, MA_N4 = NMo*NV, KL_K4 = NO*NO;
    std::vector<real_t> ct_uamci_t4;
    bool uamci_t4_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)CI_M4*KL_K4), hB((size_t)MA_N4*KL_K4);
        #pragma omp parallel for collapse(2)
        for (int c=0;c<NV;++c) for (int i=0;i<NO;++i)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hA[(size_t)(c*NO+i)*KL_K4+(k*NO+l)] = wooov(k,l,i,c);   // FIX Wooov
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int a=0;a<NV;++a)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hB[(size_t)(m*NV+a)*KL_K4+(k*NO+l)] = siP(m,k,l,a);     // FIX s_IP[m][k,l,a]
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)CI_M4*KL_K4*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)MA_N4*KL_K4*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)CI_M4*MA_N4*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)CI_M4*KL_K4*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)MA_N4*KL_K4*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MA_N4,CI_M4,KL_K4,&one,dB,KL_K4,dA,KL_K4,&zero,dC,MA_N4);
        ct_uamci_t4.assign((size_t)CI_M4*MA_N4,0.0);
        cudaMemcpy(ct_uamci_t4.data(),dC,(size_t)CI_M4*MA_N4*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        uamci_t4_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int c=0;c<NV;c+=(NV/2>0?NV/2:1)) for (int i=0;i<NO;++i)
                for (int m=0;m<NMo;++m) for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) {
                    real_t t=0.0;
                    for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                        t += wooov(k,l,i,c)*siP(m,k,l,a);
                    dmax=std::max(dmax,std::fabs(t-ct_uamci_t4[(size_t)(c*NO+i)*MA_N4+(m*NV+a)]));
                }
            std::cout << "[W_eff_and_G self-check] UAMCI T4 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int m=0;m<NMo;++m)
        for (int a=0;a<NV;++a) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i){
            real_t t=0.0;
            for (int k=0;k<NO;++k) t -= fov(k,c)*siP(m,i,k,a);                    // T1
            if (uamci_gpu) {
                t += ct_uamci[(size_t)(a*NV+c)*MI_N+(m*NO+i)];                    // T2+T3 (GEMM)
            } else {
                for (int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                    const real_t st = 2.0*siP(m,i,l,d)-siP(m,l,i,d);
                    t += wvovv(a,l,c,d)*st;                                        // T2
                    t -= wvovv(a,l,d,c)*siP(m,i,l,d);                              // T3
                }
            }
            // T4 (Wovoo->Wooov fix 2026-06-20): + Wooov[k,l,i,c] s_IP[m][k,l,a]. GEMM re-ported.
            if (uamci_t4_gpu) {
                t += ct_uamci_t4[(size_t)(c*NO+i)*MA_N4+(m*NV+a)];               // T4 (GEMM, Wooov)
            } else {
                for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                    t += wooov(k,l,i,c)*siP(m,k,l,a);                            // T4 (host, Wooov)
            }
            UAMCI(a,m,c,i)=t;
        }
    // UAKEI T4 (Σ_cd wvovv(a,k,c,d)·seA(e,i,c,d), the largest phph sub-term) as a single
    // GEMM: wvovv is [(ak)×(cd)] and seA is [(ei)×(cd)] row-major, result [(ak)×(ei)] lands
    // exactly in UAKEI layout (no repack/scatter). C_cm[N×M]=seAᵀ·wvovv → u_akei.
    std::vector<real_t> uakei_t4;   // [NV·NO × NMv·NO] = u_akei layout
    bool uakei_t4_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available() && d_Wvovv_ != nullptr) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const int M = NV * NO, Ncol = NMv * NO, K = NV * NV;
        real_t *d_seA = nullptr, *d_out = nullptr;
        tracked_cudaMalloc(&d_seA, sEA.size() * sizeof(real_t));
        tracked_cudaMalloc(&d_out, (size_t)M * Ncol * sizeof(real_t));
        cudaMemcpy(d_seA, sEA.data(), sEA.size() * sizeof(real_t), cudaMemcpyHostToDevice);
        const real_t one = 1.0, zero = 0.0;
        cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, Ncol, M, K, &one,
                    d_seA, K, d_Wvovv_, K, &zero, d_out, Ncol);
        uakei_t4.assign((size_t)M * Ncol, 0.0);
        cudaMemcpy(uakei_t4.data(), d_out, (size_t)M * Ncol * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_seA); tracked_cudaFree(d_out);
        uakei_t4_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax = 0.0;
            for (int e = 0; e < NMv; e += (NMv/2>0?NMv/2:1))
                for (int a = 0; a < NV; a += (NV/2>0?NV/2:1))
                    for (int k = 0; k < NO; ++k) for (int i = 0; i < NO; ++i) {
                        real_t t = 0.0;
                        for (int c = 0; c < NV; ++c) for (int d = 0; d < NV; ++d)
                            t += wvovv(a,k,c,d) * seA(e,i,c,d);
                        dmax = std::max(dmax, std::fabs(
                            t - uakei_t4[(((size_t)a*NO+k)*NMv+e)*NO+i]));
                    }
            std::cout << "[W_eff_and_G self-check] UAKEI T4 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)"
                      << std::defaultfloat << std::endl;
        }
    }
#endif
    // (old UAKEI T2+T3 GEMM + 4-term fallback removed 2026-06-20: the g_phph EA route is now
    // the simplified CFOUR-ujaie computed directly in the assembly below, reusing the uakei_t4
    // GEMM for its Wvovv term. u_akei[]/UAKEI() are now unused.)
    // GPU GEMM port of UAMEI T3+T4 (the last fully-host phph block, O(NMo·NMv·NV²·NO²)):
    //   T3:  Σ_{l,d} UMLID(m,l,i,d)·(2·seA(e,l,a,d)−seA(e,l,d,a))
    //   T4: −Σ_{l,d} UKMID(l,m,i,d)·seA(e,l,a,d)
    //   C[(m,i),(e,a)] = A3·B3ᵀ − A4·B4ᵀ, contract (l,d). Scattered into UAMEI(a,m,e,i).
    std::vector<real_t> ct_uamei_td;
    bool uamei_td_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const int MI=NMo*NO, EA=NMv*NV, LD=NO*NV;
        std::vector<real_t> hA3((size_t)MI*LD), hB3((size_t)EA*LD),
                            hA4((size_t)MI*LD), hB4((size_t)EA*LD);
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(m*NO+i)*LD+(l*NV+d);
                hA3[o]=UMLID(m,l,i,d); hA4[o]=UKMID(l,m,i,d);
            }
        #pragma omp parallel for collapse(2)
        for (int e=0;e<NMv;++e) for (int a=0;a<NV;++a)
            for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                const size_t o=(size_t)(e*NV+a)*LD+(l*NV+d);
                hB3[o]=2.0*seA(e,l,a,d)-seA(e,l,d,a); hB4[o]=seA(e,l,a,d);
            }
        real_t *dA3=nullptr,*dB3=nullptr,*dA4=nullptr,*dB4=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA3,(size_t)MI*LD*sizeof(real_t));
        tracked_cudaMalloc(&dB3,(size_t)EA*LD*sizeof(real_t));
        tracked_cudaMalloc(&dA4,(size_t)MI*LD*sizeof(real_t));
        tracked_cudaMalloc(&dB4,(size_t)EA*LD*sizeof(real_t));
        tracked_cudaMalloc(&dC, (size_t)MI*EA*sizeof(real_t));
        cudaMemcpy(dA3,hA3.data(),(size_t)MI*LD*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB3,hB3.data(),(size_t)EA*LD*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dA4,hA4.data(),(size_t)MI*LD*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB4,hB4.data(),(size_t)EA*LD*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,negone=-1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,EA,MI,LD,&one,   dB3,LD,dA3,LD,&zero,dC,EA);
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,EA,MI,LD,&negone,dB4,LD,dA4,LD,&one, dC,EA);
        ct_uamei_td.assign((size_t)MI*EA,0.0);
        cudaMemcpy(ct_uamei_td.data(),dC,(size_t)MI*EA*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA3);tracked_cudaFree(dB3);tracked_cudaFree(dA4);tracked_cudaFree(dB4);tracked_cudaFree(dC);
        uamei_td_gpu=true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int m=0;m<NMo;++m) for (int i=0;i<NO;++i)
                for (int e=0;e<NMv;e+=(NMv/2>0?NMv/2:1)) for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) {
                    real_t t=0.0;
                    for (int l=0;l<NO;++l) for (int d=0;d<NV;++d) {
                        const real_t st=2.0*seA(e,l,a,d)-seA(e,l,d,a);
                        t += UMLID(m,l,i,d)*st - UKMID(l,m,i,d)*seA(e,l,a,d);
                    }
                    dmax=std::max(dmax,std::fabs(t-ct_uamei_td[(size_t)(m*NO+i)*EA+(e*NV+a)]));
                }
            std::cout << "[W_eff_and_G self-check] UAMEI T3+T4 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // GPU GEMM port of UAMEI T5 (Σ_{k,l} UKLIE(k,l,i,e)·siP(m,l,k,a), contract (k,l)):
    //   C[(e,i),(m,a)] = A·Bᵀ; A[(e,i),(k,l)]=UKLIE(k,l,i,e), B[(m,a),(k,l)]=siP(m,l,k,a).
    std::vector<real_t> ct_uamei_t5;
    bool uamei_t5_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        const int EI=NMv*NO, MA=NMo*NV, KL=NO*NO;
        std::vector<real_t> hA((size_t)EI*KL), hB((size_t)MA*KL);
        #pragma omp parallel for collapse(2)
        for (int e=0;e<NMv;++e) for (int i=0;i<NO;++i)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hA[(size_t)(e*NO+i)*KL+(k*NO+l)] = UKLIE(k,l,i,e);
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int a=0;a<NV;++a)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hB[(size_t)(m*NV+a)*KL+(k*NO+l)] = siP(m,l,k,a);
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)EI*KL*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)MA*KL*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)EI*MA*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)EI*KL*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)MA*KL*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MA,EI,KL,&one,dB,KL,dA,KL,&zero,dC,MA);
        ct_uamei_t5.assign((size_t)EI*MA,0.0);
        cudaMemcpy(ct_uamei_t5.data(),dC,(size_t)EI*MA*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        uamei_t5_gpu=true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int e=0;e<NMv;e+=(NMv/2>0?NMv/2:1)) for (int i=0;i<NO;++i)
                for (int m=0;m<NMo;++m) for (int a=0;a<NV;a+=(NV/2>0?NV/2:1)) {
                    real_t t=0.0;
                    for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                        t += UKLIE(k,l,i,e)*siP(m,l,k,a);
                    dmax=std::max(dmax,std::fabs(t-ct_uamei_t5[(size_t)(e*NO+i)*MA+(m*NV+a)]));
                }
            std::cout << "[W_eff_and_G self-check] UAMEI T5 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    #pragma omp parallel for collapse(2)
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e)
        for (int a=0;a<NV;++a) for(int i=0;i<NO;++i){
            real_t t=0.0;
            for (int c=0;c<NV;++c) t += u_ma[(size_t)m*NV+c]*seA(e,i,a,c);         // T1
            for (int k=0;k<NO;++k) t -= u_ie[(size_t)k*NMv+e]*siP(m,i,k,a);        // T2
            if (uamei_td_gpu) {
                t += ct_uamei_td[(size_t)(m*NO+i)*(NMv*NV)+(e*NV+a)];              // T3+T4 (GEMM)
            } else {
                for (int l=0;l<NO;++l) for(int d=0;d<NV;++d){
                    const real_t st = 2.0*seA(e,l,a,d)-seA(e,l,d,a);
                    t += UMLID(m,l,i,d)*st;                                        // T3
                    t -= UKMID(l,m,i,d)*seA(e,l,a,d);                              // T4
                }
            }
            if (uamei_t5_gpu) {
                t += ct_uamei_t5[(size_t)(e*NO+i)*(NMo*NV)+(m*NV+a)];              // T5 (GEMM)
            } else {
                for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                    t += UKLIE(k,l,i,e)*siP(m,l,k,a);                             // T5
            }
            UAMEI(a,m,e,i)=t;
        }

    bmark("phph (UAMCI/UAKEI/UAMEI)");
    // ---- phhp (Eq.60-62) ----
    std::vector<real_t> u_bmjc((size_t)NV*NMo*NO*NV, 0.0); // [b][m][j][c]
    auto UBMJC=[&](int b,int m,int j,int c)->real_t&{ return u_bmjc[(((size_t)b*NMo+m)*NO+j)*NV+c]; };
    // GPU GEMM port of UBMJC Term1 (FIX 2026-06-20, g_phhp IP SO route -Wvovv·s):
    //   ct[b,c,m,j] = Σ_{k,d} wvovv(c,k,d,b)·siP(m,j,k,d)   (subtracted below)
    // A[(b,c),(k,d)] = wvovv(c,k,d,b), B[(m,j),(k,d)] = siP(m,j,k,d), contract (k,d).
    const int BC_M = NV*NV, MJ_N = NMo*NO, KD_K = NO*NV;
    std::vector<real_t> ct_ubmjc;
    bool ubmjc_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)BC_M*KD_K), hB((size_t)MJ_N*KD_K);
        #pragma omp parallel for collapse(2)
        for (int b=0;b<NV;++b) for (int c=0;c<NV;++c)
            for (int k=0;k<NO;++k) for (int d=0;d<NV;++d)
                hA[(size_t)(b*NV+c)*KD_K+(k*NV+d)] = wvovv(c,k,d,b);   // FIX
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int j=0;j<NO;++j)
            for (int k=0;k<NO;++k) for (int d=0;d<NV;++d)
                hB[(size_t)(m*NO+j)*KD_K+(k*NV+d)] = siP(m,j,k,d);     // FIX
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)BC_M*KD_K*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)MJ_N*KD_K*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)BC_M*MJ_N*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)BC_M*KD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)MJ_N*KD_K*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MJ_N,BC_M,KD_K,&one,dB,KD_K,dA,KD_K,&zero,dC,MJ_N);
        ct_ubmjc.assign((size_t)BC_M*MJ_N,0.0);
        cudaMemcpy(ct_ubmjc.data(),dC,(size_t)BC_M*MJ_N*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        ubmjc_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int b=0;b<NV;b+=(NV/2>0?NV/2:1)) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1))
                for (int m=0;m<NMo;++m) for (int j=0;j<NO;++j) {
                    real_t t=0.0; for (int k=0;k<NO;++k) for (int d=0;d<NV;++d) t += wvovv(c,k,d,b)*siP(m,j,k,d);
                    dmax=std::max(dmax,std::fabs(t-ct_ubmjc[(size_t)(b*NV+c)*MJ_N+(m*NO+j)]));
                }
            std::cout << "[W_eff_and_G self-check] UBMJC Term1 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // GPU GEMM port of UBMJC Term2 (FIX 2026-06-20, g_phhp IP SO route +0.5 Wooov·s):
    //   C[(b,j),(m,c)] = Σ_{(k,l)} A[(b,j),(k,l)]·B[(m,c),(k,l)]  (= A·Bᵀ, contract (k,l))
    //   A[(b,j),(k,l)] = wooov(k,l,j,b),  B[(m,c),(k,l)] = siP(m,k,l,c)   (×0.5 applied below)
    const int CJ_M2 = NV*NO, MB_N2 = NMo*NV, KL_K2 = NO*NO;
    std::vector<real_t> ct_ubmjc_t2;
    bool ubmjc_t2_gpu = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
        std::vector<real_t> hA((size_t)CJ_M2*KL_K2), hB((size_t)MB_N2*KL_K2);
        #pragma omp parallel for collapse(2)
        for (int b=0;b<NV;++b) for (int j=0;j<NO;++j)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hA[(size_t)(b*NO+j)*KL_K2+(k*NO+l)] = wooov(k,l,j,b);   // FIX
        #pragma omp parallel for collapse(2)
        for (int m=0;m<NMo;++m) for (int c=0;c<NV;++c)
            for (int k=0;k<NO;++k) for (int l=0;l<NO;++l)
                hB[(size_t)(m*NV+c)*KL_K2+(k*NO+l)] = siP(m,k,l,c);     // FIX
        real_t *dA=nullptr,*dB=nullptr,*dC=nullptr;
        tracked_cudaMalloc(&dA,(size_t)CJ_M2*KL_K2*sizeof(real_t));
        tracked_cudaMalloc(&dB,(size_t)MB_N2*KL_K2*sizeof(real_t));
        tracked_cudaMalloc(&dC,(size_t)CJ_M2*MB_N2*sizeof(real_t));
        cudaMemcpy(dA,hA.data(),(size_t)CJ_M2*KL_K2*sizeof(real_t),cudaMemcpyHostToDevice);
        cudaMemcpy(dB,hB.data(),(size_t)MB_N2*KL_K2*sizeof(real_t),cudaMemcpyHostToDevice);
        const real_t one=1.0,zero=0.0;
        cublasDgemm(cublas,CUBLAS_OP_T,CUBLAS_OP_N,MB_N2,CJ_M2,KL_K2,&one,dB,KL_K2,dA,KL_K2,&zero,dC,MB_N2);
        ct_ubmjc_t2.assign((size_t)CJ_M2*MB_N2,0.0);
        cudaMemcpy(ct_ubmjc_t2.data(),dC,(size_t)CJ_M2*MB_N2*sizeof(real_t),cudaMemcpyDeviceToHost);
        tracked_cudaFree(dA);tracked_cudaFree(dB);tracked_cudaFree(dC);
        ubmjc_t2_gpu = true;
        if (std::getenv("GANSU_STEOM_BUILD_VALIDATE")) {
            real_t dmax=0.0;
            for (int b=0;b<NV;b+=(NV/2>0?NV/2:1)) for (int j=0;j<NO;++j)
                for (int m=0;m<NMo;++m) for (int c=0;c<NV;c+=(NV/2>0?NV/2:1)) {
                    real_t t=0.0;
                    for (int k=0;k<NO;++k) for (int l=0;l<NO;++l) t += wooov(k,l,j,b)*siP(m,k,l,c);
                    dmax=std::max(dmax,std::fabs(t-ct_ubmjc_t2[(size_t)(b*NO+j)*MB_N2+(m*NV+c)]));
                }
            std::cout << "[W_eff_and_G self-check] UBMJC Term2 GEMM vs host: max|Δ| = "
                      << std::scientific << dmax << " (expect ≤1e-11)" << std::defaultfloat << std::endl;
        }
    }
#endif
    // u_bmjc FIX 2026-06-20 (g_phhp IP route from spin-orbital derivation; replaces the old
    // buggy 3-term {Fov,Wovoo,Wvovv}). Python build_g_canonical_full u_bmjc, cross-spin/Coulomb:
    //   g_phhp[b,j,i,a] += -Σ_kc Wvovv[a,k,c,b] s_IP[m][i,k,c] + 0.5 Σ_kl Wooov[k,l,i,b] s_IP[m][k,l,a]
    // GANSU layout g_phhp[b,k,j,c], root m -> k=occ_idx[m]; UBMJC loop vars (b,m,j,c) map to the
    // Python block as j=Python-i, c=Python-a (NO Fov term here - Fov lives in F_eff_oo).
    // GEMMs ct_ubmjc (Term1 -Wvovv·s) and ct_ubmjc_t2 (Term2 +0.5 Wooov·s) re-ported above.
    #pragma omp parallel for collapse(2)
    for (int m=0;m<NMo;++m)
        for (int b=0;b<NV;++b) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c){
            real_t t=0.0;
            if (ubmjc_gpu)                                                          // Term1 -Wvovv·s
                t -= ct_ubmjc[(size_t)(b*NV+c)*MJ_N+(m*NO+j)];
            else
                for (int k=0;k<NO;++k) for(int d=0;d<NV;++d)
                    t -= wvovv(c,k,d,b)*siP(m,j,k,d);
            if (ubmjc_t2_gpu)                                                       // Term2 +0.5 Wooov·s
                t += 0.5*ct_ubmjc_t2[(size_t)(b*NO+j)*MB_N2+(m*NV+c)];
            else
                for (int k=0;k<NO;++k) for(int l=0;l<NO;++l)
                    t += 0.5*wooov(k,l,j,b)*siP(m,k,l,c);
            UBMJC(b,m,j,c)=t;
        }
    // (UBKJE T2/T3 + UBMJE T3/T4 GEMMs and fallbacks removed 2026-06-20: their u_bkje/u_bmje
    //  outputs were the g_phhp EA + cross routes, both ZEROED in the assembly below since the
    //  base wovvo + UBMJC IP route already matches ORCA. UKLIE/UMLID/u_ma/u_ie stay for UAMEI.)

    bmark("phhp (UBMJC)");
    // ---- assemble g_phph[a,k,c,i] (Eq.59) and g_phhp[b,k,j,c] (Eq.63) ----
    std::vector<real_t> g_phph((size_t)NV*NO*NV*NO, 0.0);
    std::vector<real_t> g_phhp((size_t)NV*NO*NO*NV, 0.0);
    auto GPHPH=[&](int a,int k,int c,int i)->real_t&{ return g_phph[(((size_t)a*NO+k)*NV+c)*NO+i]; };
    auto GPHHP=[&](int b,int k,int j,int c)->real_t&{ return g_phhp[(((size_t)b*NO+k)*NO+j)*NV+c]; };
    #pragma omp parallel for collapse(2) schedule(static)
    for (int a=0;a<NV;++a) for(int k=0;k<NO;++k) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i)
        GPHPH(a,k,c,i)=wovov(k,a,i,c);
    #pragma omp parallel for schedule(static)              // distinct kf=active_occ_idx_[m] per m
    for (int m=0;m<NMo;++m){ const int kf=active_occ_idx_[m];
        for(int a=0;a<NV;++a) for(int c=0;c<NV;++c) for(int i=0;i<NO;++i)
            GPHPH(a,kf,c,i)+=UAMCI(a,m,c,i); }
    // u_akei (g_phph EA route = CFOUR ujaie), scattered root -> FIRST vir index af. The 6-term
    // 0.5(A+B) collapses algebraically (spinad cancellation, verified vs Python 1e-14) to 4 clean
    // contractions: GPHPH(af,k,c,i) += Σ_F seA(e,k,c,F)·fov(i,F)            [Fov]
    //   + Σ_{G,F} seA(e,k,G,F)·wvovv(c,i,G,F)                                [Wvovv, = UAKEI-T4 GEMM]
    //   + Σ_{N,F}(2seA(e,N,c,F)-seA(e,N,F,c))·wooov(i,N,k,F) - Σ_{N,F} seA(e,N,c,F)·wooov(N,i,k,F) [Wooov]
    // The Wvovv term reuses the existing uakei_t4 GEMM (Σ_cd wvovv(a,k,c,d)·seA(e,i,c,d)); with
    // (a,k,e,i)->(c,i,e,k) it equals the Wvovv contraction here.
    // (W^eff build perf) This u_akei block dominated the assemble phase: the (N,F)
    // Wooov contraction makes it O(NMv·NO³·NV²), and it ran serially (≈16 min on the
    // in-domain {0-9} cluster — the "stuck at W^eff build" wall). Parallelise over
    // (e,k): each (e,k) writes the distinct GPHPH(af,k,·,·) slab (af = active_vir_idx_[e]
    // is distinct per e), and every v keeps its serial inner-sum order ⇒ bit-exact.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int e=0;e<NMv;++e)
        for(int k=0;k<NO;++k){
            const int af=active_vir_idx_[e];
            for(int c=0;c<NV;++c) for(int i=0;i<NO;++i){
                real_t v=0.0;
                for(int F=0;F<NV;++F) v += seA(e,k,c,F)*fov(i,F);              // Fov
                if (uakei_t4_gpu) {                                            // Wvovv (GEMM reuse)
                    v += uakei_t4[(((size_t)c*NO+i)*NMv+e)*NO+k];
                } else {
                    for(int G=0;G<NV;++G) for(int F=0;F<NV;++F)
                        v += seA(e,k,G,F)*wvovv(c,i,G,F);
                }
                for(int N=0;N<NO;++N) for(int F=0;F<NV;++F){                   // Wooov (2 terms)
                    v += (2.0*seA(e,N,c,F)-seA(e,N,F,c))*wooov(i,N,k,F);
                    v -= seA(e,N,c,F)*wooov(N,i,k,F);
                }
                GPHPH(af,k,c,i)+=v;
            }
        }
    #pragma omp parallel for collapse(2) schedule(static)  // distinct (kf,cf) per (m,e)
    for (int m=0;m<NMo;++m) for(int e=0;e<NMv;++e){ const int kf=active_occ_idx_[m], cf=active_vir_idx_[e];
        for(int a=0;a<NV;++a) for(int i=0;i<NO;++i)
            GPHPH(a,kf,cf,i)+=UAMEI(a,m,e,i); }       // g_phph cross (kept, old form)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0;b<NV;++b) for(int k=0;k<NO;++k) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c)
        GPHHP(b,k,j,c)=wovvo(k,b,c,j);
    #pragma omp parallel for schedule(static)              // distinct kf per m
    for (int m=0;m<NMo;++m){ const int kf=active_occ_idx_[m];
        for(int b=0;b<NV;++b) for(int j=0;j<NO;++j) for(int c=0;c<NV;++c)
            GPHHP(b,kf,j,c)+=UBMJC(b,m,j,c); }        // g_phhp IP route (SO-derived)
    // u_bkje (g_phhp EA) and u_bmje (g_phhp cross) ZEROED 2026-06-20: old forms buggy; base+IP
    // (UBMJC) already matches ORCA on H2O/CH2O (n->pi* 0.04-0.08 eV). The correct small EA route
    // is derived in memory pt41 but adding it lands at the 2nd-order {e^S}, below ORCA (pt42).

    bmark("assemble g_phph/g_phhp");
    // ---- G^{1h1p} singlet: row=i*NV+a, col=j*NV+b ----
    //  G = F_eff_vv δ_ij − F_eff_oo δ_ab + 2 g_phhp[b,j,i,a] − g_phph[a,j,b,i]
    std::vector<real_t> Gmat((size_t)total_dim_*total_dim_, 0.0);
    #pragma omp parallel for collapse(2) schedule(static)  // distinct row=(i,a) per thread
    for (int i=0;i<NO;++i) for(int a=0;a<NV;++a){
        const int row=i*NV+a;
        for (int j=0;j<NO;++j) for(int b=0;b<NV;++b){
            const int col=j*NV+b;
            real_t v = 2.0*GPHHP(b,j,i,a) - GPHPH(a,j,b,i);
            if (i==j) v += Fvv[(size_t)a*NV+b];
            if (a==b) v -= Foo[(size_t)i*NO+j];
            Gmat[(size_t)row*total_dim_+col]=v;
        }
    }

    bmark("G^{1h1p} matrix");
    // ---- upload dense G + refresh diagonal ----
    tracked_cudaMalloc(&d_G_, (size_t)total_dim_*total_dim_*sizeof(real_t));
    std::vector<real_t> h_diag(total_dim_);
    for (int r=0;r<total_dim_;++r) h_diag[r]=Gmat[(size_t)r*total_dim_+r];
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()){
        cudaMemcpy(d_G_, Gmat.data(), Gmat.size()*sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_diagonal_, h_diag.data(), h_diag.size()*sizeof(real_t), cudaMemcpyHostToDevice);
    } else
#endif
    {
        for (size_t i=0;i<Gmat.size();++i) d_G_[i]=Gmat[i];
        for (int r=0;r<total_dim_;++r) d_diagonal_[r]=h_diag[r];
    }
    // Verification aid: HOMO->LUMO diagonal element. After the 2026-06-20 W^eff route
    // corrections (u_amci Wooov, u_akei ujaie, u_bmjc SO-derived) the H2O sto-3g (3,2)
    // reference n->pi* diag ~ 0.433 (lowest STEOM root 0.432663 Ha = 11.773 eV; old buggy
    // routes gave ~0.393).
    {
        const int hl = (NO - 1) * NV + 0;
        std::cout << "  STEOM-CCSD G^{1h1p} built (W^eff dressing Eq.34-63, dense "
                  << total_dim_ << "×" << total_dim_ << "); G[HOMO->LUMO diag] = "
                  << Gmat[(size_t)hl * total_dim_ + hl]
                  << "  (H2O sto-3g ref ~0.433 after route fix)." << std::endl;
    }
}

#undef H_OVOV
#undef H_OOOV
#undef H_OOVV
#undef H_OVVO
#undef H_OVVV
#undef H_OOOO
#undef H_VVVV
#undef H_T1
#undef H_T2


// ==================================================================
//  print_amplitude_norms — Ŝ Frobenius for sub-phase 3.0+3.1 smoke
// ==================================================================
namespace {
real_t frobenius_norm_device(const real_t* d_ptr, size_t n) {
    if (d_ptr == nullptr || n == 0) return 0.0;
    std::vector<real_t> h_buf(n);
    cudaMemcpy(h_buf.data(), d_ptr, n * sizeof(real_t), cudaMemcpyDeviceToHost);
    real_t s = 0.0;
    for (size_t i = 0; i < n; ++i) s += h_buf[i] * h_buf[i];
    return std::sqrt(s);
}
} // namespace

void STEOMCCSDOperator::print_amplitude_norms(std::ostream& os) const {
    const int NO = nocc_active_, NV = nvir_;
    const size_t r2_ip_sz  = (size_t)n_act_occ_ * NO * NO * NV;
    const size_t r2_ea_sz  = (size_t)n_act_vir_ * NO * NV * NV;
    const size_t r1_ip_sz  = (size_t)n_act_occ_ * NO;
    const size_t r1_ea_sz  = (size_t)n_act_vir_ * NV;
    const size_t loo_sz    = (size_t)NO * NO;
    const size_t lvv_sz    = (size_t)NV * NV;
    const size_t fov_sz    = (size_t)NO * NV;
    const size_t oooo_sz   = (size_t)NO * NO * NO * NO;
    const size_t ooov_sz   = (size_t)NO * NO * NO * NV;
    const size_t ovov_sz   = (size_t)NO * NV * NO * NV;
    const size_t ovvo_sz   = (size_t)NO * NV * NV * NO;
    const size_t wovoo_sz  = (size_t)NO * NV * NO * NO;
    const size_t wvovv_sz  = (size_t)NV * NO * NV * NV;
    const size_t wvvvv_sz  = (size_t)NV * NV * NV * NV;
    const size_t wvvvo_sz  = (size_t)NV * NV * NV * NO;
    const size_t x_ip_sz   = (size_t)n_act_occ_ * n_act_occ_;
    const size_t x_ea_sz   = (size_t)n_act_vir_ * n_act_vir_;
    const size_t u_mi_sz   = (size_t)n_act_occ_ * NO;

    os << "  [STEOM-CCSD amplitude + bar-H + F^eff_oo Frobenius norms]\n"
       << "    dims:  nocc_active=" << NO << "  nvir=" << NV
       <<           "  n_act_occ=" << n_act_occ_ << "  n_act_vir=" << n_act_vir_
       <<           "  total_dim=" << total_dim_ << "\n"
       << std::fixed << std::setprecision(8)
       << "    ‖R2^IP‖     = " << frobenius_norm_device(d_R2_IP_, r2_ip_sz) << "\n"
       << "    ‖R2^EA‖     = " << frobenius_norm_device(d_R2_EA_, r2_ea_sz) << "\n";
    if (d_R1_IP_ != nullptr)
        os << "    ‖R1^IP‖     = " << frobenius_norm_device(d_R1_IP_, r1_ip_sz) << "\n";
    if (d_R1_EA_ != nullptr)
        os << "    ‖R1^EA‖     = " << frobenius_norm_device(d_R1_EA_, r1_ea_sz) << "\n";
    if (d_Loo_ != nullptr) {
        os << "    ‖Loo‖       = " << frobenius_norm_device(d_Loo_,   loo_sz)   << "\n"
           << "    ‖Lvv‖       = " << frobenius_norm_device(d_Lvv_,   lvv_sz)   << "\n"
           << "    ‖Fov‖       = " << frobenius_norm_device(d_Fov_,   fov_sz)   << "\n"
           << "    ‖Woooo‖     = " << frobenius_norm_device(d_Woooo_, oooo_sz)  << "\n"
           << "    ‖Wooov‖     = " << frobenius_norm_device(d_Wooov_, ooov_sz)  << "\n"
           << "    ‖Wovov‖     = " << frobenius_norm_device(d_Wovov_, ovov_sz)  << "\n"
           << "    ‖Wovvo‖     = " << frobenius_norm_device(d_Wovvo_, ovvo_sz)  << "\n"
           << "    ‖Wovoo‖     = " << frobenius_norm_device(d_Wovoo_, wovoo_sz) << "\n"
           << "    ‖Wvovv‖     = " << frobenius_norm_device(d_Wvovv_, wvovv_sz) << "\n"
           << "    ‖Wvvvv‖     = "
           << (d_Wvvvv_ != nullptr ? std::to_string(frobenius_norm_device(d_Wvvvv_, wvvvv_sz))
                                   : std::string("(canonical-skip)"))
           << "\n"
           << "    ‖Wvvvo‖     = " << frobenius_norm_device(d_Wvvvo_, wvvvo_sz) << "\n";
    } else {
        os << "    (bar-H intermediates not built — d_eri_mo was nullptr; stub mode)\n";
    }
    if (d_X_IP_ != nullptr) {
        const size_t u_ea_sz = (size_t)n_act_vir_ * NV;
        os << "    ‖X(MI)‖     = " << frobenius_norm_device(d_X_IP_, x_ip_sz) << "\n"
           << "    ‖X(EA)‖     = " << frobenius_norm_device(d_X_EA_, x_ea_sz) << "\n"
           << "    ‖U(M,I)‖    = " << frobenius_norm_device(d_U_MI_, u_mi_sz) << "\n"
           << "    ‖U(E,A)‖    = " << frobenius_norm_device(d_U_EA_, u_ea_sz) << "\n"
           << "    ‖F^eff_oo‖  = " << frobenius_norm_device(d_F_eff_oo_, loo_sz) << "\n"
           << "    ‖F^eff_vv‖  = " << frobenius_norm_device(d_F_eff_vv_, lvv_sz) << "\n";
        // Frobenius of (F^eff_oo - Loo) and (F^eff_vv - Lvv) are the Ŝ-induced
        // dressing magnitudes (more interpretable than the absolute norms).
        auto pull_pair = [&](real_t* d_a, real_t* d_b, size_t n,
                             std::vector<real_t>& h_a, std::vector<real_t>& h_b) {
            h_a.resize(n); h_b.resize(n);
#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                cudaMemcpy(h_a.data(), d_a, n * sizeof(real_t), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_b.data(), d_b, n * sizeof(real_t), cudaMemcpyDeviceToHost);
            } else
#endif
            {
                for (size_t i = 0; i < n; ++i) { h_a[i] = d_a[i]; h_b[i] = d_b[i]; }
            }
        };
        std::vector<real_t> h_F_eff, h_bar;
        pull_pair(d_F_eff_oo_, d_Loo_, loo_sz, h_F_eff, h_bar);
        real_t s_oo = 0.0;
        for (size_t i = 0; i < loo_sz; ++i) {
            real_t d = h_F_eff[i] - h_bar[i];
            s_oo += d * d;
        }
        os << "    ‖F^eff_oo−Loo‖ = " << std::sqrt(s_oo) << "  (Ŝ^IP-induced dressing magnitude)\n";
        pull_pair(d_F_eff_vv_, d_Lvv_, lvv_sz, h_F_eff, h_bar);
        real_t s_vv = 0.0;
        for (size_t i = 0; i < lvv_sz; ++i) {
            real_t d = h_F_eff[i] - h_bar[i];
            s_vv += d * d;
        }
        os << "    ‖F^eff_vv−Lvv‖ = " << std::sqrt(s_vv) << "  (Ŝ^EA-induced dressing magnitude)\n";
    }
    os << "    (Sub-phase 3.4: F^eff_oo / F^eff_vv dressed per CFOUR\n"
       << "     `gmi_steom_rhf` / `gea_steom_rhf`; X(MI)/X(EA) are active R1\n"
       << "     inverses per CFOUR `renormalize`.)\n";
}

} // namespace gansu
