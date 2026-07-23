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
 * @file dlpno_ea_eom_native_operator.cu
 * @brief Native per-pair DLPNO-EA-EOM σ operator (stage B (a), Phase B-EA.1:
 *        σ1 + diagonal σ2). See the .hpp for the contract / build-up.
 */

#include "dlpno_ea_eom_native_operator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

#ifndef GANSU_CPU_ONLY
#include <cublas_v2.h>
#endif

#include <Eigen/Dense>

#include "dlpno_pair_data.hpp"          // PairSetup, PairData
#include "dlpno_ea_eom_transform.hpp"   // ea_packed_r2_to_canonical
#include "device_host_memory.hpp"       // tracked_cudaMalloc / tracked_cudaFree
#include "gpu_manager.hpp"              // gpu::gpu_available
#include "multi_gpu_manager.hpp"        // MultiGpuManager (Stage 5 multi-GPU)

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Master-switch env reader (see the IP native operator for the full rationale):
// the native operator is only constructed under the driver gate
// GANSU_DLPNO_NATIVE_EOM=1, so the validated GPU stack DEFAULTS ON inside the
// ctor — production needs only NATIVE_EOM=1. `dflt` is returned when unset; a set
// var turns the stage on unless it starts with '0' ("=0" opt-out). Diagnostics
// pass dflt=false to stay opt-in.
inline bool env_on(const char* name, bool dflt) {
    const char* e = std::getenv(name);
    if (!e || !e[0]) return dflt;
    return e[0] != '0';
}

bool uloc_is_identity(const std::vector<real_t>& U_loc, int nocc) {
    if (static_cast<int>(U_loc.size()) != nocc * nocc) return true;
    for (int I = 0; I < nocc; ++I)
        for (int k = 0; k < nocc; ++k) {
            const real_t expect = (I == k) ? 1.0 : 0.0;
            if (std::fabs(U_loc[static_cast<size_t>(I) * nocc + k] - expect) > 1e-12)
                return false;
        }
    return true;
}
#ifndef GANSU_CPU_ONLY
__global__ void dlpno_ea_native_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}

// B-a.6a Stage 3b T_tmp stage-1 helper: out[(c·nocc+l)·nvir+d] = 2 r2c[l](c,d) - r2c[l](d,c)
// — the (c,l,d)-reordered symmetrized lifted amplitude (r2c[l] = d_r2c_all_ block l,
// row-major [nvir×nvir]; zero blocks stay zero, matching the host n_pno skip).
__global__ void dlpno_ea_native_r2c_sym_re_kernel(
    const real_t* __restrict__ r2c_all, real_t* __restrict__ out, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nvir * nocc * nvir;
    if (idx >= total) return;
    const int d = idx % nvir;
    const int l = (idx / nvir) % nocc;
    const int c = idx / (nvir * nocc);
    const size_t base = static_cast<size_t>(l) * nvir * nvir;
    out[idx] = 2.0 * r2c_all[base + static_cast<size_t>(c) * nvir + d]
                   - r2c_all[base + static_cast<size_t>(d) * nvir + c];
}

// B-a.6a Stage 4 σ1 Wvovv·r2 helper: out[(l·nvir+c)·nvir+d] = 2 r2c[l](c,d) - r2c[l](d,c)
// — the (l,c,d)-ordered symmetrized lifted amplitude so the single σ1 GEMV
// Σ_{lcd} Wvovv[a,l,c,d]·out[l,c,d] folds the c↔d swap of (2Wvovv - Wvovvˢʷᵃᵖ).
// r2c[l] = d_r2c_all_ block l (row-major [nvir×nvir]); zero blocks stay zero.
__global__ void dlpno_ea_native_r2c_sym_lcd_kernel(
    const real_t* __restrict__ r2c_all, real_t* __restrict__ out, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nocc * nvir * nvir;
    if (idx >= total) return;
    const int d = idx % nvir;
    const int c = (idx / nvir) % nvir;
    const int l = idx / (nvir * nvir);
    const size_t base = static_cast<size_t>(l) * nvir * nvir;
    out[idx] = 2.0 * r2c_all[base + static_cast<size_t>(c) * nvir + d]
                   - r2c_all[base + static_cast<size_t>(d) * nvir + c];
}

// B-a.6f Stage F5b on-device ph borrow kernels (5 kernels, all grid-stride for safety
// at 100-atom scale where total can exceed INT_MAX). All take (nocc, nvir) and the
// a-chunk window (a0, act). Index math derived from the canonical/LMO/re layouts.
//   canonical Wovov[L,a,J,d] row-major: strides (nvir·nocc·nvir, nocc·nvir, nvir, 1)
//   canonical Wovvo[L,a,d,J] row-major: strides (nvir·nvir·nocc, nvir·nocc, nocc, 1)
//   d_Wovov_lmo_[l,a,j,d]   row-major: strides (nvir·nocc·nvir, nocc·nvir, nvir, 1)
//   d_Wovvo_re_ [l,j,a,d]   row-major: strides (nocc·nvir·nvir, nvir·nvir, nvir, 1)

// F5b identity transpose: d_Wovvo_re_[l,j,a,d] = canon_Wovvo[L=l, a, d, J=j]. No rotation.
__global__ void f5b_transpose_wovvo_canon_to_re_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ dst, int nocc, int nvir)
{
    const size_t total = static_cast<size_t>(nocc) * nocc * nvir * nvir;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int d = static_cast<int>(idx % nvir);
        const int a = static_cast<int>((idx / nvir) % nvir);
        const int j = static_cast<int>((idx / (static_cast<size_t>(nvir) * nvir)) % nocc);
        const int l = static_cast<int>(idx / (static_cast<size_t>(nocc) * nvir * nvir));
        const size_t src_off = (((static_cast<size_t>(l) * nvir) + a) * nvir + d) * nocc + j;
        dst[idx] = src[src_off];
    }
}

// F5b PM-path gather: contiguous-pack an (a-slab × L-slab) of canonical Wovov[L,a,J,d]
// into dst[a_rel, L_rel, J, d] (a_rel OUTERMOST so per-a_rel cuBLAS batches are
// contiguous — required for the Stage A `cublasDgemmStridedBatched` with `lda=nM1,
// strideA=Lact·nM1`; an (L_rel, a_rel, J, d) layout would force STRIDED lda).
// (L0, Lact) selects rows L ∈ [L0, L0+Lact); (L0=0, Lact=nocc) reproduces the
// pre-L-chunk full-L gather byte-for-byte.
__global__ void f5b_gather_wovov_canon_aslab_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ dst,
    int nocc, int nvir, int a0, int a_chunk, int L0, int Lact)
{
    const size_t total = static_cast<size_t>(a_chunk) * Lact * nocc * nvir;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int d = static_cast<int>(idx % nvir);
        const int J = static_cast<int>((idx / nvir) % nocc);
        const int L_rel = static_cast<int>((idx / (static_cast<size_t>(nvir) * nocc)) % Lact);
        const int a_rel = static_cast<int>(idx / (static_cast<size_t>(nvir) * nocc * Lact));
        const int a = a0 + a_rel;
        const int L = L0 + L_rel;
        const size_t src_off = (((static_cast<size_t>(L) * nvir) + a) * nocc + J) * nvir + d;
        dst[idx] = src[src_off];
    }
}

// F5b PM-path scatter: per (a_rel) rotated slab T2[a_rel, l, j, d] → d_Wovov_lmo_[l, a, j, d].
__global__ void f5b_scatter_wovov_lmo_aslab_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ dst,
    int nocc, int nvir, int a0, int a_chunk)
{
    const size_t total = static_cast<size_t>(a_chunk) * nocc * nocc * nvir;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int d = static_cast<int>(idx % nvir);
        const int j = static_cast<int>((idx / nvir) % nocc);
        const int l = static_cast<int>((idx / (static_cast<size_t>(nvir) * nocc)) % nocc);
        const int a_rel = static_cast<int>(idx / (static_cast<size_t>(nvir) * nocc * nocc));
        const int a = a0 + a_rel;
        // src layout T2[a_rel, l, j, d] strides (nocc·nocc·nvir, nocc·nvir, nvir, 1)
        const size_t src_off = ((static_cast<size_t>(a_rel) * nocc + l) * nocc + j) * nvir + d;
        // dst layout d_Wovov_lmo_[l, a, j, d] strides (nvir·nocc·nvir, nocc·nvir, nvir, 1)
        const size_t dst_off = (((static_cast<size_t>(l) * nvir) + a) * nocc + j) * nvir + d;
        dst[dst_off] = src[src_off];
    }
}

// F5b PM-path gather: contiguous-pack an (a-slab × L-slab) of canonical Wovvo[L,a,d,J]
// into dst[L_rel, a_rel, d, J] for the J-first rotation (J innermost in Wovvo canonical).
// (L0, Lact) selects rows L ∈ [L0, L0+Lact); (L0=0, Lact=nocc) reproduces the
// pre-L-chunk full-L gather byte-for-byte.
__global__ void f5b_gather_wovvo_canon_aslab_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ dst,
    int nocc, int nvir, int a0, int a_chunk, int L0, int Lact)
{
    const size_t total = static_cast<size_t>(Lact) * a_chunk * nvir * nocc;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int J = static_cast<int>(idx % nocc);
        const int d = static_cast<int>((idx / nocc) % nvir);
        const int a_rel = static_cast<int>((idx / (static_cast<size_t>(nocc) * nvir)) % a_chunk);
        const int L_rel = static_cast<int>(idx / (static_cast<size_t>(nocc) * nvir * a_chunk));
        const int a = a0 + a_rel;
        const int L = L0 + L_rel;
        const size_t src_off = (((static_cast<size_t>(L) * nvir) + a) * nvir + d) * nocc + J;
        dst[idx] = src[src_off];
    }
}

// F5b PM-path scatter+transpose: rotated slab T2[a_rel, l, d, j] → d_Wovvo_re_[l, j, a, d].
__global__ void f5b_scatter_wovvo_re_aslab_kernel(
    const real_t* __restrict__ src, real_t* __restrict__ dst,
    int nocc, int nvir, int a0, int a_chunk)
{
    const size_t total = static_cast<size_t>(a_chunk) * nocc * nvir * nocc;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int j = static_cast<int>(idx % nocc);
        const int d = static_cast<int>((idx / nocc) % nvir);
        const int l = static_cast<int>((idx / (static_cast<size_t>(nocc) * nvir)) % nocc);
        const int a_rel = static_cast<int>(idx / (static_cast<size_t>(nocc) * nvir * nocc));
        const int a = a0 + a_rel;
        // src layout T2[a_rel, l, d, j] strides (nocc·nvir·nocc, nvir·nocc, nocc, 1)
        const size_t src_off = ((static_cast<size_t>(a_rel) * nocc + l) * nvir + d) * nocc + j;
        // dst layout d_Wovvo_re_[l, j, a, d] strides (nocc·nvir·nvir, nvir·nvir, nvir, 1)
        const size_t dst_off = (((static_cast<size_t>(l) * nocc) + j) * nvir + a) * nvir + d;
        dst[dst_off] = src[src_off];
    }
}
#endif

void pull_device(const real_t* d_src, std::vector<real_t>& dst) {
    if (d_src == nullptr) { std::fill(dst.begin(), dst.end(), real_t(0.0)); return; }
    cudaMemcpy(dst.data(), d_src, dst.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
}

#ifndef GANSU_CPU_ONLY
// B-a.6g GPU rotate of a single TRAILING occ index: for a [nlead × nocc]
// row-major tensor, out[m,j] = Σ_J in[m,J] U[J,j]. (EA Wvvvo: m=(a,b,c).)
// Reads the source operator's resident `d_in` directly; result D2H'd into host
// `out` → downstream byte-identical (≈1e-13 GEMM reduction-order drift). `d_U` =
// U_loc [nocc²] row-major (U[J,j] at J·nocc+j). col-major: outᵀ(j,m) = U·inᵀ.
void rotate_last_occ_gpu(cublasHandle_t cublas, const real_t* d_in, const real_t* d_U,
                         std::vector<real_t>& out, int nlead, int nocc) {
    const size_t sz = static_cast<size_t>(nlead) * nocc;
    out.assign(sz, 0.0);
    if (d_in == nullptr) return;
    real_t* d_out = nullptr;
    tracked_cudaMalloc(&d_out, sz * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        nocc, nlead, nocc, &one,
        d_U, nocc, d_in, nocc, &zero, d_out, nocc);
    cudaMemcpy(out.data(), d_out, sz * sizeof(real_t), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_out);
}

// Max-abs diff reporter for the B-a.6g rotation self-check (VALIDATE only).
void rot_check_report(const char* tag, const char* nm,
                      const std::vector<real_t>& ref, const std::vector<real_t>& got) {
    double md = 0.0; size_t at = 0;
    const size_t n = std::min(ref.size(), got.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(ref[i] - got[i]);
        if (d > md) { md = d; at = i; }
    }
    std::cout << "  [" << tag << "] " << nm << " max|gpu-host| = "
              << std::scientific << md << std::defaultfloat
              << " (@" << at << ")" << std::endl;
}
#endif
}  // namespace

DLPNOEAEOMNativeOperator::DLPNOEAEOMNativeOperator(
    const EAEOMCCSDOperator& ea_op,
    const DLPNOLMP2Result& res,
    const DLPNOEAPacking& packing,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& eps_v,
    int num_gpus)
    : res_(res), packing_(packing),
      U_loc_(U_loc), C_vir_(C_vir),
      h_S_(h_S, h_S + static_cast<size_t>(nao) * nao),
      nao_(nao), nvir_(nvir), nocc_(packing.nocc),
      total_dim_(packing.total_dim),
      num_gpus_(num_gpus > 1 ? num_gpus : 1)   // Stage 5 multi-GPU (scaffolding; ≥1)
{
    // (D) construction profiler (env GANSU_DLPNO_NATIVE_PROF=1). Attributes the
    // EA native-operator ctor wall (~18 s at naphthalene, incl Wvvvv ring 5.3 s)
    // across phases. Default off → no output. Sync-bracketed. Mirrors the IP
    // native profiler.
    const bool _nprof = std::getenv("GANSU_DLPNO_NATIVE_PROF") != nullptr;
    auto _npclk = std::chrono::high_resolution_clock::now();
    auto _npmark = [&](const char* nm) {
        if (!_nprof) return;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
        const auto now = std::chrono::high_resolution_clock::now();
        std::cout << "  [EA-native-PROF] " << nm << " = " << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double>(now - _npclk).count() << " s"
                  << std::defaultfloat << std::endl;
        _npclk = now;
    };

    // B-EA.6c/6d/6e env flags (read first so the dense-Wvvvv borrow can be
    // skipped under the true-scaling bare path). NATIVE_RING ⊂ NATIVE_DRESSED;
    // NATIVE_BARE ⊂ NATIVE_RING (W_pair seed + native-only ring, no dense nvir⁴).
    {
        // Master-switch defaults: NATIVE_EOM=1 (driver gate) ⇒ the whole validated
        // stack defaults ON here; each stage is overridable with "=0" for bisection.
        // GPU stages also require gpu_available(). NATIVE_RING ⊂ NATIVE_DRESSED;
        // NATIVE_BARE ⊂ NATIVE_RING (W_pair seed + native-only ring, no dense nvir⁴).
        const bool gpu_ok = gpu::gpu_available();
        use_dressed_pno_ = env_on("GANSU_DLPNO_NATIVE_DRESSED", true);
        use_native_ring_ = use_dressed_pno_ && env_on("GANSU_DLPNO_NATIVE_RING", true);
        use_native_bare_ = use_native_ring_ && env_on("GANSU_DLPNO_NATIVE_BARE", true);
        // B-a.6a GPU port (Stage 1): only meaningful for the dressed PNO path.
        use_gpu_ = use_dressed_pno_ && gpu_ok && env_on("GANSU_DLPNO_NATIVE_GPU", true);
        gpu_selfcheck_ = use_gpu_ && env_on("GANSU_DLPNO_NATIVE_GPU_VALIDATE", false);
        // B-a.6a GPU port (Stage 2): the two-sided PNO projection on device.
        use_gpu_proj_ = use_gpu_ && env_on("GANSU_DLPNO_NATIVE_GPU_PROJ", true);
        // B-a.6a GPU port (Stage 3a): the per-matvec source lift on device.
        use_gpu_lift_ = use_gpu_proj_ && env_on("GANSU_DLPNO_NATIVE_GPU_LIFT", true);
        // B-a.6a GPU port (Stage 3b): cross-pair contraction on device (T_Loo first).
        use_gpu_xpair_ = use_gpu_lift_ && env_on("GANSU_DLPNO_NATIVE_GPU_XPAIR", true);
        // B-a.6a GPU port (Stage 3b T_ph2): the second cross-pair term on device.
        use_gpu_ph2_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_PH2", true);
        // B-a.6a GPU port (Stage 3b T_ph3): the third cross-pair term on device.
        use_gpu_ph3_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_PH3", true);
        // B-a.6a GPU port (Stage 3b T_ph1): the first cross-pair ph term on device.
        use_gpu_ph1_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_PH1", true);
        // B-a.6h(grouped) EA ph-ladder batched (single-device only; opt-in). num_gpus_ is the
        // ctor param (set early) → gate on ==1 (equivalent to !use_gpu_multi_, resolved later).
        // Default ON for the single-GPU operator: the grouped batched ph-ladder is bit-exact
        // (machine-eps σ2) and ~7× faster than the per-(j,l) path; the auto-solve policy builds
        // the operator single-GPU whenever it fits, so this fires automatically. =0 to force the
        // per-(j,l) loops. (Always off at num_gpus>1 — the multi path has no grouped.)
        use_ph_grouped_ = use_gpu_ph1_ && use_gpu_ph2_ && use_gpu_ph3_ && num_gpus_ == 1 &&
                          env_on("GANSU_DLPNO_NATIVE_GPU_PH_GROUPED", true);
        // B-a.6a GPU port (Stage 3b T_tmp): the two-stage T_tmp term on device.
        use_gpu_tmp_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_TMP", true);
        // B-a.6a GPU port (Stage 3c T_Lvv): pair-local T_Lvv on device.
        use_gpu_tlvv_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_TLVV", true);
        // B-a.6a GPU port (Stage 3c T_r1): the last host acc term on device.
        use_gpu_tr1_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_TR1", true);
        // B-a.6a GPU port (Stage 4 σ1): the 1p sector on device, reusing the
        // device-resident lifted r2 (d_r2c_all_, ⊂ use_gpu_xpair_). Nested:
        // S1LVV ⊃ S1FOV ⊃ S1WVOVV (bring up Lvv·r1 → Fov·r2 → Wvovv·r2 in order).
        use_gpu_s1lvv_ = use_gpu_xpair_ && env_on("GANSU_DLPNO_NATIVE_GPU_S1LVV", true);
        use_gpu_s1fov_ = use_gpu_s1lvv_ && env_on("GANSU_DLPNO_NATIVE_GPU_S1FOV", true);
        use_gpu_s1wvovv_ = use_gpu_s1fov_ && env_on("GANSU_DLPNO_NATIVE_GPU_S1WVOVV", true);
        // B-a.6a GPU port (Stage 4 full residency): only when EVERY σ2 acc term + the
        // 3 σ1 terms are on device (host builds nothing) — then the per-matvec host
        // round-trips can be removed. Requires the complete set + the env flag.
        use_gpu_resident_ = use_gpu_xpair_ && use_gpu_tlvv_ && use_gpu_tr1_ &&
                            use_gpu_ph1_ && use_gpu_ph2_ && use_gpu_ph3_ && use_gpu_tmp_ &&
                            use_gpu_s1lvv_ && use_gpu_s1fov_ && use_gpu_s1wvovv_ &&
                            env_on("GANSU_DLPNO_NATIVE_GPU_RESIDENT", true);
        // Stage 5b multi-GPU: broadcast d_input + per-device lift. Only meaningful
        // with the full residency path on AND >1 device requested (rhf.get_num_gpus()).
        use_gpu_multi_ = use_gpu_resident_ && num_gpus_ > 1 &&
                         env_on("GANSU_DLPNO_NATIVE_GPU_MULTI", true);
        multi_selfcheck_ = use_gpu_multi_ && env_on("GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE", false);
        // Stage 5c-step2: actual compute split (default ON for multi); NOSLAB forces step1.
        use_gpu_multi_slab_ = use_gpu_multi_ && !env_on("GANSU_DLPNO_NATIVE_GPU_MULTI_NOSLAB", false);
        // Per-term apply profiling (diagnostic; run with --num_gpus 1 for a clean
        // per-matvec breakdown). Adds a cudaDeviceSynchronize around each timed term.
        prof_ = use_gpu_ && env_on("GANSU_DLPNO_NATIVE_PROF", false);
#ifdef GANSU_CPU_ONLY
        use_gpu_ = false; gpu_selfcheck_ = false; use_gpu_proj_ = false;
        use_gpu_lift_ = false; use_gpu_xpair_ = false;
        use_gpu_ph2_ = false; use_gpu_ph3_ = false; use_gpu_ph1_ = false;
        use_gpu_tmp_ = false; use_gpu_tlvv_ = false; use_gpu_tr1_ = false;
        use_gpu_s1lvv_ = false; use_gpu_s1fov_ = false; use_gpu_s1wvovv_ = false;
        use_gpu_resident_ = false;
        use_gpu_multi_ = false; multi_selfcheck_ = false; use_gpu_multi_slab_ = false;
        use_ph_grouped_ = false;
#endif
    }

    // B-a.6g GPU rotation of the pre-dressed canonical→LMO transforms (env
    // GANSU_DLPNO_NATIVE_GPU_ROT=1). Replaces the host rotation loops (PROF
    // "pre-dressed", ~10.5 s at naphthalene under PM localizer) with cuBLAS GEMMs
    // reading ea_op's resident device buffers. Off by default → host path,
    // bit-exact. _ROT_VALIDATE=1 compares device vs host (max|gpu-host|).
#ifndef GANSU_CPU_ONLY
    bool use_gpu_rot = false, rot_selfcheck = false;
    cublasHandle_t cublas_rot = nullptr;
    real_t* d_U_rot = nullptr;
    {
        use_gpu_rot = gpu::gpu_available() && env_on("GANSU_DLPNO_NATIVE_GPU_ROT", true)
                      && !uloc_is_identity(U_loc_, nocc_);
        rot_selfcheck = use_gpu_rot && env_on("GANSU_DLPNO_NATIVE_GPU_ROT_VALIDATE", false);
        if (use_gpu_rot) {
            cublasCreate(&cublas_rot);
            const size_t no2 = static_cast<size_t>(nocc_) * nocc_;
            tracked_cudaMalloc(&d_U_rot, no2 * sizeof(real_t));
            cudaMemcpy(d_U_rot, U_loc_.data(), no2 * sizeof(real_t),
                       cudaMemcpyHostToDevice);
            std::cout << "[bt-PNO B-a.6g EA] pre-dressed LMO rotation on device "
                         "(Wvvvo trailing-occ cuBLAS GEMM)"
                      << (rot_selfcheck ? " [VALIDATE]" : "") << std::endl;
        }
    }
#endif

    // Borrow the canonical σ1 intermediates (bit-identical to ea_op).
    h_Lvv_.assign(static_cast<size_t>(nvir_) * nvir_, 0.0);
    h_Fov_.assign(static_cast<size_t>(nocc_) * nvir_, 0.0);
    h_Wvovv_.assign(static_cast<size_t>(nvir_) * nocc_ * nvir_ * nvir_, 0.0);
    pull_device(ea_op.get_Lvv_device(),   h_Lvv_);
    pull_device(ea_op.get_Fov_device(),   h_Fov_);
    pull_device(ea_op.get_Wvovv_device(), h_Wvovv_);

    // Wvvvo [nvir·nvir·nvir·nocc], occ index j (4th) rotated canonical→LMO for T_r1:
    //   Wvvvo_lmo[a,b,c,j] = Σ_J U_loc[J,j] Wvvvo[a,b,c,J]   (single-occ; copy for none)
    // Layout ((a*nvir+b)*nvir+c)*nocc + j.
    {
        const size_t sz = static_cast<size_t>(nvir_) * nvir_ * nvir_ * nocc_;
        if (uloc_is_identity(U_loc_, nocc_)) {
            std::vector<real_t> h_Wvvvo(sz, 0.0);
            pull_device(ea_op.get_Wvvvo_device(), h_Wvvvo);
            h_Wvvvo_lmo_ = std::move(h_Wvvvo);
#ifndef GANSU_CPU_ONLY
        } else if (use_gpu_rot) {
            // B-a.6g: rotate the trailing occ index J→j on device (single cuBLAS
            // GEMM over m=(a,b,c)) instead of the host omp loop. Reads ea_op's
            // resident Wvvvo directly; result D2H'd → downstream byte-identical.
            rotate_last_occ_gpu(cublas_rot, ea_op.get_Wvvvo_device(), d_U_rot,
                                h_Wvvvo_lmo_, nvir_ * nvir_ * nvir_, nocc_);
            if (rot_selfcheck) {
                std::vector<real_t> h_Wvvvo(sz, 0.0), ref(sz, 0.0);
                pull_device(ea_op.get_Wvvvo_device(), h_Wvvvo);
                const int nv = nvir_, no = nocc_;
                #pragma omp parallel for
                for (int ab = 0; ab < nv * nv; ++ab)
                    for (int c = 0; c < nv; ++c) {
                        const size_t base = (static_cast<size_t>(ab) * nv + c) * no;
                        for (int j = 0; j < no; ++j) {
                            real_t s = 0.0;
                            for (int J = 0; J < no; ++J)
                                s += U_loc_[static_cast<size_t>(J) * no + j] * h_Wvvvo[base + J];
                            ref[base + j] = s;
                        }
                    }
                rot_check_report("EA-ROT-CHK", "Wvvvo_lmo", ref, h_Wvvvo_lmo_);
            }
#endif
        } else {
            std::vector<real_t> h_Wvvvo(sz, 0.0);
            pull_device(ea_op.get_Wvvvo_device(), h_Wvvvo);
            h_Wvvvo_lmo_.assign(sz, 0.0);
            const int nv = nvir_, no = nocc_;
            #pragma omp parallel for
            for (int ab = 0; ab < nv * nv; ++ab)
                for (int c = 0; c < nv; ++c) {
                    const size_t base = (static_cast<size_t>(ab) * nv + c) * no;
                    for (int j = 0; j < no; ++j) {
                        real_t s = 0.0;
                        for (int J = 0; J < no; ++J)
                            s += U_loc_[static_cast<size_t>(J) * no + j] * h_Wvvvo[base + J];
                        h_Wvvvo_lmo_[base + j] = s;
                    }
                }
        }
    }

    // T_Loo: Loo [nocc²] both occ rotated canonical→LMO (U_locᵀ Loo U_loc; copy none).
    {
        std::vector<real_t> h_Loo(static_cast<size_t>(nocc_) * nocc_, 0.0);
        pull_device(ea_op.get_Loo_device(), h_Loo);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Loo_lmo_ = std::move(h_Loo);
        } else {
            Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);
            Eigen::Map<const RowMatXd> Loo(h_Loo.data(), nocc_, nocc_);
            const RowMatXd C = Uloc.transpose() * Loo * Uloc;
            h_Loo_lmo_.assign(static_cast<size_t>(nocc_) * nocc_, 0.0);
            for (int k = 0; k < nocc_; ++k)
                for (int i = 0; i < nocc_; ++i)
                    h_Loo_lmo_[static_cast<size_t>(k) * nocc_ + i] = C(k, i);
        }
    }

    // B-a.6f Stage F5b activation flag — under production BARE + GPU residency +
    // !VALIDATE, the persistent host members h_Wovov_lmo_/h_Wovvo_lmo_ are NEVER
    // materialised; instead `borrow_ph_to_device_f5b()` (called after the Stage 3b
    // device allocs below) builds d_Wovov_lmo_ + d_Wovvo_re_ on device 0 directly
    // from the canonical EA-op device buffers via D2D copy + on-device rotation.
    // B-a.6h: extended to multi-GPU (num_gpus>1) — f5b populates the device-0
    // buffers; the d>0 slab gather (below) then sources Wovov from device 0
    // (peer-copy of the j-subrange) instead of the elided host array, and Wovvo
    // already peer-copies from device-0 d_Wovvo_re_. This removes the SERIAL host
    // Wovvo/Wovov rotation (PROF "pre-dressed" ~9.5 s at naphthalene, omp-less)
    // from the num_gpus>1 path — the EA-side analogue of the IP (D) win.
    // Other modes (!resident, VALIDATE on, !BARE) keep the F5 host-rotation +
    // late-free path. ALL Stage 3b host uploads + the F5 late-free below are gated
    // by the SAME `f5b_active`.
    // ⚠ device-balancing guard: f5b builds its device-0 buffers and (multi-GPU)
    // peer-gathers slabs from device 0, which assumes the EA operator's buffers live
    // on device 0. Under GANSU_STEOM_OPERATOR_DEVICE_BALANCING the operator may be
    // redirected to another device → cross-device f5b is UNVERIFIED. Disable f5b
    // there (fall back to the validated F5 host-rotation path) until validated.
    const bool device_balancing = env_on("GANSU_STEOM_OPERATOR_DEVICE_BALANCING", false);
    const bool f5b_active = use_native_bare_ && use_gpu_resident_ && !gpu_selfcheck_
                            && !device_balancing;

    // ph-ladder: Wovvo[l,b,d,j] (occ pos 0,3) and Wovov[l,b,j,d] (occ pos 0,2),
    // each with its two occ indices rotated canonical→LMO (copy none). Borrowed
    // dense ALREADY-T2-dressed intermediates (shared with IP).
    if (f5b_active) {
        const size_t wvvo_sz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nocc_;
        const size_t wovv_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
        const double gb = static_cast<double>(wvvo_sz + wovv_sz) * sizeof(real_t) / 1.0e9;
        std::cout << "[bt-PNO B-a.6f EA F5b] dense host borrow (h_Wovvo + h_Wovov, "
                  << std::fixed << std::setprecision(2) << gb << " GB) ELIDED "
                     "(on-device ph rotation; ctor transient peak avoided)"
                  << std::defaultfloat << std::endl;
    } else {
        const size_t wvvo_sz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nocc_;
        const size_t wovv_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
        // B-a.6f Stage F5: dense host borrow (h_Wovvo + h_Wovov, ~360 GB at 100
        // atoms). Built here transiently; under BARE + full GPU residency the
        // persistent member storage is RELEASED at the end of the ctor (post
        // Stage 3b device upload + post multi-GPU slab gather), since the matvec
        // never reads the host arrays in that mode. The transient ctor peak
        // remains; the persistent footprint for the Davidson lifetime is freed.
        if (use_native_bare_) {
            const double gb = static_cast<double>(wvvo_sz + wovv_sz) * sizeof(real_t) / 1.0e9;
            std::cout << "[bt-PNO B-a.6f EA] dense host borrow (h_Wovvo + h_Wovov): "
                      << std::fixed << std::setprecision(2) << gb << " GB"
                      << "  (Stage F5: persistent host members released after device upload"
                         " under BARE + GPU residency; transient ctor peak only)"
                      << std::defaultfloat << std::endl;
        }
        std::vector<real_t> h_Wovvo(wvvo_sz, 0.0), h_Wovov(wovv_sz, 0.0);
        pull_device(ea_op.get_Wovvo_device(), h_Wovvo);
        pull_device(ea_op.get_Wovov_device(), h_Wovov);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Wovvo_lmo_ = std::move(h_Wovvo);
            h_Wovov_lmo_ = std::move(h_Wovov);
        } else {
            Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);
            RowMatXd M(nocc_, nocc_);
            h_Wovvo_lmo_.assign(wvvo_sz, 0.0);
            h_Wovov_lmo_.assign(wovv_sz, 0.0);
            for (int a = 0; a < nvir_; ++a)
                for (int d = 0; d < nvir_; ++d) {
                    for (int L = 0; L < nocc_; ++L)
                        for (int J = 0; J < nocc_; ++J)
                            M(L, J) = h_Wovvo[((static_cast<size_t>(L) * nvir_ + a) * nvir_ + d) * nocc_ + J];
                    const RowMatXd C = Uloc.transpose() * M * Uloc;
                    for (int l = 0; l < nocc_; ++l)
                        for (int j = 0; j < nocc_; ++j)
                            h_Wovvo_lmo_[((static_cast<size_t>(l) * nvir_ + a) * nvir_ + d) * nocc_ + j] = C(l, j);
                }
            for (int a = 0; a < nvir_; ++a)
                for (int d = 0; d < nvir_; ++d) {
                    for (int L = 0; L < nocc_; ++L)
                        for (int J = 0; J < nocc_; ++J)
                            M(L, J) = h_Wovov[((static_cast<size_t>(L) * nvir_ + a) * nocc_ + J) * nvir_ + d];
                    const RowMatXd C = Uloc.transpose() * M * Uloc;
                    for (int l = 0; l < nocc_; ++l)
                        for (int j = 0; j < nocc_; ++j)
                            h_Wovov_lmo_[((static_cast<size_t>(l) * nvir_ + a) * nocc_ + j) * nvir_ + d] = C(l, j);
                }
        }
    }

    // T_tmp: eri_ovov [nocc·nvir·nocc·nvir] with the L (3rd, occ) index rotated
    // canonical→LMO; and CCSD T2 with the 2nd occ index rotated. Copy for none.
    //   ovov layout ((k*nvir+c)*nocc+L)*nvir+d ;  t2 layout ((k*nocc+J)*nvir+a)*nvir+b
    {
        const size_t ovov_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
        const size_t t2_sz   = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
        std::vector<real_t> h_ovov(ovov_sz, 0.0), h_t2(t2_sz, 0.0);
        // (EA PNO-lean) the canonical operator may have elided/freed its raw
        // device ovov (native-bare lean) — it then exposes a persistent host
        // mirror with the same bytes; borrow from that instead.
        if (ea_op.get_eri_ovov_device() != nullptr) {
            pull_device(ea_op.get_eri_ovov_device(), h_ovov);
        } else if (ea_op.get_eri_ovov_host().size() == ovov_sz) {
            h_ovov = ea_op.get_eri_ovov_host();
        } else {
            throw std::runtime_error(
                "DLPNOEAEOMNativeOperator: canonical ovov elided but no host "
                "mirror exposed (lean-mode wiring bug).");
        }
        pull_device(ea_op.get_t2_device(),       h_t2);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_ovov_Llmo_ = std::move(h_ovov);
            h_t2_Jlmo_   = std::move(h_t2);
        } else {
            const int no = nocc_, nv = nvir_;
            // ovov: rotate L (3rd) → l, per (k,c,d).
            h_ovov_Llmo_.assign(ovov_sz, 0.0);
            #pragma omp parallel for
            for (int k = 0; k < no; ++k)
                for (int c = 0; c < nv; ++c)
                    for (int d = 0; d < nv; ++d)
                        for (int l = 0; l < no; ++l) {
                            real_t s = 0.0;
                            for (int L = 0; L < no; ++L)
                                s += U_loc_[static_cast<size_t>(L) * no + l]
                                     * h_ovov[((static_cast<size_t>(k) * nv + c) * no + L) * nv + d];
                            h_ovov_Llmo_[((static_cast<size_t>(k) * nv + c) * no + l) * nv + d] = s;
                        }
            // t2: rotate 2nd occ J → j, per (k,a,b).
            h_t2_Jlmo_.assign(t2_sz, 0.0);
            #pragma omp parallel for
            for (int k = 0; k < no; ++k)
                for (int a = 0; a < nv; ++a)
                    for (int b = 0; b < nv; ++b)
                        for (int j = 0; j < no; ++j) {
                            real_t s = 0.0;
                            for (int J = 0; J < no; ++J)
                                s += U_loc_[static_cast<size_t>(J) * no + j]
                                     * h_t2[((static_cast<size_t>(k) * no + J) * nv + a) * nv + b];
                            h_t2_Jlmo_[((static_cast<size_t>(k) * no + j) * nv + a) * nv + b] = s;
                        }
        }
    }

    // T_vvvv: Wvvvv [nvir⁴] is occ-free → borrow as-is. SKIPPED under the
    // B-EA.6e true-scaling bare path (Wvvvv^(jj) = W_pair + native_ring, no
    // dense nvir⁴ ever materialised — the whole point at 100 atoms).
    if (!use_native_bare_) {
        h_Wvvvv_.assign(static_cast<size_t>(nvir_) * nvir_ * nvir_ * nvir_, 0.0);
        pull_device(ea_op.get_Wvvvv_device(), h_Wvvvv_);
    }

    // Koopmans/PNO diagonal: 1p D[a]=+ε_a; 2p1h D[i,a',b']=-F_ii+Λ_a'+Λ_b'
    // (PNO eigenvalues of pair (i,i)). Preconditioner only (σ2 matvec is built
    // term by term in B-EA.2+, NOT this placeholder).
    h_diagonal_.assign(static_cast<size_t>(total_dim_), 0.0);
    for (int a = 0; a < nvir_; ++a) h_diagonal_[a] = eps_v[a];
    for (int i = 0; i < nocc_; ++i) {
        const int n = packing_.n_pno_ii[i];
        if (n == 0) continue;
        const int idx = res_.pair_lookup[static_cast<size_t>(i) * nocc_ + i];
        const real_t Fii = res_.setups[idx].F_ii;
        const std::vector<real_t>& Lam = res_.pairs[idx].Lambda;
        real_t* blk = h_diagonal_.data() + packing_.off_i[i];
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                blk[static_cast<size_t>(a) * n + b] = -Fii + Lam[a] + Lam[b];
    }
    tracked_cudaMalloc(&d_diagonal_, static_cast<size_t>(total_dim_) * sizeof(real_t));
    cudaMemcpy(d_diagonal_, h_diagonal_.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);

#ifndef GANSU_CPU_ONLY
    // B-a.6g: release the rotation scratch (all pre-dressed rotations done).
    if (d_U_rot)    { tracked_cudaFree(d_U_rot);   d_U_rot = nullptr; }
    if (cublas_rot) { cublasDestroy(cublas_rot);   cublas_rot = nullptr; }
#endif

    _npmark("pre-dressed (env flags + optional dense Wvvvv borrow)");
    // B-EA.6d/6e true-scaling path (env GANSU_DLPNO_NATIVE_DRESSED=1; flags read
    // at the ctor top). Precompute the per-occ diagonal-pair virtual transforms
    // U^(jj) = C_virᵀ S bar_Q_jj once, then the per-occ PNO Wvvvv^(jj). Off by
    // default → dense-borrow T_vvvv path, bit-exact. Mirrors the IP B-a.6c wiring.
    if (use_dressed_pno_) {
        Eigen::Map<const RowMatXd> Cv(C_vir_.data(), nao_, nvir_);
        Eigen::Map<const RowMatXd> S(h_S_.data(), nao_, nao_);
        const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]
        Uall_.assign(nocc_, {});
        for (int jocc = 0; jocc < nocc_; ++jocc) {
            const int n = packing_.n_pno_ii[jocc];
            if (n == 0) continue;
            const int idx = res_.pair_lookup[static_cast<size_t>(jocc) * nocc_ + jocc];
            if (idx < 0) continue;
            Eigen::Map<const RowMatXd> barQ(res_.pairs[idx].bar_Q.data(), nao_, n);
            const RowMatXd U = CvtS * barQ;                // [nvir × n_pno]
            Uall_[jocc].assign(static_cast<size_t>(nvir_) * n, 0.0);
            for (int a = 0; a < nvir_; ++a)
                for (int d = 0; d < n; ++d)
                    Uall_[jocc][static_cast<size_t>(a) * n + d] = U(a, d);
        }
        // B-a.6f Stage F0: time the BARE ctor builders (mirror of IP). The
        // vvvv_ring builder (subtract_dense=false) is the candidate ~hours
        // hotspot at 100 atoms; the vvvv_bare seed is small. Reported by the
        // audit print at the end of this block.
        double f0_ms_vvvv_bare = 0.0, f0_ms_vvvv_ring = 0.0;
        // Seed: B-EA.6e bare = W_pair (Phase24, NO dense nvir⁴); B-EA.6d = cong(dense Wvvvv).
        if (use_native_bare_) {
            const auto t0_vbare = std::chrono::high_resolution_clock::now();
            dressed_ = build_dressed_pno_ea_vvvv_bare(res_, packing_.n_pno_ii);
            f0_ms_vvvv_bare = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - t0_vbare).count();
            std::cout << "[bt-PNO B-EA.6e] DLPNOEAEOMNativeOperator: dense-free Wvvvv^(jj) "
                         "path ON (W_pair bare seed + native-only ring, NO dense nvir⁴; "
                         "GANSU_DLPNO_NATIVE_BARE=1)" << std::endl;
        } else if (gpu::gpu_available()) {
            // GPU 4-GEMM congruence (the host congruence4 quad-loop is the EA build hotspot).
            build_dressed_vvvv_gpu(ea_op.get_Wvvvv_device());
            std::cout << "[bt-PNO B-EA.6d] DLPNOEAEOMNativeOperator: per-occ PNO dressed "
                         "Wvvvv^(jj) path ON (GPU congruence4: 4× cublasDgemm[StridedBatched]; "
                         "GANSU_DLPNO_NATIVE_DRESSED=1)" << std::endl;
            if (gpu_selfcheck_) {
                // Compare the smallest-n nonzero occ against the host congruence4 (cheap).
                int jchk = -1, nchk = 1 << 30;
                for (int j = 0; j < nocc_; ++j) {
                    const int n = packing_.n_pno_ii[j];
                    if (n > 0 && n < nchk) { nchk = n; jchk = j; }
                }
                if (jchk >= 0) {
                    std::vector<real_t> ref;
                    congruence4(h_Wvvvv_, Uall_[jchk], nvir_, nchk, ref);
                    real_t dmax = 0.0;
                    for (size_t k = 0; k < ref.size(); ++k)
                        dmax = std::max(dmax, std::fabs(dressed_.Wvvvv_pno[jchk][k] - ref[k]));
                    std::cout << "[bt-PNO B-EA.6d GPU self-check] occ " << jchk << " (n=" << nchk
                              << ") max|Wvvvv_pno(GPU) - congruence4(host)| = "
                              << std::scientific << dmax
                              << "  (expect ≤1e-11 = GPU 4-GEMM congruence == host)"
                              << std::defaultfloat << std::endl;
                }
            }
        } else {
            dressed_ = build_dressed_pno_ea_vvvv(h_Wvvvv_, Uall_, packing_.n_pno_ii, nvir_);
            std::cout << "[bt-PNO B-EA.6d] DLPNOEAEOMNativeOperator: per-occ PNO dressed "
                         "Wvvvv^(jj) path ON (host congruence4; GANSU_DLPNO_NATIVE_DRESSED=1)" << std::endl;
        }

        // B-EA.6d native ring: swap the dense Wvvvv ring for a Phase24 V_ovov_pair
        // + two-sided-barS build (true scaling). The dense ring DR is only built
        // for the validate-gate (subtract_dense); the bare path is native-only and
        // needs NO raw (ov|ov) / T2 rotation (those feed DR alone).
        if (use_native_ring_) {
            std::vector<real_t> h_ovov_lmo2, h_t2_lmo2;
            if (use_native_bare_) {
                // Native-only ring: DR not built → leave ovov/t2 empty.
            } else if (uloc_is_identity(U_loc_, nocc_)) {
                const size_t ovov_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                const size_t t2_sz   = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
                std::vector<real_t> h_ovov(ovov_sz, 0.0), h_t2(t2_sz, 0.0);
                pull_device(ea_op.get_eri_ovov_device(), h_ovov);
                pull_device(ea_op.get_t2_device(),       h_t2);
                h_ovov_lmo2 = std::move(h_ovov);
                h_t2_lmo2   = std::move(h_t2);
            } else {
                const size_t ovov_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                const size_t t2_sz   = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
                std::vector<real_t> h_ovov(ovov_sz, 0.0), h_t2(t2_sz, 0.0);
                pull_device(ea_op.get_eri_ovov_device(), h_ovov);
                pull_device(ea_op.get_t2_device(),       h_t2);
                // ovov[P,c,Q,d] layout ((P·nvir+c)·nocc+Q)·nvir+d, occ at pos 0,2.
                Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);
                RowMatXd M(nocc_, nocc_);
                h_ovov_lmo2.assign(ovov_sz, 0.0);
                for (int c = 0; c < nvir_; ++c)
                    for (int d = 0; d < nvir_; ++d) {
                        for (int P = 0; P < nocc_; ++P)
                            for (int Q = 0; Q < nocc_; ++Q)
                                M(P, Q) = h_ovov[((static_cast<size_t>(P) * nvir_ + c) * nocc_ + Q) * nvir_ + d];
                        const RowMatXd C = Uloc.transpose() * M * Uloc;
                        for (int k = 0; k < nocc_; ++k)
                            for (int l = 0; l < nocc_; ++l)
                                h_ovov_lmo2[((static_cast<size_t>(k) * nvir_ + c) * nocc_ + l) * nvir_ + d] = C(k, l);
                    }
                // t2[P,Q,a,b] layout ((P·nocc+Q)·nvir+a)·nvir+b, occ at pos 0,1.
                h_t2_lmo2.assign(t2_sz, 0.0);
                for (int a = 0; a < nvir_; ++a)
                    for (int b = 0; b < nvir_; ++b) {
                        for (int P = 0; P < nocc_; ++P)
                            for (int Q = 0; Q < nocc_; ++Q)
                                M(P, Q) = h_t2[((static_cast<size_t>(P) * nocc_ + Q) * nvir_ + a) * nvir_ + b];
                        const RowMatXd C = Uloc.transpose() * M * Uloc;
                        for (int k = 0; k < nocc_; ++k)
                            for (int l = 0; l < nocc_; ++l)
                                h_t2_lmo2[((static_cast<size_t>(k) * nocc_ + l) * nvir_ + a) * nvir_ + b] = C(k, l);
                    }
            }
            real_t ring_max_delta = 0.0, ring_max_ref = 0.0;
            const auto t0_vring = std::chrono::high_resolution_clock::now();
            build_dressed_pno_ea_vvvv_ring(dressed_, res_, h_ovov_lmo2, h_t2_lmo2, Uall_,
                                           h_S_, packing_.n_pno_ii, nao_, nocc_, nvir_,
                                           ring_max_delta, ring_max_ref,
                                           /*subtract_dense=*/!use_native_bare_);
            f0_ms_vvvv_ring = std::chrono::duration<double, std::milli>(
                                  std::chrono::high_resolution_clock::now() - t0_vring).count();
            if (use_native_bare_)
                std::cout << "[bt-PNO B-EA.6e] native Wvvvv ring added (Phase24 V_ovov_pair + "
                             "two-sided barS, NO dense DR): max|native_ring| = "
                          << std::scientific << ring_max_ref
                          << "  (Wvvvv^(jj) = W_pair + native_ring; T1 terms deferred)" << std::endl;
            else
                std::cout << "[bt-PNO B-EA.6d] native Wvvvv ring ON (Phase24 V_ovov_pair + "
                             "two-sided barS; GANSU_DLPNO_NATIVE_RING=1): "
                             "max|native_ring - cong(dense_ring)| = "
                          << std::scientific << ring_max_delta
                          << "  (|cong(dense_ring)|_max = " << ring_max_ref
                          << "; →0 at full PNO, else = truncation correction)" << std::endl;
        }

        // B-a.6f Stage F0 audit (EA mirror of IP). Silent-host detector +
        // ctor builder wall. Diagnostic-only; no compute path changes.
        if (use_native_bare_) {
            std::cout << "[bt-PNO B-a.6f EA BARE-AUDIT]"
                      << "  use_native_bare_=" << (use_native_bare_ ? 1 : 0)
                      << "  use_gpu_resident_=" << (use_gpu_resident_ ? 1 : 0)
                      << std::endl;
            if (!use_gpu_resident_) {
                std::cout << "  [WARN] residency OFF → matvec drops to host-Eigen "
                             "compute_sigma2 (the 60 s/matvec failure mode)." << std::endl;
                std::cout << "  Disabled sub-flags (default ON under NATIVE_EOM; "
                             "listed = explicitly set =0 or GPU unavailable):";
                if (!use_gpu_)         std::cout << " GANSU_DLPNO_NATIVE_GPU";
                if (!use_gpu_proj_)    std::cout << " GANSU_DLPNO_NATIVE_GPU_PROJ";
                if (!use_gpu_lift_)    std::cout << " GANSU_DLPNO_NATIVE_GPU_LIFT";
                if (!use_gpu_xpair_)   std::cout << " GANSU_DLPNO_NATIVE_GPU_XPAIR";
                if (!use_gpu_tlvv_)    std::cout << " GANSU_DLPNO_NATIVE_GPU_TLVV";
                if (!use_gpu_tr1_)     std::cout << " GANSU_DLPNO_NATIVE_GPU_TR1";
                if (!use_gpu_ph1_)     std::cout << " GANSU_DLPNO_NATIVE_GPU_PH1";
                if (!use_gpu_ph2_)     std::cout << " GANSU_DLPNO_NATIVE_GPU_PH2";
                if (!use_gpu_ph3_)     std::cout << " GANSU_DLPNO_NATIVE_GPU_PH3";
                if (!use_gpu_tmp_)     std::cout << " GANSU_DLPNO_NATIVE_GPU_TMP";
                if (!use_gpu_s1lvv_)   std::cout << " GANSU_DLPNO_NATIVE_GPU_S1LVV";
                if (!use_gpu_s1fov_)   std::cout << " GANSU_DLPNO_NATIVE_GPU_S1FOV";
                if (!use_gpu_s1wvovv_) std::cout << " GANSU_DLPNO_NATIVE_GPU_S1WVOVV";
                if (!env_on("GANSU_DLPNO_NATIVE_GPU_RESIDENT", true))
                    std::cout << " GANSU_DLPNO_NATIVE_GPU_RESIDENT";
                std::cout << std::endl;
            }
            std::cout << "  ctor builder wall:  vvvv bare seed = " << std::fixed
                      << std::setprecision(1) << f0_ms_vvvv_bare << " ms"
                      << "  vvvv ring builder = " << f0_ms_vvvv_ring << " ms"
                      << " (subtract_dense=" << (!use_native_bare_ ? "true" : "false") << ")"
                      << std::defaultfloat << std::endl;
        }
    }

    _npmark("dressed_pno_ea + bare seed + Wvvvv ring (host, see BARE-AUDIT)");
    // B-a.6a GPU port — Stage 1 setup. Pack the per-occ diagonal Wvvvv^(jj) into
    // one device buffer + offsets, allocate the packed r2/σ2 scratch, and create
    // a cuBLAS handle. dressed_ is built above (use_gpu_ ⊂ use_dressed_pno_).
#ifndef GANSU_CPU_ONLY
    if (use_gpu_ && gpu::gpu_available()) {
        wvvvv_pno_off_.assign(nocc_ + 1, 0);
        for (int j = 0; j < nocc_; ++j) {
            const size_t n = static_cast<size_t>(packing_.n_pno_ii[j]);
            wvvvv_pno_off_[j + 1] = wvvvv_pno_off_[j] + n * n * n * n;   // n_pno(jj)⁴
        }
        const size_t wtot = wvvvv_pno_off_[nocc_];
        const size_t plen = static_cast<size_t>(total_dim_ - nvir_);
        cublasHandle_t cublas = nullptr;
        cublasCreate(&cublas);
        cublas_ = cublas;
        if (wtot > 0) {
            tracked_cudaMalloc(&d_Wvvvv_pno_pack_, wtot * sizeof(real_t));
            for (int j = 0; j < nocc_; ++j) {
                const size_t sz = wvvvv_pno_off_[j + 1] - wvvvv_pno_off_[j];
                if (sz == 0) continue;
                cudaMemcpy(d_Wvvvv_pno_pack_ + wvvvv_pno_off_[j],
                           dressed_.Wvvvv_pno[j].data(), sz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
            }
        }
        if (plen > 0) {
            tracked_cudaMalloc(&d_r2_pack_,  plen * sizeof(real_t));
            tracked_cudaMalloc(&d_sig_pack_, plen * sizeof(real_t));
        }
        // Stage 2: pack the per-occ U^(jj) [nvir × n_pno(jj)] (row-major, as built
        // in Uall_) and allocate the chained-GEMM scratch (acc upload + acc·U).
        if (use_gpu_proj_) {
            u_pno_off_.assign(nocc_ + 1, 0);
            max_n_pno_ = 0;
            for (int j = 0; j < nocc_; ++j) {
                const int n = packing_.n_pno_ii[j];
                u_pno_off_[j + 1] = u_pno_off_[j] + static_cast<size_t>(nvir_) * n;
                max_n_pno_ = std::max(max_n_pno_, n);
            }
            const size_t utot = u_pno_off_[nocc_];
            if (utot > 0) {
                tracked_cudaMalloc(&d_U_pack_, utot * sizeof(real_t));
                for (int j = 0; j < nocc_; ++j) {
                    const size_t sz = u_pno_off_[j + 1] - u_pno_off_[j];
                    if (sz == 0) continue;
                    cudaMemcpy(d_U_pack_ + u_pno_off_[j], Uall_[j].data(),
                               sz * sizeof(real_t), cudaMemcpyHostToDevice);
                }
            }
            tracked_cudaMalloc(&d_acc_, static_cast<size_t>(nvir_) * nvir_ * sizeof(real_t));
            if (max_n_pno_ > 0)
                tracked_cudaMalloc(&d_T1_, static_cast<size_t>(nvir_) * max_n_pno_ * sizeof(real_t));
            // Stage 3a: device-resident lifted r2c + the lift chained-GEMM scratch.
            if (use_gpu_lift_) {
                tracked_cudaMalloc(&d_r2c_all_,
                                   static_cast<size_t>(nocc_) * nvir_ * nvir_ * sizeof(real_t));
                if (max_n_pno_ > 0)
                    tracked_cudaMalloc(&d_lift_T_, static_cast<size_t>(nvir_) * max_n_pno_ * sizeof(real_t));
            }
            // Stage 3b: full acc stack (host part + GPU cross-pair) + the T_Loo
            // mixing matrix M_Loo[j,l] = -Loo_lmo[l,j] (row-major, = -Loo_lmoᵀ).
            if (use_gpu_xpair_) {
                tracked_cudaMalloc(&d_acc_all_,
                                   static_cast<size_t>(nocc_) * nvir_ * nvir_ * sizeof(real_t));
                std::vector<real_t> h_Loo_xpair(static_cast<size_t>(nocc_) * nocc_, 0.0);
                for (int j = 0; j < nocc_; ++j)
                    for (int l = 0; l < nocc_; ++l)
                        h_Loo_xpair[static_cast<size_t>(j) * nocc_ + l] =
                            -h_Loo_lmo_[static_cast<size_t>(l) * nocc_ + j];
                tracked_cudaMalloc(&d_Loo_xpair_, static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t));
                cudaMemcpy(d_Loo_xpair_, h_Loo_xpair.data(),
                           static_cast<size_t>(nocc_) * nocc_ * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            // Stage 3b T_ph2/T_ph1: borrow Wovov_lmo (occ pos 0,2) to device for the
            // per-(j,l) strided GEMMs (A_j[l]·r2c[l] / r2c[l]·Wovov_j[l]ᵀ).
            if (use_gpu_ph2_ || use_gpu_ph1_) {
                const size_t wsz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                tracked_cudaMalloc(&d_Wovov_lmo_, wsz * sizeof(real_t));
                // F5b: when on-device borrow is active, the H2D upload is skipped
                // (h_Wovov_lmo_ is empty under f5b_active; the buffer will be filled by
                // borrow_ph_to_device_f5b() below from the canonical EA device buffer).
                if (!f5b_active)
                    cudaMemcpy(d_Wovov_lmo_, h_Wovov_lmo_.data(), wsz * sizeof(real_t),
                               cudaMemcpyHostToDevice);
                // 5j: single-GPU / full default — readers use ldB = nocc·nvir. bind_device
                // overrides this per device on the multi path (slab device: ldB = jext·nvir).
                wovov_j_extent_ = nocc_;  wovov_j_base_ = 0;
            }
            // Stage 3b T_ph3/T_ph1: pre-transpose Wovvo_lmo[l,b,c,j] → Wovvo_re[l,j,b,c]
            // (c innermost / stride-1) so B_j[l] is a contiguous [nvir×nvir] block.
            if (use_gpu_ph3_ || use_gpu_ph1_) {
                const size_t wsz = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
                tracked_cudaMalloc(&d_Wovvo_re_, wsz * sizeof(real_t));
                // F5b: under on-device borrow, the host pre-transpose + H2D is skipped;
                // borrow_ph_to_device_f5b() builds d_Wovvo_re_ directly.
                if (!f5b_active) {
                    std::vector<real_t> h_Wovvo_re(wsz, 0.0);
                    const int nv = nvir_, no = nocc_;
                    // j is the innermost source index → gather into a (l,j,b,c) layout.
                    #pragma omp parallel for
                    for (int l = 0; l < no; ++l)
                        for (int j = 0; j < no; ++j)
                            for (int b = 0; b < nv; ++b)
                                for (int c = 0; c < nv; ++c)
                                    h_Wovvo_re[(((static_cast<size_t>(l) * no + j) * nv + b) * nv) + c] =
                                        h_Wovvo_lmo_[((static_cast<size_t>(l) * nv + b) * nv + c) * no + j];
                    cudaMemcpy(d_Wovvo_re_, h_Wovvo_re.data(), wsz * sizeof(real_t),
                               cudaMemcpyHostToDevice);
                }
                // 5h: single-GPU / full default — the readers index the full [l,j(nocc),b,c]
                // layout. bind_device overrides this per device on the multi path.
                wovvo_re_j_extent_ = nocc_;  wovvo_re_j_base_ = 0;
            }
            // B-a.6f Stage F5b: on-device ph borrow (both d_Wovov_lmo_ and d_Wovvo_re_).
            // Fires only when both buffers were allocated (controlled by use_gpu_ph*_).
            if (f5b_active && d_Wovov_lmo_ != nullptr && d_Wovvo_re_ != nullptr) {
                borrow_ph_to_device_f5b(ea_op.get_Wovvo_device(), ea_op.get_Wovov_device());
            }
            // B-a.6h(grouped): build the batched ph-ladder pointer arrays (single-device). The
            // per-(j,l) GEMMs are batched by looping the reduction index l and batching over the
            // active occ j (distinct acc[j]). All base device pointers (d_r2c_all_/d_acc_all_/
            // d_Wovov_lmo_/d_Wovvo_re_) are allocated above and operator-lifetime-fixed; the
            // arrays are laid out [l_active][j_active] (block l = [l·nact, (l+1)·nact)).
            // j_extent=nocc / j_base=0 at single-GPU (set above) so the W bases simplify.
            if (use_ph_grouped_ && d_Wovov_lmo_ != nullptr && d_Wovvo_re_ != nullptr) {
                const int nv = nvir_, no = nocc_;
                ph_act_.clear();
                for (int o = 0; o < no; ++o)
                    if (packing_.n_pno_ii[o] > 0) ph_act_.push_back(o);
                ph_nact_ = static_cast<int>(ph_act_.size());
                const int na = ph_nact_;
                const size_t tot = static_cast<size_t>(na) * na;
                std::vector<const real_t*> pR(tot), pWovov(tot), pWovvo(tot);
                std::vector<real_t*> pAcc(tot);
                for (int li = 0; li < na; ++li) {
                    const int l = ph_act_[li];
                    for (int ji = 0; ji < na; ++ji) {
                        const int j = ph_act_[ji];
                        const size_t idx = static_cast<size_t>(li) * na + ji;
                        pR[idx]     = d_r2c_all_  + static_cast<size_t>(l) * nv * nv;
                        pWovov[idx] = d_Wovov_lmo_ + (static_cast<size_t>(l) * nv * no + j) * nv;
                        pWovvo[idx] = d_Wovvo_re_  + (static_cast<size_t>(l) * no + j) * nv * nv;
                        pAcc[idx]   = d_acc_all_  + static_cast<size_t>(j) * nv * nv;
                    }
                }
                auto up = [&](auto& dptr, const void* hsrc) {
                    if (tot == 0) return;
                    tracked_cudaMalloc(&dptr, sizeof(dptr) * tot);
                    cudaMemcpy(dptr, hsrc, sizeof(dptr) * tot, cudaMemcpyHostToDevice);
                };
                up(d_pR_, pR.data());       up(d_pWovov_, pWovov.data());
                up(d_pWovvo_, pWovvo.data()); up(d_pAcc_, pAcc.data());
                std::cout << "[bt-PNO B-a.6h(grouped) EA] ph-ladder batched ON: " << na
                          << " active occ → ~" << 4 * na << " batched calls/matvec (was ~"
                          << 4 * na * na << " per-(j,l) GEMMs)" << std::endl;
            }
            // Stage 3b T_tmp: borrow ovov_Llmo + t2_Jlmo, alloc the r2c_sym_re / tmp scratch.
            if (use_gpu_tmp_) {
                const size_t ovsz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                const size_t t2sz = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
                tracked_cudaMalloc(&d_ovov_Llmo_, ovsz * sizeof(real_t));
                cudaMemcpy(d_ovov_Llmo_, h_ovov_Llmo_.data(), ovsz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_t2_Jlmo_, t2sz * sizeof(real_t));
                cudaMemcpy(d_t2_Jlmo_, h_t2_Jlmo_.data(), t2sz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
                // 5i: single-GPU / full default — add_ttmp GEMV lda = nocc·nvir². bind_device
                // overrides this per device on the multi path (slab device: lda = jext·nvir²).
                t2_jlmo_j_extent_ = nocc_;  t2_jlmo_j_base_ = 0;
                tracked_cudaMalloc(&d_r2c_sym_re_,
                                   static_cast<size_t>(nvir_) * nocc_ * nvir_ * sizeof(real_t));
                tracked_cudaMalloc(&d_tmp_, static_cast<size_t>(nocc_) * sizeof(real_t));
            }
            // Stage 3c T_Lvv / Stage 4 σ1 Lvv·r1: borrow Lvv (occ-free) to device.
            if (use_gpu_tlvv_ || use_gpu_s1lvv_) {
                const size_t lsz = static_cast<size_t>(nvir_) * nvir_;
                tracked_cudaMalloc(&d_Lvv_, lsz * sizeof(real_t));
                cudaMemcpy(d_Lvv_, h_Lvv_.data(), lsz * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            // Stage 3c T_r1: pre-transpose Wvvvo_lmo[a,b,c,j] → Wvvvo_r1[j,a,b,c]
            // (c stride 1) so M_j[(a,b),c] is contiguous.
            if (use_gpu_tr1_) {
                const size_t wsz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nvir_;
                std::vector<real_t> h_Wvvvo_r1(wsz, 0.0);
                const int nv = nvir_, no = nocc_;
                #pragma omp parallel for
                for (int j = 0; j < no; ++j)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b)
                            for (int c = 0; c < nv; ++c)
                                h_Wvvvo_r1[(((static_cast<size_t>(j) * nv + a) * nv + b) * nv) + c] =
                                    h_Wvvvo_lmo_[((static_cast<size_t>(a) * nv + b) * nv + c) * no + j];
                tracked_cudaMalloc(&d_Wvvvo_r1_, wsz * sizeof(real_t));
                cudaMemcpy(d_Wvvvo_r1_, h_Wvvvo_r1.data(), wsz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
            } else {
                // (M5c-a, 2026-07-22) HOST T_r1 fallback (GANSU_DLPNO_NATIVE_GPU_TR1=0
                // — the nocc·nvir³ device upload cannot fit at decacene scale).
                // The [(a,b,c),j] layout gathers with stride nocc per c: one
                // serial matvec walked ~1.3 TB of cache lines and pinned the EA
                // Davidson at 100% single-core (decacene run10). Pre-transpose
                // ONCE to [j][a][b][c] (contiguous reads, same transform as the
                // device path above) so the per-matvec term becomes a parallel
                // sequential sweep in compute_sigma2. Host-RAM: +nocc·nvir³
                // (alongside h_Wvvvo_lmo_; identical values, identical per-
                // element summation order → bit-identical σ).
                const size_t wsz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nvir_;
                h_Wvvvo_r1_host_.assign(wsz, 0.0);
                const int nv = nvir_, no = nocc_;
                // Tiled transpose: per (a,b) the job is an [c×j] → [j×c] panel
                // transpose (in: stride no in c / contiguous j; out: stride nv³
                // in j / contiguous c). Tiling c keeps both sides cache-resident
                // — the naive j-outer loop walked ~1.3 TB of cache lines (~1 h
                // at decacene); this is a plain permutation copy (numerics-free).
                {
                    constexpr int CT = 16;   // c-tile
                    #pragma omp parallel for collapse(2) schedule(dynamic)
                    for (int a = 0; a < nv; ++a)
                        for (int b = 0; b < nv; ++b) {
                            const size_t in_ab  = (static_cast<size_t>(a) * nv + b) * nv;      // ·no + j 残り
                            const size_t out_ab = (static_cast<size_t>(a) * nv + b) * nv;      // c 残り
                            for (int c0 = 0; c0 < nv; c0 += CT) {
                                const int c1 = (c0 + CT < nv) ? c0 + CT : nv;
                                for (int j = 0; j < no; ++j) {
                                    real_t* dst = h_Wvvvo_r1_host_.data()
                                                + static_cast<size_t>(j) * nv * nv * nv + out_ab;
                                    for (int c = c0; c < c1; ++c)
                                        dst[c] = h_Wvvvo_lmo_[(in_ab + c) * no + j];
                                }
                            }
                        }
                }
                std::cout << "[bt-PNO EA M5c-a] host T_r1 pre-transposed to [j,a,b,c] "
                             "(tiled; contiguous + OpenMP per matvec; device upload elided)"
                          << std::endl;
            }
            // Stage 3c T_r1 / Stage 4 σ1 Lvv·r1: the r1 upload scratch (shared).
            if ((use_gpu_tr1_ || use_gpu_s1lvv_) && !d_r1_)
                tracked_cudaMalloc(&d_r1_, static_cast<size_t>(nvir_) * sizeof(real_t));
            // Stage 4 σ1: the device σ1 accumulator + the per-term intermediates.
            if (use_gpu_s1lvv_)
                tracked_cudaMalloc(&d_sigma1_, static_cast<size_t>(nvir_) * sizeof(real_t));
            if (use_gpu_s1fov_) {
                const size_t fsz = static_cast<size_t>(nocc_) * nvir_;
                tracked_cudaMalloc(&d_Fov_, fsz * sizeof(real_t));
                cudaMemcpy(d_Fov_, h_Fov_.data(), fsz * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            if (use_gpu_s1wvovv_) {
                const size_t wsz = static_cast<size_t>(nvir_) * nocc_ * nvir_ * nvir_;
                tracked_cudaMalloc(&d_Wvovv_, wsz * sizeof(real_t));
                cudaMemcpy(d_Wvovv_, h_Wvovv_.data(), wsz * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_r2c_sym_lcd_,
                                   static_cast<size_t>(nocc_) * nvir_ * nvir_ * sizeof(real_t));
            }
        }
        std::cout << "[bt-PNO B-a.6a] DLPNOEAEOMNativeOperator: GPU σ2 "
                  << (use_gpu_xpair_ ? ((use_gpu_ph2_ || use_gpu_ph3_ || use_gpu_ph1_ || use_gpu_tmp_)
                                       ? "lift + T_Loo + T_ph/T_tmp (cross-pair) + projection + T_vvvv "
                                         "path ON (Stage 3b: device T_Loo stacked GEMM + per-(j,l) "
                                         "T_ph2/T_ph3 GEMMs + lift + projection; "
                                         "GANSU_DLPNO_NATIVE_GPU_PH2/PH3=1"
                                       : "lift + T_Loo + projection + T_vvvv path ON (Stage 3b: "
                                         "device cross-pair T_Loo = M_Loo·R_stack stacked GEMM "
                                         "+ lift + projection; GANSU_DLPNO_NATIVE_GPU_XPAIR=1")
                      : use_gpu_lift_ ? "lift + projection + T_vvvv path ON (per-occ chained "
                                      "cublasDgemm: source lift U·r2p·Uᵀ + projection Uᵀ·acc·U "
                                      "+ Wvvvv^(jj) cublasDgemv; GANSU_DLPNO_NATIVE_GPU_LIFT=1"
                      : use_gpu_proj_ ? "projection + T_vvvv path ON (per-occ chained "
                                        "cublasDgemm Uᵀ·acc·U + Wvvvv^(jj) cublasDgemv; "
                                        "GANSU_DLPNO_NATIVE_GPU_PROJ=1"
                                      : "T_vvvv path ON (per-occ Wvvvv^(jj) cublasDgemv; "
                                        "GANSU_DLPNO_NATIVE_GPU=1")
                  << (gpu_selfcheck_ ? ", in-process GPU-vs-host self-check" : "")
                  << ")" << std::endl;
        if (use_gpu_s1lvv_)
            std::cout << "[bt-PNO B-a.6a] DLPNOEAEOMNativeOperator: GPU σ1 path ON ("
                      << "Lvv·r1"
                      << (use_gpu_s1fov_ ? " + Fov·r2" : "")
                      << (use_gpu_s1wvovv_ ? " + Wvovv·r2" : "")
                      << "; reuses device-resident lifted r2; "
                         "GANSU_DLPNO_NATIVE_GPU_S1LVV/S1FOV/S1WVOVV=1)" << std::endl;
        if (use_gpu_resident_)
            std::cout << "[bt-PNO B-a.6a] DLPNOEAEOMNativeOperator: GPU full-residency "
                         "path ON (r1/r2 read straight from d_input, σ assembled on "
                         "device, no host round-trip; GANSU_DLPNO_NATIVE_GPU_RESIDENT=1)"
                      << std::endl;
        // Stage 5b: allocate the per-device lift workspace (broadcast input copy,
        // U^(jj) replica, chained-GEMM scratch, lifted-r2c scratch) on every d>0.
        // Production output stays on device 0; these replicas drive the broadcast +
        // per-device-lift validation in apply_resident. tracked_cudaMalloc allocates
        // on the current (DeviceGuard) device; cudaMemcpyPeer mirrors d_U_pack_ from
        // device 0. The allocation itself proves the per-device residency footprint fits.
        // Stage 5c: full per-device σ2 workspace (mirror of IP). Each d>0 gets a
        // complete replica of every σ2 device buffer (constants peer-copied, scratch
        // allocated) + a cublas handle, so the validated σ2 helper chain runs unchanged
        // on device d after bind_device(d). ws_[0] aliases the device-0 members. The
        // output-occ slab partition (weight n_pno_ii²) assigns each device its gather
        // range. Production (num_gpus=1) byte-unchanged.
        if (use_gpu_multi_) {
            auto& mgr = MultiGpuManager::instance();
            mgr.initialize(num_gpus_);
            const int nuse = std::min(num_gpus_, mgr.num_devices());
            if (nuse < 2) {
                use_gpu_multi_ = false; multi_selfcheck_ = false; use_gpu_multi_slab_ = false;
            } else {
                ws_.resize(nuse);
                const size_t r2c_len  = static_cast<size_t>(nocc_) * nvir_ * nvir_;
                const size_t lift_len = static_cast<size_t>(nvir_) * max_n_pno_;
                const size_t plen     = static_cast<size_t>(total_dim_ - nvir_);
                const size_t sym_len  = static_cast<size_t>(nvir_) * nocc_ * nvir_;
                const size_t no2      = static_cast<size_t>(nocc_) * nocc_;
                const size_t nv2      = static_cast<size_t>(nvir_) * nvir_;
                const size_t ovov_len = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                const size_t oovv_len = no2 * nv2;
                const size_t wvvvo_len= static_cast<size_t>(nocc_) * nvir_ * nvir_ * nvir_;
                const size_t utot     = u_pno_off_.empty()     ? 0 : u_pno_off_.back();
                const size_t wvvvv_tot= wvvvv_pno_off_.empty() ? 0 : wvvvv_pno_off_.back();
                // Output-occ slab partition (contiguous, weight n_pno_ii²) — computed
                // before the alloc loop so the 5e slab-only Wvvvv pack can size its range.
                std::vector<double> wcum(nocc_ + 1, 0.0);
                for (int j = 0; j < nocc_; ++j) {
                    const double n = packing_.n_pno_ii[j];
                    wcum[j + 1] = wcum[j] + n * n;
                }
                const double total = wcum[nocc_];
                occ_begin_.assign(nuse, 0); occ_end_.assign(nuse, 0);
                { int jb = 0;
                  for (int d = 0; d < nuse; ++d) {
                      occ_begin_[d] = jb;
                      if (d == nuse - 1) jb = nocc_;
                      else { const double tgt = total * (d + 1) / nuse;
                             while (jb < nocc_ && wcum[jb + 1] < tgt) ++jb; }
                      occ_end_[d] = jb;
                  } }
                // ws_[0] aliases the device-0 members.
                ws_[0].device = 0;            ws_[0].cublas = cublas_;
                ws_[0].d_input = nullptr;     ws_[0].d_r2c_all = d_r2c_all_;
                ws_[0].d_lift_T = d_lift_T_;  ws_[0].d_acc_all = d_acc_all_;
                ws_[0].d_sig_pack = d_sig_pack_; ws_[0].d_T1 = d_T1_;
                ws_[0].d_r2c_sym_re = d_r2c_sym_re_; ws_[0].d_tmp = d_tmp_;
                ws_[0].d_U_pack = d_U_pack_;  ws_[0].d_Wvvvv_pno_pack = d_Wvvvv_pno_pack_;
                ws_[0].d_Loo_xpair = d_Loo_xpair_; ws_[0].d_Wovov_lmo = d_Wovov_lmo_;
                ws_[0].d_Wovvo_re = d_Wovvo_re_;   ws_[0].d_ovov_Llmo = d_ovov_Llmo_;
                ws_[0].d_t2_Jlmo = d_t2_Jlmo_;     ws_[0].d_Lvv = d_Lvv_;
                ws_[0].d_Wvvvo_r1 = d_Wvvvo_r1_;
                ws_[0].wvvvv_shift = 0;   // device 0 holds the full Wvvvv pack
                ws_[0].wovvo_re_j_extent = nocc_; ws_[0].wovvo_re_j_base = 0;  // 5h: device 0 full
                ws_[0].t2_jlmo_j_extent = nocc_;  ws_[0].t2_jlmo_j_base = 0;   // 5i: device 0 full
                ws_[0].wovov_j_extent = nocc_;    ws_[0].wovov_j_base = 0;     // 5j: device 0 full
                // d>0 replicas.
                for (int d = 1; d < nuse; ++d) {
                    MultiGpuManager::DeviceGuard guard(d);
                    DeviceWorkspace& w = ws_[d];
                    w.device = d;  w.cublas = mgr.cublas(d);
                    cublasSetStream(mgr.cublas(d), 0);  // NULL stream → match device-0 ordering
                    auto scr = [&](real_t** p, size_t n) {
                        if (n) tracked_cudaMalloc(p, n * sizeof(real_t)); };
                    auto cpy = [&](real_t** p, const real_t* src, size_t n) {
                        if (n && src) { tracked_cudaMalloc(p, n * sizeof(real_t));
                                        cudaMemcpyPeer(*p, d, src, 0, n * sizeof(real_t)); } };
                    scr(&w.d_input, static_cast<size_t>(total_dim_));
                    scr(&w.d_r2c_all, r2c_len);  scr(&w.d_lift_T, lift_len);
                    scr(&w.d_acc_all, r2c_len);  scr(&w.d_sig_pack, plen);
                    scr(&w.d_T1, lift_len);      scr(&w.d_r2c_sym_re, sym_len);
                    scr(&w.d_tmp, static_cast<size_t>(nocc_));
                    cpy(&w.d_U_pack, d_U_pack_, utot);
                    // 5e: Wvvvv pack slab-only in slab mode (occ-contiguous subrange);
                    // else full replica. project_acc_stack uses w.wvvvv_shift to index it.
                    if (use_gpu_multi_slab_ && wvvvv_tot > 0) {
                        const size_t lo = wvvvv_pno_off_[occ_begin_[d]];
                        const size_t hi = wvvvv_pno_off_[occ_end_[d]];
                        w.wvvvv_shift = lo;
                        if (hi > lo) {
                            tracked_cudaMalloc(&w.d_Wvvvv_pno_pack, (hi - lo) * sizeof(real_t));
                            cudaMemcpyPeer(w.d_Wvvvv_pno_pack, d, d_Wvvvv_pno_pack_ + lo, 0,
                                           (hi - lo) * sizeof(real_t));
                        }
                    } else {
                        w.wvvvv_shift = 0;
                        cpy(&w.d_Wvvvv_pno_pack, d_Wvvvv_pno_pack_, wvvvv_tot);
                    }
                    cpy(&w.d_Loo_xpair, d_Loo_xpair_, no2);
                    // 5j: Wovov_lmo slab-only in slab mode. Layout [l,a,j,d] (j is the 3rd
                    // axis, stride nvir) → slicing j is a per-(l,a) deep gather. When the host
                    // array exists (F5 host-rotation path) this is a HOST repack + H2D; under
                    // the B-a.6h F5b on-device path the host array is elided, so we gather
                    // per-(l,a) directly from device-0 d_Wovov_lmo_ via cudaMemcpyPeer (the
                    // nocc·nvir tiny strided copies the host repack originally avoided, but
                    // now unavoidable and harmless at ctor time). Readers use ldB =
                    // j_extent·nvir and base (l·nvir·j_extent + (j-j_base))·nvir. else full.
                    if (use_gpu_multi_slab_ && occ_end_[d] > occ_begin_[d]) {
                        const int jb = occ_begin_[d], jext = occ_end_[d] - occ_begin_[d];
                        w.wovov_j_extent = jext; w.wovov_j_base = jb;
                        const size_t la = static_cast<size_t>(nocc_) * nvir_;   // (l,a) pairs
                        const size_t slab_sz = la * jext * nvir_;
                        tracked_cudaMalloc(&w.d_Wovov_lmo, slab_sz * sizeof(real_t));
                        if (h_Wovov_lmo_.empty()) {
                            // B-a.6h F5b multi-GPU: the host array was elided (on-device
                            // rotation). Gather the j-subrange straight from device-0
                            // d_Wovov_lmo_ ([l,a,j,d], j is the 3rd axis, stride nvir): per
                            // (l,a) the subrange [jb,jb+jext)×d is contiguous (jext·nvir block)
                            // → one peer copy each, packed to [p, j_rel, d] (p=l·nvir+a) so the
                            // result is byte-identical to the host-repack output above. Same
                            // per-row peer-copy idiom as the Wovvo/t2 slabs.
                            for (size_t p = 0; p < la; ++p)
                                cudaMemcpyPeer(
                                    w.d_Wovov_lmo + p * jext * nvir_, d,
                                    d_Wovov_lmo_ + (p * nocc_ + jb) * nvir_, 0,
                                    static_cast<size_t>(jext) * nvir_ * sizeof(real_t));
                        } else {
                            std::vector<real_t> h_slab(slab_sz);
                            #pragma omp parallel for
                            for (long long p = 0; p < static_cast<long long>(la); ++p) {  // p = l·nvir + a
                                const real_t* src = h_Wovov_lmo_.data()
                                    + (static_cast<size_t>(p) * nocc_ + jb) * nvir_;
                                std::copy(src, src + static_cast<size_t>(jext) * nvir_,
                                          h_slab.data() + static_cast<size_t>(p) * jext * nvir_);
                            }
                            cudaMemcpy(w.d_Wovov_lmo, h_slab.data(), slab_sz * sizeof(real_t),
                                       cudaMemcpyHostToDevice);
                        }
                    } else {
                        w.wovov_j_extent = nocc_; w.wovov_j_base = 0;
                        cpy(&w.d_Wovov_lmo, d_Wovov_lmo_, ovov_len);
                    }
                    // 5h: Wovvo_re slab-only in slab mode. Layout [l,j,b,c] (l outermost
                    // stride nocc·nv², j 2nd axis stride nv²); the slab restricts j to the
                    // device's output-occ range [occ_begin,occ_end) but keeps ALL l → a
                    // STRIDED per-l peer copy (each l's j-subrange is contiguous). The reader
                    // indexes it (l·j_extent + (j-j_base))·nv²; block ld stays nv. else full.
                    if (use_gpu_multi_slab_ && occ_end_[d] > occ_begin_[d]) {
                        const int jb = occ_begin_[d], jext = occ_end_[d] - occ_begin_[d];
                        w.wovvo_re_j_extent = jext; w.wovvo_re_j_base = jb;
                        const size_t slab_sz = static_cast<size_t>(nocc_) * jext * nv2;
                        tracked_cudaMalloc(&w.d_Wovvo_re, slab_sz * sizeof(real_t));
                        for (int l = 0; l < nocc_; ++l)
                            cudaMemcpyPeer(w.d_Wovvo_re + static_cast<size_t>(l) * jext * nv2, d,
                                           d_Wovvo_re_ + (static_cast<size_t>(l) * nocc_ + jb) * nv2, 0,
                                           static_cast<size_t>(jext) * nv2 * sizeof(real_t));
                    } else {
                        w.wovvo_re_j_extent = nocc_; w.wovvo_re_j_base = 0;
                        cpy(&w.d_Wovvo_re, d_Wovvo_re_, ovov_len);
                    }
                    // 5k: ovov_Llmo (nocc·nvir·nocc·nvir) is NOT replicated on d>0. It only
                    // feeds the GLOBAL tmp[K] reduction (add_ttmp stage 1), output-slab-
                    // independent; device 0 computes the full tmp[K] and broadcasts it (peer
                    // copy) before the d>0 projections. Null w.d_ovov_Llmo → add_ttmp skips
                    // its stage 1 there (the (B) memory win).
                    // 5i: t2_Jlmo slab-only in slab mode. Layout [K,j,a,b] (K outermost stride
                    // nocc·nv², j 2nd axis stride nv²); slab restricts j to [occ_begin,occ_end)
                    // keeping ALL K → STRIDED per-K peer copy. The add_ttmp GEMV reads it with
                    // lda = j_extent·nv² (FIRST cuBLAS-lda change) and A_ptr offset 0. else full.
                    if (use_gpu_multi_slab_ && occ_end_[d] > occ_begin_[d]) {
                        const int jb = occ_begin_[d], jext = occ_end_[d] - occ_begin_[d];
                        w.t2_jlmo_j_extent = jext; w.t2_jlmo_j_base = jb;
                        const size_t slab_sz = static_cast<size_t>(nocc_) * jext * nv2;
                        tracked_cudaMalloc(&w.d_t2_Jlmo, slab_sz * sizeof(real_t));
                        for (int K = 0; K < nocc_; ++K)
                            cudaMemcpyPeer(w.d_t2_Jlmo + static_cast<size_t>(K) * jext * nv2, d,
                                           d_t2_Jlmo_ + (static_cast<size_t>(K) * nocc_ + jb) * nv2, 0,
                                           static_cast<size_t>(jext) * nv2 * sizeof(real_t));
                    } else {
                        w.t2_jlmo_j_extent = nocc_; w.t2_jlmo_j_base = 0;
                        cpy(&w.d_t2_Jlmo, d_t2_Jlmo_, oovv_len);
                    }
                    cpy(&w.d_Lvv, d_Lvv_, nv2);
                    // 5l: Wvvvo_r1 slab-only in slab mode. Layout [j,a,b,c], j OUTERMOST
                    // (stride nvir³) → the device's output-occ j-slab is one CONTIGUOUS
                    // block [occ_begin·nvir³, occ_end·nvir³) → a single peer copy (no
                    // per-row strided gather). The sole reader add_tr1_gpu indexes block
                    // j at (j - wvvvo_r1_j_base_)·nvir³. else full. (7×1 GB → 7×slab.)
                    if (use_gpu_multi_slab_ && occ_end_[d] > occ_begin_[d]) {
                        const int jb = occ_begin_[d], jext = occ_end_[d] - occ_begin_[d];
                        w.wvvvo_r1_j_base = jb;
                        const size_t nv3 = static_cast<size_t>(nvir_) * nvir_ * nvir_;
                        const size_t slab_sz = static_cast<size_t>(jext) * nv3;
                        if (slab_sz && d_Wvvvo_r1_) {
                            tracked_cudaMalloc(&w.d_Wvvvo_r1, slab_sz * sizeof(real_t));
                            cudaMemcpyPeer(w.d_Wvvvo_r1, d,
                                           d_Wvvvo_r1_ + static_cast<size_t>(jb) * nv3, 0,
                                           slab_sz * sizeof(real_t));
                        }
                    } else {
                        w.wvvvo_r1_j_base = 0;
                        cpy(&w.d_Wvvvo_r1, d_Wvvvo_r1_, wvvvo_len);
                    }
                }
            }
        }
        if (num_gpus_ > 1)
            std::cout << "[bt-PNO Stage 5] DLPNOEAEOMNativeOperator: num_gpus=" << num_gpus_
                      << (use_gpu_multi_
                            ? " (Stage 5c: per-device σ2 build (full replica) + disjoint "
                              "peer gather; σ1 on device 0; GANSU_DLPNO_NATIVE_GPU_MULTI=1)"
                            : " (multi-GPU scaffolding; matvec single-device — set "
                              "GANSU_DLPNO_NATIVE_GPU_MULTI=1 with full residency to "
                              "exercise Stage 5c)")
                      << std::endl;
        // B-a.6f Stage F5 (ship): under BARE + GPU residency the Davidson matvec
        // never reads h_Wovvo_lmo_ / h_Wovov_lmo_ — use_gpu_ph1/2/3_ are all on,
        // so every skip_ph* in compute_sigma2 is true, and apply_resident
        // bypasses the host path entirely. The persistent host members were only
        // needed (i) to build d_Wovov_lmo_ + d_Wovvo_re_ at Stage 3b above and
        // (ii) for the multi-GPU slab gather just completed. Both are now done
        // → release the backing storage to relieve the 100-atom persistent host
        // footprint (~360 GB at 100 atoms; ~1.6 GB at anthracene). gpu_selfcheck_
        // keeps the host reference path live for diagnostics, so the free is
        // gated off when GANSU_DLPNO_NATIVE_GPU_VALIDATE=1.
        if (use_native_bare_ && use_gpu_resident_ && !gpu_selfcheck_ && !f5b_active) {
            const double saved_gb =
                static_cast<double>(h_Wovvo_lmo_.size() + h_Wovov_lmo_.size())
                * sizeof(real_t) / 1.0e9;
            std::vector<real_t>().swap(h_Wovvo_lmo_);   // force backing storage release
            std::vector<real_t>().swap(h_Wovov_lmo_);
            std::cout << "[bt-PNO B-a.6f EA F5] persistent host borrow released "
                         "(h_Wovvo_lmo_ + h_Wovov_lmo_, " << std::fixed
                      << std::setprecision(2) << saved_gb << " GB)"
                      << "  — device-resident d_Wovov_lmo_ + d_Wovvo_re_ supersede"
                      << std::defaultfloat << std::endl;
        } else if (f5b_active) {
            std::cout << "[bt-PNO B-a.6f EA F5b] persistent host borrow NEVER allocated "
                         "(on-device rotation; transient ctor peak also avoided)"
                      << std::endl;
        }
    } else {
        use_gpu_ = false; gpu_selfcheck_ = false; use_gpu_proj_ = false;
        use_gpu_lift_ = false; use_gpu_xpair_ = false;
        use_gpu_ph2_ = false; use_gpu_ph3_ = false; use_gpu_ph1_ = false;
        use_gpu_tmp_ = false; use_gpu_tlvv_ = false; use_gpu_tr1_ = false;
        use_gpu_s1lvv_ = false; use_gpu_s1fov_ = false; use_gpu_s1wvovv_ = false;
        use_gpu_resident_ = false;
    }
#endif
    _npmark("GPU Stage 1-4 device setup (per-occ pack + H2D)");
}

// Dump the accumulated per-term apply timings (env GANSU_DLPNO_NATIVE_PROF=1).
// Run with --num_gpus 1 → prof_calls_ = matvec count and each total is the full
// per-matvec cost of that term (the multi-GPU slab path only times a sub-slab).
void DLPNOEAEOMNativeOperator::print_profile() const {
    if (!prof_ || prof_calls_ == 0) return;
    const double n = static_cast<double>(prof_calls_);
    const double tot = prof_t_lift_ + prof_t_s1_ + prof_t_loo_ + prof_t_tlvv_ +
                       prof_t_tr1_ + prof_t_ph2_ + prof_t_ph3_ + prof_t_ph1_ +
                       prof_t_tmp_ + prof_t_proj_;
    auto row = [&](const char* name, double t) {
        std::cout << "    " << name << "  total=" << std::fixed << t << "s  per-matvec="
                  << (t / n) * 1e3 << "ms  (" << (tot > 0 ? 100.0 * t / tot : 0.0) << "%)\n";
    };
    std::cout << "[bt-PNO EA-PROF] apply per-term breakdown  matvecs=" << prof_calls_
              << "  timed-total=" << std::fixed << tot << "s\n";
    row("lift   ", prof_t_lift_);
    row("sigma1 ", prof_t_s1_);
    row("T_Loo  ", prof_t_loo_);
    row("T_Lvv  ", prof_t_tlvv_);
    row("T_r1   ", prof_t_tr1_);
    row("T_ph2  ", prof_t_ph2_);
    row("T_ph3  ", prof_t_ph3_);
    row("T_ph1  ", prof_t_ph1_);
    row("T_tmp  ", prof_t_tmp_);
    row("project(+T_vvvv)", prof_t_proj_);
}

// B-a.6f Stage F5b on-device ph borrow: build d_Wovov_lmo_ + d_Wovvo_re_ directly from
// the canonical EA-op device buffers without ever materialising the persistent host
// vectors h_Wovvo_lmo_/h_Wovov_lmo_ or the transient h_Wovvo/h_Wovov pulls. Eliminates
// the EA-ctor's ~6.4 GB transient host peak at anthracene (~1.4 TB at 100 atoms). Chunked
// over the virtual `a` axis so the device scratch stays inside a tuneable budget; identity
// uloc skips the cuBLAS GEMMs entirely. Gating + call site live in the ctor (under
// `f5b_active = use_native_bare_ && use_gpu_resident_ && !gpu_selfcheck_`). Builds the
// device-0 buffers only; under num_gpus>1 (B-a.6h) the d>0 slab gather peer-copies them.
void DLPNOEAEOMNativeOperator::borrow_ph_to_device_f5b(
    const real_t* d_Wovvo_canon, const real_t* d_Wovov_canon)
{
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const int nocc = nocc_, nvir = nvir_;
    const size_t wsz = static_cast<size_t>(nocc) * nvir * nocc * nvir;
    auto grid_for = [](size_t N) -> int {
        const size_t blocks = (N + 255) / 256;
        return static_cast<int>(std::min<size_t>(blocks, 65535));
    };
    // (A) Identity branch: pure D2D copy + transpose kernel.
    if (uloc_is_identity(U_loc_, nocc)) {
        cudaMemcpy(d_Wovov_lmo_, d_Wovov_canon, wsz * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        const size_t total = static_cast<size_t>(nocc) * nocc * nvir * nvir;
        f5b_transpose_wovvo_canon_to_re_kernel<<<grid_for(total), 256>>>(
            d_Wovvo_canon, d_Wovvo_re_, nocc, nvir);
        cudaDeviceSynchronize();
        std::cout << "[bt-PNO B-a.6f EA F5b] on-device ph borrow ON "
                     "(uloc=identity D2D + transpose; ctor borrow + Stage 3b host upload "
                     "skipped)" << std::endl;
        return;
    }

    // (B) PM rotation branch: chunked over `a`, plus an optional inner L-chunk loop
    // with β=1 accumulation for the a_chunk=1 over-budget regime (single-`a` slab
    // alone exceeds budget/3, defensive path for >>100-atom scales). Both chunk
    // dimensions accept env overrides for the 100-atom validation pass:
    //   GANSU_BA6F_BUDGET_GB  default 16, override the scratch budget (GB).
    //   GANSU_BA6F_A_CHUNK    >0 forces a_chunk (capped at nvir); skips auto.
    //   GANSU_BA6F_L_CHUNK    >0 forces L_chunk (capped at nocc); skips auto.
    // When L_chunk == nocc, the L-loop iterates once with β=0 and the GEMM call
    // sites collapse to the pre-L-chunk code path byte-for-byte (same lda/strideA/
    // batchCount/β; verified by side-by-side substitution).
    real_t* d_U_loc = nullptr;
    tracked_cudaMalloc(&d_U_loc, static_cast<size_t>(nocc) * nocc * sizeof(real_t));
    cudaMemcpy(d_U_loc, U_loc_.data(),
               static_cast<size_t>(nocc) * nocc * sizeof(real_t), cudaMemcpyHostToDevice);

    // Env overrides for the 100-atom validation pass.
    size_t budget_bytes = static_cast<size_t>(16) * 1024ull * 1024ull * 1024ull;
    const char* env_b_raw = std::getenv("GANSU_BA6F_BUDGET_GB");
    const char* env_a_raw = std::getenv("GANSU_BA6F_A_CHUNK");
    const char* env_l_raw = std::getenv("GANSU_BA6F_L_CHUNK");
    // Audit print: surfaces the literal getenv() result so env-propagation
    // failures (SLURM --export, ssh, container shells) are diagnosable from
    // the log without re-running.
    std::cout << "[bt-PNO B-a.6f EA F5b env] BUDGET_GB="
              << (env_b_raw ? env_b_raw : "(unset)")
              << "  A_CHUNK=" << (env_a_raw ? env_a_raw : "(unset)")
              << "  L_CHUNK=" << (env_l_raw ? env_l_raw : "(unset)") << std::endl;
    if (env_b_raw && env_b_raw[0]) {
        const size_t gb = std::strtoull(env_b_raw, nullptr, 10);
        if (gb > 0) budget_bytes = gb * 1024ull * 1024ull * 1024ull;
    }
    int a_chunk_force = 0, L_chunk_force = 0;
    if (env_a_raw && env_a_raw[0]) a_chunk_force = std::atoi(env_a_raw);
    if (env_l_raw && env_l_raw[0]) L_chunk_force = std::atoi(env_l_raw);

    // Chunk sizing: per-a scratch = 3 buffers × nocc²·nvir doubles in the full-L regime.
    const size_t per_a_bytes = static_cast<size_t>(nocc) * nocc * nvir * sizeof(real_t);
    int a_chunk = nvir;
    if (a_chunk_force > 0) {
        a_chunk = std::min(a_chunk_force, nvir);
    } else if (3 * per_a_bytes * static_cast<size_t>(nvir) > budget_bytes) {
        if (3 * per_a_bytes <= budget_bytes)
            a_chunk = static_cast<int>(std::max<size_t>(1, budget_bytes / (3 * per_a_bytes)));
        else
            a_chunk = 1;
    }

    // L-chunk: only activated when env-forced or a_chunk == 1 with single-`a` slab
    // over budget/3 (the defensive 200+ atom regime — d_canon shrinks to Lact rows
    // while d_rot1/d_rot2 still hold the full `l` axis to accumulate over L).
    int L_chunk = nocc;
    const size_t per_L_bytes = static_cast<size_t>(a_chunk) * nocc * nvir * sizeof(real_t);
    if (L_chunk_force > 0) {
        L_chunk = std::min(L_chunk_force, nocc);
    } else if (a_chunk == 1 && 3 * per_a_bytes > budget_bytes) {
        if (2 * per_a_bytes >= budget_bytes) {
            throw std::runtime_error(
                "[bt-PNO B-a.6f EA F5b] 2 × per_a_bytes exceeds budget; "
                "increase GANSU_BA6F_BUDGET_GB");
        }
        const size_t avail = budget_bytes - 2 * per_a_bytes;
        L_chunk = static_cast<int>(std::max<size_t>(1, avail / per_L_bytes));
        L_chunk = std::min(L_chunk, nocc);
    }

    // Buffer sizing: d_rot1/d_rot2 always hold the full `l` axis (per_a per a-slab);
    // d_canon shrinks to L_chunk rows when L-chunking is active.
    const size_t rot_elems   = static_cast<size_t>(a_chunk) * nocc * nocc * nvir;
    const size_t canon_elems = static_cast<size_t>(a_chunk) * L_chunk * nocc * nvir;
    real_t *d_canon = nullptr, *d_rot1 = nullptr, *d_rot2 = nullptr;
    tracked_cudaMalloc(&d_canon, canon_elems * sizeof(real_t));
    tracked_cudaMalloc(&d_rot1,  rot_elems   * sizeof(real_t));
    tracked_cudaMalloc(&d_rot2,  rot_elems   * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;

    // ---- Wovov pass ----
    // Stage A (per a-chunk, partial Σ over L-chunk with β=1 accumulation):
    //   T1[a_rel, l, J, d] += Σ_{L∈[L0,L0+Lact)} U[L,l]·canon[L, a_rel, J, d]
    //   per (a_rel) batched: row-major [nocc×nM1] → col-major [nM1×nocc]
    //   GEMM: T1_col[k,l] += canon_col[k,L]·U_col^T[L,l], k=(J,d) of size nM1=nocc·nvir.
    // Stage B (per a-chunk, T1 complete): T2[a_rel, l, j, d] = Σ_J U[J,j]·T1[a_rel, l, J, d]
    //   per (a_rel·l) batched: row-major [nocc×nvir] → col-major [nvir×nocc]
    //   GEMM: T2_col[d,j] = T1_col[d,J]·U_col^T[J,j].
    const long long nM1_ll = static_cast<long long>(nocc) * nvir;
    for (int a0 = 0; a0 < nvir; a0 += a_chunk) {
        const int act = std::min(a_chunk, nvir - a0);
        for (int L0 = 0; L0 < nocc; L0 += L_chunk) {
            const int Lact = std::min(L_chunk, nocc - L0);
            const size_t Nc = static_cast<size_t>(act) * Lact * nocc * nvir;
            // Step 1: gather sub-slab (a_slab × L_slab).
            f5b_gather_wovov_canon_aslab_kernel<<<grid_for(Nc), 256>>>(
                d_Wovov_canon, d_canon, nocc, nvir, a0, act, L0, Lact);
            // Step 2 (Stage A partial): per-a_rel batched L→l rotation,
            //   k = Lact, lda = nM1, strideA = Lact·nM1; B = U + L0·nocc (column-shift).
            const real_t* beta_a = (L0 == 0) ? &zero : &one;
            cublasDgemmStridedBatched(
                cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                /*m=*/ static_cast<int>(nM1_ll), /*n=*/ nocc, /*k=*/ Lact,
                &one,
                /*A=*/ d_canon, /*lda=*/ static_cast<int>(nM1_ll),
                /*strideA=*/ static_cast<long long>(Lact) * nM1_ll,
                /*B=*/ d_U_loc + static_cast<size_t>(L0) * nocc,
                /*ldb=*/ nocc, /*strideB=*/ 0,
                beta_a,
                /*C=*/ d_rot1, /*ldc=*/ static_cast<int>(nM1_ll),
                /*strideC=*/ static_cast<long long>(nocc) * nM1_ll,
                /*batchCount=*/ act);
        }
        // Step 3 (Stage B): batched J→j rotation, batchCount = act·nocc, batch stride = nocc·nvir.
        cublasDgemmStridedBatched(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            /*m=*/ nvir, /*n=*/ nocc, /*k=*/ nocc,
            &one,
            /*A=*/ d_rot1, /*lda=*/ nvir,
            /*strideA=*/ static_cast<long long>(nocc) * nvir,
            /*B=*/ d_U_loc, /*ldb=*/ nocc, /*strideB=*/ 0,
            &zero,
            /*C=*/ d_rot2, /*ldc=*/ nvir,
            /*strideC=*/ static_cast<long long>(nocc) * nvir,
            /*batchCount=*/ act * nocc);
        // Step 4: scatter the full (a-slab × full l) T2 into d_Wovov_lmo_.
        const size_t Nfull = static_cast<size_t>(act) * nocc * nocc * nvir;
        f5b_scatter_wovov_lmo_aslab_kernel<<<grid_for(Nfull), 256>>>(
            d_rot2, d_Wovov_lmo_, nocc, nvir, a0, act);
    }
    cudaDeviceSynchronize();

    // ---- Wovvo pass ---- (canonical J innermost → contract J first per L-chunk,
    // then accumulate L into d_rot2 with β=1).
    // Stage A' (per a-chunk × L-chunk, β=0 fresh-write into d_rot1's Lact slab):
    //   T1[L_rel, a_rel, d, j] = Σ_J U[J,j]·canon[L_rel, a_rel, d, J]
    //   per (L_rel·a_rel) batched: row-major [nvir×nocc] → col-major [nocc×nvir]
    //   GEMM: T1_col[j,d] = U_col[j,J]·canon_col[J,d]   (op_A=N: U[K,output]=U[J,j]).
    // Stage B' (per a-chunk × L-chunk, β=1 accumulate after first L0):
    //   T2[a_rel, l, d, j] += Σ_{L∈[L0,L0+Lact)} U[L,l]·T1[L_rel, a_rel, d, j]
    //   per (a_rel) batched STRIDED in L_rel: T1 sub-slab strides (act·nvir·nocc,
    //   nvir·nocc, nocc, 1). lda = act·nM1_wvvo (skip across L_rel for same a_rel),
    //   k = Lact, batch stride = nM1_wvvo, B = U + L0·nocc.
    const long long nM1_wvvo_ll = static_cast<long long>(nvir) * nocc;
    for (int a0 = 0; a0 < nvir; a0 += a_chunk) {
        const int act = std::min(a_chunk, nvir - a0);
        for (int L0 = 0; L0 < nocc; L0 += L_chunk) {
            const int Lact = std::min(L_chunk, nocc - L0);
            const size_t Nc = static_cast<size_t>(Lact) * act * nvir * nocc;
            // Step 1': gather sub-slab (a-slab × L-slab).
            f5b_gather_wovvo_canon_aslab_kernel<<<grid_for(Nc), 256>>>(
                d_Wovvo_canon, d_canon, nocc, nvir, a0, act, L0, Lact);
            // Step 2' (Stage A'): J→j (op_A=N reads U as U[K, output]=U[J,j]),
            //   batchCount = Lact·act, batch stride = nvir·nocc, β=0 (fresh sub-slab).
            cublasDgemmStridedBatched(
                cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                /*m=*/ nocc, /*n=*/ nvir, /*k=*/ nocc,
                &one,
                /*A=*/ d_U_loc, /*lda=*/ nocc, /*strideA=*/ 0,
                /*B=*/ d_canon, /*ldb=*/ nocc,
                /*strideB=*/ static_cast<long long>(nvir) * nocc,
                &zero,
                /*C=*/ d_rot1, /*ldc=*/ nocc,
                /*strideC=*/ static_cast<long long>(nvir) * nocc,
                /*batchCount=*/ Lact * act);
            // Step 3' (Stage B' partial): L→l, batched STRIDED per a_rel,
            //   k = Lact, B shifted to row L0; β=1 after the first L-chunk.
            const real_t* beta_b = (L0 == 0) ? &zero : &one;
            cublasDgemmStridedBatched(
                cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                /*m=*/ static_cast<int>(nM1_wvvo_ll), /*n=*/ nocc, /*k=*/ Lact,
                &one,
                /*A=*/ d_rot1, /*lda=*/ static_cast<int>(act) * static_cast<int>(nM1_wvvo_ll),
                /*strideA=*/ nM1_wvvo_ll,
                /*B=*/ d_U_loc + static_cast<size_t>(L0) * nocc,
                /*ldb=*/ nocc, /*strideB=*/ 0,
                beta_b,
                /*C=*/ d_rot2, /*ldc=*/ static_cast<int>(nM1_wvvo_ll),
                /*strideC=*/ static_cast<long long>(nocc) * nM1_wvvo_ll,
                /*batchCount=*/ act);
        }
        // Step 4': scatter+transpose the full (a-slab × full l) T2 into d_Wovvo_re_.
        const size_t Nfull = static_cast<size_t>(act) * nocc * nvir * nocc;
        f5b_scatter_wovvo_re_aslab_kernel<<<grid_for(Nfull), 256>>>(
            d_rot2, d_Wovvo_re_, nocc, nvir, a0, act);
    }
    cudaDeviceSynchronize();
    tracked_cudaFree(d_rot2);
    tracked_cudaFree(d_rot1);
    tracked_cudaFree(d_canon);
    tracked_cudaFree(d_U_loc);

    const double scratch_gb =
        (2.0 * rot_elems + canon_elems) * sizeof(real_t) / 1.0e9;
    std::cout << "[bt-PNO B-a.6f EA F5b] on-device ph borrow ON "
                 "(PM rotation, a_chunk=" << a_chunk
              << " L_chunk=" << L_chunk
              << " scratch≈"
              << std::fixed << std::setprecision(2) << scratch_gb << " GB; "
                 "ctor borrow + Stage 3b host upload skipped)"
              << std::defaultfloat << std::endl;
#else
    (void)d_Wovvo_canon; (void)d_Wovov_canon;
#endif
}

// GPU 4-index congruence Wvvvv^(jj) = U^(jj)ᵀ⊗4 · Wvvvv, per occ j, on device.
// The host congruence4 quad-loop is O(nocc·nvir⁴·n) with strided memory access (the EA
// operator-build hotspot ~100s on benzene). Here each of the 4 index transforms is a
// cublasDgemm[StridedBatched]: the contracted index is always the leading (row) index of
// a col-major view, so the transform is C[|Y|×n] = T[|Y|×nv] · Uflatᵀ[nv×n], batched over
// the already-transformed leading dims (a', then a'b', then a'b'c'). Uflat is the host
// U^(jj) [nvir×n] row-major = col-major [n×nv] → op_T gives the needed [nv×n] factor.
// Index trace matches congruence4 step-for-step (verified) + the ctor gpu_selfcheck_ gate.
void DLPNOEAEOMNativeOperator::build_dressed_vvvv_gpu(const real_t* d_Wvvvv) {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = gansu::gpu::GPUHandle::cublas();
    const int nv = nvir_;
    const size_t nv2 = static_cast<size_t>(nv) * nv, nv3 = nv2 * nv;
    const real_t one = 1.0, zero = 0.0;
    int nmax = 0;
    for (int j = 0; j < nocc_; ++j) nmax = std::max(nmax, packing_.n_pno_ii[j]);
    dressed_.Wvvvv_pno.assign(nocc_, {});
    if (nmax == 0) return;
    const size_t NM = static_cast<size_t>(nmax);
    real_t *d_U = nullptr, *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_out = nullptr;
    tracked_cudaMalloc(&d_U,   static_cast<size_t>(nv) * NM * sizeof(real_t));
    tracked_cudaMalloc(&d_A,   nv3 * NM * sizeof(real_t));                 // A[a'][bcd]
    tracked_cudaMalloc(&d_B,   nv2 * NM * NM * sizeof(real_t));            // B[a'][b'][cd]
    tracked_cudaMalloc(&d_C,   static_cast<size_t>(nv) * NM * NM * NM * sizeof(real_t)); // C[a'b'c'][d]
    tracked_cudaMalloc(&d_out, NM * NM * NM * NM * sizeof(real_t));        // out[a'b'c'd']
    for (int j = 0; j < nocc_; ++j) {
        const int n = packing_.n_pno_ii[j];
        if (n == 0) continue;
        cudaMemcpy(d_U, Uall_[j].data(), static_cast<size_t>(nv) * n * sizeof(real_t),
                   cudaMemcpyHostToDevice);
        // step1 (contract a): A[a'][bcd] = Wvvvv[a][bcd] · Uflatᵀ   (single GEMM)
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(nv3), n, nv, &one,
                    d_Wvvvv, static_cast<int>(nv3), d_U, n, &zero, d_A, static_cast<int>(nv3));
        // step2 (contract b): B[a'][b'][cd], batched over a' (n blocks)
        cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    static_cast<int>(nv2), n, nv, &one,
                    d_A, static_cast<int>(nv2), static_cast<long long>(nv3),
                    d_U, n, 0, &zero,
                    d_B, static_cast<int>(nv2), static_cast<long long>(n) * nv2, n);
        // step3 (contract c): C[a'][b'][c'][d], batched over (a',b') (n² blocks)
        cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    nv, n, nv, &one,
                    d_B, nv, static_cast<long long>(nv2),
                    d_U, n, 0, &zero,
                    d_C, nv, static_cast<long long>(n) * nv, static_cast<long long>(n) * n);
        // step4 (contract d): out[a'][b'][c'][d'], batched over (a',b',c') (n³ blocks)
        cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                    1, n, nv, &one,
                    d_C, 1, static_cast<long long>(nv),
                    d_U, n, 0, &zero,
                    d_out, 1, static_cast<long long>(n), static_cast<long long>(n) * n * n);
        const size_t n4 = static_cast<size_t>(n) * n * n * n;
        dressed_.Wvvvv_pno[j].assign(n4, 0.0);
        cudaMemcpy(dressed_.Wvvvv_pno[j].data(), d_out, n4 * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
    }
    tracked_cudaFree(d_U);   tracked_cudaFree(d_A); tracked_cudaFree(d_B);
    tracked_cudaFree(d_C);   tracked_cudaFree(d_out);
#else
    (void)d_Wvvvv;
#endif
}

DLPNOEAEOMNativeOperator::~DLPNOEAEOMNativeOperator() {
    print_profile();
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
#ifndef GANSU_CPU_ONLY
    if (d_Wvvvv_pno_pack_) tracked_cudaFree(d_Wvvvv_pno_pack_);
    if (d_r2_pack_)        tracked_cudaFree(d_r2_pack_);
    if (d_sig_pack_)       tracked_cudaFree(d_sig_pack_);
    if (d_U_pack_)         tracked_cudaFree(d_U_pack_);
    if (d_acc_)            tracked_cudaFree(d_acc_);
    if (d_T1_)             tracked_cudaFree(d_T1_);
    if (d_r2c_all_)        tracked_cudaFree(d_r2c_all_);
    if (d_lift_T_)         tracked_cudaFree(d_lift_T_);
    if (d_acc_all_)        tracked_cudaFree(d_acc_all_);
    if (d_Loo_xpair_)      tracked_cudaFree(d_Loo_xpair_);
    if (d_Wovov_lmo_)      tracked_cudaFree(d_Wovov_lmo_);
    if (d_Wovvo_re_)       tracked_cudaFree(d_Wovvo_re_);
    // B-a.6h(grouped) EA ph-ladder batched pointer arrays.
    if (d_pR_)             tracked_cudaFree(d_pR_);
    if (d_pWovov_)         tracked_cudaFree(d_pWovov_);
    if (d_pWovvo_)         tracked_cudaFree(d_pWovvo_);
    if (d_pAcc_)           tracked_cudaFree(d_pAcc_);
    if (d_ovov_Llmo_)      tracked_cudaFree(d_ovov_Llmo_);
    if (d_t2_Jlmo_)        tracked_cudaFree(d_t2_Jlmo_);
    if (d_r2c_sym_re_)     tracked_cudaFree(d_r2c_sym_re_);
    if (d_tmp_)            tracked_cudaFree(d_tmp_);
    if (d_Lvv_)            tracked_cudaFree(d_Lvv_);
    if (d_Wvvvo_r1_)       tracked_cudaFree(d_Wvvvo_r1_);
    if (d_r1_)             tracked_cudaFree(d_r1_);
    if (d_Fov_)            tracked_cudaFree(d_Fov_);
    if (d_Wvovv_)          tracked_cudaFree(d_Wvovv_);
    if (d_sigma1_)         tracked_cudaFree(d_sigma1_);
    if (d_r2c_sym_lcd_)    tracked_cudaFree(d_r2c_sym_lcd_);
    // Stage 5c: free the per-device σ2 replicas (d>0 only; ws_[0] aliases the device-0
    // members freed above, so skip it to avoid a double free).
    for (auto& w : ws_) {
        if (w.device <= 0) continue;
        MultiGpuManager::DeviceGuard guard(w.device);
        for (real_t* p : {w.d_input, w.d_r2c_all, w.d_lift_T, w.d_acc_all, w.d_sig_pack,
                          w.d_T1, w.d_r2c_sym_re, w.d_tmp, w.d_U_pack, w.d_Wvvvv_pno_pack,
                          w.d_Loo_xpair, w.d_Wovov_lmo, w.d_Wovvo_re, w.d_ovov_Llmo,
                          w.d_t2_Jlmo, w.d_Lvv, w.d_Wvvvo_r1})
            if (p) tracked_cudaFree(p);
    }
    if (cublas_) cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
#endif
}

// σ1[a] (1p sector), canonical formula (mirror of ea_eom_sigma1_full_kernel) on
// the lifted canonical r2. r2 layout r2[(l*nvir+a)*nvir + b]; Wvovv layout
// ((a*nocc+l)*nvir+c)*nvir+d.
void DLPNOEAEOMNativeOperator::compute_sigma1(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& sigma1,
    bool skip_lvv,
    bool skip_fov,
    bool skip_wvovv) const
{
    const int nocc = nocc_;
    const int nvir = nvir_;
    const size_t vstride = static_cast<size_t>(nvir);

    sigma1.assign(static_cast<size_t>(nvir), 0.0);
    // B-a.6a Stage 4: when every σ1 term runs on device, skip the host lift too
    // (the GPU reads the device-resident lifted r2; nothing left for the host).
    if (skip_lvv && skip_fov && skip_wvovv) return;

    const std::vector<real_t> r2 = ea_packed_r2_to_canonical(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, packed_r2);

    #pragma omp parallel for
    for (int a = 0; a < nvir; ++a) {
        real_t s = 0.0;
        // + Σ_c Lvv[a,c] r1[c]
        if (!skip_lvv)
            for (int c = 0; c < nvir; ++c)
                s += h_Lvv_[static_cast<size_t>(a) * nvir + c] * r1[c];
        // ± Σ Fov · r2
        if (!skip_fov)
            for (int l = 0; l < nocc; ++l)
                for (int d = 0; d < nvir; ++d) {
                    const real_t fov_ld = h_Fov_[static_cast<size_t>(l) * nvir + d];
                    s += 2.0 * fov_ld * r2[(static_cast<size_t>(l) * nvir + a) * vstride + d];
                    s -=       fov_ld * r2[(static_cast<size_t>(l) * nvir + d) * vstride + a];
                }
        // + Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2[l,c,d]
        if (!skip_wvovv)
            for (int l = 0; l < nocc; ++l)
                for (int c = 0; c < nvir; ++c)
                    for (int d = 0; d < nvir; ++d) {
                        const real_t w1 = h_Wvovv_[((static_cast<size_t>(a) * nocc + l) * nvir + c) * nvir + d];
                        const real_t w2 = h_Wvovv_[((static_cast<size_t>(a) * nocc + l) * nvir + d) * nvir + c];
                        s += (2.0 * w1 - w2) * r2[(static_cast<size_t>(l) * nvir + c) * vstride + d];
                    }
        sigma1[a] = s;
    }
}

// σ2 (2p1h), native per-pair. acc[a,b] (canonical virtuals) per output occ j,
// then σ2_packed^(jj) = U^(jj)ᵀ acc U^(jj). B-EA.2: pair-local T_Lvv + T_r1.
void DLPNOEAEOMNativeOperator::compute_sigma2(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2,
    bool skip_tvvvv,
    std::vector<real_t>* acc_export,
    const std::vector<real_t>* r2c_external,
    bool skip_loo,
    bool skip_ph2,
    bool skip_ph3,
    bool skip_ph1,
    bool skip_tmp,
    bool skip_tlvv,
    bool skip_tr1) const
{
    packed_sigma2.assign(static_cast<size_t>(total_dim_ - nvir_), 0.0);

    // B-a.6f Stage F5 defensive guard: under BARE + GPU residency the ctor
    // releases h_Wovvo_lmo_ / h_Wovov_lmo_ after device upload. The host T_ph
    // paths below would NPE if a caller forgot to set skip_ph1/2/3=true. Catch
    // it explicitly instead of silently dereferencing empty vectors.
    if ((!skip_ph1 || !skip_ph2 || !skip_ph3) &&
        (h_Wovvo_lmo_.empty() || h_Wovov_lmo_.empty())) {
        throw std::runtime_error(
            "[EA-EOM native] compute_sigma2: host T_ph path needs h_Wovvo_lmo_/"
            "h_Wovov_lmo_ but they were released by Stage F5 (BARE + GPU "
            "residency). Caller must set skip_ph1=skip_ph2=skip_ph3=true in "
            "this mode (or disable GPU residency / set NATIVE_GPU_VALIDATE=1).");
    }

    const int nocc = nocc_;
    const int nvir = nvir_;

    Eigen::Map<const RowMatXd> Cv(C_vir_.data(), nao_, nvir);
    Eigen::Map<const RowMatXd> S(h_S_.data(), nao_, nao_);
    const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]
    Eigen::Map<const RowMatXd> Lvv(h_Lvv_.data(), nvir, nvir);

    // Per-occ U^(ii) = C_virᵀ S bar_Q_ii [nvir × n_pno(ii)] and the lifted
    // r2c[i] = U^(ii) r2_packed^(ii) U^(ii)ᵀ [nvir × nvir].
    std::vector<RowMatXd> Uall(nocc), r2c(nocc);
    for (int i = 0; i < nocc; ++i) {
        const int n = packing_.n_pno_ii[i];
        r2c[i] = RowMatXd::Zero(nvir, nvir);
        if (n == 0) continue;
        if (r2c_external) {
            // Stage 3a: take the device-lifted r2c[i] (canonical nvir×nvir) directly.
            r2c[i] = Eigen::Map<const RowMatXd>(
                r2c_external->data() + static_cast<size_t>(i) * nvir * nvir, nvir, nvir);
            continue;
        }
        const int idx = res_.pair_lookup[static_cast<size_t>(i) * nocc + i];
        Eigen::Map<const RowMatXd> barQ(res_.pairs[idx].bar_Q.data(), nao_, n);
        Uall[i] = CvtS * barQ;
        Eigen::Map<const RowMatXd> r2p(packed_r2.data() + (packing_.off_i[i] - nvir_), n, n);
        r2c[i].noalias() = Uall[i] * r2p * Uall[i].transpose();
    }

    // T_tmp pre-stage: tmp[K] = Σ_{l,C,D} (2 ovov_Llmo[K,C,l,D] - ovov_Llmo[K,D,l,C]) r2c[l][C,D]
    //   (K canonical; ovov layout ((K*nvir+C)*nocc+l)*nvir+D).
    std::vector<real_t> tmp(nocc, 0.0);
    if (!skip_tmp)
      for (int K = 0; K < nocc; ++K) {
        real_t s = 0.0;
        for (int l = 0; l < nocc; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;
            const RowMatXd& R = r2c[l];
            for (int c = 0; c < nvir; ++c)
                for (int d = 0; d < nvir; ++d)
                    s += (2.0 * h_ovov_Llmo_[((static_cast<size_t>(K) * nvir + c) * nocc + l) * nvir + d]
                              - h_ovov_Llmo_[((static_cast<size_t>(K) * nvir + d) * nocc + l) * nvir + c]) * R(c, d);
        }
        tmp[K] = s;
      }

    for (int j = 0; j < nocc; ++j) {
        const int n = packing_.n_pno_ii[j];
        if (n == 0) continue;
        RowMatXd acc(nvir, nvir);
        if (skip_tlvv) {                             // T_Lvv (GPU on Stage 3c)
            acc.setZero();
        } else {
            acc = Lvv * r2c[j];                      // T_Lvv_a
            acc.noalias() += r2c[j] * Lvv.transpose();   // T_Lvv_b
        }
        // T_r1: acc[a,b] += Σ_c Wvvvo_lmo[a,b,c,j] r1[c]   (GPU on Stage 3c)
        if (!skip_tr1) {
            if (!h_Wvvvo_r1_host_.empty()) {
                // (M5c-a) contiguous pre-transposed [j,a,b,c] read + OpenMP over a.
                // Same per-(a,b) summation order over c → bit-identical.
                const real_t* Wj = h_Wvvvo_r1_host_.data()
                                 + static_cast<size_t>(j) * nvir * nvir * nvir;
                #pragma omp parallel for
                for (int a = 0; a < nvir; ++a)
                    for (int b = 0; b < nvir; ++b) {
                        const real_t* w = Wj + (static_cast<size_t>(a) * nvir + b) * nvir;
                        real_t s = 0.0;
                        for (int c = 0; c < nvir; ++c)
                            s += w[c] * r1[c];
                        acc(a, b) += s;
                    }
            } else {
                for (int a = 0; a < nvir; ++a)
                    for (int b = 0; b < nvir; ++b) {
                        const size_t base = (static_cast<size_t>(a) * nvir + b) * nvir;
                        real_t s = 0.0;
                        for (int c = 0; c < nvir; ++c)
                            s += h_Wvvvo_lmo_[(base + c) * nocc + j] * r1[c];
                        acc(a, b) += s;
                    }
            }
        }
        // Cross-pair (B-EA.3): source occ l, lifted r2c[l].
        //   T_Loo: acc -= Σ_l Loo_lmo[l,j] r2c[l]
        //   T_ph1: acc[a,b] += Σ_{l,d} (2 Wovvo_lmo[l,b,d,j] - Wovov_lmo[l,b,j,d]) r2c[l][a,d]
        //   T_ph2: acc[a,b] -= Σ_{l,c} Wovov_lmo[l,a,j,c] r2c[l][c,b]
        //   T_ph3: acc[a,b] -= Σ_{l,c} Wovvo_lmo[l,b,c,j] r2c[l][c,a]
        for (int l = 0; l < nocc; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;
            const RowMatXd& R = r2c[l];
            const real_t loo = h_Loo_lmo_[static_cast<size_t>(l) * nocc + j];
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    real_t s = skip_loo ? 0.0 : -loo * R(a, b);                // T_Loo (GPU on Stage 3b)
                    if (!skip_ph1)                                             // T_ph1 (GPU on Stage 3b)
                        for (int d = 0; d < nvir; ++d)
                            s += (2.0 * h_Wovvo_lmo_[((static_cast<size_t>(l) * nvir + b) * nvir + d) * nocc + j]
                                      - h_Wovov_lmo_[((static_cast<size_t>(l) * nvir + b) * nocc + j) * nvir + d]) * R(a, d);
                    if (!skip_ph2)                                             // T_ph2 (GPU on Stage 3b)
                        for (int c = 0; c < nvir; ++c)
                            s -= h_Wovov_lmo_[((static_cast<size_t>(l) * nvir + a) * nocc + j) * nvir + c] * R(c, b);
                    if (!skip_ph3)                                             // T_ph3 (GPU on Stage 3b)
                        for (int c = 0; c < nvir; ++c)
                            s -= h_Wovvo_lmo_[((static_cast<size_t>(l) * nvir + b) * nvir + c) * nocc + j] * R(c, a);
                    acc(a, b) += s;
                }
        }
        // T_tmp: acc[a,b] -= Σ_K tmp[K] t2_Jlmo[K,j,a,b]  (t2 layout ((K*nocc+j)*nvir+a)*nvir+b)
        // T_vvvv: acc[a,b] += Σ_{c,d} Wvvvv[a,b,c,d] r2c[j][c,d]  (Wvvvv ((a*nvir+b)*nvir+c)*nvir+d)
        // The dressed path (B-EA.6d) instead applies Wvvvv^(jj) directly in PNO
        // space after the projection below, so the dense T_vvvv is skipped here.
        {
            const RowMatXd& Rj = r2c[j];
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    real_t s = 0.0;
                    if (!skip_tmp)                                             // T_tmp (GPU on Stage 3b)
                        for (int K = 0; K < nocc; ++K)
                            s -= tmp[K] * h_t2_Jlmo_[((static_cast<size_t>(K) * nocc + j) * nvir + a) * nvir + b];
                    if (!use_dressed_pno_) {
                        const size_t wb = (static_cast<size_t>(a) * nvir + b) * nvir;
                        for (int c = 0; c < nvir; ++c)                         // T_vvvv (dense)
                            for (int d = 0; d < nvir; ++d)
                                s += h_Wvvvv_[(wb + c) * nvir + d] * Rj(c, d);
                    }
                    acc(a, b) += s;
                }
        }
        // Stage 2: export the completed acc[j] (canonical nvir×nvir) and let the
        // device do the projection + T_vvvv. acc already holds T_Lvv/T_r1/cross-
        // pair/T_tmp; the dressed path never put dense T_vvvv into acc, so the
        // device's apply_projection_gpu reproduces the host block exactly.
        if (acc_export) {
            Eigen::Map<RowMatXd>(acc_export->data() + static_cast<size_t>(j) * nvir * nvir,
                                 nvir, nvir) = acc;
            continue;
        }
        RowMatXd s2 = Uall[j].transpose() * acc * Uall[j];   // [n × n]
        if (use_dressed_pno_ && !skip_tvvvv) {
            // T_vvvv in PNO space: σ2_packed^(jj)[a',b'] += Σ_{c',d'} Wvvvv^(jj)[a',b',c',d']
            //   r2_packed^(jj)[c',d'].  Since the own-pair lift r2c[j] = U^(jj)·r2p·U^(jj)ᵀ
            //   is exact, the congruence-seeded Wvvvv^(jj) reproduces the dense T_vvvv
            //   bit-for-bit at any truncation; the native ring (B-EA.6d) makes it
            //   truncation-sensitive (true scaling) without ever building dense Wvvvv.
            Eigen::Map<const RowMatXd> r2p(packed_r2.data() + (packing_.off_i[j] - nvir_), n, n);
            const real_t* W = dressed_.Wvvvv_pno[j].data();
            for (int ap = 0; ap < n; ++ap)
                for (int bp = 0; bp < n; ++bp) {
                    real_t s = 0.0;
                    const size_t wb = (static_cast<size_t>(ap) * n + bp) * n;
                    for (int cp = 0; cp < n; ++cp)
                        for (int dp = 0; dp < n; ++dp)
                            s += W[(wb + cp) * n + dp] * r2p(cp, dp);
                    s2(ap, bp) += s;
                }
        }
        Eigen::Map<RowMatXd>(packed_sigma2.data() + (packing_.off_i[j] - nvir_), n, n) = s2;
    }
}

// B-a.6a Stage 1: GPU σ2 T_vvvv. packed_sigma2 += Σ_{c'd'} Wvvvv^(jj)[a'b'c'd']
// r2_packed^(jj)[c'd'] per output occ j, via cublasDgemv. The PNO intermediate
// is row-major [n²×n²]; cuBLAS reads it column-major (= its transpose), so op_T
// yields y = Wvvvv^(jj)·r2p — exactly the host PNO-space block (cu compute_sigma2).
void DLPNOEAEOMNativeOperator::apply_tvvvv_gpu(
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nvir_);
    if (plen == 0) return;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemset(d_sig_pack_, 0, plen * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    for (int j = 0; j < nocc_; ++j) {
        const int n = packing_.n_pno_ii[j];
        if (n == 0) continue;
        const int M = n * n;
        const size_t off = static_cast<size_t>(packing_.off_i[j]) - nvir_;  // packed r2/σ2 block start
        cublasDgemv(cublas, CUBLAS_OP_T, M, M, &one,
                    d_Wvvvv_pno_pack_ + wvvvv_pno_off_[j], M,
                    d_r2_pack_ + off, 1, &zero,
                    d_sig_pack_ + off, 1);
    }
    std::vector<real_t> add(plen, 0.0);
    cudaMemcpy(add.data(), d_sig_pack_, plen * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < plen; ++k) packed_sigma2[k] += add[k];
#else
    (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 2: GPU two-sided PNO projection + Stage-1 T_vvvv. For each output
// occ j, σ2_packed^(jj)[a'b'] = Σ_{ab} U^(jj)[a,a'] acc[j][a,b] U^(jj)[b,b']
// (= U^(jj)ᵀ acc[j] U^(jj)) then += Σ_{c'd'} Wvvvv^(jj)[a'b'c'd'] r2_packed^(jj)[c'd'].
// Everything is row-major; cuBLAS reads each buffer column-major (= its transpose),
// so the two GEMMs are arranged to land σ2 row-major (the validated convention):
//   GEMM1  T1[nv×n] = acc[nv×nv]·U[nv×n]   cublasDgemm(N,N, n,nv,nv, U,n, acc,nv, T1,n)
//   GEMM2  s2[n×n]  = Uᵀ·T1                 cublasDgemm(N,T, n,n,nv, T1,n, U,n, s2,n)  (β=0)
// then the validated Stage-1 dgemv accumulates T_vvvv into s2 (β=1).
void DLPNOEAEOMNativeOperator::apply_projection_gpu(
    const std::vector<real_t>& acc_all,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nvir_);
    if (plen == 0) return;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemset(d_sig_pack_, 0, plen * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    const int nv = nvir_;
    for (int j = 0; j < nocc_; ++j) {
        const int n = packing_.n_pno_ii[j];
        if (n == 0) continue;
        const size_t off = static_cast<size_t>(packing_.off_i[j]) - nvir_;  // packed block start
        const real_t* U = d_U_pack_ + u_pno_off_[j];                        // [nv × n] row-major
        cudaMemcpy(d_acc_, acc_all.data() + static_cast<size_t>(j) * nv * nv,
                   static_cast<size_t>(nv) * nv * sizeof(real_t), cudaMemcpyHostToDevice);
        // GEMM1: T1[nv×n] = acc·U.
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, nv, nv, &one,
                    U, n, d_acc_, nv, &zero, d_T1_, n);
        // GEMM2: s2[n×n] = Uᵀ·T1, written directly into the packed σ2 block (β=0).
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, nv, &one,
                    d_T1_, n, U, n, &zero, d_sig_pack_ + off, n);
        // T_vvvv (validated Stage 1): s2 += Wvvvv^(jj)·r2p (β=1, accumulate).
        const int M = n * n;
        cublasDgemv(cublas, CUBLAS_OP_T, M, M, &one,
                    d_Wvvvv_pno_pack_ + wvvvv_pno_off_[j], M,
                    d_r2_pack_ + off, 1, &one,
                    d_sig_pack_ + off, 1);
    }
    std::vector<real_t> add(plen, 0.0);
    cudaMemcpy(add.data(), d_sig_pack_, plen * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < plen; ++k) packed_sigma2[k] += add[k];
#else
    (void)acc_all; (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 3a: GPU source lift. For each occ l, r2c[l] = U^(ll)·r2_packed^(ll)·U^(ll)ᵀ
// (canonical nvir×nvir) — the inverse-direction chained GEMM of the Stage-2 projection:
//   GEMM1  T[nv×n]    = U[nv×n]·r2p[n×n]   cublasDgemm(N,N, n,nv,n, r2p,n, U,n, T,n)
//   GEMM2  r2c[nv×nv] = T·Uᵀ               cublasDgemm(T,N, nv,nv,n, U,n, T,n, r2c,nv)  (β=0)
// Result left device-resident in d_r2c_all_ (block l at l·nvir²) for the Stage-3b
// cross-pair contractions and copied to r2c_all for the (transitional) host build.
void DLPNOEAEOMNativeOperator::lift_r2c_gpu(
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& r2c_all) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nvir_);
    const size_t r2c_len = static_cast<size_t>(nocc_) * nvir_ * nvir_;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_ && plen > 0)
        cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r2src = resident_ ? d_r2_src_ : d_r2_pack_;  // packed r2 device source
    cudaMemset(d_r2c_all_, 0, r2c_len * sizeof(real_t));   // n_pno==0 blocks stay zero
    const real_t one = 1.0, zero = 0.0;
    const int nv = nvir_;
    for (int l = 0; l < nocc_; ++l) {
        const int n = packing_.n_pno_ii[l];
        if (n == 0) continue;
        const size_t off = static_cast<size_t>(packing_.off_i[l]) - nvir_;  // packed r2p block start
        const real_t* U = d_U_pack_ + u_pno_off_[l];                        // [nv × n] row-major
        // GEMM1: T[nv×n] = U·r2p.
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, nv, n, &one,
                    r2src + off, n, U, n, &zero, d_lift_T_, n);
        // GEMM2: r2c[nv×nv] = T·Uᵀ, written to the device-resident block (β=0).
        cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, n, &one,
                    U, n, d_lift_T_, n, &zero,
                    d_r2c_all_ + static_cast<size_t>(l) * nv * nv, nv);
    }
    if (!resident_)
        cudaMemcpy(r2c_all.data(), d_r2c_all_, r2c_len * sizeof(real_t), cudaMemcpyDeviceToHost);
#else
    (void)packed_r2; (void)r2c_all;
#endif
}

// Stage 5b: broadcast d_input to every d>0, redundantly lift each device's own
// d_r2c_all_ (identical two-GEMM loop to lift_r2c_gpu but on device d's ws_ replicas),
// and verify it matches device 0's lifted r2c. The device-0 matvec output is untouched;
// this proves the broadcast (decision B) + per-device full-r2c residency (decision A)
// before the 5c slab σ2 split + peer gather start consuming the per-device r2c. Gated
// on multi_selfcheck_ — without it the ws_ allocation alone proves the memory footprint
// and the per-device lift is skipped (no production overhead).
void DLPNOEAEOMNativeOperator::lift_r2c_multi_validate(const real_t* d_input) const {
#ifndef GANSU_CPU_ONLY
    if (!multi_selfcheck_ || ws_.empty()) return;
    auto& mgr = MultiGpuManager::instance();
    const size_t r2c_len = static_cast<size_t>(nocc_) * nvir_ * nvir_;
    const real_t one = 1.0, zero = 0.0;
    const int nv = nvir_;

    // Device-0 reference: the lift in this matvec already populated d_r2c_all_ and
    // nothing overwrites it (σ1/σ2 only read it).
    std::vector<real_t> ref(r2c_len);
    {
        MultiGpuManager::DeviceGuard guard(0);
        cudaMemcpy(ref.data(), d_r2c_all_, r2c_len * sizeof(real_t), cudaMemcpyDeviceToHost);
    }

    real_t max_diff = 0.0;
    for (int d = 1; d < static_cast<int>(ws_.size()); ++d) {
        const auto& w = ws_[d];
        if (w.device < 0) continue;
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t cublas = mgr.cublas(d);
        // Broadcast the full matvec input device 0 → ws_[d].d_input.
        cudaMemcpyPeer(w.d_input, d, d_input, 0,
                       static_cast<size_t>(total_dim_) * sizeof(real_t));
        const real_t* r2src = w.d_input + nvir_;            // packed r2 on device d
        cudaMemset(w.d_r2c_all, 0, r2c_len * sizeof(real_t));
        cudaDeviceSynchronize();                            // memset done before any GEMM writes
        for (int l = 0; l < nocc_; ++l) {
            const int n = packing_.n_pno_ii[l];
            if (n == 0) continue;
            const size_t off = static_cast<size_t>(packing_.off_i[l]) - nvir_;
            const real_t* U = w.d_U_pack + u_pno_off_[l];   // [nv × n] row-major
            // GEMM1: T[nv×n] = U·r2p.
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, nv, n, &one,
                        r2src + off, n, U, n, &zero, w.d_lift_T, n);
            // GEMM2: r2c[nv×nv] = T·Uᵀ, written to the device-resident block (β=0).
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, n, &one,
                        U, n, w.d_lift_T, n, &zero,
                        w.d_r2c_all + static_cast<size_t>(l) * nv * nv, nv);
        }
        cudaDeviceSynchronize();                            // GEMMs done before D2H
        std::vector<real_t> got(r2c_len);
        cudaMemcpy(got.data(), w.d_r2c_all, r2c_len * sizeof(real_t), cudaMemcpyDeviceToHost);
        real_t dd = 0.0;
        for (size_t k = 0; k < r2c_len; ++k) dd = std::max(dd, std::fabs(got[k] - ref[k]));
        max_diff = std::max(max_diff, dd);
    }
    std::cout << "[bt-PNO Stage 5b self-check] max|device d>0 lifted r2c - device 0| = "
              << std::scientific << max_diff
              << "  (expect ≤1e-11 = broadcast + per-device lift residency OK)" << std::endl;
#else
    (void)d_input;
#endif
}

// Stage 2/3b shared projection: per output occ j, σ2_packed^(jj) = U^(jj)ᵀ acc[j] U^(jj)
// (chained cublasDgemm) + T_vvvv, with acc[j] read from a DEVICE stack at
// d_acc_stack + j·nvir². Same convention as apply_projection_gpu, but the acc is
// already resident (no per-j host upload) — used by the Stage-3b cross-pair path.
void DLPNOEAEOMNativeOperator::project_acc_stack_gpu(
    const real_t* d_acc_stack,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nvir_);
    if (plen == 0) return;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_)
        cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r2src = resident_ ? d_r2_src_ : d_r2_pack_;  // packed r2 device source
    cudaMemset(d_sig_pack_, 0, plen * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        const int n = packing_.n_pno_ii[j];
        if (n == 0) continue;
        const size_t off = static_cast<size_t>(packing_.off_i[j]) - nvir_;
        const real_t* U = d_U_pack_ + u_pno_off_[j];
        const real_t* acc = d_acc_stack + static_cast<size_t>(j) * nv * nv;
        // GEMM1: T1[nv×n] = acc·U.
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, nv, nv, &one,
                    U, n, acc, nv, &zero, d_T1_, n);
        // GEMM2: s2[n×n] = Uᵀ·T1 (β=0) → packed σ2 block.
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, nv, &one,
                    d_T1_, n, U, n, &zero, d_sig_pack_ + off, n);
        // T_vvvv (validated Stage 1): s2 += Wvvvv^(jj)·r2p (β=1). 5e: subtract
        // wvvvv_pack_shift_ so the slab-only per-device pack is indexed correctly
        // (shift = 0 for device 0 / full / single-GPU → unchanged).
        const int M = n * n;
        cublasDgemv(cublas, CUBLAS_OP_T, M, M, &one,
                    d_Wvvvv_pno_pack_ + (wvvvv_pno_off_[j] - wvvvv_pack_shift_), M,
                    r2src + off, 1, &one,
                    d_sig_pack_ + off, 1);
    }
    if (resident_) return;   // d_sig_pack_ holds the full σ2; caller copies it D2D
    std::vector<real_t> add(plen, 0.0);
    cudaMemcpy(add.data(), d_sig_pack_, plen * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < plen; ++k) packed_sigma2[k] += add[k];
#else
    (void)d_acc_stack; (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 3b: device cross-pair T_Loo + projection. The host-built acc
// (T_Loo omitted) is uploaded to d_acc_all_, then the cross-pair T_Loo term
//   acc[j] += Σ_l (-Loo_lmo[l,j]) r2c[l]
// is added as ONE stacked GEMM ACC[nocc×nvir²] += M_Loo[nocc×nocc]·R_stack[nocc×nvir²]
// (M_Loo[j,l] = -Loo_lmo[l,j], R_stack = d_r2c_all_, β=1), and the result projected.
void DLPNOEAEOMNativeOperator::apply_xpair_projection_gpu(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& acc_all,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t acclen = static_cast<size_t>(nocc_) * nvir_ * nvir_;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    // Resident: the host builds no acc terms → start the device acc at zero (the
    // GPU cross-pair/T_Lvv/T_r1 terms below fill it). Else upload the host-built acc.
    if (resident_)
        cudaMemset(d_acc_all_, 0, acclen * sizeof(real_t));
    else
        cudaMemcpy(d_acc_all_, acc_all.data(), acclen * sizeof(real_t), cudaMemcpyHostToDevice);
    // T_Loo stacked GEMM: ACC[nocc × nvir²] += M_Loo[nocc×nocc] · R_stack[nocc × nvir²].
    // Stage 5c-step2: restrict the output-occ columns [j_lo,j_hi) to the active slab
    // (R_stack/k=nocc stays full — cross-pair reads all l). Off → all nocc columns.
    const real_t one = 1.0;
    const int nv2 = nvir_ * nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    // Per-term timer (prof_ only): sync, run, sync, accumulate. Off → just run.
    auto tick = [&](double& acc, auto&& fn) {
        if (!prof_) { fn(); return; }
        cudaDeviceSynchronize();
        const auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaDeviceSynchronize();
        acc += std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - t0).count();
    };
    tick(prof_t_loo_, [&] {
        if (j_hi > j_lo)
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv2, j_hi - j_lo, nocc_, &one,
                        d_r2c_all_, nv2, d_Loo_xpair_ + static_cast<size_t>(j_lo) * nocc_, nocc_,
                        &one, d_acc_all_ + static_cast<size_t>(j_lo) * nv2, nv2);
    });
    tick(prof_t_tlvv_, [&] { if (use_gpu_tlvv_) add_tlvv_gpu(); });
    tick(prof_t_tr1_,  [&] { if (use_gpu_tr1_)  add_tr1_gpu(r1); });
    // B-a.6h(grouped): one batched ph-ladder replaces the 3 per-(j,l) loops. Timed under
    // prof_t_ph2_ (T_ph3/T_ph1 read 0 in the PROF dump — sum them for the combined ph cost).
    if (use_ph_grouped_) {
        tick(prof_t_ph2_, [&] { add_tph_grouped_gpu(); });
    } else {
        tick(prof_t_ph2_,  [&] { if (use_gpu_ph2_)  add_tph2_gpu(); });
        tick(prof_t_ph3_,  [&] { if (use_gpu_ph3_)  add_tph3_gpu(); });
        tick(prof_t_ph1_,  [&] { if (use_gpu_ph1_)  add_tph1_gpu(); });
    }
    tick(prof_t_tmp_,  [&] { if (use_gpu_tmp_)  add_ttmp_gpu(); });
    tick(prof_t_proj_, [&] { project_acc_stack_gpu(d_acc_all_, packed_r2, packed_sigma2); });
    if (prof_) ++prof_calls_;
#else
    (void)acc_all; (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 3b T_ph2: device acc[j] -= Σ_l A_j[l]·r2c[l] where
// A_j[l][a,c] = Wovov_lmo[l,a,j,c] (a STRIDED submatrix of d_Wovov_lmo_: row-stride
// nocc·nvir, base (l·nvir·nocc+j)·nvir). One GEMM per (j,l) pair, α=-1, β=1
// accumulating onto the device acc stack. Verified by index trace:
//   C_rm[a,b] += -Σ_c Wovov_lmo[l,a,j,c]·r2c[l][c,b]  =  host T_ph2 summed over l.
void DLPNOEAEOMNativeOperator::add_tph2_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t neg_one = -1.0, one = 1.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        if (packing_.n_pno_ii[j] == 0) continue;
        real_t* C = d_acc_all_ + static_cast<size_t>(j) * nv * nv;
        for (int l = 0; l < nocc_; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;
            const real_t* R = d_r2c_all_ + static_cast<size_t>(l) * nv * nv;       // r2c[l] [nv×nv]
            // 5j: Wovov_lmo slab-only — A_j[l] base (l·nv·j_extent + (j-j_base))·nv, a-axis
            // row-stride ldB = j_extent·nv (full → (l·nv·nocc+j)·nv, ldB=nocc·nv).
            const real_t* A = d_Wovov_lmo_
                + (static_cast<size_t>(l) * nv * wovov_j_extent_ + (j - wovov_j_base_)) * nv;  // A_j[l]
            // C[a,b] += -Σ_c A[a,c] r2c[l][c,b]; A row-stride j_extent·nv, R/C contiguous nv.
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                        R, nv, A, wovov_j_extent_ * nv, &one, C, nv);
        }
    }
#endif
}

// B-a.6a Stage 3b T_ph3: device acc[a,b] -= Σ_l Σ_c Wovvo_lmo[l,b,c,j]·r2c[l](c,a)
// = -Σ_l (R[l]ᵀ·B_j[l]ᵀ)[a,b], B_j[l][b,c] = Wovvo_re[l,j,b,c] (contiguous [nvir×nvir],
// pre-transposed in the ctor). One GEMM(T,T) per (j,l), α=-1, β=1 accumulating onto
// the device acc stack. Verified by index trace:
//   acc_rm[a,b] += -Σ_c Wovvo_lmo[l,b,c,j]·r2c[l](c,a)  =  host T_ph3 summed over l.
void DLPNOEAEOMNativeOperator::add_tph3_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t neg_one = -1.0, one = 1.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        if (packing_.n_pno_ii[j] == 0) continue;
        real_t* C = d_acc_all_ + static_cast<size_t>(j) * nv * nv;
        for (int l = 0; l < nocc_; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;
            const real_t* R = d_r2c_all_ + static_cast<size_t>(l) * nv * nv;          // r2c[l] [nv×nv]
            // 5h: Wovvo_re slab-only — block (l,j) at (l·j_extent + (j-j_base))·nv²
            // (extent=nocc/base=0 on device 0 / full → original (l·nocc+j)·nv²).
            const real_t* B = d_Wovvo_re_
                + (static_cast<size_t>(l) * wovvo_re_j_extent_ + (j - wovvo_re_j_base_)) * nv * nv;  // B_j[l] [b×c]
            // C[a,b] += -(R[l]ᵀ·B_j[l]ᵀ)[a,b]; both contiguous nv, op_T on each.
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_T, nv, nv, nv, &neg_one,
                        B, nv, R, nv, &one, C, nv);
        }
    }
#endif
}

// B-a.6a Stage 3b T_ph1: device acc[a,b] += Σ_{l,d} (2 Wovvo_lmo[l,b,d,j] -
// Wovov_lmo[l,b,j,d]) r2c[l](a,d). TWO GEMM(T,N) per (j,l) (no W̃ build):
//   term A: +2·r2c[l]·Wovvo_re_j[l]ᵀ   (Wovvo_re_j[l][b,d]=Wovvo_lmo[l,b,d,j], contiguous)
//   term B: -1·r2c[l]·Wovov_j[l]ᵀ      (Wovov_j[l][b,d]=Wovov_lmo[l,b,j,d], strided ldB=nocc·nvir)
// Verified by index trace: C_rm[a,b] += (2W_ovvo - W_ovov)·r2c[l](a,d) = host T_ph1.
void DLPNOEAEOMNativeOperator::add_tph1_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t two = 2.0, neg_one = -1.0, one = 1.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        if (packing_.n_pno_ii[j] == 0) continue;
        real_t* C = d_acc_all_ + static_cast<size_t>(j) * nv * nv;
        for (int l = 0; l < nocc_; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;
            const real_t* R = d_r2c_all_ + static_cast<size_t>(l) * nv * nv;            // r2c[l] [nv×nv]
            // term A: C += 2·r2c[l]·Wovvo_re_j[l]ᵀ (contiguous [b×d], ldB=nv).
            // 5h: Wovvo_re slab-only — (l·j_extent + (j-j_base))·nv² (full → (l·nocc+j)·nv²).
            const real_t* WA = d_Wovvo_re_
                + (static_cast<size_t>(l) * wovvo_re_j_extent_ + (j - wovvo_re_j_base_)) * nv * nv;
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &two,
                        WA, nv, R, nv, &one, C, nv);
            // term B: C += -1·r2c[l]·Wovov_j[l]ᵀ (strided [b×d], ldB=j_extent·nv).
            // 5j: Wovov_lmo slab-only — base (l·nv·j_extent + (j-j_base))·nv (full → (l·nv·nocc+j)·nv).
            const real_t* WB = d_Wovov_lmo_
                + (static_cast<size_t>(l) * nv * wovov_j_extent_ + (j - wovov_j_base_)) * nv;
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                        WB, wovov_j_extent_ * nv, R, nv, &one, C, nv);
        }
    }
#endif
}

// B-a.6h(grouped): the ph-ladder (T_ph2 + T_ph3 + T_ph1) as batched cublasDgemmBatched.
// Each per-(j,l) GEMM is replicated exactly (same m/n/k=nvir, same ops, same lda/ldb/ldc,
// same α/β) but batched over the active occ j for a fixed reduction index l. Looping l
// outside and batching j inside keeps every acc[j]'s l-accumulation order identical to the
// per-(j,l) path (distinct acc[j] per batch entry → no race); ph2→ph3→ph1 with ph1's
// termA/termB interleaved per l matches the per-(j,l) A,B,A,B order → bit-exact. Single-
// device only (the pointer arrays index the device-0 buffers, built in the ctor).
void DLPNOEAEOMNativeOperator::add_tph_grouped_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0, two = 2.0, neg_one = -1.0;
    const int nv = nvir_, no = nocc_;
    const int na = ph_nact_;
    // T_ph2: C=acc[j] += -1 · R[l]·A_j[l]  (ops N,N; lda(R)=nv, ldb(Wovov)=nocc·nv, ldc=nv).
    for (int li = 0; li < na; ++li) {
        const size_t off = static_cast<size_t>(li) * na;
        cublasDgemmBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                           d_pR_ + off, nv, d_pWovov_ + off, no * nv, &one, d_pAcc_ + off, nv, na);
    }
    // T_ph3: C += -1 · Wovvo_reᵀ·Rᵀ  (ops T,T; lda(Wovvo)=nv, ldb(R)=nv, ldc=nv).
    for (int li = 0; li < na; ++li) {
        const size_t off = static_cast<size_t>(li) * na;
        cublasDgemmBatched(cublas, CUBLAS_OP_T, CUBLAS_OP_T, nv, nv, nv, &neg_one,
                           d_pWovvo_ + off, nv, d_pR_ + off, nv, &one, d_pAcc_ + off, nv, na);
    }
    // T_ph1: per l, termA (+2·Wovvo_reᵀ·R) then termB (-1·Wovovᵀ·R)  (both ops T,N).
    for (int li = 0; li < na; ++li) {
        const size_t off = static_cast<size_t>(li) * na;
        cublasDgemmBatched(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &two,
                           d_pWovvo_ + off, nv, d_pR_ + off, nv, &one, d_pAcc_ + off, nv, na);
        cublasDgemmBatched(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                           d_pWovov_ + off, no * nv, d_pR_ + off, nv, &one, d_pAcc_ + off, nv, na);
    }
#endif
}

// B-a.6a Stage 3b T_tmp: two-stage term, both stages as a single GEMV.
//   stage 1: r2c_sym_re[(c,l,d)] = 2 r2c[l](c,d) - r2c[l](d,c)  (kernel), then
//            tmp[K] = Σ_{c,l,d} ovov_Llmo[K,c,l,d]·r2c_sym_re[(c,l,d)]  (GEMV op_T,
//            ovov_Llmo is row-major [nocc_K × (c,l,d)]).
//   stage 2: acc_all[(j,a,b)] -= Σ_K t2_Jlmo[K,j,a,b]·tmp[K]  (GEMV op_N over the
//            whole acc stack, t2_Jlmo is row-major [nocc_K × (j,a,b)]).
// Verified by index trace (incl. the C↔D relabel that folds the 2nd ovov term into
// r2c_sym): reproduces the host tmp[K] + the host -Σ_K tmp[K]·t2 subtraction.
void DLPNOEAEOMNativeOperator::add_ttmp_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0, zero = 0.0, neg_one = -1.0;
    const int nv = nvir_;
    // Stage 1 (global tmp[K] reduction): computed only where ovov_Llmo is resident — device 0
    // / single-GPU. 5k: on the multi d>0 devices ovov_Llmo (nocc·nvir·nocc·nvir) is NOT
    // replicated; d_tmp_ already holds device 0's broadcast global tmp[K] (peer-copied in
    // apply_resident before the d>0 projections). Bit-identical across devices → bit-exact.
    if (d_ovov_Llmo_ != nullptr) {
        const int M = nv * nocc_ * nv;
        const int threads = 256, blocks = (M + threads - 1) / threads;
        dlpno_ea_native_r2c_sym_re_kernel<<<blocks, threads>>>(d_r2c_all_, d_r2c_sym_re_, nocc_, nv);
        cublasDgemv(cublas, CUBLAS_OP_T, M, nocc_, &one,
                    d_ovov_Llmo_, M, d_r2c_sym_re_, 1, &zero, d_tmp_, 1);
    }
    // Stage 2: acc_all -= t2_Jlmo_mat·tmp (t2_Jlmo row-major [nocc_K × (j,a,b)] →
    // col-major [(j,a,b) × nocc_K]). Stage 5c-step2: restrict the output (j,a,b) rows to the
    // active occ slab [j_lo·nv², j_hi·nv²) (K=nocc full).
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    const int m_sub = (j_hi - j_lo) * nv * nv;
    // 5i: t2_Jlmo is slab-only on d>0 → its K-axis stride (the GEMV lda) is j_extent·nv²
    // (= m_sub on the slab device since j_extent = j_hi-j_lo there), and the A_ptr offset
    // (j_lo - j_base)·nv² is 0 (storage starts at j'=0). On device 0 / full / single-GPU
    // j_extent=nocc/j_base=0 → lda = nocc·nv², offset = j_lo·nv² (the original m2 path).
    const int lda = t2_jlmo_j_extent_ * nv * nv;
    const size_t a_off = static_cast<size_t>(j_lo - t2_jlmo_j_base_) * nv * nv;
    if (m_sub > 0)
        cublasDgemv(cublas, CUBLAS_OP_N, m_sub, nocc_, &neg_one,
                    d_t2_Jlmo_ + a_off, lda,
                    d_tmp_, 1, &one, d_acc_all_ + static_cast<size_t>(j_lo) * nv * nv, 1);
#endif
}

// B-a.6a Stage 3c T_Lvv: device acc[j] += Lvv·r2c[j] + r2c[j]·Lvvᵀ, per-occ j.
//   term a: C[a,b] += Σ_c Lvv[a,c]·r2c[j][c,b]  → GEMM(N,N)
//   term b: C[a,b] += Σ_d r2c[j][a,d]·Lvv[b,d]  → GEMM(T,N) (Lvvᵀ)
// both β=1 into the device acc stack (r2c[j] = d_r2c_all_ block j, Lvv resident).
void DLPNOEAEOMNativeOperator::add_tlvv_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        if (packing_.n_pno_ii[j] == 0) continue;
        real_t* C = d_acc_all_ + static_cast<size_t>(j) * nv * nv;
        const real_t* R = d_r2c_all_ + static_cast<size_t>(j) * nv * nv;  // r2c[j] [nv×nv]
        // term a: C += Lvv·r2c[j].
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv, nv, nv, &one,
                    R, nv, d_Lvv_, nv, &one, C, nv);
        // term b: C += r2c[j]·Lvvᵀ.
        cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &one,
                    d_Lvv_, nv, R, nv, &one, C, nv);
    }
#endif
}

// B-a.6a Stage 3c T_r1: device acc[j][a,b] += Σ_c Wvvvo_lmo[a,b,c,j]·r1[c], per-occ j.
// M_j[(a,b),c] = Wvvvo_r1[j,a,b,c] (contiguous [nvir²×nvir], pre-transposed); the
// per-j GEMV op_T gives y[(a,b)] = Σ_c M_j[(a,b),c]·r1[c]. β=1 into the acc stack.
void DLPNOEAEOMNativeOperator::add_tr1_gpu(const std::vector<real_t>& r1) const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_)
        cudaMemcpy(d_r1_, r1.data(), static_cast<size_t>(nvir_) * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r1src = resident_ ? d_r1_src_ : d_r1_;   // r1 device source
    const real_t one = 1.0;
    const int nv = nvir_;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;        // Stage 5c-step2 slab
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    for (int j = j_lo; j < j_hi; ++j) {
        if (packing_.n_pno_ii[j] == 0) continue;
        // M_j row-major [nvir²×nvir] → col-major [nvir×nvir²] lda=nvir; op_T → M_j·r1.
        // 5l: d_Wvvvo_r1_ may be a j-slab (multi-GPU) → index block at (j - base)·nvir³.
        cublasDgemv(cublas, CUBLAS_OP_T, nv, nv * nv, &one,
                    d_Wvvvo_r1_ + static_cast<size_t>(j - wvvvo_r1_j_base_) * nv * nv * nv, nv,
                    r1src, 1, &one,
                    d_acc_all_ + static_cast<size_t>(j) * nv * nv, 1);
    }
#endif
}

// B-a.6a Stage 4 σ1: add the enabled GPU σ1 terms into sigma1, reusing the
// device-resident lifted r2 (d_r2c_all_, block l = r2c[l] row-major [nvir×nvir]).
//   S1LVV  : σ1[a] += Σ_c Lvv[a,c] r1[c]                        (one GEMV, op_T)
//   S1FOV  : σ1[a] += Σ_{l,d} Fov[l,d] (2 r2c[l][a][d] - r2c[l][d][a])
//            per-l: r2c[l]·Fov_l op_T (α=2) + r2c[l]ᵀ·Fov_l op_N (α=-1), both β=1
//   S1WVOVV: σ1 += Wvovv_mat·R_sym, R_sym[l,c,d]=2r2c[l](c,d)-r2c[l](d,c) (kernel),
//            Wvovv row-major [nvir×(l,c,d)] → op_T GEMV. Index traces match the host.
void DLPNOEAEOMNativeOperator::add_sigma1_gpu(
    const std::vector<real_t>& r1, std::vector<real_t>& sigma1) const
{
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const int nv = nvir_;
    const real_t one = 1.0, two = 2.0, neg_one = -1.0;
    cudaMemset(d_sigma1_, 0, static_cast<size_t>(nv) * sizeof(real_t));
    // Term 1: Lvv·r1. d_Lvv_ row-major → col-major view = Lvvᵀ; op_T → y = Lvv·r1.
    if (use_gpu_s1lvv_) {
        if (!resident_)
            cudaMemcpy(d_r1_, r1.data(), static_cast<size_t>(nv) * sizeof(real_t),
                       cudaMemcpyHostToDevice);
        const real_t* r1src = resident_ ? d_r1_src_ : d_r1_;
        cublasDgemv(cublas, CUBLAS_OP_T, nv, nv, &one, d_Lvv_, nv,
                    r1src, 1, &one, d_sigma1_, 1);
    }
    // Term 2: Fov·r2 (per occ l; r2c[l] = d_r2c_all_ block l, Fov_l = d_Fov_+l·nvir).
    if (use_gpu_s1fov_) {
        for (int l = 0; l < nocc_; ++l) {
            if (packing_.n_pno_ii[l] == 0) continue;       // zero r2c[l] block → no contribution
            const real_t* R = d_r2c_all_ + static_cast<size_t>(l) * nv * nv;
            const real_t* F = d_Fov_ + static_cast<size_t>(l) * nv;
            // +2 r2c[l]·Fov_l   (op_T → y[a] = Σ_d r2c[l][a][d] Fov_l[d])
            cublasDgemv(cublas, CUBLAS_OP_T, nv, nv, &two, R, nv, F, 1, &one, d_sigma1_, 1);
            // -1 r2c[l]ᵀ·Fov_l  (op_N → y[a] = Σ_d r2c[l][d][a] Fov_l[d])
            cublasDgemv(cublas, CUBLAS_OP_N, nv, nv, &neg_one, R, nv, F, 1, &one, d_sigma1_, 1);
        }
    }
    // Term 3: Wvovv·r2 (the dominant O(nvir⁴nocc) term). Build R_sym (l,c,d)-ordered,
    // then one GEMV: Wvovv row-major [nvir×M] → col-major (lda=M) = Wvovv_matᵀ; op_T → Wvovv_mat·R_sym.
    if (use_gpu_s1wvovv_) {
        const int M = nocc_ * nv * nv;
        const int threads = 256, blocks = (M + threads - 1) / threads;
        dlpno_ea_native_r2c_sym_lcd_kernel<<<blocks, threads>>>(d_r2c_all_, d_r2c_sym_lcd_, nocc_, nv);
        cublasDgemv(cublas, CUBLAS_OP_T, M, nv, &one, d_Wvovv_, M,
                    d_r2c_sym_lcd_, 1, &one, d_sigma1_, 1);
    }
    if (resident_) return;   // d_sigma1_ holds the full σ1; caller copies it D2D
    std::vector<real_t> add(static_cast<size_t>(nv), 0.0);
    cudaMemcpy(add.data(), d_sigma1_, static_cast<size_t>(nv) * sizeof(real_t),
               cudaMemcpyDeviceToHost);
    for (int a = 0; a < nv; ++a) sigma1[a] += add[a];
#else
    (void)r1; (void)sigma1;
#endif
}

// Stage 5c: point the σ2 members at device d's workspace (+ its cublas handle, NULL
// stream). bind_device(0) restores the device-0 members (ws_[0] aliases them). σ1
// buffers (d_sigma1_/d_Fov_/d_Wvovv_/d_r2c_sym_lcd_) stay on device 0 and are untouched.
void DLPNOEAEOMNativeOperator::bind_device(int d) const {
#ifndef GANSU_CPU_ONLY
    auto* s = const_cast<DLPNOEAEOMNativeOperator*>(this);
    const DeviceWorkspace& w = ws_[d];
    if (d != 0 && w.cublas) cublasSetStream(reinterpret_cast<cublasHandle_t>(w.cublas), 0);
    s->cublas_           = w.cublas;
    s->d_r2c_all_        = w.d_r2c_all;
    s->d_lift_T_         = w.d_lift_T;
    s->d_acc_all_        = w.d_acc_all;
    s->d_sig_pack_       = w.d_sig_pack;
    s->d_T1_             = w.d_T1;
    s->d_r2c_sym_re_     = w.d_r2c_sym_re;
    s->d_tmp_            = w.d_tmp;
    s->d_U_pack_         = w.d_U_pack;
    s->d_Wvvvv_pno_pack_ = w.d_Wvvvv_pno_pack;
    wvvvv_pack_shift_    = w.wvvvv_shift;        // 5e: slab-only Wvvvv pack offset
    s->d_Loo_xpair_      = w.d_Loo_xpair;
    s->d_Wovov_lmo_      = w.d_Wovov_lmo;
    wovov_j_extent_      = w.wovov_j_extent;     // 5j: slab-only Wovov_lmo j-slice (GEMM ldB)
    wovov_j_base_        = w.wovov_j_base;
    s->d_Wovvo_re_       = w.d_Wovvo_re;
    wovvo_re_j_extent_   = w.wovvo_re_j_extent;  // 5h: slab-only Wovvo_re j-slice params
    wovvo_re_j_base_     = w.wovvo_re_j_base;
    s->d_ovov_Llmo_      = w.d_ovov_Llmo;
    s->d_t2_Jlmo_        = w.d_t2_Jlmo;
    t2_jlmo_j_extent_    = w.t2_jlmo_j_extent;   // 5i: slab-only t2_Jlmo j-slice (GEMV lda)
    t2_jlmo_j_base_      = w.t2_jlmo_j_base;
    s->d_Lvv_            = w.d_Lvv;
    s->d_Wvvvo_r1_       = w.d_Wvvvo_r1;
    wvvvo_r1_j_base_     = w.wvvvo_r1_j_base;     // 5l: slab-only Wvvvo_r1 j base
#else
    (void)d;
#endif
}

// B-a.6a Stage 4 full residency: device-only matvec. r1 = d_input[0:nvir],
// packed_r2 = d_input[nvir:] are read straight from device (no input D2H/H2D, no
// d_r2_pack_ copy). The lift fills d_r2c_all_ (resident); add_sigma1_gpu leaves the
// full σ1 in d_sigma1_; apply_xpair_projection_gpu zeroes d_acc_all_ on device,
// adds every cross-pair/T_Lvv/T_r1 term, and projects into d_sig_pack_. The output
// is assembled with two D2D copies. NUMERICS-PRESERVING — identical math to the
// validated host-assisted path (the self-check below confirms it bit-for-bit).
//
// Stage 5c (use_gpu_multi_): each device builds the FULL σ2 into its own d_sig_pack
// (step1 = redundant compute; the slab split lands as step2); σ1 on device 0; device 0
// gathers each device's output-occ slab into d_output.
void DLPNOEAEOMNativeOperator::apply_resident(const real_t* d_input, real_t* d_output) const {
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nvir_);

    // Stage 5c multi-GPU: per-device σ2 build + disjoint output-occ gather. step2
    // (use_gpu_multi_slab_): each device computes ONLY its occ slab (lift stays full).
    // step1 (NOSLAB): each device builds the full σ2 redundantly. Both gather the same
    // disjoint slabs.
    if (use_gpu_multi_ && ws_.size() >= 2) {
        auto& mgr = MultiGpuManager::instance();
        resident_ = true;
        std::vector<real_t> unused;
        // Device 0: σ1 (device-0 only, full) + σ2 (slab when step2) into d_sig_pack_.
        {
            MultiGpuManager::DeviceGuard guard(0);
            bind_device(0);
            if (use_gpu_multi_slab_) { slab_active_ = true;
                cur_occ_begin_ = occ_begin_[0]; cur_occ_end_ = occ_end_[0]; }
            d_r1_src_ = d_input; d_r2_src_ = d_input + nvir_;
            lift_r2c_gpu(unused, unused);
            add_sigma1_gpu(unused, unused);          // σ1 ignores slab_active_ (full)
            apply_xpair_projection_gpu(unused, unused, unused, unused);
        }
        // 5k: device 0 (which holds the full ovov_Llmo) just computed the GLOBAL tmp[K] in its
        // add_ttmp stage 1; broadcast it to every d>0 (which keep NO ovov_Llmo and skip that
        // stage). Sync device 0 first so tmp is materialized; tmp is tiny ([nocc]). Bit-
        // identical across devices (same broadcast input + U) → the gathered σ stays bit-exact.
        { MultiGpuManager::DeviceGuard guard(0); cudaDeviceSynchronize(); }
        for (int d = 1; d < static_cast<int>(ws_.size()); ++d) {
            if (ws_[d].d_ovov_Llmo) continue;        // d>0 has its own (non-5k) — no broadcast
            MultiGpuManager::DeviceGuard guard(d);
            cudaMemcpyPeer(ws_[d].d_tmp, d, ws_[0].d_tmp, 0,
                           static_cast<size_t>(nocc_) * sizeof(real_t));
        }
        // Devices d>0: broadcast input, lift (full), then σ2 (slab when step2). Async
        // broadcast on device d's null stream (ordered before the lift on stream 0) so
        // the host doesn't block and the per-device pipelines overlap (sync_all below).
        for (int d = 1; d < static_cast<int>(ws_.size()); ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            bind_device(d);
            if (use_gpu_multi_slab_) { slab_active_ = true;
                cur_occ_begin_ = occ_begin_[d]; cur_occ_end_ = occ_end_[d]; }
            cudaMemcpyPeerAsync(ws_[d].d_input, d, d_input, 0,
                                static_cast<size_t>(total_dim_) * sizeof(real_t), 0);
            d_r1_src_ = ws_[d].d_input; d_r2_src_ = ws_[d].d_input + nvir_;
            lift_r2c_gpu(unused, unused);
            apply_xpair_projection_gpu(unused, unused, unused, unused);
        }
        slab_active_ = false;      // back to full for any later non-multi use
        bind_device(0);            // restore members to device 0
        mgr.sync_all();
        // Assemble: σ1 from device 0; gather each device's output-occ slab into σ2.
        {
            MultiGpuManager::DeviceGuard guard(0);
            cudaMemcpy(d_output, d_sigma1_, static_cast<size_t>(nvir_) * sizeof(real_t),
                       cudaMemcpyDeviceToDevice);
            for (int d = 0; d < static_cast<int>(ws_.size()); ++d)
                for (int j = occ_begin_[d]; j < occ_end_[d]; ++j) {
                    const int n = packing_.n_pno_ii[j];
                    if (n == 0) continue;
                    const size_t off = static_cast<size_t>(packing_.off_i[j]) - nvir_;
                    const size_t len = static_cast<size_t>(n) * n;
                    real_t* dst = d_output + nvir_ + off;
                    if (d == 0)
                        cudaMemcpy(dst, ws_[0].d_sig_pack + off, len * sizeof(real_t),
                                   cudaMemcpyDeviceToDevice);
                    else
                        cudaMemcpyPeer(dst, 0, ws_[d].d_sig_pack + off, d, len * sizeof(real_t));
                }
        }
        resident_ = false; d_r1_src_ = nullptr; d_r2_src_ = nullptr;

        if (multi_selfcheck_ && multi_check_done_ < kMultiCheckMax) {
            ++multi_check_done_;
            std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
            cudaMemcpy(h_in.data(), d_input,
                       static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
            std::vector<real_t> r1(h_in.begin(), h_in.begin() + nvir_);
            std::vector<real_t> packed_r2(h_in.begin() + nvir_, h_in.end());
            std::vector<real_t> ref1, ref2;
            compute_sigma1(r1, packed_r2, ref1);
            compute_sigma2(r1, packed_r2, ref2, /*skip_tvvvv=*/false);
            std::vector<real_t> got(static_cast<size_t>(total_dim_));
            cudaMemcpy(got.data(), d_output,
                       static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
            real_t d1 = 0.0, d2 = 0.0;
            for (int a = 0; a < nvir_; ++a) d1 = std::max(d1, std::fabs(got[a] - ref1[a]));
            for (size_t k = 0; k < plen; ++k) d2 = std::max(d2, std::fabs(got[nvir_ + k] - ref2[k]));
            std::cout << "[bt-PNO Stage 5c self-check] gathered σ vs full host: max|σ1| = "
                      << std::scientific << d1 << ", max|σ2| = " << d2
                      << "  (expect ≤1e-11 = multi-GPU gather == single-device)" << std::endl;
        }
        return;
    }

    resident_ = true;
    d_r1_src_ = d_input;             // r1     = d_input[0:nvir]
    d_r2_src_ = d_input + nvir_;     // packed_r2 = d_input[nvir:]  (d_r2_src_ + off = d_input + off_i[j])

    std::vector<real_t> unused;      // host buffers are untouched on the resident path
    if (prof_) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        lift_r2c_gpu(unused, unused);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        add_sigma1_gpu(unused, unused);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        prof_t_lift_ += std::chrono::duration<double>(t1 - t0).count();
        prof_t_s1_   += std::chrono::duration<double>(t2 - t1).count();
    } else {
        lift_r2c_gpu(unused, unused);                        // → d_r2c_all_ (resident)
        add_sigma1_gpu(unused, unused);                      // → d_sigma1_ (full σ1)
    }
    apply_xpair_projection_gpu(unused, unused, unused, unused);  // → d_sig_pack_ (full σ2)

    // Assemble d_output = [σ1 (nvir) | packed σ2 (plen)] on device.
    cudaMemcpy(d_output, d_sigma1_, static_cast<size_t>(nvir_) * sizeof(real_t),
               cudaMemcpyDeviceToDevice);
    if (plen > 0)
        cudaMemcpy(d_output + nvir_, d_sig_pack_, plen * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);

    resident_ = false;
    d_r1_src_ = nullptr; d_r2_src_ = nullptr;

    if (gpu_selfcheck_) {
        // Gate: pull r1/packed_r2 to host, run the full host path, compare against
        // the device-assembled output. Probe-independent (same matvec input).
        std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
        cudaMemcpy(h_in.data(), d_input,
                   static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
        std::vector<real_t> r1(h_in.begin(), h_in.begin() + nvir_);
        std::vector<real_t> packed_r2(h_in.begin() + nvir_, h_in.end());
        std::vector<real_t> ref1, ref2;
        compute_sigma1(r1, packed_r2, ref1);
        compute_sigma2(r1, packed_r2, ref2, /*skip_tvvvv=*/false);
        std::vector<real_t> got(static_cast<size_t>(total_dim_));
        cudaMemcpy(got.data(), d_output,
                   static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
        real_t d1 = 0.0, d2 = 0.0;
        for (int a = 0; a < nvir_; ++a) d1 = std::max(d1, std::fabs(got[a] - ref1[a]));
        for (size_t k = 0; k < plen; ++k) d2 = std::max(d2, std::fabs(got[nvir_ + k] - ref2[k]));
        std::cout << "[bt-PNO B-a.6a GPU self-check] full-residency max|σ1 - host| = "
                  << std::scientific << d1 << ", max|σ2 - host| = " << d2
                  << "  (expect ≤1e-11 = device-only matvec == host)" << std::endl;
    }
#else
    (void)d_input; (void)d_output;
#endif
}

void DLPNOEAEOMNativeOperator::apply(const real_t* d_input, real_t* d_output) const {
    // Progress logging — opt-in.  Davidson already prints per-iteration
    // eigenvalues + max|r| which is the convergence signal users want; the
    // per-matvec line is debugging-only (~300 lines/run) and noise-floor for
    // the production view.  Default OFF; set GANSU_EOM_PROGRESS=1 to enable.
    static thread_local int  ea_call_count = 0;
    static thread_local double ea_total_s = 0.0;
    const char* env_eom_prog = std::getenv("GANSU_EOM_PROGRESS");
    const bool ea_progress = env_eom_prog && env_eom_prog[0] == '1';
    const auto ea_t0 = std::chrono::high_resolution_clock::now();

    // B-a.6a Stage 4: full-residency device-only path (no host round-trip) when the
    // complete GPU term set is on. The host-assisted paths below are byte-unchanged.
    if (use_gpu_resident_) {
        apply_resident(d_input, d_output);
        if (ea_progress) {
            ++ea_call_count;
            const double dt = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - ea_t0).count();
            ea_total_s += dt;
            std::cout << "  [EA-EOM native matvec #" << ea_call_count
                      << "] last=" << std::fixed << std::setprecision(2) << dt
                      << "s  total=" << ea_total_s << "s  (resident)" << std::endl;
        }
        return;
    }

    // D2H packed input → [r1 (canonical, nvir) | packed_r2].
    std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
    cudaMemcpy(h_in.data(), d_input,
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    std::vector<real_t> r1(h_in.begin(), h_in.begin() + nvir_);
    std::vector<real_t> packed_r2(h_in.begin() + nvir_, h_in.end());

    // σ1: host builds the non-GPU terms (skip flags default false → byte-unchanged);
    // when σ1 GPU is on (⊂ use_gpu_xpair_), add_sigma1_gpu fills the skipped terms
    // below, after the lift has populated d_r2c_all_.
    std::vector<real_t> sigma1;
    compute_sigma1(r1, packed_r2, sigma1,
                   /*skip_lvv=*/use_gpu_s1lvv_, /*skip_fov=*/use_gpu_s1fov_,
                   /*skip_wvovv=*/use_gpu_s1wvovv_);

    std::vector<real_t> packed_sigma2;
    if (use_gpu_xpair_) {
        // GPU path (Stage 3b): device lift → r2c (resident), host builds acc with
        // T_Loo OMITTED, the device adds T_Loo as a stacked GEMM, then projects +
        // adds T_vvvv. NUMERICS-PRESERVING — the self-check confirms it bit-for-bit.
        std::vector<real_t> r2c_all(static_cast<size_t>(nocc_) * nvir_ * nvir_, 0.0);
        lift_r2c_gpu(packed_r2, r2c_all);
        // σ1 (Stage 4): the lift just populated d_r2c_all_; add the enabled GPU σ1
        // terms onto the host-built sigma1 (which omitted them). NUMERICS-PRESERVING.
        if (use_gpu_s1lvv_) {
            add_sigma1_gpu(r1, sigma1);
            if (gpu_selfcheck_) {
                std::vector<real_t> ref1;
                compute_sigma1(r1, packed_r2, ref1);   // full host σ1
                real_t d1 = 0.0;
                for (int a = 0; a < nvir_; ++a) d1 = std::max(d1, std::fabs(sigma1[a] - ref1[a]));
                std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ1(GPU Lvv·r1"
                          << (use_gpu_s1fov_ ? "+Fov·r2" : "")
                          << (use_gpu_s1wvovv_ ? "+Wvovv·r2" : "")
                          << ") - σ1(host)| = " << std::scientific << d1
                          << "  (expect ≤1e-11 = GPU σ1 GEMV/kernel == host)" << std::endl;
            }
        }
        std::vector<real_t> acc_all(static_cast<size_t>(nocc_) * nvir_ * nvir_, 0.0);
        std::vector<real_t> sink;
        compute_sigma2(r1, packed_r2, sink, /*skip_tvvvv=*/true, &acc_all, &r2c_all,
                       /*skip_loo=*/true, /*skip_ph2=*/use_gpu_ph2_, /*skip_ph3=*/use_gpu_ph3_,
                       /*skip_ph1=*/use_gpu_ph1_, /*skip_tmp=*/use_gpu_tmp_,
                       /*skip_tlvv=*/use_gpu_tlvv_, /*skip_tr1=*/use_gpu_tr1_);
        packed_sigma2.assign(static_cast<size_t>(total_dim_ - nvir_), 0.0);
        apply_xpair_projection_gpu(r1, acc_all, packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_tvvvv=*/false);  // full host
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU lift+T_Loo"
                      << (use_gpu_tlvv_ ? "+T_Lvv" : "") << (use_gpu_tr1_ ? "+T_r1" : "")
                      << (use_gpu_ph2_ ? "+T_ph2" : "") << (use_gpu_ph3_ ? "+T_ph3" : "")
                      << (use_gpu_ph1_ ? "+T_ph1" : "") << (use_gpu_tmp_ ? "+T_tmp" : "")
                      << "+proj+T_vvvv) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU cross-pair GEMMs == host)"
                      << std::endl;
        }
    } else if (use_gpu_lift_) {
        // GPU path (Stage 3a): device lifts the source amplitudes r2c[l] = U·r2p·Uᵀ;
        // the host acc-build consumes them (transitional round-trip), then the device
        // projects (Stage 2) + adds T_vvvv (Stage 1). NUMERICS-PRESERVING — same math
        // as the full host path; the self-check below confirms it bit-for-bit.
        std::vector<real_t> r2c_all(static_cast<size_t>(nocc_) * nvir_ * nvir_, 0.0);
        lift_r2c_gpu(packed_r2, r2c_all);
        std::vector<real_t> acc_all(static_cast<size_t>(nocc_) * nvir_ * nvir_, 0.0);
        std::vector<real_t> sink;
        compute_sigma2(r1, packed_r2, sink, /*skip_tvvvv=*/true, &acc_all, &r2c_all);
        packed_sigma2.assign(static_cast<size_t>(total_dim_ - nvir_), 0.0);
        apply_projection_gpu(acc_all, packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_tvvvv=*/false);  // full host (host lift)
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU lift+proj+T_vvvv) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU chained-GEMM lift U·r2p·Uᵀ == host)"
                      << std::endl;
        }
    } else if (use_gpu_proj_) {
        // GPU path (Stage 2): host builds acc[j] (T_Lvv/T_r1/cross-pair/T_tmp) and
        // exports it; the device does the two-sided projection Uᵀ·acc·U and adds
        // the validated Stage-1 T_vvvv. NUMERICS-PRESERVING — same math as the full
        // host path; the self-check below confirms it bit-for-bit.
        std::vector<real_t> acc_all(static_cast<size_t>(nocc_) * nvir_ * nvir_, 0.0);
        std::vector<real_t> sink;
        compute_sigma2(r1, packed_r2, sink, /*skip_tvvvv=*/true, &acc_all);
        packed_sigma2.assign(static_cast<size_t>(total_dim_ - nvir_), 0.0);
        apply_projection_gpu(acc_all, packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_tvvvv=*/false);  // full host
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU proj+T_vvvv) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU chained-GEMM projection == host Uᵀ·acc·U)"
                      << std::endl;
        }
    } else if (use_gpu_) {
        // GPU path (Stage 1): host computes σ2 with the dressed T_vvvv omitted, the
        // device adds T_vvvv. NUMERICS-PRESERVING — same math as the full host path;
        // the self-check below confirms it bit-for-bit.
        compute_sigma2(r1, packed_r2, packed_sigma2, /*skip_tvvvv=*/true);
        apply_tvvvv_gpu(packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_tvvvv=*/false);  // full host
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU T_vvvv) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU cublasDgemv == host PNO-space contraction)"
                      << std::endl;
        }
    } else {
        compute_sigma2(r1, packed_r2, packed_sigma2);
    }

    std::vector<real_t> h_out(static_cast<size_t>(total_dim_), 0.0);
    std::memcpy(h_out.data(), sigma1.data(), static_cast<size_t>(nvir_) * sizeof(real_t));
    std::memcpy(h_out.data() + nvir_, packed_sigma2.data(),
                packed_sigma2.size() * sizeof(real_t));

    cudaMemcpy(d_output, h_out.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);

    if (ea_progress) {
        ++ea_call_count;
        const double dt = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - ea_t0).count();
        ea_total_s += dt;
        std::cout << "  [EA-EOM native matvec #" << ea_call_count
                  << "] last=" << std::fixed << std::setprecision(2) << dt
                  << "s  total=" << ea_total_s << "s  (host)" << std::endl;
    }
}

void DLPNOEAEOMNativeOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        dlpno_ea_native_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

} // namespace gansu
