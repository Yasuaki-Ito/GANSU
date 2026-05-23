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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

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
#endif

void pull_device(const real_t* d_src, std::vector<real_t>& dst) {
    if (d_src == nullptr) { std::fill(dst.begin(), dst.end(), real_t(0.0)); return; }
    cudaMemcpy(dst.data(), d_src, dst.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
}
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
    // B-EA.6c/6d/6e env flags (read first so the dense-Wvvvv borrow can be
    // skipped under the true-scaling bare path). NATIVE_RING ⊂ NATIVE_DRESSED;
    // NATIVE_BARE ⊂ NATIVE_RING (W_pair seed + native-only ring, no dense nvir⁴).
    {
        const char* env = std::getenv("GANSU_DLPNO_NATIVE_DRESSED");
        use_dressed_pno_ = (env && env[0] == '1');
        const char* env_ring = std::getenv("GANSU_DLPNO_NATIVE_RING");
        use_native_ring_ = use_dressed_pno_ && (env_ring && env_ring[0] == '1');
        const char* env_bare = std::getenv("GANSU_DLPNO_NATIVE_BARE");
        use_native_bare_ = use_native_ring_ && (env_bare && env_bare[0] == '1');
        // B-a.6a GPU port (Stage 1): only meaningful for the dressed PNO path.
        const char* env_gpu = std::getenv("GANSU_DLPNO_NATIVE_GPU");
        use_gpu_ = use_dressed_pno_ && (env_gpu && env_gpu[0] == '1');
        const char* env_gv = std::getenv("GANSU_DLPNO_NATIVE_GPU_VALIDATE");
        gpu_selfcheck_ = use_gpu_ && (env_gv && env_gv[0] == '1');
        // B-a.6a GPU port (Stage 2): the two-sided PNO projection on device.
        const char* env_gp = std::getenv("GANSU_DLPNO_NATIVE_GPU_PROJ");
        use_gpu_proj_ = use_gpu_ && (env_gp && env_gp[0] == '1');
        // B-a.6a GPU port (Stage 3a): the per-matvec source lift on device.
        const char* env_gl = std::getenv("GANSU_DLPNO_NATIVE_GPU_LIFT");
        use_gpu_lift_ = use_gpu_proj_ && (env_gl && env_gl[0] == '1');
        // B-a.6a GPU port (Stage 3b): cross-pair contraction on device (T_Loo first).
        const char* env_gx = std::getenv("GANSU_DLPNO_NATIVE_GPU_XPAIR");
        use_gpu_xpair_ = use_gpu_lift_ && (env_gx && env_gx[0] == '1');
        // B-a.6a GPU port (Stage 3b T_ph2): the second cross-pair term on device.
        const char* env_gp2 = std::getenv("GANSU_DLPNO_NATIVE_GPU_PH2");
        use_gpu_ph2_ = use_gpu_xpair_ && (env_gp2 && env_gp2[0] == '1');
        // B-a.6a GPU port (Stage 3b T_ph3): the third cross-pair term on device.
        const char* env_gp3 = std::getenv("GANSU_DLPNO_NATIVE_GPU_PH3");
        use_gpu_ph3_ = use_gpu_xpair_ && (env_gp3 && env_gp3[0] == '1');
        // B-a.6a GPU port (Stage 3b T_ph1): the first cross-pair ph term on device.
        const char* env_gp1 = std::getenv("GANSU_DLPNO_NATIVE_GPU_PH1");
        use_gpu_ph1_ = use_gpu_xpair_ && (env_gp1 && env_gp1[0] == '1');
        // B-a.6a GPU port (Stage 3b T_tmp): the two-stage T_tmp term on device.
        const char* env_gtmp = std::getenv("GANSU_DLPNO_NATIVE_GPU_TMP");
        use_gpu_tmp_ = use_gpu_xpair_ && (env_gtmp && env_gtmp[0] == '1');
        // B-a.6a GPU port (Stage 3c T_Lvv): pair-local T_Lvv on device.
        const char* env_gtlvv = std::getenv("GANSU_DLPNO_NATIVE_GPU_TLVV");
        use_gpu_tlvv_ = use_gpu_xpair_ && (env_gtlvv && env_gtlvv[0] == '1');
        // B-a.6a GPU port (Stage 3c T_r1): the last host acc term on device.
        const char* env_gtr1 = std::getenv("GANSU_DLPNO_NATIVE_GPU_TR1");
        use_gpu_tr1_ = use_gpu_xpair_ && (env_gtr1 && env_gtr1[0] == '1');
        // B-a.6a GPU port (Stage 4 σ1): the 1p sector on device, reusing the
        // device-resident lifted r2 (d_r2c_all_, ⊂ use_gpu_xpair_). Nested:
        // S1LVV ⊃ S1FOV ⊃ S1WVOVV (bring up Lvv·r1 → Fov·r2 → Wvovv·r2 in order).
        const char* env_s1lvv = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1LVV");
        use_gpu_s1lvv_ = use_gpu_xpair_ && (env_s1lvv && env_s1lvv[0] == '1');
        const char* env_s1fov = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1FOV");
        use_gpu_s1fov_ = use_gpu_s1lvv_ && (env_s1fov && env_s1fov[0] == '1');
        const char* env_s1wv = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1WVOVV");
        use_gpu_s1wvovv_ = use_gpu_s1fov_ && (env_s1wv && env_s1wv[0] == '1');
        // B-a.6a GPU port (Stage 4 full residency): only when EVERY σ2 acc term + the
        // 3 σ1 terms are on device (host builds nothing) — then the per-matvec host
        // round-trips can be removed. Requires the complete set + the env flag.
        const char* env_res = std::getenv("GANSU_DLPNO_NATIVE_GPU_RESIDENT");
        use_gpu_resident_ = use_gpu_xpair_ && use_gpu_tlvv_ && use_gpu_tr1_ &&
                            use_gpu_ph1_ && use_gpu_ph2_ && use_gpu_ph3_ && use_gpu_tmp_ &&
                            use_gpu_s1lvv_ && use_gpu_s1fov_ && use_gpu_s1wvovv_ &&
                            (env_res && env_res[0] == '1');
        // Stage 5b multi-GPU: broadcast d_input + per-device lift. Only meaningful
        // with the full residency path on AND >1 device requested (rhf.get_num_gpus()).
        const char* env_multi = std::getenv("GANSU_DLPNO_NATIVE_GPU_MULTI");
        use_gpu_multi_ = use_gpu_resident_ && num_gpus_ > 1 && (env_multi && env_multi[0] == '1');
        const char* env_mv = std::getenv("GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE");
        multi_selfcheck_ = use_gpu_multi_ && (env_mv && env_mv[0] == '1');
        // Stage 5c-step2: actual compute split (default ON for multi); NOSLAB forces step1.
        const char* env_noslab = std::getenv("GANSU_DLPNO_NATIVE_GPU_MULTI_NOSLAB");
        use_gpu_multi_slab_ = use_gpu_multi_ && !(env_noslab && env_noslab[0] == '1');
#ifdef GANSU_CPU_ONLY
        use_gpu_ = false; gpu_selfcheck_ = false; use_gpu_proj_ = false;
        use_gpu_lift_ = false; use_gpu_xpair_ = false;
        use_gpu_ph2_ = false; use_gpu_ph3_ = false; use_gpu_ph1_ = false;
        use_gpu_tmp_ = false; use_gpu_tlvv_ = false; use_gpu_tr1_ = false;
        use_gpu_s1lvv_ = false; use_gpu_s1fov_ = false; use_gpu_s1wvovv_ = false;
        use_gpu_resident_ = false;
        use_gpu_multi_ = false; multi_selfcheck_ = false; use_gpu_multi_slab_ = false;
#endif
    }

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
        std::vector<real_t> h_Wvvvo(sz, 0.0);
        pull_device(ea_op.get_Wvvvo_device(), h_Wvvvo);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Wvvvo_lmo_ = std::move(h_Wvvvo);
        } else {
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

    // ph-ladder: Wovvo[l,b,d,j] (occ pos 0,3) and Wovov[l,b,j,d] (occ pos 0,2),
    // each with its two occ indices rotated canonical→LMO (copy none). Borrowed
    // dense ALREADY-T2-dressed intermediates (shared with IP).
    {
        const size_t wvvo_sz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nocc_;
        const size_t wovv_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
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
        pull_device(ea_op.get_eri_ovov_device(), h_ovov);
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
        // Seed: B-EA.6e bare = W_pair (Phase24, NO dense nvir⁴); B-EA.6d = cong(dense Wvvvv).
        if (use_native_bare_) {
            dressed_ = build_dressed_pno_ea_vvvv_bare(res_, packing_.n_pno_ii);
            std::cout << "[bt-PNO B-EA.6e] DLPNOEAEOMNativeOperator: dense-free Wvvvv^(jj) "
                         "path ON (W_pair bare seed + native-only ring, NO dense nvir⁴; "
                         "GANSU_DLPNO_NATIVE_BARE=1)" << std::endl;
        } else {
            dressed_ = build_dressed_pno_ea_vvvv(h_Wvvvv_, Uall_, packing_.n_pno_ii, nvir_);
            std::cout << "[bt-PNO B-EA.6d] DLPNOEAEOMNativeOperator: per-occ PNO dressed "
                         "Wvvvv^(jj) path ON (GANSU_DLPNO_NATIVE_DRESSED=1)" << std::endl;
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
            build_dressed_pno_ea_vvvv_ring(dressed_, res_, h_ovov_lmo2, h_t2_lmo2, Uall_,
                                           h_S_, packing_.n_pno_ii, nao_, nocc_, nvir_,
                                           ring_max_delta, ring_max_ref,
                                           /*subtract_dense=*/!use_native_bare_);
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
    }

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
                cudaMemcpy(d_Wovov_lmo_, h_Wovov_lmo_.data(), wsz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
            }
            // Stage 3b T_ph3/T_ph1: pre-transpose Wovvo_lmo[l,b,c,j] → Wovvo_re[l,j,b,c]
            // (c innermost / stride-1) so B_j[l] is a contiguous [nvir×nvir] block.
            if (use_gpu_ph3_ || use_gpu_ph1_) {
                const size_t wsz = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
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
                tracked_cudaMalloc(&d_Wovvo_re_, wsz * sizeof(real_t));
                cudaMemcpy(d_Wovvo_re_, h_Wovvo_re.data(), wsz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
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
                    cpy(&w.d_Wovov_lmo, d_Wovov_lmo_, ovov_len);
                    cpy(&w.d_Wovvo_re, d_Wovvo_re_, ovov_len);
                    cpy(&w.d_ovov_Llmo, d_ovov_Llmo_, ovov_len);
                    cpy(&w.d_t2_Jlmo, d_t2_Jlmo_, oovv_len);
                    cpy(&w.d_Lvv, d_Lvv_, nv2);
                    cpy(&w.d_Wvvvo_r1, d_Wvvvo_r1_, wvvvo_len);
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
    } else {
        use_gpu_ = false; gpu_selfcheck_ = false; use_gpu_proj_ = false;
        use_gpu_lift_ = false; use_gpu_xpair_ = false;
        use_gpu_ph2_ = false; use_gpu_ph3_ = false; use_gpu_ph1_ = false;
        use_gpu_tmp_ = false; use_gpu_tlvv_ = false; use_gpu_tr1_ = false;
        use_gpu_s1lvv_ = false; use_gpu_s1fov_ = false; use_gpu_s1wvovv_ = false;
        use_gpu_resident_ = false;
    }
#endif
}

DLPNOEAEOMNativeOperator::~DLPNOEAEOMNativeOperator() {
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
        if (!skip_tr1)
            for (int a = 0; a < nvir; ++a)
                for (int b = 0; b < nvir; ++b) {
                    const size_t base = (static_cast<size_t>(a) * nvir + b) * nvir;
                    real_t s = 0.0;
                    for (int c = 0; c < nvir; ++c)
                        s += h_Wvvvo_lmo_[(base + c) * nocc + j] * r1[c];
                    acc(a, b) += s;
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
    if (j_hi > j_lo)
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv2, j_hi - j_lo, nocc_, &one,
                    d_r2c_all_, nv2, d_Loo_xpair_ + static_cast<size_t>(j_lo) * nocc_, nocc_,
                    &one, d_acc_all_ + static_cast<size_t>(j_lo) * nv2, nv2);
    if (use_gpu_tlvv_) add_tlvv_gpu();
    if (use_gpu_tr1_)  add_tr1_gpu(r1);
    if (use_gpu_ph2_) add_tph2_gpu();
    if (use_gpu_ph3_) add_tph3_gpu();
    if (use_gpu_ph1_) add_tph1_gpu();
    if (use_gpu_tmp_) add_ttmp_gpu();
    project_acc_stack_gpu(d_acc_all_, packed_r2, packed_sigma2);
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
            const real_t* A = d_Wovov_lmo_ + (static_cast<size_t>(l) * nv * nocc_ + j) * nv;  // A_j[l]
            // C[a,b] += -Σ_c A[a,c] r2c[l][c,b]; A row-stride nocc·nv, R/C contiguous nv.
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                        R, nv, A, nocc_ * nv, &one, C, nv);
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
            const real_t* B = d_Wovvo_re_ + (static_cast<size_t>(l) * nocc_ + j) * nv * nv;  // B_j[l] [b×c]
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
            const real_t* WA = d_Wovvo_re_ + (static_cast<size_t>(l) * nocc_ + j) * nv * nv;
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &two,
                        WA, nv, R, nv, &one, C, nv);
            // term B: C += -1·r2c[l]·Wovov_j[l]ᵀ (strided [b×d], ldB=nocc·nv).
            const real_t* WB = d_Wovov_lmo_ + (static_cast<size_t>(l) * nv * nocc_ + j) * nv;
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, nv, nv, nv, &neg_one,
                        WB, nocc_ * nv, R, nv, &one, C, nv);
        }
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
    // Stage 1: r2c_sym_re kernel + tmp = ovov_mat·r2c_sym_re.
    const int M = nv * nocc_ * nv;
    const int threads = 256, blocks = (M + threads - 1) / threads;
    dlpno_ea_native_r2c_sym_re_kernel<<<blocks, threads>>>(d_r2c_all_, d_r2c_sym_re_, nocc_, nv);
    cublasDgemv(cublas, CUBLAS_OP_T, M, nocc_, &one,
                d_ovov_Llmo_, M, d_r2c_sym_re_, 1, &zero, d_tmp_, 1);
    // Stage 2: acc_all -= t2_Jlmo_mat·tmp (t2_Jlmo row-major [nocc_K × (j,a,b)] →
    // col-major [(j,a,b) × nocc_K], lda = nocc·nvir²). Stage 5c-step2: restrict the
    // output (j,a,b) rows to the active occ slab [j_lo·nv², j_hi·nv²) (k=nocc full).
    const int m2 = nocc_ * nv * nv;
    const int j_lo = slab_active_ ? cur_occ_begin_ : 0;
    const int j_hi = slab_active_ ? cur_occ_end_   : nocc_;
    const int m_sub = (j_hi - j_lo) * nv * nv;
    if (m_sub > 0)
        cublasDgemv(cublas, CUBLAS_OP_N, m_sub, nocc_, &neg_one,
                    d_t2_Jlmo_ + static_cast<size_t>(j_lo) * nv * nv, m2,
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
        cublasDgemv(cublas, CUBLAS_OP_T, nv, nv * nv, &one,
                    d_Wvvvo_r1_ + static_cast<size_t>(j) * nv * nv * nv, nv,
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
    s->d_Wovvo_re_       = w.d_Wovvo_re;
    s->d_ovov_Llmo_      = w.d_ovov_Llmo;
    s->d_t2_Jlmo_        = w.d_t2_Jlmo;
    s->d_Lvv_            = w.d_Lvv;
    s->d_Wvvvo_r1_       = w.d_Wvvvo_r1;
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
    lift_r2c_gpu(unused, unused);                            // → d_r2c_all_ (resident)
    add_sigma1_gpu(unused, unused);                          // → d_sigma1_ (full σ1)
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
    // B-a.6a Stage 4: full-residency device-only path (no host round-trip) when the
    // complete GPU term set is on. The host-assisted paths below are byte-unchanged.
    if (use_gpu_resident_) { apply_resident(d_input, d_output); return; }

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
