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
 * @file dlpno_ip_eom_native_operator.cu
 * @brief Native per-pair DLPNO-IP-EOM σ operator (stage B (a)). B-a.1: σ1.
 *        B-a.2a: + σ2 term T2 (Lvv^(ij)·r2). See the .hpp for the contract.
 */

#include "dlpno_ip_eom_native_operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#ifndef GANSU_CPU_ONLY
#include <cublas_v2.h>
#endif

#include <Eigen/Dense>

#include "dlpno_pair_data.hpp"        // PairSetup, PairData
#include "dlpno_ip_eom_transform.hpp" // ip_packed_r2_to_canonical
#include "device_host_memory.hpp"     // tracked_cudaMalloc / tracked_cudaFree
#include "gpu_manager.hpp"            // gpu::gpu_available
#include "multi_gpu_manager.hpp"      // MultiGpuManager (Stage 5 multi-GPU)

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

#ifndef GANSU_CPU_ONLY
__global__ void dlpno_ip_native_precondition_kernel(
    const real_t* __restrict__ D, const real_t* __restrict__ x,
    real_t* __restrict__ y, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        real_t d = D[idx];
        y[idx] = (fabs(d) > 1e-12) ? (x[idx] / d) : real_t(0.0);
    }
}
#endif

#ifndef GANSU_CPU_ONLY
// B-a.6a Stage 3b T8a helper: S[(k·nocc+l)·nvir+d] = 2 r2c(l,k)[d] - r2c(k,l)[d]
// — the swap-symmetrized lifted amplitude so tmp_c[c] = Σ_{kld} Woovv_lmo[k,l,d,c] S[k,l,d]
// folds the (k↔l) swap of (2 Woovv[l,k,d,c] - Woovv[k,l,d,c]). r2c(p,q) = d_r2c_all_ block
// (p·nocc+q) (row-major [nvir]); zero blocks stay zero (host n_pno skip).
__global__ void dlpno_ip_native_r2c_sym_kernel(
    const real_t* __restrict__ r2c_all, real_t* __restrict__ out, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nocc * nocc * nvir;
    if (idx >= total) return;
    const int d = idx % nvir;
    const int l = (idx / nvir) % nocc;
    const int k = idx / (nvir * nocc);
    out[idx] = 2.0 * r2c_all[(static_cast<size_t>(l) * nocc + k) * nvir + d]
                   - r2c_all[(static_cast<size_t>(k) * nocc + l) * nvir + d];
}

// B-a.6a Stage 4 σ1 Wooov·r2 helper: Ssym[(k·nocc+l)·nvir+d] = -2 r2c(k,l)[d] + r2c(l,k)[d]
// — so σ1[i] = Σ_{kld} Wooov[k,l,i,d] Ssym[k,l,d] folds the (k↔l) swap of
// (-2 Wooov[k,l,i,d] + Wooov[l,k,i,d]). r2c(p,q) = d_r2c_all_ block (p·nocc+q).
__global__ void dlpno_ip_native_ssym1_kernel(
    const real_t* __restrict__ r2c_all, real_t* __restrict__ out, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nocc * nocc * nvir;
    if (idx >= total) return;
    const int d = idx % nvir;
    const int l = (idx / nvir) % nocc;
    const int k = idx / (nvir * nocc);
    out[idx] = -2.0 * r2c_all[(static_cast<size_t>(k) * nocc + l) * nvir + d]
                    + r2c_all[(static_cast<size_t>(l) * nocc + k) * nvir + d];
}
#endif

// D2H copy of a device buffer into a host vector (sized by the caller).
void pull_device(const real_t* d_src, std::vector<real_t>& dst) {
    if (d_src == nullptr) { std::fill(dst.begin(), dst.end(), real_t(0.0)); return; }
    cudaMemcpy(dst.data(), d_src, dst.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
}

// Rotate the two LEADING occupied indices of a [nocc² × ntrail] row-major array
// canonical→LMO:  out[k,l,t] = Σ_IJ U_loc[I,k] U_loc[J,l] in[I,J,t] = (U_locᵀ M U_loc)[k,l]
// per trailing index t. Caller copies for the identity case.
void rotate_occ2(const std::vector<real_t>& in, std::vector<real_t>& out,
                 const std::vector<real_t>& U_loc, int nocc, int ntrail) {
    out.assign(in.size(), 0.0);
    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Uloc(U_loc.data(), nocc, nocc);
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> M(nocc, nocc);
    for (int t = 0; t < ntrail; ++t) {
        for (int I = 0; I < nocc; ++I)
            for (int J = 0; J < nocc; ++J)
                M(I, J) = in[(static_cast<size_t>(I) * nocc + J) * ntrail + t];
        const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            C = Uloc.transpose() * M * Uloc;
        for (int k = 0; k < nocc; ++k)
            for (int l = 0; l < nocc; ++l)
                out[(static_cast<size_t>(k) * nocc + l) * ntrail + t] = C(k, l);
    }
}

// U_loc handling identical to ip_packed_r2_to_canonical: identity if absent/wrong size.
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
}  // namespace

DLPNOIPEOMNativeOperator::DLPNOIPEOMNativeOperator(
    const IPEOMCCSDOperator& ip_op,
    const DLPNOLMP2Result& res,
    const DLPNOIPPacking& packing,
    const std::vector<real_t>& U_loc,
    const std::vector<real_t>& C_vir,
    const real_t* h_S,
    int nao, int nvir,
    const std::vector<real_t>& eps_o,
    int num_gpus)
    : res_(res), packing_(packing),
      U_loc_(U_loc), C_vir_(C_vir),
      h_S_(h_S, h_S + static_cast<size_t>(nao) * nao),
      nao_(nao), nvir_(nvir), nocc_(packing.nocc),
      total_dim_(packing.total_dim),
      num_gpus_(num_gpus > 1 ? num_gpus : 1)   // Stage 5 multi-GPU (scaffolding; ≥1)
{
    // B-a.6c env flags (read first so the dense Wovvo/Wovov borrow can be skipped
    // under the dense-free bare path). NATIVE_RING ⊂ NATIVE_DRESSED;
    // NATIVE_BARE ⊂ NATIVE_RING (Phase24 bare seed + native-only ring, no dense
    // nocc²·nvir² Wovvo/Wovov). Mirrors the EA B-EA.6e wiring.
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
        // B-a.6a GPU port (Stage 2): the single-sided PNO projection on device.
        const char* env_gp = std::getenv("GANSU_DLPNO_NATIVE_GPU_PROJ");
        use_gpu_proj_ = use_gpu_ && (env_gp && env_gp[0] == '1');
        // B-a.6a GPU port (Stage 3a): the per-matvec source lift on device.
        const char* env_gl = std::getenv("GANSU_DLPNO_NATIVE_GPU_LIFT");
        use_gpu_lift_ = use_gpu_proj_ && (env_gl && env_gl[0] == '1');
        // B-a.6a GPU port (Stage 3b): cross-pair T3/T4/T5 on device.
        const char* env_gx = std::getenv("GANSU_DLPNO_NATIVE_GPU_XPAIR");
        use_gpu_xpair_ = use_gpu_lift_ && (env_gx && env_gx[0] == '1');
        // B-a.6a GPU port (Stage 3b T1/T8): the remaining canonical-acc terms.
        const char* env_t1 = std::getenv("GANSU_DLPNO_NATIVE_GPU_T1");
        use_gpu_t1_ = use_gpu_xpair_ && (env_t1 && env_t1[0] == '1');
        const char* env_t8 = std::getenv("GANSU_DLPNO_NATIVE_GPU_T8");
        use_gpu_t8_ = use_gpu_xpair_ && (env_t8 && env_t8[0] == '1');
        // B-a.6a GPU port (Stage 3b T6/T7): the PNO-space ph-ladder.
        const char* env_phl = std::getenv("GANSU_DLPNO_NATIVE_GPU_PHL");
        use_gpu_phl_ = use_gpu_xpair_ && (env_phl && env_phl[0] == '1');
        // B-a.6a GPU port (Stage 4 σ1): the 1h sector on device (⊂ use_gpu_xpair_).
        const char* env_s1loo = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1LOO");
        use_gpu_s1loo_ = use_gpu_xpair_ && (env_s1loo && env_s1loo[0] == '1');
        const char* env_s1fov = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1FOV");
        use_gpu_s1fov_ = use_gpu_xpair_ && (env_s1fov && env_s1fov[0] == '1');
        const char* env_s1wov = std::getenv("GANSU_DLPNO_NATIVE_GPU_S1WOOOV");
        use_gpu_s1wooov_ = use_gpu_xpair_ && (env_s1wov && env_s1wov[0] == '1');
        // B-a.6a GPU port (Stage 4 residency): only when EVERY σ2 + 3 σ1 terms on device.
        const char* env_res = std::getenv("GANSU_DLPNO_NATIVE_GPU_RESIDENT");
        use_gpu_resident_ = use_gpu_xpair_ && use_gpu_t1_ && use_gpu_t8_ && use_gpu_phl_ &&
                            use_gpu_s1loo_ && use_gpu_s1fov_ && use_gpu_s1wooov_ &&
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
        use_gpu_t1_ = false; use_gpu_t8_ = false; use_gpu_phl_ = false;
        use_gpu_s1loo_ = false; use_gpu_s1fov_ = false; use_gpu_s1wooov_ = false;
        use_gpu_resident_ = false;
        use_gpu_multi_ = false; multi_selfcheck_ = false; use_gpu_multi_slab_ = false;
#endif
    }

    // Borrow the canonical dressed intermediates (bit-identical to ip_op; valid
    // because ip_op was built with a non-null MO ERI → build_dressed_intermediates ran).
    h_Loo_.assign(static_cast<size_t>(nocc_) * nocc_, 0.0);
    h_Fov_.assign(static_cast<size_t>(nocc_) * nvir_, 0.0);
    h_Wooov_.assign(static_cast<size_t>(nocc_) * nocc_ * nocc_ * nvir_, 0.0);
    h_Lvv_.assign(static_cast<size_t>(nvir_) * nvir_, 0.0);
    pull_device(ip_op.get_Loo_device(),   h_Loo_);
    pull_device(ip_op.get_Fov_device(),   h_Fov_);
    pull_device(ip_op.get_Wooov_device(), h_Wooov_);
    pull_device(ip_op.get_Lvv_device(),   h_Lvv_);

    // Wovoo [nocc·nvir·nocc²], occ I,J rotated canonical→LMO for the T1 term:
    //   Wovoo_lmo[k,a,i,j] = Σ_IJ U_loc[I,i] U_loc[J,j] Wovoo[k,a,I,J]
    // Layout (k*nvir+a)*nocc² + I*nocc + J; the [I,J] block per (k,a) is the
    // contiguous nocc² rotated by U_loc. Identity (copy) for localizer none.
    {
        const size_t wsz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nocc_;
        std::vector<real_t> h_Wovoo(wsz, 0.0);
        pull_device(ip_op.get_Wovoo_device(), h_Wovoo);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Wovoo_lmo_ = std::move(h_Wovoo);
        } else {
            h_Wovoo_lmo_.assign(wsz, 0.0);
            Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);  // U[I,i]
            const size_t nocc2 = static_cast<size_t>(nocc_) * nocc_;
            for (int k = 0; k < nocc_; ++k)
                for (int a = 0; a < nvir_; ++a) {
                    const size_t base = (static_cast<size_t>(k) * nvir_ + a) * nocc2;
                    Eigen::Map<const RowMatXd> M(h_Wovoo.data() + base, nocc_, nocc_);  // M[I,J]
                    const RowMatXd C = Uloc.transpose() * M * Uloc;                     // C[i,j]
                    for (int i = 0; i < nocc_; ++i)
                        for (int j = 0; j < nocc_; ++j)
                            h_Wovoo_lmo_[base + static_cast<size_t>(i) * nocc_ + j] = C(i, j);
                }
        }
    }

    // Loo [nocc²], occ rotated canonical→LMO for the cross-pair T3/T4 terms:
    //   Loo_lmo[k,i] = Σ_KI U_loc[K,k] U_loc[I,i] Loo[K,I] = (U_locᵀ Loo U_loc)[k,i]
    // (h_Loo_ is the canonical Loo borrowed for σ1; σ1 keeps using canonical.)
    if (uloc_is_identity(U_loc_, nocc_)) {
        h_Loo_lmo_ = h_Loo_;
    } else {
        Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);  // U[K,k]
        Eigen::Map<const RowMatXd> Loo(h_Loo_.data(), nocc_, nocc_);   // Loo[K,I]
        const RowMatXd C = Uloc.transpose() * Loo * Uloc;              // C[k,i]
        h_Loo_lmo_.assign(static_cast<size_t>(nocc_) * nocc_, 0.0);
        for (int k = 0; k < nocc_; ++k)
            for (int i = 0; i < nocc_; ++i)
                h_Loo_lmo_[static_cast<size_t>(k) * nocc_ + i] = C(k, i);
    }

    // Woooo [nocc⁴], all four occ indices rotated canonical→LMO for T5:
    //   Woooo_lmo[k,l,i,j] = Σ_KLIJ U_loc[K,k]U_loc[L,l]U_loc[I,i]U_loc[J,j] Woooo[K,L,I,J]
    // Layout ((k*nocc+l)*nocc+i)*nocc+j. Done as four sequential 1-index
    // transforms (each O(nocc⁵)); identity (copy) for localizer none.
    {
        const size_t no4 = static_cast<size_t>(nocc_) * nocc_ * nocc_ * nocc_;
        std::vector<real_t> h_Woooo(no4, 0.0);
        pull_device(ip_op.get_Woooo_device(), h_Woooo);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Woooo_lmo_ = std::move(h_Woooo);
        } else {
            const int no = nocc_;
            auto at = [no](const std::vector<real_t>& W, int p, int q, int r, int s) -> real_t {
                return W[((static_cast<size_t>(p) * no + q) * no + r) * no + s];
            };
            const std::vector<real_t>& U = U_loc_;  // U[X,x] = U[X*no + x]
            std::vector<real_t> W1(no4, 0.0), W2(no4, 0.0), W3(no4, 0.0);
            h_Woooo_lmo_.assign(no4, 0.0);
            // index 1: K→k
            for (int k = 0; k < no; ++k) for (int L = 0; L < no; ++L)
              for (int I = 0; I < no; ++I) for (int J = 0; J < no; ++J) {
                real_t s = 0.0;
                for (int K = 0; K < no; ++K) s += U[K * no + k] * at(h_Woooo, K, L, I, J);
                W1[((static_cast<size_t>(k) * no + L) * no + I) * no + J] = s;
              }
            // index 2: L→l
            for (int k = 0; k < no; ++k) for (int l = 0; l < no; ++l)
              for (int I = 0; I < no; ++I) for (int J = 0; J < no; ++J) {
                real_t s = 0.0;
                for (int L = 0; L < no; ++L) s += U[L * no + l] * at(W1, k, L, I, J);
                W2[((static_cast<size_t>(k) * no + l) * no + I) * no + J] = s;
              }
            // index 3: I→i
            for (int k = 0; k < no; ++k) for (int l = 0; l < no; ++l)
              for (int i = 0; i < no; ++i) for (int J = 0; J < no; ++J) {
                real_t s = 0.0;
                for (int I = 0; I < no; ++I) s += U[I * no + i] * at(W2, k, l, I, J);
                W3[((static_cast<size_t>(k) * no + l) * no + i) * no + J] = s;
              }
            // index 4: J→j
            for (int k = 0; k < no; ++k) for (int l = 0; l < no; ++l)
              for (int i = 0; i < no; ++i) for (int j = 0; j < no; ++j) {
                real_t s = 0.0;
                for (int J = 0; J < no; ++J) s += U[J * no + j] * at(W3, k, l, i, J);
                h_Woooo_lmo_[((static_cast<size_t>(k) * no + l) * no + i) * no + j] = s;
              }
        }
    }

    // T8: eri_oovv (Woovv) and CCSD T2, each with the 2 leading occ indices
    // rotated canonical→LMO (copy for localizer none). Both are [nocc²·nvir²].
    {
        const size_t sz = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
        std::vector<real_t> h_Woovv(sz, 0.0), h_t2(sz, 0.0);
        pull_device(ip_op.get_eri_oovv_device(), h_Woovv);
        pull_device(ip_op.get_t2_device(),       h_t2);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Woovv_lmo_ = std::move(h_Woovv);
            h_t2_lmo_    = std::move(h_t2);
        } else {
            rotate_occ2(h_Woovv, h_Woovv_lmo_, U_loc_, nocc_, nvir_ * nvir_);
            rotate_occ2(h_t2,    h_t2_lmo_,    U_loc_, nocc_, nvir_ * nvir_);
        }
    }

    // T6/T7: Wovvo[L,a,d,J] (occ at pos 0,3) and Wovov[L,a,J,d] (occ at pos 0,2),
    // each with its two occ indices rotated canonical→LMO (copy for none). These
    // are the dense ALREADY-T2-dressed intermediates borrowed from ip_op. SKIPPED
    // under the IP dense-free bare path (B-a.6c bare): the per-pair PNO Wovvo_pno/
    // Wovov_pno come from Phase24 bare + native ring, so this nocc²·nvir² borrow is
    // never needed (the whole point at 100 atoms). Mirrors EA's h_Wvvvv_ skip.
    if (!use_native_bare_) {
        const size_t wvvo_sz = static_cast<size_t>(nocc_) * nvir_ * nvir_ * nocc_;
        const size_t wovv_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
        std::vector<real_t> h_Wovvo(wvvo_sz, 0.0), h_Wovov(wovv_sz, 0.0);
        pull_device(ip_op.get_Wovvo_device(), h_Wovvo);
        pull_device(ip_op.get_Wovov_device(), h_Wovov);
        if (uloc_is_identity(U_loc_, nocc_)) {
            h_Wovvo_lmo_ = std::move(h_Wovvo);
            h_Wovov_lmo_ = std::move(h_Wovov);
        } else {
            Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);
            RowMatXd M(nocc_, nocc_);
            h_Wovvo_lmo_.assign(wvvo_sz, 0.0);
            h_Wovov_lmo_.assign(wovv_sz, 0.0);
            // Wovvo[L,a,d,J]: gather [L,J] per (a,d), rotate U_locᵀ M U_loc, scatter.
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
            // Wovov[L,a,J,d]: gather [L,J] per (a,d), rotate, scatter.
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

    // Koopmans/PNO diagonal (1h -ε_i, 2h1p -F_ii-F_jj+Λ_a). Preconditioner only
    // (the σ2 matvec is built term by term, NOT this diagonal placeholder).
    h_diagonal_.assign(static_cast<size_t>(total_dim_), 0.0);
    for (int i = 0; i < nocc_; ++i) h_diagonal_[i] = -eps_o[i];
    const int n_pairs = static_cast<int>(res_.pairs.size());
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = packing_.n_pno[idx];
        if (n == 0) continue;
        const real_t Fii = res_.setups[idx].F_ii;
        const real_t Fjj = res_.setups[idx].F_jj;
        const std::vector<real_t>& Lam = res_.pairs[idx].Lambda;
        const int o_ij = packing_.off_ij[idx];
        for (int a = 0; a < n; ++a) h_diagonal_[o_ij + a] = -Fii - Fjj + Lam[a];
        if (!packing_.diagonal(idx)) {
            const int o_ji = packing_.off_ji[idx];
            for (int a = 0; a < n; ++a) h_diagonal_[o_ji + a] = -Fjj - Fii + Lam[a];
        }
    }
    tracked_cudaMalloc(&d_diagonal_, static_cast<size_t>(total_dim_) * sizeof(real_t));
    cudaMemcpy(d_diagonal_, h_diagonal_.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);

    // B-a.6c true-scaling path (env GANSU_DLPNO_NATIVE_DRESSED=1; flags read at the
    // ctor top). Precompute the fixed per-pair virtual transforms U^(ij) = C_virᵀ S
    // bar_Q_ij once, then the per-pair PNO intermediates. Off by default →
    // dense-borrow path, bit-exact.
    if (use_dressed_pno_) {
        Eigen::Map<const RowMatXd> Cv(C_vir_.data(), nao_, nvir_);
        Eigen::Map<const RowMatXd> S(h_S_.data(), nao_, nao_);
        const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]
        const int n_pairs = static_cast<int>(res_.pairs.size());
        Uall_.assign(n_pairs, {});
        for (int idx = 0; idx < n_pairs; ++idx) {
            const int n = packing_.n_pno[idx];
            if (n == 0) continue;
            Eigen::Map<const RowMatXd> barQ(res_.pairs[idx].bar_Q.data(), nao_, n);
            const RowMatXd U = CvtS * barQ;                // [nvir × n_pno]
            Uall_[idx].assign(static_cast<size_t>(nvir_) * n, 0.0);
            for (int a = 0; a < nvir_; ++a)
                for (int d = 0; d < n; ++d)
                    Uall_[idx][static_cast<size_t>(a) * n + d] = U(a, d);
        }
        dressed_ = build_dressed_pno_ip(h_Lvv_, Uall_, packing_.n_pno, nvir_);

        // ph-ladder (T6/T7) per-pair PNO seed. Dense-free (B-a.6c bare): the bare
        // Wovvo_pno/Wovov_pno come straight from Phase24 (W_ovvo_bare/W_oovv_bare),
        // NO dense nocc²·nvir² Wovvo/Wovov. Otherwise (b1): congruence of the dense
        // already-dressed Wovvo/Wovov borrowed above.
        if (use_native_bare_) {
            build_dressed_pno_ip_bare(dressed_, res_, packing_.n_pno, nocc_);
            std::cout << "[bt-PNO B-a.6c] DLPNOIPEOMNativeOperator: dense-free ph-ladder "
                         "Wovvo/Wovov^(ij) path ON (Phase24 bare seed + native-only ring, "
                         "NO dense nocc²·nvir²; GANSU_DLPNO_NATIVE_BARE=1)" << std::endl;

            // Opt-in machine-ε validation of the bare-seed index convention
            // (GANSU_DLPNO_IP_NATIVE_VALIDATE only — pulls the dense canonical
            // eri_ovvo/eri_oovv [nocc²·nvir², gate scale] for the check, NOT in a
            // production NATIVE_EOM run). The Phase24 W_ovvo_bare/W_oovv_bare must
            // equal the congruence U^(ij)ᵀ (canonical bare ERI) U^(ij) — the ONLY
            // thing the new extract kernel could get wrong (probe-independent,
            // truncation-independent, T1-independent). Mirrors EA's W_pair==cong.
            const char* env_v = std::getenv("GANSU_DLPNO_IP_NATIVE_VALIDATE");
            if (env_v && env_v[0] == '1') {
                const size_t NV = static_cast<size_t>(nvir_), NO = static_cast<size_t>(nocc_);
                std::vector<real_t> h_ovvo(NO * NV * NV * NO, 0.0), h_oovv(NO * NO * NV * NV, 0.0);
                pull_device(ip_op.get_eri_ovvo_device(), h_ovvo);
                pull_device(ip_op.get_eri_oovv_device(), h_oovv);
                real_t dmax_vvo = 0.0, dmax_vov = 0.0;
                RowMatXd M(nvir_, nvir_);
                for (int idx = 0; idx < n_pairs; ++idx) {
                    const int n = packing_.n_pno[idx];
                    if (n == 0) continue;
                    Eigen::Map<const RowMatXd> U(Uall_[idx].data(), nvir_, n);
                    const int occ[2] = { res_.setups[idx].i, res_.setups[idx].j };
                    const std::vector<real_t>* vvo[2] = { &res_.phase24.W_ovvo_bare_i[idx],
                                                          &res_.phase24.W_ovvo_bare_j[idx] };
                    const std::vector<real_t>* vov[2] = { &res_.phase24.W_oovv_bare_i[idx],
                                                          &res_.phase24.W_oovv_bare_j[idx] };
                    const size_t need = static_cast<size_t>(nocc_) * n * n;
                    for (int r = 0; r < 2; ++r) {
                        const int I = occ[r];
                        for (int m = 0; m < nocc_; ++m) {
                            // ovvo bare: M(a,d) = (m d|a I) = h_ovvo[m,d,a,I]
                            for (int a = 0; a < nvir_; ++a)
                                for (int d = 0; d < nvir_; ++d)
                                    M(a, d) = h_ovvo[((static_cast<size_t>(m) * NV + d) * NV + a) * NO + I];
                            RowMatXd C = U.transpose() * M * U;            // [n×n]
                            const size_t base = static_cast<size_t>(m) * n * n;
                            if (vvo[r]->size() == need)
                                for (int ap = 0; ap < n; ++ap)
                                    for (int dp = 0; dp < n; ++dp)
                                        dmax_vvo = std::max(dmax_vvo, std::fabs(
                                            C(ap, dp) - (*vvo[r])[base + static_cast<size_t>(ap) * n + dp]));
                            // oovv bare: M(a,d) = (m I|a d) = h_oovv[m,I,a,d]
                            for (int a = 0; a < nvir_; ++a)
                                for (int d = 0; d < nvir_; ++d)
                                    M(a, d) = h_oovv[((static_cast<size_t>(m) * NO + I) * NV + a) * NV + d];
                            C = U.transpose() * M * U;
                            if (vov[r]->size() == need)
                                for (int ap = 0; ap < n; ++ap)
                                    for (int dp = 0; dp < n; ++dp)
                                        dmax_vov = std::max(dmax_vov, std::fabs(
                                            C(ap, dp) - (*vov[r])[base + static_cast<size_t>(ap) * n + dp]));
                        }
                    }
                }
                std::cout << "[bt-PNO B-a.6c bare-VALIDATE] max|Phase24_bare - cong(canonical bare ERI)|: "
                             "ovvo = " << std::scientific << dmax_vvo
                          << "  oovv = " << dmax_vov
                          << "  (expect ~1e-13 = index bit-exact; large = ovvo/oovv kernel mismatch)"
                          << std::endl;
            }
        } else {
            std::vector<int> pair_i(n_pairs), pair_j(n_pairs);
            for (int idx = 0; idx < n_pairs; ++idx) {
                pair_i[idx] = res_.setups[idx].i;
                pair_j[idx] = res_.setups[idx].j;
            }
            build_dressed_pno_ip_phladder(dressed_, h_Wovvo_lmo_, h_Wovov_lmo_, Uall_,
                                          pair_i, pair_j, packing_.n_pno, nocc_, nvir_);
            std::cout << "[bt-PNO B-a.6c] DLPNOIPEOMNativeOperator: per-pair PNO dressed "
                         "path ON (Lvv^(ij) + ph-ladder Wovvo/Wovov^(ij) in PNO space; "
                         "GANSU_DLPNO_NATIVE_DRESSED=1)" << std::endl;
        }

        // B-a.6c(b2): swap the ring part of the congruence-seeded ph-ladder W for
        // a native Phase24 + two-sided-barS build (true scaling). Borrow the raw
        // (ov|ov) and rotate its two occ indices canonical→LMO (copy for none).
        if (use_native_ring_) {
            // The dense ring DR (and thus the raw (ov|ov) borrow + rotation) is only
            // needed for the b2 validate gate (subtract_dense). The dense-free bare
            // path is native-only → leave h_ovov_lmo_ empty (no dense materialised).
            if (!use_native_bare_) {
                const size_t ovov_sz = static_cast<size_t>(nocc_) * nvir_ * nocc_ * nvir_;
                std::vector<real_t> h_ovov(ovov_sz, 0.0);
                pull_device(ip_op.get_eri_ovov_device(), h_ovov);
                if (uloc_is_identity(U_loc_, nocc_)) {
                    h_ovov_lmo_ = std::move(h_ovov);
                } else {
                    // ovov[p,a,q,b] layout ((p·nvir+a)·nocc+q)·nvir+b, occ at pos 0,2.
                    Eigen::Map<const RowMatXd> Uloc(U_loc_.data(), nocc_, nocc_);
                    RowMatXd M(nocc_, nocc_);
                    h_ovov_lmo_.assign(ovov_sz, 0.0);
                    for (int a = 0; a < nvir_; ++a)
                        for (int b = 0; b < nvir_; ++b) {
                            for (int p = 0; p < nocc_; ++p)
                                for (int q = 0; q < nocc_; ++q)
                                    M(p, q) = h_ovov[((static_cast<size_t>(p) * nvir_ + a) * nocc_ + q) * nvir_ + b];
                            const RowMatXd C = Uloc.transpose() * M * Uloc;
                            for (int m = 0; m < nocc_; ++m)
                                for (int nn = 0; nn < nocc_; ++nn)
                                    h_ovov_lmo_[((static_cast<size_t>(m) * nvir_ + a) * nocc_ + nn) * nvir_ + b] = C(m, nn);
                        }
                }
            }
            real_t ring_max_delta = 0.0, ring_max_ref = 0.0;
            build_dressed_pno_ip_ring(dressed_, res_, h_ovov_lmo_, h_t2_lmo_, Uall_,
                                      h_S_, packing_.n_pno, nao_, nocc_, nvir_,
                                      ring_max_delta, ring_max_ref,
                                      /*subtract_dense=*/!use_native_bare_);
            if (use_native_bare_)
                std::cout << "[bt-PNO B-a.6c] native ph-ladder ring added (Phase24 "
                             "V_ovov_pair + two-sided barS, NO dense DR): max|native_ring| = "
                          << std::scientific << ring_max_ref
                          << "  (Wovvo/Wovov^(ij) = Phase24 bare + native_ring; W2 T1 "
                             "terms deferred)" << std::endl;
            else
                std::cout << "[bt-PNO B-a.6c(b2)] native ring ON (Phase24 V_ovov_pair + "
                             "two-sided barS; GANSU_DLPNO_NATIVE_RING=1): "
                             "max|native_ring - cong(dense_ring)| = "
                          << std::scientific << ring_max_delta
                          << "  (|cong(dense_ring)|_max = " << ring_max_ref
                          << "; →0 at full PNO, else = truncation correction)" << std::endl;
        }
    }

    // B-a.6a GPU port — Stage 1 setup. Pack the per-pair Lvv^(ij) into one device
    // buffer + offsets, allocate packed r2/σ2 scratch, create a cuBLAS handle.
    // dressed_.Lvv_pno is built above (use_gpu_ ⊂ use_dressed_pno_).
#ifndef GANSU_CPU_ONLY
    if (use_gpu_ && gpu::gpu_available()) {
        const int n_pairs = static_cast<int>(res_.pairs.size());
        lvv_pno_off_.assign(n_pairs + 1, 0);
        for (int idx = 0; idx < n_pairs; ++idx) {
            const size_t n = static_cast<size_t>(packing_.n_pno[idx]);
            lvv_pno_off_[idx + 1] = lvv_pno_off_[idx] + n * n;       // n_pno²
        }
        const size_t wtot = lvv_pno_off_[n_pairs];
        const size_t plen = static_cast<size_t>(total_dim_ - nocc_);
        cublasHandle_t cublas = nullptr;
        cublasCreate(&cublas);
        cublas_ = cublas;
        if (wtot > 0) {
            tracked_cudaMalloc(&d_Lvv_pno_pack_, wtot * sizeof(real_t));
            for (int idx = 0; idx < n_pairs; ++idx) {
                const size_t sz = lvv_pno_off_[idx + 1] - lvv_pno_off_[idx];
                if (sz == 0) continue;
                cudaMemcpy(d_Lvv_pno_pack_ + lvv_pno_off_[idx],
                           dressed_.Lvv_pno[idx].data(), sz * sizeof(real_t),
                           cudaMemcpyHostToDevice);
            }
        }
        if (plen > 0) {
            tracked_cudaMalloc(&d_r2_pack_,  plen * sizeof(real_t));
            tracked_cudaMalloc(&d_sig_pack_, plen * sizeof(real_t));
        }
        // Stage 2: pack the per-pair U^(ij) [nvir × n_pno] (row-major, as built in
        // Uall_), count orientation slots, and allocate the exported-acc buffer.
        if (use_gpu_proj_) {
            u_pno_off_.assign(n_pairs + 1, 0);
            for (int idx = 0; idx < n_pairs; ++idx) {
                const int n = packing_.n_pno[idx];
                u_pno_off_[idx + 1] = u_pno_off_[idx] + static_cast<size_t>(nvir_) * n;
            }
            const size_t utot = u_pno_off_[n_pairs];
            if (utot > 0) {
                tracked_cudaMalloc(&d_U_pack_, utot * sizeof(real_t));
                for (int idx = 0; idx < n_pairs; ++idx) {
                    const size_t sz = u_pno_off_[idx + 1] - u_pno_off_[idx];
                    if (sz == 0) continue;
                    cudaMemcpy(d_U_pack_ + u_pno_off_[idx], Uall_[idx].data(),
                               sz * sizeof(real_t), cudaMemcpyHostToDevice);
                }
            }
            n_orient_ = 0;
            for (int idx = 0; idx < n_pairs; ++idx) {
                if (packing_.n_pno[idx] == 0) continue;
                n_orient_ += packing_.diagonal(idx) ? 1 : 2;
            }
            if (n_orient_ > 0)
                tracked_cudaMalloc(&d_acc_all_,
                                   static_cast<size_t>(n_orient_) * nvir_ * sizeof(real_t));
            // Stage 3a: device-resident lifted source r2c [nocc²·nvir].
            if (use_gpu_lift_)
                tracked_cudaMalloc(&d_r2c_all_,
                                   static_cast<size_t>(nocc_) * nocc_ * nvir_ * sizeof(real_t));
            // Per-orientation (oi,oj,idx,off) metadata, in the same iteration order
            // as compute_sigma2 / apply_projection_gpu (idx, ij then ji).
            orient_oi_.clear(); orient_oj_.clear(); orient_idx_.clear(); orient_off_.clear();
            max_n_pno_ = 0;
            for (int idx = 0; idx < n_pairs; ++idx) {
                const int n = packing_.n_pno[idx];
                max_n_pno_ = std::max(max_n_pno_, n);
                if (n == 0) continue;
                const int i = res_.setups[idx].i, j = res_.setups[idx].j;
                orient_oi_.push_back(i); orient_oj_.push_back(j);
                orient_idx_.push_back(idx); orient_off_.push_back(packing_.off_ij[idx] - nocc_);
                if (!packing_.diagonal(idx)) {
                    orient_oi_.push_back(j); orient_oj_.push_back(i);
                    orient_idx_.push_back(idx); orient_off_.push_back(packing_.off_ji[idx] - nocc_);
                }
            }
            // Stage 3b: Loo_lmo (T3/T4) + Woooo_lmo (T5) to device.
            if (use_gpu_xpair_) {
                const size_t no2 = static_cast<size_t>(nocc_) * nocc_;
                const size_t no4 = no2 * no2;
                tracked_cudaMalloc(&d_Loo_lmo_, no2 * sizeof(real_t));
                cudaMemcpy(d_Loo_lmo_, h_Loo_lmo_.data(), no2 * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_Woooo_lmo_, no4 * sizeof(real_t));
                cudaMemcpy(d_Woooo_lmo_, h_Woooo_lmo_.data(), no4 * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            // Stage 3b T1: pre-transpose Wovoo_lmo[k,a,oi,oj] → Wovoo_re[oi,oj,a,k]
            // (contiguous [nvir×nocc] per slot (oi,oj)) + the r1 upload scratch.
            if (use_gpu_t1_) {
                const int no = nocc_, nv = nvir_;
                const size_t no2 = static_cast<size_t>(no) * no;
                const size_t wsz = no2 * nv * no;
                std::vector<real_t> h_Wovoo_re(wsz, 0.0);
                #pragma omp parallel for
                for (int oi = 0; oi < no; ++oi)
                    for (int oj = 0; oj < no; ++oj)
                        for (int a = 0; a < nv; ++a)
                            for (int k = 0; k < no; ++k)
                                h_Wovoo_re[((static_cast<size_t>(oi) * no + oj) * nv + a) * no + k] =
                                    h_Wovoo_lmo_[(static_cast<size_t>(k) * nv + a) * no2
                                                 + static_cast<size_t>(oi) * no + oj];
                tracked_cudaMalloc(&d_Wovoo_re_, wsz * sizeof(real_t));
                cudaMemcpy(d_Wovoo_re_, h_Wovoo_re.data(), wsz * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_r1_, static_cast<size_t>(no) * sizeof(real_t));
            }
            // Stage 3b T8: Woovv_lmo + t2_lmo to device + the S/tmp_c scratch.
            if (use_gpu_t8_) {
                const size_t sz = static_cast<size_t>(nocc_) * nocc_ * nvir_ * nvir_;
                tracked_cudaMalloc(&d_Woovv_lmo_, sz * sizeof(real_t));
                cudaMemcpy(d_Woovv_lmo_, h_Woovv_lmo_.data(), sz * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_t2_lmo_, sz * sizeof(real_t));
                cudaMemcpy(d_t2_lmo_, h_t2_lmo_.data(), sz * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_S_, static_cast<size_t>(nocc_) * nocc_ * nvir_ * sizeof(real_t));
                tracked_cudaMalloc(&d_tmp_c_, static_cast<size_t>(nvir_) * sizeof(real_t));
            }
            // Stage 3b T6/T7: pack the 4 per-pair PNO ph-ladder W (each [nocc·n²]) +
            // the one-sided barS RP scratch.
            if (use_gpu_phl_) {
                wovvo_off_.assign(n_pairs + 1, 0);
                for (int idx = 0; idx < n_pairs; ++idx) {
                    const size_t n = static_cast<size_t>(packing_.n_pno[idx]);
                    wovvo_off_[idx + 1] = wovvo_off_[idx] + static_cast<size_t>(nocc_) * n * n;
                }
                const size_t wtot = wovvo_off_[n_pairs];
                auto pack_W = [&](real_t** dptr, const std::vector<std::vector<real_t>>& src) {
                    tracked_cudaMalloc(dptr, wtot * sizeof(real_t));
                    for (int idx = 0; idx < n_pairs; ++idx) {
                        const size_t sz = wovvo_off_[idx + 1] - wovvo_off_[idx];
                        if (sz == 0) continue;
                        cudaMemcpy(*dptr + wovvo_off_[idx], src[idx].data(), sz * sizeof(real_t),
                                   cudaMemcpyHostToDevice);
                    }
                };
                if (wtot > 0) {
                    pack_W(&d_Wovvo_occi_, dressed_.Wovvo_pno_occi);
                    pack_W(&d_Wovvo_occj_, dressed_.Wovvo_pno_occj);
                    pack_W(&d_Wovov_occi_, dressed_.Wovov_pno_occi);
                    pack_W(&d_Wovov_occj_, dressed_.Wovov_pno_occj);
                }
                const size_t rpsz = static_cast<size_t>(nocc_) * max_n_pno_;
                if (rpsz > 0) {
                    tracked_cudaMalloc(&d_RP_oim_, rpsz * sizeof(real_t));
                    tracked_cudaMalloc(&d_RP_moi_, rpsz * sizeof(real_t));
                    tracked_cudaMalloc(&d_RP_moj_, rpsz * sizeof(real_t));
                }
            }
            // Stage 4 σ1: device accumulator + per-term intermediates.
            const bool any_s1 = use_gpu_s1loo_ || use_gpu_s1fov_ || use_gpu_s1wooov_;
            if (any_s1)
                tracked_cudaMalloc(&d_sigma1_, static_cast<size_t>(nocc_) * sizeof(real_t));
            // S1LOO needs canonical Loo + r1 (r1 shared with T1; alloc if not already).
            if (use_gpu_s1loo_) {
                const size_t no2 = static_cast<size_t>(nocc_) * nocc_;
                tracked_cudaMalloc(&d_Loo_canon_, no2 * sizeof(real_t));
                cudaMemcpy(d_Loo_canon_, h_Loo_.data(), no2 * sizeof(real_t), cudaMemcpyHostToDevice);
                if (!d_r1_) tracked_cudaMalloc(&d_r1_, static_cast<size_t>(nocc_) * sizeof(real_t));
            }
            if (use_gpu_s1fov_) {
                const size_t fsz = static_cast<size_t>(nocc_) * nvir_;
                tracked_cudaMalloc(&d_Fov_, fsz * sizeof(real_t));
                cudaMemcpy(d_Fov_, h_Fov_.data(), fsz * sizeof(real_t), cudaMemcpyHostToDevice);
            }
            if (use_gpu_s1wooov_) {
                // Pre-transpose Wooov[k,l,i,d] → Wooov_re[i,k,l,d] (i outermost) so the
                // contraction Σ_{kld} Wooov[k,l,i,d] Ssym[k,l,d] is a single GEMV per i.
                const int no = nocc_, nv = nvir_;
                const size_t wsz = static_cast<size_t>(no) * no * no * nv;
                std::vector<real_t> h_re(wsz, 0.0);
                #pragma omp parallel for
                for (int i = 0; i < no; ++i)
                    for (int k = 0; k < no; ++k)
                        for (int l = 0; l < no; ++l)
                            for (int d = 0; d < nv; ++d)
                                h_re[(((static_cast<size_t>(i) * no + k) * no + l) * nv) + d] =
                                    h_Wooov_[(((static_cast<size_t>(k) * no + l) * no + i) * nv) + d];
                tracked_cudaMalloc(&d_Wooov_re_, wsz * sizeof(real_t));
                cudaMemcpy(d_Wooov_re_, h_re.data(), wsz * sizeof(real_t), cudaMemcpyHostToDevice);
                tracked_cudaMalloc(&d_Ssym1_,
                                   static_cast<size_t>(no) * no * nv * sizeof(real_t));
            }
        }
        std::cout << "[bt-PNO B-a.6a] DLPNOIPEOMNativeOperator: GPU σ2 "
                  << (use_gpu_xpair_
                        ? "lift + xpair(T3/T4/T5) + projection + T2 path ON (cross-pair "
                          "Loo·r2c + Woooo·r2c strided GEMVs + lift + projection; "
                          "GANSU_DLPNO_NATIVE_GPU_XPAIR=1"
                        : use_gpu_lift_
                        ? "lift + projection + T2 path ON (per-pair source lift U·r2s + "
                          "per-orientation U^(ij)ᵀ·acc + Lvv^(ij) T2 cublasDgemv; "
                          "GANSU_DLPNO_NATIVE_GPU_LIFT=1"
                        : use_gpu_proj_
                        ? "projection + T2 path ON (per-orientation U^(ij)ᵀ·acc cublasDgemv "
                          "+ Lvv^(ij) T2; GANSU_DLPNO_NATIVE_GPU_PROJ=1"
                        : "T2 path ON (per-orientation Lvv^(ij) cublasDgemv; "
                          "GANSU_DLPNO_NATIVE_GPU=1")
                  << (gpu_selfcheck_ ? ", in-process GPU-vs-host self-check" : "")
                  << ")" << std::endl;
        if (use_gpu_s1loo_)
            std::cout << "[bt-PNO B-a.6a] DLPNOIPEOMNativeOperator: GPU σ1 path ON ("
                      << "Loo·r1" << (use_gpu_s1fov_ ? " + Fov·r2" : "")
                      << (use_gpu_s1wooov_ ? " + Wooov·r2" : "")
                      << "; reuses device-resident lifted r2; "
                         "GANSU_DLPNO_NATIVE_GPU_S1LOO/S1FOV/S1WOOOV=1)" << std::endl;
        if (use_gpu_resident_)
            std::cout << "[bt-PNO B-a.6a] DLPNOIPEOMNativeOperator: GPU full-residency "
                         "path ON (r1/r2 read straight from d_input, σ assembled on "
                         "device, no host round-trip; GANSU_DLPNO_NATIVE_GPU_RESIDENT=1)"
                      << std::endl;
        // Stage 5c: full per-device σ2 workspace. Each d>0 gets a complete replica of
        // every σ2 device buffer (constants peer-copied from device 0, scratch freshly
        // allocated) + a cublas handle, so the validated σ2 helper chain runs unchanged
        // on device d after bind_device(d). ws_[0] aliases the device-0 members so
        // bind_device(0) restores. The orientation-slot slab partition (weight n_pno²)
        // assigns each device its gather range. Production (num_gpus=1) byte-unchanged.
        if (use_gpu_multi_) {
            auto& mgr = MultiGpuManager::instance();
            mgr.initialize(num_gpus_);
            const int nuse = std::min(num_gpus_, mgr.num_devices());
            if (nuse < 2) {
                use_gpu_multi_ = false; multi_selfcheck_ = false; use_gpu_multi_slab_ = false;
            } else {
                ws_.resize(nuse);
                const size_t r2c_len  = static_cast<size_t>(nocc_) * nocc_ * nvir_;
                const size_t acc_len  = static_cast<size_t>(n_orient_) * nvir_;
                const size_t plen     = static_cast<size_t>(total_dim_ - nocc_);
                const size_t s_len    = static_cast<size_t>(nocc_) * nocc_ * nvir_;
                const size_t no2      = static_cast<size_t>(nocc_) * nocc_;
                const size_t no4      = no2 * no2;
                const size_t wovoo_len= no2 * nvir_ * nocc_;
                const size_t oovv_len = no2 * static_cast<size_t>(nvir_) * nvir_;
                const size_t rpsz     = static_cast<size_t>(nocc_) * max_n_pno_;
                const size_t utot     = u_pno_off_.empty()   ? 0 : u_pno_off_.back();
                const size_t lvv_tot  = lvv_pno_off_.empty() ? 0 : lvv_pno_off_.back();
                const size_t wtot     = wovvo_off_.empty()   ? 0 : wovvo_off_.back();
                // Orientation-slot slab partition (contiguous, weight n_pno²) — computed
                // before the alloc loop so the 5e slab-only packs can size their range.
                std::vector<double> wcum(n_orient_ + 1, 0.0);
                for (int s = 0; s < n_orient_; ++s) {
                    const double n = packing_.n_pno[orient_idx_[s]];
                    wcum[s + 1] = wcum[s] + n * n;
                }
                const double total = wcum[n_orient_];
                slot_begin_.assign(nuse, 0); slot_end_.assign(nuse, 0);
                { int sb = 0;
                  for (int d = 0; d < nuse; ++d) {
                      slot_begin_[d] = sb;
                      if (d == nuse - 1) sb = n_orient_;
                      else { const double tgt = total * (d + 1) / nuse;
                             while (sb < n_orient_ && wcum[sb + 1] < tgt) ++sb; }
                      slot_end_[d] = sb;
                  } }
                // ws_[0] aliases the device-0 members (exact restore for bind_device(0)).
                ws_[0].device = 0;            ws_[0].cublas = cublas_;
                ws_[0].d_input = nullptr;     ws_[0].d_r2c_all = d_r2c_all_;
                ws_[0].d_acc_all = d_acc_all_;  ws_[0].d_sig_pack = d_sig_pack_;
                ws_[0].d_U_pack = d_U_pack_;    ws_[0].d_Lvv_pno_pack = d_Lvv_pno_pack_;
                ws_[0].d_Loo_lmo = d_Loo_lmo_;  ws_[0].d_Woooo_lmo = d_Woooo_lmo_;
                ws_[0].d_Wovoo_re = d_Wovoo_re_; ws_[0].d_Woovv_lmo = d_Woovv_lmo_;
                ws_[0].d_t2_lmo = d_t2_lmo_;    ws_[0].d_S = d_S_;  ws_[0].d_tmp_c = d_tmp_c_;
                ws_[0].d_Wovvo_occi = d_Wovvo_occi_; ws_[0].d_Wovvo_occj = d_Wovvo_occj_;
                ws_[0].d_Wovov_occi = d_Wovov_occi_; ws_[0].d_Wovov_occj = d_Wovov_occj_;
                ws_[0].d_RP_oim = d_RP_oim_;    ws_[0].d_RP_moi = d_RP_moi_; ws_[0].d_RP_moj = d_RP_moj_;
                ws_[0].lvv_shift = 0; ws_[0].wovvo_shift = 0;   // device 0 holds the full packs
                // d>0 replicas (constants peer-copied; scratch allocated only).
                for (int d = 1; d < nuse; ++d) {
                    MultiGpuManager::DeviceGuard guard(d);
                    DeviceWorkspace& w = ws_[d];
                    w.device = d;
                    w.cublas = mgr.cublas(d);
                    cublasSetStream(mgr.cublas(d), 0);  // NULL stream → match device-0 ordering
                    auto scr = [&](real_t** p, size_t n) {
                        if (n) tracked_cudaMalloc(p, n * sizeof(real_t)); };
                    auto cpy = [&](real_t** p, const real_t* src, size_t n) {
                        if (n && src) { tracked_cudaMalloc(p, n * sizeof(real_t));
                                        cudaMemcpyPeer(*p, d, src, 0, n * sizeof(real_t)); } };
                    // 5e: copy only a [lo,hi) subrange of an output-indexed pack.
                    auto cpy_slab = [&](real_t** p, const real_t* src, size_t lo, size_t hi) {
                        if (hi > lo && src) { tracked_cudaMalloc(p, (hi - lo) * sizeof(real_t));
                                              cudaMemcpyPeer(*p, d, src + lo, 0, (hi - lo) * sizeof(real_t)); } };
                    scr(&w.d_input, static_cast<size_t>(total_dim_));
                    scr(&w.d_r2c_all, r2c_len);  scr(&w.d_acc_all, acc_len);
                    scr(&w.d_sig_pack, plen);    scr(&w.d_S, s_len);
                    scr(&w.d_tmp_c, static_cast<size_t>(nvir_));
                    scr(&w.d_RP_oim, rpsz); scr(&w.d_RP_moi, rpsz); scr(&w.d_RP_moj, rpsz);
                    cpy(&w.d_U_pack, d_U_pack_, utot);
                    cpy(&w.d_Loo_lmo, d_Loo_lmo_, no2);
                    cpy(&w.d_Woooo_lmo, d_Woooo_lmo_, no4);
                    cpy(&w.d_Wovoo_re, d_Wovoo_re_, wovoo_len);
                    cpy(&w.d_Woovv_lmo, d_Woovv_lmo_, oovv_len);
                    cpy(&w.d_t2_lmo, d_t2_lmo_, oovv_len);
                    // 5e: Lvv_pno + ph-ladder packs slab-only in slab mode. The slab's
                    // pairs form a contiguous idx range [orient_idx_[slot_begin],
                    // orient_idx_[slot_end-1]] (orient slots are in idx order), so the
                    // packs are contiguous subranges. else full replica.
                    if (use_gpu_multi_slab_ && slot_end_[d] > slot_begin_[d]) {
                        const int idx_lo = orient_idx_[slot_begin_[d]];
                        const int idx_hi = orient_idx_[slot_end_[d] - 1];
                        const size_t lvlo = lvv_pno_off_[idx_lo], lvhi = lvv_pno_off_[idx_hi + 1];
                        const size_t wlo  = wovvo_off_[idx_lo],   whi  = wovvo_off_[idx_hi + 1];
                        w.lvv_shift = lvlo;  w.wovvo_shift = wlo;
                        cpy_slab(&w.d_Lvv_pno_pack, d_Lvv_pno_pack_, lvlo, lvhi);
                        cpy_slab(&w.d_Wovvo_occi, d_Wovvo_occi_, wlo, whi);
                        cpy_slab(&w.d_Wovvo_occj, d_Wovvo_occj_, wlo, whi);
                        cpy_slab(&w.d_Wovov_occi, d_Wovov_occi_, wlo, whi);
                        cpy_slab(&w.d_Wovov_occj, d_Wovov_occj_, wlo, whi);
                    } else {
                        w.lvv_shift = 0; w.wovvo_shift = 0;
                        cpy(&w.d_Lvv_pno_pack, d_Lvv_pno_pack_, lvv_tot);
                        cpy(&w.d_Wovvo_occi, d_Wovvo_occi_, wtot);
                        cpy(&w.d_Wovvo_occj, d_Wovvo_occj_, wtot);
                        cpy(&w.d_Wovov_occi, d_Wovov_occi_, wtot);
                        cpy(&w.d_Wovov_occj, d_Wovov_occj_, wtot);
                    }
                }
            }
        }
        if (num_gpus_ > 1)
            std::cout << "[bt-PNO Stage 5] DLPNOIPEOMNativeOperator: num_gpus=" << num_gpus_
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
        use_gpu_t1_ = false; use_gpu_t8_ = false; use_gpu_phl_ = false;
        use_gpu_s1loo_ = false; use_gpu_s1fov_ = false; use_gpu_s1wooov_ = false;
        use_gpu_resident_ = false;
    }
#endif
}

DLPNOIPEOMNativeOperator::~DLPNOIPEOMNativeOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
#ifndef GANSU_CPU_ONLY
    if (d_Lvv_pno_pack_) tracked_cudaFree(d_Lvv_pno_pack_);
    if (d_r2_pack_)      tracked_cudaFree(d_r2_pack_);
    if (d_sig_pack_)     tracked_cudaFree(d_sig_pack_);
    if (d_U_pack_)       tracked_cudaFree(d_U_pack_);
    if (d_acc_all_)      tracked_cudaFree(d_acc_all_);
    if (d_r2c_all_)      tracked_cudaFree(d_r2c_all_);
    if (d_Loo_lmo_)      tracked_cudaFree(d_Loo_lmo_);
    if (d_Woooo_lmo_)    tracked_cudaFree(d_Woooo_lmo_);
    if (d_Wovoo_re_)     tracked_cudaFree(d_Wovoo_re_);
    if (d_r1_)           tracked_cudaFree(d_r1_);
    if (d_Woovv_lmo_)    tracked_cudaFree(d_Woovv_lmo_);
    if (d_t2_lmo_)       tracked_cudaFree(d_t2_lmo_);
    if (d_S_)            tracked_cudaFree(d_S_);
    if (d_tmp_c_)        tracked_cudaFree(d_tmp_c_);
    if (d_Wovvo_occi_)   tracked_cudaFree(d_Wovvo_occi_);
    if (d_Wovvo_occj_)   tracked_cudaFree(d_Wovvo_occj_);
    if (d_Wovov_occi_)   tracked_cudaFree(d_Wovov_occi_);
    if (d_Wovov_occj_)   tracked_cudaFree(d_Wovov_occj_);
    if (d_RP_oim_)       tracked_cudaFree(d_RP_oim_);
    if (d_RP_moi_)       tracked_cudaFree(d_RP_moi_);
    if (d_RP_moj_)       tracked_cudaFree(d_RP_moj_);
    if (d_Loo_canon_)    tracked_cudaFree(d_Loo_canon_);
    if (d_Fov_)          tracked_cudaFree(d_Fov_);
    if (d_Wooov_re_)     tracked_cudaFree(d_Wooov_re_);
    if (d_Ssym1_)        tracked_cudaFree(d_Ssym1_);
    if (d_sigma1_)       tracked_cudaFree(d_sigma1_);
    // Stage 5c: free the per-device σ2 replicas (d>0 only; ws_[0] aliases the device-0
    // members freed above, so skip it to avoid a double free).
    for (auto& w : ws_) {
        if (w.device <= 0) continue;
        MultiGpuManager::DeviceGuard guard(w.device);
        for (real_t* p : {w.d_input, w.d_r2c_all, w.d_acc_all, w.d_sig_pack, w.d_U_pack,
                          w.d_Lvv_pno_pack, w.d_Loo_lmo, w.d_Woooo_lmo, w.d_Wovoo_re,
                          w.d_Woovv_lmo, w.d_t2_lmo, w.d_S, w.d_tmp_c,
                          w.d_Wovvo_occi, w.d_Wovvo_occj, w.d_Wovov_occi, w.d_Wovov_occj,
                          w.d_RP_oim, w.d_RP_moi, w.d_RP_moj})
            if (p) tracked_cudaFree(p);
    }
    if (cublas_) cublasDestroy(reinterpret_cast<cublasHandle_t>(cublas_));
#endif
}

// σ1[i] (1h sector), canonical formula (mirror of ip_eom_sigma1_full_kernel) on
// the lifted canonical r2. r2 layout r2[(k*nocc+l)*nvir + d].
void DLPNOIPEOMNativeOperator::compute_sigma1(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& sigma1,
    bool skip_loo,
    bool skip_fov,
    bool skip_wooov) const
{
    const int nocc = nocc_;
    const int nvir = nvir_;
    const size_t vstride = static_cast<size_t>(nvir);

    sigma1.assign(static_cast<size_t>(nocc), 0.0);
    // B-a.6a Stage 4: when every σ1 term runs on device, skip the host lift too.
    if (skip_loo && skip_fov && skip_wooov) return;

    const std::vector<real_t> r2 = ip_packed_r2_to_canonical(
        res_, packing_, U_loc_, C_vir_, h_S_.data(), nao_, nvir_, packed_r2);

    #pragma omp parallel for
    for (int i = 0; i < nocc; ++i) {
        real_t s = 0.0;
        if (!skip_loo)
            for (int k = 0; k < nocc; ++k)
                s -= h_Loo_[static_cast<size_t>(k) * nocc + i] * r1[k];
        if (!skip_fov)
            for (int l = 0; l < nocc; ++l)
                for (int d = 0; d < nvir; ++d) {
                    const real_t fov_ld = h_Fov_[static_cast<size_t>(l) * nvir + d];
                    s += 2.0 * fov_ld * r2[(static_cast<size_t>(i) * nocc + l) * vstride + d];
                    s -=       fov_ld * r2[(static_cast<size_t>(l) * nocc + i) * vstride + d];
                }
        if (!skip_wooov)
            for (int k = 0; k < nocc; ++k)
                for (int l = 0; l < nocc; ++l)
                    for (int d = 0; d < nvir; ++d) {
                        const real_t w1 = h_Wooov_[(static_cast<size_t>(k) * nocc + l) * nocc * nvir
                                                   + static_cast<size_t>(i) * nvir + d];
                        const real_t w2 = h_Wooov_[(static_cast<size_t>(l) * nocc + k) * nocc * nvir
                                                   + static_cast<size_t>(i) * nvir + d];
                        const real_t r2_kld = r2[(static_cast<size_t>(k) * nocc + l) * vstride + d];
                        s += (-2.0 * w1 + w2) * r2_kld;
                    }
        sigma1[i] = s;
    }
}

// σ2 (2h1p), native per-pair. Terms accumulate into a canonical-virtual buffer
// `acc` that is finally projected to the target pair PNO by U^(ij)ᵀ:
//   acc =  Lvv·(U^(ij)·r2_orient)                                      (T2, B-a.2a)
//        + w_T1,   w[a] = -Σ_k r1[k] Wovoo_lmo[k,a,oi,oj]              (T1, B-a.2b)
//        - Σ_k Loo_lmo[k,oi]·(U_{src(k,oj)}·r2_{src(k,oj)})            (T3, B-a.3a)
//        - Σ_l Loo_lmo[l,oj]·(U_{src(oi,l)}·r2_{src(oi,l)})           (T4, B-a.3a)
//   σ2_packed^(orient)[a''] += (U^(ij)ᵀ acc)[a'']
// barS^(ij,kj) = U^(ij)ᵀ U^(kj) exactly (bar_Q = C_vir U, C_virᵀ S C_vir = I),
// so T3/T4 equal the project-down of canonical Loo·r2 on the lifted r2 — bit-
// exact even with PNO truncation. Source orientation mirrors the CCSD T2 sweep
// (dlpno_pair_data.cu:399-435): off_ij if setups[idx_s].i == first-occ else off_ji.
void DLPNOIPEOMNativeOperator::compute_sigma2(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2,
    bool skip_t2,
    std::vector<real_t>* acc_export,
    const std::vector<real_t>* r2c_external,
    bool skip_xpair,
    bool skip_t1,
    bool skip_t8,
    bool skip_phl) const
{
    packed_sigma2.assign(static_cast<size_t>(total_dim_ - nocc_), 0.0);

    const int nocc = nocc_;
    const int nvir = nvir_;
    const size_t nocc2 = static_cast<size_t>(nocc) * nocc;
    const int n_pairs = static_cast<int>(res_.pairs.size());
    int orient_slot = 0;   // B-a.6a Stage 2: running orientation index for acc_export

    Eigen::Map<const RowMatXd> Cv(C_vir_.data(), nao_, nvir);
    Eigen::Map<const RowMatXd> S(h_S_.data(), nao_, nao_);
    const RowMatXd CvtS = Cv.transpose() * S;          // [nvir × nao]
    Eigen::Map<const RowMatXd> Lvv(h_Lvv_.data(), nvir, nvir);  // Lvv[a,d]

    // Precompute U^(idx) = C_virᵀ S bar_Q_idx [nvir × n_pno] for every pair
    // (needed by both the target projection and the cross-pair source lift).
    std::vector<RowMatXd> Uall(n_pairs);
    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = packing_.n_pno[idx];
        if (n == 0) continue;
        Eigen::Map<const RowMatXd> barQ(res_.pairs[idx].bar_Q.data(), nao_, n);
        Uall[idx] = CvtS * barQ;
    }

    // Precompute the canonical-virtual lift of every source amplitude r2[p,q,:]
    // (orientation per storage; zero if the pair is absent/screened) once per
    // matvec, into r2c_all[p*nocc+q].
    std::vector<Eigen::VectorXd> r2c_all(static_cast<size_t>(nocc) * nocc,
                                         Eigen::VectorXd::Zero(nvir));
    for (int p = 0; p < nocc; ++p)
        for (int q = 0; q < nocc; ++q) {
            const size_t pq = static_cast<size_t>(p) * nocc + q;
            if (r2c_external) {
                // Stage 3a: take the device-lifted r2c[p,q] (canonical nvir) directly.
                r2c_all[pq] = Eigen::Map<const Eigen::VectorXd>(
                    r2c_external->data() + pq * nvir, nvir);
                continue;
            }
            const int idx_s = res_.pair_lookup[pq];
            if (idx_s < 0) continue;
            const int ns = packing_.n_pno[idx_s];
            if (ns == 0) continue;
            const int off = (res_.setups[idx_s].i == p) ? packing_.off_ij[idx_s]
                                                        : packing_.off_ji[idx_s];
            Eigen::Map<const Eigen::VectorXd> r2s(packed_r2.data() + (off - nocc_), ns);
            r2c_all[pq].noalias() = Uall[idx_s] * r2s;
        }
    auto r2c = [&](int p, int q) -> const Eigen::VectorXd& {
        return r2c_all[static_cast<size_t>(p) * nocc + q];
    };

    // T8a: tmp_c[c] = Σ_{k,l,d} (2 Woovv_lmo[l,k,d,c] - Woovv_lmo[k,l,d,c]) r2[k,l,d]
    // (one canonical-virtual [nvir] vector per matvec; basis-invariant).
    // Woovv layout ((a*nocc+b)*nvir+d')*nvir+c for Woovv[a,b,d',c].
    Eigen::VectorXd tmp_c = Eigen::VectorXd::Zero(nvir);
    if (!skip_t8)
    for (int k = 0; k < nocc; ++k)
        for (int l = 0; l < nocc; ++l) {
            const Eigen::VectorXd& r2kl = r2c(k, l);   // indexed by d
            const size_t base_lk = (static_cast<size_t>(l) * nocc + k) * nvir;  // [l,k,*,*]
            const size_t base_kl = (static_cast<size_t>(k) * nocc + l) * nvir;  // [k,l,*,*]
            for (int c = 0; c < nvir; ++c) {
                real_t s = 0.0;
                for (int d = 0; d < nvir; ++d)
                    s += (2.0 * h_Woovv_lmo_[(base_lk + d) * nvir + c]
                              - h_Woovv_lmo_[(base_kl + d) * nvir + c]) * r2kl(d);
                tmp_c(c) += s;
            }
        }

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = packing_.n_pno[idx];
        if (n == 0) continue;
        const int i = res_.setups[idx].i;
        const int j = res_.setups[idx].j;
        const RowMatXd& U_ij = Uall[idx];

        // Accumulate all σ2 terms for one orientation (oi,oj) into the canonical
        // -virtual buffer, then project to the pair PNO.  off = packed offset of
        // this orientation's σ2 block.
        auto accumulate_orientation = [&](int oi, int oj, int off) {
            // T2 (Lvv): dense-borrow path contracts the full Lvv on the lifted r2;
            // the B-a.6c dressed path instead applies Lvv^(ij) in PNO space at the
            // end (see below), so the canonical-virtual acc starts at zero here.
            Eigen::VectorXd acc = use_dressed_pno_
                                      ? Eigen::VectorXd(Eigen::VectorXd::Zero(nvir))
                                      : Eigen::VectorXd(Lvv * r2c(oi, oj));   // T2
            // T1: w[a] = -Σ_k r1[k] Wovoo_lmo[k,a,oi,oj]    (GPU on Stage 3b)
            if (!skip_t1)
            for (int a = 0; a < nvir; ++a) {
                real_t s = 0.0;
                for (int k = 0; k < nocc; ++k)
                    s -= r1[k] * h_Wovoo_lmo_[(static_cast<size_t>(k) * nvir + a) * nocc2
                                              + static_cast<size_t>(oi) * nocc + oj];
                acc(a) += s;
            }
            // T3: -Σ_k Loo_lmo[k,oi] r2[k,oj,:]    (GPU on Stage 3b)
            if (!skip_xpair)
            for (int k = 0; k < nocc; ++k) {
                const real_t c = h_Loo_lmo_[static_cast<size_t>(k) * nocc + oi];
                if (c != 0.0) acc.noalias() += (-c) * r2c(k, oj);
            }
            // T4: -Σ_l Loo_lmo[l,oj] r2[oi,l,:]    (GPU on Stage 3b)
            if (!skip_xpair)
            for (int l = 0; l < nocc; ++l) {
                const real_t c = h_Loo_lmo_[static_cast<size_t>(l) * nocc + oj];
                if (c != 0.0) acc.noalias() += (-c) * r2c(oi, l);
            }
            // T5: +Σ_{k,l} Woooo_lmo[k,l,oi,oj] r2[k,l,:]    (GPU on Stage 3b)
            if (!skip_xpair)
            for (int k = 0; k < nocc; ++k)
                for (int l = 0; l < nocc; ++l) {
                    const real_t w = h_Woooo_lmo_[((static_cast<size_t>(k) * nocc + l) * nocc
                                                   + oi) * nocc + oj];
                    if (w != 0.0) acc.noalias() += w * r2c(k, l);
                }
            // T8b: acc[a] -= Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]    (GPU on Stage 3b)
            //   t2 layout ((i*nocc+j)*nvir+c)*nvir+a.
            if (!skip_t8) {
                const size_t base = (static_cast<size_t>(oi) * nocc + oj) * nvir;  // [oi,oj,*,*]
                for (int a = 0; a < nvir; ++a) {
                    real_t s = 0.0;
                    for (int c = 0; c < nvir; ++c)
                        s += tmp_c(c) * h_t2_lmo_[(base + c) * nvir + a];
                    acc(a) -= s;
                }
            }
            // T6 (ph-ladder): acc[a] += Σ_{m,d} Wovvo_lmo[m,a,d,oj] (2 r2c(oi,m)[d] - r2c(m,oi)[d])
            //   Wovvo layout ((m*nvir+a)*nvir+d)*nocc+oj.
            // T7 (ph-ladder): acc[a] += -Σ_{m,d} Wovov_lmo[m,a,oj,d] r2c(oi,m)[d]
            //                          -Σ_{m,d} Wovov_lmo[m,a,oi,d] r2c(m,oj)[d]
            //   Wovov layout ((m*nvir+a)*nocc+J)*nvir+d.
            // Dense-borrow path only; the dressed path does T6/T7 in PNO space below.
            if (!use_dressed_pno_)
            for (int m = 0; m < nocc; ++m) {
                const Eigen::VectorXd& r_oim = r2c(oi, m);
                const Eigen::VectorXd& r_moi = r2c(m, oi);
                const Eigen::VectorXd& r_moj = r2c(m, oj);
                for (int a = 0; a < nvir; ++a) {
                    real_t s = 0.0;
                    const size_t wvvo_b = (static_cast<size_t>(m) * nvir + a) * nvir;       // [m,a,*]·nocc+oj
                    const size_t wov_oj_b = ((static_cast<size_t>(m) * nvir + a) * nocc + oj) * nvir;  // [m,a,oj,*]
                    const size_t wov_oi_b = ((static_cast<size_t>(m) * nvir + a) * nocc + oi) * nvir;  // [m,a,oi,*]
                    for (int d = 0; d < nvir; ++d) {
                        s += h_Wovvo_lmo_[(wvvo_b + d) * nocc + oj]
                             * (2.0 * r_oim(d) - r_moi(d));                  // T6
                        s -= h_Wovov_lmo_[wov_oj_b + d] * r_oim(d);          // T7 first
                        s -= h_Wovov_lmo_[wov_oi_b + d] * r_moj(d);          // T7 second
                    }
                    acc(a) += s;
                }
            }
            // B-a.6a Stage 2: when exporting, store the canonical-virtual acc for
            // this orientation and let the device do U^(ij)ᵀ·acc + T2; the host s
            // starts at zero and accumulates only the PNO-space ph-ladder T6/T7.
            Eigen::VectorXd s;
            if (acc_export) {
                std::memcpy(acc_export->data() + static_cast<size_t>(orient_slot) * nvir,
                            acc.data(), static_cast<size_t>(nvir) * sizeof(real_t));
                s = Eigen::VectorXd::Zero(n);
            } else {
                s = U_ij.transpose() * acc;   // [n_pno]
            }
            ++orient_slot;
            if (use_dressed_pno_) {
                // T2 in PNO space: σ2_packed^(orient) += Lvv^(ij) · r2_packed^(orient).
                // Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij); since r2c(own orientation) =
                // U^(ij)·r2_packed exactly, this equals the dense-borrow T2
                // (U^(ij)ᵀ·Lvv·U^(ij)·r2_packed) bit-for-bit — but never builds
                // the canonical-virtual nvir vector. Both orientations of a pair
                // share dressed_.Lvv_pno[idx] (Lvv is occ-free).
                // T2: skipped on host when the GPU path computes it (apply_t2_gpu).
                if (!skip_t2) {
                    Eigen::Map<const RowMatXd> Lpno(dressed_.Lvv_pno[idx].data(), n, n);
                    Eigen::Map<const Eigen::VectorXd> r2p(packed_r2.data() + (off - nocc_), n);
                    s.noalias() += Lpno * r2p;
                }

                // T6/T7 (ph-ladder) in PNO space. The W lives in PNO(ij) (the
                // U^(ij)ᵀ projection is folded into the build), and the source
                // amplitudes enter via the ONE-SIDED barS projection
                //   r2c_pno(p,q) = U^(ij)ᵀ r2c(p,q)   [n]  (= barS^(ij,src)·r2_src).
                // Role select: T6 + T7-first use the fixed occupied = oj,
                // T7-second uses occupied = oi (i,j = res_.setups[idx].(i,j)).
                if (!skip_phl) {
                const std::vector<real_t>& Wvvo_oj = (oj == j) ? dressed_.Wovvo_pno_occj[idx]
                                                               : dressed_.Wovvo_pno_occi[idx];
                const std::vector<real_t>& Wov_oj  = (oj == j) ? dressed_.Wovov_pno_occj[idx]
                                                               : dressed_.Wovov_pno_occi[idx];
                const std::vector<real_t>& Wov_oi  = (oi == j) ? dressed_.Wovov_pno_occj[idx]
                                                               : dressed_.Wovov_pno_occi[idx];
                for (int m = 0; m < nocc; ++m) {
                    const Eigen::VectorXd rp_oim = U_ij.transpose() * r2c(oi, m);  // [n]
                    const Eigen::VectorXd rp_moi = U_ij.transpose() * r2c(m, oi);
                    const Eigen::VectorXd rp_moj = U_ij.transpose() * r2c(m, oj);
                    const size_t base = static_cast<size_t>(m) * n * n;
                    for (int ap = 0; ap < n; ++ap) {
                        real_t acc_p = 0.0;
                        const size_t row = base + static_cast<size_t>(ap) * n;
                        for (int dp = 0; dp < n; ++dp) {
                            acc_p += Wvvo_oj[row + dp] * (2.0 * rp_oim(dp) - rp_moi(dp));  // T6
                            acc_p -= Wov_oj[row + dp] * rp_oim(dp);                        // T7 first
                            acc_p -= Wov_oi[row + dp] * rp_moj(dp);                        // T7 second
                        }
                        s(ap) += acc_p;
                    }
                }
                }  // if (!skip_phl)
            }  // if (use_dressed_pno_)
            real_t* dst = packed_sigma2.data() + (off - nocc_);
            for (int a = 0; a < n; ++a) dst[a] += s(a);
        };

        accumulate_orientation(i, j, packing_.off_ij[idx]);
        if (!packing_.diagonal(idx))
            accumulate_orientation(j, i, packing_.off_ji[idx]);
    }
}

// B-a.6a Stage 1: GPU σ2 T2. For each pair and orientation, packed_sigma2^(orient)
// += Lvv^(ij)·r2_packed^(orient) via cublasDgemv. Lvv_pno is row-major [n×n];
// cuBLAS reads it column-major (= its transpose), so op_T yields y = Lvv^(ij)·r2p
// — exactly the host PNO-space T2 block. Both orientations share Lvv_pno[idx].
void DLPNOIPEOMNativeOperator::apply_t2_gpu(
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nocc_);
    if (plen == 0) return;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemset(d_sig_pack_, 0, plen * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    const int n_pairs = static_cast<int>(res_.pairs.size());
    auto do_orient = [&](int idx, int off) {
        const int n = packing_.n_pno[idx];
        if (n == 0) return;
        const size_t o = static_cast<size_t>(off) - nocc_;   // packed r2/σ2 block start
        cublasDgemv(cublas, CUBLAS_OP_T, n, n, &one,
                    d_Lvv_pno_pack_ + lvv_pno_off_[idx], n,
                    d_r2_pack_ + o, 1, &zero, d_sig_pack_ + o, 1);
    };
    for (int idx = 0; idx < n_pairs; ++idx) {
        if (packing_.n_pno[idx] == 0) continue;
        do_orient(idx, packing_.off_ij[idx]);
        if (!packing_.diagonal(idx)) do_orient(idx, packing_.off_ji[idx]);
    }
    std::vector<real_t> add(plen, 0.0);
    cudaMemcpy(add.data(), d_sig_pack_, plen * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < plen; ++k) packed_sigma2[k] += add[k];
#else
    (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 2: GPU single-sided PNO projection + Stage-1 T2. For each orientation
// (slot s in the same iteration order as compute_sigma2), σ2_packed^(orient)[a'] =
// Σ_a U^(ij)[a,a'] acc_slot[a]  (= U^(ij)ᵀ·acc). U^(ij) is row-major [nvir × n]; its
// column-major view (ld=n) is exactly U^(ij)ᵀ [n × nvir], so OP_N yields U^(ij)ᵀ·acc
// — the host projection bit-for-bit. Then the validated T2 (Lvv^(ij)·r2_packed) is
// accumulated (β=1). The host already wrote the PNO-space ph-ladder T6/T7 into
// packed_sigma2; this routine adds proj + T2 on top.
void DLPNOIPEOMNativeOperator::apply_projection_gpu(
    const std::vector<real_t>& r1,
    const std::vector<real_t>& acc_all,
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& packed_sigma2) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nocc_);
    if (plen == 0) return;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_)
        cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r2src = resident_ ? d_r2_src_ : d_r2_pack_;   // packed r2 device source
    // Resident: host builds no acc → start the device acc at zero; else upload host acc.
    if (resident_)
        cudaMemset(d_acc_all_, 0, static_cast<size_t>(n_orient_) * nvir_ * sizeof(real_t));
    else
        cudaMemcpy(d_acc_all_, acc_all.data(),
                   static_cast<size_t>(n_orient_) * nvir_ * sizeof(real_t), cudaMemcpyHostToDevice);
    // Stage 3b: add the cross-pair T3/T4/T5 (+ T1/T8) onto the device acc stack.
    if (use_gpu_xpair_) add_xpair_gpu();
    if (use_gpu_t1_) add_t1_gpu(r1);
    if (use_gpu_t8_) add_t8_gpu();
    cudaMemset(d_sig_pack_, 0, plen * sizeof(real_t));
    const real_t one = 1.0, zero = 0.0;
    // Direct per-slot loop (orient_idx_/orient_off_ match the acc slot order). Stage
    // 5c-step2: restrict to the active device's slab when slab_active_ (else all slots
    // → single-GPU / step1 byte-unchanged).
    const int s_lo = slab_active_ ? cur_slot_begin_ : 0;
    const int s_hi = slab_active_ ? cur_slot_end_   : n_orient_;
    for (int s = s_lo; s < s_hi; ++s) {
        const int idx = orient_idx_[s];
        const int n = packing_.n_pno[idx];
        const size_t o = orient_off_[s];   // packed r2/σ2 block start
        // proj: s[n] = U^(ij)ᵀ·acc_slot  (U row-major [nvir×n] → col-major [n×nvir]=U^ᵀ, OP_N).
        cublasDgemv(cublas, CUBLAS_OP_N, n, nvir_, &one,
                    d_U_pack_ + u_pno_off_[idx], n,
                    d_acc_all_ + static_cast<size_t>(s) * nvir_, 1, &zero,
                    d_sig_pack_ + o, 1);
        // T2 (validated Stage 1): s += Lvv^(ij)·r2_packed (β=1). 5e: subtract
        // lvv_pack_shift_ for the slab-only per-device pack (0 = full/single-GPU).
        cublasDgemv(cublas, CUBLAS_OP_T, n, n, &one,
                    d_Lvv_pno_pack_ + (lvv_pno_off_[idx] - lvv_pack_shift_), n,
                    r2src + o, 1, &one, d_sig_pack_ + o, 1);
    }
    // Stage 3b T6/T7: add the PNO-space ph-ladder onto the projected σ2 (d_sig_pack_).
    if (use_gpu_phl_) add_phl_gpu();
    if (resident_) return;   // d_sig_pack_ holds the full σ2; caller copies it D2D
    std::vector<real_t> add(plen, 0.0);
    cudaMemcpy(add.data(), d_sig_pack_, plen * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (size_t k = 0; k < plen; ++k) packed_sigma2[k] += add[k];
#else
    (void)acc_all; (void)packed_r2; (void)packed_sigma2;
#endif
}

// B-a.6a Stage 3a: GPU source lift. For each occupied pair (p,q) with a stored
// source pair idx_s (orientation off = off_ij if setups.i==p else off_ji), the
// lifted canonical amplitude r2c[p,q] = U_{idx_s}·r2_packed_src. U_{idx_s} is
// row-major [nvir × ns]; its column-major view (ld=ns) is its transpose, so OP_T
// yields y = U_{idx_s}·r2s — the host Uall[idx_s]·r2s bit-for-bit. Result left
// device-resident in d_r2c_all_ (block (p,q) at (p·nocc+q)·nvir) for the later
// cross-pair sub-stages and copied to r2c_all_host for the (transitional) host build.
void DLPNOIPEOMNativeOperator::lift_r2c_gpu(
    const std::vector<real_t>& packed_r2,
    std::vector<real_t>& r2c_all_host) const
{
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nocc_);
    const size_t r2c_len = static_cast<size_t>(nocc_) * nocc_ * nvir_;
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_ && plen > 0)
        cudaMemcpy(d_r2_pack_, packed_r2.data(), plen * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r2src = resident_ ? d_r2_src_ : d_r2_pack_;   // packed r2 device source
    cudaMemset(d_r2c_all_, 0, r2c_len * sizeof(real_t));   // absent/screened (p,q) stay zero
    const real_t one = 1.0, zero = 0.0;
    for (int p = 0; p < nocc_; ++p)
        for (int q = 0; q < nocc_; ++q) {
            const int idx_s = res_.pair_lookup[static_cast<size_t>(p) * nocc_ + q];
            if (idx_s < 0) continue;
            const int ns = packing_.n_pno[idx_s];
            if (ns == 0) continue;
            const int off = (res_.setups[idx_s].i == p) ? packing_.off_ij[idx_s]
                                                        : packing_.off_ji[idx_s];
            const size_t o = static_cast<size_t>(off) - nocc_;   // packed r2s block start
            // y[nvir] = U_{idx_s}·r2s; U row-major [nvir×ns] → col-major [ns×nvir], OP_T.
            cublasDgemv(cublas, CUBLAS_OP_T, ns, nvir_, &one,
                        d_U_pack_ + u_pno_off_[idx_s], ns,
                        r2src + o, 1, &zero,
                        d_r2c_all_ + (static_cast<size_t>(p) * nocc_ + q) * nvir_, 1);
        }
    if (!resident_)
        cudaMemcpy(r2c_all_host.data(), d_r2c_all_, r2c_len * sizeof(real_t), cudaMemcpyDeviceToHost);
#else
    (void)packed_r2; (void)r2c_all_host;
#endif
}

// Stage 5b: broadcast d_input to every d>0, redundantly lift each device's own
// d_r2c_all_ (identical loop to lift_r2c_gpu but on device d's ws_ replicas), and
// verify it matches device 0's lifted r2c. The device-0 matvec output is untouched;
// this proves the broadcast (decision B) + per-device full-r2c residency (decision
// A) before the 5c slab σ2 split + peer gather start consuming the per-device r2c.
// Gated on multi_selfcheck_ — without it the ws_ allocation alone proves the memory
// footprint and the per-device lift is skipped (no production overhead).
void DLPNOIPEOMNativeOperator::lift_r2c_multi_validate(const real_t* d_input) const {
#ifndef GANSU_CPU_ONLY
    if (!multi_selfcheck_ || ws_.empty()) return;
    auto& mgr = MultiGpuManager::instance();
    const size_t r2c_len = static_cast<size_t>(nocc_) * nocc_ * nvir_;
    const real_t one = 1.0, zero = 0.0;

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
        const real_t* r2src = w.d_input + nocc_;            // packed r2 on device d
        cudaMemset(w.d_r2c_all, 0, r2c_len * sizeof(real_t));
        cudaDeviceSynchronize();                            // memset done before any GEMV writes
        for (int p = 0; p < nocc_; ++p)
            for (int q = 0; q < nocc_; ++q) {
                const int idx_s = res_.pair_lookup[static_cast<size_t>(p) * nocc_ + q];
                if (idx_s < 0) continue;
                const int ns = packing_.n_pno[idx_s];
                if (ns == 0) continue;
                const int off = (res_.setups[idx_s].i == p) ? packing_.off_ij[idx_s]
                                                            : packing_.off_ji[idx_s];
                const size_t o = static_cast<size_t>(off) - nocc_;
                cublasDgemv(cublas, CUBLAS_OP_T, ns, nvir_, &one,
                            w.d_U_pack + u_pno_off_[idx_s], ns,
                            r2src + o, 1, &zero,
                            w.d_r2c_all + (static_cast<size_t>(p) * nocc_ + q) * nvir_, 1);
            }
        cudaDeviceSynchronize();                            // GEMVs done before D2H
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

// B-a.6a Stage 3b: add the cross-pair T3/T4/T5 to the device acc stack d_acc_all_,
// per orientation slot s = (orient_oi_[s], orient_oj_[s]). All three are OP_N GEMVs
// over a column-major view of the device-resident lifted r2c (d_r2c_all_), with the
// occ weight taken as a strided column of Loo_lmo / Woooo_lmo. Verified by index trace:
//   T3: acc[s][a] -= Σ_k r2c(k,oj)[a] Loo_lmo[k,oi]   (r2c(k,oj): base oj·nvir, ld nocc·nvir; Loo col oi, inc nocc)
//   T4: acc[s][a] -= Σ_l r2c(oi,l)[a] Loo_lmo[l,oj]   (r2c(oi,l): base oi·nocc·nvir, ld nvir; Loo col oj, inc nocc)
//   T5: acc[s][a] += Σ_kl r2c(k,l)[a] Woooo_lmo[k,l,oi,oj]  (full nocc² stack, ld nvir; Woooo col (oi·nocc+oj), inc nocc²)
void DLPNOIPEOMNativeOperator::add_xpair_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0, neg_one = -1.0;
    const int nv = nvir_, no = nocc_;
    const int no2 = no * no;
    const int s_lo = slab_active_ ? cur_slot_begin_ : 0;        // Stage 5c-step2 slab
    const int s_hi = slab_active_ ? cur_slot_end_   : n_orient_;
    for (int s = s_lo; s < s_hi; ++s) {
        const int oi = orient_oi_[s], oj = orient_oj_[s];
        real_t* acc = d_acc_all_ + static_cast<size_t>(s) * nv;
        // T3: acc -= Σ_k r2c(k,oj) Loo_lmo[k,oi].
        cublasDgemv(cublas, CUBLAS_OP_N, nv, no, &neg_one,
                    d_r2c_all_ + static_cast<size_t>(oj) * nv, no * nv,
                    d_Loo_lmo_ + oi, no, &one, acc, 1);
        // T4: acc -= Σ_l r2c(oi,l) Loo_lmo[l,oj].
        cublasDgemv(cublas, CUBLAS_OP_N, nv, no, &neg_one,
                    d_r2c_all_ + static_cast<size_t>(oi) * no * nv, nv,
                    d_Loo_lmo_ + oj, no, &one, acc, 1);
        // T5: acc += Σ_kl r2c(k,l) Woooo_lmo[k,l,oi,oj].
        cublasDgemv(cublas, CUBLAS_OP_N, nv, no2, &one,
                    d_r2c_all_, nv,
                    d_Woooo_lmo_ + (static_cast<size_t>(oi) * no + oj), no2, &one, acc, 1);
    }
#endif
}

// B-a.6a Stage 3b T1: add -Σ_k Wovoo_lmo[k,a,oi,oj] r1[k] to the device acc, per slot.
// Wovoo_re block (oi,oj) is row-major [nvir×nocc] (M[a,k]=Wovoo_lmo[k,a,oi,oj]); its
// column-major view (ld=nocc) is [nocc×nvir], so OP_T gives Σ_k M[a,k] r1[k].
void DLPNOIPEOMNativeOperator::add_t1_gpu(const std::vector<real_t>& r1) const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    if (!resident_)
        cudaMemcpy(d_r1_, r1.data(), static_cast<size_t>(nocc_) * sizeof(real_t), cudaMemcpyHostToDevice);
    const real_t* r1src = resident_ ? d_r1_src_ : d_r1_;   // r1 device source
    const real_t neg_one = -1.0, one = 1.0;
    const int nv = nvir_, no = nocc_;
    const int s_lo = slab_active_ ? cur_slot_begin_ : 0;        // Stage 5c-step2 slab
    const int s_hi = slab_active_ ? cur_slot_end_   : n_orient_;
    for (int s = s_lo; s < s_hi; ++s) {
        const int oi = orient_oi_[s], oj = orient_oj_[s];
        cublasDgemv(cublas, CUBLAS_OP_T, no, nv, &neg_one,
                    d_Wovoo_re_ + (static_cast<size_t>(oi) * no + oj) * nv * no, no,
                    r1src, 1, &one, d_acc_all_ + static_cast<size_t>(s) * nv, 1);
    }
#endif
}

// B-a.6a Stage 3b T8: tmp_c[c] = Σ_{kld} Woovv_lmo[k,l,d,c] S[k,l,d] (kernel builds
// S = 2r2c(l,k)-r2c(k,l), then a GEMV over (k,l,d)), then per slot acc[a] -=
// Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]. Both GEMVs OP_N over the contiguous row-major arrays.
void DLPNOIPEOMNativeOperator::add_t8_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0, zero = 0.0, neg_one = -1.0;
    const int nv = nvir_, no = nocc_;
    // T8a stage 1: S kernel.
    const int M = no * no * nv;
    const int threads = 256, blocks = (M + threads - 1) / threads;
    dlpno_ip_native_r2c_sym_kernel<<<blocks, threads>>>(d_r2c_all_, d_S_, no, nv);
    // T8a stage 2: tmp_c[c] = Σ_m Woovv[m,c] S[m]  (Woovv row-major [M×nvir] → col-major [nvir×M]).
    cublasDgemv(cublas, CUBLAS_OP_N, nv, M, &one, d_Woovv_lmo_, nv, d_S_, 1, &zero, d_tmp_c_, 1);
    // T8b: per slot acc -= Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]  (t2 block [c,a] row-major, OP_N → a).
    // (T8a tmp_c above is a global reduction → stays full on every device.)
    const int s_lo = slab_active_ ? cur_slot_begin_ : 0;        // Stage 5c-step2 slab
    const int s_hi = slab_active_ ? cur_slot_end_   : n_orient_;
    for (int s = s_lo; s < s_hi; ++s) {
        const int oi = orient_oi_[s], oj = orient_oj_[s];
        cublasDgemv(cublas, CUBLAS_OP_N, nv, nv, &neg_one,
                    d_t2_lmo_ + (static_cast<size_t>(oi) * no + oj) * nv * nv, nv,
                    d_tmp_c_, 1, &one, d_acc_all_ + static_cast<size_t>(s) * nv, 1);
    }
#endif
}

// B-a.6a Stage 3b T6/T7 ph-ladder: add the PNO-space ph-ladder DIRECTLY to d_sig_pack_
// (after the projection writes proj+T2). Per orientation slot (oi,oj) with (i,j)=
// setups[idx] and PNO dim n: the one-sided barS projections RP_*[nocc×n] = R_*·U^(ij)
// are formed by 3 GEMMs (row-major C trick: C[nocc×n]=R[nocc×nvir]·U[nvir×n]; R_oi
// contiguous ld=nvir, R_moi/R_moj strided ld=nocc·nvir), then per m four GEMVs (W·rp,
// OP_T) accumulate into σ2:  +2·W6·rp_oim  -W6·rp_moi  -W7oj·rp_oim  -W7oi·rp_moj.
// W role select W6/W7oj on (oj==j), W7oi on (oi==j). Verified by index trace vs host.
void DLPNOIPEOMNativeOperator::add_phl_gpu() const {
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const real_t one = 1.0, two = 2.0, neg_one = -1.0, zero = 0.0;
    const int nv = nvir_, no = nocc_;
    const int s_lo = slab_active_ ? cur_slot_begin_ : 0;        // Stage 5c-step2 slab
    const int s_hi = slab_active_ ? cur_slot_end_   : n_orient_;
    for (int s = s_lo; s < s_hi; ++s) {
        const int idx = orient_idx_[s], oi = orient_oi_[s], oj = orient_oj_[s];
        const int n = packing_.n_pno[idx];
        if (n == 0) continue;
        const int j = res_.setups[idx].j;
        const real_t* U = d_U_pack_ + u_pno_off_[idx];          // [nvir × n] row-major
        const size_t woff = wovvo_off_[idx] - wovvo_pack_shift_;  // 5e: slab-only ph-ladder pack
        const real_t* W6   = (oj == j ? d_Wovvo_occj_ : d_Wovvo_occi_) + woff;
        const real_t* W7oj = (oj == j ? d_Wovov_occj_ : d_Wovov_occi_) + woff;
        const real_t* W7oi = (oi == j ? d_Wovov_occj_ : d_Wovov_occi_) + woff;
        // One-sided barS: RP[nocc×n] = R·U  (row-major C trick).
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, no, nv, &one,
                    U, n, d_r2c_all_ + static_cast<size_t>(oi) * no * nv, nv, &zero, d_RP_oim_, n);
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, no, nv, &one,
                    U, n, d_r2c_all_ + static_cast<size_t>(oi) * nv, no * nv, &zero, d_RP_moi_, n);
        cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, no, nv, &one,
                    U, n, d_r2c_all_ + static_cast<size_t>(oj) * nv, no * nv, &zero, d_RP_moj_, n);
        real_t* sig = d_sig_pack_ + orient_off_[s];
        for (int m = 0; m < no; ++m) {
            const real_t* Wm6   = W6   + static_cast<size_t>(m) * n * n;
            const real_t* Wm7oj = W7oj + static_cast<size_t>(m) * n * n;
            const real_t* Wm7oi = W7oi + static_cast<size_t>(m) * n * n;
            const real_t* rp_oim = d_RP_oim_ + static_cast<size_t>(m) * n;
            const real_t* rp_moi = d_RP_moi_ + static_cast<size_t>(m) * n;
            const real_t* rp_moj = d_RP_moj_ + static_cast<size_t>(m) * n;
            // W·rp via OP_T (W row-major [ap×dp] → col-major [dp×ap]; OP_T → Σ_dp W[ap,dp] rp[dp]).
            cublasDgemv(cublas, CUBLAS_OP_T, n, n, &two,     Wm6,   n, rp_oim, 1, &one, sig, 1);  // +2 W6·rp_oim
            cublasDgemv(cublas, CUBLAS_OP_T, n, n, &neg_one, Wm6,   n, rp_moi, 1, &one, sig, 1);  // -W6·rp_moi
            cublasDgemv(cublas, CUBLAS_OP_T, n, n, &neg_one, Wm7oj, n, rp_oim, 1, &one, sig, 1);  // -W7oj·rp_oim
            cublasDgemv(cublas, CUBLAS_OP_T, n, n, &neg_one, Wm7oi, n, rp_moj, 1, &one, sig, 1);  // -W7oi·rp_moj
        }
    }
#endif
}

// B-a.6a Stage 4 σ1: add the enabled GPU σ1 terms into sigma1, reusing the device-
// resident lifted r2 (d_r2c_all_). S1LOO: -Loo·r1 (OP_N). S1FOV: +2 r2c(i,l)·Fov (one
// GEMV OP_T over (l,d)) - per-l r2c(l,:)·Fov_l (OP_T). S1WOOOV: kernel Ssym + Wooov_re·Ssym
// (OP_T). Index traces match compute_sigma1.
void DLPNOIPEOMNativeOperator::add_sigma1_gpu(
    const std::vector<real_t>& r1, std::vector<real_t>& sigma1) const
{
#ifndef GANSU_CPU_ONLY
    cublasHandle_t cublas = reinterpret_cast<cublasHandle_t>(cublas_);
    const int no = nocc_, nv = nvir_;
    const real_t one = 1.0, two = 2.0, neg_one = -1.0;
    cudaMemset(d_sigma1_, 0, static_cast<size_t>(no) * sizeof(real_t));
    // S1LOO: σ1[i] -= Σ_k Loo[k,i] r1[k]. Loo row-major → col-major view, OP_N → Σ_k Loo[k,i]r1[k].
    if (use_gpu_s1loo_) {
        if (!resident_)
            cudaMemcpy(d_r1_, r1.data(), static_cast<size_t>(no) * sizeof(real_t), cudaMemcpyHostToDevice);
        const real_t* r1src = resident_ ? d_r1_src_ : d_r1_;
        cublasDgemv(cublas, CUBLAS_OP_N, no, no, &neg_one, d_Loo_canon_, no,
                    r1src, 1, &one, d_sigma1_, 1);
    }
    // S1FOV: partA +2 Σ_{l,d} r2c(i,l)[d] Fov[l,d] (one GEMV); partB -Σ_l r2c(l,:)·Fov_l (per-l).
    if (use_gpu_s1fov_) {
        cublasDgemv(cublas, CUBLAS_OP_T, no * nv, no, &two, d_r2c_all_, no * nv,
                    d_Fov_, 1, &one, d_sigma1_, 1);
        for (int l = 0; l < no; ++l)
            cublasDgemv(cublas, CUBLAS_OP_T, nv, no, &neg_one,
                        d_r2c_all_ + static_cast<size_t>(l) * no * nv, nv,
                        d_Fov_ + static_cast<size_t>(l) * nv, 1, &one, d_sigma1_, 1);
    }
    // S1WOOOV: Ssym = -2 r2c(k,l) + r2c(l,k); σ1[i] += Σ_{kld} Wooov_re[i,k,l,d] Ssym[k,l,d].
    if (use_gpu_s1wooov_) {
        const int M2 = no * no * nv;
        const int threads = 256, blocks = (M2 + threads - 1) / threads;
        dlpno_ip_native_ssym1_kernel<<<blocks, threads>>>(d_r2c_all_, d_Ssym1_, no, nv);
        cublasDgemv(cublas, CUBLAS_OP_T, M2, no, &one, d_Wooov_re_, M2,
                    d_Ssym1_, 1, &one, d_sigma1_, 1);
    }
    if (resident_) return;   // d_sigma1_ holds the full σ1; caller copies it D2D
    std::vector<real_t> add(static_cast<size_t>(no), 0.0);
    cudaMemcpy(add.data(), d_sigma1_, static_cast<size_t>(no) * sizeof(real_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < no; ++i) sigma1[i] += add[i];
#else
    (void)r1; (void)sigma1;
#endif
}

// Stage 5c: point the σ2 members at device d's workspace (+ its cublas handle on the
// NULL stream so memset/kernel/cublas order exactly like device 0). bind_device(0)
// restores the device-0 members (ws_[0] aliases them). Reassigns through const_cast —
// the operator is not actually const (Davidson holds it mutably); apply() is const only
// by the LinearOperator interface. d_sigma1_/σ1 buffers stay on device 0 and are untouched.
void DLPNOIPEOMNativeOperator::bind_device(int d) const {
#ifndef GANSU_CPU_ONLY
    auto* s = const_cast<DLPNOIPEOMNativeOperator*>(this);
    const DeviceWorkspace& w = ws_[d];
    if (d != 0 && w.cublas) cublasSetStream(reinterpret_cast<cublasHandle_t>(w.cublas), 0);
    s->cublas_         = w.cublas;
    s->d_r2c_all_      = w.d_r2c_all;
    s->d_acc_all_      = w.d_acc_all;
    s->d_sig_pack_     = w.d_sig_pack;
    s->d_U_pack_       = w.d_U_pack;
    s->d_Lvv_pno_pack_ = w.d_Lvv_pno_pack;
    s->d_Loo_lmo_      = w.d_Loo_lmo;
    s->d_Woooo_lmo_    = w.d_Woooo_lmo;
    s->d_Wovoo_re_     = w.d_Wovoo_re;
    s->d_Woovv_lmo_    = w.d_Woovv_lmo;
    s->d_t2_lmo_       = w.d_t2_lmo;
    s->d_S_            = w.d_S;
    s->d_tmp_c_        = w.d_tmp_c;
    s->d_Wovvo_occi_   = w.d_Wovvo_occi;
    s->d_Wovvo_occj_   = w.d_Wovvo_occj;
    s->d_Wovov_occi_   = w.d_Wovov_occi;
    s->d_Wovov_occj_   = w.d_Wovov_occj;
    s->d_RP_oim_       = w.d_RP_oim;
    s->d_RP_moi_       = w.d_RP_moi;
    s->d_RP_moj_       = w.d_RP_moj;
    lvv_pack_shift_    = w.lvv_shift;        // 5e: slab-only Lvv_pno / ph-ladder pack offsets
    wovvo_pack_shift_  = w.wovvo_shift;
#else
    (void)d;
#endif
}

// B-a.6a Stage 4 full residency: device-only matvec. r1 = d_input[0:nocc], packed_r2
// = d_input[nocc:] read straight from device (no input D2H/H2D, no d_r2_pack_ copy).
// lift fills d_r2c_all_; add_sigma1_gpu leaves the full σ1 in d_sigma1_; apply_projection_gpu
// zeroes d_acc_all_, adds every cross-pair/T1/T8 term, projects + T2, and adds the ph-ladder
// into d_sig_pack_. Output assembled with two D2D copies. EA apply_resident mirror.
//
// Stage 5c (use_gpu_multi_): each device d builds the FULL σ2 into its own d_sig_pack
// (step1 = redundant compute, proving replication + per-device identical σ2 + disjoint
// gather; the slab-restricted compute split lands as step2). σ1 is built on device 0.
// Device 0 then gathers each device's assigned orientation-slot slab into d_output.
void DLPNOIPEOMNativeOperator::apply_resident(const real_t* d_input, real_t* d_output) const {
#ifndef GANSU_CPU_ONLY
    const size_t plen = static_cast<size_t>(total_dim_ - nocc_);

    // Stage 5c multi-GPU: per-device σ2 build + disjoint orientation-slot gather.
    // step2 (use_gpu_multi_slab_): each device computes ONLY its slab (real compute
    // split); the lift stays full (cross-pair reads all pairs). step1 (NOSLAB): each
    // device builds the full σ2 redundantly. Both gather the same disjoint slabs.
    if (use_gpu_multi_ && ws_.size() >= 2) {
        auto& mgr = MultiGpuManager::instance();
        resident_ = true;
        std::vector<real_t> unused;
        // Device 0: σ1 (device-0 only, full) + σ2 (slab when step2) into d_sig_pack_.
        {
            MultiGpuManager::DeviceGuard guard(0);
            bind_device(0);
            if (use_gpu_multi_slab_) { slab_active_ = true;
                cur_slot_begin_ = slot_begin_[0]; cur_slot_end_ = slot_end_[0]; }
            d_r1_src_ = d_input; d_r2_src_ = d_input + nocc_;
            lift_r2c_gpu(unused, unused);
            add_sigma1_gpu(unused, unused);          // σ1 ignores slab_active_ (full)
            apply_projection_gpu(unused, unused, unused, unused);
        }
        // Devices d>0: broadcast input, lift (full), then σ2 (slab when step2). The
        // broadcast is async on device d's null stream (ordered before the lift, which
        // also runs on stream 0), so the host does not block and the per-device compute
        // pipelines overlap across GPUs (sync_all below ensures completion before gather).
        for (int d = 1; d < static_cast<int>(ws_.size()); ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            bind_device(d);
            if (use_gpu_multi_slab_) { slab_active_ = true;
                cur_slot_begin_ = slot_begin_[d]; cur_slot_end_ = slot_end_[d]; }
            cudaMemcpyPeerAsync(ws_[d].d_input, d, d_input, 0,
                                static_cast<size_t>(total_dim_) * sizeof(real_t), 0);
            d_r1_src_ = ws_[d].d_input; d_r2_src_ = ws_[d].d_input + nocc_;
            lift_r2c_gpu(unused, unused);
            apply_projection_gpu(unused, unused, unused, unused);
        }
        slab_active_ = false;      // back to full for any later non-multi use
        bind_device(0);            // restore members to device 0
        mgr.sync_all();
        // Assemble: σ1 from device 0; gather each device's orientation-slot slab into σ2.
        {
            MultiGpuManager::DeviceGuard guard(0);
            cudaMemcpy(d_output, d_sigma1_, static_cast<size_t>(nocc_) * sizeof(real_t),
                       cudaMemcpyDeviceToDevice);
            for (int d = 0; d < static_cast<int>(ws_.size()); ++d)
                for (int sslot = slot_begin_[d]; sslot < slot_end_[d]; ++sslot) {
                    const int n = packing_.n_pno[orient_idx_[sslot]];
                    if (n == 0) continue;
                    const size_t o = orient_off_[sslot];
                    real_t* dst = d_output + nocc_ + o;
                    if (d == 0)
                        cudaMemcpy(dst, ws_[0].d_sig_pack + o, static_cast<size_t>(n) * sizeof(real_t),
                                   cudaMemcpyDeviceToDevice);
                    else
                        cudaMemcpyPeer(dst, 0, ws_[d].d_sig_pack + o, d,
                                       static_cast<size_t>(n) * sizeof(real_t));
                }
        }
        resident_ = false; d_r1_src_ = nullptr; d_r2_src_ = nullptr;

        if (multi_selfcheck_ && multi_check_done_ < kMultiCheckMax) {
            ++multi_check_done_;
            std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
            cudaMemcpy(h_in.data(), d_input,
                       static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
            std::vector<real_t> r1(h_in.begin(), h_in.begin() + nocc_);
            std::vector<real_t> packed_r2(h_in.begin() + nocc_, h_in.end());
            std::vector<real_t> ref1, ref2;
            compute_sigma1(r1, packed_r2, ref1);
            compute_sigma2(r1, packed_r2, ref2, /*skip_t2=*/false);
            std::vector<real_t> got(static_cast<size_t>(total_dim_));
            cudaMemcpy(got.data(), d_output,
                       static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
            real_t d1 = 0.0, d2 = 0.0;
            for (int i = 0; i < nocc_; ++i) d1 = std::max(d1, std::fabs(got[i] - ref1[i]));
            for (size_t k = 0; k < plen; ++k) d2 = std::max(d2, std::fabs(got[nocc_ + k] - ref2[k]));
            std::cout << "[bt-PNO Stage 5c self-check] gathered σ vs full host: max|σ1| = "
                      << std::scientific << d1 << ", max|σ2| = " << d2
                      << "  (expect ≤1e-11 = multi-GPU gather == single-device)" << std::endl;
        }
        return;
    }

    resident_ = true;
    d_r1_src_ = d_input;             // r1        = d_input[0:nocc]
    d_r2_src_ = d_input + nocc_;     // packed_r2 = d_input[nocc:]

    std::vector<real_t> unused;      // host buffers untouched on the resident path
    lift_r2c_gpu(unused, unused);                              // → d_r2c_all_ (resident)
    add_sigma1_gpu(unused, unused);                            // → d_sigma1_ (full σ1)
    apply_projection_gpu(unused, unused, unused, unused);      // → d_sig_pack_ (full σ2)

    cudaMemcpy(d_output, d_sigma1_, static_cast<size_t>(nocc_) * sizeof(real_t),
               cudaMemcpyDeviceToDevice);
    if (plen > 0)
        cudaMemcpy(d_output + nocc_, d_sig_pack_, plen * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);

    resident_ = false;
    d_r1_src_ = nullptr; d_r2_src_ = nullptr;

    if (gpu_selfcheck_) {
        std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
        cudaMemcpy(h_in.data(), d_input,
                   static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
        std::vector<real_t> r1(h_in.begin(), h_in.begin() + nocc_);
        std::vector<real_t> packed_r2(h_in.begin() + nocc_, h_in.end());
        std::vector<real_t> ref1, ref2;
        compute_sigma1(r1, packed_r2, ref1);
        compute_sigma2(r1, packed_r2, ref2, /*skip_t2=*/false);
        std::vector<real_t> got(static_cast<size_t>(total_dim_));
        cudaMemcpy(got.data(), d_output,
                   static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
        real_t d1 = 0.0, d2 = 0.0;
        for (int i = 0; i < nocc_; ++i) d1 = std::max(d1, std::fabs(got[i] - ref1[i]));
        for (size_t k = 0; k < plen; ++k) d2 = std::max(d2, std::fabs(got[nocc_ + k] - ref2[k]));
        std::cout << "[bt-PNO B-a.6a GPU self-check] full-residency max|σ1 - host| = "
                  << std::scientific << d1 << ", max|σ2 - host| = " << d2
                  << "  (expect ≤1e-11 = device-only matvec == host)" << std::endl;
    }
#else
    (void)d_input; (void)d_output;
#endif
}

void DLPNOIPEOMNativeOperator::apply(const real_t* d_input, real_t* d_output) const {
    // B-a.6a Stage 4: full-residency device-only path (no host round-trip) when the
    // complete GPU term set is on. The host-assisted paths below are byte-unchanged.
    if (use_gpu_resident_) { apply_resident(d_input, d_output); return; }

    // D2H packed input → [r1 (canonical, nocc) | packed_r2].
    std::vector<real_t> h_in(static_cast<size_t>(total_dim_));
    cudaMemcpy(h_in.data(), d_input,
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyDeviceToHost);
    std::vector<real_t> r1(h_in.begin(), h_in.begin() + nocc_);
    std::vector<real_t> packed_r2(h_in.begin() + nocc_, h_in.end());

    // σ1: host builds non-GPU terms (skip flags default false → byte-unchanged); when
    // σ1 GPU is on (⊂ use_gpu_xpair_), add_sigma1_gpu fills the skipped terms after the
    // lift has populated d_r2c_all_.
    std::vector<real_t> sigma1;
    compute_sigma1(r1, packed_r2, sigma1,
                   /*skip_loo=*/use_gpu_s1loo_, /*skip_fov=*/use_gpu_s1fov_,
                   /*skip_wooov=*/use_gpu_s1wooov_);

    std::vector<real_t> packed_sigma2;
    if (use_gpu_lift_) {
        // GPU path (Stage 3a): device lifts the source amplitudes r2c[p,q] = U·r2s;
        // the host acc-build consumes them (transitional round-trip), then the device
        // projects (Stage 2) + adds T2 (Stage 1). The host writes the PNO-space ph-
        // ladder T6/T7. NUMERICS-PRESERVING — the self-check confirms it bit-for-bit.
        std::vector<real_t> r2c_all(static_cast<size_t>(nocc_) * nocc_ * nvir_, 0.0);
        lift_r2c_gpu(packed_r2, r2c_all);
        // σ1 (Stage 4): the lift just populated d_r2c_all_; add the enabled GPU σ1 terms
        // onto the host-built sigma1 (which omitted them). NUMERICS-PRESERVING.
        if (use_gpu_s1loo_ || use_gpu_s1fov_ || use_gpu_s1wooov_) {
            add_sigma1_gpu(r1, sigma1);
            if (gpu_selfcheck_) {
                std::vector<real_t> ref1;
                compute_sigma1(r1, packed_r2, ref1);   // full host σ1
                real_t d1 = 0.0;
                for (int i = 0; i < nocc_; ++i) d1 = std::max(d1, std::fabs(sigma1[i] - ref1[i]));
                std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ1(GPU"
                          << (use_gpu_s1loo_ ? " Loo·r1" : "")
                          << (use_gpu_s1fov_ ? "+Fov·r2" : "")
                          << (use_gpu_s1wooov_ ? "+Wooov·r2" : "")
                          << ") - σ1(host)| = " << std::scientific << d1
                          << "  (expect ≤1e-11 = GPU σ1 GEMV/kernel == host)" << std::endl;
            }
        }
        std::vector<real_t> acc_all(static_cast<size_t>(n_orient_) * nvir_, 0.0);
        compute_sigma2(r1, packed_r2, packed_sigma2, /*skip_t2=*/true, &acc_all, &r2c_all,
                       /*skip_xpair=*/use_gpu_xpair_, /*skip_t1=*/use_gpu_t1_,
                       /*skip_t8=*/use_gpu_t8_, /*skip_phl=*/use_gpu_phl_);
        apply_projection_gpu(r1, acc_all, packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_t2=*/false);  // full host (host lift)
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU lift"
                      << (use_gpu_xpair_ ? "+xpair(T3/T4/T5)" : "")
                      << (use_gpu_t1_ ? "+T1" : "") << (use_gpu_t8_ ? "+T8" : "")
                      << (use_gpu_phl_ ? "+T6/T7" : "")
                      << "+proj+T2) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU "
                      << (use_gpu_xpair_ ? "cross-pair GEMVs" : "source lift U·r2s")
                      << " == host)" << std::endl;
        }
    } else if (use_gpu_proj_) {
        // GPU path (Stage 2): host builds the canonical-virtual acc (T1/T3/T4/T5/T8b)
        // and exports it per orientation, and writes the PNO-space ph-ladder T6/T7
        // into packed_sigma2; the device adds U^(ij)ᵀ·acc (projection) + T2.
        // NUMERICS-PRESERVING — the self-check confirms it bit-for-bit.
        std::vector<real_t> acc_all(static_cast<size_t>(n_orient_) * nvir_, 0.0);
        compute_sigma2(r1, packed_r2, packed_sigma2, /*skip_t2=*/true, &acc_all);
        apply_projection_gpu(r1, acc_all, packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_t2=*/false);  // full host
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU proj+T2) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU chained projection == host U^(ij)ᵀ·acc)"
                      << std::endl;
        }
    } else if (use_gpu_) {
        // GPU path: host computes σ2 with the dressed T2 omitted, the device adds
        // T2 (Stage 1). NUMERICS-PRESERVING; the self-check confirms it bit-for-bit.
        compute_sigma2(r1, packed_r2, packed_sigma2, /*skip_t2=*/true);
        apply_t2_gpu(packed_r2, packed_sigma2);
        if (gpu_selfcheck_) {
            std::vector<real_t> ref;
            compute_sigma2(r1, packed_r2, ref, /*skip_t2=*/false);  // full host
            real_t dmax = 0.0;
            for (size_t k = 0; k < ref.size(); ++k)
                dmax = std::max(dmax, std::fabs(packed_sigma2[k] - ref[k]));
            std::cout << "[bt-PNO B-a.6a GPU self-check] max|σ2(GPU T2) - σ2(host)| = "
                      << std::scientific << dmax
                      << "  (expect ≤1e-11 = GPU cublasDgemv == host PNO-space contraction)"
                      << std::endl;
        }
    } else {
        compute_sigma2(r1, packed_r2, packed_sigma2);
    }

    std::vector<real_t> h_out(static_cast<size_t>(total_dim_), 0.0);
    std::memcpy(h_out.data(), sigma1.data(), static_cast<size_t>(nocc_) * sizeof(real_t));
    std::memcpy(h_out.data() + nocc_, packed_sigma2.data(),
                packed_sigma2.size() * sizeof(real_t));
    cudaMemcpy(d_output, h_out.data(),
               static_cast<size_t>(total_dim_) * sizeof(real_t), cudaMemcpyHostToDevice);
}

void DLPNOIPEOMNativeOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
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
        dlpno_ip_native_precondition_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, total_dim_);
#endif
    }
}

} // namespace gansu
