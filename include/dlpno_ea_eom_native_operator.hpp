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
 * @file dlpno_ea_eom_native_operator.hpp
 * @brief Native per-pair DLPNO-EA-EOM-CCSD σ operator (bt-PNO-STEOM stage B
 *        (a), Dutta-Saitow-Riplinger-Neese-Izsák 2018 JCP 148, 244101).
 *
 * The EA analog of DLPNOIPEOMNativeOperator. The 2p1h amplitude r^{ab}_i keeps
 * BOTH virtuals in the diagonal-pair PNO basis a,b ∈ PNO(i,i); the packed
 * vector is [ R1 (nvir) | per-i n_pno(ii)² blocks ] (DLPNOEAPacking). The σ2
 * terms accumulate into a 2-virtual canonical buffer acc[a,b] (one nvir×nvir
 * matrix per output occ j) which is finally projected to the pair PNO:
 *     σ2_packed^(jj) = U^(jj)ᵀ · acc · U^(jj)         (two-sided, U^(jj)=C_virᵀ S bar_Q_jj)
 * Source amplitudes are lifted r2_canon[l] = U^(ll) · r2_packed^(ll) · U^(ll)ᵀ.
 *
 * Build-up (each sub-phase gated bit-exact vs the (b) projected reference at no
 * truncation, mirroring the IP B-a sequence):
 *   B-EA.1 (this commit): σ1 (1p sector) via lift + canonical formula on the
 *     borrowed (bit-identical) intermediates; σ2 = diagonal placeholder.
 *     σ1[a] = Σ_c Lvv[a,c] r1[c] + 2 Σ_{l,d} Fov[l,d] r2c[l,a,d]
 *           - Σ_{l,d} Fov[l,d] r2c[l,d,a]
 *           + Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2c[l,c,d]
 *     (r2c = ea_packed_r2_to_canonical). Stands up the operator + borrowed
 *     intermediates + the per-matvec σ1 gate before the σ2 terms land.
 *   B-EA.2 (this commit): + σ2 pair-local terms (only the output pair j's own
 *     lifted r2, or global r1 — no cross-pair barS):
 *       T_Lvv_a: acc += Lvv · r2c[j]              (Σ_c Lvv[a,c] r2c[j][c,b])
 *       T_Lvv_b: acc += r2c[j] · Lvvᵀ             (Σ_d Lvv[b,d] r2c[j][a,d])
 *       T_r1   : acc[a,b] += Σ_c Wvvvo_lmo[a,b,c,j] r1[c]   (1p→2p1h coupling)
 *     where acc[a,b] (canonical virtuals) → σ2_packed^(jj) = U^(jj)ᵀ acc U^(jj),
 *     r2c[i] = U^(ii) r2_packed^(ii) U^(ii)ᵀ. Wvvvo_lmo carries the single-occ
 *     U_loc rotation of the j index (copy for none).
 *   B-EA.3 (this commit): + cross-pair (source occ l ≠ j, uses r2c[l]):
 *       T_Loo : acc -= Σ_l Loo_lmo[l,j] r2c[l]
 *       T_ph1 : acc[a,b] += Σ_{l,d} (2 Wovvo_lmo[l,b,d,j] - Wovov_lmo[l,b,j,d]) r2c[l][a,d]
 *       T_ph2 : acc[a,b] -= Σ_{l,c} Wovov_lmo[l,a,j,c] r2c[l][c,b]
 *       T_ph3 : acc[a,b] -= Σ_{l,c} Wovvo_lmo[l,b,c,j] r2c[l][c,a]
 *     Loo_lmo = U_locᵀ Loo U_loc; Wovvo_lmo (occ pos 0,3) / Wovov_lmo (occ pos 0,2)
 *     2-occ-index rotations (copy for none) — identical to the IP B-a.4 forms.
 *   B-EA.4 (this commit): + the last two σ2 terms → native σ complete:
 *       T_tmp : tmp[K] = Σ_{l,C,D} (2 ovov_Llmo[K,C,l,D] - ovov_Llmo[K,D,l,C]) r2c[l][C,D]
 *               acc[a,b] -= Σ_K tmp[K] t2_Jlmo[K,j,a,b]      (global K-reduction)
 *       T_vvvv: acc[a,b] += Σ_{c,d} Wvvvv[a,b,c,d] r2c[j][c,d]   (pair-local, 4-virtual)
 *     tmp keeps K canonical; ovov_Llmo rotates ovov's L (3rd-occ) index → LMO,
 *     t2_Jlmo rotates t2's 2nd-occ index → LMO (single-occ; copy for none).
 *     Wvvvv is occ-free (no rotation). After this commit native σ == (b)
 *     reference directly (full σ1 + σ2).
 *
 * Until all σ2 terms land the operator is INCOMPLETE, so it is NOT wired into
 * the production driver — only GANSU_DLPNO_EA_NATIVE_VALIDATE exercises it.
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "dlpno_mp2.hpp"             // DLPNOLMP2Result
#include "dlpno_ea_packing.hpp"      // DLPNOEAPacking
#include "ea_eom_ccsd_operator.hpp"  // EAEOMCCSDOperator (borrowed intermediates)
#include "dlpno_eom_dressed_pno.hpp" // DressedPnoEA (B-EA.6d true-scaling Wvvvv)

namespace gansu {

class DLPNOEAEOMNativeOperator : public LinearOperator {
public:
    /**
     * @param ea_op    Canonical EA-EOM operator (built from the same DLPNO
     *                 back-transformed T1/T2) whose dressed intermediates this
     *                 operator borrows. Must outlive this operator and have been
     *                 constructed with a non-null MO ERI.
     * @param res      Converged DLPNO result (per-pair bar_Q/Lambda/setups). Ref held.
     * @param packing  EA packed-vector offset table. Ref held.
     * @param U_loc    [nocc²] localization rotation (copied).
     * @param C_vir    [nao·nvir] canonical virtual coefficients (copied).
     * @param h_S      [nao²] AO overlap (copied).
     * @param nao,nvir dimensions.
     * @param eps_v    [nvir] virtual orbital energies (1p Koopmans diagonal).
     */
    DLPNOEAEOMNativeOperator(const EAEOMCCSDOperator& ea_op,
                             const DLPNOLMP2Result& res,
                             const DLPNOEAPacking& packing,
                             const std::vector<real_t>& U_loc,
                             const std::vector<real_t>& C_vir,
                             const real_t* h_S,
                             int nao, int nvir,
                             const std::vector<real_t>& eps_v,
                             int num_gpus = 1);

    ~DLPNOEAEOMNativeOperator();

    DLPNOEAEOMNativeOperator(const DLPNOEAEOMNativeOperator&) = delete;
    DLPNOEAEOMNativeOperator& operator=(const DLPNOEAEOMNativeOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;

    /// Stage 4 full residency: device-only matvec reading r1/r2 from d_input and
    /// assembling σ into d_output on device (no host round-trip). Used when
    /// use_gpu_resident_ (the complete GPU term set + GANSU_DLPNO_NATIVE_GPU_RESIDENT).
    void apply_resident(const real_t* d_input, real_t* d_output) const;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "DLPNOEAEOMNativeOperator"; }

private:
    const DLPNOLMP2Result& res_;
    const DLPNOEAPacking& packing_;
    std::vector<real_t> U_loc_;
    std::vector<real_t> C_vir_;
    std::vector<real_t> h_S_;
    int nao_ = 0;
    int nvir_ = 0;
    int nocc_ = 0;
    int total_dim_ = 0;

    // B-a.6a Stage 5 multi-GPU: device count threaded from rhf.get_num_gpus() (≥1).
    // Stage 5a scaffolding only stores it; the per-device fan-out (broadcast input,
    // redundant lift, slab σ2, peer gather) lands in later sub-stages. See
    // project_p5_stage5_multigpu_plan.md. Single-GPU path (num_gpus_==1) byte-unchanged.
    int num_gpus_ = 1;

    // Stage 5b multi-GPU (env GANSU_DLPNO_NATIVE_GPU_MULTI=1; requires
    // use_gpu_resident_ && num_gpus_>1). Per-device replica of the lift constant
    // (d_U_pack_) + the chained-GEMM scratch (d_lift_T) + a broadcast input copy +
    // a d_r2c_all_ scratch on every d>0, so each device can redundantly lift its own
    // copy of the source amplitudes (residency decision A) from a cudaMemcpyPeer
    // broadcast of d_input (decision B). In 5b the σ1/σ2 build and the assembled
    // output stay 100% on device 0 (production byte-identical); the d>0 lifts run
    // only as a validation of the broadcast + per-device residency (each device's
    // d_r2c_all_ vs device 0, ≤1e-11) under GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE.
    // The slab σ2 split + peer gather land in Stage 5c. Mirrors the IP operator.
    bool use_gpu_multi_ = false;       ///< broadcast + per-device lift wired (num_gpus_>1)
    bool multi_selfcheck_ = false;     ///< gathered σ vs full device-0 host reference (first matvecs)
    // Cap the 5c self-check at the first few matvecs (the per-matvec host recompute is
    // ruinous at scale — it's exactly the host path multi-GPU avoids). Keeps
    // GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE usable for large-system timing. Mirrors IP.
    mutable int multi_check_done_ = 0;
    static constexpr int kMultiCheckMax = 3;
    // Stage 5c: each device d holds a FULL replica of every σ2 device buffer (constants
    // peer-copied from device 0, scratch freshly allocated) + a cublas handle, so the
    // validated σ2 helper chain runs unchanged on device d after bind_device(d). ws_[0]
    // aliases the device-0 members so bind_device(0) is an exact restore. Mirrors IP.
    struct DeviceWorkspace {
        int     device    = -1;        ///< CUDA device id (ws_[0].device = 0, aliases members)
        void*   cublas     = nullptr;  ///< cublasHandle_t for this device (device 0 = cublas_)
        real_t* d_input    = nullptr;  ///< [total_dim] broadcast copy of the matvec input (d>0)
        real_t* d_r2c_all  = nullptr;  ///< [nocc·nvir²] per-device lifted source r2c
        real_t* d_lift_T   = nullptr;  ///< [nvir·max_n_pno] lift chained-GEMM scratch
        real_t* d_acc_all  = nullptr;  ///< [nocc·nvir²] acc stack scratch
        real_t* d_sig_pack = nullptr;  ///< [total_dim-nvir] σ2 output scratch
        real_t* d_T1       = nullptr;  ///< [nvir·max_n_pno] projection chained-GEMM scratch
        real_t* d_r2c_sym_re = nullptr;///< [nvir·nocc·nvir] T_tmp scratch
        real_t* d_tmp      = nullptr;  ///< [nocc] T_tmp scratch
        real_t* d_U_pack   = nullptr;
        real_t* d_Wvvvv_pno_pack = nullptr;
        size_t  wvvvv_shift = 0;       ///< 5e: subtract from wvvvv_pno_off_[j] for slab-only pack (0 = full)
        real_t* d_Loo_xpair = nullptr;
        real_t* d_Wovov_lmo = nullptr;
        real_t* d_Wovvo_re = nullptr;
        real_t* d_ovov_Llmo = nullptr;
        real_t* d_t2_Jlmo  = nullptr;
        real_t* d_Lvv      = nullptr;
        real_t* d_Wvvvo_r1 = nullptr;
        int     wovvo_re_j_extent = 0; ///< 5h: j-dim extent of the d_Wovvo_re slice (nocc full, occ_end-occ_begin slab)
        int     wovvo_re_j_base   = 0; ///< 5h: j-dim base of the d_Wovvo_re slice (0 full, occ_begin slab)
        int     t2_jlmo_j_extent  = 0; ///< 5i: j-dim extent of the d_t2_Jlmo slice (nocc full, slab) → GEMV lda = extent·nvir²
        int     t2_jlmo_j_base    = 0; ///< 5i: j-dim base of the d_t2_Jlmo slice (0 full, occ_begin slab)
        int     wovov_j_extent    = 0; ///< 5j: j-dim extent of the d_Wovov_lmo slice (nocc full, slab) → GEMM ldB = extent·nvir
        int     wovov_j_base      = 0; ///< 5j: j-dim base of the d_Wovov_lmo slice (0 full, occ_begin slab)
    };
    std::vector<DeviceWorkspace> ws_;  ///< size = #devices used (ws_[0] aliases device 0)
    std::vector<int> occ_begin_;       ///< [#dev] output-occ slab start per device
    std::vector<int> occ_end_;         ///< [#dev] output-occ slab end (exclusive) per device
    // Stage 5c-step2 (compute split): when use_gpu_multi_slab_, each device computes σ2
    // ONLY for its output-occ slab. slab_active_ + cur_occ_* are set per device in
    // apply_resident; the σ2 helpers read them (full [0,nocc_) when slab_active_ is
    // false → single-GPU + step1 byte-unchanged). The global T_tmp stage-1 tmp[K]
    // reduction stays full. The T_Loo stacked GEMM and T_tmp stage-2 GEMV (which write
    // the whole acc in one call) restrict their output-occ column range to the slab.
    // Default ON for multi; GANSU_DLPNO_NATIVE_GPU_MULTI_NOSLAB=1 forces step1.
    bool use_gpu_multi_slab_ = false;
    mutable bool slab_active_ = false;
    mutable int  cur_occ_begin_ = 0;
    mutable int  cur_occ_end_   = 0;
    // 5e: output-indexed Wvvvv pack slab-only. d>0 in slab mode hold only their occ
    // slab's n_pno⁴ packs; project_acc_stack reads d_Wvvvv_pno_pack_[wvvvv_pno_off_[j] -
    // wvvvv_pack_shift_]. 0 on device 0 / full / single-GPU. Set per device in bind_device.
    mutable size_t wvvvv_pack_shift_ = 0;
    // 5h: d_Wovvo_re is sliced on d>0 in slab mode — the j (2nd) axis is restricted to the
    // device's output-occ slab, ALL l (outermost) kept. The readers (add_tph3 / add_tph1)
    // index it as (l·wovvo_re_j_extent_ + (j - wovvo_re_j_base_))·nvir². extent = nocc /
    // base = 0 on device 0 / full / single-GPU (→ the original (l·nocc + j)·nvir²). Set in
    // the ctor (single-GPU default) and per device in bind_device.
    mutable int wovvo_re_j_extent_ = 0;
    mutable int wovvo_re_j_base_   = 0;
    // 5i: d_t2_Jlmo is sliced on d>0 in slab mode — the j (2nd) axis restricted to the
    // device's output-occ slab, ALL K (outermost) kept. The add_ttmp stage-2 GEMV reads it
    // col-major with lda = t2_jlmo_j_extent_·nvir² (the K-axis stride; the FIRST cuBLAS-lda
    // change of the domain-sparse work) and A_ptr offset (j_lo - t2_jlmo_j_base_)·nvir² (= 0
    // on the slab device, j_lo·nvir² on full). extent = nocc / base = 0 on device 0 / full /
    // single-GPU (→ lda = nocc·nvir², the original m2). Set in the ctor + per device in bind_device.
    mutable int t2_jlmo_j_extent_ = 0;
    mutable int t2_jlmo_j_base_   = 0;
    // 5j: d_Wovov_lmo is sliced on d>0 in slab mode — the j (3rd) axis (layout [l,a,j,d],
    // j stride nvir) restricted to the device's output-occ slab. Because j is deep (not the
    // 2nd axis), the slice is a HOST repack from h_Wovov_lmo_ + H2D (a device peer copy would
    // be nocc·nvir tiny strided copies). The readers (add_tph2 / add_tph1 term B) use the
    // a-axis row-stride ldB = wovov_j_extent_·nvir and base (l·nvir·extent + (j - base))·nvir.
    // extent = nocc / base = 0 on device 0 / full / single-GPU (→ ldB = nocc·nvir, original).
    mutable int wovov_j_extent_ = 0;
    mutable int wovov_j_base_   = 0;
    /// Stage 5c: point the (const_cast) σ2 members at device d's workspace buffers +
    /// cublas handle (NULL stream); bind_device(0) restores the device-0 members.
    void bind_device(int d) const;
    /// Stage 5b legacy (kept for the num_gpus=1 byte path / standalone validation;
    /// superseded by the 5c gathered-σ self-check).
    void lift_r2c_multi_validate(const real_t* d_input) const;

    // Borrowed canonical intermediates (host copies; bit-identical to ea_op).
    std::vector<real_t> h_Lvv_;       // [nvir²]                σ1, σ2 T_Lvv
    std::vector<real_t> h_Fov_;       // [nocc·nvir]            σ1
    std::vector<real_t> h_Wvovv_;     // [nvir·nocc·nvir·nvir]  σ1
    std::vector<real_t> h_Wvvvo_lmo_; // [nvir·nvir·nvir·nocc]  σ2 T_r1 (occ j rotated → LMO)
    std::vector<real_t> h_Loo_lmo_;   // [nocc²]                σ2 T_Loo (U_locᵀ Loo U_loc)
    std::vector<real_t> h_Wovvo_lmo_; // [nocc·nvir·nvir·nocc]  σ2 ph (occ pos 0,3 rotated → LMO)
    std::vector<real_t> h_Wovov_lmo_; // [nocc·nvir·nocc·nvir]  σ2 ph (occ pos 0,2 rotated → LMO)
    std::vector<real_t> h_ovov_Llmo_; // [nocc·nvir·nocc·nvir]  σ2 T_tmp (eri_ovov, L=3rd occ rotated → LMO)
    std::vector<real_t> h_t2_Jlmo_;   // [nocc²·nvir²]          σ2 T_tmp (CCSD T2, 2nd occ rotated → LMO)
    std::vector<real_t> h_Wvvvv_;     // [nvir⁴]                σ2 T_vvvv (occ-free)

    // B-EA.6d true-scaling path (env GANSU_DLPNO_NATIVE_DRESSED=1, native ring
    // sub-flag GANSU_DLPNO_NATIVE_RING=1). When on, the σ2 T_vvvv term contracts
    // a per-occ diagonal-PNO Wvvvv^(jj) (built in the ctor) directly in PNO space
    // instead of the dense h_Wvvvv_. Off by default → dense path, bit-exact.
    bool use_dressed_pno_ = false;
    bool use_native_ring_ = false;
    bool use_native_bare_ = false;               ///< B-EA.6e: W_pair bare seed + native-only ring, NO dense Wvvvv/DR
    DressedPnoEA dressed_;                       ///< per-occ Wvvvv^(jj) [n_pno(jj)⁴]
    std::vector<std::vector<real_t>> Uall_;      ///< [nocc] of U^(jj) flat [nvir × n_pno(jj)]

    // B-a.6a GPU port — Stage 1 (env GANSU_DLPNO_NATIVE_GPU=1; requires
    // use_dressed_pno_). The pair-local PNO-space σ2 T_vvvv term contracts the
    // per-occ diagonal Wvvvv^(jj) on device (per-occ cublasDgemv; row-major W
    // viewed col-major + op_T → y = W·x). Off by default → the host T_vvvv path
    // is byte-unchanged. GANSU_DLPNO_NATIVE_GPU_VALIDATE adds an in-process
    // GPU-vs-host σ2 diff each matvec (probe-independent gate, ≤1e-11).
    bool use_gpu_ = false;
    bool gpu_selfcheck_ = false;
    // Per-term apply profiling (env GANSU_DLPNO_NATIVE_PROF=1). Off by default →
    // zero overhead (no cudaDeviceSynchronize). Run with --num_gpus 1 for a clean
    // per-matvec breakdown (prof_calls_ = matvec count); printed in the destructor.
    mutable bool   prof_ = false;
    mutable long   prof_calls_ = 0;
    mutable double prof_t_lift_ = 0, prof_t_s1_ = 0, prof_t_loo_ = 0, prof_t_tlvv_ = 0,
                   prof_t_tr1_ = 0, prof_t_ph2_ = 0, prof_t_ph3_ = 0, prof_t_ph1_ = 0,
                   prof_t_tmp_ = 0, prof_t_proj_ = 0;
    void print_profile() const;          ///< dump the accumulated per-term timings (if prof_)
    /// GPU 4-index congruence: fills dressed_.Wvvvv_pno[j] = U^(jj)ᵀ⊗4 · Wvvvv for every
    /// occ j, reading the dense Wvvvv straight off the device (4 cublasDgemm[StridedBatched]
    /// per occ). Replaces the host congruence4 quad-loop (the EA operator-build hotspot).
    void build_dressed_vvvv_gpu(const real_t* d_Wvvvv);
    /// Stage F5b on-device ph-ladder rotation: build d_Wovov_lmo_ + d_Wovvo_re_ directly
    /// from ea_op.get_W{ovov,ovvo}_device() WITHOUT ever materialising the persistent host
    /// members h_Wovov_lmo_/h_Wovvo_lmo_ or the transient h_Wovvo/h_Wovov pulls /
    /// h_Wovvo_re transpose. Eliminates the EA-ctor's ~6.4 GB transient host peak at
    /// anthracene / ~1.4 TB at 100 atoms (F5 only released the persistent footprint after
    /// ctor; F5b also avoids the transient peak). Chunked over the virtual `a` axis so
    /// the device intermediate stays ≤ ~16 GB. Identity uloc skips the GEMMs.
    /// Active ONLY when use_native_bare_ && use_gpu_resident_ && !gpu_selfcheck_
    /// && num_gpus_ == 1; otherwise the F5 (host-rotation + late-free) path runs.
    void borrow_ph_to_device_f5b(const real_t* d_Wovvo_canon,
                                 const real_t* d_Wovov_canon);
    void* cublas_ = nullptr;             ///< cublasHandle_t (opaque; keeps cublas out of this header)
    real_t* d_Wvvvv_pno_pack_ = nullptr; ///< packed per-occ Wvvvv^(jj), n_pno(jj)⁴ back-to-back
    std::vector<size_t> wvvvv_pno_off_;  ///< [nocc+1] prefix offsets (n⁴) into d_Wvvvv_pno_pack_
    real_t* d_r2_pack_  = nullptr;       ///< [total_dim-nvir] device scratch (T_vvvv GEMV input)
    real_t* d_sig_pack_ = nullptr;       ///< [total_dim-nvir] device scratch (T_vvvv GEMV output)

    // B-a.6a GPU port — Stage 2 (env GANSU_DLPNO_NATIVE_GPU_PROJ=1; requires
    // use_gpu_). The two-sided PNO projection σ2_packed^(jj) = U^(jj)ᵀ acc U^(jj)
    // (the dominant EA 100-atom cost — two-sided projection of the nvir² acc) is
    // moved to a per-occ chained cublasDgemm pair; the validated Stage-1 T_vvvv is
    // then accumulated into the same packed σ2 block. The host still builds acc
    // (T_Lvv/T_r1/cross-pair/T_tmp) and exports it; only the projection + T_vvvv
    // run on device. Off by default → the host projection path is byte-unchanged.
    bool use_gpu_proj_ = false;
    real_t* d_U_pack_ = nullptr;         ///< packed per-occ U^(jj), nvir·n_pno(jj) row-major back-to-back
    std::vector<size_t> u_pno_off_;      ///< [nocc+1] prefix offsets (nvir·n) into d_U_pack_
    real_t* d_acc_ = nullptr;            ///< [nvir²] per-occ acc upload scratch
    real_t* d_T1_  = nullptr;            ///< [nvir·max_n_pno] chained-GEMM intermediate (acc·U)
    int max_n_pno_ = 0;                  ///< max_j n_pno(jj) (sizes d_T1_/d_lift_T_)

    // B-a.6a GPU port — Stage 3a (env GANSU_DLPNO_NATIVE_GPU_LIFT=1; requires
    // use_gpu_proj_). The per-matvec source lift r2c[l] = U^(ll)·r2_packed^(ll)·U^(ll)ᵀ
    // (the inverse-direction chained GEMM of the Stage-2 projection) is moved to a
    // per-occ cublasDgemm pair, leaving the result device-resident (d_r2c_all_) for
    // the Stage-3b cross-pair contractions. For now the lifted r2c is also pulled
    // back to host so the existing host acc-build consumes it (validated end-to-end
    // by the σ2 self-check); the round-trip is removed once the contractions move
    // to device. Off by default → the host lift path is byte-unchanged.
    bool use_gpu_lift_ = false;
    real_t* d_r2c_all_ = nullptr;        ///< [nocc·nvir²] device-resident lifted r2c (block l at l·nvir²)
    real_t* d_lift_T_  = nullptr;        ///< [nvir·max_n_pno] chained-GEMM intermediate (U·r2p)

    // B-a.6a GPU port — Stage 3b (env GANSU_DLPNO_NATIVE_GPU_XPAIR=1; requires
    // use_gpu_lift_). The first cross-pair term T_Loo (acc[j] += Σ_l -Loo_lmo[l,j]
    // r2c[l]) is moved to a single stacked GEMM ACC[nocc×nvir²] += M_Loo·R_stack
    // (M_Loo[j,l] = -Loo_lmo[l,j], R_stack = d_r2c_all_), accumulated onto the
    // host-built acc (which now omits T_Loo). The remaining cross-pair terms
    // (T_ph1-3, T_tmp) follow in later sub-stages. Off by default → byte-unchanged.
    bool use_gpu_xpair_ = false;
    real_t* d_acc_all_   = nullptr;      ///< [nocc·nvir²] full acc stack (host part + GPU cross-pair)
    real_t* d_Loo_xpair_ = nullptr;      ///< [nocc²] M_Loo[j,l] = -Loo_lmo[l,j] (T_Loo mixing matrix)

    // B-a.6a GPU port — Stage 3b T_ph2 (env GANSU_DLPNO_NATIVE_GPU_PH2=1; requires
    // use_gpu_xpair_). T_ph2 (acc[a,b] -= Σ_{l,c} Wovov_lmo[l,a,j,c] r2c[l][c,b]) is
    // a per-(j,l) GEMM acc[j] += (-1)·A_j[l]·r2c[l] with A_j[l][a,c] = Wovov_lmo[l,a,j,c]
    // taken as a STRIDED submatrix view of d_Wovov_lmo_ (row-stride nocc·nvir, base
    // (l·nvir·nocc+j)·nvir). Host omits T_ph2 when on. Off by default.
    bool use_gpu_ph2_ = false;
    real_t* d_Wovov_lmo_ = nullptr;      ///< [nocc·nvir·nocc·nvir] borrowed ph intermediate (occ pos 0,2)

    // B-a.6a GPU port — Stage 3b T_ph3 (env GANSU_DLPNO_NATIVE_GPU_PH3=1; requires
    // use_gpu_xpair_). T_ph3 (acc[a,b] -= Σ_{l,c} Wovvo_lmo[l,b,c,j] r2c[l][c,a]) is
    // a per-(j,l) GEMM acc[j] += (-1)·R[l]ᵀ·B_j[l]ᵀ. Wovvo_lmo has j as its innermost
    // (stride-1) index, so the strided-view trick (T_ph2) does not apply; the ctor
    // pre-transposes it to Wovvo_re[l,j,b,c] (c stride-1) so B_j[l] is contiguous.
    bool use_gpu_ph3_ = false;
    real_t* d_Wovvo_re_ = nullptr;       ///< [nocc·nocc·nvir·nvir] Wovvo_re[l,j,b,c]=Wovvo_lmo[l,b,c,j]

    // B-a.6a GPU port — Stage 3b T_ph1 (env GANSU_DLPNO_NATIVE_GPU_PH1=1; requires
    // use_gpu_xpair_). T_ph1 (acc[a,b] += Σ_{l,d} (2 Wovvo_lmo[l,b,d,j] -
    // Wovov_lmo[l,b,j,d]) r2c[l][a,d]) is TWO GEMMs per (j,l) (no W̃ build):
    //   +2·r2c[l]·Wovvo_re_j[l]ᵀ  (contiguous, shares d_Wovvo_re_)
    //   -1·r2c[l]·Wovov_j[l]ᵀ     (strided submatrix of d_Wovov_lmo_, ldB=nocc·nvir)
    // both GEMM(T,N), β=1. Borrows d_Wovvo_re_ (ph3) + d_Wovov_lmo_ (ph2); the ctor
    // uploads either if ph1 is on regardless of ph2/ph3.
    bool use_gpu_ph1_ = false;

    // B-a.6a GPU port — Stage 3b T_tmp (env GANSU_DLPNO_NATIVE_GPU_TMP=1; requires
    // use_gpu_xpair_). The two-stage T_tmp term, both stages as a single GEMV:
    //   stage 1  tmp[K] = Σ_{l,c,d} ovov_Llmo[K,c,l,d]·r2c_sym[l](c,d)   (r2c_sym = 2R-Rᵀ,
    //            reordered to (c,l,d) by a kernel) → GEMV ovov_mat·r2c_sym_re
    //   stage 2  acc_all -= t2_Jlmo_mat·tmp                              → GEMV over the
    //            whole acc stack (t2_Jlmo is contiguous [nocc_K × (j,a,b)]).
    bool use_gpu_tmp_ = false;
    real_t* d_ovov_Llmo_  = nullptr;     ///< [nocc·nvir·nocc·nvir] ovov_Llmo (= [nocc_K × (c,l,d)])
    real_t* d_t2_Jlmo_    = nullptr;     ///< [nocc²·nvir²] t2_Jlmo (= [nocc_K × (j,a,b)])
    real_t* d_r2c_sym_re_ = nullptr;     ///< [nvir·nocc·nvir] (c,l,d)-ordered 2R-Rᵀ scratch
    real_t* d_tmp_        = nullptr;     ///< [nocc] tmp[K] scratch

    // B-a.6a GPU port — Stage 3c T_Lvv (env GANSU_DLPNO_NATIVE_GPU_TLVV=1; requires
    // use_gpu_xpair_). The pair-local T_Lvv (acc[j] += Lvv·r2c[j] + r2c[j]·Lvvᵀ) as
    // two per-j GEMMs into the device acc stack (r2c[j] resident, Lvv uploaded once).
    bool use_gpu_tlvv_ = false;
    real_t* d_Lvv_ = nullptr;            ///< [nvir²] Lvv (occ-free σ1/σ2 intermediate)

    // B-a.6a GPU port — Stage 3c T_r1 (env GANSU_DLPNO_NATIVE_GPU_TR1=1; requires
    // use_gpu_xpair_). The last host acc term T_r1 (acc[a,b] += Σ_c Wvvvo_lmo[a,b,c,j]
    // r1[c]) as a per-j GEMV. Wvvvo_lmo has j innermost (stride 1), so the ctor
    // pre-transposes it to Wvvvo_r1[j,a,b,c] (c stride 1) → M_j[(a,b),c] contiguous.
    // With T_r1 on device the acc is fully device-built (no host acc-build remains).
    bool use_gpu_tr1_ = false;
    real_t* d_Wvvvo_r1_ = nullptr;       ///< [nocc·nvir³] Wvvvo_r1[j,a,b,c]=Wvvvo_lmo[a,b,c,j]
    real_t* d_r1_ = nullptr;             ///< [nvir] r1 scratch (uploaded each matvec)

    // B-a.6a GPU port — Stage 4 σ1 (env GANSU_DLPNO_NATIVE_GPU_S1LVV / _S1FOV /
    // _S1WVOVV; each ⊂ the previous, all ⊂ use_gpu_xpair_ so d_r2c_all_ already
    // holds the device-resident lifted r2 — no re-lift). The 1p σ1 sector on
    // device:
    //   S1LVV  : σ1[a] += Σ_c Lvv[a,c] r1[c]                        (one GEMV, op_T)
    //   S1FOV  : σ1[a] += Σ_{l,d} Fov[l,d] (2 r2c[l][a][d] - r2c[l][d][a])
    //            (per-l GEMV pair: r2c[l]·Fov_l op_T (α=2) + op_N (α=-1))
    //   S1WVOVV: σ1[a] += Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2c[l][c][d]
    //            = Σ_{lcd} Wvovv[a,l,c,d] R_sym[l,c,d], R_sym = 2r2c[l](c,d)-r2c[l](d,c)
    //            (a kernel builds R_sym (l,c,d)-ordered, then ONE GEMV op_T over
    //            the dominant O(nvir⁴nocc) term — the biggest host σ1 cost).
    // Host compute_sigma1 omits the enabled terms; off by default → byte-unchanged.
    // A separate σ1 self-check (GPU-assisted vs full host) gates it ≤1e-11.
    bool use_gpu_s1lvv_   = false;
    bool use_gpu_s1fov_   = false;
    bool use_gpu_s1wvovv_ = false;
    real_t* d_Fov_         = nullptr;    ///< [nocc·nvir] σ1 Fov
    real_t* d_Wvovv_       = nullptr;    ///< [nvir·nocc·nvir·nvir] σ1 Wvovv
    real_t* d_sigma1_      = nullptr;    ///< [nvir] σ1 device accumulator
    real_t* d_r2c_sym_lcd_ = nullptr;    ///< [nocc·nvir·nvir] (l,c,d)-ordered 2R-Rᵀ for Wvovv·r2

    // B-a.6a GPU port — Stage 4 full residency (env GANSU_DLPNO_NATIVE_GPU_RESIDENT=1;
    // requires the COMPLETE GPU term set — every σ2 acc term + the 3 σ1 terms — so
    // the host builds nothing). When on, apply() routes to apply_resident: the packed
    // r1/r2 are read straight from d_input (no input D2H/H2D, no d_r2_pack_ copy), the
    // lifted r2c / acc / σ1 / σ2 stay device-resident (no intermediate D2H), and the
    // output is assembled on device (two D2D copies). NUMERICS-PRESERVING — same math
    // as the validated host-assisted path. Off by default → the existing apply() path
    // (compute_sigma1 + branch + h_out round-trip) is byte-unchanged.
    bool use_gpu_resident_ = false;
    mutable bool resident_ = false;            ///< set true only inside apply_resident
    mutable const real_t* d_r2_src_ = nullptr; ///< resident packed-r2 source (= d_input+nvir)
    mutable const real_t* d_r1_src_ = nullptr; ///< resident r1 source (= d_input)

    /// Stage 1 GPU T_vvvv: packed_sigma2 += Σ_{c'd'} Wvvvv^(jj)[a'b'c'd'] r2_packed^(jj)[c'd']
    /// on device (per-occ cublasDgemv). Exactly mirrors the host PNO-space block.
    void apply_tvvvv_gpu(const std::vector<real_t>& packed_r2,
                         std::vector<real_t>& packed_sigma2) const;

    /// Stage 3a GPU source lift: r2c_all[l·nvir²] = U^(ll)·r2_packed^(ll)·U^(ll)ᵀ
    /// (per-occ chained cublasDgemm), left in d_r2c_all_ and copied to r2c_all.
    void lift_r2c_gpu(const std::vector<real_t>& packed_r2,
                      std::vector<real_t>& r2c_all) const;

    /// Stage 2/3b shared: per-occ projection σ2_packed^(jj) = U^(jj)ᵀ acc[j] U^(jj)
    /// + T_vvvv, reading acc[j] from a DEVICE stack (d_acc_stack + j·nvir²).
    void project_acc_stack_gpu(const real_t* d_acc_stack,
                               const std::vector<real_t>& packed_r2,
                               std::vector<real_t>& packed_sigma2) const;

    /// Stage 3b: upload host acc (GPU terms omitted) to d_acc_all_, add the device
    /// terms (T_Loo, T_Lvv, T_r1, T_ph1-3, T_tmp as enabled), then project. @param r1
    /// is needed by T_r1 (the device 1p→2p1h coupling).
    void apply_xpair_projection_gpu(const std::vector<real_t>& r1,
                                    const std::vector<real_t>& acc_all,
                                    const std::vector<real_t>& packed_r2,
                                    std::vector<real_t>& packed_sigma2) const;

    /// Stage 3b T_ph2: d_acc_all_[j] -= Σ_l A_j[l]·r2c[l], A_j[l][a,c]=Wovov_lmo[l,a,j,c]
    /// (strided submatrix), accumulated per (j,l) GEMM into the device acc stack.
    void add_tph2_gpu() const;

    /// Stage 3b T_ph3: d_acc_all_[j] -= Σ_l R[l]ᵀ·B_j[l]ᵀ, B_j[l][b,c]=Wovvo_re[l,j,b,c]
    /// (contiguous), per (j,l) GEMM(T,T) accumulated into the device acc stack.
    void add_tph3_gpu() const;

    /// Stage 3b T_ph1: d_acc_all_[j] += Σ_l (2 r2c[l]·Wovvo_re_j[l]ᵀ - r2c[l]·Wovov_j[l]ᵀ),
    /// two per-(j,l) GEMM(T,N) accumulated into the device acc stack.
    void add_tph1_gpu() const;

    /// Stage 3b T_tmp: kernel builds r2c_sym_re, GEMV tmp = ovov_mat·r2c_sym_re,
    /// then GEMV d_acc_all_ -= t2_Jlmo_mat·tmp (whole acc stack).
    void add_ttmp_gpu() const;

    /// Stage 3c T_Lvv: d_acc_all_[j] += Lvv·r2c[j] + r2c[j]·Lvvᵀ, two per-j GEMMs.
    void add_tlvv_gpu() const;

    /// Stage 3c T_r1: d_acc_all_[j] += M_j·r1, M_j[(a,b),c]=Wvvvo_r1[j,a,b,c] (per-j GEMV).
    void add_tr1_gpu(const std::vector<real_t>& r1) const;

    /// Stage 4 σ1: add the enabled GPU σ1 terms (Lvv·r1 / Fov·r2 / Wvovv·r2) into
    /// sigma1, reading the device-resident lifted r2 (d_r2c_all_, filled by
    /// lift_r2c_gpu). @param r1 is uploaded to d_r1_ for the Lvv·r1 GEMV.
    void add_sigma1_gpu(const std::vector<real_t>& r1, std::vector<real_t>& sigma1) const;

    /// Stage 2 GPU projection: per output occ j, σ2_packed^(jj) = U^(jj)ᵀ acc[j] U^(jj)
    /// (chained cublasDgemm) + Stage-1 T_vvvv accumulated, into packed_sigma2.
    /// @param acc_all  [nocc·nvir²] host-built acc, block j at j·nvir² (row-major nvir×nvir).
    void apply_projection_gpu(const std::vector<real_t>& acc_all,
                              const std::vector<real_t>& packed_r2,
                              std::vector<real_t>& packed_sigma2) const;

    real_t* d_diagonal_ = nullptr;     ///< [total_dim] Koopmans/PNO diagonal (preconditioner / σ2 B-EA.1)
    std::vector<real_t> h_diagonal_;   ///< host mirror

    /// σ1[a] (1p): lift packed r2 → canonical, apply the canonical σ1 formula.
    /// B-a.6a Stage 4: @param skip_lvv / skip_fov / skip_wvovv omit the matching
    /// term from the host build (the GPU adds it via add_sigma1_gpu). When all
    /// three are skipped the host lift is skipped too (sigma1 = 0, GPU fills it).
    void compute_sigma1(const std::vector<real_t>& r1,
                        const std::vector<real_t>& packed_r2,
                        std::vector<real_t>& sigma1,
                        bool skip_lvv = false,
                        bool skip_fov = false,
                        bool skip_wvovv = false) const;

    /// σ2 (2p1h), native per-pair, into the packed buffer (length total_dim-nvir).
    /// B-EA.2: pair-local T_Lvv (a,b) + T_r1 (Wvvvo·r1).
    /// @param skip_tvvvv  B-a.6a Stage 1: when true, omit the dressed PNO-space
    ///        T_vvvv block (the GPU path computes it via apply_tvvvv_gpu instead).
    /// @param acc_export  B-a.6a Stage 2: when non-null, the per-occ acc[j]
    ///        (canonical nvir×nvir, row-major) is stored at acc_export[j·nvir²]
    ///        and the host projection + dressed T_vvvv + packed_sigma2 write are
    ///        skipped (the device does those via apply_projection_gpu).
    /// @param r2c_external  B-a.6a Stage 3a: when non-null, the lifted source
    ///        amplitudes r2c[l] (canonical nvir×nvir at r2c_external[l·nvir²]) are
    ///        taken from this buffer (GPU lift) instead of being built on host.
    /// @param skip_loo  B-a.6a Stage 3b: when true, omit the cross-pair T_Loo term
    ///        from acc (the device adds it via the stacked M_Loo·R_stack GEMM).
    /// @param skip_ph2  B-a.6a Stage 3b: when true, omit the cross-pair T_ph2 term
    ///        from acc (the device adds it via per-(j,l) strided GEMMs).
    /// @param skip_ph3  B-a.6a Stage 3b: when true, omit the cross-pair T_ph3 term
    ///        from acc (the device adds it via per-(j,l) GEMM(T,T)).
    /// @param skip_ph1  B-a.6a Stage 3b: when true, omit the cross-pair T_ph1 term
    ///        from acc (the device adds it via two per-(j,l) GEMM(T,N)).
    /// @param skip_tmp  B-a.6a Stage 3b: when true, omit the T_tmp term (both the
    ///        tmp[K] pre-stage and the t2 subtraction; the device adds it).
    /// @param skip_tlvv  B-a.6a Stage 3c: when true, omit the pair-local T_Lvv term
    ///        (acc starts at zero; the device adds Lvv·r2c[j] + r2c[j]·Lvvᵀ).
    /// @param skip_tr1  B-a.6a Stage 3c: when true, omit the T_r1 term (the device
    ///        adds Σ_c Wvvvo_lmo[a,b,c,j] r1[c]).
    void compute_sigma2(const std::vector<real_t>& r1,
                        const std::vector<real_t>& packed_r2,
                        std::vector<real_t>& packed_sigma2,
                        bool skip_tvvvv = false,
                        std::vector<real_t>* acc_export = nullptr,
                        const std::vector<real_t>* r2c_external = nullptr,
                        bool skip_loo = false,
                        bool skip_ph2 = false,
                        bool skip_ph3 = false,
                        bool skip_ph1 = false,
                        bool skip_tmp = false,
                        bool skip_tlvv = false,
                        bool skip_tr1 = false) const;
};

} // namespace gansu
