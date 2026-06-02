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
 * @file dlpno_ip_eom_native_operator.hpp
 * @brief Native per-pair DLPNO-IP-EOM-CCSD σ operator (bt-PNO-STEOM stage B
 *        (a), Dutta-Saitow-Riplinger-Neese-Izsák 2018 JCP 148, 244101).
 *
 * Unlike the project-up reference (DLPNOIPEOMProjectedOperator), which forms
 * the *full* canonical σ each matvec (non-scaling), this operator computes σ
 * directly, term by term, so the expensive σ2 contractions never touch the
 * canonical nao⁴ / N⁵⁻⁶ surface. The (b) reference is the bit-exact gate at no
 * truncation (n_pno = nvir → projection is a bijection → P·σ_canon·P = σ_canon
 * exactly, so the two operators must agree to ~1e-10, term by term).
 *
 * Build-up (each sub-phase gated bit-exact vs the (b) reference at no trunc):
 *   B-a.1: σ1 (1h sector) via lift + canonical formula on borrowed (bit-
 *          identical) intermediates; σ2 = 0 (was a diagonal placeholder).
 *   B-a.2a: + σ2 term T2 (Lvv^(ij)·r2), the first GENUINE native per-pair
 *          term — pure pair-local, no barS / no canonical-r2 lift:
 *            σ2_packed^(ij) += U^(ij)ᵀ · Lvv · (U^(ij) · r2_packed^(ij))
 *          (U^(ij) = C_virᵀ S bar_Q_ij). Both orientations use the pair's own
 *          packed r2. U_loc-independent (Lvv is occ-free + U_loc orthogonal).
 *   B-a.2b (this commit): + σ2 term T1 (Wovoo·r1) — the first term with an
 *          OCCUPIED U_loc rotation:
 *            σ2_packed^(ij)[a'] += -Σ_k r1[k] (U^(ij)ᵀ Wovoo_lmo[:,:,i,j])[a',k]
 *          where Wovoo_lmo[k,a,i,j] = Σ_IJ U_loc[I,i] U_loc[J,j] Wovoo[k,a,I,J]
 *          (occ I,J rotated canonical→LMO; identity for localizer none). The
 *          virtual index a is projected to the pair PNO by U^(ij).
 *   B-a.3a (this commit): + cross-pair scalar T3/T4 (Loo·r2) — the first terms
 *          needing the single-index barS projection of a SOURCE pair's packed
 *          amplitude into the TARGET pair PNO. barS^(ij,kj) = U^(ij)ᵀ U^(kj)
 *          (exact, since bar_Q = C_vir U and C_virᵀ S C_vir = I), so the whole
 *          σ2 for an orientation factors through a canonical-virtual accumulator:
 *            acc = Lvv·(U·r2) + w_T1
 *                  - Σ_k Loo_lmo[k,oi]·(U_{src(k,oj)}·r2_{src})       (T3)
 *                  - Σ_l Loo_lmo[l,oj]·(U_{src(oi,l)}·r2_{src})       (T4)
 *            σ2_packed^(orient) += U^(ij)ᵀ · acc
 *          Source orientation: off_ij if setups[idx_s].i == first-occ else off_ji
 *          (mirrors dlpno_pair_data.cu:399-435). Loo_lmo = U_locᵀ Loo U_loc.
 *   B-a.3b (this commit): + cross-pair T5 (Woooo·r2):
 *            acc += Σ_{k,l} Woooo_lmo[k,l,oi,oj] · (U_{src(k,l)}·r2_{src(k,l)})
 *          Woooo_lmo = the full 4-occ-index U_loc rotation of canonical Woooo
 *          (identity for localizer none). All source lifts U_{src}·r2_{src} are
 *          precomputed once per matvec into r2c_all[p,q].
 *   B-a.3c (this commit): + T8 (the global tmp_c reduction + tmp·t2):
 *            tmp_c[c] = Σ_{k,l,d} (2 Woovv[l,k,d,c] - Woovv[k,l,d,c]) r2[k,l,d]
 *                       (one [nvir] vector per matvec; basis-invariant)
 *            acc[a] += -Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]
 *          Woovv_lmo / t2_lmo carry the 2-occ-index U_loc rotation (copy for
 *          localizer none). t2 is the borrowed CCSD T2 (= canonical σ2 kernel's).
 *   B-a.4 (this commit): + ph-ladder T6/T7 (Wovvo/Wovov) — the final σ2 terms:
 *            T6:  acc[a] += Σ_{m,d} Wovvo_lmo[m,a,d,oj]·(2·r2c(oi,m)[d] - r2c(m,oi)[d])
 *            T7:  acc[a] += -Σ_{m,d} Wovov_lmo[m,a,oj,d]·r2c(oi,m)[d]
 *                          -Σ_{m,d} Wovov_lmo[m,a,oi,d]·r2c(m,oj)[d]
 *          Wovvo/Wovov are the dense ALREADY-T2-DRESSED canonical intermediates
 *          borrowed from ip_op; _lmo carries the 2-occ-index U_loc rotation
 *          (copy for none). (The per-pair-PNO dressed Wovvo^(ij) is the B-a.6
 *          scale optimisation, not needed for the gate.) After this commit the
 *          native σ equals the (b) reference's full σ directly.
 *   B-a.4: + ph-ladder (T6/T7).  Then full σ == (b) reference directly.
 *
 * Until all σ2 terms land the operator is INCOMPLETE (a partial σ2), so it is
 * NOT wired into the production driver — only GANSU_DLPNO_IP_NATIVE_VALIDATE
 * exercises it, comparing the implemented σ rows against an in-gate reference.
 *
 * apply() round-trips through the host (Eigen-free / Eigen loops); acceptable
 * at the gate scale, matching the (b) reference. The preconditioner uses the
 * Koopmans / PNO diagonal (B0.2).
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "dlpno_mp2.hpp"            // DLPNOLMP2Result
#include "dlpno_ip_packing.hpp"     // DLPNOIPPacking
#include "ip_eom_ccsd_operator.hpp" // IPEOMCCSDOperator (borrowed intermediates)
#include "dlpno_eom_dressed_pno.hpp" // DressedPnoIP (B-a.6c true-scaling per-pair W)

namespace gansu {

class DLPNOIPEOMNativeOperator : public LinearOperator {
public:
    /**
     * @param ip_op     Canonical IP-EOM operator (built from the same DLPNO
     *                  back-transformed T1/T2) whose dressed intermediates this
     *                  operator borrows (bit-identical, no re-derivation). Must
     *                  outlive this operator. Must have been constructed with a
     *                  non-null MO ERI (so build_dressed_intermediates ran).
     * @param res       Converged DLPNO result (per-pair bar_Q/Lambda/setups,
     *                  phase24 for the cross-pair σ2 terms). Ref held.
     * @param packing   Packed-vector offset table. Ref held.
     * @param U_loc     [nocc²] localization rotation (copied).
     * @param C_vir     [nao·nvir] canonical virtual coefficients (copied).
     * @param h_S       [nao²] AO overlap (copied).
     * @param nao,nvir  dimensions.
     * @param eps_o     [nocc] active-occupied energies (1h preconditioner diagonal).
     */
    DLPNOIPEOMNativeOperator(const IPEOMCCSDOperator& ip_op,
                             const DLPNOLMP2Result& res,
                             const DLPNOIPPacking& packing,
                             const std::vector<real_t>& U_loc,
                             const std::vector<real_t>& C_vir,
                             const real_t* h_S,
                             int nao, int nvir,
                             const std::vector<real_t>& eps_o,
                             int num_gpus = 1);

    ~DLPNOIPEOMNativeOperator();

    DLPNOIPEOMNativeOperator(const DLPNOIPEOMNativeOperator&) = delete;
    DLPNOIPEOMNativeOperator& operator=(const DLPNOIPEOMNativeOperator&) = delete;

    void apply(const real_t* d_input, real_t* d_output) const override;

    /// Stage 4 full residency: device-only matvec reading r1/r2 from d_input and
    /// assembling σ into d_output on device (no host round-trip). Used when
    /// use_gpu_resident_ (the complete GPU term set + GANSU_DLPNO_NATIVE_GPU_RESIDENT).
    void apply_resident(const real_t* d_input, real_t* d_output) const;

    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "DLPNOIPEOMNativeOperator"; }

private:
    const DLPNOLMP2Result& res_;
    const DLPNOIPPacking& packing_;
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
    // (d_U_pack_) plus a broadcast input copy and a d_r2c_all_ scratch on every
    // d>0, so each device can redundantly lift its own copy of the source
    // amplitudes (residency decision A) from a cudaMemcpyPeer broadcast of d_input
    // (decision B). In 5b the σ1/σ2 build and the assembled output stay 100% on
    // device 0 (production byte-identical); the d>0 lifts run only as a validation
    // of the broadcast + per-device residency (each device's d_r2c_all_ vs device
    // 0, ≤1e-11) under GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE. The slab σ2 split +
    // peer gather land in Stage 5c. Mirrors the EA operator.
    bool use_gpu_multi_ = false;       ///< multi-GPU σ2 (broadcast + per-device build + gather)
    bool multi_selfcheck_ = false;     ///< gathered σ vs full device-0 host reference (first matvecs)
    // The 5c self-check recomputes the full host σ each matvec — fine to prove the path,
    // but ruinous as a per-matvec cost (the host path is exactly what multi-GPU avoids at
    // scale). Cap it at the first few matvecs so GANSU_DLPNO_NATIVE_GPU_MULTI_VALIDATE
    // stays usable for timing on large systems.
    mutable int multi_check_done_ = 0;
    static constexpr int kMultiCheckMax = 3;
    // Stage 5c: each device d holds a FULL replica of every σ2 device buffer (constants
    // peer-copied from device 0, scratch freshly allocated) plus a cublas handle, so the
    // validated σ2 helper chain can run unchanged on device d after bind_device(d) points
    // the (rebound) members at this device's buffers. ws_[0] aliases the device-0 members
    // so bind_device(0) is an exact restore. d_input is the broadcast input copy; d_r2_src
    // = d_input+nocc on each device.
    struct DeviceWorkspace {
        int     device    = -1;        ///< CUDA device id (ws_[0].device = 0, aliases members)
        void*   cublas     = nullptr;  ///< cublasHandle_t for this device (device 0 = cublas_)
        real_t* d_input    = nullptr;  ///< [total_dim] broadcast copy of the matvec input (d>0)
        real_t* d_r2c_all  = nullptr;  ///< [nocc²·nvir] per-device lifted source r2c
        real_t* d_acc_all  = nullptr;  ///< [n_orient·nvir] acc scratch
        real_t* d_sig_pack = nullptr;  ///< [total_dim-nocc] σ2 output scratch
        real_t* d_U_pack   = nullptr;  ///< [u_pno_off_.back()] U^(ij) replica
        real_t* d_Lvv_pno_pack = nullptr;
        real_t* d_Loo_lmo  = nullptr;
        real_t* d_Woooo_lmo= nullptr;
        real_t* d_Wovoo_slot = nullptr; ///< 5g: slot-packed Wovoo_re (slab subrange in slab mode)
        real_t* d_Woovv_lmo= nullptr;
        real_t* d_t2_slot  = nullptr;  ///< 5f: slot-packed t2 (slab subrange in slab mode)
        real_t* d_S        = nullptr;  ///< [nocc²·nvir] T8a scratch
        real_t* d_tmp_c    = nullptr;  ///< [nvir] T8 scratch
        real_t* d_Wovvo_occi = nullptr;
        real_t* d_Wovvo_occj = nullptr;
        real_t* d_Wovov_occi = nullptr;
        real_t* d_Wovov_occj = nullptr;
        real_t* d_RP_oim   = nullptr;  ///< [nocc·max_n_pno] ph-ladder scratch
        real_t* d_RP_moi   = nullptr;
        real_t* d_RP_moj   = nullptr;
        size_t  lvv_shift  = 0;        ///< 5e: subtract from lvv_pno_off_[idx] for slab-only Lvv pack (0=full)
        size_t  wovvo_shift= 0;        ///< 5e: subtract from wovvo_off_[idx] for slab-only ph-ladder packs (0=full)
        size_t  t2_shift   = 0;        ///< 5f: subtract from slot·nvir² for slab-only slot-packed t2 (0=full)
        size_t  wovoo_t1_shift = 0;    ///< 5g: subtract from slot·nvir·nocc for slab-only slot-packed Wovoo (0=full)
    };
    std::vector<DeviceWorkspace> ws_;  ///< size = #devices used (ws_[0] aliases device 0)
    std::vector<int> slot_begin_;      ///< [#dev] orientation-slot slab start per device
    std::vector<int> slot_end_;        ///< [#dev] orientation-slot slab end (exclusive) per device
    // Stage 5c-step2 (compute split): when use_gpu_multi_slab_, each device computes σ2
    // ONLY for its orientation-slot slab (no redundant full build). slab_active_ +
    // cur_slot_* are set per device in apply_resident; the σ2 helpers read them (full
    // [0,n_orient_) when slab_active_ is false → single-GPU + step1 byte-unchanged). The
    // global T8a tmp_c reduction stays full on every device. Default ON for the multi
    // path; GANSU_DLPNO_NATIVE_GPU_MULTI_NOSLAB=1 forces the step1 redundant build.
    bool use_gpu_multi_slab_ = false;
    mutable bool slab_active_ = false;
    mutable int  cur_slot_begin_ = 0;
    mutable int  cur_slot_end_   = 0;
    // 5e: output-indexed pack slab-only. d>0 in slab mode hold only their slab's pairs'
    // Lvv_pno (lvv_pno_off_) + ph-ladder (wovvo_off_) packs; the helpers index them as
    // d_..._ + (off[idx] - shift). 0 on device 0 / full / single-GPU. Set in bind_device.
    mutable size_t lvv_pack_shift_   = 0;
    mutable size_t wovvo_pack_shift_ = 0;
    // 5f: cross-pair t2 stored slot-packed (one nvir² block per orientation slot, in slot
    // order) instead of the full occ-pair nocc²·nvir² dense tensor. add_t8_gpu indexes it
    // by slot; d>0 slab replicas hold only their contiguous slot subrange and subtract this
    // shift (= slot_begin·nvir²). 0 on device 0 / full / single-GPU. Set in bind_device.
    mutable size_t t2_slot_shift_    = 0;
    // 5g: cross-pair Wovoo_re stored slot-packed (one nvir·nocc block per orientation slot,
    // in slot order) instead of the full occ-pair nocc²·nvir·nocc dense tensor. add_t1_gpu
    // indexes it by slot; d>0 slab replicas keep only their contiguous slot subrange and
    // subtract this shift (= slot_begin·nvir·nocc). 0 on device 0 / full / single-GPU.
    mutable size_t wovoo_slot_shift_ = 0;
    /// Stage 5c: point the (const_cast) σ2 members at device d's workspace buffers +
    /// cublas handle (NULL stream) so the validated helper chain runs on device d.
    /// bind_device(0) restores the device-0 members. resident_/d_r2_src_/d_r1_src_ are
    /// set for the resident multi path.
    void bind_device(int d) const;
    /// Stage 5b legacy: per-device lift residency check (kept for the num_gpus=1 byte
    /// path / standalone validation; superseded by the 5c gathered-σ self-check).
    void lift_r2c_multi_validate(const real_t* d_input) const;

    // Borrowed canonical intermediates (host copies; bit-identical to ip_op).
    std::vector<real_t> h_Loo_;    // [nocc²]              σ1
    std::vector<real_t> h_Fov_;    // [nocc·nvir]          σ1
    std::vector<real_t> h_Wooov_;  // [nocc²·nocc·nvir]    σ1
    std::vector<real_t> h_Lvv_;    // [nvir²]              σ2 T2
    std::vector<real_t> h_Wovoo_lmo_;  // [nocc·nvir·nocc²]  σ2 T1 (occ I,J rotated → LMO i,j)
    std::vector<real_t> h_Loo_lmo_;    // [nocc²]            σ2 T3/T4 (Loo rotated → LMO: U_locᵀ Loo U_loc)
    std::vector<real_t> h_Woooo_lmo_;  // [nocc⁴]            σ2 T5 (Woooo 4-occ-index rotated → LMO)
    std::vector<real_t> h_Woovv_lmo_;  // [nocc²·nvir²]      σ2 T8a (eri_oovv, 2-occ-index rotated → LMO)
    std::vector<real_t> h_t2_lmo_;     // [nocc²·nvir²]      σ2 T8b (CCSD T2, 2-occ-index rotated → LMO)
    std::vector<real_t> h_Wovvo_lmo_;  // [nocc·nvir·nvir·nocc]  σ2 T6 (occ pos 0,3 rotated → LMO)
    std::vector<real_t> h_Wovov_lmo_;  // [nocc·nvir·nocc·nvir]  σ2 T7 (occ pos 0,2 rotated → LMO)
    std::vector<real_t> h_ovov_lmo_;   // [nocc·nvir·nocc·nvir]  raw (ov|ov), occ pos 0,2 → LMO (B-a.6c(b2) ring)

    // B-a.6c true-scaling path (env GANSU_DLPNO_NATIVE_DRESSED=1). When enabled,
    // per-pair PNO-basis intermediates replace the dense-borrow contractions term
    // by term, so σ2 never touches the canonical nvir surface. Default off → the
    // dense-borrow path above is used and every B-a.0..B-a.5 gate stays bit-exact.
    bool use_dressed_pno_ = false;
    // B-a.6c(b2) sub-flag (env GANSU_DLPNO_NATIVE_RING=1, only meaningful when
    // use_dressed_pno_): replace the ring part of the congruence-seeded ph-ladder
    // W with a native Phase24 + two-sided-barS build (true scaling). Off → b1
    // congruence seed (bit-exact vs the dense ph-ladder at full PNO).
    bool use_native_ring_ = false;
    // IP dense-free sub-flag (env GANSU_DLPNO_NATIVE_BARE=1, only meaningful when
    // use_native_ring_): seed the ph-ladder Wovvo_pno/Wovov_pno from Phase24 bare
    // blocks (W_ovvo_bare/W_oovv_bare) + native-only ring, so the dense nocc²·nvir²
    // Wovvo/Wovov is NEVER borrowed/materialised (true scaling). Off → dense-borrow
    // congruence seed + dense-ring subtraction (b2). EA B-EA.6e analog.
    bool use_native_bare_ = false;
    DressedPnoIP dressed_;             ///< per-pair PNO intermediates (built in ctor when enabled)
    // U^(ij) = C_virᵀ S bar_Q_ij [nvir × n_pno], flat row-major per pair. Built
    // once in the ctor (fixed across matvecs); used by the dressed build and the
    // dressed-path projection. Empty when the dressed path is off.
    std::vector<std::vector<real_t>> Uall_;

    // B-a.6a GPU port — Stage 1 (env GANSU_DLPNO_NATIVE_GPU=1; requires
    // use_dressed_pno_). The pair-local PNO-space σ2 T2 term (Lvv^(ij)·r2_packed,
    // a 1-virtual [n×n]·[n] GEMV per orientation) on device. Both orientations of
    // a pair share Lvv_pno[idx] (Lvv is occ-free). Off by default → host T2 path
    // byte-unchanged. GANSU_DLPNO_NATIVE_GPU_VALIDATE adds an in-process GPU-vs-host
    // σ2 diff each matvec (≤1e-11). Mirrors the EA T_vvvv port (B-a.6a).
    bool use_gpu_ = false;
    bool gpu_selfcheck_ = false;
    void* cublas_ = nullptr;            ///< cublasHandle_t (opaque; keeps cublas out of this header)
    real_t* d_Lvv_pno_pack_ = nullptr;  ///< packed per-pair Lvv^(ij), n_pno² back-to-back
    std::vector<size_t> lvv_pno_off_;   ///< [n_pairs+1] prefix offsets (n²) into d_Lvv_pno_pack_
    real_t* d_r2_pack_  = nullptr;      ///< [total_dim-nocc] device scratch (T2 GEMV input)
    real_t* d_sig_pack_ = nullptr;      ///< [total_dim-nocc] device scratch (T2 GEMV output)

    // B-a.6a GPU port — Stage 2 (env GANSU_DLPNO_NATIVE_GPU_PROJ=1; requires
    // use_gpu_). The per-orientation single-sided PNO projection
    //   σ2_packed^(orient)[a'] = Σ_a U^(ij)[a,a'] acc[a]   (= U^(ij)ᵀ acc, one GEMV)
    // is moved to device; the validated Stage-1 T2 (Lvv^(ij)·r2_packed) is then
    // accumulated into the same packed σ2 block. The host still builds the canonical
    // -virtual acc (T1/T3/T4/T5/T8b) and exports it per orientation, and still adds
    // the PNO-space ph-ladder T6/T7 directly to packed_sigma2 (those move to device
    // in a later sub-stage). Off by default → the host projection path is byte-
    // unchanged. Mirrors the EA Stage-2 projection port.
    bool use_gpu_proj_ = false;
    real_t* d_U_pack_   = nullptr;      ///< packed per-pair U^(ij) [nvir·n_pno] row-major back-to-back
    std::vector<size_t> u_pno_off_;     ///< [n_pairs+1] prefix offsets (nvir·n) into d_U_pack_
    real_t* d_acc_all_  = nullptr;      ///< [n_orient·nvir] host-exported acc, one nvir vector per orientation slot
    int n_orient_ = 0;                  ///< total orientation slots (Σ pairs n_pno>0: diagonal?1:2)

    // B-a.6a GPU port — Stage 3a (env GANSU_DLPNO_NATIVE_GPU_LIFT=1; requires
    // use_gpu_proj_). The per-matvec source lift r2c[p,q] = U_{src(p,q)}·r2_packed_src
    // (one GEMV per occupied pair (p,q): U row-major [nvir×ns] → OP_T gives U·r2s) is
    // moved to device, leaving the lifted r2c device-resident (d_r2c_all_, block (p,q)
    // at (p·nocc+q)·nvir) for later cross-pair sub-stages. For now the lifted r2c is
    // also pulled back to host so the existing host acc-build consumes it (validated
    // end-to-end by the σ2 self-check); the round-trip is removed once the cross-pair
    // contractions move to device. Off by default → the host lift path is byte-unchanged.
    bool use_gpu_lift_ = false;
    real_t* d_r2c_all_ = nullptr;       ///< [nocc²·nvir] device-resident lifted source r2c (block (p,q) at (p·nocc+q)·nvir)

    // B-a.6a GPU port — Stage 3b cross-pair T3/T4/T5 (env GANSU_DLPNO_NATIVE_GPU_XPAIR=1;
    // requires use_gpu_lift_ so d_r2c_all_ holds the lifted source amplitudes). The
    // three occ-weighted r2c-sum terms are added to the device acc (per orientation
    // slot s = (oi,oj)) as strided GEMVs, all OP_N over a column-major view of d_r2c_all_:
    //   T3: acc[s] -= Σ_k Loo_lmo[k,oi]·r2c(k,oj)   (r2c(k,oj): base oj·nvir, ld nocc·nvir)
    //   T4: acc[s] -= Σ_l Loo_lmo[l,oj]·r2c(oi,l)   (r2c(oi,l): base oi·nocc·nvir, ld nvir)
    //   T5: acc[s] += Σ_kl Woooo_lmo[k,l,oi,oj]·r2c(k,l)  (full nocc² stack, ld nvir)
    // The host omits T3/T4/T5 when on. Off by default → byte-unchanged.
    bool use_gpu_xpair_ = false;
    real_t* d_Loo_lmo_   = nullptr;     ///< [nocc²] Loo_lmo (T3/T4 occ weight)
    real_t* d_Woooo_lmo_ = nullptr;     ///< [nocc⁴] Woooo_lmo (T5 occ-pair weight)
    std::vector<int> orient_oi_;        ///< [n_orient] per-slot first occ (matches compute_sigma2 order)
    std::vector<int> orient_oj_;        ///< [n_orient] per-slot second occ

    // B-a.6a GPU port — Stage 3b T1 (env GANSU_DLPNO_NATIVE_GPU_T1=1; ⊂ use_gpu_xpair_).
    // T1: acc[s][a] -= Σ_k Wovoo_lmo[k,a,oi,oj] r1[k] (1p→2h1p coupling). Wovoo_lmo has
    // (oi,oj) innermost (stride nocc²), so the ctor pre-transposes it to Wovoo_re[oi,oj,a,k]
    // (contiguous [nvir×nocc] per slot) → per-slot GEMV OP_T over r1.
    bool use_gpu_t1_ = false;
    real_t* d_Wovoo_slot_ = nullptr;    ///< [n_orient·nvir·nocc] Wovoo_re slot-packed (T1; 5g)
    real_t* d_r1_ = nullptr;            ///< [nocc] r1 upload scratch

    // B-a.6a GPU port — Stage 3b T8 (env GANSU_DLPNO_NATIVE_GPU_T8=1; ⊂ use_gpu_xpair_).
    // T8a: tmp_c[c] = Σ_kld Woovv_lmo[k,l,d,c] (2 r2c(l,k)[d] - r2c(k,l)[d])  (global [nvir],
    //      via a kernel building S[k,l,d]=2r2c(l,k)-r2c(k,l) + one GEMV over (k,l,d)).
    // T8b: acc[s][a] -= Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]  (per-slot GEMV, t2 block contiguous).
    bool use_gpu_t8_ = false;
    real_t* d_Woovv_lmo_ = nullptr;     ///< [nocc²·nvir²] Woovv_lmo (T8a)
    real_t* d_t2_slot_   = nullptr;     ///< [n_orient·nvir²] CCSD T2 slot-packed (T8b; 5f)
    real_t* d_S_         = nullptr;     ///< [nocc²·nvir] symmetrized r2c scratch (T8a)
    real_t* d_tmp_c_     = nullptr;     ///< [nvir] tmp_c scratch (T8)

    // B-a.6a GPU port — Stage 3b T6/T7 ph-ladder (env GANSU_DLPNO_NATIVE_GPU_PHL=1;
    // ⊂ use_gpu_xpair_). The final σ2 terms, added DIRECTLY to the PNO-space output
    // s (not the canonical acc): per orientation (oi,oj) with PNO dim n and (i,j)=
    // setups[idx], the one-sided barS source projections rp_oim[m]=U^(ij)ᵀ·r2c(oi,m),
    // rp_moi[m]=U^(ij)ᵀ·r2c(m,oi), rp_moj[m]=U^(ij)ᵀ·r2c(m,oj) are built as 3 GEMMs
    // (RP[nocc×n]=R·U; R_oi contiguous, R_moi/R_moj strided), then per m four GEMVs
    // accumulate onto d_sig_pack_+off:
    //   +2·W6[m]·rp_oim  -W6[m]·rp_moi  -W7oj[m]·rp_oim  -W7oi[m]·rp_moj
    // W role select: W6=Wovvo_pno_occ{j if oj==j else i}, W7oj=Wovov_pno_occ{j if oj==j
    // else i}, W7oi=Wovov_pno_occ{j if oi==j else i}. Host omits T6/T7 when on.
    bool use_gpu_phl_ = false;
    real_t* d_Wovvo_occi_ = nullptr;    ///< packed per-pair Wovvo_pno_occi [nocc·n²] back-to-back
    real_t* d_Wovvo_occj_ = nullptr;    ///< packed per-pair Wovvo_pno_occj
    real_t* d_Wovov_occi_ = nullptr;    ///< packed per-pair Wovov_pno_occi
    real_t* d_Wovov_occj_ = nullptr;    ///< packed per-pair Wovov_pno_occj
    std::vector<size_t> wovvo_off_;     ///< [n_pairs+1] prefix offsets (nocc·n²) into the W packs
    real_t* d_RP_oim_ = nullptr;        ///< [nocc·max_n] one-sided barS scratch (U^ᵀ·r2c(oi,m))
    real_t* d_RP_moi_ = nullptr;        ///< [nocc·max_n] (U^ᵀ·r2c(m,oi))
    real_t* d_RP_moj_ = nullptr;        ///< [nocc·max_n] (U^ᵀ·r2c(m,oj))
    std::vector<int> orient_idx_;       ///< [n_orient] per-slot pair idx
    std::vector<int> orient_off_;       ///< [n_orient] per-slot packed block start (off - nocc)
    int max_n_pno_ = 0;                 ///< max_idx n_pno[idx] (sizes d_RP_* scratch)

    // B-a.6a GPU port — Stage 4 σ1 (env GANSU_DLPNO_NATIVE_GPU_S1LOO / _S1FOV /
    // _S1WOOOV; each ⊂ use_gpu_xpair_ so d_r2c_all_ holds the lifted r2 — no re-lift).
    // The 1h σ1 sector on device (EA σ1 mirror):
    //   S1LOO  : σ1[i] -= Σ_k Loo[k,i] r1[k]                          (canonical Loo, OP_N GEMV)
    //   S1FOV  : σ1[i] += Σ_{l,d} Fov[l,d] (2 r2c(i,l)[d] - r2c(l,i)[d])
    //            (partA: 1 GEMV OP_T over (l,d); partB: per-l GEMV OP_T)
    //   S1WOOOV: σ1[i] += Σ_{kld} Wooov[k,l,i,d] (-2 r2c(k,l)[d] + r2c(l,k)[d])
    //            (kernel builds Ssym, Wooov pre-transposed to [i,k,l,d] → one GEMV OP_T)
    // Host compute_sigma1 omits the enabled terms; off by default → byte-unchanged.
    // A separate σ1 self-check (GPU-assisted vs full host) gates it ≤1e-11.
    bool use_gpu_s1loo_   = false;
    bool use_gpu_s1fov_   = false;
    bool use_gpu_s1wooov_ = false;
    real_t* d_Loo_canon_ = nullptr;     ///< [nocc²] canonical Loo (σ1 S1LOO; ≠ d_Loo_lmo_)
    real_t* d_Fov_       = nullptr;     ///< [nocc·nvir] σ1 Fov
    real_t* d_Wooov_re_  = nullptr;     ///< [nocc³·nvir] Wooov_re[i,k,l,d]=Wooov[k,l,i,d]
    real_t* d_Ssym1_     = nullptr;     ///< [nocc²·nvir] σ1 Wooov Ssym scratch
    real_t* d_sigma1_    = nullptr;     ///< [nocc] σ1 device accumulator

    // B-a.6a GPU port — Stage 4 full residency (env GANSU_DLPNO_NATIVE_GPU_RESIDENT=1;
    // requires the COMPLETE GPU term set — every σ2 term + the 3 σ1 terms). r1/packed_r2
    // are read straight from d_input (no input D2H/H2D, no d_r2_pack_ copy), the lifted
    // r2c / acc / σ1 / σ2 stay device-resident, and d_output is assembled on device (D2D).
    // EA apply_resident mirror. Off by default → the host-assisted apply() is byte-unchanged.
    bool use_gpu_resident_ = false;
    mutable bool resident_ = false;            ///< set true only inside apply_resident
    mutable const real_t* d_r2_src_ = nullptr; ///< resident packed-r2 source (= d_input+nocc)
    mutable const real_t* d_r1_src_ = nullptr; ///< resident r1 source (= d_input)

    /// Stage 1 GPU T2: packed_sigma2^(orient) += Lvv^(ij)·r2_packed^(orient) on
    /// device (per-orientation cublasDgemv). Exactly mirrors the host PNO block.
    void apply_t2_gpu(const std::vector<real_t>& packed_r2,
                      std::vector<real_t>& packed_sigma2) const;

    /// Stage 2 GPU projection: per orientation, σ2_packed^(orient) = U^(ij)ᵀ·acc
    /// (single GEMV) + the validated Stage-1 T2, accumulated into packed_sigma2.
    /// @param acc_all  [n_orient·nvir] host-built acc, slot s at s·nvir (canonical nvir vector).
    /// @param r1       needed by the Stage-3b T1 device term (Wovoo·r1).
    void apply_projection_gpu(const std::vector<real_t>& r1,
                              const std::vector<real_t>& acc_all,
                              const std::vector<real_t>& packed_r2,
                              std::vector<real_t>& packed_sigma2) const;

    /// Stage 3a GPU source lift: r2c_all[(p·nocc+q)·nvir] = U_{src(p,q)}·r2_packed_src
    /// (per-pair GEMV), left in d_r2c_all_ and copied to r2c_all_host.
    void lift_r2c_gpu(const std::vector<real_t>& packed_r2,
                      std::vector<real_t>& r2c_all_host) const;

    /// Stage 3b cross-pair: add T3/T4/T5 (Loo·r2c, Woooo·r2c) to the device acc
    /// stack d_acc_all_ (per orientation slot, strided GEMVs over d_r2c_all_).
    void add_xpair_gpu() const;

    /// Stage 3b T1: add -Σ_k Wovoo_lmo[k,a,oi,oj] r1[k] to the device acc (per-slot
    /// GEMV over the pre-transposed Wovoo_re). @param r1 uploaded to d_r1_.
    void add_t1_gpu(const std::vector<real_t>& r1) const;

    /// Stage 3b T8: build tmp_c (kernel + GEMV) then add -Σ_c tmp_c[c] t2_lmo[oi,oj,c,a]
    /// to the device acc (per-slot GEMV).
    void add_t8_gpu() const;

    /// Stage 3b T6/T7 ph-ladder: add the PNO-space ph-ladder directly to d_sig_pack_
    /// (one-sided barS GEMMs + per-(slot,m) GEMVs over the per-pair PNO Wovvo/Wovov).
    void add_phl_gpu() const;

    /// Stage 4 σ1: add the enabled GPU σ1 terms (Loo·r1 / Fov·r2 / Wooov·r2) into
    /// sigma1, reading the device-resident lifted r2 (d_r2c_all_). @param r1 uploaded.
    void add_sigma1_gpu(const std::vector<real_t>& r1, std::vector<real_t>& sigma1) const;

    real_t* d_diagonal_ = nullptr;     ///< [total_dim] Koopmans/PNO diagonal (preconditioner)
    std::vector<real_t> h_diagonal_;   ///< host mirror

    /// σ1[i] (1h): lift packed r2 → canonical, apply canonical σ1 formula on
    /// the borrowed intermediates. B-a.6a Stage 4: skip_loo / skip_fov / skip_wooov
    /// omit the matching term (the GPU adds it via add_sigma1_gpu); all three skipped
    /// also skips the host lift (sigma1 = 0, GPU fills it).
    void compute_sigma1(const std::vector<real_t>& r1,
                        const std::vector<real_t>& packed_r2,
                        std::vector<real_t>& sigma1,
                        bool skip_loo = false,
                        bool skip_fov = false,
                        bool skip_wooov = false) const;

    /// σ2 (2h1p), native per-pair. Accumulates the implemented terms into the
    /// packed σ2 buffer (length total_dim - nocc). B-a.2a: T2 (Lvv^(ij)·r2).
    /// @param skip_t2  B-a.6a Stage 1: when true, omit the dressed PNO-space T2
    ///        block (Lvv^(ij)·r2); the GPU path adds it via apply_t2_gpu instead.
    /// @param acc_export  B-a.6a Stage 2: when non-null, the per-orientation
    ///        canonical-virtual acc (T1/T3/T4/T5/T8b) is stored at acc_export[slot·nvir]
    ///        (slot = running orientation index) and the host U^(ij)ᵀ·acc projection
    ///        is skipped (the device does it via apply_projection_gpu); the host still
    ///        writes the PNO-space ph-ladder T6/T7 into packed_sigma2.
    /// @param r2c_external  B-a.6a Stage 3a: when non-null, the lifted source
    ///        amplitudes r2c[p,q] (canonical nvir at r2c_external[(p·nocc+q)·nvir]) are
    ///        taken from this buffer (GPU lift) instead of being built on host.
    /// @param skip_xpair  B-a.6a Stage 3b: when true, omit the cross-pair T3/T4/T5
    ///        terms from acc (the device adds them via add_xpair_gpu).
    /// @param skip_t1  B-a.6a Stage 3b: when true, omit T1 (Wovoo·r1; GPU adds it).
    /// @param skip_t8  B-a.6a Stage 3b: when true, omit T8 (both the tmp_c precompute
    ///        and the t2 subtraction; the device adds it).
    void compute_sigma2(const std::vector<real_t>& r1,
                        const std::vector<real_t>& packed_r2,
                        std::vector<real_t>& packed_sigma2,
                        bool skip_t2 = false,
                        std::vector<real_t>* acc_export = nullptr,
                        const std::vector<real_t>* r2c_external = nullptr,
                        bool skip_xpair = false,
                        bool skip_t1 = false,
                        bool skip_t8 = false,
                        bool skip_phl = false) const;
};

} // namespace gansu
