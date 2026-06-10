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
 * @file ea_eom_ccsd_operator.hpp
 * @brief EA-EOM-CCSD linear operator for Davidson (bt-PNO-STEOM Phase P2).
 *
 * Excitation manifold (closed-shell RHF reference):
 *   R = Σ_a r_a a†_a + (1/2) Σ_{abi} r^{ab}_i a†_a a†_b a_i
 *
 * Davidson solves: σ = bar H · R with eigenvalues ω = EA value (positive Ha).
 *
 * State layout (host/device row-major):
 *   total_dim = p_dim + p2h_dim
 *   p_dim     = nvir                                  ← 1p sector
 *   p2h_dim   = nocc_active * nvir * nvir             ← 2p1h sector (FULL layout)
 *
 * The 2p1h sector keeps the full r^{ab}_i (both a<b and a>b) and is
 * anti-symmetrized on the fly inside apply(). This matches PySCF's
 * `eom_rccsd.EOMEA` layout and keeps debugging simple at the cost of 2×
 * memory in the 2p1h sector (negligible vs the n²v² EE-EOM doubles sector).
 *
 * Sub-phase 2.0+2.1: apply() implements a diagonal-only matvec. Davidson
 * eigenvalues therefore equal the sorted diagonal entries — for the 1p
 * sector these are Koopmans +ε_a values (the EA Koopmans estimates), which
 * validates the plumbing end-to-end. Full bar-H matvec lands in sub-phases
 * 2.3-2.6.
 */

#pragma once

#include <string>
#include <vector>

#include "steom_barh_cache.hpp"   // (A) shared dressed bar-H cache
#include "linear_operator.hpp"
#include "types.hpp"

namespace gansu {

class ERI_RI;  // Phase 0: on-the-fly MO-ERI block source (build_B_mo / mo_eri_block_into)

class EAEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the EA-EOM-CCSD operator.
     *
     * @param d_eri_mo       Device pointer to active-space MO ERIs [nao_active^4]
     *                       (chemist notation; trimmed if frozen_core was used).
     * @param d_orbital_energies Device pointer to active-space orbital energies [nao_active].
     * @param d_t1           Device pointer to CCSD T1 [nocc_active * nvir]
     *                       (ownership transferred to operator, freed in destructor).
     * @param d_t2           Device pointer to CCSD T2 [nocc_active² * nvir²]
     *                       (ownership transferred).
     * @param nocc           Active-occupied count (= full_occ - num_frozen).
     * @param nvir           Virtual count.
     * @param nao            Active-space basis count (= nocc + nvir).
     */
    EAEOMCCSDOperator(const real_t* d_eri_mo,
                      const real_t* d_orbital_energies,
                      real_t* d_t1, real_t* d_t2,
                      int nocc, int nvir, int nao,
                      const ERI_RI* eri_block_src = nullptr,
                      const real_t* d_B_mo_blocks = nullptr,
                      int nmo_full = 0,
                      int num_gpus = 1,    // multi-GPU σ (Stage EA-5; 1 = legacy single-GPU)
                      // Ship 12: per-device d_eri_vvvv slabs (driver allocates
                      // + extracts; operator takes ownership and uses them in
                      // canonical-skip Term A GEMM).  Non-null + size ≥ 2
                      // enables slab mode (forces canonical_skip_wvvvv_=true).
                      std::vector<real_t*>* d_eri_vvvv_slabs_input = nullptr,
                      // (A) shared bar-H: when non-null, publish the 3 EA-unique
                      // dressed intermediates (Wvovv/Wvvvv/Wvvvo) here and skip
                      // freeing them in the dtor (the cache owns them).
                      SteomBarHCache* barh_cache = nullptr,
                      // Frozen core: MO-index offset (= num_frozen) added to every
                      // on-the-fly mo_eri_block_into range so the active blocks are
                      // read from the full-C B_mo at columns [num_frozen, num_basis).
                      // 0 ⇒ no frozen core (byte-identical). Lets the block path
                      // replace the nao⁴ full-tensor build under frozen core.
                      int frozen_off = 0);

    ~EAEOMCCSDOperator();

    EAEOMCCSDOperator(const EAEOMCCSDOperator&) = delete;
    EAEOMCCSDOperator& operator=(const EAEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "EAEOMCCSDOperator"; }

    // --- Accessors ---
    int get_nocc()    const { return nocc_; }
    int get_nvir()    const { return nvir_; }
    int get_p_dim()   const { return p_dim_; }    // 1p sector dim   = nvir
    int get_p2h_dim() const { return p2h_dim_; }  // 2p1h sector dim = nocc · nvir² (FULL layout)

    // --- Dressed-intermediate device pointers (const, read-only) ---
    // Borrowed by the native per-pair DLPNO-EA-EOM σ (bt-PNO-STEOM stage B (a)).
    // Valid after construction with d_eri_mo != nullptr; nullptr otherwise.
    const real_t* get_Loo_device()   const { return d_Loo_; }    // [nocc²]
    const real_t* get_Lvv_device()   const { return d_Lvv_; }    // [nvir²]
    const real_t* get_Fov_device()   const { return d_Fov_; }    // [nocc·nvir]
    const real_t* get_Wovov_device() const { return d_Wovov_; }  // [nocc·nvir·nocc·nvir]
    const real_t* get_Wovvo_device() const { return d_Wovvo_; }  // [nocc·nvir·nvir·nocc]
    const real_t* get_Wvovv_device() const { return d_Wvovv_; }  // [nvir·nocc·nvir·nvir]
    const real_t* get_Wvvvv_device() const { return d_Wvvvv_; }  // [nvir⁴]
    const real_t* get_Wvvvo_device() const { return d_Wvvvo_; }  // [nvir·nvir·nvir·nocc]
    const real_t* get_eri_ovov_device() const { return d_eri_ovov_; }  // [nocc·nvir·nocc·nvir] (kc|ld) → tmp_k
    const real_t* get_t2_device()     const { return d_t2_; }    // [nocc²·nvir²]

    /// Print intermediate Frobenius norms (used by ea_eom_verbose ≥ 2 for
    /// PySCF cross-validation in sub-phases 2.3-2.6). Sub-phase 2.0+2.1
    /// returns the identity-stub annotation only.
    void print_intermediate_norms(std::ostream& os) const;

private:
    int nocc_;
    int nvir_;
    int nao_;
    int p_dim_;
    int p2h_dim_;
    int total_dim_;

    // === CCSD amplitudes (owned — freed in destructor) ===
    real_t* d_t1_ = nullptr;   // [nocc * nvir]
    real_t* d_t2_ = nullptr;   // [nocc² * nvir²]

    // === MO ERI blocks (subset of EE — EA needs vvvv + the standard 2-virtual blocks) ===
    // Allocated in sub-phase 2.2 (currently constructor extracts none —
    // diagonal uses orbital energies only).
    real_t* d_eri_oooo_ = nullptr;
    real_t* d_eri_ooov_ = nullptr;  // used by Wvvvo via (lj|kc) = ovoo identity
    real_t* d_eri_oovv_ = nullptr;
    real_t* d_eri_ovov_ = nullptr;
    real_t* d_eri_ovvo_ = nullptr;
    real_t* d_eri_ovvv_ = nullptr;
    real_t* d_eri_vvvv_ = nullptr;  // ★ EA-only: needed for Wvvvv build

    // === Dressed intermediates (EA-EOM-CCSD versions, PySCF rintermediates.py
    //  definitions). Loo/Lvv/Fov/Wovov/Wovvo are shared with IP-EOM (P1);
    //  Wvovv/Wvvvv/Wvvvo are EA-specific.  Built in build_dressed_intermediates.
    real_t* d_Loo_    = nullptr;  // [nocc²]
    real_t* d_Lvv_    = nullptr;  // [nvir²]
    real_t* d_Fov_    = nullptr;  // [nocc · nvir]
    real_t* d_Wovov_  = nullptr;  // [nocc · nvir · nocc · nvir]
    real_t* d_Wovvo_  = nullptr;  // [nocc · nvir · nvir · nocc]
    real_t* d_Wvovv_  = nullptr;  // [nvir · nocc · nvir · nvir]
    real_t* d_Wvvvv_  = nullptr;  // [nvir^4]
    real_t* d_Wvvvo_  = nullptr;  // [nvir · nvir · nvir · nocc]

    // === (A) shared bar-H publishing ===
    // When barh_cache_ != nullptr, build_dressed_intermediates publishes the 3
    // EA-unique intermediates (Wvovv/Wvvvv/Wvvvo) into the cache (IP already
    // published the 8 shared ones) and sets barh_published_; the destructor then
    // skips freeing ONLY those 3 (Loo/Lvv/Fov/Wovov/Wovvo are EA-owned copies,
    // always freed — the cache holds IP's bit-identical versions).
    SteomBarHCache* barh_cache_ = nullptr;
    bool barh_published_ = false;

    // === σ2 ring-term GEMM acceleration (reorganized W matrices, built once) ===
    // Lift the 3 O(nocc²·nvir³) ring contractions of σ2 out of the per-thread
    // matvec kernel into single GEMMs.  Each M is [nocc·nvir × nocc·nvir].
    //   M_ringA[(jb),(ld)] = 2 Wovvo[l,b,d,j] - Wovov[l,b,j,d]
    //   M_ringB[(ja),(lc)] = Wovov[l,a,j,c]
    //   M_ringC[(jb),(lc)] = Wovvo[l,b,c,j]
    real_t* d_M_ringA_ = nullptr;
    real_t* d_M_ringB_ = nullptr;
    real_t* d_M_ringC_ = nullptr;

    // === Diagonal & denominators ===
    real_t* d_D_p_   = nullptr;  // [nvir]            ≈ +ε_a
    real_t* d_D_p2h_ = nullptr;  // [nocc · nvir²]    ≈ -ε_j + ε_a + ε_b
    real_t* d_diagonal_ = nullptr;  // [total_dim] = [D_p | D_p2h] (used by preconditioner)
    real_t* d_f_oo_  = nullptr;  // [nocc] diagonal Fock-occ
    real_t* d_f_vv_  = nullptr;  // [nvir] diagonal Fock-vir

    // Phase 0: optional on-the-fly MO-ERI block source (single-GPU RI). When
    // set, extract_eri_blocks builds the 7 blocks from d_B_mo_blocks_ via
    // eri_block_src_->mo_eri_block_into, never the full nmo⁴. Frozen core: B_mo
    // is built over the full C (nmo_full_ MOs); frozen_off_ (= num_frozen)
    // shifts every block range into the active window [num_frozen, num_basis).
    const ERI_RI* eri_block_src_ = nullptr;
    const real_t* d_B_mo_blocks_ = nullptr;
    int nmo_full_ = 0;
    int frozen_off_ = 0;
    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_denominators_and_fock(const real_t* d_orbital_energies);
    void build_diagonal();
    void build_dressed_intermediates();   // PySCF EA intermediates — placeholder in 2.0+2.1, body in 2.2

    // === Multi-GPU σ (Stage EA-5) ===========================================
    // The EA solve apply() is the single largest hybrid-P5b hotspot.  When
    // GANSU_STEOM_EOM_GPUS=N>1 (decoupled from the RI/CIS-NTO --num_gpus, which
    // is forced to 1), the σ2 sector [j,a,b] is split over the output-occ j
    // across N physical GPUs: each device d>0 holds a full replica of every
    // intermediate apply() reads (peer-copied once here), broadcasts the input,
    // computes its j-slab of σ2, and disjoint-gathers to device 0.  num_gpus_==1
    // leaves ws_ empty and apply() takes the exact legacy single-device path
    // (byte-identical).  Mirrors the validated native-operator Stage 5.
    int num_gpus_ = 1;
    bool use_gpu_multi_ = false;   // num_gpus_>1 AND GANSU_STEOM_EOM_GPUS>1 AND nuse>1
    // P5 canonical-skip: skip the dressed nvir⁴ Wvvvv build (host + device) when
    // (GANSU_DLPNO_NATIVE_EOM=1 AND GANSU_DLPNO_NATIVE_BARE=1 AND GANSU_DLPNO_CANONICAL_SKIP=1).
    // In that mode the native per-pair operator handles σ2 (Wvvvv replaced by per-pair
    // PNO bare seed + native ring), so the canonical dense Wvvvv is never materialized.
    // Wvvvo's Σ_d Wvvvv[a,b,c,d]·t1[j,d] is recomputed term-by-term without a nvir⁴
    // intermediate (build_dressed_intermediates fused refactor). Default off → bit-exact.
    bool canonical_skip_wvvvv_ = false;

    // (RI Term A) When true, the Wvvvo·t1 dressing (the sole canonical-skip
    // consumer of the nvir⁴ (ab|cd) block) is evaluated on the fly from the RI
    // B-factors — Σ_d (ab|cd) t1[j,d] = Σ_P B_vv[P,ab] (Σ_d B_vv[P,cd] t1[j,d]).
    // d_eri_vvvv_ is then never materialised (64.5 GB at Pentacene), which also
    // removes the auto device-balancing trigger. Gated on
    // canonical_skip_wvvvv_ && RI block source && GANSU_DLPNO_EA_VVVV_RI.
    bool ri_vvvv_term_a_ = false;

    // === Ship 12: d_eri_vvvv n-slab distribution =============================
    // For 30+ atom workloads (Pentacene NV=327 → NV⁴·8B = 91.5 GB) the bare
    // ERI vvvv tensor exceeds the free memory available on any single GPU
    // (139.8 GB H200 ceiling, ~120 GB practical after persistent DLPNO state +
    // other d_eri_o* blocks). When `GANSU_EA_VVVV_NSLAB=N (N=#GPUs)` is set,
    // we split d_eri_vvvv along the outermost a-axis: slab d holds
    // [a_starts_[d] .. a_ends_[d]) × NV × NV × NV doubles on device d. The
    // only consumer with multi-GPU support is canonical-skip Term A; non-skip
    // paths (line 813 D2H to h_vvvv, line 1929 d_Wvvvv_ upload) are gated by
    // canonical_skip_wvvvv_=true when slab mode is active. Each slab is built
    // via the per-device B_mo replica + mo_eri_block_into (P4b RI DLPNO path).
    // Legacy single-pointer d_eri_vvvv_ left null in slab mode. Default OFF.
    int eri_vvvv_nslab_ = 1;                  // N>1 = slab mode
    std::vector<real_t*> d_eri_vvvv_slabs_;   // per-device slab device pointers (size = eri_vvvv_nslab_)
    std::vector<int> a_starts_;               // slab boundary a_starts_[d]
    std::vector<int> a_ends_;                 // slab boundary a_ends_[d]

    struct DeviceWorkspace {
        int     device  = -1;
        void*   cublas  = nullptr;   // cublasHandle_t for this device (ws_[0] = shared handle)
        // scratch (allocated per device; not copied)
        real_t* d_input    = nullptr;  // [total_dim]  broadcast matvec input
        real_t* d_s1       = nullptr;  // [p_dim]      σ1 scratch
        real_t* d_s2       = nullptr;  // [p2h_dim]    σ2 (only this device's j-slab is authoritative)
        real_t* d_tmp_k    = nullptr;  // [nocc]       tmp[k] (broadcast from device 0)
        real_t* d_r2T      = nullptr;  // [p2h_dim]    ring r2-swap scratch
        real_t* d_ring_tmp = nullptr;  // [p2h_dim]    R_A+R_C accumulator
        // replicated constants (peer-copied once from device 0; ws_[0] aliases members)
        real_t* d_Lvv = nullptr;   real_t* d_Loo = nullptr;   real_t* d_Fov = nullptr;
        real_t* d_Wovov = nullptr; real_t* d_Wovvo = nullptr; real_t* d_Wvovv = nullptr;
        real_t* d_Wvvvo = nullptr; real_t* d_Wvvvv = nullptr; real_t* d_t2 = nullptr;
        real_t* d_eri_ovov = nullptr;
        real_t* d_M_ringA = nullptr; real_t* d_M_ringB = nullptr; real_t* d_M_ringC = nullptr;
        int     j_begin = 0, j_end = 0;   // this device's output-occ slab [j_begin, j_end)
    };
    std::vector<DeviceWorkspace> ws_;     // size = #devices used (ws_[0] aliases device 0)
    mutable int multi_check_count_ = 0;   // capped per-matvec self-check counter (EA-5b/5c)
    void setup_multi_gpu();               // Stage EA-5a: alloc + peer-copy replicas (once after build)
    // Stage EA-5b: device-parametric GPU σ (explicit pointers + per-device cuBLAS handle),
    // so the validated single-GPU σ chain runs unchanged on any device.  `cublas` is a
    // cublasHandle_t passed as void* (hpp avoids the cuBLAS include).
    // j_begin/j_end restrict the σ2 output-occ slab (EA-5c); do_sigma1 gates the σ1
    // sector (computed only on device 0).  Defaults = full σ (single-GPU legacy).
    void apply_sigma_gpu(const real_t* d_r1, const real_t* d_r2, real_t* d_s1, real_t* d_s2,
                         const real_t* Lvv, const real_t* Loo, const real_t* Fov,
                         const real_t* Wovov, const real_t* Wovvo, const real_t* Wvovv,
                         const real_t* Wvvvo, const real_t* Wvvvv, const real_t* t2,
                         const real_t* eri_ovov, const real_t* M_ringA, const real_t* M_ringB,
                         const real_t* M_ringC, void* cublas,
                         int j_begin, int j_end, bool do_sigma1,
                         real_t* scr_tmp_k = nullptr, real_t* scr_r2T = nullptr,
                         real_t* scr_tmp = nullptr) const;  // EA-5d: persistent scratch (else malloc)
    void apply_multi(const real_t* d_input, real_t* d_output) const;  // EA-5b/5c
};

} // namespace gansu
