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
 * @file steom_ccsd_operator.hpp
 * @brief STEOM-CCSD effective Hamiltonian linear operator for Davidson
 *        (bt-PNO-STEOM Phase P3).
 *
 * The G^{1h1p} matvec acts on a singles excitation vector R[i,a] with
 *   total_dim = nocc_active * nvir
 * (= ordinary CIS dimension).
 *
 * Sub-phase progression:
 *   3.0+3.1: LinearOperator scaffolding + diagonal-only apply().
 *   3.2:     bar-H rebuild (11 intermediates union of IP+EA).
 *   3.3a:    Ŝ amplitude intermediate normalization (R2/(U·R1) single-divide).
 *   3.4:     F^eff_oo dressing per CFOUR `gmi_steom_rhf` —
 *              ・raw R2_IP / R2_EA + R1_IP / R1_EA + active MO indices on input
 *              ・X(MI) matrix = inverse of active R1 (n_act_occ × n_act_occ)
 *              ・U(M,I) = +2 Fov·R2 − Fov·R2 − 2 Wooov·R2 + Wooov·R2
 *              ・F^eff_oo[M_active, I] = Loo[M_idx, I] − Σ_N U(N,I) · X(N,M)
 *              ・inactive rows of F^eff_oo = bar Loo (no dressing)
 *
 * Full Ŝ-dressed bar-H build + G^{1h1p} matvec lands in sub-phases 3.5-3.7.
 *
 * IMPORTANT design note:
 *   Starting sub-phase 3.4 the constructor takes the **RAW** R2 amplitudes
 *   (NOT the single-divide-normalised Ŝ used in 3.3a). The X(MI) / X(EA)
 *   matrix inverses are built internally and stored on device. Caller passes
 *   R1, R2, and the active MO index map; ownership is by COPY into
 *   device-owned storage (caller may free source vectors after construction).
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"
#include "steom_barh_cache.hpp"   // (A) shared dressed bar-H cache

namespace gansu {

class ERI_RI;  // Phase 0: on-the-fly MO-ERI block source (build_B_mo / mo_eri_block_into)

class STEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the STEOM-CCSD G^{1h1p} operator.
     *
     * @param d_eri_mo              Device pointer to active-space MO ERIs [nao_active^4]
     *                              (chemist notation). May be nullptr — in that case
     *                              bar-H intermediates / F^eff_oo are not built and apply()
     *                              falls back to the sub-phase 3.0+3.1 diagonal-only stub
     *                              (used by unit tests).
     * @param d_orbital_energies    Device pointer to active-space orbital energies [nao_active].
     * @param d_t1                  Device pointer to CCSD T1 [nocc_active * nvir]
     *                              (ownership transferred). May be nullptr if d_eri_mo is nullptr.
     * @param d_t2                  Device pointer to CCSD T2 [nocc_active² * nvir²]
     *                              (ownership transferred). May be nullptr if d_eri_mo is nullptr.
     * @param h_R2_IP_amplitudes    Host pointer to RAW R2_IP amplitudes (no Ŝ normalization)
     *                              concatenated [n_act_occ * (nocc_active² * nvir)] with
     *                              stride (((ã*nocc_active + i)*nocc_active + j)*nvir + a).
     *                              Caller retains ownership.
     * @param h_R2_EA_amplitudes    Host pointer to RAW R2_EA amplitudes
     *                              [n_act_vir * (nocc_active * nvir²)] with stride
     *                              (((ẽ*nocc_active + j)*nvir + a)*nvir + b).
     * @param h_R1_IP_amplitudes    Host pointer to R1_IP amplitudes [n_act_occ * nocc_active]
     *                              with stride (ã*nocc_active + i). Used to build X(MI).
     *                              May be nullptr only if d_eri_mo is nullptr.
     * @param h_R1_EA_amplitudes    Host pointer to R1_EA amplitudes [n_act_vir * nvir]
     *                              with stride (ẽ*nvir + a). Used to build X(EA).
     * @param h_active_occ_idx      Host pointer to active occupied MO indices [n_act_occ]
     *                              (each entry is in [0, nocc_active)). Identifies which
     *                              MO each active NTO maps to. May be nullptr if d_eri_mo
     *                              is nullptr.
     * @param h_active_vir_idx      Host pointer to active virtual MO indices [n_act_vir]
     *                              (each entry is in [0, nvir)). May be nullptr if d_eri_mo
     *                              is nullptr.
     * @param nocc_active           Active-occupied count (= full_occ - num_frozen).
     * @param nvir                  Virtual count.
     * @param nao_active            Active-basis count (= nocc_active + nvir).
     * @param n_act_occ             Number of active occupied NTOs (== number of IP roots used).
     * @param n_act_vir             Number of active virtual NTOs (== number of EA roots used).
     */
    STEOMCCSDOperator(const real_t* d_eri_mo,
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
                      const ERI_RI* eri_block_src = nullptr,
                      const real_t* d_B_mo_blocks = nullptr,
                      int nmo_full = 0,
                      // Ship 14: per-device d_eri_vvvv slabs (driver allocates
                      // + extracts; operator takes ownership and uses them in
                      // canonical-skip Term A GEMM). Non-null + size ≥ 2
                      // enables slab mode (forces canonical_skip_wvvvv_=true).
                      std::vector<real_t*>* d_eri_vvvv_slabs_input = nullptr,
                      // (A) shared bar-H: when non-null and complete(), borrow
                      // all 11 dressed intermediates from the cache (published by
                      // IP+EA) and SKIP build_dressed_intermediates entirely. The
                      // dtor then skips freeing the 11 (the driver owns them).
                      SteomBarHCache* barh_cache = nullptr,
                      // Frozen core: MO-index offset (= num_frozen) added to every
                      // on-the-fly mo_eri_block_into range so the active blocks are
                      // read from the full-C B_mo at columns [num_frozen, num_basis).
                      // 0 ⇒ no frozen core (byte-identical).
                      int frozen_off = 0);

    ~STEOMCCSDOperator();

    STEOMCCSDOperator(const STEOMCCSDOperator&) = delete;
    STEOMCCSDOperator& operator=(const STEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "STEOMCCSDOperator"; }

    // --- Accessors ---
    int get_nocc_active() const { return nocc_active_; }
    int get_nvir()        const { return nvir_; }
    int get_n_act_occ()   const { return n_act_occ_; }
    int get_n_act_vir()   const { return n_act_vir_; }
    int get_dim()         const { return total_dim_; }  // = nocc_active * nvir

    /// Dense non-symmetric G^{1h1p} matrix [total_dim × total_dim] row-major,
    /// or nullptr if build_W_eff_and_G() has not run. Used by the optional
    /// dense-diagonalization path (GANSU_STEOM_DENSE_DIAG) to deterministically
    /// extract all eigenvalues via eigenDecompositionNonSymmetric, bypassing
    /// the run-to-run nondeterminism of the non-Hermitian Davidson.
    const real_t* get_G_device() const { return d_G_; }

    /// Print Ŝ amplitude Frobenius norms (sub-phase 3.0+3.1 smoke check).
    void print_amplitude_norms(std::ostream& os) const;

private:
    int nocc_active_;
    int nvir_;
    int nao_active_;
    int n_act_occ_;
    int n_act_vir_;
    int total_dim_;

    // Build-phase multi-GPU device count (env GANSU_STEOM_BUILD_GPUS=N>1).
    // Decoupled from the solve-phase GANSU_STEOM_EOM_GPUS and from --num_gpus
    // (the MultiGpuManager singleton is pinned to 1 device by the CIS-NTO
    // --num_gpus 1 path). 1 = legacy single-GPU build (byte-identical).
    int build_gpus_ = 1;

    // P5 canonical-skip: skip the dressed nvir⁴ Wvvvv build (host + device) when
    // (GANSU_DLPNO_NATIVE_EOM=1 AND GANSU_DLPNO_NATIVE_BARE=1 AND GANSU_DLPNO_CANONICAL_SKIP=1).
    // STEOM's d_Wvvvv_ is only read by print_intermediate_norms (verbose); the
    // dense G^{1h1p} matvec does not consume it. The Wvvvo·t1 contribution
    // (Σ_d Wvvvv[a,b,c,d]·t1[j,d]) is recomputed via a 4-term Wick refactor
    // (mirror of the EA canonical-skip) without ever materialising the nvir⁴
    // dressed intermediate. Saves ~nvir⁴·8B host + ~nvir⁴·8B device = ~16.5 GB
    // each at anthracene, decisive at 100 atoms. Default off → bit-exact.
    bool canonical_skip_wvvvv_ = false;

    // === Active NTO ↔ MO index maps (host-side, copied in constructor) ===
    std::vector<int> active_occ_idx_;   // [n_act_occ]   (each ∈ [0, nocc_active))
    std::vector<int> active_vir_idx_;   // [n_act_vir]   (each ∈ [0, nvir))

    // === Raw R1 / R2 amplitudes (device-owned, copied from host inputs) ===
    // Sub-phase 3.4: store RAW R2 (no single-divide Ŝ normalization). The
    // X(MI) / X(EA) matrices built below provide the matrix-inverse form
    // of the active-NTO normalization, matching CFOUR's `renormalize`.
    real_t* d_R2_IP_ = nullptr;   // [n_act_occ · nocc_active² · nvir]
    real_t* d_R2_EA_ = nullptr;   // [n_act_vir · nocc_active · nvir²]
    real_t* d_R1_IP_ = nullptr;   // [n_act_occ · nocc_active]
    real_t* d_R1_EA_ = nullptr;   // [n_act_vir · nvir]

    // === X matrices = active R1 inverse (n_act_* × n_act_*) ============
    // CFOUR `renormalize`: X_IP = inverse of R1_active[m_NTO, n_root] where
    // R1_active[m_NTO, n_root] = R1_IP^(n_root)[active_occ_idx[m_NTO]].
    real_t* d_X_IP_ = nullptr;    // [n_act_occ × n_act_occ] (row-major)
    real_t* d_X_EA_ = nullptr;    // [n_act_vir × n_act_vir] (row-major)

    // === F^eff_oo / F^eff_vv (full nvir²/nocc² with active rows dressed) ===
    real_t* d_F_eff_oo_ = nullptr;   // [nocc_active × nocc_active]
    real_t* d_U_MI_     = nullptr;   // [n_act_occ × nocc_active]  (IP intermediate)
    real_t* d_F_eff_vv_ = nullptr;   // [nvir × nvir]
    real_t* d_U_EA_     = nullptr;   // [n_act_vir × nvir]         (EA intermediate)

    // === Sub-phase 3.5-3.7: explicit dense G^{1h1p} singlet matrix ===
    // [total_dim × total_dim] row-major (row = i*nvir+a, col = j*nvir+b),
    // non-Hermitian. Built by build_W_eff_and_G() (port of Python
    // build_g_canonical_full, Nooijen Eq.34-63 with normalized s). When
    // present, apply() performs the dense matvec; the diagonal stub is bypassed.
    real_t* d_G_ = nullptr;          // [total_dim × total_dim]

    // === CCSD amplitudes (owned — freed in destructor) ===
    real_t* d_t1_ = nullptr;   // [nocc_active * nvir]
    real_t* d_t2_ = nullptr;   // [nocc_active² * nvir²]

    // === MO ERI blocks (union of IP + EA needs: 7 blocks) ===
    // Built by sub-phase 3.2 `extract_eri_blocks`.
    real_t* d_eri_oooo_ = nullptr;
    real_t* d_eri_ooov_ = nullptr;
    real_t* d_eri_oovv_ = nullptr;
    real_t* d_eri_ovov_ = nullptr;
    real_t* d_eri_ovvo_ = nullptr;
    real_t* d_eri_ovvv_ = nullptr;
    real_t* d_eri_vvvv_ = nullptr;

    // === Ship 14: d_eri_vvvv n-slab distribution (mirror of EA Ship 12) =====
    // For Pentacene-class workloads (NV=327) the bare ERI vvvv tensor +
    // canonical-skip Term A scratch d_inter (NV³·NO·8B ≈ 20 GB) exceed the
    // single-device budget after d_eri_vvvv (91 GB) is also resident.  When
    // GANSU_STEOM_EA_VVVV_NSLAB=N (or GANSU_EA_VVVV_NSLAB=N — shared env)
    // is set, we split d_eri_vvvv along the outermost a-axis: slab d holds
    // [a_starts_[d] .. a_ends_[d]) × NV³ doubles on device d.  Only consumer
    // with multi-GPU support is canonical-skip Term A; non-skip paths (h_vvvv
    // host alloc, d_Wvvvv_ device upload) are gated by canonical_skip_wvvvv_
    // = true when slab mode is active.  Slab ownership transfers from driver.
    int eri_vvvv_nslab_ = 1;                  // N>1 = slab mode
    std::vector<real_t*> d_eri_vvvv_slabs_;   // per-device slab device pointers
    std::vector<int> a_starts_;               // slab boundary a_starts_[d]
    std::vector<int> a_ends_;                 // slab boundary a_ends_[d]

    // === Dressed bar-H intermediates (union of IP + EA, PySCF rintermediates.py
    //  definitions; 11 distinct quantities). Built in 3.2; used by 3.3-3.7
    //  Ŝ-dressing + G^{1h1p} matvec.
    real_t* d_Loo_    = nullptr;
    real_t* d_Lvv_    = nullptr;
    real_t* d_Fov_    = nullptr;
    // IP-side intermediates (also used by STEOM bar-H, even if EA matvec didn't need them)
    real_t* d_Woooo_  = nullptr;
    real_t* d_Wooov_  = nullptr;
    real_t* d_Wovov_  = nullptr;
    real_t* d_Wovvo_  = nullptr;
    real_t* d_Wovoo_  = nullptr;
    // EA-side intermediates
    real_t* d_Wvovv_  = nullptr;
    real_t* d_Wvvvv_  = nullptr;
    real_t* d_Wvvvo_  = nullptr;

    // === (A) shared bar-H borrowing ===
    // When barh_cache_ != nullptr and complete() (dims + canonical_skip match),
    // build_dressed_intermediates points the 11 d_* members at the cache's
    // device buffers and returns without building; barh_borrowed_ is set so the
    // destructor skips freeing them (the STEOM driver owns + frees the cache).
    SteomBarHCache* barh_cache_ = nullptr;
    bool barh_borrowed_ = false;

    // === Fock diagonals (for bar-H build) ===
    real_t* d_f_oo_  = nullptr;  // [nocc_active]
    real_t* d_f_vv_  = nullptr;  // [nvir]

    // === Diagonal (stub for sub-phase 3.0+3.1, replaced in 3.7) ===
    real_t* d_diagonal_ = nullptr;  // [total_dim]
    real_t* d_eps_occ_  = nullptr;  // [nocc_active]
    real_t* d_eps_vir_  = nullptr;  // [nvir]

    void build_diagonal(const real_t* d_orbital_energies);
    // Phase 0: optional on-the-fly block source (single-GPU RI, num_frozen==0).
    // When eri_block_src_ != nullptr, extract_eri_blocks forms each ERI sub-block
    // from d_B_mo_blocks_ (the naux×nmo² half-transform) via
    // eri_block_src_->mo_eri_block_into, never allocating the full nmo⁴ tensor.
    // nullptr => legacy gather from a caller-supplied full d_eri_mo.
    const ERI_RI* eri_block_src_ = nullptr;
    const real_t* d_B_mo_blocks_ = nullptr;
    int nmo_full_ = 0;
    int frozen_off_ = 0;   // frozen-core MO offset for block ranges (0 = none)
    void extract_eri_blocks(const real_t* d_eri_mo);
    void build_dressed_intermediates();

    /// Sub-phase 3.4: build X(MI) and X(EA) on host then upload to device.
    /// Done with hand-rolled Gauss-Jordan since n_act_* is tiny (≤ ~20).
    void build_x_matrices(const real_t* h_R1_IP_amplitudes,
                          const real_t* h_R1_EA_amplitudes);

    /// Sub-phase 3.4: build U(M,I) and F^eff_oo per CFOUR `gmi_steom_rhf`.
    /// Requires bar-H intermediates + R2_IP + X(MI) already built.
    void build_F_eff_oo();

    /// Sub-phase 3.4 (extended): build U(E,A) and F^eff_vv per CFOUR
    /// `gea_steom_rhf`. Symmetric to F^eff_oo with PySCF EA-EOM σ1[a] form:
    ///   U(E,A) = +2 Fov·R2 − Fov·R2 + 2 Wvovv·R2 − Wvovv·R2
    ///   F^eff_vv[A_idx, A] = Lvv[A_idx, A] + Σ_F U(F,A) · X(F,E_NTO)
    void build_F_eff_vv();

    /// Sub-phase 3.5-3.7: build the full W^eff dressing (hp/hhhp/phph/phhp
    /// intermediates + cross), assemble g_phph/g_phhp, rebuild F^eff_oo/vv
    /// with normalized s for consistency, and form the dense G^{1h1p} singlet
    /// matrix d_G_. Direct host port of Python build_g_canonical_full
    /// (Nooijen Eq.34-63). Validation gate (H2O sto-3g): lowest singlet
    /// eigenvalues of d_G_ == 0.392886 / 0.449061 (Python reference).
    void build_W_eff_and_G();
};

} // namespace gansu
