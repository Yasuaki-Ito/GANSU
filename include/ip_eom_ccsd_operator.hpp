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
 * @file ip_eom_ccsd_operator.hpp
 * @brief IP-EOM-CCSD linear operator for Davidson (bt-PNO-STEOM Phase P1).
 *
 * Excitation manifold (closed-shell RHF reference):
 *   R = Σ_i r_i a_i + (1/2) Σ_{ija} r^a_{ij} a†_a a_j a_i
 *
 * Davidson solves: σ = bar H · R with eigenvalues ω = IP value (positive Ha).
 *
 * State layout (host/device row-major):
 *   total_dim = h_dim + h2p_dim
 *   h_dim     = nocc_active                                  ← 1h sector
 *   h2p_dim   = nocc_active * nocc_active * nvir             ← 2h1p sector (FULL layout per design Q3)
 *
 * The 2h1p sector keeps the full r^a_{ij} (both i<j and i>j) and is
 * anti-symmetrized on the fly inside apply(). This matches PySCF's
 * `eom_rccsd.EOMIP` layout and keeps debugging simple at the cost of 2×
 * memory in the 2h1p sector (negligible vs the n²v² EE-EOM doubles
 * sector).
 *
 * Sub-phase 1.0+1.1: apply() implements a diagonal-only matvec. Davidson
 * eigenvalues therefore equal the sorted diagonal entries — for the 1h
 * sector these are Koopmans -ε_i values, which validates the plumbing
 * end-to-end. Full bar-H matvec lands in sub-phases 1.3–1.6.
 */

#pragma once

#include <string>
#include <vector>

#include "linear_operator.hpp"
#include "types.hpp"

namespace gansu {

class ERI_RI;  // Phase 0: on-the-fly MO-ERI block source (build_B_mo / mo_eri_block_into)

class IPEOMCCSDOperator : public LinearOperator {
public:
    /**
     * @brief Construct the IP-EOM-CCSD operator.
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
    IPEOMCCSDOperator(const real_t* d_eri_mo,
                      const real_t* d_orbital_energies,
                      real_t* d_t1, real_t* d_t2,
                      int nocc, int nvir, int nao,
                      const ERI_RI* eri_block_src = nullptr,
                      const real_t* d_B_mo_blocks = nullptr,
                      int nmo_full = 0,
                      int num_gpus = 1);   // multi-GPU σ (Stage IP-5; 1 = legacy single-GPU)

    ~IPEOMCCSDOperator();

    IPEOMCCSDOperator(const IPEOMCCSDOperator&) = delete;
    IPEOMCCSDOperator& operator=(const IPEOMCCSDOperator&) = delete;

    // --- LinearOperator interface ---
    void apply(const real_t* d_input, real_t* d_output) const override;
    void apply_preconditioner(const real_t* d_input, real_t* d_output) const override;
    const real_t* get_diagonal_device() const override { return d_diagonal_; }
    int dimension() const override { return total_dim_; }
    std::string name() const override { return "IPEOMCCSDOperator"; }

    // --- Accessors ---
    int get_nocc()    const { return nocc_; }
    int get_nvir()    const { return nvir_; }
    int get_h_dim()   const { return h_dim_; }    // 1h sector dim   = nocc
    int get_h2p_dim() const { return h2p_dim_; }  // 2h1p sector dim = nocc² · nvir (FULL layout)

    // --- Dressed-intermediate device pointers (const, read-only) ---
    // Borrowed by the native per-pair DLPNO-IP-EOM σ (bt-PNO-STEOM stage B (a))
    // so the occupied-heavy intermediates are bit-identical to this validated
    // operator (no re-derivation). Valid after construction with d_eri_mo !=
    // nullptr (build_dressed_intermediates ran); nullptr otherwise.
    const real_t* get_Loo_device()   const { return d_Loo_; }    // [nocc²]
    const real_t* get_Lvv_device()   const { return d_Lvv_; }    // [nvir²]
    const real_t* get_Fov_device()   const { return d_Fov_; }    // [nocc·nvir]
    const real_t* get_Woooo_device() const { return d_Woooo_; }  // [nocc⁴]
    const real_t* get_Wooov_device() const { return d_Wooov_; }  // [nocc²·nocc·nvir]
    const real_t* get_Wovov_device() const { return d_Wovov_; }  // [nocc·nvir·nocc·nvir]
    const real_t* get_Wovvo_device() const { return d_Wovvo_; }  // [nocc·nvir·nvir·nocc]
    const real_t* get_Wovoo_device() const { return d_Wovoo_; }  // [nocc·nvir·nocc²]
    const real_t* get_eri_oovv_device() const { return d_eri_oovv_; }  // [nocc²·nvir²] = Woovv
    const real_t* get_eri_ovov_device() const { return d_eri_ovov_; }  // [nocc·nvir·nocc·nvir] (ov|ov), raw ERI
    const real_t* get_eri_ovvo_device() const { return d_eri_ovvo_; }  // [nocc·nvir·nvir·nocc] (ov|vo), raw ERI
    const real_t* get_t2_device()     const { return d_t2_; }    // [nocc²·nvir²]

    /// Print intermediate Frobenius norms (used by ip_eom_verbose ≥ 2 for
    /// PySCF cross-validation in sub-phases 1.3-1.6). Sub-phase 1.0+1.1
    /// returns the identity-stub annotation only.
    void print_intermediate_norms(std::ostream& os) const;

private:
    int nocc_;
    int nvir_;
    int nao_;
    int h_dim_;
    int h2p_dim_;
    int total_dim_;

    // === CCSD amplitudes (owned — freed in destructor) ===
    real_t* d_t1_ = nullptr;   // [nocc * nvir]
    real_t* d_t2_ = nullptr;   // [nocc² * nvir²]

    // === MO ERI blocks (subset of EE — vvvv is NOT needed by IP-EOM 2h1p) ===
    // Allocated in sub-phase 1.2 (currently constructor extracts only what
    // build_diagonal needs, which is none — diagonal uses orbital energies only).
    real_t* d_eri_oooo_ = nullptr;
    real_t* d_eri_oovv_ = nullptr;
    real_t* d_eri_ovov_ = nullptr;
    real_t* d_eri_ovvo_ = nullptr;
    real_t* d_eri_ooov_ = nullptr;
    real_t* d_eri_ovvv_ = nullptr;

    // === Dressed intermediates (IP-EOM-CCSD versions, PySCF rintermediates.py
    //  definitions; NOT the EE-EOM versions). Built in build_dressed_intermediates.
    real_t* d_Loo_    = nullptr;  // [nocc²]                Loo = cc_Foo + 2*ovoo·t1 - ovoo·t1   (PySCF Loo)
    real_t* d_Lvv_    = nullptr;  // [nvir²]                Lvv = cc_Fvv + 2*ovvv·t1 - ovvv·t1   (PySCF Lvv)
    real_t* d_Fov_    = nullptr;  // [nocc · nvir]          Fov = cc_Fov                          (PySCF cc_Fov)
    real_t* d_Woooo_  = nullptr;  // [nocc^4]               Woooo (PySCF IP version, no t1·t1 symmetrization)
    real_t* d_Wooov_  = nullptr;  // [nocc² · nocc · nvir]  Wooov = ooov + t1·ovov                (PySCF Wooov)
    real_t* d_Wovov_  = nullptr;  // [nocc · nvir · nocc · nvir]  Wovov = W1ovov + W2ovov         (PySCF Wovov, IP)
    real_t* d_Wovvo_  = nullptr;  // [nocc · nvir · nvir · nocc]  Wovvo = W1ovvo + W2ovvo         (PySCF Wovvo, IP)
    real_t* d_Wovoo_  = nullptr;  // [nocc · nvir · nocc²]  Wovoo (PySCF, used in 1h↔2h1p coupling)

    // === Diagonal & denominators ===
    real_t* d_D_h_   = nullptr;  // [nocc]            ≈ -ε_i
    real_t* d_D_h2p_ = nullptr;  // [nocc² · nvir]    ≈ -ε_i - ε_j + ε_a
    real_t* d_diagonal_ = nullptr;  // [total_dim] = [D_h | D_h2p] (used by preconditioner)
    real_t* d_f_oo_  = nullptr;  // [nocc] diagonal Fock-occ
    real_t* d_f_vv_  = nullptr;  // [nvir] diagonal Fock-vir

    // Phase 0: optional on-the-fly MO-ERI block source (single-GPU RI,
    // num_frozen==0). When set, extract_eri_blocks builds the 6 blocks from
    // d_B_mo_blocks_ via eri_block_src_->mo_eri_block_into, never the full nmo⁴.
    const ERI_RI* eri_block_src_ = nullptr;
    const real_t* d_B_mo_blocks_ = nullptr;
    int nmo_full_ = 0;
    void extract_eri_blocks(const real_t* d_eri_mo);
    void compute_denominators_and_fock(const real_t* d_orbital_energies);
    void build_diagonal();
    void build_dressed_intermediates();   // duplicated inline from EE-EOM (design Q2 = duplicate)

    // === Multi-GPU σ (Stage IP-5) — mirrors the validated EA-EOM 5a-5d ===
    // GANSU_STEOM_EOM_GPUS=N>1 (decoupled from the RI/CIS-NTO --num_gpus, forced
    // to 1) splits the σ2 sector [i,j,a] (nocc²·nvir) over the outer-occ i across
    // N physical GPUs: each d>0 holds a full replica of every intermediate apply()
    // reads (all small — no nvir⁴; Woooo is only nocc⁴), computes its i-slab of σ2,
    // and disjoint-gathers to device 0.  IP apply() is 3 kernels (σ1, tmp_c, σ2),
    // no GEMM → no cuBLAS handle needed.  num_gpus_==1 → ws_ empty, legacy path
    // (byte-identical).
    int num_gpus_ = 1;
    bool use_gpu_multi_ = false;
    mutable int multi_check_count_ = 0;
    struct DeviceWorkspace {
        int     device  = -1;
        real_t* d_input = nullptr;   // [total_dim]   broadcast matvec input
        real_t* d_s1    = nullptr;   // [h_dim]       σ1 scratch
        real_t* d_s2    = nullptr;   // [h2p_dim]     σ2 (only this device's i-slab authoritative)
        real_t* d_tmp_c = nullptr;   // [nvir]        tmp[c] scratch (full reduction)
        real_t* d_Loo = nullptr;   real_t* d_Lvv = nullptr;   real_t* d_Fov = nullptr;
        real_t* d_Woooo = nullptr; real_t* d_Wooov = nullptr; real_t* d_Wovov = nullptr;
        real_t* d_Wovvo = nullptr; real_t* d_Wovoo = nullptr; real_t* d_eri_oovv = nullptr;
        real_t* d_t2 = nullptr;
        int     i_begin = 0, i_end = 0;
    };
    std::vector<DeviceWorkspace> ws_;
    void setup_multi_gpu();   // Stage IP-5: per-device replicas (no-op when num_gpus_==1)
    void apply_sigma_gpu(const real_t* d_r1, const real_t* d_r2, real_t* d_s1, real_t* d_s2,
                         const real_t* Loo, const real_t* Lvv, const real_t* Fov,
                         const real_t* Woooo, const real_t* Wooov, const real_t* Wovov,
                         const real_t* Wovvo, const real_t* Wovoo, const real_t* eri_oovv,
                         const real_t* t2, int i_begin, int i_end, bool do_sigma1,
                         real_t* scr_tmp_c = nullptr) const;
    void apply_multi(const real_t* d_input, real_t* d_output) const;
};

} // namespace gansu
