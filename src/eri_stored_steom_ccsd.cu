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
 * @file eri_stored_steom_ccsd.cu
 * @brief STEOM-CCSD driver (bt-PNO-STEOM Phase P3 sub-phase 3.0+3.1).
 *
 * Composite dispatch — when `--post_hf_method steom_ccsd` is called, the
 * driver checks whether the CIS-NTO active space + IP-EOM-CCSD + EA-EOM-CCSD
 * prerequisites have been populated on the HF object. If any are missing it
 * runs them in sequence (`compute_cis_nto`, `compute_ip_eom_ccsd`,
 * `compute_ea_eom_ccsd`) before assembling Ŝ amplitudes and dispatching to
 * Davidson on the STEOM operator.
 *
 * Sub-phase 3.0+3.1: Davidson runs on a diagonal-only STEOM operator (= ε_a -
 * ε_i), so excited-state energies match the CIS limit (= sorted singles
 * eigenvalues). Ŝ amplitudes are copied raw from IP/EA per_active R2 (no
 * normalisation yet — that lands in 3.3). This validates the full
 * P0→P1→P2→P3 composite chain end-to-end.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <limits>
#include <cmath>
#include <cstring>

#include "rhf.hpp"
#include "steom_ccsd_operator.hpp"
#include "steom_result.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "profiler.hpp"
#include "utils.hpp"

namespace gansu {

// Forward declarations from eri_stored.cu (shared with all EOM modules).
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);

real_t ccsd_spatial_orbital(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed = nullptr,
                            int num_frozen = 0,
                            const real_t* h_fov_active = nullptr);

extern __global__ void trim_eri_frozen_core_kernel(const real_t* __restrict__ eri_full,
                                                    real_t* __restrict__ eri_trimmed,
                                                    int N_full, int na_active, int offset);


// Collision-free greedy assignment of EOM roots to DISTINCT canonical MOs for
// the STEOM active NTO↔MO matching. Process roots in descending max|R1|; each
// takes its largest-|R1| still-unused orbital. This equals the per-root
// argmax|R1| EXACTLY when that is already collision-free (so non-degenerate
// cases — H2O, naphthalene — are bit-identical to the validated path), and only
// (near-)degenerate roots sharing an argmax MO get disambiguated (the stronger
// |R1| keeps argmax, the weaker takes its 2nd-best). Result: distinct columns →
// non-singular active R1, without the global-Hungarian's complexity. (Matches
// Python assign_active_1to1's intent; differs only in the tie/collision break.)
static std::vector<int> steom_assign_distinct(
        const std::vector<const std::vector<real_t>*>& r1_list, int n_orb) {
    const int n = static_cast<int>(r1_list.size());
    std::vector<int>    assign(n, -1);
    std::vector<int>    order(n);
    std::vector<double> rmax(n, 0.0);
    for (int r = 0; r < n; ++r) {
        order[r] = r;
        double mx = -1.0;
        for (int i = 0; i < n_orb; ++i) mx = std::max(mx, std::fabs((double)(*r1_list[r])[i]));
        rmax[r] = mx;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b){ return rmax[a] > rmax[b]; });
    std::vector<char> used(n_orb, 0);
    for (int oi = 0; oi < n; ++oi) {
        const int r = order[oi];
        int best = -1; double best_abs = -1.0;
        for (int i = 0; i < n_orb; ++i) {
            if (used[i]) continue;
            const double v = std::fabs((double)(*r1_list[r])[i]);
            if (v > best_abs) { best_abs = v; best = i; }
        }
        assign[r] = best;
        if (best >= 0) used[best] = 1;
    }
    return assign;
}

// `eri_method` is taken by ERI base reference so the composite auto-dispatch
// (compute_cis_nto / compute_ip_eom_ccsd / compute_ea_eom_ccsd) resolves
// polymorphically — stored uses AO→MO transform, RI builds the MO ERI from B.
// When `d_eri_mo_precomputed` is non-null (RI path) the AO→MO transform is
// skipped and the caller-owned MO ERI tensor is used (and not freed here).
static void compute_steom_ccsd_impl(RHF& rhf,
                                    ERI& eri_method,
                                    const real_t* d_eri_ao,
                                    int n_states_requested,
                                    real_t* d_eri_mo_precomputed = nullptr)
{
    PROFILE_FUNCTION();

    const int verbose = rhf.get_steom_verbose();

    // Sub-phase 3.12 (early stub): spin warn-and-ignore.
    {
        const std::string& st = rhf.get_spin_type();
        if (st != "singlet") {
            std::cout << "Warning: STEOM-CCSD currently runs the singlet block only. "
                         "--spin_type \"" << st << "\" is ignored at sub-phase 3.0+3.1." << std::endl;
        }
    }

    // ----------------------------------------------------------------------
    // (A) build_dressed de-duplication. The IP operator publishes its 8 dressed
    // bar-H tensors into the HF-owned SteomBarHCache, EA publishes its 3 extra
    // (Wvovv/Wvvvv/Wvvvo), and the STEOM operator borrows all 11 — skipping its
    // own build_dressed (~30s at naphthalene). All 11 are bit-identical across
    // IP/EA/STEOM, so the share is mathematically inert; the borrow is fail-safe
    // (completeness + dims/skip checks → per-operator build on any mismatch).
    // Master-switch consolidation (2026-06-03): DEFAULTS ON for every STEOM
    // dispatch — opt out with GANSU_STEOM_SHARE_BARH=0. Guarded OFF under device
    // balancing (STEOM may be redirected to another device → cross-device borrow
    // of the IP/EA-device bar-H is UNVERIFIED). Flag is set BEFORE the IP/EA
    // dispatch so the auto-run operators see it; freed at the end of this function
    // after the STEOM operator (which only reads bar-H at build time) is finished.
    // ----------------------------------------------------------------------
    {
        const char* env     = std::getenv("GANSU_STEOM_SHARE_BARH");
        const char* env_bal = std::getenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
        const bool balancing = (env_bal && env_bal[0] == '1');
        const bool share = (!env || env[0] != '0') && !balancing;  // default ON, "=0" opt-out
        rhf.set_steom_share_barh(share);
        if (share)
            std::cout << "  [STEOM share-barH] ON (default; GANSU_STEOM_SHARE_BARH=0 to "
                         "disable) — IP/EA publish dressed bar-H, STEOM borrows "
                         "(skips its build_dressed)." << std::endl;
        else if (balancing)
            std::cout << "  [STEOM share-barH] OFF (device-balancing on → cross-device "
                         "bar-H borrow unverified; per-operator build retained)." << std::endl;
    }

    // ----------------------------------------------------------------------
    // Composite dispatch: ensure CIS-NTO + IP-EOM + EA-EOM are populated.
    // Sub-phase 3.0+3.1 auto-runs each phase if its result struct is empty;
    // future P3 optimization (sub-phase 3.x) can pipe T2 / MO ERI through
    // to avoid the triple CCSD ground-state recomputation.
    // ----------------------------------------------------------------------
    if (rhf.get_cis_nto_result().n_act_occ == 0) {
        int n_cis = rhf.get_steom_n_root_cis();
        if (n_cis <= 0) n_cis = n_states_requested + 4;  // STEOM.md §7.3 default
        std::cout << "\n---- STEOM-CCSD composite dispatch: stage 1/3 = CIS-NTO active space ----" << std::endl;
        eri_method.compute_cis_nto(n_cis);
    }
    if (rhf.get_ip_eom_result().per_active.empty()) {
        std::cout << "\n---- STEOM-CCSD composite dispatch: stage 2/3 = IP-EOM-CCSD (per active occ NTO) ----" << std::endl;
        eri_method.compute_ip_eom_ccsd(n_states_requested);
    }
    if (rhf.get_ea_eom_result().per_active.empty()) {
        std::cout << "\n---- STEOM-CCSD composite dispatch: stage 3/3 = EA-EOM-CCSD (per active vir NTO) ----" << std::endl;
        eri_method.compute_ea_eom_ccsd(n_states_requested);
    }

    // Sanity check — both per_active vectors must now have entries.
    const IPEOMResult& ip_result = rhf.get_ip_eom_result();
    const EAEOMResult& ea_result = rhf.get_ea_eom_result();
    const CISNTOResult& cis_nto  = rhf.get_cis_nto_result();
    if (ip_result.per_active.empty() || ea_result.per_active.empty()) {
        throw std::runtime_error(
            "STEOM-CCSD: P1 IP-EOM and/or P2 EA-EOM did not populate per_active "
            "(no active root survived %singles filter). Try lowering "
            "--ip_eom_ip_thresh / --ea_eom_ea_thresh, or raising "
            "--cis_nto_o_thresh / --cis_nto_v_thresh.");
    }

    const int num_frozen   = rhf.get_num_frozen_core();
    const int nocc_active  = ip_result.nocc_active;
    const int nvir         = ip_result.nvir;
    const int nao_active   = nocc_active + nvir;
    const int n_act_occ    = static_cast<int>(ip_result.per_active.size());
    const int n_act_vir    = static_cast<int>(ea_result.per_active.size());
    const int total_dim    = nocc_active * nvir;
    int n_states_to_compute = std::min(n_states_requested, total_dim);
    if (n_states_to_compute <= 0) n_states_to_compute = 1;

    if (ea_result.nocc_active != nocc_active || ea_result.nvir != nvir) {
        throw std::runtime_error(
            "STEOM-CCSD: IPEOMResult and EAEOMResult disagree on (nocc_active, nvir) — "
            "they must come from the same orbital partition.");
    }
    if (cis_nto.n_act_occ != n_act_occ || cis_nto.n_act_vir != n_act_vir) {
        // Soft warning, not hard error — sub-phase 3.4 filters unassigned
        // NTOs separately. CIS-NTO active count may legitimately differ
        // from per_active size if some NTOs failed the %singles filter.
        std::cout << "  STEOM-CCSD note: CISNTOResult (n_act_occ="
                  << cis_nto.n_act_occ << ", n_act_vir=" << cis_nto.n_act_vir
                  << ") differs from IP/EA per_active sizes ("
                  << n_act_occ << ", " << n_act_vir << ")." << std::endl;
    }

    std::cout << "\n---- STEOM-CCSD ----  "
              << "nocc_active=" << nocc_active;
    if (num_frozen > 0) std::cout << " (frozen=" << num_frozen << ")";
    std::cout << ", nvir=" << nvir
              << ", n_act_occ=" << n_act_occ << ", n_act_vir=" << n_act_vir
              << ", total_dim=" << total_dim
              << ", nroots=" << n_states_to_compute << std::endl;

    // ----------------------------------------------------------------------
    // Step 1: Assemble RAW R2 / R1 amplitudes + active MO index map per
    // sub-phase 3.4 (CFOUR `renormalize`). The operator builds X(MI) /
    // X(EA) internally (matrix inverse), superseding the sub-phase 3.3a
    // single-divide Ŝ normalisation that worked only for the diagonal
    // case (NTO = identity over active MOs).
    //
    // CIS-NTO may produce more "active" NTOs than IP/EA-EOM can assign to
    // Davidson roots (e.g. if some NTOs fail the %singles filter). We
    // filter those out here — only NTOs with valid `canonical_*_label`
    // contribute to X(MI)/X(EA). Mathematically: unassigned NTOs do not
    // participate in the second similarity transform.
    // ----------------------------------------------------------------------
    const size_t r2_ip_per_root = (size_t)nocc_active * nocc_active * nvir;
    const size_t r2_ea_per_root = (size_t)nocc_active * nvir * nvir;

    // First pass — count valid NTOs (canonical_*_label >= 0) and validate
    // sizes. Unassigned NTOs are silently dropped from the STEOM dressing.
    int n_act_occ_eff = 0;
    for (int m = 0; m < n_act_occ; ++m) {
        const auto& pr = ip_result.per_active[m];
        if (pr.canonical_occ_label < 0) continue;
        if (pr.R2.size() != r2_ip_per_root) {
            throw std::runtime_error(
                "STEOM-CCSD: IPEOMResult.per_active[" + std::to_string(m)
                + "].R2 has unexpected size " + std::to_string(pr.R2.size())
                + " (expected " + std::to_string(r2_ip_per_root) + ")");
        }
        if ((int)pr.R1.size() != nocc_active) {
            throw std::runtime_error(
                "STEOM-CCSD: IPEOMResult.per_active[" + std::to_string(m)
                + "].R1 size " + std::to_string(pr.R1.size())
                + " ≠ nocc_active=" + std::to_string(nocc_active));
        }
        ++n_act_occ_eff;
    }
    int n_act_vir_eff = 0;
    for (int e = 0; e < n_act_vir; ++e) {
        const auto& pr = ea_result.per_active[e];
        if (pr.canonical_vir_label < 0) continue;
        if (pr.R2.size() != r2_ea_per_root) {
            throw std::runtime_error(
                "STEOM-CCSD: EAEOMResult.per_active[" + std::to_string(e)
                + "].R2 has unexpected size " + std::to_string(pr.R2.size())
                + " (expected " + std::to_string(r2_ea_per_root) + ")");
        }
        if ((int)pr.R1.size() != nvir) {
            throw std::runtime_error(
                "STEOM-CCSD: EAEOMResult.per_active[" + std::to_string(e)
                + "].R1 size " + std::to_string(pr.R1.size())
                + " ≠ nvir=" + std::to_string(nvir));
        }
        ++n_act_vir_eff;
    }

    if (n_act_occ_eff == 0 || n_act_vir_eff == 0) {
        throw std::runtime_error(
            "STEOM-CCSD: after filtering, no active IP or EA root remained "
            "(n_act_occ_eff=" + std::to_string(n_act_occ_eff)
            + ", n_act_vir_eff=" + std::to_string(n_act_vir_eff)
            + "). Loosen %singles thresholds or check P1/P2 root assignment.");
    }

    if (n_act_occ_eff != n_act_occ || n_act_vir_eff != n_act_vir) {
        std::cout << "  STEOM-CCSD: filtering unassigned NTOs — "
                  << "IP " << n_act_occ << "→" << n_act_occ_eff
                  << ", EA " << n_act_vir << "→" << n_act_vir_eff
                  << " (NTOs without canonical_*_label do not enter Ŝ)." << std::endl;
    }

    std::vector<real_t> h_R2_IP((size_t)n_act_occ_eff * r2_ip_per_root, 0.0);
    std::vector<real_t> h_R2_EA((size_t)n_act_vir_eff * r2_ea_per_root, 0.0);
    std::vector<real_t> h_R1_IP((size_t)n_act_occ_eff * nocc_active, 0.0);
    std::vector<real_t> h_R1_EA((size_t)n_act_vir_eff * nvir,        0.0);
    std::vector<int>    h_active_occ_idx(n_act_occ_eff, -1);
    std::vector<int>    h_active_vir_idx(n_act_vir_eff, -1);

    // Copy the filtered active R2/R1 amplitudes and gather each filtered root's
    // R1 for the active NTO↔canonical-MO assignment below. (canonical_occ_label
    // / canonical_vir_label < 0 means the NTO was not routed to an EOM root and
    // does not enter Ŝ — skip it.)
    std::vector<const std::vector<real_t>*> ip_r1_filtered, ea_r1_filtered;
    int m_eff = 0;
    for (int m = 0; m < n_act_occ; ++m) {
        const auto& pr = ip_result.per_active[m];
        if (pr.canonical_occ_label < 0) continue;
        std::memcpy(h_R2_IP.data() + (size_t)m_eff * r2_ip_per_root,
                    pr.R2.data(), r2_ip_per_root * sizeof(real_t));
        std::memcpy(h_R1_IP.data() + (size_t)m_eff * nocc_active,
                    pr.R1.data(), nocc_active * sizeof(real_t));
        ip_r1_filtered.push_back(&pr.R1);
        ++m_eff;
    }
    int e_eff = 0;
    for (int e = 0; e < n_act_vir; ++e) {
        const auto& pr = ea_result.per_active[e];
        if (pr.canonical_vir_label < 0) continue;
        std::memcpy(h_R2_EA.data() + (size_t)e_eff * r2_ea_per_root,
                    pr.R2.data(), r2_ea_per_root * sizeof(real_t));
        std::memcpy(h_R1_EA.data() + (size_t)e_eff * nvir,
                    pr.R1.data(), nvir * sizeof(real_t));
        ea_r1_filtered.push_back(&pr.R1);
        ++e_eff;
    }

    // Primary assignment = per-root argmax|R1| (the validated path; bit-identical
    // to 0b/0c for non-degenerate cases). Only if argmax collides (two roots →
    // same MO → singular active R1) do we fall back to the collision-free greedy
    // (steom_assign_distinct), and announce it. This preserves the validated
    // result exactly when there is no collision, and replaces the old hard throw
    // with a usable disambiguation when there is.
    auto argmax_abs = [](const std::vector<real_t>& r1) {
        int best = 0; double b = -1.0;
        for (int i = 0; i < (int)r1.size(); ++i) {
            const double v = std::fabs((double)r1[i]);
            if (v > b) { b = v; best = i; }
        }
        return best;
    };
    auto has_dup = [](const std::vector<int>& v, int n) {
        std::vector<int> s(v.begin(), v.begin() + n);
        std::sort(s.begin(), s.end());
        for (int i = 1; i < n; ++i) if (s[i] == s[i-1]) return true;
        return false;
    };
    for (int r = 0; r < m_eff; ++r) h_active_occ_idx[r] = argmax_abs(*ip_r1_filtered[r]);
    for (int r = 0; r < e_eff; ++r) h_active_vir_idx[r] = argmax_abs(*ea_r1_filtered[r]);
    if (m_eff > 0 && has_dup(h_active_occ_idx, m_eff)) {
        const std::vector<int> a = steom_assign_distinct(ip_r1_filtered, nocc_active);
        for (int r = 0; r < m_eff; ++r) h_active_occ_idx[r] = a[r];
        std::cout << "  [STEOM] active-occ argmax|R1| collision → greedy distinct-MO assignment "
                     "(near-degenerate roots)" << std::endl;
    }
    if (e_eff > 0 && has_dup(h_active_vir_idx, e_eff)) {
        const std::vector<int> a = steom_assign_distinct(ea_r1_filtered, nvir);
        for (int r = 0; r < e_eff; ++r) h_active_vir_idx[r] = a[r];
        std::cout << "  [STEOM] active-vir argmax|R1| collision → greedy distinct-MO assignment "
                     "(near-degenerate roots)" << std::endl;
    }
    // Hungarian yields a permutation (distinct columns) by construction; assert
    // no collision remains as a defensive guard (must never fire).
    {
        std::vector<int> s = h_active_occ_idx; std::sort(s.begin(), s.end());
        for (int i = 1; i < (int)s.size(); ++i)
            if (s[i] == s[i-1])
                throw std::runtime_error("STEOM-CCSD: active occ MO assignment collision (MO "
                    + std::to_string(s[i]) + ") after Hungarian — internal error.");
        std::vector<int> t = h_active_vir_idx; std::sort(t.begin(), t.end());
        for (int i = 1; i < (int)t.size(); ++i)
            if (t[i] == t[i-1])
                throw std::runtime_error("STEOM-CCSD: active vir MO assignment collision (MO "
                    + std::to_string(t[i]) + ") after Hungarian — internal error.");
    }

    // Suppress unused-variable warning when CIS-NTO data is not consulted
    // here anymore (active_occ_idx now comes from R1 dominant index).
    (void)cis_nto;

    // ----------------------------------------------------------------------
    // Step 2: Build STEOM operator. Sub-phase 3.2 now extracts MO ERIs and
    // builds the union of IP+EA bar-H intermediates inside the operator
    // constructor (apply() still uses the diagonal-only stub until 3.4-3.7
    // wire in the full G^{1h1p} matvec).
    //
    // CCSD ground state is re-computed here (the IP/EA operators already
    // freed their T1/T2 when their Davidsons returned). Future sub-phase 3.x
    // optimization can pipe T1/T2 between phases to avoid the triple solve.
    // ----------------------------------------------------------------------
    Timer ccsd_timer;
    const int full_occ = rhf.get_num_electrons() / 2;
    const int num_basis = rhf.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies   = rhf.get_orbital_energies();
    const real_t* d_C   = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    if (rhf.use_dlpno_amplitudes()) {
        // Hybrid DLPNO-STEOM (P5b): inject DLPNO-CCSD T1/T2 (canonical, own copy).
        const BTAmplitudes& bt = rhf.get_dlpno_bt_amplitudes();
        if (bt.nocc != nocc_active || bt.nvir != nvir)
            throw std::runtime_error("STEOM-CCSD: DLPNO amplitude dims ("
                + std::to_string(bt.nocc) + "," + std::to_string(bt.nvir)
                + ") mismatch active space (" + std::to_string(nocc_active)
                + "," + std::to_string(nvir) + ").");
        const size_t t1n = (size_t)nocc_active * nvir;
        const size_t t2n = (size_t)nocc_active * nocc_active * nvir * nvir;
        tracked_cudaMalloc(&d_t1, t1n * sizeof(real_t));
        tracked_cudaMalloc(&d_t2, t2n * sizeof(real_t));
        cudaMemcpy(d_t1, bt.T1.data(), t1n * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t2, bt.T2.data(), t2n * sizeof(real_t), cudaMemcpyHostToDevice);
        std::cout << "  STEOM using DLPNO back-transformed canonical T1/T2 (hybrid bt-PNO-STEOM P5b)." << std::endl;
    } else {
        real_t E_CCSD = ccsd_spatial_orbital(
            d_eri_ao, d_C, d_eps, num_basis, full_occ,
            /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
            &d_t1, &d_t2,
            d_eri_mo_precomputed,
            num_frozen);
        std::cout << "  STEOM CCSD ground-state re-solve: " << std::fixed << std::setprecision(10)
                  << E_CCSD << " Ha   (in " << std::setprecision(3)
                  << ccsd_timer.elapsed_seconds() << " s)" << std::endl;
    }

    // P4b: RI DLPNO path builds MO-ERI sub-blocks on the fly from B_mo
    // (naux×nmo²) inside the STEOM operator, never the full nmo⁴ tensor.
    // Single-GPU uses intermediate_matrix_B_; distributed-RI's build_B_mo
    // lazily replicates B to each GPU and returns nullptr only if the
    // replication budget would fail. Frozen core: the operator reads the active
    // window from the full-C B_mo via its frozen_off (num_frozen) shift, so the
    // block path is no longer gated on num_frozen==0 (avoids the nao⁴ tensor).
    const ERI_RI* eri_ri_block = nullptr;
    const real_t* d_B_mo_blocks = nullptr;
    if (rhf.use_dlpno_amplitudes() && gpu::gpu_available()) {
        eri_ri_block = dynamic_cast<const ERI_RI*>(&eri_method);
        if (eri_ri_block) {
            d_B_mo_blocks = eri_ri_block->build_B_mo(d_C, num_basis);
            if (!d_B_mo_blocks) eri_ri_block = nullptr;  // CPU / budget fail
        }
    }

    Timer mo_timer;
    real_t* d_eri_mo = nullptr;
    bool free_eri_mo = false;
    if (!eri_ri_block) {
        if (d_eri_mo_precomputed) {
            d_eri_mo = d_eri_mo_precomputed;  // RI path — caller owns / frees it
        } else {
            tracked_cudaMalloc(&d_eri_mo,
                               (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
            transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
            free_eri_mo = true;
        }
    }

    real_t* d_eri_for_op = d_eri_mo;
    bool free_eri_for_op = false;
    // Full-tensor frozen-core trim ONLY when not on the on-the-fly block path.
    // With eri_ri_block the operator reads active blocks straight from the
    // full-C B_mo via the frozen_off (num_frozen) shift — no nao⁴ tensor.
    if (num_frozen > 0 && !eri_ri_block) {
        const size_t na4 = (size_t)nao_active * nao_active * nao_active * nao_active;
        tracked_cudaMalloc(&d_eri_for_op, na4 * sizeof(real_t));
        free_eri_for_op = true;
        if (!gpu::gpu_available()) {
            const size_t N = num_basis;
            #pragma omp parallel for collapse(2)
            for (int p = 0; p < nao_active; ++p)
                for (int q = 0; q < nao_active; ++q)
                    for (int r = 0; r < nao_active; ++r)
                        for (int s = 0; s < nao_active; ++s) {
                            size_t src = ((size_t)(num_frozen+p)*N + (num_frozen+q))*N*N
                                       + (size_t)(num_frozen+r)*N + (num_frozen+s);
                            size_t dst = ((size_t)p*nao_active*nao_active + (size_t)q*nao_active + r)*(size_t)nao_active + s;
                            d_eri_for_op[dst] = d_eri_mo[src];
                        }
        } else {
            const int threads = 256;
            const int blocks  = (int)((na4 + threads - 1) / threads);
            trim_eri_frozen_core_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_for_op,
                                                              num_basis, nao_active, num_frozen);
            cudaDeviceSynchronize();
        }
    }
    const real_t* d_eps_active = (num_frozen > 0) ? d_eps + num_frozen : d_eps;
    std::cout << "  MO transform + frozen-core trim time: " << std::fixed << std::setprecision(3)
              << mo_timer.elapsed_seconds() << " s" << std::endl;

    // Ship 11 — STEOM operator device-redirect (mirror of EA impl). The STEOM
    // operator construction allocates d_eri_vvvv_ (nvir⁴·8B), wvvvo_w_t1
    // (nvir³·nocc·8B) and the dense G^{1h1p} matrix on the current device.
    // For Pentacene-class workloads (NV=327), nvir⁴·8B ≈ 91 GB exceeds the
    // free memory on GPU 0 (biased by persistent DLPNO ground state slab),
    // while peer GPUs sit ~131 GB free. With GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
    // we pick the device with maximum driver-free memory and redirect all
    // subsequent operator allocs there. Restored at impl exit.
    bool steom_dev_balance_active = false;
    int  steom_dev_balance_restore = 0;
    int  steom_dev_balance_target  = 0;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        const size_t tracked_global = GlobalGpuMemoryTracker::get_current();
        const size_t tracked_peak   = GlobalGpuMemoryTracker::get_peak();
        const size_t nvir_sz = (size_t)nvir * nvir * nvir * nvir * sizeof(real_t);
        const double need_vvvv_gb = nvir_sz / (1024.0*1024.0*1024.0);
        std::cout << "  [STEOM device audit] tracked sum-of-GPUs current = "
                  << std::fixed << std::setprecision(2)
                  << (tracked_global / (1024.0*1024.0*1024.0)) << " GB"
                  << "   peak = " << (tracked_peak / (1024.0*1024.0*1024.0)) << " GB"
                  << "   (operator will alloc d_eri_vvvv_ = " << need_vvvv_gb << " GB next)"
                  << std::defaultfloat << std::endl;
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        int saved_dev = 0;
        cudaGetDevice(&saved_dev);
        steom_dev_balance_restore = saved_dev;
        std::vector<size_t> per_dev_free(n_dev, 0);
        for (int d = 0; d < n_dev; ++d) {
            cudaSetDevice(d);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                per_dev_free[d] = free_b;
                const double used_gb  = (total_b - free_b) / (1024.0*1024.0*1024.0);
                const double free_gb  = free_b / (1024.0*1024.0*1024.0);
                const double total_gb = total_b / (1024.0*1024.0*1024.0);
                std::cout << "    GPU " << d << " driver:   used="
                          << std::fixed << std::setprecision(2) << used_gb << " GB"
                          << "  free=" << free_gb << " GB"
                          << "  total=" << total_gb << " GB"
                          << "  d_eri_vvvv_ alloc "
                          << (free_b >= nvir_sz ? "fits" : "WILL OOM")
                          << std::defaultfloat << std::endl;
            }
        }
        const char* env_balance = std::getenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
        if (env_balance && env_balance[0] == '1' && n_dev > 1) {
            int best_dev = saved_dev;
            size_t best_free = (saved_dev >= 0 && saved_dev < n_dev) ? per_dev_free[saved_dev] : 0;
            for (int d = 0; d < n_dev; ++d) {
                if (per_dev_free[d] > best_free) {
                    best_free = per_dev_free[d];
                    best_dev  = d;
                }
            }
            if (best_dev != saved_dev) {
                cudaSetDevice(best_dev);
                // Rebuild thread_local GPUHandle (cuBLAS + cuSOLVER) on the
                // new device — otherwise STEOM operator's many DGEMMs run
                // with a handle bound to the original device while pointers
                // live on the new one (CUBLAS_STATUS_EXECUTION_FAILED).
                gpu::GPUHandle::reset();
                steom_dev_balance_active = true;
                steom_dev_balance_target = best_dev;
                std::cout << "  [STEOM device-balance] redirecting operator to GPU "
                          << best_dev
                          << " (free=" << std::fixed << std::setprecision(2)
                          << (best_free / (1024.0*1024.0*1024.0)) << " GB"
                          << ", was on GPU " << saved_dev
                          << " with free=" << (per_dev_free[saved_dev] / (1024.0*1024.0*1024.0))
                          << " GB; GPUHandle rebuilt for new device)"
                          << std::defaultfloat << std::endl;
            } else {
                cudaSetDevice(saved_dev);
                std::cout << "  [STEOM device-balance] GPU " << saved_dev
                          << " already has max free memory; no redirect."
                          << std::endl;
            }
        } else {
            cudaSetDevice(saved_dev);
        }
    }
#endif

    // Ship 14: vvvv n-slab distribution (mirror of EA Ship 12 for STEOM).
    // When GANSU_EA_VVVV_NSLAB=N is set and we have the P4b on-the-fly path,
    // allocate + extract per-device d_eri_vvvv slabs immediately so the STEOM
    // operator's canonical-skip Term A GEMM can run per-device against its
    // own slab.  For Pentacene (NV=327) this avoids the single-device
    // bottleneck after Ship 11's redirect: GPU 3 alone can't hold d_eri_vvvv
    // (91 GB) + Term A scratch d_inter (20 GB) + other state.  Slab ownership
    // transfers to the operator ctor below.
    std::vector<real_t*> steom_d_eri_vvvv_slabs;
#ifndef GANSU_CPU_ONLY
    if (eri_ri_block && gpu::gpu_available()) {
        const char* env_nslab = std::getenv("GANSU_EA_VVVV_NSLAB");
        if (env_nslab && env_nslab[0]) {
            const int Nreq = std::atoi(env_nslab);
            int n_dev = 0;
            cudaGetDeviceCount(&n_dev);
            const int N = std::min(Nreq, n_dev);
            if (N >= 2) {
                int saved = 0;
                cudaGetDevice(&saved);
                steom_d_eri_vvvv_slabs.assign(N, nullptr);
                bool all_ok = true;
                for (int d = 0; d < N; ++d) {
                    cudaSetDevice(d);
                    const real_t* B_d = eri_ri_block->build_B_mo(d_C, num_basis);
                    if (!B_d) {
                        all_ok = false;
                        std::cout << "  [STEOM Ship 14] WARNING: build_B_mo returned null "
                                     "for device " << d << "; disabling slab mode." << std::endl;
                        break;
                    }
                    const int a_start = (int)((int64_t)d * nvir / N);
                    const int a_end   = (int)((int64_t)(d + 1) * nvir / N);
                    const int an      = a_end - a_start;
                    const size_t slab_sz = (size_t)an * nvir * nvir * nvir;
                    tracked_cudaMalloc(&steom_d_eri_vvvv_slabs[d], slab_sz * sizeof(real_t));
                    // Frozen core: virtual block starts at full_occ = nocc_active
                    // + num_frozen in the full-C B_mo (B_d built over num_basis MOs).
                    eri_ri_block->mo_eri_block_into(B_d, num_basis,
                        nocc_active + num_frozen + a_start, an,   // a range (slab)
                        nocc_active + num_frozen, nvir,           // b range
                        nocc_active + num_frozen, nvir,           // c range
                        nocc_active + num_frozen, nvir,           // d range
                        steom_d_eri_vvvv_slabs[d]);
                    cudaDeviceSynchronize();
                    std::cout << "  [STEOM Ship 14] slab d=" << d
                              << " allocated " << std::fixed << std::setprecision(2)
                              << (slab_sz * sizeof(real_t) / (1024.0*1024.0*1024.0))
                              << " GB on GPU " << d
                              << " (a∈[" << a_start << "," << a_end << "))"
                              << std::defaultfloat << std::endl;
                }
                cudaSetDevice(saved);
                if (!all_ok) {
                    for (int d = 0; d < N; ++d) {
                        if (steom_d_eri_vvvv_slabs[d]) {
                            cudaSetDevice(d);
                            tracked_cudaFree(steom_d_eri_vvvv_slabs[d]);
                            steom_d_eri_vvvv_slabs[d] = nullptr;
                        }
                    }
                    cudaSetDevice(saved);
                    steom_d_eri_vvvv_slabs.clear();
                } else {
                    // Refresh build_B_mo on the saved (Ship 11 target) device so
                    // the operator's stored d_B_mo_blocks is valid for the other
                    // 6 d_eri_* extractions in extract_eri_blocks.
                    d_B_mo_blocks = eri_ri_block->build_B_mo(d_C, num_basis);
                    if (!d_B_mo_blocks) {
                        std::cout << "  [STEOM Ship 14] WARNING: post-loop "
                                     "build_B_mo refresh on saved device returned null; "
                                     "operator extract_eri_blocks will likely fail."
                                  << std::endl;
                    }
                }
            } else {
                std::cout << "  [STEOM Ship 14] GANSU_EA_VVVV_NSLAB=" << Nreq
                          << " but only " << n_dev << " GPUs visible; needs ≥2 to slab"
                          << std::endl;
            }
        }
    }
#endif

    Timer build_timer;
    STEOMCCSDOperator steom_op(d_eri_for_op, d_eps_active,
                               d_t1, d_t2,
                               h_R2_IP.data(), h_R2_EA.data(),
                               h_R1_IP.data(), h_R1_EA.data(),
                               h_active_occ_idx.data(), h_active_vir_idx.data(),
                               nocc_active, nvir, nao_active,
                               n_act_occ_eff, n_act_vir_eff,
                               eri_ri_block, d_B_mo_blocks, num_basis,
                               steom_d_eri_vvvv_slabs.empty()
                                   ? nullptr : &steom_d_eri_vvvv_slabs,
                               // (A) shared bar-H: borrow all 11, skip build_dressed
                               rhf.steom_share_barh() ? &rhf.steom_barh_cache() : nullptr,
                               // Frozen core: block ranges read [num_frozen, num_basis)
                               // of the full-C B_mo (only used on the block path).
                               num_frozen);

    // Operator owns T1/T2 + has copied bar-H intermediates; we can free the
    // trimmed / full MO ERI tensor (operator pulled the sub-blocks it needs).
    if (free_eri_for_op) tracked_cudaFree(d_eri_for_op);
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);

    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s "
              << "(sub-phase 3.5-3.7: ERI blocks + 11 bar-H intermediates + X(MI)/X(EA) "
                 "+ F^eff_oo/vv + full W^eff-dressed G^{1h1p}; apply = dense non-Hermitian "
                 "matvec)" << std::endl;
    if (verbose >= 2) steom_op.print_amplitude_norms(std::cout);

    // ----------------------------------------------------------------------
    // Step 3: diagonalize G^{1h1p}.
    //
    // DEFAULT for small/mid G = full dense non-Hermitian diagonalization
    // (eigenDecompositionNonSymmetric: Eigen CPU / cusolverDnXgeev GPU). It is
    // deterministic and returns ALL eigenvalues sorted ascending by real part;
    // we then select the lowest n_states real roots ≥ min_eigenvalue. This was
    // promoted to the default after the non-Hermitian Davidson was shown to be
    // unreliable on (near-)degenerate G: it both MISSES roots and fails to
    // reach the true lowest root (naphthalene Davidson 3.4527/…/5.8994 eV vs
    // dense 3.4402/…/5.5770 — Davidson dropped the 5.4313 eV root and overshot
    // k0 by +0.0125 eV; its "reproducible" answer was a reproducible error).
    //
    // Dense memory ~ 5·total_dim² doubles + geev workspace and cost ~ total_dim³,
    // so for LARGE G we fall back to the iterative non-Hermitian Davidson.
    // Overrides: GANSU_STEOM_DENSE_DIAG=1 forces dense regardless of size;
    // GANSU_STEOM_DAVIDSON=1 forces Davidson regardless.
    // ----------------------------------------------------------------------
    Timer solve_timer;
    const int  dense_auto_max = 12000;  // total_dim threshold for auto dense (~5.8 GB at 12000)
    const bool force_dense    = (std::getenv("GANSU_STEOM_DENSE_DIAG") != nullptr);
    const bool force_davidson = (std::getenv("GANSU_STEOM_DAVIDSON")  != nullptr);
    const bool dense_diag = (steom_op.get_G_device() != nullptr)
                            && !force_davidson
                            && (force_dense || total_dim <= dense_auto_max);

    std::vector<real_t> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_states_to_compute * total_dim);

    if (dense_diag) {
        const real_t min_eigenvalue = 0.0;  // STEOM excitation energies are positive
        // eigenDecompositionNonSymmetric expects column-major input. d_G_ is
        // row-major [total_dim × total_dim]; its linear buffer is exactly the
        // column-major storage of Gᵀ. eig(Gᵀ) == eig(G), so passing the buffer
        // directly yields the correct eigenvalues; to also recover the RIGHT
        // eigenvectors of G we transpose the copy into true column-major G.
        real_t* d_G_cm = nullptr;
        tracked_cudaMalloc(&d_G_cm, (size_t)total_dim * total_dim * sizeof(real_t));
        cudaMemcpy(d_G_cm, steom_op.get_G_device(),
                   (size_t)total_dim * total_dim * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        gpu::transposeMatrixInPlace(d_G_cm, total_dim);  // row-major G → column-major G

        real_t* d_all_evals = nullptr;
        real_t* d_all_evecs = nullptr;
        tracked_cudaMalloc(&d_all_evals, (size_t)total_dim * sizeof(real_t));
        tracked_cudaMalloc(&d_all_evecs, (size_t)total_dim * total_dim * sizeof(real_t));

        int info = gpu::eigenDecompositionNonSymmetric(d_G_cm, d_all_evals, d_all_evecs, total_dim);
        if (info != 0) {
            std::cout << "Warning: STEOM dense diagonalization (geev) returned info="
                      << info << "." << std::endl;
        }

        // eigenvalues are sorted ascending by real part; complex/missing slots
        // were filled with 1e30 at the tail. Eigenvectors are in transposed
        // layout (eigenvector i = row i, stride = total_dim).
        std::vector<real_t> h_all_evals(total_dim);
        cudaMemcpy(h_all_evals.data(), d_all_evals,
                   (size_t)total_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

        // Select the lowest n_states real roots ≥ min_eigenvalue.
        std::vector<int> sel;
        sel.reserve(n_states_to_compute);
        for (int i = 0; i < total_dim && (int)sel.size() < n_states_to_compute; ++i) {
            if (h_all_evals[i] >= min_eigenvalue && h_all_evals[i] < 1e29)
                sel.push_back(i);
        }
        if ((int)sel.size() < n_states_to_compute) {
            std::cout << "Warning: STEOM dense diagonalization found only " << sel.size()
                      << " real roots ≥ " << min_eigenvalue << " (requested "
                      << n_states_to_compute << ")." << std::endl;
        }

        eigenvalues.assign(n_states_to_compute, 1e30);
        for (int n = 0; n < (int)sel.size(); ++n)
            eigenvalues[n] = h_all_evals[sel[n]];
        // Strided D2H copy of the selected eigenvectors (transposed layout:
        // element j of eigenvector idx is at d_all_evecs[idx + j*total_dim]).
        {
            std::vector<real_t> h_all_evecs((size_t)total_dim * total_dim);
            cudaMemcpy(h_all_evecs.data(), d_all_evecs,
                       (size_t)total_dim * total_dim * sizeof(real_t), cudaMemcpyDeviceToHost);
            for (int n = 0; n < (int)sel.size(); ++n) {
                int idx = sel[n];
                for (int j = 0; j < total_dim; ++j)
                    h_eigenvectors[(size_t)n * total_dim + j] = h_all_evecs[(size_t)idx + (size_t)j * total_dim];
            }
        }

        tracked_cudaFree(d_G_cm);
        tracked_cudaFree(d_all_evals);
        tracked_cudaFree(d_all_evecs);

        std::cout << "  STEOM-CCSD solve time: " << std::fixed << std::setprecision(3)
                  << solve_timer.elapsed_seconds() << " s "
                  << "(dense non-Hermitian geev, deterministic; "
                  << (force_dense ? "forced via GANSU_STEOM_DENSE_DIAG"
                                  : "auto: total_dim=" + std::to_string(total_dim)
                                    + " ≤ " + std::to_string(dense_auto_max))
                  << ")" << std::endl;
    } else {
        if (steom_op.get_G_device() != nullptr && !force_davidson)
            std::cout << "  STEOM-CCSD: total_dim=" << total_dim << " > " << dense_auto_max
                      << " → iterative non-Hermitian Davidson (dense geev too large; "
                      << "note Davidson may miss/over-shoot roots — force dense with "
                      << "GANSU_STEOM_DENSE_DIAG=1 if affordable)." << std::endl;
        DavidsonConfig config;
        config.num_eigenvalues       = n_states_to_compute;
        config.convergence_threshold = rhf.get_steom_d_tol();
        config.max_subspace_size     = std::min(total_dim, std::max(80, 20 * n_states_to_compute));
        config.max_iterations        = rhf.get_steom_max_iter();
        config.use_preconditioner    = true;
        config.symmetric             = false;
        config.min_eigenvalue        = 0.0;
        config.verbose               = (verbose >= 2) ? 2 : (verbose >= 1 ? 1 : 0);

        DavidsonSolver solver(steom_op, config);
        bool converged = solver.solve();
        if (!converged) {
            std::cout << "Warning: STEOM-CCSD Davidson did not converge for all roots." << std::endl;
        }

        eigenvalues = solver.get_eigenvalues();
        solver.copy_eigenvectors_to_host(h_eigenvectors.data());

        std::cout << "  STEOM-CCSD solve time: " << std::fixed << std::setprecision(3)
                  << solve_timer.elapsed_seconds() << " s" << std::endl;
    }

    // ----------------------------------------------------------------------
    // Step 4: Build STEOMResult
    // ----------------------------------------------------------------------
    STEOMResult result;
    result.nocc_active = nocc_active;
    result.nvir        = nvir;
    result.num_frozen  = num_frozen;
    result.n_states    = n_states_to_compute;
    result.per_root.resize(n_states_to_compute);
    for (int n = 0; n < n_states_to_compute; ++n) {
        auto& pr = result.per_root[n];
        pr.omega = eigenvalues[n];
        pr.R1.assign(&h_eigenvectors[(size_t)n * total_dim],
                     &h_eigenvectors[(size_t)n * total_dim + total_dim]);
        // η / percent_active_occ / percent_active_vir stay at sentinel -1.0
        // (sub-phase 3.10 populates them).
    }

    // ----------------------------------------------------------------------
    // Step 5: human-readable report
    // ----------------------------------------------------------------------
    std::ostringstream os;
    os << "[STEOM-CCSD] sub-phase 3.5-3.7 — full W^eff-dressed G^{1h1p} "
          "(Nooijen-Bartlett Eq.34-63: F^eff_oo/vv + hp/hhhp/phph/phhp + IP×EA cross), "
          "diagonalized by "
       << (dense_diag ? "dense non-Hermitian geev (deterministic)"
                      : "non-Hermitian Davidson")
       << ". Validated vs Python reference: "
          "lowest two roots bit-exact (H2O sto-3g 0.392886 / 0.449061).\n"
       << "  STEOM excited-state energies:\n"
       << "   k   omega (Ha)        omega (eV)\n";
    for (int n = 0; n < n_states_to_compute; ++n) {
        const auto& pr = result.per_root[n];
        os << "  " << std::setw(2) << n
           << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
           << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114)
           << "\n";
    }
    os << "  (η = % active character lands in sub-phase 3.10.)\n";

    result.report = os.str();
    std::cout << result.report;

    rhf.append_excited_state_report(result.report);
    rhf.set_steom_result(std::move(result));

    // (A) shared bar-H: release the cache device buffers. Safe here — the STEOM
    // operator only reads bar-H during its build (build_F_eff_*/build_W_eff_and_G);
    // the apply()/matvec consumed by the solve above uses the dense d_G_ only, so
    // the borrowed bar-H pointers are no longer referenced. steom_op (still in
    // scope) borrowed them and its dtor skips freeing (barh_borrowed_), so this is
    // the single owner-side release with no double-free.
    if (rhf.steom_share_barh()) {
        rhf.steom_barh_cache().free();
        rhf.set_steom_share_barh(false);
    }

    // Ship 11 — restore caller's device after STEOM operator + Davidson are
    // both destructed (RAII), so downstream output / API code sees the same
    // device it set before calling STEOM.
#ifndef GANSU_CPU_ONLY
    if (steom_dev_balance_active) {
        cudaSetDevice(steom_dev_balance_restore);
        gpu::GPUHandle::reset();
        std::cout << "  [STEOM device-balance] restored to GPU " << steom_dev_balance_restore
                  << " (operator ran on GPU " << steom_dev_balance_target
                  << "; GPUHandle rebuilt for restored device)" << std::endl;
    }
#endif
}

void ERI_Stored_RHF::compute_steom_ccsd(int n_states) {
    compute_steom_ccsd_impl(rhf_, *this, eri_matrix_.device_ptr(), n_states);
}

// bt-PNO-STEOM P4 (RI path): build the full MO ERI from the RI B factors once
// (matching RI-CCSD), then run the identical canonical STEOM-CCSD impl. The
// composite auto-dispatch (CIS-NTO → IP-EOM → EA-EOM) resolves polymorphically
// to the RI overrides, so every stage uses RI-approximated integrals. Result
// matches canonical STEOM to the RI fitting tolerance (~1e-4 Ha, STEOM.md §14.6).
// Note: each auto-dispatch stage rebuilds its own MO ERI internally (correctness
// over speed, mirroring the existing triple CCSD re-solve); a future sub-phase
// can pipe a single MO ERI through all stages.
void ERI_RI_RHF::compute_steom_ccsd(int n_states) {
    const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
    const int num_basis = rhf_.get_num_basis();
    real_t* d_eri_mo = build_mo_eri(d_C, num_basis);
    compute_steom_ccsd_impl(rhf_, *this, /*d_eri_ao=*/nullptr, n_states, d_eri_mo);
    tracked_cudaFree(d_eri_mo);
}

// Hybrid DLPNO-STEOM-CCSD (bt-PNO-STEOM Phase P5b). Stage 1: DLPNO-CCSD ground
// state, back-transformed to canonical MO (bt_pno_to_canonical) and stashed on
// the HF object. Stage 2: the canonical STEOM auto-chain (CIS-NTO → IP-EOM →
// EA-EOM → STEOM), where the IP/EA/STEOM operators consume the stashed DLPNO
// amplitudes instead of solving a fresh canonical CCSD. CIS-NTO is canonical
// (CIS-based, independent of the CCSD ground state). The excited-state
// machinery + final diagonalization stay canonical (the "hybrid"); a future
// stage B replaces the IP/EA-EOM with PNO-basis DLPNO versions for 100-atom
// scaling. Like RI-STEOM, requires --num_gpus 1 (RI CIS-NTO is single-GPU).
void ERI_RI_RHF::compute_dlpno_steom_ccsd(int n_states) {
    std::cout << "\n==== DLPNO-STEOM-CCSD (hybrid bt-PNO-STEOM P5b) ====" << std::endl;

    // Stage 1: DLPNO-CCSD ground state → back-transform → stash on HF.
    std::cout << "---- DLPNO-STEOM stage 1: DLPNO-CCSD ground state (back-transformed to canonical) ----" << std::endl;
    rhf_.set_collect_dlpno_bt(true);
    const real_t E_dlpno = compute_dlpno_ccsd();
    rhf_.set_collect_dlpno_bt(false);
    rhf_.set_post_hf_energy(E_dlpno);  // report the DLPNO-CCSD ground-state correlation
    if (!rhf_.use_dlpno_amplitudes()) {
        throw std::runtime_error(
            "DLPNO-STEOM-CCSD: DLPNO-CCSD did not produce back-transformed canonical "
            "amplitudes (collect path failed). Check the DLPNO-CCSD run.");
    }
    std::cout << "  DLPNO-CCSD correlation energy = " << std::fixed << std::setprecision(10)
              << E_dlpno << " Ha  (fed to the canonical STEOM machinery)" << std::endl;

    // Stage B opt-in (env GANSU_DLPNO_PROJECTED_EOM=1): run the Galerkin-
    // projected DLPNO-IP-EOM and DLPNO-EA-EOM (per-pair PNO 2h1p / per-i PNO(ii)
    // 2p1h spaces) instead of the canonical IP/EA Davidsons → full DLPNO-bt-
    // STEOM. Default false → validated hybrid P5b path unchanged. (The legacy
    // GANSU_DLPNO_PROJECTED_IP name is still honoured.)
    const bool proj_eom = []() {
        const char* e = std::getenv("GANSU_DLPNO_PROJECTED_EOM");
        if (!e) e = std::getenv("GANSU_DLPNO_PROJECTED_IP");
        return e && e[0] == '1';
    }();
    if (proj_eom) {
        rhf_.set_use_dlpno_projected_eom(true);
        std::cout << "  (projected DLPNO-IP/EA-EOM enabled — per-pair PNO spaces)" << std::endl;
    }

    // Stage B (a) opt-in (env GANSU_DLPNO_NATIVE_EOM=1; legacy GANSU_DLPNO_NATIVE_IP
    // honoured): run the NATIVE per-pair σ for BOTH the IP (DLPNOIPEOMNativeOperator)
    // and EA (DLPNOEAEOMNativeOperator) sectors → full native DLPNO-bt-STEOM (the
    // true-scaling path, no per-matvec canonical σ). Native σ == projected σ to
    // machine epsilon (validated), so STEOM roots match the projected path
    // (modulo the documented STEOM final-stage run-to-run nondeterminism).
    const bool native_eom = []() {
        const char* e = std::getenv("GANSU_DLPNO_NATIVE_EOM");
        if (!e) e = std::getenv("GANSU_DLPNO_NATIVE_IP");  // legacy name
        return e && e[0] == '1';
    }();
    if (native_eom) {
        rhf_.set_use_dlpno_native_eom(true);   // both IP and EA impls take the native branch
        std::cout << "  (native DLPNO-IP/EA-EOM enabled — full per-pair σ bt-PNO-STEOM)" << std::endl;
    }

    // Stage 2: canonical CIS-NTO + IP/EA/STEOM, consuming the DLPNO amplitudes.
    std::cout << "---- DLPNO-STEOM stage 2: CIS-NTO + IP/EA-EOM + STEOM ----" << std::endl;
    const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
    const int num_basis = rhf_.get_num_basis();
    // P4b: STEOM operator's MO-ERI sub-blocks built on the fly inside
    // compute_steom_ccsd_impl from B_mo (no full nmo⁴), single-GPU and
    // distributed (lazy replicate). Probe build_B_mo here; if it returns
    // nullptr (CPU / budget refused) we must pre-build the full nmo⁴ tensor
    // because compute_steom_ccsd_impl has no d_eri_ao to AO→MO transform from.
    // Frozen core: the block path reads the active window from the full-C B_mo
    // via the operator's frozen_off shift, so it is NO LONGER gated on
    // num_frozen==0 — avoids the nao⁴ MO-ERI tensor (OOM for ~tetracene+).
    const bool want_block = rhf_.use_dlpno_amplitudes()
                            && gpu::gpu_available();
    const real_t* probe = want_block ? build_B_mo(d_C, num_basis) : nullptr;
    real_t* d_eri_mo = (probe != nullptr) ? nullptr : build_mo_eri(d_C, num_basis);
    compute_steom_ccsd_impl(rhf_, *this, /*d_eri_ao=*/nullptr, n_states, d_eri_mo);
    if (d_eri_mo) tracked_cudaFree(d_eri_mo);

    rhf_.set_use_dlpno_projected_eom(false);
    rhf_.set_use_dlpno_native_eom(false);
    rhf_.clear_dlpno_amplitudes();
}

} // namespace gansu
