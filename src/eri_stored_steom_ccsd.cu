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

    // For active_occ_idx we use the canonical MO index of the dominant R1
    // component (argmax |R1|), NOT `pr.canonical_occ_label` — the latter is
    // tied to the CIS-NTO basis, which may be hybrid/degenerate (e.g. H2O
    // sto-3g produces 4-fold degenerate NTOs with eigenvalue 0.2222), and
    // can collide across different IP roots, making R1_active singular.
    // The dominant-canonical-MO label, by contrast, is uniquely defined for
    // each non-degenerate IP root and matches the convention of the Python
    // reference (`script/pyscf_steom_feff_reference.py`).
    auto dominant_idx = [](const std::vector<real_t>& r1) {
        int best = 0;
        real_t best_abs = -1.0;
        for (int i = 0; i < (int)r1.size(); ++i) {
            const real_t v = std::fabs(r1[i]);
            if (v > best_abs) { best_abs = v; best = i; }
        }
        return best;
    };

    int m_eff = 0;
    for (int m = 0; m < n_act_occ; ++m) {
        const auto& pr = ip_result.per_active[m];
        if (pr.canonical_occ_label < 0) continue;
        h_active_occ_idx[m_eff] = dominant_idx(pr.R1);
        std::memcpy(h_R2_IP.data() + (size_t)m_eff * r2_ip_per_root,
                    pr.R2.data(), r2_ip_per_root * sizeof(real_t));
        std::memcpy(h_R1_IP.data() + (size_t)m_eff * nocc_active,
                    pr.R1.data(), nocc_active * sizeof(real_t));
        ++m_eff;
    }
    int e_eff = 0;
    for (int e = 0; e < n_act_vir; ++e) {
        const auto& pr = ea_result.per_active[e];
        if (pr.canonical_vir_label < 0) continue;
        h_active_vir_idx[e_eff] = dominant_idx(pr.R1);
        std::memcpy(h_R2_EA.data() + (size_t)e_eff * r2_ea_per_root,
                    pr.R2.data(), r2_ea_per_root * sizeof(real_t));
        std::memcpy(h_R1_EA.data() + (size_t)e_eff * nvir,
                    pr.R1.data(), nvir * sizeof(real_t));
        ++e_eff;
    }

    // Detect duplicate dominant indices — these would make R1_active singular.
    {
        std::vector<int> sorted_occ = h_active_occ_idx;
        std::sort(sorted_occ.begin(), sorted_occ.end());
        for (int i = 1; i < (int)sorted_occ.size(); ++i) {
            if (sorted_occ[i] == sorted_occ[i-1]) {
                throw std::runtime_error(
                    "STEOM-CCSD: two filtered IP roots share dominant canonical "
                    "MO index " + std::to_string(sorted_occ[i])
                    + ". This would make the active R1 matrix singular. Likely "
                      "the IP-EOM Davidson assigned multiple roots to the same "
                      "canonical orbital — check %singles and FollowCIS thresholds.");
            }
        }
        std::vector<int> sorted_vir = h_active_vir_idx;
        std::sort(sorted_vir.begin(), sorted_vir.end());
        for (int i = 1; i < (int)sorted_vir.size(); ++i) {
            if (sorted_vir[i] == sorted_vir[i-1]) {
                throw std::runtime_error(
                    "STEOM-CCSD: two filtered EA roots share dominant canonical "
                    "MO index " + std::to_string(sorted_vir[i]) + ".");
            }
        }
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

    Timer mo_timer;
    real_t* d_eri_mo = nullptr;
    bool free_eri_mo = false;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;  // RI path — caller owns / frees it
    } else {
        tracked_cudaMalloc(&d_eri_mo,
                           (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
        free_eri_mo = true;
    }

    real_t* d_eri_for_op = d_eri_mo;
    bool free_eri_for_op = false;
    if (num_frozen > 0) {
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

    Timer build_timer;
    STEOMCCSDOperator steom_op(d_eri_for_op, d_eps_active,
                               d_t1, d_t2,
                               h_R2_IP.data(), h_R2_EA.data(),
                               h_R1_IP.data(), h_R1_EA.data(),
                               h_active_occ_idx.data(), h_active_vir_idx.data(),
                               nocc_active, nvir, nao_active,
                               n_act_occ_eff, n_act_vir_eff);

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
    // Step 3: Non-Hermitian Davidson on G^{1h1p} (stub matvec = diagonal D · x)
    // ----------------------------------------------------------------------
    Timer solve_timer;
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

    const auto& eigenvalues = solver.get_eigenvalues();
    std::vector<real_t> h_eigenvectors((size_t)n_states_to_compute * total_dim);
    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    std::cout << "  STEOM-CCSD solve time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

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
          "diagonalized by non-Hermitian Davidson. Validated vs Python reference: "
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
    real_t* d_eri_mo = build_mo_eri(d_C, num_basis);
    compute_steom_ccsd_impl(rhf_, *this, /*d_eri_ao=*/nullptr, n_states, d_eri_mo);
    tracked_cudaFree(d_eri_mo);

    rhf_.set_use_dlpno_projected_eom(false);
    rhf_.set_use_dlpno_native_eom(false);
    rhf_.clear_dlpno_amplitudes();
}

} // namespace gansu
