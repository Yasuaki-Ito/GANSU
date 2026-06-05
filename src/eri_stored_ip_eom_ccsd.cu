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
 * @file eri_stored_ip_eom_ccsd.cu
 * @brief IP-EOM-CCSD driver (bt-PNO-STEOM Phase P1 sub-phase 1.0+1.1).
 *
 * Workflow (matches eri_stored_eom_ccsd.cu pattern):
 *   1. Solve CCSD ground state (reuse ccsd_spatial_orbital) → T1, T2
 *   2. Construct IPEOMCCSDOperator (sub-phase 1.0+1.1: diagonal-only stub)
 *   3. Run non-Hermitian Davidson on the operator
 *   4. Pull eigenvalues + R1 from device, print summary, persist into HF
 *
 * Sub-phase 1.0+1.1 is intentionally a smoke-test scaffold:
 *  - apply() is diagonal-only, so eigenvalues equal sorted `-ε_i` (Koopmans)
 *    for the 1h sector. This validates the plumbing without the bar-H
 *    intermediates.
 *  - FollowCIS / %singles filter / IPEOMResult.per_active are wired but only
 *    populated when CISNTOResult is present; otherwise we fall through to
 *    n_excited_states lowest IPs (passive mode).
 *  - Full bar-H matvec lands in sub-phases 1.2-1.6.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "rhf.hpp"
#include "dlpno_ip_packing.hpp"               // bt-PNO-STEOM stage B: packed layout
#include "dlpno_ip_eom_projected_operator.hpp" // stage B: projected DLPNO-IP operator
#include "dlpno_ip_eom_native_operator.hpp"    // stage B (a): native per-pair DLPNO-IP σ operator
#include "dlpno_ip_eom_transform.hpp"          // stage B: packed↔canonical R2 transform
#include <memory>
#include "ip_eom_ccsd_operator.hpp"
#include "ip_eom_result.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "profiler.hpp"
#include "utils.hpp"

namespace gansu {

// Forward declarations (defined in eri_stored.cu / shared with EE-EOM-CCSD)
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

// GPU kernel for trimming MO ERI to frozen-core active space
// (defined in eri_stored.cu, also used by EE-EOM-CCSD driver)
extern __global__ void trim_eri_frozen_core_kernel(const real_t* __restrict__ eri_full,
                                                    real_t* __restrict__ eri_trimmed,
                                                    int N_full, int na_active, int offset);


static void compute_ip_eom_ccsd_impl(RHF& rhf,
                                     const real_t* d_eri_ao,
                                     int n_roots_requested,
                                     real_t* d_eri_mo_precomputed = nullptr,
                                     const ERI_RI* eri_block_src = nullptr,
                                     const real_t* d_B_mo_blocks = nullptr)
{
    PROFILE_FUNCTION();

    const int num_frozen  = rhf.get_num_frozen_core();
    const int num_basis   = rhf.get_num_basis();
    const int full_occ    = rhf.get_num_electrons() / 2;
    const int nocc_active = full_occ - num_frozen;
    const int nvir        = num_basis - full_occ;
    const int nao_active  = nocc_active + nvir;
    const int h_dim       = nocc_active;
    const int h2p_dim     = nocc_active * nocc_active * nvir;
    const int total_dim   = h_dim + h2p_dim;

    if (nocc_active <= 0 || nvir <= 0) {
        throw std::runtime_error(
            "IP-EOM-CCSD: invalid orbital partition (nocc_active or nvir <= 0)");
    }

    const int verbose = rhf.get_ip_eom_verbose();

    // Sub-phase 1.11 — spin_type warn-and-ignore. IP from closed-shell RHF is
    // doublet (N-1 electron, S=1/2); singlet/triplet make no sense here. If the
    // user passed --spin_type triplet (or anything other than the default
    // "singlet"), emit a warning and proceed as doublet.
    {
        const std::string& st = rhf.get_spin_type();
        if (st != "singlet") {
            std::cout << "Warning: IP-EOM-CCSD from a closed-shell RHF reference always yields "
                         "doublet (N-1 electron) states. --spin_type \"" << st
                      << "\" is ignored." << std::endl;
        }
    }

    // CIS-NTO hand-off (active mode). If absent, fall through to passive mode
    // where we request `n_roots_requested` lowest IPs without FollowCIS.
    const CISNTOResult& cis_nto = rhf.get_cis_nto_result();
    const bool active_mode = (cis_nto.n_act_occ > 0) && rhf.get_ip_eom_followcis();
    const int safety_margin = rhf.get_ip_eom_safety_margin();
    int n_roots_to_compute = active_mode
        ? std::max(n_roots_requested, cis_nto.n_act_occ + safety_margin)
        : n_roots_requested;
    if (n_roots_to_compute > h_dim) {
        std::cout << "Warning: Requested " << n_roots_to_compute << " IP roots but 1h dim is "
                  << h_dim << ". Reducing to " << h_dim << "." << std::endl;
        n_roots_to_compute = h_dim;
    }

    if (active_mode && cis_nto.nocc_active != nocc_active) {
        throw std::runtime_error(
            "IP-EOM-CCSD: CISNTOResult.nocc_active ("
            + std::to_string(cis_nto.nocc_active)
            + ") does not match IP-EOM nocc_active ("
            + std::to_string(nocc_active)
            + ") — was the same frozen-core partition used for both?");
    }

    std::cout << "\n---- IP-EOM-CCSD ----  "
              << "nocc=" << nocc_active;
    if (num_frozen > 0) std::cout << " (frozen=" << num_frozen << ")";
    std::cout << ", nvir=" << nvir
              << ", h_dim=" << h_dim << ", h2p_dim=" << h2p_dim
              << ", total=" << total_dim
              << ", nroots=" << n_roots_to_compute
              << (active_mode ? "  (active mode)" : "  (passive mode)")
              << std::endl;

    // Step 1: CCSD ground state → T1, T2 (device-owned, transferred to operator)
    Timer ccsd_timer;
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies   = rhf.get_orbital_energies();
    const real_t* d_C   = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    real_t E_CCSD = 0.0;
    if (rhf.use_dlpno_amplitudes()) {
        // Hybrid DLPNO-STEOM (P5b): use DLPNO-CCSD T1/T2 back-transformed to
        // canonical instead of a fresh canonical CCSD solve. Hand the operator
        // its own device copy (it takes ownership + frees in its destructor).
        const BTAmplitudes& bt = rhf.get_dlpno_bt_amplitudes();
        if (bt.nocc != nocc_active || bt.nvir != nvir)
            throw std::runtime_error("IP-EOM-CCSD: DLPNO amplitude dims ("
                + std::to_string(bt.nocc) + "," + std::to_string(bt.nvir)
                + ") mismatch active space (" + std::to_string(nocc_active)
                + "," + std::to_string(nvir) + ").");
        const size_t t1n = (size_t)nocc_active * nvir;
        const size_t t2n = (size_t)nocc_active * nocc_active * nvir * nvir;
        tracked_cudaMalloc(&d_t1, t1n * sizeof(real_t));
        tracked_cudaMalloc(&d_t2, t2n * sizeof(real_t));
        cudaMemcpy(d_t1, bt.T1.data(), t1n * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t2, bt.T2.data(), t2n * sizeof(real_t), cudaMemcpyHostToDevice);
        E_CCSD = rhf.get_post_hf_energy();  // DLPNO-CCSD energy (set by stage 1)
        std::cout << "  Using DLPNO back-transformed canonical T1/T2 (hybrid bt-PNO-STEOM P5b)." << std::endl;
    } else {
        E_CCSD = ccsd_spatial_orbital(
            d_eri_ao, d_C, d_eps, num_basis, full_occ,
            /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
            &d_t1, &d_t2,
            d_eri_mo_precomputed,
            num_frozen);

        std::cout << "  CCSD correlation energy: " << std::fixed << std::setprecision(10)
                  << E_CCSD << " Ha   (in " << std::setprecision(3)
                  << ccsd_timer.elapsed_seconds() << " s)" << std::endl;
        rhf.set_post_hf_energy(E_CCSD);
    }

    // Step 2: Build MO ERI (matches EE-EOM pattern) and trim for frozen core.
    // Phase 0: eri_block_src (single-GPU RI DLPNO) builds the operator's blocks
    // on the fly from B_mo — skip the full nmo⁴ tensor entirely.
    Timer mo_timer;
    real_t* d_eri_mo = nullptr;
    bool free_eri_mo = false;
    if (!eri_block_src) {
        if (d_eri_mo_precomputed) {
            d_eri_mo = d_eri_mo_precomputed;
        } else {
            tracked_cudaMalloc(&d_eri_mo,
                               (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
            transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
            free_eri_mo = true;
        }
    }
    std::cout << "  MO transform time: " << std::fixed << std::setprecision(3)
              << mo_timer.elapsed_seconds() << " s" << std::endl;

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
    const real_t* d_eps_for_op = (num_frozen > 0) ? d_eps + num_frozen : d_eps;

    // Step 3: Build IPEOMCCSDOperator. Sub-phase 1.2: ERI blocks + dressed
    // intermediates ARE now built in the constructor; apply() still uses the
    // diagonal-only matvec until sub-phases 1.3-1.6 wire in the full sigma.
    Timer build_timer;
    // Stage IP-5: multi-GPU IP σ opt-in via GANSU_STEOM_EOM_GPUS=N>1 (shared with EA),
    // decoupled from the RI/CIS-NTO --num_gpus (forced to 1). Default unset → single-GPU.
    int eom_gpus = 1;
    if (const char* e = std::getenv("GANSU_STEOM_EOM_GPUS"))
        if (e[0]) eom_gpus = std::max(1, std::atoi(e));
    IPEOMCCSDOperator ip_op(d_eri_for_op, d_eps_for_op,
                            d_t1, d_t2,
                            nocc_active, nvir, nao_active,
                            eri_block_src, d_B_mo_blocks, num_basis, eom_gpus,
                            // (A) shared bar-H: publish 8 IP-side intermediates
                            rhf.steom_share_barh() ? &rhf.steom_barh_cache() : nullptr);

    // The intermediates have been built; we no longer need the trimmed /
    // full MO ERI tensor (the operator owns the extracted sub-blocks).
    if (free_eri_for_op) tracked_cudaFree(d_eri_for_op);
    if (free_eri_mo)     tracked_cudaFree(d_eri_mo);

    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s" << std::endl;
    if (verbose >= 2) ip_op.print_intermediate_norms(std::cout);

    // Step 3: Non-Hermitian Davidson
    Timer solve_timer;
    DavidsonConfig config;
    config.num_eigenvalues       = n_roots_to_compute;
    config.convergence_threshold = rhf.get_ip_eom_d_tol();
    config.max_subspace_size     = std::min(total_dim, std::max(80, 20 * n_roots_to_compute));
    config.max_iterations        = rhf.get_ip_eom_max_iter();
    config.use_preconditioner    = true;
    config.symmetric             = false;   // bar-H is non-Hermitian (STEOM.md §14.1)
    // IP eigenvalues are physical IPs (positive Ha) — NO spurious near-zero
    // root to filter out, so leave min_eigenvalue at default 0.
    config.min_eigenvalue        = 0.0;
    // Davidson per-iter progress is always printed (one line/iter) so the
    // user sees eigenvalue stabilisation + residual + ETA during the long
    // native-operator silent stretch (anthracene-scale: 10-30 min/solve).
    config.verbose               = (verbose >= 2) ? 2 : 1;

    std::vector<real_t> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_roots_to_compute * total_dim, 0.0);
    bool converged = false;

    if (rhf.use_dlpno_projected_eom() || rhf.use_dlpno_native_eom()) {
        // bt-PNO-STEOM stage B: DLPNO-IP-EOM in the per-pair PNO-packed space.
        // Davidson runs on either the project-up reference (B1b) or the NATIVE
        // per-pair σ operator (B-a, env GANSU_DLPNO_NATIVE_IP); both wrap the
        // canonical ip_op and produce packed roots, which are back-transformed
        // to canonical so the downstream root selection (FollowCIS / %singles)
        // is byte-for-byte unchanged. (Native σ == projected σ to machine
        // epsilon, validated by GANSU_DLPNO_IP_NATIVE_VALIDATE.)
        const bool native = rhf.use_dlpno_native_eom();
        // Frozen core: the canonical ip_op above is already built in the active space
        // (ERI trimmed + d_eps offset by num_frozen). The native/projected path's own
        // orbital data must match that convention — virtuals are MO columns
        // [full_occ, nao), active-occ energies are eps[num_frozen + i]. full_occ =
        // nocc_active + num_frozen. (num_frozen==0 ⇒ byte-unchanged.)
        const int full_occ_ip = nocc_active + num_frozen;
        const DLPNOLMP2Result& dres = rhf.get_dlpno_res();
        if (dres.pairs.empty())
            throw std::runtime_error("DLPNO IP-EOM: DLPNO pair state not stowed (set_dlpno_res).");

        DeviceHostMatrix<real_t>& Cmat = rhf.get_coefficient_matrix();
        Cmat.toHost();
        const real_t* C_full = Cmat.host_ptr();
        std::vector<real_t> C_vir((size_t)num_basis * nvir, 0.0);
        for (int mu = 0; mu < num_basis; ++mu)
            for (int a = 0; a < nvir; ++a)
                C_vir[(size_t)mu * nvir + a] = C_full[(size_t)mu * num_basis + (full_occ_ip + a)];
        rhf.get_overlap_matrix().toHost();
        const real_t* h_S = rhf.get_overlap_matrix().host_ptr();
        orbital_energies.toHost();
        const real_t* h_eps = orbital_energies.host_ptr();
        std::vector<real_t> eps_o(nocc_active);
        for (int i = 0; i < nocc_active; ++i) eps_o[i] = h_eps[num_frozen + i];

        const DLPNOIPPacking pack = build_ip_packing(dres);
        std::unique_ptr<LinearOperator> dlpno_op;
        // Auto-policy (B-a.6h): run the SOLVE single-GPU when the operator fits on device 0
        // (the grouped single-GPU solve is 3-5× faster than the multi-GPU slab solve at
        // small/medium scale; multi-GPU only pays off when the operator does not fit). The IP
        // operator has no nvir⁴ buffer, so its footprint is the nocc²·nvir² packs. Override:
        // GANSU_DLPNO_NATIVE_EOM_SOLVE1=1 force single / =0 force multi.
        int eom_solve_gpus = rhf.get_num_gpus();
#ifndef GANSU_CPU_ONLY
        if (native && eom_solve_gpus > 1) {
            const char* e = std::getenv("GANSU_DLPNO_NATIVE_EOM_SOLVE1");
            const int forced = e ? (e[0] == '0' ? -1 : 1) : 0;
            if (forced == 1) {
                eom_solve_gpus = 1;
                std::cout << "[bt-PNO auto-solve IP] forced single-GPU grouped solve "
                          << "(GANSU_DLPNO_NATIVE_EOM_SOLVE1)" << std::endl;
            } else if (forced == 0) {
                const size_t nv = static_cast<size_t>(nvir), no = static_cast<size_t>(nocc_active);
                const size_t est = (6 * no * no * nv * nv) * sizeof(real_t)
                                 + static_cast<size_t>(1) * 1024 * 1024 * 1024;  // packs + 1 GB
                int cur = 0; cudaGetDevice(&cur); cudaSetDevice(0);
                size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb); cudaSetDevice(cur);
                if (est < static_cast<size_t>(freeb * 0.7)) {
                    eom_solve_gpus = 1;
                    std::cout << "[bt-PNO auto-solve IP] operator fits on GPU 0 (est "
                              << std::fixed << std::setprecision(1) << est / 1e9 << " GB < 0.7×free "
                              << freeb * 0.7 / 1e9 << " GB) → single-GPU grouped solve" << std::endl;
                } else {
                    std::cout << "[bt-PNO auto-solve IP] operator too large for GPU 0 (est "
                              << std::fixed << std::setprecision(1) << est / 1e9
                              << " GB) → multi-GPU slab solve (" << eom_solve_gpus << " GPUs)" << std::endl;
                }
            }
        }
#endif
        if (native)
            dlpno_op = std::make_unique<DLPNOIPEOMNativeOperator>(
                ip_op, dres, pack, dres.U_loc, C_vir, h_S, num_basis, nvir, eps_o,
                eom_solve_gpus);
        else
            dlpno_op = std::make_unique<DLPNOIPEOMProjectedOperator>(
                ip_op, dres, pack, dres.U_loc, C_vir, h_S, num_basis, nvir, eps_o);
        LinearOperator& dop = *dlpno_op;

        DavidsonConfig pcfg = config;
        pcfg.max_subspace_size = std::min(dop.dimension(), std::max(80, 20 * n_roots_to_compute));
        DavidsonSolver solver(dop, pcfg);
        converged = solver.solve();
        eigenvalues = solver.get_eigenvalues();

        const int pdim = dop.dimension();
        std::vector<real_t> packed_evec((size_t)n_roots_to_compute * pdim);
        solver.copy_eigenvectors_to_host(packed_evec.data());
        for (int k = 0; k < n_roots_to_compute; ++k) {
            const real_t* ev = packed_evec.data() + (size_t)k * pdim;
            std::copy(ev, ev + h_dim, h_eigenvectors.begin() + (size_t)k * total_dim);  // R1
            std::vector<real_t> packed_r2(ev + h_dim, ev + pdim);
            std::vector<real_t> R2_canon = ip_packed_r2_to_canonical(
                dres, pack, dres.U_loc, C_vir, h_S, num_basis, nvir, packed_r2);
            std::copy(R2_canon.begin(), R2_canon.end(),
                      h_eigenvectors.begin() + (size_t)k * total_dim + h_dim);
        }
        std::cout << "  DLPNO-" << (native ? "native" : "projected") << " IP-EOM (packed_dim=" << pdim
                  << ", canon_dim=" << total_dim << ")." << std::endl;
    } else {
        DavidsonSolver solver(ip_op, config);
        converged = solver.solve();
        eigenvalues = solver.get_eigenvalues();
        solver.copy_eigenvectors_to_host(h_eigenvectors.data());
    }
    if (!converged) {
        std::cout << "Warning: IP-EOM-CCSD Davidson did not converge for all roots." << std::endl;
    }

    std::cout << "  IP-EOM-CCSD solve time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

    // Step 4: Build per-root candidates (one PerRoot per Davidson root,
    // with %singles and the raw R1 / R2 already populated). Routing into
    // per_active vs. auxiliary happens after this loop.
    const real_t ip_thresh = rhf.get_ip_eom_ip_thresh();
    std::vector<IPEOMResult::PerRoot> all_roots(n_roots_to_compute);
    for (int k = 0; k < n_roots_to_compute; ++k) {
        IPEOMResult::PerRoot& pr = all_roots[k];
        pr.omega = eigenvalues[k];
        pr.R1.assign(&h_eigenvectors[(size_t)k * total_dim],
                     &h_eigenvectors[(size_t)k * total_dim + h_dim]);
        pr.R2.assign(&h_eigenvectors[(size_t)k * total_dim + h_dim],
                     &h_eigenvectors[(size_t)k * total_dim + total_dim]);
        real_t n1 = 0.0, n2 = 0.0;
        for (real_t v : pr.R1) n1 += v * v;
        for (real_t v : pr.R2) n2 += v * v;
        pr.percent_singles     = (n1 + n2 > 0.0) ? n1 / (n1 + n2) : real_t(0.0);
        pr.followcis_overlap   = 0.0;
        pr.canonical_occ_label = -1;
    }

    // Sub-phase 1.9 + 1.10 + 1.13 — FollowCIS overlap selector + %singles
    // filter + active-root routing. The matching logic is in
    // `select_active_ip_roots` (src/ip_eom_routing.cpp); the driver only
    // copies the chosen roots into per_active and the leftovers into
    // auxiliary.
    IPEOMResult result;
    result.nocc_active = nocc_active;
    result.nvir        = nvir;
    result.num_frozen  = num_frozen;
    result.n_active    = active_mode ? cis_nto.n_act_occ : 0;

    IPEOMRoutingDecision decision;
    if (active_mode) {
        decision = select_active_ip_roots(cis_nto, all_roots, nocc_active, ip_thresh);

        // Write back follow-up tags onto the chosen roots and populate
        // per_active in NTO order (m̃ = 0 .. n_act_occ-1). Unassigned slots
        // remain as -1 sentinels for downstream STEOM consumers.
        const int n_act_occ = cis_nto.n_act_occ;
        result.per_active.resize(n_act_occ);
        for (int m = 0; m < n_act_occ; ++m) {
            const int k = decision.assigned_root_for_m[m];
            if (k >= 0) {
                all_roots[k].followcis_overlap   = decision.overlap_for_m[m];
                all_roots[k].canonical_occ_label = m;
                result.per_active[m] = all_roots[k];
            } else {
                result.per_active[m].omega = 0.0;
                result.per_active[m].percent_singles    = 0.0;
                result.per_active[m].followcis_overlap  = 0.0;
                result.per_active[m].canonical_occ_label = -1;
                result.per_active[m].R1.assign(nocc_active, 0.0);
                result.per_active[m].R2.assign((size_t)nocc_active * nocc_active * nvir, 0.0);
            }
        }
    } else {
        decision.root_taken.assign(n_roots_to_compute, false);
    }

    // Whatever wasn't assigned to an active NTO goes into auxiliary (still
    // sorted by ascending omega, matching Davidson's natural order).
    for (int k = 0; k < n_roots_to_compute; ++k) {
        if (!decision.root_taken[k]) result.auxiliary.push_back(all_roots[k]);
    }

    // Step 5: human-readable report
    std::ostringstream os;
    os << "[IP-EOM-CCSD] canonical PySCF IP-EOM matvec (Loo, Lvv, Wooov, Wovov, Wovvo, Woooo, Wovoo + 1h↔2h1p)\n"
       << "  CCSD correlation = " << std::setprecision(10) << std::fixed << E_CCSD << " Ha\n";
    if (active_mode) {
        os << "  Active NTOs (m̃ = 0.." << (cis_nto.n_act_occ - 1)
           << "), %singles threshold = " << std::setprecision(3) << ip_thresh << "\n"
           << "  Active root assignment (FollowCIS overlap + %singles filter):\n"
           << "   m̃   omega (Ha)        omega (eV)      %singles   overlap   root\n";
        for (int m = 0; m < cis_nto.n_act_occ; ++m) {
            const auto& pr = result.per_active[m];
            os << "  " << std::setw(2) << m;
            if (pr.canonical_occ_label < 0) {
                os << "   (no Davidson root passed %singles ≥ " << std::setprecision(3) << ip_thresh
                   << " — try lowering ip_eom_ip_thresh or raising ip_eom_safety_margin)\n";
            } else {
                os << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
                   << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114)
                   << "    " << std::setw(6) << std::setprecision(4) << pr.percent_singles
                   << "   " << std::setw(7) << std::setprecision(4) << pr.followcis_overlap
                   << "    "  << decision.assigned_root_for_m[m] << "\n";
            }
        }
        if (!result.auxiliary.empty()) {
            os << "  Auxiliary IP roots (Davidson found, not routed to an active NTO):\n"
               << "   k   omega (Ha)        omega (eV)      %singles\n";
            for (int k = 0; k < (int)result.auxiliary.size(); ++k) {
                const auto& pr = result.auxiliary[k];
                os << "  " << std::setw(2) << k
                   << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
                   << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114)
                   << "    " << std::setw(6) << std::setprecision(4) << pr.percent_singles
                   << "\n";
            }
        }
    } else {
        os << "  IP roots (sorted ascending; target = PySCF eom_rccsd.EOMIP 1 mHa agreement):\n"
           << "   k   omega (Ha)        omega (eV)      %singles\n";
        for (int k = 0; k < (int)result.auxiliary.size(); ++k) {
            const auto& pr = result.auxiliary[k];
            os << "  " << std::setw(2) << k
               << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
               << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114)
               << "    " << std::setw(6) << std::setprecision(4) << pr.percent_singles
               << "\n";
        }
    }
    result.report = os.str();
    std::cout << result.report;

    rhf.append_excited_state_report(result.report);
    rhf.set_ip_eom_result(std::move(result));
}

void ERI_Stored_RHF::compute_ip_eom_ccsd(int n_states) {
    compute_ip_eom_ccsd_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

// bt-PNO-STEOM P4 (RI path): build the full MO ERI from the RI B factors
// (matching RI-CCSD), then run the identical canonical IP-EOM-CCSD impl. The
// only difference vs the stored path is that T1/T2 + bar-H come from the
// RI-approximated integrals, so IP roots match canonical IP-EOM to the RI
// fitting tolerance (~1e-4 Ha). build_mo_eri produces the same chemist's-
// notation [nmo⁴] tensor layout as transform_ao_eri_to_mo_eri_full.
void ERI_RI_RHF::compute_ip_eom_ccsd(int n_states) {
    const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
    const int num_basis = rhf_.get_num_basis();
    // P4b: RI DLPNO builds the operator's MO-ERI blocks on the fly from B_mo
    // (no full nmo⁴). Single-GPU uses intermediate_matrix_B_; distributed-RI's
    // build_B_mo lazily replicates B to each GPU (~580 MB at anthracene scale).
    // build_B_mo returns nullptr when budget exceeded → fallback to full tensor.
    const bool want_block = rhf_.use_dlpno_amplitudes()
                            && rhf_.get_num_frozen_core() == 0
                            && gpu::gpu_available();
    const real_t* d_B_mo = want_block ? build_B_mo(d_C, num_basis) : nullptr;
    const bool block     = want_block && d_B_mo != nullptr;
    real_t* d_eri_mo     = block ? nullptr : build_mo_eri(d_C, num_basis);
    compute_ip_eom_ccsd_impl(rhf_, /*d_eri_ao=*/nullptr, n_states, d_eri_mo,
                             block ? this : nullptr, d_B_mo);
    if (d_eri_mo) tracked_cudaFree(d_eri_mo);
}

} // namespace gansu
