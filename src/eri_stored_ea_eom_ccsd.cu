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
 * @file eri_stored_ea_eom_ccsd.cu
 * @brief EA-EOM-CCSD driver (bt-PNO-STEOM Phase P2 sub-phase 2.0+2.1).
 *
 * Mirrors `eri_stored_ip_eom_ccsd.cu`. Sub-phase 2.0+2.1 ships a smoke-test
 * scaffold:
 *  - operator.apply() is diagonal-only, so eigenvalues equal the sorted
 *    Koopmans EA values (+ε_a for the 1p sector, -ε_j+ε_a+ε_b for the
 *    2p1h sector).
 *  - FollowCIS / %singles filter / EAEOMResult.per_active are wired but
 *    populated only when CISNTOResult is present; otherwise we fall
 *    through to n_excited_states lowest EAs (passive mode).
 *  - Full bar-H matvec lands in sub-phases 2.2-2.6.
 */

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "rhf.hpp"
#include "ea_eom_ccsd_operator.hpp"
#include "ea_eom_result.hpp"
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

extern __global__ void trim_eri_frozen_core_kernel(const real_t* __restrict__ eri_full,
                                                    real_t* __restrict__ eri_trimmed,
                                                    int N_full, int na_active, int offset);


static void compute_ea_eom_ccsd_impl(RHF& rhf,
                                     const real_t* d_eri_ao,
                                     int n_roots_requested,
                                     real_t* d_eri_mo_precomputed = nullptr)
{
    PROFILE_FUNCTION();

    const int num_frozen  = rhf.get_num_frozen_core();
    const int num_basis   = rhf.get_num_basis();
    const int full_occ    = rhf.get_num_electrons() / 2;
    const int nocc_active = full_occ - num_frozen;
    const int nvir        = num_basis - full_occ;
    const int nao_active  = nocc_active + nvir;
    const int p_dim       = nvir;
    const int p2h_dim     = nocc_active * nvir * nvir;
    const int total_dim   = p_dim + p2h_dim;

    if (nocc_active <= 0 || nvir <= 0) {
        throw std::runtime_error(
            "EA-EOM-CCSD: invalid orbital partition (nocc_active or nvir <= 0)");
    }

    const int verbose = rhf.get_ea_eom_verbose();

    // Sub-phase 2.11 — spin_type warn-and-ignore. EA from a closed-shell RHF
    // reference is doublet (N+1 electron, S=1/2); singlet/triplet make no
    // sense here. If the user passed --spin_type triplet (or anything other
    // than the default "singlet"), emit a warning and proceed as doublet.
    {
        const std::string& st = rhf.get_spin_type();
        if (st != "singlet") {
            std::cout << "Warning: EA-EOM-CCSD from a closed-shell RHF reference always yields "
                         "doublet (N+1 electron) states. --spin_type \"" << st
                      << "\" is ignored." << std::endl;
        }
    }

    // CIS-NTO hand-off (active mode). If absent, fall through to passive mode.
    const CISNTOResult& cis_nto = rhf.get_cis_nto_result();
    const bool active_mode = (cis_nto.n_act_vir > 0) && rhf.get_ea_eom_followcis();
    const int safety_margin = rhf.get_ea_eom_safety_margin();
    int n_roots_to_compute = active_mode
        ? std::max(n_roots_requested, cis_nto.n_act_vir + safety_margin)
        : n_roots_requested;
    if (n_roots_to_compute > p_dim) {
        std::cout << "Warning: Requested " << n_roots_to_compute << " EA roots but 1p dim is "
                  << p_dim << ". Reducing to " << p_dim << "." << std::endl;
        n_roots_to_compute = p_dim;
    }

    std::cout << "\n---- EA-EOM-CCSD ----  "
              << "nocc=" << nocc_active;
    if (num_frozen > 0) std::cout << " (frozen=" << num_frozen << ")";
    std::cout << ", nvir=" << nvir
              << ", p_dim=" << p_dim << ", p2h_dim=" << p2h_dim
              << ", total=" << total_dim
              << ", nroots=" << n_roots_to_compute
              << (active_mode ? "  (active mode)" : "  (passive mode)")
              << std::endl;

    if (active_mode && cis_nto.nvir != nvir) {
        throw std::runtime_error(
            "EA-EOM-CCSD: CISNTOResult.nvir ("
            + std::to_string(cis_nto.nvir)
            + ") does not match EA-EOM nvir ("
            + std::to_string(nvir)
            + ") — was the same frozen-core / basis partition used for both?");
    }

    // Step 1: CCSD ground state → T1, T2
    Timer ccsd_timer;
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies   = rhf.get_orbital_energies();
    const real_t* d_C   = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    real_t E_CCSD = ccsd_spatial_orbital(
        d_eri_ao, d_C, d_eps, num_basis, full_occ,
        /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
        &d_t1, &d_t2,
        d_eri_mo_precomputed,
        num_frozen);

    std::cout << "  CCSD correlation energy: " << std::fixed << std::setprecision(10)
              << E_CCSD << " Ha   (in " << std::setprecision(3)
              << ccsd_timer.elapsed_seconds() << " s)" << std::endl;
    rhf.set_post_hf_energy(E_CCSD);

    // Step 2: Build MO ERI (matches EE-EOM pattern) and trim for frozen core.
    Timer mo_timer;
    real_t* d_eri_mo = nullptr;
    bool free_eri_mo = false;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
    } else {
        tracked_cudaMalloc(&d_eri_mo,
                           (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
        free_eri_mo = true;
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

    // Step 3: Build EAEOMCCSDOperator. Sub-phase 2.2: ERI blocks +
    // dressed intermediates ARE now built in the constructor; apply() still
    // uses the diagonal-only matvec until sub-phases 2.3-2.6 wire in the
    // full sigma.
    Timer build_timer;
    EAEOMCCSDOperator ea_op(d_eri_for_op, d_eps_for_op,
                            d_t1, d_t2,
                            nocc_active, nvir, nao_active);

    if (free_eri_for_op) tracked_cudaFree(d_eri_for_op);
    if (free_eri_mo)     tracked_cudaFree(d_eri_mo);

    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s" << std::endl;
    if (verbose >= 2) ea_op.print_intermediate_norms(std::cout);

    // Step 4: Non-Hermitian Davidson (EA bar-H is also non-Hermitian)
    Timer solve_timer;
    DavidsonConfig config;
    config.num_eigenvalues       = n_roots_to_compute;
    config.convergence_threshold = rhf.get_ea_eom_d_tol();
    config.max_subspace_size     = std::min(total_dim, std::max(80, 20 * n_roots_to_compute));
    config.max_iterations        = rhf.get_ea_eom_max_iter();
    config.use_preconditioner    = true;
    config.symmetric             = false;
    config.min_eigenvalue        = 0.0;
    config.verbose               = (verbose >= 2) ? 2 : (verbose >= 1 ? 1 : 0);

    DavidsonSolver solver(ea_op, config);
    bool converged = solver.solve();
    if (!converged) {
        std::cout << "Warning: EA-EOM-CCSD Davidson did not converge for all roots." << std::endl;
    }

    const auto& eigenvalues = solver.get_eigenvalues();
    std::vector<real_t> h_eigenvectors((size_t)n_roots_to_compute * total_dim);
    solver.copy_eigenvectors_to_host(h_eigenvectors.data());

    std::cout << "  EA-EOM-CCSD solve time: " << std::fixed << std::setprecision(3)
              << solve_timer.elapsed_seconds() << " s" << std::endl;

    // Step 5: Build per-root candidates (one PerRoot per Davidson root, with
    // %singles and the raw R1 / R2 already populated). Routing into
    // per_active vs. auxiliary happens after this loop.
    const real_t ea_thresh = rhf.get_ea_eom_ea_thresh();
    std::vector<EAEOMResult::PerRoot> all_roots(n_roots_to_compute);
    for (int k = 0; k < n_roots_to_compute; ++k) {
        EAEOMResult::PerRoot& pr = all_roots[k];
        pr.omega = eigenvalues[k];
        pr.R1.assign(&h_eigenvectors[(size_t)k * total_dim],
                     &h_eigenvectors[(size_t)k * total_dim + p_dim]);
        pr.R2.assign(&h_eigenvectors[(size_t)k * total_dim + p_dim],
                     &h_eigenvectors[(size_t)k * total_dim + total_dim]);
        real_t n1 = 0.0, n2 = 0.0;
        for (real_t v : pr.R1) n1 += v * v;
        for (real_t v : pr.R2) n2 += v * v;
        pr.percent_singles     = (n1 + n2 > 0.0) ? n1 / (n1 + n2) : real_t(0.0);
        pr.followcis_overlap   = 0.0;
        pr.canonical_vir_label = -1;
    }

    // Sub-phase 2.9 + 2.10 + 2.13 — FollowCIS overlap selector + %singles
    // filter + active-root routing. Matching logic lives in
    // `select_active_ea_roots` (src/ea_eom_routing.cpp); the driver only
    // copies the chosen roots into per_active and the leftovers into
    // auxiliary.
    EAEOMResult result;
    result.nocc_active = nocc_active;
    result.nvir        = nvir;
    result.num_frozen  = num_frozen;
    result.n_active    = active_mode ? cis_nto.n_act_vir : 0;

    EAEOMRoutingDecision decision;
    if (active_mode) {
        decision = select_active_ea_roots(cis_nto, all_roots, nvir, ea_thresh);

        const int n_act_vir = cis_nto.n_act_vir;
        result.per_active.resize(n_act_vir);
        for (int m = 0; m < n_act_vir; ++m) {
            const int k = decision.assigned_root_for_a[m];
            if (k >= 0) {
                all_roots[k].followcis_overlap   = decision.overlap_for_a[m];
                all_roots[k].canonical_vir_label = m;
                result.per_active[m] = all_roots[k];
            } else {
                result.per_active[m].omega = 0.0;
                result.per_active[m].percent_singles    = 0.0;
                result.per_active[m].followcis_overlap  = 0.0;
                result.per_active[m].canonical_vir_label = -1;
                result.per_active[m].R1.assign(nvir, 0.0);
                result.per_active[m].R2.assign((size_t)nocc_active * nvir * nvir, 0.0);
            }
        }
    } else {
        decision.root_taken.assign(n_roots_to_compute, false);
    }

    for (int k = 0; k < n_roots_to_compute; ++k) {
        if (!decision.root_taken[k]) result.auxiliary.push_back(all_roots[k]);
    }

    // Step 6: human-readable report
    std::ostringstream os;
    os << "[EA-EOM-CCSD] sub-phase 2.3-2.6 — full PySCF EA matvec (Lvv, Fov, Wvovv, Wvvvo, Wovov, Wovvo, Wvvvv + tmp·t2)\n"
       << "  CCSD correlation = " << std::setprecision(10) << std::fixed << E_CCSD << " Ha\n";
    if (active_mode) {
        os << "  Active virtual NTOs (ã = 0.." << (cis_nto.n_act_vir - 1)
           << "), %singles threshold = " << std::setprecision(3) << ea_thresh << "\n"
           << "  Active root assignment (FollowCIS overlap + %singles filter):\n"
           << "   ã   omega (Ha)        omega (eV)      %singles   overlap   root\n";
        for (int m = 0; m < cis_nto.n_act_vir; ++m) {
            const auto& pr = result.per_active[m];
            os << "  " << std::setw(2) << m;
            if (pr.canonical_vir_label < 0) {
                os << "   (no Davidson root passed %singles ≥ " << std::setprecision(3) << ea_thresh
                   << " — try lowering ea_eom_ea_thresh or raising ea_eom_safety_margin)\n";
            } else {
                os << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
                   << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114)
                   << "    " << std::setw(6) << std::setprecision(4) << pr.percent_singles
                   << "   " << std::setw(7) << std::setprecision(4) << pr.followcis_overlap
                   << "    " << decision.assigned_root_for_a[m] << "\n";
            }
        }
        if (!result.auxiliary.empty()) {
            os << "  Auxiliary EA roots (Davidson found, not routed to an active virtual NTO):\n"
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
        os << "  EA roots (sorted ascending; target = PySCF eom_rccsd.EOMEA 1 mHa agreement):\n"
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
    rhf.set_ea_eom_result(std::move(result));
}

void ERI_Stored_RHF::compute_ea_eom_ccsd(int n_states) {
    compute_ea_eom_ccsd_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

} // namespace gansu
