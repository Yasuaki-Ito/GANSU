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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "rhf.hpp"
#include "dlpno_ea_packing.hpp"                // bt-PNO-STEOM stage B: EA packed layout
#include "dlpno_ea_eom_projected_operator.hpp" // stage B: projected DLPNO-EA operator
#include "dlpno_ea_eom_native_operator.hpp"    // stage B (a): native per-pair DLPNO-EA σ operator
#include "dlpno_ea_eom_transform.hpp"          // stage B: packed↔canonical EA R2 transform
#include <memory>
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

    // Device buffer audit (diagnostic for Pentacene-class OOM at d_eri_vvvv_
    // alloc: prior IP-EOM operator destructor SHOULD have released its
    // d_eri_vvvv_ + native operator state, but the persistent DLPNO state
    // (T2 amplitudes + bt-PNO PI cache + B replica) survives across all 3
    // (IP/EA/STEOM) consumers.  This audit prints per-device free/used
    // memory + the global tracked-malloc counter, so the gap between what
    // we EXPECT to be freed and what's actually still alive is visible
    // before we attempt the NV⁴·8B = 91 GB d_eri_vvvv_ alloc at NV=327.
    // Always-on (1 short paragraph per run, low log noise).
    //
    // Ship 11 — operator device-redirect: if GANSU_STEOM_OPERATOR_DEVICE_BALANCING=1
    // is set, after the audit pick the device with maximum driver-free memory and
    // cudaSetDevice to it BEFORE we proceed to CCSD/MO transform/operator ctor.
    // This addresses the Pentacene GPU-0 bias (DLPNO ground state's heaviest T2
    // slab persists on device 0 while devices 1-3 sit ~131 GB free). Operator
    // tracked allocs land on the chosen device; restored at impl exit.
    bool dev_balance_active = false;
    int dev_balance_restore = 0;
    int dev_balance_target  = 0;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        // Global tracker snapshot (sum across all devices, via the GANSU memory tracker).
        const size_t tracked_global = GlobalGpuMemoryTracker::get_current();
        const size_t tracked_peak = GlobalGpuMemoryTracker::get_peak();
        const size_t nvir_sz = (size_t)nvir * nvir * nvir * nvir * sizeof(real_t);
        const double need_vvvv_gb = nvir_sz / (1024.0*1024.0*1024.0);
        std::cout << "  [EA-EOM device audit] tracked sum-of-GPUs current = "
                  << std::fixed << std::setprecision(2)
                  << (tracked_global / (1024.0*1024.0*1024.0)) << " GB"
                  << "   peak = " << (tracked_peak / (1024.0*1024.0*1024.0)) << " GB"
                  << "   (will alloc d_eri_vvvv_ = " << need_vvvv_gb << " GB next)"
                  << std::defaultfloat << std::endl;
        // Per-device breakdown from tracker (only devices that saw a tracked alloc).
        const auto per_dev = GlobalGpuMemoryTracker::get_per_device_snapshot();
        for (const auto& kv : per_dev) {
            const int dev = kv.first;
            const auto& s = kv.second;  // {current, total, peak}
            std::cout << "    GPU " << dev << " tracker:  current="
                      << std::fixed << std::setprecision(2)
                      << (s[0] / (1024.0*1024.0*1024.0)) << " GB"
                      << "  peak=" << (s[2] / (1024.0*1024.0*1024.0)) << " GB"
                      << std::defaultfloat << std::endl;
        }
        // Live cudaMemGetInfo per device (= ground truth from driver).
        int n_dev = 0;
        cudaGetDeviceCount(&n_dev);
        int saved_dev = 0;
        cudaGetDevice(&saved_dev);
        dev_balance_restore = saved_dev;
        std::vector<size_t> per_dev_free(n_dev, 0);
        for (int d = 0; d < n_dev; ++d) {
            cudaSetDevice(d);
            size_t free_b = 0, total_b = 0;
            if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
                per_dev_free[d] = free_b;
                const double used_gb = (total_b - free_b) / (1024.0*1024.0*1024.0);
                const double free_gb = free_b / (1024.0*1024.0*1024.0);
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
        // Ship 11: optional device-redirect based on max driver-free memory.
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
                // new device — otherwise build_dressed_intermediates DGEMMs
                // run with a handle bound to the original device while
                // pointers live on the new one, which fails with
                // CUBLAS_STATUS_EXECUTION_FAILED (status=13).
                gpu::GPUHandle::reset();
                dev_balance_active = true;
                dev_balance_target = best_dev;
                std::cout << "  [EA-EOM device-balance] redirecting operator to GPU "
                          << best_dev
                          << " (free=" << std::fixed << std::setprecision(2)
                          << (best_free / (1024.0*1024.0*1024.0)) << " GB"
                          << ", was on GPU " << saved_dev
                          << " with free=" << (per_dev_free[saved_dev] / (1024.0*1024.0*1024.0))
                          << " GB; GPUHandle rebuilt for new device)"
                          << std::defaultfloat << std::endl;
            } else {
                cudaSetDevice(saved_dev);
                std::cout << "  [EA-EOM device-balance] GPU " << saved_dev
                          << " already has max free memory; no redirect."
                          << std::endl;
            }
        } else {
            cudaSetDevice(saved_dev);
        }
    }
#endif

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
    real_t E_CCSD = 0.0;
    if (rhf.use_dlpno_amplitudes()) {
        // Hybrid DLPNO-STEOM (P5b): inject DLPNO-CCSD T1/T2 (canonical, own copy).
        const BTAmplitudes& bt = rhf.get_dlpno_bt_amplitudes();
        if (bt.nocc != nocc_active || bt.nvir != nvir)
            throw std::runtime_error("EA-EOM-CCSD: DLPNO amplitude dims ("
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
    Timer mo_timer;
    // Phase 0: eri_block_src (single-GPU RI DLPNO) builds the operator's blocks
    // on the fly from B_mo — skip the full nmo⁴ tensor entirely.
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
    // Full-tensor frozen-core trim ONLY when not using the on-the-fly block
    // path. With eri_block_src the operator reads active blocks straight from
    // the full-C B_mo via the frozen_off (num_frozen) shift — no nao⁴ tensor.
    if (num_frozen > 0 && !eri_block_src) {
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
    // Stage EA-5: multi-GPU EA σ is opt-in via GANSU_STEOM_EOM_GPUS=N>1, decoupled
    // from the RI/CIS-NTO --num_gpus (forced to 1). Default unset → single-GPU.
    int eom_gpus = 1;
    if (const char* e = std::getenv("GANSU_STEOM_EOM_GPUS"))
        if (e[0]) eom_gpus = std::max(1, std::atoi(e));

    // Ship 12: vvvv n-slab distribution.  When GANSU_EA_VVVV_NSLAB=N is set
    // and we have the P4b on-the-fly path, allocate + extract per-device
    // d_eri_vvvv slabs immediately (build_B_mo has thread-local workspace,
    // so we must extract each slab before moving to the next device).  This
    // avoids the single-device 91 GB ceiling for Pentacene (NV=327): each
    // slab ≈ NV⁴/N · 8B fits comfortably on each H200 (140 GB).  Slab
    // ownership transfers to the operator ctor below.
    std::vector<real_t*> d_eri_vvvv_slabs;
#ifndef GANSU_CPU_ONLY
    if (eri_block_src && gpu::gpu_available()) {
        const char* env_nslab = std::getenv("GANSU_EA_VVVV_NSLAB");
        if (env_nslab && env_nslab[0]) {
            const int Nreq = std::atoi(env_nslab);
            int n_dev = 0;
            cudaGetDeviceCount(&n_dev);
            const int N = std::min(Nreq, n_dev);
            if (N >= 2) {
                int saved = 0;
                cudaGetDevice(&saved);
                d_eri_vvvv_slabs.assign(N, nullptr);
                bool all_ok = true;
                for (int d = 0; d < N; ++d) {
                    cudaSetDevice(d);
                    const real_t* B_d = eri_block_src->build_B_mo(d_C, num_basis);
                    if (!B_d) {
                        all_ok = false;
                        std::cout << "  [EA-EOM Ship 12] WARNING: build_B_mo returned null "
                                     "for device " << d << "; disabling slab mode." << std::endl;
                        break;
                    }
                    const int a_start = (int)((int64_t)d * nvir / N);
                    const int a_end   = (int)((int64_t)(d + 1) * nvir / N);
                    const int an      = a_end - a_start;
                    const size_t slab_sz = (size_t)an * nvir * nvir * nvir;
                    tracked_cudaMalloc(&d_eri_vvvv_slabs[d], slab_sz * sizeof(real_t));
                    // Frozen core: virtual block starts at full_occ = nocc_active
                    // + num_frozen in the full-C B_mo (B_d built over num_basis MOs).
                    eri_block_src->mo_eri_block_into(B_d, num_basis,
                        nocc_active + num_frozen + a_start, an,   // a range (slab)
                        nocc_active + num_frozen, nvir,           // b range
                        nocc_active + num_frozen, nvir,           // c range
                        nocc_active + num_frozen, nvir,           // d range
                        d_eri_vvvv_slabs[d]);
                    cudaDeviceSynchronize();
                    std::cout << "  [EA-EOM Ship 12] slab d=" << d
                              << " allocated " << std::fixed << std::setprecision(2)
                              << (slab_sz * sizeof(real_t) / (1024.0*1024.0*1024.0))
                              << " GB on GPU " << d
                              << " (a∈[" << a_start << "," << a_end << "))"
                              << std::defaultfloat << std::endl;
                }
                cudaSetDevice(saved);
                if (!all_ok) {
                    // Free any partially-allocated slabs.
                    for (int d = 0; d < N; ++d) {
                        if (d_eri_vvvv_slabs[d]) {
                            cudaSetDevice(d);
                            tracked_cudaFree(d_eri_vvvv_slabs[d]);
                            d_eri_vvvv_slabs[d] = nullptr;
                        }
                    }
                    cudaSetDevice(saved);
                    d_eri_vvvv_slabs.clear();
                } else {
                    // Ship 12: the build_B_mo thread-local workspace cache is
                    // single-device — our per-device loop above invalidated the
                    // GPU-0 (or whichever was original) ws_B_mo and now points
                    // at the LAST device's data.  Refresh on `saved` (the Ship 11
                    // redirect target, or original device) so the operator's
                    // stored d_B_mo_blocks is valid for the other 6 d_eri_*
                    // extractions in extract_eri_blocks.
                    d_B_mo_blocks = eri_block_src->build_B_mo(d_C, num_basis);
                    if (!d_B_mo_blocks) {
                        std::cout << "  [EA-EOM Ship 12] WARNING: post-loop "
                                     "build_B_mo refresh on saved device returned null; "
                                     "operator extract_eri_blocks will likely fail."
                                  << std::endl;
                    }
                }
            } else {
                std::cout << "  [EA-EOM Ship 12] GANSU_EA_VVVV_NSLAB=" << Nreq
                          << " but only " << n_dev << " GPUs visible; needs ≥2 to slab"
                          << std::endl;
            }
        }
    }
#endif
    EAEOMCCSDOperator ea_op(d_eri_for_op, d_eps_for_op,
                            d_t1, d_t2,
                            nocc_active, nvir, nao_active,
                            eri_block_src, d_B_mo_blocks, num_basis, eom_gpus,
                            d_eri_vvvv_slabs.empty() ? nullptr : &d_eri_vvvv_slabs,
                            // (A) shared bar-H: publish 3 EA-unique intermediates
                            rhf.steom_share_barh() ? &rhf.steom_barh_cache() : nullptr,
                            // Frozen core: block ranges read [num_frozen, num_basis)
                            // of the full-C B_mo (only used on the block path).
                            num_frozen);

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
    // Davidson per-iter progress is always printed (one line/iter) so the
    // user sees eigenvalue stabilisation + residual + ETA during the long
    // native-operator silent stretch (anthracene-scale: 10-30 min/solve).
    config.verbose               = (verbose >= 2) ? 2 : 1;

    std::vector<real_t> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_roots_to_compute * total_dim, 0.0);
    bool converged = false;

    if (rhf.use_dlpno_projected_eom() || rhf.use_dlpno_native_eom()) {
        // bt-PNO-STEOM stage B: DLPNO-EA-EOM in the per-i PNO(i,i) 2p1h space,
        // via the project-up reference (B1b) or the NATIVE per-pair σ operator
        // (B-EA, env GANSU_DLPNO_NATIVE_EOM). Both wrap ea_op and produce packed
        // roots, back-transformed to canonical (downstream selection unchanged).
        // Native σ == projected σ to machine epsilon (GANSU_DLPNO_EA_NATIVE_VALIDATE).
        const bool native = rhf.use_dlpno_native_eom();
        // Frozen core (validated naphthalene cc-pVDZ, num_frozen=10): the EA path
        // is built entirely in the active space. The virtual columns / energies use
        // the [full_occ, nao) offset (full_occ_ea below), and build_ea_packing is
        // given the TRUE nvir (= num_basis - full_occ) — without that the packed 1p
        // sector was overcounted by num_frozen, producing spurious zero modes + a
        // NaN R2 back-transform. EA root0 3.0978 eV ≈ non-frozen 3.0983.
        // (num_frozen==0 ⇒ byte-unchanged.)
        const int full_occ_ea = nocc_active + num_frozen;
        const DLPNOLMP2Result& dres = rhf.get_dlpno_res();
        if (dres.pairs.empty())
            throw std::runtime_error("DLPNO EA-EOM: DLPNO pair state not stowed (set_dlpno_res).");

        DeviceHostMatrix<real_t>& Cmat = rhf.get_coefficient_matrix();
        Cmat.toHost();
        const real_t* C_full = Cmat.host_ptr();
        std::vector<real_t> C_vir((size_t)num_basis * nvir, 0.0);
        for (int mu = 0; mu < num_basis; ++mu)
            for (int a = 0; a < nvir; ++a)
                C_vir[(size_t)mu * nvir + a] = C_full[(size_t)mu * num_basis + (full_occ_ea + a)];
        rhf.get_overlap_matrix().toHost();
        const real_t* h_S = rhf.get_overlap_matrix().host_ptr();
        orbital_energies.toHost();
        const real_t* h_eps = orbital_energies.host_ptr();
        std::vector<real_t> eps_v(nvir);
        for (int a = 0; a < nvir; ++a) eps_v[a] = h_eps[full_occ_ea + a];

        // Pass the TRUE virtual count (num_basis - full_occ). With frozen core
        // dres.nao (full AO) - dres.nocc (active occ) overcounts nvir by num_frozen,
        // which would inflate the 1p sector → num_frozen spurious zero modes + an
        // out-of-bounds C_vir read (NaN) in ea_packed_r2_to_canonical.
        const DLPNOEAPacking pack = build_ea_packing(dres, nvir);
        std::unique_ptr<LinearOperator> dlpno_op;

        // Ship 13 — secondary device-balance for the DLPNO native operator +
        // Davidson workspace.  The canonical EA operator (ea_op) holds ~120 GB
        // of persistent state on the Ship 11 target device; the DLPNO native
        // operator's d_Wvvvv_pno_pack_ (Σ_j n_pno(jj)⁴·8B, ~5 GB at Pentacene)
        // plus Davidson subspace+sigma (packed_dim·max_sub·8B·2, ~9 GB at
        // Pentacene EA scale) would push that single device over the 139.8 GB
        // ceiling.  By picking a different device with maximum free memory
        // for the native operator+Davidson, we let peer-access NVLink read
        // ea_op's persistent state cross-device.  Gate by same env as Ship 11.
        bool ea_native_dev_balance_active = false;
        int  ea_native_dev_balance_restore = 0;
        int  ea_native_dev_balance_target  = 0;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            const char* env_bal = std::getenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
            if (env_bal && env_bal[0] == '1') {
                int n_dev = 0;
                cudaGetDeviceCount(&n_dev);
                if (n_dev > 1) {
                    int saved = 0;
                    cudaGetDevice(&saved);
                    ea_native_dev_balance_restore = saved;
                    std::vector<size_t> per_dev_free(n_dev, 0);
                    for (int d = 0; d < n_dev; ++d) {
                        cudaSetDevice(d);
                        size_t free_b = 0, total_b = 0;
                        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess)
                            per_dev_free[d] = free_b;
                    }
                    int best_dev = saved;
                    size_t best_free = per_dev_free[saved];
                    for (int d = 0; d < n_dev; ++d) {
                        if (per_dev_free[d] > best_free) {
                            best_free = per_dev_free[d];
                            best_dev  = d;
                        }
                    }
                    if (best_dev != saved) {
                        cudaSetDevice(best_dev);
                        // Ship 13 critical: GPUHandle is thread_local cuBLAS +
                        // cuSOLVER bound to the device current at first call
                        // (likely GPU 0).  After our cudaSetDevice the handle
                        // is still bound to the wrong device → Davidson
                        // build_subspace_matrix cublasDgemm fails with
                        // CUBLAS_STATUS_EXECUTION_FAILED (status=13).  reset()
                        // recreates the handles on the new current device.
                        // Note: it nullifies (does not cublasDestroy) the old
                        // handles — acceptable one-shot leak for a long-lived
                        // EOM solve.  EA native operator owns its own cuBLAS
                        // handle internally, so unaffected.
                        gpu::GPUHandle::reset();
                        ea_native_dev_balance_active = true;
                        ea_native_dev_balance_target = best_dev;
                        std::cout << "  [EA-EOM Ship 13 native-balance] redirecting "
                                     "DLPNO native operator + Davidson workspace to GPU "
                                  << best_dev << " (free=" << std::fixed
                                  << std::setprecision(2)
                                  << (best_free / (1024.0*1024.0*1024.0)) << " GB"
                                  << ", canonical ea_op stays on GPU " << saved
                                  << " with free=" << (per_dev_free[saved] / (1024.0*1024.0*1024.0))
                                  << " GB; GPUHandle rebuilt for new device)"
                                  << std::defaultfloat << std::endl;
                    } else {
                        cudaSetDevice(saved);
                    }
                }
            }
        }
#endif

        // Auto-policy (B-a.6h): the grouped single-GPU EOM solve is 3-5× faster than the
        // multi-GPU slab solve at small/medium scale — the per-matvec broadcast/gather/sync
        // overhead dominates the tiny per-device compute, so multi-GPU is only worth it when
        // the operator does not fit on one device. The build phases already ran multi-GPU; run
        // the SOLVE single-GPU (device 0) whenever the operator fits. Override:
        // GANSU_DLPNO_NATIVE_EOM_SOLVE1=1 force single / =0 force multi.
        int eom_solve_gpus = rhf.get_num_gpus();
#ifndef GANSU_CPU_ONLY
        if (native && eom_solve_gpus > 1) {
            const char* e = std::getenv("GANSU_DLPNO_NATIVE_EOM_SOLVE1");
            const int forced = e ? (e[0] == '0' ? -1 : 1) : 0;
            if (forced == 1) {
                eom_solve_gpus = 1;
                std::cout << "[bt-PNO auto-solve EA] forced single-GPU grouped solve "
                          << "(GANSU_DLPNO_NATIVE_EOM_SOLVE1)" << std::endl;
            } else if (forced == 0) {   // auto: balanced-device footprint vs free memory
                const size_t nv = static_cast<size_t>(nvir), no = static_cast<size_t>(nocc_active);
                const size_t est = (nv*nv*nv*nv + 4*no*no*nv*nv) * sizeof(real_t)
                                 + static_cast<size_t>(2) * 1024 * 1024 * 1024;  // d_eri_vvvv + packs + 2 GB
                // Probe the device the native operator + Davidson will ACTUALLY run
                // on, i.e. the current device — which is the Ship 13 native-balance
                // target when balancing redirected (free GPU), else the caller's
                // device. Probing GPU 0 unconditionally under-counts free memory on
                // the balanced device and wrongly forces the (fragile, cross-device)
                // multi-GPU slab solve when the operator fits on the balanced GPU.
                int probe_dev = 0; cudaGetDevice(&probe_dev);
                size_t freeb = 0, totalb = 0; cudaMemGetInfo(&freeb, &totalb);
                if (est < static_cast<size_t>(freeb * 0.7)) {
                    eom_solve_gpus = 1;
                    std::cout << "[bt-PNO auto-solve EA] operator fits on GPU " << probe_dev << " (est "
                              << std::fixed << std::setprecision(1) << est / 1e9 << " GB < 0.7×free "
                              << freeb * 0.7 / 1e9 << " GB) → single-GPU grouped solve" << std::endl;
                } else {
                    std::cout << "[bt-PNO auto-solve EA] operator too large for GPU " << probe_dev << " (est "
                              << std::fixed << std::setprecision(1) << est / 1e9
                              << " GB) → multi-GPU slab solve (" << eom_solve_gpus << " GPUs)" << std::endl;
                }
            }
        }
#endif
        if (native)
            dlpno_op = std::make_unique<DLPNOEAEOMNativeOperator>(
                ea_op, dres, pack, dres.U_loc, C_vir, h_S, num_basis, nvir, eps_v,
                eom_solve_gpus);
        else
            dlpno_op = std::make_unique<DLPNOEAEOMProjectedOperator>(
                ea_op, dres, pack, dres.U_loc, C_vir, h_S, num_basis, eps_v);
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
            std::copy(ev, ev + p_dim, h_eigenvectors.begin() + (size_t)k * total_dim);  // R1
            std::vector<real_t> packed_r2(ev + p_dim, ev + pdim);
            std::vector<real_t> R2_canon = ea_packed_r2_to_canonical(
                dres, pack, dres.U_loc, C_vir, h_S, num_basis, packed_r2);
            std::copy(R2_canon.begin(), R2_canon.end(),
                      h_eigenvectors.begin() + (size_t)k * total_dim + p_dim);
        }
        std::cout << "  DLPNO-" << (native ? "native" : "projected") << " EA-EOM (packed_dim=" << pdim
                  << ", canon_dim=" << total_dim << ")." << std::endl;
    } else {
        DavidsonSolver solver(ea_op, config);
        converged = solver.solve();
        eigenvalues = solver.get_eigenvalues();
        solver.copy_eigenvectors_to_host(h_eigenvectors.data());
    }
    if (!converged) {
        std::cout << "Warning: EA-EOM-CCSD Davidson did not converge for all roots." << std::endl;
    }

    // Deterministic phase convention per root: force the largest-magnitude
    // component of each full Davidson eigenvector (R1+R2 contiguous in
    // h_eigenvectors) positive — removes the arbitrary Davidson sign gauge on
    // R2/X. Math-inert for sign-covariant observables (mirror of the IP path).
    for (int k = 0; k < n_roots_to_compute; ++k) {
        real_t* ev = &h_eigenvectors[(size_t)k * total_dim];
        int imax = 0; real_t amax = -1.0;
        for (int i = 0; i < total_dim; ++i) {
            const real_t a = std::fabs(ev[i]);
            if (a > amax) { amax = a; imax = i; }
        }
        if (ev[imax] < 0.0)
            for (int i = 0; i < total_dim; ++i) ev[i] = -ev[i];
    }

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

    // Ship 11 — operator device-redirect: restore caller's device so that
    // any downstream code (STEOM dispatch, output, gansu_api) sees the same
    // device it set before calling EA-EOM. Operator + DLPNO native state are
    // already destructed (RAII via ea_op / dlpno_op scope), so this is a
    // pure context switch with no straggler allocs.
#ifndef GANSU_CPU_ONLY
    if (dev_balance_active) {
        cudaSetDevice(dev_balance_restore);
        // Rebuild thread_local GPUHandle for the restored device so STEOM
        // dispatch / downstream code sees handles on the correct device.
        gpu::GPUHandle::reset();
        std::cout << "  [EA-EOM device-balance] restored to GPU " << dev_balance_restore
                  << " (operator ran on GPU " << dev_balance_target
                  << "; GPUHandle rebuilt for restored device)" << std::endl;
    }
#endif
}

void ERI_Stored_RHF::compute_ea_eom_ccsd(int n_states) {
    compute_ea_eom_ccsd_impl(rhf_, eri_matrix_.device_ptr(), n_states);
}

// bt-PNO-STEOM P4 (RI path): build the full MO ERI from the RI B factors
// (matching RI-CCSD), then run the identical canonical EA-EOM-CCSD impl. EA
// roots match canonical EA-EOM to the RI fitting tolerance (~1e-4 Ha).
void ERI_RI_RHF::compute_ea_eom_ccsd(int n_states) {
    const real_t* d_C   = rhf_.get_coefficient_matrix().device_ptr();
    const int num_basis = rhf_.get_num_basis();
    // P4b: RI DLPNO builds the operator's MO-ERI blocks on the fly from B_mo
    // (no full nmo⁴). Single-GPU uses intermediate_matrix_B_; distributed-RI's
    // build_B_mo lazily replicates B to each GPU (~580 MB at anthracene scale).
    // build_B_mo returns nullptr when budget exceeded → fallback to full tensor.
    // Frozen core: the block path reads the active window from the full-C B_mo
    // via the operator's frozen_off shift, so it is NO LONGER gated on
    // num_frozen==0 — avoids the nao⁴ MO-ERI tensor (OOM for ~tetracene+).
    const bool want_block = rhf_.use_dlpno_amplitudes()
                            && gpu::gpu_available();
    const real_t* d_B_mo = want_block ? build_B_mo(d_C, num_basis) : nullptr;
    const bool block     = want_block && d_B_mo != nullptr;
    real_t* d_eri_mo     = block ? nullptr : build_mo_eri(d_C, num_basis);
    compute_ea_eom_ccsd_impl(rhf_, /*d_eri_ao=*/nullptr, n_states, d_eri_mo,
                             block ? this : nullptr, d_B_mo);
    if (d_eri_mo) tracked_cudaFree(d_eri_mo);
}

} // namespace gansu
