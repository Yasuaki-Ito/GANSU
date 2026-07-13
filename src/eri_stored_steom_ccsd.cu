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
#include <fstream>

#include "rhf.hpp"
#include "steom_ccsd_operator.hpp"
#include "eom_chain_context.hpp"   // DMET-STEOM: standalone cluster electronic state
#include "dmet.hpp"                // DMET-STEOM Phase 1: cluster embedding driver
#include "dlpno_localizer.hpp"     // DMET×DLPNO P0-3: rectangular-C localization probe
#include "dlpno_mp2.hpp"           // DMET×DLPNO Phase 1a: cluster-space DLPNO-LMP2
#include "dlpno_ccsd.hpp"          // DMET×DLPNO Phase 1b: cluster-space DLPNO-CCSD ground
#include "dlpno_params.hpp"        //   (resolve_dlpno_params for the cluster DLPNO hooks)
#include "steom_result.hpp"
#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "profiler.hpp"
#include "utils.hpp"
#ifdef GANSU_MPI
#include <mpi.h>
#endif
#if defined(GANSU_MPI) && defined(GANSU_MULTI_GPU)
#include "nccl_comm.hpp"          // also pulls in multi_gpu_manager.hpp
#include "steom_barh_cache.hpp"
#endif
#if defined(GANSU_MPI) && defined(_OPENMP)
#include <omp.h>
// OpenBLAS thread cap (weak: no-op if not linked against OpenBLAS).
extern "C" void openblas_set_num_threads(int) __attribute__((weak));
#endif

namespace gansu {

#ifdef GANSU_MPI
// ============================================================================
//  MPI Step 4 — STEOM stage-2 rank0=IP / rank1=EA parallel split.
//  rank 0 solves IP-EOM, rank 1 solves EA-EOM concurrently; rank 1 ships its
//  EAEOMResult (host amplitudes) to rank 0, which then runs the STEOM second
//  transform + diagonalization. See MPI_DESIGN.md Step 4.
// ============================================================================
namespace {

void steom_mpi_rank_size(int& rank, int& size) {
    rank = 0; size = 1;
    int inited = 0; MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
}

constexpr int STEOM_EA_TAG = 0x57A4;  // arbitrary unique tag for the EA hand-off

// Send one PerRoot's scalar metadata + R1 + R2 to `dest`.
void send_ea_perroot(const EAEOMResult::PerRoot& pr, int dest) {
    double meta[3] = { pr.omega, pr.percent_singles, pr.followcis_overlap };
    long sizes[3]  = { (long)pr.canonical_vir_label,
                       (long)pr.R1.size(), (long)pr.R2.size() };
    MPI_Send(meta,  3, MPI_DOUBLE, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
    MPI_Send(sizes, 3, MPI_LONG,   dest, STEOM_EA_TAG, MPI_COMM_WORLD);
    if (!pr.R1.empty()) MPI_Send(pr.R1.data(), (int)pr.R1.size(), MPI_DOUBLE, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
    if (!pr.R2.empty()) MPI_Send(pr.R2.data(), (int)pr.R2.size(), MPI_DOUBLE, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
}

void recv_ea_perroot(EAEOMResult::PerRoot& pr, int src) {
    double meta[3]; long sizes[3];
    MPI_Recv(meta,  3, MPI_DOUBLE, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(sizes, 3, MPI_LONG,   src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    pr.omega = meta[0]; pr.percent_singles = meta[1]; pr.followcis_overlap = meta[2];
    pr.canonical_vir_label = (int)sizes[0];
    pr.R1.resize((size_t)sizes[1]);
    pr.R2.resize((size_t)sizes[2]);
    if (!pr.R1.empty()) MPI_Recv(pr.R1.data(), (int)pr.R1.size(), MPI_DOUBLE, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!pr.R2.empty()) MPI_Recv(pr.R2.data(), (int)pr.R2.size(), MPI_DOUBLE, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// rank 1 → rank 0: ship the EAEOMResult consumed by the STEOM second transform.
// Only the fields STEOM reads (dims + per_active) are sent; `auxiliary` is
// diagnostic-only and skipped. `report` is sent so rank 0's final summary
// includes the EA table.
void send_ea_result(const EAEOMResult& ea, int dest) {
    int header[5] = { ea.nocc_active, ea.nvir, ea.num_frozen, ea.n_active,
                      (int)ea.per_active.size() };
    MPI_Send(header, 5, MPI_INT, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
    for (const auto& pr : ea.per_active) send_ea_perroot(pr, dest);
    int rlen = (int)ea.report.size();
    MPI_Send(&rlen, 1, MPI_INT, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
    if (rlen > 0) MPI_Send(ea.report.data(), rlen, MPI_CHAR, dest, STEOM_EA_TAG, MPI_COMM_WORLD);
}

void recv_ea_result(EAEOMResult& ea, int src) {
    int header[5];
    MPI_Recv(header, 5, MPI_INT, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ea.nocc_active = header[0]; ea.nvir = header[1];
    ea.num_frozen  = header[2]; ea.n_active = header[3];
    ea.per_active.resize((size_t)header[4]);
    for (auto& pr : ea.per_active) recv_ea_perroot(pr, src);
    int rlen = 0;
    MPI_Recv(&rlen, 1, MPI_INT, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    ea.report.resize((size_t)std::max(0, rlen));
    if (rlen > 0) MPI_Recv(&ea.report[0], rlen, MPI_CHAR, src, STEOM_EA_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// --- CIS-NTO active space hand-off (rank 0 → rank 1, point-to-point) ---------
// rank 0 computes the state-averaged CIS-NTO active space and SENDS it to
// rank 1 so the EA root routing (select_active_ea_roots, which reads
// cis_nto.U_vir) uses the SAME active space as rank 0's STEOM, keeping the EA
// canonical_vir_label indices consistent. Without this, each rank's independent
// eigendecomposition of (near-)degenerate NTO blocks could order the active
// virtuals differently and corrupt the second transform. Point-to-point (NOT
// MPI_Bcast) so idle ranks ≥2 can exit early without deadlocking a collective.
constexpr int STEOM_CIS_TAG = 0x57C1;

void send_vec_double(const std::vector<real_t>& v, int dest) {
    long n = (long)v.size();
    MPI_Send(&n, 1, MPI_LONG, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
    if (n > 0) MPI_Send(v.data(), (int)n, MPI_DOUBLE, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
}
void recv_vec_double(std::vector<real_t>& v, int src) {
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    v.resize((size_t)std::max(0L, n));
    if (n > 0) MPI_Recv(v.data(), (int)n, MPI_DOUBLE, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
void send_vec_int(const std::vector<int>& v, int dest) {
    long n = (long)v.size();
    MPI_Send(&n, 1, MPI_LONG, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
    if (n > 0) MPI_Send(v.data(), (int)n, MPI_INT, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
}
void recv_vec_int(std::vector<int>& v, int src) {
    long n = 0;
    MPI_Recv(&n, 1, MPI_LONG, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    v.resize((size_t)std::max(0L, n));
    if (n > 0) MPI_Recv(v.data(), (int)n, MPI_INT, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void send_cis_nto(const CISNTOResult& cis, int dest) {
    int hi[5] = { cis.nocc_active, cis.nvir, cis.num_frozen, cis.n_act_occ, cis.n_act_vir };
    MPI_Send(hi, 5, MPI_INT, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
    double hd[5] = { cis.trace_occ, cis.trace_vir, cis.weight_sum, cis.o_thresh, cis.v_thresh };
    MPI_Send(hd, 5, MPI_DOUBLE, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
    send_vec_double(cis.nto_occ_occupations, dest);
    send_vec_double(cis.nto_vir_occupations, dest);
    send_vec_double(cis.U_occ, dest);
    send_vec_double(cis.U_vir, dest);
    send_vec_int(cis.active_occ_indices, dest);
    send_vec_int(cis.active_vir_indices, dest);
    int rlen = (int)cis.report.size();
    MPI_Send(&rlen, 1, MPI_INT, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
    if (rlen > 0) MPI_Send(cis.report.data(), rlen, MPI_CHAR, dest, STEOM_CIS_TAG, MPI_COMM_WORLD);
}

void recv_cis_nto(CISNTOResult& cis, int src) {
    int hi[5];
    MPI_Recv(hi, 5, MPI_INT, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cis.nocc_active = hi[0]; cis.nvir = hi[1]; cis.num_frozen = hi[2];
    cis.n_act_occ = hi[3]; cis.n_act_vir = hi[4];
    double hd[5];
    MPI_Recv(hd, 5, MPI_DOUBLE, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cis.trace_occ = hd[0]; cis.trace_vir = hd[1]; cis.weight_sum = hd[2];
    cis.o_thresh = hd[3]; cis.v_thresh = hd[4];
    recv_vec_double(cis.nto_occ_occupations, src);
    recv_vec_double(cis.nto_vir_occupations, src);
    recv_vec_double(cis.U_occ, src);
    recv_vec_double(cis.U_vir, src);
    recv_vec_int(cis.active_occ_indices, src);
    recv_vec_int(cis.active_vir_indices, src);
    int rlen = 0;
    MPI_Recv(&rlen, 1, MPI_INT, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cis.report.resize((size_t)std::max(0, rlen));
    if (rlen > 0) MPI_Recv(&cis.report[0], rlen, MPI_CHAR, src, STEOM_CIS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

// --- Stage-1 ground-state hand-off (rank 0 → rank 1, host MPI) ----------------
// Ships the DLPNO-CCSD ground state so rank 1 can SKIP stage-1 (the expensive
// DLPNO ground) entirely: only rank 0 computes it (at full node threads while
// rank 1 waits), then sends the canonical coefficient matrix C, orbital energies
// eps, and the back-transformed T1/T2 (BTAmplitudes). rank 1 MUST adopt rank 0's
// C/eps because the canonical basis (degenerate-subspace rotations / signs) can
// differ between independent SCF runs, and the transferred T2 lives in rank 0's
// basis. Eliminates redundant stage-1 that otherwise runs at half threads on
// both ranks and eats the IP||EA win. All host-side MPI.
constexpr int STEOM_GROUND_TAG = 0x57C9;

void send_ground_state(RHF& rhf, double E_corr, int dest) {
    const int nb = rhf.get_num_basis();
    auto& C   = rhf.get_coefficient_matrix();  C.toHost();
    auto& eps = rhf.get_orbital_energies();    eps.toHost();
    const BTAmplitudes& bt = rhf.get_dlpno_bt_amplitudes();
    long hdr[5] = { (long)nb, (long)bt.nocc, (long)bt.nvir,
                    (long)bt.T1.size(), (long)bt.T2.size() };
    MPI_Send(hdr, 5, MPI_LONG, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
    MPI_Send(&E_corr, 1, MPI_DOUBLE, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
    MPI_Send(C.host_ptr(),   (int)((size_t)nb * nb), MPI_DOUBLE, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
    MPI_Send(eps.host_ptr(), nb,                     MPI_DOUBLE, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
    if (!bt.T1.empty()) MPI_Send(bt.T1.data(), (int)bt.T1.size(), MPI_DOUBLE, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
    if (!bt.T2.empty()) MPI_Send(bt.T2.data(), (int)bt.T2.size(), MPI_DOUBLE, dest, STEOM_GROUND_TAG, MPI_COMM_WORLD);
}

double recv_ground_state(RHF& rhf, int src) {
    long hdr[5];
    MPI_Recv(hdr, 5, MPI_LONG, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int nb = (int)hdr[0];
    double E_corr = 0.0;
    MPI_Recv(&E_corr, 1, MPI_DOUBLE, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    auto& C   = rhf.get_coefficient_matrix();  C.toHost();   // ensure host buffer exists
    auto& eps = rhf.get_orbital_energies();    eps.toHost();
    MPI_Recv(C.host_ptr(),   (int)((size_t)nb * nb), MPI_DOUBLE, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(eps.host_ptr(), nb,                     MPI_DOUBLE, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    C.toDevice();  eps.toDevice();   // adopt rank 0's canonical basis
    BTAmplitudes bt;
    bt.nocc = (int)hdr[1]; bt.nvir = (int)hdr[2];
    bt.T1.resize((size_t)hdr[3]);
    bt.T2.resize((size_t)hdr[4]);
    if (!bt.T1.empty()) MPI_Recv(bt.T1.data(), (int)bt.T1.size(), MPI_DOUBLE, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!bt.T2.empty()) MPI_Recv(bt.T2.data(), (int)bt.T2.size(), MPI_DOUBLE, src, STEOM_GROUND_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    rhf.set_dlpno_bt_amplitudes(std::move(bt));
    return E_corr;
}

#ifdef GANSU_MULTI_GPU
// --- Step 4b: EA bar-H device-tensor hand-off (rank 1 → rank 0, NCCL) ---------
// Ships the 2 EA-side dressed bar-H tensors (Wvovv, Wvvvo; each [nvir³·nocc])
// from rank 1's cache to rank 0 so rank 0's STEOM borrows all 11 and skips its
// bar-H rebuild — restoring the share-barH fast path under the IP||EA split.
// Wvvvv is nullptr under canonical-skip (the DLPNO-STEOM production path) and is
// not transferred. dims go over MPI (host), the tensors over the NCCL world comm.
constexpr int STEOM_BARH_TAG = 0x57B6;

// Handshake: dims are sent UNCONDITIONALLY (nvir==0 signals "no EA bar-H", e.g.
// build_dressed was skipped); the NCCL tensor transfer happens only when both
// ranks agree nvir>0. This keeps send/recv matched and deadlock-free.
// After dims, the receiver replies with a 1-int status (1=buffers allocated OK,
// 0=alloc failed / no transfer). The sender only enters the NCCL group when the
// receiver is ready, so an OOM at the receiver becomes a clean skip on BOTH
// ranks instead of a deadlock (NCCL send blocked forever on a recv that the
// OOM'd rank never posts).
void send_ea_barh(const SteomBarHCache& cache, int dest) {
    const bool ok = cache.has_ea && cache.d_Wvovv && cache.d_Wvvvo && cache.nvir > 0;
    int dims[3] = { ok ? cache.nocc : 0, ok ? cache.nvir : 0,
                    cache.canonical_skip_wvvvv ? 1 : 0 };
    MPI_Send(dims, 3, MPI_INT, dest, STEOM_BARH_TAG, MPI_COMM_WORLD);
    if (!ok) return;
    int rx_ready = 0;
    MPI_Recv(&rx_ready, 1, MPI_INT, dest, STEOM_BARH_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (!rx_ready) {
        std::cout << "  [STEOM MPI] rank " << dest << " could not allocate EA bar-H "
                     "(OOM) — transfer skipped (rank 0 will rebuild)." << std::endl;
        return;
    }
    const size_t n = (size_t)cache.nvir * cache.nvir * cache.nvir * cache.nocc;
    auto& mgr = MultiGpuManager::instance();
    nccl::group_start();
    nccl::send(cache.d_Wvovv, n, dest, 0, mgr.comm_stream(0));
    nccl::send(cache.d_Wvvvo, n, dest, 0, mgr.comm_stream(0));
    nccl::group_end();
    cudaStreamSynchronize(mgr.comm_stream(0));
}

void recv_ea_barh(SteomBarHCache& cache, int src) {
    int dims[3];
    MPI_Recv(dims, 3, MPI_INT, src, STEOM_BARH_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const int nocc = dims[0], nvir = dims[1];
    const bool skip = dims[2] != 0;
    if (nvir == 0) return;   // sender had no EA bar-H → cache stays incomplete (STEOM rebuilds)
    const size_t n = (size_t)nvir * nvir * nvir * nocc;
    // Try to allocate the receive buffers; on OOM, tell the sender to skip so it
    // does not block forever in its NCCL send.
    real_t* d_Wvovv = nullptr; real_t* d_Wvvvo = nullptr;
    int ready = 1;
    try {
        tracked_cudaMalloc(&d_Wvovv, n * sizeof(real_t));
        tracked_cudaMalloc(&d_Wvvvo, n * sizeof(real_t));
    } catch (...) {
        if (d_Wvovv) { tracked_cudaFree(d_Wvovv); d_Wvovv = nullptr; }
        if (d_Wvvvo) { tracked_cudaFree(d_Wvvvo); d_Wvvvo = nullptr; }
        ready = 0;
    }
    MPI_Send(&ready, 1, MPI_INT, src, STEOM_BARH_TAG, MPI_COMM_WORLD);
    if (!ready) return;   // cache stays incomplete → STEOM rebuilds (or errors cleanly)
    auto& mgr = MultiGpuManager::instance();
    nccl::group_start();
    nccl::recv(d_Wvovv, n, src, 0, mgr.comm_stream(0));
    nccl::recv(d_Wvvvo, n, src, 0, mgr.comm_stream(0));
    nccl::group_end();
    cudaStreamSynchronize(mgr.comm_stream(0));
    cache.d_Wvovv = d_Wvovv;
    cache.d_Wvvvo = d_Wvvvo;
    cache.d_Wvvvv = nullptr;   // canonical-skip
    cache.canonical_skip_wvvvv = skip;
    if (cache.nocc == 0) { cache.nocc = nocc; cache.nvir = nvir; }
    cache.has_ea = true;
}
#endif // GANSU_MULTI_GPU

}  // anonymous namespace
#endif // GANSU_MPI

// Forward declarations from eri_stored.cu (shared with all EOM modules).
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);

class ERI_RI;  // RI-CCSD B-native block source
real_t ccsd_spatial_orbital(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed = nullptr,
                            int num_frozen = 0,
                            const real_t* h_fov_active = nullptr,
                            const ERI_RI* eri_ri = nullptr,
                            const real_t level_shift = 0.0);

// (bt-polish) stage warm-start amplitudes for the NEXT ccsd_spatial_orbital
// call (defined in eri_stored.cu; consumed-and-cleared there). conv_override > 0
// sets that call's |dE| threshold (GANSU_CCSD_CONV still wins if set).
void ccsd_set_initial_guess(const real_t* h_t1, const real_t* h_t2,
                            int max_iter_override, real_t conv_override);

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

// ============================================================================
//  (debug accelerator) DMET-STEOM checkpoint — cache the CIS-NTO/IP/EA results +
//  cluster CCSD T1/T2 + published bar-H so a re-run skips the ~24 min CCSD+IP+EA
//  prefix and iterates on the STEOM stage in ~5 min. Gated by GANSU_DMET_STEOM_CKPT
//  =<file>. The loaded state is bit-identical to the recomputed one (deterministic
//  inputs), so STEOM roots are unchanged; this is purely to shorten the debug loop.
// ============================================================================
namespace {
constexpr int kSteomCkptMagic = 0x53544D32;  // "STM2"

template<class T> void ckpt_wr(std::ofstream& f, const T& v){ f.write((const char*)&v, sizeof(T)); }
template<class T> void ckpt_rd(std::ifstream& f, T& v){ f.read((char*)&v, sizeof(T)); }
template<class T> void ckpt_wr_vec(std::ofstream& f, const std::vector<T>& v){
    size_t n = v.size(); f.write((const char*)&n, sizeof(n));
    if (n) f.write((const char*)v.data(), n * sizeof(T)); }
template<class T> void ckpt_rd_vec(std::ifstream& f, std::vector<T>& v){
    size_t n = 0; f.read((char*)&n, sizeof(n)); v.resize(n);
    if (n) f.read((char*)v.data(), n * sizeof(T)); }
// Device buffer: D2H then write (with element count). Null ⇒ count 0.
inline void ckpt_wr_dev(std::ofstream& f, const real_t* d, size_t n){
    size_t nn = (d ? n : 0); f.write((const char*)&nn, sizeof(nn));
    if (nn) { std::vector<real_t> h(nn);
        cudaMemcpy(h.data(), d, nn * sizeof(real_t), cudaMemcpyDeviceToHost);
        f.write((const char*)h.data(), nn * sizeof(real_t)); } }
// Read into a fresh tracked device buffer (nullptr if count 0).
inline real_t* ckpt_rd_dev(std::ifstream& f){
    size_t n = 0; f.read((char*)&n, sizeof(n)); if (!n) return nullptr;
    std::vector<real_t> h(n); f.read((char*)h.data(), n * sizeof(real_t));
    real_t* d = nullptr; tracked_cudaMalloc(&d, n * sizeof(real_t));
    cudaMemcpy(d, h.data(), n * sizeof(real_t), cudaMemcpyHostToDevice); return d; }

// per-root label accessor overloads (IP uses canonical_occ_label, EA canonical_vir_label)
inline int ckpt_eom_label(const IPEOMResult::PerRoot& pr){ return pr.canonical_occ_label; }
inline int ckpt_eom_label(const EAEOMResult::PerRoot& pr){ return pr.canonical_vir_label; }
inline void ckpt_set_label(IPEOMResult::PerRoot& pr, int v){ pr.canonical_occ_label = v; }
inline void ckpt_set_label(EAEOMResult::PerRoot& pr, int v){ pr.canonical_vir_label = v; }
template<class Res> void ckpt_wr_eom(std::ofstream& f, const Res& r){
    ckpt_wr(f, r.nocc_active); ckpt_wr(f, r.nvir); ckpt_wr(f, r.num_frozen); ckpt_wr(f, r.n_active);
    size_t n = r.per_active.size(); ckpt_wr(f, n);
    for (const auto& pr : r.per_active) {
        ckpt_wr(f, pr.omega); ckpt_wr(f, pr.percent_singles); ckpt_wr(f, pr.followcis_overlap);
        int label = ckpt_eom_label(pr);
        ckpt_wr(f, label);
        ckpt_wr_vec(f, pr.R1); ckpt_wr_vec(f, pr.R2);
    } }
template<class Res> void ckpt_rd_eom(std::ifstream& f, Res& r){
    ckpt_rd(f, r.nocc_active); ckpt_rd(f, r.nvir); ckpt_rd(f, r.num_frozen); ckpt_rd(f, r.n_active);
    size_t n = 0; ckpt_rd(f, n); r.per_active.resize(n);
    for (auto& pr : r.per_active) {
        ckpt_rd(f, pr.omega); ckpt_rd(f, pr.percent_singles); ckpt_rd(f, pr.followcis_overlap);
        int label = -1; ckpt_rd(f, label); ckpt_set_label(pr, label);
        ckpt_rd_vec(f, pr.R1); ckpt_rd_vec(f, pr.R2);
    } }

void steom_ckpt_save(const EOMChainContext& ctx, const char* path){
    std::ofstream f(path, std::ios::binary);
    if (!f) { std::cout << "  [STEOM ckpt] WARN cannot open " << path << " for write." << std::endl; return; }
    int magic = kSteomCkptMagic; ckpt_wr(f, magic);
    // cluster CCSD ground
    int cached = ctx.cc_ground_cached ? 1 : 0; ckpt_wr(f, cached);
    ckpt_wr(f, ctx.cc_t1n); ckpt_wr(f, ctx.cc_t2n); ckpt_wr(f, ctx.cc_E);
    ckpt_wr_dev(f, ctx.cc_t1, ctx.cc_t1n); ckpt_wr_dev(f, ctx.cc_t2, ctx.cc_t2n);
    // CIS-NTO result
    { const auto& c = ctx.cis_nto_result;
      ckpt_wr(f, c.nocc_active); ckpt_wr(f, c.nvir); ckpt_wr(f, c.num_frozen);
      ckpt_wr(f, c.n_act_occ); ckpt_wr(f, c.n_act_vir);
      ckpt_wr_vec(f, c.nto_occ_occupations); ckpt_wr_vec(f, c.nto_vir_occupations);
      ckpt_wr_vec(f, c.U_occ); ckpt_wr_vec(f, c.U_vir);
      ckpt_wr_vec(f, c.active_occ_indices); ckpt_wr_vec(f, c.active_vir_indices);
      ckpt_wr(f, c.trace_occ); ckpt_wr(f, c.trace_vir); ckpt_wr(f, c.weight_sum);
      ckpt_wr(f, c.o_thresh); ckpt_wr(f, c.v_thresh); }
    ckpt_wr_eom(f, ctx.ip_eom_result);
    ckpt_wr_eom(f, ctx.ea_eom_result);
    // bar-H (partial-capable: after IP only the 8 IP buffers are present; each
    // ckpt_wr_dev writes count 0 for a null slot, so partial state round-trips).
    const auto& b = ctx.barh;
    int hip = b.has_ip ? 1 : 0, hea = b.has_ea ? 1 : 0; ckpt_wr(f, hip); ckpt_wr(f, hea);
    ckpt_wr(f, b.nocc); ckpt_wr(f, b.nvir); int cs = b.canonical_skip_wvvvv ? 1 : 0; ckpt_wr(f, cs);
    const size_t NO = (size_t)b.nocc, NV = (size_t)b.nvir;
    ckpt_wr_dev(f, b.d_Loo, NO*NO);   ckpt_wr_dev(f, b.d_Lvv, NV*NV);   ckpt_wr_dev(f, b.d_Fov, NO*NV);
    ckpt_wr_dev(f, b.d_Woooo, NO*NO*NO*NO); ckpt_wr_dev(f, b.d_Wooov, NO*NO*NO*NV);
    ckpt_wr_dev(f, b.d_Wovov, NO*NV*NO*NV); ckpt_wr_dev(f, b.d_Wovvo, NO*NV*NV*NO);
    ckpt_wr_dev(f, b.d_Wovoo, NO*NV*NO*NO); ckpt_wr_dev(f, b.d_Wvovv, NV*NO*NV*NV);
    ckpt_wr_dev(f, b.d_Wvvvv, NV*NV*NV*NV); ckpt_wr_dev(f, b.d_Wvvvo, NV*NV*NV*NO);
    f.flush();
    std::cout << "  [STEOM ckpt] saved to " << path << " (cc=" << (ctx.cc_ground_cached?"y":"n")
              << " ip=" << ctx.ip_eom_result.per_active.size()
              << " ea=" << ctx.ea_eom_result.per_active.size()
              << " barH ip/ea=" << hip << "/" << hea << ")." << std::endl;
}

bool steom_ckpt_load(EOMChainContext& ctx, const char* path){
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int magic = 0; ckpt_rd(f, magic);
    if (magic != kSteomCkptMagic) {
        std::cout << "  [STEOM ckpt] " << path << " bad magic — ignoring." << std::endl; return false; }
    int cached = 0; ckpt_rd(f, cached);
    ckpt_rd(f, ctx.cc_t1n); ckpt_rd(f, ctx.cc_t2n); ckpt_rd(f, ctx.cc_E);
    ctx.cc_t1 = ckpt_rd_dev(f); ctx.cc_t2 = ckpt_rd_dev(f);
    ctx.cc_ground_cached = (cached != 0);
    { auto& c = ctx.cis_nto_result;
      ckpt_rd(f, c.nocc_active); ckpt_rd(f, c.nvir); ckpt_rd(f, c.num_frozen);
      ckpt_rd(f, c.n_act_occ); ckpt_rd(f, c.n_act_vir);
      ckpt_rd_vec(f, c.nto_occ_occupations); ckpt_rd_vec(f, c.nto_vir_occupations);
      ckpt_rd_vec(f, c.U_occ); ckpt_rd_vec(f, c.U_vir);
      ckpt_rd_vec(f, c.active_occ_indices); ckpt_rd_vec(f, c.active_vir_indices);
      ckpt_rd(f, c.trace_occ); ckpt_rd(f, c.trace_vir); ckpt_rd(f, c.weight_sum);
      ckpt_rd(f, c.o_thresh); ckpt_rd(f, c.v_thresh); }
    ckpt_rd_eom(f, ctx.ip_eom_result);
    ckpt_rd_eom(f, ctx.ea_eom_result);
    { auto& b = ctx.barh;
      int hip = 0, hea = 0; ckpt_rd(f, hip); ckpt_rd(f, hea);
      ckpt_rd(f, b.nocc); ckpt_rd(f, b.nvir);
      int cs = 0; ckpt_rd(f, cs); b.canonical_skip_wvvvv = (cs != 0);
      b.d_Loo = ckpt_rd_dev(f); b.d_Lvv = ckpt_rd_dev(f); b.d_Fov = ckpt_rd_dev(f);
      b.d_Woooo = ckpt_rd_dev(f); b.d_Wooov = ckpt_rd_dev(f); b.d_Wovov = ckpt_rd_dev(f);
      b.d_Wovvo = ckpt_rd_dev(f); b.d_Wovoo = ckpt_rd_dev(f); b.d_Wvovv = ckpt_rd_dev(f);
      b.d_Wvvvv = ckpt_rd_dev(f); b.d_Wvvvo = ckpt_rd_dev(f);
      b.has_ip = (hip != 0); b.has_ea = (hea != 0); }
    const bool ip_done = !ctx.ip_eom_result.per_active.empty();
    const bool ea_done = !ctx.ea_eom_result.per_active.empty();
    std::cout << "  [STEOM ckpt] loaded from " << path << " — skip CCSD"
              << (ip_done ? "+IP" : "") << (ea_done ? "+EA" : "")
              << (ea_done ? "" : " (IP-only checkpoint: EA re-runs)")
              << (ctx.barh.has_ip ? "; bar-H restored" : "") << "." << std::endl;
    return true;
}
}  // anonymous namespace

// `eri_method` is taken by ERI base reference so the composite auto-dispatch
// (compute_cis_nto / compute_ip_eom_ccsd / compute_ea_eom_ccsd) resolves
// polymorphically — stored uses AO→MO transform, RI builds the MO ERI from B.
// When `d_eri_mo_precomputed` is non-null (RI path) the AO→MO transform is
// skipped and the caller-owned MO ERI tensor is used (and not freed here).
static void compute_steom_ccsd_impl(RHF& rhf,
                                    ERI& eri_method,
                                    const real_t* d_eri_ao,
                                    int n_states_requested,
                                    real_t* d_eri_mo_precomputed = nullptr,
                                    EOMChainContext* ctx = nullptr)
{
    PROFILE_FUNCTION();

    // DMET-STEOM: when ctx is non-null this is a standalone CLUSTER STEOM run.
    // The composite chain (CIS-NTO → IP-EOM → EA-EOM) is driven via the
    // ctx-aware cluster-stage free functions over the precomputed cluster MO-ERI
    // (`d_eri_mo_precomputed`) instead of the polymorphic eri_method dispatch;
    // the cluster electronic state + inter-stage results live in `ctx`, while
    // `rhf` supplies read-only config. The multi-GPU auto-scale / device-balance
    // / MPI-split / share-barH machinery is bypassed (a cluster is small and
    // single-GPU). `ctx == nullptr` ⇒ byte-identical legacy plain-STEOM path.
    const int verbose = rhf.get_steom_verbose();

    // Spin block (--spin_type): singlet (default) or triplet. Both share the
    // whole chain (CIS-NTO active-space selection — itself spin-adapted via
    // rhf.is_triplet() — CCSD, IP/EA-EOM, bar-H, W^eff routes); only the final
    // G^{1h1p} assembly differs (triplet drops the 2·g_phhp Coulomb channel).
    const bool steom_triplet = rhf.is_triplet();
    {
        const std::string& st = rhf.get_spin_type();
        if (st != "singlet" && st != "triplet") {
            std::cout << "Warning: STEOM-CCSD supports --spin_type singlet|triplet; \""
                      << st << "\" falls back to singlet." << std::endl;
        }
        if (steom_triplet) {
            std::cout << "STEOM-CCSD: TRIPLET block requested (G = F_eff − g_phph; "
                         "CIS-NTO active-space selection uses triplet CIS)." << std::endl;
        }
    }

    // Auto-scale (env-free): the EA/STEOM d_eri_vvvv (nvir⁴·8B) is the memory wall.
    // When it is a large fraction of the tightest GPU's TOTAL memory and >1 GPU is
    // available, auto-enable the multi-GPU BUILD distribution that otherwise needs
    // GANSU_EA_VVVV_NSLAB / _OPERATOR_DEVICE_BALANCING / _STEOM_BUILD_GPUS.
    // CRITICAL: this runs BEFORE the share-barH decision + the IP/EA dispatch, so
    // device-balancing is decided consistently for the WHOLE chain — setting it
    // mid-chain (e.g. at EA start) makes IP publish bar-H balancing-OFF then EA/STEOM
    // borrow cross-device → garbage G (all-zero STEOM roots). setenv(overwrite=0):
    // explicit env always wins; small systems untouched → single-GPU build, byte-
    // identical. Threshold 0.40×total (not free) is GPU-size-scaled / resident-
    // independent: H200 140GB ⇒ 56 GB (pentacene nvir=305⇒64 GB distributes,
    // tetracene 30 GB single-GPU); A100 80GB ⇒ 32 GB (anthracene 28 GB single-GPU,
    // keeping the share-barH fast path).
#ifndef GANSU_CPU_ONLY
    if (!ctx && gpu::gpu_available() && rhf.get_num_gpus() > 1) {   // cluster: small, single-GPU → no auto-scale
        const int full_occ = rhf.get_num_electrons() / 2;
        const size_t nv = static_cast<size_t>(rhf.get_num_basis() - full_occ);
        const size_t vvvv_bytes = nv * nv * nv * nv * sizeof(real_t);
        int n_dev = 0; cudaGetDeviceCount(&n_dev);
        const int ng = std::min(rhf.get_num_gpus(), n_dev);
        size_t min_total = 0; bool have = false;
        int saved = 0; cudaGetDevice(&saved);
        for (int d = 0; d < ng; ++d) {
            cudaSetDevice(d);
            size_t fb = 0, tb = 0;
            if (cudaMemGetInfo(&fb, &tb) == cudaSuccess) {
                min_total = have ? std::min(min_total, tb) : tb; have = true;
            }
        }
        cudaSetDevice(saved);
        // (RI Term A) When the EA/STEOM operators evaluate Wvvvo·t1 from the RI
        // B-factors, d_eri_vvvv (nvir⁴) is never materialised — there is no memory
        // wall to distribute, so keep single-GPU + share-barH ON (STEOM borrows,
        // no ~nvir⁴ rebuild) and avoid the device-balancing chain entirely. This
        // driver IS the RI path, so the operators' eri_block_src_ is non-null ⇒
        // ri_vvvv_term_a_ active under the same env flags checked here.
        auto envon = [](const char* n, bool d){ const char* e = std::getenv(n);
            return (!e || !e[0]) ? d : (e[0] != '0'); };
        const bool ri_handles_vvvv = envon("GANSU_DLPNO_NATIVE_EOM", false)
            && envon("GANSU_DLPNO_NATIVE_BARE", true)
            && envon("GANSU_DLPNO_CANONICAL_SKIP", true)
            && envon("GANSU_DLPNO_EA_VVVV_RI", true);
        if (have && ng >= 2 && !ri_handles_vvvv && vvvv_bytes > static_cast<size_t>(0.40 * min_total)) {
            const std::string ns = std::to_string(ng);
            setenv("GANSU_EA_VVVV_NSLAB", ns.c_str(), 0);
            setenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING", "1", 0);
            setenv("GANSU_STEOM_BUILD_GPUS", ns.c_str(), 0);
            std::cout << "  [STEOM auto-scale] d_eri_vvvv = " << std::fixed
                      << std::setprecision(1) << vvvv_bytes / 1e9
                      << " GB > 0.40×total " << (0.40 * min_total) / 1e9
                      << " GB → auto-enabling NSLAB=" << ng
                      << " + device-balancing + STEOM_BUILD_GPUS=" << ng
                      << " (explicit env overrides)." << std::defaultfloat << std::endl;
        } else if (have && ng >= 2 && ri_handles_vvvv
                   && vvvv_bytes > static_cast<size_t>(0.40 * min_total)) {
            std::cout << "  [STEOM auto-scale] d_eri_vvvv = " << std::fixed
                      << std::setprecision(1) << vvvv_bytes / 1e9
                      << " GB handled by RI B-factors (Wvvvo·t1) → single-GPU build, "
                      << "no slab / device-balancing; share-barH stays ON."
                      << std::defaultfloat << std::endl;
        }
    }
#endif

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
    // MPI Step 4: rank topology for the IP||EA stage-2 split (see MPI_DESIGN.md).
    // rank 0 solves IP, rank 1 solves EA; rank 0 runs the STEOM transform.
    // OPT-IN (GANSU_STEOM_MPI_SPLIT=1) so we can validate in two stages: with the
    // split OFF every rank runs the full STEOM redundantly (validates the X-model
    // full-local B under MPI), then ON validates the IP/EA parallel + transfers.
    int mpi_rank = 0, mpi_size = 1;
#ifdef GANSU_MPI
    steom_mpi_rank_size(mpi_rank, mpi_size);
#endif
    bool mpi_steom = false;
#ifdef GANSU_MPI
    if (mpi_size > 1) {
        const char* e = std::getenv("GANSU_STEOM_MPI_SPLIT");
        mpi_steom = (e && e[0] == '1');
    }
#endif
    // Step 4b: with NCCL available, rank 1 ships its EA bar-H to rank 0 so
    // share-barH stays ON (rank 0 borrows, no ~30s rebuild). Without NCCL we
    // fall back to Step 4a Option A (share OFF, rank 0 rebuilds).
    bool mpi_barh_xfer = false;
#if defined(GANSU_MPI) && defined(GANSU_MULTI_GPU)
    mpi_barh_xfer = mpi_steom;
#endif

    // DMET-STEOM cluster: run a single self-contained STEOM (no MPI IP||EA split).
    // Fragment-parallelism is handled one level up (DMET fragment loop / MPI rank).
    if (ctx) {
        mpi_steom = false; mpi_barh_xfer = false;
        // (perf) Cluster share-barH: the cluster IP publishes its 8 dressed bar-H and
        // EA its 3 unique ones into ctx->barh (same validated mechanism as the non-
        // cluster path — see eri_stored_ip/ea_eom_ccsd.cu), so the STEOM operator
        // borrows all 11 and SKIPS its ~320 s build_dressed_intermediates.
        // ⚠ OPT-IN for the cluster (GANSU_STEOM_SHARE_BARH=1): keeping IP's 8 + EA's 3
        // bar-H resident on device 0 through the STEOM build adds ~28 GB that, together
        // with the RI B replica (~17 GB/GPU), OOMs the STEOM G build on a large cluster
        // (log174: needed 13.55 GB, 2.42 free). Realising the win needs device-balancing
        // the STEOM operator onto a freer GPU (cross-device bar-H) or freeing B first —
        // future work. Default OFF ⇒ STEOM rebuilds its bar-H (safe, no OOM).
        const char* e = std::getenv("GANSU_STEOM_SHARE_BARH");
        ctx->share_barh = (e && e[0] == '1');
        if (ctx->share_barh)
            std::cout << "  [DMET-STEOM share-barH] ON (opt-in) — IP/EA publish dressed bar-H; "
                         "STEOM borrows all 11 (skips ~320 s build_dressed). Multi-GPU: the "
                         "STEOM operator is device-balanced onto the freest GPU (bar-H/T1/T2 "
                         "migrated) to avoid device-0 OOM." << std::endl;
    }

    if (!ctx) {
        const char* env     = std::getenv("GANSU_STEOM_SHARE_BARH");
        const char* env_bal = std::getenv("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
        const bool balancing = (env_bal && env_bal[0] == '1');
        // Under MPI the IP and EA operators run on DIFFERENT ranks. Step 4b: when
        // NCCL is available, rank 1 ships its EA bar-H to rank 0 (recv_ea_barh
        // below) so share-barH stays ON and rank 0 borrows all 11 — no rebuild.
        // Without NCCL (mpi_barh_xfer false) we fall back to Option A: share OFF,
        // rank 0 rebuilds its bar-H from the CCSD ground + active space.
        const bool share = (!env || env[0] != '0') && !balancing
                           && (!mpi_steom || mpi_barh_xfer);
        rhf.set_steom_share_barh(share);
        if (share && mpi_steom)
            std::cout << "  [STEOM share-barH] ON (MPI IP||EA split + Step 4b: rank 1 "
                         "NCCL-transfers EA bar-H to rank 0, which borrows all 11)." << std::endl;
        else if (share)
            std::cout << "  [STEOM share-barH] ON (default; GANSU_STEOM_SHARE_BARH=0 to "
                         "disable) — IP/EA publish dressed bar-H, STEOM borrows "
                         "(skips its build_dressed)." << std::endl;
        else if (mpi_steom)
            std::cout << "  [STEOM share-barH] OFF (MPI IP||EA split, no NCCL → rank 0 "
                         "rebuilds bar-H locally; Option A)." << std::endl;
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
#ifdef GANSU_MPI
    // MPI Step 4a: only ranks 0 (IP→STEOM) and 1 (EA) work in stage-2; higher
    // ranks ran stage-1 redundantly and have nothing to do. They exit BEFORE any
    // point-to-point hand-off so no message is left dangling. NOTE: this assumes
    // the IP/EA solves are self-contained per rank (no world collective expecting
    // all ranks) — see MPI_DESIGN.md Step 4 open question.
    if (mpi_steom && mpi_rank >= 2) {
        std::cout << "  [STEOM MPI] rank " << mpi_rank
                  << " idle in stage-2 (IP||EA uses ranks 0,1 only)." << std::endl;
        return;
    }
#endif

    // Inc3: resolve the cluster B source up front so every stage (CIS, IP, EA,
    // STEOM build) can pull MO-ERI blocks from the cluster B instead of a dense
    // n_emb⁴ tensor. Only the stable engine pointer is cached here — each stage
    // rebuilds the transient B_mo workspace right before it consumes it (build_B_mo
    // shares one thread-local buffer that a stage's internal CCSD re-solve would
    // otherwise clobber; the rebuilds live in the CIS helper + IP/EA operators).
    if (ctx && ctx->prefer_ri_block && gpu::gpu_available() && !ctx->eri_block_src) {
        ctx->eri_block_src = dynamic_cast<const ERI_RI*>(&eri_method);
        if (!ctx->eri_block_src)
            std::cout << "  [DMET-STEOM] GANSU_DMET_STEOM_RI_BLOCK set but eri_method "
                         "is not RI — falling back to the dense cluster MO-ERI." << std::endl;
    }

    // (debug accelerator) Resume from a STEOM checkpoint: load CIS-NTO / IP / EA
    // results + cluster CCSD T1/T2 + published bar-H, so the CIS/IP/EA dispatch
    // below no-ops (populated results) and we jump straight to the STEOM stage.
    bool ckpt_loaded = false;
    bool ckpt_had_ea = false;   // loaded ckpt already contained the EA stage
    if (ctx) {
        if (const char* ck = std::getenv("GANSU_DMET_STEOM_CKPT")) {
            ckpt_loaded = steom_ckpt_load(*ctx, ck);
            ckpt_had_ea = ckpt_loaded && !ctx->ea_eom_result.per_active.empty();
            if (ckpt_loaded && ctx->barh.complete()) ctx->share_barh = true;  // borrow the restored bar-H
        }
    }

    // Stage 1/3 — CIS-NTO active space.
    if (!mpi_steom) {
        const bool need_cis = ctx ? (ctx->cis_nto_result.n_act_occ == 0)
                                   : (rhf.get_cis_nto_result().n_act_occ == 0);
        if (need_cis) {
            int n_cis = rhf.get_steom_n_root_cis();
            if (n_cis <= 0) n_cis = n_states_requested + 4;  // STEOM.md §7.3 default
            std::cout << "\n---- STEOM-CCSD composite dispatch: stage 1/3 = CIS-NTO active space ----" << std::endl;
            if (ctx) compute_cluster_cis_nto(rhf, *ctx, d_eri_mo_precomputed, n_cis);
            else     eri_method.compute_cis_nto(n_cis);
        }
    }
#ifdef GANSU_MPI
    else if (mpi_rank == 0) {
        // rank 0 computes CIS-NTO and ships it to rank 1 (consistent active space).
        if (rhf.get_cis_nto_result().n_act_occ == 0) {
            int n_cis = rhf.get_steom_n_root_cis();
            if (n_cis <= 0) n_cis = n_states_requested + 4;
            std::cout << "\n---- STEOM-CCSD composite dispatch: stage 1/3 = CIS-NTO active space [rank 0 → rank 1] ----" << std::endl;
            eri_method.compute_cis_nto(n_cis);
        }
        send_cis_nto(rhf.get_cis_nto_result(), 1);
    } else { // rank 1: use rank 0's CIS-NTO for EA routing
        CISNTOResult cis;
        recv_cis_nto(cis, 0);
        rhf.set_cis_nto_result(std::move(cis));
        std::cout << "  [STEOM MPI] rank 1 received CIS-NTO active space (n_act_vir="
                  << rhf.get_cis_nto_result().n_act_vir << ") from rank 0." << std::endl;
    }
#endif
    // (measure-first) Device-memory checkpoint after CIS-NTO. Placed AFTER the full
    // stage-1 if/else-if chain (must not sit between `if(!mpi_steom){}` and the
    // `#ifdef GANSU_MPI else if`, or the else binds to this dump's if — that
    // mis-fires the MPI stage-1 in a non-MPI run; cf. log169 MPI_Send crash).
    if (std::getenv("GANSU_STEOM_MEM_DUMP"))
        dump_tracked_allocations("after stage 1 (CIS-NTO)");

    // Stage 2/3 (IP, rank 0) and 3/3 (EA, rank 1) run concurrently under MPI.
    const bool do_ip = (!mpi_steom) || (mpi_rank == 0);
    const bool do_ea = (!mpi_steom) || (mpi_rank == 1);

    const bool need_ip = ctx ? ctx->ip_eom_result.per_active.empty()
                              : rhf.get_ip_eom_result().per_active.empty();
    if (do_ip && need_ip) {
        std::cout << "\n---- STEOM-CCSD composite dispatch: stage 2/3 = IP-EOM-CCSD (per active occ NTO)"
                  << (mpi_steom ? " [rank 0]" : "") << " ----" << std::endl;
        if (ctx) compute_cluster_ip_eom_ccsd(rhf, *ctx, d_eri_mo_precomputed, n_states_requested);
        else     eri_method.compute_ip_eom_ccsd(n_states_requested);
    }
    if (std::getenv("GANSU_STEOM_MEM_DUMP"))
        dump_tracked_allocations("after stage 2 (IP-EOM)");
    // (debug accelerator) Save an IP-stage checkpoint now (before EA) so a re-run can
    // resume past CCSD+IP and iterate on EA (+STEOM) — reached even if EA later OOMs.
    // The after-EA save below overwrites this with the full state once EA completes.
    if (ctx && !ckpt_loaded) {
        if (const char* ck = std::getenv("GANSU_DMET_STEOM_CKPT"))
            steom_ckpt_save(*ctx, ck);
    }
#ifndef GANSU_CPU_ONLY
    // (perf) EA-solve device relief (user request): with share-barH, IP's 8 published
    // bar-H (~27 GB at n_emb=490) pin the EA-solve device and OOM the EA Davidson
    // subspace there (log174/175; p-DDPA {0-29}: needed 17.24 GB, 1.66 free). IP's
    // operator is already destroyed (the cache owns the buffers — no alias), so move
    // the 8 to the freest peer GPU now; EA then builds+solves on its device with room.
    // STEOM later pulls EA's 3 onto the same GPU (barh.ip_dev) and builds there.
    // Only under sharing + a complete IP set + >1 GPU.
    //
    // (2026-07-13) Generalized from the hardcoded `ip_dev == 0`: in mode-2 the EA
    // operator builds+solves on the CLUSTER device (GANSU_DMET_STEOM_CLUSTER_GPU,
    // e.g. GPU 1 — device 0 keeps the RI 3c), and IP publishes its bar-H there too,
    // so IP+EA co-reside and OOM the Davidson. Target = the current device (the EA
    // solve device); migrate IP's bar-H to the freest OTHER GPU. Plain path
    // (cluster device == 0) is byte-unchanged.
    if (ctx && ctx->share_barh && gpu::gpu_available()
        && ctx->barh.has_ip && !ctx->barh.has_ea) {
        int ea_dev = 0; cudaGetDevice(&ea_dev);   // EA builds+solves here
        int n_dev = 0; cudaGetDeviceCount(&n_dev);
        if (n_dev > 1 && ctx->barh.ip_dev == ea_dev) {
            std::vector<size_t> freeb(n_dev, 0);
            for (int d = 0; d < n_dev; ++d) {
                if (d == ea_dev) continue;
                cudaSetDevice(d); size_t fb=0,tb=0;
                if (cudaMemGetInfo(&fb,&tb) == cudaSuccess) freeb[d] = fb;
            }
            cudaSetDevice(ea_dev);
            int D = -1; size_t bf = 0;
            for (int d = 0; d < n_dev; ++d) {
                if (d == ea_dev) continue;
                if (D < 0 || freeb[d] > bf) { bf = freeb[d]; D = d; }
            }
            if (D >= 0) {
                ctx->barh.migrate_ip_to(D);
                std::cout << "  [DMET-STEOM share-barH] moved IP bar-H (8) off the "
                             "EA-solve device " << ea_dev << " → GPU " << D
                          << " (free=" << std::fixed << std::setprecision(1)
                          << (bf/(1024.0*1024.0*1024.0))
                          << " GB) so the EA Davidson has room." << std::defaultfloat
                          << std::endl;
            }
        }
    }
    // (mode-2 / fresh-run relief) The DLPNO per-pair fan-out (or any earlier
    // build_mo_eri) may have replicated the FULL AO-basis B to every GPU
    // (naux·nao² ≈ 12.8 GB at Doxorubicin). From here on the cluster chain pulls
    // MO-ERI exclusively from the cluster B_mo (ctx->d_B_mo_blocks, which
    // survives — release_bmo_ao_replica frees only the lazy AO replicas), so
    // the AO copies are dead weight on the EA device: run8 measured projected
    // free −1.2 GiB vs +11.5 canonical — exactly the EA-Davidson OOM margin.
    // A later build_B_mo (e.g. the STEOM choreography's rebuild on the bar-H
    // GPU) lazily re-replicates from d_B_local_. Opt-out with
    // GANSU_STEOM_KEEP_B_REPLICA (same knob as the dense-diag reclaim).
    if (ctx && ctx->eri_block_src && gpu::gpu_available()
        && !std::getenv("GANSU_STEOM_KEEP_B_REPLICA")) {
        if (ERI_RI* rel_ri = dynamic_cast<ERI_RI*>(&eri_method)) {
            int dev_saved = 0; cudaGetDevice(&dev_saved);
            const size_t before = GlobalGpuMemoryTracker::get_current();
            rel_ri->release_bmo_ao_replica();
            const size_t after = GlobalGpuMemoryTracker::get_current();
            // free_replicated_B may reset the current device — restore + rebind.
            cudaSetDevice(dev_saved);
            gpu::GPUHandle::reset();
            if (before > after)
                std::cout << "  [DMET-STEOM] released RI AO-B replicas before the EA stage: "
                          << CudaMemoryManager<real_t>::format_bytes(before - after)
                          << " reclaimed (global)." << std::endl;
        }
    }
#endif
    const bool need_ea = ctx ? ctx->ea_eom_result.per_active.empty()
                              : rhf.get_ea_eom_result().per_active.empty();
    if (do_ea && need_ea) {
        std::cout << "\n---- STEOM-CCSD composite dispatch: stage 3/3 = EA-EOM-CCSD (per active vir NTO)"
                  << (mpi_steom ? " [rank 1]" : "") << " ----" << std::endl;
        if (ctx) compute_cluster_ea_eom_ccsd(rhf, *ctx, d_eri_mo_precomputed, n_states_requested);
        else     eri_method.compute_ea_eom_ccsd(n_states_requested);
    }
    if (std::getenv("GANSU_STEOM_MEM_DUMP"))
        dump_tracked_allocations("after stage 3 (EA-EOM)");

#ifdef GANSU_MPI
    // EA hand-off rank 1 → rank 0; rank 1 is then done with stage-2.
    if (mpi_steom) {
        if (mpi_rank == 1) {
            send_ea_result(rhf.get_ea_eom_result(), 0);
#ifdef GANSU_MULTI_GPU
            // Step 4b: ship EA bar-H so rank 0 borrows (no rebuild), then free
            // this rank's cache (it owns the published EA tensors; rank 0 keeps
            // its own received copies).
            if (mpi_barh_xfer) {
                send_ea_barh(rhf.steom_barh_cache(), 0);
                std::cout << "  [STEOM MPI] rank 1 sent EA bar-H (Wvovv/Wvvvo) to rank 0." << std::endl;
                rhf.steom_barh_cache().free();
            }
#endif
            std::cout << "  [STEOM MPI] rank 1 sent EA result ("
                      << rhf.get_ea_eom_result().per_active.size()
                      << " active roots) to rank 0; stage-2 done." << std::endl;
            return;  // rank 1 does not run the STEOM second transform
        } else { // rank 0 receives EA result, then proceeds to STEOM
            EAEOMResult ea;
            recv_ea_result(ea, 1);
            std::cout << "  [STEOM MPI] rank 0 received EA result ("
                      << ea.per_active.size() << " active roots) from rank 1." << std::endl;
            rhf.set_ea_eom_result(std::move(ea));
#ifdef GANSU_MULTI_GPU
            // Step 4b: receive rank 1's EA bar-H into rank 0's cache so STEOM
            // borrows all 11 (rank 0's IP already published its 8) — no rebuild.
            if (mpi_barh_xfer) {
                recv_ea_barh(rhf.steom_barh_cache(), 1);
                std::cout << "  [STEOM MPI] rank 0 received EA bar-H from rank 1"
                          << (rhf.steom_barh_cache().complete()
                                  ? " → cache complete, STEOM borrows (no rebuild)."
                                  : " (incomplete → STEOM rebuilds).") << std::endl;
            }
#endif
#if defined(_OPENMP)
            // rank 1 has returned (sent its results) — its CPU share is now free.
            // The STEOM build (G^{1h1p} dressing + geev) is host-heavy; bump rank 0
            // back to the full node so it isn't starved by the per-rank thread cap
            // used during the IP||EA overlap (launcher sets OMP=ncores/local_size).
            {
                const int full = omp_get_num_procs();
                omp_set_num_threads(full);
                if (openblas_set_num_threads) openblas_set_num_threads(full);
                std::cout << "  [STEOM MPI] rank 0 solo phase — threads bumped to "
                          << full << " for the STEOM build." << std::endl;
            }
#endif
        }
    }
#endif

    // (debug accelerator) Save the STEOM checkpoint now that CCSD+IP+EA (+bar-H when
    // sharing) are populated in ctx — a subsequent run with the same file resumes here.
    // Also UPGRADE an IP-only checkpoint after resuming from it: EA was just
    // recomputed (expensive), so persist it — otherwise every resume re-runs EA
    // and the IP-only file never gains the EA stage.
    if (ctx && (!ckpt_loaded || !ckpt_had_ea)) {
        if (const char* ck = std::getenv("GANSU_DMET_STEOM_CKPT"))
            steom_ckpt_save(*ctx, ck);
    }

    // Sanity check — both per_active vectors must now have entries.
    const IPEOMResult& ip_result = ctx ? ctx->ip_eom_result : rhf.get_ip_eom_result();
    const EAEOMResult& ea_result = ctx ? ctx->ea_eom_result : rhf.get_ea_eom_result();
    const CISNTOResult& cis_nto  = ctx ? ctx->cis_nto_result : rhf.get_cis_nto_result();
    if (ip_result.per_active.empty() || ea_result.per_active.empty()) {
        throw std::runtime_error(
            "STEOM-CCSD: P1 IP-EOM and/or P2 EA-EOM did not populate per_active "
            "(no active root survived %singles filter). Try lowering "
            "--ip_eom_ip_thresh / --ea_eom_ea_thresh, or raising "
            "--cis_nto_o_thresh / --cis_nto_v_thresh.");
    }

    const int num_frozen   = ctx ? ctx->get_num_frozen_core() : rhf.get_num_frozen_core();
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
    const int full_occ = (ctx ? ctx->get_num_electrons() : rhf.get_num_electrons()) / 2;
    const int num_basis = ctx ? ctx->get_num_basis() : rhf.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = ctx ? *ctx->C   : rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies   = ctx ? *ctx->eps : rhf.get_orbital_energies();
    const real_t* d_C   = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    if (ctx ? ctx->use_dlpno_amplitudes : rhf.use_dlpno_amplitudes()) {
        // Hybrid DLPNO-STEOM (P5b): inject DLPNO-CCSD T1/T2 (canonical, own copy).
        // Cluster (ctx) runs source the BT set from the context (square-C
        // reduction: points at the RHF's stowed set; rectangular: cluster-space).
        const BTAmplitudes& bt = (ctx && ctx->dlpno_bt) ? *ctx->dlpno_bt
                                                        : rhf.get_dlpno_bt_amplitudes();
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
    } else if (ctx && ctx->cc_ground_cached) {
        // (solve-once) reuse the cluster CCSD ground cached by an earlier stage —
        // IP/EA/STEOM share one cluster ⇒ identical T1/T2. Skips the re-solve.
        const size_t t1n = (size_t)nocc_active*nvir;
        const size_t t2n = (size_t)nocc_active*nocc_active*(size_t)nvir*nvir;
        tracked_cudaMalloc(&d_t1, t1n*sizeof(real_t));
        tracked_cudaMalloc(&d_t2, t2n*sizeof(real_t));
        cudaMemcpy(d_t1, ctx->cc_t1, t1n*sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_t2, ctx->cc_t2, t2n*sizeof(real_t), cudaMemcpyDeviceToDevice);
        std::cout << "  STEOM CCSD ground reused from cache (solve-once): " << std::fixed
                  << std::setprecision(10) << ctx->cc_E << " Ha" << std::endl;
    } else {
        // Inc3: cluster storage-free ground CCSD — when the cluster MO-ERI is NOT
        // precomputed (block mode) hand the RI engine to ccsd_spatial_orbital so it
        // builds its own cluster B (build_B_mo over d_C = C_can) and runs the
        // B-native residual (Inc1/2) instead of needing a dense n_emb⁴ tensor.
        const ERI_RI* ccsd_eri_ri =
            (ctx && ctx->prefer_ri_block && d_eri_mo_precomputed == nullptr)
                ? dynamic_cast<const ERI_RI*>(&eri_method) : nullptr;
        real_t E_CCSD = ccsd_spatial_orbital(
            d_eri_ao, d_C, d_eps, num_basis, full_occ,
            /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
            &d_t1, &d_t2,
            d_eri_mo_precomputed,
            num_frozen,
            /*h_fov_active=*/nullptr, /*eri_ri=*/ccsd_eri_ri,
            /*level_shift=*/(ctx ? ctx->level_shift : 0.0));
        std::cout << "  STEOM CCSD ground-state re-solve: " << std::fixed << std::setprecision(10)
                  << E_CCSD << " Ha   (in " << std::setprecision(3)
                  << ccsd_timer.elapsed_seconds() << " s)" << std::endl;
        // (solve-once) cache the cluster ground for any later reuse.
        if (ctx && !std::getenv("GANSU_STEOM_NO_CCSD_CACHE")) {
            const size_t t1n = (size_t)nocc_active*nvir;
            const size_t t2n = (size_t)nocc_active*nocc_active*(size_t)nvir*nvir;
            tracked_cudaMalloc(&ctx->cc_t1, t1n*sizeof(real_t));
            tracked_cudaMalloc(&ctx->cc_t2, t2n*sizeof(real_t));
            cudaMemcpy(ctx->cc_t1, d_t1, t1n*sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(ctx->cc_t2, d_t2, t2n*sizeof(real_t), cudaMemcpyDeviceToDevice);
            ctx->cc_t1n=t1n; ctx->cc_t2n=t2n; ctx->cc_E=E_CCSD; ctx->cc_ground_cached=true;
        }
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
    // Inc3: the cluster (ctx) path also takes the block source when
    // GANSU_DMET_STEOM_RI_BLOCK is set (ctx->prefer_ri_block) — d_C = ctx->C
    // (cluster C_can, [nao_ao × n_emb]) and num_basis = n_emb, so build_B_mo
    // produces the cluster B_mo (naux × n_emb²), never the n_emb⁴ tensor.
    const bool want_block = ((ctx ? ctx->use_dlpno_amplitudes : rhf.use_dlpno_amplitudes())
                             || (ctx && ctx->prefer_ri_block))
                            && gpu::gpu_available();
    if (want_block) {
        eri_ri_block = dynamic_cast<const ERI_RI*>(&eri_method);
        if (eri_ri_block) {
            d_B_mo_blocks = eri_ri_block->build_B_mo(d_C, num_basis);
            if (!d_B_mo_blocks) eri_ri_block = nullptr;  // CPU / budget fail
        }
    }
    // Inc3: publish the cluster B so the cluster-stage free functions
    // (compute_cluster_cis_nto / ip / ea) hand it to their B-source-capable
    // operators instead of the dense d_eri_mo (which the driver no longer built).
    if (ctx) { ctx->eri_block_src = eri_ri_block; ctx->d_B_mo_blocks = d_B_mo_blocks; }

    Timer mo_timer;
    real_t* d_eri_mo = nullptr;
    bool free_eri_mo = false;
    if (!eri_ri_block) {
        if (d_eri_mo_precomputed) {
            d_eri_mo = d_eri_mo_precomputed;  // RI path — caller owns / frees it
        } else if (ctx) {
            // Inc3 cluster fallback: block requested but build_B_mo refused (CPU /
            // budget). There is no d_eri_ao to AO→MO transform here — build the
            // dense cluster MO-ERI from B instead. Small clusters only; large ones
            // are exactly why the block path exists (so this should not be reached).
            d_eri_mo = eri_method.build_mo_eri(d_C, num_basis);
            free_eri_mo = true;
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
    // (denom-only level shift — EOM ε un-shift) See eri_stored_ip_eom_ccsd.cu: the
    // virtual-ε shift (+s, dmet.cu) that stabilises the cluster CCSD must not leak into
    // the STEOM operator (F^eff_vv / G^{1h1p} diagonal), else every root gains ~+s
    // (verified log184: roots drift by exactly s). Un-shift ε here — the ground CCSD
    // already ran with its shifted ε above; done before the device-balance peer copy so
    // the best-device ε copy is un-shifted too. Default-on (correctness fix, validated
    // log185); opt-out GANSU_DMET_STEOM_EOM_UNSHIFT=0. level_shift==0 ⇒ untouched.
    real_t* d_eps_unshifted = nullptr;
    const char* eu_env = std::getenv("GANSU_DMET_STEOM_EOM_UNSHIFT");
    if (ctx && ctx->level_shift != 0.0 && !(eu_env && eu_env[0] == '0')) {
        std::vector<real_t> h_eps(num_basis);
        cudaMemcpy(h_eps.data(), d_eps, (size_t)num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);
        for (int i = full_occ; i < num_basis; ++i) h_eps[i] -= ctx->level_shift;
        tracked_cudaMalloc(&d_eps_unshifted, (size_t)num_basis * sizeof(real_t));
        cudaMemcpy(d_eps_unshifted, h_eps.data(), (size_t)num_basis * sizeof(real_t), cudaMemcpyHostToDevice);
        d_eps = d_eps_unshifted;
        std::cout << "  [DMET-STEOM] STEOM operator ε un-shifted (−s=" << std::fixed
                  << std::setprecision(4) << ctx->level_shift << ") — true excitation spectrum"
                  << std::defaultfloat << std::endl;
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
    real_t* d_eps_bd = nullptr;   // cluster share-barH: best-device ε copy (freed at restore)
#ifndef GANSU_CPU_ONLY
    if (!ctx && gpu::gpu_available()) {   // cluster: small, single-GPU → no device audit/balance
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

#ifndef GANSU_CPU_ONLY
    // (perf, user request) Cluster share-barH: STEOM is built on the GPU that already
    // holds IP's 8 bar-H (barh.ip_dev — moved off device 0 before the EA solve). Bring
    // EA's 3 bar-H there too, migrate the operator's other device inputs (T1/T2 in place,
    // ε, and the B_mo block source), then cudaSetDevice + GPUHandle::reset so all W^eff
    // GEMMs + geev run there (device 0 stays free of the ~28 GB bar-H). Restored at impl
    // exit. Skips the ~320 s build_dressed (borrow) AND avoids the device-0 OOM.
    if (ctx && ctx->share_barh && gpu::gpu_available()
        && ctx->barh.complete() && !steom_dev_balance_active) {
        int saved = 0; cudaGetDevice(&saved);
        // Target GPU: where IP's bar-H already are (choreography), else the freest
        // peer (e.g. a full-checkpoint resume that loaded all 11 onto device 0).
        int best = ctx->barh.ip_dev;
        // (run4 wall) The bar-H device may lack room for the W^eff build working
        // set on top of the 11 borrowed bar-H (Doxorubicin cc-pVDZ: GPU2 OOM at a
        // 5.87 GB alloc after F_eff/hp). GANSU_STEOM_BARH_GPU=<n> forces the STEOM
        // build (and the bar-H migration) onto GPU n instead.
        if (const char* e = std::getenv("GANSU_STEOM_BARH_GPU")) {
            if (e[0] >= '0' && e[0] <= '9') {
                best = std::atoi(e);
                std::cout << "  [DMET-STEOM share-barH] target GPU " << best
                          << " forced (GANSU_STEOM_BARH_GPU)." << std::endl;
            }
        }
        if (best == saved) {
            int n_dev = 0; cudaGetDeviceCount(&n_dev);
            if (n_dev > 1) {
                std::vector<size_t> freeb(n_dev, 0);
                for (int d = 1; d < n_dev; ++d) { cudaSetDevice(d); size_t fb=0,tb=0;
                    if (cudaMemGetInfo(&fb,&tb) == cudaSuccess) freeb[d] = fb; }
                cudaSetDevice(saved);
                best = 1; size_t bf = freeb[1];
                for (int d = 2; d < n_dev; ++d) if (freeb[d] > bf) { bf = freeb[d]; best = d; }
            }
        }
        if (best != saved) {
            std::cout << "  [DMET-STEOM share-barH] building STEOM operator on GPU " << best
                      << " (bar-H device); migrating bar-H + T1/T2 there." << std::endl;
            ctx->barh.migrate_ip_to(best);                     // no-op if already on best
            ctx->barh.migrate_ea_to(best);                     // EA's 3 join IP's 8 on best
            auto move_ip = [&](real_t*& p, size_t n) {          // in-place peer migrate
                if (!p) return;
                cudaSetDevice(best);
                real_t* np = nullptr; tracked_cudaMalloc(&np, n * sizeof(real_t));
                cudaMemcpyPeer(np, best, p, saved, n * sizeof(real_t));
                tracked_cudaFree(p); p = np;
                cudaSetDevice(saved);
            };
            move_ip(d_t1, (size_t)nocc_active * nvir);                       // operator owns
            move_ip(d_t2, (size_t)nocc_active * nocc_active * (size_t)nvir * nvir);
            cudaSetDevice(best);
            tracked_cudaMalloc(&d_eps_bd, (size_t)nao_active * sizeof(real_t));
            cudaMemcpyPeer(d_eps_bd, best, d_eps_active, saved,
                           (size_t)nao_active * sizeof(real_t));
            gpu::GPUHandle::reset();                            // cuBLAS/cuSOLVER on best
            if (eri_ri_block) {
                // Rebuild the B_mo block source on best (uses best's B replica);
                // build_B_mo needs C on best — copy it transiently.
                const size_t Csz = coefficient_matrix.rows() * coefficient_matrix.cols();
                real_t* d_C_bd = nullptr; tracked_cudaMalloc(&d_C_bd, Csz * sizeof(real_t));
                cudaMemcpyPeer(d_C_bd, best, d_C, saved, Csz * sizeof(real_t));
                d_B_mo_blocks = eri_ri_block->build_B_mo(d_C_bd, num_basis);
                tracked_cudaFree(d_C_bd);
            }
            d_eps_active = d_eps_bd;                            // ctor reads ε on best
            steom_dev_balance_active  = true;
            steom_dev_balance_restore = saved;
            steom_dev_balance_target  = best;
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
                               (ctx ? ctx->share_barh : rhf.steom_share_barh())
                                   ? (ctx ? &ctx->barh : &rhf.steom_barh_cache()) : nullptr,
                               // Frozen core: block ranges read [num_frozen, num_basis)
                               // of the full-C B_mo (only used on the block path).
                               num_frozen,
                               // Spin block: triplet G = F_eff − g_phph (--spin_type).
                               steom_triplet);

    // Operator owns T1/T2 + has copied bar-H intermediates; we can free the
    // trimmed / full MO ERI tensor (operator pulled the sub-blocks it needs).
    if (free_eri_for_op) tracked_cudaFree(d_eri_for_op);
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    if (d_eps_unshifted) tracked_cudaFree(d_eps_unshifted);  // EOM ε un-shift copy (ctor consumed it)

    std::cout << "  Operator build time: " << std::fixed << std::setprecision(3)
              << build_timer.elapsed_seconds() << " s "
              << "(sub-phase 3.5-3.7: ERI blocks + 11 bar-H intermediates + X(MI)/X(EA) "
                 "+ F^eff_oo/vv + full W^eff-dressed G^{1h1p}; apply = dense non-Hermitian "
                 "matvec)" << std::endl;
    if (verbose >= 2) steom_op.print_amplitude_norms(std::cout);
    // (measure-first) Checkpoint after the STEOM operator (G^{1h1p}) is built.
    // At this point G is dense in device memory; the dense geev below needs
    // ~3·total_dim² more. This dump shows whether stale IP/EA/B buffers can be
    // freed first to make the dense path fit at total_dim≈22200 ({0-9}).
    if (std::getenv("GANSU_STEOM_MEM_DUMP"))
        dump_tracked_allocations("after STEOM operator build (G ready, before diag)");

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
    // Final non-Hermitian geev: GPU cusolverXgeev by default (fast). Eigen CPU geev
    // is deterministic but O(n^3) single-threaded — far too slow at total_dim≈3500+
    // to be a default. Opt into the deterministic CPU path with GANSU_STEOM_GEEV_HOST=1
    // (only viable for small total_dim).
    const char* geev_host_env = std::getenv("GANSU_STEOM_GEEV_HOST");
    const bool  geev_host = (geev_host_env != nullptr) && (std::atoi(geev_host_env) != 0);
    const bool dense_diag = (steom_op.get_G_device() != nullptr)
                            && !force_davidson
                            && (force_dense || total_dim <= dense_auto_max);

    std::vector<real_t> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_states_to_compute * total_dim);

    if (dense_diag) {
        const real_t min_eigenvalue = 0.0;  // STEOM excitation energies are positive

        // (blocker (a) fix) Reclaim the RI AO-B replica before the dense geev.
        // The STEOM operator's dense G is already built; from here on the diag is
        // pure matvec on G and never pulls MO-ERI blocks again, so the per-GPU
        // AO-B replica (naux·nao², ~17.6 GB/GPU on {0-9}) is dead weight that
        // otherwise OOMs the device-0 dense buffers (G copy + evecs + geev
        // workspace ≈ tens of GB at total_dim≈22200). release_bmo_ao_replica is
        // the established lazy-replica reclaim hook (no-op for single-GPU /
        // non-distributed; a later build_B_mo would re-replicate — none follows
        // STEOM in this run). Default on; opt out with GANSU_STEOM_KEEP_B_REPLICA.
        if (!std::getenv("GANSU_STEOM_KEEP_B_REPLICA")) {
            // release_bmo_ao_replica is declared on ERI_RI (not the ERI base), so
            // resolve via dynamic_cast; no-op for non-RI / single-GPU engines.
            ERI_RI* eri_ri = dynamic_cast<ERI_RI*>(&eri_method);
            if (eri_ri) {
                const size_t before = GlobalGpuMemoryTracker::get_current();
                eri_ri->release_bmo_ao_replica();
                const size_t after = GlobalGpuMemoryTracker::get_current();
                if (before > after)
                    std::cout << "  [STEOM dense diag] released RI AO-B replica before geev: "
                              << CudaMemoryManager<real_t>::format_bytes(before - after)
                              << " reclaimed (global)." << std::endl;
#ifndef GANSU_CPU_ONLY
                // free_replicated_B resets the current device to 0; when the operator +
                // G live on a device-balanced GPU, restore it so the geev below runs on
                // the device that holds d_G_ (else cusolverXgeev fails, info=-1).
                if (steom_dev_balance_active) cudaSetDevice(steom_dev_balance_target);
#endif
            }
        }

        // eigenDecompositionNonSymmetric expects column-major input. d_G_ is
        // row-major [total_dim × total_dim]; its linear buffer is exactly the
        // column-major storage of Gᵀ. eig(Gᵀ) == eig(G), so passing the buffer
        // directly yields the correct eigenvalues; to also recover the RIGHT
        // eigenvectors of G we transpose the copy into true column-major G.
        if (std::getenv("GANSU_STEOM_MEM_DUMP")) {
            const size_t need = (size_t)total_dim * total_dim * sizeof(real_t);
            std::cout << "  [STEOM dense diag] about to allocate G copy + eval/evec/"
                         "workspace; d_G_cm alone = "
                      << CudaMemoryManager<real_t>::format_bytes(need)
                      << " (total_dim=" << total_dim << ")" << std::endl;
            dump_tracked_allocations("just before dense geev allocations");
        }
        real_t* d_G_cm = nullptr;
        tracked_cudaMalloc(&d_G_cm, (size_t)total_dim * total_dim * sizeof(real_t));
        cudaMemcpy(d_G_cm, steom_op.get_G_device(),
                   (size_t)total_dim * total_dim * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        gpu::transposeMatrixInPlace(d_G_cm, total_dim);  // row-major G → column-major G

        // Robustness rescue (GANSU_STEOM_SYMMETRIZE=1, OPT-IN, default OFF ⇒
        // byte-identical). The analytic 2nd-order STEOM G is non-Hermitian and, at
        // near-degeneracies (floppy / poorly-optimised geometries), a low physical
        // state can emerge as a spurious complex-conjugate pair. That complex-ness is
        // a TRUNCATION ARTIFACT — the true all-orders STEOM G (implicit triples) is
        // real-diagonalisable (verified: script/steom_es_complex_test.py — H2O triplet
        // analytic max|Im|=0.78 eV → determinant {e^S} 0.0). Replacing G by ½(G+Gᵀ)
        // yields a real spectrum whose complex-derived roots land near the true {e^S}
        // values (2-octanone idealised n→π* 5.40+complex → 4.86 real ≈ ORCA 4.64).
        // ⚠ It is a UNIFORM approximation: it also shifts the already-real states by
        // ~0.1-0.3 eV (2-octanone_opt 4.92→4.65), so it is NOT auto-applied — on a
        // GOOD geometry the non-Hermitian roots match the full canonical STEOM and
        // should be trusted. Enable it only to turn complex garbage into a usable
        // (approximate) real spectrum on a run that reports near-defective roots.
        // The true fix for the energy is implicit triples (not this). One host pass.
#ifndef GANSU_CPU_ONLY
        if (std::getenv("GANSU_STEOM_SYMMETRIZE")) {
            std::vector<real_t> hG((size_t)total_dim * total_dim);
            cudaMemcpy(hG.data(), d_G_cm,
                       (size_t)total_dim * total_dim * sizeof(real_t), cudaMemcpyDeviceToHost);
            for (int i = 0; i < total_dim; ++i)          // ½(G+Gᵀ); transpose-invariant
                for (int j = i + 1; j < total_dim; ++j) {
                    const size_t a = (size_t)i * total_dim + j, b = (size_t)j * total_dim + i;
                    const real_t s = real_t(0.5) * (hG[a] + hG[b]);
                    hG[a] = s; hG[b] = s;
                }
            cudaMemcpy(d_G_cm, hG.data(),
                       (size_t)total_dim * total_dim * sizeof(real_t), cudaMemcpyHostToDevice);
            std::cout << "  [STEOM] G symmetrized ½(G+Gᵀ) before geev "
                         "(GANSU_STEOM_SYMMETRIZE, opt-in rescue) — real eigenvalues; "
                         "complex near-defective roots regularized (approximate; also "
                         "shifts real states ~0.2 eV — use only on near-defective runs)."
                      << std::endl;
        }
#endif

        real_t* d_all_evals = nullptr;
        real_t* d_all_evals_imag = nullptr;
        real_t* d_all_evecs = nullptr;
        tracked_cudaMalloc(&d_all_evals, (size_t)total_dim * sizeof(real_t));
        tracked_cudaMalloc(&d_all_evals_imag, (size_t)total_dim * sizeof(real_t));
        tracked_cudaMalloc(&d_all_evecs, (size_t)total_dim * total_dim * sizeof(real_t));

        // Complex-root recovery (default ON; opt out with GANSU_STEOM_DROP_COMPLEX=1).
        // The non-Hermitian STEOM G is near-defective at D2h near-degeneracies, so
        // genuine low valence states (La/Lb) emerge as complex-CONJUGATE eigenvalue
        // pairs. The legacy path silently dropped them (the P2 "low states missing"
        // bug → root0 jumped to a higher real root). Now that the doubles bug is
        // fixed the real parts are physical, so we keep the FULL spectrum and
        // collapse each c.c. pair to one real state (its real part = the physical
        // excitation energy), with a "handle with care" warning.
        const bool drop_complex = (std::getenv("GANSU_STEOM_DROP_COMPLEX") != nullptr);
        int info = gpu::eigenDecompositionNonSymmetric(
            d_G_cm, d_all_evals, d_all_evecs, total_dim, geev_host,
            drop_complex ? nullptr : d_all_evals_imag);
        if (info != 0) {
            std::cout << "Warning: STEOM dense diagonalization (geev) returned info="
                      << info << "." << std::endl;
        }

        // Eigenvalues sorted ascending by real part. In recovery mode the FULL
        // spectrum is kept and h_all_evals_imag holds the imag parts; in legacy
        // (drop) mode complex roots were filled with 1e30 and imag stays 0.
        std::vector<real_t> h_all_evals(total_dim);
        std::vector<real_t> h_all_evals_imag(total_dim, 0.0);
        cudaMemcpy(h_all_evals.data(), d_all_evals,
                   (size_t)total_dim * sizeof(real_t), cudaMemcpyDeviceToHost);
        if (!drop_complex)
            cudaMemcpy(h_all_evals_imag.data(), d_all_evals_imag,
                       (size_t)total_dim * sizeof(real_t), cudaMemcpyDeviceToHost);

        // (diagnostic) Dump the lowest G eigenvalues (real + imag parts) so we can
        // see where the physical low states (e.g. azobenzene n→π* ≈ 0.103 Ha) sit
        // and which roots are complex / near-defective. Gated by env, behaviour-inert.
        if (std::getenv("GANSU_STEOM_DUMP_SPECTRUM")) {
            const int ndump = std::min(total_dim, 30);
            std::cout << "  [STEOM spectrum dump] lowest " << ndump
                      << " G eigenvalues (real-sorted; |Im|>0 ⇒ near-defective complex root):\n"
                      << "      k      Re(Ha)        Re(eV)        Im(Ha)        Im(eV)" << std::endl;
            for (int i = 0; i < ndump; ++i) {
                std::cout << "    " << std::setw(4) << i
                          << "  " << std::fixed << std::setprecision(8) << std::setw(12) << h_all_evals[i]
                          << "  " << std::setprecision(4) << std::setw(10) << (h_all_evals[i] * 27.2114)
                          << "  " << std::scientific << std::setprecision(3) << std::setw(11) << h_all_evals_imag[i]
                          << "  " << std::fixed << std::setprecision(4) << std::setw(10)
                          << (h_all_evals_imag[i] * 27.2114)
                          << std::endl;
            }
            std::cout << std::defaultfloat;
        }

        // Select the lowest n_states roots ≥ min_eigenvalue by real part,
        // collapsing each complex-conjugate pair to a single state (skip the
        // partner with the same real part and opposite imag).
        const real_t kImagTol = 1e-6;   // |Im| ≥ this ⇒ complex root
        const real_t kPairTol = 1e-6;   // c.c.-partner match tolerance (Ha)
        std::vector<int>  sel;
        std::vector<bool> sel_complex;
        sel.reserve(n_states_to_compute);
        for (int i = 0; i < total_dim && (int)sel.size() < n_states_to_compute; ++i) {
            if (!(h_all_evals[i] >= min_eigenvalue && h_all_evals[i] < 1e29)) continue;
            const bool cplx = std::abs(h_all_evals_imag[i]) >= kImagTol;
            if (cplx) {
                bool partner_already = false;
                for (int s : sel) {
                    if (std::abs(h_all_evals[s]      - h_all_evals[i])      < kPairTol &&
                        std::abs(h_all_evals_imag[s] + h_all_evals_imag[i]) < kPairTol) {
                        partner_already = true; break;   // c.c. partner already counted
                    }
                }
                if (partner_already) continue;
            }
            sel.push_back(i);
            sel_complex.push_back(cplx);
        }
        if ((int)sel.size() < n_states_to_compute) {
            std::cout << "Warning: STEOM dense diagonalization found only " << sel.size()
                      << " states ≥ " << min_eigenvalue << " (requested "
                      << n_states_to_compute << ")." << std::endl;
        }
        {
            int n_cplx = 0; real_t max_imag = 0.0;
            for (size_t n = 0; n < sel.size(); ++n)
                if (sel_complex[n]) {
                    ++n_cplx;
                    max_imag = std::max(max_imag, std::abs(h_all_evals_imag[sel[n]]));
                }
            if (n_cplx > 0)
                std::cout << "  [STEOM complex-root recovery] " << n_cplx << "/" << sel.size()
                          << " reported states are complex-conjugate pairs collapsed to their "
                             "real part (near-defective G — handle with care; max|Im|="
                          << std::scientific << std::setprecision(2) << max_imag << " Ha = "
                          << std::fixed << std::setprecision(3) << (max_imag * 27.2114)
                          << " eV)." << std::endl;
        }

        eigenvalues.assign(n_states_to_compute, 1e30);
        for (int n = 0; n < (int)sel.size(); ++n)
            eigenvalues[n] = h_all_evals[sel[n]];
        // Strided D2H copy of the selected eigenvectors (transposed layout:
        // element j of eigenvector idx is at d_all_evecs[idx + j*total_dim]).
        // For a recovered complex root this is the real part of the (complex)
        // right eigenvector — exact for the energy; approximate for %active.
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
        tracked_cudaFree(d_all_evals_imag);
        tracked_cudaFree(d_all_evecs);

        std::cout << "  STEOM-CCSD solve time: " << std::fixed << std::setprecision(3)
                  << solve_timer.elapsed_seconds() << " s "
                  << "(dense non-Hermitian geev on "
                  << (geev_host ? "CPU/Eigen [deterministic]" : "GPU/cusolverXgeev")
                  << "; "
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

    // sub-phase 3.10 — per-root % active character η. Project each STEOM right
    // eigenvector R1[i,a] (i∈nocc_active, a∈nvir) onto the active CIS-NTO
    // subspace: the first cis_nto.n_act_occ columns of U_occ (occ side) and the
    // first n_act_vir columns of U_vir (vir side). η = the JOINT active-active
    // weight fraction; percent_active_occ / _vir are the per-side marginals. A
    // low η flags a root the active space (≡ the DMET cluster / bath, §4.3) does
    // not describe — its STEOM energy is then untrustworthy. Computed only when
    // the CIS-NTO basis matches the STEOM (nocc_active, nvir); else sentinels.
    const bool can_eta = (cis_nto.nocc_active == nocc_active) && (cis_nto.nvir == nvir)
                         && !cis_nto.U_occ.empty() && !cis_nto.U_vir.empty()
                         && cis_nto.n_act_occ > 0 && cis_nto.n_act_vir > 0;
    const int nao_occ = can_eta ? cis_nto.n_act_occ : 0;   // # active occ NTOs
    const int nav_vir = can_eta ? cis_nto.n_act_vir : 0;   // # active vir NTOs
    for (int n = 0; n < n_states_to_compute; ++n) {
        auto& pr = result.per_root[n];
        pr.omega = eigenvalues[n];
        pr.R1.assign(&h_eigenvectors[(size_t)n * total_dim],
                     &h_eigenvectors[(size_t)n * total_dim + total_dim]);
        if (!can_eta) continue;
        const real_t* R1 = pr.R1.data();
        double norm2 = 0.0;
        for (int x = 0; x < total_dim; ++x) norm2 += (double)R1[x] * R1[x];
        if (!(norm2 > 0.0)) continue;
        // T[ã,a] = Σ_i U_occ[i,ã] R1[i,a]  (occ index → active occ NTO basis)
        std::vector<double> T((size_t)nao_occ * nvir, 0.0);
        for (int am = 0; am < nao_occ; ++am)
            for (int i = 0; i < nocc_active; ++i) {
                const double u = (double)cis_nto.U_occ[(size_t)i * nocc_active + am];
                if (u == 0.0) continue;
                for (int a = 0; a < nvir; ++a)
                    T[(size_t)am * nvir + a] += u * (double)R1[(size_t)i * nvir + a];
            }
        double p_occ = 0.0;      // Σ_{active ã, all a} T²  (occ-side active fraction)
        for (size_t x = 0; x < T.size(); ++x) p_occ += T[x] * T[x];
        double joint = 0.0;      // Σ_{active ã, active ẽ} (Σ_a T[ã,a] U_vir[a,ẽ])²
        for (int am = 0; am < nao_occ; ++am)
            for (int em = 0; em < nav_vir; ++em) {
                double s = 0.0;
                for (int a = 0; a < nvir; ++a)
                    s += T[(size_t)am * nvir + a] * (double)cis_nto.U_vir[(size_t)a * nvir + em];
                joint += s * s;
            }
        double p_vir = 0.0;      // Σ_{active ẽ, all i} (Σ_a R1[i,a] U_vir[a,ẽ])²  (vir-side)
        for (int em = 0; em < nav_vir; ++em)
            for (int i = 0; i < nocc_active; ++i) {
                double s = 0.0;
                for (int a = 0; a < nvir; ++a)
                    s += (double)R1[(size_t)i * nvir + a] * (double)cis_nto.U_vir[(size_t)a * nvir + em];
                p_vir += s * s;
            }
        pr.percent_active_occ = (real_t)(p_occ / norm2);
        pr.percent_active_vir = (real_t)(p_vir / norm2);
        pr.eta                = (real_t)(joint / norm2);
    }

    // ----------------------------------------------------------------------
    // Step 5: human-readable report
    // ----------------------------------------------------------------------
    std::ostringstream os;
    os << "[STEOM-CCSD] sub-phase 3.5-3.7 — full W^eff-dressed G^{1h1p} "
       << (steom_triplet ? "TRIPLET" : "singlet") << " block "
          "(Nooijen-Bartlett Eq.34-63: F^eff_oo/vv + hp/hhhp/phph"
       << (steom_triplet ? "" : "/phhp") << " + IP×EA cross), "
          "diagonalized by "
       << (dense_diag ? "dense non-Hermitian geev (deterministic)"
                      : "non-Hermitian Davidson")
       << (steom_triplet
           ? ". Validated vs Python reference (triplet = F_eff − g_phph, routes "
             "corrected 2026-06-20): CH2O sto-3g fc2 triplets 4.223 / 5.904 eV "
             "(ORCA 4.197 / 5.959); H2O sto-3g FC1 lowest ~10.25 eV.\n"
           : ". Validated vs Python reference (W^eff routes corrected 2026-06-20): "
             "H2O sto-3g FC1 lowest two roots 0.432663 / 0.496991 Ha "
             "(11.773 / 13.524 eV; ORCA 11.849 / 13.60).\n")
       << "  STEOM " << (steom_triplet ? "TRIPLET " : "") << "excited-state energies"
       << (can_eta ? "  (η = % active character; low η ⇒ active space / DMET bath "
                     "under-describes this root, energy untrustworthy)" : "")
       << ":\n"
       << "   k   omega (Ha)        omega (eV)"
       << (can_eta ? "      η        %act_o   %act_v" : "") << "\n";
    // ORCA convergence criterion for a STEOM root w.r.t. the active space:
    // the root is "converged" when its % active character η ≥ 0.96 (DLPNO-STEOM)
    // / 0.98 (canonical STEOM). Below that ORCA flags the excitation energy as
    // unreliable; the remedy is to enlarge the active space — tighten
    // OThresh/VThresh, raise the IP/EA safety margin, or lower TCutPNOSingles
    // (Dutta, Nooijen, Neese, Izsák, "Automatic active space selection for the
    // STEOM-CCSD method", J. Chem. Theory Comput. 2018). Report-only, numerically
    // inert; override the flag threshold with GANSU_STEOM_ETA_THRESH.
    real_t eta_thresh = static_cast<real_t>(0.96);
    if (const char* e = std::getenv("GANSU_STEOM_ETA_THRESH")) {
        const double v = std::atof(e);
        if (v > 0.0 && v <= 1.0) eta_thresh = static_cast<real_t>(v);
    }
    // (diagnostic, gated by GANSU_STEOM_DUMP_SPECTRUM) dominant 1h1p components
    // per root in canonical MO labels (0-based, same numbering as ORCA's STEOM
    // amplitude lists: HOMO = num_occ−1, LUMO = num_occ) — for state assignment
    // against external references. Report-only, numerically inert.
    const bool dump_comp = (std::getenv("GANSU_STEOM_DUMP_SPECTRUM") != nullptr);
    int n_below_eta = 0;
    for (int n = 0; n < n_states_to_compute; ++n) {
        const auto& pr = result.per_root[n];
        os << "  " << std::setw(2) << n
           << "   " << std::setw(12) << std::setprecision(8) << std::fixed << pr.omega
           << "   " << std::setw(10) << std::setprecision(4) << (pr.omega * 27.2114);
        if (can_eta && pr.eta >= 0.0) {
            os << "   " << std::setw(7) << std::setprecision(4) << pr.eta
               << "  " << std::setw(7) << std::setprecision(4) << pr.percent_active_occ
               << "  " << std::setw(7) << std::setprecision(4) << pr.percent_active_vir;
            if (pr.eta < eta_thresh) { os << "  ⚠ <" << std::setprecision(2) << eta_thresh; ++n_below_eta; }
        }
        os << "\n";
        if (dump_comp && (int)pr.R1.size() == total_dim && total_dim > 0) {
            const real_t* R1 = pr.R1.data();
            double norm2 = 0.0;
            for (int x = 0; x < total_dim; ++x) norm2 += (double)R1[x] * R1[x];
            const double inv = (norm2 > 0.0) ? 1.0 / std::sqrt(norm2) : 0.0;
            const int ntop = std::min(6, total_dim);
            std::vector<int> ord(total_dim);
            for (int x = 0; x < total_dim; ++x) ord[x] = x;
            std::partial_sort(ord.begin(), ord.begin() + ntop, ord.end(),
                [&](int x, int y) {
                    return std::fabs(R1[x]) > std::fabs(R1[y]);
                });
            for (int t = 0; t < ntop; ++t) {
                const int x = ord[t];
                const double c = (double)R1[x] * inv;
                if (std::fabs(c) < 0.10) break;
                const int i = x / nvir, a = x % nvir;
                os << "        " << (c >= 0.0 ? "+" : "-") << std::setw(6)
                   << std::setprecision(4) << std::fixed << std::fabs(c)
                   << "   " << std::setw(3) << (num_frozen + i) << " -> "
                   << std::setw(3) << (num_frozen + nocc_active + a) << "\n";
            }
        }
    }
    if (!can_eta)
        os << "  (η = % active character: not computed — CIS-NTO basis "
              "dimensions did not match the STEOM active space.)\n";

    // Active-space health summary (ORCA-style, report-only). Two things a low-η
    // spectrum tells us: (1) how many computed roots fall below the ORCA η
    // threshold (their energies are untrustworthy — enlarge the active space),
    // and (2) the active-space provenance chain — the requested CIS-NTO count
    // vs the EFFECTIVE count that actually enters the second similarity
    // transform. NTOs are silently dropped when their IP/EA-EOM root fails the
    // %singles filter or gets no FollowCIS assignment (canonical_*_label < 0),
    // so a spectrum can "lose" low states not because the physics is missing but
    // because the effective active space quietly shrank below the requested one.
    // This is exactly the diagnostic to read when low valence states go missing.
    if (can_eta) {
        os << "  active-space health: " << n_below_eta << "/" << n_states_to_compute
           << " root(s) below η = " << std::setprecision(2) << eta_thresh
           << (n_below_eta > 0
                 ? "  ⚠ enlarge active space (tighten OThresh/VThresh, raise IP/EA "
                   "safety margin, lower TCutPNOSingles)"
                 : "  (all roots converged w.r.t. active space)")
           << "\n";
        os << "  active occ: requested(CIS-NTO)=" << cis_nto.n_act_occ
           << " → routed=" << n_act_occ << " → effective=" << n_act_occ_eff
           << "   |   active vir: requested=" << cis_nto.n_act_vir
           << " → routed=" << n_act_vir << " → effective=" << n_act_vir_eff << "\n";
        if (n_act_occ_eff < cis_nto.n_act_occ || n_act_vir_eff < cis_nto.n_act_vir)
            os << "  ⚠ effective active space is SMALLER than requested — "
               << (cis_nto.n_act_occ - n_act_occ_eff) << " occ / "
               << (cis_nto.n_act_vir - n_act_vir_eff) << " vir NTO(s) dropped "
                  "(unassigned or failed the IP/EA %singles filter). Enlarging the "
                  "CIS-NTO active space will NOT help these back in; raise the IP/EA "
                  "safety margin and/or loosen ip_thresh/ea_thresh instead.\n";
    }

    result.report = os.str();
    std::cout << result.report;

    if (ctx) { ctx->excited_state_report += result.report;
               ctx->steom_result = std::move(result); }
    else     { rhf.append_excited_state_report(result.report);
               rhf.set_steom_result(std::move(result)); }

    // (A) shared bar-H: release the cache device buffers. Safe here — the STEOM
    // operator only reads bar-H during its build (build_F_eff_*/build_W_eff_and_G);
    // the apply()/matvec consumed by the solve above uses the dense d_G_ only, so
    // the borrowed bar-H pointers are no longer referenced. steom_op (still in
    // scope) borrowed them and its dtor skips freeing (barh_borrowed_), so this is
    // the single owner-side release with no double-free.
    if (ctx) {
        if (ctx->share_barh) { ctx->barh.free(); ctx->share_barh = false; }
    } else if (rhf.steom_share_barh()) {
        rhf.steom_barh_cache().free();
        rhf.set_steom_share_barh(false);
    }

    // Ship 11 — restore caller's device after STEOM operator + Davidson are
    // both destructed (RAII), so downstream output / API code sees the same
    // device it set before calling STEOM.
#ifndef GANSU_CPU_ONLY
    if (d_eps_bd) tracked_cudaFree(d_eps_bd);   // cluster share-barH best-device ε copy
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

// DMET-STEOM standalone cluster entry (declared in eom_chain_context.hpp). Wraps
// the raw cluster arrays in an EOMChainContext + DeviceHostMatrix views and runs
// the canonical STEOM chain over the precomputed cluster MO-ERI. The full chain
// (CIS-NTO/IP/EA/STEOM) reads its electronic state from ctx, never from `cfg`
// (which supplies only config + AO basis). Mirrors ccsd_spatial_orbital (dmet.cu).
STEOMResult steom_spatial_orbital(RHF& cfg, ERI& eri_method,
                                  const real_t* d_C_can,
                                  const real_t* d_eps,
                                  real_t* d_eri_mo,
                                  int nao, int n_emb, int n_emb_occ,
                                  int n_states, int n_frozen,
                                  real_t level_shift)
{
    PROFILE_FUNCTION();
    if (n_emb_occ <= n_frozen || n_emb_occ >= n_emb)
        throw std::runtime_error("steom_spatial_orbital: invalid cluster occupation "
            "(need n_frozen < n_emb_occ < n_emb).");

    // (device placement) Run the WHOLE cluster chain (CIS-NTO/CCSD/IP/EA/STEOM) on a
    // chosen or freest GPU. At large n_emb the cluster CCSD device working set
    // (O(nocc²·nvir²) ≈ 90 GB at cc-pVDZ n_emb=427) plus the RI 3c resident on device 0
    // exceed one GPU (~1 GB over, log194). device 0 keeps the RI 3c; a free peer runs
    // the solve. Switch device HERE, before C/ε alloc, so the DeviceHostMatrix views —
    // and hence build_B_mo (distributed RI replicates the AO B per-GPU) + every operator
    // — are born on the target device: no cross-device reads. GPUHandle::reset() rebinds
    // the thread_local cuBLAS/cuSOLVER to the target (CRITICAL — cross-device DGEMM dies
    // otherwise). Env GANSU_DMET_STEOM_CLUSTER_GPU=<N>|auto; unset ⇒ current device
    // (byte-identical). Best paired with share-barH OFF (single-device, no choreography).
    int cl_dev_saved = 0; bool cl_dev_active = false;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        if (const char* e = std::getenv("GANSU_DMET_STEOM_CLUSTER_GPU")) {
            cudaGetDevice(&cl_dev_saved);
            int n_dev = 0; cudaGetDeviceCount(&n_dev);
            int target = -1;
            if (e[0] >= '0' && e[0] <= '9') target = std::atoi(e);       // explicit GPU id
            else {                                                        // "auto" → freest
                size_t bf = 0;
                for (int d = 0; d < n_dev; ++d) { cudaSetDevice(d); size_t fb = 0, tb = 0;
                    if (cudaMemGetInfo(&fb, &tb) == cudaSuccess && fb > bf) { bf = fb; target = d; } }
                cudaSetDevice(cl_dev_saved);
            }
            if (target >= 0 && target < n_dev && target != cl_dev_saved) {
                cudaSetDevice(target);
                gpu::GPUHandle::reset();   // rebind cuBLAS/cuSOLVER to the target device
                cl_dev_active = true;
                std::cout << "  [DMET-STEOM] cluster solve on GPU " << target
                          << " (device " << cl_dev_saved << " keeps RI 3c; frees the "
                          << "solve from single-device OOM)." << std::endl;
            }
        }
    }
#endif

    // Own the cluster C / ε in DeviceHostMatrix views (the chain calls
    // .device_ptr()/.toHost()/.host_ptr() on them). Device→device copy from the
    // caller's buffers; the views free on scope exit, after the chain returns.
    DeviceHostMatrix<real_t> C(static_cast<size_t>(nao), static_cast<size_t>(n_emb));
    DeviceHostMemory<real_t> eps(static_cast<size_t>(n_emb));
    cudaMemcpy(C.device_ptr(),  d_C_can, (size_t)nao * n_emb * sizeof(real_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(eps.device_ptr(), d_eps,  (size_t)n_emb        * sizeof(real_t), cudaMemcpyDeviceToDevice);

    EOMChainContext ctx;
    ctx.C        = &C;
    ctx.eps      = &eps;
    ctx.nmo      = n_emb;
    ctx.n_elec   = 2 * n_emb_occ;
    ctx.n_frozen = n_frozen;
    ctx.nao_ao   = nao;   // == n_emb on the Phase-0 reduction test (square cluster C)
    // Inc3: storage-free cluster integrals (opt-in). When set, the chain pulls
    // MO-ERI blocks from the cluster B (build_B_mo) instead of the dense n_emb⁴
    // tensor — the caller (dmet.cu) correspondingly passes d_eri_mo == nullptr.
    ctx.prefer_ri_block = (std::getenv("GANSU_DMET_STEOM_RI_BLOCK") != nullptr);
    ctx.level_shift     = level_shift;   // (B / denominator-only) 0 ⇒ legacy direct form

    // (DMET×DLPNO P0-3 probe, env GANSU_DMET_STEOM_LOC_TEST=1 — diagnostic only,
    // results printed and discarded). Rectangular-C make-or-break check:
    // Pipek-Mezey-localize the cluster's correlated occupied orbitals IN THE
    // FULL AO BASIS. The cluster MOs are genuine LCAO over the real AOs, so
    // PM's per-atom Mulliken populations stay well defined even for
    // non-atom-centered bath orbitals; the localizer API is shape-agnostic
    // (C_occ [nao_ao × nocc_c] + full-AO S + atom ranges). This verifies the
    // occupied half of the DLPNO ground pipeline on an embedding (rectangular)
    // C without touching dlpno_mp2.cu.
    if (const char* lt = std::getenv("GANSU_DMET_STEOM_LOC_TEST")) {
        if (lt[0] == '1') {
            C.toHost();
            cfg.get_overlap_matrix().toHost();
            const real_t* h_Cc = C.host_ptr();
            const real_t* h_S  = cfg.get_overlap_matrix().host_ptr();
            const int nocc_c = n_emb_occ - n_frozen;   // correlated cluster occ
            std::vector<real_t> C_occ_c((size_t)nao * nocc_c);
            for (int mu = 0; mu < nao; ++mu)
                for (int i = 0; i < nocc_c; ++i)
                    C_occ_c[(size_t)mu * nocc_c + i] =
                        h_Cc[(size_t)mu * n_emb + (n_frozen + i)];
            std::vector<std::pair<int,int>> atom_ranges;
            for (const auto& r : cfg.get_atom_to_basis_range())
                atom_ranges.emplace_back((int)r.start_index, (int)r.end_index);
            auto loc = localize_occupied("pm", C_occ_c.data(), h_S,
                                         /*Dx*/nullptr, /*Dy*/nullptr, /*Dz*/nullptr,
                                         nao, nocc_c, atom_ranges,
                                         /*max_sweep=*/200, /*conv_tol=*/1e-8,
                                         /*verbose=*/1);
            std::cout << "  [DMET-STEOM loc-test] PM on cluster occ (C "
                      << nao << "x" << n_emb << (nao == n_emb ? " square" : " RECTANGULAR")
                      << ", nocc_c=" << nocc_c << "): L "
                      << std::fixed << std::setprecision(6) << loc.functional_initial
                      << " -> " << loc.functional_final
                      << "  sweeps=" << loc.n_sweeps
                      << "  converged=" << (loc.converged ? "yes" : "NO") << std::endl;
        }
    }

    // (DMET×DLPNO Phase 1a/1b) Cluster-space DLPNO machinery on the embedding
    // orbital space (rectangular C):
    //   - GANSU_DMET_STEOM_DLPNO_LMP2_TEST=1 : Phase 1a probe — run cluster
    //     DLPNO-LMP2, print E(PAO)/E(PNO), discard. Gates: square reduction ==
    //     plain [DLPNO-MP2] to F_eff accuracy (~1 μHa = SCF residual);
    //     rectangular + FULL_DOMAIN + TCutPNO=0 == cluster canonical MP2
    //     (validated Δ1.5e-10, acetone {0,1}).
    //   - GANSU_DMET_STEOM_DLPNO=2 : Phase 1b — solve the cluster ground with
    //     DLPNO-CCSD (cluster space) + bt-polish (canonical cluster CCSD
    //     warm-start), stow the polished BT set on the RHF and point the ctx at
    //     it (the P0-1-validated wiring). Mode 1 (full-molecule DLPNO ground,
    //     square-C reduction anchor) is handled by compute_dmet_steom_ccsd.
    const char* lmp2_test_e = std::getenv("GANSU_DMET_STEOM_DLPNO_LMP2_TEST");
    const char* dl_mode_e   = std::getenv("GANSU_DMET_STEOM_DLPNO");
    const bool  want_lmp2_test  = lmp2_test_e && lmp2_test_e[0] == '1';
    const bool  want_cluster_dl = dl_mode_e && dl_mode_e[0] == '2';
    if (want_lmp2_test || want_cluster_dl) {
        // Cap OpenMP threads for the whole cluster-DLPNO section — the plain
        // entry points (compute_dlpno_ccsd/_t) apply this guard, but this
        // direct DLPNOCCSD construction bypassed them: on a >128-core host
        // (s177 nproc=128) the per-pair OpenBLAS calls then exceed the
        // library's 128 per-caller-thread buffer limit and abort (the known
        // dlpno_cpu_threads crash; previously masked by OMP_NUM_THREADS=64 in
        // the run recipes).
        OmpThreadCapGuard dl_omp_cap(cfg.get_dlpno_cpu_threads());
#ifndef GANSU_CPU_ONLY
        // The DLPNO machinery below (per-pair build_mo_eri fan-out,
        // replicate_B_to_all_gpus, distributed polish) leaves the CUDA current
        // device wherever its last multi-GPU loop ended (typically 0). Without a
        // restore, the whole downstream chain (CIS/IP/EA/STEOM) silently runs on
        // GPU 0 instead of the GANSU_DMET_STEOM_CLUSTER_GPU target — run9 audit:
        // GPU0 peak 113 GB (vs 15 GB canonical) and the EA Davidson OOM'd there
        // while GPU1 sat at 124.9 GB free. Save/restore + handle rebind.
        int dl_dev_saved = 0;
        if (gpu::gpu_available()) cudaGetDevice(&dl_dev_saved);
#endif
        C.toHost(); eps.toHost();
        cfg.get_overlap_matrix().toHost();
        const real_t* h_Cc  = C.host_ptr();
        const real_t* h_S   = cfg.get_overlap_matrix().host_ptr();
        // UN-SHIFT the cluster ε for the DLPNO ground: the ctx eps carry the
        // DMET stability level shift (+s on virtuals). Shifted denominators
        // systematically shrink the pair energies AND misguide the PNO
        // selection — run8 (dox, s=0.1553) measured E_corr(DLPNO) −2.455 vs
        // true −2.746 (−291 mHa), so the warm start bought nothing (polish
        // 1706 s ≈ cold 1777 s). The DLPNO ground must see the TRUE spectrum
        // (same convention as the "ε un-shifted (−s)" EOM operators); the
        // bt-polish below keeps the SHIFTED denominators + denominator-only
        // correction exactly like the canonical cluster CCSD.
        std::vector<real_t> eps_c(eps.host_ptr(), eps.host_ptr() + n_emb);
        if (level_shift != 0.0)
            for (int p1 = n_emb_occ; p1 < n_emb; ++p1) eps_c[p1] -= level_shift;
        const real_t* h_ec = eps_c.data();
        // F_eff = S·C·diag(eps)·C^T·S  (AO-covariant embedding Fock; == the
        // molecular F in the square limit since F·C = S·C·eps).
        std::vector<real_t> SC((size_t)nao * n_emb, 0.0);
        #pragma omp parallel for
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu) {
                const real_t s = h_S[(size_t)mu * nao + nu];
                if (s == 0.0) continue;
                for (int p = 0; p < n_emb; ++p)
                    SC[(size_t)mu * n_emb + p] += s * h_Cc[(size_t)nu * n_emb + p];
            }
        std::vector<real_t> F_eff((size_t)nao * nao, 0.0);
        #pragma omp parallel for
        for (int mu = 0; mu < nao; ++mu)
            for (int nu = 0; nu < nao; ++nu) {
                real_t v = 0.0;
                for (int p = 0; p < n_emb; ++p)
                    v += SC[(size_t)mu * n_emb + p] * h_ec[p]
                       * SC[(size_t)nu * n_emb + p];
                F_eff[(size_t)mu * nao + nu] = v;
            }
        DLPNOClusterSpace cs;
        cs.h_C = h_Cc; cs.h_eps = h_ec; cs.h_F_eff = F_eff.data();
        cs.nmo = n_emb; cs.nocc = n_emb_occ - n_frozen; cs.nfrozen = n_frozen;
        DLPNOParams p = resolve_dlpno_params(
            cfg.get_dlpno_preset(), cfg.get_dlpno_localizer(),
            cfg.get_dlpno_t_cut_pno(), cfg.get_dlpno_t_cut_do(),
            cfg.get_dlpno_t_cut_pairs(), cfg.get_dlpno_t_cut_mkn(),
            cfg.get_dlpno_t_cut_triples(), cfg.get_dlpno_t_cut_tno(),
            cfg.get_dlpno_pair_distance_cutoff(), cfg.get_dlpno_max_iter(),
            cfg.get_dlpno_diis_size(), cfg.get_dlpno_localizer_max_sweep(),
            cfg.get_dlpno_localizer_conv(), cfg.get_dlpno_lmp2_max_iter(),
            cfg.get_dlpno_lmp2_conv(), cfg.get_dlpno_sc_pno_iter(),
            cfg.get_dlpno_pno_os_only(), cfg.get_dlpno_verbose());
        if (want_lmp2_test) {
            auto lm = solve_dlpno_lmp2(cfg, eri_method, p, &cs);
            std::cout << "  [DMET-STEOM lmp2-test] cluster DLPNO-LMP2 (C "
                      << nao << "x" << n_emb << (nao == n_emb ? " square" : " RECTANGULAR")
                      << ", nocc_c=" << cs.nocc << ", nfrozen_c=" << cs.nfrozen
                      << "): E(PAO) = " << std::fixed << std::setprecision(10)
                      << lm.E_pao_total << "  E(PNO/LMP2) = " << lm.E_pno_total
                      << " Ha" << std::endl;
        }
        if (want_cluster_dl) {
            std::cout << "---- DMET-STEOM cluster-DLPNO ground (Phase 1b): "
                         "DLPNO-CCSD on the cluster space (C " << nao << "x" << n_emb
                      << (nao == n_emb ? " square" : " RECTANGULAR") << ") ----" << std::endl;
            cfg.set_collect_dlpno_bt(true);
            real_t E_dl = 0.0;
            {
                DLPNOCCSD drv(cfg, eri_method, p, &cs);
                E_dl = drv.compute_energy();
            }
            cfg.set_collect_dlpno_bt(false);
            if (!cfg.use_dlpno_amplitudes())
                throw std::runtime_error("DMET-STEOM (cluster-DLPNO): cluster "
                    "DLPNO-CCSD did not produce back-transformed amplitudes.");
            std::cout << "  [cluster-DLPNO] E_corr(DLPNO) = " << std::fixed
                      << std::setprecision(10) << E_dl << " Ha" << std::endl;
            real_t E_final = E_dl;
            // Cluster bt-polish: warm-start the CANONICAL cluster CCSD from the
            // cluster BT amplitudes (same recipe as the plain chain's
            // dlpno_bt_polish_stage, but in the embedding dimensions; the
            // B-native RI path mirrors the ctx chain's fresh-solve branch).
            const char* pe = std::getenv("GANSU_DLPNO_BT_POLISH");
            const bool pol_on = (pe == nullptr) || (pe[0] != '0');
            // (DMET×DLPNO Phase A) cluster-(T) validation hook:
            //   GANSU_DMET_STEOM_CLUSTER_T = canonical | dlpno | both
            // "canonical" evaluates the (T) correction inside the polish's
            // canonical cluster CCSD (converged amplitudes; run with the polish
            // cap OFF for a clean reference). "dlpno" runs the cluster-space
            // DLPNO-CCSD(T) (compute_dlpno_ccsd_t_impl(&cs); re-solves the
            // ground internally — validation-grade). Results are printed only;
            // the STEOM chain is unaffected.
            const char* tket = std::getenv("GANSU_DMET_STEOM_CLUSTER_T");
            const bool want_T_can = tket && (tket[0] == 'c' || tket[0] == 'b');
            if (pol_on && gpu::gpu_available()) {
                Timer pol_t;
                BTAmplitudes bt = cfg.get_dlpno_bt_amplitudes();   // copy; polished below
                const int cap = pe ? std::atoi(pe) : 0;
                std::cout << "---- DMET-STEOM cluster bt-polish: canonical cluster CCSD "
                             "warm-started from cluster-DLPNO amplitudes ----" << std::endl;
                ccsd_set_initial_guess(bt.T1.data(), bt.T2.data(),
                                       cap > 1 ? cap : -1, /*conv_override=*/1e-7);
                real_t* d_t1p = nullptr;
                real_t* d_t2p = nullptr;
                real_t E_T_can = 0.0;
                const ERI_RI* pol_ri = dynamic_cast<const ERI_RI*>(&eri_method);
                const real_t E_pol = ccsd_spatial_orbital(
                    /*d_eri_ao=*/nullptr, C.device_ptr(), eps.device_ptr(),
                    n_emb, n_emb_occ,
                    /*computing_ccsd_t=*/want_T_can,
                    /*ccsd_t_energy=*/want_T_can ? &E_T_can : nullptr,
                    &d_t1p, &d_t2p, /*d_eri_mo_precomputed=*/nullptr,
                    n_frozen, /*h_fov_active=*/nullptr,
                    /*eri_ri=*/pol_ri, /*level_shift=*/level_shift);
                if (want_T_can)
                    std::cout << "  [cluster (T) canonical] E(CCSD) = " << std::fixed
                              << std::setprecision(10) << E_pol << "  E((T)) = "
                              << E_T_can << "  E(total) = " << E_pol + E_T_can
                              << " Ha" << std::endl;
                cudaMemcpy(bt.T1.data(), d_t1p, bt.T1.size() * sizeof(real_t),
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(bt.T2.data(), d_t2p, bt.T2.size() * sizeof(real_t),
                           cudaMemcpyDeviceToHost);
                tracked_cudaFree(d_t1p);
                tracked_cudaFree(d_t2p);
                cfg.set_dlpno_bt_amplitudes(std::move(bt));
                E_final = E_pol;
                std::cout << "  [cluster bt-polish] E_corr = " << std::fixed
                          << std::setprecision(10) << E_pol << " Ha  (DLPNO was "
                          << E_dl << ", dE = " << std::scientific << std::setprecision(2)
                          << (E_pol - E_dl) << ";  " << std::fixed << std::setprecision(1)
                          << pol_t.elapsed_seconds() << " s)" << std::endl;
            } else if (pol_on) {
                std::cout << "  [cluster bt-polish] skipped (no GPU)." << std::endl;
            }
            ctx.use_dlpno_amplitudes = true;
            ctx.dlpno_bt = &cfg.get_dlpno_bt_amplitudes();
            ctx.dlpno_E  = E_final;
            std::cout << "  [DMET-STEOM] cluster chain consumes cluster-DLPNO "
                         "(bt-polished) amplitudes (GANSU_DMET_STEOM_DLPNO=2)." << std::endl;
            // (DMET×DLPNO Phase A) cluster DLPNO-CCSD(T) validation hook — see
            // the GANSU_DMET_STEOM_CLUSTER_T note above. Prints E(CCSD)/E((T)).
            if (tket && (tket[0] == 'd' || tket[0] == 'b')) {
                if (auto* ri_rhf = dynamic_cast<ERI_RI_RHF*>(&eri_method)) {
                    std::cout << "---- DMET-STEOM cluster DLPNO-(T) "
                                 "(validation hook, ground re-solved) ----" << std::endl;
                    const real_t e_tot = ri_rhf->compute_dlpno_ccsd_t_impl(&cs);
                    std::cout << "  [cluster DLPNO-(T)] E(CCSD)+E((T)) = " << std::fixed
                              << std::setprecision(10) << e_tot << " Ha" << std::endl;
                } else {
                    std::cout << "  [cluster DLPNO-(T)] skipped (non-RI ERI backend)."
                              << std::endl;
                }
            }
        }
#ifndef GANSU_CPU_ONLY
        // Release the DLPNO fan-out's AO-B replicas (naux·nao² per GPU ≈ 12.2 GB
        // ×4 at Doxorubicin) right here: the ground + polish are done with them,
        // and leaving them resident collided with the checkpoint bar-H restore
        // (run11: 1.24 GB alloc OOM at global 211 GB right after ckpt load,
        // which happens BEFORE the EA-stage relief). The chain's build_B_mo
        // lazily re-replicates on demand. Same opt-out knob as the other sites.
        if (gpu::gpu_available() && !std::getenv("GANSU_STEOM_KEEP_B_REPLICA")) {
            if (ERI_RI* dl_ri = dynamic_cast<ERI_RI*>(&eri_method)) {
                const size_t before = GlobalGpuMemoryTracker::get_current();
                dl_ri->release_bmo_ao_replica();
                const size_t after = GlobalGpuMemoryTracker::get_current();
                if (before > after)
                    std::cout << "  [DMET-STEOM] released RI AO-B replicas after the "
                                 "cluster-DLPNO ground: "
                              << CudaMemoryManager<real_t>::format_bytes(before - after)
                              << " reclaimed (global)." << std::endl;
            }
        }
        // Re-pin the cluster device for the downstream chain (see note above;
        // release_bmo_ao_replica may also reset the current device).
        if (gpu::gpu_available()) {
            int dl_dev_now = 0; cudaGetDevice(&dl_dev_now);
            if (dl_dev_now != dl_dev_saved) {
                cudaSetDevice(dl_dev_saved);
                gpu::GPUHandle::reset();
                std::cout << "  [DMET-STEOM] restored cluster device " << dl_dev_saved
                          << " after the DLPNO ground stage (was left on GPU "
                          << dl_dev_now << ")." << std::endl;
            }
        }
#endif
    }

    // (DMET×DLPNO P0) Cluster chain consumes DLPNO bt-polished amplitudes: the
    // driver (compute_dmet_steom_ccsd) already ran the DLPNO ground + polish and
    // stowed the BT set on the RHF. Point the ctx at it — the IP/EA/STEOM stages
    // then skip the cluster canonical-CCSD re-solve (use_dlpno_amplitudes branch).
    // Square-C reduction: the RHF's set IS the cluster's; a rectangular cluster
    // stows a cluster-space BT set here instead (Phase 1+). Native/projected EOM
    // stay off (per-pair PNO state is square-C bound). Default OFF = canonical.
    if (const char* dl = std::getenv("GANSU_DMET_STEOM_DLPNO")) {
        if (dl[0] == '1' && cfg.use_dlpno_amplitudes()) {
            ctx.use_dlpno_amplitudes = true;
            ctx.dlpno_bt = &cfg.get_dlpno_bt_amplitudes();
            ctx.dlpno_E  = cfg.get_post_hf_energy();
            std::cout << "  [DMET-STEOM] cluster chain consumes DLPNO bt-polished "
                         "amplitudes (GANSU_DMET_STEOM_DLPNO=1)." << std::endl;
        }
    }

    compute_steom_ccsd_impl(cfg, eri_method, /*d_eri_ao=*/nullptr, n_states,
                            /*d_eri_mo_precomputed=*/d_eri_mo, &ctx);

#ifndef GANSU_CPU_ONLY
    // (device placement) restore the caller's device + rebind handles. C/ε (on the
    // target device) free by pointer in their destructors regardless of current device.
    if (cl_dev_active) { cudaSetDevice(cl_dev_saved); gpu::GPUHandle::reset(); }
#endif

    // (solve-once) release the cached cluster CCSD ground amplitudes.
    if (ctx.cc_t1) tracked_cudaFree(ctx.cc_t1);
    if (ctx.cc_t2) tracked_cudaFree(ctx.cc_t2);

    // Carry the cluster CCSD ground correlation energy out so the caller can report
    // it as the post-HF correction (was lost with ctx ⇒ summary showed 0).
    ctx.steom_result.ground_corr_energy = ctx.post_hf_energy;

    return std::move(ctx.steom_result);
}

// ---- bt-polish (GANSU_DLPNO_BT_POLISH: DEFAULT ON since 2026-07-09; =0 to
// disable, =N to cap iterations) ----
// Warm-start a canonical CCSD from the back-transformed DLPNO amplitudes to
// erase the PNO-truncation error before the EOM chain (the DLPNO-vs-canonical
// STEOM gap is T2-truncation borne: naphthalene root0 5.08@normal /
// 4.90@very_tight vs canonical 4.68 — singles audit, 2026-07-08). The polish
// re-solve defaults to |dE| < 1e-7 (12 iters vs 46 at 1e-10; STEOM roots
// agree to 4 digits — 続46/47); GANSU_CCSD_CONV overrides.
// Few DIIS iterations from the warm start; the polished amplitudes REPLACE
// the stored BT set, so CIS-NTO / IP / EA / STEOM all consume them (native
// per-pair σ is defaulted OFF for consistency).
// Shared verbatim by compute_dlpno_steom_ccsd (plain path) and the DMET
// cluster-DLPNO mode (GANSU_DMET_STEOM_DLPNO — square-C reduction anchor).
static void dlpno_bt_polish_stage(RHF& rhf, const ERI_RI* eri_ri, real_t E_dlpno) {
    const char* pe = std::getenv("GANSU_DLPNO_BT_POLISH");
    const bool pol_on = (pe == nullptr) || (pe[0] != '0');
    if (pol_on && rhf.use_dlpno_amplitudes()
        && gpu::gpu_available()) {
        Timer polish_timer;
        BTAmplitudes bt = rhf.get_dlpno_bt_amplitudes();  // copy; polished below
        const int cap = pe ? std::atoi(pe) : 0;
        std::cout << "---- DLPNO-STEOM bt-polish: canonical CCSD warm-started "
                     "from DLPNO amplitudes ----" << std::endl;
        ccsd_set_initial_guess(bt.T1.data(), bt.T2.data(),
                               cap > 1 ? cap : -1, /*conv_override=*/1e-7);
        real_t* d_t1p = nullptr;
        real_t* d_t2p = nullptr;
        const real_t E_pol = ccsd_spatial_orbital(
            /*d_eri_ao=*/nullptr,
            rhf.get_coefficient_matrix().device_ptr(),
            rhf.get_orbital_energies().device_ptr(),
            rhf.get_num_basis(), rhf.get_num_electrons() / 2,
            /*computing_ccsd_t=*/false, /*ccsd_t_energy=*/nullptr,
            &d_t1p, &d_t2p, /*d_eri_mo_precomputed=*/nullptr,
            rhf.get_num_frozen_core(), /*h_fov_active=*/nullptr,
            /*eri_ri=*/eri_ri, /*level_shift=*/0.0);
        cudaMemcpy(bt.T1.data(), d_t1p, bt.T1.size() * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(bt.T2.data(), d_t2p, bt.T2.size() * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_t1p);
        tracked_cudaFree(d_t2p);
        rhf.set_dlpno_bt_amplitudes(std::move(bt));
        rhf.set_post_hf_energy(E_pol);
        std::cout << "  [bt-polish] E_corr = " << std::fixed
                  << std::setprecision(10) << E_pol << " Ha  (DLPNO was "
                  << E_dlpno << ", dE = " << std::scientific
                  << std::setprecision(2) << (E_pol - E_dlpno) << ";  "
                  << std::fixed << std::setprecision(1)
                  << polish_timer.elapsed_seconds() << " s)" << std::endl;
    } else if (pol_on) {
        std::cout << "  [bt-polish] skipped (no GPU or no BT amplitudes)."
                  << std::endl;
    }
}

// DMET-STEOM driver (RI path). Phase 1: build the chromophore fragment's
// ground-state Schmidt bath (single-shot, μ=0) → canonical cluster → cluster
// STEOM via steom_spatial_orbital (DMET::compute_steom). With no --dmet_fragments
// the cluster is the whole molecule → plain STEOM bit-exact (Phase 0 anchor).
void ERI_RI_RHF::compute_dmet_steom_ccsd(int n_states) {
    // (DMET×DLPNO P0, env GANSU_DMET_STEOM_DLPNO=1): solve the cluster with the
    // DLPNO-STEOM recipe instead of a fresh canonical cluster CCSD — run the
    // DLPNO-CCSD ground (+ bt-polish), exactly as compute_dlpno_steom_ccsd
    // stage 1, and stow the polished BT set on the RHF; steom_spatial_orbital
    // then points the cluster ctx at it (square-C reduction: with no
    // --dmet_fragments the cluster is the whole molecule, so this must
    // reproduce plain dlpno_steom_ccsd). A rectangular-C cluster will instead
    // run a cluster-space DLPNO ground here (Phase 1+). Default OFF =
    // canonical cluster (byte-identical).
    const char* dmet_dlpno = std::getenv("GANSU_DMET_STEOM_DLPNO");
    const bool  dl_on  = dmet_dlpno && dmet_dlpno[0] == '1';   // mode 1: full-molecule ground (P0 anchor)
    const bool  dl2_on = dmet_dlpno && dmet_dlpno[0] == '2';   // mode 2: cluster-space ground (Phase 1b, in steom_spatial_orbital)
    if (dl_on) {
        std::cout << "---- DMET-STEOM cluster-DLPNO mode: DLPNO-CCSD ground state "
                     "(back-transformed to canonical) ----" << std::endl;
        rhf_.set_collect_dlpno_bt(true);
        const real_t E_dlpno = compute_dlpno_ccsd();
        rhf_.set_collect_dlpno_bt(false);
        rhf_.set_post_hf_energy(E_dlpno);
        if (!rhf_.use_dlpno_amplitudes())
            throw std::runtime_error(
                "DMET-STEOM (cluster-DLPNO): DLPNO-CCSD did not produce "
                "back-transformed canonical amplitudes (collect path failed).");
        std::cout << "  DLPNO-CCSD correlation energy = " << std::fixed
                  << std::setprecision(10) << E_dlpno
                  << " Ha  (fed to the cluster STEOM chain)" << std::endl;
        dlpno_bt_polish_stage(rhf_, this, E_dlpno);
    }
    DMET dmet(rhf_, *this);
    STEOMResult r = dmet.compute_steom(*this, n_states);
    if (dl_on || dl2_on) rhf_.clear_dlpno_amplitudes();
    // Report the cluster CCSD ground correlation energy as the post-HF correction
    // (otherwise the final summary prints 0 even though the cluster CCSD ran).
    rhf_.set_post_hf_energy(r.ground_corr_energy);
    rhf_.append_excited_state_report(r.report);
    rhf_.set_steom_result(std::move(r));
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

    // Default the native DLPNO-IP/EA-EOM operator path ON (the validated
    // production-fast path: per-pair σ + share-barH + GPU rotations + f5b; ~1.3-2×
    // faster than the dense canonical σ). Set the env here — BEFORE the master
    // switch (below) and every sub-optimisation read in compute_steom_ccsd_impl /
    // the operator ctors — so they all see it consistently, exactly replicating an
    // explicit GANSU_DLPNO_NATIVE_EOM=1. Opt out with GANSU_DLPNO_NATIVE_EOM=0.
    // Scoped to this DLPNO-STEOM run: standalone IP/EA-EOM use a different entry
    // and are unaffected.
    if (std::getenv("GANSU_DLPNO_NATIVE_EOM") == nullptr) {
        // bt-polish (GANSU_DLPNO_BT_POLISH): the polish replaces the CANONICAL
        // back-transformed amplitudes only; the native per-pair σ would still
        // consume the UNPOLISHED per-pair DLPNO state. Default the native path
        // OFF under polish so IP/EA/STEOM consistently consume the polished
        // canonical amplitudes (explicit GANSU_DLPNO_NATIVE_EOM=1 overrides).
        const char* pol = std::getenv("GANSU_DLPNO_BT_POLISH");
        const bool polish_on = (pol == nullptr) || (pol[0] != '0');   // DEFAULT ON (2026-07-09)
        const char* nat_def = polish_on ? "0" : "1";
#ifdef _WIN32
        _putenv_s("GANSU_DLPNO_NATIVE_EOM", nat_def);
#else
        setenv("GANSU_DLPNO_NATIVE_EOM", nat_def, /*overwrite=*/0);
#endif
        std::cout << "  [DLPNO-STEOM] native EOM path defaulted "
                  << (polish_on ? "OFF (bt-polish active; canonical σ from "
                                  "polished amplitudes)"
                                : "ON (set GANSU_DLPNO_NATIVE_EOM=0 to disable)")
                  << std::endl;
    }

    // MPI Step 4 stage-1 broadcast (OPT-IN, GANSU_STEOM_STAGE1_BCAST=1): rank 0
    // alone computes the DLPNO-CCSD ground at full node threads while rank 1
    // waits, then ships the canonical ground state so rank 1 skips stage-1 —
    // removing the redundant stage-1 that otherwise runs at half threads on both
    // ranks and eats the IP||EA win. ⚠ INCOMPLETE: the NATIVE EOM operators also
    // need the per-pair DLPNO state (dlpno_res_, set_dlpno_res) which is NOT yet
    // transferred (DLPNOLMP2Result + phase24 serialization, TODO). Until then this
    // flag errors with the canonical-skip native path; default OFF keeps the
    // validated redundant-stage-1 behavior. See MPI_DESIGN.md.
    int mpi_rank = 0, mpi_size = 1;
    bool stage1_bcast = false;
#ifdef GANSU_MPI
    { int inited = 0; MPI_Initialized(&inited);
      if (inited) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); } }
    if (mpi_size > 1) {
        const char* es = std::getenv("GANSU_STEOM_MPI_SPLIT");
        const char* eb = std::getenv("GANSU_STEOM_STAGE1_BCAST");
        stage1_bcast = (es && es[0] == '1') && (eb && eb[0] == '1');
    }
    // With the broadcast on, ranks >= 2 take no part — skip the whole post-HF.
    if (stage1_bcast && mpi_rank >= 2) {
        std::cout << "  [STEOM MPI] rank " << mpi_rank
                  << " idle for DLPNO-STEOM (stage-1 broadcast: ranks 0,1 only)." << std::endl;
        return;
    }
#endif

    real_t E_dlpno = 0.0;
    if (!stage1_bcast || mpi_rank == 0) {
        // Stage 1 on rank 0 (or whenever the broadcast is off → all ranks compute).
        std::cout << "---- DLPNO-STEOM stage 1: DLPNO-CCSD ground state (back-transformed to canonical)"
                  << (stage1_bcast ? " [rank 0, full-thread; rank 1 waits]" : "") << " ----" << std::endl;
#if defined(GANSU_MPI) && defined(_OPENMP)
        const int omp_overlap = omp_get_max_threads();   // per-rank cap (launcher) for IP||EA overlap
        if (stage1_bcast) {
            const int full = omp_get_num_procs();        // rank 1 idle → rank 0 takes all cores
            omp_set_num_threads(full);
            if (openblas_set_num_threads) openblas_set_num_threads(full);
        }
#endif
        rhf_.set_collect_dlpno_bt(true);
        E_dlpno = compute_dlpno_ccsd();
        rhf_.set_collect_dlpno_bt(false);
        rhf_.set_post_hf_energy(E_dlpno);
        if (!rhf_.use_dlpno_amplitudes()) {
            throw std::runtime_error(
                "DLPNO-STEOM-CCSD: DLPNO-CCSD did not produce back-transformed canonical "
                "amplitudes (collect path failed). Check the DLPNO-CCSD run.");
        }
#if defined(GANSU_MPI) && defined(_OPENMP)
        if (stage1_bcast) {   // back to per-rank threads for the upcoming IP||EA overlap
            omp_set_num_threads(omp_overlap);
            if (openblas_set_num_threads) openblas_set_num_threads(omp_overlap);
        }
#endif
#ifdef GANSU_MPI
        if (stage1_bcast) {
            send_ground_state(rhf_, E_dlpno, 1);
            std::cout << "  [STEOM MPI] rank 0 sent DLPNO ground (C/eps/T1/T2) to rank 1 — "
                         "stage-1 not duplicated. ⚠ dlpno_res_ NOT sent (native EOM TODO)." << std::endl;
        }
#endif
    }
#ifdef GANSU_MPI
    else {   // rank 1: skip stage-1, adopt rank 0's ground state
        std::cout << "---- DLPNO-STEOM stage 1: SKIPPED on rank 1 (adopting rank 0's ground) ----" << std::endl;
        E_dlpno = recv_ground_state(rhf_, 0);
        rhf_.set_post_hf_energy(E_dlpno);
        std::cout << "  [STEOM MPI] rank 1 received DLPNO ground from rank 0 (stage-1 skipped)." << std::endl;
    }
#endif
    std::cout << "  DLPNO-CCSD correlation energy = " << std::fixed << std::setprecision(10)
              << E_dlpno << " Ha  (fed to the canonical STEOM machinery)" << std::endl;

    // ---- bt-polish (GANSU_DLPNO_BT_POLISH: DEFAULT ON since 2026-07-09; =0 to
    // disable, =N to cap iterations) ----
    // Warm-start a canonical CCSD from the back-transformed DLPNO amplitudes to
    // erase the PNO-truncation error before the EOM chain (the DLPNO-vs-canonical
    // STEOM gap is T2-truncation borne: naphthalene root0 5.08@normal /
    // 4.90@very_tight vs canonical 4.68 — singles audit, 2026-07-08). The polish
    // re-solve defaults to |dE| < 1e-7 (12 iters vs 46 at 1e-10; STEOM roots
    // agree to 4 digits — 続46/47); GANSU_CCSD_CONV overrides.
    // Few DIIS iterations from the warm start; the polished amplitudes REPLACE
    // the stored BT set, so CIS-NTO / IP / EA / STEOM all consume them (native
    // per-pair σ is defaulted OFF above for consistency).
    dlpno_bt_polish_stage(rhf_, this, E_dlpno);

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
