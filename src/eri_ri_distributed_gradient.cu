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

#ifdef GANSU_MULTI_GPU

#include <cuda.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <omp.h>

#include "rhf.hpp"
#include "eri.hpp"
#include "hf.hpp"
#include "gpu_manager.hpp"
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "gradients.hpp"
#include "grad_2c.hpp"
#include "grad_3c.hpp"
#include "int2e.hpp"  // comb_max(L)
#include <algorithm>

namespace gansu {

namespace {

// ----------------------------------------------------------------------------
// Replicate a single device buffer (currently on GPU 0) to all GPUs via
// cudaMemcpyPeer. Returns a vector of per-device pointers; index 0 is the
// caller-supplied source pointer (unchanged).
// ----------------------------------------------------------------------------
template <typename T>
std::vector<T*> replicate_to_all_gpus(const T* d_src_gpu0,
                                       size_t count,
                                       int num_gpus)
{
    std::vector<T*> result(num_gpus, nullptr);
    result[0] = const_cast<T*>(d_src_gpu0);
    for (int g = 1; g < num_gpus; g++) {
        cudaSetDevice(g);
        T* p = nullptr;
        gansu::tracked_cudaMalloc(&p, count * sizeof(T));
        result[g] = p;
    }
    for (int g = 1; g < num_gpus; g++) {
        cudaMemcpyPeer(result[g], g, d_src_gpu0, 0, count * sizeof(T));
    }
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
    return result;
}

template <typename T>
void free_per_gpu_replicas(std::vector<T*>& bufs)
{
    for (int g = 1; g < (int)bufs.size(); g++) {
        if (bufs[g]) {
            cudaSetDevice(g);
            gansu::tracked_cudaFree(bufs[g]);
        }
    }
    bufs.clear();
    cudaSetDevice(0);
}

// ----------------------------------------------------------------------------
// Restrict an aux-side ShellTypeInfo to the local *basis-function* range.
// Within each shell-type bucket, primitives are ordered with monotonically
// non-decreasing basis_idx (multiple atoms contributing one shell type each).
// We scan the bucket and pick the contiguous sub-range of primitives whose
// basis_idx falls within [basis_lo, basis_hi).
//
// Returns a ShellTypeInfo whose start_index/count still refer to PRIMITIVE
// shell indices (the units the kernel expects). The chosen sub-range is
// contiguous in primitive order even though the basis_idx may have jumps,
// because basis_idx is monotonic within a bucket.
// ----------------------------------------------------------------------------
inline gansu::ShellTypeInfo restrict_aux_shell_by_basis(
    const gansu::ShellTypeInfo& info,
    const gansu::PrimitiveShell* h_aux_prim,
    size_t basis_lo, size_t basis_hi)
{
    const size_t lo = info.start_index;
    const size_t hi = info.start_index + info.count;
    size_t first = hi;
    size_t last  = lo;   // exclusive
    for (size_t i = lo; i < hi; i++) {
        const size_t b = (size_t)h_aux_prim[i].basis_index;
        if (b >= basis_lo && b < basis_hi) {
            if (i < first) first = i;
            last = i + 1;
        }
    }
    gansu::ShellTypeInfo sub = info;
    if (first >= last) sub.count = 0;
    else { sub.start_index = first; sub.count = (int)(last - first); }
    return sub;
}

// ----------------------------------------------------------------------------
// Γ^(3)_{P_loc,μν} = γ[P_start + P_loc] · D_{μν} − ½ A_local[P_loc, μν]
// Mirrors k_assemble_gamma3 in eri_ri_gradient.cu, but reads γ with a P_start
// offset so the same kernel works on a local row-slab of Γ^(3).
// ----------------------------------------------------------------------------
__global__
void k_assemble_gamma3_local(real_t* d_gamma3_local,
                             const real_t* d_A_local,
                             const real_t* d_gamma_full,
                             const real_t* d_D,
                             const int naux_local,
                             const int nbas,
                             const size_t P_start)
{
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
    const size_t total = (size_t)naux_local * nbas2;
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    const size_t P_loc = tid / nbas2;
    const size_t idx   = tid % nbas2;

    d_gamma3_local[tid] = d_gamma_full[P_start + P_loc] * d_D[idx]
                        - 0.5 * d_A_local[tid];
}

// ----------------------------------------------------------------------------
// Per-device launch of the 3c2e gradient kernel for the [P_start, P_end)
// aux-primitive range. Uses local replicated device pointers.
//
// d_gamma3_local is a local row-slab of size (naux_local × nbas²). The kernel
// expects global P indices into a (naux × nbas²) buffer, so we pass
// d_gamma3_local - P_start * nbas² as the kernel's `g_gamma3` pointer (same
// trick used in the SCF distributed B-build with d_chunk).
// ----------------------------------------------------------------------------
void launch_3c2e_grad_local(
    double* d_grad_local,
    const real_t* d_gamma3_local,
    const std::vector<gansu::ShellTypeInfo>& shell_type_infos,
    const std::vector<gansu::ShellTypeInfo>& aux_shell_type_infos,
    const gansu::PrimitiveShell* d_pshell,
    const gansu::PrimitiveShell* d_pshell_aux,
    const real_t* d_cgto,
    const real_t* d_aux_cgto,
    int nbas, int naux,
    const real_t* d_boys_grid,
    const gansu::PrimitiveShell* h_aux_prim,
    size_t basis_P_start, size_t basis_P_end)
{
    const int n_shell_mu  = (int)shell_type_infos.size();
    const int n_shell_aux = (int)aux_shell_type_infos.size();
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;

    // Phantom pointer: kernel accesses Γ⁽³⁾[global_basis_P * nbas² + ...]
    // (P.basis_index, not primitive index). Slab is sized to local basis-
    // function rows starting at basis_P_start.
    const real_t* d_gamma3_phantom = d_gamma3_local - basis_P_start * nbas2;

    for (int s_mu = 0; s_mu < n_shell_mu; s_mu++) {
        for (int s_nu = 0; s_nu < n_shell_mu; s_nu++) {
            for (int s_P = 0; s_P < n_shell_aux; s_P++) {
                const auto& info_mu = shell_type_infos[s_mu];
                const auto& info_nu = shell_type_infos[s_nu];
                // Restrict to local BASIS-function range by basis_idx of each
                // primitive in this bucket.
                const auto info_aux_local =
                    restrict_aux_shell_by_basis(aux_shell_type_infos[s_P],
                                                 h_aux_prim,
                                                 basis_P_start, basis_P_end);
                const size_t n_threads = (size_t)info_mu.count *
                                         (size_t)info_nu.count *
                                         (size_t)info_aux_local.count;
                if (n_threads == 0) continue;

                const int threads_per_block = 64;
                const size_t blocks = (n_threads + threads_per_block - 1) / threads_per_block;
                gpu::compute_gradients_3c2e<<<(unsigned int)blocks, threads_per_block>>>(
                    d_grad_local, d_gamma3_phantom,
                    d_pshell, d_pshell_aux,
                    d_cgto, d_aux_cgto,
                    info_mu, info_nu, info_aux_local,
                    n_threads, nbas, naux,
                    d_boys_grid);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Per-device 2c2e gradient launch with no_pair_symmetry = true.
//
// d_gamma2_local is a local row-slab of size (naux_local × naux). Same pointer
// trick: pass d_gamma2_local - P_start * naux so kernel's global P index works.
// ----------------------------------------------------------------------------
void launch_2c2e_grad_local(
    double* d_grad_local,
    const real_t* d_gamma2_local,
    const std::vector<gansu::ShellTypeInfo>& aux_shell_type_infos,
    const gansu::PrimitiveShell* d_pshell_aux,
    const real_t* d_aux_cgto,
    int naux,
    const real_t* d_boys_grid,
    const gansu::PrimitiveShell* h_aux_prim,
    size_t basis_P_start, size_t basis_P_end)
{
    const int n_shell_aux = (int)aux_shell_type_infos.size();
    // Phantom pointer: kernel accesses Γ⁽²⁾[global_basis_P * naux + Q]; slab
    // is sized to local basis rows starting at basis_P_start.
    const real_t* d_gamma2_phantom = d_gamma2_local - basis_P_start * (size_t)naux;

    for (int s_P = 0; s_P < n_shell_aux; s_P++) {
        const auto info_P_local =
            restrict_aux_shell_by_basis(aux_shell_type_infos[s_P],
                                         h_aux_prim,
                                         basis_P_start, basis_P_end);
        if (info_P_local.count == 0) continue;
        for (int s_Q = 0; s_Q < n_shell_aux; s_Q++) {
            const auto& info_Q = aux_shell_type_infos[s_Q];
            const size_t n_threads = (size_t)info_P_local.count * (size_t)info_Q.count;
            if (n_threads == 0) continue;

            const int threads_per_block = 128;
            const size_t blocks = (n_threads + threads_per_block - 1) / threads_per_block;
            gpu::compute_gradients_2c2e<<<(unsigned int)blocks, threads_per_block>>>(
                d_grad_local, d_gamma2_phantom,
                d_pshell_aux, d_aux_cgto,
                info_P_local, info_Q,
                n_threads, naux,
                d_boys_grid,
                /*no_pair_symmetry=*/true);
        }
    }
}

} // anonymous namespace


// ============================================================================
// ERI_RI_Distributed_RHF::compute_ri_gradient — Phase 4 v4
//
// True per-GPU distributed RI gradient. v3 introduced row-block-local prep
// (Bbar, A, Γ^(3), Γ^(2)) on each GPU, eliminating the GPU-0 prep bottleneck
// and the (naux × nbas²) Γ^(3) broadcast that crippled v2.
//
// v4 adds two optimizations on top of v3:
//   • NCCL Broadcast for the full-B replication (in replicate_B_to_all_gpus
//     in eri_ri_distributed.cu) — replaces 64 serialized pairwise peer-
//     copies with 8 collective Broadcasts. 5.25× faster (2506 → 477 ms).
//   • Distributed 1-electron + nuclear gradient. W matrix replicated; each
//     GPU handles a round-robin subset of the (s0, s1) shell-type pair loop
//     and accumulates into its own d_grad_per_gpu[d]. The trailing NCCL
//     AllReduce already sums them.
//
// Intra-OMP `cudaStreamSynchronize(stream)` calls between sub-stages are
// LEFT IN even after diagnostic instrumentation was removed: empirically
// they are required for performance — without them the 3c2e/2c2e kernel
// launches see ~2× slower per-GPU times due to bad stream interaction
// across GPUs (probably resource contention without explicit hand-off).
//
// Validated on cholesterol (cc-pvdz / cc-pvdz-rifit, 8×H200):
//   1 GPU      14834 ms (baseline)
//   v2  8 GPU  13412 ms (1.11×) — but partition was broken, GPU 0 did all work
//   v3  8 GPU   7524 ms (1.97×) — correct multi-GPU; numerically validated
//   v4  8 GPU   3385 ms (4.38×) — NCCL Broadcast + parallel 1e grad
//
// Algorithm (after replicate_B_to_all_gpus()):
//
//   1) Replicate small SCF buffers (prim shells, cgto norms, atoms, Boys grid,
//      aux Schwarz bounds, D matrix) to every GPU.
//   2) Build L⁻¹ on GPU 0 (rebuild 2c2e + Cholesky + inverse), peer-copy to
//      every GPU. naux × naux is small (≲ 200 MB for naux ≈ 5000).
//   3) Per-GPU 1-electron and nuclear-repulsion gradient on GPU 0 only (cheap).
//   4) OpenMP parallel section, each GPU d operates only on its [P_start[d],
//      P_start[d] + naux_local[d]) aux primitive range:
//
//        γ_full       = L⁻ᵀ · (B · vec(D))          (replicated full naux vec)
//        Bbar_local   = (L⁻ᵀ)[P_loc, :] · B          (DGEMM, naux_local × nbas²)
//        X_local      = D · Bbar_local               (batched DGEMM)
//        A_local      = X_local · D                  (batched DGEMM, overwrites
//                                                     Bbar_local buffer)
//        Γ^(3)_local  = γ[P_loc] D − ½ A_local       (custom kernel)
//        T_local      = A_local · B^T                (DGEMM, naux_local × naux)
//        Γ^(2)_local  = ¼ T_local · L⁻¹ − ½ γ_local γᵀ
//
//        3c2e/2c2e gradient kernels launched with phantom-pointer trick so the
//        kernels (which use global P indices) write into the local row-slab.
//
//   5) NCCL AllReduce the per-device gradient buffers (n3 doubles — tiny).
//   6) Copy reduced gradient from GPU 0 to host, free per-GPU scratch.
//
// Memory per GPU at peak (cholesterol scale, naux ≈ 5000, nbas ≈ 650, 8 GPU):
//   full B (replicated)       ≈ 17 GB   ← shared SCF cost
//   L⁻¹ replica               ≈ 200 MB
//   Bbar_local / A_local pair ≈ 4.2 GB  (one buffer reused for both)
//   X scratch                 ≈ 2.1 GB
//   Γ^(3)_local               ≈ 2.1 GB
//   Γ^(2)_local + T scratch   ≈ 50 MB
//   small SCF replicas        ≈ MB
//   total                     ≈ 26 GB / GPU  (v2 GPU 0 alone needed ≈ 70 GB)
//
// Fallback: replicate_B_to_all_gpus() failing → gather on GPU 0 and run v1.
// ============================================================================
std::vector<double> ERI_RI_Distributed_RHF::compute_ri_gradient(
    const real_t* d_density_matrix,
    const real_t* d_coefficient_matrix,
    const real_t* d_orbital_energies,
    const int num_electron)
{
    const int nbas = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
    const int num_atoms = (int)hf_.get_atoms().size();
    const int n3 = 3 * num_atoms;

    auto& mgr = MultiGpuManager::instance();

    if (hf_.get_verbose()) {
        std::cout << "[RI-Distributed Gradient v4] num_gpus=" << num_gpus_
                  << " naux=" << naux << std::endl;
    }

    const bool replicated = replicate_B_to_all_gpus();

    // ========================= v1 fallback =========================
    if (!replicated) {
        if (hf_.get_verbose()) {
            std::cout << "[RI-Distributed Gradient v3] B replication declined; "
                         "falling back to gather-on-GPU0 (v1)." << std::endl;
        }
        cudaSetDevice(0);
        real_t* d_B_full = nullptr;
        tracked_cudaMalloc(&d_B_full, (size_t)naux * nbas2 * sizeof(real_t));
        std::vector<size_t> offsets(num_gpus_, 0);
        for (int g = 1; g < num_gpus_; g++)
            offsets[g] = offsets[g - 1] + (size_t)naux_local_[g - 1] * nbas2;
        for (int src = 0; src < num_gpus_; src++) {
            const size_t bytes_src = (size_t)naux_local_[src] * nbas2 * sizeof(real_t);
            if (bytes_src == 0) continue;
            if (src == 0) {
                cudaMemcpy(d_B_full + offsets[src], d_B_local_[src],
                           bytes_src, cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpyPeer(d_B_full + offsets[src], 0,
                               d_B_local_[src], src, bytes_src);
            }
        }
        cudaDeviceSynchronize();

        std::vector<double> grad = compute_ri_gradient_impl(
            d_density_matrix, d_coefficient_matrix, d_orbital_energies, num_electron,
            d_B_full,
            /*P_local_start=*/0,
            /*P_local_end=*/(size_t)naux,
            /*include_one_electron=*/true);

        tracked_cudaFree(d_B_full);
        return grad;
    }

    // ====================== v3: per-GPU distributed prep =====================

    // ---- 1) Replicate small SCF buffers ----
    const auto& primitive_shells = hf_.get_primitive_shells();
    const auto& cgto_norms       = hf_.get_cgto_normalization_factors();
    const auto& atoms_dh         = hf_.get_atoms();
    const auto& boys_grid        = hf_.get_boys_grid();
    const auto& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<gansu::ShellTypeInfo>& aux_shell_type_infos =
        auxiliary_shell_type_infos_;

    const size_t n_prim_ao  = primitive_shells.size();
    const size_t n_cgto     = (size_t)nbas;
    const size_t n_atoms_dh = atoms_dh.size();
    const size_t n_boys     = boys_grid.size();
    const size_t n_prim_aux = auxiliary_primitive_shells_.size();
    const size_t n_cgto_aux = (size_t)naux;

    auto d_prim_per_gpu     = replicate_to_all_gpus<gansu::PrimitiveShell>(
        primitive_shells.device_ptr(), n_prim_ao, num_gpus_);
    auto d_cgto_per_gpu     = replicate_to_all_gpus<real_t>(
        cgto_norms.device_ptr(), n_cgto, num_gpus_);
    auto d_atoms_per_gpu    = replicate_to_all_gpus<gansu::Atom>(
        atoms_dh.device_ptr(), n_atoms_dh, num_gpus_);
    auto d_boys_per_gpu     = replicate_to_all_gpus<real_t>(
        boys_grid.device_ptr(), n_boys, num_gpus_);
    auto d_aux_prim_per_gpu = replicate_to_all_gpus<gansu::PrimitiveShell>(
        auxiliary_primitive_shells_.device_ptr(), n_prim_aux, num_gpus_);
    auto d_aux_cgto_per_gpu = replicate_to_all_gpus<real_t>(
        auxiliary_cgto_normalization_factors_.device_ptr(), n_cgto_aux, num_gpus_);
    // NOTE: auxiliary_schwarz_upper_bound_factors is sized by *primitive* shell
    // count (one entry per aux primitive shell), NOT by basis-function count.
    // Passing n_cgto_aux (naux) reads past the end → cudaMemcpyPeer returns
    // cudaErrorInvalidValue and the AllReduce later surfaces as "unhandled
    // cuda error". Use n_prim_aux.
    auto d_aux_schwarz_per_gpu = replicate_to_all_gpus<real_t>(
        auxiliary_schwarz_upper_bound_factors.device_ptr(), n_prim_aux, num_gpus_);
    auto d_D_per_gpu        = replicate_to_all_gpus<real_t>(
        d_density_matrix, nbas2, num_gpus_);


    // ---- 2) Build L⁻¹ on GPU 0, peer-copy to every GPU ----
    //
    // Stored mode releases d_cached_L_inv_ after distributed_build_B(); only
    // OnTheFly mode keeps it across SCF iterations. So we unconditionally
    // rebuild here (single 2c2e + Cholesky + inverse on GPU 0). Cost: ~150 ms
    // for naux ≲ 6000; followed by peer-copy at NVLink rate.
    cudaSetDevice(0);
    std::vector<real_t*> d_L_inv_per_gpu(num_gpus_, nullptr);
    {
        real_t* d_L = nullptr;
        tracked_cudaMalloc(&d_L, (size_t)naux * naux * sizeof(real_t));
        cudaMemset(d_L, 0, (size_t)naux * naux * sizeof(real_t));
        gpu::computeTwoCenterERIs(
            aux_shell_type_infos,
            d_aux_prim_per_gpu[0],
            d_aux_cgto_per_gpu[0],
            d_L, naux,
            d_boys_per_gpu[0],
            d_aux_schwarz_per_gpu[0],
            hf_.get_schwarz_screening_threshold(),
            hf_.get_verbose());
        gpu::choleskyDecomposition(d_L, naux);

        tracked_cudaMalloc(&d_L_inv_per_gpu[0], (size_t)naux * naux * sizeof(real_t));
        gpu::computeInverseByDtrsm(d_L, d_L_inv_per_gpu[0], naux);
        tracked_cudaFree(d_L);

        for (int g = 1; g < num_gpus_; g++) {
            cudaSetDevice(g);
            tracked_cudaMalloc(&d_L_inv_per_gpu[g], (size_t)naux * naux * sizeof(real_t));
        }
        for (int g = 1; g < num_gpus_; g++) {
            cudaMemcpyPeer(d_L_inv_per_gpu[g], g, d_L_inv_per_gpu[0], 0,
                           (size_t)naux * naux * sizeof(real_t));
        }
        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
    }


    // ---- 3) Allocate per-device gradient buffers ----
    std::vector<double*> d_grad_per_gpu(num_gpus_, nullptr);
    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        tracked_cudaMalloc(&d_grad_per_gpu[g], n3 * sizeof(double));
        cudaMemset(d_grad_per_gpu[g], 0, n3 * sizeof(double));
    }

    // ---- 3a) Contracted-shell-aligned partition for gradient kernels ----
    //
    // The gradient kernels iterate PRIMITIVE shells (ShellTypeInfo::start_index
    // is in primitive units), but Γ⁽²⁾/Γ⁽³⁾ are indexed by BASIS FUNCTION
    // (P.basis_index). Multiple primitives can share a basis_index when they
    // come from the same contracted shell — splitting them across GPUs would
    // double-count gradient contributions and break the math.
    //
    // We therefore group primitives into "contracted shells" (runs of
    // consecutive primitives sharing the same basis_index), partition that
    // list of groups across GPUs, and derive each GPU's primitive range and
    // basis-function range from its contracted shells. Adjacent GPUs are
    // disjoint and exhaustively cover all primitives + all basis functions.
    const PrimitiveShell* h_aux_prim = auxiliary_primitive_shells_.host_ptr();
    if (h_aux_prim == nullptr) {
        throw std::runtime_error("[RI-Distributed Gradient v3] auxiliary_primitive_shells_.host_ptr() is null");
    }

    // Build per-contracted-shell list (one entry per unique basis_idx in the
    // primitive array). Primitives sharing a basis_idx are the contracted
    // primitives of one shell — they must NOT be split across GPUs.
    //
    // Note on ordering: in GANSU the primitive array is grouped by shell-type
    // (all S, then all P, ...), but basis functions are laid out per-atom
    // (all of atom-i's shells consecutively). Thus basis_idx is monotonic
    // *within* each shell-type bucket but NOT across the full primitive array
    // (jumps when crossing shell-type boundaries).
    //
    // To get a CONTIGUOUS local basis-function range per GPU we sort cshells
    // by basis_idx before partitioning. The kernel restrict then has to do a
    // basis_idx-based scan within each shell-type bucket (instead of a
    // primitive-index range), since the partition is no longer aligned with
    // primitive order.
    struct CShell { size_t basis_idx; int L; };
    std::vector<CShell> cshells;
    cshells.reserve(n_prim_aux);
    for (size_t i = 0; i < n_prim_aux; i++) {
        if (i == 0 || h_aux_prim[i].basis_index != h_aux_prim[i-1].basis_index) {
            cshells.push_back({h_aux_prim[i].basis_index, h_aux_prim[i].shell_type});
        }
    }
    std::sort(cshells.begin(), cshells.end(),
              [](const CShell& a, const CShell& b){ return a.basis_idx < b.basis_idx; });
    const size_t n_cshells = cshells.size();

    // Cshell-count balanced partition (revert from work-weighted). With
    // comb_max(L)^2 weighting, 3c2e imbalance dropped from 1.80× to 1.25× but
    // the avg (= effective parallel work) was 1791 vs 1813 ms — no net win
    // because max barely moved. The remaining bottleneck is per-GPU 3c2e
    // compute time (~1800 ms), not load imbalance.
    std::vector<size_t> basis_start(num_gpus_), basis_end(num_gpus_);
    std::vector<int>    nl_basis(num_gpus_, 0);
    for (int g = 0; g < num_gpus_; g++) {
        auto [cs, ce] = aux_partition(n_cshells, num_gpus_, g);
        if (cs >= ce) {
            basis_start[g] = naux;
            basis_end[g]   = naux;
            nl_basis[g]    = 0;
            continue;
        }
        basis_start[g] = cshells[cs].basis_idx;
        basis_end[g]   = cshells[ce-1].basis_idx + (size_t)gpu::comb_max(cshells[ce-1].L);
        nl_basis[g]    = (int)(basis_end[g] - basis_start[g]);
    }
    if (hf_.get_verbose()) {
        std::cout << "[RI-Distributed Gradient v3] n_prim_aux=" << n_prim_aux
                  << " naux=" << naux << " cshells=" << n_cshells
                  << ", per-GPU basis ranges:";
        for (int g = 0; g < num_gpus_; g++) {
            std::cout << " [" << g << "]:b[" << basis_start[g] << "," << basis_end[g]
                      << ") nl=" << nl_basis[g];
        }
        std::cout << std::endl;
    }

    // ---- 4) 1-electron + nuclear-repulsion gradient, distributed across GPUs ----
    // W matrix built on GPU 0 and replicated. Each GPU handles a subset of the
    // (s0, s1) shell-type pair loop via round-robin assignment. Nuclear-
    // repulsion gradient remains on GPU 0 only (cheap). Each GPU's
    // contributions accumulate into its own d_grad_per_gpu[d]; the final NCCL
    // AllReduce sums them.
    real_t* d_W_per_gpu[64] = {nullptr};  // small; num_gpus_ ≤ 64 in practice
    {
        cudaSetDevice(0);
        cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
        tracked_cudaMalloc(&d_W_per_gpu[0], nbas2 * sizeof(real_t));
        cudaMemset(d_W_per_gpu[0], 0, nbas2 * sizeof(real_t));
        gpu::compute_W(d_W_per_gpu[0], d_coefficient_matrix, d_orbital_energies, nbas, num_electron);
        cudaDeviceSynchronize();

        for (int g = 1; g < num_gpus_; g++) {
            cudaSetDevice(g);
            tracked_cudaMalloc(&d_W_per_gpu[g], nbas2 * sizeof(real_t));
        }
        for (int g = 1; g < num_gpus_; g++) {
            cudaMemcpyPeer(d_W_per_gpu[g], g, d_W_per_gpu[0], 0, nbas2 * sizeof(real_t));
        }
        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
    }

    #pragma omp parallel num_threads(num_gpus_)
    {
        const int d = omp_get_thread_num();
        cudaSetDevice(d);
        cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);

        const int shell_type_count = (int)shell_type_infos.size();
        const int threads_per_block = 128;
        int pair_idx = 0;
        for (int s0 = shell_type_count - 1; s0 >= 0; s0--) {
            for (int s1 = shell_type_count - 1; s1 >= s0; s1--) {
                if (pair_idx % num_gpus_ != d) { pair_idx++; continue; }
                pair_idx++;

                const auto info0 = shell_type_infos[s0];
                const auto info1 = shell_type_infos[s1];
                const size_t n_pairs = (s0 == s1)
                    ? (size_t)info0.count * (info0.count + 1) / 2
                    : (size_t)info0.count * info1.count;
                if (n_pairs == 0) continue;
                const size_t blocks = (n_pairs + threads_per_block - 1) / threads_per_block;

                gpu::compute_gradients_overlap<<<(unsigned int)blocks, threads_per_block>>>(
                    d_grad_per_gpu[d], d_W_per_gpu[d],
                    d_prim_per_gpu[d], d_cgto_per_gpu[d],
                    nbas, info0, info1, n_pairs);
                gpu::compute_gradients_kinetic<<<(unsigned int)blocks, threads_per_block>>>(
                    d_grad_per_gpu[d], d_D_per_gpu[d],   // use replicated D, NOT d_density_matrix (GPU 0 only)
                    d_prim_per_gpu[d], d_cgto_per_gpu[d],
                    nbas, info0, info1, n_pairs);
                gpu::compute_gradients_nuclear<<<(unsigned int)blocks, threads_per_block>>>(
                    d_grad_per_gpu[d], d_D_per_gpu[d],   // replicated D
                    d_prim_per_gpu[d], d_cgto_per_gpu[d],
                    d_atoms_per_gpu[d], num_atoms, nbas, info0, info1, n_pairs,
                    d_boys_per_gpu[d]);
            }
        }

        // Nuclear repulsion gradient only on GPU 0 (cheap, atom × atom).
        if (d == 0) {
            const size_t nr_threads = (size_t)num_atoms * (size_t)num_atoms;
            const size_t nr_blocks  = (nr_threads + threads_per_block - 1) / threads_per_block;
            gpu::compute_nuclear_repulsion_gradient_kernel<<<(unsigned int)nr_blocks, threads_per_block>>>(
                d_grad_per_gpu[0], d_atoms_per_gpu[0], num_atoms);
        }
        cudaDeviceSynchronize();
    }

    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        if (d_W_per_gpu[g]) tracked_cudaFree(d_W_per_gpu[g]);
    }
    cudaSetDevice(0);


    // ---- 5) OpenMP parallel: each GPU builds its local prep + 2e gradient ----
    //
    // The `cudaStreamSynchronize(stream)` calls between sub-stages below are
    // PERFORMANCE-CRITICAL, not just for diagnostics. Removing them causes
    // the 3c2e/2c2e kernels to take ~2× longer due to bad cross-GPU stream
    // interaction (resource contention, suspected). Keep them.
    #pragma omp parallel num_threads(num_gpus_)
    {
        const int d = omp_get_thread_num();
        cudaSetDevice(d);
        cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);

        cublasHandle_t handle = mgr.cublas(d);
        cudaStream_t stream = mgr.compute_stream(d);
        cublasSetStream(handle, stream);

        // Contiguous basis-function range — rows of Γ⁽²⁾/Γ⁽³⁾, A_local,
        // Bbar_local. Derived from cshell-sorted partition (see step 3a).
        const size_t P_start    = basis_start[d];   // basis-function row start
        const size_t P_end      = basis_end[d];     // basis-function row end (excl)
        const int    nl         = nl_basis[d];      // rows of Γ_local

        real_t* d_B    = d_B_full_per_gpu_[d];     // full B (naux × nbas²)
        real_t* d_Linv = d_L_inv_per_gpu[d];       // L⁻¹ (naux × naux)
        real_t* d_D    = d_D_per_gpu[d];           // D matrix (nbas × nbas)

        const double one = 1.0;
        const double zero = 0.0;

        // ----- γ_full = L⁻ᵀ · (B · vec(D)) -----
        //
        //   wB[P]    = Σ_{μν} B[P, μν] D[μν]  via DGEMV (OP=T over B_cm)
        //   γ[P]     = Σ_R L⁻ᵀ[P, R] wB[R]    = Σ_R L⁻¹[R, P] wB[R]
        //              via DGEMV (OP=N over L⁻¹_cm)
        real_t* d_wB    = nullptr;
        real_t* d_gamma = nullptr;
        tracked_cudaMalloc(&d_wB,    (size_t)naux * sizeof(real_t));
        tracked_cudaMalloc(&d_gamma, (size_t)naux * sizeof(real_t));
        cublasDgemv(handle, CUBLAS_OP_T,
                    (int)nbas2, naux,
                    &one, d_B, (int)nbas2,
                    d_D, 1,
                    &zero, d_wB, 1);
        cublasDgemv(handle, CUBLAS_OP_N,
                    naux, naux,
                    &one, d_Linv, naux,
                    d_wB, 1,
                    &zero, d_gamma, 1);
        tracked_cudaFree(d_wB);
        cudaStreamSynchronize(stream);

        // ----- Bbar_local = (L⁻ᵀ)[P_loc, :] · B_full -----
        //
        // Row-major math:  Bbar_local[P_loc, μν] = Σ_Q L⁻¹[Q, P_start + P_loc] · B[Q, μν]
        // Column-major DGEMM:
        //   C_cm = op(A_cm) · op(B_cm), shape (m, n)
        //   m = nbas², n = nl, k = naux
        //   A_cm = d_B (col-major view of row-major B[naux, nbas²]) — shape (nbas², naux), ld=nbas²
        //   B_cm = sub-block of d_Linv at row offset P_start (col-major view) — shape (nl, naux), ld=naux
        //   op_A = N, op_B = T
        //   pointer for B_cm sub-block at column-major (P_start, 0) is d_Linv + P_start
        real_t* d_Bbar_local = nullptr;
        tracked_cudaMalloc(&d_Bbar_local, (size_t)nl * nbas2 * sizeof(real_t));
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    (int)nbas2, nl, naux,
                    &one,
                    d_B, (int)nbas2,
                    d_Linv + P_start, naux,
                    &zero,
                    d_Bbar_local, (int)nbas2);
        cudaStreamSynchronize(stream);

        // ----- X_local = D · Bbar_local  (batched DGEMM) -----
        //
        // For each P_loc: X[P_loc] = D · Bbar[P_loc], shape (nbas, nbas).
        // Row-major batched: X[P_loc, μ, ν] = Σ_α D[μ, α] · Bbar[P_loc, α, ν]
        // Use cublasDgemmStridedBatched with strideA = 0 (broadcast D),
        // strideB = nbas² (Bbar slab stride), strideC = nbas² (X slab stride).
        //
        // Row→col-major: for C_rm = A_rm · B_rm, call cublasDgemm with operands
        // swapped (B then A) and OP=N both. Same for batched.
        real_t* d_X_local = nullptr;
        tracked_cudaMalloc(&d_X_local, (size_t)nl * nbas2 * sizeof(real_t));
        cublasDgemmStridedBatched(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  nbas, nbas, nbas,
                                  &one,
                                  d_Bbar_local, nbas, (long long)nbas2,   // B_rm view
                                  d_D,          nbas, 0LL,                // A_rm view (broadcast)
                                  &zero,
                                  d_X_local,    nbas, (long long)nbas2,   // C_rm view
                                  nl);
        cudaStreamSynchronize(stream);

        // ----- A_local = X_local · D  (batched DGEMM, overwrites Bbar_local) -----
        //
        // A[P_loc, μ, ν] = Σ_α X[P_loc, μ, α] · D[α, ν]
        // We no longer need Bbar_local: reuse its buffer for A_local.
        real_t* d_A_local = d_Bbar_local;   // alias / buffer reuse
        cublasDgemmStridedBatched(handle,
                                  CUBLAS_OP_N, CUBLAS_OP_N,
                                  nbas, nbas, nbas,
                                  &one,
                                  d_D,        nbas, 0LL,                // B_rm view (broadcast)
                                  d_X_local,  nbas, (long long)nbas2,   // A_rm view
                                  &zero,
                                  d_A_local,  nbas, (long long)nbas2,
                                  nl);
        tracked_cudaFree(d_X_local);
        cudaStreamSynchronize(stream);

        // ----- Γ^(3)_local[P_loc, μν] = γ[P_start + P_loc] · D − ½ A_local -----
        real_t* d_gamma3_local = nullptr;
        tracked_cudaMalloc(&d_gamma3_local, (size_t)nl * nbas2 * sizeof(real_t));
        {
            const size_t total = (size_t)nl * nbas2;
            const int threads = 256;
            const size_t blocks = (total + threads - 1) / threads;
            k_assemble_gamma3_local<<<(unsigned int)blocks, threads, 0, stream>>>(
                d_gamma3_local, d_A_local, d_gamma, d_D,
                nl, nbas, P_start);
        }
        cudaStreamSynchronize(stream);

        // ----- Γ^(2)_local[P_loc, Q]
        //         = ¼ (A_local · B^T · L⁻¹)[P_loc, Q] − ½ γ[P_start+P_loc] γ[Q] -----
        //
        // Avoid materializing full Bbar: derive
        //   (A_local · Bbar^T)[P_loc, Q] = Σ_R (A_local · B^T)[P_loc, R] · L⁻¹[R, Q]
        //
        // Step 1: T_local = A_local · B^T               size (nl × naux)
        // Step 2: Γ^(2)_local = ¼ T_local · L⁻¹         size (nl × naux)
        // Step 3: Γ^(2)_local += −½ γ_slab γᵀ           outer product via DGER
        real_t* d_T_local = nullptr;
        real_t* d_gamma2_local = nullptr;
        tracked_cudaMalloc(&d_T_local,      (size_t)nl * naux * sizeof(real_t));
        tracked_cudaMalloc(&d_gamma2_local, (size_t)nl * naux * sizeof(real_t));

        // Row-major: T[P_loc, R] = Σ_μν A_local[P_loc, μν] · B[R, μν]
        //          = A_local · B^T.
        // C_cm[R, P_loc] = Σ_μν B_cm[μν, R] · A_local_cm[μν, P_loc]
        //                = (B_cm^T · A_local_cm)[R, P_loc]
        // Col-major DGEMM: m = naux, n = nl, k = nbas²
        //   A_param = d_B (col-major view of row-major B); op_A = T → B_cm^T (naux × nbas²)
        //   B_param = d_A_local;                            op_B = N → A_local_cm (nbas² × nl)
        //   ld_A = nbas², ld_B = nbas², ld_C = naux
        cublasDgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    naux, nl, (int)nbas2,
                    &one,
                    d_B,        (int)nbas2,
                    d_A_local,  (int)nbas2,
                    &zero,
                    d_T_local,  naux);

        // Γ^(2)_local = ¼ T_local · L⁻¹
        // Row-major: Γ[P_loc, Q] = Σ_R T[P_loc, R] · L⁻¹[R, Q]
        // Col-major: m = naux, n = nl, k = naux
        //   A_cm = L⁻¹ (col-major view) — ld=naux, OP=N
        //   B_cm = T_local (col-major view) — ld=naux, OP=N
        const double quarter = 0.25;
        cublasDgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    naux, nl, naux,
                    &quarter,
                    d_Linv,         naux,
                    d_T_local,      naux,
                    &zero,
                    d_gamma2_local, naux);
        tracked_cudaFree(d_T_local);

        // Γ^(2)_local += −½ γ_slab γᵀ      (outer product, naux_local × naux)
        //
        // Row-major target: Γ[P_loc, Q] += −½ γ_full[P_start + P_loc] · γ_full[Q]
        // cublasDger col-major: A[i, j] += α x[i] y[j], ld_A = m.
        //   Place A_cm = d_gamma2_local with m=naux, n=nl, ld=naux.
        //   A_cm[i, j] = A_rm[j, i] = Γ[j, i] for j ∈ [0,nl), i ∈ [0, naux).
        //   With α = −½, x = γ_full (naux), y = γ_full + P_start (nl):
        //     A_cm[i, j] += −½ γ_full[i] · γ_full[P_start + j]
        //   i.e. Γ_rm[j, i] += −½ γ_full[P_start + j] · γ_full[i] ✓
        const double minus_half = -0.5;
        cublasDger(handle,
                   naux, nl,
                   &minus_half,
                   d_gamma,           1,
                   d_gamma + P_start, 1,
                   d_gamma2_local,    naux);
        cudaStreamSynchronize(stream);

        // Ensure prep is complete before kernel launches (they read d_gammaN
        // via phantom pointers, no further DGEMM after this point).
        cudaStreamSynchronize(stream);
        tracked_cudaFree(d_gamma);

        // ----- 2-electron gradient kernels (3c2e + 2c2e), local P range -----
        // Kernel reads Γ via P.basis_index (global), so restrict by basis
        // range. h_aux_prim is host data (cudaMallocHost-backed, safe to read
        // from OMP threads).
        launch_3c2e_grad_local(
            d_grad_per_gpu[d], d_gamma3_local,
            shell_type_infos, aux_shell_type_infos,
            d_prim_per_gpu[d], d_aux_prim_per_gpu[d],
            d_cgto_per_gpu[d], d_aux_cgto_per_gpu[d],
            nbas, naux, d_boys_per_gpu[d],
            h_aux_prim, P_start, P_end);
        cudaDeviceSynchronize();  // local sub-stage timing barrier
        cudaStreamSynchronize(stream);

        launch_2c2e_grad_local(
            d_grad_per_gpu[d], d_gamma2_local,
            aux_shell_type_infos,
            d_aux_prim_per_gpu[d], d_aux_cgto_per_gpu[d],
            naux, d_boys_per_gpu[d],
            h_aux_prim, P_start, P_end);

        cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);

        tracked_cudaFree(d_gamma3_local);
        tracked_cudaFree(d_gamma2_local);
        tracked_cudaFree(d_A_local);  // same buffer as d_Bbar_local (alias)
    }


    // ---- 6) NCCL AllReduce the gradient across all GPUs ----
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        nccl::all_reduce(d_grad_per_gpu[d], d_grad_per_gpu[d],
                         (size_t)n3, ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();
    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        cudaStreamSynchronize(mgr.comm_stream(d));
    }


    // ---- 7) Copy result from GPU 0 to host ----
    cudaSetDevice(0);
    std::vector<double> gradient(n3);
    cudaMemcpy(gradient.data(), d_grad_per_gpu[0], n3 * sizeof(double),
               cudaMemcpyDeviceToHost);

    // ---- 8) Cleanup ----
    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        if (d_grad_per_gpu[g])  tracked_cudaFree(d_grad_per_gpu[g]);
        if (d_L_inv_per_gpu[g]) tracked_cudaFree(d_L_inv_per_gpu[g]);
    }
    cudaSetDevice(0);

    free_per_gpu_replicas(d_D_per_gpu);
    free_per_gpu_replicas(d_aux_schwarz_per_gpu);
    free_per_gpu_replicas(d_aux_cgto_per_gpu);
    free_per_gpu_replicas(d_aux_prim_per_gpu);
    free_per_gpu_replicas(d_boys_per_gpu);
    free_per_gpu_replicas(d_atoms_per_gpu);
    free_per_gpu_replicas(d_cgto_per_gpu);
    free_per_gpu_replicas(d_prim_per_gpu);

    if (hf_.get_verbose()) {
        std::cout << "[RI-Distributed Gradient v4] Done." << std::endl;
    }
    return gradient;
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
