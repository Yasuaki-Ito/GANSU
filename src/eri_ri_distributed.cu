/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file eri_ri_distributed.cu
 * @brief Distributed RI-HF Fock build across multiple GPUs
 *
 * Distributed B construction (no full B on any single GPU):
 *   1. GPU 0: 2c2e → Cholesky → L⁻¹ → broadcast L⁻¹ to all GPUs
 *   2. For each aux shell type c (chunked, small memory):
 *      - All GPUs: compute 3c2e chunk for shell type c
 *      - All GPUs: B_local[P_local] += L⁻¹[P_local, Q_c] × 3c_chunk  (DGEMM)
 *   3. Result: each GPU holds B_local [naux_local × nbas²]
 *
 * Memory per GPU: B_local + L⁻¹ + 3c_chunk (one shell type)
 * Limit (4×H200): ~2700 basis functions
 */

#ifdef GANSU_MULTI_GPU

#include "rhf.hpp"
#include "dmet.hpp"
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "spherical_transform.hpp"
#include "ri_adc2_schur_distributed_operator.hpp"
#include "sos_laplace_adc2_distributed_operator.hpp"
#include "davidson_solver.hpp"
#include "oscillator_strength.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

// Forward declarations from gpu_manager.cu
namespace gpu {
    void computeTwoCenterERIs(
        const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
        const PrimitiveShell* d_auxiliary_primitive_shells,
        const real_t* d_auxiliary_cgto_normalization_factors,
        real_t* d_two_center_eri,
        const int num_auxiliary_basis,
        const real_t* d_boys_grid,
        const real_t* d_auxiliary_schwarz_upper_bound_factors,
        const real_t schwarz_screening_threshold,
        const bool verbose);

    void choleskyDecomposition(real_t* d_A, int n);

    void computeThreeCenterERIs(
        const std::vector<ShellTypeInfo>& shell_type_infos,
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
        const PrimitiveShell* d_primitive_shells,
        const real_t* d_cgto_normalization_factors,
        const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
        const PrimitiveShell* d_auxiliary_primitive_shells,
        const real_t* d_auxiliary_cgto_normalization_factors,
        real_t* d_three_center_eri,
        const size_t2* d_primitive_shell_pair_indices,
        const int num_basis,
        const int num_auxiliary_basis,
        const real_t* d_boys_grid,
        const real_t* d_schwarz_upper_bound_factors,
        const real_t* d_auxiliary_schwarz_upper_bound_factors,
        const real_t schwarz_screening_threshold,
        const bool verbose);

    void computeThreeCenterERIs_for_aux_type(
        const std::vector<ShellTypeInfo>& shell_type_infos,
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
        const PrimitiveShell* d_primitive_shells,
        const real_t* d_cgto_normalization_factors,
        const ShellTypeInfo& aux_shell_info,
        int aux_type_angular_momentum,
        const PrimitiveShell* d_auxiliary_primitive_shells,
        const real_t* d_auxiliary_cgto_normalization_factors,
        real_t* d_chunk,
        const size_t2* d_primitive_shell_pair_indices,
        const int num_basis,
        const int num_auxiliary_basis,
        const real_t* d_boys_grid,
        const real_t* d_schwarz_upper_bound_factors,
        const real_t* d_auxiliary_schwarz_upper_bound_factors,
        const real_t schwarz_screening_threshold,
        size_t aux_basis_offset,
        int nfunc_chunk,
        cudaStream_t stream);
}

// ============================================================
//  Kernels
// ============================================================

__global__ void distributed_fock_assemble_kernel(
    const double* __restrict__ H,
    const double* __restrict__ J,
    const double* __restrict__ K,
    double* __restrict__ F, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    F[idx] = H[idx] + J[idx] - 0.5 * K[idx];
}

__global__ void distributed_J_accumulate_kernel(
    double* __restrict__ J,
    const double* __restrict__ B_local,
    const double* __restrict__ W_local,
    int nbas, int naux_local)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nbas2 = nbas * nbas;
    if (idx >= nbas2) return;
    double val = 0.0;
    for (int p = 0; p < naux_local; p++)
        val += W_local[p] * B_local[(size_t)p * nbas2 + idx];
    J[idx] = val;
}

__global__ void distributed_pack_X_kernel(
    const double* __restrict__ X,
    double* __restrict__ X_packed,
    int nbas, int naux_local, int nocc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= naux_local * nbas * nocc) return;
    int i = idx % nocc;
    int mu = (idx / nocc) % nbas;
    int P = idx / (nocc * nbas);
    X_packed[(size_t)mu * naux_local * nocc + P * nocc + i] = X[idx];
}

// ============================================================
//  Constructor / Destructor
// ============================================================
ERI_RI_Distributed_RHF::ERI_RI_Distributed_RHF(RHF& rhf, const Molecular& auxiliary_molecular)
    : ERI_RI_RHF(rhf, auxiliary_molecular, LightweightTag{})
{
    auto& mgr = MultiGpuManager::instance();
    num_gpus_ = mgr.num_devices();

    naux_local_.resize(num_gpus_);
    P_start_.resize(num_gpus_);
    if (mgr.is_mpi()) {
        // MPI mode: this rank owns a single local GPU (num_gpus_ == 1) and a
        // single aux slab. The aux axis is partitioned across WORLD RANKS, not
        // local devices; the world NCCL comm sums the per-rank J/K contributions.
        // Every device-loop below runs once on this rank's GPU.
        auto [start, end] = aux_partition(num_auxiliary_basis_, mgr.world_size(), mgr.world_rank());
        P_start_[0] = start;
        naux_local_[0] = (int)(end - start);
    } else {
        for (int d = 0; d < num_gpus_; d++) {
            auto [start, end] = aux_partition(num_auxiliary_basis_, num_gpus_, d);
            P_start_[d] = start;
            naux_local_[d] = (int)(end - start);
        }
    }

    d_B_local_.resize(num_gpus_, nullptr);
    d_W_local_.resize(num_gpus_, nullptr);
    d_J_local_.resize(num_gpus_, nullptr);
    d_K_local_.resize(num_gpus_, nullptr);
    d_X_local_.resize(num_gpus_, nullptr);
    d_X_packed_local_.resize(num_gpus_, nullptr);

    if (mgr.is_mpi()) {
        if (mgr.world_rank() == 0)
            std::cout << "[RI-Distributed] " << mgr.world_size() << " MPI rank(s) x 1 GPU, naux="
                      << num_auxiliary_basis_ << " (~" << naux_local_[0] << "/rank)" << std::endl;
    } else {
        std::cout << "[RI-Distributed] " << num_gpus_ << " GPUs, naux=" << num_auxiliary_basis_;
        for (int d = 0; d < num_gpus_; d++)
            std::cout << " [" << d << "]:" << naux_local_[d];
        std::cout << std::endl;
    }
}

ERI_RI_Distributed_RHF::~ERI_RI_Distributed_RHF() {
    free_host_partitions();
    free_chunked_workspace();
    free_per_device_workspace();
    free_per_gpu_data();
    if (d_cached_L_inv_) {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_cached_L_inv_);
        d_cached_L_inv_ = nullptr;
    }
    if (d_cached_M_) {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_cached_M_);
        d_cached_M_ = nullptr;
    }
}

void ERI_RI_Distributed_RHF::allocate_chunked_workspace() {
    if (!chunked_ws_.empty()) return;  // already allocated

    const int nbas = num_basis_;
    const int nocc = num_occ_;
    const size_t nbas2 = (size_t)nbas * nbas;
    const bool has_C = rhf_.get_hasMatrixC();

    // Determine N_rows from free memory.
    // Budget: B_row (N_rows × nbas²) + 3c_chunk (max_nfunc × nbas²) + X + Xp + W
    size_t free_mem = 0, total_mem = 0;
    { cudaSetDevice(0); cudaMemGetInfo(&free_mem, &total_mem); }
    const size_t reserved = nbas2 * 3 * sizeof(double) + 512ULL * 1024 * 1024;  // J,K,D + margin
    const size_t available = (free_mem > reserved) ? free_mem - reserved : 512ULL * 1024 * 1024;

    // 3c_chunk is the biggest fixed cost (per aux type)
    const size_t chunk_3c_bytes = (size_t)max_nfunc_chunk_ * nbas2 * sizeof(double);
    const size_t remaining = (available > chunk_3c_bytes) ? available - chunk_3c_bytes : available / 2;

    const size_t bytes_per_row = nbas2 * sizeof(double)
                               + (has_C ? 2 * (size_t)nbas * nocc * sizeof(double) : 2 * nbas2 * sizeof(double))
                               + sizeof(double);
    int max_rows = (bytes_per_row > 0) ? (int)(remaining / bytes_per_row) : num_auxiliary_basis_;
    int max_naux_local = 0;
    for (int d = 0; d < num_gpus_; d++)
        if (naux_local_[d] > max_naux_local) max_naux_local = naux_local_[d];
    chunked_N_rows_ = std::max(1, std::min(max_rows, max_naux_local));

    // If 3c_chunk alone exceeds available memory, fall back to stored RI build + scatter
    if (chunk_3c_bytes > available * 9 / 10) {
        std::cout << "[RI-Dist] 3c chunk too large for chunked Fock. "
                  << "Falling back to B_local build + standard Fock." << std::endl;
        distributed_build_B();
        // Use stored RI Fock path (coefficient or density-matrix based)
        if (has_C) {
            // Delegate to the stored RI coefficient-based Fock build (below this function)
            // by breaking out and letting compute_fock_matrix handle it
        }
        // NOTE: For this fallback, we set scattered_ and return to compute_fock_matrix
        // which will proceed with the stored RI J/K build.
        return;
    }

    chunked_ws_.resize(num_gpus_);
    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        auto& w = chunked_ws_[d];
        // J/K accumulators (used by AllReduce)
        if (!d_J_local_[d]) cudaMalloc(&d_J_local_[d], nbas2 * sizeof(double));
        if (!d_K_local_[d]) cudaMalloc(&d_K_local_[d], nbas2 * sizeof(double));
        cudaMalloc(&w.d_B_row, (size_t)chunked_N_rows_ * nbas2 * sizeof(double));
        cudaMalloc(&w.d_3c_chunk, (size_t)max_nfunc_chunk_ * nbas2 * sizeof(double));
        cudaMalloc(&w.d_W, chunked_N_rows_ * sizeof(double));
        if (has_C) {
            cudaMalloc(&w.d_X, (size_t)chunked_N_rows_ * nbas * nocc * sizeof(double));
            cudaMalloc(&w.d_Xp, (size_t)chunked_N_rows_ * nbas * nocc * sizeof(double));
        } else {
            cudaMalloc(&w.d_X, (size_t)chunked_N_rows_ * nbas2 * sizeof(double));
            cudaMalloc(&w.d_Xp, (size_t)chunked_N_rows_ * nbas2 * sizeof(double));
        }
        if (d > 0) {
            cudaMalloc(&w.d_D, nbas2 * sizeof(double));
            if (has_C) cudaMalloc(&w.d_C, nbas2 * sizeof(double));
        }
    }

    int total_sub = 0;
    for (int d = 0; d < num_gpus_; d++)
        total_sub += (naux_local_[d] + chunked_N_rows_ - 1) / chunked_N_rows_;
    std::cout << "[RI-Dist] Chunked Fock: max_rows=" << chunked_N_rows_
              << ", " << total_sub << " sub-chunk(s) across " << num_gpus_ << " GPU(s)" << std::endl;
    cudaSetDevice(0);
}

void ERI_RI_Distributed_RHF::free_chunked_workspace() {
    for (int d = 0; d < (int)chunked_ws_.size(); d++) {
        cudaSetDevice(d);
        auto& w = chunked_ws_[d];
        if (w.d_B_row) cudaFree(w.d_B_row);
        if (w.d_3c_chunk) cudaFree(w.d_3c_chunk);
        if (w.d_W) cudaFree(w.d_W);
        if (w.d_X) cudaFree(w.d_X);
        if (w.d_Xp) cudaFree(w.d_Xp);
        if (w.d_D) cudaFree(w.d_D);
        if (w.d_C) cudaFree(w.d_C);
    }
    chunked_ws_.clear();
    cudaSetDevice(0);
}

void ERI_RI_Distributed_RHF::replicate_data_to_gpus() {
    if (per_gpu_data_ready_) return;

    const int naux = num_auxiliary_basis_;
    const size_t nbas2 = (size_t)num_basis_ * num_basis_;
    per_gpu_data_.resize(num_gpus_);

    // Phase 2b spherical-distributed-RI extra per-device buffers.
    const bool use_spherical = hf_.get_use_spherical();
    const int  nc_orb = use_spherical ? hf_.get_num_basis_cart()    : num_basis_;
    const int  ns_orb = num_basis_;                  // = nc_orb in Cart mode
    const int  nc_aux = use_spherical ? num_auxiliary_basis_cart_   : num_auxiliary_basis_;
    const size_t nc_orb2 = (size_t)nc_orb * nc_orb;

    const size_t n_pshells = hf_.get_primitive_shells().size();
    const size_t n_aux_pshells = auxiliary_primitive_shells_.size();
    const size_t n_cgto = hf_.get_cgto_normalization_factors().size();
    const size_t n_aux_cgto = auxiliary_cgto_normalization_factors_.size();
    const size_t n_boys = hf_.get_boys_grid().size();

    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& g = per_gpu_data_[d];
        if (d == 0) {
            g.d_L_inv = d_cached_L_inv_;
            g.d_M = d_cached_M_;   // borrowed (nullptr in Cart mode)
            g.d_pshells = const_cast<PrimitiveShell*>(hf_.get_primitive_shells().device_ptr());
            g.d_cgto_norms = const_cast<real_t*>(hf_.get_cgto_normalization_factors().device_ptr());
            g.d_aux_pshells = auxiliary_primitive_shells_.device_ptr();
            g.d_aux_cgto_norms = auxiliary_cgto_normalization_factors_.device_ptr();
            g.d_shell_pairs = d_persistent_shell_pair_indices_;
            g.d_schwarz = schwarz_upper_bound_factors.device_ptr();
            g.d_aux_schwarz = auxiliary_schwarz_upper_bound_factors.device_ptr();
            g.d_boys = const_cast<real_t*>(hf_.get_boys_grid().device_ptr());
        } else {
            if (use_spherical) {
                // Spherical: replicate M = L_sph⁻¹·U_aux [ns_aux × nc_aux]; L⁻¹ unused.
                cudaMalloc(&g.d_M, (size_t)naux * nc_aux * sizeof(real_t));
                cudaMemcpy(g.d_M, d_cached_M_, (size_t)naux * nc_aux * sizeof(real_t), cudaMemcpyDefault);
            } else {
                cudaMalloc(&g.d_L_inv, (size_t)naux * naux * sizeof(real_t));
                cudaMemcpy(g.d_L_inv, d_cached_L_inv_, (size_t)naux * naux * sizeof(real_t), cudaMemcpyDefault);
            }
            cudaMalloc(&g.d_pshells, n_pshells * sizeof(PrimitiveShell));
            cudaMemcpy(g.d_pshells, hf_.get_primitive_shells().device_ptr(), n_pshells * sizeof(PrimitiveShell), cudaMemcpyDefault);
            cudaMalloc(&g.d_cgto_norms, n_cgto * sizeof(real_t));
            cudaMemcpy(g.d_cgto_norms, hf_.get_cgto_normalization_factors().device_ptr(), n_cgto * sizeof(real_t), cudaMemcpyDefault);
            cudaMalloc(&g.d_aux_pshells, n_aux_pshells * sizeof(PrimitiveShell));
            cudaMemcpy(g.d_aux_pshells, auxiliary_primitive_shells_.device_ptr(), n_aux_pshells * sizeof(PrimitiveShell), cudaMemcpyDefault);
            cudaMalloc(&g.d_aux_cgto_norms, n_aux_cgto * sizeof(real_t));
            cudaMemcpy(g.d_aux_cgto_norms, auxiliary_cgto_normalization_factors_.device_ptr(), n_aux_cgto * sizeof(real_t), cudaMemcpyDefault);
            cudaMalloc(&g.d_shell_pairs, num_persistent_shell_pairs_ * sizeof(size_t2));
            cudaMemcpy(g.d_shell_pairs, d_persistent_shell_pair_indices_, num_persistent_shell_pairs_ * sizeof(size_t2), cudaMemcpyDefault);
            cudaMalloc(&g.d_schwarz, schwarz_upper_bound_factors.size() * sizeof(real_t));
            cudaMemcpy(g.d_schwarz, schwarz_upper_bound_factors.device_ptr(), schwarz_upper_bound_factors.size() * sizeof(real_t), cudaMemcpyDefault);
            cudaMalloc(&g.d_aux_schwarz, auxiliary_schwarz_upper_bound_factors.size() * sizeof(real_t));
            cudaMemcpy(g.d_aux_schwarz, auxiliary_schwarz_upper_bound_factors.device_ptr(), auxiliary_schwarz_upper_bound_factors.size() * sizeof(real_t), cudaMemcpyDefault);
            cudaMalloc(&g.d_boys, n_boys * sizeof(real_t));
            cudaMemcpy(g.d_boys, hf_.get_boys_grid().device_ptr(), n_boys * sizeof(real_t), cudaMemcpyDefault);
        }
        // Pre-allocate chunk buffer (max aux type size) on every GPU.
        // In spherical mode this holds the Sph-transformed chunk [max_nfunc × ns_orb²].
        cudaMalloc(&g.d_chunk, (size_t)max_nfunc_chunk_ * nbas2 * sizeof(real_t));

        if (use_spherical) {
            // Cart 3c2e kernel output (pre-transform) + Stage-A orbital scratch +
            // a per-device prebuilt orbital U.  Allocated on every GPU (scratch,
            // not borrowed) so each device transforms its own chunks concurrently.
            cudaMalloc(&g.d_chunk_cart, (size_t)max_nfunc_chunk_ * nc_orb2 * sizeof(real_t));
            cudaMalloc(&g.d_T_orb,      (size_t)max_nfunc_chunk_ * (size_t)ns_orb * nc_orb * sizeof(real_t));
            spherical::build_cart_to_sph_U_device(
                hf_.get_shell_types(), hf_.get_shell_offsets_cart(), hf_.get_shell_offsets_sph(),
                &g.d_U_orb);
        }
    }
    MultiGpuManager::instance().sync_all();
    per_gpu_data_ready_ = true;
}

void ERI_RI_Distributed_RHF::free_per_gpu_data() {
    for (int d = 0; d < (int)per_gpu_data_.size(); d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& g = per_gpu_data_[d];
        if (g.d_chunk) { cudaFree(g.d_chunk); g.d_chunk = nullptr; }
        // Spherical scratch is owned on every device (incl. GPU 0).
        if (g.d_chunk_cart) { cudaFree(g.d_chunk_cart); g.d_chunk_cart = nullptr; }
        if (g.d_T_orb)      { cudaFree(g.d_T_orb);      g.d_T_orb = nullptr; }
        if (g.d_U_orb)      { cudaFree(g.d_U_orb);      g.d_U_orb = nullptr; }
        if (d == 0) continue;  // GPU 0 data is borrowed, not owned
        if (g.d_L_inv) cudaFree(g.d_L_inv);
        if (g.d_M) cudaFree(g.d_M);
        if (g.d_pshells) cudaFree(g.d_pshells);
        if (g.d_cgto_norms) cudaFree(g.d_cgto_norms);
        if (g.d_aux_pshells) cudaFree(g.d_aux_pshells);
        if (g.d_aux_cgto_norms) cudaFree(g.d_aux_cgto_norms);
        if (g.d_shell_pairs) cudaFree(g.d_shell_pairs);
        if (g.d_schwarz) cudaFree(g.d_schwarz);
        if (g.d_aux_schwarz) cudaFree(g.d_aux_schwarz);
        if (g.d_boys) cudaFree(g.d_boys);
    }
    per_gpu_data_.clear();
    per_gpu_data_ready_ = false;
}

void ERI_RI_Distributed_RHF::allocate_per_device_workspace() {
    if (d_J_local_[0]) return;
    const size_t nbas2 = (size_t)num_basis_ * num_basis_;
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        int nl = naux_local_[d];
        cudaMalloc(&d_W_local_[d], nl * sizeof(double));
        cudaMalloc(&d_J_local_[d], nbas2 * sizeof(double));
        cudaMalloc(&d_K_local_[d], nbas2 * sizeof(double));
        cudaMalloc(&d_X_local_[d], (size_t)nl * num_basis_ * num_occ_ * sizeof(double));
        cudaMalloc(&d_X_packed_local_[d], (size_t)num_basis_ * nl * num_occ_ * sizeof(double));
    }
}

void ERI_RI_Distributed_RHF::free_per_device_workspace() {
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d_B_local_[d]) { cudaFree(d_B_local_[d]); d_B_local_[d] = nullptr; }
        if (d_W_local_[d]) { cudaFree(d_W_local_[d]); d_W_local_[d] = nullptr; }
        if (d_J_local_[d]) { cudaFree(d_J_local_[d]); d_J_local_[d] = nullptr; }
        if (d_K_local_[d]) { cudaFree(d_K_local_[d]); d_K_local_[d] = nullptr; }
        if (d_X_local_[d]) { cudaFree(d_X_local_[d]); d_X_local_[d] = nullptr; }
        if (d_X_packed_local_[d]) { cudaFree(d_X_packed_local_[d]); d_X_packed_local_[d] = nullptr; }
    }
}

// ============================================================
//  Compute per-auxiliary-shell-type basis function ranges
// ============================================================
void ERI_RI_Distributed_RHF::compute_aux_type_ranges() {
    const auto& aux_types = auxiliary_shell_type_infos_;
    const int n_aux_types = (int)aux_types.size();
    const int nbas = num_basis_;
    const size_t nbas2 = (size_t)nbas * nbas;

    // Determine max batch size: free memory minus B_local and workspace
    size_t free_mem = 0, total_mem = 0;
    { cudaSetDevice(0); cudaMemGetInfo(&free_mem, &total_mem); }
    const size_t B_local_bytes = (size_t)naux_local_[0] * nbas2 * sizeof(real_t);
    const size_t overhead_bytes = 2ULL * 1024 * 1024 * 1024;  // L⁻¹, J, K, W, X, safety margin
    const size_t used = B_local_bytes + overhead_bytes;
    const size_t available_for_chunk = (free_mem > used) ? free_mem - used : free_mem / 4;
    // Per-aux-function chunk footprint.  In spherical mode each batch is held
    // in THREE buffers before the M-DGEMM: the Cart kernel output (nc_orb²),
    // the orbital Stage-A scratch (ns_orb·nc_orb) and the Sph output (ns_orb²).
    // Budget against their sum so the batch limit stays within memory at scale.
    size_t per_func_bytes = nbas2 * sizeof(real_t);
    if (hf_.get_use_spherical()) {
        const size_t nc_orb = (size_t)hf_.get_num_basis_cart();
        const size_t ns_orb = (size_t)num_basis_;
        per_func_bytes = (nc_orb * nc_orb + ns_orb * nc_orb + ns_orb * ns_orb) * sizeof(real_t);
    }
    const int max_batch_nfunc = std::max(1, (int)(available_for_chunk / per_func_bytes));

    // Sort auxiliary primitives by basis_index within each type (on host).
    // This ensures consecutive primitives have contiguous basis_index → efficient batching.
    // (Replaces Schwarz sort order for the purpose of 3c2e chunking.)
    PrimitiveShell* h_aux = auxiliary_primitive_shells_.host_ptr();
    for (int c = 0; c < n_aux_types; c++) {
        const size_t start = aux_types[c].start_index;
        const size_t count = aux_types[c].count;
        std::sort(h_aux + start, h_aux + start + count,
            [](const PrimitiveShell& a, const PrimitiveShell& b) {
                return a.basis_index < b.basis_index;
            });
    }
    // Upload sorted order to device
    auxiliary_primitive_shells_.toDevice();

    // Build batches: split each aux type into chunks of max_batch_nfunc basis functions
    aux_batches_.clear();
    for (int c = 0; c < n_aux_types; c++) {
        const size_t type_start = aux_types[c].start_index;
        const size_t type_count = aux_types[c].count;
        const int L = h_aux[type_start].shell_type;
        const int n_cart = (L + 1) * (L + 2) / 2;

        // Primitives are now sorted by basis_index within this type.
        // Split into batches where each batch covers at most max_batch_nfunc basis functions.
        size_t batch_prim_start = 0;
        while (batch_prim_start < type_count) {
            const size_t first_basis = h_aux[type_start + batch_prim_start].basis_index;
            const size_t max_basis = first_basis + max_batch_nfunc - n_cart;

            // Find how many primitives fit in this batch
            size_t batch_prim_end = batch_prim_start;
            while (batch_prim_end < type_count &&
                   h_aux[type_start + batch_prim_end].basis_index <= max_basis) {
                batch_prim_end++;
            }
            if (batch_prim_end == batch_prim_start) batch_prim_end++;  // at least 1

            // Compute actual nfunc for this batch
            const size_t last_basis = h_aux[type_start + batch_prim_end - 1].basis_index;
            const int batch_nfunc = (int)(last_basis - first_basis) + n_cart;
            const int batch_count = (int)(batch_prim_end - batch_prim_start);

            AuxBatch batch;
            batch.shell_info.start_index = (int)(type_start + batch_prim_start);
            batch.shell_info.count = batch_count;
            batch.basis_start = first_basis;
            batch.nfunc = batch_nfunc;
            batch.angular_momentum = L;
            aux_batches_.push_back(batch);

            batch_prim_start = batch_prim_end;
        }
    }

    // Compute max_nfunc_chunk_ across all batches
    max_nfunc_chunk_ = 0;
    for (const auto& b : aux_batches_)
        if (b.nfunc > max_nfunc_chunk_) max_nfunc_chunk_ = b.nfunc;

    std::cout << "[RI-Dist] Aux batches: " << aux_batches_.size() << " (max_nfunc=" << max_nfunc_chunk_
              << ", limit=" << max_batch_nfunc << ")" << std::endl;
}

// ============================================================
//  Precomputation: Schwarz + shell pairs + 2c2e/Cholesky
//  Then distributed B build (no full B on any single GPU)
// ============================================================
void ERI_RI_Distributed_RHF::precomputation() {
    const bool use_spherical = hf_.get_use_spherical();

    // Spherical support: GPU_Resident (Phase 2b) + OnTheFly/OutOfCore (Phase 2c)
    // all use the M = L_sph⁻¹·U_aux folding (Step 3 builds M; the per-chunk
    // Cart→Sph orbital transform lives in distributed_build_B / chunked_fock_build
    // / build_out_of_core_B).

    const int naux = num_auxiliary_basis_;        // = ns_aux when use_spherical
    const int nbas = num_basis_;                  // = ns_orb when use_spherical

    // Ensure GPU 0 is active (base class data lives on GPU 0)
    cudaSetDevice(0);

    // Geometry may have changed since a previous precomputation (geometry
    // optimization / gradient line search re-invokes precompute_eri_matrix per
    // trial geometry). Drop the previously-built distributed state so B is
    // rebuilt for the CURRENT geometry — otherwise distributed_build_B()
    // early-returns on scattered_ and the SCF / gradient silently uses stale
    // (previous-geometry) B (manifests as energies below the variational
    // minimum + diverging optimization). d_B_local_ buffers are reused in place
    // (re-zeroed + refilled), so only the cached metrics / replicated copy are
    // freed here. At first construction these are all no-ops.
    scattered_ = false;
    free_replicated_B();
    if (d_cached_M_)     { tracked_cudaFree(d_cached_M_);     d_cached_M_ = nullptr; }
    if (d_cached_L_inv_) { tracked_cudaFree(d_cached_L_inv_); d_cached_L_inv_ = nullptr; }

    // Step 1: Schwarz + shell pairs + aux Schwarz (skip full B build)
    precompute_schwarz_and_shell_pairs();

    std::cout << "[RI-Dist] Skipped full B build (lightweight constructor)" << std::endl;

    // Step 2: Compute auxiliary type ranges for chunked 3c2e
    auxiliary_primitive_shells_.toHost();
    compute_aux_type_ranges();

    // Step 3: Compute and cache the per-aux contraction matrix on GPU 0.
    //  Cartesian : d_cached_L_inv_ = L⁻¹            [naux × naux],  L = chol(V_cart)
    //  Spherical : d_cached_M_     = L_sph⁻¹·U_aux  [ns_aux × nc_aux]
    //
    //  The spherical fold is required because V_sph^{-1/2} ≠ U·V_cart^{-1/2}·U^T
    //  (a matrix function does not commute with the Cart→Sph projection), so we
    //  must form V_sph = U·V_cart·U^T first, factor it, and only then absorb the
    //  remaining auxiliary Cart→Sph map U_aux into M.  With M precomputed, each
    //  3c2e chunk contracts in Cart-aux space (folded) and needs only its two
    //  orbital indices transformed — see distributed_build_B().
    {
        cudaSetDevice(0);
        const real_t schwarz_threshold = hf_.get_schwarz_screening_threshold();

        if (!use_spherical) {
            real_t* d_two_center_eri;
            tracked_cudaMalloc(&d_two_center_eri, (size_t)naux * naux * sizeof(real_t));
            cudaMemset(d_two_center_eri, 0, (size_t)naux * naux * sizeof(real_t));

            gpu::computeTwoCenterERIs(
                auxiliary_shell_type_infos_,
                auxiliary_primitive_shells_.device_ptr(),
                auxiliary_cgto_normalization_factors_.device_ptr(),
                d_two_center_eri, naux,
                hf_.get_boys_grid().device_ptr(),
                auxiliary_schwarz_upper_bound_factors.device_ptr(),
                schwarz_threshold, false);

            gpu::choleskyDecomposition(d_two_center_eri, naux);

            tracked_cudaMalloc(&d_cached_L_inv_, (size_t)naux * naux * sizeof(real_t));
            gpu::computeInverseByDtrsm(d_two_center_eri, d_cached_L_inv_, naux);

            tracked_cudaFree(d_two_center_eri);
            cudaDeviceSynchronize();
            std::cout << "[RI-Dist] Cached L^-1 on GPU 0 ("
                      << (size_t)naux * naux * sizeof(real_t) / (1024 * 1024) << " MB)" << std::flush << std::endl;
        } else {
            const int ns_aux = num_auxiliary_basis_;          // Sph aux count
            const int nc_aux = num_auxiliary_basis_cart_;     // Cart aux count

            // (P|Q) in Cart aux, on GPU 0
            real_t* d_2c_cart = nullptr;
            tracked_cudaMalloc(&d_2c_cart, (size_t)nc_aux * nc_aux * sizeof(real_t));
            cudaMemset(d_2c_cart, 0, (size_t)nc_aux * nc_aux * sizeof(real_t));
            gpu::computeTwoCenterERIs(
                auxiliary_shell_type_infos_,
                auxiliary_primitive_shells_.device_ptr(),
                auxiliary_cgto_normalization_factors_.device_ptr(),
                d_2c_cart, nc_aux,
                hf_.get_boys_grid().device_ptr(),
                auxiliary_schwarz_upper_bound_factors.device_ptr(),
                schwarz_threshold, false);

            // V_sph = U_aux · (P|Q)_cart · U_aux^T   [ns_aux × ns_aux]
            real_t* d_V_sph = nullptr;
            tracked_cudaMalloc(&d_V_sph, (size_t)ns_aux * ns_aux * sizeof(real_t));
            spherical::transform_matrix_cart_to_sph_device(
                d_2c_cart, d_V_sph,
                aux_shell_types_, aux_shell_offsets_cart_, aux_shell_offsets_sph_);
            tracked_cudaFree(d_2c_cart);

            // L_sph = chol(V_sph);  L_sph⁻¹   [ns_aux × ns_aux]
            gpu::choleskyDecomposition(d_V_sph, ns_aux);
            real_t* d_Linv_sph = nullptr;
            tracked_cudaMalloc(&d_Linv_sph, (size_t)ns_aux * ns_aux * sizeof(real_t));
            gpu::computeInverseByDtrsm(d_V_sph, d_Linv_sph, ns_aux);
            tracked_cudaFree(d_V_sph);

            // U_aux  [ns_aux × nc_aux]  (block-diagonal Cart→Sph for the aux basis)
            real_t* d_U_aux = nullptr;
            spherical::build_cart_to_sph_U_device(
                aux_shell_types_, aux_shell_offsets_cart_, aux_shell_offsets_sph_, &d_U_aux);

            // M = L_sph⁻¹ · U_aux   [ns_aux × nc_aux]   (row-major C = A·B trick)
            tracked_cudaMalloc(&d_cached_M_, (size_t)ns_aux * nc_aux * sizeof(real_t));
            {
                cublasHandle_t h = gpu::GPUHandle::cublas();
                const double one = 1.0, zero = 0.0;
                cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    nc_aux, ns_aux, ns_aux,
                    &one,
                    d_U_aux,    nc_aux,    // B = U_aux [ns_aux × nc_aux], ldb = nc_aux
                    d_Linv_sph, ns_aux,    // A = L⁻¹   [ns_aux × ns_aux], lda = ns_aux
                    &zero,
                    d_cached_M_, nc_aux);  // C = M     [ns_aux × nc_aux], ldc = nc_aux
            }
            cudaFree(d_U_aux);
            tracked_cudaFree(d_Linv_sph);
            cudaDeviceSynchronize();
            std::cout << "[RI-Dist] Cached M = L_sph^-1 * U_aux on GPU 0 (ns_aux="
                      << ns_aux << ", nc_aux=" << nc_aux << ", "
                      << (size_t)ns_aux * nc_aux * sizeof(real_t) / (1024 * 1024)
                      << " MB)" << std::flush << std::endl;
        }
    }

    // Step 4: Build B based on storage mode
    if (storage_mode_ == StorageMode::GPU_Resident) {
        // Check if B_local itself fits on GPU (chunk sizing is handled by compute_aux_type_ranges)
        const size_t nbas2_local = (size_t)nbas * nbas;
        const size_t B_local_bytes = (size_t)naux_local_[0] * nbas2_local * sizeof(real_t);
        size_t free_check = 0, total_check = 0;
        { cudaSetDevice(0); cudaMemGetInfo(&free_check, &total_check); }

        const size_t overhead = 4ULL * 1024 * 1024 * 1024;  // L⁻¹ + workspace + chunk headroom
        if (B_local_bytes + overhead > free_check) {
            // OutOfCore B build is Cart→Sph aware (Phase 2c, M-folding).
            std::cout << "[RI] Auto-switching to out-of-core (B_local=" << B_local_bytes / (1024*1024)
                      << " MB, GPU free=" << free_check / (1024*1024) << " MB)" << std::endl;
            storage_mode_ = StorageMode::OutOfCore;
        }
    }

    switch (storage_mode_) {
    case StorageMode::GPU_Resident:
        distributed_build_B();
        break;
    case StorageMode::OutOfCore:
        build_out_of_core_B();
        break;
    case StorageMode::OnTheFly:
        // B rebuilt per iteration in compute_fock_matrix
        break;
    }
}

// ============================================================
//  Distributed B build: each GPU independently computes B_local
//  via chunked 3c2e + L⁻¹ DGEMM (no full B needed after first Fock)
// ============================================================
void ERI_RI_Distributed_RHF::distributed_build_B() {
    if (scattered_) return;

    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const size_t nbas2 = (size_t)nbas * nbas;
    auto& mgr = MultiGpuManager::instance();

    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int n_aux_types = (int)auxiliary_shell_type_infos_.size();

    // Phase 2b spherical: the 3c2e kernel and chunk indexing run in Cartesian
    // (nc_orb, nc_aux); only the final B rows are Spherical (naux = ns_aux,
    // nbas = ns_orb).  The aux Cart→Sph map is folded into M (= d_M).
    const bool use_spherical = hf_.get_use_spherical();
    const int  nc_orb = use_spherical ? hf_.get_num_basis_cart()  : nbas;
    const int  nc_aux = use_spherical ? num_auxiliary_basis_cart_ : naux;
    const size_t nc_orb2 = (size_t)nc_orb * nc_orb;

    // Batch sizing in compute_aux_type_ranges() ensures chunks fit alongside B_local.
    std::cout << "[RI-Dist] Building B_local on " << num_gpus_ << " GPUs (chunked 3c2e + "
              << (use_spherical ? "orb-transform + M" : "L^-1") << " DGEMM)..." << std::endl;

    // ---- Step 1: Replicate data to all GPUs ----
    replicate_data_to_gpus();

    // ---- Step 2: Allocate B_local on each GPU and zero-initialize ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        size_t local_size = (size_t)naux_local_[d] * nbas2;
        if (!d_B_local_[d])
            cudaMalloc(&d_B_local_[d], local_size * sizeof(double));
        cudaMemsetAsync(d_B_local_[d], 0, local_size * sizeof(double), mgr.compute_stream(d));
    }

    // ---- Step 3: Chunked 3c2e + DGEMM — all GPUs launch concurrently ----
    for (const auto& batch : aux_batches_) {
        const size_t Q_c_start = batch.basis_start;
        const int nfunc_c = batch.nfunc;

        // Launch on all GPUs (non-blocking)
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cudaStream_t cs = mgr.compute_stream(d);
            cublasHandle_t handle = mgr.cublas(d);
            cublasSetStream(handle, cs);
            auto& g = per_gpu_data_[d];

            const double one = 1.0;
            if (!use_spherical) {
                // Zero pre-allocated chunk buffer
                cudaMemsetAsync(g.d_chunk, 0, (size_t)nfunc_c * nbas2 * sizeof(real_t), cs);

                // Compute 3c2e for this batch (async)
                gpu::computeThreeCenterERIs_for_aux_type(
                    shell_type_infos, shell_pair_type_infos,
                    g.d_pshells, g.d_cgto_norms,
                    batch.shell_info, batch.angular_momentum,
                    g.d_aux_pshells, g.d_aux_cgto_norms,
                    g.d_chunk, g.d_shell_pairs,
                    nbas, naux, g.d_boys,
                    g.d_schwarz, g.d_aux_schwarz,
                    schwarz_screening_threshold,
                    Q_c_start, nfunc_c, cs);

                // DGEMM on the same stream: B_local += L⁻¹_rows × chunk
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)nbas2, naux_local_[d], nfunc_c,
                    &one,
                    g.d_chunk, (int)nbas2,
                    &g.d_L_inv[P_start_[d] * naux + Q_c_start], naux,
                    &one,
                    d_B_local_[d], (int)nbas2);
            } else {
                // --- Spherical: Cart 3c2e → orbital Cart→Sph → fold-M DGEMM ---
                // (1) Cart 3c2e chunk [nfunc_c × nc_orb²] (nfunc_c is Cart aux,
                //     Q_c_start is a Cart aux offset; both come from aux_batches_).
                cudaMemsetAsync(g.d_chunk_cart, 0, (size_t)nfunc_c * nc_orb2 * sizeof(real_t), cs);
                gpu::computeThreeCenterERIs_for_aux_type(
                    shell_type_infos, shell_pair_type_infos,
                    g.d_pshells, g.d_cgto_norms,
                    batch.shell_info, batch.angular_momentum,
                    g.d_aux_pshells, g.d_aux_cgto_norms,
                    g.d_chunk_cart, g.d_shell_pairs,
                    nc_orb, nc_aux, g.d_boys,
                    g.d_schwarz, g.d_aux_schwarz,
                    schwarz_screening_threshold,
                    Q_c_start, nfunc_c, cs);

                // (2) Transform the two orbital indices: [nfunc_c × nc_orb²]
                //     → g.d_chunk [nfunc_c × ns_orb²].  Aux axis (nfunc_c) untouched.
                spherical::transform_orbital_pair_cart_to_sph_device(
                    handle, g.d_chunk_cart, g.d_chunk, g.d_T_orb, g.d_U_orb,
                    nfunc_c, nc_orb, nbas /* = ns_orb */);

                // (3) B_local_sph += M[P_sph_rows, Q_cart_chunk] × chunk_sph.
                //     M is [ns_aux × nc_aux] row-major; slice base/ld use nc_aux,
                //     contraction dim is nfunc_c (Cart aux).  B rows are Sph aux,
                //     already partitioned via P_start_/naux_local_ (over ns_aux).
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)nbas2, naux_local_[d], nfunc_c,
                    &one,
                    g.d_chunk, (int)nbas2,
                    &g.d_M[P_start_[d] * (size_t)nc_aux + Q_c_start], nc_aux,
                    &one,
                    d_B_local_[d], (int)nbas2);
            }
        }

        // Sync all GPUs before next aux type (chunk reused)
        mgr.sync_all();
    }

    // ---- Step 4: Cleanup ----
    // In stored mode, free replicated data + cached L⁻¹/M (no longer needed).
    // In direct mode, keep them for per-iteration rebuilds.
    if (storage_mode_ != StorageMode::OnTheFly) {
        free_per_gpu_data();
        if (d_cached_L_inv_) {
            MultiGpuManager::DeviceGuard guard(0);
            tracked_cudaFree(d_cached_L_inv_);
            d_cached_L_inv_ = nullptr;
        }
        if (d_cached_M_) {
            MultiGpuManager::DeviceGuard guard(0);
            tracked_cudaFree(d_cached_M_);
            d_cached_M_ = nullptr;
        }
    }

    // Release intermediate_matrix_B_ dummy from lightweight constructor
    { MultiGpuManager::DeviceGuard guard(0); intermediate_matrix_B_.release(); }

    scattered_ = true;
    allocate_per_device_workspace();

    for (int d = 0; d < num_gpus_; d++) {
        double mb = (double)naux_local_[d] * nbas2 * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  [GPU " << d << "] B_local: " << std::fixed << std::setprecision(1)
                  << mb << " MB (" << naux_local_[d] << " aux)" << std::endl;
    }
    std::cout << "[RI-Dist] B distributed to " << num_gpus_ << " GPUs (independent build)" << std::endl;
}

// ============================================================
//  Distributed Fock build
// ============================================================
void ERI_RI_Distributed_RHF::compute_fock_matrix() {
    // OnTheFly: rebuild B per iteration (semi_direct_ri / direct_ri)
    if (storage_mode_ == StorageMode::OnTheFly) {
        chunked_fock_build();
        return;
    }
    // OutOfCore: B stored on host, streamed to GPU per iteration
    if (storage_mode_ == StorageMode::OutOfCore) {
        out_of_core_fock_build();
        return;
    }
    // GPU_Resident: B on GPU(s)
    if (!scattered_) distributed_build_B();

    // Density-matrix-based Fock (before coefficient matrix is available, e.g. SAD guess)
    if (!rhf_.get_hasMatrixC()) {
        const int nbas = num_basis_;
        const size_t nbas2 = (size_t)nbas * nbas;
        const int threads = 256;
        auto& mgr = MultiGpuManager::instance();
        const real_t* d_D_gpu0 = rhf_.get_density_matrix().device_ptr();
        const real_t* d_H_gpu0 = rhf_.get_core_hamiltonian_matrix().device_ptr();

        // Replicate D to all GPUs
        std::vector<real_t*> d_D(num_gpus_, nullptr);
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            if (d == 0) { d_D[d] = const_cast<real_t*>(d_D_gpu0); }
            else {
                cudaMalloc(&d_D[d], nbas2 * sizeof(double));
                cudaMemcpy(d_D[d], d_D_gpu0, nbas2 * sizeof(double), cudaMemcpyDefault);
            }
        }

        // MPI: every rank must contract its aux slab against the SAME density
        // (rank 0's), so the AllReduce over rank slabs reconstructs the correct
        // global J/K. Broadcast rank 0's D into every rank's buffer.
        if (mgr.is_mpi()) {
            MultiGpuManager::DeviceGuard guard(0);
            nccl::broadcast(d_D[0], nbas2, 0, 0, mgr.comm_stream(0));
            cudaStreamSynchronize(mgr.comm_stream(0));
        }

        // ---- Distributed J build (same as coefficient-based path) ----
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cublasHandle_t handle = mgr.cublas(d);
            int nl = naux_local_[d];
            const double one = 1.0, zero = 0.0;
            cublasDgemv(handle, CUBLAS_OP_T, (int)nbas2, nl, &one,
                        d_B_local_[d], (int)nbas2, d_D[d], 1, &zero, d_W_local_[d], 1);
            int blk = ((int)nbas2 + threads - 1) / threads;
            distributed_J_accumulate_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(
                d_J_local_[d], d_B_local_[d], d_W_local_[d], nbas, nl);
        }
        mgr.sync_all();
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            nccl::all_reduce(d_J_local_[d], d_J_local_[d], nbas2, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();

        // ---- Distributed K build (density-matrix based) ----
        // T_P[μν] = Σ_λ D^T[μλ] × B_P[λν],  K_local = Σ_P T_P^T × B_P, then AllReduce
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cublasHandle_t handle = mgr.cublas(d);
            cublasSetStream(handle, mgr.compute_stream(d));
            int nl = naux_local_[d];
            const double one = 1.0, zero = 0.0;
            cudaMemset(d_K_local_[d], 0, nbas2 * sizeof(double));

            // Allocate T and V temporaries
            real_t *d_T = nullptr, *d_V = nullptr;
            cudaMalloc(&d_T, (size_t)nl * nbas2 * sizeof(double));
            cudaMalloc(&d_V, (size_t)nl * nbas2 * sizeof(double));

            // T_P = D^T × B_P  (batched: naux_local batches of [nbas×nbas] × [nbas×nbas])
            cublasDgemmStridedBatched(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                nbas, nbas, nbas, &one,
                d_D[d], nbas, 0LL,
                d_B_local_[d], nbas, (long long)nbas2,
                &zero, d_T, nbas, (long long)nbas2, nl);

            // V_P = T_P^T × B_P
            cublasDgemmStridedBatched(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                nbas, nbas, nbas, &one,
                d_T, nbas, (long long)nbas2,
                d_B_local_[d], nbas, (long long)nbas2,
                &zero, d_V, nbas, (long long)nbas2, nl);

            // K_local = Σ_P V_P  (= V × ones, using DGEMV on [nbas² × nl] col-major)
            {
                real_t* d_ones = nullptr;
                cudaMalloc(&d_ones, nl * sizeof(double));
                // Fill with 1.0
                std::vector<double> ones(nl, 1.0);
                cudaMemcpy(d_ones, ones.data(), nl * sizeof(double), cudaMemcpyHostToDevice);
                cublasDgemv(handle, CUBLAS_OP_N, (int)nbas2, nl, &one,
                            d_V, (int)nbas2, d_ones, 1, &zero, d_K_local_[d], 1);
                cudaFree(d_ones);
            }
            cudaFree(d_T); cudaFree(d_V);
        }
        mgr.sync_all();
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            nccl::all_reduce(d_K_local_[d], d_K_local_[d], nbas2, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();

        // ---- Fock assembly on GPU 0 ----
        {
            MultiGpuManager::DeviceGuard guard(0);
            cudaStreamSynchronize(mgr.comm_stream(0));
            int blk = ((int)nbas2 + threads - 1) / threads;
            distributed_fock_assemble_kernel<<<blk, threads>>>(
                d_H_gpu0, d_J_local_[0], d_K_local_[0], rhf_.get_fock_matrix().device_ptr(), nbas);
            cudaDeviceSynchronize();
        }
        for (int d = 1; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cudaFree(d_D[d]);
        }
        return;
    }

    const int nbas = num_basis_;
    const int nocc = num_occ_;
    const size_t nbas2 = (size_t)nbas * nbas;
    const int threads = 256;
    auto& mgr = MultiGpuManager::instance();

    const real_t* d_D_gpu0 = rhf_.get_density_matrix().device_ptr();
    const real_t* d_C_gpu0 = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_H_gpu0 = rhf_.get_core_hamiltonian_matrix().device_ptr();

    // Replicate D and C to all GPUs
    std::vector<real_t*> d_D(num_gpus_, nullptr);
    std::vector<real_t*> d_C(num_gpus_, nullptr);
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d == 0) {
            d_D[d] = const_cast<real_t*>(d_D_gpu0);
            d_C[d] = const_cast<real_t*>(d_C_gpu0);
        } else {
            cudaMalloc(&d_D[d], nbas2 * sizeof(double));
            cudaMalloc(&d_C[d], nbas2 * sizeof(double));
            cudaMemcpy(d_D[d], d_D_gpu0, nbas2 * sizeof(double), cudaMemcpyDefault);
            cudaMemcpy(d_C[d], d_C_gpu0, nbas2 * sizeof(double), cudaMemcpyDefault);
        }
    }

    // MPI: contract every rank's aux slab against rank 0's D and C, so the
    // AllReduce over rank slabs reconstructs the correct global J (from D) and
    // K (from C). Broadcast rank 0's copies into every rank.
    if (mgr.is_mpi()) {
        MultiGpuManager::DeviceGuard guard(0);
        nccl::broadcast(d_D[0], nbas2, 0, 0, mgr.comm_stream(0));
        nccl::broadcast(d_C[0], nbas2, 0, 0, mgr.comm_stream(0));
        cudaStreamSynchronize(mgr.comm_stream(0));
    }

    // ---- J build ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t handle = mgr.cublas(d);
        int nl = naux_local_[d];
        const double one = 1.0, zero = 0.0;

        // W_local[P] = Σ_{μν} B_local[P,μν] × D[μν]
        // B_local as [nbas² × nl] col-major: cublasDgemv(OP_T, nbas², nl, ...)
        cublasDgemv(handle, CUBLAS_OP_T,
                    (int)nbas2, nl,
                    &one, d_B_local_[d], (int)nbas2,
                    d_D[d], 1,
                    &zero, d_W_local_[d], 1);

        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_J_accumulate_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(
            d_J_local_[d], d_B_local_[d], d_W_local_[d], nbas, nl);
    }
    mgr.sync_all();

    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(d_J_local_[d], d_J_local_[d], nbas2,
                         ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();

    // ---- K build ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t handle = mgr.cublas(d);
        cublasSetStream(handle, mgr.compute_stream(d));
        int nl = naux_local_[d];
        const double one = 1.0, zero = 0.0, two = 2.0;

        cudaMemset(d_K_local_[d], 0, nbas2 * sizeof(double));
        cublasDgemmStridedBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            nocc, nbas, nbas,
            &one,
            d_C[d], nbas, 0LL,
            d_B_local_[d], nbas, (long long)nbas2,
            &zero,
            d_X_local_[d], nocc, (long long)(nbas * nocc),
            nl);

        int total_xpack = nl * nbas * nocc;
        int blk = (total_xpack + threads - 1) / threads;
        distributed_pack_X_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(
            d_X_local_[d], d_X_packed_local_[d], nbas, nl, nocc);

        size_t nl_nocc = (size_t)nl * nocc;
        cublasDgemm(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            nbas, nbas, (int)nl_nocc,
            &two,
            d_X_packed_local_[d], (int)nl_nocc,
            d_X_packed_local_[d], (int)nl_nocc,
            &zero,
            d_K_local_[d], nbas);
    }
    mgr.sync_all();

    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(d_K_local_[d], d_K_local_[d], nbas2,
                         ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();

    // ---- Fock assembly on GPU 0 ----
    {
        MultiGpuManager::DeviceGuard guard(0);
        cudaStreamSynchronize(mgr.comm_stream(0));
        real_t* d_F = rhf_.get_fock_matrix().device_ptr();
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_fock_assemble_kernel<<<blk, threads>>>(
            d_H_gpu0, d_J_local_[0], d_K_local_[0], d_F, nbas);
        cudaDeviceSynchronize();
    }

    for (int d = 1; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_D[d]);
        cudaFree(d_C[d]);
    }
}

// ============================================================
//  Out-of-core B: build partitions to pinned host memory (once).
//  If full B fits on GPU, keep it there instead (no transfer overhead).
// ============================================================
void ERI_RI_Distributed_RHF::free_host_partitions() {
    for (auto& p : host_partitions_) {
        if (p.h_B) { cudaFreeHost(p.h_B); p.h_B = nullptr; }
    }
    host_partitions_.clear();
    out_of_core_ready_ = false;
}

void ERI_RI_Distributed_RHF::build_out_of_core_B() {
    if (out_of_core_ready_) return;

    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const size_t nbas2 = (size_t)nbas * nbas;
    auto& mgr = MultiGpuManager::instance();

    // Phase 2c spherical (mirrors distributed_build_B / chunked_fock_build):
    // Cart 3c2e per chunk → orbital Cart→Sph → fold-M DGEMM → Sph B partitions.
    const bool use_spherical = hf_.get_use_spherical();
    const int  nc_orb = use_spherical ? hf_.get_num_basis_cart()  : nbas;
    const int  nc_aux = use_spherical ? num_auxiliary_basis_cart_ : naux;
    const size_t nc_orb2 = (size_t)nc_orb * nc_orb;

    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();

    replicate_data_to_gpus();

    // Determine partition size: fit B_partition + 3c_chunk + workspace on GPU
    size_t free_mem = 0, total_mem = 0;
    { cudaSetDevice(0); cudaMemGetInfo(&free_mem, &total_mem); }
    const size_t reserved = 512ULL * 1024 * 1024;
    const size_t chunk_3c_bytes = (size_t)max_nfunc_chunk_ * nbas2 * sizeof(real_t);
    const size_t available = (free_mem > reserved + chunk_3c_bytes) ? free_mem - reserved - chunk_3c_bytes : free_mem / 3;
    int max_rows = std::max(1, (int)(available / (nbas2 * sizeof(real_t))));
    max_rows = std::min(max_rows, naux);

    // Check if full B fits on GPU (no out-of-core needed)
    const bool fits_on_gpu = ((size_t)naux * nbas2 * sizeof(real_t) + chunk_3c_bytes + reserved < free_mem);

    // The fused compute_RI_IntermediateMatrixB builder is Cartesian-only; for
    // spherical, always take the chunked path below (M-folding per batch).
    if (fits_on_gpu && !use_spherical) {
        // Full B fits on GPU — use optimized compute_RI_IntermediateMatrixB (parallel streams)
        std::cout << "[RI-Dist] Full B fits on GPU (" << naux * nbas2 * sizeof(real_t) / (1024*1024)
                  << " MB). Building with optimized path." << std::endl;

        real_t* d_B_full = nullptr;
        tracked_cudaMalloc(&d_B_full, (size_t)naux * nbas2 * sizeof(real_t));
        cudaMemset(d_B_full, 0, (size_t)naux * nbas2 * sizeof(real_t));

        gpu::compute_RI_IntermediateMatrixB(
            shell_type_infos, shell_pair_type_infos,
            hf_.get_primitive_shells().device_ptr(),
            hf_.get_cgto_normalization_factors().device_ptr(),
            auxiliary_shell_type_infos_,
            auxiliary_primitive_shells_.device_ptr(),
            auxiliary_cgto_normalization_factors_.device_ptr(),
            d_B_full, d_persistent_shell_pair_indices_,
            schwarz_upper_bound_factors.device_ptr(),
            auxiliary_schwarz_upper_bound_factors.device_ptr(),
            schwarz_screening_threshold,
            nbas, naux,
            hf_.get_boys_grid().device_ptr(), false);

        d_B_local_[0] = d_B_full;
        naux_local_[0] = naux;
        P_start_[0] = 0;
        host_partitions_.resize(1);
        host_partitions_[0] = {nullptr, 0, naux};
        out_of_core_ready_ = true;
        std::cout << "[RI-Dist] Out-of-core B ready (GPU-resident)" << std::endl;
        return;
    }

    const int n_partitions = (naux + max_rows - 1) / max_rows;
    std::cout << "[RI-Dist] Out-of-core B: " << n_partitions << " partition(s), "
              << max_rows << " rows each" << std::endl;

    // Allocate host partitions
    host_partitions_.resize(n_partitions);
    for (int k = 0; k < n_partitions; k++) {
        const int P_start = k * max_rows;
        const int P_count = std::min(max_rows, naux - P_start);
        host_partitions_[k].P_start = P_start;
        host_partitions_[k].nrows = P_count;
        if (fits_on_gpu) {
            host_partitions_[k].h_B = nullptr;  // will use GPU-resident B
        } else {
            cudaMallocHost(&host_partitions_[k].h_B, (size_t)P_count * nbas2 * sizeof(real_t));
        }
    }

    // Build each partition: 3c2e + L⁻¹ DGEMM on GPU 0, then copy to host
    cudaSetDevice(0);
    auto& g = per_gpu_data_[0];

    // GPU buffers (reused per partition)
    real_t* d_B_row = nullptr;
    real_t* d_3c_chunk = nullptr;
    if (fits_on_gpu) {
        // Allocate full B on GPU
        cudaMalloc(&d_B_row, (size_t)naux * nbas2 * sizeof(real_t));
        cudaMemset(d_B_row, 0, (size_t)naux * nbas2 * sizeof(real_t));
    } else {
        cudaMalloc(&d_B_row, (size_t)max_rows * nbas2 * sizeof(real_t));
    }
    cudaMalloc(&d_3c_chunk, (size_t)max_nfunc_chunk_ * nbas2 * sizeof(real_t));

    // Use default stream — operations are ordered, no explicit sync between batches needed
    cublasHandle_t h0 = gpu::GPUHandle::cublas();

    for (int k = 0; k < n_partitions; k++) {
        const int P_start = host_partitions_[k].P_start;
        const int P_count = host_partitions_[k].nrows;
        real_t* d_B_dest = fits_on_gpu ? (d_B_row + (size_t)P_start * nbas2) : d_B_row;

        if (!fits_on_gpu)
            cudaMemset(d_B_dest, 0, (size_t)P_count * nbas2 * sizeof(real_t));

        for (const auto& batch : aux_batches_) {
            const double one_val = 1.0;
            if (!use_spherical) {
                cudaMemset(d_3c_chunk, 0, (size_t)batch.nfunc * nbas2 * sizeof(real_t));
                gpu::computeThreeCenterERIs_for_aux_type(
                    shell_type_infos, shell_pair_type_infos,
                    g.d_pshells, g.d_cgto_norms,
                    batch.shell_info, batch.angular_momentum,
                    g.d_aux_pshells, g.d_aux_cgto_norms,
                    d_3c_chunk, g.d_shell_pairs,
                    nbas, naux, g.d_boys,
                    g.d_schwarz, g.d_aux_schwarz,
                    schwarz_screening_threshold,
                    batch.basis_start, batch.nfunc, 0);

                cublasDgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)nbas2, P_count, batch.nfunc,
                    &one_val,
                    d_3c_chunk, (int)nbas2,
                    &g.d_L_inv[P_start * naux + batch.basis_start], naux,
                    &one_val,
                    d_B_dest, (int)nbas2);
            } else {
                // Spherical: Cart 3c2e → orbital Cart→Sph → fold-M DGEMM.
                cudaMemset(g.d_chunk_cart, 0, (size_t)batch.nfunc * nc_orb2 * sizeof(real_t));
                gpu::computeThreeCenterERIs_for_aux_type(
                    shell_type_infos, shell_pair_type_infos,
                    g.d_pshells, g.d_cgto_norms,
                    batch.shell_info, batch.angular_momentum,
                    g.d_aux_pshells, g.d_aux_cgto_norms,
                    g.d_chunk_cart, g.d_shell_pairs,
                    nc_orb, nc_aux, g.d_boys,
                    g.d_schwarz, g.d_aux_schwarz,
                    schwarz_screening_threshold,
                    batch.basis_start, batch.nfunc, 0);

                spherical::transform_orbital_pair_cart_to_sph_device(
                    h0, g.d_chunk_cart, g.d_chunk, g.d_T_orb, g.d_U_orb,
                    batch.nfunc, nc_orb, nbas);

                cublasDgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)nbas2, P_count, batch.nfunc,
                    &one_val,
                    g.d_chunk, (int)nbas2,
                    &g.d_M[P_start * (size_t)nc_aux + batch.basis_start], nc_aux,
                    &one_val,
                    d_B_dest, (int)nbas2);
            }
        }

        // Copy partition to host (unless GPU-resident)
        if (!fits_on_gpu) {
            cudaDeviceSynchronize();
            cudaMemcpy(host_partitions_[k].h_B, d_B_dest,
                       (size_t)P_count * nbas2 * sizeof(real_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaFree(d_3c_chunk);

    if (fits_on_gpu) {
        // Keep d_B_row as the single GPU-resident partition
        // Store pointer in first partition's h_B (repurposed as device pointer flag)
        // Actually, store in d_B_local_[0] for the stored RI J/K path
        d_B_local_[0] = d_B_row;
        naux_local_[0] = naux;
        P_start_[0] = 0;
    } else {
        cudaFree(d_B_row);
    }

    out_of_core_ready_ = true;
    std::cout << "[RI-Dist] Out-of-core B ready ("
              << (fits_on_gpu ? "GPU-resident" : "host-streamed") << ")" << std::endl;
}

void ERI_RI_Distributed_RHF::out_of_core_fock_build() {
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const int nocc = num_occ_;
    const size_t nbas2 = (size_t)nbas * nbas;
    const int threads = 256;
    const bool has_C = rhf_.get_hasMatrixC();

    cudaSetDevice(0);
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    cublasSetStream(handle, 0);  // ensure cuBLAS uses default stream for synchronization with transfer

    const real_t* d_D = rhf_.get_density_matrix().device_ptr();
    const real_t* d_C = has_C ? rhf_.get_coefficient_matrix().device_ptr() : nullptr;
    const real_t* d_H = rhf_.get_core_hamiltonian_matrix().device_ptr();

    const bool fits_on_gpu = (host_partitions_[0].h_B == nullptr);
    const int n_part = (int)host_partitions_.size();

    // Allocate J/K accumulators + workspace (cached across iterations)
    if (chunked_ws_.empty()) {
        int max_part_rows = 0;
        for (const auto& p : host_partitions_)
            if (p.nrows > max_part_rows) max_part_rows = p.nrows;
        chunked_N_rows_ = max_part_rows;

        chunked_ws_.resize(1);
        auto& w = chunked_ws_[0];
        if (!d_J_local_[0]) cudaMalloc(&d_J_local_[0], nbas2 * sizeof(double));
        if (!d_K_local_[0]) cudaMalloc(&d_K_local_[0], nbas2 * sizeof(double));
        if (!fits_on_gpu)
            cudaMalloc(&w.d_B_row, (size_t)max_part_rows * nbas2 * sizeof(double));
        cudaMalloc(&w.d_W, max_part_rows * sizeof(double));
        if (has_C) {
            cudaMalloc(&w.d_X, (size_t)max_part_rows * nbas * nocc * sizeof(double));
            cudaMalloc(&w.d_Xp, (size_t)max_part_rows * nbas * nocc * sizeof(double));
        }
    }
    auto& ws = chunked_ws_[0];
    cudaMemset(d_J_local_[0], 0, nbas2 * sizeof(double));
    cudaMemset(d_K_local_[0], 0, nbas2 * sizeof(double));

    for (const auto& part : host_partitions_) {
        const int P_count = part.nrows;

        real_t* d_B;
        if (fits_on_gpu) {
            d_B = d_B_local_[0] + (size_t)part.P_start * nbas2;
        } else {
            d_B = ws.d_B_row;
            cudaMemcpy(d_B, part.h_B, (size_t)P_count * nbas2 * sizeof(double), cudaMemcpyHostToDevice);
        }

        const double one = 1.0, zero = 0.0, two = 2.0;
        cublasDgemv(handle, CUBLAS_OP_T, (int)nbas2, P_count, &one,
                    d_B, (int)nbas2, d_D, 1, &zero, ws.d_W, 1);
        cublasDgemv(handle, CUBLAS_OP_N, (int)nbas2, P_count, &one,
                    d_B, (int)nbas2, ws.d_W, 1, &one, d_J_local_[0], 1);

        if (has_C) {
            cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nocc, nbas, nbas, &one, d_C, nbas, 0LL,
                d_B, nbas, (long long)nbas2, &zero, ws.d_X, nocc, (long long)(nbas*nocc), P_count);
            int total = P_count * nbas * nocc;
            distributed_pack_X_kernel<<<(total+threads-1)/threads, threads>>>(ws.d_X, ws.d_Xp, nbas, P_count, nocc);
            size_t pc = (size_t)P_count * nocc;
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nbas, nbas, (int)pc,
                &two, ws.d_Xp, (int)pc, ws.d_Xp, (int)pc, &one, d_K_local_[0], nbas);
        } else {
            cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nbas, nbas, nbas, &one, d_D, nbas, 0LL, d_B, nbas, (long long)nbas2,
                &zero, ws.d_X, nbas, (long long)nbas2, P_count);
            cublasDgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                nbas, nbas, nbas, &one, ws.d_X, nbas, (long long)nbas2, d_B, nbas, (long long)nbas2,
                &zero, ws.d_Xp, nbas, (long long)nbas2, P_count);
            std::vector<double> ones(P_count, 1.0);
            cudaMemcpy(ws.d_W, ones.data(), P_count * sizeof(double), cudaMemcpyHostToDevice);
            cublasDgemv(handle, CUBLAS_OP_N, (int)nbas2, P_count, &one,
                        ws.d_Xp, (int)nbas2, ws.d_W, 1, &one, d_K_local_[0], 1);
        }
    }

    // ---- Fock assembly ----
    {
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_fock_assemble_kernel<<<blk, threads>>>(
            d_H, d_J_local_[0], d_K_local_[0], rhf_.get_fock_matrix().device_ptr(), nbas);
        cudaDeviceSynchronize();
    }
}

// ============================================================
//  Chunked Fock build: partition B by rows (P-index), build each
//  row-partition from 3c2e + L⁻¹ DGEMM, compute J/K contribution,
//  then discard.  J and K decompose perfectly by row partition.
//
//  Memory: B_row_chunk (N_rows × nbas²) + X workspace per chunk.
//  N_rows auto-determined from free GPU memory.
// ============================================================
void ERI_RI_Distributed_RHF::chunked_fock_build() {
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const int nocc = num_occ_;
    const size_t nbas2 = (size_t)nbas * nbas;
    const int threads = 256;
    auto& mgr = MultiGpuManager::instance();

    // Phase 2c spherical: the 3c2e kernel runs in Cartesian (nc_orb, nc_aux);
    // each chunk's orbital pair is transformed Cart→Sph and the aux Cart→Sph
    // mixing is folded into M (= d_M), so B_row rows come out Spherical.
    // Mirrors distributed_build_B().  (Cart: nc_*==n* and the M branch is unused.)
    const bool use_spherical = hf_.get_use_spherical();
    const int  nc_orb = use_spherical ? hf_.get_num_basis_cart()  : nbas;
    const int  nc_aux = use_spherical ? num_auxiliary_basis_cart_ : naux;
    const size_t nc_orb2 = (size_t)nc_orb * nc_orb;

    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const bool has_C = rhf_.get_hasMatrixC();

    // ---- One-time setup (cached across iterations) ----
    replicate_data_to_gpus();
    allocate_chunked_workspace();

    const int N_rows = chunked_N_rows_;

    // ---- Replicate D/C to all GPUs (copy into cached buffers) ----
    const real_t* d_D_gpu0 = rhf_.get_density_matrix().device_ptr();
    const real_t* d_C_gpu0 = has_C ? rhf_.get_coefficient_matrix().device_ptr() : nullptr;
    const real_t* d_H_gpu0 = rhf_.get_core_hamiltonian_matrix().device_ptr();

    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        auto& w = chunked_ws_[d];
        if (d > 0) {
            cudaMemcpy(w.d_D, d_D_gpu0, nbas2 * sizeof(double), cudaMemcpyDefault);
            if (has_C && w.d_C)
                cudaMemcpy(w.d_C, d_C_gpu0, nbas2 * sizeof(double), cudaMemcpyDefault);
        }
    }

    // D/C pointer per GPU (GPU 0 uses originals)
    std::vector<real_t*> d_D(num_gpus_), d_C(num_gpus_);
    d_D[0] = const_cast<real_t*>(d_D_gpu0);
    d_C[0] = has_C ? const_cast<real_t*>(d_C_gpu0) : nullptr;
    for (int d = 1; d < num_gpus_; d++) {
        d_D[d] = chunked_ws_[d].d_D;
        d_C[d] = chunked_ws_[d].d_C;
    }

    // ---- Zero J/K ----
    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        cudaMemsetAsync(d_J_local_[d], 0, nbas2 * sizeof(double), mgr.compute_stream(d));
        cudaMemsetAsync(d_K_local_[d], 0, nbas2 * sizeof(double), mgr.compute_stream(d));
    }

    // ---- Main loop: each GPU handles its fixed naux_local rows ----
    // Within each GPU, sub-chunk if naux_local > N_rows.
    // All GPUs run concurrently (async launch, sync at end).
    for (int d = 0; d < num_gpus_; d++) {
        cudaSetDevice(d);
        cudaStream_t cs = mgr.compute_stream(d);
        cublasHandle_t handle = mgr.cublas(d);
        cublasSetStream(handle, cs);
        auto& g = per_gpu_data_[d];
        auto& w = chunked_ws_[d];

        const int gpu_P_start = (int)P_start_[d];  // global start row for this GPU
        const int gpu_P_total = naux_local_[d];     // total rows for this GPU
        const int n_sub = (gpu_P_total + N_rows - 1) / N_rows;

        for (int sub = 0; sub < n_sub; sub++) {
            const int local_offset = sub * N_rows;
            const int P_start_global = gpu_P_start + local_offset;
            const int P_count = std::min(N_rows, gpu_P_total - local_offset);

            // Zero B_row
            cudaMemsetAsync(w.d_B_row, 0, (size_t)P_count * nbas2 * sizeof(double), cs);

            // Build B_row by accumulating over aux batches:
            for (const auto& batch : aux_batches_) {
                const size_t Q_c_start = batch.basis_start;
                const int nfunc_c = batch.nfunc;

                const double one = 1.0;
                if (!use_spherical) {
                    cudaMemsetAsync(w.d_3c_chunk, 0, (size_t)nfunc_c * nbas2 * sizeof(double), cs);
                    gpu::computeThreeCenterERIs_for_aux_type(
                        shell_type_infos, shell_pair_type_infos,
                        g.d_pshells, g.d_cgto_norms,
                        batch.shell_info, batch.angular_momentum,
                        g.d_aux_pshells, g.d_aux_cgto_norms,
                        w.d_3c_chunk, g.d_shell_pairs,
                        nbas, naux, g.d_boys,
                        g.d_schwarz, g.d_aux_schwarz,
                        schwarz_screening_threshold,
                        Q_c_start, nfunc_c, cs);

                    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)nbas2, P_count, nfunc_c,
                        &one,
                        w.d_3c_chunk, (int)nbas2,
                        &g.d_L_inv[P_start_global * naux + Q_c_start], naux,
                        &one,
                        w.d_B_row, (int)nbas2);
                } else {
                    // Spherical: Cart 3c2e → orbital Cart→Sph → fold-M DGEMM.
                    cudaMemsetAsync(g.d_chunk_cart, 0, (size_t)nfunc_c * nc_orb2 * sizeof(real_t), cs);
                    gpu::computeThreeCenterERIs_for_aux_type(
                        shell_type_infos, shell_pair_type_infos,
                        g.d_pshells, g.d_cgto_norms,
                        batch.shell_info, batch.angular_momentum,
                        g.d_aux_pshells, g.d_aux_cgto_norms,
                        g.d_chunk_cart, g.d_shell_pairs,
                        nc_orb, nc_aux, g.d_boys,
                        g.d_schwarz, g.d_aux_schwarz,
                        schwarz_screening_threshold,
                        Q_c_start, nfunc_c, cs);

                    spherical::transform_orbital_pair_cart_to_sph_device(
                        handle, g.d_chunk_cart, g.d_chunk, g.d_T_orb, g.d_U_orb,
                        nfunc_c, nc_orb, nbas);

                    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)nbas2, P_count, nfunc_c,
                        &one,
                        g.d_chunk, (int)nbas2,
                        &g.d_M[P_start_global * (size_t)nc_aux + Q_c_start], nc_aux,
                        &one,
                        w.d_B_row, (int)nbas2);
                }
            }

            // ---- J contribution ----
            const double one = 1.0, zero = 0.0, two = 2.0;
            cublasDgemv(handle, CUBLAS_OP_T,
                        (int)nbas2, P_count, &one,
                        w.d_B_row, (int)nbas2,
                        d_D[d], 1, &zero, w.d_W, 1);
            cublasDgemv(handle, CUBLAS_OP_N,
                        (int)nbas2, P_count, &one,
                        w.d_B_row, (int)nbas2,
                        w.d_W, 1, &one, d_J_local_[d], 1);

            // ---- K contribution ----
            if (has_C) {
                cublasDgemmStridedBatched(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    nocc, nbas, nbas, &one,
                    d_C[d], nbas, 0LL,
                    w.d_B_row, nbas, (long long)nbas2,
                    &zero, w.d_X, nocc, (long long)(nbas * nocc),
                    P_count);

                int total = P_count * nbas * nocc;
                int blk = (total + threads - 1) / threads;
                distributed_pack_X_kernel<<<blk, threads, 0, cs>>>(
                    w.d_X, w.d_Xp, nbas, P_count, nocc);

                size_t pc_nocc = (size_t)P_count * nocc;
                cublasDgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    nbas, nbas, (int)pc_nocc,
                    &two,
                    w.d_Xp, (int)pc_nocc,
                    w.d_Xp, (int)pc_nocc,
                    &one,
                    d_K_local_[d], nbas);
            } else {
                cublasDgemmStridedBatched(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N, nbas, nbas, nbas, &one,
                    d_D[d], nbas, 0LL,
                    w.d_B_row, nbas, (long long)nbas2,
                    &zero, w.d_X, nbas, (long long)nbas2, P_count);
                cublasDgemmStridedBatched(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N, nbas, nbas, nbas, &one,
                    w.d_X, nbas, (long long)nbas2,
                    w.d_B_row, nbas, (long long)nbas2,
                    &zero, w.d_Xp, nbas, (long long)nbas2, P_count);

                // K_local += Σ V_p via DGEMV with ones
                real_t* d_ones = w.d_W;  // reuse W buffer (P_count <= N_rows)
                std::vector<double> ones_vec(P_count, 1.0);
                cudaMemcpyAsync(d_ones, ones_vec.data(), P_count * sizeof(double), cudaMemcpyHostToDevice, cs);
                cublasDgemv(handle, CUBLAS_OP_N, (int)nbas2, P_count, &one,
                            w.d_Xp, (int)nbas2, d_ones, 1, &one, d_K_local_[d], 1);
            }
        }  // sub-chunk loop
    }  // GPU loop
    mgr.sync_all();

    // ---- AllReduce J/K across GPUs ----
    if (num_gpus_ > 1) {
        // NCCL group calls: set device before each ncclAllReduce (required by NCCL)
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            cudaSetDevice(d);
            nccl::all_reduce(d_J_local_[d], d_J_local_[d], nbas2, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            cudaSetDevice(d);
            nccl::all_reduce(d_K_local_[d], d_K_local_[d], nbas2, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();
        // Sync comm streams
        for (int d = 0; d < num_gpus_; d++) {
            cudaSetDevice(d);
            cudaStreamSynchronize(mgr.comm_stream(d));
        }
    }

    // ---- Fock assembly on GPU 0 ----
    {
        cudaSetDevice(0);
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_fock_assemble_kernel<<<blk, threads>>>(
            d_H_gpu0, d_J_local_[0], d_K_local_[0], rhf_.get_fock_matrix().device_ptr(), nbas);
        cudaDeviceSynchronize();
    }

    // Check for async CUDA errors
    cudaSetDevice(0);
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "[RI-Dist] CUDA error after chunked Fock: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

// ============================================================
//  Semi-Direct RI Distributed
// ============================================================

// Forward declaration from gpu_manager.cu
namespace gpu {
    void computeFockMatrix_RI_Direct_v2(
        const real_t* d_density_matrix, const real_t* d_coefficient_matrix,
        const real_t* d_two_center_eris_cholesky, const real_t* d_L_inv,
        const real_t* d_core_hamiltonian_matrix, real_t* d_fock_matrix,
        const std::vector<ShellTypeInfo>& shell_type_infos,
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
        const PrimitiveShell* d_primitive_shells, const real_t* d_cgto_normalization_factors,
        const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
        const PrimitiveShell* d_auxiliary_primitive_shells, const real_t* d_auxiliary_cgto_normalization_factors,
        const size_t2* d_primitive_shell_pair_indices,
        int num_basis, int num_auxiliary_basis, int num_occ,
        const real_t* d_boys_grid,
        double schwarz_screening_threshold,
        const real_t* d_schwarz_upper_bound_factors, const real_t* d_auxiliary_schwarz_upper_bound_factors,
        bool verbose);
}

ERI_RI_SemiDirect_Distributed_RHF::ERI_RI_SemiDirect_Distributed_RHF(
    RHF& rhf, const Molecular& auxiliary_molecular)
    : ERI_RI_SemiDirect_RHF(rhf, auxiliary_molecular)
{
    auto& mgr = MultiGpuManager::instance();
    num_gpus_ = mgr.num_devices();
    naux_local_.resize(num_gpus_);
    P_start_.resize(num_gpus_);
    for (int d = 0; d < num_gpus_; d++) {
        auto [start, end] = aux_partition(num_auxiliary_basis_, num_gpus_, d);
        P_start_[d] = start;
        naux_local_[d] = (int)(end - start);
    }
    std::cout << "[Semi-Direct-RI-Dist] " << num_gpus_ << " GPUs, naux=" << num_auxiliary_basis_ << std::endl;
}

ERI_RI_SemiDirect_Distributed_RHF::~ERI_RI_SemiDirect_Distributed_RHF() = default;

void ERI_RI_SemiDirect_Distributed_RHF::compute_fock_matrix() {
    // Before coefficient matrix is available: use precomputed B for density-matrix Fock
    if (!rhf_.get_hasMatrixC()) {
        cudaSetDevice(0);
        const size_t nbas2 = (size_t)num_basis_ * num_basis_;
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();

        // Allocate temporary workspace
        real_t *dJ, *dK, *dW, *dT, *dV;
        tracked_cudaMalloc(&dJ, nbas2 * sizeof(real_t));
        tracked_cudaMalloc(&dK, nbas2 * sizeof(real_t));
        tracked_cudaMalloc(&dW, num_auxiliary_basis_ * sizeof(real_t));
        tracked_cudaMalloc(&dT, nbas2 * sizeof(real_t));
        tracked_cudaMalloc(&dV, nbas2 * sizeof(real_t));

        gpu::computeFockMatrix_RI_RHF_with_density_matrix(
            density_matrix.device_ptr(), core_hamiltonian_matrix.device_ptr(),
            intermediate_matrix_B_cpu_.device_ptr(), fock_matrix.device_ptr(),
            num_basis_, num_auxiliary_basis_, dJ, dK, dW, dT, dV);

        tracked_cudaFree(dJ); tracked_cudaFree(dK);
        tracked_cudaFree(dW); tracked_cudaFree(dT); tracked_cudaFree(dV);
        return;
    }

    // Strategy: GPU 0 runs existing computeFockMatrix_RI_Direct_v2 (builds temp B, computes J/K).
    // This works because Semi-Direct allocates/frees B each iteration.
    // For multi-GPU: after GPU 0 computes the full B internally, we intercept
    // and use the distributed J/K build.
    //
    // Simpler approach for now: just call the single-GPU path on GPU 0.
    // The multi-GPU benefit comes from future optimization where B is distributed.
    // For now, this acts as a correct baseline that validates the factory wiring.

    // Actually, let's do the proper distribution:
    // 1. GPU 0: compute 3c2e → B (using existing code)
    // 2. Scatter B_local to all GPUs
    // 3. Distributed J/K + AllReduce

    const int nbas = num_basis_;
    const int naux = num_auxiliary_basis_;
    const int nocc = rhf_.get_num_electrons() / 2;
    const size_t nbas2 = (size_t)nbas * nbas;
    const int threads = 256;
    auto& mgr = MultiGpuManager::instance();

    // Step 1: Compute full B on GPU 0 using existing Semi-Direct path
    real_t* d_B_full = nullptr;
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaMalloc(&d_B_full, (size_t)naux * nbas2 * sizeof(real_t));
        cudaMemset(d_B_full, 0, (size_t)naux * nbas2 * sizeof(real_t));

        // Compute 3c2e using existing function
        gpu::computeThreeCenterERIs(
            hf_.get_shell_type_infos(), hf_.get_shell_pair_type_infos(),
            hf_.get_primitive_shells().device_ptr(),
            hf_.get_cgto_normalization_factors().device_ptr(),
            auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
            auxiliary_cgto_normalization_factors_.device_ptr(),
            d_B_full, primitive_shell_pair_indices.device_ptr(),
            nbas, naux, hf_.get_boys_grid().device_ptr(),
            schwarz_upper_bound_factors.device_ptr(),
            auxiliary_schwarz_upper_bound_factors.device_ptr(),
            hf_.get_schwarz_screening_threshold(), false);

        // B = L⁻¹ × 3c2e via trsm
        cublasHandle_t h0 = gpu::GPUHandle::cublas();
        const double one = 1.0;
        cublasDtrsm(h0, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                    (int)nbas2, naux, &one, two_center_eris.device_ptr(), naux, d_B_full, (int)nbas2);
    }

    // Step 2: Scatter B_local to all GPUs
    std::vector<real_t*> d_B_local(num_gpus_, nullptr);
    for (int d = 0; d < num_gpus_; d++) {
        size_t local_size = (size_t)naux_local_[d] * nbas2;
        size_t offset = P_start_[d] * nbas2;
        MultiGpuManager::DeviceGuard guard(d);
        cudaMalloc(&d_B_local[d], local_size * sizeof(double));
        cudaMemcpy(d_B_local[d], d_B_full + offset, local_size * sizeof(double), cudaMemcpyDefault);
    }
    { MultiGpuManager::DeviceGuard guard(0); tracked_cudaFree(d_B_full); }

    // Step 3: Replicate D, C
    const real_t* d_D0 = rhf_.get_density_matrix().device_ptr();
    const real_t* d_C0 = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_H0 = rhf_.get_core_hamiltonian_matrix().device_ptr();
    std::vector<real_t*> d_D(num_gpus_), d_C(num_gpus_);
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d == 0) { d_D[d] = const_cast<real_t*>(d_D0); d_C[d] = const_cast<real_t*>(d_C0); }
        else {
            cudaMalloc(&d_D[d], nbas2 * sizeof(double));
            cudaMalloc(&d_C[d], nbas2 * sizeof(double));
            cudaMemcpy(d_D[d], d_D0, nbas2 * sizeof(double), cudaMemcpyDefault);
            cudaMemcpy(d_C[d], d_C0, nbas2 * sizeof(double), cudaMemcpyDefault);
        }
    }

    // Step 4: Distributed J build
    std::vector<real_t*> d_J_local(num_gpus_), d_K_local(num_gpus_), d_W_local(num_gpus_);
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        int nl = naux_local_[d];
        cudaMalloc(&d_W_local[d], nl * sizeof(double));
        cudaMalloc(&d_J_local[d], nbas2 * sizeof(double));
        cudaMalloc(&d_K_local[d], nbas2 * sizeof(double));
        cublasHandle_t handle = mgr.cublas(d);
        cublasSetStream(handle, mgr.compute_stream(d));
        const double one = 1.0, zero = 0.0;
        cublasDgemv(handle, CUBLAS_OP_T, (int)nbas2, nl, &one, d_B_local[d], (int)nbas2, d_D[d], 1, &zero, d_W_local[d], 1);
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_J_accumulate_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(d_J_local[d], d_B_local[d], d_W_local[d], nbas, nl);
    }
    mgr.sync_all();
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) { MultiGpuManager::DeviceGuard guard(d); nccl::all_reduce(d_J_local[d], d_J_local[d], nbas2, ncclSum, d, mgr.comm_stream(d)); }
    nccl::group_end();
    mgr.sync_all();

    // Step 5: Distributed K build
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t handle = mgr.cublas(d);
        cublasSetStream(handle, mgr.compute_stream(d));
        int nl = naux_local_[d];
        const double one = 1.0, zero = 0.0, two = 2.0;
        cudaMemset(d_K_local[d], 0, nbas2 * sizeof(double));
        real_t* d_X = nullptr; real_t* d_Xp = nullptr;
        cudaMalloc(&d_X, (size_t)nl * nbas * nocc * sizeof(double));
        cudaMalloc(&d_Xp, (size_t)nl * nbas * nocc * sizeof(double));
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, nocc, nbas, nbas, &one,
            d_C[d], nbas, 0LL, d_B_local[d], nbas, (long long)nbas2, &zero, d_X, nocc, (long long)(nbas*nocc), nl);
        int total = nl * nbas * nocc;
        distributed_pack_X_kernel<<<(total+threads-1)/threads, threads, 0, mgr.compute_stream(d)>>>(d_X, d_Xp, nbas, nl, nocc);
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, nbas, nbas, nl*nocc, &two, d_Xp, nl*nocc, d_Xp, nl*nocc, &zero, d_K_local[d], nbas);
        cudaFree(d_X); cudaFree(d_Xp);
    }
    mgr.sync_all();
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) { MultiGpuManager::DeviceGuard guard(d); nccl::all_reduce(d_K_local[d], d_K_local[d], nbas2, ncclSum, d, mgr.comm_stream(d)); }
    nccl::group_end();
    mgr.sync_all();

    // Step 6: Fock assembly on GPU 0
    { MultiGpuManager::DeviceGuard guard(0);
      int blk = ((int)nbas2 + threads - 1) / threads;
      distributed_fock_assemble_kernel<<<blk, threads>>>(d_H0, d_J_local[0], d_K_local[0], rhf_.get_fock_matrix().device_ptr(), nbas);
      cudaDeviceSynchronize(); }

    // Cleanup
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_B_local[d]); cudaFree(d_W_local[d]); cudaFree(d_J_local[d]); cudaFree(d_K_local[d]);
        if (d > 0) { cudaFree(d_D[d]); cudaFree(d_C[d]); }
    }
}

// ============================================================
//  Distributed RI-ADC(2) Schur complement
// ============================================================

void ERI_RI_Distributed_RHF::compute_sos_adc2(int n_states) {
    const int nocc = rhf_.get_num_electrons() / 2 - rhf_.get_num_frozen_core();
    const int nvir = rhf_.get_num_basis() - rhf_.get_num_electrons() / 2;
    const int singles_dim = nocc * nvir;

    std::cout << "\n---- Distributed RI-ADC(2) Schur ----"
              << " nocc=" << nocc << ", nvir=" << nvir
              << ", num_gpus=" << num_gpus_
              << ", singles=" << singles_dim
              << ", nstates=" << n_states << std::endl;

    if (rhf_.get_num_frozen_core() != 0) {
        std::cerr << "[Distributed ADC(2)] Error: frozen core not yet supported." << std::endl;
        return;
    }

    auto& mgr = MultiGpuManager::instance();
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();  // on GPU 0
    const int nbas = num_basis_;

    // --- Step 1: Build MO-transformed B blocks on each GPU ---
    Timer setup_timer;
    std::vector<real_t*> d_B_ia_local(num_gpus_, nullptr);
    std::vector<real_t*> d_B_ab_local(num_gpus_, nullptr);
    std::vector<real_t*> d_B_ij_local(num_gpus_, nullptr);

    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);

        // Need C on this GPU
        real_t* d_C_dev = d_C;
        if (d > 0) {
            cudaMalloc(&d_C_dev, (size_t)nbas * nbas * sizeof(real_t));
            cudaMemcpyPeer(d_C_dev, d, d_C, 0, (size_t)nbas * nbas * sizeof(real_t));
        }

        d_B_ia_local[d] = build_B_ia_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);
        d_B_ab_local[d] = build_B_ab_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);
        d_B_ij_local[d] = build_B_ij_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);

        if (d > 0) cudaFree(d_C_dev);
    }
    std::cout << "  B-block build time: " << std::fixed << std::setprecision(3)
              << setup_timer.elapsed_seconds() << " s" << std::endl;

    // --- Step 2: Build M11 (distributed) ---
    Timer m11_timer;
    real_t* d_M11 = nullptr;
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaMalloc(&d_M11, (size_t)singles_dim * singles_dim * sizeof(real_t));
    }
    RIADC2SchurDistributedOperator::build_M11_distributed(
        d_M11, num_gpus_,
        d_B_ia_local, d_B_ab_local, d_B_ij_local,
        naux_local_,
        rhf_.get_orbital_energies().device_ptr(),
        nocc, nvir);
    std::cout << "  M11 build time: " << std::fixed << std::setprecision(3)
              << m11_timer.elapsed_seconds() << " s" << std::endl;

    // --- Step 3: Davidson + omega iteration ---
    std::vector<double> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_states * singles_dim, 0.0);
    {
        Timer dav_timer;
        std::cout << "\n  --- Distributed RI-ADC(2) Schur Davidson ---" << std::endl;

        RIADC2SchurDistributedOperator op(
            num_gpus_,
            d_B_ia_local, d_B_ab_local, d_B_ij_local,
            naux_local_, d_M11,
            rhf_.get_orbital_energies().device_ptr(),
            nocc, nvir);

        DavidsonConfig dav_config;
        dav_config.num_eigenvalues = n_states;
        dav_config.convergence_threshold = 1e-6;
        dav_config.max_subspace_size = std::min(singles_dim, std::max(30, 4 * n_states));
        dav_config.max_iterations = 200;
        dav_config.use_preconditioner = true;
        dav_config.symmetric = true;
        dav_config.verbose = 1;

        // Initial solve (omega=0)
        op.set_omega(0.0);
        {
            DavidsonSolver solver(op, dav_config);
            solver.solve();
            eigenvalues.resize(n_states);
            const auto& evals = solver.get_eigenvalues();
            for (int k = 0; k < n_states && k < (int)evals.size(); k++)
                eigenvalues[k] = evals[k];
        }

        std::cout << "  [Dist] Initial (omega=0):";
        for (int k = 0; k < n_states; k++)
            std::cout << " " << std::fixed << std::setprecision(6) << eigenvalues[k];
        std::cout << std::endl;

        // Per-root omega iteration
        const double omega_thr = 1e-8;
        const int max_omega_iter = 15;

        for (int root = 0; root < n_states; root++) {
            double omega = eigenvalues[root];
            bool converged = false;

            for (int iter = 0; iter < max_omega_iter; iter++) {
                op.set_omega(omega);
                DavidsonSolver solver(op, dav_config);
                solver.solve();

                const auto& evals = solver.get_eigenvalues();
                double omega_new = (root < (int)evals.size()) ? evals[root] : omega;
                double delta = std::abs(omega_new - omega);

                std::cout << "  [Dist] Root " << root + 1 << " iter " << std::setw(2) << iter + 1
                          << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                          << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                          << std::defaultfloat << std::endl;

                if (delta < omega_thr) {
                    eigenvalues[root] = omega_new;
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
                    converged = true;
                    std::cout << "  [Dist] Root " << root + 1 << ": converged in "
                              << iter + 1 << " iterations" << std::endl;
                    break;
                }
                omega = omega_new;
                if (iter == max_omega_iter - 1)
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
            }
            if (!converged) {
                eigenvalues[root] = omega;
                std::cout << "  [Dist] Root " << root + 1 << ": NOT converged after "
                          << max_omega_iter << " iterations" << std::endl;
            }
        }

        std::cout << "  [Dist] Davidson time: " << std::fixed << std::setprecision(3)
                  << dav_timer.elapsed_seconds() << " s" << std::endl;
    }

    // --- Step 4: Excited state properties ---
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_M11);
    }

    rhf_.set_excitation_energies(eigenvalues);

    rhf_.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "Distributed-RI-ADC(2)",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        rhf_.get_coefficient_matrix().host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, rhf_.get_num_basis(), nocc, nvir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);

    // Cleanup MO-transformed B blocks
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_B_ia_local[d]);
        cudaFree(d_B_ab_local[d]);
        cudaFree(d_B_ij_local[d]);
    }
}

// ============================================================
//  Distributed SOS-Laplace-ADC(2) — O(N⁴) with Laplace-point parallelism
// ============================================================

void ERI_RI_Distributed_RHF::compute_sos_laplace_adc2(int n_states) {
    const int nocc = rhf_.get_num_electrons() / 2 - rhf_.get_num_frozen_core();
    const int nvir = rhf_.get_num_basis() - rhf_.get_num_electrons() / 2;
    const int singles_dim = nocc * nvir;
    const int naux_total = num_auxiliary_basis_;

    std::cout << "\n---- Distributed SOS-Laplace-ADC(2) ----"
              << " nocc=" << nocc << ", nvir=" << nvir
              << ", naux=" << naux_total
              << ", num_gpus=" << num_gpus_
              << ", singles=" << singles_dim
              << ", nstates=" << n_states << std::endl;

    if (rhf_.get_num_frozen_core() != 0) {
        std::cerr << "[SOS-Laplace-ADC(2)] Error: frozen core not yet supported." << std::endl;
        return;
    }

    auto& mgr = MultiGpuManager::instance();
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const int nbas = num_basis_;

    // --- Step 1: Build B_ia on each GPU ---
    Timer setup_timer;
    std::vector<real_t*> d_B_ia_local(num_gpus_, nullptr);
    // Also build B_ab, B_ij for M11 (then free them)
    std::vector<real_t*> d_B_ab_local(num_gpus_, nullptr);
    std::vector<real_t*> d_B_ij_local(num_gpus_, nullptr);

    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        real_t* d_C_dev = d_C;
        if (d > 0) {
            cudaMalloc(&d_C_dev, (size_t)nbas * nbas * sizeof(real_t));
            cudaMemcpyPeer(d_C_dev, d, d_C, 0, (size_t)nbas * nbas * sizeof(real_t));
        }
        d_B_ia_local[d] = build_B_ia_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);
        d_B_ab_local[d] = build_B_ab_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);
        d_B_ij_local[d] = build_B_ij_local(d_B_local_[d], naux_local_[d],
                                           d_C_dev, nbas, nocc, nvir);
        if (d > 0) cudaFree(d_C_dev);
    }
    std::cout << "  B-block build time: " << std::fixed << std::setprecision(3)
              << setup_timer.elapsed_seconds() << " s" << std::endl;

    // --- Step 2: Build M11 distributed ---
    Timer m11_timer;
    real_t* d_M11 = nullptr;
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaMalloc(&d_M11, (size_t)singles_dim * singles_dim * sizeof(real_t));
    }
    RIADC2SchurDistributedOperator::build_M11_distributed(
        d_M11, num_gpus_,
        d_B_ia_local, d_B_ab_local, d_B_ij_local,
        naux_local_,
        rhf_.get_orbital_energies().device_ptr(),
        nocc, nvir);
    std::cout << "  M11 build time: " << std::fixed << std::setprecision(3)
              << m11_timer.elapsed_seconds() << " s" << std::endl;

    // AllGather B_ab_local → B_ab_full on GPU 0 for A3-Coulomb
    real_t* d_B_ab_full = nullptr;
    {
        const int vv = nvir * nvir;
        std::vector<size_t> P_start(num_gpus_ + 1, 0);
        for (int d = 0; d < num_gpus_; d++)
            P_start[d + 1] = P_start[d] + naux_local_[d];

        MultiGpuManager::DeviceGuard guard(0);
        cudaMalloc(&d_B_ab_full, (size_t)vv * naux_total * sizeof(real_t));
        for (int d_src = 0; d_src < num_gpus_; d_src++) {
            size_t offset = P_start[d_src] * vv;
            size_t bytes = (size_t)naux_local_[d_src] * vv * sizeof(real_t);
            if (d_src == 0)
                cudaMemcpy(d_B_ab_full + offset, d_B_ab_local[d_src],
                           bytes, cudaMemcpyDeviceToDevice);
            else
                cudaMemcpyPeer(d_B_ab_full + offset, 0,
                               d_B_ab_local[d_src], d_src, bytes);
        }
    }
    // Free B_ab_local
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_B_ab_local[d]);
    }

    // AllGather B_ij_local → B_ij_full on GPU 0 for B3-exchange
    real_t* d_B_ij_full = nullptr;
    {
        const int oo = nocc * nocc;
        std::vector<size_t> P_start(num_gpus_ + 1, 0);
        for (int d = 0; d < num_gpus_; d++)
            P_start[d + 1] = P_start[d] + naux_local_[d];

        MultiGpuManager::DeviceGuard guard(0);
        cudaMalloc(&d_B_ij_full, (size_t)oo * naux_total * sizeof(real_t));
        for (int d_src = 0; d_src < num_gpus_; d_src++) {
            size_t offset = P_start[d_src] * oo;
            size_t bytes = (size_t)naux_local_[d_src] * oo * sizeof(real_t);
            if (d_src == 0)
                cudaMemcpy(d_B_ij_full + offset, d_B_ij_local[d_src],
                           bytes, cudaMemcpyDeviceToDevice);
            else
                cudaMemcpyPeer(d_B_ij_full + offset, 0,
                               d_B_ij_local[d_src], d_src, bytes);
        }
    }
    // Free B_ij_local
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_B_ij_local[d]);
    }

    // --- Step 3: Davidson + omega iteration ---
    std::vector<double> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_states * singles_dim, 0.0);
    {
        Timer dav_timer;
        std::cout << "\n  --- Distributed SOS-Laplace-ADC(2) Davidson ---" << std::endl;

        const double c_c = rhf_.get_adc_c_c();
        SOSLaplaceADC2DistributedOperator op(
            num_gpus_, d_B_ia_local, naux_local_,
            d_B_ij_full, d_B_ab_full, d_M11,
            rhf_.get_orbital_energies().device_ptr(),
            nocc, nvir, naux_total, c_c);

        DavidsonConfig dav_config;
        dav_config.num_eigenvalues = n_states;
        dav_config.convergence_threshold = 1e-6;
        dav_config.max_subspace_size = std::min(singles_dim, std::max(30, 4 * n_states));
        dav_config.max_iterations = 200;
        dav_config.use_preconditioner = true;
        dav_config.symmetric = true;
        dav_config.verbose = 1;

        op.set_omega(0.0);
        op.update_laplace_quadrature();
        {
            DavidsonSolver solver(op, dav_config);
            solver.solve();
            eigenvalues.resize(n_states);
            const auto& evals = solver.get_eigenvalues();
            for (int k = 0; k < n_states && k < (int)evals.size(); k++)
                eigenvalues[k] = evals[k];
        }

        std::cout << "  [SOS-LT] Initial (omega=0):";
        for (int k = 0; k < n_states; k++)
            std::cout << " " << std::fixed << std::setprecision(6) << eigenvalues[k];
        std::cout << std::endl;

        const double omega_thr = 1e-8;
        const int max_omega_iter = 15;

        for (int root = 0; root < n_states; root++) {
            double omega = eigenvalues[root];
            bool converged = false;
            for (int iter = 0; iter < max_omega_iter; iter++) {
                op.set_omega(omega);
                op.update_laplace_quadrature();
                DavidsonSolver solver(op, dav_config);
                solver.solve();
                const auto& evals = solver.get_eigenvalues();
                double omega_new = (root < (int)evals.size()) ? evals[root] : omega;
                double delta = std::abs(omega_new - omega);
                std::cout << "  [SOS-LT] Root " << root + 1 << " iter " << std::setw(2) << iter + 1
                          << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                          << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                          << std::defaultfloat << std::endl;
                if (delta < omega_thr) {
                    eigenvalues[root] = omega_new;
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
                    converged = true;
                    std::cout << "  [SOS-LT] Root " << root + 1 << ": converged in "
                              << iter + 1 << " iterations" << std::endl;
                    break;
                }
                omega = omega_new;
                if (iter == max_omega_iter - 1)
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
            }
            if (!converged) {
                eigenvalues[root] = omega;
                std::cout << "  [SOS-LT] Root " << root + 1 << ": NOT converged after "
                          << max_omega_iter << " iterations" << std::endl;
            }
        }
        std::cout << "  [SOS-LT] Davidson time: " << std::fixed << std::setprecision(3)
                  << dav_timer.elapsed_seconds() << " s" << std::endl;
    }

    // --- Step 4: Excited state properties ---
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_M11);
        cudaFree(d_B_ij_full);
        cudaFree(d_B_ab_full);
    }

    rhf_.set_excitation_energies(eigenvalues);
    rhf_.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "Distributed-SOS-Laplace-ADC(2)",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        rhf_.get_coefficient_matrix().host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, rhf_.get_num_basis(), nocc, nvir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);

    // Cleanup
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_B_ia_local[d]);
    }
}

// ============================================================
//  build_B_*_local: MO-transform B_local for distributed ADC(2)
// ============================================================

// Forward declaration from eri_ri.cu (no-handle overload)
void transform_intermediate_matrix(int norbs, int nocc, int nvir, int naux,
                                   double* d_C, double* d_B, double* d_tmp);

real_t* ERI_RI_Distributed_RHF::build_B_ia_local(
    const real_t* d_B_ao, int naux_local,
    real_t* d_C, int nbas, int nocc, int nvir)
{
    const size_t B_ao_size = (size_t)naux_local * nbas * nbas;

    // Copy AO-basis B (transform is destructive)
    real_t* d_B_copy = nullptr;
    tracked_cudaMalloc(&d_B_copy, B_ao_size * sizeof(real_t));
    cudaMemcpy(d_B_copy, d_B_ao, B_ao_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Workspace for intermediate result
    real_t* d_tmp = nullptr;
    tracked_cudaMalloc(&d_tmp, (size_t)nbas * nvir * naux_local * sizeof(real_t));

    // nu→a, then mu→i: B^P_{mu,nu} → B^P_{i,a}
    transform_intermediate_matrix(nbas, nocc, nvir, naux_local, d_C, d_B_copy, d_tmp);

    tracked_cudaFree(d_tmp);
    return d_B_copy;  // [ov × naux_local] col-major
}

real_t* ERI_RI_Distributed_RHF::build_B_ab_local(
    const real_t* d_B_ao, int naux_local,
    real_t* d_C, int nbas, int nocc, int nvir)
{
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;
    const size_t vv = (size_t)nvir * nvir;

    // Copy AO-basis B
    const size_t B_ao_size = (size_t)naux_local * nbas * nbas;
    real_t* d_B_ao_copy = nullptr;
    tracked_cudaMalloc(&d_B_ao_copy, B_ao_size * sizeof(real_t));
    cudaMemcpy(d_B_ao_copy, d_B_ao, B_ao_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Step 1: nu→a (virtual): B^P_{mu,a} = sum_nu C_{nu,nocc+a} * B^P_{mu,nu}
    real_t* d_B_mu_a = nullptr;
    tracked_cudaMalloc(&d_B_mu_a, (size_t)nbas * nvir * naux_local * sizeof(real_t));
    cudaMemset(d_B_mu_a, 0, (size_t)nbas * nvir * naux_local * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nvir, naux_local * nbas, nbas,
                &alpha, &d_C[nocc], nbas,
                d_B_ao_copy, nbas,
                &beta, d_B_mu_a, nvir);
    tracked_cudaFree(d_B_ao_copy);

    // Step 2: Transpose [nvir × naux_local*nbas] → [naux_local*nbas × nvir]
    real_t* d_B_transposed = nullptr;
    tracked_cudaMalloc(&d_B_transposed, (size_t)nbas * nvir * naux_local * sizeof(real_t));
    {
        int row = naux_local * nbas, col = nvir;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    row, col,
                    &alpha, d_B_mu_a, col,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, row);
    }

    // Step 3: mu→b (virtual): B^P_{b,a} = sum_mu C_{mu,nocc+b} * B^P_{mu,a}
    cudaMemset(d_B_mu_a, 0, (size_t)nbas * nbas * naux_local * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nvir, naux_local * nvir, nbas,
                &alpha, &d_C[nocc], nbas,
                d_B_transposed, nbas,
                &beta, d_B_mu_a, nvir);

    // Step 4: Transpose to get [vv × naux_local] col-major layout
    {
        int row = naux_local * nvir, col = nvir;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    col, row,
                    &alpha, d_B_mu_a, row,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, col);
    }
    // d_B_transposed[P*vv + b*nvir + a] = B^P_{b,a}

    tracked_cudaFree(d_B_mu_a);
    return d_B_transposed;  // [vv × naux_local] col-major
}

real_t* ERI_RI_Distributed_RHF::build_B_ij_local(
    const real_t* d_B_ao, int naux_local,
    real_t* d_C, int nbas, int nocc, int nvir)
{
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;
    const size_t oo = (size_t)nocc * nocc;

    // Copy AO-basis B
    const size_t B_ao_size = (size_t)naux_local * nbas * nbas;
    real_t* d_B_ao_copy = nullptr;
    tracked_cudaMalloc(&d_B_ao_copy, B_ao_size * sizeof(real_t));
    cudaMemcpy(d_B_ao_copy, d_B_ao, B_ao_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Step 1: nu→j (occupied): B^P_{mu,j} = sum_nu C_{nu,j} * B^P_{mu,nu}
    real_t* d_B_mu_j = nullptr;
    tracked_cudaMalloc(&d_B_mu_j, (size_t)nbas * nocc * naux_local * sizeof(real_t));
    cudaMemset(d_B_mu_j, 0, (size_t)nbas * nocc * naux_local * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nocc, naux_local * nbas, nbas,
                &alpha, d_C, nbas,      // C_occ: no offset
                d_B_ao_copy, nbas,
                &beta, d_B_mu_j, nocc);
    tracked_cudaFree(d_B_ao_copy);

    // Step 2: Transpose [nocc × naux_local*nbas] → [naux_local*nbas × nocc]
    real_t* d_B_transposed = nullptr;
    tracked_cudaMalloc(&d_B_transposed, (size_t)nbas * nocc * naux_local * sizeof(real_t));
    {
        int row = naux_local * nbas, col = nocc;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    row, col,
                    &alpha, d_B_mu_j, col,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, row);
    }

    // Step 3: mu→i (occupied): B^P_{i,j} = sum_mu C_{mu,i} * B^P_{mu,j}
    cudaMemset(d_B_mu_j, 0, (size_t)nbas * nbas * naux_local * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nocc, naux_local * nocc, nbas,
                &alpha, d_C, nbas,
                d_B_transposed, nbas,
                &beta, d_B_mu_j, nocc);

    // Step 4: Transpose to get [oo × naux_local] col-major layout
    {
        int row = naux_local * nocc, col = nocc;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    col, row,
                    &alpha, d_B_mu_j, row,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, col);
    }
    // d_B_transposed[P*oo + i*nocc + j] = B^P_{i,j}

    tracked_cudaFree(d_B_mu_j);
    return d_B_transposed;  // [oo × naux_local] col-major
}

// ============================================================================
//  Distributed post-HF: MP2 via build_mo_eri (avoids broken intermediate_matrix_B_)
// ============================================================================

// Forward declaration (defined in eri_ri.cu)
double mp2_from_full_moeri(
    const double* d_eri_mo, const double* d_C, const double* d_eps,
    int nao, int occ, int frozen);

real_t ERI_RI_Distributed_RHF::compute_mp2_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), num_basis_);
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
    int nocc = rhf_.get_num_electrons() / 2;
    int frozen = rhf_.get_num_frozen_core();
    real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc, frozen);
    tracked_cudaFree(d_mo_eri);
    std::cout << "MP2 energy: " << E << " Hartree" << std::endl;
    return E;
}

real_t ERI_RI_Distributed_RHF::compute_scs_mp2_energy() {
    // TODO: distributed build_B_ia for proper SCS-MP2
    // For now, use full MO ERI fallback (same as stored ERI path)
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), num_basis_);
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
    int nocc = rhf_.get_num_electrons() / 2;
    real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc, 0);
    tracked_cudaFree(d_mo_eri);
    std::cout << "SCS-MP2 (fallback to MP2 via full MO ERI): " << E << " Hartree" << std::endl;
    return E;
}

real_t ERI_RI_Distributed_RHF::compute_sos_mp2_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), num_basis_);
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
    int nocc = rhf_.get_num_electrons() / 2;
    real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc, 0);
    tracked_cudaFree(d_mo_eri);
    std::cout << "SOS-MP2 (fallback to MP2 via full MO ERI): " << E << " Hartree" << std::endl;
    return E;
}

// ============================================================================
//  build_mo_eri: three paths
//    Replicated:   full B replicated on every GPU → run pipeline locally on caller
//    Distributed:  B sliced by aux index → each GPU runs partial pipeline, NCCL AllReduce
//    Single-GPU:   num_gpus_ == 1 → run pipeline on GPU 0
// ============================================================================

// Step 6.3 — single CUDA kernel that replaces the per-Q cublasDgeam transpose
// loop in build_eri_from_B_pipeline / ERI_RI::build_mo_eri.
//
// Input  d_B_tmp  layout (naux × nao × nmo) row-major,  element [Q, ν, p]
//                                                       at Q·nao·nmo + ν·nmo + p
// Output d_B_tmp2 layout (naux × nmo × nao) row-major,  element [Q, p, ν]
//                                                       at Q·nmo·nao + p·nao + ν
//
// Threads tile (p, ν) per Q. Reads are coalesced along p (innermost in input);
// writes are strided by nao but the transposed buffer is small enough that
// memory bandwidth, not coalescing, sets the lower bound here.
__global__ void transpose_b_tmp_per_aux_kernel(
    const real_t* __restrict__ d_in,
    real_t*       __restrict__ d_out,
    int naux, int nmo, int nao)
{
    const int Q  = blockIdx.z;
    const int p  = blockIdx.x * blockDim.x + threadIdx.x;
    const int nu = blockIdx.y * blockDim.y + threadIdx.y;
    if (Q >= naux || p >= nmo || nu >= nao) return;

    const size_t in_off  = (static_cast<size_t>(Q) * nao + nu)
                         * static_cast<size_t>(nmo) + p;
    const size_t out_off = (static_cast<size_t>(Q) * nmo + p)
                         * static_cast<size_t>(nao) + nu;
    d_out[out_off] = d_in[in_off];
}

static void launch_transpose_b_tmp(const real_t* d_in, real_t* d_out,
                                   int naux, int nmo, int nao,
                                   cudaStream_t stream = 0)
{
    constexpr int TILE_X = 32;
    constexpr int TILE_Y = 8;
    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((nmo + TILE_X - 1) / TILE_X,
              (nao + TILE_Y - 1) / TILE_Y,
              naux);
    transpose_b_tmp_per_aux_kernel<<<grid, block, 0, stream>>>(
        d_in, d_out, naux, nmo, nao);
}

// Helper: standard (right half-transform → transpose → left half-transform → ERI)
// pipeline using a full-B pointer on the calling device. Returns eri_mo on the
// caller's device.
//
// Step 6.3b — per-pair cudaMalloc/Free for the d_B_tmp / d_B_tmp2 / d_B_mo
// scratch buffers cost ~10 ms/pair at hexamer scale (3 × cudaMalloc + 3 ×
// cudaMemset(zero) + 3 × cudaFree). We cache them in a thread-local pool that
// grows on demand.
//
// Step 6.3c — if d_eri_out is non-null, the final DGEMM writes there directly
// and the function returns d_eri_out (caller-owned buffer). When null (legacy
// path), a fresh buffer is allocated as before and the caller frees it.
static real_t* build_eri_from_B_pipeline(
    const real_t* d_C, int nmo, int nao, int naux,
    const real_t* d_B, cublasHandle_t handle,
    real_t* d_eri_out = nullptr)
{
    const double alpha = 1.0, beta = 0.0;
    const size_t nmo2 = (size_t)nmo * nmo;

    // ---- Step 6.3b: thread-local scratch pool (per device) ----
    static thread_local real_t* ws_B_tmp       = nullptr;
    static thread_local real_t* ws_B_tmp2      = nullptr;
    static thread_local real_t* ws_B_mo        = nullptr;
    static thread_local size_t  ws_B_tmp_bytes = 0;   // size of B_tmp / B_tmp2
    static thread_local size_t  ws_B_mo_bytes  = 0;
    static thread_local int     ws_device      = -1;

    int curr_dev = 0;
    cudaGetDevice(&curr_dev);
    const size_t need_B_tmp_bytes = (size_t)naux * nao * nmo * sizeof(real_t);
    const size_t need_B_mo_bytes  = (size_t)naux * nmo2 * sizeof(real_t);

    auto free_ws = [&]() {
        if (ws_B_tmp)  { tracked_cudaFree(ws_B_tmp);  ws_B_tmp  = nullptr; }
        if (ws_B_tmp2) { tracked_cudaFree(ws_B_tmp2); ws_B_tmp2 = nullptr; }
        if (ws_B_mo)   { tracked_cudaFree(ws_B_mo);   ws_B_mo   = nullptr; }
        ws_B_tmp_bytes = 0;
        ws_B_mo_bytes  = 0;
    };

    if (curr_dev != ws_device) {
        // Workspace is on a different device — free it (under that device's
        // context) and re-allocate fresh on the current one.
        if (ws_device >= 0 && (ws_B_tmp || ws_B_tmp2 || ws_B_mo)) {
            const int saved = curr_dev;
            cudaSetDevice(ws_device);
            free_ws();
            cudaSetDevice(saved);
        } else {
            // ws_device == -1, no buffers yet.
            free_ws();
        }
        ws_device = curr_dev;
    }

    if (need_B_tmp_bytes > ws_B_tmp_bytes) {
        if (ws_B_tmp)  { tracked_cudaFree(ws_B_tmp);  ws_B_tmp  = nullptr; }
        if (ws_B_tmp2) { tracked_cudaFree(ws_B_tmp2); ws_B_tmp2 = nullptr; }
        tracked_cudaMalloc(&ws_B_tmp,  need_B_tmp_bytes);
        tracked_cudaMalloc(&ws_B_tmp2, need_B_tmp_bytes);
        ws_B_tmp_bytes = need_B_tmp_bytes;
    }
    if (need_B_mo_bytes > ws_B_mo_bytes) {
        if (ws_B_mo) { tracked_cudaFree(ws_B_mo); ws_B_mo = nullptr; }
        tracked_cudaMalloc(&ws_B_mo, need_B_mo_bytes);
        ws_B_mo_bytes = need_B_mo_bytes;
    }

    real_t* d_B_tmp  = ws_B_tmp;
    real_t* d_B_tmp2 = ws_B_tmp2;
    real_t* d_B_mo   = ws_B_mo;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nao, nao,
        &alpha, d_C, nmo, d_B, nao, &beta, d_B_tmp, nmo);

    // Step 6.3: single kernel launch in lieu of naux × cublasDgeam (the
    // per-Q dispatch overhead was ~1.3s out of 21s pair_setup at hexamer
    // scale, 576 × 465 = 268k cublasDgeam calls).
    {
        cudaStream_t stream = nullptr;
        cublasGetStream(handle, &stream);
        launch_transpose_b_tmp(d_B_tmp, d_B_tmp2, naux, nmo, nao, stream);
    }

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nmo, nao,
        &alpha, d_C, nmo, d_B_tmp2, nao, &beta, d_B_mo, nmo);

    real_t* d_eri_mo = d_eri_out;
    if (d_eri_mo == nullptr) {
        // Legacy path: allocate a fresh output buffer; caller frees.
        tracked_cudaMalloc(&d_eri_mo, nmo2 * nmo2 * sizeof(real_t));
    }
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nmo2, nmo2, naux,
        &alpha, d_B_mo, nmo2, d_B_mo, nmo2, &beta, d_eri_mo, nmo2);

    // ws_B_tmp / ws_B_tmp2 / ws_B_mo are NOT freed here — they live in the
    // thread-local cache and are reused across calls. d_eri_mo is either
    // a freshly allocated buffer (caller frees) or the caller-supplied
    // d_eri_out (no free needed).
    return d_eri_mo;
}

real_t* ERI_RI_Distributed_RHF::build_mo_eri(const real_t* d_C, int nmo) const {
    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nmo2 = (size_t)nmo * nmo;

    // ---- Replicated path: full B on caller's current device ----
    if (b_replicated_) {
        int curr_dev;
        cudaGetDevice(&curr_dev);
        if (curr_dev < 0 || curr_dev >= num_gpus_ || !d_B_full_per_gpu_[curr_dev]) {
            throw std::runtime_error("build_mo_eri: replicated B requested but device "
                                     + std::to_string(curr_dev) + " has no copy");
        }
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        real_t* d_eri_mo = build_eri_from_B_pipeline(
            d_C, nmo, nao, naux, d_B_full_per_gpu_[curr_dev], handle);
        cudaDeviceSynchronize();
        return d_eri_mo;
    }

    // ---- Distributed path: collective build via NCCL AllReduce ----
    // Each GPU does a partial DGEMM on its B_local slice → eri_local.
    // AllReduce(SUM) combines across GPUs since
    //     eri[i,j,k,l] = Σ_P B_mo[P,i,j] B_mo[P,k,l]
    //                  = Σ_g Σ_{P∈slice g} B_mo[P,i,j] B_mo[P,k,l]
    // After AllReduce only GPU 0 keeps eri (others freed). Trades
    // num_gpus × naux × nao² (replication memory) for one nmo⁴ AllReduce.
    if (num_gpus_ > 1) {
        auto& mgr = MultiGpuManager::instance();
        std::vector<real_t*> d_C_per_gpu(num_gpus_, nullptr);
        std::vector<real_t*> d_eri_per_gpu(num_gpus_, nullptr);

        const size_t C_bytes = (size_t)nao * nmo * sizeof(real_t);
        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            if (g == 0) {
                d_C_per_gpu[g] = const_cast<real_t*>(d_C);
            } else {
                tracked_cudaMalloc(&d_C_per_gpu[g], C_bytes);
                cudaMemcpyPeer(d_C_per_gpu[g], g, d_C, 0, C_bytes);
            }
        }

        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            d_eri_per_gpu[g] = build_eri_from_B_pipeline(
                d_C_per_gpu[g], nmo, nao, naux_local_[g],
                d_B_local_[g], mgr.cublas(g));
        }

        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }

        nccl::group_start();
        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            nccl::all_reduce<real_t>(d_eri_per_gpu[g], d_eri_per_gpu[g],
                                     nmo2 * nmo2, ncclSum, g,
                                     mgr.comm_stream(g));
        }
        nccl::group_end();

        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            cudaStreamSynchronize(mgr.comm_stream(g));
        }

        for (int g = 0; g < num_gpus_; g++) {
            cudaSetDevice(g);
            if (g != 0 && d_C_per_gpu[g]) tracked_cudaFree(d_C_per_gpu[g]);
            if (g != 0) tracked_cudaFree(d_eri_per_gpu[g]);
        }

        cudaSetDevice(0);
        return d_eri_per_gpu[0];
    }

    // ---- Single-GPU fallback (num_gpus_ == 1) ----
    cudaSetDevice(0);
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    real_t* d_eri_mo = build_eri_from_B_pipeline(
        d_C, nmo, nao, naux_local_[0], d_B_local_[0], handle);
    cudaDeviceSynchronize();
    return d_eri_mo;
}

// P4b — half-transformed B_mo for distributed RI. The base ERI_RI::build_B_mo
// reads intermediate_matrix_B_ which is empty in the distributed class (we
// store B as per-GPU slices d_B_local_[g]). Lazily call replicate_B_to_all_gpus()
// so each GPU has the full naux×nao² B (≈ 580 MB at anthracene cc-pVDZ scale),
// then delegate to the base build_B_mo_impl using the current-device copy.
// Used by IP/EA/STEOM operator block-mode (P4b) so no nmo⁴ MO ERI is built.
const real_t* ERI_RI_Distributed_RHF::build_B_mo(const real_t* d_C, int nmo) const {
#ifdef GANSU_CPU_ONLY
    (void)d_C; (void)nmo;
    return nullptr;
#else
    if (!gpu::gpu_available()) return nullptr;
    if (!b_replicated_) {
        // const_cast is acceptable: replicate_B_to_all_gpus mutates only the
        // lazy-replication cache (d_B_full_per_gpu_, b_replicated_), not the
        // logical state of the operator. Memory check inside; returns false
        // if replication would blow the per-GPU budget.
        const bool ok = const_cast<ERI_RI_Distributed_RHF*>(this)->
                        replicate_B_to_all_gpus();
        if (!ok) {
            std::cout << "  [P4b] build_B_mo: replicate_B_to_all_gpus refused "
                         "(budget); falling back to full nmo⁴ MO ERI tensor."
                      << std::endl;
            return nullptr;   // caller falls back to build_mo_eri (legacy)
        }
    }
    int curr_dev = 0;
    cudaGetDevice(&curr_dev);
    // Device-0 fallback: if the calling context left the current device set to
    // something outside [0, num_gpus_) (e.g., CIS-NTO finalize), or to a device
    // that doesn't hold a replica (only seen with non-uniform multi-GPU layouts),
    // gracefully fall back to device 0 where the canonical replica always lives
    // after replicate_B_to_all_gpus().  Without this guard, build_B_mo silently
    // returns nullptr and the caller allocates a full nmo⁴ MO ERI buffer
    // (~88 GB at tetracene cc-pVDZ, OOM on a single H200 141 GB after DLPNO
    // ground state residency).  Print diagnostics for the nullptr paths so the
    // failure mode is visible if device 0 itself ever lacks a replica.
    const bool curr_dev_bad   = (curr_dev < 0 || curr_dev >= num_gpus_);
    const bool curr_dev_empty = (!curr_dev_bad) && !d_B_full_per_gpu_[curr_dev];
    if (curr_dev_bad || curr_dev_empty) {
        if (num_gpus_ > 0 && (int)d_B_full_per_gpu_.size() > 0 && d_B_full_per_gpu_[0]) {
            std::cout << "  [P4b] build_B_mo: curr_dev=" << curr_dev
                      << " "
                      << (curr_dev_bad ? "out of range" : "has no replica")
                      << "; using device-0 replica" << std::endl;
            cudaSetDevice(0);
            return build_B_mo_impl(d_B_full_per_gpu_[0], d_C, nmo);
        }
        std::cout << "  [P4b] build_B_mo: nullptr (curr_dev=" << curr_dev
                  << ", num_gpus_=" << num_gpus_
                  << ", replica vec size=" << d_B_full_per_gpu_.size()
                  << ", device-0 replica="
                  << (d_B_full_per_gpu_.empty() ? "n/a" : (d_B_full_per_gpu_[0] ? "ok" : "null"))
                  << ") — caller will fall back to nmo⁴ MO ERI" << std::endl;
        return nullptr;
    }
    return build_B_mo_impl(d_B_full_per_gpu_[curr_dev], d_C, nmo);
#endif
}

// P4b source lookup for the asymmetric V-block builder (build_B_mo_asym). Same
// lazy-replication + current-device selection as build_B_mo above, but returns
// the AO-basis B pointer instead of the full B_mo. Keeping build_B_mo untouched
// (it duplicates this small logic) preserves the validated P4b path exactly.
const real_t* ERI_RI_Distributed_RHF::B_ao_src_for_mo() const {
#ifdef GANSU_CPU_ONLY
    return nullptr;
#else
    if (!gpu::gpu_available()) return nullptr;
    if (!b_replicated_) {
        const bool ok = const_cast<ERI_RI_Distributed_RHF*>(this)->
                        replicate_B_to_all_gpus();
        if (!ok) return nullptr;
    }
    int curr_dev = 0;
    cudaGetDevice(&curr_dev);
    const bool curr_dev_bad   = (curr_dev < 0 || curr_dev >= num_gpus_);
    const bool curr_dev_empty = (!curr_dev_bad) && !d_B_full_per_gpu_[curr_dev];
    if (curr_dev_bad || curr_dev_empty) {
        if (num_gpus_ > 0 && (int)d_B_full_per_gpu_.size() > 0
            && d_B_full_per_gpu_[0]) {
            cudaSetDevice(0);
            return d_B_full_per_gpu_[0];
        }
        return nullptr;
    }
    return d_B_full_per_gpu_[curr_dev];
#endif
}

// Step 6.3c — workspace variant for the replicated-B path. Writes the MO ERI
// directly into the caller-supplied d_eri_out buffer (must be ≥ nmo⁴ doubles).
// For the multi-GPU NCCL distributed path the caller-provided buffer can't
// be used directly (each GPU produces a partial result that's AllReduced
// in-place), so we fall back to the legacy build_mo_eri + cudaMemcpy.
void ERI_RI_Distributed_RHF::build_mo_eri_into(const real_t* d_C, int nmo,
                                               real_t* d_eri_out) const
{
    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nmo2 = (size_t)nmo * nmo;

    if (b_replicated_) {
        // Fast path: pipeline writes into d_eri_out, no extra alloc/copy.
        int curr_dev;
        cudaGetDevice(&curr_dev);
        if (curr_dev < 0 || curr_dev >= num_gpus_ || !d_B_full_per_gpu_[curr_dev]) {
            throw std::runtime_error("build_mo_eri_into: replicated B requested but device "
                                     + std::to_string(curr_dev) + " has no copy");
        }
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        build_eri_from_B_pipeline(
            d_C, nmo, nao, naux, d_B_full_per_gpu_[curr_dev], handle,
            d_eri_out);
        cudaDeviceSynchronize();
        return;
    }

    // Distributed multi-GPU or single-GPU non-replicated paths: fall back
    // to the base-class default (build_mo_eri + cudaMemcpyDeviceToDevice
    // into d_eri_out + tracked_cudaFree). DLPNO pair_setup never reaches
    // here in practice because Step 6.3 Phase A pre-replicates B.
    ERI::build_mo_eri_into(d_C, nmo, d_eri_out);
}

// ============================================================================
//  Replicated-B mode: replicate full B to every GPU for fragment-parallel DMET.
//
//  After calling, each GPU holds the complete B[naux × nao²] tensor and
//  build_mo_eri can run independently per device (no peer copies). This
//  trades memory (factor of num_gpus) for fully parallel fragment solves.
// ============================================================================

bool ERI_RI_Distributed_RHF::replicate_B_to_all_gpus() {
    if (b_replicated_) return true;

    // Restore the CALLER's current device on every exit (not a hardcoded device 0):
    // DMET-STEOM can run the whole cluster chain on a free peer GPU
    // (GANSU_DMET_STEOM_CLUSTER_GPU); leaving the device at 0 here would leave the
    // Davidson / operator DGEMMs running with a device-0-bound handle against
    // device-N data → cublas status=13. All legacy callers are already on device 0,
    // so restore-to-caller == restore-to-0 for them (byte-identical).
    int caller_dev = 0; cudaGetDevice(&caller_dev);

    // Debug override: force the Distributed path even when B would fit.
    // Useful for exercising the collective build_mo_eri on small test systems
    // where memory is plentiful (otherwise Replicated is always selected).
    if (std::getenv("GANSU_DMET_FORCE_DISTRIBUTED")) {
        std::cout << "[Multi-GPU DMET] Distributed mode forced via "
                     "GANSU_DMET_FORCE_DISTRIBUTED" << std::endl;
        return false;
    }

    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nao2 = (size_t)nao * nao;
    const size_t full_bytes = (size_t)naux * nao2 * sizeof(real_t);

    // Memory check: confirm each GPU can hold a full B copy with margin.
    size_t min_free = SIZE_MAX;
    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        min_free = std::min(min_free, free_bytes);
    }
    cudaSetDevice(caller_dev);

    // Need to leave headroom for fragment-local working memory (CCSD intermediates,
    // MO ERI, B_tmp, etc). Require full_bytes < 60% of free on the tightest GPU.
    if (full_bytes > min_free * 6 / 10) {
        std::cout << "[Multi-GPU DMET] B replication skipped: full B = "
                  << (full_bytes >> 20) << " MB exceeds 60% of free GPU memory ("
                  << (min_free >> 20) << " MB). Falling back to gather-to-GPU0 path."
                  << std::endl;
        cudaSetDevice(caller_dev);
        return false;
    }

    d_B_full_per_gpu_.assign(num_gpus_, nullptr);

    // Allocate full-B buffer on every GPU.
    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        tracked_cudaMalloc(&d_B_full_per_gpu_[g], full_bytes);
    }

    // Each source GPU's slice is broadcast to every destination GPU.
    // Layout: d_B_full[offset_g .. offset_g + naux_local[g]*nao²] = d_B_local_[g].
    std::vector<size_t> offsets(num_gpus_, 0);
    for (int g = 1; g < num_gpus_; g++)
        offsets[g] = offsets[g - 1] + (size_t)naux_local_[g - 1] * nao2;

    // NCCL Broadcast path: launch one Broadcast per source root, grouped so
    // the underlying ring/tree topology is exploited instead of 64 pairwise
    // peer-copies serialized on default streams. On NVLink fabrics this is
    // typically 5-10× faster than the pairwise approach; on PCIe-only it is
    // similar to (or marginally better than) pairwise.
    auto& mgr_b = MultiGpuManager::instance();
    nccl::group_start();
    for (int src = 0; src < num_gpus_; src++) {
        const size_t count_src = (size_t)naux_local_[src] * nao2;
        for (int d = 0; d < num_gpus_; d++) {
            cudaSetDevice(d);
            // On the root rank we send from d_B_local_[src]; on others we
            // receive into d_B_full_per_gpu_[d] + offsets[src]. ncclBroadcast
            // takes one (sendbuff, recvbuff) pair per rank; we use the same
            // recv-side pointer on all ranks (including root, which copies
            // its sendbuff into the local recv-slot first).
            const real_t* sendbuff = (d == src) ? d_B_local_[src]
                                                : (d_B_full_per_gpu_[d] + offsets[src]);
            real_t* recvbuff = d_B_full_per_gpu_[d] + offsets[src];
            ncclResult_t rc = ncclBroadcast(sendbuff, recvbuff, count_src,
                                            ncclFloat64, src,
                                            mgr_b.nccl_comm(d),
                                            mgr_b.comm_stream(d));
            if (rc != ncclSuccess) {
                throw std::runtime_error(std::string("[replicate_B NCCL] ") +
                                         ncclGetErrorString(rc));
            }
        }
    }
    nccl::group_end();
    for (int g = 0; g < num_gpus_; g++) {
        cudaSetDevice(g);
        cudaStreamSynchronize(mgr_b.comm_stream(g));
    }
    cudaSetDevice(caller_dev);

    b_replicated_ = true;
    std::cout << "[Multi-GPU DMET] B replicated across " << num_gpus_
              << " GPUs (" << (full_bytes >> 20) << " MB each)" << std::endl;

    // (DMET-STEOM device-0 fit) The full replica now supersedes the per-GPU 3c
    // slices d_B_local_ for build_B_mo (the cluster chain reads d_B_full_per_gpu_,
    // never d_B_local_ — see build_B_mo above). The SCF (the only B_local consumer)
    // is finished before this lazy replication runs, so freeing B_local reclaims
    // ~naux_local·nao² (~3 GB/GPU) — enough to close the ~1 GB by which a device-0
    // cluster CCSD is short at cc-pVDZ (n_emb=427). Env-gated (DMET-STEOM sets it);
    // other flows keep B_local. Irreversible for this object (no post-cluster reuse).
    if (std::getenv("GANSU_DMET_STEOM_FREE_BLOCAL")) {
        size_t freed = 0;
        for (int g = 0; g < (int)d_B_local_.size(); g++) {
            if (d_B_local_[g]) {
                cudaSetDevice(g);
                tracked_cudaFree(d_B_local_[g]);
                d_B_local_[g] = nullptr;
                freed += (size_t)naux_local_[g] * nao2;
            }
        }
        cudaSetDevice(caller_dev);
        std::cout << "[Multi-GPU DMET] freed per-GPU B_local slices post-replication "
                  << "(GANSU_DMET_STEOM_FREE_BLOCAL) — reclaimed ~"
                  << ((freed * sizeof(real_t)) >> 20) / std::max(1, num_gpus_)
                  << " MB/GPU." << std::endl;
    }
    return true;
}

void ERI_RI_Distributed_RHF::free_replicated_B() {
    if (!b_replicated_) return;
    int caller_dev = 0; cudaGetDevice(&caller_dev);  // restore caller (not hardcoded 0):
    for (int g = 0; g < (int)d_B_full_per_gpu_.size(); g++) {   // DMET-STEOM cluster may
        if (d_B_full_per_gpu_[g]) {                             // run on a free peer GPU,
            cudaSetDevice(g);                                   // where the following geev
            tracked_cudaFree(d_B_full_per_gpu_[g]);            // must stay on that device.
        }
    }
    d_B_full_per_gpu_.clear();
    b_replicated_ = false;
    cudaSetDevice(caller_dev);
}

// (ERI_RI hook) Free the AO-B replica after build_B_mo. const_cast mirrors
// build_B_mo: this only touches the lazy-replication cache, not logical state.
// The next build_B_mo re-replicates. Used by the DMET cluster CCSD ground to
// reclaim ~naux·nao² per GPU before allocating its residual pool.
void ERI_RI_Distributed_RHF::release_bmo_ao_replica() const {
#ifndef GANSU_CPU_ONLY
    if (b_replicated_)
        const_cast<ERI_RI_Distributed_RHF*>(this)->free_replicated_B();
#endif
}

// DMET-CCSD entry point. Forwards to DMET; the multi-GPU optimization is
// applied inside DMET::compute_energy() when it detects this ERI subclass.
real_t ERI_RI_Distributed_RHF::compute_dmet_ccsd() {
    DMET dmet(rhf_, *this);
    return dmet.compute_energy(/*with_triples=*/false);
}

real_t ERI_RI_Distributed_RHF::compute_dmet_ccsd_t() {
    DMET dmet(rhf_, *this);
    return dmet.compute_energy(/*with_triples=*/true);
}

// ============================================================
//  Multi-GPU RI CIS / CIS-NTO (bt-PNO-STEOM stage 2 unblock)
// ============================================================
// In distributed mode B is aux-partitioned (d_B_local_[d] = [naux_local × nbas²]),
// and intermediate_matrix_B_ is null, so the single-GPU CIS core fails its guard.
// We gather the slabs into a full B on device 0, point cis_B_override_ at it, run
// the validated single-GPU CIS core, then free. The aux partition is contiguous and
// ascending (device 0 = lowest aux), so the slabs concatenate directly.

real_t* ERI_RI_Distributed_RHF::gather_full_B_device0() const {
    const size_t nbas2 = static_cast<size_t>(num_basis_) * num_basis_;
    real_t* d_B_full = nullptr;
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaMalloc(&d_B_full,
                           static_cast<size_t>(num_auxiliary_basis_) * nbas2 * sizeof(real_t));
    }
    size_t aux_off = 0;
    for (int d = 0; d < num_gpus_; ++d) {
        const size_t n = static_cast<size_t>(naux_local_[d]) * nbas2;
        if (n > 0)
            cudaMemcpyPeer(d_B_full + aux_off * nbas2, 0, d_B_local_[d], d, n * sizeof(real_t));
        aux_off += static_cast<size_t>(naux_local_[d]);
    }
    cudaDeviceSynchronize();
    return d_B_full;
}

void ERI_RI_Distributed_RHF::compute_cis(int n_states) {
    MultiGpuManager::DeviceGuard guard(0);
    real_t* d_B_full = gather_full_B_device0();
    cis_B_override_ = d_B_full;
    ERI_RI_RHF::compute_cis(n_states);
    cis_B_override_ = nullptr;
    tracked_cudaFree(d_B_full);
}

void ERI_RI_Distributed_RHF::compute_cis_nto(int n_states_cis) {
    MultiGpuManager::DeviceGuard guard(0);
    real_t* d_B_full = gather_full_B_device0();
    cis_B_override_ = d_B_full;
    ERI_RI_RHF::compute_cis_nto(n_states_cis);
    cis_B_override_ = nullptr;
    tracked_cudaFree(d_B_full);
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
