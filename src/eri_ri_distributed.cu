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
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "ri_adc2_schur_distributed_operator.hpp"
#include "sos_laplace_adc2_distributed_operator.hpp"
#include "davidson_solver.hpp"
#include "oscillator_strength.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

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
    for (int d = 0; d < num_gpus_; d++) {
        auto [start, end] = aux_partition(num_auxiliary_basis_, num_gpus_, d);
        P_start_[d] = start;
        naux_local_[d] = (int)(end - start);
    }

    d_B_local_.resize(num_gpus_, nullptr);
    d_W_local_.resize(num_gpus_, nullptr);
    d_J_local_.resize(num_gpus_, nullptr);
    d_K_local_.resize(num_gpus_, nullptr);
    d_X_local_.resize(num_gpus_, nullptr);
    d_X_packed_local_.resize(num_gpus_, nullptr);

    std::cout << "[RI-Distributed] " << num_gpus_ << " GPUs, naux=" << num_auxiliary_basis_;
    for (int d = 0; d < num_gpus_; d++)
        std::cout << " [" << d << "]:" << naux_local_[d];
    std::cout << std::endl;
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
            g.d_pshells = const_cast<PrimitiveShell*>(hf_.get_primitive_shells().device_ptr());
            g.d_cgto_norms = const_cast<real_t*>(hf_.get_cgto_normalization_factors().device_ptr());
            g.d_aux_pshells = auxiliary_primitive_shells_.device_ptr();
            g.d_aux_cgto_norms = auxiliary_cgto_normalization_factors_.device_ptr();
            g.d_shell_pairs = d_persistent_shell_pair_indices_;
            g.d_schwarz = schwarz_upper_bound_factors.device_ptr();
            g.d_aux_schwarz = auxiliary_schwarz_upper_bound_factors.device_ptr();
            g.d_boys = const_cast<real_t*>(hf_.get_boys_grid().device_ptr());
        } else {
            cudaMalloc(&g.d_L_inv, (size_t)naux * naux * sizeof(real_t));
            cudaMemcpy(g.d_L_inv, d_cached_L_inv_, (size_t)naux * naux * sizeof(real_t), cudaMemcpyDefault);
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
        // Pre-allocate chunk buffer (max aux type size) on every GPU
        cudaMalloc(&g.d_chunk, (size_t)max_nfunc_chunk_ * nbas2 * sizeof(real_t));
    }
    MultiGpuManager::instance().sync_all();
    per_gpu_data_ready_ = true;
}

void ERI_RI_Distributed_RHF::free_per_gpu_data() {
    for (int d = 0; d < (int)per_gpu_data_.size(); d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& g = per_gpu_data_[d];
        if (g.d_chunk) { cudaFree(g.d_chunk); g.d_chunk = nullptr; }
        if (d == 0) continue;  // GPU 0 data is borrowed, not owned
        if (g.d_L_inv) cudaFree(g.d_L_inv);
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
    const int max_batch_nfunc = std::max(1, (int)(available_for_chunk / (nbas2 * sizeof(real_t))));

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
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;

    // Ensure GPU 0 is active (base class data lives on GPU 0)
    cudaSetDevice(0);

    // Step 1: Schwarz + shell pairs + aux Schwarz (skip full B build)
    precompute_schwarz_and_shell_pairs();

    std::cout << "[RI-Dist] Skipped full B build (lightweight constructor)" << std::endl;

    // Step 2: Compute auxiliary type ranges for chunked 3c2e
    auxiliary_primitive_shells_.toHost();
    compute_aux_type_ranges();

    // Step 3: Compute and cache L⁻¹ on GPU 0
    {
        cudaSetDevice(0);
        const real_t schwarz_threshold = hf_.get_schwarz_screening_threshold();

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

    // Batch sizing in compute_aux_type_ranges() ensures chunks fit alongside B_local.
    std::cout << "[RI-Dist] Building B_local on " << num_gpus_ << " GPUs (chunked 3c2e + L^-1 DGEMM)..." << std::endl;

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
            const double one = 1.0;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)nbas2, naux_local_[d], nfunc_c,
                &one,
                g.d_chunk, (int)nbas2,
                &g.d_L_inv[P_start_[d] * naux + Q_c_start], naux,
                &one,
                d_B_local_[d], (int)nbas2);
        }

        // Sync all GPUs before next aux type (chunk reused)
        mgr.sync_all();
    }

    // ---- Step 4: Cleanup ----
    // In stored mode, free replicated data + cached L⁻¹ (no longer needed).
    // In direct mode, keep them for per-iteration rebuilds.
    if (storage_mode_ != StorageMode::OnTheFly) {
        free_per_gpu_data();
        if (d_cached_L_inv_) {
            MultiGpuManager::DeviceGuard guard(0);
            tracked_cudaFree(d_cached_L_inv_);
            d_cached_L_inv_ = nullptr;
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

    if (fits_on_gpu) {
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

            const double one_val = 1.0;
            cublasDgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)nbas2, P_count, batch.nfunc,
                &one_val,
                d_3c_chunk, (int)nbas2,
                &g.d_L_inv[P_start * naux + batch.basis_start], naux,
                &one_val,
                d_B_dest, (int)nbas2);
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

                const double one = 1.0;
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)nbas2, P_count, nfunc_c,
                    &one,
                    w.d_3c_chunk, (int)nbas2,
                    &g.d_L_inv[P_start_global * naux + Q_c_start], naux,
                    &one,
                    w.d_B_row, (int)nbas2);
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
//  build_mo_eri: gather distributed B_local → full B on GPU 0, then delegate
// ============================================================================

real_t* ERI_RI_Distributed_RHF::build_mo_eri(const real_t* d_C, int nmo) const {
    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nao2 = (size_t)nao * nao;

    // Gather full B [naux × nao²] on GPU 0 from distributed B_local chunks
    cudaSetDevice(0);
    real_t* d_B_full = nullptr;
    tracked_cudaMalloc(&d_B_full, (size_t)naux * nao2 * sizeof(real_t));

    size_t offset = 0;
    for (int g = 0; g < num_gpus_; g++) {
        const size_t chunk = (size_t)naux_local_[g] * nao2;
        if (g == 0) {
            cudaMemcpy(d_B_full + offset, d_B_local_[g],
                       chunk * sizeof(real_t), cudaMemcpyDeviceToDevice);
        } else {
            // Cross-GPU: use cudaMemcpyPeer for reliable transfer
            cudaMemcpyPeer(d_B_full + offset, 0, d_B_local_[g], g,
                           chunk * sizeof(real_t));
        }
        offset += chunk;
    }
    cudaDeviceSynchronize();

    // Temporarily set intermediate_matrix_B_ to the gathered B
    // Since build_mo_eri in ERI_RI reads intermediate_matrix_B_.device_ptr(),
    // we write d_B_full into it (it was a 1×1 dummy).
    // Instead, call the GPU DGEMM sequence directly here.

    // Reuse the ERI_RI::build_mo_eri logic with d_B_full:
    // Step 1: B_tmp = C^T × B  (right half-transform)
    // Step 2: B_mo = C^T × B_tmp (left half-transform with transpose)
    // Step 3: eri_mo = B_mo^T × B_mo

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;
    const size_t nmo2 = (size_t)nmo * nmo;

    // Step 1: right half-transform
    real_t* d_B_tmp = nullptr;
    tracked_cudaMalloc(&d_B_tmp, (size_t)naux * nao * nmo * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nao, nao,
        &alpha, d_C, nmo, d_B_full, nao,
        &beta, d_B_tmp, nmo);

    tracked_cudaFree(d_B_full);

    // Step 2: transpose per Q then left half-transform
    real_t* d_B_tmp2 = nullptr;
    tracked_cudaMalloc(&d_B_tmp2, (size_t)naux * nao * nmo * sizeof(real_t));

    for (int Q = 0; Q < naux; Q++) {
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            nao, nmo, &alpha,
            d_B_tmp + (size_t)Q * nao * nmo, nmo,
            &beta, nullptr, nao,
            d_B_tmp2 + (size_t)Q * nmo * nao, nao);
    }
    tracked_cudaFree(d_B_tmp);

    real_t* d_B_mo = nullptr;
    tracked_cudaMalloc(&d_B_mo, (size_t)naux * nmo2 * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nmo, nao,
        &alpha, d_C, nmo, d_B_tmp2, nao,
        &beta, d_B_mo, nmo);

    tracked_cudaFree(d_B_tmp2);

    // Step 3: eri_mo = B_mo^T × B_mo
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_mo, nmo2 * nmo2 * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nmo2, nmo2, naux,
        &alpha, d_B_mo, nmo2, d_B_mo, nmo2,
        &beta, d_eri_mo, nmo2);

    tracked_cudaFree(d_B_mo);
    cudaDeviceSynchronize();

    return d_eri_mo;
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
