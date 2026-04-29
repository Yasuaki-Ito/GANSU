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
        const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
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
        int aux_type_index,
        size_t aux_basis_offset,
        int nfunc_chunk);
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
    : ERI_RI_RHF(rhf, auxiliary_molecular)
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
    free_per_device_workspace();
    if (d_cached_L_inv_) {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_cached_L_inv_);
        d_cached_L_inv_ = nullptr;
    }
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
    aux_type_basis_start_.resize(n_aux_types);
    aux_type_nfunc_.resize(n_aux_types);

    // Scan host-side auxiliary primitive shells to find min/max basis_index per type
    const PrimitiveShell* h_aux = auxiliary_primitive_shells_.host_ptr();
    for (int c = 0; c < n_aux_types; c++) {
        const size_t start = aux_types[c].start_index;
        const size_t count = aux_types[c].count;
        const int L = h_aux[start].shell_type;
        const int n_cart = (L + 1) * (L + 2) / 2;

        size_t min_idx = h_aux[start].basis_index;
        size_t max_idx = h_aux[start].basis_index;
        for (size_t i = 1; i < count; i++) {
            size_t bi = h_aux[start + i].basis_index;
            if (bi < min_idx) min_idx = bi;
            if (bi > max_idx) max_idx = bi;
        }
        aux_type_basis_start_[c] = min_idx;
        aux_type_nfunc_[c] = (int)(max_idx - min_idx) + n_cart;
    }
}

// ============================================================
//  Precomputation: Schwarz + shell pairs + 2c2e/Cholesky
//  Then distributed B build (no full B on any single GPU)
// ============================================================
void ERI_RI_Distributed_RHF::precomputation() {
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;

    // Step 1: Schwarz + shell pairs + aux Schwarz (skip full B build)
    precompute_schwarz_and_shell_pairs();

    // Release intermediate_matrix_B_ allocated by base constructor (never used in distributed mode)
    intermediate_matrix_B_.release();
    // Release single-GPU Fock workspace (not needed — we use distributed J/K)
    d_tmp1_.release();
    d_tmp2_.release();

    std::cout << "[RI-Dist] Skipped full B build (saved "
              << (size_t)naux * nbas * nbas * sizeof(real_t) / (1024 * 1024) << " MB)" << std::endl;

    // Step 2: Compute auxiliary type ranges for chunked 3c2e
    auxiliary_primitive_shells_.toHost();
    compute_aux_type_ranges();

    // Step 3: Compute and cache L⁻¹ on GPU 0
    {
        MultiGpuManager::DeviceGuard guard(0);
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
        std::cout << "[RI-Dist] Cached L^-1 on GPU 0 ("
                  << (size_t)naux * naux * sizeof(real_t) / (1024 * 1024) << " MB)" << std::endl;
    }

    // Step 4: Build B_local on all GPUs immediately
    distributed_build_B();
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

    std::cout << "[RI-Dist] Building B_local on " << num_gpus_ << " GPUs (chunked 3c2e + L^-1 DGEMM)..." << std::endl;

    // ---- Step 1: Use cached L⁻¹ (or recompute if not available) ----
    real_t* d_L_inv_gpu0 = nullptr;
    bool L_inv_owned = false;  // true if we allocated L⁻¹ here (need to free)
    if (d_cached_L_inv_) {
        d_L_inv_gpu0 = d_cached_L_inv_;
    } else {
        // Fallback: recompute (e.g., direct_mode_ rebuilds after cache freed)
        MultiGpuManager::DeviceGuard guard(0);
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
            schwarz_screening_threshold, false);

        gpu::choleskyDecomposition(d_two_center_eri, naux);

        tracked_cudaMalloc(&d_L_inv_gpu0, (size_t)naux * naux * sizeof(real_t));
        gpu::computeInverseByDtrsm(d_two_center_eri, d_L_inv_gpu0, naux);

        tracked_cudaFree(d_two_center_eri);
        L_inv_owned = true;
    }

    // ---- Step 2: Replicate L⁻¹ and shell data to all GPUs ----
    std::vector<real_t*> d_L_inv(num_gpus_, nullptr);
    std::vector<PrimitiveShell*> d_pshells(num_gpus_, nullptr);
    std::vector<real_t*> d_cgto_norms(num_gpus_, nullptr);
    std::vector<PrimitiveShell*> d_aux_pshells(num_gpus_, nullptr);
    std::vector<real_t*> d_aux_cgto_norms(num_gpus_, nullptr);
    std::vector<size_t2*> d_shell_pairs(num_gpus_, nullptr);
    std::vector<real_t*> d_schwarz(num_gpus_, nullptr);
    std::vector<real_t*> d_aux_schwarz(num_gpus_, nullptr);
    std::vector<real_t*> d_boys(num_gpus_, nullptr);

    const size_t n_pshells = hf_.get_primitive_shells().size();
    const size_t n_aux_pshells = auxiliary_primitive_shells_.size();
    const size_t n_cgto = hf_.get_cgto_normalization_factors().size();
    const size_t n_aux_cgto = auxiliary_cgto_normalization_factors_.size();
    const size_t n_boys = hf_.get_boys_grid().size();

    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d == 0) {
            d_L_inv[d] = d_L_inv_gpu0;
            d_pshells[d] = const_cast<PrimitiveShell*>(hf_.get_primitive_shells().device_ptr());
            d_cgto_norms[d] = const_cast<real_t*>(hf_.get_cgto_normalization_factors().device_ptr());
            d_aux_pshells[d] = auxiliary_primitive_shells_.device_ptr();
            d_aux_cgto_norms[d] = auxiliary_cgto_normalization_factors_.device_ptr();
            d_shell_pairs[d] = d_persistent_shell_pair_indices_;
            d_schwarz[d] = schwarz_upper_bound_factors.device_ptr();
            d_aux_schwarz[d] = auxiliary_schwarz_upper_bound_factors.device_ptr();
            d_boys[d] = const_cast<real_t*>(hf_.get_boys_grid().device_ptr());
        } else {
            // L⁻¹
            cudaMalloc(&d_L_inv[d], (size_t)naux * naux * sizeof(real_t));
            cudaMemcpy(d_L_inv[d], d_L_inv_gpu0, (size_t)naux * naux * sizeof(real_t), cudaMemcpyDefault);
            // Primitive shells
            cudaMalloc(&d_pshells[d], n_pshells * sizeof(PrimitiveShell));
            cudaMemcpy(d_pshells[d], hf_.get_primitive_shells().device_ptr(), n_pshells * sizeof(PrimitiveShell), cudaMemcpyDefault);
            // CGTO norms
            cudaMalloc(&d_cgto_norms[d], n_cgto * sizeof(real_t));
            cudaMemcpy(d_cgto_norms[d], hf_.get_cgto_normalization_factors().device_ptr(), n_cgto * sizeof(real_t), cudaMemcpyDefault);
            // Auxiliary primitive shells
            cudaMalloc(&d_aux_pshells[d], n_aux_pshells * sizeof(PrimitiveShell));
            cudaMemcpy(d_aux_pshells[d], auxiliary_primitive_shells_.device_ptr(), n_aux_pshells * sizeof(PrimitiveShell), cudaMemcpyDefault);
            // Auxiliary CGTO norms
            cudaMalloc(&d_aux_cgto_norms[d], n_aux_cgto * sizeof(real_t));
            cudaMemcpy(d_aux_cgto_norms[d], auxiliary_cgto_normalization_factors_.device_ptr(), n_aux_cgto * sizeof(real_t), cudaMemcpyDefault);
            // Shell pair indices
            cudaMalloc(&d_shell_pairs[d], num_persistent_shell_pairs_ * sizeof(size_t2));
            cudaMemcpy(d_shell_pairs[d], d_persistent_shell_pair_indices_, num_persistent_shell_pairs_ * sizeof(size_t2), cudaMemcpyDefault);
            // Schwarz factors
            cudaMalloc(&d_schwarz[d], schwarz_upper_bound_factors.size() * sizeof(real_t));
            cudaMemcpy(d_schwarz[d], schwarz_upper_bound_factors.device_ptr(), schwarz_upper_bound_factors.size() * sizeof(real_t), cudaMemcpyDefault);
            // Auxiliary Schwarz factors
            cudaMalloc(&d_aux_schwarz[d], auxiliary_schwarz_upper_bound_factors.size() * sizeof(real_t));
            cudaMemcpy(d_aux_schwarz[d], auxiliary_schwarz_upper_bound_factors.device_ptr(), auxiliary_schwarz_upper_bound_factors.size() * sizeof(real_t), cudaMemcpyDefault);
            // Boys grid
            cudaMalloc(&d_boys[d], n_boys * sizeof(real_t));
            cudaMemcpy(d_boys[d], hf_.get_boys_grid().device_ptr(), n_boys * sizeof(real_t), cudaMemcpyDefault);
        }
    }

    // ---- Step 3: Allocate B_local on each GPU and zero-initialize ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        size_t local_size = (size_t)naux_local_[d] * nbas2;
        if (!d_B_local_[d])
            cudaMalloc(&d_B_local_[d], local_size * sizeof(double));
        cudaMemset(d_B_local_[d], 0, local_size * sizeof(double));
    }

    // ---- Step 4: Chunked 3c2e + DGEMM on each GPU ----
    // For each auxiliary shell type: all GPUs compute the same 3c2e chunk redundantly,
    // then each GPU accumulates its P_local rows of B via DGEMM with L⁻¹.
    for (int c = 0; c < n_aux_types; c++) {
        const size_t Q_c_start = aux_type_basis_start_[c];
        const int nfunc_c = aux_type_nfunc_[c];

        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cublasHandle_t handle = mgr.cublas(d);

            // Allocate and zero chunk buffer
            real_t* d_chunk = nullptr;
            cudaMalloc(&d_chunk, (size_t)nfunc_c * nbas2 * sizeof(real_t));
            cudaMemset(d_chunk, 0, (size_t)nfunc_c * nbas2 * sizeof(real_t));

            // Compute 3c2e for this aux type into chunk (pointer-offset trick)
            gpu::computeThreeCenterERIs_for_aux_type(
                shell_type_infos, shell_pair_type_infos,
                d_pshells[d], d_cgto_norms[d],
                auxiliary_shell_type_infos_,
                d_aux_pshells[d], d_aux_cgto_norms[d],
                d_chunk,
                d_shell_pairs[d],
                nbas, naux,
                d_boys[d],
                d_schwarz[d], d_aux_schwarz[d],
                schwarz_screening_threshold,
                c, Q_c_start, nfunc_c);

            // DGEMM: B_local += L⁻¹_local_rows[:, Q_c] × 3c_chunk
            // cuBLAS col-major: C = A * B where
            //   A = 3c_chunk^T [nbas² × nfunc_c], lda=nbas²
            //   B = L⁻¹ submatrix [nfunc_c × naux_local], ldb=naux
            //   C = B_local^T [nbas² × naux_local], ldc=nbas²
            const double one = 1.0;
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)nbas2, naux_local_[d], nfunc_c,
                &one,
                d_chunk, (int)nbas2,
                &d_L_inv[d][P_start_[d] * naux + Q_c_start], naux,
                &one,
                d_B_local_[d], (int)nbas2);

            cudaFree(d_chunk);
        }
    }
    mgr.sync_all();

    // ---- Step 5: Cleanup replicated data on non-zero GPUs ----
    for (int d = 1; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_L_inv[d]);
        cudaFree(d_pshells[d]);
        cudaFree(d_cgto_norms[d]);
        cudaFree(d_aux_pshells[d]);
        cudaFree(d_aux_cgto_norms[d]);
        cudaFree(d_shell_pairs[d]);
        cudaFree(d_schwarz[d]);
        cudaFree(d_aux_schwarz[d]);
        cudaFree(d_boys[d]);
    }
    // Free L⁻¹ on GPU 0 only if we allocated it locally (not cached)
    if (L_inv_owned) {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_L_inv_gpu0);
    }

    // Release full B on GPU 0 (no longer needed)
    {
        MultiGpuManager::DeviceGuard guard(0);
        if (intermediate_matrix_B_.rows() > 0) {
            size_t freed_mb = (size_t)naux * nbas2 * sizeof(real_t) / (1024 * 1024);
            intermediate_matrix_B_.release();
            std::cout << "[RI-Dist] Freed full B on GPU 0 (" << freed_mb << " MB)" << std::endl;
        }
    }

    scattered_ = true;
    allocate_per_device_workspace();

    // In stored mode, free cached L⁻¹ (no longer needed after B_local is built).
    // In direct mode, keep it for per-iteration rebuilds.
    if (!direct_mode_ && d_cached_L_inv_) {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaFree(d_cached_L_inv_);
        d_cached_L_inv_ = nullptr;
    }

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
    if (direct_mode_) {
        distributed_build_B();
    } else {
        if (!scattered_) distributed_build_B();
    }

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

    // Direct-RI mode: free B_local after Fock build (not stored between iterations)
    if (direct_mode_) {
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            if (d_B_local_[d]) { cudaFree(d_B_local_[d]); d_B_local_[d] = nullptr; }
        }
        scattered_ = false;  // Force rebuild next iteration
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

} // namespace gansu

#endif // GANSU_MULTI_GPU
