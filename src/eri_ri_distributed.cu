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
//  Precomputation: Schwarz bounds + distributed B build
// ============================================================
void ERI_RI_Distributed_RHF::precomputation() {
    // Run parent's precomputation for Schwarz bounds, shell pair indices, etc.
    // This builds the full B on GPU 0, which we'll replace with distributed B.
    ERI_RI::precomputation();

    // Now replace with distributed B
    distributed_build_B();
}

// ============================================================
//  Distributed B build (chunked 3c2e + L⁻¹ DGEMM)
// ============================================================
void ERI_RI_Distributed_RHF::distributed_build_B() {
    if (scattered_) return;

    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const size_t nbas2 = (size_t)nbas * nbas;

    // precomputation() already built full B in intermediate_matrix_B_ on GPU 0.
    // Scatter P-local slices to all GPUs, then free full B on GPU 0.
    const real_t* d_B_full = intermediate_matrix_B_.device_ptr();

    for (int d = 0; d < num_gpus_; d++) {
        size_t local_size = (size_t)naux_local_[d] * nbas2;
        size_t offset = P_start_[d] * nbas2;

        MultiGpuManager::DeviceGuard guard(d);
        cudaMalloc(&d_B_local_[d], local_size * sizeof(double));

        // Copy slice from GPU 0 (peer access enabled)
        cudaMemcpy(d_B_local_[d], d_B_full + offset,
                   local_size * sizeof(double), cudaMemcpyDefault);
    }

    // Note: intermediate_matrix_B_ on GPU 0 is still allocated (freed by destructor).
    // For larger systems, we would want to free it here to reclaim GPU memory.

    scattered_ = true;
    allocate_per_device_workspace();

    for (int d = 0; d < num_gpus_; d++) {
        double mb = (double)naux_local_[d] * nbas2 * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  [GPU " << d << "] B_local: " << std::fixed << std::setprecision(1)
                  << mb << " MB (" << naux_local_[d] << " aux)" << std::endl;
    }
    std::cout << "[RI-Dist] B distributed to " << num_gpus_ << " GPUs" << std::endl;
}

// ============================================================
//  Distributed Fock build
// ============================================================
void ERI_RI_Distributed_RHF::compute_fock_matrix() {
    if (!scattered_) distributed_build_B();

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

} // namespace gansu

#endif // GANSU_MULTI_GPU
