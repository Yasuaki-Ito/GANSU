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
 * Strategy:
 *   1. Build full B on GPU 0 (existing precomputation, unchanged)
 *   2. Scatter P-local slices to all GPUs
 *   3. J build: local W + weighted_sum → AllReduce(J)
 *   4. K build: local batched_gemm → pack → gemm → AllReduce(K)
 *   5. F = H + J - 0.5K (on GPU 0)
 */

#ifdef GANSU_MULTI_GPU

#include "rhf.hpp"
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "device_host_memory.hpp"
#include <iostream>
#include <iomanip>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

// Fock assembly kernel: F = H + J - 0.5*K
__global__ void distributed_fock_assemble_kernel(
    const double* __restrict__ H,
    const double* __restrict__ J,
    const double* __restrict__ K,
    double* __restrict__ F,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    F[idx] = H[idx] + J[idx] - 0.5 * K[idx];
}

// J accumulation: J[μν] += W[P] * B[P, μν]  for local P range
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

// Pack X[P,μ,i] → X_packed[μ, P*nocc+i]
__global__ void distributed_pack_X_kernel(
    const double* __restrict__ X,
    double* __restrict__ X_packed,
    int nbas, int naux_local, int nocc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = naux_local * nbas * nocc;
    if (idx >= total) return;
    int i = idx % nocc;
    int mu = (idx / nocc) % nbas;
    int P = idx / (nocc * nbas);
    X_packed[(size_t)mu * naux_local * nocc + P * nocc + i] = X[idx];
}

// ============================================================
//  Constructor
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
    if (d_J_local_[0]) return;  // already allocated
    const size_t nbas2 = (size_t)num_basis_ * num_basis_;
    auto& mgr = MultiGpuManager::instance();

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
//  Scatter B from GPU 0 to all GPUs
// ============================================================
void ERI_RI_Distributed_RHF::scatter_B() {
    if (scattered_) return;
    const size_t nbas2 = (size_t)num_basis_ * num_basis_;
    const real_t* d_B_full = intermediate_matrix_B_.device_ptr();  // on GPU 0

    for (int d = 0; d < num_gpus_; d++) {
        size_t local_size = (size_t)naux_local_[d] * nbas2;
        size_t offset = P_start_[d] * nbas2;

        MultiGpuManager::DeviceGuard guard(d);
        cudaMalloc(&d_B_local_[d], local_size * sizeof(double));

        if (d == 0) {
            // GPU 0: copy from full B (same device)
            cudaMemcpy(d_B_local_[d], d_B_full + offset,
                       local_size * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            // Other GPUs: peer copy from GPU 0
            cudaMemcpy(d_B_local_[d], d_B_full + offset,
                       local_size * sizeof(double), cudaMemcpyDefault);
        }
    }

    scattered_ = true;
    allocate_per_device_workspace();

    // Report memory per device
    for (int d = 0; d < num_gpus_; d++) {
        double mb = (double)naux_local_[d] * nbas2 * sizeof(double) / (1024.0 * 1024.0);
        std::cout << "  [GPU " << d << "] B_local: " << std::fixed << std::setprecision(1)
                  << mb << " MB (" << naux_local_[d] << " aux)" << std::endl;
    }
}

// ============================================================
//  Distributed Fock build
// ============================================================
void ERI_RI_Distributed_RHF::compute_fock_matrix() {
    // First call: scatter B
    if (!scattered_) scatter_B();

    const int nbas = num_basis_;
    const int nocc = num_occ_;
    const size_t nbas2 = (size_t)nbas * nbas;
    const int threads = 256;
    auto& mgr = MultiGpuManager::instance();

    // D, C, H are on GPU 0 — replicate to all GPUs
    const real_t* d_D_gpu0 = rhf_.get_density_matrix().device_ptr();
    const real_t* d_C_gpu0 = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_H_gpu0 = rhf_.get_core_hamiltonian_matrix().device_ptr();

    // Allocate per-device D and C copies (small, temporary)
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

    // ---- J build: local W + weighted sum → AllReduce(J) ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t handle = mgr.cublas(d);
        int nl = naux_local_[d];
        const double one = 1.0, zero = 0.0;

        // W_local[P] = Σ_{μν} B_local[P,μν] × D[μν]
        cublasDgemv(handle, CUBLAS_OP_T,
                    (int)nbas2, nl,
                    &one, d_B_local_[d], (int)nbas2,
                    d_D[d], 1,
                    &zero, d_W_local_[d], 1);

        // J_local[μν] = Σ_{P_local} W_local[P] × B_local[P,μν]
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_J_accumulate_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(
            d_J_local_[d], d_B_local_[d], d_W_local_[d], nbas, nl);
    }
    mgr.sync_all();

    // AllReduce J across all GPUs
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(d_J_local_[d], d_J_local_[d], nbas2,
                         ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();

    // ---- K build: local batched DGEMM → pack → DGEMM → AllReduce(K) ----
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t handle = mgr.cublas(d);
        cublasSetStream(handle, mgr.compute_stream(d));
        int nl = naux_local_[d];
        const double one = 1.0, zero = 0.0, two = 2.0;

        // X_local[P,μ,i] = Σ_ν C[ν,i] × B_local[P,μ,ν]  (batched over P_local)
        cudaMemset(d_K_local_[d], 0, nbas2 * sizeof(double));
        cublasDgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            nocc, nbas, nbas,
            &one,
            d_C[d], nbas, 0LL,                           // C [nbas × nbas], same for all P
            d_B_local_[d], nbas, (long long)nbas2,        // B_local [nbas × nbas] per P
            &zero,
            d_X_local_[d], nocc, (long long)(nbas * nocc), // X [nocc × nbas] per P
            nl);

        // Pack: X_packed[μ, P*nocc+i] = X[P*nbas*nocc + μ*nocc + i]
        int total_xpack = nl * nbas * nocc;
        int blk = (total_xpack + threads - 1) / threads;
        distributed_pack_X_kernel<<<blk, threads, 0, mgr.compute_stream(d)>>>(
            d_X_local_[d], d_X_packed_local_[d], nbas, nl, nocc);

        // K_local = 2 × X_packed^T × X_packed
        // [nbas × (nl*nocc)]^T × [nbas × (nl*nocc)] → [nbas × nbas]
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

    // AllReduce K across all GPUs
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(d_K_local_[d], d_K_local_[d], nbas2,
                         ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();

    // ---- Fock assembly on GPU 0: F = H + J - 0.5*K ----
    {
        MultiGpuManager::DeviceGuard guard(0);
        // Wait for comm to finish
        cudaStreamSynchronize(mgr.comm_stream(0));

        real_t* d_F = rhf_.get_fock_matrix().device_ptr();
        int blk = ((int)nbas2 + threads - 1) / threads;
        distributed_fock_assemble_kernel<<<blk, threads>>>(
            d_H_gpu0, d_J_local_[0], d_K_local_[0], d_F, nbas);
        cudaDeviceSynchronize();
    }

    // Free per-device D/C copies (not GPU 0)
    for (int d = 1; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaFree(d_D[d]);
        cudaFree(d_C[d]);
    }
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
