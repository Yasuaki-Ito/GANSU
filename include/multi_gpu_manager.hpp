/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file multi_gpu_manager.hpp
 * @brief Per-device GPU handle manager for multi-GPU execution
 *
 * Manages cuBLAS/cuSOLVER handles, CUDA streams, and NCCL communicators
 * for all available GPU devices on a single node.
 *
 * Coexists with the existing thread-local GPUHandle (gpu_manager.hpp).
 * When num_devices() == 1, the distributed code path is not used and
 * the existing single-GPU code runs unchanged.
 *
 * Usage:
 *   auto& mgr = MultiGpuManager::instance();
 *   mgr.initialize();  // auto-detect GPUs, init NCCL
 *   for (int d = 0; d < mgr.num_devices(); d++) {
 *       MultiGpuManager::DeviceGuard guard(d);
 *       // ... operations on device d ...
 *   }
 */

#pragma once

#include <vector>
#include <utility>
#include <stdexcept>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#ifdef GANSU_MULTI_GPU
#include <nccl.h>
#endif
#endif

namespace gansu {

/**
 * @brief Compute the [start, end) range for a given device in an aux-axis partition.
 *
 * Distributes global_size elements as evenly as possible across num_devices.
 * First (global_size % num_devices) devices get one extra element.
 */
inline std::pair<size_t, size_t> aux_partition(size_t global_size, int num_devices, int device_id) {
    size_t base = global_size / num_devices;
    size_t remainder = global_size % num_devices;
    size_t start = device_id * base + std::min((size_t)device_id, remainder);
    size_t local_size = base + ((size_t)device_id < remainder ? 1 : 0);
    return {start, start + local_size};
}

class MultiGpuManager {
public:
    /**
     * @brief Get the singleton instance.
     */
    static MultiGpuManager& instance() {
        static MultiGpuManager mgr;
        return mgr;
    }

    /**
     * @brief Initialize multi-GPU resources.
     * @param num_devices Number of GPUs to use (-1 = auto-detect all).
     *
     * Creates per-device cuBLAS/cuSOLVER handles, CUDA streams,
     * and NCCL communicators. Safe to call multiple times (no-op after first).
     */
    void initialize(int num_devices = -1);

    /**
     * @brief Finalize and release all resources.
     * Called automatically by destructor.
     */
    void finalize();

    /// Number of active GPU devices (per-process; 1 per rank in MPI mode).
    int num_devices() const { return num_devices_; }

    /// Whether intra-process multi-GPU mode is active (num_devices > 1).
    /// NOTE: false in MPI mode (each rank owns a single GPU) — use is_mpi()
    /// to detect cross-rank distribution.
    bool is_distributed() const { return num_devices_ > 1; }

    /// This process's rank in MPI_COMM_WORLD (0 in non-MPI / single-rank runs).
    int world_rank() const { return world_rank_; }
    /// Total number of MPI ranks (1 in non-MPI / single-rank runs).
    int world_size() const { return world_size_; }
    /// Whether running under MPI with more than one rank (cross-rank NCCL world).
    bool is_mpi() const { return world_size_ > 1; }

    /// cuBLAS handle for device d.
    cublasHandle_t cublas(int d) const { return cublas_handles_[d]; }

    /// cuSOLVER handle for device d.
    cusolverDnHandle_t cusolver(int d) const { return cusolver_handles_[d]; }

    /// Compute stream for device d (used for DGEMM, kernels).
    cudaStream_t compute_stream(int d) const { return compute_streams_[d]; }

    /// Communication stream for device d (used for NCCL).
    cudaStream_t comm_stream(int d) const { return comm_streams_[d]; }

#ifdef GANSU_MULTI_GPU
    /// NCCL communicator for device d.
    ncclComm_t nccl_comm(int d) const { return nccl_comms_[d]; }
#endif

    /// Synchronize all devices.
    void sync_all();

    /**
     * @brief RAII guard for cudaSetDevice.
     * Sets device on construction, restores previous device on destruction.
     */
    struct DeviceGuard {
        DeviceGuard(int device_id) {
#ifndef GANSU_CPU_ONLY
            cudaGetDevice(&prev_device_);
            if (device_id != prev_device_)
                cudaSetDevice(device_id);
#endif
        }
        ~DeviceGuard() {
#ifndef GANSU_CPU_ONLY
            cudaSetDevice(prev_device_);
#endif
        }
        DeviceGuard(const DeviceGuard&) = delete;
        DeviceGuard& operator=(const DeviceGuard&) = delete;
    private:
        int prev_device_ = 0;
    };

    ~MultiGpuManager() { finalize(); }

private:
    MultiGpuManager() = default;
    MultiGpuManager(const MultiGpuManager&) = delete;
    MultiGpuManager& operator=(const MultiGpuManager&) = delete;

    bool initialized_ = false;
    int num_devices_ = 0;
    int world_rank_ = 0;   ///< MPI world rank (0 if not under MPI).
    int world_size_ = 1;   ///< MPI world size (1 if not under MPI).

    std::vector<cublasHandle_t> cublas_handles_;
    std::vector<cusolverDnHandle_t> cusolver_handles_;
    std::vector<cudaStream_t> compute_streams_;
    std::vector<cudaStream_t> comm_streams_;

#ifdef GANSU_MULTI_GPU
    std::vector<ncclComm_t> nccl_comms_;
#endif
};

} // namespace gansu
