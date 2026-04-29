/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

#include "multi_gpu_manager.hpp"
#include <iostream>
#include <algorithm>

namespace gansu {

void MultiGpuManager::initialize(int requested_devices) {
    if (initialized_) return;

#ifdef GANSU_CPU_ONLY
    num_devices_ = 0;
    initialized_ = true;
    return;
#else
    // Detect available GPUs
    int available = 0;
    cudaGetDeviceCount(&available);
    if (available <= 0) {
        num_devices_ = 0;
        initialized_ = true;
        std::cout << "[MultiGPU] No CUDA devices found." << std::endl;
        return;
    }

    num_devices_ = (requested_devices > 0) ? std::min(requested_devices, available) : available;

    // Check peer access (NVLink/NVSwitch)
    for (int i = 0; i < num_devices_; i++) {
        for (int j = 0; j < num_devices_; j++) {
            if (i == j) continue;
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access) {
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // Create per-device handles and streams
    cublas_handles_.resize(num_devices_);
    cusolver_handles_.resize(num_devices_);
    compute_streams_.resize(num_devices_);
    comm_streams_.resize(num_devices_);

    for (int d = 0; d < num_devices_; d++) {
        cudaSetDevice(d);

        cublasCreate(&cublas_handles_[d]);
        cusolverDnCreate(&cusolver_handles_[d]);
        cudaStreamCreate(&compute_streams_[d]);
        cudaStreamCreate(&comm_streams_[d]);

        // Bind cuBLAS to compute stream
        cublasSetStream(cublas_handles_[d], compute_streams_[d]);
    }

#ifdef GANSU_MULTI_GPU
    // Initialize NCCL
    nccl_comms_.resize(num_devices_);
    std::vector<int> dev_list(num_devices_);
    for (int d = 0; d < num_devices_; d++) dev_list[d] = d;

    ncclResult_t nccl_result = ncclCommInitAll(nccl_comms_.data(), num_devices_, dev_list.data());
    if (nccl_result != ncclSuccess) {
        std::cerr << "[MultiGPU] NCCL init failed: " << ncclGetErrorString(nccl_result) << std::endl;
        // Fallback to single GPU
        num_devices_ = 1;
    }
#endif

    // Restore to device 0
    cudaSetDevice(0);
    initialized_ = true;

    std::cout << "[MultiGPU] Initialized: " << num_devices_ << " device(s)";
    if (num_devices_ > 1) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << " (" << prop.name << ")";
#ifdef GANSU_MULTI_GPU
        std::cout << ", NCCL enabled";
#endif
    }
    std::cout << std::endl;
#endif // GANSU_CPU_ONLY
}

void MultiGpuManager::finalize() {
    if (!initialized_) return;

#ifndef GANSU_CPU_ONLY
#ifdef GANSU_MULTI_GPU
    for (int d = 0; d < num_devices_; d++) {
        if (nccl_comms_.size() > (size_t)d && nccl_comms_[d])
            ncclCommDestroy(nccl_comms_[d]);
    }
    nccl_comms_.clear();
#endif

    for (int d = 0; d < num_devices_; d++) {
        cudaSetDevice(d);
        if (cublas_handles_.size() > (size_t)d && cublas_handles_[d])
            cublasDestroy(cublas_handles_[d]);
        if (cusolver_handles_.size() > (size_t)d && cusolver_handles_[d])
            cusolverDnDestroy(cusolver_handles_[d]);
        if (compute_streams_.size() > (size_t)d && compute_streams_[d])
            cudaStreamDestroy(compute_streams_[d]);
        if (comm_streams_.size() > (size_t)d && comm_streams_[d])
            cudaStreamDestroy(comm_streams_[d]);
    }

    cublas_handles_.clear();
    cusolver_handles_.clear();
    compute_streams_.clear();
    comm_streams_.clear();

    cudaSetDevice(0);
#endif

    num_devices_ = 0;
    initialized_ = false;
}

void MultiGpuManager::sync_all() {
#ifndef GANSU_CPU_ONLY
    for (int d = 0; d < num_devices_; d++) {
        cudaSetDevice(d);
        cudaDeviceSynchronize();
    }
    cudaSetDevice(0);
#endif
}

} // namespace gansu
