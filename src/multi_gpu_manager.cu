/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

#include "multi_gpu_manager.hpp"
#include "gpu_manager.hpp"
#include <iostream>
#include <algorithm>
#ifdef GANSU_MPI
#include <mpi.h>
#endif

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

#ifdef GANSU_MPI
    {
        int mpi_on = 0;
        MPI_Initialized(&mpi_on);
        if (mpi_on) {
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        }
    }
#endif

    // In MPI mode each rank owns exactly one GPU (pinned via CUDA_VISIBLE_DEVICES
    // by the launcher), seen as device 0. The cross-rank distribution is handled
    // by a world NCCL communicator below, not by the intra-process device loop.
    if (is_mpi()) {
        num_devices_ = 1;
    } else {
        num_devices_ = (requested_devices > 0) ? std::min(requested_devices, available) : available;
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
        cublasSetStream(cublas_handles_[d], compute_streams_[d]);
    }

#ifdef GANSU_MULTI_GPU
#ifdef GANSU_MPI
    if (is_mpi()) {
        // MPI mode: one NCCL communicator per rank, spanning the whole world.
        // rank 0 generates the unique id, MPI_Bcast distributes it, every rank
        // joins with ncclCommInitRank. After CUDA_VISIBLE_DEVICES pinning each
        // rank's GPU is device 0, so the device-0 code paths run per-rank.
        nccl_comms_.resize(1);
        ncclUniqueId id;
        if (world_rank_ == 0) {
            ncclResult_t gr = ncclGetUniqueId(&id);
            if (gr != ncclSuccess)
                std::cerr << "[MultiGPU] ncclGetUniqueId failed: "
                          << ncclGetErrorString(gr) << std::endl;
        }
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        cudaSetDevice(0);
        ncclResult_t nccl_result =
            ncclCommInitRank(&nccl_comms_[0], world_size_, id, world_rank_);
        if (nccl_result != ncclSuccess) {
            std::cerr << "[MultiGPU] ncclCommInitRank failed (rank " << world_rank_
                      << "): " << ncclGetErrorString(nccl_result) << std::endl;
        } else {
            // Cross-rank AllReduce self-test: each rank contributes 1.0; the sum
            // over the world must equal world_size. Proves the comm is wired up.
            double* d_buf = nullptr;
            cudaMalloc(&d_buf, sizeof(double));
            double one = 1.0;
            cudaMemcpy(d_buf, &one, sizeof(double), cudaMemcpyHostToDevice);
            ncclAllReduce(d_buf, d_buf, 1, ncclFloat64, ncclSum,
                          nccl_comms_[0], compute_streams_[0]);
            cudaStreamSynchronize(compute_streams_[0]);
            double sum = 0.0;
            cudaMemcpy(&sum, d_buf, sizeof(double), cudaMemcpyDeviceToHost);
            cudaFree(d_buf);
            if (world_rank_ == 0) {
                std::cout << "[MPI] NCCL world AllReduce self-test: sum=" << sum
                          << " (expected " << world_size_ << ") "
                          << (sum == (double)world_size_ ? "OK" : "MISMATCH")
                          << std::endl;
            }
        }
    } else
#endif
    {
        // Single-process multi-GPU: build all communicators at once.
        nccl_comms_.resize(num_devices_);
        std::vector<int> dev_list(num_devices_);
        for (int d = 0; d < num_devices_; d++) dev_list[d] = d;

        ncclResult_t nccl_result = ncclCommInitAll(nccl_comms_.data(), num_devices_, dev_list.data());
        if (nccl_result != ncclSuccess) {
            std::cerr << "[MultiGPU] NCCL init failed: " << ncclGetErrorString(nccl_result) << std::endl;
            num_devices_ = 1;
        }
    }
#elif defined(GANSU_MPI)
    if (is_mpi()) {
        std::cerr << "[MultiGPU] WARNING: ENABLE_MPI is on but NCCL (ENABLE_MULTI_GPU) "
                     "is off — cross-rank GPU collectives are unavailable." << std::endl;
    }
#endif

    // Restore to device 0
    cudaSetDevice(0);
    initialized_ = true;

    if (is_mpi()) {
        // One line from the root rank only — each rank owns a single GPU.
        if (world_rank_ == 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "[MultiGPU] Initialized: " << world_size_ << " MPI rank(s) x 1 GPU ("
                      << prop.name << ")";
#ifdef GANSU_MULTI_GPU
            std::cout << ", NCCL world comm enabled";
#endif
            std::cout << std::endl;
        }
    } else {
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
    }
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
