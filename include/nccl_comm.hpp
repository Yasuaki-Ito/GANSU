/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file nccl_comm.hpp
 * @brief NCCL communication wrappers for multi-GPU collective operations
 *
 * Provides type-safe AllReduce and Broadcast operations that work with
 * MultiGpuManager's per-device NCCL communicators and streams.
 *
 * All operations are asynchronous (enqueued to comm_stream).
 * Call cudaStreamSynchronize(comm_stream) or MultiGpuManager::sync_all()
 * to ensure completion.
 */

#pragma once

#ifdef GANSU_MULTI_GPU

#include "multi_gpu_manager.hpp"
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace gansu::nccl {

namespace detail {
    inline ncclDataType_t nccl_dtype(double*) { return ncclFloat64; }
    inline ncclDataType_t nccl_dtype(float*)  { return ncclFloat32; }
    inline ncclDataType_t nccl_dtype(int*)    { return ncclInt32; }

    inline void check(ncclResult_t result, const char* func) {
        if (result != ncclSuccess) {
            throw std::runtime_error(
                std::string("[NCCL] ") + func + " failed: " + ncclGetErrorString(result));
        }
    }
}

/**
 * @brief AllReduce across all devices.
 *
 * Each device contributes sendbuf and receives the reduced result in recvbuf.
 * sendbuf and recvbuf may be the same pointer (in-place).
 *
 * Must be called from a thread that has set the correct CUDA device
 * (e.g., inside a DeviceGuard or after cudaSetDevice).
 *
 * @param sendbuf  Device pointer on device_id
 * @param recvbuf  Device pointer on device_id (may == sendbuf)
 * @param count    Number of elements
 * @param op       Reduction operation (ncclSum, ncclMax, etc.)
 * @param device_id  Which device this call is for
 * @param stream   CUDA stream to enqueue on (typically comm_stream)
 */
template<typename T>
void all_reduce(const T* sendbuf, T* recvbuf, size_t count,
                ncclRedOp_t op, int device_id, cudaStream_t stream)
{
    auto& mgr = MultiGpuManager::instance();
    detail::check(
        ncclAllReduce(sendbuf, recvbuf, count,
                      detail::nccl_dtype((T*)nullptr), op,
                      mgr.nccl_comm(device_id), stream),
        "ncclAllReduce");
}

/**
 * @brief Broadcast from root device to all devices.
 *
 * @param buf       Device pointer (input on root, output on all)
 * @param count     Number of elements
 * @param root      Root device ID
 * @param device_id Which device this call is for
 * @param stream    CUDA stream
 */
template<typename T>
void broadcast(T* buf, size_t count, int root, int device_id, cudaStream_t stream)
{
    auto& mgr = MultiGpuManager::instance();
    detail::check(
        ncclBroadcast(buf, buf, count,
                      detail::nccl_dtype((T*)nullptr), root,
                      mgr.nccl_comm(device_id), stream),
        "ncclBroadcast");
}

/**
 * @brief Group start for batched NCCL operations.
 * Enclose multiple NCCL calls between group_start/group_end
 * to fuse them into a single communication round.
 */
inline void group_start() { detail::check(ncclGroupStart(), "ncclGroupStart"); }
inline void group_end()   { detail::check(ncclGroupEnd(),   "ncclGroupEnd"); }

} // namespace gansu::nccl

#endif // GANSU_MULTI_GPU
