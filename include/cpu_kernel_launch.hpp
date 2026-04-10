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

/**
 * @file cpu_kernel_launch.hpp
 * @brief CPU shim to launch CUDA-style kernels as OpenMP loops.
 *
 * In hybrid builds (`ENABLE_GPU=ON` compiled + `--cpu` runtime), each CUDA
 * kernel has a `__global__` qualifier that expands to nothing under
 * `GANSU_CPU_ONLY`, so the kernel is just a regular host function whose body
 * reads `blockIdx.x`, `threadIdx.x`, `blockDim.x` from the thread-local stubs
 * defined in cuda_compat.hpp.
 *
 * This header provides launch macros that iterate a flat range of thread ids
 * in an OpenMP parallel loop, assigning the thread-local stubs on each
 * iteration so the kernel body sees `id = threadIdx.x + blockIdx.x*blockDim.x`
 * exactly like on GPU.
 *
 * Usage (inside an `if (!gpu_available()) { ... }` branch):
 *
 *     GANSU_CPU_LAUNCH_1D(my_kernel, num_threads_total,
 *                         arg0, arg1, arg2, ...);
 *
 * Only CUDA launch patterns that collapse to a single 1-D id (i.e. every
 * gradient/hessian kernel in GANSU) need the macro; anything more elaborate
 * should be written as a plain OpenMP loop directly.
 */

#pragma once

#ifdef GANSU_CPU_ONLY

#include "cuda_compat.hpp"

// Launch a 1-D CUDA kernel as an OpenMP parallel loop.  We use blockDim.x = 1
// and map the flat thread id directly to blockIdx.x, so the kernel's
// `id = threadIdx.x + blockIdx.x*blockDim.x` recovers `id = _i`.
//
// `num_threads_total` is the number of thread ids the kernel expects — it
// must match the `num_threads` parameter the kernel uses internally for its
// bounds check, so we replicate the exact GPU launch geometry.
#define GANSU_CPU_LAUNCH_1D(kernel, num_threads_total, ...)                    \
    do {                                                                       \
        const long long _gansu_total = (long long)(num_threads_total);         \
        _Pragma("omp parallel for schedule(static)")                           \
        for (long long _gansu_id = 0; _gansu_id < _gansu_total; ++_gansu_id) { \
            ::blockDim.x  = 1u;                                                \
            ::blockIdx.x  = (unsigned int)(_gansu_id);                         \
            ::threadIdx.x = 0u;                                                \
            kernel(__VA_ARGS__);                                               \
        }                                                                      \
    } while (0)

#endif // GANSU_CPU_ONLY
