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
 * @file steom_cluster_ground_cache.hpp
 * @brief (2026-09-07) Cache the cluster CCSD/bt-PNO ground T1/T2 in the EOM chain
 *        context on HOST-parked managed memory.
 *
 * The IP/EA/STEOM stages share one cluster ground. Caching it (a) lets the
 * checkpoint carry it (cc=y) so a resume is deterministic instead of re-using a
 * freshly recomputed — and, under the multi-GPU fan-out, slightly different —
 * ground, and (b) lets later stages copy it instead of re-deriving it. A plain
 * device copy costs nocc²·nvir²·8 B on the solve device (8.6 GB at MgO 374) and
 * tipped the IP operator build into OOM, so the cache lives on host-parked managed
 * memory: preferred location CPU, direct access from the current device, so
 * cudaMemcpy from it still works from any stage (same recipe as CCSD's v_oooo).
 */

#pragma once

#include <cstddef>
#include <iostream>

#include "types.hpp"
#include "eom_chain_context.hpp"
#include "device_host_memory.hpp"

namespace gansu {

inline void steom_cache_cluster_ground_host(EOMChainContext* ctx,
                                            const real_t* d_t1, const real_t* d_t2,
                                            size_t t1n, size_t t2n, real_t E) {
    if (!ctx) return;
    const size_t b1 = t1n * sizeof(real_t), b2 = t2n * sizeof(real_t);
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        int dev = 0; cudaGetDevice(&dev);
        cudaMemLocation lh{}; lh.type = cudaMemLocationTypeHost;   lh.id = 0;
        cudaMemLocation ld{}; ld.type = cudaMemLocationTypeDevice; ld.id = dev;
        tracked_cudaMallocManagedHost(&ctx->cc_t1, b1);
        tracked_cudaMallocManagedHost(&ctx->cc_t2, b2);
        cudaMemAdvise(ctx->cc_t1, b1, cudaMemAdviseSetPreferredLocation, lh);
        cudaMemAdvise(ctx->cc_t2, b2, cudaMemAdviseSetPreferredLocation, lh);
        cudaMemAdvise(ctx->cc_t1, b1, cudaMemAdviseSetAccessedBy, ld);
        cudaMemAdvise(ctx->cc_t2, b2, cudaMemAdviseSetAccessedBy, ld);
        cudaMemcpy(ctx->cc_t1, d_t1, b1, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ctx->cc_t2, d_t2, b2, cudaMemcpyDeviceToDevice);
        cudaMemPrefetchAsync(ctx->cc_t1, b1, lh, /*flags=*/0, /*stream=*/0);
        cudaMemPrefetchAsync(ctx->cc_t2, b2, lh, /*flags=*/0, /*stream=*/0);
        cudaDeviceSynchronize();
        std::cout << "  [STEOM] cluster ground T1/T2 cached on HOST-parked managed memory ("
                  << (b1 + b2) / 1e9 << " GB; device pages released)" << std::endl;
    } else
#endif
    {
        tracked_cudaMalloc(&ctx->cc_t1, b1);
        tracked_cudaMalloc(&ctx->cc_t2, b2);
        cudaMemcpy(ctx->cc_t1, d_t1, b1, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ctx->cc_t2, d_t2, b2, cudaMemcpyDeviceToDevice);
    }
    ctx->cc_t1n = t1n; ctx->cc_t2n = t2n; ctx->cc_E = E; ctx->cc_ground_cached = true;
}

} // namespace gansu
