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
 * @file steom_barh_cache.cu
 * @brief SteomBarHCache::free() — releases the shared bar-H device buffers.
 *
 * See include/steom_barh_cache.hpp for the (A) build_dressed de-duplication
 * design. free() is the single owner-side release point; it is called by the
 * driver (compute_steom_ccsd_impl) at the end of the STEOM stage after the
 * borrowing STEOM operator has been destroyed.
 */

#include "steom_barh_cache.hpp"
#include "device_host_memory.hpp"

namespace gansu {

void SteomBarHCache::migrate_to_device(int src_dev, int dst_dev) {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || src_dev == dst_dev) return;
    int cur = 0;
    cudaGetDevice(&cur);
    cudaSetDevice(dst_dev);
    const size_t NO = (size_t)nocc, NV = (size_t)nvir;
    auto move = [&](real_t*& p, size_t n) {
        if (!p) return;
        real_t* np = nullptr;
        tracked_cudaMalloc(&np, n * sizeof(real_t));         // allocated on dst_dev
        cudaMemcpyPeer(np, dst_dev, p, src_dev, n * sizeof(real_t));
        tracked_cudaFree(p);                                  // free the src_dev copy
        p = np;
    };
    move(d_Loo,   NO*NO);
    move(d_Lvv,   NV*NV);
    move(d_Fov,   NO*NV);
    move(d_Woooo, NO*NO*NO*NO);
    move(d_Wooov, NO*NO*NO*NV);
    move(d_Wovov, NO*NV*NO*NV);
    move(d_Wovvo, NO*NV*NV*NO);
    move(d_Wovoo, NO*NV*NO*NO);
    move(d_Wvovv, NV*NO*NV*NV);
    move(d_Wvvvv, NV*NV*NV*NV);   // null-safe (canonical-skip)
    move(d_Wvvvo, NV*NV*NV*NO);
    cudaDeviceSynchronize();
    cudaSetDevice(cur);
#else
    (void)src_dev; (void)dst_dev;
#endif
}

#ifndef GANSU_CPU_ONLY
// Peer-migrate one buffer p (on src_dev) to dst_dev; free the src copy, repoint.
// Caller must have cudaSetDevice(dst_dev) active (so the new alloc lands there).
static void barh_move_one(real_t*& p, size_t n, int src_dev, int dst_dev) {
    if (!p || src_dev == dst_dev) return;
    real_t* np = nullptr;
    tracked_cudaMalloc(&np, n * sizeof(real_t));               // on dst_dev (current)
    cudaMemcpyPeer(np, dst_dev, p, src_dev, n * sizeof(real_t));
    tracked_cudaFree(p);
    p = np;
}
#endif

void SteomBarHCache::migrate_ip_to(int dst) {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || ip_dev == dst) return;
    const int src = ip_dev; int cur = 0; cudaGetDevice(&cur); cudaSetDevice(dst);
    const size_t NO = (size_t)nocc, NV = (size_t)nvir;
    barh_move_one(d_Loo,   NO*NO,          src, dst);
    barh_move_one(d_Lvv,   NV*NV,          src, dst);
    barh_move_one(d_Fov,   NO*NV,          src, dst);
    barh_move_one(d_Woooo, NO*NO*NO*NO,    src, dst);
    barh_move_one(d_Wooov, NO*NO*NO*NV,    src, dst);
    barh_move_one(d_Wovov, NO*NV*NO*NV,    src, dst);
    barh_move_one(d_Wovvo, NO*NV*NV*NO,    src, dst);
    barh_move_one(d_Wovoo, NO*NV*NO*NO,    src, dst);
    cudaDeviceSynchronize(); cudaSetDevice(cur);
    ip_dev = dst;
#else
    (void)dst;
#endif
}

void SteomBarHCache::migrate_ea_to(int dst) {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || ea_dev == dst) return;
    const int src = ea_dev; int cur = 0; cudaGetDevice(&cur); cudaSetDevice(dst);
    const size_t NO = (size_t)nocc, NV = (size_t)nvir;
    barh_move_one(d_Wvovv, NV*NO*NV*NV,    src, dst);
    barh_move_one(d_Wvvvv, NV*NV*NV*NV,    src, dst);   // null-safe (canonical-skip)
    barh_move_one(d_Wvvvo, NV*NV*NV*NO,    src, dst);
    cudaDeviceSynchronize(); cudaSetDevice(cur);
    ea_dev = dst;
#else
    (void)dst;
#endif
}

void SteomBarHCache::free() {
    // tracked_cudaFree tolerates nullptr (no-op) and maps to host free() in
    // CPU-only builds, so unconditional calls are safe in both modes.
    tracked_cudaFree(d_Loo);
    tracked_cudaFree(d_Lvv);
    tracked_cudaFree(d_Fov);
    tracked_cudaFree(d_Woooo);
    tracked_cudaFree(d_Wooov);
    tracked_cudaFree(d_Wovov);
    tracked_cudaFree(d_Wovvo);
    tracked_cudaFree(d_Wovoo);
    tracked_cudaFree(d_Wvovv);
    tracked_cudaFree(d_Wvvvv);
    tracked_cudaFree(d_Wvvvo);
    reset_pointers();
}

}  // namespace gansu
