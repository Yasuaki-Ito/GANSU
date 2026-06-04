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
