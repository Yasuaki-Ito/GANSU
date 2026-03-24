/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Template-specialized Rys ERI kernels + dispatch function.
// Compiled as a separate static library to isolate long template
// compilation from gpu_manager.cu.

#include "int2e.hpp"
#include "rys_eri_specialized.hpp"
#include <cstdlib>
#include <string>

namespace gansu::gpu {

// --- Environment variable toggle ---
static bool use_md_kernel() {
    static int cached = -1;
    if (cached < 0) {
        const char* env = std::getenv("GANSU_ERI_KERNEL");
        cached = (env && std::string(env) == "md") ? 1 : 0;
    }
    return cached == 1;
}

// --- ERI kernel dispatch ---
eri_kernel_t get_eri_kernel(int a, int b, int c, int d) {
    // Canonical ordering
    if (a > b) { int t=a; a=b; b=t; }
    if (c > d) { int t=c; c=d; d=t; }
    if (a > c || (a == c && b > d)) { int t=a; a=c; c=t; t=b; b=d; d=t; }

    // S/P specialized kernels (always used)
    if (a == 0 && b == 0 && c == 0 && d == 0)      return ssss2e;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return sssp2e;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return sspp2e;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return spsp2e;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return sppp2e;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return pppp2e;

    // Legacy MD kernel
    else if (use_md_kernel()) return MD_1T1SP;

    // Template-specialized Rys kernels
    else {
        eri_kernel_t spec = get_rys_kernel_specialized(a, b, c, d);
        return spec ? spec : RysERI;
    }
}

} // namespace gansu::gpu
