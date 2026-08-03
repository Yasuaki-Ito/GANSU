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

#include "dmet_memory_policy.hpp"

#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

#include "gpu_manager.hpp"

namespace gansu {

namespace {

/// setenv that never overrides an explicitly exported value.
/// Returns true when this call is what set the variable.
bool set_if_unset(const char* name, const char* value) {
    if (std::getenv(name) != nullptr) return false;   // user's export wins
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, /*overwrite=*/0);
#endif
    return true;
}

std::string gb(size_t bytes) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f GB", (double)bytes / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

} // namespace

void dmet_apply_memory_policy(int n_emb, int n_emb_occ, int verbose) {
#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available() || n_emb <= 0) return;

    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (free_b == 0) return;

    const size_t n    = (size_t)n_emb;
    const size_t nvir = (size_t)(n_emb - n_emb_occ);

    // The two tensors that decide everything. Both are what the *unswitched*
    // code path would allocate as one contiguous block.
    const size_t dense_eri_bytes = n * n * n * n * sizeof(double);        // cluster MO-ERI
    const size_t vvvv_bytes      = nvir * nvir * nvir * nvir * sizeof(double);  // CCSD ladder

    // A tensor "does not fit" well before it exceeds free memory: the solve also
    // needs its working set, and one contiguous multi-GB block on a fragmented
    // heap fails earlier than the totals suggest. A quarter of free memory is
    // the largest single tensor we let the unswitched path attempt.
    const size_t budget = free_b / 4;

    const bool need_ri_block = dense_eri_bytes > budget;
    const bool need_ccsd_kit = vvvv_bytes      > budget;
    // The EA/STEOM intermediates scale with the same nvir³·nocc / nvir⁴ family as
    // the CCSD ladder, so they share the trigger.
    const bool need_ea_kit   = need_ccsd_kit;

    std::vector<std::string> applied;
    auto apply = [&](const char* name) {
        if (set_if_unset(name, "1")) applied.push_back(name);
    };

    if (need_ri_block) {
        // Pull cluster MO-ERI blocks from the cluster B (naux·n_emb²) instead of
        // materialising n_emb⁴. Numerically the same integrals.
        apply("GANSU_DMET_STEOM_RI_BLOCK");
    }
    if (need_ccsd_kit) {
        // Cluster CCSD from B, per-tile ladder, i-block/VR tiling.
        apply("GANSU_CCSD_RI_BNATIVE");
        apply("GANSU_CCSD_RI_LADDER_TILE");
        apply("GANSU_CCSD_OCCI");
        apply("GANSU_CCSD_VR_TILE");
    }
    if (need_ea_kit) {
        // EA-EOM build: RI ladder, W on host, Wvvvo assembled on host.
        apply("GANSU_EA_RI_LADDER");
        apply("GANSU_EA_W_HOST");
        apply("GANSU_EA_WVVVO_HOST_ASM");
        // Spread the STEOM operator build and keep the bar-H on its own device.
        apply("GANSU_STEOM_OPERATOR_DEVICE_BALANCING");
        apply("GANSU_DMET_STEOM_CLUSTER_GPU");
        apply("GANSU_STEOM_SHARE_BARH");
        apply("GANSU_IP_SIGMA_GEMM");
        if (set_if_unset("GANSU_STEOM_BARH_GPU", "3")) applied.push_back("GANSU_STEOM_BARH_GPU=3");
    }

    // Always report when switches were applied: a published energy must be
    // traceable to the layout it was computed with, and this is the only line
    // that records it. Stay quiet (unless verbose) when nothing changed.
    if (verbose > 0 || !applied.empty()) {
        std::cout << "  [DMET memory policy] n_emb=" << n_emb << " (nvir=" << nvir
                  << "), free=" << gb(free_b)
                  << " | dense cluster MO-ERI=" << gb(dense_eri_bytes)
                  << ", CCSD nvir^4=" << gb(vvvv_bytes) << std::endl;
        if (applied.empty()) {
            std::cout << "  [DMET memory policy] cluster fits the direct layout — "
                         "no layout switches needed." << std::endl;
        } else {
            std::cout << "  [DMET memory policy] auto-enabled " << applied.size()
                      << " layout switch(es) (bit-exact; export any of them to override):"
                      << std::endl;
            std::string line = "   ";
            for (const auto& s : applied) {
                if (line.size() + s.size() > 96) { std::cout << line << std::endl; line = "   "; }
                line += " " + s;
            }
            if (line.size() > 3) std::cout << line << std::endl;
        }
    }
#else
    (void)n_emb; (void)n_emb_occ; (void)verbose;
#endif
}

} // namespace gansu
