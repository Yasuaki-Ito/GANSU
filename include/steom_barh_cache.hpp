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
 * @file steom_barh_cache.hpp
 * @brief Shared dressed bar-H intermediate cache for DLPNO-STEOM-CCSD.
 *
 * (A) build_dressed de-duplication — env GANSU_STEOM_SHARE_BARH=1, default off.
 *
 * Motivation (naphthalene cc-pVDZ production profile, 2026-06-03):
 *   build_dressed_intermediates runs THREE times — once in IP-EOM (13s), once
 *   in EA-EOM (32s), once in STEOM (33s) — each rebuilding the SAME 11 bar-H
 *   tensors from the SAME inputs (d_t1/d_t2/d_eri_mo/eps over the same
 *   nocc_active×nvir space). A read-only investigation confirmed every
 *   intermediate is bit-identical across IP/EA/STEOM (same formula, same
 *   row-major layout, same inputs; nocc_active_==nocc_ since frozen core is
 *   pre-removed). The only mode-dependence is Wvvvv, which is elided
 *   (canonical_skip_wvvvv) consistently across EA + STEOM in the DLPNO-STEOM
 *   production path (both nullptr).
 *
 * Mechanism:
 *   - IP operator publishes its 8 intermediates (Loo, Lvv, Fov, Woooo, Wooov,
 *     Wovov, Wovvo, Wovoo) into this cache and relinquishes ownership (its dtor
 *     skips freeing them).
 *   - EA operator publishes the 3 it adds beyond IP (Wvovv, Wvvvv, Wvvvo).
 *   - STEOM operator, finding the cache complete(), borrows all 11 device
 *     pointers and skips its own build_dressed_intermediates entirely (its dtor
 *     skips freeing the borrowed pointers).
 *   - The driver (compute_steom_ccsd_impl) owns the cache lifetime: it lives on
 *     the HF object across the IP→EA→STEOM dispatch and is freed at the end.
 *
 * Numerically exact (bit-identical to the un-shared path) since the borrowed
 * device buffers are the very ones the un-shared STEOM build would reproduce.
 * Default off → existing behaviour byte-identical.
 *
 * Expected saving (naphthalene): the entire STEOM build_dressed (~33s) and the
 * EA/IP shared-subset rebuild become a single union build → ~78s → ~33s, i.e.
 * ~45s off the STEOM stage (post-HF wall −25%).
 *
 * NOTE: Wvvvv is legitimately nullptr under canonical-skip, so completeness is
 * tracked by the has_ip / has_ea flags rather than by pointer non-null-ness.
 */

#pragma once

#include "types.hpp"

namespace gansu {

/// Shared dressed bar-H intermediate cache (see file header). Owns the device
/// pointers once published; free() releases them (via tracked_cudaFree). All
/// pointers are over the (nocc × nvir) active-post-frozen space.
struct SteomBarHCache {
    // --- dims (set by the first publisher; consumers assert a match) ---
    int  nocc = 0;
    int  nvir = 0;
    bool canonical_skip_wvvvv = false;

    // --- owned device pointers (row-major layouts identical to the
    //     STEOMCCSDOperator d_* members; see steom_ccsd_operator.hpp) ---
    real_t* d_Loo   = nullptr;   // [nocc·nocc]
    real_t* d_Lvv   = nullptr;   // [nvir·nvir]
    real_t* d_Fov   = nullptr;   // [nocc·nvir]
    real_t* d_Woooo = nullptr;   // [nocc⁴]      (IP-side)
    real_t* d_Wooov = nullptr;   // [nocc³·nvir] (IP-side)
    real_t* d_Wovov = nullptr;   // [nocc·nvir·nocc·nvir]
    real_t* d_Wovvo = nullptr;   // [nocc·nvir·nvir·nocc]
    real_t* d_Wovoo = nullptr;   // [nocc·nvir·nocc·nocc] (IP-side)
    real_t* d_Wvovv = nullptr;   // [nvir·nocc·nvir·nvir] (EA-side)
    real_t* d_Wvvvv = nullptr;   // [nvir⁴] — nullptr under canonical-skip (EA-side)
    real_t* d_Wvvvo = nullptr;   // [nvir³·nocc] (EA-side)

    // --- population flags (Wvvvv may legitimately be nullptr) ---
    bool has_ip = false;   ///< IP published Loo/Lvv/Fov/Woooo/Wooov/Wovov/Wovvo/Wovoo
    bool has_ea = false;   ///< EA published Wvovv/Wvvvv/Wvvvo

    /// All 11 (modulo canonical-skip Wvvvv) are present → STEOM may borrow.
    bool complete() const { return has_ip && has_ea; }

    /// Reset to empty WITHOUT freeing (used when relinquishing/transferring).
    void reset_pointers() {
        d_Loo = d_Lvv = d_Fov = d_Woooo = d_Wooov = d_Wovov = nullptr;
        d_Wovvo = d_Wovoo = d_Wvovv = d_Wvvvv = d_Wvvvo = nullptr;
        has_ip = has_ea = false;
        nocc = nvir = 0;
        canonical_skip_wvvvv = false;
    }

    /// Free all owned device pointers (tracked_cudaFree) and reset. Implemented
    /// in src/steom_barh_cache.cu where the GPU allocator is visible.
    void free();

    /// Peer-migrate every owned buffer from src_dev to dst_dev (allocate on
    /// dst_dev, cudaMemcpyPeer, free the src copy, repoint). Used to relocate the
    /// borrowed IP+EA bar-H onto the GPU where the STEOM operator/G is built when
    /// the cluster share-barH path device-balances (avoids OOM on device 0).
    /// No-op if src_dev==dst_dev or on CPU. Implemented in steom_barh_cache.cu.
    void migrate_to_device(int src_dev, int dst_dev);

    // --- per-group current device (for the EA-solve-on-device-0 choreography) ---
    // The 8 IP buffers and the 3 EA buffers can live on different GPUs: we move
    // IP's off device 0 before EA solves (so EA's Davidson has room there), then
    // move EA's onto the IP device before STEOM. These track where each group is.
    int ip_dev = 0;   ///< GPU holding Loo/Lvv/Fov/Woooo/Wooov/Wovov/Wovvo/Wovoo
    int ea_dev = 0;   ///< GPU holding Wvovv/Wvvvv/Wvvvo

    /// Migrate the 8 IP-side buffers from ip_dev → dst (peer copy) and set ip_dev.
    void migrate_ip_to(int dst);
    /// Migrate the 3 EA-side buffers from ea_dev → dst (peer copy) and set ea_dev.
    void migrate_ea_to(int dst);
};

}  // namespace gansu
