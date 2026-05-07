/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <string>
#include <stdexcept>
#include "types.hpp"

namespace gansu {

/**
 * @brief Resolved DLPNO threshold parameters for a single calculation.
 *
 * Built by combining a preset (loose / normal / tight / very_tight) with
 * any user overrides. Preset values follow ORCA's published DLPNO defaults
 * (Riplinger 2013, Liakos 2015, Saitow 2017).
 */
struct DLPNOParams {
    // Preset name actually used (after defaults applied)
    std::string preset = "normal";

    // Localizer choice ("pm", "boys", "ibo")
    std::string localizer = "pm";

    // Localizer convergence
    int    localizer_max_sweep = 200;
    real_t localizer_conv = 1e-10;

    // PNO occupation cutoff: drop natural orbitals with n < t_cut_pno
    real_t t_cut_pno = 3.33e-7;
    // PAO redundancy cutoff (overlap eigenvalue threshold for per-domain
    // Löwdin orthogonalisation)
    real_t t_cut_do = 1e-2;
    // Strong/weak pair MP2 estimate cutoff (Ha)
    real_t t_cut_pairs = 1e-4;
    // Boughton-Pulay Mulliken cumulative threshold (1 - completeness target)
    real_t t_cut_mkn = 1e-3;
    // (T): triple screening based on combined MP2 pair contribution
    real_t t_cut_triples = 1e-7;
    // (T): TNO occupation cutoff
    real_t t_cut_tno = 1e-9;

    // Distance-based pair pre-screening (Bohr); 0 disables
    real_t pair_distance_cutoff = 15.0;

    // CCSD residual solver
    int max_iter = 50;
    int diis_size = 6;

    // Iterative LMP2 amplitude solver (Jacobi)
    int lmp2_max_iter = 60;
    real_t lmp2_conv = 1e-8;
    // Number of self-consistent PNO refinement rounds (0 = single shot)
    int sc_pno_iter = 1;
    // PNO selection from opposite-spin amplitudes only.
    // Default false: full LMP2 density (Riplinger 2013) is best for full MP2.
    // Set true only for SOS-MP2-style energy evaluation.
    bool pno_os_only = false;

    // Logging level: 0=summary, 1=phase, 2=per-pair, 3=residual
    int verbose = 1;
};

/// Apply ORCA-compatible preset values, then override with any non-sentinel
/// user parameters. Sentinel for the cutoff-style doubles is < 0
/// (parameter_manager defaults are -1.0 for unset).
inline DLPNOParams resolve_dlpno_params(
    const std::string& preset,
    const std::string& localizer,
    real_t t_cut_pno,
    real_t t_cut_do,
    real_t t_cut_pairs,
    real_t t_cut_mkn,
    real_t t_cut_triples,
    real_t t_cut_tno,
    real_t pair_distance_cutoff,
    int    max_iter,
    int    diis_size,
    int    localizer_max_sweep,
    real_t localizer_conv,
    int    lmp2_max_iter,
    real_t lmp2_conv,
    int    sc_pno_iter,
    bool   pno_os_only,
    int    verbose)
{
    DLPNOParams p;
    p.preset = preset.empty() ? std::string("normal") : preset;

    if (p.preset == "loose") {
        p.t_cut_pno     = 1e-6;
        p.t_cut_do      = 2e-2;
        p.t_cut_pairs   = 1e-3;
        p.t_cut_mkn     = 1e-3;
        p.t_cut_triples = 1e-6;
        p.t_cut_tno     = 1e-9;
    } else if (p.preset == "normal") {
        p.t_cut_pno     = 3.33e-7;
        p.t_cut_do      = 1e-2;
        p.t_cut_pairs   = 1e-4;
        p.t_cut_mkn     = 1e-3;
        p.t_cut_triples = 1e-7;
        p.t_cut_tno     = 1e-9;
    } else if (p.preset == "tight") {
        p.t_cut_pno     = 1e-7;
        p.t_cut_do      = 5e-3;
        p.t_cut_pairs   = 1e-5;
        p.t_cut_mkn     = 1e-3;
        p.t_cut_triples = 1e-8;
        p.t_cut_tno     = 1e-10;
    } else if (p.preset == "very_tight" || p.preset == "verytight") {
        p.preset = "very_tight";
        p.t_cut_pno     = 1e-8;
        p.t_cut_do      = 2e-3;
        p.t_cut_pairs   = 1e-6;
        p.t_cut_mkn     = 1e-4;
        p.t_cut_triples = 1e-9;
        p.t_cut_tno     = 1e-10;
    } else {
        throw std::runtime_error(
            "Unknown dlpno_preset: '" + p.preset +
            "'. Valid: loose, normal, tight, very_tight.");
    }

    if (!localizer.empty()) {
        if (localizer != "pm" && localizer != "boys" && localizer != "ibo"
            && localizer != "none" && localizer != "canonical"
            && localizer != "identity") {
            throw std::runtime_error(
                "Unknown dlpno_localizer: '" + localizer +
                "'. Valid: pm, boys, ibo, none.");
        }
        p.localizer = localizer;
    }

    if (t_cut_pno     >= 0.0) p.t_cut_pno     = t_cut_pno;
    if (t_cut_do      >= 0.0) p.t_cut_do      = t_cut_do;
    if (t_cut_pairs   >= 0.0) p.t_cut_pairs   = t_cut_pairs;
    if (t_cut_mkn     >= 0.0) p.t_cut_mkn     = t_cut_mkn;
    if (t_cut_triples >= 0.0) p.t_cut_triples = t_cut_triples;
    if (t_cut_tno     >= 0.0) p.t_cut_tno     = t_cut_tno;

    p.pair_distance_cutoff = pair_distance_cutoff;
    p.max_iter             = max_iter;
    p.diis_size            = diis_size;
    p.localizer_max_sweep  = localizer_max_sweep;
    p.localizer_conv       = localizer_conv;
    p.lmp2_max_iter        = lmp2_max_iter;
    p.lmp2_conv            = lmp2_conv;
    p.sc_pno_iter          = sc_pno_iter;
    p.pno_os_only          = pno_os_only;
    p.verbose              = verbose;
    return p;
}

} // namespace gansu
