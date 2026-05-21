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
// post_hf_method.hpp
#pragma once

namespace gansu {

enum class PostHFMethod {
    None,
    FCI,
    MP2,
    SCS_MP2,
    SOS_MP2,
    LT_MP2,
    LT_SOS_MP2,
    MP3,
    MP4,
    CC2,
    CCSD,
    CCSD_T,
    CCSD_DENSITY,  // CCSD + Lambda + 1-RDM (DMET / property analysis)
    CIS,
    ADC2,
    SOS_ADC2,
    LT_SOS_ADC2,  // SOS-Laplace-ADC(2) — O(N⁴) with Laplace quadrature
    ADC2X,
    EOM_MP2,
    EOM_CC2,
    EOM_CCSD,
    DMET_CCSD,
    DMET_CCSD_T,
    THC_MP2,      // Tensor Hypercontraction MP2 (Phase 2.0a)
    THC_SOS_MP2,  // THC + Laplace SOS-MP2 (Phase 2.1, O(N^3) per Laplace pt)
    THC_SOS_ADC2, // THC + Laplace SOS-ADC(2) excited states (Phase 2.2a, MVP)
    DLPNO_MP2,    // Domain-based Local PNO MP2 (Phase 1)
    DLPNO_CCSD,   // Domain-based Local PNO CCSD (Phase 2)
    DLPNO_CCSD_T, // Domain-based Local PNO CCSD(T) (Phase 3)
    CIS_NTO,      // State-averaged CIS natural-transition-orbital active space (bt-PNO-STEOM Phase P0)
    IP_EOM_CCSD,  // Ionization-potential EOM-CCSD (bt-PNO-STEOM Phase P1, building block for ̂S^IP)
    EA_EOM_CCSD,  // Electron-affinity EOM-CCSD (bt-PNO-STEOM Phase P2, building block for ̂S^EA)
    STEOM_CCSD    // Similarity Transformed EOM-CCSD (bt-PNO-STEOM Phase P3; auto-runs CIS_NTO + IP-EOM + EA-EOM as prerequisites)
};

} // namespace gansu
