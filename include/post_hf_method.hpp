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
    ADC2X,
    EOM_MP2,
    EOM_CC2,
    EOM_CCSD
};

} // namespace gansu
