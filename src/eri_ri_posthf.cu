/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file eri_ri_posthf.cu
 * @brief Placeholder — all ERI_RI_RHF post-HF methods have been refactored
 *        into their respective eri_stored_*.cu files via _impl free functions.
 *
 * This file is kept for future methods that may need RI-specific logic.
 */

namespace gansu {
// All ERI_RI_RHF::compute_* methods are now defined in:
//   eri_stored.cu        (MP3, CCSD, CCSD(T))
//   eri_stored_cis.cu    (CIS)
//   eri_stored_adc2.cu   (ADC(2))
//   eri_stored_adc2x.cu  (ADC(2)-X)
//   eri_stored_cc2.cu    (CC2)
//   eri_stored_fci.cu    (FCI)
//   eri_stored_mp4.cu    (MP4)
//   eri_stored_eom_*.cu  (EOM-MP2, EOM-CC2, EOM-CCSD)
} // namespace gansu
