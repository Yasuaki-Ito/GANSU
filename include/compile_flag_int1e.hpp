/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once



/* ============================================================
 * One-electron integral maximum angular momentum
 * ============================================================ */

/*
 * Priority:
 *   ENABLE_G_INT1E > ENABLE_F_INT1E > default (d)
 */

#if defined(ENABLE_G_INT1E)
  #define INT1E_MAX_L 4   // g
#elif defined(ENABLE_F_INT1E)
  #define INT1E_MAX_L 3   // f
#else
  #define INT1E_MAX_L 2   // d
#endif

// #if INT1E_MAX_L == 2
//   #pragma message("One-electron integrals: up to d orbitals")
// #elif INT1E_MAX_L == 3
//   #pragma message("One-electron integrals: up to f orbitals")
// #elif INT1E_MAX_L == 4
//   #pragma message("One-electron integrals: up to g orbitals")
// #endif

// #if defined(ENABLE_G_INT1E)
//     #pragma message("G-orbital one-electron integrals are enabled.")
// #elif defined(ENABLE_F_INT1E)
//     #pragma message("F-orbital one-electron integrals are enabled.")
// #else
//     #pragma message("One-electron integrals are limited to d orbitals.")
// #endif
