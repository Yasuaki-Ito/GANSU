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

#pragma once


/* Uncomment below if you would like to handle inputs including d-type basis functions */
//#define COMPUTE_D_BASIS

/* Uncomment below if you would like to handle inputs including g-type auxiliary basis functions */
//#define COMPUTE_G_AUX

/*
#if defined(COMPUTE_D_BASIS) && defined(COMPUTE_G_AUX)
#pragma message("D-type auxiliary basis functions and G-type auxiliary basis functions for RI approximations (density fitting) are supported.")
#pragma message("Build time will take one day or more. If you don't need them, please disable it in CMakeLists.txt.")
#elif defined(COMPUTE_D_BASIS)
#pragma message("D-type auxiliary basis functions for RI approximations (density fitting) are supported.")
#pragma message("Build time will take about one day. If you don't need them, please disable it in CMakeLists.txt.")
#elif defined(COMPUTE_G_AUX)
#pragma message("G-type auxiliary basis functions for RI approximations (density fitting) are supported.")  
#pragma message("Build time will take several hours (5 or 6 hours). If you don't need them, please disable it in CMakeLists.txt.")
#else
#pragma message("D-type auxiliary basis functions and G-type auxiliary basis functions for RI approximations (density fitting) are not supported.")
#pragma message("Build time will be short. If you need them, please enable it in CMakeLists.txt.")
#endif
*/
