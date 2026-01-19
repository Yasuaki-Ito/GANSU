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

#include <string>

namespace gansu{

#ifdef CONSOLE_COLOR_ENABLED

namespace ConsoleColor {
    const std::string RED     = "\033[31m";
    const std::string GREEN   = "\033[32m";
    const std::string YELLOW  = "\033[33m";
    const std::string BLUE    = "\033[34m";
    const std::string MAGENTA = "\033[35m";
    const std::string CYAN    = "\033[36m";
    const std::string WHITE   = "\033[37m";
    
    const std::string RESET   = "\033[0m";
}

#else

namespace ConsoleColor {
    const std::string RED     = "";
    const std::string GREEN   = "";
    const std::string YELLOW  = "";
    const std::string BLUE    = "";
    const std::string MAGENTA = "";
    const std::string CYAN    = "";
    const std::string WHITE   = "";
    
    const std::string RESET   = "";
}


#endif

}
