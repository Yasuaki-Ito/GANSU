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
  * @file optimizer.hpp
  * @brief Header file for the Optimizer class.
  */

#pragma once

#include <vector>


class Optimizer {
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    // Optimeze the structure of the molecule
    virtual void optimize(const std::vector<Atom>& initial_atoms) = 0;
    
};
