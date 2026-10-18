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
 * @file launch_audit.hpp
 * @brief Optional audit of very large kernel launches (GANSU_LAUNCH_AUDIT).
 */

#ifndef LAUNCH_AUDIT_HPP
#define LAUNCH_AUDIT_HPP

#include <cstddef>
#include "types.hpp"

namespace gansu::gpu {

/// Report (to stderr, under GANSU_LAUNCH_AUDIT) any kernel launch whose thread
/// count approaches 2^32, where a 32-bit `blockIdx.x * blockDim.x` would wrap
/// and silently recompute work.
void auditLaunchSize(const char* what, size_t num_threads);

/// Normalise a launch extent so the audit can be called on either form a launch
/// configuration takes: a scalar count or a dim3.
inline size_t launchExtent(unsigned long long n) { return (size_t)n; }
inline size_t launchExtent(const dim3& d) {
    return (size_t)d.x * d.y * d.z;
}

} // namespace gansu::gpu

#endif // LAUNCH_AUDIT_HPP
