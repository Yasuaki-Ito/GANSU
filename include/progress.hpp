/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file progress.hpp
 * @brief Thread-local progress reporting for iterative procedures.
 *
 * Any iterative loop (SCF, CCSD, Davidson, optimizer, ...) can report
 * progress by calling gansu::report_progress(). If a callback has been
 * registered (via the C API gansu_set_progress_callback), it is invoked
 * synchronously. Otherwise the call is a no-op.
 *
 * Usage inside a loop:
 *   double vals[] = {energy, delta_e};
 *   gansu::report_progress("scf", iter, 2, vals);
 */

#pragma once

namespace gansu {

/// Set the thread-local progress callback (called by C API layer).
void set_progress_callback(void (*fn)(const char*, int, int, const double*, void*), void* user_data);

/// Clear the thread-local progress callback.
void clear_progress_callback();

/// Report progress. No-op if no callback is set.
void report_progress(const char* stage, int iter, int n_values, const double* values);

} // namespace gansu
