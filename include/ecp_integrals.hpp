/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ecp_integrals.hpp
 * @brief ECP integral evaluation — declarations only (impl in src/ecp_integrals.cpp)
 */

#pragma once

#include "ecp.hpp"
#include "types.hpp"
#include <unordered_map>
#include <string>

namespace gansu {
namespace ecp_integral {

/// Primitive Gaussian normalization
double primitive_norm(double alpha, int l, int m, int n);

/// Compute the full ECP matrix (Type 1 local + Type 2 semi-local)
void compute_ecp_matrix(
    const PrimitiveShell* shells, int num_primitives,
    const double* cgto_norms,
    int num_basis,
    const Atom* atoms, int num_atoms,
    const std::unordered_map<std::string, ElementECP>& ecp_data,
    double* V_ecp,
    int n_radial = 75, int angular_order = 26);

} // namespace ecp_integral
} // namespace gansu
