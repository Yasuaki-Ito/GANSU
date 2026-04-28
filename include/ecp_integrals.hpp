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

/// Compute ECP gradient contributions via finite difference
/// grad_ecp[3*iatom + xyz] += Tr(D * dV_ECP/dR)
void compute_ecp_gradient(
    const PrimitiveShell* shells, int num_primitives,
    const double* cgto_norms,
    int num_basis,
    const Atom* atoms, int num_atoms,
    const std::unordered_map<std::string, ElementECP>& ecp_data,
    const double* density_matrix,
    double* grad_ecp);

/// GPU kernel launch for off-center Type 2 (defined in ecp_integrals_gpu.cu)
struct AngPtGPU { double x, y, z, w; };

#ifndef GANSU_CPU_ONLY
void launch_ecp_type2_offcenter_gpu(
    const AngPtGPU* d_ang_grid, int n_ang,
    const double* d_prim_exp_a, const double* d_prim_coef_a, int n_prim_a,
    int la, int ma_c, int na_c, double dAx, double dAy, double dAz,
    const double* d_prim_exp_b, const double* d_prim_coef_b, int n_prim_b,
    int lb, int mb_c, int nb_c, double dBx, double dBy, double dBz,
    const double* d_ecp_exp, const double* d_ecp_coef, const int* d_ecp_pow, int n_ecp_prim,
    const double* d_rad_nodes, const double* d_rad_weights, int n_rad,
    double R_max, int l_proj,
    double* d_output_prim_val);
#endif

} // namespace ecp_integral
} // namespace gansu
