/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_mp2.hpp
 * @brief Closed-shell MP2 energy from a THC factorisation (CPU reference).
 *
 *   E_MP2 = -sum_{ij in occ, ab in vir}
 *               (ia|jb) [2 (ia|jb) - (ib|ja)] / (eps_a + eps_b - eps_i - eps_j)
 *
 * THC representation:
 *   X^P_p = sum_mu C_{mu,p} X^P_mu
 *   (pq|rs) ~= sum_{PQ} X^P_p X^P_q Z_{PQ} X^Q_r X^Q_s
 * Same Z works in any orbital basis as long as the same MO transform is
 * applied to both bra and ket pairs.
 *
 * Phase 2.0a path: form the MO ERI tensor explicitly (N_orb^4 entries), then
 * apply the canonical MP2 contraction.  GPU port can later use a
 * factorisation-aware MP2 (no full N^4 tensor) using SOS+Laplace.
 */

#pragma once

#include <vector>
#include <memory>
#include "types.hpp"
#ifndef GANSU_CPU_ONLY
#include "device_host_memory.hpp"
#endif

namespace gansu {

/**
 * @brief Transform the AO collocation matrix to an MO/orbital basis.
 *
 *   X_mo[p, P] = sum_mu C[mu, p] * X_ao[mu, P]
 *
 * @param X_ao Collocation in AO basis [N_bas x N_g] column-major.
 * @param C    MO coefficient matrix  [N_bas x N_orb] column-major
 *             (column p is orbital p).
 * @param N_bas, N_orb, N_g
 * @return     X_mo [N_orb x N_g] column-major.
 */
std::vector<real_t> transform_X_to_mo_cpu(const std::vector<real_t>& X_ao,
                                          const std::vector<real_t>& C,
                                          int N_bas, int N_orb, int N_g);

/**
 * @brief Compute closed-shell MP2 energy from a 4-index MO ERI tensor.
 *
 *   E_MP2 = -sum_{ij in [0,n_occ), ab in [n_occ, n_orb)}
 *               (ia|jb) [2 (ia|jb) - (ib|ja)] / (eps_a + eps_b - eps_i - eps_j)
 *
 * @param eri_mo_4d Length N_orb^4, indexed (chemist notation)
 *                  eri[p + N_orb*(q + N_orb*(r + N_orb*s))] = (pq|rs).
 * @param eps       Orbital energies, length N_orb.
 * @param n_occ     Number of (doubly) occupied orbitals.
 * @param N_orb     Total number of orbitals (occ + vir).
 * @return          E_MP2 (Hartree).  Negative for closed-shell stable cases.
 */
real_t compute_mp2_energy_from_mo_eri_cpu(const std::vector<real_t>& eri_mo_4d,
                                          const std::vector<real_t>& eps,
                                          int n_occ, int N_orb);

#ifndef GANSU_CPU_ONLY

/**
 * @brief GPU MO-basis transform of X collocation: X_mo = C^T X_ao.
 *
 * @param d_X_ao   [N_bas x N_g] column-major on device.
 * @param d_C      [N_bas x N_orb] ROW-major (GANSU convention: C[mu*N_orb+p]).
 * @param N_bas, N_orb, N_g
 * @return DeviceHostMatrix [N_orb x N_g] column-major on device.
 */
std::unique_ptr<DeviceHostMatrix<real_t>>
transform_X_to_mo_gpu(const real_t* d_X_ao, const real_t* d_C,
                       int N_bas, int N_orb, int N_g);

/**
 * @brief GPU MP2 energy from MO ERI in column-major chemist notation.
 *
 *   eri[p + N*(q + N*(r + N*s))] = (pq|rs)
 *
 * @param d_eri_mo_4d  Length N_orb^4 on device.
 * @param d_eps        Orbital energies length N_orb on device.
 * @param n_occ, N_orb
 * @return E_MP2 (Hartree).
 */
real_t compute_mp2_energy_from_mo_eri_gpu(const real_t* d_eri_mo_4d,
                                           const real_t* d_eps,
                                           int n_occ, int N_orb);

/**
 * @brief THC-SOS-MP2 with Laplace decoupling (Phase 2.1).
 *
 * Computes the closed-shell opposite-spin MP2 energy, scaled by c_os, using
 *   E^OS = -c_os Σ_τ w_τ Σ_{PQRS} Z_{PQ} U_{PR}(τ) U_{QS}(τ) Z_{RS}
 *   U_{PR}(τ) = O_{PR}(τ) · V_{PR}(τ)               (Hadamard)
 *   O_{PR}(τ) = Σ_i  X^P_i  X^R_i  exp(+ε_i τ)
 *   V_{PR}(τ) = Σ_a  X^P_a  X^R_a  exp(-ε_a τ)
 *
 * Each τ step costs O(N_g^3) (two N_g×N_g DGEMMs) -- linear-scaling in N_g
 * but cubic in grid size.  For typical THC grids N_g ~ 5..10·N_bas, this is
 * O(N_bas^3) overall (no four-index ERI tensor).
 *
 * @param d_X_mo     [N_orb x N_g] column-major MO collocation on device.
 * @param d_Z        [N_g x N_g] column-major THC core on device.
 * @param d_eps      Orbital energies length N_orb on device.
 * @param n_occ, N_orb, N_g
 * @param n_laplace  Number of Laplace quadrature points (default 12).
 * @param c_os       Opposite-spin scaling factor (default 1.3, GANSU convention).
 * @return E^OS_SOS-MP2 (Hartree, negative for stable systems).
 */
real_t compute_thc_sos_mp2_energy_gpu(const real_t* d_X_mo,
                                       const real_t* d_Z,
                                       const real_t* d_eps,
                                       int n_occ, int N_orb, int N_g,
                                       int n_laplace = 12,
                                       double c_os = 1.3,
                                       int num_gpus = 1);

#endif // GANSU_CPU_ONLY

} // namespace gansu
