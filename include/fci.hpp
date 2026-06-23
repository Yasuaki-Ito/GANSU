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


double fci(double* h_Gmo1e, double* h_Gmo, int nao, int nelec, int na, long long np, double E_hf);

#ifdef GANSU_MPI
// Multi-node / multi-GPU FCI (MPI + NCCL hybrid). Same contract as fci() — all
// pointers are device pointers and the GPU is pinned per rank before the call.
// Defined in src/fci_mpi.cu, dispatched from eri_stored_fci.cu on >1 rank.
double fci_mpi(double* d_Gmo1e, double* d_Gmo, int norb, int nelec, int na, long long np, double E_rhf);
#endif
