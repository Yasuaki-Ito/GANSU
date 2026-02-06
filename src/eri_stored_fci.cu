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

 
#include <iomanip>
#include <iostream>
#include <assert.h>


#include "rhf.hpp"
#include "eri_stored.hpp"
#include "device_host_memory.hpp"

namespace gansu {



real_t ERI_Stored_RHF::compute_fci_energy() {
    PROFILE_FUNCTION();

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();


    // FCI implementation goes here.
    // 1. Convert AO integrals to MO integrals
    // 2. Compute FCI energy using the MO integrals

    return 1.0; // dummy return
}




} // namespace gansu