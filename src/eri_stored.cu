/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */


 #include "rhf.hpp"


 namespace gansu {



real_t ERI_Stored_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    // implemented here
    THROW_EXCEPTION("MP2 energy calculation is not supported with the stored ERI method.");


    return 0.0; // return mp2 energy exluding rhf energy;
}

real_t ERI_Stored_RHF::compute_mp3_energy() {
    PROFILE_FUNCTION();

    // implemented here
    // Not only MP3 but also MP2 energy calculation needs to be implemented here

    THROW_EXCEPTION("MP3 energy calculation is not supported with the stored ERI method.");

    return 0.0; // return mp2+mp3 energy exluding rhf energy;
}

















 } // namespace gansu