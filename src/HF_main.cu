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



/**
 * @file HF_main.cu
 * @brief Main function for the Hartree-Fock method.
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "hf.hpp"
#include "parameter_manager.hpp"
#include "builder.hpp"


using namespace gansu;

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Arguments
 * @return 0 if the program ends successfully
 * @details This function reads the command line arguments and calls the RHF or UHF class.
 */
int main(int argc, char* argv[]){
  

    ParameterManager parameters;
    parameters.parse_command_line_args(argc, argv); // Parse the command line arguments

    std::unique_ptr<HF> hf = HFBuilder::buildHF(parameters);

    hf->solve(); // Solve the HF equation (SCF procedure)
    hf->report(); // Print the HF results
    
    // Export the SAD density matrix to a file
    if (parameters.contains("export_sad_cache")) {
        std::cout << "Exporting SAD cache to 'temp_sad_cache.dat'..." << std::endl;
        hf->generate_sad_cache("temp_sad_cache.dat");
        std::cout << "SAD cache exported successfully." << std::endl;
    }


    return 0;

}