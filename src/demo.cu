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



#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "rhf.hpp"
#include "uhf.hpp"
#include "basis_set.hpp"

using namespace gansu;


std::string round_to_1_decimal(double value) {
    double rounded_value = std::round(value * 10.0) / 10.0;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << rounded_value;
    return oss.str();
}

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Arguments
 * @return 0 if the program ends successfully
 */
int main(int argc, char* argv[]){

    std::string gbsfilename = "../basis/sto-3g.gbs";

    // H2 molecule
    std::vector<Atom> atoms = {
        {1, {0.0, 0.0, 0.0}},
        {1, {0.0, 0.0, 0.0}}
    };
     
    
    ParameterManager parameters(true); // use default values
    parameters["gbsfilename"] = gbsfilename;
    parameters["convergence_energy_threshold"] = 1e-6;
    parameters["initial_guess_method"] = "sad";
    parameters["convergence_method"] = "DIIS";
    parameters["diis_size"] = 8;


    for(double r=0.1; r<5.0; r+=0.1){
        atoms[1].coordinate.z = r;
        Molecular mol = Molecular(atoms, gbsfilename);

        double rhf_energy;
        long long rhf_time;
        {
            RHF rhf = RHF(mol, parameters);
            rhf.solve(); 

            rhf_energy =  rhf.get_total_energy();
            rhf_time = rhf.get_solve_time_in_milliseconds();
            if(rhf.is_export_molden()){
                rhf.export_molden_file("rhf_" + round_to_1_decimal(r) + ".molden");
            }
        }

        double uhf_energy;
        long long uhf_time;

        {
            UHF uhf = UHF(mol, parameters);
            uhf.solve(); 

            uhf_energy =  uhf.get_total_energy();
            uhf_time = uhf.get_solve_time_in_milliseconds();
            if(uhf.is_export_molden()){
                uhf.export_molden_file("uhf_" + round_to_1_decimal(r) + ".molden");
            }
        }

        std::cout << "DEMO," << r << "," << rhf_energy << "," << uhf_energy << "," << rhf_time << "," << uhf_time << std::endl;
    }

    return 0;

}