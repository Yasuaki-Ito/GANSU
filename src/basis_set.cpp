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


#include <fstream>
#include <sstream> // std::istringstream
#include <cctype> // std::isalpha
#include <algorithm> // std::replace
#include <utility> // std::pair

#include "basis_set.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{

/**
 * @brief Construct of a basis set from gbs file
 * @param filename Basis set file name (Gaussian basis set file)
 * @return Basis set
 */
BasisSet BasisSet::construct_from_gbs(const std::string& filename){
    std::ifstream ifs(filename);
    if(!ifs){
        THROW_EXCEPTION("Cannot open basis set file: " + filename);
    }

    BasisSet basis_set;
    std::string line;

    ElementBasisSet current_element_basis_set;

    // Read lines until the first charactor of the line is an alphabet.
    while(std::getline(ifs, line)){
        if(std::isalpha(line[0])){
            // unread the line
            ifs.seekg(ifs.tellg() - static_cast<std::streamoff>(line.size() + 1));
            break;
        }
    }

    while(!ifs.eof()){
        if(!current_element_basis_set.get_element_name().empty()){
            basis_set.add_element_basis_set(current_element_basis_set);
            current_element_basis_set = ElementBasisSet();
        }

        
        { // Read a line for Element name
            std::getline(ifs, line);
            std::istringstream iss(line);
            // Get element name (H, He, Li, ...)
            std::string element_name;
            iss >> element_name;
            current_element_basis_set.set_element_name(element_name);
        }

        // Read lines for basis functions
        while(std::getline(ifs, line)){
            // If the line is "****", the end of the basis functions
            if(line == "****"){
                break;
            }

            std::istringstream iss(line);

            // Get the type of the basis functions and the number of primitive Gaussians
            std::string type;
            size_t num_primitives;
            iss >> type >> num_primitives;


            if(type.length() == 1){ // S, P, D, F, ...
                ContractedGauss contracted_gauss(type);
                for(size_t i = 0; i < num_primitives; i++){
                    std::getline(ifs, line);
                    // Replace all "D"s to "E"s for the exponential notation
                    std::replace(line.begin(), line.end(), 'D', 'E');

                    std::istringstream iss(line);
                    double exponent, coefficient;
                    iss >> exponent >> coefficient;
                    contracted_gauss.add_primitive_gauss(exponent, coefficient);
                }
                current_element_basis_set.add_contracted_gauss(contracted_gauss);
            }else if(type.length() == 2){ // SP, ??, ...
                ContractedGauss contracted_gauss0(std::string(1,type[0]));
                ContractedGauss contracted_gauss1(std::string(1,type[1]));
                for(size_t i = 0; i < num_primitives; i++){
                    std::getline(ifs, line);
                    // Replace all "D"s to "E"s for the exponential notation
                    std::replace(line.begin(), line.end(), 'D', 'E');

                    std::istringstream iss(line);
                    double exponent, coefficient0, coefficient1;
                    iss >> exponent >> coefficient0 >> coefficient1;
                    contracted_gauss0.add_primitive_gauss(exponent, coefficient0);
                    contracted_gauss1.add_primitive_gauss(exponent, coefficient1);
                }
                current_element_basis_set.add_contracted_gauss(contracted_gauss0);
                current_element_basis_set.add_contracted_gauss(contracted_gauss1);
            }else{ // could not find, or three or more characters
                THROW_EXCEPTION("Invalid basis function name: " + type);
            }
        }
    }

    // The last element basis set is added
    if(!current_element_basis_set.get_element_name().empty()){
        basis_set.add_element_basis_set(current_element_basis_set);
    }

    return basis_set;
}


}