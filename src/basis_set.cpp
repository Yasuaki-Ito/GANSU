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

    // Read lines until the first character of the line is an alphabet.
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


BasisSet BasisSet::generate_auxiliary_basis(const BasisSet& primary_basis_set, int max_auxiliary_shell_type) {
    BasisSet aux_basis_set;

    for (const auto& [element_name, element_basis] : primary_basis_set.element_basis_sets) {
        ElementBasisSet aux_element;
        aux_element.set_element_name(element_name);

        // Collect all primitive exponents and determine max angular momentum from primary basis
        std::vector<double> exponents;
        int max_primary_L = 0;
        for (size_t i = 0; i < element_basis.get_num_contracted_gausses(); i++) {
            const auto& cg = element_basis.get_contracted_gauss(i);
            int L = shell_name_to_shell_type(cg.get_type());
            if (L > max_primary_L) max_primary_L = L;
            for (size_t j = 0; j < cg.get_num_primitives(); j++) {
                exponents.push_back(cg.get_primitive_gauss(j).exponent);
            }
        }

        // Auto-determine max auxiliary angular momentum: 2 * L_max_primary
        // (products of basis functions with l1, l2 produce angular momenta up to l1+l2)
        int effective_max_L = std::min(2 * max_primary_L, max_auxiliary_shell_type);

        // Generate pairwise sums of exponents (product basis approach)
        std::vector<double> aux_exponents;
        for (size_t i = 0; i < exponents.size(); i++) {
            for (size_t j = i; j < exponents.size(); j++) {
                aux_exponents.push_back(exponents[i] + exponents[j]);
            }
        }

        // Sort and aggressively thin out to avoid ill-conditioning
        // Keep exponents only if they differ by a factor of >= 2.0 from the previous kept one
        std::sort(aux_exponents.begin(), aux_exponents.end());
        std::vector<double> unique_exponents;
        for (const auto& exp : aux_exponents) {
            if (unique_exponents.empty() ||
                exp / unique_exponents.back() >= 2.0) {
                unique_exponents.push_back(exp);
            }
        }

        // Create uncontracted functions for each angular momentum
        for (int L = 0; L <= effective_max_L; L++) {
            std::string shell_name = SHELL_TYPE_TO_SHELL_NAME[L];
            // Capitalize the shell name (s -> S, p -> P, etc.)
            shell_name[0] = std::toupper(shell_name[0]);

            for (const auto& exp : unique_exponents) {
                ContractedGauss cg(shell_name);
                cg.add_primitive_gauss(exp, 1.0); // uncontracted: single primitive with coefficient 1.0
                aux_element.add_contracted_gauss(cg);
            }
        }

        aux_basis_set.add_element_basis_set(aux_element);
    }

    return aux_basis_set;
}

}