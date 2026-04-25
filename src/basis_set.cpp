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

    bool found_ecp = false;
    while(!ifs.eof()){
        if(!current_element_basis_set.get_element_name().empty() &&
           current_element_basis_set.get_num_contracted_gausses() > 0){
            basis_set.add_element_basis_set(current_element_basis_set);
        }
        current_element_basis_set = ElementBasisSet();


        { // Read a line for Element name
            std::getline(ifs, line);
            if(ifs.eof()) break;
            // Check if this line is an ECP header (contains "-ECP")
            if(line.find("-ECP") != std::string::npos || line.find("-ecp") != std::string::npos) {
                // Rewind so the ECP parser can read this line
                ifs.seekg(ifs.tellg() - static_cast<std::streamoff>(line.size() + 1));
                found_ecp = true;
                break;
            }
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

            // Check if this line is an ECP header
            if(line.find("-ECP") != std::string::npos || line.find("-ecp") != std::string::npos) {
                ifs.seekg(ifs.tellg() - static_cast<std::streamoff>(line.size() + 1));
                found_ecp = true;
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
        if(found_ecp) break;
    }

    // The last element basis set is added (skip if empty, e.g., ECP-only element header)
    if(!current_element_basis_set.get_element_name().empty() &&
       current_element_basis_set.get_num_contracted_gausses() > 0){
        basis_set.add_element_basis_set(current_element_basis_set);
    }

    // Continue reading: look for ECP sections in the same file
    // ECP lines start with "ELEMENT-ECP" or "ELEMENT     0" followed by "ELEMENT-ECP"
    while(std::getline(ifs, line)){
        // Skip blank/comment lines
        if(line.empty() || line[0] == '!' || line[0] == '#') continue;

        // Check for ECP header: "XX-ECP  l_max  n_core"
        if(line.find("-ECP") != std::string::npos || line.find("-ecp") != std::string::npos){
            std::istringstream iss(line);
            std::string ecp_label;
            int l_max, n_core;
            iss >> ecp_label >> l_max >> n_core;

            // Extract element name: "BR-ECP" → "Br"
            std::string element_name = ecp_label.substr(0, ecp_label.find('-'));
            if(!element_name.empty()){
                element_name[0] = std::toupper(element_name[0]);
                for(size_t i = 1; i < element_name.size(); i++)
                    element_name[i] = std::tolower(element_name[i]);
            }

            ElementECP ecp;
            ecp.set_element_name(element_name);
            ecp.set_l_max(l_max);
            ecp.set_n_core_electrons(n_core);

            // Read l_max+1 components: first = local, rest = semi-local
            for(int comp_idx = 0; comp_idx <= l_max; comp_idx++){
                // Read label line (e.g., "f potential" or "s-f potential")
                do {
                    if(!std::getline(ifs, line)) goto ecp_done;
                } while(line.empty() || line[0] == '!' || line[0] == '#');

                // Read number of primitives
                if(!std::getline(ifs, line)) goto ecp_done;
                std::replace(line.begin(), line.end(), 'D', 'E');
                int n_prim = 0;
                { std::istringstream iss2(line); iss2 >> n_prim; }

                ECPComponent comp;
                comp.angular_momentum = (comp_idx == 0) ? -1 : (comp_idx - 1);

                for(int k = 0; k < n_prim; k++){
                    if(!std::getline(ifs, line)) goto ecp_done;
                    std::replace(line.begin(), line.end(), 'D', 'E');
                    std::istringstream iss3(line);
                    int power;
                    double exponent, coefficient;
                    iss3 >> power >> exponent >> coefficient;
                    comp.primitives.push_back({power, exponent, coefficient});
                }

                if(comp_idx == 0)
                    ecp.set_local_component(comp);
                else
                    ecp.add_semilocal_component(comp);
            }

            basis_set.add_element_ecp(ecp);
        }
    }
    ecp_done:

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


// ============================================================
//  ECP file parser (Gaussian94 format)
//
//  Format:
//    ELEMENT-ECP  l_max  n_core_electrons
//    local_label (e.g., "f potential" or "g potential")
//      N
//      n1  zeta1  d1
//      ...
//    semilocal_label (e.g., "s-f potential")
//      N
//      n1  zeta1  d1
//      ...
//    ****  (or next element, or EOF)
// ============================================================

static std::string shell_label_for_l(int l) {
    const char* labels[] = {"s", "p", "d", "f", "g", "h", "i"};
    return (l >= 0 && l <= 6) ? labels[l] : "?";
}

std::unordered_map<std::string, ElementECP> BasisSet::parse_ecp_file(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs) {
        THROW_EXCEPTION("Cannot open ECP file: " + filename);
    }

    std::unordered_map<std::string, ElementECP> ecps;
    std::string line;

    while (std::getline(ifs, line)) {
        // Skip comments and blank lines
        if (line.empty() || line[0] == '!' || line[0] == '#') continue;
        if (line.find("****") != std::string::npos) continue;

        // Look for ECP header: "ELEMENT-ECP  l_max  n_core"
        if (line.find("-ECP") != std::string::npos || line.find("-ecp") != std::string::npos) {
            std::istringstream iss(line);
            std::string ecp_label;
            int l_max, n_core;
            iss >> ecp_label >> l_max >> n_core;

            // Extract element name from "BR-ECP" → "Br"
            std::string element_name = ecp_label.substr(0, ecp_label.find('-'));
            // Capitalize first letter, lowercase rest
            if (!element_name.empty()) {
                element_name[0] = std::toupper(element_name[0]);
                for (size_t i = 1; i < element_name.size(); i++)
                    element_name[i] = std::tolower(element_name[i]);
            }

            ElementECP ecp;
            ecp.set_element_name(element_name);
            ecp.set_l_max(l_max);
            ecp.set_n_core_electrons(n_core);

            // Read l_max+1 components: first is local, rest are semi-local
            // Local is labeled e.g., "f potential" (for l_max=3) or "g potential" (for l_max=4)
            // Semi-local labeled "s-f potential", "p-f potential", etc.
            for (int comp_idx = 0; comp_idx <= l_max; comp_idx++) {
                // Read label line (e.g., "f potential" or "s-f potential")
                if (!std::getline(ifs, line)) break;
                // Skip blank/comment lines
                while (line.empty() || line[0] == '!' || line[0] == '#') {
                    if (!std::getline(ifs, line)) break;
                }

                // Read number of primitives
                if (!std::getline(ifs, line)) break;
                std::replace(line.begin(), line.end(), 'D', 'E');
                int n_prim = 0;
                {
                    std::istringstream iss2(line);
                    iss2 >> n_prim;
                }

                ECPComponent comp;
                if (comp_idx == 0) {
                    comp.angular_momentum = -1;  // local component
                } else {
                    comp.angular_momentum = comp_idx - 1;  // s=0, p=1, d=2, ...
                }

                for (int k = 0; k < n_prim; k++) {
                    if (!std::getline(ifs, line)) break;
                    std::replace(line.begin(), line.end(), 'D', 'E');
                    std::istringstream iss3(line);
                    int power;
                    double exponent, coefficient;
                    iss3 >> power >> exponent >> coefficient;
                    comp.primitives.push_back({power, exponent, coefficient});
                }

                if (comp_idx == 0) {
                    ecp.set_local_component(comp);
                } else {
                    ecp.add_semilocal_component(comp);
                }
            }

            ecps[element_name] = ecp;
        }
    }

    return ecps;
}

}