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
 * @file parameter_manager.cpp 
 */


#include <fstream>

#include <algorithm>
#include <cctype>

#include "parameter_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu {

// Function to convert a string to lowercase
std::string toLowerCase(const std::string& input) {
    std::string result = input; // Copy the input string
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}


ParameterManager::ParameterManager(bool set_default_values) {
    // Default parameters
    default_parameters_ = {
        {"parameter_file", ""},                     // string
        {"xyzfilename", ""},                        // string
        {"gbsfilename", ""},                        // string
        {"verbose", "0"},                           // int(bool)
        {"method", "rhf"},                          // string
        {"charge", "0"},                            // int
        {"beta_to_alpha", "0"},                     // int
        {"eri_method", "stored"},                   // string
        {"post_hf_method", "none"},                  // string
        {"auxiliary_gbsfilename", ""},              // string
        {"maxiter", "100"},                         // int
        {"convergence_energy_threshold", "1e-6"},   // real_t
        {"schwarz_screening_threshold", "1e-12"},   // real_t 
        {"initial_guess", "core"},                  // string
        {"convergence_method", "diis"},             // string
        {"damping_factor", "0.9"},                  // real_t (0<=damping_factor<=1)
        {"diis_size", "8"},                         // int
        {"diis_include_transform", "0"},            // int(bool)
        {"rohf_parameter_name", "roothaan"},        // string
        {"geometry_optimization", "0"},             // int (bool)
        {"geometry_optimization_method", "powell"}, // string
        {"export_sad_cache", "0"},                   // int (bool)
        {"mulliken", "0"},                          // int (bool)
        {"mayer", "0"},                             // int (bool)
        {"wiberg", "0"},                            // int (bool)
        {"export_molden", "0"},                     // int (bool)
    };


    // Default short-to-full option mappings
    short_to_full_ = {
        {"-m", "method"},
        {"-v", "verbose"},
        {"-p", "parameter_file"},
        {"-x", "xyzfilename"},
        {"-g", "gbsfilename"},
        {"-ag", "auxiliary_gbsfilename"},
        {"-c", "charge"},
    };

    if(set_default_values){
        set_default_values_to_unspecfied_parameters();
    }
}





bool ParameterManager::contains(const std::string& key) const {
    std::string key_lower = toLowerCase(key);
    return parameters_.find(key_lower) != parameters_.end();
}

bool ParameterManager::is_valid_key(const std::string& key) const {
    std::string key_lower = toLowerCase(key);
    return default_parameters_.find(key_lower) != parameters_.end();
}

void ParameterManager::set_default_values_to_unspecfied_parameters() {
    for (const auto& pair : default_parameters_) {
        if (!contains(pair.first)) {
            parameters_[pair.first] = pair.second;
        }
    }
}



std::vector<std::string> ParameterManager::keys() const {
    std::vector<std::string> key_list;
    for (const auto& pair : parameters_) {
        key_list.push_back(pair.first);
    }
    return key_list;
}


void ParameterManager::parse_command_line_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key;
        std::string value;
        if (arg.rfind("--", 0) == 0) { // Long option, starts with "--"
            key = arg.substr(2);
        } else if (arg.rfind("-", 0) == 0) { // Short option, starts with "-"
            auto it = short_to_full_.find(arg);
            if (it != short_to_full_.end()) {
                key = it->second; // Convert short option to full option
            } else {
                THROW_EXCEPTION("Unknown short option: " + arg);
            }
        } else {
            THROW_EXCEPTION("Invalid option format: " + arg);
        }
        // Check for a value
        if(key == "charge"){ // ``charge'' is a special case to accept "-1" as a value
            if (i + 1 < argc) {
                value = argv[++i]; // Consume the next argument as the value
            }else{
                THROW_EXCEPTION("No value provided for parameter: " + key);
            }
        }else if (i + 1 < argc && argv[i + 1][0] != '-') {
            value = argv[++i]; // Consume the next argument as the value
        } else {
            value = "1"; // Default to "true" for flags without values, e.g. "--verbose". "1" is used to treat it as a non-zero integer.
        }


        // convert boolean values to "1" or "0"
        if(value == "true" || value == "True" || value == "TRUE"){
            value = "1";
        }else if(value == "false" || value == "False" || value == "FALSE"){
            value = "0";
        }

        if(is_valid_key(key)){
            parameters_[key] = value;
        }else{
            THROW_EXCEPTION("Unknown parameter: " + key);
        }
    }
}



void ParameterManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        THROW_EXCEPTION("Failed to open parameter file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Remove carriage return (in case of \r\n line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip empty or whitespace-only lines
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }

        // Parse "key = value" format
        auto delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            THROW_EXCEPTION("Invalid line format: " + line);
        }

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // convert boolean values to "1" or "0"
        if(value == "true" || value == "True" || value == "TRUE"){
            value = "1";
        }else if(value == "false" || value == "False" || value == "FALSE"){
            value = "0";
        }

        std::string key_lower = toLowerCase(key);

        // add the key-value pair to the parameters map
        if(is_valid_key(key_lower)){
            if(key_lower == "xyzfilename" || key_lower == "gbsfilename" || key_lower == "parameter_file"){
                parameters_[key_lower] = value;
            }else{
                std::string value_lower = toLowerCase(value);
                parameters_[key_lower] = value_lower;
            }
        }else{
            THROW_EXCEPTION("Unknown parameter: " + key);
        }
    }
}

} // namespace gansu