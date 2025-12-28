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
 * @file parameter_manager.hpp Header file for the ParameterManager class.
 */


#pragma once
#include <iostream>
#include <string>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "utils.hpp" // THROW_EXCEPTION

namespace gansu {


// prototype of function
/**
 * @brief Function to convert a string to lowercase
 * @param input Input string
 * @return Lowercase string
 */
std::string toLowerCase(const std::string& input);



/**
 * \brief A class to store key-value pairs where all values are internally stored as strings.
 *
 * This class provides methods to set and retrieve values with type conversion.
 */
class ParameterManager {
public:
    /**
     * \brief Proxy class for handling key-value assignment.
     *
     * This proxy allows values to be assigned to a key via operator=.
     */
    class ValueProxy {
    public:
        ValueProxy(std::unordered_map<std::string, std::string>& map, const std::string& key)
            : map_(map), key_(key) {}

        // Overloaded assignment operator for setting values.
        template <typename T>
        ValueProxy& operator=(const T& value) {
            key_ = toLowerCase(key_);

            std::ostringstream oss;
            oss << value;  // Convert the value to a string

            if(key_ == "xyzfilename" || key_ == "gbsfilename" || key_ == "auxiliary_gbsfilename" || key_ == "parameter_file"){ // Special case for file/path names
                map_[key_] = oss.str();
            }else{
                map_[key_] = toLowerCase(oss.str());
            }
            return *this;
        }

        // Implicit conversion to retrieve the stored value as a string.
        operator std::string() const {
            auto it = map_.find(key_);
            if (it == map_.end()) {
                THROW_EXCEPTION("Parameter not found: " + key_);
            }
            return it->second;
        }

    private:
        std::unordered_map<std::string, std::string>& map_;
        std::string key_;
    };

    /**
     * \brief Constructor to initialize default short-to-full option mappings.
     * \param set_default_values If true, set default values.
     */
    ParameterManager(bool set_default_values=false);

    /**
     * \brief Accessor for setting or getting values associated with a key.
     *
     * \param key The key to access.
     * \return A proxy object for assigning or retrieving values.
     */
    ValueProxy operator[](const std::string& key) {
        std::string key_lower = toLowerCase(key);
        return ValueProxy(parameters_, key_lower);
    }

    /**
     * @brief Checks if the given key exists in the map.
     *
     * @param key The key to check.
     * @return True if the key exists, false otherwise.
     */
    bool contains(const std::string& key) const;

    /**
     * @brief Checks if the given key is a valid key.
     * 
     * A key is considered valid if it is present in the default parameters map.
     */
    bool is_valid_key(const std::string& key) const ;

    /**
     * @brief Gets the value associated with the given key, converted to the specified type.
     *
     * @tparam T The desired type of the value.
     * @param key The key to look up.
     * @return The value associated with the key, converted to the specified type.
     * @throws std::out_of_range If the key is not found.
     * @throws std::invalid_argument If the value cannot be converted to the specified type.
     */
    template <typename T> 
    T get(const std::string& key) const {
        std::string key_lower = toLowerCase(key);
        auto it = parameters_.find(key);
        if (it == parameters_.end()) {
            THROW_EXCEPTION("Parameter not found: " + key);
        }

        T result;
        std::istringstream iss(it->second);
        if (!(iss >> result)) {
            THROW_EXCEPTION("Failed to convert value for key: " + key);
        }

        return result;
    }


    /**
     * @brief Returns a list of all keys stored in the parameter manager.
     *
     * @return A vector of strings containing all keys.
     */
    std::vector<std::string> keys() const;

    /**
     * @brief Parse command-line arguments and populate the parameters map.
     *
     * @param argc The number of command-line arguments.
     * @param argv The array of command-line arguments.
     */
    void parse_command_line_args(int argc, char* argv[]);


    /**
     * @brief Loads key-value pairs from a configuration file and registers them in the parameter map.
     * 
     * Lines in the file should follow the "key = value" format. Lines starting with `#` or parts of a line after `#` 
     * are treated as comments and ignored.
     * 
     * @param filename The path to the configuration file.
     * @throws std::ios_base::failure If the file cannot be opened.
     * @throws std::invalid_argument If a line does not follow the "key = value" format.
     */
    void load_from_file(const std::string& filename);

    /**
     * @brief Sets default values for unspecified parameters.
     * 
     * This method sets default values for any parameters that are not already present in the parameter map.
     */
    void set_default_values_to_unspecfied_parameters();

private:
    /**
     * @brief The underlying storage for key-value pairs.
     * All values are stored as strings.
     */
    std::unordered_map<std::string, std::string> parameters_;

    /**
     * @brief A map of short option names to their full equivalent names.
     * 
     * For example, `-m` can map to `method`, allowing convenient shorthand for commonly used options.
     */
    std::unordered_map<std::string, std::string> short_to_full_;

    std::unordered_map<std::string, std::string> default_parameters_;


    void trim(std::string& str) const {
        // trim leading and trailing whitespaces
        size_t first = str.find_first_not_of(" \t");
        size_t last = str.find_last_not_of(" \t");
        str = str.substr(first, last - first + 1);
    }
};



} // namespace gansu