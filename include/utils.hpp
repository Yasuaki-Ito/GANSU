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


/**
 * @file utils.hpp
 * @brief Constants and utility functions.
 */

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <cstring>
#include <filesystem> // std::filesystem::path (C++17)
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <unordered_map>

namespace gansu{


// protoype of functions
std::pair<std::vector<double>, std::vector<double>> read_density_from_sad(const std::string& element, const std::string& basis_path, int& num_basis_out);

/**
 * @brief Exception class
 * @details This class is used to throw exceptions with the file name, line number, and function name.
 * @details The exception message is formatted as follows:
 * @details "=============================================="
 * @details "Message   : <message>"
 * @details "File      : <file>"
 * @details "Line      : <line>"
 * @details "Function  : <function>"
 * @details "=============================================="
 */
class Exception : public std::exception {
public:
    Exception(const std::string& message, const std::string& file, int line, const std::string& function)
        : message_(message), file_(file), line_(line), function_(function) {
        std::ostringstream oss;
        oss << "\n"
            << "================== Exception ==================\n"
            << "Message   : " << message_ << "\n"
            << "File      : " << file_ << "\n"
            << "Line      : " << line_ << "\n"
            << "Function  : " << function_ << "\n"
            << "==============================================";
        full_message_ = oss.str();
    }

    const char* what() const noexcept override {
        return full_message_.c_str();
    }

private:
    std::string message_;
    std::string file_;
    int line_;
    std::string function_;
    std::string full_message_;
};


/**
 * @brief Macro to get the file name without the path
 * @details This macro gets the file name without the path.
 */
#define FILENAME (std::strrchr(__FILE__, '/') ? std::strrchr(__FILE__, '/') + 1 : __FILE__)

/**
 * @brief Macro to throw an exception with the file name, line number, and function name.
 * @param message Exception message
 * @details This macro throws an exception with the file name, line number, and function name.
 */
#define THROW_EXCEPTION(message) throw Exception((message), FILENAME, __LINE__, __func__)


inline std::string addSuffixBeforeExtension(const std::string& filename, const std::string& suffix){
    std::filesystem::path path(filename);
    std::string new_filename = path.stem().string() + suffix + path.extension().string();
    return new_filename;
}




} // namespace gansu