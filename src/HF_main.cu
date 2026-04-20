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
 * @file HF_main.cu
 * @brief Main function for the Hartree-Fock method.
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <set>

#include "hf.hpp"
#include "parameter_manager.hpp"
#include "builder.hpp"
#include "gpu_manager.hpp"

namespace fs = std::filesystem;
using namespace gansu;

/**
 * @brief Resolve basis set name to full path.
 *
 * If the given path already exists as a file, return it as-is.
 * Otherwise, search for "<name>" and "<name>.gbs" in standard locations
 * relative to the executable: ../basis/, ./basis/, and the executable directory.
 */
static std::string resolve_basis_path(const std::string& name_or_path, const char* argv0) {
    // Already a valid file path
    if (fs::exists(name_or_path)) return name_or_path;

    // Candidate file names
    std::vector<std::string> names = {name_or_path};
    if (name_or_path.find(".gbs") == std::string::npos)
        names.push_back(name_or_path + ".gbs");

    // Search directories relative to executable
    fs::path exe_dir = fs::path(argv0).parent_path();
    if (exe_dir.empty()) exe_dir = ".";
    std::vector<fs::path> search_dirs = {
        exe_dir / ".." / "basis",      // build/ → ../basis/
        exe_dir / "basis",             // same dir
        fs::path("basis"),             // cwd/basis/
        fs::path("../basis"),          // cwd/../basis/
    };

    for (const auto& dir : search_dirs)
        for (const auto& n : names) {
            auto p = dir / n;
            if (fs::exists(p)) return p.string();
        }

    // Not found — return original (will produce a clear error from GANSU)
    return name_or_path;
}

/**
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Arguments
 * @return 0 if the program ends successfully
 * @details This function reads the command line arguments and calls the RHF or UHF class.
 */
int main(int argc, char* argv[]){
  try {
    // Check for special flags before parameter parsing
    bool force_cpu = false;
    bool list_basis = false;
    std::vector<char*> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--cpu") {
            force_cpu = true;
        } else if (std::string(argv[i]) == "--list-basis") {
            list_basis = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    // List available basis sets and exit
    if (list_basis) {
        fs::path exe_dir = fs::path(argv[0]).parent_path();
        if (exe_dir.empty()) exe_dir = ".";
        std::vector<fs::path> search_dirs = {
            exe_dir / ".." / "basis",
            exe_dir / "basis",
            fs::path("basis"),
            fs::path("../basis"),
        };
        std::set<std::string> names;
        for (const auto& dir : search_dirs) {
            if (!fs::is_directory(dir)) continue;
            for (const auto& entry : fs::directory_iterator(dir)) {
                auto fn = entry.path().filename().string();
                if (fn.size() > 4 && fn.substr(fn.size()-4) == ".gbs")
                    names.insert(fn.substr(0, fn.size()-4));
            }
        }
        for (const auto& n : names) std::cout << n << std::endl;
        return 0;
    }

    if (force_cpu) {
        gpu::disable_gpu();
    } else {
        gpu::initialize_gpu(); // Detect GPU availability (CPU fallback if no GPU)
    }

    ParameterManager parameters;
    parameters.parse_command_line_args(filtered_argc, filtered_argv.data());

    // Resolve basis set short names: -g cc-pvdz → ../basis/cc-pvdz.gbs
    if (parameters.contains("gbsfilename")) {
        std::string gbs = parameters.get<std::string>("gbsfilename");
        if (!gbs.empty()) {
            parameters["gbsfilename"] = resolve_basis_path(gbs, argv[0]);
        }
    }

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

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}