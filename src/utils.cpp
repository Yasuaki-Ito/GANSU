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

#include "utils.hpp"

namespace gansu{

// Read alpha/beta densities from .sad file for a specific element
namespace fs = std::filesystem;
std::pair<std::vector<double>, std::vector<double>> read_density_from_sad(
    const std::string& element,
    const std::string& basis_path,
    int& num_basis_out
) {
    fs::path sad_path = basis_path;
    sad_path.replace_extension(".sad");

    std::ifstream in(sad_path);
    if (!in) throw std::runtime_error("Failed to open SAD file: " + sad_path.string());

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string tag;
        int n;
        if (!(iss >> tag >> n)) continue;
        if (tag != element) {
            for (int i = 0; i < 2 * n; ++i) std::getline(in, line);
            continue;
        }

        std::vector<double> alpha(n * n), beta(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                in >> alpha[i * n + j];
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                in >> beta[i * n + j];

        num_basis_out = n;
        return {alpha, beta};
    }

    throw std::runtime_error("Element not found in SAD file: " + element);
}


} // namespace gansu