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


#include "parseXYZ.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{

/**
 * @brief Reads an XYZ file and returns a list of atomic information.
 * 
 * @param filename Path to the XYZ file
 * @return std::vector<Atom> List of atomic information
 * @throws std::runtime_error If the file cannot be opened or has an invalid format
 * @details The XYZ file format is as follows:
 * @details - Line 1: Number of atoms
 * @details - Line 2: Comment line (ignored)
 * @details - Remaining lines: Atomic data (symbol, x, y, z)
 * @details - Example:
 * @code
 * 3
 * Water molecule
 * O          0.00000        0.00000        0.11779
 * H          0.00000        0.75545       -0.47116
 * H          0.00000       -0.75545       -0.47116
 * @endcode
 * @details The atomic coordinates are converted from angstrom to bohr.
 * @details The atomic symbols are converted to atomic numbers.
 * @details Multiple molecules are not supported.
 */
std::vector<Atom> parseXYZ(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        THROW_EXCEPTION("Failed to open xyz file: " + filename);
    }

    std::vector<Atom> atoms;
    std::string line;

    // First line: Number of atoms
    if (!std::getline(file, line)) {
        THROW_EXCEPTION("Failed to read atom count from file: " + filename);
    }

    int atom_count = 0;
    try {
        atom_count = std::stoi(line);
    } catch (...) {
        THROW_EXCEPTION("Invalid atom count in file: " + filename);
    }

    // Second line: Comment line (ignored)
    if (!std::getline(file, line)) {
        THROW_EXCEPTION("Failed to read comment line from file: " + filename);
    }

    // Remaining lines: Atomic data
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string symbol;
        Coordinate coordinate;

        // skip empty lines
        if (line.empty()) {
            continue;
        }

        if (!(iss >> symbol >> coordinate.x >> coordinate.y >> coordinate.z)) {
            THROW_EXCEPTION("Invalid line format in file: " + line);
        }

        // Convert angstrom to bohr
        coordinate.x = angstrom_to_bohr(coordinate.x);
        coordinate.y = angstrom_to_bohr(coordinate.y);
        coordinate.z = angstrom_to_bohr(coordinate.z);

        Atom atom{
            .atomic_number = element_name_to_atomic_number(symbol),
            .coordinate = coordinate
        };

        atoms.push_back(atom);
    }

    // Verify that the number of atoms matches the declared count
    if (atoms.size() != static_cast<size_t>(atom_count)) {
        THROW_EXCEPTION("Atom count mismatch in file: " + filename);
    }

    return atoms;
}



} // namespace gansu