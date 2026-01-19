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



#pragma once


#include <string>
#include <unordered_map>
#include <vector>


namespace gansu{


// Define the type of real number (Due to library limitations, probably not available except for double)
typedef double real_t;



/**
 * @brief Map of elements
 * @details The atomic number is given by the element name.
 */
const std::unordered_map<std::string, int> ELEMENT_NAME_TO_ATOMIC_NUMBER = {
    {"H", 1}, {"He", 2}, {"Li", 3}, {"Be", 4}, {"B", 5},
    {"C", 6}, {"N", 7}, {"O", 8}, {"F", 9}, {"Ne", 10},
    {"Na", 11}, {"Mg", 12}, {"Al", 13}, {"Si", 14}, {"P", 15},
    {"S", 16}, {"Cl", 17}, {"Ar", 18}, {"K", 19}, {"Ca", 20},
    {"Sc", 21}, {"Ti", 22}, {"V", 23}, {"Cr", 24}, {"Mn", 25},
    {"Fe", 26}, {"Co", 27}, {"Ni", 28}, {"Cu", 29}, {"Zn", 30},
    {"Ga", 31}, {"Ge", 32}, {"As", 33}, {"Se", 34}, {"Br", 35},
    {"Kr", 36}, {"Rb", 37}, {"Sr", 38}, {"Y", 39}, {"Zr", 40},
    {"Nb", 41}, {"Mo", 42}, {"Tc", 43}, {"Ru", 44}, {"Rh", 45},
    {"Pd", 46}, {"Ag", 47}, {"Cd", 48}, {"In", 49}, {"Sn", 50},
    {"Sb", 51}, {"Te", 52}, {"I", 53}, {"Xe", 54}, {"Cs", 55},
    {"Ba", 56}, {"La", 57}, {"Ce", 58}, {"Pr", 59}, {"Nd", 60},
    {"Pm", 61}, {"Sm", 62}, {"Eu", 63}, {"Gd", 64}, {"Tb", 65},
    {"Dy", 66}, {"Ho", 67}, {"Er", 68}, {"Tm", 69}, {"Yb", 70},
    {"Lu", 71}, {"Hf", 72}, {"Ta", 73}, {"W", 74}, {"Re", 75},
    {"Os", 76}, {"Ir", 77}, {"Pt", 78}, {"Au", 79}, {"Hg", 80},
    {"Tl", 81}, {"Pb", 82}, {"Bi", 83}, {"Po", 84}, {"At", 85},
    {"Rn", 86}, {"Fr", 87}, {"Ra", 88}, {"Ac", 89}, {"Th", 90},
    {"Pa", 91}, {"U", 92}, {"Np", 93}, {"Pu", 94}, {"Am", 95},
    {"Cm", 96}, {"Bk", 97}, {"Cf", 98}, {"Es", 99}, {"Fm", 100},
    {"Md", 101}, {"No", 102}, {"Lr", 103}, {"Rf", 104}, {"Db", 105},
    {"Sg", 106}, {"Bh", 107}, {"Hs", 108}, {"Mt", 109}, {"Ds", 110},
    {"Rg", 111}, {"Cn", 112}, {"Nh", 113}, {"Fl", 114}, {"Mc", 115},
    {"Lv", 116}, {"Ts", 117}, {"Og", 118}
};

inline int element_name_to_atomic_number(const std::string& element_name){
    return ELEMENT_NAME_TO_ATOMIC_NUMBER.at(element_name);
}

/**
 * @brief Map of atomic numbers
 * @details The element name is given by the atomic number.
 */
const std::vector<std::string> ATOMIC_NUMBER_MINUS_ONE_TO_ELEMENT_NAME ={
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
};

inline std::string atomic_number_to_element_name(const int atomic_number){
    return ATOMIC_NUMBER_MINUS_ONE_TO_ELEMENT_NAME[atomic_number-1];
}


/**
 * @brief Map of shell types
 * @details The shell type (Azimuthal quantum number) is given by the shell name.
 */
const std::unordered_map<std::string, int> SHELL_NAME_TO_SHELL_TYPE ={
   {"s", 0}, {"p", 1}, {"d", 2}, {"f", 3}, {"g", 4},
   {"h", 5}, {"i", 6},
   {"S", 0}, {"P", 1}, {"D", 2}, {"F", 3}, {"G", 4},
   {"H", 5}, {"I", 6},
};

inline int shell_name_to_shell_type(const std::string& shell_name){
    return SHELL_NAME_TO_SHELL_TYPE.at(shell_name);
}

/**
 * @brief Map of shell types
 * @details The shell name is given by the shell type (Azimuthal quantum number).
 */
const std::vector<std::string> SHELL_TYPE_TO_SHELL_NAME = {
    "s", "p", "d", "f", "g", "h", "i"
};

inline std::string shell_type_to_shell_name(int shell_type){
    return SHELL_TYPE_TO_SHELL_NAME[shell_type];
}

/**
 * @brief Compute the number of the basis functions from the shell type
 * @details The number of the basis functions from the shell type (Azimuthal quantum number) is given by 2*shell_type + 1.
 */
inline int shell_type_to_num_basis(int shell_type){
    return (shell_type + 1)*(shell_type + 2)/2;
}

/**
 * @brief 3D coordinate struct
 * @details This struct represents the 3D coordinates.
 */
struct Coordinate {
    real_t x, y, z; ///< x, y, z coordinate
};


/**
 * @brief Primitive Shell struct
 * @details This class represents a primitive shell.
 */
struct PrimitiveShell {
    real_t exponent; ///< Exponent of the primitive shell
    real_t coefficient; ///< Coefficient of the primitive shell
    Coordinate coordinate; ///< Center coordinate of the primitive shell
    int shell_type; ///< Azimuthal quantum number, 0=s, 1=p, 2=d, 3=f, ...
    size_t basis_index; ///< basis index (consecutive index) through all the basis functions
//    real_t cgto_normalize_factor; ///< Normalization factor of CGTO
    int atom_index; ///< Atom index (atom index in the molecule)
};


/**
 * @brief Atom struct
 * @details This class represents an atom.
 */
struct Atom {
    int atomic_number; ///< atomic number
    Coordinate coordinate; ///< coordinate=(x, y, z)
};

/**
 * @brief Constant values
 */

inline real_t angstrom_to_bohr(const real_t angstrom){
    const real_t a_to_b = 1.8897259886; ///< 1 Angstrom = 1.8897259886 Bohr
    return angstrom * a_to_b;
}

/**
 * @brief Constant values
 */

inline real_t bohr_to_angstrom(const real_t bohr){
    const real_t b_to_a = 0.529177210903; ///< 1 bohr = 0.529177210903 angstrom
    return bohr * b_to_a;
}

/**
 * @brief Shell type info in the primitive shell list
 */
struct ShellTypeInfo {
    int count; ///< Number of the shell type
    size_t start_index; ///< Offset of the shell type in the primitive shell list
};

/**
 * @brief Shell pair type info in the primitive shell pair list
 */
struct ShellPairTypeInfo {
    size_t count; ///< Number of the shell pair type
    size_t start_index; ///< Offset of the shell pair type in the primitive shell pair list
};


struct ROHF_ParameterSet {
    real_t Acc;
    real_t Bcc;
    real_t Aoo;
    real_t Boo;
    real_t Avv;
    real_t Bvv;
};

struct ROHF_Parameters{
    static const std::unordered_map<std::string, ROHF_ParameterSet> sets;
};



struct size_t2{
    size_t x;
    size_t y;
};

struct size_t4{
    size_t x;
    size_t y;
    size_t z;
    size_t w;
};

/**
 * @brief Exponents in the respective Cartesian direction
 */
const std::vector<std::vector<std::vector<int>>> AngularMomentums = {
    // s
    {{0,0,0}}, 
    // p (x, y, z)
    {{1,0,0}, {0,1,0}, {0,0,1}}, 
    // d (xx, yy, zz, xy, xz, yz)
    {{2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {1,0,1}, {0,1,1}}, 
    // f (xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz)
    {{3,0,0}, {0,3,0}, {0,0,3}, {1,2,0}, {2,1,0}, {2,0,1}, {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1}},
    // g (xxxx, yyyy, zzzz, 
    //    xxxy, xxxz, xyyy, yyyz, xzzz,yzzz, 
    //    xxyy, xxzz, yyzz, 
    //    xxyz, xyyz, xyzz)
    {
        {4,0,0}, {0,4,0}, {0,0,4}, 
        {3,1,0}, {3,0,1}, {1,3,0}, {0,3,1}, {1,0,3}, {0,1,3}, 
        {2,2,0}, {2,0,2}, {0,2,2}, 
        {2,1,1}, {1,2,1}, {1,1,2}
    },
    // h (xxxxx, yyyyy, zzzzz,
    //    xxxxy, xxxxz, xyyyy, yyyyz, xzzzz, yzzzz,
    //    xxxyy, xxxzz, xxyyy, yyyzz, xxzzz, yyzzz,
    //    xxxyz, xyyyz, xyzzz,
    //    xxyyz, xxyzz, xyyzz,
    //    
    {
        {5,0,0}, {0,5,0}, {0,0,5},
        {4,1,0}, {4,0,1}, {1,4,0}, {0,4,1}, {1,0,4}, {0,1,4},
        {3,2,0}, {3,0,2}, {2,3,0}, {0,3,2}, {2,0,3}, {0,2,3},
        {3,1,1}, {1,3,1}, {1,1,3},
        {2,2,1}, {2,1,2}, {1,2,2}
    },
    // i (xxxxxx, yyyyyy, zzzzzz,
    //    xxxxxy, xxxxxz, xyyyyy, yyyyyz, xzzzzz, yzzzzz,
    //    xxxxyy, xxxxzz, xxyyyy, yyyyzz, xxzzzz, yyzzzz,   
    //    xxxxyz, xyyyyz, xyzzzz,
    //    xxxyyy, xxxzzz, yyyzzz,
    //    xxxyyz, xxxyzz, xxyyyz, xyyyzz, xxyzzz, xyyzzz, 
    //    xxyyzz)
    {
        {6,0,0}, {0,6,0}, {0,0,6},
        {5,1,0}, {5,0,1}, {1,5,0}, {0,5,1}, {1,0,5}, {0,1,5},
        {4,2,0}, {4,0,2}, {2,4,0}, {0,4,2}, {2,0,4}, {0,2,4},
        {4,1,1}, {1,4,1}, {1,1,4},
        {3,3,0}, {3,0,3}, {0,3,3},
        {3,2,1}, {3,1,2}, {2,3,1}, {1,3,2}, {2,1,3}, {1,2,3},
        {2,2,2}
    }
};


} // namespace gansu