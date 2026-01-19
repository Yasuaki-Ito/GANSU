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
 * @file molecular.hpp
 * @brief Header file for the Molecular class.
 * @details This file contains the definition of the Molecular class.
 */

#pragma once

#include <vector>
#include <algorithm> // std::sort


#include "types.hpp"
#include "basis_set.hpp"
#include "parseXYZ.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{




/**
 * @brief Struct of basis range for the atom
 */
struct BasisRange {
    size_t start_index; ///< Start basis index of the atom
    size_t end_index; ///< End basis index of the atom
};

/**
 * @brief Molecular class
 * @details
 * This class represents a molecule.
 * The molecule is defined by the atoms and the basis set.
 */
class Molecular {
public:

    Molecular(const std::string xyz_filename, const std::string gbs_filename, const int charge=0, const unsigned int beta_to_alpha=0)
        : Molecular(parseXYZ(xyz_filename), gbs_filename, charge, beta_to_alpha) {}

    /**
     * @brief Constructor of the Molecular class
     * @param xyz_filename XYZ file name
     * @param gbs_filename Basis set file name (Gaussian basis set file)
     * @param charge Charge of the molecule. Default is 0. For cation, the charge is positive. For anion, the charge is negative.
     * @param beta_to_alpha Number of beta_to_alphaed electrons. Default is 0.
     * @details This function constructs the Molecular class.
     * @details How to calculate the number of electrons (alpha- and beta-spin electrons)
     * Given parameters:
        * \f$Z\f$ - The total number of positive charges in the nucleus of atoms in the molecule (= number of protons)
        * \f$c\f$ - The charge of the molecule
        * \f$u\f$ - The number of shifted beta_to_alpha electron spins


        The numbers of electrons (alpha- and beta-spin electrons) 
        * \f$N\f$ - The total number of electrons in the molecule
        * \f$N_{\alpha}\f$ - The number of electrons with alpha spin
        * \f$N_{\beta}\f$ - The number of electrons with beta spin

        are calculated as follows:
        * \f$ N = Z - c \f$
        * \f$ N_{\alpha} = \left\lceil \frac{N}{2} \right\rceil  + u\f$
        * \f$ N_{\beta} = \left\lfloor \frac{N}{2} \right\rfloor - u \f$

        When the number of electrons is odd, the number of alpha-spin electrons is greater than the number of beta-spin electrons by one.
        If any of the following conditions are met, an exception is thrown:
        * \f$ N < 1 \f$ (no electrons in the molecule)
        * \f$ N_{\beta} < 0 \f$ (the number of beta-spin electrons is negative)
     * @throws std::runtime_error if the number of electrons is less than one or the number of beta-spin electrons is negative.
     * @throws std::runtime_error if no atoms are given.
     */
    Molecular(const std::vector<Atom> atoms, const std::string gbs_filename, const int charge=0, const unsigned int beta_to_alpha=0)
        : atoms_(atoms), gbs_filename_(gbs_filename)
    {
        create_basis_set(gbs_filename);

        num_electrons_ = 0;
        for(const auto& atom : atoms_){
            num_electrons_ += atom.atomic_number;
        }
        num_electrons_ -= charge;
        if(num_electrons_ < 1){
            THROW_EXCEPTION("The number of electrons is less than one.");
        }

        num_alpha_spins_ = static_cast<int>((num_electrons_+1)/2) +  beta_to_alpha; // the number of alpha-spin electrons
        num_beta_spins_  = static_cast<int>(num_electrons_/2) - beta_to_alpha; // the number of beta-spin electrons
        
        if(num_beta_spins_ < 0){
            THROW_EXCEPTION("The number of beta-spin electrons is less than zero.");
        }


    }

    /**
     * @brief Create the basis set
     */
    void create_basis_set(const std::string gbs_filename){
        BasisSet basis_set = BasisSet::construct_from_gbs(gbs_filename);

        if(atoms_.size() == 0){
            THROW_EXCEPTION("No atoms are given.");
        }

        size_t basis_index = 0; ///< Basis index (consecutive number through all the basis functions)
//        for(const auto& atom : atoms_){ // loop over atoms
        for(int atom_index=0; atom_index<atoms_.size(); atom_index++){
            const auto& atom = atoms_[atom_index];

            BasisRange basis_range;
            basis_range.start_index = basis_index;

            const ElementBasisSet& element_basis_set = basis_set.get_element_basis_set(atomic_number_to_element_name(atom.atomic_number));
            for(size_t i=0; i<element_basis_set.get_num_contracted_gausses(); i++){ // loop over basis function (contracted Gauss functions)
                const ContractedGauss& contracted_gauss = element_basis_set.get_contracted_gauss(i);

                const size_t num_primitives = contracted_gauss.get_num_primitives();

                const int shell_type = shell_name_to_shell_type(contracted_gauss.get_type());

                for(size_t j=0; j<num_primitives; j++){
                    const PrimitiveGauss& primitive = contracted_gauss.get_primitive_gauss(j);

                    PrimitiveShell primitive_shell {
                        .exponent = primitive.exponent,
                        .coefficient = primitive.coefficient,
                        .coordinate = atom.coordinate,
                        .shell_type = shell_type,
                        .basis_index = basis_index,
                        .atom_index = atom_index
                    };
                    primitive_shells_.push_back(primitive_shell);
                }
                basis_index += shell_type_to_num_basis(shell_type);

                // store the normalization factor of the contracted Gauss function
                const std::vector<real_t> normalization_factors = contracted_gauss.get_normalization_factor();
                cgto_normalization_factors_.insert(cgto_normalization_factors_.end(), normalization_factors.begin(), normalization_factors.end());
            }
            basis_range.end_index = basis_index;
            atom_to_basis_range_.push_back(basis_range);
        }

        num_basis_ = basis_index;

        if(num_basis_ != cgto_normalization_factors_.size()){
            THROW_EXCEPTION("The number of basis functions is not equal to the number of normalization factors.");
        }


        // initialization for the primitive shells and the shell type counts

        // sort the primitive shells by the shell type (Azimuthal quantum number)
        //std::sort(primitive_shells_.begin(), primitive_shells_.end(), 
        //    [](const PrimitiveShell& a, const PrimitiveShell& b){return a.shell_type < b.shell_type;});
        std::sort(primitive_shells_.begin(), primitive_shells_.end(), [](const PrimitiveShell& a, const PrimitiveShell& b) {
            if (a.shell_type != b.shell_type) {
                return a.shell_type < b.shell_type;
            }
            else {
                return a.basis_index < b.basis_index;
            }
        });

        
        // count and store the shell type information
        int max_shell_type = primitive_shells_[primitive_shells_.size()-1].shell_type;
        shell_type_infos_.resize(max_shell_type+1, {0, 0});
        for(size_t i=0; i<primitive_shells_.size(); i++){
            shell_type_infos_[primitive_shells_[i].shell_type].count++;
        }
        shell_type_infos_[0].start_index = 0;
        for(size_t i=1; i<shell_type_infos_.size(); i++){
            shell_type_infos_[i].start_index = shell_type_infos_[i-1].start_index + shell_type_infos_[i-1].count;
        }
    }

    /**
     * @brief Get the number of basis functions
     */
    size_t get_num_basis() const { return num_basis_;}

    /**
     * @brief Get the list of atoms
     */
    const std::vector<Atom>& get_atoms() const { return atoms_; }

    /**
     * @brief Get the list of primitive shells
     */
    const std::vector<PrimitiveShell>& get_primitive_shells() const { return primitive_shells_; }

    /**
     * @brief Get the list of the numbers of shell types
     */
    const std::vector<ShellTypeInfo>& get_shell_type_infos() const { return shell_type_infos_; }

    /**
     * @brief Get the list of the basis range for each atom
     */
    const std::vector<BasisRange>& get_atom_to_basis_range() const { return atom_to_basis_range_; }

    /**
     * @brief Get the number of electrons
     */
    int get_num_electrons() const { return num_electrons_; }

    /**
     * @brief Get the number of alpha-spin electrons
     */
    int get_num_alpha_spins() const { return num_alpha_spins_; }

    /**
     * @brief Get the number of beta-spin electrons
     */
    int get_num_beta_spins() const { return num_beta_spins_; }


    /**
     * @brief Get the basis set file name (Gaussian basis set file)
     * @return Basis set file name (Gaussian basis set file)
     */
    std::string get_gbs_filename() const { return gbs_filename_; }

    /**
     * @brief Get the list of the normalization factors of the contracted Gauss functions
    */
    const std::vector<real_t>& get_cgto_normalization_factors() const { return cgto_normalization_factors_; }



    Molecular(const Molecular&) = delete; ///< copy constructor is deleted
    ~Molecular() = default; ///< destructor

    void dump() const {
        std::cout << "Number of atoms: " << atoms_.size() << std::endl;

        for(size_t i=0; i<atoms_.size(); i++){
            const auto& atom = atoms_[i];
            std::cout << "Atom[" << i << "]: {" << atomic_number_to_element_name(atom.atomic_number) << ", (" << atom.coordinate.x << ", " << atom.coordinate.y << ", " << atom.coordinate.z << ")}";
            std::cout << " Basis range: {" << atom_to_basis_range_[i].start_index << ", " << atom_to_basis_range_[i].end_index << "}"  << std::endl;
        }

        std::cout << "Number of electrons: " << num_electrons_ << std::endl;
        std::cout << "Number of alpha-spin electrons: " << num_alpha_spins_ << std::endl;
        std::cout << "Number of beta-spin electrons: " << num_beta_spins_ << std::endl;

        std::cout << "Number of basis functions: " << num_basis_ << std::endl;
        for(size_t i=0; i<cgto_normalization_factors_.size(); i++){
            std::cout << "Normalization factor[" << i << "]: " << cgto_normalization_factors_[i] << std::endl;
        }


        std::cout << "Number of primitive shells: " << primitive_shells_.size() << std::endl;

        // print properties of the primitive shells
        for(size_t i=0; i<primitive_shells_.size(); i++){
            const auto& primitive_shell = primitive_shells_[i];
            std::cout << "Primitive shell[" << i << "]: {" << primitive_shell.exponent << ", " << primitive_shell.coefficient << ", (" << primitive_shell.coordinate.x << ", " << primitive_shell.coordinate.y << ", " << primitive_shell.coordinate.z << "), " << shell_type_to_shell_name(primitive_shell.shell_type) << ", " << primitive_shell.basis_index << "," << "}" << std::endl;
        }

        // print the number of shell types
        for(size_t i=0; i<shell_type_infos_.size(); i++){
            std::cout << "Shell type[" << i << "] (" << shell_type_to_shell_name(i) << "-type orbital): " << shell_type_infos_[i].count << ", " << shell_type_infos_[i].start_index << std::endl;
        }

    }

private:
    std::vector<Atom> atoms_; ///< Atoms
    std::vector<PrimitiveShell> primitive_shells_; ///< Primitive shells
    std::vector<ShellTypeInfo> shell_type_infos_; ///< The list of shell type information
    std::vector<BasisRange> atom_to_basis_range_; ///< The list of the basis range for each atom
    size_t num_basis_; ///< Number of basis functions

    std::vector<real_t> cgto_normalization_factors_; ///< The list of the normalization factors of the contracted Gauss functions

    int num_electrons_; ///< Number of electrons
    int num_alpha_spins_; ///< Number of alpha-spin electrons
    int num_beta_spins_; ///< Number of beta-spin electrons

    const std::string gbs_filename_; ///< Basis set file name (Gaussian basis set file)

};


} // namespace gansu