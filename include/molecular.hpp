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
        // First compute full-electron count (needed by create_basis_set for ECP adjustment)
        num_electrons_ = 0;
        for(const auto& atom : atoms_){
            num_electrons_ += atom.atomic_number;
        }
        num_electrons_ -= charge;

        // Parse basis set (and ECP if present — adjusts num_electrons_)
        create_basis_set(gbs_filename);

        if(num_electrons_ < 1){
            THROW_EXCEPTION("The number of electrons is less than one.");
        }

        num_alpha_spins_ = static_cast<int>((num_electrons_+1)/2) +  beta_to_alpha;
        num_beta_spins_  = static_cast<int>(num_electrons_/2) - beta_to_alpha;

        if(num_beta_spins_ < 0){
            THROW_EXCEPTION("The number of beta-spin electrons is less than zero.");
        }
    }

    /**
     * @brief Constructor of the Molecular class from a BasisSet object (without GBS file)
     * @param atoms List of atoms
     * @param basis_set Pre-constructed basis set
     * @param charge Charge of the molecule
     * @param beta_to_alpha Number of beta_to_alpha electrons
     */
    Molecular(const std::vector<Atom> atoms, const BasisSet& basis_set, const int charge=0, const unsigned int beta_to_alpha=0)
        : atoms_(atoms), gbs_filename_("")
    {
        create_basis_set_from_object(basis_set);

        num_electrons_ = 0;
        for(const auto& atom : atoms_){
            num_electrons_ += atom.atomic_number;
        }
        num_electrons_ -= charge;
        if(num_electrons_ < 1){
            THROW_EXCEPTION("The number of electrons is less than one.");
        }

        num_alpha_spins_ = static_cast<int>((num_electrons_+1)/2) + beta_to_alpha;
        num_beta_spins_  = static_cast<int>(num_electrons_/2) - beta_to_alpha;

        if(num_beta_spins_ < 0){
            THROW_EXCEPTION("The number of beta-spin electrons is less than zero.");
        }
    }

    /**
     * @brief Create the basis set
     */
    void create_basis_set(const std::string gbs_filename){
        BasisSet basis_set = BasisSet::construct_from_gbs(gbs_filename);
        create_basis_set_from_object(basis_set);

        // If the GBS file contains ECP sections, load them
        const auto& ecps = basis_set.get_all_ecps();
        if (!ecps.empty()) {
            int n_ecp_electrons = 0;
            effective_charges_.resize(atoms_.size());

            for (size_t i = 0; i < atoms_.size(); i++) {
                std::string elem = atomic_number_to_element_name(atoms_[i].atomic_number);
                auto it = ecps.find(elem);
                if (it != ecps.end()) {
                    ecp_data_[elem] = it->second;
                    int n_core = it->second.get_n_core_electrons();
                    n_ecp_electrons += n_core;
                    effective_charges_[i] = atoms_[i].atomic_number - n_core;
                    atoms_[i].effective_charge = effective_charges_[i];
                } else {
                    effective_charges_[i] = atoms_[i].atomic_number;
                }
            }

            if (n_ecp_electrons > 0) {
                num_electrons_ -= n_ecp_electrons;
                num_alpha_spins_ = static_cast<int>((num_electrons_+1)/2);
                num_beta_spins_  = static_cast<int>(num_electrons_/2);
                has_ecp_ = true;
                std::cout << "[ECP] " << n_ecp_electrons << " core electrons replaced, "
                          << num_electrons_ << " valence electrons" << std::endl;
                for (size_t i = 0; i < atoms_.size(); i++) {
                    if (effective_charges_[i] != atoms_[i].atomic_number) {
                        std::cout << "  " << atomic_number_to_element_name(atoms_[i].atomic_number)
                                  << ": Z_eff=" << effective_charges_[i] << std::endl;
                    }
                }
            }
        }
    }

    /**
     * @brief Create basis set from a BasisSet object
     */
    void create_basis_set_from_object(const BasisSet& basis_set){

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
                // DEBUG: print shell info with primitive details
                if (atom_index == 1 && shell_type >= 1) {
                    std::cout << "[CGTO-DBG] atom=" << atom_index << " type=" << shell_type
                              << " nprim=" << contracted_gauss.get_num_primitives()
                              << " basis=" << (basis_index - shell_type_to_num_basis(shell_type))
                              << " norm=" << normalization_factors[0];
                    for (size_t jp = 0; jp < contracted_gauss.get_num_primitives(); jp++)
                        std::cout << " [exp=" << contracted_gauss.get_primitive_gauss(jp).exponent
                                  << " c=" << contracted_gauss.get_primitive_gauss(jp).coefficient << "]";
                    std::cout << std::endl;
                }
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

    /// Load ECP data from file and adjust electron count
    void load_ecp(const std::string& ecp_filename) {
        auto ecps = BasisSet::parse_ecp_file(ecp_filename);
        int n_ecp_electrons = 0;
        ecp_data_.clear();
        effective_charges_.resize(atoms_.size());

        for (size_t i = 0; i < atoms_.size(); i++) {
            std::string elem = atomic_number_to_element_name(atoms_[i].atomic_number);
            auto it = ecps.find(elem);
            if (it != ecps.end()) {
                ecp_data_[elem] = it->second;
                int n_core = it->second.get_n_core_electrons();
                n_ecp_electrons += n_core;
                effective_charges_[i] = atoms_[i].atomic_number - n_core;
                std::cout << "  ECP: " << elem << " Z_eff=" << effective_charges_[i]
                          << " (removed " << n_core << " core electrons)" << std::endl;
            } else {
                effective_charges_[i] = atoms_[i].atomic_number;  // no ECP
            }
        }

        // Adjust electron count
        num_electrons_ -= n_ecp_electrons;
        num_alpha_spins_ = static_cast<int>((num_electrons_+1)/2);
        num_beta_spins_  = static_cast<int>(num_electrons_/2);

        has_ecp_ = (n_ecp_electrons > 0);
        if (has_ecp_) {
            std::cout << "  ECP total: " << n_ecp_electrons << " core electrons removed, "
                      << num_electrons_ << " valence electrons remain" << std::endl;
        }
    }

    bool has_ecp() const { return has_ecp_; }

    /// Get effective nuclear charge for atom i (Z - n_core if ECP, else Z)
    int get_effective_charge(int atom_index) const {
        if (effective_charges_.empty()) return atoms_[atom_index].atomic_number;
        return effective_charges_[atom_index];
    }

    /// Get ECP data for an element (returns nullptr if not found)
    const ElementECP* get_ecp(const std::string& element_name) const {
        auto it = ecp_data_.find(element_name);
        return (it != ecp_data_.end()) ? &it->second : nullptr;
    }

    /// Get ECP data for atom by index
    const ElementECP* get_atom_ecp(int atom_index) const {
        std::string elem = atomic_number_to_element_name(atoms_[atom_index].atomic_number);
        return get_ecp(elem);
    }

    const std::unordered_map<std::string, ElementECP>& get_all_ecps() const { return ecp_data_; }

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

    // ECP data
    bool has_ecp_ = false;
    std::unordered_map<std::string, ElementECP> ecp_data_;
    std::vector<int> effective_charges_;  // Z_eff for each atom (Z - n_core if ECP)
};


} // namespace gansu