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
 * @brief  This function is used to calculate the ROHF energy
 */

#include "types.hpp"
#include "rohf.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION

#include <limits> // numeric_limits<double>::max();
#include <iomanip> // std::setprecision

namespace gansu{

const std::unordered_map<std::string, ROHF_ParameterSet> ROHF_Parameters::sets = {
    {"roothaan",                      {-1.0/2.0, 3.0/2.0, 1.0/2.0, 1.0/2.0, 3.0/2.0, -1.0/2.0}},
    {"mcweeny-diercksen",             { 1.0/3.0, 2.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0,  1.0/3.0}},
    {"davidson",                      { 1.0/2.0,     1.0,     0.0,     1.0,     1.0,      0.0}},
    {"guest-saunders",                { 1.0/2.0, 1.0/2.0, 1.0/2.0, 1.0/2.0, 1.0/2.0,  1.0/2.0}},
    {"binkley-pople-dobosh",          { 1.0/2.0, 1.0/2.0,     1.0,     0.0,     0.0,      1.0}},
    {"faegri-manne",                  { 1.0/2.0, 1.0/2.0,     1.0,     0.0, 1.0/2.0,  1.0/2.0}},
    {"goddard",                       { 1.0/2.0, 1.0/2.0, 1.0/2.0,     0.0,     1.0,      0.0}},
    {"plakhutin-gorelik-breslavskaya",{     0.0,     1.0,     1.0,     0.0,     1.0,      0.0}},
};

 
/**
 * @brief Constructor of the ROHF class
 * @param molecular Molecular
 * @param verbose Verbose mode
 * @details This function constructs the ROHF class.
 * @details The molecular is given as an argument.
 */
ROHF::ROHF(const Molecular& molecular, const ParameterManager& parameters) : 
    HF(molecular, parameters),
    coefficient_matrix(num_basis, num_basis),
    density_matrix_closed(num_basis, num_basis),
    density_matrix_open(num_basis, num_basis),
    density_matrix(num_basis, num_basis),
    fock_matrix_closed(num_basis, num_basis),
    fock_matrix_open(num_basis, num_basis),
    fock_matrix(num_basis, num_basis),
    orbital_energies(num_basis),
    num_closed(molecular.get_num_beta_spins()),
    num_open(molecular.get_num_alpha_spins() - molecular.get_num_beta_spins()),
    num_virtual(num_basis - num_open - num_closed),
    ROH_parameter_name(parameters.get<std::string>("rohf_parameter_name")),
    ROHF_parameters(ROHF_Parameters::sets.at(ROH_parameter_name)),
    initial_guess_method_(parameters.get<std::string>("initial_guess")),
    gbsfilename_(parameters.get<std::string>("gbsfilename"))
{
    // check the number of alpha and beta electrons and the number of basis functions
    if(molecular.get_num_alpha_spins() > num_basis || molecular.get_num_beta_spins() > num_basis){
        THROW_EXCEPTION("The number of alpha or beta electrons should be less than or equal to the number of basis functions.");
    }


    // Set an algorithm to update the Fock matrix (default: Convergence_ROHF_DIIS)
    const std::string convergence_method = parameters.get<std::string>("convergence_method");
    if(convergence_method == "damping"){ // damping algorithm with a fixed damping factor
        const double damping_factor = parameters.get<double>("damping_factor"); // damping factor
        set_convergence_method(std::make_unique<Convergence_ROHF_Damping>(*this, damping_factor));
    }else if(convergence_method == "optimaldamping"){ // optimal damping factor
        set_convergence_method(std::make_unique<Convergence_ROHF_Damping>(*this));
    }else if(convergence_method == "diis"){ // DIIS algorithm
        const size_t DIIS_size = parameters.get<size_t>("diis_size"); // DIIS size (number of previous Fock matrices)
        const bool is_include_transform = parameters.get<bool>("diis_include_transform"); // include the transformation matrix in DIIS

        set_convergence_method(std::make_unique<Convergence_ROHF_DIIS>(*this, DIIS_size, is_include_transform));
    }else{
        THROW_EXCEPTION("Invalid convergence_method: " + convergence_method);
    }

    // Set an ERI method to calculate the two-electron integrals
    const std::string eri_method = parameters.get<std::string>("eri_method");
    if(eri_method == "stored"){ // stored two-electron integrals
        set_eri_method(std::make_unique<ERI_Stored_ROHF>(*this));
    }else if(eri_method == "ri"){ // RI (Resolution of Identity) method
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename"); // auxiliary basis set file name
        Molecular auxiliary_molecular(molecular.get_atoms(), auxiliary_gbsfilename); // auxiliary molecular object
        set_eri_method(std::make_unique<ERI_RI_ROHF>(*this, auxiliary_molecular));
    }else{
        THROW_EXCEPTION("Invalid eri_method: " + eri_method);
    }
}
 
/**
 * @brief Set a convergence method to update the Fock matrix
 * @param convergence_method UpdateFockMatrix_ROHF
 * @details This function sets an algorithm to update the Fock matrix.
 */
void ROHF::set_convergence_method(std::unique_ptr<Convergence_ROHF> convergence_method) {
    this->convergence_method_ = std::move(convergence_method);
}

/**
 * @brief Set an ERI method to calculate the two-electron integrals
 * @param eri_method ERI
 * @details This function sets an algorithm to calculate the two-electron integrals.
 */
void ROHF::set_eri_method(std::unique_ptr<ERI> eri_method) {
    this->eri_method_ = std::move(eri_method);
}
 

void ROHF::precompute_eri_matrix(){
    PROFILE_FUNCTION();

    eri_method_->precomputation();
}

 
/**
 * @brief Function to guess the initial Fock matrix
 * @details This function calculates the initial Fock matrix using the core Hamiltonian matrix.
 */
void ROHF::guess_initial_fock_matrix(const real_t* density_matrix_a, const real_t* density_matrix_b, bool force_density) {
    PROFILE_FUNCTION();

    std::unique_ptr<InitialGuess_ROHF> initial_guess; // the life time is only here since initial guess is performed only once

    if(force_density == true || initial_guess_method_ == "density"){ // Initialized by the precomputed density matrix
        if(density_matrix_a == nullptr || density_matrix_b == nullptr){
            std::cerr << "The density matrix is not provided even though ``density'' is set to ``initial_guess_method'' or force_density=true. The core Hamiltonian matrix is used instead." << std::endl;
            initial_guess = std::make_unique<InitialGuess_ROHF_Core>(*this);
        }else{
            initial_guess = std::make_unique<InitialGuess_ROHF_Density>(*this, density_matrix_a, density_matrix_b);
        }
    }else if(initial_guess_method_ == "core"){ // core Hamiltonian matrix
        initial_guess = std::make_unique<InitialGuess_ROHF_Core>(*this);
    }else if(initial_guess_method_ == "gwh"){ // Generalized Wolfsberg-Helmholz (GWH) method
        initial_guess = std::make_unique<InitialGuess_ROHF_GWH>(*this);
    }else if(initial_guess_method_ == "sad"){ // Superposition of Atomic Densities (SAD) method
        if(gbsfilename_.empty()){
            THROW_EXCEPTION("The basis set file is not specified for SAD initial guess method. Please specify the basis set file name by -gbsfilename option.");
        }
        initial_guess = std::make_unique<InitialGuess_ROHF_SAD>(*this);
    }else{
        THROW_EXCEPTION("Invalid initial guess method: " + initial_guess_method_);
    }

    // Execute the initial guess method
    initial_guess->guess();
}
 
/**
 * @brief Function to calculate the coefficient matrix
 * @details This function calculates the coefficient matrix using the eigenvectors of the Fock matrix.
 */
void ROHF::compute_coefficient_matrix_impl() {
    PROFILE_FUNCTION();
    // compute coefficient matrix C
    // The function computeEigenvectors performs the following operations:
    //   1. symmetrize the Fock matrix F' = X^T F X
    //   2. diagonalize the symmetrized Fock matrix F'C' = C'E
    //   3. obtain the coefficient matrix from the eigenvectors C = X C'
    gpu::computeCoefficientMatrix(
        fock_matrix.device_ptr(), 
        transform_matrix.device_ptr(),
        coefficient_matrix.device_ptr(),
        num_basis,
        orbital_energies.device_ptr()
    );

    if (verbose) {
        std::cout << "Orbital energies:" << std::endl;
        orbital_energies.toHost();
        for(size_t i=0; i<num_basis; i++){
            std::cout << orbital_energies[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Coefficient matrix:" << std::endl;
        coefficient_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << coefficient_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}
 
/**
 * @brief Function to calculate the density matrix
 * @details This function calculates the density matrix using the coefficient matrix.
 */
void ROHF::compute_density_matrix() {
    PROFILE_FUNCTION();

    gpu::computeDensityMatrix_ROHF(
        coefficient_matrix.device_ptr(),
        density_matrix_closed.device_ptr(),
        density_matrix_open.device_ptr(),
        density_matrix.device_ptr(),
        num_closed,
        num_open,
        num_basis
    );
    

    if (verbose) {
        std::cout << "Density matrix (closed-shell):" << std::endl;
        density_matrix_closed.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << density_matrix_closed(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Density matrix (open-shell):" << std::endl;
        density_matrix_open.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << density_matrix_open(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Density matrix (sum of closed-shell and open-shell):" << std::endl;
        density_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << density_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}
 
 
/**
 * @brief Function to calculate the Fock matrix
 * @details This function calculates the Fock matrix using the core Hamiltonian matrix, the density matrix, and the two-electron integrals.
 */
void ROHF::compute_fock_matrix(){
    PROFILE_FUNCTION();
 
    eri_method_->compute_fock_matrix();
}
 
 
/**
 * @brief Function to calculate the energy
 * @details This function calculates the energy using the core Hamiltonian matrix, the density matrix, and the Fock matrix.
 */
void ROHF::compute_energy(){
    PROFILE_FUNCTION();

    energy_ = gpu::computeEnergy_ROHF(
        density_matrix_closed.device_ptr(),
        density_matrix_open.device_ptr(),
        core_hamiltonian_matrix.device_ptr(),
        fock_matrix_closed.device_ptr(),
        fock_matrix_open.device_ptr(),
        num_basis
    );

    if(verbose){
        std::cout << "Energy: " << energy_ << std::endl;
        std::cout << "Total energy: " << get_total_energy() << std::endl;
    }
}
 
/**
 * @brief Update the Fock matrix
 * @details This function updates the Fock matrix
 */
 void ROHF::update_fock_matrix(){
    PROFILE_FUNCTION();

    convergence_method_->get_new_fock_matrix();
}

void ROHF::export_density_matrix(real_t* density_matrix_a, real_t* density_matrix_b, const int num_basis) {
    if(num_basis != this->num_basis){
        THROW_EXCEPTION("The number of basis functions is different.");
    }

    // copy the density matrix to the host memory
    density_matrix_closed.toHost();
    density_matrix_open.toHost();
    {   // alpha-spin density matrix
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                density_matrix_a[i*num_basis+j] = 0.5 * density_matrix_closed(i, j) + density_matrix_open(i, j);
            }
        }
    }
    {   // beta-spin density matrix
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                density_matrix_b[i*num_basis+j] = 0.5 * density_matrix_closed(i, j);
            }
        }
    }
}


/**
 * @brief Print the results of the SCF procedure
 * @details This function prints the results of the SCF procedure.
 */
 void ROHF::report() {
    
    HF::report(); // prints the information of the input molecular and basis set

    if(is_mulliken_analysis_){
        std::cout << std::endl;
        std::cout << "[Mulliken population]" << std::endl;
        const auto& mulliken_population = analyze_mulliken_population();
        for(size_t i=0; i<atoms.size(); i++){
            std::cout << "Atom " << i << " " << atomic_number_to_element_name(atoms[i].atomic_number) << ": " << std::setprecision(6) << mulliken_population[i] << std::endl;
        }
    }

    if(is_mayer_bond_order_analysis_){ // print Mayer bond order matrix
        std::cout << std::endl;
        std::cout << "[Mayer bond order]" << std::endl;
        const auto& mayer_bond_order_matrix = compute_mayer_bond_order();

        // save the current format flags and precision
        std::ios::fmtflags old_flags = std::cout.flags();
        std::streamsize old_precision = std::cout.precision();

        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << std::fixed << std::setprecision(3) << mayer_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // restore the format flags and precision
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
    }

    if(is_wiberg_bond_order_analysis_){ // print Wiberg bond order matrix
        std::cout << std::endl;
        std::cout << "[Wiberg bond order]" << std::endl;
        const auto& wiberg_bond_order_matrix = compute_wiberg_bond_order();

        // save the current format flags and precision
        std::ios::fmtflags old_flags = std::cout.flags();
        std::streamsize old_precision = std::cout.precision();

        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << std::fixed << std::setprecision(3) << wiberg_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // restore the format flags and precision
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
    }


    std::cout << std::endl;
    std::cout << "[Calculation Summary]" << std::endl;
    std::cout << "Method: Restricted Open-shell Hartree-Fock (ROHF)" << std::endl;
    std::cout << "Schwarz screening threshold: " << schwarz_screening_threshold << std::endl;
    std::cout << "Initial guess method: " << initial_guess_method_ << std::endl;
    std::cout << "Convergence algorithm: " << convergence_method_->get_algorithm_name() << std::endl;
    std::cout << "Number of iterations: " << iter_ << std::endl;
    std::cout << "Convergence criterion: " << convergence_energy_threshold << std::endl;
    std::cout << "Energy difference: " << energy_difference_ << std::endl;
    std::cout << "Energy (without nuclear repulsion): " << std::setprecision(17) << get_energy() << " [hartree]" << std::endl;
    std::cout << "Total Energy: " << std::setprecision(17) << get_total_energy() << " [hartree]" << std::endl;
    std::cout << "Computing time: " << std::setprecision(5) << get_solve_time_in_milliseconds() << " [ms]" << std::endl;
    std::cout << "Total Spin <S^2>: " << std::setprecision(17) << get_total_spin() << std::endl;
}




void ROHF::export_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        THROW_EXCEPTION("Failed to open the file: " + filename);
    }

    // write the header
    ofs << "[Molden Format]" << std::endl;
    ofs << "generated by GANSU" << std::endl;
    ofs << "[Atoms] (AU)" << std::endl;
    for(size_t i=0; i<atoms.size(); i++){
        ofs << atomic_number_to_element_name(atoms[i].atomic_number) << " " 
            << i+1 << " "
            << atoms[i].atomic_number << " "
            << atoms[i].coordinate.x << " "
            << atoms[i].coordinate.y << " "
            << atoms[i].coordinate.z << std::endl;
    }
    ofs << "[GTO]" << std::endl;
    primitive_shells.toHost();
    std::vector<int> num_primitives(num_basis, 0);
    std::vector<int> shell_types(num_basis, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }

    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range()[i];
        for(size_t j=basis_range.start_index; j<basis_range.end_index; j++){
            if(num_primitives[j] == 0){
                continue;
            }
            ofs << " " << shell_type_to_shell_name(shell_types[j]) << " " << num_primitives[j] << " " << "1.00" << std::endl;
            for(size_t k=0; k<primitive_shells.size(); k++){
                if(primitive_shells[k].basis_index == j){
                    ofs << "\t" << primitive_shells[k].exponent << " " << primitive_shells[k].coefficient << std::endl;
                }
            }
        }
        ofs << std::endl; // empty line
    }

    ofs << std::endl; // empty line

    // write the orbital energies
    orbital_energies.toHost();
    coefficient_matrix.toHost();
    ofs << "[MO]" << std::endl;
    for(size_t i=0; i<num_basis; i++){
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= " << (i < num_beta_spins ? 2.0 : (i < num_alpha_spins ? 1.0 : 0.0)) << std::endl;
        for(size_t j=0; j<num_basis; j++){
            ofs << " " << j+1 << " " << std::setprecision(17) << coefficient_matrix(j, i) << std::endl;
        }
    }

    ofs.close();

}

std::vector<real_t> ROHF::analyze_mulliken_population() const {
    std::vector<real_t> mulliken_population_basis(num_basis);

    // calculate the Mulliken population for each basis function
    gpu::computeMullikenPopulation_RHF(
        density_matrix.device_ptr(),
        overlap_matrix.device_ptr(),
        mulliken_population_basis.data(),
        num_basis
    );

    const auto& atoms = get_atoms();
    const int num_atoms = atoms.size();

    std::vector<real_t> mulliken_population_atoms(num_atoms);
    for(int i=0; i<num_atoms; i++){
        const int basis_start = get_atom_to_basis_range()[i].start_index;
        const int basis_end = get_atom_to_basis_range()[i].end_index;
        mulliken_population_atoms[i] = atoms[i].atomic_number; // initialize with the atomic number (positive charge)
        for(int j=basis_start; j<basis_end; j++){
            mulliken_population_atoms[i] -= mulliken_population_basis[j];
        }
    }

    return mulliken_population_atoms;
}

std::vector<std::vector<real_t>> ROHF::compute_mayer_bond_order() const{
    std::vector<std::vector<real_t>> mayer_bond_order_matrix(atoms.size(), std::vector<real_t>(atoms.size(), 0.0));

    std::vector<real_t> temp_matrix(num_basis * num_basis, 0.0); // temporary matrix to store DS (product of density and overlap matrices)

    // calculate the product of density and overlap matrices
    gpu::computeDensityOverlapMatrix(
        density_matrix.device_ptr(),
        overlap_matrix.device_ptr(),
        temp_matrix.data(),
        num_basis
    );

    // sum up the contributions from basis functions to atoms
    for(size_t i=0; i<atoms.size(); i++){
        const int basis_i_start = get_atom_to_basis_range()[i].start_index;
        const int basis_i_end = get_atom_to_basis_range()[i].end_index;
        for(size_t j=0; j<atoms.size(); j++){
            const int basis_j_start = get_atom_to_basis_range()[j].start_index;
            const int basis_j_end = get_atom_to_basis_range()[j].end_index;
            real_t bond_order_ij = 0.0;
            for(int bi=basis_i_start; bi<basis_i_end; bi++){
                for(int bj=basis_j_start; bj<basis_j_end; bj++){
                    bond_order_ij += temp_matrix[bi * num_basis + bj] * temp_matrix[bj * num_basis + bi];
                }
            }
            mayer_bond_order_matrix[i][j] = bond_order_ij;
        }
    }

    if(verbose){
        std::cout << "Mayer bond order matrix:" << std::endl;
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << mayer_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


    return mayer_bond_order_matrix;
}



std::vector<std::vector<real_t>> ROHF::compute_wiberg_bond_order() {
    std::vector<std::vector<real_t>> wiberg_bond_order_matrix(atoms.size(), std::vector<real_t>(atoms.size(), 0.0));

    std::vector<real_t> temp_matrix(num_basis * num_basis, 0.0); // temporary matrix to store S^{1/2} * D * S^{1/2}

    // Compute S^{1/2}
    gpu::computeSqrtOverlapDensitySqrtOverlapMatrix(
        overlap_matrix.device_ptr(),
        density_matrix.device_ptr(),
        temp_matrix.data(),
        num_basis
    );

    // sum up the contributions from basis functions to atoms
    for(size_t i=0; i<atoms.size(); i++){
        const int basis_i_start = get_atom_to_basis_range()[i].start_index;
        const int basis_i_end = get_atom_to_basis_range()[i].end_index;
        for(size_t j=0; j<atoms.size(); j++){
            const int basis_j_start = get_atom_to_basis_range()[j].start_index;
            const int basis_j_end = get_atom_to_basis_range()[j].end_index;
            real_t bond_order_ij = 0.0;
            for(int bi=basis_i_start; bi<basis_i_end; bi++){
                for(int bj=basis_j_start; bj<basis_j_end; bj++){
                    real_t d_ij = temp_matrix[bi * num_basis + bj];
                    bond_order_ij += d_ij * d_ij;
                }
            }
            wiberg_bond_order_matrix[i][j] = bond_order_ij;
        }
    }

    if(verbose){
        std::cout << "Wiberg bond order matrix:" << std::endl;
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << wiberg_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


    return wiberg_bond_order_matrix;
}

} // namespace gansu