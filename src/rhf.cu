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
 * @brief  This function is used to calculate the rhf energy
 */

#include "rhf.hpp"


#include <limits> // numeric_limits<double>::max();
#include <iomanip> // std::setprecision

#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{



/**
 * @brief Constructor of the RHF class
 * @param molecular Molecular
 * @param verbose Verbose mode
 * @details This function constructs the RHF class.
 * @details The molecular is given as an argument.
 */
RHF::RHF(const Molecular& molecular, const ParameterManager& parameters) : 
    HF(molecular, parameters),
    coefficient_matrix(num_basis, num_basis),
    density_matrix(num_basis, num_basis),
    orbital_energies(num_basis),
    initail_guess_method_(parameters.get<std::string>("initial_guess")),
    gbsfilename_(parameters.get<std::string>("gbsfilename")),
    fock_matrix(num_basis, num_basis)
{
    // check if the number of electrons is even (condition for restricted Hartree Fock method)
    if(molecular.get_num_alpha_spins() != molecular.get_num_beta_spins()){
        THROW_EXCEPTION("In restricted Hartree Fock method, the number of alpha and beta electrons should be equal.");
    }
    if(molecular.get_num_alpha_spins() > num_basis){
        THROW_EXCEPTION("The number of alpha or beta electrons should be less than or equal to the number of basis functions.");
    }

    // Set an algorithm to update the Fock matrix (default: Convergence_RHF_DIIS)
    const std::string convergence_method = parameters.get<std::string>("convergence_method");
    if(convergence_method == "damping"){ // damping algorithm
        const double damping_factor = parameters.get<double>("damping_factor"); // damping factor, default: 0.9
        set_convergence_method(std::make_unique<Convergence_RHF_Damping>(*this, damping_factor));
    }else if(convergence_method == "optimaldamping"){
        set_convergence_method(std::make_unique<Convergence_RHF_Damping>(*this));
    }else if(convergence_method == "diis"){ // DIIS algorithm
        const size_t DIIS_size = parameters.get<int>("diis_size"); // DIIS size (number of previous Fock matrices), default: 8
        const bool is_include_transform = parameters.get<bool>("diis_include_transform"); // include the transformation matrix in DIIS, default: false

        set_convergence_method(std::make_unique<Convergence_RHF_DIIS>(*this, DIIS_size, is_include_transform));
    }else{
        THROW_EXCEPTION("Invalid convergence algorithm name: " + convergence_method);
    }

    // Set an algorithm for ERI calculation (default: ERI_Stored_RHF)
    const std::string eri_method = parameters.get<std::string>("eri_method");
    if(eri_method == "stored"){ // ERI matrices are stored in the device memory
        set_eri_method(std::make_unique<ERI_Stored_RHF>(*this));
    }else if(eri_method == "ri"){ // Resolution of Identity (RI) method
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename"); // auxiliary basis set file name
        Molecular auxiliary_molecular(molecular.get_atoms(), auxiliary_gbsfilename); // auxiliary molecular object
        set_eri_method(std::make_unique<ERI_RI_RHF>(*this, auxiliary_molecular));
    }else if(eri_method == "direct"){
        set_eri_method(std::make_unique<ERI_Direct_RHF>(*this));
    }else if(eri_method == "hash"){
        set_eri_method(std::make_unique<ERI_Hash_RHF>(*this));
    }else{
        THROW_EXCEPTION("Invalid ERI method name: " + eri_method);
    }

    // Check if the selected ERI method supports post-HF methods
    if(!eri_method_->supports_post_hf_method(get_post_hf_method())){
        THROW_EXCEPTION("The selected ERI method does not support the selected post-HF method.");
    }
}

/**
 * @brief Set a convergence method to update the Fock matrix
 * @param convergence_method Convergence_RHF
 * @details This function sets a convergence method to update the Fock matrix.
 */
void RHF::set_convergence_method(std::unique_ptr<Convergence_RHF> convergence_method) {
    this->convergence_method_ = std::move(convergence_method);
}

/**
 * @brief Set an ERI method to calculate the two-electron integrals
 * @param eri_method ERI
 * @details This function sets an ERI method to calculate the two-electron integrals.
 */
void RHF::set_eri_method(std::unique_ptr<ERI> eri_method) {
    this->eri_method_ = std::move(eri_method);
}


void RHF::precompute_eri_matrix(){
    PROFILE_FUNCTION();

    eri_method_->precomputation();
}


void RHF::post_process_after_scf() {
    PROFILE_FUNCTION();

    PostHFMethod post_hf_method = get_post_hf_method();
    if(post_hf_method == PostHFMethod::None){
        post_hf_energy_ = 0.0;
        return; // do nothing
    }else if(post_hf_method == PostHFMethod::MP2){
        post_hf_energy_ = eri_method_->compute_mp2_energy();
    }else if(post_hf_method == PostHFMethod::MP3){
        post_hf_energy_ = eri_method_->compute_mp3_energy();
    }else if(post_hf_method == PostHFMethod::MP4){
        post_hf_energy_ = eri_method_->compute_mp4_energy();
    }else if(post_hf_method == PostHFMethod::CCSD){
        post_hf_energy_ = eri_method_->compute_ccsd_energy();
    }else if(post_hf_method == PostHFMethod::CCSD_T){
        post_hf_energy_ = eri_method_->compute_ccsd_t_energy();
    }else{
        THROW_EXCEPTION("Invalid post-HF method.");
    }
}


/**
 * @brief Function to guess the initial Fock matrix
 * @param density_matrix_a Density matrix (alpha-spin)
 * @param density_matrix_b Density matrix (beta-spin)
 * @param force_density Initialized by the precomputed density matrix
 * @details This function calculates the initial Fock matrix using the core Hamiltonian matrix.
 */
void RHF::guess_initial_fock_matrix(const real_t* density_matrix_a, const real_t* density_matrix_b, bool force_density) {
    PROFILE_FUNCTION();

    std::unique_ptr<InitialGuess_RHF> initial_guess; // the life time is only here since initial guess is performed only once

    if(force_density == true || initail_guess_method_ == "density"){ // Inititialized by precomputed density matrices
        if(density_matrix_a == nullptr || density_matrix_b == nullptr){
            std::cerr << "The density matrix is not provided even though ``density'' is set to ``initail_guess_method'' or forced_density = true. The core Hamiltonian matrix is used instead." << std::endl;
            initial_guess = std::make_unique<InitialGuess_RHF_Core>(*this);
        }else{
            initial_guess = std::make_unique<InitialGuess_RHF_Density>(*this, density_matrix_a, density_matrix_b);
        }
    }else if(initail_guess_method_ == "core"){ // core Hamiltonian matrix
        initial_guess = std::make_unique<InitialGuess_RHF_Core>(*this);
    }else if(initail_guess_method_ == "gwh"){ // Generalized Wolfsberg-Helmholz (GWH) method
        initial_guess = std::make_unique<InitialGuess_RHF_GWH>(*this);
    }else if(initail_guess_method_ == "sad"){ // Superposition of Atomic Densities (SAD) method
        if(gbsfilename_.empty()){
            THROW_EXCEPTION("The basis set file is not specified for SAD initial guess method. Please specify the basis set file name by -gbsfilename option.");
        }
        initial_guess = std::make_unique<InitialGuess_RHF_SAD>(*this);
    }else{
        throw std::runtime_error("Invalid initial guess method: " + initail_guess_method_);
    }

    // Execute the initial guess method
    initial_guess->guess();

}

/**
 * @brief Function to calculate the coefficient matrix
 * @details This function calculates the coefficient matrix using the eigenvectors of the Fock matrix.
 */
void RHF::compute_coefficient_matrix_impl() {
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
void RHF::compute_density_matrix() {
    PROFILE_FUNCTION();

    gpu::computeDensityMatrix_RHF(
        coefficient_matrix.device_ptr(),
        density_matrix.device_ptr(),
        num_electrons,
        num_basis
    );
/*
    { // nan check
        density_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                if(std::isnan(density_matrix(i, j))){
                    THROW_EXCEPTION("Density matrix contains NaN value at (" + std::to_string(i) + ", " + std::to_string(j) + ").");
                }
            }
            std::cout << std::endl;
        }
    }
*/
    if (verbose) {
        std::cout << "Density matrix:" << std::endl;
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
 void RHF::compute_fock_matrix(){
    PROFILE_FUNCTION();

    eri_method_->compute_fock_matrix();
}


/**
 * @brief Function to calculate the energy
 * @details This function calculates the energy using the core Hamiltonian matrix, the density matrix, and the Fock matrix.
 */
void RHF::compute_energy(){
    PROFILE_FUNCTION();

    energy_ = gpu::computeEnergy_RHF(
        density_matrix.device_ptr(),
        core_hamiltonian_matrix.device_ptr(),
        fock_matrix.device_ptr(),
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
void RHF::update_fock_matrix(){
    PROFILE_FUNCTION();

    convergence_method_->get_new_fock_matrix();
}

void RHF::export_density_matrix(real_t* density_matrix_a, real_t* density_martix_b, const int num_basis) {
    if(num_basis != this->num_basis){
        throw std::runtime_error("The number of basis functions is different.");
    }
    if(density_matrix_a == nullptr || density_martix_b == nullptr){
        throw std::runtime_error("The density matrix is not provided.");
    }
    // copy the density matrix to the host memory
    density_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        for(size_t j=0; j<num_basis; j++){
            density_matrix_a[i*num_basis + j] = density_matrix(i, j) / 2.0;
            density_martix_b[i*num_basis + j] = density_matrix(i, j) / 2.0;
        }
    }
}





/**
 * @brief Print the results of the SCF procedure
 * @details This function prints the results of the SCF procedure.
 */
void RHF::report() {

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
    std::cout << "Method: Restricted Hartree-Fock (RHF)" << std::endl;
    std::cout << "Schwarz screening threshold: " << schwarz_screening_threshold << std::endl;
    std::cout << "Initial guess method: " << initail_guess_method_ << std::endl;
    std::cout << "Convergence algorithm: " << convergence_method_->get_algorithm_name() << std::endl;
    std::cout << "Number of iterations: " << iter_ << std::endl;
    std::cout << "Convergence criterion: " << convergence_energy_threshold << std::endl;
    std::cout << "Energy difference: " << energy_difference_ << std::endl;
    std::cout << "Energy (without nuclear repulsion): " << std::setprecision(17) << get_energy() << " [hartree]" << std::endl;
    std::cout << "Total Energy: " << std::setprecision(17) << get_total_energy() << " [hartree]" << std::endl;
    std::cout << "Computing time: " << std::setprecision(5) << get_solve_time_in_milliseconds() << " [ms]" << std::endl;

    if(get_post_hf_method() != PostHFMethod::None){
        std::cout << std::endl;
        std::cout << "[Calculation Summary (Post-HF)]" << std::endl;
        std::cout << "Post-HF method: ";
        if(get_post_hf_method() == PostHFMethod::MP2){
            std::cout << "MP2" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::MP3){
            std::cout << "MP3" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CCSD){
            std::cout << "CCSD" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CCSD_T){
            std::cout << "CCSD(T)" << std::endl;
        }
        std::cout << "Post-HF energy correction: " << std::setprecision(17) << get_post_hf_energy() << " [hartree]" << std::endl;
        std::cout << "Total Energy (including post-HF correction): " << std::setprecision(17) << get_total_energy() + get_post_hf_energy() << " [hartree]" << std::endl;
    }
}

/*
void RHF::export_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open the file: " + filename);
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
            if(num_primitives[j] == 0){ // skip non-representive basis functions (e.g. py,pz, etc.)
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
    ofs << "[MO]" << std::endl;
    orbital_energies.toHost();
    coefficient_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= " << (i < num_electrons/2 ? 2.0 : 0.0) << std::endl;
        for(size_t j=0; j<num_basis; j++){
            ofs << " " << j+1 << " " << std::setprecision(17) << coefficient_matrix(j, i) << std::endl;
        }
    }

    ofs.close();

}
*/

void RHF::export_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open the file: " + filename);
    }
    // write the header
    ofs << "[Molden Format]" << std::endl;
    ofs << "[Title]" << std::endl;
    ofs << "generated by GANSU" << std::endl;
    ofs << "[Atoms] (Angs)" << std::endl;
    for(size_t i=0; i<atoms.size(); i++){
        ofs << atomic_number_to_element_name(atoms[i].atomic_number) << " " 
            << i+1 << " "
            << atoms[i].atomic_number << " "
            << bohr_to_angstrom(atoms[i].coordinate.x) << " "
            << bohr_to_angstrom(atoms[i].coordinate.y) << " "
            << bohr_to_angstrom(atoms[i].coordinate.z) << std::endl;
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
            if(num_primitives[j] == 0){ // skip non-representive basis functions (e.g. py,pz, etc.)
                continue;
            }  
            ofs << " " << shell_type_to_shell_name(shell_types[j]) << " " << num_primitives[j] << " " << "1.00" << std::endl;
            for(size_t k=0; k<primitive_shells.size(); k++){
                if(primitive_shells[k].basis_index == j){
                    ofs << "    " << primitive_shells[k].exponent << " " << primitive_shells[k].coefficient << std::endl;
                }
            }
        }
        ofs << std::endl; // empty line
    }

//    ofs << std::endl; // empty line
    
    // write the orbital energies
    ofs << "[MO]" << std::endl;
    orbital_energies.toHost();
    coefficient_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= " << (i < num_electrons/2 ? 2.0 : 0.0) << std::endl;
        for(size_t j=0; j<num_basis; j++){
            ofs << " " << j+1 << " " << std::setprecision(17) << coefficient_matrix(j, i) << std::endl;
        }
    }

    ofs.close();

}


std::vector<real_t> RHF::analyze_mulliken_population() const {
    std::vector<real_t> mulliken_population_basis(num_basis);

    // calculate the Mulliken population for each basis function
    gpu::computeMullikenPopulation_RHF(
        density_matrix.device_ptr(),
        overlap_matrix.device_ptr(),
        mulliken_population_basis.data(),
        num_basis
    );

    if (verbose) {
        std::cout << "Mulliken population for basis functions:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            std::cout << "Basis " << i << ": " << std::setprecision(6) << mulliken_population_basis[i] << std::endl;
        }
        std::cout << std::endl;
    }

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

std::vector<std::vector<real_t>> RHF::compute_mayer_bond_order() const{
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


std::vector<std::vector<real_t>> RHF::compute_wiberg_bond_order() {
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