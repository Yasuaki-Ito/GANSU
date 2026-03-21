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
 * @file hf.cu
 * @brief Implementation of the HF class.
 * @details This file contains the implementation of the HF class.
 */
#include <cmath> // sqrt
#include <cassert>
#include <algorithm> // std::sort
#include <cmath> // sqrt
#include <iomanip> // std::setprecision

#include "hf.hpp"
#include "device_host_memory.hpp"
#include "boys/boys_30720.h"
#include "gpu_manager.hpp"

#include "profiler.hpp"
#include "geometry_optimizer.hpp"

namespace gansu{



/**
 * @brief Constructor of the HF class
 * @param molecular Molecular
 * @param verbose Verbose mode, default is false
 * @param max_iter Maximum number of iterations, default is 100
 * @param convergence_energy_threshold Convergence criterion, default is 1.0e-9
 * @details This function constructs the HF class.
 * @details The molecular is given as an argument.
 */
HF::HF(const Molecular& molecular, const ParameterManager& parameters) : 
    num_basis(molecular.get_num_basis()), 
    num_electrons(molecular.get_num_electrons()),
    num_alpha_spins(molecular.get_num_alpha_spins()),
    num_beta_spins(molecular.get_num_beta_spins()),
    atoms(molecular.get_atoms()), // construct the list directly from std::vector
    primitive_shells(molecular.get_primitive_shells()), // construct the list directly from std::vector
    shell_type_infos(molecular.get_shell_type_infos()), // construct the list directly from std::vector
    shell_pair_type_infos(shell_type_infos.size()*(shell_type_infos.size()+1)/2),
    atom_to_basis_range(molecular.get_atom_to_basis_range()), // construct the list directly from std::vector
    boys_grid(30720,true), // 30720 is the number of grid points for, true means that the host memory is allocated in advance
    cgto_normalization_factors(molecular.get_cgto_normalization_factors()), // construct the list directly from std::vector
    overlap_matrix(num_basis, num_basis), // host memory is not allocated in advance
    core_hamiltonian_matrix(num_basis, num_basis), // host memory is not allocated in advance
    transform_matrix(num_basis, num_basis), // host memory is not allocated in advance
    verbose(parameters.get<int>("verbose")),
    max_iter(parameters.get<int>("maxiter")),
    int1e_method(parameters.get<std::string>("int1e_method")),
    initial_guess_method_(parameters.get<std::string>("initial_guess")),
    convergence_energy_threshold(parameters.get<double>("convergence_energy_threshold")),
    schwarz_screening_threshold(parameters.get<double>("schwarz_screening_threshold")),
    run_type_(parameters.get<std::string>("run_type")),
    optimizer_(parameters.get<std::string>("optimizer")),
    post_hf_energy_(0.0),
    is_mulliken_analysis_(parameters.get<bool>("mulliken")),
    is_mayer_bond_order_analysis_(parameters.get<bool>("mayer")),
    is_wiberg_bond_order_analysis_(parameters.get<bool>("wiberg")),
    is_export_molden_(parameters.get<bool>("export_molden"))
{
    // Validate run_type
    if(run_type_ != "energy" && run_type_ != "gradient" && run_type_ != "optimize"){
        throw std::runtime_error("Invalid run_type: '" + run_type_ + "'. Must be 'energy', 'gradient', or 'optimize'.");
    }

    // Validate optimizer
    if(!GeometryOptimizer::is_valid_optimizer(optimizer_)){
        throw std::runtime_error("Invalid optimizer: '" + optimizer_ + "'. Valid options: bfgs, dfp, sr1, gdiis, cg-fr, cg-pr, cg-hs, cg-dy, sd");
    }

    // Set the post-HF method
    std::string post_hf_method_str = parameters.get<std::string>("post_hf_method");
    if(post_hf_method_str == "none"){
        std::cout << "Message: Post-HF method is not selected." << std::endl;
        post_hf_method_ = PostHFMethod::None;
    }else if(post_hf_method_str == "fci"){
        std::cout << "Message: Post-HF method is FCI." << std::endl;
        post_hf_method_ = PostHFMethod::FCI;
    }else if(post_hf_method_str == "mp2"){
        std::cout << "Message: Post-HF method is MP2." << std::endl;
        post_hf_method_ = PostHFMethod::MP2;
    }else if(post_hf_method_str == "mp3"){
        std::cout << "Message: Post-HF method is MP3." << std::endl;
        post_hf_method_ = PostHFMethod::MP3;
    }else if(post_hf_method_str == "mp4"){
        std::cout << "Message: Post-HF method is MP4." << std::endl;
        post_hf_method_ = PostHFMethod::MP4;
    }else if(post_hf_method_str == "cc2"){
        std::cout << "Message: Post-HF method is CC2." << std::endl;
        post_hf_method_ = PostHFMethod::CC2;
    }else if(post_hf_method_str == "ccsd"){
        std::cout << "Message: Post-HF method is CCSD." << std::endl;
        post_hf_method_ = PostHFMethod::CCSD;
    }else if(post_hf_method_str == "ccsd_t"){
        std::cout << "Message: Post-HF method is CCSD(T)." << std::endl;
        post_hf_method_ = PostHFMethod::CCSD_T;
    }else if(post_hf_method_str == "cis"){
        std::cout << "Message: Post-HF method is CIS." << std::endl;
        post_hf_method_ = PostHFMethod::CIS;
    }else if(post_hf_method_str == "adc2"){
        std::cout << "Message: Post-HF method is ADC(2)." << std::endl;
        post_hf_method_ = PostHFMethod::ADC2;
    }else if(post_hf_method_str == "eom_mp2"){
        std::cout << "Message: Post-HF method is EOM-MP2." << std::endl;
        post_hf_method_ = PostHFMethod::EOM_MP2;
    }else if(post_hf_method_str == "eom_cc2"){
        std::cout << "Message: Post-HF method is EOM-CC2." << std::endl;
        post_hf_method_ = PostHFMethod::EOM_CC2;
    }else if(post_hf_method_str == "eom_ccsd"){
        std::cout << "Message: Post-HF method is EOM-CCSD." << std::endl;
        post_hf_method_ = PostHFMethod::EOM_CCSD;
    }else{
        throw std::runtime_error("Error: Unknown post-HF method: " + post_hf_method_str);
    }

    // Set the number of excited states
    n_excited_states_ = parameters.get<int>("n_excited_states");

    // Set the ADC(2) solver mode
    adc2_solver_ = toLowerCase(parameters.get<std::string>("adc2_solver"));

    // Set the EOM-MP2 solver mode
    eom_mp2_solver_ = toLowerCase(parameters.get<std::string>("eom_mp2_solver"));


    // print all the values of boys function for the test (temporary implementation)
    if(verbose){
        std::cout << "Message: grid values for the Boys function is load from the header file." << std::endl;
    }
    for(size_t i=0; i<30720; i++){
        //std::cout << i << ": " << h_boys_grid[i] << std::endl;
        boys_grid[i] = h_boys_grid[i];
    }

    // for Schwarz screening in Stored ERI and Direct SCF
    int sum = 0;
    for (const auto& x : shell_type_infos) {
        sum += x.count;
    }
    num_primitive_shells = sum;
    num_primitive_shell_pairs = gpu::makeShellPairTypeInfo(shell_type_infos, shell_pair_type_infos);


    if(verbose){
        molecular.dump();
    }

    // initialize the overlap matrix
    cudaMemset(overlap_matrix.device_ptr(), 0, sizeof(real_t)*num_basis*num_basis);

    // initialize the core Hamiltonian matrix
    cudaMemset(core_hamiltonian_matrix.device_ptr(), 0, sizeof(real_t)*num_basis*num_basis);

    // copy the data to the device memory for preparing the computation
    boys_grid.toDevice(); // copy the grid values to the device memory
    atoms.toDevice(); // copy the list of atoms to the device memory
    primitive_shells.toDevice(); // copy the list of primitive shells to the device memory
    cgto_normalization_factors.toDevice(); // copy the list of cgto normalization factors to the device memory

}

void HF::compute_nuclear_repulsion_energy() {
    PROFILE_FUNCTION();

    real_t nuclear_repulsion_energy = 0.0;
    for(size_t i=0; i<atoms.size()-1; i++){
        for(size_t j=i+1; j<atoms.size(); j++){
            const auto& atom1 = atoms[i];
            const auto& atom2 = atoms[j];
            real_t dx = atom1.coordinate.x - atom2.coordinate.x;
            real_t dy = atom1.coordinate.y - atom2.coordinate.y;
            real_t dz = atom1.coordinate.z - atom2.coordinate.z;
            real_t r = std::sqrt(dx*dx + dy*dy + dz*dz);
            real_t z1 = atom1.atomic_number;
            real_t z2 = atom2.atomic_number;
            assert(r != 0.0);
            nuclear_repulsion_energy += z1 * z2 / r;
        }
    }
    nuclear_repulsion_energy_ = nuclear_repulsion_energy;
}


void HF::compute_core_hamiltonian_matrix() {
    PROFILE_FUNCTION();

    // compute the core Hamiltonian matrix
    gpu::computeCoreHamiltonianMatrix(shell_type_infos, atoms.device_ptr(), primitive_shells.device_ptr(), boys_grid.device_ptr(), cgto_normalization_factors.device_ptr(), overlap_matrix.device_ptr(), core_hamiltonian_matrix.device_ptr(),atoms.size(), num_basis, int1e_method, verbose);

    // print the overlap and core Hamiltonian matrix
    if(verbose){
        // copy the core Hamiltonian matrix to the host memory
        overlap_matrix.toHost();
        core_hamiltonian_matrix.toHost();

        std::cout << "=== Overlap Matrix ===" << std::endl;
        std::cout << "[\n";
        for (size_t i = 0; i < num_basis; i++) {
            std::cout << "  [";
            for (size_t j = 0; j < num_basis; j++) {
                std::cout << std::right << std::setfill(' ')
                        << std::setw(10) << std::fixed << std::setprecision(6) << overlap_matrix(i, j);
                if (j != num_basis - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i != num_basis - 1) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "]\n\n";

        std::cout << "=== Core Hamiltonian Matrix ===" << std::endl;
        std::cout << "[\n";
        for (size_t i = 0; i < num_basis; i++) {
            std::cout << "  [";
            for (size_t j = 0; j < num_basis; j++) {
                std::cout << std::right << std::setfill(' ')
                        << std::setw(10) << std::fixed << std::setprecision(6) << core_hamiltonian_matrix(i, j);
                if (j != num_basis - 1) std::cout << ", ";
            }
            std::cout << "]";
            if (i != num_basis - 1) std::cout << ",";
            std::cout << "\n";
        }
        std::cout << "]\n\n";

    }

}



void HF::compute_transform_matrix(){
    PROFILE_FUNCTION();

    DeviceHostMemory<real_t> eigenvalue(num_basis);
    DeviceHostMatrix<real_t> eigenvector(num_basis, num_basis);

    // compute the eigenvalues and eigenvectors of the overlap matrix
    gpu::eigenDecomposition(overlap_matrix.device_ptr(), eigenvalue.device_ptr(), eigenvector.device_ptr(), num_basis);

    if(verbose){
        // copy the transform matrix to the host memory
        eigenvalue.toHost();
        std::cout << "Eigenvalues:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            std::cout << eigenvalue[i] << " ";
        }
        std::cout << std::endl;

        eigenvector.toHost();
        std::cout << "Eigenvectors:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << eigenvector(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // compute the transformation matrix
    gpu::invertSqrtElements(eigenvalue.device_ptr(), num_basis);

    if(verbose){
        // copy the transform matrix to the host memory
        eigenvalue.toHost();
        std::cout << "Inverse square root of the eigenvalues:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            std::cout << eigenvalue[i] << " ";
        }
        std::cout << std::endl;
    }

    DeviceHostMatrix<real_t> eigenvalueMatrix(num_basis, num_basis);

    gpu::makeDiagonalMatrix(eigenvalue.device_ptr(), eigenvalueMatrix.device_ptr(), num_basis);

    if(verbose){
        // copy the transform matrix to the host memory
        eigenvalueMatrix.toHost();
        std::cout << "Make diagnal:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << eigenvalueMatrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // compute eigenvectors * inverse square root of the eigenvalues
    gpu::matrixMatrixProduct(eigenvector.device_ptr(), eigenvalueMatrix.device_ptr(), transform_matrix.device_ptr(), num_basis);

    if(verbose){
        // copy the transform matrix to the host memory
        transform_matrix.toHost();
        std::cout << "Transform matrix:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << transform_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}


real_t HF::single_point_energy(const real_t* density_matrix_alpha, const real_t* density_matrix_beta, bool force_density){
//    PROFILE_FUNCTION();
    // Reset convergence method state (DIIS history, damping state) for a fresh SCF cycle
    reset_convergence();

    // Start Profiling
    GlobalProfiler::initialize(); // timer starts


    Timer timer;

    // compute the nuclear repulsion energy
    compute_nuclear_repulsion_energy();
    if(verbose){
        std::cout << "Nuclear repulsion energy: " << nuclear_repulsion_energy_ << std::endl;
    }



    // compute the core Hamiltonian matrix
    compute_core_hamiltonian_matrix();

    // precompute the electron repulsion integrals
    precompute_eri_matrix();

    // compute the transformation matrix
    compute_transform_matrix();
    

    guess_initial_fock_matrix(density_matrix_alpha, density_matrix_beta, force_density); // guess the initial Fock matrix

    real_t prev_energy = std::numeric_limits<double>::max(); // set the previous energy to a large value
    real_t energy = 0.0; // initialize the energy

    // SCF iterations
    iter_ = 0;
    while(iter_ < max_iter){
        std::string str = "SCF Iteration " + std::to_string(iter_);
        PROFILE_ELAPSED_TIME(str);

        compute_coefficient_matrix(); // compute the coefficient matrix from the Fock matrix
        compute_density_matrix(); // compute the density matrix from the coefficient matrix
        compute_fock_matrix(); // compute the Fock matrix from the density matrix
        compute_energy(); // compute the energy from the Fock matrix

        energy = get_energy(); // get the energy

        // check convergence
        energy_difference_ = std::abs(energy - prev_energy);

        if(run_type_ != "optimize"){
            std::cout << "---- Iteration: " << iter_ << " ----  ";
            std::cout << "Energy: " << std::setprecision(17) << energy << " ";
            std::cout << "Total energy: " << std::setprecision(17) << get_total_energy() << " ";
            if(iter_ > 0){
                std::cout << "Energy difference: " << std::setprecision(10) << energy_difference_ << " ";
            }
            std::cout << std::endl;
        }

        if(energy_difference_ < convergence_energy_threshold){
            break;
        }
        iter_++;

        // store the previous energy
        prev_energy = energy;

        // Update the Fock matrix
        update_fock_matrix();
    }

    post_process_after_scf(); // post process after SCF convergence (Post-HF calculations, etc.)

    solve_time_in_milliseconds_ = timer.elapsed_milliseconds();

    if(run_type_ != "optimize"){
        GlobalProfiler::report(); // Print the profiling results
    }
    
    return get_total_energy();
}

real_t HF::solve(const real_t* density_matrix_alpha, const real_t* density_matrix_beta, bool force_density){
    if(run_type_ == "optimize"){
        std::cout << std::endl;
        std::cout << "============================================================" << std::endl;
        std::string opt_upper = optimizer_;
        std::transform(opt_upper.begin(), opt_upper.end(), opt_upper.begin(), ::toupper);
        std::cout << "              Geometry Optimization (" << opt_upper << ")                  " << std::endl;
        std::cout << "============================================================" << std::endl;

        GlobalProfiler::setSilent(true); // Suppress profiler output during optimization

        const int max_iter = 200;
        const double grad_threshold = 3.0e-4;    // max gradient component (Hartree/Bohr)
        const double rms_grad_threshold = 2.0e-4; // RMS gradient (Hartree/Bohr)
        const double energy_threshold = 1.0e-6;   // energy change (Hartree)
        const double disp_threshold = 3.0e-4;     // max displacement (Bohr)
        const double step_max = 0.3;               // trust radius (Bohr)

        // Initial SCF
        real_t energy = single_point_energy(density_matrix_alpha, density_matrix_beta, force_density);

        atoms.toHost();
        int num_atoms_val = static_cast<int>(atoms.size());
        int ndim = 3 * num_atoms_val; // dimension of coordinate/gradient vector

        // Create optimizer
        auto optimizer = GeometryOptimizer::create(optimizer_, ndim);

        // Initial gradient
        std::vector<double> grad = compute_Energy_Gradient();
        if(grad.empty()){
            std::cout << "Gradient not available for this method. Aborting optimization." << std::endl;
            return energy;
        }

        // Extract current coordinates as flat vector
        auto atoms_to_coords = [&](const std::vector<Atom>& atom_list) -> std::vector<double> {
            std::vector<double> coords(ndim);
            for(int i = 0; i < num_atoms_val; i++){
                coords[3*i + 0] = atom_list[i].coordinate.x;
                coords[3*i + 1] = atom_list[i].coordinate.y;
                coords[3*i + 2] = atom_list[i].coordinate.z;
            }
            return coords;
        };

        auto coords_to_atoms = [&](const std::vector<double>& coords) -> std::vector<Atom> {
            atoms.toHost();
            std::vector<Atom> atom_list(num_atoms_val);
            for(int i = 0; i < num_atoms_val; i++){
                atom_list[i] = atoms[i];
                atom_list[i].coordinate.x = coords[3*i + 0];
                atom_list[i].coordinate.y = coords[3*i + 1];
                atom_list[i].coordinate.z = coords[3*i + 2];
            }
            return atom_list;
        };

        atoms.toHost();
        std::vector<Atom> current_atoms(num_atoms_val);
        for(int i = 0; i < num_atoms_val; i++) current_atoms[i] = atoms[i];
        std::vector<double> coords = atoms_to_coords(current_atoms);

        // --- Project out translational and rotational components from a vector ---
        auto project_out_tr = [&](std::vector<double>& v, const std::vector<double>& r){
            int N = num_atoms_val;
            std::vector<std::vector<double>> basis(6, std::vector<double>(ndim, 0.0));

            for(int i = 0; i < N; i++){
                basis[0][3*i + 0] = 1.0;
                basis[1][3*i + 1] = 1.0;
                basis[2][3*i + 2] = 1.0;
            }

            double cx = 0, cy = 0, cz = 0;
            for(int i = 0; i < N; i++){
                cx += r[3*i+0]; cy += r[3*i+1]; cz += r[3*i+2];
            }
            cx /= N; cy /= N; cz /= N;

            for(int i = 0; i < N; i++){
                double x = r[3*i+0] - cx;
                double y = r[3*i+1] - cy;
                double z = r[3*i+2] - cz;
                basis[3][3*i + 1] = -z;
                basis[3][3*i + 2] =  y;
                basis[4][3*i + 0] =  z;
                basis[4][3*i + 2] = -x;
                basis[5][3*i + 0] = -y;
                basis[5][3*i + 1] =  x;
            }

            std::vector<std::vector<double>> ortho_basis;
            for(int k = 0; k < 6; k++){
                auto& bk = basis[k];
                for(const auto& ob : ortho_basis){
                    double dot = 0.0;
                    for(int i = 0; i < ndim; i++) dot += bk[i] * ob[i];
                    for(int i = 0; i < ndim; i++) bk[i] -= dot * ob[i];
                }
                double norm = 0.0;
                for(int i = 0; i < ndim; i++) norm += bk[i] * bk[i];
                norm = std::sqrt(norm);
                if(norm < 1.0e-10) continue;
                for(int i = 0; i < ndim; i++) bk[i] /= norm;
                ortho_basis.push_back(bk);
            }

            for(const auto& ek : ortho_basis){
                double dot = 0.0;
                for(int i = 0; i < ndim; i++) dot += v[i] * ek[i];
                for(int i = 0; i < ndim; i++) v[i] -= dot * ek[i];
            }
        };

        // Initialize optimizer and project out TR from initial gradient
        optimizer->initialize(coords, grad);
        project_out_tr(grad, coords);

        for(int iter = 0; iter < max_iter; iter++){
            // --- Print status ---
            double max_grad = 0.0, rms_grad = 0.0;
            for(int i = 0; i < ndim; i++){
                double g = std::abs(grad[i]);
                if(g > max_grad) max_grad = g;
                rms_grad += grad[i] * grad[i];
            }
            rms_grad = std::sqrt(rms_grad / ndim);

            std::cout << std::endl;
            std::cout << "--- Geometry Optimization Step " << iter << " ---" << std::endl;
            std::cout << std::fixed << std::setprecision(12);
            std::cout << "Energy: " << energy << " Hartree" << std::endl;
            std::cout << std::scientific << std::setprecision(6);
            std::cout << "Max gradient: " << max_grad << " Hartree/Bohr" << std::endl;
            std::cout << "RMS gradient: " << rms_grad << " Hartree/Bohr" << std::endl;

            // --- Check gradient convergence ---
            if(max_grad < grad_threshold && rms_grad < rms_grad_threshold){
                std::cout << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << "         Geometry Optimization Converged!                   " << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << std::fixed << std::setprecision(12);
                std::cout << "Final energy: " << energy << " Hartree" << std::endl;
                std::cout << "Iterations: " << iter << std::endl;
                std::cout << std::endl;
                std::cout << "Optimized Geometry (Bohr):" << std::endl;
                for(int i = 0; i < num_atoms_val; i++){
                    std::cout << std::setw(4) << atomic_number_to_element_name(current_atoms[i].atomic_number)
                              << std::setw(16) << std::setprecision(10) << std::fixed << coords[3*i+0]
                              << std::setw(16) << coords[3*i+1]
                              << std::setw(16) << coords[3*i+2] << std::endl;
                }
                std::cout << std::defaultfloat;
                return energy;
            }

            // --- Compute search direction ---
            std::vector<double> p = optimizer->compute_search_direction(coords, grad);
            project_out_tr(p, coords);

            // --- Check descent direction; fall back to steepest descent if not ---
            {
                double gp = 0.0;
                for(int i = 0; i < ndim; i++) gp += grad[i] * p[i];
                if(gp >= 0.0){
                    std::cout << "Search direction is not a descent direction (g*p=" << gp << "). Resetting to steepest descent." << std::endl;
                    for(int i = 0; i < ndim; i++) p[i] = -grad[i];
                    project_out_tr(p, coords);
                }
            }

            // --- Apply trust radius (scale p if too large) ---
            double max_disp = 0.0;
            for(int i = 0; i < num_atoms_val; i++){
                double disp = std::sqrt(p[3*i]*p[3*i] + p[3*i+1]*p[3*i+1] + p[3*i+2]*p[3*i+2]);
                if(disp > max_disp) max_disp = disp;
            }
            if(max_disp > step_max){
                double scale = step_max / max_disp;
                for(int i = 0; i < ndim; i++) p[i] *= scale;
                max_disp = step_max;
            }

            // --- Step: line search or direct step ---
            double alpha = 1.0;
            real_t new_energy = energy;
            std::vector<double> new_coords(ndim);

            if(optimizer->use_line_search()){
                // Armijo backtracking line search
                double gp = 0.0;
                for(int i = 0; i < ndim; i++) gp += grad[i] * p[i];

                const double c1 = 1.0e-4;
                const int max_ls = 10;
                bool ls_success = false;

                for(int ls = 0; ls < max_ls; ls++){
                    for(int i = 0; i < ndim; i++)
                        new_coords[i] = coords[i] + alpha * p[i];
                    auto new_atom_list = coords_to_atoms(new_coords);
                    update_geometry(new_atom_list);
                    new_energy = single_point_energy();

                    if(std::isfinite(new_energy) && new_energy <= energy + c1 * alpha * gp){
                        ls_success = true;
                        break;
                    }
                    alpha *= 0.5;
                }

                if(!ls_success){
                    // Line search failed: take the smallest step that gives finite energy
                    std::cout << "Line search: Armijo condition not satisfied, using smallest step (alpha=" << alpha << ")" << std::endl;
                    for(int i = 0; i < ndim; i++)
                        new_coords[i] = coords[i] + alpha * p[i];
                    auto new_atom_list = coords_to_atoms(new_coords);
                    update_geometry(new_atom_list);
                    new_energy = single_point_energy();

                    if(!std::isfinite(new_energy)){
                        std::cout << "Line search failed with non-finite energy. Restoring geometry and terminating." << std::endl;
                        // Restore original geometry and re-run SCF to get consistent state
                        auto orig_atom_list = coords_to_atoms(coords);
                        update_geometry(orig_atom_list);
                        single_point_energy();
                        return energy;
                    }
                }
            } else {
                // Direct step (GDIIS)
                for(int i = 0; i < ndim; i++)
                    new_coords[i] = coords[i] + p[i];
                auto new_atom_list = coords_to_atoms(new_coords);
                update_geometry(new_atom_list);
                new_energy = single_point_energy();
            }

            double delta_e = new_energy - energy;

            // Recompute max displacement with actual alpha
            max_disp = 0.0;
            for(int i = 0; i < num_atoms_val; i++){
                double dx = new_coords[3*i] - coords[3*i];
                double dy = new_coords[3*i+1] - coords[3*i+1];
                double dz = new_coords[3*i+2] - coords[3*i+2];
                double disp = std::sqrt(dx*dx + dy*dy + dz*dz);
                if(disp > max_disp) max_disp = disp;
            }

            std::cout << "Max displacement: " << max_disp << " Bohr" << std::endl;
            std::cout << "Energy change: " << delta_e << " Hartree" << std::endl;

            // --- Compute new gradient and project out TR ---
            std::vector<double> new_grad = compute_Energy_Gradient();
            project_out_tr(new_grad, new_coords);

            // --- Update optimizer state ---
            std::vector<double> s(ndim), y(ndim);
            for(int i = 0; i < ndim; i++){
                s[i] = new_coords[i] - coords[i];
                y[i] = new_grad[i] - grad[i];
            }
            optimizer->step_completed(s, y, new_grad);

            // --- Accept step ---
            energy = new_energy;
            coords = new_coords;
            grad = new_grad;
            current_atoms = coords_to_atoms(new_coords);

            // --- Check energy/displacement convergence ---
            if(std::abs(delta_e) < energy_threshold && max_disp < disp_threshold){
                std::cout << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << "         Geometry Optimization Converged!                   " << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << std::fixed << std::setprecision(12);
                std::cout << "Final energy: " << energy << " Hartree" << std::endl;
                std::cout << "Iterations: " << iter + 1 << std::endl;
                std::cout << std::endl;
                std::cout << "Optimized Geometry (Bohr):" << std::endl;
                for(int i = 0; i < num_atoms_val; i++){
                    std::cout << std::setw(4) << atomic_number_to_element_name(current_atoms[i].atomic_number)
                              << std::setw(16) << std::setprecision(10) << std::fixed << coords[3*i+0]
                              << std::setw(16) << coords[3*i+1]
                              << std::setw(16) << coords[3*i+2] << std::endl;
                }
                std::cout << std::defaultfloat;
                return energy;
            }
        }

        std::cout << std::endl;
        std::cout << "WARNING: Geometry optimization did not converge in " << max_iter << " iterations." << std::endl;
        std::cout << std::fixed << std::setprecision(12);
        std::cout << "Current energy: " << energy << " Hartree" << std::endl;
        std::cout << std::defaultfloat;
        return energy;
    } else {
        // energy or gradient mode
        real_t energy = single_point_energy(density_matrix_alpha, density_matrix_beta, force_density);

        if(run_type_ == "gradient"){
            std::vector<double> grad = compute_Energy_Gradient();
            if(!grad.empty()){
                atoms.toHost();
                int num_atoms_val = static_cast<int>(atoms.size());
                std::cout << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << "                   Energy Gradient                          " << std::endl;
                std::cout << "============================================================" << std::endl;
                std::cout << std::scientific << std::setprecision(10);
                std::cout << std::setw(6) << "Atom"
                          << std::setw(20) << "dE/dx"
                          << std::setw(20) << "dE/dy"
                          << std::setw(20) << "dE/dz" << std::endl;
                for(int i = 0; i < num_atoms_val; i++){
                    std::cout << std::setw(4) << atomic_number_to_element_name(atoms[i].atomic_number)
                              << "  " << std::setw(18) << grad[3*i+0]
                              << std::setw(20) << grad[3*i+1]
                              << std::setw(20) << grad[3*i+2] << std::endl;
                }
                std::cout << std::defaultfloat;
            } else {
                std::cout << "Gradient not available for this method." << std::endl;
            }
        }

        return energy;
    }
}


void HF::report(){
    std::cout << std::endl;
    if(atoms.size() == 1){
        std::cout << "[Atom Summary]" << std::endl;
        std::cout << "Atom: " << atomic_number_to_element_name(atoms[0].atomic_number) << " (" << atoms[0].coordinate.x << ", " << atoms[0].coordinate.y << ", " << atoms[0].coordinate.z << ")" << std::endl;
    }
    else{
        std::cout << "[Molecule Summary]" << std::endl;
        std::cout << "Number of atoms: " << atoms.size() << std::endl;
        for(size_t i=0; i<atoms.size(); i++){
            std::cout << "Atom " << i << ": " << atomic_number_to_element_name(atoms[i].atomic_number) << " (" << atoms[i].coordinate.x << ", " << atoms[i].coordinate.y << ", " << atoms[i].coordinate.z << ")" << std::endl;
        }
    }
    std::cout << "Number of electrons: " << num_electrons << std::endl;
    std::cout << "Number of alpha-spin electrons: " << num_alpha_spins << std::endl;
    std::cout << "Number of beta-spin electrons: " << num_beta_spins << std::endl;

    std::cout << std::endl;
    std::cout << "[Basis Set Summary]" << std::endl;
    std::cout << "Number of basis functions: " << num_basis << std::endl;
    std::cout << "Number of primitive basis functions: " << primitive_shells.size() << std::endl;
    if(eri_method_->get_algorithm_name() == "RI"){
        auto* ri_ptr = dynamic_cast<ERI_RI*>(eri_method_.get());
        auto& auxiliary_primitive_shells = ri_ptr->get_auxiliary_primitive_shells(); // get the auxiliary basis set
        std::cout << "Number of auxiliary basis functions: " << ri_ptr->get_num_auxiliary_basis() << std::endl;
        std::cout << "Number of primitive auxiliary basis functions: " << auxiliary_primitive_shells.size() << std::endl;
    }

    // report memory statistics
    CudaMemoryManager<double>::report_memory_statistics();

    if(is_export_molden_){
        export_molden_file("output.molden"); // Export the molecular orbitals to a molden file
    }
}



void HF::update_geometry(const std::vector<Atom>& moved_atoms){
    // update the geometry of the atoms
    atoms.toHost();
    for(size_t i=0; i<atoms.size(); i++){
        atoms[i] = moved_atoms[i];
    }
    atoms.toDevice();

    // update the primitive shells and shell type infos
    primitive_shells.toHost(); // copy the primitive shells to the host memory
    for(size_t i=0; i<primitive_shells.size(); i++){
        for(int atom_index=0; atom_index<atoms.size(); atom_index++){
            if(primitive_shells[i].atom_index == atom_index){
                primitive_shells[i].coordinate = atoms[atom_index].coordinate;
            }
        }
    }
    primitive_shells.toDevice(); // copy the list of primitive shells to the device memory

    // update the primitive shells of the auxiliary basis set if RI is used in ERI calculation, which means that the auxiliary basis set is used
    if(eri_method_->get_algorithm_name() == "RI"){
        auto* ri_ptr = dynamic_cast<ERI_RI*>(eri_method_.get());
        auto& auxiliary_primitive_shells = ri_ptr->get_auxiliary_primitive_shells(); // get the auxiliary basis set
        auxiliary_primitive_shells.toHost(); // copy the auxiliary primitive shells to the host memory
        for(size_t i=0; i<auxiliary_primitive_shells.size(); i++){
            for(int atom_index=0; atom_index<atoms.size(); atom_index++){
                if(auxiliary_primitive_shells[i].atom_index == atom_index){
                    auxiliary_primitive_shells[i].coordinate = atoms[atom_index].coordinate; // update the coordinate of the primitive shell
                }
            }
        }
        auxiliary_primitive_shells.toDevice(); // copy the list of auxiliary primitive shells to the device memory
    }

}



} // namespace gansu
