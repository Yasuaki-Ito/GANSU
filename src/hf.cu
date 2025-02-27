/*
 * Quantum GANSU: GPU Acclerated Numerical Simulation Utility
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
    cgto_nomalization_factors(molecular.get_cgto_normalization_factors()), // construct the list directly from std::vector
    overlap_matrix(num_basis, num_basis), // host memory is not allocated in advance
    core_hamiltonian_matrix(num_basis, num_basis), // host memory is not allocated in advance
    transform_matrix(num_basis, num_basis), // host memory is not allocated in advance
    verbose(parameters.get<int>("verbose")),
    max_iter(parameters.get<int>("maxiter")),
    convergence_energy_threshold(parameters.get<double>("convergence_energy_threshold")),
    schwarz_screening_threshold(parameters.get<double>("schwarz_screening_threshold"))
{
    // print all the values of boys function for the test (temporary implementation)
    if(verbose){
        std::cout << "Messege: grid values for the Boys function is load from the header file." << std::endl;
    }
    for(size_t i=0; i<30720; i++){
        //std::cout << i << ": " << h_boys_grid[i] << std::endl;
        boys_grid[i] = h_boys_grid[i];
    }

    // for Schwarz screening in Stored ERI and Direct SCF
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
    cgto_nomalization_factors.toDevice(); // copy the list of cgto normalization factors to the device memory

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
    gpu::computeCoreHamiltonianMatrix(shell_type_infos, atoms.device_ptr(), primitive_shells.device_ptr(), boys_grid.device_ptr(), cgto_nomalization_factors.device_ptr(), overlap_matrix.device_ptr(), core_hamiltonian_matrix.device_ptr(),atoms.size(), num_basis, verbose);

    // print the overlap and core Hamiltonian matrix
    if(verbose){
        // copy the core Hamiltonian matrix to the host memory
        overlap_matrix.toHost();
        core_hamiltonian_matrix.toHost();

        std::cout << "Overlap matrix:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << overlap_matrix(i, j) << " "; 
            }
            std::cout << std::endl;
        }

        std::cout << "Core Hamiltonian matrix:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << core_hamiltonian_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
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




real_t HF::solve(const real_t* density_matrix_alpha, const real_t* density_matrix_beta){
//    PROFILE_FUNCTION();
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
    

    guess_initial_fock_matrix(density_matrix_alpha, density_matrix_beta); // guess the initial Fock matrix

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

        std::cout << "---- Iteration: " << iter_ << " ----  ";
        std::cout << "Energy: " << std::setprecision(17) << energy << " ";
        std::cout << "Total energy: " << std::setprecision(17) << get_total_energy() << " ";
        std::cout << "Difference: " << energy_difference_ << std::endl;

        if(energy_difference_ < convergence_energy_threshold){
            break;
        }
        iter_++;

        // store the previous energy
        prev_energy = energy;

        // Update the Fock matrix
        update_fock_matrix();
    }

    solve_time_in_milliseconds_ = timer.elapsed_milliseconds();

    GlobalProfiler::report(); // Print the profiling results
    
    return get_total_energy();
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
            std::cout << "Atom " << i+1 << ": " << atomic_number_to_element_name(atoms[i].atomic_number) << " (" << atoms[i].coordinate.x << ", " << atoms[i].coordinate.y << ", " << atoms[i].coordinate.z << ")" << std::endl;
        }
    }
    std::cout << "Number of electrons: " << num_electrons << std::endl;
    std::cout << "Number of alpha-spin electrons: " << num_alpha_spins << std::endl;
    std::cout << "Number of beta-spin electrons: " << num_beta_spins << std::endl;

    std::cout << std::endl;
    std::cout << "[Basis Set Summary]" << std::endl;
    std::cout << "Number of basis functions: " << num_basis << std::endl;
    std::cout << "Number of primitive basis functions: " << primitive_shells.size() << std::endl;
}


} // namespace gansu