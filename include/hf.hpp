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
 * @file hf.hpp 
 * @brief This file contains the definition of the HF class and update Fock matrix class.
 */

#pragma once

#include "types.hpp"
#include "basis_set.hpp"
#include "molecular.hpp"

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "parameter_manager.hpp"
#include "eri.hpp"

namespace gansu{


// prototype of classes
class ERI;

/**
 * @brief HF class
 * @details This class is a virtual class for the Hartree-Fock method.
 * Computations related only to AO basis are implemented in this class
 */
class HF{
public:

    HF(const Molecular& molecular, const ParameterManager& parameters);
    HF(const Molecular& molecular): HF(molecular, ParameterManager()){} ///< Constructor with default parameters

    HF(const HF&) = delete; ///< copy constructor is deleted
    virtual ~HF() = default; ///< destructor

    /**
     * @brief Compute the nuclear repulsion energy
     * @details This function computes the nuclear repulsion energy.
     * @details The nuclear repulsion energy is given by \f$\displaystyle E_{\text{nuc}} = \sum_{i} \sum_{j>i} \frac{Z_i Z_j}{r_{ij}} \f$
     */
    void compute_nuclear_repulsion_energy();

    /**
     * @brief Compute the core Hamiltonian matrix
     * @details This function computes the overlap matrix and the core Hamiltonian matrix (kinetic energy + nuclear attraction)
     * @details The overlap matrix is given by \f$ S_{\mu \nu} = \int \chi_{\mu}(\mathbf{r}) \chi_{\nu}(\mathbf{r}) d\mathbf{r} \f$
     * @details The kinetic energy matrix is given by \f$ T_{\mu \nu} = -\frac{1}{2} \int \chi_{\mu}(\mathbf{r}) \nabla^2 \chi_{\nu}(\mathbf{r}) d\mathbf{r} \f$
     * @details The nuclear attraction matrix is given by \f$ V_{\mu \nu} = -\sum_{A} \int \chi_{\mu}(\mathbf{r}) \frac{Z_A}{r_A} \chi_{\nu}(\mathbf{r}) d\mathbf{r} \f$
     * @details The core Hamiltonian matrix is given by \f$ h_{\mu \nu} = T_{\mu \nu} + V_{\mu \nu} \f$
     */
    void compute_core_hamiltonian_matrix();

    /**
     * @brief Precompute the ERIs
     * @details This function precomputes the electron repulsion integrals (ERIs)
     * @details The exection depends on the algorithm in the derived class.
     */
    virtual void precompute_eri_matrix()=0;

    /**
     * @brief Compute the transformation matrix
     * @details This function computes the transform matrix \f$X\f$ by diagonalizing the overlap matrix \f$S\f$.
     * @details (1) Symetrize the overlap matrix by \f$U^{T}SU = s\f$
     * @details (2) Compute the transformation matrix by \f$X = U s^{-1/2}\f$
     */
    void compute_transform_matrix();


    /**
     * @brief Get the number of basis functions
     */
    int get_num_basis() const { return num_basis; }

    /**
     * @brief Get the number of electrons
     */
    int get_num_electrons() const { return num_electrons; }

    /**
     * @brief Get the number of alpha electrons
     */
    int get_num_alpha_spins() const { return num_alpha_spins; }

    /**
     * @brief Get the number of beta electrons
     */
    int get_num_beta_spins() const { return num_beta_spins; }


    /**
     * @brief Get the boolean value of verbose mode
     */
    bool get_verbose() const { return verbose; }

    /**
     * @brief Get the nuclear repulsion energy
     */
    real_t get_nuclear_repulsion_energy() const { return nuclear_repulsion_energy_; }


    /**
     * @brief Virtual function to get the energy
     * @return Energy
     * @details This function gets the energy.
     * @details This function must be implemented in the derived class.
    */
    virtual real_t get_energy() const = 0;

    /**
     * @brief Get the total energy
     * @details This function gets the total energy.
    */
    real_t get_total_energy() const { return get_energy() + nuclear_repulsion_energy_; }

    /**
     * @brief Get the total spin <S^2>
     * @return Total spin <S^2>
     * @details This function gets the total spin <S^2>.
     * @details In the current UHF implementation, the coeefficient matrices are copied to the host memory. Therefore, "const" is removed from the function.
     */
    virtual real_t get_total_spin() = 0;

    /**
     * @brief Get the reference to the overlap matrix
     * @return Reference to the overlap matrix
     */
    DeviceHostMatrix<real_t>& get_overlap_matrix() { return overlap_matrix; }

    /**
     * @brief Get the reference to the transform matrix
     * @return Reference to the transform matrix
     */
    DeviceHostMatrix<real_t>& get_transform_matrix() { return transform_matrix; }

    /**
     * @brief Get the core Hamiltonian matrix
     * @return Core Hamiltonian matrix
     */
    DeviceHostMatrix<real_t>& get_core_hamiltonian_matrix() { return core_hamiltonian_matrix; }

    /**
     * @brief Get the computing time to solve the HF equation in milliseconds
     * @return Time to solve the HF equation in milliseconds
    */
    long long get_solve_time_in_milliseconds() const { return solve_time_in_milliseconds_; }

    /**
     * @brief Get atom_to_basis_range
     */
    const std::vector<BasisRange>& get_atom_to_basis_range() const { return atom_to_basis_range; }

    /**
     * @brief Get Atoms
     */
    const DeviceHostMemory<Atom>& get_atoms() const { return atoms; }


    /**
     * @brief Get the shell type infos
     */
    const std::vector<ShellTypeInfo>& get_shell_type_infos() const { return shell_type_infos; }

    /**
     * @brief Get the primitive shells
     */
    const DeviceHostMemory<PrimitiveShell>& get_primitive_shells() const { return primitive_shells; }

    /**
     * @brief Get boys_grid
     */
    const DeviceHostMemory<real_t>& get_boys_grid() const { return boys_grid; }

    /**
     * @brief Get cgto_nomalization_factors
     */
    const DeviceHostMemory<real_t>& get_cgto_nomalization_factors() const { return cgto_nomalization_factors; }

    /**
     * @brief Get the Schwartz screening threshold
     */
    real_t get_schwarz_screening_threshold() const { return schwarz_screening_threshold; }

    /**
     * @brief Get Shell-pair type info
     */
    const std::vector<ShellPairTypeInfo>& get_shell_pair_type_infos() const { return shell_pair_type_infos; }

    /**
     * @brief Get num_primitive_shell_pairs
     */
    size_t get_num_primitive_shell_pairs() const { return num_primitive_shell_pairs; }


    /**
     * @brief Single point energy calculation
     * @param density_matrix_alpha Density matrix of alpha spin if UHF, otherwise the density matrix (optional)
     * @param density_matrix_beta Density matrix of beta spin (optional)
     * @param force_density Density matrix is used in the initial guess
     */
    real_t single_point_energy(const real_t* density_matrix_alpha=nullptr, const real_t* density_matrix_beta=nullptr, bool force_density=false);

protected:
    long long solve_time_in_milliseconds_; ///< Time to solve the HF equation

    const int num_basis; ///< Number of basis functions

    const int num_electrons; ///< Number of electrons
    const int num_alpha_spins; ///< Number of alpha electrons
    const int num_beta_spins; ///< Number of beta electrons

    const int verbose; ///< Verbose mode
    const real_t convergence_energy_threshold; ///< Convergence criterion
    const real_t schwarz_screening_threshold; ///< Schwartz screening threshold
    const int max_iter; ///< Maximum number of iterations
    int iter_; ///< Number of iterations
    real_t energy_difference_; ///< Energy difference between the current and the previous iteration
    const int geometry_optimization; ///< Geometry optimization flag
    const std::string geometry_optimization_method; ///< Geometry optimization method
    
    const std::vector<ShellTypeInfo> shell_type_infos; ///< Shell type info in the primitive shell list
    const std::vector<BasisRange> atom_to_basis_range; ///< Basis range for each atom

    DeviceHostMemory<Atom> atoms; ///< Atoms
    DeviceHostMemory<PrimitiveShell> primitive_shells; ///< Primitive shells

    DeviceHostMemory<real_t> boys_grid; ///< grid values for the Boys function
    DeviceHostMatrix<real_t> overlap_matrix; ///< Overlap matrix
    DeviceHostMatrix<real_t> core_hamiltonian_matrix; ///< Core Hamiltonian matrix (kinetic energy + nuclear attraction)
    DeviceHostMemory<real_t> cgto_nomalization_factors; ///< Normalization factors of the contracted Gauss functions


    DeviceHostMatrix<real_t> transform_matrix; ///< Transformation matrix

    real_t nuclear_repulsion_energy_; ///< Nuclear repulsion energy

    // for Diect SCF
    std::vector<ShellPairTypeInfo> shell_pair_type_infos;
    size_t num_primitive_shell_pairs;

    // for ERI (stored, RI, direct)
    std::unique_ptr<ERI> eri_method_; ///< ERI method

    /**
     * @brief Virtual function to compute the Fock matrix
     * @details This function computes the Fock matrix.
     * @details This function must be implemented in the derived class.
     */
    virtual void compute_fock_matrix()=0;

    /**
     * @brief Virtual function to compute the density matrix
     * @details This function computes the density matrix.
     * @details This function must be implemented in the derived class.
    */
    virtual void compute_density_matrix()=0;

    /**
     * @brief Virtual function to guess the initial Fock matrix
     * @param density_matrix_a Density matrix of alpha spin if UHF, otherwise the density matrix (optional)
     * @param density_matrix_b Density matrix of beta spin if UHF, otherwise no use (optional)
     * @param force_density Density matrix is used in the initial guess (optional)
     * @param 
     * @details This function guesses the initial Fock matrix.
     * @details This function must be implemented in the derived class.
     */
    virtual void guess_initial_fock_matrix(const real_t* density_matrix_a=nullptr, const real_t* density_matrix_b=nullptr, bool force_density=false)=0;


    /**
     * @brief Virtual function to compute the coefficient matrix
     * @details This function computes the coefficient matrix.
     * @details This function must be implemented in the derived class.
    */
    virtual void compute_coefficient_matrix()=0;

    /**
     * @brief Virtual function to compute the energy
     * @details This function computes the energy.
     * @details This function must be implemented in the derived class.
    */
    virtual void compute_energy()=0;

    /**
     * @brief Update the Fock matrix 
     * @details This function updates the Fock matrix.
     * @details This function is implemented in the derived class.
     */
    virtual void update_fock_matrix() = 0;




    /**
     * @brief Update the geometry of the molecule
     * @param moved_atoms New geometry of the atoms
     * @details This function updates the geometry of the molecule.
     * @details This function updates coordinates of atoms and primitive_shells.
     * @details This function also updates the auxiliary basis set if RI method is used.
     */
    void update_geometry(const std::vector<Atom>& moved_atoms);


    /**
     * @brief Analyze Mulliken population
     * @returns Mulliken population analysis per atom
     * @details This function analyzes the Mulliken population.
     * @details This function is a virtual function and must be implemented in the derived class.
    */
    virtual std::vector<real_t> analyze_mulliken_population() const = 0;

    /**
     * @brief Compute Mayer bond order
     * @returns Mayer bond order matrix
     * @details This function computes the Mayer bond order.
     * @details This function is a virtual function and must be implemented in the derived class.
    */
    virtual std::vector<std::vector<real_t>> compute_mayer_bond_order() const = 0;

    /**
     * @brief Compute Wiberg bond order
     * @returns Wiberg bond order matrix
     * @details This function computes the Wiberg bond order.
     * @details This function is a virtual function and must be implemented in the derived class.
    */
    virtual std::vector<std::vector<real_t>> compute_wiberg_bond_order() = 0;

public:

    /**
    * @brief Function to solve the Hartree-Fock equation by the SCF procedure
    * @param density_matrix Density matrix_alpha (optional), density_matrix_beta (optional)
    * @param force_density Density matrix is used in the initial guess (optional)
    * @return Energy of the system
    * @details This function solves the Hartree-Fock equation by iterating the SCF procedure.
    */
    virtual real_t solve(const real_t* density_matrix_alpha=nullptr, const real_t* density_matrix_beta=nullptr, bool force_density=false); ///< Solve the HF equation by the SCF method



    /**
     * @brief Print the information of the input molecular and basis set
     * @details This function prints the information of the input molecular and basis set.
     * @details The information includes the number of atoms, the number of electrons, the number of basis functions, and the number of primitive basis functions.
     * @details This function is called in the derived classes.
     */
    virtual void report(); ///< Report the results. 
  

    /**
     * @brief Export the results as a Molden format file
     * @param filename File name
     * @details This function exports the results as a Molden format file.
     * @details This function is implemented in the derived class.
     */
    virtual void export_molden_file(const std::string& filename) = 0;

    /**
     * @brief Export the density matrix
     * @param density_matrix_a Density matrix (alpha spin) if UHF, otherwise the density matrix
     * @param density_matrix_b Density matrix (beta spin) if UHF, otherwise no use
     * @param num_basis Number of basis functions
     * @details This function exports the density matrix.
     * @details This function is implemented in the derived class.
     * @details Matrix must be allocated before calling this function, and the size of the matrix must be num_basis x num_basis.
     */
    virtual void export_density_matrix(real_t* density_matrix_a, real_t* density_martix_b, const int num_basis) = 0;

    void generate_sad_cache(const std::string& sad_cache_filename) {
        // This function is called after solving the HF equation
        // to generate the SAD cache file for the SAD initial guess method.
        // The SAD cache file is used to store the density matrices of the atoms.

        std::ofstream sad_cache_file(sad_cache_filename);
        if (!sad_cache_file.is_open()) {
            throw std::runtime_error("Failed to open the SAD cache file: " + sad_cache_filename);
        }
        std::vector<double> density_matrix_alpha(num_basis * num_basis);
        std::vector<double> density_matrix_beta(num_basis * num_basis);
        export_density_matrix(density_matrix_alpha.data(), density_matrix_beta.data(), num_basis);


        sad_cache_file << num_basis << std::endl; // write the number of basis functions
        sad_cache_file << std::setprecision(10) << std::scientific;
        // Write alpha matrix (row-major)
        for (int i = 0; i < num_basis; ++i) {
            for (int j = 0; j < num_basis; ++j) {
                sad_cache_file << density_matrix_alpha[i * num_basis + j] << " ";
            }
            sad_cache_file << "\n";
        }

        // Write beta matrix (row-major)
        for (int i = 0; i < num_basis; ++i) {
            for (int j = 0; j < num_basis; ++j) {
                sad_cache_file << density_matrix_beta[i * num_basis + j] << " ";
            }
            sad_cache_file << "\n";
        }
    }
};


} // namespace gansu