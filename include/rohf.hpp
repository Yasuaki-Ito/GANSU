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
 * @file rohf.hpp 
 * @brief This file contains the definition of the ROHF class.
 */


#pragma once

#include "hf.hpp"
#include <memory> // std::unique_ptr

#include "profiler.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{

// prototype of classes
class Convergence_ROHF;
class Convergence_ROHF_Damping;
class Convergence_ROHF_DIIS;


/**
 * @brief ROHF class
 * 
 */
class ROHF : public HF {
public:

    ROHF(const Molecular& molecular, const ParameterManager& parameters);
    ROHF(const Molecular& molecular): ROHF(molecular, ParameterManager()){} ///< Constructor with default parameters
 
    ROHF(const ROHF&) = delete; ///< copy constructor is deleted
    ~ROHF() = default; ///< destructor

    void precompute_eri_matrix() override;
    void compute_fock_matrix() override;
    void compute_density_matrix() override;
    void guess_initial_fock_matrix(const real_t* density_matrix_a=nullptr, const real_t* density_matrix_b=nullptr, bool force_density=false) override;
    void compute_coefficient_matrix() override;
    void compute_energy() override;
    void update_fock_matrix() override;

    real_t get_energy() const override { return energy_; }
    real_t get_total_spin() override { return (num_open/2.0) * (num_open/2.0+1); };

    void report() override;

    void set_convergence_method(std::unique_ptr<Convergence_ROHF> convergence_method);

    void set_eri_method(std::unique_ptr<ERI> eri_method);

    std::vector<real_t> analyze_mulliken_population() const override;

    std::vector<std::vector<real_t>> compute_mayer_bond_order() const override;

    std::vector<std::vector<real_t>> compute_wiberg_bond_order() override;

    /**
     * @brief Get the reference to the coefficient matrix
     * @return Reference to the coefficient matrix
     */
    DeviceHostMatrix<real_t>& get_coefficient_matrix() { return coefficient_matrix; }

    /**
     * @brief Get the reference to the density matrix (closed orbitals)
     * @return Reference to the density matrix (closed orbitals)
     */
    DeviceHostMatrix<real_t>& get_density_matrix_closed() { return density_matrix_closed; }


    /**
     * @brief Get the reference to the density matrix (open orbitals)
     * @return Reference to the density matrix (open orbitals)
     */
    DeviceHostMatrix<real_t>& get_density_matrix_open() { return density_matrix_open; }

    /**
     * @brief Get the reference to the density matrix (sum of the closed and open orbitals)
     * @return Reference to the density matrix (sum of the closed and open orbitals)
     */
    DeviceHostMatrix<real_t>& get_density_matrix() { return density_matrix; }

    /**
     * @brief Get the reference to the Fock matrix (closed orbitals)
     * @return Reference to the Fock matrix (closed orbitals)
     */
    DeviceHostMatrix<real_t>& get_fock_matrix_closed() { return fock_matrix_closed; }

    /**
     * @brief Get the reference to the Fock matrix (open orbitals)
     * @return Reference to the Fock matrix (open orbitals)
     */
    DeviceHostMatrix<real_t>& get_fock_matrix_open() { return fock_matrix_open; }

    /**
     * @brief Get the reference to the Fock matrix (unified)
     * @return Reference to the Fock matrix (unified)
     */
    DeviceHostMatrix<real_t>& get_fock_matrix() { return fock_matrix; }

    /**
     * @brief Export the density matrix
     * @param density_matrix_a Density matrix (alpha spin) if UHF, otherwise the density matrix
     * @param density_matrix_b Density matrix (beta spin) if UHF, otherwise no use
     * @param num_basis Number of basis functions
     * @details This function exports the density matrix.
     * @details Matrix must be allocated before calling this function, and the size of the matrix must be num_basis x num_basis.
     */
    void export_density_matrix(real_t* density_matrix_a, real_t* density_martix_b, const int num_basis) override;


    /**
     * @brief Get the basis set file name (gbs)
     * @return Basis set file name as a string
     */
    std::string get_gbsfilename() const { return gbsfilename_; } ///< Get the basis set file name


    /**
     * @brief Export the molecular orbitals to the results as a Molden format file
     * @param filename File name
     */
    void export_molden_file(const std::string& filename) override;

    const ROHF_ParameterSet& get_ROHF_parameters() const { return ROHF_parameters; } ///< Get the ROHF parameters

    int get_num_open() const { return num_open; } ///< Get the number of open shells
    int get_num_closed() const { return num_closed; } ///< Get the number of closed shellss

private:
    const int num_closed; ///< Number of closed shells
    const int num_open; ///< Number of open shells
    const int num_virtual; ///< Number of virtual shells

    const std::string ROH_parameter_name; ///< ROHF parameter name for computing the unified Fock matrix
    const ROHF_ParameterSet ROHF_parameters; ///< ROHF parameters for computing the unified Fock matrix

    real_t energy_; ///< Energy
    DeviceHostMemory<real_t> orbital_energies; ///< Orbital energies

    DeviceHostMatrix<real_t> coefficient_matrix; ///< Coefficient matrix
    DeviceHostMatrix<real_t> density_matrix_closed; ///< Density matrix (closed shell)
    DeviceHostMatrix<real_t> density_matrix_open; ///< Density matrix (open shell)
    DeviceHostMatrix<real_t> density_matrix; ///< Density matrix (sum of the closed and open shell)
    DeviceHostMatrix<real_t> fock_matrix_closed; ///< Fock matrix (closed shell)
    DeviceHostMatrix<real_t> fock_matrix_open; ///< Fock matrix (open shell)
    DeviceHostMatrix<real_t> fock_matrix; ///< Fock matrix (unified)

    std::unique_ptr<Convergence_ROHF> convergence_method_; ///< Convergence_ROHF

    const std::string initail_guess_method_; ///< Initial guess method name
    const std::string gbsfilename_; ///< Basis set file name (Gaussian basis set file)

};



/**
 * @brief Convergence_ROHF class for a convergence algorithm for the ROHF method
 * @details This class is a virtual class for updating the Fock matrix of the ROHF method.
 */
class Convergence_ROHF {
public:
    /**
     * @brief Constructor of the Convergence_ROHF class
     * @param hf ROHF
     * @details This function constructs the Convergence_ROHF class.
     * @details The ROHF is given as an argument.
     */
    Convergence_ROHF(ROHF& hf) 
    : hf_(hf),
      verbose(hf.get_verbose()){}

    Convergence_ROHF(const Convergence_ROHF&) = delete; ///< copy constructor is deleted
    virtual ~Convergence_ROHF() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix.
     */
    virtual void get_new_fock_matrix() = 0;


    /**
     * @brief Get the algorithm name
     * @return Algorithm name as a string
     */
    virtual std::string get_algorithm_name() const = 0;

protected:
    ROHF& hf_; ///< ROHF
    bool verbose; ///< Verbose mode
};

/**
 * @brief Convergence_ROHF_Damping class for the restricted open-shell HF method.
 * @details This class performs the damping of the Fock matrix for the restricted open-shell HF method.
 * @details The damping factor can be constant or optimized.
 * @details The constant damping factor is given as an argument.
 * @details The optimal damping factor is obtained by the optial damping algorithm in @cite Zerner1979 and @cite Cances2000.
 */
class Convergence_ROHF_Damping : public Convergence_ROHF {
public:
    /**
     * @brief Constructor of the UpdateFockMatrix_ROHF_ROamping class with constant damping factor
     * @param hf ROHF
     * @param damping_factor Damping factor
     * @details This function constructs the Convergence_ROHF_Damping class.
     * @details The ROHF and the damping factor are given as arguments.
     */
    Convergence_ROHF_Damping(ROHF& hf, const real_t damping_factor) 
        : Convergence_ROHF(hf), 
        damping_factor_(damping_factor),
        use_optimized_(false),
        first_iteration_(true),
        prev_density_matrix(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix(hf.get_num_basis(), hf.get_num_basis()) {}
    
    /**
     * @brief Constructor of the Convergence_ROHF_Damping class with optimized damping factor
     * @param hf ROHF
     * @details This function constructs the Convergence_ROHF_Damping class.
     * @details The ROHF is given as an argument.
     */
    Convergence_ROHF_Damping(ROHF& hf) 
        : Convergence_ROHF(hf), 
        damping_factor_(0.5),
        use_optimized_(true),
        first_iteration_(true),
        prev_density_matrix(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix(hf.get_num_basis(), hf.get_num_basis()) {}

    Convergence_ROHF_Damping(const Convergence_ROHF_Damping&) = delete; ///< copy constructor is deleted
    ~Convergence_ROHF_Damping() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix with damping.
     */
    void get_new_fock_matrix() override {
        if (first_iteration_) { // First iteration: no damping, just store the density matrix and the Fock matrix
            first_iteration_ = false;
            cudaMemcpy(prev_fock_matrix.device_ptr(), hf_.get_fock_matrix().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(prev_fock_matrix.device_ptr(), hf_.get_fock_matrix().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            return;
        }else{
            real_t factor;
             if (use_optimized_) { // Optimized damping factor
                factor = gpu::computeOptimalDampingFactor_ROHF(hf_.get_fock_matrix().device_ptr(), prev_fock_matrix.device_ptr(),hf_.get_density_matrix().device_ptr(), prev_density_matrix.device_ptr(), hf_.get_num_basis());
                if(verbose) std::cout << "Damping factor (optimal): " << factor << std::endl;
                // Damping (after dammping, store the density matrix and the Fock matrix to the previous density matrix and the previous Fock matrix, respectively)
                gpu::damping(prev_fock_matrix.device_ptr(), hf_.get_fock_matrix().device_ptr(), factor, hf_.get_num_basis());
                gpu::damping(prev_density_matrix.device_ptr(), hf_.get_density_matrix().device_ptr(), factor, hf_.get_num_basis());
            }else{ 
                factor = damping_factor_;
                if(verbose) std::cout << "Damping factor (constant): " << factor << std::endl;
                // Damping (after dammping, store the Fock matrix to the previous Fock matrix)
                gpu::damping(prev_fock_matrix.device_ptr(), hf_.get_fock_matrix().device_ptr(), factor, hf_.get_num_basis());
            }
        }
    }

    /**
     * @brief Get the algorithm name
     */
    std::string get_algorithm_name() const override {
        std::string name = "";
        if(use_optimized_){
            name = "Optimal damping";
        }else{
            name = "Damping (alpha = " + std::to_string(damping_factor_) + ")";
        }
        return name;
    }

private:
    real_t damping_factor_; ///< Damping factor
    const bool use_optimized_; ///< Optimized damping factor
    bool first_iteration_; ///< First iteration

    DeviceHostMatrix<real_t> prev_density_matrix; ///< Previous density matrix (sum of the closed and open shell)
    DeviceHostMatrix<real_t> prev_fock_matrix; ///< Previous Fock matrix (unified)
};




/**
 * @brief Convergence_ROHF_DIIS class for the restricted HF method
 * @details This class performs the update the Fock matrix for the restricted HF method using DIIS @cite Pulay1980, @cite Pulay1982.
 * @details The number num_prev of the previous Fock matrices to be stored is given as an argument.
 */
class Convergence_ROHF_DIIS : public Convergence_ROHF {
public:
    /**
     * @brief Constructor of the UpdateFockMatrix_ROHF_DIIS class
     * @param hf ROHF
     * @param num_prev The number of the previous Fock matrices to be stored
     * @param is_include_transform Include the transformation matrix in the error matrix
     * @details This function constructs the Convergence_ROHF_DIIS class.
     */
    Convergence_ROHF_DIIS(ROHF& hf, const real_t num_prev=10, const bool is_include_transform=false) 
        : Convergence_ROHF(hf), 
        num_prev_(num_prev),
        is_include_transform_(is_include_transform),
        num_basis_(hf.get_num_basis()),
        iteration_(0),
        error_matrix(hf_.get_num_basis(),hf_.get_num_basis()),
        prev_error_matrices(num_prev * num_basis_ * num_basis_),
        prev_fock_matrices(num_prev * num_basis_ * num_basis_){}

    Convergence_ROHF_DIIS(const Convergence_ROHF_DIIS&) = delete; ///< copy constructor is deleted
    ~Convergence_ROHF_DIIS() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix with damping.
     */
    void get_new_fock_matrix() override {
        // Compute the error matrix
        gpu::computeDIISErrorMatrix(
            hf_.get_overlap_matrix().device_ptr(), 
            hf_.get_transform_matrix().device_ptr(), 
            hf_.get_fock_matrix().device_ptr(), 
            hf_.get_density_matrix().device_ptr(), 
            error_matrix.device_ptr(), 
            hf_.get_num_basis(),
            is_include_transform_);


        // Copy the previous error matrix and the previous Fock matrix to the new error matrix and the new Fock matrix at most num_prev matrices
        const int store_prev_index = iteration_ % num_prev_; // Overwrite the previous matrices cyclically
        cudaMemcpy(&prev_error_matrices.device_ptr()[store_prev_index * num_basis_ * num_basis_], error_matrix.device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&prev_fock_matrices.device_ptr()[store_prev_index * num_basis_ * num_basis_], hf_.get_fock_matrix().device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);

        // Compute the DIIS coefficients
        const int num_prevs = std::min(iteration_+1, num_prev_);

        if(num_prevs > 1){ // first iteration: no DIIS
            gpu::computeFockMatrixDIIS(
                prev_error_matrices.device_ptr(), 
                prev_fock_matrices.device_ptr(), 
                hf_.get_fock_matrix().device_ptr(), 
                num_prevs, 
                num_basis_);
        }
        iteration_++;
    }

    /**
     * @brief Get the algorithm name
     */
    std::string get_algorithm_name() const override {
        std::string name = "DIIS";
        name += "(";
        name +=   "diis_size: " + std::to_string(num_prev_) + ", ";
        name +=   "diis_include_transform: ";
        name +=       (is_include_transform_) ? "true" : "false";
        name += ")";
        return name;
    }

private:
    int iteration_; ///< count of iterations

    const int num_basis_; ///< Number of basis functions
    const int num_prev_; ///< Number of the previous Fock matrices to be stored
    const bool is_include_transform_; ///< Include the transformation matrix in the DIIS error matrix

    DeviceHostMatrix<real_t> error_matrix; ///< Error matrix

    DeviceHostMemory<real_t> prev_error_matrices; ///< Previous error matrices
    DeviceHostMemory<real_t> prev_fock_matrices; ///< Previous Fock matrices
};




/**
 * @brief InitialGuess_ROHF class for the restricted HF method
 * @details This class is a virtual class for the initial guess of the restricted HF method.
 * @details This class will be derived to implement the initial guess.
 * @details The initial guess is used for the initial Fock matrix.
 */
class InitialGuess_ROHF {
public:
    InitialGuess_ROHF(ROHF& hf) : hf_(hf) {}
    InitialGuess_ROHF(const InitialGuess_ROHF&) = delete;
    virtual ~InitialGuess_ROHF() = default;

    virtual void guess() = 0;

protected:
    ROHF& hf_;
};

/**
 * @brief InitialGuess_ROHF_Core class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the core Hamiltonian matrix.
 */
class InitialGuess_ROHF_Core : public InitialGuess_ROHF {
public:
    InitialGuess_ROHF_Core(ROHF& hf) : InitialGuess_ROHF(hf) {}
    InitialGuess_ROHF_Core(const InitialGuess_ROHF_Core&) = delete;
    ~InitialGuess_ROHF_Core() = default;

    void guess() override {
        // initial guess for the Coefficient matrix by the core Hamiltonian matrix
        gpu::computeCoefficientMatrix(
            hf_.get_core_hamiltonian_matrix().device_ptr(), // core Hamiltonian matrix is used instead of the Fock matrix 
            hf_.get_transform_matrix().device_ptr(),
            hf_.get_coefficient_matrix().device_ptr(),
            hf_.get_num_basis()
        );

        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

    }
};


/**
 * @brief InitialGuess_ROHF_GWH class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Generalized Wolfsberg-Helmholtz method (GWH) @cite Wolfsberg1952.
 */
class InitialGuess_ROHF_GWH : public InitialGuess_ROHF {
public:
    InitialGuess_ROHF_GWH(ROHF& hf) : InitialGuess_ROHF(hf) {}
    InitialGuess_ROHF_GWH(const InitialGuess_ROHF_GWH&) = delete;
    ~InitialGuess_ROHF_GWH() = default;

    void guess() override {
        // initial guess for the Fock matrix by the core Hamiltonian matrix
        gpu::computeInitialCoefficientMatrix_GWH(
            hf_.get_core_hamiltonian_matrix().device_ptr(),
            hf_.get_overlap_matrix().device_ptr(),
            hf_.get_transform_matrix().device_ptr(),
            hf_.get_coefficient_matrix().device_ptr(),
            hf_.get_num_basis()
        );

        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
    }
};


/**
 * @brief InitialGuess_ROHF_Density class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted open-shell HF method.
 * @details The initial Fock matrix is computed from the density matrix given as an argument.
 */
class InitialGuess_ROHF_Density : public InitialGuess_ROHF {
public:
    InitialGuess_ROHF_Density(ROHF& hf, const real_t* density_matrix_a, const real_t* density_matrix_b) : InitialGuess_ROHF(hf), density_matrix_a_(density_matrix_a), density_matrix_b_(density_matrix_b) {
        if(density_matrix_a_ == nullptr || density_matrix_b_ == nullptr){
            THROW_EXCEPTION("density_matrix is nullptr");
        }
    }
    InitialGuess_ROHF_Density(const InitialGuess_ROHF_Density&) = delete;
    ~InitialGuess_ROHF_Density() = default;

    void guess() override {
        // initial guess from the density matrix given as an argument
        { // open shell
            std::unique_ptr<real_t[]> density_matrix_open(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
            for(size_t i=0; i<hf_.get_num_basis() * hf_.get_num_basis(); i++){
                density_matrix_open[i] = density_matrix_a_[i] - density_matrix_b_[i];
            }
            cudaMemcpy(hf_.get_density_matrix_open().device_ptr(), density_matrix_open.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        }
        { // closed shell
            std::unique_ptr<real_t[]> density_matrix_closed(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
            for(size_t i=0; i<hf_.get_num_basis() * hf_.get_num_basis(); i++){
                density_matrix_closed[i] = 2.0 * density_matrix_b_[i];
            }
            cudaMemcpy(hf_.get_density_matrix_closed().device_ptr(), density_matrix_closed.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        }
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
        hf_.compute_coefficient_matrix(); // compute the coefficient matrix from the Fock matrix
        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
    }
private:
    const real_t* density_matrix_a_;
    const real_t* density_matrix_b_;
};



/**
 * @brief InitialGuess_ROHF_SAD class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Superposition of Atomic Densities (SAD) @cite Lenthe2006.
 */
class InitialGuess_ROHF_SAD : public InitialGuess_ROHF {
public:
    InitialGuess_ROHF_SAD(ROHF& hf) : InitialGuess_ROHF(hf) {}
    InitialGuess_ROHF_SAD(const InitialGuess_ROHF_SAD&) = delete;
    ~InitialGuess_ROHF_SAD() = default;

    /**
     * @brief Get or compute the density matrix of the atom
     * @param atomic_number Atomic number
     * @param monatomic_molecule Monatomic molecule
     * @return Pair of the density matrices (alpha- and beta-spins)
     * @details This function gets or computes the density matrix of the atom by solving the ROHF for the monatomic molecule.
     * @details The density matrices are stored in the cache for each atom. If the density matrices of an atom are already computed, the density matrices are returned from the cache.
     */
    std::pair<const double*, const double*> get_or_compute_density_matrix(const int atomic_number, const Molecular& monatomic_molecule){
        if(cache.find(atomic_number) != cache.end()){
            const auto& [density_matrix_alpha, density_matrix_beta] = cache[atomic_number];
            return {density_matrix_alpha.data(), density_matrix_beta.data()};
        }

        std::cout << "------ [SAD] Computing density matrix for : " << atomic_number_to_element_name(atomic_number) << " ------" << std::endl;

        ParameterManager parameters;
        parameters.set_default_values_to_unspecfied_parameters();
        parameters["gbsfilename"] = hf_.get_gbsfilename();
        parameters["initial_guess"] = "core"; // if "SAD" is used, the initial guess may be recursively called
        parameters["eri_method"] = "stored"; // use stored ERI method for the monatomic molecule
        
        ROHF atom_rohf(monatomic_molecule, parameters);

        atom_rohf.solve();

        std::vector<double> atom_density_matrix_alpha(atom_rohf.get_num_basis() * atom_rohf.get_num_basis());
        std::vector<double> atom_density_matrix_beta(atom_rohf.get_num_basis() * atom_rohf.get_num_basis());

        atom_rohf.export_density_matrix(atom_density_matrix_alpha.data(), atom_density_matrix_beta.data(), atom_rohf.get_num_basis());

        // cache the density matrices of the atom
        cache[atomic_number] = { std::move(atom_density_matrix_alpha), std::move(atom_density_matrix_beta) };

        // return the density matrices of the atom
        const auto& [density_matrix_alpha, density_matrix_beta] = cache[atomic_number];
        return {density_matrix_alpha.data(), density_matrix_beta.data()};

    }


    void guess() override {
        // allocate and initialize the density matrices of alpha and beta spins
        std::unique_ptr<real_t[]> density_matrix_closed(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        std::unique_ptr<real_t[]> density_matrix_open(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        memset(density_matrix_closed.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));
        memset(density_matrix_open.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));


        // solve ROHF for each atom to get the density matrix
        for(int i=0; i<hf_.get_atoms().size(); i++){
            const std::string element_name = atomic_number_to_element_name(hf_.get_atoms()[i].atomic_number);

            std::cout << " [SAD] Loading density matrix for : " << element_name  << std::endl;

            int atom_num_basis;
            auto [atom_density_matrix_alpha, atom_density_matrix_beta] = read_density_from_sad(element_name, hf_.get_gbsfilename(), atom_num_basis);

            // copy the density matrix of the atom to the density matrix of the molecule in the corresponding diagonal block
            for(size_t p=0; p < atom_num_basis; p++){
                for(size_t q = 0; q < atom_num_basis; q++){
                    size_t p_molecule = hf_.get_atom_to_basis_range()[i].start_index + p;
                    size_t q_molecule = hf_.get_atom_to_basis_range()[i].start_index + q;
                    density_matrix_closed[p_molecule * hf_.get_num_basis() + q_molecule] = 2.0 * atom_density_matrix_beta [p * atom_num_basis + q];
                    density_matrix_open[p_molecule * hf_.get_num_basis() + q_molecule]  = atom_density_matrix_alpha[p * atom_num_basis + q] - atom_density_matrix_beta [p * atom_num_basis + q];
                }
            }
        }
        cudaMemcpy(hf_.get_density_matrix_closed().device_ptr(), density_matrix_closed.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(hf_.get_density_matrix_open().device_ptr(), density_matrix_open.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

        // Since the above Fock matrix is not correct (the density matrix is not correct), the coefficient matrix is computed from the Fock matrix
        hf_.compute_coefficient_matrix(); // compute the coefficient matrix from the density matrix
        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

    }
private:
    std::unordered_map<int, std::pair<std::vector<double>,std::vector<double>>> cache; ///< Cache for the density matricies (alpha- and beta-spins) of each atom
};



/**
 * @brief ERI_Stored_ROHF class for the stored ERIs of the restricted open-shell HF method
 * @details This class computes the ERIs and stores them in the device memory.
 * @details The size of ERI should be reduced to \f$ {1 \over 8} \f$ using the symmetry.
 */

class ERI_Stored_ROHF : public ERI_Stored{
public:
    ERI_Stored_ROHF(ROHF& rohf): 
        ERI_Stored(rohf),
        rohf_(rohf){}   ///< Constructor

    ERI_Stored_ROHF(const ERI_Stored_ROHF&) = delete; ///< copy constructor is deleted
    ~ERI_Stored_ROHF() = default; ///< destructor

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix_closed = rohf_.get_density_matrix_closed();
        const DeviceHostMatrix<real_t>& density_matrix_open = rohf_.get_density_matrix_open();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rohf_.get_core_hamiltonian_matrix();
        const DeviceHostMatrix<real_t>& coefficient_matrix = rohf_.get_coefficient_matrix();
        const DeviceHostMatrix<real_t>& overlap_matrix = rohf_.get_overlap_matrix();
        const ROHF_ParameterSet& ROHF_parameters = rohf_.get_ROHF_parameters();
        DeviceHostMatrix<real_t>& fock_matrix_closed = rohf_.get_fock_matrix_closed();
        DeviceHostMatrix<real_t>& fock_matrix_open = rohf_.get_fock_matrix_open();
        DeviceHostMatrix<real_t>& fock_matrix = rohf_.get_fock_matrix();
        const int num_closed = rohf_.get_num_closed();
        const int num_open = rohf_.get_num_open();
        const int verbose = rohf_.get_verbose();

        gpu::computeFockMatrix_ROHF(
            density_matrix_closed.device_ptr(),
            density_matrix_open.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            coefficient_matrix.device_ptr(),
            overlap_matrix.device_ptr(),
            eri_matrix_.device_ptr(),
            ROHF_parameters,
            fock_matrix_closed.device_ptr(),
            fock_matrix_open.device_ptr(),
            fock_matrix.device_ptr(),
            num_closed,
            num_open,
            num_basis_
        );

        if(verbose){
            std::cout << "Fock matrix (closed):" << std::endl;
            fock_matrix_closed.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_closed(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix (open):" << std::endl;
            fock_matrix_open.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_open(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix (unified):" << std::endl;
            fock_matrix.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    ROHF& rohf_; ///< ROHF
};



class ERI_RI_ROHF : public ERI_RI {
public:
    ERI_RI_ROHF(ROHF& rohf, const Molecular& auxiliary_molecular): ERI_RI(rohf, auxiliary_molecular), rohf_(rohf) {} ///< Constructor
    ERI_RI_ROHF(const ERI_RI_ROHF&) = delete; ///< copy constructor is deleted
    ~ERI_RI_ROHF() = default; ///< destructor
    
    
    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix_closed = rohf_.get_density_matrix_closed();
        const DeviceHostMatrix<real_t>& density_matrix_open = rohf_.get_density_matrix_open();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rohf_.get_core_hamiltonian_matrix();
        const DeviceHostMatrix<real_t>& coefficient_matrix = rohf_.get_coefficient_matrix();
        const DeviceHostMatrix<real_t>& overlap_matrix = rohf_.get_overlap_matrix();
        const ROHF_ParameterSet& ROHF_parameters = rohf_.get_ROHF_parameters();
        DeviceHostMatrix<real_t>& fock_matrix_closed = rohf_.get_fock_matrix_closed();
        DeviceHostMatrix<real_t>& fock_matrix_open = rohf_.get_fock_matrix_open();
        DeviceHostMatrix<real_t>& fock_matrix = rohf_.get_fock_matrix();
        const int num_closed = rohf_.get_num_closed();
        const int num_open = rohf_.get_num_open();
        const int verbose = rohf_.get_verbose();

        gpu::computeFockMatrix_RI_ROHF(
            density_matrix_closed.device_ptr(),
            density_matrix_open.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            coefficient_matrix.device_ptr(),
            overlap_matrix.device_ptr(),
            intermediate_matrix_B_.device_ptr(), 
            ROHF_parameters,
            fock_matrix_closed.device_ptr(),
            fock_matrix_open.device_ptr(),
            fock_matrix.device_ptr(),
            num_closed, 
            num_open, 
            num_basis_, 
            num_auxiliary_basis_
        );


        if(verbose){
            std::cout << "Fock matrix (closed):" << std::endl;
            fock_matrix_closed.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_closed(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix (open):" << std::endl;
            fock_matrix_open.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_open(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix:" << std::endl;
            fock_matrix.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    ROHF& rohf_; ///< ROHF
};
    
} // namespace gansu