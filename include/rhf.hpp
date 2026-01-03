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
 * @file rhf.hpp This file contains the definition of the RHF class.
 */


#pragma once

#include "hf.hpp"
#include "rohf.hpp" // for SAD
#include <memory> // std::unique_ptr
#include "profiler.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION





namespace gansu{





// prototype of classes
class Convergence_RHF;
class Convergence_RHF_Damping;
class Convergence_RHF_DIIS;

/**
 * @brief RHF class
 * 
 */
class RHF : public HF {
public:

    RHF(const Molecular& molecular, const ParameterManager& parameters);
    RHF(const Molecular& molecular): RHF(molecular, ParameterManager()){} ///< Constructor with default parameters

    RHF(const RHF&) = delete; ///< copy constructor is deleted
    ~RHF() = default; ///< destructor

    void precompute_eri_matrix() override;
    void compute_fock_matrix() override;
    void compute_density_matrix() override;
    void guess_initial_fock_matrix(const real_t* density_matrix_a=nullptr, const real_t* density_matrix_b=nullptr, bool force_density=false) override;
    void compute_coefficient_matrix() override;
    void compute_energy() override;
    void update_fock_matrix() override;

    real_t get_energy() const override { return energy_; }
    real_t get_total_spin() override { return 0.0; } // always 0 for RHF

    void report() override;

    //suzuki
    //void compute_RI_RMP2();

    void set_convergence_method(std::unique_ptr<Convergence_RHF> convergence_method);

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
     * @brief Get the reference to the density matrix
     * @return Reference to the density matrix
     */
    DeviceHostMatrix<real_t>& get_density_matrix() { return density_matrix; }

    /**
     * @brief Get the reference to the Fock matrix
     * @return Reference to the Fock matrix
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

    /**
     * @brief Post process after SCF convergence
     * @details This function performs post-HF calculations after the SCF convergence, in which the selected post-HF method is applied.
     * @details This function overrides the virtual function in the base class HF.
     */
    void post_process_after_scf() override;

    /**
     * @brief Get the orbital energies
     * @return Reference to the orbital energies
     */
    DeviceHostMemory<real_t>& get_orbital_energies() { return orbital_energies; } ///< Get the orbital energies

private:
    real_t energy_; ///< Energy
    DeviceHostMemory<real_t> orbital_energies; ///< Orbital energies

    DeviceHostMatrix<real_t> coefficient_matrix; ///< Coefficient matrix
    DeviceHostMatrix<real_t> density_matrix; ///< Density matrix
    DeviceHostMatrix<real_t> fock_matrix; ///< Fock matrix

    std::unique_ptr<Convergence_RHF> convergence_method_; ///< Convergence_RHF

    const std::string initail_guess_method_; ///< Initial guess method name
    const std::string gbsfilename_; ///< Basis set file name (Gaussian basis set file)

};



/**
 * @brief Convergence_RHF class for a convergence algoritm of the restricted HF method
 * @details This class is a virtual class for a convergence algorithm to update the Fock matrix of the restricted HF method.
 * @details This class will be derived to implement the convergence algorithm.
 */
class Convergence_RHF {
public:
    /**
     * @brief Constructor of the UpdateFockMatrix_RHF class
     * @param hf RHF
     * @details This function constructs the UpdateFockMatrix_RHF class.
     * @details The RHF is given as an argument.
     */
    Convergence_RHF(RHF& hf) 
    : hf_(hf),
      verbose(hf.get_verbose()){}

    Convergence_RHF(const Convergence_RHF&) = delete; ///< copy constructor is deleted
    virtual ~Convergence_RHF() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix.
     * @details This function is a pure virtual function to be implemented in the derived classes.
     */
    virtual void get_new_fock_matrix() = 0;

    /**
     * @brief Get the algorithm name
     * @return Algorithm name as a string
     */
    virtual std::string get_algorithm_name() const = 0;

protected:
    RHF& hf_; ///< RHF
    const bool verbose; ///< Verbose mode
};

/**
 * @brief Convergence_RHF_Damping class for the restricted HF method
 * @details This class performs the damping @cite Zerner1979 of the Fock matrix for the restricted HF method.
 * @details The damping factor can be constant or optimized.
 * @details The constant damping factor is given as an argument.
 * @details If no constant damping factor is given, the optimal damping factor is obtained by the optial damping algorithm @cite Cances2000.
 */
class Convergence_RHF_Damping : public Convergence_RHF {
public:
    /**
     * @brief Constructor of the Convergence_RHF_Damping class with constant damping factor
     * @param hf RHF
     * @param damping_factor Damping factor
     * @details This function constructs the UpdateFockMatrix_RHF_Damping class.
     * @details The RHF and the damping factor are given as arguments.
     */
    Convergence_RHF_Damping(RHF& hf, const real_t damping_factor) 
        : Convergence_RHF(hf), 
        damping_factor_(damping_factor),
        use_optimized_(false),
        first_iteration_(true),
        prev_density_matrix(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix(hf.get_num_basis(), hf.get_num_basis()) {}
    
    /**
     * @brief Constructor of the Convergence_RHF_Damping class with optimized damping factor
     * @param hf RHF
     * @details This function constructs the Convergence_RHF_Damping class.
     * @details The RHF is given as an argument.
     */
    Convergence_RHF_Damping(RHF& hf) 
        : Convergence_RHF(hf), 
        damping_factor_(0.0),
        use_optimized_(true),
        first_iteration_(true),
        prev_density_matrix(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix(hf.get_num_basis(), hf.get_num_basis()) {}

    Convergence_RHF_Damping(const Convergence_RHF_Damping&) = delete; ///< copy constructor is deleted
    ~Convergence_RHF_Damping() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix with damping.
     */
    void get_new_fock_matrix() override {
        if (first_iteration_) { // First iteration: no damping, just store the density matrix and the Fock matrix
            first_iteration_ = false;
            cudaMemcpy(prev_density_matrix.device_ptr(), hf_.get_density_matrix().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(prev_fock_matrix.device_ptr(),    hf_.get_fock_matrix().device_ptr(),    hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            return;
        }else{
            if (use_optimized_) { // Optimized damping factor
                const real_t factor = gpu::computeOptimalDampingFactor_RHF(hf_.get_fock_matrix().device_ptr(), prev_fock_matrix.device_ptr(),hf_.get_density_matrix().device_ptr(), prev_density_matrix.device_ptr(), hf_.get_num_basis());
                if(verbose) std::cout << "Damping factor (optimal): " << factor << std::endl;
                // Damping (after dammping, store the density matrix and the Fock matrix to the previous density matrix and the previous Fock matrix, respectively)
                gpu::damping(prev_fock_matrix.device_ptr(), hf_.get_fock_matrix().device_ptr(), factor, hf_.get_num_basis());
                gpu::damping(prev_density_matrix.device_ptr(), hf_.get_density_matrix().device_ptr(), factor, hf_.get_num_basis());
            }else{
                const real_t factor = damping_factor_;
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
    bool use_optimized_; ///< Optimized damping factor
    bool first_iteration_; ///< First iteration

    DeviceHostMatrix<real_t> prev_density_matrix; ///< Previous density matrix
    DeviceHostMatrix<real_t> prev_fock_matrix; ///< Previous Fock matrix
};




/**
 * @brief Convergence_RHF_DIIS class for the restricted HF method
 * @details This class performs the update the Fock matrix for the restricted HF method using DIIS @cite Pulay1980, @cite Pulay1982.
 * @details The number num_prev of the previous Fock matrices to be stored is given as an argument.
 */
class Convergence_RHF_DIIS : public Convergence_RHF {
public:
    /**
     * @brief Constructor of the Convergence_RHF_DIIS class
     * @param hf RHF
     * @param num_prev The number of the previous Fock matrices to be stored
     * @param is_include_transform Include the transformation matrix in the error matrix
     * @details This function constructs the Convergence_RHF_DIIS class.
     */
    Convergence_RHF_DIIS(RHF& hf, const real_t num_prev=10, const bool is_include_transform=false) 
        : Convergence_RHF(hf), 
        num_prev_(num_prev),
        is_include_transform_(is_include_transform),
        num_basis_(hf.get_num_basis()),
        iteration_(0),
        error_matrix(hf_.get_num_basis(),hf_.get_num_basis()),
        prev_error_matrices(num_prev * num_basis_ * num_basis_),
        prev_fock_matrices(num_prev * num_basis_ * num_basis_){}

    Convergence_RHF_DIIS(const Convergence_RHF_DIIS&) = delete; ///< copy constructor is deleted
    ~Convergence_RHF_DIIS() = default; ///< destructor

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
        std::string name = "DIIS ";
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
    const bool is_include_transform_; ///< Include the transformation matrix in the error matrix

    DeviceHostMatrix<real_t> error_matrix; ///< Error matrix

    DeviceHostMemory<real_t> prev_error_matrices; ///< Previous error matrices
    DeviceHostMemory<real_t> prev_fock_matrices; ///< Previous Fock matrices
};

/**
 * @brief InitialGuess_RHF class for the restricted HF method
 * @details This class is a virtual class for the initial guess of the restricted HF method.
 * @details This class will be derived to implement the initial guess.
 * @details The initial guess is used for the initial Fock matrix.
 */
class InitialGuess_RHF {
public:
    InitialGuess_RHF(RHF& hf) : hf_(hf) {}
    InitialGuess_RHF(const InitialGuess_RHF&) = delete;
    virtual ~InitialGuess_RHF() = default;

    virtual void guess() = 0;

protected:
    RHF& hf_;
};

/**
 * @brief InitialGuess_RHF_Core class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the core Hamiltonian matrix.
 */
class InitialGuess_RHF_Core : public InitialGuess_RHF {
public:
    InitialGuess_RHF_Core(RHF& hf) : InitialGuess_RHF(hf) {}
    InitialGuess_RHF_Core(const InitialGuess_RHF_Core&) = delete;
    ~InitialGuess_RHF_Core() = default;

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
 * @brief InitialGuess_RHF_GWH class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Generalized Wolfsberg-Helmholtz method (GWH) @cite Wolfsberg1952.
 */
class InitialGuess_RHF_GWH : public InitialGuess_RHF {
public:
    InitialGuess_RHF_GWH(RHF& hf) : InitialGuess_RHF(hf) {}
    InitialGuess_RHF_GWH(const InitialGuess_RHF_GWH&) = delete;
    ~InitialGuess_RHF_GWH() = default;

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
 * @brief InitialGuess_RHF_Density class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial Fock matrix is computed from the density matrix given as an argument.
 */
class InitialGuess_RHF_Density : public InitialGuess_RHF {
public:
    InitialGuess_RHF_Density(RHF& hf, const real_t* density_matrix_a, const real_t* density_matrix_b) : InitialGuess_RHF(hf), density_matrix_a_(density_matrix_a), density_matrix_b_(density_matrix_b) {
        if(density_matrix_a_ == nullptr || density_matrix_b_ == nullptr){
            THROW_EXCEPTION("density_matrix is nullptr");
        }
    }
    InitialGuess_RHF_Density(const InitialGuess_RHF_Density&) = delete;
    ~InitialGuess_RHF_Density() = default;

    void guess() override {
        // initial guess from the density matrix given as an argument
        std::unique_ptr<real_t[]> density_matrix(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);

        for(size_t i=0; i<hf_.get_num_basis() * hf_.get_num_basis(); i++){
            density_matrix[i] = density_matrix_a_[i] + density_matrix_b_[i];
        }

        cudaMemcpy(hf_.get_density_matrix().device_ptr(), density_matrix.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

    }
private:
    const real_t* density_matrix_a_;
    const real_t* density_matrix_b_;
};


/**
 * @brief InitialGuess_RHF_SAD class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Superposition of Atomic Densities (SAD) @cite Lenthe2006.
 */
class InitialGuess_RHF_SAD : public InitialGuess_RHF {
public:
    InitialGuess_RHF_SAD(RHF& hf) : InitialGuess_RHF(hf) {}
    InitialGuess_RHF_SAD(const InitialGuess_RHF_SAD&) = delete;
    ~InitialGuess_RHF_SAD() = default;

    /**
     * @brief Get or compute the density matrix of the atom by solving the ROHF for the monatomic molecule
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
//        parameters["verbose"] = 1; // debug mode for the monatomic molecules

//        std::cout << "------------" << hf_.get_gbsfilename() << "------------" << std::endl;

        ROHF atom_rohf(monatomic_molecule, parameters);

        atom_rohf.solve();

        atom_rohf.report(); // report the results of the ROHF calculation

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
        std::unique_ptr<real_t[]> density_matrix(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        memset(density_matrix.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));


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
                    density_matrix[p_molecule * hf_.get_num_basis() + q_molecule] = atom_density_matrix_alpha[p * atom_num_basis + q] + atom_density_matrix_beta [p * atom_num_basis + q];
                }
            }
        }
        

        cudaMemcpy(hf_.get_density_matrix().device_ptr(), density_matrix.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
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
 * @brief ERI_Stored_RHF class for the stored ERIs of the restricted HF method
 * @details This class computes the ERIs and stores them in the device memory.
 * @details The size of ERI should be reduced to \f$ {1 \over 8} \f$ using the symmetry.
 */
class ERI_Stored_RHF : public ERI_Stored{
public:
    ERI_Stored_RHF(RHF& rhf): 
        ERI_Stored(rhf),
        rhf_(rhf){} ///< Constructor

    ERI_Stored_RHF(const ERI_Stored_RHF&) = delete; ///< copy constructor is deleted
    ~ERI_Stored_RHF() = default; ///< destructor

    real_t compute_mp2_energy() override;
    real_t compute_mp3_energy() override;
    real_t compute_ccsd_energy() override;
    real_t compute_ccsd_t_energy() override;


    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const int verbose = rhf_.get_verbose();

        gpu::computeFockMatrix_RHF(
            density_matrix.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            eri_matrix_.device_ptr(),
            fock_matrix.device_ptr(),
            num_basis_
        );

        if(verbose){
            // copy the fock matrix to the host memory
            fock_matrix.toHost();
            std::cout << "Fock matrix:" << std::endl;
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    RHF& rhf_; ///< RHF
};



/**
 * @brief ERI_RI_RHF class for the RI approximation of the ERIs of the restricted HF method
 * @details This class computes the ERIs using the Resolution of Identity (RI) approximation.
 */
class ERI_RI_RHF : public ERI_RI {
public:
    ERI_RI_RHF(RHF& rhf, const Molecular& auxiliary_molecular): ERI_RI(rhf, auxiliary_molecular), rhf_(rhf) {} ///< Constructor
    ERI_RI_RHF(const ERI_RI_RHF&) = delete; ///< copy constructor is deleted
    ~ERI_RI_RHF() = default; ///< destructor

    real_t compute_mp2_energy() override;

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const int verbose = rhf_.get_verbose();

        gpu::computeFockMatrix_RI_RHF(
            density_matrix.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            intermediate_matrix_B_.device_ptr(),
            fock_matrix.device_ptr(),
            num_basis_,
            num_auxiliary_basis_, 
            d_J_.device_ptr(),
            d_K_.device_ptr(),
            d_W_tmp_.device_ptr(),
            d_T_tmp_.device_ptr(),
            d_V_tmp_.device_ptr()
        );

        { // nan check
            fock_matrix.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    if(std::isnan(fock_matrix(i, j))){
                        THROW_EXCEPTION("Fock matrix contains NaN at (" + std::to_string(i) + ", " + std::to_string(j) + ")");
                    }
                }
            }
        }

        if(verbose){
            // copy the fock matrix to the host memory
            fock_matrix.toHost();
            std::cout << "Fock matrix:" << std::endl;
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }



protected:
    RHF& rhf_; ///< RHF
};


/**
 * @brief ERI_Direct_RHF class for the direct computation of the ERIs of the restricted HF method
 * @details This class computes the ERIs directly.
 * @details The ERIs are computed by the direct computation of the integrals.
 * @details The Schawarz screening is used to reduce the number of the integrals to be computed.
 */
class ERI_Direct_RHF : public ERI_Direct {
public:
    ERI_Direct_RHF(RHF& rhf): ERI_Direct(rhf), rhf_(rhf) {} ///< Constructor
    ERI_Direct_RHF(const ERI_Direct_RHF&) = delete; ///< copy constructor is deleted
    ~ERI_Direct_RHF() = default; ///< destructor

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
        const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
        const DeviceHostMemory<real_t>& cgto_nomalization_factors = hf_.get_cgto_nomalization_factors();
        const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const real_t schwarz_screening_threshold = rhf_.get_schwarz_screening_threshold();
        const int verbose = rhf_.get_verbose();

        gpu::computeFockMatrix_Direct_RHF(
            density_matrix.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            shell_type_infos, 
            shell_pair_type_infos,
            primitive_shells.device_ptr(), 
            primitive_shell_pair_indices.device_ptr(),
            cgto_nomalization_factors.device_ptr(), 
            boys_grid.device_ptr(), 
            schwarz_upper_bound_factors.device_ptr(),
            schwarz_screening_threshold,
            fock_matrix.device_ptr(),
            num_basis_,
            global_counters_,
            min_skipped_columns_,
            fock_matrix_replicas_,
            num_fock_replicas_,
            verbose
        );

        if(verbose){
            // copy the fock matrix to the host memory
            fock_matrix.toHost();
            std::cout << "Fock matrix:" << std::endl;
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    RHF& rhf_; ///< RHF
};



class ERI_Hash_RHF : public ERI_Hash {
public:
    ERI_Hash_RHF(RHF& rhf): ERI_Hash(rhf), rhf_(rhf) {} ///< Constructor
    ERI_Hash_RHF(const ERI_Hash_RHF&) = delete; ///< copy constructor is deleted
    ~ERI_Hash_RHF() = default; ///< destructor

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const int verbose = rhf_.get_verbose();

        gpu::computeFockMatrix_Hash_RHF(
            density_matrix.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            // Hash memoryへのポインタ
            fock_matrix.device_ptr(),
            num_basis_,
            verbose
        );

        if(verbose){
            // copy the fock matrix to the host memory
            fock_matrix.toHost();
            std::cout << "Fock matrix:" << std::endl;
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    RHF& rhf_; ///< RHF
};



} // namespace gansu