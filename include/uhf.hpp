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
 * @file uhf.hpp This file contains the definition of the UHF class.
 */


#pragma once

#include "hf.hpp"
#include "rohf.hpp"
#include <memory> // std::unique_ptr
#include <random> // std::random_device, std::mt19937, std::uniform_real_distribution

#include "profiler.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu{


// prototype of classes
class Convergence_UHF;
class Convergence_UHF_Damping;
class Convergence_UHF_DIIS;



/**
 * @brief UHF class
 * 
 */
class UHF : public HF {
public:

    UHF(const Molecular& molecular, const ParameterManager& parameters);
    UHF(const Molecular& molecular): UHF(molecular, ParameterManager()){} ///< Constructor with default parameters
 
    UHF(const UHF&) = delete; ///< copy constructor is deleted
    ~UHF() = default; ///< destructor

    void precompute_eri_matrix() override;
    void compute_fock_matrix() override;
    void compute_density_matrix() override;
    void guess_initial_fock_matrix(const real_t* density_matrix_a=nullptr, const real_t* density_matrix_b=nullptr, bool force_density=false) override;
    void compute_coefficient_matrix() override;
    void compute_energy() override;
    void update_fock_matrix() override;

    real_t get_energy() const override { return energy_; }

    void report() override;

    void set_convergence_method(std::unique_ptr<Convergence_UHF> convergence_method);

    void set_eri_method(std::unique_ptr<ERI> eri_method);

    std::vector<real_t> analyze_mulliken_population() const override;

    std::vector<std::vector<real_t>> compute_mayer_bond_order() const override;

    std::vector<std::vector<real_t>> compute_wiberg_bond_order() override;

    real_t get_total_spin() override;;

    /**
     * @brief Get the reference to the coefficient matrix (alpha spin)
     * @return Reference to the coefficient matrix (alpha spin)
     */
    DeviceHostMatrix<real_t>& get_coefficient_matrix_a() { return coefficient_matrix_a; }

    /**
     * @brief Get the reference to the coefficient matrix (beta spin)
     * @return Reference to the coefficient matrix (beta spin)
     */
    DeviceHostMatrix<real_t>& get_coefficient_matrix_b() { return coefficient_matrix_b; }

    /**
     * @brief Get the reference to the density matrix (alpha spin)
     * @return Reference to the density matrix (alpha spin)
     */
    DeviceHostMatrix<real_t>& get_density_matrix_a() { return density_matrix_a; }

    /**
     * @brief Get the reference to the density matrix (beta spin)
     * @return Reference to the density matrix (beta spin)
     */
    DeviceHostMatrix<real_t>& get_density_matrix_b() { return density_matrix_b; }


    /**
     * @brief Get the reference to the Fock matrix (alpha spin)
     * @return Reference to the Fock matrix (alpha spin)
     */
    DeviceHostMatrix<real_t>& get_fock_matrix_a() { return fock_matrix_a; }

    /**
     * @brief Get the reference to the Fock matrix (beta spin)
     * @return Reference to the Fock matrix (beta spin)
     */
    DeviceHostMatrix<real_t>& get_fock_matrix_b() { return fock_matrix_b; }

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

private:
    real_t energy_; ///< Energy
    DeviceHostMemory<real_t> orbital_energies_a; ///< Orbital energies (alpha spin)
    DeviceHostMemory<real_t> orbital_energies_b; ///< Orbital energies (beta spin)

    DeviceHostMatrix<real_t> coefficient_matrix_a; ///< Coefficient matrix (alpha spin)
    DeviceHostMatrix<real_t> coefficient_matrix_b; ///< Coefficient matrix (beta spin)
    DeviceHostMatrix<real_t> density_matrix_a; ///< Density matrix (alpha spin)
    DeviceHostMatrix<real_t> density_matrix_b; ///< Density matrix (beta spin)
    DeviceHostMatrix<real_t> fock_matrix_a; ///< Fock matrix (alpha spin)
    DeviceHostMatrix<real_t> fock_matrix_b; ///< Fock matrix (beta spin)

    std::unique_ptr<Convergence_UHF> convergence_method_; ///< Convergence_UHF

    const std::string initail_guess_method_; ///< Initial guess method name
    const std::string gbsfilename_; ///< Basis set file name (Gaussian basis set file)

};



/**
 * @brief UpdateFockMatrix_UHF class for a convergence algorithm of the unrestricted HF method
 * @details This class is a virtual class for updating the Fock matrix of the unrestricted HF method.
 */
class Convergence_UHF{
public:
    /**
     * @brief Constructor of the Convergence_UHF class
     * @param hf UHF
     * @details This function constructs the Convergence_UHF class.
     * @details The UHF is given as an argument.
     */
    Convergence_UHF(UHF& hf) 
    : hf_(hf),
      verbose(hf.get_verbose()){}

    Convergence_UHF(const Convergence_UHF&) = delete; ///< copy constructor is deleted
    virtual ~Convergence_UHF() = default; ///< destructor

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
    UHF& hf_; ///< UHF
    bool verbose; ///< Verbose mode
};

/**
 * @brief UpdateFockMatrix_UHF_Damping class for the unrestricted HF method
 * @details This class is a virtual class for updating the Fock matrix of the restricted UHF method with damping.
 */
class Convergence_UHF_Damping : public Convergence_UHF {
public:
    /**
     * @brief Constructor of the Convergence_UHF_Damping class with constant damping factor
     * @param hf UHF
     * @param damping_factor Damping factor
     * @details This function constructs the Convergence_UHF_Damping class.
     * @details The UHF and the damping factor are given as arguments.
     */
    Convergence_UHF_Damping(UHF& hf, const real_t damping_factor) 
        : Convergence_UHF(hf), 
        damping_factor_(damping_factor),
        use_optimized_(false),
        first_iteration_(true),
        prev_density_matrix_a(hf.get_num_basis(), hf.get_num_basis()),
        prev_density_matrix_b(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix_a(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix_b(hf.get_num_basis(), hf.get_num_basis()) {}
    
    /**
     * @brief Constructor of the Convergence_UHF_Damping class with optimized damping factor
     * @param hf UHF
     * @details This function constructs the Convergence_UHF_Damping class.
     * @details The UHF is given as an argument.
     */
    Convergence_UHF_Damping(UHF& hf) 
        : Convergence_UHF(hf), 
        damping_factor_(0.0),
//        use_optimized_(true),
        first_iteration_(true),
        prev_density_matrix_a(hf.get_num_basis(), hf.get_num_basis()),
        prev_density_matrix_b(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix_a(hf.get_num_basis(), hf.get_num_basis()),
        prev_fock_matrix_b(hf.get_num_basis(), hf.get_num_basis()) {}

    Convergence_UHF_Damping(const Convergence_UHF_Damping&) = delete; ///< copy constructor is deleted
    ~Convergence_UHF_Damping() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix with damping.
     */
    void get_new_fock_matrix() override {
        if (first_iteration_) { // First iteration: no damping, just store the density matrix and the Fock matrix
            first_iteration_ = false;
            cudaMemcpy(prev_density_matrix_a.device_ptr(), hf_.get_density_matrix_a().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(prev_density_matrix_b.device_ptr(), hf_.get_density_matrix_b().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(prev_fock_matrix_a.device_ptr(), hf_.get_fock_matrix_a().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(prev_fock_matrix_a.device_ptr(), hf_.get_fock_matrix_b().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
            return;
        }else{
            /*
               Optimized damping factor is not implemented yet.
                If implement it, code here
            */
            if(verbose) std::cout << "Damping factor (constant): " << damping_factor_ << std::endl;
            // Damping (after dammping, store the Fock matrix to the previous Fock matrix)
            gpu::damping(prev_fock_matrix_a.device_ptr(), hf_.get_fock_matrix_a().device_ptr(), damping_factor_, hf_.get_num_basis());
            gpu::damping(prev_fock_matrix_b.device_ptr(), hf_.get_fock_matrix_b().device_ptr(), damping_factor_, hf_.get_num_basis());
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

    DeviceHostMatrix<real_t> prev_density_matrix_a; ///< Previous density matrix (alpha spin)
    DeviceHostMatrix<real_t> prev_density_matrix_b; ///< Previous density matrix (beta spin)
    DeviceHostMatrix<real_t> prev_fock_matrix_a; ///< Previous Fock matrix (alpha spin)
    DeviceHostMatrix<real_t> prev_fock_matrix_b; ///< Previous Fock matrix (beta spin)
};



/**
 * @brief Convergence_UHF_DIIS class for the restricted HF method
 * @details This class performs the update the Fock matrix for the restricted UHF method using DIIS @cite Pulay1980, @cite Pulay1982.
 * @details The number num_prev of the previous Fock matrices to be stored is given as an argument.
 * @details This class is a virtual class for updating the Fock matrix of the restricted UHF method with DIIS.
 * @details This class reuses the functions for the UHF method by performing DIIS for the alpha and beta spins separately.
 */
class Convergence_UHF_DIIS : public Convergence_UHF {
public:
    /**
     * @brief Constructor of the Convergence_UHF_DIIS class
     * @param hf UHF
     * @param num_prev The number of the previous Fock matrices to be stored
     * @param is_include_transform Include the transformation matrix in the error matrix
     * @details This function constructs the Convergence_UHF_DIIS class.
     */
    Convergence_UHF_DIIS(UHF& hf, const real_t num_prev=10, const bool is_include_transform=false) 
        : Convergence_UHF(hf), 
        num_prev_(num_prev),
        is_include_transform_(is_include_transform),
        num_basis_(hf.get_num_basis()),
        iteration_(0),
        error_matrix_a_or_b(hf_.get_num_basis(),hf_.get_num_basis()),
        prev_error_matrices_a(num_prev * num_basis_ * num_basis_),
        prev_error_matrices_b(num_prev * num_basis_ * num_basis_),
        prev_fock_matrices_a(num_prev * num_basis_ * num_basis_),
        prev_fock_matrices_b(num_prev * num_basis_ * num_basis_){}

    Convergence_UHF_DIIS(const Convergence_UHF_DIIS&) = delete; ///< copy constructor is deleted
    ~Convergence_UHF_DIIS() = default; ///< destructor

    /**
     * @brief Update the Fock matrix
     * @details This function updates the Fock matrix with damping.
     */
    void get_new_fock_matrix() override {
        { // alpha spin
            // Compute the error matrix
            gpu::computeDIISErrorMatrix(
                hf_.get_overlap_matrix().device_ptr(), 
                hf_.get_transform_matrix().device_ptr(), 
                hf_.get_fock_matrix_a().device_ptr(), 
                hf_.get_density_matrix_a().device_ptr(), 
                error_matrix_a_or_b.device_ptr(), 
                hf_.get_num_basis(),
                is_include_transform_);


            // Copy the previous error matrix and the previous Fock matrix to the new error matrix and the new Fock matrix at most num_prev matrices
            const int store_prev_index = iteration_ % num_prev_; // Overwrite the previous matrices cyclically
            cudaMemcpy(&prev_error_matrices_a.device_ptr()[store_prev_index * num_basis_ * num_basis_], error_matrix_a_or_b.device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&prev_fock_matrices_a.device_ptr()[store_prev_index * num_basis_ * num_basis_], hf_.get_fock_matrix_a().device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);

            // Compute the DIIS coefficients
            const int num_prevs = std::min(iteration_+1, num_prev_);

            if(num_prevs > 1){ // first iteration: no DIIS
                gpu::computeFockMatrixDIIS(
                    prev_error_matrices_a.device_ptr(), 
                    prev_fock_matrices_a.device_ptr(), 
                    hf_.get_fock_matrix_a().device_ptr(), 
                    num_prevs, 
                    num_basis_);
            }
        }
        { // beta spin
            // Compute the error matrix
            gpu::computeDIISErrorMatrix(
                hf_.get_overlap_matrix().device_ptr(), 
                hf_.get_transform_matrix().device_ptr(), 
                hf_.get_fock_matrix_b().device_ptr(), 
                hf_.get_density_matrix_b().device_ptr(), 
                error_matrix_a_or_b.device_ptr(), 
                hf_.get_num_basis(),
                is_include_transform_);


            // Copy the previous error matrix and the previous Fock matrix to the new error matrix and the new Fock matrix at most num_prev matrices
            const int store_prev_index = iteration_ % num_prev_; // Overwrite the previous matrices cyclically
            cudaMemcpy(&prev_error_matrices_b.device_ptr()[store_prev_index * num_basis_ * num_basis_], error_matrix_a_or_b.device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(&prev_fock_matrices_b.device_ptr()[store_prev_index * num_basis_ * num_basis_], hf_.get_fock_matrix_b().device_ptr(), num_basis_ * num_basis_ * sizeof(real_t), cudaMemcpyDeviceToDevice);

            // Compute the DIIS coefficients
            const int num_prevs = std::min(iteration_+1, num_prev_);

            if(num_prevs > 1){ // first iteration: no DIIS
                gpu::computeFockMatrixDIIS(
                    prev_error_matrices_b.device_ptr(), 
                    prev_fock_matrices_b.device_ptr(), 
                    hf_.get_fock_matrix_b().device_ptr(), 
                    num_prevs, 
                    num_basis_);
            }
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
    const bool is_include_transform_; ///< Include the transformation matrix in the error matrix

    DeviceHostMatrix<real_t> error_matrix_a_or_b; ///< Error matrix (alpha or beta spin)

    DeviceHostMemory<real_t> prev_error_matrices_a; ///< Previous error matrices (alpha spin)
    DeviceHostMemory<real_t> prev_error_matrices_b; ///< Previous error matrices (beta spin)
    DeviceHostMemory<real_t> prev_fock_matrices_a; ///< Previous Fock matrices (alpha spin)
    DeviceHostMemory<real_t> prev_fock_matrices_b; ///< Previous Fock matrices (beta spin)
};


/**
 * @brief InitialGuess_UHF class for the unrestricted HF method
 * @details This class is a virtual class for the initial guess of the restricted HF method.
 * @details This class will be derived to implement the initial guess.
 * @details The initial guess is used for the initial Fock matrix.
 */
class InitialGuess_UHF {
public:
    InitialGuess_UHF(UHF& hf) : hf_(hf) {}
    InitialGuess_UHF(const InitialGuess_UHF&) = delete;
    virtual ~InitialGuess_UHF() = default;

    virtual void guess() = 0;

    /**
     * @brief Break the symmetry of the density matrix
     * @details This function breaks the symmetry of the density matrix.
     * @details Copy the density matrix of the alpha spin to the density matrix of the beta spin only for diagonal blocks.
     */
    void break_symmetry(){ 

        hf_.get_coefficient_matrix_b().toHost();
        for(int i=0; i<hf_.get_num_basis(); i++){
            hf_.get_coefficient_matrix_b().host_ptr()[(hf_.get_num_basis()-1) * hf_.get_num_basis() + i] = 0;
            hf_.get_coefficient_matrix_b().host_ptr()[i * hf_.get_num_basis() + hf_.get_num_basis()-1] = 0;
        }
        hf_.get_coefficient_matrix_b().toDevice();

/*
        hf_.get_coefficient_matrix_a().toHost();
        hf_.get_coefficient_matrix_b().toHost();

        std::unique_ptr<real_t[]> homo(new real_t[hf_.get_num_basis()]);
        std::unique_ptr<real_t[]> lumo(new real_t[hf_.get_num_basis()]);

        // alpha-spin
        const double alpha = 0.25;
        const double cos_alpha = std::cos(alpha);
        const double sin_alpha = std::cos(alpha);

        for(int i=0; i<hf_.get_num_basis(); i++){
            homo[i] = hf_.get_coefficient_matrix_a().host_ptr()[(hf_.get_num_alpha_spins()-1) * hf_.get_num_basis() + i];
            lumo[i] = hf_.get_coefficient_matrix_a().host_ptr()[hf_.get_num_alpha_spins() * hf_.get_num_basis() + i];
        }
        
        for(int i=0; i<hf_.get_num_basis(); i++){
            hf_.get_coefficient_matrix_a().host_ptr()[(hf_.get_num_alpha_spins()-1) * hf_.get_num_basis() + i] = cos_alpha * homo[i] + sin_alpha * lumo[i];
            hf_.get_coefficient_matrix_a().host_ptr()[hf_.get_num_alpha_spins() * hf_.get_num_basis() +  i] = sin_alpha * homo[i] + cos_alpha * lumo[i];
        }

        // beta-spin
        const double beta = -alpha;
        const double cos_beta = std::cos(beta);
        const double sin_beta = std::cos(beta);


        for(int i=0; i<hf_.get_num_basis(); i++){
            homo[i] = hf_.get_coefficient_matrix_b().host_ptr()[(hf_.get_num_beta_spins()-1) * hf_.get_num_basis() + i];
            lumo[i] = hf_.get_coefficient_matrix_b().host_ptr()[hf_.get_num_beta_spins() * hf_.get_num_basis() + i];
        }
        for(int i=0; i<hf_.get_num_basis(); i++){
            hf_.get_coefficient_matrix_b().host_ptr()[(hf_.get_num_beta_spins()-1) * hf_.get_num_basis() + i] = cos_beta * homo[i] + sin_beta * lumo[i];
            hf_.get_coefficient_matrix_b().host_ptr()[hf_.get_num_beta_spins() * hf_.get_num_basis() + i] = sin_beta * homo[i] + cos_beta * lumo[i];
        }

        hf_.get_coefficient_matrix_a().toDevice();
        hf_.get_coefficient_matrix_b().toDevice();
*/
    }

protected:
    UHF& hf_;
};

/**
 * @brief InitialGuess_UHF_Core class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the core Hamiltonian matrix.
 */
class InitialGuess_UHF_Core : public InitialGuess_UHF {
public:
    InitialGuess_UHF_Core(UHF& hf) : InitialGuess_UHF(hf) {}
    InitialGuess_UHF_Core(const InitialGuess_UHF_Core&) = delete;
    ~InitialGuess_UHF_Core() = default;

    void guess() override {
        // initial guess for the Fock matrix by the core Hamiltonian matrix
        gpu::computeCoefficientMatrix(
            hf_.get_core_hamiltonian_matrix().device_ptr(), // core Hamiltonian matrix is used instead of the Fock matrix 
            hf_.get_transform_matrix().device_ptr(),
            hf_.get_coefficient_matrix_a().device_ptr(),
            hf_.get_num_basis()
        );

        cudaMemcpy(hf_.get_coefficient_matrix_b().device_ptr(), hf_.get_coefficient_matrix_a().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);
        break_symmetry();
        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
    }
};


/**
 * @brief InitialGuess_UHF_GWH class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Generalized Wolfsberg-Helmholtz method (GWH) @cite Wolfsberg1952.
 */
class InitialGuess_UHF_GWH : public InitialGuess_UHF {
public:
    InitialGuess_UHF_GWH(UHF& hf) : InitialGuess_UHF(hf) {}
    InitialGuess_UHF_GWH(const InitialGuess_UHF_GWH&) = delete;
    ~InitialGuess_UHF_GWH() = default;

    void guess() override {
        // initial guess for the Fock matrix by the core Hamiltonian matrix
        gpu::computeInitialCoefficientMatrix_GWH(
            hf_.get_core_hamiltonian_matrix().device_ptr(),
            hf_.get_overlap_matrix().device_ptr(),
            hf_.get_transform_matrix().device_ptr(),
            hf_.get_coefficient_matrix_a().device_ptr(),
            hf_.get_num_basis()
        );
        cudaMemcpy(hf_.get_coefficient_matrix_b().device_ptr(), hf_.get_coefficient_matrix_a().device_ptr(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyDeviceToDevice);

        break_symmetry();
        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

    }
};


/**
 * @brief InitialGuess_UHF_Density class for the unrestricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial Fock matrix is computed from the density matrix given as an argument.
 */
class InitialGuess_UHF_Density : public InitialGuess_UHF {
public:
    InitialGuess_UHF_Density(UHF& hf, const real_t* density_matrix_a, const real_t* density_matrix_b) : InitialGuess_UHF(hf), density_matrix_a_(density_matrix_a), density_matrix_b_(density_matrix_b) {
        if(density_matrix_a_ == nullptr || density_matrix_b_ == nullptr){
            THROW_EXCEPTION("density_matrix is nullptr");
        }
    }
    InitialGuess_UHF_Density(const InitialGuess_UHF_Density&) = delete;
    ~InitialGuess_UHF_Density() = default;

    void guess() override {
        // initial guess from the density matrix given as an argument
        cudaMemcpy(hf_.get_density_matrix_a().device_ptr(), density_matrix_a_, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(hf_.get_density_matrix_b().device_ptr(), density_matrix_b_, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);


        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
        hf_.compute_coefficient_matrix(); // compute the coefficient matrix from the density matrix
        break_symmetry();
        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
    }
private:
    const real_t* density_matrix_a_;
    const real_t* density_matrix_b_;
};


/**
 * @brief InitialGuess_UHF_SAD class for the restricted HF method
 * @details This class performs the initial guess of the Fock matrix for the restricted HF method.
 * @details The initial guess is the Superposition of Atomic Densities (SAD) @cite Lenthe2006.
 */
class InitialGuess_UHF_SAD : public InitialGuess_UHF {
public:
    InitialGuess_UHF_SAD(UHF& hf) : InitialGuess_UHF(hf) {}
    InitialGuess_UHF_SAD(const InitialGuess_UHF_SAD&) = delete;
    ~InitialGuess_UHF_SAD() = default;

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


/*
    void guess() override {
        // allocate and initialize the density matrices of alpha and beta spins
        std::unique_ptr<real_t[]> density_matrix_alpha(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        std::unique_ptr<real_t[]> density_matrix_beta(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        memset(density_matrix_alpha.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));
        memset(density_matrix_beta.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));


        // solve ROHF for each atom to get the density matrix
        for(int i=0; i<hf_.get_atoms().size(); i++){
            const std::vector<Atom> monatomic_atom = { {hf_.get_atoms()[i].atomic_number, {0.0, 0.0, 0.0}} };
//            BasisSet basis_set = BasisSet::construct_from_gbs(hf_.get_gbsfilename());
            const Molecular monatomic_molecule(monatomic_atom, hf_.get_gbsfilename());

            auto [atom_density_matrix_alpha, atom_density_matrix_beta] = get_or_compute_density_matrix(hf_.get_atoms()[i].atomic_number, monatomic_molecule);

            // copy the density matrix of the atom to the density matrix of the molecule in the corresponding diagonal block
            for(size_t p=0; p < monatomic_molecule.get_num_basis(); p++){
                for(size_t q = 0; q < monatomic_molecule.get_num_basis(); q++){
                    size_t p_molecule = hf_.get_atom_to_basis_range()[i].start_index + p;
                    size_t q_molecule = hf_.get_atom_to_basis_range()[i].start_index + q;
                    density_matrix_alpha[p_molecule * hf_.get_num_basis() + q_molecule] = atom_density_matrix_alpha[p * monatomic_molecule.get_num_basis() + q];
                    density_matrix_beta[p_molecule * hf_.get_num_basis() + q_molecule]  = atom_density_matrix_beta [p * monatomic_molecule.get_num_basis() + q];
                }
            }
        }
        cudaMemcpy(hf_.get_density_matrix_a().device_ptr(), density_matrix_alpha.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(hf_.get_density_matrix_b().device_ptr(), density_matrix_beta.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);

        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
        hf_.compute_coefficient_matrix(); // compute the coefficient matrix from the density matrix
        
        break_symmetry();

        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix

    }
*/

    void guess() override {
        // allocate and initialize the density matrices of alpha and beta spins
        std::unique_ptr<real_t[]> density_matrix_alpha(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        std::unique_ptr<real_t[]> density_matrix_beta(new real_t[hf_.get_num_basis() * hf_.get_num_basis()]);
        memset(density_matrix_alpha.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));
        memset(density_matrix_beta.get(), 0, hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t));


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
                    density_matrix_alpha[p_molecule * hf_.get_num_basis() + q_molecule] = atom_density_matrix_alpha[p * atom_num_basis + q];
                    density_matrix_beta[p_molecule * hf_.get_num_basis() + q_molecule]  = atom_density_matrix_beta [p * atom_num_basis + q];                }
            }
        }

        cudaMemcpy(hf_.get_density_matrix_a().device_ptr(), density_matrix_alpha.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(hf_.get_density_matrix_b().device_ptr(), density_matrix_beta.get(), hf_.get_num_basis() * hf_.get_num_basis() * sizeof(real_t), cudaMemcpyHostToDevice);

        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
        hf_.compute_coefficient_matrix(); // compute the coefficient matrix from the density matrix
        
        break_symmetry();

        hf_.compute_density_matrix(); // compute the density matrix from the coefficient matrix
        hf_.compute_fock_matrix(); // compute the Fock matrix from the density matrix
    }

private:
    std::unordered_map<int, std::pair<std::vector<double>,std::vector<double>>> cache; ///< Cache for the density matricies (alpha- and beta-spins) of each atom
};




/**
 * @brief ERI_Stored_UHF class for the stored ERIs of the unrestricted HF method
 * @details This class computes the ERIs and stores them in the device memory.
 * @details The size of ERI should be reduced to \f$ {1 \over 8} \f$ using the symmetry.
 */

class ERI_Stored_UHF : public ERI_Stored{
public:
    ERI_Stored_UHF(UHF& uhf): 
        ERI_Stored(uhf),
        uhf_(uhf){} ///< Constructor

    ERI_Stored_UHF(const ERI_Stored_UHF&) = delete; ///< copy constructor is deleted
    ~ERI_Stored_UHF() = default; ///< destructor

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix_a = uhf_.get_density_matrix_a();
        const DeviceHostMatrix<real_t>& density_matrix_b = uhf_.get_density_matrix_b();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = uhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix_a = uhf_.get_fock_matrix_a();
        DeviceHostMatrix<real_t>& fock_matrix_b = uhf_.get_fock_matrix_b();
        const int verbose = uhf_.get_verbose();

        gpu::computeFockMatrix_UHF(
            density_matrix_a.device_ptr(),
            density_matrix_b.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            eri_matrix_.device_ptr(),
            fock_matrix_a.device_ptr(),
            fock_matrix_b.device_ptr(),
            num_basis_
        );

        if(verbose){
            std::cout << "Fock matrix (alpha spin):" << std::endl;
            fock_matrix_a.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_a(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix (beta spin):" << std::endl;
            fock_matrix_b.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_b(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    UHF& uhf_; ///< UHF
};



class ERI_RI_UHF : public ERI_RI {
public:
    ERI_RI_UHF(UHF& uhf, const Molecular& auxiliary_molecular): ERI_RI(uhf, auxiliary_molecular), uhf_(uhf) {} ///< Constructor
    ERI_RI_UHF(const ERI_RI_UHF&) = delete; ///< copy constructor is deleted
    ~ERI_RI_UHF() = default; ///< destructor


    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix_a = uhf_.get_density_matrix_a();
        const DeviceHostMatrix<real_t>& density_matrix_b = uhf_.get_density_matrix_b();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = uhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix_a = uhf_.get_fock_matrix_a();
        DeviceHostMatrix<real_t>& fock_matrix_b = uhf_.get_fock_matrix_b();
        const int verbose = uhf_.get_verbose();


        gpu::computeFockMatrix_RI_UHF(
            density_matrix_a.device_ptr(),
            density_matrix_b.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            intermediate_matrix_B_.device_ptr(),
            fock_matrix_a.device_ptr(),
            fock_matrix_b.device_ptr(),
            num_basis_,
            num_auxiliary_basis_
        );

        if(verbose){
            std::cout << "Fock matrix (alpha spin):" << std::endl;
            fock_matrix_a.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_a(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Fock matrix (beta spin):" << std::endl;
            fock_matrix_b.toHost();
            for(size_t i=0; i<num_basis_; i++){
                for(size_t j=0; j<num_basis_; j++){
                    std::cout << fock_matrix_b(i, j) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

protected:
    UHF& uhf_; ///< UHF
};


} // namespace gansu