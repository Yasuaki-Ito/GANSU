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
    void compute_coefficient_matrix_impl() override;
    void compute_energy() override;
    void update_fock_matrix() override;
    void reset_convergence() override;
    std::vector<double> compute_Energy_Gradient() override;
    std::vector<double> compute_Energy_Hessian() override;

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

    const std::string initial_guess_method_; ///< Initial guess method name
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

    virtual void reset() {} ///< Reset internal state (e.g., DIIS history) for a new SCF cycle

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

    void reset() override { first_iteration_ = true; }

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

    void reset() override { iteration_ = 0; }

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
 * @brief Second-Order SCF (SOSCF) convergence method for the restricted HF method
 * @details Uses DIIS for initial iterations, then switches to second-order orbital optimization
 *          near convergence for quadratic convergence.
 * @details Reference: P. Pulay, J. Comput. Chem. 3, 556 (1982)
 *
 * Algorithm (SOSCF phase):
 *   1. Transform Fock matrix to MO basis: F_MO = C^T · F · C
 *   2. Extract orbital gradient: g[a][i] = F_MO[a+nocc, i]
 *   3. Compute rotation: θ[a][i] = -g[a][i] / (ε[a+nocc] - ε[i])
 *   4. Update MO coefficients: C_new = C · (I + Θ)
 *   5. Re-orthogonalize (modified Gram-Schmidt in S metric)
 *   6. Build new density matrix and rebuild Fock matrix
 *
 * NOTE: This implementation uses host-side loops for clarity.
 *       Students should optimize the marked sections using cuBLAS/cuSOLVER.
 */
class Convergence_RHF_SOSCF : public Convergence_RHF {
public:
    /**
     * @brief Constructor
     * @param hf RHF object
     * @param diis_size DIIS history size for the initial phase
     * @param switch_threshold Energy difference threshold to switch from DIIS to SOSCF
     */
    Convergence_RHF_SOSCF(RHF& hf, size_t diis_size = 8, real_t switch_threshold = 1e-4)
        : Convergence_RHF(hf)
        , diis_(hf, diis_size)
        , switch_threshold_(switch_threshold)
        , using_soscf_(false)
    {}

    std::string get_algorithm_name() const override { return "SOSCF"; }

    void reset() override {
        diis_.reset();
        using_soscf_ = false;
    }

    void get_new_fock_matrix() override {
        const int nao = hf_.get_num_basis();
        const int nocc = hf_.get_num_electrons() / 2;
        const int nvir = nao - nocc;

        // =======================================================
        //  Phase 1: Use DIIS until close to convergence
        // =======================================================
        if (!using_soscf_) {
            diis_.get_new_fock_matrix();
            real_t dE = hf_.get_energy_difference();
            if (dE < switch_threshold_ && dE > 0) {
                using_soscf_ = true;
                if (verbose) std::cout << "  [SOSCF] Switching from DIIS to SOSCF" << std::endl;
            }
            return;
        }

        // =======================================================
        //  Phase 2: Second-Order SCF (SOSCF)
        // =======================================================

        // --- Copy matrices to host ---
        hf_.get_coefficient_matrix().toHost();
        hf_.get_fock_matrix().toHost();
        hf_.get_orbital_energies().toHost();
        hf_.get_overlap_matrix().toHost();

        const real_t* C   = hf_.get_coefficient_matrix().host_ptr();
        const real_t* F   = hf_.get_fock_matrix().host_ptr();
        const real_t* eps = hf_.get_orbital_energies().host_ptr();
        const real_t* S   = hf_.get_overlap_matrix().host_ptr();

        // -----------------------------------------------
        // Step 1: F_MO = C^T · F · C    (nao × nao)
        // -----------------------------------------------
        // TODO: GPU化 — cublasDgemm を2回使用: tmp=F·C, F_MO=C^T·tmp
        std::vector<real_t> tmp(nao * nao, 0.0);
        std::vector<real_t> F_MO(nao * nao, 0.0);

        // tmp[μ,q] = Σ_ν F[μ,ν] · C[ν,q]
        for (int mu = 0; mu < nao; mu++)
            for (int q = 0; q < nao; q++)
                for (int nu = 0; nu < nao; nu++)
                    tmp[mu * nao + q] += F[mu * nao + nu] * C[nu * nao + q];

        // F_MO[p,q] = Σ_μ C[μ,p] · tmp[μ,q]
        for (int p = 0; p < nao; p++)
            for (int q = 0; q < nao; q++)
                for (int mu = 0; mu < nao; mu++)
                    F_MO[p * nao + q] += C[mu * nao + p] * tmp[mu * nao + q];

        // -----------------------------------------------
        // Step 2: Orbital gradient g[a][i] = F_MO[a+nocc, i]
        //   (The off-diagonal vir-occ block of F_MO)
        //   At convergence, F_MO is diagonal, so g → 0.
        // -----------------------------------------------
        double grad_norm = 0.0;
        for (int a = 0; a < nvir; a++)
            for (int i = 0; i < nocc; i++) {
                double g = F_MO[(nocc + a) * nao + i];
                grad_norm += g * g;
            }
        grad_norm = std::sqrt(grad_norm);

        if (verbose)
            std::cout << "  [SOSCF] Orbital gradient norm = " << grad_norm << std::endl;

        // Fall back to DIIS if the gradient is too large
        if (grad_norm > 0.5) {
            if (verbose) std::cout << "  [SOSCF] Gradient too large, falling back to DIIS" << std::endl;
            diis_.get_new_fock_matrix();
            return;
        }

        // -----------------------------------------------
        // Step 3: Rotation angles (diagonal Hessian approximation)
        //   θ[a][i] = − F_MO[a+nocc, i] / (ε[a+nocc] − ε[i])
        //
        //   This is the Newton step with the approximate orbital Hessian:
        //     H[ai,ai] ≈ ε_a − ε_i    (diagonal approximation)
        // -----------------------------------------------
        // TODO: GPU化 — デバイスカーネルで並列計算
        std::vector<real_t> theta(nvir * nocc, 0.0);
        for (int a = 0; a < nvir; a++)
            for (int i = 0; i < nocc; i++) {
                real_t denom = eps[nocc + a] - eps[i];
                if (std::abs(denom) < 1e-12) denom = 1e-12;
                theta[a * nocc + i] = -F_MO[(nocc + a) * nao + i] / denom;
            }

        // -----------------------------------------------
        // Step 4: Rotate MO coefficients  C_new = C · (I + Θ)
        //   occupied:  C_new[μ,i]      = C[μ,i]      + Σ_a θ[a,i] · C[μ, a+nocc]
        //   virtual:   C_new[μ,a+nocc] = C[μ,a+nocc] − Σ_i θ[a,i] · C[μ, i]
        // -----------------------------------------------
        // TODO: GPU化 — cublasDgemm で一括計算
        //   C_occ_new = C_occ + C_vir · Θ
        //   C_vir_new = C_vir − C_occ · Θ^T
        std::vector<real_t> C_new(nao * nao);
        std::copy(C, C + nao * nao, C_new.begin());

        for (int mu = 0; mu < nao; mu++)
            for (int i = 0; i < nocc; i++)
                for (int a = 0; a < nvir; a++)
                    C_new[mu * nao + i] += theta[a * nocc + i] * C[mu * nao + (nocc + a)];

        for (int mu = 0; mu < nao; mu++)
            for (int a = 0; a < nvir; a++)
                for (int i = 0; i < nocc; i++)
                    C_new[mu * nao + (nocc + a)] -= theta[a * nocc + i] * C[mu * nao + i];

        // -----------------------------------------------
        // Step 5: Re-orthogonalize occupied orbitals
        //   Modified Gram-Schmidt in the overlap (S) metric:
        //     <i|j>_S = C[:,i]^T · S · C[:,j] = δ_{ij}
        // -----------------------------------------------
        // TODO: GPU化 — cublasDgemm で S·C を先に計算し、
        //   cublasDdot でオーバーラップ計算、cublasDaxpy で射影除去
        for (int i = 0; i < nocc; i++) {
            // Project out components of orbitals 0..i-1
            for (int j = 0; j < i; j++) {
                double overlap = 0.0;
                for (int mu = 0; mu < nao; mu++)
                    for (int nu = 0; nu < nao; nu++)
                        overlap += C_new[mu * nao + i] * S[mu * nao + nu] * C_new[nu * nao + j];
                for (int mu = 0; mu < nao; mu++)
                    C_new[mu * nao + i] -= overlap * C_new[mu * nao + j];
            }
            // Normalize
            double norm = 0.0;
            for (int mu = 0; mu < nao; mu++)
                for (int nu = 0; nu < nao; nu++)
                    norm += C_new[mu * nao + i] * S[mu * nao + nu] * C_new[nu * nao + i];
            norm = std::sqrt(norm);
            for (int mu = 0; mu < nao; mu++)
                C_new[mu * nao + i] /= norm;
        }

        // -----------------------------------------------
        // Step 6: Build new density matrix  D = 2 · C_occ · C_occ^T
        // -----------------------------------------------
        // TODO: GPU化 — cublasDsyrk: D = 2·C_occ·C_occ^T
        std::vector<real_t> D_new(nao * nao, 0.0);
        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu < nao; nu++)
                for (int i = 0; i < nocc; i++)
                    D_new[mu * nao + nu] += 2.0 * C_new[mu * nao + i] * C_new[nu * nao + i];

        // -----------------------------------------------
        // Step 7: Upload C and D to device, then rebuild Fock matrix
        // -----------------------------------------------
        cudaMemcpy(hf_.get_coefficient_matrix().device_ptr(), C_new.data(),
                   nao * nao * sizeof(real_t), cudaMemcpyHostToDevice);
        cudaMemcpy(hf_.get_density_matrix().device_ptr(), D_new.data(),
                   nao * nao * sizeof(real_t), cudaMemcpyHostToDevice);

        // Rebuild F from the SOSCF-improved density
        hf_.compute_fock_matrix();
    }

private:
    Convergence_RHF_DIIS diis_;   ///< DIIS for initial phase
    real_t switch_threshold_;      ///< Energy difference threshold for DIIS→SOSCF switch
    bool using_soscf_;             ///< true after switching to SOSCF
};


/**
 * @brief ADIIS/EDIIS convergence methods for the restricted HF method
 * @details Three modes:
 *   - ADIIS: Augmented DIIS only (JCP 132, 054109 (2010))
 *   - EDIIS: Energy DIIS only (JCP 116, 8255 (2002))
 *   - AEDIIS: Automatic EDIIS→ADIIS→DIIS switching based on error norm
 */
enum class ADIISMode { ADIIS, EDIIS, AEDIIS };

class Convergence_RHF_ADIIS : public Convergence_RHF {
public:
    Convergence_RHF_ADIIS(RHF& hf, ADIISMode mode, const int num_prev = 8)
        : Convergence_RHF(hf),
          mode_(mode),
          num_prev_(num_prev),
          num_basis_(hf.get_num_basis()),
          iteration_(0),
          error_matrix(hf.get_num_basis(), hf.get_num_basis()),
          prev_error_matrices(num_prev * hf.get_num_basis() * hf.get_num_basis()),
          prev_fock_matrices(num_prev * hf.get_num_basis() * hf.get_num_basis()),
          prev_density_matrices(num_prev * hf.get_num_basis() * hf.get_num_basis()),
          prev_energies(num_prev, 0.0) {}

    ~Convergence_RHF_ADIIS() = default;
    Convergence_RHF_ADIIS(const Convergence_RHF_ADIIS&) = delete;

    void get_new_fock_matrix() override;
    std::string get_algorithm_name() const override {
        const char* name = (mode_ == ADIISMode::ADIIS) ? "ADIIS" :
                           (mode_ == ADIISMode::EDIIS) ? "EDIIS" : "AEDIIS";
        return std::string(name) + " (size=" + std::to_string(num_prev_) + ")";
    }
    void reset() override { iteration_ = 0; }

private:
    std::vector<real_t> solve_adiis_coefficients(int n, const std::vector<real_t>& df_matrix, int newest) const;
    std::vector<real_t> solve_ediis_coefficients(int n, const std::vector<real_t>& energies,
                                                  const std::vector<real_t>& df_matrix) const;

    const ADIISMode mode_;
    int iteration_;
    const int num_prev_;
    const int num_basis_;

    DeviceHostMatrix<real_t> error_matrix;
    DeviceHostMemory<real_t> prev_error_matrices;
    DeviceHostMemory<real_t> prev_fock_matrices;
    DeviceHostMemory<real_t> prev_density_matrices;
    std::vector<real_t> prev_energies;
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
        hf_.switchHasMatrixC();

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
        hf_.switchHasMatrixC();

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

        if(hf_.get_run_type() != "optimize"){
            std::cout << "------ [SAD] Computing density matrix for : " << atomic_number_to_element_name(atomic_number) << " ------" << std::endl;
        }

        ParameterManager parameters;
        parameters.set_default_values_to_unspecified_parameters();
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

            if(hf_.get_run_type() != "optimize"){
                std::cout << " [SAD] Loading density matrix for : " << element_name  << std::endl;
            }

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
    void compute_mp2_effective_densities(real_t* d_P_eff, real_t* d_W_eff, real_t* d_Gamma_eff, real_t* d_P_2el) override;
    real_t compute_mp3_energy() override;
    real_t compute_mp4_energy() override;
    real_t compute_cc2_energy() override;
    real_t compute_ccsd_energy() override;
    real_t compute_ccsd_t_energy() override;
    real_t compute_fci_energy() override;
    void compute_cis(int n_states) override;
    void compute_adc2(int n_states) override;
    void compute_adc2x(int n_states) override;
    void compute_eom_mp2(int n_states) override;
    void compute_eom_cc2(int n_states) override;
    void compute_eom_ccsd(int n_states) override;

    /// Set CCSD algorithm: 0=spatial-optimized (default), 1=spatial-naive, 2=spin-orbital
    void set_ccsd_algorithm(int algo) { ccsd_algorithm_ = algo; }


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
    int ccsd_algorithm_ = 0; ///< 0=spatial-optimized, 1=spatial-naive, 2=spin-orbital
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
    real_t compute_mp3_energy() override;
    real_t compute_mp4_energy() override;
    real_t compute_cc2_energy() override;
    real_t compute_ccsd_energy() override;
    real_t compute_ccsd_t_energy() override;
    real_t compute_fci_energy() override;
    void compute_cis(int n_states) override;
    void compute_adc2(int n_states) override;
    void compute_adc2x(int n_states) override;
    void compute_eom_mp2(int n_states) override;
    void compute_eom_cc2(int n_states) override;
    void compute_eom_ccsd(int n_states) override;

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const int verbose = rhf_.get_verbose();

        if (rhf_.get_hasMatrixC()) {
            gpu::computeFockMatrix_RI_RHF_with_coefficient_matrix(
                coefficient_matrix.device_ptr(),
                density_matrix.device_ptr(),
                core_hamiltonian_matrix.device_ptr(),
                intermediate_matrix_B_.device_ptr(),
                fock_matrix.device_ptr(),
                num_basis_,
                num_auxiliary_basis_, 
                num_occ_,
                d_J_.device_ptr(),
                d_K_.device_ptr(),
                d_W_tmp_.device_ptr(),
                d_tmp1_.device_ptr(),
                d_tmp2_.device_ptr()
            );
        } else {
            gpu::computeFockMatrix_RI_RHF_with_density_matrix(
                density_matrix.device_ptr(),
                core_hamiltonian_matrix.device_ptr(),
                intermediate_matrix_B_.device_ptr(),
                fock_matrix.device_ptr(),
                num_basis_,
                num_auxiliary_basis_, 
                d_J_.device_ptr(),
                d_K_.device_ptr(),
                d_W_tmp_.device_ptr(),
                d_tmp1_.device_ptr(),
                d_tmp2_.device_ptr()
            );
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

    // Post-HF methods (uses lazy AO ERI reconstruction + build_mo_eri)
    real_t compute_mp2_energy() override;
    real_t compute_mp3_energy() override;
    real_t compute_mp4_energy() override;
    real_t compute_cc2_energy() override;
    real_t compute_ccsd_energy() override;
    real_t compute_ccsd_t_energy() override;
    real_t compute_fci_energy() override;
    void compute_cis(int n_states) override;
    void compute_adc2(int n_states) override;
    void compute_adc2x(int n_states) override;
    void compute_eom_mp2(int n_states) override;
    void compute_eom_cc2(int n_states) override;
    void compute_eom_ccsd(int n_states) override;

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
        const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
        const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
        const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const real_t schwarz_screening_threshold = rhf_.get_schwarz_screening_threshold();
        const int verbose = rhf_.get_verbose();

        //gpu::computeFockMatrix_Direct_RHF(
        //    density_matrix.device_ptr(),
        //    core_hamiltonian_matrix.device_ptr(),
        //    shell_type_infos, 
        //    shell_pair_type_infos,
        //    primitive_shells.device_ptr(), 
        //    primitive_shell_pair_indices.device_ptr(),
        //    cgto_normalization_factors.device_ptr(), 
        //    boys_grid.device_ptr(), 
        //    schwarz_upper_bound_factors.device_ptr(),
        //    schwarz_screening_threshold,
        //    fock_matrix.device_ptr(),
        //    num_basis_,
        //    global_counters_,
        //    min_skipped_columns_,
        //    fock_matrix_replicas_,
        //    num_fock_replicas_,
        //    verbose
        //);
        gpu::computeFockMatrix_Direct_RHF(
            density_matrix.device_ptr(),
            density_matrix_diff_.device_ptr(),
            density_matrix_diff_shell_.device_ptr(),
            core_hamiltonian_matrix.device_ptr(),
            shell_type_infos, 
            shell_pair_type_infos,
            primitive_shells.device_ptr(), 
            primitive_shell_pair_indices.device_ptr(),
            cgto_normalization_factors.device_ptr(), 
            boys_grid.device_ptr(), 
            schwarz_upper_bound_factors.device_ptr(),
            schwarz_screening_threshold,
            fock_matrix.device_ptr(),
            fock_matrix_prev_.device_ptr(),
            num_basis_,
            global_counters_,
            min_skipped_columns_,
            fock_matrix_replicas_,
            num_fock_replicas_,
            verbose,
            is_first_call_
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

        if (hash_fock_method_ == HashFockMethod::Compact) {
            gpu::computeFockMatrix_Hash_RHF(
                density_matrix.device_ptr(),
                core_hamiltonian_matrix.device_ptr(),
                d_coo_keys_, d_coo_values_, num_entries_,
                fock_matrix.device_ptr(), num_basis_, verbose);
        } else if (hash_fock_method_ == HashFockMethod::Indexed) {
            gpu::computeFockMatrix_Hash_Indexed_RHF(
                density_matrix.device_ptr(),
                core_hamiltonian_matrix.device_ptr(),
                d_hash_keys_, d_hash_values_,
                d_nonzero_indices_, num_nonzero_,
                fock_matrix.device_ptr(), num_basis_, verbose);
        } else {
            gpu::computeFockMatrix_Hash_FullScan_RHF(
                density_matrix.device_ptr(),
                core_hamiltonian_matrix.device_ptr(),
                d_hash_keys_, d_hash_values_,
                hash_capacity_mask_ + 1,
                fock_matrix.device_ptr(), num_basis_, verbose);
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




class ERI_RI_Direct_RHF : public ERI_RI_Direct {
public:
    ERI_RI_Direct_RHF(RHF& rhf, const Molecular& auxiliary_molecular)
        : ERI_RI_Direct(rhf, auxiliary_molecular), rhf_(rhf),
          coefficient_matrix_prev(rhf.get_num_basis() * rhf.get_num_basis()) {
        cudaMemset(coefficient_matrix_prev.device_ptr(), 0, sizeof(real_t) * coefficient_matrix_prev.size());
        cudaMallocHost((void**)&h_Z_tensor, sizeof(real_t) * (rhf_.get_num_electrons() / 2) * num_basis_ * num_auxiliary_basis_);
        memset(h_Z_tensor, 0, sizeof(real_t) * (rhf_.get_num_electrons() / 2) * num_basis_ * num_auxiliary_basis_);
    }
    ERI_RI_Direct_RHF(const ERI_RI_Direct_RHF&) = delete;
    ~ERI_RI_Direct_RHF() { if(h_Z_tensor) cudaFreeHost(h_Z_tensor); }

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
        const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
        const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
        const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const real_t schwarz_screening_threshold = rhf_.get_schwarz_screening_threshold();
        const int verbose = rhf_.get_verbose();
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
        const DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
        const int num_electrons = rhf_.get_num_electrons();

        if(rhf_.get_hasMatrixC() == false){
            gpu::computeInitialFockMatrix_RI_Direct_RHF(
                density_matrix.device_ptr(), coefficient_matrix.device_ptr(),
                two_center_eris_inverse.device_ptr(),
                core_hamiltonian_matrix.device_ptr(), fock_matrix.device_ptr(),
                shell_type_infos, shell_pair_type_infos,
                primitive_shells.host_ptr(), primitive_shells.device_ptr(),
                cgto_normalization_factors.device_ptr(),
                auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
                auxiliary_cgto_normalization_factors_.device_ptr(),
                primitive_shell_pair_indices.device_ptr(),
                primitive_shell_pair_indices_for_SAD_K_computation.host_ptr(),
                primitive_shell_pair_indices_for_SAD_K_computation.device_ptr(),
                num_basis_, num_auxiliary_basis_, num_electrons, primitive_shells.size(),
                boys_grid.device_ptr(), schwarz_screening_threshold,
                schwarz_upper_bound_factors.device_ptr(),
                auxiliary_schwarz_upper_bound_factors.device_ptr(),
                verbose, two_center_eris.device_ptr());
        } else {
            gpu::computeFockMatrix_RI_Direct_RHF(
                density_matrix.device_ptr(), coefficient_matrix.device_ptr(),
                two_center_eris_inverse.device_ptr(), two_center_eris.device_ptr(),
                core_hamiltonian_matrix.device_ptr(), fock_matrix.device_ptr(),
                coefficient_matrix_prev.device_ptr(), h_Z_tensor,
                shell_type_infos, shell_pair_type_infos,
                primitive_shells.host_ptr(), primitive_shells.device_ptr(),
                cgto_normalization_factors.device_ptr(),
                auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
                auxiliary_cgto_normalization_factors_.device_ptr(),
                primitive_shell_pair_indices.device_ptr(),
                num_basis_, num_auxiliary_basis_, num_electrons, primitive_shells.size(),
                boys_grid.device_ptr(), schwarz_screening_threshold,
                schwarz_upper_bound_factors.device_ptr(),
                auxiliary_schwarz_upper_bound_factors.device_ptr(),
                verbose);
        }
    }

protected:
    RHF& rhf_;
    DeviceHostMemory<real_t> coefficient_matrix_prev;
    real_t* h_Z_tensor = nullptr;
};


/// Semi-Direct RI: computes 3-center ERIs each iteration into temporary B matrix,
/// then J/K entirely via BLAS.  Faster than Direct RI but uses O(naux×nao²) peak memory.
class ERI_RI_SemiDirect_RHF : public ERI_RI_Direct {
public:
    ERI_RI_SemiDirect_RHF(RHF& rhf, const Molecular& auxiliary_molecular)
        : ERI_RI_Direct(rhf, auxiliary_molecular), rhf_(rhf) {}
    ERI_RI_SemiDirect_RHF(const ERI_RI_SemiDirect_RHF&) = delete;
    ~ERI_RI_SemiDirect_RHF() = default;

    std::string get_algorithm_name() override { return "Semi-Direct-RI"; }

    void compute_fock_matrix() override {
        const DeviceHostMatrix<real_t>& density_matrix = rhf_.get_density_matrix();
        const DeviceHostMatrix<real_t>& core_hamiltonian_matrix = rhf_.get_core_hamiltonian_matrix();
        const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
        const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
        const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
        const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
        DeviceHostMatrix<real_t>& fock_matrix = rhf_.get_fock_matrix();
        const int verbose = rhf_.get_verbose();
        const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
        const DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
        const int num_occ = rhf_.get_num_electrons() / 2;

        gpu::computeFockMatrix_RI_Direct_v2(
            density_matrix.device_ptr(), coefficient_matrix.device_ptr(),
            two_center_eris.device_ptr(), two_center_eris_inverse.device_ptr(),
            core_hamiltonian_matrix.device_ptr(), fock_matrix.device_ptr(),
            shell_type_infos, shell_pair_type_infos,
            primitive_shells.device_ptr(), cgto_normalization_factors.device_ptr(),
            auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
            auxiliary_cgto_normalization_factors_.device_ptr(),
            primitive_shell_pair_indices.device_ptr(),
            num_basis_, num_auxiliary_basis_, num_occ,
            boys_grid.device_ptr(), rhf_.get_schwarz_screening_threshold(),
            schwarz_upper_bound_factors.device_ptr(),
            auxiliary_schwarz_upper_bound_factors.device_ptr(),
            verbose);
    }

protected:
    RHF& rhf_;
};


inline void RHF::reset_convergence() { if(convergence_method_) convergence_method_->reset(); }

} // namespace gansu
