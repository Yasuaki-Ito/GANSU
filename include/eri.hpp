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



#pragma once

#include "hf.hpp"
#include "types.hpp"
#include "device_host_memory.hpp"

#include <cuda_runtime.h> // for int2 type


namespace gansu{

// prototype of classes
class HF;

/**
 * @brief ERI_RHF class for the electron repulsion integrals (ERIs) of the restricted HF method
 * @details This class computes the electron repulsion integrals (ERIs) of the restricted HF method.
 * @details The ERIs are given by \f$ (ij|kl) = \iint \chi_i(\mathbf{r}_1) \chi_j(\mathbf{r}_1) \frac{1}{r_{12}} \chi_k(\mathbf{r}_2) \chi_l(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2 \f$
 * @details This class will be derived to implement the ERI algorithm.
 */
class ERI{
public:

    ERI(){}///< Constructor
    
    ERI(const ERI&) = delete; ///< copy constructor is deleted
    virtual ~ERI() = default; ///< destructor
    
    /**
     * @brief precomputation
     * @details This function is called to initialize the ERI algorithm.
     * @details This function must be implemented in the derived class.
     */
    virtual void precomputation() = 0;

    /**
     * @brief Compute the Fock matrix
     * @details This function must be implemented in the derived class.
     */
    virtual void compute_fock_matrix() = 0;

    /**
     * @brief Get the algorithm name
     * @return Algorithm name as a string
     * @details This function must be implemented in the derived class.
    */
    virtual std::string get_algorithm_name() = 0; ///< Get the algorithm name
};

/**
 * @brief ERI_RHF class for the electron repulsion integrals (ERIs) of the restricted HF method
 * @details This class computes the electron repulsion integrals (ERIs) of the restricted HF method.
 * @details The ERIs are given by \f$ (ij|kl) = \iint \chi_i(\mathbf{r}_1) \chi_j(\mathbf{r}_1) \frac{1}{r_{12}} \chi_k(\mathbf{r}_2) \chi_l(\mathbf{r}_2) d\mathbf{r}_1 d\mathbf{r}_2 \f$
 * @details This class will be derived to implement the ERI algorithm.
 */
class ERI_Stored: public ERI {
public:

    ERI_Stored(const HF& hf); ///< Constructor
    
    ERI_Stored(const ERI_Stored&) = delete; ///< copy constructor is deleted
    virtual ~ERI_Stored() = default; ///< destructor
    
    /**
     * @brief precomputation
     * @details This function is called to initialize the ERI algorithm.
     * @details This function must be implemented in the derived class.
     */
    void precomputation() override;

    std::string get_algorithm_name() override { return "Stored"; } ///< Get the algorithm name

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;
    DeviceHostMatrix<real_t> eri_matrix_;

    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
};


/**
 * @brief ERI_RI class for the electron repulsion integrals (ERIs) using the Resolution of Identity (RI) method
 */
class ERI_RI: public ERI {
public:

    ERI_RI(const HF& hf, const Molecular& auxiliary_molecular); ///< Constructor
    
    ERI_RI(const ERI_RI&) = delete; ///< copy constructor is deleted
    virtual ~ERI_RI() = default; ///< destructor
    
    void precomputation() override;

    DeviceHostMemory<PrimitiveShell>& get_auxiliary_primitive_shells() { return auxiliary_primitive_shells_; } ///< Get the auxiliary primitive shells
    int get_num_auxiliary_basis() { return num_auxiliary_basis_; }

    std::string get_algorithm_name() override { return "RI"; } ///< Get the algorithm name

    //suzuki
    DeviceHostMatrix<real_t>& get_intermediate_matrix_B() { return intermediate_matrix_B_; }

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;
    const int num_auxiliary_basis_;

    const std::vector<ShellTypeInfo> auxiliary_shell_type_infos_; ///< Shell type info in the primitive shell list
    DeviceHostMemory<PrimitiveShell> auxiliary_primitive_shells_; ///< Primitive shells
    DeviceHostMemory<real_t> auxiliary_cgto_nomalization_factors_; ///< Normalization factors of the contracted Gauss functions

    DeviceHostMatrix<real_t> intermediate_matrix_B_; ///< intermediate matrix B (num_auxiliary_basis_ x (num_basis_x num_basis_))

    // Suzuki.
    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
    DeviceHostMemory<real_t> auxiliary_schwarz_upper_bound_factors;
    // DeviceHostMemory<real_t> two_center_eri_;
    // DeviceHostMemory<real_t> three_center_eri_;

    DeviceHostMatrix<real_t> d_J_;
    DeviceHostMatrix<real_t> d_K_;
    DeviceHostMemory<real_t> d_W_tmp_;
    DeviceHostMatrix<real_t> d_T_tmp_;
    DeviceHostMatrix<real_t> d_V_tmp_;
};



/**
 * @brief ERI_Direct class for the electron repulsion integrals (ERIs) using Direct-SCF
 */
class ERI_Direct: public ERI {
public:
    
    ERI_Direct(const HF& hf); ///< Constructor
        
    ERI_Direct(const ERI_Direct&) = delete; ///< copy constructor is deleted
    //virtual ~ERI_Direct() = default; ///< destructor
    virtual ~ERI_Direct();

        
    void precomputation() override;

    std::string get_algorithm_name() override { return "Direct"; } ///< Get the algorithm name
    
protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;

    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
    DeviceHostMemory<int2> primitive_shell_pair_indices;
    std::vector<int*> global_counters_;
    std::vector<int*> min_skipped_columns_;
    real_t* fock_matrix_replicas_;
    const int num_fock_replicas_;
};


/**
 * @brief ERI_Hash class for the electron repulsion integrals (ERIs) using hash memory
 */
class ERI_Hash: public ERI {
public:
    
    ERI_Hash(const HF& hf); ///< Constructor
        
    ERI_Hash(const ERI_Hash&) = delete; ///< copy constructor is deleted
    virtual ~ERI_Hash() = default; ///< destructor
        
    void precomputation() override;

    std::string get_algorithm_name() override { return "Hash"; } ///< Get the algorithm name
    
protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;

    // ここにHash memoryを宣言
    
};






} // namespace gansu