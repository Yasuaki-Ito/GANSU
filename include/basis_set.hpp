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
 * @file basis_set.hpp
 * @brief Read the Gaussian basis set file and construct the basis set.
 * @details It is necessary only to include this header file.
*/

#pragma once

#include <vector>
#include <string>
#include <stdexcept> // runtime_error
#include <iostream> // cout
#include <unordered_map>
#include <utility> // std::pair
#include <cmath> // std::sqrt, std::pow

#include "types.hpp"
#include "utils.hpp" // Exception

namespace gansu{



/**
 * @brief Struct of primitive Gauss function
 * @details This struct represents a primitive Gauss function.
 * @details A primitive Gauss function is defined by the exponent and the coefficient.
 * @details The primitive Gauss function is given by \f$ G(\mathbf{r}) = c \exp(-\alpha \mathbf{r}^2) \f$.
 */
struct PrimitiveGauss {
    real_t exponent;
    real_t coefficient;
};


/**
 * @brief Struct of contracted Gauss function
 * @details This struct represents a contracted Gauss function.
 * @details The contracted Gauss function is given by \f$\chi(\mathbf{r}) = \sum_i c_i G_i(\mathbf{r}) \f$.
 * @details This structure includes the type of the contracted Gauss function (S, P, D,...) and the list of primitive Gauss functions \f$G_1,G_2,\ldots \f$.
 */
class ContractedGauss {
public:
    /**
     * @brief Constructor of the ContractedGauss class
     * @param type Type of the contracted Gauss function
     */
    ContractedGauss(const std::string& type) : type(type) {}

    /**
     * @brief Add a primitive Gauss function to the contracted Gauss function
     */
    void add_primitive_gauss(const PrimitiveGauss& primitive_gauss){
        primitives.push_back(primitive_gauss);
    }

    /**
     * @brief Add a primitive Gauss function to the contracted Gauss function
     */
    void add_primitive_gauss(const real_t exponent, const real_t coefficient){
        PrimitiveGauss primitive_gauss = {exponent, coefficient};
        primitives.push_back(primitive_gauss);
    }

    /**
     * @brief Get the number of primitive Gauss functions
     */
    size_t get_num_primitives() const { return primitives.size(); }

    /**
     * @brief Get the primitive Gauss function
     * @param index Index of the primitive Gauss function
     * @return Primitive Gauss function
     */
    const PrimitiveGauss& get_primitive_gauss(size_t index) const { return primitives[index]; }

    /**
     * @brief Get the type of the contracted Gauss function
     * @return Type of the contracted Gauss function
     */
    const std::string& get_type() const { return type; }


    /**
     * @brief Compute the normalization factor of the contracted Gauss function
     * @return Normalization factor
     * @details The normalization factor is given. The order of the normalization factor is the same as the order of the Cartesian exponents (e.g., p-type: {1,0,0}, {0,1,0}, {0,0,1}) defined in "type.hpp".
     */

    std::vector<real_t> get_normalization_factor() const {
        std::vector<real_t> normalize_factors;
        const std::vector<std::vector<int>>& angular_momentum_list = AngularMomentums[shell_name_to_shell_type(type)];

        for(const auto& angular_momentums : angular_momentum_list){
            real_t normalize_factor = 0.0;
            int lx = angular_momentums[0];
            int ly = angular_momentums[1];
            int lz = angular_momentums[2];
            int L = lx + ly + lz;

            for(const auto& primitive_a : primitives){
                real_t n_a = std::pow(2.0 * M_PI, -3.0/4.0) 
                            * std::pow(4.0 * primitive_a.exponent, L/2.0+0.75) ;

                for(const auto& primitive_b : primitives){
                    real_t n_b = std::pow(2.0 * M_PI, -3.0/4.0) 
                                * std::pow(4.0 * primitive_b.exponent, L/2+0.75) ;

                    normalize_factor += primitive_a.coefficient * n_a
                                    * primitive_b.coefficient * n_b
                                    / std::pow(2.0*(primitive_a.exponent + primitive_b.exponent), L + 1.5);
                }
            }

            normalize_factor *= std::pow(2.0*M_PI, 1.5);
            
            normalize_factor = std::pow(normalize_factor,-0.5);
            normalize_factors.push_back(normalize_factor);
        }
        return normalize_factors;
    }


private:
    std::string type; // S, P, D, ...
    std::vector<PrimitiveGauss> primitives;

    real_t factorial2(int n) const {
        if(n <= 1) return 1;
        return n * factorial2(n-2);
    }

    /**
     * @brief Get the normalization factor of the contracted Gauss function
     * @param alpha Exponent of the contracted Gauss function
     * @param i Exponent in the respective Cartesian direction 
     * @return Normalization factor
     * @details The normalization factor is given by \f$ \left( \frac{2 \alpha}{\pi} \right)^{1/4} \left( \frac{(4 \alpha)^i}{(2i-1)!!} \right)^{1/2} \f$.
     */
    real_t get_normalize_factor_1d(const real_t alpha, const int i) const {
        return std::pow(2.0 * alpha / M_PI, 1.0/4.0) * std::sqrt( std::pow(4.0 * alpha, i) / factorial2(2*i-1) );
    }
};

/**
 * @brief ElementBasisSet class
 * @details This class represents the basis set of an element.
 */
class ElementBasisSet {
public:
    /**
     * @brief Constructor of the ElementBasisSet class
     * @param element_name Name of the element
     */
    ElementBasisSet() {}

    /**
     * @brief Set the name of the element
     */
    void set_element_name(const std::string& element_name){
        this->element_name = element_name;
    }

    /**
     * @brief Add a contracted Gauss function to the basis set
     * @param contracted_gauss Contracted Gauss function
     */
    void add_contracted_gauss(const ContractedGauss& contracted_gauss){
        contracted_gausses.push_back(contracted_gauss);
    }

    /**
     * @brief Get the name of the element
     * @return Name of the element
     */
    const std::string& get_element_name() const { return element_name; }

    /**
     * @brief Get the number of contracted Gauss functions
     * @return Number of contracted Gauss functions
     */
    size_t get_num_contracted_gausses() const { return contracted_gausses.size(); }

    /**
     * @brief Get the contracted Gauss function
     * @param index Index of the contracted Gauss function
     * @return Contracted Gauss function
     */
    const ContractedGauss& get_contracted_gauss(size_t index) const {
        return contracted_gausses[index];
    }

    /**
     * @brief Print the element basis set
     * @param os Output stream
     * @param element_basis_set Element basis set
     * @return Output stream
     * @details This function prints the element basis set.
     * @details An example of output is as follows:
     * @code
     * Element name: H
     * Contracted Gauss[0] Type: S, Number of primitives: 3
     *    Primitive Gauss[0]: {3.42525091, 0.15432897}
     *    Primitive Gauss[1]: {0.62391373, 0.53532814}
     *    Primitive Gauss[2]: {0.16885540, 0.44463454}
     * Contracted Gauss[1] Type: S, Number of primitives: 1
     *    Primitive Gauss[0]: {0.44463454, 1.00000000}
     * ...
     * @endcode
     * @details This function can be used as follows:
     * @code
     * ElementBasisSet element_basis_set;
     * std::cout << element_basis_set << std::endl;
     * @endcode
     */
    friend std::ostream& operator<<(std::ostream& os, const ElementBasisSet& element_basis_set){
        os << "Element name: " << element_basis_set.element_name << std::endl;
        for(size_t i=0; i<element_basis_set.get_num_contracted_gausses(); i++){
            const ContractedGauss& contracted_gauss = element_basis_set.get_contracted_gauss(i);
            os << "Contracted Gauss[" << i << "] Type: " << contracted_gauss.get_type() << ", ";
            os << "Number of primitives: " << contracted_gauss.get_num_primitives() << std::endl;

            for(size_t j=0; j<contracted_gauss.get_num_primitives(); j++){
                const PrimitiveGauss& primitive = contracted_gauss.get_primitive_gauss(j);
                os << "\tPrimitive Gauss[" << j << "]: {" << primitive.exponent << ", " << primitive.coefficient << "}" << std::endl;
            }
        }
        return os;
    }



private:
    std::string element_name;
    std::vector<ContractedGauss> contracted_gausses;
};




/**
 * @brief BasisSet class
 * @details This class represents the basis set.
 */
class BasisSet {
public:
    /**
     * @brief Add an element basis set to the basis set
     * @param element_basis_set Element basis set
     */
    void add_element_basis_set(const ElementBasisSet& element_basis_set){
        element_basis_sets.insert({element_basis_set.get_element_name(), element_basis_set});
    }

    /**
     * @brief Get the number of element basis sets
     * @return Number of element basis sets
     */
    size_t get_num_element_basis_sets() const { return element_basis_sets.size(); }

    /**
     * @brief Get the element basis set
     * @param element_name Element name of the element basis set
     * @return Element basis set
     */
    const ElementBasisSet& get_element_basis_set(std::string element_name) const {
        if(element_basis_sets.find(element_name) == element_basis_sets.end()){
            THROW_EXCEPTION("Basis set does not include " + element_name);
        }
        return element_basis_sets.at(element_name); 
    }

    /**
     * @brief Get the element name list
     * @return Element name list
     */
    std::vector<std::string> get_element_name_list() const {
        std::vector<std::string> element_name_list;
        for(const auto& element_basis_set : element_basis_sets){
            element_name_list.push_back(element_basis_set.first);
        }
        return element_name_list;
    }



    static BasisSet construct_from_gbs(const std::string& filename);

    /**
     * @brief Print all the basis set
     * @param os Output stream
     * @param basis_set Basis set
     * @return Output stream
     * @details This function prints all the basis set.
     * @details An example of output is as follows:
     * @code
     * ------------------------------------------------
     * Element name: H
     * Contracted Gauss[0] Type: S, Number of primitives: 3
     *    Primitive Gauss[0]: {3.42525091, 0.15432897}
     *    Primitive Gauss[1]: {0.62391373, 0.53532814}
     *    Primitive Gauss[2]: {0.16885540, 0.44463454}
     * Contracted Gauss[1] Type: S, Number of primitives: 1
     *    Primitive Gauss[0]: {0.44463454, 1.00000000}
     * ------------------------------------------------
     * Element name: He
     * Contracted Gauss[0] Type: S, Number of primitives: 3
     *    Primitive Gauss[0]: {6.36242139, 0.15432897}
     *    Primitive Gauss[1]: {1.15892215, 0.53532814}
     *    Primitive Gauss[2]: {0.31364979, 0.44463454}
     * ------------------------------------------------
     * ...
     * @endcode
     * @details This function can be used as follows:
     * @code
     * BasisSet basis_set = BasisSet::construct_from_gbs("sto-3g.gbs");
     * std::cout << basis_set << std::endl;
     * @endcode
     */
    friend std::ostream& operator<<(std::ostream& os, const BasisSet& basis_set){
        for(const auto& element_basis_set : basis_set.element_basis_sets){
            std::cout << "------------------------------------------------" << std::endl;
            os << element_basis_set.second << std::endl;
        }
        return os;
    }
    
private:
    std::unordered_map<std::string, ElementBasisSet> element_basis_sets;
};


} // namespace gansu