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
 * @file builder.hpp 
 */

#pragma once

#include "hf.hpp"
#include "rhf.hpp"
#include "uhf.hpp"
#include "rohf.hpp"
#include "utils.hpp" // THROW_EXCEPTION

#include "parameter_manager.hpp"
#include "gpu_manager.hpp"
#include "env.hpp" // display_env_info

namespace gansu{

/**
 * @brief HFBuilder class
 * @details This class is a virtual class for building the HF class.
 */
class HFBuilder {
public:
    /**
     * @brief Build of the HF class (Factory method)
     */
    static std::unique_ptr<HF> buildHF(const ParameterManager& arg_parameters){

        ParameterManager parameters; // Create a new ParameterManager without setting default values

        // Load the parameters if the parameter file is given
        if(arg_parameters.contains("parameter_file")){
            const std::string parameter_file = arg_parameters.get<std::string>(std::string("parameter_file"));
            parameters.load_from_file(parameter_file);
        }

        // add the command line arguments, which will overwrite the parameters from the file if the same parameter is specified.
        for(const auto& key : arg_parameters.keys()){
            parameters[key] = arg_parameters.get<std::string>(key);
        }


        if(parameters.contains("verbose")){
            std::cout << "Enumerating all parameters:" << std::endl;
            std::vector<std::string> key_list = parameters.keys();
            for (const auto& key : key_list) {
                std::cout << "parameters[" << key << "] = " << parameters.get<std::string>(key) << std::endl;
            }
        }

        // Check if required parameters are set
        if(!parameters.contains("xyzfilename")){
            THROW_EXCEPTION("xyzfilename is not set");
        }
        if(!parameters.contains("gbsfilename")){
            THROW_EXCEPTION("gbsfilename is not set");
        }

        // Set default values for unspecified parameters
        parameters.set_default_values_to_unspecified_parameters();


        // Check the environment
        {
            gpu::cusolverManager cusolver;
            gpu::cublasManager cublas;
            display_env_info();
        }


        // Read the molecular information and create the Molecular class
        const std::string xyzfilename = parameters.get<std::string>("xyzfilename");
        const std::string gbsfilename = parameters.get<std::string>("gbsfilename");
        const int charge = parameters.get<int>("charge");
        const int beta_to_alpha = parameters.get<int>("beta_to_alpha");
        
        Molecular molecular(xyzfilename, gbsfilename, charge, beta_to_alpha);

        // Select and build the HF class
        const std::string method = parameters.get<std::string>("method");
        if(method == "rhf"){
            return std::make_unique<RHF>(molecular, parameters);
        }else if(method == "uhf"){
            return std::make_unique<UHF>(molecular, parameters);
        }else if(method == "rohf"){
            return std::make_unique<ROHF>(molecular, parameters);
        }else{
            THROW_EXCEPTION("Invalid method: " + method);
        }
    }

    static real_t buildOptimizer(const ParameterManager& arg_parameters){
        ParameterManager parameters; // Create a new ParameterManager without setting default values

        // Load the parameters if the parameter file is given
        if(arg_parameters.contains("parameter_file")){
            const std::string parameter_file = arg_parameters.get<std::string>(std::string("parameter_file"));
            parameters.load_from_file(parameter_file);
        }

        // add the command line arguments, which will overwrite the parameters from the file if the same parameter is specified.
        for(const auto& key : arg_parameters.keys()){
            parameters[key] = arg_parameters.get<std::string>(key);
        }


        if(parameters.contains("verbose")){
            std::cout << "Enumerating all parameters:" << std::endl;
            std::vector<std::string> key_list = parameters.keys();
            for (const auto& key : key_list) {
                std::cout << "parameters[" << key << "] = " << parameters.get<std::string>(key) << std::endl;
            }
        }

        // Check if required parameters are set
        if(!parameters.contains("xyzfilename")){
            THROW_EXCEPTION("xyzfilename is not set");
        }
        if(!parameters.contains("gbsfilename")){
            THROW_EXCEPTION("gbsfilename is not set");
        }

        // Set default values for unspecified parameters
        parameters.set_default_values_to_unspecified_parameters();


        // Check the environment
        {
            gpu::cusolverManager cusolver;
            gpu::cublasManager cublas;
        }


        // Read the molecular information and create the Molecular class
        const std::string xyzfilename = parameters.get<std::string>("xyzfilename");
        const std::string gbsfilename = parameters.get<std::string>("gbsfilename");
        const int charge = parameters.get<int>("charge");
        const int beta_to_alpha = parameters.get<int>("beta_to_alpha");


        // Read atoms from the xyz file
        std::vector<Atom> atoms = parseXYZ(xyzfilename);

        //atoms[1].coordinate.z -= 0.5;

        // Geometry optimization calculation
        const real_t initial_step_size = 0.01; // step size for geometry optimization
        const real_t tolerance  = 1.0e-6; // tolerance of step size for geometry optimization
        const int max_iter = 100; // maximum number of iterations for geometry optimization

        std::unique_ptr<HF> hf;
        real_t prev_energy;

        size_t num_basis = 0;
        std::vector<double> best_density_matrix_alpha;
        std::vector<double> best_density_matrix_beta;


        {   // initial energy is computed
            Molecular molecular(atoms, gbsfilename, charge, beta_to_alpha);

            // Select and build the HF class
            const std::string method = parameters.get<std::string>("method");
            if(method == "rhf"){
                hf = std::make_unique<RHF>(molecular, parameters);
            }else if(method == "uhf"){
                hf = std::make_unique<UHF>(molecular, parameters);
            }else if(method == "rohf"){
                hf = std::make_unique<ROHF>(molecular, parameters);
            }else{
                THROW_EXCEPTION("Invalid method: " + method);
            }
            prev_energy = hf->solve();

            num_basis = hf->get_num_basis();
            best_density_matrix_alpha.resize(num_basis * num_basis);
            best_density_matrix_beta.resize(num_basis * num_basis);

            std::cout << "Number of basis functions: " << num_basis << std::endl;
            hf->export_density_matrix(best_density_matrix_alpha.data(), best_density_matrix_beta.data(), num_basis);

        }
        for(int iter=0; iter<max_iter; iter++){
            std::cout << "Atom iteration: " << iter << std::endl;
            bool improved = false;

            for(size_t i=1; i<atoms.size(); i++){ // Skip the first atom (the one that is fixed)
                for(size_t dim=0; dim<3; dim++){
                    for(real_t direction: {-1.0, 1.0}){
                        real_t step_size = initial_step_size;

                        std::vector<double> prev_density_matrix_alpha = best_density_matrix_alpha;
                        std::vector<double> prev_density_matrix_beta = best_density_matrix_beta;

                        while(step_size > tolerance){
                            std::cout << "Atom " << i << ": " << atoms[i].atomic_number;
                            std::cout << " Dim=" << dim << " Direction: " << direction;
                            std::cout << " Step size: " << step_size << std::endl;

                            Atom atom_moved = atoms[i];
                            // Compute the gradient using finite difference
                            real_t move_size = direction * step_size;
                            if(dim==0){
                                atoms[i].coordinate.x += move_size;
                            }else if(dim==1){
                                atoms[i].coordinate.y += move_size;
                            }else if(dim==2){
                                atoms[i].coordinate.z += move_size;
                            }
                            auto atoms_moved = atoms;
                            atoms_moved[i] = atom_moved;

                            Molecular molecular(atoms_moved, gbsfilename, charge, beta_to_alpha);
                            // Select and build the HF class
                            const std::string method = parameters.get<std::string>("method");
                            if(method == "rhf"){
                                hf = std::make_unique<RHF>(molecular, parameters);
                            }else if(method == "uhf"){
                                hf = std::make_unique<UHF>(molecular, parameters);
                            }else if(method == "rohf"){
                                hf = std::make_unique<ROHF>(molecular, parameters);
                            }else{
                                THROW_EXCEPTION("Invalid method: " + method);
                            }
                            // Compute the energy with the perturbed position
                            real_t moved_energy = hf->solve(prev_density_matrix_alpha.data(), prev_density_matrix_beta.data(), true);
                            hf->export_density_matrix(prev_density_matrix_alpha.data(), prev_density_matrix_beta.data(), num_basis);

                            if(moved_energy < prev_energy - tolerance){
                                std::cout << "Atom Improved energy: " << moved_energy << std::endl;
                                for(size_t j=0; j<atoms.size(); j++){
                                    std::cout << "Atom " << atoms[j].atomic_number << ": (" << atoms[j].coordinate.x << ", " << atoms[j].coordinate.y << ", " << atoms[j].coordinate.z << ")" << std::endl;
                                }
                                improved = true;
                                // Update the atom position
                                atoms[i] = atom_moved;
                                prev_energy = moved_energy;
                                best_density_matrix_alpha = prev_density_matrix_alpha;
                                best_density_matrix_beta = prev_density_matrix_beta;
                                //no break; try the same step size in the next iteration
                            }else{
                                step_size *= 0.5; // Reduce the step size if no improvement is found
                            }
                        }
                    }
                }
            }
            if(!improved){
                std::cout << "No improvement found. Stopping optimization." << std::endl;
                break;
            }
        }
        std::cout << "Final energy: " << prev_energy << std::endl;
        std::cout << "Final coordinates:" << std::endl;
        for(const auto& atom : atoms){
            std::cout << atom.atomic_number << ": (" << atom.coordinate.x << ", " << atom.coordinate.y << ", " << atom.coordinate.z << ")" << std::endl;
        }
        return prev_energy;
    }

};

} // namespace gansu