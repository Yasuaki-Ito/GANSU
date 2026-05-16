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
 * @file parameter_manager.cpp 
 */


#include <fstream>

#include <algorithm>
#include <cctype>

#include "parameter_manager.hpp"
#include "utils.hpp" // THROW_EXCEPTION

namespace gansu {

// Function to convert a string to lowercase
std::string toLowerCase(const std::string& input) {
    std::string result = input; // Copy the input string
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}


ParameterManager::ParameterManager(bool set_default_values) {
    // Default parameters
    default_parameters_ = {
        {"parameter_file", ""},                     // string
        {"xyzfilename", ""},                        // string
        {"gbsfilename", ""},                        // string
        {"verbose", "0"},                           // int(bool)
        {"method", "rhf"},                          // string
        {"charge", "0"},                            // int
        {"beta_to_alpha", "0"},                     // int
        {"eri_method", "stored"},                   // string
        {"hash_fock_method", "compact"},              // string: compact, indexed, fullscan
        {"post_hf_method", "none"},                  // string
        {"auxiliary_gbsfilename", ""},              // string
        {"maxiter", "100"},                         // int
        {"convergence_energy_threshold", "1e-6"},   // real_t
        {"schwarz_screening_threshold", "1e-12"},   // real_t 
        {"initial_guess", "core"},                  // string
        {"convergence_method", "diis"},             // string
        {"damping_factor", "0.9"},                  // real_t (0<=damping_factor<=1)
        {"diis_size", "8"},                         // int
        {"diis_include_transform", "0"},            // int(bool)
        {"soscf_start_threshold", "1e-4"},          // real_t: energy diff threshold to switch DIIS→SOSCF
        {"rohf_parameter_name", "roothaan"},        // string
        {"run_type", "energy"},                      // string (energy, gradient, optimize)
        {"optimizer", "bfgs"},                       // string (bfgs)
        {"export_sad_cache", "0"},                   // int (bool)
        {"mulliken", "0"},                          // int (bool)
        {"mayer", "0"},                             // int (bool)
        {"wiberg", "0"},                            // int (bool)
        {"export_molden", "0"},                     // int (bool)
        {"export_lmo_molden", "0"},                 // int (bool) — export Pipek-Mezey localized occupied orbitals as <basename>_lmo.molden for visualization (Avogadro/Jmol/VMD)
        {"int1e_method", "hybrid"},                 // string
        {"n_excited_states", "5"},                   // int
        {"adc2_solver", "auto"},                     // string (auto, schur_static, schur_omega, full)
        {"eom_mp2_solver", "auto"},                  // string (auto, schur_static, schur_omega, full)
        {"eom_cc2_solver", "auto"},                    // string (auto, schur_static, schur_omega, full)
        {"spin_type", "singlet"},                       // string (singlet, triplet)
        {"frozen_core", "none"},                         // string: "none", "auto", or integer
        {"ecp_filename", ""},                              // string: ECP file path
        {"num_gpus", "-1"},                                // int: number of GPUs (-1 = auto-detect all)
        {"adc_c_t", "1.3"},                                // real_t: SOS-ADC(2) t2 amplitude scaling
        {"adc_c_c", "0.85"},                               // real_t: SOS-ADC(2) coupling block scaling
        {"dmet_fragments", ""},                            // string: fragment specification e.g. "{0,1,2} {3,4,5}"
        {"dmet_threshold", "1e-6"},                        // real_t: SVD threshold for bath orbital selection
        {"dmet_mu_refine_ccsd", "0"},                      // bool (0/1): refine μ with CCSD-relaxed density after HF stage
        {"dmet_n_tol", "1e-5"},                            // real_t: bisection tol on |Σ N_frag − N_elec| (Vayesta-compat: 4.2e-3 for benzene)
        {"opt_max_iter", "200"},                           // int: maximum geometry optimization iterations
        {"opt_grad_threshold", "3.0e-4"},                  // real_t: max gradient component (Hartree/Bohr)
        {"opt_rms_grad_threshold", "2.0e-4"},              // real_t: RMS gradient (Hartree/Bohr)
        {"opt_energy_threshold", "1.0e-6"},                // real_t: energy change threshold (Hartree)
        {"opt_disp_threshold", "3.0e-4"},                  // real_t: max displacement (Bohr)
        {"opt_step_max", "0.3"},                           // real_t: trust-region radius (Bohr)
        {"thc_n_radial", "50"},                            // int: THC grid radial points per atom (Treutler-Ahlrichs M3)
        {"thc_lebedev_order", "194"},                      // int: THC angular grid (110, 194, 302)
        {"thc_n_laplace", "12"},                           // int: Laplace quadrature points for THC-SOS-MP2
        {"thc_rel_cutoff", "1e-7"},                        // real_t: relative SVD cutoff in LS-THC
        {"thc_sos_c_os", "1.3"},                           // real_t: opposite-spin scaling factor for THC-SOS-MP2
        {"thc_b3a3", "0"},                                 // bool (0/1): enable B3+A3 Schur correction. Off by default — present implementation over-corrects ~1.5 eV (LS-THC structural)
        {"thc_b3", "1"},                                   // bool (0/1): per-term toggle for B3 (consulted only when thc_b3a3=1, diagnostic use)
        {"thc_a3", "1"},                                   // bool (0/1): per-term toggle for A3 (consulted only when thc_b3a3=1, diagnostic use)
        {"thc_density_threshold", "0"},                    // real_t: drop grid points with electron density ρ ≤ threshold (Phase 2.3 (B); 0 = no pruning, typical 1e-8)
        {"thc_max_rank", "0"},                             // int: cap LS-THC rank with randomized SVD (avoids O(N⁴) M·M^T eigendecomp). 0 = full eigendecomp (default for small systems); positive = randomized SVD with that rank cap. Use for N_bas ≥ 200
        {"thc_rsvd_power_iter", "4"},                      // int: power iterations for randomized SVD (only used when thc_max_rank > 0). q=2 is the textbook minimum, q=4-5 is needed for slow-decay spectra (LS-THC tail)
        // ----- DLPNO parameters -----
        {"dlpno_preset", "normal"},                        // string: loose / normal / tight / very_tight (ORCA-compatible)
        {"dlpno_localizer", "pm"},                         // string: pm (Pipek-Mezey, default) / boys (Foster-Boys) / ibo
        {"dlpno_t_cut_pno", "-1"},                         // real_t: PNO occupation cutoff. -1 = use preset value
        {"dlpno_t_cut_do", "-1"},                          // real_t: PAO redundancy (overlap eigenvalue) cutoff. -1 = use preset
        {"dlpno_t_cut_pairs", "-1"},                       // real_t: strong/weak pair MP2 cutoff (Ha). -1 = use preset
        {"dlpno_t_cut_mkn", "-1"},                         // real_t: Boughton-Pulay Mulliken cumulative threshold. -1 = use preset
        {"dlpno_t_cut_triples", "-1"},                     // real_t: triple-screening threshold for (T). -1 = use preset
        {"dlpno_t_cut_tno", "-1"},                         // real_t: TNO occupation cutoff for (T). -1 = use preset
        {"dlpno_pair_distance_cutoff", "15.0"},            // real_t: distance-based pair pre-screening (Bohr). 0 = off
        {"dlpno_max_iter", "50"},                          // int: DLPNO-CCSD residual max iterations
        {"dlpno_diis_size", "6"},                          // int: DIIS subspace size for DLPNO-CCSD
        {"dlpno_localizer_max_sweep", "200"},              // int: PM/Boys Jacobi sweep upper bound
        {"dlpno_localizer_conv", "1e-10"},                 // real_t: localizer ΔP² convergence threshold
        {"dlpno_lmp2_max_iter", "100"},                    // int: iterative LMP2 amplitude solver max iterations (also caps CCSD T2 dressing iter; hexamer-class CCSD T2 needs ~70-80 iters at conv 1e-8)
        {"dlpno_lmp2_conv", "1e-8"},                       // real_t: LMP2 residual convergence (max-abs amplitude residual)
        {"dlpno_sc_pno_iter", "1"},                        // int: extra rounds of self-consistent PNO refinement (0 = single-shot PNO selection from semi-canonical guess)
        {"dlpno_pno_os_only", "0"},                        // bool: PNO selection from opposite-spin amplitudes only (D = T^T T + T T^T). Default 0 = use full LMP2 density (Riplinger 2013, T̃^T T + T̃ T^T) which slightly outperforms the OS-only form for full closed-shell MP2 energy evaluation. Set to 1 only when pairing with SOS-MP2 (c_os scaling, SS dropped).
        {"dlpno_verbose", "1"},                            // int: 0=summary, 1=phase, 2=per-pair, 3=residual
        {"dlpno_compute_density", "0"},                    // bool: build DLPNO Λ + 1-RDM after MP2/CCSD energy (Sub-phase 1+ of DLPNO-CCSD-Λ project). Default 0 = energy only (no extra cost). Set 1 for DMET / properties / dipole; print sanity block + dipole when combined with dlpno_verbose >= 1.
        {"dlpno_lambda_full_dressing", "0"}                // bool: Sub-phase 2X.2c. Enable full F-eff dressing (phase24-based dF_ki + DF_per_pair) in the DLPNO-CCSD Λ iteration. Default 0 = LMP2-limit closed-form Λ_2 = 2 Y - Y^T (agrees with canonical CCSD oo/vv to ~1e-5). Set 1 to engage the full Path A dressing to close the 6.3% off-canonical dipole gap; iteration costs ~50% more than closed-form. See DLPNO_Lambda.md §12.
    };


    // Default short-to-full option mappings
    short_to_full_ = {
        {"-m", "method"},
        {"-v", "verbose"},
        {"-p", "parameter_file"},
        {"-x", "xyzfilename"},
        {"-g", "gbsfilename"},
        {"-ag", "auxiliary_gbsfilename"},
        {"-ecp", "ecp_filename"},
        {"-c", "charge"},
        {"-r", "run_type"},
    };

    if(set_default_values){
        set_default_values_to_unspecified_parameters();
    }
}





bool ParameterManager::contains(const std::string& key) const {
    std::string key_lower = toLowerCase(key);
    return parameters_.find(key_lower) != parameters_.end();
}

bool ParameterManager::is_valid_key(const std::string& key) const {
    std::string key_lower = toLowerCase(key);
    return default_parameters_.find(key_lower) != default_parameters_.end();
}

void ParameterManager::set_default_values_to_unspecified_parameters() {
    for (const auto& pair : default_parameters_) {
        if (!contains(pair.first)) {
            parameters_[pair.first] = pair.second;
        }
    }
}



std::vector<std::string> ParameterManager::keys() const {
    std::vector<std::string> key_list;
    for (const auto& pair : parameters_) {
        key_list.push_back(pair.first);
    }
    return key_list;
}


void ParameterManager::parse_command_line_args(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string key;
        std::string value;
        if (arg.rfind("--", 0) == 0) { // Long option, starts with "--"
            key = arg.substr(2);
        } else if (arg.rfind("-", 0) == 0) { // Short option, starts with "-"
            auto it = short_to_full_.find(arg);
            if (it != short_to_full_.end()) {
                key = it->second; // Convert short option to full option
            } else {
                THROW_EXCEPTION("Unknown short option: " + arg);
            }
        } else {
            THROW_EXCEPTION("Invalid option format: " + arg);
        }
        // Check if the next argument is an option (--key or known short option like -m)
        // rather than a value. This allows values starting with '-' (e.g., "-1" for charge,
        // "cg-fr" for optimizer) to be correctly consumed as values.
        auto next_is_option = [&]() -> bool {
            if (i + 1 >= argc) return true; // no next argument
            std::string next = argv[i + 1];
            if (next.rfind("--", 0) == 0) return true; // long option
            if (next.rfind("-", 0) == 0 && short_to_full_.find(next) != short_to_full_.end()) return true; // known short option
            return false;
        };

        // Check for a value
        if (!next_is_option()) {
            value = argv[++i]; // Consume the next argument as the value
        } else {
            value = "1"; // Default to "true" for flags without values, e.g. "--verbose"
        }


        // convert boolean values to "1" or "0"
        if(value == "true" || value == "True" || value == "TRUE"){
            value = "1";
        }else if(value == "false" || value == "False" || value == "FALSE"){
            value = "0";
        }

        if(is_valid_key(key)){
            parameters_[key] = value;
        }else{
            THROW_EXCEPTION("Unknown parameter: " + key);
        }
    }
}



void ParameterManager::load_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        THROW_EXCEPTION("Failed to open parameter file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Remove comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        // Remove carriage return (in case of \r\n line endings)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // Skip empty or whitespace-only lines
        if (line.empty() || line.find_first_not_of(" \t") == std::string::npos) {
            continue;
        }

        // Parse "key = value" format
        auto delimiter_pos = line.find('=');
        if (delimiter_pos == std::string::npos) {
            THROW_EXCEPTION("Invalid line format: " + line);
        }

        std::string key = line.substr(0, delimiter_pos);
        std::string value = line.substr(delimiter_pos + 1);

        // Trim whitespace
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        // convert boolean values to "1" or "0"
        if(value == "true" || value == "True" || value == "TRUE"){
            value = "1";
        }else if(value == "false" || value == "False" || value == "FALSE"){
            value = "0";
        }

        std::string key_lower = toLowerCase(key);

        // add the key-value pair to the parameters map
        if(is_valid_key(key_lower)){
            if(key_lower == "xyzfilename" || key_lower == "gbsfilename" || key_lower == "parameter_file"){
                parameters_[key_lower] = value;
            }else{
                std::string value_lower = toLowerCase(value);
                parameters_[key_lower] = value_lower;
            }
        }else{
            THROW_EXCEPTION("Unknown parameter: " + key);
        }
    }
}

} // namespace gansu