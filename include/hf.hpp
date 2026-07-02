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
 * @file hf.hpp 
 * @brief This file contains the definition of the HF class and update Fock matrix class.
 */

#pragma once

#include "types.hpp"
#include <memory>
#include "basis_set.hpp"
#include "molecular.hpp"

#include "post_hf_method.hpp"

#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "parameter_manager.hpp"
#include "eri.hpp"
#include "cis_nto_active_space.hpp"
#include "ip_eom_result.hpp"
#include "ea_eom_result.hpp"
#include "steom_result.hpp"
#include "bt_pno_backtransform.hpp"  // BTAmplitudes (hybrid DLPNO-STEOM P5b)
#include "steom_barh_cache.hpp"      // SteomBarHCache (DLPNO-STEOM build_dressed sharing, (A))


namespace gansu{

// prototype of classes
class ERI;

/**
 * @brief Get or generate an auxiliary basis set for RI approximation
 * @param molecular Primary molecular object
 * @param auxiliary_gbsfilename Auxiliary basis set file name (if empty, auto-generate from primary basis)
 * @return Auxiliary BasisSet
 */
inline BasisSet get_auxiliary_basis(const Molecular& molecular, const std::string& auxiliary_gbsfilename) {
    if (!auxiliary_gbsfilename.empty()) {
        return BasisSet::construct_from_gbs(auxiliary_gbsfilename);
    }
    // Auto-generate auxiliary basis from primary basis
    std::cout << "[RI] No auxiliary basis set file specified. Auto-generating from primary basis..." << std::endl;
    BasisSet primary_basis = BasisSet::construct_from_gbs(molecular.get_gbs_filename());
    return BasisSet::generate_auxiliary_basis(primary_basis);
}

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

    /// Phase 1 spherical-basis accessors (forwarded from Molecular at construction).
    bool get_use_spherical()              const { return use_spherical_; }
    int  get_num_basis_cart()             const { return num_basis_cart_; }
    const std::vector<int>& get_shell_types()         const { return shell_types_; }
    const std::vector<int>& get_shell_offsets_cart()  const { return shell_offsets_cart_; }
    const std::vector<int>& get_shell_offsets_sph()   const { return shell_offsets_sph_; }

    /**
     * @brief Get the number of electrons
     */
    int get_num_electrons() const { return num_electrons; }

    real_t get_energy_difference() const { return energy_difference_; } ///< Get the energy difference from the previous iteration

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
     * @brief Get the run type (energy, gradient, optimize)
     */
    const std::string& get_run_type() const { return run_type_; }

    /**
     * @brief Get the optimizer name (bfgs)
     */
    const std::string& get_optimizer() const { return optimizer_; }

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

    /// Always-Cartesian atom→basis ranges (for Molden [GTO] block, which writes
    /// Cartesian shell info regardless of the active basis).
    const std::vector<BasisRange>& get_atom_to_basis_range_cart() const { return atom_to_basis_range_cart; }

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
     * @brief Get cgto_normalization_factors
     */
    const DeviceHostMemory<real_t>& get_cgto_normalization_factors() const { return cgto_normalization_factors; }

    /**
     * @brief Get the Schwartz screening threshold
     */
    real_t get_schwarz_screening_threshold() const { return schwarz_screening_threshold; }

    /// SOS-ADC(2) scaling: t2 amplitude scaling (default 1.3)
    double get_adc_c_t() const { return adc_c_t_; }
    /// SOS-ADC(2) scaling: coupling block scaling (default 1.17)
    double get_adc_c_c() const { return adc_c_c_; }

    /// DMET fragment specification string
    const std::string& get_dmet_fragments_str() const { return dmet_fragments_str_; }
    /// DMET SVD threshold for bath orbital selection
    double get_dmet_threshold() const { return dmet_threshold_; }
    /// DMET: refine μ with CCSD-relaxed density after HF stage (2-stage μ optimization)
    bool get_dmet_mu_refine_ccsd() const { return dmet_mu_refine_ccsd_; }
    /// DMET: bisection tolerance on |Σ N_frag − N_elec|. Default 1e-5 (tight, GANSU native).
    /// Vayesta-compat: 4.2e-3 for benzene/STO-3G (= 1e-4 × 42, matches Vayesta's max_elec_err).
    double get_dmet_n_tol() const { return dmet_n_tol_; }
    /// Geometry optimization parameters (used in optimize_geometry path)
    int get_opt_max_iter() const { return opt_max_iter_; }
    double get_opt_grad_threshold() const { return opt_grad_threshold_; }
    double get_opt_rms_grad_threshold() const { return opt_rms_grad_threshold_; }
    double get_opt_energy_threshold() const { return opt_energy_threshold_; }
    double get_opt_disp_threshold() const { return opt_disp_threshold_; }
    double get_opt_step_max() const { return opt_step_max_; }

    /// THC parameters (used by THC-MP2 / THC-SOS-MP2 / THC-ADC(2) paths)
    int    get_thc_n_radial() const { return thc_n_radial_; }
    int    get_thc_lebedev_order() const { return thc_lebedev_order_; }
    int    get_thc_n_laplace() const { return thc_n_laplace_; }
    double get_thc_rel_cutoff() const { return thc_rel_cutoff_; }
    double get_thc_sos_c_os() const { return thc_sos_c_os_; }
    bool   get_thc_b3a3() const { return thc_b3a3_; }
    bool   get_thc_b3() const { return thc_b3_; }
    bool   get_thc_a3() const { return thc_a3_; }
    double get_thc_density_threshold() const { return thc_density_threshold_; }
    int    get_thc_max_rank() const { return thc_max_rank_; }
    int    get_thc_rsvd_power_iter() const { return thc_rsvd_power_iter_; }

    /// DLPNO parameters. -1 sentinels mean "fall back to preset" (resolved in dlpno_params.hpp)
    const std::string& get_dlpno_preset() const { return dlpno_preset_; }
    const std::string& get_dlpno_localizer() const { return dlpno_localizer_; }
    double get_dlpno_t_cut_pno()     const { return dlpno_t_cut_pno_; }
    double get_dlpno_t_cut_do()      const { return dlpno_t_cut_do_; }
    double get_dlpno_t_cut_pairs()   const { return dlpno_t_cut_pairs_; }
    double get_dlpno_t_cut_mkn()     const { return dlpno_t_cut_mkn_; }
    double get_dlpno_t_cut_triples() const { return dlpno_t_cut_triples_; }
    double get_dlpno_t_cut_tno()     const { return dlpno_t_cut_tno_; }
    double get_dlpno_pair_distance_cutoff() const { return dlpno_pair_distance_cutoff_; }
    int    get_dlpno_max_iter() const { return dlpno_max_iter_; }
    int    get_dlpno_diis_size() const { return dlpno_diis_size_; }
    int    get_dlpno_localizer_max_sweep() const { return dlpno_localizer_max_sweep_; }
    double get_dlpno_localizer_conv() const { return dlpno_localizer_conv_; }
    int    get_dlpno_lmp2_max_iter() const { return dlpno_lmp2_max_iter_; }
    double get_dlpno_lmp2_conv()     const { return dlpno_lmp2_conv_; }
    int    get_dlpno_sc_pno_iter()   const { return dlpno_sc_pno_iter_; }
    bool   get_dlpno_pno_os_only()   const { return dlpno_pno_os_only_; }
    int    get_dlpno_verbose() const { return dlpno_verbose_; }
    int    get_dlpno_cpu_threads() const { return dlpno_cpu_threads_; }
    bool   get_dlpno_compute_density() const { return dlpno_compute_density_; }
    /// Sub-phase 2X.2c+: enable full F-eff dressing in the DLPNO-CCSD Λ
    /// iteration (phase24-based cross-pair dF_ki + per-pair DF). Default
    /// off — current production density uses the LMP2-limit closed-form
    /// Λ_2 = 2 Y - Y^T which agrees with canonical CCSD oo/vv blocks to
    /// ~1e-5. The full dressing path is iterated independently to close
    /// the closed-form 6.3% off-canonical dipole gap. When stable will
    /// become the production default. See DLPNO_Lambda.md §12.
    bool   get_dlpno_lambda_full_dressing() const { return dlpno_lambda_full_dressing_; }

    /// Number of GPUs requested (-1 = auto-detect, 1 = single, > 1 = multi).
    int    get_num_gpus() const { return num_gpus_; }

    /// CIS NTO (bt-PNO-STEOM Phase P0) parameters
    int                get_cis_nto_n_root_cis() const { return cis_nto_n_root_cis_; }
    double             get_cis_nto_o_thresh()   const { return cis_nto_o_thresh_;  }
    double             get_cis_nto_v_thresh()   const { return cis_nto_v_thresh_;  }
    const std::string& get_cis_nto_weights()    const { return cis_nto_weights_;   }
    int                get_cis_nto_verbose()    const { return cis_nto_verbose_;   }

    /// IP-EOM-CCSD (bt-PNO-STEOM Phase P1) parameters
    double get_ip_eom_ip_thresh()      const { return ip_eom_ip_thresh_;      }
    int    get_ip_eom_safety_margin()  const { return ip_eom_safety_margin_;  }
    bool   get_ip_eom_followcis()      const { return ip_eom_followcis_;      }
    double get_ip_eom_d_tol()          const { return ip_eom_d_tol_;          }
    double get_ip_eom_r_tol()          const { return ip_eom_r_tol_;          }
    int    get_ip_eom_max_iter()       const { return ip_eom_max_iter_;       }
    int    get_ip_eom_verbose()        const { return ip_eom_verbose_;        }

    /// EA-EOM-CCSD (bt-PNO-STEOM Phase P2) parameters
    double get_ea_eom_ea_thresh()      const { return ea_eom_ea_thresh_;      }
    int    get_ea_eom_safety_margin()  const { return ea_eom_safety_margin_;  }
    bool   get_ea_eom_followcis()      const { return ea_eom_followcis_;      }
    double get_ea_eom_d_tol()          const { return ea_eom_d_tol_;          }
    double get_ea_eom_r_tol()          const { return ea_eom_r_tol_;          }
    int    get_ea_eom_max_iter()       const { return ea_eom_max_iter_;       }
    int    get_ea_eom_verbose()        const { return ea_eom_verbose_;        }

    /// STEOM-CCSD (bt-PNO-STEOM Phase P3) parameters
    int    get_steom_n_root_cis()      const { return steom_n_root_cis_;      }
    double get_steom_active_char_warn() const { return steom_active_char_warn_; }
    double get_steom_d_tol()           const { return steom_d_tol_;           }
    double get_steom_r_tol()           const { return steom_r_tol_;           }
    int    get_steom_max_iter()        const { return steom_max_iter_;        }
    int    get_steom_verbose()         const { return steom_verbose_;         }

    /// Post-HF runtime statistics — populated by the post-HF driver as it
    /// runs so that the post-HF summary can report the actual problem size.
    /// All default to 0 if the corresponding method was not used.
    int  get_last_dmet_n_fragments() const { return last_dmet_n_fragments_; }
    void set_last_dmet_n_fragments(int n) { last_dmet_n_fragments_ = n; }

    int  get_last_dlpno_n_strong() const { return last_dlpno_n_strong_; }
    int  get_last_dlpno_n_weak()   const { return last_dlpno_n_weak_; }
    int  get_last_dlpno_n_empty()  const { return last_dlpno_n_empty_; }
    void set_last_dlpno_pairs(int strong, int weak, int empty) {
        last_dlpno_n_strong_ = strong;
        last_dlpno_n_weak_   = weak;
        last_dlpno_n_empty_  = empty;
    }

    int  get_last_dlpno_n_triples_total()  const { return last_dlpno_n_triples_total_; }
    int  get_last_dlpno_n_triples_active() const { return last_dlpno_n_triples_active_; }
    void set_last_dlpno_triples(int total, int active) {
        last_dlpno_n_triples_total_  = total;
        last_dlpno_n_triples_active_ = active;
    }

    /**
     * @brief Get Shell-pair type info
     */
    const std::vector<ShellPairTypeInfo>& get_shell_pair_type_infos() const { return shell_pair_type_infos; }

    /**
     * @brief Get num_primitive_shells
     */
    size_t get_num_primitive_shells() const { return num_primitive_shells; }

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


    /**
     * @brief Post process after SCF calculation
     * @details This function performs post processing after the SCF calculation (e.g., Post HF computation)
     * @details This function is a virtual function and can be overridden in the derived class.
     */
    virtual void post_process_after_scf(){} ///< Post process after SCF calculation (e.g., Post HF computation)

    /**
     * @brief Get the post-HF method
     */
    PostHFMethod get_post_hf_method() const { return post_hf_method_; } ///< Get the post-HF method

    /**
     * @brief Get the post-HF energy
     */
    real_t get_post_hf_energy() const { return post_hf_energy_; } ///< Get the post-HF energy

    /**
     * @brief Set the post-HF energy (called by ERI methods that compute correlation energy internally)
     */
    void set_post_hf_energy(real_t energy) { post_hf_energy_ = energy; }

    /// CCSD 1-RDM accessors (for DMET / property post-analysis)
    std::vector<real_t>& get_ccsd_1rdm_mo() { return ccsd_1rdm_mo_; }
    std::vector<real_t>& get_ccsd_1rdm_ao() { return ccsd_1rdm_ao_; }
    const std::vector<real_t>& get_ccsd_1rdm_mo() const { return ccsd_1rdm_mo_; }
    const std::vector<real_t>& get_ccsd_1rdm_ao() const { return ccsd_1rdm_ao_; }

    /**
     * @brief Get the number of excited states to compute
     */
    int get_n_excited_states() const { return n_excited_states_; }

    /**
     * @brief Get the number of frozen core orbitals
     */
    int get_num_frozen_core() const { return num_frozen_core_; }

    /**
     * @brief Get the number of active (non-frozen) occupied orbitals
     */
    int get_num_active_occ() const { return num_electrons / 2 - num_frozen_core_; }

    /**
     * @brief Get ADC(2) solver mode: "schur_static", "schur_omega", or "full"
     */
    const std::string& get_adc2_solver() const { return adc2_solver_; }

    /**
     * @brief Get EOM-MP2 solver mode: "full" or "schur"
     */
    const std::string& get_eom_mp2_solver() const { return eom_mp2_solver_; }

    /**
     * @brief Get EOM-CC2 solver mode: "auto", "schur_static", "schur_omega", or "full"
     */
    const std::string& get_eom_cc2_solver() const { return eom_cc2_solver_; }

    /**
     * @brief Get spin type for excited states: "singlet" or "triplet"
     */
    const std::string& get_spin_type() const { return spin_type_; }
    bool is_triplet() const { return spin_type_ == "triplet"; }

    /**
     * @brief Get excitation energies (populated after CIS/EOM calculations)
     */
    const std::vector<real_t>& get_excitation_energies() const { return excitation_energies_; }

    /**
     * @brief Set excitation energies (called by ERI compute_cis etc.)
     */
    void set_excitation_energies(const std::vector<real_t>& energies) { excitation_energies_ = energies; }

    /**
     * @brief Get oscillator strengths (populated after excited state calculations)
     */
    const std::vector<real_t>& get_oscillator_strengths() const { return oscillator_strengths_; }

    /**
     * @brief Set oscillator strengths (called by ERI compute_cis etc.)
     */
    void set_oscillator_strengths(const std::vector<real_t>& strengths) { oscillator_strengths_ = strengths; }

    /**
     * @brief Get/set excited state report string (printed in final summary)
     */
    const std::string& get_excited_state_report() const { return excited_state_report_; }
    void set_excited_state_report(const std::string& report) { excited_state_report_ = report; }
    /// Append a chunk to the excited-state report. Used when multiple excited
    /// post-HF stages (e.g. CIS + CIS-NTO) contribute to a single run's report.
    void append_excited_state_report(const std::string& chunk) {
        if (!chunk.empty()) {
            if (!excited_state_report_.empty()) excited_state_report_ += "\n";
            excited_state_report_ += chunk;
        }
    }

    /**
     * @brief CIS NTO active space result (bt-PNO-STEOM Phase P0).
     *
     * Populated by `ERI::compute_cis_nto`. Empty (`nocc_active==0`) if the
     * post-HF method was not CIS_NTO. Read by P1 IP-EOM-CCSD (when it lands).
     */
    const CISNTOResult& get_cis_nto_result() const { return cis_nto_result_; }
    CISNTOResult&       get_cis_nto_result()       { return cis_nto_result_; }
    void set_cis_nto_result(CISNTOResult result) { cis_nto_result_ = std::move(result); }

    /// Raw per-root CIS amplitudes [n_states × nocc_active × nvir] stashed by
    /// compute_cis_nto (RI path) so DMET-STEOM can re-weight the state-averaged
    /// NTO by fragment localization (root-targeting, AQUA/DMET_STEOM.md §Step B).
    const std::vector<real_t>& get_last_cis_amplitudes() const { return last_cis_amplitudes_; }
    void set_last_cis_amplitudes(std::vector<real_t> a) { last_cis_amplitudes_ = std::move(a); }

    /**
     * @brief IP-EOM-CCSD result (bt-PNO-STEOM Phase P1).
     *
     * Populated by `ERI::compute_ip_eom_ccsd`. Empty (`n_active==0`) if the
     * post-HF method was not IP_EOM_CCSD. Read by P3 STEOM (when it lands).
     */
    const IPEOMResult& get_ip_eom_result() const { return ip_eom_result_; }
    IPEOMResult&       get_ip_eom_result()       { return ip_eom_result_; }
    void set_ip_eom_result(IPEOMResult result) { ip_eom_result_ = std::move(result); }

    /**
     * @brief EA-EOM-CCSD result (bt-PNO-STEOM Phase P2).
     *
     * Populated by `ERI::compute_ea_eom_ccsd`. Empty (`n_active==0`) if the
     * post-HF method was not EA_EOM_CCSD. Read by P3 STEOM (when it lands).
     */
    const EAEOMResult& get_ea_eom_result() const { return ea_eom_result_; }
    EAEOMResult&       get_ea_eom_result()       { return ea_eom_result_; }
    void set_ea_eom_result(EAEOMResult result) { ea_eom_result_ = std::move(result); }

    /**
     * @brief STEOM-CCSD result (bt-PNO-STEOM Phase P3).
     *
     * Populated by `ERI::compute_steom_ccsd`. Empty (`n_states==0`) if the
     * post-HF method was not STEOM_CCSD.
     */
    const STEOMResult& get_steom_result() const { return steom_result_; }
    STEOMResult&       get_steom_result()       { return steom_result_; }
    void set_steom_result(STEOMResult result) { steom_result_ = std::move(result); }

    /**
     * @brief DLPNO-CCSD T1/T2 back-transformed to canonical MO (hybrid bt-PNO-STEOM P5b).
     *
     * When `use_dlpno_amplitudes()` is true the IP-EOM / EA-EOM / STEOM impls
     * consume these canonical amplitudes (one device copy each) instead of
     * solving a fresh canonical CCSD ground state. Set by
     * `ERI_RI_RHF::compute_dlpno_steom_ccsd`; cleared after that driver returns.
     */
    const BTAmplitudes& get_dlpno_bt_amplitudes() const { return dlpno_bt_amplitudes_; }
    void set_dlpno_bt_amplitudes(BTAmplitudes a) { dlpno_bt_amplitudes_ = std::move(a); use_dlpno_amplitudes_ = true; }
    bool use_dlpno_amplitudes() const { return use_dlpno_amplitudes_; }
    void clear_dlpno_amplitudes() { use_dlpno_amplitudes_ = false; dlpno_bt_amplitudes_ = BTAmplitudes{}; }

    /// When set, DLPNOCCSD::compute_energy back-transforms its converged
    /// amplitudes to canonical MO and stores them via set_dlpno_bt_amplitudes
    /// (used by the hybrid DLPNO-STEOM driver). Default false → plain
    /// DLPNO-CCSD runs pay no back-transform cost. When set it ALSO stows the
    /// converged DLPNOLMP2Result (set_dlpno_res) so the projected DLPNO-IP/EA
    /// path (stage B) can reach the per-pair PNOs / barS.
    void set_collect_dlpno_bt(bool b) { collect_dlpno_bt_ = b; }
    bool collect_dlpno_bt() const { return collect_dlpno_bt_; }

    /// Converged DLPNO-CCSD pair state (per-pair bar_Q/Lambda/setups/U_loc),
    /// stowed when collect_dlpno_bt is set. Consumed by the projected
    /// DLPNO-IP-EOM / DLPNO-EA-EOM operators (bt-PNO-STEOM stage B).
    void set_dlpno_res(DLPNOLMP2Result r) { dlpno_res_ = std::move(r); }
    const DLPNOLMP2Result& get_dlpno_res() const { return dlpno_res_; }

    /// When set, the IP-EOM / EA-EOM impls run the Galerkin-projected DLPNO
    /// operator (per-pair PNO 2h1p/2p1h space) instead of the full canonical
    /// Davidson. Requires get_dlpno_res() to be populated. Set by the
    /// DLPNO-STEOM driver; cleared afterwards. Default false.
    void set_use_dlpno_projected_eom(bool b) { use_dlpno_projected_eom_ = b; }
    bool use_dlpno_projected_eom() const { return use_dlpno_projected_eom_; }

    /// (A) build_dressed de-duplication (env GANSU_STEOM_SHARE_BARH=1).
    /// When set by the DLPNO-STEOM driver, the IP/EA operators publish their
    /// dressed bar-H tensors into steom_barh_cache_ and the STEOM operator
    /// borrows all 11, skipping its own build_dressed_intermediates entirely.
    /// Default false → each operator rebuilds independently (byte-identical).
    /// See include/steom_barh_cache.hpp.
    void set_steom_share_barh(bool b) { steom_share_barh_ = b; }
    bool steom_share_barh() const { return steom_share_barh_; }
    SteomBarHCache& steom_barh_cache() { return steom_barh_cache_; }
    const SteomBarHCache& steom_barh_cache() const { return steom_barh_cache_; }

    /// stage B (a): IP impl runs the NATIVE per-pair DLPNO-IP-EOM σ operator
    /// (DLPNOIPEOMNativeOperator) instead of the project-up reference. Set
    /// alongside use_dlpno_projected_eom_ (EA stays projected until native EA
    /// lands). Default false → projected/hybrid/canonical paths unchanged.
    void set_use_dlpno_native_eom(bool b) { use_dlpno_native_eom_ = b; }
    bool use_dlpno_native_eom() const { return use_dlpno_native_eom_; }

    /**
     * @brief Get whether the coefficient matrix has been computed
     */
    bool get_hasMatrixC() const { return hasMatrixC_; }

    void switchHasMatrixC() { hasMatrixC_ = true; }

    /**
     * @brief Set/clear the coefficient-matrix-valid flag
     * @details Used by the C API SCF-free energy evaluation (gansu_energy_from_density)
     * to force the density-based Fock build path even after SCF has populated C
     * (the RI Fock builders switch to the coefficient-matrix path once hasMatrixC_ is true).
     */
    void set_hasMatrixC(bool b) { hasMatrixC_ = b; }

    /**
     * @brief Get the ERI method object (nullptr until set by the builder)
     */
    ERI* get_eri_method() { return eri_method_.get(); }

    /**
     * @brief Function to compute the coefficient matrix and set hasMatrixC_ to true
     * @details This function computes the coefficient matrix by calling compute_coefficient_matrix()
     * and sets hasMatrixC_ to true.
    */
    void compute_coefficient_matrix(){
        compute_coefficient_matrix_impl();
        hasMatrixC_ = true;
    }

    /**
     * @brief Get the algorithm name for int1e
     * @return Algorithm name as a string
    */
    const std::string get_int1e_algorithm_name() const{ return int1e_method;}; ///< Get the algorithm name


    /**
     * @brief Get the algorithm name for initial_guess
     * @return Algorithm name as a string
    */
    const std::string get_initial_guess_algorithm_name() const{ return initial_guess_method_;}; ///< Get the algorithm name

protected:
    long long solve_time_in_milliseconds_; ///< Time to solve the HF equation

    const int num_basis; ///< Number of basis functions (= Spherical when use_spherical_=true, else Cartesian)

    // Phase 1 spherical-harmonic basis support (Molden 5D/7F/9G).  When
    // use_spherical_=true the integral engine still computes in Cartesian
    // (size num_basis_cart_), then compute_core_hamiltonian_matrix /
    // precompute_eri_matrix apply spherical_transform.hpp to produce the
    // matrices at size num_basis (= num_basis_sph).  SCF and post-HF then
    // operate entirely in the smaller Spherical basis.
    const bool use_spherical_;
    const int  num_basis_cart_;                   ///< Cartesian count (always set)
    const std::vector<int> shell_types_;          ///< per-shell L values
    const std::vector<int> shell_offsets_cart_;   ///< per-shell cart offsets (size n_shells+1)
    const std::vector<int> shell_offsets_sph_;    ///< per-shell sph  offsets (size n_shells+1)

    const int num_electrons; ///< Number of electrons
    const int num_alpha_spins; ///< Number of alpha electrons
    const int num_beta_spins; ///< Number of beta electrons

    const int verbose; ///< Verbose mode
    const real_t convergence_energy_threshold; ///< Convergence criterion
    const real_t schwarz_screening_threshold; ///< Schwartz screening threshold
    double adc_c_t_ = 1.3;   ///< SOS-ADC(2) t2 amplitude scaling
    double adc_c_c_ = 0.85;  ///< SOS-ADC(2) coupling block scaling
    std::string dmet_fragments_str_;    ///< DMET fragment specification string
    double dmet_threshold_ = 1e-6;     ///< SVD threshold for DMET bath orbital selection
    bool dmet_mu_refine_ccsd_ = false;  ///< DMET: refine μ with CCSD-relaxed density
    double dmet_n_tol_ = 1e-5;  ///< DMET: bisection tolerance on |Σ N_frag − N_elec|
    int opt_max_iter_ = 200;    ///< Geometry optimization max iterations
    double opt_grad_threshold_ = 3.0e-4;
    double opt_rms_grad_threshold_ = 2.0e-4;
    double opt_energy_threshold_ = 1.0e-6;
    double opt_disp_threshold_ = 3.0e-4;
    double opt_step_max_ = 0.3;

    // THC parameters
    int    thc_n_radial_ = 50;
    int    thc_lebedev_order_ = 194;
    int    thc_n_laplace_ = 12;
    double thc_rel_cutoff_ = 1.0e-7;
    double thc_sos_c_os_ = 1.3;
    bool   thc_b3a3_ = true;
    bool   thc_b3_ = true;
    bool   thc_a3_ = true;
    double thc_density_threshold_ = 0.0;
    int    thc_max_rank_ = 0;
    int    thc_rsvd_power_iter_ = 4;

    // DLPNO parameters (raw user input; -1 sentinels mean "use preset")
    std::string dlpno_preset_ = "normal";
    std::string dlpno_localizer_ = "pm";
    double dlpno_t_cut_pno_ = -1.0;
    double dlpno_t_cut_do_ = -1.0;
    double dlpno_t_cut_pairs_ = -1.0;
    double dlpno_t_cut_mkn_ = -1.0;
    double dlpno_t_cut_triples_ = -1.0;
    double dlpno_t_cut_tno_ = -1.0;
    double dlpno_pair_distance_cutoff_ = 15.0;
    int    dlpno_max_iter_ = 50;
    int    dlpno_diis_size_ = 6;
    int    dlpno_localizer_max_sweep_ = 200;
    double dlpno_localizer_conv_ = 1e-10;
    int    dlpno_lmp2_max_iter_ = 100;
    double dlpno_lmp2_conv_ = 1e-8;
    int    dlpno_sc_pno_iter_ = 1;
    bool   dlpno_pno_os_only_ = false;
    int    dlpno_verbose_ = 1;
    int    dlpno_cpu_threads_ = 0;  ///< Cap on OpenMP threads for DLPNO per-pair CPU loops; 0 = auto min(cores,64). Bounds OpenBLAS per-caller-thread buffer use (128 limit) on many-core machines.
    bool   dlpno_compute_density_ = false;  ///< Sub-phase 1+: build Λ + 1-RDM after MP2/CCSD
    bool   dlpno_lambda_full_dressing_ = false;  ///< Sub-phase 2X.2c: full Λ F-eff dressing

    // Multi-GPU
    int    num_gpus_ = 1;

    // CIS NTO active space (bt-PNO-STEOM Phase P0)
    int         cis_nto_n_root_cis_ = 0;          ///< 0 = auto (n_excited_states + 4)
    double      cis_nto_o_thresh_   = 1e-3;
    double      cis_nto_v_thresh_   = 1e-3;
    std::string cis_nto_weights_    = "uniform";
    int         cis_nto_verbose_    = 1;

    // IP-EOM-CCSD (bt-PNO-STEOM Phase P1)
    double ip_eom_ip_thresh_     = 0.80;
    int    ip_eom_safety_margin_ = 2;
    bool   ip_eom_followcis_     = true;
    double ip_eom_d_tol_         = 1e-5;
    double ip_eom_r_tol_         = 1e-7;
    int    ip_eom_max_iter_      = 500;
    int    ip_eom_verbose_       = 1;

    // EA-EOM-CCSD (bt-PNO-STEOM Phase P2)
    double ea_eom_ea_thresh_     = 0.80;
    int    ea_eom_safety_margin_ = 2;
    bool   ea_eom_followcis_     = true;
    double ea_eom_d_tol_         = 1e-5;
    double ea_eom_r_tol_         = 1e-7;
    int    ea_eom_max_iter_      = 500;
    int    ea_eom_verbose_       = 1;

    // STEOM-CCSD (bt-PNO-STEOM Phase P3)
    int    steom_n_root_cis_      = 0;     // 0 = auto (n_excited_states + 4)
    double steom_active_char_warn_ = 0.96; // warn if η < threshold
    double steom_d_tol_           = 1e-5;
    double steom_r_tol_           = 1e-7;
    int    steom_max_iter_        = 500;
    int    steom_verbose_         = 1;

    // Post-HF runtime statistics (set by drivers, read by post-HF summary).
    int    last_dmet_n_fragments_      = 0;
    int    last_dlpno_n_strong_        = 0;
    int    last_dlpno_n_weak_          = 0;
    int    last_dlpno_n_empty_         = 0;
    int    last_dlpno_n_triples_total_ = 0;
    int    last_dlpno_n_triples_active_ = 0;
    const int max_iter; ///< Maximum number of iterations
    int iter_; ///< Number of iterations
    real_t energy_difference_; ///< Energy difference between the current and the previous iteration
    const std::string run_type_; ///< Run type: "energy", "gradient", "optimize"
    const std::string optimizer_; ///< Optimizer: "bfgs"
    
    const std::vector<ShellTypeInfo> shell_type_infos; ///< Shell type info in the primitive shell list
    const std::vector<BasisRange> atom_to_basis_range; ///< Basis range for each atom (active basis: Spherical when use_spherical, else Cartesian)
    const std::vector<BasisRange> atom_to_basis_range_cart; ///< Always Cartesian-offset ranges (Molden [GTO] block writes Cart shell info)

    DeviceHostMemory<Atom> atoms; ///< Atoms
    DeviceHostMemory<PrimitiveShell> primitive_shells; ///< Primitive shells

    DeviceHostMemory<real_t> boys_grid; ///< grid values for the Boys function
    DeviceHostMatrix<real_t> overlap_matrix; ///< Overlap matrix
    DeviceHostMatrix<real_t> core_hamiltonian_matrix; ///< Core Hamiltonian matrix (kinetic energy + nuclear attraction)
    DeviceHostMemory<real_t> cgto_normalization_factors; ///< Normalization factors of the contracted Gauss functions


    DeviceHostMatrix<real_t> transform_matrix; ///< Transformation matrix

    real_t nuclear_repulsion_energy_; ///< Nuclear repulsion energy

    bool hasMatrixC_ = false; ///< Flag indicating whether the coefficient matrix has been computed. Once computed, it is set to true.

    // Post-HF methods
    PostHFMethod post_hf_method_; ///< Post-HF method
    real_t post_hf_energy_; ///< Post-HF energy
    int n_excited_states_; ///< Number of excited states to compute
    std::string adc2_solver_; ///< ADC(2) solver mode: schur_static, schur_omega, full
    std::string eom_mp2_solver_; ///< EOM-MP2 solver mode: full, schur
    std::string eom_cc2_solver_; ///< EOM-CC2 solver mode: auto, schur_static, schur_omega, full
    std::string spin_type_; ///< Spin type for excited states: "singlet" or "triplet"
    int num_frozen_core_; ///< Number of frozen core orbitals (0 = no frozen core)
    std::vector<real_t> excitation_energies_; ///< Excitation energies (CIS/EOM)
    std::vector<real_t> oscillator_strengths_; ///< Oscillator strengths (CIS/EOM)
    std::string excited_state_report_; ///< Formatted excited state report for final summary
    CISNTOResult cis_nto_result_;      ///< bt-PNO-STEOM Phase P0: state-averaged CIS NTO active space (empty unless post_hf_method=cis_nto)
    std::vector<real_t> last_cis_amplitudes_;  ///< DMET-STEOM root-targeting: raw CIS amplitudes from the last compute_cis_nto (RI)
    IPEOMResult  ip_eom_result_;       ///< bt-PNO-STEOM Phase P1: IP-EOM-CCSD roots per active occupied NTO (empty unless post_hf_method=ip_eom_ccsd)
    EAEOMResult  ea_eom_result_;       ///< bt-PNO-STEOM Phase P2: EA-EOM-CCSD roots per active virtual NTO (empty unless post_hf_method=ea_eom_ccsd)
    STEOMResult  steom_result_;        ///< bt-PNO-STEOM Phase P3: STEOM-CCSD excited states (empty unless post_hf_method=steom_ccsd)
    BTAmplitudes dlpno_bt_amplitudes_; ///< hybrid bt-PNO-STEOM P5b: DLPNO-CCSD T1/T2 in canonical MO (set during dlpno_steom_ccsd)
    bool         use_dlpno_amplitudes_ = false; ///< when true, IP/EA/STEOM impls consume dlpno_bt_amplitudes_ instead of canonical CCSD
    bool         collect_dlpno_bt_ = false;      ///< when true, DLPNOCCSD::compute_energy stores back-transformed canonical amplitudes
    DLPNOLMP2Result dlpno_res_;                  ///< stage B: converged DLPNO pair state (set when collect_dlpno_bt_)
    bool         use_dlpno_projected_eom_ = false; ///< stage B: IP/EA impls run the projected DLPNO operator
    bool         use_dlpno_native_eom_ = false; ///< stage B (a): IP impl runs the native per-pair DLPNO-IP-EOM σ
    bool         steom_share_barh_ = false;        ///< (A): IP/EA publish dressed bar-H, STEOM borrows (env GANSU_STEOM_SHARE_BARH=1)
    SteomBarHCache steom_barh_cache_;              ///< (A): shared dressed bar-H, owned across IP→EA→STEOM dispatch, freed by driver

    // ECP data (from Molecular)
    bool has_ecp_ = false;
    std::unordered_map<std::string, ElementECP> ecp_data_;

    // CCSD 1-RDM (correlation density, non-relaxed). Filled by CCSD_DENSITY method.
    std::vector<real_t> ccsd_1rdm_mo_; ///< [nao*nao] row-major, MO basis (incl. HF reference)
    std::vector<real_t> ccsd_1rdm_ao_; ///< [nao*nao] row-major, AO basis


    // for Diect SCF
    std::vector<ShellPairTypeInfo> shell_pair_type_infos;
    size_t num_primitive_shells;
    size_t num_primitive_shell_pairs;

    // for ERI (stored, RI, direct)
    std::unique_ptr<ERI> eri_method_; ///< ERI method

    // for int1e (hybrid, MD, OS)
    const std::string int1e_method; // int1e method

    // for int1e (hybrid, MD, OS)
    const std::string initial_guess_method_; // int1e method

    // Analysis flags
    const bool is_mulliken_analysis_; ///< Mulliken population analysis flag
    const bool is_mayer_bond_order_analysis_; ///< Mayer bond order analysis flag
    const bool is_wiberg_bond_order_analysis_; ///< Wiberg bond order analysis
    const bool is_export_molden_; ///< Export Molden file flag (canonical MOs)
    const bool is_export_lmo_molden_; ///< Export Pipek-Mezey localized occupied MOs as <basename>_lmo.molden

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
    virtual void compute_coefficient_matrix_impl()=0;




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

    virtual void reset_convergence() {} ///< Reset convergence method state for a new SCF cycle




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

    /// Export Pipek-Mezey localized occupied orbitals as a Molden file.
    /// Occupied block is replaced by LMOs (C_LMO = C_occ · U); virtual block
    /// retains the canonical orbitals. UHF/ROHF localise α and β separately.
    virtual void export_lmo_molden_file(const std::string& filename) = 0;


    /**
     * @brief Compute the gradient of the total electronic energy
     * @details This function evaluates the energy derivatives with respect to each nuclear coordinate (x, y, z) by summing the derivatives of the one-electron and two-electron integrals.
     * @details The result can be used for geometry optimization or force calculations.
     */
    virtual std::vector<double> compute_Energy_Gradient() = 0;

    /**
     * @brief Compute the analytic Hessian of the total electronic energy
     * @details Returns the second derivative matrix d²E/dR_i dR_j as a flat vector
     *          of size (3*num_atoms)^2 in row-major order.
     * @return Hessian matrix as flat vector
     */
    virtual std::vector<double> compute_Energy_Hessian() {
        throw std::runtime_error("Hessian not implemented for this method.");
    }

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
