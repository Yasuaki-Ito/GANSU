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
 * @brief  This function is used to calculate the rhf energy
 */

#include "rhf.hpp"
#include <memory>
#include "cphf_solver.hpp"
#include "multi_gpu_manager.hpp"   // for actual GPU count in post-HF summary
#include "progress.hpp"
#include "dlpno_localizer.hpp"     // for export_lmo_molden_file (PM localization)
#include <cassert>
#include "ao2mo.cuh"
#ifdef GANSU_MULTI_GPU
#include "multi_gpu_manager.hpp"
#endif

#include <limits> // numeric_limits<double>::max();
#include <iomanip> // std::setprecision
#include <filesystem>
#include <numeric>
#include "minao.hpp"

#include "utils.hpp" // THROW_EXCEPTION
#include "ecp_integrals.hpp"
namespace gansu{



/**
 * @brief Constructor of the RHF class
 * @param molecular Molecular
 * @param verbose Verbose mode
 * @details This function constructs the RHF class.
 * @details The molecular is given as an argument.
 */
RHF::RHF(const Molecular& molecular, const ParameterManager& parameters) : 
    HF(molecular, parameters),
    coefficient_matrix(num_basis, num_basis),
    density_matrix(num_basis, num_basis),
    orbital_energies(num_basis),
    initial_guess_method_(parameters.get<std::string>("initial_guess")),
    gbsfilename_(parameters.get<std::string>("gbsfilename")),
    fock_matrix(num_basis, num_basis)
{
    // check if the number of electrons is even (condition for restricted Hartree Fock method)
    if(molecular.get_num_alpha_spins() != molecular.get_num_beta_spins()){
        THROW_EXCEPTION("In restricted Hartree Fock method, the number of alpha and beta electrons should be equal.");
    }
    if(molecular.get_num_alpha_spins() > num_basis){
        THROW_EXCEPTION("The number of alpha or beta electrons should be less than or equal to the number of basis functions.");
    }

    // Set an algorithm to update the Fock matrix (default: Convergence_RHF_DIIS)
    const std::string convergence_method = parameters.get<std::string>("convergence_method");
    if(convergence_method == "damping"){ // damping algorithm
        const double damping_factor = parameters.get<double>("damping_factor"); // damping factor, default: 0.9
        set_convergence_method(std::make_unique<Convergence_RHF_Damping>(*this, damping_factor));
    }else if(convergence_method == "optimaldamping"){
        set_convergence_method(std::make_unique<Convergence_RHF_Damping>(*this));
    }else if(convergence_method == "diis"){ // DIIS algorithm
        const size_t DIIS_size = parameters.get<int>("diis_size"); // DIIS size (number of previous Fock matrices), default: 8
        const bool is_include_transform = parameters.get<bool>("diis_include_transform"); // include the transformation matrix in DIIS, default: false

        set_convergence_method(std::make_unique<Convergence_RHF_DIIS>(*this, DIIS_size, is_include_transform));
    }else if(convergence_method == "soscf"){ // Second-Order SCF (DIIS → SOSCF)
        const size_t DIIS_size = parameters.get<int>("diis_size");
        const double soscf_start = parameters.get<double>("soscf_start_threshold");
        set_convergence_method(std::make_unique<Convergence_RHF_SOSCF>(*this, DIIS_size, soscf_start));
    }else if(convergence_method == "adiis"){
        const size_t DIIS_size = parameters.get<int>("diis_size");
        set_convergence_method(std::make_unique<Convergence_RHF_ADIIS>(*this, ADIISMode::ADIIS, DIIS_size));
    }else if(convergence_method == "ediis"){
        const size_t DIIS_size = parameters.get<int>("diis_size");
        set_convergence_method(std::make_unique<Convergence_RHF_ADIIS>(*this, ADIISMode::EDIIS, DIIS_size));
    }else if(convergence_method == "aediis"){
        const size_t DIIS_size = parameters.get<int>("diis_size");
        set_convergence_method(std::make_unique<Convergence_RHF_ADIIS>(*this, ADIISMode::AEDIIS, DIIS_size));
    }else{
        THROW_EXCEPTION("Invalid convergence algorithm name: " + convergence_method);
    }

    // Set an algorithm for ERI calculation (default: ERI_Stored_RHF)
    const std::string eri_method = parameters.get<std::string>("eri_method");
    if(eri_method == "stored"){ // ERI matrices are stored in the device memory
        auto eri_stored = std::make_unique<ERI_Stored_RHF>(*this);
        // Set CCSD algorithm: 0=spatial-optimized, 1=spatial-naive, 2=spin-orbital
        if (parameters.contains("ccsd_algorithm")) {
            eri_stored->set_ccsd_algorithm(parameters.get<int>("ccsd_algorithm"));
        }
        set_eri_method(std::move(eri_stored));
    }else if(eri_method == "ri"){ // Resolution of Identity (RI) method
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename");
        BasisSet aux_basis = get_auxiliary_basis(molecular, auxiliary_gbsfilename);
        Molecular auxiliary_molecular(molecular.get_atoms(), aux_basis);
        if (molecular.get_use_spherical()) {
            auxiliary_molecular.set_use_spherical(true);
            std::cout << "[RI Spherical aux] nbf_cart = "
                      << auxiliary_molecular.get_num_basis_cart() << " → nbf_sph = "
                      << auxiliary_molecular.get_num_basis_sph() << std::endl;
        }
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
#ifdef GANSU_MULTI_GPU
        {
            auto& mgr = MultiGpuManager::instance();
            if (!mgr.num_devices()) {
                int num_gpus = parameters.get<int>("num_gpus");
                mgr.initialize(num_gpus);
            }
            std::cout << "[RI] " << mgr.num_devices() << " device(s)" << std::endl;
            // B-model routing (MPI_DESIGN.md "X model"):
            //  - Single GPU, single process: full-local ERI_RI_RHF (the distributed
            //    class's lightweight ctor leaves intermediate_matrix_B_ a 1x1 dummy,
            //    breaking analytical paths e.g. RI gradient).
            //  - MPI + post-HF: post-HF (build_B_mo → replicate_B_to_all_gpus) needs
            //    a FULL B on every rank, but the distributed class holds only this
            //    rank's aux slab → partial B → wrong integrals. So each rank uses the
            //    full-local ERI_RI_RHF (SCF redundant; parallelism comes from the
            //    post-HF IP/EA split + pair distribution). B is naux×nao² (~580 MB),
            //    not nvir⁴, so a full per-rank B is cheap.
            //  - Single-process multi-GPU, OR MPI pure-SCF (post_hf==none): use the
            //    distributed class (aux partitioned over devices / world ranks).
            const std::string post_hf = parameters.contains("post_hf_method")
                ? parameters.get<std::string>("post_hf_method") : std::string("none");
            const bool mpi_post_hf = mgr.is_mpi() && !post_hf.empty() && post_hf != "none";
            if ((mgr.num_devices() == 1 && !mgr.is_mpi()) || mpi_post_hf) {
                if (mpi_post_hf)
                    std::cout << "[RI] MPI post-HF (" << post_hf << "): full-local B per rank "
                                 "(redundant SCF; parallelism in post-HF stage)." << std::endl;
                set_eri_method(std::make_unique<ERI_RI_RHF>(*this, auxiliary_molecular));
            } else {
                auto eri = std::make_unique<ERI_RI_Distributed_RHF>(*this, auxiliary_molecular);
                eri->set_storage_mode(ERI_RI_Distributed_RHF::StorageMode::GPU_Resident);
                set_eri_method(std::move(eri));
            }
        }
#else
        set_eri_method(std::make_unique<ERI_RI_RHF>(*this, auxiliary_molecular));
#endif
    }else if(eri_method == "direct"){
        set_eri_method(std::make_unique<ERI_Direct_RHF>(*this));
    }else if(eri_method == "hash"){
        auto eri_hash = std::make_unique<ERI_Hash_RHF>(*this);
        if (parameters.contains("hash_fock_method")) {
            std::string hfm = parameters.get<std::string>("hash_fock_method");
            if (hfm == "compact") eri_hash->set_hash_fock_method(HashFockMethod::Compact);
            else if (hfm == "indexed") eri_hash->set_hash_fock_method(HashFockMethod::Indexed);
            else if (hfm == "fullscan") eri_hash->set_hash_fock_method(HashFockMethod::Fullscan);
            else THROW_EXCEPTION("Invalid hash_fock_method: " + hfm + ". Use compact, indexed, or fullscan.");
        }
        set_eri_method(std::move(eri_hash));
    }else if(eri_method == "direct_ri"){
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename");
        BasisSet aux_basis = get_auxiliary_basis(molecular, auxiliary_gbsfilename);
        Molecular auxiliary_molecular(molecular.get_atoms(), aux_basis);
        if (molecular.get_use_spherical()) {
            auxiliary_molecular.set_use_spherical(true);
            std::cout << "[RI Spherical aux] nbf_cart = "
                      << auxiliary_molecular.get_num_basis_cart() << " → nbf_sph = "
                      << auxiliary_molecular.get_num_basis_sph() << std::endl;
        }
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
#ifdef GANSU_MULTI_GPU
        {
            auto& mgr = MultiGpuManager::instance();
            if (!mgr.num_devices()) {
                int num_gpus = parameters.get<int>("num_gpus");
                mgr.initialize(num_gpus);
            }
            std::cout << "[Direct-RI] On-the-fly mode (" << mgr.num_devices() << " device(s))" << std::endl;
            auto eri = std::make_unique<ERI_RI_Distributed_RHF>(*this, auxiliary_molecular);
            eri->set_storage_mode(ERI_RI_Distributed_RHF::StorageMode::OnTheFly);
            set_eri_method(std::move(eri));
        }
#else
        set_eri_method(std::make_unique<ERI_RI_Direct_RHF>(*this, auxiliary_molecular));
#endif
    }else if(eri_method == "semi_direct_ri"){
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename");
        BasisSet aux_basis = get_auxiliary_basis(molecular, auxiliary_gbsfilename);
        Molecular auxiliary_molecular(molecular.get_atoms(), aux_basis);
        if (molecular.get_use_spherical()) {
            auxiliary_molecular.set_use_spherical(true);
            std::cout << "[RI Spherical aux] nbf_cart = "
                      << auxiliary_molecular.get_num_basis_cart() << " → nbf_sph = "
                      << auxiliary_molecular.get_num_basis_sph() << std::endl;
        }
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
#ifdef GANSU_MULTI_GPU
        {
            auto& mgr = MultiGpuManager::instance();
            if (!mgr.num_devices()) {
                int num_gpus = parameters.get<int>("num_gpus");
                mgr.initialize(num_gpus);
            }
            std::cout << "[Semi-Direct-RI] On-the-fly mode (" << mgr.num_devices() << " device(s))" << std::endl;
            auto eri = std::make_unique<ERI_RI_Distributed_RHF>(*this, auxiliary_molecular);
            eri->set_storage_mode(ERI_RI_Distributed_RHF::StorageMode::OnTheFly);
            set_eri_method(std::move(eri));
        }
#else
        set_eri_method(std::make_unique<ERI_RI_SemiDirect_RHF>(*this, auxiliary_molecular));
#endif
    }else if(eri_method == "hash_ri"){
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename");
        BasisSet aux_basis = get_auxiliary_basis(molecular, auxiliary_gbsfilename);
        Molecular auxiliary_molecular(molecular.get_atoms(), aux_basis);
        if (molecular.get_use_spherical()) {
            auxiliary_molecular.set_use_spherical(true);
            std::cout << "[RI Spherical aux] nbf_cart = "
                      << auxiliary_molecular.get_num_basis_cart() << " → nbf_sph = "
                      << auxiliary_molecular.get_num_basis_sph() << std::endl;
        }
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
        set_eri_method(std::make_unique<ERI_RI_Hash_RHF>(*this, auxiliary_molecular));
    }else{
        THROW_EXCEPTION("Invalid ERI method name: " + eri_method);
    }

    // Check if the selected ERI method supports post-HF methods
    if(!eri_method_->supports_post_hf_method(get_post_hf_method())){
        if(get_post_hf_method() != PostHFMethod::None)
            THROW_EXCEPTION("The selected ERI method does not support the selected post-HF method.");
    }

    // Set an algorithm for int1e calculation (default: ERI_Stored_RHF)
    const std::string int1e_method = parameters.get<std::string>("int1e_method");
    if(int1e_method == "md"){
        std::cout << "[INT1E] One electron integrals are computed using the MD method." << std::endl;
    }else if(int1e_method == "os"){
        std::cout << "[INT1E] One electron integrals are computed using the OS method." << std::endl;
    }else{ //Default hybrid
        std::cout << "[INT1E] One electron integrals are computed using the Hybrid method." << std::endl;
    }
}

/**
 * @brief Set a convergence method to update the Fock matrix
 * @param convergence_method Convergence_RHF
 * @details This function sets a convergence method to update the Fock matrix.
 */
void RHF::set_convergence_method(std::unique_ptr<Convergence_RHF> convergence_method) {
    this->convergence_method_ = std::move(convergence_method);
}

/**
 * @brief Set an ERI method to calculate the two-electron integrals
 * @param eri_method ERI
 * @details This function sets an ERI method to calculate the two-electron integrals.
 */
void RHF::set_eri_method(std::unique_ptr<ERI> eri_method) {
    this->eri_method_ = std::move(eri_method);
}


void RHF::precompute_eri_matrix(){
    PROFILE_FUNCTION();

    eri_method_->precomputation();
}


void RHF::post_process_after_scf() {
    PROFILE_FUNCTION();

    PostHFMethod post_hf_method = get_post_hf_method();
    if(post_hf_method == PostHFMethod::None){
        post_hf_energy_ = 0.0;
        return; // do nothing
    }

    // Report post-HF start
    report_progress("posthf", 0, 0, nullptr);

    if(post_hf_method == PostHFMethod::FCI){
        post_hf_energy_ = eri_method_->compute_fci_energy();
    }else if(post_hf_method == PostHFMethod::MP2){
        post_hf_energy_ = eri_method_->compute_mp2_energy();
    }else if(post_hf_method == PostHFMethod::THC_MP2){
        post_hf_energy_ = eri_method_->compute_thc_mp2_energy();
    }else if(post_hf_method == PostHFMethod::THC_SOS_MP2){
        post_hf_energy_ = eri_method_->compute_thc_sos_mp2_energy();
    }else if(post_hf_method == PostHFMethod::THC_SOS_ADC2){
        eri_method_->compute_thc_sos_adc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::SCS_MP2){
        post_hf_energy_ = eri_method_->compute_scs_mp2_energy();
    }else if(post_hf_method == PostHFMethod::SOS_MP2){
        post_hf_energy_ = eri_method_->compute_sos_mp2_energy();
    }else if(post_hf_method == PostHFMethod::LT_MP2){
        post_hf_energy_ = eri_method_->compute_lt_mp2_energy();
    }else if(post_hf_method == PostHFMethod::LT_SOS_MP2){
        post_hf_energy_ = eri_method_->compute_lt_sos_mp2_energy();
    }else if(post_hf_method == PostHFMethod::MP3){
        post_hf_energy_ = eri_method_->compute_mp3_energy();
    }else if(post_hf_method == PostHFMethod::MP4){
        post_hf_energy_ = eri_method_->compute_mp4_energy();
    }else if(post_hf_method == PostHFMethod::CC2){
        post_hf_energy_ = eri_method_->compute_cc2_energy();
    }else if(post_hf_method == PostHFMethod::CCSD){
        post_hf_energy_ = eri_method_->compute_ccsd_energy();
    }else if(post_hf_method == PostHFMethod::CCSD_T){
        post_hf_energy_ = eri_method_->compute_ccsd_t_energy();
    }else if(post_hf_method == PostHFMethod::CCSD_DENSITY){
        // compute_ccsd_density() runs CCSD internally and stores energy via
        // set_post_hf_energy(); avoid redundant CCSD solve.
        eri_method_->compute_ccsd_density();
    }else if(post_hf_method == PostHFMethod::CIS){
        eri_method_->compute_cis(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::ADC2){
        eri_method_->compute_adc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::SOS_ADC2){
        eri_method_->compute_sos_adc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::LT_SOS_ADC2){
        eri_method_->compute_sos_laplace_adc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::ADC2X){
        eri_method_->compute_adc2x(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_MP2){
        eri_method_->compute_eom_mp2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_CC2){
        eri_method_->compute_eom_cc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_CCSD){
        eri_method_->compute_eom_ccsd(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::DMET_CCSD){
        post_hf_energy_ = eri_method_->compute_dmet_ccsd();
    }else if(post_hf_method == PostHFMethod::DMET_CCSD_T){
        post_hf_energy_ = eri_method_->compute_dmet_ccsd_t();
    }else if(post_hf_method == PostHFMethod::DLPNO_MP2){
        post_hf_energy_ = eri_method_->compute_dlpno_mp2();
    }else if(post_hf_method == PostHFMethod::DLPNO_CCSD){
        post_hf_energy_ = eri_method_->compute_dlpno_ccsd();
    }else if(post_hf_method == PostHFMethod::DLPNO_CCSD_T){
        post_hf_energy_ = eri_method_->compute_dlpno_ccsd_t();
    }else if(post_hf_method == PostHFMethod::CIS_NTO){
        int n_cis = get_cis_nto_n_root_cis();
        if (n_cis <= 0) n_cis = get_n_excited_states() + 4;  // STEOM.md §7.3 default
        eri_method_->compute_cis_nto(n_cis);
    }else if(post_hf_method == PostHFMethod::IP_EOM_CCSD){
        eri_method_->compute_ip_eom_ccsd(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EA_EOM_CCSD){
        eri_method_->compute_ea_eom_ccsd(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::STEOM_CCSD){
        eri_method_->compute_steom_ccsd(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::DLPNO_STEOM_CCSD){
        eri_method_->compute_dlpno_steom_ccsd(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::DMET_STEOM){
        eri_method_->compute_dmet_steom_ccsd(get_n_excited_states());
    }else{
        THROW_EXCEPTION("Invalid post-HF method.");
    }

    // Report post-HF done
    report_progress("posthf", 1, 0, nullptr);
}


/**
 * @brief Function to guess the initial Fock matrix
 * @param density_matrix_a Density matrix (alpha-spin)
 * @param density_matrix_b Density matrix (beta-spin)
 * @param force_density Initialized by the precomputed density matrix
 * @details This function calculates the initial Fock matrix using the core Hamiltonian matrix.
 */
void RHF::guess_initial_fock_matrix(const real_t* density_matrix_a, const real_t* density_matrix_b, bool force_density) {
    PROFILE_FUNCTION();

    std::unique_ptr<InitialGuess_RHF> initial_guess; // the life time is only here since initial guess is performed only once

    if(force_density == true || initial_guess_method_ == "density"){ // Initialized by precomputed density matrices
        if(density_matrix_a == nullptr || density_matrix_b == nullptr){
            std::cerr << "The density matrix is not provided even though ``density'' is set to ``initial_guess_method'' or forced_density = true. The core Hamiltonian matrix is used instead." << std::endl;
            initial_guess = std::make_unique<InitialGuess_RHF_Core>(*this);
        }else{
            initial_guess = std::make_unique<InitialGuess_RHF_Density>(*this, density_matrix_a, density_matrix_b);
        }
    }else if(initial_guess_method_ == "core"){ // core Hamiltonian matrix
        initial_guess = std::make_unique<InitialGuess_RHF_Core>(*this);
    }else if(initial_guess_method_ == "gwh"){ // Generalized Wolfsberg-Helmholz (GWH) method
        initial_guess = std::make_unique<InitialGuess_RHF_GWH>(*this);
    }else if(initial_guess_method_ == "sad"){ // Superposition of Atomic Densities (SAD) method
        if(gbsfilename_.empty()){
            THROW_EXCEPTION("The basis set file is not specified for SAD initial guess method. Please specify the basis set file name by -gbsfilename option.");
        }
        // SAD spherical support: the guess builds the molecular density in
        // Cartesian and transforms it Cart→Sph (Phase 3c, see guess()).
        initial_guess = std::make_unique<InitialGuess_RHF_SAD>(*this);
    }else if(initial_guess_method_ == "minao"){ // MINAO (Minimal ANO) projection
        // MINAO projects a minimal ANO basis onto the AO basis; under spherical
        // basis the cross-overlap's calc axis is transformed Cart→Sph in guess().
        initial_guess = std::make_unique<InitialGuess_RHF_MINAO>(*this);
    }else{
        throw std::runtime_error("Invalid initial guess method: " + initial_guess_method_);
    }

    // Execute the initial guess method
    initial_guess->guess();

}

/**
 * @brief Function to calculate the coefficient matrix
 * @details This function calculates the coefficient matrix using the eigenvectors of the Fock matrix.
 */
void RHF::compute_coefficient_matrix_impl() {
    PROFILE_FUNCTION();

    // compute coefficient matrix C
    // The function computeEigenvectors performs the following operations:
    //   1. symmetrize the Fock matrix F' = X^T F X
    //   2. diagonalize the symmetrized Fock matrix F'C' = C'E
    //   3. obtain the coefficient matrix from the eigenvectors C = X C'
    gpu::computeCoefficientMatrix(
        fock_matrix.device_ptr(), 
        transform_matrix.device_ptr(),
        coefficient_matrix.device_ptr(),
        num_basis,
        orbital_energies.device_ptr()
    );

    if (verbose) {
        std::cout << "Orbital energies:" << std::endl;
        orbital_energies.toHost();
        for(size_t i=0; i<num_basis; i++){
            std::cout << orbital_energies[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "Coefficient matrix:" << std::endl;
        coefficient_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << coefficient_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}

/**
 * @brief Function to calculate the density matrix
 * @details This function calculates the density matrix using the coefficient matrix.
 */
void RHF::compute_density_matrix() {
    PROFILE_FUNCTION();

    gpu::computeDensityMatrix_RHF(
        coefficient_matrix.device_ptr(),
        density_matrix.device_ptr(),
        num_electrons,
        num_basis
    );
/*
    { // nan check
        density_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                if(std::isnan(density_matrix(i, j))){
                    THROW_EXCEPTION("Density matrix contains NaN value at (" + std::to_string(i) + ", " + std::to_string(j) + ").");
                }
            }
            std::cout << std::endl;
        }
    }
*/
    if (verbose) {
        std::cout << "Density matrix:" << std::endl;
        density_matrix.toHost();
        for(size_t i=0; i<num_basis; i++){
            for(size_t j=0; j<num_basis; j++){
                std::cout << density_matrix(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}


/**
 * @brief Function to calculate the Fock matrix
 * @details This function calculates the Fock matrix using the core Hamiltonian matrix, the density matrix, and the two-electron integrals.
 */
 void RHF::compute_fock_matrix(){
    PROFILE_FUNCTION();

    eri_method_->compute_fock_matrix();
}


/**
 * @brief Function to calculate the energy
 * @details This function calculates the energy using the core Hamiltonian matrix, the density matrix, and the Fock matrix.
 */
void RHF::compute_energy(){
    PROFILE_FUNCTION();

    energy_ = gpu::computeEnergy_RHF(
        density_matrix.device_ptr(),
        core_hamiltonian_matrix.device_ptr(),
        fock_matrix.device_ptr(),
        num_basis
    );

    if(verbose){
        std::cout << "Energy: " << energy_ << std::endl;
        std::cout << "Total energy: " << get_total_energy() << std::endl;
    }
}

/**
 * @brief Update the Fock matrix
 * @details This function updates the Fock matrix
 */
void RHF::update_fock_matrix(){
    PROFILE_FUNCTION();

    convergence_method_->get_new_fock_matrix();
}

void RHF::export_density_matrix(real_t* density_matrix_a, real_t* density_martix_b, const int num_basis) {
    if(num_basis != this->num_basis){
        throw std::runtime_error("The number of basis functions is different.");
    }
    if(density_matrix_a == nullptr || density_martix_b == nullptr){
        throw std::runtime_error("The density matrix is not provided.");
    }
    // copy the density matrix to the host memory
    density_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        for(size_t j=0; j<num_basis; j++){
            density_matrix_a[i*num_basis + j] = density_matrix(i, j) / 2.0;
            density_martix_b[i*num_basis + j] = density_matrix(i, j) / 2.0;
        }
    }
}


/**
 * @brief Compute the analytical Hessian (skeleton + CPHF response)
 */
std::vector<double> RHF::compute_Energy_Hessian() {
    PROFILE_FUNCTION();

    // Spherical basis: the analytical skeleton Hessian below differentiates the
    // Cartesian Gaussian integrals, so it must contract the Cartesian-expressed
    // D / C / ε (back-transform, exactly like the stored gradient). The CPHF
    // response part is computed from the spherical Fock/overlap derivatives
    // (finite difference through the spherical SCF) and spherical MO ERIs, so it
    // runs natively in the spherical active basis without any back-transform.
    const bool hess_sph = get_use_spherical();

    int num_atoms_val = static_cast<int>(atoms.size());
    int ndim = 3 * num_atoms_val;

    atoms.toHost();
    std::vector<double> orig(3 * num_atoms_val);
    for (int i = 0; i < num_atoms_val; i++) {
        orig[3*i]   = atoms[i].coordinate.x;
        orig[3*i+1] = atoms[i].coordinate.y;
        orig[3*i+2] = atoms[i].coordinate.z;
    }

    // --- Skeleton Hessian ---
    if (verbose) std::cout << "  Computing skeleton Hessian..." << std::endl;
    std::vector<double> skel_hessian;
    if (hess_sph) {
        // === Spherical skeleton Hessian via Cartesian back-transform ===
        const int nc = get_num_basis_cart();
        const int ns = num_basis;
        const size_t nc2 = (size_t)nc * nc;
        const auto& sht  = get_shell_types();
        const auto& offc = get_shell_offsets_cart();
        const auto& offs = get_shell_offsets_sph();

        real_t *d_D_cart=nullptr, *d_C_pack=nullptr, *d_C_cart=nullptr, *d_eps_cart=nullptr;
        gansu::tracked_cudaMalloc(&d_D_cart,  nc2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_C_pack,  (size_t)nc * ns * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_C_cart,  nc2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_eps_cart, (size_t)nc * sizeof(real_t));

        // D_cart = Uᵀ D_sph U ;  C_pack[nc × ns] = Uᵀ C_sph
        spherical::transform_matrix_sph_to_cart_device(
            density_matrix.device_ptr(), d_D_cart, sht, offc, offs);
        spherical::transform_coeff_sph_to_cart_device(
            coefficient_matrix.device_ptr(), d_C_pack, ns, sht, offc, offs);

        // Pad C to [nc × nc] (extra MO columns 0) and ε to [nc] (extra 0). Only
        // occupied MOs (< ns) enter the skeleton Hessian, so padding is inert.
        cudaMemset(d_C_cart, 0, nc2 * sizeof(real_t));
        cudaMemcpy2D(d_C_cart, (size_t)nc * sizeof(real_t),
                     d_C_pack, (size_t)ns * sizeof(real_t),
                     (size_t)ns * sizeof(real_t), (size_t)nc, cudaMemcpyDeviceToDevice);
        cudaMemset(d_eps_cart, 0, (size_t)nc * sizeof(real_t));
        cudaMemcpy(d_eps_cart, orbital_energies.device_ptr(),
                   (size_t)ns * sizeof(real_t), cudaMemcpyDeviceToDevice);

        skel_hessian = gpu::computeSkeletonHessian_RHF(
            shell_type_infos, shell_pair_type_infos, atoms.device_ptr(),
            d_D_cart, d_C_cart, d_eps_cart,
            primitive_shells.device_ptr(), boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()), nc, num_electrons, verbose);

        gansu::tracked_cudaFree(d_D_cart);
        gansu::tracked_cudaFree(d_C_pack);
        gansu::tracked_cudaFree(d_C_cart);
        gansu::tracked_cudaFree(d_eps_cart);
    } else {
        skel_hessian = gpu::computeSkeletonHessian_RHF(
            shell_type_infos,
            shell_pair_type_infos,
            atoms.device_ptr(),
            density_matrix.device_ptr(),
            coefficient_matrix.device_ptr(),
            orbital_energies.device_ptr(),
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()),
            num_basis,
            num_electrons,
            verbose
        );
    }

    // --- CPHF Response Hessian ---
    if (verbose) std::cout << "  Computing CPHF response Hessian..." << std::endl;

    const int nao = num_basis;
    const int nocc = num_electrons / 2;
    const int nvir = nao - nocc;
    const int nmo = nao;
    const int n_pert = ndim;

    // Save converged C, ε, D
    std::vector<double> h_C(nao * nao), h_eps(nao), h_D(nao * nao);
    cudaMemcpy(h_C.data(), coefficient_matrix.device_ptr(), nao * nao * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eps.data(), orbital_energies.device_ptr(), nao * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D.data(), density_matrix.device_ptr(), nao * nao * sizeof(double), cudaMemcpyDeviceToHost);

    // --- Step 1: Compute h1ao and s1ao via finite difference ---
    // h1ao[pert] = dF/dR at fixed D, s1ao[pert] = dS/dR
    if (verbose) std::cout << "  Step 1: Computing Fock/overlap derivatives (finite difference)..." << std::endl;

    std::vector<std::vector<double>> h1ao(n_pert, std::vector<double>(nao * nao));
    std::vector<std::vector<double>> s1ao(n_pert, std::vector<double>(nao * nao));

    const double h_fd = 1e-4;

    for (int coord = 0; coord < n_pert; coord++) {
        int aidx = coord / 3;
        int dir = coord % 3;

        auto get_F_S = [&](double delta) -> std::pair<std::vector<double>, std::vector<double>> {
            // Set displaced geometry
            atoms.toHost();
            for (int i = 0; i < num_atoms_val; i++) {
                atoms[i].coordinate.x = orig[3 * i];
                atoms[i].coordinate.y = orig[3 * i + 1];
                atoms[i].coordinate.z = orig[3 * i + 2];
            }
            if (dir == 0) atoms[aidx].coordinate.x += delta;
            else if (dir == 1) atoms[aidx].coordinate.y += delta;
            else atoms[aidx].coordinate.z += delta;
            atoms.toDevice();

            // Update primitive shell coordinates
            primitive_shells.toHost();
            for (size_t i = 0; i < primitive_shells.size(); i++) {
                int ai2 = primitive_shells[i].atom_index;
                primitive_shells[i].coordinate = atoms.host_ptr()[ai2].coordinate;
            }
            primitive_shells.toDevice();

            // Rebuild integrals at displaced geometry
            compute_core_hamiltonian_matrix();
            precompute_eri_matrix();

            // Restore density and build Fock at displaced geometry with original D
            cudaMemcpy(density_matrix.device_ptr(), h_D.data(), nao * nao * sizeof(double), cudaMemcpyHostToDevice);
            compute_fock_matrix();

            std::vector<double> F(nao * nao), S(nao * nao);
            cudaMemcpy(F.data(), fock_matrix.device_ptr(), nao * nao * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(S.data(), overlap_matrix.device_ptr(), nao * nao * sizeof(double), cudaMemcpyDeviceToHost);
            return {F, S};
        };

        auto [Fp, Sp] = get_F_S(+h_fd);
        auto [Fm, Sm] = get_F_S(-h_fd);

        for (int k = 0; k < nao * nao; k++) {
            h1ao[coord][k] = (Fp[k] - Fm[k]) / (2.0 * h_fd);
            s1ao[coord][k] = (Sp[k] - Sm[k]) / (2.0 * h_fd);
        }
    }

    // Restore original geometry and integrals
    atoms.toHost();
    for (int i = 0; i < num_atoms_val; i++) {
        atoms[i].coordinate.x = orig[3 * i];
        atoms[i].coordinate.y = orig[3 * i + 1];
        atoms[i].coordinate.z = orig[3 * i + 2];
    }
    atoms.toDevice();
    primitive_shells.toHost();
    for (size_t i = 0; i < primitive_shells.size(); i++) {
        int ai2 = primitive_shells[i].atom_index;
        primitive_shells[i].coordinate = atoms.host_ptr()[ai2].coordinate;
    }
    primitive_shells.toDevice();
    compute_core_hamiltonian_matrix();
    precompute_eri_matrix();
    cudaMemcpy(density_matrix.device_ptr(), h_D.data(), nao * nao * sizeof(double), cudaMemcpyHostToDevice);
    compute_fock_matrix();

    // --- Step 2: Transform h1ao, s1ao to MO basis and build CPHF RHS ---
    if (verbose) std::cout << "  Step 2: Building CPHF RHS..." << std::endl;

    // Helper: C^T * M * C  (all nao × nao, on CPU)
    auto ao2mo_matrix = [&](const std::vector<double>& M_ao) -> std::vector<double> {
        std::vector<double> temp(nao * nao, 0.0), M_mo(nao * nao, 0.0);
        // temp = M_ao * C
        for (int mu = 0; mu < nao; mu++)
            for (int q = 0; q < nao; q++) {
                double s = 0.0;
                for (int nu = 0; nu < nao; nu++)
                    s += M_ao[mu * nao + nu] * h_C[nu * nao + q];
                temp[mu * nao + q] = s;
            }
        // M_mo = C^T * temp
        for (int p = 0; p < nao; p++)
            for (int q = 0; q < nao; q++) {
                double s = 0.0;
                for (int mu = 0; mu < nao; mu++)
                    s += h_C[mu * nao + p] * temp[mu * nao + q];
                M_mo[p * nao + q] = s;
            }
        return M_mo;
    };

    // Compute s1oo (occ-occ overlap derivative in MO) for each perturbation
    std::vector<std::vector<double>> s1oo(n_pert);    // nocc × nocc
    std::vector<std::vector<double>> h1_mo(n_pert);   // nao × nao
    std::vector<std::vector<double>> s1_mo(n_pert);   // nao × nao

    // CPHF RHS: rhs[pert][i*nvir + a] = -(F^x_MO[a_mo, i] - ε_i * S^x_MO[a_mo, i])
    std::vector<double> h_rhs(n_pert * nocc * nvir, 0.0);

    for (int pert = 0; pert < n_pert; pert++) {
        h1_mo[pert] = ao2mo_matrix(h1ao[pert]);
        s1_mo[pert] = ao2mo_matrix(s1ao[pert]);

        // Extract occ-occ block of s1_mo
        s1oo[pert].resize(nocc * nocc);
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                s1oo[pert][i * nocc + j] = s1_mo[pert][i * nao + j];

        // Build RHS (vir-occ block)
        for (int i = 0; i < nocc; i++) {
            for (int a = 0; a < nvir; a++) {
                int a_mo = nocc + a;
                double F_ai = h1_mo[pert][a_mo * nao + i];
                double S_ai = s1_mo[pert][a_mo * nao + i];
                h_rhs[pert * nocc * nvir + i * nvir + a] = -(F_ai - h_eps[i] * S_ai);
            }
        }
    }

    // --- Step 2b: Add occ-occ density response correction to CPHF RHS ---
    // The CPHF includes the 2e response from the occ-occ density change
    // D_oo = -2 C_occ s1oo C_occ^T. We compute G(D_oo) and add its vir-occ
    // MO projection to the RHS: rhs[ai] -= (C^T G(D_oo) C)[a,i]
    {
        real_t* d_Doo = nullptr;
        real_t* d_Goo = nullptr;
        tracked_cudaMalloc(&d_Doo, nao * nao * sizeof(real_t));
        tracked_cudaMalloc(&d_Goo, nao * nao * sizeof(real_t));

        for (int pert = 0; pert < n_pert; pert++) {
            // D_oo = -2 C_occ s1oo C_occ^T
            std::vector<double> D_oo(nao * nao, 0.0);
            for (int mu = 0; mu < nao; mu++)
                for (int nu = 0; nu < nao; nu++) {
                    double s = 0.0;
                    for (int i = 0; i < nocc; i++)
                        for (int j = 0; j < nocc; j++)
                            s += h_C[mu * nao + i] * s1oo[pert][i * nocc + j] * h_C[nu * nao + j];
                    D_oo[mu * nao + nu] = -2.0 * s;
                }

            // G_oo = JK(D_oo) via ERI-method-native response
            cudaMemcpy(d_Doo, D_oo.data(), nao * nao * sizeof(double), cudaMemcpyHostToDevice);
            eri_method_->compute_jk_response(d_Doo, d_Goo, nao);
            std::vector<double> G_oo(nao * nao);
            cudaMemcpy(G_oo.data(), d_Goo, nao * nao * sizeof(double), cudaMemcpyDeviceToHost);

            // G_oo_mo = C^T G_oo C, then correct RHS
            std::vector<double> G_oo_mo = ao2mo_matrix(G_oo);
            for (int i = 0; i < nocc; i++)
                for (int a = 0; a < nvir; a++) {
                    int a_mo = nocc + a;
                    h_rhs[pert * nocc * nvir + i * nvir + a] -= G_oo_mo[a_mo * nao + i];
                }
        }

        tracked_cudaFree(d_Doo);
        tracked_cudaFree(d_Goo);
    }

    // --- Step 3: AO→MO ERI transform ---
    if (verbose) std::cout << "  Step 3: AO→MO ERI transform..." << std::endl;

    real_t* d_eri_mo = eri_method_->build_mo_eri(coefficient_matrix.device_ptr(), nao);

    // --- Step 4: Solve CPHF ---
    if (verbose) std::cout << "  Step 4: Solving CPHF equations..." << std::endl;

    real_t* d_rhs = nullptr;
    real_t* d_U = nullptr;
    tracked_cudaMalloc(&d_rhs, n_pert * nocc * nvir * sizeof(real_t));
    tracked_cudaMalloc(&d_U, n_pert * nocc * nvir * sizeof(real_t));
    cudaMemcpy(d_rhs, h_rhs.data(), n_pert * nocc * nvir * sizeof(real_t), cudaMemcpyHostToDevice);

    CPHFOperator cphf_op(d_eri_mo, orbital_energies.device_ptr(), nocc, nvir, nmo);
    solve_cphf(cphf_op, d_rhs, d_U, n_pert);

    // Download CPHF solution
    std::vector<double> h_U(n_pert * nocc * nvir);
    cudaMemcpy(h_U.data(), d_U, n_pert * nocc * nvir * sizeof(real_t), cudaMemcpyDeviceToHost);

    // --- Step 5: Build response quantities ---
    if (verbose) std::cout << "  Step 5: Building response quantities..." << std::endl;

    // For each perturbation y, build:
    // mo1[p,i]: vir-occ = U (CPHF solution), occ-occ = -½ s1oo
    // dm1[μ,ν] = Σ_p Σ_i C[μ,p] mo1[p,i] C[ν,i]  (one-sided response density)
    // dm1e[μ,ν] = Σ_p Σ_i C[μ,p] mo1[p,i] ε_i C[ν,i]  (energy-weighted)
    // D1 = 2*(dm1 + dm1^T)  (full first-order density, symmetric)
    // vhf1 = G(D1) via Fock build with zero h_core
    // mo_e1[i,j] = (C^T (h1ao + vhf1) C)[i,j] + ε_i * mo1_oo[i,j]  (Lagrange multiplier)

    std::vector<std::vector<double>> dm1_all(n_pert, std::vector<double>(nao * nao, 0.0));
    std::vector<std::vector<double>> dm1e_all(n_pert, std::vector<double>(nao * nao, 0.0));
    std::vector<std::vector<double>> mo_e1_all(n_pert, std::vector<double>(nocc * nocc, 0.0));

    // Temporary GPU buffers for vhf1 computation
    real_t* d_D1 = nullptr;
    real_t* d_vhf1 = nullptr;
    tracked_cudaMalloc(&d_D1, nao * nao * sizeof(real_t));
    tracked_cudaMalloc(&d_vhf1, nao * nao * sizeof(real_t));

    for (int pert = 0; pert < n_pert; pert++) {
        // Build mo1_MO (nmo × nocc)
        std::vector<double> mo1(nmo * nocc, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int a = 0; a < nvir; a++)
                mo1[(nocc + a) * nocc + i] = h_U[pert * nocc * nvir + i * nvir + a];
        for (int j = 0; j < nocc; j++)
            for (int i = 0; i < nocc; i++)
                mo1[j * nocc + i] = -0.5 * s1oo[pert][j * nocc + i];

        // temp[μ,i] = Σ_p C[μ,p] * mo1[p,i]  (= mo1_AO, in AO representation)
        std::vector<double> temp_mi(nao * nocc, 0.0);
        for (int mu = 0; mu < nao; mu++)
            for (int i = 0; i < nocc; i++) {
                double s = 0.0;
                for (int p = 0; p < nmo; p++)
                    s += h_C[mu * nao + p] * mo1[p * nocc + i];
                temp_mi[mu * nocc + i] = s;
            }

        // dm1[μ,ν] = Σ_i temp[μ,i] * C[ν,i]
        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu < nao; nu++) {
                double s = 0.0;
                for (int i = 0; i < nocc; i++)
                    s += temp_mi[mu * nocc + i] * h_C[nu * nao + i];
                dm1_all[pert][mu * nao + nu] = s;
            }

        // dm1e[μ,ν] = Σ_i temp[μ,i] * ε_i * C[ν,i]
        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu < nao; nu++) {
                double s = 0.0;
                for (int i = 0; i < nocc; i++)
                    s += temp_mi[mu * nocc + i] * h_eps[i] * h_C[nu * nao + i];
                dm1e_all[pert][mu * nao + nu] = s;
            }

        // D1 = 2*(dm1 + dm1^T) for vhf1 computation
        std::vector<double> D1(nao * nao);
        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu < nao; nu++)
                D1[mu * nao + nu] = 2.0 * (dm1_all[pert][mu * nao + nu] + dm1_all[pert][nu * nao + mu]);

        // vhf1 = G(D1) via ERI-method-native response
        cudaMemcpy(d_D1, D1.data(), nao * nao * sizeof(double), cudaMemcpyHostToDevice);
        eri_method_->compute_jk_response(d_D1, d_vhf1, nao);
        std::vector<double> vhf1(nao * nao);
        cudaMemcpy(vhf1.data(), d_vhf1, nao * nao * sizeof(double), cudaMemcpyDeviceToHost);

        // F_tot = h1ao + vhf1 (total first-order Fock in AO)
        std::vector<double> F_tot(nao * nao);
        for (int k = 0; k < nao * nao; k++)
            F_tot[k] = h1ao[pert][k] + vhf1[k];

        // F_tot_MO = C^T F_tot C
        std::vector<double> F_tot_mo = ao2mo_matrix(F_tot);

        // mo_e1[i,j] = F_tot_MO[i,j] - 0.5*(ε_i + ε_j)*s1oo[i,j]
        // Derived from d/dR(F C = S C ε): ε^(1)_{ij} = F^(1)_{ij} - 0.5*(ε_i+ε_j)*S^(1)_{ij}
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                mo_e1_all[pert][i * nocc + j] = F_tot_mo[i * nao + j]
                    - 0.5 * (h_eps[i] + h_eps[j]) * s1oo[pert][i * nocc + j];
    }

    // --- Step 6: Assemble response Hessian ---
    if (verbose) std::cout << "  Step 6: Assembling response Hessian..." << std::endl;

    // resp[x,y] = 4 * <h1ao_x, dm1_y>       ... Fock derivative × response density
    //           - 4 * <s1ao_x, dm1e_y>       ... overlap derivative × energy-weighted response density
    //           - 2 * <s1oo_x, mo_e1_y>      ... overlap occ-occ × Lagrange multiplier
    std::vector<double> resp_hessian(ndim * ndim, 0.0);

    for (int x = 0; x < n_pert; x++) {
        for (int y = 0; y < n_pert; y++) {
            double term1 = 0.0, term2 = 0.0, term3 = 0.0;

            for (int k = 0; k < nao * nao; k++) {
                term1 += h1ao[x][k] * dm1_all[y][k];
                term2 += s1ao[x][k] * dm1e_all[y][k];
            }
            for (int k = 0; k < nocc * nocc; k++)
                term3 += s1oo[x][k] * mo_e1_all[y][k];

            resp_hessian[x * ndim + y] = 4.0 * term1 - 4.0 * term2 - 2.0 * term3;
        }
    }

    // Symmetrize response
    for (int i = 0; i < ndim; i++)
        for (int j = i + 1; j < ndim; j++)
            resp_hessian[i * ndim + j] = resp_hessian[j * ndim + i] =
                0.5 * (resp_hessian[i * ndim + j] + resp_hessian[j * ndim + i]);

    // Full analytical Hessian = skeleton + response
    std::vector<double> hessian(ndim * ndim);
    for (int k = 0; k < ndim * ndim; k++)
        hessian[k] = skel_hessian[k] + resp_hessian[k];

    // Clean up GPU memory
    tracked_cudaFree(d_eri_mo);
    tracked_cudaFree(d_rhs);
    tracked_cudaFree(d_U);
    tracked_cudaFree(d_D1);
    tracked_cudaFree(d_vhf1);

    return hessian;
}

/**
 * @brief Compute the gradient of the total electronic energy
 */
std::vector<double> RHF::compute_Energy_Gradient() {
    PROFILE_FUNCTION();

    // Spherical-basis gradient.  The stored/Direct HF gradient is handled
    // exactly by a Cartesian back-transform (below): D_cart = Uᵀ D_sph U and
    // C_cart = Uᵀ C_sph, then the existing Cartesian gradient kernels give dE/dR
    // (U is geometry-independent and, for exact ERIs, E_sph = E_cart). The RI HF
    // gradient is spherical via the Γ back-transform path (single- and multi-GPU,
    // Phase 3d/3e). Only the MP2 gradient (nao⁴ 2-PDM) is still Cartesian-only —
    // guard it (its Cartesian path is itself not yet validated).
    const bool grad_sph = get_use_spherical();

    if (post_hf_method_ == PostHFMethod::MP2) {
        if (grad_sph)
            THROW_EXCEPTION("MP2 gradient does not yet support spherical basis "
                "(--use_spherical 1). Run in Cartesian (omit --use_spherical).");
        // MP2 gradient uses computeEnergyGradient_general with:
        //   1-electron terms: P_relaxed (HF + MP2 unrelaxed + Z-vector)
        //   Overlap term: W_MP2
        //   2-electron term: D_HF (the ERI derivative kernel computes Γ=D⊗D which is the HF 2-PDM)
        // Plus a separate non-separable 2-PDM contribution (Γ^MP2) — TODO
        const int nao = num_basis;
        const size_t nao2 = (size_t)nao * nao;
        const size_t nao4 = nao2 * nao2;

        real_t* d_P_relaxed = nullptr;   // 1-el: D_HF + dm1_relaxed (with z-vector)
        real_t* d_P_unrelaxed = nullptr; // 2-el: D_HF + dm1_unrelaxed (without z-vector)
        real_t* d_W_mp2 = nullptr;
        real_t* d_Gamma_4idx = nullptr;
        gansu::tracked_cudaMalloc(&d_P_relaxed, nao2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_P_unrelaxed, nao2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_W_mp2, nao2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_Gamma_4idx, nao4 * sizeof(real_t));

        eri_method_->compute_mp2_effective_densities(d_P_relaxed, d_W_mp2, d_Gamma_4idx, d_P_unrelaxed);

        // MP2 gradient:
        //   1-el:     P_relaxed (with z-vector, for kinetic + nuclear attraction)
        //   2-el sep: P_unrelaxed (without z-vector, for separable 2-PDM)
        //   gamma:    Γ^T2 (non-separable T2 cumulant)
        //   overlap:  W (energy-weighted density with z-vector + Lagrangian)
        auto gradient = gpu::computeEnergyGradient_general(
            shell_type_infos, shell_pair_type_infos,
            atoms.device_ptr(),
            d_P_relaxed,                    // 1-electron: relaxed density
            d_W_mp2,                        // overlap: energy-weighted density
            d_P_unrelaxed,                  // 2-electron: UNRELAXED (no z-vector)
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()),
            num_basis, verbose,
            d_Gamma_4idx                    // non-separable 2-PDM: Γ^T2
        );

        gansu::tracked_cudaFree(d_P_relaxed);
        gansu::tracked_cudaFree(d_P_unrelaxed);
        gansu::tracked_cudaFree(d_W_mp2);
        gansu::tracked_cudaFree(d_Gamma_4idx);
        return gradient;
    }

    // HF gradient: dispatch to the analytical RI path when the active ERI method
    // exposes it (currently ERI_RI; ERI_RI_Distributed_RHF added in Phase 4).
    std::vector<double> gradient;
    if (eri_method_->supports_ri_gradient()) {
        // Spherical RI gradient: compute_ri_gradient dispatches internally to
        // the back-transform path (Γ assembled in the spherical metric, then
        // back-transformed to Cartesian for the derivative kernels). Both the
        // single-GPU (compute_ri_gradient_spherical) and multi-GPU distributed
        // (compute_ri_gradient_spherical_distributed) paths support spherical.
        gradient = eri_method_->compute_ri_gradient(
            density_matrix.device_ptr(),
            coefficient_matrix.device_ptr(),
            orbital_energies.device_ptr(),
            num_electrons);
    } else if (grad_sph) {
        // === Spherical stored/Direct HF gradient via Cartesian back-transform ===
        const int nc = get_num_basis_cart();
        const int ns = num_basis;                 // spherical AO count (= #MOs)
        const size_t nc2 = (size_t)nc * nc;
        const auto& sht  = get_shell_types();
        const auto& offc = get_shell_offsets_cart();
        const auto& offs = get_shell_offsets_sph();

        real_t *d_D_cart=nullptr, *d_C_pack=nullptr, *d_C_cart=nullptr, *d_eps_cart=nullptr;
        gansu::tracked_cudaMalloc(&d_D_cart,  nc2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_C_pack,  (size_t)nc * ns * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_C_cart,  nc2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_eps_cart, (size_t)nc * sizeof(real_t));

        // D_cart = Uᵀ D_sph U ;  C_pack[nc × ns] = Uᵀ C_sph
        spherical::transform_matrix_sph_to_cart_device(
            density_matrix.device_ptr(), d_D_cart, sht, offc, offs);
        spherical::transform_coeff_sph_to_cart_device(
            coefficient_matrix.device_ptr(), d_C_pack, ns, sht, offc, offs);

        // Pad C to [nc × nc] (extra MO columns 0) and eps to [nc] (extra 0).
        // The gradient only uses occupied MOs (< ns), so padding is inert.
        cudaMemset(d_C_cart, 0, nc2 * sizeof(real_t));
        cudaMemcpy2D(d_C_cart, (size_t)nc * sizeof(real_t),
                     d_C_pack, (size_t)ns * sizeof(real_t),
                     (size_t)ns * sizeof(real_t), (size_t)nc, cudaMemcpyDeviceToDevice);
        cudaMemset(d_eps_cart, 0, (size_t)nc * sizeof(real_t));
        cudaMemcpy(d_eps_cart, orbital_energies.device_ptr(),
                   (size_t)ns * sizeof(real_t), cudaMemcpyDeviceToDevice);

        gradient = gpu::computeEnergyGradient_RHF(
            shell_type_infos, shell_pair_type_infos, atoms.device_ptr(),
            d_D_cart, d_C_cart, d_eps_cart,
            primitive_shells.device_ptr(), boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()), nc, num_electrons, verbose);

        gansu::tracked_cudaFree(d_D_cart);
        gansu::tracked_cudaFree(d_C_pack);
        gansu::tracked_cudaFree(d_C_cart);
        gansu::tracked_cudaFree(d_eps_cart);
    } else {
        gradient = gpu::computeEnergyGradient_RHF(
            shell_type_infos,
            shell_pair_type_infos,
            atoms.device_ptr(),
            density_matrix.device_ptr(),
            coefficient_matrix.device_ptr(),
            orbital_energies.device_ptr(),
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()),
            num_basis,
            num_electrons,
            verbose
        );
    }

    // Add ECP gradient contribution if present
    if (has_ecp_) {
        density_matrix.toHost();
        primitive_shells.toHost();
        cgto_normalization_factors.toHost();
        atoms.toHost();

        if (get_use_spherical()) {
            // ECP integral derivatives are built in the Cartesian primitive
            // basis, so contract them with the Cartesian density D_cart = Uᵀ D_sph U.
            const int nc = get_num_basis_cart();
            std::vector<real_t> D_cart((size_t)nc * nc);
            spherical::transform_matrix_sph_to_cart(
                density_matrix.host_ptr(), D_cart.data(),
                get_shell_types(), get_shell_offsets_cart(), get_shell_offsets_sph());
            ecp_integral::compute_ecp_gradient(
                primitive_shells.host_ptr(), primitive_shells.size(),
                cgto_normalization_factors.host_ptr(),
                nc,
                atoms.host_ptr(), atoms.size(),
                ecp_data_,
                D_cart.data(),
                gradient.data());
        } else {
            ecp_integral::compute_ecp_gradient(
                primitive_shells.host_ptr(), primitive_shells.size(),
                cgto_normalization_factors.host_ptr(),
                num_basis,
                atoms.host_ptr(), atoms.size(),
                ecp_data_,
                density_matrix.host_ptr(),
                gradient.data());
        }
    }

    return gradient;
}





/**
 * @brief Print the results of the SCF procedure
 * @details This function prints the results of the SCF procedure.
 */
void RHF::report() {

    HF::report(); // prints the information of the input molecular and basis set

    if(is_mulliken_analysis_){
        std::cout << std::endl;
        std::cout << "[Mulliken population]" << std::endl;
        const auto& mulliken_population = analyze_mulliken_population();
        for(size_t i=0; i<atoms.size(); i++){
            std::cout << "Atom " << i << " " << atomic_number_to_element_name(atoms[i].atomic_number) << ": " << std::setprecision(6) << mulliken_population[i] << std::endl;
        }
    }


    if(is_mayer_bond_order_analysis_){ // print Mayer bond order matrix
        std::cout << std::endl;
        std::cout << "[Mayer bond order]" << std::endl;
        const auto& mayer_bond_order_matrix = compute_mayer_bond_order();
        // save the current format flags and precision
        std::ios::fmtflags old_flags = std::cout.flags();
        std::streamsize old_precision = std::cout.precision();
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << std::fixed << std::setprecision(3) << mayer_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // restore the format flags and precision
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
    }

    if(is_wiberg_bond_order_analysis_){ // print Wiberg bond order matrix
        std::cout << std::endl;
        std::cout << "[Wiberg bond order]" << std::endl;
        const auto& wiberg_bond_order_matrix = compute_wiberg_bond_order();
        // save the current format flags and precision
        std::ios::fmtflags old_flags = std::cout.flags();
        std::streamsize old_precision = std::cout.precision();
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << std::fixed << std::setprecision(3) << wiberg_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
        // restore the format flags and precision
        std::cout.flags(old_flags);
        std::cout.precision(old_precision);
    }

    // Orbital energies
    {
        std::cout << std::endl;
        std::cout << "[Orbital Energies]" << std::endl;
        const int N = num_basis;
        const int num_occ = num_electrons / 2;
        std::vector<real_t> eps(N);
        cudaMemcpy(eps.data(), orbital_energies.device_ptr(), N * sizeof(real_t), cudaMemcpyDeviceToHost);
        std::ios::fmtflags old_flags = std::cout.flags();
        std::streamsize old_prec = std::cout.precision();
        for (int i = 0; i < N; ++i) {
            std::cout << "  MO " << std::setw(4) << (i + 1)
                      << (i < num_occ ? " (occ) " : " (vir) ")
                      << std::fixed << std::setprecision(6) << eps[i] << " hartree" << std::endl;
        }
        std::cout.flags(old_flags);
        std::cout.precision(old_prec);
    }

    std::cout << std::endl;
    std::cout << "[Calculation Summary]" << std::endl;
    std::cout << "Method: Restricted Hartree-Fock (RHF)" << std::endl;
    std::cout << "Schwarz screening threshold: " << std::scientific << std::setprecision(2) << schwarz_screening_threshold << std::endl;
    std::cout << "Initial guess method: " << initial_guess_method_ << std::endl;
    std::cout << "Convergence algorithm: " << convergence_method_->get_algorithm_name() << std::endl;
    std::cout << "Number of iterations: " << iter_ << std::endl;
    std::cout << "Convergence criterion: " << std::scientific << std::setprecision(2) << convergence_energy_threshold << std::endl;
    std::cout << "Energy difference: " << std::scientific << std::setprecision(2) << energy_difference_ << std::endl;
    std::cout << std::fixed;
    std::cout << "Energy (without nuclear repulsion): " << std::setprecision(17) << get_energy() << " [hartree]" << std::endl;
    std::cout << "Total Energy: " << std::setprecision(17) << get_total_energy() << " [hartree]" << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "Computing time: " << std::setprecision(5) << get_solve_time_in_milliseconds() << " [ms]" << std::endl;

    if(get_post_hf_method() != PostHFMethod::None){
        std::cout << std::endl;
        std::cout << "[Calculation Summary (Post-HF)]" << std::endl;

        // Method name — covers every PostHFMethod enum value.
        const PostHFMethod m = get_post_hf_method();
        const char* method_name = "(unknown)";
        switch (m) {
            case PostHFMethod::FCI:           method_name = "FCI"; break;
            case PostHFMethod::MP2:           method_name = "MP2"; break;
            case PostHFMethod::SCS_MP2:       method_name = "SCS-MP2"; break;
            case PostHFMethod::SOS_MP2:       method_name = "SOS-MP2"; break;
            case PostHFMethod::LT_MP2:        method_name = "LT-MP2"; break;
            case PostHFMethod::LT_SOS_MP2:    method_name = "LT-SOS-MP2"; break;
            case PostHFMethod::MP3:           method_name = "MP3"; break;
            case PostHFMethod::MP4:           method_name = "MP4"; break;
            case PostHFMethod::CC2:           method_name = "CC2"; break;
            case PostHFMethod::CCSD:          method_name = "CCSD"; break;
            case PostHFMethod::CCSD_T:        method_name = "CCSD(T)"; break;
            case PostHFMethod::CCSD_DENSITY:  method_name = "CCSD + 1-RDM (Lambda)"; break;
            case PostHFMethod::CIS:           method_name = "CIS"; break;
            case PostHFMethod::ADC2:          method_name = "ADC(2)"; break;
            case PostHFMethod::SOS_ADC2:      method_name = "SOS-ADC(2)"; break;
            case PostHFMethod::LT_SOS_ADC2:   method_name = "LT-SOS-ADC(2)"; break;
            case PostHFMethod::ADC2X:         method_name = "ADC(2)-x"; break;
            case PostHFMethod::EOM_MP2:       method_name = "EOM-MP2"; break;
            case PostHFMethod::EOM_CC2:       method_name = "EOM-CC2"; break;
            case PostHFMethod::EOM_CCSD:      method_name = "EOM-CCSD"; break;
            case PostHFMethod::DMET_CCSD:     method_name = "DMET-CCSD"; break;
            case PostHFMethod::DMET_CCSD_T:   method_name = "DMET-CCSD(T)"; break;
            case PostHFMethod::THC_MP2:       method_name = "THC-MP2"; break;
            case PostHFMethod::THC_SOS_MP2:   method_name = "THC-SOS-MP2"; break;
            case PostHFMethod::THC_SOS_ADC2:  method_name = "THC-SOS-ADC(2)"; break;
            case PostHFMethod::DLPNO_MP2:     method_name = "DLPNO-MP2"; break;
            case PostHFMethod::DLPNO_CCSD:    method_name = "DLPNO-CCSD"; break;
            case PostHFMethod::DLPNO_CCSD_T:  method_name = "DLPNO-CCSD(T)"; break;
            case PostHFMethod::CIS_NTO:       method_name = "CIS NTO active space (bt-PNO-STEOM P0)"; break;
            case PostHFMethod::IP_EOM_CCSD:   method_name = "IP-EOM-CCSD (bt-PNO-STEOM P1)"; break;
            case PostHFMethod::EA_EOM_CCSD:   method_name = "EA-EOM-CCSD (bt-PNO-STEOM P2)"; break;
            case PostHFMethod::STEOM_CCSD:    method_name = "STEOM-CCSD (bt-PNO-STEOM P3)"; break;
            case PostHFMethod::DLPNO_STEOM_CCSD: method_name = "DLPNO-STEOM-CCSD (hybrid bt-PNO-STEOM P5b)"; break;
            case PostHFMethod::DMET_STEOM:    method_name = "DMET-STEOM-CCSD"; break;
            case PostHFMethod::None:          method_name = "(none)"; break;
        }
        std::cout << "Post-HF method: " << method_name << std::endl;

        // Method-specific configuration / parallelism info.
        const bool is_dlpno = (m == PostHFMethod::DLPNO_MP2
                            || m == PostHFMethod::DLPNO_CCSD
                            || m == PostHFMethod::DLPNO_CCSD_T);
        const bool is_dmet  = (m == PostHFMethod::DMET_CCSD
                            || m == PostHFMethod::DMET_CCSD_T);
        // Resolve actual GPU count: get_num_gpus() may be -1 (auto-detect);
        // MultiGpuManager reports what was actually initialised at runtime.
        const int n_gpu_actual = MultiGpuManager::instance().num_devices();
        if (is_dlpno) {
            std::cout << "  DLPNO preset:   " << get_dlpno_preset() << std::endl;
            const int ns = get_last_dlpno_n_strong();
            const int nw = get_last_dlpno_n_weak();
            const int ne = get_last_dlpno_n_empty();
            const int n_pairs = ns + nw + ne;
            if (n_pairs > 0) {
                std::cout << "  Pairs:          " << ns << " strong + " << nw
                          << " weak";
                if (ne > 0) std::cout << " (" << ne << " empty)";
                std::cout << " / " << n_pairs << " total" << std::endl;
            }
            if (m == PostHFMethod::DLPNO_CCSD_T) {
                const int n_tot = get_last_dlpno_n_triples_total();
                const int n_act = get_last_dlpno_n_triples_active();
                if (n_tot > 0) {
                    std::cout << "  Triples (i≤j≤k): " << n_act << " active / "
                              << n_tot << " total" << std::endl;
                }
            }
            std::cout << "  GPUs used:      " << n_gpu_actual << std::endl;
        }
        if (is_dmet) {
            const std::string& frags = get_dmet_fragments_str();
            const int nf = get_last_dmet_n_fragments();
            std::cout << "  DMET fragments: "
                      << (nf > 0 ? std::to_string(nf) + " " : std::string())
                      << "(" << (frags.empty() ? "auto X-H bonds" : frags) << ")"
                      << std::endl;
            std::cout << "  GPUs used:      " << n_gpu_actual << std::endl;
        }

        const auto& exc_energies = get_excitation_energies();
        if(!exc_energies.empty()){
            // Excited state methods: print detailed report
            const auto& report = get_excited_state_report();
            if(!report.empty()){
                std::cout << report;
            }
        }else{
            // Correlation energy methods (MP2, CCSD, etc.)
            std::cout << "Post-HF energy correction: " << std::setprecision(17) << get_post_hf_energy() << " [hartree]" << std::endl;
            std::cout << "Total Energy (including post-HF correction): " << std::setprecision(17) << get_total_energy() + get_post_hf_energy() << " [hartree]" << std::endl;
        }
    }
}

/*
void RHF::export_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open the file: " + filename);
    }
    // write the header
    ofs << "[Molden Format]" << std::endl;
    ofs << "generated by GANSU" << std::endl;
    ofs << "[Atoms] (AU)" << std::endl;
    for(size_t i=0; i<atoms.size(); i++){
        ofs << atomic_number_to_element_name(atoms[i].atomic_number) << " " 
            << i+1 << " "
            << atoms[i].atomic_number << " "
            << atoms[i].coordinate.x << " "
            << atoms[i].coordinate.y << " "
            << atoms[i].coordinate.z << std::endl;
    }

    ofs << "[GTO]" << std::endl;
    primitive_shells.toHost();
    // [GTO] writes Cart-indexed primitive shell info (primitive_shells.basis_index
    // is the Cart basis function index 0..nbf_cart-1).  When use_spherical, num_basis
    // = nbf_sph < nbf_cart, so sizing these arrays by num_basis triggers OOB writes.
    const size_t nbf_for_gto = get_use_spherical() ? get_num_basis_cart() : num_basis;
    std::vector<int> num_primitives(nbf_for_gto, 0);
    std::vector<int> shell_types(nbf_for_gto, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }

    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range_cart()[i];  // Molden [GTO]: Cartesian shell layout
        for(size_t j=basis_range.start_index; j<basis_range.end_index; j++){
            if(num_primitives[j] == 0){ // skip non-representative basis functions (e.g. py,pz, etc.)
                continue;
            }  
            ofs << " " << shell_type_to_shell_name(shell_types[j]) << " " << num_primitives[j] << " " << "1.00" << std::endl;
            for(size_t k=0; k<primitive_shells.size(); k++){
                if(primitive_shells[k].basis_index == j){
                    ofs << "\t" << primitive_shells[k].exponent << " " << primitive_shells[k].coefficient << std::endl;
                }
            }
        }
        ofs << std::endl; // empty line
    }

    ofs << std::endl; // empty line
    
    // write the orbital energies
    if (get_use_spherical()) {
        // Molden spherical-basis markers (per Molden file format spec):
        // [5D] = 5 D functions in spherical order, [7F] = 7 F, [9G] = 9 G.
        // Coefficient ordering for spherical functions follows Molden's m-order
        // (d_0, d_+1, d_-1, d_+2, d_-2 etc.), which matches what GANSU's
        // Cart→Sph transform produces.
        ofs << "[5D]" << std::endl;
        ofs << "[7F]" << std::endl;
        ofs << "[9G]" << std::endl;
    }
    ofs << "[MO]" << std::endl;
    orbital_energies.toHost();
    coefficient_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= " << (i < num_electrons/2 ? 2.0 : 0.0) << std::endl;
        for(size_t j=0; j<num_basis; j++){
            ofs << " " << j+1 << " " << std::setprecision(17) << coefficient_matrix(j, i) << std::endl;
        }
    }

    ofs.close();

}
*/

void RHF::export_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open the file: " + filename);
    }
    // write the header
    ofs << "[Molden Format]" << std::endl;
    ofs << "[Title]" << std::endl;
    ofs << "generated by GANSU" << std::endl;
    ofs << "[Atoms] (Angs)" << std::endl;
    for(size_t i=0; i<atoms.size(); i++){
        ofs << atomic_number_to_element_name(atoms[i].atomic_number) << " " 
            << i+1 << " "
            << atoms[i].atomic_number << " "
            << bohr_to_angstrom(atoms[i].coordinate.x) << " "
            << bohr_to_angstrom(atoms[i].coordinate.y) << " "
            << bohr_to_angstrom(atoms[i].coordinate.z) << std::endl;
    }

    ofs << "[GTO]" << std::endl;
    primitive_shells.toHost();
    // [GTO] writes Cart-indexed primitive shell info (primitive_shells.basis_index
    // is the Cart basis function index 0..nbf_cart-1).  When use_spherical, num_basis
    // = nbf_sph < nbf_cart, so sizing these arrays by num_basis triggers OOB writes.
    const size_t nbf_for_gto = get_use_spherical() ? get_num_basis_cart() : num_basis;
    std::vector<int> num_primitives(nbf_for_gto, 0);
    std::vector<int> shell_types(nbf_for_gto, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }

    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range_cart()[i];  // Molden [GTO]: Cartesian shell layout
        for(size_t j=basis_range.start_index; j<basis_range.end_index; j++){
            if(num_primitives[j] == 0){ // skip non-representative basis functions (e.g. py,pz, etc.)
                continue;
            }  
            ofs << " " << shell_type_to_shell_name(shell_types[j]) << " " << num_primitives[j] << " " << "1.00" << std::endl;
            for(size_t k=0; k<primitive_shells.size(); k++){
                if(primitive_shells[k].basis_index == j){
                    ofs << "    " << primitive_shells[k].exponent << " " << primitive_shells[k].coefficient << std::endl;
                }
            }
        }
        ofs << std::endl; // empty line
    }

//    ofs << std::endl; // empty line

    // write the orbital energies
    if (get_use_spherical()) {
        // Molden spherical-basis markers (per Molden file format spec):
        // [5D] = 5 D functions in spherical order, [7F] = 7 F, [9G] = 9 G.
        // Coefficient ordering for spherical functions follows Molden's m-order
        // (d_0, d_+1, d_-1, d_+2, d_-2 etc.), which matches what GANSU's
        // Cart→Sph transform produces.
        ofs << "[5D]" << std::endl;
        ofs << "[7F]" << std::endl;
        ofs << "[9G]" << std::endl;
    }
    ofs << "[MO]" << std::endl;
    orbital_energies.toHost();
    coefficient_matrix.toHost();
    for(size_t i=0; i<num_basis; i++){
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= " << (i < num_electrons/2 ? 2.0 : 0.0) << std::endl;
        for(size_t j=0; j<num_basis; j++){
            ofs << " " << j+1 << " " << std::setprecision(17) << coefficient_matrix(j, i) << std::endl;
        }
    }

    ofs.close();

}

// ---------------------------------------------------------------------------
//  Pipek-Mezey localized occupied MOs → Molden file
// ---------------------------------------------------------------------------
//  Re-uses the same [Atoms]/[GTO] header as the canonical writer, but the
//  occupied block in [MO] is replaced by LMOs (C_occ · U). Virtual block keeps
//  the canonical orbitals so molden viewers (Avogadro/Jmol/VMD) still show a
//  full nao-set. Orbital energies for the occupied LMOs are the diagonal
//  Fock-in-LMO-basis values ε_i^LMO = (U^T diag(ε_can) U)_{ii} so they sort
//  meaningfully in viewers.
// ---------------------------------------------------------------------------
void RHF::export_lmo_molden_file(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Failed to open the file: " + filename);
    }

    // --- Header / Atoms / GTO (verbatim from export_molden_file) -----------
    ofs << "[Molden Format]" << std::endl;
    ofs << "[Title]" << std::endl;
    ofs << "generated by GANSU (LMO export)" << std::endl;
    ofs << "[Atoms] (Angs)" << std::endl;
    for(size_t i=0; i<atoms.size(); i++){
        ofs << atomic_number_to_element_name(atoms[i].atomic_number) << " "
            << i+1 << " "
            << atoms[i].atomic_number << " "
            << bohr_to_angstrom(atoms[i].coordinate.x) << " "
            << bohr_to_angstrom(atoms[i].coordinate.y) << " "
            << bohr_to_angstrom(atoms[i].coordinate.z) << std::endl;
    }
    ofs << "[GTO]" << std::endl;
    primitive_shells.toHost();
    // [GTO] writes Cart-indexed primitive shell info (primitive_shells.basis_index
    // is the Cart basis function index 0..nbf_cart-1).  When use_spherical, num_basis
    // = nbf_sph < nbf_cart, so sizing these arrays by num_basis triggers OOB writes.
    const size_t nbf_for_gto = get_use_spherical() ? get_num_basis_cart() : num_basis;
    std::vector<int> num_primitives(nbf_for_gto, 0);
    std::vector<int> shell_types(nbf_for_gto, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }
    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range_cart()[i];  // Molden [GTO]: Cartesian shell layout
        for(size_t j=basis_range.start_index; j<basis_range.end_index; j++){
            if(num_primitives[j] == 0) continue;
            ofs << " " << shell_type_to_shell_name(shell_types[j]) << " " << num_primitives[j] << " " << "1.00" << std::endl;
            for(size_t k=0; k<primitive_shells.size(); k++){
                if(primitive_shells[k].basis_index == j){
                    ofs << "    " << primitive_shells[k].exponent << " " << primitive_shells[k].coefficient << std::endl;
                }
            }
        }
        ofs << std::endl;
    }

    // --- Localize occupied MOs via Pipek-Mezey -----------------------------
    coefficient_matrix.toHost();
    overlap_matrix.toHost();
    orbital_energies.toHost();
    const int nao  = static_cast<int>(num_basis);
    const int nocc = static_cast<int>(num_electrons / 2);

    // Extract C_occ (nao × nocc, row-major) and S (nao × nao, row-major).
    std::vector<real_t> C_occ(static_cast<size_t>(nao) * nocc);
    for (int mu = 0; mu < nao; ++mu)
        for (int i = 0; i < nocc; ++i)
            C_occ[static_cast<size_t>(mu) * nocc + i] = coefficient_matrix(mu, i);
    std::vector<real_t> S(static_cast<size_t>(nao) * nao);
    for (int mu = 0; mu < nao; ++mu)
        for (int nu = 0; nu < nao; ++nu)
            S[static_cast<size_t>(mu) * nao + nu] = overlap_matrix(mu, nu);

    // Per-atom AO ranges (PM functional requires Mulliken-on-atom partition).
    std::vector<std::pair<int,int>> atom_ao_ranges;
    atom_ao_ranges.reserve(atoms.size());
    for(size_t a = 0; a < atoms.size(); ++a){
        const auto& br = get_atom_to_basis_range()[a];
        atom_ao_ranges.emplace_back(br.start_index, br.end_index);
    }

    auto loc = localize_pipek_mezey(
        C_occ.data(), S.data(),
        nao, nocc, atom_ao_ranges,
        /*max_sweep=*/200, /*conv_tol=*/1e-10, /*verbose=*/1);

    // LMO Fock-diagonal "pseudo-energies": ε_i^LMO = Σ_k U_{ki}² · ε_k^can
    // (this is exact for Hartree-Fock since canonical F is diagonal in the
    // canonical MO basis). Used only for sorting/display in molden viewers.
    std::vector<real_t> eps_lmo(nocc, 0.0);
    for (int i = 0; i < nocc; ++i) {
        real_t s = 0.0;
        for (int k = 0; k < nocc; ++k) {
            const real_t u = loc.U[static_cast<size_t>(k) * nocc + i];
            s += u * u * orbital_energies[k];
        }
        eps_lmo[i] = s;
    }

    // --- [MO] block: LMOs (occupied) + canonical virtuals -----------------
    if (get_use_spherical()) {
        // Molden spherical-basis markers (per Molden file format spec):
        // [5D] = 5 D functions in spherical order, [7F] = 7 F, [9G] = 9 G.
        // Coefficient ordering for spherical functions follows Molden's m-order
        // (d_0, d_+1, d_-1, d_+2, d_-2 etc.), which matches what GANSU's
        // Cart→Sph transform produces.
        ofs << "[5D]" << std::endl;
        ofs << "[7F]" << std::endl;
        ofs << "[9G]" << std::endl;
    }
    ofs << "[MO]" << std::endl;
    // Occupied: write C_LMO column by column.
    for (int i = 0; i < nocc; ++i) {
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << eps_lmo[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= 2.0" << std::endl;
        for (int mu = 0; mu < nao; ++mu) {
            ofs << " " << mu+1 << " " << std::setprecision(17)
                << loc.C_LMO[static_cast<size_t>(mu) * nocc + i] << std::endl;
        }
    }
    // Virtual: keep canonical.
    for (int i = nocc; i < nao; ++i) {
        ofs << "Sym= A" << std::endl;
        ofs << "Ene= " << orbital_energies[i] << std::endl;
        ofs << "Spin= Alpha" << std::endl;
        ofs << "Occup= 0.0" << std::endl;
        for (int mu = 0; mu < nao; ++mu) {
            ofs << " " << mu+1 << " " << std::setprecision(17)
                << coefficient_matrix(mu, i) << std::endl;
        }
    }
    ofs.close();

    std::cout << "[LMO Molden] wrote " << nocc << " Pipek-Mezey LMOs + "
              << (nao - nocc) << " canonical virtuals to " << filename
              << "  (PM: " << loc.n_sweeps << " sweeps, L = "
              << loc.functional_initial << " → " << loc.functional_final << ")"
              << std::endl;
}


std::vector<real_t> RHF::analyze_mulliken_population() const {
    std::vector<real_t> mulliken_population_basis(num_basis);

    // calculate the Mulliken population for each basis function
    gpu::computeMullikenPopulation_RHF(
        density_matrix.device_ptr(),
        overlap_matrix.device_ptr(),
        mulliken_population_basis.data(),
        num_basis
    );

    if (verbose) {
        std::cout << "Mulliken population for basis functions:" << std::endl;
        for(size_t i=0; i<num_basis; i++){
            std::cout << "Basis " << i << ": " << std::setprecision(6) << mulliken_population_basis[i] << std::endl;
        }
        std::cout << std::endl;
    }

    const auto& atoms = get_atoms();
    const int num_atoms = atoms.size();
    
    std::vector<real_t> mulliken_population_atoms(num_atoms);
    for(int i=0; i<num_atoms; i++){
        const int basis_start = get_atom_to_basis_range()[i].start_index;
        const int basis_end = get_atom_to_basis_range()[i].end_index;
        mulliken_population_atoms[i] = atoms[i].atomic_number; // initialize with the atomic number (positive charge)
        for(int j=basis_start; j<basis_end; j++){
            mulliken_population_atoms[i] -= mulliken_population_basis[j];
        }
    }

    return mulliken_population_atoms;
}

std::vector<std::vector<real_t>> RHF::compute_mayer_bond_order() const{
    std::vector<std::vector<real_t>> mayer_bond_order_matrix(atoms.size(), std::vector<real_t>(atoms.size(), 0.0));

    std::vector<real_t> temp_matrix(num_basis * num_basis, 0.0); // temporary matrix to store DS (product of density and overlap matrices)

    // calculate the product of density and overlap matrices
    gpu::computeDensityOverlapMatrix(
        density_matrix.device_ptr(),
        overlap_matrix.device_ptr(),
        temp_matrix.data(),
        num_basis
    );

    // sum up the contributions from basis functions to atoms
    for(size_t i=0; i<atoms.size(); i++){
        const int basis_i_start = get_atom_to_basis_range()[i].start_index;
        const int basis_i_end = get_atom_to_basis_range()[i].end_index;
        for(size_t j=0; j<atoms.size(); j++){
            const int basis_j_start = get_atom_to_basis_range()[j].start_index;
            const int basis_j_end = get_atom_to_basis_range()[j].end_index;
            real_t bond_order_ij = 0.0;
            for(int bi=basis_i_start; bi<basis_i_end; bi++){
                for(int bj=basis_j_start; bj<basis_j_end; bj++){
                    bond_order_ij += temp_matrix[bi * num_basis + bj] * temp_matrix[bj * num_basis + bi];
                }
            }
            mayer_bond_order_matrix[i][j] = bond_order_ij;
        }
    }

    if(verbose){
        std::cout << "Mayer bond order matrix:" << std::endl;
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << mayer_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


    return mayer_bond_order_matrix;
}


std::vector<std::vector<real_t>> RHF::compute_wiberg_bond_order() {
    std::vector<std::vector<real_t>> wiberg_bond_order_matrix(atoms.size(), std::vector<real_t>(atoms.size(), 0.0));

    std::vector<real_t> temp_matrix(num_basis * num_basis, 0.0); // temporary matrix to store S^{1/2} * D * S^{1/2}

    // Compute S^{1/2}
    gpu::computeSqrtOverlapDensitySqrtOverlapMatrix(
        overlap_matrix.device_ptr(),
        density_matrix.device_ptr(),
        temp_matrix.data(),
        num_basis
    );

    // sum up the contributions from basis functions to atoms
    for(size_t i=0; i<atoms.size(); i++){
        const int basis_i_start = get_atom_to_basis_range()[i].start_index;
        const int basis_i_end = get_atom_to_basis_range()[i].end_index;
        for(size_t j=0; j<atoms.size(); j++){
            const int basis_j_start = get_atom_to_basis_range()[j].start_index;
            const int basis_j_end = get_atom_to_basis_range()[j].end_index;
            real_t bond_order_ij = 0.0;
            for(int bi=basis_i_start; bi<basis_i_end; bi++){
                for(int bj=basis_j_start; bj<basis_j_end; bj++){
                    real_t d_ij = temp_matrix[bi * num_basis + bj];
                    bond_order_ij += d_ij * d_ij;
                }
            }
            wiberg_bond_order_matrix[i][j] = bond_order_ij;
        }
    }

    if(verbose){
        std::cout << "Wiberg bond order matrix:" << std::endl;
        for(size_t i=0; i<atoms.size(); i++){
            for(size_t j=0; j<atoms.size(); j++){
                std::cout << wiberg_bond_order_matrix[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }


    return wiberg_bond_order_matrix;
}

// ============================================================
//  ADIIS/EDIIS convergence method
// ============================================================

void Convergence_RHF_ADIIS::get_new_fock_matrix() {
    const int nb = num_basis_;
    const size_t nb2 = (size_t)nb * nb;

    // Store current Fock, Density, Energy
    const int idx = iteration_ % num_prev_;
    cudaMemcpy(&prev_fock_matrices.device_ptr()[idx * nb2],
               hf_.get_fock_matrix().device_ptr(), nb2 * sizeof(real_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(&prev_density_matrices.device_ptr()[idx * nb2],
               hf_.get_density_matrix().device_ptr(), nb2 * sizeof(real_t), cudaMemcpyDeviceToDevice);
    prev_energies[idx] = hf_.get_energy();

    // Also store DIIS error for fallback to DIIS near convergence
    gpu::computeDIISErrorMatrix(
        hf_.get_overlap_matrix().device_ptr(),
        hf_.get_transform_matrix().device_ptr(),
        hf_.get_fock_matrix().device_ptr(),
        hf_.get_density_matrix().device_ptr(),
        error_matrix.device_ptr(), nb, false);
    cudaMemcpy(&prev_error_matrices.device_ptr()[idx * nb2],
               error_matrix.device_ptr(), nb2 * sizeof(real_t), cudaMemcpyDeviceToDevice);

    const int n = std::min(iteration_ + 1, num_prev_);
    iteration_++;

    if (n <= 1) return;  // first iteration: no extrapolation

    // Compute Tr[D_i F_j] matrix on host
    // Copy all stored D and F to host
    std::vector<real_t> h_D(n * nb2), h_F(n * nb2);
    for (int k = 0; k < n; k++) {
        int ki = (iteration_ - n + k) % num_prev_;
        cudaMemcpy(&h_D[k * nb2], &prev_density_matrices.device_ptr()[ki * nb2],
                   nb2 * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_F[k * nb2], &prev_fock_matrices.device_ptr()[ki * nb2],
                   nb2 * sizeof(real_t), cudaMemcpyDeviceToHost);
    }
    // df[i][j] = Tr[D_i F_j] = Σ_{μν} D_i(μν) F_j(νμ) = Σ D_i(μν) F_j(μν) (symmetric)
    std::vector<real_t> df(n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            real_t tr = 0;
            for (size_t k = 0; k < nb2; k++)
                tr += h_D[i * nb2 + k] * h_F[j * nb2 + k];
            df[i * n + j] = tr;
        }

    // Collect energies for current window
    std::vector<real_t> energies(n);
    for (int k = 0; k < n; k++) {
        int ki = (iteration_ - n + k) % num_prev_;
        energies[k] = prev_energies[ki];
    }

    // Check DIIS error norm to decide EDIIS vs ADIIS vs DIIS
    real_t err_norm = 0;
    {
        std::vector<real_t> h_err(nb2);
        int newest_idx = (iteration_ - 1) % num_prev_;
        cudaMemcpy(h_err.data(), &prev_error_matrices.device_ptr()[newest_idx * nb2],
                   nb2 * sizeof(real_t), cudaMemcpyDeviceToHost);
        for (auto v : h_err) err_norm += v * v;
        err_norm = std::sqrt(err_norm);
    }

    std::vector<real_t> coeffs;
    std::string method_used;
    if (mode_ == ADIISMode::ADIIS) {
        coeffs = solve_adiis_coefficients(n, df, n - 1);
        method_used = "ADIIS";
    } else if (mode_ == ADIISMode::EDIIS) {
        coeffs = solve_ediis_coefficients(n, energies, df);
        method_used = "EDIIS";
    } else {  // AEDIIS: automatic switching
        if (err_norm < 1e-1) {
            gpu::computeFockMatrixDIIS(
                prev_error_matrices.device_ptr(),
                prev_fock_matrices.device_ptr(),
                hf_.get_fock_matrix().device_ptr(),
                n, nb);
            return;
        } else if (err_norm < 1.0) {
            coeffs = solve_adiis_coefficients(n, df, n - 1);
            method_used = "ADIIS";
        } else {
            coeffs = solve_ediis_coefficients(n, energies, df);
            method_used = "EDIIS";
        }
    }

    if (verbose)
        std::cout << "    [" << method_used << "] err=" << err_norm << " c=[";

    // F_new = Σ c_i F_i
    // Zero out Fock, then accumulate
    cudaMemset(hf_.get_fock_matrix().device_ptr(), 0, nb2 * sizeof(real_t));
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    for (int k = 0; k < n; k++) {
        int ki = (iteration_ - n + k) % num_prev_;
        real_t c = coeffs[k];
        if (verbose) std::cout << (k > 0 ? "," : "") << c;
        cublasDaxpy(handle, (int)nb2, &c,
                    &prev_fock_matrices.device_ptr()[ki * nb2], 1,
                    hf_.get_fock_matrix().device_ptr(), 1);
    }
    if (verbose) std::cout << "]" << std::endl;
}

std::vector<real_t> Convergence_RHF_ADIIS::solve_adiis_coefficients(
    int n, const std::vector<real_t>& df, int newest) const
{
    // ADIIS cost: minimize 2*Σ c_i*(df[i,n]-df[n,n]) + Σ c_i*c_j*(df[i,j]-df[i,n]-df[n,j]+df[n,n])
    // subject to c_i >= 0, Σ c_i = 1
    // Use parameterization c_i = x_i^2 / Σ x_j^2 and BFGS-like iteration
    // For small n, simple iterative reweighting works.
    // Simplified: use uniform coefficients as starting point, then a few gradient steps.

    real_t dn_fn = df[newest * n + newest];
    std::vector<real_t> dd_fn(n);
    std::vector<real_t> df_shifted(n * n);
    for (int i = 0; i < n; i++) {
        dd_fn[i] = df[i * n + newest] - dn_fn;
        for (int j = 0; j < n; j++)
            df_shifted[i * n + j] = df[i * n + j] - df[i * n + newest] - df[newest * n + j] + dn_fn;
    }

    // x parameterization: c_i = x_i^2 / sum(x^2)
    std::vector<real_t> x(n, 1.0);

    for (int iter = 0; iter < 50; iter++) {
        real_t x2sum = 0;
        for (int i = 0; i < n; i++) x2sum += x[i] * x[i];
        std::vector<real_t> c(n);
        for (int i = 0; i < n; i++) c[i] = x[i] * x[i] / x2sum;

        // Gradient of cost w.r.t. x
        std::vector<real_t> fc(n, 0.0);
        for (int k = 0; k < n; k++) {
            fc[k] = 2.0 * dd_fn[k];
            for (int j = 0; j < n; j++)
                fc[k] += c[j] * df_shifted[k * n + j] + c[j] * df_shifted[j * n + k];
        }

        std::vector<real_t> grad(n, 0.0);
        for (int nn = 0; nn < n; nn++) {
            for (int k = 0; k < n; k++) {
                real_t dcdk = (k == nn ? x[nn] * x2sum - x[nn] * x[nn] * x[nn] : -x[nn] * x[nn] * x[k]);
                dcdk *= 2.0 / (x2sum * x2sum);
                grad[nn] += fc[k] * dcdk;
            }
        }

        real_t grad_norm = 0;
        for (auto g : grad) grad_norm += g * g;
        if (std::sqrt(grad_norm) < 1e-10) break;

        // Simple steepest descent with small step
        real_t step = 0.1;
        for (int i = 0; i < n; i++) x[i] -= step * grad[i];
        // Keep x positive
        for (int i = 0; i < n; i++) if (x[i] < 0.01) x[i] = 0.01;
    }

    real_t x2sum = 0;
    for (int i = 0; i < n; i++) x2sum += x[i] * x[i];
    std::vector<real_t> c(n);
    for (int i = 0; i < n; i++) c[i] = x[i] * x[i] / x2sum;
    return c;
}

std::vector<real_t> Convergence_RHF_ADIIS::solve_ediis_coefficients(
    int n, const std::vector<real_t>& energies, const std::vector<real_t>& df) const
{
    // EDIIS cost: minimize Σ c_i E_i - Σ c_i c_j (diag_i + diag_j - df_ij - df_ji) / 2
    // where diag_i = df[i,i] = Tr[D_i F_i]
    std::vector<real_t> diag(n);
    for (int i = 0; i < n; i++) diag[i] = df[i * n + i];

    std::vector<real_t> df_sym(n * n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            df_sym[i * n + j] = diag[i] + diag[j] - df[i * n + j] - df[j * n + i];

    std::vector<real_t> x(n, 1.0);

    for (int iter = 0; iter < 50; iter++) {
        real_t x2sum = 0;
        for (int i = 0; i < n; i++) x2sum += x[i] * x[i];
        std::vector<real_t> c(n);
        for (int i = 0; i < n; i++) c[i] = x[i] * x[i] / x2sum;

        // fc_k = E_k - 2 Σ_i c_i df_sym[i,k]
        std::vector<real_t> fc(n, 0.0);
        for (int k = 0; k < n; k++) {
            fc[k] = energies[k];
            for (int i = 0; i < n; i++)
                fc[k] -= 2.0 * c[i] * df_sym[i * n + k];
        }

        std::vector<real_t> grad(n, 0.0);
        for (int nn = 0; nn < n; nn++) {
            for (int k = 0; k < n; k++) {
                real_t dcdk = (k == nn ? x[nn] * x2sum - x[nn] * x[nn] * x[nn] : -x[nn] * x[nn] * x[k]);
                dcdk *= 2.0 / (x2sum * x2sum);
                grad[nn] += fc[k] * dcdk;
            }
        }

        real_t grad_norm = 0;
        for (auto g : grad) grad_norm += g * g;
        if (std::sqrt(grad_norm) < 1e-10) break;

        real_t step = 0.1;
        for (int i = 0; i < n; i++) x[i] -= step * grad[i];
        for (int i = 0; i < n; i++) if (x[i] < 0.01) x[i] = 0.01;
    }

    real_t x2sum = 0;
    for (int i = 0; i < n; i++) x2sum += x[i] * x[i];
    std::vector<real_t> c(n);
    for (int i = 0; i < n; i++) c[i] = x[i] * x[i] / x2sum;
    return c;
}

// ============================================================
//  MINAO initial guess
// ============================================================
void InitialGuess_RHF_MINAO::guess() {
    const int nb = hf_.get_num_basis();
    const size_t nb2 = (size_t)nb * nb;

    // Resolve ANO-RCC-MB basis path relative to calculation basis
    std::filesystem::path gbs_path(hf_.get_gbsfilename());
    std::filesystem::path minao_path = gbs_path.parent_path() / "ano-rcc-mb.gbs";
    std::cout << " [MINAO] Loading ANO-RCC-MB from: " << minao_path << std::endl;

    BasisSet calc_basis = BasisSet::construct_from_gbs(hf_.get_gbsfilename());
    BasisSet minao_basis = BasisSet::construct_from_gbs(minao_path.string());

    // Get atoms on host
    const auto& atoms_dev = hf_.get_atoms();
    std::vector<Atom> atoms_host(atoms_dev.size());
    cudaMemcpy(atoms_host.data(), atoms_dev.device_ptr(),
               atoms_host.size() * sizeof(Atom), cudaMemcpyDeviceToHost);

    // Build MINAO occupation vector
    int n_minao = 0;
    std::vector<double> occ;
    for (size_t ia = 0; ia < atoms_host.size(); ia++) {
        const std::string elem = atomic_number_to_element_name(atoms_host[ia].atomic_number);
        const auto& ebs = minao_basis.get_element_basis_set(elem);
        auto atom_occ = get_minao_occupations(atoms_host[ia].atomic_number, ebs);
        occ.insert(occ.end(), atom_occ.begin(), atom_occ.end());
        n_minao += (int)atom_occ.size();
    }
    std::cout << " [MINAO] n_calc=" << nb << " n_minao=" << n_minao
              << " n_electrons(occ)=" << std::accumulate(occ.begin(), occ.end(), 0.0) << std::endl;

    // Compute cross overlap S_cross (n_calc_CART × n_minao) — compute_cross_overlap
    // always works in the Cartesian calc basis.
    auto S_cross = compute_cross_overlap(atoms_host, calc_basis, minao_basis);

    // Under spherical basis, S_calc and X (from the HF object) are spherical
    // (nb = nbf_sph), but S_cross is [nbf_cart × n_minao]. Transform the calc
    // (Cartesian) axis to spherical: S_cross_sph[m,j] = Σ_p U[m,p] S_cross[p,j]
    // with the block-diagonal Cart→Sph matrix U. The MINAO occupation axis j is
    // untouched. The minao axis itself stays in its own (Cartesian) basis — only
    // the projection onto the calc basis matters and is now consistent.
    if (hf_.get_use_spherical()) {
        const auto& sht  = hf_.get_shell_types();
        const auto& offc = hf_.get_shell_offsets_cart();
        const auto& offs = hf_.get_shell_offsets_sph();
        std::vector<double> S_cross_sph((size_t)nb * n_minao, 0.0);
        for (size_t ish = 0; ish < sht.size(); ish++) {
            const auto U = spherical::get_cart_to_sph_matrix(sht[ish]);  // [n_sph_i × n_cart_i]
            const int ns_i = (int)U.size();
            const int nc_i = (int)U[0].size();
            const int so = offs[ish], co = offc[ish];
            for (int a = 0; a < ns_i; a++)
                for (int b = 0; b < nc_i; b++) {
                    const double u = U[a][b];
                    if (u == 0.0) continue;
                    for (int j = 0; j < n_minao; j++)
                        S_cross_sph[(size_t)(so + a) * n_minao + j] +=
                            u * S_cross[(size_t)(co + b) * n_minao + j];
                }
        }
        S_cross = std::move(S_cross_sph);
    }

    // Compute S_calc overlap on host (reuse from HF object)
    hf_.get_overlap_matrix().toHost();
    std::vector<double> S_calc(nb2);
    std::memcpy(S_calc.data(), hf_.get_overlap_matrix().host_ptr(), nb2 * sizeof(double));

    // Projection: P = S_calc^{-1} × S_cross
    // Use Cholesky: S_calc = L L^T, solve L L^T P = S_cross
    // Or use transform matrix X: S_calc^{-1} = X X^T (where X = S^{-1/2})
    // P = X X^T S_cross = X (X^T S_cross)
    hf_.get_transform_matrix().toHost();
    std::vector<double> X(nb2);
    std::memcpy(X.data(), hf_.get_transform_matrix().host_ptr(), nb2 * sizeof(double));

    // tmp = X^T × S_cross (nb × n_minao)
    std::vector<double> tmp(nb * n_minao, 0.0);
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < n_minao; j++)
            for (int k = 0; k < nb; k++)
                tmp[i * n_minao + j] += X[k * nb + i] * S_cross[k * n_minao + j];

    // P = X × tmp (nb × n_minao)
    std::vector<double> P(nb * n_minao, 0.0);
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < n_minao; j++)
            for (int k = 0; k < nb; k++)
                P[i * n_minao + j] += X[i * nb + k] * tmp[k * n_minao + j];

    // D = P × diag(occ) × P^T (nb × nb)
    // Scale columns: P_occ[i][j] = P[i][j] * occ[j]
    std::vector<double> D(nb2, 0.0);
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < nb; j++) {
            double sum = 0;
            for (int k = 0; k < n_minao; k++)
                sum += P[i * n_minao + k] * occ[k] * P[j * n_minao + k];
            D[i * nb + j] = sum;
        }

    // Upload density to GPU
    cudaMemcpy(hf_.get_density_matrix().device_ptr(), D.data(), nb2 * sizeof(real_t), cudaMemcpyHostToDevice);
    hf_.compute_fock_matrix();
    hf_.compute_coefficient_matrix();
    hf_.compute_density_matrix();
    hf_.compute_fock_matrix();

    std::cout << " [MINAO] Initial guess ready." << std::endl;
}

} // namespace gansu