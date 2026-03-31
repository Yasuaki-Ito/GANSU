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
#include "cphf_solver.hpp"
#include <cassert>
#include "ao2mo.cuh"

#include <limits> // numeric_limits<double>::max();
#include <iomanip> // std::setprecision

#include "utils.hpp" // THROW_EXCEPTION
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
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
        set_eri_method(std::make_unique<ERI_RI_RHF>(*this, auxiliary_molecular));
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
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
        set_eri_method(std::make_unique<ERI_RI_Direct_RHF>(*this, auxiliary_molecular));
    }else if(eri_method == "semi_direct_ri"){
        const std::string auxiliary_gbsfilename = parameters.get<std::string>("auxiliary_gbsfilename");
        BasisSet aux_basis = get_auxiliary_basis(molecular, auxiliary_gbsfilename);
        Molecular auxiliary_molecular(molecular.get_atoms(), aux_basis);
        std::cout << "[RI] Auxiliary basis: " << auxiliary_molecular.get_num_basis() << " functions" << std::endl;
        set_eri_method(std::make_unique<ERI_RI_SemiDirect_RHF>(*this, auxiliary_molecular));
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
    }else if(post_hf_method == PostHFMethod::FCI){
        post_hf_energy_ = eri_method_->compute_fci_energy();
    }else if(post_hf_method == PostHFMethod::MP2){
        post_hf_energy_ = eri_method_->compute_mp2_energy();
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
    }else if(post_hf_method == PostHFMethod::CIS){
        eri_method_->compute_cis(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::ADC2){
        eri_method_->compute_adc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::ADC2X){
        eri_method_->compute_adc2x(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_MP2){
        eri_method_->compute_eom_mp2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_CC2){
        eri_method_->compute_eom_cc2(get_n_excited_states());
    }else if(post_hf_method == PostHFMethod::EOM_CCSD){
        eri_method_->compute_eom_ccsd(get_n_excited_states());
    }else{
        THROW_EXCEPTION("Invalid post-HF method.");
    }
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
        initial_guess = std::make_unique<InitialGuess_RHF_SAD>(*this);
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
    auto skel_hessian = gpu::computeSkeletonHessian_RHF(
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
    // PySCF's CPHF includes the 2e response from the occ-occ density change
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

    if (post_hf_method_ == PostHFMethod::MP2) {
        // MP2 gradient uses computeEnergyGradient_general with:
        //   1-electron terms: P_relaxed (HF + MP2 unrelaxed + Z-vector)
        //   Overlap term: W_MP2
        //   2-electron term: D_HF (the ERI derivative kernel computes Γ=D⊗D which is the HF 2-PDM)
        // Plus a separate non-separable 2-PDM contribution (Γ^MP2) — TODO
        const int nao = num_basis;
        const size_t nao2 = (size_t)nao * nao;

        real_t* d_P_relaxed = nullptr;
        real_t* d_W_mp2 = nullptr;
        real_t* d_Gamma_placeholder = nullptr;
        gansu::tracked_cudaMalloc(&d_P_relaxed, nao2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_W_mp2, nao2 * sizeof(real_t));
        gansu::tracked_cudaMalloc(&d_Gamma_placeholder, nao2 * sizeof(real_t));

        eri_method_->compute_mp2_effective_densities(d_P_relaxed, d_W_mp2, d_Gamma_placeholder);

        auto gradient = gpu::computeEnergyGradient_general(
            shell_type_infos, shell_pair_type_infos,
            atoms.device_ptr(),
            d_P_relaxed,                    // 1-electron: relaxed density
            d_W_mp2,                        // overlap: MP2 energy-weighted density
            d_P_relaxed,                    // 2-electron: relaxed density (separable 2-PDM approx)
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            static_cast<int>(atoms.size()),
            num_basis, verbose
        );

        // Non-separable MP2 2-PDM contribution: Σ_{ij} P^{ij} ⊗ Q^{ij} × d(ERI)/dX
        // P^{ij}_{μλ} = C_{μi} C_{λj}, Q^{ij}_{νσ} = Σ_{ab} T̃_{ij}^{ab} C_{ν,a+n} C_{σ,b+n}
        // Use polarization identity: P×Q = 0.5[Coulomb(P+Q) - Coulomb(P-Q)]
        // Approximated via: 0.5[kernel(P+Q) - kernel(P-Q)] ≈ Coulomb(P,Q) + small Exchange cross terms
        {
            const int nocc = num_electrons / 2;
            const int nvir = nao - nocc;
            const int n_atoms = static_cast<int>(atoms.size());
            const int n = 3 * n_atoms;

            // Get T̃ on host (recompute from OVOV — already done in effective densities, but stored in eri_matrix_)
            // For now, get T̃ via compute_mp2_effective_densities side effect or recompute.
            // Simpler: compute T̃_ij in AO here using C and the OVOV MO integrals.

            // Get coefficient matrix on host
            coefficient_matrix.toHost();
            const real_t* h_C = coefficient_matrix.host_ptr();

            // Get T̃ from the effective densities computation (stored in d_Gamma_placeholder)
            // Actually, d_Gamma_placeholder was set to zero. We need T̃ separately.
            // For now, build T̃_{ij} AO matrices from the OVOV MO integrals.

            // We need the OVOV MO integrals. Re-transform from AO ERIs.
            // This is a duplicate computation but necessary for the 2-PDM.
            // TODO: cache OVOV in compute_mp2_effective_densities

            // For now, compute T̃_AO directly and contract with gradient via polarization.
            // Build T̃_{ij}^{ab} on host from stored integrals

            // Actually, the simplest: ask ERI to provide T̃_AO or reuse existing data.
            // Skip for now — the non-separable 2-PDM is the remaining ~30% error.
            // TODO: implement T̃_AO construction and polarization-identity gradient
            std::cout << "  [MP2 Gradient] Non-separable 2-PDM not yet implemented." << std::endl;
        }
        // Σ_{μνλσ} Γ^MP2_{μνλσ} d(μν|λσ)/dX

        gansu::tracked_cudaFree(d_P_relaxed);
        gansu::tracked_cudaFree(d_W_mp2);
        gansu::tracked_cudaFree(d_Gamma_placeholder);
        return gradient;
    }

    // HF gradient (default)
    return gpu::computeEnergyGradient_RHF(
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
    std::cout << "Schwarz screening threshold: " << schwarz_screening_threshold << std::endl;
    std::cout << "Initial guess method: " << initial_guess_method_ << std::endl;
    std::cout << "Convergence algorithm: " << convergence_method_->get_algorithm_name() << std::endl;
    std::cout << "Number of iterations: " << iter_ << std::endl;
    std::cout << "Convergence criterion: " << convergence_energy_threshold << std::endl;
    std::cout << "Energy difference: " << energy_difference_ << std::endl;
    std::cout << "Energy (without nuclear repulsion): " << std::setprecision(17) << get_energy() << " [hartree]" << std::endl;
    std::cout << "Total Energy: " << std::setprecision(17) << get_total_energy() << " [hartree]" << std::endl;
    std::cout << "Computing time: " << std::setprecision(5) << get_solve_time_in_milliseconds() << " [ms]" << std::endl;

    if(get_post_hf_method() != PostHFMethod::None){
        std::cout << std::endl;
        std::cout << "[Calculation Summary (Post-HF)]" << std::endl;
        std::cout << "Post-HF method: ";
        if(get_post_hf_method() == PostHFMethod::FCI){
            std::cout << "FCI" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::MP2){
            std::cout << "MP2" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::MP3){
            std::cout << "MP3" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CC2){
            std::cout << "CC2" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CCSD){
            std::cout << "CCSD" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CCSD_T){
            std::cout << "CCSD(T)" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::CIS){
            std::cout << "CIS" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::ADC2){
            std::cout << "ADC(2)" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::ADC2X){
            std::cout << "ADC(2)-x" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::EOM_MP2){
            std::cout << "EOM-MP2" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::EOM_CC2){
            std::cout << "EOM-CC2" << std::endl;
        }else if(get_post_hf_method() == PostHFMethod::EOM_CCSD){
            std::cout << "EOM-CCSD" << std::endl;
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
    std::vector<int> num_primitives(num_basis, 0);
    std::vector<int> shell_types(num_basis, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }

    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range()[i];
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
    std::vector<int> num_primitives(num_basis, 0);
    std::vector<int> shell_types(num_basis, 0);
    for(size_t i=0; i<primitive_shells.size(); i++){
        num_primitives[primitive_shells[i].basis_index]++;
        shell_types[primitive_shells[i].basis_index] = primitive_shells[i].shell_type;
    }

    for(size_t i=0; i<atoms.size(); i++){
        ofs << i+1 << " " << 0 << std::endl;
        BasisRange basis_range = get_atom_to_basis_range()[i];
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

} // namespace gansu