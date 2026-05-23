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



#pragma once

#include "post_hf_method.hpp"

#include "hf.hpp"
#include "types.hpp"
#include "device_host_memory.hpp"

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h> // for int2 type
#endif


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
    virtual const real_t* get_eri_matrix_device() const { return nullptr; } ///< AO ERI (stored only)

    /// Build full MO ERI from AO ERI + coefficient matrix.
    /// Returns device pointer to MO ERI [nmo⁴]. Caller must free with tracked_cudaFree.
    /// Default: uses get_eri_matrix_device() + 4-index AO→MO transform.
    virtual real_t* build_mo_eri(const real_t* d_C, int nmo) const;

    /// Step 6.3c — workspace variant. Writes the (nmo⁴) MO ERI into the caller-
    /// supplied `d_eri_out` device buffer, which must hold at least
    /// `(size_t)nmo*nmo*nmo*nmo` real_t elements. Caller owns the buffer
    /// (no free per call). Lets DLPNO `pair_setup` reuse a single max-size
    /// workspace across all 465 build_mo_eri invocations, avoiding the
    /// per-pair `cudaMalloc + cudaMemset` of a 16 MB output buffer.
    /// Default: delegates to build_mo_eri + cudaMemcpy + tracked_cudaFree.
    virtual void build_mo_eri_into(const real_t* d_C, int nmo,
                                   real_t* d_eri_out) const;

    /// Compute G(D) = 2J[D] - K[D] (two-electron response) from an arbitrary density matrix.
    /// Default: uses get_eri_matrix_device() + computeFockMatrix_RHF with zero core Hamiltonian.
    virtual void compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const;

    /// Compute the full analytical RI-RHF gradient (1-electron + S + N + RI 2-electron).
    /// Default: throws — only ERI_RI (and its distributed subclass) implement this.
    /// Caller passes the density / coefficient / orbital-energy buffers (device-resident).
    /// Returns 3*num_atoms doubles in atom-major (x, y, z) order.
    virtual std::vector<double> compute_ri_gradient(
        const real_t* /*d_density_matrix*/,
        const real_t* /*d_coefficient_matrix*/,
        const real_t* /*d_orbital_energies*/,
        const int     /*num_electron*/) {
        throw std::runtime_error("compute_ri_gradient(): not implemented for this ERI method");
    }

    /// Whether this ERI method has an analytical RI gradient implementation.
    virtual bool supports_ri_gradient() const { return false; }

    /**
     * @brief Check if the post-HF method is supported
     * @param method Post-HF method
     * @return true if the method is supported, false otherwise
     * @details This function checks if the post-HF method is supported.
     * @details This function can be overridden in the derived class.
     */
    virtual bool supports_post_hf_method(PostHFMethod method) const {
        return false; // By default, no post-HF methods are supported
    }

    /**
     * @brief Compute MP2 energy
        * @return MP2 energy
        * @details This function computes the MP2 energy.
        */
    virtual real_t compute_mp2_energy(){
        THROW_EXCEPTION("MP2 energy computation is not supported for the selected ERI method.");
    }

    virtual real_t compute_thc_mp2_energy(){
        THROW_EXCEPTION("THC-MP2 energy computation is not supported for the selected ERI method (use eri_method=stored).");
    }

    virtual real_t compute_thc_sos_mp2_energy(){
        THROW_EXCEPTION("THC-SOS-MP2 energy computation is not supported for the selected ERI method (use eri_method=stored).");
    }

    virtual void compute_thc_sos_adc2(int /*n_states*/){
        THROW_EXCEPTION("THC-SOS-ADC(2) is not supported for the selected ERI method (use eri_method=stored).");
    }

    virtual real_t compute_lt_mp2_energy(){
        THROW_EXCEPTION("LT-MP2 energy computation is not supported for the selected ERI method.");
    }

    virtual real_t compute_lt_sos_mp2_energy(){
        THROW_EXCEPTION("LT-SOS-MP2 energy computation is not supported for the selected ERI method.");
    }

    virtual real_t compute_scs_mp2_energy(){
        THROW_EXCEPTION("SCS-MP2 energy computation is not supported for the selected ERI method.");
    }

    virtual real_t compute_sos_mp2_energy(){
        THROW_EXCEPTION("SOS-MP2 energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute MP2 effective density matrices for gradient calculation
     * @param d_P_eff   Output: effective 1-PDM in AO basis (num_basis × num_basis)
     * @param d_W_eff   Output: effective energy-weighted density in AO basis
     * @param d_Gamma_eff Output: effective 2-PDM density pair for ERI derivative contraction
     * @details Computes relaxed MP2 density (unrelaxed + Z-vector) and transforms to AO basis.
     *          The caller uses these to compute the MP2 gradient via existing integral derivative kernels.
     */
    virtual void compute_mp2_effective_densities(real_t* d_P_eff, real_t* d_W_eff, real_t* d_Gamma_eff, real_t* d_P_2el) {
        THROW_EXCEPTION("MP2 gradient is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute MP3 energy
        * @return MP3 energy
        * @details This function computes the MP3 energy.
        */
    virtual real_t compute_mp3_energy(){
        THROW_EXCEPTION("MP3 energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute MP4 energy
        * @return MP4 energy
        * @details This function computes the MP4 energy.
        */
    virtual real_t compute_mp4_energy(){
        THROW_EXCEPTION("MP4 energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute CCSD energy
        * @return CCSD energy
        * @details This function computes the CCSD energy.
        */
    virtual real_t compute_cc2_energy(){
        THROW_EXCEPTION("CC2 energy computation is not supported for the selected ERI method.");
    }

    virtual real_t compute_ccsd_energy(){
        THROW_EXCEPTION("CCSD energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute CCSD 1-RDM (non-relaxed correlation density)
     * @details Solves CCSD Lambda equations and constructs the spin-traced
     *          1-RDM D_pq in MO basis, then transforms to AO. Stored via
     *          RHF::set_ccsd_1rdm_{ao,mo}. Mainly for DMET solver use.
     */
    virtual void compute_ccsd_density(){
        THROW_EXCEPTION("CCSD density is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute CCSD(T) energy
        * @return CCSD(T) energy
        * @details This function computes the CCSD(T) energy.
        */
    virtual real_t compute_ccsd_t_energy(){
        THROW_EXCEPTION("CCSD(T) energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute DMET-CCSD energy
     * @return DMET-CCSD correlation energy
     */
    virtual real_t compute_dmet_ccsd(){
        THROW_EXCEPTION("DMET-CCSD is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute DMET-CCSD(T) energy (CCSD + perturbative triples per fragment)
     * @return DMET-CCSD(T) correlation energy (CCSD + (T) summed)
     */
    virtual real_t compute_dmet_ccsd_t(){
        THROW_EXCEPTION("DMET-CCSD(T) is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute DLPNO-MP2 correlation energy (Phase 1)
     * @return DLPNO-MP2 correlation energy
     */
    virtual real_t compute_dlpno_mp2(){
        THROW_EXCEPTION("DLPNO-MP2 is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute DLPNO-CCSD correlation energy (Phase 2)
     * @return DLPNO-CCSD correlation energy
     */
    virtual real_t compute_dlpno_ccsd(){
        THROW_EXCEPTION("DLPNO-CCSD is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute DLPNO-CCSD(T) correlation energy (Phase 3)
     * @return DLPNO-CCSD(T) correlation energy (CCSD + (T) summed)
     */
    virtual real_t compute_dlpno_ccsd_t(){
        THROW_EXCEPTION("DLPNO-CCSD(T) is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute FCI energy
        * @return FCI energy
        * @details This function computes the FCI energy.
        */
    virtual real_t compute_fci_energy(){
        THROW_EXCEPTION("FCI energy computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute CIS excited states
     * @param n_states Number of excited states to compute
     * @details This function computes CIS excited state energies.
     */
    virtual void compute_cis(int n_states){
        THROW_EXCEPTION("CIS computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute ADC(2) excited states
     * @param n_states Number of excited states to compute
     */
    virtual void compute_adc2(int n_states){
        THROW_EXCEPTION("ADC(2) computation is not supported for the selected ERI method.");
    }

    virtual void compute_sos_adc2(int n_states){
        THROW_EXCEPTION("SOS-ADC(2) computation is not supported for the selected ERI method.");
    }

    virtual void compute_sos_laplace_adc2(int n_states){
        THROW_EXCEPTION("SOS-Laplace-ADC(2) computation is not supported for the selected ERI method.");
    }

    virtual void compute_adc2x(int n_states){
        THROW_EXCEPTION("ADC(2)-x computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute EOM-MP2 excited states
     * @param n_states Number of excited states to compute
     */
    virtual void compute_eom_mp2(int n_states){
        THROW_EXCEPTION("EOM-MP2 computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute EOM-CC2 excited states
     * @param n_states Number of excited states to compute
     */
    virtual void compute_eom_cc2(int n_states){
        THROW_EXCEPTION("EOM-CC2 computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute EOM-CCSD excited states
     * @param n_states Number of excited states to compute
     */
    virtual void compute_eom_ccsd(int n_states){
        THROW_EXCEPTION("EOM-CCSD computation is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute state-averaged CIS NTO active space (bt-PNO-STEOM Phase P0)
     * @param n_states_cis Number of CIS roots used to build the state-averaged density
     */
    virtual void compute_cis_nto(int /*n_states_cis*/){
        THROW_EXCEPTION("CIS NTO active space is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute IP-EOM-CCSD (bt-PNO-STEOM Phase P1, building block for ̂S^IP)
     * @param n_states Number of IP roots to extract from Davidson
     */
    virtual void compute_ip_eom_ccsd(int /*n_states*/){
        THROW_EXCEPTION("IP-EOM-CCSD is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute EA-EOM-CCSD (bt-PNO-STEOM Phase P2, building block for ̂S^EA)
     * @param n_states Number of EA roots to extract from Davidson
     */
    virtual void compute_ea_eom_ccsd(int /*n_states*/){
        THROW_EXCEPTION("EA-EOM-CCSD is not supported for the selected ERI method.");
    }

    /**
     * @brief Compute canonical STEOM-CCSD (bt-PNO-STEOM Phase P3)
     * @param n_states Number of excited states to extract from Davidson
     *
     * The driver auto-runs CIS_NTO + IP-EOM + EA-EOM as prerequisites if the
     * corresponding result fields on HF are empty.
     */
    virtual void compute_steom_ccsd(int /*n_states*/){
        THROW_EXCEPTION("STEOM-CCSD is not supported for the selected ERI method.");
    }

    /**
     * @brief Hybrid DLPNO-STEOM-CCSD (bt-PNO-STEOM Phase P5b)
     * @param n_states Number of excited states
     *
     * DLPNO-CCSD ground state back-transformed to canonical MO, fed to the
     * canonical CIS-NTO + IP-EOM + EA-EOM + STEOM machinery. RI-only.
     */
    virtual void compute_dlpno_steom_ccsd(int /*n_states*/){
        THROW_EXCEPTION("DLPNO-STEOM-CCSD is not supported for the selected ERI method (RI required).");
    }

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
    const real_t* get_eri_matrix_device() const { return eri_matrix_.device_ptr(); }

    /// Build MO ERI with rectangular C [nao × nmo] support
    real_t* build_mo_eri(const real_t* d_C, int nmo) const override;

    bool supports_post_hf_method(PostHFMethod method) const override {
        if( method == PostHFMethod::None // always supported
            || method == PostHFMethod::FCI  // The stored ERI method supports FCI
            || method == PostHFMethod::MP2  // The stored ERI method supports MP2
            || method == PostHFMethod::SCS_MP2  // SCS-MP2
            || method == PostHFMethod::SOS_MP2  // SOS-MP2
            || method == PostHFMethod::MP3  // The stored ERI method supports MP3
            || method == PostHFMethod::MP4  // The stored ERI method supports MP4
            || method == PostHFMethod::CC2  // The stored ERI method supports CC2
            || method == PostHFMethod::CCSD // The stored ERI method supports CCSD
            || method == PostHFMethod::CCSD_T // The stored ERI method supports CCSD(T)
            || method == PostHFMethod::CCSD_DENSITY // CCSD + 1-RDM (Lambda)
            || method == PostHFMethod::CIS  // The stored ERI method supports CIS
            || method == PostHFMethod::ADC2 // The stored ERI method supports ADC(2)
            || method == PostHFMethod::ADC2X // The stored ERI method supports ADC(2)-x
            || method == PostHFMethod::EOM_MP2 // The stored ERI method supports EOM-MP2
            || method == PostHFMethod::EOM_CC2 // The stored ERI method supports EOM-CC2
            || method == PostHFMethod::EOM_CCSD // The stored ERI method supports EOM-CCSD
            || method == PostHFMethod::DMET_CCSD // DMET-CCSD
            || method == PostHFMethod::DMET_CCSD_T // DMET-CCSD(T)
            || method == PostHFMethod::THC_MP2 // Tensor Hypercontraction MP2
            || method == PostHFMethod::THC_SOS_MP2 // THC + Laplace SOS-MP2
            || method == PostHFMethod::THC_SOS_ADC2 // THC SOS-ADC(2) excited states
            || method == PostHFMethod::CIS_NTO // bt-PNO-STEOM P0: state-averaged CIS NTO active space
            || method == PostHFMethod::IP_EOM_CCSD // bt-PNO-STEOM P1: IP-EOM-CCSD canonical
            || method == PostHFMethod::EA_EOM_CCSD // bt-PNO-STEOM P2: EA-EOM-CCSD canonical
            || method == PostHFMethod::STEOM_CCSD  // bt-PNO-STEOM P3: STEOM-CCSD (auto-runs P0/P1/P2)
          ){
            return true;
        }
        return false;
    }

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

    /// Lightweight constructor: skips allocation of intermediate_matrix_B_, d_tmp1_, d_tmp2_.
    /// Used by distributed subclasses that don't need single-GPU workspace.
    struct LightweightTag {};
    ERI_RI(const HF& hf, const Molecular& auxiliary_molecular, LightweightTag); ///< Lightweight constructor

    ERI_RI(const ERI_RI&) = delete; ///< copy constructor is deleted
    virtual ~ERI_RI(); ///< destructor

    void precomputation() override;

    /// Compute only Schwarz bounds, shell pair indices, and auxiliary Schwarz (no full B).
    /// Subclasses (e.g., distributed) can call this instead of precomputation() to skip full B build.
    void precompute_schwarz_and_shell_pairs();

    DeviceHostMemory<PrimitiveShell>& get_auxiliary_primitive_shells() { return auxiliary_primitive_shells_; } ///< Get the auxiliary primitive shells
    int get_num_auxiliary_basis() { return num_auxiliary_basis_; }

    std::string get_algorithm_name() override { return "RI"; } ///< Get the algorithm name

    DeviceHostMatrix<real_t>& get_intermediate_matrix_B() { return intermediate_matrix_B_; }

    /// Reconstruct full AO ERI from B matrix: (μν|λσ) = Σ_Q B(Q,μν) B(Q,λσ)
    void reconstruct_ao_eri();
    const real_t* get_eri_matrix_device() const override;

    /// Build full MO ERI directly from B: B→B_mo→(pq|rs) without nao⁴ intermediate
    /// Returns device pointer to MO ERI [nmo⁴]. Caller must free with tracked_cudaFree.
    real_t* build_mo_eri(const real_t* d_C, int nmo) const override;

    /// Step 6.3c — workspace variant. Writes MO ERI directly into caller-
    /// supplied d_eri_out, avoiding the per-call cudaMalloc + cudaMemset.
    void build_mo_eri_into(const real_t* d_C, int nmo,
                           real_t* d_eri_out) const override;

    /// Compute G(D) using RI B-matrix based J/K build (no AO ERI reconstruction needed).
    void compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const override;

    /// Full analytical RI-RHF gradient. See §2.5 of RI_Gradient.md for the math.
    /// Path:
    ///   1) Rebuild (P|Q), Cholesky → L (recomputed since not persisted).
    ///   2) B̄_P = L^{-T} B_P via DTRSM on the P axis.
    ///   3) w_P = Σ_{μν} B_{P,μν} D_{μν}  (Coulomb fitting density).
    ///   4) Γ^(3)_{P,μν} = w_P D_{μν} − ½(D B̄_P D)_{μν}.
    ///   5) Γ^(2)_{PQ}   = −½ w_P w_Q + ¼ Tr[D B̄_P D B̄_Q].
    ///   6) Contract Γ^(3) with 3c2e derivatives (compute_gradients_3c2e).
    ///   7) Contract Γ^(2) with 2c2e derivatives (compute_gradients_2c2e).
    ///   8) Add 1e, S (energy-weighted W), N (nuclear repulsion).
    std::vector<double> compute_ri_gradient(
        const real_t* d_density_matrix,
        const real_t* d_coefficient_matrix,
        const real_t* d_orbital_energies,
        const int num_electron) override;

    bool supports_ri_gradient() const override { return true; }

    /// Worker that does the actual gradient computation. The public
    /// compute_ri_gradient is a thin wrapper around this with the defaults
    /// (full B from intermediate_matrix_B_, full P range, includes 1-electron).
    /// Distributed overrides supply their own d_B_full pointer (replicated
    /// per-device) and local P range, and use include_one_electron to elect
    /// which GPU contributes the 1e/S/N/W gradient (typically only GPU 0).
    ///
    /// Postcondition: returns a length-(3 × num_atoms) gradient vector on host.
    /// If include_one_electron is false, only the 2-electron RI contribution
    /// (3c2e + 2c2e kernel work for the given P range) is returned.
    std::vector<double> compute_ri_gradient_impl(
        const real_t* d_density_matrix,
        const real_t* d_coefficient_matrix,
        const real_t* d_orbital_energies,
        const int num_electron,
        const real_t* d_B_full,
        const size_t P_local_start,
        const size_t P_local_end,
        const bool include_one_electron);

    bool supports_post_hf_method(PostHFMethod method) const override {
        if( method == PostHFMethod::None
         || method == PostHFMethod::MP2
         || method == PostHFMethod::SCS_MP2
         || method == PostHFMethod::SOS_MP2
         || method == PostHFMethod::LT_MP2
         || method == PostHFMethod::LT_SOS_MP2
         || method == PostHFMethod::MP3
         || method == PostHFMethod::CC2
         || method == PostHFMethod::CCSD
         || method == PostHFMethod::CCSD_T
         || method == PostHFMethod::CIS
         || method == PostHFMethod::ADC2
         || method == PostHFMethod::SOS_ADC2
         || method == PostHFMethod::LT_SOS_ADC2
         || method == PostHFMethod::ADC2X
         || method == PostHFMethod::EOM_MP2
         || method == PostHFMethod::EOM_CC2
         || method == PostHFMethod::EOM_CCSD
         || method == PostHFMethod::FCI
         || method == PostHFMethod::DMET_CCSD
         || method == PostHFMethod::DMET_CCSD_T
         || method == PostHFMethod::THC_SOS_ADC2  // RI-Z path (Phase 2.3)
         || method == PostHFMethod::DLPNO_MP2
         || method == PostHFMethod::DLPNO_CCSD
         || method == PostHFMethod::DLPNO_CCSD_T
         || method == PostHFMethod::CIS_NTO  // bt-PNO-STEOM P0 (RI-CIS path)
         || method == PostHFMethod::IP_EOM_CCSD  // bt-PNO-STEOM P4 (RI path)
         || method == PostHFMethod::EA_EOM_CCSD  // bt-PNO-STEOM P4 (RI path)
         || method == PostHFMethod::STEOM_CCSD   // bt-PNO-STEOM P4 (RI path, auto-runs P0/P1/P2)
         || method == PostHFMethod::DLPNO_STEOM_CCSD  // hybrid bt-PNO-STEOM P5b (RI required)
          ){
            return true;
        }
        return false;
    }

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;
    const int num_auxiliary_basis_;
    const int num_occ_;

    const std::vector<ShellTypeInfo> auxiliary_shell_type_infos_; ///< Shell type info in the primitive shell list
    DeviceHostMemory<PrimitiveShell> auxiliary_primitive_shells_; ///< Primitive shells
    DeviceHostMemory<real_t> auxiliary_cgto_normalization_factors_; ///< Normalization factors of the contracted Gauss functions

    DeviceHostMatrix<real_t> intermediate_matrix_B_; ///< intermediate matrix B (num_auxiliary_basis_ x (num_basis_x num_basis_))

    // Suzuki.
    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
    DeviceHostMemory<real_t> auxiliary_schwarz_upper_bound_factors;

    /// Persistent shell pair indices (reused by distributed multi-GPU builds)
    size_t2* d_persistent_shell_pair_indices_ = nullptr;
    size_t num_persistent_shell_pairs_ = 0;

    // DeviceHostMemory<real_t> two_center_eri_;
    // DeviceHostMemory<real_t> three_center_eri_;

    DeviceHostMatrix<real_t> d_J_;
    DeviceHostMatrix<real_t> d_K_;
    DeviceHostMemory<real_t> d_W_tmp_;
    //DeviceHostMatrix<real_t> d_T_tmp_;
    //DeviceHostMatrix<real_t> d_V_tmp_;

    // d_tmp1_ and d_tmp2_ will be intermediate matrices T and V (density-matrix based).
    // d_tmp1_ and d_tmp2_ will be intermediate matrices X and X_packed (coefficient-matrix based).
    DeviceHostMatrix<real_t> d_tmp1_;
    DeviceHostMatrix<real_t> d_tmp2_;

    // Reconstructed full AO ERI (lazily allocated for post-HF)
    mutable real_t* d_eri_reconstructed_ = nullptr;
    mutable bool eri_reconstructed_ = false;
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

    /// Reconstruct full AO ERI by computing all integrals (lazy, for post-HF)
    void reconstruct_ao_eri();
    const real_t* get_eri_matrix_device() const override;

    bool supports_post_hf_method(PostHFMethod method) const override {
        if( method == PostHFMethod::None
         || method == PostHFMethod::MP2
         || method == PostHFMethod::SCS_MP2
         || method == PostHFMethod::SOS_MP2
         || method == PostHFMethod::MP3
         || method == PostHFMethod::MP4
         || method == PostHFMethod::CC2
         || method == PostHFMethod::CCSD
         || method == PostHFMethod::CCSD_T
         || method == PostHFMethod::CIS
         || method == PostHFMethod::ADC2
         || method == PostHFMethod::SOS_ADC2
         || method == PostHFMethod::LT_SOS_ADC2
         || method == PostHFMethod::ADC2X
         || method == PostHFMethod::EOM_MP2
         || method == PostHFMethod::EOM_CC2
         || method == PostHFMethod::EOM_CCSD
         || method == PostHFMethod::FCI
          ){
            return true;
        }
        return false;
    }

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;

    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
    DeviceHostMemory<int2> primitive_shell_pair_indices;
    std::vector<int*> global_counters_;
    std::vector<int*> min_skipped_columns_;
    real_t* fock_matrix_replicas_;
    const int num_fock_replicas_;
    DeviceHostMatrix<real_t> density_matrix_diff_;
    DeviceHostMatrix<real_t> density_matrix_diff_shell_;
    DeviceHostMatrix<real_t> fock_matrix_prev_;
    bool is_first_call_ = true; ///< Reset per SCF solve for density difference tracking

    // Reconstructed full AO ERI (lazily allocated for post-HF)
    mutable real_t* d_eri_reconstructed_ = nullptr;
    mutable bool eri_reconstructed_ = false;
};


/**
 * @brief ERI_Hash class for the electron repulsion integrals (ERIs) using hash memory
 */
enum class HashFockMethod { Compact, Indexed, Fullscan };

class ERI_Hash: public ERI {
public:

    ERI_Hash(const HF& hf); ///< Constructor

    ERI_Hash(const ERI_Hash&) = delete; ///< copy constructor is deleted
    virtual ~ERI_Hash(); ///< destructor

    void precomputation() override;

    std::string get_algorithm_name() override { return "Hash"; } ///< Get the algorithm name

    void set_hash_fock_method(HashFockMethod m) { hash_fock_method_ = m; }
    HashFockMethod get_hash_fock_method() const { return hash_fock_method_; }

    /// Compute G(D) = 2J[D] - K[D] using Hash ERI (reuses Fock construction with H=0)
    void compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const override;

    /// Build full MO ERI tensor from COO AO ERI
    real_t* build_mo_eri(const real_t* d_C, int nmo) const override;

    /// On CPU the hash is not built; we instead return the cached dense tensor
    /// that precomputation() produced via gpu::computeERIMatrix.
    const real_t* get_eri_matrix_device() const override {
        return d_eri_cpu_tensor_;  // nullptr on GPU (hash is used instead)
    }

    bool supports_post_hf_method(PostHFMethod method) const override {
        // With build_mo_eri, all post-HF methods are supported
        if( method == PostHFMethod::None
         || method == PostHFMethod::MP2
         || method == PostHFMethod::SCS_MP2
         || method == PostHFMethod::SOS_MP2
         || method == PostHFMethod::MP3
         || method == PostHFMethod::MP4
         || method == PostHFMethod::CC2
         || method == PostHFMethod::CCSD
         || method == PostHFMethod::CCSD_T
         || method == PostHFMethod::CIS
         || method == PostHFMethod::ADC2
         || method == PostHFMethod::SOS_ADC2
         || method == PostHFMethod::LT_SOS_ADC2
         || method == PostHFMethod::ADC2X
         || method == PostHFMethod::EOM_MP2
         || method == PostHFMethod::EOM_CC2
         || method == PostHFMethod::EOM_CCSD
         || method == PostHFMethod::FCI
          ){
            return true;
        }
        return false;
    }

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;

    // Compact COO storage (for Push-type Fock construction)
    unsigned long long* d_coo_keys_;    ///< Compact canonical keys
    real_t*             d_coo_values_;  ///< Compact ERI values
    size_t              num_entries_;   ///< Number of unique ERI entries

    // Hash table (kept for O(1) random access lookup, e.g. Post-HF)
    unsigned long long* d_hash_keys_;   ///< Full hash table keys
    real_t*             d_hash_values_; ///< Full hash table values
    size_t              hash_capacity_mask_; ///< capacity - 1

    // Indexed: non-empty slot indices (for Indexed method)
    size_t*             d_nonzero_indices_; ///< Indices of non-empty slots
    size_t              num_nonzero_;   ///< Number of non-empty slots

    HashFockMethod      hash_fock_method_ = HashFockMethod::Compact;

    // CPU fallback: a full 4D AO ERI tensor built in precomputation() when
    // gpu_available() is false.  The hash-table machinery above is skipped
    // on CPU and the Fock path uses this tensor via computeFockMatrix_RHF.
    real_t*             d_eri_cpu_tensor_ = nullptr;
};





/**
 * @brief ERI_RI class for the electron repulsion integrals (ERIs) using the Resolution of Identity (RI) method
 */
class ERI_RI_Direct: public ERI {
public:

    ERI_RI_Direct(const HF& hf, const Molecular& auxiliary_molecular); ///< Constructor

    ERI_RI_Direct(const ERI_RI_Direct&) = delete; ///< copy constructor is deleted
    virtual ~ERI_RI_Direct(); ///< destructor

    void precomputation() override;

    DeviceHostMemory<PrimitiveShell>& get_auxiliary_primitive_shells() { return auxiliary_primitive_shells_; } ///< Get the auxiliary primitive shells
    int get_num_auxiliary_basis() { return num_auxiliary_basis_; }

    std::string get_algorithm_name() override { return "Direct-RI"; } ///< Get the algorithm name

    /// On CPU, lazily reconstruct the dense AO ERI tensor from
    /// intermediate_matrix_B_cpu_ so the base-class build_mo_eri can be used
    /// as a uniform post-HF fallback.  On GPU returns nullptr (unused).
    const real_t* get_eri_matrix_device() const override;

    bool supports_post_hf_method(PostHFMethod method) const override {
        return method == PostHFMethod::None || method == PostHFMethod::MP2
            || method == PostHFMethod::SCS_MP2 || method == PostHFMethod::SOS_MP2;
    }

protected:
    const HF& hf_; ///< HF. This excludes MOs.
    const int num_basis_;
    const int num_auxiliary_basis_;

    const std::vector<ShellTypeInfo> auxiliary_shell_type_infos_; ///< Shell type info in the primitive shell list
    DeviceHostMemory<PrimitiveShell> auxiliary_primitive_shells_; ///< Primitive shells
    DeviceHostMemory<real_t> auxiliary_cgto_normalization_factors_; ///< Normalization factors of the contracted Gauss functions

    // Suzuki.
    DeviceHostMemory<real_t> schwarz_upper_bound_factors;
    DeviceHostMemory<real_t> auxiliary_schwarz_upper_bound_factors;

    //suzuki
    DeviceHostMemory<real_t> two_center_eris;
    DeviceHostMemory<real_t> two_center_eris_inverse;
    DeviceHostMemory<size_t2> primitive_shell_pair_indices;

    // 初回用
    DeviceHostMemory<real_t> schwarz_upper_bound_factors_for_SAD_K_computation;
    DeviceHostMemory<size_t2> primitive_shell_pair_indices_for_SAD_K_computation;

    // CPU-only: cached RI B matrix for the fallback Fock path.
    // On GPU the specialized Direct/SemiDirect/Hash kernels do not need this
    // buffer so we allocate it as a 1x1 placeholder; on CPU it is built in
    // precomputation() and reused every SCF iteration.
    DeviceHostMatrix<real_t> intermediate_matrix_B_cpu_;

    // CPU-only: lazily reconstructed dense AO ERI tensor (nao² x nao²) built
    // from intermediate_matrix_B_cpu_ via B^T * B.  Used by post-HF methods
    // that go through get_eri_matrix_device() + base-class build_mo_eri.
    mutable real_t* d_eri_reconstructed_cpu_ = nullptr;
};



} // namespace gansu
