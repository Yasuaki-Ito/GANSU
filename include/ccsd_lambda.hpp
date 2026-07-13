/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ccsd_lambda.hpp
 * @brief CCSD Lambda equations solver and 1-RDM (non-relaxed correlation density).
 *
 * For DMET and property calculations. Solves the Λ equations
 *     0 = ∂⟨Φ₀|(1+Λ) H̄|Φ₀⟩ / ∂t^μ
 * given converged CCSD T1/T2 amplitudes, then constructs the spatial-orbital
 * 1-RDM D_pq = ⟨Φ₀|(1+Λ) e^{-T} a†_p a_q e^T|Φ₀⟩ in both MO and AO bases.
 *
 * See AQUA/CCSD_lambda.md for derivation.
 */

#pragma once

#include "types.hpp"

namespace gansu { class ERI_RI; }   // eri.hpp — optional RI block source (DMET B-native)

namespace gansu {

/**
 * @brief Solve CCSD Lambda equations on CPU (Eigen reference implementation).
 *
 * Inputs are host-side MO integrals and T amplitudes. Lambda amplitudes are
 * written into the provided host buffers on exit. Solver iterates until
 * ||residual|| < tol or max_iter reached.
 *
 * Storage convention (row-major, spatial orbitals):
 *   t1[i*nvir + a],   λ1[i*nvir + a]
 *   t2[i,j,a,b] = t2[((i*nocc+j)*nvir+a)*nvir+b]
 *   eri[pqrs] (physicist ⟨pq|rs⟩ or chemist (pq|rs)) — see implementation
 *
 * @return true if converged
 */
bool solve_ccsd_lambda_cpu(
    int nocc, int nvir,
    const real_t* h_eps,        // [nao] orbital energies
    const real_t* h_eri_mo,     // [nao^4] full MO ERI (chemist notation (pq|rs))
    const real_t* h_t1,         // [nocc*nvir]
    const real_t* h_t2,         // [nocc^2 * nvir^2]
    real_t* h_lambda1,          // [nocc*nvir] — output
    real_t* h_lambda2,          // [nocc^2 * nvir^2] — output
    int max_iter = 100,
    real_t tol = 1e-8,
    int verbose = 1,
    const real_t* h_fov_active = nullptr);  // [nocc*nvir] off-diag Fock f_ov.
                                            // For semi-canonical (DMET cluster):
                                            //   v1 -= Σ_j fov[j,a]*t1[j,b]
                                            //   v2 += Σ_b fov[i,b]*t1[j,b]
                                            //   v4 += fov
                                            //   l1new += fov  (Brillouin)
                                            // Default nullptr → canonical Lambda.

/**
 * @brief Build non-relaxed CCSD 1-RDM in MO basis (CPU).
 *
 * D_MO is (nocc+nvir) × (nocc+nvir). HF reference is included: occupied block
 * starts from 2*δ_ij (doubly occupied), corrected by -t·λ and -τ·λ terms.
 *
 * @param D_mo_out [nao × nao] output, row-major
 */
void build_ccsd_1rdm_mo_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_lambda1, const real_t* h_lambda2,
    real_t* D_mo_out);

/**
 * @brief Transform MO density to AO basis: D_AO = C · D_MO · C^T (CPU).
 */
void transform_density_mo_to_ao_cpu(
    int nao,
    const real_t* h_C,          // [nao × nao] MO coefficients, row-major, col=MO
    const real_t* h_D_mo,       // [nao × nao]
    real_t* h_D_ao_out);        // [nao × nao]

/**
 * @brief Solve CCSD Lambda equations on GPU (cuBLAS + CUDA kernels).
 *
 * Same interface as the CPU version but takes device pointers. Implementation
 * mirrors solve_ccsd_lambda_cpu term-for-term but each einsum is a
 * CUDA kernel parallelized over output indices, with cuBLAS DGEMM for the
 * vvvv contractions in m3.
 */
bool solve_ccsd_lambda_gpu(
    int nocc, int nvir,
    const real_t* d_eps,        // [nao]
    const real_t* d_eri_mo,     // [nao^4] full MO ERI (chemist (pq|rs));
                                //   may be nullptr when eri_block_src is given
    const real_t* d_t1,         // [nocc*nvir]
    const real_t* d_t2,         // [nocc^2 * nvir^2]
    real_t* d_lambda1,          // [nocc*nvir] — output
    real_t* d_lambda2,          // [nocc^2 * nvir^2] — output
    int max_iter = 100,
    real_t tol = 1e-8,
    int verbose = 1,
    const real_t* d_fov_active = nullptr,  // [nocc*nvir] semi-canonical f_ov, nullptr for canonical
    // (DMET B-native, 2026-07-13) Optional RI block source: when both are set,
    // the 7 MO-ERI sub-blocks are built on the fly from the half-transformed
    // B_mo (ERI_RI::mo_eri_block_into) — the full NA⁴ tensor is never
    // materialized on device. nmo_full must equal nocc+nvir (the orbital set
    // B_mo was built from). Identical block index ranges to the dense extract.
    const ERI_RI* eri_block_src = nullptr,
    const real_t* d_B_mo_blocks = nullptr,
    int nmo_full = 0);

/**
 * @brief Build CCSD 1-RDM in MO basis on GPU.
 */
void build_ccsd_1rdm_mo_gpu(
    int nocc, int nvir,
    const real_t* d_t1, const real_t* d_t2,
    const real_t* d_lambda1, const real_t* d_lambda2,
    real_t* d_D_mo_out);        // [nao × nao] device

/**
 * @brief Build CCSD 2-RDM in MO basis (CPU, for DMET democratic partitioning).
 *
 * Γ[p,q,r,s] = ⟨a†_p a†_r a_s a_q⟩ in chemist notation (pq|rs).
 * Includes HF reference. For embedding spaces (n_emb ~ 10-30), CPU is fast enough.
 *
 * @param D2_mo_out [nao^4] output, row-major Γ[p*nao³+q*nao²+r*nao+s]
 */
void build_ccsd_2rdm_mo_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_l1, const real_t* h_l2,
    real_t* D2_mo_out);

/**
 * @brief Compute DMET fragment correlation energy via democratic partitioning
 *        using T amplitudes directly (no Lambda/RDM needed).
 *
 * P_can[i,j] = Σ_{p∈frag} U[p,i] * U[p,j]  (U = h_emb eigenvectors)
 * E_corr_frag = Σ_{i,i'∈occ} P_can[i,i'] * Σ_{j∈occ,a,b∈vir}
 *               (ia|jb) * (2*τ[i',j,a,b] - τ[i',j,b,a])
 * where τ = t2 + t1⊗t1.
 * For P=I, this exactly recovers the full CCSD correlation energy.
 */
/**
 * @brief Build CCSD 2-RDM in chemist convention (CPU).
 *
 * Computes dm2[p,q,r,s] = <a†_p a†_r a_s a_q> with the final transpose(1,0,3,2)
 * applied so that:
 *     E_total = einsum('pq,qp', h_core, dm1) + 0.5 * einsum('pqrs,pqrs', eri, dm2)
 *             + E_nuc
 * Includes HF reference. Verified element-wise.
 *
 * @param dm1 [na × na] CCSD 1-RDM (with HF reference) — must come from
 *            build_ccsd_1rdm_mo_cpu before calling this.
 * @param D2  [na^4] output, row-major D2[((p*na+q)*na+r)*na+s]
 */
void build_ccsd_2rdm_chemist_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_l1, const real_t* h_l2,
    const real_t* dm1,
    real_t* D2);

real_t compute_dmet_fragment_energy(
    int nocc, int nvir,
    const real_t* eri_emb,   // [na^4] canonical MO ERI, chemist: eri[p,q,r,s]=(pq|rs)
    const real_t* t1,        // [nocc_active × nvir] T1 amplitudes
    const real_t* t2,        // [nocc_active^2 × nvir^2] T2 amplitudes
    int n_frag,              // number of fragment AOs in embedding space
    const real_t* eigvecs,   // [na × na] h_emb eigenvectors: U[p,i] = eigvecs[p*na+i]
    int n_frozen = 0,        // frozen core orbitals (σ≈1, first n_frozen MOs)
    const real_t* fov_active = nullptr);  // [nocc_active × nvir] semi-canonical f_ov.
                                          // When non-null, adds the Brillouin term
                                          //   2 Σ_{i,i',a} P[i+nf,i'+nf] · f_ov[i,a] · t1[i',a]
                                          // to the democratic CCSD correlation energy
                                          // (required for non-zero f_ov, e.g. DMET clusters
                                          // with Vayesta-style canonicalization).

/**
 * @brief AO-projected DMET fragment correlation energy via full 1-RDM and 2-RDM.
 *
 * Compared to compute_dmet_fragment_energy (T-amplitude form), this:
 *   - Uses Lambda-relaxed dm1 and dm2 (full CCSD density matrices)
 *   - Includes the 1-electron (h × dm1) contribution
 *
 * Formula (chemist convention, derived via U-orthonormality so that no explicit
 *          back-transform from canonical to embedding basis is needed):
 *   E_corr_frag = E_frag(D1, D2) − E_frag(D1_HF, D2_HF)
 *   E_frag(D1, D2) = Σ_{i,i'} P[i,i'] (h_can D1)[i,i']
 *                  + (1/2) Σ_{i,i'} P[i,i'] Σ_{jkl} eri_can[i,j,k,l] D2[i',j,k,l]
 *   P[i,i'] = Σ_{p<n_frag} U[p,i] U[p,i']
 *   h_can = U^T h_emb_1e U
 *
 * Must be passed the same 1e operator that the CCSD/embedding-HF used as
 * reference (h_emb_base = C^T F C in the current code) so that
 *  − the orbital eigenstates U are exact eigenstates of h_can + 2J(D)−K(D),
 *  − E_corr_frag has the standard CCSD-correlation sign (≤ 0).
 * Passing h_core_emb instead measures the physical Hamiltonian's energy with
 * the F_emb-HF reference, which is NOT a correlation energy and can be
 * positive — verified empirically on benzene/STO-3G.
 *
 * Σ over fragments recovers the full embedding-basis CCSD energy.
 *
 * @param nocc_act     Active occupied dimension (nocc - n_frozen)
 * @param nvir         Virtual dimension
 * @param h_emb_1e     [ne × ne] embedding 1e operator (h_emb_base = C^T F C)
 * @param eri_can      [ne⁴]    canonical-basis ERI (chemist), as fed to CCSD
 * @param dm1_active   [na_act²] canonical 1-RDM with active HF reference
 * @param dm2_active   [na_act⁴] canonical 2-RDM with active HF reference
 * @param n_frag       First n_frag indices = fragment AOs in embedding
 * @param eigvecs      [ne × ne] U[p,i] (canonical i in embedding p)
 * @param n_frozen     Frozen core orbitals (placed at canonical indices [0, n_frozen))
 * @param E1_out       (optional) decomposed 1-electron contribution for diagnostics
 * @param E2_out       (optional) decomposed 2-electron contribution for diagnostics
 */
real_t compute_dmet_fragment_energy_aoproj(
    int nocc_act, int nvir,
    const real_t* h_emb_1e,
    const real_t* eri_can,
    const real_t* dm1_active,
    const real_t* dm2_active,
    int n_frag,
    const real_t* eigvecs,
    int n_frozen = 0,
    real_t* E1_out = nullptr,
    real_t* E2_out = nullptr);


/**
 * @brief Vayesta-convention DMET fragment correlation energy.
 *
 * Standard quantum-chemistry DMET energy partition (Knizia/Wouters/Vayesta):
 *
 *   e1 = Tr[P · h_avg · D1]                with h_avg = ½(h_bare + h_eff)
 *   e2 = ½ · Σ P[p,t] · eri_can[t,q,r,s] · D2[p,q,r,s]
 *   E_corr = (e1 + e2) − (e1_HF + e2_HF)
 *
 *   h_bare = U^T · (C_emb^T · h_core · C_emb) · U      (canonical bare core)
 *   h_eff  = U^T · h_emb_1e · U − v_act                 (cluster-local HF removed)
 *   v_act[i,j] = 2·Σ_{k∈occ} eri_can[k,k,i,j] − Σ_{k∈occ} eri_can[k,j,i,k]
 *
 * Differs from `compute_dmet_fragment_energy_aoproj` in two ways:
 *   (a) takes h_core_emb in addition to h_emb_1e (= F projected to embedding)
 *   (b) builds v_act from the canonical ERI to remove the cluster-internal HF
 *       interaction, recovering the bare 1-body operator on average
 *
 * Validated against Vayesta (benzene/STO-3G, 6 CH SAO fragments, oneshot CCSD)
 * which gives e_corr = −0.4310 Ha (102 % of full CCSD).  GANSU's existing
 * AO-proj formula gives only −0.067 Ha (16 %) because it uses h_F = U^T F U
 * directly, mixing in the global Hartree-exchange that should be cancelled
 * by v_act.
 */
real_t compute_dmet_fragment_energy_vayesta(
    int nocc_act, int nvir,
    const real_t* h_core_emb,    // C_emb^T · h_core · C_emb (μ-independent)
    const real_t* h_emb_1e,      // C_emb^T · F_full · C_emb (or with μ-shift)
    const real_t* eri_can,
    const real_t* dm1_active,
    const real_t* dm2_active,
    int n_frag,
    const real_t* eigvecs,
    int n_frozen = 0,
    real_t* E1_out = nullptr,
    real_t* E2_out = nullptr);

} // namespace gansu
