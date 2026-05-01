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
    int verbose = 1);

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
 * mirrors solve_ccsd_lambda_cpu term-for-term but each PySCF einsum is a
 * CUDA kernel parallelized over output indices, with cuBLAS DGEMM for the
 * vvvv contractions in m3.
 */
bool solve_ccsd_lambda_gpu(
    int nocc, int nvir,
    const real_t* d_eps,        // [nao]
    const real_t* d_eri_mo,     // [nao^4] full MO ERI (chemist (pq|rs))
    const real_t* d_t1,         // [nocc*nvir]
    const real_t* d_t2,         // [nocc^2 * nvir^2]
    real_t* d_lambda1,          // [nocc*nvir] — output
    real_t* d_lambda2,          // [nocc^2 * nvir^2] — output
    int max_iter = 100,
    real_t tol = 1e-8,
    int verbose = 1);

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
 * @brief Build CCSD 2-RDM in PySCF convention (CPU).
 * dm2[p,q,r,s] = γ[p,q]*γ[r,s] - 0.5*γ[p,s]*γ[r,q] + cumulant
 * Energy: E = einsum('pq,qp',h_core,dm1) + 0.5*einsum('pqrs,pqrs',eri,dm2) + E_nuc
 */
void build_ccsd_2rdm_pyscf_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_l1, const real_t* h_l2,
    const real_t* dm1,  // [na × na] 1-RDM
    real_t* D2);        // [na^4] output

real_t compute_dmet_fragment_energy(
    int nocc, int nvir,
    const real_t* eri_emb,   // [na^4] canonical MO ERI, chemist: eri[p,q,r,s]=(pq|rs)
    const real_t* t1,        // [nocc_active × nvir] T1 amplitudes
    const real_t* t2,        // [nocc_active^2 × nvir^2] T2 amplitudes
    int n_frag,              // number of fragment AOs in embedding space
    const real_t* eigvecs,   // [na × na] h_emb eigenvectors: U[p,i] = eigvecs[p*na+i]
    int n_frozen = 0);       // frozen core orbitals (σ≈1, first n_frozen MOs)

} // namespace gansu
