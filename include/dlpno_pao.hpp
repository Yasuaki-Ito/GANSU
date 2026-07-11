/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <vector>
#include "types.hpp"

namespace gansu {

/**
 * @brief Global Projected Atomic Orbitals (PAOs) in AO basis.
 *
 *   |μ̃⟩ = (1 − P̂_occ) |χ_μ⟩,
 *   where P̂_occ = Σ_{i ∈ occ} |φ_i⟩ ⟨φ_i| is the closed-shell occupied
 *   projector. In matrix form (assuming D_occ ≡ C_occ C_occ^T):
 *
 *       C̃^{PAO} = I − D_occ · S,
 *
 *   shape (nao × nao), with column μ giving the AO-basis coefficients of
 *   the μ-th PAO. The set of nao PAOs is redundant; rank = N_vir.
 *
 * @param C_occ  [nao × nocc] occupied MO coefficients (row-major).
 * @param S      [nao × nao]  AO overlap (row-major, symmetric).
 * @param nao, nocc  basis sizes.
 * @return [nao × nao] row-major C̃^{PAO}.
 */
std::vector<real_t> build_pao_global(
    const real_t* C_occ,
    const real_t* S,
    int nao, int nocc);

/**
 * @brief Global PAOs restricted to an embedding-cluster VIRTUAL subspace
 *        (DMET×DLPNO Phase 1a, rectangular C).
 *
 *   |μ̃⟩ = P̂_vir^emb |χ_μ⟩,  P̂_vir^emb = Σ_{a ∈ cluster vir} |φ_a⟩⟨φ_a|
 *
 *       C̃^{PAO,emb} = (C_vir · C_vir^T) · S,
 *
 *   shape (nao × nao), rank = nmo − nocc_full. In the square (complete-C)
 *   limit D_occ + D_vir = S⁻¹, so this equals the legacy I − D_occ·S exactly;
 *   for rectangular C it removes the environment complement that
 *   I − D_occ·S would leak into the PAO span. Redundancy handled downstream
 *   by orthogonalize_pao_domain (t_cut_do), unchanged.
 *
 * @param C_emb   [nao × nmo] row-major cluster MO coefficients (ALL columns).
 * @param S       [nao × nao] AO overlap (row-major, symmetric).
 * @param nao, nmo  AO count / cluster MO count.
 * @param nocc_full cluster occupied count INCLUDING frozen (virtuals =
 *                  columns [nocc_full, nmo)).
 * @return [nao × nao] row-major C̃^{PAO,emb}.
 */
std::vector<real_t> build_pao_cluster_virtual(
    const real_t* C_emb,
    const real_t* S,
    int nao, int nmo, int nocc_full);

/// Result of per-domain PAO Löwdin orthogonalisation.
struct PAODomainResult {
    /// Orthonormal PAO expansion in global AO basis: shape [nao × n_kept],
    /// row-major. Columns satisfy C^T S C = I.
    std::vector<real_t> C_pao_orth;
    int n_kept = 0;
    int n_redundant_dropped = 0;
    /// Sorted eigenvalues of the domain PAO overlap (descending).
    std::vector<real_t> overlap_eigenvalues;
};

/**
 * @brief Symmetric (Löwdin) orthogonalisation of a domain-restricted PAO set.
 *
 * Given the global PAO matrix `C_pao_global` of shape [nao × nao] and a list
 * of domain AO indices {μ_1, …, μ_d}, build an orthonormal set spanning the
 * subspace ⟨ μ̃_{μ_1}, …, μ̃_{μ_d} ⟩ with linearly dependent components
 * removed via overlap-eigenvalue truncation.
 *
 * Algorithm:
 *   1. Restrict columns of C̃^{PAO} to the domain → C̃_dom (nao × d).
 *   2. Build S_dom = C̃_dom^T · S · C̃_dom (d × d, symmetric PSD).
 *   3. Eigendecompose S_dom = V Λ V^T.
 *   4. Drop eigenvalues with λ_k < t_cut_do.
 *   5. Form M = V_kept · diag(1/√λ_kept).  Result C^{orth} = C̃_dom · M.
 */
PAODomainResult orthogonalize_pao_domain(
    const real_t* C_pao_global,
    const real_t* S,
    const std::vector<int>& domain_ao_indices,
    int nao,
    real_t t_cut_do);

} // namespace gansu
