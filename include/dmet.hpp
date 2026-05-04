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

#include <vector>
#include <string>
#include <tuple>
#include "types.hpp"

namespace gansu {

class RHF;
class ERI;

/// Fragment definition: a group of atoms mapped to AO indices
struct DMETFragment {
    std::vector<int> atom_indices;   ///< Atom indices belonging to this fragment
    std::vector<int> ao_indices;     ///< AO basis function indices (derived from atom_indices)
    int n_frag;                      ///< Number of fragment AO basis functions
};

/// Result from a single fragment CCSD solve
struct FragmentResult {
    real_t E_corr;    ///< CCSD correlation energy
    real_t N_frag;    ///< Fragment electron count from 1-RDM
};

/// DMET-CCSD driver
/// Works with any ERI method (Stored, RI, Direct, Hash) via ERI::build_mo_eri().
class DMET {
public:
    DMET(RHF& rhf, const ERI& eri);

    /// Run DMET-CCSD and return total correlation energy
    real_t compute_energy();

private:
    RHF& rhf_;
    const ERI& eri_;
    int num_basis_;
    int num_occ_;
    int num_atoms_;
    double svd_threshold_;
    std::vector<DMETFragment> fragments_;

    static std::vector<DMETFragment> parse_fragments(const std::string& spec, int num_atoms);

    /// Automatic atom-based fragmentation: each heavy atom becomes a fragment,
    /// and each H is grouped with its closest heavy atom (within bond threshold).
    /// Yields the standard "atomic fragment" partitioning used in most DMET papers.
    /// H atoms not within bond_threshold of any heavy atom become singleton fragments.
    /// @param bond_threshold_bohr Maximum X-H bond length (default 2.6 Bohr ≈ 1.38 Å,
    ///        covers C-H/N-H/O-H/S-H but not heavy-heavy bonds at ≥ 2.7 Bohr)
    static std::vector<DMETFragment> auto_fragments_by_bonds(
        const RHF& rhf, real_t bond_threshold_bohr = 2.6);

    std::tuple<std::vector<real_t>, int, int, int>
    build_bath_orbitals(const DMETFragment& frag,
                        const real_t* h_C,
                        const real_t* S_inv_half,  // precomputed S^{-1/2} [nao × nao]
                        const real_t* S_half,       // precomputed S^{1/2} [nao × nao]
                        int nao, int nocc);

    /// Solve fragment CCSD with chemical potential μ.
    /// μ is subtracted from the fragment diagonal of h_emb (μ>0 → more electrons in fragment).
    /// Returns correlation energy and fragment electron count from CCSD 1-RDM.
    FragmentResult solve_fragment_ccsd(
        const real_t* h_C_emb,     // [nao × n_emb]
        int n_emb, int n_emb_occ,
        int n_frozen,              // frozen core orbitals (σ ≈ 1)
        const real_t* h_fock,      // [nao × nao]
        const DMETFragment& frag,
        real_t mu = 0.0) const;

    // TODO: evaluate_at_mu for chemical potential optimization (requires stable Lambda/1-RDM)
};

} // namespace gansu
