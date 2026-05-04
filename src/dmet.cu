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

#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <map>
#include <utility>
#include <omp.h>
#include <Eigen/Dense>

// Compile-time guard: NVCC must propagate -fopenmp to the host compiler when
// compiling .cu files (via OpenMP::OpenMP_CUDA target in CMakeLists.txt).
// Without it, #pragma omp parallel is silently ignored and the fragment loop
// collapses to a single thread.
#ifndef _OPENMP
#  error "_OPENMP not defined — link OpenMP::OpenMP_CUDA in CMakeLists.txt"
#endif

#include "dmet.hpp"
#include "rhf.hpp"
#include "eri.hpp"
#include "gpu_manager.hpp"
#include "multi_gpu_manager.hpp"
#include "ccsd_lambda.hpp"
#include "utils.hpp"

namespace gansu {

// Forward declarations (defined in eri_stored.cu)
real_t ccsd_spatial_orbital(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed,
                            int num_frozen,
                            const real_t* h_fov_active = nullptr);

// RAII guard: redirect std::cout to a discarded sink for the lifetime of the
// object. Used to silence ccsd_spatial_orbital's per-iteration output during
// the bisection inner loop, where its many lines would drown out the bisection
// progress messages.
//
// IMPORTANT: the sink must be thread-safe — std::cout is global, and the
// fragment-parallel OpenMP region inside `evaluate_at_mu` has multiple threads
// concurrently writing to it (CCSD/Lambda iter prints). The previous
// implementation used a std::stringstream sink, whose stringbuf appends to a
// shared std::string non-atomically; concurrent writes corrupted the string's
// internal size/capacity, causing glibc heap errors (`corrupted size vs.
// prev_size`) on later allocations. The NullBuf below has no state, so
// overflow / xsputn are trivially safe under concurrent calls.
class CoutSilencer {
    struct NullBuf : std::streambuf {
        int_type overflow(int_type c) override {
            return traits_type::not_eof(c);
        }
        std::streamsize xsputn(const char_type*, std::streamsize n) override {
            return n;
        }
    };
    std::streambuf* old_buf_ = nullptr;
    NullBuf null_buf_;
public:
    explicit CoutSilencer(bool silence) {
        if (silence) old_buf_ = std::cout.rdbuf(&null_buf_);
    }
    ~CoutSilencer() {
        if (old_buf_) std::cout.rdbuf(old_buf_);
    }
    CoutSilencer(const CoutSilencer&) = delete;
    CoutSilencer& operator=(const CoutSilencer&) = delete;
};

// ============================================================================
//  Embedding HF SCF (CPU, Roothaan iteration)
//
//  When max_iter > 0: solves canonical orbitals of the embedding Hamiltonian
//  H = h_emb + V_emb, i.e. eigenstates of the *self-consistent* Fock matrix
//      F[p,q] = h_emb[p,q] + Σ_rs D[r,s] (eri[p,q,r,s] - 0.5 eri[p,r,s,q])
//  with D[r,s] = 2 Σ_{i<n_occ} C[r,i] C[s,i].
//
//  When max_iter == 0: just diagonalizes h_emb directly (no SCF). This is
//  the "Vayesta convention" path — h_emb is treated as the canonical Fock
//  itself, so ε = eig(h_emb) and the cluster CCSD effectively solves
//  H_cluster = h_eff + cluster_ERI where h_eff = h_emb − v_act (the cluster
//  internal V_HF cancels out implicitly because the CCSD reconstructs Fock
//  as h_eff + V_HF[D_HF_canonical] = h_emb = diag(ε)). Using SCF instead
//  double-counts V_HF[cluster] (once in h_emb_base = C^T F C, once in the
//  Fock update), giving 1.5× over Vayesta on benzene.
//
//  Output: U[p, i] = embedding-basis coefficient of i-th canonical orbital,
//          eps[i]  = orbital energy.
// ============================================================================
// NOTE: As of the Vayesta-canonical refactor, evaluate_at_mu uses
// canonicalize_cluster_subspaces (D_cluster split + h_emb_base canonicalize_mo)
// and no longer performs an embedding-HF SCF. This helper is retained for
// reference and potential future use (e.g. self-consistent DMET).
[[maybe_unused]] static std::pair<std::vector<real_t>, std::vector<real_t>>
run_embedding_hf_cpu(
    const std::vector<real_t>& h_emb,    // [ne × ne] row-major
    const std::vector<real_t>& eri_emb,  // [ne^4] chemist (pq|rs)
    int ne, int n_occ,
    int max_iter, real_t tol, int verbose)
{
    using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMatXd> H(h_emb.data(), ne, ne);

    // Initial guess: diagonalize h_emb
    Eigen::SelfAdjointEigenSolver<RowMatXd> sol0(H);
    RowMatXd C = sol0.eigenvectors();
    Eigen::VectorXd eps = sol0.eigenvalues();

    real_t E_prev = 0.0;
    bool converged = (max_iter == 0);  // direct-diag path: trivially "converged"

    for (int iter = 0; iter < max_iter; iter++) {
        // Density: D[r,s] = 2 Σ_{i<n_occ} C[r,i] C[s,i]
        RowMatXd Cocc = C.leftCols(n_occ);
        RowMatXd D = 2.0 * Cocc * Cocc.transpose();

        // Fock: F[p,q] = h[p,q] + Σ_rs D[r,s] (eri[p,q,r,s] - 0.5 eri[p,r,s,q])
        RowMatXd F = H;
        for (int p = 0; p < ne; p++) {
            for (int q = 0; q < ne; q++) {
                real_t v = 0.0;
                for (int r = 0; r < ne; r++) {
                    for (int s = 0; s < ne; s++) {
                        const real_t Drs = D(r, s);
                        v += Drs * eri_emb[(((size_t)p*ne+q)*ne+r)*ne+s];
                        v -= 0.5 * Drs * eri_emb[(((size_t)p*ne+r)*ne+s)*ne+q];
                    }
                }
                F(p, q) += v;
            }
        }

        // Energy: E = 0.5 Σ_pq D[p,q] (h[p,q] + F[p,q])
        real_t E = 0.5 * (D.cwiseProduct(H + F)).sum();

        // Diagonalize F
        Eigen::SelfAdjointEigenSolver<RowMatXd> sol(F);
        eps = sol.eigenvalues();
        RowMatXd C_new = sol.eigenvectors();

        real_t dE = std::abs(E - E_prev);
        if (verbose >= 2)
            std::cout << "      Emb HF iter " << iter << ": E=" << std::setprecision(10) << E
                      << " dE=" << std::setprecision(3) << dE << std::endl;

        C = C_new;
        if (iter > 0 && dE < tol) { converged = true; break; }
        E_prev = E;
    }

    if (!converged && verbose >= 1)
        std::cerr << "      [warn] embedding HF not converged (max_iter=" << max_iter
                  << ", final dE=" << std::setprecision(3) << std::abs(E_prev) << ")" << std::endl;

    // Pack into row-major std::vector. Convention matches gpu::eigenDecomposition:
    //   U[p*ne + i] = p-th component of i-th eigenvector (column i of C)
    std::vector<real_t> U(ne * ne);
    std::vector<real_t> eps_v(ne);
    for (int p = 0; p < ne; p++) {
        eps_v[p] = eps(p);
        for (int i = 0; i < ne; i++)
            U[p * ne + i] = C(p, i);
    }
    return {U, eps_v};
}

// ============================================================================
//  Vayesta-convention cluster orbital construction
//
//  Given a cluster basis with M_co = C_emb^T S C_occ_full (μ-independent),
//  produce canonical cluster orbitals U[ne × ne] and energies eps[ne]:
//
//    1. D_cluster = 2 M_co · M_co^T  (cluster representation of full HF DM).
//       Eigenvalues are between 0 and 2; sort descending.
//    2. Top n_emb_occ eigenvectors → occupied subspace U_occ [ne × n_emb_occ].
//       Remaining (ne − n_emb_occ) → virtual subspace U_vir.
//    3. Within each subspace, diagonalize h_emb (= h_emb_base − μ·P_frag) to
//       get canonical orbitals: F_sub = U_sub^T h_emb U_sub → eps, V_sub.
//       Final canonical = U_sub @ V_sub.
//    4. Combine: U[:, :n_occ] = occupied canonical, [:, n_occ:] = virtual.
//
//  This is Vayesta's `canonicalize_mo` (fragment.py:506) applied to occ/vir
//  separately (via `_get_bath_option("canonicalize", "occupied"/"virtual")`).
//  CCSD then implicitly uses h_eff = h_emb − v_act as the cluster 1-body
//  operator, recovering Vayesta's cluster Hamiltonian without V_HF[cluster]
//  double-counting.
//
//  f_ov is generally non-zero between blocks (h_emb couples occ and vir),
//  but f_oo and f_vv are diagonal — sufficient for canonical CCSD with the
//  Brillouin condition assumed for the HF reference.
// ============================================================================
static std::pair<std::vector<real_t>, std::vector<real_t>>
canonicalize_cluster_subspaces(
    const std::vector<real_t>& h_emb,    // [ne × ne] (= h_emb_base − μ·P_frag)
    const std::vector<real_t>& M_co,     // [ne × nocc_full]  C_emb^T S C_occ
    int ne, int nocc_full, int n_emb_occ)
{
    using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    Eigen::Map<const RowMatXd> H(h_emb.data(), ne, ne);
    Eigen::Map<const RowMatXd> M(M_co.data(), ne, nocc_full);

    // 1. Cluster DM: D_cluster = 2 M Mᵀ
    RowMatXd D_cluster = 2.0 * M * M.transpose();

    // 2. Diagonalize, sort descending
    Eigen::SelfAdjointEigenSolver<RowMatXd> dm_solver(D_cluster);
    Eigen::VectorXd dm_eigs = dm_solver.eigenvalues();   // ascending
    RowMatXd dm_vecs = dm_solver.eigenvectors();         // columns are eigenvectors

    // SelfAdjointEigenSolver returns ascending; reverse to get descending.
    // Top n_emb_occ → occupied subspace.
    RowMatXd U_occ(ne, n_emb_occ);
    RowMatXd U_vir(ne, ne - n_emb_occ);
    for (int i = 0; i < n_emb_occ; i++)
        U_occ.col(i) = dm_vecs.col(ne - 1 - i);
    for (int i = 0; i < ne - n_emb_occ; i++)
        U_vir.col(i) = dm_vecs.col(ne - n_emb_occ - 1 - i);

    // 3. Canonicalize each subspace via h_emb
    auto canonicalize = [&H](const RowMatXd& U_sub)
        -> std::pair<RowMatXd, Eigen::VectorXd> {
        if (U_sub.cols() == 0) return {RowMatXd(U_sub.rows(), 0), Eigen::VectorXd(0)};
        RowMatXd F_sub = U_sub.transpose() * H * U_sub;
        Eigen::SelfAdjointEigenSolver<RowMatXd> solver(F_sub);
        return {U_sub * solver.eigenvectors(), solver.eigenvalues()};
    };
    auto [U_occ_can, eps_occ] = canonicalize(U_occ);
    auto [U_vir_can, eps_vir] = canonicalize(U_vir);

    // 4. Combine: U[:, :n_occ] = occupied, U[:, n_occ:] = virtual.
    //    Convention matches GANSU eigenvecs storage: U[p*ne + i] = component p of orbital i.
    std::vector<real_t> U_out((size_t)ne * ne, 0.0);
    std::vector<real_t> eps_out(ne, 0.0);
    for (int i = 0; i < n_emb_occ; i++) {
        eps_out[i] = eps_occ(i);
        for (int p = 0; p < ne; p++) U_out[p * ne + i] = U_occ_can(p, i);
    }
    for (int i = 0; i < ne - n_emb_occ; i++) {
        const int idx = n_emb_occ + i;
        eps_out[idx] = eps_vir(i);
        for (int p = 0; p < ne; p++) U_out[p * ne + idx] = U_vir_can(p, i);
    }
    return {U_out, eps_out};
}

// ============================================================================
//  Fragment parsing (unchanged)
// ============================================================================

std::vector<DMETFragment> DMET::parse_fragments(const std::string& spec, int num_atoms) {
    std::vector<DMETFragment> fragments;
    if (spec.empty()) {
        // Empty spec without auto-detection — fall back to one fragment per atom.
        // The recommended path for empty spec is auto_fragments_by_bonds(),
        // dispatched from the DMET constructor.
        for (int i = 0; i < num_atoms; i++) {
            DMETFragment frag;
            frag.atom_indices.push_back(i);
            fragments.push_back(frag);
        }
        return fragments;
    }

    size_t pos = 0;
    while (pos < spec.size()) {
        while (pos < spec.size() && (spec[pos] == ' ' || spec[pos] == '\t')) pos++;
        if (pos >= spec.size()) break;
        if (spec[pos] != '{')
            throw std::runtime_error("DMET fragment parse error: expected '{' at position " + std::to_string(pos));
        pos++;

        DMETFragment frag;
        while (pos < spec.size() && spec[pos] != '}') {
            while (pos < spec.size() && (spec[pos] == ' ' || spec[pos] == ',')) pos++;
            if (pos >= spec.size() || spec[pos] == '}') break;
            int start_atom = 0;
            while (pos < spec.size() && spec[pos] >= '0' && spec[pos] <= '9') {
                start_atom = start_atom * 10 + (spec[pos] - '0'); pos++;
            }
            if (pos < spec.size() && spec[pos] == '-') {
                pos++;
                int end_atom = 0;
                while (pos < spec.size() && spec[pos] >= '0' && spec[pos] <= '9') {
                    end_atom = end_atom * 10 + (spec[pos] - '0'); pos++;
                }
                for (int a = start_atom; a <= end_atom; a++) {
                    if (a < 0 || a >= num_atoms) throw std::runtime_error("DMET: atom index out of range");
                    frag.atom_indices.push_back(a);
                }
            } else {
                if (start_atom < 0 || start_atom >= num_atoms) throw std::runtime_error("DMET: atom index out of range");
                frag.atom_indices.push_back(start_atom);
            }
        }
        if (pos < spec.size() && spec[pos] == '}') pos++;
        if (!frag.atom_indices.empty()) fragments.push_back(frag);
    }
    return fragments;
}

// ============================================================================
//  Automatic atom-based fragmentation by bond distance
//
//  Algorithm:
//   1. Each heavy atom (atomic_number > 1) becomes its own fragment.
//   2. Each H is bonded to its single closest heavy atom (Euclidean distance
//      < bond_threshold_bohr); the H is appended to that fragment's atom list.
//   3. H atoms with no heavy neighbor within threshold (e.g. isolated H, H₂)
//      become singleton fragments.
//
//  Default threshold 2.6 Bohr ≈ 1.38 Å covers all common X-H bonds:
//      O-H 0.96 Å, N-H 1.01 Å, C-H 1.09 Å, S-H 1.34 Å (all < 1.38 Å)
//  while excluding heavy-heavy bonds (C-C 1.54 Å = 2.91 Bohr is well above).
// ============================================================================
std::vector<DMETFragment> DMET::auto_fragments_by_bonds(
    const RHF& rhf, real_t bond_threshold_bohr)
{
    const auto& atoms = rhf.get_atoms();
    const int N = (int)atoms.size();
    const real_t r2_max = bond_threshold_bohr * bond_threshold_bohr;

    // Phase 1: heavy atoms each become a fragment, mapped via heavy_to_frag.
    std::vector<int> heavy_to_frag(N, -1);
    std::vector<DMETFragment> fragments;
    for (int i = 0; i < N; i++) {
        if (atoms[i].atomic_number > 1) {
            DMETFragment frag;
            frag.atom_indices.push_back(i);
            heavy_to_frag[i] = (int)fragments.size();
            fragments.push_back(std::move(frag));
        }
    }

    // Phase 2: assign each H to its closest heavy neighbor (within threshold).
    int orphan_h = 0;
    for (int h = 0; h < N; h++) {
        if (atoms[h].atomic_number != 1) continue;
        const auto& Hc = atoms[h].coordinate;
        int closest = -1;
        real_t min_r2 = r2_max;  // strict-less-than below preserves bond cutoff
        for (int j = 0; j < N; j++) {
            if (atoms[j].atomic_number <= 1) continue;
            const auto& Xc = atoms[j].coordinate;
            const real_t dx = Hc.x - Xc.x;
            const real_t dy = Hc.y - Xc.y;
            const real_t dz = Hc.z - Xc.z;
            const real_t r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < min_r2) { min_r2 = r2; closest = j; }
        }
        if (closest >= 0) {
            fragments[heavy_to_frag[closest]].atom_indices.push_back(h);
        } else {
            // Orphan H: no heavy atom within bond_threshold.
            DMETFragment frag;
            frag.atom_indices.push_back(h);
            fragments.push_back(std::move(frag));
            orphan_h++;
        }
    }

    // Print a concise summary so the user can verify the partitioning.
    std::cout << "  Auto-detected " << fragments.size()
              << " fragment(s) by X-H bond proximity (threshold "
              << bond_threshold_bohr << " Bohr ≈ "
              << std::fixed << std::setprecision(2)
              << bond_threshold_bohr / 1.8897259886 << " Å)";
    if (orphan_h > 0)
        std::cout << ", " << orphan_h << " orphan H atom(s)";
    std::cout << std::defaultfloat << std::endl;
    // Element-symbol summary so heteroatoms are visible at a glance.
    std::map<std::string, int> by_signature;
    for (const auto& frag : fragments) {
        std::string sig;
        std::map<std::string, int> ec;
        for (int a : frag.atom_indices)
            ec[atomic_number_to_element_name(atoms[a].atomic_number)]++;
        for (const auto& [el, n] : ec) {
            if (!sig.empty()) sig += "+";
            sig += el;
            if (n > 1) sig += std::to_string(n);
        }
        by_signature[sig]++;
    }
    std::cout << "  Fragment signatures:";
    for (const auto& [sig, n] : by_signature)
        std::cout << " " << sig << "×" << n;
    std::cout << std::endl;

    return fragments;
}

// ============================================================================
//  Constructor
// ============================================================================

DMET::DMET(RHF& rhf, const ERI& eri)
    : rhf_(rhf), eri_(eri),
      num_basis_(rhf.get_num_basis()),
      num_occ_(rhf.get_num_electrons() / 2),
      num_atoms_((int)rhf.get_atom_to_basis_range().size()),
      svd_threshold_(rhf.get_dmet_threshold())
{
    const std::string& spec = rhf.get_dmet_fragments_str();
    if (spec.empty()) {
        // Default: automatic atomic fragmentation by X-H bond proximity.
        fragments_ = auto_fragments_by_bonds(rhf);
    } else {
        // User-specified fragments override the auto-detection.
        fragments_ = parse_fragments(spec, num_atoms_);
    }
    const auto& a2b = rhf.get_atom_to_basis_range();
    for (auto& frag : fragments_) {
        frag.ao_indices.clear();
        for (int atom : frag.atom_indices)
            for (size_t mu = a2b[atom].start_index; mu < a2b[atom].end_index; mu++)
                frag.ao_indices.push_back((int)mu);
        frag.n_frag = (int)frag.ao_indices.size();
    }
}

// ============================================================================
//  Schmidt decomposition (Löwdin basis) — unchanged
// ============================================================================

std::tuple<std::vector<real_t>, int, int, int>
DMET::build_bath_orbitals(const DMETFragment& frag,
                          const real_t* h_C,
                          const real_t* S_inv_half,
                          const real_t* S_half,
                          int nao, int nocc)
{
    const int n_frag = frag.n_frag;

    if (n_frag == nao) {
        return {std::vector<real_t>(), nocc, 0, 0};
    }

    // --- C_lo_occ = S^{1/2} C_occ ---
    std::vector<real_t> C_lo_occ(nao * nocc, 0.0);
    for (int mu = 0; mu < nao; mu++)
        for (int j = 0; j < nocc; j++) {
            real_t val = 0.0;
            for (int nu = 0; nu < nao; nu++)
                val += S_half[mu * nao + nu] * h_C[nu * nao + j];
            C_lo_occ[mu * nocc + j] = val;
        }

    // --- SVD of C_lo_occ[frag,:] → all σ ∈ [0,1] guaranteed ---
    const int k = std::min(n_frag, nocc);
    std::vector<real_t> sigma(k);
    std::vector<real_t> Vt(k * nocc);
    {
        std::vector<real_t> AtA(nocc * nocc, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j <= i; j++) {
                real_t val = 0.0;
                for (int p = 0; p < n_frag; p++) {
                    int mu = frag.ao_indices[p];
                    val += C_lo_occ[mu * nocc + i] * C_lo_occ[mu * nocc + j];
                }
                AtA[i * nocc + j] = val;
                AtA[j * nocc + i] = val;
            }

        std::vector<real_t> eigvals(nocc), eigvecs(nocc * nocc);
        real_t *d_AtA = nullptr, *d_ev = nullptr, *d_evc = nullptr;
        tracked_cudaMalloc(&d_AtA, nocc * nocc * sizeof(real_t));
        tracked_cudaMalloc(&d_ev, nocc * sizeof(real_t));
        tracked_cudaMalloc(&d_evc, nocc * nocc * sizeof(real_t));
        cudaMemcpy(d_AtA, AtA.data(), nocc * nocc * sizeof(real_t), cudaMemcpyHostToDevice);
        gpu::eigenDecomposition(d_AtA, d_ev, d_evc, nocc);
        cudaMemcpy(eigvals.data(), d_ev, nocc * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(eigvecs.data(), d_evc, nocc * nocc * sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_AtA); tracked_cudaFree(d_ev); tracked_cudaFree(d_evc);

        for (int i = 0; i < k; i++) {
            int rev = nocc - 1 - i;
            sigma[i] = (eigvals[rev] > 0) ? std::sqrt(eigvals[rev]) : 0.0;
            for (int j = 0; j < nocc; j++)
                Vt[i * nocc + j] = eigvecs[j * nocc + rev];
        }
    }

    std::cout << "  SVD singular values:";
    for (int i = 0; i < k; i++) std::cout << " " << std::fixed << std::setprecision(6) << sigma[i];
    std::cout << std::defaultfloat << std::endl;

    // --- Count bath / core / discarded ---
    // The naive (C_lo_occ V) bath orbital includes weight on fragment AOs
    // (= U_A · σ from the SVD), so it is not orthogonal to the fragment basis.
    // We project out the fragment component → bath_proj has weight only on
    // environment AOs → ⟨frag | bath⟩ = 0 in Löwdin basis. Normalization
    // factor is √(1 − σ²) (environment-component norm).
    //
    // For σ → 1, the environment component vanishes (orbital is fragment-
    // localized), so we exclude it from bath and count it as n_core. Use a
    // dedicated core_threshold (1e-3) since the user-facing svd_threshold_
    // (default 1e-6) is too small to catch σ ≈ 1 ↔ Schmidt-pair degeneracy.
    const real_t core_threshold = 1e-3;
    int n_bath = 0, n_core = 0;
    for (int i = 0; i < k; i++) {
        if (sigma[i] > svd_threshold_ && sigma[i] < 1.0 - core_threshold)
            n_bath++;
        else if (sigma[i] >= 1.0 - core_threshold)
            n_core++;
    }

    int n_emb = n_frag + n_bath;
    // Cluster's HF occupation count (Schmidt convention): each σ ≈ 1 SVD pair
    // contributes 1 fully-fragment-localized occupied direction (no bath partner),
    // plus each σ < 1 pair contributes 1 occupied bath/frag pair. So the cluster's
    // D_cluster has (n_bath + n_core) eigenvalues ≈ 2 and (n_emb − n_bath − n_core)
    // eigenvalues ≈ 0. canonicalize_cluster_subspaces relies on this split, so
    // n_emb_occ MUST equal n_bath + n_core (not just n_bath, which would push the
    // σ ≈ 1 fragment direction — typically a 1s core — into the virtual subspace
    // and produce a HOMO < LUMO inversion that breaks CCSD).
    int n_emb_occ = n_bath + n_core;
    int n_frozen = 0;  // don't freeze in CCSD — keeps Lambda solver simple
                       //  (Lambda doesn't natively support frozen core); the σ ≈ 1
                       //  cluster orbital gets the deepest ε and is correlated.

    // --- Build C_emb in Löwdin basis, then convert to AO ---
    // Fragment: identity on Löwdin AO indices
    // Bath: project (C_lo_occ × V) to env subspace and normalize by √(1−σ²)
    //       → bath orbitals orthogonal to frag AOs in Löwdin basis
    // Convert: C_emb_ao = S^{-1/2} × C_emb_lo
    // Result: C_emb_ao^T S C_emb_ao = C_emb_lo^T C_emb_lo = I  (orthonormal)

    std::vector<real_t> C_emb_lo(nao * n_emb, 0.0);
    std::vector<char> is_frag_ao(nao, 0);
    for (int i = 0; i < n_frag; i++) {
        int mu = frag.ao_indices[i];
        C_emb_lo[mu * n_emb + i] = 1.0;
        is_frag_ao[mu] = 1;
    }
    int bath_col = n_frag;
    for (int i = 0; i < k; i++) {
        if (sigma[i] > svd_threshold_ && sigma[i] < 1.0 - core_threshold) {
            // (C_lo_occ V)[μ, i] for env AOs only; zero on frag AOs
            for (int mu = 0; mu < nao; mu++) {
                if (is_frag_ao[mu]) continue;
                real_t val = 0.0;
                for (int j = 0; j < nocc; j++)
                    val += C_lo_occ[mu * nocc + j] * Vt[i * nocc + j];
                C_emb_lo[mu * n_emb + bath_col] = val;
            }
            // Normalize by √(1 − σ²): (env norm)² = ||C_lo_occ V||² − ||A V||²
            //                                    = 1 − σ² (since ||V||=1, ||AV||=σ)
            const real_t inv_norm = 1.0 / std::sqrt(1.0 - sigma[i] * sigma[i]);
            for (int mu = 0; mu < nao; mu++)
                C_emb_lo[mu * n_emb + bath_col] *= inv_norm;
            bath_col++;
        }
    }

    // C_emb_ao = S^{-1/2} × C_emb_lo
    std::vector<real_t> C_emb(nao * n_emb, 0.0);
    for (int mu = 0; mu < nao; mu++)
        for (int p = 0; p < n_emb; p++) {
            real_t val = 0.0;
            for (int nu = 0; nu < nao; nu++)
                val += S_inv_half[mu * nao + nu] * C_emb_lo[nu * n_emb + p];
            C_emb[mu * n_emb + p] = val;
        }

    return {C_emb, n_emb_occ, n_frozen, n_core};
}

// ============================================================================
//  Fragment CCSD solver with chemical potential
// ============================================================================

FragmentResult DMET::solve_fragment_ccsd(
    const real_t* h_C_emb, int n_emb, int n_emb_occ,
    int n_frozen,
    const real_t* h_fock, const DMETFragment& frag,
    real_t mu) const
{
    const int nao = num_basis_;
    const int n_frag = frag.n_frag;

    if (n_emb_occ <= 0 || n_emb_occ >= n_emb)
        return {0.0, 0.0};

    // 1. h_emb = C_emb^T F C_emb, then subtract μ from fragment diagonal
    std::vector<real_t> tmp(nao * n_emb, 0.0);
    std::vector<real_t> h_emb(n_emb * n_emb, 0.0);

    for (int mu_idx = 0; mu_idx < nao; mu_idx++)
        for (int p = 0; p < n_emb; p++) {
            real_t val = 0.0;
            for (int nu = 0; nu < nao; nu++)
                val += h_fock[mu_idx * nao + nu] * h_C_emb[nu * n_emb + p];
            tmp[mu_idx * n_emb + p] = val;
        }
    for (int p = 0; p < n_emb; p++)
        for (int q = 0; q < n_emb; q++) {
            real_t val = 0.0;
            for (int mu_idx = 0; mu_idx < nao; mu_idx++)
                val += h_C_emb[mu_idx * n_emb + p] * tmp[mu_idx * n_emb + q];
            h_emb[p * n_emb + q] = val;
        }

    // Add chemical potential: h_emb[p,p] -= μ for fragment AOs (first n_frag indices)
    // μ > 0 → fragment more attractive → more electrons
    for (int p = 0; p < n_frag; p++)
        h_emb[p * n_emb + p] -= mu;

    // 2. Diagonalize h_emb → ε, U (standard eigenvalue problem)
    //    The non-orthogonality of C_emb is handled by build_mo_eri's rectangular transform.
    real_t *d_h_emb = nullptr, *d_eigvals = nullptr, *d_eigvecs = nullptr;
    tracked_cudaMalloc(&d_h_emb, n_emb * n_emb * sizeof(real_t));
    tracked_cudaMalloc(&d_eigvals, n_emb * sizeof(real_t));
    tracked_cudaMalloc(&d_eigvecs, n_emb * n_emb * sizeof(real_t));
    cudaMemcpy(d_h_emb, h_emb.data(), n_emb * n_emb * sizeof(real_t), cudaMemcpyHostToDevice);
    gpu::eigenDecomposition(d_h_emb, d_eigvals, d_eigvecs, n_emb);
    tracked_cudaFree(d_h_emb);

    // eigenDecomposition: eigenvector i in COLUMN i → U[p,i] = eigvecs[p*n_emb+i]
    // Level shift: always apply to stabilize CCSD in DMET embedding
    {
        std::vector<real_t> h_eps(n_emb);
        cudaMemcpy(h_eps.data(), d_eigvals, n_emb * sizeof(real_t), cudaMemcpyDeviceToHost);

        real_t homo = h_eps[n_emb_occ - 1];
        real_t lumo = h_eps[n_emb_occ];
        real_t gap = lumo - homo;
        const real_t target_gap = 0.5;  // Hartree — ensures stable CCSD convergence

        if (gap < target_gap) {
            real_t shift = target_gap - gap;
            for (int i = n_emb_occ; i < n_emb; i++)
                h_eps[i] += shift;
            cudaMemcpy(d_eigvals, h_eps.data(), n_emb * sizeof(real_t), cudaMemcpyHostToDevice);
            std::cout << "  HOMO-LUMO gap: " << std::fixed << std::setprecision(4) << gap
                      << " → " << target_gap << " Ha (level shift +" << shift << ")" << std::defaultfloat << std::endl;
        } else {
            std::cout << "  HOMO-LUMO gap: " << std::fixed << std::setprecision(4) << gap << " Ha" << std::defaultfloat << std::endl;
        }
    }

    // 3. C_canonical = C_emb × U  [nao × n_emb]
    std::vector<real_t> h_eigvecs(n_emb * n_emb);
    cudaMemcpy(h_eigvecs.data(), d_eigvecs, n_emb * n_emb * sizeof(real_t), cudaMemcpyDeviceToHost);

    std::vector<real_t> h_C_can(nao * n_emb, 0.0);
    for (int mu_idx = 0; mu_idx < nao; mu_idx++)
        for (int i = 0; i < n_emb; i++) {
            real_t val = 0.0;
            for (int p = 0; p < n_emb; p++)
                val += h_C_emb[mu_idx * n_emb + p] * h_eigvecs[p * n_emb + i];
            h_C_can[mu_idx * n_emb + i] = val;
        }

    real_t* d_C_can = nullptr;
    tracked_cudaMalloc(&d_C_can, nao * n_emb * sizeof(real_t));
    cudaMemcpy(d_C_can, h_C_can.data(), nao * n_emb * sizeof(real_t), cudaMemcpyHostToDevice);

    // 4. MO ERI
    real_t* d_eri_mo = eri_.build_mo_eri(d_C_can, n_emb);

    // 5. CCSD with frozen core (σ ≈ 1 orbitals are frozen)
    real_t E_CCSD = ccsd_spatial_orbital(
        nullptr, d_C_can, d_eigvals,
        n_emb, n_emb_occ,
        false, nullptr, nullptr, nullptr,
        d_eri_mo, n_frozen);

    // Cleanup
    tracked_cudaFree(d_eigvals); tracked_cudaFree(d_eigvecs);
    tracked_cudaFree(d_C_can); tracked_cudaFree(d_eri_mo);

    // N_frag placeholder (Lambda/1-RDM needed for chemical potential optimization)
    real_t N_frag = 2.0 * (n_emb_occ - n_frozen);  // approximate

    return {E_CCSD, N_frag};
}

// ============================================================================
//  Evaluate all fragments at given μ
// ============================================================================

// ============================================================================
//  DMET energy driver
// ============================================================================

real_t DMET::compute_energy() {
    const int nao = num_basis_;
    const int nocc = num_occ_;
    const int N_elec = rhf_.get_num_electrons();

    std::cout << "\n==== DMET-CCSD ====" << std::endl;
    std::cout << "  nao=" << nao << " nocc=" << nocc << " N_elec=" << N_elec
              << " num_atoms=" << num_atoms_ << std::endl;
    std::cout << "  Number of fragments: " << fragments_.size() << std::endl;
    std::cout << "  SVD threshold: " << std::scientific << svd_threshold_ << std::defaultfloat << std::endl;

    for (size_t f = 0; f < fragments_.size(); f++) {
        std::cout << "  Fragment " << f << ": atoms={";
        for (size_t i = 0; i < fragments_[f].atom_indices.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << fragments_[f].atom_indices[i];
        }
        std::cout << "} n_frag_ao=" << fragments_[f].n_frag << std::endl;
    }

    rhf_.get_coefficient_matrix().toHost();
    rhf_.get_fock_matrix().toHost();
    rhf_.get_overlap_matrix().toHost();

    const real_t* h_C = rhf_.get_coefficient_matrix().host_ptr();
    const real_t* h_F = rhf_.get_fock_matrix().host_ptr();
    const real_t* h_S = rhf_.get_overlap_matrix().host_ptr();

    rhf_.get_core_hamiltonian_matrix().toHost();
    const real_t* h_Hcore = rhf_.get_core_hamiltonian_matrix().host_ptr();

    // ================================================================
    //  Phase A: Bath construction (μ-independent)
    // ================================================================

    // S^{1/2} and S^{-1/2} (Löwdin)
    std::vector<real_t> S_half(nao * nao, 0.0), S_inv_half(nao * nao, 0.0);
    {
        std::vector<real_t> s_eigvals(nao), s_eigvecs(nao * nao);
        real_t *d_S = nullptr, *d_sv = nullptr, *d_se = nullptr;
        tracked_cudaMalloc(&d_S, nao * nao * sizeof(real_t));
        tracked_cudaMalloc(&d_sv, nao * sizeof(real_t));
        tracked_cudaMalloc(&d_se, nao * nao * sizeof(real_t));
        cudaMemcpy(d_S, h_S, nao * nao * sizeof(real_t), cudaMemcpyHostToDevice);
        gpu::eigenDecomposition(d_S, d_sv, d_se, nao);
        cudaMemcpy(s_eigvals.data(), d_sv, nao * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(s_eigvecs.data(), d_se, nao * nao * sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_S); tracked_cudaFree(d_sv); tracked_cudaFree(d_se);

        for (int mu = 0; mu < nao; mu++)
            for (int nu = 0; nu <= mu; nu++) {
                real_t vh = 0.0, vi = 0.0;
                for (int i = 0; i < nao; i++) {
                    if (s_eigvals[i] > 1e-10) {
                        real_t u = s_eigvecs[mu * nao + i] * s_eigvecs[nu * nao + i];
                        vh += u * std::sqrt(s_eigvals[i]);
                        vi += u / std::sqrt(s_eigvals[i]);
                    }
                }
                S_half[mu * nao + nu] = vh; S_half[nu * nao + mu] = vh;
                S_inv_half[mu * nao + nu] = vi; S_inv_half[nu * nao + mu] = vi;
            }
    }

    // Bath construction → C_emb per fragment (μ-independent)
    struct BathData {
        std::vector<real_t> C_emb;       // [nao × n_emb]
        std::vector<real_t> h_emb_base;  // C_emb^T F C_emb (before μ shift)
        std::vector<real_t> h_core_emb;  // C_emb^T h_core C_emb (for RDM energy, μ-independent)
        std::vector<real_t> M_co;        // [n_emb × nocc_full]: C_emb^T S C_occ_full,
                                         //  used to build cluster-DM = 2 M M^T for the
                                         //  Vayesta-style separate-subspace canonicalization.
        // ---- Vayesta-convention cached canonical-basis quantities (μ-independent) ----
        // Computed once per unique fragment in Phase A.6 by applying
        // canonicalize_cluster_subspaces(h_emb_base, M_co): D_cluster split for occ/vir
        // partition, then h_emb_base canonicalize_mo within each subspace. Reused at
        // every μ via constant-time formulas (see evaluate_at_mu).
        std::vector<real_t> h_eigvecs_can;  // [n_emb × n_emb] U_can[p,i] (embedding p, canonical i)
        std::vector<real_t> h_eps_can_h;    // [n_emb] diag(U^T h_emb_base U) — semi-canonical ε
        std::vector<real_t> h_F_can;        // [n_emb × n_emb] full U^T h_emb_base U
                                            //   (occ/vir blocks diagonal, occ-vir block ≠ 0)
        std::vector<real_t> h_P_can;        // [n_emb × n_emb] U^T P_frag U; fragment projector in canonical basis
        std::vector<real_t> h_eri_can;      // [n_emb⁴] canonical-basis ERI = build_mo_eri(C_emb · U_can)
        int n_emb, n_emb_occ, n_frozen;
        int n_core;  // σ≈1 orbitals: for N_frag counting only
        bool is_full_system = false;
    };
    std::vector<BathData> baths(fragments_.size());

    for (size_t f = 0; f < fragments_.size(); f++) {
        std::cout << "\n---- Fragment " << f << " (bath) ----" << std::endl;
        auto [C_emb, n_emb_occ, n_frozen, n_core] = build_bath_orbitals(
            fragments_[f], h_C, S_inv_half.data(), S_half.data(), nao, nocc);
        auto& bd = baths[f];
        bd.n_emb_occ = n_emb_occ;
        bd.n_frozen = n_frozen;
        bd.n_core = n_core;

        if (C_emb.empty()) {
            bd.is_full_system = true;
            bd.n_emb = nao;
            std::cout << "  1-fragment = full system: using regular CCSD" << std::endl;
            continue;
        }

        bd.n_emb = (int)(C_emb.size() / nao);
        bd.C_emb = std::move(C_emb);
        int n_bath = bd.n_emb - fragments_[f].n_frag;
        std::cout << "  n_frag=" << fragments_[f].n_frag << " n_bath=" << n_bath
                  << " n_emb=" << bd.n_emb << " n_emb_occ=" << bd.n_emb_occ
                  << " n_frozen=" << bd.n_frozen << std::endl;

        // Precompute h_emb_base = C_emb^T F C_emb and h_core_emb = C_emb^T h_core C_emb
        bd.h_emb_base.resize(bd.n_emb * bd.n_emb, 0.0);
        bd.h_core_emb.resize(bd.n_emb * bd.n_emb, 0.0);
        std::vector<real_t> tmp(nao * bd.n_emb, 0.0);
        std::vector<real_t> tmp2(nao * bd.n_emb, 0.0);
        for (int mu = 0; mu < nao; mu++)
            for (int p = 0; p < bd.n_emb; p++) {
                real_t vf = 0.0, vh = 0.0;
                for (int nu = 0; nu < nao; nu++) {
                    vf += h_F[mu * nao + nu] * bd.C_emb[nu * bd.n_emb + p];
                    vh += h_Hcore[mu * nao + nu] * bd.C_emb[nu * bd.n_emb + p];
                }
                tmp[mu * bd.n_emb + p] = vf;
                tmp2[mu * bd.n_emb + p] = vh;
            }
        for (int p = 0; p < bd.n_emb; p++)
            for (int q = 0; q < bd.n_emb; q++) {
                real_t vf = 0.0, vh = 0.0;
                for (int mu = 0; mu < nao; mu++) {
                    vf += bd.C_emb[mu * bd.n_emb + p] * tmp[mu * bd.n_emb + q];
                    vh += bd.C_emb[mu * bd.n_emb + p] * tmp2[mu * bd.n_emb + q];
                }
                bd.h_emb_base[p * bd.n_emb + q] = vf;
                bd.h_core_emb[p * bd.n_emb + q] = vh;
            }

        // Precompute M_co = C_emb^T · S · C_occ_full  [n_emb × nocc].
        // Used by canonicalize_cluster_subspaces to build cluster-DM = 2 M Mᵀ
        // and identify occupied/virtual subspaces (Vayesta convention).
        bd.M_co.assign((size_t)bd.n_emb * nocc, 0.0);
        std::vector<real_t> SC(nao * nocc, 0.0);
        for (int mu = 0; mu < nao; mu++)
            for (int i = 0; i < nocc; i++) {
                real_t v = 0.0;
                for (int nu = 0; nu < nao; nu++)
                    v += h_S[mu * nao + nu] * h_C[nu * nao + i];  // C_occ = first nocc cols of h_C
                SC[mu * nocc + i] = v;
            }
        for (int p = 0; p < bd.n_emb; p++)
            for (int i = 0; i < nocc; i++) {
                real_t v = 0.0;
                for (int mu = 0; mu < nao; mu++)
                    v += bd.C_emb[mu * bd.n_emb + p] * SC[mu * nocc + i];
                bd.M_co[p * nocc + i] = v;
            }
    }

    cudaDeviceSynchronize();

    // ================================================================
    //  Phase A.5: Multi-GPU setup + fragment equivalence clustering
    //  (μ-independent, computed once)
    // ================================================================

    // Multi-GPU strategy:
    //   Replicated (preferred): replicate full B to every GPU → fragment-parallel
    //                           across OpenMP threads with no peer copies.
    //   Distributed (fallback when B too large to replicate): keep B sliced by
    //                           aux index, each fragment's build_mo_eri uses
    //                           all GPUs collectively (NCCL AllReduce).
    //                           Fragment loop is serial but each ERI build is
    //                           multi-GPU parallel.
    int num_gpus = 1;
    auto* eri_distributed =
        dynamic_cast<ERI_RI_Distributed_RHF*>(const_cast<ERI*>(&eri_));
    if (eri_distributed && eri_distributed->num_gpus() > 1) {
        bool ok = eri_distributed->replicate_B_to_all_gpus();
        if (ok) {
            num_gpus = eri_distributed->num_gpus();
            std::cout << "  Multi-GPU strategy: Replicated (B on every GPU, "
                         "fragment-parallel " << num_gpus << " GPUs)" << std::endl;
        } else {
            std::cout << "  Multi-GPU strategy: Distributed (B sliced, collective "
                         "build_mo_eri across " << eri_distributed->num_gpus()
                      << " GPUs; fragment loop serial)" << std::endl;
        }
    } else {
        std::cout << "  Multi-GPU strategy: single GPU (serial)" << std::endl;
    }

    // Pre-cluster fragments by μ-independent equivalence. Each fragment maps to
    // a canonical representative; only canonical fragments are solved, others
    // copy the result.
    //
    // Equivalence signature: sorted eigenvalues of h_emb_base. These are
    // unitary invariants — robust against the bath SVD's non-unique basis
    // choice when several singular values are nearly degenerate (which makes
    // h_emb_base elements differ between symmetry-equivalent fragments even
    // though the matrices are unitarily equivalent). The earlier norm-only
    // check missed these for pentacene and beyond.
    std::vector<int> canonical_of(fragments_.size(), -1);
    std::vector<int> unique_fragments;  // canonical fragment indices to solve

    using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    std::vector<std::vector<real_t>> h_eigs(fragments_.size());
    for (size_t f = 0; f < fragments_.size(); f++) {
        auto& bd = baths[f];
        if (bd.is_full_system || bd.n_emb == 0) continue;
        const int ne = bd.n_emb;
        Eigen::Map<const RowMatXd> H(bd.h_emb_base.data(), ne, ne);
        Eigen::SelfAdjointEigenSolver<RowMatXd> es(H);
        // SelfAdjointEigenSolver returns eigenvalues sorted ascending.
        h_eigs[f].assign(es.eigenvalues().data(), es.eigenvalues().data() + ne);
    }

    // Element-wise comparison of sorted eigenvalues.
    auto eigs_match = [](const std::vector<real_t>& a,
                         const std::vector<real_t>& b, real_t rel_tol) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            const real_t scale = std::max({std::abs(a[i]), std::abs(b[i]), 1.0});
            if (std::abs(a[i] - b[i]) > rel_tol * scale) return false;
        }
        return true;
    };

    for (size_t f = 0; f < fragments_.size(); f++) {
        auto& bd = baths[f];
        if (bd.is_full_system) {
            canonical_of[f] = (int)f;  // never deduplicated (rare singleton case)
            unique_fragments.push_back((int)f);
            continue;
        }
        if (bd.n_emb_occ <= 0 || bd.n_emb_occ >= bd.n_emb) {
            canonical_of[f] = (int)f;  // trivial — handled in evaluate_at_mu
            continue;  // not added to unique_fragments
        }
        int canon = -1;
        for (int u : unique_fragments) {
            auto& bg = baths[u];
            if (bg.is_full_system) continue;
            if (bg.n_emb != bd.n_emb || bg.n_emb_occ != bd.n_emb_occ
                || bg.n_frozen != bd.n_frozen
                || fragments_[u].n_frag != fragments_[f].n_frag) continue;
            // Tolerance 1e-4: SCF/bath-SVD noise on eigenvalues is ~1e-5
            // relative for medium systems; physically distinct fragment types
            // (e.g. α-CH vs β-CH in polyacenes) differ by >1e-3 in their
            // eigenvalue spectrum — a comfortable separation.
            if (eigs_match(h_eigs[f], h_eigs[u], 1e-4)) {
                canon = u;
                break;
            }
        }
        if (canon < 0) {
            canonical_of[f] = (int)f;
            unique_fragments.push_back((int)f);
        } else {
            canonical_of[f] = canon;
        }
    }
    std::cout << "  Equivalence clustering: " << unique_fragments.size()
              << " unique fragment(s) of " << fragments_.size() << std::endl;

    // ================================================================
    //  Phase A.6: Precompute Vayesta-canonical orbitals + canonical ERI
    //                per unique fragment.
    //
    //  All quantities below are μ-independent because canonicalize_cluster_subspaces
    //  takes h_emb_base directly (no μ shift). At each μ, evaluate_at_mu rebuilds
    //  only the semi-canonical ε and f_ov off-diagonal in O(ne²) time, while the
    //  expensive ne⁴ canonical ERI and the U_can rotation stay cached:
    //
    //    ε(μ)            = diag(F_can) − μ · diag(P_can)        [ne]
    //    f_ov_active(μ) = F_can[occ,vir] − μ · P_can[occ,vir]   [no_act × nv]
    //
    //  The diag part of −μ·P_can in occ-occ and vir-vir blocks is folded into ε
    //  (the off-diag is dropped — semi-canonical CCSD assumes f_oo, f_vv diagonal).
    //
    //  Only canonical-of-equivalence-class fragments are precomputed; equivalent
    //  fragments copy their canonical's evaluate_at_mu results.
    //
    //  Memory footprint per fragment: ne⁴ × 8 B (eri_can) + O(ne²) for the rest.
    //  For cholesterol (n_emb ≤ ~14, 28 unique frags) this is ~10 MB total.
    //
    //  build_mo_eri uses replicated B if Replicated mode succeeded earlier, or
    //  collective NCCL AllReduce if Distributed; either way GPU 0 ends up with
    //  the result, which we copy to host.
    // ================================================================
    cudaSetDevice(0);
    for (int u : unique_fragments) {
        auto& bd = baths[u];
        if (bd.is_full_system) continue;
        const int ne = bd.n_emb;
        const int no = bd.n_emb_occ;
        if (ne == 0) continue;

        // (a) D_cluster split + canonicalize_mo within each subspace using h_emb_base.
        //     Result: U_can[p,i], eps_can_h[i] = (U^T h_emb_base U)[i,i].
        auto [U_can_vec, eps_can_h_vec] = canonicalize_cluster_subspaces(
            bd.h_emb_base, bd.M_co, ne, nocc, no);
        bd.h_eigvecs_can = std::move(U_can_vec);
        bd.h_eps_can_h.assign(eps_can_h_vec.begin(), eps_can_h_vec.end());

        // (b) F_can = U_can^T h_emb_base U_can. After canonicalize_mo:
        //     occ block diagonal (= eps_o), vir block diagonal (= eps_v),
        //     occ-vir block generally non-zero (this is the bare semi-canonical f_ov).
        using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMatXd> Hbase(bd.h_emb_base.data(), ne, ne);
        Eigen::Map<const RowMatXd> Ucan(bd.h_eigvecs_can.data(), ne, ne);
        RowMatXd F_can_mat = Ucan.transpose() * Hbase * Ucan;
        bd.h_F_can.assign(F_can_mat.data(), F_can_mat.data() + (size_t)ne * ne);

        // (c) P_can[i,j] = Σ_{p<n_frag} U_can[p,i] · U_can[p,j]  (fragment projector,
        //     symmetric ne × ne).
        const int n_frag_emb = fragments_[u].n_frag;
        bd.h_P_can.assign((size_t)ne * ne, 0.0);
        for (int i = 0; i < ne; i++)
            for (int j = 0; j <= i; j++) {
                real_t v = 0.0;
                for (int p = 0; p < n_frag_emb; p++)
                    v += bd.h_eigvecs_can[p * ne + i] *
                         bd.h_eigvecs_can[p * ne + j];
                bd.h_P_can[(size_t)i * ne + j] = v;
                bd.h_P_can[(size_t)j * ne + i] = v;
            }

        // (d) C_can_AO = C_emb · U_can, then eri_can = build_mo_eri(C_can_AO).
        std::vector<real_t> h_C_can_AO((size_t)nao * ne, 0.0);
        for (int mu_idx = 0; mu_idx < nao; mu_idx++)
            for (int i = 0; i < ne; i++) {
                real_t v = 0.0;
                for (int p = 0; p < ne; p++)
                    v += bd.C_emb[mu_idx * ne + p] *
                         bd.h_eigvecs_can[p * ne + i];
                h_C_can_AO[mu_idx * ne + i] = v;
            }
        real_t* d_C_can_AO = nullptr;
        tracked_cudaMalloc(&d_C_can_AO, (size_t)nao * ne * sizeof(real_t));
        cudaMemcpy(d_C_can_AO, h_C_can_AO.data(),
                   (size_t)nao * ne * sizeof(real_t), cudaMemcpyHostToDevice);
        real_t* d_eri_can_dev = eri_.build_mo_eri(d_C_can_AO, ne);
        tracked_cudaFree(d_C_can_AO);
        bd.h_eri_can.resize((size_t)ne * ne * ne * ne);
        cudaMemcpy(bd.h_eri_can.data(), d_eri_can_dev,
                   bd.h_eri_can.size() * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_eri_can_dev);
    }

    // ================================================================
    //  Phase B: evaluate_at_mu — solve all fragments at given μ
    //  Returns {Σ N_frag, Σ E_corr_frag}
    // ================================================================

    struct FragResult {
        real_t E_corr_frag;        // T-amplitude democratic (existing)
        real_t E_corr_frag_aoproj; // AO-projected dm2-based (E1+E2 combined)
        real_t E1_aoproj;          // 1e contribution (diagnostic)
        real_t E2_aoproj;          // 2e contribution (diagnostic)
        real_t N_frag;
    };

    // Evaluation modes for evaluate_at_mu:
    //   verbose=true:  full pipeline (CCSD + Lambda + dm1 + dm2 + Vayesta E + print).
    //                  Used for the energy-providing call at μ_final.
    //   verbose=false: CCSD-density mode for μ bisection. Runs CCSD + Lambda + dm1
    //                  to obtain the Λ-relaxed N_frag, but skips dm2/Vayesta for speed.
    //
    // The legacy `need_ccsd_density` flag is retained for API compatibility with
    // the previous 2-stage refinement, but is now a no-op: with Vayesta-canonical
    // orbitals (μ-independent), HF dm1 is also μ-independent, so an HF-fast bisect
    // would never move N_frag. We always need CCSD-relaxed dm1.
    auto evaluate_at_mu = [&](real_t mu, bool verbose,
                              [[maybe_unused]] bool need_ccsd_density = false)
        -> std::tuple<real_t, real_t, real_t> {
        const bool full_pipeline = verbose;
        std::vector<FragResult> results(fragments_.size(), {0.0, 0.0, 0.0, 0.0, 0.0});
        // char (not bool) — std::vector<bool> bit-packs and is not thread-safe
        // for concurrent writes to different elements.
        std::vector<char> solved(fragments_.size(), 0);
        std::vector<char> lambda_ok_v(fragments_.size(), 0);
        // GPU id where each fragment's CCSD/Lambda actually ran (-1 = not solved /
        // equivalent; 0 = full-system / Distributed mode).
        std::vector<int> gpu_used(fragments_.size(), -1);

        // ---- Serial: full-system fragments (rare singleton case) ----
        for (size_t f = 0; f < fragments_.size(); f++) {
            auto& bd = baths[f];
            if (!bd.is_full_system) continue;

            // Fast path for μ-bisection: full-system fragment trivially has
            // N_frag = N_elec independent of μ (no projection effect), so we
            // can skip CCSD entirely in both HF-only and CCSD-density modes.
            // Energies are computed only at the full_pipeline (verbose) call.
            if (!full_pipeline) {
                results[f] = {0.0, 0.0, 0.0, 0.0, (real_t)N_elec};
                solved[f] = true;
                gpu_used[f] = 0;
                continue;
            }

            cudaSetDevice(0);
            real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
            real_t* d_mo_eri;
            real_t E_corr;
            {
                CoutSilencer silence(false);  // verbose=true here, never silence
                d_mo_eri = eri_.build_mo_eri(d_C, nao);
                E_corr = ccsd_spatial_orbital(
                    nullptr, d_C, rhf_.get_orbital_energies().device_ptr(),
                    nao, nocc, false, nullptr, nullptr, nullptr, d_mo_eri, 0);
            }
            tracked_cudaFree(d_mo_eri);
            // Full-system fragment: AO projection is trivial (covers all AOs),
            // so both democratic forms collapse to the same full CCSD energy.
            results[f] = {E_corr, E_corr, 0.0, E_corr, (real_t)N_elec};
            solved[f] = true;
            gpu_used[f] = 0;  // full-system always runs on GPU 0
        }

        // ---- Mark trivial fragments (n_emb_occ degenerate) ----
        for (size_t f = 0; f < fragments_.size(); f++) {
            auto& bd = baths[f];
            if (bd.is_full_system) continue;
            if (bd.n_emb_occ <= 0 || bd.n_emb_occ >= bd.n_emb) {
                results[f] = {0.0, 0.0, 0.0, 0.0, 0.0};
                solved[f] = true;
            }
        }

        // ---- Parallel: solve canonical non-trivial fragments ----
        // Inner output (CCSD/Lambda iter prints) is suppressed throughout; the
        // bisection summary line per fragment is printed in-order afterwards.
        int n_threads = std::min((int)unique_fragments.size(), num_gpus);
        if (n_threads < 1) n_threads = 1;

        // Force OMP team size and disable dynamic adjustment. Eigen serial is
        // set BEFORE the parallel region to avoid its internal OMP setting
        // shrinking our team.
        omp_set_dynamic(0);
        omp_set_num_threads(n_threads);
        Eigen::setNbThreads(1);

        {
            CoutSilencer silence(true);
            #pragma omp parallel num_threads(n_threads)
            {
                #pragma omp for schedule(static, 1)
                for (int idx = 0; idx < (int)unique_fragments.size(); idx++) {
                    int my_tid = omp_get_thread_num();
                    cudaSetDevice(my_tid);

                    int f = unique_fragments[idx];
                    if (solved[f]) continue;  // full-system was handled serially
                    auto& bd = baths[f];

                    const int ne = bd.n_emb, no = bd.n_emb_occ, nv = ne - no;
                    const int no_act = no - bd.n_frozen;
                    const int nf = bd.n_frozen;
                    const int n_frag_emb = fragments_[f].n_frag;
                    const size_t t1sz = (size_t)no_act * nv;
                    const size_t t2sz = (size_t)no_act * no_act * nv * nv;

                    // -----------------------------------------------------------
                    //  Vayesta-convention cluster Hamiltonian.
                    //
                    //  Cluster orbitals U_can are μ-INDEPENDENT (D_cluster split +
                    //  canonicalize_mo on h_emb_base, all cached in Phase A.6).
                    //  Per μ we only rebuild semi-canonical ε and the off-diagonal
                    //  f_ov from constants:
                    //    F_can_full(μ) = F_can − μ · P_can
                    //    ε(μ)         = diag(F_can_full(μ))            (no × no, nv × nv)
                    //    f_ov(μ)      = (F_can_full(μ))[occ, vir]      (no_act × nv)
                    //
                    //  The diag part of −μ·P_can is folded into ε; off-diag in
                    //  occ-occ and vir-vir is dropped (semi-canonical CCSD assumes
                    //  f_oo, f_vv diagonal). The off-diag in occ-vir is kept and
                    //  fed to non-canonical CCSD/Lambda via h_fov_active.
                    // -----------------------------------------------------------
                    const std::vector<real_t>& h_eigvecs = bd.h_eigvecs_can;
                    const std::vector<real_t>& F_can     = bd.h_F_can;
                    const std::vector<real_t>& P_can     = bd.h_P_can;
                    const std::vector<real_t>& eps_h     = bd.h_eps_can_h;
                    std::vector<real_t>        h_eri     = bd.h_eri_can;  // cached

                    // (a) Semi-canonical ε with diag μ shift
                    std::vector<real_t> h_eps(ne, 0.0);
                    for (int i = 0; i < ne; i++)
                        h_eps[i] = eps_h[i] - mu * P_can[(size_t)i * ne + i];

                    // (b) f_ov_active[i, a] = F_can[i+nf, no+a] − μ · P_can[i+nf, no+a]
                    //     for i ∈ [0, no_act), a ∈ [0, nv).
                    std::vector<real_t> h_fov_active(t1sz, 0.0);
                    real_t fov_max = 0.0;
                    for (int i = 0; i < no_act; i++) {
                        const size_t row = (size_t)(i + nf);
                        for (int a = 0; a < nv; a++) {
                            const size_t col = (size_t)(no + a);
                            const real_t F_ia = F_can[row * ne + col];
                            const real_t P_ia = P_can[row * ne + col];
                            const real_t v   = F_ia - mu * P_ia;
                            h_fov_active[(size_t)i * nv + a] = v;
                            fov_max = std::max(fov_max, std::abs(v));
                        }
                    }

                    // Diagnostic: maximum |f_ov| for this fragment (μ-shifted). Use stderr
                    // since stdout is silenced. Helps gauge whether semi-canonical CCSD
                    // can converge — empirically PySCF/Vayesta handle |f_ov| ≲ 0.5 Ha.
                    std::fprintf(stderr,
                        "[DMET frag %d μ=%.4f] max|f_ov|=%.4e  ε_HOMO=%.4e ε_LUMO=%.4e\n",
                        f, mu, fov_max,
                        (no > 0 ? h_eps[no - 1] : 0.0),
                        (no < ne ? h_eps[no] : 0.0));

                    // Detailed diagnostic: full ε spectrum + D_cluster eigenvalues.
                    // Helps identify whether canonicalize_cluster_subspaces is putting
                    // the wrong directions in occ/vir (e.g., core leaking into vir).
                    if (std::getenv("GANSU_DMET_VERBOSE") != nullptr) {
                        std::fprintf(stderr, "[DMET frag %d] ε_occ:", f);
                        for (int i = 0; i < no; i++)
                            std::fprintf(stderr, " %.4e", h_eps[i]);
                        std::fprintf(stderr, "\n[DMET frag %d] ε_vir:", f);
                        for (int i = no; i < ne; i++)
                            std::fprintf(stderr, " %.4e", h_eps[i]);
                        std::fprintf(stderr, "\n[DMET frag %d] D_cluster eigenvalues (asc):", f);
                        // Recompute D_cluster eigs for diagnosis
                        using RowMatXd2 = Eigen::Matrix<real_t, Eigen::Dynamic,
                            Eigen::Dynamic, Eigen::RowMajor>;
                        Eigen::Map<const RowMatXd2> Mco(bd.M_co.data(), ne, nocc);
                        RowMatXd2 Dcl = 2.0 * Mco * Mco.transpose();
                        Eigen::SelfAdjointEigenSolver<RowMatXd2> dsolv(Dcl);
                        Eigen::VectorXd dvals = dsolv.eigenvalues();
                        for (int i = 0; i < ne; i++)
                            std::fprintf(stderr, " %.4e", dvals(i));
                        std::fprintf(stderr, "\n");
                    }

                    // Escape hatch: GANSU_DMET_NOFOV=1 disables the semi-canonical
                    // f_ov contribution (h_fov_active = nullptr → CCSD treats orbitals
                    // as canonical, ignoring Brillouin off-diagonal). The orbitals,
                    // ε, and ERI are still from canonicalize_cluster_subspaces, so this
                    // matches Approach 4's physics with the new bath canonicalization
                    // — useful to isolate whether divergence is in the f_ov branches.
                    static const bool nofov_env =
                        (std::getenv("GANSU_DMET_NOFOV") != nullptr);
                    const real_t* fov_ptr = nofov_env ? nullptr : h_fov_active.data();

                    // Level shift on virtual ε (small clusters with tiny gap).
                    // Affects only CCSD denominators, not the physical Hamiltonian.
                    if (no > 0 && no < ne) {
                        real_t gap = h_eps[no] - h_eps[no - 1];
                        const real_t target_gap = 0.1;
                        if (gap < target_gap) {
                            real_t shift = target_gap - gap;
                            for (int i = no; i < ne; i++) h_eps[i] += shift;
                        }
                    }

                    real_t* d_eigvals = nullptr;
                    tracked_cudaMalloc(&d_eigvals, ne * sizeof(real_t));
                    cudaMemcpy(d_eigvals, h_eps.data(), ne * sizeof(real_t),
                               cudaMemcpyHostToDevice);

                    // Canonical-basis ERI is μ-independent: upload the cached host copy.
                    real_t* d_eri_mo = nullptr;
                    tracked_cudaMalloc(&d_eri_mo, h_eri.size() * sizeof(real_t));
                    cudaMemcpy(d_eri_mo, h_eri.data(),
                               h_eri.size() * sizeof(real_t), cudaMemcpyHostToDevice);

                    // CCSD with non-canonical (semi-canonical) f_ov.
                    // d_C_emb is unused when d_eri_mo_precomputed is supplied and
                    // CCSD(T) is off; pass nullptr.
                    real_t *d_t1 = nullptr, *d_t2 = nullptr;
                    real_t E_CCSD = ccsd_spatial_orbital(
                        nullptr, nullptr, d_eigvals,
                        ne, no, false, nullptr, &d_t1, &d_t2,
                        d_eri_mo, bd.n_frozen,
                        fov_ptr);
                    (void)E_CCSD;

                    // Download T-amplitudes (h_eri already on host).
                    std::vector<real_t> h_t1(t1sz), h_t2(t2sz);
                    cudaMemcpy(h_t1.data(), d_t1, t1sz*sizeof(real_t),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_t2.data(), d_t2, t2sz*sizeof(real_t),
                               cudaMemcpyDeviceToHost);

                    // Lambda solver with the same f_ov for self-consistency.
                    std::vector<real_t> h_l1(h_t1), h_l2(h_t2);
                    bool lambda_ok = solve_ccsd_lambda_cpu(
                        no, nv, h_eps.data(), h_eri.data(),
                        h_t1.data(), h_t2.data(), h_l1.data(), h_l2.data(),
                        300, 1e-5, /*verbose=*/0,
                        fov_ptr);

                    // Democratic E_corr_frag (T-amplitude form). Picks up an extra
                    //   2 Σ_{i,i',a} P[i,i'] f_ov[i,a] t1[i',a]
                    // term in the semi-canonical case (Brillouin contribution).
                    // Only computed in full_pipeline mode.
                    real_t E_frag = 0.0;
                    if (full_pipeline) {
                        E_frag = compute_dmet_fragment_energy(
                            no, nv, h_eri.data(), h_t1.data(), h_t2.data(),
                            n_frag_emb, h_eigvecs.data(), bd.n_frozen,
                            fov_ptr);
                    }

                    // N_frag from relaxed 1-RDM (always computed). The Vayesta
                    // (dm2-based) fragment energy is full_pipeline-only.
                    //
                    // Note: bd.n_core is now folded into bd.n_emb_occ (Schmidt
                    // convention — see build_bath_orbitals), so the σ ≈ 1
                    // fragment direction is already part of the canonical occ
                    // block and contributes to N_frag via P_can[core, core] · 2
                    // in the sums below. No separate `+ 2 · n_core` term needed.
                    real_t N_frag = 0.0;
                    real_t E_frag_aoproj = 0.0;
                    real_t E1_ao = 0.0, E2_ao = 0.0;
                    if (lambda_ok) {
                        const int na_act = no_act + nv;
                        const size_t na_act4 =
                            (size_t)na_act * na_act * na_act * na_act;
                        std::vector<real_t> dm1(na_act * na_act, 0.0);
                        build_ccsd_1rdm_mo_cpu(no_act, nv, h_t1.data(), h_t2.data(),
                                               h_l1.data(), h_l2.data(), dm1.data());
                        for (int i = 0; i < na_act; i++)
                            for (int j = 0; j < na_act; j++) {
                                real_t P_ij = 0.0;
                                for (int p = 0; p < n_frag_emb; p++)
                                    P_ij += h_eigvecs[p * ne + (i+nf)]
                                          * h_eigvecs[p * ne + (j+nf)];
                                N_frag += P_ij * dm1[i * na_act + j];
                            }

                        if (full_pipeline) {
                            // 10b. Standard QC-DMET fragment energy (Vayesta convention).
                            // Uses h_avg = ½(h_bare + h_eff) where h_eff = F − v_act,
                            // so cluster-internal Hartree-exchange cancels and we
                            // recover the bare core operator on average. Validated
                            // against Vayesta on benzene/STO-3G (−0.431 Ha vs the
                            // old AO-proj formula's −0.067 Ha).
                            std::vector<real_t> dm2(na_act4, 0.0);
                            build_ccsd_2rdm_chemist_cpu(
                                no_act, nv, h_t1.data(), h_t2.data(),
                                h_l1.data(), h_l2.data(), dm1.data(), dm2.data());
                            E_frag_aoproj = compute_dmet_fragment_energy_vayesta(
                                no_act, nv,
                                bd.h_core_emb.data(), bd.h_emb_base.data(),
                                h_eri.data(),
                                dm1.data(), dm2.data(),
                                n_frag_emb, h_eigvecs.data(), bd.n_frozen,
                                &E1_ao, &E2_ao);
                        }
                    } else {
                        for (int i = 0; i < no; i++) {
                            real_t P_ii = 0.0;
                            for (int p = 0; p < n_frag_emb; p++)
                                P_ii += h_eigvecs[p * ne + i] * h_eigvecs[p * ne + i];
                            N_frag += 2.0 * P_ii;
                        }
                        // Without Lambda we can't form relaxed 2-RDM; fall back
                        // to the T-amplitude value so totals stay finite.
                        E_frag_aoproj = E_frag;
                    }

                    results[f] = {E_frag, E_frag_aoproj, E1_ao, E2_ao, N_frag};
                    lambda_ok_v[f] = lambda_ok;
                    gpu_used[f] = my_tid;
                    solved[f] = true;

                    tracked_cudaFree(d_t1); tracked_cudaFree(d_t2);
                    tracked_cudaFree(d_eri_mo);
                    tracked_cudaFree(d_eigvals);
                }
            } // end omp parallel
        } // end CoutSilencer

        // ---- Copy canonical results to equivalent fragments ----
        for (size_t f = 0; f < fragments_.size(); f++) {
            if (solved[f]) continue;
            int canon = canonical_of[f];
            if (canon >= 0 && solved[canon]) {
                results[f] = results[canon];
                lambda_ok_v[f] = lambda_ok_v[canon];
                solved[f] = true;
            }
        }

        // ---- Verbose summary (in fragment order) ----
        if (verbose) {
            for (size_t f = 0; f < fragments_.size(); f++) {
                auto& bd = baths[f];
                int canon = canonical_of[f];
                if (bd.is_full_system) {
                    std::cout << "  Fragment " << f << ": full system CCSD = "
                              << std::setprecision(10) << results[f].E_corr_frag
                              << " Ha [GPU 0]" << std::endl;
                } else if (canon == (int)f) {
                    if (gpu_used[f] < 0) {
                        std::cout << "  Fragment " << f << ": (trivial, skipped)"
                                  << std::endl;
                    } else {
                        std::cout << "  Fragment " << f
                                  << ": E_corr(T-amp)=" << std::setprecision(10)
                                  << results[f].E_corr_frag
                                  << " E_corr(DMET)=" << std::setprecision(10)
                                  << results[f].E_corr_frag_aoproj
                                  << " (E1=" << std::setprecision(6) << results[f].E1_aoproj
                                  << " E2=" << std::setprecision(6) << results[f].E2_aoproj
                                  << ")"
                                  << " N_frag=" << std::setprecision(4) << results[f].N_frag
                                  << (lambda_ok_v[f] ? " (Lambda OK)" : " (L=T)")
                                  << " [GPU " << gpu_used[f] << "]" << std::endl;
                    }
                } else if (canon >= 0) {
                    std::cout << "  Fragment " << f << ": equivalent to " << canon
                              << " (E_T=" << std::setprecision(10) << results[f].E_corr_frag
                              << ", E_DMET=" << std::setprecision(10)
                              << results[f].E_corr_frag_aoproj
                              << ", N=" << std::setprecision(4) << results[f].N_frag
                              << ") [from GPU " << gpu_used[canon] << "]" << std::endl;
                }
            }
        }

        // ---- Aggregate ----
        real_t N_total = 0.0, E_total = 0.0, E_total_aoproj = 0.0;
        for (size_t f = 0; f < fragments_.size(); f++) {
            N_total += results[f].N_frag;
            E_total += results[f].E_corr_frag;
            E_total_aoproj += results[f].E_corr_frag_aoproj;
        }
        return {N_total, E_total, E_total_aoproj};
    };

    // ================================================================
    //  Phase C: Chemical potential optimization (bisection)
    // ================================================================

    std::cout << "\n---- μ optimization ----" << std::endl;

    // First evaluation at μ=0
    auto [N0, E0, E0_aoproj] = evaluate_at_mu(0.0, true);
    real_t N_err = N0 - N_elec;

    std::cout << "  μ=0: Σ N_frag=" << std::setprecision(4) << N0
              << " N_elec=" << N_elec << " err=" << N_err << std::endl;

    real_t mu_opt = 0.0;
    real_t E_corr_opt = E0;
    real_t E_corr_opt_aoproj = E0_aoproj;
    // Set when a Final-eval at μ ≠ 0 has been done; controls Summary label.
    bool refinement_applied = false;

    // Vayesta-convention DMET-CCSD bisection: find μ such that the CCSD-relaxed
    //   Σ_F N_F^cluster(μ) = N_total
    //
    // The cluster orbitals are μ-INDEPENDENT (cached canonicalize_cluster_subspaces),
    // so HF dm1 in canonical basis is also μ-independent and an HF-fast bisection
    // would never move N_frag. Each μ evaluation therefore runs full CCSD/Lambda/dm1
    // (dm2 and the Vayesta formula are only computed in the final verbose call).
    //
    // The need_ccsd_density flag is now redundant — kept in the API for backward
    // compat with the old 2-stage pipeline. Stage-2 refinement (the dmet_mu_refine_ccsd
    // parameter) is now a no-op since Stage-1 already runs CCSD-density.
    const real_t N_tol = 1e-5;
    if (std::abs(N_err) > N_tol) {
        // Bracket: μ > 0 pulls electrons into the fragment, μ < 0 pushes them out.
        real_t mu_lo = -1.0, mu_hi = 1.0;

        std::cout << "  Bracket init: solving at μ=" << mu_lo << " and μ=" << mu_hi << std::endl;
        // Bracket setup uses only N — energies recomputed at μ_opt below.
        real_t N_lo = std::get<0>(evaluate_at_mu(mu_lo, false));
        std::cout << "    μ=" << std::setprecision(4) << mu_lo
                  << "  Σ N_frag=" << N_lo << "  err=" << (N_lo - N_elec) << std::endl;
        real_t N_hi = std::get<0>(evaluate_at_mu(mu_hi, false));
        std::cout << "    μ=" << mu_hi
                  << "  Σ N_frag=" << N_hi << "  err=" << (N_hi - N_elec) << std::endl;

        for (int expand = 0; expand < 5; expand++) {
            if ((N_lo - N_elec) * (N_hi - N_elec) <= 0) break;
            mu_lo *= 2.0; mu_hi *= 2.0;
            std::cout << "  Expanding bracket to μ=[" << mu_lo << ", " << mu_hi << "]" << std::endl;
            N_lo = std::get<0>(evaluate_at_mu(mu_lo, false));
            std::cout << "    μ=" << mu_lo
                      << "  Σ N_frag=" << N_lo << "  err=" << (N_lo - N_elec) << std::endl;
            N_hi = std::get<0>(evaluate_at_mu(mu_hi, false));
            std::cout << "    μ=" << mu_hi
                      << "  Σ N_frag=" << N_hi << "  err=" << (N_hi - N_elec) << std::endl;
        }

        if ((N_lo - N_elec) * (N_hi - N_elec) > 0) {
            std::cout << "  WARNING: bisection bracket failed, using μ=0" << std::endl;
        } else {
            const int max_iter = 30;
            std::cout << "  Bisecting (target |err| < " << std::scientific << N_tol
                      << std::defaultfloat << ")..." << std::endl;
            for (int iter = 0; iter < max_iter; iter++) {
                real_t mu_mid = 0.5 * (mu_lo + mu_hi);
                auto [N_mid, E_mid, E_mid_ao] = evaluate_at_mu(mu_mid, false);
                real_t err_mid = N_mid - N_elec;
                (void)E_mid; (void)E_mid_ao;  // bisection uses only N_mid

                std::cout << "    iter " << std::setw(2) << iter + 1
                          << ": μ=" << std::setprecision(6) << mu_mid
                          << "  Σ N_frag=" << std::setprecision(6) << N_mid
                          << "  err=" << std::scientific << err_mid
                          << std::defaultfloat << std::endl;

                if (std::abs(err_mid) < N_tol || (mu_hi - mu_lo) < 1e-10) {
                    mu_opt = mu_mid;
                    std::cout << "  μ converged at iter " << iter + 1 << std::endl;
                    break;
                }
                if ((N_lo - N_elec) * err_mid < 0) {
                    mu_hi = mu_mid; N_hi = N_mid;
                } else {
                    mu_lo = mu_mid; N_lo = N_mid;
                }
                if (iter == max_iter - 1) {
                    mu_opt = mu_mid;
                    std::cout << "  μ: max iterations reached, μ=" << std::setprecision(6) << mu_opt
                              << " err=" << err_mid << std::endl;
                }
            }
        }

        // Final full eval at μ_opt to produce energies (T-amp + Vayesta both).
        // The semi-canonical f_ov = F_can_ov − μ·P_can_ov absorbs the μ shift
        // consistently, so the T-amp formula now yields the correct CCSD
        // correlation energy at any μ (no need to fall back to μ=0).
        if (std::abs(mu_opt) > 1e-12) {
            std::cout << "  Final evaluation at μ_DMET = "
                      << std::setprecision(6) << mu_opt << std::endl;
            real_t N_final = 0.0, E_final = 0.0, E_final_ao = 0.0;
            std::tie(N_final, E_final, E_final_ao) = evaluate_at_mu(mu_opt, true);
            (void)N_final;
            E_corr_opt = E_final;
            E_corr_opt_aoproj = E_final_ao;
            refinement_applied = true;
        } else {
            // μ_opt = 0 — the initial verbose eval already produced the
            // correct energies in (E0, E0_aoproj).
            E_corr_opt = E0;
            E_corr_opt_aoproj = E0_aoproj;
        }
    } else {
        std::cout << "  μ=0 satisfies electron count (err=" << std::scientific << N_err
                  << std::defaultfloat << "), no optimization needed" << std::endl;
    }

    // ================================================================
    //  Summary
    // ================================================================

    std::cout << "\n---- DMET-CCSD Summary ----" << std::endl;
    if (refinement_applied) {
        std::cout << "  Chemical potential μ_DMET (CCSD-relaxed): "
                  << std::setprecision(6) << mu_opt << " Ha" << std::endl;
    } else {
        std::cout << "  Chemical potential μ: "
                  << std::setprecision(6) << mu_opt << " Ha" << std::endl;
    }
    std::cout << "  Total DMET-CCSD correlation energy (T-amp democratic): "
              << std::fixed << std::setprecision(10) << E_corr_opt << " Ha" << std::endl;
    std::cout << "  Total DMET-CCSD correlation energy (DMET, Vayesta):     "
              << std::setprecision(10) << E_corr_opt_aoproj << " Ha" << std::endl;
    std::cout << "  HF energy: " << std::setprecision(10) << rhf_.get_total_energy() << " Ha" << std::endl;
    std::cout << "  DMET-CCSD total energy (T-amp): "
              << std::setprecision(10) << rhf_.get_total_energy() + E_corr_opt << " Ha" << std::endl;
    std::cout << "  DMET-CCSD total energy (DMET):  "
              << std::setprecision(10) << rhf_.get_total_energy() + E_corr_opt_aoproj << " Ha" << std::endl;
    std::cout << std::defaultfloat;

    // Release replicated full-B copies (frees ~num_gpus × naux × nao² × 8 bytes)
    if (eri_distributed && eri_distributed->b_is_replicated()) {
        eri_distributed->free_replicated_B();
    }

    return E_corr_opt;
}

// ============================================================================
//  ERI wiring
// ============================================================================

real_t ERI_Stored_RHF::compute_dmet_ccsd() {
    DMET dmet(rhf_, *this);
    return dmet.compute_energy();
}

real_t ERI_RI_RHF::compute_dmet_ccsd() {
    DMET dmet(rhf_, *this);
    return dmet.compute_energy();
}

} // namespace gansu
