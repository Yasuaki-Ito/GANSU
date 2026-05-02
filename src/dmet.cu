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
#include <algorithm>
#include <tuple>
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
                            int num_frozen);

// RAII guard: redirect std::cout to a discarded sink for the lifetime of the
// object. Used to silence ccsd_spatial_orbital's per-iteration output during
// the bisection inner loop, where its many lines would drown out the bisection
// progress messages.
class CoutSilencer {
    std::streambuf* old_buf_ = nullptr;
    std::stringstream sink_;
public:
    explicit CoutSilencer(bool silence) {
        if (silence) old_buf_ = std::cout.rdbuf(sink_.rdbuf());
    }
    ~CoutSilencer() {
        if (old_buf_) std::cout.rdbuf(old_buf_);
    }
};

// ============================================================================
//  Embedding HF SCF (CPU, Roothaan iteration)
//
//  Solves canonical orbitals of the embedding Hamiltonian H = h_emb + V_emb,
//  i.e. eigenstates of the *self-consistent* Fock matrix
//      F[p,q] = h_emb[p,q] + Σ_rs D[r,s] (eri[p,q,r,s] - 0.5 eri[p,r,s,q])
//  with D[r,s] = 2 Σ_{i<n_occ} C[r,i] C[s,i].
//
//  Without this step, h_emb eigenstates leave F off-diagonal in the embedding
//  basis (e.g. ~10⁻¹ for benzene/STO-3G), violating the canonical-orbital
//  assumption made by ccsd_spatial_orbital and solve_ccsd_lambda_cpu.
//
//  Output: U[p, i] = embedding-basis coefficient of i-th canonical orbital,
//          eps[i]  = orbital energy (eigenvalue of converged F).
//
//  Tolerance is on the energy change between iterations; for ne ≲ 30 this
//  runs in microseconds even without DIIS.
// ============================================================================
static std::pair<std::vector<real_t>, std::vector<real_t>>
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
    bool converged = false;

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
//  Fragment parsing (unchanged)
// ============================================================================

std::vector<DMETFragment> DMET::parse_fragments(const std::string& spec, int num_atoms) {
    std::vector<DMETFragment> fragments;
    if (spec.empty()) {
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
//  Constructor
// ============================================================================

DMET::DMET(RHF& rhf, const ERI& eri)
    : rhf_(rhf), eri_(eri),
      num_basis_(rhf.get_num_basis()),
      num_occ_(rhf.get_num_electrons() / 2),
      num_atoms_((int)rhf.get_atom_to_basis_range().size()),
      svd_threshold_(rhf.get_dmet_threshold())
{
    fragments_ = parse_fragments(rhf.get_dmet_fragments_str(), num_atoms_);
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

    // --- Count bath (threshold < σ < 1-threshold) and core (σ ≥ 1-threshold) ---
    int n_bath = 0, n_core = 0;
    for (int i = 0; i < k; i++) {
        if (sigma[i] > svd_threshold_ && sigma[i] < 1.0 - svd_threshold_)
            n_bath++;
        else if (sigma[i] >= 1.0 - svd_threshold_)
            n_core++;
    }

    int n_emb = n_frag + n_bath;
    int n_emb_occ = n_bath;
    int n_frozen = 0;  // don't freeze in CCSD — all occupied orbitals correlated

    // --- Build C_emb in Löwdin basis, then convert to AO ---
    // Fragment: identity on Löwdin AO indices
    // Bath: C_lo_occ × V (in Löwdin basis)
    // Convert: C_emb_ao = S^{-1/2} × C_emb_lo
    // Result: C_emb_ao^T S C_emb_ao = C_emb_lo^T (S^{-1/2} S S^{-1/2}) C_emb_lo = C_emb_lo^T C_emb_lo = I

    std::vector<real_t> C_emb_lo(nao * n_emb, 0.0);
    for (int i = 0; i < n_frag; i++) {
        int mu = frag.ao_indices[i];
        C_emb_lo[mu * n_emb + i] = 1.0;
    }
    int bath_col = n_frag;
    for (int i = 0; i < k; i++) {
        if (sigma[i] > svd_threshold_ && sigma[i] < 1.0 - svd_threshold_) {
            for (int mu = 0; mu < nao; mu++) {
                real_t val = 0.0;
                for (int j = 0; j < nocc; j++)
                    val += C_lo_occ[mu * nocc + j] * Vt[i * nocc + j];
                C_emb_lo[mu * n_emb + bath_col] = val;
            }
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

    // Pre-cluster fragments by μ-independent equivalence (h_emb_base norm,
    // n_emb, n_emb_occ, etc). Each fragment maps to a canonical representative;
    // only canonical fragments are solved, others copy the result.
    std::vector<int> canonical_of(fragments_.size(), -1);
    std::vector<int> unique_fragments;  // canonical fragment indices to solve
    std::vector<real_t> h_norms(fragments_.size(), 0.0);
    for (size_t f = 0; f < fragments_.size(); f++) {
        for (auto v : baths[f].h_emb_base) h_norms[f] += v * v;
    }
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
            if (std::abs(h_norms[f] - h_norms[u])
                < 1e-10 * std::max(h_norms[f], h_norms[u])) {
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
    //  Phase B: evaluate_at_mu — solve all fragments at given μ
    //  Returns {Σ N_frag, Σ E_corr_frag}
    // ================================================================

    struct FragResult {
        real_t E_corr_frag;
        real_t N_frag;
    };

    auto evaluate_at_mu = [&](real_t mu, bool verbose) -> std::pair<real_t, real_t> {
        std::vector<FragResult> results(fragments_.size(), {0.0, 0.0});
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

            cudaSetDevice(0);
            real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
            real_t* d_mo_eri;
            real_t E_corr;
            {
                CoutSilencer silence(!verbose);
                d_mo_eri = eri_.build_mo_eri(d_C, nao);
                E_corr = ccsd_spatial_orbital(
                    nullptr, d_C, rhf_.get_orbital_energies().device_ptr(),
                    nao, nocc, false, nullptr, nullptr, nullptr, d_mo_eri, 0);
            }
            tracked_cudaFree(d_mo_eri);
            results[f] = {E_corr, (real_t)N_elec};
            solved[f] = true;
            gpu_used[f] = 0;  // full-system always runs on GPU 0
        }

        // ---- Mark trivial fragments (n_emb_occ degenerate) ----
        for (size_t f = 0; f < fragments_.size(); f++) {
            auto& bd = baths[f];
            if (bd.is_full_system) continue;
            if (bd.n_emb_occ <= 0 || bd.n_emb_occ >= bd.n_emb) {
                results[f] = {0.0, 0.0};
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
                    const int n_frag_emb = fragments_[f].n_frag;
                    const size_t t1sz = (size_t)no_act * nv;
                    const size_t t2sz = (size_t)no_act * no_act * nv * nv;

                    // 1. h_emb(μ) = h_emb_base - μ P_frag
                    std::vector<real_t> h_emb(bd.h_emb_base);
                    for (int p = 0; p < n_frag_emb; p++)
                        h_emb[p * ne + p] -= mu;

                    // 2. Build eri in embedding basis (current device's replicated B)
                    real_t* d_C_emb_dev = nullptr;
                    tracked_cudaMalloc(&d_C_emb_dev, nao * ne * sizeof(real_t));
                    cudaMemcpy(d_C_emb_dev, bd.C_emb.data(), nao * ne * sizeof(real_t),
                               cudaMemcpyHostToDevice);
                    real_t* d_eri_emb_dev = eri_.build_mo_eri(d_C_emb_dev, ne);
                    tracked_cudaFree(d_C_emb_dev);

                    std::vector<real_t> h_eri_emb((size_t)ne*ne*ne*ne);
                    cudaMemcpy(h_eri_emb.data(), d_eri_emb_dev,
                               h_eri_emb.size()*sizeof(real_t), cudaMemcpyDeviceToHost);
                    tracked_cudaFree(d_eri_emb_dev);

                    // 3. Embedding HF
                    auto [h_eigvecs, h_eps] = run_embedding_hf_cpu(
                        h_emb, h_eri_emb, ne, no,
                        /*max_iter=*/100, /*tol=*/1e-10, /*verbose=*/0);

                    // Level shift
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

                    // 4. C_can = C_emb × U_HF
                    std::vector<real_t> h_C_can(nao * ne, 0.0);
                    for (int mu_idx = 0; mu_idx < nao; mu_idx++)
                        for (int i = 0; i < ne; i++) {
                            real_t val = 0.0;
                            for (int p = 0; p < ne; p++)
                                val += bd.C_emb[mu_idx * ne + p] * h_eigvecs[p * ne + i];
                            h_C_can[mu_idx * ne + i] = val;
                        }

                    // 5. MO ERI in canonical basis
                    real_t* d_C_can = nullptr;
                    tracked_cudaMalloc(&d_C_can, nao * ne * sizeof(real_t));
                    cudaMemcpy(d_C_can, h_C_can.data(), nao * ne * sizeof(real_t),
                               cudaMemcpyHostToDevice);
                    real_t* d_eri_mo = eri_.build_mo_eri(d_C_can, ne);

                    // 6. CCSD
                    real_t *d_t1 = nullptr, *d_t2 = nullptr;
                    real_t E_CCSD = ccsd_spatial_orbital(
                        nullptr, d_C_can, d_eigvals,
                        ne, no, false, nullptr, &d_t1, &d_t2,
                        d_eri_mo, bd.n_frozen);
                    (void)E_CCSD;

                    // 7. Download T/ERI
                    std::vector<real_t> h_t1(t1sz), h_t2(t2sz);
                    std::vector<real_t> h_eri((size_t)ne*ne*ne*ne);
                    cudaMemcpy(h_t1.data(), d_t1, t1sz*sizeof(real_t),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_t2.data(), d_t2, t2sz*sizeof(real_t),
                               cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_eri.data(), d_eri_mo,
                               h_eri.size()*sizeof(real_t), cudaMemcpyDeviceToHost);

                    // 8. Lambda solver
                    std::vector<real_t> h_l1(h_t1), h_l2(h_t2);
                    bool lambda_ok = solve_ccsd_lambda_cpu(
                        no, nv, h_eps.data(), h_eri.data(),
                        h_t1.data(), h_t2.data(), h_l1.data(), h_l2.data(),
                        300, 1e-5, /*verbose=*/0);

                    // 9. Democratic E_corr_frag
                    real_t E_frag = compute_dmet_fragment_energy(
                        no, nv, h_eri.data(), h_t1.data(), h_t2.data(),
                        n_frag_emb, h_eigvecs.data(), bd.n_frozen);

                    // 10. N_frag from relaxed 1-RDM (or HF fallback)
                    real_t N_frag = 2.0 * bd.n_core;
                    if (lambda_ok) {
                        const int na_act = no_act + nv;
                        std::vector<real_t> dm1(na_act * na_act, 0.0);
                        build_ccsd_1rdm_mo_cpu(no_act, nv, h_t1.data(), h_t2.data(),
                                               h_l1.data(), h_l2.data(), dm1.data());
                        const int nf = bd.n_frozen;
                        for (int i = 0; i < na_act; i++)
                            for (int j = 0; j < na_act; j++) {
                                real_t P_ij = 0.0;
                                for (int p = 0; p < n_frag_emb; p++)
                                    P_ij += h_eigvecs[p * ne + (i+nf)]
                                          * h_eigvecs[p * ne + (j+nf)];
                                N_frag += P_ij * dm1[i * na_act + j];
                            }
                    } else {
                        for (int i = 0; i < no; i++) {
                            real_t P_ii = 0.0;
                            for (int p = 0; p < n_frag_emb; p++)
                                P_ii += h_eigvecs[p * ne + i] * h_eigvecs[p * ne + i];
                            N_frag += 2.0 * P_ii;
                        }
                    }

                    results[f] = {E_frag, N_frag};
                    lambda_ok_v[f] = lambda_ok;
                    gpu_used[f] = my_tid;
                    solved[f] = true;

                    tracked_cudaFree(d_t1); tracked_cudaFree(d_t2);
                    tracked_cudaFree(d_C_can); tracked_cudaFree(d_eri_mo);
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
                        std::cout << "  Fragment " << f << ": E_corr="
                                  << std::setprecision(10) << results[f].E_corr_frag
                                  << " N_frag=" << std::setprecision(4) << results[f].N_frag
                                  << (lambda_ok_v[f] ? " (Lambda OK)" : " (L=T)")
                                  << " [GPU " << gpu_used[f] << "]" << std::endl;
                    }
                } else if (canon >= 0) {
                    std::cout << "  Fragment " << f << ": equivalent to " << canon
                              << " (E=" << std::setprecision(10) << results[f].E_corr_frag
                              << ", N=" << std::setprecision(4) << results[f].N_frag
                              << ") [from GPU " << gpu_used[canon] << "]" << std::endl;
                }
            }
        }

        // ---- Aggregate ----
        real_t N_total = 0.0, E_total = 0.0;
        for (size_t f = 0; f < fragments_.size(); f++) {
            N_total += results[f].N_frag;
            E_total += results[f].E_corr_frag;
        }
        return {N_total, E_total};
    };

    // ================================================================
    //  Phase C: Chemical potential optimization (bisection)
    // ================================================================

    std::cout << "\n---- μ optimization ----" << std::endl;

    // First evaluation at μ=0
    auto [N0, E0] = evaluate_at_mu(0.0, true);
    real_t N_err = N0 - N_elec;

    std::cout << "  μ=0: Σ N_frag=" << std::setprecision(4) << N0
              << " N_elec=" << N_elec << " err=" << N_err << std::endl;

    real_t mu_opt = 0.0;
    real_t E_corr_opt = E0;

    const real_t N_tol = 1e-5;
    if (std::abs(N_err) > N_tol) {
        // Bisection: find μ such that Σ N_frag(μ) = N_elec
        // μ > 0 → more electrons in fragment, μ < 0 → fewer
        real_t mu_lo = -1.0, mu_hi = 1.0;

        // Ensure the bracket contains the root
        std::cout << "  Bracket init: solving at μ=" << mu_lo << " and μ=" << mu_hi << std::endl;
        auto [N_lo, E_lo] = evaluate_at_mu(mu_lo, false);
        std::cout << "    μ=" << std::setprecision(4) << mu_lo
                  << "  Σ N_frag=" << N_lo << "  err=" << (N_lo - N_elec) << std::endl;
        auto [N_hi, E_hi] = evaluate_at_mu(mu_hi, false);
        std::cout << "    μ=" << mu_hi
                  << "  Σ N_frag=" << N_hi << "  err=" << (N_hi - N_elec) << std::endl;

        // Expand bracket if needed
        for (int expand = 0; expand < 5; expand++) {
            if ((N_lo - N_elec) * (N_hi - N_elec) <= 0) break;
            mu_lo *= 2.0; mu_hi *= 2.0;
            std::cout << "  Expanding bracket to μ=[" << mu_lo << ", " << mu_hi << "]" << std::endl;
            std::tie(N_lo, E_lo) = evaluate_at_mu(mu_lo, false);
            std::cout << "    μ=" << mu_lo
                      << "  Σ N_frag=" << N_lo << "  err=" << (N_lo - N_elec) << std::endl;
            std::tie(N_hi, E_hi) = evaluate_at_mu(mu_hi, false);
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
                auto [N_mid, E_mid] = evaluate_at_mu(mu_mid, false);
                real_t err_mid = N_mid - N_elec;

                std::cout << "    iter " << std::setw(2) << iter + 1
                          << ": μ=" << std::setprecision(6) << mu_mid
                          << "  Σ N_frag=" << std::setprecision(6) << N_mid
                          << "  err=" << std::scientific << err_mid
                          << std::defaultfloat << std::endl;

                if (std::abs(err_mid) < N_tol || (mu_hi - mu_lo) < 1e-10) {
                    mu_opt = mu_mid;
                    E_corr_opt = E_mid;
                    std::cout << "  μ converged at iter " << iter + 1 << std::endl;
                    break;
                }

                if ((N_lo - N_elec) * err_mid < 0) {
                    mu_hi = mu_mid; N_hi = N_mid; E_hi = E_mid;
                } else {
                    mu_lo = mu_mid; N_lo = N_mid; E_lo = E_mid;
                }

                if (iter == max_iter - 1) {
                    mu_opt = mu_mid;
                    E_corr_opt = E_mid;
                    std::cout << "  μ: max iterations reached, μ=" << std::setprecision(6) << mu_opt
                              << " err=" << err_mid << std::endl;
                }
            }
        }

        // Energy: use μ=0 (T-amplitude energy is only valid for H(μ=0))
        // μ* gives density consistency but doesn't improve energy without proper 2-RDM
        E_corr_opt = E0;
    } else {
        std::cout << "  μ=0 satisfies electron count (err=" << std::scientific << N_err
                  << std::defaultfloat << "), no optimization needed" << std::endl;
    }

    // ================================================================
    //  Summary
    // ================================================================

    std::cout << "\n---- DMET-CCSD Summary ----" << std::endl;
    std::cout << "  Chemical potential μ: " << std::setprecision(6) << mu_opt << " Ha" << std::endl;
    std::cout << "  Total DMET-CCSD correlation energy: "
              << std::fixed << std::setprecision(10) << E_corr_opt << " Ha" << std::endl;
    std::cout << "  HF energy: " << std::setprecision(10) << rhf_.get_total_energy() << " Ha" << std::endl;
    std::cout << "  DMET-CCSD total energy: "
              << std::setprecision(10) << rhf_.get_total_energy() + E_corr_opt << " Ha" << std::endl;
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
