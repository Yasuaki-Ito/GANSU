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
#include <cmath>
#include <algorithm>
#include <tuple>

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
    //  Phase B: evaluate_at_mu — solve all fragments at given μ
    //  Returns {Σ N_frag, Σ E_corr_frag}
    // ================================================================

    struct FragResult {
        real_t E_corr_frag;
        real_t N_frag;
    };

    auto evaluate_at_mu = [&](real_t mu, bool verbose) -> std::pair<real_t, real_t> {
        real_t N_total_frag = 0.0, E_corr_total = 0.0;
        std::vector<FragResult> results(fragments_.size());
        std::vector<bool> solved(fragments_.size(), false);

        for (size_t f = 0; f < fragments_.size(); f++) {
            auto& bd = baths[f];

            if (bd.is_full_system) {
                // 1-fragment = full system → regular CCSD, no μ
                real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
                real_t* d_mo_eri = eri_.build_mo_eri(d_C, nao);
                real_t E_corr = ccsd_spatial_orbital(
                    nullptr, d_C, rhf_.get_orbital_energies().device_ptr(),
                    nao, nocc, false, nullptr, nullptr, nullptr, d_mo_eri, 0);
                tracked_cudaFree(d_mo_eri);
                results[f] = {E_corr, (real_t)N_elec};
                solved[f] = true;
                E_corr_total += E_corr;
                N_total_frag += N_elec;
                if (verbose)
                    std::cout << "  Fragment " << f << ": full system CCSD = "
                              << std::setprecision(10) << E_corr << " Ha" << std::endl;
                continue;
            }

            if (bd.n_emb_occ <= 0 || bd.n_emb_occ >= bd.n_emb) {
                results[f] = {0.0, 0.0};
                solved[f] = true;
                continue;
            }

            // Check equivalence with an already-solved fragment
            bool reused = false;
            for (size_t g = 0; g < f; g++) {
                if (!solved[g] || baths[g].is_full_system) continue;
                auto& bg = baths[g];
                if (bg.n_emb == bd.n_emb && bg.n_emb_occ == bd.n_emb_occ
                    && bg.n_frozen == bd.n_frozen
                    && fragments_[g].n_frag == fragments_[f].n_frag) {
                    // Compare h_emb_base norms as a quick equivalence test
                    real_t norm_f = 0.0, norm_g = 0.0;
                    for (auto v : bd.h_emb_base) norm_f += v * v;
                    for (auto v : bg.h_emb_base) norm_g += v * v;
                    if (std::abs(norm_f - norm_g) < 1e-10 * std::max(norm_f, norm_g)) {
                        results[f] = results[g];
                        solved[f] = true;
                        reused = true;
                        E_corr_total += results[f].E_corr_frag;
                        N_total_frag += results[f].N_frag;
                        if (verbose)
                            std::cout << "  Fragment " << f << ": equivalent to " << g
                                      << " (E=" << std::setprecision(10) << results[f].E_corr_frag
                                      << ", N=" << std::setprecision(4) << results[f].N_frag << ")" << std::endl;
                        break;
                    }
                }
            }
            if (reused) continue;

            const int ne = bd.n_emb, no = bd.n_emb_occ, nv = ne - no;
            const int no_act = no - bd.n_frozen;  // active occupied (CCSD T amplitude dim)
            const int n_frag_emb = fragments_[f].n_frag;
            const size_t t1sz = (size_t)no_act * nv;  // n_frozen=0 → no_act=no
            const size_t t2sz = (size_t)no_act * no_act * nv * nv;

            // 1. h_emb(μ) = h_emb_base - μ P_frag
            std::vector<real_t> h_emb(bd.h_emb_base);
            for (int p = 0; p < n_frag_emb; p++)
                h_emb[p * ne + p] -= mu;

            // 2. Diagonalize h_emb(μ) → ε, U
            real_t *d_h_emb = nullptr, *d_eigvals = nullptr, *d_eigvecs = nullptr;
            tracked_cudaMalloc(&d_h_emb, ne * ne * sizeof(real_t));
            tracked_cudaMalloc(&d_eigvals, ne * sizeof(real_t));
            tracked_cudaMalloc(&d_eigvecs, ne * ne * sizeof(real_t));
            cudaMemcpy(d_h_emb, h_emb.data(), ne * ne * sizeof(real_t), cudaMemcpyHostToDevice);
            gpu::eigenDecomposition(d_h_emb, d_eigvals, d_eigvecs, ne);
            tracked_cudaFree(d_h_emb);

            // Level shift
            std::vector<real_t> h_eps(ne);
            cudaMemcpy(h_eps.data(), d_eigvals, ne * sizeof(real_t), cudaMemcpyDeviceToHost);
            if (no > 0 && no < ne) {
                real_t gap = h_eps[no] - h_eps[no - 1];
                const real_t target_gap = 0.1;
                if (gap < target_gap) {
                    real_t shift = target_gap - gap;
                    for (int i = no; i < ne; i++) h_eps[i] += shift;
                    cudaMemcpy(d_eigvals, h_eps.data(), ne * sizeof(real_t), cudaMemcpyHostToDevice);
                }
            }

            std::vector<real_t> h_eigvecs(ne * ne);
            cudaMemcpy(h_eigvecs.data(), d_eigvecs, ne * ne * sizeof(real_t), cudaMemcpyDeviceToHost);
            tracked_cudaFree(d_eigvecs);

            // 3. C_can = C_emb × U
            std::vector<real_t> h_C_can(nao * ne, 0.0);
            for (int mu_idx = 0; mu_idx < nao; mu_idx++)
                for (int i = 0; i < ne; i++) {
                    real_t val = 0.0;
                    for (int p = 0; p < ne; p++)
                        val += bd.C_emb[mu_idx * ne + p] * h_eigvecs[p * ne + i];
                    h_C_can[mu_idx * ne + i] = val;
                }

            // 4. Build MO ERI
            cudaSetDevice(0);
            real_t* d_C_can = nullptr;
            tracked_cudaMalloc(&d_C_can, nao * ne * sizeof(real_t));
            cudaMemcpy(d_C_can, h_C_can.data(), nao * ne * sizeof(real_t), cudaMemcpyHostToDevice);
            real_t* d_eri_mo = eri_.build_mo_eri(d_C_can, ne);

            // 5. CCSD
            real_t *d_t1 = nullptr, *d_t2 = nullptr;
            real_t E_CCSD = ccsd_spatial_orbital(
                nullptr, d_C_can, d_eigvals,
                ne, no, false, nullptr, &d_t1, &d_t2,
                d_eri_mo, bd.n_frozen);

            // 6. Download T amplitudes and ERI
            std::vector<real_t> h_t1(t1sz), h_t2(t2sz);
            std::vector<real_t> h_eri((size_t)ne*ne*ne*ne);
            cudaMemcpy(h_t1.data(), d_t1, t1sz*sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_t2.data(), d_t2, t2sz*sizeof(real_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_eri.data(), d_eri_mo, h_eri.size()*sizeof(real_t), cudaMemcpyDeviceToHost);

            // 7. Lambda solver (CPU, L=T initial guess, DIIS+damping)
            std::vector<real_t> h_l1(h_t1), h_l2(h_t2);  // L=T initial guess
            bool lambda_ok = solve_ccsd_lambda_cpu(
                no, nv, h_eps.data(), h_eri.data(),
                h_t1.data(), h_t2.data(), h_l1.data(), h_l2.data(),
                300, 1e-5, verbose ? 1 : 0);

            if (!lambda_ok && verbose)
                std::cout << "  Lambda: falling back to L=T" << std::endl;

            // 8. Democratic E_corr_frag from T amplitudes
            real_t E_frag = compute_dmet_fragment_energy(
                no, nv, h_eri.data(), h_t1.data(), h_t2.data(),
                n_frag_emb, h_eigvecs.data(), bd.n_frozen);

            // 8. Fragment electron count from relaxed 1-RDM (if Lambda ok) or HF
            real_t N_frag = 2.0 * bd.n_core;
            if (lambda_ok) {
                // Relaxed 1-RDM → accurate N_frag
                const int na_act = no_act + nv;
                std::vector<real_t> dm1(na_act * na_act, 0.0);
                build_ccsd_1rdm_mo_cpu(no_act, nv, h_t1.data(), h_t2.data(),
                                       h_l1.data(), h_l2.data(), dm1.data());
                const int nf = bd.n_frozen;
                for (int i = 0; i < na_act; i++)
                    for (int j = 0; j < na_act; j++) {
                        real_t P_ij = 0.0;
                        for (int p = 0; p < n_frag_emb; p++)
                            P_ij += h_eigvecs[p * ne + (i+nf)] * h_eigvecs[p * ne + (j+nf)];
                        N_frag += P_ij * dm1[i * na_act + j];
                    }
            } else {
                // HF fallback: N_frag = 2*Σ_{i<no} P_can[i,i]
                for (int i = 0; i < no; i++) {
                    real_t P_ii = 0.0;
                    for (int p = 0; p < n_frag_emb; p++)
                        P_ii += h_eigvecs[p * ne + i] * h_eigvecs[p * ne + i];
                    N_frag += 2.0 * P_ii;
                }
            }

            results[f] = {E_frag, N_frag};
            solved[f] = true;
            E_corr_total += E_frag;
            N_total_frag += N_frag;

            // 10. RDM-based E_corr verification (PySCF convention, P=I)
            if (verbose && lambda_ok) {
                const int na_act = no_act + nv;
                const int nf = bd.n_frozen;

                // Build PySCF dm2 from relaxed 1-RDM + T/L
                std::vector<real_t> dm1_v(na_act * na_act, 0.0);
                std::vector<real_t> dm2_v((size_t)na_act*na_act*na_act*na_act, 0.0);
                build_ccsd_1rdm_mo_cpu(no_act, nv, h_t1.data(), h_t2.data(),
                                       h_l1.data(), h_l2.data(), dm1_v.data());
                build_ccsd_2rdm_pyscf_cpu(no_act, nv, h_t1.data(), h_t2.data(),
                                          h_l1.data(), h_l2.data(),
                                          dm1_v.data(), dm2_v.data());

                // h_core in canonical(μ) basis
                std::vector<real_t> hc(na_act * na_act, 0.0);
                for (int p = 0; p < na_act; p++)
                  for (int q = 0; q < na_act; q++) {
                    real_t val = 0.0;
                    for (int a = 0; a < ne; a++)
                      for (int b = 0; b < ne; b++)
                        val += h_eigvecs[a*ne+(p+nf)] * bd.h_core_emb[a*ne+b] * h_eigvecs[b*ne+(q+nf)];
                    hc[p * na_act + q] = val;
                  }

                // E_1e = Tr(h_core * dm1) - E_1e_HF
                real_t e1 = 0.0, e1hf = 0.0;
                for (int p = 0; p < na_act; p++)
                  for (int q = 0; q < na_act; q++) {
                    e1 += hc[p*na_act+q] * dm1_v[q*na_act+p];
                    real_t hf = (q == p && p < no_act) ? 2.0 : 0.0;
                    e1hf += hc[p*na_act+q] * hf;  // only diagonal survives
                  }

                // E_2e = 0.5*einsum('pqrs,pqrs', eri, dm2) - E_2e_HF
                const size_t na2a = (size_t)na_act * na_act;
                real_t e2 = 0.0, e2hf = 0.0;
                for (int p = 0; p < na_act; p++)
                  for (int q = 0; q < na_act; q++)
                    for (int r = 0; r < na_act; r++)
                      for (int s = 0; s < na_act; s++) {
                        real_t eri_v = h_eri[((size_t)(p+nf)*ne+(q+nf))*((size_t)ne*ne)
                                            +(size_t)(r+nf)*ne+(s+nf)];
                        e2 += eri_v * dm2_v[((size_t)p*na_act+q)*na2a + r*na_act+s];
                        // HF dm2: 4δ_{pq}δ_{rs} - 2δ_{ps}δ_{rq}
                        real_t hfv = 0.0;
                        if (p < no_act && q < no_act && r < no_act && s < no_act) {
                            if (p==q && r==s) hfv += 4.0;
                            if (p==s && r==q) hfv -= 2.0;
                        }
                        e2hf += eri_v * hfv;
                      }
                e2 *= 0.5; e2hf *= 0.5;

                real_t e_corr_rdm = (e1 - e1hf) + (e2 - e2hf);
                std::cout << "  Fragment " << f << ": E_T=" << std::setprecision(10) << E_frag
                          << " E_RDM=" << e_corr_rdm
                          << " N=" << std::setprecision(4) << N_frag
                          << " raw=" << std::setprecision(10) << E_CCSD << std::endl;
            } else if (verbose) {
                std::cout << "  Fragment " << f << ": E_corr=" << std::setprecision(10) << E_frag
                          << " N_frag=" << std::setprecision(4) << N_frag
                          << " (L=T)" << std::endl;
            }

            // Cleanup
            tracked_cudaFree(d_t1); tracked_cudaFree(d_t2);
            tracked_cudaFree(d_C_can); tracked_cudaFree(d_eri_mo);
            tracked_cudaFree(d_eigvals);
        }

        return {N_total_frag, E_corr_total};
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
        auto [N_lo, E_lo] = evaluate_at_mu(mu_lo, false);
        auto [N_hi, E_hi] = evaluate_at_mu(mu_hi, false);

        // Expand bracket if needed
        for (int expand = 0; expand < 5; expand++) {
            if ((N_lo - N_elec) * (N_hi - N_elec) <= 0) break;
            mu_lo *= 2.0; mu_hi *= 2.0;
            std::tie(N_lo, E_lo) = evaluate_at_mu(mu_lo, false);
            std::tie(N_hi, E_hi) = evaluate_at_mu(mu_hi, false);
        }

        if ((N_lo - N_elec) * (N_hi - N_elec) > 0) {
            std::cout << "  WARNING: bisection bracket failed, using μ=0" << std::endl;
        } else {
            const int max_iter = 30;
            for (int iter = 0; iter < max_iter; iter++) {
                real_t mu_mid = 0.5 * (mu_lo + mu_hi);
                auto [N_mid, E_mid] = evaluate_at_mu(mu_mid, false);
                real_t err_mid = N_mid - N_elec;

                if (std::abs(err_mid) < N_tol || (mu_hi - mu_lo) < 1e-10) {
                    mu_opt = mu_mid;
                    E_corr_opt = E_mid;
                    std::cout << "  μ converged: μ=" << std::setprecision(6) << mu_opt
                              << " Σ N_frag=" << std::setprecision(6) << N_mid
                              << " err=" << std::scientific << err_mid
                              << std::defaultfloat << " (iter=" << iter + 1 << ")" << std::endl;
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
