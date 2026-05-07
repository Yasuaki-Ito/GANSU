/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_localizer.hpp"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace gansu {

// ---------------------------------------------------------------------------
// Helper: in-place compute SC = S · C  (S [nao×nao], C [nao×nocc], all
// row-major). SC has shape [nao × nocc].
// ---------------------------------------------------------------------------
static void compute_SC(const real_t* S, const real_t* C,
                       int nao, int nocc, real_t* SC)
{
    for (int mu = 0; mu < nao; mu++) {
        for (int i = 0; i < nocc; i++) {
            real_t v = 0.0;
            for (int nu = 0; nu < nao; nu++) {
                v += S[mu * nao + nu] * C[nu * nocc + i];
            }
            SC[mu * nocc + i] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Pipek-Mezey functional value for diagnostic / test use.
//   L = Σ_i Σ_A (P^A_{ii})^2,  P^A_{ii} = Σ_{μ ∈ A} (SC)_{μi} C_{μi}
// ---------------------------------------------------------------------------
real_t pipek_mezey_functional(
    const real_t* C_occ,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges)
{
    std::vector<real_t> SC(static_cast<size_t>(nao) * nocc);
    compute_SC(S, C_occ, nao, nocc, SC.data());

    real_t L = 0.0;
    for (int i = 0; i < nocc; i++) {
        for (const auto& rng : atom_ao_ranges) {
            real_t p = 0.0;
            for (int mu = rng.first; mu < rng.second; mu++) {
                p += SC[mu * nocc + i] * C_occ[mu * nocc + i];
            }
            L += p * p;
        }
    }
    return L;
}

// ---------------------------------------------------------------------------
// Pipek-Mezey 2×2 Jacobi sweep.
//
// For a pair (i, j) and the symmetric Mulliken-like overlap
//     P^A_{kl} = (1/2) Σ_{μ ∈ A} [C_{μk}(SC)_{μl} + C_{μl}(SC)_{μk}]
// the change of L = Σ_i Σ_A (P^A_{ii})^2 under the rotation
//     C̃_{:,i} =  cosθ C_{:,i} + sinθ C_{:,j}
//     C̃_{:,j} = -sinθ C_{:,i} + cosθ C_{:,j}
// is
//     ΔL = A (1 − cos 4θ) + B sin 4θ
// with
//     A = Σ_A [(P^A_{ij})^2 − (1/4)(P^A_{ii} − P^A_{jj})^2]
//     B = Σ_A  P^A_{ij} (P^A_{ii} − P^A_{jj})
// The maximum lies at  cos 4θ = −A/R, sin 4θ = B/R, R = √(A²+B²),
// with gain  A + R  ≥ 0.  See Pipek & Mezey, JCP 90, 4916 (1989).
// ---------------------------------------------------------------------------
DLPNOLocalizationResult localize_pipek_mezey(
    const real_t* C_occ_in,
    const real_t* S,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    int max_sweep,
    real_t conv_tol,
    int verbose)
{
    if (nocc <= 0) {
        DLPNOLocalizationResult r;
        r.C_LMO.assign(C_occ_in, C_occ_in + static_cast<size_t>(nao) * nocc);
        r.U.clear();
        r.converged = true;
        return r;
    }

    DLPNOLocalizationResult res;
    res.C_LMO.assign(C_occ_in, C_occ_in + static_cast<size_t>(nao) * nocc);
    res.U.assign(static_cast<size_t>(nocc) * nocc, 0.0);
    for (int i = 0; i < nocc; i++) res.U[i * nocc + i] = 1.0;

    real_t* C = res.C_LMO.data();
    real_t* U = res.U.data();
    std::vector<real_t> SC(static_cast<size_t>(nao) * nocc);
    compute_SC(S, C, nao, nocc, SC.data());

    res.functional_initial = pipek_mezey_functional(
        C, S, nao, nocc, atom_ao_ranges);

    if (verbose >= 1) {
        std::cout << "[DLPNO PM] init L = "
                  << std::scientific << std::setprecision(10)
                  << res.functional_initial << std::endl;
    }

    const int n_atoms = static_cast<int>(atom_ao_ranges.size());
    std::vector<real_t> Pii(n_atoms);
    std::vector<real_t> Pjj(n_atoms);
    std::vector<real_t> Pij(n_atoms);

    constexpr real_t kSmall = 1e-30;

    int sweep;
    for (sweep = 0; sweep < max_sweep; sweep++) {
        real_t sweep_gain = 0.0;
        int rotations_done = 0;

        for (int i = 0; i < nocc - 1; i++) {
            for (int j = i + 1; j < nocc; j++) {
                // Build P^A_{ii}, P^A_{jj}, P^A_{ij} per atom.
                for (int a = 0; a < n_atoms; a++) {
                    const int mu0 = atom_ao_ranges[a].first;
                    const int mu1 = atom_ao_ranges[a].second;
                    real_t pii = 0.0, pjj = 0.0, pij = 0.0;
                    for (int mu = mu0; mu < mu1; mu++) {
                        const real_t Ci  = C[mu * nocc + i];
                        const real_t Cj  = C[mu * nocc + j];
                        const real_t SCi = SC[mu * nocc + i];
                        const real_t SCj = SC[mu * nocc + j];
                        pii += SCi * Ci;
                        pjj += SCj * Cj;
                        pij += 0.5 * (Ci * SCj + Cj * SCi);
                    }
                    Pii[a] = pii;
                    Pjj[a] = pjj;
                    Pij[a] = pij;
                }

                real_t A = 0.0, B = 0.0;
                for (int a = 0; a < n_atoms; a++) {
                    const real_t d = Pii[a] - Pjj[a];
                    A += Pij[a] * Pij[a] - 0.25 * d * d;
                    B += Pij[a] * d;
                }

                const real_t R2 = A * A + B * B;
                if (R2 < kSmall) continue;

                // 4θ_opt with cos = -A/R, sin = B/R  →  atan2(B, -A)
                const real_t four_theta = std::atan2(B, -A);
                real_t theta = 0.25 * four_theta;
                // Reduce to (-π/4, π/4] just in case of FP drift.
                const real_t kPi4 = 0.25 * static_cast<real_t>(M_PI);
                while (theta >  kPi4) theta -= 0.5 * static_cast<real_t>(M_PI);
                while (theta <= -kPi4) theta += 0.5 * static_cast<real_t>(M_PI);

                if (std::fabs(theta) < 1e-14) continue;

                const real_t R = std::sqrt(R2);
                const real_t gain = A + R;

                const real_t c = std::cos(theta);
                const real_t s = std::sin(theta);

                // Apply rotation to C and SC (linear in C, so SC also rotates).
                for (int mu = 0; mu < nao; mu++) {
                    const real_t Ci  = C[mu * nocc + i];
                    const real_t Cj  = C[mu * nocc + j];
                    C[mu * nocc + i] =  c * Ci + s * Cj;
                    C[mu * nocc + j] = -s * Ci + c * Cj;
                    const real_t SCi = SC[mu * nocc + i];
                    const real_t SCj = SC[mu * nocc + j];
                    SC[mu * nocc + i] =  c * SCi + s * SCj;
                    SC[mu * nocc + j] = -s * SCi + c * SCj;
                }
                // Apply same rotation to U (right-multiply: U' = U · G).
                for (int k = 0; k < nocc; k++) {
                    const real_t Uki = U[k * nocc + i];
                    const real_t Ukj = U[k * nocc + j];
                    U[k * nocc + i] =  c * Uki + s * Ukj;
                    U[k * nocc + j] = -s * Uki + c * Ukj;
                }

                sweep_gain += gain;
                rotations_done++;
            } // j
        } // i

        if (verbose >= 2) {
            std::cout << "[DLPNO PM] sweep " << std::setw(3) << (sweep + 1)
                      << " rotations=" << rotations_done
                      << " ΔL=" << std::scientific << std::setprecision(6)
                      << sweep_gain << std::endl;
        }

        if (sweep_gain < conv_tol) {
            res.converged = true;
            sweep++;
            break;
        }
    }

    res.n_sweeps = sweep;
    res.functional_final = pipek_mezey_functional(
        C, S, nao, nocc, atom_ao_ranges);

    if (verbose >= 1) {
        std::cout << "[DLPNO PM] " << (res.converged ? "converged" : "MAX_SWEEP")
                  << " in " << res.n_sweeps << " sweeps, final L = "
                  << std::scientific << std::setprecision(10)
                  << res.functional_final << std::endl;
    }

    return res;
}

// ---------------------------------------------------------------------------
// Foster-Boys placeholder. Implementation is deferred to a follow-up phase;
// the function exists so that the front-end accepts `--dlpno_localizer boys`
// once dipole AO integrals are wired through. For Phase 0 it throws.
// ---------------------------------------------------------------------------
DLPNOLocalizationResult localize_foster_boys(
    const real_t* /*C_occ*/,
    const real_t* /*D_x*/,
    const real_t* /*D_y*/,
    const real_t* /*D_z*/,
    int /*nao*/, int /*nocc*/,
    int /*max_sweep*/,
    real_t /*conv_tol*/,
    int /*verbose*/)
{
    throw std::runtime_error(
        "Foster-Boys localizer is not implemented yet (DLPNO Phase 0). "
        "Use --dlpno_localizer pm.");
}

DLPNOLocalizationResult localize_occupied(
    const std::string& method,
    const real_t* C_occ,
    const real_t* S,
    const real_t* D_x, const real_t* D_y, const real_t* D_z,
    int nao, int nocc,
    const std::vector<std::pair<int,int>>& atom_ao_ranges,
    int max_sweep,
    real_t conv_tol,
    int verbose)
{
    if (method == "none" || method == "canonical" || method == "identity") {
        // No-op localizer: keep canonical occupied MOs. Useful for strict
        // validation against canonical MP2 (off-diagonal LMO Fock vanishes).
        DLPNOLocalizationResult r;
        r.C_LMO.assign(C_occ, C_occ + static_cast<size_t>(nao) * nocc);
        r.U.assign(static_cast<size_t>(nocc) * nocc, 0.0);
        for (int i = 0; i < nocc; i++) r.U[i * nocc + i] = 1.0;
        r.functional_initial = pipek_mezey_functional(
            C_occ, S, nao, nocc, atom_ao_ranges);
        r.functional_final = r.functional_initial;
        r.n_sweeps = 0;
        r.converged = true;
        if (verbose >= 1) {
            std::cout << "[DLPNO loc] localizer=none (canonical MOs preserved)"
                      << std::endl;
        }
        return r;
    }
    if (method == "pm") {
        return localize_pipek_mezey(C_occ, S, nao, nocc, atom_ao_ranges,
                                    max_sweep, conv_tol, verbose);
    }
    if (method == "boys") {
        return localize_foster_boys(C_occ, D_x, D_y, D_z, nao, nocc,
                                    max_sweep, conv_tol, verbose);
    }
    if (method == "ibo") {
        // IBO requires a minimal-basis projection (Knizia 2013). Defer.
        throw std::runtime_error(
            "IBO localizer is not implemented yet (DLPNO Phase 0). "
            "Use --dlpno_localizer pm.");
    }
    throw std::runtime_error("Unknown DLPNO localizer: '" + method + "'");
}

} // namespace gansu
