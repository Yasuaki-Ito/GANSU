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
 * @file oscillator_strength.cu
 * @brief Oscillator strength computation for excited states
 *
 * AO dipole integrals are computed on CPU using McMurchie-Davidson:
 *   ⟨a|x|b⟩ = K_AB (π/p)^{3/2} [P_x E^x_0 + E^x_1] E^y_0 E^z_0
 *
 * where E^x_t(i,j) are Hermite expansion coefficients computed by
 * the Obara-Saika recurrence.
 */

#include "oscillator_strength.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace gansu {

// ============================================================
//  Host-side Hermite expansion coefficient E^t(i, j)
// ============================================================

// Angular momentum component table (same order as GANSU's loop_to_ang)
static const int h_loop_to_ang[5][15][3] = {
    // s (1 component)
    {{0,0,0}},
    // p (3 components)
    {{1,0,0}, {0,1,0}, {0,0,1}},
    // d (6 components)
    {{2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {1,0,1}, {0,1,1}},
    // f (10 components)
    {{3,0,0}, {0,3,0}, {0,0,3}, {1,2,0}, {2,1,0}, {2,0,1}, {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1}},
    // g (15 components)
    {{4,0,0}, {0,4,0}, {0,0,4}, {3,1,0}, {3,0,1}, {1,3,0}, {0,3,1}, {1,0,3}, {0,1,3}, {2,2,0}, {2,0,2}, {0,2,2}, {2,1,1}, {1,2,1}, {1,1,2}},
};

static int h_comb_max(int L) {
    return (L + 1) * (L + 2) / 2;
}

/**
 * Compute 1D overlap integral (unnormalized, excluding K_AB and (π/p)^{1/2})
 * using Obara-Saika recurrence:
 *   S(i+1, j) = X_PA * S(i, j) + 1/(2p) [i * S(i-1, j) + j * S(i, j-1)]
 *   S(i, j+1) = X_PB * S(i, j) + 1/(2p) [i * S(i-1, j) + j * S(i, j-1)]
 * with S(0, 0) = 1.
 */
static double overlap_1d(int la, int lb, double XPA, double XPB, double one_over_2p)
{
    // Build table S[i][j] for 0 <= i <= la, 0 <= j <= lb
    // Max angular momentum: g (4) + 1 (for dipole) = 5
    const int MAXL = 6;
    double S[MAXL + 1][MAXL + 1];
    std::memset(S, 0, sizeof(S));

    S[0][0] = 1.0;

    // Fill column j=0: S[i+1][0] = XPA * S[i][0] + one_over_2p * i * S[i-1][0]
    for (int i = 0; i < la + 1; i++) {  // need up to la+1 for dipole
        S[i + 1][0] = XPA * S[i][0] + one_over_2p * i * (i > 0 ? S[i - 1][0] : 0.0);
    }

    // Fill rows
    for (int j = 0; j < lb; j++) {
        for (int i = 0; i <= la + 1; i++) {
            S[i][j + 1] = XPB * S[i][j]
                         + one_over_2p * (i * (i > 0 ? S[i - 1][j] : 0.0)
                                        + j * (j > 0 ? S[i][j - 1] : 0.0));
        }
    }

    return S[la][lb];
}

/**
 * Compute double factorial (2n-1)!! = 1·3·5·...·(2n-1)
 */
static double double_factorial(int n)
{
    // (2n-1)!! for n >= 0
    // For n=0: (-1)!! = 1 by convention
    if (n <= 0) return 1.0;
    double result = 1.0;
    for (int k = 2 * n - 1; k >= 1; k -= 2) {
        result *= k;
    }
    return result;
}

/**
 * Cartesian Gaussian normalization: N(α, l, m, n)
 * Same as calcNorm in int2e.hpp but on host.
 */
static double host_calcNorm(double alpha, int l, int m, int n)
{
    // N = 2^(l+m+n) / sqrt((2l-1)!! (2m-1)!! (2n-1)!!) * (π)^{3/4} * α^{(2(l+m+n)+3)/4}
    // Note: π^{3/4} = PI3_4
    static const double PI3_4 = std::pow(2.0 / M_PI, 0.75);  // (2/π)^{3/4} ≈ 0.71270547
    double num = static_cast<double>(1 << (l + m + n));
    double den = std::sqrt(double_factorial(l) * double_factorial(m) * double_factorial(n));
    return num / den * PI3_4 * std::pow(alpha, (2.0 * (l + m + n) + 3.0) / 4.0);
}


void compute_ao_dipole_integrals(
    const PrimitiveShell* shells, size_t num_shells,
    const real_t* cgto_norms,
    int nao,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    std::vector<real_t>& dipole_x,
    std::vector<real_t>& dipole_y,
    std::vector<real_t>& dipole_z)
{
    size_t n2 = (size_t)nao * nao;
    dipole_x.assign(n2, 0.0);
    dipole_y.assign(n2, 0.0);
    dipole_z.assign(n2, 0.0);

    // Iterate over all pairs of primitive shells
    for (size_t ia = 0; ia < num_shells; ia++) {
        const PrimitiveShell& a = shells[ia];
        for (size_t ib = 0; ib < num_shells; ib++) {
            const PrimitiveShell& b = shells[ib];

            double alpha = a.exponent;
            double beta = b.exponent;
            double p = alpha + beta;
            double one_over_2p = 0.5 / p;

            double Ax = a.coordinate.x, Ay = a.coordinate.y, Az = a.coordinate.z;
            double Bx = b.coordinate.x, By = b.coordinate.y, Bz = b.coordinate.z;

            double Dx = Ax - Bx, Dy = Ay - By, Dz = Az - Bz;
            double dist2 = Dx * Dx + Dy * Dy + Dz * Dz;

            double Px = (alpha * Ax + beta * Bx) / p;
            double Py = (alpha * Ay + beta * By) / p;
            double Pz = (alpha * Az + beta * Bz) / p;

            double XPA = Px - Ax, YPA = Py - Ay, ZPA = Pz - Az;
            double XPB = Px - Bx, YPB = Py - By, ZPB = Pz - Bz;

            // K_AB * (π/p)^{3/2}
            double K_AB = std::exp(-alpha * beta / p * dist2);
            double prefactor = K_AB * std::pow(M_PI / p, 1.5);

            // Contraction coefficients
            double coeff = a.coefficient * b.coefficient;

            int La = a.shell_type;
            int Lb = b.shell_type;

            for (int lmn_a = 0; lmn_a < h_comb_max(La); lmn_a++) {
                int l1 = h_loop_to_ang[La][lmn_a][0];
                int m1 = h_loop_to_ang[La][lmn_a][1];
                int n1 = h_loop_to_ang[La][lmn_a][2];
                double Na = host_calcNorm(alpha, l1, m1, n1);

                for (int lmn_b = 0; lmn_b < h_comb_max(Lb); lmn_b++) {
                    int l2 = h_loop_to_ang[Lb][lmn_b][0];
                    int m2 = h_loop_to_ang[Lb][lmn_b][1];
                    int n2 = h_loop_to_ang[Lb][lmn_b][2];
                    double Nb = host_calcNorm(beta, l2, m2, n2);

                    size_t mu = a.basis_index + lmn_a;
                    size_t nu = b.basis_index + lmn_b;

                    double norm_coeff = coeff * prefactor * Na * Nb
                                      * cgto_norms[mu] * cgto_norms[nu];

                    // Overlap components
                    double Sx = overlap_1d(l1, l2, XPA, XPB, one_over_2p);
                    double Sy = overlap_1d(m1, m2, YPA, YPB, one_over_2p);
                    double Sz = overlap_1d(n1, n2, ZPA, ZPB, one_over_2p);

                    // Dipole: ⟨a|x|b⟩ = [S_x(la+1, lb) + Ax * S_x(la, lb)] * S_y * S_z
                    double Sx_plus = overlap_1d(l1 + 1, l2, XPA, XPB, one_over_2p);
                    double Sy_plus = overlap_1d(m1 + 1, m2, YPA, YPB, one_over_2p);
                    double Sz_plus = overlap_1d(n1 + 1, n2, ZPA, ZPB, one_over_2p);

                    dipole_x[mu * nao + nu] += norm_coeff * (Sx_plus + Ax * Sx) * Sy * Sz;
                    dipole_y[mu * nao + nu] += norm_coeff * Sx * (Sy_plus + Ay * Sy) * Sz;
                    dipole_z[mu * nao + nu] += norm_coeff * Sx * Sy * (Sz_plus + Az * Sz);
                }
            }
        }
    }
}


std::vector<real_t> transform_dipole_ao_to_mo_ov(
    const std::vector<real_t>& dipole_ao,
    const real_t* C, int nao, int nocc, int nvir)
{
    // μ^MO_{ia} = Σ_{μν} C_{μi} μ^AO_{μν} C_{νa}
    // C is row-major: C[mu * nao + p]
    std::vector<real_t> dipole_ov(nocc * nvir, 0.0);

    // First: tmp[μ][a] = Σ_ν μ^AO_{μν} C_{ν,nocc+a}
    std::vector<real_t> tmp(nao * nvir, 0.0);
    for (int mu = 0; mu < nao; mu++) {
        for (int a = 0; a < nvir; a++) {
            double val = 0.0;
            for (int nu = 0; nu < nao; nu++) {
                val += dipole_ao[mu * nao + nu] * C[nu * nao + (nocc + a)];
            }
            tmp[mu * nvir + a] = val;
        }
    }

    // Then: dipole_ov[i][a] = Σ_μ C_{μi} tmp[μ][a]
    for (int i = 0; i < nocc; i++) {
        for (int a = 0; a < nvir; a++) {
            double val = 0.0;
            for (int mu = 0; mu < nao; mu++) {
                val += C[mu * nao + i] * tmp[mu * nvir + a];
            }
            dipole_ov[i * nvir + a] = val;
        }
    }

    return dipole_ov;
}


std::vector<real_t> compute_oscillator_strengths(
    const real_t* h_eigenvectors,
    const std::vector<real_t>& excitation_energies,
    const std::vector<real_t>& dipole_mo_ov_x,
    const std::vector<real_t>& dipole_mo_ov_y,
    const std::vector<real_t>& dipole_mo_ov_z,
    int n_states, int nocc, int nvir)
{
    int singles_dim = nocc * nvir;
    std::vector<real_t> osc_strengths(n_states);

    for (int state = 0; state < n_states; state++) {
        const real_t* R1 = &h_eigenvectors[state * singles_dim];

        // Transition dipole moment: ⟨0|μ_d|n⟩ = √2 Σ_{ia} R1_{ia} μ^d_{ia}
        // Factor √2 for RHF singlet excitations (α+β contributions)
        double tdm_x = 0.0, tdm_y = 0.0, tdm_z = 0.0;
        for (int ia = 0; ia < singles_dim; ia++) {
            tdm_x += R1[ia] * dipole_mo_ov_x[ia];
            tdm_y += R1[ia] * dipole_mo_ov_y[ia];
            tdm_z += R1[ia] * dipole_mo_ov_z[ia];
        }
        tdm_x *= std::sqrt(2.0);
        tdm_y *= std::sqrt(2.0);
        tdm_z *= std::sqrt(2.0);

        double tdm2 = tdm_x * tdm_x + tdm_y * tdm_y + tdm_z * tdm_z;

        // Oscillator strength: f = (2/3) ω |⟨0|μ|n⟩|²
        osc_strengths[state] = (2.0 / 3.0) * excitation_energies[state] * tdm2;
    }

    return osc_strengths;
}


ExcitedStateResult compute_excited_state_properties(
    const std::string& method_name,
    const PrimitiveShell* shells, size_t num_shells,
    const real_t* cgto_norms,
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const real_t* C_host,
    const std::vector<real_t>& excitation_energies,
    const real_t* h_eigenvectors,
    int n_states, int nao, int nocc, int nvir)
{
    const int singles_dim = nocc * nvir;

    // --- Compute AO dipole integrals ---
    std::vector<real_t> dip_ao_x, dip_ao_y, dip_ao_z;
    compute_ao_dipole_integrals(
        shells, num_shells, cgto_norms, nao, shell_type_infos,
        dip_ao_x, dip_ao_y, dip_ao_z);

    // --- Transform to MO ov block ---
    auto dip_mo_x = transform_dipole_ao_to_mo_ov(dip_ao_x, C_host, nao, nocc, nvir);
    auto dip_mo_y = transform_dipole_ao_to_mo_ov(dip_ao_y, C_host, nao, nocc, nvir);
    auto dip_mo_z = transform_dipole_ao_to_mo_ov(dip_ao_z, C_host, nao, nocc, nvir);

    // --- Compute oscillator strengths ---
    // Triplet→singlet electric dipole transitions are spin-forbidden (f=0)
    bool is_triplet = (method_name.find("triplet") != std::string::npos);
    std::vector<real_t> osc_strengths;
    if (is_triplet) {
        osc_strengths.assign(n_states, 0.0);
    } else {
        osc_strengths = compute_oscillator_strengths(
            h_eigenvectors, excitation_energies,
            dip_mo_x, dip_mo_y, dip_mo_z,
            n_states, nocc, nvir);
    }

    // --- Build report string ---
    const real_t eV = 27.211386245988;
    std::ostringstream ss;

    ss << std::setfill(' ');
    ss << "\n============================================================" << std::endl;
    ss << "             " << method_name << " Excited States" << std::endl;
    ss << "============================================================" << std::endl;
    ss << std::endl;
    ss << std::setw(7) << "State"
       << std::setw(18) << "Energy (Ha)"
       << std::setw(16) << "Energy (eV)"
       << std::setw(10) << "f"
       << "   Dominant Transitions" << std::endl;
    ss << std::setw(7) << "-----"
       << std::setw(18) << "-----------"
       << std::setw(16) << "-----------"
       << std::setw(10) << "------"
       << "   --------------------" << std::endl;

    for (int state = 0; state < n_states; state++) {
        real_t exc_energy = excitation_energies[state];
        real_t exc_energy_eV = exc_energy * eV;

        const real_t* evec = &h_eigenvectors[state * singles_dim];

        struct Transition { int i, a; real_t coeff; };
        std::vector<Transition> transitions;
        for (int i = 0; i < nocc; i++) {
            for (int a = 0; a < nvir; a++) {
                real_t c = evec[i * nvir + a];
                if (std::abs(c) > 0.1) {
                    transitions.push_back({i, a, c});
                }
            }
        }
        std::sort(transitions.begin(), transitions.end(),
                  [](const Transition& a, const Transition& b) {
                      return std::abs(a.coeff) > std::abs(b.coeff);
                  });

        ss << std::setw(5) << state + 1
           << std::fixed << std::setprecision(6)
           << std::setw(18) << exc_energy
           << std::setprecision(4)
           << std::setw(16) << exc_energy_eV
           << std::setw(10) << osc_strengths[state]
           << "   ";

        int max_print = std::min((int)transitions.size(), 3);
        for (int t = 0; t < max_print; t++) {
            const auto& tr = transitions[t];
            int homo_offset = nocc - 1 - tr.i;
            int lumo_offset = tr.a;
            if (t > 0) ss << ", ";
            if (homo_offset == 0) ss << "HOMO";
            else ss << "HOMO-" << homo_offset;
            ss << " -> ";
            if (lumo_offset == 0) ss << "LUMO";
            else ss << "LUMO+" << lumo_offset;
            ss << " (" << std::fixed << std::setprecision(2)
               << std::abs(tr.coeff) << ")";
        }
        ss << std::endl;
    }
    ss << "============================================================" << std::endl;

    ExcitedStateResult result;
    result.oscillator_strengths = osc_strengths;
    result.report = ss.str();
    return result;
}

} // namespace gansu
