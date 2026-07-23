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

// CIS-guided automatic fragment extraction for DMET-STEOM.
//
// Runs a full-system state-averaged CIS-NTO, expands each active NTO into the
// Löwdin AO basis (same metric as the §4.3 bath-sufficiency gauge in dmet.cu, so
// the "where does the excitation live" and "does the bath capture it" measures
// share one ruler), scores every atom by its occupation-weighted hole+particle
// NTO population, and greedily selects the chromophore atoms. See
// AQUA/フラグメント自動分割for励起計算.md for the design and novelty positioning.

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>   // std::getenv
#include <vector>
#include <set>
#include <fstream>   // per-state JSON (Phase C)
#include <string>
#include <Eigen/Dense>

#include "dmet_auto_fragment.hpp"
#include "rhf.hpp"
#include "eri.hpp"
#include "cis_nto_active_space.hpp"
#include "utils.hpp"   // atomic_number_to_element_name

namespace gansu {

namespace {

// Greedy atom selection shared by the main selection and the f/2·2f sensitivity
// probe. Candidates = atoms with score ≥ floor, taken in descending order until
// the cumulative coverage target T is met (or the budget caps n_emb). Returns
// the selected atom set (sorted asc) plus achieved coverage / flags.
struct Selection {
    std::vector<int> atoms;
    double coverage   = 0.0;
    bool   budget_hit = false;
    bool   delocalized = false;
};

Selection greedy_select(const std::vector<double>& score,   // per-atom, normalized (Σ=1)
                        const std::vector<int>& ao_count,   // per-atom AO count
                        double floor, double target, int budget)
{
    const int N = (int)score.size();
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return score[a] > score[b]; });

    Selection sel;
    int n_frag_ao = 0;
    for (int a : order) {
        if (score[a] < floor) break;               // below noise floor → stop
        const int est_n_emb = 2 * (n_frag_ao + ao_count[a]) + 30;  // NTO-aug allowance
        if (est_n_emb > budget) { sel.budget_hit = true; break; }
        sel.atoms.push_back(a);
        n_frag_ao += ao_count[a];
        sel.coverage += score[a];
        if (sel.coverage >= target) break;
    }
    if (sel.coverage < target && !sel.budget_hit) sel.delocalized = true;
    std::sort(sel.atoms.begin(), sel.atoms.end());
    return sel;
}

// Connected-component count of the selected atoms under a bond-length cutoff
// (heavy-heavy bonds are ≤ ~2.9 Bohr; 3.2 Bohr keeps a fragment connected while
// separating a through-space donor…acceptor pair into distinct regions).
template <class Atoms>
int connected_components(const std::vector<int>& atoms, const Atoms& all,
                         real_t bond_cut_bohr = 3.2)
{
    const int n = (int)atoms.size();
    if (n <= 1) return n;
    const real_t r2c = bond_cut_bohr * bond_cut_bohr;
    std::vector<int> comp(n, -1);
    int ncomp = 0;
    for (int s = 0; s < n; ++s) {
        if (comp[s] >= 0) continue;
        // BFS from s.
        std::vector<int> stack{s};
        comp[s] = ncomp;
        while (!stack.empty()) {
            const int i = stack.back(); stack.pop_back();
            const auto& ci = all[atoms[i]].coordinate;
            for (int j = 0; j < n; ++j) {
                if (comp[j] >= 0) continue;
                const auto& cj = all[atoms[j]].coordinate;
                const real_t dx = ci.x - cj.x, dy = ci.y - cj.y, dz = ci.z - cj.z;
                if (dx*dx + dy*dy + dz*dz < r2c) { comp[j] = ncomp; stack.push_back(j); }
            }
        }
        ++ncomp;
    }
    return ncomp;
}

} // anonymous namespace

int dmet_steom_default_budget(const RHF& rhf) {
    bool dlpno = (rhf.get_dmet_cluster_solver() == "dlpno");
    if (const char* e = std::getenv("GANSU_DMET_STEOM_DLPNO")) if (e[0] == '2') dlpno = true;
    return dlpno ? 700 : 460;
}

DMETAutoFragmentResult dmet_steom_auto_extract_fragment(
    RHF& rhf, ERI& eri, int n_states, int num_atoms, int nao, int nocc)
{
    DMETAutoFragmentResult out;

    // ---- Parameters -------------------------------------------------------
    const double T          = rhf.get_dmet_steom_auto_coverage();
    const double floor      = rhf.get_dmet_steom_auto_atom_floor();
    const bool   include_h  = rhf.get_dmet_steom_auto_include_h();
    int          budget     = rhf.get_dmet_steom_auto_budget();
    if (budget <= 0) budget = dmet_steom_default_budget(rhf);

    int n_cis = rhf.get_dmet_steom_auto_n_cis();
    if (n_cis <= 0) {
        const int base = rhf.get_steom_n_root_cis();
        n_cis = std::max(base > 0 ? base : 0, n_states + 4);
    }
    out.n_cis_used = n_cis;

    std::cout << "\n==== DMET-STEOM auto-fragment (CIS-NTO per-atom weights) ====" << std::endl;
    std::cout << "  n_cis=" << n_cis << " coverage_target=" << T
              << " atom_floor=" << floor << " budget(n_emb)=" << budget
              << " include_h=" << (include_h ? "yes" : "no") << std::endl;

    // ---- Full-system state-averaged CIS-NTO (cached on rhf for reuse) ------
    eri.compute_cis_nto(n_cis);

    // Optionally FOCUS the extraction NTO on the lowest `focus_states` CIS roots
    // (0/1 per-state weights) instead of averaging over all n_cis. The full
    // state-average mixes π→π* with n→π*; for a molecule with low-lying lone-pair
    // excitations (e.g. dox: extra carbonyl oxygens outrank the ring carbons in the
    // average) this pulls the wrong atoms into the fragment. Focusing on the target
    // (lowest, typically the bright π→π*) states cleans the chromophore. The cached
    // state-averaged NTO on rhf is left untouched (the downstream NTO-bath gauge/
    // augmentation still uses the full average). RI path only (needs stashed amps).
    CISNTOResult focused;
    bool use_focused = false;
    const int focus = rhf.get_dmet_steom_auto_focus_states();
    if (focus > 0) {
        const CISNTOResult& savg = rhf.get_cis_nto_result();
        const std::vector<real_t>& amp = rhf.get_last_cis_amplitudes();
        const int noa = savg.nocc_active, nvr = savg.nvir, nfzc = savg.num_frozen;
        const size_t per = (size_t)noa * nvr;
        const int n_have = (per > 0) ? (int)(amp.size() / per) : 0;
        if (n_have <= 0) {
            std::cout << "  [auto-frag] focus_states=" << focus << " set but no stashed CIS "
                         "amplitudes (non-RI path?) → using state-averaged NTO." << std::endl;
        } else {
            const int nf = std::min(focus, n_have);
            std::vector<real_t> w(n_have, 0.0);
            for (int n = 0; n < nf; ++n) w[n] = 1.0;
            CISNTOActiveSpace::Params p;
            p.o_thresh = (real_t)rhf.get_cis_nto_o_thresh();
            p.v_thresh = (real_t)rhf.get_cis_nto_v_thresh();
            p.verbose  = 0;
            p.weights  = w;
            focused = CISNTOActiveSpace::compute(amp.data(), n_have, noa, nvr, nfzc, p);
            use_focused = true;
            std::cout << "  [auto-frag] focusing extraction on the lowest " << nf
                      << " CIS root(s): n_act_occ " << savg.n_act_occ << "→" << focused.n_act_occ
                      << ", n_act_vir " << savg.n_act_vir << "→" << focused.n_act_vir << std::endl;
        }
    }
    const CISNTOResult& nto = use_focused ? focused : rhf.get_cis_nto_result();
    const int nocc_act = nto.nocc_active;
    const int nfz      = nto.num_frozen;
    const int nvir     = nto.nvir;
    const int full_occ = nocc;

    auto fallback_whole = [&](const char* why) {
        std::cout << "  [auto-frag] " << why
                  << " → falling back to the whole molecule (plain STEOM)." << std::endl;
        out.atoms.resize(num_atoms);
        std::iota(out.atoms.begin(), out.atoms.end(), 0);
        out.coverage = 1.0;
        out.n_components = 1;
        return out;
    };
    if (nocc_act <= 0 || (nto.n_act_occ <= 0 && nto.n_act_vir <= 0))
        return fallback_whole("CIS-NTO produced no active excitation space");

    // ---- Löwdin S^{1/2} (Eigen, CPU one-shot) -----------------------------
    rhf.get_coefficient_matrix().toHost();
    rhf.get_overlap_matrix().toHost();
    const real_t* h_C = rhf.get_coefficient_matrix().host_ptr();
    const real_t* h_S = rhf.get_overlap_matrix().host_ptr();

    Eigen::MatrixXd Smat(nao, nao);
    for (int i = 0; i < nao; ++i)
        for (int j = 0; j < nao; ++j) Smat(i, j) = (double)h_S[(size_t)i * nao + j];
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Smat);
    Eigen::VectorXd d = es.eigenvalues();
    for (int k = 0; k < nao; ++k) d[k] = (d[k] > 1e-10) ? std::sqrt(d[k]) : 0.0;
    const Eigen::MatrixXd Shalf = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().transpose();

    // Active-occupied / virtual canonical MO blocks, then Löwdin: C_lo = S^{1/2} C.
    // Löwdin MO columns are orthonormal, so each NTO built below is unit-norm and
    // its per-AO squares partition into 1 → a clean per-atom probability weight.
    Eigen::MatrixXd Cocc(nao, nocc_act), Cvir(nao, nvir);
    for (int mu = 0; mu < nao; ++mu) {
        for (int i = 0; i < nocc_act; ++i) Cocc(mu, i) = (double)h_C[(size_t)mu * nao + (nfz + i)];
        for (int a = 0; a < nvir; ++a)     Cvir(mu, a) = (double)h_C[(size_t)mu * nao + (full_occ + a)];
    }
    const Eigen::MatrixXd Cocc_lo = Shalf * Cocc;   // nao × nocc_act
    const Eigen::MatrixXd Cvir_lo = Shalf * Cvir;   // nao × nvir

    // build_nto(occ,k): active NTO k in the Löwdin AO basis (unit norm). Mirrors
    // dmet.cu build_nto (1137-1149): occ → U_occ over the active-occupied block,
    // else U_vir over the virtual block.
    auto build_nto = [&](bool occ, int k) -> Eigen::VectorXd {
        if (occ) {
            Eigen::VectorXd u(nocc_act);
            for (int i = 0; i < nocc_act; ++i) u[i] = (double)nto.U_occ[(size_t)i * nocc_act + k];
            return Cocc_lo * u;
        } else {
            Eigen::VectorXd u(nvir);
            for (int a = 0; a < nvir; ++a) u[a] = (double)nto.U_vir[(size_t)a * nto.nvir + k];
            return Cvir_lo * u;
        }
    };

    // ---- Per-atom score: occupation-weighted hole+particle NTO population ---
    const auto& a2b = rhf.get_atom_to_basis_range();
    std::vector<double> score(num_atoms, 0.0);
    std::vector<int>    ao_count(num_atoms, 0);
    for (int A = 0; A < num_atoms; ++A)
        ao_count[A] = (int)(a2b[A].end_index - a2b[A].start_index);

    auto accumulate = [&](bool occ, int n_act, const std::vector<real_t>& occs) {
        for (int k = 0; k < n_act; ++k) {
            const double w = (k < (int)occs.size()) ? std::max(0.0, (double)occs[k]) : 0.0;
            if (w <= 0.0) continue;
            const Eigen::VectorXd phi = build_nto(occ, k);
            for (int A = 0; A < num_atoms; ++A) {
                double s = 0.0;
                for (size_t mu = a2b[A].start_index; mu < a2b[A].end_index; ++mu)
                    s += phi[(int)mu] * phi[(int)mu];
                score[A] += w * s;
            }
        }
    };
    accumulate(false, nto.n_act_vir, nto.nto_vir_occupations);  // particle
    accumulate(true,  nto.n_act_occ, nto.nto_occ_occupations);  // hole

    const double total = std::accumulate(score.begin(), score.end(), 0.0);
    if (total <= 1e-12) return fallback_whole("all per-atom NTO weights ≈ 0");
    for (double& s : score) s /= total;   // normalize so Σ_A score = 1

    // ---- Greedy selection + budget guard ----------------------------------
    Selection sel = greedy_select(score, ao_count, floor, T, budget);
    if (sel.atoms.empty())
        return fallback_whole("no atom above the per-atom floor");

    // ---- Optional H attachment (default off; env H σ is covered by the bath) -
    const auto& atoms = rhf.get_atoms();
    if (include_h) {
        std::set<int> selset(sel.atoms.begin(), sel.atoms.end());
        const real_t r2_max = real_t(2.6) * real_t(2.6);
        for (int heavy : sel.atoms) {
            if (atoms[heavy].atomic_number <= 1) continue;
            const auto& Xc = atoms[heavy].coordinate;
            for (int h = 0; h < num_atoms; ++h) {
                if (atoms[h].atomic_number != 1 || selset.count(h)) continue;
                const auto& Hc = atoms[h].coordinate;
                const real_t dx = Hc.x - Xc.x, dy = Hc.y - Xc.y, dz = Hc.z - Xc.z;
                if (dx*dx + dy*dy + dz*dz < r2_max) selset.insert(h);
            }
        }
        sel.atoms.assign(selset.begin(), selset.end());
        std::sort(sel.atoms.begin(), sel.atoms.end());
    }

    out.atoms       = sel.atoms;
    out.coverage    = sel.coverage;
    out.budget_hit  = sel.budget_hit;
    out.delocalized = sel.delocalized;
    out.n_components = connected_components(sel.atoms, atoms);

    // ---- Diagnostics (numerically inert) ----------------------------------
    {
        std::set<int> selset(sel.atoms.begin(), sel.atoms.end());
        std::vector<int> order(num_atoms);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b){ return score[a] > score[b]; });
        std::cout << "  per-atom NTO weight (occupation-weighted hole+particle, normalized):\n"
                  << "     rank  atom  elem     score     cumsum   sel" << std::endl;
        double cum = 0.0;
        const int show = std::min(num_atoms, 24);   // top rows only (large systems)
        for (int r = 0; r < show; ++r) {
            const int A = order[r];
            cum += score[A];
            std::cout << "     " << std::setw(4) << r
                      << "  " << std::setw(4) << A
                      << "  " << std::setw(4) << atomic_number_to_element_name(atoms[A].atomic_number)
                      << "   " << std::fixed << std::setprecision(5) << std::setw(9) << score[A]
                      << "   " << std::setw(8) << cum
                      << "   " << (selset.count(A) ? "*" : " ") << std::defaultfloat << std::endl;
        }
        if (show < num_atoms)
            std::cout << "     ... (" << (num_atoms - show) << " more atoms below)" << std::endl;

        std::cout << "  selected " << sel.atoms.size() << " atom(s) (coverage="
                  << std::fixed << std::setprecision(4) << sel.coverage << std::defaultfloat << "):";
        for (int A : sel.atoms) std::cout << " " << A << atomic_number_to_element_name(atoms[A].atomic_number);
        std::cout << std::endl;

        // Sensitivity: reselect at f/2 and 2f (no CIS recompute) — set change warns.
        auto set_of = [](const Selection& s){ return std::set<int>(s.atoms.begin(), s.atoms.end()); };
        const std::set<int> base_set = set_of(sel);
        const std::set<int> lo = set_of(greedy_select(score, ao_count, floor * 0.5, T, budget));
        const std::set<int> hi = set_of(greedy_select(score, ao_count, floor * 2.0, T, budget));
        if (lo != base_set || hi != base_set)
            std::cout << "  [auto-frag] WARNING: selection is floor-sensitive "
                         "(atom set changes at f/2 or 2f) — knee is soft; verify the fragment."
                      << std::endl;

        if (out.budget_hit)
            std::cout << "  [auto-frag] WARNING: cluster orbital budget (" << budget
                      << ") reached before coverage " << T
                      << " — this excitation may not be embeddable within budget "
                         "(coverage stopped at " << std::fixed << std::setprecision(4)
                      << sel.coverage << std::defaultfloat << ")." << std::endl;
        if (out.delocalized)
            std::cout << "  [auto-frag] WARNING: above-floor atoms cover only "
                      << std::fixed << std::setprecision(4) << sel.coverage << std::defaultfloat
                      << " < target " << T << " — excitation is delocalized/diffuse "
                         "(DMET localization assumption weak; expect degeneration toward full STEOM)."
                      << std::endl;
        if (out.n_components > 3)
            std::cout << "  [auto-frag] WARNING: selected atoms span " << out.n_components
                      << " disconnected regions (>3) — likely a mix of spatially distinct "
                         "excitations; consider per-state grouping (Phase C)." << std::endl;
        else if (out.n_components > 1)
            std::cout << "  [auto-frag] note: selected atoms span " << out.n_components
                      << " regions (e.g. donor…acceptor) — expected for charge-transfer." << std::endl;
    }

    // ---- Write the selected chromophore fragment as .xyz (optional) --------
    // Paper / visualization: the auto-extracted fragment geometry in Ångström.
    const std::string& xyz_path = rhf.get_dmet_steom_auto_xyz();
    if (!xyz_path.empty()) {
        const double bohr2ang = 1.0 / 1.8897259886;
        std::ofstream fx(xyz_path);
        fx << out.atoms.size() << "\n"
           << "DMET-STEOM auto-selected chromophore fragment (coverage="
           << std::fixed << std::setprecision(4) << out.coverage << ", "
           << out.atoms.size() << " atoms)\n";
        for (int A : out.atoms) {
            const auto& c = atoms[A].coordinate;
            fx << std::setw(2) << atomic_number_to_element_name(atoms[A].atomic_number)
               << " " << std::setw(14) << std::setprecision(8) << c.x * bohr2ang
               << " " << std::setw(14) << c.y * bohr2ang
               << " " << std::setw(14) << c.z * bohr2ang << "\n";
        }
        fx.close();
        std::cout << "  [auto-frag] wrote selected fragment geometry → " << xyz_path
                  << " (" << out.atoms.size() << " atoms)." << std::endl;
    }

    // ---- Phase C: per-state per-atom localization JSON (optional) ----------
    // For each CIS root n, s_A^(n) = Σ_ia |C^n_ia|² (pop_occ_A(i) + pop_vir_A(a))
    // — where the excitation lives per atom, per state. An external driver reads
    // this to cluster states by cosine similarity and split them into separate
    // fragment jobs (the dox case where the state-average scatters over regions).
    const std::string& json_path = rhf.get_dmet_steom_auto_json();
    if (!json_path.empty()) {
        const std::vector<real_t>& amp = rhf.get_last_cis_amplitudes();
        const size_t per = (size_t)nocc_act * nvir;
        const int n_have = (per > 0) ? (int)(amp.size() / per) : 0;
        if (n_have <= 0) {
            std::cout << "  [auto-frag] dmet_steom_auto_json set but no stashed CIS amplitudes "
                         "(non-RI path?) → skipping JSON." << std::endl;
        } else {
            // Per-atom Löwdin populations of the active-occ / virtual MOs.
            std::vector<std::vector<double>> pop_occ(num_atoms, std::vector<double>(nocc_act, 0.0));
            std::vector<std::vector<double>> pop_vir(num_atoms, std::vector<double>(nvir, 0.0));
            for (int A = 0; A < num_atoms; ++A) {
                for (int i = 0; i < nocc_act; ++i) {
                    double s = 0.0;
                    for (size_t mu = a2b[A].start_index; mu < a2b[A].end_index; ++mu) s += Cocc_lo((int)mu, i) * Cocc_lo((int)mu, i);
                    pop_occ[A][i] = s;
                }
                for (int a = 0; a < nvir; ++a) {
                    double s = 0.0;
                    for (size_t mu = a2b[A].start_index; mu < a2b[A].end_index; ++mu) s += Cvir_lo((int)mu, a) * Cvir_lo((int)mu, a);
                    pop_vir[A][a] = s;
                }
            }
            std::ofstream js(json_path);
            js << "{\n  \"n_states\": " << n_have << ",\n  \"num_atoms\": " << num_atoms << ",\n";
            js << "  \"elements\": [";
            for (int A = 0; A < num_atoms; ++A) js << (A ? "," : "") << "\"" << atomic_number_to_element_name(atoms[A].atomic_number) << "\"";
            js << "],\n";
            const double bohr2ang = 1.0 / 1.8897259886;
            js << "  \"coords_angstrom\": [";
            for (int A = 0; A < num_atoms; ++A) {
                const auto& c = atoms[A].coordinate;
                js << (A ? "," : "") << "[" << std::fixed << std::setprecision(6)
                   << c.x * bohr2ang << "," << c.y * bohr2ang << "," << c.z * bohr2ang << "]";
            }
            js << "],\n  \"per_state\": [\n";
            for (int n = 0; n < n_have; ++n) {
                const real_t* C = amp.data() + (size_t)n * per;
                std::vector<double> hole(nocc_act, 0.0), part(nvir, 0.0);
                for (int i = 0; i < nocc_act; ++i)
                    for (int a = 0; a < nvir; ++a) {
                        const double c2 = (double)C[(size_t)i * nvir + a] * C[(size_t)i * nvir + a];
                        hole[i] += c2; part[a] += c2;
                    }
                // Separate hole (occupied) and particle (virtual) per-atom weight
                // so a charge-transfer state (hole on the donor, particle on the
                // acceptor) is visible as spatially disjoint hole/particle regions.
                std::vector<double> s_hole(num_atoms, 0.0), s_part(num_atoms, 0.0), s_atom(num_atoms, 0.0);
                double tot = 0.0;
                for (int A = 0; A < num_atoms; ++A) {
                    double sh = 0.0, sp = 0.0;
                    for (int i = 0; i < nocc_act; ++i) sh += hole[i] * pop_occ[A][i];
                    for (int a = 0; a < nvir; ++a)     sp += part[a] * pop_vir[A][a];
                    s_hole[A] = sh; s_part[A] = sp; s_atom[A] = sh + sp; tot += sh + sp;
                }
                if (tot > 0.0) for (int A = 0; A < num_atoms; ++A) { s_hole[A] /= tot; s_part[A] /= tot; s_atom[A] /= tot; }
                js << "    {\"state\": " << n << ", \"atom_scores\": [";
                for (int A = 0; A < num_atoms; ++A) { js << (A ? "," : ""); js << std::fixed << std::setprecision(5) << s_atom[A]; }
                js << "], \"hole_scores\": [";
                for (int A = 0; A < num_atoms; ++A) { js << (A ? "," : ""); js << std::setprecision(5) << s_hole[A]; }
                js << "], \"part_scores\": [";
                for (int A = 0; A < num_atoms; ++A) { js << (A ? "," : ""); js << std::setprecision(5) << s_part[A]; }
                js << "]}" << (n + 1 < n_have ? "," : "") << "\n";
            }
            js << "  ]\n}\n";
            js.close();
            std::cout << "  [auto-frag] wrote per-state per-atom localization → " << json_path
                      << " (" << n_have << " states × " << num_atoms << " atoms) for external grouping (Phase C)." << std::endl;
        }
    }

    return out;
}

DMETBathGaugeResult dmet_steom_bath_gauge(
    RHF& rhf, const real_t* S_half, const real_t* h_C,
    int nao, int nocc, int num_atoms,
    const real_t* C_emb, int n_emb, const std::vector<char>& is_frag_ao)
{
    DMETBathGaugeResult g;
    g.atom_uncaptured.assign(num_atoms, 0.0);

    const CISNTOResult& nto = rhf.get_cis_nto_result();
    const int nocc_act = nto.nocc_active;
    const int nfz      = nto.num_frozen;
    const int nvir     = nto.nvir;
    const int full_occ = nocc;
    if (nocc_act <= 0 || (nto.n_act_occ <= 0 && nto.n_act_vir <= 0)) return g;

    // Embedding in the Löwdin basis (orthonormal columns): C_emb_lo = S^{1/2} C_emb.
    std::vector<double> C_emb_lo((size_t)nao * n_emb, 0.0);
    for (int mu = 0; mu < nao; ++mu)
        for (int p = 0; p < n_emb; ++p) {
            double v = 0.0;
            for (int nu = 0; nu < nao; ++nu)
                v += (double)S_half[(size_t)mu * nao + nu] * (double)C_emb[(size_t)nu * n_emb + p];
            C_emb_lo[(size_t)mu * n_emb + p] = v;
        }

    // Löwdin active-occ / virtual MO blocks (C_lo = S^{1/2} C).
    std::vector<double> C_lo_occ((size_t)nao * nocc_act, 0.0), C_lo_vir((size_t)nao * nvir, 0.0);
    for (int mu = 0; mu < nao; ++mu) {
        for (int i = 0; i < nocc_act; ++i) {
            double v = 0.0;
            for (int nu = 0; nu < nao; ++nu) v += (double)S_half[(size_t)mu * nao + nu] * (double)h_C[(size_t)nu * nao + (nfz + i)];
            C_lo_occ[(size_t)mu * nocc_act + i] = v;
        }
        for (int a = 0; a < nvir; ++a) {
            double v = 0.0;
            for (int nu = 0; nu < nao; ++nu) v += (double)S_half[(size_t)mu * nao + nu] * (double)h_C[(size_t)nu * nao + (full_occ + a)];
            C_lo_vir[(size_t)mu * nvir + a] = v;
        }
    }

    auto build_nto = [&](bool occ, int k) {
        std::vector<double> phi(nao, 0.0);
        if (occ) for (int mu = 0; mu < nao; ++mu) {
            double v = 0.0;
            for (int i = 0; i < nocc_act; ++i) v += C_lo_occ[(size_t)mu * nocc_act + i] * (double)nto.U_occ[(size_t)i * nocc_act + k];
            phi[mu] = v;
        } else for (int mu = 0; mu < nao; ++mu) {
            double v = 0.0;
            for (int a = 0; a < nvir; ++a) v += C_lo_vir[(size_t)mu * nvir + a] * (double)nto.U_vir[(size_t)a * nto.nvir + k];
            phi[mu] = v;
        }
        return phi;
    };

    // Per-NTO uncaptured = 1 − Σ_p ⟨col_p|φ⟩²; the residual (φ minus its cluster
    // projection) is the missing direction, attributed to environment atoms.
    const auto& a2b = rhf.get_atom_to_basis_range();
    auto scan = [&](bool occ, int n_act, const std::vector<real_t>& occs,
                    double& sumw, double& sumw_unc) {
        for (int k = 0; k < n_act; ++k) {
            const double w = (k < (int)occs.size()) ? std::max(0.0, (double)occs[k]) : 0.0;
            if (w <= 0.0) continue;
            std::vector<double> phi = build_nto(occ, k);
            // Cluster projection coefficients and captured fraction.
            double cap = 0.0;
            std::vector<double> res = phi;
            for (int p = 0; p < n_emb; ++p) {
                double d = 0.0;
                for (int mu = 0; mu < nao; ++mu) d += C_emb_lo[(size_t)mu * n_emb + p] * phi[mu];
                cap += d * d;
                for (int mu = 0; mu < nao; ++mu) res[mu] -= d * C_emb_lo[(size_t)mu * n_emb + p];
            }
            const double unc = std::max(0.0, 1.0 - cap);
            sumw += w; sumw_unc += w * unc;
            // Attribute the uncaptured residual to environment atoms.
            for (int A = 0; A < num_atoms; ++A) {
                double s = 0.0;
                for (size_t mu = a2b[A].start_index; mu < a2b[A].end_index; ++mu)
                    if (!is_frag_ao[(int)mu]) s += res[mu] * res[mu];
                g.atom_uncaptured[A] += w * s;
            }
        }
    };
    double sw_vir = 0.0, swu_vir = 0.0, sw_occ = 0.0, swu_occ = 0.0;
    scan(false, nto.n_act_vir, nto.nto_vir_occupations, sw_vir, swu_vir);
    scan(true,  nto.n_act_occ, nto.nto_occ_occupations, sw_occ, swu_occ);
    g.wunc_vir = sw_vir > 0.0 ? swu_vir / sw_vir : 0.0;
    g.wunc_occ = sw_occ > 0.0 ? swu_occ / sw_occ : 0.0;
    g.wunc     = std::max(g.wunc_vir, g.wunc_occ);   // η-aligned (worse side)
    g.verdict  = g.wunc < 0.02 ? "SUFFICIENT" : g.wunc < 0.10 ? "MARGINAL" : "INSUFFICIENT";

    // Virtual-space sufficiency: extend the particle scan past the active set to
    // occupation 1e-4 (the correlating tail). Capture-only (no atom attribution).
    const double ext_floor = 1e-4;
    int n_ext = 0;
    for (int a = 0; a < nvir && a < (int)nto.nto_vir_occupations.size(); ++a)
        if ((double)nto.nto_vir_occupations[a] > ext_floor) ++n_ext;
    double sw_ext = 0.0, swu_ext = 0.0;
    for (int k = 0; k < n_ext; ++k) {
        const double w = (double)nto.nto_vir_occupations[k];
        std::vector<double> phi = build_nto(false, k);
        double cap = 0.0;
        for (int p = 0; p < n_emb; ++p) {
            double d = 0.0;
            for (int mu = 0; mu < nao; ++mu) d += C_emb_lo[(size_t)mu * n_emb + p] * phi[mu];
            cap += d * d;
        }
        sw_ext += w; swu_ext += w * std::max(0.0, 1.0 - cap);
    }
    g.wunc_vir_ext = sw_ext > 0.0 ? swu_ext / sw_ext : 0.0;
    g.n_vir_ext    = n_ext;
    return g;
}

} // namespace gansu
