#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
#include "basis_set.hpp"
#include "types.hpp"

namespace gansu {

// ============================================================
//  Atomic occupation numbers for MINAO initial guess
//  Returns occupation per basis function for given element.
//  Aufbau filling: 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, ...
//  Partially filled subshells get fractional occupations.
// ============================================================
inline std::vector<double> get_minao_occupations(int atomic_number, const ElementBasisSet& minao_basis) {
    // Count shells in MINAO basis: how many s, p, d contractions
    int n_s = 0, n_p = 0, n_d = 0;
    for (size_t i = 0; i < minao_basis.get_num_contracted_gausses(); i++) {
        const std::string& type = minao_basis.get_contracted_gauss(i).get_type();
        if (type == "S") n_s++;
        else if (type == "P") n_p++;
        else if (type == "D") n_d++;
    }

    // Subshell filling order and capacities
    // Each entry: (shell_type, max_electrons)
    // s=0 (cap 2), p=1 (cap 6), d=2 (cap 10)
    struct Subshell { int type; int capacity; };
    // Aufbau order of subshells (matching MINAO basis order: all s first, then p, then d)
    // MINAO basis has s-shells ordered as 1s, 2s, 3s, ...; p-shells as 2p, 3p, ...; d as 3d, ...
    // We fill in Aufbau order but map to the basis function index
    std::vector<Subshell> aufbau;
    // Standard Aufbau: 1s 2s 2p 3s 3p 4s 3d 4p 5s 4d 5p 6s 4f 5d 6p 7s ...
    // We need to map each to (s_index, p_index, d_index)
    // s_index: 0-based index into s-contractions; p_index: into p; d: into d
    struct AufbauEntry { int type; int sub_index; int capacity; };
    std::vector<AufbauEntry> filling;
    // Simplified: H-Kr (Z=1-36)
    //   1s  2s  2p  3s  3p  4s  3d  4p
    int si = 0, pi = 0, di = 0;
    auto add_s = [&]() { if (si < n_s) { filling.push_back({0, si++, 2}); } };
    auto add_p = [&]() { if (pi < n_p) { filling.push_back({1, pi++, 6}); } };
    auto add_d = [&]() { if (di < n_d) { filling.push_back({2, di++, 10}); } };

    // Standard Aufbau order for Z up to 86
    add_s();  // 1s
    add_s();  // 2s
    add_p();  // 2p
    add_s();  // 3s
    add_p();  // 3p
    add_s();  // 4s
    add_d();  // 3d
    add_p();  // 4p
    add_s();  // 5s
    add_d();  // 4d
    add_p();  // 5p
    add_s();  // 6s
    add_d();  // 5d  (if available)
    add_p();  // 6p

    // Total basis functions
    int n_basis = n_s * 1 + n_p * 3 + n_d * 6;
    std::vector<double> occ(n_basis, 0.0);

    // Compute basis function offset for each contraction
    // Order in basis: all s contractions, then p, then d (matching .gbs parse order)
    auto basis_offset = [&](int type, int sub_idx) -> int {
        if (type == 0) return sub_idx;  // s: 1 func each
        if (type == 1) return n_s + sub_idx * 3;  // p: 3 funcs each
        if (type == 2) return n_s + n_p * 3 + sub_idx * 6;  // d: 6 funcs each
        return 0;
    };

    int electrons_remaining = atomic_number;
    for (const auto& entry : filling) {
        if (electrons_remaining <= 0) break;
        int offset = basis_offset(entry.type, entry.sub_index);
        int degen = (entry.type == 0) ? 1 : (entry.type == 1) ? 3 : 6;
        int cap = entry.capacity;
        int to_fill = std::min(electrons_remaining, cap);
        double per_func = static_cast<double>(to_fill) / degen;
        for (int k = 0; k < degen; k++)
            occ[offset + k] = per_func;
        electrons_remaining -= to_fill;
    }

    return occ;
}

// ============================================================
//  Obara-Saika 1D overlap integral
//  S^{i,j} between Cartesian Gaussians centered at A and B
//  with exponents alpha and beta.
// ============================================================
namespace detail {

// Compute all S^{i,j}_x for i=0..la, j=0..lb using Obara-Saika recursion
inline void obara_saika_overlap_1d(
    int la, int lb, double alpha, double beta,
    double A, double B, std::vector<double>& S)
{
    double p = alpha + beta;
    double mu = alpha * beta / p;
    double P = (alpha * A + beta * B) / p;
    double PA = P - A;
    double PB = P - B;
    double pre = std::sqrt(M_PI / p) * std::exp(-mu * (A - B) * (A - B));

    int stride = lb + 1;
    S.assign((la + 1) * stride, 0.0);
    auto idx = [stride](int i, int j) { return i * stride + j; };

    S[idx(0, 0)] = pre;

    // Upward recursion in i (first index)
    for (int i = 1; i <= la; i++)
        S[idx(i, 0)] = PA * S[idx(i-1, 0)] + (i > 1 ? (i-1) / (2.0*p) * S[idx(i-2, 0)] : 0.0);

    // Upward recursion in j (second index)
    for (int j = 1; j <= lb; j++) {
        S[idx(0, j)] = PB * S[idx(0, j-1)] + (j > 1 ? (j-1) / (2.0*p) * S[idx(0, j-2)] : 0.0);
        for (int i = 1; i <= la; i++)
            S[idx(i, j)] = PB * S[idx(i, j-1)]
                         + i / (2.0*p) * S[idx(i-1, j-1)]
                         + (j > 1 ? (j-1) / (2.0*p) * S[idx(i, j-2)] : 0.0);
    }
}

// Angular momentum component table for Cartesian Gaussians (same as kernel's loop_to_ang)
inline void get_cartesian_components(int l, std::vector<std::array<int,3>>& comps) {
    comps.clear();
    for (int i = l; i >= 0; i--)
        for (int j = l - i; j >= 0; j--)
            comps.push_back({i, j, l - i - j});
}

} // namespace detail

// ============================================================
//  Cross-overlap matrix between two basis sets on the same atoms.
//  Returns row-major matrix of size n_calc × n_minao.
// ============================================================
inline std::vector<double> compute_cross_overlap(
    const std::vector<Atom>& atoms,
    const BasisSet& basis_calc,
    const BasisSet& basis_minao)
{
    // Build basis info for both sets
    struct BasisInfo {
        int n_basis;
        struct CGTO {
            int shell_type;
            int basis_offset;
            int atom_index;
            double ax, ay, az;  // center
            std::vector<double> exponents, coefficients;
            std::vector<double> norms;  // per angular component
        };
        std::vector<CGTO> cgtos;
    };

    auto build_info = [&](const BasisSet& bs) -> BasisInfo {
        BasisInfo info;
        int offset = 0;
        for (size_t ia = 0; ia < atoms.size(); ia++) {
            const std::string elem = atomic_number_to_element_name(atoms[ia].atomic_number);
            try { bs.get_element_basis_set(elem); } catch (...) { continue; }
            const auto& ebs = bs.get_element_basis_set(elem);
            for (size_t ic = 0; ic < ebs.get_num_contracted_gausses(); ic++) {
                const auto& cg = ebs.get_contracted_gauss(ic);
                BasisInfo::CGTO cgto;
                cgto.shell_type = shell_name_to_shell_type(cg.get_type());
                cgto.basis_offset = offset;
                cgto.atom_index = (int)ia;
                cgto.ax = atoms[ia].coordinate.x;
                cgto.ay = atoms[ia].coordinate.y;
                cgto.az = atoms[ia].coordinate.z;
                for (size_t jp = 0; jp < cg.get_num_primitives(); jp++) {
                    cgto.exponents.push_back(cg.get_primitive_gauss(jp).exponent);
                    cgto.coefficients.push_back(cg.get_primitive_gauss(jp).coefficient);
                }
                cgto.norms = cg.get_normalization_factor();
                offset += shell_type_to_num_basis(cgto.shell_type);
                info.cgtos.push_back(std::move(cgto));
            }
        }
        info.n_basis = offset;
        return info;
    };

    BasisInfo calc = build_info(basis_calc);
    BasisInfo minao = build_info(basis_minao);

    std::vector<double> S(calc.n_basis * minao.n_basis, 0.0);

    // For each pair of contracted Gaussians
    for (const auto& cg_a : calc.cgtos) {
        for (const auto& cg_b : minao.cgtos) {
            int la = cg_a.shell_type, lb = cg_b.shell_type;
            std::vector<std::array<int,3>> comps_a, comps_b;
            detail::get_cartesian_components(la, comps_a);
            detail::get_cartesian_components(lb, comps_b);

            // Loop over primitive pairs
            for (size_t ip = 0; ip < cg_a.exponents.size(); ip++) {
                double alpha = cg_a.exponents[ip];
                double ca = cg_a.coefficients[ip];
                for (size_t jp = 0; jp < cg_b.exponents.size(); jp++) {
                    double beta = cg_b.exponents[jp];
                    double cb = cg_b.coefficients[jp];

                    // 1D overlaps for each direction
                    std::vector<double> Sx, Sy, Sz;
                    detail::obara_saika_overlap_1d(la, lb, alpha, beta, cg_a.ax, cg_b.ax, Sx);
                    detail::obara_saika_overlap_1d(la, lb, alpha, beta, cg_a.ay, cg_b.ay, Sy);
                    detail::obara_saika_overlap_1d(la, lb, alpha, beta, cg_a.az, cg_b.az, Sz);

                    int stride = lb + 1;
                    // Loop over angular components
                    for (size_t ia_c = 0; ia_c < comps_a.size(); ia_c++) {
                        int lx_a = comps_a[ia_c][0], ly_a = comps_a[ia_c][1], lz_a = comps_a[ia_c][2];
                        for (size_t ib_c = 0; ib_c < comps_b.size(); ib_c++) {
                            int lx_b = comps_b[ib_c][0], ly_b = comps_b[ib_c][1], lz_b = comps_b[ib_c][2];
                            double s3d = Sx[lx_a * stride + lx_b]
                                       * Sy[ly_a * stride + ly_b]
                                       * Sz[lz_a * stride + lz_b];
                            int mu = cg_a.basis_offset + (int)ia_c;
                            int nu = cg_b.basis_offset + (int)ib_c;
                            S[mu * minao.n_basis + nu] += ca * cb
                                * cg_a.norms[ia_c] * cg_b.norms[ib_c] * s3d;
                        }
                    }
                }
            }
        }
    }

    return S;
}

} // namespace gansu
