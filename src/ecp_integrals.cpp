/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ecp_integrals.cpp
 * @brief ECP integral evaluation (CPU, compiled with host compiler to avoid nvcc issues)
 *
 * Type 1 (local, n=2): 3-center overlap integral via Obara-Saika recursion
 * Type 2 (semi-local, same-center): analytical radial + Lebedev angular projection
 * Type 2 (semi-local, off-center): numerical GL-radial + product-angular quadrature
 */

#include "ecp_integrals.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace gansu {
namespace ecp_integral {

// ============================================================
//  Utility functions
// ============================================================
static double double_factorial(int n) {
    double result = 1.0;
    for (int i = n; i > 1; i -= 2) result *= i;
    return result;
}

double primitive_norm(double alpha, int l, int m, int n) {
    int L = l + m + n;
    return std::pow(2.0 / M_PI, 0.75)
         * std::pow(4.0 * alpha, L / 2.0)
         * std::pow(alpha, 0.75)
         / std::sqrt(double_factorial(2*l-1) * double_factorial(2*m-1) * double_factorial(2*n-1));
}

static double contracted_shell_norm(const PrimitiveShell* shells,
                                     const std::vector<size_t>& prim_indices,
                                     int lx, int ly, int lz) {
    double S = 0.0;
    for (size_t ip : prim_indices) {
        double Na = primitive_norm(shells[ip].exponent, lx, ly, lz);
        for (size_t jp : prim_indices) {
            double Nb = primitive_norm(shells[jp].exponent, lx, ly, lz);
            double p = shells[ip].exponent + shells[jp].exponent;
            double S_prim = std::pow(M_PI / p, 1.5);
            double E_prod = 1.0;
            for (int k = 0; k < lx; k++) E_prod *= (2.0*k+1) / (2.0*p);
            for (int k = 0; k < ly; k++) E_prod *= (2.0*k+1) / (2.0*p);
            for (int k = 0; k < lz; k++) E_prod *= (2.0*k+1) / (2.0*p);
            S += shells[ip].coefficient * shells[jp].coefficient * Na * Nb * S_prim * E_prod;
        }
    }
    return (S > 0) ? 1.0 / std::sqrt(S) : 1.0;
}

// ============================================================
//  Gauss-Legendre quadrature
// ============================================================
static void gauss_legendre_nodes(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n); weights.resize(n);
    for (int i = 0; i < (n+1)/2; i++) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
        double pp, z1;
        do {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 0; j < n; j++) { double p3 = p2; p2 = p1; p1 = ((2.0*j+1)*z*p2 - j*p3)/(j+1); }
            pp = n * (z*p1 - p2) / (z*z - 1.0);
            z1 = z; z -= p1/pp;
        } while (std::abs(z-z1) > 1e-15);
        nodes[i] = -z;     weights[i] = 2.0/((1-z*z)*pp*pp);
        nodes[n-1-i] = z;  weights[n-1-i] = weights[i];
    }
}

// ============================================================
//  Real solid harmonics
// ============================================================
static std::vector<double> solid_harmonics(int l, double x, double y, double z) {
    if (l == 0) return {1.0 / std::sqrt(4*M_PI)};
    if (l == 1) return {std::sqrt(3.0/(4*M_PI))*y, std::sqrt(3.0/(4*M_PI))*z, std::sqrt(3.0/(4*M_PI))*x};
    if (l == 2) {
        double f = std::sqrt(15.0/(4*M_PI));
        double f0 = std::sqrt(5.0/(16*M_PI));
        return {f*x*y, f*y*z, f0*(3*z*z-1), f*x*z, 0.5*f*(x*x-y*y)};
    }
    if (l == 3) {
        std::vector<double> S(7, 0.0);
        S[0] = std::sqrt(35/(32*M_PI))*y*(3*x*x-y*y);
        S[1] = std::sqrt(105/(4*M_PI))*x*y*z;
        S[2] = std::sqrt(21/(32*M_PI))*y*(5*z*z-1);
        S[3] = std::sqrt(7/(16*M_PI))*z*(5*z*z-3);
        S[4] = std::sqrt(21/(32*M_PI))*x*(5*z*z-1);
        S[5] = std::sqrt(105/(16*M_PI))*z*(x*x-y*y);
        S[6] = std::sqrt(35/(32*M_PI))*x*(x*x-3*y*y);
        return S;
    }
    return std::vector<double>(2*l+1, 0.0);
}

// ============================================================
//  Angular grids
// ============================================================
struct AngPt { double x, y, z, w; };

static std::vector<AngPt> lebedev_26() {
    std::vector<AngPt> pts;
    double w1 = 4.0*M_PI * 1.0/21.0;
    double w2 = 4.0*M_PI * 4.0/105.0;
    double w3 = 4.0*M_PI * 27.0/840.0;
    double a = 1.0/std::sqrt(2.0), b = 1.0/std::sqrt(3.0);
    for (int s : {-1,1}) { pts.push_back({(double)s,0,0,w1}); pts.push_back({0,(double)s,0,w1}); pts.push_back({0,0,(double)s,w1}); }
    for (int s1 : {-1,1}) for (int s2 : {-1,1}) {
        pts.push_back({s1*a, s2*a, 0, w2}); pts.push_back({s1*a, 0, s2*a, w2}); pts.push_back({0, s1*a, s2*a, w2});
    }
    for (int s1 : {-1,1}) for (int s2 : {-1,1}) for (int s3 : {-1,1})
        pts.push_back({s1*b, s2*b, s3*b, w3});
    return pts;
}

static std::vector<AngPt> product_angular_grid(int n_theta, int n_phi) {
    std::vector<double> ct_nodes, ct_weights;
    gauss_legendre_nodes(n_theta, ct_nodes, ct_weights);
    std::vector<AngPt> grid;
    grid.reserve(n_theta * n_phi);
    double dphi = 2.0 * M_PI / n_phi;
    for (int i = 0; i < n_theta; i++) {
        double cos_theta = ct_nodes[i];
        double s2 = 1.0 - cos_theta * cos_theta;
        double sin_theta = std::sqrt(s2 > 0.0 ? s2 : 0.0);
        for (int j = 0; j < n_phi; j++) {
            double phi = dphi * (j + 0.5);
            grid.push_back({sin_theta * std::cos(phi),
                           sin_theta * std::sin(phi),
                           cos_theta,
                           ct_weights[i] * dphi});
        }
    }
    return grid;
}

// ============================================================
//  Cartesian angular components for shell type l
// ============================================================
static std::vector<std::array<int,3>> cartesian_components(int l) {
    if (l == 0) return {{0,0,0}};
    if (l == 1) return {{1,0,0}, {0,1,0}, {0,0,1}};
    if (l == 2) return {{2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {1,0,1}, {0,1,1}};
    if (l == 3) return {{3,0,0}, {0,3,0}, {0,0,3}, {1,2,0}, {2,1,0}, {2,0,1}, {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1}};
    std::vector<std::array<int,3>> comps;
    for (int lx = l; lx >= 0; lx--)
        for (int ly = l-lx; ly >= 0; ly--)
            comps.push_back({lx, ly, l-lx-ly});
    return comps;
}

// ============================================================
//  Boys function
// ============================================================
static double boys_function(int n, double T) {
    if (T < 1e-15) return 1.0 / (2.0 * n + 1.0);
    if (T > 30.0 + n) {
        double val = 1.0;
        for (int i = 1; i <= n; i++) val *= (2*i - 1);
        return val * 0.5 * std::sqrt(M_PI / std::pow(T, 2*n+1));
    }
    int max_n = n + 20;
    double term = 1.0 / (2.0 * max_n + 1.0);
    double sum = term;
    for (int k = 1; k < 100; k++) {
        term *= T / (max_n + k + 0.5);
        sum += term;
        if (std::abs(term) < 1e-16 * std::abs(sum)) break;
    }
    double F = std::exp(-T) * sum;
    double expT = std::exp(-T);
    for (int i = max_n - 1; i >= n; i--)
        F = (2.0 * T * F + expT) / (2.0 * i + 1.0);
    return F;
}

// ============================================================
//  McMurchie-Davidson E coefficients
// ============================================================
static double E_coeff(int i, int j, int t, double p, double XPA, double XPB) {
    if (t < 0 || t > i + j) return 0.0;
    if (i == 0 && j == 0 && t == 0) return 1.0;
    if (i > 0) {
        return (1.0/(2.0*p)) * E_coeff(i-1, j, t-1, p, XPA, XPB)
             + XPA * E_coeff(i-1, j, t, p, XPA, XPB)
             + (t+1) * E_coeff(i-1, j, t+1, p, XPA, XPB);
    }
    return (1.0/(2.0*p)) * E_coeff(i, j-1, t-1, p, XPA, XPB)
         + XPB * E_coeff(i, j-1, t, p, XPA, XPB)
         + (t+1) * E_coeff(i, j-1, t+1, p, XPA, XPB);
}

// ============================================================
//  Type 1 ECP integrals (local potential)
// ============================================================
static double type1_n2_primitive(
    double alpha, const Coordinate& A, int la, int ma, int na_,
    double beta, const Coordinate& B, int lb, int mb, int nb_,
    double zeta, const Coordinate& C, double d_coeff)
{
    double p = alpha + beta;
    double Px = (alpha*A.x + beta*B.x) / p;
    double Py = (alpha*A.y + beta*B.y) / p;
    double Pz = (alpha*A.z + beta*B.z) / p;
    double gamma = p + zeta;
    double Qx = (p*Px + zeta*C.x) / gamma;
    double Qy = (p*Py + zeta*C.y) / gamma;
    double Qz = (p*Pz + zeta*C.z) / gamma;
    double AB2 = (A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z);
    double PC2 = (Px-C.x)*(Px-C.x) + (Py-C.y)*(Py-C.y) + (Pz-C.z)*(Pz-C.z);
    double K_AB = std::exp(-alpha*beta/p * AB2);
    double K_PC = std::exp(-p*zeta/gamma * PC2);
    double QAx = Qx-A.x, QAy = Qy-A.y, QAz = Qz-A.z;
    double QBx = Qx-B.x, QBy = Qy-B.y, QBz = Qz-B.z;
    double Ex = E_coeff(la, lb, 0, gamma, QAx, QBx);
    double Ey = E_coeff(ma, mb, 0, gamma, QAy, QBy);
    double Ez = E_coeff(na_, nb_, 0, gamma, QAz, QBz);
    return d_coeff * K_AB * K_PC * std::pow(M_PI / gamma, 1.5) * Ex * Ey * Ez;
}

// ============================================================
//  Contracted shell info
// ============================================================
struct ContractedShellInfo {
    int shell_type;
    size_t basis_start;
    int num_basis_funcs;
    Coordinate center;
    std::vector<size_t> prim_indices;
};

static std::vector<ContractedShellInfo> build_contracted_shells(
    const PrimitiveShell* shells, int num_primitives)
{
    std::vector<ContractedShellInfo> contracted;
    std::map<std::pair<size_t,int>, size_t> key_to_idx;
    for (int i = 0; i < num_primitives; i++) {
        auto key = std::make_pair(shells[i].basis_index, shells[i].shell_type);
        auto it = key_to_idx.find(key);
        if (it == key_to_idx.end()) {
            key_to_idx[key] = contracted.size();
            ContractedShellInfo cs;
            cs.shell_type = shells[i].shell_type;
            cs.basis_start = shells[i].basis_index;
            cs.num_basis_funcs = shell_type_to_num_basis(shells[i].shell_type);
            cs.center = shells[i].coordinate;
            cs.prim_indices.push_back(i);
            contracted.push_back(cs);
        } else {
            contracted[it->second].prim_indices.push_back(i);
        }
    }
    return contracted;
}

// ============================================================
//  Main ECP matrix computation
// ============================================================
void compute_ecp_matrix(
    const PrimitiveShell* shells, int num_primitives,
    const double* cgto_norms,
    int num_basis,
    const Atom* atoms, int num_atoms,
    const std::unordered_map<std::string, ElementECP>& ecp_data,
    double* V_ecp,
    int /*n_radial*/, int /*angular_order*/)
{
    auto contracted = build_contracted_shells(shells, num_primitives);
    std::map<int, std::vector<std::array<int,3>>> cart_map;
    for (const auto& cs : contracted) {
        if (cart_map.find(cs.shell_type) == cart_map.end())
            cart_map[cs.shell_type] = cartesian_components(cs.shell_type);
    }

    for (int iatom = 0; iatom < num_atoms; iatom++) {
        std::string elem = atomic_number_to_element_name(atoms[iatom].atomic_number);
        auto it = ecp_data.find(elem);
        if (it == ecp_data.end()) continue;

        const ElementECP& ecp = it->second;
        const Coordinate& C = atoms[iatom].coordinate;

        // ---- Type 1: local component ----
        auto compute_type1 = [&](const ECPComponent& comp) {
            for (const auto& ecp_prim : comp.primitives) {
                for (size_t si = 0; si < contracted.size(); si++) {
                    const auto& cs_a = contracted[si];
                    const auto& comps_a = cart_map[cs_a.shell_type];
                    for (size_t sj = si; sj < contracted.size(); sj++) {
                        const auto& cs_b = contracted[sj];
                        const auto& comps_b = cart_map[cs_b.shell_type];
                        for (int ca = 0; ca < cs_a.num_basis_funcs; ca++) {
                            int mu = cs_a.basis_start + ca;
                            int la = comps_a[ca][0], ma = comps_a[ca][1], na = comps_a[ca][2];
                            for (int cb = 0; cb < cs_b.num_basis_funcs; cb++) {
                                int nu = cs_b.basis_start + cb;
                                if (si == sj && nu < mu) continue;
                                int lb = comps_b[cb][0], mb = comps_b[cb][1], nb = comps_b[cb][2];
                                double val = 0.0;
                                for (size_t ip : cs_a.prim_indices) {
                                    for (size_t jp : cs_b.prim_indices) {
                                        double prim_val = 0.0;
                                        if (ecp_prim.power == 2) {
                                            prim_val = type1_n2_primitive(
                                                shells[ip].exponent, cs_a.center, la, ma, na,
                                                shells[jp].exponent, cs_b.center, lb, mb, nb,
                                                ecp_prim.exponent, C, ecp_prim.coefficient);
                                        }
                                        double Na = primitive_norm(shells[ip].exponent, la, ma, na);
                                        double Nb = primitive_norm(shells[jp].exponent, lb, mb, nb);
                                        val += shells[ip].coefficient * shells[jp].coefficient * Na * Nb * prim_val;
                                    }
                                }
                                val *= cgto_norms[mu] * cgto_norms[nu];
                                V_ecp[mu * num_basis + nu] += val;
                                if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                            }
                        }
                    }
                }
            }
        };
        compute_type1(ecp.get_local());

        // ---- Type 2: semi-local components ----
        {
            auto ang_grid = lebedev_26();
            int n_ang = ang_grid.size();

            // Precompute off-center angular grid and Y_lm (reused across all shell pairs)
            auto off_ang = product_angular_grid(25, 50);
            int off_n_ang = (int)off_ang.size();
            const int off_n_rad = 80;
            std::vector<double> off_rn, off_rw;
            gauss_legendre_nodes(off_n_rad, off_rn, off_rw);

            // Precompute Y_lm for all angular points and all l_proj values
            int off_max_lproj = ecp.get_l_max(); // semi-local l = 0 .. l_max-1
            // Flat Y_lm storage: off_Ylm_flat[ia * stride + offset[l] + m]
            int off_stride = 0;
            std::vector<int> off_Ylm_offset(off_max_lproj, 0);
            for (int l = 0; l < off_max_lproj; l++) {
                off_Ylm_offset[l] = off_stride;
                off_stride += 2*l+1;
            }
            std::vector<double> off_Ylm_flat(off_n_ang * off_stride, 0.0);
            for (int ia = 0; ia < off_n_ang; ia++) {
                for (int l = 0; l < off_max_lproj; l++) {
                    auto Ylm = solid_harmonics(l, off_ang[ia].x, off_ang[ia].y, off_ang[ia].z);
                    for (int m = 0; m < 2*l+1; m++)
                        off_Ylm_flat[ia * off_stride + off_Ylm_offset[l] + m] = Ylm[m];
                }
            }

            auto compute_proj_matrix_cross = [&](int La_, int Lb_, int l_proj) -> std::vector<double> {
                int nA = shell_type_to_num_basis(La_);
                int nB = shell_type_to_num_basis(Lb_);
                std::vector<double> proj(nA * nB, 0.0);
                auto comps_A = cartesian_components(La_);
                auto comps_B = cartesian_components(Lb_);
                int n_m = 2 * l_proj + 1;
                std::vector<std::vector<double>> C_a(nA, std::vector<double>(n_m, 0.0));
                std::vector<std::vector<double>> C_b(nB, std::vector<double>(n_m, 0.0));
                for (int ia = 0; ia < n_ang; ia++) {
                    double ox = ang_grid[ia].x, oy = ang_grid[ia].y, oz = ang_grid[ia].z;
                    double w = ang_grid[ia].w;
                    auto Ylm = solid_harmonics(l_proj, ox, oy, oz);
                    for (int a = 0; a < nA; a++) {
                        double Sa = 1.0;
                        for (int k = 0; k < comps_A[a][0]; k++) Sa *= ox;
                        for (int k = 0; k < comps_A[a][1]; k++) Sa *= oy;
                        for (int k = 0; k < comps_A[a][2]; k++) Sa *= oz;
                        for (int m = 0; m < n_m; m++) C_a[a][m] += w * Sa * Ylm[m];
                    }
                    for (int b = 0; b < nB; b++) {
                        double Sb = 1.0;
                        for (int k = 0; k < comps_B[b][0]; k++) Sb *= ox;
                        for (int k = 0; k < comps_B[b][1]; k++) Sb *= oy;
                        for (int k = 0; k < comps_B[b][2]; k++) Sb *= oz;
                        for (int m = 0; m < n_m; m++) C_b[b][m] += w * Sb * Ylm[m];
                    }
                }
                for (int a = 0; a < nA; a++)
                    for (int b = 0; b < nB; b++)
                        for (int m = 0; m < n_m; m++)
                            proj[a * nB + b] += C_a[a][m] * C_b[b][m];
                return proj;
            };

            for (size_t il = 0; il < ecp.num_semilocal(); il++) {
                const auto& sl_comp = ecp.get_semilocal()[il];
                int l_proj = sl_comp.angular_momentum;

                for (const auto& ecp_prim : sl_comp.primitives) {
                    if (ecp_prim.power != 2) continue;
                    double zeta = ecp_prim.exponent;
                    double d_ecp = ecp_prim.coefficient;

                    for (size_t si = 0; si < contracted.size(); si++) {
                        const auto& cs_a = contracted[si];
                        const auto& comps_a = cart_map[cs_a.shell_type];
                        int La = cs_a.shell_type;

                        for (size_t sj = si; sj < contracted.size(); sj++) {
                            const auto& cs_b = contracted[sj];
                            const auto& comps_b = cart_map[cs_b.shell_type];
                            int Lb = cs_b.shell_type;

                            double Da = std::sqrt(
                                (cs_a.center.x-C.x)*(cs_a.center.x-C.x) +
                                (cs_a.center.y-C.y)*(cs_a.center.y-C.y) +
                                (cs_a.center.z-C.z)*(cs_a.center.z-C.z));
                            double Db = std::sqrt(
                                (cs_b.center.x-C.x)*(cs_b.center.x-C.x) +
                                (cs_b.center.y-C.y)*(cs_b.center.y-C.y) +
                                (cs_b.center.z-C.z)*(cs_b.center.z-C.z));

                            // ---- Same-center: analytical ----
                            if (Da < 1e-10 && Db < 1e-10) {
                                if (l_proj > La || l_proj > Lb) continue;
                                if ((La - l_proj) % 2 != 0) continue;
                                if ((Lb - l_proj) % 2 != 0) continue;

                                auto proj = compute_proj_matrix_cross(La, Lb, l_proj);
                                int nA = cs_a.num_basis_funcs;
                                int nB = cs_b.num_basis_funcs;
                                int n_rad = 1 + (La + Lb) / 2;
                                auto gamma_half_int = [](int n_) -> double {
                                    double val = std::sqrt(M_PI);
                                    for (int k = 1; k <= n_; k++) val *= (2.0*k - 1.0) / 2.0;
                                    return val;
                                };

                                for (int ca = 0; ca < nA; ca++) {
                                    int mu = cs_a.basis_start + ca;
                                    int la = comps_a[ca][0], ma_c = comps_a[ca][1], na_c = comps_a[ca][2];
                                    for (int cb = 0; cb < nB; cb++) {
                                        int nu = cs_b.basis_start + cb;
                                        if (si == sj && nu < mu) continue;
                                        int lb = comps_b[cb][0], mb_c = comps_b[cb][1], nb_c = comps_b[cb][2];
                                        double P_ab = proj[ca * nB + cb];
                                        if (std::abs(P_ab) < 1e-15) continue;
                                        double val = 0.0;
                                        for (size_t ip : cs_a.prim_indices) {
                                            for (size_t jp : cs_b.prim_indices) {
                                                double gamma = shells[ip].exponent + shells[jp].exponent + zeta;
                                                double rad = gamma_half_int(n_rad) / (2.0 * std::pow(gamma, n_rad + 0.5));
                                                double Na = primitive_norm(shells[ip].exponent, la, ma_c, na_c);
                                                double Nb = primitive_norm(shells[jp].exponent, lb, mb_c, nb_c);
                                                val += shells[ip].coefficient * shells[jp].coefficient * Na * Nb
                                                     * d_ecp * P_ab * rad;
                                            }
                                        }
                                        val *= cgto_norms[mu] * cgto_norms[nu];
                                        V_ecp[mu * num_basis + nu] += val;
                                        if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                                    }
                                }
                            }
                            // ---- Off-center: numerical quadrature ----
                            // Angular grid and Y_lm precomputed above; radial GL per pair
                            else if (Da > 1e-10 || Db > 1e-10) {
                                double dAx = cs_a.center.x-C.x, dAy = cs_a.center.y-C.y, dAz = cs_a.center.z-C.z;
                                double dBx = cs_b.center.x-C.x, dBy = cs_b.center.y-C.y, dBz = cs_b.center.z-C.z;

                                // Screening: skip if both shells decay too much at ECP center
                                double alpha_min_a = 1e30, alpha_min_b = 1e30;
                                for (size_t ip : cs_a.prim_indices)
                                    if (shells[ip].exponent < alpha_min_a) alpha_min_a = shells[ip].exponent;
                                for (size_t jp : cs_b.prim_indices)
                                    if (shells[jp].exponent < alpha_min_b) alpha_min_b = shells[jp].exponent;
                                double screen_a = std::exp(-alpha_min_a * Da * Da);
                                double screen_b = std::exp(-alpha_min_b * Db * Db);
                                if (screen_a * screen_b < 1e-20) continue;

                                int nA = cs_a.num_basis_funcs;
                                int nB = cs_b.num_basis_funcs;
                                int n_m = 2 * l_proj + 1;

                                // R_max based on ECP decay: beyond this range ECP factor ≈ 0
                                double zm = (zeta > 0.01) ? zeta : 0.01;
                                double R_max = std::sqrt(35.0 / zm);

                                for (int ca = 0; ca < nA; ca++) {
                                    int mu = cs_a.basis_start + ca;
                                    int la = comps_a[ca][0], ma_c = comps_a[ca][1], na_c = comps_a[ca][2];
                                    for (int cb = 0; cb < nB; cb++) {
                                        int nu = cs_b.basis_start + cb;
                                        if (si == sj && nu < mu) continue;
                                        int lb = comps_b[cb][0], mb_c = comps_b[cb][1], nb_c = comps_b[cb][2];

                                        double val = 0.0;
                                        for (int ir = 0; ir < off_n_rad; ir++) {
                                            double t = (off_rn[ir] + 1.0) / 2.0;
                                            double r = R_max * t;
                                            double wr = off_rw[ir] * R_max / 2.0;
                                            if (r < 1e-15) continue;

                                            double ecp_rad = std::pow(r, ecp_prim.power) * std::exp(-zeta * r * r);
                                            if (ecp_rad < 1e-30) continue;

                                            double Alm_mu[7] = {}, Alm_nu[7] = {};  // max 2*3+1=7

                                            for (int ia = 0; ia < off_n_ang; ia++) {
                                                double ox = off_ang[ia].x;
                                                double oy = off_ang[ia].y;
                                                double oz = off_ang[ia].z;
                                                double w_ang = off_ang[ia].w;

                                                double rx = r*ox - dAx, ry = r*oy - dAy, rz = r*oz - dAz;
                                                double r2A = rx*rx + ry*ry + rz*rz;
                                                double sx = r*ox - dBx, sy = r*oy - dBy, sz = r*oz - dBz;
                                                double r2B = sx*sx + sy*sy + sz*sz;

                                                double pow_mu = 1.0;
                                                for (int k = 0; k < la;   k++) pow_mu *= rx;
                                                for (int k = 0; k < ma_c; k++) pow_mu *= ry;
                                                for (int k = 0; k < na_c; k++) pow_mu *= rz;

                                                double pow_nu = 1.0;
                                                for (int k = 0; k < lb;   k++) pow_nu *= sx;
                                                for (int k = 0; k < mb_c; k++) pow_nu *= sy;
                                                for (int k = 0; k < nb_c; k++) pow_nu *= sz;

                                                double mu_val = 0.0;
                                                for (size_t ip : cs_a.prim_indices) {
                                                    double Na = primitive_norm(shells[ip].exponent, la, ma_c, na_c);
                                                    mu_val += shells[ip].coefficient * Na * pow_mu
                                                              * std::exp(-shells[ip].exponent * r2A);
                                                }
                                                double nu_val = 0.0;
                                                for (size_t jp : cs_b.prim_indices) {
                                                    double Nb = primitive_norm(shells[jp].exponent, lb, mb_c, nb_c);
                                                    nu_val += shells[jp].coefficient * Nb * pow_nu
                                                              * std::exp(-shells[jp].exponent * r2B);
                                                }

                                                const double* Ylm_ptr = &off_Ylm_flat[ia * off_stride + off_Ylm_offset[l_proj]];
                                                for (int m = 0; m < n_m; m++) {
                                                    Alm_mu[m] += w_ang * mu_val * Ylm_ptr[m];
                                                    Alm_nu[m] += w_ang * nu_val * Ylm_ptr[m];
                                                }
                                            }

                                            double ang_sum = 0.0;
                                            for (int m = 0; m < n_m; m++)
                                                ang_sum += Alm_mu[m] * Alm_nu[m];
                                            val += wr * ecp_rad * ang_sum;
                                        }

                                        val *= d_ecp * cgto_norms[mu] * cgto_norms[nu];
                                        V_ecp[mu * num_basis + nu] += val;
                                        if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        std::cout << "  ECP integrals for " << elem << " computed" << std::endl;
    }
}

} // namespace ecp_integral
} // namespace gansu
