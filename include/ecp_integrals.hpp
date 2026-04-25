/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ecp_integrals.hpp
 * @brief ECP integral evaluation (CPU, analytical for n=2 + Boys function for n=0,1)
 *
 * Type 1 (local, n=2): 3-center overlap integral via Obara-Saika recursion
 * Type 1 (local, n=0,1): modified nuclear attraction via Boys function
 * Type 2 (semi-local): angular projection via analytical radial integrals
 *
 * Key formula (Type 1, n=2):
 *   <mu_A|d*exp(-zeta*r_C^2)|nu_B> = d * K_AB * K_PC * (pi/gamma)^(3/2) * Hermite(...)
 *   where gamma = alpha + beta + zeta
 */

#pragma once

#include "ecp.hpp"
#include "types.hpp"
#include <cmath>
#include <vector>
#include <array>
#include <map>
#include <iostream>
#include <iomanip>

namespace gansu {
namespace ecp_integral {

// ============================================================
//  Primitive Gaussian normalization
//  N = (2/pi)^{3/4} * (4*alpha)^{L/2} * alpha^{3/4} / sqrt(df(2l-1)*df(2m-1)*df(2n-1))
// ============================================================
inline double double_factorial(int n) {
    double result = 1.0;
    for (int i = n; i > 1; i -= 2) result *= i;
    return result;
}

// Primitive Gaussian normalization (CPU convention, matches PySCF)
// N = (2/pi)^{3/4} * (4*alpha)^{L/2} * alpha^{3/4} / sqrt(df(2l-1)*df(2m-1)*df(2n-1))
inline double primitive_norm(double alpha, int l, int m, int n) {
    int L = l + m + n;
    return std::pow(2.0 / M_PI, 0.75)
         * std::pow(4.0 * alpha, L / 2.0)
         * std::pow(alpha, 0.75)
         / std::sqrt(double_factorial(2*l-1) * double_factorial(2*m-1) * double_factorial(2*n-1));
}

// Compute contracted shell normalization (self-contained, not using cgto_norms array)
// This avoids index mapping issues with GANSU's shell_type-sorted primitive array
inline double contracted_shell_norm(const PrimitiveShell* shells,
                                     const std::vector<size_t>& prim_indices,
                                     int lx, int ly, int lz) {
    int L = lx + ly + lz;
    double S = 0.0;
    for (size_t ip : prim_indices) {
        double Na = primitive_norm(shells[ip].exponent, lx, ly, lz);
        for (size_t jp : prim_indices) {
            double Nb = primitive_norm(shells[jp].exponent, lx, ly, lz);
            double p = shells[ip].exponent + shells[jp].exponent;
            // Overlap: (pi/p)^{3/2} * E(lx,lx,0) * E(ly,ly,0) * E(lz,lz,0)  [same center]
            // = (pi/p)^{3/2} * df(2lx-1)!!*df(2ly-1)!!*df(2lz-1)!! / (2p)^L
            double S_prim = std::pow(M_PI / p, 1.5);
            double E_prod = 1.0;
            // For same-center: E(l,l,0,p,0,0) = df(2l-1)!! / (2p)^l (closed form)
            for (int k = 0; k < lx; k++) E_prod *= (2.0*k+1) / (2.0*p);
            for (int k = 0; k < ly; k++) E_prod *= (2.0*k+1) / (2.0*p);
            for (int k = 0; k < lz; k++) E_prod *= (2.0*k+1) / (2.0*p);
            S += shells[ip].coefficient * shells[jp].coefficient * Na * Nb * S_prim * E_prod;
        }
    }
    return (S > 0) ? 1.0 / std::sqrt(S) : 1.0;
}

// ============================================================
//  Modified spherical Bessel function i_l(x)
//  i_l(x) = sqrt(pi/(2x)) * I_{l+1/2}(x) = sum_k x^(2k+l) / (k! (2k+2l+1)!!)
// ============================================================
inline double mod_sph_bessel(int l, double x) {
    if (x < 1e-15) return (l == 0) ? 1.0 : 0.0;

    if (x < 15.0) {
        // Series expansion: i_l(x) = x^l * sum_k (x^2)^k / (k! * prod_{j=1}^{k} (2l+2j+1))
        // = x^l / (2l+1)!! * sum_k (x^2/2)^k / (k! * (l+3/2)_k) ... actually simpler:
        // i_l(x) = sum_{k=0}^inf x^{2k+l} / (2^k k! (2k+2l+1)!!)
        double xl = 1.0;
        for (int i = 0; i < l; i++) xl *= x;
        double df = 1.0;
        for (int i = 1; i <= 2*l+1; i += 2) df *= i; // (2l+1)!!

        double term = xl / df;
        double sum = term;
        double x2 = x * x;
        for (int k = 1; k < 50; k++) {
            term *= x2 / (2.0 * k * (2*(k+l) + 1));
            sum += term;
            if (std::abs(term) < 1e-16 * std::abs(sum)) break;
        }
        return sum;
    }

    // Large x: i_l(x) ≈ exp(x)/(2x) (leading term)
    // Use recurrence: i_{l+1}(x) = i_{l-1}(x) - (2l+1)/x * i_l(x)
    double i0 = std::sinh(x) / x;     // i_0
    if (l == 0) return i0;
    double i1 = std::cosh(x)/x - std::sinh(x)/(x*x);  // i_1 = cosh(x)/x - sinh(x)/x^2
    if (l == 1) return i1;

    // Upward recurrence (stable for moderate x)
    double i_prev = i0, i_curr = i1;
    for (int n = 1; n < l; n++) {
        double i_next = i_prev - (2.0*n + 1.0) / x * i_curr;
        i_prev = i_curr;
        i_curr = i_next;
    }
    return i_curr;
}

// ============================================================
//  Gauss-Legendre quadrature for 1D radial integrals
// ============================================================
inline void gauss_legendre_nodes(int n, std::vector<double>& nodes, std::vector<double>& weights) {
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
//  Real solid harmonics S_lm(x, y, z) (unnormalized on unit sphere)
// ============================================================
inline std::vector<double> solid_harmonics(int l, double x, double y, double z) {
    if (l == 0) return {1.0 / std::sqrt(4*M_PI)};
    if (l == 1) return {std::sqrt(3.0/(4*M_PI))*y, std::sqrt(3.0/(4*M_PI))*z, std::sqrt(3.0/(4*M_PI))*x};
    if (l == 2) {
        double f = std::sqrt(15.0/(4*M_PI));
        double f0 = std::sqrt(5.0/(16*M_PI));
        return {f*x*y, f*y*z, f0*(3*z*z-1), f*x*z, 0.5*f*(x*x-y*y)};
    }
    if (l == 3) {
        // Simplified l=3 (7 components)
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
//  Lebedev 26-point angular quadrature (exact for l<=5)
// ============================================================
struct AngPt { double x, y, z, w; };

inline std::vector<AngPt> lebedev_26() {
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

// ============================================================
//  Cartesian angular components for shell type l
// ============================================================
inline std::vector<std::array<int,3>> cartesian_components(int l) {
    // Must match GANSU's AngularMomentums ordering in types.hpp
    if (l == 0) return {{0,0,0}};
    if (l == 1) return {{1,0,0}, {0,1,0}, {0,0,1}};
    if (l == 2) return {{2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {1,0,1}, {0,1,1}};
    if (l == 3) return {{3,0,0}, {0,3,0}, {0,0,3}, {1,2,0}, {2,1,0}, {2,0,1}, {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1}};
    // Generic fallback (not matching GANSU for l>=4)
    std::vector<std::array<int,3>> comps;
    for (int lx = l; lx >= 0; lx--)
        for (int ly = l-lx; ly >= 0; ly--)
            comps.push_back({lx, ly, l-lx-ly});
    return comps;
}

// ============================================================
//  Boys function F_n(T) = integral_0^1 t^(2n) exp(-T*t^2) dt
// ============================================================
inline double boys_function(int n, double T) {
    if (T < 1e-15) return 1.0 / (2.0 * n + 1.0);

    // Asymptotic formula for large T
    if (T > 30.0 + n) {
        double val = 1.0;
        for (int i = 1; i <= n; i++) val *= (2*i - 1);
        return val * 0.5 * std::sqrt(M_PI / std::pow(T, 2*n+1));
    }

    // Downward recursion from high n
    int max_n = n + 20;
    double F = 0.0;
    // Series expansion for F_{max_n}
    double term = 1.0 / (2.0 * max_n + 1.0);
    double sum = term;
    for (int k = 1; k < 100; k++) {
        term *= T / (max_n + k + 0.5);
        sum += term;
        if (std::abs(term) < 1e-16 * std::abs(sum)) break;
    }
    F = std::exp(-T) * sum;

    // Downward recursion: F_n = (2T F_{n+1} + exp(-T)) / (2n+1)
    double expT = std::exp(-T);
    for (int i = max_n - 1; i >= n; i--) {
        F = (2.0 * T * F + expT) / (2.0 * i + 1.0);
    }
    return F;
}

// ============================================================
//  McMurchie-Davidson E coefficients for Hermite expansion
//  E^{ij}_t(p, XPA, XPB) where p = alpha + beta
// ============================================================
inline double E_coeff(int i, int j, int t, double p, double XPA, double XPB) {
    if (t < 0 || t > i + j) return 0.0;
    if (i == 0 && j == 0 && t == 0) return 1.0;

    if (i > 0) {
        // Recursion on i
        return (1.0/(2.0*p)) * E_coeff(i-1, j, t-1, p, XPA, XPB)
             + XPA * E_coeff(i-1, j, t, p, XPA, XPB)
             + (t+1) * E_coeff(i-1, j, t+1, p, XPA, XPB);
    }
    // i == 0, j > 0
    return (1.0/(2.0*p)) * E_coeff(i, j-1, t-1, p, XPA, XPB)
         + XPB * E_coeff(i, j-1, t, p, XPA, XPB)
         + (t+1) * E_coeff(i, j-1, t+1, p, XPA, XPB);
}

// ============================================================
//  Hermite Coulomb integral R^n_{tuv}(p, RPC)
//  Used for n=0 and n=1 ECP terms (Boys function based)
// ============================================================
inline double R_hermite(int t, int u, int v, int n, double p, double RPC_x, double RPC_y, double RPC_z) {
    double T = p * (RPC_x*RPC_x + RPC_y*RPC_y + RPC_z*RPC_z);
    if (t == 0 && u == 0 && v == 0) {
        return std::pow(-2.0*p, n) * boys_function(n, T);
    }
    if (t > 0) {
        return (t-1)*R_hermite(t-2,u,v,n+1,p,RPC_x,RPC_y,RPC_z)
             + RPC_x*R_hermite(t-1,u,v,n+1,p,RPC_x,RPC_y,RPC_z);
    }
    if (u > 0) {
        return (u-1)*R_hermite(t,u-2,v,n+1,p,RPC_x,RPC_y,RPC_z)
             + RPC_y*R_hermite(t,u-1,v,n+1,p,RPC_x,RPC_y,RPC_z);
    }
    // v > 0
    return (v-1)*R_hermite(t,u,v-2,n+1,p,RPC_x,RPC_y,RPC_z)
         + RPC_z*R_hermite(t,u,v-1,n+1,p,RPC_x,RPC_y,RPC_z);
}

// ============================================================
//  Type 1 ECP integral: <mu_A|d * r^(n-2) * exp(-zeta*r_C^2)|nu_B>
//  For a single primitive pair and single ECP primitive
// ============================================================

// n=2 case: pure Gaussian → 3-center overlap
inline double type1_n2_primitive(
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

    // McMurchie-Davidson E coefficients with combined center Q and exponent gamma
    double QAx = Qx - A.x, QAy = Qy - A.y, QAz = Qz - A.z;
    double QBx = Qx - B.x, QBy = Qy - B.y, QBz = Qz - B.z;

    // For 3-center overlap (pure Gaussian), only t=u=v=0 Hermite component survives:
    // integral H_t exp(-gamma r^2) dr = (pi/gamma)^(1/2) only for t=0
    double Ex = E_coeff(la, lb, 0, gamma, QAx, QBx);
    double Ey = E_coeff(ma, mb, 0, gamma, QAy, QBy);
    double Ez = E_coeff(na_, nb_, 0, gamma, QAz, QBz);

    return d_coeff * K_AB * K_PC * std::pow(M_PI / gamma, 1.5) * Ex * Ey * Ez;
}

// n=0 case: r^(-2) * exp(-zeta*r^2) → related to Boys function
inline double type1_n0_primitive(
    double alpha, const Coordinate& A, int la, int ma, int na_,
    double beta, const Coordinate& B, int lb, int mb, int nb_,
    double zeta, const Coordinate& C, double d_coeff)
{
    double p = alpha + beta;
    double Px = (alpha*A.x + beta*B.x) / p;
    double Py = (alpha*A.y + beta*B.y) / p;
    double Pz = (alpha*A.z + beta*B.z) / p;

    double gamma = p + zeta;
    // Effective nuclear attraction with modified exponent
    double AB2 = (A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z);
    double K_AB = std::exp(-alpha*beta/p * AB2);

    double RPCx = Px - C.x, RPCy = Py - C.y, RPCz = Pz - C.z;

    // <mu|r_C^{-2} exp(-zeta r_C^2)|nu> = (2pi/p) K_AB sum E_t R^0_{tuv}(gamma, RPC)
    // where the R integrals use the modified exponent gamma = p + zeta
    // and the Boys function argument T = gamma * |Q-C|^2...
    // Actually this needs careful derivation. For now use approximate:
    // Treat as nuclear attraction with modified Boys: T = p*zeta/gamma * |P-C|^2
    double T = p * zeta / gamma * (RPCx*RPCx + RPCy*RPCy + RPCz*RPCz);

    double result = 0.0;
    for (int t = 0; t <= la + lb; t++)
        for (int u = 0; u <= ma + mb; u++)
            for (int v = 0; v <= na_ + nb_; v++) {
                double Ex = E_coeff(la, lb, t, p, Px-A.x, Px-B.x);
                double Ey = E_coeff(ma, mb, u, p, Py-A.y, Py-B.y);
                double Ez = E_coeff(na_, nb_, v, p, Pz-A.z, Pz-B.z);
                // Modified R integral with gamma instead of p for the Boys function
                double Rtuv = R_hermite(t, u, v, 0, gamma, RPCx*p/gamma, RPCy*p/gamma, RPCz*p/gamma);
                result += Ex * Ey * Ez * Rtuv;
            }

    return d_coeff * K_AB * (2.0 * M_PI / p) * result;
}

// n=1 case: r^(-1) * exp(-zeta*r^2)
inline double type1_n1_primitive(
    double alpha, const Coordinate& A, int la, int ma, int na_,
    double beta, const Coordinate& B, int lb, int mb, int nb_,
    double zeta, const Coordinate& C, double d_coeff)
{
    // Similar to nuclear attraction but with additional Gaussian damping
    // For now, approximate: for large zeta, this term is negligible
    // TODO: implement exact formula
    double p = alpha + beta;
    double AB2 = (A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z);
    double K_AB = std::exp(-alpha*beta/p * AB2);
    double Px = (alpha*A.x + beta*B.x) / p;
    double Py = (alpha*A.y + beta*B.y) / p;
    double Pz = (alpha*A.z + beta*B.z) / p;
    double gamma = p + zeta;

    double RPCx = Px-C.x, RPCy = Py-C.y, RPCz = Pz-C.z;

    double result = 0.0;
    for (int t = 0; t <= la + lb; t++)
        for (int u = 0; u <= ma + mb; u++)
            for (int v = 0; v <= na_ + nb_; v++) {
                double Ex = E_coeff(la, lb, t, p, Px-A.x, Px-B.x);
                double Ey = E_coeff(ma, mb, u, p, Py-A.y, Py-B.y);
                double Ez = E_coeff(na_, nb_, v, p, Pz-A.z, Pz-B.z);
                double Rtuv = R_hermite(t, u, v, 0, gamma, RPCx*p/gamma, RPCy*p/gamma, RPCz*p/gamma);
                result += Ex * Ey * Ez * Rtuv;
            }

    // Factor: 2*pi*sqrt(pi/gamma)/p * ... (approximate)
    return d_coeff * K_AB * (2.0 * M_PI / p) * std::sqrt(M_PI / gamma) * result;
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

inline std::vector<ContractedShellInfo> build_contracted_shells(
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
//  Main: compute ECP matrix (Type 1 only, analytical)
//  Type 2 (semi-local) requires angular projection — TODO
// ============================================================
inline void compute_ecp_matrix(
    const PrimitiveShell* shells, int num_primitives,
    const double* cgto_norms,
    int num_basis,
    const Atom* atoms, int num_atoms,
    const std::unordered_map<std::string, ElementECP>& ecp_data,
    double* V_ecp,
    int /*n_radial*/ = 75, int /*angular_order*/ = 26)
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

        // Combine local + semilocal components for Type 1 evaluation
        // V_ECP = V_local + sum_l dV_l P_l
        // Type 1: <mu|V_local|nu> — computed analytically
        // Type 2: <mu|dV_l P_l|nu> — TODO (for now, only Type 1)

        // DEBUG: check contracted shells for basis=7
        for (size_t si = 0; si < contracted.size(); si++) {
            if (contracted[si].basis_start == 7)
                std::cout << "[ECP-DBG] shell " << si << ": basis=" << contracted[si].basis_start
                          << " type=" << contracted[si].shell_type
                          << " nprims=" << contracted[si].prim_indices.size()
                          << " center=(" << contracted[si].center.x << "," << contracted[si].center.y << "," << contracted[si].center.z << ")" << std::endl;
        }

        // DEBUG: print all ECP components
        std::cout << "[ECP-DBG] l_max=" << ecp.get_l_max() << " n_semilocal=" << ecp.num_semilocal() << std::endl;
        std::cout << "[ECP-DBG] Local: " << ecp.get_local().primitives.size() << " primitives, l=" << ecp.get_local().angular_momentum << std::endl;
        for (const auto& p : ecp.get_local().primitives)
            std::cout << "  n=" << p.power << " z=" << p.exponent << " d=" << p.coefficient << std::endl;
        for (size_t il = 0; il < ecp.num_semilocal(); il++) {
            std::cout << "[ECP-DBG] Semilocal[" << il << "]: " << ecp.get_semilocal()[il].primitives.size() << " primitives, l=" << ecp.get_semilocal()[il].angular_momentum << std::endl;
            for (const auto& p : ecp.get_semilocal()[il].primitives)
                std::cout << "  n=" << p.power << " z=" << p.exponent << " d=" << p.coefficient << std::endl;
        }

        // DEBUG: verify cart_map for d-type
        if (cart_map.count(2)) {
            std::cout << "[CART-DBG] d-type components:" << std::endl;
            for (int i = 0; i < (int)cart_map[2].size(); i++)
                std::cout << "  " << i << ": (" << cart_map[2][i][0] << "," << cart_map[2][i][1] << "," << cart_map[2][i][2] << ")" << std::endl;
        }

        auto compute_type1 = [&](const ECPComponent& comp) {
            // DEBUG: track d-type contributions
            bool dbg_first = true;
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
                                        double prim_val;
                                        if (ecp_prim.power == 2) {
                                            prim_val = type1_n2_primitive(
                                                shells[ip].exponent, cs_a.center, la, ma, na,
                                                shells[jp].exponent, cs_b.center, lb, mb, nb,
                                                ecp_prim.exponent, C, ecp_prim.coefficient);
                                        } else {
                                            // n=0, n=1: skip for now (formulas need verification)
                                            prim_val = 0.0;
                                        }
                                        double Na = primitive_norm(shells[ip].exponent, la, ma, na);
                                        double Nb = primitive_norm(shells[jp].exponent, lb, mb, nb);
                                        val += shells[ip].coefficient * shells[jp].coefficient * Na * Nb * prim_val;
                                    }
                                }
                                // Use self-computed contracted norm (avoids cgto_norms index mapping issues)
                                double norm_mu = contracted_shell_norm(shells, cs_a.prim_indices, la, ma, na);
                                double norm_nu = contracted_shell_norm(shells, cs_b.prim_indices, lb, mb, nb);
                                val *= norm_mu * norm_nu;

                                V_ecp[mu * num_basis + nu] += val;
                                if (mu != nu)
                                    V_ecp[nu * num_basis + mu] += val;
                            }
                        }
                    }
                }
            }
        };

        // Type 1: local component
        compute_type1(ecp.get_local());

        // Type 2 (semi-local): <mu|dV_l P_l|nu>
        // Using angular projector matrix computed via Lebedev grid:
        // P_l^{ab} = Σ_m [∫ S_a Y_lm dΩ] [∫ S_b Y_lm dΩ]
        // Radial: ∫ r^{2+La+Lb} exp(-γ r²) dr = Γ((La+Lb+3)/2) / (2γ^{(La+Lb+3)/2})
        {
            // Precompute angular projector matrices for same-center using Lebedev grid
            // proj_l[L][l] is a flat array of size nL × nL, where nL = num Cartesian components for L
            auto ang_grid = lebedev_26();
            int n_ang = ang_grid.size();

            // For each pair of shell angular momenta and ECP projector l:
            // compute the projector matrix P_l^{cart_a, cart_b}
            auto compute_proj_matrix = [&](int L, int l_proj) -> std::vector<double> {
                int nL = shell_type_to_num_basis(L);
                std::vector<double> proj(nL * nL, 0.0);
                auto comps_L = cartesian_components(L);
                int n_m = 2 * l_proj + 1;

                // C_{a,lm} = ∫ S_a(Ω) Y_lm(Ω) dΩ ≈ Σ_k w_k S_a(Ω_k) Y_lm(Ω_k)
                std::vector<std::vector<double>> C_alm(nL, std::vector<double>(n_m, 0.0));
                for (int ia = 0; ia < n_ang; ia++) {
                    double ox = ang_grid[ia].x, oy = ang_grid[ia].y, oz = ang_grid[ia].z;
                    double w = ang_grid[ia].w;
                    auto Ylm = solid_harmonics(l_proj, ox, oy, oz);

                    for (int a = 0; a < nL; a++) {
                        // S_a(Ω) = Ω_x^lx Ω_y^ly Ω_z^lz
                        double Sa = 1.0;
                        for (int k = 0; k < comps_L[a][0]; k++) Sa *= ox;
                        for (int k = 0; k < comps_L[a][1]; k++) Sa *= oy;
                        for (int k = 0; k < comps_L[a][2]; k++) Sa *= oz;
                        for (int m = 0; m < n_m; m++)
                            C_alm[a][m] += w * Sa * Ylm[m];
                    }
                }

                // P_l^{ab} = Σ_m C_{a,lm} C_{b,lm}
                for (int a = 0; a < nL; a++)
                    for (int b = 0; b < nL; b++)
                        for (int m = 0; m < n_m; m++)
                            proj[a * nL + b] += C_alm[a][m] * C_alm[b][m];

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

                            // Same-center case: both shells on ECP atom
                            if (Da < 1e-10 && Db < 1e-10 && La == Lb) {
                                // Angular projector must match: only L = l contributes for same-center
                                // But l can also differ from L: e.g., d-type has l=0 and l=2 components
                                // The projector matrix handles this correctly
                                if (l_proj > La) continue; // angular momentum selection
                                if ((La - l_proj) % 2 != 0) continue; // parity

                                auto proj = compute_proj_matrix(La, l_proj);
                                int nL = cs_a.num_basis_funcs;

                                // Radial: ∫_0^∞ r^{2+La+Lb} exp(-γ r²) dr = Γ(n+1/2) / (2γ^{n+1/2})
                                // where n = 1 + (La+Lb)/2
                                int n_rad = 1 + (La + Lb) / 2;
                                auto gamma_half_int = [](int n) -> double {
                                    // Γ(n + 1/2) = (2n-1)!! / 2^n * √π
                                    double val = std::sqrt(M_PI);
                                    for (int k = 1; k <= n; k++)
                                        val *= (2.0*k - 1.0) / 2.0;
                                    return val;
                                };

                                for (int ca = 0; ca < nL; ca++) {
                                    int mu = cs_a.basis_start + ca;
                                    int la = comps_a[ca][0], ma_c = comps_a[ca][1], na_c = comps_a[ca][2];

                                    for (int cb = 0; cb < nL; cb++) {
                                        int nu = cs_b.basis_start + cb;
                                        if (si == sj && nu < mu) continue;
                                        int lb = comps_b[cb][0], mb_c = comps_b[cb][1], nb_c = comps_b[cb][2];

                                        double P_ab = proj[ca * nL + cb];
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
                                        double nm = contracted_shell_norm(shells, cs_a.prim_indices, la, ma_c, na_c);
                                        double nn = contracted_shell_norm(shells, cs_b.prim_indices, lb, mb_c, nb_c);
                                        val *= nm * nn;

                                        V_ecp[mu * num_basis + nu] += val;
                                        if (mu != nu)
                                            V_ecp[nu * num_basis + mu] += val;
                                    }
                                }
                            }
                            // Off-center s-s: Bessel expansion
                            else if (La == 0 && Lb == 0 && (Da > 1e-10 || Db > 1e-10)) {
                                double DAx = cs_a.center.x-C.x, DAy = cs_a.center.y-C.y, DAz = cs_a.center.z-C.z;
                                double DBx = cs_b.center.x-C.x, DBy = cs_b.center.y-C.y, DBz = cs_b.center.z-C.z;
                                double cos_gamma = (Da > 1e-10 && Db > 1e-10) ?
                                    (DAx*DBx + DAy*DBy + DAz*DBz) / (Da*Db) : 1.0;

                                double Pl = 1.0;
                                if (l_proj == 1) Pl = cos_gamma;
                                else if (l_proj == 2) Pl = (3*cos_gamma*cos_gamma - 1) / 2;
                                else if (l_proj == 3) Pl = (5*cos_gamma*cos_gamma*cos_gamma - 3*cos_gamma) / 2;

                                int mu = cs_a.basis_start, nu = cs_b.basis_start;
                                if (si == sj && nu < mu) continue;

                                double val = 0.0;
                                for (size_t ip : cs_a.prim_indices) {
                                    for (size_t jp : cs_b.prim_indices) {
                                        double aa = shells[ip].exponent, ab = shells[jp].exponent;
                                        double gamma_r = aa + ab + zeta;
                                        double ka = 2*aa*Da, kb = 2*ab*Db;
                                        double R_max = std::sqrt(35.0 / std::max(std::min({aa, ab, zeta}), 0.01));

                                        static thread_local std::vector<double> rn, rw;
                                        static thread_local bool ri = false;
                                        if (!ri) { gauss_legendre_nodes(50, rn, rw); ri = true; }

                                        double rad_int = 0.0;
                                        for (int ir = 0; ir < 50; ir++) {
                                            double t = (rn[ir]+1)/2.0, r = R_max*t;
                                            double wr = rw[ir] * R_max/2.0 * r*r;
                                            if (r < 1e-15) continue;
                                            double xa = ka*r, xb = kb*r;
                                            if (xa > 500 || xb > 500) continue;
                                            double log_v = -gamma_r*r*r - aa*Da*Da - ab*Db*Db;
                                            double ila, ilb;
                                            if (xa < 1e-10) { ila = (l_proj==0)?1:0; }
                                            else { ila = mod_sph_bessel(l_proj,xa)*std::exp(-xa); log_v += xa; }
                                            if (xb < 1e-10) { ilb = (l_proj==0)?1:0; }
                                            else { ilb = mod_sph_bessel(l_proj,xb)*std::exp(-xb); log_v += xb; }
                                            double ig = wr * std::exp(log_v) * ila * ilb;
                                            if (std::isfinite(ig)) rad_int += ig;
                                        }

                                        double Na = primitive_norm(aa, 0, 0, 0);
                                        double Nb = primitive_norm(ab, 0, 0, 0);
                                        val += shells[ip].coefficient * shells[jp].coefficient * Na * Nb
                                             * d_ecp * (2*l_proj+1) * 4*M_PI * Pl * rad_int;
                                    }
                                }
                                double nm = contracted_shell_norm(shells, cs_a.prim_indices, 0, 0, 0);
                                double nn = contracted_shell_norm(shells, cs_b.prim_indices, 0, 0, 0);
                                val *= nm * nn;
                                V_ecp[mu * num_basis + nu] += val;
                                if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                            }
                            // TODO: off-center p/d-type Type 2
                        }
                    }
                }
            }
        }

        std::cout << "  ECP integrals for " << elem << " computed (analytical Type 1)" << std::endl;
    }
}

} // namespace ecp_integral
} // namespace gansu
