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
#include <omp.h>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
namespace gansu { namespace gpu { bool gpu_available(); } }
#endif

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
    using A = std::array<int,3>;
    if (l == 0) return {A{0,0,0}};
    if (l == 1) return {A{1,0,0}, A{0,1,0}, A{0,0,1}};
    if (l == 2) return {A{2,0,0}, A{0,2,0}, A{0,0,2}, A{1,1,0}, A{1,0,1}, A{0,1,1}};
    if (l == 3) return {A{3,0,0}, A{0,3,0}, A{0,0,3}, A{1,2,0}, A{2,1,0}, A{2,0,1}, A{1,0,2}, A{0,1,2}, A{0,2,1}, A{1,1,1}};
    std::vector<A> comps;
    for (int lx = l; lx >= 0; lx--)
        for (int ly = l-lx; ly >= 0; ly--)
            comps.push_back(A{lx, ly, l-lx-ly});
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
//  Hermite Coulomb integral R^n_{tuv}(p, RPC)
// ============================================================
static double R_hermite(int t, int u, int v, int n, double p,
                        double RPC_x, double RPC_y, double RPC_z) {
    if (t < 0 || u < 0 || v < 0) return 0.0;
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
    return (v-1)*R_hermite(t,u,v-2,n+1,p,RPC_x,RPC_y,RPC_z)
         + RPC_z*R_hermite(t,u,v-1,n+1,p,RPC_x,RPC_y,RPC_z);
}

// ============================================================
//  Type 1 ECP integrals (local potential) for all powers n
//
//  n=2: <μ|d exp(-ζr²)|ν>  → 3-center overlap (pure Gaussian)
//  n=1: <μ|d r⁻¹ exp(-ζr²)|ν>  → modified nuclear attraction
//  n=0: <μ|d r⁻² exp(-ζr²)|ν>  → Boys function based
// ============================================================
static double type1_primitive(
    double alpha, const Coordinate& A, int la, int ma, int na_,
    double beta, const Coordinate& B, int lb, int mb, int nb_,
    double zeta, const Coordinate& C, double d_coeff, int power)
{
    double p = alpha + beta;
    double Px = (alpha*A.x + beta*B.x) / p;
    double Py = (alpha*A.y + beta*B.y) / p;
    double Pz = (alpha*A.z + beta*B.z) / p;
    double gamma = p + zeta;
    double AB2 = (A.x-B.x)*(A.x-B.x) + (A.y-B.y)*(A.y-B.y) + (A.z-B.z)*(A.z-B.z);
    double K_AB = std::exp(-alpha*beta/p * AB2);

    // Helper lambda: 3-center overlap integral with arbitrary zeta (n=2 kernel)
    auto overlap_3c = [&](double z) -> double {
        double g = p + z;
        double qx = (p*Px + z*C.x) / g;
        double qy = (p*Py + z*C.y) / g;
        double qz = (p*Pz + z*C.z) / g;
        double PC2 = (Px-C.x)*(Px-C.x) + (Py-C.y)*(Py-C.y) + (Pz-C.z)*(Pz-C.z);
        double Kpc = std::exp(-p*z/g * PC2);
        return K_AB * Kpc * std::pow(M_PI / g, 1.5)
             * E_coeff(la, lb, 0, g, qx-A.x, qx-B.x)
             * E_coeff(ma, mb, 0, g, qy-A.y, qy-B.y)
             * E_coeff(na_, nb_, 0, g, qz-A.z, qz-B.z);
    };

    if (power == 2) {
        // Direct analytical computation (fast path, no lambda overhead)
        double g = p + zeta;
        double qx = (p*Px + zeta*C.x) / g;
        double qy = (p*Py + zeta*C.y) / g;
        double qz = (p*Pz + zeta*C.z) / g;
        double PC2 = (Px-C.x)*(Px-C.x) + (Py-C.y)*(Py-C.y) + (Pz-C.z)*(Pz-C.z);
        double Kpc = std::exp(-p*zeta/g * PC2);
        return d_coeff * K_AB * Kpc * std::pow(M_PI / g, 1.5)
             * E_coeff(la, lb, 0, g, qx-A.x, qx-B.x)
             * E_coeff(ma, mb, 0, g, qy-A.y, qy-B.y)
             * E_coeff(na_, nb_, 0, g, qz-A.z, qz-B.z);
    }

    // GL nodes for 1D quadrature (cached)
    static std::vector<double> gl_n, gl_w;
    static bool gl_init = false;
    if (!gl_init) { gauss_legendre_nodes(50, gl_n, gl_w); gl_init = true; }

    if (power == 1) {
        // r^{-1} exp(-ζr²) = (2/√π) ∫_0^∞ exp(-(ζ+u²)r²) du
        // → <μ|V|ν> = d × (2/√π) ∫_0^∞ overlap_3c(ζ+u²) du
        double R = std::sqrt(35.0 + p + zeta);
        double val = 0.0;
        for (int i = 0; i < 50; i++) {
            double u = R * (gl_n[i] + 1.0) / 2.0;
            val += gl_w[i] * R / 2.0 * overlap_3c(zeta + u*u);
        }
        return d_coeff * (2.0 / std::sqrt(M_PI)) * val;
    }

    // power == 0: r^{-2} exp(-ζr²) = ∫_ζ^∞ exp(-t r²) dt
    // → <μ|V|ν> = d × ∫_ζ^∞ overlap_3c(t) dt
    {
        double R = 35.0 + p;
        double val = 0.0;
        for (int i = 0; i < 50; i++) {
            double t_ = zeta + R * (gl_n[i] + 1.0) / 2.0;
            val += gl_w[i] * R / 2.0 * overlap_3c(t_);
        }
        return d_coeff * val;
    }
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
                                        double prim_val = type1_primitive(
                                                shells[ip].exponent, cs_a.center, la, ma, na,
                                                shells[jp].exponent, cs_b.center, lb, mb, nb,
                                                ecp_prim.exponent, C, ecp_prim.coefficient,
                                                ecp_prim.power);
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
            auto off_ang = product_angular_grid(15, 30);
            int off_n_ang = (int)off_ang.size();
            const int off_n_rad = 80;
            std::vector<double> off_rn, off_rw;
            gauss_legendre_nodes(off_n_rad, off_rn, off_rw);

#ifndef GANSU_CPU_ONLY
            // GPU resources (allocated once per ECP atom, freed at end of Type 2 block)
            bool use_gpu = gpu::gpu_available();
            AngPtGPU* d_ang_grid = nullptr;
            double* d_rad_nodes = nullptr;
            double* d_rad_weights = nullptr;
            double* d_output_prim_val = nullptr;
            int max_ecp_prims = 0;

            if (use_gpu) {
                // Upload angular grid
                std::vector<AngPtGPU> ang_gpu(off_n_ang);
                for (int i = 0; i < off_n_ang; i++)
                    ang_gpu[i] = {off_ang[i].x, off_ang[i].y, off_ang[i].z, off_ang[i].w};
                cudaMalloc(&d_ang_grid, off_n_ang * sizeof(AngPtGPU));
                cudaMemcpy(d_ang_grid, ang_gpu.data(), off_n_ang * sizeof(AngPtGPU), cudaMemcpyHostToDevice);

                // Upload radial grid
                cudaMalloc(&d_rad_nodes, off_n_rad * sizeof(double));
                cudaMalloc(&d_rad_weights, off_n_rad * sizeof(double));
                cudaMemcpy(d_rad_nodes, off_rn.data(), off_n_rad * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_rad_weights, off_rw.data(), off_n_rad * sizeof(double), cudaMemcpyHostToDevice);

                // Find max ECP primitives across all components
                for (size_t il = 0; il < ecp.num_semilocal(); il++)
                    if ((int)ecp.get_semilocal()[il].primitives.size() > max_ecp_prims)
                        max_ecp_prims = (int)ecp.get_semilocal()[il].primitives.size();
                cudaMalloc(&d_output_prim_val, max_ecp_prims * sizeof(double));
            }
#else
            bool use_gpu = false;
#endif

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

            auto radial_integral = [](int M, double g) -> double {
                if (M % 2 == 0) {
                    int n = M / 2;
                    double val = std::sqrt(M_PI);
                    for (int k = 1; k <= n; k++) val *= (2.0*k - 1.0) / 2.0;
                    return val / (2.0 * std::pow(g, n + 0.5));
                } else {
                    int n = (M - 1) / 2;
                    double val = 1.0;
                    for (int k = 1; k <= n; k++) val *= k;
                    return val / (2.0 * std::pow(g, n + 1.0));
                }
            };

            for (size_t il = 0; il < ecp.num_semilocal(); il++) {
                const auto& sl_comp = ecp.get_semilocal()[il];
                int l_proj = sl_comp.angular_momentum;
                int n_m = 2 * l_proj + 1;

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

                            for (const auto& ecp_prim : sl_comp.primitives) {
                                int rad_exp = La + Lb + ecp_prim.power;
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
                                                double gamma = shells[ip].exponent + shells[jp].exponent + ecp_prim.exponent;
                                                double Na = primitive_norm(shells[ip].exponent, la, ma_c, na_c);
                                                double Nb = primitive_norm(shells[jp].exponent, lb, mb_c, nb_c);
                                                val += shells[ip].coefficient * shells[jp].coefficient * Na * Nb
                                                     * ecp_prim.coefficient * P_ab * radial_integral(rad_exp, gamma);
                                            }
                                        }
                                        val *= cgto_norms[mu] * cgto_norms[nu];
                                        V_ecp[mu * num_basis + nu] += val;
                                        if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                                    }
                                }
                            }
                        }
                        // ---- Off-center: numerical quadrature (GPU or OpenMP) ----
                        else if (Da > 1e-10 || Db > 1e-10) {
                            double dAx = cs_a.center.x-C.x, dAy = cs_a.center.y-C.y, dAz = cs_a.center.z-C.z;
                            double dBx = cs_b.center.x-C.x, dBy = cs_b.center.y-C.y, dBz = cs_b.center.z-C.z;

                            double alpha_min_a = 1e30, alpha_min_b = 1e30;
                            for (size_t ip : cs_a.prim_indices)
                                if (shells[ip].exponent < alpha_min_a) alpha_min_a = shells[ip].exponent;
                            for (size_t jp : cs_b.prim_indices)
                                if (shells[jp].exponent < alpha_min_b) alpha_min_b = shells[jp].exponent;
                            if (std::exp(-alpha_min_a*Da*Da) * std::exp(-alpha_min_b*Db*Db) < 1e-20) continue;

                            int nA = cs_a.num_basis_funcs;
                            int nB = cs_b.num_basis_funcs;
                            int n_prims = (int)sl_comp.primitives.size();

                            double zeta_min_ecp = 1e30;
                            for (const auto& ep : sl_comp.primitives)
                                if (ep.exponent < zeta_min_ecp) zeta_min_ecp = ep.exponent;
                            double R_max = std::sqrt(35.0 / (zeta_min_ecp > 0.01 ? zeta_min_ecp : 0.01));

                            int n_pairs = nA * nB;
                            std::vector<double> results(n_pairs * n_prims, 0.0);

#ifndef GANSU_CPU_ONLY
                            if (use_gpu) {
                                // Prepare ECP primitive data for GPU
                                std::vector<double> ecp_exp_v(n_prims), ecp_coef_v(n_prims);
                                std::vector<int> ecp_pow_v(n_prims);
                                for (int k = 0; k < n_prims; k++) {
                                    ecp_exp_v[k] = sl_comp.primitives[k].exponent;
                                    ecp_coef_v[k] = sl_comp.primitives[k].coefficient;
                                    ecp_pow_v[k] = sl_comp.primitives[k].power;
                                }
                                double *d_ecp_exp, *d_ecp_coef; int *d_ecp_pow;
                                cudaMalloc(&d_ecp_exp, n_prims*sizeof(double));
                                cudaMalloc(&d_ecp_coef, n_prims*sizeof(double));
                                cudaMalloc(&d_ecp_pow, n_prims*sizeof(int));
                                cudaMemcpy(d_ecp_exp, ecp_exp_v.data(), n_prims*sizeof(double), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_ecp_coef, ecp_coef_v.data(), n_prims*sizeof(double), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_ecp_pow, ecp_pow_v.data(), n_prims*sizeof(int), cudaMemcpyHostToDevice);

                                // Prepare basis primitive data
                                std::vector<double> exp_a, coef_a, exp_b, coef_b;
                                for (size_t ip : cs_a.prim_indices) { exp_a.push_back(shells[ip].exponent); coef_a.push_back(shells[ip].coefficient); }
                                for (size_t jp : cs_b.prim_indices) { exp_b.push_back(shells[jp].exponent); coef_b.push_back(shells[jp].coefficient); }
                                double *d_exp_a, *d_coef_a, *d_exp_b, *d_coef_b;
                                cudaMalloc(&d_exp_a, exp_a.size()*sizeof(double));
                                cudaMalloc(&d_coef_a, coef_a.size()*sizeof(double));
                                cudaMalloc(&d_exp_b, exp_b.size()*sizeof(double));
                                cudaMalloc(&d_coef_b, coef_b.size()*sizeof(double));
                                cudaMemcpy(d_exp_a, exp_a.data(), exp_a.size()*sizeof(double), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_coef_a, coef_a.data(), coef_a.size()*sizeof(double), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_exp_b, exp_b.data(), exp_b.size()*sizeof(double), cudaMemcpyHostToDevice);
                                cudaMemcpy(d_coef_b, coef_b.data(), coef_b.size()*sizeof(double), cudaMemcpyHostToDevice);

                                for (int pair = 0; pair < n_pairs; pair++) {
                                    int ca = pair / nB, cb = pair % nB;
                                    int mu = cs_a.basis_start + ca, nu = cs_b.basis_start + cb;
                                    if (si == sj && nu < mu) continue;

                                    launch_ecp_type2_offcenter_gpu(
                                        d_ang_grid, off_n_ang,
                                        d_exp_a, d_coef_a, (int)exp_a.size(),
                                        comps_a[ca][0], comps_a[ca][1], comps_a[ca][2], dAx, dAy, dAz,
                                        d_exp_b, d_coef_b, (int)exp_b.size(),
                                        comps_b[cb][0], comps_b[cb][1], comps_b[cb][2], dBx, dBy, dBz,
                                        d_ecp_exp, d_ecp_coef, d_ecp_pow, n_prims,
                                        d_rad_nodes, d_rad_weights, off_n_rad, R_max, l_proj,
                                        d_output_prim_val);

                                    cudaMemcpy(&results[pair*n_prims], d_output_prim_val, n_prims*sizeof(double), cudaMemcpyDeviceToHost);
                                }
                                cudaFree(d_ecp_exp); cudaFree(d_ecp_coef); cudaFree(d_ecp_pow);
                                cudaFree(d_exp_a); cudaFree(d_coef_a); cudaFree(d_exp_b); cudaFree(d_coef_b);
                            } else
#endif
                            {
                                // CPU path (OpenMP)
                                #pragma omp parallel for schedule(dynamic)
                                for (int pair = 0; pair < n_pairs; pair++) {
                                    int ca = pair / nB, cb = pair % nB;
                                    int mu = cs_a.basis_start + ca;
                                    int nu = cs_b.basis_start + cb;
                                    if (si == sj && nu < mu) continue;
                                    int la = comps_a[ca][0], ma_c = comps_a[ca][1], na_c = comps_a[ca][2];
                                    int lb = comps_b[cb][0], mb_c = comps_b[cb][1], nb_c = comps_b[cb][2];
                                    double* pv = &results[pair * n_prims];

                                    for (int ir = 0; ir < off_n_rad; ir++) {
                                        double t_ = (off_rn[ir] + 1.0) / 2.0;
                                        double r = R_max * t_;
                                        double wr = off_rw[ir] * R_max / 2.0;
                                        if (r < 1e-15) continue;
                                        double Alm_mu[7] = {}, Alm_nu[7] = {};
                                        for (int ia = 0; ia < off_n_ang; ia++) {
                                            double ox = off_ang[ia].x, oy = off_ang[ia].y, oz = off_ang[ia].z;
                                            double w_ang = off_ang[ia].w;
                                            double rx = r*ox-dAx, ry = r*oy-dAy, rz = r*oz-dAz;
                                            double r2A = rx*rx + ry*ry + rz*rz;
                                            double sx = r*ox-dBx, sy = r*oy-dBy, sz = r*oz-dBz;
                                            double r2B = sx*sx + sy*sy + sz*sz;
                                            double pow_mu = 1.0;
                                            for (int k=0;k<la;k++) pow_mu*=rx;
                                            for (int k=0;k<ma_c;k++) pow_mu*=ry;
                                            for (int k=0;k<na_c;k++) pow_mu*=rz;
                                            double pow_nu = 1.0;
                                            for (int k=0;k<lb;k++) pow_nu*=sx;
                                            for (int k=0;k<mb_c;k++) pow_nu*=sy;
                                            for (int k=0;k<nb_c;k++) pow_nu*=sz;
                                            double mu_v = 0.0;
                                            for (size_t ip : cs_a.prim_indices)
                                                mu_v += shells[ip].coefficient * primitive_norm(shells[ip].exponent, la, ma_c, na_c)
                                                        * pow_mu * std::exp(-shells[ip].exponent * r2A);
                                            double nu_v = 0.0;
                                            for (size_t jp : cs_b.prim_indices)
                                                nu_v += shells[jp].coefficient * primitive_norm(shells[jp].exponent, lb, mb_c, nb_c)
                                                        * pow_nu * std::exp(-shells[jp].exponent * r2B);
                                            const double* Ylm_ptr = &off_Ylm_flat[ia * off_stride + off_Ylm_offset[l_proj]];
                                            for (int m = 0; m < n_m; m++) {
                                                Alm_mu[m] += w_ang * mu_v * Ylm_ptr[m];
                                                Alm_nu[m] += w_ang * nu_v * Ylm_ptr[m];
                                            }
                                        }
                                        double ang_sum = 0.0;
                                        for (int m = 0; m < n_m; m++)
                                            ang_sum += Alm_mu[m] * Alm_nu[m];
                                        for (int k = 0; k < n_prims; k++) {
                                            double ecp_rad = std::pow(r, sl_comp.primitives[k].power)
                                                           * std::exp(-sl_comp.primitives[k].exponent * r * r);
                                            pv[k] += wr * ecp_rad * ang_sum;
                                        }
                                    }
                                }
                            }

                            // Collect results (sequential, no race condition)
                            for (int pair = 0; pair < n_pairs; pair++) {
                                int ca = pair / nB, cb = pair % nB;
                                int mu = cs_a.basis_start + ca;
                                int nu = cs_b.basis_start + cb;
                                if (si == sj && nu < mu) continue;
                                double val = 0.0;
                                for (int k = 0; k < n_prims; k++)
                                    val += sl_comp.primitives[k].coefficient * results[pair * n_prims + k];
                                val *= cgto_norms[mu] * cgto_norms[nu];
                                V_ecp[mu * num_basis + nu] += val;
                                if (mu != nu) V_ecp[nu * num_basis + mu] += val;
                            }
                        }
                    }
                }
            }

#ifndef GANSU_CPU_ONLY
            if (use_gpu) {
                cudaFree(d_ang_grid);
                cudaFree(d_rad_nodes);
                cudaFree(d_rad_weights);
                cudaFree(d_output_prim_val);
            }
#endif
        }

        std::cout << "  ECP integrals for " << elem << " computed" << std::endl;
    }
}

// ============================================================
//  ECP gradient via finite difference of V_ECP matrix
//  grad_ecp[3*A + d] = Tr(D * dV_ECP/dR_Ad)
//  dV_ECP/dR_Ad ≈ (V_ECP(R_A + h*e_d) - V_ECP(R_A - h*e_d)) / (2h)
// ============================================================
void compute_ecp_gradient(
    const PrimitiveShell* shells, int num_primitives,
    const double* cgto_norms,
    int num_basis,
    const Atom* atoms, int num_atoms,
    const std::unordered_map<std::string, ElementECP>& ecp_data,
    const double* density_matrix,
    double* grad_ecp)
{
    const double h = 1e-5; // finite difference step (bohr)
    const size_t mat_size = (size_t)num_basis * num_basis;

    // For each atom and each direction (x,y,z)
    for (int iatom = 0; iatom < num_atoms; iatom++) {
        for (int dir = 0; dir < 3; dir++) {
            // Create displaced atom AND shell arrays
            std::vector<Atom> atoms_plus(atoms, atoms + num_atoms);
            std::vector<Atom> atoms_minus(atoms, atoms + num_atoms);
            std::vector<PrimitiveShell> shells_plus(shells, shells + num_primitives);
            std::vector<PrimitiveShell> shells_minus(shells, shells + num_primitives);

            // Displace atom coordinates
            if (dir == 0) { atoms_plus[iatom].coordinate.x += h; atoms_minus[iatom].coordinate.x -= h; }
            if (dir == 1) { atoms_plus[iatom].coordinate.y += h; atoms_minus[iatom].coordinate.y -= h; }
            if (dir == 2) { atoms_plus[iatom].coordinate.z += h; atoms_minus[iatom].coordinate.z -= h; }

            // Also displace shell coordinates for shells on this atom
            for (int ip = 0; ip < num_primitives; ip++) {
                if (shells[ip].atom_index == iatom) {
                    if (dir == 0) { shells_plus[ip].coordinate.x += h; shells_minus[ip].coordinate.x -= h; }
                    if (dir == 1) { shells_plus[ip].coordinate.y += h; shells_minus[ip].coordinate.y -= h; }
                    if (dir == 2) { shells_plus[ip].coordinate.z += h; shells_minus[ip].coordinate.z -= h; }
                }
            }

            // Compute V_ECP at displaced geometries
            std::vector<double> V_plus(mat_size, 0.0), V_minus(mat_size, 0.0);

            compute_ecp_matrix(shells_plus.data(), num_primitives, cgto_norms, num_basis,
                               atoms_plus.data(), num_atoms, ecp_data, V_plus.data());
            compute_ecp_matrix(shells_minus.data(), num_primitives, cgto_norms, num_basis,
                               atoms_minus.data(), num_atoms, ecp_data, V_minus.data());

            // dV/dR = (V+ - V-) / 2h, then contract with density: Tr(D * dV/dR)
            double grad_val = 0.0;
            for (size_t ij = 0; ij < mat_size; ij++) {
                grad_val += density_matrix[ij] * (V_plus[ij] - V_minus[ij]) / (2.0 * h);
            }

            grad_ecp[3 * iatom + dir] += grad_val;
        }
    }
}

} // namespace ecp_integral
} // namespace gansu
