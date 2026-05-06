/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file thc_grid.cu
 * @brief Implementation of the THC molecular numerical-integration grid.
 *
 * Lebedev tables: V. I. Lebedev & D. N. Laikov, Dokl. Math. 59, 477 (1999).
 *   Encoded as octahedral orbit specifications (a1, a2, a3, bk, ck, dk) with
 *   parameters and weights.  Each orbit is expanded at runtime by
 *   emit_orbit().  This keeps the table compact and verifiable.
 *
 * Radial: Treutler-Ahlrichs M3 logarithmic mapping over Gauss-Chebyshev-2 nodes,
 *   per O. Treutler & R. Ahlrichs, J. Chem. Phys. 102, 346 (1995).
 *
 * Partition: A. D. Becke, J. Chem. Phys. 88, 2547 (1988), 3-fold polynomial
 *   smoothing with Becke's atomic-size adjustment.
 */

#include "thc_grid.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cassert>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#endif

#include "utils.hpp"

namespace gansu {

namespace {

// -----------------------------------------------------------------------------
// Lebedev orbit specification
// -----------------------------------------------------------------------------

enum class OrbitClass : int {
    A1, // 6 pts: (+/-1, 0, 0) + permutations
    A2, // 8 pts: (+/-1/sqrt(3))^3
    A3, // 12 pts: (+/-1/sqrt(2), +/-1/sqrt(2), 0) + perms
    BK, // 24 pts: (+/-l, +/-l, +/-m), m = sqrt(1-2*l^2), perms
    CK, // 24 pts: (+/-p, +/-q, 0), q = sqrt(1-p^2), perms
    DK  // 48 pts: (+/-r, +/-s, +/-t), all distinct, perms; r^2+s^2+t^2=1
};

struct LebedevOrbit {
    OrbitClass cls;
    double p1;   // l (BK), p (CK), r (DK), unused for A*
    double p2;   // unused except DK (s)
    double w;    // surface-element weight (sum of all orbit weights = 4*pi)
};

// emit_orbit: expand one orbit into Cartesian {x,y,z,w} on the unit sphere.
static void emit_orbit(const LebedevOrbit& o, std::vector<GridPoint>& out)
{
    auto add = [&](double x, double y, double z, double w) {
        out.push_back(GridPoint{
            static_cast<real_t>(x), static_cast<real_t>(y), static_cast<real_t>(z),
            static_cast<real_t>(w), -1});
    };

    const double w = o.w;

    switch (o.cls) {
    case OrbitClass::A1:
        for (int s : {-1, 1}) {
            add(s, 0, 0, w);
            add(0, s, 0, w);
            add(0, 0, s, w);
        }
        break;

    case OrbitClass::A2: {
        const double a = 1.0 / std::sqrt(3.0);
        for (int sx : {-1, 1}) for (int sy : {-1, 1}) for (int sz : {-1, 1})
            add(sx*a, sy*a, sz*a, w);
        break;
    }

    case OrbitClass::A3: {
        const double a = 1.0 / std::sqrt(2.0);
        for (int s1 : {-1, 1}) for (int s2 : {-1, 1}) {
            add(s1*a, s2*a, 0, w);
            add(s1*a, 0, s2*a, w);
            add(0, s1*a, s2*a, w);
        }
        break;
    }

    case OrbitClass::BK: {
        const double l = o.p1;
        const double m = std::sqrt(1.0 - 2.0*l*l);
        for (int sx : {-1, 1}) for (int sy : {-1, 1}) for (int sz : {-1, 1}) {
            add(sx*l, sy*l, sz*m, w);
            add(sx*l, sy*m, sz*l, w);
            add(sx*m, sy*l, sz*l, w);
        }
        break;
    }

    case OrbitClass::CK: {
        const double p = o.p1;
        const double q = std::sqrt(1.0 - p*p);
        for (int s1 : {-1, 1}) for (int s2 : {-1, 1}) {
            add(s1*p, s2*q, 0, w);
            add(s1*q, s2*p, 0, w);
            add(s1*p, 0, s2*q, w);
            add(s1*q, 0, s2*p, w);
            add(0, s1*p, s2*q, w);
            add(0, s1*q, s2*p, w);
        }
        break;
    }

    case OrbitClass::DK: {
        const double r = o.p1;
        const double s = o.p2;
        const double t = std::sqrt(1.0 - r*r - s*s);
        const double abc[3] = {r, s, t};
        // 6 permutations * 8 sign combinations = 48
        const int perms[6][3] = {
            {0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}
        };
        for (int p = 0; p < 6; ++p) {
            for (int sx : {-1, 1}) for (int sy : {-1, 1}) for (int sz : {-1, 1})
                add(sx*abc[perms[p][0]], sy*abc[perms[p][1]], sz*abc[perms[p][2]], w);
        }
        break;
    }
    }
}

// -----------------------------------------------------------------------------
// Lebedev-Laikov 1999 orbit tables.
//
// Source: Burkardt's public-domain "sphere_lebedev_rule" data files
// (lebedev_017.txt, lebedev_023.txt, lebedev_029.txt) under GNU LGPL,
// at https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/ .
// These files are the standard tabulation of the Lebedev-Laikov 1999 quadrature
// in (theta, phi, w) form (theta, phi in degrees; w sums to 1 over each grid).
//
// The orbit-class parameters (l for BK, p for CK, r/s for DK) are the
// equal-pair magnitude (BK), the smaller of the two non-zero magnitudes (CK),
// and the two smallest of the three non-zero magnitudes (DK), so they pair
// directly with emit_orbit() above.
//
// Weights are reproduced at the file's 13-digit precision, then multiplied
// by 4*pi to convert from sum-to-1 to sum-to-4*pi (surface element).
// -----------------------------------------------------------------------------

static const std::vector<LebedevOrbit> LEBEDEV_110 = {
    { OrbitClass::A1, 0.0, 0.0, 0.003828270494937 * 4.0 * M_PI },
    { OrbitClass::A2, 0.0, 0.0, 0.009793737512488 * 4.0 * M_PI },
    { OrbitClass::BK, 0.1851156353447363, 0.0, 0.008211737283191 * 4.0 * M_PI },
    { OrbitClass::BK, 0.3956894730559419, 0.0, 0.009595471336071 * 4.0 * M_PI },
    { OrbitClass::BK, 0.6904210483822922, 0.0, 0.009942814891178 * 4.0 * M_PI },
    { OrbitClass::CK, 0.4783690288121502, 0.0, 0.009694996361663 * 4.0 * M_PI }
};

static const std::vector<LebedevOrbit> LEBEDEV_194 = {
    { OrbitClass::A1, 0.0, 0.0, 0.001782340447245 * 4.0 * M_PI },
    { OrbitClass::A2, 0.0, 0.0, 0.005573383178849 * 4.0 * M_PI },
    { OrbitClass::A3, 0.0, 0.0, 0.005716905949977 * 4.0 * M_PI },
    { OrbitClass::BK, 0.1299335447650066, 0.0, 0.004106777028169 * 4.0 * M_PI },
    { OrbitClass::BK, 0.2892465627575440, 0.0, 0.005158237711805 * 4.0 * M_PI },
    { OrbitClass::BK, 0.4446933178717438, 0.0, 0.005518771467274 * 4.0 * M_PI },
    { OrbitClass::BK, 0.6712973442695226, 0.0, 0.005608704082588 * 4.0 * M_PI },
    { OrbitClass::CK, 0.3457702197611283, 0.0, 0.005051846064615 * 4.0 * M_PI },
    { OrbitClass::DK, 0.1590417105383530, 0.5251185724436420, 0.005530248916233 * 4.0 * M_PI }
};

static const std::vector<LebedevOrbit> LEBEDEV_302 = {
    { OrbitClass::A1, 0.0, 0.0, 0.000854591172513 * 4.0 * M_PI },
    { OrbitClass::A2, 0.0, 0.0, 0.003599119285026 * 4.0 * M_PI },
    { OrbitClass::BK, 0.0961830852261480, 0.0, 0.002352101413700 * 4.0 * M_PI },
    { OrbitClass::BK, 0.2219645236294177, 0.0, 0.003108953122414 * 4.0 * M_PI },
    { OrbitClass::BK, 0.3515640345570106, 0.0, 0.003449788424306 * 4.0 * M_PI },
    { OrbitClass::BK, 0.4729054132581006, 0.0, 0.003576729661743 * 4.0 * M_PI },
    { OrbitClass::BK, 0.6566329410219612, 0.0, 0.003604822601420 * 4.0 * M_PI },
    { OrbitClass::BK, 0.7011766416089545, 0.0, 0.003650045807677 * 4.0 * M_PI },
    { OrbitClass::CK, 0.2644152887060662, 0.0, 0.002982344963172 * 4.0 * M_PI },
    { OrbitClass::CK, 0.5718955891878960, 0.0, 0.003600820932216 * 4.0 * M_PI },
    { OrbitClass::DK, 0.1233548532583327, 0.4127724083168531, 0.003392312205006 * 4.0 * M_PI },
    { OrbitClass::DK, 0.2510034751770464, 0.5448677372580775, 0.003571540554273 * 4.0 * M_PI }
};

// -----------------------------------------------------------------------------
// Bragg-Slater atomic radii (in Bohr).
//
// Indexed by atomic number Z (Z=1 -> [1]).  Source: Treutler & Ahlrichs
// 1995, table 1, converted from Angstrom to Bohr (x 1.8897259886).  Beyond
// Xe (Z=54) we fall back to the heaviest tabulated value.
// -----------------------------------------------------------------------------

static double bragg_slater_radius_bohr(int z)
{
    // Angstrom values, [0] padded so [Z] gives element Z.
    static const double r_ang[] = {
        0.00,
        0.35, 0.35,                                             // H, He
        1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,         // Li-Ne
        1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.00,         // Na-Ar
        2.20, 1.80,                                             // K, Ca
        1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, // Sc-Zn
        1.30, 1.25, 1.15, 1.15, 1.15, 1.10,                     // Ga-Kr
        2.35, 2.00,                                             // Rb, Sr
        1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, // Y-Cd
        1.55, 1.45, 1.45, 1.40, 1.40, 1.30                      // In-Xe
    };
    constexpr double ang_to_bohr = 1.8897259886;
    if (z < 1) return 0.5 * ang_to_bohr;
    const int z_max = static_cast<int>(sizeof(r_ang)/sizeof(r_ang[0])) - 1;
    if (z > z_max) z = z_max;
    return r_ang[z] * ang_to_bohr;
}

// -----------------------------------------------------------------------------
// Treutler-Ahlrichs M3 radial node generation
//
// For Gauss-Chebyshev-2 nodes x_k = cos(k*pi/(N+1)), the integral
//   I = int_{-1}^{1} f(x) dx
// is approximated as
//   I ~ sum_k (pi/(N+1)) * sin^2(k*pi/(N+1)) * f(x_k) / sqrt(1-x_k^2)
// using the standard transformation.  The mapping
//   r(x) = (xi/ln 2) * (1+x)^alpha * ln(2/(1-x))
// (alpha = 0.6, M3) sends [-1, 1] -> [0, infty), giving
//   int_0^infty g(r) r^2 dr = int_{-1}^{1} g(r(x)) r(x)^2 (dr/dx) dx
// so the radial weight (with the r^2 Jacobian absorbed) is
//   w_rad(k) = w_cheb_k * r_k^2 * (dr/dx)_k.
// -----------------------------------------------------------------------------

} // anonymous namespace (close so the public symbol below is in `gansu`)

void treutler_ahlrichs_m3(int atomic_number,
                          int n_radial,
                          std::vector<real_t>& r_out,
                          std::vector<real_t>& w_out)
{
    if (n_radial < 1) {
        THROW_EXCEPTION("treutler_ahlrichs_m3: n_radial must be >= 1");
    }

    const double xi = bragg_slater_radius_bohr(atomic_number) / std::log(2.0);
    const double alpha = 0.6;

    r_out.resize(n_radial);
    w_out.resize(n_radial);

    const double inv_np1 = 1.0 / static_cast<double>(n_radial + 1);
    for (int k = 1; k <= n_radial; ++k) {
        const double theta = k * M_PI * inv_np1;
        const double x = std::cos(theta);
        const double sin_theta = std::sin(theta);

        // Gauss-Chebyshev-2 weight on x in [-1, 1] (with sqrt(1-x^2) absorbed)
        // int f dx ~ sum_k (pi/(N+1)) sin^2(theta_k) f(x_k) / sqrt(1-x_k^2)
        //         = sum_k (pi/(N+1)) sin(theta_k) f(x_k)
        const double w_cheb = M_PI * inv_np1 * sin_theta;

        const double one_plus_x = 1.0 + x;
        const double one_minus_x = 1.0 - x;
        if (one_minus_x < 1.0e-15 || one_plus_x < 1.0e-15) {
            // Endpoint guard; use a tiny radius (will get pruned by weight_eps).
            r_out[k-1] = static_cast<real_t>(1.0e-10);
            w_out[k-1] = static_cast<real_t>(0.0);
            continue;
        }

        const double ln_term = std::log(2.0 / one_minus_x);
        const double pow_term = std::pow(one_plus_x, alpha);
        const double r = xi * pow_term * ln_term;

        // dr/dx = xi * [ alpha * (1+x)^(alpha-1) * ln(2/(1-x))
        //              + (1+x)^alpha / (1-x) ]
        const double drdx = xi * (alpha * std::pow(one_plus_x, alpha - 1.0) * ln_term
                                  + pow_term / one_minus_x);

        const double w_rad = w_cheb * r * r * drdx;

        r_out[k-1] = static_cast<real_t>(r);
        w_out[k-1] = static_cast<real_t>(w_rad);
    }
}

namespace { // reopen anonymous namespace for the remaining file-private helpers

// -----------------------------------------------------------------------------
// Becke fuzzy-cell partition (1988)
//
// For each pair of atoms (A, B), define
//   mu_AB(r) = (|r - R_A| - |r - R_B|) / |R_A - R_B|
// and Becke's atomic-size adjustment
//   chi = R_A / R_B
//   u   = (chi - 1) / (chi + 1)
//   a   = u / (u^2 - 1)        (clamped to |a| <= 0.5)
//   nu  = mu + a * (1 - mu^2)
// then iteratively smooth k=3 times:
//   p_1(x) = 1.5 x - 0.5 x^3
//   p_k(x) = p_1(p_{k-1}(x))
// and assign the cell switch s_AB = 0.5 * (1 - p_3(nu_AB)).
//
// Atom A's ownership weight at point r is
//   P_A(r) = prod_{B != A} s_AB(r),  W_A(r) = P_A / sum_C P_C.
// -----------------------------------------------------------------------------

static double becke_polynomial_3(double mu)
{
    auto p = [](double x) { return 1.5 * x - 0.5 * x * x * x; };
    return p(p(p(mu)));
}

static std::vector<double> becke_partition_weights(
    const Coordinate& r,
    const std::vector<Atom>& atoms)
{
    const std::size_t n = atoms.size();
    std::vector<double> dist(n);
    for (std::size_t a = 0; a < n; ++a) {
        const double dx = r.x - atoms[a].coordinate.x;
        const double dy = r.y - atoms[a].coordinate.y;
        const double dz = r.z - atoms[a].coordinate.z;
        dist[a] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    std::vector<double> P(n, 1.0);

    for (std::size_t a = 0; a < n; ++a) {
        const double Ra = bragg_slater_radius_bohr(atoms[a].atomic_number);
        for (std::size_t b = 0; b < n; ++b) {
            if (a == b) continue;
            const double Rb = bragg_slater_radius_bohr(atoms[b].atomic_number);
            const double dab_x = atoms[a].coordinate.x - atoms[b].coordinate.x;
            const double dab_y = atoms[a].coordinate.y - atoms[b].coordinate.y;
            const double dab_z = atoms[a].coordinate.z - atoms[b].coordinate.z;
            const double dab = std::sqrt(dab_x*dab_x + dab_y*dab_y + dab_z*dab_z);
            if (dab < 1.0e-12) continue;

            double mu = (dist[a] - dist[b]) / dab;

            const double chi = Ra / Rb;
            const double u = (chi - 1.0) / (chi + 1.0);
            double a_size = u / (u*u - 1.0);
            if (a_size >  0.5) a_size =  0.5;
            if (a_size < -0.5) a_size = -0.5;

            const double nu = mu + a_size * (1.0 - mu*mu);
            const double s = 0.5 * (1.0 - becke_polynomial_3(nu));
            P[a] *= s;
        }
    }

    double sum = 0.0;
    for (std::size_t a = 0; a < n; ++a) sum += P[a];

    std::vector<double> W(n, 0.0);
    if (sum < 1.0e-300) {
        // Degenerate -- assign to nearest atom.
        std::size_t nearest = 0;
        for (std::size_t a = 1; a < n; ++a)
            if (dist[a] < dist[nearest]) nearest = a;
        W[nearest] = 1.0;
    } else {
        const double inv_sum = 1.0 / sum;
        for (std::size_t a = 0; a < n; ++a) W[a] = P[a] * inv_sum;
    }
    return W;
}

} // anonymous namespace

// -----------------------------------------------------------------------------
// Public Lebedev accessor (also expanded form for tests)
// -----------------------------------------------------------------------------

const std::vector<GridPoint>& get_lebedev_grid(LebedevOrder order)
{
    static std::vector<GridPoint> cached_110;
    static std::vector<GridPoint> cached_194;
    static std::vector<GridPoint> cached_302;

    auto build = [](const std::vector<LebedevOrbit>& orbits,
                    std::vector<GridPoint>& cache)
    {
        if (!cache.empty()) return;
        for (const auto& o : orbits) emit_orbit(o, cache);
    };

    switch (order) {
    case LebedevOrder::L110: build(LEBEDEV_110, cached_110); return cached_110;
    case LebedevOrder::L194: build(LEBEDEV_194, cached_194); return cached_194;
    case LebedevOrder::L302: build(LEBEDEV_302, cached_302); return cached_302;
    }
    THROW_EXCEPTION("get_lebedev_grid: unknown LebedevOrder");
}

// -----------------------------------------------------------------------------
// Driver: build_molecular_grid
// -----------------------------------------------------------------------------

MolecularGrid build_molecular_grid(const std::vector<Atom>& atoms,
                                    const ThcGridOptions& options)
{
    MolecularGrid grid;
    grid.atom_ranges.reserve(atoms.size());

    const auto& angular = get_lebedev_grid(options.lebedev);

    std::vector<real_t> r_radial, w_radial;

    for (std::size_t a = 0; a < atoms.size(); ++a) {
        const auto& atom = atoms[a];
        treutler_ahlrichs_m3(atom.atomic_number, options.n_radial, r_radial, w_radial);

        MolecularGrid::AtomRange range;
        range.start = grid.points.size();

        for (int k = 0; k < options.n_radial; ++k) {
            const double r = static_cast<double>(r_radial[k]);
            const double w_rad = static_cast<double>(w_radial[k]);
            if (w_rad == 0.0) continue;

            for (const auto& ang : angular) {
                const Coordinate p {
                    static_cast<real_t>(atom.coordinate.x + r * ang.x),
                    static_cast<real_t>(atom.coordinate.y + r * ang.y),
                    static_cast<real_t>(atom.coordinate.z + r * ang.z)
                };

                const auto W = becke_partition_weights(p, atoms);
                const double w_total = w_rad * static_cast<double>(ang.w) * W[a];
                if (std::abs(w_total) < static_cast<double>(options.weight_eps)) continue;

                grid.points.push_back(GridPoint{
                    p.x, p.y, p.z, static_cast<real_t>(w_total),
                    static_cast<int>(a)});
            }
        }

        range.count = grid.points.size() - range.start;
        grid.atom_ranges.push_back(range);
    }

    grid.num_points = grid.points.size();
    return grid;
}

// -----------------------------------------------------------------------------
// Device buffer management (RAII)
// -----------------------------------------------------------------------------

void MolecularGrid::host_to_device()
{
#ifndef GANSU_CPU_ONLY
    release_device();

    if (num_points == 0) return;

    const std::size_t bytes = num_points * sizeof(real_t);
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_z, bytes);
    cudaMalloc(&d_w, bytes);

    std::vector<real_t> hx(num_points), hy(num_points), hz(num_points), hw(num_points);
    for (std::size_t i = 0; i < num_points; ++i) {
        hx[i] = points[i].x;
        hy[i] = points[i].y;
        hz[i] = points[i].z;
        hw[i] = points[i].w;
    }

    cudaMemcpy(d_x, hx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, hy.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, hz.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, hw.data(), bytes, cudaMemcpyHostToDevice);
#else
    (void)num_points;
#endif
}

void MolecularGrid::release_device()
{
#ifndef GANSU_CPU_ONLY
    if (d_x) { cudaFree(d_x); d_x = nullptr; }
    if (d_y) { cudaFree(d_y); d_y = nullptr; }
    if (d_z) { cudaFree(d_z); d_z = nullptr; }
    if (d_w) { cudaFree(d_w); d_w = nullptr; }
#endif
}

MolecularGrid::~MolecularGrid()
{
    release_device();
}

MolecularGrid::MolecularGrid(MolecularGrid&& other) noexcept
    : points(std::move(other.points))
    , num_points(other.num_points)
    , atom_ranges(std::move(other.atom_ranges))
    , d_x(other.d_x), d_y(other.d_y), d_z(other.d_z), d_w(other.d_w)
{
    other.num_points = 0;
    other.d_x = other.d_y = other.d_z = other.d_w = nullptr;
}

MolecularGrid& MolecularGrid::operator=(MolecularGrid&& other) noexcept
{
    if (this != &other) {
        release_device();
        points = std::move(other.points);
        num_points = other.num_points;
        atom_ranges = std::move(other.atom_ranges);
        d_x = other.d_x; d_y = other.d_y; d_z = other.d_z; d_w = other.d_w;
        other.num_points = 0;
        other.d_x = other.d_y = other.d_z = other.d_w = nullptr;
    }
    return *this;
}

} // namespace gansu
