/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_collocation.cu
 * @brief CPU reference implementation of the THC collocation matrix.
 *
 * The GPU kernel will mirror this loop structure once the CPU version is
 * cross-checked against GANSU's analytic overlap matrix.
 */

#include "thc_collocation.hpp"
#include <memory>

#include <cmath>
#include <cassert>
#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_manager.hpp"
#include "spherical_transform.hpp"  // Cart→Sph AO-axis transform of the collocation
#include <map>
#include <iostream>
#include <stdexcept>
#endif

namespace gansu {

namespace {

// Cartesian polynomial powers in GANSU canonical order, matching
//   - include/types.hpp::AngularMomentums (used by basis_set normalisation)
//   - include/int2e.hpp::loop_to_ang     (used by all integral kernels)
//
// CAUTION: this is *not* the lex-descending order produced by
// minao::get_cartesian_components.  The two orders agree for L<=1 but diverge
// at d (L=2): GANSU groups (xx, yy, zz) before (xy, xz, yz), while lex-desc
// interleaves them.  Using lex-desc here gives correct s/p but garbage d/f.
//
// Returns 3 * (L+1)*(L+2)/2 ints, flat (lx, ly, lz, lx, ly, lz, ...).
inline void cartesian_powers(int L, int* out)
{
    static const int s_powers[1][3]  = { {0,0,0} };
    static const int p_powers[3][3]  = { {1,0,0}, {0,1,0}, {0,0,1} };
    static const int d_powers[6][3]  = { {2,0,0}, {0,2,0}, {0,0,2},
                                         {1,1,0}, {1,0,1}, {0,1,1} };
    static const int f_powers[10][3] = { {3,0,0}, {0,3,0}, {0,0,3},
                                         {1,2,0}, {2,1,0}, {2,0,1},
                                         {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1} };
    static const int g_powers[15][3] = { {4,0,0}, {0,4,0}, {0,0,4},
                                         {3,1,0}, {3,0,1}, {1,3,0}, {0,3,1},
                                         {1,0,3}, {0,1,3},
                                         {2,2,0}, {2,0,2}, {0,2,2},
                                         {2,1,1}, {1,2,1}, {1,1,2} };
    const int (*src)[3] = nullptr;
    int n_comps = 0;
    switch (L) {
        case 0: src = s_powers; n_comps = 1;  break;
        case 1: src = p_powers; n_comps = 3;  break;
        case 2: src = d_powers; n_comps = 6;  break;
        case 3: src = f_powers; n_comps = 10; break;
        case 4: src = g_powers; n_comps = 15; break;
        default:
            // h, i, ... not yet wired here; fall back to the generic
            // descending order, which agrees with neither GANSU nor lex-desc.
            // The assert below makes sure callers don't reach this path.
            break;
    }
    assert(src != nullptr && "thc_collocation: shell L >= 5 not yet supported");
    for (int c = 0; c < n_comps; ++c) {
        out[3*c + 0] = src[c][0];
        out[3*c + 1] = src[c][1];
        out[3*c + 2] = src[c][2];
    }
}

inline real_t pow_int(real_t x, int n)
{
    real_t r = 1.0;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
}

// Double factorial (2k-1)!! with the convention (-1)!! = 1.  Matches
// factorial2_gpu in include/int2e.hpp.
inline double dfact2_2km1(int k)
{
    if (k <= 0) return 1.0;
    double v = 1.0;
    for (int i = 2*k - 1; i > 0; i -= 2) v *= static_cast<double>(i);
    return v;
}

// Per-primitive, per-Cartesian-component GTO normalisation factor used by
// GANSU's integral kernels (calcNorm() in include/int2e.hpp):
//
//   N_prim(alpha, l, m, n)
//      = 2^(l+m+n) / sqrt((2l-1)!! (2m-1)!! (2n-1)!!)
//      * (2/pi)^(3/4)
//      * alpha^((2(l+m+n)+3)/4)
//
// The full GANSU AO is
//   phi_mu(r) = N_cgto[mu] * sum_p c_p * N_prim(alpha_p, lx, ly, lz)
//                          * (x-Ax)^lx (y-Ay)^ly (z-Az)^lz * exp(-alpha_p r_A^2)
inline double prim_norm_factor(double alpha, int lx, int ly, int lz)
{
    static const double pi34 = std::pow(2.0 / M_PI, 0.75); // (2/pi)^(3/4) = PI3_4
    const int L = lx + ly + lz;
    const double pow2L = static_cast<double>(1u << L); // 2^L
    const double dfprod = dfact2_2km1(lx) * dfact2_2km1(ly) * dfact2_2km1(lz);
    return pow2L / std::sqrt(dfprod) * pi34
           * std::pow(alpha, (2.0 * L + 3.0) / 4.0);
}

} // anonymous namespace

std::vector<real_t> compute_X_ao_cpu(int N_bas,
                                     const std::vector<PrimitiveShell>& prims,
                                     const std::vector<real_t>& norms,
                                     const MolecularGrid& grid)
{
    const std::size_t N_g = grid.num_points;

    std::vector<real_t> X(static_cast<std::size_t>(N_bas) * N_g, 0.0);

    // Buffer for cartesian powers up to a generous max L.
    constexpr int MAX_L = 7;
    constexpr int MAX_COMPS = (MAX_L + 1) * (MAX_L + 2) / 2;
    int cart_pow[3 * MAX_COMPS];

    for (std::size_t P = 0; P < N_g; ++P) {
        const real_t rx = grid.points[P].x;
        const real_t ry = grid.points[P].y;
        const real_t rz = grid.points[P].z;
        real_t* X_col = X.data() + P * static_cast<std::size_t>(N_bas);

        for (const auto& prim : prims) {
            const real_t dx = rx - prim.coordinate.x;
            const real_t dy = ry - prim.coordinate.y;
            const real_t dz = rz - prim.coordinate.z;
            const real_t r2 = dx*dx + dy*dy + dz*dz;
            const real_t radial = prim.coefficient * std::exp(-prim.exponent * r2);

            const int L = prim.shell_type;
            assert(L <= MAX_L);
            const int n_comps = (L + 1) * (L + 2) / 2;
            cartesian_powers(L, cart_pow);

            for (int c = 0; c < n_comps; ++c) {
                const int lx = cart_pow[3*c + 0];
                const int ly = cart_pow[3*c + 1];
                const int lz = cart_pow[3*c + 2];
                const std::size_t mu = prim.basis_index + c;
                const real_t poly = pow_int(dx, lx) * pow_int(dy, ly) * pow_int(dz, lz);
                const real_t Nprim = static_cast<real_t>(
                    prim_norm_factor(prim.exponent, lx, ly, lz));
                X_col[mu] += radial * Nprim * poly * norms[mu];
            }
        }
    }

    return X;
}

std::vector<real_t> build_overlap_thc_cpu(const std::vector<real_t>& X_ao,
                                          const MolecularGrid& grid,
                                          int N_bas)
{
    const std::size_t N_g = grid.num_points;
    std::vector<real_t> S(static_cast<std::size_t>(N_bas) * N_bas, 0.0);

    for (std::size_t P = 0; P < N_g; ++P) {
        const real_t w = grid.points[P].w;
        const real_t* col = X_ao.data() + P * static_cast<std::size_t>(N_bas);
        for (int nu = 0; nu < N_bas; ++nu) {
            const real_t xn = col[nu];
            if (xn == 0.0) continue;
            const real_t wxn = w * xn;
            for (int mu = 0; mu < N_bas; ++mu) {
                S[mu + nu * static_cast<std::size_t>(N_bas)] += col[mu] * wxn;
            }
        }
    }

    return S;
}

#ifndef GANSU_CPU_ONLY

// =============================================================================
// GPU collocation kernel.
//
// One thread per grid point P; thread loops over all primitive shells, and for
// each primitive expands all (lx, ly, lz) Cartesian components and accumulates
// into X[mu + P*N_bas].  Same math as compute_X_ao_cpu().
//
// Cartesian-component table is replicated as a __constant__ array; matches
// AngularMomentums in include/types.hpp (NOT minao's lex-desc ordering).
// =============================================================================

namespace {

__constant__ int kThcLoopToAng[5][15][3] = {
    // L = 0 (1 component)
    {{0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
    // L = 1 (3 components)
    {{1,0,0}, {0,1,0}, {0,0,1}, {0,0,0}, {0,0,0},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
    // L = 2 (6 components)
    {{2,0,0}, {0,2,0}, {0,0,2}, {1,1,0}, {1,0,1},
     {0,1,1}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
    // L = 3 (10 components)
    {{3,0,0}, {0,3,0}, {0,0,3}, {1,2,0}, {2,1,0},
     {2,0,1}, {1,0,2}, {0,1,2}, {0,2,1}, {1,1,1},
     {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}, {0,0,0}},
    // L = 4 (15 components)
    {{4,0,0}, {0,4,0}, {0,0,4}, {3,1,0}, {3,0,1},
     {1,3,0}, {0,3,1}, {1,0,3}, {0,1,3}, {2,2,0},
     {2,0,2}, {0,2,2}, {2,1,1}, {1,2,1}, {1,1,2}}
};

__device__ __forceinline__ int n_comps_for(int L)
{
    return (L + 1) * (L + 2) / 2;
}

__device__ __forceinline__ double dfact_2km1_dev(int k)
{
    if (k <= 0) return 1.0;
    double v = 1.0;
    for (int i = 2*k - 1; i > 0; i -= 2) v *= static_cast<double>(i);
    return v;
}

// Per-primitive per-component normalisation (matches calcNorm in int2e.hpp /
// prim_norm_factor in the CPU path):
//   N(alpha, l, m, n) = 2^(l+m+n) / sqrt((2l-1)!! (2m-1)!! (2n-1)!!)
//                     * (2/pi)^(3/4)
//                     * alpha^((2(l+m+n)+3)/4)
__device__ __forceinline__ double prim_norm_factor_dev(double alpha,
                                                       int lx, int ly, int lz)
{
    const double pi34 = 0.71270547035499016035339845; // (2/pi)^(3/4)
    const int L = lx + ly + lz;
    const double pow2L = static_cast<double>(1u << L);
    const double dfprod = dfact_2km1_dev(lx) * dfact_2km1_dev(ly) * dfact_2km1_dev(lz);
    return pow2L / std::sqrt(dfprod) * pi34
           * std::pow(alpha, (2.0 * L + 3.0) / 4.0);
}

__device__ __forceinline__ double pow_int_dev(double x, int n)
{
    double r = 1.0;
    for (int i = 0; i < n; ++i) r *= x;
    return r;
}

__global__ void thc_collocation_kernel(
    int N_bas, int N_g, int N_prim,
    const real_t* __restrict__ d_grid_x,
    const real_t* __restrict__ d_grid_y,
    const real_t* __restrict__ d_grid_z,
    const PrimitiveShell* __restrict__ d_prims,
    const real_t* __restrict__ d_norms,
    real_t* d_X)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    if (P >= N_g) return;

    const real_t rx = d_grid_x[P];
    const real_t ry = d_grid_y[P];
    const real_t rz = d_grid_z[P];

    real_t* X_col = d_X + static_cast<std::size_t>(P) * N_bas;

    for (int s = 0; s < N_prim; ++s) {
        const PrimitiveShell prim = d_prims[s];
        const real_t dx = rx - prim.coordinate.x;
        const real_t dy = ry - prim.coordinate.y;
        const real_t dz = rz - prim.coordinate.z;
        const real_t r2 = dx*dx + dy*dy + dz*dz;
        const real_t radial = prim.coefficient * exp(-prim.exponent * r2);

        const int L = prim.shell_type;
        const int n_comps = n_comps_for(L);
        const std::size_t base = prim.basis_index;

        for (int c = 0; c < n_comps; ++c) {
            const int lx = kThcLoopToAng[L][c][0];
            const int ly = kThcLoopToAng[L][c][1];
            const int lz = kThcLoopToAng[L][c][2];
            const double Nprim = prim_norm_factor_dev(prim.exponent, lx, ly, lz);
            const double poly  = pow_int_dev(dx, lx) * pow_int_dev(dy, ly)
                                * pow_int_dev(dz, lz);
            X_col[base + c] += static_cast<real_t>(radial * Nprim * poly)
                              * d_norms[base + c];
        }
    }
}

} // anonymous namespace

void compute_X_ao_gpu_impl(int N_bas, int N_g, int N_prim,
                            const real_t* d_grid_x,
                            const real_t* d_grid_y,
                            const real_t* d_grid_z,
                            const PrimitiveShell* d_prims,
                            const real_t* d_cgto_norms,
                            real_t* d_X_out)
{
    cudaMemset(d_X_out, 0,
               static_cast<std::size_t>(N_bas) * N_g * sizeof(real_t));

    const int threads = 128;
    const int blocks  = (N_g + threads - 1) / threads;
    thc_collocation_kernel<<<blocks, threads>>>(
        N_bas, N_g, N_prim,
        d_grid_x, d_grid_y, d_grid_z,
        d_prims, d_cgto_norms, d_X_out);

    cudaDeviceSynchronize();
}

std::unique_ptr<DeviceHostMatrix<real_t>>
compute_X_ao_gpu(int N_bas,
                 const std::vector<PrimitiveShell>& prims,
                 const std::vector<real_t>& norms,
                 MolecularGrid& grid)
{
    if (grid.d_x == nullptr) grid.host_to_device();

    const int N_g   = static_cast<int>(grid.num_points);
    const int N_prim = static_cast<int>(prims.size());

    // The collocation kernel is inherently Cartesian: it writes X[basis_index+c]
    // with the Cartesian basis_index and indexes the Cartesian-sized cgto norms.
    // Reconstruct the contracted-shell layout from the primitive shells to get
    // the true Cartesian count nc_orb; if the requested N_bas is smaller (the
    // Spherical count) we build X in Cartesian and transform the AO axis
    // Cart→Sph (X_sph = U · X_cart, U block-diagonal per shell).  `norms` is the
    // Cartesian-sized cgto_normalization_factors regardless.
    std::map<size_t,int> shell_L;
    for (const auto& p : prims) shell_L[p.basis_index] = p.shell_type;
    std::vector<int> sh_types, off_cart, off_sph;
    off_cart.push_back(0); off_sph.push_back(0);
    int nc_orb = 0, ns_orb = 0;
    for (const auto& kv : shell_L) {
        const int L = kv.second;
        sh_types.push_back(L);
        nc_orb += (L + 1) * (L + 2) / 2;
        ns_orb += 2 * L + 1;
        off_cart.push_back(nc_orb);
        off_sph.push_back(ns_orb);
    }
    const bool spherical = (N_bas != nc_orb);   // N_bas == ns_orb < nc_orb in spherical mode

    // Mirror primitive shells + (Cartesian-sized) cgto norms to device.
    PrimitiveShell* d_prims = nullptr;
    cudaMalloc(&d_prims, N_prim * sizeof(PrimitiveShell));
    cudaMemcpy(d_prims, prims.data(),
               N_prim * sizeof(PrimitiveShell), cudaMemcpyHostToDevice);

    real_t* d_norms = nullptr;
    cudaMalloc(&d_norms, (size_t)nc_orb * sizeof(real_t));
    cudaMemcpy(d_norms, norms.data(),
               (size_t)nc_orb * sizeof(real_t), cudaMemcpyHostToDevice);

    if (!spherical) {
        auto X = std::make_unique<DeviceHostMatrix<real_t>>(N_bas, N_g);
        compute_X_ao_gpu_impl(N_bas, N_g, N_prim,
                              grid.d_x, grid.d_y, grid.d_z,
                              d_prims, d_norms, X->device_ptr());
        cudaFree(d_prims);
        cudaFree(d_norms);
        return X;
    }

    // Spherical: build X_cart [nc_orb × N_g], then X_sph = U · X_cart.
    real_t* d_X_cart = nullptr;
    cudaMalloc(&d_X_cart, (size_t)nc_orb * N_g * sizeof(real_t));
    compute_X_ao_gpu_impl(nc_orb, N_g, N_prim,
                          grid.d_x, grid.d_y, grid.d_z,
                          d_prims, d_norms, d_X_cart);

    real_t* d_U = nullptr;   // block-diagonal U [ns_orb × nc_orb] (row-major)
    spherical::build_cart_to_sph_U_device(sh_types, off_cart, off_sph, &d_U);

    auto X = std::make_unique<DeviceHostMatrix<real_t>>(N_bas, N_g);  // N_bas == ns_orb
    {
        // X_sph[ns × N_g] (col-major) = U[ns × nc] · X_cart[nc × N_g].
        // d_U is row-major [ns × nc] = col-major U^T [nc × ns] → use op_A=T.
        cublasHandle_t h = gpu::GPUHandle::cublas();
        const double one = 1.0, zero = 0.0;
        cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ns_orb, N_g, nc_orb,
                    &one,
                    d_U,      nc_orb,
                    d_X_cart, nc_orb,
                    &zero,
                    X->device_ptr(), ns_orb);
        cudaDeviceSynchronize();
    }

    cudaFree(d_U);
    cudaFree(d_X_cart);
    cudaFree(d_prims);
    cudaFree(d_norms);
    return X;
}

// =============================================================================
//  Density-based grid pruning (Phase 2.3 (B))
// =============================================================================

namespace {

// rho[P] = sum_mu X[mu, P] * Y[mu, P]   (per-column dot product)
__global__ void thc_density_per_point_kernel(
    const real_t* __restrict__ d_X,
    const real_t* __restrict__ d_Y,
    real_t* d_rho,
    int N_bas, int N_g)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    if (P >= N_g) return;
    real_t acc = 0.0;
    const std::size_t base = static_cast<std::size_t>(P) * N_bas;
    for (int mu = 0; mu < N_bas; ++mu) {
        acc += d_X[base + mu] * d_Y[base + mu];
    }
    d_rho[P] = acc;
}

// keep[P] = (rho[P] > threshold) ? 1 : 0
__global__ void thc_density_mask_kernel(
    const real_t* __restrict__ d_rho,
    int* d_keep,
    real_t threshold,
    int N_g)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    if (P >= N_g) return;
    d_keep[P] = (d_rho[P] > threshold) ? 1 : 0;
}

// Compact columns of X according to a CPU-side index list of kept positions.
// d_X_in : [N_bas × N_g] col-major
// d_X_out: [N_bas × N_g_kept] col-major
// d_kept_idx: int[N_g_kept] giving the original column index for each kept slot.
__global__ void thc_density_gather_kernel(
    const real_t* __restrict__ d_X_in,
    const int* __restrict__ d_kept_idx,
    real_t* d_X_out,
    int N_bas, int N_g_kept)
{
    const int Pnew = blockIdx.x * blockDim.x + threadIdx.x;
    const int mu   = blockIdx.y * blockDim.y + threadIdx.y;
    if (Pnew >= N_g_kept || mu >= N_bas) return;
    const int Pold = d_kept_idx[Pnew];
    d_X_out[mu + static_cast<std::size_t>(Pnew) * N_bas] =
        d_X_in[mu + static_cast<std::size_t>(Pold) * N_bas];
}

} // anonymous namespace

std::unique_ptr<DeviceHostMatrix<real_t>>
prune_X_by_density_gpu(const real_t* d_X_ao,
                       const real_t* d_density,
                       int N_bas, int N_g,
                       real_t threshold,
                       int* N_g_kept_out)
{
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;

    if (threshold <= 0.0) {
        // Disabled: just deep-copy X.
        auto X_out = std::make_unique<DeviceHostMatrix<real_t>>(N_bas, N_g);
        cudaMemcpy(X_out->device_ptr(), d_X_ao,
                   static_cast<std::size_t>(N_bas) * N_g * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        if (N_g_kept_out) *N_g_kept_out = N_g;
        return X_out;
    }

    // 1) Y = P × X  (N_bas × N_g)
    real_t* d_Y = nullptr;
    cudaMalloc(&d_Y, static_cast<std::size_t>(N_bas) * N_g * sizeof(real_t));
    if (cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     N_bas, N_g, N_bas,
                     &one, d_density, N_bas, d_X_ao, N_bas,
                     &zero, d_Y, N_bas) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_Y);
        throw std::runtime_error("prune_X_by_density_gpu: P × X DGEMM failed");
    }

    // 2) rho[P] = column-dot(X[:,P], Y[:,P])
    real_t* d_rho = nullptr;
    cudaMalloc(&d_rho, N_g * sizeof(real_t));
    {
        const int threads = 128;
        const int blocks  = (N_g + threads - 1) / threads;
        thc_density_per_point_kernel<<<blocks, threads>>>(
            d_X_ao, d_Y, d_rho, N_bas, N_g);
    }
    cudaFree(d_Y);

    // 3) mask + host-side compaction (cheap: N_g is at most a few × 10⁵)
    int* d_keep = nullptr;
    cudaMalloc(&d_keep, N_g * sizeof(int));
    {
        const int threads = 256;
        const int blocks  = (N_g + threads - 1) / threads;
        thc_density_mask_kernel<<<blocks, threads>>>(
            d_rho, d_keep, threshold, N_g);
    }

    std::vector<int> h_keep(N_g);
    cudaMemcpy(h_keep.data(), d_keep, N_g * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_keep);
    cudaFree(d_rho);

    std::vector<int> kept_idx;
    kept_idx.reserve(N_g);
    for (int P = 0; P < N_g; ++P) if (h_keep[P]) kept_idx.push_back(P);
    const int N_g_kept = static_cast<int>(kept_idx.size());
    if (N_g_kept_out) *N_g_kept_out = N_g_kept;

    if (N_g_kept == 0) {
        throw std::runtime_error("prune_X_by_density_gpu: zero grid points retained "
                                  "(threshold too high?)");
    }

    // 4) gather columns into a fresh DeviceHostMatrix (N_bas × N_g_kept)
    int* d_kept_idx = nullptr;
    cudaMalloc(&d_kept_idx, N_g_kept * sizeof(int));
    cudaMemcpy(d_kept_idx, kept_idx.data(),
               N_g_kept * sizeof(int), cudaMemcpyHostToDevice);

    auto X_out = std::make_unique<DeviceHostMatrix<real_t>>(N_bas, N_g_kept);
    {
        const dim3 threads(16, 16);
        const dim3 blocks((N_g_kept + 15) / 16, (N_bas + 15) / 16);
        thc_density_gather_kernel<<<blocks, threads>>>(
            d_X_ao, d_kept_idx, X_out->device_ptr(),
            N_bas, N_g_kept);
    }
    cudaFree(d_kept_idx);

    return X_out;
}

#endif // GANSU_CPU_ONLY

} // namespace gansu
