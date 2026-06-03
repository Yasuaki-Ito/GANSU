/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file spherical_transform.cpp
 * @brief Cartesian → Spherical basis transformation routines (host).
 *
 * Implements forward-only (Cart → Sph) transformations for matrices (S, T,
 * V, H_core, F) and the 4-index ERI tensor.  Host implementation using
 * Eigen DGEMM (BLAS) for each of the 4 sequential single-index transforms,
 * with OpenMP parallelism across the outer non-contracted axis when the
 * contracted index is not the leading dimension.
 *
 * For 2-index matrices:
 *     M_sph[i,j] = Σ_kl U_block[i,k] · M_cart[k,l] · U_block[j,l]
 *
 * For 4-index ERI, 4 sequential single-index transforms:
 *     T1[p,q',r',s'] = Σ_{p'} U[p,p'] · ERI_cart[p',q',r',s']
 *     T2[p,q,r',s']  = Σ_{q'} U[q,q'] · T1[p,q',r',s']
 *     T3[p,q,r,s']   = Σ_{r'} U[r,r'] · T2[p,q,r',s']
 *     ERI_sph[p,q,r,s] = Σ_{s'} U[s,s'] · T3[p,q,r,s']
 *
 * Per stage: build a dense (nbf_sph × nbf_cart) matrix U_full (block-diagonal
 * in shells, ~16K entries for cc-pVQZ-class basis) and apply one DGEMM
 * (Stages 1 & 4 — contracted index leads or trails) or one DGEMM per outer
 * slice (Stages 2 & 3) with OpenMP parallelism over the outer index.
 *
 * Complexity unchanged O(nbf_cart^5) per 4-index transform, but per stage
 * is dominated by BLAS DGEMM (16-32× speedup vs scalar loops on multi-core
 * with OpenBLAS).
 */

#include "spherical_transform.hpp"

#include <Eigen/Dense>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace gansu::spherical {

namespace {

using RowMatrix =
    Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Build a dense U_full[nbf_sph, nbf_cart] block-diagonal matrix from the
// per-shell U_L matrices.  Zero everywhere except the per-shell blocks.
RowMatrix build_U_full(const std::vector<int>& shell_types,
                       const std::vector<int>& shell_offsets_cart,
                       const std::vector<int>& shell_offsets_sph) {
    const int n_shells = (int)shell_types.size();
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];
    RowMatrix U_full = RowMatrix::Zero(nbf_sph, nbf_cart);
    for (int i_shell = 0; i_shell < n_shells; ++i_shell) {
        const auto U_i     = get_cart_to_sph_matrix(shell_types[i_shell]);
        const int n_sph_i  = (int)U_i.size();
        const int n_cart_i = (int)U_i[0].size();
        const int sph_off  = shell_offsets_sph[i_shell];
        const int cart_off = shell_offsets_cart[i_shell];
        for (int a = 0; a < n_sph_i; ++a)
            for (int b = 0; b < n_cart_i; ++b)
                U_full(sph_off + a, cart_off + b) = U_i[a][b];
    }
    return U_full;
}

} // anonymous namespace


void transform_matrix_cart_to_sph(
    const real_t* M_cart,
    real_t* M_sph,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_matrix_cart_to_sph: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    const RowMatrix U_full = build_U_full(shell_types, shell_offsets_cart,
                                          shell_offsets_sph);

    // M_sph = U_full · M_cart · U_full^T   (one DGEMM per stage via Eigen/BLAS)
    Eigen::Map<const RowMatrix> M_cart_mat(M_cart, nbf_cart, nbf_cart);
    Eigen::Map<RowMatrix>       M_sph_mat (M_sph,  nbf_sph,  nbf_sph);
    M_sph_mat.noalias() = U_full * M_cart_mat * U_full.transpose();
}


void transform_matrix_sph_to_cart(
    const real_t* M_sph,
    real_t* M_cart,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_matrix_sph_to_cart: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    const RowMatrix U_full = build_U_full(shell_types, shell_offsets_cart,
                                          shell_offsets_sph);

    // M_cart = U_full^T · M_sph · U_full  (contravariant back-transform; the
    // adjoint of the covariant cart→sph operator transform).
    Eigen::Map<const RowMatrix> M_sph_mat (M_sph,  nbf_sph,  nbf_sph);
    Eigen::Map<RowMatrix>       M_cart_mat(M_cart, nbf_cart, nbf_cart);
    M_cart_mat.noalias() = U_full.transpose() * M_sph_mat * U_full;
}


void transform_eri_cart_to_sph(
    const real_t* ERI_cart,
    real_t* ERI_sph,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_eri_cart_to_sph: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    const RowMatrix U_full = build_U_full(shell_types, shell_offsets_cart,
                                          shell_offsets_sph);

    const size_t nc = (size_t)nbf_cart;
    const size_t ns = (size_t)nbf_sph;

    // ============================================================
    // Stage 1: T1[p_s, q_c, r_c, s_c] = U[p_s, p_c] · ERI[p_c, q_c, r_c, s_c]
    //   ERI viewed as (nc) × (nc·nc·nc) row-major matrix
    //   T1  viewed as (ns) × (nc·nc·nc) row-major matrix
    //   Single DGEMM: M = ns, K = nc, N = nc³
    // ============================================================
    std::vector<real_t> T1(ns * nc * nc * nc);
    {
        Eigen::Map<const RowMatrix> ERI_mat(ERI_cart, nc, nc * nc * nc);
        Eigen::Map<RowMatrix>       T1_mat (T1.data(), ns, nc * nc * nc);
        T1_mat.noalias() = U_full * ERI_mat;
    }

    // ============================================================
    // Stage 2: T2[p_s, q_s, r_c, s_c] = U[q_s, q_c] · T1[p_s, q_c, r_c, s_c]
    //   Contracted axis is the 2nd (q_c), not leading.  Loop over p_s as
    //   outer index; for each p_s, view the slab T1[p_s, :, :, :] as
    //   (nc) × (nc²) and do a single DGEMM with U.
    //   Each DGEMM is independent → OpenMP parallel over p_s.
    // ============================================================
    std::vector<real_t> T2(ns * ns * nc * nc);
    {
        const size_t slab_in  = nc * nc * nc;        // T1[p_s, :, :, :] size
        const size_t slab_out = ns * nc * nc;        // T2[p_s, :, :, :] size
        #pragma omp parallel for schedule(static)
        for (int p_s = 0; p_s < nbf_sph; ++p_s) {
            Eigen::Map<const RowMatrix> T1_slab(
                T1.data() + (size_t)p_s * slab_in, nc, nc * nc);
            Eigen::Map<RowMatrix>       T2_slab(
                T2.data() + (size_t)p_s * slab_out, ns, nc * nc);
            T2_slab.noalias() = U_full * T1_slab;
        }
    }

    // ============================================================
    // Stage 3: T3[p_s, q_s, r_s, s_c] = U[r_s, r_c] · T2[p_s, q_s, r_c, s_c]
    //   Same pattern as Stage 2 but contraction now on the 3rd axis (r_c).
    //   Outer is (p_s, q_s) flat → ns·ns independent DGEMMs of shape
    //   (ns × nc × nc).  OpenMP collapse(2).
    // ============================================================
    std::vector<real_t> T3(ns * ns * ns * nc);
    {
        const size_t slab_in  = nc * nc;             // T2[p_s, q_s, :, :] size
        const size_t slab_out = ns * nc;             // T3[p_s, q_s, :, :] size
        #pragma omp parallel for collapse(2) schedule(static)
        for (int p_s = 0; p_s < nbf_sph; ++p_s)
            for (int q_s = 0; q_s < nbf_sph; ++q_s) {
                const size_t in_off  = ((size_t)p_s * ns + q_s) * slab_in;
                const size_t out_off = ((size_t)p_s * ns + q_s) * slab_out;
                Eigen::Map<const RowMatrix> T2_slab(T2.data() + in_off,  nc, nc);
                Eigen::Map<RowMatrix>       T3_slab(T3.data() + out_off, ns, nc);
                T3_slab.noalias() = U_full * T2_slab;
            }
    }

    // ============================================================
    // Stage 4: ERI_sph[p_s, q_s, r_s, s_s] = Σ_{s_c} U[s_s, s_c]·T3[p_s, q_s, r_s, s_c]
    //   Contracted axis is the last (s_c) → trailing index.
    //   View T3 as (ns³) × (nc) and ERI_sph as (ns³) × (ns).
    //   ERI_sph = T3 · U^T  (single DGEMM with K = nc, M = ns³, N = ns)
    // ============================================================
    {
        Eigen::Map<const RowMatrix> T3_mat (T3.data(), ns * ns * ns, nc);
        Eigen::Map<RowMatrix>       ERI_sph_mat(ERI_sph, ns * ns * ns, ns);
        ERI_sph_mat.noalias() = T3_mat * U_full.transpose();
    }
}

} // namespace gansu::spherical
