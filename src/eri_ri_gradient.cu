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

#ifdef GANSU_CPU_ONLY
#include "cuda_compat.hpp"
#else
#include <cuda.h>
#include <cublas_v2.h>
#endif
#include <cmath>
#include <vector>
#include <stdexcept>

#include <Eigen/Dense>

#include "eri.hpp"
#include "hf.hpp"
#include "gpu_manager.hpp"
#include "gradients.hpp"
#include "grad_2c.hpp"
#include "grad_3c.hpp"

namespace gansu {

// ============================================================================
// Small device kernel: Γ^(3)_{P,μν} = w_P · D_{μν} − ½ · A_{P,μν}
//
// Layout: A and Γ^(3) are row-major [naux × nbas²] flat. D is (nbas, nbas).
// One thread per (P, idx) element of Γ^(3).
// ============================================================================
#ifndef GANSU_CPU_ONLY
__global__
void k_assemble_gamma3(real_t* d_gamma3,
                       const real_t* d_A,
                       const real_t* d_w,
                       const real_t* d_D,
                       const int naux,
                       const int nbas)
{
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
    const size_t total = (size_t)naux * nbas2;
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= total) return;

    const size_t P   = tid / nbas2;
    const size_t idx = tid % nbas2;  // μν flat index, row-major

    d_gamma3[tid] = d_w[P] * d_D[idx] - 0.5 * d_A[tid];
}
#endif

static void launch_assemble_gamma3(real_t* d_gamma3,
                                   const real_t* d_A,
                                   const real_t* d_w,
                                   const real_t* d_D,
                                   const int naux,
                                   const int nbas)
{
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        const size_t total = (size_t)naux * (size_t)nbas * (size_t)nbas;
        const int threads = 256;
        const size_t blocks = (total + threads - 1) / threads;
        k_assemble_gamma3<<<(unsigned int)blocks, threads>>>(d_gamma3, d_A, d_w, d_D, naux, nbas);
        return;
    }
#endif
    // CPU fallback (OpenMP)
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
    #pragma omp parallel for schedule(static)
    for (long long P_ll = 0; P_ll < (long long)naux; P_ll++) {
        const size_t P = (size_t)P_ll;
        for (size_t idx = 0; idx < nbas2; idx++) {
            d_gamma3[P * nbas2 + idx] = d_w[P] * d_D[idx] - 0.5 * d_A[P * nbas2 + idx];
        }
    }
}

// ============================================================================
// Apply L^{-T} to the P axis of B in-place.
// Math:  Bbar = L^{-T} B  ⇔  L^T · Bbar = B.
// L is row-major lower-triangular, shape (naux, naux). B is row-major
// (naux, nbas²); the P axis is row-major leading.
//
// cuBLAS is column-major: a row-major (naux, nbas²) matrix is, in column-major
// terms, a (nbas², naux) matrix with leading dimension nbas². We want to solve
// op(A)·X = B with op(A) = L^T (in row-major sense) which is U (upper triangle
// in column-major). One way: directly call DTRSM with SIDE=RIGHT and OP=N on
// the column-major view, which acts on the P axis.
// ============================================================================
#ifndef GANSU_CPU_ONLY
static void apply_LinvT_to_Bbar_inplace_gpu(real_t* d_L, real_t* d_Bbar,
                                            const int naux, const int nbas2)
{
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0;

    // The row-major (naux, nbas²) matrix d_Bbar, viewed in column-major, is
    // (nbas², naux) with ld = nbas². We want to apply L^{-T} on the P axis
    // (which is the trailing/column axis in column-major view).
    //
    // Solve (op_B X) = α B in cuBLAS: cublasDtrsm with SIDE=RIGHT, FILL=LOWER,
    // OP_A=OP_T, applies (op_A(L))^{-1} on the right, i.e., X · L^T = B
    // (column-major). Recall: in column-major terms, L (originally lower
    // in row-major) is upper in column-major. We pass FILL=UPPER.
    //
    // Concretely: solve  Bbar_cm · op(L_cm) = α B_cm
    //   where L_cm is column-major view of the row-major lower-triangular L,
    //   which appears as upper-triangular in column-major. op = T.
    // After this, Bbar_cm contains B_cm · (L_cm^T)^{-1}.
    // In row-major terms (transpose): Bbar_rm = (L_rm^{-T}) · B_rm. ✓
    //
    // Note: choleskyDecomposition() stores L in the lower triangle of d_L
    // (row-major). We pass FILL_MODE_UPPER to cuBLAS since the column-major
    // view inverts the fill mode.

    cublasStatus_t st = cublasDtrsm(
        handle,
        CUBLAS_SIDE_RIGHT,
        CUBLAS_FILL_MODE_UPPER,  // row-major lower → column-major upper
        CUBLAS_OP_T,             // apply L^T (column-major)
        CUBLAS_DIAG_NON_UNIT,
        nbas2, naux,             // shape of d_Bbar in column-major view
        &alpha,
        d_L, naux,               // L: (naux, naux), ld=naux
        d_Bbar, nbas2);          // Bbar: column-major view (nbas², naux), ld=nbas²
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("apply_LinvT_to_Bbar_inplace_gpu: cublasDtrsm failed");
    }
}
#endif

// ============================================================================
// In-place: w ← L^{-T} · w  (length-naux vector).
//
// Used to convert wB = B · D = L^{-1} d into the true Coulomb fitting coefficient
// γ = J^{-1} d = L^{-T} L^{-1} d = L^{-T} wB.
//
// L is row-major lower-triangular (= column-major upper-triangular). For the
// column-major view, we solve L_cm · γ_cm = w_cm with cublasDtrsv (FILL=UPPER,
// OP=N), which in row-major terms is L_rm^T · γ_rm = w_rm — exactly L^{-T}·w.
// ============================================================================
static void apply_LinvT_to_w_inplace(real_t* d_L, real_t* d_w, const int naux)
{
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        cublasStatus_t st = cublasDtrsv(
            handle,
            CUBLAS_FILL_MODE_UPPER,   // row-major lower → column-major upper
            CUBLAS_OP_N,              // L_cm · γ_cm = w_cm  ⇔  L_rm^T · γ_rm = w_rm
            CUBLAS_DIAG_NON_UNIT,
            naux,
            d_L, naux,
            d_w, 1);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("apply_LinvT_to_w_inplace: cublasDtrsv failed");
        }
        return;
    }
#endif
    using RM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using Vec = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
    Eigen::Map<RM>  L(d_L, naux, naux);
    Eigen::Map<Vec> w(d_w, naux);
    Vec g = L.transpose().triangularView<Eigen::Upper>().solve(w);
    w = g;
}

static void apply_LinvT_to_Bbar_inplace(real_t* d_L, real_t* d_Bbar,
                                         const int naux, const int nbas2)
{
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        apply_LinvT_to_Bbar_inplace_gpu(d_L, d_Bbar, naux, nbas2);
        return;
    }
#endif
    // CPU path: Eigen back-substitution.
    using RM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<RM> L(d_L, naux, naux);
    Eigen::Map<RM> Bb(d_Bbar, naux, nbas2);
    // L is lower triangular (row-major). Solve L^T · X = Bb for X.
    // Note: take Transpose view first, then select Upper triangle. Eigen's
    // TriangularView::transpose() requires an lvalue underlying matrix and
    // trips a static assert if called on a Map<const> (read-only) target.
    RM X = L.transpose().triangularView<Eigen::Upper>().solve(Bb);
    Bb = X;
}

// ============================================================================
// Build A_{P,μν} = (D · B̄_P · D)_{μν} for all P via two batched DGEMMs.
//   1) X_P = D · B̄_P    (each B̄_P is (nbas, nbas), strided)
//   2) A_P = X_P · D
// All matrices are row-major (nbas, nbas).
//
// cuBLAS column-major reinterpretation: a row-major (nbas, nbas) matrix M is
// column-major M^T, with ld = nbas. The product (row-major) X = D · B̄_P
// becomes, in column-major, X^T = B̄_P^T · D^T. We can use this with cuBLAS by
// passing OP_N for both and swapping operand order, or equivalently use OP_N
// on transposed lda. Simplest: use SIDE-aware GEMM with row-major helper.
//
// Approach: matrixMatrixProductRect already handles row-major. But it's
// per-call (not batched). For naux DGEMMs, that's naux cublas calls — fine for
// 100s of aux primitives. Use a single batched call when possible.
// ============================================================================
static void build_A_from_DBbarD(real_t* d_A,
                                const real_t* d_D,
                                const real_t* d_Bbar,
                                const int naux,
                                const int nbas)
{
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        // Per-P DGEMM pair via matrixMatrixProductRect.
        // X = D · B̄_P, then A_P = X · D.
        // X uses a temporary buffer; reuse d_A for X (its final contents are
        // overwritten by the second DGEMM).
        real_t* d_X = nullptr;
        gansu::tracked_cudaMalloc(&d_X, nbas2 * sizeof(real_t));

        for (int P = 0; P < naux; P++) {
            const real_t* d_Bbar_P = d_Bbar + (size_t)P * nbas2;
            real_t* d_A_P = d_A + (size_t)P * nbas2;

            // Step 1: X = D · B̄_P  (both nbas × nbas, row-major)
            gpu::matrixMatrixProductRect(
                d_D, d_Bbar_P, d_X,
                nbas, nbas, nbas,
                /*transpose_A=*/false, /*transpose_B=*/false,
                /*accumulate=*/false, /*alpha=*/1.0);

            // Step 2: A_P = X · D
            gpu::matrixMatrixProductRect(
                d_X, d_D, d_A_P,
                nbas, nbas, nbas,
                /*transpose_A=*/false, /*transpose_B=*/false,
                /*accumulate=*/false, /*alpha=*/1.0);
        }
        gansu::tracked_cudaFree(d_X);
        return;
    }
#endif
    // CPU path: Eigen per P.
    using RM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RM> D(d_D, nbas, nbas);
    #pragma omp parallel for schedule(static)
    for (long long P_ll = 0; P_ll < (long long)naux; P_ll++) {
        const size_t P = (size_t)P_ll;
        Eigen::Map<const RM> Bp(d_Bbar + P * nbas2, nbas, nbas);
        Eigen::Map<RM>       Ap(d_A    + P * nbas2, nbas, nbas);
        Ap = D * Bp * D;
    }
}

// ============================================================================
// Γ^(2)_{PQ} = -½ w_P w_Q + ¼ (A · B̄^T)_{PQ}
//
// A and B̄ are row-major (naux, nbas²). A·B̄^T is (naux, naux), with the
// inner contraction over the flat μν index.
//
// (A · B̄^T)_{P,Q} = Σ_{ij} A_{P,ij} B̄_{Q,ij}
// In row-major, this is `matrixMatrixProductRect(A, B̄, out, naux, naux, nbas²,
//                                                  transpose_A=false, transpose_B=true)`.
// ============================================================================
static void build_gamma2(real_t* d_gamma2,
                         const real_t* d_A,
                         const real_t* d_Bbar,
                         const real_t* d_w,
                         const int naux,
                         const int nbas)
{
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;

#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        // Step 1: Γ^(2) = ¼ A · B̄^T
        gpu::matrixMatrixProductRect(
            d_A, d_Bbar, d_gamma2,
            naux, naux, (int)nbas2,
            /*transpose_A=*/false, /*transpose_B=*/true,
            /*accumulate=*/false, /*alpha=*/0.25);

        // Step 2: Γ^(2) += −½ w wᵀ  via cublasDger
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        const double minus_half = -0.5;
        // Row-major (naux × naux) ≡ column-major (naux × naux) with ld=naux,
        // but the outer-product `w wᵀ` is the same in both layouts. cublasDger
        // expects column-major: A += α x yᵀ with x of length m, y of length n.
        // For symmetric outer product of a single vector, layout matters not.
        cublasStatus_t st = cublasDger(handle, naux, naux, &minus_half,
                                       d_w, 1, d_w, 1, d_gamma2, naux);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("build_gamma2: cublasDger failed");
        }
        return;
    }
#endif
    using RM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RM> A_mat(d_A,    naux, (int)nbas2);
    Eigen::Map<const RM> B_mat(d_Bbar, naux, (int)nbas2);
    Eigen::Map<RM>       G_mat(d_gamma2, naux, naux);
    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, 1>> w_vec(d_w, naux);
    G_mat = 0.25 * (A_mat * B_mat.transpose());
    G_mat.noalias() += -0.5 * (w_vec * w_vec.transpose());
}

// ============================================================================
// w_P = Σ_{μν} B_{P,μν} D_{μν}.  In flat: w = B · vec(D), with B row-major
// (naux, nbas²) and vec(D) of length nbas². This is a single DGEMV.
// ============================================================================
static void build_w(real_t* d_w,
                    const real_t* d_B,
                    const real_t* d_D,
                    const int naux,
                    const int nbas)
{
    const int nbas2 = nbas * nbas;
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t handle = gpu::GPUHandle::cublas();
        const double one = 1.0, zero = 0.0;
        // Row-major (naux, nbas²) B  ≡  column-major (nbas², naux) with ld=nbas².
        // We want y = B · vec(D), i.e., y_P = Σ_i B[P, i] · D[i].
        // In column-major view: y^T = vec(D)^T · B_cm, where B_cm = B^T (col-major
        // of row-major B). So y = B_cm^T · vec(D), i.e., DGEMV with OP=T and
        // m=nbas², n=naux.
        cublasStatus_t st = cublasDgemv(handle, CUBLAS_OP_T,
                                        nbas2, naux,
                                        &one, d_B, nbas2,
                                        d_D, 1,
                                        &zero, d_w, 1);
        if (st != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("build_w: cublasDgemv failed");
        }
        return;
    }
#endif
    using RM = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RM> B_mat(d_B, naux, nbas2);
    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, 1>> Dv(d_D, nbas2);
    Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>> wv(d_w, naux);
    wv = B_mat * Dv;
}

// ============================================================================
// Compute the intersection of a shell-type's primitive range
// [info.start_index, info.start_index + info.count) with the requested local
// aux range [P_local_start, P_local_end). Returns a (possibly empty) sub-shell
// ShellTypeInfo. If the intersection is empty, the result's count is 0 and
// callers should skip the kernel launch.
// ============================================================================
static inline gansu::ShellTypeInfo restrict_aux_shell_info(
    const gansu::ShellTypeInfo& info,
    const size_t P_local_start,
    const size_t P_local_end)
{
    const size_t shell_first = info.start_index;
    const size_t shell_last  = info.start_index + info.count;  // exclusive
    const size_t lo = std::max(shell_first, P_local_start);
    const size_t hi = std::min(shell_last,  P_local_end);
    gansu::ShellTypeInfo sub = info;
    if (lo >= hi) {
        sub.count = 0;
    } else {
        sub.start_index = lo;
        sub.count = (int)(hi - lo);
    }
    return sub;
}

// ============================================================================
// Launch the 3c2e gradient kernel for every (s_μ, s_ν, s_P_aux) shell-type
// combination, restricted to aux primitive indices in [P_local_start,
// P_local_end). For single-GPU callers this is the full [0, naux) range.
// The Γ^(3) symmetry in (μ, ν) means we iterate all (s_μ, s_ν) without an
// upper-triangle restriction — see grad_3c.hpp design note.
// ============================================================================
static void launch_3c2e_grad(
    double* d_grad_2el,
    const real_t* d_gamma3,
    const std::vector<gansu::ShellTypeInfo>& shell_type_infos,
    const std::vector<gansu::ShellTypeInfo>& aux_shell_type_infos,
    const gansu::PrimitiveShell* d_pshell,
    const gansu::PrimitiveShell* d_pshell_aux,
    const real_t* d_cgto,
    const real_t* d_aux_cgto,
    const int nbas, const int naux,
    const real_t* d_boys_grid,
    const size_t P_local_start = 0,
    const size_t P_local_end = static_cast<size_t>(-1))
{
    const int n_shell_mu = (int)shell_type_infos.size();
    const int n_shell_aux = (int)aux_shell_type_infos.size();
    const size_t Pend = (P_local_end == (size_t)-1) ? (size_t)naux : P_local_end;

    for (int s_mu = 0; s_mu < n_shell_mu; s_mu++) {
        for (int s_nu = 0; s_nu < n_shell_mu; s_nu++) {
            for (int s_P = 0; s_P < n_shell_aux; s_P++) {
                const auto& info_mu  = shell_type_infos[s_mu];
                const auto& info_nu  = shell_type_infos[s_nu];
                const auto info_aux_local =
                    restrict_aux_shell_info(aux_shell_type_infos[s_P], P_local_start, Pend);

                const size_t n_threads = (size_t)info_mu.count *
                                         (size_t)info_nu.count *
                                         (size_t)info_aux_local.count;
                if (n_threads == 0) continue;

#ifndef GANSU_CPU_ONLY
                if (gpu::gpu_available()) {
                    const int threads_per_block = 64;
                    const size_t blocks = (n_threads + threads_per_block - 1) / threads_per_block;
                    gpu::compute_gradients_3c2e<<<(unsigned int)blocks, threads_per_block>>>(
                        d_grad_2el, d_gamma3,
                        d_pshell, d_pshell_aux,
                        d_cgto, d_aux_cgto,
                        info_mu, info_nu, info_aux_local,
                        n_threads, nbas, naux,
                        d_boys_grid);
                } else
#endif
                {
                    gpu::compute_gradients_3c2e_cpu(
                        d_grad_2el, d_gamma3,
                        d_pshell, d_pshell_aux,
                        d_cgto, d_aux_cgto,
                        info_mu, info_nu, info_aux_local,
                        n_threads, nbas, naux,
                        d_boys_grid);
                }
            }
        }
    }
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
}

// ============================================================================
// Launch the 2c2e gradient kernel.
//
//   Single-GPU mode (P_local_start = 0, P_local_end = -1 or naux):
//     Iterate the upper triangle of (s_P, s_Q) shell-type pairs (s_P ≤ s_Q).
//     The kernel uses sym_factor = 2 for off-diagonal primitive pairs.
//
//   Distributed mode (P_local_start, P_local_end define a strict sub-range):
//     Iterate ALL (s_P, s_Q) shell-type combinations (Cartesian product),
//     restricting only s_P to local aux primitive indices. Each ordered
//     (prim_P, prim_Q) pair is visited exactly once across all GPUs combined.
//     Kernel uses sym_factor = 1 (via no_pair_symmetry = true).
// ============================================================================
static void launch_2c2e_grad(
    double* d_grad_2el,
    const real_t* d_gamma2,
    const std::vector<gansu::ShellTypeInfo>& aux_shell_type_infos,
    const gansu::PrimitiveShell* d_pshell_aux,
    const real_t* d_aux_cgto,
    const int naux,
    const real_t* d_boys_grid,
    const size_t P_local_start = 0,
    const size_t P_local_end = static_cast<size_t>(-1))
{
    const int n_shell_aux = (int)aux_shell_type_infos.size();
    const size_t Pend = (P_local_end == (size_t)-1) ? (size_t)naux : P_local_end;
    const bool distributed = (P_local_start != 0) || (Pend != (size_t)naux);

    auto launch = [&](const gansu::ShellTypeInfo& info_P,
                      const gansu::ShellTypeInfo& info_Q,
                      size_t n_threads,
                      bool no_pair_symmetry) {
        if (n_threads == 0) return;
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            const int threads_per_block = 128;
            const size_t blocks = (n_threads + threads_per_block - 1) / threads_per_block;
            gpu::compute_gradients_2c2e<<<(unsigned int)blocks, threads_per_block>>>(
                d_grad_2el, d_gamma2,
                d_pshell_aux, d_aux_cgto,
                info_P, info_Q,
                n_threads, naux,
                d_boys_grid,
                no_pair_symmetry);
        } else
#endif
        {
            gpu::compute_gradients_2c2e_cpu(
                d_grad_2el, d_gamma2,
                d_pshell_aux, d_aux_cgto,
                info_P, info_Q,
                n_threads, naux,
                d_boys_grid,
                no_pair_symmetry);
        }
    };

    if (!distributed) {
        // Single-GPU: upper triangle, factor 2 for off-diagonals (default).
        for (int s_P = 0; s_P < n_shell_aux; s_P++) {
            for (int s_Q = s_P; s_Q < n_shell_aux; s_Q++) {
                const auto& info_P = aux_shell_type_infos[s_P];
                const auto& info_Q = aux_shell_type_infos[s_Q];
                size_t n_threads;
                if (s_P == s_Q) {
                    n_threads = (size_t)info_P.count * ((size_t)info_P.count + 1) / 2;
                } else {
                    n_threads = (size_t)info_P.count * (size_t)info_Q.count;
                }
                launch(info_P, info_Q, n_threads, /*no_pair_symmetry=*/false);
            }
        }
    } else {
        // Distributed: all (s_P, s_Q) combos, P restricted to local range,
        // each ordered pair visited exactly once (combined across GPUs).
        for (int s_P = 0; s_P < n_shell_aux; s_P++) {
            const auto info_P_local =
                restrict_aux_shell_info(aux_shell_type_infos[s_P], P_local_start, Pend);
            if (info_P_local.count == 0) continue;
            for (int s_Q = 0; s_Q < n_shell_aux; s_Q++) {
                const auto& info_Q = aux_shell_type_infos[s_Q];
                const size_t n_threads = (size_t)info_P_local.count * (size_t)info_Q.count;
                launch(info_P_local, info_Q, n_threads, /*no_pair_symmetry=*/true);
            }
        }
    }

#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) cudaDeviceSynchronize();
#endif
}

// ============================================================================
// ERI_RI::compute_ri_gradient_impl — worker with extra parameters for
// the distributed override. The public compute_ri_gradient is a thin wrapper
// below.
// ============================================================================
std::vector<double> ERI_RI::compute_ri_gradient_impl(
    const real_t* d_density_matrix,
    const real_t* d_coefficient_matrix,
    const real_t* d_orbital_energies,
    const int num_electron,
    const real_t* d_B_full,
    const size_t P_local_start,
    const size_t P_local_end,
    const bool include_one_electron)
{
    const int nbas = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nbas2 = (size_t)nbas * (size_t)nbas;
    const int num_atoms = (int)hf_.get_atoms().size();
    const int n3 = 3 * num_atoms;
    const size_t grad_bytes = n3 * sizeof(double);

    // ----- Stack limit for the 3c2e gradient kernel (uses ~56KB / thread) -----
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cudaDeviceSetLimit(cudaLimitStackSize, 64 * 1024);
    }
#endif

    // ----- Allocate gradient buffers (1e/S/N/2e/total) -----
    double *d_grad_total = nullptr, *d_grad_N = nullptr, *d_grad_S = nullptr;
    double *d_grad_K = nullptr, *d_grad_V = nullptr, *d_grad_2el = nullptr;
    real_t *d_W = nullptr;
    gansu::tracked_cudaMalloc(&d_grad_total, grad_bytes);
    gansu::tracked_cudaMalloc(&d_grad_N, grad_bytes);
    gansu::tracked_cudaMalloc(&d_grad_S, grad_bytes);
    gansu::tracked_cudaMalloc(&d_grad_K, grad_bytes);
    gansu::tracked_cudaMalloc(&d_grad_V, grad_bytes);
    gansu::tracked_cudaMalloc(&d_grad_2el, grad_bytes);
    if (include_one_electron) {
        gansu::tracked_cudaMalloc(&d_W, nbas2 * sizeof(real_t));
        cudaMemset(d_W, 0, nbas2 * sizeof(real_t));
    }
    cudaMemset(d_grad_total, 0, grad_bytes);
    cudaMemset(d_grad_N,     0, grad_bytes);
    cudaMemset(d_grad_S,     0, grad_bytes);
    cudaMemset(d_grad_K,     0, grad_bytes);
    cudaMemset(d_grad_V,     0, grad_bytes);
    cudaMemset(d_grad_2el,   0, grad_bytes);

    const auto& shell_type_infos = hf_.get_shell_type_infos();
    const auto& primitive_shells = hf_.get_primitive_shells();
    const auto& cgto_norms       = hf_.get_cgto_normalization_factors();
    const auto& boys_grid        = hf_.get_boys_grid();
    const auto& atoms_dh         = hf_.get_atoms();

    if (include_one_electron) {
        // ----- W matrix (energy-weighted density) -----
        gpu::compute_W(d_W, d_coefficient_matrix, d_orbital_energies, nbas, num_electron);

        // ----- 1-electron + S + N gradient (reuse the 4c launchers) -----
        const int shell_type_count = (int)shell_type_infos.size();
        const int threads_per_block = 128;

#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            for (int s0 = shell_type_count - 1; s0 >= 0; s0--) {
                for (int s1 = shell_type_count - 1; s1 >= s0; s1--) {
                    const auto info0 = shell_type_infos[s0];
                    const auto info1 = shell_type_infos[s1];
                    const size_t n_pairs = (s0 == s1) ? (size_t)info0.count*(info0.count+1)/2
                                                      : (size_t)info0.count*info1.count;
                    if (n_pairs == 0) continue;
                    const size_t blocks = (n_pairs + threads_per_block - 1) / threads_per_block;

                    gpu::compute_gradients_overlap<<<(unsigned int)blocks, threads_per_block>>>(
                        d_grad_S, d_W, primitive_shells.device_ptr(), cgto_norms.device_ptr(),
                        nbas, info0, info1, n_pairs);
                    gpu::compute_gradients_kinetic<<<(unsigned int)blocks, threads_per_block>>>(
                        d_grad_K, d_density_matrix, primitive_shells.device_ptr(), cgto_norms.device_ptr(),
                        nbas, info0, info1, n_pairs);
                    gpu::compute_gradients_nuclear<<<(unsigned int)blocks, threads_per_block>>>(
                        d_grad_V, d_density_matrix, primitive_shells.device_ptr(), cgto_norms.device_ptr(),
                        atoms_dh.device_ptr(), num_atoms, nbas, info0, info1, n_pairs,
                        boys_grid.device_ptr());
                }
            }

            const size_t nr_threads = (size_t)num_atoms * (size_t)num_atoms;
            const size_t nr_blocks  = (nr_threads + threads_per_block - 1) / threads_per_block;
            gpu::compute_nuclear_repulsion_gradient_kernel<<<(unsigned int)nr_blocks, threads_per_block>>>(
                d_grad_N, atoms_dh.device_ptr(), num_atoms);
            cudaDeviceSynchronize();
        } else
#endif
        {
            for (int s0 = shell_type_count - 1; s0 >= 0; s0--) {
                for (int s1 = shell_type_count - 1; s1 >= s0; s1--) {
                    const auto info0 = shell_type_infos[s0];
                    const auto info1 = shell_type_infos[s1];
                    const size_t n_pairs = (s0 == s1) ? (size_t)info0.count*(info0.count+1)/2
                                                      : (size_t)info0.count*info1.count;
                    if (n_pairs == 0) continue;
                    gpu::compute_gradients_overlap_cpu(
                        d_grad_S, d_W, primitive_shells.host_ptr(), cgto_norms.host_ptr(),
                        nbas, info0, info1, n_pairs);
                    gpu::compute_gradients_kinetic_cpu(
                        d_grad_K, d_density_matrix, primitive_shells.host_ptr(), cgto_norms.host_ptr(),
                        nbas, info0, info1, n_pairs);
                    gpu::compute_gradients_nuclear_cpu(
                        d_grad_V, d_density_matrix, primitive_shells.host_ptr(), cgto_norms.host_ptr(),
                        atoms_dh.host_ptr(), num_atoms, nbas, info0, info1, n_pairs,
                        boys_grid.host_ptr());
                }
            }
            gpu::compute_nuclear_repulsion_gradient_cpu(d_grad_N, atoms_dh.host_ptr(), num_atoms);
        }
    }

    // ===================== RI 2-electron gradient =====================
    //
    // (1) Rebuild (P|Q) and Cholesky → L  (L not persisted by SCF path).
    real_t* d_L = nullptr;
    gansu::tracked_cudaMalloc(&d_L, (size_t)naux * naux * sizeof(real_t));
    cudaMemset(d_L, 0, (size_t)naux * naux * sizeof(real_t));
    gpu::computeTwoCenterERIs(
        auxiliary_shell_type_infos_,
        auxiliary_primitive_shells_.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        d_L, naux,
        boys_grid.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        hf_.get_schwarz_screening_threshold(),
        hf_.get_verbose());
    gpu::choleskyDecomposition(d_L, naux);

    // (2) B̄ = L^{-T} · B  (P-axis backward solve, in-place on a copy of B)
    real_t* d_Bbar = nullptr;
    gansu::tracked_cudaMalloc(&d_Bbar, (size_t)naux * nbas2 * sizeof(real_t));
    cudaMemcpy(d_Bbar, d_B_full,
               (size_t)naux * nbas2 * sizeof(real_t), cudaMemcpyDeviceToDevice);
    apply_LinvT_to_Bbar_inplace(d_L, d_Bbar, naux, (int)nbas2);

    // (3) γ_P = J^{-1} d_P = L^{-T} (B · D)_P  — Coulomb fitting coefficient.
    //     See note in single-GPU compute_ri_gradient about wB vs γ.
    real_t* d_gamma = nullptr;
    gansu::tracked_cudaMalloc(&d_gamma, (size_t)naux * sizeof(real_t));
    cudaMemset(d_gamma, 0, (size_t)naux * sizeof(real_t));
    build_w(d_gamma, d_B_full, d_density_matrix, naux, nbas);  // wB
    apply_LinvT_to_w_inplace(d_L, d_gamma, naux);  // wB → γ
    gansu::tracked_cudaFree(d_L);

    real_t* d_w = d_gamma;  // alias for downstream readability

    // (4) A_{P,μν} = (D · B̄_P · D)_{μν}
    real_t* d_A = nullptr;
    gansu::tracked_cudaMalloc(&d_A, (size_t)naux * nbas2 * sizeof(real_t));
    build_A_from_DBbarD(d_A, d_density_matrix, d_Bbar, naux, nbas);

    // (5) Γ^(3)_{P,μν} = w_P · D_{μν} − ½ · A_{P,μν}
    real_t* d_gamma3 = nullptr;
    gansu::tracked_cudaMalloc(&d_gamma3, (size_t)naux * nbas2 * sizeof(real_t));
    launch_assemble_gamma3(d_gamma3, d_A, d_w, d_density_matrix, naux, nbas);

    // (6) Γ^(2)_{PQ} = -½ w_P w_Q + ¼ (A · B̄^T)_{PQ}
    real_t* d_gamma2 = nullptr;
    gansu::tracked_cudaMalloc(&d_gamma2, (size_t)naux * naux * sizeof(real_t));
    build_gamma2(d_gamma2, d_A, d_Bbar, d_w, naux, nbas);

    gansu::tracked_cudaFree(d_A);
    gansu::tracked_cudaFree(d_Bbar);
    gansu::tracked_cudaFree(d_w);

    // (7) 3c2e gradient kernel: shell-type triple sweep, restricted to local P
    launch_3c2e_grad(
        d_grad_2el, d_gamma3,
        shell_type_infos, auxiliary_shell_type_infos_,
        primitive_shells.device_ptr(),
        auxiliary_primitive_shells_.device_ptr(),
        cgto_norms.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        nbas, naux, boys_grid.device_ptr(),
        P_local_start, P_local_end);

    gansu::tracked_cudaFree(d_gamma3);

    // (8) 2c2e gradient kernel: aux shell-type pair sweep, restricted to local P
    launch_2c2e_grad(
        d_grad_2el, d_gamma2,
        auxiliary_shell_type_infos_,
        auxiliary_primitive_shells_.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        naux, boys_grid.device_ptr(),
        P_local_start, P_local_end);

    gansu::tracked_cudaFree(d_gamma2);

    // ----- Sum total = N + S + K + V + 2el -----
#ifndef GANSU_CPU_ONLY
    if (gpu::gpu_available()) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        const double one = 1.0;
        cudaMemcpy(d_grad_total, d_grad_N, grad_bytes, cudaMemcpyDeviceToDevice);
        cublasDaxpy(handle, n3, &one, d_grad_S,   1, d_grad_total, 1);
        cublasDaxpy(handle, n3, &one, d_grad_K,   1, d_grad_total, 1);
        cublasDaxpy(handle, n3, &one, d_grad_V,   1, d_grad_total, 1);
        cublasDaxpy(handle, n3, &one, d_grad_2el, 1, d_grad_total, 1);
        cublasDestroy(handle);
    } else
#endif
    {
        for (int i = 0; i < n3; i++) {
            d_grad_total[i] = d_grad_N[i] + d_grad_S[i] + d_grad_K[i] +
                              d_grad_V[i] + d_grad_2el[i];
        }
    }

    std::vector<double> gradient(n3);
    cudaMemcpy(gradient.data(), d_grad_total, grad_bytes, cudaMemcpyDeviceToHost);

    if (d_W) gansu::tracked_cudaFree(d_W);
    gansu::tracked_cudaFree(d_grad_N);
    gansu::tracked_cudaFree(d_grad_S);
    gansu::tracked_cudaFree(d_grad_K);
    gansu::tracked_cudaFree(d_grad_V);
    gansu::tracked_cudaFree(d_grad_2el);
    gansu::tracked_cudaFree(d_grad_total);

    return gradient;
}

// ============================================================================
// ERI_RI::compute_ri_gradient — public thin wrapper.
//
// Uses the persisted intermediate_matrix_B_ as the source for full B, runs
// over the entire aux primitive range, and includes the 1-electron gradient.
// ============================================================================
std::vector<double> ERI_RI::compute_ri_gradient(
    const real_t* d_density_matrix,
    const real_t* d_coefficient_matrix,
    const real_t* d_orbital_energies,
    const int num_electron)
{
    return compute_ri_gradient_impl(
        d_density_matrix, d_coefficient_matrix, d_orbital_energies, num_electron,
        intermediate_matrix_B_.device_ptr(),
        /*P_local_start=*/0,
        /*P_local_end=*/(size_t)num_auxiliary_basis_,
        /*include_one_electron=*/true);
}

} // namespace gansu
