/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file thc_decomposition.cu
 * @brief CPU LS-THC pipeline (Phase 2.0a).
 *
 * All matrix operations go through Eigen (already pulled in via
 * FetchContent for the rest of GANSU's CPU paths).  Matrices are stored
 * column-major in std::vector<real_t> at this layer; we wrap them with
 * Eigen::Map for in-place computation.
 */

#include "thc_decomposition.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "gpu_manager.hpp"

namespace gansu {
namespace {
struct PhaseTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    void log(const char* tag, const char* msg) {
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "    [" << tag << " t=" << std::fixed
                  << std::setprecision(1) << s << "s] " << msg << std::endl;
    }
};
} // anonymous namespace
} // namespace gansu
#endif

namespace gansu {

namespace {

using ColMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using ColVecXd = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;

// Wrap a column-major std::vector<real_t> as an Eigen matrix view.
inline Eigen::Map<const ColMatXd> map_const(const std::vector<real_t>& v,
                                             int rows, int cols)
{
    return Eigen::Map<const ColMatXd>(v.data(), rows, cols);
}

inline Eigen::Map<ColMatXd> map_mut(std::vector<real_t>& v, int rows, int cols)
{
    return Eigen::Map<ColMatXd>(v.data(), rows, cols);
}

} // anonymous namespace

// -----------------------------------------------------------------------------

std::vector<real_t> build_gram_cpu(const std::vector<real_t>& X,
                                   int N_bas, int N_g)
{
    std::vector<real_t> G(static_cast<std::size_t>(N_g) * N_g, 0.0);
    auto Xm = map_const(X, N_bas, N_g);
    auto Gm = map_mut(G, N_g, N_g);
    Gm.noalias() = Xm.transpose() * Xm;
    return G;
}

std::vector<real_t> hadamard_square(const std::vector<real_t>& A)
{
    std::vector<real_t> R(A.size());
    for (std::size_t i = 0; i < A.size(); ++i) R[i] = A[i] * A[i];
    return R;
}

// -----------------------------------------------------------------------------
//   Build E_{P,Q} = sum_{abcd} (ab|cd) X^P_a X^P_b X^Q_c X^Q_d
//
//   Stage 1: K[(ab), P] = X^P_a * X^P_b   -- Khatri-Rao columns,  N_bas^2 x N_g
//   Stage 2: T = V_flat * K               -- N_bas^2 x N_g    (DGEMM)
//   Stage 3: E = K^T * T                  -- N_g    x N_g     (DGEMM)
//
// V_flat is the ERI reshaped as [N_bas^2 x N_bas^2] with row index
// (a + N_bas*b) and column index (c + N_bas*d) in column-major storage.
// -----------------------------------------------------------------------------

std::vector<real_t> build_E_from_eri_cpu(const std::vector<real_t>& X,
                                          const std::vector<real_t>& eri_4d,
                                          int N_bas, int N_g)
{
    const int N2 = N_bas * N_bas;

    // Stage 1: Khatri-Rao K[(ab), P] = X^P_a * X^P_b  (N2 x N_g col-major)
    std::vector<real_t> K(static_cast<std::size_t>(N2) * N_g, 0.0);
    auto Xm = map_const(X, N_bas, N_g);
    {
        auto Km = map_mut(K, N2, N_g);
        for (int P = 0; P < N_g; ++P) {
            // Column P of X is a vector of length N_bas; outer product with itself,
            // flattened column-major into K[*, P].
            for (int b = 0; b < N_bas; ++b) {
                const real_t Xb = Xm(b, P);
                for (int a = 0; a < N_bas; ++a) {
                    Km(a + N_bas * b, P) = Xm(a, P) * Xb;
                }
            }
        }
    }

    // Stage 2: T = V_flat * K  (N2 x N_g)
    // V_flat[(ab),(cd)] reshaped from eri_4d.  Column-major storage:
    //   V_flat[r + N2 * c] = eri_4d[a + N_bas*(b + N_bas*(cd_a + N_bas*cd_b))]
    // where r = a + N_bas*b, c = cd_a + N_bas*cd_b.  This is naturally the
    // same memory layout as eri_4d as a length-N_bas^4 array.
    auto Vm = Eigen::Map<const ColMatXd>(eri_4d.data(), N2, N2);

    std::vector<real_t> T(static_cast<std::size_t>(N2) * N_g, 0.0);
    {
        auto Tm = map_mut(T, N2, N_g);
        auto Km = map_const(K, N2, N_g);
        Tm.noalias() = Vm * Km;
    }

    // Stage 3: E = K^T * T  (N_g x N_g)
    std::vector<real_t> E(static_cast<std::size_t>(N_g) * N_g, 0.0);
    {
        auto Km = map_const(K, N2, N_g);
        auto Tm = map_const(T, N2, N_g);
        auto Em = map_mut(E, N_g, N_g);
        Em.noalias() = Km.transpose() * Tm;
    }

    return E;
}

// -----------------------------------------------------------------------------
//   Z = S^+ E S^+ via symmetric eigendecomposition + relative cutoff
// -----------------------------------------------------------------------------

std::vector<real_t> solve_Z_pinv_cpu(const std::vector<real_t>& S,
                                     const std::vector<real_t>& E,
                                     int N_g, double rel_cutoff,
                                     int* rank_out,
                                     real_t* sigma_max_out,
                                     real_t* sigma_min_kept_out)
{
    auto Sm = map_const(S, N_g, N_g);
    auto Em = map_const(E, N_g, N_g);

    // S is symmetric PSD by construction (Hadamard square of Gram).
    // Symmetrise defensively to suppress tiny off-symmetric noise.
    ColMatXd Ssym = (Sm + Sm.transpose()) * 0.5;

    Eigen::SelfAdjointEigenSolver<ColMatXd> es(Ssym);
    if (es.info() != Eigen::Success) {
        throw std::runtime_error("solve_Z_pinv_cpu: eigendecomposition of S failed");
    }
    const ColVecXd& evals = es.eigenvalues();   // ascending
    const ColMatXd& evecs = es.eigenvectors();

    const real_t s_max = evals.tail(1)(0);
    const real_t cutoff = static_cast<real_t>(rel_cutoff) * s_max;

    ColVecXd inv(N_g);
    int n_kept = 0;
    real_t sigma_min_kept = std::numeric_limits<real_t>::infinity();
    for (int i = 0; i < N_g; ++i) {
        if (evals(i) > cutoff) {
            inv(i) = static_cast<real_t>(1.0) / evals(i);
            ++n_kept;
            if (evals(i) < sigma_min_kept) sigma_min_kept = evals(i);
        } else {
            inv(i) = static_cast<real_t>(0.0);
        }
    }

    if (rank_out)            *rank_out = n_kept;
    if (sigma_max_out)       *sigma_max_out = s_max;
    if (sigma_min_kept_out)  *sigma_min_kept_out = sigma_min_kept;

    ColMatXd UinvUt = evecs * inv.asDiagonal() * evecs.transpose();
    ColMatXd Z_eig = UinvUt * Em * UinvUt;

    std::vector<real_t> Z(static_cast<std::size_t>(N_g) * N_g);
    Eigen::Map<ColMatXd>(Z.data(), N_g, N_g) = Z_eig;
    return Z;
}

// -----------------------------------------------------------------------------
//   Reconstruct (mu nu | lambda sigma)_THC = sum_{PQ} X^P_mu X^P_nu Z X^Q ...
//
//   Build M[(mu,nu), P] = X^P_mu X^P_nu  (N_bas^2 x N_g)
//   eri_thc_flat = M * Z * M^T            (N_bas^2 x N_bas^2)
//   reshape back to N_bas^4.
// -----------------------------------------------------------------------------

std::vector<real_t> reconstruct_eri_thc_cpu(const std::vector<real_t>& X,
                                            const std::vector<real_t>& Z,
                                            int N_bas, int N_g)
{
    const int N2 = N_bas * N_bas;

    // Build Khatri-Rao M[(ab), P] = X^P_a X^P_b
    std::vector<real_t> M(static_cast<std::size_t>(N2) * N_g, 0.0);
    auto Xm = map_const(X, N_bas, N_g);
    {
        auto Mm = map_mut(M, N2, N_g);
        for (int P = 0; P < N_g; ++P) {
            for (int b = 0; b < N_bas; ++b) {
                const real_t Xb = Xm(b, P);
                for (int a = 0; a < N_bas; ++a) {
                    Mm(a + N_bas * b, P) = Xm(a, P) * Xb;
                }
            }
        }
    }

    auto Mm = map_const(M, N2, N_g);
    auto Zm = map_const(Z, N_g, N_g);

    // T = M * Z   (N2 x N_g)
    ColMatXd T = Mm * Zm;

    // V_thc_flat = T * M^T   (N2 x N2)
    ColMatXd V_thc = T * Mm.transpose();

    std::vector<real_t> eri_thc(static_cast<std::size_t>(N_bas) * N_bas * N_bas * N_bas);
    Eigen::Map<ColMatXd>(eri_thc.data(), N2, N2) = V_thc;
    return eri_thc;
}

// -----------------------------------------------------------------------------
//   compute_Z_via_M_svd_cpu: thin SVD of the Khatri-Rao matrix M
//
// Build M (N_bas^2 x N_g), economy SVD, build Z = M^+ V_eri (M^+)^T.
// O(N_bas^4 * N_g) work -- dominates the older S-eigendecomposition path
// whenever N_g >> N_bas^2.
// -----------------------------------------------------------------------------

std::vector<real_t> compute_Z_via_M_svd_cpu(const std::vector<real_t>& X,
                                            const std::vector<real_t>& eri_4d,
                                            int N_bas, int N_g,
                                            double rel_cutoff,
                                            int* rank_out,
                                            real_t* sigma_max_out,
                                            real_t* sigma_min_kept_out)
{
    const int N2 = N_bas * N_bas;

    // ---- 1. Build M[(ab), P] = X^P_a * X^P_b  (N2 x N_g col-major)
    ColMatXd M(N2, N_g);
    {
        auto Xm = map_const(X, N_bas, N_g);
        for (int P = 0; P < N_g; ++P) {
            for (int b = 0; b < N_bas; ++b) {
                const real_t Xb = Xm(b, P);
                for (int a = 0; a < N_bas; ++a) {
                    M(a + N_bas * b, P) = Xm(a, P) * Xb;
                }
            }
        }
    }

    // ---- 2. Economy SVD: M = U Sigma V^T  (U is N2 x r0, V is N_g x r0, r0 = N2)
    Eigen::BDCSVD<ColMatXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    if (svd.info() != Eigen::Success) {
        throw std::runtime_error("compute_Z_via_M_svd_cpu: SVD failed");
    }
    const ColVecXd& sigma = svd.singularValues();   // descending
    const ColMatXd& U = svd.matrixU();               // N2 x r0
    const ColMatXd& V = svd.matrixV();               // N_g x r0

    const real_t s_max = sigma.size() > 0 ? sigma(0) : real_t(0);
    const real_t cutoff = static_cast<real_t>(rel_cutoff) * s_max;

    int r = 0;
    real_t s_min_kept = std::numeric_limits<real_t>::infinity();
    for (int i = 0; i < sigma.size(); ++i) {
        if (sigma(i) > cutoff) {
            ++r;
            if (sigma(i) < s_min_kept) s_min_kept = sigma(i);
        } else {
            break; // descending; stop at first below
        }
    }

    if (rank_out)            *rank_out = r;
    if (sigma_max_out)       *sigma_max_out = s_max;
    if (sigma_min_kept_out)  *sigma_min_kept_out = (r > 0 ? s_min_kept : real_t(0));

    if (r == 0) {
        throw std::runtime_error("compute_Z_via_M_svd_cpu: zero rank");
    }

    // ---- 3. Inner contraction in r-dim space: K = Sigma_r^{-1} U_r^T V_eri U_r Sigma_r^{-1}
    // V_eri stored as N2 x N2 (column-major) at the same memory as eri_4d.
    auto Vm = Eigen::Map<const ColMatXd>(eri_4d.data(), N2, N2);
    ColMatXd Ur = U.leftCols(r);                     // N2 x r
    ColVecXd inv_sigma_r = sigma.head(r).cwiseInverse(); // r

    ColMatXd UtV = Ur.transpose() * Vm;              // r x N2
    ColMatXd UtVU = UtV * Ur;                        // r x r
    ColMatXd K = inv_sigma_r.asDiagonal() * UtVU * inv_sigma_r.asDiagonal(); // r x r

    // ---- 4. Z = V_r * K * V_r^T  (N_g x N_g, but rank r << N_g)
    ColMatXd Vr = V.leftCols(r);                     // N_g x r
    ColMatXd VK = Vr * K;                            // N_g x r
    ColMatXd Z_eig = VK * Vr.transpose();            // N_g x N_g

    std::vector<real_t> Z(static_cast<std::size_t>(N_g) * N_g);
    Eigen::Map<ColMatXd>(Z.data(), N_g, N_g) = Z_eig;
    return Z;
}

#ifndef GANSU_CPU_ONLY

// =============================================================================
// GPU LS-THC pipeline
// =============================================================================
//
// Layout convention here is column-major throughout (matches my CPU code and
// cuBLAS native).  The Khatri-Rao matrix is M[(a + N_bas * b), P] = X[a,P]*X[b,P]
// in col-major.  We eigendecompose the small Gram MM^T of size N_bas^2 x N_bas^2,
// derive U_r and Sigma_r for the kept rank, build V = M^T U Sigma^{-1}, then
//   Z = V (Sigma^{-1} U^T V_eri U Sigma^{-1}) V^T
//
// CAUTION on eigenvector layout:
//   gpu::eigenDecomposition writes "row-major eigvec j is column j",
//   which in column-major view of the same memory means evec j is in ROW j.
//   We call transposeMatrixInPlace once after the call to obtain U with
//   column-major eigvec j in column j (the standard cuBLAS-friendly form).

namespace {

// Khatri-Rao: M[a + N_bas*b, P] = X[a, P] * X[b, P]; M is N_bas^2 x N_g col-major.
__global__ void build_khatri_rao_M_kernel(const real_t* __restrict__ d_X,
                                          int N_bas, int N_g,
                                          real_t* d_M)
{
    const int N2 = N_bas * N_bas;
    const std::size_t total = static_cast<std::size_t>(N2) * N_g;
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int P  = static_cast<int>(idx / N2);
    const int ab = static_cast<int>(idx % N2);
    const int b  = ab / N_bas;
    const int a  = ab % N_bas;

    d_M[idx] = d_X[a + P * N_bas] * d_X[b + P * N_bas];
}

// K[i, j] /= sigma[i] * sigma[j], K is rank x rank col-major.
__global__ void scale_K_inv_sigma_kernel(real_t* d_K,
                                         const real_t* d_sigma,
                                         int rank)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rank || j >= rank) return;
    d_K[i + j * rank] /= (d_sigma[i] * d_sigma[j]);
}

// V[r_idx, c_idx] /= sigma[c_idx] for V col-major of shape (rows, rank).
__global__ void scale_columns_inv_sigma_kernel(real_t* d_V,
                                               const real_t* d_sigma,
                                               int rows, int rank)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= rank) return;
    d_V[i + j * rows] /= d_sigma[j];
}

inline void check_cuda(cudaError_t e, const char* tag)
{
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + tag
                                 + ": " + cudaGetErrorString(e));
    }
}

inline void check_cublas(cublasStatus_t s, const char* tag)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error: ") + tag);
    }
}

} // anonymous namespace

std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_M_svd_gpu(const real_t* d_X,
                         const real_t* d_eri_4d,
                         int N_bas, int N_g,
                         double rel_cutoff,
                         int* rank_out,
                         real_t* sigma_max_out,
                         real_t* sigma_min_kept_out)
{
    const int N2 = N_bas * N_bas;
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;

    // ---- 1. Build M (N2 x N_g col-major)
    real_t* d_M = nullptr;
    check_cuda(cudaMalloc(&d_M, static_cast<std::size_t>(N2) * N_g * sizeof(real_t)),
               "alloc d_M");
    {
        const int threads = 256;
        const std::size_t total = static_cast<std::size_t>(N2) * N_g;
        const int blocks  = static_cast<int>((total + threads - 1) / threads);
        build_khatri_rao_M_kernel<<<blocks, threads>>>(d_X, N_bas, N_g, d_M);
        check_cuda(cudaGetLastError(), "build_khatri_rao_M_kernel");
    }

    // ---- 2. MMT = M M^T (N2 x N2 col-major).  cuBLAS: M is (N2, N_g) col-major.
    //         MMT = M * M^T : op_A = N, op_B = T.  C[N2 x N2] = A[N2 x N_g] B[N_g x N2 from M^T]
    real_t* d_MMT = nullptr;
    check_cuda(cudaMalloc(&d_MMT, static_cast<std::size_t>(N2) * N2 * sizeof(real_t)),
               "alloc d_MMT");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              N2, N2, N_g,
                              &one, d_M, N2, d_M, N2,
                              &zero, d_MMT, N2),
                 "M M^T");

    // ---- 3. Eigen-decompose MMT (symmetric).  Output: evals ascending, evecs
    //         in row-major-eigvec-j-as-col-j form (= col-major: eigvec j in ROW j).
    //         We transpose to col-major standard (column j = eigvec j).
    real_t* d_evals = nullptr;
    real_t* d_evecs = nullptr;
    check_cuda(cudaMalloc(&d_evals, N2 * sizeof(real_t)), "alloc d_evals");
    check_cuda(cudaMalloc(&d_evecs, static_cast<std::size_t>(N2) * N2 * sizeof(real_t)),
               "alloc d_evecs");
    int info = gpu::eigenDecomposition(d_MMT, d_evals, d_evecs, N2);
    if (info != 0) {
        cudaFree(d_M); cudaFree(d_MMT); cudaFree(d_evals); cudaFree(d_evecs);
        throw std::runtime_error("compute_Z_via_M_svd_gpu: eigenDecomposition failed");
    }
    gpu::transposeMatrixInPlace(d_evecs, N2);   // now col-major: column j = eigvec j

    // ---- 4. Determine rank from eigenvalues
    std::vector<real_t> h_evals(N2);
    check_cuda(cudaMemcpy(h_evals.data(), d_evals, N2 * sizeof(real_t),
                          cudaMemcpyDeviceToHost),
               "copy d_evals -> host");
    const real_t lambda_max = h_evals[N2 - 1];   // ascending: max at end
    const real_t sigma_max  = std::sqrt(std::max<real_t>(0, lambda_max));
    const real_t cutoff     = static_cast<real_t>(rel_cutoff) * sigma_max;
    const real_t lambda_cutoff = cutoff * cutoff;

    int rank = 0;
    real_t sigma_min_kept = std::numeric_limits<real_t>::infinity();
    for (int i = 0; i < N2; ++i) {
        if (h_evals[i] > lambda_cutoff) {
            ++rank;
            const real_t s = std::sqrt(h_evals[i]);
            if (s < sigma_min_kept) sigma_min_kept = s;
        }
    }
    if (rank_out)            *rank_out = rank;
    if (sigma_max_out)       *sigma_max_out = sigma_max;
    if (sigma_min_kept_out)  *sigma_min_kept_out = (rank > 0 ? sigma_min_kept : real_t(0));

    if (rank == 0) {
        cudaFree(d_M); cudaFree(d_MMT); cudaFree(d_evals); cudaFree(d_evecs);
        throw std::runtime_error("compute_Z_via_M_svd_gpu: zero rank");
    }

    // Top-r columns of d_evecs (col-major): columns [N2-rank, N2-1].
    // Memory range: d_evecs + (N2 - rank) * N2 .. d_evecs + N2 * N2.
    real_t* d_U_r = d_evecs + static_cast<std::size_t>(N2 - rank) * N2;

    // sigma_r vector on device
    real_t* d_sigma_r = nullptr;
    check_cuda(cudaMalloc(&d_sigma_r, rank * sizeof(real_t)), "alloc d_sigma_r");
    std::vector<real_t> h_sigma_r(rank);
    for (int i = 0; i < rank; ++i) h_sigma_r[i] = std::sqrt(h_evals[N2 - rank + i]);
    check_cuda(cudaMemcpy(d_sigma_r, h_sigma_r.data(),
                          rank * sizeof(real_t), cudaMemcpyHostToDevice),
               "copy h_sigma_r");

    // ---- 5. Inner contraction K = Sigma^{-1} U_r^T V_eri U_r Sigma^{-1}.
    // V_eri viewed as (N2, N2) col-major.
    //   A = U_r^T V_eri   (rank x N2)
    //   B = A   U_r       (rank x rank)
    //   K = (1/sigma) B (1/sigma)
    real_t* d_A = nullptr;
    check_cuda(cudaMalloc(&d_A, static_cast<std::size_t>(rank) * N2 * sizeof(real_t)),
               "alloc d_A");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              rank, N2, N2,
                              &one, d_U_r, N2, d_eri_4d, N2,
                              &zero, d_A, rank),
                 "U_r^T V_eri");

    real_t* d_K = nullptr;
    check_cuda(cudaMalloc(&d_K, static_cast<std::size_t>(rank) * rank * sizeof(real_t)),
               "alloc d_K");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              rank, rank, N2,
                              &one, d_A, rank, d_U_r, N2,
                              &zero, d_K, rank),
                 "(U_r^T V_eri) U_r");
    cudaFree(d_A);

    {
        const dim3 threads(16, 16);
        const dim3 blocks((rank + 15) / 16, (rank + 15) / 16);
        scale_K_inv_sigma_kernel<<<blocks, threads>>>(d_K, d_sigma_r, rank);
        check_cuda(cudaGetLastError(), "scale_K_inv_sigma_kernel");
    }

    // ---- 6. V = M^T U_r Sigma^{-1}  (N_g x rank, col-major)
    real_t* d_V = nullptr;
    check_cuda(cudaMalloc(&d_V, static_cast<std::size_t>(N_g) * rank * sizeof(real_t)),
               "alloc d_V");
    // M^T U_r: cuBLAS. M is (N2, N_g) col-major; M^T is (N_g, N2).
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              N_g, rank, N2,
                              &one, d_M, N2, d_U_r, N2,
                              &zero, d_V, N_g),
                 "M^T U_r");
    {
        const dim3 threads(16, 16);
        const dim3 blocks((N_g + 15) / 16, (rank + 15) / 16);
        scale_columns_inv_sigma_kernel<<<blocks, threads>>>(d_V, d_sigma_r, N_g, rank);
        check_cuda(cudaGetLastError(), "scale_columns_inv_sigma_kernel");
    }
    cudaFree(d_M);

    // ---- 7. VK = V K  (N_g x rank)
    real_t* d_VK = nullptr;
    check_cuda(cudaMalloc(&d_VK, static_cast<std::size_t>(N_g) * rank * sizeof(real_t)),
               "alloc d_VK");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              N_g, rank, rank,
                              &one, d_V, N_g, d_K, rank,
                              &zero, d_VK, N_g),
                 "V K");

    // ---- 8. Z = VK V^T  (N_g x N_g)
    auto Z = std::make_unique<DeviceHostMatrix<real_t>>(N_g, N_g);
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              N_g, N_g, rank,
                              &one, d_VK, N_g, d_V, N_g,
                              &zero, Z->device_ptr(), N_g),
                 "VK V^T");

    cudaFree(d_VK);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_sigma_r);
    cudaFree(d_evals);
    cudaFree(d_evecs);
    cudaFree(d_MMT);

    return Z;
}

// =============================================================================
// compute_Z_via_M_svd_ri_gpu (Phase 2.3): RI-based variant.
//
// Replaces V_{(μν)(λσ)} with V_RI = B B^T where B is the AO 3-index tensor
//   d_B_ao : [N_bas² × naux] column-major lda=N_bas²
// (i.e. the ERI_RI::intermediate_matrix_B_ row-major (naux × N²) viewed as
//  column-major (N² × naux)).
//
// Z = M^+ V_RI (M^+)^T = M^+ B (M^+ B)^T
//   M^+ = V_M Σ_M^{-1} U_M^T   (thin SVD with relative cutoff)
//   K   = (Σ_M^{-1} U_M^T B) (Σ_M^{-1} U_M^T B)^T = T T^T  (rank × rank)
//   V   = M^T U_M Σ_M^{-1}                                  (N_g × rank)
//   Z   = V K V^T                                           (N_g × N_g)
// =============================================================================
std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_M_svd_ri_gpu(const real_t* d_X,
                            const real_t* d_B_ao,
                            int N_bas, int N_g, int naux,
                            double rel_cutoff,
                            int* rank_out,
                            real_t* sigma_max_out,
                            real_t* sigma_min_kept_out)
{
    const int N2 = N_bas * N_bas;
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;

    // ---- 1. Build M (N2 × N_g col-major)
    real_t* d_M = nullptr;
    check_cuda(cudaMalloc(&d_M, static_cast<std::size_t>(N2) * N_g * sizeof(real_t)),
               "alloc d_M (RI-Z)");
    {
        const int threads = 256;
        const std::size_t total = static_cast<std::size_t>(N2) * N_g;
        const int blocks  = static_cast<int>((total + threads - 1) / threads);
        build_khatri_rao_M_kernel<<<blocks, threads>>>(d_X, N_bas, N_g, d_M);
        check_cuda(cudaGetLastError(), "build_khatri_rao_M_kernel (RI-Z)");
    }

    // ---- 2. MMT = M M^T (N2 × N2 col-major).
    real_t* d_MMT = nullptr;
    check_cuda(cudaMalloc(&d_MMT, static_cast<std::size_t>(N2) * N2 * sizeof(real_t)),
               "alloc d_MMT (RI-Z)");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              N2, N2, N_g,
                              &one, d_M, N2, d_M, N2,
                              &zero, d_MMT, N2),
                 "M M^T (RI-Z)");

    // ---- 3. Eigen-decompose MMT (symmetric).
    real_t* d_evals = nullptr;
    real_t* d_evecs = nullptr;
    check_cuda(cudaMalloc(&d_evals, N2 * sizeof(real_t)), "alloc d_evals (RI-Z)");
    check_cuda(cudaMalloc(&d_evecs, static_cast<std::size_t>(N2) * N2 * sizeof(real_t)),
               "alloc d_evecs (RI-Z)");
    int info = gpu::eigenDecomposition(d_MMT, d_evals, d_evecs, N2);
    if (info != 0) {
        cudaFree(d_M); cudaFree(d_MMT); cudaFree(d_evals); cudaFree(d_evecs);
        throw std::runtime_error("compute_Z_via_M_svd_ri_gpu: eigenDecomposition failed");
    }
    gpu::transposeMatrixInPlace(d_evecs, N2);

    // ---- 4. Determine rank from eigenvalues
    std::vector<real_t> h_evals(N2);
    check_cuda(cudaMemcpy(h_evals.data(), d_evals, N2 * sizeof(real_t),
                          cudaMemcpyDeviceToHost),
               "copy d_evals -> host (RI-Z)");
    const real_t lambda_max = h_evals[N2 - 1];
    const real_t sigma_max  = std::sqrt(std::max<real_t>(0, lambda_max));
    const real_t cutoff     = static_cast<real_t>(rel_cutoff) * sigma_max;
    const real_t lambda_cutoff = cutoff * cutoff;

    int rank = 0;
    real_t sigma_min_kept = std::numeric_limits<real_t>::infinity();
    for (int i = 0; i < N2; ++i) {
        if (h_evals[i] > lambda_cutoff) {
            ++rank;
            const real_t s = std::sqrt(h_evals[i]);
            if (s < sigma_min_kept) sigma_min_kept = s;
        }
    }
    if (rank_out)            *rank_out = rank;
    if (sigma_max_out)       *sigma_max_out = sigma_max;
    if (sigma_min_kept_out)  *sigma_min_kept_out = (rank > 0 ? sigma_min_kept : real_t(0));

    if (rank == 0) {
        cudaFree(d_M); cudaFree(d_MMT); cudaFree(d_evals); cudaFree(d_evecs);
        throw std::runtime_error("compute_Z_via_M_svd_ri_gpu: zero rank");
    }

    real_t* d_U_r = d_evecs + static_cast<std::size_t>(N2 - rank) * N2;

    real_t* d_sigma_r = nullptr;
    check_cuda(cudaMalloc(&d_sigma_r, rank * sizeof(real_t)), "alloc d_sigma_r (RI-Z)");
    std::vector<real_t> h_sigma_r(rank);
    for (int i = 0; i < rank; ++i) h_sigma_r[i] = std::sqrt(h_evals[N2 - rank + i]);
    check_cuda(cudaMemcpy(d_sigma_r, h_sigma_r.data(),
                          rank * sizeof(real_t), cudaMemcpyHostToDevice),
               "copy h_sigma_r (RI-Z)");

    // ---- 5. Inner contraction K = (Σ^{-1} U^T B) (Σ^{-1} U^T B)^T.
    //
    //   T = U_r^T B            (rank × naux)
    //   T_scaled = Σ^{-1} T    (scale rows by 1/σ)
    //   K = T_scaled T_scaled^T
    real_t* d_T = nullptr;
    check_cuda(cudaMalloc(&d_T, static_cast<std::size_t>(rank) * naux * sizeof(real_t)),
               "alloc d_T (RI-Z)");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              rank, naux, N2,
                              &one, d_U_r, N2, d_B_ao, N2,
                              &zero, d_T, rank),
                 "T = U_r^T B (RI-Z)");

    // Scale rows of T by 1/σ_r via cublasDdgmm (T ← diag(1/σ) × T).
    {
        real_t* d_sigma_inv = nullptr;
        check_cuda(cudaMalloc(&d_sigma_inv, rank * sizeof(real_t)),
                   "alloc d_sigma_inv (RI-Z)");
        std::vector<real_t> h_sigma_inv(rank);
        for (int i = 0; i < rank; ++i) h_sigma_inv[i] = real_t(1.0) / h_sigma_r[i];
        check_cuda(cudaMemcpy(d_sigma_inv, h_sigma_inv.data(),
                              rank * sizeof(real_t), cudaMemcpyHostToDevice),
                   "copy h_sigma_inv (RI-Z)");
        // T (rank × naux col-major) ← diag(1/σ) × T   (left side ⇒ side mode = LEFT)
        check_cublas(cublasDdgmm(cublas, CUBLAS_SIDE_LEFT,
                                  rank, naux,
                                  d_T, rank,
                                  d_sigma_inv, 1,
                                  d_T, rank),
                     "diag(1/σ) × T (RI-Z)");
        cudaFree(d_sigma_inv);
    }

    real_t* d_K = nullptr;
    check_cuda(cudaMalloc(&d_K, static_cast<std::size_t>(rank) * rank * sizeof(real_t)),
               "alloc d_K (RI-Z)");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              rank, rank, naux,
                              &one, d_T, rank, d_T, rank,
                              &zero, d_K, rank),
                 "K = T T^T (RI-Z)");
    cudaFree(d_T);

    // ---- 6. V = M^T U_r Σ^{-1}  (N_g × rank, col-major)
    real_t* d_V = nullptr;
    check_cuda(cudaMalloc(&d_V, static_cast<std::size_t>(N_g) * rank * sizeof(real_t)),
               "alloc d_V (RI-Z)");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              N_g, rank, N2,
                              &one, d_M, N2, d_U_r, N2,
                              &zero, d_V, N_g),
                 "M^T U_r (RI-Z)");
    {
        const dim3 threads(16, 16);
        const dim3 blocks((N_g + 15) / 16, (rank + 15) / 16);
        scale_columns_inv_sigma_kernel<<<blocks, threads>>>(d_V, d_sigma_r, N_g, rank);
        check_cuda(cudaGetLastError(), "scale_columns_inv_sigma_kernel (RI-Z)");
    }
    cudaFree(d_M);

    // ---- 7. VK = V K  (N_g × rank)
    real_t* d_VK = nullptr;
    check_cuda(cudaMalloc(&d_VK, static_cast<std::size_t>(N_g) * rank * sizeof(real_t)),
               "alloc d_VK (RI-Z)");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              N_g, rank, rank,
                              &one, d_V, N_g, d_K, rank,
                              &zero, d_VK, N_g),
                 "V K (RI-Z)");

    // ---- 8. Z = VK V^T  (N_g × N_g)
    auto Z = std::make_unique<DeviceHostMatrix<real_t>>(N_g, N_g);
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              N_g, N_g, rank,
                              &one, d_VK, N_g, d_V, N_g,
                              &zero, Z->device_ptr(), N_g),
                 "VK V^T (RI-Z)");

    cudaFree(d_VK);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_sigma_r);
    cudaFree(d_evals);
    cudaFree(d_evecs);
    cudaFree(d_MMT);

    return Z;
}

// =============================================================================
// compute_Z_via_rand_svd_ri_gpu (Phase 2.3 large-system path):
//
// Halko-Martinsson-Tropp randomized SVD on M = Khatri-Rao(X, X) (N_bas² × N_g),
// avoiding the O(N_bas^4) M·M^T eigendecomposition that the dense path requires.
// Followed by the same RI-Z post-processing as compute_Z_via_M_svd_ri_gpu.
// =============================================================================
namespace {

// Simple host-side standard-normal generator → device.  cuRAND's
// curandGenerateNormalDouble has parity restrictions per memory note, and the
// random matrix is small (N_g × max_rank, typically a few GB), so a single
// host fill + cudaMemcpy is fine.
void fill_random_normal_to_device(real_t* d_out, std::size_t n,
                                   uint64_t seed = 0x5151515151515151ULL)
{
    std::vector<real_t> h(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nrm(0.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) h[i] = static_cast<real_t>(nrm(rng));
    cudaMemcpy(d_out, h.data(), n * sizeof(real_t), cudaMemcpyHostToDevice);
}

// Thin QR of A (m × n, m ≥ n), in-place: on return A holds Q (m × n orthonormal
// columns).  Throws on cuSOLVER failure.
void thin_qr_in_place(real_t* d_A, int m, int n)
{
    cusolverDnHandle_t cusolver = gpu::GPUHandle::cusolver();
    if (!cusolver) {
        throw std::runtime_error("thin_qr_in_place: cuSOLVER handle not initialised");
    }

    int lwork_geqrf = 0, lwork_orgqr = 0;
    if (cusolverDnDgeqrf_bufferSize(cusolver, m, n, d_A, m, &lwork_geqrf)
            != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cusolverDnDgeqrf_bufferSize failed");
    }
    if (cusolverDnDorgqr_bufferSize(cusolver, m, n, n, d_A, m, nullptr, &lwork_orgqr)
            != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error("cusolverDnDorgqr_bufferSize failed");
    }
    const int lwork = std::max(lwork_geqrf, lwork_orgqr);

    real_t* d_tau = nullptr;
    real_t* d_work = nullptr;
    int* d_info = nullptr;
    cudaMalloc(&d_tau, n * sizeof(real_t));
    cudaMalloc(&d_work, lwork * sizeof(real_t));
    cudaMalloc(&d_info, sizeof(int));

    if (cusolverDnDgeqrf(cusolver, m, n, d_A, m, d_tau, d_work, lwork, d_info)
            != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
        throw std::runtime_error("cusolverDnDgeqrf failed");
    }
    int h_info = 0;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
        throw std::runtime_error("cusolverDnDgeqrf info=" + std::to_string(h_info));
    }

    if (cusolverDnDorgqr(cusolver, m, n, n, d_A, m, d_tau, d_work, lwork, d_info)
            != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
        throw std::runtime_error("cusolverDnDorgqr failed");
    }
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_tau); cudaFree(d_work); cudaFree(d_info);
    if (h_info != 0) {
        throw std::runtime_error("cusolverDnDorgqr info=" + std::to_string(h_info));
    }
}

} // anonymous namespace

std::unique_ptr<DeviceHostMatrix<real_t>>
compute_Z_via_rand_svd_ri_gpu(const real_t* d_X,
                               const real_t* d_B_ao,
                               int N_bas, int N_g, int naux,
                               int max_rank,
                               int n_power_iter,
                               double rel_cutoff,
                               int* rank_out,
                               real_t* sigma_max_out,
                               real_t* sigma_min_kept_out)
{
    const int N2 = N_bas * N_bas;
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;

    if (max_rank <= 0 || max_rank > std::min(N2, N_g)) {
        throw std::runtime_error("compute_Z_via_rand_svd_ri_gpu: invalid max_rank "
                                  + std::to_string(max_rank)
                                  + " (must be in 1.." + std::to_string(std::min(N2, N_g)) + ")");
    }

    // Oversampling: ko = max_rank + p, p ≥ 5 (Halko et al. recommend 5-10).
    constexpr int over_p = 10;
    const int ko = std::min(max_rank + over_p, std::min(N2, N_g));

    PhaseTimer pt;

    // ---- 1. Build M (N² × N_g col-major)
    real_t* d_M = nullptr;
    cudaMalloc(&d_M, static_cast<std::size_t>(N2) * N_g * sizeof(real_t));
    {
        const int threads = 256;
        const std::size_t total = static_cast<std::size_t>(N2) * N_g;
        const int blocks  = static_cast<int>((total + threads - 1) / threads);
        build_khatri_rao_M_kernel<<<blocks, threads>>>(d_X, N_bas, N_g, d_M);
    }
    cudaDeviceSynchronize();
    pt.log("rand-SVD", "Khatri-Rao M built");

    // ---- 2. Random Gaussian Ω (N_g × ko)
    real_t* d_Omega = nullptr;
    cudaMalloc(&d_Omega, static_cast<std::size_t>(N_g) * ko * sizeof(real_t));
    fill_random_normal_to_device(d_Omega, static_cast<std::size_t>(N_g) * ko);

    // ---- 3. Y = M · Ω    (N² × ko)
    real_t* d_Y = nullptr;
    cudaMalloc(&d_Y, static_cast<std::size_t>(N2) * ko * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                N2, ko, N_g,
                &one, d_M, N2, d_Omega, N_g,
                &zero, d_Y, N2);
    cudaDeviceSynchronize();
    pt.log("rand-SVD", "Y = M·Ω done");

    // ---- 4. Power iteration: Y = M · (M^T · Y)
    if (n_power_iter > 0) {
        real_t* d_Z_tmp = nullptr;  // (N_g × ko) intermediate
        cudaMalloc(&d_Z_tmp, static_cast<std::size_t>(N_g) * ko * sizeof(real_t));
        for (int q = 0; q < n_power_iter; ++q) {
            // Z_tmp = M^T · Y     (N_g × ko)
            cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        N_g, ko, N2,
                        &one, d_M, N2, d_Y, N2,
                        &zero, d_Z_tmp, N_g);
            // Y = M · Z_tmp       (N² × ko)
            cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        N2, ko, N_g,
                        &one, d_M, N2, d_Z_tmp, N_g,
                        &zero, d_Y, N2);
            cudaDeviceSynchronize();
            pt.log("rand-SVD",
                   ("power-iter " + std::to_string(q + 1) + "/"
                    + std::to_string(n_power_iter) + " done").c_str());
        }
        cudaFree(d_Z_tmp);
    }
    cudaFree(d_Omega);

    // ---- 5. Q = thin QR(Y)   (N² × ko, in-place)
    thin_qr_in_place(d_Y, N2, ko);
    real_t* d_Q = d_Y;  // reuse buffer
    pt.log("rand-SVD", "thin QR done");

    // ---- 6. B = Q^T M     (ko × N_g)
    real_t* d_B = nullptr;
    cudaMalloc(&d_B, static_cast<std::size_t>(ko) * N_g * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                ko, N_g, N2,
                &one, d_Q, N2, d_M, N2,
                &zero, d_B, ko);
    cudaFree(d_M);
    cudaDeviceSynchronize();
    pt.log("rand-SVD", "B = Q^T·M done");

    // ---- 7. C = B B^T     (ko × ko symmetric)
    real_t* d_C = nullptr;
    cudaMalloc(&d_C, static_cast<std::size_t>(ko) * ko * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                ko, ko, N_g,
                &one, d_B, ko, d_B, ko,
                &zero, d_C, ko);
    cudaDeviceSynchronize();
    pt.log("rand-SVD", "C = B·B^T done (small ko²)");

    // ---- 8. Eigendecompose C → eigenvalues (ascending) σ²_i, eigvecs U_C
    real_t* d_evals = nullptr;
    real_t* d_U_C = nullptr;
    cudaMalloc(&d_evals, ko * sizeof(real_t));
    cudaMalloc(&d_U_C, static_cast<std::size_t>(ko) * ko * sizeof(real_t));
    int info = gpu::eigenDecomposition(d_C, d_evals, d_U_C, ko);
    if (info != 0) {
        cudaFree(d_Q); cudaFree(d_B); cudaFree(d_C);
        cudaFree(d_evals); cudaFree(d_U_C);
        throw std::runtime_error("compute_Z_via_rand_svd_ri_gpu: small eigendecomp failed");
    }
    gpu::transposeMatrixInPlace(d_U_C, ko);   // column j = eigvec j (ascending)
    cudaFree(d_C);
    pt.log("rand-SVD", "small eigendecomp done");

    // ---- 9. Determine rank by rel_cutoff
    std::vector<real_t> h_evals(ko);
    cudaMemcpy(h_evals.data(), d_evals, ko * sizeof(real_t), cudaMemcpyDeviceToHost);
    const real_t lambda_max = h_evals[ko - 1];
    const real_t sigma_max  = std::sqrt(std::max<real_t>(0, lambda_max));
    const real_t cutoff     = static_cast<real_t>(rel_cutoff) * sigma_max;
    const real_t lambda_cutoff = cutoff * cutoff;

    int rank = 0;
    real_t sigma_min_kept = std::numeric_limits<real_t>::infinity();
    for (int i = 0; i < ko; ++i) {
        if (h_evals[i] > lambda_cutoff) {
            ++rank;
            const real_t s = std::sqrt(h_evals[i]);
            if (s < sigma_min_kept) sigma_min_kept = s;
        }
    }
    if (rank > max_rank) rank = max_rank;  // cap at user-requested
    if (rank_out)            *rank_out = rank;
    if (sigma_max_out)       *sigma_max_out = sigma_max;
    if (sigma_min_kept_out)  *sigma_min_kept_out = (rank > 0 ? sigma_min_kept : real_t(0));
    if (rank == 0) {
        cudaFree(d_Q); cudaFree(d_B); cudaFree(d_evals); cudaFree(d_U_C);
        throw std::runtime_error("compute_Z_via_rand_svd_ri_gpu: zero rank");
    }

    // Top-r eigvecs: U_C columns [ko - rank, ko - 1]
    real_t* d_U_C_top = d_U_C + static_cast<std::size_t>(ko - rank) * ko;

    std::vector<real_t> h_sigma_r(rank);
    for (int i = 0; i < rank; ++i) h_sigma_r[i] = std::sqrt(h_evals[ko - rank + i]);
    real_t* d_sigma_r = nullptr;
    cudaMalloc(&d_sigma_r, rank * sizeof(real_t));
    cudaMemcpy(d_sigma_r, h_sigma_r.data(), rank * sizeof(real_t), cudaMemcpyHostToDevice);

    // ---- 10. U_THC = Q · U_C_top    (N² × rank)
    real_t* d_U_THC = nullptr;
    cudaMalloc(&d_U_THC, static_cast<std::size_t>(N2) * rank * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                N2, rank, ko,
                &one, d_Q, N2, d_U_C_top, ko,
                &zero, d_U_THC, N2);
    cudaFree(d_Q);
    cudaFree(d_U_C);

    // ---- 11. RI-Z post-processing: T = Σ⁻¹ U_THC^T B_AO   (rank × naux)
    real_t* d_T = nullptr;
    cudaMalloc(&d_T, static_cast<std::size_t>(rank) * naux * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                rank, naux, N2,
                &one, d_U_THC, N2, d_B_ao, N2,
                &zero, d_T, rank);
    {
        real_t* d_sigma_inv = nullptr;
        cudaMalloc(&d_sigma_inv, rank * sizeof(real_t));
        std::vector<real_t> h_sigma_inv(rank);
        for (int i = 0; i < rank; ++i) h_sigma_inv[i] = real_t(1.0) / h_sigma_r[i];
        cudaMemcpy(d_sigma_inv, h_sigma_inv.data(),
                   rank * sizeof(real_t), cudaMemcpyHostToDevice);
        cublasDdgmm(cublas, CUBLAS_SIDE_LEFT, rank, naux,
                    d_T, rank, d_sigma_inv, 1, d_T, rank);
        cudaFree(d_sigma_inv);
    }

    real_t* d_K = nullptr;
    cudaMalloc(&d_K, static_cast<std::size_t>(rank) * rank * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                rank, rank, naux,
                &one, d_T, rank, d_T, rank,
                &zero, d_K, rank);
    cudaFree(d_T);

    // V_THC = M^T U_THC Σ⁻¹    (N_g × rank)
    // We discarded d_M; reuse d_B (ko × N_g) was already different.  Need M
    // re-built or kept.  Optimisation: rebuild M just for V (saves 2 × N² × ko
    // memory at the QR stage).  Rebuild here.
    real_t* d_M2 = nullptr;
    cudaMalloc(&d_M2, static_cast<std::size_t>(N2) * N_g * sizeof(real_t));
    {
        const int threads = 256;
        const std::size_t total = static_cast<std::size_t>(N2) * N_g;
        const int blocks  = static_cast<int>((total + threads - 1) / threads);
        build_khatri_rao_M_kernel<<<blocks, threads>>>(d_X, N_bas, N_g, d_M2);
    }
    real_t* d_V = nullptr;
    cudaMalloc(&d_V, static_cast<std::size_t>(N_g) * rank * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                N_g, rank, N2,
                &one, d_M2, N2, d_U_THC, N2,
                &zero, d_V, N_g);
    cudaFree(d_M2);
    cudaFree(d_U_THC);
    {
        const dim3 threads(16, 16);
        const dim3 blocks((N_g + 15) / 16, (rank + 15) / 16);
        scale_columns_inv_sigma_kernel<<<blocks, threads>>>(d_V, d_sigma_r, N_g, rank);
    }

    // VK = V K
    real_t* d_VK = nullptr;
    cudaMalloc(&d_VK, static_cast<std::size_t>(N_g) * rank * sizeof(real_t));
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                N_g, rank, rank,
                &one, d_V, N_g, d_K, rank,
                &zero, d_VK, N_g);

    // Z = VK V^T
    auto Z = std::make_unique<DeviceHostMatrix<real_t>>(N_g, N_g);
    cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                N_g, N_g, rank,
                &one, d_VK, N_g, d_V, N_g,
                &zero, Z->device_ptr(), N_g);
    cudaDeviceSynchronize();
    pt.log("rand-SVD", "Z assembled (rank-truncated)");

    cudaFree(d_VK);
    cudaFree(d_V);
    cudaFree(d_K);
    cudaFree(d_sigma_r);
    cudaFree(d_evals);
    cudaFree(d_B);

    return Z;
}

// =============================================================================
// reconstruct_eri_thc_gpu: build the full N_bas^2 x N_bas^2 THC-ERI tensor
//
//   (ab|cd)_THC = sum_{PQ} X^P_a X^P_b Z_{PQ} X^Q_c X^Q_d
//
// Same kernel as the LS-THC pipeline:
//   M[(ab), P] = X^P_a * X^P_b   (N_bas^2 x N_g col-major)
//   T = M * Z                    (N_bas^2 x N_g)
//   V_THC = T * M^T              (N_bas^2 x N_bas^2)
// =============================================================================

std::unique_ptr<DeviceHostMatrix<real_t>>
reconstruct_eri_thc_gpu(const real_t* d_X, const real_t* d_Z,
                         int N_bas, int N_g)
{
    const int N2 = N_bas * N_bas;
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;

    // M (Khatri-Rao)
    real_t* d_M = nullptr;
    check_cuda(cudaMalloc(&d_M, static_cast<std::size_t>(N2) * N_g * sizeof(real_t)),
               "alloc d_M (reconstruct)");
    {
        const int threads = 256;
        const std::size_t total = static_cast<std::size_t>(N2) * N_g;
        const int blocks  = static_cast<int>((total + threads - 1) / threads);
        build_khatri_rao_M_kernel<<<blocks, threads>>>(d_X, N_bas, N_g, d_M);
        check_cuda(cudaGetLastError(), "build_khatri_rao_M_kernel (reconstruct)");
    }

    // T = M * Z   (N2 x N_g)
    real_t* d_T = nullptr;
    check_cuda(cudaMalloc(&d_T, static_cast<std::size_t>(N2) * N_g * sizeof(real_t)),
               "alloc d_T");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              N2, N_g, N_g,
                              &one, d_M, N2, d_Z, N_g,
                              &zero, d_T, N2),
                 "M Z");

    // V_THC = T * M^T  (N2 x N2), output as DeviceHostMatrix
    auto V = std::make_unique<DeviceHostMatrix<real_t>>(N2, N2);
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              N2, N2, N_g,
                              &one, d_T, N2, d_M, N2,
                              &zero, V->device_ptr(), N2),
                 "T M^T");

    cudaFree(d_T);
    cudaFree(d_M);
    return V;
}

#endif // GANSU_CPU_ONLY

} // namespace gansu
