/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

// Host symmetric eigendecomposition via LAPACK divide-and-conquer (dsyevd),
// as a faster drop-in for Eigen::SelfAdjointEigenSolver in the DLPNO per-pair
// setup hot loops (S_dom orthogonalisation, F_PAO semi-canonicalisation, PNO
// density). Eigen's solver uses implicit QR; LAPACK's D&C is typically 2-4x
// faster for the n≈130-450 matrices that dominate DLPNO pair_setup / SC-PNO.
//
// dsyevd_ is resolved from the already-linked BLAS library (OpenBLAS ships
// LAPACK in the same .so), so no extra CMake linking is required.
//
// Default ON (opt-out with GANSU_DLPNO_LAPACK_EIG=0). Validated on Decacene:
// LAPACK D&C is ~57x faster than Eigen's QR for the n≈130-458 per-pair eigs
// (Sdom_eig 2607→47 s, FPAO_eig 1326→21 s thread-summed; pair_setup phase1 host
// 73→3.6 s), with naphthalene ΔE 2.4e-14 (eigenvalues match Eigen; eigenvector
// sign / degenerate-subspace ambiguity is invariant under the downstream
// M=V·diag(1/√λ) / PNO occupation-threshold use). The earlier suspicion that
// OpenBLAS thread oversubscription was the cause was ruled out — OPENBLAS_
// NUM_THREADS=1 had no effect; the entire win is the eigensolver algorithm.

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace gansu {

extern "C" void dsyevd_(const char* jobz, const char* uplo, const int* n,
                        double* a, const int* lda, double* w,
                        double* work, const int* lwork,
                        int* iwork, const int* liwork, int* info);

// Runtime gate (evaluated once). Default ON; opt out with
// GANSU_DLPNO_LAPACK_EIG=0 to fall back to the Eigen eigensolver.
inline bool use_lapack_eig() {
    static const bool on = []() {
        const char* e = std::getenv("GANSU_DLPNO_LAPACK_EIG");
        return !(e && e[0] == '0');
    }();
    return on;
}

// Symmetric eigendecomposition matching Eigen::SelfAdjointEigenSolver's output
// convention exactly:
//   - evals: n eigenvalues in ASCENDING order.
//   - evecs_rowmajor: n×n row-major with column k = the k-th eigenvector, i.e.
//     evecs_rowmajor[i*n + k] = component i of eigenvector k. This is what
//     `Eigen::Map<RowMatXd>(evecs.data(), n, n).col(k)` returns, matching the
//     existing eig_sym / build_pno_from_T / orthogonalize_pao_domain callers.
//
// mat_in is the n×n symmetric matrix (row-major; symmetric ⇒ the col-major
// view LAPACK takes is identical, so no pre-transpose is needed). Throws on
// LAPACK failure (info != 0).
inline void lapack_syevd(const double* mat_in, int n,
                         std::vector<double>& evals,
                         std::vector<double>& evecs_rowmajor)
{
    if (n <= 0) { evals.clear(); evecs_rowmajor.clear(); return; }
    evals.assign(static_cast<size_t>(n), 0.0);
    // dsyevd overwrites the matrix with the eigenvectors (column-major).
    std::vector<double> a(mat_in, mat_in + static_cast<size_t>(n) * n);

    const char jobz = 'V';
    const char uplo = 'L';  // symmetric input ⇒ either triangle is fine.
    int info = 0;

    // Workspace size query (lwork = liwork = -1).
    double work_query = 0.0;
    int    iwork_query = 0;
    int    lwork = -1, liwork = -1;
    dsyevd_(&jobz, &uplo, &n, a.data(), &n, evals.data(),
            &work_query, &lwork, &iwork_query, &liwork, &info);
    if (info != 0)
        throw std::runtime_error("lapack_syevd: workspace query failed");
    lwork  = static_cast<int>(work_query);
    liwork = iwork_query;
    std::vector<double> work(static_cast<size_t>(std::max(1, lwork)));
    std::vector<int>    iwork(static_cast<size_t>(std::max(1, liwork)));

    dsyevd_(&jobz, &uplo, &n, a.data(), &n, evals.data(),
            work.data(), &lwork, iwork.data(), &liwork, &info);
    if (info != 0)
        throw std::runtime_error("lapack_syevd: eigendecomposition failed");

    // a holds eigenvectors column-major: a[i + k*n] = component i of evec k.
    // Transpose into the row-major (column k = evec k) convention.
    evecs_rowmajor.assign(static_cast<size_t>(n) * n, 0.0);
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            evecs_rowmajor[static_cast<size_t>(i) * n + k] =
                a[static_cast<size_t>(i) + static_cast<size_t>(k) * n];
}

}  // namespace gansu
