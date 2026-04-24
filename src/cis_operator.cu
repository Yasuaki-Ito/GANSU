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

/**
 * @file cis_operator.cu
 * @brief GPU implementation of CIS (Configuration Interaction Singles) operator
 *
 * CIS A-matrix for RHF excited states:
 *   Singlet: A_{ia,jb} = delta_{ij} delta_{ab} (eps_a - eps_i) + 2(ia|jb) - (ij|ab)
 *   Triplet: A_{ia,jb} = delta_{ij} delta_{ab} (eps_a - eps_i) - (ij|ab)
 *
 * Equations derived by spin2spatial tool (CIS_RHF.md):
 *   sigma^a_i = -2 f_ji r^a_j + 2 f_ab r^b_i + 4(ai|jb) r^b_j - 2(ab|ji) r^b_j
 * Divided by 2 (spin degeneracy factor) to get eigenvalues = excitation energies.
 */

#include <cmath>
#include <Eigen/Dense>
#include "cis_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

namespace gansu {

// ========================================================================
//  CUDA kernel: Build CIS A-matrix
// ========================================================================

/**
 * @brief Build CIS A-matrix elements
 *
 * Singlet: A[ia, jb] = delta_{ij} * delta_{ab} * (eps_a - eps_i) + 2*(ia|jb) - (ij|ab)
 * Triplet: A[ia, jb] = delta_{ij} * delta_{ab} * (eps_a - eps_i) - (ij|ab)
 *
 * Index mapping: ia = i * nvir + a_rel, where a_rel = a_abs - nocc
 * ERI access: eri_mo[(p*nao + q)*nao*nao + r*nao + s] = (pq|rs)
 */
__global__ void cis_build_A_matrix_kernel(
    const real_t* __restrict__ d_eri_mo,
    const real_t* __restrict__ d_orbital_energies,
    real_t* __restrict__ d_A_matrix,
    int nocc, int nvir, int nao,
    bool is_triplet,
    int occ_offset, int vir_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim = nocc * nvir;
    int total = dim * dim;
    if (idx >= total) return;

    int ia = idx / dim;
    int jb = idx % dim;

    int i = ia / nvir;
    int a_rel = ia % nvir;
    int j = jb / nvir;
    int b_rel = jb % nvir;

    // Apply frozen core offsets for MO-ERI and orbital energy indexing
    int i_abs = occ_offset + i;
    int j_abs = occ_offset + j;
    int a_abs = vir_start + a_rel;
    int b_abs = vir_start + b_rel;

    size_t nao2 = (size_t)nao * nao;

    // (ij|ab) = eri_mo[(i*nao + j)*nao*nao + a_abs*nao + b_abs]
    real_t ij_ab = d_eri_mo[((size_t)i_abs * nao + j_abs) * nao2 + (size_t)a_abs * nao + b_abs];

    real_t val;
    if (is_triplet) {
        // Triplet: -(ij|ab)
        val = -ij_ab;
    } else {
        // Singlet: 2*(ia|jb) - (ij|ab)
        real_t ia_jb = d_eri_mo[((size_t)i_abs * nao + a_abs) * nao2 + (size_t)j_abs * nao + b_abs];
        val = 2.0 * ia_jb - ij_ab;
    }

    // Diagonal contribution
    if (i == j && a_rel == b_rel) {
        val += d_orbital_energies[a_abs] - d_orbital_energies[i_abs];
    }

    d_A_matrix[idx] = val;
}


// ========================================================================
//  CUDA kernel: Extract diagonal and apply preconditioner
// ========================================================================

__global__ void cis_extract_diagonal_kernel(
    const real_t* __restrict__ d_A_matrix,
    real_t* __restrict__ d_diagonal,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    d_diagonal[idx] = d_A_matrix[(size_t)idx * dim + idx];
}

__global__ void cis_preconditioner_kernel(
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t diag = d_diagonal[idx];
    d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
}


// ========================================================================
//  CISOperator Implementation
// ========================================================================

CISOperator::CISOperator(
    const real_t* d_eri_mo,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int nao,
    bool is_triplet,
    int occ_offset, int vir_start)
    : nocc_(nocc), nvir_(nvir), nao_(nao),
      dim_(nocc * nvir),
      occ_offset_(occ_offset),
      vir_start_(vir_start < 0 ? occ_offset + nocc : vir_start),
      is_triplet_(is_triplet),
      d_A_matrix_(nullptr),
      d_diagonal_(nullptr)
{
    build_A_matrix(d_eri_mo, d_orbital_energies);
}

CISOperator::~CISOperator() {
    if (d_A_matrix_) tracked_cudaFree(d_A_matrix_);
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void CISOperator::build_A_matrix(const real_t* d_eri_mo, const real_t* d_orbital_energies) {
    size_t matrix_size = (size_t)dim_ * dim_;

    // Allocate A-matrix and diagonal
    tracked_cudaMalloc(&d_A_matrix_, matrix_size * sizeof(real_t));
    tracked_cudaMalloc(&d_diagonal_, dim_ * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback: build A-matrix
        const real_t* eri = d_eri_mo;
        const real_t* eps = d_orbital_energies;
        size_t nao2 = (size_t)nao_ * nao_;
        for (int ia = 0; ia < dim_; ia++) {
            int i = ia / nvir_;
            int a_rel = ia % nvir_;
            int i_abs = occ_offset_ + i;
            int a_abs = vir_start_ + a_rel;
            for (int jb = 0; jb < dim_; jb++) {
                int j = jb / nvir_;
                int b_rel = jb % nvir_;
                int j_abs = occ_offset_ + j;
                int b_abs = vir_start_ + b_rel;
                size_t idx = (size_t)ia * dim_ + jb;

                real_t ij_ab = eri[((size_t)i_abs * nao_ + j_abs) * nao2 + (size_t)a_abs * nao_ + b_abs];
                real_t val;
                if (is_triplet_) {
                    val = -ij_ab;
                } else {
                    real_t ia_jb = eri[((size_t)i_abs * nao_ + a_abs) * nao2 + (size_t)j_abs * nao_ + b_abs];
                    val = 2.0 * ia_jb - ij_ab;
                }
                if (i == j && a_rel == b_rel) {
                    val += eps[a_abs] - eps[i_abs];
                }
                d_A_matrix_[idx] = val;
            }
        }
        // Extract diagonal
        for (int i = 0; i < dim_; i++) {
            d_diagonal_[i] = d_A_matrix_[(size_t)i * dim_ + i];
        }
        return;
    }

    // Build A-matrix
    int threads = 256;
    int blocks = (matrix_size + threads - 1) / threads;
    cis_build_A_matrix_kernel<<<blocks, threads>>>(
        d_eri_mo, d_orbital_energies, d_A_matrix_,
        nocc_, nvir_, nao_, is_triplet_, occ_offset_, vir_start_);
    cudaDeviceSynchronize();

    // Extract diagonal
    blocks = (dim_ + threads - 1) / threads;
    cis_extract_diagonal_kernel<<<blocks, threads>>>(
        d_A_matrix_, d_diagonal_, dim_);
    cudaDeviceSynchronize();
}

void CISOperator::apply(const real_t* d_input, real_t* d_output) const {
    // sigma = A * r  using cuBLAS DGEMV
    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    if (!gpu::gpu_available()) {
        // CPU fallback: y = alpha * A * x + beta * y (column-major)
        using Eigen::Map;
        using Eigen::MatrixXd;
        using Eigen::VectorXd;
        Map<VectorXd>(d_output, dim_) = alpha * Map<const MatrixXd>(d_A_matrix_, dim_, dim_) * Map<const VectorXd>(d_input, dim_);
        return;
    }

    // cuBLAS uses column-major, but our A is row-major.
    // For row-major A: y = A*x is equivalent to column-major y = A^T * x
    // But since A is symmetric (CIS matrix is symmetric), A = A^T, so:
    // y = A * x (row-major) = A^T * x (col-major) = A * x (col-major, since A = A^T)
    cublasDgemv(gpu::GPUHandle::cublas(), CUBLAS_OP_N,
                dim_, dim_, &alpha,
                d_A_matrix_, dim_,
                d_input, 1,
                &beta, d_output, 1);
}

void CISOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    if (!gpu::gpu_available()) {
        for (int i = 0; i < dim_; i++) {
            real_t diag = d_diagonal_[i];
            d_output[i] = (std::fabs(diag) > 1e-12) ? d_input[i] / diag : 0.0;
        }
        return;
    }

    int threads = 256;
    int blocks = (dim_ + threads - 1) / threads;
    cis_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, dim_);
}

} // namespace gansu
