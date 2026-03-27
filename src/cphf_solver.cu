/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "cphf_solver.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include <iostream>
#include <cmath>

namespace gansu {

// ============================================================
//  CPHF Operator: A * U
//
//  A_{ai,bj} U_{bj} = (ε_a - ε_i) U_{ai}
//                    + Σ_{bj} [4(ai|bj) - (ab|ij) - (aj|ib)] U_{bj}
//
//  Input/Output: vectors of size nocc * nvir
//  U is indexed as U[i * nvir + a] (occ × vir)
// ============================================================

// Kernel: apply orbital energy diagonal
__global__
void cphf_apply_diagonal_kernel(
    real_t* d_output, const real_t* d_input,
    const real_t* d_orbital_energies,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    double eps_diff = d_orbital_energies[nocc + a] - d_orbital_energies[i];
    d_output[idx] = eps_diff * d_input[idx];
}

// Kernel: apply 2-electron part
// Σ_{bj} [4(ai|bj) - (ab|ij) - (aj|ib)] U_{bj}
__global__
void cphf_apply_2e_kernel(
    real_t* d_output, const real_t* d_input,
    const real_t* d_eri_mo,
    int nocc, int nvir, int nmo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    int a_mo = nocc + a; // MO index for virtual orbital

    double sum = 0.0;
    for (int j = 0; j < nocc; j++) {
        for (int b = 0; b < nvir; b++) {
            int b_mo = nocc + b;
            double U_bj = d_input[j * nvir + b];

            // ERI in Mulliken notation: (pq|rs) at d_eri_mo[p*nmo³ + q*nmo² + r*nmo + s]
            // (ai|bj): p=a_mo, q=i, r=b_mo, s=j
            double eri_aibj = d_eri_mo[((size_t)a_mo*nmo + i)*nmo*nmo + (size_t)b_mo*nmo + j];
            // (ab|ij): p=a_mo, q=b_mo, r=i, s=j
            double eri_abij = d_eri_mo[((size_t)a_mo*nmo + b_mo)*nmo*nmo + (size_t)i*nmo + j];
            // (aj|ib): p=a_mo, q=j, r=i, s=b_mo
            double eri_ajib = d_eri_mo[((size_t)a_mo*nmo + j)*nmo*nmo + (size_t)i*nmo + b_mo];

            sum += (4.0 * eri_aibj - eri_abij - eri_ajib) * U_bj;
        }
    }
    d_output[idx] += sum;
}

// Kernel: compute diagonal (ε_a - ε_i)
__global__
void cphf_compute_diagonal_kernel(
    real_t* d_diagonal, const real_t* d_orbital_energies,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    d_diagonal[idx] = d_orbital_energies[nocc + a] - d_orbital_energies[i];
}

// Kernel: preconditioner (divide by diagonal)
__global__
void cphf_preconditioner_kernel(
    real_t* d_output, const real_t* d_input, const real_t* d_diagonal,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double diag = d_diagonal[idx];
    d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : d_input[idx];
}

// ============================================================
//  CPHFOperator implementation
// ============================================================

CPHFOperator::CPHFOperator(const real_t* d_eri_mo, const real_t* d_orbital_energies,
                           int nocc, int nvir, int nmo)
    : d_eri_mo_(d_eri_mo), d_orbital_energies_(d_orbital_energies),
      nocc_(nocc), nvir_(nvir), nmo_(nmo), d_diagonal_(nullptr)
{
    // Compute diagonal
    tracked_cudaMalloc(&d_diagonal_, nocc * nvir * sizeof(real_t));
    int threads = 256;
    int blocks = (nocc * nvir + threads - 1) / threads;
    cphf_compute_diagonal_kernel<<<blocks, threads>>>(d_diagonal_, d_orbital_energies, nocc, nvir);
    cudaDeviceSynchronize();
}

CPHFOperator::~CPHFOperator() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
}

void CPHFOperator::apply(const real_t* d_input, real_t* d_output) const {
    int n = nocc_ * nvir_;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Step 1: diagonal part
    cphf_apply_diagonal_kernel<<<blocks, threads>>>(d_output, d_input, d_orbital_energies_, nocc_, nvir_);

    // Step 2: 2-electron part (adds to d_output)
    cphf_apply_2e_kernel<<<blocks, threads>>>(d_output, d_input, d_eri_mo_, nocc_, nvir_, nmo_);

    cudaDeviceSynchronize();
}

void CPHFOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int n = nocc_ * nvir_;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cphf_preconditioner_kernel<<<blocks, threads>>>(d_output, d_input, d_diagonal_, n);
}

// ============================================================
//  Preconditioned Conjugate Gradient solver for CPHF
// ============================================================

void solve_cphf(const CPHFOperator& cphf_op,
                const real_t* d_rhs, real_t* d_U,
                int n_pert, double tol, int max_iter)
{
    int n = cphf_op.dimension();
    cublasHandle_t handle = gpu::GPUHandle::cublas();

    // Workspace per perturbation
    real_t *d_r, *d_z, *d_p, *d_Ap;
    tracked_cudaMalloc(&d_r, n * sizeof(real_t));
    tracked_cudaMalloc(&d_z, n * sizeof(real_t));
    tracked_cudaMalloc(&d_p, n * sizeof(real_t));
    tracked_cudaMalloc(&d_Ap, n * sizeof(real_t));

    std::cout << "  CPHF: solving " << n_pert << " perturbation directions (dim=" << n << ")" << std::endl;

    for (int pert = 0; pert < n_pert; pert++) {
        const real_t* d_b = d_rhs + (size_t)pert * n;  // RHS for this perturbation
        real_t* d_x = d_U + (size_t)pert * n;          // Solution for this perturbation

        // Initialize x = 0
        cudaMemset(d_x, 0, n * sizeof(real_t));

        // r = b (since x = 0, r = b - A*0 = b)
        cudaMemcpy(d_r, d_b, n * sizeof(real_t), cudaMemcpyDeviceToDevice);

        // z = M^{-1} r
        cphf_op.apply_preconditioner(d_r, d_z);

        // p = z
        cudaMemcpy(d_p, d_z, n * sizeof(real_t), cudaMemcpyDeviceToDevice);

        // rz = r^T z
        double rz;
        cublasDdot(handle, n, d_r, 1, d_z, 1, &rz);

        double r_norm;
        cublasDnrm2(handle, n, d_r, 1, &r_norm);
        double b_norm = r_norm;

        for (int iter = 0; iter < max_iter; iter++) {
            // Ap = A * p
            cphf_op.apply(d_p, d_Ap);

            // alpha = rz / (p^T Ap)
            double pAp;
            cublasDdot(handle, n, d_p, 1, d_Ap, 1, &pAp);
            double alpha_cg = rz / (fabs(pAp) > 1e-30 ? pAp : 1e-30);

            // x = x + alpha * p
            cublasDaxpy(handle, n, &alpha_cg, d_p, 1, d_x, 1);

            // r = r - alpha * Ap
            double neg_alpha = -alpha_cg;
            cublasDaxpy(handle, n, &neg_alpha, d_Ap, 1, d_r, 1);

            // Check convergence
            cublasDnrm2(handle, n, d_r, 1, &r_norm);
            if (r_norm / (b_norm > 1e-15 ? b_norm : 1.0) < tol) {
                if (pert == 0 || pert == n_pert - 1)
                    std::cout << "    pert " << pert << ": converged in " << iter + 1 << " iterations (|r|=" << r_norm << ")" << std::endl;
                break;
            }

            // z = M^{-1} r
            cphf_op.apply_preconditioner(d_r, d_z);

            // beta = (r^T z_new) / rz_old
            double rz_new;
            cublasDdot(handle, n, d_r, 1, d_z, 1, &rz_new);
            double beta_cg = rz_new / (fabs(rz) > 1e-30 ? rz : 1e-30);

            // p = z + beta * p
            cublasDscal(handle, n, &beta_cg, d_p, 1);
            double one = 1.0;
            cublasDaxpy(handle, n, &one, d_z, 1, d_p, 1);

            rz = rz_new;
        }
    }

    tracked_cudaFree(d_r);
    tracked_cudaFree(d_z);
    tracked_cudaFree(d_p);
    tracked_cudaFree(d_Ap);
}

} // namespace gansu
