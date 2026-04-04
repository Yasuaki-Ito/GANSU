/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file cis_operator_ri.cu
 * @brief B-matrix based CIS operator for RI approximation
 *
 * Computes CIS sigma vector without forming nmo^4 MO ERI tensor.
 * Uses RI B-matrix intermediates: B_ov(Q,ia), B_oo(Q,ij), B_vv(Q,ab).
 *
 * Singlet: σ(ia) = (ε_a - ε_i)r(ia) + 2*J(ia) - K(ia)
 * Triplet: σ(ia) = (ε_a - ε_i)r(ia) - K(ia)
 *
 * Coulomb: J(ia) = Σ_Q B_ov(Q,ia) × w(Q)  where w(Q) = Σ_{jb} B_ov(Q,jb) r(jb)
 * Exchange: K(ia) = Σ_Q Σ_j B_oo(Q,ij) × [Σ_b B_vv(Q,ab) r(jb)]
 */

#include "cis_operator_ri.hpp"
#include "device_host_memory.hpp"
#include "utils.hpp"

namespace gansu {

// Diagonal kernel: d(ia) = eps_a - eps_i
__global__ void cis_ri_diagonal_kernel(
    const real_t* __restrict__ d_eps,
    real_t* __restrict__ d_diag,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    d_diag[idx] = d_eps[nocc + a] - d_eps[i];
}

// Preconditioner: output = input / diagonal
__global__ void cis_ri_precond_kernel(
    const real_t* __restrict__ d_diag,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t d = d_diag[idx];
    d_output[idx] = (fabs(d) > 1e-12) ? d_input[idx] / d : 0.0;
}

// sigma += diag * r (element-wise)
__global__ void cis_ri_diag_apply_kernel(
    const real_t* __restrict__ d_diag,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    d_output[idx] = d_diag[idx] * d_input[idx];
}

// ========================================================================

CISOperator_RI::CISOperator_RI(
    const real_t* d_B_ov, const real_t* d_B_oo, const real_t* d_B_vv,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int naux,
    bool is_triplet)
    : nocc_(nocc), nvir_(nvir), naux_(naux),
      dim_(nocc * nvir), is_triplet_(is_triplet),
      d_B_ov_(d_B_ov), d_B_oo_(d_B_oo), d_B_vv_(d_B_vv),
      d_diagonal_(nullptr), d_work_(nullptr)
{
    tracked_cudaMalloc(&d_diagonal_, dim_ * sizeof(real_t));
    // Workspace: max of naux (Coulomb w) and naux*nocc*nvir (Exchange tmp)
    size_t work_size = (size_t)naux_ * nocc_ * nvir_;
    tracked_cudaMalloc(&d_work_, work_size * sizeof(real_t));

    build_diagonal(d_orbital_energies);
}

CISOperator_RI::~CISOperator_RI() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
    if (d_work_) tracked_cudaFree(d_work_);
}

void CISOperator_RI::build_diagonal(const real_t* d_orbital_energies) {
    int threads = 256, blocks = (dim_ + threads - 1) / threads;
    cis_ri_diagonal_kernel<<<blocks, threads>>>(d_orbital_energies, d_diagonal_, nocc_, nvir_);
    cudaDeviceSynchronize();
}

void CISOperator_RI::apply(const real_t* d_input, real_t* d_output) const {
    // d_input  = r(ia),  shape (dim,)   = (nocc*nvir,)
    // d_output = σ(ia), shape (dim,)

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const real_t one = 1.0, zero = 0.0, two = 2.0, minus_one = -1.0;
    const int ov = nocc_ * nvir_;
    const int oo = nocc_ * nocc_;
    const int vv = nvir_ * nvir_;

    // (1) Diagonal: σ = diag × r
    {
        int threads = 256, blocks = (dim_ + threads - 1) / threads;
        cis_ri_diag_apply_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, dim_);
    }

    // (2) Coulomb: σ += 2 × B_ov^T × (B_ov × r)
    //   B_ov: (naux, ov) row-major → cuBLAS col-major: (ov, naux)
    //   Step 2a: w(Q) = Σ_{ia} B_ov(Q,ia) r(ia) = B_ov × r
    //            cuBLAS: w = B_ov^T_cm × r = (row-major B_ov viewed as col-major)^T × r
    //            → cublasDgemv(CUBLAS_OP_T, ov, naux, 1, B_ov, ov, r, 1, 0, w, 1)
    //   Step 2b: σ += 2 × B_ov^T × w
    //            cuBLAS: σ += 2 × B_ov_cm × w = cublasDgemv(N, ov, naux, 2, B_ov, ov, w, 1, 1, σ, 1)
    if (!is_triplet_) {
        real_t* d_w = d_work_;  // reuse workspace, only naux elements needed
        cublasDgemv(handle, CUBLAS_OP_T, ov, naux_, &one,
                    d_B_ov_, ov, d_input, 1, &zero, d_w, 1);
        cublasDgemv(handle, CUBLAS_OP_N, ov, naux_, &two,
                    d_B_ov_, ov, d_w, 1, &one, d_output, 1);
    }

    // (3) Exchange: σ -= K(ia)
    //   K(ia) = Σ_Q Σ_j B_oo(Q,ij) [Σ_b B_vv(Q,ab) r(jb)]
    //
    //   r(jb) viewed as R(j,b) shape (nocc, nvir)
    //   Step 3a: tmp(Q,j,a) = Σ_b B_vv(Q,ab) R(j,b)
    //            For each Q: tmp_Q(a,j) = B_vv_Q(a,b) × R^T(b,j)   [nvir×nvir × nvir×nocc → nvir×nocc]
    //            Batched: B_vv (naux, nvir, nvir) × R^T (nvir, nocc) → tmp (naux, nvir, nocc)
    //            As single DGEMM: reshape B_vv as (naux*nvir, nvir), multiply by R^T (nvir, nocc)
    //            → result (naux*nvir, nocc), then view as (naux, nvir, nocc)
    //
    //   Step 3b: K(i,a) = Σ_Q Σ_j B_oo(Q,ij) tmp(Q,j,a)
    //            = Σ_Q B_oo_Q(i,j) × tmp_Q(j,a)   → for each Q: (nocc,nocc) × (nocc,nvir) → (nocc,nvir)
    //            Reshape: B_oo as (naux*nocc, nocc), tmp as (naux*nocc, nvir)?
    //            Not straightforward. Use batched DGEMM or loop over Q.
    //
    //   Alternative single-DGEMM approach for Step 3b:
    //   Transpose tmp to tmp2(Q, a, j) = tmp(Q, j, a)^T per Q slice
    //   Then K(i,a) = Σ_Q B_oo(Q,i,j) tmp2(Q,a,j) — still per-Q.
    //
    //   Efficient approach: use the identity
    //   K(ia) = Σ_Q Σ_j B_oo(Q,ij) tmp(Q,j,a)
    //         = Σ_j [Σ_Q B_oo(Q,ij) tmp(Q,j,a)]
    //   Reshape B_oo(Q,i,j) as M(Q*nocc+i, j) = (naux*nocc, nocc)  [Q is slow, i is fast within Q]
    //   Actually B_oo is stored as (naux, nocc*nocc) row-major = (naux, oo).
    //   B_oo(Q,i,j) = B_oo[Q * oo + i * nocc + j]
    //   Reshape as (naux*nocc, nocc) by viewing Q*nocc+i as row index. This works if oo = nocc*nocc.
    //
    //   tmp(Q,j,a): from Step 3a, stored as (naux*nvir, nocc) but we need (naux*nocc, nvir).
    //   The layout from Step 3a gives us: for each Q, a nvir×nocc block.
    //   We need: for each Q, a nocc×nvir block (tmp(Q,j,a)).
    //   These are transposes of each other per Q-slice.
    //
    //   Simpler: do Step 3a as tmp(Q,j,a) directly.
    //   R(j,b) = d_input[j*nvir + b]. B_vv(Q,a,b) = d_B_vv[Q*vv + a*nvir + b].
    //   tmp(Q,j,a) = Σ_b B_vv(Q,a,b) R(j,b)
    //   For fixed Q: tmp_Q = R × B_vv_Q^T  [nocc×nvir × nvir×nvir → nocc×nvir]
    //   All Q: view R as (nocc, nvir), B_vv as (naux*nvir, nvir).
    //   Actually: B_vv_Q^T(b,a) = B_vv[Q*vv+a*nvir+b]^T ... row-major B_vv(Q,a,b).
    //
    //   Let me just use explicit DGEMM per Q-slice. For small naux this is fine.
    //   For larger systems, batched DGEMM can be used.

    {
        // d_work_ has room for naux*nocc*nvir elements
        real_t* d_tmp = d_work_;  // (naux, nocc, nvir) or equivalently naux blocks of (nocc, nvir)

        // Step 3a: For each Q, tmp_Q(j,a) = Σ_b R(j,b) × B_vv_Q(a,b)^T
        //   = R × B_vv_Q^T where R is (nocc, nvir), B_vv_Q is (nvir, nvir)
        //   Result: (nocc, nvir)
        //   cuBLAS (col-major): B_vv_Q_cm(nvir,nvir) is B_vv_Q^T_rm.
        //   R_cm(nvir,nocc) is R^T_rm.
        //   tmp_cm = B_vv_cm × R_cm = (nvir,nvir)×(nvir,nocc) → (nvir,nocc)
        //   In row-major: this gives tmp(nocc,nvir) = R × B_vv^T. Correct!
        for (int Q = 0; Q < naux_; Q++) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        nvir_, nocc_, nvir_,
                        &one,
                        &d_B_vv_[Q * vv], nvir_,  // B_vv_Q col-major = B_vv_Q^T row-major
                        d_input, nvir_,            // R col-major = R^T row-major
                        &zero,
                        &d_tmp[Q * ov], nvir_);    // tmp_Q col-major (nvir, nocc)
        }
        // Now d_tmp[Q*ov + a*nocc + j] in col-major = tmp(Q,j,a) in the math sense
        // Actually in col-major output: d_tmp[Q*ov + a + j*nvir]?
        // cuBLAS DGEMM output C(nvir, nocc): C[a + j*nvir] = C_cm(a,j).
        // So d_tmp[Q*ov + a + j*nvir] = tmp_Q(a,j) in col-major = tmp_Q(j,a) transposed.
        // We need K(i,a) = Σ_Q Σ_j B_oo(Q,i,j) tmp_Q(j,a)
        // = Σ_Q B_oo_Q(i,j) × tmp_Q_cm^T(j,a)
        // In col-major with B_oo_Q stored as (nocc,nocc) row-major = col-major transposed:
        // B_oo_Q_cm(j,i) = B_oo_Q_rm(i,j)

        // Step 3b: K(i,a) = Σ_Q Σ_j B_oo(Q,i,j) × tmp(Q,j,a)
        // tmp_Q in memory: col-major (nvir, nocc) → element (a,j) at offset a + j*nvir
        // B_oo_Q in memory: row-major (nocc, nocc) → cuBLAS col-major: (nocc, nocc)^T

        // K = Σ_Q B_oo_Q_rm × tmp_Q_as_matrix(nocc, nvir)
        // where tmp_Q_as_matrix(j,a) = d_tmp[Q*ov + a + j*nvir] → this is col-major (nvir,nocc),
        // so row-major is (nocc, nvir) with tmp_rm(j,a) = d_tmp[Q*ov + j*nvir + a].
        // But d_tmp is col-major from DGEMM output: d_tmp[Q*ov + a + j*nvir].
        // row-major view: row j, col a → offset j*nvir + a ≠ a + j*nvir. Actually they are the same!
        // j*nvir + a = a + j*nvir. So tmp_rm(j,a) = d_tmp[Q*ov + j*nvir + a]. Good.

        // K(i,a) += B_oo_Q(i,j) × tmp(j,a) for each Q
        // cuBLAS: K_cm += B_oo_cm^T × tmp_cm
        //   B_oo_cm = B_oo_rm^T. B_oo_cm^T = B_oo_rm.
        //   cublasDgemm(T, N, nvir, nocc, nocc, 1, tmp, nvir, B_oo, nocc, 1, K, nvir)
        // Wait, let me think more carefully.
        //
        // K(i,a) = Σ_j B_oo(i,j) × tmp(j,a)  →  K_rm = B_oo_rm × tmp_rm
        //   K_rm(nocc,nvir) = B_oo_rm(nocc,nocc) × tmp_rm(nocc,nvir)
        // In cuBLAS col-major:
        //   K_cm(nvir,nocc) = tmp_cm(nvir,nocc) × B_oo_cm(nocc,nocc)^?
        //   No: K_rm = A_rm × B_rm → K_cm^T = B_cm^T × A_cm^T → K_cm = (B^T A^T)^T?
        // Simplest: cublasDgemm with transpositions.
        // K_rm = B_oo_rm × tmp_rm. In cublas:
        // C_cm = tmp_cm × B_oo_cm^T where B_oo_cm = B_oo_rm^T, so B_oo_cm^T = B_oo_rm.
        // cublasDgemm(N, T, nvir, nocc, nocc, alpha, tmp, nvir, B_oo, nocc, beta, K, nvir)

        // Initialize K = d_output (already has diagonal part). We accumulate -K into it.
        for (int Q = 0; Q < naux_; Q++) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        nvir_, nocc_, nocc_,
                        &minus_one,
                        &d_tmp[Q * ov], nvir_,
                        &d_B_oo_[Q * oo], nocc_,
                        &one,
                        d_output, nvir_);
        }
    }
}

void CISOperator_RI::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256, blocks = (dim_ + threads - 1) / threads;
    cis_ri_precond_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, dim_);
}

} // namespace gansu
