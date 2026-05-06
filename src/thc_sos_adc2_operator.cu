/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "thc_sos_adc2_operator.hpp"
#include "gpu_manager.hpp"
#include "multi_gpu_manager.hpp"
#include "laplace_quadrature.hpp"
#include "utils.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace gansu {

namespace {

inline void check_cuda(cudaError_t e, const char* tag) {
    if (e != cudaSuccess) {
        THROW_EXCEPTION(std::string("CUDA error: ") + tag
                        + ": " + cudaGetErrorString(e));
    }
}
inline void check_cublas(cublasStatus_t s, const char* tag) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        THROW_EXCEPTION(std::string("cuBLAS error: ") + tag);
    }
}

// =============================================================================
//  Device kernels
// =============================================================================

// X_occ_t[i, P] = X_mo[i, P] * exp(+ eps[i] * tau / 2)        (occ rows)
// X_vir_t[a, P] = X_mo[n_occ + a, P] * exp(-eps[n_occ + a] * tau / 2)
__global__ void thc_adc2_scale_X_kernel(
    const real_t* __restrict__ d_X_mo,
    const real_t* __restrict__ d_eps,
    real_t tau,
    int n_occ, int n_vir, int N_orb, int N_g,
    real_t* d_X_occ_t,
    real_t* d_X_vir_t)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    if (P >= N_g || p >= N_orb) return;

    const real_t x = d_X_mo[p + P * N_orb];
    const real_t half_tau = tau * real_t(0.5);
    if (p < n_occ) {
        d_X_occ_t[p + P * n_occ] = x * exp(d_eps[p] * half_tau);
    } else {
        const int a_off = p - n_occ;
        d_X_vir_t[a_off + P * n_vir] = x * exp(-d_eps[p] * half_tau);
    }
}

// Khatri-Rao: M[(ia), P] = X_occ[i, P] * X_vir[a, P]
//   M is (ov, N_g) col-major,   X_occ (n_occ, N_g),  X_vir (n_vir, N_g).
__global__ void thc_adc2_khatri_rao_kernel(
    const real_t* __restrict__ d_X_occ,
    const real_t* __restrict__ d_X_vir,
    int n_occ, int n_vir, int N_g,
    real_t* d_M)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    const int P  = blockIdx.y * blockDim.y + threadIdx.y;
    const int ov = n_occ * n_vir;
    if (ia >= ov || P >= N_g) return;

    const int i = ia % n_occ;
    const int a = ia / n_occ;
    d_M[ia + P * ov] = d_X_occ[i + P * n_occ] * d_X_vir[a + P * n_vir];
}

// F[(ia), P] = M[(ia), P] * x[ia]   (broadcast trial vector along grid)
__global__ void thc_adc2_broadcast_x_kernel(
    const real_t* __restrict__ d_M,
    const real_t* __restrict__ d_x,
    int ov, int N_g,
    real_t* d_F)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    const int P  = blockIdx.y * blockDim.y + threadIdx.y;
    if (ia >= ov || P >= N_g) return;
    d_F[ia + P * ov] = d_M[ia + P * ov] * d_x[ia];
}

// Row-wise dot:  d_sigma[ia] += pref * sum_P d_M[ia, P] * d_MT[ia, P]
__global__ void thc_adc2_rowdot_kernel(
    const real_t* __restrict__ d_M,
    const real_t* __restrict__ d_MT,
    real_t* d_sigma,
    real_t pref, int ov, int N_g)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= ov) return;
    real_t acc = 0.0;
    for (int P = 0; P < N_g; ++P) {
        acc += d_M[ia + P * ov] * d_MT[ia + P * ov];
    }
    d_sigma[ia] += pref * acc;
}

// d_diag[ia] = D1[ia] + M11_diagonal_addition  -- but here we just copy D1
__global__ void thc_adc2_d1_kernel(
    const real_t* __restrict__ d_eps,
    int n_occ, int n_vir,
    real_t* d_D1)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    const int ov = n_occ * n_vir;
    if (ia >= ov) return;
    const int i = ia % n_occ;
    const int a = ia / n_occ;
    d_D1[ia] = d_eps[n_occ + a] - d_eps[i];
}

// Preconditioner: out[ia] = in[ia] / (D1[ia] - omega), guarded against
// near-zero denominators.
__global__ void thc_adc2_precondition_kernel(
    const real_t* __restrict__ d_in,
    const real_t* __restrict__ d_D1,
    real_t omega,
    real_t* d_out, int ov)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= ov) return;
    const real_t denom = d_D1[ia] - omega;
    d_out[ia] = (fabs(static_cast<double>(denom)) > 1.0e-8)
                ? d_in[ia] / denom
                : d_in[ia];
}

// d_M11[(ia, jb)] = D1[ia] * delta_{ia,jb}  + raw_J * 2 - raw_K
//   where raw_J[ia, jb] = (ia|jb), raw_K[ia, jb] = (ij|ab)
//   (called after the J and K matrices have been formed in d_M11 area)
// Here we pass: d_M11 = -raw_K + 2 * raw_J  (already accumulated), and add D1
__global__ void thc_adc2_add_d1_diag_kernel(
    const real_t* __restrict__ d_D1,
    real_t* d_M11,
    int ov)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= ov) return;
    d_M11[ia + ia * ov] += d_D1[ia];
}

// Permute (ij|ab) tensor into M11[(ia), (jb)] position with a SUBTRACTION:
//   Given d_ERI_oovv[ij, ab]  (oo, vv) col-major where
//     d_ERI_oovv[i + j*n_occ + (a + b*n_vir)*oo] = (ij|ab),
//   subtract from d_M11[(ia), (jb)] = d_M11[i + a*n_occ + (j + b*n_occ)*ov]
//   the value (ij|ab).
__global__ void thc_adc2_subtract_K_into_M11_kernel(
    const real_t* __restrict__ d_ERI_oovv,
    real_t* d_M11,
    int n_occ, int n_vir)
{
    const int n_ov = n_occ * n_vir;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_ov * n_ov;
    if (idx >= total) return;

    const int row = idx % n_ov;       // (ia)
    const int col = idx / n_ov;       // (jb)
    const int i = row % n_occ;
    const int a = row / n_occ;
    const int j = col % n_occ;
    const int b = col / n_occ;

    // (ij|ab) at d_ERI_oovv[i + j*n_occ + (a + b*n_vir)*oo]
    const int oo = n_occ * n_occ;
    const std::size_t k_idx = static_cast<std::size_t>(i + j * n_occ)
                              + static_cast<std::size_t>(a + b * n_vir) * oo;
    d_M11[idx] -= d_ERI_oovv[k_idx];
}

// Diagonal kernel for the preconditioner:
//   d_diagonal[ia] = D1[ia]  (rough; ignore M11 off-diagonal contributions)
__global__ void thc_adc2_set_diagonal_from_D1_kernel(
    const real_t* __restrict__ d_D1,
    real_t* d_diagonal,
    int ov)
{
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= ov) return;
    d_diagonal[ia] = d_D1[ia];
}

// =============================================================================
//  B3 / A3 helper kernels  (Phase 2.2b)
// =============================================================================

// Full-Laplace virtual collocation:
//   X̃̃_vir[a, P] = X[n_occ + a, P] * exp(-eps[n_occ+a] * tau)
__global__ void thc_adc2_scale_X_full_vir_kernel(
    const real_t* __restrict__ d_X_mo,
    const real_t* __restrict__ d_eps,
    real_t tau,
    int n_occ, int n_vir, int N_orb, int N_g,
    real_t* d_X2_vir)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    const int a = blockIdx.y * blockDim.y + threadIdx.y;
    if (P >= N_g || a >= n_vir) return;
    const int p = n_occ + a;
    const real_t x = d_X_mo[p + P * N_orb];
    d_X2_vir[a + P * n_vir] = x * exp(-d_eps[p] * tau);
}

// Full-Laplace occupied collocation:
//   X̃̃_occ[i, P] = X[i, P] * exp(+eps[i] * tau)
__global__ void thc_adc2_scale_X_full_occ_kernel(
    const real_t* __restrict__ d_X_mo,
    const real_t* __restrict__ d_eps,
    real_t tau,
    int n_occ, int N_orb, int N_g,
    real_t* d_X2_occ)
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (P >= N_g || i >= n_occ) return;
    const real_t x = d_X_mo[i + P * N_orb];
    d_X2_occ[i + P * n_occ] = x * exp(d_eps[i] * tau);
}

// Element-wise Hadamard A *= B  (in-place; A and B both N x N col-major).
__global__ void thc_adc2_hadamard_inplace_kernel(
    real_t* d_A,
    const real_t* __restrict__ d_B,
    std::size_t total)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    d_A[idx] *= d_B[idx];
}

// Fused B3 post-scale + axpy:
//   sigma[i + a*n_occ] += factor * exp(-eps_vir[a] * tau) * raw[i + a*n_occ]
//
// (THC index convention: ia = a * n_occ + i, i.e. (n_occ × n_vir) col-major)
__global__ void thc_adc2_b3_postscale_axpy_kernel(
    const real_t* __restrict__ d_raw,
    const real_t* __restrict__ d_eps,
    real_t* d_sigma,
    real_t factor, real_t tau,
    int n_occ, int n_vir)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ov = n_occ * n_vir;
    if (idx >= ov) return;
    const int a = idx / n_occ;
    const real_t scale = exp(-d_eps[n_occ + a] * tau);
    d_sigma[idx] += factor * scale * d_raw[idx];
}

} // anonymous namespace

// =============================================================================
//  Constructor / destructor
// =============================================================================

THCSOSADC2Operator::THCSOSADC2Operator(const real_t* d_X_mo,
                                       const real_t* d_Z,
                                       const real_t* d_orbital_energies,
                                       int n_occ, int n_vir, int N_orb, int N_g,
                                       double c_os,
                                       int n_laplace,
                                       int num_gpus,
                                       bool enable_b3a3,
                                       bool enable_b3,
                                       bool enable_a3)
    : n_occ_(n_occ), n_vir_(n_vir), N_orb_(N_orb), N_g_(N_g),
      singles_dim_(n_occ * n_vir),
      c_os_(c_os), n_laplace_(n_laplace),
      enable_b3a3_(enable_b3a3),
      enable_b3_(enable_b3a3 && enable_b3),
      enable_a3_(enable_b3a3 && enable_a3),
      d_X_mo_(d_X_mo), d_Z_(d_Z), d_orbital_energies_(d_orbital_energies),
      num_gpus_(num_gpus < 1 ? 1 : num_gpus)
{
    eps_h_.resize(N_orb_);
    check_cuda(cudaMemcpy(eps_h_.data(), d_orbital_energies_,
                          N_orb_ * sizeof(real_t), cudaMemcpyDeviceToHost),
               "copy eps to host");

    const std::size_t ov = static_cast<std::size_t>(singles_dim_);
    const std::size_t Ng = static_cast<std::size_t>(N_g_);

    // ---- GPU 0 only: M11, D1, diagonal ----
    check_cuda(cudaMalloc(&d_D1_, ov * sizeof(real_t)), "alloc D1");
    check_cuda(cudaMalloc(&d_diagonal_, ov * sizeof(real_t)), "alloc diagonal");
    check_cuda(cudaMalloc(&d_M11_, ov * ov * sizeof(real_t)), "alloc M11");

    // Negotiate active GPU count via MultiGpuManager (auto cap at available).
    if (num_gpus_ > 1) {
        auto& mgr = MultiGpuManager::instance();
        mgr.initialize(num_gpus_);
        num_gpus_ = mgr.num_devices();
    }

    // ---- Allocate per-GPU buffer vectors ----
    d_X_mo_per_.assign(num_gpus_, nullptr);
    d_Z_per_.assign(num_gpus_, nullptr);
    d_eps_per_.assign(num_gpus_, nullptr);
    d_X_occ_t_per_.assign(num_gpus_, nullptr);
    d_X_vir_t_per_.assign(num_gpus_, nullptr);
    d_M_per_.assign(num_gpus_, nullptr);
    d_F_per_.assign(num_gpus_, nullptr);
    d_Y_PQ_per_.assign(num_gpus_, nullptr);
    d_ZY_per_.assign(num_gpus_, nullptr);
    d_T_per_.assign(num_gpus_, nullptr);
    d_MT_per_.assign(num_gpus_, nullptr);
    d_input_per_.assign(num_gpus_, nullptr);
    d_sigma_partial_per_.assign(num_gpus_, nullptr);

    // B3/A3 per-GPU vectors (always sized; pointers stay nullptr if disabled).
    d_Z_occ_per_.assign(num_gpus_, nullptr);
    d_W_per_.assign(num_gpus_, nullptr);
    d_U_per_.assign(num_gpus_, nullptr);
    d_yB3_per_.assign(num_gpus_, nullptr);
    d_X2vir_per_.assign(num_gpus_, nullptr);
    d_X2occ_per_.assign(num_gpus_, nullptr);
    d_tmp1_per_.assign(num_gpus_, nullptr);
    d_tmpB3_per_.assign(num_gpus_, nullptr);
    d_sig_corr_per_.assign(num_gpus_, nullptr);

    // GPU 0: alias the caller's pointers (no copy needed for inputs).
    d_X_mo_per_[0] = const_cast<real_t*>(d_X_mo_);
    d_Z_per_[0]    = const_cast<real_t*>(d_Z_);
    d_eps_per_[0]  = const_cast<real_t*>(d_orbital_energies_);

    auto alloc_workspaces_on_current_device = [&](int d) {
        check_cuda(cudaMalloc(&d_X_occ_t_per_[d], static_cast<std::size_t>(n_occ_) * N_g_ * sizeof(real_t)), "alloc X_occ_t");
        check_cuda(cudaMalloc(&d_X_vir_t_per_[d], static_cast<std::size_t>(n_vir_) * N_g_ * sizeof(real_t)), "alloc X_vir_t");
        check_cuda(cudaMalloc(&d_M_per_[d], ov * Ng * sizeof(real_t)), "alloc M");
        check_cuda(cudaMalloc(&d_F_per_[d], ov * Ng * sizeof(real_t)), "alloc F");
        check_cuda(cudaMalloc(&d_Y_PQ_per_[d], Ng * Ng * sizeof(real_t)), "alloc Y");
        check_cuda(cudaMalloc(&d_ZY_per_[d],   Ng * Ng * sizeof(real_t)), "alloc ZY");
        check_cuda(cudaMalloc(&d_T_per_[d],    Ng * Ng * sizeof(real_t)), "alloc T");
        check_cuda(cudaMalloc(&d_MT_per_[d],   ov * Ng * sizeof(real_t)), "alloc MT");
        check_cuda(cudaMalloc(&d_input_per_[d], ov * sizeof(real_t)), "alloc input");
        check_cuda(cudaMalloc(&d_sigma_partial_per_[d], ov * sizeof(real_t)), "alloc sigma_partial");

        if (enable_b3a3_) {
            check_cuda(cudaMalloc(&d_Z_occ_per_[d], Ng * Ng * sizeof(real_t)), "alloc Z_occ (B3/A3)");
            check_cuda(cudaMalloc(&d_W_per_[d],     Ng * Ng * sizeof(real_t)), "alloc W (B3/A3)");
            check_cuda(cudaMalloc(&d_U_per_[d],     Ng * Ng * sizeof(real_t)), "alloc U (B3/A3)");
            check_cuda(cudaMalloc(&d_yB3_per_[d],
                                  static_cast<std::size_t>(N_g_) * n_vir_ * sizeof(real_t)),
                       "alloc yB3 (B3/A3)");
            check_cuda(cudaMalloc(&d_X2vir_per_[d],
                                  static_cast<std::size_t>(n_vir_) * N_g_ * sizeof(real_t)),
                       "alloc X2_vir (B3/A3)");
            check_cuda(cudaMalloc(&d_X2occ_per_[d],
                                  static_cast<std::size_t>(n_occ_) * N_g_ * sizeof(real_t)),
                       "alloc X2_occ (B3/A3)");
            check_cuda(cudaMalloc(&d_tmp1_per_[d],
                                  static_cast<std::size_t>(n_vir_) * N_g_ * sizeof(real_t)),
                       "alloc tmp1 (A3)");
            check_cuda(cudaMalloc(&d_tmpB3_per_[d],
                                  static_cast<std::size_t>(n_occ_) * N_g_ * sizeof(real_t)),
                       "alloc tmpB3 (B3)");
            check_cuda(cudaMalloc(&d_sig_corr_per_[d], ov * sizeof(real_t)),
                       "alloc sig_corr (B3/A3)");
        }
    };

    // GPU 0 workspaces
    alloc_workspaces_on_current_device(0);

#ifndef GANSU_CPU_ONLY
    // Peer replicas + workspaces on GPUs > 0.
    if (num_gpus_ > 1) {
        const std::size_t X_size   = static_cast<std::size_t>(N_orb_) * N_g_ * sizeof(real_t);
        const std::size_t Z_size   = Ng * Ng * sizeof(real_t);
        const std::size_t eps_size = static_cast<std::size_t>(N_orb_) * sizeof(real_t);
        for (int d = 1; d < num_gpus_; ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            check_cuda(cudaMalloc(&d_X_mo_per_[d], X_size), "alloc X_mo peer");
            check_cuda(cudaMalloc(&d_Z_per_[d],    Z_size), "alloc Z peer");
            check_cuda(cudaMalloc(&d_eps_per_[d],  eps_size), "alloc eps peer");
            cudaMemcpy(d_X_mo_per_[d], d_X_mo_, X_size,   cudaMemcpyDefault);
            cudaMemcpy(d_Z_per_[d],    d_Z_,    Z_size,   cudaMemcpyDefault);
            cudaMemcpy(d_eps_per_[d],  d_orbital_energies_, eps_size, cudaMemcpyDefault);
            alloc_workspaces_on_current_device(d);
        }
    }
#endif

    compute_D1();   // runs on current device (GPU 0)
    build_M11();    // runs on GPU 0 with d_X_mo_/d_Z_

    update_laplace_quadrature();
}

THCSOSADC2Operator::~THCSOSADC2Operator()
{
    // GPU 0 only: M11, D1, diagonal
    if (d_M11_)      cudaFree(d_M11_);
    if (d_D1_)       cudaFree(d_D1_);
    if (d_diagonal_) cudaFree(d_diagonal_);

    // Per-GPU workspaces and peer replicas
    for (int d = 0; d < num_gpus_; ++d) {
#ifndef GANSU_CPU_ONLY
        MultiGpuManager::DeviceGuard guard(d);
#endif
        if (d_X_occ_t_per_[d])        cudaFree(d_X_occ_t_per_[d]);
        if (d_X_vir_t_per_[d])        cudaFree(d_X_vir_t_per_[d]);
        if (d_M_per_[d])              cudaFree(d_M_per_[d]);
        if (d_F_per_[d])              cudaFree(d_F_per_[d]);
        if (d_Y_PQ_per_[d])           cudaFree(d_Y_PQ_per_[d]);
        if (d_ZY_per_[d])             cudaFree(d_ZY_per_[d]);
        if (d_T_per_[d])              cudaFree(d_T_per_[d]);
        if (d_MT_per_[d])             cudaFree(d_MT_per_[d]);
        if (d_input_per_[d])          cudaFree(d_input_per_[d]);
        if (d_sigma_partial_per_[d])  cudaFree(d_sigma_partial_per_[d]);

        if (d_Z_occ_per_[d])          cudaFree(d_Z_occ_per_[d]);
        if (d_W_per_[d])              cudaFree(d_W_per_[d]);
        if (d_U_per_[d])              cudaFree(d_U_per_[d]);
        if (d_yB3_per_[d])            cudaFree(d_yB3_per_[d]);
        if (d_X2vir_per_[d])          cudaFree(d_X2vir_per_[d]);
        if (d_X2occ_per_[d])          cudaFree(d_X2occ_per_[d]);
        if (d_tmp1_per_[d])           cudaFree(d_tmp1_per_[d]);
        if (d_tmpB3_per_[d])          cudaFree(d_tmpB3_per_[d]);
        if (d_sig_corr_per_[d])       cudaFree(d_sig_corr_per_[d]);

        if (d > 0) {
            // Free peer replicas (GPU 0 aliases caller's pointers).
            if (d_X_mo_per_[d])  cudaFree(d_X_mo_per_[d]);
            if (d_Z_per_[d])     cudaFree(d_Z_per_[d]);
            if (d_eps_per_[d])   cudaFree(d_eps_per_[d]);
        }
    }
}

void THCSOSADC2Operator::compute_D1()
{
    const int threads = 256;
    const int blocks  = (singles_dim_ + threads - 1) / threads;
    thc_adc2_d1_kernel<<<blocks, threads>>>(d_orbital_energies_, n_occ_, n_vir_, d_D1_);
    check_cuda(cudaGetLastError(), "thc_adc2_d1_kernel");

    thc_adc2_set_diagonal_from_D1_kernel<<<blocks, threads>>>(d_D1_, d_diagonal_, singles_dim_);
    check_cuda(cudaGetLastError(), "thc_adc2_set_diagonal_from_D1_kernel");
}

// =============================================================================
//  M11 = D1 + 2(ia|jb) - (ij|ab)  via THC
// =============================================================================
void THCSOSADC2Operator::build_M11()
{
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    const real_t one = 1.0;
    const real_t zero = 0.0;
    const real_t two = 2.0;
    const real_t neg_one = -1.0;

    const int ov = singles_dim_;
    const int oo = n_occ_ * n_occ_;
    const int vv = n_vir_ * n_vir_;
    const std::size_t Ng = static_cast<std::size_t>(N_g_);

    auto t0 = std::chrono::steady_clock::now();
    auto plog = [&t0](const char* msg) {
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::cout << "    [build_M11 t=" << std::fixed << std::setprecision(1)
                  << s << "s, GPU 0] " << msg << std::endl;
    };

    // ---- 1. Build M_ia[(ia), P] = X^P_i X^P_a  (ov x N_g col-major)
    // We need separate X_occ and X_vir.  At this point we use the unscaled MO
    // collocation (no Laplace), so we can just take the relevant rows of d_X_mo_.
    // X_occ  = first n_occ rows of d_X_mo_ (col-major lda = N_orb_)
    // X_vir  = next n_vir rows.
    // Our Khatri-Rao kernel expects standalone X_occ/X_vir buffers; reuse
    // d_X_occ_t_ / d_X_vir_t_ workspace by writing directly here.
    {
        const dim3 threads(16, 16);
        const dim3 blocks((N_g_ + 15) / 16, (N_orb_ + 15) / 16);
        thc_adc2_scale_X_kernel<<<blocks, threads>>>(
            d_X_mo_, d_orbital_energies_, /*tau=*/0.0, // tau=0 -> exp(0) = 1
            n_occ_, n_vir_, N_orb_, N_g_,
            d_X_occ_t_per_[0], d_X_vir_t_per_[0]);
        check_cuda(cudaGetLastError(), "scale_X (build_M11, tau=0)");
    }
    {
        const dim3 threads(16, 16);
        const dim3 blocks((ov + 15) / 16, (N_g_ + 15) / 16);
        thc_adc2_khatri_rao_kernel<<<blocks, threads>>>(
            d_X_occ_t_per_[0], d_X_vir_t_per_[0], n_occ_, n_vir_, N_g_, d_M_per_[0]);
        check_cuda(cudaGetLastError(), "khatri_rao (build_M11)");
    }

    // ---- 2. (ia|jb) tensor as (ov × ov) matrix:
    //         J = M_ia · Z · M_ia^T     (ov × ov col-major)
    //   step 2a: tmp = M Z          (ov × N_g)
    //   step 2b: J   = tmp M^T      (ov × ov)
    real_t* d_J = nullptr;
    check_cuda(cudaMalloc(&d_J, static_cast<std::size_t>(ov) * ov * sizeof(real_t)),
               "alloc J (M11)");

    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              ov, N_g_, N_g_,
                              &one, d_M_per_[0], ov, d_Z_per_[0], N_g_,
                              &zero, d_MT_per_[0], ov),  // reuse MT as tmp
                 "M Z");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              ov, ov, N_g_,
                              &one, d_MT_per_[0], ov, d_M_per_[0], ov,
                              &zero, d_J, ov),
                 "(M Z) M^T");
    cudaDeviceSynchronize();
    plog("J = M·Z·M^T done");

    // ---- 3. (ij|ab) tensor (oo × vv col-major):
    //         M_ij[(ij), P] = X^P_i X^P_j
    //         M_ab[(ab), P] = X^P_a X^P_b
    //         K_oovv = M_ij · Z · M_ab^T
    //
    // Memory-friendly ordering:
    //   (a) Build M_ij, then tmp = M_ij · Z, then FREE M_ij;
    //   (b) Build M_ab, then K_oovv = tmp · M_ab^T, then FREE tmp + M_ab.
    // This caps the peak at d_J + max(M_ij+tmp, tmp+M_ab+K_oovv) instead of
    // having all four (M_ij + M_ab + tmp + K_oovv) alive simultaneously.
    real_t* d_M_ij = nullptr;
    check_cuda(cudaMalloc(&d_M_ij, static_cast<std::size_t>(oo) * Ng * sizeof(real_t)),
               "alloc M_ij");
    {
        const dim3 threads(16, 16);
        const dim3 blocks((oo + 15) / 16, (N_g_ + 15) / 16);
        thc_adc2_khatri_rao_kernel<<<blocks, threads>>>(
            d_X_occ_t_per_[0], d_X_occ_t_per_[0], n_occ_, n_occ_, N_g_, d_M_ij);
        check_cuda(cudaGetLastError(), "khatri_rao M_ij");
    }

    real_t* d_tmp = nullptr;
    check_cuda(cudaMalloc(&d_tmp, static_cast<std::size_t>(oo) * Ng * sizeof(real_t)),
               "alloc tmp K");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                              oo, N_g_, N_g_,
                              &one, d_M_ij, oo, d_Z_per_[0], N_g_,
                              &zero, d_tmp, oo),
                 "M_ij Z");
    cudaFree(d_M_ij);  // freed early — saves oo × N_g × 8 bytes

    real_t* d_M_ab = nullptr;
    check_cuda(cudaMalloc(&d_M_ab, static_cast<std::size_t>(vv) * Ng * sizeof(real_t)),
               "alloc M_ab");
    {
        const dim3 threads(16, 16);
        const dim3 blocks((vv + 15) / 16, (N_g_ + 15) / 16);
        thc_adc2_khatri_rao_kernel<<<blocks, threads>>>(
            d_X_vir_t_per_[0], d_X_vir_t_per_[0], n_vir_, n_vir_, N_g_, d_M_ab);
        check_cuda(cudaGetLastError(), "khatri_rao M_ab");
    }

    real_t* d_K_oovv = nullptr;
    check_cuda(cudaMalloc(&d_K_oovv, static_cast<std::size_t>(oo) * vv * sizeof(real_t)),
               "alloc K_oovv");
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              oo, vv, N_g_,
                              &one, d_tmp, oo, d_M_ab, vv,
                              &zero, d_K_oovv, oo),
                 "(M_ij Z) M_ab^T");
    cudaFree(d_tmp);
    cudaFree(d_M_ab);  // freed early — saves vv × N_g × 8 bytes
    cudaDeviceSynchronize();
    plog("K_oovv = M_ij·Z·M_ab^T done");

    // ---- 4. Assemble M11 = 2 J - K (in (ia, jb) ordering) + D1 on diagonal
    // Start with M11 = 2 J
    {
        const std::size_t total = static_cast<std::size_t>(ov) * ov;
        check_cublas(cublasDcopy(cublas, static_cast<int>(total),
                                  d_J, 1, d_M11_, 1),
                     "copy J -> M11");
        check_cublas(cublasDscal(cublas, static_cast<int>(total),
                                  &two, d_M11_, 1),
                     "scale M11 by 2");
    }

    // M11 -= K_permuted_to_(ia,jb)
    {
        const int total = ov * ov;
        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;
        thc_adc2_subtract_K_into_M11_kernel<<<blocks, threads>>>(
            d_K_oovv, d_M11_, n_occ_, n_vir_);
        check_cuda(cudaGetLastError(), "subtract_K_into_M11");
    }

    // Add D1 to diagonal
    {
        const int threads = 256;
        const int blocks  = (ov + threads - 1) / threads;
        thc_adc2_add_d1_diag_kernel<<<blocks, threads>>>(d_D1_, d_M11_, ov);
        check_cuda(cudaGetLastError(), "add_d1_diag");
    }

    cudaFree(d_J);
    cudaFree(d_K_oovv);
    // d_M_ij and d_M_ab freed early above (memory-friendly ordering)
    plog("M11 = 2J - K + D1 assembled");
}

// =============================================================================
//  ω management
// =============================================================================
void THCSOSADC2Operator::set_omega(real_t omega)
{
    omega_ = omega;
    update_laplace_quadrature();
}

void THCSOSADC2Operator::update_laplace_quadrature()
{
    const real_t e_homo = eps_h_[n_occ_ - 1];
    const real_t e_lumo = eps_h_[n_occ_];
    const real_t e_occ_min = eps_h_[0];
    const real_t e_vir_max = eps_h_[N_orb_ - 1];
    // ω-shifted denominator: x = Δ - ω, x_min = (e_lumo - e_homo) - ω, etc.
    double x_min = static_cast<double>(e_lumo - e_homo) - static_cast<double>(omega_);
    double x_max = static_cast<double>(e_vir_max - e_occ_min) - static_cast<double>(omega_);
    if (x_min < 1.0e-3) x_min = 1.0e-3;
    if (x_max < x_min + 0.1) x_max = x_min + 10.0;

    LaplaceQuadrature q = generate_laplace_quadrature(x_min, x_max, n_laplace_);
    laplace_t_.assign(q.points.begin(), q.points.end());
    laplace_w_.assign(q.weights.begin(), q.weights.end());
}

// =============================================================================
//  Sigma vector application: σ = M_eff(ω) x
//
//   σ = M11 x  +  c_os Σ_τ w_τ e^{ω t_τ} · σ_τ(x)
//   σ_τ(x)_ia = Σ_PR M̃[(ia),P] M̃[(ia),R] [Z Y(τ,x) Z^T]_{P,R}
//   Y_{Q,S}(τ,x) = Σ_kc M̃[(kc),Q] M̃[(kc),S] x_kc      (= F^T M̃)
//   F[(kc),Q] = M̃[(kc),Q] * x_kc
//   M̃[(kc),P] = X̃^P_k X̃^P_c   (Laplace-scaled MO collocation × Khatri-Rao)
// =============================================================================
namespace {

// Per-GPU σ build for a τ subset.  Writes the τ contribution to d_sigma_partial
// (overwrites; caller is responsible for zeroing first if accumulation desired).
// Uses workspaces / inputs that already live on the current device.
//
// If enable_b3a3 is true, the B3-exchange and A3-Coulomb correction terms are
// added inside the τ loop; the corresponding workspace pointers must be
// non-null in that case.
void thc_sos_adc2_sigma_partial(
    int tau_start, int tau_end,
    const std::vector<double>& laplace_t,
    const std::vector<double>& laplace_w,
    real_t omega, real_t c_os,
    int n_occ, int n_vir, int N_orb, int N_g, int ov,
    const real_t* d_X_mo,
    const real_t* d_Z,
    const real_t* d_eps,
    const real_t* d_input,
    real_t* d_sigma_partial,
    real_t* d_X_occ_t, real_t* d_X_vir_t,
    real_t* d_M, real_t* d_F,
    real_t* d_Y_PQ, real_t* d_ZY, real_t* d_T, real_t* d_MT,
    cublasHandle_t cublas,
    bool enable_b3a3, bool enable_b3, bool enable_a3,
    real_t* d_Z_occ, real_t* d_W, real_t* d_U,
    real_t* d_yB3, real_t* d_X2vir, real_t* d_X2occ,
    real_t* d_tmp1, real_t* d_tmpB3, real_t* d_sig_corr)
{
    const real_t one = 1.0, zero = 0.0;

    // Initialise partial σ to zero
    cudaMemset(d_sigma_partial, 0, ov * sizeof(real_t));

    for (int it = tau_start; it < tau_end; ++it) {
        const real_t tau = static_cast<real_t>(laplace_t[it]);
        const real_t w   = static_cast<real_t>(laplace_w[it]);
        // M_eff = M11 - M12 (D2-ω)^{-1} M21^T  →  Schur subtracts (negative pref).
        // (Phase 2.2a inadvertently used +c_os; symptom hidden by small magnitude.)
        const real_t pref = -c_os * w
                          * static_cast<real_t>(std::exp(static_cast<double>(omega) * static_cast<double>(tau)));

        // Scale X
        {
            const dim3 threads(16, 16);
            const dim3 blocks((N_g + 15) / 16, (N_orb + 15) / 16);
            thc_adc2_scale_X_kernel<<<blocks, threads>>>(
                d_X_mo, d_eps, tau, n_occ, n_vir, N_orb, N_g,
                d_X_occ_t, d_X_vir_t);
            check_cuda(cudaGetLastError(), "scale_X (partial)");
        }
        // Khatri-Rao M = X_occ_t * X_vir_t
        {
            const dim3 threads(16, 16);
            const dim3 blocks((ov + 15) / 16, (N_g + 15) / 16);
            thc_adc2_khatri_rao_kernel<<<blocks, threads>>>(
                d_X_occ_t, d_X_vir_t, n_occ, n_vir, N_g, d_M);
            check_cuda(cudaGetLastError(), "khatri_rao (partial)");
        }
        // F = M ⊙ x_input
        {
            const dim3 threads(16, 16);
            const dim3 blocks((ov + 15) / 16, (N_g + 15) / 16);
            thc_adc2_broadcast_x_kernel<<<blocks, threads>>>(
                d_M, d_input, ov, N_g, d_F);
            check_cuda(cudaGetLastError(), "broadcast_x (partial)");
        }
        // Y = F^T M
        check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  N_g, N_g, ov,
                                  &one, d_F, ov, d_M, ov,
                                  &zero, d_Y_PQ, N_g),
                     "Y = F^T M (partial)");
        // ZY = Z Y
        check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N_g, N_g, N_g,
                                  &one, d_Z, N_g, d_Y_PQ, N_g,
                                  &zero, d_ZY, N_g),
                     "Z Y (partial)");
        // T = ZY Z^T
        check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                  N_g, N_g, N_g,
                                  &one, d_ZY, N_g, d_Z, N_g,
                                  &zero, d_T, N_g),
                     "ZY Z^T (partial)");
        // MT = M T
        check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  ov, N_g, N_g,
                                  &one, d_M, ov, d_T, N_g,
                                  &zero, d_MT, ov),
                     "M T (partial)");
        // σ_partial_ia += pref Σ_P M[ia,P] MT[ia,P]
        {
            const int threads = 256;
            const int blocks  = (ov + threads - 1) / threads;
            thc_adc2_rowdot_kernel<<<blocks, threads>>>(
                d_M, d_MT, d_sigma_partial, pref, ov, N_g);
            check_cuda(cudaGetLastError(), "rowdot (partial)");
        }

        // -------------------------------------------------------------------
        //  Phase 2.2b: B3-exchange + A3-Coulomb correction  (THC analog)
        //
        //  Both reuse the Laplace-scaled Khatri-Rao factors (X̃_occ, X̃_vir)
        //  and the LS-THC core Z that are already in scope.  The shared
        //  intermediate matrices for one τ are
        //
        //     Z̃_occ_PP' = X̃_occ^T X̃_occ            (N_g × N_g)
        //     Z̃_vir_PP' = X̃_vir^T X̃_vir            (N_g × N_g)
        //     W_PP'     = Z̃_occ ⊙ Z̃_vir            (jb-sum in THC view)
        //     U_PP'     = Z W Z^T                   (N_g × N_g)
        //     y_P'E     = X_occ_unscaled^T x        (N_g × n_vir)
        //
        //  A3 is run before B3 so that U is still alive (B3 overwrites U with
        //  the Hadamard U ⊙ Z̃_occ).
        // -------------------------------------------------------------------
        if (enable_b3a3 && (enable_b3 || enable_a3)) {
            const std::size_t Ng2 = static_cast<std::size_t>(N_g) * N_g;

            // 1) Z̃_occ = X̃_occ^T X̃_occ          (N_g × N_g)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N_g, N_g, n_occ,
                                      &one, d_X_occ_t, n_occ,
                                            d_X_occ_t, n_occ,
                                      &zero, d_Z_occ, N_g),
                         "Z_occ = X̃_occ^T X̃_occ");

            // 2) W = X̃_vir^T X̃_vir   (write into d_W; will become Hadamard target)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N_g, N_g, n_vir,
                                      &one, d_X_vir_t, n_vir,
                                            d_X_vir_t, n_vir,
                                      &zero, d_W, N_g),
                         "Z_vir = X̃_vir^T X̃_vir");

            // 3) W ⊙= Z̃_occ                       (W now holds Z̃_occ ⊙ Z̃_vir)
            {
                const int threads = 256;
                const int blocks  = static_cast<int>((Ng2 + threads - 1) / threads);
                thc_adc2_hadamard_inplace_kernel<<<blocks, threads>>>(
                    d_W, d_Z_occ, Ng2);
                check_cuda(cudaGetLastError(), "Hadamard W ⊙= Z_occ");
            }

            // 4) U = Z W Z^T   (two DGEMMs through d_T as scratch)
            //    d_T is the existing 2.2a workspace; safe to reuse here because
            //    the Coulomb step has already finished consuming it.
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      N_g, N_g, N_g,
                                      &one, d_Z, N_g,
                                            d_W, N_g,
                                      &zero, d_T, N_g),
                         "ZW (B3/A3)");
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      N_g, N_g, N_g,
                                      &one, d_T, N_g,
                                            d_Z, N_g,
                                      &zero, d_U, N_g),
                         "U = (ZW) Z^T (B3/A3)");

            // 5) y[P', E] = X_occ_unscaled^T · x       (N_g × n_vir col-major)
            //    X_occ_unscaled lives in d_X_mo with lda = N_orb (occ rows 0..n_occ-1).
            //    x viewed as (n_occ × n_vir) col-major lda = n_occ.
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N_g, n_vir, n_occ,
                                      &one, d_X_mo, N_orb,
                                            d_input, n_occ,
                                      &zero, d_yB3, N_g),
                         "y = X_occ^T x (B3/A3)");

            // ---------------- A3-Coulomb (BEFORE U is consumed) ----------------
            if (enable_a3) {
            // X̃̃_vir[a, P] = X[n_occ+a, P] × exp(-eps[n_occ+a] τ)   (full-t)
            {
                const dim3 threads(16, 16);
                const dim3 blocks((N_g + 15) / 16, (n_vir + 15) / 16);
                thc_adc2_scale_X_full_vir_kernel<<<blocks, threads>>>(
                    d_X_mo, d_eps, tau, n_occ, n_vir, N_orb, N_g, d_X2vir);
                check_cuda(cudaGetLastError(), "scale_X_full_vir (A3)");
            }
            // X̃̃_occ[i, P] = X[i, P] × exp(+eps[i] τ)                (full-t)
            {
                const dim3 threads(16, 16);
                const dim3 blocks((N_g + 15) / 16, (n_occ + 15) / 16);
                thc_adc2_scale_X_full_occ_kernel<<<blocks, threads>>>(
                    d_X_mo, d_eps, tau, n_occ, N_orb, N_g, d_X2occ);
                check_cuda(cudaGetLastError(), "scale_X_full_occ (A3)");
            }

            //   z_A3[P, P'] = Σ_C X̃̃_vir[C, P] · y[P', C]
            //              = X̃̃_vir^T · y^T   (matmul of N_g × n_vir and n_vir × N_g)
            //   We store z_A3 in d_W (the Z̃_occ ⊙ Z̃_vir buffer is no longer needed
            //   because U has already been formed).
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_T,
                                      N_g, N_g, n_vir,
                                      &one, d_X2vir, n_vir,
                                            d_yB3, N_g,
                                      &zero, d_W, N_g),
                         "z_A3 (A3)");

            //   V = z_A3 ⊙ U   (in-place into d_W)
            {
                const int threads = 256;
                const int blocks  = static_cast<int>((Ng2 + threads - 1) / threads);
                thc_adc2_hadamard_inplace_kernel<<<blocks, threads>>>(
                    d_W, d_U, Ng2);
                check_cuda(cudaGetLastError(), "Hadamard V = z_A3 ⊙ U (A3)");
            }

            //   tmp1[E, P'] = X_vir_unscaled · V       (n_vir × N_g)
            //   X_vir_unscaled = d_X_mo + n_occ rows, lda=N_orb, shape (n_vir × N_g)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n_vir, N_g, N_g,
                                      &one, d_X_mo + n_occ, N_orb,
                                            d_W, N_g,
                                      &zero, d_tmp1, n_vir),
                         "tmp1 = X_vir V (A3)");

            //   σ_a3[K, E] = X̃̃_occ · tmp1^T          (n_occ × n_vir col-major)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      n_occ, n_vir, N_g,
                                      &one, d_X2occ, n_occ,
                                            d_tmp1, n_vir,
                                      &zero, d_sig_corr, n_occ),
                         "σ_a3 = X̃̃_occ tmp1^T (A3)");

            //   σ_partial += a3_factor · σ_a3
            //   a3_factor = +2 c_os w e^{ωt}   (Coulomb と逆符号 ×2、RI と一致)
            {
                const real_t a3_factor = static_cast<real_t>(2.0)
                                       * c_os * w
                                       * static_cast<real_t>(std::exp(static_cast<double>(omega)
                                                                       * static_cast<double>(tau)));
                check_cublas(cublasDaxpy(cublas, ov, &a3_factor,
                                          d_sig_corr, 1,
                                          d_sigma_partial, 1),
                             "axpy A3 -> sigma_partial");
            }
            } // enable_a3

            // ---------------- B3-exchange (consumes U) ----------------
            if (enable_b3) {
            //   N_B3 = U ⊙ Z̃_occ                     (in-place into d_U)
            {
                const int threads = 256;
                const int blocks  = static_cast<int>((Ng2 + threads - 1) / threads);
                thc_adc2_hadamard_inplace_kernel<<<blocks, threads>>>(
                    d_U, d_Z_occ, Ng2);
                check_cuda(cudaGetLastError(), "Hadamard N_B3 = U ⊙ Z_occ (B3)");
            }

            //   tmp_B3[K, P'] = X_occ_unscaled · N_B3      (n_occ × N_g)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n_occ, N_g, N_g,
                                      &one, d_X_mo, N_orb,
                                            d_U, N_g,
                                      &zero, d_tmpB3, n_occ),
                         "tmp_B3 = X_occ N_B3 (B3)");

            //   σ_b3_raw[K, E] = tmp_B3 · y           (n_occ × n_vir)
            check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      n_occ, n_vir, N_g,
                                      &one, d_tmpB3, n_occ,
                                            d_yB3, N_g,
                                      &zero, d_sig_corr, n_occ),
                         "σ_b3_raw = tmp_B3 y (B3)");

            //   σ_partial += b3_factor · exp(-εE τ) · σ_b3_raw
            //   b3_factor = -2 c_os w e^{ωt}    (Coulomb と同符号 ×2、RI と一致)
            {
                const real_t b3_factor = -static_cast<real_t>(2.0)
                                       * c_os * w
                                       * static_cast<real_t>(std::exp(static_cast<double>(omega)
                                                                       * static_cast<double>(tau)));
                const int threads = 256;
                const int blocks  = (ov + threads - 1) / threads;
                thc_adc2_b3_postscale_axpy_kernel<<<blocks, threads>>>(
                    d_sig_corr, d_eps, d_sigma_partial,
                    b3_factor, tau, n_occ, n_vir);
                check_cuda(cudaGetLastError(), "B3 postscale + axpy");
            }
            } // enable_b3
        }
    }
    // Ensure all kernels on this device's stream complete before the host
    // returns -- otherwise GPU 0's later sum-of-partials may read stale data.
    cudaDeviceSynchronize();
}

} // anonymous namespace

void THCSOSADC2Operator::apply(const real_t* d_input, real_t* d_output) const
{
    const real_t one = 1.0;
    const real_t zero = 0.0;
    const int ov = singles_dim_;

    // ---- Step 1: σ = M11 · x  on GPU 0 ----
#ifndef GANSU_CPU_ONLY
    {
        MultiGpuManager::DeviceGuard guard(0);
#endif
        cublasHandle_t cublas0 = gpu::GPUHandle::cublas();
        check_cublas(cublasDgemv(cublas0, CUBLAS_OP_N,
                                  ov, ov,
                                  &one, d_M11_, ov,
                                  d_input, 1,
                                  &zero, d_output, 1),
                     "M11 * x");
        // Copy d_input into GPU 0's per-device buffer (used in partial sigma).
        cudaMemcpy(d_input_per_[0], d_input, ov * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
#ifndef GANSU_CPU_ONLY
    }

    // ---- Step 2: broadcast d_input to peer GPUs, and sync source GPU. ----
    if (num_gpus_ > 1) {
        for (int d = 1; d < num_gpus_; ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            cudaMemcpy(d_input_per_[d], d_input, ov * sizeof(real_t),
                       cudaMemcpyDefault);
        }
    }
    // Ensure GPU 0's default-stream M11*x and d_input copy are visible to
    // the per-device compute streams used in the next OpenMP region.
    {
        MultiGpuManager::DeviceGuard guard(0);
        cudaDeviceSynchronize();
    }
#endif

    // ---- Step 3: τ-parallel σ partial build per GPU ----
    if (num_gpus_ <= 1) {
        cublasHandle_t cublas0 = gpu::GPUHandle::cublas();
        thc_sos_adc2_sigma_partial(
            0, n_laplace_, laplace_t_, laplace_w_,
            omega_, static_cast<real_t>(c_os_),
            n_occ_, n_vir_, N_orb_, N_g_, ov,
            d_X_mo_per_[0], d_Z_per_[0], d_eps_per_[0],
            d_input_per_[0], d_sigma_partial_per_[0],
            d_X_occ_t_per_[0], d_X_vir_t_per_[0],
            d_M_per_[0], d_F_per_[0],
            d_Y_PQ_per_[0], d_ZY_per_[0], d_T_per_[0], d_MT_per_[0],
            cublas0,
            enable_b3a3_, enable_b3_, enable_a3_,
            d_Z_occ_per_[0], d_W_per_[0], d_U_per_[0],
            d_yB3_per_[0], d_X2vir_per_[0], d_X2occ_per_[0],
            d_tmp1_per_[0], d_tmpB3_per_[0], d_sig_corr_per_[0]);
    } else {
#ifndef GANSU_CPU_ONLY
        std::vector<std::string> err_msg(num_gpus_);
#ifdef _OPENMP
        #pragma omp parallel num_threads(num_gpus_)
        {
            const int d = omp_get_thread_num();
#else
        for (int d = 0; d < num_gpus_; ++d) {
#endif
            try {
                MultiGpuManager::DeviceGuard guard(d);
                const int t0 = (d * n_laplace_) / num_gpus_;
                const int t1 = ((d + 1) * n_laplace_) / num_gpus_;
                thc_sos_adc2_sigma_partial(
                    t0, t1, laplace_t_, laplace_w_,
                    omega_, static_cast<real_t>(c_os_),
                    n_occ_, n_vir_, N_orb_, N_g_, ov,
                    d_X_mo_per_[d], d_Z_per_[d], d_eps_per_[d],
                    d_input_per_[d], d_sigma_partial_per_[d],
                    d_X_occ_t_per_[d], d_X_vir_t_per_[d],
                    d_M_per_[d], d_F_per_[d],
                    d_Y_PQ_per_[d], d_ZY_per_[d], d_T_per_[d], d_MT_per_[d],
                    MultiGpuManager::instance().cublas(d),
                    enable_b3a3_, enable_b3_, enable_a3_,
                    d_Z_occ_per_[d], d_W_per_[d], d_U_per_[d],
                    d_yB3_per_[d], d_X2vir_per_[d], d_X2occ_per_[d],
                    d_tmp1_per_[d], d_tmpB3_per_[d], d_sig_corr_per_[d]);
            } catch (const std::exception& e) {
                err_msg[d] = e.what();
            }
#ifdef _OPENMP
        }
#else
        }
#endif
        for (int d = 0; d < num_gpus_; ++d) {
            if (!err_msg[d].empty())
                THROW_EXCEPTION("THCSOSADC2 partial sigma (GPU " + std::to_string(d) + "): " + err_msg[d]);
        }
#endif
    }

    // ---- Step 4: sum partials into d_output (on GPU 0) ----
#ifndef GANSU_CPU_ONLY
    {
        MultiGpuManager::DeviceGuard guard(0);
#endif
        cublasHandle_t cublas0 = gpu::GPUHandle::cublas();
        const real_t alpha = 1.0;
        // GPU 0's partial first
        check_cublas(cublasDaxpy(cublas0, ov,
                                  &alpha, d_sigma_partial_per_[0], 1,
                                  d_output, 1),
                     "axpy partial[0] -> output");
#ifndef GANSU_CPU_ONLY
        // Peer partials: copy to GPU 0's input buffer (reuse), axpy
        for (int d = 1; d < num_gpus_; ++d) {
            cudaMemcpy(d_input_per_[0], d_sigma_partial_per_[d],
                       ov * sizeof(real_t), cudaMemcpyDefault);
            check_cublas(cublasDaxpy(cublas0, ov,
                                      &alpha, d_input_per_[0], 1,
                                      d_output, 1),
                         "axpy partial[d] -> output");
        }
    }
#endif
}

void THCSOSADC2Operator::apply_preconditioner(const real_t* d_input, real_t* d_output) const
{
    // Jacobi preconditioner: out[ia] = in[ia] / (D1[ia] - omega), entirely on device.
    const int threads = 256;
    const int blocks  = (singles_dim_ + threads - 1) / threads;
    thc_adc2_precondition_kernel<<<blocks, threads>>>(
        d_input, d_D1_, omega_, d_output, singles_dim_);
    check_cuda(cudaGetLastError(), "preconditioner");
}

} // namespace gansu
