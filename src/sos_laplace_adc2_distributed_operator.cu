/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file sos_laplace_adc2_distributed_operator.cu
 * @brief Multi-GPU SOS-Laplace-ADC(2) — Laplace-point-parallel sigma
 *
 * Each GPU holds the full B_ia (AllGathered from local pieces).
 * The Laplace τ loop is partitioned across GPUs: each GPU
 * computes sigma contributions for its assigned τ points,
 * then AllReduce(sigma_schur) [ov doubles].
 */

#ifdef GANSU_MULTI_GPU

#include "sos_laplace_adc2_distributed_operator.hpp"
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include "laplace_quadrature.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

// ============================================================
//  Kernels (same as sos_laplace_adc2_operator.cu)
// ============================================================

__global__ void dist_sos_scale_B_kernel(
    const real_t* __restrict__ d_B,
    real_t* __restrict__ d_C,
    const real_t* __restrict__ d_eps_occ,
    const real_t* __restrict__ d_eps_vir,
    int nocc, int nvir, int naux, double t_half)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ov = (size_t)nocc * nvir;
    if (idx >= ov * naux) return;
    size_t P = idx / ov;
    size_t ia = idx % ov;
    int i = (int)(ia / nvir);
    int a = (int)(ia % nvir);
    double scale = exp(-t_half * (d_eps_vir[a] - d_eps_occ[i]));
    d_C[idx] = d_B[idx] * scale;
}

__global__ void dist_sos_scale_by_trial_kernel(
    const real_t* __restrict__ d_B_scaled,
    const real_t* __restrict__ d_x,
    real_t* __restrict__ d_F,
    int nov, int naux)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)nov * naux) return;
    size_t ia = idx % nov;
    d_F[idx] = d_B_scaled[idx] * d_x[ia];
}

__global__ void dist_sos_rowdot_kernel(
    const real_t* __restrict__ d_A,
    const real_t* __restrict__ d_B,
    real_t* __restrict__ d_sigma,
    double scale_factor,
    int nov, int naux)
{
    int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= nov) return;
    double sum = 0.0;
    for (int P = 0; P < naux; P++)
        sum += d_A[(size_t)P * nov + ia] * d_B[(size_t)P * nov + ia];
    d_sigma[ia] += scale_factor * sum;
}

// ============================================================
//  Constructor / Destructor
// ============================================================

SOSLaplaceADC2DistributedOperator::SOSLaplaceADC2DistributedOperator(
    int num_gpus,
    const std::vector<real_t*>& d_B_ia_local,
    const std::vector<int>& naux_local,
    const real_t* d_M11,
    const real_t* d_orbital_energies,
    int nocc, int nvir, int naux_total,
    double c_os, int n_laplace)
    : num_gpus_(num_gpus), nocc_(nocc), nvir_(nvir), naux_(naux_total),
      ov_(nocc * nvir), c_os_(c_os), n_laplace_(n_laplace)
{
    // Read orbital energies from GPU 0
    std::vector<double> eps(nocc + nvir);
    cudaMemcpy(eps.data(), d_orbital_energies, (nocc + nvir) * sizeof(double),
               cudaMemcpyDeviceToHost);
    eps_occ_.assign(eps.begin(), eps.begin() + nocc);
    eps_vir_.assign(eps.begin() + nocc, eps.end());

    // GPU 0 allocations
    {
        MultiGpuManager::DeviceGuard guard(0);
        tracked_cudaMalloc(&d_M11_, (size_t)ov_ * ov_ * sizeof(real_t));
        cudaMemcpy(d_M11_, d_M11, (size_t)ov_ * ov_ * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        tracked_cudaMalloc(&d_D1_, ov_ * sizeof(real_t));
        tracked_cudaMalloc(&d_diagonal_, ov_ * sizeof(real_t));
    }
    compute_D1();
    compute_diagonal();

    allocate_workspace();
    allgather_B_ia(d_B_ia_local, naux_local);
    update_laplace_quadrature();

    std::cout << "[SOS-LT-ADC(2)-Distributed] Initialized: nocc=" << nocc
              << " nvir=" << nvir << " naux=" << naux_total
              << " num_gpus=" << num_gpus << " n_laplace=" << n_laplace
              << " c_os=" << c_os << std::endl;
}

SOSLaplaceADC2DistributedOperator::~SOSLaplaceADC2DistributedOperator() {
    free_workspace();
    MultiGpuManager::DeviceGuard guard(0);
    tracked_cudaFree(d_M11_);
    tracked_cudaFree(d_D1_);
    tracked_cudaFree(d_diagonal_);
}

// ============================================================
//  Workspace
// ============================================================

void SOSLaplaceADC2DistributedOperator::allocate_workspace() {
    ws_.resize(num_gpus_);
    size_t ov_aux = (size_t)ov_ * naux_;
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& w = ws_[d];
        cudaMalloc(&w.d_B_ia_full, ov_aux * sizeof(real_t));
        cudaMalloc(&w.d_x, ov_ * sizeof(real_t));
        cudaMalloc(&w.d_B_scaled, ov_aux * sizeof(real_t));
        cudaMalloc(&w.d_F, ov_aux * sizeof(real_t));
        cudaMalloc(&w.d_X_PQ, (size_t)naux_ * naux_ * sizeof(real_t));
        cudaMalloc(&w.d_temp, ov_aux * sizeof(real_t));
        cudaMalloc(&w.d_sigma_local, ov_ * sizeof(real_t));
        cudaMalloc(&w.d_eps_occ, nocc_ * sizeof(real_t));
        cudaMalloc(&w.d_eps_vir, nvir_ * sizeof(real_t));
        cudaMemcpy(w.d_eps_occ, eps_occ_.data(), nocc_ * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(w.d_eps_vir, eps_vir_.data(), nvir_ * sizeof(double),
                   cudaMemcpyHostToDevice);
    }
}

void SOSLaplaceADC2DistributedOperator::free_workspace() {
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& w = ws_[d];
        cudaFree(w.d_B_ia_full);
        cudaFree(w.d_x);
        cudaFree(w.d_B_scaled);
        cudaFree(w.d_F);
        cudaFree(w.d_X_PQ);
        cudaFree(w.d_temp);
        cudaFree(w.d_sigma_local);
        cudaFree(w.d_eps_occ);
        cudaFree(w.d_eps_vir);
        w = {};
    }
    ws_.clear();
}

// ============================================================
//  AllGather B_ia_local → B_ia_full on each GPU
// ============================================================

void SOSLaplaceADC2DistributedOperator::allgather_B_ia(
    const std::vector<real_t*>& d_B_ia_local,
    const std::vector<int>& naux_local)
{
    // Compute P_start for each GPU
    std::vector<size_t> P_start(num_gpus_ + 1, 0);
    for (int d = 0; d < num_gpus_; d++)
        P_start[d + 1] = P_start[d] + naux_local[d];

    // Each GPU receives all pieces
    for (int d_dst = 0; d_dst < num_gpus_; d_dst++) {
        MultiGpuManager::DeviceGuard guard(d_dst);
        for (int d_src = 0; d_src < num_gpus_; d_src++) {
            size_t offset = P_start[d_src] * ov_;  // col-major offset
            size_t bytes = (size_t)naux_local[d_src] * ov_ * sizeof(real_t);
            if (d_src == d_dst) {
                cudaMemcpy(ws_[d_dst].d_B_ia_full + offset,
                           d_B_ia_local[d_src], bytes,
                           cudaMemcpyDeviceToDevice);
            } else {
                cudaMemcpyPeer(ws_[d_dst].d_B_ia_full + offset, d_dst,
                               d_B_ia_local[d_src], d_src, bytes);
            }
        }
    }

    auto& mgr = MultiGpuManager::instance();
    mgr.sync_all();
    std::cout << "[SOS-LT-ADC(2)-Distributed] AllGather B_ia complete ("
              << (double)ov_ * naux_ * sizeof(real_t) / (1024.0 * 1024.0)
              << " MB/GPU)" << std::endl;
}

// ============================================================
//  D1, diagonal, preconditioner, Laplace quadrature
// ============================================================

void SOSLaplaceADC2DistributedOperator::compute_D1() {
    std::vector<double> D1(ov_);
    for (int i = 0; i < nocc_; i++)
        for (int a = 0; a < nvir_; a++)
            D1[i * nvir_ + a] = eps_vir_[a] - eps_occ_[i];
    cudaMemcpy(d_D1_, D1.data(), ov_ * sizeof(double), cudaMemcpyHostToDevice);
}

void SOSLaplaceADC2DistributedOperator::compute_diagonal() {
    cudaMemcpy(d_diagonal_, d_D1_, ov_ * sizeof(double), cudaMemcpyDeviceToDevice);
}

void SOSLaplaceADC2DistributedOperator::apply_preconditioner(
    const real_t* d_input, real_t* d_output) const
{
    std::vector<double> D1(ov_), input(ov_);
    cudaMemcpy(D1.data(), d_D1_, ov_ * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(input.data(), d_input, ov_ * sizeof(double), cudaMemcpyDeviceToHost);
    for (int ia = 0; ia < ov_; ia++) {
        double denom = D1[ia] - omega_;
        if (std::abs(denom) < 1e-10) denom = (denom >= 0) ? 1e-10 : -1e-10;
        input[ia] /= denom;
    }
    cudaMemcpy(d_output, input.data(), ov_ * sizeof(double), cudaMemcpyHostToDevice);
}

void SOSLaplaceADC2DistributedOperator::set_omega(real_t omega) {
    omega_ = omega;
}

void SOSLaplaceADC2DistributedOperator::update_laplace_quadrature() {
    double Delta_min = 2.0 * eps_vir_[0] - 2.0 * eps_occ_[nocc_ - 1];
    double Delta_max = 2.0 * eps_vir_[nvir_ - 1] - 2.0 * eps_occ_[0];
    double x_min = Delta_min - omega_;
    double x_max = Delta_max - omega_;
    if (x_min < 1e-4) {
        std::cerr << "[SOS-LT-ADC(2)-Dist] Warning: omega=" << omega_
                  << " near threshold Delta_min=" << Delta_min << std::endl;
        x_min = 1e-4;
    }
    if (x_max < x_min + 0.1) x_max = x_min + 10.0;
    auto quad = generate_laplace_quadrature(x_min, x_max, n_laplace_);
    laplace_t_.resize(quad.num_points);
    laplace_w_.resize(quad.num_points);
    for (int k = 0; k < quad.num_points; k++) {
        laplace_t_[k] = quad.points[k];
        laplace_w_[k] = quad.weights[k];
    }
}

// ============================================================
//  apply() — Laplace-point-parallel sigma build
// ============================================================

void SOSLaplaceADC2DistributedOperator::apply(
    const real_t* d_input, real_t* d_output) const
{
    auto& mgr = MultiGpuManager::instance();
    const double one = 1.0, zero = 0.0;
    const int threads = 256;
    const int n_tau = (int)laplace_t_.size();

    // Step 1: σ = M11 · x  (GPU 0)
    {
        MultiGpuManager::DeviceGuard guard(0);
        cublasHandle_t h0 = mgr.cublas(0);
        cublasDgemv(h0, CUBLAS_OP_N, ov_, ov_,
                    &one, d_M11_, ov_, d_input, 1, &zero, d_output, 1);
    }

    // Broadcast x + zero sigma_local on each GPU
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d == 0)
            cudaMemcpy(ws_[0].d_x, d_input, ov_ * sizeof(real_t),
                       cudaMemcpyDeviceToDevice);
        else
            cudaMemcpyPeer(ws_[d].d_x, d, d_input, 0, ov_ * sizeof(real_t));
        cudaMemset(ws_[d].d_sigma_local, 0, ov_ * sizeof(real_t));
    }

    // Step 2: Distribute Laplace points across GPUs
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t h = mgr.cublas(d);
        auto& w = ws_[d];

        // This GPU handles τ points: k = d, d+num_gpus, d+2*num_gpus, ...
        for (int k = d; k < n_tau; k += num_gpus_) {
            double t = laplace_t_[k];
            double wt = laplace_w_[k];
            double t_half = t / 2.0;
            double prefactor = -c_os_ * wt * std::exp(omega_ * t);

            size_t total = (size_t)ov_ * naux_;

            // Scale B → B̃
            {
                int blk = (int)((total + threads - 1) / threads);
                dist_sos_scale_B_kernel<<<blk, threads>>>(
                    w.d_B_ia_full, w.d_B_scaled, w.d_eps_occ, w.d_eps_vir,
                    nocc_, nvir_, naux_, t_half);
            }

            // F = B̃ × x (element-wise)
            {
                int blk = (int)((total + threads - 1) / threads);
                dist_sos_scale_by_trial_kernel<<<blk, threads>>>(
                    w.d_B_scaled, w.d_x, w.d_F, ov_, naux_);
            }

            // X = F^T · B̃  [naux × naux]
            cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                        naux_, naux_, ov_,
                        &one, w.d_F, ov_,
                        w.d_B_scaled, ov_,
                        &zero, w.d_X_PQ, naux_);

            // temp = B̃ · X  [ov × naux]
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        ov_, naux_, naux_,
                        &one, w.d_B_scaled, ov_,
                        w.d_X_PQ, naux_,
                        &zero, w.d_temp, ov_);

            // σ_local += prefactor × rowdot(B̃, temp)
            {
                int blk = (ov_ + threads - 1) / threads;
                dist_sos_rowdot_kernel<<<blk, threads>>>(
                    w.d_B_scaled, w.d_temp, w.d_sigma_local,
                    prefactor, ov_, naux_);
            }
        }
    }

    // Step 3: AllReduce sigma_local → GPU 0
    mgr.sync_all();
    nccl::group_start();
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(ws_[d].d_sigma_local, ws_[d].d_sigma_local,
                         ov_, ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();
    mgr.sync_all();

    // Step 4: σ += sigma_schur (GPU 0)
    {
        MultiGpuManager::DeviceGuard guard(0);
        cublasHandle_t h0 = mgr.cublas(0);
        cublasDaxpy(h0, ov_, &one, ws_[0].d_sigma_local, 1, d_output, 1);
    }
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
