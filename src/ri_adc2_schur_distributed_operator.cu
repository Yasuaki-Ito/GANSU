/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ri_adc2_schur_distributed_operator.cu
 * @brief Multi-GPU distributed RI-ADC(2) Schur complement operator
 *
 * See ri_adc2_schur_distributed_operator.hpp for algorithm overview.
 */

#ifdef GANSU_MULTI_GPU

#include "ri_adc2_schur_distributed_operator.hpp"
#include "multi_gpu_manager.hpp"
#include "nccl_comm.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace gansu {

// ============================================================
//  Forward declarations of M11 kernels from ri_adc2_schur_operator.cu
//  (__global__ functions have external linkage in CUDA)
// ============================================================

__global__ void ri_adc2_build_cis_kernel(
    const double* __restrict__ ovov, const double* __restrict__ oovv,
    const double* __restrict__ eps, double* __restrict__ M11,
    int nocc, int nvir);

__global__ void ri_adc2_build_t2_kernel(
    const double* __restrict__ ovov, const double* __restrict__ eps,
    double* __restrict__ t2, int nocc, int nvir);

__global__ void ri_adc2_ovov_to_4d_kernel(
    const double* __restrict__ ovov_matrix, double* __restrict__ ovov_4d,
    int nocc, int nvir);

__global__ void ri_adc2_add_self_energy_kernel(
    const double* __restrict__ sigma_oo, const double* __restrict__ sigma_vv,
    double* __restrict__ M11, int nocc, int nvir);

__global__ void ri_m11_ISR_kernel(
    const real_t* __restrict__ d_t2, const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_ISR_corr, int nocc, int nvir);

__global__ void ri_m11_sigma_oo_kernel(
    const real_t* __restrict__ d_t2, const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_oo, int nocc, int nvir);

__global__ void ri_m11_sigma_vv_kernel(
    const real_t* __restrict__ d_t2, const real_t* __restrict__ d_eri_ovov,
    real_t* __restrict__ d_sigma_vv, int nocc, int nvir);

// ============================================================
//  Kernels (same as ri_adc2_schur_operator.cu, duplicated for
//  link independence under GANSU_MULTI_GPU guard)
// ============================================================

__global__ void dist_adc2_weight_all_kernel(
    const double* __restrict__ R_all,
    double* __restrict__ W_all,
    const double* __restrict__ eps_occ,
    const double* __restrict__ eps_vir,
    double omega_J, int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov * nvir) return;
    int ic = idx % ov;
    int D = idx / ov;
    int I = ic / nvir;
    int C = ic % nvir;
    double denom = omega_J + eps_occ[I] - eps_vir[C] - eps_vir[D];
    if (fabs(denom) < 1e-12) denom = (denom >= 0.0) ? 1e-12 : -1e-12;
    W_all[idx] = R_all[idx] / denom;
}

__global__ void dist_adc2_group_A_all_kernel(
    const double* __restrict__ eri_vvov,
    const double* __restrict__ W_all,
    double* __restrict__ sigma,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    if (idx >= ov) return;
    int I = idx / nvir;
    int E = idx % nvir;
    int vv = nvir * nvir;

    double coul = 0.0, xc = 0.0;
    for (int D = 0; D < nvir; D++) {
        for (int C = 0; C < nvir; C++) {
            double w = W_all[(I * nvir + C) + D * ov];
            coul += eri_vvov[E * nvir + C + D * vv] * w;
            xc   += eri_vvov[D * nvir + E + C * vv] * w;
        }
    }
    sigma[idx] += 2.0 * coul - xc;
}

__global__ void dist_adc2_group_B_all_kernel(
    const double* __restrict__ ooov1_all,
    const double* __restrict__ ooov2_all,
    const double* __restrict__ W_all,
    double* __restrict__ sigma,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ov = nocc * nvir;
    int oo = nocc * nocc;
    if (idx >= ov) return;
    int K = idx / nvir;
    int E = idx % nvir;

    double val = 0.0;
    for (int I = 0; I < nocc; I++) {
        for (int D = 0; D < nvir; D++) {
            double jk_id = ooov1_all[K + (size_t)(I * nvir + D) * nocc];
            double ik_jd = ooov2_all[(I * nocc + K) + (size_t)D * oo];
            double w = W_all[(I * nvir + E) + (size_t)D * ov];
            val += (jk_id - 2.0 * ik_jd) * w;
        }
    }
    sigma[idx] += val;
}

// ============================================================
//  Constructor / Destructor
// ============================================================

RIADC2SchurDistributedOperator::RIADC2SchurDistributedOperator(
    int num_gpus,
    const std::vector<real_t*>& d_B_ia_local,
    const std::vector<real_t*>& d_B_ab_local,
    const std::vector<real_t*>& d_B_ij_local,
    const std::vector<int>& naux_local,
    const real_t* d_M11,
    const real_t* d_orbital_energies,
    int nocc, int nvir)
    : num_gpus_(num_gpus), nocc_(nocc), nvir_(nvir),
      ov_(nocc * nvir), vv_(nvir * nvir), oo_(nocc * nocc),
      naux_local_(naux_local)
{
    // Store RI block pointers (not owned)
    d_B_ia_local_.resize(num_gpus);
    d_B_ab_local_.resize(num_gpus);
    d_B_ij_local_.resize(num_gpus);
    for (int d = 0; d < num_gpus; d++) {
        d_B_ia_local_[d] = d_B_ia_local[d];
        d_B_ab_local_[d] = d_B_ab_local[d];
        d_B_ij_local_[d] = d_B_ij_local[d];
    }

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

        tracked_cudaMalloc(&d_eps_occ_dev_, nocc * sizeof(real_t));
        tracked_cudaMalloc(&d_eps_vir_dev_, nvir * sizeof(real_t));
        cudaMemcpy(d_eps_occ_dev_, eps_occ_.data(), nocc * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps_vir_dev_, eps_vir_.data(), nvir * sizeof(double),
                   cudaMemcpyHostToDevice);

        tracked_cudaMalloc(&d_D1_, ov_ * sizeof(real_t));
        tracked_cudaMalloc(&d_diagonal_, ov_ * sizeof(real_t));
        tracked_cudaMalloc(&d_W_all_, (size_t)ov_ * nvir * sizeof(real_t));
    }

    compute_D1();
    compute_diagonal();
    allocate_workspace();

    std::cout << "[RI-ADC(2)-Schur-Distributed] Initialized: nocc=" << nocc
              << " nvir=" << nvir << " num_gpus=" << num_gpus << " naux_local=[";
    for (int d = 0; d < num_gpus; d++) {
        if (d) std::cout << ",";
        std::cout << naux_local[d];
    }
    std::cout << "]" << std::endl;
}

RIADC2SchurDistributedOperator::~RIADC2SchurDistributedOperator() {
    free_workspace();
    MultiGpuManager::DeviceGuard guard(0);
    tracked_cudaFree(d_M11_);
    tracked_cudaFree(d_D1_);
    tracked_cudaFree(d_diagonal_);
    tracked_cudaFree(d_W_all_);
    tracked_cudaFree(d_eps_occ_dev_);
    tracked_cudaFree(d_eps_vir_dev_);
}

// ============================================================
//  Workspace management
// ============================================================

void RIADC2SchurDistributedOperator::allocate_workspace() {
    ws_.resize(num_gpus_);
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& w = ws_[d];
        int nl = naux_local_[d];

        cudaMalloc(&w.d_x, ov_ * sizeof(real_t));
        cudaMalloc(&w.d_phi, (size_t)nvir_ * nocc_ * nl * sizeof(real_t));
        cudaMalloc(&w.d_psi, (size_t)nvir_ * nocc_ * nl * sizeof(real_t));
        cudaMalloc(&w.d_alpha, (size_t)ov_ * nl * sizeof(real_t));
        cudaMalloc(&w.d_beta, (size_t)nvir_ * nl * sizeof(real_t));

        // Buffers for partial → AllReduced intermediates
        cudaMalloc(&w.d_eri_vvov, (size_t)vv_ * nvir_ * sizeof(real_t));
        cudaMalloc(&w.d_R_all, (size_t)ov_ * nvir_ * sizeof(real_t));
        cudaMalloc(&w.d_ooov1, (size_t)nocc_ * ov_ * sizeof(real_t));
        cudaMalloc(&w.d_ooov2, (size_t)oo_ * nvir_ * sizeof(real_t));
    }
}

void RIADC2SchurDistributedOperator::free_workspace() {
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        auto& w = ws_[d];
        cudaFree(w.d_x);
        cudaFree(w.d_phi);
        cudaFree(w.d_psi);
        cudaFree(w.d_alpha);
        cudaFree(w.d_beta);
        cudaFree(w.d_eri_vvov);
        cudaFree(w.d_R_all);
        cudaFree(w.d_ooov1);
        cudaFree(w.d_ooov2);
        w = {};
    }
    ws_.clear();
}

// ============================================================
//  D1, diagonal, preconditioner
// ============================================================

void RIADC2SchurDistributedOperator::compute_D1() {
    std::vector<double> D1(ov_);
    for (int i = 0; i < nocc_; i++)
        for (int a = 0; a < nvir_; a++)
            D1[i * nvir_ + a] = eps_vir_[a] - eps_occ_[i];
    cudaMemcpy(d_D1_, D1.data(), ov_ * sizeof(double), cudaMemcpyHostToDevice);
}

void RIADC2SchurDistributedOperator::compute_diagonal() {
    cudaMemcpy(d_diagonal_, d_D1_, ov_ * sizeof(double), cudaMemcpyDeviceToDevice);
}

void RIADC2SchurDistributedOperator::apply_preconditioner(
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

// ============================================================
//  apply() — distributed sigma build
// ============================================================

void RIADC2SchurDistributedOperator::apply(
    const real_t* d_input, real_t* d_output) const
{
    auto& mgr = MultiGpuManager::instance();
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const int threads = 256;

    // -----------------------------------------------------------
    //  Step 0: σ = M11 · x   (GPU 0)
    // -----------------------------------------------------------
    {
        MultiGpuManager::DeviceGuard guard(0);
        cublasHandle_t h0 = mgr.cublas(0);
        cublasDgemv(h0, CUBLAS_OP_N, ov_, ov_,
                    &one, d_M11_, ov_, d_input, 1, &zero, d_output, 1);
    }

    // -----------------------------------------------------------
    //  Broadcast x to all GPUs
    // -----------------------------------------------------------
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        if (d == 0) {
            cudaMemcpy(ws_[0].d_x, d_input, ov_ * sizeof(real_t),
                       cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyPeer(ws_[d].d_x, d, d_input, 0, ov_ * sizeof(real_t));
        }
    }

    // -----------------------------------------------------------
    //  Steps 1-2: phi, psi, alpha on each GPU (local P slices)
    // -----------------------------------------------------------
    for (int d = 0; d < num_gpus_; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t h = mgr.cublas(d);
        int nl = naux_local_[d];
        auto& w = ws_[d];

        // phi^P[C,I] = Σ_F B_ab^P[C,F] × x[F,I]  (naux_local batches)
        cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
            nvir_, nocc_, nvir_,
            &one,
            d_B_ab_local_[d], nvir_, (long long)vv_,
            w.d_x, nvir_, 0LL,
            &zero,
            w.d_phi, nvir_, (long long)(nvir_ * nocc_),
            nl);

        // psi^P[C,I] = Σ_L x[C,L] × B_ij^P[L,I]^T  (naux_local batches)
        cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T,
            nvir_, nocc_, nocc_,
            &one,
            w.d_x, nvir_, 0LL,
            d_B_ij_local_[d], nocc_, (long long)oo_,
            &zero,
            w.d_psi, nvir_, (long long)(nvir_ * nocc_),
            nl);

        // alpha = phi - psi  [ov × naux_local]
        {
            int total = ov_ * nl;
            cublasDgeam(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        total, 1,
                        &one, w.d_phi, total,
                        &neg_one, w.d_psi, total,
                        w.d_alpha, total);
        }
    }

    // -----------------------------------------------------------
    //  J loop
    // -----------------------------------------------------------
    const int lda_phi = nvir_ * nocc_;

    for (int J = 0; J < nocc_; J++) {

        // ---- Phase 1: partial DGEMMs on all GPUs ----
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cublasHandle_t h = mgr.cublas(d);
            int nl = naux_local_[d];
            auto& w = ws_[d];

            // b_J on this GPU: row J of B_ia_local, stride = ov_
            const real_t* b_J = &d_B_ia_local_[d][J * nvir_];

            // eri_vvov = B_ab_local × b_J^T  [vv × nvir]  (partial over P)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        vv_, nvir_, nl,
                        &one, d_B_ab_local_[d], vv_,
                        b_J, ov_,
                        &zero, w.d_eri_vvov, vv_);

            // beta_J = phi_J - psi_J  [nvir × naux_local]
            cublasDgeam(h, CUBLAS_OP_N, CUBLAS_OP_N,
                        nvir_, nl,
                        &one,     &w.d_phi[J * nvir_], lda_phi,
                        &neg_one, &w.d_psi[J * nvir_], lda_phi,
                        w.d_beta, nvir_);

            // R = alpha × b_J^T  [ov × nvir]  (partial)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        ov_, nvir_, nl,
                        &one, w.d_alpha, ov_,
                        b_J, ov_,
                        &zero, w.d_R_all, ov_);

            // R += B_ia_local × beta_J^T  [ov × nvir]  (partial)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        ov_, nvir_, nl,
                        &one, d_B_ia_local_[d], ov_,
                        w.d_beta, nvir_,
                        &one, w.d_R_all, ov_);
        }

        // ---- AllReduce eri_vvov and R_all ----
        mgr.sync_all();
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            nccl::all_reduce(ws_[d].d_eri_vvov, ws_[d].d_eri_vvov,
                             (size_t)vv_ * nvir_, ncclSum, d, mgr.comm_stream(d));
            nccl::all_reduce(ws_[d].d_R_all, ws_[d].d_R_all,
                             (size_t)ov_ * nvir_, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();
        mgr.sync_all();

        // ---- Weight + Group A on GPU 0 ----
        {
            MultiGpuManager::DeviceGuard guard(0);
            double omega_J = omega_ + eps_occ_[J];
            int total = ov_ * nvir_;
            int blk = (total + threads - 1) / threads;
            dist_adc2_weight_all_kernel<<<blk, threads>>>(
                ws_[0].d_R_all, d_W_all_,
                d_eps_occ_dev_, d_eps_vir_dev_,
                omega_J, nocc_, nvir_);

            blk = (ov_ + threads - 1) / threads;
            dist_adc2_group_A_all_kernel<<<blk, threads>>>(
                ws_[0].d_eri_vvov, d_W_all_, d_output, nocc_, nvir_);
        }

        // ---- Phase 2: OOOV partial DGEMMs on all GPUs ----
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            cublasHandle_t h = mgr.cublas(d);
            int nl = naux_local_[d];
            auto& w = ws_[d];

            const real_t* b_J = &d_B_ia_local_[d][J * nvir_];
            const real_t* b_ij_J = &d_B_ij_local_[d][J * nocc_];

            // ooov1[K, I*nvir+D] = b_ij_J × B_ia^T  [nocc × ov]  (partial)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        nocc_, ov_, nl,
                        &one, b_ij_J, oo_,
                        d_B_ia_local_[d], ov_,
                        &zero, w.d_ooov1, nocc_);

            // ooov2[I*nocc+K, D] = B_ij × b_J^T  [oo × nvir]  (partial)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                        oo_, nvir_, nl,
                        &one, d_B_ij_local_[d], oo_,
                        b_J, ov_,
                        &zero, w.d_ooov2, oo_);
        }

        // ---- AllReduce ooov1 and ooov2 ----
        mgr.sync_all();
        nccl::group_start();
        for (int d = 0; d < num_gpus_; d++) {
            MultiGpuManager::DeviceGuard guard(d);
            nccl::all_reduce(ws_[d].d_ooov1, ws_[d].d_ooov1,
                             (size_t)nocc_ * ov_, ncclSum, d, mgr.comm_stream(d));
            nccl::all_reduce(ws_[d].d_ooov2, ws_[d].d_ooov2,
                             (size_t)oo_ * nvir_, ncclSum, d, mgr.comm_stream(d));
        }
        nccl::group_end();
        mgr.sync_all();

        // ---- Group B on GPU 0 ----
        {
            MultiGpuManager::DeviceGuard guard(0);
            int blk = (ov_ + threads - 1) / threads;
            dist_adc2_group_B_all_kernel<<<blk, threads>>>(
                ws_[0].d_ooov1, ws_[0].d_ooov2, d_W_all_,
                d_output, nocc_, nvir_);
        }

    } // J loop
}

// ============================================================
//  build_M11_distributed — distributed OVOV/OOVV + GPU 0 kernels
// ============================================================

void RIADC2SchurDistributedOperator::build_M11_distributed(
    real_t* d_M11_out,
    int num_gpus,
    const std::vector<real_t*>& d_B_ia_local,
    const std::vector<real_t*>& d_B_ab_local,
    const std::vector<real_t*>& d_B_ij_local,
    const std::vector<int>& naux_local,
    const real_t* d_orbital_energies,
    int nocc, int nvir)
{
    auto& mgr = MultiGpuManager::instance();
    const int ov = nocc * nvir;
    const int vv = nvir * nvir;
    const int oo = nocc * nocc;
    const int dd = nocc * nocc * nvir * nvir;
    const size_t matrix_size = (size_t)ov * ov;
    const double one = 1.0, zero = 0.0;
    const int threads = 256;

    // --- Per-GPU workspace for partial OVOV and OOVV ---
    std::vector<real_t*> d_ovov(num_gpus, nullptr);
    std::vector<real_t*> d_oovv(num_gpus, nullptr);

    for (int d = 0; d < num_gpus; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cudaMalloc(&d_ovov[d], matrix_size * sizeof(real_t));
        cudaMalloc(&d_oovv[d], (size_t)oo * vv * sizeof(real_t));
    }

    // --- Step 1-2: Partial DGEMM on each GPU ---
    for (int d = 0; d < num_gpus; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        cublasHandle_t h = mgr.cublas(d);
        int nl = naux_local[d];

        // OVOV_partial = B_ia_local × B_ia_local^T  [ov × ov], K=naux_local
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                    ov, ov, nl,
                    &one, d_B_ia_local[d], ov,
                    d_B_ia_local[d], ov,
                    &zero, d_ovov[d], ov);

        // OOVV_partial = B_ij_local × B_ab_local^T  [oo × vv], K=naux_local
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T,
                    oo, vv, nl,
                    &one, d_B_ij_local[d], oo,
                    d_B_ab_local[d], vv,
                    &zero, d_oovv[d], oo);
    }

    // --- AllReduce OVOV and OOVV ---
    mgr.sync_all();
    nccl::group_start();
    for (int d = 0; d < num_gpus; d++) {
        MultiGpuManager::DeviceGuard guard(d);
        nccl::all_reduce(d_ovov[d], d_ovov[d], matrix_size,
                         ncclSum, d, mgr.comm_stream(d));
        nccl::all_reduce(d_oovv[d], d_oovv[d], (size_t)oo * vv,
                         ncclSum, d, mgr.comm_stream(d));
    }
    nccl::group_end();
    mgr.sync_all();

    // --- Steps 3-9: kernel work on GPU 0 (identical to single-GPU) ---
    {
        MultiGpuManager::DeviceGuard guard(0);
        cublasHandle_t h0 = mgr.cublas(0);

        // Step 3: CIS matrix
        {
            int blocks = ((int)matrix_size + threads - 1) / threads;
            ri_adc2_build_cis_kernel<<<blocks, threads>>>(
                d_ovov[0], d_oovv[0], d_orbital_energies, d_M11_out, nocc, nvir);
        }

        // Free OOVV (no longer needed); free non-GPU-0 OVOV
        for (int d = 1; d < num_gpus; d++) {
            MultiGpuManager::DeviceGuard g(d);
            cudaFree(d_oovv[d]);
            cudaFree(d_ovov[d]);
        }
        { // GPU 0
            cudaFree(d_oovv[0]);
        }

        // Step 4: Convert OVOV [ov×ov] → 4D layout
        real_t* d_ovov_4d = nullptr;
        tracked_cudaMalloc(&d_ovov_4d, matrix_size * sizeof(real_t));
        {
            int blocks = ((int)matrix_size + threads - 1) / threads;
            ri_adc2_ovov_to_4d_kernel<<<blocks, threads>>>(
                d_ovov[0], d_ovov_4d, nocc, nvir);
        }

        // Step 5: Build T2 (reuse AllReduced OVOV — no second DGEMM needed)
        real_t* d_t2 = nullptr;
        tracked_cudaMalloc(&d_t2, (size_t)dd * sizeof(real_t));
        {
            int blocks = (dd + threads - 1) / threads;
            ri_adc2_build_t2_kernel<<<blocks, threads>>>(
                d_ovov[0], d_orbital_energies, d_t2, nocc, nvir);
        }
        cudaFree(d_ovov[0]);  // OVOV no longer needed

        // Step 6: ISR correction
        real_t* d_ISR_corr = nullptr;
        tracked_cudaMalloc(&d_ISR_corr, matrix_size * sizeof(real_t));
        {
            int blocks = ((int)matrix_size + threads - 1) / threads;
            ri_m11_ISR_kernel<<<blocks, threads>>>(
                d_t2, d_ovov_4d, d_ISR_corr, nocc, nvir);
            cudaDeviceSynchronize();
        }
        cublasDaxpy(h0, (int)matrix_size, &one, d_ISR_corr, 1, d_M11_out, 1);
        tracked_cudaFree(d_ISR_corr);

        // Step 7: Σ_oo
        real_t* d_sigma_oo = nullptr;
        tracked_cudaMalloc(&d_sigma_oo, (size_t)oo * sizeof(real_t));
        {
            int blocks = (oo + threads - 1) / threads;
            ri_m11_sigma_oo_kernel<<<blocks, threads>>>(
                d_t2, d_ovov_4d, d_sigma_oo, nocc, nvir);
            cudaDeviceSynchronize();
        }

        // Step 8: Σ_vv
        real_t* d_sigma_vv = nullptr;
        tracked_cudaMalloc(&d_sigma_vv, (size_t)vv * sizeof(real_t));
        {
            int blocks = (vv + threads - 1) / threads;
            ri_m11_sigma_vv_kernel<<<blocks, threads>>>(
                d_t2, d_ovov_4d, d_sigma_vv, nocc, nvir);
            cudaDeviceSynchronize();
        }

        // Step 9: Add Σ corrections
        {
            int blocks = ((int)matrix_size + threads - 1) / threads;
            ri_adc2_add_self_energy_kernel<<<blocks, threads>>>(
                d_sigma_oo, d_sigma_vv, d_M11_out, nocc, nvir);
        }

        tracked_cudaFree(d_sigma_oo);
        tracked_cudaFree(d_sigma_vv);
        tracked_cudaFree(d_t2);
        tracked_cudaFree(d_ovov_4d);
    }

    std::cout << "[RI-ADC(2)] M11 built from distributed RI ("
              << num_gpus << " GPUs)" << std::endl;
}

} // namespace gansu

#endif // GANSU_MULTI_GPU
