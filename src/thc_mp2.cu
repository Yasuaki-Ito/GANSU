/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "thc_mp2.hpp"
#include <memory>

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

#ifndef GANSU_CPU_ONLY
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "gpu_manager.hpp"
#include "laplace_quadrature.hpp"
#include "multi_gpu_manager.hpp"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

namespace gansu {

namespace {

using ColMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

inline Eigen::Map<const ColMatXd> map_const(const std::vector<real_t>& v,
                                             int rows, int cols)
{
    return Eigen::Map<const ColMatXd>(v.data(), rows, cols);
}

} // anonymous namespace

// X_ao col-major: X_ao[mu + P*N_bas] = X[mu, P]
// C    row-major (GANSU convention): C[mu*N_orb + p] = C_{mu p}
// X_mo col-major: X_mo[p + P*N_orb] = sum_mu C[mu, p] * X[mu, P]
std::vector<real_t> transform_X_to_mo_cpu(const std::vector<real_t>& X_ao,
                                          const std::vector<real_t>& C,
                                          int N_bas, int N_orb, int N_g)
{
    if ((int)X_ao.size() != N_bas * N_g)
        throw std::runtime_error("transform_X_to_mo_cpu: X_ao size mismatch");
    if ((int)C.size() != N_bas * N_orb)
        throw std::runtime_error("transform_X_to_mo_cpu: C size mismatch");

    std::vector<real_t> X_mo(static_cast<std::size_t>(N_orb) * N_g, 0.0);
    for (int P = 0; P < N_g; ++P) {
        for (int p = 0; p < N_orb; ++p) {
            real_t s = 0.0;
            for (int mu = 0; mu < N_bas; ++mu) {
                s += C[mu * N_orb + p] * X_ao[mu + P * N_bas];
            }
            X_mo[p + P * N_orb] = s;
        }
    }
    return X_mo;
}

real_t compute_mp2_energy_from_mo_eri_cpu(const std::vector<real_t>& eri_mo_4d,
                                          const std::vector<real_t>& eps,
                                          int n_occ, int N_orb)
{
    if ((int)eri_mo_4d.size() != N_orb * N_orb * N_orb * N_orb)
        throw std::runtime_error("compute_mp2_energy: eri size mismatch");
    if ((int)eps.size() != N_orb)
        throw std::runtime_error("compute_mp2_energy: eps size mismatch");
    if (n_occ <= 0 || n_occ >= N_orb)
        throw std::runtime_error("compute_mp2_energy: invalid n_occ");

    auto idx = [N_orb](int p, int q, int r, int s) {
        return p + N_orb * (q + N_orb * (r + N_orb * s));
    };

    real_t E = 0.0;
    for (int i = 0; i < n_occ; ++i) {
        for (int a = n_occ; a < N_orb; ++a) {
            for (int j = 0; j < n_occ; ++j) {
                for (int b = n_occ; b < N_orb; ++b) {
                    const real_t iajb = eri_mo_4d[idx(i, a, j, b)];
                    const real_t ibja = eri_mo_4d[idx(i, b, j, a)];
                    const real_t denom = eps[a] + eps[b] - eps[i] - eps[j];
                    E -= iajb * (real_t(2.0) * iajb - ibja) / denom;
                }
            }
        }
    }
    return E;
}

#ifndef GANSU_CPU_ONLY

namespace {

inline void check_cuda_mp2(cudaError_t e, const char* tag)
{
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error: ") + tag
                                 + ": " + cudaGetErrorString(e));
    }
}

inline void check_cublas_mp2(cublasStatus_t s, const char* tag)
{
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS error: ") + tag);
    }
}

// Each thread accumulates one MP2 amplitude contribution and reduces with
// atomicAdd into d_E[0].  4-loop indexing matches the CPU reference:
//   eri[idx(i,a,j,b)] = i + N*(a + N*(j + N*b))
__global__ void mp2_reduction_kernel(
    const real_t* __restrict__ d_eri_mo, // length N_orb^4
    const real_t* __restrict__ d_eps,    // length N_orb
    int n_occ, int n_vir, int N_orb,
    real_t* d_E_partial)
{
    const std::size_t total =
        static_cast<std::size_t>(n_occ) * n_vir * n_occ * n_vir;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                          + threadIdx.x;
    if (idx >= total) return;

    // Decompose: idx = ((i * n_vir + a_off) * n_occ + j) * n_vir + b_off
    const int b_off = static_cast<int>(idx % n_vir);
    const int j     = static_cast<int>((idx / n_vir) % n_occ);
    const int a_off = static_cast<int>((idx / (n_vir * n_occ)) % n_vir);
    const int i     = static_cast<int>(idx / (static_cast<std::size_t>(n_vir) * n_occ * n_vir));

    const int a = n_occ + a_off;
    const int b = n_occ + b_off;

    const std::size_t N1 = N_orb;
    const std::size_t N2 = N1 * N_orb;
    const std::size_t N3 = N2 * N_orb;
    const std::size_t idx_iajb = i + N1*a + N2*j + N3*b;
    const std::size_t idx_ibja = i + N1*b + N2*j + N3*a;

    const real_t iajb = d_eri_mo[idx_iajb];
    const real_t ibja = d_eri_mo[idx_ibja];
    const real_t denom = d_eps[a] + d_eps[b] - d_eps[i] - d_eps[j];
    const real_t contrib = -iajb * (real_t(2.0) * iajb - ibja) / denom;

    atomicAdd(d_E_partial, contrib);
}

} // anonymous namespace

std::unique_ptr<DeviceHostMatrix<real_t>>
transform_X_to_mo_gpu(const real_t* d_X_ao, const real_t* d_C,
                       int N_bas, int N_orb, int N_g)
{
    cublasHandle_t cublas = gpu::GPUHandle::cublas();
    auto X_mo = std::make_unique<DeviceHostMatrix<real_t>>(N_orb, N_g);

    // C is stored row-major (GANSU convention) as C[mu * N_orb + p].
    // In column-major view of the same memory: C_view[p, mu] = C[mu, p],
    // i.e. C_view = C^T (shape N_orb x N_bas col-major).
    //
    // X_mo[p, P] = sum_mu C[mu, p] * X_ao[mu, P]
    //            = sum_mu C_view[p, mu] * X_ao[mu, P]
    //            = (C_view * X_ao)[p, P]
    // So a single DGEMM with op_N op_N suffices, treating d_C with leading
    // dimension N_orb (col-major rows count).
    const real_t one = 1.0;
    const real_t zero = 0.0;
    check_cublas_mp2(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N_orb, N_g, N_bas,
                                  &one, d_C, N_orb, d_X_ao, N_bas,
                                  &zero, X_mo->device_ptr(), N_orb),
                      "transform_X_to_mo_gpu");
    return X_mo;
}

real_t compute_mp2_energy_from_mo_eri_gpu(const real_t* d_eri_mo_4d,
                                           const real_t* d_eps,
                                           int n_occ, int N_orb)
{
    if (n_occ <= 0 || n_occ >= N_orb)
        throw std::runtime_error("compute_mp2_energy_gpu: invalid n_occ");

    const int n_vir = N_orb - n_occ;
    const std::size_t total =
        static_cast<std::size_t>(n_occ) * n_vir * n_occ * n_vir;

    real_t* d_E = nullptr;
    check_cuda_mp2(cudaMalloc(&d_E, sizeof(real_t)), "alloc d_E");
    check_cuda_mp2(cudaMemset(d_E, 0, sizeof(real_t)), "zero d_E");

    const int threads = 256;
    const int blocks  = static_cast<int>((total + threads - 1) / threads);
    mp2_reduction_kernel<<<blocks, threads>>>(
        d_eri_mo_4d, d_eps, n_occ, n_vir, N_orb, d_E);
    check_cuda_mp2(cudaGetLastError(), "mp2_reduction_kernel");

    real_t E = 0.0;
    check_cuda_mp2(cudaMemcpy(&E, d_E, sizeof(real_t), cudaMemcpyDeviceToHost),
                    "copy d_E -> host");
    cudaFree(d_E);
    return E;
}

// =============================================================================
//   THC-SOS-MP2 + Laplace (Phase 2.1)
// =============================================================================

namespace {

// Build scaled occupied and virtual collocation matrices for one Laplace τ:
//   X_occ_t[i, P] = X_mo[i, P] * exp(+ eps[i] * tau / 2)
//   X_vir_t[a, P] = X_mo[n_occ + a, P] * exp(-eps[n_occ + a] * tau / 2)
// (the /2 distributes the symmetric exponent across the two occurrences of
//  each index in the (ia|jb)^2 product; see theory_THC.md §8.2).
// X_mo is [N_orb x N_g] col-major.
__global__ void scale_X_for_laplace_kernel(
    const real_t* __restrict__ d_X_mo,
    const real_t* __restrict__ d_eps,
    real_t tau,
    int n_occ, int n_vir, int N_orb, int N_g,
    real_t* d_X_occ_t,    // [n_occ x N_g] col-major
    real_t* d_X_vir_t)    // [n_vir x N_g] col-major
{
    const int P = blockIdx.x * blockDim.x + threadIdx.x;
    const int p = blockIdx.y * blockDim.y + threadIdx.y;
    if (P >= N_g || p >= N_orb) return;

    const real_t x = d_X_mo[p + P * N_orb];
    const real_t half_tau = tau * real_t(0.5);
    if (p < n_occ) {
        const real_t scale = exp(d_eps[p] * half_tau);
        d_X_occ_t[p + P * n_occ] = x * scale;
    } else {
        const int a_off = p - n_occ;
        const real_t scale = exp(-d_eps[p] * half_tau);
        d_X_vir_t[a_off + P * n_vir] = x * scale;
    }
}

// C[i] = A[i] * B[i] elementwise.
__global__ void hadamard_kernel(
    const real_t* __restrict__ A,
    const real_t* __restrict__ B,
    real_t* C, std::size_t total)
{
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                          + threadIdx.x;
    if (idx >= total) return;
    C[idx] = A[idx] * B[idx];
}

} // anonymous namespace

// Per-device worker: processes a subset of Laplace τ points on the current
// device.  Returns local sum sum_{tau in [tau_start, tau_end)} w_tau * E_tau.
// The current device must be set by the caller (e.g. via DeviceGuard).
static real_t thc_sos_mp2_partial_on_device(
    const real_t* d_X_mo, const real_t* d_Z, const real_t* d_eps,
    int n_occ, int n_vir, int N_orb, int N_g,
    int tau_start, int tau_end,
    const LaplaceQuadrature& laplace,
    cublasHandle_t cublas)
{
    const real_t one = 1.0;
    const real_t zero = 0.0;
    const std::size_t Ng2 = static_cast<std::size_t>(N_g) * N_g;

    real_t* d_X_occ_t = nullptr;
    real_t* d_X_vir_t = nullptr;
    real_t* d_O = nullptr;
    real_t* d_V = nullptr;
    real_t* d_U = nullptr;
    real_t* d_ZU = nullptr;
    real_t* d_T = nullptr;
    check_cuda_mp2(cudaMalloc(&d_X_occ_t, static_cast<std::size_t>(n_occ) * N_g * sizeof(real_t)), "alloc X_occ_t");
    check_cuda_mp2(cudaMalloc(&d_X_vir_t, static_cast<std::size_t>(n_vir) * N_g * sizeof(real_t)), "alloc X_vir_t");
    check_cuda_mp2(cudaMalloc(&d_O, Ng2 * sizeof(real_t)), "alloc O");
    check_cuda_mp2(cudaMalloc(&d_V, Ng2 * sizeof(real_t)), "alloc V");
    check_cuda_mp2(cudaMalloc(&d_U, Ng2 * sizeof(real_t)), "alloc U");
    check_cuda_mp2(cudaMalloc(&d_ZU, Ng2 * sizeof(real_t)), "alloc ZU");
    check_cuda_mp2(cudaMalloc(&d_T, Ng2 * sizeof(real_t)), "alloc T");

    real_t E_OS_local = 0.0;

    for (int it = tau_start; it < tau_end; ++it) {
        const real_t tau = static_cast<real_t>(laplace.points[it]);
        const real_t w   = static_cast<real_t>(laplace.weights[it]);

        // 1. Scaled X_occ, X_vir
        {
            const dim3 threads(16, 16);
            const dim3 blocks((N_g + 15) / 16, (N_orb + 15) / 16);
            scale_X_for_laplace_kernel<<<blocks, threads>>>(
                d_X_mo, d_eps, tau, n_occ, n_vir, N_orb, N_g,
                d_X_occ_t, d_X_vir_t);
            check_cuda_mp2(cudaGetLastError(), "scale_X_for_laplace_kernel");
        }

        // 2. O = X_occ_t^T X_occ_t  (N_g x N_g)
        check_cublas_mp2(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N_g, N_g, n_occ,
                                      &one, d_X_occ_t, n_occ, d_X_occ_t, n_occ,
                                      &zero, d_O, N_g),
                          "X_occ_t^T X_occ_t");

        // 3. V = X_vir_t^T X_vir_t  (N_g x N_g)
        check_cublas_mp2(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                      N_g, N_g, n_vir,
                                      &one, d_X_vir_t, n_vir, d_X_vir_t, n_vir,
                                      &zero, d_V, N_g),
                          "X_vir_t^T X_vir_t");

        // 4. U = O ∘ V
        {
            const int threads = 256;
            const int blocks  = static_cast<int>((Ng2 + threads - 1) / threads);
            hadamard_kernel<<<blocks, threads>>>(d_O, d_V, d_U, Ng2);
            check_cuda_mp2(cudaGetLastError(), "hadamard O∘V");
        }

        // 5. ZU = Z U,  T = ZU Z^T  (N_g x N_g)
        check_cublas_mp2(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                      N_g, N_g, N_g,
                                      &one, d_Z, N_g, d_U, N_g,
                                      &zero, d_ZU, N_g),
                          "Z U");
        check_cublas_mp2(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                      N_g, N_g, N_g,
                                      &one, d_ZU, N_g, d_Z, N_g,
                                      &zero, d_T, N_g),
                          "ZU Z^T");

        // 6. E_tau = sum(U ∘ T) = dot(U_flat, T_flat)
        real_t E_tau = 0.0;
        check_cublas_mp2(cublasDdot(cublas, static_cast<int>(Ng2),
                                     d_U, 1, d_T, 1, &E_tau),
                          "dot U,T");

        E_OS_local += w * E_tau;
    }

    cudaFree(d_T);
    cudaFree(d_ZU);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_X_vir_t);
    cudaFree(d_X_occ_t);

    return E_OS_local;
}

real_t compute_thc_sos_mp2_energy_gpu(const real_t* d_X_mo,
                                       const real_t* d_Z,
                                       const real_t* d_eps,
                                       int n_occ, int N_orb, int N_g,
                                       int n_laplace,
                                       double c_os,
                                       int num_gpus)
{
    if (n_occ <= 0 || n_occ >= N_orb)
        throw std::runtime_error("compute_thc_sos_mp2_energy_gpu: invalid n_occ");
    const int n_vir = N_orb - n_occ;

    // ---- Determine Laplace integration range from orbital energies (host side)
    std::vector<real_t> h_eps(N_orb);
    check_cuda_mp2(cudaMemcpy(h_eps.data(), d_eps, N_orb * sizeof(real_t),
                               cudaMemcpyDeviceToHost),
                    "copy d_eps");
    const real_t e_homo = h_eps[n_occ - 1];
    const real_t e_lumo = h_eps[n_occ];
    const real_t e_occ_min = h_eps[0];
    const real_t e_vir_max = h_eps[N_orb - 1];
    const double Delta_min = static_cast<double>(e_lumo - e_homo);
    const double Delta_max = static_cast<double>(e_vir_max - e_occ_min);
    if (Delta_min <= 0.0)
        throw std::runtime_error("compute_thc_sos_mp2_energy_gpu: non-positive HOMO-LUMO gap");

    LaplaceQuadrature laplace =
        generate_laplace_quadrature(Delta_min, Delta_max, n_laplace);

    real_t E_OS = 0.0;

    if (num_gpus <= 1) {
        // ---- Single-GPU path ---------------------------------------------------
        E_OS = thc_sos_mp2_partial_on_device(
            d_X_mo, d_Z, d_eps,
            n_occ, n_vir, N_orb, N_g,
            0, n_laplace,
            laplace,
            gpu::GPUHandle::cublas());
    } else {
        // ---- Multi-GPU tau-parallel path --------------------------------------
        // Each device processes a contiguous subset of tau points. Inputs
        // (X_mo, Z, eps) are replicated to peer devices; outputs are scalars
        // summed back on host.
        auto& mgr = MultiGpuManager::instance();
        mgr.initialize(num_gpus);
        const int N = mgr.num_devices();
        if (N <= 1) {
            // Manager downgraded; fall back to single GPU.
            return compute_thc_sos_mp2_energy_gpu(
                d_X_mo, d_Z, d_eps, n_occ, N_orb, N_g,
                n_laplace, c_os, 1);
        }

        const std::size_t X_size   = static_cast<std::size_t>(N_orb) * N_g * sizeof(real_t);
        const std::size_t Z_size   = static_cast<std::size_t>(N_g)   * N_g * sizeof(real_t);
        const std::size_t eps_size = static_cast<std::size_t>(N_orb) * sizeof(real_t);

        std::vector<real_t*> d_X_mo_per(N, nullptr);
        std::vector<real_t*> d_Z_per(N,   nullptr);
        std::vector<real_t*> d_eps_per(N, nullptr);
        d_X_mo_per[0] = const_cast<real_t*>(d_X_mo);
        d_Z_per[0]    = const_cast<real_t*>(d_Z);
        d_eps_per[0]  = const_cast<real_t*>(d_eps);

        // Replicate inputs to peer GPUs (fast over NVLink with cudaMemcpyDefault).
        for (int d = 1; d < N; ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            check_cuda_mp2(cudaMalloc(&d_X_mo_per[d], X_size),  "alloc X_mo peer");
            check_cuda_mp2(cudaMalloc(&d_Z_per[d],    Z_size),  "alloc Z peer");
            check_cuda_mp2(cudaMalloc(&d_eps_per[d],  eps_size), "alloc eps peer");
            cudaMemcpy(d_X_mo_per[d], d_X_mo,  X_size,   cudaMemcpyDefault);
            cudaMemcpy(d_Z_per[d],    d_Z,     Z_size,   cudaMemcpyDefault);
            cudaMemcpy(d_eps_per[d],  d_eps,   eps_size, cudaMemcpyDefault);
        }

        // Distribute tau points across GPUs.
        std::vector<real_t> E_per(N, 0.0);
        std::vector<std::string> err_msg(N);

#ifdef _OPENMP
        #pragma omp parallel num_threads(N)
        {
            const int d = omp_get_thread_num();
#else
        for (int d = 0; d < N; ++d) {
#endif
            try {
                MultiGpuManager::DeviceGuard guard(d);
                const int t0 = (d * n_laplace) / N;
                const int t1 = ((d + 1) * n_laplace) / N;
                E_per[d] = thc_sos_mp2_partial_on_device(
                    d_X_mo_per[d], d_Z_per[d], d_eps_per[d],
                    n_occ, n_vir, N_orb, N_g,
                    t0, t1,
                    laplace,
                    mgr.cublas(d));
            } catch (const std::exception& e) {
                err_msg[d] = e.what();
            }
#ifdef _OPENMP
        }
#else
        }
#endif

        for (int d = 0; d < N; ++d) {
            if (!err_msg[d].empty()) {
                throw std::runtime_error("compute_thc_sos_mp2_energy_gpu (GPU "
                                          + std::to_string(d) + "): "
                                          + err_msg[d]);
            }
            E_OS += E_per[d];
        }

        // Free peer-replicated inputs (skip d == 0; that is the caller's buffer).
        for (int d = 1; d < N; ++d) {
            MultiGpuManager::DeviceGuard guard(d);
            cudaFree(d_X_mo_per[d]);
            cudaFree(d_Z_per[d]);
            cudaFree(d_eps_per[d]);
        }
    }

    // GANSU SOS-MP2 convention: SOS-MP2 = c_os * E_os, no spin-2 factor.
    // E_OS accumulator holds sum_tau w_tau E_tau (= |E_os| > 0 in the limit
    // of exact Laplace + LS-THC), so SOS-MP2 = -c_os * E_OS.
    return static_cast<real_t>(-c_os) * E_OS;
}

#endif // GANSU_CPU_ONLY

} // namespace gansu
