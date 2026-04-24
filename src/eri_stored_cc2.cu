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
 * @file eri_stored_cc2.cu
 * @brief Standalone CC2 ground state energy calculation using stored ERIs
 *
 * Computes the CC2 correlation energy by solving the CC2 amplitude equations.
 * Reuses the CC2 solver from EOM-CC2 infrastructure.
 */

#include <iomanip>
#include <iostream>

#include "rhf.hpp"
#include "cc2_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"
#include "profiler.hpp"

namespace gansu {

// Forward declarations
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);

// Reuse EOM-MP2 kernels for ERI block extraction
extern __global__ void eom_mp2_extract_eri_ovov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_ooov_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_oovv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_ovvo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_extract_eri_vvvv_kernel(
    const real_t*, real_t*, int, int, int, int, int);
extern __global__ void eom_mp2_extract_eri_oooo_kernel(
    const real_t*, real_t*, int, int, int);
extern __global__ void eom_mp2_compute_D1_kernel(
    const real_t*, real_t*, int, int);
extern __global__ void eom_mp2_extract_fock_kernel(
    const real_t*, real_t*, real_t*, int, int);
extern __global__ void cc2_standard_D2_kernel(
    const real_t*, real_t*, int, int);


static real_t compute_cc2_energy_impl(RHF& rhf, const real_t* d_eri_ao, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_basis = rhf.get_num_basis();
    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const int singles_dim = num_occ * num_vir;
    const int doubles_dim = num_occ * num_occ * num_vir * num_vir;

    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    const real_t* d_C = coefficient_matrix.device_ptr();

    std::cout << "\n---- CC2 ground state ---- "
              << "nocc=" << num_occ << ", nvir=" << num_vir << std::endl;

    // Step 1: Transform AO ERIs to MO ERIs
    real_t* d_eri_mo;
    bool free_eri_mo;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        tracked_cudaMalloc(&d_eri_mo,
                           (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(real_t));
        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, num_basis, d_eri_mo);
        free_eri_mo = true;
    }

    // Step 2: Extract ERI blocks for CC2 solver
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_orbital_energies = orbital_energies.device_ptr();

    int threads = 256;
    int blocks;

    // OVOV
    size_t ovov_size = (size_t)num_occ * num_vir * num_occ * num_vir;
    real_t* d_eri_ovov = nullptr;
    tracked_cudaMalloc(&d_eri_ovov, ovov_size * sizeof(real_t));

    // VVOV
    size_t vvov_size = (size_t)num_vir * num_vir * num_occ * num_vir;
    real_t* d_eri_vvov = nullptr;
    tracked_cudaMalloc(&d_eri_vvov, vvov_size * sizeof(real_t));

    // OOOV
    size_t ooov_size = (size_t)num_occ * num_occ * num_occ * num_vir;
    real_t* d_eri_ooov = nullptr;
    tracked_cudaMalloc(&d_eri_ooov, ooov_size * sizeof(real_t));

    // OOVV
    size_t oovv_size = (size_t)num_occ * num_occ * num_vir * num_vir;
    real_t* d_eri_oovv = nullptr;
    tracked_cudaMalloc(&d_eri_oovv, oovv_size * sizeof(real_t));

    // OVVO
    size_t ovvo_size = (size_t)num_occ * num_vir * num_vir * num_occ;
    real_t* d_eri_ovvo = nullptr;
    tracked_cudaMalloc(&d_eri_ovvo, ovvo_size * sizeof(real_t));

    // VVVV
    size_t vvvv_size = (size_t)num_vir * num_vir * num_vir * num_vir;
    real_t* d_eri_vvvv = nullptr;
    tracked_cudaMalloc(&d_eri_vvvv, vvvv_size * sizeof(real_t));

    // OOOO
    size_t oooo_size = (size_t)num_occ * num_occ * num_occ * num_occ;
    real_t* d_eri_oooo = nullptr;
    tracked_cudaMalloc(&d_eri_oooo, oooo_size * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback for all ERI block extractions
        // eri_mo[p*N³ + r*N² + q*N + s] (chemist notation, 2nd/3rd swapped)
        size_t nao2 = (size_t)num_basis * num_basis;
        // OVOV: (ia|jb)
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovov_size; idx++) {
            int i = (int)(idx / (num_vir * num_occ * num_vir));
            int rem = (int)(idx % (num_vir * num_occ * num_vir));
            int a = rem / (num_occ * num_vir); rem %= (num_occ * num_vir);
            int j = rem / num_vir; int b = rem % num_vir;
            d_eri_ovov[idx] = d_eri_mo[((size_t)i*num_basis + a+num_occ)*nao2 + (size_t)j*num_basis + b+num_occ];
        }
        // VVOV: (ab|ic)
        #pragma omp parallel for
        for (size_t idx = 0; idx < vvov_size; idx++) {
            int a = (int)(idx / (num_vir * num_occ * num_vir));
            int rem = (int)(idx % (num_vir * num_occ * num_vir));
            int b = rem / (num_occ * num_vir); rem %= (num_occ * num_vir);
            int i = rem / num_vir; int c = rem % num_vir;
            d_eri_vvov[idx] = d_eri_mo[((size_t)(a+num_occ)*num_basis + b+num_occ)*nao2 + (size_t)i*num_basis + c+num_occ];
        }
        // OOOV: (ji|kb)
        #pragma omp parallel for
        for (size_t idx = 0; idx < ooov_size; idx++) {
            int j = (int)(idx / (num_occ * num_occ * num_vir));
            int rem = (int)(idx % (num_occ * num_occ * num_vir));
            int i = rem / (num_occ * num_vir); rem %= (num_occ * num_vir);
            int k = rem / num_vir; int b = rem % num_vir;
            d_eri_ooov[idx] = d_eri_mo[((size_t)j*num_basis + i)*nao2 + (size_t)k*num_basis + b+num_occ];
        }
        // OOVV: (ij|ab)
        #pragma omp parallel for
        for (size_t idx = 0; idx < oovv_size; idx++) {
            int i = (int)(idx / (num_occ * num_vir * num_vir));
            int rem = (int)(idx % (num_occ * num_vir * num_vir));
            int j = rem / (num_vir * num_vir); rem %= (num_vir * num_vir);
            int a = rem / num_vir; int b = rem % num_vir;
            d_eri_oovv[idx] = d_eri_mo[((size_t)i*num_basis + j)*nao2 + (size_t)(a+num_occ)*num_basis + b+num_occ];
        }
        // OVVO: (ia|bj)
        #pragma omp parallel for
        for (size_t idx = 0; idx < ovvo_size; idx++) {
            int i = (int)(idx / (num_vir * num_vir * num_occ));
            int rem = (int)(idx % (num_vir * num_vir * num_occ));
            int a = rem / (num_vir * num_occ); rem %= (num_vir * num_occ);
            int b = rem / num_occ; int j = rem % num_occ;
            d_eri_ovvo[idx] = d_eri_mo[((size_t)i*num_basis + a+num_occ)*nao2 + (size_t)(b+num_occ)*num_basis + j];
        }
        // VVVV: (ab|cd)
        #pragma omp parallel for
        for (size_t idx = 0; idx < vvvv_size; idx++) {
            int a = (int)(idx / (num_vir * num_vir * num_vir));
            int rem = (int)(idx % (num_vir * num_vir * num_vir));
            int b = rem / (num_vir * num_vir); rem %= (num_vir * num_vir);
            int c = rem / num_vir; int d = rem % num_vir;
            d_eri_vvvv[idx] = d_eri_mo[((size_t)(a+num_occ)*num_basis + b+num_occ)*nao2 + (size_t)(c+num_occ)*num_basis + d+num_occ];
        }
        // OOOO: (ij|kl)
        #pragma omp parallel for
        for (size_t idx = 0; idx < oooo_size; idx++) {
            int i = (int)(idx / (num_occ * num_occ * num_occ));
            int rem = (int)(idx % (num_occ * num_occ * num_occ));
            int j = rem / (num_occ * num_occ); rem %= (num_occ * num_occ);
            int k = rem / num_occ; int l = rem % num_occ;
            d_eri_oooo[idx] = d_eri_mo[((size_t)i*num_basis + j)*nao2 + (size_t)k*num_basis + l];
        }
    } else {
        blocks = (ovov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovov, num_occ, num_vir, num_basis);
        blocks = (vvov_size + threads - 1) / threads;
        eom_mp2_extract_eri_vvov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvov, num_occ, num_vir, num_basis);
        blocks = (ooov_size + threads - 1) / threads;
        eom_mp2_extract_eri_ooov_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ooov, num_occ, num_vir, num_basis);
        blocks = (oovv_size + threads - 1) / threads;
        eom_mp2_extract_eri_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oovv, num_occ, num_vir, num_basis, 0, -1);
        blocks = (ovvo_size + threads - 1) / threads;
        eom_mp2_extract_eri_ovvo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_ovvo, num_occ, num_vir, num_basis);
        blocks = (vvvv_size + threads - 1) / threads;
        eom_mp2_extract_eri_vvvv_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_vvvv, num_occ, num_vir, num_basis, 0, -1);
        blocks = (oooo_size + threads - 1) / threads;
        eom_mp2_extract_eri_oooo_kernel<<<blocks, threads>>>(d_eri_mo, d_eri_oooo, num_occ, num_basis, 0);
    }

    // Free full MO ERIs
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    // D1, D2, Fock
    real_t* d_D1 = nullptr;
    tracked_cudaMalloc(&d_D1, (size_t)singles_dim * sizeof(real_t));

    real_t* d_D2 = nullptr;
    tracked_cudaMalloc(&d_D2, (size_t)doubles_dim * sizeof(real_t));

    real_t* d_f_oo = nullptr;
    real_t* d_f_vv = nullptr;
    tracked_cudaMalloc(&d_f_oo, (size_t)num_occ * sizeof(real_t));
    tracked_cudaMalloc(&d_f_vv, (size_t)num_vir * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // D1
        for (int idx = 0; idx < singles_dim; idx++)
            d_D1[idx] = d_orbital_energies[idx % num_vir + num_occ] - d_orbital_energies[idx / num_vir];
        // D2 (standard: eps_a + eps_b - eps_i - eps_j)
        for (int idx = 0; idx < doubles_dim; idx++) {
            int i = idx / (num_occ * num_vir * num_vir);
            int rem = idx % (num_occ * num_vir * num_vir);
            int j = rem / (num_vir * num_vir); rem %= (num_vir * num_vir);
            int a = rem / num_vir; int b = rem % num_vir;
            d_D2[idx] = d_orbital_energies[a+num_occ] + d_orbital_energies[b+num_occ]
                      - d_orbital_energies[i] - d_orbital_energies[j];
        }
        // Fock
        for (int i = 0; i < num_occ; i++) d_f_oo[i] = d_orbital_energies[i];
        for (int a = 0; a < num_vir; a++) d_f_vv[a] = d_orbital_energies[a + num_occ];
    } else {
        blocks = (singles_dim + threads - 1) / threads;
        eom_mp2_compute_D1_kernel<<<blocks, threads>>>(d_orbital_energies, d_D1, num_occ, num_vir);
        blocks = (doubles_dim + threads - 1) / threads;
        cc2_standard_D2_kernel<<<blocks, threads>>>(d_orbital_energies, d_D2, num_occ, num_vir);
        blocks = (num_basis + threads - 1) / threads;
        eom_mp2_extract_fock_kernel<<<blocks, threads>>>(d_orbital_energies, d_f_oo, d_f_vv, num_occ, num_vir);
    }

    cudaDeviceSynchronize();

    // Step 3: Solve CC2 ground state amplitudes
    Timer cc2_timer;
    CC2Result cc2 = solve_cc2(
        d_eri_ovov, d_eri_vvov, d_eri_ooov, d_eri_oovv, d_eri_ovvo,
        d_eri_vvvv, d_eri_oooo,
        d_f_oo, d_f_vv, d_D1, d_D2,
        num_occ, num_vir);

    std::cout << "  CC2 solver time: " << std::fixed << std::setprecision(3)
              << cc2_timer.elapsed_seconds() << " s" << std::endl;

    real_t cc2_energy = cc2.cc2_energy;

    // Cleanup
    tracked_cudaFree(d_eri_ovov);
    tracked_cudaFree(d_eri_vvov);
    tracked_cudaFree(d_eri_ooov);
    tracked_cudaFree(d_eri_oovv);
    tracked_cudaFree(d_eri_ovvo);
    tracked_cudaFree(d_eri_vvvv);
    tracked_cudaFree(d_eri_oooo);
    tracked_cudaFree(d_D1);
    tracked_cudaFree(d_D2);
    tracked_cudaFree(d_f_oo);
    tracked_cudaFree(d_f_vv);
    tracked_cudaFree(cc2.d_t1);
    tracked_cudaFree(cc2.d_t2);

    return cc2_energy;
}

real_t ERI_Stored_RHF::compute_cc2_energy() {
    return compute_cc2_energy_impl(rhf_, eri_matrix_.device_ptr());
}

real_t ERI_RI_RHF::compute_cc2_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_cc2_energy_impl(rhf_, nullptr, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Direct_RHF::compute_cc2_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_cc2_energy_impl(rhf_, nullptr, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Hash_RHF::compute_cc2_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_cc2_energy_impl(rhf_, nullptr, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

} // namespace gansu
