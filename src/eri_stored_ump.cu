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

#include <iomanip>
#include <iostream>
#include <assert.h>

#include "uhf.hpp"
#include "eri_stored.hpp"
#include "device_host_memory.hpp"

#include "ao2mo.cuh"

#define FULLMASK 0xffffffff

namespace gansu {





//*
__global__ void compute_ump2_energy_contrib_ss(
    double* g_energy_second, 
    const double* g_eri_mo, const double* g_eps, 
    const int num_occupied, const int num_virtual)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_tmp = 0;
    }
    __syncthreads();

    double tmp = 0.0;
    const size_t seq = (((size_t)blockDim.x * blockDim.y) * blockIdx.x) + blockDim.x * threadIdx.y + threadIdx.x;
    if (seq < (size_t)num_occupied * num_virtual * (size_t)num_occupied * num_virtual) {
        const int ia = seq / (num_occupied * num_virtual);
        const int jb = seq % (num_occupied * num_virtual);
        const int i = ia / num_virtual;
        const int a = ia % num_virtual;
        const int j = jb / num_virtual;
        const int b = jb % num_virtual;

        const double iajb = g_eri_mo[ovov2seq(i, a, j, b, num_occupied, num_virtual)];
        const double jaib = g_eri_mo[ovov2seq(j, a, i, b, num_occupied, num_virtual)];
        //tmp = iajb * (2 * iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
        tmp = iajb * (iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += __shfl_down_sync(FULLMASK, tmp, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_tmp, tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_second, s_tmp * 0.5);
    }
}
/**/


//*
__global__ void compute_ump2_energy_contrib_os(
    double* g_energy_second, const double* g_eri_mo, 
    const double* g_eps_al, const double* g_eps_be, 
    const int num_occupied_al, const int num_virtual_al, 
    const int num_occupied_be, const int num_virtual_be)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_tmp = 0;
    }
    __syncthreads();

    double tmp = 0.0;
    const size_t seq = (((size_t)blockDim.x * blockDim.y) * blockIdx.x) + blockDim.x * threadIdx.y + threadIdx.x;
    if (seq < (size_t)num_occupied_al * num_virtual_al * (size_t)num_occupied_be * num_virtual_be) {
        const int ia = seq / (num_occupied_be * num_virtual_be);
        const int jb = seq % (num_occupied_be * num_virtual_be);
        const int i = ia / num_virtual_al;
        const int a = ia % num_virtual_al;
        const int j = jb / num_virtual_be;
        const int b = jb % num_virtual_be;

        const double iajb = g_eri_mo[ovov2seq_aabb(i, a, j, b, num_occupied_al, num_virtual_al, num_occupied_be, num_virtual_be)];
        tmp = (iajb * iajb) / (g_eps_al[i] + g_eps_be[j] - g_eps_al[num_occupied_al + a] - g_eps_be[num_occupied_be + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += __shfl_down_sync(FULLMASK, tmp, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_tmp, tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_second, s_tmp);
    }
}
/**/





double ump2_from_aoeri_via_required_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis, 
    const int num_occupied_orbitals_al,
    const int num_occupied_orbitals_be)
{
    double* d_eri_tmp1 = nullptr;
    double* d_eri_tmp2 = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const int max_num_occ = std::max(num_occupied_orbitals_al, num_occupied_orbitals_be);
    tracked_cudaMalloc(&d_eri_tmp1, sizeof(double) * num_basis_2 * num_basis_2);
    tracked_cudaMalloc(&d_eri_tmp2, sizeof(double) * max_num_occ * num_basis_2 * num_basis);
    if (!d_eri_tmp1) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp_1."); }
    if (!d_eri_tmp2) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp_2."); }

    const int num_virtual_orbitals_al = num_basis - num_occupied_orbitals_al;
    const int num_virtual_orbitals_be = num_basis - num_occupied_orbitals_be;

    double* d_second_energy = nullptr;
    tracked_cudaMalloc(&d_second_energy, sizeof(double));
    cudaMemset(d_second_energy, 0, sizeof(double));

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;

    float time_aa, time_bb, time_ab;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);
    // Compute alpha-alpha energy contribution
    {
        std::string str = "Computing 1st term... ";
        PROFILE_ELAPSED_TIME(str);

        cudaMemcpy(d_eri_tmp1, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2, cudaMemcpyDeviceToDevice);
        cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, num_occupied_orbitals_al, num_virtual_orbitals_al);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_aa = d_eri_tmp1;

        const size_t total = (size_t)num_occupied_orbitals_al * num_virtual_orbitals_al * num_occupied_orbitals_al * num_virtual_orbitals_al;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aaaa
        compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occupied_orbitals_al, num_virtual_orbitals_al);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_aa, begin, end);
    printf("alpha-alpha: %.2f [ms]\n", time_aa);


    cudaEventRecord(begin);
    // Compute beta-beta energy contribution
    {
        std::string str = "Computing 2nd term... ";
        PROFILE_ELAPSED_TIME(str);

        cudaMemcpy(d_eri_tmp1, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2, cudaMemcpyDeviceToDevice);
        cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_be, num_occupied_orbitals_be, num_virtual_orbitals_be);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_bb = d_eri_tmp1;

        const size_t total = (size_t)num_occupied_orbitals_be * num_virtual_orbitals_be * num_occupied_orbitals_be * num_virtual_orbitals_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // bbbb
        compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occupied_orbitals_be, num_virtual_orbitals_be);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_bb, begin, end);
    printf("beta-beta: %.2f [ms]\n", time_bb);


    cudaEventRecord(begin);
    // Compute alpha-beta energy contribution
    {
        std::string str = "Computing 3rd term... ";
        PROFILE_ELAPSED_TIME(str);

        cudaMemcpy(d_eri_tmp1, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2, cudaMemcpyDeviceToDevice);
        cudaMemset(d_eri_tmp2, 0, sizeof(double) * max_num_occ * num_basis_2 * num_basis);

        // AO ERIs (d_eri_tmp1) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov_os(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, d_coefficient_matrix_be, num_occupied_orbitals_al, num_virtual_orbitals_al, num_occupied_orbitals_be, num_virtual_orbitals_be);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_ab = d_eri_tmp1;

        const size_t total = (size_t)num_occupied_orbitals_al * num_virtual_orbitals_al * num_occupied_orbitals_be * num_virtual_orbitals_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aabb
        compute_ump2_energy_contrib_os<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occupied_orbitals_al, num_virtual_orbitals_al, num_occupied_orbitals_be, num_virtual_orbitals_be);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ab, begin, end);
    printf("alpha-beta: %.2f [ms]\n", time_ab);


    double h_second_energy = 0.0;
    cudaMemcpy(&h_second_energy, d_second_energy, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "UMP2 correlation energy: " << std::setprecision(12) << h_second_energy << std::endl;

    tracked_cudaFree(d_eri_tmp1);
    tracked_cudaFree(d_eri_tmp2);
    tracked_cudaFree(d_second_energy);

    return h_second_energy;
}










real_t ERI_Stored_UHF::compute_mp2_energy() 
{
    PROFILE_FUNCTION();

    const int num_basis = uhf_.get_num_basis();
    const int num_occ_al = uhf_.get_num_alpha_spins();
    const int num_occ_be = uhf_.get_num_beta_spins();

    DeviceHostMatrix<real_t>& coefficient_matrix_al = uhf_.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& coefficient_matrix_be = uhf_.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& orbital_energies_al = uhf_.get_orbital_energies_a();
    DeviceHostMemory<real_t>& orbital_energies_be = uhf_.get_orbital_energies_b();

    //const real_t* d_C = coefficient_matrix.device_ptr();
    //const real_t* d_eps = orbital_energies.device_ptr();
    //real_t* d_eri = eri_matrix_.device_ptr();

    //const real_t E_UMP2 = 1.0;
    const real_t E_UMP2 = ump2_from_aoeri_via_required_moeri(
        eri_matrix_.device_ptr(), 
        coefficient_matrix_al.device_ptr(), 
        coefficient_matrix_be.device_ptr(), 
        orbital_energies_al.device_ptr(),
        orbital_energies_be.device_ptr(),
        num_basis, 
        num_occ_al, 
        num_occ_be
    );

    std::cout << "UMP2 energy test" << std::endl;

    return E_UMP2;
}














}   // namespace gansu
