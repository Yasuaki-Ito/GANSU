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

static inline dim3 make_2d_grid_from_1d_blocks(const size_t num_blocks_1d, const cudaDeviceProp& prop)
{
    if (num_blocks_1d == 0) {
        return dim3(1, 1, 1);
    }

    const size_t max_x = static_cast<size_t>(prop.maxGridSize[0]);
    const size_t max_y = static_cast<size_t>(prop.maxGridSize[1]);

    if (num_blocks_1d <= max_x) {
        return dim3(static_cast<unsigned int>(num_blocks_1d), 1, 1);
    }

    const size_t grid_x = max_x;
    const size_t grid_y = (num_blocks_1d + grid_x - 1) / grid_x;
    if (grid_y > max_y) {
        THROW_EXCEPTION("Error: Too many blocks for the 2D grid size.");
    }

    return dim3(static_cast<unsigned int>(grid_x), static_cast<unsigned int>(grid_y), 1);
}


void transform_ump3_full_mo_eris(
    double* d_eri_ao,
    double* d_g_aaaa_full,
    double* d_g_aabb_full,
    double* d_g_bbaa_full,
    double* d_g_bbbb_full,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const size_t num_basis_4,
    const int num_basis)
{
    // Reuse d_g_bbbb_full as temporary storage for the original AO ERIs.
    double* d_eri_tmp = d_g_bbbb_full;
    cudaMemcpy(d_eri_tmp, d_eri_ao, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);

    transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_aaaa_full, d_coefficient_matrix_al, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_aabb_full, d_coefficient_matrix_al, d_coefficient_matrix_be, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_bbaa_full, d_coefficient_matrix_be, d_coefficient_matrix_al, num_basis);

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_bbbb_full, d_coefficient_matrix_be, num_basis);

    cudaDeviceSynchronize();
}


void transform_ump3_single_mo_eri(
    double* d_eri_ao,
    double* d_g_full,
    double* d_eri_tmp,
    const double* d_coefficient_matrix_1,
    const double* d_coefficient_matrix_2,
    const size_t num_basis_4,
    const int num_basis,
    const bool same_spin)
{
    cudaMemcpy(d_eri_tmp, d_eri_ao, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);

    if (same_spin) {
        transform_eri_ao2mo_dgemm_full(d_eri_ao, d_g_full, d_coefficient_matrix_1, num_basis);
    } else {
        transform_eri_ao2mo_dgemm_full_os(d_eri_ao, d_g_full, d_coefficient_matrix_1, d_coefficient_matrix_2, num_basis);
    }

    cudaMemcpy(d_eri_ao, d_eri_tmp, sizeof(double) * num_basis_4, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}






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
    const int num_occ_al,
    const int num_occ_be)
{
    double* d_eri_tmp1 = nullptr;
    double* d_eri_tmp2 = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const int max_num_occ = std::max(num_occ_al, num_occ_be);
    tracked_cudaMalloc(&d_eri_tmp1, sizeof(double) * num_basis_2 * num_basis_2);
    tracked_cudaMalloc(&d_eri_tmp2, sizeof(double) * max_num_occ * num_basis_2 * num_basis);
    if (!d_eri_tmp1) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp_1."); }
    if (!d_eri_tmp2) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp_2."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

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
        transform_eri_ao2mo_dgemm_ovov(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, num_occ_al, num_vir_al);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_aa = d_eri_tmp1;

        const size_t total = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aaaa
        compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_aa, d_orbital_energies_al, num_occ_al, num_vir_al);
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
        transform_eri_ao2mo_dgemm_ovov(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_be, num_occ_be, num_vir_be);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_bb = d_eri_tmp1;

        const size_t total = (size_t)num_occ_be * num_vir_be * num_occ_be * num_vir_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // bbbb
        compute_ump2_energy_contrib_ss<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_bb, d_orbital_energies_be, num_occ_be, num_vir_be);
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
        transform_eri_ao2mo_dgemm_ovov_os(d_eri_tmp1, d_eri_tmp2, d_coefficient_matrix_al, d_coefficient_matrix_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        cudaDeviceSynchronize();
        double* d_eri_mo_ovov_ab = d_eri_tmp1;

        const size_t total = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
        const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
        const dim3 blocks(num_blocks);
        const dim3 threads(num_threads_per_warp, num_warps_per_block);

        // aabb
        compute_ump2_energy_contrib_os<<<blocks, threads>>>(d_second_energy, d_eri_mo_ovov_ab, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
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












__global__ void compute_4h2p_ss(
    double* g_energy_4h2p, 
    const double* g_int2e_aaaa, const double* g_eps_al, 
    const int num_occ_al, const int num_vir_al) 
{
    __shared__ double s_energy_4h2p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_4h2p = 0;
    }
    __syncthreads();

    const size_t tid_4h2p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t serial_4h2p = num_occ_al_2 * num_occ_al_2 * num_vir_al_2;

    double energy_4h2p = 0.0;
    if (tid_4h2p < serial_4h2p) {
        const size_t ijkl = tid_4h2p / num_vir_al_2;
        const int ab = tid_4h2p % num_vir_al_2;
        const int ij = ijkl / num_occ_al_2;
        const int kl = ijkl % num_occ_al_2;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int k = kl / num_occ_al;
        const int l = kl % num_occ_al;
        const int a = num_occ_al + (ab / num_vir_al);
        const int b = num_occ_al + (ab % num_vir_al);

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_klab = g_eps_al[k] + g_eps_al[l] - g_eps_al[a] - g_eps_al[b];
        const double numerator = 
            g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] * \
            g_int2e_aaaa[q2s(i, k, j, l, num_orbitals)] * \
            (g_int2e_aaaa[q2s(k, a, l, b, num_orbitals)] - g_int2e_aaaa[q2s(k, b, l, a, num_orbitals)]);
        energy_4h2p = numerator / (eps_ijab * eps_klab);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_4h2p += __shfl_down_sync(FULLMASK, energy_4h2p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_4h2p, energy_4h2p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_4h2p, s_energy_4h2p);
    }
}


__global__ void compute_4h2p_os(
    double* g_energy_4h2p,
    const double* g_int2e_aabb, 
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_4h2p;
    if (threadIdx.x == 0) {
        s_energy_4h2p = 0.0;
    }
    __syncthreads();

    //const size_t tid_4h2p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t tid_4h2p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;

    const size_t occa2 = num_occ_al * num_occ_al;
    const size_t occb2 = num_occ_be * num_occ_be;
    const size_t virab = num_vir_al * num_vir_be;
    const size_t serial_4h2p_os = occa2 * occb2 * virab;

    double energy_4h2p = 0.0;
    if (tid_4h2p < serial_4h2p_os) {
        const size_t ikjl = tid_4h2p / virab;
        const size_t ab   = tid_4h2p % virab;
        const size_t ik = ikjl / occb2;
        const size_t jl = ikjl % occb2;
        const int i = ik / num_occ_al;
        const int k = ik % num_occ_al;
        const int j = jl / num_occ_be;
        const int l = jl % num_occ_be;
        const int a = num_occ_al + (ab / num_vir_be);
        const int b = num_occ_be + (ab % num_vir_be);

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_klab = g_eps_al[k] + g_eps_be[l] - g_eps_al[a] - g_eps_be[b];
        const double numerator =
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(i, k, j, l, num_orbitals)] *
            g_int2e_aabb[q2s(k, a, l, b, num_orbitals)];

        energy_4h2p = numerator / (eps_ijab * eps_klab);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_4h2p += __shfl_down_sync(FULLMASK, energy_4h2p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_4h2p, energy_4h2p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_4h2p, s_energy_4h2p);
    }
}


__global__ void compute_2h4p_ss(
    double* g_energy_2h4p, 
    const double* g_int2e_aaaa, const double* g_eps_al, 
    const int num_occ_al, const int num_vir_al) 
{
    __shared__ double s_energy_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_2h4p = 0;
    }
    __syncthreads();

    const size_t tid_2h4p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t serial_2h4p = num_occ_al_2 * num_vir_al_2 * num_vir_al_2;

    double energy_2h4p = 0.0;
    if (tid_2h4p < serial_2h4p) {
        const int ij = tid_2h4p / (num_vir_al_2 * num_vir_al_2);
        const size_t abcd = tid_2h4p % (num_vir_al_2 * num_vir_al_2);
        const int ab = abcd / num_vir_al_2;
        const int cd = abcd % num_vir_al_2;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int a = num_occ_al + (ab / num_vir_al);
        const int b = num_occ_al + (ab % num_vir_al);
        const int c = num_occ_al + (cd / num_vir_al);
        const int d = num_occ_al + (cd % num_vir_al);

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ijcd = g_eps_al[i] + g_eps_al[j] - g_eps_al[c] - g_eps_al[d];
        const double numerator = 
            g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] * \
            g_int2e_aaaa[q2s(a, c, b, d, num_orbitals)] * \
            (g_int2e_aaaa[q2s(i, c, j, d, num_orbitals)] - g_int2e_aaaa[q2s(i, d, j, c, num_orbitals)]);

        energy_2h4p = numerator / (eps_ijab * eps_ijcd);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_2h4p += __shfl_down_sync(FULLMASK, energy_2h4p, offset);
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_2h4p, energy_2h4p);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_2h4p, s_energy_2h4p);
    }
}



__global__ void compute_2h4p_os(
    double* g_energy_2h4p,
    const double* g_int2e_aabb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_2h4p = 0;
    }
    __syncthreads();

    const size_t tid_2h4p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;
    const size_t serial_2h4p = num_occ_al * num_occ_be * num_vir_al_2 * num_vir_be_2;

    double energy_2h4p = 0.0;
    if (tid_2h4p < serial_2h4p) {
        const int ij = tid_2h4p / (num_vir_al_2 * num_vir_be_2);
        const size_t abcd = tid_2h4p % (num_vir_al_2 * num_vir_be_2);
        const int ac = abcd / num_vir_be_2;
        const int bd = abcd % num_vir_be_2;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int a = num_occ_al + (ac / num_vir_al);
        const int c = num_occ_al + (ac % num_vir_al);
        const int b = num_occ_be + (bd / num_vir_be);
        const int d = num_occ_be + (bd % num_vir_be);

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ijcd = g_eps_al[i] + g_eps_be[j] - g_eps_al[c] - g_eps_be[d];
        const double numerator =
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(a, c, b, d, num_orbitals)] *
            g_int2e_aabb[q2s(i, c, j, d, num_orbitals)];

        energy_2h4p = numerator / (eps_ijab * eps_ijcd);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_2h4p += __shfl_down_sync(FULLMASK, energy_2h4p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_2h4p, energy_2h4p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_2h4p, s_energy_2h4p);
    }
}











__global__ void compute_3h3p_aaaaaa(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa,
    const double* g_eps_al,
    const int num_occ_al, const int num_vir_al)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t num_occ_al_3 = (size_t)num_occ_al * num_occ_al * num_occ_al;
    const size_t num_vir_al_3 = (size_t)num_vir_al * num_vir_al * num_vir_al;
    const size_t serial_3h3p = num_occ_al_3 * num_vir_al_3;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / num_vir_al_3;
        const size_t abc = tid_3h3p % num_vir_al_3;
        //const int i = ijk / (num_occ_al * num_occ_al);
        //const int j = (ijk / num_occ_al) % num_occ_al;
        //const int k = ijk % num_occ_al;
        //const int a = num_occ_al + abc / (num_vir_al * num_vir_al);
        //const int b = num_occ_al + (abc / num_vir_al) % num_vir_al;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_al;
        const int k = ijk % num_occ_al;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_al + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_al[k] - g_eps_al[a] - g_eps_al[c];
        const double numerator = 
            (g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] - g_int2e_aaaa[q2s(i, b, j, a, num_orbitals)]) *
            (g_int2e_aaaa[q2s(i, a, k, c, num_orbitals)] - g_int2e_aaaa[q2s(i, c, k, a, num_orbitals)]) *
            (g_int2e_aaaa[q2s(k, c, b, j, num_orbitals)] - g_int2e_aaaa[q2s(k, j, b, c, num_orbitals)]);
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_aabaab(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa, 
    const double* g_int2e_aabb,
    const double* g_int2e_bbaa,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_al * num_occ_be * num_vir_al * num_vir_al * num_vir_be;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_al * num_vir_be);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_al * num_vir_be);
        //const int i = ijk / (num_occ_al * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_al;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_al + abc / (num_vir_al * num_vir_be);
        //const int b = num_occ_al + (abc / num_vir_be) % num_vir_al;
        //const int c = num_occ_be + abc % num_vir_be;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int ab = abc / num_vir_be;
        const int c = num_occ_be + abc % num_vir_be;
        const int a = num_occ_al + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_al[a] - g_eps_be[c];
        const double numerator = 
            (g_int2e_aaaa[q2s(i, a, j, b, num_orbitals)] - g_int2e_aaaa[q2s(i, b, j, a, num_orbitals)]) *
            g_int2e_aabb[q2s(i, a, k, c, num_orbitals)] *
            g_int2e_bbaa[q2s(k, c, b, j, num_orbitals)];
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_abaaba(
    double* g_energy_3h3p,
    const double* g_int2e_aaaa, 
    const double* g_int2e_aabb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_al * num_vir_al * num_vir_be * num_vir_al;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_be * num_vir_al);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_be * num_vir_al);
        //const int i = ijk / (num_occ_be * num_occ_al);
        //const int j = (ijk / num_occ_al) % num_occ_be;
        //const int k = ijk % num_occ_al;
        //const int a = num_occ_al + abc / (num_vir_be * num_vir_al);
        //const int b = num_occ_be + (abc / num_vir_al) % num_vir_be;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_al;
        const int k = ijk % num_occ_al;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_al + ab / num_vir_be;
        const int b = num_occ_be + ab % num_vir_be;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ikac = g_eps_al[i] + g_eps_al[k] - g_eps_al[a] - g_eps_al[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            (g_int2e_aaaa[q2s(i, a, k, c, num_orbitals)] - g_int2e_aaaa[q2s(i, c, k, a, num_orbitals)]) *
            g_int2e_aabb[q2s(k, c, b, j, num_orbitals)];
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}

__global__ void compute_3h3p_abbabb(
    double* g_energy_3h3p,
    const double* g_int2e_aabb,
    const double* g_int2e_bbbb,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_be * num_vir_al * num_vir_be * num_vir_be;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_al * num_vir_be * num_vir_be);
        const size_t abc = tid_3h3p % ((size_t)num_vir_al * num_vir_be * num_vir_be);
        //const int i = ijk / (num_occ_be * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_be;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_al + abc / (num_vir_be * num_vir_be);
        //const int b = num_occ_be + (abc / num_vir_be) % num_vir_be;
        //const int c = num_occ_be + abc % num_vir_be;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_be;
        const int c = num_occ_be + abc % num_vir_be;
        const int a = num_occ_al + ab / num_vir_be;
        const int b = num_occ_be + ab % num_vir_be;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_al[a] - g_eps_be[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, a, j, b, num_orbitals)] *
            g_int2e_aabb[q2s(i, a, k, c, num_orbitals)] *
            (g_int2e_bbbb[q2s(k, c, b, j, num_orbitals)] - g_int2e_bbbb[q2s(k, j, b, c, num_orbitals)]);
        energy_3h3p = numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}


__global__ void compute_3h3p_abbbaa(
    double* g_energy_3h3p,
    const double* g_int2e_aabb,
    const double* g_int2e_bbaa,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0.0;
    }
    __syncthreads();

    const size_t tid_3h3p = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const int num_orbitals = num_occ_al + num_vir_al;
    const size_t serial_3h3p = (size_t)num_occ_al * num_occ_be * num_occ_be * num_vir_be * num_vir_al * num_vir_al;

    double energy_3h3p = 0.0;
    if (tid_3h3p < serial_3h3p) {
        const size_t ijk = tid_3h3p / ((size_t)num_vir_be * num_vir_al * num_vir_al);
        const size_t abc = tid_3h3p % ((size_t)num_vir_be * num_vir_al * num_vir_al);
        //const int i = ijk / (num_occ_be * num_occ_be);
        //const int j = (ijk / num_occ_be) % num_occ_be;
        //const int k = ijk % num_occ_be;
        //const int a = num_occ_be + abc / (num_vir_al * num_vir_al);
        //const int b = num_occ_al + (abc / num_vir_al) % num_vir_al;
        //const int c = num_occ_al + abc % num_vir_al;
        const int ij = ijk / num_occ_be;
        const int k = ijk % num_occ_be;
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int ab = abc / num_vir_al;
        const int c = num_occ_al + abc % num_vir_al;
        const int a = num_occ_be + ab / num_vir_al;
        const int b = num_occ_al + ab % num_vir_al;

        const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_be[a] - g_eps_al[b];
        const double eps_ikac = g_eps_al[i] + g_eps_be[k] - g_eps_be[a] - g_eps_al[c];
        const double numerator = 
            g_int2e_aabb[q2s(i, b, j, a, num_orbitals)] *
            g_int2e_aabb[q2s(i, c, k, a, num_orbitals)] *
            g_int2e_bbaa[q2s(k, j, b, c, num_orbitals)];
        energy_3h3p = (-1) * numerator / (eps_ijab * eps_ikac);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        energy_3h3p += __shfl_down_sync(FULLMASK, energy_3h3p, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_energy_3h3p, energy_3h3p);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_3h3p, s_energy_3h3p);
    }
}









double ump3_from_aoeri_via_full_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al, const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al, const double* d_orbital_energies_be,
    const int num_basis, const int num_occ_al, const int num_occ_be)
{
    double* d_g_aaaa_full = nullptr;
    double* d_g_aabb_full = nullptr;
    double* d_g_bbaa_full = nullptr;
    double* d_g_bbbb_full = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_aaaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_aabb_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbbb_full, sizeof(double) * num_basis_4);
    if (!d_g_aaaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_full."); }
    if (!d_g_aabb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_full."); }
    if (!d_g_bbaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_full."); }
    if (!d_g_bbbb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_full."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    double* d_energy_4h2p = nullptr;
    double* d_energy_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p, 0, sizeof(double));
    cudaMemset(d_energy_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    transform_ump3_full_mo_eris(
        d_eri_ao,
        d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
        d_coefficient_matrix_al, d_coefficient_matrix_be,
        num_basis_4, num_basis
    );

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 0.5;

    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_occ_be_2 = num_occ_be * num_occ_be;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;
    const size_t num_occ_al_3 = num_occ_al_2 * num_occ_al;
    const size_t num_occ_be_3 = num_occ_be_2 * num_occ_be;
    const size_t num_vir_al_3 = num_vir_al_2 * num_vir_al;
    const size_t num_vir_be_3 = num_vir_be_2 * num_vir_be;




    // 4h2p contributions
    double* d_energy_4h2p_aa = nullptr;
    double* d_energy_4h2p_ab = nullptr;
    double* d_energy_4h2p_ba = nullptr;
    double* d_energy_4h2p_bb = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_aa, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_ab, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_ba, sizeof(double));
    tracked_cudaMalloc(&d_energy_4h2p_bb, sizeof(double));
    cudaMemset(d_energy_4h2p_aa, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_ab, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_ba, 0, sizeof(double));
    cudaMemset(d_energy_4h2p_bb, 0, sizeof(double));

    const size_t num_4h2p_aa = num_occ_al_2 * num_occ_al_2 * num_vir_al_2;
    const size_t num_4h2p_bb = num_occ_be_2 * num_occ_be_2 * num_vir_be_2;
    const size_t num_4h2p_ab = num_occ_al_2 * num_occ_be_2 * num_vir_al * num_vir_be;
    const size_t num_4h2p_ba = num_occ_be_2 * num_occ_al_2 * num_vir_be * num_vir_al;
    const size_t num_blocks_4h2p_aa = (num_4h2p_aa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_bb = (num_4h2p_bb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_ab = (num_4h2p_ab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_4h2p_ba = (num_4h2p_ba + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_4h2p_aa = make_2d_grid_from_1d_blocks(num_blocks_4h2p_aa, prop);
    const dim3 blocks_4h2p_bb = make_2d_grid_from_1d_blocks(num_blocks_4h2p_bb, prop);
    const dim3 blocks_4h2p_ab = make_2d_grid_from_1d_blocks(num_blocks_4h2p_ab, prop);
    const dim3 blocks_4h2p_ba = make_2d_grid_from_1d_blocks(num_blocks_4h2p_ba, prop);
    // 4h2p-aa
    compute_4h2p_ss<<<blocks_4h2p_aa, threads>>>(
        d_energy_4h2p, d_g_aaaa_full, d_orbital_energies_al, 
        //d_energy_4h2p_aa, d_g_aaaa_full, d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 4h2p-ab
    compute_4h2p_os<<<blocks_4h2p_ab, threads>>>(
        d_energy_4h2p, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        //d_energy_4h2p_ab, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 4h2p-ba
    compute_4h2p_os<<<blocks_4h2p_ba, threads>>>(
        d_energy_4h2p, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        //d_energy_4h2p_ba, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 4h2p-bb
    compute_4h2p_ss<<<blocks_4h2p_bb, threads>>>(
        d_energy_4h2p, d_g_bbbb_full, d_orbital_energies_be, 
        //d_energy_4h2p_bb, d_g_bbbb_full, d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );
    cudaDeviceSynchronize();
    // Scale the 4h2p energy by 0.5
    cublasDscal(handle, 1, &alpha, d_energy_4h2p, 1);





    // 2h4p contributions
    double* d_energy_2h4p_aa = nullptr;
    double* d_energy_2h4p_ab = nullptr;
    double* d_energy_2h4p_ba = nullptr;
    double* d_energy_2h4p_bb = nullptr;
    tracked_cudaMalloc(&d_energy_2h4p_aa, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_ab, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_ba, sizeof(double));
    tracked_cudaMalloc(&d_energy_2h4p_bb, sizeof(double));
    cudaMemset(d_energy_2h4p_aa, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_ab, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_ba, 0, sizeof(double));
    cudaMemset(d_energy_2h4p_bb, 0, sizeof(double));


    const size_t num_2h4p_aa = num_occ_al_2 * num_vir_al_2 * num_vir_al_2;
    const size_t num_2h4p_ab = num_occ_al * num_occ_be * num_vir_al_2 * num_vir_be_2;
    const size_t num_2h4p_ba = num_occ_be * num_occ_al * num_vir_be_2 * num_vir_al_2;
    const size_t num_2h4p_bb = num_occ_be_2 * num_vir_be_2 * num_vir_be_2;
    const size_t num_blocks_2h4p_aa = (num_2h4p_aa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_ab = (num_2h4p_ab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_ba = (num_2h4p_ba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_2h4p_bb = (num_2h4p_bb + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_2h4p_aa = make_2d_grid_from_1d_blocks(num_blocks_2h4p_aa, prop);
    const dim3 blocks_2h4p_ab = make_2d_grid_from_1d_blocks(num_blocks_2h4p_ab, prop);
    const dim3 blocks_2h4p_ba = make_2d_grid_from_1d_blocks(num_blocks_2h4p_ba, prop);
    const dim3 blocks_2h4p_bb = make_2d_grid_from_1d_blocks(num_blocks_2h4p_bb, prop);
    // 2h4p-aa
    compute_2h4p_ss<<<blocks_2h4p_aa, threads>>>(
        d_energy_2h4p, d_g_aaaa_full, d_orbital_energies_al, 
        //d_energy_2h4p_aa, d_g_aaaa_full, d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 2h4p-ab
    compute_2h4p_os<<<blocks_2h4p_ab, threads>>>(
        d_energy_2h4p, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        //d_energy_2h4p_ab, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be,
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 2h4p-ba
    compute_2h4p_os<<<blocks_2h4p_ba, threads>>>(
        d_energy_2h4p, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        //d_energy_2h4p_ba, d_g_bbaa_full, d_orbital_energies_be, d_orbital_energies_al,
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 2h4p-bb
    compute_2h4p_ss<<<blocks_2h4p_bb, threads>>>(
        d_energy_2h4p, d_g_bbbb_full, d_orbital_energies_be, 
        //d_energy_2h4p_bb, d_g_bbbb_full, d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );
    // Scale the 2h4p energy by 0.5
    cublasDscal(handle, 1, &alpha, d_energy_2h4p, 1);



    // 3h3p contributions
    double* d_energy_aaaaaa = nullptr;
    double* d_energy_aabaab = nullptr;
    double* d_energy_abaaba = nullptr;
    double* d_energy_abbabb = nullptr;
    double* d_energy_abbbaa = nullptr;
    double* d_energy_baaabb = nullptr;
    double* d_energy_baabaa = nullptr;
    double* d_energy_babbab = nullptr;
    double* d_energy_bbabba = nullptr;
    double* d_energy_bbbbbb = nullptr;
    tracked_cudaMalloc(&d_energy_aaaaaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_aabaab, sizeof(double));
    tracked_cudaMalloc(&d_energy_abaaba, sizeof(double));
    tracked_cudaMalloc(&d_energy_abbabb, sizeof(double));
    tracked_cudaMalloc(&d_energy_abbbaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_baaabb, sizeof(double));
    tracked_cudaMalloc(&d_energy_baabaa, sizeof(double));
    tracked_cudaMalloc(&d_energy_babbab, sizeof(double));
    tracked_cudaMalloc(&d_energy_bbabba, sizeof(double));
    tracked_cudaMalloc(&d_energy_bbbbbb, sizeof(double));
    cudaMemset(d_energy_aaaaaa, 0, sizeof(double));
    cudaMemset(d_energy_aabaab, 0, sizeof(double));
    cudaMemset(d_energy_abaaba, 0, sizeof(double));
    cudaMemset(d_energy_abbabb, 0, sizeof(double));
    cudaMemset(d_energy_abbbaa, 0, sizeof(double));
    cudaMemset(d_energy_baaabb, 0, sizeof(double));
    cudaMemset(d_energy_baabaa, 0, sizeof(double));
    cudaMemset(d_energy_babbab, 0, sizeof(double));
    cudaMemset(d_energy_bbabba, 0, sizeof(double));
    cudaMemset(d_energy_bbbbbb, 0, sizeof(double));

    const size_t num_3h3p_aaaaaa = num_occ_al_3                * num_vir_al_3;
    const size_t num_3h3p_aabaab = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_abaaba = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_abbabb = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_abbbaa = num_occ_al   * num_occ_be_2 * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_baaabb = num_occ_al_2 * num_occ_be   * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_baabaa = num_occ_al_2 * num_occ_be   * num_vir_al_2 * num_vir_be;
    const size_t num_3h3p_babbab = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_bbabba = num_occ_al   * num_occ_be_2 * num_vir_al   * num_vir_be_2;
    const size_t num_3h3p_bbbbbb =                num_occ_be_3                * num_vir_be_3;
    const size_t num_blocks_3h3p_aaaaaa = (num_3h3p_aaaaaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_aabaab = (num_3h3p_aabaab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abaaba = (num_3h3p_abaaba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abbabb = (num_3h3p_abbabb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_abbbaa = (num_3h3p_abbbaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_baaabb = (num_3h3p_baaabb + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_baabaa = (num_3h3p_baabaa + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_babbab = (num_3h3p_babbab + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_bbabba = (num_3h3p_bbabba + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_3h3p_bbbbbb = (num_3h3p_bbbbbb + num_threads_per_block - 1) / num_threads_per_block;
    const dim3 blocks_3h3p_aaaaaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_aaaaaa, prop);
    const dim3 blocks_3h3p_aabaab = make_2d_grid_from_1d_blocks(num_blocks_3h3p_aabaab, prop);
    const dim3 blocks_3h3p_abaaba = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abaaba, prop);
    const dim3 blocks_3h3p_abbabb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abbabb, prop);
    const dim3 blocks_3h3p_abbbaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_abbbaa, prop);
    const dim3 blocks_3h3p_baaabb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_baaabb, prop);
    const dim3 blocks_3h3p_baabaa = make_2d_grid_from_1d_blocks(num_blocks_3h3p_baabaa, prop);
    const dim3 blocks_3h3p_babbab = make_2d_grid_from_1d_blocks(num_blocks_3h3p_babbab, prop);
    const dim3 blocks_3h3p_bbabba = make_2d_grid_from_1d_blocks(num_blocks_3h3p_bbabba, prop);
    const dim3 blocks_3h3p_bbbbbb = make_2d_grid_from_1d_blocks(num_blocks_3h3p_bbbbbb, prop);
    // 3h3p-aaaaaa
    compute_3h3p_aaaaaa<<<blocks_3h3p_aaaaaa, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, 
        //d_energy_aaaaaa, d_g_aaaa_full, 
        d_orbital_energies_al, 
        num_occ_al, num_vir_al
    );
    // 3h3p-aabaab
    compute_3h3p_aabaab<<<blocks_3h3p_aabaab, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, 
        //d_energy_aabaab, d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abaaba
    compute_3h3p_abaaba<<<blocks_3h3p_abaaba, threads>>>(
        d_energy_3h3p, d_g_aaaa_full, d_g_aabb_full, 
        //d_energy_abaaba, d_g_aaaa_full, d_g_aabb_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abbabb
    compute_3h3p_abbabb<<<blocks_3h3p_abbabb, threads>>>(
        d_energy_3h3p, d_g_aabb_full, d_g_bbbb_full, 
        //d_energy_abbabb, d_g_aabb_full, d_g_bbbb_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-abbbaa
    compute_3h3p_abbbaa<<<blocks_3h3p_abbbaa, threads>>>(
        d_energy_3h3p, d_g_aabb_full, d_g_bbaa_full, 
        //d_energy_abbbaa, d_g_aabb_full, d_g_bbaa_full, 
        d_orbital_energies_al, d_orbital_energies_be, 
        num_occ_al, num_vir_al, num_occ_be, num_vir_be
    );
    // 3h3p-baaabb
    compute_3h3p_abbbaa<<<blocks_3h3p_baaabb, threads>>>(
        d_energy_3h3p, d_g_bbaa_full, d_g_aabb_full, 
        //d_energy_baaabb, d_g_bbaa_full, d_g_aabb_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-baabaa
    compute_3h3p_abbabb<<<blocks_3h3p_baabaa, threads>>>(
        d_energy_3h3p, d_g_bbaa_full, d_g_aaaa_full, 
        //d_energy_baabaa, d_g_bbaa_full, d_g_aaaa_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-babbab
    compute_3h3p_abaaba<<<blocks_3h3p_babbab, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, d_g_bbaa_full, 
        //d_energy_babbab, d_g_bbbb_full, d_g_bbaa_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-bbabba
    compute_3h3p_aabaab<<<blocks_3h3p_bbabba, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, 
        //d_energy_bbabba, d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, 
        d_orbital_energies_be, d_orbital_energies_al, 
        num_occ_be, num_vir_be, num_occ_al, num_vir_al
    );
    // 3h3p-bbbbbb
    compute_3h3p_aaaaaa<<<blocks_3h3p_bbbbbb, threads>>>(
        d_energy_3h3p, d_g_bbbb_full, 
        //d_energy_bbbbbb, d_g_bbbb_full, 
        d_orbital_energies_be, 
        num_occ_be, num_vir_be
    );


    cudaDeviceSynchronize();
    double h_energy_4h2p = 0.0;
    double h_energy_2h4p = 0.0;
    double h_energy_3h3p = 0.0;
    //double h_energy = 0.0;

    /*
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_aa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_aa: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_ab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_ab: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_ba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_ba: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p_bb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p_bb: " << 0.5 * h_energy_4h2p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_4h2p;
    std::cout << "Total E_4h2p: " << h_energy << " [hartree]" << std::endl;
    h_energy = 0.0;
    /**/

    /*
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_aa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_aa: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_ab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_ab: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_ba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_ba: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p_bb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_2h4p_bb: " << 0.5 * h_energy_2h4p << " [hartree]" << std::endl;
    h_energy += 0.5 * h_energy_2h4p;
    std::cout << "Total E_2h4p: " << h_energy << " [hartree]" << std::endl;
    h_energy = 0.0;
    /**/

    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_aaaaaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_aaaaaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_aabaab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_aabaab: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abaaba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abaaba: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abbabb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abbabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_abbbaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_abbbaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_baaabb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_baaabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_baabaa, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_baabaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_babbab, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_babbab: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_bbabba, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_bbabba: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    cudaMemcpy(&h_energy_3h3p, d_energy_bbbbbb, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_bbbbbb: " << h_energy_3h3p << " [hartree]" << std::endl;
    h_energy += h_energy_3h3p;
    std::cout << "Total E_3h3p: " << h_energy << " [hartree]" << std::endl;
    /**/

    //*
    cudaMemcpy(&h_energy_4h2p, d_energy_4h2p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_2h4p, d_energy_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_4h2p: " << h_energy_4h2p << " [hartree]" << std::endl;
    std::cout << "E_2h4p: " << h_energy_2h4p << " [hartree]" << std::endl;
    std::cout << "E_3h3p: " << h_energy_3h3p << " [hartree]" << std::endl;
    /**/

    tracked_cudaFree(d_g_aaaa_full);
    tracked_cudaFree(d_g_aabb_full);
    tracked_cudaFree(d_g_bbaa_full);
    tracked_cudaFree(d_g_bbbb_full);
    tracked_cudaFree(d_energy_4h2p);
    tracked_cudaFree(d_energy_2h4p);
    tracked_cudaFree(d_energy_3h3p);

    tracked_cudaFree(d_energy_4h2p_aa);
    tracked_cudaFree(d_energy_4h2p_ab);
    tracked_cudaFree(d_energy_4h2p_ba);
    tracked_cudaFree(d_energy_4h2p_bb);
    tracked_cudaFree(d_energy_2h4p_aa);
    tracked_cudaFree(d_energy_2h4p_ab);
    tracked_cudaFree(d_energy_2h4p_ba);
    tracked_cudaFree(d_energy_2h4p_bb);
    tracked_cudaFree(d_energy_aaaaaa);
    tracked_cudaFree(d_energy_aabaab);
    tracked_cudaFree(d_energy_abaaba);
    tracked_cudaFree(d_energy_abbabb);
    tracked_cudaFree(d_energy_abbbaa);
    tracked_cudaFree(d_energy_baaabb);
    tracked_cudaFree(d_energy_baabaa);
    tracked_cudaFree(d_energy_babbab);
    tracked_cudaFree(d_energy_bbabba);
    tracked_cudaFree(d_energy_bbbbbb);

    cublasDestroy(handle);

    //return 0.0;
    return h_energy_4h2p + h_energy_2h4p + h_energy_3h3p;
}













double compute_aaaa_contributions(
    double* d_energy_4h2p_2h4p,
    double* d_energy_3h3p,
    const double* d_g_aaaa_full, 
    const double* d_orbital_energies_al,
    const int num_occ_al,
    const int num_vir_al)
{
    // g: (ik|jl), (ac|bd)
    // u: (kc||bj)
    double* d_g_aaaa_oooo = nullptr;
    double* d_g_aaaa_vvvv = nullptr;
    double* d_u_aaaa_ovvo = nullptr;
    double* d_x_aaaa_ovov = nullptr;
    double* d_y_aaaa_ovov = nullptr;
    double* d_tmp_1 = nullptr;
    double* d_tmp_2 = nullptr;

    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_ov_al_2 = num_occ_al * num_vir_al;

    const size_t num_oooo = num_occ_al_2 * num_occ_al_2;
    const size_t num_vvvv = num_vir_al_2 * num_vir_al_2;
    const size_t num_ovov = num_occ_al_2 * num_vir_al_2;
    const size_t num_ovvo = num_occ_al_2 * num_vir_al_2;

    tracked_cudaMalloc(&d_g_aaaa_oooo, sizeof(double) * num_oooo);
    tracked_cudaMalloc(&d_g_aaaa_vvvv, sizeof(double) * num_vvvv);
    tracked_cudaMalloc(&d_u_aaaa_ovvo, sizeof(double) * num_ovvo);
    tracked_cudaMalloc(&d_x_aaaa_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_tmp_1, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_tmp_2, sizeof(double) * num_ovov);

    if (!d_g_aaaa_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_oooo."); }
    if (!d_g_aaaa_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_vvvv."); }
    if (!d_u_aaaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_aaaa_ovvo."); }
    if (!d_x_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aaaa_ovov."); }
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_tmp_1) { THROW_EXCEPTION("cudaMalloc failed for d_t_aaaa_ovov."); }
    if (!d_tmp_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_2."); }

    cudaMemset(d_g_aaaa_oooo, 0, sizeof(double) * num_oooo);
    cudaMemset(d_g_aaaa_vvvv, 0, sizeof(double) * num_vvvv);
    cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_ovvo);
    cudaMemset(d_x_aaaa_ovov, 0, sizeof(double) * num_ovov);
    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_ovov);
    cudaMemset(d_tmp_1, 0, sizeof(double) * num_ovov);
    cudaMemset(d_tmp_2, 0, sizeof(double) * num_ovov);

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_blocks_oooo = (num_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_vvvv = (num_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_ovov = (num_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_ovvo = (num_ovvo + num_threads_per_block - 1) / num_threads_per_block;

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    if (num_blocks_oooo > prop.maxGridSize[0] ||
        num_blocks_vvvv > prop.maxGridSize[0] ||
        num_blocks_ovov > prop.maxGridSize[0] ||
        num_blocks_ovvo > prop.maxGridSize[0]) {
        THROW_EXCEPTION("Error: Too many blocks for the grid size.");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    tensorize_g_aaaa_oooo<<<num_blocks_oooo, threads>>>(d_g_aaaa_oooo, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_g_aaaa_vvvv<<<num_blocks_vvvv, threads>>>(d_g_aaaa_vvvv, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_u_aaaa_ovvo<<<num_blocks_ovvo, threads>>>(d_u_aaaa_ovvo, d_g_aaaa_full, num_occ_al, num_vir_al);
    tensorize_x_aaaa_ovov<<<num_blocks_ovov, threads>>>(d_x_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);
    tensorize_y_aaaa_ovov<<<num_blocks_ovov, threads>>>(d_y_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

    // Y_iakc * u_kcbj --> t_iabj (d_tmp_1)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_ov_al_2, num_ov_al_2, num_ov_al_2,
        &alpha, d_y_aaaa_ovov, num_ov_al_2,
        d_u_aaaa_ovvo, num_ov_al_2,
        &beta, d_tmp_1, num_ov_al_2
    );
    const double* d_t_aaaa_ovvo = d_tmp_1;
    contract_3h3p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_3h3p, d_y_aaaa_ovov, d_t_aaaa_ovvo, num_occ_al, num_vir_al);

    // Y_klab --> d_tmp_1
    kalb2klab_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_aaaa_ovov, num_occ_al, num_vir_al);
    // g_ijkl * Y_klab --> d_tmp_2
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al_2, num_vir_al_2, num_occ_al_2,
        &alpha, d_g_aaaa_oooo, num_occ_al_2,
        d_tmp_1, num_vir_al_2,
        &beta, d_tmp_2, num_vir_al_2
    );
    const double* d_t_aaaa_oovv = d_tmp_2;

    double* d_tmp_3 = d_u_aaaa_ovvo;
    // Y_cdij --> d_tmp_1
    icjd2cdij_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_aaaa_ovov, num_occ_al, num_vir_al);
    // g_abcd * Y_cdij --> d_tmp_3
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_al_2, num_occ_al_2, num_vir_al_2,
        &alpha, d_g_aaaa_vvvv, num_vir_al_2,
        d_tmp_1, num_occ_al_2,
        &beta, d_tmp_3, num_occ_al_2
    );
    const double* d_t_aaaa_vvoo = d_tmp_3;
    contract_4h2p_2h4p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_aaaa_ovov, d_t_aaaa_oovv, d_t_aaaa_vvoo, num_occ_al, num_vir_al);

    cudaDeviceSynchronize();

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
    //std::cout << "E_3h3p_aaaaaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    //std::cout << "E_4h2p_2h4p_aaaaaa: " << h_energy_4h2p_2h4p << " [hartree]" << std::endl;

    cublasDestroy(cublasH);

    tracked_cudaFree(d_g_aaaa_oooo);
    tracked_cudaFree(d_g_aaaa_vvvv);
    tracked_cudaFree(d_u_aaaa_ovvo);
    tracked_cudaFree(d_x_aaaa_ovov);
    tracked_cudaFree(d_y_aaaa_ovov);
    tracked_cudaFree(d_tmp_1);
    tracked_cudaFree(d_tmp_2);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}





double compute_aabb_contributions(
    double* d_energy_4h2p_2h4p, double* d_energy_3h3p,
    const double* d_g_aaaa_full, const double* d_g_aabb_full, 
    const double* d_g_bbaa_full, const double* d_g_bbbb_full,
    const double* d_orbital_energies_al, const double* d_orbital_energies_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be, 
    const bool compute_4h2p_2h4p)
{
    // g: (ik|jl), (ac|bd), (kc|bj), (kj|bc)
    // u: (kc||bj)
    double* d_g_aabb_oooo = nullptr;
    double* d_g_aabb_vvvv = nullptr;
    double* d_g_bbaa_ovvo = nullptr;
    double* d_g_bbaa_oovv = nullptr;
    double* d_u_bbbb_ovvo = nullptr;
    double* d_x_aabb_ovov = nullptr;
    double* d_y_aaaa_ovov = nullptr;
    double* d_tmp_1 = nullptr;
    double* d_tmp_2 = nullptr;
    double* d_tmp_4 = nullptr;

    const int num_basis = num_occ_al + num_vir_al;
    const size_t num_occ_al_2 = num_occ_al * num_occ_al;
    const size_t num_vir_al_2 = num_vir_al * num_vir_al;
    const size_t num_occ_be_2 = num_occ_be * num_occ_be;
    const size_t num_vir_be_2 = num_vir_be * num_vir_be;

    const size_t num_aabb_oooo = num_occ_al_2 * num_occ_be_2;
    const size_t num_aabb_vvvv = num_vir_al_2 * num_vir_be_2;
    const size_t num_bbaa_ovvo = num_occ_be * num_vir_be * num_vir_al * num_occ_al;
    const size_t num_bbaa_oovv = num_occ_be_2 * num_vir_al_2;
    const size_t num_bbbb_ovvo = num_occ_be * num_vir_be * num_vir_be * num_occ_be;
    const size_t num_aabb_ovov = num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    const size_t num_aaaa_ovov = num_occ_al * num_vir_al * num_occ_al * num_vir_al;

    tracked_cudaMalloc(&d_g_aabb_oooo, sizeof(double) * num_aabb_oooo);
    tracked_cudaMalloc(&d_g_aabb_vvvv, sizeof(double) * num_aabb_vvvv);
    tracked_cudaMalloc(&d_g_bbaa_ovvo, sizeof(double) * num_bbaa_ovvo);
    tracked_cudaMalloc(&d_g_bbaa_oovv, sizeof(double) * num_bbaa_oovv);
    tracked_cudaMalloc(&d_u_bbbb_ovvo, sizeof(double) * num_bbbb_ovvo);
    tracked_cudaMalloc(&d_x_aabb_ovov, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_tmp_1, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_tmp_2, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_4, sizeof(double) * num_aabb_ovov);

    if (!d_g_aabb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oooo."); }
    if (!d_g_aabb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_vvvv."); }
    if (!d_g_bbaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_ovvo."); }
    if (!d_g_bbaa_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_oovv."); }
    if (!d_u_bbbb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_bbbb_ovvo."); }
    if (!d_x_aabb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aabb_ovov."); }
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_tmp_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_1."); }
    if (!d_tmp_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_2."); }
    if (!d_tmp_4) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_4."); }

    cudaMemset(d_g_aabb_oooo, 0, sizeof(double) * num_aabb_oooo);
    cudaMemset(d_g_aabb_vvvv, 0, sizeof(double) * num_aabb_vvvv);
    cudaMemset(d_g_bbaa_ovvo, 0, sizeof(double) * num_bbaa_ovvo);
    cudaMemset(d_g_bbaa_oovv, 0, sizeof(double) * num_bbaa_oovv);
    cudaMemset(d_u_bbbb_ovvo, 0, sizeof(double) * num_bbbb_ovvo);
    cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_tmp_1, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_tmp_2, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_tmp_4, 0, sizeof(double) * num_aabb_ovov);

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    const size_t num_blocks_aabb_oooo = (num_aabb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_vvvv = (num_aabb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_ovvo = (num_bbaa_ovvo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_oovv = (num_bbaa_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_ovvo = (num_bbbb_ovvo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_ovov = (num_aabb_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_ovov = (num_aaaa_ovov + num_threads_per_block - 1) / num_threads_per_block;

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    if (num_blocks_aabb_oooo > prop.maxGridSize[0] ||
        num_blocks_aabb_vvvv > prop.maxGridSize[0] ||
        num_blocks_bbaa_ovvo > prop.maxGridSize[0] ||
        num_blocks_bbaa_oovv > prop.maxGridSize[0] ||
        num_blocks_bbbb_ovvo > prop.maxGridSize[0] ||
        num_blocks_aabb_ovov > prop.maxGridSize[0] ||
        num_blocks_aaaa_ovov > prop.maxGridSize[0]) {
        THROW_EXCEPTION("Error: Too many blocks for the grid size.");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    tensorize_g_aabb_oooo<<<num_blocks_aabb_oooo, threads>>>(d_g_aabb_oooo, d_g_aabb_full, num_occ_al, num_occ_be, num_basis);
    tensorize_g_aabb_vvvv<<<num_blocks_aabb_vvvv, threads>>>(d_g_aabb_vvvv, d_g_aabb_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);
    tensorize_g_bbaa_ovvo<<<num_blocks_bbaa_ovvo, threads>>>(d_g_bbaa_ovvo, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
    tensorize_g_bbaa_oovv<<<num_blocks_bbaa_oovv, threads>>>(d_g_bbaa_oovv, d_g_bbaa_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
    tensorize_u_bbbb_ovvo<<<num_blocks_bbbb_ovvo, threads>>>(d_u_bbbb_ovvo, d_g_bbbb_full, num_occ_be, num_vir_be);
    tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_aabb_ovov, d_g_aabb_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_basis);
    tensorize_y_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_y_aaaa_ovov, d_g_aaaa_full, d_orbital_energies_al, num_occ_al, num_vir_al);

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;

    // X_iakc^aabb * g_kcbj^bbaa --> t_iabj^aaaa (d_tmp_1)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_al, num_vir_al * num_occ_al, num_occ_be * num_vir_be,
        &alpha, d_x_aabb_ovov, num_occ_be * num_vir_be,
        d_g_bbaa_ovvo, num_vir_al * num_occ_al,
        &beta, d_tmp_1, num_vir_al * num_occ_al
    );
    const double* d_t_aaaa_ovvo = d_tmp_1;
    contract_3h3p_aabaab_abaaba<<<num_blocks_aaaa_ovov, threads>>>(d_energy_3h3p, d_y_aaaa_ovov, d_t_aaaa_ovvo, num_occ_al, num_vir_al);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_aabaab: " << (0.5) * h_energy_3h3p << " [hartree]" << std::endl;
    std::cout << "E_3h3p_abaaba: " << (0.5) * h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/


    double* d_tmp_3 = d_g_bbaa_ovvo;
    // X_iakc^aabb * u_kcbj^bbbb --> t_iabj^aabb (d_tmp_3)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_al, num_vir_be * num_occ_be, num_occ_be * num_vir_be,
        &alpha, d_x_aabb_ovov, num_occ_be * num_vir_be,
        d_u_bbbb_ovvo, num_vir_be * num_occ_be,
        &beta, d_tmp_3, num_vir_be * num_occ_be
    );
    const double* d_t_aabb_ovvo = d_tmp_3;
    contract_3h3p_abbabb<<<num_blocks_aabb_ovov, threads>>>(d_energy_3h3p, d_x_aabb_ovov, d_t_aabb_ovvo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_abbabb: " << h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/

    // X_icka^aabb --> X_iakc^abba (d_tmp_3)
    aabb_icka2abba_iakc<<<num_blocks_aabb_ovov, threads>>>(d_tmp_3, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
    // X_iakc^abba * g_kcbj^baab --> t_iabj^abab (d_tmp_2)
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_al * num_vir_be, num_vir_al * num_occ_be, num_occ_be * num_vir_al,
        &alpha, d_tmp_3, num_occ_be * num_vir_al,
        d_g_bbaa_oovv, num_vir_al * num_occ_be,
        &beta, d_tmp_2, num_vir_al * num_occ_be
    );
    const double* d_t_abab_ovvo = d_tmp_2;
    contract_3h3p_abbbaa<<<num_blocks_aabb_ovov, threads>>>(d_energy_3h3p, d_x_aabb_ovov, d_t_abab_ovvo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
    cudaDeviceSynchronize();
    /*
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "E_3h3p_abbbaa: " << h_energy_3h3p << " [hartree]" << std::endl;
    cudaMemset(d_energy_3h3p, 0, sizeof(double));
    /**/

    if (compute_4h2p_2h4p) {
        // X_kalb^aabb --> X_klab^abab (d_tmp_2)
        aabb_kalb2abab_klab<<<num_blocks_aabb_ovov, threads>>>(d_tmp_2, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        // g_ijkl^abab * X_klab^abab --> t_ijab^abab (d_tmp_3)
        dgemm_device_row_major(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            num_occ_al * num_occ_be, num_vir_al * num_vir_be, num_occ_al * num_occ_be,
            &alpha, d_g_aabb_oooo, num_occ_al * num_occ_be,
            d_tmp_2, num_vir_al * num_vir_be,
            &beta, d_tmp_3, num_vir_al * num_vir_be
        );
        const double* d_t_abab_oovv = d_tmp_3;
        // X_icjd^aabb --> X_cdij^abab (d_tmp_2)
        aabb_icjd2abab_cdij<<<num_blocks_aabb_ovov, threads>>>(d_tmp_2, d_x_aabb_ovov, num_occ_al, num_vir_al, num_occ_be, num_vir_be);
        // g_abcd^abab * X_cdij^abab --> t_abij^abab (d_tmp_4)
        dgemm_device_row_major(
            cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
            num_vir_al * num_vir_be, num_occ_al * num_occ_be, num_vir_al * num_vir_be,
            &alpha, d_g_aabb_vvvv, num_vir_al * num_vir_be,
            d_tmp_2, num_occ_al * num_occ_be,
            &beta, d_tmp_4, num_occ_al * num_occ_be
        );
        const double* d_t_abab_vvoo = d_tmp_4;
        contract_4h2p_2h4p_ababab_bababa<<<num_blocks_aabb_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_aabb_ovov, d_t_abab_oovv, d_t_abab_vvoo, num_occ_al, num_occ_be, num_vir_al, num_vir_be);
        cudaDeviceSynchronize();
        /*
        cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "E_4h2p_2h4p_ababab: " << h_energy_4h2p_2h4p * 0.5 << " [hartree]" << std::endl;
        std::cout << "E_4h2p_2h4p_bababa: " << h_energy_4h2p_2h4p * 0.5 << " [hartree]" << std::endl;
        /**/
    }

    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(cublasH);

    tracked_cudaFree(d_g_aabb_oooo);
    tracked_cudaFree(d_g_aabb_vvvv);
    tracked_cudaFree(d_g_bbaa_ovvo);
    tracked_cudaFree(d_g_bbaa_oovv);
    tracked_cudaFree(d_u_bbbb_ovvo);
    tracked_cudaFree(d_x_aabb_ovov);
    tracked_cudaFree(d_y_aaaa_ovov);
    tracked_cudaFree(d_tmp_1);
    tracked_cudaFree(d_tmp_2);
    tracked_cudaFree(d_tmp_4);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}





double ump3_from_aoeri_via_full_moeri_dgemm(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis, 
    const int num_occ_al,
    const int num_occ_be)
{
    double* d_g_aaaa_full = nullptr;
    double* d_g_aabb_full = nullptr;
    double* d_g_bbaa_full = nullptr;
    double* d_g_bbbb_full = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_aaaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_aabb_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbaa_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_g_bbbb_full, sizeof(double) * num_basis_4);
    if (!d_g_aaaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_full."); }
    if (!d_g_aabb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_full."); }
    if (!d_g_bbaa_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_full."); }
    if (!d_g_bbbb_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_full."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    double* d_energy_4h2p_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    transform_ump3_full_mo_eris(
        d_eri_ao,
        d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
        d_coefficient_matrix_al, d_coefficient_matrix_be,
        num_basis_4, num_basis
    );

    // (alpha, alpha, alpha, alpha)
    const double energy_aaaa = 
        compute_aaaa_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_aaaa_full, d_orbital_energies_al, 
            num_occ_al, num_vir_al
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    //*
    // (alpha, alpha, beta, beta)
    const double energy_aabb = 
        compute_aabb_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_aaaa_full, d_g_aabb_full, d_g_bbaa_full, d_g_bbbb_full,
            d_orbital_energies_al, d_orbital_energies_be,
            num_occ_al, num_vir_al, num_occ_be, num_vir_be, 
            true
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));


    // (beta, beta, alpha, alpha)
    const double energy_bbaa = 
        compute_aabb_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_bbbb_full, d_g_bbaa_full, d_g_aabb_full, d_g_aaaa_full,
            d_orbital_energies_be, d_orbital_energies_al,
            num_occ_be, num_vir_be, num_occ_al, num_vir_al, 
            false
        );
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));


    // (beta, beta, beta, beta)
    const double energy_bbbb = 
        compute_aaaa_contributions(
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_g_bbbb_full, d_orbital_energies_be, 
            num_occ_be, num_vir_be
        );
    /**/

    tracked_cudaFree(d_g_aaaa_full);
    tracked_cudaFree(d_g_aabb_full);
    tracked_cudaFree(d_g_bbaa_full);
    tracked_cudaFree(d_g_bbbb_full);
    tracked_cudaFree(d_energy_4h2p_2h4p);
    tracked_cudaFree(d_energy_3h3p);


    return energy_aaaa + energy_aabb + energy_bbaa + energy_bbbb;
}


static void contract_same_spin_contributions_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_4h2p_2h4p,
    double* d_energy_3h3p,
    const double* d_x_ovov,
    const double* d_y_ovov,
    const double* d_g_oooo,
    const double* d_g_vvvv,
    double* d_u_ovvo,
    double* d_tmp_1,
    double* d_tmp_2,
    double* d_tmp_3,
    const dim3 threads,
    const size_t num_blocks_ovov,
    const int num_occ,
    const int num_vir)
{
    const size_t num_occ_2 = num_occ * num_occ;
    const size_t num_vir_2 = num_vir * num_vir;
    const size_t num_ov_2 = num_occ * num_vir;
    const double alpha = 1.0;
    const double beta = 0.0;

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_ov_2, num_ov_2, num_ov_2,
        &alpha, d_y_ovov, num_ov_2,
        d_u_ovvo, num_ov_2,
        &beta, d_tmp_1, num_ov_2
    );
    const double* d_t_ovvo = d_tmp_1;
    contract_3h3p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_3h3p, d_y_ovov, d_t_ovvo, num_occ, num_vir);

    kalb2klab_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_ovov, num_occ, num_vir);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_2, num_vir_2, num_occ_2,
        &alpha, d_g_oooo, num_occ_2,
        d_tmp_1, num_vir_2,
        &beta, d_tmp_2, num_vir_2
    );
    const double* d_t_oovv = d_tmp_2;

    icjd2cdij_aaaa<<<num_blocks_ovov, threads>>>(d_tmp_1, d_y_ovov, num_occ, num_vir);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_2, num_occ_2, num_vir_2,
        &alpha, d_g_vvvv, num_vir_2,
        d_tmp_1, num_occ_2,
        &beta, d_tmp_3, num_occ_2
    );
    const double* d_t_vvoo = d_tmp_3;
    contract_4h2p_2h4p_aaaaaa<<<num_blocks_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_ovov, d_t_oovv, d_t_vvoo, num_occ, num_vir);

    cudaDeviceSynchronize();
}


static void contract_mixed_yxg_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_y_same_ovov,
    const double* d_x_mixed_ovov,
    const double* d_g_rev_ovvo,
    double* d_tmp_same_ovov,
    const dim3 threads,
    const size_t num_blocks_same_ovov,
    const int num_occ_same,
    const int num_vir_same,
    const int num_occ_other,
    const int num_vir_other)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_same * num_vir_same, num_vir_same * num_occ_same, num_occ_other * num_vir_other,
        &alpha, d_x_mixed_ovov, num_occ_other * num_vir_other,
        d_g_rev_ovvo, num_vir_same * num_occ_same,
        &beta, d_tmp_same_ovov, num_vir_same * num_occ_same
    );
    const double* d_t_same_ovvo = d_tmp_same_ovov;
    contract_3h3p_aabaab_abaaba<<<num_blocks_same_ovov, threads>>>(d_energy_3h3p, d_y_same_ovov, d_t_same_ovvo, num_occ_same, num_vir_same);

    cudaDeviceSynchronize();
}


static void contract_mixed_xu_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_x_mixed_ovov,
    const double* d_u_same_ovvo,
    double* d_tmp_mixed_ovov,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_vir_1, num_vir_2 * num_occ_2, num_occ_2 * num_vir_2,
        &alpha, d_x_mixed_ovov, num_occ_2 * num_vir_2,
        d_u_same_ovvo, num_vir_2 * num_occ_2,
        &beta, d_tmp_mixed_ovov, num_vir_2 * num_occ_2
    );
    const double* d_t_mixed_ovvo = d_tmp_mixed_ovov;
    contract_3h3p_abbabb<<<num_blocks_mixed_ovov, threads>>>(d_energy_3h3p, d_x_mixed_ovov, d_t_mixed_ovvo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


static void contract_mixed_xg_oovv_3h3p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_3h3p,
    const double* d_x_mixed_ovov,
    const double* d_g_rev_oovv,
    double* d_tmp_reordered,
    double* d_tmp_mixed_ovov,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    aabb_icka2abba_iakc<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_vir_2, num_vir_1 * num_occ_2, num_occ_2 * num_vir_1,
        &alpha, d_tmp_reordered, num_occ_2 * num_vir_1,
        d_g_rev_oovv, num_vir_1 * num_occ_2,
        &beta, d_tmp_mixed_ovov, num_vir_1 * num_occ_2
    );
    const double* d_t_mixed_ovvo = d_tmp_mixed_ovov;
    contract_3h3p_abbbaa<<<num_blocks_mixed_ovov, threads>>>(d_energy_3h3p, d_x_mixed_ovov, d_t_mixed_ovvo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


static void contract_mixed_4h2p_2h4p_from_tensors(
    cublasHandle_t cublasH,
    double* d_energy_4h2p_2h4p,
    const double* d_x_mixed_ovov,
    const double* d_g_mixed_oooo,
    const double* d_g_mixed_vvvv,
    double* d_tmp_reordered,
    double* d_tmp_oovv,
    double* d_tmp_vvoo,
    const dim3 threads,
    const size_t num_blocks_mixed_ovov,
    const int num_occ_1,
    const int num_occ_2,
    const int num_vir_1,
    const int num_vir_2)
{
    const double alpha = 1.0;
    const double beta = 0.0;

    aabb_kalb2abab_klab<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_occ_1 * num_occ_2, num_vir_1 * num_vir_2, num_occ_1 * num_occ_2,
        &alpha, d_g_mixed_oooo, num_occ_1 * num_occ_2,
        d_tmp_reordered, num_vir_1 * num_vir_2,
        &beta, d_tmp_oovv, num_vir_1 * num_vir_2
    );
    const double* d_t_oovv = d_tmp_oovv;

    aabb_icjd2abab_cdij<<<num_blocks_mixed_ovov, threads>>>(d_tmp_reordered, d_x_mixed_ovov, num_occ_1, num_vir_1, num_occ_2, num_vir_2);
    dgemm_device_row_major(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        num_vir_1 * num_vir_2, num_occ_1 * num_occ_2, num_vir_1 * num_vir_2,
        &alpha, d_g_mixed_vvvv, num_vir_1 * num_vir_2,
        d_tmp_reordered, num_occ_1 * num_occ_2,
        &beta, d_tmp_vvoo, num_occ_1 * num_occ_2
    );
    const double* d_t_vvoo = d_tmp_vvoo;

    contract_4h2p_2h4p_ababab_bababa<<<num_blocks_mixed_ovov, threads>>>(d_energy_4h2p_2h4p, d_x_mixed_ovov, d_t_oovv, d_t_vvoo, num_occ_1, num_occ_2, num_vir_1, num_vir_2);

    cudaDeviceSynchronize();
}


double ump3_from_aoeri_via_full_moeri_dgemm_eff(
    double* d_eri_ao,
    const double* d_coefficient_matrix_al,
    const double* d_coefficient_matrix_be,
    const double* d_orbital_energies_al,
    const double* d_orbital_energies_be,
    const int num_basis,
    const int num_occ_al,
    const int num_occ_be)
{
    double* d_g_full = nullptr;
    double* d_eri_tmp = nullptr;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_4 = num_basis_2 * num_basis_2;
    tracked_cudaMalloc(&d_g_full, sizeof(double) * num_basis_4);
    tracked_cudaMalloc(&d_eri_tmp, sizeof(double) * num_basis_4);
    if (!d_g_full) { THROW_EXCEPTION("cudaMalloc failed for d_g_full."); }
    if (!d_eri_tmp) { THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp."); }

    const int num_vir_al = num_basis - num_occ_al;
    const int num_vir_be = num_basis - num_occ_be;

    double* d_energy_4h2p_2h4p = nullptr;
    double* d_energy_3h3p = nullptr;
    tracked_cudaMalloc(&d_energy_4h2p_2h4p, sizeof(double));
    tracked_cudaMalloc(&d_energy_3h3p, sizeof(double));
    cudaMemset(d_energy_4h2p_2h4p, 0, sizeof(double));
    cudaMemset(d_energy_3h3p, 0, sizeof(double));

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    const dim3 threads(num_threads_per_warp, num_warps_per_block);

    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    const auto check_num_blocks = [&prop](const size_t num_blocks) {
        if (num_blocks > static_cast<size_t>(prop.maxGridSize[0])) {
            THROW_EXCEPTION("Error: Too many blocks for the grid size.");
        }
    };

    const size_t num_aaaa_oooo = (size_t)num_occ_al * num_occ_al * num_occ_al * num_occ_al;
    const size_t num_aaaa_vvvv = (size_t)num_vir_al * num_vir_al * num_vir_al * num_vir_al;
    const size_t num_aaaa_ovov = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
    const size_t num_aabb_oooo = (size_t)num_occ_al * num_occ_be * num_occ_al * num_occ_be;
    const size_t num_aabb_vvvv = (size_t)num_vir_al * num_vir_be * num_vir_al * num_vir_be;
    const size_t num_aabb_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    const size_t num_aabb_oovv = (size_t)num_occ_al * num_occ_al * num_vir_be * num_vir_be;
    const size_t num_bbaa_oovv = (size_t)num_occ_be * num_occ_be * num_vir_al * num_vir_al;
    const size_t num_bbbb_oooo = (size_t)num_occ_be * num_occ_be * num_occ_be * num_occ_be;
    const size_t num_bbbb_vvvv = (size_t)num_vir_be * num_vir_be * num_vir_be * num_vir_be;
    const size_t num_bbbb_ovov = (size_t)num_occ_be * num_vir_be * num_occ_be * num_vir_be;
    const size_t num_same_ovov_max = std::max(num_aaaa_ovov, num_bbbb_ovov);

    const size_t num_blocks_aaaa_oooo = (num_aaaa_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_vvvv = (num_aaaa_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aaaa_ovov = (num_aaaa_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oooo = (num_aabb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_vvvv = (num_aabb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_ovov = (num_aabb_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_aabb_oovv = (num_aabb_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbaa_oovv = (num_bbaa_oovv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_oooo = (num_bbbb_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_vvvv = (num_bbbb_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const size_t num_blocks_bbbb_ovov = (num_bbbb_ovov + num_threads_per_block - 1) / num_threads_per_block;

    check_num_blocks(num_blocks_aaaa_oooo);
    check_num_blocks(num_blocks_aaaa_vvvv);
    check_num_blocks(num_blocks_aaaa_ovov);
    check_num_blocks(num_blocks_aabb_oooo);
    check_num_blocks(num_blocks_aabb_vvvv);
    check_num_blocks(num_blocks_aabb_ovov);
    check_num_blocks(num_blocks_aabb_oovv);
    check_num_blocks(num_blocks_bbaa_oovv);
    check_num_blocks(num_blocks_bbbb_oooo);
    check_num_blocks(num_blocks_bbbb_vvvv);
    check_num_blocks(num_blocks_bbbb_ovov);

    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    double* d_tmp_same_1 = nullptr;
    double* d_tmp_same_2 = nullptr;
    double* d_tmp_same_3 = nullptr;
    double* d_tmp_mixed_1 = nullptr;
    double* d_tmp_mixed_2 = nullptr;
    double* d_tmp_mixed_3 = nullptr;
    tracked_cudaMalloc(&d_tmp_same_1, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_2, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_same_3, sizeof(double) * num_same_ovov_max);
    tracked_cudaMalloc(&d_tmp_mixed_1, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_2, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_tmp_mixed_3, sizeof(double) * num_aabb_ovov);
    if (!d_tmp_same_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_1."); }
    if (!d_tmp_same_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_2."); }
    if (!d_tmp_same_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_same_3."); }
    if (!d_tmp_mixed_1) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_1."); }
    if (!d_tmp_mixed_2) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_2."); }
    if (!d_tmp_mixed_3) { THROW_EXCEPTION("cudaMalloc failed for d_tmp_mixed_3."); }

    double* d_y_aaaa_ovov = nullptr;
    double* d_u_aaaa_ovvo = nullptr;
    double* d_x_aabb_ovov = nullptr;
    double* d_g_aabb_oovv = nullptr;
    double* d_g_aabb_ovvo = nullptr;
    double* d_x_bbaa_ovov = nullptr;

    tracked_cudaMalloc(&d_y_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_u_aaaa_ovvo, sizeof(double) * num_aaaa_ovov);
    tracked_cudaMalloc(&d_x_aabb_ovov, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_g_aabb_oovv, sizeof(double) * num_aabb_oovv);
    tracked_cudaMalloc(&d_g_aabb_ovvo, sizeof(double) * num_aabb_ovov);
    tracked_cudaMalloc(&d_x_bbaa_ovov, sizeof(double) * num_aabb_ovov);
    if (!d_y_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_aaaa_ovov."); }
    if (!d_u_aaaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_aaaa_ovvo."); }
    if (!d_x_aabb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aabb_ovov."); }
    if (!d_g_aabb_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oovv."); }
    if (!d_g_aabb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_ovvo."); }
    if (!d_x_bbaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbaa_ovov."); }

    cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);
    cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
    cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);
    cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);

    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_al, d_coefficient_matrix_al,
            num_basis_4, num_basis, true
        );

        double* d_g_aaaa_oooo = nullptr;
        double* d_g_aaaa_vvvv = nullptr;
        double* d_x_aaaa_ovov = nullptr;
        tracked_cudaMalloc(&d_g_aaaa_oooo, sizeof(double) * num_aaaa_oooo);
        tracked_cudaMalloc(&d_g_aaaa_vvvv, sizeof(double) * num_aaaa_vvvv);
        tracked_cudaMalloc(&d_x_aaaa_ovov, sizeof(double) * num_aaaa_ovov);
        if (!d_g_aaaa_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_oooo."); }
        if (!d_g_aaaa_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aaaa_vvvv."); }
        if (!d_x_aaaa_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_aaaa_ovov."); }

        cudaMemset(d_g_aaaa_oooo, 0, sizeof(double) * num_aaaa_oooo);
        cudaMemset(d_g_aaaa_vvvv, 0, sizeof(double) * num_aaaa_vvvv);
        cudaMemset(d_x_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_y_aaaa_ovov, 0, sizeof(double) * num_aaaa_ovov);
        cudaMemset(d_u_aaaa_ovvo, 0, sizeof(double) * num_aaaa_ovov);

        tensorize_g_aaaa_oooo<<<num_blocks_aaaa_oooo, threads>>>(d_g_aaaa_oooo, d_g_full, num_occ_al, num_vir_al);
        tensorize_g_aaaa_vvvv<<<num_blocks_aaaa_vvvv, threads>>>(d_g_aaaa_vvvv, d_g_full, num_occ_al, num_vir_al);
        tensorize_u_aaaa_ovvo<<<num_blocks_aaaa_ovov, threads>>>(d_u_aaaa_ovvo, d_g_full, num_occ_al, num_vir_al);
        tensorize_x_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_x_aaaa_ovov, d_g_full, d_orbital_energies_al, num_occ_al, num_vir_al);
        tensorize_y_aaaa_ovov<<<num_blocks_aaaa_ovov, threads>>>(d_y_aaaa_ovov, d_g_full, d_orbital_energies_al, num_occ_al, num_vir_al);

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_aaaa_ovov, d_y_aaaa_ovov,
            d_g_aaaa_oooo, d_g_aaaa_vvvv,
            d_u_aaaa_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_aaaa_ovov,
            num_occ_al, num_vir_al
        );

        tracked_cudaFree(d_g_aaaa_oooo);
        tracked_cudaFree(d_g_aaaa_vvvv);
        tracked_cudaFree(d_x_aaaa_ovov);
    }

    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_al, d_coefficient_matrix_be,
            num_basis_4, num_basis, false
        );

        double* d_g_aabb_oooo = nullptr;
        double* d_g_aabb_vvvv = nullptr;
        tracked_cudaMalloc(&d_g_aabb_oooo, sizeof(double) * num_aabb_oooo);
        tracked_cudaMalloc(&d_g_aabb_vvvv, sizeof(double) * num_aabb_vvvv);
        if (!d_g_aabb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_oooo."); }
        if (!d_g_aabb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_aabb_vvvv."); }

        cudaMemset(d_g_aabb_oooo, 0, sizeof(double) * num_aabb_oooo);
        cudaMemset(d_g_aabb_vvvv, 0, sizeof(double) * num_aabb_vvvv);
        cudaMemset(d_x_aabb_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_aabb_oovv, 0, sizeof(double) * num_aabb_oovv);
        cudaMemset(d_g_aabb_ovvo, 0, sizeof(double) * num_aabb_ovov);

        tensorize_g_aabb_oooo<<<num_blocks_aabb_oooo, threads>>>(d_g_aabb_oooo, d_g_full, num_occ_al, num_occ_be, num_basis);
        tensorize_g_aabb_vvvv<<<num_blocks_aabb_vvvv, threads>>>(d_g_aabb_vvvv, d_g_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);
        tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_aabb_ovov, d_g_full, d_orbital_energies_al, d_orbital_energies_be, num_occ_al, num_vir_al, num_occ_be, num_vir_be, num_basis);
        tensorize_g_bbaa_oovv<<<num_blocks_aabb_oovv, threads>>>(d_g_aabb_oovv, d_g_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);
        tensorize_g_bbaa_ovvo<<<num_blocks_aabb_ovov, threads>>>(d_g_aabb_ovvo, d_g_full, num_occ_al, num_occ_be, num_vir_al, num_vir_be, num_basis);

        contract_mixed_4h2p_2h4p_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p,
            d_x_aabb_ovov,
            d_g_aabb_oooo,
            d_g_aabb_vvvv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            d_tmp_mixed_3,
            threads,
            num_blocks_aabb_ovov,
            num_occ_al, num_occ_be, num_vir_al, num_vir_be
        );

        tracked_cudaFree(d_g_aabb_oooo);
        tracked_cudaFree(d_g_aabb_vvvv);
    }

    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_be, d_coefficient_matrix_al,
            num_basis_4, num_basis, false
        );

        double* d_g_bbaa_ovvo = nullptr;
        double* d_g_bbaa_oovv = nullptr;
        tracked_cudaMalloc(&d_g_bbaa_ovvo, sizeof(double) * num_aabb_ovov);
        tracked_cudaMalloc(&d_g_bbaa_oovv, sizeof(double) * num_bbaa_oovv);
        if (!d_g_bbaa_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_ovvo."); }
        if (!d_g_bbaa_oovv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbaa_oovv."); }

        cudaMemset(d_x_bbaa_ovov, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_ovvo, 0, sizeof(double) * num_aabb_ovov);
        cudaMemset(d_g_bbaa_oovv, 0, sizeof(double) * num_bbaa_oovv);

        tensorize_x_aabb_ovov<<<num_blocks_aabb_ovov, threads>>>(d_x_bbaa_ovov, d_g_full, d_orbital_energies_be, d_orbital_energies_al, num_occ_be, num_vir_be, num_occ_al, num_vir_al, num_basis);
        tensorize_g_bbaa_ovvo<<<num_blocks_aabb_ovov, threads>>>(d_g_bbaa_ovvo, d_g_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);
        tensorize_g_bbaa_oovv<<<num_blocks_bbaa_oovv, threads>>>(d_g_bbaa_oovv, d_g_full, num_occ_be, num_occ_al, num_vir_be, num_vir_al, num_basis);

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_aaaa_ovov,
            d_x_aabb_ovov,
            d_g_bbaa_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_aaaa_ovov,
            num_occ_al, num_vir_al,
            num_occ_be, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_g_bbaa_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            num_occ_al, num_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_xg_oovv_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_g_aabb_oovv,
            d_tmp_mixed_1,
            d_tmp_mixed_2,
            threads,
            num_blocks_aabb_ovov,
            num_occ_be, num_occ_al, num_vir_be, num_vir_al
        );

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_bbaa_ovov,
            d_u_aaaa_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            num_occ_be, num_occ_al, num_vir_be, num_vir_al
        );

        tracked_cudaFree(d_g_bbaa_ovvo);
        tracked_cudaFree(d_g_bbaa_oovv);
        tracked_cudaFree(d_y_aaaa_ovov);
        tracked_cudaFree(d_u_aaaa_ovvo);
        tracked_cudaFree(d_g_aabb_oovv);
        d_y_aaaa_ovov = nullptr;
        d_u_aaaa_ovvo = nullptr;
        d_g_aabb_oovv = nullptr;
    }

    {
        transform_ump3_single_mo_eri(
            d_eri_ao, d_g_full, d_eri_tmp,
            d_coefficient_matrix_be, d_coefficient_matrix_be,
            num_basis_4, num_basis, true
        );

        double* d_g_bbbb_oooo = nullptr;
        double* d_g_bbbb_vvvv = nullptr;
        double* d_u_bbbb_ovvo = nullptr;
        double* d_x_bbbb_ovov = nullptr;
        double* d_y_bbbb_ovov = nullptr;
        tracked_cudaMalloc(&d_g_bbbb_oooo, sizeof(double) * num_bbbb_oooo);
        tracked_cudaMalloc(&d_g_bbbb_vvvv, sizeof(double) * num_bbbb_vvvv);
        tracked_cudaMalloc(&d_u_bbbb_ovvo, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_x_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        tracked_cudaMalloc(&d_y_bbbb_ovov, sizeof(double) * num_bbbb_ovov);
        if (!d_g_bbbb_oooo) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_oooo."); }
        if (!d_g_bbbb_vvvv) { THROW_EXCEPTION("cudaMalloc failed for d_g_bbbb_vvvv."); }
        if (!d_u_bbbb_ovvo) { THROW_EXCEPTION("cudaMalloc failed for d_u_bbbb_ovvo."); }
        if (!d_x_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_x_bbbb_ovov."); }
        if (!d_y_bbbb_ovov) { THROW_EXCEPTION("cudaMalloc failed for d_y_bbbb_ovov."); }

        cudaMemset(d_g_bbbb_oooo, 0, sizeof(double) * num_bbbb_oooo);
        cudaMemset(d_g_bbbb_vvvv, 0, sizeof(double) * num_bbbb_vvvv);
        cudaMemset(d_u_bbbb_ovvo, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_x_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);
        cudaMemset(d_y_bbbb_ovov, 0, sizeof(double) * num_bbbb_ovov);

        tensorize_g_aaaa_oooo<<<num_blocks_bbbb_oooo, threads>>>(d_g_bbbb_oooo, d_g_full, num_occ_be, num_vir_be);
        tensorize_g_aaaa_vvvv<<<num_blocks_bbbb_vvvv, threads>>>(d_g_bbbb_vvvv, d_g_full, num_occ_be, num_vir_be);
        tensorize_u_aaaa_ovvo<<<num_blocks_bbbb_ovov, threads>>>(d_u_bbbb_ovvo, d_g_full, num_occ_be, num_vir_be);
        tensorize_x_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_x_bbbb_ovov, d_g_full, d_orbital_energies_be, num_occ_be, num_vir_be);
        tensorize_y_aaaa_ovov<<<num_blocks_bbbb_ovov, threads>>>(d_y_bbbb_ovov, d_g_full, d_orbital_energies_be, num_occ_be, num_vir_be);

        contract_mixed_xu_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_x_aabb_ovov,
            d_u_bbbb_ovvo,
            d_tmp_mixed_1,
            threads,
            num_blocks_aabb_ovov,
            num_occ_al, num_occ_be, num_vir_al, num_vir_be
        );

        contract_mixed_yxg_3h3p_from_tensors(
            cublasH,
            d_energy_3h3p,
            d_y_bbbb_ovov,
            d_x_bbaa_ovov,
            d_g_aabb_ovvo,
            d_tmp_same_1,
            threads,
            num_blocks_bbbb_ovov,
            num_occ_be, num_vir_be,
            num_occ_al, num_vir_al
        );

        contract_same_spin_contributions_from_tensors(
            cublasH,
            d_energy_4h2p_2h4p, d_energy_3h3p,
            d_x_bbbb_ovov, d_y_bbbb_ovov,
            d_g_bbbb_oooo, d_g_bbbb_vvvv,
            d_u_bbbb_ovvo,
            d_tmp_same_1, d_tmp_same_2, d_tmp_same_3,
            threads, num_blocks_bbbb_ovov,
            num_occ_be, num_vir_be
        );

        tracked_cudaFree(d_g_bbbb_oooo);
        tracked_cudaFree(d_g_bbbb_vvvv);
        tracked_cudaFree(d_u_bbbb_ovvo);
        tracked_cudaFree(d_x_bbbb_ovov);
        tracked_cudaFree(d_y_bbbb_ovov);
    }

    double h_energy_3h3p = 0.0;
    double h_energy_4h2p_2h4p = 0.0;
    cudaMemcpy(&h_energy_3h3p, d_energy_3h3p, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_energy_4h2p_2h4p, d_energy_4h2p_2h4p, sizeof(double), cudaMemcpyDeviceToHost);

    cublasDestroy(cublasH);

    tracked_cudaFree(d_g_full);
    tracked_cudaFree(d_eri_tmp);
    tracked_cudaFree(d_energy_4h2p_2h4p);
    tracked_cudaFree(d_energy_3h3p);
    tracked_cudaFree(d_tmp_same_1);
    tracked_cudaFree(d_tmp_same_2);
    tracked_cudaFree(d_tmp_same_3);
    tracked_cudaFree(d_tmp_mixed_1);
    tracked_cudaFree(d_tmp_mixed_2);
    tracked_cudaFree(d_tmp_mixed_3);
    tracked_cudaFree(d_x_aabb_ovov);
    tracked_cudaFree(d_g_aabb_ovvo);
    tracked_cudaFree(d_x_bbaa_ovov);

    return h_energy_4h2p_2h4p + h_energy_3h3p;
}











real_t ERI_Stored_UHF::compute_mp3_energy() 
{
    PROFILE_FUNCTION();

    const int num_basis = uhf_.get_num_basis();
    const int num_occ_al = uhf_.get_num_alpha_spins();
    const int num_occ_be = uhf_.get_num_beta_spins();

    DeviceHostMatrix<real_t>& coefficient_matrix_al = uhf_.get_coefficient_matrix_a();
    DeviceHostMatrix<real_t>& coefficient_matrix_be = uhf_.get_coefficient_matrix_b();
    DeviceHostMemory<real_t>& orbital_energies_al = uhf_.get_orbital_energies_a();
    DeviceHostMemory<real_t>& orbital_energies_be = uhf_.get_orbital_energies_b();

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

    //const real_t E_UMP3 = ump3_from_aoeri_via_full_moeri(
    //const real_t E_UMP3 = ump3_from_aoeri_via_full_moeri_dgemm(
    const real_t E_UMP3 = ump3_from_aoeri_via_full_moeri_dgemm_eff(
        eri_matrix_.device_ptr(), 
        coefficient_matrix_al.device_ptr(), 
        coefficient_matrix_be.device_ptr(), 
        orbital_energies_al.device_ptr(),
        orbital_energies_be.device_ptr(),
        num_basis, 
        num_occ_al, 
        num_occ_be
    );

    std::cout << "UMP3 energy: " << E_UMP3 << " [hartree]" << std::endl;

    return E_UMP2 + E_UMP3;
}



























}   // namespace gansu
