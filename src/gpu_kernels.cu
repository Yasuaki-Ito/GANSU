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


#include "gpu_kernels.hpp"
#include "gpu_hash_table.cuh"
#include "utils.hpp"

namespace gansu::gpu{


/**
 * @brief CUDA kernel for inverse of square root for individual values of input vectors
 * @param d_eigenvalues Device pointer storing the eigenvalues as a vector
 * @param size Size of the input vector
 * @details This function computes the inverse of the square root of each element of the input vector.
 *         The input vector is modified in place.
 */
 __global__ void inverseSqrt_kernel(real_t* d_eigenvalues, const size_t size, const double threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double value = d_eigenvalues[idx];
        if (value < threshold) {
            d_eigenvalues[idx] = 0.0; // Avoid division by zero
        }else{
            d_eigenvalues[idx] = 1.0 / __dsqrt_rn(value);
        }
    }
}

/**
 * @brief CUDA kernel for square root for individual values of input vectors
 * @param d_eigenvalues Device pointer storing the eigenvalues as a vector
 * @param size Size of the input vector
 * @param threshold Threshold value to avoid negative square roots or zero
 * @details This function computes the square root of each element of the input vector.
 *         The input vector is modified in place.
 */
 __global__ void sqrt_kernel(real_t* d_eigenvalues, const size_t size, const double threshold) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if(d_eigenvalues[idx] < threshold){
            d_eigenvalues[idx] = threshold; // Avoid negative square roots or zero
        }
        d_eigenvalues[idx] = __dsqrt_rn(d_eigenvalues[idx]);
    }
}



/**
 * @brief CUDA kernel for computing the density matrix for restricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix Device pointer to the density matrix, each of orbital elements has exactly 2 electrons
 * @param num_electron Number of electrons, must be even
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = 2 \sum_{k=1}^{N/2} C_{ik} C_{jk} \f$.
 */
 __global__ void computeDensityMatrix_RHF_kernel(const real_t* d_coefficient_matrix, real_t* d_density_matrix, const int num_electron, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_electron / 2; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = 2.0 * sum;
}

/**
 * @brief CUDA kernel for computing the density matrix for unrestricted Hartree-Fock
 * @param d_coefficient_matrix Device pointer to the coefficient matrix (alpha or beta)
 * @param d_density_matrix Device pointer to the density matrix (alpha or beta)
 * @param num_spin Number of electrons, must be number of electrons for the alpha or beta spin
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 * @details The density matrix is given by \f$ D_{ij} = \sum_{k=1}^{N} C_{ik} C_{jk} \f$.
 */
 __global__ void computeDensityMatrix_UHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix, const int num_spin, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum = 0.0;
    for (size_t k = 0; k < num_spin; k++) {
        sum += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    d_density_matrix[id] = sum;
}



/**
 * @brief CUDA kernel for computing the density matrix for ROHF
 * @param d_coefficient_matrix Device pointer to the coefficient matrix
 * @param d_density_matrix_closed Device pointer to the density matrix (closed-shell)
 * @param d_density_matrix_oepn Device pointer to the density matrix (open-shell)
 * @param d_density_matrix Device pointer to the density matrix (sum of closed-shell and open-shell)
 * @param num_closed Number of closed-shell orbitals
 * @param num_open Number of open-shell orbitals
 * @param num_basis Number of basis functions
 * @details This function computes the density matrix using the coefficient matrix.
 */
 __global__ void computeDensityMatrix_ROHF_kernel(const double* d_coefficient_matrix, double* d_density_matrix_closed, double* d_density_matrix_open, double* d_density_matrix, const int num_closed, const int num_open, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t i = id / num_basis;
    size_t j = id % num_basis;

    real_t sum_closed = 0.0;
    for (size_t k = 0; k < num_closed; k++) {
        sum_closed += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_closed *= 2.0; // closed shell (2 electrons per orbital)
    d_density_matrix_closed[id] = sum_closed; 

    real_t sum_open = 0.0;
    for (size_t k = num_closed; k < num_closed+num_open; k++) {
        sum_open += d_coefficient_matrix[i * num_basis + k] * d_coefficient_matrix[j * num_basis + k];
    }
    sum_open *= 1.0; // open shell (1 electron per orbital)
    d_density_matrix_open[id] = sum_open;

    d_density_matrix[id] = sum_closed + sum_open;
}



/**
 * @brief transposeMatrixInPlace_kernel CUDA kernel for transposing a matrix in place
 * @param d_matrix Device pointer to the matrix
 * @param size Size of the matrix
 */
__global__ void transposeMatrixInPlace_kernel(real_t* d_matrix, int size)
{
    if (blockIdx.x < blockIdx.y) {
        return;
    }
    const int xid = blockDim.x * blockIdx.x + threadIdx.x;
    const int yid = blockDim.y * blockIdx.y + threadIdx.y;
    if (xid < yid || xid >= size || yid >= size) {
        return;
    }

    //__shared__ real_t s_src[WARP_SIZE][WARP_SIZE];
    //__shared__ real_t s_dst[WARP_SIZE][WARP_SIZE];
    __shared__ real_t s_src[WARP_SIZE][WARP_SIZE + 1];
    __shared__ real_t s_dst[WARP_SIZE][WARP_SIZE + 1];
    s_src[threadIdx.y][threadIdx.x] = d_matrix[size * yid + xid];
    s_dst[threadIdx.y][threadIdx.x] = d_matrix[size * xid + yid];

    __syncthreads();

    d_matrix[size * yid + xid] = s_dst[threadIdx.y][threadIdx.x];
    d_matrix[size * xid + yid] = s_src[threadIdx.y][threadIdx.x];
}

/**
 * @brief CUDA kernel for computing weight sum matices sum(W[i] * B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param d_W Scalars (size N).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
__global__ void weighted_sum_matrices_kernel(double* d_J, const double* d_B, const double* d_W, const int M, const int N, const bool accumulated) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= M * M) return;

    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
        sum += d_W[j] * d_B[(size_t)j * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_J[id] += sum;
    }else{
        d_J[id] = sum;
    }
}


/**
 * @brief CUDA kernel for computing sum matices sum(B[i]).
 *
 * @param d_J Output result matrix (MxM) in device memory.
 * @param d_B Input matrices (N matrices of size MxM).
 * @param M Dimension of matrices (M x M).
 * @param N Number of matrices.
 * @param accumulated If true, the result is accumulated to the output matrix.
 */
__global__ void sum_matrices_kernel(double* d_K, const double* d_B, const int M, const int N, const bool accumulated) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= M * M) return;

    double sum = 0.0;
    for (int p = 0; p < N; p++) {
        //sum += d_B[p * M * M + id];  // Apply scalar multiplication and accumulate
        sum += d_B[(size_t)p * M * M + id];  // Apply scalar multiplication and accumulate
    }

    if(accumulated){
        d_K[id] += sum;
    }else{
        d_K[id] = sum;
    }
}


__global__ void computeFockMatrix_RHF_kernel(const real_t* d_density_matrix, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_F_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_F_ij[0] = 0.0;
    }
    __syncthreads();

    real_t sum = 0.0;
    size_t eid1, eid2;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            //eid2 = get_1d_indexM4(i, l, k, j, num_basis);
            eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum += (d_eri[eid1] - 0.5 * d_eri[eid2]) * d_density_matrix[num_basis * k + l];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_F_ij, sum);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix[bra] = s_F_ij[0] + d_core_hamiltonian_matrix[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + d_core_hamiltonian_matrix[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}


__global__ void computeFockMatrix_UHF_kernel(const real_t* d_density_matrix_a, const real_t* d_density_matrix_b, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_a, real_t* d_fock_matrix_b, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_Fa_ij[1];
    __shared__ real_t s_Fb_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_Fa_ij[0] = 0.0;
        s_Fb_ij[0] = 0.0;
    }
    __syncthreads();

    real_t sum_a = 0.0;
    real_t sum_b = 0.0;
    size_t eid1, eid2;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            eid1 = get_1d_indexM4(i, j, k, l, num_basis);
            //eid2 = get_1d_indexM4(i, l, k, j, num_basis);
            eid2 = get_1d_indexM4(i, k, j, l, num_basis);
            sum_a += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_a[num_basis * k + l] * d_eri[eid2];
            sum_b += (d_density_matrix_a[num_basis * k + l]+d_density_matrix_b[num_basis * k + l]) * d_eri[eid1] - d_density_matrix_b[num_basis * k + l] * d_eri[eid2];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum_a += __shfl_down_sync(FULL_MASK, sum_a, offset);
        sum_b += __shfl_down_sync(FULL_MASK, sum_b, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_Fa_ij, sum_a);
        atomicAdd(s_Fb_ij, sum_b);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix_a[bra] = s_Fa_ij[0] + d_core_hamiltonian_matrix[bra];
        d_fock_matrix_b[bra] = s_Fb_ij[0] + d_core_hamiltonian_matrix[bra];
        //g_fock[uid] = g_fock[lid] = s_F_ij[0] + d_core_hamiltonian_matrix[uid];   // 2-fold symmetry
        //g_fock[bra] = s_F_ij[0];  // use cuBLAS
    }
}



__global__ void computeFockMatrix_ROHF_kernel(const real_t* d_density_matrix_closed, const real_t* d_density_matrix_open, const real_t* d_core_hamiltonian_matrix, const real_t* d_eri, real_t* d_fock_matrix_closed, real_t* d_fock_matrix_open, int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    // 2-fold symmetry (vertical)
    /*
    const short j = __real_t2int_rn((__dsqrt_rn(8 * bra + 1) - 1) / 2);
    const short i = bra - j * (j + 1) / 2;
    const int uid = num_basis * i + j;
    const int lid = num_basis * j + i;
    */

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_J_closed_ij[1];
    __shared__ real_t s_J_open_ij[1];
    __shared__ real_t s_K_closed_ij[1];
    __shared__ real_t s_K_open_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_J_closed_ij[0] = 0.0;
        s_J_open_ij[0] = 0.0;
        s_K_closed_ij[0] = 0.0;
        s_K_open_ij[0] = 0.0;
    }
    __syncthreads();

    real_t J_closed = 0.0;
    real_t J_open = 0.0;
    real_t K_closed = 0.0;
    real_t K_open = 0.0;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            const real_t eri_ijkl = d_eri[get_1d_indexM4(i, j, k, l, num_basis)];
            const real_t eri_ikjl = d_eri[get_1d_indexM4(i, k, j, l, num_basis)];
            J_closed += d_density_matrix_closed[num_basis * k + l] * eri_ijkl;
            J_open   += d_density_matrix_open  [num_basis * k + l] * eri_ijkl;
            K_closed += d_density_matrix_closed[num_basis * k + l] * eri_ikjl;
            K_open   += d_density_matrix_open  [num_basis * k + l] * eri_ikjl;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        J_closed += __shfl_down_sync(FULL_MASK, J_closed, offset);
        J_open  += __shfl_down_sync(FULL_MASK, J_open,  offset);
        K_closed += __shfl_down_sync(FULL_MASK, K_closed, offset);
        K_open  += __shfl_down_sync(FULL_MASK, K_open,  offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_J_closed_ij, J_closed);
        atomicAdd(s_J_open_ij, J_open);
        atomicAdd(s_K_closed_ij, K_closed);
        atomicAdd(s_K_open_ij, K_open);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix_closed[bra] = d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - 0.5 * s_K_open_ij[0];
        d_fock_matrix_open[bra]  = 0.5 * (d_core_hamiltonian_matrix[bra] + s_J_closed_ij[0] - 0.5 * s_K_closed_ij[0] + s_J_open_ij[0] - s_K_open_ij[0]);
    }
}


__global__ void computeUnifiedFockMatrix_ROHF_kernel(const real_t* d_fock_mo_closed_matrix, const real_t* d_fock_mo_open_matrix, const ROHF_ParameterSet rohf_params, real_t* d_unified_fock_matrix, const int num_closed, const int num_open, const size_t num_basis) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * (num_basis+1) / 2) return;

    const size_t2 ij = index1to2(id, true);
    size_t i,j;
    if(ij.x < ij.y){
        i = ij.x;
        j = ij.y;
    }else{
        i = ij.y;
        j = ij.x;
    }

    enum SHELL_TYPE {CLOSED, OPEN, VIRTUAL};
    SHELL_TYPE shell_i, shell_j;
    if(i < num_closed) shell_i = CLOSED;
    else if(i < num_closed+num_open) shell_i = OPEN;
    else shell_i = VIRTUAL;
    if(j < num_closed) shell_j = CLOSED;
    else if(j < num_closed+num_open) shell_j = OPEN;
    else shell_j = VIRTUAL;

    const auto Acc = rohf_params.Acc;
    const auto Bcc = rohf_params.Bcc;
    const auto Aoo = rohf_params.Aoo;
    const auto Boo = rohf_params.Boo;
    const auto Avv = rohf_params.Avv;
    const auto Bvv = rohf_params.Bvv;

    real_t d = 0.0;

    if(shell_i == CLOSED && shell_j == CLOSED){ // closed-closed
        d = 2.0 * (Acc*d_fock_mo_open_matrix[i*num_basis+j] + Bcc*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == CLOSED && shell_j == OPEN){ // closed-open
        d = 2.0 * (d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]);
    }else if(shell_i == CLOSED && shell_j == VIRTUAL){ // closed-virtual
        d = d_fock_mo_closed_matrix[i*num_basis+j];
    }else if(shell_i == OPEN && shell_j == OPEN){ // open-open
        d = 2.0 * (Aoo*d_fock_mo_open_matrix[i*num_basis+j] + Boo*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }else if(shell_i == OPEN && shell_j == VIRTUAL){ // open-virtual
        d = 2.0 * d_fock_mo_open_matrix[i*num_basis+j];
    }else if(shell_i == VIRTUAL && shell_j == VIRTUAL){ // virtual-virtual
        d = 2.0 * (Avv*d_fock_mo_open_matrix[i*num_basis+j] + Bvv*(d_fock_mo_closed_matrix[i*num_basis+j] - d_fock_mo_open_matrix[i*num_basis+j]));
    }

    // 2-fold symmetry
    d_unified_fock_matrix[i*num_basis+j] = d;
    if(i != j) d_unified_fock_matrix[j*num_basis+i] = d;
}



/**
 * @brief CUDA kernel for computing the trace of a matrix
 * @param d_matrix Device pointer to the matrix
 * @param d_trace Device pointer to the trace
 * @param num_basis Number of basis functions
 * @details This function computes the trace of a matrix.
 */
__global__ void getMatrixTrace(const double* g_matrix, double* g_trace, const int num_basis)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double s_trace;
    if (threadIdx.x == 0) {
        s_trace = 0.0;
    }
    __syncthreads();

    double val = 0.0;
    if (tid < num_basis) {
        val = g_matrix[num_basis * tid + tid];
    }

    atomicAdd(&s_trace, val);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(g_trace, s_trace);
    }
}

/**
 * @brief CUDA kernel for computing the initial Fock matrix in GWH method
 * @param d_core_hamiltonian_matrix Device pointer to the core Hamiltonian matrix
 * @param d_overlap_matrix Device pointer to the overlap matrix
 * @param d_fock_matrix Device pointer to the initial Fock matrix
 * @param num_basis Number of basis functions
 * @param c_x Constant c_x
 */
__global__ void computeInitialFockMatrix_GWH_kernel(const double* d_core_hamiltonian_matrix, const double* d_overlap_matrix, double* d_fock_matrix, const int num_basis, const double c_x) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_basis * num_basis) return;

    size_t p = id / num_basis;
    size_t q = id % num_basis;

    d_fock_matrix[id] = c_x * d_overlap_matrix[id] * (d_core_hamiltonian_matrix[p*num_basis+p] + d_core_hamiltonian_matrix[q*num_basis+q]) / 2.0;
}


__global__ void computeRIIntermediateMatrixB_kernel(const double* d_three_center_eri, const double* d_matrix_L, double* d_matrix_B, const int num_basis, const int num_auxiliary_basis){
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_auxiliary_basis * num_basis * num_basis) return;

    const size_t p = id / (num_basis*num_basis);
    const size_t id2 = (id % (num_basis*num_basis)) ;
    const size_t mu = id2 / num_basis;
    const size_t nu = id2 % num_basis;

    real_t sum = 0.0;
    for (int q = 0; q < num_auxiliary_basis; q++) {
        sum += d_three_center_eri[q*num_basis*num_basis + mu*num_basis + nu] * d_matrix_L[q*num_auxiliary_basis + p];
    }
    d_matrix_B[id] = sum;
}



__global__ void computeFockMatrix_RI_RHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix[id];
}


__global__ void computeFockMatrix_RI_UHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix, double* d_Fock_matrix, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix[id];
}

__global__ void computeFockMatrix_RI_ROHF_kernel(const double* d_core_hamiltonian_matrix, const double* d_J_matrix, const double* d_K_matrix_closed, const double* d_K_matrix_open, double* d_Fock_matrix_closed, double* d_Fock_matrix_open, const int num_basis) {
    size_t id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= num_basis * num_basis) return;

    d_Fock_matrix_closed[id] = d_core_hamiltonian_matrix[id] + d_J_matrix[id] - 0.5*d_K_matrix_closed[id];
    d_Fock_matrix_open[id] = 0.5 * (d_core_hamiltonian_matrix[id] + d_J_matrix[id] - d_K_matrix_open[id]);
}

/*
 * @brief Sets zeros to the upper triangular part of the matrix
 *
 * @param d_A Pointer to the N x N matrix in device memory (input/output).
 * @param N The size of the matrix (number of rows/columns).
 */
 __global__ void setZeroUpperTriangle(double* d_A, const int N) {
    const size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = id / N;
    const size_t col = id % N;
    if (row < N && col < N && col > row) {
        d_A[row * N + col] = 0.0;
    }
}


/**
 * @brief CUDA kernel for computing the diagonal of the product of two matrices A and B
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
__global__ void compute_diagonal_of_product(const double* A, const double* B, double* diag, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        double sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[i * N + k] * B[k * N + i];  // Diagonal element of the product matrix stored in row-major order
        }
        diag[i] = sum;
    }
}


/**
 * @brief CUDA kernel for computing the diagonal of the sum of two matrices A and B, multiplied by a third matrix C
 * @param A Device pointer to the first matrix (row-major)
 * @param B Device pointer to the second matrix (row-major)
 * @param C Device pointer to the third matrix (row-major)
 * @param diag Device pointer to the output diagonal vector
 * @param N Size of the matrices (N x N)
 */
__global__ void compute_diagonal_of_product_sum(const double* A, const double* B, const double* C, double* diag, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        double a_plus_b = A[i * N + k] + B[i * N + k]; // (A + B)[i][k]
        double c = C[k * N + i];                       // C[k][i] (row-major)
        sum += a_plus_b * c;
    }
    diag[i] = sum;
}



/**
 * @brief Kernel to copy non-zero ERI values from dense N^4 array to hash table
 * @details Each thread handles one symmetry-unique (i,j,k,l) index combination.
 *          Indices are enumerated as: i<=j, k<=l, (i,j)<=(k,l).
 */
__global__ void denseToHash_kernel(const real_t* d_dense_eri,
    unsigned long long* d_hash_keys, real_t* d_hash_values,
    size_t hash_capacity_mask, const int num_basis)
{
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t N = num_basis;
    const size_t num_pairs = N * (N + 1) / 2;
    const size_t num_unique = num_pairs * (num_pairs + 1) / 2;
    if (tid >= num_unique) return;

    // Convert linear index to (bra_idx, ket_idx) where bra_idx <= ket_idx
    // Using triangular index formula: ket_idx = floor((sqrt(8*tid+1)-1)/2)
    const size_t ket_idx = static_cast<size_t>(__double2ll_rd((__dsqrt_rn(8.0 * tid + 1.0) - 1.0) / 2.0));
    const size_t bra_idx = tid - ket_idx * (ket_idx + 1) / 2;

    // Convert pair indices to (i,j) and (k,l) where i<=j, k<=l
    const size_t j = static_cast<size_t>(__double2ll_rd((__dsqrt_rn(8.0 * bra_idx + 1.0) - 1.0) / 2.0));
    const size_t i = bra_idx - j * (j + 1) / 2;
    const size_t l = static_cast<size_t>(__double2ll_rd((__dsqrt_rn(8.0 * ket_idx + 1.0) - 1.0) / 2.0));
    const size_t k = ket_idx - l * (l + 1) / 2;

    // Read ERI value from dense array
    const size_t dense_idx = get_1d_indexM4(i, j, k, l, N);
    const real_t value = d_dense_eri[dense_idx];

    // Only insert non-zero values
    if (value != 0.0) {
        const unsigned long long key = canonical_eri_key(i, j, k, l);
        hash_insert(d_hash_keys, d_hash_values, hash_capacity_mask, key, value);
    }
}

/**
 * @brief Compute Fock matrix for RHF using hash-stored ERIs
 * @details Same structure as computeFockMatrix_RHF_kernel but uses hash_lookup
 *          instead of direct dense array access.
 *          F_ij = H_ij + sum_{kl} D_kl * ((ij|kl) - 0.5*(ik|jl))
 */
__global__ void computeFockMatrix_Hash_RHF_kernel(const real_t* d_density_matrix,
    const real_t* d_core_hamiltonian_matrix,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    size_t hash_capacity_mask,
    real_t* d_fock_matrix, const int num_basis)
{
    const int bra = blockIdx.x;
    const int i = bra / num_basis;
    const int j = bra % num_basis;

    const size_t l = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ real_t s_F_ij[1];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_F_ij[0] = 0.0;
    }
    __syncthreads();

    real_t sum = 0.0;
    if (l < num_basis) {
        for (int k = 0; k < num_basis; ++k) {
            const real_t eri_ijkl = hash_lookup(d_hash_keys, d_hash_values, hash_capacity_mask,
                                                 canonical_eri_key(i, j, k, l));
            const real_t eri_ikjl = hash_lookup(d_hash_keys, d_hash_values, hash_capacity_mask,
                                                 canonical_eri_key(i, k, j, l));
            sum += (eri_ijkl - 0.5 * eri_ikjl) * d_density_matrix[num_basis * k + l];
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(s_F_ij, sum);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_fock_matrix[bra] = s_F_ij[0] + d_core_hamiltonian_matrix[bra];
    }
}



/**
 * @brief Push-based Fock matrix construction from hash-stored ERIs (RHF)
 * @details Each thread processes one hash table entry. For each non-zero ERI (a,b,c,d)=v,
 *          it adds Coulomb and Exchange contributions to the Fock matrix via atomicAdd.
 *          The 8-fold symmetry is fully expanded:
 *            Coulomb: F[a,b] += 2*D[c,d]*v, F[c,d] += 2*D[a,b]*v (if bra!=ket)
 *            Exchange: F[a,c] -= D[b,d]*v, F[a,d] -= D[b,c]*v (if c!=d),
 *                      F[b,c] -= D[a,d]*v (if a!=b), F[b,d] -= D[a,c]*v (if a!=b && c!=d)
 */
__global__ void computeFockMatrix_Hash_Push_RHF_kernel(
    const real_t* d_density_matrix,
    const unsigned long long* d_hash_keys,
    const real_t* d_hash_values,
    size_t hash_table_capacity,
    real_t* d_fock_matrix,
    const int num_basis,
    const int num_fock_replicas)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= hash_table_capacity) return;

    const unsigned long long key = d_hash_keys[tid];
    if (key == EMPTY_KEY) return;

    const real_t v = d_hash_values[tid];
    if (v == 0.0) return;

    int a, b, c, d;
    decode_eri_key(key, a, b, c, d);

    const int N = num_basis;

    const real_t* D = d_density_matrix;
    real_t* F = d_fock_matrix + N * N * (threadIdx.x % num_fock_replicas);

    const real_t D_cd = D[c * N + d];
    const real_t D_ab = D[a * N + b];
    const real_t D_bd = D[b * N + d];
    const real_t D_ad = D[a * N + d];
    const real_t D_bc = D[b * N + c];
    const real_t D_ac = D[a * N + c];
    const real_t hv = -0.5 * v;
    const bool braket_diff = (a != c || b != d);

    // --- Coulomb: enumerate unique (ij|kl) permutations ---
    // (a,b,c,d) and (a,b,d,c) → F[a,b] += (c==d ? 1 : 2) * D[c,d] * v
    atomicAdd(&F[a * N + b], (c != d ? 2.0 : 1.0) * D_cd * v);
    // (b,a,c,d) and (b,a,d,c) → F[b,a], if a != b
    if (a != b)
        atomicAdd(&F[b * N + a], (c != d ? 2.0 : 1.0) * D_cd * v);
    // (c,d,a,b) and (c,d,b,a) → F[c,d], if bra != ket
    if (braket_diff)
        atomicAdd(&F[c * N + d], (a != b ? 2.0 : 1.0) * D_ab * v);
    // (d,c,a,b) and (d,c,b,a) → F[d,c], if bra != ket && c != d
    if (braket_diff && c != d)
        atomicAdd(&F[d * N + c], (a != b ? 2.0 : 1.0) * D_ab * v);

    // --- Exchange: enumerate all 8 unique (ik|jl) permutations ---
    // #1: (ik|jl)=(a,b,c,d) → F[a,c] -= 0.5*D[b,d]*v
    atomicAdd(&F[a * N + c], hv * D_bd);
    // #2: (ik|jl)=(b,a,c,d) → F[b,c] -= 0.5*D[a,d]*v, if a!=b
    if (a != b)
        atomicAdd(&F[b * N + c], hv * D_ad);
    // #3: (ik|jl)=(a,b,d,c) → F[a,d] -= 0.5*D[b,c]*v, if c!=d
    if (c != d)
        atomicAdd(&F[a * N + d], hv * D_bc);
    // #4: (ik|jl)=(b,a,d,c) → F[b,d] -= 0.5*D[a,c]*v, if a!=b && c!=d
    if (a != b && c != d)
        atomicAdd(&F[b * N + d], hv * D_ac);
    // #5: (ik|jl)=(c,d,a,b) → F[c,a] -= 0.5*D[d,b]*v, if bra!=ket
    if (braket_diff)
        atomicAdd(&F[c * N + a], hv * D_bd);
    // #6: (ik|jl)=(d,c,a,b) → F[d,a] -= 0.5*D[c,b]*v, if bra!=ket && c!=d
    if (braket_diff && c != d)
        atomicAdd(&F[d * N + a], hv * D_bc);
    // #7: (ik|jl)=(c,d,b,a) → F[c,b] -= 0.5*D[d,a]*v, if bra!=ket && a!=b
    if (braket_diff && a != b)
        atomicAdd(&F[c * N + b], hv * D_ad);
    // #8: (ik|jl)=(d,c,b,a) → F[d,b] -= 0.5*D[c,a]*v, if bra!=ket && a!=b && c!=d
    if (braket_diff && a != b && c != d)
        atomicAdd(&F[d * N + b], hv * D_ac);
}

/**
 * @brief Push-based Fock matrix using pre-built non-zero index list.
 *        Each thread processes one non-zero hash entry (no empty slot waste).
 */
__global__ void computeFockMatrix_Hash_Push_Indexed_RHF_kernel(
    const real_t* d_density_matrix,
    const unsigned long long* d_hash_keys,
    const real_t* d_hash_values,
    const size_t* d_nonzero_indices,
    size_t num_nonzero,
    real_t* d_fock_matrix,
    const int num_basis,
    const int num_fock_replicas)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= num_nonzero) return;

    const size_t slot = d_nonzero_indices[tid];
    const real_t v = d_hash_values[slot];
    if (v == 0.0) return;

    int a, b, c, d;
    decode_eri_key(d_hash_keys[slot], a, b, c, d);

    const int N = num_basis;
    const real_t* D = d_density_matrix;
    real_t* F = d_fock_matrix + N * N * (threadIdx.x % num_fock_replicas);

    const real_t D_cd = D[c * N + d];
    const real_t D_ab = D[a * N + b];
    const real_t D_bd = D[b * N + d];
    const real_t D_ad = D[a * N + d];
    const real_t D_bc = D[b * N + c];
    const real_t D_ac = D[a * N + c];
    const real_t hv = -0.5 * v;
    const bool braket_diff = (a != c || b != d);

    // Coulomb
    atomicAdd(&F[a * N + b], (c != d ? 2.0 : 1.0) * D_cd * v);
    if (a != b)
        atomicAdd(&F[b * N + a], (c != d ? 2.0 : 1.0) * D_cd * v);
    if (braket_diff)
        atomicAdd(&F[c * N + d], (a != b ? 2.0 : 1.0) * D_ab * v);
    if (braket_diff && c != d)
        atomicAdd(&F[d * N + c], (a != b ? 2.0 : 1.0) * D_ab * v);

    // Exchange
    atomicAdd(&F[a * N + c], hv * D_bd);
    if (a != b)
        atomicAdd(&F[b * N + c], hv * D_ad);
    if (c != d)
        atomicAdd(&F[a * N + d], hv * D_bc);
    if (a != b && c != d)
        atomicAdd(&F[b * N + d], hv * D_ac);
    if (braket_diff)
        atomicAdd(&F[c * N + a], hv * D_bd);
    if (braket_diff && c != d)
        atomicAdd(&F[d * N + a], hv * D_bc);
    if (braket_diff && a != b)
        atomicAdd(&F[c * N + b], hv * D_ad);
    if (braket_diff && a != b && c != d)
        atomicAdd(&F[d * N + b], hv * D_ac);
}

/**
 * @brief Push-based Fock matrix from COO sparse ERIs.
 *        Each thread processes one COO entry directly (no index indirection).
 */
__global__ void computeFockMatrix_COO_Push_RHF_kernel(
    const real_t* d_density_matrix,
    const unsigned long long* d_coo_keys,
    const real_t* d_coo_values,
    size_t num_entries,
    real_t* d_fock_matrix,
    const int num_basis,
    const int num_fock_replicas)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= num_entries) return;

    const real_t v = d_coo_values[tid];
    if (v == 0.0) return;

    int a, b, c, d;
    decode_eri_key(d_coo_keys[tid], a, b, c, d);

    const int N = num_basis;
    const real_t* D = d_density_matrix;
    real_t* F = d_fock_matrix + N * N * (threadIdx.x % num_fock_replicas);

    const real_t D_cd = D[c * N + d];
    const real_t D_ab = D[a * N + b];
    const real_t D_bd = D[b * N + d];
    const real_t D_ad = D[a * N + d];
    const real_t D_bc = D[b * N + c];
    const real_t D_ac = D[a * N + c];
    const real_t hv = -0.5 * v;
    const bool braket_diff = (a != c || b != d);

    // Coulomb
    atomicAdd(&F[a * N + b], (c != d ? 2.0 : 1.0) * D_cd * v);
    if (a != b)
        atomicAdd(&F[b * N + a], (c != d ? 2.0 : 1.0) * D_cd * v);
    if (braket_diff)
        atomicAdd(&F[c * N + d], (a != b ? 2.0 : 1.0) * D_ab * v);
    if (braket_diff && c != d)
        atomicAdd(&F[d * N + c], (a != b ? 2.0 : 1.0) * D_ab * v);

    // Exchange
    atomicAdd(&F[a * N + c], hv * D_bd);
    if (a != b) atomicAdd(&F[b * N + c], hv * D_ad);
    if (c != d) atomicAdd(&F[a * N + d], hv * D_bc);
    if (a != b && c != d) atomicAdd(&F[b * N + d], hv * D_ac);
    if (braket_diff) atomicAdd(&F[c * N + a], hv * D_bd);
    if (braket_diff && c != d) atomicAdd(&F[d * N + a], hv * D_bc);
    if (braket_diff && a != b) atomicAdd(&F[c * N + b], hv * D_ad);
    if (braket_diff && a != b && c != d) atomicAdd(&F[d * N + b], hv * D_ac);
}

/**
 * @brief Add core Hamiltonian to Fock matrix: F[i] += H[i]
 */
__global__ void addCoreHamiltonian_kernel(
    const real_t* d_core_hamiltonian_matrix,
    real_t* d_fock_matrix,
    const int size)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        d_fock_matrix[tid] += d_core_hamiltonian_matrix[tid];
    }
}

/**
 * @brief Remove hash entries with |value| < threshold.
 *        Resets key to EMPTY_KEY and value to 0.0 for negligible entries.
 */
__global__ void cleanupHashTable_kernel(
    unsigned long long* d_hash_keys,
    real_t* d_hash_values,
    size_t hash_table_capacity,
    real_t threshold)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= hash_table_capacity) return;
    if (d_hash_keys[tid] == EMPTY_KEY) return;

    if (fabs(d_hash_values[tid]) < threshold) {
        d_hash_keys[tid] = EMPTY_KEY;
        d_hash_values[tid] = 0.0;
    }
}

__global__ void countNonEmptySlots_kernel(
    const unsigned long long* d_hash_keys,
    size_t hash_table_capacity,
    size_t* d_count)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= hash_table_capacity) return;
    if (d_hash_keys[tid] != EMPTY_KEY) {
        atomicAdd((unsigned long long*)d_count, 1ULL);
    }
}

__global__ void collectNonEmptyIndices_kernel(
    const unsigned long long* d_hash_keys,
    size_t hash_table_capacity,
    size_t* d_indices,
    size_t* d_write_pos)
{
    const size_t tid = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (tid >= hash_table_capacity) return;
    if (d_hash_keys[tid] != EMPTY_KEY) {
        size_t pos = atomicAdd((unsigned long long*)d_write_pos, 1ULL);
        d_indices[pos] = tid;
    }
}

} // namespace gansu::gpu