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

#include <algorithm>
#include <memory>

#include <cstdlib>  // std::getenv
#include <cstdio>   // std::printf
#include <cmath>
#include <vector>
#include <string>   // std::string
#include <fstream>
#include <iomanip>
#include <chrono>

//#include <omp.h>


#include "rhf.hpp"
#include "device_host_memory.hpp"
#include "int3c2e.hpp"
#include "laplace_quadrature.hpp"
#include "ri_adc2_schur_operator.hpp"
#include "sos_laplace_adc2_operator.hpp"
#include "adc2_operator.hpp"
#include "davidson_solver.hpp"
#include "oscillator_strength.hpp"
#include "profiler.hpp"
#include "thc_grid.hpp"
#include "thc_collocation.hpp"
#include "thc_decomposition.hpp"
#include "thc_mp2.hpp"   // for transform_X_to_mo_gpu
#include "thc_sos_adc2_operator.hpp"
#include "multi_gpu_manager.hpp"

namespace gansu{

// Forward declarations of stored-path helpers defined in eri_stored.cu.
// These are used as CPU fallbacks by the RI-based post-HF methods.
double mp2_from_full_moeri(
    const double* d_eri_mo, const double* d_C, const double* d_eps,
    int nao, int occ, int frozen = 0);


// // #threads = M * Mvir * Maux
__global__
void nu2a_(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a)
{
    long long seq = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (seq >= (long long)norbs * nvir * naux) {
        return;
    }

    const int p = seq / (norbs * nvir);
    seq %= (norbs * nvir);

    const int a = seq % nvir;
    const int mu = seq / nvir;

    double tmp = 0.0;
    for (int nu = 0; nu < norbs; ++nu) {
        tmp += d_C[norbs * nu + (a + nocc)] * d_B_p_mu_nu[p*(norbs*norbs) + mu*norbs + nu];
    }
    d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a] = tmp;
}


// #threads = Mocc * Mvir * Maux
__global__
void mu2i_(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a)
{
    long long seq = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (seq >= (long long)nocc * nvir * naux) {
        return;
    }

    const int p = seq / (nocc * nvir);
    seq %= (nocc * nvir);

    const int a = seq % nvir;
    const int i = seq / nvir;

    double tmp = 0.0;
    for (int mu = 0; mu < norbs; ++mu) {
        tmp += d_C[norbs * mu + i] * d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a];
    }
    d_B_p_i_a[p*(nocc*nvir) + i*nvir + a] = tmp;
}


 void nu2a_dgemm(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a, cublasHandle_t &handle){
    // cublasManager cublas;
    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // if(col_A != row_B) throw exception("行数と列数が不一致\n");

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_mu_a, 0, norbs * (size_t)nvir * naux * sizeof(double));
	

    cublasDgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        nvir, naux * norbs, norbs, 
        &alpha, 
        &d_C[nocc], norbs, 
        d_B_p_mu_nu, norbs, 
        &beta, 
        d_B_p_mu_a, nvir
    );
    
    // cublasDestroy(handle);
}


void mu2i_dgemm(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a, cublasHandle_t &handle){
    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_i_a, 0, norbs * (size_t)nvir * naux * sizeof(double));

    int row = naux * norbs, col = nvir;
    cublasDgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        row, col,
        &alpha,
        d_B_p_mu_a, col,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B_p_i_a, row
    );

    // beta=0 in DGEMM fully overwrites output; no memset needed
    // (previous memset had buffer overflow: norbs*norbs*naux > allocated norbs*nvir*naux)

    cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nocc, naux * nvir, norbs,
        &alpha,
        d_C, norbs,
        d_B_p_i_a, norbs,
        &beta,
        d_B_p_mu_a, nocc
    );

    row = naux * nocc, col = nvir;
    cublasDgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        col, row,
        &alpha,
        d_B_p_mu_a, row,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B_p_i_a, col
    );

    // cublasDestroy(handle);
}




void transform_intermediate_matrix(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B, double* d_tmp, cublasHandle_t &handle){
    nu2a_dgemm(norbs, nocc, nvir, naux, d_C, d_B, d_tmp, handle);
    mu2i_dgemm(norbs, nocc, nvir, naux, d_C, d_tmp, d_B, handle);
}






 void nu2a_dgemm(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a){
    // cublasManager cublas;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // if(col_A != row_B) throw exception("行数と列数が不一致\n");

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_mu_a, 0, norbs * (size_t)nvir * naux * sizeof(double));
	

    cublasDgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        nvir, naux * norbs, norbs, 
        &alpha, 
        &d_C[nocc], norbs, 
        d_B_p_mu_nu, norbs, 
        &beta, 
        d_B_p_mu_a, nvir
    );
    
    cublasDestroy(handle);
}


void mu2i_dgemm(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a){
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_i_a, 0, norbs * (size_t)nvir * naux * sizeof(double));

    int row = naux * norbs, col = nvir;
    cublasDgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        row, col,
        &alpha,
        d_B_p_mu_a, col,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B_p_i_a, row
    );

    // beta=0 in DGEMM fully overwrites output; no memset needed
    // (previous memset had buffer overflow: norbs*norbs*naux > allocated norbs*nvir*naux)

    cublasDgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nocc, naux * nvir, norbs,
        &alpha,
        d_C, norbs,
        d_B_p_i_a, norbs,
        &beta,
        d_B_p_mu_a, nocc
    );

    row = naux * nocc, col = nvir;
    cublasDgeam(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        col, row,
        &alpha,
        d_B_p_mu_a, row,
        &beta,
        nullptr, (row >= col) ? row : col,
        d_B_p_i_a, col
    );

    cublasDestroy(handle);
}




void transform_intermediate_matrix(int norbs, int nocc, int nvir, int naux, double* d_C, double* d_B, double* d_tmp){
    nu2a_dgemm(norbs, nocc, nvir, naux, d_C, d_B, d_tmp);
    mu2i_dgemm(norbs, nocc, nvir, naux, d_C, d_tmp, d_B);
}








__device__ inline size_t2 index1to2_upper_wo_trace(const uint64_t index, const int n){
    size_t r2 = (2.0*n - 1.0 - sqrt((2.0*n - 1.0)*(2.0*n - 1.0) - 8.0*index)) / 2.0;
    size_t r1 = (r2*r2 - (2.0*n - 3.0)*r2 + 2.0*index) / 2.0 + 1.0;

    return {r2,r1};
}


/*
__device__ 
inline uint64_t calc_i(uint64_t id, int s, int k) {   // s:nocc_stride, k: nocc_block
    return ((uint64_t)1.0 + sqrt(1.0 + 4.0*(2.0*id + s*(s-1)))) / 2.0;
}


__device__ 
inline uint64_t calc_exclusive_prefix_num_j(int i, int s){
    return (i*(i-1) - s*(s-1)) / 2.0;
}

__device__
inline uint64_t calc_j(uint64_t id, int i, int s){
    return id - calc_exclusive_prefix_num_j(i,s);
}
/**/

__device__ 
inline uint64_t calc_i(uint64_t id, int s, int k) {   // s:nocc_stride, k: nocc_block
    return ((uint64_t)1.0 + sqrt(1.0 + 4.0*(2.0*id + (size_t)s*(s-1)))) / 2.0;
}


__device__ 
inline uint64_t calc_exclusive_prefix_num_j(int i, int s){
    return ((size_t)i*(i-1) - (size_t)s*(s-1)) / 2.0;
}

__device__
inline uint64_t calc_j(uint64_t id, int i, int s){
    return id - calc_exclusive_prefix_num_j(i,s);
}



// i>j, a>b
__global__
void calc_RI_RMP2_energy_kernel1(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy){
    __shared__ double sh_tmp[1];
    if(!threadIdx.x) sh_tmp[0] = 0.0;
    __syncthreads();

    uint64_t id = blockDim.x * (uint64_t)blockIdx.x + threadIdx.x;
    // NOTE: out-of-range threads must NOT `return` before the warp reduction —
    // __shfl_down_sync(0xffffffff,…) and __syncthreads() require every lane of
    // the block to participate.  Early-return left inactive lanes in the full
    // mask (undefined on Volta+), which corrupted the sum for some dimensions
    // (e.g. spherical nvir=19).  Instead, compute val (0 if out of range) and
    // let all threads reduce.
    double val = 0.0;
    if (id < (((uint64_t)nocc_block*nocc_stride + nocc_block*(nocc_block-1)/2) *  nvir*(nvir-1)/2)) {
        size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
        uint64_t idp = id / (nvir * (nvir-1) / 2);
        const size_t i = calc_i(idp, nocc_stride, nocc_block); // full-rangeの値 (stride~stride+k)
        const size_t j = calc_j(idp, i, nocc_stride);
        if (i < nocc) {
            double iajb = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + (size_t)ab.x*nocc*nvir + j*nvir + ab.y];
            double ibja = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + (size_t)ab.y*nocc*nvir + j*nvir + ab.x];
            val = 4.0 * ((iajb-ibja)*(iajb-ibja) + iajb*ibja) / (d_eps[i] + d_eps[j] - d_eps[ab.x+nocc] - d_eps[ab.y+nocc]);
        }
    }

    // warp-wide reduction
    val += __shfl_down_sync(0xffffffff, val, 16, 32);    
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);    
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);    

    // block-wide reduction
    if(!(threadIdx.x%32))
        atomicAdd(&sh_tmp[0], val);
    __syncthreads();

    // device-wide reduction
    if(!(threadIdx.x))
        atomicAdd(energy, sh_tmp[0]);
}





// i>j, a
__global__
void calc_RI_RMP2_energy_kernel2(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy){
    __shared__ double sh_tmp[1];

    if(!threadIdx.x) sh_tmp[0] = 0.0;
    __syncthreads();

    uint64_t id = blockDim.x * (uint64_t)blockIdx.x + threadIdx.x;
    // (see kernel1) all lanes must reach the reduction — no early return.
    // BUGFIX: the bound is EXCLUSIVE ( id < N_pairs·nvir ), matching kernels
    // 1/3/4.  The original `if (id > bound) return` (inclusive) processed one
    // extra thread (id == bound) that decodes to a NEXT-block (i,j) pair whose
    // (i a | j a) lives outside this block's d_iajb chunk → out-of-bounds read.
    // For Cartesian dims that OOB slot happened to read ~0 (benign); for
    // spherical (nvir=19) it read garbage, inflating E2 (-0.0022 → -0.9163).
    double val = 0.0;
    if (id < (((uint64_t)nocc_block*nocc_stride + nocc_block*(nocc_block-1)/2) *  nvir)) {
        const size_t a = id % nvir;
        uint64_t idp = id / nvir;
        const size_t i = calc_i(idp, nocc_stride, nocc_block);
        const size_t j = calc_j(idp, i, nocc_stride);
        if (i < nocc) {
            double iaja = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + a*(size_t)nocc*nvir + j*nvir + a];
            val = 2.0*iaja*iaja / (d_eps[i] + d_eps[j] - 2.0*d_eps[a+nocc]);
        }
    }

    // warp-wide reduction
    val += __shfl_down_sync(0xffffffff, val, 16, 32);    
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);    
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);    

    // block-wide reduction
    if(!(threadIdx.x%32))
        atomicAdd(&sh_tmp[0], val);
    __syncthreads();

    // device-wide reduction
    if(!(threadIdx.x))
        // atomicAdd(&energy[blockIdx.x % ARRAY_SIZE], sh_tmp[0]);
        atomicAdd(energy, sh_tmp[0]);
}




// i, a>b
__global__
void calc_RI_RMP2_energy_kernel3(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy){
    __shared__ double sh_tmp[1];

    if(!threadIdx.x) sh_tmp[0] = 0.0;
    __syncthreads();

    uint64_t id = blockDim.x * (uint64_t)blockIdx.x + threadIdx.x;
    // (see kernel1) all lanes must reach the reduction — no early return.
    double val = 0.0;
    if (id < (uint64_t)(nocc_block * nvir * (nvir - 1.0) / 2)) {
        size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
        const size_t a = ab.x, b = ab.y;
        const size_t i = id / (nvir * (nvir-1) / 2);  // このiは0~k-1
        if (i + nocc_stride < (size_t)nocc) {
            double iaib = d_iajb[i*nvir*(size_t)nocc*nvir + a*(size_t)nocc*nvir + (i+nocc_stride)*nvir + b];
            val = 2.0*iaib*iaib / (2.0*d_eps[i+nocc_stride] - d_eps[a+nocc] - d_eps[b+nocc]);
        }
    }

    // warp-wide reduction
    val += __shfl_down_sync(0xffffffff, val, 16, 32);    
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);    
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);    

    // block-wide reduction
    if(!(threadIdx.x%32))
        atomicAdd(&sh_tmp[0], val);
    __syncthreads();

    // device-wide reduction
    if(!(threadIdx.x))
        // atomicAdd(&energy[blockIdx.x % ARRAY_SIZE], sh_tmp[0]);
        atomicAdd(energy, sh_tmp[0]);
}


// i, a
__global__
void calc_RI_RMP2_energy_kernel4(int nocc, int nocc_block, int nvir, int nocc_stride, int naux, double* d_iajb, double* d_eps, double* energy){
    __shared__ double sh_tmp[1];

    if(!threadIdx.x) sh_tmp[0] = 0.0;
    __syncthreads();

    uint64_t id = blockDim.x * (uint64_t)blockIdx.x + threadIdx.x;
    // (see kernel1) all lanes must reach the reduction — no early return.
    double val = 0.0;
    if (id < (uint64_t)nocc_block * nvir) {
        const size_t a = id % nvir;
        const size_t i = id / nvir;  // このiは0~k-1
        if (i + nocc_stride < (size_t)nocc) {
            double iaia = d_iajb[i*nvir*(size_t)nocc*nvir + a*(size_t)nocc*nvir + (i+nocc_stride)*nvir + a];
            val = 0.5*iaia*iaia / (d_eps[i+nocc_stride] - d_eps[a+nocc]);
        }
    }

    // warp-wide reduction
    val += __shfl_down_sync(0xffffffff, val, 16, 32);
    val += __shfl_down_sync(0xffffffff, val, 8, 32);
    val += __shfl_down_sync(0xffffffff, val, 4, 32);    
    val += __shfl_down_sync(0xffffffff, val, 2, 32);
    val += __shfl_down_sync(0xffffffff, val, 1, 32);    

    // block-wide reduction
    if(!(threadIdx.x%32))
        atomicAdd(&sh_tmp[0], val);
    __syncthreads();

    // device-wide reduction
    if(!(threadIdx.x))
        // atomicAdd(&energy[blockIdx.x % ARRAY_SIZE], sh_tmp[0]);
        atomicAdd(energy, sh_tmp[0]);
}






/*
int search_maximum_k(int mocc, int mvir) {
    size_t free_mem_bytes, total_mem_bytes;
    cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes);
    
    return std::min(free_mem_bytes/(mocc * mvir * mvir * sizeof(double)), (size_t)mocc);    
}

void search_k_and_cudamalloc_4cERI(int mocc, int mvir, int &k, double **d_iajb, cudaStream_t &stream) {
    k = search_maximum_k(mocc, mvir);
    // k = (int)(k*mvir / 32) * 32;
    // k = 10;

    while(tracked_cudaMallocAsync((void**)d_iajb, sizeof(double) * k * mvir * mocc * mvir, stream) != cudaSuccess){
        k *= 0.9;
    }

    // printf("k = %d\n",k);
}
/**/


int search_maximum_k(int mocc, int mvir) {
    size_t free_mem_bytes, total_mem_bytes;
    cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes);
    
    return std::min(free_mem_bytes/(mocc * (size_t)mvir * mvir * sizeof(double)), (size_t)mocc);    
}

void search_k_and_cudamalloc_4cERI(int mocc, int mvir, int &k, double **d_iajb, cudaStream_t &stream) {
    k = search_maximum_k(mocc, mvir) * 0.9;
    // k = (int)(k*mvir / 32) * 32;
    // k = 10;

    //while(cudaMallocAsync((void**)d_iajb, sizeof(double) * k * mvir * mocc * mvir, stream) != cudaSuccess){
    //    k *= 0.9;
    //}

    cudaError_t err = tracked_cudaMalloc((void**)d_iajb, sizeof(double) * k * (size_t)mvir * mocc * mvir);
    if (err != cudaSuccess) {
        THROW_EXCEPTION(std::string("Failed to allocate device memory for d_iajb matrix: ") + std::string(cudaGetErrorString(err)));
    }

    printf("k = %d\n",k);
}




using RI_RMP2_energy_kernel_t = void(*)(int, int, int, int, int, double*, double*, double*);
struct KernelPair{
    RI_RMP2_energy_kernel_t kernel;
    size_t num_blocks;
};


real_t ERI_RI_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();


    real_t *d_C = coefficient_matrix.device_ptr();
    real_t *d_eps = orbital_energies.device_ptr();
    const int num_auxiliary_basis = num_auxiliary_basis_;

    // === CPU fallback: build full MO ERI from B and use stored MP2 ===
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(d_C, num_basis_);
        real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc);
        tracked_cudaFree(d_mo_eri);
        return E;
    }

    // Copy B matrix to avoid in-place destruction of intermediate_matrix_B_
    const size_t B_size = (size_t)num_auxiliary_basis * num_basis_ * num_basis_;
    real_t* d_B_copy = nullptr;
    tracked_cudaMalloc(&d_B_copy, B_size * sizeof(real_t));
    cudaMemcpy(d_B_copy, intermediate_matrix_B_.device_ptr(), B_size * sizeof(real_t), cudaMemcpyDeviceToDevice);
    real_t *d_intermediate_matrix_B = d_B_copy;

    real_t* d_tmp;
    tracked_cudaMalloc((void**)&d_tmp, sizeof(double) * num_basis_ * (size_t)nvir * num_auxiliary_basis);

    double *d_energy;
    tracked_cudaMalloc((void**)&d_energy, sizeof(double));
    cudaMemset(d_energy, 0, sizeof(double));

    const int num_threads = 1024;
    cudaEvent_t events[2];
    for (int i=0; i<2; i++) cudaEventCreate(&events[i]);

    cudaStream_t streams[4];
    for(int i=0; i<4; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);


    double *d_iajb = nullptr;
    int nocc_block;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, streams[0]);




    search_k_and_cudamalloc_4cERI(nocc, nvir, nocc_block, &d_iajb, streams[0]);

    transform_intermediate_matrix(num_basis_, nocc, nvir, num_auxiliary_basis, d_C, d_intermediate_matrix_B, d_tmp);
    tracked_cudaFree(d_tmp);


    size_t num_blocks_3 = ((size_t)(nocc_block * (size_t)nvir * (nvir - 1.0) / 2) + num_threads - 1) / num_threads, 
           num_blocks_4 = ((size_t)(nocc_block * (size_t)nvir) + num_threads - 1) / num_threads;


    // int nocc_block =  std::stoi(std::getenv("NUM"));



    int niter = ((double)nocc + nocc_block - 1) / nocc_block;
    cudaEvent_t *events_for_sync = new cudaEvent_t[niter * 4];
    for(int i=0; i<niter*4; i++) cudaEventCreate(&events_for_sync[i]);



    std::vector<KernelPair> num_blocks_list{{calc_RI_RMP2_energy_kernel1, 0},
                                            {calc_RI_RMP2_energy_kernel2, 0},
                                            {calc_RI_RMP2_energy_kernel3, num_blocks_3},
                                            {calc_RI_RMP2_energy_kernel4, num_blocks_4}};


    


    int iter_count = 0;
    const double alpha = 1.0;
    const double beta = 0.0;


    cudaDeviceSynchronize();

    cudaEventRecord(events[0], streams[0]);
    for(int i = 0; i < nocc; i+=nocc_block){  // iは資料でいう「stride」
        num_blocks_list[0].num_blocks = (((size_t)(nocc_block*i + (size_t)nocc_block*(nocc_block-1)/2) *  nvir*(nvir-1)/2) + num_threads -1) / num_threads;
        num_blocks_list[1].num_blocks = (((size_t)(nocc_block*i + (size_t)nocc_block*(nocc_block-1)/2) *  nvir) + num_threads - 1) / num_threads;

        cublasDgemm(
            handle, 
            CUBLAS_OP_N, CUBLAS_OP_T, 
            nocc * nvir,  
            ((nocc_block < nocc - i) ? nocc_block : nocc - i) * nvir ,  
            num_auxiliary_basis,  
            &alpha, 
            d_intermediate_matrix_B, nocc * nvir, 
            &d_intermediate_matrix_B[i * nvir], nocc * nvir,
            &beta, 
            d_iajb, nocc * nvir
        );

        cudaEventRecord(events_for_sync[iter_count*4 + 0], streams[0]);
        for(int s = 1; s < 4; s++) cudaStreamWaitEvent(streams[s], events_for_sync[iter_count*4 + 0], 0);

        for (int j = 0; j < num_blocks_list.size(); j++){
            num_blocks_list[j].kernel<<<num_blocks_list[j].num_blocks, num_threads, 0, streams[j]>>>(nocc, nocc_block, nvir, i, num_auxiliary_basis, d_iajb, d_eps, d_energy);
            cudaEventRecord(events_for_sync[iter_count*4 + j], streams[j]);
            cudaStreamWaitEvent(streams[0], events_for_sync[iter_count*4 + j], 0);
        }

        iter_count++;
    }


    cudaEventRecord(events[1], streams[0]);
    cudaEventSynchronize(events[1]);

    tracked_cudaFree(d_iajb);
    tracked_cudaFree(d_B_copy);
    cublasDestroy(handle);






    // cudaStreamSynchronize(streams[0]);
    double energy;
    cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);
    printf("RMP2_energy: %.10f\n", energy);
    printf("RMP2_total_energy: %.10f\n", rhf_.get_total_energy() + energy);



    // timeに、エネルギー計算部分の実行時間
    float time;
    cudaEventElapsedTime(&time, events[0], events[1]);
    std::cout << "Execution time: " << std::setprecision(15) << time / 1000.0 << " [s]" << std::endl;


    printf("(nocc, nvir, naux) = (%d, %d, %d)\n",nocc, nvir, num_auxiliary_basis);
    tracked_cudaFree(d_energy);


    for (int i = 0; i < 4; i++) cudaStreamDestroy(streams[i]);
    

    // events_for_sync
    for(int i=0; i<niter*4; i++) cudaEventDestroy(events_for_sync[i]);
    for(int i=0; i<2; i++) cudaEventDestroy(events[i]);

    return energy;
}



// ============================================================
//  Laplace-transformed MP2 from B_ia^P
//  E_os via Laplace: ||C(t)^T C(t)||_F^2 per quadrature point
//  E_ss via Laplace: sum_{ij} ||C_i(t)^T C_j(t)||_F^2 per point
// ============================================================

// GPU kernel: scale B_ia^P → C_ia^P(t) = B_ia^P * exp(-t * (eps[nocc+a] - eps[i]) / 2)
__global__ void laplace_scale_B_kernel(
    const real_t* __restrict__ d_B,
    real_t* __restrict__ d_C,
    const real_t* __restrict__ d_eps,
    int nocc, int nvir, int naux, double t_half)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t ov = (size_t)nocc * nvir;
    if (idx >= ov * naux) return;

    size_t P = idx / ov;
    size_t ia = idx % ov;
    int i = (int)(ia / nvir);
    int a = (int)(ia % nvir);

    double scale = exp(-t_half * (d_eps[nocc + a] - d_eps[i]));
    d_C[idx] = d_B[idx] * scale;
}

// Compute LT-MP2 spin components: E_os and E_ss
// d_B: B_ia^P in column-major [nocc*nvir, naux] (after MO transform)
// Returns {E_os, E_ss}
static std::pair<real_t, real_t> compute_lt_mp2_spin_components(
    real_t* d_B, int naux, int nocc, int nvir, const real_t* d_eps,
    int n_laplace_points)
{
    const size_t ov = (size_t)nocc * nvir;

    // Determine denominator range for quadrature
    std::vector<double> h_eps(nocc + nvir);
    cudaMemcpy(h_eps.data(), d_eps, (nocc + nvir) * sizeof(double), cudaMemcpyDeviceToHost);

    double eps_min = h_eps[nocc] - h_eps[nocc - 1]; // HOMO-LUMO gap
    double eps_max = h_eps[nocc + nvir - 1] - h_eps[0]; // max denominator
    if (eps_min < 1e-6) eps_min = 1e-6; // safety

    auto quad = generate_laplace_quadrature(eps_min, eps_max, n_laplace_points);

    std::cout << "  LT-MP2: " << quad.num_points << " quadrature points, "
              << "eps_range=[" << std::scientific << std::setprecision(3)
              << eps_min << ", " << eps_max << "]" << std::defaultfloat << std::endl;

    // Allocate scaled B matrix C(t)
    real_t* d_C = nullptr;
    tracked_cudaMalloc(&d_C, ov * naux * sizeof(real_t));

    // Allocate X = C^T C for OS term [naux × naux]
    real_t* d_X = nullptr;
    tracked_cudaMalloc(&d_X, (size_t)naux * naux * sizeof(real_t));

    // Allocate N^{ij} and its transpose for exchange [naux × naux]
    real_t* d_Nij = nullptr;
    real_t* d_Nij_T = nullptr;
    tracked_cudaMalloc(&d_Nij, (size_t)naux * naux * sizeof(real_t));
    tracked_cudaMalloc(&d_Nij_T, (size_t)naux * naux * sizeof(real_t));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0, beta = 0.0;

    double E_os_total = 0.0, E_ss_total = 0.0;
    int threads = 256;

    for (int k = 0; k < quad.num_points; k++) {
        double t = quad.points[k];
        double w = quad.weights[k];
        double t_half = t / 2.0;

        // Scale B → C(t): C_ia^P = B_ia^P * exp(-t/2 * (eps_a - eps_i))
        {
            size_t total = ov * naux;
            int blocks = (int)((total + threads - 1) / threads);
            laplace_scale_B_kernel<<<blocks, threads>>>(d_B, d_C, d_eps, nocc, nvir, naux, t_half);
        }

        // === Opposite-spin (Coulomb): X = C^T C, E_os = ||X||_F^2 ===
        // C is col-major [ov, naux], so X = C^T C is [naux, naux]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    naux, naux, (int)ov,
                    &alpha, d_C, (int)ov, d_C, (int)ov,
                    &beta, d_X, naux);

        // ||X||_F^2 = dot(X, X)
        double X_norm_sq = 0.0;
        cublasDdot(handle, naux * naux, d_X, 1, d_X, 1, &X_norm_sq);
        E_os_total -= w * X_norm_sq;

        // === Same-spin (exchange): Σ_{ij} ||C_i^T C_j||_F^2 ===
        // C_i is a [nvir, naux] submatrix: C[i*nvir:(i+1)*nvir, :]
        // N^{ij} = C_i^T C_j is [naux, naux]
        // E_ex = Σ_{ij} ||N^{ij}||_F^2
        double E_ex_t = 0.0;
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                // C_i starts at offset i*nvir in the ov dimension
                const real_t* d_Ci = d_C + (size_t)i * nvir;  // col-major stride = ov
                const real_t* d_Cj = d_C + (size_t)j * nvir;

                // N^{ij} = C_i^T @ C_j where C_i is [nvir, naux] col-major with lda=ov
                cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            naux, naux, nvir,
                            &alpha, d_Ci, (int)ov, d_Cj, (int)ov,
                            &beta, d_Nij, naux);

                // Tr(N^2) = dot(vec(N), vec(N^T)) — need transpose
                const double zero = 0.0, one = 1.0;
                cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            naux, naux, &one, d_Nij, naux, &zero, d_Nij, naux, d_Nij_T, naux);

                double trN2 = 0.0;
                cublasDdot(handle, naux * naux, d_Nij, 1, d_Nij_T, 1, &trN2);
                E_ex_t += trN2;
            }
        }
        // E_ss = E_os - E_exchange (for closed-shell RHF)
        // Standard MP2: E = 2*E_os - E_exchange, so E_ss = E_os - E_exchange
        double E_os_t = -X_norm_sq;
        double E_ex_t_neg = -E_ex_t;
        double E_ss_t = E_os_t - E_ex_t_neg; // = -X_norm_sq + E_ex_t

        E_ss_total += w * E_ss_t;
    }

    cublasDestroy(handle);
    tracked_cudaFree(d_C);
    tracked_cudaFree(d_X);
    tracked_cudaFree(d_Nij);
    tracked_cudaFree(d_Nij_T);

    std::cout << "  LT-MP2: E_os = " << std::setprecision(12) << E_os_total
              << "  E_ss = " << E_ss_total
              << "  E_MP2 = " << (E_os_total + E_ss_total) << std::endl;

    return {E_os_total, E_ss_total};
}

// LT-SOS-MP2: E_os only via Laplace — no exchange loop, DGEMM only
static real_t compute_lt_sos_mp2_energy_from_B(
    real_t* d_B, int naux, int nocc, int nvir, const real_t* d_eps,
    int n_laplace_points)
{
    const size_t ov = (size_t)nocc * nvir;

    std::vector<double> h_eps(nocc + nvir);
    cudaMemcpy(h_eps.data(), d_eps, (nocc + nvir) * sizeof(double), cudaMemcpyDeviceToHost);

    double eps_min = h_eps[nocc] - h_eps[nocc - 1];
    double eps_max = h_eps[nocc + nvir - 1] - h_eps[0];
    if (eps_min < 1e-6) eps_min = 1e-6;

    auto quad = generate_laplace_quadrature(eps_min, eps_max, n_laplace_points);

    std::cout << "  LT-SOS-MP2: " << quad.num_points << " quadrature points" << std::endl;

    real_t* d_C = nullptr;
    tracked_cudaMalloc(&d_C, ov * naux * sizeof(real_t));

    real_t* d_X = nullptr;
    tracked_cudaMalloc(&d_X, (size_t)naux * naux * sizeof(real_t));

    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = 1.0, beta = 0.0;

    double E_os = 0.0;
    int threads = 256;

    for (int k = 0; k < quad.num_points; k++) {
        double t_half = quad.points[k] / 2.0;
        double w = quad.weights[k];

        // Scale B → C(t)
        size_t total = ov * naux;
        int blocks = (int)((total + threads - 1) / threads);
        laplace_scale_B_kernel<<<blocks, threads>>>(d_B, d_C, d_eps, nocc, nvir, naux, t_half);

        // X = C^T C [naux × naux]
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    naux, naux, (int)ov,
                    &alpha, d_C, (int)ov, d_C, (int)ov,
                    &beta, d_X, naux);

        // E_os(t) = ||X||_F^2
        double X_norm_sq = 0.0;
        cublasDdot(handle, naux * naux, d_X, 1, d_X, 1, &X_norm_sq);
        E_os -= w * X_norm_sq;
    }

    cublasDestroy(handle);
    tracked_cudaFree(d_C);
    tracked_cudaFree(d_X);

    real_t E_SOS = 1.3 * E_os;
    std::cout << "  LT-SOS-MP2: E_os = " << std::setprecision(12) << E_os
              << "  SOS-MP2 (1.3*E_os) = " << E_SOS << std::endl;
    return E_SOS;
}

// Helper: copy B, transform AO→MO, return B_ia^P on device (caller must free)
real_t* ERI_RI_RHF::build_B_ia() {
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const int naux = num_auxiliary_basis_;

    const size_t B_size = (size_t)naux * num_basis_ * num_basis_;
    real_t* d_B_copy = nullptr;
    tracked_cudaMalloc(&d_B_copy, B_size * sizeof(real_t));
    cudaMemcpy(d_B_copy, intermediate_matrix_B_.device_ptr(), B_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    real_t* d_tmp = nullptr;
    tracked_cudaMalloc(&d_tmp, sizeof(double) * num_basis_ * (size_t)nvir * naux);
    transform_intermediate_matrix(num_basis_, nocc, nvir, naux, d_C, d_B_copy, d_tmp);
    tracked_cudaFree(d_tmp);

    return d_B_copy;  // now contains B_ia^P [nocc*nvir, naux] col-major
}

// Helper: build B_ab^P (VV block) on device. Caller must free.
// Layout: [nvir*nvir × naux] col-major: element B^P_{a,b} at P*vv + a*nvir + b
real_t* ERI_RI_RHF::build_B_ab() {
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const size_t vv = (size_t)nvir * nvir;

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;

    // Copy AO-basis B
    const size_t B_ao_size = (size_t)naux * nbas * nbas;
    real_t* d_B_ao = nullptr;
    tracked_cudaMalloc(&d_B_ao, B_ao_size * sizeof(real_t));
    cudaMemcpy(d_B_ao, intermediate_matrix_B_.device_ptr(), B_ao_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Step 1: ν→a (same as nu2a_dgemm)
    // B^P_{μ,a} = Σ_ν C_{ν,nocc+a} × B^P_{μ,ν}
    // Layout: [nvir × naux*nbas] col-major
    real_t* d_B_mu_a = nullptr;
    tracked_cudaMalloc(&d_B_mu_a, (size_t)nbas * nvir * naux * sizeof(real_t));
    cudaMemset(d_B_mu_a, 0, (size_t)nbas * nvir * naux * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nvir, naux * nbas, nbas,
                &alpha, &d_C[nocc], nbas,
                d_B_ao, nbas,
                &beta, d_B_mu_a, nvir);
    tracked_cudaFree(d_B_ao);

    // Step 2: Transpose [nvir × naux*nbas] → [naux*nbas × nvir]
    real_t* d_B_transposed = nullptr;
    tracked_cudaMalloc(&d_B_transposed, (size_t)nbas * nvir * naux * sizeof(real_t));
    {
        int row = naux * nbas, col = nvir;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    row, col,
                    &alpha, d_B_mu_a, col,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, row);
    }

    // Step 3: μ→b (virtual): B^P_{b,a} = Σ_μ C_{μ,nocc+b} × B^P_{μ,a}
    // DGEMM: [nvir × nbas] × [nbas × naux*nvir] → [nvir × naux*nvir]
    cudaMemset(d_B_mu_a, 0, (size_t)nbas * nbas * naux * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nvir, naux * nvir, nbas,
                &alpha, &d_C[nocc], nbas,
                d_B_transposed, nbas,
                &beta, d_B_mu_a, nvir);

    // Step 4: Transpose [nvir × naux*nvir] → [naux*nvir × nvir]
    // Then reinterpret as [vv × naux] col-major: B^P_{b,a} at P*vv + b*nvir + a
    // Wait — the DGEMM output is [nvir_b × naux*nvir_a] col-major.
    // Element: d_B_mu_a[b + (P + a*naux)*nvir] = B^P_{b,a}... actually:
    // batch = P + a*naux? Let me check: ldb=nbas for the input, output has ldc=nvir.
    // d_B_mu_a[b + batch*nvir] where batch = P*nvir/... hmm.
    // Actually the DGEMM n=naux*nvir means the result has columns indexed by
    // (batch) = 0..naux*nvir-1, but the mapping of batch→(P,a) depends on the input layout.
    //
    // Input B (d_B_transposed): element B[μ, P+a*naux] = B^P_{μ,a}  (from step 2 transpose)
    // Wait, need to re-derive. After step 2 transpose:
    // d_B_transposed[(P*nbas+μ) + a*(naux*nbas)] = B^P_{μ,a}
    // In DGEMM with ldb=nbas: B[μ, batch] = d_B_transposed[μ + batch*nbas]
    //   batch*nbas = P*nbas + a*naux*nbas → batch = P + a*naux
    //   So n = naux*nvir with batch = a*naux + P
    //
    // DGEMM output: d_B_mu_a[b + batch*nvir] = Σ_μ C_{μ,b}^vir × B^P_{μ,a}
    //   = B^P_{b,a} where batch = a*naux + P
    //   d_B_mu_a[b + (a*naux + P)*nvir] = B^P_{b,a}
    //
    // We want: result[P*vv + b*nvir + a]
    // Current: result[b + a*naux*nvir + P*nvir]
    // These differ. Need a final transpose/reshape.
    //
    // Alternative: transpose to get [naux*nvir × nvir] and reinterpret.

    {
        int row = naux * nvir, col = nvir;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    col, row,
                    &alpha, d_B_mu_a, row,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, col);
    }
    // After transpose: d_B_transposed[a + (P*nvir+b)*nvir]... hmm.
    // cublasDgeam(T, _, col=nvir, row=naux*nvir): output [col × row] = [nvir × naux*nvir]
    // input A^T: A is [nvir × naux*nvir] (the d_B_mu_a), so A^T is [naux*nvir × nvir].
    // The call transposes to [col × row] = [nvir × naux*nvir]:
    // d_B_transposed[a + (P*nvir+b)*nvir]?? No:
    // geam output C[i,j] = A^T[i,j] = A[j,i]
    // With m=col=nvir, n=row=naux*nvir:
    // C[i, j] = A[j, i], C is [nvir × naux*nvir] with ldc=nvir
    // C[a, P*nvir+b] = A[P*nvir+b, a] = d_B_mu_a[(P*nvir+b) + a*(naux*nvir)]
    //                = d_B_mu_a[P*nvir + b + a*naux*nvir]
    // But d_B_mu_a[idx] where idx = b + (a*naux+P)*nvir = b + a*naux*nvir + P*nvir ✓
    // So C[a, P*nvir+b] = B^P_{b,a}
    // And d_B_transposed[a + (P*nvir+b)*nvir] = B^P_{b,a}
    //
    // We want: d_result[P*vv + b*nvir + a] = B^P_{b,a}
    // Current: d_B_transposed[a + P*nvir*nvir + b*nvir] = d_B_transposed[P*vv + b*nvir + a] ✓ !!

    tracked_cudaFree(d_B_mu_a);
    return d_B_transposed;  // B_ab^P [vv × naux] col-major: B^P_{a,b} at P*vv + a*nvir + b
}

// Helper: build B_ij^P (OO block) on device. Caller must free.
// Layout: [nocc*nocc × naux] col-major: element B^P_{i,j} at P*oo + i*nocc + j
real_t* ERI_RI_RHF::build_B_ij() {
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const int naux = num_auxiliary_basis_;
    const int nbas = num_basis_;
    const size_t oo = (size_t)nocc * nocc;

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;

    // Copy AO-basis B
    const size_t B_ao_size = (size_t)naux * nbas * nbas;
    real_t* d_B_ao = nullptr;
    tracked_cudaMalloc(&d_B_ao, B_ao_size * sizeof(real_t));
    cudaMemcpy(d_B_ao, intermediate_matrix_B_.device_ptr(), B_ao_size * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // Step 1: ν→j (occupied): B^P_{μ,j} = Σ_ν C_{ν,j} × B^P_{μ,ν}
    // Like nu2a but with C_occ (m=nocc, no offset)
    real_t* d_B_mu_j = nullptr;
    tracked_cudaMalloc(&d_B_mu_j, (size_t)nbas * nocc * naux * sizeof(real_t));
    cudaMemset(d_B_mu_j, 0, (size_t)nbas * nocc * naux * sizeof(real_t));

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nocc, naux * nbas, nbas,
                &alpha, d_C, nbas,      // C_occ: no offset
                d_B_ao, nbas,
                &beta, d_B_mu_j, nocc);
    tracked_cudaFree(d_B_ao);

    // Step 2: Transpose [nocc × naux*nbas] → [naux*nbas × nocc]
    real_t* d_B_transposed = nullptr;
    tracked_cudaMalloc(&d_B_transposed, (size_t)nbas * nocc * naux * sizeof(real_t));
    {
        int row = naux * nbas, col = nocc;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    row, col,
                    &alpha, d_B_mu_j, col,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, row);
    }

    // Step 3: μ→i (occupied): B^P_{i,j} = Σ_μ C_{μ,i} × B^P_{μ,j}
    cudaMemset(d_B_mu_j, 0, (size_t)nbas * nbas * naux * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                nocc, naux * nocc, nbas,
                &alpha, d_C, nbas,
                d_B_transposed, nbas,
                &beta, d_B_mu_j, nocc);

    // Step 4: Transpose [nocc × naux*nocc] → [nocc × naux*nocc] reshaped
    {
        int row = naux * nocc, col = nocc;
        cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    col, row,
                    &alpha, d_B_mu_j, row,
                    &beta, nullptr, (row >= col) ? row : col,
                    d_B_transposed, col);
    }
    // d_B_transposed[P*oo + i*nocc + j] = B^P_{i,j}

    tracked_cudaFree(d_B_mu_j);
    return d_B_transposed;  // B_ij^P [oo × naux] col-major
}

real_t ERI_RI_RHF::compute_lt_sos_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_B_ia = build_B_ia();
    real_t E_SOS = compute_lt_sos_mp2_energy_from_B(d_B_ia, num_auxiliary_basis_, nocc, nvir,
                                                     rhf_.get_orbital_energies().device_ptr(), 30);
    tracked_cudaFree(d_B_ia);
    return E_SOS;
}

real_t ERI_RI_RHF::compute_lt_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_B_ia = build_B_ia();
    auto [E_os, E_ss] = compute_lt_mp2_spin_components(d_B_ia, num_auxiliary_basis_, nocc, nvir,
                                                        rhf_.get_orbital_energies().device_ptr(), 30);
    tracked_cudaFree(d_B_ia);
    return E_os + E_ss;
}

real_t ERI_RI_RHF::compute_scs_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_B_ia = build_B_ia();
    auto [E_os, E_ss] = compute_lt_mp2_spin_components(d_B_ia, num_auxiliary_basis_, nocc, nvir,
                                                        rhf_.get_orbital_energies().device_ptr(), 30);
    tracked_cudaFree(d_B_ia);
    real_t E_SCS = (6.0/5.0) * E_os + (1.0/3.0) * E_ss;
    std::cout << "  RI-SCS-MP2 (c_os=6/5, c_ss=1/3): " << std::setprecision(12) << E_SCS << std::endl;
    return E_SCS;
}

real_t ERI_RI_RHF::compute_sos_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = rhf_.get_num_basis() - nocc;
    real_t* d_B_ia = build_B_ia();
    auto [E_os, E_ss] = compute_lt_mp2_spin_components(d_B_ia, num_auxiliary_basis_, nocc, nvir,
                                                        rhf_.get_orbital_energies().device_ptr(), 30);
    tracked_cudaFree(d_B_ia);
    real_t E_SOS = 1.3 * E_os;
    std::cout << "  RI-SOS-MP2 (c_os=1.3): " << std::setprecision(12) << E_SOS << std::endl;
    return E_SOS;
}

// ============================================================
//  Common RI-MP2 computation from a given B matrix pointer.
//  Used by Stored RI, Semi-Direct RI, and Direct RI.
// ============================================================
real_t compute_ri_mp2_from_B(
    real_t* d_B, int num_basis, int num_auxiliary_basis,
    int nocc, int nvir, real_t* d_C, real_t* d_eps)
{
    // MO transform: B(Q,μν) → B(Q,ia)
    real_t* d_tmp;
    tracked_cudaMalloc((void**)&d_tmp, sizeof(double) * num_basis * (size_t)nvir * num_auxiliary_basis);
    transform_intermediate_matrix(num_basis, nocc, nvir, num_auxiliary_basis, d_C, d_B, d_tmp);
    tracked_cudaFree(d_tmp);

    // Allocate energy accumulator
    double *d_energy;
    tracked_cudaMalloc((void**)&d_energy, sizeof(double));
    cudaMemset(d_energy, 0, sizeof(double));

    const int num_threads = 1024;
    cudaStream_t streams[4];
    for(int i = 0; i < 4; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    double *d_iajb = nullptr;
    int nocc_block;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, streams[0]);

    search_k_and_cudamalloc_4cERI(nocc, nvir, nocc_block, &d_iajb, streams[0]);

    size_t num_blocks_3 = ((size_t)(nocc_block * (size_t)nvir * (nvir - 1.0) / 2) + num_threads - 1) / num_threads;
    size_t num_blocks_4 = ((size_t)(nocc_block * (size_t)nvir) + num_threads - 1) / num_threads;

    int niter = ((double)nocc + nocc_block - 1) / nocc_block;
    cudaEvent_t events[2];
    for (int i = 0; i < 2; i++) cudaEventCreate(&events[i]);
    cudaEvent_t *events_for_sync = new cudaEvent_t[niter * 4];
    for(int i = 0; i < niter * 4; i++) cudaEventCreate(&events_for_sync[i]);

    std::vector<KernelPair> num_blocks_list{
        {calc_RI_RMP2_energy_kernel1, 0},
        {calc_RI_RMP2_energy_kernel2, 0},
        {calc_RI_RMP2_energy_kernel3, num_blocks_3},
        {calc_RI_RMP2_energy_kernel4, num_blocks_4}};

    int iter_count = 0;
    const double alpha = 1.0, beta = 0.0;

    cudaDeviceSynchronize();
    cudaEventRecord(events[0], streams[0]);

    for(int i = 0; i < nocc; i += nocc_block) {
        num_blocks_list[0].num_blocks = (((size_t)(nocc_block*i + (size_t)nocc_block*(nocc_block-1)/2) * nvir*(nvir-1)/2) + num_threads - 1) / num_threads;
        num_blocks_list[1].num_blocks = (((size_t)(nocc_block*i + (size_t)nocc_block*(nocc_block-1)/2) * nvir) + num_threads - 1) / num_threads;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            nocc * nvir,
            ((nocc_block < nocc - i) ? nocc_block : nocc - i) * nvir,
            num_auxiliary_basis,
            &alpha, d_B, nocc * nvir,
            &d_B[i * nvir], nocc * nvir,
            &beta, d_iajb, nocc * nvir);

        cudaEventRecord(events_for_sync[iter_count*4 + 0], streams[0]);
        for(int s = 1; s < 4; s++) cudaStreamWaitEvent(streams[s], events_for_sync[iter_count*4 + 0], 0);

        for (size_t j = 0; j < num_blocks_list.size(); j++) {
            num_blocks_list[j].kernel<<<num_blocks_list[j].num_blocks, num_threads, 0, streams[j]>>>(
                nocc, nocc_block, nvir, i, num_auxiliary_basis, d_iajb, d_eps, d_energy);
            cudaEventRecord(events_for_sync[iter_count*4 + j], streams[j]);
            cudaStreamWaitEvent(streams[0], events_for_sync[iter_count*4 + j], 0);
        }
        iter_count++;
    }

    cudaEventRecord(events[1], streams[0]);
    cudaEventSynchronize(events[1]);

    tracked_cudaFree(d_iajb);
    cublasDestroy(handle);

    double energy;
    cudaMemcpy(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_energy);

    for (int i = 0; i < 4; i++) cudaStreamDestroy(streams[i]);
    for (int i = 0; i < niter * 4; i++) cudaEventDestroy(events_for_sync[i]);
    for (int i = 0; i < 2; i++) cudaEventDestroy(events[i]);
    delete[] events_for_sync;

    return energy;
}

// ============================================================
//  Build temporary B matrix for Semi-Direct/Direct RI
//  Uses the same 3-center + L^{-1} logic as computeFockMatrix_RI_Direct_v2
// ============================================================
static inline int calcIdx_triangular_local(int a, int b, int N) {
    return a * N - a * (a + 1) / 2 + b;
}

static real_t* build_temporary_B_matrix(
    const std::vector<ShellTypeInfo>& shell_type_infos,
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos,
    const PrimitiveShell* d_primitive_shells,
    const real_t* d_cgto_normalization_factors,
    const std::vector<ShellTypeInfo>& auxiliary_shell_type_infos,
    const PrimitiveShell* d_auxiliary_primitive_shells,
    const real_t* d_auxiliary_cgto_normalization_factors,
    const size_t2* d_primitive_shell_pair_indices,
    const real_t* d_schwarz_upper_bound_factors,
    const real_t* d_auxiliary_schwarz_upper_bound_factors,
    const real_t* d_two_center_eris_cholesky,
    double schwarz_screening_threshold,
    int num_basis, int num_auxiliary_basis,
    const real_t* d_boys_grid)
{
    const int nao = num_basis;
    const int naux = num_auxiliary_basis;
    const size_t nao2 = (size_t)nao * nao;
    const double one = 1.0;

    // Step 1: Compute raw 3-center ERIs
    real_t* d_B = nullptr;
    tracked_cudaMalloc(&d_B, (size_t)naux * nao2 * sizeof(real_t));
    cudaMemset(d_B, 0, (size_t)naux * nao2 * sizeof(real_t));

    {
        const int threads_per_block = 128;
        const int shell_type_count = shell_type_infos.size();
        const int auxiliary_shell_type_count = auxiliary_shell_type_infos.size();

        for (int s0 = 0; s0 < shell_type_count; ++s0) {
            for (int s1 = s0; s1 < shell_type_count; ++s1) {
                for (int s2 = 0; s2 < auxiliary_shell_type_count; ++s2) {
                    const ShellTypeInfo shell_s0 = shell_type_infos[s0];
                    const ShellTypeInfo shell_s1 = shell_type_infos[s1];
                    const ShellTypeInfo shell_s2 = auxiliary_shell_type_infos[s2];

                    const int64_t num_tasks = ((s0 == s1)
                        ? ((int64_t)shell_s0.count * (shell_s0.count + 1) / 2)
                        : ((int64_t)shell_s0.count * shell_s1.count))
                        * (int64_t)shell_s2.count;
                    const int num_blocks = (int)((num_tasks + threads_per_block - 1) / threads_per_block);
                    const int pair_idx = calcIdx_triangular_local(s0, s1, shell_type_count);

                    ShellTypeInfo shell_s0_nooff = shell_s0;  shell_s0_nooff.start_index = 0;
                    ShellTypeInfo shell_s1_nooff = shell_s1;  shell_s1_nooff.start_index = 0;

                    gpu::get_3center_kernel(s0, s1, s2)<<<num_blocks, threads_per_block>>>(
                        d_B, d_primitive_shells, d_auxiliary_primitive_shells,
                        d_cgto_normalization_factors, d_auxiliary_cgto_normalization_factors,
                        shell_s0_nooff, shell_s1_nooff, shell_s2,
                        num_tasks, nao,
                        &d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index],
                        &d_schwarz_upper_bound_factors[shell_pair_type_infos[pair_idx].start_index],
                        d_auxiliary_schwarz_upper_bound_factors,
                        schwarz_screening_threshold, naux, d_boys_grid);
                }
            }
        }
        cudaDeviceSynchronize();
    }

    // Step 2: B = L^{-1} × raw_3center (via DTRSM)
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    cublasDtrsm(handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                nao2, naux, &one, d_two_center_eris_cholesky, naux, d_B, nao2);

    return d_B;
}

// ============================================================
//  Semi-Direct RI MP2
// ============================================================
real_t ERI_RI_SemiDirect_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = num_basis_ - nocc;
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    real_t* d_eps = rhf_.get_orbital_energies().device_ptr();

    // === CPU fallback: use stored B matrix cached in ERI_RI_Direct precomputation ===
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(d_C, num_basis_);
        real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc);
        tracked_cudaFree(d_mo_eri);
        return E;
    }

    std::cout << "  [Semi-Direct RI-MP2] Building temporary B matrix..." << std::endl;

    real_t* d_B = build_temporary_B_matrix(
        hf_.get_shell_type_infos(), hf_.get_shell_pair_type_infos(),
        hf_.get_primitive_shells().device_ptr(), hf_.get_cgto_normalization_factors().device_ptr(),
        auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        primitive_shell_pair_indices.device_ptr(),
        schwarz_upper_bound_factors.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        two_center_eris.device_ptr(),  // Cholesky factor L
        rhf_.get_schwarz_screening_threshold(),
        num_basis_, num_auxiliary_basis_,
        hf_.get_boys_grid().device_ptr());

    real_t energy = compute_ri_mp2_from_B(d_B, num_basis_, num_auxiliary_basis_, nocc, nvir, d_C, d_eps);
    tracked_cudaFree(d_B);

    std::cout << "h_E: " << std::setprecision(12) << energy << std::endl;
    return energy;
}

// ============================================================
//  Direct RI MP2 (same as Semi-Direct for MP2 — both build B temporarily)
// ============================================================
real_t ERI_RI_Direct_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = num_basis_ - nocc;
    real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    real_t* d_eps = rhf_.get_orbital_energies().device_ptr();

    // === CPU fallback: use stored B matrix cached in ERI_RI_Direct precomputation ===
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(d_C, num_basis_);
        real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, num_basis_, nocc);
        tracked_cudaFree(d_mo_eri);
        return E;
    }

    std::cout << "  [Direct RI-MP2] Building temporary B matrix..." << std::endl;

    real_t* d_B = build_temporary_B_matrix(
        hf_.get_shell_type_infos(), hf_.get_shell_pair_type_infos(),
        hf_.get_primitive_shells().device_ptr(), hf_.get_cgto_normalization_factors().device_ptr(),
        auxiliary_shell_type_infos_, auxiliary_primitive_shells_.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        primitive_shell_pair_indices.device_ptr(),
        schwarz_upper_bound_factors.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        two_center_eris.device_ptr(),  // Cholesky factor L
        rhf_.get_schwarz_screening_threshold(),
        num_basis_, num_auxiliary_basis_,
        hf_.get_boys_grid().device_ptr());

    real_t energy = compute_ri_mp2_from_B(d_B, num_basis_, num_auxiliary_basis_, nocc, nvir, d_C, d_eps);
    tracked_cudaFree(d_B);

    std::cout << "h_E: " << std::setprecision(12) << energy << std::endl;
    return energy;
}

// ============================================================
//  SOS-ADC(2) via Laplace transform + RI
// ============================================================
void ERI_RI_RHF::compute_sos_adc2(int n_states) {
    const int nocc = rhf_.get_num_electrons() / 2 - rhf_.get_num_frozen_core();
    const int nvir = rhf_.get_num_basis() - rhf_.get_num_electrons() / 2;
    const int singles_dim = nocc * nvir;

    std::cout << "\n---- SOS-ADC(2) Laplace ---- "
              << "nocc=" << nocc << ", nvir=" << nvir
              << ", naux=" << num_auxiliary_basis_
              << ", singles=" << singles_dim
              << ", nstates=" << n_states << std::endl;

    // build_B_ia() uses full occupied space (no frozen core subtraction).
    // Assert frozen_core == 0 to avoid dimension mismatch.
    if (rhf_.get_num_frozen_core() != 0) {
        std::cerr << "[SOS-ADC(2)] Error: frozen core not yet supported in SOS-Laplace path." << std::endl;
        return;
    }

    // Build B_ia^P on device [ov × naux] col-major
    Timer setup_timer;
    real_t* d_B_ia = build_B_ia();
    real_t* d_B_ab = build_B_ab();
    real_t* d_B_ij = build_B_ij();
    std::cout << "  B-block build time: " << std::fixed << std::setprecision(3)
              << setup_timer.elapsed_seconds() << " s" << std::endl;

    // --- Build M11 from RI (no nao⁴ MO-ERI needed) ---
    Timer m11_timer;
    real_t* d_M11_from_adc2 = nullptr;
    tracked_cudaMalloc(&d_M11_from_adc2, (size_t)singles_dim * singles_dim * sizeof(real_t));
    RIADC2SchurOperator::build_M11_from_RI(
        d_M11_from_adc2, d_B_ia, d_B_ab, d_B_ij,
        rhf_.get_orbital_energies().device_ptr(), nocc, nvir, num_auxiliary_basis_);
    std::cout << "  M11 build time: " << std::fixed << std::setprecision(3)
              << m11_timer.elapsed_seconds() << " s" << std::endl;

    // ============================================================
    //  RI-ADC(2) exact Schur + Davidson + omega-iteration
    // ============================================================
    std::vector<double> sos_eigenvalues;
    std::vector<real_t> h_sos_eigenvectors((size_t)n_states * singles_dim, 0.0);
    {
        Timer ri_timer;
        std::cout << "\n  --- RI-ADC(2) Schur Davidson ---" << std::endl;

        RIADC2SchurOperator ri_op(
            d_B_ia, d_B_ab, d_B_ij, d_M11_from_adc2,
            rhf_.get_orbital_energies().device_ptr(),
            nocc, nvir, num_auxiliary_basis_);

        DavidsonConfig dav_config;
        dav_config.num_eigenvalues = n_states;
        dav_config.convergence_threshold = 1e-6;
        dav_config.max_subspace_size = std::min(singles_dim, std::max(30, 4 * n_states));
        dav_config.max_iterations = 200;
        dav_config.use_preconditioner = true;
        dav_config.symmetric = true;
        dav_config.verbose = 1;

        // Initial Davidson solve (omega=0)
        ri_op.set_omega(0.0);
        {
            DavidsonSolver init_solver(ri_op, dav_config);
            init_solver.solve();
            sos_eigenvalues.resize(n_states);
            const auto& evals = init_solver.get_eigenvalues();
            for (int k = 0; k < n_states && k < (int)evals.size(); k++)
                sos_eigenvalues[k] = evals[k];
        }

        std::cout << "  [RI] Initial (omega=0):";
        for (int k = 0; k < n_states; k++)
            std::cout << " " << std::fixed << std::setprecision(6) << sos_eigenvalues[k];
        std::cout << std::endl;

        // Per-root omega-iteration
        const double omega_thr = 1e-8;
        const int max_omega_iter = 15;

        for (int root = 0; root < n_states; root++) {
            double omega = sos_eigenvalues[root];
            bool converged = false;

            for (int iter = 0; iter < max_omega_iter; iter++) {
                ri_op.set_omega(omega);

                DavidsonSolver solver(ri_op, dav_config);
                solver.solve();

                const auto& evals = solver.get_eigenvalues();
                double omega_new = (root < (int)evals.size()) ? evals[root] : omega;
                double delta = std::abs(omega_new - omega);

                std::cout << "  [RI] Root " << root + 1 << " iter " << std::setw(2) << iter + 1
                          << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                          << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                          << std::defaultfloat << std::endl;

                if (delta < omega_thr) {
                    sos_eigenvalues[root] = omega_new;
                    solver.copy_eigenvectors_to_host(h_sos_eigenvectors.data());
                    converged = true;
                    std::cout << "  [RI] Root " << root + 1 << ": converged in "
                              << iter + 1 << " iterations" << std::endl;
                    break;
                }
                omega = omega_new;
                if (iter == max_omega_iter - 1)
                    solver.copy_eigenvectors_to_host(h_sos_eigenvectors.data());
            }
            if (!converged) {
                sos_eigenvalues[root] = omega;
                std::cout << "  [RI] Root " << root + 1 << ": NOT converged after "
                          << max_omega_iter << " iterations" << std::endl;
            }
        }

        std::cout << "  [RI] RI-ADC(2) Schur time: " << std::fixed << std::setprecision(3)
                  << ri_timer.elapsed_seconds() << " s" << std::endl;
    }

    tracked_cudaFree(d_M11_from_adc2);

    // ============================================================
    //  Store results and compute oscillator strengths
    // ============================================================
    rhf_.set_excitation_energies(sos_eigenvalues);

    rhf_.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "RI-ADC(2)",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        rhf_.get_coefficient_matrix().host_ptr(),
        sos_eigenvalues, h_sos_eigenvectors.data(),
        n_states, rhf_.get_num_basis(), nocc, nvir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);

    tracked_cudaFree(d_B_ab);
    tracked_cudaFree(d_B_ij);
    tracked_cudaFree(d_B_ia);
}

void ERI_RI_RHF::compute_sos_laplace_adc2(int n_states) {
    const int nocc = rhf_.get_num_electrons() / 2 - rhf_.get_num_frozen_core();
    const int nvir = rhf_.get_num_basis() - rhf_.get_num_electrons() / 2;
    const int singles_dim = nocc * nvir;

    std::cout << "\n---- SOS-Laplace-ADC(2) ----"
              << " nocc=" << nocc << ", nvir=" << nvir
              << ", naux=" << num_auxiliary_basis_
              << ", singles=" << singles_dim
              << ", nstates=" << n_states << std::endl;

    if (rhf_.get_num_frozen_core() != 0) {
        std::cerr << "[SOS-Laplace-ADC(2)] Error: frozen core not yet supported." << std::endl;
        return;
    }

    // Build B blocks and M11
    Timer setup_timer;
    real_t* d_B_ia = build_B_ia();
    real_t* d_B_ab = build_B_ab();
    real_t* d_B_ij = build_B_ij();
    std::cout << "  B-block build time: " << std::fixed << std::setprecision(3)
              << setup_timer.elapsed_seconds() << " s" << std::endl;

    Timer m11_timer;
    real_t* d_M11 = nullptr;
    tracked_cudaMalloc(&d_M11, (size_t)singles_dim * singles_dim * sizeof(real_t));
    RIADC2SchurOperator::build_M11_from_RI(
        d_M11, d_B_ia, d_B_ab, d_B_ij,
        rhf_.get_orbital_energies().device_ptr(), nocc, nvir, num_auxiliary_basis_);
    std::cout << "  M11 build time: " << std::fixed << std::setprecision(3)
              << m11_timer.elapsed_seconds() << " s" << std::endl;

    // Keep d_B_ab and d_B_ij alive for A3/B3 in SOS-Laplace operator

    // Davidson + omega iteration with SOS-Laplace operator
    std::vector<double> eigenvalues;
    std::vector<real_t> h_eigenvectors((size_t)n_states * singles_dim, 0.0);
    {
        Timer dav_timer;
        std::cout << "\n  --- SOS-Laplace-ADC(2) Davidson ---" << std::endl;

        const double c_c = rhf_.get_adc_c_c();
        SOSLaplaceADC2Operator op(
            d_B_ia, d_B_ij, d_B_ab, d_M11,
            rhf_.get_orbital_energies().device_ptr(),
            nocc, nvir, num_auxiliary_basis_, c_c);

        DavidsonConfig dav_config;
        dav_config.num_eigenvalues = n_states;
        dav_config.convergence_threshold = 1e-6;
        dav_config.max_subspace_size = std::min(singles_dim, std::max(30, 4 * n_states));
        dav_config.max_iterations = 200;
        dav_config.use_preconditioner = true;
        dav_config.symmetric = true;
        dav_config.verbose = 1;

        op.set_omega(0.0);
        op.update_laplace_quadrature();
        {
            DavidsonSolver solver(op, dav_config);
            solver.solve();
            eigenvalues.resize(n_states);
            const auto& evals = solver.get_eigenvalues();
            for (int k = 0; k < n_states && k < (int)evals.size(); k++)
                eigenvalues[k] = evals[k];
        }

        std::cout << "  [SOS-LT] Initial (omega=0):";
        for (int k = 0; k < n_states; k++)
            std::cout << " " << std::fixed << std::setprecision(6) << eigenvalues[k];
        std::cout << std::endl;

        const double omega_thr = 1e-8;
        const int max_omega_iter = 15;
        for (int root = 0; root < n_states; root++) {
            double omega = eigenvalues[root];
            bool converged = false;
            for (int iter = 0; iter < max_omega_iter; iter++) {
                op.set_omega(omega);
                op.update_laplace_quadrature();
                DavidsonSolver solver(op, dav_config);
                solver.solve();
                const auto& evals = solver.get_eigenvalues();
                double omega_new = (root < (int)evals.size()) ? evals[root] : omega;
                double delta = std::abs(omega_new - omega);
                std::cout << "  [SOS-LT] Root " << root + 1 << " iter " << std::setw(2) << iter + 1
                          << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                          << ", d_omega=" << std::scientific << std::setprecision(2) << delta
                          << std::defaultfloat << std::endl;
                if (delta < omega_thr) {
                    eigenvalues[root] = omega_new;
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
                    converged = true;
                    break;
                }
                omega = omega_new;
                if (iter == max_omega_iter - 1)
                    solver.copy_eigenvectors_to_host(h_eigenvectors.data());
            }
            if (!converged)
                eigenvalues[root] = omega;
        }
        std::cout << "  [SOS-LT] Davidson time: " << std::fixed << std::setprecision(3)
                  << dav_timer.elapsed_seconds() << " s" << std::endl;
    }

    tracked_cudaFree(d_M11);

    rhf_.set_excitation_energies(eigenvalues);
    rhf_.get_coefficient_matrix().toHost();
    const auto& prim_shells = rhf_.get_primitive_shells();
    const auto& cgto_norms = rhf_.get_cgto_normalization_factors();
    const_cast<DeviceHostMemory<PrimitiveShell>&>(prim_shells).toHost();
    const_cast<DeviceHostMemory<real_t>&>(cgto_norms).toHost();

    auto es_result = compute_excited_state_properties(
        "SOS-Laplace-ADC(2)",
        prim_shells.host_ptr(), prim_shells.size(),
        cgto_norms.host_ptr(),
        rhf_.get_shell_type_infos(),
        rhf_.get_coefficient_matrix().host_ptr(),
        eigenvalues, h_eigenvectors.data(),
        n_states, rhf_.get_num_basis(), nocc, nvir);
    rhf_.set_oscillator_strengths(es_result.oscillator_strengths);
    rhf_.set_excited_state_report(es_result.report);

    tracked_cudaFree(d_B_ab);
    tracked_cudaFree(d_B_ij);
    tracked_cudaFree(d_B_ia);
}

// ============================================================
//  THC-SOS-ADC(2) excited states (Phase 2.3: RI-Z path)
//
//  Same Davidson + ω-iteration driver as ERI_Stored_RHF::compute_thc_sos_adc2,
//  but the LS-THC core matrix Z is built from RI's 3-index AO tensor instead
//  of the analytic 4-index ERI:
//
//     Z = M^+ V (M^+)^T = (M^+ B) (M^+ B)^T,
//
//  with B = intermediate_matrix_B_ ((N_bas² × naux) col-major lda=N_bas²).
//  Avoids materialising the O(N_bas^4) ERI tensor; cost is O(N_bas² × N_g × naux).
// ============================================================
void ERI_RI_RHF::compute_thc_sos_adc2(int n_states) {
#ifdef GANSU_CPU_ONLY
    THROW_EXCEPTION("THC-SOS-ADC(2) requires GPU build.");
#else
    // Spherical: compute_X_ao_gpu builds the collocation in Cartesian and
    // transforms the AO axis Cart→Sph internally, so N_bas (sph) is consistent.
    const int N_bas = rhf_.get_num_basis();
    const int n_occ = rhf_.get_num_electrons() / 2;
    const int n_vir = N_bas - n_occ;
    const int N_orb = N_bas;
    const int singles_dim = n_occ * n_vir;
    const int naux = num_auxiliary_basis_;

    if (rhf_.get_num_frozen_core() != 0) {
        THROW_EXCEPTION("THC-SOS-ADC(2): frozen core not yet supported.");
    }

    // Spin: SINGLET block only — warn-and-ignore other --spin_type.
    if (rhf_.get_spin_type() != "singlet") {
        std::cout << "Warning: THC-SOS-ADC(2) computes SINGLET excited states only. "
                     "--spin_type \"" << rhf_.get_spin_type()
                  << "\" is ignored (use STEOM-CCSD / ADC(2) / CIS for triplets)." << std::endl;
    }

    const bool enable_b3a3_log = rhf_.get_thc_b3a3();
    std::cout << "\n---- THC-SOS-ADC(2) ("
              << (enable_b3a3_log ? "Coulomb + B3 + A3" : "Coulomb-only")
              << ", RI-Z) ----" << std::endl;
    std::cout << "  nocc=" << n_occ << ", nvir=" << n_vir
              << ", singles=" << singles_dim
              << ", naux=" << naux
              << ", nstates=" << n_states << std::endl;

    // ---- Phase 2.0a infrastructure (grid + X) ----
    auto& atoms_dh = const_cast<DeviceHostMemory<Atom>&>(rhf_.get_atoms());
    atoms_dh.toHost();
    std::vector<Atom> atoms_h(atoms_dh.host_ptr(),
                               atoms_dh.host_ptr() + atoms_dh.size());

    auto& prims_dh = const_cast<DeviceHostMemory<PrimitiveShell>&>(rhf_.get_primitive_shells());
    prims_dh.toHost();
    std::vector<PrimitiveShell> prims_h(prims_dh.host_ptr(),
                                         prims_dh.host_ptr() + prims_dh.size());

    auto& norms_dh = const_cast<DeviceHostMemory<real_t>&>(rhf_.get_cgto_normalization_factors());
    norms_dh.toHost();
    std::vector<real_t> norms_h(norms_dh.host_ptr(),
                                 norms_dh.host_ptr() + norms_dh.size());

    ThcGridOptions opts;
    {
        const int lev = rhf_.get_thc_lebedev_order();
        switch (lev) {
            case 110: opts.lebedev = LebedevOrder::L110; break;
            case 194: opts.lebedev = LebedevOrder::L194; break;
            case 302: opts.lebedev = LebedevOrder::L302; break;
            default: THROW_EXCEPTION("thc_lebedev_order must be 110, 194, or 302");
        }
    }
    opts.n_radial = rhf_.get_thc_n_radial();
    opts.weight_eps = 0.0;
    MolecularGrid grid = build_molecular_grid(atoms_h, opts);
    grid.host_to_device();

    const auto t_fit_start = std::chrono::steady_clock::now();
    auto report_elapsed = [&](const char* label) {
        const double s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_fit_start).count();
        std::cout << "  [THC t=" << std::fixed << std::setprecision(1) << s
                  << "s] " << label << std::endl;
    };
    std::cout << "  [THC] Grid: " << grid.num_points
              << " points (" << atoms_h.size() << " atoms × n_radial="
              << opts.n_radial << " × Lebedev=" << static_cast<int>(opts.lebedev)
              << ")" << std::endl;
    auto X_full_dh = compute_X_ao_gpu(N_bas, prims_h, norms_h, grid);
    report_elapsed("AO collocation X built (GPU 0)");

    // Optional Phase 2.3 (B) density-based pruning.
    int N_g_eff = static_cast<int>(grid.num_points);
    auto* X_for_thc = X_full_dh.get();
    std::unique_ptr<DeviceHostMatrix<real_t>> X_pruned_dh;
    const double rho_thr = rhf_.get_thc_density_threshold();
    if (rho_thr > 0.0) {
        const auto& density_dh = rhf_.get_density_matrix();
        int kept = 0;
        X_pruned_dh = prune_X_by_density_gpu(
            X_full_dh->device_ptr(), density_dh.device_ptr(),
            N_bas, N_g_eff, static_cast<real_t>(rho_thr), &kept);
        std::cout << "  [density-prune] " << N_g_eff << " → " << kept
                  << " points (ρ > " << rho_thr << ")" << std::endl;
        N_g_eff = kept;
        X_for_thc = X_pruned_dh.get();
        report_elapsed("density pruning done (GPU 0)");
    }

    // ---- Acquire AO B-tensor pointer.
    //   ERI_RI_RHF (single-GPU non-distributed): intermediate_matrix_B_ holds full B.
    //   ERI_RI_Distributed_RHF: intermediate_matrix_B_ is a 1×1 dummy; the AO B
    //   lives in d_B_local_ (per-GPU naux slices).  For multi-GPU we gather them
    //   onto GPU 0 into d_B_gathered.
    const real_t* d_B_ao_full = intermediate_matrix_B_.device_ptr();
    real_t* d_B_gathered = nullptr;
#ifdef GANSU_MULTI_GPU
    if (auto* dist = dynamic_cast<ERI_RI_Distributed_RHF*>(this)) {
        const auto& B_local = dist->get_d_B_local();
        const auto& naux_local = dist->get_naux_local();
        const int num_gpus_dist = dist->num_gpus();
        const std::size_t N2 = static_cast<std::size_t>(N_bas) * N_bas;

        if (num_gpus_dist == 1) {
            d_B_ao_full = B_local[0];
            if (naux_local[0] != naux) {
                THROW_EXCEPTION("THC-SOS-ADC(2) RI-Z: naux_local[0] mismatch with naux.");
            }
        } else {
            // Multi-GPU: gather B onto GPU 0 as a contiguous (N² × naux) col-major block.
            // Each GPU g contributes naux_local[g] cols, in order.
            MultiGpuManager::DeviceGuard guard0(0);
            tracked_cudaMalloc(&d_B_gathered,
                               N2 * static_cast<std::size_t>(naux) * sizeof(real_t));
            std::size_t col_offset = 0;  // running aux-col index on GPU 0
            for (int g = 0; g < num_gpus_dist; ++g) {
                const std::size_t bytes =
                    N2 * static_cast<std::size_t>(naux_local[g]) * sizeof(real_t);
                cudaMemcpyPeer(d_B_gathered + col_offset * N2, /*dst device*/ 0,
                               B_local[g], /*src device*/ g, bytes);
                col_offset += naux_local[g];
            }
            cudaDeviceSynchronize();
            if (col_offset != static_cast<std::size_t>(naux)) {
                tracked_cudaFree(d_B_gathered);
                THROW_EXCEPTION("THC-SOS-ADC(2) RI-Z gather: naux mismatch.");
            }
            d_B_ao_full = d_B_gathered;
            std::cout << "  [RI-Z] Gathered B from " << num_gpus_dist
                      << " GPUs (" << (N2 * naux * sizeof(real_t) / (1024 * 1024))
                      << " MB on GPU 0)" << std::endl;
            report_elapsed("B-gather done");
        }
    }
#endif // GANSU_MULTI_GPU

    // ---- LS-THC Z via RI 3-index B (no analytic ERI needed) ----
    const int thc_max_rank = rhf_.get_thc_max_rank();
    int rank = 0;
    real_t s_max = 0, s_min = 0;
    std::unique_ptr<DeviceHostMatrix<real_t>> Z_dh;
    if (thc_max_rank > 0) {
        const int q = rhf_.get_thc_rsvd_power_iter();
        std::cout << "  [LS-THC fit] randomized SVD on GPU 0, max_rank="
                  << thc_max_rank
                  << ", power_iter=" << q
                  << " (N²=" << (N_bas * N_bas)
                  << ", N_g=" << N_g_eff << ")" << std::endl;
        Z_dh = compute_Z_via_rand_svd_ri_gpu(
            X_for_thc->device_ptr(),
            d_B_ao_full,
            N_bas, N_g_eff, naux,
            thc_max_rank, /*n_power_iter=*/q,
            rhf_.get_thc_rel_cutoff(), &rank, &s_max, &s_min);
    } else {
        std::cout << "  [LS-THC fit] dense eigendecomp on GPU 0 (N²="
                  << (N_bas * N_bas) << ", N_g=" << N_g_eff << ")" << std::endl;
        Z_dh = compute_Z_via_M_svd_ri_gpu(
            X_for_thc->device_ptr(),
            d_B_ao_full,
            N_bas, N_g_eff, naux,
            rhf_.get_thc_rel_cutoff(), &rank, &s_max, &s_min);
    }
    report_elapsed("LS-THC Z built");

    std::cout << "  THC: N_g=" << N_g_eff
              << ", rank(M)=" << rank << "/" << N_bas * (N_bas + 1) / 2
              << ", sigma_max=" << s_max
              << ", sigma_min_kept=" << s_min << std::endl;

    auto& C_dh   = rhf_.get_coefficient_matrix();
    auto& eps_dh = rhf_.get_orbital_energies();
    auto X_mo_dh = transform_X_to_mo_gpu(
        X_for_thc->device_ptr(), C_dh.device_ptr(),
        N_bas, N_orb, N_g_eff);
    report_elapsed("X_mo (MO collocation) built");

    // Z is built; the AO-side scaffolding is no longer needed.  Freeing now
    // saves several GB on GPU 0 before the (large) operator buffers go up.
    if (d_B_gathered) {
        MultiGpuManager::DeviceGuard guard0(0);
        tracked_cudaFree(d_B_gathered);
        d_B_gathered = nullptr;
    }
    X_full_dh.reset();
    X_pruned_dh.reset();

    // ---- Build THC-SOS-ADC(2) operator ----
    const double c_os = 1.17;
    const int n_laplace = rhf_.get_thc_n_laplace();

    const int num_gpus = rhf_.get_num_gpus();
    const bool enable_b3a3 = enable_b3a3_log;
    const bool enable_b3 = rhf_.get_thc_b3();
    const bool enable_a3 = rhf_.get_thc_a3();
    if (enable_b3a3) {
        std::cout << "  B3=" << (enable_b3 ? "on" : "off")
                  << " A3=" << (enable_a3 ? "on" : "off") << std::endl;
    }
    int num_gpus_eff = num_gpus;
#ifndef GANSU_CPU_ONLY
    if (num_gpus_eff < 1) {
        num_gpus_eff = MultiGpuManager::instance().num_devices();
        if (num_gpus_eff < 1) num_gpus_eff = 1;
    }
#else
    if (num_gpus_eff < 1) num_gpus_eff = 1;
#endif
    std::cout << "  [THC] Constructing operator (M11 + per-GPU buffers on "
              << num_gpus_eff << " GPU(s))..." << std::endl;
    THCSOSADC2Operator op(
        X_mo_dh->device_ptr(),
        Z_dh->device_ptr(),
        eps_dh.device_ptr(),
        n_occ, n_vir, N_orb,
        N_g_eff,
        c_os, n_laplace,
        num_gpus, enable_b3a3, enable_b3, enable_a3);
    report_elapsed("operator built — entering Davidson");

    // ---- Davidson + ω-iteration ----
    DavidsonConfig dav_config;
    dav_config.num_eigenvalues = n_states;
    dav_config.convergence_threshold = 1e-6;
    dav_config.max_subspace_size = std::min(singles_dim, std::max(30, 4 * n_states));
    dav_config.max_iterations = 200;
    dav_config.use_preconditioner = true;
    dav_config.symmetric = true;
    dav_config.verbose = 2;  // print per-iteration sigma timing + residual

    std::vector<real_t> excitation_energies(n_states, 0.0);

    op.set_omega(0.0);
    {
        DavidsonSolver init_solver(op, dav_config);
        init_solver.solve();
        const auto& evals = init_solver.get_eigenvalues();
        for (int k = 0; k < n_states && k < (int)evals.size(); ++k) {
            excitation_energies[k] = static_cast<real_t>(evals[k]);
        }
    }

    std::cout << "  [THC] Initial (omega=0):";
    for (int k = 0; k < n_states; ++k) {
        std::cout << " " << std::fixed << std::setprecision(6) << excitation_energies[k];
    }
    std::cout << std::endl;

    const double omega_thr = 1e-6;
    const int max_omega_iter = 15;

    for (int root = 0; root < n_states; ++root) {
        double omega = static_cast<double>(excitation_energies[root]);
        for (int iter = 0; iter < max_omega_iter; ++iter) {
            op.set_omega(static_cast<real_t>(omega));
            DavidsonSolver solver(op, dav_config);
            solver.solve();
            const auto& evals = solver.get_eigenvalues();
            const double omega_new = (root < (int)evals.size())
                                      ? evals[root] : omega;
            const double delta = std::abs(omega_new - omega);
            std::cout << "  [THC] root " << (root + 1) << " iter " << (iter + 1)
                      << ": omega=" << std::fixed << std::setprecision(8) << omega_new
                      << " d=" << std::scientific << std::setprecision(2) << delta
                      << std::defaultfloat << std::endl;
            if (delta < omega_thr) {
                excitation_energies[root] = static_cast<real_t>(omega_new);
                std::cout << "  [THC] root " << (root + 1) << ": converged" << std::endl;
                break;
            }
            omega = omega_new;
        }
        excitation_energies[root] = static_cast<real_t>(omega);
    }

    rhf_.set_excitation_energies(std::vector<double>(
        excitation_energies.begin(), excitation_energies.end()));

    std::cout << "\n  THC-SOS-ADC(2) excitation energies (Hartree):" << std::endl;
    for (int k = 0; k < n_states; ++k) {
        std::cout << "    state " << (k + 1) << ": "
                  << std::fixed << std::setprecision(8)
                  << excitation_energies[k] << " Ha = "
                  << std::setprecision(4)
                  << (excitation_energies[k] * 27.2114) << " eV"
                  << std::endl;
    }

    // d_B_gathered already freed right after Z build (above).
#endif
}

} // namespace gansu
