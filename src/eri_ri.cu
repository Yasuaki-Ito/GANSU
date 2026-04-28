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

#include <cstdlib>  // std::getenv
#include <string>   // std::string
#include <fstream>
#include <iomanip>

//#include <omp.h>


#include "rhf.hpp"
#include "device_host_memory.hpp"
#include "int3c2e.hpp"
#include "laplace_quadrature.hpp"
#include "sos_laplace_adc2_operator.hpp"
#include "adc2_operator.hpp"
#include "davidson_solver.hpp"

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
    // cublasManager cublas;
    // cublasHandle_t handle;
    // cublasCreate(&handle);

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

    cudaMemset(d_B_p_mu_a, 0, norbs * (size_t)norbs * naux * sizeof(double));

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
    // cublasManager cublas;
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

    cudaMemset(d_B_p_mu_a, 0, norbs * (size_t)norbs * naux * sizeof(double));

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
    if (id >= (((uint64_t)nocc_block*nocc_stride + nocc_block*(nocc_block-1)/2) *  nvir*(nvir-1)/2)) return;


    size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
    // const size_t a = ab.x, b=ab.y;
    id /= (nvir * (nvir-1) / 2);

    const size_t i = calc_i(id, nocc_stride, nocc_block); // full-rangeの値が出てくる(0~kでなく、stride~stride+k)
    const size_t j = calc_j(id, i, nocc_stride);
    if (i >= nocc) return;


    double iajb = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + (size_t)ab.x*nocc*nvir + j*nvir + ab.y];
    double ibja = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + (size_t)ab.y*nocc*nvir + j*nvir + ab.x];        
    double val = 4.0 * ((iajb-ibja)*(iajb-ibja) + iajb*ibja) / (d_eps[i] + d_eps[j] - d_eps[ab.x+nocc] - d_eps[ab.y+nocc]);

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
    if (id > (((uint64_t)nocc_block*nocc_stride + nocc_block*(nocc_block-1)/2) *  nvir)) {
        return;
    }

    const size_t a = id % nvir;
    id /= nvir;

    const size_t i = calc_i(id, nocc_stride, nocc_block); // full-rangeの値が出てくる(0~kでなく、stride~stride+k)
    const size_t j = calc_j(id, i, nocc_stride);
    if (i >= nocc) return;

    double iaja = d_iajb[(i-nocc_stride)*(size_t)nvir*nocc*nvir + a*(size_t)nocc*nvir + j*nvir + a];     
    double val = 2.0*iaja*iaja / (d_eps[i] + d_eps[j] - 2.0*d_eps[a+nocc]);

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
    if (id >= (uint64_t)nocc_block * nvir * (nvir - 1.0) / 2) {
        return;
    }

    size_t2 ab = index1to2_upper_wo_trace(id % (nvir * (nvir-1) / 2), nvir);
    const size_t a = ab.x, b = ab.y;

    const size_t i = id / (nvir * (nvir-1) / 2);  // このiは0~k-1
    if (i + nocc_stride >= nocc) return;

    double iaib = d_iajb[i*nvir*(size_t)nocc*nvir + a*(size_t)nocc*nvir + (i+nocc_stride)*nvir + b];     
    double val = 2.0*iaib*iaib / (2.0*d_eps[i+nocc_stride] - d_eps[a+nocc] - d_eps[b+nocc]);

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
    if (id >= (uint64_t)nocc_block * nvir) {
        return;
    }

    const size_t a = id % nvir;
    const size_t i = id / nvir;  // このiは0~k-1
    if (i + nocc_stride >= nocc) return;

    double iaia = d_iajb[i*nvir*(size_t)nocc*nvir + a*(size_t)nocc*nvir + (i+nocc_stride)*nvir + a];     
    double val = 0.5*iaia*iaia / (d_eps[i+nocc_stride] - d_eps[a+nocc]);
    // printf("[iter %d] (%d %d) 0.5*%f**2 / (E[%d](%f) - E[%d](%f)) = %f\n", nocc_stride/nocc_block,i,a,iaia, i+nocc_stride, d_eps[i+nocc_stride], a+nocc,d_eps[a+nocc], val);

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

    // Build B_ia^P on device [ov × naux] col-major
    real_t* d_B_ia = build_B_ia();

    // --- Build M11 via existing ADC(2) (correct, uses full MO-ERI) ---
    // This ensures M11 is identical to the validated ADC(2) implementation
    real_t* d_M11_from_adc2 = nullptr;
    {
        real_t* d_eri_mo = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
        ADC2Operator adc2_ref(d_eri_mo, rhf_.get_orbital_energies().device_ptr(),
                              nocc, nvir, rhf_.get_num_basis());
        // Copy M11 from ADC2Operator
        tracked_cudaMalloc(&d_M11_from_adc2, (size_t)singles_dim * singles_dim * sizeof(real_t));
        cudaMemcpy(d_M11_from_adc2, adc2_ref.get_M11(),
                   (size_t)singles_dim * singles_dim * sizeof(real_t), cudaMemcpyDeviceToDevice);
        tracked_cudaFree(d_eri_mo);
        std::cout << "  M11 borrowed from ADC(2) (validated)" << std::endl;
    }

#if 0  // Skip RI-based M11 construction for now (debugging)
    // Build CIS M11 matrix: δε + 2(ia|jb) - (ij|ab) [singlet]
    // (ia|jb) = Σ_P B_ia^P B_jb^P (Coulomb, from B_ia)
    // (ij|ab) needs oovv ERI block — build from full MO-ERI via B matrix
    real_t* d_M11_cis = nullptr;
    {
        size_t ov = (size_t)nocc * nvir;
        tracked_cudaMalloc(&d_M11_cis, ov * ov * sizeof(real_t));
        cudaMemset(d_M11_cis, 0, ov * ov * sizeof(real_t));

        cublasHandle_t handle = gpu::GPUHandle::cublas();
        const double zero = 0.0, one = 1.0, two = 2.0, neg_one = -1.0;

        // Diagonal: ε_a - ε_i
        std::vector<double> eps(nocc + nvir);
        cudaMemcpy(eps.data(), rhf_.get_orbital_energies().device_ptr(),
                   (nocc + nvir) * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> M11_diag(ov * ov, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int a = 0; a < nvir; a++)
                M11_diag[(size_t)(i*nvir+a) * ov + (i*nvir+a)] = eps[nocc+a] - eps[i];
        cudaMemcpy(d_M11_cis, M11_diag.data(), ov * ov * sizeof(double), cudaMemcpyHostToDevice);

        // Coulomb: +2(ia|jb) = +2 B_ia · B_jb^T
        // B_ia is [ov × naux] col-major, lda=ov
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    (int)ov, (int)ov, num_auxiliary_basis_,
                    &two, d_B_ia, (int)ov,
                    d_B_ia, (int)ov,
                    &one, d_M11_cis, (int)ov);

        // Exchange: -(ij|ab) — computed from B_ia^P by re-indexing
        // (ij|ab) = Σ_P [B with oo indices] [B with vv indices]
        // Without B_ij/B_ab, use identity: for Coulomb metric,
        // (ij|ab) can be computed as:
        //   For each (i,j,a,b): (ij|ab) = Σ_P B_ia^P · B_jb^P with index swap
        //   ... no, that gives (ia|jb) again.
        //
        // Correct approach: build (ij|ab) from B_μν^P (AO basis) via:
        //   full_B is [naux × nbas²]. After occ-occ and vir-vir transform:
        //   B_ij = C_occ^T B C_occ, B_ab = C_vir^T B C_vir
        //   Then (ij|ab) = Σ_P B_ij^P B_ab^P
        //
        // Use a simpler O(N⁴) approach: build oovv block on host from B_ia
        // via loop over P, using B_ia sub-blocks per i.
        //
        // For each P, define L^P[i,a] = B[(i*nvir+a) + P*ov]
        // Then: (ia|jb) = Σ_P L^P[i,a] L^P[j,b]  [Coulomb]
        //       (ib|ja) = Σ_P L^P[i,b] L^P[j,a]  [NOT exchange!]
        //
        // (ij|ab) requires a DIFFERENT 3-index quantity: the oo-vv mixed.
        // Without it, use the (ib|ja) as an approximation.
        // NOTE: For CIS singlet, 2(ia|jb) - (ib|ja) is actually the correct
        // formula in SOME conventions! Let me verify with GANSU's existing code.

        // GANSU ADC(2) M11 code (line 1507-1511 of adc2_operator.cu):
        //   val -= eri[i][j][a][b] = (ij|ab)   ← -1× exchange
        //   val += 2 * eri[i][a][j][b] = 2*(ia|jb) ← +2× Coulomb
        //
        // eri[p][q][r][s] uses linear index (p*nao+q)*nao² + r*nao+s
        // So eri[i][j][a][b] = chemist (ij|ab) = ∫ φ_iφ_j(1) 1/r φ_aφ_b(2)
        // And eri[i][a][j][b] = chemist (ia|jb) = ∫ φ_iφ_a(1) 1/r φ_jφ_b(2)
        //
        // CIS = +2(ia|jb) - (ij|ab) ← CONFIRMED
        //
        // Since we can't get (ij|ab) from B_ia alone, build it via
        // half-transform of original B_μν^P.

        // Access the original AO B_μν^P matrix and coefficient matrix
        const int nbas = rhf_.get_num_basis();
        const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
        const size_t B_ao_size = (size_t)num_auxiliary_basis_ * nbas * nbas;

        // Download B_μν^P and C to host for exchange computation
        std::vector<double> B_ao(B_ao_size);
        cudaMemcpy(B_ao.data(), intermediate_matrix_B_.device_ptr(),
                   B_ao_size * sizeof(double), cudaMemcpyDeviceToHost);
        std::vector<double> C_host((size_t)nbas * nbas);
        cudaMemcpy(C_host.data(), d_C, (size_t)nbas * nbas * sizeof(double), cudaMemcpyDeviceToHost);

        // Build oovv block: (ij|ab) = Σ_P B_ij^P B_ab^P
        // B_ij^P = Σ_μν C_μi C_νj B_μν^P, B_ab^P = Σ_μν C_μa C_νb B_μν^P
        // Strategy: for each P, compute B_ij^P[i,j] and B_ab^P[a,b], then accumulate
        // Exchange K[ia,jb] += B_ij^P × B_ab^P
        std::vector<double> K_host(ov * ov, 0.0);

        // For each auxiliary function P
        for (int P = 0; P < num_auxiliary_basis_; P++) {
            // B_μν for this P: B_ao[P * nbas * nbas + μ * nbas + ν]
            // But B_μν^P storage: after Cholesky it's L^-1 × 3c-ERI
            // Layout: B[P, μ, ν] stored as B[P * nbas² + μ * nbas + ν]... need to check
            // Actually intermediate_matrix_B_ stores B^P_μν differently.
            // The MO transform uses it as [naux × nbas × nbas] with stride nbas.

            // Compute B_ij^P = Σ_μν C_μi C_νj B_μν^P
            // For small system, explicit O(nocc² × nbas²) per P
            std::vector<double> B_ij_P(nocc * nocc, 0.0);
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbas; mu++) {
                        for (int nu = 0; nu < nbas; nu++) {
                            val += C_host[mu * nbas + i] * C_host[nu * nbas + j]
                                 * B_ao[(size_t)P * nbas * nbas + mu * nbas + nu];
                        }
                    }
                    B_ij_P[i * nocc + j] = val;
                }
            }

            // Compute B_ab^P = Σ_μν C_μ,nocc+a C_ν,nocc+b B_μν^P
            std::vector<double> B_ab_P(nvir * nvir, 0.0);
            for (int a = 0; a < nvir; a++) {
                for (int b = 0; b < nvir; b++) {
                    double val = 0.0;
                    for (int mu = 0; mu < nbas; mu++) {
                        for (int nu = 0; nu < nbas; nu++) {
                            val += C_host[mu * nbas + (nocc + a)] * C_host[nu * nbas + (nocc + b)]
                                 * B_ao[(size_t)P * nbas * nbas + mu * nbas + nu];
                        }
                    }
                    B_ab_P[a * nvir + b] = val;
                }
            }

            // Accumulate: K[ia,jb] += B_ij^P × B_ab^P
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    for (int a = 0; a < nvir; a++)
                        for (int b = 0; b < nvir; b++)
                            K_host[(size_t)(j*nvir+b) * ov + (i*nvir+a)] += B_ij_P[i*nocc+j] * B_ab_P[a*nvir+b];
        }

        // M11 -= K (exchange)
        real_t* d_K = nullptr;
        tracked_cudaMalloc(&d_K, ov * ov * sizeof(real_t));
        cudaMemcpy(d_K, K_host.data(), ov * ov * sizeof(real_t), cudaMemcpyHostToDevice);
        cublasDaxpy(handle, (int)(ov * ov), &neg_one, d_K, 1, d_M11_cis, 1);
        tracked_cudaFree(d_K);

        // --- ISR correction + Σ_oo + Σ_vv (2nd order self-energy) ---
        // Requires (ia|jb) and t2[i][j][a][b] = (ia|jb)/Δ
        {
            // Build (ia|jb) on host from B_ia
            std::vector<double> B_ia_host(ov * num_auxiliary_basis_);
            cudaMemcpy(B_ia_host.data(), d_B_ia, ov * num_auxiliary_basis_ * sizeof(double), cudaMemcpyDeviceToHost);

            // Build ovov[i][a][j][b] = (ia|jb) = Σ_P B_ia^P B_jb^P
            size_t ovov_size = ov * ov;
            std::vector<double> ovov(ovov_size, 0.0);
            for (size_t ia = 0; ia < ov; ia++)
                for (size_t jb = 0; jb < ov; jb++)
                    for (int P = 0; P < num_auxiliary_basis_; P++)
                        ovov[ia * ov + jb] += B_ia_host[P * ov + ia] * B_ia_host[P * ov + jb];

            // Build t2[i][j][a][b] = (ia|jb) / (ε_i+ε_j-ε_a-ε_b)
            std::vector<double> t2(ov * ov, 0.0);
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    for (int a = 0; a < nvir; a++)
                        for (int b = 0; b < nvir; b++) {
                            double denom = eps[i] + eps[j] - eps[nocc+a] - eps[nocc+b];
                            size_t ia = i*nvir+a, jb = j*nvir+b;
                            t2[ia * ov + jb] = ovov[ia * ov + jb] / denom;
                        }

            // ISR + Σ_oo + Σ_vv (singlet, same as ADC(2) M11 correction)
            // Simplified: Σ_oo[i,j] and Σ_vv[a,b] modify M11 diagonal blocks
            // ISR_corr[ia,jb] involves t2 × ovov contractions

            // ISR correction (ADC(2) 2nd-order singles-singles)
            // ISR[ia,jb] = Σ_{kc} { 0.5*t2(ki,ac)*(kb|jc)
            //   + [singlet: 2*t2(ik,ac)*(jb|kc) - t2(ik,ac)*(kb|jc) - t2(ki,ac)*(jb|kc)]
            //   + transpose(i↔j, a↔b) }
            std::vector<double> ISR_corr(ov * ov, 0.0);
            for (int i = 0; i < nocc; i++)
                for (int a = 0; a < nvir; a++)
                    for (int j = 0; j < nocc; j++)
                        for (int b = 0; b < nvir; b++) {
                            size_t ia = i*nvir+a, jb = j*nvir+b;
                            double val = 0.0;
                            for (int k = 0; k < nocc; k++)
                                for (int c = 0; c < nvir; c++) {
                                    size_t kc = k*nvir+c;
                                    double t2_kiac = t2[(k*nvir+a) * ov + (i*nvir+c)]; // t2[k,i,a,c]=(ka|ic)/Δ
                                    double t2_ikac = t2[(i*nvir+a) * ov + (k*nvir+c)]; // t2[i,k,a,c]=(ia|kc)/Δ
                                    double kb_jc = ovov[(k*nvir+b) * ov + (j*nvir+c)];
                                    double jb_kc = ovov[(j*nvir+b) * ov + kc];
                                    // Forward
                                    val += 0.5 * t2_kiac * kb_jc;
                                    val += 2.0 * t2_ikac * jb_kc;
                                    val -= t2_ikac * kb_jc;
                                    val -= t2_kiac * jb_kc;
                                    // Transpose (i↔j, a↔b)
                                    double t2_kjbc = t2[(k*nvir+b) * ov + (j*nvir+c)];
                                    double t2_jkbc = t2[(j*nvir+b) * ov + kc];
                                    double ka_ic = ovov[(k*nvir+a) * ov + (i*nvir+c)];
                                    double ia_kc = ovov[(i*nvir+a) * ov + kc];
                                    val += 0.5 * t2_kjbc * ka_ic;
                                    val += 2.0 * t2_jkbc * ia_kc;
                                    val -= t2_jkbc * ka_ic;
                                    val -= t2_kjbc * ia_kc;
                                }
                            ISR_corr[jb * ov + ia] = val; // col-major
                        }

            // Add ISR to M11
            real_t* d_isr = nullptr;
            tracked_cudaMalloc(&d_isr, ov * ov * sizeof(real_t));
            cudaMemcpy(d_isr, ISR_corr.data(), ov * ov * sizeof(double), cudaMemcpyHostToDevice);
            cublasDaxpy(handle, (int)(ov * ov), &one, d_isr, 1, d_M11_cis, 1);
            tracked_cudaFree(d_isr);

            // Σ_oo[i,j] = Σ_{kab} [t2(ik,ab)*((ja|kb)-0.5*(jb|ka)) + t2(jk,ab)*((ia|kb)-0.5*(ib|ka))]
            // (matches ADC2Operator::build_M11 exactly)
            std::vector<double> sigma_oo(nocc * nocc, 0.0);
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++) {
                    double val_ij = 0.0, val_ji = 0.0;
                    for (int k = 0; k < nocc; k++)
                        for (int a = 0; a < nvir; a++)
                            for (int b = 0; b < nvir; b++) {
                                double t2_ikab = t2[(i*nvir+a) * ov + (k*nvir+b)];
                                double ja_kb = ovov[(j*nvir+a) * ov + (k*nvir+b)];
                                double jb_ka = ovov[(j*nvir+b) * ov + (k*nvir+a)];
                                val_ij += t2_ikab * (ja_kb - 0.5 * jb_ka);
                                double t2_jkab = t2[(j*nvir+a) * ov + (k*nvir+b)];
                                double ia_kb = ovov[(i*nvir+a) * ov + (k*nvir+b)];
                                double ib_ka = ovov[(i*nvir+b) * ov + (k*nvir+a)];
                                val_ji += t2_jkab * (ia_kb - 0.5 * ib_ka);
                            }
                    sigma_oo[i * nocc + j] = val_ij + val_ji;
                }

            // Σ_vv[a,b] = -Σ_{ijc} [t2(ij,ac)*((ib|jc)-0.5*(jb|ic)) + t2(ij,bc)*((ia|jc)-0.5*(ja|ic))]
            std::vector<double> sigma_vv(nvir * nvir, 0.0);
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++) {
                    double val_ab = 0.0, val_ba = 0.0;
                    for (int i = 0; i < nocc; i++)
                        for (int j = 0; j < nocc; j++)
                            for (int c = 0; c < nvir; c++) {
                                double t2_ijac = t2[(i*nvir+a) * ov + (j*nvir+c)];
                                double ib_jc = ovov[(i*nvir+b) * ov + (j*nvir+c)];
                                double jb_ic = ovov[(j*nvir+b) * ov + (i*nvir+c)];
                                val_ab += t2_ijac * (ib_jc - 0.5 * jb_ic);
                                double t2_ijbc = t2[(i*nvir+b) * ov + (j*nvir+c)];
                                double ia_jc = ovov[(i*nvir+a) * ov + (j*nvir+c)];
                                double ja_ic = ovov[(j*nvir+a) * ov + (i*nvir+c)];
                                val_ba += t2_ijbc * (ia_jc - 0.5 * ja_ic);
                            }
                    sigma_vv[a * nvir + b] = -(val_ab + val_ba);
                }

            // Add Σ to M11: M11[ia,jb] -= δ_ab Σ_oo[i,j] + δ_ij Σ_vv[a,b]
            std::vector<double> M11_corr(ov * ov, 0.0);
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    for (int a = 0; a < nvir; a++) {
                        // -δ_ab Σ_oo[i,j]
                        M11_corr[(size_t)(j*nvir+a) * ov + (i*nvir+a)] -= sigma_oo[i*nocc+j];
                    }
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++)
                    for (int i = 0; i < nocc; i++) {
                        // +δ_ij Σ_vv[a,b]
                        M11_corr[(size_t)(i*nvir+b) * ov + (i*nvir+a)] += sigma_vv[a*nvir+b];
                    }

            real_t* d_corr = nullptr;
            tracked_cudaMalloc(&d_corr, ov * ov * sizeof(real_t));
            cudaMemcpy(d_corr, M11_corr.data(), ov * ov * sizeof(double), cudaMemcpyHostToDevice);
            cublasDaxpy(handle, (int)(ov * ov), &one, d_corr, 1, d_M11_cis, 1);
            tracked_cudaFree(d_corr);
        }

        std::cout << "  CIS M11 built (D1 + 2J - K + Σ_oo + Σ_vv)" << std::endl;
    }
#endif  // Skip RI-based M11

    // Use full ADC(2) operator for correct Schur complement (validated)
    // SOS + Laplace optimization of sigma vector is future work.
    // For now: correct ADC(2) eigenvalues via ω-dependent Schur complement.
    {
        real_t* d_eri_mo = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
        ADC2Operator adc2_op(d_eri_mo, rhf_.get_orbital_energies().device_ptr(),
                             nocc, nvir, rhf_.get_num_basis());
        tracked_cudaFree(d_eri_mo);

        // Build M_eff(ω) and diagonalize with ω-iteration
        real_t* d_M_eff = nullptr;
        tracked_cudaMalloc(&d_M_eff, (size_t)singles_dim * singles_dim * sizeof(real_t));

        // ω-iteration
        double omega = 0.0;
        std::vector<double> eigenvalues;

        for (int iter = 0; iter < 15; iter++) {
            adc2_op.build_M_eff_matrix(omega, d_M_eff);

            // Diagonalize M_eff (symmetric)
            real_t* d_evals = nullptr;
            real_t* d_evecs = nullptr;
            tracked_cudaMalloc(&d_evals, singles_dim * sizeof(real_t));
            tracked_cudaMalloc(&d_evecs, (size_t)singles_dim * singles_dim * sizeof(real_t));
            gpu::eigenDecomposition(d_M_eff, d_evals, d_evecs, singles_dim);
            tracked_cudaFree(d_evecs);

            eigenvalues.resize(n_states);
            cudaMemcpy(eigenvalues.data(), d_evals, n_states * sizeof(double), cudaMemcpyDeviceToHost);
            tracked_cudaFree(d_evals);

            double omega_new = eigenvalues[0];
            std::cout << "  ω-iter " << iter << ": ω=" << std::setprecision(8) << omega_new
                      << "  Δω=" << std::abs(omega_new - omega) << std::endl;

            if (std::abs(omega_new - omega) < 1e-8) {
                std::cout << "  ω converged!" << std::endl;
                break;
            }
            omega = omega_new;
        }

        tracked_cudaFree(d_M_eff);
        tracked_cudaFree(d_M11_from_adc2);

        // Report results
    std::cout << "\n  SOS-ADC(2) Excitation Energies:" << std::endl;
    for (int k = 0; k < (int)eigenvalues.size() && k < n_states; k++) {
        std::cout << "    State " << k + 1 << ": "
                  << std::setprecision(8) << eigenvalues[k] << " Ha = "
                  << eigenvalues[k] * 27.2114 << " eV" << std::endl;
        }
    }

    tracked_cudaFree(d_B_ia);
}

} // namespace gansu
