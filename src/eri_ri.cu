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

} // namespace gansu
