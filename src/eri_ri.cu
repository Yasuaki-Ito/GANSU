/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
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

//#include <omp.h>


#include "rhf.hpp"

namespace gansu{

// // #threads = M * Mvir * Maux
__global__
void nu2a_(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a)
{
    long long seq = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (seq >= (long long)norbs * nvir * naux) {
        return;
    }

    const short p = seq / (norbs * nvir);
    seq %= (norbs * nvir);

    const int a = seq % nvir;
    const int mu = seq / nvir;

    double tmp = 0.0;
    for (short nu = 0; nu < norbs; ++nu) {
        tmp += d_C[norbs * nu + (a + nocc)] * d_B_p_mu_nu[p*(norbs*norbs) + mu*norbs + nu];
    }
    d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a] = tmp;
}


// #threads = Mocc * Mvir * Maux
__global__
void mu2i_(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a)
{
    long long seq = blockDim.x * (long long)blockIdx.x + threadIdx.x;
    if (seq >= (long long)nocc * nvir * naux) {
        return;
    }

    const short p = seq / (nocc * nvir);
    seq %= (nocc * nvir);

    const int a = seq % nvir;
    const int i = seq / nvir;

    double tmp = 0.0;
    for (short mu = 0; mu < norbs; ++mu) {
        tmp += d_C[norbs * mu + i] * d_B_p_mu_a[p*(norbs*nvir) + mu*nvir + a];
    }
    d_B_p_i_a[p*(nocc*nvir) + i*nvir + a] = tmp;
}


 void nu2a_dgemm(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a, cublasHandle_t &handle){
    // cublasManager cublas;
    // cublasHandle_t handle;
    // cublasCreate(&handle);

    // if(col_A != row_B) throw exception("行数と列数が不一致\n");

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double));
	

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


void mu2i_dgemm(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a, cublasHandle_t &handle){
    // cublasManager cublas;
    // cublasHandle_t handle;
    // cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_i_a, 0, norbs * norbs * naux * sizeof(double));

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

    cudaMemset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double));

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




void transform_intermediate_matrix(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B, double* d_tmp, cublasHandle_t &handle){
    nu2a_dgemm(norbs, nocc, nvir, naux, d_C, d_B, d_tmp, handle);
    mu2i_dgemm(norbs, nocc, nvir, naux, d_C, d_tmp, d_B, handle);
}






 void nu2a_dgemm(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_nu, double* d_B_p_mu_a){
    // cublasManager cublas;
    cublasHandle_t handle;
    cublasCreate(&handle);

    // if(col_A != row_B) throw exception("行数と列数が不一致\n");

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double));
	

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


void mu2i_dgemm(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B_p_mu_a, double* d_B_p_i_a){
    // cublasManager cublas;
    cublasHandle_t handle;
    cublasCreate(&handle);

    const double alpha = 1.0;
    const double beta = 0.0;

    cudaMemset(d_B_p_i_a, 0, norbs * norbs * naux * sizeof(double));

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

    cudaMemset(d_B_p_mu_a, 0, norbs * norbs * naux * sizeof(double));

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




void transform_intermediate_matrix(short norbs, short nocc, short nvir, short naux, double* d_C, double* d_B, double* d_tmp){
    nu2a_dgemm(norbs, nocc, nvir, naux, d_C, d_B, d_tmp);
    mu2i_dgemm(norbs, nocc, nvir, naux, d_C, d_tmp, d_B);
}








__device__ inline size_t2 index1to2_upper_wo_trace(const uint64_t index, const int n){
    size_t r2 = (2.0*n - 1.0 - sqrt((2.0*n - 1.0)*(2.0*n - 1.0) - 8.0*index)) / 2.0;
    size_t r1 = (r2*r2 - (2.0*n - 3.0)*r2 + 2.0*index) / 2.0 + 1.0;

    return {r2,r1};
}



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


    double iajb = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + ab.x*nocc*nvir + j*nvir + ab.y];
    double ibja = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + ab.y*nocc*nvir + j*nvir + ab.x];        
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

    double iaja = d_iajb[(i-nocc_stride)*nvir*nocc*nvir + a*nocc*nvir + j*nvir + a];     
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

    double iaib = d_iajb[i*nvir*nocc*nvir + a*nocc*nvir + (i+nocc_stride)*nvir + b];     
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

    double iaia = d_iajb[i*nvir*nocc*nvir + a*nocc*nvir + (i+nocc_stride)*nvir + a];     
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







int search_maximime_k(int mocc, int mvir) {
    size_t free_mem_bytes, total_mem_bytes;
    cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes);
    
    return std::min(free_mem_bytes/(mocc * mvir * mvir * sizeof(double)), (size_t)mocc);    
}






void search_k_and_cudamalloc_4cERI(int mocc, int mvir, int &k, double **d_iajb, cudaStream_t &stream) {
    k = search_maximime_k(mocc, mvir);
    // k = (int)(k*mvir / 32) * 32;
    // k = 10;

    while(cudaMallocAsync((void**)d_iajb, sizeof(double) * k * mvir * mocc * mvir, stream) != cudaSuccess){
        k *= 0.9;
    }

    // printf("k = %d\n",k);
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
    real_t *d_intermediate_matrix_B = intermediate_matrix_B_.device_ptr();
    const int num_auxiliary_basis = num_auxiliary_basis_;


    real_t* d_tmp;
    cudaMalloc((void**)&d_tmp, sizeof(double) * num_auxiliary_basis*num_basis_*num_basis_);

    double *d_energy;
    cudaMalloc((void**)&d_energy, sizeof(double));
    cudaMemset(d_energy, 0.0, sizeof(double));



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
    cudaFree(d_tmp);





    size_t num_blocks_3 = ((size_t)(nocc_block * nvir * (nvir - 1.0) / 2) + num_threads - 1) / num_threads, 
        num_blocks_4 = ((size_t)(nocc_block * nvir) + num_threads - 1) / num_threads;


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
        num_blocks_list[0].num_blocks = (((size_t)(nocc_block*i + nocc_block*(nocc_block-1)/2) *  nvir*(nvir-1)/2) + num_threads -1) / num_threads;
        num_blocks_list[1].num_blocks = (((size_t)(nocc_block*i + nocc_block*(nocc_block-1)/2) *  nvir) + num_threads - 1) / num_threads;

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

    cudaFree(d_iajb);            
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
    cudaFree(d_energy);


    for (int i = 0; i < 4; i++) cudaStreamDestroy(streams[i]);
    

    // events_for_sync
    for(int i=0; i<niter*4; i++) cudaEventDestroy(events_for_sync[i]);
    for(int i=0; i<2; i++) cudaEventDestroy(events[i]);

    return energy;
}



} // namespace gansu