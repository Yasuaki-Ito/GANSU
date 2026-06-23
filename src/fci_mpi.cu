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

// Multi-node / multi-GPU FCI (MPI + NCCL hybrid). The whole translation unit is
// compiled only in an MPI build (ENABLE_MPI -> GANSU_MPI); otherwise it is empty
// so CPU-only and single-GPU builds need no MPI/NCCL headers. The single public
// symbol fci_mpi() is dispatched from eri_stored_fci.cu when running on more than
// one rank; the proven single-GPU fci() (fci.cu) is used otherwise. All internal
// kernels/helpers live in an anonymous namespace to avoid link collisions with
// the identically named symbols in fci.cu.
#ifdef GANSU_MPI

#ifndef GANSU_CPU_ONLY
#include <cuda.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#endif
#include <omp.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <utility>
#include <iostream>
#include "gpu_manager.hpp"
#include "utils.hpp"
#include<sys/time.h>
#include <cusolverDn.h>

#include "mpi.h"
#include <cublas_v2.h>
#include <nccl.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "device_host_memory.hpp"
#include "fci.hpp"

#include <Eigen/Dense>

using gansu::gpu::gpu_available;

// --- Internal linkage: isolate all kernels/helpers from fci.cu (which defines
//     many identically named symbols). Only fci_mpi() below is external. ---
namespace {



#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                              \
  if (r != ncclSuccess) {                            \
    printf("NCCL error %s:%d '%s'\n",                \
           __FILE__, __LINE__, ncclGetErrorString(r)); \
    MPI_Abort(MPI_COMM_WORLD, -1);                   \
  }                                                  \
} while(0)

#define CUBLAS_CHECK(call)                                \
do {                                                      \
    cublasStatus_t status = call;                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                \
        fprintf(stderr,                                   \
                "CUBLAS error %s:%d: %d\n",               \
                __FILE__, __LINE__, status);               \
        MPI_Abort(MPI_COMM_WORLD, -1);                    \
    }                                                     \
} while (0)

#define CUDA_CHECK(call) \
do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d %s\n", \
      __FILE__, __LINE__, cudaGetErrorString(err)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while(0)

#define MPI_CHECK(call)                                      \
do {                                                         \
    int err = call;                                         \
    if (err != MPI_SUCCESS) {                               \
        char errstr[MPI_MAX_ERROR_STRING];                  \
        int sz;                                             \
        MPI_Error_string(err, errstr, &sz);                 \
        fprintf(stderr, "MPI error %s:%d: %s\n",            \
                __FILE__, __LINE__, errstr);                \
        MPI_Abort(MPI_COMM_WORLD, err);                     \
    }                                                        \
} while (0)

#define CUSOLVER_CHECK(call)                                      \
do {                                                              \
    cusolverStatus_t status = (call);                             \
    if (status != CUSOLVER_STATUS_SUCCESS) {                      \
        fprintf(stderr,                                          \
                "cuSOLVER error %s:%d code=%d\n",                 \
                __FILE__, __LINE__, status);                      \
        cudaDeviceReset();                                        \
        MPI_Abort(MPI_COMM_WORLD, -1);                            \
    }                                                             \
} while (0)



__global__ void FCImake_hdiag_uhf_part_kernel_large(
    double *hdiag,
    size_t size,
    const double *h1e,
    const double *jdiag,
    const double *kdiag,
    int32_t norb,
    int32_t nstra,
    int32_t nstrb,
    int32_t starta,
    int32_t nocc,
    const int32_t *occslist,
    int rank)
{
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = tid; i < size; i += stride) {

        int32_t ia = (int32_t)(i / nstrb);
        int32_t ib = (int32_t)(i % nstrb);

        if (ia + starta >= nstrb) continue;

        const int32_t *paocc = occslist + (size_t)(ia + starta) * nocc;
        const int32_t *pbocc = occslist + (size_t)ib * nocc;

        double e1 = 0.0, e2 = 0.0;

        for (int32_t j0 = 0; j0 < nocc; j0++) {
            int32_t j  = paocc[j0];
            int32_t jb = pbocc[j0];

            e1 += h1e[(size_t)j  * norb + j]
                + h1e[(size_t)jb * norb + jb];

            for (int32_t k0 = 0; k0 < nocc; k0++) {
                int32_t jk;

                jk = j * norb + paocc[k0];
                e2 += jdiag[jk] - kdiag[jk];

                jk = j * norb + pbocc[k0];
                e2 += 2.0 * jdiag[jk];

                jk = jb * norb + pbocc[k0];
                e2 += jdiag[jk] - kdiag[jk];
            }
        }

        hdiag[(size_t)ia * nstrb + ib] = e1 + 0.5 * e2;
    }
}



void computeEigenvaluesAndVectorsn(cusolverDnHandle_t cusolverH, int32_t N, double* d_A, double* values, double* vectors, int* devInfo, double* d_W) {
    double *d_work = NULL;
    int lwork = 0;

    // Query workspace size
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_A, N, d_W, &lwork));
    //printf("lwork = %d\n", lwork);
    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
    

    CUSOLVER_CHECK(cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, N, d_A, N, d_W, d_work, lwork, devInfo));
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(vectors, d_A, sizeof(double) * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values, d_W, sizeof(double) * 1, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_work));
}


__global__ void f1e_kernel(double* f1e, double* eri, double* h1e, int32_t norb, int32_t d2, int32_t d3, double norm_factor){
      int32_t jk=blockIdx.x * blockDim.x + threadIdx.x;
      int32_t j=jk/norb;
      int32_t k=jk%norb;
      double sum = 0.0;
      for (int32_t i = 0; i < norb; i++) {
                sum += eri[j * d3 + i * d2 + i * norb + k];
      }
      f1e[j * norb + k] = (h1e[j * norb + k] - 0.5 * sum)*norm_factor;
}
__global__ void adderi_kernel(double* f1e, double* eri, int32_t norb, int32_t d2, int32_t d3){
       int32_t ijk=blockIdx.x * blockDim.x + threadIdx.x;
       int32_t i = ijk/d2;
       int32_t j = (ijk-i*d2)/norb;
       int32_t k = (ijk-i*d2)%norb;
       //printf("norb:%d, ijk:%d, i:%d, j:%d, k:%d\n", norb, ijk, i, j, k);
       atomicAdd(&eri[k * d3 + k * d2 + i * norb + j], f1e[i * norb + j]);
       atomicAdd(&eri[i * d3 + j * d2 + k * norb + k], f1e[i * norb + j]);
}

__global__ void nr1to4_kernel(double* eri1, double* eri4, int32_t norb, int32_t d1, int32_t d2, int32_t d3, size_t npair, double fac)
{
        int32_t idx=blockIdx.x * blockDim.x + threadIdx.x;
        //for (int idx=0; idx<npair*npair; idx++){
        int32_t i, j, k, l, ij, kl;
        ij = idx / npair;
        kl = idx % npair;
        i = (int32_t)((sqrt(8.0 * ij + 1) - 1) / 2);
        j = ij - i * (i + 1) / 2;
        k = (int32_t)((sqrt(8.0 * kl + 1) - 1) / 2);
        l = kl - k * (k + 1) / 2;
        eri4[ij*npair+kl] = eri1[i*d3+j*d2+k*d1+l]*fac;
        //}
}

void absorb_h1e(double* d_h1e,  double* d_erio, double* d_eri, int32_t norb, int32_t nelec, int32_t nnorb, double fac) {
    //double* f1e = (double*)malloc(norb * norb * sizeof(double));
    int32_t d2 = norb * norb;
    int32_t d3 = norb * norb * norb;
    double norm_factor = 1.0 / (nelec + 1e-100);
    double *d_f1e;
    cudaMalloc((void **)&d_f1e, d2* sizeof(double));
    f1e_kernel<<<d2, 1>>>(d_f1e, d_erio, d_h1e, norb, d2, d3, norm_factor);
    adderi_kernel<<<d3, 1>>>(d_f1e, d_erio, norb, d2, d3);
    
    nr1to4_kernel<<<nnorb*nnorb, 1>>>(d_erio, d_eri, norb, norb, d2, d3, nnorb, fac);

    //free(f1e);
    cudaFree(d_f1e);
}


__device__ inline void sort_small_int64(int32_t* arr, int32_t n) {
    // insertion sort
    for (int32_t i = 1; i < n; i++) {
        int32_t key = arr[i];
        int32_t j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Global function (kernel) for the GPU
__global__ void propgate1e_kernel(
    int32_t  nelec,        int32_t  norb,         int32_t  na,
    int      amin,         int      nstring,       int32_t *d_link_index,
    int32_t *occslst,      int32_t *d_link_nnorb,  int32_t  nvir,
    int32_t  nlink,
    int32_t *d_scratch  // size: nstring * (nvir * 2 + nvir * nelec) * sizeof(int32_t)
)
{
    const int32_t id = blockIdx.x * blockDim.x + threadIdx.x + amin;
    if (id >= na + amin) return;

    // Each thread gets its own scratch memory block
    const int32_t per_thread_scratch = nvir * 2 + nvir * nelec;  // in int32_t units
    int32_t *scratch = &d_scratch[id * per_thread_scratch];

    int32_t *vir      = scratch;
    int32_t *where_vir = scratch + nvir;
    // str1buf: nvir arrays of length nelec, used as nvir * nelec scratch space
    int32_t *str1buf  = scratch + nvir * 2;

    int32_t  parity_occ_orb = 1;
    int32_t *str0 = &occslst[id * nelec];
    int32_t  a, idx;
    int32_t  i, j, n;

    // -----------------
    // Compute virtual orbital list vir[]
    // -----------------
    int32_t j_vir = 0;
    for (i = 0; i < norb; ++i) {
        bool found = false;
        for (n = 0; n < nelec; ++n) {
            if (str0[n] == i) { found = true; break; }
        }
        if (!found) vir[j_vir++] = i;
    }

    // -----------------
    // Compute where_vir[]: number of occupied orbitals below each virtual
    // -----------------
    for (i = 0; i < nvir; ++i) {
        where_vir[i] = 0;
        for (j = 0; j < nelec; ++j) {
            if (str0[j] < vir[i]) where_vir[i]++;
        }
    }

    // -----------------
    // Fill diagonal entries of link_index (i -> i transitions)
    // -----------------
    for (i = 0; i < nelec; ++i) {
        a = str0[i];
        a = a * (a + 1) / 2 + a;
        d_link_index[(id * nlink + i) * 3 + 0] = a;
        d_link_index[(id * nlink + i) * 3 + 1] = id;
        d_link_index[(id * nlink + i) * 3 + 2] = 1;
    }

    // -----------------
    // Loop over electrons: build excitation entries in link_index
    // -----------------
    for (n = 0; n < nelec; ++n) {

        // Build excited strings str1buf[i] = str0 with str0[n] -> vir[i]
        for (i = 0; i < nvir; ++i) {
            int32_t *s1 = str1buf + i * nelec;
            for (int32_t k = 0; k < nelec; ++k) s1[k] = str0[k];
            s1[n] = vir[i];
            sort_small_int64(s1, nelec);
        }

        for (i = 0; i < nvir; ++i) {
            int32_t *s1 = str1buf + i * nelec;

            // Compute parity of the excitation str0[n] -> vir[i]
            const int32_t comp   = (vir[i] > str0[n]) ? 1 : 0;
            const int32_t sum    = where_vir[i] + comp + 1;
            const int32_t parity = (sum % 2 == 0 ? -1 : 1) * parity_occ_orb;

            // Search for the excited string s1 in occslst
            int32_t s_index = -1;
            for (int32_t k = 0; k < nstring; ++k) {
                bool eq = true;
                for (int32_t x = 0; x < nelec; ++x) {
                    if (s1[x] != occslst[k * nelec + x]) { eq = false; break; }
                }
                if (eq) { s_index = k; break; }
            }

            // Write excitation entry: (orbital pair index, target string, parity)
            const int32_t pos = nelec + n * nvir + i;
            a   = vir[i];
            idx = str0[n];
            a   = max(a * (a + 1) / 2 + idx, idx * (idx + 1) / 2 + a);
            d_link_index[(id * nlink + pos) * 3 + 0] = a;
            d_link_index[(id * nlink + pos) * 3 + 1] = s_index;
            d_link_index[(id * nlink + pos) * 3 + 2] = parity;
        }

        parity_occ_orb *= -1;
    }

    // -----------------
    // Build d_link_nnorb from d_link_index
    // -----------------
    for (int32_t rlink = 0; rlink < nlink; ++rlink) {
        const int32_t flat   = id * nlink + rlink;
        const int32_t ia_aj  = d_link_index[flat * 3];
        const size_t  id_st  = (size_t)ia_aj * (size_t)nstring + (size_t)id;
        d_link_nnorb[2 * id_st]     = d_link_index[flat * 3 + 1];
        d_link_nnorb[2 * id_st + 1] = d_link_index[flat * 3 + 2];
    }
}


void gen_linkstr_index(int32_t nelec, int32_t norb, int32_t nstring, int amin, int na, int32_t* d_occslst,  int32_t* d_link_index, int32_t* d_link_nnorb) {

    int32_t nvir = norb - nelec;
    int32_t nlink = nelec + nelec * nvir;
    int32_t  *d_scratch;
    int32_t threads = 512;
    int blocks = (nstring + threads - 1) / threads;
    size_t per_thread_scratch_size = (nvir * 2 + nvir * nelec); // in int32_t units
    
    size_t total_scratch_bytes = na * per_thread_scratch_size * sizeof(int32_t);
    

    CUDA_CHECK(cudaMalloc((void **)&d_scratch, total_scratch_bytes));
    if (total_scratch_bytes == 0) {
        fprintf(stderr, "ERROR: total_scratch_bytes == 0\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    CUDA_CHECK(cudaMemset(d_scratch, 0, total_scratch_bytes));
    
    propgate1e_kernel<<<blocks, threads>>>(nelec, norb, nstring, amin, na,
        d_link_index,  d_occslst,  d_link_nnorb, nvir, nlink, d_scratch);

    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaFree(d_scratch));
}




__global__ void Dcopy_kernel(double *in, double *out, int heff_size, int space){
     int32_t id=blockIdx.x * blockDim.x + threadIdx.x;
     int32_t j=id/space;
     out[id] = in[j*heff_size+id%space];     
}

__global__ void dot_partial_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ partial,
    size_t n)
{
    __shared__ double sdata[256];
    if (threadIdx.x == 0 && blockDim.x != 256) {
        printf("ERROR: blockDim.x = %d\n", blockDim.x);
    }
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + tid;
    size_t stride = gridDim.x * blockDim.x;

    double sum = 0.0;
    for (size_t i = idx; i < n; i += stride) {
        sum += a[i] * b[i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        partial[blockIdx.x] = sdata[0];
}

void dot_func(double* d_partial, double* h_partial, double* d_ci0, double* d_ci1, double* d_inner_local, size_t mynp, int blocks, int rank, cudaStream_t computeStream){
    dot_partial_kernel<<<blocks, 256, 0, computeStream>>>(
        d_ci0, d_ci1, d_partial, mynp
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Use asynchronous copy with stream
    CUDA_CHECK(cudaMemcpyAsync(h_partial, d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    
    double result = 0.0;
    for (int i = 0; i < blocks; i++)
        result += h_partial[i];
    
    // Use asynchronous copy with stream
    CUDA_CHECK(cudaMemcpyAsync(d_inner_local, &result, sizeof(double), cudaMemcpyHostToDevice, computeStream));
}

void dot_func_h(double* d_partial, double* h_partial, const double* __restrict__ d_ci0, const double* __restrict__ d_ci1, double* h_local, size_t mynp, int blocks, int rank, cudaStream_t computeStream){
    dot_partial_kernel<<<blocks, 256, 0, computeStream>>>(
        d_ci0, d_ci1, d_partial, mynp
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Use asynchronous copy with stream
    CUDA_CHECK(cudaMemcpyAsync(h_partial, d_partial, blocks * sizeof(double), cudaMemcpyDeviceToHost, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    
    double result = 0.0;
    for (int i = 0; i < blocks; i++)
        result += h_partial[i];
    h_local[0] = result;
}


__global__ void write_heff(double* d_heff_tmp, double* val,
                           int r, int c, int ld)
{
    if (threadIdx.x == 0)
        d_heff_tmp[r * ld + c] = *val;
        d_heff_tmp[c * ld + r] = *val;
}



void fill_heff_hermitian_gpu_fast(cublasHandle_t handle, ncclComm_t ncclComm, cudaStream_t computeStream, double* d_heff_tmp, double* d_heff, double* d_ci0, double* d_ci1, double* d_ci1_list, double* h_ci1_list, double* d_tmp, double* d_inner_global, double* d_inner_local, double* d_partial, double* h_partial, int32_t row1, int32_t nrow, int32_t heff_size, int nprocs, int in_cpu, size_t mynp, int blocks, int rank){
    
    int32_t row0 = row1 - nrow;
    
    // Compute diagonal element heff[row0, row0]
    dot_func(d_partial, h_partial, d_ci0, d_ci1, d_inner_local, mynp, blocks, rank, computeStream);
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    if (nprocs > 1) {
        NCCL_CHECK(ncclAllReduce(d_inner_local, d_inner_global, 1, ncclDouble, ncclSum, ncclComm, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        write_heff<<<1, 1, 0, computeStream>>>(d_heff_tmp, d_inner_global, row0, row0, heff_size);
    } else {
        write_heff<<<1, 1, 0, computeStream>>>(d_heff_tmp, d_inner_local, row0, row0, heff_size);
    }
    CUDA_CHECK(cudaGetLastError());

    // Compute off-diagonal elements
    for (int32_t i = 0; i < row0; i++) {
        if (in_cpu == 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_tmp, h_ci1_list + i * mynp, sizeof(double) * mynp, cudaMemcpyHostToDevice, computeStream));
            dot_func(d_partial, h_partial, d_ci0, d_tmp, d_inner_local, mynp, blocks, rank, computeStream);
        } else {
            dot_func(d_partial, h_partial, d_ci0, d_ci1_list + i * mynp, d_inner_local, mynp, blocks, rank, computeStream);
        }   
        
        if (nprocs > 1) {
            NCCL_CHECK(ncclAllReduce(d_inner_local, d_inner_global, 1, ncclDouble, ncclSum, ncclComm, computeStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            write_heff<<<1, 1, 0, computeStream>>>(d_heff_tmp, d_inner_global, row0, i, heff_size);
        } else {
            write_heff<<<1, 1, 0, computeStream>>>(d_heff_tmp, d_inner_local, row0, i, heff_size);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Copy to final heff matrix
    Dcopy_kernel<<<row1 * row1, 1, 0, computeStream>>>(d_heff_tmp, d_heff, heff_size, row1);
    CUDA_CHECK(cudaGetLastError());
}



__global__ void Dscal_kernel(double *in, double *out, double k, size_t np)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t id = tid; id < np; id += stride) {
        out[id] = k * in[id];
    }
}
__global__ void Dscalplus_kernel(double *in, double *out, double k, size_t np)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t id = tid; id < np; id += stride) {
        out[id] += k * in[id];
    }
}


__global__ void Dscal_accum_kernel(const double* __restrict__ d_ci0_list, const double* __restrict__ d_ci1_list, double* __restrict__ d_ci0, double* __restrict__ d_ci1, const  double* __restrict__ v, int space, size_t np)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= np) return;
    double x = d_ci0[id] * v[space - 1];
    double y = d_ci1[id] * v[space - 1];

    #pragma unroll
    for (int i = 0; i < space - 1; ++i){
        x += v[i] * d_ci0_list[i * np + id];
        y += v[i] * d_ci1_list[i * np + id];
    }
    d_ci0[id] = x;
    d_ci1[id] = y;
}

void gen_x0_gpu(double *v, double *d_ci0_list,  double *d_ci1_list, double *d_ci0, double *d_ci1, int space, size_t np, cudaStream_t computeStream)
{
    const int threads = 256;
    const int blocks  = (int)((np + threads - 1) / threads);

    // BUG FIX: use cudaMallocAsync and cudaMemcpyAsync to stay on computeStream
    double *d_v = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_v, space * sizeof(double), computeStream));
    CUDA_CHECK(cudaMemcpyAsync(d_v, v, space * sizeof(double),
                               cudaMemcpyHostToDevice, computeStream));

    Dscal_accum_kernel<<<blocks, threads, 0, computeStream>>>(
        d_ci0_list, d_ci1_list, d_ci0, d_ci1, d_v, space, np);
    CUDA_CHECK(cudaGetLastError());

    // free after kernel completes 
    CUDA_CHECK(cudaFreeAsync(d_v, computeStream));
}


void gen_x0_gpu_incpu(double *h_v, double *h_c_list, double *d_x0, double* d_tmp, int space, size_t np, cudaStream_t computeStream){
    size_t nthread = 256;
    size_t max_blocks = 65535; 
    const size_t nblock = (np + nthread - 1) / nthread < max_blocks
                        ? (np + nthread - 1) / nthread : max_blocks;

    // Initialize with the last term
    CUDA_CHECK(cudaMemcpyAsync(d_tmp, h_c_list + (space - 1) * np, sizeof(double) * np, cudaMemcpyHostToDevice, computeStream));
    Dscal_kernel<<<nblock, nthread, 0, computeStream>>>(d_tmp, d_x0, h_v[space - 1], np);
    
    // Accumulate remaining terms
    for (int i = space - 2; i >= 0; i--) {
        CUDA_CHECK(cudaMemcpyAsync(d_tmp, h_c_list + i * np, sizeof(double) * np, cudaMemcpyHostToDevice, computeStream));
        Dscalplus_kernel<<<nblock, nthread, 0, computeStream>>>(d_tmp, d_x0, h_v[i], np);
    }
}



__global__ void precond_kernel(const double* __restrict__ diag, double* __restrict__ dx, double e, double level_shift, size_t np){
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    
    const double threshold = 1e-8;
    const double shift = e - level_shift;
    
    for (size_t i = tid; i < np; i += stride) {
        double diag_val = diag[i] - shift;
        double abs_diag_val = fabs(diag_val);
        dx[i] /= (abs_diag_val < threshold) ? threshold : diag_val;
    }     
}


__global__ void Dscalminus_kernel(double *in, double *out, double k,  size_t np){
     size_t id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np) return;
     out[id]-=k*in[id];
}
__global__ void Ddiv_kernel(double *in, double *out, double k,  size_t np){
     size_t id=blockIdx.x * blockDim.x + threadIdx.x;
     if (id>=np) return;
     out[id]=in[id]/k;
}
__global__ void Dscalminus_largekernel(const double* __restrict__ in, double* __restrict__ out, double k, size_t np){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t id = tid; id < np; id += stride) {
        out[id] -= k * in[id];
    }
}

__global__ void Ddiv_largekernel(const double* __restrict__ in, double* __restrict__ out, double k, size_t np){
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t id = tid; id < np; id += stride) {
        out[id] = in[id] / k;
    }
}


void normalize_xt_gpu_large(
    cublasHandle_t handle,   double *d_ci0,      double *d_ci0_list,
    double        *d_tmp,    double  lindep,      double  norm_min,
    int            space,    size_t  np,          int     nprocs,
    ncclComm_t     ncclComm, int     rank,        size_t  mynp,
    double        *d_single, double *d_global,    double *h_single,
    int            in_cpu,   int     blocks_dot,  double *d_partial,
    double        *h_partial, cudaStream_t computeStream)
{
    const size_t nthread    = 256;
    const size_t max_blocks = 65535;
    const size_t nblock     = (mynp + nthread - 1) / nthread < max_blocks
                              ? (mynp + nthread - 1) / nthread : max_blocks;

    for (int i = 0; i < space; i++) {

        // Compute dot product <ci0_list[i] | ci0>
        if (in_cpu == 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_tmp, &d_ci0_list[i * mynp],
                                       mynp * sizeof(double),
                                       cudaMemcpyHostToDevice, computeStream));
            dot_func_h(d_partial, h_partial, d_tmp, d_ci0,
                       h_single, mynp, blocks_dot, rank, computeStream);
        } else {
            dot_func_h(d_partial, h_partial, d_ci0_list + i * mynp, d_ci0,
                       h_single, mynp, blocks_dot, rank, computeStream);
        }
        CUDA_CHECK(cudaStreamSynchronize(computeStream));

        // AllReduce across processes
        if (nprocs > 1) {
            CUDA_CHECK(cudaMemcpy(d_single, h_single, sizeof(double),
                                  cudaMemcpyHostToDevice));
            NCCL_CHECK(ncclAllReduce(d_single, d_global, 1, ncclDouble,
                                     ncclSum, ncclComm, computeStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            CUDA_CHECK(cudaMemcpy(h_single, d_global, sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        // Subtract projection: ci0 -= tmp * ci0_list[i]
        const double tmp = h_single[0];
        if (in_cpu == 1) {
            CUDA_CHECK(cudaMemcpyAsync(d_tmp, &d_ci0_list[i * mynp],
                                       mynp * sizeof(double),
                                       cudaMemcpyHostToDevice, computeStream));
            Dscalminus_largekernel<<<nblock, nthread, 0, computeStream>>>(
                d_tmp, d_ci0, tmp, mynp);
        } else {
            Dscalminus_largekernel<<<nblock, nthread, 0, computeStream>>>(
                d_ci0_list + i * mynp, d_ci0, tmp, mynp);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    // Normalize ci0
    dot_func_h(d_partial, h_partial, d_ci0, d_ci0,
               h_single, mynp, blocks_dot, rank, computeStream);
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    if (nprocs > 1) {
        CUDA_CHECK(cudaMemcpy(d_single, h_single, sizeof(double),
                              cudaMemcpyHostToDevice));
        NCCL_CHECK(ncclAllReduce(d_single, d_global, 1, ncclDouble,
                                 ncclSum, ncclComm, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaMemcpy(h_single, d_global, sizeof(double),
                              cudaMemcpyDeviceToHost));
    }

    const double norm_sq = h_single[0];
    if (norm_sq > lindep && sqrt(norm_sq) > 1e-14) {
        const double norm = sqrt(norm_sq);
        Ddiv_largekernel<<<nblock, nthread, 0, computeStream>>>(
            d_ci0, d_ci0, norm, mynp);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }
}

void normalize_xt_gpu(
    cublasHandle_t handle,  double *d_ci0,     double *d_ci0_list,
    double        *d_tmp,   double  lindep,     double  norm_min,
    int            space,   size_t  np,         int     nprocs,
    ncclComm_t     ncclComm, int    rank,       size_t  mynp,
    double        *d_single, double *d_global,  int     in_cpu,
    cudaStream_t   computeStream)
{
    const int    nthread = 256;
    const size_t nblock  = (mynp + nthread - 1) / nthread;

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    for (int i = 0; i < space; i++) {

        // Compute dot product <ci0_list[i] | ci0>
        if (in_cpu == 1) {
            CUDA_CHECK(cudaMemcpy(d_tmp, &d_ci0_list[i * mynp],
                                  mynp * sizeof(double), cudaMemcpyHostToDevice));
            CUBLAS_CHECK(cublasDdot(handle, mynp, d_tmp, 1, d_ci0, 1, d_single));
        } else {
            CUBLAS_CHECK(cublasDdot(handle, mynp, d_ci0_list + i * mynp, 1,
                                    d_ci0, 1, d_single));
        }

        // AllReduce across processes
        if (nprocs > 1) {
            NCCL_CHECK(ncclAllReduce(d_single, d_global, 1, ncclDouble,
                                     ncclSum, ncclComm, computeStream));
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        } else {
            CUDA_CHECK(cudaMemcpy(d_global, d_single, sizeof(double),
                                  cudaMemcpyDeviceToDevice));
        }

        // Copy result to host
        double tmp;
        CUDA_CHECK(cudaMemcpy(&tmp, d_global, sizeof(double), cudaMemcpyDeviceToHost));

        // Subtract projection: ci0 -= tmp * ci0_list[i]
        if (in_cpu == 1) {
            Dscalminus_kernel<<<nblock, nthread, 0, computeStream>>>(
                d_tmp, d_ci0, tmp, mynp);
        } else {
            Dscalminus_kernel<<<nblock, nthread, 0, computeStream>>>(
                d_ci0_list + i * mynp, d_ci0, tmp, mynp);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    }

    // Normalize ci0
    CUBLAS_CHECK(cublasDdot(handle, mynp, d_ci0, 1, d_ci0, 1, d_single));

    if (nprocs > 1) {
        NCCL_CHECK(ncclAllReduce(d_single, d_global, 1, ncclDouble,
                                 ncclSum, ncclComm, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
    } else {
        CUDA_CHECK(cudaMemcpy(d_global, d_single, sizeof(double),
                              cudaMemcpyDeviceToDevice));
    }

    double norm_sq;
    CUDA_CHECK(cudaMemcpy(&norm_sq, d_global, sizeof(double), cudaMemcpyDeviceToHost));

    if (norm_sq > lindep) {
        const double norm = sqrt(norm_sq);
        Ddiv_kernel<<<nblock, nthread, 0, computeStream>>>(d_ci0, d_ci0, norm, mynp);
        CUDA_CHECK(cudaGetLastError());
    }
}


__global__ void _build_t1(double *ci0, double *t1,
    int32_t strb0, int32_t na, int32_t nb, int32_t nnorb,
    int32_t *d_linknn,  int32_t chunk, int32_t nab)
{
    int32_t thread_id=blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id>=na*chunk){
	    return;
    }
    int32_t tx = thread_id%chunk;
    int32_t stra = thread_id/chunk ;
    int32_t strb = strb0 + tx;
    int32_t ab_id = stra * chunk + tx;
    int8_t signa, signb;
    int32_t str1a, str1b, j;
  
    if (stra < na && strb < nb) {
            for (j = 0; j < nnorb; j++) {
                str1a = d_linknn[2*(j*na+stra)];
                signa = d_linknn[2*(j*na+stra)+1];

                str1b = d_linknn[2*(j*na+strb)];
                signb = d_linknn[2*(j*na+strb)+1];
                t1[j*nab + ab_id] = signa * ci0[str1a*nb+strb] + signb * ci0[stra*nb+str1b];
            }
    }
}

__global__ void _gather(double *out, double *t1,
    int32_t strb0, int32_t na, int32_t nb, int32_t nnorb,
    int32_t chunk, int32_t nab, int32_t *d_clink_index, int32_t nlink)
{
    int32_t thread_id=blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id>=na*chunk) return;
    int32_t tx = thread_id%chunk;
    int32_t stra = thread_id/chunk ;
    int32_t strb = strb0 + tx;
    int32_t ab_id = stra * chunk + tx;
    int32_t str1, j, ia;
    double val = 0.;
    int32_t *tabb = d_clink_index + strb * nlink*3;
    int32_t *taba = d_clink_index + stra * nlink*3;
    int8_t signa, signb;
    

    if (stra < na && strb < nb) {
        for (j = 0; j < nlink; j++) {
            ia = taba[j*3];
            str1 = taba[j*3+1];
            signa = taba[j*3+2];
            val += signa * t1[ia*nab + (str1*chunk+tx)];

            ia = tabb[j*3];
            str1 = tabb[j*3+1];
            signb = tabb[j*3+2];
            atomicAdd(&out[stra*nb+str1], signb * t1[ia*nab + ab_id]);
        }
        atomicAdd(&out[stra*nb+strb], val);
    }

}


void contract_2e_spin1_gpu(
    cublasHandle_t handle,
    double        *d_eri,
    double        *d_ci0,    double  *d_ci1,
    double        *d_t1,     double  *d_vt1,
    int32_t        norb,     int32_t  na,
    int32_t        nb,       int32_t  nlink,
    int32_t       *d_clink,  int32_t *d_linknn,
    int32_t        na_self,
    int32_t        chunk,    int      debug_mode,
    cudaStream_t   computeStream)
{

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetStream(handle, computeStream));

    const int32_t nnorb          = norb * (norb + 1) / 2;
    const double  D0             = 0.0, D1 = 1.0;
    const int32_t nab            = na_self * chunk;
    const int32_t threadsPerBlock = 256;
    const int32_t nblocks        = (nab + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemsetAsync(d_ci1, 0, (size_t)na_self * nb * sizeof(double), computeStream));

    // CUDA Event for timing (debug_mode > 1 only)
    cudaEvent_t start1, stop1;
    if (debug_mode > 1) {
        CUDA_CHECK(cudaEventCreate(&start1));
        CUDA_CHECK(cudaEventCreate(&stop1));
        CUDA_CHECK(cudaEventRecord(start1, computeStream));
    }

    for (int32_t strb0 = 0; strb0 < nb; strb0 += chunk) {
        _build_t1<<<nblocks, threadsPerBlock, 0, computeStream>>>(
            d_ci0, d_t1, strb0, na, nb, nnorb, d_linknn, chunk, nab);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            nab, nnorb, nnorb, &D1, d_t1, nab, d_eri, nnorb, &D0, d_vt1, nab));

        _gather<<<nblocks, threadsPerBlock, 0, computeStream>>>(
            d_ci1, d_vt1, strb0, na, nb, nnorb, chunk, nab, d_clink, nlink);
        CUDA_CHECK(cudaGetLastError());
    }

    if (debug_mode > 1) {
        CUDA_CHECK(cudaEventRecord(stop1, computeStream));
        CUDA_CHECK(cudaEventSynchronize(stop1));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start1, stop1));
        printf("contract_2e_spin1_gpu time: %.3f ms\n", elapsed_ms);
        CUDA_CHECK(cudaEventDestroy(start1));
        CUDA_CHECK(cudaEventDestroy(stop1));
    }
    
}



__global__ void _build_t1_buf(
    const double* __restrict__ ci0,
    const double* __restrict__ ci0_buf,
    double* __restrict__ t1,
    const int32_t* __restrict__ d_link_nnorb,
    const int32_t* __restrict__ natomax,
    int32_t strb0,
    int32_t na,
    int32_t nb,
    int32_t nnorb,
    int32_t ntile,
    int32_t ntile2,
    int32_t amin,
    int32_t amax,
    int32_t na_self,
    int32_t nab,
    int mode,
    int rank)
{
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ntile_sz = (size_t)ntile;
    const size_t ntile2_sz = (size_t)ntile2;
    const size_t total_threads = (size_t)na_self * ntile2_sz;
    
    if (thread_id >= total_threads) return;
    
    const size_t tx = thread_id % ntile2_sz;
    const size_t stra_base = thread_id / ntile2_sz;
    
    // Early bounds checking
    if (stra_base >= na_self || tx >= ntile2_sz) return;
    
    const size_t strb = (size_t)strb0 + tx;
    const size_t stra = stra_base + (size_t)amin;
    
    // Bounds checking
    if (stra >= amax || strb >= nb) return;
    
    // Pre-calculate commonly used values
    const size_t ab_id = stra_base * ntile_sz + tx;
    const size_t nab_sz = (size_t)nab;
    const size_t na_sz = (size_t)na;
    const size_t nb_sz = (size_t)nb;
    const size_t stra_base_nb = stra_base * nb_sz;
    
    // Optimize based on mode to avoid branching inside the loop
    if (mode == 0) {
        // Mode 0: direct indexing
        //#pragma unroll 4
        for (int32_t j = 0; j < nnorb; j++) {
            const size_t link_idx_a = 2 * (j * na_sz + stra);
            const size_t link_idx_b = 2 * (j * na_sz + strb);
            
            const size_t str1a = (size_t)d_link_nnorb[link_idx_a];
            const int8_t signa = (int8_t)d_link_nnorb[link_idx_a + 1];
            const size_t str1b = (size_t)d_link_nnorb[link_idx_b];
            const int8_t signb = (int8_t)d_link_nnorb[link_idx_b + 1];
            
            const double term1 = (double)signa * ci0_buf[str1a * ntile_sz + tx];
            const double term2 = (double)signb * ci0[stra_base_nb + str1b];
            
            t1[j * nab_sz + ab_id] = term1 + term2;
        }
    } else {
        // Mode 1: with natomax mapping
        //#pragma unroll 4
        for (int32_t j = 0; j < nnorb; j++) {
            const size_t link_idx_a = 2 * (j * na_sz + stra);
            const size_t link_idx_b = 2 * (j * na_sz + strb);
            
            size_t str1a = (size_t)d_link_nnorb[link_idx_a];
            const int8_t signa = (int8_t)d_link_nnorb[link_idx_a + 1];
            const size_t str1b = (size_t)d_link_nnorb[link_idx_b];
            const int8_t signb = (int8_t)d_link_nnorb[link_idx_b + 1];
            
            // Apply mapping for str1a
            str1a = (size_t)natomax[str1a];
            
            const double term1 = (double)signa * ci0_buf[str1a * ntile_sz + tx];
            const double term2 = (double)signb * ci0[stra_base_nb + str1b];
            
            t1[j * nab_sz + ab_id] = term1 + term2;
        }
    }
}

__global__ void _gather_buf(
    double* __restrict__ out,
    double* __restrict__ ci1_buf,
    double* __restrict__ d_cbuf_large,
    const double* __restrict__ t1,
    const int32_t* __restrict__ d_clink_index,
    const int32_t* __restrict__ natomax,
    int32_t strb0,
    int32_t na,
    int32_t nb,
    int32_t nnorb,
    int32_t ntile,
    int32_t ntile2,
    int32_t nab,
    int32_t nlink,
    int32_t amin,
    int32_t amax,
    int32_t na_self,
    int mode,
    int rank)
{
    const size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total_threads = (size_t)na_self * ntile2;
    
    if (thread_id >= total_threads) return;
    
    const size_t tx = thread_id % ntile2;
    const size_t stra0 = thread_id / ntile2;
    
    // Early bounds checking
    if (stra0 >= na_self) return;
    
    const size_t strb = (size_t)strb0 + tx;
    const size_t stra = stra0 + (size_t)amin;
    
    // Comprehensive bounds checking
    if (stra < amin || stra >= amax || strb >= nb) return;
    
    // Pre-calculate commonly used values
    const size_t ntile_sz = (size_t)ntile;
    const size_t nab_sz = (size_t)nab;
    const size_t nb_sz = (size_t)nb;
    const size_t nlink_sz = (size_t)nlink;
    
    // Pre-calculate base indices for link tables
    const int32_t* __restrict__ taba = d_clink_index + stra * nlink_sz * 3;
    const int32_t* __restrict__ tabb = d_clink_index + strb * nlink_sz * 3;
    
    // Pre-calculate common array indices
    const size_t out_base_idx = stra0 * nb_sz;
    const size_t t1_base_idx = stra0 * ntile_sz + tx;
    
    // ReduceScatter mode
    for (size_t j = 0; j < nlink_sz; j++) {
        const size_t j3 = j * 3;

        const size_t  ia    = (size_t)taba[j3];
        size_t        str1a = (size_t)taba[j3 + 1];
        const int8_t  signa = taba[j3 + 2];

        const size_t  ib    = (size_t)tabb[j3];
        const size_t  str1b = (size_t)tabb[j3 + 1];
        const int8_t  signb = tabb[j3 + 2];

        if (mode != 0) str1a = (size_t)natomax[str1a];

        const double t1_val_a = t1[ia * nab_sz + t1_base_idx];
        const double t1_val_b = t1[ib * nab_sz + t1_base_idx];

        atomicAdd(&d_cbuf_large[str1a * ntile_sz + tx], signa * t1_val_a);
        atomicAdd(&out[out_base_idx + str1b],            signb * t1_val_b);
    }
}


__global__ void _copy_buf(
    double  *in,          double  *out,
    int32_t  strb0,       int32_t  ntile,
    int32_t  amin,        int32_t  amax,
    int32_t  na_per_node, int32_t  na_max_node,
    int32_t  na,          int32_t  nb,
    int      rank)
{
    const int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t ntile2    = (nb - strb0 < ntile) ? (nb - strb0) : ntile;

    if (thread_id >= na_max_node * ntile2) return;

    const int32_t stra = thread_id / ntile2;
    const int32_t tx   = thread_id % ntile2;
    const int32_t strb = strb0 + tx;

    if (strb >= nb) return;

    if (stra < na_per_node) {
        const size_t index_in = (size_t)stra * (size_t)nb + (size_t)strb;
        out[stra * ntile + tx] = in[index_in];
    } else if (stra < na_max_node) {
        out[stra * ntile + tx] = 0.0;
    }
}


__global__
void _add_to_c_option(
    const double* __restrict__ ci1_buf,
    const double* __restrict__ cbuf_local,
    double* __restrict__ out,
    int32_t strb0,
    int32_t ntile,
    int32_t ntile2,
    int32_t nb,
    int32_t amin,
    int32_t amax,
    int32_t na_per_node,
    int32_t na_max,
    int nprocs,
    int rank)
{
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = (size_t)na_per_node * ntile2;
    
    if (thread_id >= total_threads) return;
    
    size_t stra0 = thread_id / ntile2;
    size_t tx = thread_id % ntile2;
    
    int32_t stra0_i32 = (int32_t)stra0;
    int32_t tx_i32 = (int32_t)tx;
    int32_t strb = strb0 + tx_i32;
    
    // Bounds checking
    if (stra0_i32 >= na_per_node || tx_i32 >= ntile2 || strb >= nb) {
        return;
    }
    
    // Output index calculation
    size_t out_idx = stra0 * (size_t)nb + strb;
    
    size_t buf_idx = stra0 * (size_t)ntile + tx_i32;
    out[out_idx] += cbuf_local[buf_idx];
            
}

// ================================================================
// Structs for davidson and contract_2e
// ================================================================

// Problem dimensions and solver parameters
struct SolverConfig {
    // Problem dimensions
    int32_t na;          // total alpha strings
    int32_t na_self;     // alpha strings owned by this rank
    int32_t norb;        // number of orbitals
    int32_t nelec;       // number of electrons (alpha)
    int32_t nlinka;      // number of link entries per string
    int32_t nnorb;       // norb*(norb+1)/2
    int32_t ntile;       // tile (chunk) size
    size_t  np;          // na * na (total CI space)
    // Davidson parameters
    int     nroots;
    int     max_space;
    int     max_cycle;
    int     heff_size;   // max_space + nroots
    double  tol;
    double  E_rhf;
    // Distribution across ranks
    int32_t *counts_na;
    int32_t *starts_na;
    int32_t *ends_na;
    // Runtime flags
    int     in_cpu;
    int     debug_mode;
    int     rank;
    int     nprocs;
};

struct PipelineConfig {
    // Tile / buffer sizes (all derived from SolverConfig in davidson)
    int32_t na_max;       // counts_na[0]: max strings per rank
    int32_t nab;          // na_self * ntile
    int     mode;         // na % nprocs
    int32_t ntiles;       // (na + ntile - 1) / ntile
    size_t  sendcount;    // na_max * ntile
    size_t  recvlarge;    // sendcount * nprocs
    size_t  base_size;    // na_self * na * sizeof(double)
    size_t  nblocks_buf;  // grid size for _copy_buf
    size_t  nblocks;      // grid size for _build_t1_buf / _gather_buf
    int     threads;      // threads per block (256)
};

void contract_2e_spin1_gpu_buf(
    cublasHandle_t  handle,
    double         *d_eri,
    double        *d_t1,          double        *d_vt1,
    int32_t        *d_clink,       int32_t     *d_link_nnorb,
    double        *d_ci1_buf,      double       *d_cbuf_local,
    double        *d_cbuf_large,
    double         *d_ci0,          double         *d_ci1,
    int32_t        *natomax,
    // Configuration structs (replaces ~20 scalar arguments)
    const SolverConfig   &cfg,
    const PipelineConfig &pc,
    // Streams
    cudaStream_t    computeStream,
    ncclComm_t      ncclComm)
{
    const double D0 = 0.0, D1 = 1.0;

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetStream(handle, computeStream));
 
    const int32_t total_tiles     = (cfg.na + cfg.ntile - 1) / cfg.ntile;
    const int     print_period     = 100;

    const int32_t amin   = cfg.starts_na[cfg.rank];
    const int32_t amax   = cfg.ends_na[cfg.rank];
    // Timing variables (only used when debug_mode > 1)
    double total_allgather_time = 0.0;
    double total_allreduce_time = 0.0;
    double tile_start_time      = 0.0;
    double allgather_start      = 0.0;
    double allreduce_start      = 0.0;
    double allgather_time       = 0.0;
    double allreduce_time       = 0.0;
    double tile_total_time      = 0.0;
    double total_start_time     = 0.0;

    // Record total start time
    if (cfg.debug_mode > 1) {
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        total_start_time = MPI_Wtime();
    }

    // Initialize output buffer
    CUDA_CHECK(cudaMemsetAsync(d_ci1, 0, pc.base_size, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    // Main computation loop
    int32_t tile_id = 0;
    for (int32_t strb0 = 0; strb0 < cfg.na; strb0 += cfg.ntile) {
        const int32_t ntile2 = (cfg.na - strb0 < cfg.ntile) ? (cfg.na - strb0) : cfg.ntile;

        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            tile_start_time = MPI_Wtime();
            allgather_time  = 0.0;
            allreduce_time  = 0.0;
        }
        // ----------------------------------------------------------------
        // Compute: copy buffer
        // ----------------------------------------------------------------
        _copy_buf<<<pc.nblocks_buf, pc.threads, 0, computeStream>>>(
            d_ci0, d_cbuf_local, strb0, cfg.ntile,
            amin, amax, cfg.na_self, pc.na_max, cfg.na, cfg.na, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------------------------------
        // Communication: AllGather
        // ----------------------------------------------------------------
        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            allgather_start = MPI_Wtime();
        }
        NCCL_CHECK(ncclAllGather(
            d_cbuf_local, d_cbuf_large,
            pc.sendcount, ncclDouble, ncclComm, computeStream));

        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            allgather_time        = MPI_Wtime() - allgather_start;
            total_allgather_time += allgather_time;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }

        // ----------------------------------------------------------------
        // Compute: build t1 + dgemm + clear buffer + gather
        // ----------------------------------------------------------------

        _build_t1_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
                d_ci0, d_cbuf_large, d_t1, d_link_nnorb, natomax,
                strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, ntile2,
                amin, amax, cfg.na_self, pc.nab, pc.mode, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                pc.nab, cfg.nnorb, cfg.nnorb,
                &D1, d_t1, pc.nab, d_eri, cfg.nnorb,
                &D0, d_vt1, pc.nab));

        CUDA_CHECK(cudaMemsetAsync(d_cbuf_large, 0,
                    pc.recvlarge * sizeof(double), computeStream));
        
        _gather_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
                d_ci1, d_ci1_buf, d_cbuf_large, d_vt1,
                d_clink, natomax,
                strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, ntile2,
                pc.nab, cfg.nlinka, amin, amax, cfg.na_self, pc.mode, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------------------------------
        // Communication:  ReduceScatter / 
        // ----------------------------------------------------------------
        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            allreduce_start = MPI_Wtime();
        }
        NCCL_CHECK(ncclReduceScatter(
                d_cbuf_large, d_cbuf_local,
                pc.sendcount, ncclDouble, ncclSum, ncclComm, computeStream));
                
        if (cfg.debug_mode > 2) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            allreduce_time        = MPI_Wtime() - allreduce_start;
            total_allreduce_time += allreduce_time;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
        }


        // ----------------------------------------------------------------
        // Compute: add results to output
        // ----------------------------------------------------------------

        _add_to_c_option<<<pc.nblocks, pc.threads, 0, computeStream>>>(
               d_ci1_buf, d_cbuf_local, d_ci1,
               strb0, cfg.ntile, ntile2, cfg.na,
               amin, amax, cfg.na_self, pc.na_max, cfg.nprocs, cfg.rank);

        CUDA_CHECK(cudaGetLastError());

        // ----------------------------------------------------------------
        // Per-tile timing report: compute = tile_total - allgather - allreduce
        // ----------------------------------------------------------------
        if (cfg.debug_mode > 2 && cfg.rank == 0) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            tile_total_time          = MPI_Wtime() - tile_start_time;
            const double tile_comm   = allgather_time + allreduce_time;
            const double tile_compute = tile_total_time - tile_comm;

            if (tile_id % print_period == 0 || tile_id == total_tiles - 1 || total_tiles <= 20) {
                printf("Tile[%d/%d]: total=%.3fms, allgather=%.3fms, compute=%.3fms, reducescatter=%.3fms\n",
                       tile_id + 1, total_tiles, 
                       tile_total_time * 1000.0,
                       allgather_time  * 1000.0,
                       tile_compute    * 1000.0,
                       allreduce_time  * 1000.0);
            }
        }

        tile_id++;
    }

    // ----------------------------------------------------------------
    // Final timing summary: compute = total - allgather - allreduce
    // ----------------------------------------------------------------
    if (cfg.debug_mode > 2 && cfg.rank == 0) {
        const double total_time    = MPI_Wtime() - total_start_time;
        const double total_comm    = total_allgather_time + total_allreduce_time;
        const double total_compute = total_time - total_comm;
        if (cfg.debug_mode > 3) {
            printf("contract_2e_total_t: %.3fs\n", total_time);
            printf("  allgather: %.3fs (%.1f%%), ",
               total_allgather_time,
               100.0 * total_allgather_time / (total_time + 1e-9));
            printf("  allreduce: %.3fs (%.1f%%)\n",
               total_allreduce_time,
               100.0 * total_allreduce_time / (total_time + 1e-9));
        }
        printf("Compute - total: %.3fs (%.1f%%), ",
               total_compute,
               100.0 * total_compute / (total_time + 1e-9));
        printf("Communication -total: %.3fs (%.1f%%)\n",
               total_comm,
               100.0 * total_comm / (total_time + 1e-9));
    }
}




// ------------------------------------------------------------------
// Communication/computation pipeline
//
// commStream:
//   copy_buf -> AllGather(tile N)
//
// computeStream:
//   build/dgemm/gather(tile N-1)
//   -> ReduceScatter(tile N-1)
//
// ev_ag_done : AllGather complete
// ev_rs_done : buffer reusable
//
// Overlap:
//   AllGather(tile N)
//      ||
//   build/dgemm/gather(tile N-1)
// ------------------------------------------------------------------
void contract_2e_spin1_gpu_buf_overlap(
    cublasHandle_t  handle,
    double         *d_eri,
    double        **d_t1,          double        **d_vt1,
    int32_t        *d_clink,        int32_t        *d_link_nnorb,
    double        **d_ci1_buf,      double        **d_cbuf_local,
    double        **d_cbuf_large,
    double         *d_ci0,          double         *d_ci1,
    int32_t        *natomax,
    // Configuration structs (replaces ~20 scalar arguments)
    const SolverConfig   &cfg,
    const PipelineConfig &pc,
    // Streams
    cudaStream_t    computeStream,
    cudaStream_t    commStream,
    ncclComm_t      ncclComm)
{
    const double D0 = 0.0, D1 = 1.0;

    CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetStream(handle, computeStream));

    // ----------------------------------------------------------------
    // Events
    // ----------------------------------------------------------------
    cudaEvent_t ev_ag_done[2], ev_rs_done[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_ag_done[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_rs_done[i], cudaEventDisableTiming));
    }
    // Initialize ev_rs_done so tile 0 commStream does not wait on unrecorded event
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaEventRecord(ev_rs_done[i], computeStream));

    CUDA_CHECK(cudaMemsetAsync(d_ci1, 0, pc.base_size, computeStream));
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    // ----------------------------------------------------------------
    // Convenience aliases
    // ----------------------------------------------------------------
    const int32_t amin   = cfg.starts_na[cfg.rank];
    const int32_t amax   = cfg.ends_na[cfg.rank];
    if (pc.ntiles == 0) goto cleanup;
    // ----------------------------------------------------------------
    // Main loop
    // ----------------------------------------------------------------
    for (int32_t tile = 0; tile < pc.ntiles; tile++) {
        const int     b     = tile & 1;
        const int     p     = 1 - b;
        const int32_t strb0 = tile * cfg.ntile;

        // commStream: copy_buf + AllGather for tile N
        CUDA_CHECK(cudaStreamWaitEvent(commStream, ev_rs_done[b], 0));

        _copy_buf<<<pc.nblocks_buf, pc.threads, 0, commStream>>>(
            d_ci0, d_cbuf_local[b], strb0, cfg.ntile,
            amin, amax, cfg.na_self, pc.na_max, cfg.na, cfg.na, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        NCCL_CHECK(ncclAllGather(
            d_cbuf_local[b], d_cbuf_large[b],
            pc.sendcount, ncclDouble, ncclComm, commStream));
        CUDA_CHECK(cudaEventRecord(ev_ag_done[b], commStream));

        // computeStream: build_t1 + dgemm + gather + ReduceScatter for tile N-1
        if (tile > 0) {
            const int32_t prev_strb0  = (tile - 1) * cfg.ntile;
            const int32_t prev_ntile2 = (cfg.na - prev_strb0 < cfg.ntile)
                                        ? (cfg.na - prev_strb0) : cfg.ntile;

            CUDA_CHECK(cudaStreamWaitEvent(computeStream, ev_ag_done[p], 0));

            _build_t1_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
                d_ci0, d_cbuf_large[p], d_t1[p], d_link_nnorb, natomax,
                prev_strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, prev_ntile2,
                amin, amax, cfg.na_self, pc.nab, pc.mode, cfg.rank);
            CUDA_CHECK(cudaGetLastError());

            CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                pc.nab, cfg.nnorb, cfg.nnorb,
                &D1, d_t1[p], pc.nab, d_eri, cfg.nnorb,
                &D0, d_vt1[p], pc.nab));

            CUDA_CHECK(cudaMemsetAsync(d_cbuf_large[p], 0,
                                       pc.recvlarge * sizeof(double), computeStream));

            _gather_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
                d_ci1, d_ci1_buf[p], d_cbuf_large[p], d_vt1[p],
                d_clink, natomax,
                prev_strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, prev_ntile2,
                pc.nab, cfg.nlinka, amin, amax, cfg.na_self, pc.mode, cfg.rank);
            CUDA_CHECK(cudaGetLastError());

            NCCL_CHECK(ncclReduceScatter(
                d_cbuf_large[p], d_cbuf_local[p],
                pc.sendcount, ncclDouble, ncclSum, ncclComm, computeStream));

            _add_to_c_option<<<pc.nblocks, pc.threads, 0, computeStream>>>(
                d_ci1_buf[p], d_cbuf_local[p], d_ci1,
                prev_strb0, cfg.ntile, prev_ntile2, cfg.na,
                amin, amax, cfg.na_self, pc.na_max, cfg.nprocs, cfg.rank);
            CUDA_CHECK(cudaEventRecord(ev_rs_done[p], computeStream));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // ----------------------------------------------------------------
    // Process last tile (ntiles-1)
    // ----------------------------------------------------------------
    {
        const int     p           = (pc.ntiles - 1) & 1;
        const int32_t prev_strb0  = (pc.ntiles - 1) * cfg.ntile;
        const int32_t prev_ntile2 = (cfg.na - prev_strb0 < cfg.ntile)
                                    ? (cfg.na - prev_strb0) : cfg.ntile;
        const int32_t amin        = cfg.starts_na[cfg.rank];
        const int32_t amax        = cfg.ends_na[cfg.rank];

        CUDA_CHECK(cudaStreamWaitEvent(computeStream, ev_ag_done[p], 0));

        _build_t1_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
            d_ci0, d_cbuf_large[p], d_t1[p], d_link_nnorb, natomax,
            prev_strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, prev_ntile2,
            amin, amax, cfg.na_self, pc.nab, pc.mode, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            pc.nab, cfg.nnorb, cfg.nnorb,
            &D1, d_t1[p], pc.nab, d_eri, cfg.nnorb,
            &D0, d_vt1[p], pc.nab));

        CUDA_CHECK(cudaMemsetAsync(d_cbuf_large[p], 0,
                                   pc.recvlarge * sizeof(double), computeStream));

        _gather_buf<<<pc.nblocks, pc.threads, 0, computeStream>>>(
            d_ci1, d_ci1_buf[p], d_cbuf_large[p], d_vt1[p],
            d_clink, natomax,
            prev_strb0, cfg.na, cfg.na, cfg.nnorb, cfg.ntile, prev_ntile2,
            pc.nab, cfg.nlinka, amin, amax, cfg.na_self, pc.mode, cfg.rank);
        CUDA_CHECK(cudaGetLastError());

        NCCL_CHECK(ncclReduceScatter(
            d_cbuf_large[p], d_cbuf_local[p],
            pc.sendcount, ncclDouble, ncclSum, ncclComm, computeStream));

    
        _add_to_c_option<<<pc.nblocks, pc.threads, 0, computeStream>>>(
            d_ci1_buf[p], d_cbuf_local[p], d_ci1,
            prev_strb0, cfg.ntile, prev_ntile2, cfg.na,
            amin, amax, cfg.na_self, pc.na_max, cfg.nprocs, cfg.rank);
        CUDA_CHECK(cudaEventRecord(ev_rs_done[p], computeStream));
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    CUDA_CHECK(cudaStreamSynchronize(commStream));

cleanup:
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaEventDestroy(ev_ag_done[i]));
        CUDA_CHECK(cudaEventDestroy(ev_rs_done[i]));
    }
}


__global__ void normalize_kernel(double* __restrict__ ci0, size_t np,
                                 const double* __restrict__ d_inner,
                                 double lindep)
{
    const double inner_val = d_inner[0];
    const double norm = sqrt(inner_val);
    
    // Early exit if normalization is not needed
    if (inner_val <= lindep || norm <= 1e-14) return;
    
    const double inv_norm = 1.0 / norm;
    
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t id = tid; id < np; id += stride) {
        ci0[id] *= inv_norm;
    }
}



__global__
void _convertids(int32_t *natomax, int32_t *starts_aid, int32_t*ends_aid, int32_t na, int32_t na_max_total, int32_t na_max_node, int rank)
{
    int32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= na_max_total) return;
    
    int32_t proc_id = thread_id / na_max_node;
    int32_t id_in_proc = thread_id % na_max_node;
    
    // Bounds check for proc_id to prevent out-of-bounds access
    if (proc_id >= gridDim.x * blockDim.x / na_max_node) return;
    
    int32_t amin = starts_aid[proc_id];
    int32_t amax = ends_aid[proc_id];
    int32_t local_count = amax - amin;
    
    if (id_in_proc < local_count) {
        int32_t global_id = amin + id_in_proc;
        
        // Bounds check before writing to natomax
        if (global_id < na) {
            natomax[global_id] = thread_id;
        }
    } 
}

__global__ void dr_kernel_safe(
    const double* __restrict__ ci0,
    const double* __restrict__ ci1,
    double* __restrict__ r,
    double lambda,
    size_t n)
{
    size_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        r[i] = ci1[i] - lambda * ci0[i];
    }
}


__global__ void jkcopy_kernel(
    double *d_Gmo, double *d_jdiag, 
    double *d_kdiag, int32_t norb, 
    int32_t norb_sq, int32_t norb_t)
{
    int32_t k=blockIdx.x * blockDim.x + threadIdx.x;
    if (k>=norb_sq) return;
    int32_t i = k/norb;
    int32_t j = k%norb;
    d_jdiag[i*norb + j] = d_Gmo[i*norb_t + i*norb_sq + j*norb + j];
    d_kdiag[i*norb + j] = d_Gmo[i*norb_t + j*norb_sq + j*norb + i];
}




// ================================================================
// Davidson eigenvalue solver
// ================================================================
void davidson(
    // CI vector storage
    double  *d_ci0_list, double *d_ci1_list,
    double  *h_ci0_list, double *h_ci1_list,
    double  *d_ci0,      double *d_ci1,
    // Integrals and link indices
    double  *eri,
    double  *d_hdiag,    double *hdiag,
    int32_t *d_link_index, int32_t *d_link_nnorb,
    // Output
    double  *e,          double *e_check,
    // Work arrays
    double  *d_tmp,      double *d_heff,   double *d_heff_tmp,
    // Configuration (replaces ~20 scalar arguments)
    const SolverConfig &cfg,
    // Communication
    ncclComm_t   ncclComm,
    cudaStream_t computeStream)
{
    // ========== TIMING ==========
    struct timespec ts_begin, ts_end, ts_iter_begin, ts_iter_end;
    double t_sigma_cal_total = 0.0, t_subspace_proj_total = 0.0;
    double t_subspace_diag_total = 0.0, t_subspace_expans_total = 0.0, t_total = 0.0;
    double t_sigma_cal, t_subspace_proj, t_subspace_diag, t_subspace_expans;

    if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &ts_begin);

    // ========== CONSTANTS ==========
    const size_t  mynp        = (size_t)cfg.na_self * (size_t)cfg.na;
    const double  lindep      = 1e-10;
    const double  level_shift = 1e-3;
    const double  toloose     = sqrt(cfg.tol) / 100.0;
    const int     nthread     = 256;
    const size_t  t1_size     = (size_t)cfg.na_self * cfg.ntile * cfg.nnorb * sizeof(double);
    const size_t  nblock      = (mynp + nthread - 1) / nthread;
    const int     nblock_small = (int)((nblock < 65535) ? nblock : 65535);

    // Local alias for buffer allocation (used before main loop)
    const int32_t na_max = cfg.counts_na[0];

    // Pipeline config (used when calling contract_2e)
    PipelineConfig pc;
    pc.na_max      = cfg.counts_na[0];
    pc.nab         = cfg.na_self * cfg.ntile;
    pc.mode        = cfg.na % cfg.nprocs;
    pc.ntiles      = (cfg.na + cfg.ntile - 1) / cfg.ntile;
    pc.sendcount   = (size_t)pc.na_max * cfg.ntile;
    pc.recvlarge   = pc.sendcount * cfg.nprocs;
    pc.base_size   = (size_t)cfg.na_self * cfg.na * sizeof(double);
    pc.nblocks_buf = ((size_t)pc.na_max * cfg.ntile + nthread - 1) / nthread;
    pc.nblocks     = ((size_t)cfg.na_self * cfg.ntile + nthread - 1) / nthread;
    pc.threads     = nthread;

    // ========== SOLVER STATE ==========
    int    space = 0, conv = 0, reset_state = 0, conv_last = 0;
    double dx_norm = 0.0, de = 0.0, e_last = 0.0;

    bool overlap_state = true; // Use overlap pipeline or not (overlap_state = true enables overlap)
    

    // ========== DOT PRODUCT SETUP ==========
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cur_device));
    const int blocks_dot = prop.multiProcessorCount * 8;

    if (cfg.debug_mode > 1 && cfg.rank == 0) {
        printf("Davidson: rank=%d na_self=%d mynp=%zu blocks_dot=%d\n",
               cfg.rank, cfg.na_self, mynp, blocks_dot);
    }

    // ========== MEMORY ==========
    double  *d_ci1_buf[2]    = {nullptr};
    double  *d_cbuf_local[2] = {nullptr};
    double  *d_cbuf_large[2] = {nullptr};
    double  *dl_t1[2]        = {nullptr};
    double  *dl_vt1[2]       = {nullptr};
    double  *d_W             = nullptr;
    double  *d_inner_local   = nullptr;
    double  *d_inner_global  = nullptr;
    double  *d_partial       = nullptr;
    double  *h_partial       = nullptr;
    double  *h_single        = nullptr;
    double  *v               = nullptr;
    double  *v_last          = nullptr;
    int32_t *natomax         = nullptr;
    int     *devInfo         = nullptr;
    int     *count_buf       = nullptr;
    cublasHandle_t     handle    = nullptr;
    cusolverDnHandle_t cusolverH = nullptr;
    cudaStream_t       commStream = nullptr;

    // Host allocations
    h_single  = (double*)malloc(sizeof(double));
    v         = (double*)malloc(cfg.heff_size * cfg.heff_size * sizeof(double));
    v_last    = (double*)calloc(cfg.heff_size, cfg.nroots * sizeof(double));
    count_buf = (int*)malloc(cfg.nprocs * sizeof(int));
    h_partial = (double*)malloc(blocks_dot * sizeof(double));
    if (!h_single || !v || !v_last || !count_buf || !h_partial) {
        fprintf(stderr, "Host memory allocation failed\n");
        goto cleanup;
    }

    // commStream: created here, used only by contract_2e pipeline
    CUDA_CHECK(cudaStreamCreate(&commStream));

    // Device allocations
    CUDA_CHECK(cudaMalloc((void**)&d_W,            cfg.max_space * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&devInfo,         sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_inner_local,   sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_inner_global,  sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_partial,       blocks_dot * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&dl_t1[0],  t1_size));
    CUDA_CHECK(cudaMalloc((void**)&dl_vt1[0], t1_size));

    // Multi-process pipeline buffers
    if (cfg.nprocs > 1) {
        if (overlap_state) { 
            CUDA_CHECK(cudaMalloc((void**)&dl_t1[1],  t1_size));
            CUDA_CHECK(cudaMalloc((void**)&dl_vt1[1], t1_size));
        }

        const int32_t na_max_total = na_max * cfg.nprocs;
        const size_t  buflocalsize = (size_t)na_max * cfg.ntile * sizeof(double);
        const size_t  buflargesize = (size_t)na_max * cfg.ntile * cfg.nprocs * sizeof(double);
        int num_buf = 2; // Double buffering
        if (!overlap_state) num_buf = 1; // If not overlapping, only one set of buffers is needed
        for (int i = 0; i < num_buf; i++) {
            CUDA_CHECK(cudaMalloc((void**)&d_ci1_buf[i],
                                  (size_t)cfg.na * cfg.ntile * sizeof(double)));
            CUDA_CHECK(cudaMalloc((void**)&d_cbuf_local[i], buflocalsize));
            CUDA_CHECK(cudaMalloc((void**)&d_cbuf_large[i], buflargesize));
        }

        CUDA_CHECK(cudaMalloc((void**)&natomax, cfg.na        * sizeof(int32_t)));

        int32_t *d_starta = nullptr, *d_enda = nullptr;
        CUDA_CHECK(cudaMalloc((void**)&d_starta, cfg.nprocs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc((void**)&d_enda,   cfg.nprocs * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_starta, cfg.starts_na,
                              cfg.nprocs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_enda,   cfg.ends_na,
                              cfg.nprocs * sizeof(int32_t), cudaMemcpyHostToDevice));

        const int nblock_conv = (na_max_total + min(nthread, na_max) - 1)
                                / min(nthread, na_max);
        _convertids<<<nblock_conv, min(nthread, na_max), 0, computeStream>>>(
            natomax, d_starta, d_enda,
            cfg.na, na_max_total, na_max, cfg.rank);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        cudaFree(d_starta);
        cudaFree(d_enda);

        for (int i = 0; i < cfg.nprocs; i++) count_buf[i] = na_max * cfg.ntile;
    }

    // Handles
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, computeStream));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, computeStream));

    // ========== INITIAL NORMALIZATION ==========
    dot_func(d_partial, h_partial, d_ci0, d_ci0, d_inner_local,
             mynp, blocks_dot, cfg.rank, computeStream);
    if (cfg.nprocs > 1) {
        NCCL_CHECK(ncclAllReduce(d_inner_local, d_inner_global, 1,
                                 ncclDouble, ncclSum, ncclComm, computeStream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(d_inner_global, d_inner_local, sizeof(double),
                                   cudaMemcpyDeviceToDevice, computeStream));
    }
    normalize_kernel<<<nblock_small, nthread, 0, computeStream>>>(
        d_ci0, mynp, d_inner_global, lindep);
    CUDA_CHECK(cudaGetLastError());
    

    if (cfg.debug_mode > 1 && cfg.rank == 0)
        printf("Davidson initialization complete for rank %d\n", cfg.rank);

    // ========== MAIN LOOP ==========
    for (int icyc = 0; icyc < cfg.max_cycle; icyc++) {

        if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &ts_iter_begin);
        if (cfg.rank == 0 && cfg.debug_mode > 0) printf("\nDavidson iteration %d\n", icyc);

        // --- Hamiltonian application ---
        struct timespec t1, t2;
        if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &t1);

        if (reset_state == 1) {
            if (cfg.rank == 0 && cfg.debug_mode > 0) printf("reset state\n");
        } else if (cfg.nprocs > 1 && overlap_state) {
            contract_2e_spin1_gpu_buf_overlap(
                handle, eri, dl_t1, dl_vt1,                 
                d_link_index, d_link_nnorb,
                d_ci1_buf, d_cbuf_local, d_cbuf_large,
                d_ci0, d_ci1, natomax, 
                cfg, pc, computeStream, commStream, ncclComm);
        } else if (cfg.nprocs > 1) {
            contract_2e_spin1_gpu_buf(
                handle, eri, dl_t1[0], dl_vt1[0],
                d_link_index, d_link_nnorb, 
                d_ci1_buf[0], d_cbuf_local[0], d_cbuf_large[0], 
                d_ci0, d_ci1, natomax,
                cfg, pc, computeStream, ncclComm);
        }
        else {
            contract_2e_spin1_gpu(
                handle, eri, d_ci0, d_ci1, dl_t1[0], dl_vt1[0],
                cfg.norb, cfg.na, cfg.na, cfg.nlinka, 
                d_link_index, d_link_nnorb, cfg.na_self, 
                cfg.ntile, cfg.debug_mode, computeStream);
        }

        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            clock_gettime(CLOCK_MONOTONIC, &t2);
            t_sigma_cal = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
            t_sigma_cal_total += t_sigma_cal;
        }
        if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &t1);
        // --- Store vectors ---
        const size_t offset = space * mynp;
        if (cfg.in_cpu == 1) {
            CUDA_CHECK(cudaMemcpyAsync(h_ci1_list + offset, d_ci1,
                                       mynp * sizeof(double),
                                       cudaMemcpyDeviceToHost, computeStream));
            CUDA_CHECK(cudaMemcpyAsync(h_ci0_list + offset, d_ci0,
                                       mynp * sizeof(double),
                                       cudaMemcpyDeviceToHost, computeStream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_ci1_list + offset, d_ci1,
                                       mynp * sizeof(double),
                                       cudaMemcpyDeviceToDevice, computeStream));
            CUDA_CHECK(cudaMemcpyAsync(d_ci0_list + offset, d_ci0,
                                       mynp * sizeof(double),
                                       cudaMemcpyDeviceToDevice, computeStream));
        }
        space++;

        // --- Effective Hamiltonian ---
        
        fill_heff_hermitian_gpu_fast(
            handle, ncclComm, computeStream,
            d_heff_tmp, d_heff, d_ci0, d_ci1,
            d_ci1_list, h_ci1_list, d_tmp,
            d_inner_global, d_inner_local, d_partial, h_partial,
            space, cfg.nroots, cfg.heff_size,
            cfg.nprocs, cfg.in_cpu, mynp, blocks_dot, cfg.rank);
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        if (cfg.debug_mode > 1) {
            clock_gettime(CLOCK_MONOTONIC, &t2);
            t_subspace_proj = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
            t_subspace_proj_total += t_subspace_proj;
        }

        // --- Eigenvalue decomposition ---
        e_last    = e[0];
        conv_last = conv;
        if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &t1);
        computeEigenvaluesAndVectorsn(cusolverH, space, d_heff, e, v, devInfo, d_W);
        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            clock_gettime(CLOCK_MONOTONIC, &t2);
            t_subspace_diag = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
            t_subspace_diag_total += t_subspace_diag;
        }
        if (cfg.rank == 0 && cfg.debug_mode > 0)
            printf("Iteration %d: E=%.15f FCI_E=%.15f\n",
                   icyc, e[0], e[0] + cfg.E_rhf);

        // --- Generate new trial vector ---
        if (cfg.debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &t1);
        if (cfg.in_cpu == 1) {
            gen_x0_gpu_incpu(v, h_ci0_list, d_ci0, d_tmp, space, mynp, computeStream);
            gen_x0_gpu_incpu(v, h_ci1_list, d_ci1, d_tmp, space, mynp, computeStream);
        } else {
            gen_x0_gpu(v, d_ci0_list, d_ci1_list, d_ci0, d_ci1,
                       space, mynp, computeStream);
        }
        

        reset_state = 0;
        de = (icyc == 0) ? e[0] : e[0] - e_last;

        // Abort on divergence
        if (icyc > 1 && fabs(de) > 10.0) {
            e_check[1] = 100;
            if (cfg.rank == 0) printf("Energy diverged: de=%f\n", de);
            break;
        }

        // Helper: AllReduce d_inner_local -> h_single
        auto allreduce_to_host = [&]() {
            if (cfg.nprocs > 1) {
                NCCL_CHECK(ncclAllReduce(d_inner_local, d_inner_global, 1,
                                         ncclDouble, ncclSum, ncclComm, computeStream));
                CUDA_CHECK(cudaStreamSynchronize(computeStream));
                CUDA_CHECK(cudaMemcpy(h_single, d_inner_global,
                                      sizeof(double), cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaStreamSynchronize(computeStream));
                CUDA_CHECK(cudaMemcpy(h_single, d_inner_local,
                                      sizeof(double), cudaMemcpyDeviceToHost));
            }
        };

        // --- Space management ---
        if (space >= cfg.max_space) {
            if (fabs(de) >= cfg.tol) {
                if (cfg.rank == 0 && cfg.debug_mode > 1){
                    CUDA_CHECK(cudaStreamSynchronize(computeStream));
                    clock_gettime(CLOCK_MONOTONIC, &t2);
                    t_subspace_expans = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
                    t_subspace_expans_total += t_subspace_expans;
                }
                space = 0; reset_state = 1; continue;
            }
            dr_kernel_safe<<<nblock_small, nthread, 0, computeStream>>>(
                d_ci0, d_ci1, d_tmp, e[0], mynp);
            CUDA_CHECK(cudaGetLastError());
            dot_func(d_partial, h_partial, d_tmp, d_tmp, d_inner_local,
                     mynp, blocks_dot, cfg.rank, computeStream);
            allreduce_to_host();
            dx_norm = sqrt(fabs(h_single[0]));
            conv    = (fabs(de) < cfg.tol && dx_norm < toloose) ? 1 : 0;
            if (cfg.rank == 0)
                printf("Final check: dx_norm=%e de=%e conv=%d\n", dx_norm, de, conv);
            if (conv == 1) break;
            space = 0; reset_state = 1; continue;
        }

        // --- Residual ---
        
        dr_kernel_safe<<<nblock_small, nthread, 0, computeStream>>>(
            d_ci0, d_ci1, d_ci0, e[0], mynp);
        CUDA_CHECK(cudaGetLastError());
        dot_func(d_partial, h_partial, d_ci0, d_ci0, d_inner_local,
                 mynp, blocks_dot, cfg.rank, computeStream);
        allreduce_to_host();
        dx_norm = sqrt(fabs(h_single[0]));
        conv    = (fabs(de) < cfg.tol && dx_norm < toloose) ? 1 : 0;
        

        // --- Preconditioning ---
        if (conv == 0 && dx_norm * dx_norm > lindep) {
            if (cfg.in_cpu == 1) {
                CUDA_CHECK(cudaMemcpyAsync(d_tmp, hdiag, mynp * sizeof(double),
                                           cudaMemcpyHostToDevice, computeStream));
                precond_kernel<<<nblock_small, nthread, 0, computeStream>>>(
                    d_tmp, d_ci0, e[0], level_shift, mynp);
            } else {
                precond_kernel<<<nblock_small, nthread, 0, computeStream>>>(
                    d_hdiag, d_ci0, e[0], level_shift, mynp);
            }
            CUDA_CHECK(cudaGetLastError());
            dot_func(d_partial, h_partial, d_ci0, d_ci0, d_inner_local,
                     mynp, blocks_dot, cfg.rank, computeStream);
            allreduce_to_host();
            if (h_single[0] > 0.0) {
                const double tmpk = pow(h_single[0], -0.5);
                Dscal_kernel<<<nblock_small, nthread, 0, computeStream>>>(
                    d_ci0, d_ci0, tmpk, mynp);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            if (cfg.rank == 0 && cfg.debug_mode > 0)
                printf("Vector is linearly dependent or converged\n");
            break;
        }

        // --- Orthogonalization ---
        if (cfg.in_cpu == 1) {
            normalize_xt_gpu_large(
                handle, d_ci0, h_ci0_list, d_tmp, lindep, 1.0,
                space, cfg.np, cfg.nprocs, ncclComm, cfg.rank, mynp,
                d_inner_local, d_inner_global, h_single, cfg.in_cpu,
                blocks_dot, d_partial, h_partial, computeStream);
        } else {
            normalize_xt_gpu(
                handle, d_ci0, d_ci0_list, d_tmp, lindep, 1.0,
                space, cfg.np, cfg.nprocs, ncclComm, cfg.rank, mynp,
                d_inner_local, d_inner_global, cfg.in_cpu, computeStream);
        }
        if (cfg.debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            clock_gettime(CLOCK_MONOTONIC, &t2);
            t_subspace_expans = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
            t_subspace_expans_total += t_subspace_expans;
        }

        // --- Per-iteration report ---
        if (cfg.rank == 0 && cfg.debug_mode > 0) {
            if (cfg.debug_mode > 1) {
                clock_gettime(CLOCK_MONOTONIC, &ts_iter_end);
                const double iter_time =
                    (ts_iter_end.tv_sec  - ts_iter_begin.tv_sec) +
                    (ts_iter_end.tv_nsec - ts_iter_begin.tv_nsec) / 1e9;
                printf("Iteration %d: dx_norm=%e, dE=%e, conv=%d, time=%.3fs\n",
                       icyc, dx_norm, de, conv, iter_time);
                printf("  Step timing (ms): sigma_cal=%.1f, subspace_proj=%.1f, subspace_diag=%.1f, subspace_expans=%.1f\n",
                       t_sigma_cal*1000, t_subspace_proj*1000, t_subspace_diag*1000, t_subspace_expans*1000);
            } else {
                printf("Iteration %d: dx_norm=%e dE=%e conv=%d\n",
                       icyc, dx_norm, de, conv);
            }
        }

        if (conv == 1 && conv_last == 0) {
            if (cfg.rank == 0 && cfg.debug_mode > 0) printf("Convergence achieved!\n");
            break;
        }
    }

    // ========== FINAL TIMING ==========
    if (cfg.debug_mode > 1) {
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        t_total = (ts_end.tv_sec  - ts_begin.tv_sec) +
                  (ts_end.tv_nsec - ts_begin.tv_nsec) / 1e9;
        if (cfg.rank == 0) {
            printf("Davidson completed: total=%.3fs\n", t_total);
            printf("  sigma_calculation       : %.3fs (%.1f%%)\n", t_sigma_cal_total,     100.0*t_sigma_cal_total/t_total);
            printf("  subspace_projection     : %.3fs (%.1f%%)\n", t_subspace_proj_total, 100.0*t_subspace_proj_total/t_total);
            printf("  subspace_diagonalization: %.3fs (%.1f%%)\n", t_subspace_diag,       100.0*t_subspace_diag/t_total);
            printf("  subspace_expansion      : %.3fs (%.1f%%)\n", t_subspace_expans,     100.0*t_subspace_expans/t_total);
        }
    }

cleanup:
    if (handle)     { cublasDestroy(handle);        handle     = nullptr; }
    if (cusolverH)  { cusolverDnDestroy(cusolverH); cusolverH  = nullptr; }
    if (commStream) { CUDA_CHECK(cudaStreamDestroy(commStream)); commStream = nullptr; }

    auto safe_free = [](void *&ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };
    safe_free((void*&)d_inner_local);
    safe_free((void*&)d_inner_global);
    safe_free((void*&)devInfo);
    safe_free((void*&)d_W);
    safe_free((void*&)d_partial);
    safe_free((void*&)dl_t1[0]);
    safe_free((void*&)dl_vt1[0]);

    if (cfg.nprocs > 1) {
        if (overlap_state) {
            safe_free((void*&)dl_t1[1]);
            safe_free((void*&)dl_vt1[1]);
            for (int i = 0; i < 2; i++) {
                safe_free((void*&)d_ci1_buf[i]);
                safe_free((void*&)d_cbuf_local[i]);
                safe_free((void*&)d_cbuf_large[i]);
            }
        } else {
            safe_free((void*&)d_ci1_buf[0]);
            safe_free((void*&)d_cbuf_local[0]);
            safe_free((void*&)d_cbuf_large[0]);
        }
        safe_free((void*&)natomax);
    }

    auto safe_host_free = [](void *&ptr) {
        if (ptr) { free(ptr); ptr = nullptr; }
    };
    safe_host_free((void*&)h_single);
    safe_host_free((void*&)v);
    safe_host_free((void*&)v_last);
    safe_host_free((void*&)count_buf);
    safe_host_free((void*&)h_partial);
}

void combinationUtil(int *occslst_flat, int *index, int *arr, int n, int r, int *data, int data_index, int start) {
    if (data_index == r) {
        // 現在の組み合わせを occslst_flat に保存
        for (int i = 0; i < r; i++) {
            occslst_flat[(*index)++] = data[i];
        }
        return;
    }

    for (int i = start; i < n; i++) {
        data[data_index] = arr[i];
        combinationUtil(occslst_flat, index, arr, n, r, data, data_index + 1, i + 1);
    }
}


void gen_occs_iter_ci_new(int *occslst_flat, int *index, int *orb_list, int nelec, int norb) {
 
    int *data = (int*)malloc(nelec * sizeof(int));
    combinationUtil(occslst_flat, index, orb_list, norb, nelec, data, 0, 0);

    free(data);
}

void range(int* in, int size) {
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<size; i++){
        in[i] = i;
    }
}

} // anonymous namespace

double fci_mpi(
    double* d_Gmo1e, double* d_Gmo, int norb, int nelec, int na, long long np, double E_rhf)
{
    // =========================================================
    // Parameters
    // =========================================================
    const int    max_space  = 12;
    const int    max_cycle  = 100;
    const int    in_cpu     = 0;
    const int    tile       = 256;
    const int    debug_mode = 2;
    const double tol        = 1e-10;
    const int    nroots     = 1;
    const int    neleca     = nelec / 2;

    // =========================================================
    // Timing
    // =========================================================
    struct timespec ts_begin, ts_end_init, ts_end_other, ts_end;
    float t_init = 0.0f, t_other_init = 0.0f, t_davidson = 0.0f, t_total = 0.0f;

    if (debug_mode > 1) clock_gettime(CLOCK_MONOTONIC, &ts_begin);

    // =========================================================
    // MPI / GPU / NCCL Initialization
    // =========================================================
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(NULL, NULL);
    }
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_CHECK(MPI_Comm_rank(comm, &rank));
    MPI_CHECK(MPI_Comm_size(comm, &nprocs));
    int current_device = -1; cudaGetDevice(&current_device);
    printf("rank=%d using GPU device=%d\n", rank, current_device); fflush(stdout);

    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm));

    int local_rank;
    MPI_CHECK(MPI_Comm_rank(node_comm, &local_rank));

    int ngpu;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    // cudaSetDevice: already set by gpu_manager::initialize_gpu()

    ncclUniqueId ncclId;
    ncclComm_t   ncclComm = nullptr;
    if (nprocs > 1) {
        if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&ncclId));
        MPI_CHECK(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, comm));
        NCCL_CHECK(ncclCommInitRank(&ncclComm, nprocs, ncclId, rank));
    }

    cudaStream_t computeStream = nullptr, transferStream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&computeStream));
    CUDA_CHECK(cudaStreamCreate(&transferStream));

    // =========================================================
    // Generate occslst internally (fci_mpi style)
    // =========================================================
    int *occslst  = (int*)malloc((size_t)na * neleca * sizeof(int));
    int *orb_list = (int*)malloc(norb * sizeof(int));
    if (!occslst || !orb_list) {
        fprintf(stderr, "Failed to allocate occslst/orb_list\n");
        MPI_Abort(comm, 1);
    }
    {
        int index = 0;
        range(orb_list, norb);
        gen_occs_iter_ci_new(occslst, &index, orb_list, neleca, norb);
    }
    free(orb_list);

    // =========================================================
    // Size calculations
    // =========================================================
    const int32_t na_ave = na / nprocs;
    const int32_t na_res = na % nprocs;

    int32_t *counts_na = (int32_t*)malloc(nprocs * sizeof(int32_t));
    int32_t *starts_na = (int32_t*)malloc(nprocs * sizeof(int32_t));
    int32_t *ends_na   = (int32_t*)malloc(nprocs * sizeof(int32_t));
    if (!counts_na || !starts_na || !ends_na) {
        fprintf(stderr, "Failed to allocate distribution arrays\n");
        MPI_Abort(comm, 1);
    }

    for (int i = 0; i < nprocs; i++) {
        counts_na[i] = na_ave + (i < na_res ? 1 : 0);
        starts_na[i] = (i == 0) ? 0 : starts_na[i-1] + counts_na[i-1];
        ends_na[i]   = starts_na[i] + counts_na[i];
    }

    const int32_t amin  = starts_na[rank];
    const int32_t my_na = counts_na[rank];
    const size_t  mynp  = (size_t)my_na * (size_t)na;

    const size_t norb_sq = (size_t)norb * norb;
    const size_t norb_t  = norb_sq * norb;
    const size_t norb4   = norb_t  * norb;

    const int     heff_size = max_space + nroots;
    const int32_t nnorb     = norb * (norb + 1) / 2;
    const int32_t nlinka    = neleca + neleca * (norb - neleca);
    const int32_t chunk     = (na > tile) ? tile : na;

    const size_t link_size   = (size_t)nlinka * (size_t)na * 3    * sizeof(int32_t);
    const size_t linknn_size = (size_t)na     * (size_t)nnorb * 2 * sizeof(int32_t);
    const size_t occs_size   = (size_t)na     * neleca             * sizeof(int32_t);
    const size_t base_size   = mynp                                * sizeof(double);
    const size_t list_size   = (size_t)max_space * mynp            * sizeof(double);

    const int    nthread      = 256;
    const size_t nblockp      = (mynp + nthread - 1) / nthread;
    const size_t large_blocks = (nblockp > 65535) ? 65535 : nblockp;

    // =========================================================
    // GPU memory allocation
    // When nprocs>1, rank!=0 needs receive buffers for d_Gmo / d_Gmo1e
    // =========================================================
    double *d_Gmo_local   = NULL;
    double *d_Gmo1e_local = NULL;
    bool    owns_Gmo      = false;

    if (nprocs > 1 && rank != 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_Gmo_local,   norb4   * sizeof(double)));
        CUDA_CHECK(cudaMalloc((void**)&d_Gmo1e_local, norb_sq * sizeof(double)));
        owns_Gmo = true;
    } else {
        // rank==0 (or nprocs==1) uses the argument pointer directly
        d_Gmo_local   = d_Gmo;
        d_Gmo1e_local = d_Gmo1e;
    }

    double  *d_jdiag      = NULL, *d_kdiag      = NULL, *d_hdiag    = NULL;
    double  *d_eri        = NULL, *d_heff        = NULL, *d_heff_tmp = NULL;
    double  *d_ci0        = NULL, *d_ci1         = NULL, *d_tmp      = NULL;
    double  *d_ci0_list   = NULL, *d_ci1_list    = NULL;
    int32_t *d_clink      = NULL, *d_link_nnorb  = NULL, *d_occslst  = NULL;
    double  *e            = NULL;
    cudaEvent_t occs_ready;

    e = (double*)calloc(heff_size, sizeof(double));
    if (!e) {
        fprintf(stderr, "Failed to allocate energy array\n");
        goto cleanup;
    }

    CUDA_CHECK(cudaMalloc((void**)&d_occslst,    occs_size));
    CUDA_CHECK(cudaMalloc((void**)&d_clink,      link_size));
    CUDA_CHECK(cudaMalloc((void**)&d_link_nnorb, linknn_size));
    CUDA_CHECK(cudaMalloc((void**)&d_jdiag,      norb_sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_kdiag,      norb_sq * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_eri,        nnorb * nnorb * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_ci0,        base_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ci1,        base_size));
    CUDA_CHECK(cudaMalloc((void**)&d_tmp,        base_size));
    CUDA_CHECK(cudaMalloc((void**)&d_heff,       heff_size * heff_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_heff_tmp,   heff_size * heff_size * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_heff_tmp, 0,         heff_size * heff_size * sizeof(double)));
    // in_cpu=0 fixed: GPU memory only
    CUDA_CHECK(cudaMalloc((void**)&d_ci0_list, list_size));
    CUDA_CHECK(cudaMalloc((void**)&d_ci1_list, list_size));
    CUDA_CHECK(cudaMalloc((void**)&d_hdiag,    base_size));

    // =========================================================
    // NCCL broadcast (only when nprocs>1)
    // Distribute d_Gmo / d_Gmo1e from rank=0 to all ranks
    // =========================================================
    if (nprocs > 1) {
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclBcast(rank == 0 ? d_Gmo   : d_Gmo_local,
                             norb4,   ncclDouble, 0, ncclComm, computeStream));
        NCCL_CHECK(ncclBcast(rank == 0 ? d_Gmo1e : d_Gmo1e_local,
                             norb_sq, ncclDouble, 0, ncclComm, transferStream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        CUDA_CHECK(cudaStreamSynchronize(transferStream));
    }

    // Transfer occslst from host to device
    CUDA_CHECK(cudaEventCreate(&occs_ready));
    CUDA_CHECK(cudaMemcpyAsync(d_occslst, occslst, occs_size, cudaMemcpyHostToDevice, transferStream));

    if (debug_mode > 1) {
        CUDA_CHECK(cudaStreamSynchronize(transferStream));
        clock_gettime(CLOCK_MONOTONIC, &ts_end_init);
        t_init = (float)(ts_end_init.tv_sec  - ts_begin.tv_sec) +
                 (float)(ts_end_init.tv_nsec - ts_begin.tv_nsec) / 1e9f;
    }

    // =========================================================
    // Compute J/K diagonal elements
    // =========================================================
    CUDA_CHECK(cudaEventRecord(occs_ready, transferStream));
    CUDA_CHECK(cudaStreamWaitEvent(computeStream, occs_ready, 0));
    jkcopy_kernel<<<norb_sq, 1, 0, computeStream>>>(
        d_Gmo_local, d_jdiag, d_kdiag, norb, norb_sq, norb_t);
    CUDA_CHECK(cudaGetLastError());

    // =========================================================
    // Compute Hamiltonian diagonal elements (in_cpu=0 fixed: GPU path only)
    // =========================================================
    FCImake_hdiag_uhf_part_kernel_large<<<large_blocks, nthread, 0, computeStream>>>(
        d_hdiag, mynp, d_Gmo1e_local, d_jdiag, d_kdiag,
        norb, my_na, na, amin, neleca, d_occslst, rank);
    CUDA_CHECK(cudaGetLastError());

    // =========================================================
    // Generate link indices and process 2-electron integrals
    // =========================================================
    absorb_h1e(d_Gmo1e_local, d_Gmo_local, d_eri, norb, nelec, nnorb, 0.5);

    if (nprocs > 1) {
        CUDA_CHECK(cudaMemset(d_clink,      0, link_size));
        CUDA_CHECK(cudaMemset(d_link_nnorb, 0, linknn_size));
        gen_linkstr_index(neleca, norb, my_na, amin, na, d_occslst, d_clink, d_link_nnorb);
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclBcast(d_eri,
                             nnorb * nnorb,
                             ncclDouble, 0, ncclComm, computeStream));
        NCCL_CHECK(ncclAllReduce(d_clink, d_clink,
                                 (size_t)nlinka * (size_t)na * 3,
                                 ncclInt32, ncclSum, ncclComm, computeStream));
        NCCL_CHECK(ncclAllReduce(d_link_nnorb, d_link_nnorb,
                                 (size_t)na * (size_t)nnorb * 2,
                                 ncclInt32, ncclSum, ncclComm, computeStream));
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        NCCL_CHECK(ncclGroupEnd());
    } else {
        gen_linkstr_index(neleca, norb, na, 0, na, d_occslst, d_clink, d_link_nnorb);
    }

    // Free unused GPU memory early
    CUDA_CHECK(cudaFree(d_jdiag));   d_jdiag   = NULL;
    CUDA_CHECK(cudaFree(d_kdiag));   d_kdiag   = NULL;
    CUDA_CHECK(cudaFree(d_occslst)); d_occslst = NULL;
    if (owns_Gmo) {
        CUDA_CHECK(cudaFree(d_Gmo1e_local)); d_Gmo1e_local = NULL;
        CUDA_CHECK(cudaFree(d_Gmo_local));   d_Gmo_local   = NULL;
    }

    if (debug_mode > 1) {
        CUDA_CHECK(cudaStreamSynchronize(computeStream));
        clock_gettime(CLOCK_MONOTONIC, &ts_end_other);
        t_other_init = (float)(ts_end_other.tv_sec  - ts_end_init.tv_sec) +
                       (float)(ts_end_other.tv_nsec - ts_end_init.tv_nsec) / 1e9f;
    }

    // =========================================================
    // Initialize CI vectors
    // =========================================================
    {
        CUDA_CHECK(cudaMemset(d_ci0, 0, base_size));
        const double firstv =  1.0 + 1e-5;
        const double lastv  =  0.0 - 1e-5;

        if (rank == 0) {
            CUDA_CHECK(cudaMemcpy(&d_ci0[0],      &firstv, sizeof(double), cudaMemcpyHostToDevice));
        }
        if (rank == nprocs - 1) {
            CUDA_CHECK(cudaMemcpy(&d_ci0[mynp-1], &lastv,  sizeof(double), cudaMemcpyHostToDevice));
        }

        // =========================================================
        // Davidson eigenvalue solver
        // =========================================================
        SolverConfig cfg;
        cfg.na         = na;
        cfg.na_self    = my_na;
        cfg.norb       = norb;
        cfg.nelec      = neleca;
        cfg.nlinka     = nlinka;
        cfg.nnorb      = nnorb;
        cfg.ntile      = chunk;
        cfg.np         = np;
        cfg.nroots     = nroots;
        cfg.max_space  = max_space;
        cfg.max_cycle  = max_cycle;
        cfg.heff_size  = heff_size;
        cfg.tol        = tol;
        cfg.E_rhf      = E_rhf;
        cfg.counts_na  = counts_na;
        cfg.starts_na  = starts_na;
        cfg.ends_na    = ends_na;
        cfg.in_cpu     = in_cpu;
        cfg.debug_mode = debug_mode;
        cfg.rank       = rank;
        cfg.nprocs     = nprocs;

        davidson(
            d_ci0_list, d_ci1_list, nullptr, nullptr,
            d_ci0, d_ci1, d_eri, d_hdiag, nullptr,
            d_clink, d_link_nnorb, e, nullptr, d_tmp, d_heff, d_heff_tmp,
            cfg, ncclComm, computeStream);

        if (rank == 0) printf("E_fci: %.15lf, nprocs: %d\n\n", e[0] + E_rhf, nprocs);

        if (debug_mode > 1) {
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            t_davidson = (float)(ts_end.tv_sec  - ts_end_other.tv_sec) +
                         (float)(ts_end.tv_nsec - ts_end_other.tv_nsec) / 1e9f;
            t_total    = (float)(ts_end.tv_sec  - ts_begin.tv_sec) +
                         (float)(ts_end.tv_nsec - ts_begin.tv_nsec) / 1e9f;
        }
        
        if (debug_mode > 1 && rank == 0) {
            printf("==================================================\n");
            printf("FCI solver completed:\n");
            printf("  Final energy: %.15f\n", e[0] + E_rhf);
            printf("  FCI time:   %.3f s\n", t_total);
            printf("  Timing breakdown:\n");
            printf("    MPI init + GPU alloc/copy:  %.3f s (%.1f%%)\n",
                   t_init,       100.0f * t_init       / t_total);
            printf("    Other initialization:       %.3f s (%.1f%%)\n",
                   t_other_init, 100.0f * t_other_init / t_total);
            printf("    Davidson solver:            %.3f s (%.1f%%)\n",
                   t_davidson,   100.0f * t_davidson   / t_total);
            printf("==================================================\n");
        }
        E_rhf = e[0];
    }

cleanup:
    // =========================================================
    // Resource cleanup
    // =========================================================
    auto safe_free_host = [](void *&ptr) {
        if (ptr) { free(ptr); ptr = nullptr; }
    };
    auto safe_free_dev = [](void *&ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };

    safe_free_host((void*&)occslst);
    safe_free_host((void*&)counts_na);
    safe_free_host((void*&)starts_na);
    safe_free_host((void*&)ends_na);
    safe_free_host((void*&)e);

    safe_free_dev((void*&)d_heff);
    safe_free_dev((void*&)d_heff_tmp);
    safe_free_dev((void*&)d_ci0);
    safe_free_dev((void*&)d_ci1);
    safe_free_dev((void*&)d_tmp);
    safe_free_dev((void*&)d_link_nnorb);
    safe_free_dev((void*&)d_clink);
    safe_free_dev((void*&)d_eri);
    safe_free_dev((void*&)d_jdiag);
    safe_free_dev((void*&)d_kdiag);
    safe_free_dev((void*&)d_occslst);
    // in_cpu=0 fixed
    safe_free_dev((void*&)d_ci0_list);
    safe_free_dev((void*&)d_ci1_list);
    safe_free_dev((void*&)d_hdiag);
    // Free only buffers allocated by rank!=0
    if (owns_Gmo) {
        safe_free_dev((void*&)d_Gmo1e_local);
        safe_free_dev((void*&)d_Gmo_local);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (computeStream) cudaStreamDestroy(computeStream);
    if (transferStream) cudaStreamDestroy(transferStream);

    if (nprocs > 1 && ncclComm)     ncclCommDestroy(ncclComm);
    if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
    if (!mpi_initialized) { MPI_Finalize(); }
    return E_rhf;
}

#endif // GANSU_MPI




