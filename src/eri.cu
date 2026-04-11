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


#include "eri.hpp"
#include "utils_cuda.hpp"
#include "rys_eri.hpp"
#include "gpu_kernels.hpp"
#include "gpu_manager.hpp"
#include "ao2mo.cuh"
#include <cassert>
#include <cmath>
#ifndef GANSU_CPU_ONLY
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#endif

#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

// CPU replacement for thrust::sort_by_key (descending order).
// Always defined so runtime (ENABLE_GPU=ON + --cpu) can use it too.
template <typename KeyT, typename ValueT>
void cpu_sort_by_key_descending(KeyT* keys, ValueT* values, size_t count) {
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return keys[a] > keys[b];
    });
    std::vector<KeyT> sorted_keys(count);
    std::vector<ValueT> sorted_values(count);
    for (size_t i = 0; i < count; i++) {
        sorted_keys[i] = keys[indices[i]];
        sorted_values[i] = values[indices[i]];
    }
    std::copy(sorted_keys.begin(), sorted_keys.end(), keys);
    std::copy(sorted_values.begin(), sorted_values.end(), values);
}

#ifdef GANSU_CPU_ONLY
// CPU replacement for thrust::sort_by_key (ascending, using operator<)
template <typename KeyT, typename ValueT>
void cpu_sort_by_key_ascending(KeyT* keys, ValueT* values, size_t count) {
    std::vector<size_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return keys[a] < keys[b];
    });
    std::vector<KeyT> sorted_keys(count);
    std::vector<ValueT> sorted_values(count);
    for (size_t i = 0; i < count; i++) {
        sorted_keys[i] = keys[indices[i]];
        sorted_values[i] = values[indices[i]];
    }
    std::copy(sorted_keys.begin(), sorted_keys.end(), keys);
    std::copy(sorted_values.begin(), sorted_values.end(), values);
}
#endif

struct HashKeyIsNonEmpty {
    __host__ __device__ bool operator()(unsigned long long key) const {
        return key != 0xFFFFFFFFFFFFFFFFULL;
    }
};

namespace gansu{

__device__ inline size_t2 index1to2(const size_t index, bool is_symmetric, size_t num_basis=0){
//    assert(is_symmetric or num_basis > 0);
    if(is_symmetric){
        const size_t r2 = __double2ll_rd((__dsqrt_rn(8 * index + 1) - 1) / 2);
        const size_t r1 = index - r2 * (r2 + 1) / 2;
        return {r1, r2};
    }else{
        return {index / num_basis, index % num_basis};
    }
}

__global__ void generatePrimitiveShellPairIndices(size_t2* d_indices_array, size_t num_threads, bool is_symmetric, size_t num_basis){
    const size_t id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_threads) return;
    d_indices_array[id] = index1to2(id, is_symmetric, num_basis);
}


__global__ void generatePrimitiveShellPairIndices(size_t2* d_indices_array, size_t num_threads, bool is_symmetric, size_t num_basis, bool if_full_range, size_t start_index_a, size_t start_index_b){
    const size_t id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_threads) return;
    d_indices_array[id] = index1to2(id, is_symmetric, num_basis);


    d_indices_array[id].x += start_index_a;
    d_indices_array[id].y += start_index_b;
}

//__global__ void initializePrimitiveShellPairIndices(int2* d_indices_array, int num_threads, bool is_symmetric, int num_basis) {
//    const int id = blockDim.x * blockIdx.x + threadIdx.x;
//    if (id >= num_threads) return;
//    size_t2 index_pair = index1to2(id, is_symmetric, num_basis);
//    d_indices_array[id] = make_int2(static_cast<int>(index_pair.x), static_cast<int>(index_pair.y));
//}
__global__ void initializePrimitiveShellPairIndices(
    int2* d_indices_array, int num_threads, bool is_symmetric, 
    //int num_shells)
    int num_shells, const size_t start_index_a, const size_t start_index_b)
    {
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_threads) return;
    size_t2 index_pair = index1to2(id, is_symmetric, num_shells);
    //d_indices_array[id] = make_int2(static_cast<int>(index_pair.x), static_cast<int>(index_pair.y));
    d_indices_array[id] = make_int2(static_cast<int>(index_pair.x + start_index_a), static_cast<int>(index_pair.y + start_index_b));
}


ERI_Stored::ERI_Stored(const HF& hf): 
        hf_(hf),
        num_basis_(hf.get_num_basis()),
        eri_matrix_(num_basis_*num_basis_, num_basis_*num_basis_),
        schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs())
{
    // nothing to do
}


void ERI_Stored::precomputation() {
    // compute the electron repulsion integrals
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int verbose = hf_.get_verbose();

    // Compute Schwarz Upper Bounds
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );


    //gpu::computeERIMatrix(shell_type_infos, primitive_shells.device_ptr(), boys_grid.device_ptr(), cgto_normalization_factors.device_ptr(), eri_matrix_.device_ptr(), schwarz_screening_threshold, num_basis_, verbose);

    gpu::computeERIMatrix(
        shell_type_infos, 
        shell_pair_type_infos, 
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(),
        cgto_normalization_factors.device_ptr(),   
        eri_matrix_.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold, 
        num_basis_, 
        verbose
        );

    // print the eri matrix
    if(verbose){
        // copy the eri matrix to the host memory
        eri_matrix_.toHost();

        std::cout << "ERI matrix:" << std::endl;
        for(int l=0; l<num_basis_; l++){
            for(int k=0; k<=l; k++){
                for(int j=0; j<=l; j++){
                    const auto i_max = (l==j) ? k : j;
                    for(int i=0; i<=i_max; i++){
                        std::cout << "i: " << i << ", j: " << j << ", k: " << k << ", l: " << l << ": " << eri_matrix_(i*num_basis_+j, k*num_basis_+l) << std::endl;
                    }
                }
            }
        }
    }
}



ERI_RI::ERI_RI(const HF& hf, const Molecular& auxiliary_molecular): 
        hf_(hf),
        num_basis_(hf.get_num_basis()),
        num_auxiliary_basis_(auxiliary_molecular.get_num_basis()),
        num_occ_(hf.get_num_electrons() / 2),
        auxiliary_shell_type_infos_(auxiliary_molecular.get_shell_type_infos()),
        auxiliary_primitive_shells_(auxiliary_molecular.get_primitive_shells()),
        auxiliary_cgto_normalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
        intermediate_matrix_B_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_J_(num_basis_, num_basis_),
        d_K_(num_basis_, num_basis_),
        d_W_tmp_(num_auxiliary_basis_),
        //d_T_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        //d_V_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_tmp1_(
            num_auxiliary_basis_,
            hf_.get_hasMatrixC()
                ? num_basis_ * num_occ_
                : num_basis_ * num_basis_
        ),
        d_tmp2_(
            hf_.get_hasMatrixC() ? num_basis_ : num_auxiliary_basis_,
            hf_.get_hasMatrixC()
                ? num_auxiliary_basis_ * num_occ_
                : num_basis_ * num_basis_
        ),
        schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
        auxiliary_schwarz_upper_bound_factors(auxiliary_molecular.get_primitive_shells().size())
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
}

ERI_RI::~ERI_RI() {
    if (d_eri_reconstructed_) tracked_cudaFree(d_eri_reconstructed_);
}

void ERI_RI::reconstruct_ao_eri() {
    if (eri_reconstructed_) return;

    const size_t nao2 = (size_t)num_basis_ * num_basis_;
    const size_t required_bytes = nao2 * nao2 * sizeof(real_t);

    // Memory check
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (required_bytes > free_mem * 8 / 10) {
        THROW_EXCEPTION("Not enough GPU memory to reconstruct AO ERI from RI. "
            "Required: " + std::to_string(required_bytes / (1024*1024)) + " MB, "
            "Available: " + std::to_string(free_mem / (1024*1024)) + " MB. "
            "Use stored ERI method for post-HF on this system.");
    }

    tracked_cudaMalloc(&d_eri_reconstructed_, required_bytes);

    // ERI[nao^2 x nao^2] = B^T * B  where B is (naux x nao^2) row-major.
    // matrixMatrixProductRect handles both GPU (cuBLAS) and CPU (Eigen) paths.
    // Row-major view: ERI_rm(nao², nao²) = B_rm^T(nao², naux) * B_rm(naux, nao²)
    gpu::matrixMatrixProductRect(
        intermediate_matrix_B_.device_ptr(),   // (naux, nao²) row-major
        intermediate_matrix_B_.device_ptr(),   // (naux, nao²) row-major
        d_eri_reconstructed_,                  // (nao², nao²) row-major
        (int)nao2, (int)nao2, num_auxiliary_basis_,
        /*transpose_A=*/true, /*transpose_B=*/false,
        /*accumulate=*/false, /*alpha=*/1.0);
    cudaDeviceSynchronize();

    eri_reconstructed_ = true;

    if (hf_.get_verbose()) {
        std::cout << "[RI] Reconstructed AO ERI: " << nao2 << " x " << nao2
                  << " (" << required_bytes / (1024*1024) << " MB)" << std::endl;
    }
}

// ============================================================
// ERI base class default implementations
// ============================================================

real_t* ERI::build_mo_eri(const real_t* d_C, int nmo) const {
    const real_t* d_eri_ao = get_eri_matrix_device();
    if (!d_eri_ao) {
        THROW_EXCEPTION("build_mo_eri: no AO ERI available for this ERI method.");
    }
    const size_t N = nmo;
    const size_t n4 = N * N * N * N;
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_mo, n4 * sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU: 4-index AO→MO transform using O(N^5) algorithm
        // (pq|rs) = sum_{mu,nu,la,si} C(mu,p)*C(nu,q)*(mu nu|la si)*C(la,r)*C(si,s)
        // Step-by-step quarter transforms to keep O(N^5):
        real_t* tmp1 = nullptr;
        real_t* tmp2 = nullptr;
        tracked_cudaMalloc(&tmp1, n4 * sizeof(real_t));
        tracked_cudaMalloc(&tmp2, n4 * sizeof(real_t));

        // Step 1: contract 4th index: tmp1(mu,nu,la,s) = sum_si eri(mu,nu,la,si) * C(si,s)
        std::memset(tmp1, 0, n4 * sizeof(real_t));
        #pragma omp parallel for schedule(dynamic)
        for (size_t mu = 0; mu < N; mu++)
            for (size_t nu = 0; nu < N; nu++)
                for (size_t la = 0; la < N; la++)
                    for (size_t s = 0; s < N; s++) {
                        double val = 0.0;
                        for (size_t si = 0; si < N; si++)
                            val += d_eri_ao[mu*N*N*N + nu*N*N + la*N + si] * d_C[si*N + s];
                        tmp1[mu*N*N*N + nu*N*N + la*N + s] = val;
                    }

        // Step 2: contract 3rd index: tmp2(mu,nu,r,s) = sum_la tmp1(mu,nu,la,s) * C(la,r)
        std::memset(tmp2, 0, n4 * sizeof(real_t));
        #pragma omp parallel for schedule(dynamic)
        for (size_t mu = 0; mu < N; mu++)
            for (size_t nu = 0; nu < N; nu++)
                for (size_t r = 0; r < N; r++)
                    for (size_t s = 0; s < N; s++) {
                        double val = 0.0;
                        for (size_t la = 0; la < N; la++)
                            val += tmp1[mu*N*N*N + nu*N*N + la*N + s] * d_C[la*N + r];
                        tmp2[mu*N*N*N + nu*N*N + r*N + s] = val;
                    }

        // Step 3: contract 2nd index: tmp1(mu,q,r,s) = sum_nu tmp2(mu,nu,r,s) * C(nu,q)
        std::memset(tmp1, 0, n4 * sizeof(real_t));
        #pragma omp parallel for schedule(dynamic)
        for (size_t mu = 0; mu < N; mu++)
            for (size_t q = 0; q < N; q++)
                for (size_t r = 0; r < N; r++)
                    for (size_t s = 0; s < N; s++) {
                        double val = 0.0;
                        for (size_t nu = 0; nu < N; nu++)
                            val += tmp2[mu*N*N*N + nu*N*N + r*N + s] * d_C[nu*N + q];
                        tmp1[mu*N*N*N + q*N*N + r*N + s] = val;
                    }

        // Step 4: contract 1st index: eri_mo(p,q,r,s) = sum_mu tmp1(mu,q,r,s) * C(mu,p)
        std::memset(d_eri_mo, 0, n4 * sizeof(real_t));
        #pragma omp parallel for schedule(dynamic)
        for (size_t p = 0; p < N; p++)
            for (size_t q = 0; q < N; q++)
                for (size_t r = 0; r < N; r++)
                    for (size_t s = 0; s < N; s++) {
                        double val = 0.0;
                        for (size_t mu = 0; mu < N; mu++)
                            val += d_C[mu*N + p] * tmp1[mu*N*N*N + q*N*N + r*N + s];
                        d_eri_mo[p*N*N*N + q*N*N + r*N + s] = val;
                    }

        tracked_cudaFree(tmp1);
        tracked_cudaFree(tmp2);
        return d_eri_mo;
    }

    // GPU path
    real_t* d_eri_work = nullptr;
    tracked_cudaMalloc(&d_eri_work, n4 * sizeof(real_t));
    cudaMemcpy(d_eri_work, d_eri_ao, n4 * sizeof(real_t), cudaMemcpyDeviceToDevice);
    transform_eri_ao2mo_dgemm_full(d_eri_work, d_eri_mo, d_C, nmo);
    tracked_cudaFree(d_eri_work);
    return d_eri_mo;
}

void ERI::compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const {
    const real_t* d_eri_ao = get_eri_matrix_device();
    if (!d_eri_ao) {
        THROW_EXCEPTION("compute_jk_response: no AO ERI available for this ERI method.");
    }
    real_t* d_zero = nullptr;
    tracked_cudaMalloc(&d_zero, (size_t)nao * nao * sizeof(real_t));
    cudaMemset(d_zero, 0, (size_t)nao * nao * sizeof(real_t));
    // computeFockMatrix_RHF computes F = H + 2J - K. With H=0, output = 2J - K.
    gpu::computeFockMatrix_RHF(d_D, d_zero, d_eri_ao, d_G, nao);
    tracked_cudaFree(d_zero);
}

// ============================================================
// ERI_RI implementations
// ============================================================

const real_t* ERI_RI::get_eri_matrix_device() const {
    if (!eri_reconstructed_) {
        const_cast<ERI_RI*>(this)->reconstruct_ao_eri();
    }
    return d_eri_reconstructed_;
}

void ERI_RI::compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const {
    // Use RI B-matrix based Fock build: no AO ERI reconstruction needed.
    // Allocate temporary buffers locally (CPHF is called infrequently).
    const size_t nao2 = (size_t)nao * nao;
    real_t* d_zero = nullptr;
    real_t* d_J = nullptr;
    real_t* d_K = nullptr;
    real_t* d_W = nullptr;
    real_t* d_T = nullptr;
    real_t* d_V = nullptr;
    tracked_cudaMalloc(&d_zero, nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_J, nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_K, nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_W, num_auxiliary_basis_ * sizeof(real_t));
    tracked_cudaMalloc(&d_T, (size_t)num_auxiliary_basis_ * nao2 * sizeof(real_t));
    tracked_cudaMalloc(&d_V, (size_t)num_auxiliary_basis_ * nao2 * sizeof(real_t));
    cudaMemset(d_zero, 0, nao2 * sizeof(real_t));

    gpu::computeFockMatrix_RI_RHF_with_density_matrix(
        d_D, d_zero, intermediate_matrix_B_.device_ptr(),
        d_G, num_basis_, num_auxiliary_basis_,
        d_J, d_K, d_W, d_T, d_V);

    tracked_cudaFree(d_zero);
    tracked_cudaFree(d_J);
    tracked_cudaFree(d_K);
    tracked_cudaFree(d_W);
    tracked_cudaFree(d_T);
    tracked_cudaFree(d_V);
}

real_t* ERI_RI::build_mo_eri(const real_t* d_C, int nmo) const {
    // CPU fallback: delegate to the base-class implementation which uses
    // get_eri_matrix_device() (via reconstruct_ao_eri) + O(N^5) quarter transform.
    if (!gpu::gpu_available()) {
        return ERI::build_mo_eri(d_C, nmo);
    }

    // Build MO ERI directly: B(Q,μν) → B_mo(Q,pq) → (pq|rs) = B_mo^T B_mo
    // Avoids the nao⁴ AO ERI intermediate.
    //
    // Step 1: Right half-transform  B(Q,μ,ν) · C(ν,q) → B_tmp(Q,μ,q)
    // Step 2: Left half-transform   C(μ,p)^T · B_tmp(Q,μ,q) → B_mo(Q,p,q)
    // Step 3: MO ERI = B_mo^T · B_mo
    //
    // B layout: row-major (naux, nao, nao) → cuBLAS sees column-major (nao, nao*naux)
    // C layout: row-major (nao, nmo) → cuBLAS sees column-major (nmo, nao)

    const int nao = num_basis_;
    const int naux = num_auxiliary_basis_;
    const size_t nmo2 = (size_t)nmo * nmo;

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta = 0.0;

    // Allocate workspace: B_tmp (naux * nao * nmo)
    real_t* d_B_tmp = nullptr;
    tracked_cudaMalloc(&d_B_tmp, (size_t)naux * nao * nmo * sizeof(real_t));

    // Step 1: Right half-transform (batched over Q slices)
    // For each Q: B_tmp(Q, μ, q) = Σ_ν B(Q, μ, ν) · C(ν, q)
    // Treat as single DGEMM: B is (naux*nao, nao), C is (nao, nmo)
    // Result: B_tmp is (naux*nao, nmo)
    // cuBLAS col-major: C_cm(nmo,nao) * B_cm(nao, naux*nao) → impossible directly
    // Instead: B_tmp = B · C in row-major = C^T · B^T in col-major
    // Actually simpler: treat as batched with naux*nao rows
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nao, nao,
        &alpha,
        d_C, nmo,           // C is (nao, nmo) row-major → col-major (nmo, nao)
        intermediate_matrix_B_.device_ptr(), nao,  // B is (naux*nao, nao) row-major → col-major (nao, naux*nao)
        &beta,
        d_B_tmp, nmo);       // result (naux*nao, nmo) row-major → col-major (nmo, naux*nao)

    // Step 2: Left half-transform
    // For each Q: B_mo(Q, p, q) = Σ_μ C(μ, p) · B_tmp(Q, μ, q)
    // B_tmp is (naux, nao, nmo) → treat as naux*nmo columns of length nao
    // B_mo  is (naux, nmo, nmo) → treat as naux*nmo columns of length nmo
    //
    // We need to transpose B_tmp from (naux*nao, nmo) to (nmo, naux*nao)
    // then multiply C^T · B_tmp_reshaped. But this is complex.
    //
    // Alternative: use the transposed approach.
    // B_tmp is stored as (Q*nao+μ, q) in row-major, i.e., (nmo, naux*nao) in col-major.
    // We want B_mo(Q*nmo+p, q) = Σ_μ C(μ,p) B_tmp(Q*nao+μ, q)
    //
    // For fixed q: B_mo(:,q) = [C^T ⊗ I_naux] B_tmp(:,q) ... this needs a reshape.
    //
    // Simplest: transpose B_tmp to get B_tmp2(Q, q, μ) then multiply by C
    // B_tmp2(naux*nmo, nao) then B_mo2 = B_tmp2 · C → (naux*nmo, nmo)

    real_t* d_B_tmp2 = nullptr;
    tracked_cudaMalloc(&d_B_tmp2, (size_t)naux * nao * nmo * sizeof(real_t));

    // Transpose: for each Q, transpose (nao, nmo) → (nmo, nao)
    // B_tmp[Q*nao*nmo + μ*nmo + q] → B_tmp2[Q*nmo*nao + q*nao + μ]
    // Use cublasDgeam for batch transpose
    for (int Q = 0; Q < naux; Q++) {
        cublasDgeam(handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            nao, nmo,
            &alpha,
            d_B_tmp + (size_t)Q * nao * nmo, nmo,  // col-major (nmo, nao) → transpose to (nao, nmo)
            &beta,
            nullptr, nao,
            d_B_tmp2 + (size_t)Q * nmo * nao, nao); // result (nao, nmo) in col-major
    }

    tracked_cudaFree(d_B_tmp);

    // Now B_tmp2 is (naux*nmo, nao) in row-major = (nao, naux*nmo) in col-major
    // Step 2 DGEMM: B_mo = B_tmp2 · C → (naux*nmo, nmo) in row-major
    real_t* d_B_mo = nullptr;
    tracked_cudaMalloc(&d_B_mo, (size_t)naux * nmo2 * sizeof(real_t));

    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        nmo, (long long)naux * nmo, nao,
        &alpha,
        d_C, nmo,            // (nmo, nao) col-major
        d_B_tmp2, nao,       // (nao, naux*nmo) col-major
        &beta,
        d_B_mo, nmo);        // (nmo, naux*nmo) col-major = (naux*nmo, nmo) row-major

    tracked_cudaFree(d_B_tmp2);

    // Step 3: MO ERI = B_mo^T · B_mo
    // B_mo is (naux, nmo²) row-major = (nmo², naux) col-major
    // ERI = B_cm · B_cm^T = (nmo², naux) × (naux, nmo²) = (nmo², nmo²) col-major
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_mo, nmo2 * nmo2 * sizeof(real_t));

    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nmo2, nmo2, naux,
        &alpha,
        d_B_mo, nmo2,
        d_B_mo, nmo2,
        &beta,
        d_eri_mo, nmo2);

    tracked_cudaFree(d_B_mo);
    cudaDeviceSynchronize();

    if (hf_.get_verbose()) {
        std::cout << "[RI] Built MO ERI directly: " << nmo << "^4 = "
                  << (nmo2 * nmo2 * sizeof(real_t)) / (1024*1024) << " MB"
                  << " (skipped nao^4 = " << ((size_t)nao*nao*nao*nao*sizeof(real_t))/(1024*1024) << " MB)" << std::endl;
    }

    return d_eri_mo;
}

void ERI_RI::precomputation() {
    // compute the intermediate matrix B of the auxiliary basis functions
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();

    // compute upper bounds of primitive-shell-pair
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );


    const size_t num_primitive_shell_pairs = primitive_shells.size() * (primitive_shells.size() + 1) / 2;
    size_t2* d_primitive_shell_pair_indices;
    // Use tracked_cudaMalloc so hybrid --cpu mode falls back to host calloc.
    tracked_cudaMalloc(&d_primitive_shell_pair_indices, sizeof(size_t2) * num_primitive_shell_pairs);

    int pair_idx = 0;
    const int threads_per_block = 1024;
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const size_t count = shell_pair_type_infos[pair_idx].count;
            const size_t start = shell_pair_type_infos[pair_idx].start_index;

#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                const int num_blocks = (count + threads_per_block - 1) / threads_per_block;
                generatePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[start], count, s0 == s1, shell_type_infos[s1].count);
                thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[start]);
                thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[start] + count);
                thrust::device_ptr<size_t2> values_begin(&d_primitive_shell_pair_indices[start]);
                thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
            } else
#endif
            {
                // === CPU path: generate indices in original order, skip Schwarz sort ===
                // In CPU mode schwarz_upper_bound_factors are all 1.0 (no screening),
                // so a sort would only scramble primitives to no effect.  Leave them
                // in basis-index order so group_contracted_shells sees them in the
                // same sequence as the untouched GPU case (important for determinism).
                size_t2* h_pair = &d_primitive_shell_pair_indices[start];
                const bool is_symmetric = (s0 == s1);
                const size_t ns1 = shell_type_infos[s1].count;
                for (size_t id = 0; id < count; ++id) {
                    size_t r1, r2;
                    if (is_symmetric) {
                        r2 = (size_t)((std::sqrt(8.0 * (double)id + 1.0) - 1.0) / 2.0);
                        while ((r2 + 1) * (r2 + 2) / 2 <= id) ++r2;
                        while (r2 > 0 && r2 * (r2 + 1) / 2 > id) --r2;
                        r1 = id - r2 * (r2 + 1) / 2;
                    } else {
                        r1 = id / ns1;
                        r2 = id % ns1;
                    }
                    h_pair[id] = {r1, r2};
                }
            }

            pair_idx++;
        }
    }
    cudaDeviceSynchronize();




    // compute upper bounds of  aux-shell
    gpu::computeAuxiliarySchwarzUpperBounds(
        auxiliary_shell_type_infos_,
        auxiliary_primitive_shells_.device_ptr(),
        boys_grid.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),   // auxiliary_schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );

    for(const auto& s : auxiliary_shell_type_infos_){
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            thrust::device_ptr<real_t> keys_begin(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index]);
            thrust::device_ptr<real_t> keys_end(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index] + s.count);
            thrust::device_ptr<PrimitiveShell> values_begin(&auxiliary_primitive_shells_.device_ptr()[s.start_index]);
            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
        }
        // CPU path: skip sort (all Schwarz bounds = 1.0 in the stub).
#endif
    }


    // Zero-initialize B matrix before computation (defensive: prevent stale GPU memory)
    cudaMemset(intermediate_matrix_B_.device_ptr(), 0,
               (size_t)num_auxiliary_basis_ * num_basis_ * num_basis_ * sizeof(real_t));

    gpu::compute_RI_IntermediateMatrixB(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(),
        cgto_normalization_factors.device_ptr(),
        auxiliary_shell_type_infos_,
        auxiliary_primitive_shells_.device_ptr(),
        auxiliary_cgto_normalization_factors_.device_ptr(),
        intermediate_matrix_B_.device_ptr(),
        d_primitive_shell_pair_indices,
        schwarz_upper_bound_factors.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold,
        num_basis_,
        num_auxiliary_basis_,
        boys_grid.device_ptr(),
        verbose
        );


    tracked_cudaFree(d_primitive_shell_pair_indices);
    cudaDeviceSynchronize();
}



ERI_Direct::ERI_Direct(const HF& hf):
    hf_(hf),
    num_basis_(hf.get_num_basis()),
    schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
    primitive_shell_pair_indices(hf.get_num_primitive_shell_pairs()),
    num_fock_replicas_(8),
    density_matrix_diff_(num_basis_, num_basis_),
    density_matrix_diff_shell_(hf.get_num_primitive_shells(), hf.get_num_primitive_shells()),
    fock_matrix_prev_(num_basis_, num_basis_)
{
    // for distributed atomicAdd operations
    cudaMalloc(&fock_matrix_replicas_, sizeof(real_t) * num_basis_ * num_basis_ * num_fock_replicas_);
    //cudaMemset(fock_matrix_replicas_, 0.0, sizeof(real_t) * num_basis_ * num_basis_ * num_fock_replicas_);
}

ERI_Direct::~ERI_Direct() {
    for (auto p : global_counters_) { if (p) cudaFree(p); }
    for (auto p : min_skipped_columns_) { if (p) cudaFree(p); }
    global_counters_.clear();
    min_skipped_columns_.clear();

    if (fock_matrix_replicas_) {
        cudaFree(fock_matrix_replicas_);
        fock_matrix_replicas_ = nullptr;
    }
    if (d_eri_reconstructed_) tracked_cudaFree(d_eri_reconstructed_);
}

void ERI_Direct::reconstruct_ao_eri() {
    if (eri_reconstructed_) return;

    const size_t nao2 = (size_t)num_basis_ * num_basis_;
    const size_t required_bytes = nao2 * nao2 * sizeof(real_t);

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (required_bytes > free_mem * 8 / 10) {
        THROW_EXCEPTION("Not enough GPU memory to reconstruct AO ERI from Direct SCF. "
            "Required: " + std::to_string(required_bytes / (1024*1024)) + " MB, "
            "Available: " + std::to_string(free_mem / (1024*1024)) + " MB.");
    }

    tracked_cudaMalloc(&d_eri_reconstructed_, required_bytes);

    const auto& shell_type_infos = hf_.get_shell_type_infos();
    const auto& shell_pair_type_infos = hf_.get_shell_pair_type_infos();

    // Recompute unsorted Schwarz factors — Direct SCF's schwarz_upper_bound_factors
    // have been reordered by thrust::sort_by_key in precomputation(), so they cannot
    // be used for computeERIMatrix which expects the original shell-pair ordering.
    DeviceHostMemory<real_t> schwarz_unsorted(hf_.get_num_primitive_shell_pairs());
    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        hf_.get_primitive_shells().device_ptr(),
        hf_.get_boys_grid().device_ptr(),
        hf_.get_cgto_normalization_factors().device_ptr(),
        schwarz_unsorted.device_ptr(),
        false
    );

    gpu::computeERIMatrix(
        shell_type_infos,
        shell_pair_type_infos,
        hf_.get_primitive_shells().device_ptr(),
        hf_.get_boys_grid().device_ptr(),
        hf_.get_cgto_normalization_factors().device_ptr(),
        d_eri_reconstructed_,
        schwarz_unsorted.device_ptr(),
        hf_.get_schwarz_screening_threshold(),
        num_basis_,
        hf_.get_verbose()
    );

    eri_reconstructed_ = true;

    if (hf_.get_verbose()) {
        std::cout << "[Direct] Reconstructed AO ERI: " << nao2 << " x " << nao2
                  << " (" << required_bytes / (1024*1024) << " MB)" << std::endl;
    }
}

const real_t* ERI_Direct::get_eri_matrix_device() const {
    if (!eri_reconstructed_) {
        const_cast<ERI_Direct*>(this)->reconstruct_ao_eri();
    }
    return d_eri_reconstructed_;
}

void ERI_Direct::precomputation() 
{
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    // for dynamic Schwarz screening
    const int shell_type_count = shell_type_infos.size();
    std::vector<std::tuple<int, int, int, int>> shell_quadruples;
    for (int a = 0; a < shell_type_count; ++a) {
        for (int b = a; b < shell_type_count; ++b) {
            for (int c = 0; c < shell_type_count; ++c) {
                for (int d = c; d < shell_type_count; ++d) {
                    if (a < c || (a == c && b <= d)) {
                        shell_quadruples.emplace_back(a, b, c, d);
                    }
                }
            }
        }
    }
    const int task_group_size = 16;
    const int num_braket_types = shell_quadruples.size();
    global_counters_.resize(num_braket_types, nullptr);
    min_skipped_columns_.resize(num_braket_types, nullptr);
    int s0, s1, s2, s3;
    ShellTypeInfo shell_s0, shell_s1; //shell_s2, shell_s3;
    int num_bra, num_bra_groups;
    for (int idx = 0; idx < num_braket_types; ++idx) {
        std::tie(s0, s1, s2, s3) = shell_quadruples[idx];
        shell_s0 = shell_type_infos[s0];
        shell_s1 = shell_type_infos[s1];
        num_bra = (s0 == s1) ? shell_s0.count * (shell_s0.count + 1) / 2 : shell_s0.count * shell_s1.count;
        num_bra_groups = (num_bra + task_group_size - 1) / task_group_size;
        cudaMalloc(&global_counters_[idx], sizeof(int) * num_bra_groups);
        cudaMalloc(&min_skipped_columns_[idx], sizeof(int) * num_bra_groups);
    }

    gpu::computeSchwarzUpperBounds(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        schwarz_upper_bound_factors.device_ptr(), 
        verbose
        );

    // Create an array for storing pairs of primitive shell indices
    const size_t num_primitive_shell_pairs = primitive_shells.size() * (primitive_shells.size() + 1) / 2;
    int2* d_primitive_shell_pair_indices = primitive_shell_pair_indices.device_ptr();

    // Store the pairs of primitive shell indices and sort them based on the Schwarz upper bound factors
    int pair_idx = 0;
    const int threads_per_block = 1024;
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const size_t count = shell_pair_type_infos[pair_idx].count;
            const size_t start = shell_pair_type_infos[pair_idx].start_index;

#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                const int num_blocks = (count + threads_per_block - 1) / threads_per_block;
                initializePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[start], count, s0 == s1, shell_type_infos[s1].count, shell_type_infos[s0].start_index, shell_type_infos[s1].start_index);
                thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[start]);
                thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[start] + count);
                thrust::device_ptr<int2> values_begin(&d_primitive_shell_pair_indices[start]);
                thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
            } else
#endif
            {
                // === CPU path: generate indices on host, skip Schwarz sort ===
                // In CPU mode schwarz_upper_bound_factors are 1.0 stubs so
                // sorting is meaningless and std::sort could scramble order.
                int2* h_pair = &d_primitive_shell_pair_indices[start];
                const bool is_symmetric = (s0 == s1);
                const size_t ns1 = shell_type_infos[s1].count;
                const size_t off_a = shell_type_infos[s0].start_index;
                const size_t off_b = shell_type_infos[s1].start_index;
                for (size_t id = 0; id < count; ++id) {
                    size_t r1, r2;
                    if (is_symmetric) {
                        r2 = (size_t)((std::sqrt(8.0 * (double)id + 1.0) - 1.0) / 2.0);
                        while ((r2 + 1) * (r2 + 2) / 2 <= id) ++r2;
                        while (r2 > 0 && r2 * (r2 + 1) / 2 > id) --r2;
                        r1 = id - r2 * (r2 + 1) / 2;
                    } else {
                        r1 = id / ns1;
                        r2 = id % ns1;
                    }
                    h_pair[id] = make_int2((int)(r1 + off_a), (int)(r2 + off_b));
                }
            }
            pair_idx++;
        }
    }
    cudaDeviceSynchronize();

    // === CPU fallback: build the full 4D AO ERI tensor once ===
    // Direct SCF on GPU avoids storing the ERI for memory reasons, but on CPU
    // we fall back to the stored-ERI Fock path.  reconstruct_ao_eri() already
    // runs through gpu::computeERIMatrix which has a CPU implementation.
    if (!gpu::gpu_available()) {
        reconstruct_ao_eri();
    }
}


ERI_Hash::ERI_Hash(const HF& hf):
    hf_(hf),
    num_basis_(hf.get_num_basis()),
    d_coo_keys_(nullptr),
    d_coo_values_(nullptr),
    num_entries_(0),
    d_hash_keys_(nullptr),
    d_hash_values_(nullptr),
    hash_capacity_mask_(0),
    d_nonzero_indices_(nullptr),
    num_nonzero_(0)
{
}

ERI_Hash::~ERI_Hash() {
    if (d_eri_cpu_tensor_) tracked_cudaFree(d_eri_cpu_tensor_);
    if (d_coo_keys_) tracked_cudaFree(d_coo_keys_);
    if (d_coo_values_) tracked_cudaFree(d_coo_values_);
    if (d_hash_keys_) tracked_cudaFree(d_hash_keys_);
    if (d_hash_values_) tracked_cudaFree(d_hash_values_);
    if (d_nonzero_indices_) tracked_cudaFree(d_nonzero_indices_);
}

void ERI_Hash::compute_jk_response(const real_t* d_D, real_t* d_G, int nao) const {
    // G = 2J[D] - K[D] = Fock(D, H=0) using Hash ERI
    // Use Compact (COO) method — always available after precomputation
    real_t* d_zero = nullptr;
    tracked_cudaMalloc(&d_zero, (size_t)nao * nao * sizeof(real_t));
    cudaMemset(d_zero, 0, (size_t)nao * nao * sizeof(real_t));
    gpu::computeFockMatrix_Hash_RHF(d_D, d_zero,
        d_coo_keys_, d_coo_values_, num_entries_,
        d_G, nao, false);
    tracked_cudaFree(d_zero);
}

// Forward declaration of kernel in eri_stored.cu
__global__ void hash_coo_to_dense_kernel(
    const unsigned long long* g_coo_keys, const double* g_coo_values,
    size_t num_entries, double* g_eri_dense, int nao);

real_t* ERI_Hash::build_mo_eri(const real_t* d_C, int nmo) const {
    // CPU fallback: on CPU we skip the hash machinery entirely (see
    // ERI_Hash::precomputation) and cache the full AO ERI tensor in
    // d_eri_cpu_tensor_.  The base class build_mo_eri will pick it up
    // via get_eri_matrix_device() and run the O(N^5) transform.
    if (!gpu::gpu_available()) {
        return ERI::build_mo_eri(d_C, nmo);
    }

    const size_t n4 = (size_t)nmo * nmo * nmo * nmo;

    // Step 1: COO → dense AO ERI (O(nnz) expansion)
    real_t* d_eri_work = nullptr;
    tracked_cudaMalloc(&d_eri_work, n4 * sizeof(real_t));
    cudaMemset(d_eri_work, 0, n4 * sizeof(real_t));
    {
        const int threads = 256;
        const int blocks = ((int)num_entries_ + threads - 1) / threads;
        hash_coo_to_dense_kernel<<<blocks, threads>>>(
            d_coo_keys_, d_coo_values_, num_entries_,
            d_eri_work, nmo);
        cudaDeviceSynchronize();
    }

    // Step 2: AO → MO transform via DGEMM (same as Stored ERI)
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_mo, n4 * sizeof(real_t));
    transform_eri_ao2mo_dgemm_full(d_eri_work, d_eri_mo, d_C, nmo);
    tracked_cudaFree(d_eri_work);

    return d_eri_mo;
}

void ERI_Hash::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int verbose = hf_.get_verbose();

    // === CPU fallback: skip hash construction and build the full 4D tensor ===
    // The hash-table machinery below uses raw CUDA kernels that are not
    // exercised on CPU.  Instead, compute the dense AO ERI tensor once
    // via gpu::computeERIMatrix (which already has a CPU implementation)
    // and let ERI_Hash_RHF::compute_fock_matrix() reuse the stored-ERI path.
    if (!gpu::gpu_available()) {
        const size_t nao2 = (size_t)num_basis_ * num_basis_;
        const size_t tensor_bytes = nao2 * nao2 * sizeof(real_t);
        tracked_cudaMalloc(&d_eri_cpu_tensor_, tensor_bytes);
        cudaMemset(d_eri_cpu_tensor_, 0, tensor_bytes);

        DeviceHostMemory<real_t> schwarz_unsorted(hf_.get_num_primitive_shell_pairs());
        gpu::computeSchwarzUpperBounds(
            shell_type_infos, shell_pair_type_infos,
            primitive_shells.device_ptr(), boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            schwarz_unsorted.device_ptr(), verbose);

        gpu::computeERIMatrix(
            shell_type_infos, shell_pair_type_infos,
            primitive_shells.device_ptr(), boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            d_eri_cpu_tensor_, schwarz_unsorted.device_ptr(),
            schwarz_screening_threshold, num_basis_, verbose);

        // Leave the hash-related members null/zero; they are unused in CPU mode.
        return;
    }

    // Compute Schwarz upper bounds
    DeviceHostMemory<real_t> schwarz_upper_bound_factors(hf_.get_num_primitive_shell_pairs());
    gpu::computeSchwarzUpperBounds(
        shell_type_infos, shell_pair_type_infos,
        primitive_shells.device_ptr(), boys_grid.device_ptr(),
        cgto_normalization_factors.device_ptr(),
        schwarz_upper_bound_factors.device_ptr(), verbose);

    // === Phase 1: Allocate hash table ===
    const size_t N = num_basis_;
    const size_t num_pairs = N * (N + 1) / 2;
    size_t estimated_entries = num_pairs * (num_pairs + 1) / 2;

    // Hash table capacity: next power of 2, at least 2x estimated entries (load factor ~0.5)
    size_t hash_capacity = 1;
    while (hash_capacity < estimated_entries * 2) hash_capacity <<= 1;

    // Cap by GPU memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t bytes_per_entry = sizeof(unsigned long long) + sizeof(real_t);
    const size_t mem_cap_entries = (free_mem * 3 / 4) / bytes_per_entry;
    size_t mem_cap_pow2 = 1;
    while (mem_cap_pow2 * 2 <= mem_cap_entries) mem_cap_pow2 <<= 1;
    if (hash_capacity > mem_cap_pow2) hash_capacity = mem_cap_pow2;

    const size_t hash_capacity_mask = hash_capacity - 1;

    // Allocate hash table
    unsigned long long* d_hash_keys = nullptr;
    real_t* d_hash_values = nullptr;
    tracked_cudaMalloc(&d_hash_keys, hash_capacity * sizeof(unsigned long long));
    tracked_cudaMalloc(&d_hash_values, hash_capacity * sizeof(real_t));
    cudaMemset(d_hash_keys, 0xFF, hash_capacity * sizeof(unsigned long long));
    cudaMemset(d_hash_values, 0, hash_capacity * sizeof(real_t));

    if (verbose) {
        std::cout << "ERI Hash: capacity = " << hash_capacity << " entries ("
                  << (hash_capacity * bytes_per_entry) / (1024*1024) << " MB)" << std::endl;
    }

    // === Phase 2: Compute ERIs into hash table ===
    gpu::constructERIHash(
        shell_type_infos, shell_pair_type_infos,
        primitive_shells.device_ptr(), boys_grid.device_ptr(),
        cgto_normalization_factors.device_ptr(),
        schwarz_upper_bound_factors.device_ptr(), schwarz_screening_threshold,
        d_hash_keys, d_hash_values, hash_capacity_mask,
        num_basis_, verbose);

    // === Phase 3: Cleanup — remove near-zero entries ===
    const real_t cleanup_threshold = 1e-15;
    {
        const int threads_per_block = 256;
        const int num_blocks = (hash_capacity + threads_per_block - 1) / threads_per_block;
        gpu::cleanupHashTable_kernel<<<num_blocks, threads_per_block>>>(
            d_hash_keys, d_hash_values, hash_capacity, cleanup_threshold);
        cudaDeviceSynchronize();
    }

    // === Phase 4: Collect non-empty entries (method-dependent) ===
#ifdef GANSU_CPU_ONLY
    size_t num_nonzero = 0;
    for (size_t i = 0; i < hash_capacity; i++) {
        if (HashKeyIsNonEmpty()(d_hash_keys[i])) num_nonzero++;
    }
#else
    thrust::device_ptr<unsigned long long> keys_ptr(d_hash_keys);
    size_t num_nonzero = thrust::count_if(keys_ptr, keys_ptr + hash_capacity, HashKeyIsNonEmpty());
#endif

    // Keep hash table (needed for all methods including MP2 half-transform)
    d_hash_keys_ = d_hash_keys;
    d_hash_values_ = d_hash_values;
    hash_capacity_mask_ = hash_capacity_mask;
    num_nonzero_ = num_nonzero;  // Store for all methods (used by Indexed MP2 half-transform)

    // Build all auxiliary data structures for SCF Fock and MP2 half-transform
    // COO (Compact) — always built (needed for compute_jk_response and MP2 compact)
    tracked_cudaMalloc(&d_coo_keys_, num_nonzero * sizeof(unsigned long long));
    tracked_cudaMalloc(&d_coo_values_, num_nonzero * sizeof(real_t));
#ifdef GANSU_CPU_ONLY
    {
        size_t write_pos = 0;
        for (size_t i = 0; i < hash_capacity; i++) {
            if (HashKeyIsNonEmpty()(d_hash_keys[i])) {
                d_coo_keys_[write_pos] = d_hash_keys[i];
                d_coo_values_[write_pos] = d_hash_values[i];
                write_pos++;
            }
        }
    }
#else
    {
        thrust::device_ptr<real_t> vals_ptr(d_hash_values);
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_ptr, vals_ptr));
        auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(keys_ptr + hash_capacity, vals_ptr + hash_capacity));
        auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_ptr<unsigned long long>(d_coo_keys_),
            thrust::device_ptr<real_t>(d_coo_values_)));
        thrust::copy_if(zip_begin, zip_end, keys_ptr, out_zip, HashKeyIsNonEmpty());
    }
#endif
    num_entries_ = num_nonzero;

    // Indexed — always built (needed for MP2 indexed half-transform)
    tracked_cudaMalloc(&d_nonzero_indices_, num_nonzero * sizeof(size_t));
#ifdef GANSU_CPU_ONLY
    {
        size_t write_pos = 0;
        for (size_t i = 0; i < hash_capacity; i++) {
            if (HashKeyIsNonEmpty()(d_hash_keys[i])) {
                d_nonzero_indices_[write_pos++] = i;
            }
        }
    }
#else
    {
        thrust::counting_iterator<size_t> count_begin(0);
        thrust::device_ptr<size_t> out_indices(d_nonzero_indices_);
        thrust::copy_if(count_begin, count_begin + hash_capacity, keys_ptr, out_indices, HashKeyIsNonEmpty());
    }
#endif

    if (verbose) {
        std::cout << "ERI Hash: " << num_nonzero << " unique entries after cleanup ("
                  << (num_nonzero * bytes_per_entry) / (1024*1024) << " MB)" << std::endl;
    }
}













// full_range
// All of CGTO Idx pair {a,b} satisfying a < b
__global__ void generatePrimitiveShellPairIndices_for_SAD_K_computation(size_t2* d_primitive_shell_pair_indices_for_SAD_K_computation, const PrimitiveShell* d_primitive_shells, int num_primitive_shells, size_t num_threads){
    const size_t id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_threads) return;
    
    size_t2 res = index1to2(id, false, num_primitive_shells);

    d_primitive_shell_pair_indices_for_SAD_K_computation[id] = res;
}


__global__ void copySchwarzUpperBoundFactors_for_SAD_K_computation(real_t* d_schwarz_upper_bound_factors_for_SAD_K_computation, ShellPairSorter* d_shell_pair_sorter_for_SAD_K_computation, const size_t num_primitive_shells) {
    const size_t id = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_primitive_shells * num_primitive_shells) return;

    d_schwarz_upper_bound_factors_for_SAD_K_computation[id] = d_shell_pair_sorter_for_SAD_K_computation[id].schwarz_upper_bound_ab;
}



ERI_RI_Direct::ERI_RI_Direct(const HF& hf, const Molecular& auxiliary_molecular): 
    hf_(hf),
    num_basis_(hf.get_num_basis()),
    num_auxiliary_basis_(auxiliary_molecular.get_num_basis()),
    auxiliary_shell_type_infos_(auxiliary_molecular.get_shell_type_infos()),
    auxiliary_primitive_shells_(auxiliary_molecular.get_primitive_shells()),
    auxiliary_cgto_normalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
    schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
    auxiliary_schwarz_upper_bound_factors(auxiliary_molecular.get_primitive_shells().size()),
    two_center_eris(num_auxiliary_basis_ * num_auxiliary_basis_), 
    two_center_eris_inverse(num_auxiliary_basis_ * num_auxiliary_basis_), 
    primitive_shell_pair_indices(hf_.get_primitive_shells().size() * (hf_.get_primitive_shells().size() + 1) / 2),
    schwarz_upper_bound_factors_for_SAD_K_computation((hf_.get_initial_guess_algorithm_name() == "sad") ? hf_.get_primitive_shells().size() * hf_.get_primitive_shells().size() : 0),
    primitive_shell_pair_indices_for_SAD_K_computation((hf_.get_initial_guess_algorithm_name() == "sad") ? hf_.get_primitive_shells().size() * hf_.get_primitive_shells().size() : 0),
    // CPU-only B matrix cache: allocate full size on CPU, 1x1 placeholder on GPU.
    intermediate_matrix_B_cpu_(
        gpu::gpu_available() ? 1 : num_auxiliary_basis_,
        gpu::gpu_available() ? 1 : num_basis_ * num_basis_)
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
}

ERI_RI_Direct::~ERI_RI_Direct() {
    if (d_eri_reconstructed_cpu_) {
        tracked_cudaFree(d_eri_reconstructed_cpu_);
        d_eri_reconstructed_cpu_ = nullptr;
    }
}

const real_t* ERI_RI_Direct::get_eri_matrix_device() const {
    if (gpu::gpu_available()) {
        // On GPU the Direct/SemiDirect/Hash variants don't maintain a dense
        // AO ERI tensor; post-HF methods have their own GPU pipelines.
        return nullptr;
    }
    if (d_eri_reconstructed_cpu_) return d_eri_reconstructed_cpu_;

    // Lazily reconstruct: (μν|λσ) = Σ_Q B(Q,μν) · B(Q,λσ)
    // where B is intermediate_matrix_B_cpu_ with shape (naux, nao²).
    const size_t nao2 = (size_t)num_basis_ * num_basis_;
    const size_t required_bytes = nao2 * nao2 * sizeof(real_t);
    tracked_cudaMalloc(&d_eri_reconstructed_cpu_, required_bytes);

    gpu::matrixMatrixProductRect(
        intermediate_matrix_B_cpu_.device_ptr(),
        intermediate_matrix_B_cpu_.device_ptr(),
        d_eri_reconstructed_cpu_,
        (int)nao2, (int)nao2, num_auxiliary_basis_,
        /*transpose_A=*/true, /*transpose_B=*/false,
        /*accumulate=*/false, /*alpha=*/1.0);

    return d_eri_reconstructed_cpu_;
}


void ERI_RI_Direct::precomputation() {
    // compute the intermediate matrix B of the auxiliary basis functions
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();

    const int threads_per_block = 1024;

    // K 計算用のソートに使用
    const size_t num_primitive_shells = primitive_shells.size();


    // K計算用のペア配列生成
    // The SAD-K pre-sort below feeds the GPU computeInitialFockMatrix_RI_Direct_RHF
    // path which is entirely skipped on CPU (ERI_RI_Direct_RHF::compute_fock_matrix
    // short-circuits to the cached intermediate_matrix_B_cpu_).  So in CPU mode we
    // just compute the regular Schwarz bounds (stub = 1.0) and skip the SAD-K work.
    if(schwarz_upper_bound_factors_for_SAD_K_computation.size() > 0 && gpu::gpu_available()){
#ifndef GANSU_CPU_ONLY
        size_t num_tasks = num_primitive_shells*num_primitive_shells;
        size_t num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;
        generatePrimitiveShellPairIndices_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(primitive_shell_pair_indices_for_SAD_K_computation.device_ptr(), primitive_shells.device_ptr(), num_primitive_shells, num_tasks);

        // Sort用構造体配列
        ShellPairSorter* d_shell_pair_sorter_for_SAD_K_computation;
        cudaMalloc((void**)&d_shell_pair_sorter_for_SAD_K_computation, sizeof(ShellPairSorter)*num_tasks);


        // compute upper bounds of primitive-shell-pair
        // 通常のshell pairの上界計算も行う
        gpu::computeSchwarzUpperBounds_for_SAD_K_computation(
            shell_type_infos,
            shell_pair_type_infos,
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
            d_shell_pair_sorter_for_SAD_K_computation,
            num_primitive_shells,
            verbose
        );

        // K計算用のshell-pair配列ソート
        thrust::device_ptr<ShellPairSorter> keys_begin(d_shell_pair_sorter_for_SAD_K_computation);
        thrust::device_ptr<ShellPairSorter> keys_end(d_shell_pair_sorter_for_SAD_K_computation + num_tasks);
        thrust::device_ptr<size_t2> values_begin(primitive_shell_pair_indices_for_SAD_K_computation.device_ptr());
        thrust::sort_by_key(keys_begin, keys_end, values_begin);

        copySchwarzUpperBoundFactors_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(schwarz_upper_bound_factors_for_SAD_K_computation.device_ptr(), d_shell_pair_sorter_for_SAD_K_computation, num_primitive_shells);

        primitive_shell_pair_indices_for_SAD_K_computation.toHost();
        cudaFree(d_shell_pair_sorter_for_SAD_K_computation);
#endif // !GANSU_CPU_ONLY
    }else{
        gpu::computeSchwarzUpperBounds(
            shell_type_infos,
            shell_pair_type_infos,
            primitive_shells.device_ptr(),
            boys_grid.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            schwarz_upper_bound_factors.device_ptr(),   // schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
            verbose
        );
    }



    // shell-pair sort
    int pair_idx = 0;
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const size_t count = shell_pair_type_infos[pair_idx].count;
            const size_t start = shell_pair_type_infos[pair_idx].start_index;

#ifndef GANSU_CPU_ONLY
            if (gpu::gpu_available()) {
                const int num_blocks = (count + threads_per_block - 1) / threads_per_block;
                generatePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&primitive_shell_pair_indices.device_ptr()[start], count, s0 == s1, shell_type_infos[s1].count, true, shell_type_infos[s0].start_index, shell_type_infos[s1].start_index);
                thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[start]);
                thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[start] + count);
                thrust::device_ptr<size_t2> values_begin(&primitive_shell_pair_indices.device_ptr()[start]);
                thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
            } else
#endif
            {
                // === CPU path: generate indices (with start offsets) + sort on host ===
                size_t2* h_pair = &primitive_shell_pair_indices.device_ptr()[start];
                real_t* h_keys  = &schwarz_upper_bound_factors.device_ptr()[start];
                const bool is_symmetric = (s0 == s1);
                const size_t ns1 = shell_type_infos[s1].count;
                const size_t off_a = shell_type_infos[s0].start_index;
                const size_t off_b = shell_type_infos[s1].start_index;
                for (size_t id = 0; id < count; ++id) {
                    size_t r1, r2;
                    if (is_symmetric) {
                        r2 = (size_t)((std::sqrt(8.0 * (double)id + 1.0) - 1.0) / 2.0);
                        while ((r2 + 1) * (r2 + 2) / 2 <= id) ++r2;
                        while (r2 > 0 && r2 * (r2 + 1) / 2 > id) --r2;
                        r1 = id - r2 * (r2 + 1) / 2;
                    } else {
                        r1 = id / ns1;
                        r2 = id % ns1;
                    }
                    h_pair[id] = {r1 + off_a, r2 + off_b};
                }
                cpu_sort_by_key_descending(h_keys, h_pair, count);
            }

            pair_idx++;
        }
    }
    cudaDeviceSynchronize();







    
    
    // compute upper bounds of  aux-shell
    gpu::computeAuxiliarySchwarzUpperBounds(
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        boys_grid.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        auxiliary_schwarz_upper_bound_factors.device_ptr(),   // auxiliary_schwarz_upper_bound_factorsに√(pq|pq)の値がはいっている
        verbose
    );

    for(const auto& s : auxiliary_shell_type_infos_){
#ifndef GANSU_CPU_ONLY
        if (gpu::gpu_available()) {
            thrust::device_ptr<real_t> keys_begin(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index]);
            thrust::device_ptr<real_t> keys_end(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index] + s.count);
            thrust::device_ptr<PrimitiveShell> values_begin(&auxiliary_primitive_shells_.device_ptr()[s.start_index]);
            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
        } else
#endif
        {
            real_t*         h_keys = &auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index];
            PrimitiveShell* h_vals = &auxiliary_primitive_shells_.device_ptr()[s.start_index];
            cpu_sort_by_key_descending(h_keys, h_vals, s.count);
        }
    }



    gpu::computeTwoCenterERIs(
        auxiliary_shell_type_infos_, 
        auxiliary_primitive_shells_.device_ptr(), 
        auxiliary_cgto_normalization_factors_.device_ptr(), 
        two_center_eris.device_ptr(),
        num_auxiliary_basis_,
        boys_grid.device_ptr(),
        auxiliary_schwarz_upper_bound_factors.device_ptr(),
        schwarz_screening_threshold,
        verbose
    );


    gpu::choleskyDecomposition(two_center_eris.device_ptr(), num_auxiliary_basis_);
    gpu::computeInverseByDtrsm(two_center_eris.device_ptr(), two_center_eris_inverse.device_ptr(), num_auxiliary_basis_);

    // === CPU fallback: also precompute the full B matrix ===
    // Direct-RI on GPU avoids storing B for memory reasons, but on CPU we
    // reuse the stored RHF Fock path which needs B.  Build it once here
    // (3-center ERI + triangular solve against the Cholesky factor L).
    if (!gpu::gpu_available()) {
        const size_t naux2_times_nao2 =
            (size_t)num_auxiliary_basis_ * num_basis_ * num_basis_;
        // Allocate a scratch buffer for the 3-center ERIs.  On CPU,
        // tracked_cudaMalloc falls back to std::calloc, so this is host memory.
        real_t* d_three_center_eri = nullptr;
        tracked_cudaMalloc(&d_three_center_eri, naux2_times_nao2 * sizeof(real_t));
        cudaMemset(d_three_center_eri, 0, naux2_times_nao2 * sizeof(real_t));

        gpu::computeThreeCenterERIs(
            shell_type_infos,
            shell_pair_type_infos,
            primitive_shells.device_ptr(),
            cgto_normalization_factors.device_ptr(),
            auxiliary_shell_type_infos_,
            auxiliary_primitive_shells_.device_ptr(),
            auxiliary_cgto_normalization_factors_.device_ptr(),
            d_three_center_eri,
            primitive_shell_pair_indices.device_ptr(),
            num_basis_,
            num_auxiliary_basis_,
            boys_grid.device_ptr(),
            schwarz_upper_bound_factors.device_ptr(),
            auxiliary_schwarz_upper_bound_factors.device_ptr(),
            schwarz_screening_threshold,
            verbose);

        // Solve L * B = T (T is the 3-center ERI matrix viewed as naux x nao²).
        // two_center_eris currently holds L (Cholesky factor, lower triangular).
        gpu::solve_lower_triangular(
            two_center_eris.device_ptr(),
            d_three_center_eri,
            num_auxiliary_basis_,
            num_basis_ * num_basis_);

        // Copy the result into intermediate_matrix_B_cpu_.
        cudaMemcpy(intermediate_matrix_B_cpu_.device_ptr(), d_three_center_eri,
                   naux2_times_nao2 * sizeof(real_t), cudaMemcpyDeviceToDevice);

        tracked_cudaFree(d_three_center_eri);
    }
}


} // namespace gansu
