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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

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

    // ERI[nao^2 x nao^2] = B^T * B  where B is (naux x nao^2) row-major
    // cuBLAS column-major: B_cm is (nao^2 x naux), so ERI_cm = B_cm * B_cm^T
    const double alpha = 1.0, beta = 0.0;
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    cublasDgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        nao2, nao2, num_auxiliary_basis_,
        &alpha,
        intermediate_matrix_B_.device_ptr(), nao2,
        intermediate_matrix_B_.device_ptr(), nao2,
        &beta,
        d_eri_reconstructed_, nao2);
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
    const size_t n4 = (size_t)nmo * nmo * nmo * nmo;
    real_t* d_eri_work = nullptr;
    real_t* d_eri_mo = nullptr;
    tracked_cudaMalloc(&d_eri_work, n4 * sizeof(real_t));
    tracked_cudaMalloc(&d_eri_mo, n4 * sizeof(real_t));
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
    cudaMalloc((void**)&d_primitive_shell_pair_indices, sizeof(size_t2) * num_primitive_shell_pairs);

    int pair_idx = 0;
    const int threads_per_block = 1024;
    for(int s0 = 0; s0 < shell_type_infos.size(); s0++){
        for(int s1 = s0; s1 < shell_type_infos.size(); s1++){
            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            generatePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count);

            thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);  
            thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index] + shell_pair_type_infos[pair_idx].count);
            thrust::device_ptr<size_t2> values_begin(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index]);

            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());

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
        thrust::device_ptr<real_t> keys_begin(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index]);  
        thrust::device_ptr<real_t> keys_end(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index] + s.count);
        thrust::device_ptr<PrimitiveShell> values_begin(&auxiliary_primitive_shells_.device_ptr()[s.start_index]);

        thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
    }


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


    cudaFree(d_primitive_shell_pair_indices);
    /*
    if(1){
        // copy the intermediate matrix B to the host memory
        intermediate_matrix_B_.toHost();

        std::cout << "Intermediate matrix B:" << std::endl;
        for(int i=0; i<num_auxiliary_basis_; i++){
            for(int j=0; j<num_basis_; j++){
                for(int k=0; k<num_basis_; k++){
                    auto value = intermediate_matrix_B_(i, j*num_basis_+k);
                    if (std::isnan(value)) {
                        std::cout << "NaN found at (" << i << "," << j << "): " << value << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    */
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
            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            //initializePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count);
            initializePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count, shell_type_infos[s0].start_index, shell_type_infos[s1].start_index);
            thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);  
            thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index] + shell_pair_type_infos[pair_idx].count);
            thrust::device_ptr<int2> values_begin(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index]);
            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
            pair_idx++;
        }
    }
    cudaDeviceSynchronize();
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
    if (d_coo_keys_) tracked_cudaFree(d_coo_keys_);
    if (d_coo_values_) tracked_cudaFree(d_coo_values_);
    if (d_hash_keys_) tracked_cudaFree(d_hash_keys_);
    if (d_hash_values_) tracked_cudaFree(d_hash_values_);
    if (d_nonzero_indices_) tracked_cudaFree(d_nonzero_indices_);
}

void ERI_Hash::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const real_t schwarz_screening_threshold = hf_.get_schwarz_screening_threshold();
    const int verbose = hf_.get_verbose();

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
    thrust::device_ptr<unsigned long long> keys_ptr(d_hash_keys);
    size_t num_nonzero = thrust::count_if(keys_ptr, keys_ptr + hash_capacity, HashKeyIsNonEmpty());

    // Keep hash table
    d_hash_keys_ = d_hash_keys;
    d_hash_values_ = d_hash_values;
    hash_capacity_mask_ = hash_capacity_mask;

    if (hash_fock_method_ == HashFockMethod::Compact) {
        tracked_cudaMalloc(&d_coo_keys_, num_nonzero * sizeof(unsigned long long));
        tracked_cudaMalloc(&d_coo_values_, num_nonzero * sizeof(real_t));
        {
            thrust::device_ptr<real_t> vals_ptr(d_hash_values);
            auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(keys_ptr, vals_ptr));
            auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(keys_ptr + hash_capacity, vals_ptr + hash_capacity));
            auto out_zip = thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_ptr<unsigned long long>(d_coo_keys_),
                thrust::device_ptr<real_t>(d_coo_values_)));
            thrust::copy_if(zip_begin, zip_end, keys_ptr, out_zip, HashKeyIsNonEmpty());
        }
        num_entries_ = num_nonzero;
    } else if (hash_fock_method_ == HashFockMethod::Indexed) {
        tracked_cudaMalloc(&d_nonzero_indices_, num_nonzero * sizeof(size_t));
        thrust::counting_iterator<size_t> count_begin(0);
        thrust::device_ptr<size_t> out_indices(d_nonzero_indices_);
        thrust::copy_if(count_begin, count_begin + hash_capacity, keys_ptr, out_indices, HashKeyIsNonEmpty());
        num_nonzero_ = num_nonzero;
    }
    // Fullscan: no additional data structure needed

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
    primitive_shell_pair_indices_for_SAD_K_computation((hf_.get_initial_guess_algorithm_name() == "sad") ? hf_.get_primitive_shells().size() * hf_.get_primitive_shells().size() : 0)
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
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
    if(schwarz_upper_bound_factors_for_SAD_K_computation.size() > 0){
        size_t num_tasks = num_primitive_shells*num_primitive_shells;
        size_t num_blocks = (num_tasks + threads_per_block - 1) / threads_per_block;
        generatePrimitiveShellPairIndices_for_SAD_K_computation<<<num_blocks, threads_per_block>>>(primitive_shell_pair_indices_for_SAD_K_computation.device_ptr(), primitive_shells.device_ptr(), num_primitive_shells, num_tasks);
    
        // Sort用構造体配列
        ShellPairSorter* d_shell_pair_sorter_for_SAD_K_computation;
        if(schwarz_upper_bound_factors_for_SAD_K_computation.size() > 0) cudaMalloc((void**)&d_shell_pair_sorter_for_SAD_K_computation, sizeof(ShellPairSorter)*num_tasks);


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
            const int num_blocks = (shell_pair_type_infos[pair_idx].count + threads_per_block - 1) / threads_per_block; // the number of blocks
            generatePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&primitive_shell_pair_indices.device_ptr()[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count, true, shell_type_infos[s0].start_index, shell_type_infos[s1].start_index);

            thrust::device_ptr<real_t> keys_begin(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);  
            thrust::device_ptr<real_t> keys_end(&schwarz_upper_bound_factors.device_ptr()[shell_pair_type_infos[pair_idx].start_index] + shell_pair_type_infos[pair_idx].count);
            thrust::device_ptr<size_t2> values_begin(&primitive_shell_pair_indices.device_ptr()[shell_pair_type_infos[pair_idx].start_index]);

            thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());

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
        thrust::device_ptr<real_t> keys_begin(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index]);
        thrust::device_ptr<real_t> keys_end(&auxiliary_schwarz_upper_bound_factors.device_ptr()[s.start_index] + s.count);
        thrust::device_ptr<PrimitiveShell> values_begin(&auxiliary_primitive_shells_.device_ptr()[s.start_index]);

        thrust::sort_by_key(keys_begin, keys_end, values_begin, thrust::greater<real_t>());
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
}


} // namespace gansu
