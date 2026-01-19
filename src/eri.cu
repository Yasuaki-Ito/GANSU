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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

__global__ void initializePrimitiveShellPairIndices(int2* d_indices_array, int num_threads, bool is_symmetric, int num_basis) {
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id >= num_threads) return;
    size_t2 index_pair = index1to2(id, is_symmetric, num_basis);
    d_indices_array[id] = make_int2(static_cast<int>(index_pair.x), static_cast<int>(index_pair.y));
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
        auxiliary_shell_type_infos_(auxiliary_molecular.get_shell_type_infos()),
        auxiliary_primitive_shells_(auxiliary_molecular.get_primitive_shells()),
        auxiliary_cgto_normalization_factors_(auxiliary_molecular.get_cgto_normalization_factors()),
        intermediate_matrix_B_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_J_(num_basis_, num_basis_),
        d_K_(num_basis_, num_basis_),
        d_W_tmp_(num_auxiliary_basis_),
        d_T_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        d_V_tmp_(num_auxiliary_basis_, num_basis_*num_basis_),
        schwarz_upper_bound_factors(hf.get_num_primitive_shell_pairs()),
        auxiliary_schwarz_upper_bound_factors(auxiliary_molecular.get_primitive_shells().size())
{
    // to device memory
    auxiliary_primitive_shells_.toDevice();
    auxiliary_cgto_normalization_factors_.toDevice();
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
    num_fock_replicas_(8)
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
            initializePrimitiveShellPairIndices<<<num_blocks, threads_per_block>>>(&d_primitive_shell_pair_indices[shell_pair_type_infos[pair_idx].start_index], shell_pair_type_infos[pair_idx].count, s0 == s1, shell_type_infos[s1].count);
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
    num_basis_(hf.get_num_basis())
{
    // ここでHash memoryの初期化をおこなう
}

void ERI_Hash::precomputation() {
    const std::vector<ShellTypeInfo>& shell_type_infos = hf_.get_shell_type_infos();
    const std::vector<ShellPairTypeInfo>& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    const DeviceHostMemory<PrimitiveShell>& primitive_shells = hf_.get_primitive_shells();
    const DeviceHostMemory<real_t>& cgto_normalization_factors = hf_.get_cgto_normalization_factors();
    const DeviceHostMemory<real_t>& boys_grid = hf_.get_boys_grid();
    const int verbose = hf_.get_verbose();

    gpu::constructERIHash(
        shell_type_infos,
        shell_pair_type_infos,
        primitive_shells.device_ptr(), 
        boys_grid.device_ptr(), 
        cgto_normalization_factors.device_ptr(), 
        // Hash memoryのポインタを渡す
        verbose
    );
}


} // namespace gansu