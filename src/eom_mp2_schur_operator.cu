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

/**
 * @file eom_mp2_schur_operator.cu
 * @brief GPU implementation of Schur complement operator for EOM methods
 *
 * apply(R1):
 *   1. σ([R1|0]) → σ1_from_R1 = M11×R1, σ2 = M21×R1
 *   2. Scale σ2 by 1/(ω-D2): temp_R2 = (ωI-M22)⁻¹ × (M21×R1)
 *   3. σ([0|temp_R2]) → σ1_from_R2 = M12×temp_R2
 *   4. result = σ1_from_R1 + σ1_from_R2
 *
 * This gives M_eff(ω) × R1 = M11×R1 + M12×(ωI-M22)⁻¹×M21×R1
 * For ω=0: reduces to M11×R1 - M12×D2⁻¹×M21×R1 (standard Schur)
 */

#include <cmath>

#include "eom_mp2_schur_operator.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"

namespace gansu {

// ========================================================================
//  CUDA kernels
// ========================================================================

/**
 * @brief Scale doubles vector for Schur complement: out[idx] = in[idx] / (ω - D2[idx])
 *
 * For ω = 0: reduces to -in[idx] / D2[idx] (standard Schur complement)
 * For ω ≠ 0: frequency-dependent Schur complement M_eff(ω) = M11 + M12·(ωI-M22)⁻¹·M21
 */
__global__ void eom_mp2_schur_scale_kernel(
    const real_t* __restrict__ d_input,
    const real_t* __restrict__ d_D2,
    real_t* __restrict__ d_output,
    real_t omega,
    int doubles_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= doubles_dim) return;
    real_t denom = omega - d_D2[idx];
    d_output[idx] = (fabs(denom) > 1e-12) ? d_input[idx] / denom : 0.0;
}

/**
 * @brief Add two vectors: out[idx] += in[idx]
 */
__global__ void eom_mp2_schur_add_kernel(
    const real_t* __restrict__ d_in,
    real_t* __restrict__ d_out,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    d_out[idx] += d_in[idx];
}

/**
 * @brief Preconditioner: out[idx] = in[idx] / D1[idx]
 */
__global__ void eom_mp2_schur_preconditioner_kernel(
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t diag = d_diagonal[idx];
    d_output[idx] = (fabs(diag) > 1e-12) ? d_input[idx] / diag : 0.0;
}

// ========================================================================
//  EOMMP2SchurOperator implementation
// ========================================================================

void EOMMP2SchurOperator::init(
    LinearOperator& op, const real_t* d_D1, const real_t* d_D2,
    int singles_dim, int doubles_dim, int total_dim)
{
    // Allocate workspace
    tracked_cudaMalloc(&d_full_input1_,  (size_t)total_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_full_output1_, (size_t)total_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_full_input2_,  (size_t)total_dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_full_output2_, (size_t)total_dim_ * sizeof(real_t));

    // Diagonal = D1 (orbital energy differences, used as preconditioner)
    tracked_cudaMalloc(&d_diagonal_, (size_t)singles_dim_ * sizeof(real_t));
    cudaMemcpy(d_diagonal_, d_D1,
               (size_t)singles_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
}

EOMMP2SchurOperator::EOMMP2SchurOperator(EOMMP2Operator& full_op)
    : full_op_(full_op),
      d_D2_ptr_(full_op.get_D2()),
      singles_dim_(full_op.get_singles_dim()),
      doubles_dim_(full_op.get_doubles_dim()),
      total_dim_(full_op.dimension()),
      d_full_input1_(nullptr), d_full_output1_(nullptr),
      d_full_input2_(nullptr), d_full_output2_(nullptr),
      d_diagonal_(nullptr)
{
    init(full_op, full_op.get_D1(), full_op.get_D2(),
         singles_dim_, doubles_dim_, total_dim_);
}

EOMMP2SchurOperator::EOMMP2SchurOperator(EOMCC2Operator& full_op)
    : full_op_(full_op),
      d_D2_ptr_(full_op.get_D2()),
      singles_dim_(full_op.get_singles_dim()),
      doubles_dim_(full_op.get_doubles_dim()),
      total_dim_(full_op.dimension()),
      d_full_input1_(nullptr), d_full_output1_(nullptr),
      d_full_input2_(nullptr), d_full_output2_(nullptr),
      d_diagonal_(nullptr)
{
    init(full_op, full_op.get_D1(), full_op.get_D2(),
         singles_dim_, doubles_dim_, total_dim_);
}

EOMMP2SchurOperator::~EOMMP2SchurOperator() {
    if (d_full_input1_)  tracked_cudaFree(d_full_input1_);
    if (d_full_output1_) tracked_cudaFree(d_full_output1_);
    if (d_full_input2_)  tracked_cudaFree(d_full_input2_);
    if (d_full_output2_) tracked_cudaFree(d_full_output2_);
    if (d_diagonal_)     tracked_cudaFree(d_diagonal_);
}

void EOMMP2SchurOperator::apply(const real_t* d_input, real_t* d_output) const {
    int threads = 256;

    // Step 1: Prepare [R1 | 0] and compute σ([R1|0])
    cudaMemcpy(d_full_input1_, d_input,
               (size_t)singles_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
    cudaMemset(d_full_input1_ + singles_dim_, 0,
               (size_t)doubles_dim_ * sizeof(real_t));
    full_op_.apply(d_full_input1_, d_full_output1_);
    // d_full_output1_[0..singles_dim-1] = M11 × R1
    // d_full_output1_[singles_dim..] = M21 × R1

    // Step 2: Scale doubles part by 1/(ω - D2)
    //   temp_R2 = (ωI - M22)⁻¹ × (M21 × R1)
    {
        int blocks = (doubles_dim_ + threads - 1) / threads;
        eom_mp2_schur_scale_kernel<<<blocks, threads>>>(
            d_full_output1_ + singles_dim_,
            d_D2_ptr_,
            d_full_input2_ + singles_dim_,
            omega_,
            doubles_dim_);
    }

    // Step 3: Prepare [0 | temp_R2] and compute σ([0|temp_R2])
    cudaMemset(d_full_input2_, 0,
               (size_t)singles_dim_ * sizeof(real_t));
    full_op_.apply(d_full_input2_, d_full_output2_);
    // d_full_output2_[0..singles_dim-1] = M12 × temp_R2

    // Step 4: result = M11×R1 + M12×temp_R2
    cudaMemcpy(d_output, d_full_output1_,
               (size_t)singles_dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
    {
        int blocks = (singles_dim_ + threads - 1) / threads;
        eom_mp2_schur_add_kernel<<<blocks, threads>>>(
            d_full_output2_, d_output, singles_dim_);
    }

    cudaDeviceSynchronize();
}

void EOMMP2SchurOperator::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256;
    int blocks = (singles_dim_ + threads - 1) / threads;
    eom_mp2_schur_preconditioner_kernel<<<blocks, threads>>>(
        d_diagonal_, d_input, d_output, singles_dim_);
}

} // namespace gansu
