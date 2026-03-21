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

#include "davidson_solver.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"
#include "utils.hpp"

#include <cublas_v2.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>

namespace gansu {

// Symmetrize subspace matrix: H = (H + H^T) / 2
__global__ void symmetrize_matrix_kernel(real_t* d_matrix, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n || i >= j) return;
    real_t avg = 0.5 * (d_matrix[i + j * n] + d_matrix[j + i * n]);
    d_matrix[i + j * n] = avg;
    d_matrix[j + i * n] = avg;
}

// Davidson-Jacobi correction kernel: correction[j] = -residual[j] / (diagonal[j] - eigenvalue)
__global__ void davidson_jacobi_correction_kernel(
    const real_t* __restrict__ d_diagonal,
    const real_t* __restrict__ d_residual,
    real_t* __restrict__ d_correction,
    real_t eigenvalue, int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t denom = d_diagonal[idx] - eigenvalue;
    d_correction[idx] = (fabs(denom) > 1e-12) ? -d_residual[idx] / denom : 0.0;
}

// ========== Constructor ==========

DavidsonSolver::DavidsonSolver(const LinearOperator& linear_op,
                               const DavidsonConfig& config)
    : linear_op_(linear_op),
      config_(config),
      dim_(linear_op.dimension()),
      subspace_dim_(0),
      num_iterations_(0),
      d_subspace_vectors_(nullptr),
      d_sigma_vectors_(nullptr),
      d_subspace_matrix_(nullptr),
      d_subspace_eigenvalues_(nullptr),
      d_subspace_eigenvectors_(nullptr),
      d_residuals_(nullptr),
      d_eigenvectors_(nullptr)
{
    // Validate configuration
    if (config_.num_eigenvalues <= 0) {
        THROW_EXCEPTION("DavidsonSolver: num_eigenvalues must be positive");
    }
    if (config_.num_eigenvalues > dim_) {
        THROW_EXCEPTION("DavidsonSolver: num_eigenvalues cannot exceed operator dimension");
    }
    // Cap max_subspace_size to dimension (cannot have more orthogonal vectors than dim)
    if (config_.max_subspace_size > dim_) {
        config_.max_subspace_size = dim_;
    }
    if (config_.max_subspace_size < config_.num_eigenvalues) {
        THROW_EXCEPTION("DavidsonSolver: max_subspace_size must be >= num_eigenvalues");
    }
    if (config_.convergence_threshold <= 0.0) {
        THROW_EXCEPTION("DavidsonSolver: convergence_threshold must be positive");
    }

    // Set default initial subspace size if not specified
    if (config_.initial_subspace_size == 0) {
        config_.initial_subspace_size = std::min(2 * config_.num_eigenvalues,
                                                 config_.max_subspace_size);
    }

    // Cap initial_subspace_size to dimension
    if (config_.initial_subspace_size > dim_) {
        config_.initial_subspace_size = dim_;
    }

    if (config_.initial_subspace_size < config_.num_eigenvalues) {
        THROW_EXCEPTION("DavidsonSolver: initial_subspace_size must be >= num_eigenvalues");
    }

    // Allocate host memory
    h_eigenvalues_.resize(config_.num_eigenvalues);
    residual_norms_.resize(config_.num_eigenvalues);

    // Allocate GPU memory
    allocate_memory();

    if (config_.verbose > 1) {
        std::cout << "  Davidson: dim=" << dim_
                  << ", nev=" << config_.num_eigenvalues
                  << ", max_sub=" << config_.max_subspace_size
                  << ", tol=" << std::scientific << config_.convergence_threshold
                  << std::defaultfloat << std::endl;
    }
}

// ========== Destructor ==========

DavidsonSolver::~DavidsonSolver() {
    free_memory();
}

// ========== Memory Management ==========

void DavidsonSolver::allocate_memory() {
    using gansu::tracked_cudaMalloc;

    // Pre-check total memory requirement
    size_t total_bytes = (
        static_cast<size_t>(dim_) * config_.max_subspace_size * 2 +  // subspace + sigma
        static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * 2 +  // matrix + eigvecs
        static_cast<size_t>(dim_) * config_.num_eigenvalues * 2 +    // residuals + eigenvectors
        config_.max_subspace_size                                    // eigenvalues
    ) * sizeof(real_t);

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    if (total_bytes > free_mem) {
        std::ostringstream oss;
        oss << "Davidson solver requires "
            << CudaMemoryManager<real_t>::format_bytes(total_bytes)
            << " but only "
            << CudaMemoryManager<real_t>::format_bytes(free_mem)
            << " available on GPU. "
            << "Consider using schur_omega solver for large systems.";
        throw std::runtime_error(oss.str());
    }

    // Subspace vectors: dim × max_subspace_size (use size_t to avoid int overflow)
    tracked_cudaMalloc(&d_subspace_vectors_,
                       static_cast<size_t>(dim_) * config_.max_subspace_size * sizeof(real_t));

    // Sigma vectors: dim × max_subspace_size
    tracked_cudaMalloc(&d_sigma_vectors_,
                       static_cast<size_t>(dim_) * config_.max_subspace_size * sizeof(real_t));

    // Subspace matrix: max_subspace_size × max_subspace_size
    tracked_cudaMalloc(&d_subspace_matrix_,
                       static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * sizeof(real_t));

    // Subspace eigenvalues: max_subspace_size
    tracked_cudaMalloc(&d_subspace_eigenvalues_,
                       config_.max_subspace_size * sizeof(real_t));

    // Subspace eigenvectors: max_subspace_size × max_subspace_size
    tracked_cudaMalloc(&d_subspace_eigenvectors_,
                       static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * sizeof(real_t));

    // Residuals: dim × num_eigenvalues
    tracked_cudaMalloc(&d_residuals_,
                       static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));

    // Final eigenvectors: dim × num_eigenvalues
    tracked_cudaMalloc(&d_eigenvectors_,
                       static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));

    if (config_.verbose > 2) {
        size_t total_bytes = (
            static_cast<size_t>(dim_) * config_.max_subspace_size * 2 +  // subspace + sigma
            static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * 2 +  // matrix + eigvecs
            static_cast<size_t>(dim_) * config_.num_eigenvalues * 2 +    // residuals + eigenvectors
            config_.max_subspace_size                                    // eigenvalues
        ) * sizeof(real_t);

        std::cout << "  Davidson memory: "
                  << CudaMemoryManager<real_t>::format_bytes(total_bytes) << std::endl;
    }
}

void DavidsonSolver::free_memory() {
    using gansu::tracked_cudaFree;

    if (d_subspace_vectors_) tracked_cudaFree(d_subspace_vectors_);
    if (d_sigma_vectors_) tracked_cudaFree(d_sigma_vectors_);
    if (d_subspace_matrix_) tracked_cudaFree(d_subspace_matrix_);
    if (d_subspace_eigenvalues_) tracked_cudaFree(d_subspace_eigenvalues_);
    if (d_subspace_eigenvectors_) tracked_cudaFree(d_subspace_eigenvectors_);
    if (d_residuals_) tracked_cudaFree(d_residuals_);
    if (d_eigenvectors_) tracked_cudaFree(d_eigenvectors_);

    d_subspace_vectors_ = nullptr;
    d_sigma_vectors_ = nullptr;
    d_subspace_matrix_ = nullptr;
    d_subspace_eigenvalues_ = nullptr;
    d_subspace_eigenvectors_ = nullptr;
    d_residuals_ = nullptr;
    d_eigenvectors_ = nullptr;
}

// ========== Main Solve Method ==========

bool DavidsonSolver::solve(const real_t* d_initial_guess) {
    // Initialize subspace
    initialize_subspace(d_initial_guess);
    num_iterations_ = 0;
    int sigma_computed_ = 0;  // Track how many sigma vectors are up-to-date

    // Eigenvalue-stability convergence tracking for non-Hermitian problems.
    // The residual can oscillate near the threshold without converging,
    // but if eigenvalues are stable, the solution is effectively converged.
    std::vector<real_t> prev_eigenvalues(config_.num_eigenvalues, 0.0);
    int eigenvalue_stable_count = 0;
    const int eigenvalue_stable_required = 3;

    bool converged = false;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        num_iterations_ = iter + 1;

        if (config_.verbose > 2) {
            std::cout << "  Davidson sub_dim=" << subspace_dim_ << std::endl;
        }

        // Apply operator only to NEW subspace vectors (not already computed)
        for (int i = sigma_computed_; i < subspace_dim_; ++i) {
            linear_op_.apply(&d_subspace_vectors_[static_cast<size_t>(i) * dim_],
                           &d_sigma_vectors_[static_cast<size_t>(i) * dim_]);
        }
        sigma_computed_ = subspace_dim_;
        cudaDeviceSynchronize();

        // Build subspace matrix
        build_subspace_matrix();

        // Solve subspace eigenvalue problem
        solve_subspace_eigenproblem();

        // Compute Ritz vectors and residuals
        compute_ritz_vectors_and_residuals();

        // Check convergence (residual-based)
        converged = check_convergence();

        // Eigenvalue-stability check: if eigenvalues haven't changed by more than
        // convergence_threshold for several consecutive iterations AND the residual
        // is within 10× of the threshold, declare convergence.
        // This handles non-Hermitian problems where residuals oscillate.
        if (!converged && iter > 0) {
            real_t max_eval_change = 0.0;
            for (int i = 0; i < config_.num_eigenvalues; ++i) {
                // Skip spurious eigenvalues
                if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue)
                    continue;
                max_eval_change = std::max(max_eval_change,
                    std::abs(h_eigenvalues_[i] - prev_eigenvalues[i]));
            }

            real_t max_res = *std::max_element(residual_norms_.begin(), residual_norms_.end());

            if (max_eval_change < config_.convergence_threshold &&
                max_res < 10.0 * config_.convergence_threshold) {
                eigenvalue_stable_count++;
            } else {
                eigenvalue_stable_count = 0;
            }

            if (eigenvalue_stable_count >= eigenvalue_stable_required) {
                converged = true;
            }
        }
        prev_eigenvalues = h_eigenvalues_;

        if (config_.verbose > 1) {
            // Find max residual norm
            real_t max_res = *std::max_element(residual_norms_.begin(), residual_norms_.end());
            std::cout << "---- Davidson iteration " << std::setw(3) << (iter + 1) << " ---- ";
            for (int i = 0; i < std::min(5, config_.num_eigenvalues); ++i) {
                std::cout << std::fixed << std::setprecision(6) << h_eigenvalues_[i];
                if (i < std::min(5, config_.num_eigenvalues) - 1) std::cout << ", ";
            }
            std::cout << "  max|r|=" << std::scientific << std::setprecision(2) << max_res
                      << std::defaultfloat << std::endl;
        }

        if (converged) {
            if (config_.verbose > 0) {
                std::cout << "  Davidson converged in " << num_iterations_
                         << " iterations" << std::endl;
            }
            break;
        }

        // Restart if subspace is full (but not if it spans the entire space,
        // since the next iteration will give the exact solution).
        // Must restart BEFORE add_correction_vectors so that
        // subspace_dim_ still matches the eigenvector stride from solve_subspace_eigenproblem.
        if (subspace_dim_ >= config_.max_subspace_size && subspace_dim_ < dim_) {
            // Save pre-restart eigenvalues for collapse detection
            std::vector<real_t> pre_restart_evals(h_eigenvalues_.begin(),
                h_eigenvalues_.begin() + config_.num_eigenvalues);

            if (config_.verbose > 2) {
                std::cout << "  Davidson: restarting subspace" << std::endl;
            }
            restart_subspace();
            sigma_computed_ = 0;  // Ritz vectors replaced old basis — recompute sigma

            // Detect eigenvalue collapse after restart for non-Hermitian problems
            if (config_.min_eigenvalue > 0.0) {
                // Recompute sigma and subspace matrix to check new eigenvalues
                for (int ii = sigma_computed_; ii < subspace_dim_; ++ii) {
                    linear_op_.apply(&d_subspace_vectors_[static_cast<size_t>(ii) * dim_],
                                   &d_sigma_vectors_[static_cast<size_t>(ii) * dim_]);
                }
                sigma_computed_ = subspace_dim_;
                cudaDeviceSynchronize();
                build_subspace_matrix();
                solve_subspace_eigenproblem();

                // Check if eigenvalues collapsed (dropped by >50% or below threshold)
                bool collapsed = false;
                for (int ii = 0; ii < config_.num_eigenvalues; ++ii) {
                    if (h_eigenvalues_[ii] < config_.min_eigenvalue ||
                        h_eigenvalues_[ii] < 0.5 * pre_restart_evals[ii]) {
                        collapsed = true;
                        break;
                    }
                }

                if (collapsed) {
                    if (config_.verbose > 0) {
                        std::cout << "  Davidson: eigenvalue collapse detected after restart, reinitializing" << std::endl;
                    }
                    // Reinitialize with fresh unit vectors
                    initialize_subspace(nullptr);
                    sigma_computed_ = 0;
                }
            }
        }

        // Add correction vectors
        add_correction_vectors();
    }

    if (!converged && config_.verbose > 0) {
        std::cout << "  Warning: Davidson did not converge in "
                 << config_.max_iterations << " iterations" << std::endl;
    }

    return converged;
}

// ========== Initialization ==========

void DavidsonSolver::initialize_subspace(const real_t* d_initial_guess) {
    if (d_initial_guess != nullptr) {
        // Use provided guess vectors
        cudaMemcpy(d_subspace_vectors_, d_initial_guess,
                   static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);
        subspace_dim_ = config_.num_eigenvalues;
    } else {
        int initial_size = config_.initial_subspace_size;
        const real_t* d_diagonal = linear_op_.get_diagonal_device();

        if (d_diagonal != nullptr) {
            // Use unit vectors for the smallest diagonal elements
            // This targets the lowest eigenvalues directly
            std::vector<real_t> h_diagonal(dim_);
            cudaMemcpy(h_diagonal.data(), d_diagonal,
                       dim_ * sizeof(real_t), cudaMemcpyDeviceToHost);

            // Find indices of smallest diagonal elements
            // When min_eigenvalue is set, exclude diagonal elements below
            // that threshold to avoid targeting spurious near-zero eigenvalues
            // (e.g., doubles-dominated states in EOM methods)
            std::vector<int> indices;
            indices.reserve(dim_);
            for (int idx = 0; idx < dim_; ++idx) {
                if (config_.min_eigenvalue > 0.0 && h_diagonal[idx] < config_.min_eigenvalue) {
                    continue;
                }
                indices.push_back(idx);
            }
            // Fallback: if too few valid indices, include all
            if (static_cast<int>(indices.size()) < initial_size) {
                indices.resize(dim_);
                std::iota(indices.begin(), indices.end(), 0);
            }
            std::partial_sort(indices.begin(), indices.begin() + initial_size,
                              indices.end(),
                              [&h_diagonal](int a, int b) {
                                  return h_diagonal[a] < h_diagonal[b];
                              });

            // Create unit vectors at those indices
            cudaMemset(d_subspace_vectors_, 0,
                       static_cast<size_t>(dim_) * initial_size * sizeof(real_t));
            for (int i = 0; i < initial_size; ++i) {
                real_t one = 1.0;
                cudaMemcpy(&d_subspace_vectors_[static_cast<size_t>(i) * dim_ + indices[i]],
                           &one, sizeof(real_t), cudaMemcpyHostToDevice);
            }
        } else {
            // Fallback: random initial vectors
            size_t total_elements = static_cast<size_t>(dim_) * initial_size;
            std::vector<real_t> h_random(total_elements);
            std::mt19937_64 rng(1234ULL);
            std::normal_distribution<real_t> dist(0.0, 1.0);
            for (size_t i = 0; i < total_elements; ++i) {
                h_random[i] = dist(rng);
            }
            cudaMemcpy(d_subspace_vectors_, h_random.data(),
                       total_elements * sizeof(real_t), cudaMemcpyHostToDevice);
        }

        subspace_dim_ = initial_size;
    }

    // Orthogonalize initial vectors
    orthogonalize_vectors(0, subspace_dim_);

    if (config_.verbose > 2) {
        std::cout << "  Davidson: initialized with " << subspace_dim_
                 << " vectors" << std::endl;
    }
}

// ========== Orthogonalization ==========

void DavidsonSolver::orthogonalize_vectors(int start_index, int num_vectors) {
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    for (int i = start_index; i < start_index + num_vectors; ++i) {
        real_t* d_vec_i = &d_subspace_vectors_[static_cast<size_t>(i) * dim_];

        // Orthogonalize against all previous vectors
        for (int j = 0; j < i; ++j) {
            const real_t* d_vec_j = &d_subspace_vectors_[static_cast<size_t>(j) * dim_];

            // Compute projection: proj = <v_j | v_i>
            real_t proj;
            cublasDdot(handle, dim_, d_vec_j, 1, d_vec_i, 1, &proj);

            // Subtract projection: v_i -= proj * v_j
            real_t alpha = -proj;
            cublasDaxpy(handle, dim_, &alpha, d_vec_j, 1, d_vec_i, 1);
        }

        // Normalize v_i
        real_t norm;
        cublasDnrm2(handle, dim_, d_vec_i, 1, &norm);

        if (norm < 1e-12) {
            if (config_.verbose > 0) {
                std::cout << "Warning: Linear dependence detected in vector "
                         << i << ", replacing with random vector" << std::endl;
            }
            // Replace with random vector (host-side generation to avoid curand even-count issue)
            std::vector<real_t> h_random(dim_);
            std::mt19937_64 rng(1234ULL + i);
            std::normal_distribution<real_t> dist(0.0, 1.0);
            for (int k = 0; k < dim_; ++k) {
                h_random[k] = dist(rng);
            }
            cudaMemcpy(d_vec_i, h_random.data(), dim_ * sizeof(real_t), cudaMemcpyHostToDevice);

            // Re-orthogonalize this single vector
            orthogonalize_vectors(i, 1);
            continue;  // Continue to next vector (don't re-normalize)
        }

        real_t inv_norm = 1.0 / norm;
        cublasDscal(handle, dim_, &inv_norm, d_vec_i, 1);
    }
}

// ========== Subspace Matrix Construction ==========

void DavidsonSolver::build_subspace_matrix() {
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    // Compute H = V^T * Σ using DGEMM (single cuBLAS call)
    // V: dim_ × subspace_dim_ stored column-major with lda = dim_
    // Σ: dim_ × subspace_dim_ stored column-major with lda = dim_
    // H: subspace_dim_ × subspace_dim_ stored column-major with lda = subspace_dim_
    // H_ij = <v_i | σ_j> = V[:,i]^T * Σ[:,j]
    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    cublasStatus_t status = cublasDgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                subspace_dim_, subspace_dim_, dim_,
                &alpha,
                d_subspace_vectors_, dim_,
                d_sigma_vectors_, dim_,
                &beta,
                d_subspace_matrix_, subspace_dim_);

    if (status != CUBLAS_STATUS_SUCCESS) {
        // Check for pending CUDA errors (e.g., OOM from previous kernel/allocation)
        cudaError_t cuda_err = cudaGetLastError();
        std::ostringstream oss;
        oss << "DavidsonSolver::build_subspace_matrix: cublasDgemm failed"
            << " (cuBLAS status=" << static_cast<int>(status)
            << ", CUDA error=" << cudaGetErrorString(cuda_err)
            << ", m=" << subspace_dim_ << ", n=" << subspace_dim_ << ", k=" << dim_ << ")";
        THROW_EXCEPTION(oss.str());
    }

    // Symmetrize H = (H + H^T) / 2 to handle operators with slight numerical asymmetry
    // (e.g., ADC(2) in spatial orbital representation where M12 ≠ M21^T structurally)
    if (config_.symmetric && subspace_dim_ > 1) {
        dim3 threads(16, 16);
        dim3 blocks((subspace_dim_ + 15) / 16, (subspace_dim_ + 15) / 16);
        symmetrize_matrix_kernel<<<blocks, threads>>>(d_subspace_matrix_, subspace_dim_);
        cudaDeviceSynchronize();
    }
}

// ========== Subspace Eigenvalue Problem ==========

void DavidsonSolver::solve_subspace_eigenproblem() {
    int status;
    if (config_.symmetric) {
        status = gpu::eigenDecomposition(
            d_subspace_matrix_,
            d_subspace_eigenvalues_,
            d_subspace_eigenvectors_,
            subspace_dim_
        );
    } else {
        status = gpu::eigenDecompositionNonSymmetric(
            d_subspace_matrix_,
            d_subspace_eigenvalues_,
            d_subspace_eigenvectors_,
            subspace_dim_
        );
    }

    if (status != 0) {
        THROW_EXCEPTION("DavidsonSolver: Subspace eigenvalue decomposition failed");
    }

    // Copy eigenvalues to host
    cudaMemcpy(h_eigenvalues_.data(), d_subspace_eigenvalues_,
               config_.num_eigenvalues * sizeof(real_t),
               cudaMemcpyDeviceToHost);

    if (config_.verbose > 2) {
        std::cout << "Subspace eigenvalues:" << std::endl;
        for (int i = 0; i < std::min(5, subspace_dim_); ++i) {
            real_t eval;
            cudaMemcpy(&eval, &d_subspace_eigenvalues_[i],
                      sizeof(real_t), cudaMemcpyDeviceToHost);
            std::cout << "  λ[" << i << "] = " << std::scientific
                     << std::setprecision(10) << eval << std::endl;
        }
    }
}

// ========== Ritz Vectors and Residuals ==========

void DavidsonSolver::compute_ritz_vectors_and_residuals() {
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        // Compute Ritz vector: ψ_i = V * c_i (linear combination of basis vectors)
        // eigenDecomposition transposes eigenvectors to row-major, so eigenvector i
        // is in COLUMN i (stride = subspace_dim_, starting at index i)
        const real_t alpha = 1.0;
        const real_t beta = 0.0;

        // Use Dgemv: y = alpha * A * x + beta * y
        cublasDgemv(handle, CUBLAS_OP_N,
                   dim_, subspace_dim_,
                   &alpha,
                   d_subspace_vectors_, dim_,
                   &d_subspace_eigenvectors_[i], subspace_dim_,
                   &beta,
                   &d_eigenvectors_[static_cast<size_t>(i) * dim_], 1);

        // Compute Hψ_i = Σ * c_i (linear combination of sigma vectors)
        cublasDgemv(handle, CUBLAS_OP_N,
                   dim_, subspace_dim_,
                   &alpha,
                   d_sigma_vectors_, dim_,
                   &d_subspace_eigenvectors_[i], subspace_dim_,
                   &beta,
                   &d_residuals_[static_cast<size_t>(i) * dim_], 1);

        // Compute residual: r_i = Hψ_i - λ_i * ψ_i
        real_t minus_lambda = -h_eigenvalues_[i];
        cublasDaxpy(handle, dim_, &minus_lambda,
                   &d_eigenvectors_[static_cast<size_t>(i) * dim_], 1,
                   &d_residuals_[static_cast<size_t>(i) * dim_], 1);

        // Compute residual norm
        real_t res_norm;
        cublasDnrm2(handle, dim_, &d_residuals_[static_cast<size_t>(i) * dim_], 1, &res_norm);
        residual_norms_[i] = res_norm;
    }
}

// ========== Convergence Check ==========

bool DavidsonSolver::check_convergence() {
    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        // Skip spurious eigenvalues below threshold (e.g., ground state in EOM)
        if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue) {
            return false;  // Not yet converged to valid excited states
        }
        if (residual_norms_[i] > config_.convergence_threshold) {
            return false;
        }
    }
    return true;
}

// ========== Correction Vectors ==========

void DavidsonSolver::add_correction_vectors() {
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();
    const real_t* d_diagonal = linear_op_.get_diagonal_device();
    int num_new_vectors = 0;

    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        // Skip spurious eigenvalues below threshold (e.g., ground state in EOM)
        if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue) {
            continue;
        }
        if (residual_norms_[i] > config_.convergence_threshold) {
            // Check if we have space for more vectors
            if (subspace_dim_ + num_new_vectors >= config_.max_subspace_size) {
                break;
            }

            real_t* d_correction = &d_subspace_vectors_[static_cast<size_t>(subspace_dim_ + num_new_vectors) * dim_];

            if (config_.use_preconditioner && d_diagonal != nullptr) {
                // Davidson-Jacobi correction: δ[j] = -r[j] / (H_jj - θ_i)
                int threads = 256;
                int blocks = (dim_ + threads - 1) / threads;
                davidson_jacobi_correction_kernel<<<blocks, threads>>>(
                    d_diagonal, &d_residuals_[static_cast<size_t>(i) * dim_], d_correction,
                    h_eigenvalues_[i], dim_);
                cudaDeviceSynchronize();
            } else if (config_.use_preconditioner) {
                // Fallback: operator-provided preconditioner (no eigenvalue shift)
                linear_op_.apply_preconditioner(&d_residuals_[static_cast<size_t>(i) * dim_], d_correction);
            } else {
                // No preconditioning: just copy residual
                cudaMemcpy(d_correction, &d_residuals_[static_cast<size_t>(i) * dim_],
                          dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
            }

            // Normalize correction vector
            real_t norm;
            cublasDnrm2(handle, dim_, d_correction, 1, &norm);

            if (norm > 1e-12) {
                real_t inv_norm = 1.0 / norm;
                cublasDscal(handle, dim_, &inv_norm, d_correction, 1);
                num_new_vectors++;
            }
        }
    }

    if (num_new_vectors > 0) {
        // Orthogonalize new vectors against existing subspace
        orthogonalize_vectors(subspace_dim_, num_new_vectors);
        subspace_dim_ += num_new_vectors;

        if (config_.verbose > 2) {
            std::cout << "Added " << num_new_vectors
                     << " correction vectors (subspace dim: "
                     << subspace_dim_ << ")" << std::endl;
        }
    }
}

// ========== Subspace Restart ==========

void DavidsonSolver::restart_subspace() {
    // Keep 2*num_eigenvalues Ritz vectors (or all if subspace is small)
    // This preserves more spectral information across restarts
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();
    int num_keep = std::min(2 * config_.num_eigenvalues, subspace_dim_);

    // Allocate temporary buffer
    real_t* d_temp;
    tracked_cudaMalloc(&d_temp, static_cast<size_t>(dim_) * num_keep * sizeof(real_t));

    // Copy already-computed Ritz vectors for the first num_eigenvalues
    int n_from_eigvecs = std::min(config_.num_eigenvalues, num_keep);
    cudaMemcpy(d_temp, d_eigenvectors_,
               static_cast<size_t>(dim_) * n_from_eigvecs * sizeof(real_t),
               cudaMemcpyDeviceToDevice);

    // Compute additional Ritz vectors (indices num_eigenvalues..num_keep-1)
    // using Dgemv with the subspace eigenvector coefficients
    // eigenDecomposition layout: eigenvector i starts at index i, stride = subspace_dim_
    const real_t alpha = 1.0;
    const real_t beta = 0.0;
    for (int i = n_from_eigvecs; i < num_keep; ++i) {
        cublasDgemv(handle, CUBLAS_OP_N,
                    dim_, subspace_dim_,
                    &alpha,
                    d_subspace_vectors_, dim_,
                    &d_subspace_eigenvectors_[i], subspace_dim_,
                    &beta,
                    &d_temp[static_cast<size_t>(i) * dim_], 1);
    }

    // Copy back to subspace
    cudaMemcpy(d_subspace_vectors_, d_temp,
               static_cast<size_t>(dim_) * num_keep * sizeof(real_t),
               cudaMemcpyDeviceToDevice);
    tracked_cudaFree(d_temp);

    subspace_dim_ = num_keep;
}

// ========== Public Getters ==========

void DavidsonSolver::copy_eigenvectors_to_host(real_t* h_output) const {
    if (!h_output) {
        THROW_EXCEPTION("DavidsonSolver::copy_eigenvectors_to_host: null pointer");
    }

    cudaMemcpy(h_output, d_eigenvectors_,
               static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t),
               cudaMemcpyDeviceToHost);
}

} // namespace gansu
