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
#include "progress.hpp"

#include <Eigen/Dense>
#include <chrono>

#ifndef GANSU_CPU_ONLY
#include <cublas_v2.h>
#endif
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cstdlib>

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
      d_subspace_eigenvalues_imag_(nullptr),
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
    h_eigenvalues_imag_.assign(config_.num_eigenvalues, 0.0);
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
    // 2026-09-04: when the subspace does not fit the device, put the four
    // dim-sized buffers (subspace, sigma, residuals, eigenvectors) on MANAGED
    // memory instead of throwing. The per-iteration DGEMMs (V^T sigma,
    // projections) then stream the oversubscribed pages over the interconnect
    // — seconds per iteration, against a solve that could not run at all
    // (MgO n_emb=374 IP-EOM: dim 7.7M x 89 roots = 71 GB vs 46 GB free after
    // the DLPNO ground). GANSU_DAVIDSON_UVM=1 forces it, =0 restores the throw.
    bool use_managed = false;
    // (2026-09-16) GANSU_DAVIDSON_UVM=2: HOST-preferred managed buffers (host
    // first-touch, preferred location CPU, direct access from this device).
    // The =1 mode is device-preferred and thrashes once oversubscribed (MgO
    // pob IP at 1.5x stalled); host-preferred pages are simply streamed by the
    // subspace DGEMMs and the sigma reads — seconds per iteration — so the
    // subspace cap no longer competes with the operator for device memory
    // (MgO 3x3x3 EA: cap 5-7 on device stalled at max|r| ~0.1; cap 10 = 45 GB).
    bool host_managed = false;
    {
        const char* e = std::getenv("GANSU_DAVIDSON_UVM");
        const bool force = e && (e[0] == '1' || e[0] == '2');
        host_managed = e && e[0] == '2';
        const bool allow = !(e && e[0] == '0');
        if (force || (allow && total_bytes > free_mem)) use_managed = true;
        if (total_bytes > free_mem && !use_managed) {
            std::ostringstream oss;
            oss << "Davidson solver requires "
                << CudaMemoryManager<real_t>::format_bytes(total_bytes)
                << " but only "
                << CudaMemoryManager<real_t>::format_bytes(free_mem)
                << " available on GPU. "
                << "Consider using schur_omega solver for large systems.";
            throw std::runtime_error(oss.str());
        }
        if (use_managed) {
            std::cout << "  Davidson: subspace " << CudaMemoryManager<real_t>::format_bytes(total_bytes)
                      << " vs free " << CudaMemoryManager<real_t>::format_bytes(free_mem)
                      << " -> dim-sized buffers on MANAGED memory (oversubscribable)"
                      << (force ? "  (forced)" : "") << std::endl;
            if (total_bytes > free_mem)
                gansu::dump_tracked_allocations("Davidson managed fallback: device residents");
        }
    }
    auto alloc_dim = [&](real_t** p, size_t bytes) {
        if (use_managed && host_managed) {
            gansu::tracked_cudaMallocManagedHost(p, bytes);
            int dev = 0; cudaGetDevice(&dev);
            cudaMemLocation lh{}; lh.type = cudaMemLocationTypeHost;   lh.id = 0;
            cudaMemLocation ld{}; ld.type = cudaMemLocationTypeDevice; ld.id = dev;
            cudaMemAdvise(*p, bytes, cudaMemAdviseSetPreferredLocation, lh);
            cudaMemAdvise(*p, bytes, cudaMemAdviseSetAccessedBy, ld);
        }
        else if (use_managed) gansu::tracked_cudaMallocManaged(p, bytes);
        else                  tracked_cudaMalloc(p, bytes);
    };
    if (use_managed && host_managed)
        std::cout << "  Davidson: dim-sized buffers on HOST-preferred managed memory (GANSU_DAVIDSON_UVM=2): "
                  << CudaMemoryManager<real_t>::format_bytes(
                         static_cast<size_t>(dim_) * (2 * config_.max_subspace_size + 2 * config_.num_eigenvalues) * sizeof(real_t))
                  << " off device" << std::endl;

    // Subspace vectors: dim × max_subspace_size (use size_t to avoid int overflow)
    alloc_dim(&d_subspace_vectors_,
              static_cast<size_t>(dim_) * config_.max_subspace_size * sizeof(real_t));

    // Sigma vectors: dim × max_subspace_size
    alloc_dim(&d_sigma_vectors_,
              static_cast<size_t>(dim_) * config_.max_subspace_size * sizeof(real_t));

    // Subspace matrix: max_subspace_size × max_subspace_size
    tracked_cudaMalloc(&d_subspace_matrix_,
                       static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * sizeof(real_t));

    // Subspace eigenvalues: max_subspace_size (+ imag parts, non-Hermitian FULL mode)
    tracked_cudaMalloc(&d_subspace_eigenvalues_,
                       config_.max_subspace_size * sizeof(real_t));
    tracked_cudaMalloc(&d_subspace_eigenvalues_imag_,
                       config_.max_subspace_size * sizeof(real_t));

    // Subspace eigenvectors: max_subspace_size × max_subspace_size
    tracked_cudaMalloc(&d_subspace_eigenvectors_,
                       static_cast<size_t>(config_.max_subspace_size) * config_.max_subspace_size * sizeof(real_t));

    // Residuals / final eigenvectors: dim × num_eigenvalues. In the streamable
    // mode these stay on the DEVICE (2 × nev × dim, e.g. 5.8 GB at MgO 3x3x3 EA):
    // they are the working set of every iteration, and the blocked Ritz/residual
    // GEMMs write straight into them.
    host_managed_ = use_managed && host_managed;
    if (host_managed_) {
        tracked_cudaMalloc(&d_residuals_,    static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));
        tracked_cudaMalloc(&d_eigenvectors_, static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));
        tracked_cudaMalloc(&d_block_scratch_, static_cast<size_t>(dim_) * 2 * config_.num_eigenvalues * sizeof(real_t));
    } else {
        alloc_dim(&d_residuals_,
                  static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));
        alloc_dim(&d_eigenvectors_,
                  static_cast<size_t>(dim_) * config_.num_eigenvalues * sizeof(real_t));
    }

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
    if (d_subspace_eigenvalues_imag_) tracked_cudaFree(d_subspace_eigenvalues_imag_);
    if (d_subspace_eigenvectors_) tracked_cudaFree(d_subspace_eigenvectors_);
    if (d_residuals_) tracked_cudaFree(d_residuals_);
    if (d_eigenvectors_) tracked_cudaFree(d_eigenvectors_);
    if (d_block_scratch_) tracked_cudaFree(d_block_scratch_);
    d_block_scratch_ = nullptr;

    d_subspace_vectors_ = nullptr;
    d_sigma_vectors_ = nullptr;
    d_subspace_matrix_ = nullptr;
    d_subspace_eigenvalues_ = nullptr;
    d_subspace_eigenvalues_imag_ = nullptr;
    d_subspace_eigenvectors_ = nullptr;
    d_residuals_ = nullptr;
    d_eigenvectors_ = nullptr;
}

// (2026-09-16) See the header note: stage the sigma block through the device
// scratch in the streamable mode; plain pass-through otherwise (byte-identical).
void DavidsonSolver::apply_sigma_block(int first, int n) {
    if (n <= 0) return;
    if (!host_managed_ || !d_block_scratch_) {
        linear_op_.apply_batch(&d_subspace_vectors_[static_cast<size_t>(first) * dim_],
                               &d_sigma_vectors_[static_cast<size_t>(first) * dim_], n);
        return;
    }
    const int nev = config_.num_eigenvalues;
    real_t* d_in  = d_block_scratch_;
    real_t* d_out = d_block_scratch_ + static_cast<size_t>(nev) * dim_;
    for (int off = 0; off < n; off += nev) {
        const int nb = std::min(nev, n - off);
        const size_t bytes = static_cast<size_t>(nb) * dim_ * sizeof(real_t);
        cudaMemcpy(d_in, &d_subspace_vectors_[static_cast<size_t>(first + off) * dim_], bytes, cudaMemcpyDefault);
        linear_op_.apply_batch(d_in, d_out, nb);
        cudaDeviceSynchronize();
        cudaMemcpy(&d_sigma_vectors_[static_cast<size_t>(first + off) * dim_], d_out, bytes, cudaMemcpyDefault);
    }
    cudaDeviceSynchronize();
}

// ========== Main Solve Method ==========

bool DavidsonSolver::solve(const real_t* d_initial_guess) {
    // Initialize subspace
    initialize_subspace(d_initial_guess);
    num_iterations_ = 0;
    int sigma_computed_ = 0;  // Track how many sigma vectors are up-to-date
    int subspace_matrix_built_ = 0;  // Track how many columns of H are up-to-date

    // Eigenvalue-stability convergence tracking for non-Hermitian problems.
    // The residual can oscillate near the threshold without converging,
    // but if eigenvalues are stable, the solution is effectively converged.
    std::vector<real_t> prev_eigenvalues(config_.num_eigenvalues, 0.0);
    int eigenvalue_stable_count = 0;
    const int eigenvalue_stable_required = 3;
    // Softer fallback: if eigenvalues are stable to tol for many consecutive
    // iterations even though the residual has not converged, accept the result.
    // Common for non-Hermitian / dense-spectrum problems (EOM-CCSD large basis).
    int eigenvalue_only_stable_count = 0;
    const int eigenvalue_only_stable_required = 10;

    bool converged = false;

    // Per-phase timing (printed at solve end when verbose>0). One sync per phase/iter;
    // negligible vs the phases themselves. Reveals where a large-dim EOM solve spends
    // time outside the operator apply (orthogonalization / subspace ops / restart).
    double t_sigma = 0, t_build = 0, t_eig = 0, t_ritz = 0, t_corr = 0, t_restart = 0;
    auto phase = [](double& acc, auto&& fn) {
        cudaDeviceSynchronize();
        const auto p0 = std::chrono::high_resolution_clock::now();
        fn();
        cudaDeviceSynchronize();
        acc += std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - p0).count();
    };

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        num_iterations_ = iter + 1;

        if (config_.verbose > 2) {
            std::cout << "  Davidson sub_dim=" << subspace_dim_ << std::endl;
        }

        // Apply operator only to NEW subspace vectors (not already computed)
        {
            int n_new = subspace_dim_ - sigma_computed_;
            auto t_start = std::chrono::high_resolution_clock::now();
            // Batched matvec: apply the operator to all n_new NEW vectors at once.
            // Default apply_batch loops apply() (bit-identical); operators with a
            // streamed out-of-core intermediate (host-staged EA σ) share each slab
            // across the batch, amortizing H2D by n_new.
            if (n_new > 0) {
                apply_sigma_block(sigma_computed_, n_new);
            }
            sigma_computed_ = subspace_dim_;
            cudaDeviceSynchronize();
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t_start).count();
            t_sigma += ms / 1000.0;
            if (config_.verbose > 1 && n_new > 0) {
                std::cout << "  sigma: " << n_new << " vecs, "
                          << std::fixed << std::setprecision(0) << ms << "ms ("
                          << std::setprecision(0) << ms / n_new << "ms/vec)" << std::endl;
            }
        }

        // Build subspace matrix (incremental: only new columns)
        phase(t_build, [&] { build_subspace_matrix_incremental(subspace_matrix_built_); });
        subspace_matrix_built_ = subspace_dim_;

        // Solve subspace eigenvalue problem
        phase(t_eig, [&] { solve_subspace_eigenproblem(); });

        // Compute Ritz vectors and residuals
        phase(t_ritz, [&] { compute_ritz_vectors_and_residuals(); });

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

            // (2026-07-15 hardening) A complex Ritz pair among the reported
            // roots is NOT an eigenvalue approximation — its real part can sit
            // perfectly stable while the residual stalls (490 STEOM: artifacts
            // at max|r|=4e-2 were accepted by the stability fallbacks and the
            // dense geev later showed they were nowhere in the true spectrum).
            // Stability-based acceptance therefore requires an all-real window;
            // a persistent complex pair runs to max_iterations and reports an
            // honest non-convergence (+ per-root complex warnings at exit).
            // Purely-real spectra are unaffected (Im = 0 exactly).
            bool any_complex = false;
            for (int i = 0; i < config_.num_eigenvalues; ++i) {
                if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue)
                    continue;
                if (std::abs(h_eigenvalues_imag_[i]) > 1e-8) { any_complex = true; break; }
            }

            if (max_eval_change < config_.convergence_threshold &&
                max_res < 10.0 * config_.convergence_threshold) {
                eigenvalue_stable_count++;
            } else {
                eigenvalue_stable_count = 0;
            }

            if (eigenvalue_stable_count >= eigenvalue_stable_required && !any_complex) {
                converged = true;
            }

            // (2026-09-06) Complex-pair stagnation acceptance. A complex Ritz pair's
            // "residual" (of the real-part vector against the real-part eigenvalue) has
            // a floor ~|Im λ|·‖V Im c‖ that no iteration can lower, so with any_complex
            // the strict rule above can never fire and the solve burns max_iterations
            // (MgO 374 EA: max|r| pinned at 1.79e-5 for 70 iterations, 4.8 h wasted).
            // Accept when (a) every REAL root is below tol, (b) the eigenvalues are
            // stable, and (c) max|r| has not moved by >0.1% for 10 consecutive
            // iterations — i.e. the only thing left is the complex-pair floor. The
            // returned vectors are exactly what max_iterations would have returned.
            // GANSU_EOM_NO_COMPLEX_STAGNATION_ACCEPT=1 disables.
            {
                static const bool allow_stag = [] {
                    const char* e = std::getenv("GANSU_EOM_NO_COMPLEX_STAGNATION_ACCEPT");
                    return !(e && e[0] == '1');
                }();
                if (stag_prev_max_res_ > 0.0
                    && std::fabs(max_res - stag_prev_max_res_) < 1e-3 * max_res)
                    ++stag_count_;
                else
                    stag_count_ = 0;
                stag_prev_max_res_ = max_res;
                if (!converged && allow_stag && any_complex && stag_count_ >= 10
                    && max_eval_change < config_.convergence_threshold) {
                    bool real_roots_ok = true;
                    int  n_complex = 0;
                    for (int i = 0; i < config_.num_eigenvalues; ++i) {
                        if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue)
                            continue;
                        if (std::abs(h_eigenvalues_imag_[i]) > 1e-8) { ++n_complex; continue; }
                        if (residual_norms_[i] > config_.convergence_threshold) { real_roots_ok = false; break; }
                    }
                    if (real_roots_ok) {
                        if (config_.verbose > 0)
                            std::cout << "  Davidson: all real roots converged; " << n_complex
                                      << " complex-pair root(s) hold max|r| at " << std::scientific
                                      << std::setprecision(2) << max_res
                                      << " (stagnant 10 iters) — accepting" << std::defaultfloat
                                      << std::endl;
                        converged = true;
                    }
                }
                if (converged) { stag_count_ = 0; stag_prev_max_res_ = -1.0; }
            }

            // Softer fallback: eigenvalues stable to 10*tol for N consecutive iters,
            // accept regardless of residual. (1e-6 boundary noise is common at
            // the 6th decimal; 10*tol = 1e-5 still gives 5-digit eigenvalues.)
            if (max_eval_change < 10.0 * config_.convergence_threshold) {
                eigenvalue_only_stable_count++;
            } else {
                eigenvalue_only_stable_count = 0;
            }
            // (2026-09-05) GANSU_EOM_NO_SOFT_ACCEPT=1 disables this fallback:
            // on MgO 374 the EA solve was accepted at max|r| ~ 1e-2 (tol 1e-5)
            // and the unconverged vectors shifted the STEOM root 0 by 10 meV
            // between runs. With the switch the solve runs to true convergence
            // or max_iterations (then the caller sees unconverged status).
            static const bool no_soft_accept = [] {
                const char* e = std::getenv("GANSU_EOM_NO_SOFT_ACCEPT");
                return e && e[0] == '1';
            }();
            if (!converged && !any_complex &&
                eigenvalue_only_stable_count >= eigenvalue_only_stable_required) {
                if (no_soft_accept) {
                    if (config_.verbose > 0 && eigenvalue_only_stable_count == eigenvalue_only_stable_required)
                        std::cout << "  Davidson: eigenvalues stable for "
                                  << eigenvalue_only_stable_required
                                  << " consecutive iters but max|r|=" << std::scientific
                                  << std::setprecision(2) << max_res
                                  << " > tol — NOT accepting (GANSU_EOM_NO_SOFT_ACCEPT=1), continuing"
                                  << std::defaultfloat << std::endl;
                } else {
                if (config_.verbose > 0) {
                    std::cout << "  Davidson: eigenvalues stable for "
                              << eigenvalue_only_stable_required
                              << " consecutive iters (max|r|=" << std::scientific
                              << std::setprecision(2) << max_res
                              << "), accepting" << std::defaultfloat << std::endl;
                }
                converged = true;
                }   // !no_soft_accept
            }
        }
        prev_eigenvalues = h_eigenvalues_;

        {
            // Find max residual norm
            real_t max_res = *std::max_element(residual_norms_.begin(), residual_norms_.end());
            // Per-iteration progress at verbose > 0 (one line/iter): the EOM/STEOM
            // solves set verbose=1 expecting this (each sigma apply is tens of seconds
            // on large clusters — without it the multi-minute solve looks frozen).
            if (config_.verbose > 0) {
                std::cout << "---- Davidson iteration " << std::setw(3) << (iter + 1) << " ---- ";
                for (int i = 0; i < std::min(5, config_.num_eigenvalues); ++i) {
                    std::cout << std::fixed << std::setprecision(6) << h_eigenvalues_[i];
                    if (i < std::min(5, config_.num_eigenvalues) - 1) std::cout << ", ";
                }
                std::cout << "  max|r|=" << std::scientific << std::setprecision(2) << max_res
                          << std::defaultfloat << std::endl;
            }
            // Progress callback: [eigenvalue_0, ..., eigenvalue_k, max_residual]
            int nev = std::min(config_.num_eigenvalues, (int)h_eigenvalues_.size());
            std::vector<real_t> vals(nev + 1);
            for (int i = 0; i < nev; i++) vals[i] = h_eigenvalues_[i];
            vals[nev] = max_res;
            report_progress("davidson", iter + 1, nev + 1, vals.data());
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
            phase(t_restart, [&] { restart_subspace(); });
            sigma_computed_ = 0;  // Ritz vectors replaced old basis — recompute sigma
            subspace_matrix_built_ = 0;  // Subspace matrix must be rebuilt after restart

            // (2026-09-06) GANSU_DAVIDSON_RESTART_CHECK=1: re-evaluate the restarted
            // basis immediately (sigma + H + Ritz + residuals) and print the lowest
            // eigenvalues and max|r| next to the pre-restart values. A sound thick
            // restart reproduces both; a jump here localises the defect to the
            // restart itself (MgO 374 EA: max|r| 1e-4 → 1e-2..1 at every restart).
            static const bool restart_check = [] {
                const char* e = std::getenv("GANSU_DAVIDSON_RESTART_CHECK");
                return e && e[0] == '1';
            }();
            const real_t pre_restart_max_res =
                *std::max_element(residual_norms_.begin(), residual_norms_.end());
            // Detect eigenvalue collapse after restart for non-Hermitian problems
            if (config_.min_eigenvalue > 0.0 || restart_check) {
                // Recompute sigma and subspace matrix to check new eigenvalues
                if (subspace_dim_ - sigma_computed_ > 0) {
                    apply_sigma_block(sigma_computed_, subspace_dim_ - sigma_computed_);
                }
                sigma_computed_ = subspace_dim_;
                cudaDeviceSynchronize();
                build_subspace_matrix();
                subspace_matrix_built_ = subspace_dim_;
                solve_subspace_eigenproblem();
                if (restart_check) {
                    compute_ritz_vectors_and_residuals();
                    const real_t post_max_res =
                        *std::max_element(residual_norms_.begin(), residual_norms_.end());
                    std::cout << "  Davidson: RESTART CHECK  kept=" << subspace_dim_
                              << "  max|r| pre=" << std::scientific << std::setprecision(2)
                              << pre_restart_max_res << " post=" << post_max_res
                              << "  ev0 pre=" << std::fixed << std::setprecision(6)
                              << pre_restart_evals[0] << " post=" << h_eigenvalues_[0]
                              << "  ev" << (config_.num_eigenvalues - 1) << " pre="
                              << pre_restart_evals[config_.num_eigenvalues - 1]
                              << " post=" << h_eigenvalues_[config_.num_eigenvalues - 1]
                              << std::defaultfloat << std::endl;
                }

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

        // Add correction vectors (Jacobi precondition + Gram-Schmidt orthogonalize)
        phase(t_corr, [&] { add_correction_vectors(); });
    }

    if (!converged && config_.verbose > 0) {
        std::cout << "  Warning: Davidson did not converge in "
                 << config_.max_iterations << " iterations" << std::endl;
    }

    // (2026-09-06) Exit residual summary by root class. The per-iteration line
    // prints only max|r|, which a complex pair can pin (MgO 374 EA: 1.8e-5 for
    // 70+ iterations) while saying nothing about the quality of the REAL roots
    // that STEOM actually consumes. Print both so a log records what was reached.
    if (config_.verbose > 0) {
        real_t max_real = 0.0, max_cplx = 0.0;
        int n_real = 0, n_cplx = 0, n_real_unconv = 0;
        for (int i = 0; i < config_.num_eigenvalues; ++i) {
            if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue) continue;
            const bool cplx = (!config_.symmetric && std::abs(h_eigenvalues_imag_[i]) > 1e-8);
            if (cplx) { ++n_cplx; max_cplx = std::max(max_cplx, residual_norms_[i]); }
            else      { ++n_real; max_real = std::max(max_real, residual_norms_[i]);
                        if (residual_norms_[i] > config_.convergence_threshold) ++n_real_unconv; }
        }
        std::cout << "  Davidson exit: real roots " << n_real << " (max|r|=" << std::scientific
                  << std::setprecision(2) << max_real << ", " << n_real_unconv << " above tol "
                  << config_.convergence_threshold << ")";
        if (n_cplx > 0)
            std::cout << "; complex-pair roots " << n_cplx << " (max|r|=" << max_cplx << ")";
        std::cout << std::defaultfloat << std::endl;
    }

    // (FULL mode) Unresolved complex Ritz pairs at exit: the reported real part
    // is only the pair's real projection — flag it (the legacy path silently
    // returned a garbage 1e30 root here instead).
    if (!config_.symmetric) {
        for (int i = 0; i < config_.num_eigenvalues; ++i) {
            if (std::abs(h_eigenvalues_imag_[i]) > 1e-8) {
                std::cout << "  Warning: Davidson root " << i << " is part of a complex"
                          << " Ritz pair at exit (" << std::scientific << std::setprecision(6)
                          << h_eigenvalues_[i] << " ± " << std::abs(h_eigenvalues_imag_[i])
                          << "i) — the operator is non-normal/near-defective here; treat"
                          << " this root's energy as unresolved." << std::defaultfloat
                          << std::endl;
            }
        }
    }

    if (config_.verbose > 0) {
        const double tot = t_sigma + t_build + t_eig + t_ritz + t_corr + t_restart;
        std::cout << std::fixed << std::setprecision(3)
                  << "  [Davidson-PROF] dim=" << dim_ << " nev=" << config_.num_eigenvalues
                  << " iters=" << num_iterations_ << " phase-total=" << tot << "s\n"
                  << "    sigma(apply)=" << t_sigma << "s  build_subspace=" << t_build
                  << "s  subspace_eig=" << t_eig << "s\n"
                  << "    ritz+resid=" << t_ritz << "s  add_correction(orth)=" << t_corr
                  << "s  restart=" << t_restart << "s" << std::defaultfloat << std::endl;
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
    using Eigen::Map;
    using Eigen::VectorXd;

    if (!gpu::gpu_available()) {
        // CPU implementation using Eigen
        for (int i = start_index; i < start_index + num_vectors; ++i) {
            real_t* vec_i = &d_subspace_vectors_[static_cast<size_t>(i) * dim_];
            Map<VectorXd> vi(vec_i, dim_);

            // Orthogonalize against all previous vectors (CGS2 — see the GPU path
            // note; GANSU_DAVIDSON_CGS1=1 restores the single pass)
            static const int n_gs_passes_cpu = [] {
                const char* e = std::getenv("GANSU_DAVIDSON_CGS1");
                return (e && e[0] == '1') ? 1 : 2;
            }();
            for (int pass = 0; pass < n_gs_passes_cpu; ++pass)
            for (int j = 0; j < i; ++j) {
                const real_t* vec_j = &d_subspace_vectors_[static_cast<size_t>(j) * dim_];
                Map<const VectorXd> vj(vec_j, dim_);
                real_t proj = vj.dot(vi);
                vi -= proj * vj;
            }

            // Normalize
            real_t norm = vi.norm();

            if (norm < 1e-12) {
                if (config_.verbose > 0) {
                    std::cout << "Warning: Linear dependence detected in vector "
                             << i << ", replacing with random vector" << std::endl;
                }
                std::vector<real_t> h_random(dim_);
                std::mt19937_64 rng(1234ULL + i);
                std::normal_distribution<real_t> dist(0.0, 1.0);
                for (int k = 0; k < dim_; ++k) {
                    h_random[k] = dist(rng);
                }
                cudaMemcpy(vec_i, h_random.data(), dim_ * sizeof(real_t), cudaMemcpyHostToDevice);

                orthogonalize_vectors(i, 1);
                continue;
            }

            vi *= 1.0 / norm;
        }
        return;
    }

    // GPU implementation
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    // (2026-09-16) Streamable mode: block CGS2 against the old basis with two
    // DGEMMs per pass (the host-resident basis is read ONCE per GEMM), then a
    // small in-block Gram-Schmidt on the device scratch. The per-vector ddot/axpy
    // loop below reads the whole basis per new vector and, with the basis on the
    // host, took > 5 h for one iteration at MgO 3x3x3 EA (dim 7.9e6, 46 roots).
    if (host_managed_ && d_block_scratch_ && num_vectors > 0
        && num_vectors <= 2 * config_.num_eigenvalues) {
        const int k = num_vectors, m = start_index;
        real_t* d_blk = d_block_scratch_;
        const size_t blk_bytes = static_cast<size_t>(dim_) * k * sizeof(real_t);
        cudaMemcpy(d_blk, &d_subspace_vectors_[static_cast<size_t>(m) * dim_], blk_bytes, cudaMemcpyDefault);
        const real_t one = 1.0, zero = 0.0, minus_one = -1.0;
        static const int n_gs_passes_blk = [] {
            const char* e = std::getenv("GANSU_DAVIDSON_CGS1");
            return (e && e[0] == '1') ? 1 : 2;
        }();
        real_t* d_P = nullptr;
        if (m > 0) {
            tracked_cudaMalloc(&d_P, static_cast<size_t>(m) * k * sizeof(real_t));
            for (int pass = 0; pass < n_gs_passes_blk; ++pass) {
                // P (m x k) = V_old^T (m x dim) * B (dim x k)
                cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, k, dim_,
                            &one, d_subspace_vectors_, dim_, d_blk, dim_, &zero, d_P, m);
                // B -= V_old (dim x m) * P (m x k)
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim_, k, m,
                            &minus_one, d_subspace_vectors_, dim_, d_P, m, &one, d_blk, dim_);
            }
        }
        // in-block CGS2 + normalisation on the device scratch
        for (int i = 0; i < k; ++i) {
            real_t* d_vi = d_blk + static_cast<size_t>(i) * dim_;
            for (int pass = 0; pass < n_gs_passes_blk; ++pass)
                for (int j = 0; j < i; ++j) {
                    const real_t* d_vj = d_blk + static_cast<size_t>(j) * dim_;
                    real_t proj = 0.0;
                    cublasDdot(handle, dim_, d_vj, 1, d_vi, 1, &proj);
                    const real_t alpha = -proj;
                    cublasDaxpy(handle, dim_, &alpha, d_vj, 1, d_vi, 1);
                }
            real_t norm = 0.0;
            cublasDnrm2(handle, dim_, d_vi, 1, &norm);
            if (norm < 1e-12) {
                if (config_.verbose > 0)
                    std::cout << "Warning: Linear dependence detected in vector "
                              << (m + i) << ", replacing with random vector" << std::endl;
                std::vector<real_t> h_random(dim_);
                std::mt19937_64 rng(1234ULL + m + i);
                std::normal_distribution<real_t> dist(0.0, 1.0);
                for (int q = 0; q < dim_; ++q) h_random[q] = dist(rng);
                cudaMemcpy(d_vi, h_random.data(), dim_ * sizeof(real_t), cudaMemcpyHostToDevice);
                // project the random vector out of the old basis (2 GEMV passes) and the block
                if (m > 0) {
                    for (int pass = 0; pass < 2; ++pass) {
                        cublasDgemv(handle, CUBLAS_OP_T, dim_, m, &one, d_subspace_vectors_, dim_,
                                    d_vi, 1, &zero, d_P, 1);
                        cublasDgemv(handle, CUBLAS_OP_N, dim_, m, &minus_one, d_subspace_vectors_, dim_,
                                    d_P, 1, &one, d_vi, 1);
                    }
                }
                for (int pass = 0; pass < 2; ++pass)
                    for (int j = 0; j < i; ++j) {
                        const real_t* d_vj = d_blk + static_cast<size_t>(j) * dim_;
                        real_t proj = 0.0;
                        cublasDdot(handle, dim_, d_vj, 1, d_vi, 1, &proj);
                        const real_t alpha = -proj;
                        cublasDaxpy(handle, dim_, &alpha, d_vj, 1, d_vi, 1);
                    }
                cublasDnrm2(handle, dim_, d_vi, 1, &norm);
            }
            const real_t inv_norm = 1.0 / norm;
            cublasDscal(handle, dim_, &inv_norm, d_vi, 1);
        }
        if (d_P) tracked_cudaFree(d_P);
        cudaMemcpy(&d_subspace_vectors_[static_cast<size_t>(m) * dim_], d_blk, blk_bytes, cudaMemcpyDefault);
        cudaDeviceSynchronize();
        return;
    }

    // (2026-09-06) Re-orthogonalisation ("twice is enough", CGS2). A single
    // classical Gram-Schmidt pass loses orthogonality as the basis grows; with
    // ~1000 vectors of dimension 4.6e6 (MgO 374 EA-EOM, subspace 1140) V^T V
    // drifted far enough from I that the Rayleigh-Ritz problem grew GHOST roots
    // (a 0.000240 Ha eigenvalue in the lowest slot, max|r| jumping 1e-4 -> 1e-2..1
    // mid-cycle, not at restarts — RESTART CHECK showed the restart itself sound).
    // Two passes keep V orthonormal to working precision. GANSU_DAVIDSON_CGS1=1
    // restores the single pass (byte-identical legacy) for A/B.
    static const int n_gs_passes = [] {
        const char* e = std::getenv("GANSU_DAVIDSON_CGS1");
        return (e && e[0] == '1') ? 1 : 2;
    }();

    for (int i = start_index; i < start_index + num_vectors; ++i) {
        real_t* d_vec_i = &d_subspace_vectors_[static_cast<size_t>(i) * dim_];

        // Orthogonalize against all previous vectors (CGS2: two passes)
        for (int pass = 0; pass < n_gs_passes; ++pass)
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

// (2026-09-05) Restart-basis orthonormalisation with compaction.
//
// Why a separate routine: the kept Ritz vectors of a NON-Hermitian problem are
// not orthogonal and can be nearly dependent (Re/Im parts of an almost-real
// complex pair, near-degenerate roots). orthogonalize_vectors() is a single-pass
// classical Gram-Schmidt with an ABSOLUTE 1e-12 threshold, so a vector whose
// projected remainder is 1e-6 of its length is kept and scaled up by 1e6 — pure
// roundoff noise enters the basis. Observed on the MgO 374 EA-EOM solve: every
// restart kicked max|r| from 1e-4 back to 1e-2..1, and the fourth restart put a
// spurious 0.000000 eigenvalue into the lowest slot. Here: two projection passes
// (CGS2, "twice is enough") and a RELATIVE dependence test; a dependent vector
// is dropped and the basis compacted (span of the kept vectors is unchanged
// up to the dropped, already-represented direction). Restart-only, so runs that
// never fill the subspace are byte-identical.
int DavidsonSolver::orthonormalize_compact(int num_vectors, real_t* base) {
    using Eigen::Map;
    using Eigen::VectorXd;
    const real_t rel_tol = 1e-8;
    const real_t abs_tol = 1e-14;
    int kept = 0;
    real_t* const Vb = base ? base : d_subspace_vectors_;   // (2026-09-16) streamable mode passes the device scratch

    if (!gpu::gpu_available()) {
        for (int i = 0; i < num_vectors; ++i) {
            real_t* vec_i = &d_subspace_vectors_[static_cast<size_t>(i) * dim_];
            Map<VectorXd> vi(vec_i, dim_);
            const real_t norm0 = vi.norm();
            if (!(norm0 > abs_tol)) continue;
            for (int pass = 0; pass < 2; ++pass)
                for (int j = 0; j < kept; ++j) {
                    Map<const VectorXd> vj(&d_subspace_vectors_[static_cast<size_t>(j) * dim_], dim_);
                    vi -= vj.dot(vi) * vj;
                }
            const real_t norm = vi.norm();
            if (norm < rel_tol * norm0 || norm < abs_tol) continue;   // dependent → drop
            vi *= 1.0 / norm;
            if (i != kept) {
                Map<VectorXd> vk(&d_subspace_vectors_[static_cast<size_t>(kept) * dim_], dim_);
                vk = vi;
            }
            ++kept;
        }
        return kept;
    }

    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();
    for (int i = 0; i < num_vectors; ++i) {
        real_t* d_vec_i = &Vb[static_cast<size_t>(i) * dim_];
        real_t norm0 = 0.0;
        cublasDnrm2(handle, dim_, d_vec_i, 1, &norm0);
        if (!(norm0 > abs_tol)) continue;
        for (int pass = 0; pass < 2; ++pass) {
            for (int j = 0; j < kept; ++j) {
                const real_t* d_vec_j = &Vb[static_cast<size_t>(j) * dim_];
                real_t proj = 0.0;
                cublasDdot(handle, dim_, d_vec_j, 1, d_vec_i, 1, &proj);
                const real_t alpha = -proj;
                cublasDaxpy(handle, dim_, &alpha, d_vec_j, 1, d_vec_i, 1);
            }
        }
        real_t norm = 0.0;
        cublasDnrm2(handle, dim_, d_vec_i, 1, &norm);
        if (norm < rel_tol * norm0 || norm < abs_tol) continue;       // dependent → drop
        const real_t inv_norm = 1.0 / norm;
        cublasDscal(handle, dim_, &inv_norm, d_vec_i, 1);
        if (i != kept) {
            cudaMemcpy(&Vb[static_cast<size_t>(kept) * dim_], d_vec_i,
                       static_cast<size_t>(dim_) * sizeof(real_t), cudaMemcpyDeviceToDevice);
        }
        ++kept;
    }
    return kept;
}

// ========== Subspace Matrix Construction ==========

void DavidsonSolver::build_subspace_matrix_incremental(int old_dim) {
    using Eigen::Map;
    using Eigen::MatrixXd;

    if (old_dim <= 0 || old_dim >= subspace_dim_) {
        // Full rebuild needed (first call or after restart)
        build_subspace_matrix();
        return;
    }

    // Incremental update: only compute entries involving new vectors.
    // H is subspace_dim_ × subspace_dim_ (packed, lda = subspace_dim_).
    // Old H was old_dim × old_dim (packed, lda = old_dim).
    // We need to:
    //   1. Repack old entries from lda=old_dim to lda=subspace_dim_
    //   2. Compute new column block: H[0:subspace_dim_, old_dim:subspace_dim_] = V_all^T * Σ_new
    //   3. Compute new row block: H[old_dim:subspace_dim_, 0:old_dim] = V_new^T * Σ_old

    int new_cols = subspace_dim_ - old_dim;

    if (!gpu::gpu_available()) {
        // Step 1: Repack old H (old_dim × old_dim, lda=old_dim) → top-left of new H (lda=subspace_dim_)
        // Work backwards to avoid overwriting data we still need
        std::vector<real_t> old_H(old_dim * old_dim);
        std::memcpy(old_H.data(), d_subspace_matrix_, old_dim * old_dim * sizeof(real_t));
        for (int j = old_dim - 1; j >= 0; j--)
            for (int i = old_dim - 1; i >= 0; i--)
                d_subspace_matrix_[i + j * subspace_dim_] = old_H[i + j * old_dim];

        Map<MatrixXd> V_all(d_subspace_vectors_, dim_, subspace_dim_);
        Map<MatrixXd> S_all(d_sigma_vectors_, dim_, subspace_dim_);

        // Step 2: H[:, old_dim:] = V_all^T * Σ_new
        Map<MatrixXd> S_new(d_sigma_vectors_ + (size_t)old_dim * dim_, dim_, new_cols);
        Map<MatrixXd> H_right(d_subspace_matrix_ + (size_t)old_dim * subspace_dim_, subspace_dim_, new_cols);
        H_right.noalias() = V_all.transpose() * S_new;

        // Step 3: H[old_dim:, :old_dim] = V_new^T * Σ_old
        Map<MatrixXd> V_new(d_subspace_vectors_ + (size_t)old_dim * dim_, dim_, new_cols);
        Map<MatrixXd> S_old(d_sigma_vectors_, dim_, old_dim);
        for (int j = 0; j < old_dim; j++)
            for (int i = 0; i < new_cols; i++) {
                real_t val = 0.0;
                for (int k = 0; k < dim_; k++)
                    val += V_new(k, i) * S_old(k, j);
                d_subspace_matrix_[(old_dim + i) + j * subspace_dim_] = val;
            }

        // Symmetrize if needed
        if (config_.symmetric && subspace_dim_ > 1) {
            Map<MatrixXd> H(d_subspace_matrix_, subspace_dim_, subspace_dim_);
            H = (H + H.transpose().eval()) * 0.5;
        }
        return;
    }

    // GPU path
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();
    const real_t alpha = 1.0;
    const real_t beta = 0.0;

    // Step 1: Repack old H
    std::vector<real_t> old_H(old_dim * old_dim);
    cudaMemcpy(old_H.data(), d_subspace_matrix_, old_dim * old_dim * sizeof(real_t), cudaMemcpyDeviceToHost);
    // Zero the new matrix area
    cudaMemset(d_subspace_matrix_, 0, (size_t)subspace_dim_ * subspace_dim_ * sizeof(real_t));
    // Copy old columns back with new stride
    for (int j = 0; j < old_dim; j++)
        cudaMemcpy(d_subspace_matrix_ + j * subspace_dim_,
                   old_H.data() + j * old_dim,
                   old_dim * sizeof(real_t), cudaMemcpyHostToDevice);

    // Step 2: H[:, old_dim:] = V_all^T * Σ_new
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                subspace_dim_, new_cols, dim_,
                &alpha,
                d_subspace_vectors_, dim_,
                d_sigma_vectors_ + (size_t)old_dim * dim_, dim_,
                &beta,
                d_subspace_matrix_ + (size_t)old_dim * subspace_dim_, subspace_dim_);

    // Step 3: H[old_dim:, :old_dim] = V_new^T * Σ_old
    // Write into rows [old_dim..subspace_dim_) of columns [0..old_dim)
    // Use a temp buffer since lda of the destination is subspace_dim_
    real_t* d_temp = nullptr;
    tracked_cudaMalloc(&d_temp, (size_t)new_cols * old_dim * sizeof(real_t));
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                new_cols, old_dim, dim_,
                &alpha,
                d_subspace_vectors_ + (size_t)old_dim * dim_, dim_,
                d_sigma_vectors_, dim_,
                &beta,
                d_temp, new_cols);
    // Copy rows into H
    for (int j = 0; j < old_dim; j++)
        cudaMemcpy(d_subspace_matrix_ + old_dim + j * subspace_dim_,
                   d_temp + j * new_cols,
                   new_cols * sizeof(real_t), cudaMemcpyDeviceToDevice);
    tracked_cudaFree(d_temp);

    // Symmetrize if needed
    if (config_.symmetric && subspace_dim_ > 1) {
        dim3 threads(16, 16);
        dim3 blocks((subspace_dim_ + 15) / 16, (subspace_dim_ + 15) / 16);
        symmetrize_matrix_kernel<<<blocks, threads>>>(d_subspace_matrix_, subspace_dim_);
        cudaDeviceSynchronize();
    }
}

void DavidsonSolver::build_subspace_matrix() {
    using Eigen::Map;
    using Eigen::MatrixXd;

    if (!gpu::gpu_available()) {
        // CPU implementation using Eigen
        // V and Sigma are column-major: dim_ × subspace_dim_
        Map<MatrixXd> V(d_subspace_vectors_, dim_, subspace_dim_);
        Map<MatrixXd> S(d_sigma_vectors_, dim_, subspace_dim_);
        Map<MatrixXd> H(d_subspace_matrix_, subspace_dim_, subspace_dim_);

        // H = V^T * Sigma
        H.noalias() = V.transpose() * S;

        // Symmetrize H = (H + H^T) / 2
        if (config_.symmetric && subspace_dim_ > 1) {
            H = (H + H.transpose().eval()) * 0.5;
        }
        return;
    }

    // GPU implementation
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
        // FULL mode (imag array passed): complex Ritz pairs are KEPT sorted by
        // real part instead of dropped-to-1e30 with zero eigenvectors (the drop
        // made zero Ritz vectors → residual 0 → fake convergence on garbage
        // roots; see d_subspace_eigenvalues_imag_ in the header).  Purely-real
        // subspace spectra are bit-identical (real roots have Im = 0 exactly).
        status = gpu::eigenDecompositionNonSymmetric(
            d_subspace_matrix_,
            d_subspace_eigenvalues_,
            d_subspace_eigenvectors_,
            subspace_dim_,
            /*force_host=*/false,
            d_subspace_eigenvalues_imag_
        );
    }

    if (status != 0) {
        THROW_EXCEPTION("DavidsonSolver: Subspace eigenvalue decomposition failed");
    }

    // Copy eigenvalues to host
    cudaMemcpy(h_eigenvalues_.data(), d_subspace_eigenvalues_,
               config_.num_eigenvalues * sizeof(real_t),
               cudaMemcpyDeviceToHost);
    if (config_.symmetric) {
        std::fill(h_eigenvalues_imag_.begin(), h_eigenvalues_imag_.end(), 0.0);
    } else {
        cudaMemcpy(h_eigenvalues_imag_.data(), d_subspace_eigenvalues_imag_,
                   config_.num_eigenvalues * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
    }

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
    using Eigen::Map;
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    if (!gpu::gpu_available()) {
        // CPU implementation using Eigen
        // Subspace vectors: column-major dim_ × subspace_dim_
        Map<MatrixXd> V(d_subspace_vectors_, dim_, subspace_dim_);
        Map<MatrixXd> Sigma(d_sigma_vectors_, dim_, subspace_dim_);

        for (int i = 0; i < config_.num_eigenvalues; ++i) {
            // Eigenvector i: starts at index i, stride = subspace_dim_ (column i in col-major layout)
            Map<VectorXd, 0, Eigen::InnerStride<>> ci(&d_subspace_eigenvectors_[i], subspace_dim_,
                             Eigen::InnerStride<>(subspace_dim_));

            // Ritz vector: ψ_i = V * c_i
            Map<VectorXd> psi(&d_eigenvectors_[static_cast<size_t>(i) * dim_], dim_);
            psi.noalias() = V * ci;

            // Hψ_i = Sigma * c_i
            Map<VectorXd> res(&d_residuals_[static_cast<size_t>(i) * dim_], dim_);
            res.noalias() = Sigma * ci;

            // Residual: r_i = Hψ_i - λ_i * ψ_i
            res -= h_eigenvalues_[i] * psi;

            // Residual norm
            residual_norms_[i] = res.norm();
        }
        return;
    }

    // GPU implementation
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    // (2026-09-16) Streamable mode: Psi = V C and H Psi = Sigma C as two DGEMMs
    // (each reads the host-resident basis / sigma once), then r_i = H psi_i -
    // lambda_i psi_i on the device buffers. The per-root Dgemv loop below reads V
    // and Sigma once PER ROOT (4 TB per iteration at MgO 3x3x3 EA).
    if (host_managed_) {
        const real_t one = 1.0, zero = 0.0;
        const int nev = config_.num_eigenvalues;
        // Eigenvector i lives in COLUMN i of the row-major (subspace x subspace)
        // buffer, i.e. c_i[j] = buf[i + j*ld] (the Dgemv path reads it with
        // incx = subspace_dim_). Viewed column-major that buffer is C^T, so the
        // coefficient operand is OP_T here (2026-09-16: OP_N made the 2x1x1 gate
        // wander for 394 iterations at max|r| ~0.7).
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_, nev, subspace_dim_,
                    &one, d_subspace_vectors_, dim_, d_subspace_eigenvectors_, subspace_dim_,
                    &zero, d_eigenvectors_, dim_);
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_, nev, subspace_dim_,
                    &one, d_sigma_vectors_, dim_, d_subspace_eigenvectors_, subspace_dim_,
                    &zero, d_residuals_, dim_);
        for (int i = 0; i < nev; ++i) {
            const real_t minus_lambda = -h_eigenvalues_[i];
            cublasDaxpy(handle, dim_, &minus_lambda,
                        &d_eigenvectors_[static_cast<size_t>(i) * dim_], 1,
                        &d_residuals_[static_cast<size_t>(i) * dim_], 1);
            real_t res_norm = 0.0;
            cublasDnrm2(handle, dim_, &d_residuals_[static_cast<size_t>(i) * dim_], 1, &res_norm);
            residual_norms_[i] = res_norm;
        }
        return;
    }

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
    int num_valid = 0;
    for (int i = 0; i < config_.num_eigenvalues; ++i) {
        // Skip spurious eigenvalues below threshold (e.g., ground state in EOM)
        if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue) {
            continue;  // This slot has a spurious eigenvalue, skip it
        }
        ++num_valid;
        if (residual_norms_[i] > config_.convergence_threshold) {
            return false;
        }
    }
    // Need at least one valid eigenvalue to declare convergence
    return num_valid > 0;
}

// ========== Correction Vectors ==========

void DavidsonSolver::add_correction_vectors() {
    using Eigen::Map;
    using Eigen::VectorXd;

    if (!gpu::gpu_available()) {
        // CPU implementation using Eigen
        const real_t* diagonal = linear_op_.get_diagonal_device();
        int num_new_vectors = 0;

        for (int i = 0; i < config_.num_eigenvalues; ++i) {
            if (config_.min_eigenvalue > 0.0 && h_eigenvalues_[i] < config_.min_eigenvalue) {
                continue;
            }
            if (residual_norms_[i] > config_.convergence_threshold) {
                if (subspace_dim_ + num_new_vectors >= config_.max_subspace_size) {
                    break;
                }

                real_t* correction = &d_subspace_vectors_[static_cast<size_t>(subspace_dim_ + num_new_vectors) * dim_];
                const real_t* residual = &d_residuals_[static_cast<size_t>(i) * dim_];

                if (config_.use_preconditioner && diagonal != nullptr) {
                    // Davidson-Jacobi correction: δ[j] = -r[j] / (H_jj - θ_i)
                    real_t eigenvalue = h_eigenvalues_[i];
                    for (int j = 0; j < dim_; ++j) {
                        real_t denom = diagonal[j] - eigenvalue;
                        correction[j] = (std::fabs(denom) > 1e-12) ? -residual[j] / denom : 0.0;
                    }
                } else if (config_.use_preconditioner) {
                    linear_op_.apply_preconditioner(residual, correction);
                } else {
                    cudaMemcpy(correction, residual,
                              dim_ * sizeof(real_t), cudaMemcpyDeviceToDevice);
                }

                // Normalize correction vector
                Map<VectorXd> corr_vec(correction, dim_);
                real_t norm = corr_vec.norm();

                if (norm > 1e-12) {
                    corr_vec *= 1.0 / norm;
                    num_new_vectors++;
                }
            }
        }

        if (num_new_vectors > 0) {
            orthogonalize_vectors(subspace_dim_, num_new_vectors);
            subspace_dim_ += num_new_vectors;

            if (config_.verbose > 2) {
                std::cout << "Added " << num_new_vectors
                         << " correction vectors (subspace dim: "
                         << subspace_dim_ << ")" << std::endl;
            }
        }
        return;
    }

    // GPU implementation
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
    using Eigen::Map;
    using Eigen::VectorXd;
    using Eigen::MatrixXd;

    // Keep 2*num_eigenvalues Ritz vectors (or all if subspace is small)
    // This preserves more spectral information across restarts
    int num_keep = std::min(2 * config_.num_eigenvalues, subspace_dim_);

    if (!gpu::gpu_available()) {
        // CPU implementation using Eigen.
        // (2026-07-11) Scratch = d_sigma_vectors_: the caller invalidates every
        // sigma after a restart (sigma_computed_ = 0, all recomputed for the new
        // basis), so its dim × max_subspace storage is free to borrow — and
        // num_keep ≤ max_subspace always holds. Avoids a dim × num_keep
        // allocation at the moment of peak memory pressure (Doxorubicin EA:
        // 1.24 GiB restart transient OOM'd with the Davidson pool resident).
        real_t* temp = d_sigma_vectors_;

        // Copy already-computed Ritz vectors for the first num_eigenvalues
        int n_from_eigvecs = std::min(config_.num_eigenvalues, num_keep);
        cudaMemcpy(temp, d_eigenvectors_,
                   static_cast<size_t>(dim_) * n_from_eigvecs * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);

        // Compute additional Ritz vectors using Eigen
        Map<MatrixXd> V(d_subspace_vectors_, dim_, subspace_dim_);
        for (int i = n_from_eigvecs; i < num_keep; ++i) {
            // Eigenvector i: starts at index i, stride = subspace_dim_
            Map<VectorXd, 0, Eigen::InnerStride<>> ci(&d_subspace_eigenvectors_[i], subspace_dim_,
                             Eigen::InnerStride<>(subspace_dim_));
            Map<VectorXd> out(&temp[static_cast<size_t>(i) * dim_], dim_);
            out.noalias() = V * ci;
        }

        // Copy back to subspace
        cudaMemcpy(d_subspace_vectors_, temp,
                   static_cast<size_t>(dim_) * num_keep * sizeof(real_t),
                   cudaMemcpyDeviceToDevice);

        subspace_dim_ = num_keep;
        // (2026-07-10) Re-orthonormalize the restart basis. For a NON-Hermitian
        // operator (IP/EA/EE-EOM) the subspace eigenvectors are not mutually
        // orthogonal, so the kept Ritz vectors V·c_i are not an orthonormal set —
        // yet build_subspace_matrix (H = V^T·Σ) and the Gram-Schmidt in
        // add_correction_vectors both assume orthonormal V. Restarting without
        // this silently degenerates the basis (observed: EA roots collapsing to
        // 0.000000 after the first restart under GANSU_EOM_MAX_SUB_PER_ROOT).
        // Gram-Schmidt preserves span(kept Ritz vectors), so the restart stays
        // mathematically sound; for Hermitian problems the Ritz vectors are
        // already orthonormal and this is a numerical no-op.
        // (2026-09-05) CGS2 + drop dependent kept vectors (see orthonormalize_compact).
        {
            const int kept = orthonormalize_compact(num_keep);
            if (kept < num_keep && config_.verbose > 0)
                std::cout << "  Davidson: restart dropped " << (num_keep - kept)
                          << " dependent Ritz vector(s) (kept " << kept << ")" << std::endl;
            subspace_dim_ = std::max(kept, 1);
        }
        return;
    }

    // GPU implementation
    cublasHandle_t handle = gansu::gpu::GPUHandle::cublas();

    // (2026-09-16) Streamable mode: the kept Ritz vectors are formed with ONE
    // DGEMM into the device scratch (dim x 2 nev >= dim x num_keep), orthonormalised
    // there, and written back to the host-resident basis once.
    if (host_managed_ && d_block_scratch_) {
        real_t* d_tmp = d_block_scratch_;
        const int n_from_eigvecs = std::min(config_.num_eigenvalues, num_keep);
        cudaMemcpy(d_tmp, d_eigenvectors_,
                   static_cast<size_t>(dim_) * n_from_eigvecs * sizeof(real_t), cudaMemcpyDeviceToDevice);
        if (num_keep > n_from_eigvecs) {
            const real_t one = 1.0, zero = 0.0;
            // same layout as compute_ritz_vectors_and_residuals: OP_T on C^T,
            // eigenvectors n_from.. are ROWS n_from.. of the stored buffer (+n_from offset)
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim_, num_keep - n_from_eigvecs, subspace_dim_,
                        &one, d_subspace_vectors_, dim_,
                        &d_subspace_eigenvectors_[static_cast<size_t>(n_from_eigvecs)], subspace_dim_,
                        &zero, &d_tmp[static_cast<size_t>(n_from_eigvecs) * dim_], dim_);
        }
        const int kept = orthonormalize_compact(num_keep, d_tmp);
        if (kept < num_keep && config_.verbose > 0)
            std::cout << "  Davidson: restart dropped " << (num_keep - kept)
                      << " dependent Ritz vector(s) (kept " << kept << ")" << std::endl;
        subspace_dim_ = std::max(kept, 1);
        cudaMemcpy(d_subspace_vectors_, d_tmp,
                   static_cast<size_t>(dim_) * subspace_dim_ * sizeof(real_t), cudaMemcpyDefault);
        cudaDeviceSynchronize();
        return;
    }

    // (2026-07-11) Scratch = d_sigma_vectors_ (dim × max_subspace ≥ dim × num_keep):
    // every sigma is invalidated/recomputed after a restart, so borrowing it here
    // is free — and avoids a dim × num_keep tracked_cudaMalloc at the moment of
    // peak pressure (Doxorubicin EA: the 1.24 GiB restart transient OOM'd with
    // the Davidson pool resident).
    real_t* d_temp = d_sigma_vectors_;

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

    subspace_dim_ = num_keep;
    // (2026-07-10) Re-orthonormalize the restart basis — see the CPU-path note
    // above: non-Hermitian Ritz vectors are not mutually orthogonal, and every
    // downstream step assumes an orthonormal V. Span-preserving, so restarts
    // stay mathematically sound; Hermitian case ≈ numerical no-op.
    // (2026-09-05) CGS2 + drop dependent kept vectors (see orthonormalize_compact).
    {
        const int kept = orthonormalize_compact(num_keep);
        if (kept < num_keep && config_.verbose > 0)
            std::cout << "  Davidson: restart dropped " << (num_keep - kept)
                      << " dependent Ritz vector(s) (kept " << kept << ")" << std::endl;
        subspace_dim_ = std::max(kept, 1);
    }
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
