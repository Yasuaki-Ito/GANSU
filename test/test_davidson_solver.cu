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

#include <gtest/gtest.h>
#include "davidson_solver.hpp"
#include "test_operators.hpp"
#include "fci_hamiltonian.hpp"
#include "gpu_manager.hpp"
#include "device_host_memory.hpp"
#include <vector>
#include <cmath>

using namespace gansu;

// ========== Step 1: Test operator apply ==========

TEST(DavidsonSolverTest, Step1_DiagonalOperatorApply) {
    // Verify that DiagonalOperator correctly computes y = D * x
    std::vector<real_t> diagonal = {1.0, 2.0, 3.0, 4.0};
    DiagonalOperator op(diagonal);

    EXPECT_EQ(op.dimension(), 4);

    // x = [1, 1, 1, 1]
    std::vector<real_t> h_input = {1.0, 1.0, 1.0, 1.0};
    std::vector<real_t> h_output(4, 0.0);

    real_t* d_input = nullptr;
    real_t* d_output = nullptr;
    tracked_cudaMalloc(&d_input, 4 * sizeof(real_t));
    tracked_cudaMalloc(&d_output, 4 * sizeof(real_t));

    cudaMemcpy(d_input, h_input.data(), 4 * sizeof(real_t), cudaMemcpyHostToDevice);

    op.apply(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, 4 * sizeof(real_t), cudaMemcpyDeviceToHost);

    // y = D * [1,1,1,1] = [1,2,3,4]
    EXPECT_DOUBLE_EQ(h_output[0], 1.0);
    EXPECT_DOUBLE_EQ(h_output[1], 2.0);
    EXPECT_DOUBLE_EQ(h_output[2], 3.0);
    EXPECT_DOUBLE_EQ(h_output[3], 4.0);

    tracked_cudaFree(d_input);
    tracked_cudaFree(d_output);
}

// ========== Step 2: Test eigenDecomposition directly ==========

TEST(DavidsonSolverTest, Step2_EigenDecomposition) {
    // Known 3x3 diagonal matrix (row-major)
    // eigenDecomposition expects row-major input (it handles column-major internally)
    std::vector<real_t> h_matrix = {
        3.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 2.0
    };

    real_t* d_matrix = nullptr;
    real_t* d_eigenvalues = nullptr;
    real_t* d_eigenvectors = nullptr;

    tracked_cudaMalloc(&d_matrix, 9 * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvalues, 3 * sizeof(real_t));
    tracked_cudaMalloc(&d_eigenvectors, 9 * sizeof(real_t));

    cudaMemcpy(d_matrix, h_matrix.data(), 9 * sizeof(real_t), cudaMemcpyHostToDevice);

    int status = gpu::eigenDecomposition(d_matrix, d_eigenvalues, d_eigenvectors, 3);
    EXPECT_EQ(status, 0);

    // Check eigenvalues (should be sorted ascending: 1, 2, 3)
    std::vector<real_t> h_eigenvalues(3);
    cudaMemcpy(h_eigenvalues.data(), d_eigenvalues, 3 * sizeof(real_t), cudaMemcpyDeviceToHost);

    EXPECT_NEAR(h_eigenvalues[0], 1.0, 1e-10);
    EXPECT_NEAR(h_eigenvalues[1], 2.0, 1e-10);
    EXPECT_NEAR(h_eigenvalues[2], 3.0, 1e-10);

    // Check eigenvectors layout after transpose
    // eigenDecomposition transposes to row-major, so eigenvector k is in COLUMN k
    std::vector<real_t> h_eigvecs(9);
    cudaMemcpy(h_eigvecs.data(), d_eigenvectors, 9 * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Print layout for debugging
    std::cout << "Eigenvector matrix (row-major, after transpose):" << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cout << "  Row " << i << ": ";
        for (int j = 0; j < 3; ++j) {
            std::cout << h_eigvecs[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    // Eigenvalue 0 (=1.0): eigenvector should be e_1 = [0,1,0]
    // In COLUMN 0 (stride 3): h_eigvecs[0], h_eigvecs[3], h_eigvecs[6]
    std::cout << "Eigenvector 0 (column 0): "
              << h_eigvecs[0] << " " << h_eigvecs[3] << " " << h_eigvecs[6] << std::endl;

    // Eigenvalue 1 (=2.0): eigenvector should be e_2 = [0,0,1]
    // In COLUMN 1 (stride 3): h_eigvecs[1], h_eigvecs[4], h_eigvecs[7]
    std::cout << "Eigenvector 1 (column 1): "
              << h_eigvecs[1] << " " << h_eigvecs[4] << " " << h_eigvecs[7] << std::endl;

    // Eigenvalue 2 (=3.0): eigenvector should be e_0 = [1,0,0]
    // In COLUMN 2 (stride 3): h_eigvecs[2], h_eigvecs[5], h_eigvecs[8]
    std::cout << "Eigenvector 2 (column 2): "
              << h_eigvecs[2] << " " << h_eigvecs[5] << " " << h_eigvecs[8] << std::endl;

    tracked_cudaFree(d_matrix);
    tracked_cudaFree(d_eigenvalues);
    tracked_cudaFree(d_eigenvectors);
}

// ========== Step 3: Simplest Davidson test - 2x2 diagonal ==========

TEST(DavidsonSolverTest, Step3_Diagonal2x2) {
    // Simplest possible: 2x2 diagonal, find 1 eigenvalue
    std::vector<real_t> diagonal = {3.0, 1.0};
    DiagonalOperator op(diagonal);

    DavidsonConfig config;
    config.num_eigenvalues = 1;
    config.convergence_threshold = 1e-6;
    config.max_iterations = 50;
    config.verbose = 2;  // Verbose for debugging

    DavidsonSolver solver(op, config);
    bool converged = solver.solve();

    const auto& eigenvalues = solver.get_eigenvalues();
    std::cout << "Final eigenvalue: " << eigenvalues[0] << " (expected: 1.0)" << std::endl;
    std::cout << "Residual: " << solver.get_residual_norms()[0] << std::endl;
    std::cout << "Iterations: " << solver.get_num_iterations() << std::endl;

    EXPECT_TRUE(converged);
    EXPECT_NEAR(eigenvalues[0], 1.0, 1e-4);
}

// ========== Step 4: 4x4 diagonal, 1 eigenvalue ==========

TEST(DavidsonSolverTest, Step4_Diagonal4x4_OneEigenvalue) {
    std::vector<real_t> diagonal = {5.0, 2.0, 8.0, 1.0};
    DiagonalOperator op(diagonal);

    DavidsonConfig config;
    config.num_eigenvalues = 1;
    config.convergence_threshold = 1e-6;
    config.max_iterations = 50;
    config.verbose = 0;

    DavidsonSolver solver(op, config);
    bool converged = solver.solve();

    EXPECT_TRUE(converged);
    EXPECT_NEAR(solver.get_eigenvalues()[0], 1.0, 1e-4);
    std::cout << "4x4 diagonal: eigenvalue = " << solver.get_eigenvalues()[0]
              << ", iterations = " << solver.get_num_iterations() << std::endl;
}

// ========== Step 5: 10x10 diagonal, 3 eigenvalues ==========

TEST(DavidsonSolverTest, Step5_Diagonal10x10_ThreeEigenvalues) {
    std::vector<real_t> diagonal(10);
    for (int i = 0; i < 10; ++i) {
        diagonal[i] = static_cast<real_t>(i + 1);
    }
    DiagonalOperator op(diagonal);

    DavidsonConfig config;
    config.num_eigenvalues = 3;
    config.convergence_threshold = 1e-6;
    config.max_iterations = 100;
    config.verbose = 0;

    DavidsonSolver solver(op, config);
    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    const auto& eigenvalues = solver.get_eigenvalues();
    EXPECT_NEAR(eigenvalues[0], 1.0, 1e-4);
    EXPECT_NEAR(eigenvalues[1], 2.0, 1e-4);
    EXPECT_NEAR(eigenvalues[2], 3.0, 1e-4);
    std::cout << "10x10 diagonal: eigenvalues = "
              << eigenvalues[0] << ", " << eigenvalues[1] << ", " << eigenvalues[2]
              << ", iterations = " << solver.get_num_iterations() << std::endl;
}

// ========== Step 6: Symmetric matrix ==========

TEST(DavidsonSolverTest, Step6_SymmetricMatrix) {
    // 4x4 tridiagonal: known eigenvalues 2 - 2*cos(k*pi/5)
    std::vector<real_t> matrix = {
        2.0, -1.0,  0.0,  0.0,
       -1.0,  2.0, -1.0,  0.0,
        0.0, -1.0,  2.0, -1.0,
        0.0,  0.0, -1.0,  2.0
    };
    SymmetricMatrixOperator op(matrix, 4);

    DavidsonConfig config;
    config.num_eigenvalues = 2;
    config.convergence_threshold = 1e-6;
    config.max_iterations = 100;
    config.verbose = 0;

    DavidsonSolver solver(op, config);
    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    const auto& eigenvalues = solver.get_eigenvalues();
    double pi = 3.14159265358979323846;
    double expected0 = 2.0 - 2.0 * cos(pi / 5.0);
    double expected1 = 2.0 - 2.0 * cos(2.0 * pi / 5.0);

    EXPECT_NEAR(eigenvalues[0], expected0, 1e-4);
    EXPECT_NEAR(eigenvalues[1], expected1, 1e-4);
    std::cout << "Symmetric 4x4: eigenvalues = " << eigenvalues[0] << " (expected " << expected0
              << "), " << eigenvalues[1] << " (expected " << expected1 << ")" << std::endl;
}

// ========== Step 7: FCI Hamiltonian with 2 orbitals ==========

TEST(FCIHamiltonianTest, TwoOrbitalTwoElectron) {
    // M=2 spatial orbitals, 1 alpha, 1 beta (like H2 minimal basis)
    // FCI dimension = C(2,1)^2 = 4
    //
    // Determinants (alpha_idx * num_beta_det + beta_idx):
    //   0: alpha=|10> beta=|10>  (both in orbital 0)
    //   1: alpha=|10> beta=|01>  (alpha=0, beta=1)
    //   2: alpha=|01> beta=|10>  (alpha=1, beta=0)
    //   3: alpha=|01> beta=|01>  (both in orbital 1)
    //
    // h1 = [[-2, 0], [0, -1]]
    // ERI (pq|rs): (00|00)=1.0, (11|11)=0.5, (00|11)=(11|00)=0.3,
    //              (01|01)=(10|10)=0.2, (01|10)=(10|01)=0.2
    //
    // Expected 4x4 matrix:
    //   H = [[-3.0,   0,     0,     0.2 ],
    //        [ 0,    -2.7,   0.2,   0   ],
    //        [ 0,     0.2,  -2.7,   0   ],
    //        [ 0.2,   0,     0,    -1.5 ]]
    //
    // Eigenvalues: -3.02621, -2.9, -2.5, -1.47379

    const int M = 2;

    // 1-electron MO integrals [M x M] row-major
    std::vector<real_t> h_h1 = {
        -2.0,  0.0,
         0.0, -1.0
    };

    // 2-electron MO integrals [M^4] = 16 elements
    // (pq|rs) = h_eri[p*8 + q*4 + r*2 + s]
    std::vector<real_t> h_eri(16, 0.0);
    // (00|00) = 1.0
    h_eri[0*8 + 0*4 + 0*2 + 0] = 1.0;
    // (11|11) = 0.5
    h_eri[1*8 + 1*4 + 1*2 + 1] = 0.5;
    // (00|11) = (11|00) = 0.3
    h_eri[0*8 + 0*4 + 1*2 + 1] = 0.3;
    h_eri[1*8 + 1*4 + 0*2 + 0] = 0.3;
    // (01|01) = (10|10) = 0.2
    h_eri[0*8 + 1*4 + 0*2 + 1] = 0.2;
    h_eri[1*8 + 0*4 + 1*2 + 0] = 0.2;
    // (01|10) = (10|01) = 0.2
    h_eri[0*8 + 1*4 + 1*2 + 0] = 0.2;
    h_eri[1*8 + 0*4 + 0*2 + 1] = 0.2;

    // Copy to device
    real_t* d_h1 = nullptr;
    real_t* d_eri = nullptr;
    tracked_cudaMalloc(&d_h1, M * M * sizeof(real_t));
    tracked_cudaMalloc(&d_eri, M * M * M * M * sizeof(real_t));
    cudaMemcpy(d_h1, h_h1.data(), M * M * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_eri, h_eri.data(), M * M * M * M * sizeof(real_t), cudaMemcpyHostToDevice);

    // Create FCI operator
    FCIHamiltonianOperator fci_op(d_h1, d_eri, M, 1, 1);
    EXPECT_EQ(fci_op.dimension(), 4);

    // Test apply: compute H * e_0 (first basis vector)
    std::vector<real_t> h_input(4, 0.0);
    h_input[0] = 1.0;
    std::vector<real_t> h_output(4, 0.0);

    real_t* d_input = nullptr;
    real_t* d_output = nullptr;
    tracked_cudaMalloc(&d_input, 4 * sizeof(real_t));
    tracked_cudaMalloc(&d_output, 4 * sizeof(real_t));

    cudaMemcpy(d_input, h_input.data(), 4 * sizeof(real_t), cudaMemcpyHostToDevice);
    fci_op.apply(d_input, d_output);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output.data(), d_output, 4 * sizeof(real_t), cudaMemcpyDeviceToHost);

    // H * e_0 should be column 0 of H: [-3.0, 0, 0, 0.2]
    std::cout << "H * e_0 = [" << h_output[0] << ", " << h_output[1]
              << ", " << h_output[2] << ", " << h_output[3] << "]" << std::endl;
    EXPECT_NEAR(h_output[0], -3.0, 1e-10);
    EXPECT_NEAR(h_output[1],  0.0, 1e-10);
    EXPECT_NEAR(h_output[2],  0.0, 1e-10);
    EXPECT_NEAR(h_output[3],  0.2, 1e-10);

    // Test Davidson solver
    DavidsonConfig config;
    config.num_eigenvalues = 1;
    config.convergence_threshold = 1e-8;
    config.max_iterations = 100;
    config.verbose = 0;

    DavidsonSolver solver(fci_op, config);
    bool converged = solver.solve();

    EXPECT_TRUE(converged);

    // Expected lowest eigenvalue: -3.02621
    // Exact: (-4.5 - sqrt(2.41)) / 2
    double expected = (-4.5 - std::sqrt(2.41)) / 2.0;
    real_t fci_eigenvalue = solver.get_eigenvalues()[0];
    std::cout << "FCI eigenvalue: " << fci_eigenvalue
              << " (expected: " << expected << ")" << std::endl;
    EXPECT_NEAR(fci_eigenvalue, expected, 1e-6);

    tracked_cudaFree(d_input);
    tracked_cudaFree(d_output);
    tracked_cudaFree(d_h1);
    tracked_cudaFree(d_eri);
}

int main(int argc, char **argv) {
    gansu::gpu::cusolverManager cusolver;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
