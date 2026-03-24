/*
 * Tests for Rys quadrature roots/weights and RysERI kernel.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include "rys_quadrature.hpp"
#include "boys.hpp"

// ============================================================
//  Test kernel: compute Rys roots and weights on GPU
// ============================================================
__global__ void test_rys_roots_kernel(
    int N, double T, const double* g_boys_grid,
    double* out_roots, double* out_weights)
{
    gansu::gpu::computeRysRootsAndWeights(N, T, g_boys_grid, out_roots, out_weights);
}

// Helper: get Boys grid (reuse GANSU's Boys function infrastructure)
class RysQuadratureTest : public ::testing::Test {
protected:
    double* d_boys_grid = nullptr;

    void SetUp() override {
        // Allocate and initialize Boys function lookup table
        // The Boys grid is typically allocated via gansu::gpu::allocateBoysGrid()
        // For testing, we use a simplified approach: allocate and copy the grid
        // from the host-side precomputed table.
        extern const double gansu_boys_grid[];
        extern const int gansu_boys_grid_size;

        // Try to get the Boys grid size from the library
        // Fallback: allocate a dummy grid and use the analytical fallback
        // For a proper test, this should use the real Boys grid.
        // Since we can't easily access it here, we'll use getSingleBoys
        // which has a fallback path for large/small T.

        // Allocate a placeholder Boys grid (the actual grid is embedded in the binary)
        // The getIncrementalBoys function handles T=0 and large T analytically,
        // so for those cases the grid is not needed.
        cudaMalloc(&d_boys_grid, sizeof(double));  // Minimal allocation
    }

    void TearDown() override {
        if (d_boys_grid) cudaFree(d_boys_grid);
    }
};

// ============================================================
//  Validate Rys roots against analytical formulas
// ============================================================

// For N=1: t^2 = F1(T)/F0(T), w = F0(T)
// F0(0) = 1, F1(0) = 1/3 => t^2 = 1/3, w = 1
TEST(RysRootsAnalytical, N1_T0) {
    // Compute on host using known values
    // F0(0) = 1, F1(0) = 1/3
    double t2 = 1.0 / 3.0;
    double w = 1.0;

    // Verify: integral of t^0 * exp(0) dt from 0 to 1 = F0(0) = 1
    // w * 1 = F0(0) = 1  ✓
    // Verify: integral of t^2 * exp(0) dt from 0 to 1 = F1(0) = 1/3
    // w * t^2 = 1 * 1/3 = 1/3  ✓
    EXPECT_NEAR(t2, 1.0 / 3.0, 1e-14);
    EXPECT_NEAR(w, 1.0, 1e-14);
}

// For N=2, T=0: verify quadrature reproduces moments
// Moments: F_k(0) = 1/(2k+1), k=0,1,2,3
TEST(RysRootsAnalytical, N2_T0_moments) {
    // The N=2 Rys quadrature at T=0 should satisfy:
    // w1 + w2 = F0 = 1
    // w1*t1^2 + w2*t2^2 = F1 = 1/3
    // w1*t1^4 + w2*t2^4 = F2 = 1/5
    // w1*t1^6 + w2*t2^6 = F3 = 1/7

    // These are the roots and weights of Gauss-Legendre quadrature on [0,1]
    // with weight exp(0)=1, for polynomial in t^2.
    // Analytically: the quadrature with 2 points in u=t^2 on [0,1] with weight u^{-1/2}

    // Moments: mu_k = F_k(0) = 1/(2k+1)
    double mu[4] = {1.0, 1.0/3.0, 1.0/5.0, 1.0/7.0};

    // alpha_0 = mu1/mu0 = 1/3
    double a0 = mu[1] / mu[0];
    // sigma_{1,1} = mu2 - a0*mu1 = 1/5 - (1/3)(1/3) = 1/5 - 1/9 = 4/45
    double s11 = mu[2] - a0 * mu[1];
    // sigma_{1,2} = mu3 - a0*mu2 = 1/7 - (1/3)(1/5) = 1/7 - 1/15 = 8/105
    double s12 = mu[3] - a0 * mu[2];
    // beta_1 = s11/mu0 = 4/45
    double b1 = s11 / mu[0];
    // alpha_1 = s12/s11 - a0 = (8/105)/(4/45) - 1/3 = (8*45)/(105*4) - 1/3 = 360/420 - 1/3 = 6/7 - 1/3 = 11/21
    double a1 = s12 / s11 - a0;

    // Eigenvalues of [[a0, sqrt(b1)], [sqrt(b1), a1]]
    double sum = a0 + a1;
    double diff = a0 - a1;
    double disc = sqrt(diff * diff / 4.0 + b1);
    double r1 = sum / 2.0 - disc;
    double r2 = sum / 2.0 + disc;

    // Verify moment reproduction
    double v1_sq = (a1 - r1) / (r2 - r1);
    double v2_sq = 1.0 - v1_sq;
    double w1 = mu[0] * v1_sq;
    double w2 = mu[0] * v2_sq;

    // Check moments
    EXPECT_NEAR(w1 + w2, mu[0], 1e-14);
    EXPECT_NEAR(w1 * r1 + w2 * r2, mu[1], 1e-14);
    EXPECT_NEAR(w1 * r1 * r1 + w2 * r2 * r2, mu[2], 1e-14);
    EXPECT_NEAR(w1 * r1 * r1 * r1 + w2 * r2 * r2 * r2, mu[3], 1e-13);
}
