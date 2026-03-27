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
 * @file geometry_optimizer.hpp
 * @brief Geometry optimization algorithms for molecular structure optimization.
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>
#include <stdexcept>

namespace gansu {

/**
 * @brief Abstract base class for geometry optimization algorithms.
 */
class GeometryOptimizer {
public:
    GeometryOptimizer(int ndim) : ndim_(ndim) {}
    virtual ~GeometryOptimizer() = default;

    virtual void initialize(const std::vector<double>& coords, const std::vector<double>& grad) = 0;

    virtual std::vector<double> compute_search_direction(
        const std::vector<double>& coords, const std::vector<double>& grad) = 0;

    virtual void step_completed(
        const std::vector<double>& s, const std::vector<double>& y,
        const std::vector<double>& grad_new) = 0;

    virtual bool use_line_search() const { return true; }

    virtual std::string name() const = 0;

    static std::unique_ptr<GeometryOptimizer> create(const std::string& optimizer_name, int ndim);

    static bool is_valid_optimizer(const std::string& name) {
        static const char* valid[] = {
            "bfgs", "dfp", "sr1", "gdiis",
            "cg-fr", "cg-pr", "cg-hs", "cg-dy", "sd"
        };
        for(const auto& v : valid) {
            if(name == v) return true;
        }
        return false;
    }

protected:
    int ndim_;

    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for(size_t i = 0; i < a.size(); i++) s += a[i] * b[i];
        return s;
    }
};


// =============================================================================
// Quasi-Newton base class (BFGS, DFP, SR1)
// =============================================================================

class QuasiNewtonOptimizer : public GeometryOptimizer {
public:
    QuasiNewtonOptimizer(int ndim) : GeometryOptimizer(ndim), H_inv_(ndim * ndim, 0.0) {}

    void initialize(const std::vector<double>&, const std::vector<double>&) override {
        std::fill(H_inv_.begin(), H_inv_.end(), 0.0);
        for(int i = 0; i < ndim_; i++) H_inv_[i * ndim_ + i] = 1.0;
    }

    std::vector<double> compute_search_direction(
        const std::vector<double>&, const std::vector<double>& grad) override {
        // p = -H_inv * grad
        std::vector<double> p(ndim_, 0.0);
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                p[i] -= H_inv_[i * ndim_ + j] * grad[j];
        return p;
    }

protected:
    std::vector<double> H_inv_;

    std::vector<double> H_inv_times(const std::vector<double>& v) const {
        std::vector<double> r(ndim_, 0.0);
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                r[i] += H_inv_[i * ndim_ + j] * v[j];
        return r;
    }
};


// =============================================================================
// BFGS (Broyden-Fletcher-Goldfarb-Shanno)
// =============================================================================

class BFGSOptimizer : public QuasiNewtonOptimizer {
public:
    using QuasiNewtonOptimizer::QuasiNewtonOptimizer;

    void step_completed(const std::vector<double>& s, const std::vector<double>& y,
                       const std::vector<double>&) override {
        double sy = dot(s, y);
        if(sy <= 1.0e-10) return;

        double rho = 1.0 / sy;
        auto Hy = H_inv_times(y);
        double yHy = dot(y, Hy);

        // Sherman-Morrison-Woodbury form
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                H_inv_[i * ndim_ + j] += (sy + yHy) * rho * rho * s[i] * s[j]
                                       - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
    }

    std::string name() const override { return "BFGS"; }
};


// =============================================================================
// DFP (Davidon-Fletcher-Powell)
// =============================================================================

class DFPOptimizer : public QuasiNewtonOptimizer {
public:
    using QuasiNewtonOptimizer::QuasiNewtonOptimizer;

    void step_completed(const std::vector<double>& s, const std::vector<double>& y,
                       const std::vector<double>&) override {
        double sy = dot(s, y);
        if(sy <= 1.0e-10) return;

        auto Hy = H_inv_times(y);
        double yHy = dot(y, Hy);

        // DFP formula: H' = H - (H*y*y^T*H)/(y^T*H*y) + (s*s^T)/(y^T*s)
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                H_inv_[i * ndim_ + j] += s[i] * s[j] / sy - Hy[i] * Hy[j] / yHy;
    }

    std::string name() const override { return "DFP"; }
};


// =============================================================================
// SR1 (Symmetric Rank-1)
// =============================================================================

class SR1Optimizer : public QuasiNewtonOptimizer {
public:
    using QuasiNewtonOptimizer::QuasiNewtonOptimizer;

    void step_completed(const std::vector<double>& s, const std::vector<double>& y,
                       const std::vector<double>&) override {
        // SR1: H' = H + (s - H*y)(s - H*y)^T / ((s - H*y)^T * y)
        auto Hy = H_inv_times(y);

        std::vector<double> diff(ndim_);
        for(int i = 0; i < ndim_; i++) diff[i] = s[i] - Hy[i];

        double denom = dot(diff, y);

        // Skip if denominator is too small (Nocedal & Wright criterion)
        double norm_y = std::sqrt(dot(y, y));
        double norm_diff = std::sqrt(dot(diff, diff));
        if(std::abs(denom) < 1.0e-8 * norm_y * norm_diff) return;

        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                H_inv_[i * ndim_ + j] += diff[i] * diff[j] / denom;
    }

    std::string name() const override { return "SR1"; }
};


// =============================================================================
// Conjugate Gradient base class
// =============================================================================

class ConjugateGradientOptimizer : public GeometryOptimizer {
public:
    ConjugateGradientOptimizer(int ndim)
        : GeometryOptimizer(ndim), prev_grad_(ndim, 0.0), prev_direction_(ndim, 0.0), first_step_(true) {}

    void initialize(const std::vector<double>&, const std::vector<double>&) override {
        first_step_ = true;
    }

    std::vector<double> compute_search_direction(
        const std::vector<double>&, const std::vector<double>& grad) override {

        std::vector<double> p(ndim_);

        if(first_step_) {
            for(int i = 0; i < ndim_; i++) p[i] = -grad[i];
            first_step_ = false;
        } else {
            double beta = compute_beta(prev_grad_, grad, prev_direction_);
            if(beta < 0.0) beta = 0.0; // restart to steepest descent

            for(int i = 0; i < ndim_; i++)
                p[i] = -grad[i] + beta * prev_direction_[i];

            // Descent direction check
            if(dot(grad, p) >= 0.0) {
                for(int i = 0; i < ndim_; i++) p[i] = -grad[i];
            }
        }

        prev_direction_ = p;
        prev_grad_ = grad;
        return p;
    }

    void step_completed(const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&) override {}

protected:
    virtual double compute_beta(const std::vector<double>& g_old,
                                const std::vector<double>& g_new,
                                const std::vector<double>& d_old) const = 0;

    std::vector<double> prev_grad_;
    std::vector<double> prev_direction_;
    bool first_step_;
};


// =============================================================================
// Fletcher-Reeves: beta = ||g_{k+1}||^2 / ||g_k||^2
// =============================================================================

class CGFletcherReeves : public ConjugateGradientOptimizer {
public:
    using ConjugateGradientOptimizer::ConjugateGradientOptimizer;
    std::string name() const override { return "CG-FR"; }
protected:
    double compute_beta(const std::vector<double>& g_old,
                       const std::vector<double>& g_new,
                       const std::vector<double>&) const override {
        double g_old_sq = dot(g_old, g_old);
        if(g_old_sq < 1.0e-30) return 0.0;
        return dot(g_new, g_new) / g_old_sq;
    }
};


// =============================================================================
// Polak-Ribiere: beta = g_{k+1}^T (g_{k+1} - g_k) / ||g_k||^2
// =============================================================================

class CGPolakRibiere : public ConjugateGradientOptimizer {
public:
    using ConjugateGradientOptimizer::ConjugateGradientOptimizer;
    std::string name() const override { return "CG-PR"; }
protected:
    double compute_beta(const std::vector<double>& g_old,
                       const std::vector<double>& g_new,
                       const std::vector<double>&) const override {
        double g_old_sq = dot(g_old, g_old);
        if(g_old_sq < 1.0e-30) return 0.0;
        double gTy = 0.0;
        for(int i = 0; i < ndim_; i++) gTy += g_new[i] * (g_new[i] - g_old[i]);
        return gTy / g_old_sq;
    }
};


// =============================================================================
// Hestenes-Stiefel: beta = g_{k+1}^T (g_{k+1} - g_k) / d_k^T (g_{k+1} - g_k)
// =============================================================================

class CGHestenesStifel : public ConjugateGradientOptimizer {
public:
    using ConjugateGradientOptimizer::ConjugateGradientOptimizer;
    std::string name() const override { return "CG-HS"; }
protected:
    double compute_beta(const std::vector<double>& g_old,
                       const std::vector<double>& g_new,
                       const std::vector<double>& d_old) const override {
        double gTy = 0.0, dTy = 0.0;
        for(int i = 0; i < ndim_; i++) {
            double yi = g_new[i] - g_old[i];
            gTy += g_new[i] * yi;
            dTy += d_old[i] * yi;
        }
        if(std::abs(dTy) < 1.0e-30) return 0.0;
        return gTy / dTy;
    }
};


// =============================================================================
// Dai-Yuan: beta = ||g_{k+1}||^2 / d_k^T (g_{k+1} - g_k)
// =============================================================================

class CGDaiYuan : public ConjugateGradientOptimizer {
public:
    using ConjugateGradientOptimizer::ConjugateGradientOptimizer;
    std::string name() const override { return "CG-DY"; }
protected:
    double compute_beta(const std::vector<double>& g_old,
                       const std::vector<double>& g_new,
                       const std::vector<double>& d_old) const override {
        double dTy = 0.0;
        for(int i = 0; i < ndim_; i++) dTy += d_old[i] * (g_new[i] - g_old[i]);
        if(std::abs(dTy) < 1.0e-30) return 0.0;
        return dot(g_new, g_new) / dTy;
    }
};


// =============================================================================
// Steepest Descent: p = -grad
// =============================================================================

class SteepestDescentOptimizer : public GeometryOptimizer {
public:
    using GeometryOptimizer::GeometryOptimizer;

    void initialize(const std::vector<double>&, const std::vector<double>&) override {}

    std::vector<double> compute_search_direction(
        const std::vector<double>&, const std::vector<double>& grad) override {
        std::vector<double> p(ndim_);
        for(int i = 0; i < ndim_; i++) p[i] = -grad[i];
        return p;
    }

    void step_completed(const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&) override {}

    std::string name() const override { return "SD"; }
};


// =============================================================================
// GDIIS (Geometry Direct Inversion in the Iterative Subspace)
// Uses BFGS Hessian update + DIIS extrapolation instead of line search.
// =============================================================================

class GDIISOptimizer : public GeometryOptimizer {
public:
    GDIISOptimizer(int ndim, int max_subspace = 6)
        : GeometryOptimizer(ndim), H_inv_(ndim * ndim, 0.0), max_subspace_(max_subspace) {}

    void initialize(const std::vector<double>&, const std::vector<double>&) override {
        std::fill(H_inv_.begin(), H_inv_.end(), 0.0);
        for(int i = 0; i < ndim_; i++) H_inv_[i * ndim_ + i] = 1.0;
        history_.clear();
    }

    std::vector<double> compute_search_direction(
        const std::vector<double>& coords, const std::vector<double>& grad) override {

        // Compute error vector: e = -H_inv * grad (quasi-Newton step)
        std::vector<double> error(ndim_, 0.0);
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                error[i] -= H_inv_[i * ndim_ + j] * grad[j];

        // Add to history
        history_.push_back({coords, grad, error});
        if(static_cast<int>(history_.size()) > max_subspace_)
            history_.erase(history_.begin());

        int m = static_cast<int>(history_.size());

        if(m < 2) {
            // First step: just use quasi-Newton direction
            return error;
        }

        // Build DIIS matrix B_ij = e_i . e_j
        // Augmented system: [B, 1; 1^T, 0] * [c; -lambda] = [0; 1]
        int n = m + 1;
        std::vector<double> A(n * n, 0.0);
        std::vector<double> rhs(n, 0.0);

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < m; j++) {
                A[i * n + j] = dot(history_[i].error, history_[j].error);
            }
            A[i * n + m] = 1.0;
            A[m * n + i] = 1.0;
        }
        rhs[m] = 1.0;

        if(!solve_linear_system(A, rhs, n)) {
            // DIIS failed, fall back to quasi-Newton step
            return error;
        }

        // x_new = sum(c_i * (x_i + e_i))
        // displacement = x_new - x_current
        std::vector<double> displacement(ndim_, 0.0);
        for(int i = 0; i < m; i++) {
            double c = rhs[i];
            for(int j = 0; j < ndim_; j++)
                displacement[j] += c * (history_[i].coords[j] + history_[i].error[j]);
        }
        for(int j = 0; j < ndim_; j++)
            displacement[j] -= coords[j];

        return displacement;
    }

    void step_completed(const std::vector<double>& s, const std::vector<double>& y,
                       const std::vector<double>&) override {
        // BFGS update of internal inverse Hessian
        double sy = dot(s, y);
        if(sy <= 1.0e-10) return;

        double rho = 1.0 / sy;
        std::vector<double> Hy(ndim_, 0.0);
        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                Hy[i] += H_inv_[i * ndim_ + j] * y[j];
        double yHy = dot(y, Hy);

        for(int i = 0; i < ndim_; i++)
            for(int j = 0; j < ndim_; j++)
                H_inv_[i * ndim_ + j] += (sy + yHy) * rho * rho * s[i] * s[j]
                                       - rho * (Hy[i] * s[j] + s[i] * Hy[j]);
    }

    bool use_line_search() const override { return false; }

    std::string name() const override { return "GDIIS"; }

private:
    struct DiisEntry {
        std::vector<double> coords;
        std::vector<double> grad;
        std::vector<double> error;
    };

    std::vector<double> H_inv_;
    std::vector<DiisEntry> history_;
    int max_subspace_;

    // Gaussian elimination with partial pivoting
    static bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n) {
        for(int k = 0; k < n; k++) {
            int max_row = k;
            double max_val = std::abs(A[k * n + k]);
            for(int i = k + 1; i < n; i++) {
                if(std::abs(A[i * n + k]) > max_val) {
                    max_val = std::abs(A[i * n + k]);
                    max_row = i;
                }
            }
            if(max_val < 1.0e-14) return false;

            if(max_row != k) {
                for(int j = 0; j < n; j++) std::swap(A[k * n + j], A[max_row * n + j]);
                std::swap(b[k], b[max_row]);
            }

            for(int i = k + 1; i < n; i++) {
                double factor = A[i * n + k] / A[k * n + k];
                for(int j = k; j < n; j++) A[i * n + j] -= factor * A[k * n + j];
                b[i] -= factor * b[k];
            }
        }

        for(int i = n - 1; i >= 0; i--) {
            for(int j = i + 1; j < n; j++) b[i] -= A[i * n + j] * b[j];
            b[i] /= A[i * n + i];
        }
        return true;
    }
};


// =============================================================================
// Newton-Raphson: p = -H^{-1} g using exact analytical Hessian
// Requires a callback to compute the Hessian at each step.
// =============================================================================

class NewtonOptimizer : public GeometryOptimizer {
public:
    using HessianCallback = std::function<std::vector<double>()>;

    NewtonOptimizer(int ndim, HessianCallback hessian_fn)
        : GeometryOptimizer(ndim), hessian_fn_(std::move(hessian_fn)) {}

    void initialize(const std::vector<double>&, const std::vector<double>&) override {}

    std::vector<double> compute_search_direction(
        const std::vector<double>&, const std::vector<double>& grad) override {
        // Compute analytical Hessian
        auto hessian = hessian_fn_();

        // Solve H * p = -g via Gaussian elimination
        std::vector<double> H(hessian);
        std::vector<double> neg_g(ndim_);
        for (int i = 0; i < ndim_; i++) neg_g[i] = -grad[i];

        if (!solve_linear_system(H, neg_g, ndim_)) {
            // Hessian singular — fall back to steepest descent
            for (int i = 0; i < ndim_; i++) neg_g[i] = -grad[i];
        }
        return neg_g;
    }

    void step_completed(const std::vector<double>&, const std::vector<double>&,
                       const std::vector<double>&) override {}

    bool use_line_search() const override { return false; }

    std::string name() const override { return "Newton"; }

private:
    HessianCallback hessian_fn_;

    static bool solve_linear_system(std::vector<double>& A, std::vector<double>& b, int n) {
        for (int k = 0; k < n; k++) {
            int max_row = k;
            double max_val = std::abs(A[k * n + k]);
            for (int i = k + 1; i < n; i++) {
                if (std::abs(A[i * n + k]) > max_val) {
                    max_val = std::abs(A[i * n + k]);
                    max_row = i;
                }
            }
            if (max_val < 1.0e-14) return false;
            if (max_row != k) {
                for (int j = 0; j < n; j++) std::swap(A[k * n + j], A[max_row * n + j]);
                std::swap(b[k], b[max_row]);
            }
            for (int i = k + 1; i < n; i++) {
                double factor = A[i * n + k] / A[k * n + k];
                for (int j = k; j < n; j++) A[i * n + j] -= factor * A[k * n + j];
                b[i] -= factor * b[k];
            }
        }
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) b[i] -= A[i * n + j] * b[j];
            b[i] /= A[i * n + i];
        }
        return true;
    }
};


// =============================================================================
// Factory method
// =============================================================================

inline std::unique_ptr<GeometryOptimizer> GeometryOptimizer::create(
    const std::string& optimizer_name, int ndim) {

    if(optimizer_name == "bfgs")   return std::make_unique<BFGSOptimizer>(ndim);
    if(optimizer_name == "dfp")    return std::make_unique<DFPOptimizer>(ndim);
    if(optimizer_name == "sr1")    return std::make_unique<SR1Optimizer>(ndim);
    if(optimizer_name == "gdiis")  return std::make_unique<GDIISOptimizer>(ndim);
    if(optimizer_name == "cg-fr")  return std::make_unique<CGFletcherReeves>(ndim);
    if(optimizer_name == "cg-pr")  return std::make_unique<CGPolakRibiere>(ndim);
    if(optimizer_name == "cg-hs")  return std::make_unique<CGHestenesStifel>(ndim);
    if(optimizer_name == "cg-dy")  return std::make_unique<CGDaiYuan>(ndim);
    if(optimizer_name == "sd")     return std::make_unique<SteepestDescentOptimizer>(ndim);
    // "newton" is handled separately in hf.cu because it needs the HF object

    throw std::runtime_error("Unknown optimizer: " + optimizer_name
        + ". Valid options: bfgs, dfp, sr1, gdiis, cg-fr, cg-pr, cg-hs, cg-dy, sd, newton");
}

} // namespace gansu
