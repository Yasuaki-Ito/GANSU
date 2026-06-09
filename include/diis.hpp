#pragma once

#include <vector>

/**
 * @brief DIIS class for Direct Inversion in the Iterative Subspace
 * @details This class implements the DIIS algorithm for accelerating convergence in iterative methods.
 */
// How to use:
// 1. Create an instance of the DIIS class with desired history sizes.
//  e.g.,
//      DIIS diis(max_history_size=8, min_history_size=2);
//
// 2. In each iteration, after computing the new solution vector x and error vector e,
//    call diis.push(x, e);
//   e.g., 
//      diis.push(current_solution_vector, current_error_vector);
//
// 3. Before updating the solution vector, check if diis.can_extrapolate() returns true.
//    e.g., 
//      if(diis.can_extrapolate()) { ... }
//
//    This ensures that there is enough history to perform extrapolation.
//
// 4. If true, get the extrapolated solution vector by calling diis.extrapolate().
//    e.g.,
//      auto new_solution_vector = diis.extrapolate();
//    Use this new_solution_vector as the updated solution for the next iteration.

class DIIS {
public:
    DIIS(int max_history_size = 8, int min_history_size = 2)
        : max_hist(max_history_size), min_hist(min_history_size) {}
    ~DIIS() = default;

    // add new history of x and e
    void push(const std::vector<double>& x, const std::vector<double>& e) {
        xs.push_back(x);
        es.push_back(e);
        if ((int)xs.size() > max_hist) {
            xs.erase(xs.begin());
            es.erase(es.begin());
        }
    }

    // Clear all history. Used by callers that detect convergence pathologies
    // (residual rebound, singular B) and want to restart DIIS from a clean
    // subspace rather than silently fall back to plain Jacobi.
    void clear() {
        xs.clear();
        es.clear();
    }

    int history_size() const { return (int)xs.size(); }

    // dot product of two vectors
    static double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t i=0;i<a.size();++i) s += a[i]*b[i];
        return s;
    }

    // Gauss elimination to solve B c = rhs
    static std::vector<double> solve_diis_coeffs(const std::vector<std::vector<double>>& B) {
        // B: (n+1)x(n+1) matrix (enlarged B matrix)
        int n = (int)B.size();
        std::vector<std::vector<double>> A = B;
        std::vector<double> rhs(n, 0.0);
        rhs[n-1] = -1.0;

        // Solve by Gauss elimination A c = rhs
        for (int k=0;k<n;k++){
            int piv = k;
            for (int i=k+1;i<n;i++) if (std::abs(A[i][k]) > std::abs(A[piv][k])) piv = i;
            std::swap(A[k], A[piv]);
            std::swap(rhs[k], rhs[piv]);

            double diag = A[k][k];
            if (std::abs(diag) < 1e-14) throw std::runtime_error("DIIS singular B matrix");
            for (int j=k;j<n;j++) A[k][j] /= diag;
            rhs[k] /= diag;

            for (int i=0;i<n;i++){
                if (i==k) continue;
                double f = A[i][k];
                for (int j=k;j<n;j++) A[i][j] -= f*A[k][j];
                rhs[i] -= f*rhs[k];
            }
        }
        return rhs; // solution vector
    }

    bool can_extrapolate() const { return (int)xs.size() >= min_hist; }

    // return extrapolated x
    std::vector<double> extrapolate() const {
        int m = (int)xs.size();
        int dim = (int)xs[0].size();

        // enlarge B matrix. The error-vector Gram is symmetric, so compute
        // only the upper triangle and mirror, parallelising over the unique
        // (i,j) dots. Each dot keeps its serial summation order ⇒ the B matrix
        // (and hence the coefficients and extrapolated x) is bit-identical to
        // the serial build. dim is large (≈ Σ n_pno² for DLPNO CCSD), so this
        // O(m²·dim) build dominated the per-iter DIIS cost.
        std::vector<std::vector<double>> B(m+1, std::vector<double>(m+1, -1.0));
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int i=0;i<m;i++){
            for (int j=0;j<m;j++){
                if (j < i) continue;
                const double d = dot(es[i], es[j]);
                B[i][j] = d;
                B[j][i] = d;
            }
        }
        for (int i=0;i<m;i++){
            B[i][i] += 1e-12; // to avoid singularity
            B[i][m] = -1.0;
            B[m][i] = -1.0;
        }
        B[m][m] = 0.0;

        auto c_full = solve_diis_coeffs(B); // the size is m+1
        // cut off the last element
        std::vector<double> c(m);
        for (int i=0;i<m;i++) c[i] = c_full[i];

        // x = Σ c_i x_i. Each output element is an independent reduction over
        // i in the same order as the serial loop ⇒ bit-identical.
        std::vector<double> x(dim, 0.0);
        #pragma omp parallel for schedule(static)
        for (int k=0;k<dim;k++){
            double acc = 0.0;
            for (int i=0;i<m;i++) acc += c[i]*xs[i][k];
            x[k] = acc;
        }
        return x;
    }

private:
    int max_hist; ///< maximum history size
    int min_hist; ///< minimum history size to start DIIS
    std::vector<std::vector<double>> xs; ///< history of x vectors
    std::vector<std::vector<double>> es; ///< history of error vectors
};
