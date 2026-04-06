/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * CIS operator using OVOV/OOVV sub-blocks from half-transformation.
 * No full MO ERI (nao⁴) needed.
 *
 * Singlet: σ(ia) = (ε_a - ε_i) r(ia) + Σ_{jb} [2(ia|jb) - (ij|ab)] r(jb)
 * Triplet: σ(ia) = (ε_a - ε_i) r(ia) - Σ_{jb} (ij|ab) r(jb)
 */

#include "cis_operator_jk.hpp"
#include "device_host_memory.hpp"

namespace gansu {

__global__ void cis_ht_diagonal_kernel(
    const real_t* __restrict__ d_eps, real_t* __restrict__ d_diag,
    int nocc, int nvir)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nocc * nvir) return;
    int i = idx / nvir;
    int a = idx % nvir;
    d_diag[idx] = d_eps[nocc + a] - d_eps[i];
}

__global__ void cis_ht_precond_kernel(
    const real_t* __restrict__ d_diag,
    const real_t* __restrict__ d_input,
    real_t* __restrict__ d_output,
    int dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    real_t dval = d_diag[idx];
    d_output[idx] = (fabs(dval) > 1e-12) ? d_input[idx] / dval : 0.0;
}

/**
 * CIS sigma vector kernel using OVOV and OOVV sub-blocks.
 *
 * Singlet: σ(ia) = diag(ia)*r(ia) + Σ_{jb} [2*(ia|jb) - (ij|ab)] r(jb)
 * Triplet: σ(ia) = diag(ia)*r(ia) - Σ_{jb} (ij|ab) r(jb)
 *
 * ovov layout: [i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]  (a,b 0-based in vir)
 * oovv layout: [i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b]  (a,b 0-based in vir)
 * r layout:    [j*nvir + b]
 */
__global__ void cis_ht_sigma_kernel(
    const real_t* __restrict__ d_ovov,
    const real_t* __restrict__ d_oovv,
    const real_t* __restrict__ d_diag,
    const real_t* __restrict__ d_r,
    real_t* __restrict__ d_sigma,
    int nocc, int nvir, bool is_triplet)
{
    const int dim = nocc * nvir;
    const int ia = blockIdx.x * blockDim.x + threadIdx.x;
    if (ia >= dim) return;

    const int i = ia / nvir;
    const int a = ia % nvir;

    real_t sigma_val = d_diag[ia] * d_r[ia];

    for (int j = 0; j < nocc; j++) {
        for (int b = 0; b < nvir; b++) {
            const int jb = j * nvir + b;
            const real_t r_jb = d_r[jb];

            // Exchange: (ij|ab)
            const real_t oovv_ijab = d_oovv[(size_t)i * nocc * nvir * nvir
                                          + (size_t)j * nvir * nvir
                                          + (size_t)a * nvir + b];
            sigma_val -= oovv_ijab * r_jb;

            if (!is_triplet) {
                // Coulomb: 2(ia|jb)
                const real_t ovov_iajb = d_ovov[(size_t)i * nvir * nocc * nvir
                                              + (size_t)a * nocc * nvir
                                              + (size_t)j * nvir + b];
                sigma_val += 2.0 * ovov_iajb * r_jb;
            }
        }
    }

    d_sigma[ia] = sigma_val;
}


// ========================================================================

CISOperator_HalfTransform::CISOperator_HalfTransform(
    const real_t* d_ovov, const real_t* d_oovv,
    const real_t* d_orbital_energies,
    int nocc, int nvir, bool is_triplet)
    : d_ovov_(d_ovov), d_oovv_(d_oovv),
      nocc_(nocc), nvir_(nvir), dim_(nocc * nvir),
      is_triplet_(is_triplet),
      d_diagonal_(nullptr), d_work_(nullptr)
{
    tracked_cudaMalloc(&d_diagonal_, dim_ * sizeof(real_t));
    tracked_cudaMalloc(&d_work_, dim_ * sizeof(real_t));

    build_diagonal(d_orbital_energies);
}

CISOperator_HalfTransform::~CISOperator_HalfTransform() {
    if (d_diagonal_) tracked_cudaFree(d_diagonal_);
    if (d_work_) tracked_cudaFree(d_work_);
}

void CISOperator_HalfTransform::build_diagonal(const real_t* d_orbital_energies) {
    int threads = 256, blocks = (dim_ + threads - 1) / threads;
    cis_ht_diagonal_kernel<<<blocks, threads>>>(d_orbital_energies, d_diagonal_, nocc_, nvir_);
    cudaDeviceSynchronize();
}

void CISOperator_HalfTransform::apply(const real_t* d_input, real_t* d_output) const {
    int threads = 256, blocks = (dim_ + threads - 1) / threads;
    cis_ht_sigma_kernel<<<blocks, threads>>>(
        d_ovov_, d_oovv_, d_diagonal_, d_input, d_output,
        nocc_, nvir_, is_triplet_);
    cudaDeviceSynchronize();
}

void CISOperator_HalfTransform::apply_preconditioner(const real_t* d_input, real_t* d_output) const {
    int threads = 256, blocks = (dim_ + threads - 1) / threads;
    cis_ht_precond_kernel<<<blocks, threads>>>(d_diagonal_, d_input, d_output, dim_);
}

} // namespace gansu
