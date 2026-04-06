/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 * Hash ERI half-transformation kernels for MP2
 *
 * Replaces on-the-fly integral computation in Direct MP2 with Hash table lookup.
 * H(μ,ν,λ,j) = Σ_σ (μν|λσ) × C(σ,j)  where (μν|λσ) comes from Hash ERI.
 */

#include "gpu_hash_table.cuh"
#include "types.hpp"

namespace gansu {

// ============================================================
//  Core: process one AO ERI (μν|λσ) and accumulate into H
// ============================================================
__device__ inline void hash_half_transform_accumulate(
    int mu, int nu, int la, int si, double val,
    const double* __restrict__ g_C,
    double* __restrict__ g_half,
    int nao, int j_start, int block_j)
{
    const size_t nao2 = (size_t)nao * nao;
    const size_t nao2bj = nao2 * block_j;

    // H(μ,ν,λ,j) += (μν|λσ) × C(σ,j)  for j in [j_start, j_start+block_j)
    // 8-fold symmetry: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = ...
    // Each permutation gives a different (m,n,l,s) → different H entry.
    struct Perm { int m, n, l, s; };
    Perm perms[8] = {
        {mu,nu,la,si}, {nu,mu,la,si}, {mu,nu,si,la}, {nu,mu,si,la},
        {la,si,mu,nu}, {si,la,mu,nu}, {la,si,nu,mu}, {si,la,nu,mu}
    };

    // Deduplicate
    int n_unique = 0;
    Perm unique[8];
    for (int p = 0; p < 8; p++) {
        bool dup = false;
        for (int q = 0; q < n_unique; q++)
            if (perms[p].m == unique[q].m && perms[p].n == unique[q].n &&
                perms[p].l == unique[q].l && perms[p].s == unique[q].s) { dup = true; break; }
        if (!dup) unique[n_unique++] = perms[p];
    }

    for (int p = 0; p < n_unique; p++) {
        int m = unique[p].m, n = unique[p].n, l = unique[p].l, s = unique[p].s;
        // H(m, n, l, j) += val * C(s, j)  for j in block
        // H layout: H[m * nao2 * block_j + n * nao * block_j + l * block_j + (j - j_start)]
        //         = H[(m*nao + n) * nao * block_j + l * block_j + jj]
        // Actually: H(mu, nu*la*bj) row-major → H[mu * (nao*nao*bj) + nu*(nao*bj) + la*bj + jj]
        for (int jj = 0; jj < block_j; jj++) {
            double Csj = g_C[s * nao + (j_start + jj)];
            if (fabs(Csj) < 1e-15) continue;
            atomicAdd(&g_half[(size_t)m * nao2bj + (size_t)n * nao * block_j + (size_t)l * block_j + jj],
                      val * Csj);
        }
    }
}


// ============================================================
//  Compact (COO) version
// ============================================================
__global__ void hash_half_transform_compact_kernel(
    const unsigned long long* __restrict__ g_coo_keys,
    const double* __restrict__ g_coo_values,
    const size_t num_entries,
    const double* __restrict__ g_C,
    double* __restrict__ g_half,
    int nao, int j_start, int block_j)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    const unsigned long long key = g_coo_keys[tid];
    const double val = g_coo_values[tid];
    if (fabs(val) < 1e-18) return;

    int mu, nu, la, si;
    gpu::decode_eri_key(key, mu, nu, la, si);

    hash_half_transform_accumulate(mu, nu, la, si, val, g_C, g_half, nao, j_start, block_j);
}


// ============================================================
//  Indexed version
// ============================================================
__global__ void hash_half_transform_indexed_kernel(
    const unsigned long long* __restrict__ g_hash_keys,
    const double* __restrict__ g_hash_values,
    const size_t* __restrict__ g_nonzero_indices,
    const size_t num_nonzero,
    const double* __restrict__ g_C,
    double* __restrict__ g_half,
    int nao, int j_start, int block_j)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nonzero) return;

    const size_t slot = g_nonzero_indices[tid];
    const unsigned long long key = g_hash_keys[slot];
    const double val = g_hash_values[slot];
    if (key == 0xFFFFFFFFFFFFFFFFULL || fabs(val) < 1e-18) return;

    int mu, nu, la, si;
    gpu::decode_eri_key(key, mu, nu, la, si);

    hash_half_transform_accumulate(mu, nu, la, si, val, g_C, g_half, nao, j_start, block_j);
}


// ============================================================
//  Fullscan version
// ============================================================
__global__ void hash_half_transform_fullscan_kernel(
    const unsigned long long* __restrict__ g_hash_keys,
    const double* __restrict__ g_hash_values,
    const size_t hash_capacity,
    const double* __restrict__ g_C,
    double* __restrict__ g_half,
    int nao, int j_start, int block_j)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hash_capacity) return;

    const unsigned long long key = g_hash_keys[tid];
    if (key == 0xFFFFFFFFFFFFFFFFULL) return;
    const double val = g_hash_values[tid];
    if (fabs(val) < 1e-18) return;

    int mu, nu, la, si;
    gpu::decode_eri_key(key, mu, nu, la, si);

    hash_half_transform_accumulate(mu, nu, la, si, val, g_C, g_half, nao, j_start, block_j);
}

} // namespace gansu
