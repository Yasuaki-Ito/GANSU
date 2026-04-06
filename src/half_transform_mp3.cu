/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 * Generalized half-transformation for MP3
 *
 * Builds any MO ERI block (pq|rs) from AO ERI via:
 *   Step 1: H(μ,ν,λ,s) = Σ_σ (μν|λσ) × C_s(σ,s)     ← on-the-fly or Hash
 *   Step 2: K(p,ν,λ,s) = Σ_μ C_p(μ,p) × H(μ,ν,λ,s)  ← DGEMM
 *   Step 3: L(p,q,λ,s) = Σ_ν C_q(ν,q) × K(p,ν,λ,s)  ← DGEMM
 *   Step 4: V(p,q,r,s) = Σ_λ C_r(λ,r) × L(p,q,λ,s)  ← DGEMM
 *
 * C_p, C_q, C_r, C_s can be C_occ or C_vir (or any subblock).
 * n_p, n_q, n_r, n_s are their respective dimensions.
 */

#include "rhf.hpp"
#include "device_host_memory.hpp"
#include "gpu_manager.hpp"

namespace gansu {

// Forward declarations for Hash half-transform kernels
__global__ void hash_half_transform_compact_kernel(
    const unsigned long long* g_coo_keys, const double* g_coo_values, size_t num_entries,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);
__global__ void hash_half_transform_indexed_kernel(
    const unsigned long long* g_hash_keys, const double* g_hash_values,
    const size_t* g_nonzero_indices, size_t num_nonzero,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);
__global__ void hash_half_transform_fullscan_kernel(
    const unsigned long long* g_hash_keys, const double* g_hash_values, size_t hash_capacity,
    const double* g_C, double* g_half, int nao, int j_start, int block_j);

/**
 * Build a general MO ERI block using half-transformation.
 *
 * @param d_result Output: (n_p, n_q, n_r, n_s) tensor, row-major
 *                 Layout: result[p * n_q*n_r*n_s + q * n_r*n_s + r * n_s + s]
 *                 The 4th index (s) is the one contracted in Step 1.
 * @param d_C     Full MO coefficient matrix [nao × nao], row-major
 * @param p_start, n_p  Range for index p in C  (Step 2 contraction)
 * @param q_start, n_q  Range for index q in C  (Step 3 contraction)
 * @param r_start, n_r  Range for index r in C  (Step 4 contraction)
 * @param s_start, n_s  Range for index s in C  (Step 1 contraction)
 * @param nao     Total number of AOs
 * @param block_s Block size for s-index loop
 *
 * Step 1 source: one of Direct (on-the-fly), Hash, or pre-computed.
 * This function handles Steps 2-4 DGEMM chain. The caller provides Step 1 (d_half).
 */
void half_transform_steps234(
    const real_t* d_half,  // H(μ, ν*λ*bs) from Step 1, size nao * nao * nao * bs
    const real_t* d_C,     // Full C matrix [nao × nao]
    real_t* d_result,      // Output block, size n_p * n_q * n_r * bs
    real_t* d_Ki,          // Workspace: nao * nao * bs
    real_t* d_Li,          // Workspace: n_q * nao * bs
    int nao, int bs,       // bs = actual block size for this iteration
    int p_start, int n_p,
    int q_start, int n_q,
    int r_start, int n_r)
{
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta_zero = 0.0;

    const size_t M_nao2bs = (size_t)nao * nao * bs;
    const size_t M_naobs = (size_t)nao * bs;

    for (int p = 0; p < n_p; p++) {
        // Step 2: K_p(ν, λ*bs) = Σ_μ C(μ, p+p_start) × H(μ, ν*λ*bs)
        cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            M_nao2bs, 1, nao,
            &alpha,
            d_half, M_nao2bs,
            d_C + (p_start + p), nao,
            &beta_zero,
            d_Ki, M_nao2bs);

        // Step 3: L_p(q, λ*bs) = Σ_ν C(ν, q+q_start) × K_p(ν, λ*bs)
        cublasDgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            M_naobs, n_q, nao,
            &alpha,
            d_Ki, M_naobs,
            d_C + q_start, nao,
            &beta_zero,
            d_Li, M_naobs);

        // Step 4: V_p(q, r, bs) = Σ_λ C(λ, r+r_start) × L_p(q, λ*bs)
        for (int q = 0; q < n_q; q++) {
            cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                bs, n_r, nao,
                &alpha,
                d_Li + (size_t)q * nao * bs, bs,
                d_C + r_start, nao,
                &beta_zero,
                d_result + (size_t)p * n_q * n_r * bs + (size_t)q * n_r * bs, bs);
        }
    }
}

/**
 * Complete generalized half-transformation for Direct SCF (on-the-fly AO ERI).
 * Builds MO ERI block (pq|rs) where p,q,r indices are arbitrary MO ranges.
 * The s-index is looped in blocks, with AO ERI half-transformed on-the-fly.
 */
void build_mo_block_direct(
    real_t* d_result,   // Output: (n_p, n_q, n_r, n_s) row-major
    const HF& hf,
    const real_t* d_C, const real_t* d_eps,
    int nao, int p_start, int n_p, int q_start, int n_q,
    int r_start, int n_r, int s_start, int n_s,
    int block_s,
    const real_t* d_schwarz)
{
    const size_t nao3 = (size_t)nao * nao * nao;
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;

    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)n_q * nao * block_s * sizeof(real_t));

    for (int s_blk = 0; s_blk < n_s; s_blk += block_s) {
        int bs = std::min(block_s, n_s - s_blk);
        int s_abs = s_start + s_blk;

        // Step 1: H(μ,ν,λ,s) = Σ_σ (μν|λσ) × C(σ, s_abs+ss) — on-the-fly
        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        gpu::computeHalfTransformedERI(
            hf.get_shell_type_infos(), hf.get_shell_pair_type_infos(),
            hf.get_primitive_shells().device_ptr(),
            hf.get_boys_grid().device_ptr(),
            hf.get_cgto_normalization_factors().device_ptr(),
            d_half, d_schwarz,
            hf.get_schwarz_screening_threshold(),
            nao, d_C, s_abs, bs);

        // Steps 2-4
        half_transform_steps234(
            d_half, d_C,
            d_result + (size_t)s_blk,  // offset into the s-dimension
            d_Ki, d_Li,
            nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);
    }

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
}

/**
 * Complete generalized half-transformation for Hash ERI.
 * Same as Direct but Step 1 reads from Hash table.
 */
void build_mo_block_hash(
    real_t* d_result,
    const real_t* d_C,
    int nao, int p_start, int n_p, int q_start, int n_q,
    int r_start, int n_r, int s_start, int n_s,
    int block_s,
    // Hash data
    const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    const size_t* d_nonzero_indices, size_t num_nonzero,
    size_t hash_capacity_mask,
    HashFockMethod method)
{
    const size_t nao3 = (size_t)nao * nao * nao;
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;

    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)n_q * nao * block_s * sizeof(real_t));

    for (int s_blk = 0; s_blk < n_s; s_blk += block_s) {
        int bs = std::min(block_s, n_s - s_blk);
        int s_abs = s_start + s_blk;

        // Step 1: H from Hash
        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        {
            const int threads = 256;
            if (method == HashFockMethod::Compact) {
                const int blocks = ((int)num_entries + threads - 1) / threads;
                hash_half_transform_compact_kernel<<<blocks, threads>>>(
                    d_coo_keys, d_coo_values, num_entries,
                    d_C, d_half, nao, s_abs, bs);
            } else if (method == HashFockMethod::Indexed) {
                const int blocks = ((int)num_nonzero + threads - 1) / threads;
                hash_half_transform_indexed_kernel<<<blocks, threads>>>(
                    d_hash_keys, d_hash_values,
                    d_nonzero_indices, num_nonzero,
                    d_C, d_half, nao, s_abs, bs);
            } else {
                const size_t capacity = hash_capacity_mask + 1;
                const int blocks = ((int)capacity + threads - 1) / threads;
                hash_half_transform_fullscan_kernel<<<blocks, threads>>>(
                    d_hash_keys, d_hash_values, capacity,
                    d_C, d_half, nao, s_abs, bs);
            }
            cudaDeviceSynchronize();
        }

        // Steps 2-4
        half_transform_steps234(
            d_half, d_C,
            d_result + (size_t)s_blk,
            d_Ki, d_Li,
            nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);
    }

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
}

} // namespace gansu
