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

// ============================================================
//  Kernels for half-transform-based MP3
// ============================================================

/**
 * Scatter s-block from temporary buffer to correct positions in full 4D tensor.
 * src layout: src[p * n_q*n_r*bs + q * n_r*bs + r * bs + s_local]
 * dst layout: dst[p * n_q*n_r*n_s + q * n_r*n_s + r * n_s + (s_blk + s_local)]
 */
__global__ void scatter_s_block_kernel(
    const double* __restrict__ src, double* __restrict__ dst,
    int n_p, int n_q, int n_r, int n_s, int s_blk, int bs)
{
    const size_t total = (size_t)n_p * n_q * n_r * bs;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int s_local = t % bs; t /= bs;
    const int r = t % n_r; t /= n_r;
    const int q = t % n_q; t /= n_q;
    const int p = (int)t;

    const size_t dst_idx = (size_t)p * n_q * n_r * n_s + (size_t)q * n_r * n_s + (size_t)r * n_s + (s_blk + s_local);
    dst[dst_idx] = src[gid];
}

/**
 * Permute 4D tensor: out[i,j,k,l] = in[i,k,j,l]  (swap indices 1↔2)
 * Used to convert (ij|kl) → oooo[i,j,k,l] = (ik|jl)
 *             and (ab|cd) → vvvv[a,b,c,d] = (ac|bd)
 */
__global__ void permute_swap12_kernel(
    const double* __restrict__ in, double* __restrict__ out,
    int n0, int n1, int n2, int n3)
{
    const size_t total = (size_t)n0 * n1 * n2 * n3;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    // Decode gid → (i, j, k, l) in output layout
    size_t t = gid;
    const int l = t % n3; t /= n3;
    const int k = t % n2; t /= n2;
    const int j = t % n1; t /= n1;
    const int i = (int)t;

    // out[i,j,k,l] = in[i,k,j,l]  (swap j↔k, i.e. indices 1↔2)
    const size_t in_idx = (size_t)i * n2 * n1 * n3 + (size_t)k * n1 * n3 + (size_t)j * n3 + l;
    out[gid] = in[in_idx];
}

/**
 * Build s_ovov and t_ovov from ovov block V(i,a,j,b) = (ia|jb)
 * ovov layout: [i * nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]  (a,b are 0-based in vir)
 *
 * s_ovov[i,a,j,b] = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)
 * t_ovov[i,a,j,b] = (2*(ia|jb) - (ib|ja)) / (eps_i + eps_j - eps_a - eps_b)
 */
__global__ void build_st_ovov_from_block(
    const double* __restrict__ V_ovov,   // (ia|jb) size nocc*nvir*nocc*nvir
    const double* __restrict__ eps,
    double* __restrict__ s_ovov,
    double* __restrict__ t_ovov,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int b = t % nvir; t /= nvir;
    const int j = t % nocc; t /= nocc;
    const int a = t % nvir; t /= nvir;
    const int i = (int)t;

    const double eps_ijab = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
    const double iajb = V_ovov[gid];
    // (ib|ja) = V_ovov[i * nvir*nocc*nvir + b * nocc*nvir + j * nvir + a]
    const double ibja = V_ovov[(size_t)i * nvir * nocc * nvir + (size_t)b * nocc * nvir + (size_t)j * nvir + a];

    s_ovov[gid] = iajb / eps_ijab;
    t_ovov[gid] = (2.0 * iajb - ibja) / eps_ijab;
}

/**
 * Compute MP2 energy from ovov block V(i,a,j,b) = (ia|jb)
 * E_MP2 = Σ_{iajb} (ia|jb) * (2*(ia|jb) - (ib|ja)) / (eps_i + eps_j - eps_a - eps_b)
 */
__global__ void mp2_from_ovov_block_kernel(
    const double* __restrict__ V_ovov,
    const double* __restrict__ eps,
    int nocc, int nvir, double* __restrict__ E_out)
{
    __shared__ double s_E;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_E = 0.0;
    __syncthreads();

    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (gid < total) {
        size_t t = gid;
        const int b = t % nvir; t /= nvir;
        const int j = t % nocc; t /= nocc;
        const int a = t % nvir; t /= nvir;
        const int i = (int)t;

        const double denom = eps[i] + eps[j] - eps[nocc + a] - eps[nocc + b];
        const double iajb = V_ovov[gid];
        const double ibja = V_ovov[(size_t)i * nvir * nocc * nvir + (size_t)b * nocc * nvir + (size_t)j * nvir + a];
        val = iajb * (2.0 * iajb - ibja) / denom;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if (threadIdx.x == 0) atomicAdd(&s_E, val);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(E_out, s_E);
}

/**
 * Contract tensors for MP3 energy using sub-blocks instead of full MO ERI.
 *
 * Replaces contract_iajb_tensors which reads from g_int2e (full nao^4).
 * Instead reads (ia|jb) from V_ovov and (ij|ab) from V_oovv.
 *
 * V_ovov layout: [i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]  (a,b 0-based in vir)
 * V_oovv layout: [i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b]  (a,b 0-based in vir)
 */
__global__ void contract_iajb_tensors_blocks(
    int nocc, int nvir,
    const double* __restrict__ V_ovov,  // (ia|jb), size nocc*nvir*nocc*nvir
    const double* __restrict__ V_oovv,  // (ij|ab), size nocc*nocc*nvir*nvir
    const double* __restrict__ s_ovov,  // ovov layout
    const double* __restrict__ mm1,     // oovv layout
    const double* __restrict__ mm2,     // vvoo layout
    const double* __restrict__ mm3,     // ovov layout
    const double* __restrict__ mm4,     // ovov layout
    double* __restrict__ E_3rd)
{
    __shared__ double s_E_3rd;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_E_3rd = 0.0;
    __syncthreads();

    const long long total = (long long)nocc * nocc * nvir * nvir;
    const long long ijab = (long long)(blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x)
                         + (blockDim.x * threadIdx.y + threadIdx.x);
    if (ijab >= total) return;

    const int ij = ijab / (nvir * nvir);
    const int ab = ijab % (nvir * nvir);
    const int i = ij / nocc;
    const int j = ij % nocc;
    const int a = ab / nvir;  // 0-based in vir
    const int b = ab % nvir;  // 0-based in vir

    // ovov index: i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b
    const size_t idx_ovov_iajb = (size_t)i * nvir * nocc * nvir + (size_t)a * nocc * nvir + (size_t)j * nvir + b;
    const double s_iajb = s_ovov[idx_ovov_iajb];
    const double e_iajb = V_ovov[idx_ovov_iajb]; // (ia|jb)

    // oovv index: i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
    const size_t idx_oovv_ijab = (size_t)i * nocc * nvir * nvir + (size_t)j * nvir * nvir + (size_t)a * nvir + b;
    const double e_ijab = V_oovv[idx_oovv_ijab]; // (ij|ab)

    // vvoo index: a*nvir*nocc*nocc + b*nocc*nocc + i*nocc + j  (a,b 0-based)
    const size_t idx_vvoo_abij = (size_t)a * nvir * nocc * nocc + (size_t)b * nocc * nocc + (size_t)i * nocc + j;

    double energy = 0.0;
    energy += s_iajb * (mm1[idx_oovv_ijab] + mm2[idx_vvoo_abij]);
    energy += (2.0 * e_iajb - e_ijab) * mm3[idx_ovov_iajb];
    energy += (-3.0) * e_ijab * mm4[idx_ovov_iajb];

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        energy += __shfl_down_sync(0xFFFFFFFF, energy, offset);

    if (threadIdx.x == 0) atomicAdd(&s_E_3rd, energy);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(E_3rd, s_E_3rd);
}

/**
 * kalb2klab for 0-based vir indices: ovov → oovv
 * Input ovov: [k*nvir*nocc*nvir + a*nocc*nvir + l*nvir + b]
 * Output oovv: [k*nocc*nvir*nvir + l*nvir*nvir + a*nvir + b]
 */
__global__ void kalb2klab_block(
    const double* __restrict__ t_ovov, double* __restrict__ t_oovv,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int b = t % nvir; t /= nvir;
    const int l = t % nocc; t /= nocc;
    const int a = t % nvir; t /= nvir;
    const int k = (int)t;

    t_oovv[(size_t)k * nocc * nvir * nvir + (size_t)l * nvir * nvir + (size_t)a * nvir + b]
        = t_ovov[(size_t)k * nvir * nocc * nvir + (size_t)a * nocc * nvir + (size_t)l * nvir + b];
}

/**
 * icjd2cdij for 0-based vir indices: ovov → vvoo
 * Input ovov: [i*nvir*nocc*nvir + c*nocc*nvir + j*nvir + d]
 * Output vvoo: [c*nvir*nocc*nocc + d*nocc*nocc + i*nocc + j]
 */
__global__ void icjd2cdij_block(
    const double* __restrict__ t_ovov, double* __restrict__ t_vvoo,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int d = t % nvir; t /= nvir;
    const int j = t % nocc; t /= nocc;
    const int c = t % nvir; t /= nvir;
    const int i = (int)t;

    t_vvoo[(size_t)c * nvir * nocc * nocc + (size_t)d * nocc * nocc + (size_t)i * nocc + j]
        = t_ovov[(size_t)i * nvir * nocc * nvir + (size_t)c * nocc * nvir + (size_t)j * nvir + d];
}

/**
 * kaic2iakc for 0-based vir indices: ovov → ovov (permutation)
 * Input ovov: [k*nvir*nocc*nvir + a*nocc*nvir + i*nvir + c]
 * Output ovov: [i*nvir*nocc*nvir + a*nocc*nvir + k*nvir + c]
 */
__global__ void kaic2iakc_block(
    const double* __restrict__ in, double* __restrict__ out,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int c = t % nvir; t /= nvir;
    const int i = t % nocc; t /= nocc;
    const int a = t % nvir; t /= nvir;
    const int k = (int)t;

    out[(size_t)i * nvir * nocc * nvir + (size_t)a * nocc * nvir + (size_t)k * nvir + c]
        = in[gid];
}

/**
 * kbjc2kcjb for 0-based vir indices: ovov → ovov (permutation)
 * Input ovov: [k*nvir*nocc*nvir + b*nocc*nvir + j*nvir + c]
 * Output ovov: [k*nvir*nocc*nvir + c*nocc*nvir + j*nvir + b]
 */
__global__ void kbjc2kcjb_block(
    const double* __restrict__ in, double* __restrict__ out,
    int nocc, int nvir)
{
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    const size_t gid = (size_t)blockIdx.x * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    if (gid >= total) return;

    size_t t = gid;
    const int c = t % nvir; t /= nvir;
    const int j = t % nocc; t /= nocc;
    const int b = t % nvir; t /= nvir;
    const int k = (int)t;

    out[(size_t)k * nvir * nocc * nvir + (size_t)c * nocc * nvir + (size_t)j * nvir + b]
        = in[gid];
}


// ============================================================
//  MP3 energy via half-transformation (avoids full nao^4 MO ERI)
// ============================================================

/**
 * Build 4 MO ERI sub-blocks via half-transformation and compute MP3 energy.
 * Sub-blocks: oooo, vvvv, ovov, oovv.
 *
 * Phase 1: s ∈ occ → build oooo
 * Phase 2: s ∈ vir → build ovov, oovv, vvvv (sharing Step 1 half-transform)
 *
 * @param step1_func  Callable: step1_func(d_half, nao, s_abs, bs) performs Step 1
 */
template <typename Step1Func>
real_t mp3_half_transform_impl(
    RHF& rhf, Step1Func step1_func, int block_s)
{
    const int nao = rhf.get_num_basis();
    const int nocc = rhf.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf.get_orbital_energies().device_ptr();

    std::cout << "  [Half-Transform MP3] nao=" << nao << " nocc=" << nocc << " nvir=" << nvir
              << " block_s=" << block_s << std::endl;

    const size_t nao3 = (size_t)nao * nao * nao;
    const size_t num_oooo = (size_t)nocc * nocc * nocc * nocc;
    const size_t num_vvvv = (size_t)nvir * nvir * nvir * nvir;
    const size_t num_ovov = (size_t)nocc * nvir * nocc * nvir;
    const size_t num_oovv = (size_t)nocc * nocc * nvir * nvir;

    // --- Allocate sub-blocks ---
    real_t* d_oooo = nullptr;   // (ij|kl) then permuted to (ik|jl)
    real_t* d_vvvv = nullptr;   // (ab|cd) then permuted to (ac|bd)
    real_t* d_ovov = nullptr;   // (ia|jb) direct
    real_t* d_oovv = nullptr;   // (ij|ab) direct

    tracked_cudaMalloc(&d_oooo, num_oooo * sizeof(real_t));
    tracked_cudaMalloc(&d_vvvv, num_vvvv * sizeof(real_t));
    tracked_cudaMalloc(&d_ovov, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_oovv, num_oovv * sizeof(real_t));
    cudaMemset(d_oooo, 0, num_oooo * sizeof(real_t));
    cudaMemset(d_vvvv, 0, num_vvvv * sizeof(real_t));
    cudaMemset(d_ovov, 0, num_ovov * sizeof(real_t));
    cudaMemset(d_oovv, 0, num_oovv * sizeof(real_t));

    // --- Workspace for half-transform ---
    // Li needs max(n_q) * nao * block_s; max n_q = nvir
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;
    real_t* d_tmp_block = nullptr;  // Temporary buffer for s-block output
    tracked_cudaMalloc(&d_half, nao3 * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_s * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)std::max(nocc, nvir) * nao * block_s * sizeof(real_t));
    // Largest sub-block per s-block: oooo(nocc³) can be larger than vvvv(nvir³) when nocc > nvir
    const size_t max_block_elems = std::max({
        (size_t)nocc * nocc * nocc * block_s,   // oooo (Phase 1)
        (size_t)nocc * nvir * nocc * block_s,    // ovov (Phase 2)
        (size_t)nocc * nocc * nvir * block_s,    // oovv (Phase 2)
        (size_t)nvir * nvir * nvir * block_s     // vvvv (Phase 2)
    });
    tracked_cudaMalloc(&d_tmp_block, max_block_elems * sizeof(real_t));

    constexpr int scatter_threads = 256;

    // Helper lambda: Steps 2-4 into temp buffer, then scatter to output
    auto build_and_scatter = [&](real_t* d_dst, int n_p, int n_q, int n_r, int n_s,
                                  int s_blk, int bs,
                                  int p_start, int q_start, int r_start) {
        half_transform_steps234(d_half, d_C, d_tmp_block,
            d_Ki, d_Li, nao, bs, p_start, n_p, q_start, n_q, r_start, n_r);

        const size_t block_total = (size_t)n_p * n_q * n_r * bs;
        const size_t scatter_blocks = (block_total + scatter_threads - 1) / scatter_threads;
        scatter_s_block_kernel<<<scatter_blocks, scatter_threads>>>(
            d_tmp_block, d_dst, n_p, n_q, n_r, n_s, s_blk, bs);
    };

    // ============================================================
    //  Phase 1: s ∈ occ → build oooo = (ij|kl)
    // ============================================================
    std::cout << "  [Phase 1] Building oooo block..." << std::flush;
    for (int s_blk = 0; s_blk < nocc; s_blk += block_s) {
        int bs = std::min(block_s, nocc - s_blk);
        int s_abs = s_blk;  // s_start = 0 for occ

        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        step1_func(d_half, nao, d_C, s_abs, bs);

        build_and_scatter(d_oooo, nocc, nocc, nocc, nocc, s_blk, bs, 0, 0, 0);
    }
    std::cout << " done" << std::endl;

    // ============================================================
    //  Phase 2: s ∈ vir → build ovov, oovv, vvvv
    //  Share Step 1 half-transform H(μ,ν,λ,s) across 3 blocks
    // ============================================================
    std::cout << "  [Phase 2] Building ovov, oovv, vvvv blocks..." << std::flush;
    for (int s_blk = 0; s_blk < nvir; s_blk += block_s) {
        int bs = std::min(block_s, nvir - s_blk);
        int s_abs = nocc + s_blk;  // s_start = nocc for vir

        cudaMemset(d_half, 0, nao3 * bs * sizeof(real_t));
        step1_func(d_half, nao, d_C, s_abs, bs);

        // ovov: p∈occ(nocc), q∈vir(nvir), r∈occ(nocc)  → V(i, a, j, b) = (ia|jb)
        build_and_scatter(d_ovov, nocc, nvir, nocc, nvir, s_blk, bs, 0, nocc, 0);

        // oovv: p∈occ(nocc), q∈occ(nocc), r∈vir(nvir)  → V(i, j, a, b) = (ij|ab)
        build_and_scatter(d_oovv, nocc, nocc, nvir, nvir, s_blk, bs, 0, 0, nocc);

        // vvvv: p∈vir(nvir), q∈vir(nvir), r∈vir(nvir)  → V(a, b, c, d) = (ab|cd)
        build_and_scatter(d_vvvv, nvir, nvir, nvir, nvir, s_blk, bs, nocc, nocc, nocc);
    }
    std::cout << " done" << std::endl;

    // --- Free half-transform workspace ---
    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_tmp_block);

    // ============================================================
    //  Permute oooo and vvvv: swap indices 1↔2
    //  oooo: V(i,j,k,l)=(ij|kl) → perm[i,j,k,l]=(ik|jl)
    //  vvvv: V(a,b,c,d)=(ab|cd) → perm[a,b,c,d]=(ac|bd)
    // ============================================================
    {
        real_t* d_tmp = nullptr;
        const size_t max_size = std::max(num_oooo, num_vvvv);
        tracked_cudaMalloc(&d_tmp, max_size * sizeof(real_t));

        int threads = 256;
        // Permute oooo
        {
            size_t blocks = (num_oooo + threads - 1) / threads;
            cudaMemcpy(d_tmp, d_oooo, num_oooo * sizeof(real_t), cudaMemcpyDeviceToDevice);
            permute_swap12_kernel<<<blocks, threads>>>(d_tmp, d_oooo, nocc, nocc, nocc, nocc);
        }
        // Permute vvvv
        {
            size_t blocks = (num_vvvv + threads - 1) / threads;
            cudaMemcpy(d_tmp, d_vvvv, num_vvvv * sizeof(real_t), cudaMemcpyDeviceToDevice);
            permute_swap12_kernel<<<blocks, threads>>>(d_tmp, d_vvvv, nvir, nvir, nvir, nvir);
        }
        tracked_cudaFree(d_tmp);
    }
    cudaDeviceSynchronize();

    // ============================================================
    //  Compute MP2 energy from ovov block
    // ============================================================
    real_t* d_E_mp2 = nullptr;
    tracked_cudaMalloc(&d_E_mp2, sizeof(real_t));
    cudaMemset(d_E_mp2, 0, sizeof(real_t));

    mp2_from_ovov_block_kernel<<<(num_ovov + 1023) / 1024, dim3(32, 32)>>>(
        d_ovov, d_eps, nocc, nvir, d_E_mp2);
    cudaDeviceSynchronize();

    real_t h_E_mp2 = 0.0;
    cudaMemcpy(&h_E_mp2, d_E_mp2, sizeof(real_t), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_E_mp2);
    std::cout << "  [Half-Transform MP3] MP2 energy: " << std::setprecision(12) << h_E_mp2 << std::endl;

    // ============================================================
    //  Build s_ovov and t_ovov from ovov block
    // ============================================================
    real_t* d_s_ovov = nullptr;
    real_t* d_t_ovov = nullptr;
    real_t* d_t_tmp  = nullptr;
    tracked_cudaMalloc(&d_s_ovov, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_t_ovov, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_t_tmp,  num_ovov * sizeof(real_t));

    {
        int threads = 256;
        size_t blocks = (num_ovov + threads - 1) / threads;
        build_st_ovov_from_block<<<blocks, threads>>>(d_ovov, d_eps, d_s_ovov, d_t_ovov, nocc, nvir);
        cudaDeviceSynchronize();
    }

    // ============================================================
    //  DGEMM contractions (same as existing MP3)
    // ============================================================
    real_t* d_mm1 = nullptr;  // oovv layout
    real_t* d_mm2 = nullptr;  // vvoo layout
    real_t* d_mm3 = nullptr;  // ovov layout
    real_t* d_mm4 = nullptr;  // ovov layout
    tracked_cudaMalloc(&d_mm1, num_oovv * sizeof(real_t));
    tracked_cudaMalloc(&d_mm2, num_oovv * sizeof(real_t));
    tracked_cudaMalloc(&d_mm3, num_ovov * sizeof(real_t));
    tracked_cudaMalloc(&d_mm4, num_ovov * sizeof(real_t));

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta_zero = 0.0;
    const int num_oo = nocc * nocc;
    const int num_vv = nvir * nvir;
    const int num_ov = nocc * nvir;

    constexpr int num_threads_per_block = 32 * 32;
    const size_t num_blocks_ovov = (num_ovov + num_threads_per_block - 1) / num_threads_per_block;
    dim3 threads_2d(32, 32);

    // mm1 = oooo * t_oovv (t_ovov permuted to oovv via kalb2klab)
    kalb2klab_block<<<num_blocks_ovov, threads_2d>>>(d_t_ovov, d_t_tmp, nocc, nvir);
    cudaDeviceSynchronize();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_vv, num_oo, num_oo,
                &alpha, d_t_tmp, num_vv, d_oooo, num_oo, &beta_zero, d_mm1, num_vv);

    // mm2 = t_vvoo * vvvv (t_ovov permuted to vvoo via icjd2cdij)
    icjd2cdij_block<<<num_blocks_ovov, threads_2d>>>(d_t_ovov, d_t_tmp, nocc, nvir);
    cudaDeviceSynchronize();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_oo, num_vv, num_vv,
                &alpha, d_t_tmp, num_oo, d_vvvv, num_vv, &beta_zero, d_mm2, num_oo);

    // mm3 = t_ovov^T * t_ovov
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, num_ov, num_ov, num_ov,
                &alpha, d_t_ovov, num_ov, d_t_ovov, num_ov, &beta_zero, d_mm3, num_ov);

    // mm4 = s_ovov(kbjc→kcjb) * s_ovov(kaic→iakc)
    real_t* d_s_tmp1 = d_t_ovov;  // Reuse t_ovov as workspace
    real_t* d_s_tmp2 = d_t_tmp;   // Reuse t_tmp as workspace
    kaic2iakc_block<<<num_blocks_ovov, threads_2d>>>(d_s_ovov, d_s_tmp1, nocc, nvir);
    kbjc2kcjb_block<<<num_blocks_ovov, threads_2d>>>(d_s_ovov, d_s_tmp2, nocc, nvir);
    cudaDeviceSynchronize();
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_ov, num_ov, num_ov,
                &alpha, d_s_tmp2, num_ov, d_s_tmp1, num_ov, &beta_zero, d_mm4, num_ov);

    // ============================================================
    //  Final contraction: 3 MP3 terms
    // ============================================================
    real_t* d_E_3rd = nullptr;
    tracked_cudaMalloc(&d_E_3rd, sizeof(real_t));
    cudaMemset(d_E_3rd, 0, sizeof(real_t));

    {
        const size_t num_blocks = (num_oovv + num_threads_per_block - 1) / num_threads_per_block;
        contract_iajb_tensors_blocks<<<num_blocks, threads_2d>>>(
            nocc, nvir, d_ovov, d_oovv, d_s_ovov, d_mm1, d_mm2, d_mm3, d_mm4, d_E_3rd);
        cudaDeviceSynchronize();
    }

    real_t h_E_3rd = 0.0;
    cudaMemcpy(&h_E_3rd, d_E_3rd, sizeof(real_t), cudaMemcpyDeviceToHost);
    std::cout << "  [Half-Transform MP3] 3rd perturbation energy: " << std::setprecision(12) << h_E_3rd << std::endl;

    real_t E_MP3 = h_E_mp2 + h_E_3rd;
    std::cout << "MP3 energy: " << E_MP3 << " Hartree" << std::endl;

    // --- Cleanup ---
    tracked_cudaFree(d_oooo);
    tracked_cudaFree(d_vvvv);
    tracked_cudaFree(d_ovov);
    tracked_cudaFree(d_oovv);
    tracked_cudaFree(d_s_ovov);
    tracked_cudaFree(d_t_ovov);
    tracked_cudaFree(d_t_tmp);
    tracked_cudaFree(d_mm1);
    tracked_cudaFree(d_mm2);
    tracked_cudaFree(d_mm3);
    tracked_cudaFree(d_mm4);
    tracked_cudaFree(d_E_3rd);

    return E_MP3;
}


// ============================================================
//  Direct SCF version
// ============================================================
real_t mp3_half_transform_direct(
    RHF& rhf, const HF& hf, int block_s)
{
    // Compute Schwarz factors
    DeviceHostMemory<real_t> schwarz_unsorted(hf.get_num_primitive_shell_pairs());
    gpu::computeSchwarzUpperBounds(
        hf.get_shell_type_infos(), hf.get_shell_pair_type_infos(),
        hf.get_primitive_shells().device_ptr(),
        hf.get_boys_grid().device_ptr(),
        hf.get_cgto_normalization_factors().device_ptr(),
        schwarz_unsorted.device_ptr(), false);

    const auto& shell_type_infos = hf.get_shell_type_infos();
    const auto& shell_pair_type_infos = hf.get_shell_pair_type_infos();
    const auto* d_primitive_shells = hf.get_primitive_shells().device_ptr();
    const auto* d_boys_grid = hf.get_boys_grid().device_ptr();
    const auto* d_cgto_norm = hf.get_cgto_normalization_factors().device_ptr();
    const auto* d_schwarz = schwarz_unsorted.device_ptr();
    const real_t schwarz_thresh = hf.get_schwarz_screening_threshold();

    auto step1 = [&](real_t* d_half, int nao, const real_t* d_C, int s_abs, int bs) {
        gpu::computeHalfTransformedERI(
            shell_type_infos, shell_pair_type_infos,
            d_primitive_shells, d_boys_grid, d_cgto_norm,
            d_half, d_schwarz, schwarz_thresh,
            nao, d_C, s_abs, bs);
    };

    return mp3_half_transform_impl(rhf, step1, block_s);
}


// ============================================================
//  Hash ERI version
// ============================================================
real_t mp3_half_transform_hash(
    RHF& rhf,
    const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    const size_t* d_nonzero_indices, size_t num_nonzero,
    size_t hash_capacity_mask, HashFockMethod method,
    int block_s)
{
    auto step1 = [&](real_t* d_half, int nao, const real_t* d_C, int s_abs, int bs) {
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
    };

    return mp3_half_transform_impl(rhf, step1, block_s);
}

} // namespace gansu
