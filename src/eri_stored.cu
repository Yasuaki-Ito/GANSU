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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <assert.h>


#include "rhf.hpp"
#include "diis.hpp"
#include "eri_stored.hpp"
#include "device_host_memory.hpp"
#include "cphf_solver.hpp"
#include "ccsd_lambda.hpp"

#include "ao2mo.cuh"

#include <Eigen/Dense>

#define FULLMASK 0xffffffff

namespace gansu {

// Forward declaration: full MO-ERI based MP2 helper defined later in this file.
// Referenced by the CPU fallback in ERI_Hash_RHF / ERI_Direct_RHF compute_mp2_energy.
double mp2_from_full_moeri(
    const double* d_eri_mo, const double* d_C, const double* d_eps,
    int nao, int occ);

// Atomic max for double precision (since CUDA does not provide atomicMax for double)
__device__ double atomicMaxDouble(double* address, double val) {
    // cast address to unsigned long long int pointer
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        // Choose the larger of the existing value and the argument as the new candidate value
        double current_val = __longlong_as_double(assumed);
        double max_val = fmax(current_val, val);
        
        // atomicCAS(compare address, expected old value, new value to write)
        // If the value at address matches assumed, write max_val, and return the old value.
        // If not, return the current value at address.
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(max_val));

    } while (assumed != old); // Loop until successful write

    return __longlong_as_double(old);
}


__host__ __device__ double eri_mo_bruteforce(const double* __restrict__ eri_ao,
                         const double* __restrict__ C,
                         int num_basis,
                         int i, int j, int a, int b)
{
  double sum = 0.0;
  for(int mu=0; mu<num_basis; ++mu){
    const double Cmu_i = C[(size_t)num_basis*mu + i];
    for(int nu=0; nu<num_basis; ++nu){
      const double Cnu_j = C[(size_t)num_basis*nu + j];
      const double pref_mn = Cmu_i * Cnu_j;
      for(int la=0; la<num_basis; ++la){
        const double Cla_a = C[(size_t)num_basis*la + a];
        const double pref_mnl = pref_mn * Cla_a;
        for(int si=0; si<num_basis; ++si){
          const double Csi_b = C[(size_t)num_basis*si + b];
          const double v = eri_ao[idx4_to_1(num_basis, mu, nu, la, si)];
          sum += pref_mnl * Csi_b * v;
        }
      }
    }
  }
  return sum;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////// Integral Transformation (Full stored AO ERI to MO ERI)
__global__ void build_kron_C_C(
    const double* __restrict__ C, // [nao x nao], row-major
    int nao,
    double* __restrict__ D        // [N x N], N=nao^2, row-major
){
    int N = nao * nao;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)N * (size_t)N;
    if(idx >= total) return;

    int R = idx % N;   // p*nao + q
    int P = idx / N;   // mu*nao + nu

    int mu = P / nao;
    int nu = P % nao;

    int p  = R / nao;
    int q  = R % nao;

    D[(size_t)P * N + R] = C[(size_t)mu * nao + p] * C[(size_t)nu * nao + q];
}

/**
 * @brief Full AO->MO 4-index ERI transformation (naive, memory-heavy).
 *
 * Computes MO ERI G = D^T * A * D, where
 *  A : AO ERI as (mu nu | la si), viewed as N x N matrix
 *  D : Kronecker product of MO coefficients C ⊗ C
 *
 * All matrices are row-major.
 *
 * @param d_eri_ao  AO ERI array, size nao^4, row-major
 * @param d_C       MO coefficient matrix C(mu,p), size nao x nao, row-major
 * @param nao       Number of AO (and MO) basis functions
 * @param d_eri_mo  Output MO ERI array, size nao^4, row-major (allocated outside)
 */
void transform_ao_eri_to_mo_eri_full(
    const double* d_eri_ao,
    const double* d_C,
    int nao,
    double* d_eri_mo
){
    const int N = nao * nao;

    // Temporary buffers (must fit in GPU memory)
    double* d_D = nullptr;
    double* d_T = nullptr;

    tracked_cudaMalloc((void**)&d_D, (size_t)N * N * sizeof(double));
    if(!d_D){
        THROW_EXCEPTION("tracked_cudaMalloc failed for d_D.");
    }
    tracked_cudaMalloc((void**)&d_T, (size_t)N * N * sizeof(double));
    if(!d_T){
        tracked_cudaFree(d_D);
        THROW_EXCEPTION("tracked_cudaMalloc failed for d_T.");
    }

    // ------------------------------------------------------------------
    // Step 1: Build D = kron(C, C)
    // ------------------------------------------------------------------
    if (!gpu::gpu_available()) {
        // CPU fallback: D[P,R] = C[mu,p] * C[nu,q]  where P=mu*nao+nu, R=p*nao+q
        size_t total = (size_t)N * (size_t)N;
        #pragma omp parallel for
        for (size_t idx = 0; idx < total; idx++) {
            int R = idx % N;
            int P = idx / N;
            int mu = P / nao;
            int nu = P % nao;
            int p = R / nao;
            int q = R % nao;
            d_D[(size_t)P * N + R] = d_C[(size_t)mu * nao + p] * d_C[(size_t)nu * nao + q];
        }
    } else {
        size_t total = (size_t)N * (size_t)N;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        build_kron_C_C<<<blocks, threads>>>(d_C, nao, d_D);
    }

    // ------------------------------------------------------------------
    // Step 2: T = A * D
    //   A : d_eri_ao (N x N)
    //   D : d_D      (N x N)
    //   T : d_T      (N x N)
    // ------------------------------------------------------------------
    gpu::matrixMatrixProduct(
        d_eri_ao,  // A
        d_D,       // B
        d_T,       // C = A * D
        N,
        false,     // transpose A
        false,     // transpose B
        false      // overwrite C
    );

    // ------------------------------------------------------------------
    // Step 3: G = D^T * T
    //   D^T : transpose of D
    //   T   : d_T
    //   G   : d_eri_mo
    // ------------------------------------------------------------------
    gpu::matrixMatrixProduct(
        d_D,       // A
        d_T,       // B
        d_eri_mo,  // C = D^T * T
        N,
        true,      // transpose A
        false,     // transpose B
        false      // overwrite C
    );

    tracked_cudaFree(d_D);
    tracked_cudaFree(d_T);
}


/**
 * @brief GPU kernel: extract a sub-block of MO integrals from the full N⁴ tensor.
 *
 * Maps physicist's notation v(p,q,r,s) = eri_mo[p*N³ + r*N² + q*N + s]
 * to a contiguous output array out[i0*sz1*sz2*sz3 + i1*sz2*sz3 + i2*sz3 + i3]
 * = v(off0+i0, off1+i1, off2+i2, off3+i3).
 */
__global__ void extract_subblock_4d(const double* __restrict__ eri_mo,
                                     double* __restrict__ out,
                                     int N, int off0, int sz0,
                                     int off1, int sz1,
                                     int off2, int sz2,
                                     int off3, int sz3) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)sz0 * sz1 * sz2 * sz3;
    if (gid >= total) return;

    int i3 = gid % sz3; size_t rem = gid / sz3;
    int i2 = rem % sz2; rem /= sz2;
    int i1 = rem % sz1;
    int i0 = (int)(rem / sz1);

    int p = off0 + i0, q = off1 + i1, r = off2 + i2, s = off3 + i3;
    size_t N3 = (size_t)N * N * N;
    out[gid] = eri_mo[(size_t)p * N3 + (size_t)r * N * N + (size_t)q * N + s];
}

/**
 * @brief GPU kernel: extract w_oovv = 2*v(k,l,c,d) - v(k,l,d,c) directly from MO integrals.
 */
__global__ void extract_w_oovv_kernel(const double* __restrict__ eri_mo,
                                       double* __restrict__ w_oovv,
                                       int N, int nocc, int nvir) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)nocc * nocc * nvir * nvir;
    if (gid >= total) return;

    int d = gid % nvir; size_t rem = gid / nvir;
    int c = rem % nvir; rem /= nvir;
    int l = rem % nocc;
    int k = (int)(rem / nocc);

    size_t N3 = (size_t)N * N * N;
    size_t N2 = (size_t)N * N;
    int oc = nocc + c, od = nocc + d;
    double v_cd = eri_mo[(size_t)k * N3 + (size_t)oc * N2 + (size_t)l * N + od];
    double v_dc = eri_mo[(size_t)k * N3 + (size_t)od * N2 + (size_t)l * N + oc];
    w_oovv[gid] = 2.0 * v_cd - v_dc;
}

/**
 * @brief GPU kernel: compute (T) perturbative triples energy on GPU.
 *
 * Each thread block processes one (a,b,c) triple (a >= b >= c).
 * Shared memory layout: wt[6*o3] + zt[6*o3] + r3buf[o3] + red[blockDim.x]
 * where o3 = nocc^3.
 */
__global__ void ccsd_t_energy_kernel(
    const double* __restrict__ F_sum, int F_cols_int,
    const double* __restrict__ M_sum, int M_cols_int,
    const double* __restrict__ v_oovv,
    const double* __restrict__ t1,
    const double* __restrict__ eps,
    int nocc, int nvir,
    const int* __restrict__ abc_triples,
    int num_triples,
    double* __restrict__ block_E_T,
    double* __restrict__ g_wt,     // global memory: num_triples * 6 * o3
    double* __restrict__ g_zt)     // global memory: num_triples * 6 * o3
{
    int triple_id = blockIdx.x;
    if (triple_id >= num_triples) return;

    const int a = abc_triples[triple_id * 3];
    const int b = abc_triples[triple_id * 3 + 1];
    const int c = abc_triples[triple_id * 3 + 2];

    const int oo = nocc * nocc;
    const int o3 = oo * nocc;
    const int vv = nvir * nvir;
    const size_t F_cols = (size_t)F_cols_int;
    const size_t M_cols = (size_t)M_cols_int;

    double d3_scale = 1.0;
    if (a == c) d3_scale = 6.0;
    else if (a == b || b == c) d3_scale = 2.0;

    int perms[6][3] = {{a,b,c},{a,c,b},{b,a,c},{b,c,a},{c,a,b},{c,b,a}};

    // Per-block pointers into global memory
    const size_t block_offset = (size_t)triple_id * 6 * o3;
    double* wt = g_wt + block_offset;
    double* zt = g_zt + block_offset;

    // Shared memory: only r3buf (o3) + reduction buffer (blockDim.x)
    extern __shared__ double smem[];
    double* r3buf = smem;              // o3
    double* red   = smem + o3;         // blockDim.x

    // Phase 1 & 2: for each permutation, compute wt[p] and zt[p]
    for (int p = 0; p < 6; p++) {
        int aa = perms[p][0], bb = perms[p][1], cc = perms[p][2];

        // Phase 1: compute wt[p] and store wpv in zt[p] temporarily
        for (int ijk = threadIdx.x; ijk < o3; ijk += blockDim.x) {
            int k = ijk % nocc;
            int j = (ijk / nocc) % nocc;
            int i = ijk / oo;

            size_t f_row = (size_t)i * vv + (size_t)aa * nvir + bb;
            size_t f_col = ((size_t)k * nocc + j) * nvir + cc;
            double wval = F_sum[f_row * F_cols + f_col];

            size_t m_row = (size_t)aa * oo + (size_t)j * nocc + i;
            size_t m_col = (size_t)k * vv + (size_t)bb * nvir + cc;
            wval += M_sum[m_row * M_cols + m_col];

            double vval = v_oovv[((size_t)i * nocc + j) * vv + (size_t)aa * nvir + bb]
                        * t1[k * nvir + cc];

            wt[p * o3 + ijk] = wval;
            zt[p * o3 + ijk] = wval + 0.5 * vval;
        }
        __syncthreads();

        // Phase 2a: compute r3out from wpv (stored in zt[p]) into shared r3buf
        for (int ijk = threadIdx.x; ijk < o3; ijk += blockDim.x) {
            double wpv_self = zt[p * o3 + ijk];
            int k = ijk % nocc;
            int j = (ijk / nocc) % nocc;
            int i = ijk / oo;
            int idx1 = (i*nocc+k)*nocc+j;
            int idx2 = (j*nocc+i)*nocc+k;
            int idx3 = (j*nocc+k)*nocc+i;
            int idx4 = (k*nocc+i)*nocc+j;
            int idx5 = (k*nocc+j)*nocc+i;

            r3buf[ijk] = 4.0*wpv_self + zt[p*o3+idx3] + zt[p*o3+idx4]
                       - 2.0*zt[p*o3+idx5] - 2.0*zt[p*o3+idx1] - 2.0*zt[p*o3+idx2];
        }
        __syncthreads();

        // Phase 2b: write zt[p] = r3out / D
        for (int ijk = threadIdx.x; ijk < o3; ijk += blockDim.x) {
            int k = ijk % nocc;
            int j = (ijk / nocc) % nocc;
            int i = ijk / oo;
            double D = (eps[i] + eps[j] + eps[k]
                      - eps[nocc+perms[p][0]] - eps[nocc+perms[p][1]] - eps[nocc+perms[p][2]]) * d3_scale;
            zt[p * o3 + ijk] = r3buf[ijk] / D;
        }
        __syncthreads();
    }

    // Phase 3: compute 36 dot products for energy
    const int comp[6][6] = {
        {0,1,2,3,4,5}, {1,0,4,5,2,3}, {2,3,0,1,5,4},
        {4,5,1,0,3,2}, {3,2,5,4,0,1}, {5,4,3,2,1,0}
    };

    double thread_E = 0.0;
    for (int q = 0; q < 6; q++) {
        for (int pp = 0; pp < 6; pp++) {
            int s = comp[q][pp];
            for (int r = threadIdx.x; r < o3; r += blockDim.x) {
                int kr = r % nocc;
                int jr = (r / nocc) % nocc;
                int ir = r / oo;
                int sr;
                switch(s) {
                    case 0: sr = r; break;
                    case 1: sr = (ir*nocc+kr)*nocc+jr; break;
                    case 2: sr = (jr*nocc+ir)*nocc+kr; break;
                    case 3: sr = (jr*nocc+kr)*nocc+ir; break;
                    case 4: sr = (kr*nocc+ir)*nocc+jr; break;
                    default: sr = (kr*nocc+jr)*nocc+ir; break;
                }
                thread_E += wt[pp*o3+sr] * zt[q*o3+r];
            }
        }
    }

    // Block reduction (shared memory)
    red[threadIdx.x] = thread_E;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride)
            red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        block_E_T[triple_id] = red[0];
}

/**
 * @brief GPU kernel: permute indices of a 4D tensor of size N×N×N×N.
 *
 * For input tensor in[i0][i1][i2][i3], produces output such that
 * out[j0][j1][j2][j3] = in[i0][i1][i2][i3] where j_{p_k} = i_k.
 * E.g., (p0,p1,p2,p3) = (1,0,2,3) swaps the first two indices.
 */
__global__ void tensor4d_permute_kernel(const double* __restrict__ in,
                                        double* __restrict__ out,
                                        int N, int p0, int p1, int p2, int p3) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t N4 = (size_t)N * N * N * N;
    if (gid >= N4) return;

    int i3 = gid % N; size_t rem = gid / N;
    int i2 = rem % N; rem /= N;
    int i1 = rem % N;
    int i0 = (int)(rem / N);

    int out_i[4];
    out_i[p0] = i0;
    out_i[p1] = i1;
    out_i[p2] = i2;
    out_i[p3] = i3;

    size_t out_gid = ((size_t)out_i[0]*N + out_i[1])*(size_t)N*N + (size_t)out_i[2]*N + out_i[3];
    out[out_gid] = in[gid];
}

/**
 * @brief GPU kernel: build Wabcd from v_vvvv and ovvv_t1 (DGEMM output).
 *
 * Wabcd[a,b,c,d] = v_vvvv[a,b,c,d] - ovvv_t1[(a*vv+d*nvir+c), b] - ovvv_t1[(b*vv+c*nvir+d), a]
 * Eliminates ovvv_t1 download + Wabcd upload per CCSD iteration.
 */
/**
 * @brief GPU kernel: build tau = t2 + t1⊗t1 directly on GPU.
 * tau[((i*nocc+j)*nvir+a)*nvir+b] = t2v[same] + t1[i*nvir+a] * t1[j*nvir+b]
 */
__global__ void build_tau_kernel(const double* __restrict__ t2v,
                                  const double* __restrict__ t1,
                                  double* __restrict__ tau,
                                  int nocc, int nvir) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int vv = nvir * nvir;
    const size_t total = (size_t)nocc * nocc * vv;
    if (gid >= total) return;

    int b = gid % nvir; size_t rem = gid / nvir;
    int a = rem % nvir; rem /= nvir;
    int j = rem % nocc;
    int i = (int)(rem / nocc);

    tau[gid] = t2v[gid] + t1[i * nvir + a] * t1[j * nvir + b];
}

__global__ void build_Wabcd_kernel(const double* __restrict__ v_vvvv,
                                    const double* __restrict__ ovvv_t1,
                                    double* __restrict__ Wabcd,
                                    int nvir) {
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t vv = (size_t)nvir * nvir;
    size_t vv2 = vv * vv;
    if (gid >= vv2) return;

    int d = gid % nvir; size_t rem = gid / nvir;
    int c = rem % nvir; rem /= nvir;
    int b = rem % nvir;
    int a = (int)(rem / nvir);

    // ovvv_t1[(x*vv + y*nvir + z), w] = sum_k v_ovvv[k, x, y, z] * t1[k, w]
    Wabcd[gid] = v_vvvv[gid]
                 - ovvv_t1[((size_t)a*vv + (size_t)d*nvir + c)*nvir + b]
                 - ovvv_t1[((size_t)b*vv + (size_t)c*nvir + d)*nvir + a];
}

/**
 * @brief GPU kernel: compute Fac = -sum_{kl,d} w_oovv[(kl),(cd)] * tau[T2(k,l,a,d)]
 *
 * Both w_oovv and tau are already on GPU — no data transfer needed.
 */
__global__ void compute_Fac_kernel(const double* __restrict__ w_oovv,
                                    const double* __restrict__ tau,
                                    double* __restrict__ Fac,
                                    int nocc, int nvir) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= nvir || c >= nvir) return;

    int oo = nocc * nocc;
    int vv = nvir * nvir;
    double val = 0.0;
    for (int kl = 0; kl < oo; kl++)
        for (int d = 0; d < nvir; d++)
            val -= w_oovv[kl*vv + c*nvir + d] * tau[kl*vv + a*nvir + d];
    Fac[a*nvir + c] = val;
}

/**
 * @brief GPU kernel: compute Fkc = sum_{l,d} w_oovv[(k*nocc+l)*vv + c*nvir+d] * t1[l*nvir+d]
 *
 * Both w_oovv and t1 are already on GPU — no data transfer needed.
 */
__global__ void compute_Fkc_kernel(const double* __restrict__ w_oovv,
                                    const double* __restrict__ t1,
                                    double* __restrict__ Fkc,
                                    int nocc, int nvir) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (k >= nocc || c >= nvir) return;

    int vv = nvir * nvir;
    double val = 0.0;
    for (int l = 0; l < nocc; l++)
        for (int d = 0; d < nvir; d++)
            val += w_oovv[(k*nocc + l)*vv + c*nvir + d] * t1[l*nvir + d];
    Fkc[k*nvir + c] = val;
}

/**
 * @brief 4-stage AO->MO ERI transformation using half-transforms.
 *
 * Contracts each AO index with C one at a time via DGEMM:
 *   Stage 1: (μνλσ) → (pνλσ)  via C^T × ERI
 *   Stage 2: (pνλσ) → (pqλσ)  via C^T × permuted
 *   Stage 3: (pqλσ) → (pqrσ)  via C^T × permuted
 *   Stage 4: (pqrσ) → (pqrs)  via permuted × C
 *
 * Cost: O(N^5) vs O(N^6) for the Kronecker product method.
 * Memory: 2 × N^4 (ping-pong buffers) vs 3 × N^4 (Kronecker: D + T + G).
 */
/**
 * @brief Extract a sub-block of MO integrals from d_eri_mo on GPU.
 *
 * Extracts v(off0..off0+sz0-1, off1..off1+sz1-1, off2..off2+sz2-1, off3..off3+sz3-1)
 * in physicist's notation, stored contiguously in d_out.
 */
static void gpu_extract_subblock(const double* d_eri_mo, double* d_out, int N,
                                  int off0, int sz0, int off1, int sz1,
                                  int off2, int sz2, int off3, int sz3) {
    if (!gpu::gpu_available()) {
        // CPU: extract 4D sub-block matching GPU kernel indexing:
        // GPU: out[gid] = eri_mo[p*N³ + r*N² + q*N + s]  where p=off0+i0, q=off1+i1, r=off2+i2, s=off3+i3
        // Note: 2nd and 3rd indices are swapped in memory layout (chemist's notation)
        #pragma omp parallel for collapse(2)
        for (int i0 = 0; i0 < sz0; i0++)
            for (int i1 = 0; i1 < sz1; i1++)
                for (int i2 = 0; i2 < sz2; i2++)
                    for (int i3 = 0; i3 < sz3; i3++) {
                        int p = off0+i0, q = off1+i1, r = off2+i2, s = off3+i3;
                        size_t src = ((size_t)p*N*N*N + (size_t)r*N*N + (size_t)q*N + s);
                        size_t dst = ((size_t)i0*sz1*sz2*sz3 + i1*sz2*sz3 + i2*sz3 + i3);
                        d_out[dst] = d_eri_mo[src];
                    }
        return;
    }
    size_t total = (size_t)sz0 * sz1 * sz2 * sz3;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    extract_subblock_4d<<<blocks, threads>>>(d_eri_mo, d_out, N,
                                              off0, sz0, off1, sz1,
                                              off2, sz2, off3, sz3);
}

void transform_ao_eri_to_mo_eri_4stage(
    const double* d_eri_ao,
    const double* d_C,
    int nao,
    double* d_eri_mo
){
    const int N = nao;
    const size_t N3 = (size_t)N * N * N;
    const size_t N4 = N3 * N;

    double* d_tmp = nullptr;
    tracked_cudaMalloc((void**)&d_tmp, N4 * sizeof(double));

    auto permute4d = [&](const double* in, double* out, int p0, int p1, int p2, int p3) {
        if (!gpu::gpu_available()) {
            // CPU fallback: out[j0,j1,j2,j3] = in[i0,i1,i2,i3] where j_{p_k} = i_k
            #pragma omp parallel for
            for (size_t gid = 0; gid < N4; gid++) {
                int i3 = gid % N; size_t rem = gid / N;
                int i2 = rem % N; rem /= N;
                int i1 = rem % N;
                int i0 = (int)(rem / N);
                int out_i[4];
                out_i[p0] = i0; out_i[p1] = i1; out_i[p2] = i2; out_i[p3] = i3;
                size_t out_gid = ((size_t)out_i[0]*N + out_i[1])*(size_t)N*N + (size_t)out_i[2]*N + out_i[3];
                out[out_gid] = in[gid];
            }
        } else {
            int threads = 256;
            int blocks = (int)((N4 + threads - 1) / threads);
            tensor4d_permute_kernel<<<blocks, threads>>>(in, out, N, p0, p1, p2, p3);
        }
    };

    // Stage 1: half1[p,ν,λ,σ] = sum_μ C^T[p,μ] × eri_ao[μ, ν*N²+λ*N+σ]
    gpu::matrixMatrixProductRect(d_C, d_eri_ao, d_eri_mo,
                                N, (int)N3, N,
                                true, false, false, 1.0);

    // Transpose: [p,ν,λ,σ] → [ν,p,λ,σ]  (swap indices 0,1)
    permute4d(d_eri_mo, d_tmp, 1, 0, 2, 3);

    // Stage 2: half2[q,p,λ,σ] = sum_ν C^T[q,ν] × tmp[ν, p*N²+λ*N+σ]
    gpu::matrixMatrixProductRect(d_C, d_tmp, d_eri_mo,
                                N, (int)N3, N,
                                true, false, false, 1.0);

    // Transpose: [q,p,λ,σ] → [λ,p,q,σ]  (swap indices 0,2)
    permute4d(d_eri_mo, d_tmp, 2, 1, 0, 3);

    // Stage 3: half3[r,p,q,σ] = sum_λ C^T[r,λ] × tmp[λ, p*N²+q*N+σ]
    gpu::matrixMatrixProductRect(d_C, d_tmp, d_eri_mo,
                                N, (int)N3, N,
                                true, false, false, 1.0);

    // Transpose: [r,p,q,σ] → [p,q,r,σ]  (cyclic rotate first 3: perm={2,0,1,3})
    permute4d(d_eri_mo, d_tmp, 2, 0, 1, 3);

    // Stage 4: eri_mo[p*N²+q*N+r, s] = sum_σ tmp[p*N²+q*N+r, σ] × C[σ,s]
    gpu::matrixMatrixProductRect(d_tmp, d_C, d_eri_mo,
                                (int)N3, N, N,
                                false, false, false, 1.0);

    tracked_cudaFree(d_tmp);
}


//// debug for MO ERI
__global__ void check_moeri_kernel(const double* __restrict__ eri_mo,
    const double* __restrict__ eri_ao,
    const double* __restrict__ C,
    int num_basis)
{
  // Flattened index over (p,q,r,s) 
  size_t total = (size_t)num_basis * num_basis * num_basis * num_basis;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < total){
    size_t t = gid;
    int s  = (int)(t % num_basis); t /= num_basis;
    int r  = (int)(t % num_basis); t /= num_basis;
    int q  = (int)(t % num_basis); t /= num_basis;
    int p  = (int)(t % num_basis);

    double eri_mo_val = eri_mo[idx4_to_1(num_basis, p, q, r, s)];
    double eri_mo_val_bruteforce = eri_mo_bruteforce(eri_ao, C, num_basis, p, q, r, s);

    if(fabs(eri_mo_val - eri_mo_val_bruteforce) > 1e-10){
      printf("Mismatch: (%d,%d,%d,%d): eri_mo=%18.10f, eri_mo_bruteforce=%18.10f\n", p, q, r, s, eri_mo_val, eri_mo_val_bruteforce);
    }else{
        //printf("Match: (%d,%d,%d,%d): eri_mo=%18.10f, eri_mo_bruteforce=%18.10f\n", p, q, r, s, eri_mo_val, eri_mo_val_bruteforce);
    }

  }
}

void check_moeri(const double* d_eri_mo,
    const double* d_eri_ao,
    const double* d_C,
    int num_basis)
{
    if (!gpu::gpu_available()) {
        // CPU fallback: brute-force MO ERI check (debug only)
        size_t total = (size_t)num_basis * num_basis * num_basis * num_basis;
        for (size_t gid = 0; gid < total; gid++) {
            size_t t = gid;
            int s  = (int)(t % num_basis); t /= num_basis;
            int r  = (int)(t % num_basis); t /= num_basis;
            int q  = (int)(t % num_basis); t /= num_basis;
            int p  = (int)(t % num_basis);
            double eri_mo_val = d_eri_mo[idx4_to_1(num_basis, p, q, r, s)];
            double eri_mo_val_bruteforce = eri_mo_bruteforce(d_eri_ao, d_C, num_basis, p, q, r, s);
            if (fabs(eri_mo_val - eri_mo_val_bruteforce) > 1e-10) {
                printf("Mismatch: (%d,%d,%d,%d): eri_mo=%18.10f, eri_mo_bruteforce=%18.10f\n",
                       p, q, r, s, eri_mo_val, eri_mo_val_bruteforce);
            }
        }
    } else {
    const int num_threads = 256;

    size_t total = (size_t)num_basis * num_basis * num_basis * num_basis;
    const int num_blocks = (int)((total + num_threads - 1) / num_threads);

    check_moeri_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eri_ao, d_C, num_basis);

    cudaDeviceSynchronize();
    } // else (gpu_available)
}



///////////////////////////////////////////////////////////////////////////////////////// MP2 energy calculation (from stored full MO ERI)

__global__ void mp2_from_moeri_kernel(
    const double* __restrict__ eri_mo,  // device, nao^4, row-major
    const double* __restrict__ eps,     // device, nao
    int nao, int occ,
    double* __restrict__ E_out)
{
    const int vir = nao - occ;

    size_t total = (size_t)occ * occ * (size_t)vir * (size_t)vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;

    if(gid < total){
        size_t t = gid;

        int b_ = (int)(t % vir); t /= vir;
        int a_ = (int)(t % vir); t /= vir;
        int j  = (int)(t % occ); t /= occ;
        int i  = (int)(t % occ);

        int a = occ + a_;
        int b = occ + b_;

        double denom = eps[i] + eps[j] - eps[a] - eps[b];
        if(fabs(denom) > 1e-14){
            double iajb = eri_mo[idx4_to_1(nao, i, a, j, b)]; // (ia|jb)
            double ibja = eri_mo[idx4_to_1(nao, i, b, j, a)]; // (ib|ja)

            contrib = iajb * (2.0 * iajb - ibja) / denom;
        }
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(E_out, block_sum);
    }
}

double mp2_from_aoeri_via_full_moeri(
    const double* d_eri_ao,   // device, size nao^4, row-major (mu nu | la si)
    const double* d_C,        // device, size nao*nao, row-major (mu,p)
    const double* d_eps,      // device, size nao
    int nao,
    int occ)
{
    const int N = nao * nao;

    // ------------------------------------------------------------
    // 1) allocate full MO ERI on device: d_eri_mo (N x N)
    // ------------------------------------------------------------
    double* d_eri_mo = nullptr;
    size_t bytes_mo = (size_t)N * (size_t)N * sizeof(double);
    tracked_cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("tracked_cudaMalloc failed for d_eri_mo.");
    }

    // ------------------------------------------------------------
    // 2) AO -> MO full transformation (writes into d_eri_mo)
    // Note: MP2 does not used all MO ERIs, but we compute all for simplicity.
    // ------------------------------------------------------------
    {
        std::string str = "Computing AO -> MO full integral transformation... ";
        PROFILE_ELAPSED_TIME(str);

        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_C, nao, d_eri_mo);
        cudaDeviceSynchronize();
    }

    // show all MO ERI
    /*
    real_t* h_eri_ao = new real_t[N * N];
    for(int p = 0; p < nao; ++p){
        for(int q = 0; q < nao; ++q){
            for(int r = 0; r < nao; ++r){
                for(int s = 0; s < nao; ++s){
                    size_t idx = p * N * N * N + q * N * N + r * N + s;
                    h_eri_ao[idx] =-10.0;
                }
            }
        }
    }
    cudaMemcpy(h_eri_ao, d_eri_ao, bytes_mo, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int p = 0; p < nao; ++p){
        for(int q = 0; q < nao; ++q){
            for(int r = 0; r < nao; ++r){
                for(int s = 0; s < nao; ++s){
                    size_t idx = p * N * N * N + q * N * N + r * N + s;
                    std::cout << "ERI(" << p << "," << q << "," << r << "," << s << ") = " << h_eri_ao[idx] << std::endl;
                }
            }
        }
    }
    delete[] h_eri_ao;
    */

    // ------------------------------------------------------------
    // 3) MP2 energy from full MO ERI
    // ------------------------------------------------------------
    int vir = nao - occ;
    size_t total = (size_t)occ * (size_t)occ * (size_t)vir * (size_t)vir;

    double* d_E = nullptr;
    tracked_cudaMalloc((void**)&d_E, sizeof(double));
    cudaMemset(d_E, 0, sizeof(double));

    int threads = 128;
    int blocks  = (int)((total + threads - 1) / threads);
    size_t shmem = (size_t)threads * sizeof(double);

    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            // CPU fallback: MP2 energy from full MO ERI
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = occ + a_, b = occ + b_;
                double denom = d_eps[i] + d_eps[j] - d_eps[a] - d_eps[b];
                if (fabs(denom) > 1e-14) {
                    double iajb = d_eri_mo[idx4_to_1(nao, i, a, j, b)];
                    double ibja = d_eri_mo[idx4_to_1(nao, i, b, j, a)];
                    cpu_E += iajb * (2.0 * iajb - ibja) / denom;
                }
            }
            *d_E = cpu_E;
        } else {
            mp2_from_moeri_kernel<<<blocks, threads, shmem>>>(d_eri_mo, d_eps, nao, occ, d_E);
            cudaDeviceSynchronize();
        }
    }



    double h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(double), cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------
    // 4) cleanup
    // ------------------------------------------------------------
    tracked_cudaFree(d_E);
    tracked_cudaFree(d_eri_mo);

    return h_E;
}




/////////////////////////////////////////////////////////////////////////////////////////////////////////// MP2 energy calculation (naive implementation with on-the-fly integral transformation)
__global__ void mp2_naive_kernel(const double* __restrict__ eri_ao,
                      const double* __restrict__ C,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,a,b) with i,j in occ, a,b in vir (offset by occ)
  size_t total = (size_t)occ * occ * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;

    double denom = eps[i] + eps[j] - eps[a] - eps[b];
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo_bruteforce(eri_ao, C, num_basis, i, a, j, b);
      double ibja = eri_mo_bruteforce(eri_ao, C, num_basis, i, b, j, a);
      contrib = (iajb * (2.0*iajb-ibja)) / denom;       // sum_{ijab} (ia|jb)(2*(ia|jb)-(ib|ja)) / denom
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}

real_t mp2_naive(const real_t* d_eri, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_occ) {
    const int num_threads = 1024;

    size_t occ = (size_t)num_occ;
    size_t vir = (size_t)(num_basis - num_occ);
    size_t total = (size_t)occ * occ * vir * vir;
    const int num_blocks = (int)((total + num_threads - 1) / num_threads);
    size_t shmem = (size_t)num_threads * sizeof(double);

    real_t* d_mp2_energy;
    tracked_cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
    if(d_mp2_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP2 energy.");
    }
    cudaMemset(d_mp2_energy, 0.0, sizeof(real_t));

    if (!gpu::gpu_available()) {
        // CPU fallback: brute-force MP2 with on-the-fly AO→MO transformation
        double cpu_E = 0.0;
        #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
        for (size_t gid = 0; gid < total; gid++) {
            size_t t = gid;
            int b_ = (int)(t % vir); t /= vir;
            int a_ = (int)(t % vir); t /= vir;
            int j  = (int)(t % occ); t /= occ;
            int i  = (int)(t % occ);
            int a = num_occ + a_, b = num_occ + b_;
            double denom = d_orbital_energies[i] + d_orbital_energies[j] - d_orbital_energies[a] - d_orbital_energies[b];
            if (fabs(denom) > 1e-14) {
                // Brute-force AO→MO integral (ia|jb)
                auto eri_bf = [&](int p, int q, int r, int s) -> double {
                    double sum = 0.0;
                    for (int mu = 0; mu < num_basis; mu++)
                        for (int nu = 0; nu < num_basis; nu++)
                            for (int la = 0; la < num_basis; la++)
                                for (int si = 0; si < num_basis; si++)
                                    sum += d_coefficient_matrix[(size_t)num_basis*mu+p]
                                         * d_coefficient_matrix[(size_t)num_basis*nu+q]
                                         * d_coefficient_matrix[(size_t)num_basis*la+r]
                                         * d_coefficient_matrix[(size_t)num_basis*si+s]
                                         * d_eri[idx4_to_1(num_basis, mu, nu, la, si)];
                    return sum;
                };
                double iajb = eri_bf(i, a, j, b);
                double ibja = eri_bf(i, b, j, a);
                cpu_E += iajb * (2.0*iajb - ibja) / denom;
            }
        }
        *d_mp2_energy = cpu_E;
    } else {
        mp2_naive_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, d_mp2_energy);
    }

    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_mp2_energy);

    return h_mp2_energy;

}

__global__ void mp2_moeri_kernel(const double* __restrict__ eri_mo,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,a,b) with i,j in occ, a,b in vir (offset by occ)
  size_t total = (size_t)occ * occ * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;

    double denom = eps[i] + eps[j] - eps[a] - eps[b];
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo[idx4_to_1(num_basis, i, a, j, b)];
      double ibja = eri_mo[idx4_to_1(num_basis, i, b, j, a)];
      contrib = (iajb * (2.0*iajb-ibja)) / denom;       // sum_{ijab} (ia|jb)(2*(ia|jb)-(ib|ja)) / denom
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}

// Direct MP2 energy accumulation kernel
// V is (nocc, nvir, nvir, bj) where j ranges from j_start to j_start+bj-1
// V[i*nvir*nvir*bj + a*nvir*bj + b*bj + jj] = (ia|jb) for j = j_start + jj
__global__ void mp2_stored_kernel_ovov_direct(
    double* g_energy, const double* V, const double* g_eps,
    const int nocc, const int nvir, const int j_start, const int bj)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_tmp = 0;
    __syncthreads();

    double tmp = 0.0;
    const size_t seq = (size_t)blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    const size_t total = (size_t)nocc * nvir * nocc * nvir;
    if (seq < total) {
        // Decode (i, a, j_local_within_nocc, b) from flat index over full (nocc,nvir,nocc,nvir)
        const int ia = seq / ((size_t)nocc * nvir);
        const int jb = seq % ((size_t)nocc * nvir);
        const int i = ia / nvir;
        const int a = ia % nvir;
        const int j = jb / nvir;
        const int b = jb % nvir;

        // Check if j is in current block
        const int jj = j - j_start;
        if (jj >= 0 && jj < bj) {
            const double iajb = V[(size_t)i*nvir*nvir*bj + (size_t)a*nvir*bj + (size_t)b*bj + jj];
            const double ibja = V[(size_t)i*nvir*nvir*bj + (size_t)b*nvir*bj + (size_t)a*bj + jj];
            tmp = iajb * (2.0 * iajb - ibja) / (g_eps[i] + g_eps[j] - g_eps[nocc + a] - g_eps[nocc + b]);
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
        tmp += __shfl_down_sync(0xFFFFFFFF, tmp, offset);
    if (threadIdx.x == 0) atomicAdd(&s_tmp, tmp);
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(g_energy, s_tmp);
}

// ============================================================
//  Hash COO → dense OVOV transformation kernel
//  Each thread processes one COO entry (μν|λσ) and accumulates
//  contributions to (ia|jb) using MO coefficients.
// ============================================================
__global__ void hash_coo_to_ovov_kernel(
    const unsigned long long* __restrict__ g_coo_keys,
    const double* __restrict__ g_coo_values,
    const size_t num_entries,
    const double* __restrict__ g_C,   // MO coefficients [nao × nao] row-major
    double* __restrict__ g_ovov,       // output OVOV [nocc*nvir*nocc*nvir]
    const int nao, const int nocc, const int nvir)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    const unsigned long long key = g_coo_keys[tid];
    const double val = g_coo_values[tid];
    if (fabs(val) < 1e-18) return;

    // Decode canonical (mu<=nu, la<=si, mn<=ls) indices
    int mu = (int)((key >> 48) & 0xFFFF);
    int nu = (int)((key >> 32) & 0xFFFF);
    int la = (int)((key >> 16) & 0xFFFF);
    int si = (int)(key & 0xFFFF);

    // Enumerate all unique permutations of (μν|λσ) under 8-fold symmetry.
    // Canonical: mu<=nu, la<=si, (mu,nu)<=(la,si).
    // 8 permutations: (mn|ls), (nm|ls), (mn|sl), (nm|sl),
    //                 (ls|mn), (sl|mn), (ls|nm), (sl|nm)
    // Use a small buffer and deduplicate.
    struct Perm { int m, n, l, s; };
    Perm perms[8] = {
        {mu,nu,la,si}, {nu,mu,la,si}, {mu,nu,si,la}, {nu,mu,si,la},
        {la,si,mu,nu}, {si,la,mu,nu}, {la,si,nu,mu}, {si,la,nu,mu}
    };
    // Deduplicate: mark visited by comparing encoded (m,n,l,s)
    int n_unique = 0;
    Perm unique[8];
    for (int p = 0; p < 8; p++) {
        bool dup = false;
        for (int q = 0; q < n_unique; q++) {
            if (perms[p].m == unique[q].m && perms[p].n == unique[q].n &&
                perms[p].l == unique[q].l && perms[p].s == unique[q].s) {
                dup = true; break;
            }
        }
        if (!dup) unique[n_unique++] = perms[p];
    }

    for (int p = 0; p < n_unique; p++) {
        int m = unique[p].m, n = unique[p].n, l = unique[p].l, s = unique[p].s;
        // (ia|jb) += C(m,i)*C(n,a+nocc) * val * C(l,j)*C(s,b+nocc)
        for (int i = 0; i < nocc; i++) {
            double Cmi = g_C[m * nao + i];
            if (fabs(Cmi) < 1e-15) continue;
            for (int a = 0; a < nvir; a++) {
                double CnA = g_C[n * nao + (a + nocc)];
                if (fabs(CnA) < 1e-15) continue;
                double fac1 = Cmi * CnA * val;
                for (int j = 0; j < nocc; j++) {
                    double ClJ = g_C[l * nao + j];
                    if (fabs(ClJ) < 1e-15) continue;
                    double fac2 = fac1 * ClJ;
                    for (int b = 0; b < nvir; b++) {
                        double CsB = g_C[s * nao + (b + nocc)];
                        atomicAdd(&g_ovov[((size_t)i * nvir + a) * nocc * nvir + j * nvir + b],
                                  fac2 * CsB);
                    }
                }
            }
        }
    }
}

// ============================================================
//  Hash COO → full dense MO ERI transformation kernel
//  Each thread processes one COO entry and accumulates all
//  MO ERI contributions (pq|rs) = Σ C(μ,p)C(ν,q)(μν|λσ)C(λ,r)C(σ,s)
// ============================================================
__global__ void hash_coo_to_mo_eri_kernel(
    const unsigned long long* __restrict__ g_coo_keys,
    const double* __restrict__ g_coo_values,
    const size_t num_entries,
    const double* __restrict__ g_C,
    double* __restrict__ g_mo_eri,
    const int nao)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    const unsigned long long key = g_coo_keys[tid];
    const double val = g_coo_values[tid];
    if (fabs(val) < 1e-18) return;

    int mu = (int)((key >> 48) & 0xFFFF);
    int nu = (int)((key >> 32) & 0xFFFF);
    int la = (int)((key >> 16) & 0xFFFF);
    int si = (int)(key & 0xFFFF);

    // 8-fold symmetry: enumerate unique permutations
    struct Perm { int m, n, l, s; };
    Perm perms[8] = {
        {mu,nu,la,si}, {nu,mu,la,si}, {mu,nu,si,la}, {nu,mu,si,la},
        {la,si,mu,nu}, {si,la,mu,nu}, {la,si,nu,mu}, {si,la,nu,mu}
    };
    int n_unique = 0;
    Perm unique[8];
    for (int pp = 0; pp < 8; pp++) {
        bool dup = false;
        for (int q = 0; q < n_unique; q++)
            if (perms[pp].m == unique[q].m && perms[pp].n == unique[q].n &&
                perms[pp].l == unique[q].l && perms[pp].s == unique[q].s) { dup = true; break; }
        if (!dup) unique[n_unique++] = perms[pp];
    }

    const size_t nao2 = (size_t)nao * nao;
    for (int pp = 0; pp < n_unique; pp++) {
        int m = unique[pp].m, n = unique[pp].n, l = unique[pp].l, s = unique[pp].s;
        for (int p = 0; p < nao; p++) {
            double Cmp = g_C[m * nao + p];
            if (fabs(Cmp) < 1e-15) continue;
            for (int q = 0; q < nao; q++) {
                double CnQ = g_C[n * nao + q];
                if (fabs(CnQ) < 1e-15) continue;
                double fac1 = Cmp * CnQ * val;
                for (int r = 0; r < nao; r++) {
                    double ClR = g_C[l * nao + r];
                    if (fabs(ClR) < 1e-15) continue;
                    double fac2 = fac1 * ClR;
                    for (int ss = 0; ss < nao; ss++) {
                        double CsS = g_C[s * nao + ss];
                        atomicAdd(&g_mo_eri[((size_t)p * nao + q) * nao2 + r * nao + ss],
                                  fac2 * CsS);
                    }
                }
            }
        }
    }
}

//*
__global__ void mp2_stored_kernel_ovov(
    double* g_energy_second, 
    const double* g_eri_mo, const double* g_eps, 
    const int num_occupied, const int num_virtual)
{
    __shared__ double s_tmp;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_tmp = 0;
    }
    __syncthreads();

    double tmp = 0.0;
    //const int num_orbitals = num_occupied + num_virtual;
    const size_t seq = (((size_t)blockDim.x * blockDim.y) * blockIdx.x) + blockDim.x * threadIdx.y + threadIdx.x;
    if (seq < (size_t)num_occupied * num_virtual * (size_t)num_occupied * num_virtual) {
        const int ia = seq / (num_occupied * num_virtual);
        const int jb = seq % (num_occupied * num_virtual);
        const int i = ia / num_virtual;
        const int a = ia % num_virtual;
        const int j = jb / num_virtual;
        const int b = jb % num_virtual;

        const double iajb = g_eri_mo[ovov2seq(i, a, j, b, num_occupied, num_virtual)];
        const double jaib = g_eri_mo[ovov2seq(j, a, i, b, num_occupied, num_virtual)];
        tmp = iajb * (2 * iajb - jaib) / (g_eps[i] + g_eps[j] - g_eps[num_occupied + a] - g_eps[num_occupied + b]);
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        tmp += __shfl_down_sync(FULLMASK, tmp, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(&s_tmp, tmp);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_energy_second, s_tmp);
    }
}
/**/


double mp2_from_aoeri_via_required_moeri(
    double* d_eri_ao,
    const double* d_coefficient_matrix,
    const double* d_orbital_energies,
    int num_orbitals, int num_occupied)
{
    double* d_eri_tmp;
    tracked_cudaMalloc(&d_eri_tmp, sizeof(double) * num_occupied * (size_t)num_orbitals * num_orbitals * num_orbitals);
    if(!d_eri_tmp){ THROW_EXCEPTION("cudaMalloc failed for d_eri_tmp."); }
    const int num_virtual = num_orbitals - num_occupied;

    {
        std::string str = "Computing AO -> MO (ia|jb) integral transformation... ";
        PROFILE_ELAPSED_TIME(str);

        // AO ERIs (d_eri_ao) will be overwritten with (ia|jb) MO ERIs (d_eri_mo_ovov)
        transform_eri_ao2mo_dgemm_ovov(d_eri_ao, d_eri_tmp, d_coefficient_matrix, num_occupied, num_virtual);
        cudaDeviceSynchronize();
    }
    double* d_eri_mo_ovov = d_eri_ao;
    tracked_cudaFree(d_eri_tmp);

    size_t total = (size_t)num_occupied * num_virtual * num_occupied * num_virtual;

    double* d_E = nullptr;
    tracked_cudaMalloc((void**)&d_E, sizeof(double));
    cudaMemset(d_E, 0, sizeof(double));

    const int num_threads_per_warp = 32;
    const int num_warps_per_block = 32;
    const int num_threads_per_block = num_threads_per_warp * num_warps_per_block;
    const size_t num_blocks = (total + num_threads_per_block - 1) / num_threads_per_block;
    dim3 blocks(num_blocks);
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    {
        std::string str = "Computing MP2 energy from (ia|jb) MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            // CPU fallback: MP2 from (ia|jb) stored MO ERI
            auto ovov2seq_cpu = [&](int i, int a, int j, int b) -> size_t {
                return (((size_t)i * num_virtual + a) * num_occupied + j) * num_virtual + b;
            };
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t seq = 0; seq < total; seq++) {
                int ia = (int)(seq / (num_occupied * num_virtual));
                int jb = (int)(seq % (num_occupied * num_virtual));
                int i = ia / num_virtual, a = ia % num_virtual;
                int j = jb / num_virtual, b = jb % num_virtual;
                double iajb = d_eri_mo_ovov[ovov2seq_cpu(i, a, j, b)];
                double jaib = d_eri_mo_ovov[ovov2seq_cpu(j, a, i, b)];
                cpu_E += iajb * (2.0 * iajb - jaib) / (d_orbital_energies[i] + d_orbital_energies[j] - d_orbital_energies[num_occupied + a] - d_orbital_energies[num_occupied + b]);
            }
            *d_E = cpu_E;
        } else {
            mp2_stored_kernel_ovov<<<blocks, threads>>>(d_E, d_eri_mo_ovov, d_orbital_energies, num_occupied, num_virtual);
            cudaDeviceSynchronize();
        }
    }

    double h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "h_E: " << std::setprecision(12) << h_E << std::endl;

    tracked_cudaFree(d_E);

    return h_E;
}





/////////////////////////// MP2 energy calculation 


real_t ERI_Stored_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    const int num_occ = rhf_.get_num_electrons() / 2;
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    real_t* d_eri = eri_matrix_.device_ptr();

#ifndef GANSU_CPU_ONLY
    if (!gpu::gpu_available()) {
#endif
        // === CPU MP2 via O(N^5) quarter-transform AO→MO ===
        // Standard 4-step scheme: contract one C-index at a time so each
        // step is O(N^5) instead of the O(N^8) naive loop.
        //   Step 1: T1[nu,la,si, i]   = sum_mu C[mu,i] * (mu nu|la si)
        //   Step 2: T2[la,si, i,a]    = sum_nu C[nu,a+nocc] * T1[nu,la,si, i]
        //   Step 3: T3[si, i,a, j]    = sum_la C[la,j] * T2[la,si, i,a]
        //   Step 4: G[i,a,j,b]        = sum_si C[si,b+nocc] * T3[si, i,a, j]
        // Then accumulate E_MP2 = sum_{iajb} (ia|jb)*(2(ia|jb)-(ib|ja))/denom.
        const int N = num_basis;
        const int nocc = num_occ;
        const int nvir = N - nocc;
        const size_t N2 = (size_t)N * N;
        const size_t N3 = (size_t)N * N * N;

        // Step 1: contract mu.  T1 is (N × N × N × nocc) with layout
        //   T1[((nu*N + la)*N + si)*nocc + i]
        std::vector<double> T1((size_t)N3 * nocc, 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int nu = 0; nu < N; nu++) {
            for (int la = 0; la < N; la++) {
                for (int si = 0; si < N; si++) {
                    double* out = &T1[(((size_t)nu * N + la) * N + si) * nocc];
                    for (int mu = 0; mu < N; mu++) {
                        const double val = d_eri[((size_t)mu * N + nu) * N2 + (size_t)la * N + si];
                        if (val == 0.0) continue;
                        const double* Cmu = &d_C[mu * N];
                        for (int i = 0; i < nocc; i++) {
                            out[i] += Cmu[i] * val;
                        }
                    }
                }
            }
        }

        // Step 2: contract nu over virtual MOs.  T2 is (N × N × nocc × nvir)
        //   T2[((la*N + si)*nocc + i)*nvir + a]
        std::vector<double> T2((size_t)N2 * nocc * nvir, 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int la = 0; la < N; la++) {
            for (int si = 0; si < N; si++) {
                for (int i = 0; i < nocc; i++) {
                    double* out = &T2[(((size_t)la * N + si) * nocc + i) * nvir];
                    for (int nu = 0; nu < N; nu++) {
                        const double t1 = T1[(((size_t)nu * N + la) * N + si) * nocc + i];
                        if (t1 == 0.0) continue;
                        const double* Cnu_vir = &d_C[nu * N + nocc];
                        for (int a = 0; a < nvir; a++) {
                            out[a] += Cnu_vir[a] * t1;
                        }
                    }
                }
            }
        }
        // T1 no longer needed.
        std::vector<double>().swap(T1);

        // Step 3: contract la over occupied MOs.  T3 is (N × nocc × nvir × nocc)
        //   T3[((si*nocc + i)*nvir + a)*nocc + j]
        std::vector<double> T3((size_t)N * nocc * nvir * nocc, 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int si = 0; si < N; si++) {
            for (int i = 0; i < nocc; i++) {
                for (int a = 0; a < nvir; a++) {
                    double* out = &T3[(((size_t)si * nocc + i) * nvir + a) * nocc];
                    for (int la = 0; la < N; la++) {
                        const double t2 = T2[(((size_t)la * N + si) * nocc + i) * nvir + a];
                        if (t2 == 0.0) continue;
                        const double* Cla_occ = &d_C[la * N];
                        for (int j = 0; j < nocc; j++) {
                            out[j] += Cla_occ[j] * t2;
                        }
                    }
                }
            }
        }
        std::vector<double>().swap(T2);

        // Step 4: contract si over virtual MOs.  G[(i*nvir+a)*nocc*nvir + j*nvir + b] = (ia|jb)
        std::vector<double> G((size_t)nocc * nvir * nocc * nvir, 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < nocc; i++) {
            for (int a = 0; a < nvir; a++) {
                for (int j = 0; j < nocc; j++) {
                    double* out = &G[(((size_t)i * nvir + a) * nocc + j) * nvir];
                    for (int si = 0; si < N; si++) {
                        const double t3 = T3[(((size_t)si * nocc + i) * nvir + a) * nocc + j];
                        if (t3 == 0.0) continue;
                        const double* Csi_vir = &d_C[si * N + nocc];
                        for (int b = 0; b < nvir; b++) {
                            out[b] += Csi_vir[b] * t3;
                        }
                    }
                }
            }
        }
        std::vector<double>().swap(T3);

        // Step 5: accumulate MP2 energy.
        // E_MP2 = sum_{iajb} (ia|jb) * [2*(ia|jb) - (ib|ja)] / (eps_i + eps_j - eps_a - eps_b)
        // Index convention: G[((i*nvir+a)*nocc+j)*nvir + b]
        double E_MP2 = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:E_MP2) schedule(static)
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                const double ei_plus_ej = d_eps[i] + d_eps[j];
                for (int a = 0; a < nvir; a++) {
                    const double ea = d_eps[nocc + a];
                    for (int b = 0; b < nvir; b++) {
                        const double eb = d_eps[nocc + b];
                        const double iajb = G[(((size_t)i * nvir + a) * nocc + j) * nvir + b];
                        const double ibja = G[(((size_t)i * nvir + b) * nocc + j) * nvir + a];
                        const double denom = ei_plus_ej - ea - eb;
                        E_MP2 += iajb * (2.0 * iajb - ibja) / denom;
                    }
                }
            }
        }
        return E_MP2;
#ifndef GANSU_CPU_ONLY
    }
#endif

    // === GPU path ===
    real_t E_MP2 = mp2_from_aoeri_via_required_moeri(d_eri, d_C, d_eps, num_basis, num_occ);

//    if(fabs(E_MP2_naive - E_MP2_stored) > 1e-8){
//        std::cerr << "Warning: MP2 energy mismatch between naive and stored MOERI methods." << std::endl;
//        std::cerr << "  E_MP2_naive  = " << E_MP2_naive << std::endl;
//        std::cerr << "  E_MP2_stored = " << E_MP2_stored << std::endl;
//    }

    return E_MP2;
}


// ============================================================
//  MP2 Effective Densities for Gradient Calculation
//
//  Computes:
//    1. MO integrals (ia|jb) and T2 amplitudes
//    2. Unrelaxed MP2 density: P_oo, P_vv (via DGEMM)
//    3. Z-vector Lagrangian L_ai
//    4. Z-vector solve via CPHF
//    5. Relaxed density → AO basis → P_eff
//    6. Energy-weighted density → W_eff
//    7. 2-PDM contribution → Gamma_eff
// ============================================================
void ERI_Stored_RHF::compute_mp2_effective_densities(
    real_t* d_P_eff, real_t* d_W_eff, real_t* d_Gamma_eff, real_t* d_P_2el)
{
    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const int nmo = nao;
    const size_t nao2 = (size_t)nao * nao;
    const size_t novov = (size_t)nocc * nvir * nocc * nvir;
    const double zero = 0.0;

    const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();
    real_t* d_eri_ao = eri_matrix_.device_ptr();

    std::cout << "  [MP2 Gradient] Computing effective densities..." << std::endl;
    std::cout << "    nao=" << nao << " nocc=" << nocc << " nvir=" << nvir << std::endl;

    // -------------------------------------------------------
    // Step 1: OVOV MO integrals
    //   compute_mp2_energy() has already been called and overwrote
    //   eri_matrix_ with (ia|jb) OVOV integrals in the first novov elements.
    //   We just copy them — no re-transformation needed.
    // -------------------------------------------------------
    real_t* d_eri_ovov = nullptr;
    gansu::tracked_cudaMalloc(&d_eri_ovov, novov * sizeof(real_t));
    cudaMemcpy(d_eri_ovov, d_eri_ao, novov * sizeof(real_t), cudaMemcpyDeviceToDevice);

    // -------------------------------------------------------
    // Step 2: T2 amplitudes and T2_tilde
    //   t_{ij}^{ab} = (ia|jb) / D_{ijab}
    //   t̃_{ij}^{ab} = 2*t - t(swap a,b)
    // -------------------------------------------------------
    real_t* d_T2 = nullptr;
    real_t* d_T2tilde = nullptr;
    gansu::tracked_cudaMalloc(&d_T2, novov * sizeof(real_t));
    gansu::tracked_cudaMalloc(&d_T2tilde, novov * sizeof(real_t));

    // Copy orbital energies to host for denominator computation
    std::vector<real_t> h_eps(nmo);
    cudaMemcpy(h_eps.data(), d_eps, nmo * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Compute T2 and T2tilde on host (small enough for STO-3G/cc-pVDZ)
    std::vector<real_t> h_ovov(novov), h_T2(novov), h_T2tilde(novov);
    cudaMemcpy(h_ovov.data(), d_eri_ovov, novov * sizeof(real_t), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nocc; i++) {
        for (int a = 0; a < nvir; a++) {
            for (int j = 0; j < nocc; j++) {
                for (int b = 0; b < nvir; b++) {
                    const size_t idx = ((size_t)i * nvir + a) * nocc * nvir + (size_t)j * nvir + b;
                    const size_t idx_swap = ((size_t)i * nvir + b) * nocc * nvir + (size_t)j * nvir + a;
                    const real_t D = h_eps[i] + h_eps[j] - h_eps[nocc + a] - h_eps[nocc + b];
                    h_T2[idx] = h_ovov[idx] / D;
                    h_T2tilde[idx] = 2.0 * h_ovov[idx] / D - h_ovov[idx_swap] / D;
                }
            }
        }
    }
    cudaMemcpy(d_T2, h_T2.data(), novov * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T2tilde, h_T2tilde.data(), novov * sizeof(real_t), cudaMemcpyHostToDevice);

    // Diagnostic: T2 and OVOV norms
    {
        double t2_norm = 0, t2t_norm = 0, ovov_norm = 0;
        for (auto v : h_T2) t2_norm += v * v;
        for (auto v : h_T2tilde) t2t_norm += v * v;
        for (auto v : h_ovov) ovov_norm += v * v;
        std::cout << "    |T2| = " << std::sqrt(t2_norm)
                  << ", |T2tilde| = " << std::sqrt(t2t_norm)
                  << ", |OVOV| = " << std::sqrt(ovov_norm) << std::endl;
    }

    // -------------------------------------------------------
    // Step 3: Unrelaxed MP2 density (MO basis)
    //   P_ij = -2 Σ_{kab} T̃_{ik}^{ab} T_{jk}^{ab}
    //   P_ab =  2 Σ_{ijc} T̃_{ij}^{ac} T_{ij}^{bc}
    //
    //   P_oo(i,j) = -Σ_{kab} T̃(i,a,k,b) * T(j,a,k,b)
    //   P_vv(a,b) =  Σ_{ijc} T̃(i,a,j,c) * T(i,b,j,c)
    //
    //   T2 layout: T2[i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]
    //   Reshape as mat(nocc, nvir*nocc*nvir) for P_oo via DGEMM.
    // -------------------------------------------------------
    real_t* d_P_oo = nullptr;
    gansu::tracked_cudaMalloc(&d_P_oo, (size_t)nocc * nocc * sizeof(real_t));

    // P_oo = -1 * T̃_mat × T_mat^T, where both are (nocc, nvir*nocc*nvir) in row-major
    // cuBLAS col-major: P_cm(nocc,nocc) = -1 * T_cm^T(nocc,N) × T̃_cm(N,nocc)
    {
        const int N = (int)(nvir * nocc * nvir);
        if (gpu::gpu_available()) {
            const double minus_one = -1.0;
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        nocc, nocc, N,
                        &minus_one, d_T2, N,
                        d_T2tilde, N,
                        &zero, d_P_oo, nocc);
        } else {
            // Row-major: P_oo(nocc,nocc) = -1 * T̃(nocc,N) * T2(nocc,N)^T
            gpu::matrixMatrixProductRect(d_T2tilde, d_T2, d_P_oo,
                                         nocc, nocc, N,
                                         false, true, false, -1.0);
        }
    }

    // P_vv(a,b) = Σ_{ijc} T̃(i,a,j,c) * T(i,b,j,c)
    // Compute on host (requires permuted layout for DGEMM)
    std::vector<real_t> h_P_vv(nvir * nvir, 0.0);
    for (int a = 0; a < nvir; a++) {
        for (int b = 0; b < nvir; b++) {
            real_t sum = 0.0;
            for (int i = 0; i < nocc; i++) {
                for (int j = 0; j < nocc; j++) {
                    for (int c = 0; c < nvir; c++) {
                        const size_t idx_ac = ((size_t)i * nvir + a) * nocc * nvir + (size_t)j * nvir + c;
                        const size_t idx_bc = ((size_t)i * nvir + b) * nocc * nvir + (size_t)j * nvir + c;
                        sum += h_T2tilde[idx_ac] * h_T2[idx_bc];
                    }
                }
            }
            h_P_vv[a * nvir + b] = sum;
        }
    }
    real_t* d_P_vv = nullptr;
    gansu::tracked_cudaMalloc(&d_P_vv, (size_t)nvir * nvir * sizeof(real_t));
    cudaMemcpy(d_P_vv, h_P_vv.data(), nvir * nvir * sizeof(real_t), cudaMemcpyHostToDevice);

    // Diagnostic: print P_oo and P_vv elements for PySCF comparison
    {
        std::vector<real_t> h_Poo(nocc * nocc);
        cudaMemcpy(h_Poo.data(), d_P_oo, nocc * nocc * sizeof(real_t), cudaMemcpyDeviceToHost);
        double poo_norm = 0, pvv_norm = 0;
        for (auto v : h_Poo) poo_norm += v * v;
        for (auto v : h_P_vv) pvv_norm += v * v;
        std::cout << "    |P_oo| = " << std::sqrt(poo_norm) << ", |P_vv| = " << std::sqrt(pvv_norm) << std::endl;
        std::cout << "    P_oo diag:";
        for (int i = 0; i < nocc; i++) std::cout << " " << h_Poo[i * nocc + i];
        std::cout << std::endl;
        std::cout << "    P_vv diag:";
        for (int a = 0; a < nvir; a++) std::cout << " " << h_P_vv[a * nvir + a];
        std::cout << std::endl;
    }
    std::cout << "    Unrelaxed density computed." << std::endl;

    // -------------------------------------------------------
    // Step 4: Z-vector equation (canonical HF)
    //   Xvo(a,i) = vhf_mo(a,i) + I(i,a) - I(a,i)
    //   where vhf = (2J-K) response from unrelaxed density, scaled by 2
    //   z(a,i) = Xvo(a,i) / (ε_a - ε_i)
    // -------------------------------------------------------
    // Copy P_oo to host (needed here and in Step 5)
    std::vector<real_t> h_P_oo(nocc * nocc);
    cudaMemcpy(h_P_oo.data(), d_P_oo, nocc * nocc * sizeof(real_t), cudaMemcpyDeviceToHost);

    // Build unrelaxed density in AO for vhf computation
    std::vector<real_t> h_dm1_unrelaxed_MO(nmo * nmo, 0.0);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            h_dm1_unrelaxed_MO[i * nmo + j] = h_P_oo[i * nocc + j] + h_P_oo[j * nocc + i];
    for (int a = 0; a < nvir; a++)
        for (int b = 0; b < nvir; b++)
            h_dm1_unrelaxed_MO[(a + nocc) * nmo + (b + nocc)] = h_P_vv[a * nvir + b] + h_P_vv[b * nvir + a];

    // Compute vhf_MO(a,i) on host using full MO ERI.
    // AO ERI was overwritten by MP2, so rebuild full MO ERI from scratch.
    // For small molecules this is acceptable (nmo^4 storage).
    rhf_.get_coefficient_matrix().toHost();
    const real_t* h_C = rhf_.get_coefficient_matrix().host_ptr();

    // Rebuild AO ERI on host from scratch using int2e computation
    // Actually, use get_eri_matrix_host which may have AO ERI cached
    // Since eri_matrix_ is overwritten, we need to recompute AO ERI.
    // For now, compute full MO ERI via a fresh AO→MO transform.
    // Precompute the AO ERI again (stored in eri_matrix_)
    precomputation();

    // Now eri_matrix_ has fresh AO ERI. Build full MO ERI.
    real_t* d_eri_mo_full = build_mo_eri(d_C, nmo);
    std::vector<real_t> h_eri_mo(nmo * nmo * nmo * nmo);
    cudaMemcpy(h_eri_mo.data(), d_eri_mo_full, h_eri_mo.size() * sizeof(real_t), cudaMemcpyDeviceToHost);
    gansu::tracked_cudaFree(d_eri_mo_full);

    // Compute vhf_MO(p,q) = Σ_{rs} dm1(r,s) * [2*(pq|rs) - (ps|rq)]
    // dm1 only has oo and vv blocks
    std::vector<real_t> h_vhf_MO(nmo * nmo, 0.0);
    auto eri4 = [&](int p, int q, int r, int s) -> real_t {
        return h_eri_mo[((size_t)p * nmo + q) * nmo * nmo + (size_t)r * nmo + s];
    };
    for (int p = 0; p < nmo; p++) {
        for (int q = 0; q < nmo; q++) {
            real_t val = 0.0;
            // oo block
            for (int j = 0; j < nocc; j++)
                for (int k = 0; k < nocc; k++) {
                    real_t dm = h_dm1_unrelaxed_MO[j * nmo + k];
                    if (fabs(dm) < 1e-15) continue;
                    val += dm * (2.0 * eri4(p, q, j, k) - eri4(p, k, j, q));
                }
            // vv block
            for (int c = 0; c < nvir; c++)
                for (int d = 0; d < nvir; d++) {
                    real_t dm = h_dm1_unrelaxed_MO[(c+nocc) * nmo + (d+nocc)];
                    if (fabs(dm) < 1e-15) continue;
                    val += dm * (2.0 * eri4(p, q, c+nocc, d+nocc) - eri4(p, d+nocc, c+nocc, q));
                }
            h_vhf_MO[p * nmo + q] = val;  // (2J-K) convention — will be halved for CPHF
        }
    }

    // I_oo(i,j) = Σ_{akb} T̃(i,a,k,b) × (ja|kb)  [OVOV integrals]
    // I_vv(a,b) = Σ_{ijc} T̃(i,a,j,c) × (ib|jc)
    // Used in both W matrix (Step 6) and Z-vector RHS.
    std::vector<real_t> h_I_oo(nocc * nocc, 0.0);
    std::vector<real_t> h_I_vv(nvir * nvir, 0.0);
    for (int i = 0; i < nocc; i++) {
        for (int j = 0; j < nocc; j++) {
            real_t sum = 0.0;
            for (int a = 0; a < nvir; a++)
                for (int k = 0; k < nocc; k++)
                    for (int b = 0; b < nvir; b++) {
                        const size_t idx_iakb = ((size_t)i * nvir + a) * nocc * nvir + (size_t)k * nvir + b;
                        const size_t idx_jakb = ((size_t)j * nvir + a) * nocc * nvir + (size_t)k * nvir + b;
                        sum += h_T2tilde[idx_iakb] * h_ovov[idx_jakb];
                    }
            h_I_oo[i * nocc + j] = sum;
        }
    }
    for (int a = 0; a < nvir; a++) {
        for (int b = 0; b < nvir; b++) {
            real_t sum = 0.0;
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    for (int c = 0; c < nvir; c++) {
                        const size_t idx_iajc = ((size_t)i * nvir + a) * nocc * nvir + (size_t)j * nvir + c;
                        const size_t idx_ibjc = ((size_t)i * nvir + b) * nocc * nvir + (size_t)j * nvir + c;
                        sum += h_T2tilde[idx_iajc] * h_ovov[idx_ibjc];
                    }
            h_I_vv[a * nvir + b] = sum;
        }
    }

    // Coupled-Perturbed HF (CPHF) Z-vector solve.
    // Full equation: (diag(ε_a - ε_i) + A) * z = -Xvo
    // where A_{ai,bj} = 4*(ai|bj) - (ab|ij) - (aj|bi) is the RHF orbital Hessian.
    // Xvo(a,i) = vhf_MO(a+nocc, i)
    // Note: The T2-derived Lagrangian (L1, L2) enters through the 2-RDM
    // (sep(P_unrelaxed) + Gamma^T2), NOT through the CPHF RHS.
    const int nvo = nvir * nocc;
    std::vector<real_t> h_z(nvo, 0.0);

    // Build RHS: Xvo = 0.5 * vhf_MO(vo) [J - 0.5K convention, matching PySCF]
    // vhf_MO is (2J-K); PySCF uses (J-0.5K) = 0.5*(2J-K)
    std::vector<real_t> h_Xvo(nvo, 0.0);
    for (int a = 0; a < nvir; a++)
        for (int i = 0; i < nocc; i++)
            h_Xvo[a * nocc + i] = 0.5 * h_vhf_MO[(a + nocc) * nmo + i];

    // Build M = diag(ε_a - ε_i) + A (CPHF matrix, PySCF convention)
    // A_{ai,bj} = 2*(a+n,i|b+n,j) - 0.5*(a+n,b+n|i,j) - 0.5*(a+n,j|b+n,i)
    // This matches PySCF's (J - 0.5K) convention for the orbital Hessian.
    std::vector<real_t> h_M(nvo * nvo, 0.0);
    for (int a = 0; a < nvir; a++) {
        for (int i = 0; i < nocc; i++) {
            int ai = a * nocc + i;
            for (int b = 0; b < nvir; b++) {
                for (int j = 0; j < nocc; j++) {
                    int bj = b * nocc + j;
                    real_t A_val = 2.0 * eri4(a + nocc, i, b + nocc, j)
                                 - 0.5 * eri4(a + nocc, b + nocc, i, j)
                                 - 0.5 * eri4(a + nocc, j, b + nocc, i);
                    // Column-major: M[ai, bj] stored at h_M[ai + bj*nvo]
                    h_M[ai + bj * nvo] = A_val;
                    if (ai == bj)
                        h_M[ai + bj * nvo] += (h_eps[nocc + a] - h_eps[i]);
                }
            }
        }
    }

    // Solve M * z = -Xvo using Eigen
    {
        Eigen::Map<Eigen::MatrixXd> M(h_M.data(), nvo, nvo);
        Eigen::VectorXd rhs(nvo);
        for (int k = 0; k < nvo; k++) rhs(k) = -h_Xvo[k];
        Eigen::VectorXd z = M.colPivHouseholderQr().solve(rhs);
        h_z.resize(nvo);
        for (int k = 0; k < nvo; k++) h_z[k] = z(k);
    }

    {
        double z_norm = 0;
        for (auto v : h_z) z_norm += v * v;
        std::cout << "    |z| (CPHF) = " << std::sqrt(z_norm) << " (PySCF: 0.000784)" << std::endl;
    }

    // -------------------------------------------------------
    // Step 5: Build relaxed density in MO basis → transform to AO
    //   dm1mo: MP2 correction to 1-RDM (no factor 2, no HF part)
    //   P_MO = D_HF_MO + 2*dm1mo  (factor 2 for spin-sum, matching PySCF dm1p)
    // -------------------------------------------------------
    // First build dm1mo (used for W matrix, factor-1)
    std::vector<real_t> h_dm1mo(nmo * nmo, 0.0);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            h_dm1mo[i * nmo + j] = h_P_oo[i * nocc + j] + h_P_oo[j * nocc + i];
    for (int a = 0; a < nvir; a++)
        for (int b = 0; b < nvir; b++)
            h_dm1mo[(a + nocc) * nmo + (b + nocc)] = h_P_vv[a * nvir + b] + h_P_vv[b * nvir + a];
    for (int a = 0; a < nvir; a++)
        for (int i = 0; i < nocc; i++) {
            h_dm1mo[(a + nocc) * nmo + i] = h_z[a * nocc + i];
            h_dm1mo[i * nmo + (a + nocc)] = h_z[a * nocc + i];
        }

    // Diagnostic: dm1mo norm (PySCF: 0.04174 with z-vector)
    {
        double dm1_norm = 0;
        for (auto v : h_dm1mo) dm1_norm += v * v;
        std::cout << "    |dm1mo| = " << std::sqrt(dm1_norm) << " (PySCF: 0.02903)" << std::endl;
    }

    // Build P_MO = D_HF_MO + dm1mo (factor 1, for 1-el gradient)
    // PySCF: dm1 = D_HF + dm1_ao (1-el uses this)
    std::vector<real_t> h_P_MO(nmo * nmo, 0.0);
    for (int i = 0; i < nocc; i++) h_P_MO[i * nmo + i] = 2.0;
    for (int p = 0; p < nmo; p++)
        for (int q = 0; q < nmo; q++)
            h_P_MO[p * nmo + q] += h_dm1mo[p * nmo + q];

    // Transform P_MO to AO: P_AO = C × P_MO × C^T
    real_t* d_P_MO = nullptr;
    real_t* d_tmp = nullptr;
    gansu::tracked_cudaMalloc(&d_P_MO, nao2 * sizeof(real_t));
    gansu::tracked_cudaMalloc(&d_tmp, nao2 * sizeof(real_t));
    cudaMemcpy(d_P_MO, h_P_MO.data(), nao2 * sizeof(real_t), cudaMemcpyHostToDevice);

    // P_AO = C × P_MO × C^T (row-major)
    // cuBLAS: tmp = P_MO_cm × C_cm = (C_rm^T × P_rm^T)  -- but we need C × P × C^T
    // In row-major: tmp = C × P_MO (nao × nmo × nmo × nmo = nao × nmo)
    // Then P_AO = tmp × C^T
    // cuBLAS col-major: C_rm → C^T_cm, P_rm → P^T_cm
    // tmp_cm = C^T_cm × P^T_cm = (P × C^T)^T_cm   ... complex
    // Simpler: use gpu::matrixMatrixProduct which handles row-major
    gpu::matrixMatrixProduct(d_C, d_P_MO, d_tmp, nao, false, false);  // tmp = C × P_MO
    gpu::matrixMatrixProduct(d_tmp, d_C, d_P_eff, nao, false, true);  // P_eff = tmp × C^T

    // Build unrelaxed density for 2-electron term (NO z-vector)
    // PySCF's make_rdm2() uses unrelaxed dm1 for the separable 2-PDM.
    // The z-vector response enters only through 1-el and overlap terms.
    {
        std::vector<real_t> h_P_MO_2el(nmo * nmo, 0.0);
        for (int i = 0; i < nocc; i++) h_P_MO_2el[i * nmo + i] = 2.0;
        // Add oo and vv blocks (unrelaxed), but NOT ov block (z-vector)
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                h_P_MO_2el[i * nmo + j] += h_P_oo[i * nocc + j] + h_P_oo[j * nocc + i];
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++)
                h_P_MO_2el[(a + nocc) * nmo + (b + nocc)] += h_P_vv[a * nvir + b] + h_P_vv[b * nvir + a];
        cudaMemcpy(d_P_MO, h_P_MO_2el.data(), nao2 * sizeof(real_t), cudaMemcpyHostToDevice);
        gpu::matrixMatrixProduct(d_C, d_P_MO, d_tmp, nao, false, false);
        gpu::matrixMatrixProduct(d_tmp, d_C, d_P_2el, nao, false, true);
    }

    std::cout << "    Relaxed density in AO basis." << std::endl;

    // -------------------------------------------------------
    // Step 6: Energy-weighted density W
    //   W_MO(p,q) = (ε_p + ε_q)/2 × P_total(p,q)
    //   where P_total = D_HF (2*δ_ij) + dm1mo (MP2 correction).
    // -------------------------------------------------------
    std::vector<real_t> h_W_MO(nmo * nmo, 0.0);

    // HF energy-weighted density: W_HF(i,i) = 2*ε_i
    for (int i = 0; i < nocc; i++)
        h_W_MO[i * nmo + i] = 2.0 * h_eps[i];

    // MP2 correction: ε_q × dm1mo(p,q)
    for (int p = 0; p < nmo; p++)
        for (int q = 0; q < nmo; q++)
            h_W_MO[p * nmo + q] += h_eps[q] * h_dm1mo[p * nmo + q];

    // Note: Lagrangian I terms (I_oo, I_vv) are NOT added to W here.
    // They enter the gradient through the 2-RDM (cross term in sep(P_unrelaxed))
    // rather than through the overlap derivative. Adding them to W would
    // double-count their contribution.
    // TODO: Verify this against PySCF's exact dme0 construction.

    // Symmetrize: W = (W + W^T) / 2
    for (int p = 0; p < nmo; p++)
        for (int q = p + 1; q < nmo; q++) {
            real_t avg = 0.5 * (h_W_MO[p * nmo + q] + h_W_MO[q * nmo + p]);
            h_W_MO[p * nmo + q] = avg;
            h_W_MO[q * nmo + p] = avg;
        }

    real_t* d_W_MO = nullptr;
    gansu::tracked_cudaMalloc(&d_W_MO, nao2 * sizeof(real_t));
    cudaMemcpy(d_W_MO, h_W_MO.data(), nao2 * sizeof(real_t), cudaMemcpyHostToDevice);

    gpu::matrixMatrixProduct(d_C, d_W_MO, d_tmp, nao, false, false);
    gpu::matrixMatrixProduct(d_tmp, d_C, d_W_eff, nao, false, true);

    // -------------------------------------------------------
    // Step 7: Non-separable 2-PDM: Γ^T2_{μνλσ} = Σ_{ijab} C_{μi} C_{ν,a+n} T̃_{ij}^{ab} C_{λj} C_{σ,b+n}
    //   Build symmetrized AO 4-index density for ERI derivative contraction.
    //   Symmetrized for 8-fold ERI symmetry: (μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν) etc.
    // -------------------------------------------------------
    {
        // Get C on host
        rhf_.get_coefficient_matrix().toHost();
        const real_t* h_C = rhf_.get_coefficient_matrix().host_ptr();
        const size_t nao4 = nao2 * nao2;

        // Build Γ^T2 on host (nao^4 elements)
        std::vector<real_t> h_Gamma(nao4, 0.0);

        // Γ(μ,ν,λ,σ) = Σ_{ijab} C(μ,i) C(ν,a+nocc) T̃(i,a,j,b) C(λ,j) C(σ,b+nocc)
        // Use efficient contraction: for each (i,j), form rank-1 × T̃-contracted outer product
        for (int i = 0; i < nocc; i++) {
            for (int j = 0; j < nocc; j++) {
                // Q^{ij}(ν,σ) = Σ_{ab} T̃_{ij}^{ab} C(ν,a+nocc) C(σ,b+nocc)
                // First: t̃_ab = T̃(i,a,j,b) for this (i,j)
                for (int a = 0; a < nvir; a++) {
                    for (int b = 0; b < nvir; b++) {
                        const size_t t_idx = ((size_t)i * nvir + a) * nocc * nvir + (size_t)j * nvir + b;
                        real_t t_ab = h_T2tilde[t_idx];
                        if (fabs(t_ab) < 1e-15) continue;
                        // Add contribution to Γ: t_ab * C(μ,i)*C(ν,a+n)*C(λ,j)*C(σ,b+n)
                        for (int mu = 0; mu < nao; mu++) {
                            real_t Cmi = h_C[mu * nao + i];
                            if (fabs(Cmi) < 1e-15) continue;
                            real_t Cmi_t = Cmi * t_ab;
                            for (int nu = 0; nu < nao; nu++) {
                                real_t CnA = h_C[nu * nao + (a + nocc)];
                                if (fabs(CnA) < 1e-15) continue;
                                real_t CnA_Cmi_t = CnA * Cmi_t;
                                for (int la = 0; la < nao; la++) {
                                    real_t ClJ = h_C[la * nao + j];
                                    if (fabs(ClJ) < 1e-15) continue;
                                    real_t factor = CnA_Cmi_t * ClJ;
                                    for (int si = 0; si < nao; si++) {
                                        h_Gamma[((size_t)mu * nao + nu) * nao2 + la * nao + si]
                                            += factor * h_C[si * nao + (b + nocc)];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4-fold symmetrize for ERI symmetry: (μν|λσ)=(νμ|λσ)=(μν|σλ)=(νμ|σλ)
        std::vector<real_t> h_Gamma_sym(nao4, 0.0);
        for (int mu = 0; mu < nao; mu++) {
            for (int nu = 0; nu < nao; nu++) {
                for (int la = 0; la < nao; la++) {
                    for (int si = 0; si < nao; si++) {
                        const size_t i0 = ((size_t)mu*nao + nu)*nao2 + la*nao + si;  // (μν|λσ)
                        const size_t i1 = ((size_t)nu*nao + mu)*nao2 + la*nao + si;  // (νμ|λσ)
                        const size_t i2 = ((size_t)mu*nao + nu)*nao2 + si*nao + la;  // (μν|σλ)
                        const size_t i3 = ((size_t)nu*nao + mu)*nao2 + si*nao + la;  // (νμ|σλ)
                        h_Gamma_sym[i0] = 0.25 * (
                            h_Gamma[i0] + h_Gamma[i1] + h_Gamma[i2] + h_Gamma[i3]);
                    }
                }
            }
        }
        // gamma_4idx = Γ^T2_sym only.
        // The kernel computes sep(P_relaxed) + gamma_4idx.
        // Since P_relaxed = D_HF + ΔP (full relaxed 1-RDM), sep(P_relaxed)
        // already includes sep(D_HF) + cross(D_HF, ΔP) + sep(ΔP).
        // The full 2-RDM is sep(P_total) + Γ^T2 (cumulant), so gamma_4idx = Γ^T2.
        // No sep(ΔP) subtraction needed.
        cudaMemcpy(d_Gamma_eff, h_Gamma_sym.data(), nao4 * sizeof(real_t), cudaMemcpyHostToDevice);
    }

    // Cleanup
    gansu::tracked_cudaFree(d_eri_ovov);
    gansu::tracked_cudaFree(d_T2);
    gansu::tracked_cudaFree(d_T2tilde);
    gansu::tracked_cudaFree(d_P_oo);
    gansu::tracked_cudaFree(d_P_vv);
    gansu::tracked_cudaFree(d_P_MO);
    gansu::tracked_cudaFree(d_W_MO);
    gansu::tracked_cudaFree(d_tmp);

    std::cout << "  [MP2 Gradient] Effective densities ready." << std::endl;
}


// ============================================================
//  Direct MP2: compute MP2 energy without storing nao^4 AO ERI
//
//  Algorithm (block over occupied index j):
//    For each j-block [j_start, j_start + block_j):
//      1. Half-transform: H(mu,nu,la,j~) = sum_sigma (mu nu|la sigma) C(sigma,j~)
//         Memory: nao^3 * block_j  (GPU kernel: RysERI_half_transform)
//      2. Contract mu -> i: K(i,nu,la,j~) = sum_mu C(mu,i) * H(mu,nu,la,j~)
//         Memory: nocc * nao^2 * block_j  (cublasDgemm)
//      3. Contract nu -> a: L(i,a,la,j~) = sum_nu C(nu,a+nocc) * K(i,nu,la,j~)
//         Memory: nocc * nvir * nao * block_j  (cublasDgemm)
//      4. Contract la -> b: V(i,a,b,j~) = sum_la C(la,b+nocc) * L(i,a,la,j~)
//         Memory: nocc * nvir * nvir * block_j  (cublasDgemm)
//      5. Accumulate: E += sum_{i,a,b,j} V(ia|jb)(2*V(ia|jb)-V(ib|ja)) / D_ijab
//
//  Memory peak: nao^3 * block_j (Step 1 buffer)
//  Total integral evaluations: ceil(nocc / block_j) full AO integral passes
//
// ============================================================
//  Hash ERI MP2: COO → dense OVOV → MP2 energy
// ============================================================
// COO → dense AO ERI expansion kernel
__global__ void hash_coo_to_dense_kernel(
    const unsigned long long* __restrict__ g_coo_keys,
    const double* __restrict__ g_coo_values,
    const size_t num_entries,
    double* __restrict__ g_eri_dense,
    const int nao)
{
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    const unsigned long long key = g_coo_keys[tid];
    const double val = g_coo_values[tid];
    if (fabs(val) < 1e-18) return;

    int mu = (int)((key >> 48) & 0xFFFF);
    int nu = (int)((key >> 32) & 0xFFFF);
    int la = (int)((key >> 16) & 0xFFFF);
    int si = (int)(key & 0xFFFF);

    const size_t nao2 = (size_t)nao * nao;
    // Expand 8-fold symmetry
    struct Perm { int a, b, c, d; };
    Perm perms[8] = {
        {mu,nu,la,si}, {nu,mu,la,si}, {mu,nu,si,la}, {nu,mu,si,la},
        {la,si,mu,nu}, {si,la,mu,nu}, {la,si,nu,mu}, {si,la,nu,mu}
    };
    for (int p = 0; p < 8; p++) {
        size_t idx = ((size_t)perms[p].a*nao + perms[p].b)*nao2 + perms[p].c*nao + perms[p].d;
        g_eri_dense[idx] = val;  // Direct write (no race: each permutation maps to unique index)
    }
}

// Forward declarations (hash_half_transform.cu) — in gansu namespace (same as this file)
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

real_t ERI_Hash_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();

    // === CPU fallback: build full MO ERI via AO reconstruction + stored MP2 ===
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(d_C, nao);
        real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, nao, nocc);
        tracked_cudaFree(d_mo_eri);
        return E;
    }

    // Determine block_j (same as Direct MP2)
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t nao3 = (size_t)nao * nao * nao;
    int block_j = std::max(1, (int)(free_mem * 6 / 10 / (nao3 * sizeof(real_t))));
    block_j = std::min(block_j, nocc);
    if (block_j > 8) block_j = 8;

    const char* method_name = (hash_fock_method_ == HashFockMethod::Compact) ? "Compact" :
                              (hash_fock_method_ == HashFockMethod::Indexed) ? "Indexed" : "Fullscan";
    std::cout << "  [Hash MP2 / " << method_name << "] nao=" << nao << " nocc=" << nocc
              << " nvir=" << nvir << " nnz=" << num_entries_ << " block_j=" << block_j << std::endl;

    // Allocate buffers (same as Direct MP2)
    real_t* d_half = nullptr;
    real_t* d_Ki = nullptr;
    real_t* d_Li = nullptr;
    real_t* d_V = nullptr;
    real_t* d_E = nullptr;

    tracked_cudaMalloc(&d_half, nao3 * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)nvir * nao * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_V, (size_t)nocc * nvir * nvir * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_E, sizeof(real_t));
    cudaMemset(d_E, 0, sizeof(real_t));

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta_zero = 0.0;

    for (int j_start = 0; j_start < nocc; j_start += block_j) {
        int bj = std::min(block_j, nocc - j_start);

        // Step 1: Half-transform via Hash lookup (replaces on-the-fly Rys computation)
        cudaMemset(d_half, 0, nao3 * bj * sizeof(real_t));
        if (!gpu::gpu_available()) {
            // CPU fallback: hash half-transform accumulate
            // H(m,n,l,j) += (mn|ls) * C(s,j) with 8-fold symmetry deduplication
            auto cpu_hash_accumulate = [&](int mu, int nu, int la, int si, double val) {
                if (fabs(val) < 1e-18) return;
                const size_t nao2bj = (size_t)nao * nao * bj;
                struct Perm { int m, n, l, s; };
                Perm perms[8] = {
                    {mu,nu,la,si}, {nu,mu,la,si}, {mu,nu,si,la}, {nu,mu,si,la},
                    {la,si,mu,nu}, {si,la,mu,nu}, {la,si,nu,mu}, {si,la,nu,mu}
                };
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
                    for (int jj = 0; jj < bj; jj++) {
                        double Csj = d_C[s * nao + (j_start + jj)];
                        if (fabs(Csj) < 1e-15) continue;
                        d_half[(size_t)m * nao2bj + (size_t)n * nao * bj + (size_t)l * bj + jj] += val * Csj;
                    }
                }
            };
            auto decode_key = [](unsigned long long key, int& a, int& b, int& c, int& d) {
                a = (int)((key >> 48) & 0xFFFF);
                b = (int)((key >> 32) & 0xFFFF);
                c = (int)((key >> 16) & 0xFFFF);
                d = (int)(key & 0xFFFF);
            };

            if (hash_fock_method_ == HashFockMethod::Compact) {
                for (size_t tid = 0; tid < num_entries_; tid++) {
                    int mu, nu, la, si;
                    decode_key(d_coo_keys_[tid], mu, nu, la, si);
                    cpu_hash_accumulate(mu, nu, la, si, d_coo_values_[tid]);
                }
            } else if (hash_fock_method_ == HashFockMethod::Indexed) {
                for (size_t tid = 0; tid < num_nonzero_; tid++) {
                    size_t slot = d_nonzero_indices_[tid];
                    unsigned long long key = d_hash_keys_[slot];
                    if (key == 0xFFFFFFFFFFFFFFFFULL) continue;
                    int mu, nu, la, si;
                    decode_key(key, mu, nu, la, si);
                    cpu_hash_accumulate(mu, nu, la, si, d_hash_values_[slot]);
                }
            } else { // Fullscan
                const size_t capacity = hash_capacity_mask_ + 1;
                for (size_t tid = 0; tid < capacity; tid++) {
                    unsigned long long key = d_hash_keys_[tid];
                    if (key == 0xFFFFFFFFFFFFFFFFULL) continue;
                    int mu, nu, la, si;
                    decode_key(key, mu, nu, la, si);
                    cpu_hash_accumulate(mu, nu, la, si, d_hash_values_[tid]);
                }
            }
        } else {
            const int threads = 256;
            if (hash_fock_method_ == HashFockMethod::Compact) {
                const int blocks = ((int)num_entries_ + threads - 1) / threads;
                hash_half_transform_compact_kernel<<<blocks, threads>>>(
                    d_coo_keys_, d_coo_values_, num_entries_,
                    d_C, d_half, nao, j_start, bj);
            } else if (hash_fock_method_ == HashFockMethod::Indexed) {
                const int blocks = ((int)num_nonzero_ + threads - 1) / threads;
                hash_half_transform_indexed_kernel<<<blocks, threads>>>(
                    d_hash_keys_, d_hash_values_,
                    d_nonzero_indices_, num_nonzero_,
                    d_C, d_half, nao, j_start, bj);
            } else { // Fullscan
                const size_t capacity = hash_capacity_mask_ + 1;
                const int blocks = ((int)capacity + threads - 1) / threads;
                hash_half_transform_fullscan_kernel<<<blocks, threads>>>(
                    d_hash_keys_, d_hash_values_, capacity,
                    d_C, d_half, nao, j_start, bj);
            }
            cudaDeviceSynchronize();
        }

        // Steps 2-4: DGEMM chain (identical to Direct MP2)
        {
            const size_t M_nao2bj = (size_t)nao * nao * bj;
            const size_t M_naobj = (size_t)nao * bj;

            for (int i = 0; i < nocc; i++) {
                if (gpu::gpu_available()) {
                    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        M_nao2bj, 1, nao, &alpha,
                        d_half, M_nao2bj, d_C + i, nao,
                        &beta_zero, d_Ki, M_nao2bj);

                    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        M_naobj, nvir, nao, &alpha,
                        d_Ki, M_naobj, d_C + nocc, nao,
                        &beta_zero, d_Li, M_naobj);

                    for (int a = 0; a < nvir; a++) {
                        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            bj, nvir, nao, &alpha,
                            d_Li + (size_t)a * nao * bj, bj,
                            d_C + nocc, nao,
                            &beta_zero,
                            d_V + (size_t)i * nvir * nvir * bj + (size_t)a * nvir * bj, bj);
                    }
                } else {
                    // CPU fallback: cuBLAS col-major → row-major reinterpretation.
                    // C_rm(nao, nao) row-major.  Column i of C_rm (non-contiguous) is needed.
                    // cuBLAS d_C+i with lda=nao accesses column i of C_rm.
                    // In row-major, rows nocc..nao-1 of C start at d_C + nocc*nao (contiguous, shape nvir x nao).
                    // half in memory: col-major (M_nao2bj, nao) == row-major (nao, M_nao2bj).
                    // Ki in memory:   col-major (M_nao2bj, 1)   == row-major (1, M_nao2bj).
                    // Li in memory:   col-major (M_naobj, nvir)  == row-major (nvir, M_naobj).
                    // V  in memory:   col-major (bj, nvir)       == row-major (nvir, bj).

                    // Step 2: Ki(1, M_nao2bj) = C_col_i^T(1, nao) * half_rm(nao, M_nao2bj)
                    // Extract column i of C_rm to contiguous temp buffer
                    std::vector<real_t> c_col_i(nao);
                    for (int mu = 0; mu < nao; mu++) c_col_i[mu] = d_C[mu * nao + i];
                    gpu::matrixMatrixProductRect(c_col_i.data(), d_half, d_Ki,
                        1, (int)M_nao2bj, nao, false, false, false, 1.0);

                    // Step 3: Li_rm(nvir, M_naobj) = C_vir_rm(nvir, nao) * Ki_rm(nao, M_naobj)
                    // C_vir_rm = rows nocc..nao-1 of C_rm, at d_C + nocc*nao, shape (nvir, nao)
                    gpu::matrixMatrixProductRect(d_C + (size_t)nocc * nao, d_Ki, d_Li,
                        nvir, (int)M_naobj, nao, false, false, false, 1.0);

                    // Step 4: V_rm(nvir, bj) = C_vir_rm(nvir, nao) * Li_a_rm(nao, bj)
                    for (int a = 0; a < nvir; a++) {
                        gpu::matrixMatrixProductRect(d_C + (size_t)nocc * nao,
                            d_Li + (size_t)a * nao * bj,
                            d_V + (size_t)i * nvir * nvir * bj + (size_t)a * nvir * bj,
                            nvir, bj, nao, false, false, false, 1.0);
                    }
                }
            }
        }

        // Step 5: MP2 energy accumulation
        if (!gpu::gpu_available()) {
            const size_t total_ovov = (size_t)nocc * nvir * nocc * nvir;
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t seq = 0; seq < total_ovov; seq++) {
                int ia = (int)(seq / ((size_t)nocc * nvir));
                int jb = (int)(seq % ((size_t)nocc * nvir));
                int i = ia / nvir, a = ia % nvir;
                int j = jb / nvir, b = jb % nvir;
                int jj = j - j_start;
                if (jj >= 0 && jj < bj) {
                    double iajb = d_V[(size_t)i*nvir*nvir*bj + (size_t)a*nvir*bj + (size_t)b*bj + jj];
                    double ibja = d_V[(size_t)i*nvir*nvir*bj + (size_t)b*nvir*bj + (size_t)a*bj + jj];
                    cpu_E += iajb * (2.0*iajb - ibja) / (d_eps[i] + d_eps[j] - d_eps[nocc+a] - d_eps[nocc+b]);
                }
            }
            *d_E += cpu_E;
        } else {
            mp2_stored_kernel_ovov_direct<<<(nocc*nvir*nocc*nvir + 1023)/1024, dim3(32,32)>>>(
                d_E, d_V, d_eps, nocc, nvir, j_start, bj);
            cudaDeviceSynchronize();
        }
    }

    real_t h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(real_t), cudaMemcpyDeviceToHost);
    std::cout << "h_E: " << std::setprecision(12) << h_E << std::endl;

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_V);
    tracked_cudaFree(d_E);

    return h_E;
}

//  NOTE: Uses Rys quadrature for all angular momenta (no s/p specialization).
//        TODO: GPU最適化 — s/p特化カーネルの追加、block_jの自動調整
// ============================================================
real_t ERI_Direct_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();
    const int nao = rhf_.get_num_basis();
    const int nocc = rhf_.get_num_electrons() / 2;
    const int nvir = nao - nocc;
    const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();

    // === CPU fallback: reconstruct full AO ERI + stored MP2 path ===
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(d_C, nao);
        real_t E = mp2_from_full_moeri(d_mo_eri, d_C, d_eps, nao, nocc);
        tracked_cudaFree(d_mo_eri);
        return E;
    }

    // --- Determine block size from available GPU memory ---
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t nao3 = (size_t)nao * nao * nao;
    // Reserve memory for H (nao^3 * block_j) + intermediates
    // Use at most 60% of free memory for H
    int block_j = std::max(1, (int)(free_mem * 6 / 10 / (nao3 * sizeof(real_t))));
    block_j = std::min(block_j, nocc);
    if (block_j > 8) block_j = 8; // Cap for reasonable # of atomicAdd per integral

    std::cout << "  [Direct MP2] nao=" << nao << " nocc=" << nocc << " nvir=" << nvir
              << " block_j=" << block_j << std::endl;

    // --- Recompute unsorted Schwarz factors ---
    const auto& shell_type_infos = hf_.get_shell_type_infos();
    const auto& shell_pair_type_infos = hf_.get_shell_pair_type_infos();
    DeviceHostMemory<real_t> schwarz_unsorted(hf_.get_num_primitive_shell_pairs());
    gpu::computeSchwarzUpperBounds(
        shell_type_infos, shell_pair_type_infos,
        hf_.get_primitive_shells().device_ptr(),
        hf_.get_boys_grid().device_ptr(),
        hf_.get_cgto_normalization_factors().device_ptr(),
        schwarz_unsorted.device_ptr(), false);

    // --- Allocate buffers ---
    // Steps 2-3-4 are fused per occupied orbital i to minimize memory.
    real_t* d_half = nullptr;    // H(mu,nu,la,j~): nao^3 * block_j
    real_t* d_Ki = nullptr;      // K_i(nu,la,j~):  nao^2 * block_j  (per i)
    real_t* d_Li = nullptr;      // L_i(a,la,j~):   nvir * nao * block_j  (per i)
    real_t* d_V = nullptr;       // V(i,a,b,j~):    nocc * nvir * nvir * block_j
    real_t* d_E = nullptr;       // scalar energy accumulator

    tracked_cudaMalloc(&d_half, nao3 * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_Ki, (size_t)nao * nao * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_Li, (size_t)nvir * nao * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_V, (size_t)nocc * nvir * nvir * block_j * sizeof(real_t));
    tracked_cudaMalloc(&d_E, sizeof(real_t));
    cudaMemset(d_E, 0, sizeof(real_t));

    cublasHandle_t handle = gpu::GPUHandle::cublas();
    const double alpha = 1.0, beta_zero = 0.0;

    // --- Main loop over j-blocks ---
    for (int j_start = 0; j_start < nocc; j_start += block_j) {
        int bj = std::min(block_j, nocc - j_start);

        // Step 1: Half-transform H(mu,nu,la,j~) = sum_sigma (mu nu|la sigma) C(sigma,j~)
        cudaMemset(d_half, 0, nao3 * bj * sizeof(real_t));
        gpu::computeHalfTransformedERI(
            shell_type_infos, shell_pair_type_infos,
            hf_.get_primitive_shells().device_ptr(),
            hf_.get_boys_grid().device_ptr(),
            hf_.get_cgto_normalization_factors().device_ptr(),
            d_half, schwarz_unsorted.device_ptr(),
            hf_.get_schwarz_screening_threshold(),
            nao, d_C, j_start, bj);

        // Steps 2-3-4 fused per occupied orbital i (minimizes memory).
        // Step 2: K_i(nu,la,j~) = sum_mu C(mu,i) * H(mu,nu,la,j~)     — rank-1 contraction
        // Step 3: L_i(a,la,j~) = sum_nu C_vir(nu,a) * K_i(nu,la,j~)   — DGEMM
        // Step 4: V_i(a,b,j~) = sum_la C_vir(la,b) * L_i(a,la,j~)     — DGEMM per a
        {
            const size_t M_nao2bj = (size_t)nao * nao * bj;
            const size_t M_naobj = (size_t)nao * bj;

            for (int i = 0; i < nocc; i++) {
                if (gpu::gpu_available()) {
                    // Step 2: K_i(nu*la*bj) = sum_mu C(mu,i) * H(mu, nu*la*bj)
                    //   Rank-1 contraction: K_i = C_col_i^T * H
                    //   cuBLAS: K_cm(nao^2*bj, 1) = H_cm(nao^2*bj, nao) * C_i_cm^T(nao, 1)
                    cublasDgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        M_nao2bj, 1, nao,
                        &alpha,
                        d_half, M_nao2bj,
                        d_C + i, nao,       // column i of C (row-major stride = nao)
                        &beta_zero,
                        d_Ki, M_nao2bj);

                    // Step 3: L_i(a, la*bj) = sum_nu C_vir(nu,a) * K_i(nu, la*bj)
                    //   cuBLAS: L_cm(nao*bj, nvir) = K_cm(nao*bj, nao) * C_vir_cm^T(nao, nvir)
                    cublasDgemm(handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        M_naobj, nvir, nao,
                        &alpha,
                        d_Ki, M_naobj,
                        d_C + nocc, nao,    // C_vir: columns nocc..nao-1
                        &beta_zero,
                        d_Li, M_naobj);

                    // Step 4: V_i(a, b, j~) = sum_la C_vir(la,b) * L_i(a, la, j~)
                    for (int a = 0; a < nvir; a++) {
                        cublasDgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            bj, nvir, nao,
                            &alpha,
                            d_Li + (size_t)a * nao * bj, bj,
                            d_C + nocc, nao,
                            &beta_zero,
                            d_V + (size_t)i * nvir * nvir * bj + (size_t)a * nvir * bj, bj);
                    }
                } else {
                    // CPU fallback: cuBLAS col-major → row-major reinterpretation.
                    // C_rm(nao, nao) row-major. half_rm(nao, M_nao2bj), Ki_rm(1, M_nao2bj),
                    // Li_rm(nvir, M_naobj), V_rm(nvir, bj).
                    // C_vir_rm = rows nocc..nao-1 of C_rm at d_C + nocc*nao, shape (nvir, nao).

                    // Step 2: Ki(1, M_nao2bj) = C_col_i^T(1, nao) * half_rm(nao, M_nao2bj)
                    // Extract column i of C_rm to contiguous temp buffer
                    std::vector<real_t> c_col_i(nao);
                    for (int mu = 0; mu < nao; mu++) c_col_i[mu] = d_C[mu * nao + i];
                    gpu::matrixMatrixProductRect(c_col_i.data(), d_half, d_Ki,
                        1, (int)M_nao2bj, nao, false, false, false, 1.0);

                    // Step 3: Li_rm(nvir, M_naobj) = C_vir_rm(nvir, nao) * Ki_rm(nao, M_naobj)
                    gpu::matrixMatrixProductRect(d_C + (size_t)nocc * nao, d_Ki, d_Li,
                        nvir, (int)M_naobj, nao, false, false, false, 1.0);

                    // Step 4: V_rm(nvir, bj) = C_vir_rm(nvir, nao) * Li_a_rm(nao, bj)
                    for (int a = 0; a < nvir; a++) {
                        gpu::matrixMatrixProductRect(d_C + (size_t)nocc * nao,
                            d_Li + (size_t)a * nao * bj,
                            d_V + (size_t)i * nvir * nvir * bj + (size_t)a * nvir * bj,
                            nvir, bj, nao, false, false, false, 1.0);
                    }
                }
            }
        }

        // Step 5: Accumulate MP2 energy
        // V(i, a, b, j~) at d_V[i*nvir*nvir*bj + a*nvir*bj + b*bj + jj]
        // (ia|jb) = V[i][a][b][j-j_start]
        // E_MP2 += (ia|jb) * (2*(ia|jb) - (ib|ja)) / (eps_i + eps_j - eps_a - eps_b)
        if (!gpu::gpu_available()) {
            const size_t total_ovov = (size_t)nocc * nvir * nocc * nvir;
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t seq = 0; seq < total_ovov; seq++) {
                int ia = (int)(seq / ((size_t)nocc * nvir));
                int jb = (int)(seq % ((size_t)nocc * nvir));
                int i_d = ia / nvir, a_d = ia % nvir;
                int j_d = jb / nvir, b_d = jb % nvir;
                int jj = j_d - j_start;
                if (jj >= 0 && jj < bj) {
                    double iajb = d_V[(size_t)i_d*nvir*nvir*bj + (size_t)a_d*nvir*bj + (size_t)b_d*bj + jj];
                    double ibja = d_V[(size_t)i_d*nvir*nvir*bj + (size_t)b_d*nvir*bj + (size_t)a_d*bj + jj];
                    cpu_E += iajb * (2.0*iajb - ibja) / (d_eps[i_d] + d_eps[j_d] - d_eps[nocc+a_d] - d_eps[nocc+b_d]);
                }
            }
            *d_E += cpu_E;
        } else {
            mp2_stored_kernel_ovov_direct<<<(nocc*nvir*nocc*nvir + 1023)/1024, dim3(32,32)>>>(
                d_E, d_V, d_eps, nocc, nvir, j_start, bj);
            cudaDeviceSynchronize();
        }

        std::cout << "  [Direct MP2] j_block " << j_start << ".." << j_start+bj-1
                  << " / " << nocc << " done" << std::endl;
    }

    // --- Read final energy ---
    real_t h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(real_t), cudaMemcpyDeviceToHost);
    std::cout << "  [Direct MP2] E_MP2 = " << std::setprecision(12) << h_E << std::endl;

    tracked_cudaFree(d_half);
    tracked_cudaFree(d_Ki);
    tracked_cudaFree(d_Li);
    tracked_cudaFree(d_V);
    tracked_cudaFree(d_E);

    return h_E;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// MP3 energy calculation (naive implementation with on-the-fly integral transformation)
__global__ void mp3_naive_4h2p_kernel(const double* __restrict__ eri_ao,
                      const double* __restrict__ C,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,k,l,a,b) with i,j,k,l in occ, a,b in vir (offset by occ)
  size_t total = (size_t)occ * occ * occ * occ * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int l  = (int)(t % occ); t /= occ;
    int k  = (int)(t % occ); t /= occ;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;

    double denom1 = eps[i] + eps[j] - eps[a] - eps[b];
    double denom2 = eps[k] + eps[l] - eps[a] - eps[b];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo_bruteforce(eri_ao, C, num_basis, i, a, j, b);
      double ikjl = eri_mo_bruteforce(eri_ao, C, num_basis, i, k, j, l);
      double kalb = eri_mo_bruteforce(eri_ao, C, num_basis, k, a, l, b);
      double kbla = eri_mo_bruteforce(eri_ao, C, num_basis, k, b, l, a);
      contrib = iajb*ikjl*(2.0*kalb-kbla) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}


__global__ void mp3_naive_2h4p_kernel(const double* __restrict__ eri_ao,
                      const double* __restrict__ C,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,a,b,c,d) with i,j in occ, a,b,c,d in vir (offset by occ)
  size_t total = (size_t)occ * occ * vir * vir * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int d_ = (int)(t % vir); t /= vir;
    int c_ = (int)(t % vir); t /= vir;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;
    int c = occ + c_;
    int d = occ + d_;

    double denom1 = eps[i] + eps[j] - eps[a] - eps[b];
    double denom2 = eps[i] + eps[j] - eps[c] - eps[d];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo_bruteforce(eri_ao, C, num_basis, i, a, j, b);
      double acbd = eri_mo_bruteforce(eri_ao, C, num_basis, a, c, b, d);
      double icjd = eri_mo_bruteforce(eri_ao, C, num_basis, i, c, j, d);
      double idjc = eri_mo_bruteforce(eri_ao, C, num_basis, i, d, j, c);
      contrib = iajb*acbd*(2.0*icjd-idjc) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}

__global__ void mp3_naive_3h3p_kernel(const double* __restrict__ eri_ao,
                      const double* __restrict__ C,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,k,a,b,c) with i,j,k in occ, a,b,c in vir (offset by occ)
  size_t total = (size_t)occ * occ * occ * vir * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int c_ = (int)(t % vir); t /= vir;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int k  = (int)(t % occ); t /= occ;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;
    int c = occ + c_;

    double denom1 = eps[i] + eps[k] - eps[a] - eps[c];
    double denom2 = eps[k] + eps[j] - eps[b] - eps[c];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo_bruteforce(eri_ao, C, num_basis, i, a, j, b);
      double ijab = eri_mo_bruteforce(eri_ao, C, num_basis, i, j, a, b);
      double kcia = eri_mo_bruteforce(eri_ao, C, num_basis, k, c, i, a);
      double kaic = eri_mo_bruteforce(eri_ao, C, num_basis, k, a, i, c);
      double kcjb = eri_mo_bruteforce(eri_ao, C, num_basis, k, c, j, b);
      double kbjc = eri_mo_bruteforce(eri_ao, C, num_basis, k, b, j, c);
      contrib = ((2.0*iajb-ijab)*(2.0*kcia-kaic)*(2.0*kcjb-kbjc) - 3.0*ijab*kaic*kbjc) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}


real_t mp3_naive(const real_t* d_eri, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_occ) {
    const int num_threads = 512; // if 1024, shared memory exceeds the limit


    real_t* d_mp3_energy;
    tracked_cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);

    // Host brute-force MO integral lambda (for CPU fallback of naive kernels)
    auto eri_bf_host = [&](const real_t* eri_ao, const real_t* C, int nb, int p, int q, int r, int s) -> double {
        double sum = 0.0;
        for (int mu = 0; mu < nb; mu++)
            for (int nu = 0; nu < nb; nu++)
                for (int la = 0; la < nb; la++)
                    for (int si = 0; si < nb; si++)
                        sum += C[(size_t)nb*mu+p] * C[(size_t)nb*nu+q] * C[(size_t)nb*la+r] * C[(size_t)nb*si+s]
                             * eri_ao[idx4_to_1(nb, mu, nu, la, si)];
        return sum;
    };

    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int l  = (int)(t % occ); t /= occ;
                int k  = (int)(t % occ); t /= occ;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b])
                             * (d_orbital_energies[k]+d_orbital_energies[l]-d_orbital_energies[a]-d_orbital_energies[b]);
                if (fabs(denom) > 1e-14) {
                    double iajb = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,a,j,b);
                    double ikjl = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,k,j,l);
                    double kalb = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,a,l,b);
                    double kbla = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,b,l,a);
                    cpu_E += iajb*ikjl*(2.0*kalb-kbla) / denom;
                }
            }
            d_mp3_energy[0] = cpu_E;
        } else {
            const int num_blocks = (int)((total + num_threads - 1) / num_threads);
            size_t shmem = (size_t)num_threads * sizeof(double);
            mp3_naive_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);
            cudaDeviceSynchronize();
        }
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int d_ = (int)(t % vir); t /= vir;
                int c_ = (int)(t % vir); t /= vir;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_, c = num_occ+c_, d = num_occ+d_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b])
                             * (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[c]-d_orbital_energies[d]);
                if (fabs(denom) > 1e-14) {
                    double iajb = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,a,j,b);
                    double acbd = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, a,c,b,d);
                    double icjd = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,c,j,d);
                    double idjc = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,d,j,c);
                    cpu_E += iajb*acbd*(2.0*icjd-idjc) / denom;
                }
            }
            d_mp3_energy[1] = cpu_E;
        } else {
            const int num_blocks = (int)((total + num_threads - 1) / num_threads);
            size_t shmem = (size_t)num_threads * sizeof(double);
            mp3_naive_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);
            cudaDeviceSynchronize();
        }
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int c_ = (int)(t % vir); t /= vir;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int k  = (int)(t % occ); t /= occ;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_, c = num_occ+c_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[k]-d_orbital_energies[a]-d_orbital_energies[c])
                             * (d_orbital_energies[k]+d_orbital_energies[j]-d_orbital_energies[b]-d_orbital_energies[c]);
                if (fabs(denom) > 1e-14) {
                    double iajb = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,a,j,b);
                    double ijab = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, i,j,a,b);
                    double kcia = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,c,i,a);
                    double kaic = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,a,i,c);
                    double kcjb = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,c,j,b);
                    double kbjc = eri_bf_host(d_eri, d_coefficient_matrix, num_basis, k,b,j,c);
                    cpu_E += ((2.0*iajb-ijab)*(2.0*kcia-kaic)*(2.0*kcjb-kbjc) - 3.0*ijab*kaic*kbjc) / denom;
                }
            }
            d_mp3_energy[2] = cpu_E;
        } else {
            const int num_blocks = (int)((total + num_threads - 1) / num_threads);
            size_t shmem = (size_t)num_threads * sizeof(double);
            mp3_naive_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);
            cudaDeviceSynchronize();
        }
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    tracked_cudaFree(d_mp3_energy);


    std::cout << "4h2p term: " << h_mp3_energy[0] << " Hartree" << std::endl;
    std::cout << "2h4p term: " << h_mp3_energy[1] << " Hartree" << std::endl;
    std::cout << "3h3p term: " << h_mp3_energy[2] << " Hartree" << std::endl;

    return h_mp3_energy[0] + h_mp3_energy[1] + h_mp3_energy[2];
}


///////////////////////////////////////////////////////////////////////////////////// MP3 energy calculation  (from stored full MO ERI)
__global__ void mp3_moeri_4h2p_kernel(const double* __restrict__ eri_mo,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,k,l,a,b) with i,j,k,l in occ, a,b in vir (offset by occ)
  size_t total = (size_t)occ * occ * occ * occ * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int l  = (int)(t % occ); t /= occ;
    int k  = (int)(t % occ); t /= occ;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;

    double denom1 = eps[i] + eps[j] - eps[a] - eps[b];
    double denom2 = eps[k] + eps[l] - eps[a] - eps[b];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo[idx4_to_1(num_basis, i, a, j, b)];
      double ikjl = eri_mo[idx4_to_1(num_basis, i, k, j, l)];
      double kalb = eri_mo[idx4_to_1(num_basis, k, a, l, b)];
      double kbla = eri_mo[idx4_to_1(num_basis, k, b, l, a)];
      contrib = iajb*ikjl*(2.0*kalb-kbla) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}


__global__ void mp3_moeri_2h4p_kernel(const double* __restrict__ eri_mo,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,a,b,c,d) with i,j in occ, a,b,c,d in vir (offset by occ)
  size_t total = (size_t)occ * occ * vir * vir * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int d_ = (int)(t % vir); t /= vir;
    int c_ = (int)(t % vir); t /= vir;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;
    int c = occ + c_;
    int d = occ + d_;

    double denom1 = eps[i] + eps[j] - eps[a] - eps[b];
    double denom2 = eps[i] + eps[j] - eps[c] - eps[d];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo[idx4_to_1(num_basis, i, a, j, b)];
      double acbd = eri_mo[idx4_to_1(num_basis, a, c, b, d)];
      double icjd = eri_mo[idx4_to_1(num_basis, i, c, j, d)];
      double idjc = eri_mo[idx4_to_1(num_basis, i, d, j, c)];
      contrib = iajb*acbd*(2.0*icjd-idjc) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}

__global__ void mp3_moeri_3h3p_kernel(const double* __restrict__ eri_mo,
                      const double* __restrict__ eps,
                      const int num_basis, const int occ,
                      double* __restrict__ E_out)
{
  int vir = num_basis - occ;
  // Flattened index over (i,j,k,a,b,c) with i,j,k in occ, a,b,c in vir (offset by occ)
  size_t total = (size_t)occ * occ * occ * vir * vir * vir;
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  double contrib = 0.0;
  if(gid < total){
    size_t t = gid;
    int c_ = (int)(t % vir); t /= vir;
    int b_ = (int)(t % vir); t /= vir;
    int a_ = (int)(t % vir); t /= vir;
    int k  = (int)(t % occ); t /= occ;
    int j  = (int)(t % occ); t /= occ;
    int i  = (int)(t % occ);

    int a = occ + a_;
    int b = occ + b_;
    int c = occ + c_;

    double denom1 = eps[i] + eps[k] - eps[a] - eps[c];
    double denom2 = eps[k] + eps[j] - eps[b] - eps[c];
    double denom = denom1 * denom2;
    // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
    if(fabs(denom) > 1e-14){
      double iajb = eri_mo[idx4_to_1(num_basis, i, a, j, b)];
      double ijab = eri_mo[idx4_to_1(num_basis, i, j, a, b)];
      double kcia = eri_mo[idx4_to_1(num_basis, k, c, i, a)];
      double kaic = eri_mo[idx4_to_1(num_basis, k, a, i, c)];
      double kcjb = eri_mo[idx4_to_1(num_basis, k, c, j, b)];
      double kbjc = eri_mo[idx4_to_1(num_basis, k, b, j, c)];
      contrib = ((2.0*iajb-ijab)*(2.0*kcia-kaic)*(2.0*kcjb-kbjc) - 3.0*ijab*kaic*kbjc) / denom;
    }
  }

  double block_sum = block_reduce_sum(contrib);
  if(threadIdx.x == 0){
    atomicAdd(E_out, block_sum);
  }
}


real_t mp3_from_aoeri_via_full_moeri(const real_t* d_eri_ao, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_occ) {
    const int num_threads = 256;

    const int N = num_basis * num_basis;

    // ------------------------------------------------------------
    // 1) allocate full MO ERI on device: d_eri_mo (N x N)
    // ------------------------------------------------------------
    double* d_eri_mo = nullptr;
    size_t bytes_mo = (size_t)N * (size_t)N * sizeof(double);
    tracked_cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("tracked_cudaMalloc failed for d_eri_mo.");
    }


    // ------------------------------------------------------------
    // 2) AO -> MO full transformation (writes into d_eri_mo)
    // ------------------------------------------------------------
    {
        std::string str = "Computing AO -> MO full integral transformation... ";
        PROFILE_ELAPSED_TIME(str);

        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix, num_basis, d_eri_mo);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    //debug: checking MO ERI by comparing with brute-force transformation and stored MO ERI
    // std::cout << "Checking MO ERI..." << std::endl;
    // check_moeri(d_eri_mo, d_eri_ao, d_coefficient_matrix, num_basis);

    // ------------------------------------------------------------
    // 3) MP2 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp2_energy;
    tracked_cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
    if(d_mp2_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP2 energy.");
    }
    cudaMemset(d_mp2_energy, 0.0, sizeof(real_t));
    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        if (!gpu::gpu_available()) {
            auto idx4 = [&](int p, int q, int r, int s) -> size_t {
                return (((size_t)p*num_basis + q)*num_basis + r)*num_basis + s;
            };
            double cpu_E = 0.0;
            int vir_i = num_basis - num_occ;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int b_ = (int)(t % vir_i); t /= vir_i;
                int a_ = (int)(t % vir_i); t /= vir_i;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_;
                double denom = d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b];
                if (fabs(denom) > 1e-14) {
                    double iajb = d_eri_mo[idx4(i,a,j,b)];
                    double ibja = d_eri_mo[idx4(i,b,j,a)];
                    cpu_E += iajb*(2.0*iajb - ibja)/denom;
                }
            }
            *d_mp2_energy = cpu_E;
        } else {
            mp2_moeri_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, d_mp2_energy);
            cudaDeviceSynchronize();
        }
    }
    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    tracked_cudaFree(d_mp2_energy);
    std::cout << "MP2 energy: " << h_mp2_energy << " Hartree" << std::endl;



    // ------------------------------------------------------------
    // 4) MP3 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp3_energy;
    tracked_cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);
    cudaDeviceSynchronize();

    auto idx4m = [&](int p, int q, int r, int s) -> size_t {
        return (((size_t)p*num_basis + q)*num_basis + r)*num_basis + s;
    };
    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int l  = (int)(t % occ); t /= occ;
                int k  = (int)(t % occ); t /= occ;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b])
                             * (d_orbital_energies[k]+d_orbital_energies[l]-d_orbital_energies[a]-d_orbital_energies[b]);
                if (fabs(denom) > 1e-14) {
                    cpu_E += d_eri_mo[idx4m(i,a,j,b)] * d_eri_mo[idx4m(i,k,j,l)]
                           * (2.0*d_eri_mo[idx4m(k,a,l,b)] - d_eri_mo[idx4m(k,b,l,a)]) / denom;
                }
            }
            d_mp3_energy[0] = cpu_E;
        } else {
            const int num_blocks = (int)((total + num_threads - 1) / num_threads);
            size_t shmem = (size_t)num_threads * sizeof(double);
            mp3_moeri_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);
            cudaDeviceSynchronize();
        }
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int d_ = (int)(t % vir); t /= vir;
                int c_ = (int)(t % vir); t /= vir;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_, c = num_occ+c_, d = num_occ+d_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b])
                             * (d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[c]-d_orbital_energies[d]);
                if (fabs(denom) > 1e-14) {
                    cpu_E += d_eri_mo[idx4m(i,a,j,b)] * d_eri_mo[idx4m(a,c,b,d)]
                           * (2.0*d_eri_mo[idx4m(i,c,j,d)] - d_eri_mo[idx4m(i,d,j,c)]) / denom;
                }
            }
            d_mp3_energy[1] = cpu_E;
        } else {
            const int num_blocks = (int)((total + num_threads - 1) / num_threads);
            size_t shmem = (size_t)num_threads * sizeof(double);
            mp3_moeri_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);
            cudaDeviceSynchronize();
        }
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E) schedule(dynamic)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int c_ = (int)(t % vir); t /= vir;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int k  = (int)(t % occ); t /= occ;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = num_occ+a_, b = num_occ+b_, c = num_occ+c_;
                double denom = (d_orbital_energies[i]+d_orbital_energies[k]-d_orbital_energies[a]-d_orbital_energies[c])
                             * (d_orbital_energies[k]+d_orbital_energies[j]-d_orbital_energies[b]-d_orbital_energies[c]);
                if (fabs(denom) > 1e-14) {
                    double iajb = d_eri_mo[idx4m(i,a,j,b)];
                    double ijab = d_eri_mo[idx4m(i,j,a,b)];
                    double kcia = d_eri_mo[idx4m(k,c,i,a)];
                    double kaic = d_eri_mo[idx4m(k,a,i,c)];
                    double kcjb = d_eri_mo[idx4m(k,c,j,b)];
                    double kbjc = d_eri_mo[idx4m(k,b,j,c)];
                    cpu_E += ((2.0*iajb-ijab)*(2.0*kcia-kaic)*(2.0*kcjb-kbjc) - 3.0*ijab*kaic*kbjc) / denom;
                }
            }
            d_mp3_energy[2] = cpu_E;
        } else {
            mp3_moeri_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);
            cudaDeviceSynchronize();
        }
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    tracked_cudaFree(d_mp3_energy);
    tracked_cudaFree(d_eri_mo);

    std::cout << "4h2p term: " << h_mp3_energy[0] << " Hartree" << std::endl;
    std::cout << "2h4p term: " << h_mp3_energy[1] << " Hartree" << std::endl;
    std::cout << "3h3p term: " << h_mp3_energy[2] << " Hartree" << std::endl;


    return h_mp2_energy + h_mp3_energy[0] + h_mp3_energy[1] + h_mp3_energy[2];
}






double mp2_from_full_moeri(
    const double* d_eri_mo,   // device, size nao^4, row-major (mu nu | la si)
    const double* d_C,        // device, size nao*nao, row-major (mu,p)
    const double* d_eps,      // device, size nao
    int nao,
    int occ)
{
    int vir = nao - occ;
    size_t total = (size_t)occ * (size_t)occ * (size_t)vir * (size_t)vir;

    double* d_E = nullptr;
    tracked_cudaMalloc((void**)&d_E, sizeof(double));
    cudaMemset(d_E, 0, sizeof(double));

    int threads = 1024;
    size_t blocks  = (size_t)((total + threads - 1) / threads);
    size_t shmem = (size_t)threads * sizeof(double);

    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            double cpu_E = 0.0;
            #pragma omp parallel for reduction(+:cpu_E)
            for (size_t gid = 0; gid < total; gid++) {
                size_t t = gid;
                int b_ = (int)(t % vir); t /= vir;
                int a_ = (int)(t % vir); t /= vir;
                int j  = (int)(t % occ); t /= occ;
                int i  = (int)(t % occ);
                int a = occ + a_, b = occ + b_;
                double denom = d_eps[i] + d_eps[j] - d_eps[a] - d_eps[b];
                if (fabs(denom) > 1e-14) {
                    size_t idx_iajb = (((size_t)i*nao + a)*nao + j)*nao + b;
                    size_t idx_ibja = (((size_t)i*nao + b)*nao + j)*nao + a;
                    double iajb = d_eri_mo[idx_iajb];
                    double ibja = d_eri_mo[idx_ibja];
                    cpu_E += iajb * (2.0 * iajb - ibja) / denom;
                }
            }
            *d_E = cpu_E;
        } else {
            mp2_from_moeri_kernel<<<blocks, threads, shmem>>>(d_eri_mo, d_eps, nao, occ, d_E);
            cudaDeviceSynchronize();
        }
    }

    double h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "h_E: " << std::setprecision(12) << h_E << std::endl;

    tracked_cudaFree(d_E);

    return h_E;
}






__global__
void kalb2klab(double* d_ovov, double* d_oovv, const int num_occ, const int num_vir)
{
    const long long kalb = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_ovov = (long long)num_occ * num_vir * num_occ * num_vir;
    if (kalb >= num_ovov) {
        return;
    }

    const int ka = kalb / (num_occ * num_vir);
    const int lb = kalb % (num_occ * num_vir);
    const int k = ka / num_vir;
    const int a = ka % num_vir + num_occ;
    const int l = lb / num_vir;
    const int b = lb % num_vir + num_occ;

    d_oovv[oovv2s(k, l, a, b, num_occ, num_vir)] = d_ovov[ovov2s(k, a, l, b, num_occ, num_vir)];
}

__global__
void icjd2cdij(double* d_ovov, double* d_vvoo, const int num_occ, const int num_vir)
{
    const long long icjd = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_ovov = (long long)num_occ * num_vir * num_occ * num_vir;
    if (icjd >= num_ovov) {
        return;
    }

    const int ic = icjd / (num_occ * num_vir);
    const int jd = icjd % (num_occ * num_vir);
    const int i = ic / num_vir;
    const int c = ic % num_vir + num_occ;
    const int j = jd / num_vir;
    const int d = jd % num_vir + num_occ;

    d_vvoo[vvoo2s(c, d, i, j, num_occ, num_vir)] = d_ovov[ovov2s(i, c, j, d, num_occ, num_vir)];
}

__global__
void kaic2iakc(double* d_ovov_in, double* d_ovov_out, const int num_occ, const int num_vir)
{
    const long long kaic = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_ovov = (long long)num_occ * num_vir * num_occ * num_vir;
    if (kaic >= num_ovov) {
        return;
    }

    const int ka = kaic / (num_occ * num_vir);
    const int ic = kaic % (num_occ * num_vir);
    const int k = ka / num_vir;
    const int a = ka % num_vir + num_occ;
    const int i = ic / num_vir;
    const int c = ic % num_vir + num_occ;

    d_ovov_out[ovov2s(i, a, k, c, num_occ, num_vir)] = d_ovov_in[ovov2s(k, a, i, c, num_occ, num_vir)];
}



__global__
void kbjc2kcjb(double* d_ovov_in, double* d_ovov_out, const int num_occ, const int num_vir)
{
    const long long kbjc = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_ovov = (long long)num_occ * num_vir * num_occ * num_vir;
    if (kbjc >= num_ovov) {
        return;
    }

    const int kb = kbjc / (num_occ * num_vir);
    const int jc = kbjc % (num_occ * num_vir);
    const int k = kb / num_vir;
    const int b = kb % num_vir + num_occ;
    const int j = jc / num_vir;
    const int c = jc % num_vir + num_occ;

    d_ovov_out[ovov2s(k, c, j, b, num_occ, num_vir)] = d_ovov_in[ovov2s(k, b, j, c, num_occ, num_vir)];
}




__global__
void contract_iajb_tensors(   // 4h2p, 2h4p, 3h3p
    const int num_orbitals, const int num_occ, const int num_vir, double* g_int2e, 
    double* g_s_ovov, double* g_mm1, double* g_mm2, double* g_mm3, double* g_mm4, double* g_E_3rd)
{
    __shared__ double s_E_3rd;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_E_3rd = 0;
    }
    __syncthreads();

    const long long ijab = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_oovv = (long long)num_occ * num_occ * num_vir * num_vir;
    if (ijab >= num_oovv) {
        return;
    }

    const int ij = ijab / (num_vir * num_vir);
    const int ab = ijab % (num_vir * num_vir);
    const int i = ij / num_occ;
    const int j = ij % num_occ;
    const int a = ab / num_vir + num_occ;
    const int b = ab % num_vir + num_occ;

    const double s_iajb = g_s_ovov[ovov2s(i, a, j, b, num_occ, num_vir)];
    const double e_ijab = g_int2e[q2s(i, j, a, b, num_orbitals)];
    const double e_iajb = g_int2e[q2s(i, a, j, b, num_orbitals)];

    double energy = 0.0;
    energy += s_iajb * (g_mm1[oovv2s(i, j, a, b, num_occ, num_vir)] + g_mm2[vvoo2s(a, b, i, j, num_occ, num_vir)]);
    energy += (2 * e_iajb - e_ijab) * g_mm3[ovov2s(i, a, j, b, num_occ, num_vir)];
    energy += (-3) * e_ijab * g_mm4[ovov2s(i, a, j, b, num_occ, num_vir)];

    for (int offset = 16; offset > 0; offset /= 2) {
        energy += __shfl_down_sync(FULLMASK, energy, offset);
    }

    if (threadIdx.x == 0) {
        atomicAdd(&s_E_3rd, energy);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(g_E_3rd, s_E_3rd);
    }
}




//*
real_t mp3_from_aoeri_via_full_moeri_dgemm(
    real_t* d_eri_ao, const real_t* d_coefficient_matrix,
    const real_t* d_orbital_energies, const int num_basis, const int num_occ,
    real_t* d_eri_mo_precomputed = nullptr)
{
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    double* d_groundE_3rd;
    double* h_groundE_3rd;
    tracked_cudaMalloc(&d_groundE_3rd, sizeof(double));
    cudaMallocHost(&h_groundE_3rd, sizeof(double));
    cudaMemset(d_groundE_3rd, 0, sizeof(double));

    const int num_vir = num_basis - num_occ;
    printf("#orbitals_occ: %d, #orbitals_vir: %d\n", num_occ, num_vir);

    // Full MO ERI transformation
    double* d_eri_mo;
    bool free_eri_mo;
    const size_t num_basis_2 = num_basis * num_basis;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        d_eri_mo = nullptr;
        tracked_cudaMalloc((void**)&d_eri_mo, sizeof(double) * num_basis_2 * num_basis_2);
        if (!d_eri_mo) {
            THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
        }
        transform_eri_ao2mo_dgemm_full(d_eri_ao, d_eri_mo, d_coefficient_matrix, num_basis);
        cudaDeviceSynchronize();
        free_eri_mo = true;
    }

    // MP2 energy from full MO ERI
    real_t E_MP2 = mp2_from_full_moeri(d_eri_mo, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ);
    printf("MP2 energy from full MO ERI: %.12f\n", E_MP2);
    //return E_MP2;


    const long long num_oooo = (long long)num_occ * num_occ * num_occ * num_occ;
    const long long num_vvvv = (long long)num_vir * num_vir * num_vir * num_vir;
    const long long num_ovov = (long long)num_occ * num_vir * num_occ * num_vir;
    const long long num_oovv = (long long)num_occ * num_occ * num_vir * num_vir;
    //printf("num_oooo: %lld\n", num_oooo);
    //printf("num_vvvv: %lld\n", num_vvvv);
    //printf("num_ovov: %lld\n", num_ovov);
    //printf("num_oovv: %lld\n", num_oovv);

    double* d_oooo;
    double* d_vvvv;
    double* d_s_ovov;
    double* d_t_ovov;
    double* d_t_tmp;
    double* d_mm1;
    double* d_mm2;
    double* d_mm3;
    double* d_mm4;
    tracked_cudaMalloc(&d_oooo, sizeof(double) * num_oooo);
    tracked_cudaMalloc(&d_vvvv, sizeof(double) * num_vvvv);
    tracked_cudaMalloc(&d_s_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_t_ovov, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_t_tmp, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_mm1, sizeof(double) * num_oovv);
    tracked_cudaMalloc(&d_mm2, sizeof(double) * num_oovv);
    tracked_cudaMalloc(&d_mm3, sizeof(double) * num_ovov);
    tracked_cudaMalloc(&d_mm4, sizeof(double) * num_ovov);
    double* d_s_tmp1 = d_t_ovov;
    double* d_s_tmp2 = d_t_tmp;

    constexpr int num_threads_per_warp = 32;
    constexpr int num_warps_per_block = 32;
    constexpr int num_threads_per_block = num_threads_per_warp * num_warps_per_block;

    const long long num_blocks_oooo = (num_oooo + num_threads_per_block - 1) / num_threads_per_block;
    const long long num_blocks_vvvv = (num_vvvv + num_threads_per_block - 1) / num_threads_per_block;
    const long long num_blocks_ovov = (num_ovov + num_threads_per_block - 1) / num_threads_per_block;
    const long long num_blocks_oovv = (num_oovv + num_threads_per_block - 1) / num_threads_per_block;

    if (num_blocks_oooo > prop.maxGridSize[0] ||
        num_blocks_vvvv > prop.maxGridSize[0] ||
        num_blocks_ovov > prop.maxGridSize[0] ||
        num_blocks_oovv > prop.maxGridSize[0]) {
        printf("Error: Too many blocks for the grid size.\n");
        return 0;
    }

    dim3 blocks_oooo(num_blocks_oooo);
    dim3 blocks_vvvv(num_blocks_vvvv);
    dim3 blocks_ovov(num_blocks_ovov);
    dim3 blocks_oovv(num_blocks_oovv);
    dim3 threads(num_threads_per_warp, num_warps_per_block);

    float time_tensor, time_dgemm, time_ijab;
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);

    cudaEventRecord(begin);
    if (!gpu::gpu_available()) {
        // CPU fallback: tensorize_oooo, tensorize_vvvv, tensorize_ovov
        const int num_orbitals = num_occ + num_vir;
        auto q2s_cpu = [&](int mu, int nu, int la, int si) -> size_t {
            return ((size_t)num_orbitals*num_orbitals*num_orbitals)*mu + (size_t)num_orbitals*num_orbitals*nu + (size_t)num_orbitals*la + si;
        };
        // tensorize_oooo: g_oooo[i*o^3+j*o^2+k*o+l] = g_int2e[q2s(i,k,j,l)]
        const long long num_oooo_tot = (long long)num_occ*num_occ*num_occ*num_occ;
        #pragma omp parallel for
        for (long long ijkl = 0; ijkl < num_oooo_tot; ijkl++) {
            int kl = ijkl % (num_occ*num_occ);
            int ij = ijkl / (num_occ*num_occ);
            int i=ij/num_occ, j=ij%num_occ, k=kl/num_occ, l=kl%num_occ;
            d_oooo[ijkl] = d_eri_mo[q2s_cpu(i,k,j,l)];
        }
        // tensorize_vvvv: g_vvvv[(a-o)*v^3+(b-o)*v^2+(c-o)*v+(d-o)] = g_int2e[q2s(a,c,b,d)]
        const long long num_vvvv_tot = (long long)num_vir*num_vir*num_vir*num_vir;
        #pragma omp parallel for
        for (long long abcd = 0; abcd < num_vvvv_tot; abcd++) {
            int cd = abcd % (num_vir*num_vir);
            int ab = abcd / (num_vir*num_vir);
            int a=ab/num_vir+num_occ, b=ab%num_vir+num_occ;
            int c=cd/num_vir+num_occ, d=cd%num_vir+num_occ;
            d_vvvv[abcd] = d_eri_mo[q2s_cpu(a,c,b,d)];
        }
        // tensorize_ovov
        auto ovov2s_cpu = [&](int i, int a, int j, int b) -> size_t {
            return (size_t)i*num_occ*num_vir*num_vir + (size_t)(a-num_occ)*num_occ*num_vir + (size_t)j*num_vir + (b-num_occ);
        };
        const long long num_ovov_tot = (long long)num_occ*num_vir*num_occ*num_vir;
        #pragma omp parallel for
        for (long long idx = 0; idx < num_ovov_tot; idx++) {
            int jb = idx % (num_occ*num_vir);
            int ia = idx / (num_occ*num_vir);
            int i=ia/num_vir, a=ia%num_vir+num_occ;
            int j=jb/num_vir, b=jb%num_vir+num_occ;
            double iajb = d_eri_mo[q2s_cpu(i,a,j,b)];
            double ibja = d_eri_mo[q2s_cpu(i,b,j,a)];
            double eps_ijab = d_orbital_energies[i]+d_orbital_energies[j]-d_orbital_energies[a]-d_orbital_energies[b];
            d_s_ovov[ovov2s_cpu(i,a,j,b)] = iajb / eps_ijab;
            d_t_ovov[ovov2s_cpu(i,a,j,b)] = (2.0*iajb - ibja) / eps_ijab;
        }
    } else {
        tensorize_oooo<<<blocks_oooo, threads>>>(d_eri_mo, d_oooo, num_occ, num_vir);
        tensorize_vvvv<<<blocks_vvvv, threads>>>(d_eri_mo, d_vvvv, num_occ, num_vir);
        tensorize_ovov<<<blocks_ovov, threads>>>(d_eri_mo, d_orbital_energies, d_s_ovov, d_t_ovov, num_occ, num_vir);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_tensor, begin, end);
    printf("tensorize: %.2f [msec]\n", time_tensor);

    const double alpha = 1.0;
    const double beta = 0.0;
    const int num_oo = num_occ * num_occ;
    const int num_vv = num_vir * num_vir;
    const int num_ov = num_occ * num_vir;
    cublasHandle_t cublasH = NULL;
    if (gpu::gpu_available()) cublasCreate(&cublasH);

    cudaEventRecord(begin);

    if (!gpu::gpu_available()) {
        // CPU fallback: kalb2klab permutation
        auto ovov2s_c = [&](int i, int a, int j, int b) -> size_t {
            return (size_t)i*num_occ*num_vir*num_vir + (size_t)(a-num_occ)*num_occ*num_vir + (size_t)j*num_vir + (b-num_occ);
        };
        auto oovv2s_c = [&](int i, int j, int a, int b) -> size_t {
            return (size_t)i*num_vir*num_vir*num_occ + (size_t)j*num_vir*num_vir + (size_t)(a-num_occ)*num_vir + (b-num_occ);
        };
        const long long num_ovov_p = (long long)num_occ*num_vir*num_occ*num_vir;
        #pragma omp parallel for
        for (long long kalb = 0; kalb < num_ovov_p; kalb++) {
            int lb = kalb % (num_occ*num_vir);
            int ka = kalb / (num_occ*num_vir);
            int k=ka/num_vir, a=ka%num_vir+num_occ;
            int l=lb/num_vir, b=lb%num_vir+num_occ;
            d_t_tmp[oovv2s_c(k,l,a,b)] = d_t_ovov[ovov2s_c(k,a,l,b)];
        }
    } else {
        kalb2klab<<<blocks_ovov, threads>>>(d_t_ovov, d_t_tmp, num_occ, num_vir);
        cudaDeviceSynchronize();
    }
    if (gpu::gpu_available()) {
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, num_vv, num_oo, num_oo,
                    &alpha, d_t_tmp, num_vv, d_oooo, num_oo, &beta, d_mm1, num_vv);
    } else {
        // Col-major C = A * B → row-major: C_rm = B_rm * A_rm
        // mm1_rm(oo, vv) = oooo_rm(oo, oo) * t_tmp_rm(oo, vv)
        gpu::matrixMatrixProductRect(d_oooo, d_t_tmp, d_mm1,
            num_oo, num_vv, num_oo, false, false, false, 1.0);
    }

    if (!gpu::gpu_available()) {
        // CPU fallback: icjd2cdij permutation
        auto ovov2s_c = [&](int i, int a, int j, int b) -> size_t {
            return (size_t)i*num_occ*num_vir*num_vir + (size_t)(a-num_occ)*num_occ*num_vir + (size_t)j*num_vir + (b-num_occ);
        };
        auto vvoo2s_c = [&](int c, int d, int i, int j) -> size_t {
            return (size_t)(c-num_occ)*num_occ*num_occ*num_vir + (size_t)(d-num_occ)*num_occ*num_occ + (size_t)i*num_occ + j;
        };
        const long long num_ovov_p = (long long)num_occ*num_vir*num_occ*num_vir;
        #pragma omp parallel for
        for (long long icjd = 0; icjd < num_ovov_p; icjd++) {
            int jd = icjd % (num_occ*num_vir);
            int ic = icjd / (num_occ*num_vir);
            int i=ic/num_vir, c=ic%num_vir+num_occ;
            int j=jd/num_vir, d=jd%num_vir+num_occ;
            d_t_tmp[vvoo2s_c(c,d,i,j)] = d_t_ovov[ovov2s_c(i,c,j,d)];
        }
    } else {
        icjd2cdij<<<blocks_ovov, threads>>>(d_t_ovov, d_t_tmp, num_occ, num_vir);
        cudaDeviceSynchronize();
    }
    if (gpu::gpu_available()) {
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, num_oo, num_vv, num_vv,
                    &alpha, d_t_tmp, num_oo, d_vvvv, num_vv, &beta, d_mm2, num_oo);
    } else {
        // mm2_rm(vv, oo) = vvvv_rm(vv, vv) * t_tmp_rm(vv, oo)
        gpu::matrixMatrixProductRect(d_vvvv, d_t_tmp, d_mm2,
            num_vv, num_oo, num_vv, false, false, false, 1.0);
    }

    if (gpu::gpu_available()) {
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_ov, num_ov, num_ov,
                    &alpha, d_t_ovov, num_ov, d_t_ovov, num_ov, &beta, d_mm3, num_ov);
    } else {
        // Col-major C = A * B^T → row-major: C_rm = B_rm^T * A_rm
        // mm3_rm(ov, ov) = t_ovov_rm^T(ov, ov) * t_ovov_rm(ov, ov)
        gpu::matrixMatrixProductRect(d_t_ovov, d_t_ovov, d_mm3,
            num_ov, num_ov, num_ov, true, false, false, 1.0);
    }

    if (!gpu::gpu_available()) {
        // CPU fallback: kaic2iakc and kbjc2kcjb permutations
        auto ovov2s_c = [&](int i, int a, int j, int b) -> size_t {
            return (size_t)i*num_occ*num_vir*num_vir + (size_t)(a-num_occ)*num_occ*num_vir + (size_t)j*num_vir + (b-num_occ);
        };
        const long long num_ovov_p = (long long)num_occ*num_vir*num_occ*num_vir;
        #pragma omp parallel for
        for (long long kaic = 0; kaic < num_ovov_p; kaic++) {
            int ic = kaic % (num_occ*num_vir);
            int ka = kaic / (num_occ*num_vir);
            int k=ka/num_vir, a=ka%num_vir+num_occ;
            int i=ic/num_vir, c=ic%num_vir+num_occ;
            d_s_tmp1[ovov2s_c(i,a,k,c)] = d_s_ovov[ovov2s_c(k,a,i,c)];
        }
        #pragma omp parallel for
        for (long long kbjc = 0; kbjc < num_ovov_p; kbjc++) {
            int jc = kbjc % (num_occ*num_vir);
            int kb = kbjc / (num_occ*num_vir);
            int k=kb/num_vir, b=kb%num_vir+num_occ;
            int j=jc/num_vir, c=jc%num_vir+num_occ;
            d_s_tmp2[ovov2s_c(k,c,j,b)] = d_s_ovov[ovov2s_c(k,b,j,c)];
        }
    } else {
        kaic2iakc<<<blocks_ovov, threads>>>(d_s_ovov, d_s_tmp1, num_occ, num_vir);
        kbjc2kcjb<<<blocks_ovov, threads>>>(d_s_ovov, d_s_tmp2, num_occ, num_vir);
        cudaDeviceSynchronize();
    }
    if (gpu::gpu_available()) {
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, num_ov, num_ov, num_ov,
                    &alpha, d_s_tmp2, num_ov, d_s_tmp1, num_ov, &beta, d_mm4, num_ov);
    } else {
        // mm4_rm(ov, ov) = s_tmp1_rm(ov, ov) * s_tmp2_rm(ov, ov)
        gpu::matrixMatrixProductRect(d_s_tmp1, d_s_tmp2, d_mm4,
            num_ov, num_ov, num_ov, false, false, false, 1.0);
    }

    if (cublasH) cublasDestroy(cublasH);

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_dgemm, begin, end);
    printf("dgemm: %.2f [msec]\n", time_dgemm);

    cudaEventRecord(begin);
    if (!gpu::gpu_available()) {
        // CPU fallback: contract_iajb_tensors (4h2p + 2h4p + 3h3p reduction)
        const int num_orbitals = num_basis;
        auto q2s_c = [&](int mu, int nu, int la, int si) -> size_t {
            return ((size_t)num_orbitals*num_orbitals*num_orbitals)*mu + (size_t)num_orbitals*num_orbitals*nu + (size_t)num_orbitals*la + si;
        };
        auto ovov2s_c = [&](int i, int a, int j, int b) -> size_t {
            return (size_t)i*num_occ*num_vir*num_vir + (size_t)(a-num_occ)*num_occ*num_vir + (size_t)j*num_vir + (b-num_occ);
        };
        auto oovv2s_c = [&](int i, int j, int a, int b) -> size_t {
            return (size_t)i*num_vir*num_vir*num_occ + (size_t)j*num_vir*num_vir + (size_t)(a-num_occ)*num_vir + (b-num_occ);
        };
        auto vvoo2s_c = [&](int c, int d, int i, int j) -> size_t {
            return (size_t)(c-num_occ)*num_occ*num_occ*num_vir + (size_t)(d-num_occ)*num_occ*num_occ + (size_t)i*num_occ + j;
        };
        const long long num_oovv_tot = (long long)num_occ*num_occ*num_vir*num_vir;
        double cpu_E3 = 0.0;
        #pragma omp parallel for reduction(+:cpu_E3)
        for (long long ijab = 0; ijab < num_oovv_tot; ijab++) {
            int ab = ijab % (num_vir*num_vir);
            int ij = ijab / (num_vir*num_vir);
            int i=ij/num_occ, j=ij%num_occ;
            int a=ab/num_vir+num_occ, b=ab%num_vir+num_occ;
            double s_iajb = d_s_ovov[ovov2s_c(i,a,j,b)];
            double e_ijab = d_eri_mo[q2s_c(i,j,a,b)];
            double e_iajb = d_eri_mo[q2s_c(i,a,j,b)];
            double energy = s_iajb * (d_mm1[oovv2s_c(i,j,a,b)] + d_mm2[vvoo2s_c(a,b,i,j)]);
            energy += (2.0*e_iajb - e_ijab) * d_mm3[ovov2s_c(i,a,j,b)];
            energy += (-3.0) * e_ijab * d_mm4[ovov2s_c(i,a,j,b)];
            cpu_E3 += energy;
        }
        *d_groundE_3rd += cpu_E3;
    } else {
        contract_iajb_tensors<<<blocks_ovov, threads>>>(num_basis, num_occ, num_vir, d_eri_mo, d_s_ovov, d_mm1, d_mm2, d_mm3, d_mm4, d_groundE_3rd);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ijab, begin, end);

    printf("three terms (2h2p): %.2f [msec]\n", time_ijab);
    printf("mp3 total: %.2f [msec]\n", (time_tensor + time_dgemm + time_ijab));
    printf("mp3 correlation energy: %.4f [sec]\n", (time_tensor + time_dgemm + time_ijab) * 1e-3);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    cudaMemcpy(h_groundE_3rd, d_groundE_3rd, sizeof(double), cudaMemcpyDeviceToHost);
    const double correlationE_3rd = *h_groundE_3rd;
    printf("3rd perturbation energy: %.12f [hartree]\n", correlationE_3rd);

    tracked_cudaFree(d_groundE_3rd);
    cudaFreeHost(h_groundE_3rd);

    tracked_cudaFree(d_oooo);
    tracked_cudaFree(d_vvvv);
    tracked_cudaFree(d_s_ovov);
    tracked_cudaFree(d_t_ovov);
    tracked_cudaFree(d_t_tmp);
    tracked_cudaFree(d_mm1);
    tracked_cudaFree(d_mm2);
    tracked_cudaFree(d_mm3);
    tracked_cudaFree(d_mm4);

    if (free_eri_mo) tracked_cudaFree(d_eri_mo);

    return E_MP2 + correlationE_3rd;
}
/**/








//////////////////////////////////////////////////////////////////////////////////////// MP3 energy calculation

real_t compute_mp3_energy_impl(RHF& rhf, real_t* d_eri, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_basis = rhf.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();

    real_t E_MP3 = mp3_from_aoeri_via_full_moeri_dgemm(d_eri, d_C, d_eps, num_basis, num_occ, d_eri_mo_precomputed);

    std::cout << "MP3 energy: " << E_MP3 << " Hartree" << std::endl;
    return E_MP3;
}

real_t ERI_Stored_RHF::compute_mp3_energy() {
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
        real_t result = compute_mp3_energy_impl(rhf_, nullptr, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return result;
    }
    return compute_mp3_energy_impl(rhf_, eri_matrix_.device_ptr());
}

real_t ERI_RI_RHF::compute_mp3_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_mp3_energy_impl(rhf_, nullptr, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

// Forward declarations for half-transform MP3 (defined in half_transform_mp3.cu)
real_t mp3_half_transform_direct(RHF& rhf, const HF& hf, int block_s);
real_t mp3_half_transform_hash(
    RHF& rhf,
    const unsigned long long* d_coo_keys, const real_t* d_coo_values, size_t num_entries,
    const unsigned long long* d_hash_keys, const real_t* d_hash_values,
    const size_t* d_nonzero_indices, size_t num_nonzero,
    size_t hash_capacity_mask, HashFockMethod method, int block_s);

real_t ERI_Direct_RHF::compute_mp3_energy() {
    PROFILE_FUNCTION();
    const int nao = rhf_.get_num_basis();

    // CPU fallback: reuse stored MP3 path via build_mo_eri
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        real_t result = compute_mp3_energy_impl(rhf_, nullptr, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return result;
    }

    // Determine block_s from available GPU memory
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t nao3 = (size_t)nao * nao * nao;
    int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
    block_s = std::min(block_s, nao);
    if (block_s > 8) block_s = 8;

    return mp3_half_transform_direct(rhf_, hf_, block_s);
}

real_t ERI_Hash_RHF::compute_mp3_energy() {
    PROFILE_FUNCTION();
    const int nao = rhf_.get_num_basis();

    // CPU fallback: reuse stored MP3 path via build_mo_eri
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), nao);
        real_t result = compute_mp3_energy_impl(rhf_, nullptr, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return result;
    }

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    const size_t nao3 = (size_t)nao * nao * nao;
    int block_s = std::max(1, (int)(free_mem * 4 / 10 / (nao3 * sizeof(real_t))));
    block_s = std::min(block_s, nao);
    if (block_s > 8) block_s = 8;

    return mp3_half_transform_hash(rhf_,
        d_coo_keys_, d_coo_values_, num_entries_,
        d_hash_keys_, d_hash_values_,
        d_nonzero_indices_, num_nonzero_,
        hash_capacity_mask_, hash_fock_method_, block_s);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////// CCSD energy calculation


__device__ __forceinline__ real_t t1_amplitude(const real_t* __restrict__ t_ia,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const int i, const int a_) // a_ is index in virtual space (0 to num_spin_vir-1)
{
    return t_ia[i * num_spin_vir + a_];
}

__device__ __forceinline__ real_t t2_amplitude(const real_t* __restrict__ t_ijab,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    return t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)];
}

__device__ real_t U_ijab(const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    real_t sum = 0.0;
    
    // t_ij^ab contribution
    real_t t_ijab_val = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
    sum += t_ijab_val;
    

    // 0.5 * (t_i^a t_j^b - t_i^b t_j^a)
    real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_);
    real_t t_jb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, b_);
    real_t t_ib_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, b_);
    real_t t_ja_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, a_);

    sum += (1.0) / (2.0) * (t_ia_val * t_jb_val - t_ib_val * t_ja_val);
    return sum;
}


__device__ real_t T_ijab(const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    real_t sum = 0.0;

    // t_ij^ab contribution
    real_t t_ijab_val = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
    sum += t_ijab_val;

    // t_i^a * t_jb
    real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_);
    real_t t_jb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, b_);
    sum += t_ia_val * t_jb_val;

    // - t_i^b * t_ja
    real_t t_ib_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, b_);
    real_t t_ja_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, a_);
    sum -= t_ib_val * t_ja_val;

    return sum;
}

// ============================================================
//  Host-side helper functions for CPU fallback paths
// ============================================================
namespace cpu_helpers {

static inline size_t idx4_to_1_host(int num_basis, int mu, int nu, int la, int si) {
    return (((size_t(mu) * num_basis + nu) * num_basis + la) * num_basis + si);
}

static inline real_t antisym_eri_host(const real_t* eri_mo, int num_basis, int p, int q, int r, int s) {
    real_t prqs = ((p % 2) == (r % 2) && (q % 2) == (s % 2)) ? eri_mo[idx4_to_1_host(num_basis, p / 2, r / 2, q / 2, s / 2)] : 0.0;
    real_t psqr = ((p % 2) == (s % 2) && (q % 2) == (r % 2)) ? eri_mo[idx4_to_1_host(num_basis, p / 2, s / 2, q / 2, r / 2)] : 0.0;
    return prqs - psqr;
}

static inline real_t t1_host(const real_t* t_ia, int num_spin_occ, int num_spin_vir, int i, int a_) {
    return t_ia[i * num_spin_vir + a_];
}

static inline real_t t2_host(const real_t* t_ijab, int num_spin_occ, int num_spin_vir, int i, int j, int a_, int b_) {
    return t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)];
}

static inline real_t U_ijab_host(const real_t* t_ia, const real_t* t_ijab, int num_spin_occ, int num_spin_vir, int i, int j, int a_, int b_) {
    real_t sum = t2_host(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
    real_t t_ia_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, a_);
    real_t t_jb_val = t1_host(t_ia, num_spin_occ, num_spin_vir, j, b_);
    real_t t_ib_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, b_);
    real_t t_ja_val = t1_host(t_ia, num_spin_occ, num_spin_vir, j, a_);
    sum += 0.5 * (t_ia_val * t_jb_val - t_ib_val * t_ja_val);
    return sum;
}

static inline real_t T_ijab_host(const real_t* t_ia, const real_t* t_ijab, int num_spin_occ, int num_spin_vir, int i, int j, int a_, int b_) {
    real_t sum = t2_host(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
    real_t t_ia_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, a_);
    real_t t_jb_val = t1_host(t_ia, num_spin_occ, num_spin_vir, j, b_);
    sum += t_ia_val * t_jb_val;
    real_t t_ib_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, b_);
    real_t t_ja_val = t1_host(t_ia, num_spin_occ, num_spin_vir, j, a_);
    sum -= t_ib_val * t_ja_val;
    return sum;
}

} // namespace cpu_helpers

__global__ void compute_F_ae_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ F_ae)
{
    size_t total = (size_t)num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir);

        int e = num_spin_occ + e_;
        int a = num_spin_occ + a_;

        real_t sum = 0.0;

        // (1-delta_ae) * f_ae
        // but always zero for RHF
        
        // sum over m
        // - 0.5 * f_me * t_m^a, but f_me = 0 for RHF
        // omitted

        // sum over m, f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                 
                real_t mafe = antisym_eri(d_eri_mo, num_basis, m, a, f, e); // <ma||fe> = (mf|ae) - (me|af)
                real_t t_mf_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, m, f_);
                sum += mafe * t_mf_val; // <ma||fe> * t_m^f
            }
        }

        // sum over m,n,f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int f = num_spin_occ + f_;
                    
                    real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                    real_t U_mnaf_val = U_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, m, n, a_, f_); // U_mnaf
                    sum -= 0.5 * mnef * U_mnaf_val; // -0.5 * <mn||ef> * U_mnaf
                }
            }
        }

        F_ae[gid] = sum;
    }
}

__global__ void compute_F_mi_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ F_mi)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int i  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int m  = (int)(t % num_spin_occ);

        real_t sum = 0.0;

        // (1-delta_mi) * f_mi
        // but always zero for RHF

        // sum over e, but RHF symmetry makes this zero (f_ia = 0)
        // 0.5*sum_e f_me * t_i^e
        // omitted

        // sum over n, e
        for(int n = 0; n < num_spin_occ; ++n){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = num_spin_occ + e_;
                
                real_t mnie = antisym_eri(d_eri_mo, num_basis, m, n, i, e); // <mn||ie> = (mi|ne) - (me|ni)
                real_t t_ne_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, n, e_);

                sum += mnie * t_ne_val; // <mn||ie> * t_n^e
            }
        }

        // sum over n, e, f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int e = num_spin_occ + e_;
                    int f = num_spin_occ + f_;
                    
                    real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                    real_t U_inef_val = U_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, i, n, e_, f_); // U_inef

                    sum += 0.5 * mnef * U_inef_val; // +0.5 * <mn||ef> * U_nief
                }
            }
        }

        F_mi[gid] = sum;
    }
}

__global__ void compute_F_me_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* F_me)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int m  = (int)(t % num_spin_occ);

        int e = num_spin_occ + e_;

        real_t sum = 0.0;

        // f_me
        // f_ia = 0 for RHF
        // omitted

        // sum over n, f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                real_t t_nf_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, n, f_);
                sum += mnef * t_nf_val; // <mn||ef> * t_n^f
            }
        }

        F_me[gid] = sum;
    }
}

__global__ void compute_W_mnij_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ W_mnij)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ; // 1d index is (i * num_spin_occ + j)  * num_spin_occ * num_spin_occ + k * num_spin_occ + n
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i = (int)(t % num_spin_occ); t /= num_spin_occ;
        int n  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int m  = (int)(t % num_spin_occ);

        real_t sum = 0.0;

        real_t mnij = antisym_eri(d_eri_mo, num_basis, m, n, i, j); // <mn||ij> = (mi|nj) - (mj|ni)
        sum += mnij;

        // sum ove e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = num_spin_occ + e_;
            
            // i, j (identity)
            real_t mnie = antisym_eri(d_eri_mo, num_basis, m, n, i, e); // <mn||ie> = (mi|ne) - (me|ni)
            real_t t_je_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, e_);
            sum += mnie * t_je_val; // <mn||ie> * t_j^e

            // swap i and j
            real_t mnje = antisym_eri(d_eri_mo, num_basis, m, n, j, e); // <mn||je> = (mj|ne) - (me|nj)
            real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, e_);
            sum -= mnje * t_ia_val; // - <mn||je> * t_i^e
        }

        // sum over e,f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int e = num_spin_occ + e_;
                int f = num_spin_occ + f_;
                
                real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                real_t t_ijef_val = T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, i, j, e_, f_);
                sum += (1.0) / (4.0) * mnef * t_ijef_val; // (1/4) * <mn||ef> * T_ij^ef
            }
        }
        W_mnij[gid] = sum;
    }
}


__global__ void compute_W_abef_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ W_abef)
{
    size_t total = (size_t)num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir; // 1d index is (a * num_spin_vir + b_)  * num_spin_vir * num_spin_vir + e_ * num_spin_vir + f_
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int f_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;
        int e = num_spin_occ + e_;
        int f = num_spin_occ + f_;

        real_t sum = 0.0;

        real_t abef = antisym_eri(d_eri_mo, num_basis, a, b, e, f); // <ab||ef> = (ae|bf) - (af|be)
        sum += abef;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            // a, b (identity)
            real_t amef = antisym_eri(d_eri_mo, num_basis, a, m, e, f); // <am||ef> = (ae|mf) - (af|me)
            real_t t_mb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, m, b_);

            sum -= amef * t_mb_val; // - <am||ef> * t_m^b

            // swap a and b
            real_t bmef = antisym_eri(d_eri_mo, num_basis, b, m, e, f); // <bm||ef> = (be|mf) - (bf|me)
            real_t t_ma_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, m, a_);

            sum += bmef * t_ma_val; // + <bm||ef> * t_m^a
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){   
            for(int n = 0; n < num_spin_occ; ++n){
                real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                real_t t_mnab_val = T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, m, n, a_, b_);

                sum += (1.0) / (4.0) * mnef * t_mnab_val; // (1/4) * <mn||ef> * T_mn^ab
            }
        }
        W_abef[gid] = sum;
    }
}


__global__ void compute_W_mbej_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ W_mbej)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ; // 1d index is (m * num_spin_vir + b_)  * num_spin_vir * num_spin_occ + e_ * num_spin_occ + j
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int m  = (int)(t % num_spin_occ);

        int e = num_spin_occ + e_;
        int b = num_spin_occ + b_;

        real_t sum = 0.0;

        real_t mbej = antisym_eri(d_eri_mo, num_basis, m, b, e, j); // <mb||ej> = (me|bj) - (mj|be)
        sum += mbej;

        // sum over f
        for(int f_ = 0; f_ < num_spin_vir; ++f_){
            int f = num_spin_occ + f_;
            
            real_t mbef = antisym_eri(d_eri_mo, num_basis, m, b, e, f); // <mb||ef> = (me|bf) - (mf|be)
            real_t t_jf_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, f_);

            sum += mbef * t_jf_val; // <mb||ef> * t_j^f
        }

        // sum over n
        for(int n = 0; n < num_spin_occ; ++n){
            real_t mnej = antisym_eri(d_eri_mo, num_basis, m, n, e, j); // <mn||ej> = (me|nj) - (mj|ne)
            real_t t_nb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, n, b_);

            sum -= mnej * t_nb_val; // - <mn||ej> * t_n^b
        }

        // sum over n,f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                real_t t_jnfb_val = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, j, n, f_, b_);
                real_t t_jf_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, f_);
                real_t t_nb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, n, b_);

                sum -= mnef  
                    * (0.5 * t_jnfb_val + t_jf_val * t_nb_val); // - <mn||ef> * (0.5 * T_jn^fb + t_j^f * t_n^b)
            }
        }

        W_mbej[gid] = sum;
    }
}



__global__ void compute_t_ia_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ d_eps, 
                                    const real_t* __restrict__ t_ia_old,
                                    const real_t* __restrict__ t_ijab_old,
                                    const real_t* __restrict__ F_ae,
                                    const real_t* __restrict__ F_mi,
                                    const real_t* __restrict__ F_me,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ t_ia_new)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;


        // skip spin incompatible combinations
        int sa = a_ % 2;
        int si = i % 2;
        if(sa != si){
            return;
        }


        real_t numerator = 0.0;

        // f_ia contribution is zero due to RHF symmetry

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            real_t t_ie_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
            real_t F_ae_val = F_ae[a_ * num_spin_vir + e_];

            numerator += F_ae_val * t_ie_val; // F_ae * t_i^e
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            real_t t_ma_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
            real_t F_mi_val = F_mi[m * num_spin_occ + i];

            numerator -= F_mi_val * t_ma_val; // - F_mi * t_m^a
        }

        // sum over m,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                real_t F_me_val = F_me[m * num_spin_vir + e_];
                real_t t_imae_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, e_);

                numerator += F_me_val * t_imae_val; // F_me * t_im^ae
            }
        }

        // sum over n,f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                real_t naif = antisym_eri(d_eri_mo, num_basis, n, a, i, f); // <na||if> = (ni|af) - (nf|ai)
                real_t t_nf_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, n, f_);

                numerator -= naif * t_nf_val; // - <na||if> * t_n^f
            }
        }

        // sum over m,e,f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int e = num_spin_occ + e_;
                    int f = num_spin_occ + f_;
                    
                    real_t maef = antisym_eri(d_eri_mo, num_basis, m, a, e, f); // <ma||ef> = (me|af) - (mf|ae)
                    real_t t_imef_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, m, e_, f_);

                    numerator -= (1.0) / (2.0) * maef * t_imef_val; // - (1/2) * <ma||ef> * t_im^ef
                }
            }
        }

        // sum over m,n,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                for(int e_ = 0; e_ < num_spin_vir; ++e_){
                    int e = num_spin_occ + e_;
                    
                    real_t nmei = antisym_eri(d_eri_mo, num_basis, n, m, e, i); // <nm||ei> = (ne|mi) - (ni|me)
                    real_t t_mnae_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, e_);

                    numerator -= (1.0) / (2.0) *  nmei * t_mnae_val; // - (1/2) * <nm||ei> * t_mn^ae
                }
            }
        }

        double denom = d_eps[i/2] - d_eps[a/2];
        // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
        if(fabs(denom) > 1e-14){
            t_ia_new[gid] = numerator / denom;
        } else {
            t_ia_new[gid] = 0.0;
        }
    }
}


__global__ void compute_t_ijab_kernel(const real_t* __restrict__ d_eri_mo, const real_t* __restrict__ d_eps, 
                                    const real_t* __restrict__ t_ia_old,
                                    const real_t* __restrict__ t_ijab_old,
                                    const real_t* __restrict__ F_ae,
                                    const real_t* __restrict__ F_mi,
                                    const real_t* __restrict__ F_me,
                                    const real_t* __restrict__ W_mnij,
                                    const real_t* __restrict__ W_abef,
                                    const real_t* __restrict__ W_mbej,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ t_ijab_new)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;

        // skip redundant calculations due to antisymmetry
        if(j <= i || b_ <= a_){ // other threads will fill in by antisymmetry
            return;
        }
        int si = (i%2);
        int sj = (j%2);
        int sa = (a_%2);
        int sb = (b_%2);
        if((si+sj)!=(sa+sb)){ // spin incompatible (0(aa), 1(ab), 1(ba), 2(bb))
            return;
        }



        real_t numerator = 0.0;

        real_t ijab = antisym_eri(d_eri_mo, num_basis, i, j, a, b); // <ij||ab> = (ia|jb) - (ib|ja)
        numerator += ijab;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            // a_, b_ (identity)
            real_t sum2 = 0.0;
            
            real_t F_be = F_ae[(b_ * num_spin_vir + e_)]; // F_be
            sum2 += F_be;

            // sum over m
            for(int m = 0; m < num_spin_occ; ++m){
                real_t t_mb_val = t_ia_old[m * num_spin_vir + b_];
                real_t F_me_val = F_me[m * num_spin_vir + e_];

                sum2 -= (1.0) / (2.0) * F_me_val * t_mb_val; // - (1/2) * F_me * t_m^b
            }

            real_t t_ijae_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, j, a_, e_);
            numerator += t_ijae_val * sum2; // + t_ij^ae * (...)

            // swap a_ and b_ for antisymmetry
            real_t sum2_asym = 0.0;
            
            real_t F_ae_val = F_ae[(a_ * num_spin_vir + e_)]; // F_ae
            sum2_asym += F_ae_val;

            // sum over m
            for(int m = 0; m < num_spin_occ; ++m){
                real_t t_ma_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                real_t F_me_val = F_me[(m * num_spin_vir + e_)];
                sum2_asym -= (1.0) / (2.0) * F_me_val * t_ma_val; // - (1/2) * F_me * t_m^a
            }

            real_t t_ijbe_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, j, b_, e_);
            numerator -= t_ijbe_val * sum2_asym; // - t_ij^be * (...)
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            // i, j (identity)
            real_t sum2 = 0.0;

            real_t F_mj_val = F_mi[(m * num_spin_occ + j)]; // F_mj
            sum2 += F_mj_val;   
            
            // sum over e
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                real_t t_je_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                real_t F_me_val = F_me[(m * num_spin_vir + e_)];
                sum2 += (1.0) / (2.0) * F_me_val * t_je_val; // + (1/2) * F_me * t_j^e
            }

            real_t t_imab_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, b_);

            numerator -= t_imab_val * sum2; // - t_im^ab * (...)

            // swap i and j for antisymmetry
            real_t sum2_asym = 0.0;
            real_t F_mi_val = F_mi[(m * num_spin_occ + i)]; // F_mi
            sum2_asym += F_mi_val;

            // sum over e
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                real_t t_ie_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                real_t F_me_val = F_me[(m * num_spin_vir + e_)];
                sum2_asym += (1.0) / (2.0) * F_me_val * t_ie_val; // + (1/2) * F_me * t_i^e
            }

            real_t t_jmab_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, j, m, a_, b_);
            numerator += t_jmab_val * sum2_asym; // + t_jm^ab * (...)
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                real_t T_mnab_val = T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, b_);
                real_t W_mnij_val = W_mnij[(m * num_spin_occ + n) * num_spin_occ * num_spin_occ + (i * num_spin_occ + j)];
                numerator += (1.0) / (2.0) * T_mnab_val * W_mnij_val; // + (1/2) * T_ij^ab * W_mnij
            }
        }

        // sum over e,f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                real_t T_ijef_val = T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, j, e_, f_);
                real_t W_abef_val = W_abef[(a_ * num_spin_vir + b_) * num_spin_vir * num_spin_vir + (e_ * num_spin_vir + f_)];
                numerator += (1.0) / (2.0) * T_ijef_val * W_abef_val; // + (1/2) * T_ij^ef * W_abef
            }
        }

        // sum over m,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = num_spin_occ + e_;

                // identity part
                real_t t_imae_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, e_);
                real_t W_mbej_val = W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)];
                real_t t_ie_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                real_t t_ma_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                real_t mbej = antisym_eri(d_eri_mo, num_basis, m, b, e, j); // <mb||ej> = (me|bj) - (mj|be)

                numerator += t_imae_val * W_mbej_val; // + t_im^ae * W_mbej
                numerator -= t_ie_val * t_ma_val * mbej; // - t_i^e *

                // swap a_ and b_ 
                real_t t_imbe_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, m, b_, e_);
                real_t W_maej_val = W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)];
                // already have t_ie
                real_t t_mb_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, b_);
                real_t maej = antisym_eri(d_eri_mo, num_basis, m, a, e, j); // <ma||ej> = (me|aj) - (mj|ae)

                numerator -= t_imbe_val * W_maej_val; // - t_im^be * W_maej
                numerator += t_ie_val * t_mb_val * maej; // + t_i^e * t_m^b * <ma||ej>

                // swap i and j
                real_t t_jmae_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, j, m, a_, e_);
                real_t W_mbei_val = W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)];
                real_t t_je_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                // already have t_ma
                real_t mbei = antisym_eri(d_eri_mo, num_basis, m, b, e, i); // <mb||ei> = (me|bi) - (mi|be)

                numerator -= t_jmae_val * W_mbei_val; // - t_jm^ae * W_mbei
                numerator += t_je_val * t_ma_val * mbei; // + t_j^e * t_m^a * <mb||ei>

                // swap a_ and b_, i and j
                real_t t_jmbe_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, j, m, b_, e_);
                real_t W_maei_val = W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)];
                // already have t_je
                // already have t_mb
                real_t maei = antisym_eri(d_eri_mo, num_basis, m, a, e, i); // <ma||ei> = (me|ai) - (mi|ae)
                
                numerator += t_jmbe_val * W_maei_val; // + t_jm^be * W_maei
                numerator -= t_je_val * t_mb_val * maei; // - t_j^e * t_m^b * <ma||ei>
            }
        }

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = num_spin_occ + e_;
            
            // i, j (identity)
            real_t abej = antisym_eri(d_eri_mo, num_basis, a, b, e, j); // <ab||ej> = (ae|bj) - (aj|be)
            real_t t_ie_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, i, e_);

            numerator += abej * t_ie_val; // + <ab||ej> * t_i^e

            // swap i and j
            real_t abei = antisym_eri(d_eri_mo, num_basis, a, b, e, i); // <ab||ei> = (ae|bi) - (ai|be)
            real_t t_je_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, j, e_);

            numerator -= abei * t_je_val; // - <ab||ei> * t_j^e
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            // a_, b_ (identity)                
            real_t mbij = antisym_eri(d_eri_mo, num_basis, m, b, i, j); // <mb||ij> = (mi|bj) - (mj|bi)
            real_t t_ma_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, a_);

            numerator -= mbij * t_ma_val; // - <mb||ij> * t_m^a

            // swap a_ and b_
            real_t maij = antisym_eri(d_eri_mo, num_basis, m, a, i, j); // <ma||ij> = (mi|aj) - (mj|ai)
            real_t t_mb_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, m, b_);

            numerator += maij * t_mb_val; // + <ma||ij> * t_m^b
        }



        real_t denom = d_eps[i/2] + d_eps[j/2] - d_eps[a/2] - d_eps[b/2];
        real_t t_ijab_val = 0.0;
        // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
        if(fabs(denom) > 1e-14){
            t_ijab_val = numerator / denom;
        } else {
            t_ijab_val = 0.0;
        }
        // Assign with antisymmetry t_ij^ab = - t_ji^ab = - t_ij^ba = t_ji^ba
        t_ijab_new[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = t_ijab_val;  // t_ij^ab
        t_ijab_new[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = -t_ijab_val; // t_ji^ab (= - t_ij^ab)
        t_ijab_new[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = -t_ijab_val; // t_ij^ba (= - t_ij^ab)
        t_ijab_new[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = t_ijab_val;  // t_ji^ba (= t_ij^ab)

    }
}

__global__ void compute_t_amplitude_max_norm_kernel(const real_t* __restrict__ t_ia_new,
                                        const real_t* __restrict__ t_ijab_new,
                                        const real_t* __restrict__ t_ia_old,
                                        const real_t* __restrict__ t_ijab_old,
                                        const int num_spin_occ,
                                        const int num_spin_vir,
                                        real_t* max_norm)
{
    __shared__ real_t local_max;

    if(threadIdx.x == 0){
        local_max = 0.0;
    }
    __syncthreads();

    size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total_ia){
        real_t diff = fabs(t_ia_new[gid] - t_ia_old[gid]);
        atomicMaxDouble(max_norm, diff);
    }
    size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    if(gid < total_ijab){
        real_t diff = fabs(t_ijab_new[gid] - t_ijab_old[gid]);
        atomicMaxDouble(max_norm, diff);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicMaxDouble(max_norm, local_max);
    }
}

void compute_t_amplitude(const real_t* __restrict__ d_eri_mo,
                            const real_t* __restrict__ d_eps,
                            const int num_basis,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            real_t* __restrict__ t_ia_old,
                            real_t* __restrict__ t_ijab_old,
                            real_t* __restrict__ t_ia_new,
                            real_t* __restrict__ t_ijab_new,
                            real_t* __restrict__ F_ae,
                            real_t* __restrict__ F_mi,
                            real_t* __restrict__ F_me,
                            real_t* __restrict__ W_mnij,
                            real_t* __restrict__ W_abef,
                            real_t* __restrict__ W_mbej)
{
    using namespace cpu_helpers;
    const int num_intermediates = 8; // F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, t_ia, t_ijab
    int computed_intermediates = 0;
    { // F_ae
        std::string str = "Computing F_ae intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_vir * num_spin_vir;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int a_ = (int)(t % num_spin_vir);
                int e = num_spin_occ + e_;
                int a = num_spin_occ + a_;
                real_t sum = 0.0;
                for (int m = 0; m < num_spin_occ; ++m) {
                    for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                        int f = num_spin_occ + f_;
                        sum += antisym_eri_host(d_eri_mo, num_basis, m, a, f, e) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, f_);
                    }
                }
                for (int m = 0; m < num_spin_occ; ++m) {
                    for (int n = 0; n < num_spin_occ; ++n) {
                        for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                            int f = num_spin_occ + f_;
                            sum -= 0.5 * antisym_eri_host(d_eri_mo, num_basis, m, n, e, f) * U_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, f_);
                        }
                    }
                }
                F_ae[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_vir * num_spin_vir;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_F_ae_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_ae);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // F_mi
        std::string str = "Computing F_mi intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_occ;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int i = (int)(t % num_spin_occ); t /= num_spin_occ;
                int m = (int)(t % num_spin_occ);
                real_t sum = 0.0;
                for (int n = 0; n < num_spin_occ; ++n) {
                    for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                        int e = num_spin_occ + e_;
                        sum += antisym_eri_host(d_eri_mo, num_basis, m, n, i, e) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, n, e_);
                    }
                }
                for (int n = 0; n < num_spin_occ; ++n) {
                    for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                        for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                            int e = num_spin_occ + e_;
                            int f = num_spin_occ + f_;
                            sum += 0.5 * antisym_eri_host(d_eri_mo, num_basis, m, n, e, f) * U_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, n, e_, f_);
                        }
                    }
                }
                F_mi[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_occ;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_F_mi_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_mi);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // F_me
        std::string str = "Computing F_me intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_vir;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int m = (int)(t % num_spin_occ);
                int e = num_spin_occ + e_;
                real_t sum = 0.0;
                for (int n = 0; n < num_spin_occ; ++n) {
                    for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                        int f = num_spin_occ + f_;
                        sum += antisym_eri_host(d_eri_mo, num_basis, m, n, e, f) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, n, f_);
                    }
                }
                F_me[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_vir;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_F_me_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_me);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // W_mnij
        std::string str = "Computing W_mnij intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int j = (int)(t % num_spin_occ); t /= num_spin_occ;
                int i = (int)(t % num_spin_occ); t /= num_spin_occ;
                int n = (int)(t % num_spin_occ); t /= num_spin_occ;
                int m = (int)(t % num_spin_occ);
                real_t sum = antisym_eri_host(d_eri_mo, num_basis, m, n, i, j);
                for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                    int e = num_spin_occ + e_;
                    sum += antisym_eri_host(d_eri_mo, num_basis, m, n, i, e) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                    sum -= antisym_eri_host(d_eri_mo, num_basis, m, n, j, e) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                }
                for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                    for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                        int e = num_spin_occ + e_;
                        int f = num_spin_occ + f_;
                        sum += 0.25 * antisym_eri_host(d_eri_mo, num_basis, m, n, e, f) * T_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, j, e_, f_);
                    }
                }
                W_mnij[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_W_mnij_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_mnij);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // W_abef
        std::string str = "Computing W_abef intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int f_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int a_ = (int)(t % num_spin_vir);
                int a = num_spin_occ + a_;
                int b = num_spin_occ + b_;
                int e = num_spin_occ + e_;
                int f = num_spin_occ + f_;
                real_t sum = antisym_eri_host(d_eri_mo, num_basis, a, b, e, f);
                for (int m = 0; m < num_spin_occ; ++m) {
                    sum -= antisym_eri_host(d_eri_mo, num_basis, a, m, e, f) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, b_);
                    sum += antisym_eri_host(d_eri_mo, num_basis, b, m, e, f) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                }
                for (int m = 0; m < num_spin_occ; ++m) {
                    for (int n = 0; n < num_spin_occ; ++n) {
                        sum += 0.25 * antisym_eri_host(d_eri_mo, num_basis, m, n, e, f) * T_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, b_);
                    }
                }
                W_abef[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_W_abef_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_abef);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // W_mbej
        std::string str = "Computing W_mbej intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
                int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int m  = (int)(t % num_spin_occ);
                int e = num_spin_occ + e_;
                int b = num_spin_occ + b_;
                real_t sum = antisym_eri_host(d_eri_mo, num_basis, m, b, e, j);
                for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                    int f = num_spin_occ + f_;
                    sum += antisym_eri_host(d_eri_mo, num_basis, m, b, e, f) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, f_);
                }
                for (int n = 0; n < num_spin_occ; ++n) {
                    sum -= antisym_eri_host(d_eri_mo, num_basis, m, n, e, j) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, n, b_);
                }
                for (int n = 0; n < num_spin_occ; ++n) {
                    for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                        int f = num_spin_occ + f_;
                        real_t mnef = antisym_eri_host(d_eri_mo, num_basis, m, n, e, f);
                        real_t t_jnfb = t2_host(t_ijab_old, num_spin_occ, num_spin_vir, j, n, f_, b_);
                        real_t t_jf = t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, f_);
                        real_t t_nb = t1_host(t_ia_old, num_spin_occ, num_spin_vir, n, b_);
                        sum -= mnef * (0.5 * t_jnfb + t_jf * t_nb);
                    }
                }
                W_mbej[gid] = sum;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_W_mbej_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_mbej);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    // Compute t_ia and t_ijab amplitudes
    { // t_ia_new
        std::string str = "Computing t_ia amplitudes... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_vir;
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int i  = (int)(t % num_spin_occ);
                int a = num_spin_occ + a_;
                if ((a_ % 2) != (i % 2)) { t_ia_new[gid] = 0.0; continue; }
                real_t numerator = 0.0;
                for (int e_ = 0; e_ < num_spin_vir; ++e_)
                    numerator += F_ae[a_ * num_spin_vir + e_] * t1_host(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                for (int m = 0; m < num_spin_occ; ++m)
                    numerator -= F_mi[m * num_spin_occ + i] * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                for (int m = 0; m < num_spin_occ; ++m)
                    for (int e_ = 0; e_ < num_spin_vir; ++e_)
                        numerator += F_me[m * num_spin_vir + e_] * t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, e_);
                for (int n = 0; n < num_spin_occ; ++n)
                    for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                        int f = num_spin_occ + f_;
                        numerator -= antisym_eri_host(d_eri_mo, num_basis, n, a, i, f) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, n, f_);
                    }
                for (int m = 0; m < num_spin_occ; ++m)
                    for (int e_ = 0; e_ < num_spin_vir; ++e_)
                        for (int f_ = 0; f_ < num_spin_vir; ++f_) {
                            int e = num_spin_occ + e_;
                            int f = num_spin_occ + f_;
                            numerator -= 0.5 * antisym_eri_host(d_eri_mo, num_basis, m, a, e, f) * t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, m, e_, f_);
                        }
                for (int m = 0; m < num_spin_occ; ++m)
                    for (int n = 0; n < num_spin_occ; ++n)
                        for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                            int e = num_spin_occ + e_;
                            numerator -= 0.5 * antisym_eri_host(d_eri_mo, num_basis, n, m, e, i) * t2_host(t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, e_);
                        }
                double denom = d_eps[i / 2] - d_eps[a / 2];
                t_ia_new[gid] = (fabs(denom) > 1e-14) ? numerator / denom : 0.0;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_vir;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_t_ia_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, t_ia_old, t_ijab_old, F_ae, F_mi, F_me, num_basis, num_spin_occ, num_spin_vir, t_ia_new);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
    { // t_ijab_new
        std::string str = "Computing t_ijab amplitudes... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        if (!gpu::gpu_available()) {
            const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
            // Zero out t_ijab_new first (elements skipped by the loop need to be zero)
            std::memset(t_ijab_new, 0, total * sizeof(real_t));
            #pragma omp parallel for schedule(dynamic)
            for (size_t gid = 0; gid < total; ++gid) {
                size_t t = gid;
                int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
                int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
                int i  = (int)(t % num_spin_occ);
                int a = num_spin_occ + a_;
                int b = num_spin_occ + b_;
                if (j <= i || b_ <= a_) continue;
                if (((i % 2) + (j % 2)) != ((a_ % 2) + (b_ % 2))) continue;

                real_t numerator = antisym_eri_host(d_eri_mo, num_basis, i, j, a, b);

                // sum over e (F_ae terms)
                for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                    real_t sum2 = F_ae[b_ * num_spin_vir + e_];
                    for (int m = 0; m < num_spin_occ; ++m)
                        sum2 -= 0.5 * F_me[m * num_spin_vir + e_] * t_ia_old[m * num_spin_vir + b_];
                    numerator += t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, j, a_, e_) * sum2;
                    real_t sum2_asym = F_ae[a_ * num_spin_vir + e_];
                    for (int m = 0; m < num_spin_occ; ++m)
                        sum2_asym -= 0.5 * F_me[m * num_spin_vir + e_] * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                    numerator -= t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, j, b_, e_) * sum2_asym;
                }
                // sum over m (F_mi terms)
                for (int m = 0; m < num_spin_occ; ++m) {
                    real_t sum2 = F_mi[m * num_spin_occ + j];
                    for (int e_ = 0; e_ < num_spin_vir; ++e_)
                        sum2 += 0.5 * F_me[m * num_spin_vir + e_] * t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                    numerator -= t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, b_) * sum2;
                    real_t sum2_asym = F_mi[m * num_spin_occ + i];
                    for (int e_ = 0; e_ < num_spin_vir; ++e_)
                        sum2_asym += 0.5 * F_me[m * num_spin_vir + e_] * t1_host(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                    numerator += t2_host(t_ijab_old, num_spin_occ, num_spin_vir, j, m, a_, b_) * sum2_asym;
                }
                // sum over m,n (W_mnij)
                for (int m = 0; m < num_spin_occ; ++m)
                    for (int n = 0; n < num_spin_occ; ++n)
                        numerator += 0.5 * T_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, b_)
                                   * W_mnij[(m * num_spin_occ + n) * num_spin_occ * num_spin_occ + (i * num_spin_occ + j)];
                // sum over e,f (W_abef)
                for (int e_ = 0; e_ < num_spin_vir; ++e_)
                    for (int f_ = 0; f_ < num_spin_vir; ++f_)
                        numerator += 0.5 * T_ijab_host(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, j, e_, f_)
                                   * W_abef[(a_ * num_spin_vir + b_) * num_spin_vir * num_spin_vir + (e_ * num_spin_vir + f_)];
                // sum over m,e (W_mbej terms with 4 permutations)
                for (int m = 0; m < num_spin_occ; ++m) {
                    for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                        int e = num_spin_occ + e_;
                        real_t t_ie = t1_host(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                        real_t t_je = t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                        real_t t_ma = t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                        real_t t_mb = t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, b_);
                        // identity
                        numerator += t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, m, a_, e_) * W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)];
                        numerator -= t_ie * t_ma * antisym_eri_host(d_eri_mo, num_basis, m, b, e, j);
                        // swap a,b
                        numerator -= t2_host(t_ijab_old, num_spin_occ, num_spin_vir, i, m, b_, e_) * W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)];
                        numerator += t_ie * t_mb * antisym_eri_host(d_eri_mo, num_basis, m, a, e, j);
                        // swap i,j
                        numerator -= t2_host(t_ijab_old, num_spin_occ, num_spin_vir, j, m, a_, e_) * W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)];
                        numerator += t_je * t_ma * antisym_eri_host(d_eri_mo, num_basis, m, b, e, i);
                        // swap a,b and i,j
                        numerator += t2_host(t_ijab_old, num_spin_occ, num_spin_vir, j, m, b_, e_) * W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)];
                        numerator -= t_je * t_mb * antisym_eri_host(d_eri_mo, num_basis, m, a, e, i);
                    }
                }
                // sum over e (direct ERI terms)
                for (int e_ = 0; e_ < num_spin_vir; ++e_) {
                    int e = num_spin_occ + e_;
                    numerator += antisym_eri_host(d_eri_mo, num_basis, a, b, e, j) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, i, e_);
                    numerator -= antisym_eri_host(d_eri_mo, num_basis, a, b, e, i) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                }
                // sum over m (direct ERI terms)
                for (int m = 0; m < num_spin_occ; ++m) {
                    numerator -= antisym_eri_host(d_eri_mo, num_basis, m, b, i, j) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, a_);
                    numerator += antisym_eri_host(d_eri_mo, num_basis, m, a, i, j) * t1_host(t_ia_old, num_spin_occ, num_spin_vir, m, b_);
                }

                real_t denom = d_eps[i / 2] + d_eps[j / 2] - d_eps[a / 2] - d_eps[b / 2];
                real_t t_ijab_val = (fabs(denom) > 1e-14) ? numerator / denom : 0.0;
                t_ijab_new[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = t_ijab_val;
                t_ijab_new[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = -t_ijab_val;
                t_ijab_new[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = -t_ijab_val;
                t_ijab_new[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = t_ijab_val;
            }
        } else {
            const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
            const int num_threads = 256;
            const int num_blocks = (total + num_threads - 1) / num_threads;
            compute_t_ijab_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, t_ia_old, t_ijab_old, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, num_basis, num_spin_occ, num_spin_vir, t_ijab_new);
            cudaDeviceSynchronize();
        }
        computed_intermediates++;
    }
}

real_t compute_t_amplitude_diff(const real_t* __restrict__ t_ia_new, const real_t* __restrict__ t_ijab_new,
                                    const real_t* __restrict__ t_ia_old, const real_t* __restrict__ t_ijab_old,
                                    const int num_spin_occ,
                                    const int num_spin_vir)
{
    if (!gpu::gpu_available()) {
        real_t max_norm = 0.0;
        const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
        const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        for (size_t gid = 0; gid < total_ia; ++gid) {
            real_t diff = fabs(t_ia_new[gid] - t_ia_old[gid]);
            if (diff > max_norm) max_norm = diff;
        }
        for (size_t gid = 0; gid < total_ijab; ++gid) {
            real_t diff = fabs(t_ijab_new[gid] - t_ijab_old[gid]);
            if (diff > max_norm) max_norm = diff;
        }
        return max_norm;
    } else {
        real_t h_max_norm = 0.0;
        real_t* d_max_norm = nullptr;
        tracked_cudaMalloc((void**)&d_max_norm, sizeof(real_t));
        if(!d_max_norm){
            THROW_EXCEPTION("tracked_cudaMalloc failed for d_max_norm.");
        }
        cudaMemset(d_max_norm, 0.0, sizeof(real_t));

        const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
        const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        const size_t total = (total_ia > total_ijab) ? total_ia : total_ijab;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_t_amplitude_max_norm_kernel<<<num_blocks, num_threads>>>(t_ia_new, t_ijab_new, t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, d_max_norm);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_max_norm, d_max_norm, sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_max_norm);

        return h_max_norm;
    }
}




__global__ void compute_t_amplitude_rms_kernel(const real_t* __restrict__ t_ia_new,
                                        const real_t* __restrict__ t_ijab_new,
                                        const real_t* __restrict__ t_ia_old,
                                        const real_t* __restrict__ t_ijab_old,
                                        const int num_spin_occ,
                                        const int num_spin_vir,
                                        real_t* rms)
{
    __shared__ real_t local_rms;

    if(threadIdx.x == 0){
        local_rms = 0.0;
    }
    __syncthreads();

    size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total_ia){
        real_t diff = t_ia_new[gid] - t_ia_old[gid];
        atomicAdd(&local_rms, diff * diff);
        
    }
    size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    if(gid < total_ijab){
        real_t diff = t_ijab_new[gid] - t_ijab_old[gid];
        atomicAdd(&local_rms, diff * diff);
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(rms, local_rms);
    }
}

real_t compute_t_amplitude_rms(const real_t* __restrict__ t_ia_new, const real_t* __restrict__ t_ijab_new,
                                    const real_t* __restrict__ t_ia_old, const real_t* __restrict__ t_ijab_old,
                                    const int num_spin_occ,
                                    const int num_spin_vir)
{
    if (!gpu::gpu_available()) {
        real_t rms_sum = 0.0;
        const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
        const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        for (size_t gid = 0; gid < total_ia; ++gid) {
            real_t diff = t_ia_new[gid] - t_ia_old[gid];
            rms_sum += diff * diff;
        }
        for (size_t gid = 0; gid < total_ijab; ++gid) {
            real_t diff = t_ijab_new[gid] - t_ijab_old[gid];
            rms_sum += diff * diff;
        }
        return sqrt(rms_sum);
    } else {
        real_t h_rms = 0.0;
        real_t* d_rms = nullptr;
        tracked_cudaMalloc((void**)&d_rms, sizeof(real_t));
        if(!d_rms){
            THROW_EXCEPTION("tracked_cudaMalloc failed for d_rms.");
        }
        cudaMemset(d_rms, 0.0, sizeof(real_t));

        const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
        const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        const size_t total = (total_ia > total_ijab) ? total_ia : total_ijab;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_t_amplitude_rms_kernel<<<num_blocks, num_threads>>>(t_ia_new, t_ijab_new, t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, d_rms);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_rms, d_rms, sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_rms);

        return sqrt(h_rms);
    }
}


__global__ void update_t_amplitude_damping_kernel(const real_t* __restrict__ t_new,
                                                real_t* __restrict__ t_old,
                                                const int dim1,
                                                const int dim2,
                                                const real_t damping_factor)
{
    size_t total = (size_t)dim1 * dim2;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        t_old[gid] = (1.0 - damping_factor) * t_old[gid] + damping_factor * t_new[gid];
    }
}

void update_t_amplitude_damping(const real_t* t_ia_new, const real_t* t_ijab_new,
                                real_t* t_ia_old, real_t* t_ijab_old,
                                const int num_spin_occ,
                                const int num_spin_vir,
                                const real_t damping_factor)
{
    const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
    const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;

    if (!gpu::gpu_available()) {
        // t_ia_old = (1 - damping_factor) * t_ia_old + damping_factor * t_ia_new
        #pragma omp parallel for
        for (size_t gid = 0; gid < total_ia; ++gid)
            t_ia_old[gid] = (1.0 - damping_factor) * t_ia_old[gid] + damping_factor * t_ia_new[gid];
        // t_ijab_old = (1 - damping_factor) * t_ijab_old + damping_factor * t_ijab_new
        #pragma omp parallel for
        for (size_t gid = 0; gid < total_ijab; ++gid)
            t_ijab_old[gid] = (1.0 - damping_factor) * t_ijab_old[gid] + damping_factor * t_ijab_new[gid];
    } else {
        // t_ia_old = (1 - damping_factor) * t_ia_old + damping_factor * t_ia_new
        const int num_threads = 256;
        const int num_blocks_ia = (total_ia + num_threads - 1) / num_threads;
        update_t_amplitude_damping_kernel<<<num_blocks_ia, num_threads>>>(t_ia_new, t_ia_old, num_spin_occ, num_spin_vir, damping_factor);
        cudaDeviceSynchronize();

        // t_ijab_old = (1 - damping_factor) * t_ijab_old + damping_factor * t_ijab_new
        const int num_blocks_ijab = (total_ijab + num_threads - 1) / num_threads;
        update_t_amplitude_damping_kernel<<<num_blocks_ijab, num_threads>>>(t_ijab_new, t_ijab_old, num_spin_occ * num_spin_occ, num_spin_vir * num_spin_vir, damping_factor);
        cudaDeviceSynchronize();
    }
}

__global__ void compute_ccsd_energy_kernel(const real_t* __restrict__ d_eri_mo,
                                            const int num_basis,
                                            const int num_spin_occ,
                                            const int num_spin_vir,
                                            const real_t* __restrict__ t_ia,
                                            const real_t* __restrict__ t_ijab,
                                            real_t* d_ccsd_energy)
{
    assert(blockDim.x <= 256); // ensure local_sum size is sufficient

    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    real_t contrib = 0.0;

    // loop over all i,j,a,b
    if(gid < total){
        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;

        
        // <ij||ab> = (ia|jb) - (ib|ja)
        real_t ijab = antisym_eri(d_eri_mo, num_basis, i, j, a, b);
        
        real_t t_ijab_val = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
        real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_);
        real_t t_jb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, b_);

        contrib += 0.5 * ijab * t_ia_val * t_jb_val; // 0.5 * <ij||ab> * t_i^a * t_j^b
        contrib += 0.25 * ijab * t_ijab_val; // 0.25 * <ij||ab> * t_ij^ab
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_ccsd_energy, block_sum);
    }
}


real_t compute_ccsd_energy(const real_t* __restrict__ d_eri_mo,
                            const int num_basis,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            const real_t* __restrict__ t_ia,
                            const real_t* __restrict__ t_ijab)
{
    using namespace cpu_helpers;

    if (!gpu::gpu_available()) {
        real_t energy = 0.0;
        const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        for (size_t gid = 0; gid < total; ++gid) {
            size_t t = gid;
            int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
            int i  = (int)(t % num_spin_occ);
            int a = num_spin_occ + a_;
            int b = num_spin_occ + b_;
            real_t ijab = antisym_eri_host(d_eri_mo, num_basis, i, j, a, b);
            real_t t_ijab_val = t2_host(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
            real_t t_ia_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, a_);
            real_t t_jb_val = t1_host(t_ia, num_spin_occ, num_spin_vir, j, b_);
            energy += 0.5 * ijab * t_ia_val * t_jb_val;
            energy += 0.25 * ijab * t_ijab_val;
        }
        return energy;
    } else {
        real_t h_ccsd_energy = 0.0;
        real_t* d_ccsd_energy = nullptr;
        tracked_cudaMalloc((void**)&d_ccsd_energy, sizeof(real_t));
        if(!d_ccsd_energy){
            THROW_EXCEPTION("tracked_cudaMalloc failed for d_ccsd_energy.");
        }
        cudaMemset(d_ccsd_energy, 0.0, sizeof(real_t));

        const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        const int shmem_size = num_threads * sizeof(real_t);
        compute_ccsd_energy_kernel<<<num_blocks, num_threads, shmem_size>>>(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, d_ccsd_energy);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_ccsd_energy, d_ccsd_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_ccsd_energy);

        return h_ccsd_energy;
    }
}


void allocate_ccsd_intermediates(const int num_spin_occ, const int num_spin_vir,
                                        real_t** F_ae,
                                        real_t** F_mi,
                                        real_t** F_me,
                                        real_t** W_mnij,
                                        real_t** W_abef,
                                        real_t** W_mbej)
{
    // intermediates
    tracked_cudaMalloc((void**)F_ae, sizeof(real_t) * num_spin_vir * num_spin_vir);
    tracked_cudaMalloc((void**)F_mi, sizeof(real_t) * num_spin_occ * num_spin_occ);
    tracked_cudaMalloc((void**)F_me, sizeof(real_t) * num_spin_occ * num_spin_vir);
    tracked_cudaMalloc((void**)W_mnij, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ);
    tracked_cudaMalloc((void**)W_abef, sizeof(real_t) * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir);
    tracked_cudaMalloc((void**)W_mbej, sizeof(real_t) * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ);

    // error checks
    if(!(*F_ae) || !(*F_mi) || !(*F_me) || !(*W_mnij) || !(*W_abef) || !(*W_mbej)){
        THROW_EXCEPTION("tracked_cudaMalloc failed for CCSD intermediates.");
    }
}

void allocate_ccsd_amplitudes(const int num_spin_occ, const int num_spin_vir,
                                        real_t** t_ia_new,
                                        real_t** t_ia_old,
                                        real_t** t_ijab_new,
                                        real_t** t_ijab_old)
{
    // amplitudes
    // Allocate a single buffer for t_ia and t_ijab as a contiguous block for both new and old amplitudes
    size_t num_t1 = num_spin_occ * num_spin_vir;
    size_t num_t2 = num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;

    real_t* t1t2_new_buffer = nullptr;
    real_t* t1t2_old_buffer = nullptr;

    tracked_cudaMalloc((void**)&t1t2_new_buffer, sizeof(real_t) * (num_t1 + num_t2));
    tracked_cudaMalloc((void**)&t1t2_old_buffer, sizeof(real_t) * (num_t1 + num_t2));
    if(!t1t2_new_buffer || !t1t2_old_buffer){
        THROW_EXCEPTION("tracked_cudaMalloc failed for CCSD amplitudes buffer.");
    }
    *t_ia_new = t1t2_new_buffer;
    *t_ijab_new = t1t2_new_buffer + num_t1;
    *t_ia_old = t1t2_old_buffer;
    *t_ijab_old = t1t2_old_buffer + num_t1;

}

void deallocate_ccsd_intermediates(real_t* __restrict__ F_ae,
                                                real_t* __restrict__ F_mi,
                                                real_t* __restrict__ F_me,
                                                real_t* __restrict__ W_mnij,
                                                real_t* __restrict__ W_abef,
                                                real_t* __restrict__ W_mbej)
{
    tracked_cudaFree(F_ae);
    tracked_cudaFree(F_mi);
    tracked_cudaFree(F_me);
    tracked_cudaFree(W_mnij);
    tracked_cudaFree(W_abef);
    tracked_cudaFree(W_mbej);
}


void deallocate_ccsd_amplitudes(real_t* __restrict__ t_ia_new,
                                real_t* __restrict__ t_ia_old,
                                real_t* __restrict__ t_ijab_new,
                                real_t* __restrict__ t_ijab_old)
{
    // t_ijab_new and t_ijab_old are part of t_ia_new and t_ia_old buffers, so no need to free them separately    
    tracked_cudaFree(t_ia_new); // free both t_ia_new and t_ijab_new as they are in the same buffer
    tracked_cudaFree(t_ia_old); // free both t_ia_old and t_ijab_old as they are in the same buffer
}

__global__ void initialize_ccsd_amplitudes_kernel(const real_t* __restrict__ d_eri_mo,
                                    const real_t* __restrict__ d_eps,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    real_t* __restrict__ t_ijab)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;

        // skip antisymmetric cases
        if( (i > j) || (a_ > b_) ){
            return;
        }
        // skip spin-incompatible cases
        int spin_i = i % 2;
        int spin_j = j % 2;
        int spin_a = a_ % 2;
        int spin_b = b_ % 2;
        if( (spin_i + spin_j) != (spin_a + spin_b) ){ // 0(alpha,alpha), 2(beta,beta) or 1(alpha,beta)
            t_ijab[gid] = 0.0;
            return;
        }

        // <ij||ab> = (ia|jb) - (ib|ja)
        real_t ijab = antisym_eri(d_eri_mo, num_basis, i, j, a, b);

        double denom = d_eps[i/2] + d_eps[j/2] - d_eps[a/2] - d_eps[b/2];
        // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
        if(fabs(denom) > 1e-14){
            double t_ijab_val = ijab / denom;

            // Assign with antisymmetry t_ij^ab = - t_ji^ab = - t_ij^ba = t_ji^ba
            t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = t_ijab_val;  // t_ij^ab
            t_ijab[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = -t_ijab_val; // t_ji^ab (= - t_ij^ab)
            t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = -t_ijab_val; // t_ij^ba (= - t_ij^ab)
            t_ijab[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = t_ijab_val;  // t_ji^ba (= t_ij^ab)
        } else {
            t_ijab[gid] = 0.0;
        }
        // debug
        /*
        printf("iajb = %f\n", iajb);
        printf("ibja = %f\n", ibja);
        printf("daemon = %f\n", denom);
         printf("t_ijab(%d,%d,%d,%d) = %f\n", i, j, a_, b_, t_ijab[gid]);
         */
    }
}

void intialize_ccsd_amplitudes(const real_t* __restrict__ d_eri_mo,
                                const real_t* __restrict__ d_eps,
                                const int num_basis,
                                const int num_spin_occ,
                                const int num_spin_vir,
                                real_t* __restrict__ t_ijab)
{
    using namespace cpu_helpers;
    const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;

    if (!gpu::gpu_available()) {
        std::memset(t_ijab, 0, total * sizeof(real_t));
        #pragma omp parallel for schedule(dynamic)
        for (size_t gid = 0; gid < total; ++gid) {
            size_t t = gid;
            int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
            int i  = (int)(t % num_spin_occ);
            int a = num_spin_occ + a_;
            int b = num_spin_occ + b_;
            if (i > j || a_ > b_) continue;
            if (((i % 2) + (j % 2)) != ((a_ % 2) + (b_ % 2))) { t_ijab[gid] = 0.0; continue; }
            real_t ijab = antisym_eri_host(d_eri_mo, num_basis, i, j, a, b);
            double denom = d_eps[i / 2] + d_eps[j / 2] - d_eps[a / 2] - d_eps[b / 2];
            if (fabs(denom) > 1e-14) {
                double t_ijab_val = ijab / denom;
                t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = t_ijab_val;
                t_ijab[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] = -t_ijab_val;
                t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = -t_ijab_val;
                t_ijab[(j * num_spin_occ + i) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + a_)] = t_ijab_val;
            }
        }
    } else {
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        initialize_ccsd_amplitudes_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ijab);
        cudaDeviceSynchronize();
    }
}



/////////////////// CCSD(T) Energy Calculation ///////////////////
// Ref. Chapter 9.5 in Many-Body Methods in Chemistry and Physics by I. Shavitt and R.J. Bartlett
// t_ijk^abc in Eq. (10.35)


// Precomputed permutations for 3 indices P(i|jk)f(ijk) = f(ijk) - f(jik) - f(kji)
__device__ __constant__ int perms3[3][3] = {
    {0,1,2}, // f(ijk)
    {1,0,2}, // -f(jik)
    {2,1,0}  // -f(kji)
};

__device__ __constant__ int parity3[3] = {
    +1,  // f(ijk)
    -1,  // -f(jik)
    -1  //  -f(kji)
};


__global__ void compute_ccsd_t_energy_kernel(const real_t* __restrict__ d_eri_mo,
                                    const real_t* __restrict__ d_eps,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const real_t* __restrict__ t_ia,
                                    const real_t* __restrict__ t_ijab,
                                    real_t* d_ccsd_t_energy)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ real_t local_sum[256]; // assuming max 256 threads per block
    if(threadIdx.x < 256){
        local_sum[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if(gid < total){
        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;
        int c = num_spin_occ + c_;
/*
        // skip spin-incompatible cases
        int spin_i = i % 2;
        int spin_j = j % 2;
        int spin_k = k % 2;
        int spin_a = a_ % 2;
        int spin_b = b_ % 2;
        int spin_c = c_ % 2;
        if( (spin_i + spin_j + spin_k) != (spin_a + spin_b + spin_c) ){ // 0(alpha,alpha,alpha), 3(beta,beta,beta) or 1/3(mixed)
            return;
        }
        // skip when same indices appear
        if( (i == j) || (i == k) || (j == k) || (a_ == b_) || (a_ == c_) || (b_ == c_) ){
            return;
        }
*/
        // Compute the contribution to CCSD(T) energy from (i,j,k,a,b,c)
        double contrib = 0.0;

        double denom = d_eps[i/2] + d_eps[j/2] + d_eps[k/2] - d_eps[a/2] - d_eps[b/2] - d_eps[c/2];
        if(fabs(denom) < 1e-14){
            denom = 1e-14; // avoid division by zero
        }

        double T_ijk_abc = 0.0;

        { // first part: compute T_ijk^abc
            // P(k|ij) P(a|bc) 
            int occ[3] = {k, i, j};
            int vir_[3] = {a_, b_, c_};
            int vir[3] = {a, b, c};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(k|ij) 
                int kk = occ[ perms3[p1][0] ];
                int ii = occ[ perms3[p1][1] ];
                int jj = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    int aa_ = vir_[ perms3[p2][0] ];
                    //int bb_ = vir_[ perms3[p2][1] ];
                    //int cc_ = vir_[ perms3[p2][2] ];
                    //int aa = vir[ perms3[p2][0] ];
                    int bb = vir[ perms3[p2][1] ];
                    int cc = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over d
                    for(int d_ = 0; d_ < num_spin_vir; ++d_){
                        int d = num_spin_occ + d_;

                        real_t bcdk = antisym_eri(d_eri_mo, num_basis, bb, cc, d, kk);// <bc||dk> = (bd|ck) - (bk|dc)
                        real_t t_ijad = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, jj, aa_, d_); // t_ij^ad
                        
                        T_ijk_abc += sign * bcdk * t_ijad; // sign * <bc||dk> * t_ij^ad
                    }
                }
            }
        }
        { // second part: compute T_ijk^abc
            // P(i|jk) P(c|ab) 
            int occ[3] = {i, j, k};
            int vir_[3] = {c_, a_, b_};
            int vir[3] = {c, a, b};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(i|jk) 
                int ii = occ[ perms3[p1][0] ];
                int jj = occ[ perms3[p1][1] ];
                int kk = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    //int cc_ = vir_[ perms3[p2][0] ];
                    int aa_ = vir_[ perms3[p2][1] ];
                    int bb_ = vir_[ perms3[p2][2] ];
                    int cc = vir[ perms3[p2][0] ];
                    //int aa = vir[ perms3[p2][1] ];
                    //int bb = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over l
                    for(int l = 0; l < num_spin_occ; ++l){
                        real_t lcjk = antisym_eri(d_eri_mo, num_basis, l, cc, jj, kk); // <lc||jk> = (lj|ck) - (lk|cj)
                        real_t t_ilab = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, l, aa_, bb_); // t_il^ab

                        T_ijk_abc -= sign * lcjk * t_ilab; // sign * <lc||jk> * t_il^ab
                    }
                }
            }
        }
        
        T_ijk_abc /= denom;

        // E(4) contribution
        {
            contrib += (1.0/36.0) * T_ijk_abc * T_ijk_abc * denom;
        }

        // E(5) contribution
        {
            real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_); // t_i^a
            real_t jkbc = antisym_eri(d_eri_mo, num_basis, j, k, b, c); // <jk||bc> = (jb|kc) - (jc|kb)

            contrib += (1.0/4.0) * T_ijk_abc * t_ia_val * jkbc; // (1/4) * T_ijk^abc * t_i^a * <jk||bc>            

        }

        local_sum[threadIdx.x] += contrib;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        real_t block_sum = 0.0;
        for(int i = 0; i < blockDim.x; ++i){
            block_sum += local_sum[i];
        }
        atomicAdd(d_ccsd_t_energy, block_sum);
    }
}


__global__ void compute_ccsd_t_energy_vir1_kernel(const real_t* __restrict__ d_eri_mo,
                                    const real_t* __restrict__ d_eps,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const real_t* __restrict__ t_ia,
                                    const real_t* __restrict__ t_ijab,
                                    const int vir1,
                                    real_t* d_ccsd_t_energy)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ real_t local_sum[256]; // assuming max 256 threads per block
    if(threadIdx.x < 256){
        local_sum[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if(gid < total){
        size_t t = gid;
        int c_ = vir1;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;
        int c = num_spin_occ + c_;
/*
        // skip spin-incompatible cases
        int spin_i = i % 2;
        int spin_j = j % 2;
        int spin_k = k % 2;
        int spin_a = a_ % 2;
        int spin_b = b_ % 2;
        int spin_c = c_ % 2;
        if( (spin_i + spin_j + spin_k) != (spin_a + spin_b + spin_c) ){ // 0(alpha,alpha,alpha), 3(beta,beta,beta) or 1/3(mixed)
            return;
        }
        // skip when same indices appear
        if( (i == j) || (i == k) || (j == k) || (a_ == b_) || (a_ == c_) || (b_ == c_) ){
            return;
        }
*/
        // Compute the contribution to CCSD(T) energy from (i,j,k,a,b,c)
        double contrib = 0.0;

        double denom = d_eps[i/2] + d_eps[j/2] + d_eps[k/2] - d_eps[a/2] - d_eps[b/2] - d_eps[c/2];
        if(fabs(denom) < 1e-14){
            denom = 1e-14; // avoid division by zero
        }

        double T_ijk_abc = 0.0;

        { // first part: compute T_ijk^abc
            // P(k|ij) P(a|bc) 
            int occ[3] = {k, i, j};
            int vir_[3] = {a_, b_, c_};
            int vir[3] = {a, b, c};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(k|ij) 
                int kk = occ[ perms3[p1][0] ];
                int ii = occ[ perms3[p1][1] ];
                int jj = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    int aa_ = vir_[ perms3[p2][0] ];
                    //int bb_ = vir_[ perms3[p2][1] ];
                    //int cc_ = vir_[ perms3[p2][2] ];
                    //int aa = vir[ perms3[p2][0] ];
                    int bb = vir[ perms3[p2][1] ];
                    int cc = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over d
                    for(int d_ = 0; d_ < num_spin_vir; ++d_){
                        int d = num_spin_occ + d_;

                        real_t bcdk = antisym_eri(d_eri_mo, num_basis, bb, cc, d, kk);// <bc||dk> = (bd|ck) - (bk|dc)
                        real_t t_ijad = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, jj, aa_, d_); // t_ij^ad
                        
                        T_ijk_abc += sign * bcdk * t_ijad; // sign * <bc||dk> * t_ij^ad
                    }
                }
            }
        }
        { // second part: compute T_ijk^abc
            // P(i|jk) P(c|ab) 
            int occ[3] = {i, j, k};
            int vir_[3] = {c_, a_, b_};
            int vir[3] = {c, a, b};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(i|jk) 
                int ii = occ[ perms3[p1][0] ];
                int jj = occ[ perms3[p1][1] ];
                int kk = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    //int cc_ = vir_[ perms3[p2][0] ];
                    int aa_ = vir_[ perms3[p2][1] ];
                    int bb_ = vir_[ perms3[p2][2] ];
                    int cc = vir[ perms3[p2][0] ];
                    //int aa = vir[ perms3[p2][1] ];
                    //int bb = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over l
                    for(int l = 0; l < num_spin_occ; ++l){
                        real_t lcjk = antisym_eri(d_eri_mo, num_basis, l, cc, jj, kk); // <lc||jk> = (lj|ck) - (lk|cj)
                        real_t t_ilab = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, l, aa_, bb_); // t_il^ab

                        T_ijk_abc -= sign * lcjk * t_ilab; // sign * <lc||jk> * t_il^ab
                    }
                }
            }
        }
        
        T_ijk_abc /= denom;

        // E(4) contribution
        {
            contrib += (1.0/36.0) * T_ijk_abc * T_ijk_abc * denom;
        }

        // E(5) contribution
        {
            real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_); // t_i^a
            real_t jkbc = antisym_eri(d_eri_mo, num_basis, j, k, b, c); // <jk||bc> = (jb|kc) - (jc|kb)

            contrib += (1.0/4.0) * T_ijk_abc * t_ia_val * jkbc; // (1/4) * T_ijk^abc * t_i^a * <jk||bc>            

        }

        local_sum[threadIdx.x] += contrib;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        real_t block_sum = 0.0;
        for(int i = 0; i < blockDim.x; ++i){
            block_sum += local_sum[i];
        }
        atomicAdd(d_ccsd_t_energy, block_sum);
    }
}



__global__ void compute_ccsd_t_energy_vir2_kernel(const real_t* __restrict__ d_eri_mo,
                                    const real_t* __restrict__ d_eps,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const real_t* __restrict__ t_ia,
                                    const real_t* __restrict__ t_ijab,
                                    const int vir1,
                                    const int vir2,
                                    real_t* d_ccsd_t_energy)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ real_t local_sum[256]; // assuming max 256 threads per block
    if(threadIdx.x < 256){
        local_sum[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if(gid < total){
        size_t t = gid;
        int c_ = vir2;
        int b_ = vir1;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;
        int c = num_spin_occ + c_;
/*
        // skip spin-incompatible cases
        int spin_i = i % 2;
        int spin_j = j % 2;
        int spin_k = k % 2;
        int spin_a = a_ % 2;
        int spin_b = b_ % 2;
        int spin_c = c_ % 2;
        if( (spin_i + spin_j + spin_k) != (spin_a + spin_b + spin_c) ){ // 0(alpha,alpha,alpha), 3(beta,beta,beta) or 1/3(mixed)
            return;
        }
        // skip when same indices appear
        if( (i == j) || (i == k) || (j == k) || (a_ == b_) || (a_ == c_) || (b_ == c_) ){
            return;
        }
*/
        // Compute the contribution to CCSD(T) energy from (i,j,k,a,b,c)
        double contrib = 0.0;

        double denom = d_eps[i/2] + d_eps[j/2] + d_eps[k/2] - d_eps[a/2] - d_eps[b/2] - d_eps[c/2];
        if(fabs(denom) < 1e-14){
            denom = 1e-14; // avoid division by zero
        }

        double T_ijk_abc = 0.0;

        { // first part: compute T_ijk^abc
            // P(k|ij) P(a|bc) 
            int occ[3] = {k, i, j};
            int vir_[3] = {a_, b_, c_};
            int vir[3] = {a, b, c};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(k|ij) 
                int kk = occ[ perms3[p1][0] ];
                int ii = occ[ perms3[p1][1] ];
                int jj = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    int aa_ = vir_[ perms3[p2][0] ];
                    //int bb_ = vir_[ perms3[p2][1] ];
                    //int cc_ = vir_[ perms3[p2][2] ];
                    //int aa = vir[ perms3[p2][0] ];
                    int bb = vir[ perms3[p2][1] ];
                    int cc = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over d
                    for(int d_ = 0; d_ < num_spin_vir; ++d_){
                        int d = num_spin_occ + d_;

                        real_t bcdk = antisym_eri(d_eri_mo, num_basis, bb, cc, d, kk);// <bc||dk> = (bd|ck) - (bk|dc)
                        real_t t_ijad = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, jj, aa_, d_); // t_ij^ad
                        
                        T_ijk_abc += sign * bcdk * t_ijad; // sign * <bc||dk> * t_ij^ad
                    }
                }
            }
        }
        { // second part: compute T_ijk^abc
            // P(i|jk) P(c|ab) 
            int occ[3] = {i, j, k};
            int vir_[3] = {c_, a_, b_};
            int vir[3] = {c, a, b};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(i|jk) 
                int ii = occ[ perms3[p1][0] ];
                int jj = occ[ perms3[p1][1] ];
                int kk = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    //int cc_ = vir_[ perms3[p2][0] ];
                    int aa_ = vir_[ perms3[p2][1] ];
                    int bb_ = vir_[ perms3[p2][2] ];
                    int cc = vir[ perms3[p2][0] ];
                    //int aa = vir[ perms3[p2][1] ];
                    //int bb = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over l
                    for(int l = 0; l < num_spin_occ; ++l){
                        real_t lcjk = antisym_eri(d_eri_mo, num_basis, l, cc, jj, kk); // <lc||jk> = (lj|ck) - (lk|cj)
                        real_t t_ilab = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, l, aa_, bb_); // t_il^ab

                        T_ijk_abc -= sign * lcjk * t_ilab; // sign * <lc||jk> * t_il^ab
                    }
                }
            }
        }
        
        T_ijk_abc /= denom;

        // E(4) contribution
        {
            contrib += (1.0/36.0) * T_ijk_abc * T_ijk_abc * denom;
        }

        // E(5) contribution
        {
            real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_); // t_i^a
            real_t jkbc = antisym_eri(d_eri_mo, num_basis, j, k, b, c); // <jk||bc> = (jb|kc) - (jc|kb)

            contrib += (1.0/4.0) * T_ijk_abc * t_ia_val * jkbc; // (1/4) * T_ijk^abc * t_i^a * <jk||bc>            

        }

        local_sum[threadIdx.x] += contrib;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        real_t block_sum = 0.0;
        for(int i = 0; i < blockDim.x; ++i){
            block_sum += local_sum[i];
        }
        atomicAdd(d_ccsd_t_energy, block_sum);
    }
}



__global__ void compute_ccsd_t_energy_vir3_kernel(const real_t* __restrict__ d_eri_mo,
                                    const real_t* __restrict__ d_eps,
                                    const int num_basis,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const real_t* __restrict__ t_ia,
                                    const real_t* __restrict__ t_ijab,
                                    const int vir1,
                                    const int vir2,
                                    const int vir3,
                                    real_t* d_ccsd_t_energy)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ real_t local_sum[256]; // assuming max 256 threads per block
    if(threadIdx.x < 256){
        local_sum[threadIdx.x] = 0.0;
    }
    __syncthreads();

    if(gid < total){
        size_t t = gid;
        int c_ = vir3;
        int b_ = vir2;
        int a_ = vir1;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;
        int c = num_spin_occ + c_;
/*
        // skip spin-incompatible cases
        int spin_i = i % 2;
        int spin_j = j % 2;
        int spin_k = k % 2;
        int spin_a = a_ % 2;
        int spin_b = b_ % 2;
        int spin_c = c_ % 2;
        if( (spin_i + spin_j + spin_k) != (spin_a + spin_b + spin_c) ){ // 0(alpha,alpha,alpha), 3(beta,beta,beta) or 1/3(mixed)
            return;
        }
        // skip when same indices appear
        if( (i == j) || (i == k) || (j == k) || (a_ == b_) || (a_ == c_) || (b_ == c_) ){
            return;
        }
*/
        // Compute the contribution to CCSD(T) energy from (i,j,k,a,b,c)
        double contrib = 0.0;

        double denom = d_eps[i/2] + d_eps[j/2] + d_eps[k/2] - d_eps[a/2] - d_eps[b/2] - d_eps[c/2];
        if(fabs(denom) < 1e-14){
            denom = 1e-14; // avoid division by zero
        }

        double T_ijk_abc = 0.0;

        { // first part: compute T_ijk^abc
            // P(k|ij) P(a|bc) 
            int occ[3] = {k, i, j};
            int vir_[3] = {a_, b_, c_};
            int vir[3] = {a, b, c};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(k|ij) 
                int kk = occ[ perms3[p1][0] ];
                int ii = occ[ perms3[p1][1] ];
                int jj = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    int aa_ = vir_[ perms3[p2][0] ];
                    //int bb_ = vir_[ perms3[p2][1] ];
                    //int cc_ = vir_[ perms3[p2][2] ];
                    //int aa = vir[ perms3[p2][0] ];
                    int bb = vir[ perms3[p2][1] ];
                    int cc = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over d
                    for(int d_ = 0; d_ < num_spin_vir; ++d_){
                        int d = num_spin_occ + d_;

                        real_t bcdk = antisym_eri(d_eri_mo, num_basis, bb, cc, d, kk);// <bc||dk> = (bd|ck) - (bk|dc)
                        real_t t_ijad = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, jj, aa_, d_); // t_ij^ad
                        
                        T_ijk_abc += sign * bcdk * t_ijad; // sign * <bc||dk> * t_ij^ad
                    }
                }
            }
        }
        { // second part: compute T_ijk^abc
            // P(i|jk) P(c|ab) 
            int occ[3] = {i, j, k};
            int vir_[3] = {c_, a_, b_};
            int vir[3] = {c, a, b};

            for(int p1 = 0; p1 < 3; ++p1){ // permutations over (i,j,k)
                // P(i|jk) 
                int ii = occ[ perms3[p1][0] ];
                int jj = occ[ perms3[p1][1] ];
                int kk = occ[ perms3[p1][2] ];
                double sign1 = parity3[p1];

                for(int p2 = 0; p2 < 3; ++p2){
                    //int cc_ = vir_[ perms3[p2][0] ];
                    int aa_ = vir_[ perms3[p2][1] ];
                    int bb_ = vir_[ perms3[p2][2] ];
                    int cc = vir[ perms3[p2][0] ];
                    //int aa = vir[ perms3[p2][1] ];
                    //int bb = vir[ perms3[p2][2] ];
                    double sign2 = parity3[p2];

                    double sign = sign1 * sign2;

                    // sum over l
                    for(int l = 0; l < num_spin_occ; ++l){
                        real_t lcjk = antisym_eri(d_eri_mo, num_basis, l, cc, jj, kk); // <lc||jk> = (lj|ck) - (lk|cj)
                        real_t t_ilab = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, ii, l, aa_, bb_); // t_il^ab

                        T_ijk_abc -= sign * lcjk * t_ilab; // sign * <lc||jk> * t_il^ab
                    }
                }
            }
        }
        
        T_ijk_abc /= denom;

        // E(4) contribution
        {
            contrib += (1.0/36.0) * T_ijk_abc * T_ijk_abc * denom;
        }

        // E(5) contribution
        {
            real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_); // t_i^a
            real_t jkbc = antisym_eri(d_eri_mo, num_basis, j, k, b, c); // <jk||bc> = (jb|kc) - (jc|kb)

            contrib += (1.0/4.0) * T_ijk_abc * t_ia_val * jkbc; // (1/4) * T_ijk^abc * t_i^a * <jk||bc>            

        }

        local_sum[threadIdx.x] += contrib;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        real_t block_sum = 0.0;
        for(int i = 0; i < blockDim.x; ++i){
            block_sum += local_sum[i];
        }
        atomicAdd(d_ccsd_t_energy, block_sum);
    }
}





real_t compute_ccsd_t_energy(const real_t* __restrict__ d_eri_mo,
                                const real_t* __restrict__ d_eps,
                                const int num_basis,
                                const int num_spin_occ,
                                const int num_spin_vir,
                                const real_t* __restrict__ t_ia,
                                const real_t* __restrict__ t_ijab)
{
    using namespace cpu_helpers;

    if (!gpu::gpu_available()) {
        // CPU fallback for CCSD(T) perturbative triples
        // Permutation tables (host-side equivalent of __device__ __constant__)
        static const int h_perms3[3][3] = { {0,1,2}, {1,0,2}, {2,1,0} };
        static const int h_parity3[3] = { +1, -1, -1 };

        std::cout << "Computing CCSD(T) energy on CPU ..." << std::endl;
        const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ
                           * num_spin_vir * num_spin_vir * num_spin_vir;
        real_t energy = 0.0;
        #pragma omp parallel for schedule(dynamic) reduction(+:energy)
        for (size_t gid = 0; gid < total; ++gid) {
            size_t t = gid;
            int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
            int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
            int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
            int i  = (int)(t % num_spin_occ);
            int a = num_spin_occ + a_;
            int b = num_spin_occ + b_;
            int c = num_spin_occ + c_;

            double denom = d_eps[i/2] + d_eps[j/2] + d_eps[k/2] - d_eps[a/2] - d_eps[b/2] - d_eps[c/2];
            if (fabs(denom) < 1e-14) denom = 1e-14;

            double T_ijk_abc = 0.0;
            { // first part: P(k|ij) P(a|bc)
                int occ[3] = {k, i, j};
                int vir_loc[3] = {a_, b_, c_};
                int vir_full[3] = {a, b, c};
                for (int p1 = 0; p1 < 3; ++p1) {
                    int kk = occ[h_perms3[p1][0]];
                    int ii = occ[h_perms3[p1][1]];
                    int jj = occ[h_perms3[p1][2]];
                    double sign1 = h_parity3[p1];
                    for (int p2 = 0; p2 < 3; ++p2) {
                        int aa_ = vir_loc[h_perms3[p2][0]];
                        int bb = vir_full[h_perms3[p2][1]];
                        int cc = vir_full[h_perms3[p2][2]];
                        double sign = sign1 * h_parity3[p2];
                        for (int d_ = 0; d_ < num_spin_vir; ++d_) {
                            int d = num_spin_occ + d_;
                            T_ijk_abc += sign * antisym_eri_host(d_eri_mo, num_basis, bb, cc, d, kk)
                                             * t2_host(t_ijab, num_spin_occ, num_spin_vir, ii, jj, aa_, d_);
                        }
                    }
                }
            }
            { // second part: P(i|jk) P(c|ab)
                int occ[3] = {i, j, k};
                int vir_loc[3] = {c_, a_, b_};
                int vir_full[3] = {c, a, b};
                for (int p1 = 0; p1 < 3; ++p1) {
                    int ii = occ[h_perms3[p1][0]];
                    int jj = occ[h_perms3[p1][1]];
                    int kk = occ[h_perms3[p1][2]];
                    double sign1 = h_parity3[p1];
                    for (int p2 = 0; p2 < 3; ++p2) {
                        int aa_ = vir_loc[h_perms3[p2][1]];
                        int bb_ = vir_loc[h_perms3[p2][2]];
                        int cc = vir_full[h_perms3[p2][0]];
                        double sign = sign1 * h_parity3[p2];
                        for (int l = 0; l < num_spin_occ; ++l) {
                            T_ijk_abc -= sign * antisym_eri_host(d_eri_mo, num_basis, l, cc, jj, kk)
                                              * t2_host(t_ijab, num_spin_occ, num_spin_vir, ii, l, aa_, bb_);
                        }
                    }
                }
            }
            T_ijk_abc /= denom;

            double contrib = 0.0;
            // E(4) contribution
            contrib += (1.0 / 36.0) * T_ijk_abc * T_ijk_abc * denom;
            // E(5) contribution
            real_t t_ia_val = t1_host(t_ia, num_spin_occ, num_spin_vir, i, a_);
            real_t jkbc = antisym_eri_host(d_eri_mo, num_basis, j, k, b, c);
            contrib += (1.0 / 4.0) * T_ijk_abc * t_ia_val * jkbc;

            energy += contrib;
        }
        return energy;
    } else {
        // GPU path
        real_t h_E_CCSD_T = 0.0;
        real_t* d_E_CCSD_T = nullptr;
        tracked_cudaMalloc((void**)&d_E_CCSD_T, sizeof(real_t));
        if(!d_E_CCSD_T){
            THROW_EXCEPTION("tracked_cudaMalloc failed for d_E_CCSD_T.");
        }
        cudaMemset(d_E_CCSD_T, 0.0, sizeof(real_t));

        const int num_threads = 256;
        const size_t max_blocks = (1ULL << 31) - 1;

        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
        size_t num_blocks = (total + num_threads - 1) / num_threads;
        int num_fixed_vir = 0;

        for(num_fixed_vir=0; num_fixed_vir<=3; ++num_fixed_vir){
            if(num_blocks <= max_blocks){
                break;
            }
            total /= num_spin_vir;
            num_blocks = (total + num_threads - 1) / num_threads;
        }
        if(num_fixed_vir == 0){
            std::cout << "Computing CCSD(T) energy with full parallelization over (i,j,k,a,b,c) ..." << std::endl;
            compute_ccsd_t_energy_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, d_E_CCSD_T);
        }else if(num_fixed_vir == 1){
            std::cout << "Computing CCSD(T) energy with partial parallelization over (i,j,k,a,b) x c times ..." << std::endl;
            for(int vir1 = 0; vir1 < num_spin_vir; ++vir1){
                compute_ccsd_t_energy_vir1_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, vir1, d_E_CCSD_T);
            }
        }else if(num_fixed_vir == 2){
            std::cout << "Computing CCSD(T) energy with partial parallelization over (i,j,k,a) x (b,c) times ..." << std::endl;
            for(int vir1 = 0; vir1 < num_spin_vir; ++vir1){
                for(int vir2 = 0; vir2 < num_spin_vir; ++vir2){
                    compute_ccsd_t_energy_vir2_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, vir1, vir2, d_E_CCSD_T);
                }
            }
        }else if(num_fixed_vir == 3){
            std::cout << "Computing CCSD(T) energy with partial parallelization over (i,j,k) x (a,b,c) times ..." << std::endl;
            for(int vir1 = 0; vir1 < num_spin_vir; ++vir1){
                for(int vir2 = 0; vir2 < num_spin_vir; ++vir2){
                    for(int vir3 = 0; vir3 < num_spin_vir; ++vir3){
                        compute_ccsd_t_energy_vir3_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, vir1, vir2, vir3, d_E_CCSD_T);
                    }
                }
            }
        }else{
            THROW_EXCEPTION("Error in compute_ccsd_t_energy: num_spin_vir is too large.");
        }

        cudaMemcpy(&h_E_CCSD_T, d_E_CCSD_T, sizeof(real_t), cudaMemcpyDeviceToHost);
        tracked_cudaFree(d_E_CCSD_T);

        return h_E_CCSD_T;
    }
}




// ============================================================
//  Spatial-orbital closed-shell CCSD implementation
// ============================================================
//
// MO integrals: (pq|rs) in chemist's notation, stored as d_eri_mo[p*N*N*N + q*N*N + r*N + s]
// where N = num_basis (spatial orbitals).
//
// T1 amplitudes: t1[i*nvir + a]  (nocc × nvir)
// T2 amplitudes: t2[i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b]  i.e. (ia,jb) ordering
//   This stores T2(αβ). T2(αα) = t2(ia,jb) - t2(ib,ja).
//
// For canonical HF: f_ia = 0, f_ab = eps_a * delta_ab, f_ij = eps_i * delta_ij.
//
// References: Stanton, Gauss, Watts, Bartlett, JCP 94, 4334 (1991)
//             Scuseria, Janssen, Schaefer, JCP 89, 7382 (1988)
// ============================================================

// ============================================================
//  Naive spatial-orbital CCSD (no GPU DGEMM, no pre-built sub-blocks)
//  Kept for benchmarking against the optimized version.
// ============================================================
real_t ccsd_spatial_orbital_naive(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t* d_eri_mo_precomputed = nullptr)
{
    const int N = num_basis;
    const int nocc = num_occ;
    const int nvir = N - nocc;
    const size_t NN = (size_t)N * N;
    const size_t NNN = NN * N;
    const size_t N4 = NN * NN;

    std::cout << "CCSD spatial-orbital (naive, no GPU optimization): N=" << N
              << " nocc=" << nocc << " nvir=" << nvir << std::endl;

    // AO->MO transform on GPU, download to host
    real_t* d_eri_mo;
    bool free_eri_mo;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        d_eri_mo = nullptr;
        tracked_cudaMalloc((void**)&d_eri_mo, N4 * sizeof(real_t));
        {
            std::string str = "Computing AO -> MO full integral transformation... ";
            PROFILE_ELAPSED_TIME(str);
            transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix, N, d_eri_mo);
            cudaDeviceSynchronize();
        }
        free_eri_mo = true;
    }
    std::vector<real_t> eri(N4);
    cudaMemcpy(eri.data(), d_eri_mo, N4 * sizeof(real_t), cudaMemcpyDeviceToHost);
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);

    std::vector<real_t> eps(N);
    cudaMemcpy(eps.data(), d_orbital_energies, N * sizeof(real_t), cudaMemcpyDeviceToHost);

    auto v = [&](int p, int q, int r, int s) -> real_t {
        return eri[(size_t)p * NNN + (size_t)r * NN + (size_t)q * N + s];
    };
    auto w = [&](int p, int q, int r, int s) -> real_t {
        return 2.0 * v(p,q,r,s) - v(p,q,s,r);
    };
    auto T2 = [&](int i, int j, int a, int b) -> size_t {
        return ((size_t)i * nocc + j) * nvir * nvir + (size_t)a * nvir + b;
    };

    const size_t t1Size = (size_t)nocc * nvir;
    const size_t t2Size = (size_t)nocc * nocc * nvir * nvir;

    std::vector<real_t> Dia(t1Size);
    for (int i = 0; i < nocc; i++)
        for (int a = 0; a < nvir; a++)
            Dia[i*nvir+a] = eps[i] - eps[nocc+a];

    std::vector<real_t> Dijab(t2Size);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++)
                    Dijab[T2(i,j,a,b)] = eps[i] + eps[j] - eps[nocc+a] - eps[nocc+b];

    // MP2 initial guess
    std::vector<real_t> t1(t1Size, 0.0);
    std::vector<real_t> t2v(t2Size);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++)
                    t2v[T2(i,j,a,b)] = v(i,j,nocc+a,nocc+b) / Dijab[T2(i,j,a,b)];

    auto energy = [&]() -> real_t {
        real_t E = 0.0;
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++)
                        E += w(i,j,nocc+a,nocc+b) * (t2v[T2(i,j,a,b)] + t1[i*nvir+a]*t1[j*nvir+b]);
        return E;
    };

    real_t Ecc = energy();
    std::cout << "CCSD iter  0: E = " << std::fixed << std::setprecision(12)
              << Ecc << " (MP2 initial guess)" << std::endl;

    DIIS diis(8, 2);
    size_t num_amps = t1Size + t2Size;
    const int MAX_ITER = 100;
    const real_t CONV = 1e-10;

    for (int iter = 1; iter <= MAX_ITER; iter++) {

        // ---- F intermediates ----
        std::vector<real_t> Fkc(nocc * nvir, 0.0);
        for (int k = 0; k < nocc; k++)
            for (int c = 0; c < nvir; c++) {
                real_t val = 0.0;
                for (int l = 0; l < nocc; l++)
                    for (int d = 0; d < nvir; d++)
                        val += w(k,l,nocc+c,nocc+d) * t1[l*nvir+d];
                Fkc[k*nvir+c] = val;
            }

        std::vector<real_t> Fki(nocc * nocc, 0.0);
        for (int k = 0; k < nocc; k++)
            for (int i = 0; i < nocc; i++) {
                real_t val = 0.0;
                for (int l = 0; l < nocc; l++)
                    for (int c = 0; c < nvir; c++)
                        for (int d = 0; d < nvir; d++) {
                            real_t ww = w(k,l,nocc+c,nocc+d);
                            val += ww * (t2v[T2(i,l,c,d)] + t1[i*nvir+c]*t1[l*nvir+d]);
                        }
                Fki[k*nocc+i] = val;
            }

        std::vector<real_t> Fac(nvir * nvir, 0.0);
        for (int a = 0; a < nvir; a++)
            for (int c = 0; c < nvir; c++) {
                real_t val = 0.0;
                for (int k = 0; k < nocc; k++)
                    for (int l = 0; l < nocc; l++)
                        for (int d = 0; d < nvir; d++) {
                            real_t ww = w(k,l,nocc+c,nocc+d);
                            val -= ww * (t2v[T2(k,l,a,d)] + t1[k*nvir+a]*t1[l*nvir+d]);
                        }
                Fac[a*nvir+c] = val;
            }

        // ---- L intermediates ----
        std::vector<real_t> Lki(nocc * nocc, 0.0);
        for (int k = 0; k < nocc; k++)
            for (int i = 0; i < nocc; i++) {
                real_t val = Fki[k*nocc+i];
                for (int l = 0; l < nocc; l++)
                    for (int c = 0; c < nvir; c++)
                        val += w(l,k,nocc+c,i) * t1[l*nvir+c];
                Lki[k*nocc+i] = val;
            }

        std::vector<real_t> Lac(nvir * nvir, 0.0);
        for (int a = 0; a < nvir; a++)
            for (int c = 0; c < nvir; c++) {
                real_t val = Fac[a*nvir+c];
                for (int k = 0; k < nocc; k++)
                    for (int d = 0; d < nvir; d++)
                        val += w(k,nocc+a,nocc+d,nocc+c) * t1[k*nvir+d];
                Lac[a*nvir+c] = val;
            }

        // ---- W^{kl}_{ij} ----
        std::vector<real_t> Wklij((size_t)nocc*nocc*nocc*nocc);
        for (int k = 0; k < nocc; k++)
            for (int l = 0; l < nocc; l++)
                for (int i = 0; i < nocc; i++)
                    for (int j = 0; j < nocc; j++) {
                        real_t val = v(k,l,i,j);
                        for (int c = 0; c < nvir; c++) {
                            int C = nocc+c;
                            val += v(l,k,C,i) * t1[j*nvir+c];
                            val += v(k,l,C,j) * t1[i*nvir+c];
                        }
                        for (int c = 0; c < nvir; c++)
                            for (int d = 0; d < nvir; d++)
                                val += v(k,l,nocc+c,nocc+d) * (t2v[T2(i,j,c,d)] + t1[i*nvir+c]*t1[j*nvir+d]);
                        Wklij[((size_t)k*nocc+l)*nocc*nocc + (size_t)i*nocc+j] = val;
                    }

        // ---- W^{ab}_{cd} ----
        std::vector<real_t> Wabcd((size_t)nvir*nvir*nvir*nvir);
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++) {
                int A=nocc+a, B=nocc+b;
                for (int c = 0; c < nvir; c++)
                    for (int d = 0; d < nvir; d++) {
                        int C=nocc+c, D=nocc+d;
                        real_t val = v(A,B,C,D);
                        for (int k = 0; k < nocc; k++) {
                            val -= v(k,A,D,C) * t1[k*nvir+b];
                            val -= v(k,B,C,D) * t1[k*nvir+a];
                        }
                        Wabcd[((size_t)a*nvir+b)*nvir*nvir + (size_t)c*nvir+d] = val;
                    }
            }

        // ---- W^{ak}_{ic} ----
        std::vector<real_t> Wakic((size_t)nvir*nocc*nocc*nvir);
        for (int a = 0; a < nvir; a++)
            for (int k = 0; k < nocc; k++)
                for (int i = 0; i < nocc; i++)
                    for (int c = 0; c < nvir; c++) {
                        int A=nocc+a, C=nocc+c;
                        real_t val = v(A,k,i,C);
                        for (int l = 0; l < nocc; l++)
                            val -= v(k,l,C,i) * t1[l*nvir+a];
                        for (int d = 0; d < nvir; d++)
                            val += v(k,A,C,nocc+d) * t1[i*nvir+d];
                        for (int l = 0; l < nocc; l++)
                            for (int d = 0; d < nvir; d++) {
                                real_t vlk = v(l,k,nocc+d,C);
                                val -= 0.5 * vlk * t2v[T2(i,l,d,a)];
                                val -= vlk * t1[i*nvir+d] * t1[l*nvir+a];
                                val += 0.5 * w(l,k,nocc+d,C) * t2v[T2(i,l,a,d)];
                            }
                        Wakic[((size_t)a*nocc+k)*nocc*nvir + (size_t)i*nvir+c] = val;
                    }

        // ---- W^{ak}_{ci} ----
        std::vector<real_t> Wakci((size_t)nvir*nocc*nvir*nocc);
        for (int a = 0; a < nvir; a++)
            for (int k = 0; k < nocc; k++)
                for (int c = 0; c < nvir; c++)
                    for (int i = 0; i < nocc; i++) {
                        int A=nocc+a, C=nocc+c;
                        real_t val = v(A,k,C,i);
                        for (int l = 0; l < nocc; l++)
                            val -= v(l,k,C,i) * t1[l*nvir+a];
                        for (int d = 0; d < nvir; d++)
                            val += v(k,A,nocc+d,C) * t1[i*nvir+d];
                        for (int l = 0; l < nocc; l++)
                            for (int d = 0; d < nvir; d++) {
                                real_t vlk = v(l,k,C,nocc+d);
                                val -= 0.5 * vlk * t2v[T2(i,l,d,a)];
                                val -= vlk * t1[i*nvir+d] * t1[l*nvir+a];
                            }
                        Wakci[((size_t)a*nocc+k)*nvir*nocc + (size_t)c*nocc+i] = val;
                    }

        // ---- T1 update ----
        std::vector<real_t> newT1(t1Size, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int a = 0; a < nvir; a++) {
                int A = nocc+a;
                real_t val = 0.0;
                for (int c = 0; c < nvir; c++)
                    val += Fac[a*nvir+c] * t1[i*nvir+c];
                for (int k = 0; k < nocc; k++)
                    val -= Fki[k*nocc+i] * t1[k*nvir+a];
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++) {
                        real_t fc = Fkc[k*nvir+c];
                        val += fc * (2.0*t2v[T2(k,i,c,a)] - t2v[T2(i,k,c,a)] + t1[i*nvir+c]*t1[k*nvir+a]);
                    }
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++)
                        val += w(A,k,i,nocc+c) * t1[k*nvir+c];
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++)
                        for (int d = 0; d < nvir; d++)
                            val += w(A,k,nocc+c,nocc+d) * (t2v[T2(i,k,c,d)] + t1[i*nvir+c]*t1[k*nvir+d]);
                for (int k = 0; k < nocc; k++)
                    for (int l = 0; l < nocc; l++)
                        for (int c = 0; c < nvir; c++)
                            val -= w(k,l,i,nocc+c) * (t2v[T2(k,l,a,c)] + t1[k*nvir+a]*t1[l*nvir+c]);
                newT1[i*nvir+a] = val / Dia[i*nvir+a];
            }

        // ---- T2 update ----
        std::vector<real_t> raw(t2Size, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int a = 0; a < nvir; a++) {
                int A = nocc+a;
                for (int j = 0; j < nocc; j++)
                    for (int b = 0; b < nvir; b++) {
                        int B = nocc+b;
                        real_t val = 0.5 * v(i,j,A,B);
                        // Wklij * tau
                        for (int k = 0; k < nocc; k++)
                            for (int l = 0; l < nocc; l++)
                                val += 0.5 * Wklij[((size_t)k*nocc+l)*nocc*nocc+(size_t)i*nocc+j]
                                     * (t2v[T2(k,l,a,b)] + t1[k*nvir+a]*t1[l*nvir+b]);
                        // Wabcd * tau
                        for (int c = 0; c < nvir; c++)
                            for (int d = 0; d < nvir; d++)
                                val += 0.5 * Wabcd[((size_t)a*nvir+b)*nvir*nvir+(size_t)c*nvir+d]
                                     * (t2v[T2(i,j,c,d)] + t1[i*nvir+c]*t1[j*nvir+d]);
                        for (int c = 0; c < nvir; c++)
                            val += Lac[a*nvir+c] * t2v[T2(i,j,c,b)];
                        for (int k = 0; k < nocc; k++)
                            val -= Lki[k*nocc+i] * t2v[T2(k,j,a,b)];
                        for (int c = 0; c < nvir; c++)
                            val += v(A,B,i,nocc+c) * t1[j*nvir+c];
                        for (int k = 0; k < nocc; k++)
                            for (int c = 0; c < nvir; c++)
                                val -= v(k,B,i,nocc+c) * t1[k*nvir+a] * t1[j*nvir+c];
                        for (int k = 0; k < nocc; k++)
                            val -= v(A,k,i,j) * t1[k*nvir+b];
                        for (int k = 0; k < nocc; k++)
                            for (int c = 0; c < nvir; c++)
                                val -= v(A,k,i,nocc+c) * t1[j*nvir+c] * t1[k*nvir+b];
                        for (int k = 0; k < nocc; k++)
                            for (int c = 0; c < nvir; c++) {
                                real_t w1 = Wakic[((size_t)a*nocc+k)*nocc*nvir + (size_t)i*nvir+c];
                                real_t w2 = Wakci[((size_t)a*nocc+k)*nvir*nocc + (size_t)c*nocc+i];
                                real_t w3 = Wakci[((size_t)b*nocc+k)*nvir*nocc + (size_t)c*nocc+i];
                                val += 2.0 * w1 * t2v[T2(k,j,c,b)];
                                val -= w2 * t2v[T2(k,j,c,b)];
                                val -= w1 * t2v[T2(k,j,b,c)];
                                val -= w3 * t2v[T2(k,j,a,c)];
                            }
                        raw[T2(i,j,a,b)] = val;
                    }
            }

        std::vector<real_t> newT2(t2Size);
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++) {
                        size_t idx = T2(i,j,a,b);
                        newT2[idx] = (raw[idx] + raw[T2(j,i,b,a)]) / Dijab[idx];
                    }

        // ---- DIIS ----
        std::vector<real_t> ampVec(num_amps);
        std::vector<real_t> errVec(num_amps);
        for (size_t k = 0; k < t1Size; k++) { ampVec[k] = newT1[k]; errVec[k] = newT1[k] - t1[k]; }
        for (size_t k = 0; k < t2Size; k++) { ampVec[t1Size+k] = newT2[k]; errVec[t1Size+k] = newT2[k] - t2v[k]; }
        diis.push(ampVec, errVec);
        if (diis.can_extrapolate()) {
            auto extrap = diis.extrapolate();
            for (size_t k = 0; k < t1Size; k++) newT1[k] = extrap[k];
            for (size_t k = 0; k < t2Size; k++) newT2[k] = extrap[t1Size + k];
        }

        for (size_t k = 0; k < t1Size; k++) t1[k] = newT1[k];
        for (size_t k = 0; k < t2Size; k++) t2v[k] = newT2[k];

        real_t newEcc = energy();
        real_t deltaE = newEcc - Ecc;
        Ecc = newEcc;
        std::cout << "CCSD iter " << std::setw(2) << iter
                  << ": E = " << std::fixed << std::setprecision(12) << Ecc
                  << ", dE = " << std::scientific << std::setprecision(4) << deltaE << std::endl;
        if (std::abs(deltaE) < CONV) {
            std::cout << "CCSD converged after " << iter << " iterations" << std::endl;
            break;
        }
    }

    // ---- (T) correction ----
    if (computing_ccsd_t && ccsd_t_energy) {
        std::cout << "---- Computing (T) correction (naive spatial orbital) ----" << std::endl;
        std::string str = "Computing (T) correction energy... ";
        PROFILE_ELAPSED_TIME(str);
        const size_t o3 = (size_t)nocc * nocc * nocc;
        real_t E_T = 0.0;
        std::vector<std::vector<size_t>> idx(6, std::vector<size_t>(o3));
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int k = 0; k < nocc; k++) {
                    size_t ijk = ((size_t)i*nocc+j)*nocc+k;
                    idx[0][ijk] = ijk;
                    idx[1][ijk] = ((size_t)i*nocc+k)*nocc+j;
                    idx[2][ijk] = ((size_t)j*nocc+i)*nocc+k;
                    idx[3][ijk] = ((size_t)j*nocc+k)*nocc+i;
                    idx[4][ijk] = ((size_t)k*nocc+i)*nocc+j;
                    idx[5][ijk] = ((size_t)k*nocc+j)*nocc+i;
                }
        int comp[6][6] = {
            {0,1,2,3,4,5},{1,0,4,5,2,3},{2,3,0,1,5,4},
            {4,5,1,0,3,2},{3,2,5,4,0,1},{5,4,3,2,1,0}
        };
        std::vector<std::vector<real_t>> wt(6, std::vector<real_t>(o3));
        std::vector<std::vector<real_t>> zt(6, std::vector<real_t>(o3));
        std::vector<real_t> wpv(o3), r3out(o3);
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b <= a; b++)
                for (int c = 0; c <= b; c++) {
                    real_t d3_scale = 1.0;
                    if (a == c) d3_scale = 6.0;
                    else if (a == b || b == c) d3_scale = 2.0;
                    int perms[6][3] = {{a,b,c},{a,c,b},{b,a,c},{b,c,a},{c,a,b},{c,b,a}};
                    for (int p = 0; p < 6; p++) {
                        int aa=perms[p][0], bb=perms[p][1], cc=perms[p][2];
                        int AA=nocc+aa, BB=nocc+bb;
                        for (int i = 0; i < nocc; i++)
                            for (int j = 0; j < nocc; j++)
                                for (int k = 0; k < nocc; k++) {
                                    size_t ijk = ((size_t)i*nocc+j)*nocc+k;
                                    real_t wval = 0.0;
                                    for (int f = 0; f < nvir; f++)
                                        wval += v(i,nocc+f,AA,BB) * t2v[T2(k,j,cc,f)];
                                    for (int m = 0; m < nocc; m++)
                                        wval -= v(AA,j,i,m) * t2v[T2(m,k,bb,cc)];
                                    real_t vval = v(AA,BB,i,j) * t1[k*nvir+cc];
                                    wpv[ijk] = wval + 0.5 * vval;
                                    wt[p][ijk] = wval;
                                }
                        for (size_t q = 0; q < o3; q++)
                            r3out[q] = 4.0*wpv[q] + wpv[idx[3][q]] + wpv[idx[4][q]]
                                     - 2.0*wpv[idx[5][q]] - 2.0*wpv[idx[1][q]] - 2.0*wpv[idx[2][q]];
                        for (int i = 0; i < nocc; i++)
                            for (int j = 0; j < nocc; j++)
                                for (int k = 0; k < nocc; k++) {
                                    real_t D = (eps[i]+eps[j]+eps[k]-eps[nocc+aa]-eps[nocc+bb]-eps[nocc+cc])*d3_scale;
                                    zt[p][((size_t)i*nocc+j)*nocc+k] = r3out[((size_t)i*nocc+j)*nocc+k] / D;
                                }
                    }
                    for (int q = 0; q < 6; q++)
                        for (int p = 0; p < 6; p++) {
                            int s = comp[q][p];
                            real_t eterm = 0.0;
                            for (size_t r = 0; r < o3; r++)
                                eterm += wt[p][idx[s][r]] * zt[q][r];
                            E_T += eterm;
                        }
                }
        E_T *= 2.0;
        *ccsd_t_energy = E_T;
        std::cout << "(T) correction energy: " << std::fixed << std::setprecision(12) << E_T << std::endl;
    }

    return Ecc;
}


// ============================================================
//  Optimized spatial-orbital CCSD (GPU DGEMM + pre-built integral sub-blocks)
// ============================================================
// v^{pq}_{rs} = (pr|qs),  w^{pq}_{rs} = 2*(pr|qs) - (ps|qr)
// P(ia,jb) f = f(i,a,j,b) + f(j,b,i,a)

real_t ccsd_spatial_orbital(const real_t* __restrict__ d_eri_ao,
                            const real_t* __restrict__ d_coefficient_matrix,
                            const real_t* __restrict__ d_orbital_energies,
                            const int num_basis, const int num_occ,
                            const bool computing_ccsd_t, real_t* ccsd_t_energy,
                            real_t** d_t1_out, real_t** d_t2_out,
                            real_t* d_eri_mo_precomputed = nullptr)
{
    const int N = num_basis;
    const int nocc = num_occ;
    const int nvir = N - nocc;
    const size_t N4 = (size_t)N * N * N * N;

    std::cout << "CCSD spatial-orbital: N=" << N << " nocc=" << nocc
              << " nvir=" << nvir << std::endl;

    // AO->MO transform on GPU (4-stage half-transform, O(N^5))
    real_t* d_eri_mo;
    bool free_eri_mo;
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        d_eri_mo = nullptr;
        tracked_cudaMalloc((void**)&d_eri_mo, N4 * sizeof(real_t));
        {
            std::string str = "Computing AO -> MO 4-stage integral transformation... ";
            PROFILE_ELAPSED_TIME(str);
            transform_ao_eri_to_mo_eri_4stage(d_eri_ao, d_coefficient_matrix, N, d_eri_mo);
            cudaDeviceSynchronize();
        }
        free_eri_mo = true;
    }

    std::vector<real_t> eps(N);
    cudaMemcpy(eps.data(), d_orbital_energies, N * sizeof(real_t), cudaMemcpyDeviceToHost);

    // t2[i,j,a,b] index
    auto T2 = [&](int i, int j, int a, int b) -> size_t {
        return ((size_t)i * nocc + j) * nvir * nvir + (size_t)a * nvir + b;
    };

    // Denominators
    const size_t t1Size = (size_t)nocc * nvir;
    const size_t t2Size = (size_t)nocc * nocc * nvir * nvir;

    std::vector<real_t> Dia(t1Size);
    for (int i = 0; i < nocc; i++)
        for (int a = 0; a < nvir; a++)
            Dia[i*nvir+a] = eps[i] - eps[nocc+a];

    std::vector<real_t> Dijab(t2Size);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++)
                    Dijab[T2(i,j,a,b)] = eps[i] + eps[j] - eps[nocc+a] - eps[nocc+b];

    DIIS diis(8, 2);
    size_t num_amps = t1Size + t2Size;
    const int MAX_ITER = 100;
    const real_t CONV = 1e-10;

    // Pre-build contiguous integral sub-blocks (constant, computed once)
    const size_t oo = (size_t)nocc * nocc;
    const size_t vv = (size_t)nvir * nvir;
    const size_t vvv = vv * nvir;
    const size_t vo = (size_t)nvir * nocc;
    const size_t ov = (size_t)nocc * nvir;

    // ---- GPU sub-block extraction from d_eri_mo (no full N⁴ download) ----
    // Use a single temporary GPU buffer for CPU-bound sub-blocks
    const size_t max_cpu_block = std::max({(size_t)nocc*vvv, vv*ov, ov*ov, vo*vo, vo*oo, oo*ov, oo*oo});
    double* d_extract_tmp = nullptr;
    tracked_cudaMalloc((void**)&d_extract_tmp, max_cpu_block * sizeof(double));

    // Declare vectors outside profiling scope so they survive
    std::vector<real_t> v_oovv(oo * vv);
    std::vector<real_t> w_oovv(oo * vv);
    std::vector<real_t> v_ovvv((size_t)nocc * vvv);
    std::vector<real_t> v_voov((size_t)nvir * nocc * nocc * nvir);
    std::vector<real_t> v_oovo((size_t)nocc * nocc * nvir * nocc);
    const size_t oooo = oo * oo;
    std::vector<real_t> v_oooo(oooo);
    std::vector<real_t> v_vovo(vo * vo);
    std::vector<real_t> v_vvov(vv * ov);
    std::vector<real_t> v_ovov(ov * ov);
    std::vector<real_t> v_vooo(vo * oo);
    std::vector<real_t> v_ooov(oo * ov);
    {
        std::string str = "Extracting MO integral sub-blocks on GPU... ";
        PROFILE_ELAPSED_TIME(str);

    // --- GPU-extract CPU-needed sub-blocks → download ---
    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, 0,nocc, nocc,nvir, nocc,nvir);
    cudaMemcpy(v_oovv.data(), d_extract_tmp, oo*vv*sizeof(double), cudaMemcpyDeviceToHost);

    if (!gpu::gpu_available()) {
        // CPU fallback: w_oovv[k,l,c,d] = 2*(kc|ld) - (kd|lc)
        const size_t N3 = (size_t)N * N * N;
        const size_t N2 = (size_t)N * N;
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < nocc; k++)
            for (int l = 0; l < nocc; l++)
                for (int c = 0; c < nvir; c++)
                    for (int d = 0; d < nvir; d++) {
                        int oc = nocc + c, od = nocc + d;
                        double v_cd = d_eri_mo[(size_t)k * N3 + (size_t)oc * N2 + (size_t)l * N + od];
                        double v_dc = d_eri_mo[(size_t)k * N3 + (size_t)od * N2 + (size_t)l * N + oc];
                        size_t gid = ((size_t)k * nocc + l) * vv + (size_t)c * nvir + d;
                        d_extract_tmp[gid] = 2.0 * v_cd - v_dc;
                    }
    } else {
        int threads = 256;
        int blocks = (int)((oo*vv + threads - 1) / threads);
        extract_w_oovv_kernel<<<blocks, threads>>>(d_eri_mo, d_extract_tmp, N, nocc, nvir);
    }
    cudaMemcpy(w_oovv.data(), d_extract_tmp, oo*vv*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, nocc,nvir, nocc,nvir, nocc,nvir);
    cudaMemcpy(v_ovvv.data(), d_extract_tmp, (size_t)nocc*vvv*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, nocc,nvir, 0,nocc, 0,nocc, nocc,nvir);
    cudaMemcpy(v_voov.data(), d_extract_tmp, (size_t)nvir*nocc*nocc*nvir*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, 0,nocc, nocc,nvir, 0,nocc);
    cudaMemcpy(v_oovo.data(), d_extract_tmp, (size_t)nocc*nocc*nvir*nocc*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, 0,nocc, 0,nocc, 0,nocc);
    cudaMemcpy(v_oooo.data(), d_extract_tmp, oooo*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, nocc,nvir, 0,nocc, nocc,nvir, 0,nocc);
    cudaMemcpy(v_vovo.data(), d_extract_tmp, vo*vo*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, nocc,nvir, nocc,nvir, 0,nocc, nocc,nvir);
    cudaMemcpy(v_vvov.data(), d_extract_tmp, vv*ov*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, nocc,nvir, 0,nocc, nocc,nvir);
    cudaMemcpy(v_ovov.data(), d_extract_tmp, ov*ov*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, nocc,nvir, 0,nocc, 0,nocc, 0,nocc);
    cudaMemcpy(v_vooo.data(), d_extract_tmp, vo*oo*sizeof(double), cudaMemcpyDeviceToHost);

    gpu_extract_subblock(d_eri_mo, d_extract_tmp, N, 0,nocc, 0,nocc, 0,nocc, nocc,nvir);
    cudaMemcpy(v_ooov.data(), d_extract_tmp, oo*ov*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    } // end PROFILE_ELAPSED_TIME for sub-block extraction

    tracked_cudaFree(d_extract_tmp);

    // MP2 initial guess: t2(i,j,a,b) = v_oovv[i,j,a,b] / Dijab
    std::vector<real_t> t1(t1Size, 0.0);
    std::vector<real_t> t2v(t2Size);
    for (int i = 0; i < nocc; i++)
        for (int j = 0; j < nocc; j++)
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b < nvir; b++) {
                    size_t idx = T2(i,j,a,b);
                    t2v[idx] = v_oovv[((size_t)i*nocc+j)*vv + (size_t)a*nvir+b] / Dijab[idx];
                }

    // v_ovvv reshaped for Wabcd DGEMM: v_ovvv_T[nvir³, nocc]
    std::vector<real_t> v_ovvv_T(vvv * nocc);
    for (int a = 0; a < nvir; a++)
        for (int c = 0; c < nvir; c++)
            for (int d = 0; d < nvir; d++)
                for (int k = 0; k < nocc; k++)
                    v_ovvv_T[((size_t)a*vv + (size_t)c*nvir + d)*nocc + k] = v_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)c*nvir + d];

    // Pre-built w-variants: w = 2*v - v_exchange
    // w_voov[a,k,i,c] = 2*v_voov[a,k,i,c] - v_vovo[a,k,c,i]
    std::vector<real_t> w_voov((size_t)nvir * nocc * nocc * nvir);
    for (int a = 0; a < nvir; a++)
        for (int k = 0; k < nocc; k++)
            for (int i = 0; i < nocc; i++)
                for (int c = 0; c < nvir; c++) {
                    size_t idx_voov = ((size_t)a*nocc+k)*(size_t)nocc*nvir + (size_t)i*nvir+c;
                    size_t idx_vovo = ((size_t)a*nocc+k)*vo + (size_t)c*nocc+i;
                    w_voov[idx_voov] = 2.0 * v_voov[idx_voov] - v_vovo[idx_vovo];
                }

    // w_ooov[k,l,i,c] = 2*v_ooov[k,l,i,c] - v_oovo[k,l,c,i]
    std::vector<real_t> w_ooov(oo * ov);
    for (int k = 0; k < nocc; k++)
        for (int l = 0; l < nocc; l++)
            for (int i = 0; i < nocc; i++)
                for (int c = 0; c < nvir; c++) {
                    size_t idx_ooov = ((size_t)k*nocc+l)*ov + (size_t)i*nvir+c;
                    size_t idx_oovo = ((size_t)k*nocc+l)*(size_t)nvir*nocc + (size_t)c*nocc+i;
                    w_ooov[idx_ooov] = 2.0 * v_ooov[idx_ooov] - v_oovo[idx_oovo];
                }

    // w_ovvv[(k*nvir+a)*vv + c*nvir+d] = 2*v_ovvv[k,a,c,d] - v_ovvv[k,a,d,c]
    std::vector<real_t> w_ovvv((size_t)nocc * vvv);
    for (int k = 0; k < nocc; k++)
        for (int a = 0; a < nvir; a++)
            for (int c = 0; c < nvir; c++)
                for (int d = 0; d < nvir; d++)
                    w_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)c*nvir+d] =
                        2.0 * v_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)c*nvir+d]
                            - v_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)d*nvir+c];

    // Pre-allocate all GPU buffers in a single allocation to reduce cudaMalloc overhead
    const size_t OV2 = ov * ov;
    const size_t sz_tau = t2Size, sz_Wabcd = vv*vv, sz_Wklij = oo*oo, sz_raw = t2Size;
    const size_t sz_w_oovv = oo*vv, sz_v_oovv = oo*vv, sz_Fki = oo;
    const size_t sz_v_ovvv_T = vvv*nocc, sz_t1 = t1Size, sz_ovvv_t1 = std::max(vvv*std::max((size_t)nvir,(size_t)nocc), t2Size);
    const size_t sz_v_vvvv = vv*vv, sz_Fac = vv, sz_Fkc = ov;
    const size_t sz_t2v = t2Size, sz_Lac = vv, sz_Z = vv*ov;
    const size_t sz_Wex = OV2;  // each of A, B, C1, C2, V_R, W_R, V_R2
    const size_t sz_v_ovvv = (size_t)nocc * vvv;       // persistent v_ovvv
    const size_t sz_v_ovvv_perm = (size_t)nocc * vvv;   // persistent v_ovvv_perm
    const size_t sz_w_ovvv_R = (size_t)nvir * nocc * vv; // persistent w_ovvv_R
    const size_t total_gpu_doubles = sz_tau + sz_Wabcd + sz_Wklij + sz_raw
        + sz_w_oovv + sz_v_oovv + sz_Fki + sz_v_ovvv_T + sz_t1 + sz_ovvv_t1
        + sz_v_vvvv + sz_Fac + sz_Fkc + sz_t2v + sz_Lac + sz_Z
        + 7 * sz_Wex  // Wex_A, Wex_B, Wex_C1, Wex_C2, V_R, W_R, V_R2
        + sz_v_ovvv + sz_v_ovvv_perm + sz_w_ovvv_R;

    double *d_gpu_pool = nullptr;
    tracked_cudaMalloc((void**)&d_gpu_pool, total_gpu_doubles * sizeof(double));
    double *d_ptr = d_gpu_pool;
    auto carve = [&](size_t n) -> double* { double *p = d_ptr; d_ptr += n; return p; };

    double *d_tau       = carve(sz_tau);
    double *d_Wabcd     = carve(sz_Wabcd);
    double *d_Wklij     = carve(sz_Wklij);
    double *d_raw       = carve(sz_raw);
    double *d_w_oovv    = carve(sz_w_oovv);
    double *d_v_oovv    = carve(sz_v_oovv);
    double *d_Fki       = carve(sz_Fki);
    double *d_v_ovvv_T  = carve(sz_v_ovvv_T);
    double *d_t1        = carve(sz_t1);
    double *d_ovvv_t1   = carve(sz_ovvv_t1);
    double *d_v_vvvv    = carve(sz_v_vvvv);
    double *d_Fac       = carve(sz_Fac);
    double *d_Fkc       = carve(sz_Fkc);
    double *d_t2v       = carve(sz_t2v);
    double *d_Lac       = carve(sz_Lac);
    double *d_Z         = carve(sz_Z);
    double *d_Wex_A     = carve(sz_Wex);
    double *d_Wex_B     = carve(sz_Wex);
    double *d_Wex_C1    = carve(sz_Wex);
    double *d_Wex_C2    = carve(sz_Wex);
    double *d_V_R       = carve(sz_Wex);
    double *d_W_R       = carve(sz_Wex);
    double *d_V_R2      = carve(sz_Wex);
    double *d_v_ovvv    = carve(sz_v_ovvv);       // persistent v_ovvv
    double *d_v_ovvv_perm = carve(sz_v_ovvv_perm); // persistent v_ovvv_perm
    double *d_w_ovvv_R  = carve(sz_w_ovvv_R);     // persistent w_ovvv_R

    // Constant reshaped integrals for Wakic/Wakci ld-sum DGEMM
    // V_R[(l*nvir+d), (k*nvir+c)] = v_oovv[(l*nocc+k)*vv + d*nvir+c]
    // W_R[(l*nvir+d), (k*nvir+c)] = w_oovv[(l*nocc+k)*vv + d*nvir+c]
    // V_R2[(l*nvir+d), (k*nvir+c)] = v_oovv[(l*nocc+k)*vv + c*nvir+d]  (c,d swapped for Wakci)
    {
        std::vector<real_t> V_R(OV2), W_R(OV2), V_R2(OV2);
        for (int l = 0; l < nocc; l++)
            for (int d = 0; d < nvir; d++)
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++) {
                        size_t row = l*nvir+d;
                        size_t col = k*nvir+c;
                        V_R[row*ov + col] = v_oovv[((size_t)l*nocc+k)*vv + (size_t)d*nvir+c];
                        W_R[row*ov + col] = w_oovv[((size_t)l*nocc+k)*vv + (size_t)d*nvir+c];
                        V_R2[row*ov + col] = v_oovv[((size_t)l*nocc+k)*vv + (size_t)c*nvir+d];
                    }
        cudaMemcpy(d_V_R, V_R.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W_R, W_R.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V_R2, V_R2.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
    }

    // GPU-direct extraction for GPU-resident sub-blocks (no CPU round-trip)
    // v_vvvv: v(nocc+a, nocc+b, nocc+c, nocc+d) → d_v_vvvv (largest: nvir⁴)
    gpu_extract_subblock(d_eri_mo, d_v_vvvv, N, nocc,nvir, nocc,nvir, nocc,nvir, nocc,nvir);
    // v_oovv and w_oovv → d_v_oovv, d_w_oovv
    gpu_extract_subblock(d_eri_mo, d_v_oovv, N, 0,nocc, 0,nocc, nocc,nvir, nocc,nvir);
    if (!gpu::gpu_available()) {
        // CPU fallback: w_oovv[k,l,c,d] = 2*(kc|ld) - (kd|lc)
        const size_t N3 = (size_t)N * N * N;
        const size_t N2 = (size_t)N * N;
        #pragma omp parallel for collapse(2)
        for (int k = 0; k < nocc; k++)
            for (int l = 0; l < nocc; l++)
                for (int c = 0; c < nvir; c++)
                    for (int d = 0; d < nvir; d++) {
                        int oc = nocc + c, od = nocc + d;
                        double v_cd = d_eri_mo[(size_t)k * N3 + (size_t)oc * N2 + (size_t)l * N + od];
                        double v_dc = d_eri_mo[(size_t)k * N3 + (size_t)od * N2 + (size_t)l * N + oc];
                        size_t gid = ((size_t)k * nocc + l) * vv + (size_t)c * nvir + d;
                        d_w_oovv[gid] = 2.0 * v_cd - v_dc;
                    }
    } else {
        int threads = 256;
        int blocks_w = (int)((oo*vv + threads - 1) / threads);
        extract_w_oovv_kernel<<<blocks_w, threads>>>(d_eri_mo, d_w_oovv, N, nocc, nvir);
    }
    cudaDeviceSynchronize();
    // Now d_eri_mo is no longer needed — free it
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);
    d_eri_mo = nullptr;

    cudaMemcpy(d_v_ovvv_T, v_ovvv_T.data(), vvv * nocc * sizeof(double), cudaMemcpyHostToDevice);

    // Pre-compute v_ovvv_perm (constant — transpose d,c within each (k,a) block)
    std::vector<real_t> v_ovvv_perm((size_t)nocc * vvv);
    for (int k = 0; k < nocc; k++)
        for (int a = 0; a < nvir; a++)
            for (int c = 0; c < nvir; c++)
                for (int d = 0; d < nvir; d++)
                    v_ovvv_perm[(size_t)k*vvv + (size_t)a*vv + (size_t)c*nvir+d] =
                        v_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)d*nvir+c];

    // Pre-compute w_ovvv_perm reshaped as [nvir, nocc*vv] for T1 DGEMM
    // w_ovvv_R[a, k*vv+c*nvir+d] = w_ovvv[k*vvv + a*vv + d*nvir+c]
    // This transposes (c,d) in w_ovvv and reshapes for DGEMM: Result[a,i] = w_ovvv_R × tau^T
    std::vector<real_t> w_ovvv_R((size_t)nvir * nocc * vv);
    for (int a = 0; a < nvir; a++)
        for (int k = 0; k < nocc; k++)
            for (int c = 0; c < nvir; c++)
                for (int d = 0; d < nvir; d++)
                    w_ovvv_R[(size_t)a * nocc * vv + (size_t)k * vv + (size_t)c * nvir + d] =
                        w_ovvv[(size_t)k * vvv + (size_t)a * vv + (size_t)d * nvir + c];

    // Upload constant integral arrays to persistent GPU buffers (once, not per iteration)
    cudaMemcpy(d_v_ovvv, v_ovvv.data(), sz_v_ovvv * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_ovvv_perm, v_ovvv_perm.data(), sz_v_ovvv_perm * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_ovvv_R, w_ovvv_R.data(), sz_w_ovvv_R * sizeof(double), cudaMemcpyHostToDevice);

    // Energy: E = sum_{ijab} w_oovv[ij,ab] * (t2(i,j,a,b) + t1(i,a)*t1(j,b))
    auto energy = [&]() -> real_t {
        real_t E = 0.0;
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++) {
                        E += w_oovv[((size_t)i*nocc+j)*vv + (size_t)a*nvir+b]
                           * (t2v[T2(i,j,a,b)] + t1[i*nvir+a]*t1[j*nvir+b]);
                    }
        return E;
    };

    real_t Ecc = energy();
    std::cout << "CCSD iter  0: E = " << std::fixed << std::setprecision(12)
              << Ecc << " (MP2 initial guess)" << std::endl;

    for (int iter = 1; iter <= MAX_ITER; iter++) {

        // Upload t1 and t2v to GPU, build tau on GPU (avoids CPU tau computation + upload)
        cudaMemcpy(d_t1, t1.data(), t1Size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t2v, t2v.data(), t2Size * sizeof(double), cudaMemcpyHostToDevice);
        if (!gpu::gpu_available()) {
            // CPU fallback: tau[ij,ab] = t2v[ij,ab] + t1[i,a]*t1[j,b]
            #pragma omp parallel for
            for (size_t gid = 0; gid < t2Size; gid++) {
                int b = gid % nvir; size_t rem = gid / nvir;
                int a = rem % nvir; rem /= nvir;
                int j = rem % nocc;
                int i = (int)(rem / nocc);
                d_tau[gid] = d_t2v[gid] + d_t1[i * nvir + a] * d_t1[j * nvir + b];
            }
        } else {
            int threads = 256;
            int blocks_tau = (int)((t2Size + threads - 1) / threads);
            build_tau_kernel<<<blocks_tau, threads>>>(d_t2v, d_t1, d_tau, nocc, nvir);
        }

        // ---- F intermediates (GPU kernels) ----
        // F^k_c = sum_{ld} w_oovv[kl,cd] * t1(l,d) — GPU kernel, no transfer needed
        std::vector<real_t> Fkc(nocc * nvir);
        if (!gpu::gpu_available()) {
            // CPU fallback: Fkc[k,c] = sum_{l,d} w_oovv[(k*nocc+l)*vv + c*nvir+d] * t1[l*nvir+d]
            #pragma omp parallel for collapse(2)
            for (int k = 0; k < nocc; k++)
                for (int c = 0; c < nvir; c++) {
                    double val = 0.0;
                    for (int l = 0; l < nocc; l++)
                        for (int d = 0; d < nvir; d++)
                            val += d_w_oovv[(k*nocc + l)*(int)vv + c*nvir + d] * d_t1[l*nvir + d];
                    d_Fkc[k*nvir + c] = val;
                }
            cudaMemcpy(Fkc.data(), d_Fkc, ov * sizeof(double), cudaMemcpyDeviceToHost);
        } else {
            dim3 block_fkc(16, 16);
            dim3 grid_fkc((nocc + 15) / 16, (nvir + 15) / 16);
            compute_Fkc_kernel<<<grid_fkc, block_fkc>>>(d_w_oovv, d_t1, d_Fkc, nocc, nvir);
            cudaMemcpy(Fkc.data(), d_Fkc, ov * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // F^k_i = sum_{lcd} w_oovv[k,(l*vv+cd)] * tau[i,(l*vv+cd)]
        // DGEMM: Fki(nocc×nocc) = w_oovv(nocc×nocc*vv) × tau^T(nocc*vv×nocc)
        std::vector<real_t> Fki(nocc * nocc);
        gpu::matrixMatrixProductRect(d_w_oovv, d_tau, d_Fki,
                                nocc, nocc, (int)(nocc * vv),
                                false, true, false, 1.0);
        cudaMemcpy(Fki.data(), d_Fki, oo * sizeof(double), cudaMemcpyDeviceToHost);

        // F^a_c = -sum_{kld} w_oovv[(kl),(cd)] * tau[T2(k,l,a,d)] — GPU kernel
        std::vector<real_t> Fac(nvir * nvir);
        if (!gpu::gpu_available()) {
            // CPU fallback: Fac[a,c] = -sum_{kl,d} w_oovv[kl*vv + c*nvir+d] * tau[kl*vv + a*nvir+d]
            const int oo_int = nocc * nocc;
            const int vv_int = nvir * nvir;
            #pragma omp parallel for collapse(2)
            for (int a = 0; a < nvir; a++)
                for (int c = 0; c < nvir; c++) {
                    double val = 0.0;
                    for (int kl = 0; kl < oo_int; kl++)
                        for (int d = 0; d < nvir; d++)
                            val -= d_w_oovv[kl*vv_int + c*nvir + d] * d_tau[kl*vv_int + a*nvir + d];
                    d_Fac[a*nvir + c] = val;
                }
            cudaMemcpy(Fac.data(), d_Fac, vv * sizeof(double), cudaMemcpyDeviceToHost);
        } else {
            dim3 block_fac(16, 16);
            dim3 grid_fac((nvir + 15) / 16, (nvir + 15) / 16);
            compute_Fac_kernel<<<grid_fac, block_fac>>>(d_w_oovv, d_tau, d_Fac, nocc, nvir);
            cudaMemcpy(Fac.data(), d_Fac, vv * sizeof(double), cudaMemcpyDeviceToHost);
        }

        // ---- L intermediates ----
        // L^k_i = F^k_i + sum_{lc} w(l,k,C,i) * t1(l,c)
        // w(l,k,C,i) = 2*v(l,k,C,i) - v(l,k,i,C) = 2*v_oovo[(l*nocc+k)*nvir*nocc + c*nocc+i] - v_ooov[(l*nocc+k)*ov + i*nvir+c]
        std::vector<real_t> Lki(nocc * nocc, 0.0);
        for (int k = 0; k < nocc; k++)
            for (int i = 0; i < nocc; i++) {
                real_t val = Fki[k*nocc+i];
                for (int l = 0; l < nocc; l++)
                    for (int c = 0; c < nvir; c++) {
                        real_t wval = 2.0 * v_oovo[((size_t)l*nocc+k)*(size_t)nvir*nocc + (size_t)c*nocc+i]
                                         - v_ooov[((size_t)l*nocc+k)*ov + (size_t)i*nvir+c];
                        val += wval * t1[l*nvir+c];
                    }
                Lki[k*nocc+i] = val;
            }

        // L^a_c = F^a_c + sum_{kd} w_ovvv[k,a,d,c] * t1(k,d)
        std::vector<real_t> Lac(nvir * nvir, 0.0);
        for (int a = 0; a < nvir; a++)
            for (int c = 0; c < nvir; c++) {
                real_t val = Fac[a*nvir+c];
                for (int k = 0; k < nocc; k++)
                    for (int d = 0; d < nvir; d++)
                        val += w_ovvv[(size_t)k*vvv + (size_t)a*vv + (size_t)d*nvir+c] * t1[k*nvir+d];
                Lac[a*nvir+c] = val;
            }

        // ---- W^{kl}_{ij} ----
        // W^{kl}_{ij} = v^{kl}_{ij} + sum_c v^{lk}_{ci} * t1(j,c) + sum_c v^{kl}_{cj} * t1(i,c)
        //             + sum_{cd} v_oovv[(kl),(cd)] * tau[(ij),(cd)]
        // DGEMM for vv contraction: Wklij_base[kl,ij] = v_oovv[kl,cd] × tau^T[cd,ij]
        std::vector<real_t> Wklij(oo * oo);
        gpu::matrixMatrixProductRect(d_v_oovv, d_tau, d_Wklij,
                                (int)oo, (int)oo, (int)vv,
                                false, true, false, 1.0);
        cudaMemcpy(Wklij.data(), d_Wklij, oo * oo * sizeof(double), cudaMemcpyDeviceToHost);
        // Add remaining terms on CPU: v_oooo + t1 terms
        for (int k = 0; k < nocc; k++)
            for (int l = 0; l < nocc; l++)
                for (int i = 0; i < nocc; i++)
                    for (int j = 0; j < nocc; j++) {
                        real_t val = v_oooo[((size_t)k*nocc+l)*oo + (size_t)i*nocc+j];
                        for (int c = 0; c < nvir; c++) {
                            val += v_oovo[((size_t)l*nocc+k)*(size_t)nvir*nocc + (size_t)c*nocc+i] * t1[j*nvir+c];
                            val += v_oovo[((size_t)k*nocc+l)*(size_t)nvir*nocc + (size_t)c*nocc+j] * t1[i*nvir+c];
                        }
                        Wklij[((size_t)k*nocc+l)*oo + (size_t)i*nocc+j] += val;
                    }

        // ---- W^{ab}_{cd} ---- (fully on GPU: DGEMM + kernel, no host transfer)
        // GPU DGEMM: d_ovvv_t1[nvir³, nvir] = v_ovvv_T[nvir³, nocc] × t1[nocc, nvir]
        // d_t1 already uploaded at top of iteration
        gpu::matrixMatrixProductRect(d_v_ovvv_T, d_t1, d_ovvv_t1,
                                (int)vvv, nvir, nocc,
                                false, false, false, 1.0);
        // GPU kernel: Wabcd = v_vvvv - ovvv_t1 permutations (no download/upload)
        if (!gpu::gpu_available()) {
            // CPU fallback: Wabcd[a,b,c,d] = v_vvvv[a,b,c,d]
            //   - ovvv_t1[(a*vv + d*nvir+c), b] - ovvv_t1[(b*vv + c*nvir+d), a]
            size_t vv2 = vv * vv;
            #pragma omp parallel for
            for (size_t gid = 0; gid < vv2; gid++) {
                int d = gid % nvir; size_t rem = gid / nvir;
                int c = rem % nvir; rem /= nvir;
                int b = rem % nvir;
                int a = (int)(rem / nvir);
                d_Wabcd[gid] = d_v_vvvv[gid]
                    - d_ovvv_t1[((size_t)a*vv + (size_t)d*nvir + c)*nvir + b]
                    - d_ovvv_t1[((size_t)b*vv + (size_t)c*nvir + d)*nvir + a];
            }
        } else {
            size_t vv2 = vv * vv;
            int threads = 256;
            int blocks = (int)((vv2 + threads - 1) / threads);
            build_Wabcd_kernel<<<blocks, threads>>>(d_v_vvvv, d_ovvv_t1, d_Wabcd, nvir);
        }

        // ---- W^{ak}_{ic} and W^{ak}_{ci} (two exchange intermediates) ----
        // d-sum via DGEMM: ovvv_t1_ic[(k*vv + a*nvir+c), i] = sum_d v_ovvv[k,a,c,d] * t1[i,d]
        // v_ovvv viewed as [nocc*vv, nvir], t1 as [nocc, nvir]
        // DGEMM: [nocc*vv, nocc] = v_ovvv[nocc*vv, nvir] × t1^T[nvir, nocc]
        // Use persistent d_v_ovvv buffer (uploaded once before the loop)
        // Result stored in d_ovvv_t1 as scratch (size sz_ovvv_t1 ≥ nocc*vv*nocc = oo*vv) ✓
        // This avoids overwriting d_t2v which holds t2v data
        gpu::matrixMatrixProductRect(d_v_ovvv, d_t1, d_ovvv_t1,
                                (int)(nocc * vv), nocc, nvir,
                                false, true, false, 1.0);
        // Download d-sum result: ovvv_t1_ic[k*vv*nocc + ac*nocc + i]
        std::vector<real_t> ovvv_t1_ic((size_t)nocc * vv * nocc);
        cudaMemcpy(ovvv_t1_ic.data(), d_ovvv_t1, (size_t)nocc * vv * nocc * sizeof(double), cudaMemcpyDeviceToHost);

        // ovvv_t1_dc for Wakci: sum_d v_ovvv_perm[k,a,c,d] * t1[i,d]
        // Use persistent d_v_ovvv_perm buffer (uploaded once before the loop)
        gpu::matrixMatrixProductRect(d_v_ovvv_perm, d_t1, d_ovvv_t1,
                                (int)(nocc * vv), nocc, nvir,
                                false, true, false, 1.0);
        std::vector<real_t> ovvv_t1_dc((size_t)nocc * vv * nocc);
        cudaMemcpy(ovvv_t1_dc.data(), d_ovvv_t1, (size_t)nocc * vv * nocc * sizeof(double), cudaMemcpyDeviceToHost);

        // ld-sum terms computed via DGEMM (below)
        std::vector<real_t> Wakic((size_t)nvir*nocc*nocc*nvir, 0.0);
        std::vector<real_t> Wakci((size_t)nvir*nocc*nvir*nocc, 0.0);

        // CPU: single-index terms for Wakic and Wakci (d-sum replaced by DGEMM result lookup)
        for (int a = 0; a < nvir; a++)
            for (int k = 0; k < nocc; k++) {
                for (int i = 0; i < nocc; i++)
                    for (int c = 0; c < nvir; c++) {
                        real_t val = v_voov[((size_t)a*nocc+k)*(size_t)nocc*nvir + (size_t)i*nvir+c];
                        for (int l = 0; l < nocc; l++)
                            val -= v_oovo[((size_t)k*nocc+l)*(size_t)nvir*nocc + (size_t)c*nocc+i] * t1[l*nvir+a];
                        // d-sum from DGEMM: ovvv_t1_ic[(k*vv + a*nvir+c)*nocc + i]
                        val += ovvv_t1_ic[((size_t)k*vv + (size_t)a*nvir+c)*nocc + i];
                        Wakic[((size_t)a*nocc+k)*nocc*nvir + (size_t)i*nvir+c] = val;
                    }
                for (int c = 0; c < nvir; c++)
                    for (int i = 0; i < nocc; i++) {
                        real_t val = v_vovo[((size_t)a*nocc+k)*vo + (size_t)c*nocc+i];
                        for (int l = 0; l < nocc; l++)
                            val -= v_oovo[((size_t)l*nocc+k)*(size_t)nvir*nocc + (size_t)c*nocc+i] * t1[l*nvir+a];
                        // d-sum from DGEMM: ovvv_t1_dc[(k*vv + a*nvir+c)*nocc + i]
                        val += ovvv_t1_dc[((size_t)k*vv + (size_t)a*nvir+c)*nocc + i];
                        Wakci[((size_t)a*nocc+k)*nvir*nocc + (size_t)c*nocc+i] = val;
                    }
            }

        // DGEMM for Wakic/Wakci ld-sum terms
        // eff_t2[T2(i,l,d,a)] = -0.5*t2[i,l,d,a] - t1[i,d]*t1[l,a]
        // Reshape: eff_t2_R[(l*nvir+d), (i*nvir+a)] = eff_t2[T2(i,l,d,a)]
        //          t2_C[(l*nvir+d), (i*nvir+a)] = t2[T2(i,l,a,d)]
        {
            std::vector<real_t> eff_t2_R(OV2);
            std::vector<real_t> t2_C(OV2);
            for (int l = 0; l < nocc; l++)
                for (int d = 0; d < nvir; d++)
                    for (int i = 0; i < nocc; i++)
                        for (int a = 0; a < nvir; a++) {
                            size_t row = l*nvir+d;
                            size_t col = i*nvir+a;
                            eff_t2_R[row*ov + col] = -0.5*t2v[T2(i,l,d,a)] - t1[i*nvir+d]*t1[l*nvir+a];
                            t2_C[row*ov + col] = t2v[T2(i,l,a,d)];
                        }

            // Wakic ld-sum: R_AB[ia,kc] = eff_t2_R^T × V_R (terms A+B)
            // Upload eff_t2_R to d_Wex_B (persists for reuse in Wakci DGEMM below)
            cudaMemcpy(d_Wex_B, eff_t2_R.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            gpu::matrixMatrixProductRect(d_Wex_B, d_V_R, d_Wex_C1,
                                    (int)ov, (int)ov, (int)ov, true, false, false, 1.0);
            // Wakic ld-sum: R_C[ia,kc] += 0.5 * t2_C^T × W_R (term C)
            cudaMemcpy(d_Wex_A, t2_C.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            gpu::matrixMatrixProductRect(d_Wex_A, d_W_R, d_Wex_C1,
                                    (int)ov, (int)ov, (int)ov, true, false, true, 0.5);
            // Wakci ld-sum: R_Wakci[ia,kc] = eff_t2_R^T × V_R2 (reuse d_Wex_B)
            gpu::matrixMatrixProductRect(d_Wex_B, d_V_R2, d_Wex_C2,
                                    (int)ov, (int)ov, (int)ov, true, false, false, 1.0);

            // Download and scatter into Wakic/Wakci
            std::vector<real_t> R_Wakic(OV2), R_Wakci(OV2);
            cudaMemcpy(R_Wakic.data(), d_Wex_C1, OV2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(R_Wakci.data(), d_Wex_C2, OV2 * sizeof(double), cudaMemcpyDeviceToHost);
            for (int a = 0; a < nvir; a++)
                for (int k = 0; k < nocc; k++)
                    for (int i = 0; i < nocc; i++)
                        for (int c = 0; c < nvir; c++) {
                            Wakic[((size_t)a*nocc+k)*nocc*nvir + (size_t)i*nvir+c] +=
                                R_Wakic[(size_t)(i*nvir+a)*ov + k*nvir+c];
                            Wakci[((size_t)a*nocc+k)*nvir*nocc + (size_t)c*nocc+i] +=
                                R_Wakci[(size_t)(i*nvir+a)*ov + k*nvir+c];
                        }
        }

        // ---- T1 update ----
        // t1(i,a)*D = Ltilde^a_c*t1(i,c) - Ltilde^k_i*t1(k,a)
        //           + 2*F^k_c*t2(k,i,c,a) - F^k_c*t2(i,k,c,a) + F^k_c*t1(i,c)*t1(k,a)
        //           + w^{ak}_{ic}*t1(k,c)
        //           + w^{ak}_{cd}*t2(i,k,c,d) + w^{ak}_{cd}*t1(i,c)*t1(k,d)   -- wait: should be sum
        //           - w^{kl}_{ic}*t2(k,l,a,c) - w^{kl}_{ic}*t1(k,a)*t1(l,c)
        // where Ltilde = L - diagonal part (but for canonical HF, f_diag is removed, so Ltilde = L - 0? No...)
        // Actually Ltilde^a_c = L^a_c for c!=a, Ltilde^a_a = L^a_a (no diagonal subtraction needed since
        // we compute F without the diagonal Fock contribution, but actually the formula says
        // "Ftilde^a_c = F^a_c - delta_{ac} * D_i^a" ... hmm.
        // For canonical HF with the way we defined F (without diagonal Fock), Ltilde = L.
        // The D_i^a denominator is applied at the end, so Ltilde = L.
        // T1 w_ovvv DGEMM: Result[a,i] = w_ovvv_R[nvir, nocc*vv] × tau[nocc, nocc*vv]^T
        // w_ovvv_R[a, k*vv+c*nvir+d] = w_ovvv[k*vvv+a*vv+d*nvir+c] (pre-computed)
        // Use persistent d_w_ovvv_R buffer (uploaded once before the loop)
        // d_Fkc reuse as result buffer (size ov = nvir*nocc, sufficient)
        gpu::matrixMatrixProductRect(d_w_ovvv_R, d_tau, d_Fkc,
                                nvir, nocc, (int)(nocc * vv),
                                false, true, false, 1.0);
        std::vector<real_t> t1_wovvv(ov);
        cudaMemcpy(t1_wovvv.data(), d_Fkc, ov * sizeof(double), cudaMemcpyDeviceToHost);

        std::vector<real_t> newT1(t1Size, 0.0);
        for (int i = 0; i < nocc; i++)
            for (int a = 0; a < nvir; a++) {
                real_t val = 0.0;
                // Fac * t1
                for (int c = 0; c < nvir; c++)
                    val += Fac[a*nvir+c] * t1[i*nvir+c];
                // -Fki * t1
                for (int k = 0; k < nocc; k++)
                    val -= Fki[k*nocc+i] * t1[k*nvir+a];
                // 2*F^k_c*t2(ki,ca) - F^k_c*t2(ik,ca) + F^k_c*t1(i,c)*t1(k,a)
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++) {
                        real_t fc = Fkc[k*nvir+c];
                        val += fc * (2.0*t2v[T2(k,i,c,a)] - t2v[T2(i,k,c,a)] + t1[i*nvir+c]*t1[k*nvir+a]);
                    }
                // w_voov[a,k,i,c] * t1(k,c)
                for (int k = 0; k < nocc; k++)
                    for (int c = 0; c < nvir; c++)
                        val += w_voov[((size_t)a*nocc+k)*(size_t)nocc*nvir + (size_t)i*nvir+c] * t1[k*nvir+c];
                // w_ovvv term via DGEMM (computed above)
                val += t1_wovvv[a*nocc+i];
                // -w_ooov[k,l,i,c] * (t2(kl,ac) + t1(k,a)*t1(l,c))
                for (int k = 0; k < nocc; k++)
                    for (int l = 0; l < nocc; l++)
                        for (int c = 0; c < nvir; c++)
                            val -= w_ooov[((size_t)k*nocc+l)*ov + (size_t)i*nvir+c] * (t2v[T2(k,l,a,c)] + t1[k*nvir+a]*t1[l*nvir+c]);
                newT1[i*nvir+a] = val / Dia[i*nvir+a];
            }

        // ---- T2 update ----
        // raw(i,a,j,b) computed, then t2_new(i,j,a,b) = [raw(i,a,j,b) + raw(j,b,i,a)] / D
        // This is the P(ia,jb) symmetrization.
        //
        // raw(i,a,j,b) = 0.5*v^{ij}_{ab}
        //   + 0.5*sum_{kl} W^{kl}_{ij} * (t2(k,l,a,b) + t1(k,a)*t1(l,b))
        //   + 0.5*sum_{cd} W^{ab}_{cd} * (t2(i,j,c,d) + t1(i,c)*t1(j,d))
        //   + Ltilde^a_c * t2(i,j,c,b)
        //   - Ltilde^k_i * t2(k,j,a,b)
        //   + v^{ab}_{ic} * t1(j,c)
        //   - sum_k v^{kb}_{ic} * t1(k,a) * t1(j,c)  -- wait, this should be -v^{kb}_{ic}*t1(k,a)*t1(j,c)
        //   - v^{ak}_{ij} * t1(k,b)
        //   - v^{ak}_{ic} * t1(j,c) * t1(k,b)   -- from the formula
        //   + sum_k 2*W^{ak}_{ic}*t2(k,j,c,b) - W^{ak}_{ci}*t2(k,j,c,b) - W^{ak}_{ic}*t2(k,j,b,c)
        //   - sum_k W^{bk}_{ci}*t2(k,j,a,c)

        // ---- GPU DGEMM for Wabcd×tau and Wklij×tau ----
        // d_Wabcd already on GPU from build_Wabcd_kernel above
        cudaMemcpy(d_Wklij, Wklij.data(), oo * oo * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_raw, 0, t2Size * sizeof(double));

        // raw[ij,ab] += 0.5 * tau[ij,cd] * Wabcd[ab,cd]^T
        gpu::matrixMatrixProductRect(d_tau, d_Wabcd, d_raw,
                                (int)oo, (int)vv, (int)vv,
                                false, true, true, 0.5);

        // raw[ij,ab] += 0.5 * Wklij^T[ij,kl] * tau[kl,ab]
        gpu::matrixMatrixProductRect(d_Wklij, d_tau, d_raw,
                                (int)oo, (int)vv, (int)oo,
                                true, false, true, 0.5);

        // ---- GPU batched DGEMM for Lac×t2 → d_raw (before download) ----
        // For each ij: raw[ij,ab] += Lac[a,c] × t2v[ij,cb]
        // d_t2v already contains t2v from upload at start of iteration
        cudaMemcpy(d_Lac, Lac.data(), vv * sizeof(double), cudaMemcpyHostToDevice);
        gpu::matrixMatrixProductBatched(d_Lac, d_t2v, d_raw,
                                    nvir, nvir, nvir,
                                    0, (long long)vv, (long long)vv,
                                    (int)oo,
                                    false, false, true, 1.0);

        // Download raw (now includes Wabcd×tau + Wklij×tau + Lac×t2)
        std::vector<real_t> raw(t2Size);
        cudaMemcpy(raw.data(), d_raw, t2Size * sizeof(double), cudaMemcpyDeviceToHost);

        // ---- W exchange terms via DGEMM ----
        // Term 1+2: sum_{kc} (2*Wakic[a,k,i,c] - Wakci[a,k,c,i]) * t2[k,j,c,b]
        //   DGEMM: Weff[ai,kc] × t2_R1[kc,jb]  →  R12[ai,jb]
        // Term 3: -sum_{kc} Wakic[a,k,i,c] * t2[k,j,b,c]
        //   DGEMM: -Wakic_R[ai,kc] × t2_R3[kc,jb]  →  R12[ai,jb] (accumulate)
        // Term 4: -sum_{kc} Wakci[b,k,c,i] * t2[k,j,a,c]
        //   DGEMM: -Wakci_R[bi,kc] × t2_R4[kc,ja]  →  R4[bi,ja]
        {
            std::vector<real_t> Weff(OV2), Wakic_R(OV2), Wakci_R_mat(OV2);
            std::vector<real_t> t2_R1(OV2), t2_R3(OV2), t2_R4(OV2);

            // Build reshaped W intermediates
            for (int a = 0; a < nvir; a++)
                for (int i = 0; i < nocc; i++)
                    for (int k = 0; k < nocc; k++)
                        for (int c = 0; c < nvir; c++) {
                            size_t row = a*nocc+i;
                            size_t col = k*nvir+c;
                            real_t wic = Wakic[((size_t)a*nocc+k)*nocc*nvir + (size_t)i*nvir+c];
                            real_t wci = Wakci[((size_t)a*nocc+k)*nvir*nocc + (size_t)c*nocc+i];
                            Weff[row*ov + col] = 2.0*wic - wci;
                            Wakic_R[row*ov + col] = wic;
                        }
            for (int b = 0; b < nvir; b++)
                for (int i = 0; i < nocc; i++)
                    for (int k = 0; k < nocc; k++)
                        for (int c = 0; c < nvir; c++)
                            Wakci_R_mat[(b*nocc+i)*ov + k*nvir+c] =
                                Wakci[((size_t)b*nocc+k)*nvir*nocc + (size_t)c*nocc+i];

            // Build reshaped t2
            for (int k = 0; k < nocc; k++)
                for (int c = 0; c < nvir; c++)
                    for (int j = 0; j < nocc; j++)
                        for (int b = 0; b < nvir; b++) {
                            size_t row = k*nvir+c;
                            size_t col = j*nvir+b;
                            t2_R1[row*ov + col] = t2v[T2(k,j,c,b)];
                            t2_R3[row*ov + col] = t2v[T2(k,j,b,c)];
                        }
            for (int k = 0; k < nocc; k++)
                for (int c = 0; c < nvir; c++)
                    for (int j = 0; j < nocc; j++)
                        for (int a = 0; a < nvir; a++)
                            t2_R4[(k*nvir+c)*ov + j*nvir+a] = t2v[T2(k,j,a,c)];

            // DGEMM 1: R12 = Weff × t2_R1  (terms 1+2)
            cudaMemcpy(d_Wex_A, Weff.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wex_B, t2_R1.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            gpu::matrixMatrixProductRect(d_Wex_A, d_Wex_B, d_Wex_C1,
                                    (int)ov, (int)ov, (int)ov, false, false, false, 1.0);
            // DGEMM 2: R12 -= Wakic_R × t2_R3  (term 3, accumulate)
            cudaMemcpy(d_Wex_A, Wakic_R.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wex_B, t2_R3.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            gpu::matrixMatrixProductRect(d_Wex_A, d_Wex_B, d_Wex_C1,
                                    (int)ov, (int)ov, (int)ov, false, false, true, -1.0);
            // DGEMM 3: R4 = -Wakci_R × t2_R4  (term 4)
            cudaMemcpy(d_Wex_A, Wakci_R_mat.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(d_Wex_B, t2_R4.data(), OV2 * sizeof(double), cudaMemcpyHostToDevice);
            gpu::matrixMatrixProductRect(d_Wex_A, d_Wex_B, d_Wex_C2,
                                    (int)ov, (int)ov, (int)ov, false, false, false, -1.0);

            // Download results
            std::vector<real_t> R12(OV2), R4(OV2);
            cudaMemcpy(R12.data(), d_Wex_C1, OV2 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(R4.data(), d_Wex_C2, OV2 * sizeof(double), cudaMemcpyDeviceToHost);

            // Scatter W exchange DGEMM results into raw
            for (int i = 0; i < nocc; i++)
                for (int j = 0; j < nocc; j++)
                    for (int a = 0; a < nvir; a++)
                        for (int b = 0; b < nvir; b++)
                            raw[T2(i,j,a,b)] += R12[(a*nocc+i)*ov + j*nvir+b]
                                               + R4[(b*nocc+i)*ov + j*nvir+a];
        }

        // ---- GPU DGEMM for Z×t1 → raw contribution ----
        // Z[(ab)*ov + i*nvir+c] stored as [vv*nocc, nvir] (c is contiguous)
        // result[(ab*nocc+i), j] = sum_c Z[(ab*nocc+i), c] × t1^T[c, j]
        // = DGEMM: [vv*nocc, nocc] = Z[vv*nocc, nvir] × t1^T[nvir, nocc]
        // Then: raw[T2(i,j,a,b)] += result[(a*nvir+b)*nocc + i, j]
        std::vector<real_t> Z(vv * ov);
        std::copy(v_vvov.begin(), v_vvov.end(), Z.begin());
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++)
                for (int i = 0; i < nocc; i++)
                    for (int c = 0; c < nvir; c++) {
                        size_t z_idx = ((size_t)a*nvir+b)*ov + (size_t)i*nvir+c;
                        for (int k = 0; k < nocc; k++) {
                            Z[z_idx] -= v_ovov[((size_t)k*nvir+b)*ov + (size_t)i*nvir+c] * t1[k*nvir+a];
                            Z[z_idx] -= v_voov[((size_t)a*nocc+k)*(size_t)nocc*nvir + (size_t)i*nvir+c] * t1[k*nvir+b];
                        }
                    }

        // Z×t1 DGEMM: result[vv*nocc, nocc] = Z[vv*nocc, nvir] × t1^T[nvir, nocc]
        cudaMemcpy(d_Z, Z.data(), vv * ov * sizeof(double), cudaMemcpyHostToDevice);
        // d_t1 already on GPU; result goes into d_ovvv_t1 (scratch, avoids overwriting d_t2v)
        gpu::matrixMatrixProductRect(d_Z, d_t1, d_ovvv_t1,
                                (int)(vv * nocc), nocc, nvir,
                                false, true, false, 1.0);
        std::vector<real_t> Zt1_result(vv * oo);
        cudaMemcpy(Zt1_result.data(), d_ovvv_t1, vv * oo * sizeof(double), cudaMemcpyDeviceToHost);

        // Build Q (small: O(nocc² × vv))
        std::vector<real_t> Q(vv * oo);
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b < nvir; b++)
                for (int i = 0; i < nocc; i++)
                    for (int j = 0; j < nocc; j++) {
                        real_t val = 0.0;
                        for (int k = 0; k < nocc; k++)
                            val += v_vooo[((size_t)a*nocc+k)*oo + (size_t)i*nocc+j] * t1[k*nvir+b];
                        Q[((size_t)a*nvir+b)*oo + (size_t)i*nocc+j] = val;
                    }

        // Reduced inner loop: only O(nocc) per (i,j,a,b) — Lac×t2 and Z×t1 done via DGEMM
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++) {
                        size_t idx = T2(i,j,a,b);
                        real_t val = 0.5 * v_oovv[((size_t)i*nocc+j)*vv + (size_t)a*nvir+b];

                        // -Lki * t2 (O(nocc) inner loop — small)
                        for (int k = 0; k < nocc; k++)
                            val -= Lki[k*nocc+i] * t2v[T2(k,j,a,b)];

                        // Z×t1 DGEMM result scatter
                        val += Zt1_result[((size_t)a*nvir+b)*oo + (size_t)i*nocc+j];

                        // -Q[ab,ij]
                        val -= Q[((size_t)a*nvir+b)*oo + (size_t)i*nocc+j];

                        raw[idx] += val;
                    }

        // Symmetrize: t2_new(i,j,a,b) = [raw(i,a,j,b) + raw(j,b,i,a)] / D
        // raw is stored as raw[T2(i,j,a,b)] = raw(i,a,j,b)
        // so raw(j,b,i,a) = raw[T2(j,i,b,a)]
        std::vector<real_t> newT2(t2Size);
        for (int i = 0; i < nocc; i++)
            for (int j = 0; j < nocc; j++)
                for (int a = 0; a < nvir; a++)
                    for (int b = 0; b < nvir; b++) {
                        size_t idx = T2(i,j,a,b);
                        newT2[idx] = (raw[idx] + raw[T2(j,i,b,a)]) / Dijab[idx];
                    }

        // ---- DIIS ----
        std::vector<real_t> ampVec(num_amps);
        std::vector<real_t> errVec(num_amps);
        for (size_t k = 0; k < t1Size; k++) {
            ampVec[k] = newT1[k];
            errVec[k] = newT1[k] - t1[k];
        }
        for (size_t k = 0; k < t2Size; k++) {
            ampVec[t1Size + k] = newT2[k];
            errVec[t1Size + k] = newT2[k] - t2v[k];
        }
        diis.push(ampVec, errVec);
        if (diis.can_extrapolate()) {
            auto extrap = diis.extrapolate();
            for (size_t k = 0; k < t1Size; k++) newT1[k] = extrap[k];
            for (size_t k = 0; k < t2Size; k++) newT2[k] = extrap[t1Size + k];
        }

        for (size_t k = 0; k < t1Size; k++) t1[k] = newT1[k];
        for (size_t k = 0; k < t2Size; k++) t2v[k] = newT2[k];

        real_t newEcc = energy();
        real_t deltaE = newEcc - Ecc;
        Ecc = newEcc;

        std::cout << "CCSD iter " << std::setw(2) << iter
                  << ": E = " << std::fixed << std::setprecision(12) << Ecc
                  << ", dE = " << std::scientific << std::setprecision(4) << deltaE
                  << std::endl;

        if (std::abs(deltaE) < CONV) {
            std::cout << "CCSD converged after " << iter << " iterations" << std::endl;
            break;
        }
    }

    // Optionally copy converged T1/T2 to new device allocations for EOM-CCSD
    if (d_t1_out) {
        tracked_cudaMalloc((void**)d_t1_out, t1Size * sizeof(real_t));
        cudaMemcpy(*d_t1_out, t1.data(), t1Size * sizeof(real_t), cudaMemcpyHostToDevice);
    }
    if (d_t2_out) {
        tracked_cudaMalloc((void**)d_t2_out, t2Size * sizeof(real_t));
        cudaMemcpy(*d_t2_out, t2v.data(), t2Size * sizeof(real_t), cudaMemcpyHostToDevice);
    }

    // Free pre-allocated GPU buffers
    tracked_cudaFree(d_gpu_pool);  // single deallocation for all GPU buffers

    // ---- (T) perturbative triples correction (spatial orbital, DGEMM-accelerated) ----
    // Pre-compute ALL f-sum and m-sum contractions via 2 large DGEMMs:
    //   F_sum[(i*vv+ab), (kj*nvir+c)] = sum_f V_full[i*vv+ab, f] * t2v[kj*nvir+c, f]
    //   M_sum[(aa*oo+ji), (k*vv+bc)]  = -v_vooo[aa*oo+ji, m] * t2v_H[m, k*vv+bc]
    // Then the inner loop over (a,b,c) does only lookups (no contractions).
    if (computing_ccsd_t && ccsd_t_energy) {
        std::cout << "---- Computing (T) correction (spatial orbital, DGEMM) ----" << std::endl;
        std::string str = "Computing (T) correction energy... ";
        PROFILE_ELAPSED_TIME(str);

        const size_t o3 = (size_t)nocc * nocc * nocc;
        real_t E_T = 0.0;

        // ---- DGEMM 1: F_sum for all f-sums ----
        // V_full[(i*vv+ab), f] = v_ovvv[i*vvv + f*vv + ab]  (transpose f,ab within each i)
        // T2_mat[(kj*nvir+c), f] = t2v[kj*vv + c*nvir + f]  (= t2v as-is, since vv=nvir²)
        // F_sum = V_full × T2_mat^T  →  [nocc*vv × oo*nvir]
        const size_t F_rows = (size_t)nocc * vv;    // nocc * nvir²
        const size_t F_cols = oo * nvir;             // nocc² * nvir
        std::vector<real_t> V_full(F_rows * nvir);
        for (int i = 0; i < nocc; i++)
            for (int ab = 0; ab < (int)vv; ab++)
                for (int f = 0; f < nvir; f++)
                    V_full[((size_t)i*vv + ab)*nvir + f] = v_ovvv[(size_t)i*vvv + (size_t)f*vv + ab];

        // t2v is already in the right layout for T2_mat: t2v[row*nvir + f] where row = kj*nvir+c
        // DGEMM: F_sum[F_rows × F_cols] = V_full[F_rows × nvir] × t2v^T[nvir × F_cols]
        const size_t F_total = F_rows * F_cols;
        double *d_V_full=nullptr, *d_T2_mat=nullptr, *d_F_sum=nullptr;
        tracked_cudaMalloc((void**)&d_V_full, F_rows * nvir * sizeof(double));
        tracked_cudaMalloc((void**)&d_T2_mat, t2Size * sizeof(double));
        tracked_cudaMalloc((void**)&d_F_sum, F_total * sizeof(double));

        cudaMemcpy(d_V_full, V_full.data(), F_rows * nvir * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_T2_mat, t2v.data(), t2Size * sizeof(double), cudaMemcpyHostToDevice);
        gpu::matrixMatrixProductRect(d_V_full, d_T2_mat, d_F_sum,
                                (int)F_rows, (int)F_cols, nvir,
                                false, true, false, 1.0);

        tracked_cudaFree(d_V_full);
        // d_F_sum stays on GPU for the kernel

        // ---- DGEMM 2: M_sum for all m-sums ----
        // G[(aa*oo+ji), m] = v_vooo[(aa*nocc+j)*oo + i*nocc + m]  (= v_vooo as-is)
        // H[m, k*vv+bc] = t2v[(m*nocc+k)*vv + bc]  (= t2v viewed as [nocc × nocc*vv])
        // M_sum = -G × H  →  [(nvir*oo) × (nocc*vv)]
        const size_t M_rows = (size_t)nvir * oo;  // = vo * nocc
        const size_t M_cols = (size_t)nocc * vv;   // = t2Size / nocc actually = nocc * vv
        const size_t M_total = M_rows * M_cols;

        double *d_G=nullptr, *d_M_sum=nullptr;
        tracked_cudaMalloc((void**)&d_G, M_rows * nocc * sizeof(double));
        tracked_cudaMalloc((void**)&d_M_sum, M_total * sizeof(double));

        cudaMemcpy(d_G, v_vooo.data(), M_rows * nocc * sizeof(double), cudaMemcpyHostToDevice);
        // d_T2_mat is still valid (t2v as [nocc × nocc*vv])
        gpu::matrixMatrixProductRect(d_G, d_T2_mat, d_M_sum,
                                (int)M_rows, (int)M_cols, nocc,
                                false, false, false, -1.0);

        tracked_cudaFree(d_G);
        tracked_cudaFree(d_T2_mat);
        // d_M_sum stays on GPU for the kernel

        // ---- GPU kernel for inner loop ----
        // Enumerate all (a,b,c) triples with a >= b >= c
        int num_triples = 0;
        for (int a = 0; a < nvir; a++)
            for (int b = 0; b <= a; b++)
                for (int c = 0; c <= b; c++)
                    num_triples++;

        std::vector<int> abc_triples(num_triples * 3);
        {
            int idx = 0;
            for (int a = 0; a < nvir; a++)
                for (int b = 0; b <= a; b++)
                    for (int c = 0; c <= b; c++) {
                        abc_triples[idx*3]   = a;
                        abc_triples[idx*3+1] = b;
                        abc_triples[idx*3+2] = c;
                        idx++;
                    }
        }

        // Upload auxiliary arrays to GPU
        double *d_v_oovv_t=nullptr, *d_t1_t=nullptr, *d_eps_t=nullptr;
        int *d_abc=nullptr;
        double *d_block_ET=nullptr;
        tracked_cudaMalloc((void**)&d_v_oovv_t, oo * vv * sizeof(double));
        tracked_cudaMalloc((void**)&d_t1_t, ov * sizeof(double));
        tracked_cudaMalloc((void**)&d_eps_t, N * sizeof(double));
        tracked_cudaMalloc((void**)&d_abc, num_triples * 3 * sizeof(int));
        tracked_cudaMalloc((void**)&d_block_ET, num_triples * sizeof(double));

        cudaMemcpy(d_v_oovv_t, v_oovv.data(), oo * vv * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t1_t, t1.data(), ov * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_eps_t, eps.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_abc, abc_triples.data(), num_triples * 3 * sizeof(int), cudaMemcpyHostToDevice);

        // Allocate global memory for wt/zt (6 * o3 per triple)
        const size_t wt_zt_size = (size_t)num_triples * 6 * o3;
        double *d_g_wt = nullptr, *d_g_zt = nullptr;
        tracked_cudaMalloc((void**)&d_g_wt, wt_zt_size * sizeof(double));
        tracked_cudaMalloc((void**)&d_g_zt, wt_zt_size * sizeof(double));

        if (!gpu::gpu_available()) {
            // CPU fallback for ccsd_t_energy_kernel
            // Process each (a,b,c) triple independently (parallelized over triples)
            #pragma omp parallel for reduction(+:E_T) schedule(dynamic)
            for (int triple_id = 0; triple_id < num_triples; triple_id++) {
                const int a_t = d_abc[triple_id * 3];
                const int b_t = d_abc[triple_id * 3 + 1];
                const int c_t = d_abc[triple_id * 3 + 2];

                double d3_scale = 1.0;
                if (a_t == c_t) d3_scale = 6.0;
                else if (a_t == b_t || b_t == c_t) d3_scale = 2.0;

                int perms[6][3] = {{a_t,b_t,c_t},{a_t,c_t,b_t},{b_t,a_t,c_t},
                                   {b_t,c_t,a_t},{c_t,a_t,b_t},{c_t,b_t,a_t}};

                const size_t block_offset = (size_t)triple_id * 6 * o3;
                double* wt = d_g_wt + block_offset;
                double* zt = d_g_zt + block_offset;
                std::vector<double> r3buf(o3);

                // Phase 1 & 2: for each permutation, compute wt[p] and zt[p]
                for (int p = 0; p < 6; p++) {
                    int aa = perms[p][0], bb = perms[p][1], cc = perms[p][2];

                    // Phase 1: compute wt and wpv (stored in zt temporarily)
                    for (int ijk = 0; ijk < o3; ijk++) {
                        int k = ijk % nocc;
                        int j = (ijk / nocc) % nocc;
                        int i_t = ijk / oo;

                        size_t f_row = (size_t)i_t * vv + (size_t)aa * nvir + bb;
                        size_t f_col = ((size_t)k * nocc + j) * nvir + cc;
                        double wval = d_F_sum[f_row * F_cols + f_col];

                        size_t m_row = (size_t)aa * oo + (size_t)j * nocc + i_t;
                        size_t m_col = (size_t)k * vv + (size_t)bb * nvir + cc;
                        wval += d_M_sum[m_row * M_cols + m_col];

                        double vval = d_v_oovv_t[((size_t)i_t * nocc + j) * vv + (size_t)aa * nvir + bb]
                                    * d_t1_t[k * nvir + cc];

                        wt[p * o3 + ijk] = wval;
                        zt[p * o3 + ijk] = wval + 0.5 * vval;
                    }

                    // Phase 2a: compute r3out from wpv (stored in zt) into r3buf
                    for (int ijk = 0; ijk < o3; ijk++) {
                        int k = ijk % nocc;
                        int j = (ijk / nocc) % nocc;
                        int i_t = ijk / oo;
                        int idx1 = (i_t*nocc+k)*nocc+j;
                        int idx2 = (j*nocc+i_t)*nocc+k;
                        int idx3 = (j*nocc+k)*nocc+i_t;
                        int idx4 = (k*nocc+i_t)*nocc+j;
                        int idx5 = (k*nocc+j)*nocc+i_t;

                        r3buf[ijk] = 4.0*zt[p*o3+ijk] + zt[p*o3+idx3] + zt[p*o3+idx4]
                                   - 2.0*zt[p*o3+idx5] - 2.0*zt[p*o3+idx1] - 2.0*zt[p*o3+idx2];
                    }

                    // Phase 2b: write zt[p] = r3out / D
                    for (int ijk = 0; ijk < o3; ijk++) {
                        int k = ijk % nocc;
                        int j = (ijk / nocc) % nocc;
                        int i_t = ijk / oo;
                        double D = (d_eps_t[i_t] + d_eps_t[j] + d_eps_t[k]
                                  - d_eps_t[nocc+perms[p][0]] - d_eps_t[nocc+perms[p][1]] - d_eps_t[nocc+perms[p][2]]) * d3_scale;
                        zt[p * o3 + ijk] = r3buf[ijk] / D;
                    }
                }

                // Phase 3: compute 36 dot products for energy
                const int comp[6][6] = {
                    {0,1,2,3,4,5}, {1,0,4,5,2,3}, {2,3,0,1,5,4},
                    {4,5,1,0,3,2}, {3,2,5,4,0,1}, {5,4,3,2,1,0}
                };

                double triple_E = 0.0;
                for (int q = 0; q < 6; q++) {
                    for (int pp = 0; pp < 6; pp++) {
                        int s = comp[q][pp];
                        for (int r = 0; r < o3; r++) {
                            int kr = r % nocc;
                            int jr = (r / nocc) % nocc;
                            int ir = r / oo;
                            int sr;
                            switch(s) {
                                case 0: sr = r; break;
                                case 1: sr = (ir*nocc+kr)*nocc+jr; break;
                                case 2: sr = (jr*nocc+ir)*nocc+kr; break;
                                case 3: sr = (jr*nocc+kr)*nocc+ir; break;
                                case 4: sr = (kr*nocc+ir)*nocc+jr; break;
                                default: sr = (kr*nocc+jr)*nocc+ir; break;
                            }
                            triple_E += wt[pp*o3+sr] * zt[q*o3+r];
                        }
                    }
                }
                E_T += triple_E;
            }
        } else {
            // Launch kernel: one block per (a,b,c) triple
            // Shared memory: r3buf (o3) + reduction buffer (blockSize)
            const int blockSize = 128;
            size_t smem_size = (o3 + blockSize) * sizeof(double);
            ccsd_t_energy_kernel<<<num_triples, blockSize, smem_size>>>(
                d_F_sum, (int)F_cols,
                d_M_sum, (int)M_cols,
                d_v_oovv_t, d_t1_t, d_eps_t,
                nocc, nvir,
                d_abc, num_triples,
                d_block_ET,
                d_g_wt, d_g_zt);
            cudaDeviceSynchronize();

            // Download partial sums and accumulate
            std::vector<double> block_ET(num_triples);
            cudaMemcpy(block_ET.data(), d_block_ET, num_triples * sizeof(double), cudaMemcpyDeviceToHost);

            for (int i = 0; i < num_triples; i++)
                E_T += block_ET[i];
        }

        // Free GPU memory
        tracked_cudaFree(d_g_wt);
        tracked_cudaFree(d_g_zt);
        tracked_cudaFree(d_F_sum);
        tracked_cudaFree(d_M_sum);
        tracked_cudaFree(d_v_oovv_t);
        tracked_cudaFree(d_t1_t);
        tracked_cudaFree(d_eps_t);
        tracked_cudaFree(d_abc);
        tracked_cudaFree(d_block_ET);

        E_T *= 2.0;
        *ccsd_t_energy = E_T;
        std::cout << "(T) correction energy: " << std::fixed << std::setprecision(12) << E_T << std::endl;
    }

    return Ecc;
}


// ============================================================
//  Legacy spin-orbital CCSD implementation (kept for reference/fallback)
// ============================================================
real_t ccsd_from_aoeri_via_full_moeri(const real_t* __restrict__ d_eri_ao, const real_t* __restrict__ d_coefficient_matrix, const real_t* __restrict__ d_orbital_energies, const int num_basis, const int num_occ, const bool computing_ccsd_t=false, real_t* ccsd_t_energy=nullptr, real_t* d_eri_mo_precomputed=nullptr) {

    const int num_spin_mo = num_basis * 2;
    const int num_spin_occ = num_occ * 2;
    const int num_spin_vir = num_spin_mo - num_spin_occ;

    // for DIIS convergence acceleration
    DIIS diis(6); // DIIS with max 6 error vectors
    size_t num_ccsd_amplitudes = (size_t)num_spin_occ * num_spin_vir + (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    std::vector<real_t> h_t_old(num_ccsd_amplitudes, 0.0); // host buffer for DIIS of t amplitudes
    std::vector<real_t> h_t_new(num_ccsd_amplitudes); // host buffer for DIIS of t amplitudes
    std::vector<real_t> h_residual(num_ccsd_amplitudes); // host buffer for DIIS residuals

    // ------------------------------------------------------------
    // 1) allocate full MO ERI on device: d_eri_mo (N x N) for RHF (closed-shell)
    // 2) AO -> MO full transformation (writes into d_eri_mo) for RHF (closed-shell)
    // ------------------------------------------------------------
    double* d_eri_mo;
    bool free_eri_mo;
    size_t bytes_mo = (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(double);
    if (d_eri_mo_precomputed) {
        d_eri_mo = d_eri_mo_precomputed;
        free_eri_mo = false;
    } else {
        d_eri_mo = nullptr;
        tracked_cudaMalloc((void**)&d_eri_mo, bytes_mo);
        if(!d_eri_mo){
            THROW_EXCEPTION("tracked_cudaMalloc failed for d_eri_mo.");
        }
        {
            std::string str = "Computing AO -> MO full integral transformation... ";
            PROFILE_ELAPSED_TIME(str);

            transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix, num_basis, d_eri_mo);
            cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        }
        free_eri_mo = true;
    }


    //debug: checking MO ERI by comparing with brute-force transformation and stored MO ERI
    // std::cout << "Checking MO ERI..." << std::endl;
    // check_moeri(d_eri_mo, d_eri_ao, d_coefficient_matrix, num_basis);

    // show all MO ERI
    /*
    real_t* h_eri = new real_t[N * N];
    cudaMemcpy(h_eri, d_eri_mo, bytes_mo, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int p = 0; p < num_basis; ++p){
        for(int q = 0; q < num_basis; ++q){
            for(int r = 0; r < num_basis; ++r){
                for(int s = 0; s < num_basis; ++s){
                    size_t idx = p * num_basis * num_basis * num_basis + q * num_basis * num_basis + r * num_basis + s;
                    std::cout << "ERI(" << p << "," << q << "," << r << "," << s << ") = " << h_eri[idx] << std::endl;
                }
            }
        }
    }
    delete[] h_eri;
    */

    // ------------------------------------------------------------
    // 3) CCSD energy from full MO ERI
    // ------------------------------------------------------------


    // ------------------------------------------------------------
    // 3-1) Memory allocation for intermediates and amplitudes
    // ------------------------------------------------------------
    
    // memory allocation for intermediates and amplitudes inside ccsd_from_moeri_full function
    real_t* F_ae = nullptr;
    real_t* F_mi = nullptr;
    real_t* F_me = nullptr;
    real_t* W_mnij = nullptr;
    real_t* W_abef = nullptr;
    real_t* W_mbej = nullptr;

    real_t* t_ia_new = nullptr;
    real_t* t_ia_old = nullptr;
    real_t* t_ijab_new = nullptr;
    real_t* t_ijab_old = nullptr;

    allocate_ccsd_intermediates(num_spin_occ, num_spin_vir,
                                &F_ae, &F_mi, &F_me,
                                &W_mnij, &W_abef, &W_mbej);
    allocate_ccsd_amplitudes(num_spin_occ, num_spin_vir,
                            &t_ia_new, &t_ia_old,
                            &t_ijab_new, &t_ijab_old);


    // ------------------------------------------------------------
    // 3-2) Computes initial values of t_ia and t_ijab amplitudes
    // ------------------------------------------------------------
    {
        std::string str = "Computing initial t_ia and t_ijab amplitudes... ";
        PROFILE_ELAPSED_TIME(str);

        // t_ia = 0
        cudaMemset(t_ia_old, 0.0, sizeof(real_t) * num_spin_occ * num_spin_vir);

        // t_ijab = <ij||ab> / (epsilon_i + epsilon_j - epsilon_a - epsilon_b)
        cudaMemset(t_ijab_old, 0.0, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir); // Never skip this zeroing because some elements may not be set in the kernel
        intialize_ccsd_amplitudes(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir, t_ijab_old);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }

    // ------------------------------------------------------------
    // 3-3) CCSD iterations
    // ------------------------------------------------------------
    int max_ccsd_iterations = 50;
    real_t convergence_threshold = 1e-7;
    int loops = 0;

    //real_t diff = 0.0;
    real_t rms = 0.0;


    real_t E_CCSD_old = compute_ccsd_energy(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia_old, t_ijab_old); // initial energy
    real_t E_CCSD_new = E_CCSD_old;

    for(loops = 0; loops < max_ccsd_iterations; ++loops){
        std::string str = "---- CCSD iteration " + std::to_string(loops+1) + " ---- ";
        if(loops == 0){
            str += "E_CCSD: " + std::to_string(E_CCSD_new) + " Hartree. ";
            str += "(initial amplitudes)";
            std::cout << str << std::endl;
        }else{
            //std::streamsize old_prec = std::cout.precision(); // save old precision
            //std::ios::fmtflags old_flags = std::cout.flags();      

            std::cout << str
              //<< "E_CCSD: " << E_CCSD_new << " Hartree. "
              << "E_CCSD difference: " << fabs(E_CCSD_new - E_CCSD_old) << " Hartree. "
              //<< "T-amplitude difference: "
              << "T-amplitude RMS: "
              //<< std::scientific        // or std::fixed
              //<< std::setprecision(12)  // number of digits
              //<< diff
              << rms
              << std::endl;

            //std::cout.precision(old_prec); // restore old precision
            //std::cout.flags(old_flags);
        }


        //debug: print t_ia and t_ijab amplitudes
        /*
        real_t* h_t_ia_old = new real_t[num_spin_occ * num_spin_vir];
        real_t* h_t_ijab_old = new real_t[num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir];
        cudaMemcpy(h_t_ia_old, t_ia_old, sizeof(real_t) * num_spin_occ * num_spin_vir, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_t_ijab_old, t_ijab_old, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir, cudaMemcpyDeviceToHost);
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = num_spin_occ + a_;
                std::cout << "t_ia[" << i << "," << a << "] = " << h_t_ia_old[i * num_spin_vir + a_] << std::endl;
            }
        }
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){
                for(int a_ = 0; a_ < num_spin_vir; ++a_){
                    for(int b_ = 0; b_ < num_spin_vir; ++b_){
                        if(i>=j || a_>=b_){ // print only unique amplitudes
                            continue;
                        }
                        int a = num_spin_occ + a_;
                        int b = num_spin_occ + b_;
                        std::cout << "t_ijab[" << i << "," << j << "," << a << "," << b << "] = " 
                                  << h_t_ijab_old[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] 
                                  << std::endl;
                    }
                }
            }
        }
   
        delete[] h_t_ia_old;
        delete[] h_t_ijab_old;
        */


        compute_t_amplitude(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir,
                            t_ia_old, t_ijab_old,
                            t_ia_new, t_ijab_new,
                            F_ae, F_mi, F_me,
                            W_mnij, W_abef, W_mbej);
        

        cudaDeviceSynchronize();

        // CCSD energy calculation
        E_CCSD_old = E_CCSD_new;
        E_CCSD_new = compute_ccsd_energy(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia_new, t_ijab_new);
        
        /////////// DIIS procedure ///////////
        // Copy new amplitudes to host for DIIS
        cudaMemcpy(h_t_new.data(), t_ia_new, sizeof(real_t) * num_ccsd_amplitudes, cudaMemcpyDeviceToHost); // t_ia_new and t_ijab_new are in contiguous buffer
        // Compute residuals for DIIS and rms difference
        rms = 0.0;
        for(size_t idx = 0; idx < num_ccsd_amplitudes; ++idx){
            h_residual[idx] = h_t_new[idx] - h_t_old[idx];
            rms += h_residual[idx] * h_residual[idx];
        }
        rms = std::sqrt(rms/num_ccsd_amplitudes);
        //std::cout << "DIIS RMS of residuals: " << rms << std::endl;
        real_t E_CCSD_diff = fabs(E_CCSD_new - E_CCSD_old);
        //std::cout << "CCSD Energy difference: " << E_CCSD_diff << " Hartree" <<  std::endl;

        if(rms < convergence_threshold || E_CCSD_diff < convergence_threshold){
            std::cout << "CCSD converged in " << (loops+1) << " iterations." << std::endl;
            break;
        }

        // Add new amplitudes and residuals to DIIS history
        //diis.push(h_t_new, h_residual);
        diis.push(h_t_old, h_residual);

        // DIIS extrapolation to get improved amplitudes
        if(loops > 4 && diis.can_extrapolate()){
            auto h_t_diis = diis.extrapolate();
            h_t_new = h_t_diis; // update new amplitudes with DIIS result
        }else{
            // damping if DIIS is not used
            real_t damping_factor = 0.3; // 0.0 ~ 1.0
            for(size_t idx = 0; idx < num_ccsd_amplitudes; ++idx){
                h_t_new[idx] = (1.0 - damping_factor) * h_t_old[idx] + damping_factor * h_t_new[idx];
            }
        }
        // Copy back to device
        cudaMemcpy(t_ia_old, h_t_new.data(), sizeof(real_t) * num_ccsd_amplitudes, cudaMemcpyHostToDevice);
        // Update old amplitudes on device for next iteration
        h_t_old = h_t_new; // update host old amplitudes

    
        // check convergence and damping
        /*
        E_CCSD_new = compute_ccsd_energy(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia_new, t_ijab_new);
        diff = compute_t_amplitude_diff(t_ia_new, t_ijab_new,
                                                    t_ia_old, t_ijab_old,
                                                    num_spin_occ,
                                                    num_spin_vir);
    
        if(diff < convergence_threshold){
            std::cout << "CCSD converged in " << (loops+1) << " iterations." << std::endl;
            break;
        }

        // update amplitudes by damping
        real_t damping_factor = 0.6;//0.9; // 0.0 ~ 1.0
        // t_old = (1 - damping_factor) * t_old + damping_factor * t_new
        update_t_amplitude_damping(t_ia_new, t_ijab_new,
                                    t_ia_old, t_ijab_old,
                                    num_spin_occ,
                                    num_spin_vir,
                                    damping_factor);
        */
    }

    deallocate_ccsd_intermediates(F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej);

    // ------------------------------------------------------------
    // 4) CCSD(T) energy calculation (optional)
    // ------------------------------------------------------------
    if(computing_ccsd_t){
        if(!ccsd_t_energy){
            THROW_EXCEPTION("ccsd_t_energy pointer is null in computing CCSD(T) energy.");
        }

        std::cout << "---- CCSD(T) correction ---- " << std::endl;
        std::string str = "Computing CCSD(T) correction energy... ";
        PROFILE_ELAPSED_TIME(str);

        real_t E_CCSD_T = compute_ccsd_t_energy(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir,
                                                t_ia_new, t_ijab_new);
        //std::cout << "CCSD correction energy: " << E_CCSD_new << " Hartree" << std::endl;
        //std::cout << "(T) correction energy: " << E_CCSD_T << " Hartree" << std::endl;
        *ccsd_t_energy = E_CCSD_T; // return CCSD(T) correction energy
    }


    deallocate_ccsd_amplitudes(t_ia_new, t_ia_old, t_ijab_new, t_ijab_old);
    if (free_eri_mo) tracked_cudaFree(d_eri_mo);

    return E_CCSD_new;
}




// ccsd_algorithm: 0 = spatial-orbital optimized (GPU DGEMM + sub-blocks)
//                 1 = spatial-orbital naive (pure CPU)
//                 2 = spin-orbital (legacy)

static real_t compute_ccsd_energy_impl(RHF& rhf, const real_t* d_eri, int ccsd_algorithm, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_basis = rhf.get_num_basis();
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf.get_orbital_energies().device_ptr();

    real_t E_CCSD;
    if (ccsd_algorithm == 2) {
        E_CCSD = ccsd_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ, false, nullptr, d_eri_mo_precomputed);
    } else if (ccsd_algorithm == 1) {
        E_CCSD = ccsd_spatial_orbital_naive(d_eri, d_C, d_eps, num_basis, num_occ, false, nullptr, d_eri_mo_precomputed);
    } else {
        E_CCSD = ccsd_spatial_orbital(d_eri, d_C, d_eps, num_basis, num_occ, false, nullptr, nullptr, nullptr, d_eri_mo_precomputed);
    }

    std::cout << "CCSD energy: " << E_CCSD << " Hartree" << std::endl;
    return E_CCSD;
}

real_t ERI_Stored_RHF::compute_ccsd_energy() {
    if (!gpu::gpu_available()) {
        // CPU: use build_mo_eri for AO→MO transform (avoids cuBLAS-only ao2mo functions)
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
        real_t result = compute_ccsd_energy_impl(rhf_, nullptr, ccsd_algorithm_, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return result;
    }
    return compute_ccsd_energy_impl(rhf_, eri_matrix_.device_ptr(), ccsd_algorithm_);
}

// ----------------------------------------------------------------------------
//  CCSD 1-RDM (Lambda density) for ERI_Stored_RHF — DMET / property analysis
// ----------------------------------------------------------------------------

void ERI_Stored_RHF::compute_ccsd_density() {
    PROFILE_FUNCTION();

    const int num_basis = rhf_.get_num_basis();
    const int num_occ = rhf_.get_num_electrons() / 2;
    const int num_vir = num_basis - num_occ;
    const size_t N4 = (size_t)num_basis * num_basis * num_basis * num_basis;
    const size_t t1_sz = (size_t)num_occ * num_vir;
    const size_t t2_sz = (size_t)num_occ * num_occ * num_vir * num_vir;

    std::cout << "\n---- CCSD 1-RDM (Lambda density) ----" << std::endl;
    std::cout << "  nocc=" << num_occ << " nvir=" << num_vir
              << " nao=" << num_basis << std::endl;

    const real_t* d_C = rhf_.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf_.get_orbital_energies().device_ptr();

    // Step 1: Build full MO ERI (used by both CCSD solver and Lambda step)
    real_t* d_eri_mo = nullptr;
    if (gpu::gpu_available()) {
        tracked_cudaMalloc(&d_eri_mo, N4 * sizeof(real_t));
        transform_ao_eri_to_mo_eri_4stage(eri_matrix_.device_ptr(), d_C, num_basis, d_eri_mo);
    } else {
        d_eri_mo = build_mo_eri(d_C, num_basis);  // CPU AO->MO via Eigen
    }

    // Step 2: Solve CCSD T amplitudes (returned as device pointers)
    real_t* d_t1 = nullptr;
    real_t* d_t2 = nullptr;
    real_t E_CCSD = ccsd_spatial_orbital(eri_matrix_.device_ptr(), d_C, d_eps,
                                          num_basis, num_occ, false, nullptr,
                                          &d_t1, &d_t2, d_eri_mo);
    rhf_.set_post_hf_energy(E_CCSD);
    std::cout << "  CCSD energy: " << std::fixed << std::setprecision(10)
              << E_CCSD << " Ha" << std::endl;

    // Step 3: Lambda solver (GPU when available, CPU fallback)
    auto& D_mo = rhf_.get_ccsd_1rdm_mo();
    auto& D_ao = rhf_.get_ccsd_1rdm_ao();
    D_mo.assign((size_t)num_basis * num_basis, 0.0);
    D_ao.assign((size_t)num_basis * num_basis, 0.0);

    if (gpu::gpu_available()) {
        // GPU path: keep amplitudes/eri on device, no host roundtrip
        real_t *d_l1=nullptr, *d_l2=nullptr, *d_D_mo=nullptr;
        tracked_cudaMalloc(&d_l1, t1_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_l2, t2_sz * sizeof(real_t));
        tracked_cudaMalloc(&d_D_mo, (size_t)num_basis * num_basis * sizeof(real_t));

        bool conv = solve_ccsd_lambda_gpu(num_occ, num_vir, d_eps, d_eri_mo,
                                          d_t1, d_t2, d_l1, d_l2,
                                          100, 1e-8, 1);
        if (!conv) std::cout << "  Warning: Lambda did not fully converge" << std::endl;

        // Diagnostic ||λ|| via cuBLAS
        {
            real_t n1, n2;
            cublasDnrm2(gpu::GPUHandle::cublas(), (int)t1_sz, d_l1, 1, &n1);
            cublasDnrm2(gpu::GPUHandle::cublas(), (int)t2_sz, d_l2, 1, &n2);
            std::cout << "  ||l1||=" << std::scientific << std::setprecision(8) << n1
                      << "  ||l2||=" << n2 << std::defaultfloat << std::endl;
        }

        build_ccsd_1rdm_mo_gpu(num_occ, num_vir, d_t1, d_t2, d_l1, d_l2, d_D_mo);

        // Download D_MO and AO transform on host (cheap)
        cudaMemcpy(D_mo.data(), d_D_mo, (size_t)num_basis * num_basis * sizeof(real_t),
                   cudaMemcpyDeviceToHost);
        std::vector<real_t> h_C(num_basis * num_basis);
        cudaMemcpy(h_C.data(), d_C, num_basis * num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);
        transform_density_mo_to_ao_cpu(num_basis, h_C.data(), D_mo.data(), D_ao.data());

        tracked_cudaFree(d_l1); tracked_cudaFree(d_l2); tracked_cudaFree(d_D_mo);
    } else {
        // CPU fallback
        std::vector<real_t> h_t1(t1_sz), h_t2(t2_sz), h_eri_mo(N4), h_eps(num_basis), h_C(num_basis*num_basis);
        cudaMemcpy(h_t1.data(),     d_t1,     t1_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_t2.data(),     d_t2,     t2_sz * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eri_mo.data(), d_eri_mo, N4    * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eps.data(),    d_eps,    num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C.data(),      d_C,      num_basis*num_basis * sizeof(real_t), cudaMemcpyDeviceToHost);

        std::vector<real_t> h_l1(t1_sz), h_l2(t2_sz);
        bool conv = solve_ccsd_lambda_cpu(num_occ, num_vir, h_eps.data(), h_eri_mo.data(),
                                          h_t1.data(), h_t2.data(),
                                          h_l1.data(), h_l2.data(), 100, 1e-8, 1);
        if (!conv) std::cout << "  Warning: Lambda did not fully converge" << std::endl;
        {
            real_t n1 = 0.0, n2 = 0.0;
            for (size_t k = 0; k < t1_sz; k++) n1 += h_l1[k]*h_l1[k];
            for (size_t k = 0; k < t2_sz; k++) n2 += h_l2[k]*h_l2[k];
            std::cout << "  ||l1||=" << std::scientific << std::setprecision(8)
                      << std::sqrt(n1) << "  ||l2||=" << std::sqrt(n2)
                      << std::defaultfloat << std::endl;
        }
        build_ccsd_1rdm_mo_cpu(num_occ, num_vir, h_t1.data(), h_t2.data(),
                               h_l1.data(), h_l2.data(), D_mo.data());
        transform_density_mo_to_ao_cpu(num_basis, h_C.data(), D_mo.data(), D_ao.data());
    }

    // Step 6: Sanity check — Tr(D_AO @ S) should equal Ne = 2*nocc
    const real_t* d_S = rhf_.get_overlap_matrix().device_ptr();
    std::vector<real_t> h_S((size_t)num_basis * num_basis);
    cudaMemcpy(h_S.data(), d_S, num_basis*num_basis*sizeof(real_t), cudaMemcpyDeviceToHost);
    real_t trace = 0.0;
    for (int p = 0; p < num_basis; p++)
      for (int q = 0; q < num_basis; q++)
        trace += D_ao[p*num_basis + q] * h_S[q*num_basis + p];
    std::cout << "  Tr(D_AO·S) = " << std::fixed << std::setprecision(8) << trace
              << " (expected " << (2*num_occ) << ")" << std::endl;

    // Diagnostic: D_MO diagonal (for PySCF comparison)
    std::cout << "  D_MO diagonal:";
    for (int p = 0; p < num_basis; p++)
        std::cout << " " << std::fixed << std::setprecision(6) << D_mo[p*num_basis + p];
    std::cout << std::endl;

    // Cleanup
    tracked_cudaFree(d_t1);
    tracked_cudaFree(d_t2);
    tracked_cudaFree(d_eri_mo);
}

real_t ERI_RI_RHF::compute_ccsd_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Direct_RHF::compute_ccsd_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Hash_RHF::compute_ccsd_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////// CCSD(T) implementation

static real_t compute_ccsd_t_energy_impl(RHF& rhf, const real_t* d_eri, int ccsd_algorithm, real_t* d_eri_mo_precomputed = nullptr) {
    PROFILE_FUNCTION();

    const int num_occ = rhf.get_num_electrons() / 2;
    const int num_basis = rhf.get_num_basis();
    const real_t* d_C = rhf.get_coefficient_matrix().device_ptr();
    const real_t* d_eps = rhf.get_orbital_energies().device_ptr();

    real_t ccsd_t_energy = 0.0;
    real_t E_CCSD;
    if (ccsd_algorithm == 2) {
        E_CCSD = ccsd_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ, true, &ccsd_t_energy, d_eri_mo_precomputed);
    } else if (ccsd_algorithm == 1) {
        E_CCSD = ccsd_spatial_orbital_naive(d_eri, d_C, d_eps, num_basis, num_occ, true, &ccsd_t_energy, d_eri_mo_precomputed);
    } else {
        E_CCSD = ccsd_spatial_orbital(d_eri, d_C, d_eps, num_basis, num_occ, true, &ccsd_t_energy, nullptr, nullptr, d_eri_mo_precomputed);
    }

    std::cout << "CCSD correction energy: " << E_CCSD << " Hartree" << std::endl;
    std::cout << "(T) correction energy: " << ccsd_t_energy << " Hartree" << std::endl;
    std::cout << "CCSD(T) correction energy: " << E_CCSD+ccsd_t_energy << " Hartree" << std::endl;

    return E_CCSD+ccsd_t_energy;
}

real_t ERI_Stored_RHF::compute_ccsd_t_energy() {
    if (!gpu::gpu_available()) {
        real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
        real_t result = compute_ccsd_t_energy_impl(rhf_, nullptr, ccsd_algorithm_, d_mo_eri);
        tracked_cudaFree(d_mo_eri);
        return result;
    }
    return compute_ccsd_t_energy_impl(rhf_, eri_matrix_.device_ptr(), ccsd_algorithm_);
}

real_t ERI_RI_RHF::compute_ccsd_t_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_t_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Direct_RHF::compute_ccsd_t_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_t_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

real_t ERI_Hash_RHF::compute_ccsd_t_energy() {
    real_t* d_mo_eri = build_mo_eri(rhf_.get_coefficient_matrix().device_ptr(), rhf_.get_num_basis());
    real_t result = compute_ccsd_t_energy_impl(rhf_, nullptr, 0, d_mo_eri);
    tracked_cudaFree(d_mo_eri);
    return result;
}

 } // namespace gansu
