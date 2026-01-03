/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * You may obtain a copy of the license in the LICENSE file
 * located in the root directory of this source tree or at:
 * https://opensource.org/licenses/BSD-3-Clause
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iomanip>
#include <iostream>
#include <assert.h>
#include "rhf.hpp"

#include "diis.hpp"

namespace gansu {

__device__  size_t idx4_to_1(int num_basis, int mu, int nu, int la, int si){
  return ( ( (size_t(mu)*num_basis + nu)*num_basis + la)*num_basis + si );
}


__device__ double eri_mo_bruteforce(const double* __restrict__ eri_ao,
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


__device__ double block_reduce_sum(double x){
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  sdata[tid] = x;
  __syncthreads();

  for(int s = blockDim.x/2; s>0; s>>=1){
    if(tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  return sdata[0];
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

    cudaMalloc((void**)&d_D, (size_t)N * N * sizeof(double));
    if(!d_D){
        THROW_EXCEPTION("cudaMalloc failed for d_D.");
    }
    cudaMalloc((void**)&d_T, (size_t)N * N * sizeof(double));
    if(!d_T){
        cudaFree(d_D);
        THROW_EXCEPTION("cudaMalloc failed for d_T.");
    }

    // ------------------------------------------------------------------
    // Step 1: Build D = kron(C, C)
    // ------------------------------------------------------------------
    {
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

    cudaFree(d_D);
    cudaFree(d_T);
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
    const int num_threads = 256;

    size_t total = (size_t)num_basis * num_basis * num_basis * num_basis;
    const int num_blocks = (int)((total + num_threads - 1) / num_threads);

    check_moeri_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eri_ao, d_C, num_basis);

    cudaDeviceSynchronize();
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
    cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
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

    // ------------------------------------------------------------
    // 3) MP2 energy from full MO ERI
    // ------------------------------------------------------------
    int vir = nao - occ;
    size_t total = (size_t)occ * (size_t)occ * (size_t)vir * (size_t)vir;

    double* d_E = nullptr;
    cudaMalloc((void**)&d_E, sizeof(double));
    cudaMemset(d_E, 0, sizeof(double));

    int threads = 128;
    int blocks  = (int)((total + threads - 1) / threads);
    size_t shmem = (size_t)threads * sizeof(double);

    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        mp2_from_moeri_kernel<<<blocks, threads, shmem>>>(d_eri_mo, d_eps, nao, occ, d_E);
        cudaDeviceSynchronize();
    }



    double h_E = 0.0;
    cudaMemcpy(&h_E, d_E, sizeof(double), cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------
    // 4) cleanup
    // ------------------------------------------------------------
    cudaFree(d_E);
    cudaFree(d_eri_mo);

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
    cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
    if(d_mp2_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP2 energy.");
    }
    cudaMemset(d_mp2_energy, 0.0, sizeof(real_t));

    mp2_naive_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, d_mp2_energy);

    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_mp2_energy);

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


/////////////////////////// MP2 energy calculation 


real_t ERI_Stored_RHF::compute_mp2_energy() {
    PROFILE_FUNCTION();

    // Naive implementation for MP2 energy calculation 
    // Note: Integral transformation is performed on-the-fly.

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();




//    real_t E_MP2_naive = mp2_naive(d_eri, d_C, d_eps, num_basis, num_occ);
    real_t E_MP2_stored = mp2_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ);

//    if(fabs(E_MP2_naive - E_MP2_stored) > 1e-8){
//        std::cerr << "Warning: MP2 energy mismatch between naive and stored MOERI methods." << std::endl;
//        std::cerr << "  E_MP2_naive  = " << E_MP2_naive << std::endl;
//        std::cerr << "  E_MP2_stored = " << E_MP2_stored << std::endl;
//    }

    return E_MP2_stored;
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
    cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);

    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);
       
        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_naive_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);

        cudaDeviceSynchronize();  // It is for PROFILE_ELAPSED_TIME
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_naive_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_naive_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri, d_coefficient_matrix, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);

        cudaDeviceSynchronize();  // It is for PROFILE_ELAPSED_TIME
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    cudaFree(d_mp3_energy);


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
    cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
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
    cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
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

        mp2_moeri_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, d_mp2_energy);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_mp2_energy);
    std::cout << "MP2 energy: " << h_mp2_energy << " Hartree" << std::endl;



    // ------------------------------------------------------------
    // 4) MP3 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp3_energy;
    cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);
    cudaDeviceSynchronize();

    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);
       
        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_mp3_energy);
    cudaFree(d_eri_mo);

    std::cout << "4h2p term: " << h_mp3_energy[0] << " Hartree" << std::endl;
    std::cout << "2h4p term: " << h_mp3_energy[1] << " Hartree" << std::endl;
    std::cout << "3h3p term: " << h_mp3_energy[2] << " Hartree" << std::endl;


    return h_mp2_energy + h_mp3_energy[0] + h_mp3_energy[1] + h_mp3_energy[2];
}

//////////////////////////////////////////////////////////////////////////////////////// MP3 energy calculation

real_t ERI_Stored_RHF::compute_mp3_energy() {
    PROFILE_FUNCTION();


    // MP3 energy calculation 
    // MP2 energy is also calculated inside this function.

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();

    //real_t E_MP3_naive = mp3_naive(d_eri, d_C, d_eps, num_basis, num_occ);
    real_t E_MP3 = mp3_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ);

//    if(fabs(E_MP3 - E_MP3_stored) > 1e-8){
//        std::cerr << "Warning: MP3 energy mismatch between naive and stored MOERI methods." << std::endl;
//        std::cerr << "  E_MP3_naive  = " << E_MP3 << std::endl;
//        std::cerr << "  E_MP3_stored = " << E_MP3_stored << std::endl;
//    }


    std::cout << "MP3 energy: " << E_MP3 << " Hartree" << std::endl;

    return E_MP3;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////// CCSD energy calculation

__device__ __forceinline__ real_t antisym_eri(const real_t* __restrict__ eri_mo,
                                    const int num_basis,
                                    const int p, const int q, const int r, const int s)
{
    assert(p >= 0 && p < num_basis);
    assert(q >= 0 && q < num_basis);
    assert(r >= 0 && r < num_basis);
    assert(s >= 0 && s < num_basis);

    // <pq||rs> = (pr|qs) - (ps|qr)
    real_t prqs = ((p%2)==(r%2) && ((q%2)==(s%2))) ? eri_mo[idx4_to_1(num_basis, p/2, r/2, q/2, s/2)] : 0.0;
    real_t psqr = ((p%2)==(s%2) && ((q%2)==(r%2))) ? eri_mo[idx4_to_1(num_basis, p/2, s/2, q/2, r/2)] : 0.0;
    return prqs - psqr;
}

__device__ __forceinline__ real_t t1_amplitude(const real_t* __restrict__ t_ia,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const int i, const int a_) // a_ is index in virtual space (0 to num_spin_vir-1)
{
    assert(i >= 0 && i < num_spin_occ);
    assert(a_ >= 0 && a_ < num_spin_vir);

    return t_ia[i * num_spin_vir + a_];
}

__device__ __forceinline__ real_t t2_amplitude(const real_t* __restrict__ t_ijab,
                                    const int num_spin_occ,
                                    const int num_spin_vir,
                                    const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    assert(i >= 0 && i < num_spin_occ);
    assert(j >= 0 && j < num_spin_occ);
    assert(a_ >= 0 && a_ < num_spin_vir);
    assert(b_ >= 0 && b_ < num_spin_vir);

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

    sum += 0.5 * (t_ia_val * t_jb_val - t_ib_val * t_ja_val);
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
            
            real_t mnie = antisym_eri(d_eri_mo, num_basis, m, n, i, e); // <mn||ie> = (mi|ne) - (me|ni)
            real_t t_je_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, e_);
            sum += mnie * t_je_val; // <mn||ie> * t_j^e

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
                sum += 0.25 * mnef * t_ijef_val; // 0.25 * <mn||ef> * T_ij^ef
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
            real_t amef = antisym_eri(d_eri_mo, num_basis, a, m, e, f); // <am||ef> = (ae|mf) - (af|me)
            real_t t_mb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, m, b_);

            sum -= amef * t_mb_val; // - <am||ef> * t_m^b

            // swap a and b
            real_t bemf = antisym_eri(d_eri_mo, num_basis, b, m, e, f); // <bm||ef> = (be|mf) - (bf|me)
            real_t t_ma_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, m, a_);

            sum += bemf * t_ma_val; // + <bm||ef> * t_m^a
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){   
            for(int n = 0; n < num_spin_occ; ++n){
                real_t mnef = antisym_eri(d_eri_mo, num_basis, m, n, e, f); // <mn||ef> = (me|nf) - (mf|ne)
                real_t t_mnab_val = T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, m, n, a_, b_);

                sum += 0.25 * mnef * t_mnab_val; // 0.25 * <mn||ef> * T_mn^ab
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
            
            real_t mebf = antisym_eri(d_eri_mo, num_basis, m, e, b, f); // <mb||ef> = (me|bf) - (mf|be)
            real_t t_jf_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, f_);

            sum += mebf * t_jf_val; // <mb||ef> * t_j^f
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
                real_t t_jnfb_val = T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, j, n, f_, b_); // T_jn^fb
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
            real_t t_ie_val = t_ia_old[i * num_spin_vir + e_];
            real_t F_ae_val = F_ae[a_ * num_spin_vir + e_];

            numerator += F_ae_val * t_ie_val; // F_ae * t_i^e
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            real_t t_ma_val = t_ia_old[m * num_spin_vir + a_];
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

                    numerator -= 0.5 * maef * t_imef_val; // - 0.5 * <ma||ef> * t_im^ef
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

                    numerator -= 0.5 *  nmei * t_mnae_val; // - 0.5 * <nm||ei> * t_mn^ae
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
            real_t sum2 = 0.0;
            
            real_t F_be = F_ae[(b_ * num_spin_vir + e_)]; // F_be
            sum2 += F_be;

            // sum over m
            for(int m = 0; m < num_spin_occ; ++m){
                real_t t_mb_val = t_ia_old[m * num_spin_vir + b_];
                real_t F_me_val = F_me[m * num_spin_vir + e_];

                sum2 -= 0.5 * F_me_val * t_mb_val; // -0.5 * F_me * t_m^b
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
                sum2_asym -= 0.5 * F_me_val * t_ma_val; // -0.5 * F_me * t_m^a
            }

            real_t t_ijbe_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, i, j, b_, e_);
            numerator -= t_ijbe_val * sum2_asym; // - t_ij^be * (...)
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            real_t sum2 = 0.0;

            real_t F_mj_val = F_mi[(m * num_spin_occ + j)]; // F_mj
            sum2 += F_mj_val;   
            
            // sum over e
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                real_t t_je_val = t1_amplitude(t_ia_old, num_spin_occ, num_spin_vir, j, e_);
                real_t F_me_val = F_me[(m * num_spin_vir + e_)];
                sum2 += 0.5 * F_me_val * t_je_val; // +0.5 * F_me * t_j^e
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
                sum2_asym += 0.5 * F_me_val * t_ie_val; // +0.5 * F_me * t_i^e
            }

            real_t t_jmab_val = t2_amplitude(t_ijab_old, num_spin_occ, num_spin_vir, j, m, a_, b_);
            numerator += t_jmab_val * sum2_asym; // + t_jm^ab * (...)
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                real_t T_mnab_val = T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, b_);
                real_t W_mnij_val = W_mnij[(m * num_spin_occ + n) * num_spin_occ * num_spin_occ + (i * num_spin_occ + j)];
                numerator += 0.5 * T_mnab_val * W_mnij_val; // +0.5 * T_ij^ab * W_mnij
            }
        }

        // sum over e,f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                real_t T_ijef_val = T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, j, e_, f_);
                real_t W_abef_val = W_abef[(a_ * num_spin_vir + b_) * num_spin_vir * num_spin_vir + (e_ * num_spin_vir + f_)];
                numerator += 0.5 * T_ijef_val * W_abef_val; // +0.5 * T_ij^ef * W_abef
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
        atomicMax((unsigned long long int*)&local_max, __double_as_longlong(diff));
    }
    size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    if(gid < total_ijab){
        real_t diff = fabs(t_ijab_new[gid] - t_ijab_old[gid]);
        atomicMax((unsigned long long int*)&local_max, __double_as_longlong(diff));
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicMax((unsigned long long int*)max_norm, __double_as_longlong(local_max));
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
    const int num_intermediates = 8; // F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, t_ia, t_ijab
    int computed_intermediates = 0;
    { // F_ae
        std::string str = "Computing F_ae intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_vir * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_F_ae_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_ae);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    { // F_mi
        std::string str = "Computing F_mi intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_occ;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_F_mi_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_mi);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    { // F_me
        std::string str = "Computing F_me intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_F_me_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, F_me);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    { // W_mnij
        std::string str = "Computing W_mnij intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_W_mnij_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_mnij);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    { // W_abef
        std::string str = "Computing W_abef intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_W_abef_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_abef);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;   
    }
    { // W_mbej
        std::string str = "Computing W_mbej intermediate... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_W_mbej_kernel<<<num_blocks, num_threads>>>(d_eri_mo, t_ia_old, t_ijab_old, num_basis, num_spin_occ, num_spin_vir, W_mbej);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    // Compute t_ia and t_ijab amplitudes
    { // t_ia_new
        std::string str = "Computing t_ia amplitudes... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_t_ia_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, t_ia_old, t_ijab_old, F_ae, F_mi, F_me, num_basis, num_spin_occ, num_spin_vir, t_ia_new);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
    { // t_ijab_new
        std::string str = "Computing t_ijab amplitudes... " + std::to_string(computed_intermediates+1) + "/" + std::to_string(num_intermediates);
        PROFILE_ELAPSED_TIME(str);

        const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
        const int num_threads = 256;
        const int num_blocks = (total + num_threads - 1) / num_threads;
        compute_t_ijab_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, t_ia_old, t_ijab_old, F_ae, F_mi, F_me, W_mnij, W_abef, W_mbej, num_basis, num_spin_occ, num_spin_vir, t_ijab_new);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
        computed_intermediates++;
    }
}

real_t compute_t_amplitude_diff(const real_t* __restrict__ t_ia_new, const real_t* __restrict__ t_ijab_new,
                                    const real_t* __restrict__ t_ia_old, const real_t* __restrict__ t_ijab_old,
                                    const int num_spin_occ,
                                    const int num_spin_vir)
{
    real_t h_max_norm = 0.0;
    real_t* d_max_norm = nullptr;
    cudaMalloc((void**)&d_max_norm, sizeof(real_t));
    if(!d_max_norm){
        THROW_EXCEPTION("cudaMalloc failed for d_max_norm.");
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
    cudaFree(d_max_norm);

    return h_max_norm;

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
    real_t h_rms = 0.0;
    real_t* d_rms = nullptr;
    cudaMalloc((void**)&d_rms, sizeof(real_t));
    if(!d_rms){
        THROW_EXCEPTION("cudaMalloc failed for d_rms.");
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
    cudaFree(d_rms);

    return sqrt(h_rms);
}


__global__ void update_t_amplitude_dumping_kernel(const real_t* __restrict__ t_new,
                                                real_t* __restrict__ t_old,
                                                const int dim1,
                                                const int dim2,
                                                const real_t dumping_factor)
{
    size_t total = (size_t)dim1 * dim2;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < total){
        t_old[gid] = (1.0 - dumping_factor) * t_old[gid] + dumping_factor * t_new[gid];
    }
}

void update_t_amplitude_dumping(const real_t* t_ia_new, const real_t* t_ijab_new,
                                real_t* t_ia_old, real_t* t_ijab_old,
                                const int num_spin_occ,
                                const int num_spin_vir,
                                const real_t dumping_factor)
{
    // t_ia_old = (1 - dumping_factor) * t_ia_old + dumping_factor * t_ia_new
    const size_t total_ia = (size_t)num_spin_occ * num_spin_vir;
    const int num_threads = 256;
    const int num_blocks_ia = (total_ia + num_threads - 1) / num_threads;
    update_t_amplitude_dumping_kernel<<<num_blocks_ia, num_threads>>>(t_ia_new, t_ia_old, num_spin_occ, num_spin_vir, dumping_factor);
    cudaDeviceSynchronize();

    // t_ijab_old = (1 - dumping_factor) * t_ijab_old + dumping_factor * t_ijab_new
    const size_t total_ijab = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    const int num_blocks_ijab = (total_ijab + num_threads - 1) / num_threads;
    update_t_amplitude_dumping_kernel<<<num_blocks_ijab, num_threads>>>(t_ijab_new, t_ijab_old, num_spin_occ * num_spin_occ, num_spin_vir * num_spin_vir, dumping_factor);
    cudaDeviceSynchronize();
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

    __shared__ real_t local_sum[256]; // assuming max 256 threads per block
    if(threadIdx.x < 256){
        local_sum[threadIdx.x] = 0.0;
    }
    __syncthreads();

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
        real_t ijab = antisym_eri(d_eri_mo, num_basis, i, a, j, b);
        
        real_t t_ijab_val = t2_amplitude(t_ijab, num_spin_occ, num_spin_vir, i, j, a_, b_);
        real_t t_ia_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, i, a_);
        real_t t_jb_val = t1_amplitude(t_ia, num_spin_occ, num_spin_vir, j, b_);

        double contrib = 0.5 * ijab * t_ia_val * t_jb_val; // 0.5 * <ij||ab> * t_i^a * t_j^b
        contrib += 0.25 * ijab * t_ijab_val; // 0.25 * <ij||ab> * t_ij^ab
        local_sum[threadIdx.x] += contrib;
    }
    __syncthreads();
    if(threadIdx.x == 0){
        real_t block_sum = 0.0;
        for(int i = 0; i < blockDim.x; ++i){
            block_sum += local_sum[i];
        }
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
    real_t h_ccsd_energy = 0.0;
    real_t* d_ccsd_energy = nullptr;
    cudaMalloc((void**)&d_ccsd_energy, sizeof(real_t));
    if(!d_ccsd_energy){
        THROW_EXCEPTION("cudaMalloc failed for d_ccsd_energy.");
    }
    cudaMemset(d_ccsd_energy, 0.0, sizeof(real_t));

    const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    const int num_threads = 256;
    const int num_blocks = (total + num_threads - 1) / num_threads;
    compute_ccsd_energy_kernel<<<num_blocks, num_threads>>>(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, d_ccsd_energy);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_ccsd_energy, d_ccsd_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_ccsd_energy);

    return h_ccsd_energy;
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
    cudaMalloc((void**)F_ae, sizeof(real_t) * num_spin_vir * num_spin_vir);
    cudaMalloc((void**)F_mi, sizeof(real_t) * num_spin_occ * num_spin_occ);
    cudaMalloc((void**)F_me, sizeof(real_t) * num_spin_occ * num_spin_vir);
    cudaMalloc((void**)W_mnij, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ);
    cudaMalloc((void**)W_abef, sizeof(real_t) * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir);
    cudaMalloc((void**)W_mbej, sizeof(real_t) * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ);

    // error checks
    if(!(*F_ae) || !(*F_mi) || !(*F_me) || !(*W_mnij) || !(*W_abef) || !(*W_mbej)){
        THROW_EXCEPTION("cudaMalloc failed for CCSD intermediates.");
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

    cudaMalloc((void**)&t1t2_new_buffer, sizeof(real_t) * (num_t1 + num_t2));
    cudaMalloc((void**)&t1t2_old_buffer, sizeof(real_t) * (num_t1 + num_t2));
    if(!t1t2_new_buffer || !t1t2_old_buffer){
        THROW_EXCEPTION("cudaMalloc failed for CCSD amplitudes buffer.");
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
    cudaFree(F_ae);
    cudaFree(F_mi);
    cudaFree(F_me);
    cudaFree(W_mnij);
    cudaFree(W_abef);
    cudaFree(W_mbej);
}


void deallocate_ccsd_amplitudes(real_t* __restrict__ t_ia_new,
                                real_t* __restrict__ t_ia_old,
                                real_t* __restrict__ t_ijab_new,
                                real_t* __restrict__ t_ijab_old)
{
    // t_ijab_new and t_ijab_old are part of t_ia_new and t_ia_old buffers, so no need to free them separately    
    cudaFree(t_ia_new); // free both t_ia_new and t_ijab_new as they are in the same buffer
    cudaFree(t_ia_old); // free both t_ia_old and t_ijab_old as they are in the same buffer
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
        real_t ijab = antisym_eri(d_eri_mo, num_basis, i, a, j, b);

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
    const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    const int num_threads = 256;
    const int num_blocks = (total + num_threads - 1) / num_threads;
    initialize_ccsd_amplitudes_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ijab);
    cudaDeviceSynchronize();
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


real_t compute_ccsd_t_energy(const real_t* __restrict__ d_eri_mo,
                                const real_t* __restrict__ d_eps,
                                const int num_basis,
                                const int num_spin_occ,
                                const int num_spin_vir,
                                const real_t* __restrict__ t_ia,
                                const real_t* __restrict__ t_ijab)
{

    // Compute CCSD(T) energy
    real_t h_E_CCSD_T = 0.0;
    real_t* d_E_CCSD_T = nullptr;
    cudaMalloc((void**)&d_E_CCSD_T, sizeof(real_t));
    if(!d_E_CCSD_T){
        THROW_EXCEPTION("cudaMalloc failed for d_E_CCSD_T.");
    }
    cudaMemset(d_E_CCSD_T, 0.0, sizeof(real_t));

    const size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    const int num_threads = 256;
    const int num_blocks = (total + num_threads - 1) / num_threads;
    compute_ccsd_t_energy_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ia, t_ijab, d_E_CCSD_T);
    
    cudaMemcpy(&h_E_CCSD_T, d_E_CCSD_T, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(d_E_CCSD_T);

    return h_E_CCSD_T;
}




real_t ccsd_from_aoeri_via_full_moeri(const real_t* __restrict__ d_eri_ao, const real_t* __restrict__ d_coefficient_matrix, const real_t* __restrict__ d_orbital_energies, const int num_basis, const int num_occ, const bool computing_ccsd_t=false, real_t* ccsd_t_energy=nullptr) {

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
    // ------------------------------------------------------------
    double* d_eri_mo = nullptr;
    size_t bytes_mo = (size_t)num_basis * num_basis * num_basis * num_basis * sizeof(double);
    cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
    }


    // ------------------------------------------------------------
    // 2) AO -> MO full transformation (writes into d_eri_mo) for RHF (closed-shell)
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
            std::streamsize old_prec = std::cout.precision(); // save old precision
            std::ios::fmtflags old_flags = std::cout.flags();      

            std::cout << str
              //<< "E_CCSD: " << E_CCSD_new << " Hartree. "
              << "E_CCSD difference: " << fabs(E_CCSD_new - E_CCSD_old) << " Hartree. "
              //<< "T-amplitude difference: "
              << "T-amplitude RMS: "
              << std::scientific        // or std::fixed
              << std::setprecision(12)  // number of digits
              //<< diff
              << rms
              << std::endl;

            std::cout.precision(old_prec); // restore old precision
            std::cout.flags(old_flags);
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
        for(size_t idx = 0; idx < num_ccsd_amplitudes; ++idx){
            h_residual[idx] = h_t_new[idx] - h_t_old[idx];
            rms += h_residual[idx] * h_residual[idx];
        }
        rms = std::sqrt(rms/num_ccsd_amplitudes);
        std::cout << "DIIS RMS of residuals: " << std::scientific << std::setprecision(12) << rms << std::endl;
        real_t E_CCSD_diff = fabs(E_CCSD_new - E_CCSD_old);
        std::cout << "CCSD Energy difference: " << std::scientific << std::setprecision(12) << E_CCSD_diff << " Hartree" <<  std::endl;

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
            // dumping if DIIS is not used
            real_t dumping_factor = 0.3; // 0.0 ~ 1.0
            for(size_t idx = 0; idx < num_ccsd_amplitudes; ++idx){
                h_t_new[idx] = (1.0 - dumping_factor) * h_t_old[idx] + dumping_factor * h_t_new[idx];
            }
        }
        // Copy back to device
        cudaMemcpy(t_ia_old, h_t_new.data(), sizeof(real_t) * num_ccsd_amplitudes, cudaMemcpyHostToDevice);
        // Update old amplitudes on device for next iteration
        h_t_old = h_t_new; // update host old amplitudes

    
        // check convergence and dumping
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

        // update amplitudes by dumping
        real_t dumping_factor = 0.6;//0.9; // 0.0 ~ 1.0
        // t_old = (1 - dumping_factor) * t_old + dumping_factor * t_new
        update_t_amplitude_dumping(t_ia_new, t_ijab_new,
                                    t_ia_old, t_ijab_old,
                                    num_spin_occ,
                                    num_spin_vir,
                                    dumping_factor);
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
    cudaFree(d_eri_mo);

    return E_CCSD_new;
}




real_t ERI_Stored_RHF::compute_ccsd_energy() {
    PROFILE_FUNCTION();


    // CCSD energy calculation 

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();

    bool including_ccsd_t = false;

    real_t E_CCSD = ccsd_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ, including_ccsd_t);

    std::cout << "CCSD energy: " << E_CCSD << " Hartree" << std::endl;

    return E_CCSD;
}


////////////////////////////////////////////////////////////////////////////////////////////////// CCSD(T) implementation





real_t ERI_Stored_RHF::compute_ccsd_t_energy() {
    PROFILE_FUNCTION();

    // CCSD energy calculation 

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();
    
    bool computing_ccsd_t = true;
    real_t ccsd_t_energy = 0.0;

    real_t E_CCSD = ccsd_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ, computing_ccsd_t, &ccsd_t_energy);


    std::cout << "CCSD correction energy: " << E_CCSD << " Hartree" << std::endl;
    std::cout << "(T) correction energy: " << ccsd_t_energy << " Hartree" << std::endl;
    std::cout << "CCSD(T) correction energy: " << E_CCSD+ccsd_t_energy << " Hartree" << std::endl;

    return E_CCSD+ccsd_t_energy;

    return 0.0;
}


 } // namespace gansu