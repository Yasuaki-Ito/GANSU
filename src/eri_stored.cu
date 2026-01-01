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

__device__ real_t U_ijab(const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    double sum = 0.0;

    assert(i >= 0 && i < num_spin_occ);
    assert(j >= 0 && j < num_spin_occ);
    assert(a_ >= 0 && a_ < num_spin_vir);
    assert(b_ >= 0 && b_ < num_spin_vir);

    // t_ij^ab contribution
    sum += t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)];

    // 0.5 * (t_i^a t_j^b - t_i^b t_j^a)
    sum += 0.5 * (t_ia[i * num_spin_vir + a_] * t_ia[j * num_spin_vir + b_] - t_ia[i * num_spin_vir + b_] * t_ia[j * num_spin_vir + a_]);

    return sum;
}


__device__ real_t T_ijab(const real_t* __restrict__ t_ia, const real_t* __restrict__ t_ijab,
                            const int num_spin_occ,
                            const int num_spin_vir,
                            const int i, const int j, const int a_, const int b_) // a_ and b_ are indices in virtual space (0 to num_spin_vir-1)
{
    double sum = 0.0;

    assert(i >= 0 && i < num_spin_occ);
    assert(j >= 0 && j < num_spin_occ);
    assert(a_ >= 0 && a_ < num_spin_vir);
    assert(b_ >= 0 && b_ < num_spin_vir);

    // t_ij^ab contribution
    sum += t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)];

    // t_i^a * t_jb
    sum += t_ia[i * num_spin_vir + a_] * t_ia[j * num_spin_vir + b_];

    // - t_i^b * t_ja
    sum -= t_ia[i * num_spin_vir + b_] * t_ia[j * num_spin_vir + a_];

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

        double sum = 0.0;

        // (1-delta_ae) * f_ae
        // but always zero for RHF
        
        // sum over m
        // f_me * t_m^a, but f_me = 0 for RHF
        // omitted

        // sum over m, f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                // <ma||fe> = (mf|ae) - (me|af)
                double mfae = ((m%2)==(f%2) && ((a%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, a/2, e/2)] : 0.0;
                double meaf = ((m%2)==(e%2) && ((a%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, a/2, f/2)] : 0.0;

                sum += (mfae - meaf) * t_ia[m * num_spin_vir + f_]; // <ma||fe> * t_m^f
            }
        }

        // sum over m,n,f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int f = num_spin_occ + f_;
                    
                    // <mn||ef> = (me|nf) - (mf|ne)
                    double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                    double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                    sum -= 0.5 * (menf - mfne) * U_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, m, n, a_, f_); // -0.5 * <mn||ef> * U_mnaf
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

        double sum = 0.0;

        // (1-delta_mi) * f_mi
        // but always zero for RHF

        // sum over e, but RHF symmetry makes this zero (f_ia = 0)
        // - sum_e f_me * t_i^e
        // omitted

        // sum over n, e
        for(int n = 0; n < num_spin_occ; ++n){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = num_spin_occ + e_;
                
                // <mn||ie> = (mi|ne) - (me|ni)
                double mine = ((m%2)==(i%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, n/2, e/2)] : 0.0;
                double meni = ((m%2)==(e%2) && ((n%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, i/2)] : 0.0;

                sum += (mine - meni) * t_ia[n * num_spin_vir + e_]; // <mn||ie> * t_n^e
            }
        }

        // sum over n, e, f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int e = num_spin_occ + e_;
                    int f = num_spin_occ + f_;
                    
                    // <mn||ef> = (me|nf) - (mf|ne)
                    double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                    double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                    sum += 0.5 * (menf - mfne) * U_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, n, i, e_, f_); // +0.5 * <mn||ef> * U_nief
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

        double sum = 0.0;

        // f_me
        // RHF symmetry makes this zero
        // omitted

        // sum over n, f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                // <mn||ef> = (me|nf) - (mf|ne)
                double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                sum += (menf - mfne) * t_ia[n * num_spin_vir + f_]; // <mn||ef> * t_n^f
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

        double sum = 0.0;

        // <mn||ij> = (mi|nj) - (mj|ni)
        double minj = ((m%2)==(i%2) && ((n%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, n/2, j/2)] : 0.0;
        double mjni = ((m%2)==(j%2) && ((n%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, n/2, i/2)] : 0.0;

        sum += (minj - mjni);

        // sum ove e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = num_spin_occ + e_;
            
            // <mn||ie> = (mi|ne) - (me|ni)
            double mine = ((m%2)==(i%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, n/2, e/2)] : 0.0;
            double meni = ((m%2)==(e%2) && ((n%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, i/2)] : 0.0;

            sum += (mine - meni) * t_ia[j * num_spin_vir + e_]; // <mn||ie> * t_n^e

            // <mn||je> = (mj|ne) - (me|nj)
            double mjne = ((m%2)==(j%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, n/2, e/2)] : 0.0;
            double menj = ((m%2)==(e%2) && ((n%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, j/2)] : 0.0;

            sum -= (mjne - menj) * t_ia[i * num_spin_vir + e_]; // - <mn||je> * t_n^e
        }

        // sum over e,f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int e = num_spin_occ + e_;
                int f = num_spin_occ + f_;
                
                // <mn||ef> = (me|nf) - (mf|ne)
                double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                sum += 0.25 * (menf - mfne) * T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, i, j, e_, f_); // 0.25 * <mn||ef> * T_ij^ef
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

        double sum = 0.0;

        // <ab||ef> = (ae|bf) - (af|be)
        double aebf = ((a%2)==(e%2) && ((b%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, e/2, b/2, f/2)] : 0.0;
        double afbe = ((a%2)==(f%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, f/2, b/2, e/2)] : 0.0;

        sum += (aebf - afbe);

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            // <am||ef> = (ae|mf) - (af|me)
            double aemf = ((a%2)==(e%2) && ((m%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, e/2, m/2, f/2)] : 0.0;
            double afme = ((a%2)==(f%2) && ((m%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, f/2, m/2, e/2)] : 0.0;

            sum -= (aemf - afme) * t_ia[m * num_spin_vir + b_]; // - <am||ef> * t_m^b

            // swap a and b
            // <bm||ef> = (be|mf) - (bf|me)
            double bemf = ((b%2)==(e%2) && ((m%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, b/2, e/2, m/2, f/2)] : 0.0;
            double bfme = ((b%2)==(f%2) && ((m%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, b/2, f/2, m/2, e/2)] : 0.0;

            sum += (bemf - bfme) * t_ia[m * num_spin_vir + a_]; // + <bm||ef> * t_m^a
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){   
            for(int n = 0; n < num_spin_occ; ++n){
                // <mn||ef> = (me|nf) - (mf|ne)
                double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                sum += 0.25 * (menf - mfne) * T_ijab(t_ia, t_ijab, num_spin_occ, num_spin_vir, m, n, a_, b_); // 0.25 * <mn||ef> * T_mn^ab
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

        double sum = 0.0;

        // <mb||ej> = (me|bj) - (mj|be)
        double mebj = ((m%2)==(e%2) && ((b%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, b/2, j/2)] : 0.0;
        double mjbe = ((m%2)==(j%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, b/2, e/2)] : 0.0;
        sum += (mebj - mjbe);

        // sum over f
        for(int f_ = 0; f_ < num_spin_vir; ++f_){
            int f = num_spin_occ + f_;
            
            // <mb||ef> = (me|bf) - (mf|be)
            double mebf = ((m%2)==(e%2) && ((b%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, b/2, f/2)] : 0.0;
            double mfbe = ((m%2)==(f%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, b/2, e/2)] : 0.0;

            sum += (mebf - mfbe) * t_ia[j * num_spin_vir + f_]; // <mb||ef> * t_j^f
        }

        // sum over n
        for(int n = 0; n < num_spin_occ; ++n){
            // <mn||ej> = (me|nj) - (mj|ne)
            double menj = ((m%2)==(e%2) && ((n%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, j/2)] : 0.0;
            double mjne = ((m%2)==(j%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, n/2, e/2)] : 0.0;

            sum -= (menj - mjne) * t_ia[n * num_spin_vir + b_]; // - <mn||ej> * t_n^b
        }

        // sum over n,f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                // <mn||ef> = (me|nf) - (mf|ne)
                double menf = ((m%2)==(e%2) && ((n%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, n/2, f/2)] : 0.0;
                double mfne = ((m%2)==(f%2) && ((n%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, n/2, e/2)] : 0.0;

                sum -= (menf - mfne)  
                    * (0.5 * t_ijab[(j * num_spin_occ + n) * num_spin_vir * num_spin_vir + (f_ * num_spin_vir + b_)] + t_ia[j * num_spin_vir + f_] * t_ia[n * num_spin_vir + b_]); // - <mn||ef> * (0.5 * t_jn^fb + t_j^f * t_n^b)
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

        double numerator = 0.0;

        // f_ia contribution is zero due to RHF symmetry

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            numerator += F_ae[(a_ * num_spin_vir + e_)] * t_ia_old[i * num_spin_vir + e_]; // F_ae * t_i^e
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            numerator += F_mi[(m * num_spin_occ + i)] * t_ia_old[m * num_spin_vir + a_]; // F_mi * t_m^a
        }

        // sum over m,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                numerator += F_me[(m * num_spin_vir + e_)] * t_ijab_old[(i * num_spin_occ + m) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + e_)]; // F_me * t_im^ae
            }
        }

        // sum over n,f
        for(int n = 0; n < num_spin_occ; ++n){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = num_spin_occ + f_;
                
                // <na||if> = (ni|af) - (nf|ai)
                double niaf = ((n%2)==(i%2) && ((a%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, n/2, i/2, a/2, f/2)] : 0.0;
                double nfai = ((n%2)==(f%2) && ((a%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, n/2, f/2, a/2, i/2)] : 0.0;

                numerator -= (niaf - nfai) * t_ia_old[n * num_spin_vir + f_]; // - <na||if> * t_n^f
            }
        }

        // sum over m,e,f
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                for(int f_ = 0; f_ < num_spin_vir; ++f_){
                    int e = num_spin_occ + e_;
                    int f = num_spin_occ + f_;
                    
                    // <ma||ef> = (me|af) - (mf|ae)
                    double meaf = ((m%2)==(e%2) && ((a%2)==(f%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, a/2, f/2)] : 0.0;
                    double mfae = ((m%2)==(f%2) && ((a%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, f/2, a/2, e/2)] : 0.0;

                    numerator -= 0.5 * (meaf - mfae) * t_ijab_old[(i * num_spin_occ + m) * num_spin_vir * num_spin_vir + (e_ * num_spin_vir + f_)]; // - 0.5 * <ma||ef> * t_im^ef
                }
            }
        }

        // sum over m,n,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                for(int e_ = 0; e_ < num_spin_vir; ++e_){
                    int e = num_spin_occ + e_;
                    
                    // <nm||ei> = (ne|mi) - (ni|me)
                    double nemi = ((n%2)==(e%2) && ((m%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, n/2, e/2, m/2, i/2)] : 0.0;
                    double nime = ((n%2)==(i%2) && ((m%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, n/2, i/2, m/2, e/2)] : 0.0;

                    numerator += (nemi - nime) * t_ijab_old[(m * num_spin_occ + n) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + e_)]; // + <nm||ei> * t_mn^ae
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

        double numerator = 0.0;

        // <ij||ab> = (ia|jb) - (ib|ja)
        double iajb = ((i%2)==(a%2) && ((j%2)==(b%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, a/2, j/2, b/2)] : 0.0;
        double ibja = ((i%2)==(b%2) && ((j%2)==(a%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, b/2, j/2, a/2)] : 0.0;

        numerator += (iajb - ibja);

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            double sum2 = F_ae[(b_ * num_spin_vir + e_)]; // F_be
            // sum over m
            for(int m = 0; m < num_spin_occ; ++m){
                sum2 -= 0.5 * F_me[(m * num_spin_vir + e_)] * t_ia_old[m * num_spin_vir + b_]; // -0.5 * F_me * t_m^b
            }

            numerator += t_ijab_old[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + e_)] * sum2; // t_ij^ae * (...)

            // swap a_ and b_ for antisymmetry
            double sum2_asym = F_ae[(a_ * num_spin_vir + e_)]; // F_ae
            // sum over m
            for(int m = 0; m < num_spin_occ; ++m){
                sum2_asym -= 0.5 * F_me[(m * num_spin_vir + e_)] * t_ia_old[m * num_spin_vir + a_]; // -0.5 * F_me * t_m^a
            }
            numerator -= t_ijab_old[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + e_)] * sum2_asym; // - t_ij^be * (...)

        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            double sum2 = F_mi[(m * num_spin_occ + i)]; // F_mi
            // sum over e
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                sum2 += 0.5 * F_me[(m * num_spin_vir + e_)] * t_ia_old[j * num_spin_vir + e_]; // +0.5 * F_me * t_j^e
            }

            numerator += t_ijab_old[(i * num_spin_occ + m) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] * sum2; // t_im^ab * (...)

            // swap i and j for antisymmetry
            double sum2_asym = F_mi[(m * num_spin_occ + j)]; // F_mj
            // sum over e
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                sum2_asym += 0.5 * F_me[(m * num_spin_vir + e_)] * t_ia_old[i * num_spin_vir + e_]; // +0.5 * F_me * t_i^e
            }

            numerator -= t_ijab_old[(j * num_spin_occ + m) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)] * sum2_asym; // - t_jm^ab * (...)
        }

        // sum over m,n
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){
                numerator += 0.5 * T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, m, n, a_, b_) * W_mnij[(m * num_spin_occ + n) * num_spin_occ * num_spin_occ + (i * num_spin_occ + j)]; // +0.5 * T_ij^ab * W_mnij
            }
        }

        // sum over e,f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                numerator += 0.5 * T_ijab(t_ia_old, t_ijab_old, num_spin_occ, num_spin_vir, i, j, e_, f_) * W_abef[(a_ * num_spin_vir + b_) * num_spin_vir * num_spin_vir + (e_ * num_spin_vir + f_)]; // +0.5 * T_ij^ef * W_abef
            }
        }

        // sum over m,e
        for(int m = 0; m < num_spin_occ; ++m){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = num_spin_occ + e_;

                // <mb||ej> = (me|bj) - (mj|be)
                double mebj = ((m%2)==(e%2) && ((b%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, b/2, j/2)] : 0.0;
                double mjbe = ((m%2)==(j%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, b/2, e/2)] : 0.0;

                numerator += (t_ijab_old[(i * num_spin_occ + m) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + e_)] * W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)]
                           - t_ia_old[i * num_spin_vir + e_] * t_ia_old[m * num_spin_vir + b_] * (mebj - mjbe)); // + t_im^ef * W_mbej - t_i^e * t_m^b * <mb||ej>

                // swap a_ and b_ 
                // <ma||ej> = (me|aj) - (mj|ae)
                double meaj = ((m%2)==(e%2) && ((a%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, a/2, j/2)] : 0.0;
                double mjae = ((m%2)==(j%2) && ((a%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, a/2, e/2)] : 0.0;
                
                numerator -= (t_ijab_old[(i * num_spin_occ + m) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + e_)] * W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + j)]
                              - t_ia_old[i * num_spin_vir + e_] * t_ia_old[m * num_spin_vir + a_] * (meaj - mjae)); // -(t_im^ef * W_mbej - t_i^e * t_m^a * <ma||ej>)

                // swap i and j
                // <mb||ei> = (me|bi) - (mi|be)
                double mebi = ((m%2)==(e%2) && ((b%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, b/2, i/2)] : 0.0;
                double mibe = ((m%2)==(i%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, b/2, e/2)] : 0.0;

                numerator -= (t_ijab_old[(j * num_spin_occ + m) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + e_)] * W_mbej[(m * num_spin_vir + b_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)]
                               - t_ia_old[j * num_spin_vir + e_] * t_ia_old[m * num_spin_vir + b_] * (mebi - mibe)); // -(t_jm^ef * W_mbej - t_j^e * t_m^b * <mb||ei>)
                
                // swap i and j, a_ and b_
                // <ma||ei> = (me|ai) - (mi|ae)
                double meai = ((m%2)==(e%2) && ((a%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, e/2, a/2, i/2)] : 0.0;
                double miae = ((m%2)==(i%2) && ((a%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, a/2, e/2)] : 0.0;

                numerator += (t_ijab_old[(j * num_spin_occ + m) * num_spin_vir * num_spin_vir + (b_ * num_spin_vir + e_)] * W_mbej[(m * num_spin_vir + a_) * num_spin_vir * num_spin_occ + (e_ * num_spin_occ + i)]
                           - t_ia_old[j * num_spin_vir + e_] * t_ia_old[m * num_spin_vir + a_] * (meai - miae)); // + t_jm^ef * W_mbej - t_j^e * t_m^a * <ma||ei>
            }
        }

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = num_spin_occ + e_;
            
            // <ab||ej> = (ae|bj) - (aj|be)
            double aebj = ((a%2)==(e%2) && ((b%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, e/2, b/2, j/2)] : 0.0;
            double ajbe = ((a%2)==(j%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, j/2, b/2, e/2)] : 0.0;

            numerator += (aebj - ajbe) * t_ia_old[i * num_spin_vir + e_]; // + <ab||ej> * t_i^e

            // swap i and j
            // <ab||ei> = (ae|bi) - (ai|be)
            double aebi = ((a%2)==(e%2) && ((b%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, e/2, b/2, i/2)] : 0.0;
            double aibe = ((a%2)==(i%2) && ((b%2)==(e%2))) ? d_eri_mo[idx4_to_1(num_basis, a/2, i/2, b/2, e/2)] : 0.0;

            numerator -= (aebi - aibe) * t_ia_old[j * num_spin_vir + e_]; // - <ab||ei> * t_j^e
        }

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){
            // <mb||ij> = (mi|bj) - (mj|bi)
            double mibj = ((m%2)==(i%2) && ((b%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, b/2, j/2)] : 0.0;
            double mjbi = ((m%2)==(j%2) && ((b%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, b/2, i/2)] : 0.0;

            numerator -= (mibj - mjbi) * t_ia_old[m * num_spin_vir + a_]; // - <mb||ij> * t_m^a

            // swap a_ and b_
            // <ma||ij> = (mi|aj) - (mj|ai)
            double miaj = ((m%2)==(i%2) && ((a%2)==(j%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, i/2, a/2, j/2)] : 0.0;
            double mjai = ((m%2)==(j%2) && ((a%2)==(i%2))) ? d_eri_mo[idx4_to_1(num_basis, m/2, j/2, a/2, i/2)] : 0.0;

            numerator += (miaj - mjai) * t_ia_old[m * num_spin_vir + b_]; // + <ma||ij> * t_m^b
        }



        double denom = d_eps[i/2] + d_eps[j/2] - d_eps[a/2] - d_eps[b/2];
        // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
        if(fabs(denom) > 1e-14){
            t_ijab_new[gid] = numerator / denom;
        } else {
            t_ijab_new[gid] = 0.0;
        }
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

    if(gid < total){
        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int a = num_spin_occ + a_;
        int b = num_spin_occ + b_;

        // <ij||ab> = (ia|jb) - (ib|ja)
        double iajb = ((i%2)==(a%2) && ((j%2)==(b%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, a/2, j/2, b/2)] : 0.0;
        double ibja = ((i%2)==(b%2) && ((j%2)==(a%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, b/2, j/2, a/2)] : 0.0;

        double ij_ab = (iajb - ibja);

        double contrib = 0.5 * ij_ab * t_ia[i * num_spin_vir + a_] * t_ia[j * num_spin_vir + b_]; // 0.5 * <ij||ab> * t_i^a * t_j^b
        contrib += 0.25 * ij_ab * t_ijab[(i * num_spin_occ + j) * num_spin_vir * num_spin_vir + (a_ * num_spin_vir + b_)]; // 0.25 * <ij||ab> * t_ij^ab
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


void allocate_ccsd_intermediates_and_amplitudes(const int num_spin_occ, const int num_spin_vir,
                                        real_t** F_ae,
                                        real_t** F_mi,
                                        real_t** F_me,
                                        real_t** W_mnij,
                                        real_t** W_abef,
                                        real_t** W_mbej,
                                        real_t** t_ia_new,
                                        real_t** t_ia_old,
                                        real_t** t_ijab_new,
                                        real_t** t_ijab_old)
{
    // intermediates
    cudaMalloc((void**)F_ae, sizeof(real_t) * num_spin_vir * num_spin_vir);
    cudaMalloc((void**)F_mi, sizeof(real_t) * num_spin_occ * num_spin_occ);
    cudaMalloc((void**)F_me, sizeof(real_t) * num_spin_occ * num_spin_vir);
    cudaMalloc((void**)W_mnij, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ);
    cudaMalloc((void**)W_abef, sizeof(real_t) * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir);
    cudaMalloc((void**)W_mbej, sizeof(real_t) * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_occ);

    // amplitudes
    cudaMalloc((void**)t_ia_new, sizeof(real_t) * num_spin_occ * num_spin_vir);
    cudaMalloc((void**)t_ia_old, sizeof(real_t) * num_spin_occ * num_spin_vir);
    cudaMalloc((void**)t_ijab_new, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir);
    cudaMalloc((void**)t_ijab_old, sizeof(real_t) * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir);

    // error checks
    if(!(*F_ae) || !(*F_mi) || !(*F_me) || !(*W_mnij) || !(*W_abef) || !(*W_mbej) ||
       !(*t_ia_new) || !(*t_ia_old) || !(*t_ijab_new) || !(*t_ijab_old)){
        THROW_EXCEPTION("cudaMalloc failed for CCSD intermediates or amplitudes.");
    }
}

void deallocate_ccsd_intermediates_and_amplitudes(real_t* __restrict__ F_ae,
                                                real_t* __restrict__ F_mi,
                                                real_t* __restrict__ F_me,
                                                real_t* __restrict__ W_mnij,
                                                real_t* __restrict__ W_abef,
                                                real_t* __restrict__ W_mbej,
                                                real_t* __restrict__ t_ia_new,
                                                real_t* __restrict__ t_ia_old,
                                                real_t* __restrict__ t_ijab_new,
                                                real_t* __restrict__ t_ijab_old)
{
    cudaFree(F_ae);
    cudaFree(F_mi);
    cudaFree(F_me);
    cudaFree(W_mnij);
    cudaFree(W_abef);
    cudaFree(W_mbej);
    cudaFree(t_ia_new);
    cudaFree(t_ia_old);
    cudaFree(t_ijab_new);
    cudaFree(t_ijab_old);
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

        // <ij||ab> = (ia|jb) - (ib|ja)
        double iajb = ((i%2)==(a%2) && ((j%2)==(b%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, a/2, j/2, b/2)] : 0.0;
        double ibja = ((i%2)==(b%2) && ((j%2)==(a%2))) ? d_eri_mo[idx4_to_1(num_basis, i/2, b/2, j/2, a/2)] : 0.0;

        double denom = d_eps[i/2] + d_eps[j/2] - d_eps[a/2] - d_eps[b/2];
        // Avoid division by tiny denom (shouldn't happen in normal canonical RHF unless degenerate)
        if(fabs(denom) > 1e-14){
            t_ijab[gid] = (2.0*iajb - ibja) / denom;
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
    const int total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    const int num_threads = 256;
    const int num_blocks = (total + num_threads - 1) / num_threads;
    initialize_ccsd_amplitudes_kernel<<<num_blocks, num_threads>>>(d_eri_mo, d_eps, num_basis, num_spin_occ, num_spin_vir, t_ijab);
    cudaDeviceSynchronize();
}


real_t ccsd_from_aoeri_via_full_moeri(const real_t* __restrict__ d_eri_ao, const real_t* __restrict__ d_coefficient_matrix, const real_t* __restrict__ d_orbital_energies, const int num_basis, const int num_occ) {

    const int num_spin_mo = num_basis * 2;
    const int num_spin_occ = num_occ * 2;
    const int num_spin_vir = num_spin_mo - num_spin_occ;

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

    allocate_ccsd_intermediates_and_amplitudes(num_spin_occ, num_spin_vir,
                                                &F_ae, &F_mi, &F_me,
                                                &W_mnij, &W_abef, &W_mbej,
                                                &t_ia_new, &t_ia_old,
                                                &t_ijab_new, &t_ijab_old
                                                );


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

    real_t diff = 0.0;
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
              << "E_CCSD: " << E_CCSD_new << " Hartree. "
              << "T-amplitude difference: "
              << std::scientific        // or std::fixed
              << std::setprecision(12)  // number of digits
              << diff
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
        
    
        // check convergence
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
        real_t dumping_factor = 0.8;//0.9; // 0.0 ~ 1.0
        // t_old = (1 - dumping_factor) * t_old + dumping_factor * t_new
        update_t_amplitude_dumping(t_ia_new, t_ijab_new,
                                    t_ia_old, t_ijab_old,
                                    num_spin_occ,
                                    num_spin_vir,
                                    dumping_factor);
    }

    // ------------------------------------------------------------
    // 3-4) CCSD energy calculation
    // ------------------------------------------------------------
    real_t E_CCSD = compute_ccsd_energy(d_eri_mo, num_basis, num_spin_occ, num_spin_vir, t_ia_new, t_ijab_new);

    deallocate_ccsd_intermediates_and_amplitudes(
        F_ae, F_mi, F_me,
        W_mnij, W_abef, W_mbej,
        t_ia_new, t_ia_old,
        t_ijab_new, t_ijab_old
    );

    return E_CCSD;
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

    real_t E_CCSD = ccsd_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ);

    std::cout << "CCSD energy: " << E_CCSD << " Hartree" << std::endl;

    return E_CCSD;
}





 } // namespace gansu