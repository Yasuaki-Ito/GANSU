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


    // Naive implementation for MP3 energy calculation 
    // Note: Integral transformation is performed on-the-fly.

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








 } // namespace gansu