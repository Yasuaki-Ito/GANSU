/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file spherical_transform_gpu.cu
 * @brief GPU (cuBLAS) implementation of the 4-index ERI Cart→Sph transform.
 *
 * Replaces the host loop / Eigen DGEMM path of the same transform.  Each
 * of the 4 sequential single-index transforms becomes one cuBLAS DGEMM:
 *
 *   Stage 1 : T1[p,q',r',s'] = U[p,p'] · ERI_cart[p',q',r',s']
 *             single dgemm  (M=ns, K=nc, N=nc³)
 *
 *   Stage 2 : T2[p,q,r',s']  = U[q,q'] · T1[p,q',r',s']
 *             strided-batched dgemm — batch over outer p (ns batches),
 *             each batch is (M=ns, K=nc, N=nc²)
 *
 *   Stage 3 : T3[p,q,r,s']   = U[r,r'] · T2[p,q,r',s']
 *             strided-batched dgemm — batch over (p,q) (ns² batches),
 *             each batch is (M=ns, K=nc, N=nc)
 *
 *   Stage 4 : ERI_sph[p,q,r,s] = T3[p,q,r,s']·U^T[s',s]
 *             single dgemm  (M=ns³, K=nc, N=ns)
 *
 * Memory peak: max(|ERI_cart| + |T1|, |T1| + |T2|, …) — intermediates
 * are released as soon as the next stage consumes them.
 *
 * Row-major arrays are passed to cuBLAS using the standard swap trick:
 *   To compute  C_row[M×N] = A_row[M×K] · B_row[K×N]  (row-major), call
 *     cublasDgemm(N, N, m_cu=N, n_cu=M, k_cu=K,
 *                 α, B_row_ptr, ldb=N,
 *                    A_row_ptr, lda=K,
 *                 β, C_row_ptr, ldc=N);
 * For ops involving transposes we mirror this rule.
 */

#include "spherical_transform.hpp"
#include "gpu_manager.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace gansu::spherical {

namespace {

// Build the dense U_full[nbf_sph × nbf_cart] (row-major host buffer) and
// upload to device. Block-diagonal in shell index; zeros elsewhere.
// Returns a unique_ptr-style raii via DeviceHostMatrix-equivalent pattern,
// but for this short-lived buffer a raw cudaMalloc/cudaFree pair suffices.
void build_and_upload_U_full(const std::vector<int>& shell_types,
                             const std::vector<int>& shell_offsets_cart,
                             const std::vector<int>& shell_offsets_sph,
                             int nbf_cart, int nbf_sph,
                             real_t** d_U_out) {
    const int n_shells = (int)shell_types.size();
    std::vector<real_t> U_host((size_t)nbf_sph * nbf_cart, 0.0);
    for (int i_shell = 0; i_shell < n_shells; ++i_shell) {
        const auto U_i     = get_cart_to_sph_matrix(shell_types[i_shell]);
        const int n_sph_i  = (int)U_i.size();
        const int n_cart_i = (int)U_i[0].size();
        const int sph_off  = shell_offsets_sph[i_shell];
        const int cart_off = shell_offsets_cart[i_shell];
        for (int a = 0; a < n_sph_i; ++a)
            for (int b = 0; b < n_cart_i; ++b)
                U_host[(size_t)(sph_off + a) * nbf_cart + (cart_off + b)] = U_i[a][b];
    }
    const size_t bytes = (size_t)nbf_sph * nbf_cart * sizeof(real_t);
    cudaError_t err = cudaMalloc((void**)d_U_out, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("transform_eri_cart_to_sph_device: "
            "cudaMalloc for U_full failed: ") + cudaGetErrorString(err));
    }
    err = cudaMemcpy(*d_U_out, U_host.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*d_U_out);
        throw std::runtime_error(std::string("transform_eri_cart_to_sph_device: "
            "cudaMemcpy U_full H2D failed: ") + cudaGetErrorString(err));
    }
}

} // anonymous namespace


void transform_eri_cart_to_sph_device(
    const real_t* d_eri_cart,
    real_t* d_eri_sph,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_eri_cart_to_sph_device: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    // Upload block-diagonal U_full (nbf_sph × nbf_cart).
    real_t* d_U = nullptr;
    build_and_upload_U_full(shell_types, shell_offsets_cart, shell_offsets_sph,
                            nbf_cart, nbf_sph, &d_U);

    const long long nc = (long long)nbf_cart;
    const long long ns = (long long)nbf_sph;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // ============================================================
    // Stage 1: T1[ns × nc³] = U[ns × nc] · ERI_cart[nc × nc³]   (row-major)
    // cuBLAS row-major trick:
    //   cublasDgemm(N, N, m_cu=nc³, n_cu=ns, k_cu=nc,
    //               α, ERI_cart, ldb=nc³, U, lda=nc, β, T1, ldc=nc³)
    // ============================================================
    real_t* d_T1 = nullptr;
    const size_t t1_bytes = (size_t)ns * nc * nc * nc * sizeof(real_t);
    cudaError_t cerr = cudaMalloc((void**)&d_T1, t1_bytes);
    if (cerr != cudaSuccess) {
        cudaFree(d_U);
        throw std::runtime_error(std::string("Stage 1 T1 alloc failed: ")
            + cudaGetErrorString(cerr));
    }
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nc * nc * nc, ns, nc,
        &alpha,
        d_eri_cart, nc * nc * nc,
        d_U,        nc,
        &beta,
        d_T1,       nc * nc * nc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T1); cudaFree(d_U);
        throw std::runtime_error("Stage 1 cublasDgemm failed: status="
            + std::to_string((int)st));
    }

    // ============================================================
    // Stage 2: T2[ns × ns × nc²] from T1[ns × nc × nc²], slab over outer p.
    //   Per-slab row-major: T2_slab[ns × nc²] = U[ns × nc] · T1_slab[nc × nc²]
    //   cuBLAS strided batched (batch = ns over the outer p):
    //     per batch: m_cu = nc², n_cu = ns, k_cu = nc
    //                A_arg = T1, lda = nc², strideA = nc * nc² = nc³
    //                B_arg = U,  ldb = nc,  strideB = 0   (shared U)
    //                C_arg = T2, ldc = nc², strideC = ns * nc²
    // ============================================================
    real_t* d_T2 = nullptr;
    const size_t t2_bytes = (size_t)ns * ns * nc * nc * sizeof(real_t);
    cerr = cudaMalloc((void**)&d_T2, t2_bytes);
    if (cerr != cudaSuccess) {
        cudaFree(d_T1); cudaFree(d_U);
        throw std::runtime_error(std::string("Stage 2 T2 alloc failed: ")
            + cudaGetErrorString(cerr));
    }
    st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nc * nc, ns, nc,
        &alpha,
        d_T1, nc * nc, nc * nc * nc,    // strideA = T1 slab size
        d_U,  nc,      0,               // strideB = 0 (U shared)
        &beta,
        d_T2, nc * nc, ns * nc * nc,    // strideC = T2 slab size
        nbf_sph);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T2); cudaFree(d_T1); cudaFree(d_U);
        throw std::runtime_error("Stage 2 cublasDgemmStridedBatched failed: status="
            + std::to_string((int)st));
    }
    cudaFree(d_T1);  // T1 no longer needed
    d_T1 = nullptr;

    // ============================================================
    // Stage 3: T3[ns × ns × ns × nc] from T2[ns × ns × nc × nc], slab over
    //          outer (p, q).  ns² batches.
    //   Per-slab: T3_slab[ns × nc] = U[ns × nc] · T2_slab[nc × nc]
    //   Batched: m_cu = nc, n_cu = ns, k_cu = nc
    //            A_arg = T2, lda = nc, strideA = nc²
    //            B_arg = U,  ldb = nc, strideB = 0
    //            C_arg = T3, ldc = nc, strideC = ns·nc
    // ============================================================
    real_t* d_T3 = nullptr;
    const size_t t3_bytes = (size_t)ns * ns * ns * nc * sizeof(real_t);
    cerr = cudaMalloc((void**)&d_T3, t3_bytes);
    if (cerr != cudaSuccess) {
        cudaFree(d_T2); cudaFree(d_U);
        throw std::runtime_error(std::string("Stage 3 T3 alloc failed: ")
            + cudaGetErrorString(cerr));
    }
    st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nc, ns, nc,
        &alpha,
        d_T2, nc, nc * nc,              // strideA = nc²
        d_U,  nc, 0,                    // strideB = 0
        &beta,
        d_T3, nc, ns * nc,              // strideC = ns·nc
        nbf_sph * nbf_sph);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T3); cudaFree(d_T2); cudaFree(d_U);
        throw std::runtime_error("Stage 3 cublasDgemmStridedBatched failed: status="
            + std::to_string((int)st));
    }
    cudaFree(d_T2);  // T2 no longer needed
    d_T2 = nullptr;

    // ============================================================
    // Stage 4: ERI_sph[ns³ × ns] = T3[ns³ × nc] · U^T[nc × ns]   (row-major)
    //   In row-major terms: C = A · B^T  where B (= U) is stored as ns × nc.
    //   Using the cuBLAS swap trick for row-major (C = A · op_row_B(B)):
    //     cublasDgemm(op_row_B=T, op_row_A=N,
    //                 m_cu = ns (cols of C_row),
    //                 n_cu = ns³ (rows of C_row),
    //                 k_cu = nc,
    //                 α, U_ptr,  ldb = nc (B's row-stride = its ncol),
    //                    T3_ptr, lda = nc (A's row-stride),
    //                 β, ERI_sph_ptr, ldc = ns)
    // ============================================================
    st = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        ns, ns * ns * ns, nc,
        &alpha,
        d_U,  nc,
        d_T3, nc,
        &beta,
        d_eri_sph, ns);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T3); cudaFree(d_U);
        throw std::runtime_error("Stage 4 cublasDgemm failed: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();

    cudaFree(d_T3);
    cudaFree(d_U);
}


void transform_matrix_cart_to_sph_device(
    const real_t* d_M_cart,
    real_t* d_M_sph,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_matrix_cart_to_sph_device: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    real_t* d_U = nullptr;
    build_and_upload_U_full(shell_types, shell_offsets_cart, shell_offsets_sph,
                            nbf_cart, nbf_sph, &d_U);

    const long long nc = (long long)nbf_cart;
    const long long ns = (long long)nbf_sph;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // Stage 1: T[ns × nc] = U[ns × nc] · M_cart[nc × nc]
    real_t* d_T = nullptr;
    cudaError_t cerr = cudaMalloc((void**)&d_T, (size_t)ns * nc * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_U);
        throw std::runtime_error(std::string("matrix Stage 1 alloc: ")
            + cudaGetErrorString(cerr));
    }
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nc, ns, nc, &alpha, d_M_cart, nc, d_U, nc, &beta, d_T, nc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T); cudaFree(d_U);
        throw std::runtime_error("matrix Stage 1 dgemm: status="
            + std::to_string((int)st));
    }

    // Stage 2: M_sph[ns × ns] = T[ns × nc] · U^T[nc × ns]
    st = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        ns, ns, nc, &alpha, d_U, nc, d_T, nc, &beta, d_M_sph, ns);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T); cudaFree(d_U);
        throw std::runtime_error("matrix Stage 2 dgemm: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();

    cudaFree(d_T);
    cudaFree(d_U);
}


void transform_matrix_sph_to_cart_device(
    const real_t* d_M_sph,
    real_t* d_M_cart,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_matrix_sph_to_cart_device: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    real_t* d_U = nullptr;   // row-major U[ns × nc]  (= col-major Uᵀ [nc × ns])
    build_and_upload_U_full(shell_types, shell_offsets_cart, shell_offsets_sph,
                            nbf_cart, nbf_sph, &d_U);

    const long long nc = (long long)nbf_cart;
    const long long ns = (long long)nbf_sph;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // M_cart = Uᵀ · M_sph · U.  With the d_U buffer interpreted col-major as Ub
    // (= Uᵀ [nc × ns]):  M_cart = Ub · M_sph · Ubᵀ.
    real_t* d_T = nullptr;   // [nc × ns]
    cudaError_t cerr = cudaMalloc((void**)&d_T, (size_t)nc * ns * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_U);
        throw std::runtime_error(std::string("sph→cart Stage 1 alloc: ")
            + cudaGetErrorString(cerr));
    }
    // Stage 1: T[nc × ns] = Ub[nc × ns] · M_sph[ns × ns]
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        nc, ns, ns, &alpha, d_U, nc, d_M_sph, ns, &beta, d_T, nc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T); cudaFree(d_U);
        throw std::runtime_error("sph→cart Stage 1 dgemm: status="
            + std::to_string((int)st));
    }
    // Stage 2: M_cart[nc × nc] = T[nc × ns] · Ubᵀ[ns × nc]
    st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nc, nc, ns, &alpha, d_T, nc, d_U, nc, &beta, d_M_cart, nc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T); cudaFree(d_U);
        throw std::runtime_error("sph→cart Stage 2 dgemm: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();

    cudaFree(d_T);
    cudaFree(d_U);
}


void transform_coeff_sph_to_cart_device(
    const real_t* d_C_sph,
    real_t* d_C_cart,
    int nmo,
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size() != n_shells + 1) {
        throw std::runtime_error(
            "transform_coeff_sph_to_cart_device: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    real_t* d_U = nullptr;   // row-major U[ns × nc] (= col-major Ub [nc × ns] = Uᵀ)
    build_and_upload_U_full(shell_types, shell_offsets_cart, shell_offsets_sph,
                            nbf_cart, nbf_sph, &d_U);

    const long long nc = (long long)nbf_cart;
    const long long ns = (long long)nbf_sph;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // C_cart[nc × nmo] = Uᵀ · C_sph[ns × nmo].  Col-major: C_sph (row-major
    // [ns × nmo]) is Cs [nmo × ns]; output C_cart (row-major [nc × nmo]) is Cc
    // [nmo × nc].  Cc = Cs · Ubᵀ where Ub (= d_U col-major [nc × ns]) = Uᵀ.
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nmo, nc, ns,
        &alpha,
        d_C_sph, nmo,
        d_U,     nc,
        &beta,
        d_C_cart, nmo);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_U);
        throw std::runtime_error("coeff sph→cart dgemm: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();
    cudaFree(d_U);
}


void transform_3index_cart_to_sph_device(
    const real_t* d_T_cart,
    real_t* d_T_sph,
    const std::vector<int>& aux_shell_types,
    const std::vector<int>& aux_shell_offsets_cart,
    const std::vector<int>& aux_shell_offsets_sph,
    const std::vector<int>& orb_shell_types,
    const std::vector<int>& orb_shell_offsets_cart,
    const std::vector<int>& orb_shell_offsets_sph)
{
    const int n_aux_shells = (int)aux_shell_types.size();
    const int n_orb_shells = (int)orb_shell_types.size();
    if ((int)aux_shell_offsets_cart.size() != n_aux_shells + 1 ||
        (int)aux_shell_offsets_sph.size()  != n_aux_shells + 1 ||
        (int)orb_shell_offsets_cart.size() != n_orb_shells + 1 ||
        (int)orb_shell_offsets_sph.size()  != n_orb_shells + 1) {
        throw std::runtime_error(
            "transform_3index_cart_to_sph_device: offset arrays must have size n_shells+1");
    }
    const int nc_aux = aux_shell_offsets_cart[n_aux_shells];
    const int ns_aux = aux_shell_offsets_sph[n_aux_shells];
    const int nc_orb = orb_shell_offsets_cart[n_orb_shells];
    const int ns_orb = orb_shell_offsets_sph[n_orb_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    real_t* d_U_aux = nullptr;
    real_t* d_U_orb = nullptr;
    build_and_upload_U_full(aux_shell_types, aux_shell_offsets_cart,
                            aux_shell_offsets_sph, nc_aux, ns_aux, &d_U_aux);
    build_and_upload_U_full(orb_shell_types, orb_shell_offsets_cart,
                            orb_shell_offsets_sph, nc_orb, ns_orb, &d_U_orb);

    const long long ncA = (long long)nc_aux;
    const long long nsA = (long long)ns_aux;
    const long long ncO = (long long)nc_orb;
    const long long nsO = (long long)ns_orb;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // ============================================================
    // Stage 1 (P axis): T1[ns_aux × nc_orb²] = U_aux[ns_aux × nc_aux] · in[nc_aux × nc_orb²]
    //   Single DGEMM in row-major sense
    // ============================================================
    real_t* d_T1 = nullptr;
    cudaError_t cerr = cudaMalloc((void**)&d_T1,
        (size_t)nsA * ncO * ncO * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error(std::string("3idx Stage 1 alloc: ")
            + cudaGetErrorString(cerr));
    }
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ncO * ncO, nsA, ncA,
        &alpha,
        d_T_cart, ncO * ncO,
        d_U_aux,  ncA,
        &beta,
        d_T1,     ncO * ncO);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx Stage 1 dgemm: status="
            + std::to_string((int)st));
    }

    // ============================================================
    // Stage 2 (μ axis): T2[ns_aux × ns_orb × nc_orb] from T1[ns_aux × nc_orb × nc_orb]
    //   Strided batched over outer P axis.  Per-slab DGEMM:
    //     T2_slab[ns_orb × nc_orb] = U_orb[ns_orb × nc_orb] · T1_slab[nc_orb × nc_orb]
    //     (same as Stage 3 in the 4-index transform)
    // ============================================================
    real_t* d_T2 = nullptr;
    cerr = cudaMalloc((void**)&d_T2,
        (size_t)nsA * nsO * ncO * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error(std::string("3idx Stage 2 alloc: ")
            + cudaGetErrorString(cerr));
    }
    st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ncO, nsO, ncO,
        &alpha,
        d_T1,    ncO, ncO * ncO,         // strideA = nc_orb²
        d_U_orb, ncO, 0,                 // strideB = 0
        &beta,
        d_T2,    ncO, nsO * ncO,         // strideC = ns_orb · nc_orb
        ns_aux);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T2); cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx Stage 2 dgemm: status="
            + std::to_string((int)st));
    }
    cudaFree(d_T1);
    d_T1 = nullptr;

    // ============================================================
    // Stage 3 (ν axis): out[ns_aux × ns_orb × ns_orb] = T2[ns_aux × ns_orb × nc_orb] · U_orb^T[nc_orb × ns_orb]
    //   View T2 as (ns_aux · ns_orb) × nc_orb and out as (ns_aux · ns_orb) × ns_orb.
    //   Single DGEMM with op_row_B = T on U_orb.
    // ============================================================
    st = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        nsO, nsA * nsO, ncO,
        &alpha,
        d_U_orb, ncO,
        d_T2,    ncO,
        &beta,
        d_T_sph, nsO);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T2); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx Stage 3 dgemm: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();

    cudaFree(d_T2);
    cudaFree(d_U_aux);
    cudaFree(d_U_orb);
}


void transform_3index_sph_to_cart_device(
    const real_t* d_T_sph,
    real_t* d_T_cart,
    const std::vector<int>& aux_shell_types,
    const std::vector<int>& aux_shell_offsets_cart,
    const std::vector<int>& aux_shell_offsets_sph,
    const std::vector<int>& orb_shell_types,
    const std::vector<int>& orb_shell_offsets_cart,
    const std::vector<int>& orb_shell_offsets_sph)
{
    const int n_aux_shells = (int)aux_shell_types.size();
    const int n_orb_shells = (int)orb_shell_types.size();
    if ((int)aux_shell_offsets_cart.size() != n_aux_shells + 1 ||
        (int)aux_shell_offsets_sph.size()  != n_aux_shells + 1 ||
        (int)orb_shell_offsets_cart.size() != n_orb_shells + 1 ||
        (int)orb_shell_offsets_sph.size()  != n_orb_shells + 1) {
        throw std::runtime_error(
            "transform_3index_sph_to_cart_device: offset arrays must have size n_shells+1");
    }
    const int nc_aux = aux_shell_offsets_cart[n_aux_shells];
    const int ns_aux = aux_shell_offsets_sph[n_aux_shells];
    const int nc_orb = orb_shell_offsets_cart[n_orb_shells];
    const int ns_orb = orb_shell_offsets_sph[n_orb_shells];

    cublasHandle_t handle = gpu::GPUHandle::cublas();

    real_t* d_U_aux = nullptr;   // row-major U_aux[ns_aux × nc_aux]
    real_t* d_U_orb = nullptr;   // row-major U_orb[ns_orb × nc_orb]
    build_and_upload_U_full(aux_shell_types, aux_shell_offsets_cart,
                            aux_shell_offsets_sph, nc_aux, ns_aux, &d_U_aux);
    build_and_upload_U_full(orb_shell_types, orb_shell_offsets_cart,
                            orb_shell_offsets_sph, nc_orb, ns_orb, &d_U_orb);

    const long long ncA = (long long)nc_aux;
    const long long nsA = (long long)ns_aux;
    const long long ncO = (long long)nc_orb;
    const long long nsO = (long long)ns_orb;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // ============================================================
    // Stage 1 (P aux): T1[nc_aux × ns_orb²] = U_auxᵀ · in[ns_aux × ns_orb²]
    //   Row-major C[nc_aux × N] = A[nc_aux × ns_aux]·B[ns_aux × N] with A = U_auxᵀ
    //   (N = ns_orb²).  Col-major: C_cm[N × nc_aux] = in_cm[N × ns_aux] · U_aux_cmᵀ,
    //   so opA = N on in, opB = T on U_aux.
    // ============================================================
    real_t* d_T1 = nullptr;
    cudaError_t cerr = cudaMalloc((void**)&d_T1,
        (size_t)ncA * nsO * nsO * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error(std::string("3idx-rev Stage 1 alloc: ")
            + cudaGetErrorString(cerr));
    }
    cublasStatus_t st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        nsO * nsO, ncA, nsA,
        &alpha,
        d_T_sph, nsO * nsO,
        d_U_aux, ncA,
        &beta,
        d_T1,    nsO * nsO);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx-rev Stage 1 dgemm: status="
            + std::to_string((int)st));
    }

    // ============================================================
    // Stage 2 (ν orb): T2[nc_aux × ns_orb × nc_orb] = T1[nc_aux × ns_orb × ns_orb]
    //                  · U_orb[ns_orb × nc_orb]   (trailing ν axis ns→nc)
    //   View T1 as (nc_aux·ns_orb) × ns_orb, out T2 as (nc_aux·ns_orb) × nc_orb.
    //   Row-major C = A·B: cublasDgemm(N,N, m_cu=nc_orb, n_cu=nc_aux·ns_orb,
    //                                  k_cu=ns_orb, U_orb(ldb=nc_orb), T1(lda=ns_orb)).
    // ============================================================
    real_t* d_T2 = nullptr;
    cerr = cudaMalloc((void**)&d_T2,
        (size_t)ncA * nsO * ncO * sizeof(real_t));
    if (cerr != cudaSuccess) {
        cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error(std::string("3idx-rev Stage 2 alloc: ")
            + cudaGetErrorString(cerr));
    }
    st = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ncO, ncA * nsO, nsO,
        &alpha,
        d_U_orb, ncO,
        d_T1,    nsO,
        &beta,
        d_T2,    ncO);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T2); cudaFree(d_T1); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx-rev Stage 2 dgemm: status="
            + std::to_string((int)st));
    }
    cudaFree(d_T1);
    d_T1 = nullptr;

    // ============================================================
    // Stage 3 (μ orb): out[nc_aux × nc_orb × nc_orb] from T2[nc_aux × ns_orb × nc_orb]
    //   Per nc_aux slab: out_slab[nc_orb × nc_orb] = U_orbᵀ · T2_slab[ns_orb × nc_orb].
    //   Col-major: out_cm = T2_cm · U_orb_cmᵀ  → opA=N on T2, opB=T on U_orb.
    // ============================================================
    st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
        ncO, ncO, nsO,
        &alpha,
        d_T2,    ncO, nsO * ncO,        // strideA = T2 slab (ns_orb · nc_orb)
        d_U_orb, ncO, 0,                // strideB = 0 (shared)
        &beta,
        d_T_cart, ncO, ncO * ncO,       // strideC = nc_orb²
        nc_aux);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaFree(d_T2); cudaFree(d_U_aux); cudaFree(d_U_orb);
        throw std::runtime_error("3idx-rev Stage 3 dgemm: status="
            + std::to_string((int)st));
    }
    cudaDeviceSynchronize();

    cudaFree(d_T2);
    cudaFree(d_U_aux);
    cudaFree(d_U_orb);
}


void build_cart_to_sph_U_device(
    const std::vector<int>& shell_types,
    const std::vector<int>& shell_offsets_cart,
    const std::vector<int>& shell_offsets_sph,
    real_t** d_U_out)
{
    const int n_shells = (int)shell_types.size();
    if ((int)shell_offsets_cart.size() != n_shells + 1 ||
        (int)shell_offsets_sph.size()  != n_shells + 1) {
        throw std::runtime_error(
            "build_cart_to_sph_U_device: offset arrays must have size n_shells+1");
    }
    const int nbf_cart = shell_offsets_cart[n_shells];
    const int nbf_sph  = shell_offsets_sph[n_shells];
    build_and_upload_U_full(shell_types, shell_offsets_cart, shell_offsets_sph,
                            nbf_cart, nbf_sph, d_U_out);
}


void transform_orbital_pair_cart_to_sph_device(
    cublasHandle_t handle,
    const real_t* d_in,
    real_t* d_out,
    real_t* d_T_work,
    const real_t* d_U_orb,
    int n_lead,
    int nc_orb,
    int ns_orb)
{
    const long long ncO = (long long)nc_orb;
    const long long nsO = (long long)ns_orb;
    const real_t alpha = 1.0;
    const real_t beta  = 0.0;

    // ============================================================
    // Stage A (μ axis): T[n_lead × ns_orb × nc_orb] from in[n_lead × nc_orb × nc_orb]
    //   Strided batched over the leading axis.  Per-slab row-major DGEMM:
    //     T_slab[ns_orb × nc_orb] = U_orb[ns_orb × nc_orb] · in_slab[nc_orb × nc_orb]
    //   (identical to Stage 2 of transform_3index_cart_to_sph_device with
    //    the aux axis replaced by n_lead).
    // ============================================================
    cublasStatus_t st = cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ncO, nsO, ncO,
        &alpha,
        d_in,    ncO, ncO * ncO,        // A = in,    lda = nc_orb, strideA = nc_orb²
        d_U_orb, ncO, 0,                // B = U_orb, ldb = nc_orb, strideB = 0 (shared)
        &beta,
        d_T_work, ncO, nsO * ncO,       // C = T,     ldc = nc_orb, strideC = ns_orb · nc_orb
        n_lead);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("transform_orbital_pair Stage A dgemm: status="
            + std::to_string((int)st));
    }

    // ============================================================
    // Stage B (ν axis): out[n_lead × ns_orb × ns_orb] = T[n_lead × ns_orb × nc_orb] · U_orb^T
    //   View T as (n_lead · ns_orb) × nc_orb and out as (n_lead · ns_orb) × ns_orb,
    //   single DGEMM with op_row_B = T on U_orb (identical to Stage 3).
    // ============================================================
    st = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        nsO, (long long)n_lead * nsO, ncO,
        &alpha,
        d_U_orb,  ncO,
        d_T_work, ncO,
        &beta,
        d_out,    nsO);
    if (st != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("transform_orbital_pair Stage B dgemm: status="
            + std::to_string((int)st));
    }
    // No device sync here: the caller's stream sequences this against the
    // subsequent B_local DGEMM; mgr.sync_all() barriers between aux batches.
}

} // namespace gansu::spherical
