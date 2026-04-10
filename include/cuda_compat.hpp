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

/**
 * @file cuda_compat.hpp
 * @brief CPU-only stubs for CUDA runtime, cuBLAS, and cuSOLVER APIs.
 *
 * When GANSU_CPU_ONLY is defined, this header replaces <cuda_runtime.h>,
 * <cublas_v2.h>, and <cusolverDn.h> with minimal stubs so that .cu files
 * can be compiled as plain C++ without any CUDA dependency.
 */

#pragma once

#ifndef GANSU_CPU_ONLY
#error "cuda_compat.hpp must only be included when GANSU_CPU_ONLY is defined"
#endif

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>

// ============================================================================
// CUDA Qualifiers — expand to nothing on CPU
// ============================================================================
#define __global__
#define __device__
#define __host__
#define __shared__
#define __constant__
#define __forceinline__ inline
#define __restrict__

// ============================================================================
// CUDA Types
// ============================================================================

// Error type
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInvalidValue = 11,
};

// Memory copy direction
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

// Stream (opaque pointer, unused on CPU)
using cudaStream_t = void*;

// Event (opaque pointer, unused on CPU)
using cudaEvent_t = void*;

// Stream flags
constexpr unsigned int cudaStreamNonBlocking = 0x01;

// Dim3 for kernel launch dimensions
struct dim3 {
    unsigned int x, y, z;
    dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};

// CUDA vector types
struct int2 {
    int x, y;
};

struct int3 {
    int x, y, z;
};

struct int4 {
    int x, y, z, w;
};

struct double2 {
    double x, y;
};

struct double3 {
    double x, y, z;
};

struct double4 {
    double x, y, z, w;
};

inline int2 make_int2(int x, int y) { return {x, y}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
inline double2 make_double2(double x, double y) { return {x, y}; }
inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }
inline double4 make_double4(double x, double y, double z, double w) { return {x, y, z, w}; }


// ============================================================================
// CUDA Memory Functions — redirect to standard C
// ============================================================================

inline cudaError_t cudaMalloc(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    return (*ptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMallocHost(void** ptr, size_t size) {
    *ptr = std::malloc(size);
    return (*ptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFreeHost(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind /*kind*/) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind /*kind*/, cudaStream_t /*stream*/ = nullptr) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemset(void* ptr, int value, size_t count) {
    std::memset(ptr, value, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* ptr, int value, size_t count, cudaStream_t /*stream*/ = nullptr) {
    std::memset(ptr, value, count);
    return cudaSuccess;
}

// Async alloc/free — same as sync on CPU
inline cudaError_t cudaMallocAsync(void** ptr, size_t size, cudaStream_t /*stream*/) {
    *ptr = std::malloc(size);
    return (*ptr) ? cudaSuccess : cudaErrorMemoryAllocation;
}

inline cudaError_t cudaFreeAsync(void* ptr, cudaStream_t /*stream*/) {
    std::free(ptr);
    return cudaSuccess;
}


// ============================================================================
// CUDA Stream Functions — no-op on CPU
// ============================================================================

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    *stream = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int /*flags*/) {
    *stream = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t /*stream*/) {
    return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t /*stream*/) {
    return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

// ============================================================================
// CUDA Event Functions — no-op on CPU
// ============================================================================

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = nullptr;
    return cudaSuccess;
}

inline cudaError_t cudaEventDestroy(cudaEvent_t /*event*/) {
    return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t /*event*/, cudaStream_t /*stream*/ = nullptr) {
    return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t /*event*/) {
    return cudaSuccess;
}

inline cudaError_t cudaStreamWaitEvent(cudaStream_t /*stream*/, cudaEvent_t /*event*/, unsigned int /*flags*/ = 0) {
    return cudaSuccess;
}


// ============================================================================
// CUDA Device / Version Query — stubs
// ============================================================================

struct cudaDeviceProp {
    char name[256] = "CPU (no GPU)";
    size_t totalGlobalMem = 0;
    int major = 0;
    int minor = 0;
    int multiProcessorCount = 0;
};

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int /*device*/) {
    *prop = cudaDeviceProp{};
    return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int /*device*/) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }

inline cudaError_t cudaRuntimeGetVersion(int* version) { *version = 0; return cudaSuccess; }
inline cudaError_t cudaDriverGetVersion(int* version) { *version = 0; return cudaSuccess; }

inline const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorInvalidValue: return "invalid argument";
        default: return "unknown error";
    }
}

inline cudaError_t cudaGetLastError() { return cudaSuccess; }


// ============================================================================
// CUDA Device Intrinsics — standard C++ equivalents
// ============================================================================

inline double __dsqrt_rn(double x) { return std::sqrt(x); }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __dadd_rn(double a, double b) { return a + b; }
inline double __fma_rn(double a, double b, double c) { return std::fma(a, b, c); }
inline long long __double2ll_rd(double x) { return static_cast<long long>(std::floor(x)); }
inline double __ll2double_rn(long long x) { return static_cast<double>(x); }

// Warp-level primitives — single-threaded on CPU
inline double __shfl_down_sync(unsigned int /*mask*/, double val, int /*delta*/, int /*width*/ = 32) { return val; }
inline unsigned int __ballot_sync(unsigned int /*mask*/, int predicate) { return predicate ? 1u : 0u; }
inline int __popc(int x) {
    // Kernighan bit count
    int count = 0;
    unsigned int ux = static_cast<unsigned int>(x);
    while (ux) { ux &= (ux - 1); count++; }
    return count;
}

// Synchronization — no-op on CPU (single-threaded kernel execution)
inline void __syncthreads() {}

// Atomic operations — when a CUDA-style kernel is launched as an OpenMP
// parallel loop (see cpu_kernel_launch.hpp), multiple host threads can hit
// the same `atomicAdd` simultaneously, so we MUST honour atomicity here or
// gradient accumulation races will silently corrupt results.
template <typename T>
inline T atomicAdd(T* address, T val) {
    T old;
    #pragma omp atomic capture
    { old = *address; *address += val; }
    return old;
}

template <typename T>
inline T atomicCAS(T* address, T compare, T val) {
    T old = *address;
    if (old == compare) *address = val;
    return old;
}

// Thread/block indexing — used by CUDA kernels, stubbed for CPU.
// On CPU, __global__ kernels are invoked via the GANSU_CPU_LAUNCH_* macros
// (see cpu_kernel_launch.hpp) which iterate a plain OpenMP loop and assign
// these thread_local stubs on each iteration, mimicking a CUDA grid.
// They are shared across translation units via `inline` linkage so the
// launcher macro (in any .cu) and the kernel body (in another .cu) refer to
// the same thread-local instance per OpenMP thread.
struct _cuda_dim3_stub {
    unsigned int x = 0, y = 0, z = 0;
};
inline thread_local _cuda_dim3_stub blockIdx;
inline thread_local _cuda_dim3_stub blockDim;
inline thread_local _cuda_dim3_stub threadIdx;
inline thread_local _cuda_dim3_stub gridDim;


// ============================================================================
// cuBLAS Stubs — types and handles only (functions are in gpu_manager.cu)
// ============================================================================

using cublasHandle_t = void*;

enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
};

enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
};

enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2,
};

enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
};

enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1,
};

// Library property enum (used by cublasGetProperty / cusolverGetProperty)
enum libraryPropertyType {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
};

// cuBLAS handle management — no-op
inline cublasStatus_t cublasCreate(cublasHandle_t* handle) { *handle = nullptr; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t /*handle*/) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t /*handle*/, cudaStream_t /*stream*/) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasGetProperty(libraryPropertyType /*type*/, int* value) { *value = 0; return CUBLAS_STATUS_SUCCESS; }


// ============================================================================
// cuSOLVER Stubs — types and handles only (functions are in gpu_manager.cu)
// ============================================================================

using cusolverDnHandle_t = void*;
using cusolverDnParams_t = void*;

enum cusolverStatus_t {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
};

enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
};

// cuSOLVER handle management — no-op
inline cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) { *handle = nullptr; return CUSOLVER_STATUS_SUCCESS; }
inline cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t /*handle*/) { return CUSOLVER_STATUS_SUCCESS; }
inline cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) { *params = nullptr; return CUSOLVER_STATUS_SUCCESS; }
inline cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t /*params*/) { return CUSOLVER_STATUS_SUCCESS; }
inline cusolverStatus_t cusolverGetProperty(libraryPropertyType /*type*/, int* value) { *value = 0; return CUSOLVER_STATUS_SUCCESS; }


// ============================================================================
// cuRAND Stubs (used in tests only)
// ============================================================================

using curandGenerator_t = void*;
enum curandRngType { CURAND_RNG_PSEUDO_DEFAULT = 100 };
enum curandStatus_t { CURAND_STATUS_SUCCESS = 0 };

inline curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType /*rng_type*/) { *gen = nullptr; return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandDestroyGenerator(curandGenerator_t /*gen*/) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t /*gen*/, unsigned long long /*seed*/) { return CURAND_STATUS_SUCCESS; }
inline curandStatus_t curandGenerateNormalDouble(curandGenerator_t /*gen*/, double* /*outputPtr*/, size_t /*n*/, double /*mean*/, double /*stddev*/) { return CURAND_STATUS_SUCCESS; }
