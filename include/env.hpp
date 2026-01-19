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
 * @file env.hpp
 * @brief Display the environment information (e.g., versions of CUDA, cuBLAS, and cuSOLVER)
 */

#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace gansu{

/**
 * @brief Display the environment information
 */
inline void display_env_info(){
    // CUDA Runtime Version
    int cuda_runtime_version;
    cudaRuntimeGetVersion(&cuda_runtime_version);
    int cuda_runtime_major = cuda_runtime_version / 1000;
    int cuda_runtime_minor = (cuda_runtime_version % 1000) / 10;
    int cuda_runtime_patch = cuda_runtime_version % 10;

    // CUDA Driver Version
    int cuda_driver_version;
    cudaDriverGetVersion(&cuda_driver_version);
    int cuda_driver_major = cuda_driver_version / 1000;
    int cuda_driver_minor = (cuda_driver_version % 1000) / 10;
    int cuda_driver_patch = cuda_driver_version % 10;

    // cuBLAS Version
    int cublas_major=-1, cublas_minor=-1, cublas_patch=-1;
    cublasGetProperty(MAJOR_VERSION, &cublas_major);
    cublasGetProperty(MINOR_VERSION, &cublas_minor);
    cublasGetProperty(PATCH_LEVEL, &cublas_patch);

    // cuSOLVER Version
    int cusolver_major=-1, cusolver_minor=-1, cusolver_patch=-1;
    cusolverGetProperty(MAJOR_VERSION, &cusolver_major);
    cusolverGetProperty(MINOR_VERSION, &cusolver_minor);
    cusolverGetProperty(PATCH_LEVEL, &cusolver_patch);

    std::cout << "CUDA Runtime Version: " << cuda_runtime_major << "." << cuda_runtime_minor << "." << cuda_runtime_patch << std::endl;
    std::cout << "CUDA Driver Version: " << cuda_driver_major << "." << cuda_driver_minor << "." << cuda_driver_patch << std::endl;
    std::cout << "cuBLAS Version: " << cublas_major << "." << cublas_minor << "." << cublas_patch << std::endl;
    std::cout << "cuSOLVER Version: " << cusolver_major << "." << cusolver_minor << "." << cusolver_patch << std::endl;

}


} // namespace gansu
