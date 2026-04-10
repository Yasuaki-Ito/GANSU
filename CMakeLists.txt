#
# GANSU: GPU Accelerated Numerical Simulation Utility
#
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# All rights reserved.
#
# This software is licensed under the BSD 3-Clause License.
# You may obtain a copy of the license in the LICENSE file
# located in the root directory of this source tree or at:
# https://opensource.org/licenses/BSD-3-Clause
#
# SPDX-License-Identifier: BSD-3-Clause
#
cmake_minimum_required(VERSION 3.31)

# GPU support option (must be set before project())
option(ENABLE_GPU "Build with CUDA GPU support" ON)

if(ENABLE_GPU)
    project(GANSU LANGUAGES CUDA CXX)
else()
    project(GANSU LANGUAGES CXX)
endif()


# Uncomment the following line if f- or g-orbital basis functions are present.
# add_definitions(-DENABLE_F_INT1E)
# add_definitions(-DENABLE_G_INT1E)

# Uncomment the following line to enable to support D-type basis functions for RI approximations (density fitting)
# Note: Build time will take several hours (5-6 hours)
#
# add_definitions(-DCOMPUTE_D_BASIS)

# Uncomment the following line to enable to support G-type auxiliary basis functions for RI approximations (density fitting)
# Note: Build time will take one day or more
#
# add_definitions(-DCOMPUTE_G_AUX)


if(ENABLE_GPU)
    # set compute capability
    set(CMAKE_CUDA_ARCHITECTURES 80 86 90)
    # Enable line info and debug info for CUDA when in Debug mode
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
    set(CMAKE_CUDA_STANDARD 17)
    # --- Suppress Eigen/CUDA interop warnings ---
    # Eigen headers included from .cu files (for the CPU fallback paths) trigger
    # spurious NVCC diagnostics because Eigen marks some methods __host__ __device__
    # via EIGEN_DEVICE_FUNC but the full template instantiation pulls in plain
    # __host__ helpers.  These code paths are never reached from actual CUDA
    # kernels (they only run inside `if (!gpu_available())` branches on the host),
    # so the warnings are false positives.
    #
    #   20011/20014: calling a __host__ function from a __host__ __device__ function
    #   20012      : __host__ annotation on a defaulted function (legacy suppression)
    #   20013/20015: constexpr variants of the above; Eigen suggests
    #                --expt-relaxed-constexpr to allow, which we enable below.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20011,20012,20013,20014,20015")
else()
    add_definitions(-DGANSU_CPU_ONLY)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 17)

# Set include directories (global)
include_directories(include)


# ============================================================================
# Dependencies: Eigen (header-only) and OpenMP
# ============================================================================
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

find_package(OpenMP REQUIRED)


# ============================================================================
# Source files for integral static libraries
# ============================================================================
set(INTEGRAL_SOURCES
    src/int1e.cu
    src/int2e.cu
    src/int2e_hash.cu
    src/int2c2e.cu
    src/int3c2e.cu
    src/int2e_direct.cu
    src/rys_eri.cu
    src/rys_eri_direct.cu
    src/rys_eri_mp2.cu
    src/rys_int3c2e.cu
    src/rys_int2c2e.cu
)

set(GRADIENT_SOURCES
    src/grad/grad_w.cu
    src/grad/grad_s.cu
    src/grad/grad_k.cu
    src/grad/grad_v.cu
    src/grad/grad_g.cu
    src/grad/rys_grad_g.cu
    src/grad/rys_hessian_g.cu
    src/grad/hessian_s.cu
    src/grad/hessian_k.cu
    src/grad/hessian_v.cu
    src/grad/hessian_w.cu
)

# When GPU is disabled, compile .cu files as C++
if(NOT ENABLE_GPU)
    set_source_files_properties(${INTEGRAL_SOURCES} ${GRADIENT_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

# Static libraries (integrals)
foreach(src ${INTEGRAL_SOURCES})
    get_filename_component(lib_name ${src} NAME_WE)
    add_library(${lib_name} STATIC ${src})
    target_link_libraries(${lib_name} Eigen3::Eigen OpenMP::OpenMP_CXX)
endforeach()

# Static libraries (gradients)
foreach(src ${GRADIENT_SOURCES})
    get_filename_component(lib_name ${src} NAME_WE)
    add_library(${lib_name} STATIC ${src})
    target_link_libraries(${lib_name} Eigen3::Eigen OpenMP::OpenMP_CXX)
endforeach()


# ============================================================================
# Main GANSU library
# ============================================================================
set(SOURCES
    src/basis_set.cpp
    src/parseXYZ.cpp
    src/hf.cu
    src/rhf.cu
    src/uhf.cu
    src/rohf.cu
    src/gpu_manager.cu
    src/gpu_kernels.cu
    src/profiler.cu
    src/parameter_manager.cpp
    src/eri.cu
    src/eri_ri.cu
    src/eri_ri_posthf.cu
    src/eri_stored.cu
    src/eri_stored_fci.cu
    src/eri_stored_cis.cu
    src/cis_operator.cu
    src/cis_operator_ri.cu
    src/cis_operator_jk.cu
    src/hash_half_transform.cu
    src/half_transform_mp3.cu
    src/eri_ri_hash.cu
    src/eri_stored_adc2.cu
    src/adc2_operator.cu
    src/adc2_full_operator.cu
    src/adc2x_operator.cu
    src/eri_stored_adc2x.cu
    src/eom_mp2_operator.cu
    src/eom_mp2_schur_operator.cu
    src/eri_stored_eom_mp2.cu
    src/cc2_solver.cu
    src/eom_cc2_operator.cu
    src/eri_stored_cc2.cu
    src/eri_stored_eom_cc2.cu
    src/eom_ccsd_operator.cu
    src/eri_stored_eom_ccsd.cu
    src/oscillator_strength.cu
    src/eri_stored_mp4.cu
    src/cphf_solver.cu
    src/eri_stored_ump.cu
    src/davidson_solver.cu
    src/fci.cu
    src/test_operators.cu
    src/utils.cpp
)

if(NOT ENABLE_GPU)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(gansu STATIC ${SOURCES})
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED)
target_compile_definitions(gansu PRIVATE CONSOLE_COLOR_ENABLED)
target_link_libraries(gansu
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        int1e
        int2e
        int2e_hash
        int2c2e
        int3c2e
        int2e_direct
        rys_eri
        rys_eri_direct
        rys_eri_mp2
        rys_int3c2e
        rys_int2c2e
        grad_w
        grad_s
        grad_k
        grad_v
        grad_g
        rys_grad_g
        rys_hessian_g
        hessian_s
        hessian_k
        hessian_v
        hessian_w
    )

if(ENABLE_GPU)
    target_link_libraries(gansu cublas cusolver ${CUDA_LIBRARIES})
    # Delay-load cuBLAS/cuSOLVER DLLs (Windows only) so binary runs without GPU
    if(WIN32)
        target_link_options(gansu PUBLIC /DELAYLOAD:cublas64_12.dll /DELAYLOAD:cusolver64_11.dll)
        target_link_libraries(gansu delayimp)
    endif()
endif()


# ============================================================================
# Main executable
# ============================================================================
set(MAIN_SOURCE src/HF_main.cu)
if(NOT ENABLE_GPU)
    set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
endif()

add_executable(HF_main ${MAIN_SOURCE})
target_compile_definitions(HF_main PRIVATE PROFILE_ENABLED)
target_compile_definitions(HF_main PRIVATE CONSOLE_COLOR_ENABLED)
target_link_libraries(HF_main gansu)


# ============================================================================
# Tests (optional)
# ============================================================================
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()


# GANSU-UI (Web UI) is maintained as a separate project under ui/.
# To build and run the UI, see ui/README.md.
