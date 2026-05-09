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
option(ENABLE_MULTI_GPU "Build with multi-GPU support (requires NCCL)" OFF)

if(ENABLE_GPU)
    project(GANSU LANGUAGES CUDA CXX)
else()
    project(GANSU LANGUAGES CXX)
endif()


# Enable PIC globally (required for shared library libgansu.so)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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
    # Only override CUDA archs if the user has not given an explicit value
    # (e.g. via -DCMAKE_CUDA_ARCHITECTURES=... or in their cache). The string
    # comparison guards against CMake 3.24+'s autodetect leaving the variable
    # set to an unhelpful default — we want our 80;86;90 to win in fresh
    # builds while still letting the wheel CI pass 80;86;89;90;100;120.
    if(NOT CMAKE_CUDA_ARCHITECTURES OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "native" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "all" OR
       CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
        set(CMAKE_CUDA_ARCHITECTURES 80 86 90 CACHE STRING "" FORCE)
    endif()
    message(STATUS "CUDA archs: ${CMAKE_CUDA_ARCHITECTURES}")
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

# BLAS (optional): Eigen uses it automatically for DGEMM etc. if available
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
    add_definitions(-DEIGEN_USE_BLAS)
else()
    message(STATUS "BLAS not found — Eigen will use built-in (slower) implementations")
endif()

find_package(OpenMP REQUIRED)
# Note: link both OpenMP::OpenMP_CXX (for .cpp) and OpenMP::OpenMP_CUDA
# (for .cu files). Without OpenMP_CUDA, NVCC won't pass -fopenmp through to
# the host compiler and #pragma omp parallel is silently ignored, yielding
# team_size=1 at runtime.


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
    target_link_libraries(${lib_name} Eigen3::Eigen OpenMP::OpenMP_CXX
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CUDA>:OpenMP::OpenMP_CUDA>)
endforeach()

# Static libraries (gradients)
foreach(src ${GRADIENT_SOURCES})
    get_filename_component(lib_name ${src} NAME_WE)
    add_library(${lib_name} STATIC ${src})
    target_link_libraries(${lib_name} Eigen3::Eigen OpenMP::OpenMP_CXX
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CUDA>:OpenMP::OpenMP_CUDA>)
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
    src/ccsd_lambda.cu
    src/ccsd_lambda_gpu.cu
    src/progress.cu
    src/oscillator_strength.cu
    src/eri_stored_mp4.cu
    src/cphf_solver.cu
    src/eri_stored_ump.cu
    src/davidson_solver.cu
    src/fci.cu
    src/test_operators.cu
    src/utils.cpp
    src/ecp_integrals.cpp
    src/ecp_integrals_gpu.cu
    src/sos_laplace_adc2_operator.cu
    src/ri_adc2_schur_operator.cu
    src/ri_adc2_schur_distributed_operator.cu
    src/sos_laplace_adc2_distributed_operator.cu
    src/multi_gpu_manager.cu
    src/eri_ri_distributed.cu
    src/dmet.cu
    src/thc_grid.cu
    src/thc_collocation.cu
    src/thc_decomposition.cu
    src/thc_mp2.cu
    src/thc_sos_adc2_operator.cu
    src/dlpno_localizer.cu
    src/dlpno_pao.cu
    src/dlpno_domain.cu
    src/dlpno_pno.cu
    src/dlpno_pair_data.cu
    src/dlpno_picache_gpu.cu
    src/dlpno_resid_gpu.cu
    src/dlpno_mp2.cu
    src/dlpno_ccsd.cu
)

if(NOT ENABLE_GPU)
    set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
endif()

add_library(gansu STATIC ${SOURCES})
set_target_properties(gansu PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED)
target_compile_definitions(gansu PRIVATE CONSOLE_COLOR_ENABLED)
target_link_libraries(gansu
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        $<$<TARGET_EXISTS:OpenMP::OpenMP_CUDA>:OpenMP::OpenMP_CUDA>
        $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>
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
    # Make the CUDA toolkit headers (cuda_runtime.h etc.) visible to plain
    # .cpp sources of this target — by default they are only on the include
    # path for .cu files. Required on minimal toolchains (manylinux/UBI8
    # Docker images) where /usr/local/cuda/include is not in the system
    # search path; harmless on distros where it already is.
    target_include_directories(gansu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(gansu cublas cusolver ${CUDA_LIBRARIES})
    # Delay-load cuBLAS/cuSOLVER DLLs (Windows only) so binary runs without GPU
    if(WIN32)
        target_link_options(gansu PUBLIC /DELAYLOAD:cublas64_12.dll /DELAYLOAD:cusolver64_11.dll)
        target_link_libraries(gansu delayimp)
    endif()
    # Multi-GPU support (NCCL)
    if(ENABLE_MULTI_GPU)
        find_library(NCCL_LIBRARY nccl HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        if(NCCL_LIBRARY)
            target_compile_definitions(gansu PUBLIC GANSU_MULTI_GPU)
            target_link_libraries(gansu ${NCCL_LIBRARY})
            message(STATUS "Multi-GPU enabled: NCCL found at ${NCCL_LIBRARY}")
        else()
            message(WARNING "ENABLE_MULTI_GPU is ON but NCCL not found. Multi-GPU disabled.")
        endif()
    endif()
endif()


# ============================================================================
# Shared library for C API / Python binding
# ============================================================================
set(API_SOURCE src/gansu_api.cu)
if(NOT ENABLE_GPU)
    set_source_files_properties(${API_SOURCE} PROPERTIES LANGUAGE CXX)
endif()

add_library(gansu_shared SHARED ${API_SOURCE})
target_compile_definitions(gansu_shared PRIVATE PROFILE_ENABLED)
target_link_libraries(gansu_shared gansu)
set_target_properties(gansu_shared PROPERTIES
    OUTPUT_NAME "gansu"
    POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Main executable
# ============================================================================
set(MAIN_SOURCE src/HF_main.cu)
if(NOT ENABLE_GPU)
    set_source_files_properties(${MAIN_SOURCE} PROPERTIES LANGUAGE CXX)
endif()

add_executable(gansu_cli ${MAIN_SOURCE})
set_target_properties(gansu_cli PROPERTIES OUTPUT_NAME "gansu")
target_compile_definitions(gansu_cli PRIVATE PROFILE_ENABLED)
target_compile_definitions(gansu_cli PRIVATE CONSOLE_COLOR_ENABLED)
target_link_libraries(gansu_cli gansu)

# Legacy name (backward compatibility)
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
