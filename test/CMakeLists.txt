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

# Note: This CMakeLists.txt is designed to be used as a subdirectory from the main project.
# If building tests standalone, you'll need to build the main project first.

# Download Google Test using FetchContent
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
)
# Build settings for GTest
FetchContent_MakeAvailable(googletest)


# GPU test source files
set(GPU_TEST_SOURCES
    test_gpu_manager.cu
    test_gpu_kernels.cu
    test_davidson_solver.cu
    test_validation.cu
    test_rys_quadrature.cu
)

# CPU backend test source files (separate executable to avoid GPU state pollution)
set(CPU_TEST_SOURCES
    test_cpu_backend.cu
)

if(NOT ENABLE_GPU)
    set_source_files_properties(${GPU_TEST_SOURCES} ${CPU_TEST_SOURCES} PROPERTIES LANGUAGE CXX)
endif()

# GPU test executable
add_executable(test_runner ${GPU_TEST_SOURCES})
target_link_libraries(test_runner gansu gtest gtest_main)
if(ENABLE_GPU)
    target_link_libraries(test_runner curand)
endif()

# CPU backend test executable (no GPU state interference)
add_executable(test_cpu ${CPU_TEST_SOURCES})
target_link_libraries(test_cpu gansu gtest gtest_main)
if(ENABLE_GPU)
    target_link_libraries(test_cpu curand)
endif()

# Enable CTest framework
enable_testing()

# Fast tests (daily use, ~1 min)
add_test(NAME GANSU_GPU_Tests COMMAND test_runner
    --gtest_filter=-Benchmark*:LargeBenchmark*:*ccpVTZ*:ValidationEOMCCSD.H2O_ccpVDZ:ValidationEOMCC2.H2O_ccpVDZ:ValidationEOMMP2.H2O_STO3G:ValidationRI_PostHF.H2O_EOM_CCSD*)
add_test(NAME GANSU_CPU_Tests COMMAND test_cpu)

# Slow tests (run separately when needed)
add_test(NAME GANSU_Slow_Tests COMMAND test_runner
    --gtest_filter=*ccpVTZ*:ValidationEOMCCSD.H2O_ccpVDZ:ValidationEOMCC2.H2O_ccpVDZ:ValidationEOMMP2.H2O_STO3G:ValidationRI_PostHF.H2O_EOM_CCSD*)
set_tests_properties(GANSU_Slow_Tests PROPERTIES LABELS "slow")

# Benchmarks (run separately)
add_test(NAME GANSU_Benchmarks COMMAND test_runner --gtest_filter=Benchmark*:LargeBenchmark*)
set_tests_properties(GANSU_Benchmarks PROPERTIES LABELS "benchmark")