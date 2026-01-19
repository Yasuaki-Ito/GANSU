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
project(GANSU LANGUAGES CUDA CXX)

# Download Google Test using FetchContent
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# Build settings for GTest
FetchContent_MakeAvailable(googletest)


# set compute capability
set(CMAKE_CUDA_ARCHITECTURES 80 86)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set include directories (global)
include_directories(../include)


# Static libraries
add_library(int1e STATIC ../src/int1e.cu)
add_library(int2e STATIC ../src/int2e.cu)


set(SOURCES
    ../src/basis_set.cpp
    ../src/parseXYZ.cpp
    ../src/hf.cu
    ../src/rhf.cu
    ../src/uhf.cu
    ../src/rohf.cu
    ../src/gpu_manager.cu
    ../src/gpu_kernels.cu
    ../src/profiler.cu
    ../src/parameter_manager.cpp
    ../src/eri.cu
)

add_library(gansu STATIC ${SOURCES})
target_link_libraries(gansu
        cublas
        cusolver
        ${CUDA_LIBRARIES}
        int1e
        int2e
    )


set(TEST_SOURCES
    test_gpu_manager.cu
    test_gpu_kernels.cu
)

add_executable(test_runner ${TEST_SOURCES})
target_link_libraries(test_runner
        gansu
        gtest
        gtest_main
    )


# Enable CTest framework
enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)