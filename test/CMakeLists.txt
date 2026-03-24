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


# Test source files
set(TEST_SOURCES
    test_gpu_manager.cu
    test_gpu_kernels.cu
    test_davidson_solver.cu
    test_validation.cu
    test_rys_quadrature.cu
)

# Create test executable
add_executable(test_runner ${TEST_SOURCES})

# Link against parent project's gansu library and Google Test
target_link_libraries(test_runner
    gansu          # Uses the gansu library from parent project
    curand         # Required by Davidson solver
    gtest
    gtest_main
)

# Enable CTest framework
enable_testing()
add_test(NAME GANSU_Tests COMMAND test_runner)