# 
# GANSU: GPU Acclerated Numerical Simulation Utility
# 
# Copyright (c) 2025, Hiroshima University and Fujitsu Limited
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


# Uncomment the following line to enable to support D-type basis functions for RI approximations (density fitting)
# Note: Build time will take several hours (5-6 hours)
#
# add_definitions(-DCOMPUTE_D_BASIS)

# Uncomment the following line to enable to support G-type auxiliary basis functions for RI approximations (density fitting)
# Note: Build time will take one day or more
# 
# add_definitions(-DCOMPUTE_G_AUX)


# set compute capability
set(CMAKE_CUDA_ARCHITECTURES 80 86 90)

# Enable line info and debug info for CUDA when in Debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Set include directories (global)
include_directories(include)


# Static libraries
add_library(int1e STATIC src/int1e.cu)
add_library(int2e STATIC src/int2e.cu)
add_library(int2c2e STATIC src/int2c2e.cu)
add_library(int3c2e STATIC src/int3c2e.cu)
add_library(int2e_direct STATIC src/int2e_direct.cu)

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
    src/utils.cpp
)

add_library(gansu STATIC ${SOURCES})
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED) # add option -DPROFILE_ENABLED 
target_compile_definitions(gansu PRIVATE CONSOLE_COLOR_ENABLED) # add option -DCONSOLE_COLOR_ENABLED 
target_link_libraries(gansu
        cublas
        cusolver
        ${CUDA_LIBRARIES}
        int1e
        int2e
        int2c2e
        int3c2e
        int2e_direct
    )


add_executable(HF_main src/HF_main.cu)
target_compile_definitions(HF_main PRIVATE PROFILE_ENABLED) # add option -DPROFILE_ENABLED 
target_compile_definitions(HF_main PRIVATE CONSOLE_COLOR_ENABLED) # add option -DCONSOLE_COLOR_ENABLED 
target_link_libraries(HF_main
        gansu
    )

# add_executable(demo src/demo.cu ${SOURCES})
# target_link_libraries(demo
#         gansu
#     )


# add_executable(HF_main_no_profile src/HF_main.cu ${SOURCES})
# target_link_libraries(HF_main_no_profile
#         cublas
#         cusolver
#         ${CUDA_LIBRARIES}
#         int1e
#         int2e
#     )