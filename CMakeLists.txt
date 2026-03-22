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


add_library(grad_w STATIC src/grad/grad_w.cu)
add_library(grad_s STATIC src/grad/grad_s.cu)
add_library(grad_k STATIC src/grad/grad_k.cu)
add_library(grad_v STATIC src/grad/grad_v.cu)
add_library(grad_g STATIC src/grad/grad_g.cu)


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
    src/eri_stored.cu
    src/eri_stored_fci.cu
    src/eri_stored_cis.cu
    src/cis_operator.cu
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
    src/eri_stored_ump.cu
    src/davidson_solver.cu
    src/fci.cu
    src/test_operators.cu
    src/utils.cpp
)

add_library(gansu STATIC ${SOURCES})
target_compile_definitions(gansu PRIVATE PROFILE_ENABLED) # add option -DPROFILE_ENABLED
target_compile_definitions(gansu PRIVATE CONSOLE_COLOR_ENABLED) # add option -DCONSOLE_COLOR_ENABLED
find_package(LAPACK REQUIRED)

target_link_libraries(gansu
        cublas
        cusolver
        ${LAPACK_LIBRARIES}
        ${CUDA_LIBRARIES}
        int1e
        int2e
        int2c2e
        int3c2e
        int2e_direct
        grad_w
        grad_s
        grad_k
        grad_v
        grad_g
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

# Add test directory (optional, can be enabled/disabled)
option(BUILD_TESTS "Build the test suite" OFF)
if(BUILD_TESTS)
    add_subdirectory(test)
endif()


# === GANSU-UI (Web UI) ===
set(UI_SRC_DIR "${CMAKE_SOURCE_DIR}/ui")
set(UI_BUILD_DIR "${CMAKE_BINARY_DIR}/gansu-ui")
set(NODE_VERSION "v22.14.0")

# Find npm; if not found, download Node.js to build/_node/
find_program(NPM_EXECUTABLE npm)
if(NOT NPM_EXECUTABLE)
    set(NODE_DIR "${CMAKE_BINARY_DIR}/_node")
    if(NOT EXISTS "${NODE_DIR}/bin/npm")
        message(STATUS "npm not found. Downloading Node.js ${NODE_VERSION}...")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(NODE_ARCH "arm64")
        else()
            set(NODE_ARCH "x64")
        endif()
        set(NODE_TAR "node-${NODE_VERSION}-linux-${NODE_ARCH}.tar.xz")
        file(DOWNLOAD
            "https://nodejs.org/dist/${NODE_VERSION}/${NODE_TAR}"
            "${CMAKE_BINARY_DIR}/${NODE_TAR}"
            SHOW_PROGRESS
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xf "${CMAKE_BINARY_DIR}/${NODE_TAR}"
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
        file(RENAME "${CMAKE_BINARY_DIR}/node-${NODE_VERSION}-linux-${NODE_ARCH}" "${NODE_DIR}")
        file(REMOVE "${CMAKE_BINARY_DIR}/${NODE_TAR}")
    endif()
    set(NPM_EXECUTABLE "${NODE_DIR}/bin/npm")
    set(NODE_PATH_PREFIX "${NODE_DIR}/bin:")
endif()

# Frontend build (only when source changes)
file(GLOB_RECURSE UI_FRONTEND_SOURCES "${UI_SRC_DIR}/frontend/src/*.ts")
add_custom_command(
    OUTPUT "${UI_BUILD_DIR}/frontend/dist/index.html"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${UI_SRC_DIR}/frontend" "${UI_BUILD_DIR}/_frontend_build"
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${NODE_PATH_PREFIX}$ENV{PATH}"
        "${NPM_EXECUTABLE}" install --prefix "${UI_BUILD_DIR}/_frontend_build"
    COMMAND ${CMAKE_COMMAND} -E env "PATH=${NODE_PATH_PREFIX}$ENV{PATH}"
        "${NPM_EXECUTABLE}" run build --prefix "${UI_BUILD_DIR}/_frontend_build"
        -- --outDir "${UI_BUILD_DIR}/frontend/dist"
    DEPENDS ${UI_FRONTEND_SOURCES} "${UI_SRC_DIR}/frontend/index.html"
    COMMENT "Building GANSU-UI frontend"
)

# Backend copy + frontend build
add_custom_target(gansu-ui ALL
    DEPENDS "${UI_BUILD_DIR}/frontend/dist/index.html"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${UI_BUILD_DIR}/backend"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${UI_SRC_DIR}/backend/main.py"
        "${UI_SRC_DIR}/backend/runner.py"
        "${UI_SRC_DIR}/backend/parser.py"
        "${UI_SRC_DIR}/backend/requirements.txt"
        "${UI_BUILD_DIR}/backend/"
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_SRC_DIR}/run.sh" "${UI_BUILD_DIR}/"
    COMMENT "Setting up GANSU-UI"
)

# Generate .env with correct paths (only if not already present)
if(NOT EXISTS "${UI_BUILD_DIR}/.env")
    file(WRITE "${UI_BUILD_DIR}/.env"
        "GANSU_PATH=${CMAKE_SOURCE_DIR}\n"
        "HF_MAIN_PATH=${CMAKE_BINARY_DIR}/HF_main\n"
    )
endif()
