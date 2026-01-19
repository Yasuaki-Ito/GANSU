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
 * @file int2c2e.hpp This file contains the functions for computing the two-center two-electron repulsion integrals.
 */


#ifndef INT2C2E_CPP
#define INT2C2E_CPP

#define N_ORBITAL_TYPE_AUX 5

#include <stdexcept>

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"
#include "parameters.h"
#include "compile_flag.hpp"
#include <cuda.h>
 
namespace gansu::gpu{


    // 順序合わせたやつ
__constant__ int loop_to_ang_RI[7][28][3] = {
    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{3,0,0}, {0,3,0}, {0,0,3}, {2,1,0}, {2,0,1}, {0,2,1}, {1,2,0}, {0,1,2}, {1,0,2}, {1,1,1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3,1,0}, {3,0,1}, {0,3,1}, {1,3,0}, {0,1,3}, {1,0,3}, {2,2,0}, {2,0,2}, {0,2,2}, {2,1,1}, {1,2,1}, {1,1,2}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {2, 3, 0}, {1, 4, 0}, {0, 5, 0}, {4, 0, 1}, {3, 1, 1}, {2, 2, 1}, {1, 3, 1}, {0, 4, 1}, {3, 0, 2}, {2, 1, 2}, {1, 2, 2}, {0, 3, 2}, {2, 0, 3}, {1, 1, 3}, {0, 2, 3}, {1, 0, 4}, {0, 1, 4}, {0, 0, 5}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{6, 0, 0}, {5, 1, 0}, {4, 2, 0}, {3, 3, 0}, {2, 4, 0}, {1, 5, 0}, {0, 6, 0}, {5, 0, 1}, {4, 1, 1}, {3, 2, 1}, {2, 3, 1}, {1, 4, 1}, {0, 5, 1}, {4, 0, 2}, {3, 1, 2}, {2, 2, 2}, {1, 3, 2}, {0, 4, 2}, {3, 0, 3}, {2, 1, 3}, {1, 2, 3}, {0, 3, 3}, {2, 0, 4}, {1, 1, 4}, {0, 2, 4}, {1, 0, 5}, {0, 1, 5}, {0, 0, 6}}
};

 
    extern __global__ void calc_ss_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_sp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_sd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_sf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_sg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_pp_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_pd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_pf_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_pg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_dd_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_df_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_dg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_ff_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_fg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void calc_gg_gpu(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    extern __global__ void MD_int2c2e_1T1SP(real_t* g_result, const PrimitiveShell* g_pshell_aux, const real_t* d_auxiliary_cgto_normalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, int num_shell_pairs, const double* g_upper_bound_factors, const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);


    inline int calcIdx_triangular(int a, int b, int N){
        return (int)(a*N - (a*(a-1))/2) + (b-a);
    }

    using eri_2center_kernel_t = void (*)(real_t*, const PrimitiveShell*, const real_t*, ShellTypeInfo, ShellTypeInfo, int, const double*, const double, int, const double*);
    inline eri_2center_kernel_t get_2center_kernel(int a, int b){
        eri_2center_kernel_t kernels[15] = {calc_ss_gpu, calc_sp_gpu, calc_sd_gpu, calc_sf_gpu, calc_sg_gpu, 
                                                         calc_pp_gpu, calc_pd_gpu, calc_pf_gpu, calc_pg_gpu, 
                                                                      calc_dd_gpu, calc_df_gpu, calc_dg_gpu, 
                                                                                   calc_ff_gpu, calc_fg_gpu, 
                                                                                                calc_gg_gpu};
                                                                                        
        if (a < N_ORBITAL_TYPE_AUX && b < N_ORBITAL_TYPE_AUX){
#if !defined(COMPUTE_G_AUX)
            if (a >= 4 || b >= 4){
                // printf("Caution: calling generic int2c2e kernel.\n");
                return MD_int2c2e_1T1SP;
            }
#endif
            return kernels[calcIdx_triangular(a,b,N_ORBITAL_TYPE_AUX)];
        } else { 
            throw std::runtime_error("Invalid call for 2center eri.\n");
        }
    }



    __device__ __inline__ double calcNormsWOFact2_2center(double alpha, double gamma, int ang1, int ang2){
        return pow(2.0, ang1+ang2) 
            * pow(2.0 / M_PI, 1.5)
            * pow(alpha, (2.0*ang1+3.0)/4.0)
            * pow(gamma, (2.0*ang2+3.0)/4.0);
    }
    
    

    __device__ __inline__ void addToResult_2center(double res, double *g_result, int p, int r, int nAux, bool is_prim_id_not_equal, const real_t* d_auxiliary_cgto_normalization_factors){
        res *= d_auxiliary_cgto_normalization_factors[p] * d_auxiliary_cgto_normalization_factors[r];
        
        atomicAdd(&g_result[p*nAux + r], res);
        if(is_prim_id_not_equal) atomicAdd(&g_result[r*nAux + p], res);
    }
 
} // namespace gansu::gpu
 
 #endif