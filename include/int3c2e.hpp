/*
 * GANSU: GPU Acclerated Numerical Simulation Utility
 *
 * Copyright (c) 2025, Hiroshima University and Fujitsu Limited
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
 * @file int3c2e.hpp This file contains the functions for computing the two-center two-electron repulsion integrals.
 */


 #ifndef INT3C2E_CPP
 #define INT3C2E_CPP

 #define N_ORBITAL_TYPE_BASIS 3   // (s,1), (p,2), ...

 #ifndef N_ORBITAL_TYPE_AUX
 #define N_ORBITAL_TYPE_AUX 5
 #endif

 #include "int2c2e.hpp"
 #include "boys.hpp"
 #include "types.hpp"
 #include "utils_cuda.hpp"
 #include "parameters.h"
 #include "compile_flag.hpp"


 #include <cuda.h>
  
 namespace gansu::gpu{
  

    /* (dd|g) */
    extern __global__ void calc_ddg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|f) */
    extern __global__ void calc_ddf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|d) */
    extern __global__ void calc_ddd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|p) */
    extern __global__ void calc_ddp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (dd|s) */
    extern __global__ void calc_dds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|g) */
    extern __global__ void calc_pdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|f) */
    extern __global__ void calc_pdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|d) */
    extern __global__ void calc_pdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|p) */
    extern __global__ void calc_pdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pd|s) */
    extern __global__ void calc_pds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|g) */
    extern __global__ void calc_ppg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|f) */
    extern __global__ void calc_ppf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|d) */
    extern __global__ void calc_ppd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|p) */
    extern __global__ void calc_ppp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (pp|s) */
    extern __global__ void calc_pps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|g) */
    extern __global__ void calc_sdg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|f) */
    extern __global__ void calc_sdf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|d) */
    extern __global__ void calc_sdd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|p) */
    extern __global__ void calc_sdp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sd|s) */
    extern __global__ void calc_sds_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|g) */
    extern __global__ void calc_spg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|f) */
    extern __global__ void calc_spf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|d) */
    extern __global__ void calc_spd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|p) */
    extern __global__ void calc_spp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (sp|s) */
    extern __global__ void calc_sps_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|g) */
    extern __global__ void calc_ssg_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|f) */
    extern __global__ void calc_ssf_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|d) */
    extern __global__ void calc_ssd_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|p) */
    extern __global__ void calc_ssp_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
    /* (ss|s) */
    extern __global__ void calc_sss_gpu(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);
 
    extern __global__ void MD_int3c2e_1T1SP(real_t* g_result, const PrimitiveShell* g_pshell, const PrimitiveShell* g_pshell_aux, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, ShellTypeInfo shell_s2, int64_t num_tasks, int num_basis, const size_t2* d_primitive_shell_pair_indices, const double* g_upper_bound_factors,  const double* g_auxiliary_upper_bound_factors,  const double schwarz_screening_threshold, int num_auxiliary_basis, const double* g_boys_grid);

    using eri_3center_kernel_t = void (*)(real_t*, const PrimitiveShell*, const PrimitiveShell*, const real_t*, const real_t*, ShellTypeInfo, ShellTypeInfo, ShellTypeInfo, int64_t, int, const size_t2*, const double*, const double*, const double, int, const double*);
    

    inline eri_3center_kernel_t get_3center_kernel(int a, int b, int c){
        eri_3center_kernel_t kernels[30] = {calc_sss_gpu, calc_ssp_gpu, calc_ssd_gpu, calc_ssf_gpu, calc_ssg_gpu, 
                                            calc_sps_gpu, calc_spp_gpu, calc_spd_gpu, calc_spf_gpu, calc_spg_gpu, 
                                            calc_sds_gpu, calc_sdp_gpu, calc_sdd_gpu, calc_sdf_gpu, calc_sdg_gpu, 
                                            calc_pps_gpu, calc_ppp_gpu, calc_ppd_gpu, calc_ppf_gpu, calc_ppg_gpu, 
                                            calc_pds_gpu, calc_pdp_gpu, calc_pdd_gpu, calc_pdf_gpu, calc_pdg_gpu, 
                                            calc_dds_gpu, calc_ddp_gpu, calc_ddd_gpu, calc_ddf_gpu, calc_ddg_gpu};
    

        if (a < N_ORBITAL_TYPE_BASIS && b < N_ORBITAL_TYPE_BASIS && c < N_ORBITAL_TYPE_AUX){
#if !defined(COMPUTE_D_BASIS)
            if (a >= 2 || b >= 2){
                // printf("Caution: calling generic int3c2e kernel.\n");
                return MD_int3c2e_1T1SP;
            }
#endif
#if !defined(COMPUTE_G_AUX)
            if (c >= 4){
                // printf("Caution: calling generic int3c2e kernel.\n");
                return MD_int3c2e_1T1SP;
            }
#endif  
            return kernels[calcIdx_triangular(a, b, N_ORBITAL_TYPE_BASIS) * N_ORBITAL_TYPE_AUX + c];
        } else { 
            throw std::runtime_error("Invalid call for 3center eri.\n");
        }
    }

    
    
    __device__ __inline__ double calcNormsWOFact2_3center(double alpha, double beta, double gamma, int sum_ang1,  int sum_ang2,  int sum_ang3){
        return pow(2.0, sum_ang1+sum_ang2+sum_ang3) 
            // * pow(factorial2_gpu(2.0*i1-1.0)*factorial2_gpu(2.0*j1-1.0)*factorial2_gpu(2.0*k1-1.0)*factorial2_gpu(2.0*l1-1.0)*factorial2_gpu(2.0*m1-1.0)*factorial2_gpu(2.0*n1-1.0)*factorial2_gpu(2.0*i2-1.0)*factorial2_gpu(2.0*k2-1.0)*factorial2_gpu(2.0*m2-1.0), -0.5) 
            * pow(2.0 / M_PI, 2.25)
            * pow(alpha, (2.0*(sum_ang1)+3.0)/4.0)
            * pow(beta, (2.0*(sum_ang2)+3.0)/4.0)
            * pow(gamma, (2.0*(sum_ang3)+3.0)/4.0);
    }
    

    __device__ __inline__ void addToResult_3center(double res, double *g_result, int p, int q, int r, int nCGTO, int nAux, bool is_prim_id_not_equal, const real_t* d_cgto_nomalization_factors, const real_t* d_auxiliary_cgto_nomalization_factors){
        res *= d_cgto_nomalization_factors[p] * d_cgto_nomalization_factors[q] * d_auxiliary_cgto_nomalization_factors[r];
        //atomicAdd(&g_result[r*nCGTO*nCGTO + p*nCGTO + q], res);
        atomicAdd(&g_result[(size_t)r*nCGTO*nCGTO + p*nCGTO + q], res);
        //if(is_prim_id_not_equal) atomicAdd(&g_result[r*nCGTO*nCGTO + q*nCGTO + p], res);
        if(is_prim_id_not_equal) atomicAdd(&g_result[(size_t)r*nCGTO*nCGTO + q*nCGTO + p], res);
    }

  
 } // namespace gansu::gpu
  
  #endif