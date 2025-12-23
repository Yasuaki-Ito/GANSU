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


#ifndef INT2E_CUH
#define INT2E_CUH

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

#ifndef M_PI_2_5
    #define M_PI_2_5 17.49341832762486284622				
#endif

#ifndef PI3_4
    #define PI3_4 0.71270547035499016035339845
#endif

namespace gansu::gpu{

__global__ void ssss2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sssp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sspp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void spsp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void sppp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void pppp2e(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);
__global__ void MD_1T1SP(double* g_int2e, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const real_t schwarz_screening_threshold, const double* g_upper_bound_factors, const int num_basis, const double* g_boys_grid, const size_t head_bra, const size_t head_ket);

__global__ void get_schwarz_upper_bound_factors_ss(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_sp(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_pp(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_general(const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors);
__global__ void get_schwarz_upper_bound_factors_aux_general(const PrimitiveShell* g_shell_aux, const real_t* g_aux_cgto_normalization_factors, const ShellTypeInfo shell_s0, const size_t head, const size_t num_bra, const double* g_boys_grid, double* g_max_upper_bound_factors_aux);


// define the kernel functions as function pointers for two electron integrals
using eri_kernel_t = void (*)(double*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const real_t, const double*, const int, const double*, const size_t, const size_t);
using schwarz_kernel_t = void (*)(const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const size_t, const size_t, const double*, double*);
using schwarz_aux_kernel_t = void (*)(const PrimitiveShell*, const real_t*, const ShellTypeInfo, const size_t, const size_t, const double*, double*);

// ここを対角，非対角の分岐にするぐらい？
inline eri_kernel_t get_eri_kernel(int a, int b, int c, int d){
    if(a>b) std::swap(a, b);
    if(c>d) std::swap(c, d);
    if(a>c || (a==c && b>d)){
        std::swap(a, c);
        std::swap(b, d);
    }
    
    //if(a==c && b==d) return UTM_1T1SP;
    //else return RCT_1T1SP;

    // if(a==0 && b==0 && c==0 && d==0)      return ssss2e;
    // else if(a==0 && b==0 && c==0 && d==1) return sssp2e;
    // else if(a==0 && b==0 && c==1 && d==1) return sspp2e;
    // else if(a==0 && b==1 && c==0 && d==1) return spsp2e;
    // else if(a==0 && b==1 && c==1 && d==1) return sppp2e;
    // else if(a==1 && b==1 && c==1 && d==1) return pppp2e;
    // else throw std::runtime_error("Invalid shell type");

    // Hybrid
    if (a == 0 && b == 0 && c == 0 && d == 0)      return ssss2e;
    else if (a == 0 && b == 0 && c == 0 && d == 1) return sssp2e;
    else if (a == 0 && b == 0 && c == 1 && d == 1) return sspp2e;
    else if (a == 0 && b == 1 && c == 0 && d == 1) return spsp2e;
    else if (a == 0 && b == 1 && c == 1 && d == 1) return sppp2e;
    else if (a == 1 && b == 1 && c == 1 && d == 1) return pppp2e;
    else return MD_1T1SP;
}

inline schwarz_kernel_t get_schwarz_kernel(int a, int b)
{
    if(a > b) std::swap(a, b);
    if (a == 0 && b == 0)       return get_schwarz_upper_bound_factors_ss;
    else if (a == 0 && b == 1)  return get_schwarz_upper_bound_factors_sp;
    else if (a == 1 && b == 1)  return get_schwarz_upper_bound_factors_pp;
    else                        return get_schwarz_upper_bound_factors_general;
}


inline schwarz_aux_kernel_t get_schwarz_aux_kernel(int a)
{
    return get_schwarz_upper_bound_factors_aux_general;
}



inline __device__
int factorial2_gpu(int n)
{
    return (n <= 1) ?  1 : n*factorial2_gpu(n-2);
}

inline __device__
long long int calcIdx4Dim(size_t i, size_t j, size_t k, size_t l, size_t nao)
//long long int calcIdx4Dim(int i, int j, int k, int l, int nao)
{
    return nao * nao * nao * i + nao * nao * j + nao * k + l;
}

inline __device__
double calcNorm(double exp, int l, int m, int n){
    return static_cast<double>(1 << (l+m+n)) / static_cast<double>(sqrt(static_cast<double>(factorial2_gpu(2.0*l-1.0) * factorial2_gpu(2.0*m-1.0) * factorial2_gpu(2.0*n-1.0)))) * PI3_4 * pow(exp, (2.0*(l+m+n)+3.0)/4.0);
}

inline __device__
double calcNorms(double alpha, double beta, double gamma, double delta, int i1, int k1, int m1, int j1, int l1, int n1, int i2, int k2, int m2, int j2, int l2, int n2)
{
    return pow(2.0, i1+i2+j1+j2+k1+k2+l1+l2+m1+m2+n1+n2) 
           * pow(factorial2_gpu(2.0*i1-1.0)*factorial2_gpu(2.0*j1-1.0)*factorial2_gpu(2.0*k1-1.0)*factorial2_gpu(2.0*l1-1.0)*factorial2_gpu(2.0*m1-1.0)*factorial2_gpu(2.0*n1-1.0)*factorial2_gpu(2.0*i2-1.0)*factorial2_gpu(2.0*j2-1.0)*factorial2_gpu(2.0*k2-1.0)*factorial2_gpu(2.0*l2-1.0)*factorial2_gpu(2.0*m2-1.0)*factorial2_gpu(2.0*n2-1.0), -0.5) 
           * 8.0 / (M_PI*M_PI*M_PI)
           * pow(alpha, (2.0*(i1+k1+m1)+3.0)/4.0)
           * pow(beta, (2.0*(j1+l1+n1)+3.0)/4.0)
           * pow(gamma, (2.0*(i2+k2+m2)+3.0)/4.0)
           * pow(delta, (2.0*(j2+l2+n2)+3.0)/4.0);
}




/* case1. 全てひっくり返すか判定([ss|ss], [pp|pp])　*/
inline __device__
void addToResult_case1(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case2. bra側のみ判定 ([ss|sp]) */
inline __device__
void addToResult_case2(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case3. ket側のみ判定 ([sp|pp]) */
inline __device__
void addToResult_case3(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_ket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case4. bra側ket側，それぞれ判定 ([ss|pp]) */
inline __device__
void addToResult_case4(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket)
{    
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    }
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    }
    if(!sym_bra && !sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case5. bra-ketを判定 ([sp|sp]) */
inline __device__
void addToResult_case5(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_braket)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}

/* case6. 判定しない ([sp|sd]) */
inline __device__
void addToResult_case6(double res, double *g_G, int p, int q, int r, int s, int nao)
{
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
}


/* case1. 全てひっくり返すか判定([ss|ss], [pp|pp])　*/
inline __device__
void addToResult_case1(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case2. bra側のみ判定 ([ss|sp]) */
inline __device__
void addToResult_case2(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case3. ket側のみ判定 ([sp|pp]) */
inline __device__
void addToResult_case3(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_ket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case4. bra側ket側，それぞれ判定 ([ss|pp]) */
inline __device__
void addToResult_case4(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, const double* g_cgto_normalization_factors)
{    
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    if(!sym_bra) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    }
    if(!sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    }
    if(!sym_bra && !sym_ket) {
        atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}


/* case5. bra-ketを判定 ([sp|sp]) */
inline __device__
void addToResult_case5(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}

/* case6. 判定しない ([sp|sd]) */
inline __device__
void addToResult_case6(double res, double *g_G, int p, int q, int r, int s, int nao, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];
    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
    atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
}



inline __device__
void addToResult(double res, double *g_G, int p, int q, int r, int s, int nao, bool sym_bra, bool sym_ket, bool sym_braket, const double* g_cgto_normalization_factors)
{
    res *= g_cgto_normalization_factors[p] * g_cgto_normalization_factors[q] * g_cgto_normalization_factors[r] * g_cgto_normalization_factors[s];

    atomicAdd(&g_G[calcIdx4Dim(p, q, r, s, nao)], res);
    if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(q, p, r, s, nao)], res);
    if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(p, q, s, r, nao)], res);
    if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(q, p, s, r, nao)], res);
    if(!sym_braket) {
        atomicAdd(&g_G[calcIdx4Dim(r, s, p, q, nao)], res);
        if(!sym_bra) atomicAdd(&g_G[calcIdx4Dim(r, s, q, p, nao)], res);
        if(!sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, p, q, nao)], res);
        if(!sym_bra && !sym_ket) atomicAdd(&g_G[calcIdx4Dim(s, r, q, p, nao)], res);
    }
}






// BatchアルゴリズムにおけるR_midへのindexを計算
inline __device__
int calc_Idx_Rmid(int k, int u, int v, int i, int comb_max, int size_mid){
    return ((k+3)%3)*size_mid + (i)*comb_max + v*(k+2) - v*(v+1)/2 + (u);
}

inline __device__
int comb_max(int k){
    return ((k+2)*(k+1))/2;
}

// ループ変数(2)⇒角運動量(0,0,1)への置換 総和6=i軌道まで対応可能
//順に1, 3, 6, 10, 15, 21, 28要素
// __constant__ int loop_to_ang[7][28][3] = {
//     {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{2, 0, 0}, {1, 1, 0}, {0, 2, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 2}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{3, 0, 0}, {2, 1, 0}, {1, 2, 0}, {0, 3, 0}, {2, 0, 1}, {1, 1, 1}, {0, 2, 1}, {1, 0, 2}, {0, 1, 2}, {0, 0, 3}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{4, 0, 0}, {3, 1, 0}, {2, 2, 0}, {1, 3, 0}, {0, 4, 0}, {3, 0, 1}, {2, 1, 1}, {1, 2, 1}, {0, 3, 1}, {2, 0, 2}, {1, 1, 2}, {0, 2, 2}, {1, 0, 3}, {0, 1, 3}, {0, 0, 4}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {2, 3, 0}, {1, 4, 0}, {0, 5, 0}, {4, 0, 1}, {3, 1, 1}, {2, 2, 1}, {1, 3, 1}, {0, 4, 1}, {3, 0, 2}, {2, 1, 2}, {1, 2, 2}, {0, 3, 2}, {2, 0, 3}, {1, 1, 3}, {0, 2, 3}, {1, 0, 4}, {0, 1, 4}, {0, 0, 5}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
//     {{6, 0, 0}, {5, 1, 0}, {4, 2, 0}, {3, 3, 0}, {2, 4, 0}, {1, 5, 0}, {0, 6, 0}, {5, 0, 1}, {4, 1, 1}, {3, 2, 1}, {2, 3, 1}, {1, 4, 1}, {0, 5, 1}, {4, 0, 2}, {3, 1, 2}, {2, 2, 2}, {1, 3, 2}, {0, 4, 2}, {3, 0, 3}, {2, 1, 3}, {1, 2, 3}, {0, 3, 3}, {2, 0, 4}, {1, 1, 4}, {0, 2, 4}, {1, 0, 5}, {0, 1, 5}, {0, 0, 6}}
// };

// 順序合わせたやつ
__constant__ int loop_to_ang[7][28][3] = {
    {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{3, 0, 0}, {0, 3, 0}, {0, 0, 3}, {1, 2, 0}, {2, 1, 0}, {2, 0, 1}, {1, 0, 2}, {0, 1, 2}, {0, 2, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{4, 0, 0}, {0, 4, 0}, {0, 0, 4}, {3, 1, 0}, {3, 0, 1}, {1, 3, 0}, {0, 3, 1}, {1, 0, 3}, {0, 1, 3}, {2, 2, 0}, {2, 0, 2}, {0, 2, 2}, {2, 1, 1}, {1, 2, 1}, {1, 1, 2}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{5, 0, 0}, {4, 1, 0}, {3, 2, 0}, {2, 3, 0}, {1, 4, 0}, {0, 5, 0}, {4, 0, 1}, {3, 1, 1}, {2, 2, 1}, {1, 3, 1}, {0, 4, 1}, {3, 0, 2}, {2, 1, 2}, {1, 2, 2}, {0, 3, 2}, {2, 0, 3}, {1, 1, 3}, {0, 2, 3}, {1, 0, 4}, {0, 1, 4}, {0, 0, 5}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
    {{6, 0, 0}, {5, 1, 0}, {4, 2, 0}, {3, 3, 0}, {2, 4, 0}, {1, 5, 0}, {0, 6, 0}, {5, 0, 1}, {4, 1, 1}, {3, 2, 1}, {2, 3, 1}, {1, 4, 1}, {0, 5, 1}, {4, 0, 2}, {3, 1, 2}, {2, 2, 2}, {1, 3, 2}, {0, 4, 2}, {3, 0, 3}, {2, 1, 3}, {1, 2, 3}, {0, 3, 3}, {2, 0, 4}, {1, 1, 4}, {0, 2, 4}, {1, 0, 5}, {0, 1, 5}, {0, 0, 6}}
};








//msbを取得
inline __device__ 
int get_msb(int n)
{
  int i = 0;
 
  while (n > 0) {
    n >>= 1;
    i++;
  }
 
  /* return -1 if a == 0. */
  return i - 1;
}

//double型変数の"整数乗"を計算 powより多分速い
inline __device__ 
double right2left_binary_woif(double a, int n)
{
  double x = 1;
  int i = get_msb(n);
  int j = 0;
 
  for (j = 0; j <= i; j++) {
    x *=(((n>>j)&1)*(a-1) + 1);
    a *= a;
  }

  return x;
}


//*
__constant__ int tuv_list[2925][3] =
    {{0, 0, 0},
    {0, 0, 1}, {0, 1, 0}, {1, 0, 0},
    {0, 0, 2}, {0, 1, 1}, {0, 2, 0}, {1, 0, 1}, {1, 1, 0}, {2, 0, 0},
    {0, 0, 3}, {0, 1, 2}, {0, 2, 1}, {0, 3, 0}, {1, 0, 2}, {1, 1, 1}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}, {3, 0, 0},
    {0, 0, 4}, {0, 1, 3}, {0, 2, 2}, {0, 3, 1}, {0, 4, 0}, {1, 0, 3}, {1, 1, 2}, {1, 2, 1}, {1, 3, 0}, {2, 0, 2}, {2, 1, 1}, {2, 2, 0}, {3, 0, 1}, {3, 1, 0}, {4, 0, 0},
    {0, 0, 5}, {0, 1, 4}, {0, 2, 3}, {0, 3, 2}, {0, 4, 1}, {0, 5, 0}, {1, 0, 4}, {1, 1, 3}, {1, 2, 2}, {1, 3, 1}, {1, 4, 0}, {2, 0, 3}, {2, 1, 2}, {2, 2, 1}, {2, 3, 0}, {3, 0, 2}, {3, 1, 1}, {3, 2, 0}, {4, 0, 1}, {4, 1, 0}, {5, 0, 0},
    {0, 0, 6}, {0, 1, 5}, {0, 2, 4}, {0, 3, 3}, {0, 4, 2}, {0, 5, 1}, {0, 6, 0}, {1, 0, 5}, {1, 1, 4}, {1, 2, 3}, {1, 3, 2}, {1, 4, 1}, {1, 5, 0}, {2, 0, 4}, {2, 1, 3}, {2, 2, 2}, {2, 3, 1}, {2, 4, 0}, {3, 0, 3}, {3, 1, 2}, {3, 2, 1}, {3, 3, 0}, {4, 0, 2}, {4, 1, 1}, {4, 2, 0}, {5, 0, 1}, {5, 1, 0}, {6, 0, 0},
    {0, 0, 7}, {0, 1, 6}, {0, 2, 5}, {0, 3, 4}, {0, 4, 3}, {0, 5, 2}, {0, 6, 1}, {0, 7, 0}, {1, 0, 6}, {1, 1, 5}, {1, 2, 4}, {1, 3, 3}, {1, 4, 2}, {1, 5, 1}, {1, 6, 0}, {2, 0, 5}, {2, 1, 4}, {2, 2, 3}, {2, 3, 2}, {2, 4, 1}, {2, 5, 0}, {3, 0, 4}, {3, 1, 3}, {3, 2, 2}, {3, 3, 1}, {3, 4, 0}, {4, 0, 3}, {4, 1, 2}, {4, 2, 1}, {4, 3, 0}, {5, 0, 2}, {5, 1, 1}, {5, 2, 0}, {6, 0, 1}, {6, 1, 0}, {7, 0, 0},
    {0, 0, 8}, {0, 1, 7}, {0, 2, 6}, {0, 3, 5}, {0, 4, 4}, {0, 5, 3}, {0, 6, 2}, {0, 7, 1}, {0, 8, 0}, {1, 0, 7}, {1, 1, 6}, {1, 2, 5}, {1, 3, 4}, {1, 4, 3}, {1, 5, 2}, {1, 6, 1}, {1, 7, 0}, {2, 0, 6}, {2, 1, 5}, {2, 2, 4}, {2, 3, 3}, {2, 4, 2}, {2, 5, 1}, {2, 6, 0}, {3, 0, 5}, {3, 1, 4}, {3, 2, 3}, {3, 3, 2}, {3, 4, 1}, {3, 5, 0}, {4, 0, 4}, {4, 1, 3}, {4, 2, 2}, {4, 3, 1}, {4, 4, 0}, {5, 0, 3}, {5, 1, 2}, {5, 2, 1}, {5, 3, 0}, {6, 0, 2}, {6, 1, 1}, {6, 2, 0}, {7, 0, 1}, {7, 1, 0}, {8, 0, 0},
    {0, 0, 9}, {0, 1, 8}, {0, 2, 7}, {0, 3, 6}, {0, 4, 5}, {0, 5, 4}, {0, 6, 3}, {0, 7, 2}, {0, 8, 1}, {0, 9, 0}, {1, 0, 8}, {1, 1, 7}, {1, 2, 6}, {1, 3, 5}, {1, 4, 4}, {1, 5, 3}, {1, 6, 2}, {1, 7, 1}, {1, 8, 0}, {2, 0, 7}, {2, 1, 6}, {2, 2, 5}, {2, 3, 4}, {2, 4, 3}, {2, 5, 2}, {2, 6, 1}, {2, 7, 0}, {3, 0, 6}, {3, 1, 5}, {3, 2, 4}, {3, 3, 3}, {3, 4, 2}, {3, 5, 1}, {3, 6, 0}, {4, 0, 5}, {4, 1, 4}, {4, 2, 3}, {4, 3, 2}, {4, 4, 1}, {4, 5, 0}, {5, 0, 4}, {5, 1, 3}, {5, 2, 2}, {5, 3, 1}, {5, 4, 0}, {6, 0, 3}, {6, 1, 2}, {6, 2, 1}, {6, 3, 0}, {7, 0, 2}, {7, 1, 1}, {7, 2, 0}, {8, 0, 1}, {8, 1, 0}, {9, 0, 0},
    {0, 0, 10}, {0, 1, 9}, {0, 2, 8}, {0, 3, 7}, {0, 4, 6}, {0, 5, 5}, {0, 6, 4}, {0, 7, 3}, {0, 8, 2}, {0, 9, 1}, {0, 10, 0}, {1, 0, 9}, {1, 1, 8}, {1, 2, 7}, {1, 3, 6}, {1, 4, 5}, {1, 5, 4}, {1, 6, 3}, {1, 7, 2}, {1, 8, 1}, {1, 9, 0}, {2, 0, 8}, {2, 1, 7}, {2, 2, 6}, {2, 3, 5}, {2, 4, 4}, {2, 5, 3}, {2, 6, 2}, {2, 7, 1}, {2, 8, 0}, {3, 0, 7}, {3, 1, 6}, {3, 2, 5}, {3, 3, 4}, {3, 4, 3}, {3, 5, 2}, {3, 6, 1}, {3, 7, 0}, {4, 0, 6}, {4, 1, 5}, {4, 2, 4}, {4, 3, 3}, {4, 4, 2}, {4, 5, 1}, {4, 6, 0}, {5, 0, 5}, {5, 1, 4}, {5, 2, 3}, {5, 3, 2}, {5, 4, 1}, {5, 5, 0}, {6, 0, 4}, {6, 1, 3}, {6, 2, 2}, {6, 3, 1}, {6, 4, 0}, {7, 0, 3}, {7, 1, 2}, {7, 2, 1}, {7, 3, 0}, {8, 0, 2}, {8, 1, 1}, {8, 2, 0}, {9, 0, 1}, {9, 1, 0}, {10, 0, 0},
    {0, 0, 11}, {0, 1, 10}, {0, 2, 9}, {0, 3, 8}, {0, 4, 7}, {0, 5, 6}, {0, 6, 5}, {0, 7, 4}, {0, 8, 3}, {0, 9, 2}, {0, 10, 1}, {0, 11, 0}, {1, 0, 10}, {1, 1, 9}, {1, 2, 8}, {1, 3, 7}, {1, 4, 6}, {1, 5, 5}, {1, 6, 4}, {1, 7, 3}, {1, 8, 2}, {1, 9, 1}, {1, 10, 0}, {2, 0, 9}, {2, 1, 8}, {2, 2, 7}, {2, 3, 6}, {2, 4, 5}, {2, 5, 4}, {2, 6, 3}, {2, 7, 2}, {2, 8, 1}, {2, 9, 0}, {3, 0, 8}, {3, 1, 7}, {3, 2, 6}, {3, 3, 5}, {3, 4, 4}, {3, 5, 3}, {3, 6, 2}, {3, 7, 1}, {3, 8, 0}, {4, 0, 7}, {4, 1, 6}, {4, 2, 5}, {4, 3, 4}, {4, 4, 3}, {4, 5, 2}, {4, 6, 1}, {4, 7, 0}, {5, 0, 6}, {5, 1, 5}, {5, 2, 4}, {5, 3, 3}, {5, 4, 2}, {5, 5, 1}, {5, 6, 0}, {6, 0, 5}, {6, 1, 4}, {6, 2, 3}, {6, 3, 2}, {6, 4, 1}, {6, 5, 0}, {7, 0, 4}, {7, 1, 3}, {7, 2, 2}, {7, 3, 1}, {7, 4, 0}, {8, 0, 3}, {8, 1, 2}, {8, 2, 1}, {8, 3, 0}, {9, 0, 2}, {9, 1, 1}, {9, 2, 0}, {10, 0, 1}, {10, 1, 0}, {11, 0, 0},
    {0, 0, 12}, {0, 1, 11}, {0, 2, 10}, {0, 3, 9}, {0, 4, 8}, {0, 5, 7}, {0, 6, 6}, {0, 7, 5}, {0, 8, 4}, {0, 9, 3}, {0, 10, 2}, {0, 11, 1}, {0, 12, 0}, {1, 0, 11}, {1, 1, 10}, {1, 2, 9}, {1, 3, 8}, {1, 4, 7}, {1, 5, 6}, {1, 6, 5}, {1, 7, 4}, {1, 8, 3}, {1, 9, 2}, {1, 10, 1}, {1, 11, 0}, {2, 0, 10}, {2, 1, 9}, {2, 2, 8}, {2, 3, 7}, {2, 4, 6}, {2, 5, 5}, {2, 6, 4}, {2, 7, 3}, {2, 8, 2}, {2, 9, 1}, {2, 10, 0}, {3, 0, 9}, {3, 1, 8}, {3, 2, 7}, {3, 3, 6}, {3, 4, 5}, {3, 5, 4}, {3, 6, 3}, {3, 7, 2}, {3, 8, 1}, {3, 9, 0}, {4, 0, 8}, {4, 1, 7}, {4, 2, 6}, {4, 3, 5}, {4, 4, 4}, {4, 5, 3}, {4, 6, 2}, {4, 7, 1}, {4, 8, 0}, {5, 0, 7}, {5, 1, 6}, {5, 2, 5}, {5, 3, 4}, {5, 4, 3}, {5, 5, 2}, {5, 6, 1}, {5, 7, 0}, {6, 0, 6}, {6, 1, 5}, {6, 2, 4}, {6, 3, 3}, {6, 4, 2}, {6, 5, 1}, {6, 6, 0}, {7, 0, 5}, {7, 1, 4}, {7, 2, 3}, {7, 3, 2}, {7, 4, 1}, {7, 5, 0}, {8, 0, 4}, {8, 1, 3}, {8, 2, 2}, {8, 3, 1}, {8, 4, 0}, {9, 0, 3}, {9, 1, 2}, {9, 2, 1}, {9, 3, 0}, {10, 0, 2}, {10, 1, 1}, {10, 2, 0}, {11, 0, 1}, {11, 1, 0}, {12, 0, 0},
    {0, 0, 13}, {0, 1, 12}, {0, 2, 11}, {0, 3, 10}, {0, 4, 9}, {0, 5, 8}, {0, 6, 7}, {0, 7, 6}, {0, 8, 5}, {0, 9, 4}, {0, 10, 3}, {0, 11, 2}, {0, 12, 1}, {0, 13, 0}, {1, 0, 12}, {1, 1, 11}, {1, 2, 10}, {1, 3, 9}, {1, 4, 8}, {1, 5, 7}, {1, 6, 6}, {1, 7, 5}, {1, 8, 4}, {1, 9, 3}, {1, 10, 2}, {1, 11, 1}, {1, 12, 0}, {2, 0, 11}, {2, 1, 10}, {2, 2, 9}, {2, 3, 8}, {2, 4, 7}, {2, 5, 6}, {2, 6, 5}, {2, 7, 4}, {2, 8, 3}, {2, 9, 2}, {2, 10, 1}, {2, 11, 0}, {3, 0, 10}, {3, 1, 9}, {3, 2, 8}, {3, 3, 7}, {3, 4, 6}, {3, 5, 5}, {3, 6, 4}, {3, 7, 3}, {3, 8, 2}, {3, 9, 1}, {3, 10, 0}, {4, 0, 9}, {4, 1, 8}, {4, 2, 7}, {4, 3, 6}, {4, 4, 5}, {4, 5, 4}, {4, 6, 3}, {4, 7, 2}, {4, 8, 1}, {4, 9, 0}, {5, 0, 8}, {5, 1, 7}, {5, 2, 6}, {5, 3, 5}, {5, 4, 4}, {5, 5, 3}, {5, 6, 2}, {5, 7, 1}, {5, 8, 0}, {6, 0, 7}, {6, 1, 6}, {6, 2, 5}, {6, 3, 4}, {6, 4, 3}, {6, 5, 2}, {6, 6, 1}, {6, 7, 0}, {7, 0, 6}, {7, 1, 5}, {7, 2, 4}, {7, 3, 3}, {7, 4, 2}, {7, 5, 1}, {7, 6, 0}, {8, 0, 5}, {8, 1, 4}, {8, 2, 3}, {8, 3, 2}, {8, 4, 1}, {8, 5, 0}, {9, 0, 4}, {9, 1, 3}, {9, 2, 2}, {9, 3, 1}, {9, 4, 0}, {10, 0, 3}, {10, 1, 2}, {10, 2, 1}, {10, 3, 0}, {11, 0, 2}, {11, 1, 1}, {11, 2, 0}, {12, 0, 1}, {12, 1, 0}, {13, 0, 0},
    {0, 0, 14}, {0, 1, 13}, {0, 2, 12}, {0, 3, 11}, {0, 4, 10}, {0, 5, 9}, {0, 6, 8}, {0, 7, 7}, {0, 8, 6}, {0, 9, 5}, {0, 10, 4}, {0, 11, 3}, {0, 12, 2}, {0, 13, 1}, {0, 14, 0}, {1, 0, 13}, {1, 1, 12}, {1, 2, 11}, {1, 3, 10}, {1, 4, 9}, {1, 5, 8}, {1, 6, 7}, {1, 7, 6}, {1, 8, 5}, {1, 9, 4}, {1, 10, 3}, {1, 11, 2}, {1, 12, 1}, {1, 13, 0}, {2, 0, 12}, {2, 1, 11}, {2, 2, 10}, {2, 3, 9}, {2, 4, 8}, {2, 5, 7}, {2, 6, 6}, {2, 7, 5}, {2, 8, 4}, {2, 9, 3}, {2, 10, 2}, {2, 11, 1}, {2, 12, 0}, {3, 0, 11}, {3, 1, 10}, {3, 2, 9}, {3, 3, 8}, {3, 4, 7}, {3, 5, 6}, {3, 6, 5}, {3, 7, 4}, {3, 8, 3}, {3, 9, 2}, {3, 10, 1}, {3, 11, 0}, {4, 0, 10}, {4, 1, 9}, {4, 2, 8}, {4, 3, 7}, {4, 4, 6}, {4, 5, 5}, {4, 6, 4}, {4, 7, 3}, {4, 8, 2}, {4, 9, 1}, {4, 10, 0}, {5, 0, 9}, {5, 1, 8}, {5, 2, 7}, {5, 3, 6}, {5, 4, 5}, {5, 5, 4}, {5, 6, 3}, {5, 7, 2}, {5, 8, 1}, {5, 9, 0}, {6, 0, 8}, {6, 1, 7}, {6, 2, 6}, {6, 3, 5}, {6, 4, 4}, {6, 5, 3}, {6, 6, 2}, {6, 7, 1}, {6, 8, 0}, {7, 0, 7}, {7, 1, 6}, {7, 2, 5}, {7, 3, 4}, {7, 4, 3}, {7, 5, 2}, {7, 6, 1}, {7, 7, 0}, {8, 0, 6}, {8, 1, 5}, {8, 2, 4}, {8, 3, 3}, {8, 4, 2}, {8, 5, 1}, {8, 6, 0}, {9, 0, 5}, {9, 1, 4}, {9, 2, 3}, {9, 3, 2}, {9, 4, 1}, {9, 5, 0}, {10, 0, 4}, {10, 1, 3}, {10, 2, 2}, {10, 3, 1}, {10, 4, 0}, {11, 0, 3}, {11, 1, 2}, {11, 2, 1}, {11, 3, 0}, {12, 0, 2}, {12, 1, 1}, {12, 2, 0}, {13, 0, 1}, {13, 1, 0}, {14, 0, 0},
    {0, 0, 15}, {0, 1, 14}, {0, 2, 13}, {0, 3, 12}, {0, 4, 11}, {0, 5, 10}, {0, 6, 9}, {0, 7, 8}, {0, 8, 7}, {0, 9, 6}, {0, 10, 5}, {0, 11, 4}, {0, 12, 3}, {0, 13, 2}, {0, 14, 1}, {0, 15, 0}, {1, 0, 14}, {1, 1, 13}, {1, 2, 12}, {1, 3, 11}, {1, 4, 10}, {1, 5, 9}, {1, 6, 8}, {1, 7, 7}, {1, 8, 6}, {1, 9, 5}, {1, 10, 4}, {1, 11, 3}, {1, 12, 2}, {1, 13, 1}, {1, 14, 0}, {2, 0, 13}, {2, 1, 12}, {2, 2, 11}, {2, 3, 10}, {2, 4, 9}, {2, 5, 8}, {2, 6, 7}, {2, 7, 6}, {2, 8, 5}, {2, 9, 4}, {2, 10, 3}, {2, 11, 2}, {2, 12, 1}, {2, 13, 0}, {3, 0, 12}, {3, 1, 11}, {3, 2, 10}, {3, 3, 9}, {3, 4, 8}, {3, 5, 7}, {3, 6, 6}, {3, 7, 5}, {3, 8, 4}, {3, 9, 3}, {3, 10, 2}, {3, 11, 1}, {3, 12, 0}, {4, 0, 11}, {4, 1, 10}, {4, 2, 9}, {4, 3, 8}, {4, 4, 7}, {4, 5, 6}, {4, 6, 5}, {4, 7, 4}, {4, 8, 3}, {4, 9, 2}, {4, 10, 1}, {4, 11, 0}, {5, 0, 10}, {5, 1, 9}, {5, 2, 8}, {5, 3, 7}, {5, 4, 6}, {5, 5, 5}, {5, 6, 4}, {5, 7, 3}, {5, 8, 2}, {5, 9, 1}, {5, 10, 0}, {6, 0, 9}, {6, 1, 8}, {6, 2, 7}, {6, 3, 6}, {6, 4, 5}, {6, 5, 4}, {6, 6, 3}, {6, 7, 2}, {6, 8, 1}, {6, 9, 0}, {7, 0, 8}, {7, 1, 7}, {7, 2, 6}, {7, 3, 5}, {7, 4, 4}, {7, 5, 3}, {7, 6, 2}, {7, 7, 1}, {7, 8, 0}, {8, 0, 7}, {8, 1, 6}, {8, 2, 5}, {8, 3, 4}, {8, 4, 3}, {8, 5, 2}, {8, 6, 1}, {8, 7, 0}, {9, 0, 6}, {9, 1, 5}, {9, 2, 4}, {9, 3, 3}, {9, 4, 2}, {9, 5, 1}, {9, 6, 0}, {10, 0, 5}, {10, 1, 4}, {10, 2, 3}, {10, 3, 2}, {10, 4, 1}, {10, 5, 0}, {11, 0, 4}, {11, 1, 3}, {11, 2, 2}, {11, 3, 1}, {11, 4, 0}, {12, 0, 3}, {12, 1, 2}, {12, 2, 1}, {12, 3, 0}, {13, 0, 2}, {13, 1, 1}, {13, 2, 0}, {14, 0, 1}, {14, 1, 0}, {15, 0, 0},
    {0, 0, 16}, {0, 1, 15}, {0, 2, 14}, {0, 3, 13}, {0, 4, 12}, {0, 5, 11}, {0, 6, 10}, {0, 7, 9}, {0, 8, 8}, {0, 9, 7}, {0, 10, 6}, {0, 11, 5}, {0, 12, 4}, {0, 13, 3}, {0, 14, 2}, {0, 15, 1}, {0, 16, 0}, {1, 0, 15}, {1, 1, 14}, {1, 2, 13}, {1, 3, 12}, {1, 4, 11}, {1, 5, 10}, {1, 6, 9}, {1, 7, 8}, {1, 8, 7}, {1, 9, 6}, {1, 10, 5}, {1, 11, 4}, {1, 12, 3}, {1, 13, 2}, {1, 14, 1}, {1, 15, 0}, {2, 0, 14}, {2, 1, 13}, {2, 2, 12}, {2, 3, 11}, {2, 4, 10}, {2, 5, 9}, {2, 6, 8}, {2, 7, 7}, {2, 8, 6}, {2, 9, 5}, {2, 10, 4}, {2, 11, 3}, {2, 12, 2}, {2, 13, 1}, {2, 14, 0}, {3, 0, 13}, {3, 1, 12}, {3, 2, 11}, {3, 3, 10}, {3, 4, 9}, {3, 5, 8}, {3, 6, 7}, {3, 7, 6}, {3, 8, 5}, {3, 9, 4}, {3, 10, 3}, {3, 11, 2}, {3, 12, 1}, {3, 13, 0}, {4, 0, 12}, {4, 1, 11}, {4, 2, 10}, {4, 3, 9}, {4, 4, 8}, {4, 5, 7}, {4, 6, 6}, {4, 7, 5}, {4, 8, 4}, {4, 9, 3}, {4, 10, 2}, {4, 11, 1}, {4, 12, 0}, {5, 0, 11}, {5, 1, 10}, {5, 2, 9}, {5, 3, 8}, {5, 4, 7}, {5, 5, 6}, {5, 6, 5}, {5, 7, 4}, {5, 8, 3}, {5, 9, 2}, {5, 10, 1}, {5, 11, 0}, {6, 0, 10}, {6, 1, 9}, {6, 2, 8}, {6, 3, 7}, {6, 4, 6}, {6, 5, 5}, {6, 6, 4}, {6, 7, 3}, {6, 8, 2}, {6, 9, 1}, {6, 10, 0}, {7, 0, 9}, {7, 1, 8}, {7, 2, 7}, {7, 3, 6}, {7, 4, 5}, {7, 5, 4}, {7, 6, 3}, {7, 7, 2}, {7, 8, 1}, {7, 9, 0}, {8, 0, 8}, {8, 1, 7}, {8, 2, 6}, {8, 3, 5}, {8, 4, 4}, {8, 5, 3}, {8, 6, 2}, {8, 7, 1}, {8, 8, 0}, {9, 0, 7}, {9, 1, 6}, {9, 2, 5}, {9, 3, 4}, {9, 4, 3}, {9, 5, 2}, {9, 6, 1}, {9, 7, 0}, {10, 0, 6}, {10, 1, 5}, {10, 2, 4}, {10, 3, 3}, {10, 4, 2}, {10, 5, 1}, {10, 6, 0}, {11, 0, 5}, {11, 1, 4}, {11, 2, 3}, {11, 3, 2}, {11, 4, 1}, {11, 5, 0}, {12, 0, 4}, {12, 1, 3}, {12, 2, 2}, {12, 3, 1}, {12, 4, 0}, {13, 0, 3}, {13, 1, 2}, {13, 2, 1}, {13, 3, 0}, {14, 0, 2}, {14, 1, 1}, {14, 2, 0}, {15, 0, 1}, {15, 1, 0}, {16, 0, 0},
    {0, 0, 17}, {0, 1, 16}, {0, 2, 15}, {0, 3, 14}, {0, 4, 13}, {0, 5, 12}, {0, 6, 11}, {0, 7, 10}, {0, 8, 9}, {0, 9, 8}, {0, 10, 7}, {0, 11, 6}, {0, 12, 5}, {0, 13, 4}, {0, 14, 3}, {0, 15, 2}, {0, 16, 1}, {0, 17, 0}, {1, 0, 16}, {1, 1, 15}, {1, 2, 14}, {1, 3, 13}, {1, 4, 12}, {1, 5, 11}, {1, 6, 10}, {1, 7, 9}, {1, 8, 8}, {1, 9, 7}, {1, 10, 6}, {1, 11, 5}, {1, 12, 4}, {1, 13, 3}, {1, 14, 2}, {1, 15, 1}, {1, 16, 0}, {2, 0, 15}, {2, 1, 14}, {2, 2, 13}, {2, 3, 12}, {2, 4, 11}, {2, 5, 10}, {2, 6, 9}, {2, 7, 8}, {2, 8, 7}, {2, 9, 6}, {2, 10, 5}, {2, 11, 4}, {2, 12, 3}, {2, 13, 2}, {2, 14, 1}, {2, 15, 0}, {3, 0, 14}, {3, 1, 13}, {3, 2, 12}, {3, 3, 11}, {3, 4, 10}, {3, 5, 9}, {3, 6, 8}, {3, 7, 7}, {3, 8, 6}, {3, 9, 5}, {3, 10, 4}, {3, 11, 3}, {3, 12, 2}, {3, 13, 1}, {3, 14, 0}, {4, 0, 13}, {4, 1, 12}, {4, 2, 11}, {4, 3, 10}, {4, 4, 9}, {4, 5, 8}, {4, 6, 7}, {4, 7, 6}, {4, 8, 5}, {4, 9, 4}, {4, 10, 3}, {4, 11, 2}, {4, 12, 1}, {4, 13, 0}, {5, 0, 12}, {5, 1, 11}, {5, 2, 10}, {5, 3, 9}, {5, 4, 8}, {5, 5, 7}, {5, 6, 6}, {5, 7, 5}, {5, 8, 4}, {5, 9, 3}, {5, 10, 2}, {5, 11, 1}, {5, 12, 0}, {6, 0, 11}, {6, 1, 10}, {6, 2, 9}, {6, 3, 8}, {6, 4, 7}, {6, 5, 6}, {6, 6, 5}, {6, 7, 4}, {6, 8, 3}, {6, 9, 2}, {6, 10, 1}, {6, 11, 0}, {7, 0, 10}, {7, 1, 9}, {7, 2, 8}, {7, 3, 7}, {7, 4, 6}, {7, 5, 5}, {7, 6, 4}, {7, 7, 3}, {7, 8, 2}, {7, 9, 1}, {7, 10, 0}, {8, 0, 9}, {8, 1, 8}, {8, 2, 7}, {8, 3, 6}, {8, 4, 5}, {8, 5, 4}, {8, 6, 3}, {8, 7, 2}, {8, 8, 1}, {8, 9, 0}, {9, 0, 8}, {9, 1, 7}, {9, 2, 6}, {9, 3, 5}, {9, 4, 4}, {9, 5, 3}, {9, 6, 2}, {9, 7, 1}, {9, 8, 0}, {10, 0, 7}, {10, 1, 6}, {10, 2, 5}, {10, 3, 4}, {10, 4, 3}, {10, 5, 2}, {10, 6, 1}, {10, 7, 0}, {11, 0, 6}, {11, 1, 5}, {11, 2, 4}, {11, 3, 3}, {11, 4, 2}, {11, 5, 1}, {11, 6, 0}, {12, 0, 5}, {12, 1, 4}, {12, 2, 3}, {12, 3, 2}, {12, 4, 1}, {12, 5, 0}, {13, 0, 4}, {13, 1, 3}, {13, 2, 2}, {13, 3, 1}, {13, 4, 0}, {14, 0, 3}, {14, 1, 2}, {14, 2, 1}, {14, 3, 0}, {15, 0, 2}, {15, 1, 1}, {15, 2, 0}, {16, 0, 1}, {16, 1, 0}, {17, 0, 0},
    {0, 0, 18}, {0, 1, 17}, {0, 2, 16}, {0, 3, 15}, {0, 4, 14}, {0, 5, 13}, {0, 6, 12}, {0, 7, 11}, {0, 8, 10}, {0, 9, 9}, {0, 10, 8}, {0, 11, 7}, {0, 12, 6}, {0, 13, 5}, {0, 14, 4}, {0, 15, 3}, {0, 16, 2}, {0, 17, 1}, {0, 18, 0}, {1, 0, 17}, {1, 1, 16}, {1, 2, 15}, {1, 3, 14}, {1, 4, 13}, {1, 5, 12}, {1, 6, 11}, {1, 7, 10}, {1, 8, 9}, {1, 9, 8}, {1, 10, 7}, {1, 11, 6}, {1, 12, 5}, {1, 13, 4}, {1, 14, 3}, {1, 15, 2}, {1, 16, 1}, {1, 17, 0}, {2, 0, 16}, {2, 1, 15}, {2, 2, 14}, {2, 3, 13}, {2, 4, 12}, {2, 5, 11}, {2, 6, 10}, {2, 7, 9}, {2, 8, 8}, {2, 9, 7}, {2, 10, 6}, {2, 11, 5}, {2, 12, 4}, {2, 13, 3}, {2, 14, 2}, {2, 15, 1}, {2, 16, 0}, {3, 0, 15}, {3, 1, 14}, {3, 2, 13}, {3, 3, 12}, {3, 4, 11}, {3, 5, 10}, {3, 6, 9}, {3, 7, 8}, {3, 8, 7}, {3, 9, 6}, {3, 10, 5}, {3, 11, 4}, {3, 12, 3}, {3, 13, 2}, {3, 14, 1}, {3, 15, 0}, {4, 0, 14}, {4, 1, 13}, {4, 2, 12}, {4, 3, 11}, {4, 4, 10}, {4, 5, 9}, {4, 6, 8}, {4, 7, 7}, {4, 8, 6}, {4, 9, 5}, {4, 10, 4}, {4, 11, 3}, {4, 12, 2}, {4, 13, 1}, {4, 14, 0}, {5, 0, 13}, {5, 1, 12}, {5, 2, 11}, {5, 3, 10}, {5, 4, 9}, {5, 5, 8}, {5, 6, 7}, {5, 7, 6}, {5, 8, 5}, {5, 9, 4}, {5, 10, 3}, {5, 11, 2}, {5, 12, 1}, {5, 13, 0}, {6, 0, 12}, {6, 1, 11}, {6, 2, 10}, {6, 3, 9}, {6, 4, 8}, {6, 5, 7}, {6, 6, 6}, {6, 7, 5}, {6, 8, 4}, {6, 9, 3}, {6, 10, 2}, {6, 11, 1}, {6, 12, 0}, {7, 0, 11}, {7, 1, 10}, {7, 2, 9}, {7, 3, 8}, {7, 4, 7}, {7, 5, 6}, {7, 6, 5}, {7, 7, 4}, {7, 8, 3}, {7, 9, 2}, {7, 10, 1}, {7, 11, 0}, {8, 0, 10}, {8, 1, 9}, {8, 2, 8}, {8, 3, 7}, {8, 4, 6}, {8, 5, 5}, {8, 6, 4}, {8, 7, 3}, {8, 8, 2}, {8, 9, 1}, {8, 10, 0}, {9, 0, 9}, {9, 1, 8}, {9, 2, 7}, {9, 3, 6}, {9, 4, 5}, {9, 5, 4}, {9, 6, 3}, {9, 7, 2}, {9, 8, 1}, {9, 9, 0}, {10, 0, 8}, {10, 1, 7}, {10, 2, 6}, {10, 3, 5}, {10, 4, 4}, {10, 5, 3}, {10, 6, 2}, {10, 7, 1}, {10, 8, 0}, {11, 0, 7}, {11, 1, 6}, {11, 2, 5}, {11, 3, 4}, {11, 4, 3}, {11, 5, 2}, {11, 6, 1}, {11, 7, 0}, {12, 0, 6}, {12, 1, 5}, {12, 2, 4}, {12, 3, 3}, {12, 4, 2}, {12, 5, 1}, {12, 6, 0}, {13, 0, 5}, {13, 1, 4}, {13, 2, 3}, {13, 3, 2}, {13, 4, 1}, {13, 5, 0}, {14, 0, 4}, {14, 1, 3}, {14, 2, 2}, {14, 3, 1}, {14, 4, 0}, {15, 0, 3}, {15, 1, 2}, {15, 2, 1}, {15, 3, 0}, {16, 0, 2}, {16, 1, 1}, {16, 2, 0}, {17, 0, 1}, {17, 1, 0}, {18, 0, 0},
    {0, 0, 19}, {0, 1, 18}, {0, 2, 17}, {0, 3, 16}, {0, 4, 15}, {0, 5, 14}, {0, 6, 13}, {0, 7, 12}, {0, 8, 11}, {0, 9, 10}, {0, 10, 9}, {0, 11, 8}, {0, 12, 7}, {0, 13, 6}, {0, 14, 5}, {0, 15, 4}, {0, 16, 3}, {0, 17, 2}, {0, 18, 1}, {0, 19, 0}, {1, 0, 18}, {1, 1, 17}, {1, 2, 16}, {1, 3, 15}, {1, 4, 14}, {1, 5, 13}, {1, 6, 12}, {1, 7, 11}, {1, 8, 10}, {1, 9, 9}, {1, 10, 8}, {1, 11, 7}, {1, 12, 6}, {1, 13, 5}, {1, 14, 4}, {1, 15, 3}, {1, 16, 2}, {1, 17, 1}, {1, 18, 0}, {2, 0, 17}, {2, 1, 16}, {2, 2, 15}, {2, 3, 14}, {2, 4, 13}, {2, 5, 12}, {2, 6, 11}, {2, 7, 10}, {2, 8, 9}, {2, 9, 8}, {2, 10, 7}, {2, 11, 6}, {2, 12, 5}, {2, 13, 4}, {2, 14, 3}, {2, 15, 2}, {2, 16, 1}, {2, 17, 0}, {3, 0, 16}, {3, 1, 15}, {3, 2, 14}, {3, 3, 13}, {3, 4, 12}, {3, 5, 11}, {3, 6, 10}, {3, 7, 9}, {3, 8, 8}, {3, 9, 7}, {3, 10, 6}, {3, 11, 5}, {3, 12, 4}, {3, 13, 3}, {3, 14, 2}, {3, 15, 1}, {3, 16, 0}, {4, 0, 15}, {4, 1, 14}, {4, 2, 13}, {4, 3, 12}, {4, 4, 11}, {4, 5, 10}, {4, 6, 9}, {4, 7, 8}, {4, 8, 7}, {4, 9, 6}, {4, 10, 5}, {4, 11, 4}, {4, 12, 3}, {4, 13, 2}, {4, 14, 1}, {4, 15, 0}, {5, 0, 14}, {5, 1, 13}, {5, 2, 12}, {5, 3, 11}, {5, 4, 10}, {5, 5, 9}, {5, 6, 8}, {5, 7, 7}, {5, 8, 6}, {5, 9, 5}, {5, 10, 4}, {5, 11, 3}, {5, 12, 2}, {5, 13, 1}, {5, 14, 0}, {6, 0, 13}, {6, 1, 12}, {6, 2, 11}, {6, 3, 10}, {6, 4, 9}, {6, 5, 8}, {6, 6, 7}, {6, 7, 6}, {6, 8, 5}, {6, 9, 4}, {6, 10, 3}, {6, 11, 2}, {6, 12, 1}, {6, 13, 0}, {7, 0, 12}, {7, 1, 11}, {7, 2, 10}, {7, 3, 9}, {7, 4, 8}, {7, 5, 7}, {7, 6, 6}, {7, 7, 5}, {7, 8, 4}, {7, 9, 3}, {7, 10, 2}, {7, 11, 1}, {7, 12, 0}, {8, 0, 11}, {8, 1, 10}, {8, 2, 9}, {8, 3, 8}, {8, 4, 7}, {8, 5, 6}, {8, 6, 5}, {8, 7, 4}, {8, 8, 3}, {8, 9, 2}, {8, 10, 1}, {8, 11, 0}, {9, 0, 10}, {9, 1, 9}, {9, 2, 8}, {9, 3, 7}, {9, 4, 6}, {9, 5, 5}, {9, 6, 4}, {9, 7, 3}, {9, 8, 2}, {9, 9, 1}, {9, 10, 0}, {10, 0, 9}, {10, 1, 8}, {10, 2, 7}, {10, 3, 6}, {10, 4, 5}, {10, 5, 4}, {10, 6, 3}, {10, 7, 2}, {10, 8, 1}, {10, 9, 0}, {11, 0, 8}, {11, 1, 7}, {11, 2, 6}, {11, 3, 5}, {11, 4, 4}, {11, 5, 3}, {11, 6, 2}, {11, 7, 1}, {11, 8, 0}, {12, 0, 7}, {12, 1, 6}, {12, 2, 5}, {12, 3, 4}, {12, 4, 3}, {12, 5, 2}, {12, 6, 1}, {12, 7, 0}, {13, 0, 6}, {13, 1, 5}, {13, 2, 4}, {13, 3, 3}, {13, 4, 2}, {13, 5, 1}, {13, 6, 0}, {14, 0, 5}, {14, 1, 4}, {14, 2, 3}, {14, 3, 2}, {14, 4, 1}, {14, 5, 0}, {15, 0, 4}, {15, 1, 3}, {15, 2, 2}, {15, 3, 1}, {15, 4, 0}, {16, 0, 3}, {16, 1, 2}, {16, 2, 1}, {16, 3, 0}, {17, 0, 2}, {17, 1, 1}, {17, 2, 0}, {18, 0, 1}, {18, 1, 0}, {19, 0, 0},
    {0, 0, 20}, {0, 1, 19}, {0, 2, 18}, {0, 3, 17}, {0, 4, 16}, {0, 5, 15}, {0, 6, 14}, {0, 7, 13}, {0, 8, 12}, {0, 9, 11}, {0, 10, 10}, {0, 11, 9}, {0, 12, 8}, {0, 13, 7}, {0, 14, 6}, {0, 15, 5}, {0, 16, 4}, {0, 17, 3}, {0, 18, 2}, {0, 19, 1}, {0, 20, 0}, {1, 0, 19}, {1, 1, 18}, {1, 2, 17}, {1, 3, 16}, {1, 4, 15}, {1, 5, 14}, {1, 6, 13}, {1, 7, 12}, {1, 8, 11}, {1, 9, 10}, {1, 10, 9}, {1, 11, 8}, {1, 12, 7}, {1, 13, 6}, {1, 14, 5}, {1, 15, 4}, {1, 16, 3}, {1, 17, 2}, {1, 18, 1}, {1, 19, 0}, {2, 0, 18}, {2, 1, 17}, {2, 2, 16}, {2, 3, 15}, {2, 4, 14}, {2, 5, 13}, {2, 6, 12}, {2, 7, 11}, {2, 8, 10}, {2, 9, 9}, {2, 10, 8}, {2, 11, 7}, {2, 12, 6}, {2, 13, 5}, {2, 14, 4}, {2, 15, 3}, {2, 16, 2}, {2, 17, 1}, {2, 18, 0}, {3, 0, 17}, {3, 1, 16}, {3, 2, 15}, {3, 3, 14}, {3, 4, 13}, {3, 5, 12}, {3, 6, 11}, {3, 7, 10}, {3, 8, 9}, {3, 9, 8}, {3, 10, 7}, {3, 11, 6}, {3, 12, 5}, {3, 13, 4}, {3, 14, 3}, {3, 15, 2}, {3, 16, 1}, {3, 17, 0}, {4, 0, 16}, {4, 1, 15}, {4, 2, 14}, {4, 3, 13}, {4, 4, 12}, {4, 5, 11}, {4, 6, 10}, {4, 7, 9}, {4, 8, 8}, {4, 9, 7}, {4, 10, 6}, {4, 11, 5}, {4, 12, 4}, {4, 13, 3}, {4, 14, 2}, {4, 15, 1}, {4, 16, 0}, {5, 0, 15}, {5, 1, 14}, {5, 2, 13}, {5, 3, 12}, {5, 4, 11}, {5, 5, 10}, {5, 6, 9}, {5, 7, 8}, {5, 8, 7}, {5, 9, 6}, {5, 10, 5}, {5, 11, 4}, {5, 12, 3}, {5, 13, 2}, {5, 14, 1}, {5, 15, 0}, {6, 0, 14}, {6, 1, 13}, {6, 2, 12}, {6, 3, 11}, {6, 4, 10}, {6, 5, 9}, {6, 6, 8}, {6, 7, 7}, {6, 8, 6}, {6, 9, 5}, {6, 10, 4}, {6, 11, 3}, {6, 12, 2}, {6, 13, 1}, {6, 14, 0}, {7, 0, 13}, {7, 1, 12}, {7, 2, 11}, {7, 3, 10}, {7, 4, 9}, {7, 5, 8}, {7, 6, 7}, {7, 7, 6}, {7, 8, 5}, {7, 9, 4}, {7, 10, 3}, {7, 11, 2}, {7, 12, 1}, {7, 13, 0}, {8, 0, 12}, {8, 1, 11}, {8, 2, 10}, {8, 3, 9}, {8, 4, 8}, {8, 5, 7}, {8, 6, 6}, {8, 7, 5}, {8, 8, 4}, {8, 9, 3}, {8, 10, 2}, {8, 11, 1}, {8, 12, 0}, {9, 0, 11}, {9, 1, 10}, {9, 2, 9}, {9, 3, 8}, {9, 4, 7}, {9, 5, 6}, {9, 6, 5}, {9, 7, 4}, {9, 8, 3}, {9, 9, 2}, {9, 10, 1}, {9, 11, 0}, {10, 0, 10}, {10, 1, 9}, {10, 2, 8}, {10, 3, 7}, {10, 4, 6}, {10, 5, 5}, {10, 6, 4}, {10, 7, 3}, {10, 8, 2}, {10, 9, 1}, {10, 10, 0}, {11, 0, 9}, {11, 1, 8}, {11, 2, 7}, {11, 3, 6}, {11, 4, 5}, {11, 5, 4}, {11, 6, 3}, {11, 7, 2}, {11, 8, 1}, {11, 9, 0}, {12, 0, 8}, {12, 1, 7}, {12, 2, 6}, {12, 3, 5}, {12, 4, 4}, {12, 5, 3}, {12, 6, 2}, {12, 7, 1}, {12, 8, 0}, {13, 0, 7}, {13, 1, 6}, {13, 2, 5}, {13, 3, 4}, {13, 4, 3}, {13, 5, 2}, {13, 6, 1}, {13, 7, 0}, {14, 0, 6}, {14, 1, 5}, {14, 2, 4}, {14, 3, 3}, {14, 4, 2}, {14, 5, 1}, {14, 6, 0}, {15, 0, 5}, {15, 1, 4}, {15, 2, 3}, {15, 3, 2}, {15, 4, 1}, {15, 5, 0}, {16, 0, 4}, {16, 1, 3}, {16, 2, 2}, {16, 3, 1}, {16, 4, 0}, {17, 0, 3}, {17, 1, 2}, {17, 2, 1}, {17, 3, 0}, {18, 0, 2}, {18, 1, 1}, {18, 2, 0}, {19, 0, 1}, {19, 1, 0}, {20, 0, 0},
    {0, 0, 21}, {0, 1, 20}, {0, 2, 19}, {0, 3, 18}, {0, 4, 17}, {0, 5, 16}, {0, 6, 15}, {0, 7, 14}, {0, 8, 13}, {0, 9, 12}, {0, 10, 11}, {0, 11, 10}, {0, 12, 9}, {0, 13, 8}, {0, 14, 7}, {0, 15, 6}, {0, 16, 5}, {0, 17, 4}, {0, 18, 3}, {0, 19, 2}, {0, 20, 1}, {0, 21, 0}, {1, 0, 20}, {1, 1, 19}, {1, 2, 18}, {1, 3, 17}, {1, 4, 16}, {1, 5, 15}, {1, 6, 14}, {1, 7, 13}, {1, 8, 12}, {1, 9, 11}, {1, 10, 10}, {1, 11, 9}, {1, 12, 8}, {1, 13, 7}, {1, 14, 6}, {1, 15, 5}, {1, 16, 4}, {1, 17, 3}, {1, 18, 2}, {1, 19, 1}, {1, 20, 0}, {2, 0, 19}, {2, 1, 18}, {2, 2, 17}, {2, 3, 16}, {2, 4, 15}, {2, 5, 14}, {2, 6, 13}, {2, 7, 12}, {2, 8, 11}, {2, 9, 10}, {2, 10, 9}, {2, 11, 8}, {2, 12, 7}, {2, 13, 6}, {2, 14, 5}, {2, 15, 4}, {2, 16, 3}, {2, 17, 2}, {2, 18, 1}, {2, 19, 0}, {3, 0, 18}, {3, 1, 17}, {3, 2, 16}, {3, 3, 15}, {3, 4, 14}, {3, 5, 13}, {3, 6, 12}, {3, 7, 11}, {3, 8, 10}, {3, 9, 9}, {3, 10, 8}, {3, 11, 7}, {3, 12, 6}, {3, 13, 5}, {3, 14, 4}, {3, 15, 3}, {3, 16, 2}, {3, 17, 1}, {3, 18, 0}, {4, 0, 17}, {4, 1, 16}, {4, 2, 15}, {4, 3, 14}, {4, 4, 13}, {4, 5, 12}, {4, 6, 11}, {4, 7, 10}, {4, 8, 9}, {4, 9, 8}, {4, 10, 7}, {4, 11, 6}, {4, 12, 5}, {4, 13, 4}, {4, 14, 3}, {4, 15, 2}, {4, 16, 1}, {4, 17, 0}, {5, 0, 16}, {5, 1, 15}, {5, 2, 14}, {5, 3, 13}, {5, 4, 12}, {5, 5, 11}, {5, 6, 10}, {5, 7, 9}, {5, 8, 8}, {5, 9, 7}, {5, 10, 6}, {5, 11, 5}, {5, 12, 4}, {5, 13, 3}, {5, 14, 2}, {5, 15, 1}, {5, 16, 0}, {6, 0, 15}, {6, 1, 14}, {6, 2, 13}, {6, 3, 12}, {6, 4, 11}, {6, 5, 10}, {6, 6, 9}, {6, 7, 8}, {6, 8, 7}, {6, 9, 6}, {6, 10, 5}, {6, 11, 4}, {6, 12, 3}, {6, 13, 2}, {6, 14, 1}, {6, 15, 0}, {7, 0, 14}, {7, 1, 13}, {7, 2, 12}, {7, 3, 11}, {7, 4, 10}, {7, 5, 9}, {7, 6, 8}, {7, 7, 7}, {7, 8, 6}, {7, 9, 5}, {7, 10, 4}, {7, 11, 3}, {7, 12, 2}, {7, 13, 1}, {7, 14, 0}, {8, 0, 13}, {8, 1, 12}, {8, 2, 11}, {8, 3, 10}, {8, 4, 9}, {8, 5, 8}, {8, 6, 7}, {8, 7, 6}, {8, 8, 5}, {8, 9, 4}, {8, 10, 3}, {8, 11, 2}, {8, 12, 1}, {8, 13, 0}, {9, 0, 12}, {9, 1, 11}, {9, 2, 10}, {9, 3, 9}, {9, 4, 8}, {9, 5, 7}, {9, 6, 6}, {9, 7, 5}, {9, 8, 4}, {9, 9, 3}, {9, 10, 2}, {9, 11, 1}, {9, 12, 0}, {10, 0, 11}, {10, 1, 10}, {10, 2, 9}, {10, 3, 8}, {10, 4, 7}, {10, 5, 6}, {10, 6, 5}, {10, 7, 4}, {10, 8, 3}, {10, 9, 2}, {10, 10, 1}, {10, 11, 0}, {11, 0, 10}, {11, 1, 9}, {11, 2, 8}, {11, 3, 7}, {11, 4, 6}, {11, 5, 5}, {11, 6, 4}, {11, 7, 3}, {11, 8, 2}, {11, 9, 1}, {11, 10, 0}, {12, 0, 9}, {12, 1, 8}, {12, 2, 7}, {12, 3, 6}, {12, 4, 5}, {12, 5, 4}, {12, 6, 3}, {12, 7, 2}, {12, 8, 1}, {12, 9, 0}, {13, 0, 8}, {13, 1, 7}, {13, 2, 6}, {13, 3, 5}, {13, 4, 4}, {13, 5, 3}, {13, 6, 2}, {13, 7, 1}, {13, 8, 0}, {14, 0, 7}, {14, 1, 6}, {14, 2, 5}, {14, 3, 4}, {14, 4, 3}, {14, 5, 2}, {14, 6, 1}, {14, 7, 0}, {15, 0, 6}, {15, 1, 5}, {15, 2, 4}, {15, 3, 3}, {15, 4, 2}, {15, 5, 1}, {15, 6, 0}, {16, 0, 5}, {16, 1, 4}, {16, 2, 3}, {16, 3, 2}, {16, 4, 1}, {16, 5, 0}, {17, 0, 4}, {17, 1, 3}, {17, 2, 2}, {17, 3, 1}, {17, 4, 0}, {18, 0, 3}, {18, 1, 2}, {18, 2, 1}, {18, 3, 0}, {19, 0, 2}, {19, 1, 1}, {19, 2, 0}, {20, 0, 1}, {20, 1, 0}, {21, 0, 0},
    {0, 0, 22}, {0, 1, 21}, {0, 2, 20}, {0, 3, 19}, {0, 4, 18}, {0, 5, 17}, {0, 6, 16}, {0, 7, 15}, {0, 8, 14}, {0, 9, 13}, {0, 10, 12}, {0, 11, 11}, {0, 12, 10}, {0, 13, 9}, {0, 14, 8}, {0, 15, 7}, {0, 16, 6}, {0, 17, 5}, {0, 18, 4}, {0, 19, 3}, {0, 20, 2}, {0, 21, 1}, {0, 22, 0}, {1, 0, 21}, {1, 1, 20}, {1, 2, 19}, {1, 3, 18}, {1, 4, 17}, {1, 5, 16}, {1, 6, 15}, {1, 7, 14}, {1, 8, 13}, {1, 9, 12}, {1, 10, 11}, {1, 11, 10}, {1, 12, 9}, {1, 13, 8}, {1, 14, 7}, {1, 15, 6}, {1, 16, 5}, {1, 17, 4}, {1, 18, 3}, {1, 19, 2}, {1, 20, 1}, {1, 21, 0}, {2, 0, 20}, {2, 1, 19}, {2, 2, 18}, {2, 3, 17}, {2, 4, 16}, {2, 5, 15}, {2, 6, 14}, {2, 7, 13}, {2, 8, 12}, {2, 9, 11}, {2, 10, 10}, {2, 11, 9}, {2, 12, 8}, {2, 13, 7}, {2, 14, 6}, {2, 15, 5}, {2, 16, 4}, {2, 17, 3}, {2, 18, 2}, {2, 19, 1}, {2, 20, 0}, {3, 0, 19}, {3, 1, 18}, {3, 2, 17}, {3, 3, 16}, {3, 4, 15}, {3, 5, 14}, {3, 6, 13}, {3, 7, 12}, {3, 8, 11}, {3, 9, 10}, {3, 10, 9}, {3, 11, 8}, {3, 12, 7}, {3, 13, 6}, {3, 14, 5}, {3, 15, 4}, {3, 16, 3}, {3, 17, 2}, {3, 18, 1}, {3, 19, 0}, {4, 0, 18}, {4, 1, 17}, {4, 2, 16}, {4, 3, 15}, {4, 4, 14}, {4, 5, 13}, {4, 6, 12}, {4, 7, 11}, {4, 8, 10}, {4, 9, 9}, {4, 10, 8}, {4, 11, 7}, {4, 12, 6}, {4, 13, 5}, {4, 14, 4}, {4, 15, 3}, {4, 16, 2}, {4, 17, 1}, {4, 18, 0}, {5, 0, 17}, {5, 1, 16}, {5, 2, 15}, {5, 3, 14}, {5, 4, 13}, {5, 5, 12}, {5, 6, 11}, {5, 7, 10}, {5, 8, 9}, {5, 9, 8}, {5, 10, 7}, {5, 11, 6}, {5, 12, 5}, {5, 13, 4}, {5, 14, 3}, {5, 15, 2}, {5, 16, 1}, {5, 17, 0}, {6, 0, 16}, {6, 1, 15}, {6, 2, 14}, {6, 3, 13}, {6, 4, 12}, {6, 5, 11}, {6, 6, 10}, {6, 7, 9}, {6, 8, 8}, {6, 9, 7}, {6, 10, 6}, {6, 11, 5}, {6, 12, 4}, {6, 13, 3}, {6, 14, 2}, {6, 15, 1}, {6, 16, 0}, {7, 0, 15}, {7, 1, 14}, {7, 2, 13}, {7, 3, 12}, {7, 4, 11}, {7, 5, 10}, {7, 6, 9}, {7, 7, 8}, {7, 8, 7}, {7, 9, 6}, {7, 10, 5}, {7, 11, 4}, {7, 12, 3}, {7, 13, 2}, {7, 14, 1}, {7, 15, 0}, {8, 0, 14}, {8, 1, 13}, {8, 2, 12}, {8, 3, 11}, {8, 4, 10}, {8, 5, 9}, {8, 6, 8}, {8, 7, 7}, {8, 8, 6}, {8, 9, 5}, {8, 10, 4}, {8, 11, 3}, {8, 12, 2}, {8, 13, 1}, {8, 14, 0}, {9, 0, 13}, {9, 1, 12}, {9, 2, 11}, {9, 3, 10}, {9, 4, 9}, {9, 5, 8}, {9, 6, 7}, {9, 7, 6}, {9, 8, 5}, {9, 9, 4}, {9, 10, 3}, {9, 11, 2}, {9, 12, 1}, {9, 13, 0}, {10, 0, 12}, {10, 1, 11}, {10, 2, 10}, {10, 3, 9}, {10, 4, 8}, {10, 5, 7}, {10, 6, 6}, {10, 7, 5}, {10, 8, 4}, {10, 9, 3}, {10, 10, 2}, {10, 11, 1}, {10, 12, 0}, {11, 0, 11}, {11, 1, 10}, {11, 2, 9}, {11, 3, 8}, {11, 4, 7}, {11, 5, 6}, {11, 6, 5}, {11, 7, 4}, {11, 8, 3}, {11, 9, 2}, {11, 10, 1}, {11, 11, 0}, {12, 0, 10}, {12, 1, 9}, {12, 2, 8}, {12, 3, 7}, {12, 4, 6}, {12, 5, 5}, {12, 6, 4}, {12, 7, 3}, {12, 8, 2}, {12, 9, 1}, {12, 10, 0}, {13, 0, 9}, {13, 1, 8}, {13, 2, 7}, {13, 3, 6}, {13, 4, 5}, {13, 5, 4}, {13, 6, 3}, {13, 7, 2}, {13, 8, 1}, {13, 9, 0}, {14, 0, 8}, {14, 1, 7}, {14, 2, 6}, {14, 3, 5}, {14, 4, 4}, {14, 5, 3}, {14, 6, 2}, {14, 7, 1}, {14, 8, 0}, {15, 0, 7}, {15, 1, 6}, {15, 2, 5}, {15, 3, 4}, {15, 4, 3}, {15, 5, 2}, {15, 6, 1}, {15, 7, 0}, {16, 0, 6}, {16, 1, 5}, {16, 2, 4}, {16, 3, 3}, {16, 4, 2}, {16, 5, 1}, {16, 6, 0}, {17, 0, 5}, {17, 1, 4}, {17, 2, 3}, {17, 3, 2}, {17, 4, 1}, {17, 5, 0}, {18, 0, 4}, {18, 1, 3}, {18, 2, 2}, {18, 3, 1}, {18, 4, 0}, {19, 0, 3}, {19, 1, 2}, {19, 2, 1}, {19, 3, 0}, {20, 0, 2}, {20, 1, 1}, {20, 2, 0}, {21, 0, 1}, {21, 1, 0}, {22, 0, 0},
    {0, 0, 23}, {0, 1, 22}, {0, 2, 21}, {0, 3, 20}, {0, 4, 19}, {0, 5, 18}, {0, 6, 17}, {0, 7, 16}, {0, 8, 15}, {0, 9, 14}, {0, 10, 13}, {0, 11, 12}, {0, 12, 11}, {0, 13, 10}, {0, 14, 9}, {0, 15, 8}, {0, 16, 7}, {0, 17, 6}, {0, 18, 5}, {0, 19, 4}, {0, 20, 3}, {0, 21, 2}, {0, 22, 1}, {0, 23, 0}, {1, 0, 22}, {1, 1, 21}, {1, 2, 20}, {1, 3, 19}, {1, 4, 18}, {1, 5, 17}, {1, 6, 16}, {1, 7, 15}, {1, 8, 14}, {1, 9, 13}, {1, 10, 12}, {1, 11, 11}, {1, 12, 10}, {1, 13, 9}, {1, 14, 8}, {1, 15, 7}, {1, 16, 6}, {1, 17, 5}, {1, 18, 4}, {1, 19, 3}, {1, 20, 2}, {1, 21, 1}, {1, 22, 0}, {2, 0, 21}, {2, 1, 20}, {2, 2, 19}, {2, 3, 18}, {2, 4, 17}, {2, 5, 16}, {2, 6, 15}, {2, 7, 14}, {2, 8, 13}, {2, 9, 12}, {2, 10, 11}, {2, 11, 10}, {2, 12, 9}, {2, 13, 8}, {2, 14, 7}, {2, 15, 6}, {2, 16, 5}, {2, 17, 4}, {2, 18, 3}, {2, 19, 2}, {2, 20, 1}, {2, 21, 0}, {3, 0, 20}, {3, 1, 19}, {3, 2, 18}, {3, 3, 17}, {3, 4, 16}, {3, 5, 15}, {3, 6, 14}, {3, 7, 13}, {3, 8, 12}, {3, 9, 11}, {3, 10, 10}, {3, 11, 9}, {3, 12, 8}, {3, 13, 7}, {3, 14, 6}, {3, 15, 5}, {3, 16, 4}, {3, 17, 3}, {3, 18, 2}, {3, 19, 1}, {3, 20, 0}, {4, 0, 19}, {4, 1, 18}, {4, 2, 17}, {4, 3, 16}, {4, 4, 15}, {4, 5, 14}, {4, 6, 13}, {4, 7, 12}, {4, 8, 11}, {4, 9, 10}, {4, 10, 9}, {4, 11, 8}, {4, 12, 7}, {4, 13, 6}, {4, 14, 5}, {4, 15, 4}, {4, 16, 3}, {4, 17, 2}, {4, 18, 1}, {4, 19, 0}, {5, 0, 18}, {5, 1, 17}, {5, 2, 16}, {5, 3, 15}, {5, 4, 14}, {5, 5, 13}, {5, 6, 12}, {5, 7, 11}, {5, 8, 10}, {5, 9, 9}, {5, 10, 8}, {5, 11, 7}, {5, 12, 6}, {5, 13, 5}, {5, 14, 4}, {5, 15, 3}, {5, 16, 2}, {5, 17, 1}, {5, 18, 0}, {6, 0, 17}, {6, 1, 16}, {6, 2, 15}, {6, 3, 14}, {6, 4, 13}, {6, 5, 12}, {6, 6, 11}, {6, 7, 10}, {6, 8, 9}, {6, 9, 8}, {6, 10, 7}, {6, 11, 6}, {6, 12, 5}, {6, 13, 4}, {6, 14, 3}, {6, 15, 2}, {6, 16, 1}, {6, 17, 0}, {7, 0, 16}, {7, 1, 15}, {7, 2, 14}, {7, 3, 13}, {7, 4, 12}, {7, 5, 11}, {7, 6, 10}, {7, 7, 9}, {7, 8, 8}, {7, 9, 7}, {7, 10, 6}, {7, 11, 5}, {7, 12, 4}, {7, 13, 3}, {7, 14, 2}, {7, 15, 1}, {7, 16, 0}, {8, 0, 15}, {8, 1, 14}, {8, 2, 13}, {8, 3, 12}, {8, 4, 11}, {8, 5, 10}, {8, 6, 9}, {8, 7, 8}, {8, 8, 7}, {8, 9, 6}, {8, 10, 5}, {8, 11, 4}, {8, 12, 3}, {8, 13, 2}, {8, 14, 1}, {8, 15, 0}, {9, 0, 14}, {9, 1, 13}, {9, 2, 12}, {9, 3, 11}, {9, 4, 10}, {9, 5, 9}, {9, 6, 8}, {9, 7, 7}, {9, 8, 6}, {9, 9, 5}, {9, 10, 4}, {9, 11, 3}, {9, 12, 2}, {9, 13, 1}, {9, 14, 0}, {10, 0, 13}, {10, 1, 12}, {10, 2, 11}, {10, 3, 10}, {10, 4, 9}, {10, 5, 8}, {10, 6, 7}, {10, 7, 6}, {10, 8, 5}, {10, 9, 4}, {10, 10, 3}, {10, 11, 2}, {10, 12, 1}, {10, 13, 0}, {11, 0, 12}, {11, 1, 11}, {11, 2, 10}, {11, 3, 9}, {11, 4, 8}, {11, 5, 7}, {11, 6, 6}, {11, 7, 5}, {11, 8, 4}, {11, 9, 3}, {11, 10, 2}, {11, 11, 1}, {11, 12, 0}, {12, 0, 11}, {12, 1, 10}, {12, 2, 9}, {12, 3, 8}, {12, 4, 7}, {12, 5, 6}, {12, 6, 5}, {12, 7, 4}, {12, 8, 3}, {12, 9, 2}, {12, 10, 1}, {12, 11, 0}, {13, 0, 10}, {13, 1, 9}, {13, 2, 8}, {13, 3, 7}, {13, 4, 6}, {13, 5, 5}, {13, 6, 4}, {13, 7, 3}, {13, 8, 2}, {13, 9, 1}, {13, 10, 0}, {14, 0, 9}, {14, 1, 8}, {14, 2, 7}, {14, 3, 6}, {14, 4, 5}, {14, 5, 4}, {14, 6, 3}, {14, 7, 2}, {14, 8, 1}, {14, 9, 0}, {15, 0, 8}, {15, 1, 7}, {15, 2, 6}, {15, 3, 5}, {15, 4, 4}, {15, 5, 3}, {15, 6, 2}, {15, 7, 1}, {15, 8, 0}, {16, 0, 7}, {16, 1, 6}, {16, 2, 5}, {16, 3, 4}, {16, 4, 3}, {16, 5, 2}, {16, 6, 1}, {16, 7, 0}, {17, 0, 6}, {17, 1, 5}, {17, 2, 4}, {17, 3, 3}, {17, 4, 2}, {17, 5, 1}, {17, 6, 0}, {18, 0, 5}, {18, 1, 4}, {18, 2, 3}, {18, 3, 2}, {18, 4, 1}, {18, 5, 0}, {19, 0, 4}, {19, 1, 3}, {19, 2, 2}, {19, 3, 1}, {19, 4, 0}, {20, 0, 3}, {20, 1, 2}, {20, 2, 1}, {20, 3, 0}, {21, 0, 2}, {21, 1, 1}, {21, 2, 0}, {22, 0, 1}, {22, 1, 0}, {23, 0, 0},
    {0, 0, 24}, {0, 1, 23}, {0, 2, 22}, {0, 3, 21}, {0, 4, 20}, {0, 5, 19}, {0, 6, 18}, {0, 7, 17}, {0, 8, 16}, {0, 9, 15}, {0, 10, 14}, {0, 11, 13}, {0, 12, 12}, {0, 13, 11}, {0, 14, 10}, {0, 15, 9}, {0, 16, 8}, {0, 17, 7}, {0, 18, 6}, {0, 19, 5}, {0, 20, 4}, {0, 21, 3}, {0, 22, 2}, {0, 23, 1}, {0, 24, 0}, {1, 0, 23}, {1, 1, 22}, {1, 2, 21}, {1, 3, 20}, {1, 4, 19}, {1, 5, 18}, {1, 6, 17}, {1, 7, 16}, {1, 8, 15}, {1, 9, 14}, {1, 10, 13}, {1, 11, 12}, {1, 12, 11}, {1, 13, 10}, {1, 14, 9}, {1, 15, 8}, {1, 16, 7}, {1, 17, 6}, {1, 18, 5}, {1, 19, 4}, {1, 20, 3}, {1, 21, 2}, {1, 22, 1}, {1, 23, 0}, {2, 0, 22}, {2, 1, 21}, {2, 2, 20}, {2, 3, 19}, {2, 4, 18}, {2, 5, 17}, {2, 6, 16}, {2, 7, 15}, {2, 8, 14}, {2, 9, 13}, {2, 10, 12}, {2, 11, 11}, {2, 12, 10}, {2, 13, 9}, {2, 14, 8}, {2, 15, 7}, {2, 16, 6}, {2, 17, 5}, {2, 18, 4}, {2, 19, 3}, {2, 20, 2}, {2, 21, 1}, {2, 22, 0}, {3, 0, 21}, {3, 1, 20}, {3, 2, 19}, {3, 3, 18}, {3, 4, 17}, {3, 5, 16}, {3, 6, 15}, {3, 7, 14}, {3, 8, 13}, {3, 9, 12}, {3, 10, 11}, {3, 11, 10}, {3, 12, 9}, {3, 13, 8}, {3, 14, 7}, {3, 15, 6}, {3, 16, 5}, {3, 17, 4}, {3, 18, 3}, {3, 19, 2}, {3, 20, 1}, {3, 21, 0}, {4, 0, 20}, {4, 1, 19}, {4, 2, 18}, {4, 3, 17}, {4, 4, 16}, {4, 5, 15}, {4, 6, 14}, {4, 7, 13}, {4, 8, 12}, {4, 9, 11}, {4, 10, 10}, {4, 11, 9}, {4, 12, 8}, {4, 13, 7}, {4, 14, 6}, {4, 15, 5}, {4, 16, 4}, {4, 17, 3}, {4, 18, 2}, {4, 19, 1}, {4, 20, 0}, {5, 0, 19}, {5, 1, 18}, {5, 2, 17}, {5, 3, 16}, {5, 4, 15}, {5, 5, 14}, {5, 6, 13}, {5, 7, 12}, {5, 8, 11}, {5, 9, 10}, {5, 10, 9}, {5, 11, 8}, {5, 12, 7}, {5, 13, 6}, {5, 14, 5}, {5, 15, 4}, {5, 16, 3}, {5, 17, 2}, {5, 18, 1}, {5, 19, 0}, {6, 0, 18}, {6, 1, 17}, {6, 2, 16}, {6, 3, 15}, {6, 4, 14}, {6, 5, 13}, {6, 6, 12}, {6, 7, 11}, {6, 8, 10}, {6, 9, 9}, {6, 10, 8}, {6, 11, 7}, {6, 12, 6}, {6, 13, 5}, {6, 14, 4}, {6, 15, 3}, {6, 16, 2}, {6, 17, 1}, {6, 18, 0}, {7, 0, 17}, {7, 1, 16}, {7, 2, 15}, {7, 3, 14}, {7, 4, 13}, {7, 5, 12}, {7, 6, 11}, {7, 7, 10}, {7, 8, 9}, {7, 9, 8}, {7, 10, 7}, {7, 11, 6}, {7, 12, 5}, {7, 13, 4}, {7, 14, 3}, {7, 15, 2}, {7, 16, 1}, {7, 17, 0}, {8, 0, 16}, {8, 1, 15}, {8, 2, 14}, {8, 3, 13}, {8, 4, 12}, {8, 5, 11}, {8, 6, 10}, {8, 7, 9}, {8, 8, 8}, {8, 9, 7}, {8, 10, 6}, {8, 11, 5}, {8, 12, 4}, {8, 13, 3}, {8, 14, 2}, {8, 15, 1}, {8, 16, 0}, {9, 0, 15}, {9, 1, 14}, {9, 2, 13}, {9, 3, 12}, {9, 4, 11}, {9, 5, 10}, {9, 6, 9}, {9, 7, 8}, {9, 8, 7}, {9, 9, 6}, {9, 10, 5}, {9, 11, 4}, {9, 12, 3}, {9, 13, 2}, {9, 14, 1}, {9, 15, 0}, {10, 0, 14}, {10, 1, 13}, {10, 2, 12}, {10, 3, 11}, {10, 4, 10}, {10, 5, 9}, {10, 6, 8}, {10, 7, 7}, {10, 8, 6}, {10, 9, 5}, {10, 10, 4}, {10, 11, 3}, {10, 12, 2}, {10, 13, 1}, {10, 14, 0}, {11, 0, 13}, {11, 1, 12}, {11, 2, 11}, {11, 3, 10}, {11, 4, 9}, {11, 5, 8}, {11, 6, 7}, {11, 7, 6}, {11, 8, 5}, {11, 9, 4}, {11, 10, 3}, {11, 11, 2}, {11, 12, 1}, {11, 13, 0}, {12, 0, 12}, {12, 1, 11}, {12, 2, 10}, {12, 3, 9}, {12, 4, 8}, {12, 5, 7}, {12, 6, 6}, {12, 7, 5}, {12, 8, 4}, {12, 9, 3}, {12, 10, 2}, {12, 11, 1}, {12, 12, 0}, {13, 0, 11}, {13, 1, 10}, {13, 2, 9}, {13, 3, 8}, {13, 4, 7}, {13, 5, 6}, {13, 6, 5}, {13, 7, 4}, {13, 8, 3}, {13, 9, 2}, {13, 10, 1}, {13, 11, 0}, {14, 0, 10}, {14, 1, 9}, {14, 2, 8}, {14, 3, 7}, {14, 4, 6}, {14, 5, 5}, {14, 6, 4}, {14, 7, 3}, {14, 8, 2}, {14, 9, 1}, {14, 10, 0}, {15, 0, 9}, {15, 1, 8}, {15, 2, 7}, {15, 3, 6}, {15, 4, 5}, {15, 5, 4}, {15, 6, 3}, {15, 7, 2}, {15, 8, 1}, {15, 9, 0}, {16, 0, 8}, {16, 1, 7}, {16, 2, 6}, {16, 3, 5}, {16, 4, 4}, {16, 5, 3}, {16, 6, 2}, {16, 7, 1}, {16, 8, 0}, {17, 0, 7}, {17, 1, 6}, {17, 2, 5}, {17, 3, 4}, {17, 4, 3}, {17, 5, 2}, {17, 6, 1}, {17, 7, 0}, {18, 0, 6}, {18, 1, 5}, {18, 2, 4}, {18, 3, 3}, {18, 4, 2}, {18, 5, 1}, {18, 6, 0}, {19, 0, 5}, {19, 1, 4}, {19, 2, 3}, {19, 3, 2}, {19, 4, 1}, {19, 5, 0}, {20, 0, 4}, {20, 1, 3}, {20, 2, 2}, {20, 3, 1}, {20, 4, 0}, {21, 0, 3}, {21, 1, 2}, {21, 2, 1}, {21, 3, 0}, {22, 0, 2}, {22, 1, 1}, {22, 2, 0}, {23, 0, 1}, {23, 1, 0}, {24, 0, 0}};
//*/






} // namespace gansu::gpu

#endif