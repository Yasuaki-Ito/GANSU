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



#ifndef GRADIENTS
#define GRADIENTS


#include "int2e.hpp"
#include "Et_functions.hpp"
#include "Et_grad_functions.hpp"
#include "parameters.h"

// 1電子部分・・・i軌道の場合，13(6+6+1)
#ifndef boys_one_size
    #define boys_one_size 13
#endif

#ifndef size_one_R
    #define size_one_R 455
#endif

#ifndef size_one_Rmid
    #define size_one_Rmid 225
#endif


// 2電子部分・・・i軌道の場合，25(12+12+1)
#ifndef boys_size
    #define boys_size 25
#endif

#ifndef size_Rmid
    #define size_Rmid 1377
#endif


#ifndef size_R
    #define size_R 2925
#endif


namespace gansu::gpu{

__global__ void printGradientMatrix_Kernel(const double* g_grad, int num_atoms);

// define the kernel to calculate nuclear repulsion gradient
__global__ void compute_nuclear_repulsion_gradient_kernel(double* g_grad, const Atom* g_atom, const int num_atoms);


// define the kernel to calculate W matrix for overlap gradient
__global__ void compute_W_Matrix_kernel(real_t* d_W_matrix, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_electron, const int num_basis);


// define the kernel to calculate the gradients of moliucular integrals
__global__ void compute_gradients_overlap(double* g_gradients, const real_t* g_W_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads);
__global__ void compute_gradients_kinetic(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads);
__global__ void compute_gradients_nuclear(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const int num_basis, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads, const real_t* g_boys_grid);


// define the kernel to calculate the gradient of the two-electron part
__global__ void compute_gradients_two_electron(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const int num_basis, const double* g_boys_grid);
__global__ void Rys_compute_gradients_two_electron(double* g_gradients, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const int num_basis, const double* g_boys_grid);

// UHF version: takes alpha and beta density matrices separately
__global__ void compute_gradients_two_electron_uhf(double* g_gradients, const real_t* g_density_matrix_a, const real_t* g_density_matrix_b, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const int num_basis, const double* g_boys_grid);
__global__ void Rys_compute_gradients_two_electron_uhf(double* g_gradients, const real_t* g_density_matrix_a, const real_t* g_density_matrix_b, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const ShellTypeInfo shell_s2, const ShellTypeInfo shell_s3, const size_t num_threads, const int num_basis, const double* g_boys_grid);


// define the kernel functions as function poconst inters for one electron const integrals
using compute_basis_deriv_overlap = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const int, ShellTypeInfo, ShellTypeInfo, const size_t);
using compute_basis_deriv_kinetic = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const int, ShellTypeInfo, ShellTypeInfo, const size_t);
using compute_basis_deriv_nuclear = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const Atom*, const int, const int, ShellTypeInfo, ShellTypeInfo, const size_t, const real_t*);

using compute_basis_deriv_repulsion = void (*)(double*, const real_t*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int, const double*);

using compute_basis_deriv_repulsion_uhf = void (*)(double*, const real_t*, const real_t*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int, const double*);




// 2点間の距離を求める関数（2乗済み）・・・関数のオーバーロードを使用
__device__ __forceinline__ double calc_dist_GPU(const Coordinate& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}

__device__ __forceinline__ double calc_dist_GPU(const double3& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}

__device__ __forceinline__ double calc_dist_GPU(const double3& coord1, const double3& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}



// 該当箇所に排他的に加算する関数
inline 
__device__ void AddToResult(double* g_result, size_t index, double result, bool flag) {
    double val = flag ? 2.0 * result : result;
    atomicAdd(&g_result[index], val);
}


// TEI・・・Two Electron Integral
inline __device__
void AddToResult_TEI(double* g_result, size_t index, double result, bool sym_bra, bool sym_ket, bool sym_braket) {
    int f = 1 + static_cast<int>(!sym_bra) + static_cast<int>(!sym_ket) + static_cast<int>(!sym_bra && !sym_ket) + static_cast<int>(!sym_braket) * ( 1 + static_cast<int>(!sym_bra) + static_cast<int>(!sym_ket) + static_cast<int>(!sym_bra && !sym_ket) );
    atomicAdd(&g_result[index], result * f);
}

// // 係数部E_t(i,l)を計算する関数
inline 
__device__ double Et_GPU_Recursion(int i, int l, int t, const double alpha, const double beta, const double dist){
    if(t==0 && i==0 && l==0){
        // return 1.0;
        return exp(-(alpha*beta/(alpha + beta))*dist*dist);  
    }else if(t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else if(i>0){ // iに関して求める
        return 1/(2*(alpha+beta))*Et_GPU_Recursion(i-1, l, t-1, alpha, beta, dist) - (beta * dist/(alpha+beta))*Et_GPU_Recursion(i-1, l, t, alpha, beta, dist) + (t+1)*Et_GPU_Recursion(i-1, l, t+1, alpha, beta, dist);
    }else{ // lに関して求める
        return 1/(2*(alpha+beta))*Et_GPU_Recursion(i, l-1, t-1, alpha, beta, dist) + (alpha * dist / (alpha+beta))*Et_GPU_Recursion(i, l-1, t, alpha, beta, dist) + (t+1)*Et_GPU_Recursion(i, l-1, t+1, alpha, beta, dist);
    }
}


// // 係数部E_t(i,l)の微分に関する影響を計算する関数
inline 
__device__ double Et_GPU_gradients(int i, int l, int t, const double alpha, const double beta, const double dist){
    if(t==0 && i==0 && l==0){
        return -2*alpha*beta/(alpha+beta)*dist*exp(-(alpha*beta/(alpha + beta))*dist*dist);
    }else if(t<0 || i+l<t){ // 範囲外の処理
        return 0.0;
    }else if(i>0){ // iに関して求める
        return 1/(2*(alpha+beta))*Et_GPU_gradients(i-1, l, t-1, alpha, beta, dist) - (beta * dist/(alpha+beta))*Et_GPU_gradients(i-1, l, t, alpha, beta, dist) - (beta/(alpha+beta))*Et_GPU_Recursion(i-1, l, t, alpha, beta, dist) + (t+1)*Et_GPU_gradients(i-1, l, t+1, alpha, beta, dist);
    }else{ // lに関して求める
        return 1/(2*(alpha+beta))*Et_GPU_gradients(i, l-1, t-1, alpha, beta, dist) + (alpha * dist/(alpha+beta))*Et_GPU_gradients(i, l-1, t, alpha, beta, dist) + (alpha/(alpha+beta))*Et_GPU_Recursion(i, l-1, t, alpha, beta, dist) + (t+1)*Et_GPU_gradients(i, l-1, t+1, alpha, beta, dist);
    }
}


// R(t,u,v)の計算
inline 
__device__ double R_GPU_Recursion(int n, int t, int u, int v, const double3& P, const Coordinate& atom_pos, double* Boys){
    if(t==0 and u==0 and v==0){
        return Boys[n];
    }else if(t>0){
        return ((t-1)*R_GPU_Recursion(n+1, t-2, u, v, P, atom_pos, Boys) + (P.x-atom_pos.x)*R_GPU_Recursion(n+1, t-1, u, v, P, atom_pos, Boys));
    }else if(u>0){
        return ((u-1)*R_GPU_Recursion(n+1, t, u-2, v, P, atom_pos, Boys) + (P.y-atom_pos.y)*R_GPU_Recursion(n+1, t, u-1, v, P, atom_pos, Boys));
    }else if(v>0){
        return ((v-1)*R_GPU_Recursion(n+1, t, u, v-2, P, atom_pos, Boys) + (P.z-atom_pos.z)*R_GPU_Recursion(n+1, t, u, v-1, P, atom_pos, Boys));
    }else{
        return 0.0;
    }
}


inline 
__device__ double R_GPU_Recursion(int n, int t, int u, int v, const double3& P, const double3& Q, double* Boys){
    if(t==0 and u==0 and v==0){
        return Boys[n];
    }else if(t>0){
        return ((t-1)*R_GPU_Recursion(n+1, t-2, u, v, P, Q, Boys) + (P.x - Q.x)*R_GPU_Recursion(n+1, t-1, u, v, P, Q, Boys));
    }else if(u>0){
        return ((u-1)*R_GPU_Recursion(n+1, t, u-2, v, P, Q, Boys) + (P.y - Q.y)*R_GPU_Recursion(n+1, t, u-1, v, P, Q, Boys));
    }else if(v>0){
        return ((v-1)*R_GPU_Recursion(n+1, t, u, v-2, P, Q, Boys) + (P.z - Q.z)*R_GPU_Recursion(n+1, t, u, v-1, P, Q, Boys));
    }else{
        return 0.0;
    }
}


// MD法のRの再帰関係をトリプルバッファリングで計算
inline 
__device__ void compute_R_TripleBuffer(real_t* R, real_t* R_mid, const real_t* Boys, const double3& P, const Coordinate& coord, const int K, const int t_max, const int u_max, const int v_max){
    //Step 0: Boys関数評価
    R[0]=Boys[0];
    for(int i=0; i <= K; i++){
        R_mid[i]=Boys[i];
    } 

    //Step 1~Kの計算
    for(int k=1; k <= K; k++){
        for(int z=0; z<=(K+1)*comb_max(k); z++){
            int i = z/comb_max(k);
            if(i <= K-k){
                int t = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                int u = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                int v = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
                if((t <= t_max) && (u <= u_max) && (v <= v_max)){
                    if(t >= 1){
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.x-coord.x)*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_one_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_one_Rmid)];
                    }
                    else if(u >= 1){
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.y-coord.y)*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_one_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_one_Rmid)];
                    }
                    else{
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.z-coord.z)*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_one_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_one_Rmid)];
                    }
                }
            }
        }

        //必要な結果を配列Rに書き込み
        for(int i=0; i<=comb_max(k); i++){
            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_one_Rmid) + i];
        }
    }
}


inline 
__device__ void compute_R_TripleBuffer(double* R, double* R_mid, const double* Boys, const double3& P, const double3& Q, const int K, const int t_max, const int u_max, const int v_max){
    //Step 0: Boys関数評価
    R[0]=Boys[0];
    for(int i=0; i <= K; i++){
        R_mid[i]=Boys[i];
    }

    //Step 1~Kの計算
    for(int k=1; k <= K; k++){
        for(int z=0; z<=(K+1)*comb_max(k); z++){
            int i = z/comb_max(k);
            if(i <= K-k){
                int t = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                int u = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                int v = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
                if((t <= t_max) && (u <= u_max) && (v <= v_max)){
                    if(t >= 1){
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (P.x - Q.x)*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_Rmid)];
                    }
                    else if(u >= 1){
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (P.y - Q.y)*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_Rmid)];
                    }
                    else{
                        R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_Rmid)] = (P.z - Q.z)*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_Rmid)];
                    }
                }
            }
        }

        //必要な結果を配列Rに書き込み
        for(int i=0; i<=comb_max(k); i++){
            R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_Rmid) + i];
        }

    }
}





inline compute_basis_deriv_overlap get_compute_gradients_overlap() {
    return compute_gradients_overlap;
}

inline compute_basis_deriv_kinetic get_compute_gradients_kinetic() {
    return compute_gradients_kinetic;
}

inline compute_basis_deriv_nuclear get_compute_gradients_nuclear() {
    return compute_gradients_nuclear;
}

inline compute_basis_deriv_repulsion get_compute_gradients_repulsion() {
    return Rys_compute_gradients_two_electron;
}

inline compute_basis_deriv_repulsion_uhf get_compute_gradients_repulsion_uhf() {
    return Rys_compute_gradients_two_electron_uhf;
}


// Hessian kernels (1-electron)
__global__ void compute_hessian_overlap(double* g_hessian, const real_t* g_W_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const int num_basis, const int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads);
__global__ void compute_hessian_kinetic(double* g_hessian, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const int num_basis, const int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads);
__global__ void compute_hessian_nuclear_attraction(double* g_hessian, const real_t* g_density_matrix, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atoms, const int num_basis, const int num_atoms, ShellTypeInfo shell_s0, ShellTypeInfo shell_s1, const size_t num_threads, const double* g_boys_grid);
__global__ void compute_hessian_nuclear_repulsion(double* g_hessian, const Atom* g_atoms, const int num_atoms);

} // namespace gansu::gpu

#endif