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


#include <cuda.h>
#include <cmath>

#include "boys.hpp"
#include "types.hpp"
#include "utils_cuda.hpp"

#include "int2e.hpp"
#include "Et_functions.hpp"


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

namespace gansu::gpu{

// 二条階乗を返す関数
__device__ int factorial2gpu(int i){
    if(i<1){
        return 1;
    }else{
        return i*factorial2gpu(i-2);
    }
}

// 2点間の距離を求める関数（2乗済み）
__device__ double calc_dist_GPU(const Coordinate& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}
__device__ double calc_dist_GPU(const double3& coord1, const Coordinate& coord2){
    return (coord1.x-coord2.x)*(coord1.x-coord2.x) + (coord1.y-coord2.y)*(coord1.y-coord2.y) + (coord1.z-coord2.z)*(coord1.z-coord2.z);
}

__device__ double calc_Norms(double alpha, double beta, int ijk, int lmn){
    return pow(2.0, ijk+lmn) 
        * pow(2.0/M_PI, 1.5)
        * pow(alpha, (2.0*ijk+3.0)/4.0)
        * pow(beta, (2.0*lmn+3.0)/4.0);
}


//__global__ void Matrix_Symmetrization(double* matrix, int n){
//    __shared__ double sh_mem[32][33];
//
//    if(blockIdx.y > blockIdx.x) return;
//
//    int src_block = blockIdx.y*32*n + blockIdx.x*32;
//    int dst_block = blockIdx.x*32*n + blockIdx.y*32;
//
//    if(blockIdx.x*32+threadIdx.x < n || blockIdx.y*32+threadIdx.y < n){
//    //if(blockIdx.x*32+threadIdx.x < n && blockIdx.y*32+threadIdx.y < n){
//        sh_mem[threadIdx.y][threadIdx.x] = matrix[src_block + threadIdx.y*n+threadIdx.x];
//    }
//    __syncthreads();
//
//    if (blockIdx.y==blockIdx.x && threadIdx.y <= threadIdx.x || (dst_block + threadIdx.y*n+threadIdx.x >=n*n) ) return;
//
//    matrix[dst_block + threadIdx.y*n+threadIdx.x] = sh_mem[threadIdx.x][threadIdx.y];
//}

__global__ void matrixSymmetrization(double* g_matrix, const int num_basis) 
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mu = idx / num_basis;
    const int nu = idx % num_basis;

    if (mu < nu) {
        g_matrix[num_basis * nu + mu] = g_matrix[num_basis * mu + nu];
    }

}


__device__ int calc_result_index(int y, int x, int sumCGTO){
    return (y<=x) ? y*sumCGTO + x : x*sumCGTO + y;
}


// MD method
#include "MD_kernel.txt"
// OS method
#include "OS_kernel.txt"





// すべての行列要素にatomicAddを行う（汎用カーネルで使用）
__device__ void AddToResult(double result, double *g_V, int y, int x, int sumCGTO, bool flag){
    atomicAdd(&g_V[y*sumCGTO + x], result);
    if(flag){
        atomicAdd(&g_V[x*sumCGTO + y], result);
    }
}


// // normに球面調和に関わるマジックナンバーをかける
// __device__ inline double normalizationFactor(int l, int m, int n) {
//     int total = l + m + n;

//     if (total == 2) { // d-type
//         if(l == 2 || m == 2 || n == 2){
//             return D_NORMALIZATION_CONSTANT_2_INV;
//         }else{
//             return D_NORMALIZATION_CONSTANT_1_INV;
//         }

//     } else if (total == 3) { // f-type
//         if (l == 3 || m == 3 || n == 3)
//             return F_NORMALIZATION_CONSTANT_3_INV;
//         else if (l == 2 || m == 2 || n == 2)
//             return F_NORMALIZATION_CONSTANT_2_INV;
//         else
//             return F_NORMALIZATION_CONSTANT_1_INV;

//     } else if (total == 4) { // g-type
//         if (l == 4 || m == 4 || n == 4) // i.e., (4, 0, 0)
//             return G_NORMALIZATION_CONSTANT_4;
//         else if (l == 3 || m == 3 || n == 3) // i.e., (3, 1, 0)
//             return G_NORMALIZATION_CONSTANT_3_INV;
//         else if (l == 0 || m == 0 || n == 0) // i.e., (2, 2, 0)
//             return G_NORMALIZATION_CONSTANT_2_2_INV;
//         else
//             return G_NORMALIZATION_CONSTANT_2_INV;
//     }

//     // それ以外は補正なし（1.0を掛けるのと同じ）
//     return 1.0;
// }



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


__global__
void compute_kinetic_energy_integral(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                        const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis)
{
	const size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= num_threads) return;

	size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count); // Convert 1D index to 2D index a,b of [a|b]

	const size_t primitive_index_a = ab.y+shell_s0.start_index;
	const size_t primitive_index_b = ab.x+shell_s1.start_index;
	const PrimitiveShell a = g_shell[primitive_index_a];
	const PrimitiveShell b = g_shell[primitive_index_b];

	const size_t size_a = a.basis_index; // Obtain basis index (i|j)
	const size_t size_b = b.basis_index;

	const double p = a.exponent + b.exponent;

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

    double result_S;
    double result_K;
   
    double Norm_A, Norm_B;
    const double coefandNorm = a.coefficient * b.coefficient * M_PI/p * sqrt(M_PI/p) * exp(-(a.exponent*b.exponent/p)*calc_dist_GPU(a.coordinate, b.coordinate)) * g_cgto_normalization_factors[size_a]*g_cgto_normalization_factors[size_b];

    for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
        int l1=loop_to_ang[a.shell_type][lmn_a][0];
        int m1=loop_to_ang[a.shell_type][lmn_a][1]; 
        int n1=loop_to_ang[a.shell_type][lmn_a][2];
        Norm_A = calcNorm(a.exponent, l1, m1, n1);
        // Norm_A *= normalizationFactor(l1, m1, n1);
        for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){                  
            int l2=loop_to_ang[b.shell_type][lmn_b][0]; 
            int m2=loop_to_ang[b.shell_type][lmn_b][1]; 
            int n2=loop_to_ang[b.shell_type][lmn_b][2];
            Norm_B = calcNorm(b.exponent, l2, m2, n2);
            // Norm_B *= normalizationFactor(l2, m2, n2);

            result_S = coefandNorm * Norm_A * Norm_B * Et_GPU(l1, l2, 0, a.exponent, b.exponent, Dx) * Et_GPU(m1, m2, 0, a.exponent, b.exponent, Dy) * Et_GPU(n1, n2, 0, a.exponent, b.exponent, Dz);

            result_K = coefandNorm * Norm_A * Norm_B * (
                    ( (-2.0)*b.exponent*b.exponent*Et_GPU(l1, l2+2, 0, a.exponent, b.exponent, Dx) + b.exponent*(2*l2+1)*Et_GPU(l1, l2, 0, a.exponent, b.exponent, Dx) - ((l2*(l2-1))/2)*Et_GPU(l1, l2-2, 0, a.exponent, b.exponent, Dx) )*Et_GPU(m1, m2, 0, a.exponent, b.exponent, Dy)*Et_GPU(n1, n2, 0, a.exponent, b.exponent, Dz) +
                    Et_GPU(l1, l2, 0, a.exponent, b.exponent, Dx)*( (-2.0)*b.exponent*b.exponent*Et_GPU(m1, m2+2, 0, a.exponent, b.exponent, Dy) + b.exponent*(2*m2+1)*Et_GPU(m1, m2, 0, a.exponent, b.exponent, Dy) - ((m2*(m2-1))/2)*Et_GPU(m1, m2-2, 0, a.exponent, b.exponent, Dy) )*Et_GPU(n1, n2, 0, a.exponent, b.exponent, Dz) +
                    Et_GPU(l1, l2, 0, a.exponent, b.exponent, Dx)*Et_GPU(m1, m2, 0, a.exponent, b.exponent, Dy)*( (-2.0)*b.exponent*b.exponent*Et_GPU(n1, n2+2, 0, a.exponent, b.exponent, Dz) + b.exponent*(2*n2+1)*Et_GPU(n1, n2, 0, a.exponent, b.exponent, Dz) - ((n2*(n2-1))/2)*Et_GPU(n1, n2-2, 0, a.exponent, b.exponent, Dz))
            );

            // AddToResult(
            //     result_S, 
            //     g_overlap, 
            //     size_a+lmn_a, size_b+lmn_b, 
            //     num_basis, 
            //     primitive_index_a != primitive_index_b
            // );
            // AddToResult(
            //     result_K, 
            //     g_kinetic, 
            //     size_a+lmn_a, size_b+lmn_b, 
            //     num_basis, 
            //     primitive_index_a != primitive_index_b
            // );

            if( (a.shell_type == b.shell_type) && (size_a==size_b) && (lmn_a > lmn_b)) continue;

            atomicAdd(&g_overlap[calc_result_index(size_a+lmn_a, size_b+lmn_b, num_basis)], result_S*(1.0+int((size_a==size_b) && (primitive_index_a!=primitive_index_b))));
            atomicAdd(&g_kinetic[calc_result_index(size_a+lmn_a, size_b+lmn_b, num_basis)], result_K*(1.0+int((size_a==size_b) && (primitive_index_a!=primitive_index_b))));
        }
    }
}



__global__
void compute_nuclear_attraction_integral(real_t* g_nucattr, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                        const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                                        const size_t num_threads, 
                                        const int num_basis, const real_t* g_boys_grid)
{
	const size_t id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id >= num_threads) return;

	size_t2 ab = index1to2_one_electron(id, shell_s0.start_index == shell_s1.start_index, shell_s1.count); // Convert 1D index to 2D index a,b of [a|b]

	const size_t primitive_index_a = ab.y+shell_s0.start_index;
	const size_t primitive_index_b = ab.x+shell_s1.start_index;
	const PrimitiveShell a = g_shell[primitive_index_a];
	const PrimitiveShell b = g_shell[primitive_index_b];

	const size_t size_a = a.basis_index; // Obtain basis index (i|j)
	const size_t size_b = b.basis_index;

	const double p = a.exponent + b.exponent;

	const double3 P = make_double3((a.exponent*a.coordinate.x + b.exponent*b.coordinate.x)/p, (a.exponent*a.coordinate.y + b.exponent*b.coordinate.y)/p, (a.exponent*a.coordinate.z + b.exponent*b.coordinate.z)/p);

    const double Dx = a.coordinate.x - b.coordinate.x;
    const double Dy = a.coordinate.y - b.coordinate.y;
    const double Dz = a.coordinate.z - b.coordinate.z;

	const int K = a.shell_type + b.shell_type;
	double Boys[boys_one_size];

    double Et, Eu, Ev;

    double temp;
    double result_V = 0.0;
   
    double Norm_A, Norm_B;
    const double coefandNorm = a.coefficient * b.coefficient * ((2*M_PI)/p)* exp(-(a.exponent*b.exponent/p)*calc_dist_GPU(a.coordinate, b.coordinate)) * g_cgto_normalization_factors[size_a]*g_cgto_normalization_factors[size_b];

    double R_mid[3*size_one_Rmid];
    double R[size_one_R];

    for(int atom_index=0; atom_index<num_atoms; atom_index++){
        getIncrementalBoys(K, p*calc_dist_GPU(P, g_atom[atom_index].coordinate), g_boys_grid, Boys);
        for(int x=0; x <= K; x++){
            Boys[x] *= (right2left_binary_woif((-2*p), x));
        }

        for(int lmn_a=0; lmn_a<comb_max(a.shell_type); lmn_a++){
            int l1=loop_to_ang[a.shell_type][lmn_a][0];
            int m1=loop_to_ang[a.shell_type][lmn_a][1]; 
            int n1=loop_to_ang[a.shell_type][lmn_a][2];
            Norm_A = calcNorm(a.exponent, l1, m1, n1);
            // Norm_A *= normalizationFactor(l1, m1, n1);
            for(int lmn_b=0; lmn_b<comb_max(b.shell_type); lmn_b++){                 
                int l2=loop_to_ang[b.shell_type][lmn_b][0]; 
                int m2=loop_to_ang[b.shell_type][lmn_b][1]; 
                int n2=loop_to_ang[b.shell_type][lmn_b][2];
                Norm_B = calcNorm(b.exponent, l2, m2, n2);
                // Norm_B *= normalizationFactor(l2, m2, n2);

                if( (a.shell_type == b.shell_type) && (size_a==size_b) && (lmn_a > lmn_b)) continue;

                compute_R_TripleBuffer(R, R_mid, Boys, P, g_atom[atom_index].coordinate, K, l1+l2, m1+m2, n1+n2);

                // //Step 0: Boys関数評価
                // R[0]=Boys[0];
                // for(int i=0; i <= K; i++){
                //     R_mid[i]=Boys[i];
                // } 

                // //Step 1~Kの計算
                // for(int k=1; k <= K; k++){
                //     for(int z=0; z<=(K+1)*comb_max(k); z++){
                //         int i = z/comb_max(k);
                //         if(i <= K-k){
                //             int t = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][0];
                //             int u = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][1];
                //             int v = tuv_list[(k*(k+1)*(k+2))/6 + z%comb_max(k)][2];
                //             if((t <= l1+l2) && (u <= m1+m2) && (v <= n1+n2)){
                //                 if(t >= 1){
                //                     R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.x-g_atom[atom_index].coordinate.x)*R_mid[calc_Idx_Rmid(k-1,u,v,i+1,comb_max(k-1),size_one_Rmid)] + (t-1)*R_mid[calc_Idx_Rmid(k-2,u,v,i+1,comb_max(k-2),size_one_Rmid)];
                //                 }
                //                 else if(u >= 1){
                //                     R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.y-g_atom[atom_index].coordinate.y)*R_mid[calc_Idx_Rmid(k-1,u-1,v,i+1,comb_max(k-1),size_one_Rmid)] + (u-1)*R_mid[calc_Idx_Rmid(k-2,u-2,v,i+1,comb_max(k-2),size_one_Rmid)];
                //                 }
                //                 else{
                //                     R_mid[calc_Idx_Rmid(k,u,v,i,comb_max(k),size_one_Rmid)] = (P.z-g_atom[atom_index].coordinate.z)*R_mid[calc_Idx_Rmid(k-1,u,v-1,i+1,comb_max(k-1),size_one_Rmid)] + (v-1)*R_mid[calc_Idx_Rmid(k-2,u,v-2,i+1,comb_max(k-2),size_one_Rmid)];
                //                 }
                //             }
                //         }
                //     }

                //     //必要な結果を配列Rに書き込み
                //     for(int i=0; i<=comb_max(k); i++){
                //         R[static_cast<int>(k*(k+1)*(k+2)/6) + i] = R_mid[(k%3)*static_cast<int>(size_one_Rmid) + i];
                //     }
                // }

                result_V = 0.0;

                // Rの事前計算終了
                temp = 0.0;
                for(int t = 0; t < l1+l2+1; t++){
                    Et = Et_GPU(l1, l2, t, a.exponent, b.exponent, Dx);
                    for(int u = 0; u < m1+m2+1; u++){
                        Eu = Et_GPU(m1, m2, u, a.exponent, b.exponent, Dy);
                        for(int v = 0; v < n1+n2+1; v++){
                            Ev = Et_GPU(n1, n2, v, a.exponent, b.exponent, Dz);
                            int k = t + u + v;
                            temp += Et * Eu * Ev * R[k*(k+1)*(k+2)/6 + calc_Idx_Rmid(k,u,v,0,0,0)];
                        }
                    }
                }
                result_V = (-g_atom[atom_index].atomic_number) * coefandNorm * Norm_A * Norm_B * temp;


                // AddToResult(
                //     result_V, 
                //     g_nucattr, 
                //     size_a+lmn_a, size_b+lmn_b, 
                //     num_basis, 
                //     primitive_index_a != primitive_index_b
                // );

                atomicAdd(&g_nucattr[calc_result_index(size_a+lmn_a, size_b+lmn_b, num_basis)], result_V*(1.0+int((size_a==size_b) && (primitive_index_a!=primitive_index_b))));
            }
        }
    }
}



} // namespace gansu::gpu
