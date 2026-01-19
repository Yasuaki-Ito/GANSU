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



#ifndef INT1E_CUH
#define INT1E_CUH

#include "types.hpp"
#include "utils_cuda.hpp"

namespace gansu::gpu{

// MD method(overlap and kinetic integral)
__global__
void overlap_kinetic_MDss(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__ 
void overlap_kinetic_MDsp(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__ 
void overlap_kinetic_MDpp(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDsd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDpd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDdd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDsf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDpf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDdf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_MDff(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

// OS method(overlap and kinetic integral)
__global__
void overlap_kinetic_OSss(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__ 
void overlap_kinetic_OSsp(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__ 
void overlap_kinetic_OSpp(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSsd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSpd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSdd(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSsf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSpf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSdf(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);

__global__
void overlap_kinetic_OSff(real_t* g_overlap, real_t* g_kinetic, 
                        const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                        const size_t num_threads,
                        const int num_basis);



// MD method(nuclear attraction integral)
__global__ 
void nuclear_attraction_MDss(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDsp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDpp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDsd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDpd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDdd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDsf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDpf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDdf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_MDff(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);


// OS method(nuclear attraction integral)
__global__ 
void nuclear_attraction_OSss(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSsp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSpp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSsd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSpd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSdd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSsf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSpf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSdf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

__global__ 
void nuclear_attraction_OSff(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  
                           const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1,
                           const size_t num_threads,
                           const int num_basis, const real_t* g_boys_grid);

// 2025-05-26 define a function to target matrices
//__global__ void Matrix_Symmetrization(real_t* g_matrix, const int num_basis);
__global__ void matrixSymmetrization(real_t* g_matrix, const int num_basis);

__global__
void compute_kinetic_energy_integral(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);

__global__
void compute_nuclear_attraction_integral(real_t* g_nucattr, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                        const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                                        const size_t num_threads, 
                                        const int num_basis, const real_t* g_boys_grid);


// define the kernel functions as function pointers for one electron integrals
using overlap_kinetic_kernel_t     = void (*)(real_t*, real_t*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int);
using nuclear_attraction_kernel_t = void (*)(real_t*, const PrimitiveShell*, const real_t*, const Atom*, const int, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int, const real_t*);


inline overlap_kinetic_kernel_t get_overlap_kinetic_kernel(int a, int b){
    int flag=0;

    if(flag){
        if(a==0 && b==0)      return overlap_kinetic_MDss;
        else if(a==0 && b==1) return overlap_kinetic_MDsp;
        else if(a==1 && b==1) return overlap_kinetic_MDpp;
        else if(a==0 && b==2) return overlap_kinetic_MDsd;
        else if(a==1 && b==2) return overlap_kinetic_MDpd;
        else if(a==2 && b==2) return overlap_kinetic_MDdd;
        else if(a==0 && b==3) return overlap_kinetic_MDsf;
        else if(a==1 && b==3) return overlap_kinetic_MDpf;
        else if(a==2 && b==3) return overlap_kinetic_MDdf;
        else if(a==3 && b==3) return overlap_kinetic_MDff;
        else return compute_kinetic_energy_integral;
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for overlap and kinetic integrals");
    }
    else{
        if(a==0 && b==0)      return overlap_kinetic_OSss;
        else if(a==0 && b==1) return overlap_kinetic_OSsp;
        else if(a==1 && b==1) return overlap_kinetic_OSpp;
        else if(a==0 && b==2) return overlap_kinetic_OSsd;
        else if(a==1 && b==2) return overlap_kinetic_OSpd;
        else if(a==2 && b==2) return overlap_kinetic_OSdd;
        else if(a==0 && b==3) return overlap_kinetic_OSsf;
        else if(a==1 && b==3) return overlap_kinetic_OSpf;
        else if(a==2 && b==3) return overlap_kinetic_OSdf;
        else if(a==3 && b==3) return overlap_kinetic_OSff;
        else return compute_kinetic_energy_integral;
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for overlap and kinetic integrals");
    }
}

inline nuclear_attraction_kernel_t get_nuclear_attraction_kernel(int a, int b){
    int flag=0;

    if(flag){
        if(a==0 && b==0)      return nuclear_attraction_MDss;
        else if(a==0 && b==1) return nuclear_attraction_MDsp;
        else if(a==1 && b==1) return nuclear_attraction_MDpp;
        else if(a==0 && b==2) return nuclear_attraction_MDsd;
        else if(a==1 && b==2) return nuclear_attraction_MDpd;
        else if(a==2 && b==2) return nuclear_attraction_MDdd;
        else if(a==0 && b==3) return nuclear_attraction_MDsf;
        else if(a==1 && b==3) return nuclear_attraction_MDpf;
        else if(a==2 && b==3) return nuclear_attraction_MDdf;
        else if(a==3 && b==3) return nuclear_attraction_MDff;
        else return compute_nuclear_attraction_integral;
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in MD method for nuclear attraction integrals");
    }
    else{
        if(a==0 && b==0)      return nuclear_attraction_OSss;
        else if(a==0 && b==1) return nuclear_attraction_OSsp;
        else if(a==1 && b==1) return nuclear_attraction_OSpp;
        else if(a==0 && b==2) return nuclear_attraction_OSsd;
        else if(a==1 && b==2) return nuclear_attraction_OSpd;
        else if(a==2 && b==2) return nuclear_attraction_OSdd;
        else if(a==0 && b==3) return nuclear_attraction_OSsf;
        else if(a==1 && b==3) return nuclear_attraction_OSpf;
        else if(a==2 && b==3) return nuclear_attraction_OSdf;
        else if(a==3 && b==3) return nuclear_attraction_OSff;
        else return compute_nuclear_attraction_integral;
        // else THROW_EXCEPTION("Only up to f-orbitals are supported in OS method for nuclear attraction integrals");
    }

}





} // namespace gansu::gpu

#endif
