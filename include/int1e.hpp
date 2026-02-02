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



#ifndef INT1E_CUH
#define INT1E_CUH

#include <iostream>
#include "types.hpp"
#include "utils_cuda.hpp"
#include "compile_flag_int1e.hpp"


namespace gansu::gpu{

// MD and OS method(overlap and kinetic integral: up tp d orbital)
__global__ void overlap_kinetic_MDss(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDsp(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDpp(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDsd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDpd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDdd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);

__global__ void overlap_kinetic_OSss(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSsp(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSpp(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSsd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSpd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSdd(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);


// MD and OS method(nuclear attraction integral: up to d orbital)
__global__ void nuclear_attraction_MDss(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDsp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDpp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDsd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDpd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDdd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);

__global__ void nuclear_attraction_OSss(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSsp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSpp(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSsd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSpd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSdd(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);


// ENABLE_F_INT1E
#if INT1E_MAX_L >= 3
// F kernels
__global__ void overlap_kinetic_MDsf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDpf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDdf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDff(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void nuclear_attraction_MDsf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDpf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDdf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDff(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);

__global__ void overlap_kinetic_OSsf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSpf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSdf(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSff(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void nuclear_attraction_OSsf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSpf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSdf(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSff(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors,  const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);

#endif

// ENABLE_G_INT1E
#if INT1E_MAX_L >= 4
// G kernels
__global__ void overlap_kinetic_MDsg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDpg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDdg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDfg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_MDgg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void nuclear_attraction_MDsg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDpg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDdg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDfg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_MDgg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);

__global__ void overlap_kinetic_OSsg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSpg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSdg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSfg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void overlap_kinetic_OSgg(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
__global__ void nuclear_attraction_OSsg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSpg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSdg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSfg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);
__global__ void nuclear_attraction_OSgg(real_t* g_nucattr, const PrimitiveShell *g_shell, const real_t* g_cgto_normalization_factors, const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis, const real_t* g_boys_grid);

#endif // ENABLE_G_INT1E




// 2025-05-26 define a function to target matrices
__global__ void matrixSymmetrization(real_t* g_matrix, const int num_basis);

// f以上の軌道の場合，汎用カーネルで処理する
__global__
void compute_kinetic_energy_integral(real_t* g_overlap, real_t* g_kinetic, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                    const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, const size_t num_threads, const int num_basis);
// f以上の軌道の場合，汎用カーネルで処理する
__global__
void compute_nuclear_attraction_integral(real_t* g_nucattr, const PrimitiveShell* g_shell, const real_t* g_cgto_normalization_factors, 
                                        const Atom* g_atom, const int num_atoms, const ShellTypeInfo shell_s0, const ShellTypeInfo shell_s1, 
                                        const size_t num_threads, 
                                        const int num_basis, const real_t* g_boys_grid);


// define the kernel functions as function pointers for one electron integrals
using overlap_kinetic_kernel_t     = void (*)(real_t*, real_t*, const PrimitiveShell*, const real_t*, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int);
using nuclear_attraction_kernel_t = void (*)(real_t*, const PrimitiveShell*, const real_t*, const Atom*, const int, const ShellTypeInfo, const ShellTypeInfo, const size_t, const int, const real_t*);



inline int idx_ab(int a, int b){
    return b*(b+1)/2 + a;
}


inline overlap_kinetic_kernel_t MD_table[] = {
    /* 0: ss */ overlap_kinetic_MDss,
    /* 1: sp */ overlap_kinetic_MDsp,
    /* 2: pp */ overlap_kinetic_MDpp,
    /* 3: sd */ overlap_kinetic_MDsd,
    /* 4: pd */ overlap_kinetic_MDpd,
    /* 5: dd */ overlap_kinetic_MDdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ overlap_kinetic_MDsf,
    /* 7: pf */ overlap_kinetic_MDpf,
    /* 8: df */ overlap_kinetic_MDdf,
    /* 9: ff */ overlap_kinetic_MDff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ overlap_kinetic_MDsg,
    /* 11: pg */ overlap_kinetic_MDpg,
    /* 12: dg */ overlap_kinetic_MDdg,
    /* 13: fg */ overlap_kinetic_MDfg,
    /* 14: gg */ overlap_kinetic_MDgg,
#endif
};

inline overlap_kinetic_kernel_t OS_table[] = {
    /* 0: ss */ overlap_kinetic_OSss,
    /* 1: sp */ overlap_kinetic_OSsp,
    /* 2: pp */ overlap_kinetic_OSpp,
    /* 3: sd */ overlap_kinetic_OSsd,
    /* 4: pd */ overlap_kinetic_OSpd,
    /* 5: dd */ overlap_kinetic_OSdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ overlap_kinetic_OSsf,
    /* 7: pf */ overlap_kinetic_OSpf,
    /* 8: df */ overlap_kinetic_OSdf,
    /* 9: ff */ overlap_kinetic_OSff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ overlap_kinetic_OSsg,
    /* 11: pg */ overlap_kinetic_OSpg,
    /* 12: dg */ overlap_kinetic_OSdg,
    /* 13: fg */ overlap_kinetic_OSfg,
    /* 14: gg */ overlap_kinetic_OSgg,
#endif
};

inline overlap_kinetic_kernel_t Hybrid_table[] = {
    /* 0: ss */ overlap_kinetic_OSss,
    /* 1: sp */ overlap_kinetic_OSsp,
    /* 2: pp */ overlap_kinetic_OSpp,
    /* 3: sd */ overlap_kinetic_MDsd,
    /* 4: pd */ overlap_kinetic_OSpd,
    /* 5: dd */ overlap_kinetic_OSdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ overlap_kinetic_MDsf,
    /* 7: pf */ overlap_kinetic_OSpf,
    /* 8: df */ overlap_kinetic_OSdf,
    /* 9: ff */ overlap_kinetic_OSff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ overlap_kinetic_MDsg,
    /* 11: pg */ overlap_kinetic_OSpg,
    /* 12: dg */ overlap_kinetic_OSdg,
    /* 13: fg */ overlap_kinetic_OSfg,
    /* 14: gg */ overlap_kinetic_OSgg,
#endif
};


inline overlap_kinetic_kernel_t
get_overlap_kinetic_kernel(int a, int b, const std::string& int1e_method)
{
    const int idx = idx_ab(a,b);

    const overlap_kinetic_kernel_t* table = nullptr;

    if(int1e_method == "md"){
        // std::cout << int1e_method << std::endl;
        table = MD_table;
    }else if(int1e_method == "os"){
        // std::cout << int1e_method << std::endl;
        table = OS_table;
    }else { // "hybrid" or default
        // std::cout << int1e_method << std::endl;
        table = Hybrid_table;
    }

    constexpr int table_size = (INT1E_MAX_L + 1) * (INT1E_MAX_L + 2) / 2;

    if(idx < table_size) return table[idx];

    return compute_kinetic_energy_integral;
}



inline nuclear_attraction_kernel_t MD_nuc_table[] = {
    /* 0: ss */ nuclear_attraction_MDss,
    /* 1: sp */ nuclear_attraction_MDsp,
    /* 2: pp */ nuclear_attraction_MDpp,
    /* 3: sd */ nuclear_attraction_MDsd,
    /* 4: pd */ nuclear_attraction_MDpd,
    /* 5: dd */ nuclear_attraction_MDdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ nuclear_attraction_MDsf,
    /* 7: pf */ nuclear_attraction_MDpf,
    /* 8: df */ nuclear_attraction_MDdf,
    /* 9: ff */ nuclear_attraction_MDff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ nuclear_attraction_MDsg,
    /* 11: pg */ nuclear_attraction_MDpg,
    /* 12: dg */ nuclear_attraction_MDdg,
    /* 13: fg */ nuclear_attraction_MDfg,
    /* 14: gg */ nuclear_attraction_MDgg,
#endif
};

inline nuclear_attraction_kernel_t OS_nuc_table[] = {
    /* 0: ss */ nuclear_attraction_OSss,
    /* 1: sp */ nuclear_attraction_OSsp,
    /* 2: pp */ nuclear_attraction_OSpp,
    /* 3: sd */ nuclear_attraction_OSsd,
    /* 4: pd */ nuclear_attraction_OSpd,
    /* 5: dd */ nuclear_attraction_OSdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ nuclear_attraction_OSsf,
    /* 7: pf */ nuclear_attraction_OSpf,
    /* 8: df */ nuclear_attraction_OSdf,
    /* 9: ff */ nuclear_attraction_OSff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ nuclear_attraction_OSsg,
    /* 11: pg */ nuclear_attraction_OSpg,
    /* 12: dg */ nuclear_attraction_OSdg,
    /* 13: fg */ nuclear_attraction_OSfg,
    /* 14: gg */ nuclear_attraction_OSgg,
#endif
};

inline nuclear_attraction_kernel_t Hybrid_nuc_table[] = {
    /* 0: ss */ nuclear_attraction_OSss,
    /* 1: sp */ nuclear_attraction_OSsp,
    /* 2: pp */ nuclear_attraction_OSpp,
    /* 3: sd */ nuclear_attraction_OSsd,
    /* 4: pd */ nuclear_attraction_OSpd,
    /* 5: dd */ nuclear_attraction_OSdd,

#if INT1E_MAX_L >= 3
    /* 6: sf */ nuclear_attraction_OSsf,
    /* 7: pf */ nuclear_attraction_OSpf,
    /* 8: df */ nuclear_attraction_OSdf,
    /* 9: ff */ nuclear_attraction_MDff,
#endif

#if INT1E_MAX_L >= 4
    /* 10: sg */ nuclear_attraction_OSsg,
    /* 11: pg */ nuclear_attraction_OSpg,
    /* 12: dg */ nuclear_attraction_MDdg,
    /* 13: fg */ nuclear_attraction_MDfg,
    /* 14: gg */ nuclear_attraction_MDgg,
#endif
};


inline nuclear_attraction_kernel_t
get_nuclear_attraction_kernel(int a, int b, const std::string& int1e_method)
{
    const int idx = idx_ab(a, b);

    const nuclear_attraction_kernel_t* table = nullptr;

    if (int1e_method == "md") {
        // std::cout << int1e_method << std::endl;
        table = MD_nuc_table;
    }
    else if (int1e_method == "os") {
        // std::cout << int1e_method << std::endl;
        table = OS_nuc_table;
    }
    else { // "hybrid" or default
        // std::cout << int1e_method << std::endl;
        table = Hybrid_nuc_table;
    }

    constexpr int table_size = (INT1E_MAX_L + 1) * (INT1E_MAX_L + 2) / 2;

    if (idx < table_size) return table[idx];

    return compute_nuclear_attraction_integral;
}

} // namespace gansu::gpu

#endif
