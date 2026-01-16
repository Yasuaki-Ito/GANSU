#include <iomanip>
#include <iostream>
#include <assert.h>
#include "rhf.hpp"
#include "eri_stored.hpp"

namespace gansu {


// Defined in eri_stored_mp4.cu
void transform_ao_eri_to_mo_eri_full(const double* d_eri_ao, const double* d_C, int nao, double* d_eri_mo);
__global__ void mp2_moeri_kernel(const double* __restrict__ eri_mo, const double* __restrict__ eps, const int num_basis, const int occ, double* __restrict__ E_out);
__global__ void mp3_moeri_4h2p_kernel(const double* __restrict__ eri_mo, const double* __restrict__ eps, const int num_basis, const int occ,  double* __restrict__ E_out);
__global__ void mp3_moeri_2h4p_kernel(const double* __restrict__ eri_mo, const double* __restrict__ eps, const int num_basis, const int occ, double* __restrict__ E_out);
__global__ void mp3_moeri_3h3p_kernel(const double* __restrict__ eri_mo, const double* __restrict__ eps,  const int num_basis, const int occ, double* __restrict__ E_out);




/*



// <p0 p1||p2 p3><q0 q1||q2 q3><r0 r1||r2 r3><s0 s1||s2 s3> / (eps_t0t1^t2t3 * eps_u0^u1 * eps_v0v1^v2v3)
__device__ __forceinline__ real_t contribution_mp4_single(const real_t* __restrict__ eri_mo,
                                               const real_t* __restrict__ orbital_energies,
                                               const int num_basis,
                                               const int p0, const int p1, const int p2, const int p3,
                                               const int q0, const int q1, const int q2, const int q3,
                                               const int r0, const int r1, const int r2, const int r3,
                                               const int s0, const int s1, const int s2, const int s3,
                                               const int t0, const int t1, const int t2, const int t3,
                                               const int u0, const int u1,
                                               const int v0, const int v1, const int v2, const int v3) 
{
    // compute antisymmetrized ERIs
    const real_t eri_p = antisym_eri(eri_mo, num_basis, p0, p1, p2, p3);
    const real_t eri_q = antisym_eri(eri_mo, num_basis, q0, q1, q2, q3);
    const real_t eri_r = antisym_eri(eri_mo, num_basis, r0, r1, r2, r3);
    const real_t eri_s = antisym_eri(eri_mo, num_basis, s0, s1, s2, s3);

    // compute energy denominators
    const real_t eps_t = (orbital_energies[t0/2] + orbital_energies[t1/2] - orbital_energies[t2/2] - orbital_energies[t3/2]);
    const real_t eps_u = (orbital_energies[u0/2] - orbital_energies[u1/2]);
    const real_t eps_v = (orbital_energies[v0/2] + orbital_energies[v1/2] - orbital_energies[v2/2] - orbital_energies[v3/2]);
    const real_t denom = eps_t * eps_u * eps_v;

    // compute contribution
    const real_t contribution = eri_p * eri_q * eri_r * eri_s / denom;

    return contribution;
}

// <p0 p1||p2 p3><q0 q1||q2 q3><r0 r1||r2 r3><s0 s1||s2 s3> / (eps_t0t1^t2t3 * eps_u0u1^u2u3 * eps_v0v1^v2v3)
__device__ __forceinline__ real_t contribution_mp4_double(const real_t* __restrict__ eri_mo,
                                               const real_t* __restrict__ orbital_energies,
                                               const int num_basis,
                                               const int p0, const int p1, const int p2, const int p3,
                                               const int q0, const int q1, const int q2, const int q3,
                                               const int r0, const int r1, const int r2, const int r3,
                                               const int s0, const int s1, const int s2, const int s3,
                                               const int t0, const int t1, const int t2, const int t3,
                                               const int u0, const int u1, const int u2, const int u3,
                                               const int v0, const int v1, const int v2, const int v3) 
{
    // compute antisymmetrized ERIs
    const real_t eri_p = antisym_eri(eri_mo, num_basis, p0, p1, p2, p3);
    const real_t eri_q = antisym_eri(eri_mo, num_basis, q0, q1, q2, q3);
    const real_t eri_r = antisym_eri(eri_mo, num_basis, r0, r1, r2, r3);
    const real_t eri_s = antisym_eri(eri_mo, num_basis, s0, s1, s2, s3);

    // compute energy denominators
    const real_t eps_t = (orbital_energies[t0/2] + orbital_energies[t1/2] - orbital_energies[t2/2] - orbital_energies[t3/2]);
    const real_t eps_u = (orbital_energies[u0/2] + orbital_energies[u1/2] - orbital_energies[u2/2] - orbital_energies[u3/2]);
    const real_t eps_v = (orbital_energies[v0/2] + orbital_energies[v1/2] - orbital_energies[v2/2] - orbital_energies[v3/2]);
    const real_t denom = eps_t * eps_u * eps_v;

    // compute contribution
    const real_t contribution = eri_p * eri_q * eri_r * eri_s / denom;
    
    return contribution;
}

// <p0 p1||p2 p3><q0 q1||q2 q3><r0 r1||r2 r3><s0 s1||s2 s3> / (eps_t0t1^t2t3 * eps_u0u1u2^u3u4u5 * eps_v0v1^v2v3)
__device__ __forceinline__ real_t contribution_mp4_triple(const real_t* __restrict__ eri_mo,
                                               const real_t* __restrict__ orbital_energies,
                                               const int num_basis,
                                               const int p0, const int p1, const int p2, const int p3,
                                               const int q0, const int q1, const int q2, const int q3,
                                               const int r0, const int r1, const int r2, const int r3,
                                               const int s0, const int s1, const int s2, const int s3,
                                               const int t0, const int t1, const int t2, const int t3,
                                               const int u0, const int u1, const int u2, const int u3, const int u4, const int u5,
                                               const int v0, const int v1, const int v2, const int v3) 
{
    // compute antisymmetrized ERIs
    const real_t eri_p = antisym_eri(eri_mo, num_basis, p0, p1, p2, p3);
    const real_t eri_q = antisym_eri(eri_mo, num_basis, q0, q1, q2, q3);
    const real_t eri_r = antisym_eri(eri_mo, num_basis, r0, r1, r2, r3);
    const real_t eri_s = antisym_eri(eri_mo, num_basis, s0, s1, s2, s3);

    // compute energy denominators
    const real_t eps_t = (orbital_energies[t0/2] + orbital_energies[t1/2] - orbital_energies[t2/2] - orbital_energies[t3/2]);
    const real_t eps_u = (orbital_energies[u0/2] + orbital_energies[u1/2] + orbital_energies[u2/2] - orbital_energies[u3/2] - orbital_energies[u4/2] - orbital_energies[u5/2]);
    const real_t eps_v = (orbital_energies[v0/2] + orbital_energies[v1/2] - orbital_energies[v2/2] - orbital_energies[v3/2]);
    const real_t denom = eps_t * eps_u * eps_v;

    // compute contribution
    const real_t contribution = eri_p * eri_q * eri_r * eri_s / denom;
    
    return contribution;
}


// <p0 p1||p2 p3><q0 q1||q2 q3><r0 r1||r2 r3><s0 s1||s2 s3> / (eps_t0t1^t2t3 * eps_u0u1u2u3^u4u5u6u7 * eps_v0v1^v2v3)
__device__ __forceinline__ real_t contribution_mp4_quadruple(const real_t* __restrict__ eri_mo,
                                               const real_t* __restrict__ orbital_energies,
                                               const int num_basis,
                                               const int p0, const int p1, const int p2, const int p3,
                                               const int q0, const int q1, const int q2, const int q3,
                                               const int r0, const int r1, const int r2, const int r3,
                                               const int s0, const int s1, const int s2, const int s3,
                                               const int t0, const int t1, const int t2, const int t3,
                                               const int u0, const int u1, const int u2, const int u3, const int u4, const int u5, const int u6, const int u7,
                                               const int v0, const int v1, const int v2, const int v3) 
{
    // compute antisymmetrized ERIs
    const real_t eri_p = antisym_eri(eri_mo, num_basis, p0, p1, p2, p3);
    const real_t eri_q = antisym_eri(eri_mo, num_basis, q0, q1, q2, q3);
    const real_t eri_r = antisym_eri(eri_mo, num_basis, r0, r1, r2, r3);
    const real_t eri_s = antisym_eri(eri_mo, num_basis, s0, s1, s2, s3);

    // compute energy denominators
    const real_t eps_t = (orbital_energies[t0/2] + orbital_energies[t1/2] - orbital_energies[t2/2] - orbital_energies[t3/2]);
    const real_t eps_u = (orbital_energies[u0/2] + orbital_energies[u1/2] + orbital_energies[u2/2] + orbital_energies[u3/2] - orbital_energies[u4/2] - orbital_energies[u5/2] - orbital_energies[u6/2] - orbital_energies[u7/2]);
    const real_t eps_v = (orbital_energies[v0/2] + orbital_energies[v1/2] - orbital_energies[v2/2] - orbital_energies[v3/2]);
    const real_t denom = eps_t * eps_u * eps_v;

    // compute contribution
    const real_t contribution = eri_p * eri_q * eri_r * eri_s / denom;
    
    return contribution;
}


__global__ void compute_mp4_2h6p_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_2h6p)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int f_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int f = f_ + num_spin_occ;
        int e = e_ + num_spin_occ;
        int d = d_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // E2,1
        contrib += (1.0) / (16.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, b,
                                         e, f, c, d,
                                         i, j, e, f,
                                         i, j, a, b,
                                         i, j, c, d, 
                                         i, j, e, f);


    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_2h6p, block_sum);
    }
}



__global__ void compute_mp4_3h5p_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_3h5p)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int e_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int e = e_ + num_spin_occ;
        int d = d_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // E1,1
        contrib += (1.0) / (4.0) * contribution_mp4_single(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, b,
                                         d, e, c, k,
                                         j, k, d, e,
                                         i, j, a, b,
                                         j, c, 
                                         j, k, d, e);

        // E2,11
        contrib += (1.0) / (2.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, k,
                                         d, e, b, c,
                                         j, k, d, e,
                                         i, j, a, b,
                                         j, k, b, c, 
                                         j, k, d, e);

        // E2,12
        contrib += (1.0) / (2.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, b,
                                         i, e, c, k,
                                         j, k, d, e,
                                         i, j, a, b,
                                         i, j, c, d, 
                                         j, k, d, e);

        // E3,8
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, d, i, j,
                                         b, c, a, k,
                                         i, e, b, c,
                                         j, k, d, e,
                                         i, j, a, d,
                                         i, j, k, b, c, d, 
                                         j, k, d, e);

        // E3,10
        contrib += (1.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, k,
                                         i, e, b, c,
                                         j, k, d, e,
                                         i, j, a, b,
                                         i, j, k, b, c, d, 
                                         j, k, d, e);

        // E3,11
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, k,
                                         k, e, b, c,
                                         j, j, d, e,
                                         i, j, a, b,
                                         i, j, k, b, c, d, 
                                         i, j, d, e);

        // E3,15
        contrib += (1.0) / (4.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, d, i, j,
                                         b, c, a, k,
                                         k, e, b, c,
                                         j, j, d, e,
                                         i, j, a, d,
                                         i, j, k, b, c, d, 
                                         i, j, d, e);


    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_3h5p, block_sum);
    }
}



__global__ void compute_mp4_4h4p_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_4h4p)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;


    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int d = d_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // E1,3
        contrib += - (1.0) / (4.0) * contribution_mp4_single(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, j, a, k,
                                         c, d, b, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         k, b, 
                                         k, l, c, d);

        // E1,4
        contrib += - (1.0) / (4.0) * contribution_mp4_single(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, b,
                                         j, d, k, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         j, c, 
                                         k, l, c, d);

        // E2,3
        contrib += (1.0) / (16.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, b,
                                         i, j, k, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         i, j, c, d, 
                                         k, l, c, d);

        // E2,5
        contrib += - (1.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, k,
                                         k, d, b, l,
                                         j, l, c, d,
                                         i, j, a, b,
                                         j, k, b, c, 
                                         j, l, c, d);

        // E2,6
        contrib += (1.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, k,
                                         j, d, b, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         j, k, b, c, 
                                         k, l, c, d);

        // E2,8
        contrib += - (1.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         i, b, a, k,
                                         j, d, b, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         j, k, b, c, 
                                         k, l, c, d);

        // E2,9
        contrib += (1.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         i, b, a, k,
                                         k, d, b, l,
                                         j, l, c, d,
                                         i, j, a, c,
                                         j, k, b, c, 
                                         j, l, c, d);

        // E2,10
        contrib += (1.0) / (16.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, j, k, l,
                                         c, d, a, b,
                                         k, l, c, d,
                                         i, j, a, b,
                                         k, l, a, b, 
                                         k, l, c, d);

        // E3,1
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         i, b, k, l,
                                         j, d, a, b,
                                         k, l, c, d,
                                         i, j, a, c,
                                         j, k, l, a, b, c, 
                                         k, l, c, d);

        // E3,2
        contrib += - (1.0) / (4.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, k, l,
                                         j, d, a, b,
                                         k, l, c, d,
                                         i, j, a, b,
                                         j, k, l, a, b, c, 
                                         k, l, c, d);

        // E3,3
        contrib += - (1.0) / (4.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, k,
                                         i, j, b, l,
                                         k, l, c, d,
                                         i, j, a, b,
                                         i, j, k, b, c, d, 
                                         k, l, c, d);

        // E3,5
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, a, k,
                                         i, k, b, l,
                                         j, l, c, d,
                                         i, j, a, b,
                                         i, j, k, b, c, d, 
                                         j, l, c, d);

        // E3,9
        contrib += - (1.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         b, d, a, k,
                                         i, k, b, l,
                                         j, l, c, d,
                                         i, j, a, c,
                                         j, k, l, b, c, d, 
                                         j, l, c, d);

        // E3,12
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         b, d, a, k,
                                         i, j, b, l,
                                         k, l, c, d,
                                         i, j, a, c,
                                         i, j, k, a, b, c, 
                                         k, l, c, d);

        // E3,14
        contrib += - (1.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, c, i, j,
                                         i, b, k, l,
                                         k, d, a, b,
                                         j, l, c, d,
                                         i, j, a, c,
                                         j, k, l, a, b, c, 
                                         j, l, c, d);

        // E3,16
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, k, l,
                                         k, d, a, b,
                                         j, l, c, d,
                                         i, j, a, b,
                                         j, k, l, a, b, c, 
                                         j, l, c, d);

        // E4,1
        contrib += (1.0) / (16.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         c, d, i, j,
                                         a, b, k, l,
                                         i, j, a, b,
                                         k, l, c, d,
                                         i, j, c, d,
                                         i, j, k, l, a, b, c, d,
                                         k, l, c, d);

        // E4,2
        contrib += (1.0) / (16.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         k, l, a, b,
                                         i, j, c, d,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         i, j, c, d);

        // E4,3
        contrib += - (1.0) / (4.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         i, j, a, c,
                                         k, l, b, d,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         k, l, b, d);

        // E4,4
        contrib += - (1.0) / (4.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         k, l, b, d,
                                         i, j, a, c,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         i, j, a, c);

        // E4,5
        contrib += - (1.0) / (4.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         j, l, a, b,
                                         i, k, c, d,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         i, k, c, d);

        // E4,6
        contrib += - (1.0) / (4.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         i, k, c, d,
                                         j, l, a, b,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         j, l, a, b);

        // E4,7
        contrib += (1.0) * contribution_mp4_quadruple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         c, d, k, l,
                                         j, l, b, d,
                                         i, k, a, c,
                                         i, j, a, b,
                                         i, j, k, l, a, b, c, d,
                                         i, k, a, c);

    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_4h4p, block_sum);
    }
}



__global__ void compute_mp4_5h3p_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_5h3p)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int m  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int l  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // E1,2
        contrib += (1.0) / (4.0) * contribution_mp4_single(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, j, a, k,
                                         k, c, l, m,
                                         l, m, b, c,
                                         i, j, a, b,
                                         k, b, 
                                         l, m, b, c);

        // E2,2
        contrib += (1.0) / (2.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, j, k, l,
                                         k, c, a, m,
                                         l, m, b, c,
                                         i, j, a, b,
                                         k, l, a, b, 
                                         l, m, b, c);

        // E2,4
        contrib += (1.0) / (2.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, a, k,
                                         j, k, l, m,
                                         l, m, b, c,
                                         i, j, a, b,
                                         j, k, b, c, 
                                         l, m, b, c);

        // E3,4
        contrib += (1.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, k, l,
                                         j, k, a, m,
                                         l, m, b, c,
                                         i, j, a, b,
                                         j, k, l, a, b, c, 
                                         l, m, b, c);

        // E3,6
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         b, c, i, j,
                                         i, a, k, l,
                                         j, k, a, m,
                                         l, m, b, c,
                                         i, j, b, c,
                                         j, k, l, a, b, c, 
                                         l, m, b, c);

        // E3,7
        contrib += (1.0) / (4.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         b, c, i, j,
                                         i, a, k, l,
                                         k, l, a, m,
                                         j, m, b, c,
                                         i, j, b, c,
                                         j, k, l, a, b, c, 
                                         j, m, b, c);


        // E3,13
        contrib += (1.0) / (2.0) * contribution_mp4_triple(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, c, k, l,
                                         k, l, a, m,
                                         j, m, b, c,
                                         i, j, a, b,
                                         j, k, l, a, b, c, 
                                         j, m, b, c);

    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_5h3p, block_sum);
    }
}



__global__ void compute_mp4_6h2p_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_6h2p)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;

    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int n  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int m  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int l  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;


        // E2,7
        contrib += (1.0) / (16.0) * contribution_mp4_double(eri_mo, orbital_energies, num_basis,
                                         a, b, i, j,
                                         i, j, k, l,
                                         k, l, m, n,
                                         m, n, a, b,
                                         i, j, a, b,
                                         k, l, a, b, 
                                         m, n, a, b);


    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_6h2p, block_sum);
    }
}

real_t mp4_from_aoeri_via_full_moeri(const real_t* d_eri_ao, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_occ) {
    const int num_threads = 256;

    const int N = num_basis * num_basis;

    // ------------------------------------------------------------
    // 1) allocate full MO ERI on device: d_eri_mo (N x N)
    // ------------------------------------------------------------
    double* d_eri_mo = nullptr;
    size_t bytes_mo = (size_t)N * (size_t)N * sizeof(double);
    cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
    }


    // ------------------------------------------------------------
    // 2) AO -> MO full transformation (writes into d_eri_mo)
    // ------------------------------------------------------------
    {
        std::string str = "Computing AO -> MO full integral transformation... ";
        PROFILE_ELAPSED_TIME(str);

        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix, num_basis, d_eri_mo);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    //debug: checking MO ERI by comparing with brute-force transformation and stored MO ERI
    // std::cout << "Checking MO ERI..." << std::endl;
    // check_moeri(d_eri_mo, d_eri_ao, d_coefficient_matrix, num_basis);

    // ------------------------------------------------------------
    // 3) MP2 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp2_energy;
    cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
    if(d_mp2_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP2 energy.");
    }
    cudaMemset(d_mp2_energy, 0.0, sizeof(real_t));
    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp2_moeri_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, d_mp2_energy);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_mp2_energy);
    std::cout << "MP2 energy: " << h_mp2_energy << " Hartree" << std::endl;



    // ------------------------------------------------------------
    // 4) MP3 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp3_energy;
    cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);
    cudaDeviceSynchronize();

    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);
       
        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_mp3_energy);
    cudaFree(d_eri_mo);

    std::cout << "4h2p term: " << h_mp3_energy[0] << " Hartree" << std::endl;
    std::cout << "2h4p term: " << h_mp3_energy[1] << " Hartree" << std::endl;
    std::cout << "3h3p term: " << h_mp3_energy[2] << " Hartree" << std::endl;

    real_t mp3_energy = h_mp2_energy + h_mp3_energy[0] + h_mp3_energy[1] + h_mp3_energy[2];


    // ------------------------------------------------------------
    // 5) compute MP4 energy
    // ------------------------------------------------------------
    real_t* d_mp4_energy;
    cudaMalloc((void**)&d_mp4_energy, sizeof(real_t) * 5); // Allocate space for 5 kernels
    if(d_mp4_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP4 energy.");
    }
    cudaMemset(d_mp4_energy, 0.0, sizeof(real_t)*5);
    cudaDeviceSynchronize();

    { // 2h6p
        std::string str = "Computing MP4 energy 2h6p term... (1/5) ";
        PROFILE_ELAPSED_TIME(str);
        size_t num_spin_occ = (size_t)num_occ * 2;
        size_t num_spin_vir = (size_t)(num_basis - num_occ) * 2;
        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);
        
        compute_mp4_2h6p_kernel<<<num_blocks, num_threads,shmem>>>(d_eri_mo, d_orbital_energies, num_basis, (int)num_spin_occ, (int)num_spin_vir, &d_mp4_energy[0]);
    }
    { // 3h5p
        std::string str = "Computing MP4 energy 3h5p term... (2/5) ";
        PROFILE_ELAPSED_TIME(str);
        size_t num_spin_occ = (size_t)num_occ * 2;
        size_t num_spin_vir = (size_t)(num_basis - num_occ) * 2;
        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);
        
        compute_mp4_3h5p_kernel<<<num_blocks, num_threads,shmem>>>(d_eri_mo, d_orbital_energies, num_basis, (int)num_spin_occ, (int)num_spin_vir, &d_mp4_energy[1]);
    }
    { // 4h4p
        std::string str = "Computing MP4 energy 4h4p term... (3/5) ";
        PROFILE_ELAPSED_TIME(str);
        size_t num_spin_occ = (size_t)num_occ * 2;
        size_t num_spin_vir = (size_t)(num_basis - num_occ) * 2;
        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);
        
        compute_mp4_4h4p_kernel<<<num_blocks, num_threads,shmem>>>(d_eri_mo, d_orbital_energies, num_basis, (int)num_spin_occ, (int)num_spin_vir, &d_mp4_energy[2]);
    }
    { // 5h3p
        std::string str = "Computing MP4 energy 5h3p term... (4/5) ";
        PROFILE_ELAPSED_TIME(str);
        size_t num_spin_occ = (size_t)num_occ * 2;
        size_t num_spin_vir = (size_t)(num_basis - num_occ) * 2;
        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);
        
        compute_mp4_5h3p_kernel<<<num_blocks, num_threads,shmem>>>(d_eri_mo, d_orbital_energies, num_basis, (int)num_spin_occ, (int)num_spin_vir, &d_mp4_energy[3]);
    }
    { // 6h2p
        std::string str = "Computing MP4 energy 6h2p term... (5/5) ";
        PROFILE_ELAPSED_TIME(str);
        size_t num_spin_occ = (size_t)num_occ * 2;
        size_t num_spin_vir = (size_t)(num_basis - num_occ) * 2;
        size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir * num_spin_vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);
        
        compute_mp4_6h2p_kernel<<<num_blocks, num_threads,shmem>>>(d_eri_mo, d_orbital_energies, num_basis, (int)num_spin_occ, (int)num_spin_vir, &d_mp4_energy[4]);
    }
    real_t h_mp4_energy[5];
    cudaMemcpy(h_mp4_energy, d_mp4_energy, sizeof(real_t)*5, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_mp4_energy);
    cudaFree(d_eri_mo);
    std::cout << "2h6p term: " << h_mp4_energy[0] << " Hartree" << std::endl;
    std::cout << "3h5p term: " << h_mp4_energy[1] << " Hartree" << std::endl;
    std::cout << "4h4p term: " << h_mp4_energy[2] << " Hartree" << std::endl;
    std::cout << "5h3p term: " << h_mp4_energy[3] << " Hartree" << std::endl;
    std::cout << "6h2p term: " << h_mp4_energy[4] << " Hartree" << std::endl; 

    real_t mp4_corr_energy = h_mp4_energy[0] + h_mp4_energy[1] + h_mp4_energy[2] + h_mp4_energy[3] + h_mp4_energy[4];
    std::cout << "MP4 correlation energy: " << mp4_corr_energy << " Hartree" << std::endl;
    real_t mp4_total_energy = mp3_energy + mp4_corr_energy;
    std::cout << "MP4 total energy: " << mp4_total_energy << " Hartree" << std::endl;

    return mp4_total_energy;
}


*/






/////////////////////////////////////////////////////////////////////////////////// factorization MP4 kernels
// Note: The number of kernels may be small. So, CUDA streams shuld be used to overlap their executions.


__global__ void compute_mp4_E1_1_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E1_1)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;

        // S_jc
        real_t S_jc = 0.0;

        // sum over k, d, e
        for(int k = 0; k < num_spin_occ; ++k){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;
                for(int e_ = 0; e_ < num_spin_vir; ++e_){
                    int e = e_ + num_spin_occ;

                    real_t eri_deck = antisym_eri(eri_mo, num_basis, d, e, c, k);
                    real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
                    real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

                    S_jc += eri_deck * eri_jkde / e_jkde;
                }
            }
        }

        // T_jc
        real_t T_jc = 0.0;

        // sum over i, a, b
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;
                for(int b_ = 0; b_ < num_spin_vir; ++b_){
                    int b = b_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_icab = antisym_eri(eri_mo, num_basis, i, c, a, b);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                    T_jc += eri_abij * eri_icab / e_ijab;
                }
            }
        }

        real_t e_jc = orbital_energies[j/2] - orbital_energies[c/2];
        contrib += (1.0) / (4.0) * S_jc * T_jc / e_jc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E1_1, block_sum);
    }
}


__global__ void compute_mp4_E1_2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E1_2)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;

        // S_kb
        real_t S_kb = 0.0;

        // sum over l, m, c
        for(int l = 0; l < num_spin_occ; ++l){
            for(int m = 0; m < num_spin_occ; ++m){
                for(int c_ = 0; c_ < num_spin_vir; ++c_){
                    int c = c_ + num_spin_occ;

                    real_t eri_kclm = antisym_eri(eri_mo, num_basis, k, c, l, m);
                    real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
                    real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

                    S_kb += eri_kclm * eri_lmbc / e_lmbc;
                }
            }
        }

        // T_kb
        real_t T_kb = 0.0;

        // sum over i, j, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){
                for(int a_ = 0; a_ < num_spin_vir; ++a_){
                    int a = a_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_ijak = antisym_eri(eri_mo, num_basis, i, j, a, k);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                    T_kb += eri_abij * eri_ijak / e_ijab;
                }
            }
        }

        real_t e_kb = orbital_energies[k/2] - orbital_energies[b/2];
        contrib += (1.0) / (4.0) * S_kb * T_kb / e_kb;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E1_2, block_sum);
    }
}



__global__ void compute_mp4_E1_3_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E1_3)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;

        // S_kb
        real_t S_kb = 0.0;

        // sum over l, c, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int c_ = 0; c_ < num_spin_vir; ++c_){
                int c = c_ + num_spin_occ;
                for(int d_ = 0; d_ < num_spin_vir; ++d_){
                    int d = d_ + num_spin_occ;

                    real_t eri_cdbl = antisym_eri(eri_mo, num_basis, c, d, b, l);
                    real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                    real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                    S_kb += eri_cdbl * eri_klcd / e_klcd;
                }
            }
        }

        // T_kb
        real_t T_kb = 0.0;

        // sum over i, j, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){
                for(int a_ = 0; a_ < num_spin_vir; ++a_){
                    int a = a_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_ijak = antisym_eri(eri_mo, num_basis, i, j, a, k);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                    T_kb += eri_abij * eri_ijak / e_ijab;
                }
            }
        }

        real_t e_kb = orbital_energies[k/2] - orbital_energies[b/2];
        contrib += - (1.0) / (4.0) * S_kb * T_kb / e_kb;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E1_3, block_sum);
    }
}



__global__ void compute_mp4_E1_4_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E1_4)
{
    size_t total = (size_t)num_spin_occ * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;

        // S_jc
        real_t S_jc = 0.0;

        // sum over k,l,d
        for(int k = 0; k < num_spin_occ; ++k){
            for(int l = 0; l < num_spin_occ; ++l){
                for(int d_ = 0; d_ < num_spin_vir; ++d_){
                    int d = d_ + num_spin_occ;

                    real_t eri_jdkl = antisym_eri(eri_mo, num_basis, j, d, k, l);
                    real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                    real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                    S_jc += eri_jdkl * eri_klcd / e_klcd;
                }
            }
        }

        // T_jc
        real_t T_jc = 0.0;

        // sum over i, a, b
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;
                for(int b_ = 0; b_ < num_spin_vir; ++b_){
                    int b = b_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_icab = antisym_eri(eri_mo, num_basis, i, c, a, b);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                    T_jc += eri_abij * eri_icab / e_ijab;
                }
            }
        }

        real_t e_jc = orbital_energies[j/2] - orbital_energies[c/2];
        contrib += - (1.0) / (4.0) * S_jc * T_jc / e_jc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E1_4, block_sum);
    }
}



__global__ void compute_mp4_E2_1_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_1)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijcd
        real_t S_ijcd = 0.0;

        // sum over e, f
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;
            for(int f_ = 0; f_ < num_spin_vir; ++f_){
                int f = f_ + num_spin_occ;

                real_t eri_ijef = antisym_eri(eri_mo, num_basis, i, j, e, f);
                real_t eri_efcd = antisym_eri(eri_mo, num_basis, e, f, c, d);
                real_t e_ijef = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[e/2] - orbital_energies[f/2];

                S_ijcd += eri_ijef * eri_efcd / e_ijef;
            }
        }

        // no T term

        real_t e_ijcd = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (16.0) * S_ijcd * S_ijcd / e_ijcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_1, block_sum);
    }
}




__global__ void compute_mp4_E2_2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_2)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_klab
        real_t S_klab = 0.0;

        // sum over m, c
        for(int m = 0; m < num_spin_occ; ++m){
            for(int c_ = 0; c_ < num_spin_vir; ++c_){
                int c = c_ + num_spin_occ;

                real_t eri_kcam = antisym_eri(eri_mo, num_basis, k, c, a, m);
                real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
                real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

                S_klab += eri_kcam * eri_lmbc / e_lmbc;
            }
        }

        // T_klab
        real_t T_klab = 0.0;

        // sum over i, j
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_ijkl = antisym_eri(eri_mo, num_basis, i, j, k, l);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_klab += eri_abij * eri_ijkl / e_ijab;
            }
        }

        real_t e_klab = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2];
        contrib += (1.0) / (2.0) * (S_klab * T_klab) / e_klab;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_2, block_sum);
    }
}




__global__ void compute_mp4_E2_3_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_3)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijcd
        real_t S_ijcd = 0.0;

        // sum over k, l
        for(int k = 0; k < num_spin_occ; ++k){
            for(int l = 0; l < num_spin_occ; ++l){

                real_t eri_ijkl = antisym_eri(eri_mo, num_basis, i, j, k, l);
                real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_ijcd += eri_ijkl * eri_klcd / e_klcd;
            }
        }

        // T_ijcd
        real_t T_ijcd = 0.0;

        // sum over a, b
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;
            for(int b_ = 0; b_ < num_spin_vir; ++b_){
                int b = b_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_cdab = antisym_eri(eri_mo, num_basis, c, d, a, b);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_ijcd += eri_abij * eri_cdab / e_ijab;
            }
        }

        real_t e_ijcd = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (16.0) * (S_ijcd * T_ijcd) / e_ijcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_3, block_sum);
    }
}




__global__ void compute_mp4_E2_4_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_4)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, m
        for(int l = 0; l < num_spin_occ; ++l){
            for(int m = 0; m < num_spin_occ; ++m){

                real_t eri_jklm = antisym_eri(eri_mo, num_basis, j, k, l, m);
                real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
                real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

                S_jkbc += eri_jklm * eri_lmbc / e_lmbc;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_icak = antisym_eri(eri_mo, num_basis, i, c, a, k);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_jkbc += eri_abij * eri_icak / e_ijab;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * (S_jkbc * T_jkbc) / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_4, block_sum);
    }
}




__global__ void compute_mp4_E2_5_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_5)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_kdbl = antisym_eri(eri_mo, num_basis, k, d, b, l);
                real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
                real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_jkbc += eri_kdbl * eri_jlcd / e_jlcd;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_icak = antisym_eri(eri_mo, num_basis, i, c, a, k);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_jkbc += eri_abij * eri_icak / e_ijab;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib +=  - (1.0) * (S_jkbc * T_jkbc) / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_5, block_sum);
    }
}



__global__ void compute_mp4_E2_6_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_6)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_jdbl = antisym_eri(eri_mo, num_basis, j, d, b, l);
                real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_jkbc += eri_jdbl * eri_klcd / e_klcd;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_icak = antisym_eri(eri_mo, num_basis, i, c, a, k);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_jkbc += eri_abij * eri_icak / e_ijab;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib +=  (1.0) * (S_jkbc * T_jkbc) / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_6, block_sum);
    }
}





__global__ void compute_mp4_E2_7_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_7)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_klab
        real_t S_klab = 0.0;

        // sum over m, n
        for(int m = 0; m < num_spin_occ; ++m){
            for(int n = 0; n < num_spin_occ; ++n){

                real_t eri_abmn = antisym_eri(eri_mo, num_basis, a, b, m, n);
                real_t eri_mnkl = antisym_eri(eri_mo, num_basis, m, n, k, l);
                real_t e_mnab = orbital_energies[m/2] + orbital_energies[n/2] - orbital_energies[a/2] - orbital_energies[b/2];

                S_klab += eri_abmn * eri_mnkl / e_mnab;
            }
        }

        // No T term

        real_t e_klab = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2];
        contrib += (1.0) / (16.0) * S_klab * S_klab / e_klab;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_7, block_sum);
    }
}



__global__ void compute_mp4_E2_8_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_8)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_jdbl = antisym_eri(eri_mo, num_basis, j, d, b, l);
                real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_jkbc += eri_jdbl * eri_klcd / e_klcd;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
                real_t eri_ibak = antisym_eri(eri_mo, num_basis, i, b, a, k);
                real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

                T_jkbc += eri_acij * eri_ibak / e_ijac;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += - (1.0) * S_jkbc * T_jkbc / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_8, block_sum);
    }
}


__global__ void compute_mp4_E2_9_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_9)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_kdbl = antisym_eri(eri_mo, num_basis, k, d, b, l);
                real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
                real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_jkbc += eri_kdbl * eri_jlcd / e_jlcd;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
                real_t eri_ibak = antisym_eri(eri_mo, num_basis, i, b, a, k);
                real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

                T_jkbc += eri_acij * eri_ibak / e_ijac;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) * S_jkbc * T_jkbc / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_9, block_sum);
    }
}


__global__ void compute_mp4_E2_10_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_10)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_klab
        real_t S_klab = 0.0;

        // sum over c, d
        for(int c_ = 0; c_ < num_spin_vir; ++c_){
            int c = c_ + num_spin_occ;
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_cdab = antisym_eri(eri_mo, num_basis, c, d, a, b);
                real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                S_klab += eri_cdab * eri_klcd / e_klcd;
            }
        }

        // T_klab
        real_t T_klab = 0.0;

        // sum over i, j
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_ijkl = antisym_eri(eri_mo, num_basis, i, j, k, l);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_klab += eri_abij * eri_ijkl / e_ijab;
            }
        }

        real_t e_klab = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2];
        contrib += (1.0) / (16.0) * S_klab * T_klab / e_klab;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_10, block_sum);
    }
}



__global__ void compute_mp4_E2_11_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_11)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over d, e
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = e_ + num_spin_occ;

                real_t eri_debc = antisym_eri(eri_mo, num_basis, d, e, b, c);
                real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
                real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

                S_jkbc += eri_debc * eri_jkde / e_jkde;
            }
        }

        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_icak = antisym_eri(eri_mo, num_basis, i, c, a, k);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_jkbc += eri_abij * eri_icak / e_ijab;
            }
        }

        real_t e_jkbc = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jkbc * T_jkbc / e_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_11, block_sum);
    }
}




__global__ void compute_mp4_E2_12_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E2_12)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijcd
        real_t S_ijcd = 0.0;

        // sum over k, e
        for(int k = 0; k < num_spin_occ; ++k){
            for(int e_ = 0; e_ < num_spin_vir; ++e_){
                int e = e_ + num_spin_occ;

                real_t eri_ieck = antisym_eri(eri_mo, num_basis, i, e, c, k);
                real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
                real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

                S_ijcd += eri_ieck * eri_jkde / e_jkde;
            }
        }

        // T_ijcd
        real_t T_ijcd = 0.0;

        // sum over a, b
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;
            for(int b_ = 0; b_ < num_spin_vir; ++b_){
                int b = b_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_cdab = antisym_eri(eri_mo, num_basis, c, d, a, b);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                T_ijcd += eri_abij * eri_cdab / e_ijab;
            }
        }

        real_t e_ijcd = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijcd * T_ijcd / e_ijcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E2_12, block_sum);
    }
}



__global__ void compute_mp4_E3_1_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_1)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_jdab = antisym_eri(eri_mo, num_basis, j, d, a, b);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_jdab * eri_klcd / e_klcd;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_ibkl = antisym_eri(eri_mo, num_basis, i, b, k, l);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_jklabc += eri_acij * eri_ibkl / e_ijac;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_1, block_sum);
    }
}




__global__ void compute_mp4_E3_1_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          const int vir_b,
                                          const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_1)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_jdab = antisym_eri(eri_mo, num_basis, j, d, a, b);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_jdab * eri_klcd / e_klcd;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_ibkl = antisym_eri(eri_mo, num_basis, i, b, k, l);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_jklabc += eri_acij * eri_ibkl / e_ijac;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_1, block_sum);
    }
}

__global__ void compute_mp4_E3_2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_2)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_jdab = antisym_eri(eri_mo, num_basis, j, d, a, b);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_jdab * eri_klcd / e_klcd;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += - (1.0) / (4.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_2, block_sum);
    }
}


__global__ void compute_mp4_E3_2_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          const int vir_b,
                                          const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_2)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int a = a_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_jdab = antisym_eri(eri_mo, num_basis, j, d, a, b);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_jdab * eri_klcd / e_klcd;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += - (1.0) / (4.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_2, block_sum);
    }
}



__global__ void compute_mp4_E3_3_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_3)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ijbl = antisym_eri(eri_mo, num_basis, i, j, b, l);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ijbl * eri_klcd / e_klcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += - (1.0) / (4.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_3, block_sum);
    }
}

__global__ void compute_mp4_E3_3_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_3)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ijbl = antisym_eri(eri_mo, num_basis, i, j, b, l);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ijbl * eri_klcd / e_klcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += - (1.0) / (4.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_3, block_sum);
    }
}



__global__ void compute_mp4_E3_4_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_4)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_jkam = antisym_eri(eri_mo, num_basis, j, k, a, m);
            real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
            real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_jkam * eri_lmbc / e_lmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_4, block_sum);
    }
}


__global__ void compute_mp4_E3_4_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_4)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c;//(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b;//(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_jkam = antisym_eri(eri_mo, num_basis, j, k, a, m);
            real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
            real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_jkam * eri_lmbc / e_lmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_4, block_sum);
    }
}


__global__ void compute_mp4_E3_5_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_5)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ikbl = antisym_eri(eri_mo, num_basis, i, k, b, l);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ikbl * eri_jlcd / e_jlcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_5, block_sum);
    }
}



__global__ void compute_mp4_E3_5_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_5)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d;//(int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c;//(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ikbl = antisym_eri(eri_mo, num_basis, i, k, b, l);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ikbl * eri_jlcd / e_jlcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_5, block_sum);
    }
}




__global__ void compute_mp4_E3_6_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_6)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_jkam = antisym_eri(eri_mo, num_basis, j, k, a, m);
            real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
            real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_jkam * eri_lmbc / e_lmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_bcij = antisym_eri(eri_mo, num_basis, b, c, i, j);
            real_t eri_iakl = antisym_eri(eri_mo, num_basis, i, a, k, l);
            real_t e_ijbc = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[b/2] - orbital_energies[c/2];

            T_jklabc += eri_bcij * eri_iakl / e_ijbc;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_6, block_sum);
    }
}


__global__ void compute_mp4_E3_6_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_6)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ =  vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ =  vir_b; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_jkam = antisym_eri(eri_mo, num_basis, j, k, a, m);
            real_t eri_lmbc = antisym_eri(eri_mo, num_basis, l, m, b, c);
            real_t e_lmbc = orbital_energies[l/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_jkam * eri_lmbc / e_lmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_bcij = antisym_eri(eri_mo, num_basis, b, c, i, j);
            real_t eri_iakl = antisym_eri(eri_mo, num_basis, i, a, k, l);
            real_t e_ijbc = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[b/2] - orbital_energies[c/2];

            T_jklabc += eri_bcij * eri_iakl / e_ijbc;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_6, block_sum);
    }
}


__global__ void compute_mp4_E3_7_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_7)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_klam = antisym_eri(eri_mo, num_basis, k, l, a, m);
            real_t eri_jmbc = antisym_eri(eri_mo, num_basis, j, m, b, c);
            real_t e_jmbc = orbital_energies[j/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_klam * eri_jmbc / e_jmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_bcij = antisym_eri(eri_mo, num_basis, b, c, i, j);
            real_t eri_iakl = antisym_eri(eri_mo, num_basis, i, a, k, l);
            real_t e_ijbc = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[b/2] - orbital_energies[c/2];

            T_jklabc += eri_bcij * eri_iakl / e_ijbc;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (4.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_7, block_sum);
    }
}


__global__ void compute_mp4_E3_7_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_7)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_klam = antisym_eri(eri_mo, num_basis, k, l, a, m);
            real_t eri_jmbc = antisym_eri(eri_mo, num_basis, j, m, b, c);
            real_t e_jmbc = orbital_energies[j/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_klam * eri_jmbc / e_jmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_bcij = antisym_eri(eri_mo, num_basis, b, c, i, j);
            real_t eri_iakl = antisym_eri(eri_mo, num_basis, i, a, k, l);
            real_t e_ijbc = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[b/2] - orbital_energies[c/2];

            T_jklabc += eri_bcij * eri_iakl / e_ijbc;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (4.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_7, block_sum);
    }
}


__global__ void compute_mp4_E3_8_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_8)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_iebc = antisym_eri(eri_mo, num_basis, i, e, b, c);
            real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
            real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_iebc * eri_jkde / e_jkde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_adij = antisym_eri(eri_mo, num_basis, a, d, i, j);
            real_t eri_bcak = antisym_eri(eri_mo, num_basis, b, c, a, k);
            real_t e_ijad = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[d/2];

            T_ijkbcd += eri_adij * eri_bcak / e_ijad;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_8, block_sum);
    }
}


__global__ void compute_mp4_E3_8_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_8)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_iebc = antisym_eri(eri_mo, num_basis, i, e, b, c);
            real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
            real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_iebc * eri_jkde / e_jkde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_adij = antisym_eri(eri_mo, num_basis, a, d, i, j);
            real_t eri_bcak = antisym_eri(eri_mo, num_basis, b, c, a, k);
            real_t e_ijad = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[d/2];

            T_ijkbcd += eri_adij * eri_bcak / e_ijad;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_8, block_sum);
    }
}



__global__ void compute_mp4_E3_9_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_9)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ikbl = antisym_eri(eri_mo, num_basis, i, k, b, l);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ikbl * eri_jlcd / e_jlcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_bdak = antisym_eri(eri_mo, num_basis, b, d, a, k);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_ijkbcd += eri_acij * eri_bdak / e_ijac;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += - (1.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_9, block_sum);
    }
}


__global__ void compute_mp4_E3_9_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_9)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ikbl = antisym_eri(eri_mo, num_basis, i, k, b, l);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ikbl * eri_jlcd / e_jlcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_bdak = antisym_eri(eri_mo, num_basis, b, d, a, k);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_ijkbcd += eri_acij * eri_bdak / e_ijac;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += - (1.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_9, block_sum);
    }
}



__global__ void compute_mp4_E3_10_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_10)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_iebc = antisym_eri(eri_mo, num_basis, i, e, b, c);
            real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
            real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_iebc * eri_jkde / e_jkde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_10, block_sum);
    }
}


__global__ void compute_mp4_E3_10_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_10)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_iebc = antisym_eri(eri_mo, num_basis, i, e, b, c);
            real_t eri_jkde = antisym_eri(eri_mo, num_basis, j, k, d, e);
            real_t e_jkde = orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_iebc * eri_jkde / e_jkde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_10, block_sum);
    }
}


__global__ void compute_mp4_E3_11_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_11)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_kebc = antisym_eri(eri_mo, num_basis, k, e, b, c);
            real_t eri_ijde = antisym_eri(eri_mo, num_basis, i, j, d, e);
            real_t e_ijde = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_kebc * eri_ijde / e_ijde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_11, block_sum);
    }
}


__global__ void compute_mp4_E3_11_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_11)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; // (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_kebc = antisym_eri(eri_mo, num_basis, k, e, b, c);
            real_t eri_ijde = antisym_eri(eri_mo, num_basis, i, j, d, e);
            real_t e_ijde = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_kebc * eri_ijde / e_ijde;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_cdak = antisym_eri(eri_mo, num_basis, c, d, a, k);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_ijkbcd += eri_abij * eri_cdak / e_ijab;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_11, block_sum);
    }
}




__global__ void compute_mp4_E3_12_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_12)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ijbl = antisym_eri(eri_mo, num_basis, i, j, b, l);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ijbl * eri_klcd / e_klcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_bdak = antisym_eri(eri_mo, num_basis, b, d, a, k);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_ijkbcd += eri_acij * eri_bdak / e_ijac;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_12, block_sum);
    }
}



__global__ void compute_mp4_E3_12_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_12)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir; // * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over l
        for(int l = 0; l < num_spin_occ; ++l){

            real_t eri_ijbl = antisym_eri(eri_mo, num_basis, i, j, b, l);
            real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
            real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_ijkbcd += eri_ijbl * eri_klcd / e_klcd;
        }

        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_bdak = antisym_eri(eri_mo, num_basis, b, d, a, k);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_ijkbcd += eri_acij * eri_bdak / e_ijac;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (2.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_12, block_sum);
    }
}



__global__ void compute_mp4_E3_13_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_13)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_klam = antisym_eri(eri_mo, num_basis, k, l, a, m);
            real_t eri_jmbc = antisym_eri(eri_mo, num_basis, j, m, b, c);
            real_t e_jmbc = orbital_energies[j/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_klam * eri_jmbc / e_jmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_13, block_sum);
    }
}



__global__ void compute_mp4_E3_13_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_13)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over m
        for(int m = 0; m < num_spin_occ; ++m){

            real_t eri_klam = antisym_eri(eri_mo, num_basis, k, l, a, m);
            real_t eri_jmbc = antisym_eri(eri_mo, num_basis, j, m, b, c);
            real_t e_jmbc = orbital_energies[j/2] + orbital_energies[m/2] - orbital_energies[b/2] - orbital_energies[c/2];

            S_jklabc += eri_klam * eri_jmbc / e_jmbc;
        }

        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_13, block_sum);
    }
}



__global__ void compute_mp4_E3_14_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_14)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_kdab = antisym_eri(eri_mo, num_basis, k, d, a, b);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_kdab * eri_jlcd / e_jlcd;
        }


        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_ibkl = antisym_eri(eri_mo, num_basis, i, b, k, l);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_jklabc += eri_acij * eri_ibkl / e_ijac;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += - (1.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_14, block_sum);
    }
}


__global__ void compute_mp4_E3_14_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_14)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_kdab = antisym_eri(eri_mo, num_basis, k, d, a, b);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_kdab * eri_jlcd / e_jlcd;
        }


        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_acij = antisym_eri(eri_mo, num_basis, a, c, i, j);
            real_t eri_ibkl = antisym_eri(eri_mo, num_basis, i, b, k, l);
            real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

            T_jklabc += eri_acij * eri_ibkl / e_ijac;
        }

        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += - (1.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_14, block_sum);
    }
}




__global__ void compute_mp4_E3_15_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_15)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_kebc = antisym_eri(eri_mo, num_basis, k, e, b, c);
            real_t eri_ijde = antisym_eri(eri_mo, num_basis, i, j, d, e);
            real_t e_ijde = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_kebc * eri_ijde / e_ijde;
        }


        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_adij = antisym_eri(eri_mo, num_basis, a, d, i, j);
            real_t eri_bcak = antisym_eri(eri_mo, num_basis, b, c, a, k);
            real_t e_ijad = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[d/2];

            T_ijkbcd += eri_adij * eri_bcak / e_ijad;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (4.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_15, block_sum);
    }
}



__global__ void compute_mp4_E3_15_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_c,
                                            const int vir_d,
                                          real_t* __restrict__ d_mp4_energy_E3_15)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int d_ = vir_d; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;
        int d = d_ + num_spin_occ;

        // S_ijkbcd
        real_t S_ijkbcd = 0.0;

        // sum over e
        for(int e_ = 0; e_ < num_spin_vir; ++e_){
            int e = e_ + num_spin_occ;

            real_t eri_kebc = antisym_eri(eri_mo, num_basis, k, e, b, c);
            real_t eri_ijde = antisym_eri(eri_mo, num_basis, i, j, d, e);
            real_t e_ijde = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[d/2] - orbital_energies[e/2];

            S_ijkbcd += eri_kebc * eri_ijde / e_ijde;
        }


        // T_ijkbcd
        real_t T_ijkbcd = 0.0;

        // sum over a
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;

            real_t eri_adij = antisym_eri(eri_mo, num_basis, a, d, i, j);
            real_t eri_bcak = antisym_eri(eri_mo, num_basis, b, c, a, k);
            real_t e_ijad = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[d/2];

            T_ijkbcd += eri_adij * eri_bcak / e_ijad;
        }

        real_t e_ijkbcd = orbital_energies[i/2] + orbital_energies[j/2] + orbital_energies[k/2] - orbital_energies[b/2] - orbital_energies[c/2] - orbital_energies[d/2];
        contrib += (1.0) / (4.0) * S_ijkbcd * T_ijkbcd / e_ijkbcd;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_15, block_sum);
    }
}



__global__ void compute_mp4_E3_16_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E3_16)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_kdab = antisym_eri(eri_mo, num_basis, k, d, a, b);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_kdab * eri_jlcd / e_jlcd;
        }


        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }


        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_16, block_sum);
    }
}



__global__ void compute_mp4_E3_16_vir2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                            const int vir_b,
                                            const int vir_c,
                                          real_t* __restrict__ d_mp4_energy_E3_16)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = vir_c; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = vir_b; //(int)(t % num_spin_vir); t /= num_spin_vir;
        int a_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j  = (int)(t % num_spin_occ);

        int a = a_ + num_spin_occ;
        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_jklabc
        real_t S_jklabc = 0.0;

        // sum over d
        for(int d_ = 0; d_ < num_spin_vir; ++d_){
            int d = d_ + num_spin_occ;

            real_t eri_kdab = antisym_eri(eri_mo, num_basis, k, d, a, b);
            real_t eri_jlcd = antisym_eri(eri_mo, num_basis, j, l, c, d);
            real_t e_jlcd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

            S_jklabc += eri_kdab * eri_jlcd / e_jlcd;
        }


        // T_jklabc
        real_t T_jklabc = 0.0;

        // sum over i
        for(int i = 0; i < num_spin_occ; ++i){

            real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
            real_t eri_ickl = antisym_eri(eri_mo, num_basis, i, c, k, l);
            real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

            T_jklabc += eri_abij * eri_ickl / e_ijab;
        }


        real_t e_jklabc = orbital_energies[j/2] + orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2] - orbital_energies[c/2];
        contrib += (1.0) / (2.0) * S_jklabc * T_jklabc / e_jklabc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E3_16, block_sum);
    }
}




__global__ void compute_mp4_E4_1_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E4_1)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int k = (int)(t % num_spin_occ); t /= num_spin_occ;
        int j = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i  = (int)(t % num_spin_occ);

        // S_ijkl
        real_t S_ijkl = 0.0;

        // sum over a, b
        for(int a_ = 0; a_ < num_spin_vir; ++a_){
            int a = a_ + num_spin_occ;
            for(int b_ = 0; b_ < num_spin_vir; ++b_){
                int b = b_ + num_spin_occ;

                real_t eri_abkl = antisym_eri(eri_mo, num_basis, a, b, k, l);
                real_t eri_ijab = antisym_eri(eri_mo, num_basis, i, j, a, b);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];

                S_ijkl += eri_abkl * eri_ijab / e_ijab;
            }
        }


        // T_ijkl
        real_t T_ijkl = 0.0;

        // sum over c, d
        for(int c_ = 0; c_ < num_spin_vir; ++c_){
            int c = c_ + num_spin_occ;
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_cdij = antisym_eri(eri_mo, num_basis, c, d, i, j);
                real_t eri_klcd = antisym_eri(eri_mo, num_basis, k, l, c, d);
                real_t e_ijcd = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[c/2] - orbital_energies[d/2];
                real_t e_klcd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[c/2] - orbital_energies[d/2];

                T_ijkl += eri_cdij * eri_klcd / (e_ijcd * e_klcd);
            }
        }

        contrib += (1.0) / (16.0) * S_ijkl * T_ijkl;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E4_1, block_sum);
    }
}




__global__ void compute_mp4_E4_2_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E4_2)
{
    size_t total = (size_t)num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir);

        int b = b_ + num_spin_occ;
        int c = c_ + num_spin_occ;

        // S_bc
        real_t S_bc = 0.0;

        // sum over k, l, d
        for(int k = 0; k < num_spin_occ; ++k){
            for(int l = 0; l < num_spin_occ; ++l){
                for(int d_ = 0; d_ < num_spin_vir; ++d_){
                    int d = d_ + num_spin_occ;

                    real_t eri_cdkl = antisym_eri(eri_mo, num_basis, c, d, k, l);
                    real_t eri_klbd = antisym_eri(eri_mo, num_basis, k, l, b, d);
                    real_t e_klbd = orbital_energies[k/2] + orbital_energies[l/2] - orbital_energies[b/2] - orbital_energies[d/2];

                    S_bc += eri_cdkl * eri_klbd / e_klbd;
                }
            }
        }


        // T_bc
        real_t T_bc = 0.0;

        // sum over i, j, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int j = 0; j < num_spin_occ; ++j){
                for(int a_ = 0; a_ < num_spin_vir; ++a_){
                    int a = a_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_ijac = antisym_eri(eri_mo, num_basis, i, j, a, c);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];
                    real_t e_ijac = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[c/2];

                    T_bc += eri_abij * eri_ijac / (e_ijab * e_ijac);
                }
            }
        }


        contrib += - (1.0) / (4.0) * S_bc * T_bc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E4_2, block_sum);
    }
}




__global__ void compute_mp4_E4_3_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E4_3)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int l = (int)(t % num_spin_occ); t /= num_spin_occ;
        int i = (int)(t % num_spin_occ);


        // S_il
        real_t S_il = 0.0;

        // sum over k, c, d
        for(int k = 0; k < num_spin_occ; ++k){
            for(int c_ = 0; c_ < num_spin_vir; ++c_){
                int c = c_ + num_spin_occ;
                for(int d_ = 0; d_ < num_spin_vir; ++d_){
                    int d = d_ + num_spin_occ;

                    real_t eri_cdkl = antisym_eri(eri_mo, num_basis, c, d, k, l);
                    real_t eri_ikcd = antisym_eri(eri_mo, num_basis, i, k, c, d);
                    real_t e_ikcd = orbital_energies[i/2] + orbital_energies[k/2] - orbital_energies[c/2] - orbital_energies[d/2];

                    S_il += eri_cdkl * eri_ikcd / e_ikcd;
                }
            }
        }


        // T_il
        real_t T_il = 0.0;

        // sum over j, a, b
        for(int j = 0; j < num_spin_occ; ++j){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;
                for(int b_ = 0; b_ < num_spin_vir; ++b_){
                    int b = b_ + num_spin_occ;

                    real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                    real_t eri_jlab = antisym_eri(eri_mo, num_basis, j, l, a, b);
                    real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];
                    real_t e_jlab = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[a/2] - orbital_energies[b/2];

                    T_il += eri_abij * eri_jlab / (e_ijab * e_jlab);
                }
            }
        }


        contrib += - (1.0) / (4.0) * S_il * T_il;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E4_3, block_sum);
    }
}



__global__ void compute_mp4_E4_4_kernel(const real_t* __restrict__ eri_mo,
                                          const real_t* __restrict__ orbital_energies,
                                          const int num_basis,
                                          const int num_spin_occ,
                                          const int num_spin_vir,
                                          real_t* __restrict__ d_mp4_energy_E4_4)
{
    size_t total = (size_t)num_spin_occ * num_spin_occ* num_spin_vir * num_spin_vir;
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    double contrib = 0.0;
    if(gid < total){

        size_t t = gid;
        int c_ = (int)(t % num_spin_vir); t /= num_spin_vir;
        int b_ = (int)(t % num_spin_vir); t /= num_spin_vir; 
        int k  = (int)(t % num_spin_occ); t /= num_spin_occ; 
        int j  = (int)(t % num_spin_occ);

        int c = c_ + num_spin_occ;
        int b = b_ + num_spin_occ;

        // S_jkbc
        real_t S_jkbc = 0.0;

        // sum over l, d
        for(int l = 0; l < num_spin_occ; ++l){
            for(int d_ = 0; d_ < num_spin_vir; ++d_){
                int d = d_ + num_spin_occ;

                real_t eri_cdkl = antisym_eri(eri_mo, num_basis, c, d, k, l);
                real_t eri_jlbd = antisym_eri(eri_mo, num_basis, j, l, b, d);
                real_t e_jlbd = orbital_energies[j/2] + orbital_energies[l/2] - orbital_energies[b/2] - orbital_energies[d/2];

                S_jkbc += eri_cdkl * eri_jlbd / e_jlbd;
            }
        }


        // T_jkbc
        real_t T_jkbc = 0.0;

        // sum over i, a
        for(int i = 0; i < num_spin_occ; ++i){
            for(int a_ = 0; a_ < num_spin_vir; ++a_){
                int a = a_ + num_spin_occ;

                real_t eri_abij = antisym_eri(eri_mo, num_basis, a, b, i, j);
                real_t eri_ikac = antisym_eri(eri_mo, num_basis, i, k, a, c);
                real_t e_ijab = orbital_energies[i/2] + orbital_energies[j/2] - orbital_energies[a/2] - orbital_energies[b/2];
                real_t e_ikac = orbital_energies[i/2] + orbital_energies[k/2] - orbital_energies[a/2] - orbital_energies[c/2];

                T_jkbc += eri_abij * eri_ikac / (e_ijab * e_ikac);
            }
        }

        contrib += (1.0) / (2.0) * S_jkbc * T_jkbc;
    }

    double block_sum = block_reduce_sum(contrib);
    if(threadIdx.x == 0){
        atomicAdd(d_mp4_energy_E4_4, block_sum);
    }
}


// define the kernel functions as function pointers for mp4 terms
using mp4_term_kernel_t = void (*)(const real_t*, const real_t*, const int, const int, const int, real_t* __restrict__);
using mp4_term_vir2_kernel_t = void (*)(const real_t*, const real_t*, const int, const int, const int, const int, const int, real_t* __restrict__);

const int num_mp4_terms_single   = 4;
const int num_mp4_terms_double   = 12;
const int num_mp4_terms_triple   = 16;
const int num_mp4_terms_quadrule = 4;

const mp4_term_kernel_t mp4_term_kernels_single[num_mp4_terms_single] = {
    compute_mp4_E1_1_kernel,
    compute_mp4_E1_2_kernel,
    compute_mp4_E1_3_kernel,
    compute_mp4_E1_4_kernel
};


const mp4_term_kernel_t mp4_term_kernels_double[num_mp4_terms_double] = {
    compute_mp4_E2_1_kernel,
    compute_mp4_E2_2_kernel,
    compute_mp4_E2_3_kernel,
    compute_mp4_E2_4_kernel,
    compute_mp4_E2_5_kernel,
    compute_mp4_E2_6_kernel,
    compute_mp4_E2_7_kernel,
    compute_mp4_E2_8_kernel,
    compute_mp4_E2_9_kernel,
    compute_mp4_E2_10_kernel,
    compute_mp4_E2_11_kernel,
    compute_mp4_E2_12_kernel
};


const mp4_term_vir2_kernel_t mp4_term_kernels_triple[num_mp4_terms_triple] = {
    compute_mp4_E3_1_vir2_kernel,
    compute_mp4_E3_2_vir2_kernel,
    compute_mp4_E3_3_vir2_kernel,
    compute_mp4_E3_4_vir2_kernel,
    compute_mp4_E3_5_vir2_kernel,
    compute_mp4_E3_6_vir2_kernel,
    compute_mp4_E3_7_vir2_kernel,
    compute_mp4_E3_8_vir2_kernel,
    compute_mp4_E3_9_vir2_kernel,
    compute_mp4_E3_10_vir2_kernel,
    compute_mp4_E3_11_vir2_kernel,
    compute_mp4_E3_12_vir2_kernel,
    compute_mp4_E3_13_vir2_kernel,
    compute_mp4_E3_14_vir2_kernel,
    compute_mp4_E3_15_vir2_kernel,
    compute_mp4_E3_16_vir2_kernel,
};


const mp4_term_kernel_t mp4_term_kernels_quadrule[num_mp4_terms_quadrule] = {
    compute_mp4_E4_1_kernel,
    compute_mp4_E4_2_kernel,
    compute_mp4_E4_3_kernel,
    compute_mp4_E4_4_kernel
};

void get_mp4_term_num_block_thread_shmem(int term_index, const int num_spin_occ, const int num_spin_vir, int& num_threads, int& num_blocks, size_t& shared_mem_size) {
    size_t total_threads = 0;
    const int default_num_threads = 256; // default

    switch(term_index) {
        case 0: // E1_1
        case 1: // E1_2
        case 2: // E1_3
        case 3: // E1_4
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_vir;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 4: // E2_1
        case 5: // E2_2
        case 6: // E2_3
        case 7: // E2_4
        case 8: // E2_5
        case 9: // E2_6
        case 10: // E2_7
        case 11: // E2_8
        case 12: // E2_9
        case 13: // E2_10
        case 14: // E2_11
        case 15: // E2_12
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 16: // E3_1
        case 17: // E3_2
        case 18: // E3_3
        case 19: // E3_4
        case 20: // E3_5
        case 21: // E3_6
        case 22: // E3_7
        case 23: // E3_8
        case 24: // E3_9
        case 25: // E3_10
        case 26: // E3_11
        case 27: // E3_12
        case 28: // E3_13
        case 29: // E3_14
        case 30: // E3_15
        case 31: // E3_16
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_vir;// * num_spin_vir * num_spin_vir;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 32: // E4_1
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_occ * num_spin_occ * num_spin_occ;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 33: // E4_2
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_vir * num_spin_vir;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 34: // E4_3
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_occ;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
        case 35: // E4_4
            num_threads = default_num_threads;
            total_threads = (size_t)num_spin_occ * num_spin_occ * num_spin_vir * num_spin_vir;
            num_blocks = (int)((total_threads + num_threads - 1) / num_threads);
            shared_mem_size = (size_t)num_threads * sizeof(double);
            return;
    }
    THROW_EXCEPTION("Invalid MP4 term index: " + std::to_string(term_index));
};



real_t mp4_from_aoeri_via_full_moeri_factorization(const real_t* d_eri_ao, const real_t* d_coefficient_matrix, const real_t* d_orbital_energies, const int num_basis, const int num_occ) {
    const int num_threads = 256;

    const int N = num_basis * num_basis;

    // ------------------------------------------------------------
    // 1) allocate full MO ERI on device: d_eri_mo (N x N)
    // ------------------------------------------------------------
    double* d_eri_mo = nullptr;
    size_t bytes_mo = (size_t)N * (size_t)N * sizeof(double);
    cudaMalloc((void**)&d_eri_mo, bytes_mo);
    if(!d_eri_mo){
        THROW_EXCEPTION("cudaMalloc failed for d_eri_mo.");
    }


    // ------------------------------------------------------------
    // 2) AO -> MO full transformation (writes into d_eri_mo)
    // ------------------------------------------------------------
    {
        std::string str = "Computing AO -> MO full integral transformation... ";
        PROFILE_ELAPSED_TIME(str);

        transform_ao_eri_to_mo_eri_full(d_eri_ao, d_coefficient_matrix, num_basis, d_eri_mo);
        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    //debug: checking MO ERI by comparing with brute-force transformation and stored MO ERI
    // std::cout << "Checking MO ERI..." << std::endl;
    // check_moeri(d_eri_mo, d_eri_ao, d_coefficient_matrix, num_basis);

    // ------------------------------------------------------------
    // 3) MP2 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp2_energy;
    cudaMalloc((void**)&d_mp2_energy, sizeof(real_t));
    if(d_mp2_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP2 energy.");
    }
    cudaMemset(d_mp2_energy, 0.0, sizeof(real_t));
    {
        std::string str = "Computing MP2 energy from full MO ERI... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp2_moeri_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, d_mp2_energy);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    real_t h_mp2_energy;
    cudaMemcpy(&h_mp2_energy, d_mp2_energy, sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_mp2_energy);
    std::cout << "MP2 energy: " << h_mp2_energy << " Hartree" << std::endl;



    // ------------------------------------------------------------
    // 4) MP3 energy from full MO ERI
    // ------------------------------------------------------------
    real_t* d_mp3_energy;
    cudaMalloc((void**)&d_mp3_energy, sizeof(real_t) * 3); // Allocate space for 3 terms
    if(d_mp3_energy == nullptr) {
        THROW_EXCEPTION("Failed to allocate device memory for MP3 energy.");
    }
    cudaMemset(d_mp3_energy, 0.0, sizeof(real_t)*3);
    cudaDeviceSynchronize();

    { // 4h2p term
        std::string str = "Computing MP3 (1/3) 4h2p term... ";
        PROFILE_ELAPSED_TIME(str);
       
        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * occ * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_4h2p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[0]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 2h4p term
        std::string str = "Computing MP3 (2/3) 2h4p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * vir * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_2h4p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[1]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }
    { // 3h3p term
        std::string str = "Computing MP3 (3/3) 3h3p term... ";
        PROFILE_ELAPSED_TIME(str);

        size_t occ = (size_t)num_occ;
        size_t vir = (size_t)(num_basis - num_occ);
        size_t total = (size_t)occ * occ * occ * vir * vir * vir;
        const int num_blocks = (int)((total + num_threads - 1) / num_threads);
        size_t shmem = (size_t)num_threads * sizeof(double);

        mp3_moeri_3h3p_kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_occ, &d_mp3_energy[2]);

        cudaDeviceSynchronize(); // It is for PROFILE_ELAPSED_TIME
    }


    real_t h_mp3_energy[3];
    cudaMemcpy(h_mp3_energy, d_mp3_energy, sizeof(real_t)*3, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    cudaFree(d_mp3_energy);

    std::cout << "4h2p term: " << h_mp3_energy[0] << " Hartree" << std::endl;
    std::cout << "2h4p term: " << h_mp3_energy[1] << " Hartree" << std::endl;
    std::cout << "3h3p term: " << h_mp3_energy[2] << " Hartree" << std::endl;

    real_t mp3_energy = h_mp2_energy + h_mp3_energy[0] + h_mp3_energy[1] + h_mp3_energy[2];


    // ------------------------------------------------------------
    // 5) compute MP4 energy
    // ------------------------------------------------------------
    int num_spin_occ = num_occ * 2;
    int num_spin_vir = (num_basis - num_occ) * 2;


    real_t h_mp4_energy[4]; // single, double, triple, quadruple
    memset(h_mp4_energy, 0, sizeof(real_t)*4);
   
    { // single excitation terms
        std::string str = "Launch kernels of MP4 single excitation terms... ";
        PROFILE_ELAPSED_TIME(str);
        int num_threads;
        int num_blocks;
        size_t shmem;

        int kernel_offset = 0;

        real_t* d_contrib;
        cudaMalloc((void**)&d_contrib, sizeof(real_t));
        cudaMemset(d_contrib, 0.0, sizeof(real_t));        

        // Launch the kernel for single excitation terms
        for(int i=0; i<num_mp4_terms_single; i++){
            // Get the number of blocks, threads, and shared memory size for single excitation terms
            get_mp4_term_num_block_thread_shmem(kernel_offset + i, num_spin_occ, num_spin_vir, num_threads, num_blocks, shmem);

            mp4_term_kernel_t kernel = mp4_term_kernels_single[i];
            kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir, d_contrib);
            cudaDeviceSynchronize();

            real_t h_contrib;
            cudaMemcpy(&h_contrib, d_contrib, sizeof(real_t), cudaMemcpyDeviceToHost);

            std::cout << "  E1 " << i+1 << " contribution: " << h_contrib << " Hartree" << std::endl;

            h_mp4_energy[0] += h_contrib;
        }

        std::cout << "Total MP4 Single excitation energy: " << h_mp4_energy[0] << " Hartree" << std::endl;
    }
    { // double excitation terms
        std::string str = "Launch kernels of MP4 double excitation terms... ";
        PROFILE_ELAPSED_TIME(str);
        int num_threads;
        int num_blocks;
        size_t shmem;

        int kernel_offset = num_mp4_terms_single;

        real_t* d_contrib;
        cudaMalloc((void**)&d_contrib, sizeof(real_t));
        cudaMemset(d_contrib, 0.0, sizeof(real_t));        

        // Launch the kernel for double excitation terms
        for(int i=0; i<num_mp4_terms_double; i++){
            
            // Get the number of blocks, threads, and shared memory size for double excitation terms
            get_mp4_term_num_block_thread_shmem(kernel_offset + i, num_spin_occ, num_spin_vir, num_threads, num_blocks, shmem);

            mp4_term_kernel_t kernel = mp4_term_kernels_double[i];
            kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir, d_contrib);
            cudaDeviceSynchronize();

            real_t h_contrib;
            cudaMemcpy(&h_contrib, d_contrib, sizeof(real_t), cudaMemcpyDeviceToHost);

            std::cout << "  E2 " << i+1 << " contribution: " << h_contrib << " Hartree" << std::endl;

            h_mp4_energy[1] += h_contrib;
        }

        std::cout << "Total MP4 Double excitation energy: " << h_mp4_energy[1] << " Hartree" << std::endl;
    }
    { // triple excitation terms
        std::string str = "Launch kernels of MP4 triple excitation terms... ";
        PROFILE_ELAPSED_TIME(str);
        int num_threads;
        int num_blocks;
        size_t shmem;

        int kernel_offset = num_mp4_terms_single + num_mp4_terms_double;

        real_t* d_contrib;
        cudaMalloc((void**)&d_contrib, sizeof(real_t));
        cudaMemset(d_contrib, 0.0, sizeof(real_t));        

        // Launch the kernel for triple excitation terms
        for(int i=0; i<num_mp4_terms_triple; i++){
            // Get the number of blocks, threads, and shared memory size for triple excitation terms
            get_mp4_term_num_block_thread_shmem(kernel_offset + i, num_spin_occ, num_spin_vir, num_threads, num_blocks, shmem);

            mp4_term_vir2_kernel_t kernel = mp4_term_kernels_triple[i];
            for(int v1=0; v1<num_spin_vir; v1++){
                for(int v2=0; v2<num_spin_vir; v2++){
                    kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir, v1, v2, d_contrib);
                }
            }
            cudaDeviceSynchronize();

            real_t h_contrib;
            cudaMemcpy(&h_contrib, d_contrib, sizeof(real_t), cudaMemcpyDeviceToHost);

            std::cout << "  E3 " << i+1 << " contribution: " << h_contrib << " Hartree" << std::endl;

            h_mp4_energy[2] += h_contrib;
        }

        std::cout << "Total MP4 Triple excitation energy: " << h_mp4_energy[2] << " Hartree" << std::endl;
    }
    { // quadruple excitation terms
        std::string str = "Launch kernels of MP4 quadruple excitation terms... ";
        PROFILE_ELAPSED_TIME(str);
        int num_threads;
        int num_blocks;
        size_t shmem;

        int kernel_offset = num_mp4_terms_single + num_mp4_terms_double + num_mp4_terms_triple;

        real_t* d_contrib;
        cudaMalloc((void**)&d_contrib, sizeof(real_t));
        cudaMemset(d_contrib, 0.0, sizeof(real_t));        

        // Launch the kernel for quadruple excitation terms
        for(int i=0; i<num_mp4_terms_quadrule; i++){
            // Get the number of blocks, threads, and shared memory size for quadruple excitation terms
            get_mp4_term_num_block_thread_shmem(kernel_offset + i, num_spin_occ, num_spin_vir, num_threads, num_blocks, shmem);

            mp4_term_kernel_t kernel = mp4_term_kernels_quadrule[i];
            kernel<<<num_blocks, num_threads, shmem>>>(d_eri_mo, d_orbital_energies, num_basis, num_spin_occ, num_spin_vir, d_contrib);
            cudaDeviceSynchronize();

            real_t h_contrib;
            cudaMemcpy(&h_contrib, d_contrib, sizeof(real_t), cudaMemcpyDeviceToHost);

            std::cout << "  E4 " << i+1 << " contribution: " << h_contrib << " Hartree" << std::endl;

            h_mp4_energy[3] += h_contrib;
        }

        std::cout << "Total MP4 Quadruple excitation energy: " << h_mp4_energy[3] << " Hartree" << std::endl;
    }

    real_t mp4_corr_energy = 0.0;
    std::cout << "  E_S = " << h_mp4_energy[0] << " Hartree" << std::endl;
    std::cout << "  E_D = " << h_mp4_energy[1] << " Hartree" << std::endl;
    std::cout << "  E_T = " << h_mp4_energy[2] << " Hartree" << std::endl;
    std::cout << "  E_Q = " << h_mp4_energy[3] << " Hartree" << std::endl;

    for(int i=0; i<4; i++){
        mp4_corr_energy += h_mp4_energy[i];
    }
    //std::cout << "E_MP4 = E_S + E_D + E_T + E_Q = " << mp4_corr_energy << " Hartree" << std::endl;



    cudaFree(d_eri_mo);




    std::cout << "MP4 correlation energy: " << mp4_corr_energy << " Hartree" << std::endl;
    real_t mp4_total_energy = mp3_energy + mp4_corr_energy;
    std::cout << "MP4 total energy: " << mp4_total_energy << " Hartree" << std::endl;

    return mp4_total_energy;
}




real_t ERI_Stored_RHF::compute_mp4_energy() {
    PROFILE_FUNCTION();

    // Naive implementation for MP4 energy calculation 

    const int num_occ = rhf_.get_num_electrons() / 2; // number of occupied orbitals for RHF
    const int num_basis = rhf_.get_num_basis();
    DeviceHostMatrix<real_t>& coefficient_matrix = rhf_.get_coefficient_matrix();
    DeviceHostMemory<real_t>& orbital_energies = rhf_.get_orbital_energies();
    const real_t* d_C = coefficient_matrix.device_ptr();
    const real_t* d_eps = orbital_energies.device_ptr();
    const real_t* d_eri = eri_matrix_.device_ptr();


//    real_t E_MP4 = mp4_from_aoeri_via_full_moeri(d_eri, d_C, d_eps, num_basis, num_occ);
    real_t E_MP4 = mp4_from_aoeri_via_full_moeri_factorization(d_eri, d_C, d_eps, num_basis, num_occ);

//    if(fabs(E_MP2_naive - E_MP2_stored) > 1e-8){
//        std::cerr << "Warning: MP2 energy mismatch between naive and stored MOERI methods." << std::endl;
//        std::cerr << "  E_MP2_naive  = " << E_MP2_naive << std::endl;
//        std::cerr << "  E_MP2_stored = " << E_MP2_stored << std::endl;
//    }

    return E_MP4;
}



} // namespace gansu
