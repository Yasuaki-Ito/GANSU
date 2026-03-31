#ifndef AO2MO_CUH
#define AO2MO_CUH

#include "eri_stored.hpp"

#define FULLMASK 0xffffffff

namespace gansu {





__device__ inline size_t q2s(int mu, int nu, int la, int si, int num_orbitals)
{
    return ((size_t)num_orbitals * num_orbitals * num_orbitals) * mu + \
           (num_orbitals * num_orbitals) * nu + \
           (num_orbitals) * la + si;
}

__device__ inline size_t ovov2s(int i, int a, int j, int b, int num_occ, int num_vir)
{
    return ((size_t)num_occ * num_vir * num_vir) * i + \
           (num_occ * num_vir) * (a - num_occ) + \
           (num_vir) * j + (b - num_occ);
}

__device__ inline size_t oovv2s(int i, int j, int a, int b, int num_occ, int num_vir)
{
    return ((size_t)num_vir * num_vir * num_occ) * i + \
           (num_vir * num_vir) * j + \
           (num_vir) * (a - num_occ) + \
           (b - num_occ);
}

__device__ inline size_t vvoo2s(int c, int d, int i, int j, int num_occ, int num_vir)
{
    return ((size_t)num_occ * num_occ * num_vir) * (c - num_occ) + \
           (num_occ * num_occ) * (d - num_occ) + \
           (num_occ) * i + j;
}

__device__ inline size_t ovvo2s(int k, int c, int b, int j, int num_occ, int num_vir)
{
    return ((size_t)num_vir * num_vir * num_occ) * k + \
           (num_vir * num_occ) * (c - num_occ) + \
           (num_occ) * (b - num_occ) + j;
}

__device__ inline size_t oooo2s(int i, int j, int k, int l, int num_occ)
{
    return ((size_t)num_occ * num_occ * num_occ) * i + \
           (num_occ * num_occ) * j + \
           (num_occ) * k + l;
}

__device__ inline size_t vvvv2s(int a, int b, int c, int d, int num_occ, int num_vir)
{
    return ((size_t)num_vir * num_vir * num_vir) * (a - num_occ) + \
           (num_vir * num_vir) * (b - num_occ) + \
           (num_vir) * (c - num_occ) + (d - num_occ);
}





// for aaaa contributions

static __global__
void tensorize_oooo(
    double* g_int2e, double* g_oooo, const int num_occ, const int num_vir)
{
    const long long ijkl = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_oooo = (long long)num_occ * num_occ * num_occ * num_occ;
    if (ijkl >= num_oooo) {
        return;
    }

    const int ij = ijkl / (num_occ * num_occ);
    const int kl = ijkl % (num_occ * num_occ);
    const int i = ij / num_occ;
    const int j = ij % num_occ;
    const int k = kl / num_occ;
    const int l = kl % num_occ;

    const int num_orbitals = num_occ + num_vir;

    g_oooo[oooo2s(i, j, k, l, num_occ)] = g_int2e[q2s(i, k, j, l, num_orbitals)];
}


static __global__
void tensorize_vvvv(
    double* g_int2e, double* g_vvvv, const int num_occ, const int num_vir)
{
    const long long abcd = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_vvvv = (long long)num_vir * num_vir * num_vir * num_vir;
    if (abcd >= num_vvvv) {
        return;
    }

    const int ab = abcd / (num_vir * num_vir);
    const int cd = abcd % (num_vir * num_vir);
    const int a = ab / num_vir + num_occ;
    const int b = ab % num_vir + num_occ;
    const int c = cd / num_vir + num_occ;
    const int d = cd % num_vir + num_occ;

    const int num_orbitals = num_occ + num_vir;

    g_vvvv[vvvv2s(a, b, c, d, num_occ, num_vir)] = g_int2e[q2s(a, c, b, d, num_orbitals)];
}


static __global__
void tensorize_ovov(
    double* g_int2e, const double* g_eps, double* g_s_ovov, double* g_t_ovov, 
    const int num_occ, const int num_vir)
{
    const long long iajb = ((long long)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const long long num_unique_elements = (long long)num_occ * num_vir * num_occ * num_vir;
    if (iajb >= num_unique_elements) {
        return;
    }

    const int ia = iajb / (num_occ * num_vir);
    const int jb = iajb % (num_occ * num_vir);
    const int i = ia / num_vir;
    const int a = ia % num_vir + num_occ;
    const int j = jb / num_vir;
    const int b = jb % num_vir + num_occ;

    const int num_orbitals = num_occ + num_vir;
    const double int2e_iajb = g_int2e[q2s(i, a, j, b, num_orbitals)];
    const double int2e_ibja = g_int2e[q2s(i, b, j, a, num_orbitals)];
    const double eps_ijab = g_eps[i] + g_eps[j] - g_eps[a] - g_eps[b];

    g_s_ovov[ovov2s(i, a, j, b, num_occ, num_vir)] = int2e_iajb / eps_ijab;
    g_t_ovov[ovov2s(i, a, j, b, num_occ, num_vir)] = (2 * int2e_iajb - int2e_ibja) / eps_ijab;
}




static __global__ void tensorize_g_aaaa_oooo(
    double* g_g_aaaa_oooo, const double* g_g_aaaa_full, 
    const int num_occ_al, const int num_vir_al)
{
    const size_t ijkl = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oooo = (size_t)num_occ_al * num_occ_al * num_occ_al * num_occ_al;
    if (ijkl >= num_oooo) {
        return;
    }
    const int num_orbitals = num_occ_al + num_vir_al;

    const int ij = ijkl / (num_occ_al * num_occ_al);
    const int kl = ijkl % (num_occ_al * num_occ_al);
    const int i = ij / num_occ_al;
    const int j = ij % num_occ_al;
    const int k = kl / num_occ_al;
    const int l = kl % num_occ_al;

    g_g_aaaa_oooo[oooo2s(i, j, k, l, num_occ_al)] = g_g_aaaa_full[q2s(i, k, j, l, num_orbitals)];
}


static __global__ void tensorize_g_aaaa_vvvv(
    double* g_g_aaaa_vvvv, const double* g_g_aaaa_full,
    const int num_occ_al, const int num_vir_al)
{
    const size_t abcd = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_vvvv = (size_t)num_vir_al * num_vir_al * num_vir_al * num_vir_al;
    if (abcd >= num_vvvv) {
        return;
    }
    const int num_orbitals = num_occ_al + num_vir_al;

    const int ab = abcd / (num_vir_al * num_vir_al);
    const int cd = abcd % (num_vir_al * num_vir_al);
    const int a = ab / num_vir_al + num_occ_al;
    const int b = ab % num_vir_al + num_occ_al;
    const int c = cd / num_vir_al + num_occ_al;
    const int d = cd % num_vir_al + num_occ_al;

    g_g_aaaa_vvvv[vvvv2s(a, b, c, d, num_occ_al, num_vir_al)] = g_g_aaaa_full[q2s(a, c, b, d, num_orbitals)];
}


static __global__ void tensorize_u_aaaa_ovvo(
    double* g_u_aaaa_ovvo, const double* g_g_aaaa_full,
    const int num_occ_al, const int num_vir_al)
{
    const size_t kcbj = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovvo = (size_t)num_occ_al * num_vir_al * num_vir_al * num_occ_al;
    if (kcbj >= num_ovvo) {
        return;
    }
    const int num_orbitals = num_occ_al + num_vir_al;

    const int kc = kcbj / (num_vir_al * num_occ_al);
    const int bj = kcbj % (num_vir_al * num_occ_al);
    const int k = kc / num_vir_al;
    const int c = kc % num_vir_al + num_occ_al;
    const int b = bj / num_occ_al + num_occ_al;
    const int j = bj % num_occ_al;

    const double g_kcbj = g_g_aaaa_full[q2s(k, c, b, j, num_orbitals)];
    const double g_kjbc = g_g_aaaa_full[q2s(k, j, b, c, num_orbitals)];

    g_u_aaaa_ovvo[ovvo2s(k, c, b, j, num_occ_al, num_vir_al)] = g_kcbj - g_kjbc;
}


static __global__ void tensorize_x_aaaa_ovov(
    double* g_x_aaaa_ovov, const double* g_g_aaaa_full, 
    const double* g_eps_al, const int num_occ_al, const int num_vir_al)
{
    const size_t iajb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
    if (iajb >= num_ovov) {
        return;
    }
    const int num_orbitals = num_occ_al + num_vir_al;

    const int ia = iajb / (num_occ_al * num_vir_al);
    const int jb = iajb % (num_occ_al * num_vir_al);
    const int i = ia / num_vir_al;
    const int a = ia % num_vir_al + num_occ_al;
    const int j = jb / num_vir_al;
    const int b = jb % num_vir_al + num_occ_al;

    const double g_iajb = g_g_aaaa_full[q2s(i, a, j, b, num_orbitals)];
    const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];

    g_x_aaaa_ovov[ovov2s(i, a, j, b, num_occ_al, num_vir_al)] = g_iajb / eps_ijab;
}


static __global__ void tensorize_y_aaaa_ovov(
    double* g_y_aaaa_ovov, const double* g_g_aaaa_full,
    const double* g_eps_al, const int num_occ_al, const int num_vir_al)
{
    const size_t iajb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
    if (iajb >= num_ovov) {
        return;
    }
    const int num_orbitals = num_occ_al + num_vir_al;

    const int ia = iajb / (num_occ_al * num_vir_al);
    const int jb = iajb % (num_occ_al * num_vir_al);
    const int i = ia / num_vir_al;
    const int a = ia % num_vir_al + num_occ_al;
    const int j = jb / num_vir_al;
    const int b = jb % num_vir_al + num_occ_al;

    const double g_iajb = g_g_aaaa_full[q2s(i, a, j, b, num_orbitals)];
    const double g_ibja = g_g_aaaa_full[q2s(i, b, j, a, num_orbitals)];
    const double eps_ijab = g_eps_al[i] + g_eps_al[j] - g_eps_al[a] - g_eps_al[b];

    g_y_aaaa_ovov[ovov2s(i, a, j, b, num_occ_al, num_vir_al)] = (g_iajb - g_ibja) / eps_ijab;
}


static __global__ void kalb2klab_aaaa(
    double* d_aaaa_oovv, const double* d_aaaa_ovov, 
    const int num_occ_al, const int num_vir_al)
{
    const size_t kalb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
    if (kalb >= num_ovov) {
        return;
    }

    const int ka = kalb / (num_occ_al * num_vir_al);
    const int lb = kalb % (num_occ_al * num_vir_al);
    const int k = ka / num_vir_al;
    const int a = ka % num_vir_al + num_occ_al;
    const int l = lb / num_vir_al;
    const int b = lb % num_vir_al + num_occ_al;

    d_aaaa_oovv[oovv2s(k, l, a, b, num_occ_al, num_vir_al)] = d_aaaa_ovov[ovov2s(k, a, l, b, num_occ_al, num_vir_al)];
}

static __global__ void icjd2cdij_aaaa(
    double* d_aaaa_vvoo, const double* d_aaaa_ovov, 
    const int num_occ_al, const int num_vir_al)
{
    const size_t icjd = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_al * num_vir_al;
    if (icjd >= num_ovov) {
        return;
    }

    const int ic = icjd / (num_occ_al * num_vir_al);
    const int jd = icjd % (num_occ_al * num_vir_al);
    const int i = ic / num_vir_al;
    const int c = ic % num_vir_al + num_occ_al;
    const int j = jd / num_vir_al;
    const int d = jd % num_vir_al + num_occ_al;

    d_aaaa_vvoo[vvoo2s(c, d, i, j, num_occ_al, num_vir_al)] = d_aaaa_ovov[ovov2s(i, c, j, d, num_occ_al, num_vir_al)];
}






// for aabb contributions

__device__ inline size_t oooo2s_abab(
    int i, int j, int k, int l, int num_occ_al, int num_occ_be)
{
    return ((size_t)num_occ_be * num_occ_al * num_occ_be) * i + \
           ((size_t)num_occ_al * num_occ_be) * j + \
           (num_occ_be) * k + l;
}

__device__ inline size_t vvvv2s_abab(
    int a, int b, int c, int d, int num_occ_al, int num_occ_be, int num_vir_al, int num_vir_be)
{
    return ((size_t)num_vir_be * num_vir_al * num_vir_be) * (a - num_occ_al) + \
           ((size_t)num_vir_al * num_vir_be) * (b - num_occ_be) + \
           (num_vir_be) * (c - num_occ_al) + (d - num_occ_be);
}

__device__ inline size_t ovvo2s_bbaa(
    int k, int c, int b, int j, int num_occ_be, int num_occ_al, int num_vir_be, int num_vir_al)
{
    return ((size_t)num_vir_be * num_vir_al * num_occ_al) * k + \
           ((size_t)num_vir_al * num_occ_al) * (c - num_occ_be) + \
           (num_occ_al) * (b - num_occ_al) + j;
}

__device__ inline size_t ovvo2s_baab(
    int k, int c, int b, int j, int num_occ_be, int num_occ_al, int num_vir_al, int num_vir_be)
{
    return ((size_t)num_vir_al * num_vir_al * num_occ_be) * k + \
           ((size_t)num_vir_al * num_occ_be) * (c - num_occ_al) + \
           (num_occ_be) * (b - num_occ_al) + j;
}

__device__ inline size_t ovvo2s_aabb(
    int i, int a, int b, int j, int num_occ_al, int num_vir_al, int num_vir_be, int num_occ_be)
{
    return ((size_t)num_vir_al * num_vir_be * num_occ_be) * i + \
           ((size_t)num_vir_be * num_occ_be) * (a - num_occ_al) + \
           (num_occ_be) * (b - num_occ_be) + j;
}

__device__ inline size_t ovov2s_aabb(
    int i, int a, int j, int b, int num_occ_al, int num_vir_al, int num_occ_be, int num_vir_be)
{
    return ((size_t)num_vir_al * num_occ_be * num_vir_be) * i + \
           ((size_t)num_occ_be * num_vir_be) * (a - num_occ_al) + \
           (num_vir_be) * j + (b - num_occ_be);
}




static __global__ void tensorize_g_aabb_oooo(
    double* g_g_aabb_oooo, const double* g_g_aabb_full,
    const int num_occ_al, const int num_occ_be, const int num_basis)
{
    const size_t ijkl = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oooo = (size_t)num_occ_al * num_occ_be * num_occ_al * num_occ_be;
    if (ijkl >= num_oooo) {
        return;
    }

    const int ij = ijkl / (num_occ_al * num_occ_be);
    const int kl = ijkl % (num_occ_al * num_occ_be);
    const int i = ij / num_occ_be;
    const int j = ij % num_occ_be;
    const int k = kl / num_occ_be;
    const int l = kl % num_occ_be;

    g_g_aabb_oooo[oooo2s_abab(i, j, k, l, num_occ_al, num_occ_be)] = g_g_aabb_full[q2s(i, k, j, l, num_basis)];
}


static __global__ void tensorize_g_aabb_vvvv(
    double* g_g_aabb_vvvv, const double* g_g_aabb_full,
    const int num_occ_al, const int num_occ_be, 
    const int num_vir_al, const int num_vir_be, const int num_basis)
{
    const size_t abcd = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_vvvv = (size_t)num_vir_al * num_vir_be * num_vir_al * num_vir_be;
    if (abcd >= num_vvvv) {
        return;
    }

    const int ab = abcd / (num_vir_al * num_vir_be);
    const int cd = abcd % (num_vir_al * num_vir_be);
    const int a = ab / num_vir_be + num_occ_al;
    const int b = ab % num_vir_be + num_occ_be;
    const int c = cd / num_vir_be + num_occ_al;
    const int d = cd % num_vir_be + num_occ_be;

    g_g_aabb_vvvv[vvvv2s_abab(a, b, c, d, num_occ_al, num_occ_be, num_vir_al, num_vir_be)] = g_g_aabb_full[q2s(a, c, b, d, num_basis)];
}


static __global__ void tensorize_g_bbaa_ovvo(
    double* g_g_bbaa_ovvo, const double* g_g_bbaa_full,
    const int num_occ_be, const int num_occ_al,
    const int num_vir_be, const int num_vir_al, const int num_basis)
{
    const size_t kcbj = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovvo = (size_t)num_occ_be * num_vir_be * num_vir_al * num_occ_al;
    if (kcbj >= num_ovvo) {
        return;
    }

    const int kc = kcbj / (num_vir_al * num_occ_al);
    const int bj = kcbj % (num_vir_al * num_occ_al);
    const int k = kc / num_vir_be;
    const int c = kc % num_vir_be + num_occ_be;
    const int b = bj / num_occ_al + num_occ_al;
    const int j = bj % num_occ_al;

    g_g_bbaa_ovvo[ovvo2s_bbaa(k, c, b, j, num_occ_be, num_occ_al, num_vir_be, num_vir_al)] = g_g_bbaa_full[q2s(k, c, b, j, num_basis)];
}


static __global__ void tensorize_g_bbaa_oovv(
    double* g_g_bbaa_oovv, const double* g_g_bbaa_full,
    const int num_occ_be, const int num_occ_al,
    const int num_vir_be, const int num_vir_al, const int num_basis)
{
    const size_t kjbc = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_be * num_occ_be * num_vir_al * num_vir_al;
    if (kjbc >= num_oovv) {
        return;
    }

    const int kj = kjbc / (num_vir_al * num_vir_al);
    const int bc = kjbc % (num_vir_al * num_vir_al);
    const int k = kj / num_occ_be;
    const int j = kj % num_occ_be;
    const int b = bc / num_vir_al + num_occ_al;
    const int c = bc % num_vir_al + num_occ_al;

    g_g_bbaa_oovv[ovvo2s_baab(k, c, b, j, num_occ_be, num_occ_al, num_vir_al, num_vir_be)] = g_g_bbaa_full[q2s(k, j, b, c, num_basis)];
}


static __global__ void tensorize_u_bbbb_ovvo(
    double* g_u_bbbb_ovvo, const double* g_g_bbbb_full,
    const int num_occ_be, const int num_vir_be)
{
    const size_t kcbj = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovvo = (size_t)num_occ_be * num_vir_be * num_vir_be * num_occ_be;
    if (kcbj >= num_ovvo) {
        return;
    }
    const int num_orbitals = num_occ_be + num_vir_be;

    const int kc = kcbj / (num_vir_be * num_occ_be);
    const int bj = kcbj % (num_vir_be * num_occ_be);
    const int k = kc / num_vir_be;
    const int c = kc % num_vir_be + num_occ_be;
    const int b = bj / num_occ_be + num_occ_be;
    const int j = bj % num_occ_be;

    const double g_kcbj = g_g_bbbb_full[q2s(k, c, b, j, num_orbitals)];
    const double g_kjbc = g_g_bbbb_full[q2s(k, j, b, c, num_orbitals)];

    g_u_bbbb_ovvo[ovvo2s(k, c, b, j, num_occ_be, num_vir_be)] = g_kcbj - g_kjbc;
}


static __global__ void tensorize_x_aabb_ovov(
    double* g_x_aabb_ovov, const double* g_g_aabb_full,
    const double* g_eps_al, const double* g_eps_be,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be, const int num_basis)
{
    const size_t iajb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    if (iajb >= num_ovov) {
        return;
    }

    const int ia = iajb / (num_occ_be * num_vir_be);
    const int jb = iajb % (num_occ_be * num_vir_be);
    const int i = ia / num_vir_al;
    const int a = ia % num_vir_al + num_occ_al;
    const int j = jb / num_vir_be;
    const int b = jb % num_vir_be + num_occ_be;

    const double g_iajb = g_g_aabb_full[q2s(i, a, j, b, num_basis)];
    const double eps_ijab = g_eps_al[i] + g_eps_be[j] - g_eps_al[a] - g_eps_be[b];

    g_x_aabb_ovov[ovov2s_aabb(i, a, j, b, num_occ_al, num_vir_al, num_occ_be, num_vir_be)] = g_iajb / eps_ijab;
}




static __device__ inline 
double warp_reduce_sum_32(double x)
{
    for (int offset = 16; offset > 0; offset /= 2) {
        x += __shfl_down_sync(FULLMASK, x, offset);
    }
    return x;
}

static __device__ inline 
void accumulate_block_sum_2d(
    double x, double* shared_sum, double* global_sum)
{
    x = warp_reduce_sum_32(x);

    if (threadIdx.x == 0) {
        atomicAdd(shared_sum, x);
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(global_sum, *shared_sum);
    }
}



static __global__ void contract_3h3p_aaaaaa(
    double* g_energy_3h3p, 
    const double* g_y_aaaa_ovov, const double* g_t_aaaa_ovvo,
    const int num_occ_al, const int num_vir_al)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_al * num_vir_al * num_vir_al;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_al * num_vir_al);
        const int ab = ijab % (num_vir_al * num_vir_al);
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int a = ab / num_vir_al + num_occ_al;
        const int b = ab % num_vir_al + num_occ_al;

        const double y_aaaa_iajb = g_y_aaaa_ovov[ovov2s(i, a, j, b, num_occ_al, num_vir_al)];
        const double t_aaaa_iabj = g_t_aaaa_ovvo[ovvo2s(i, a, b, j, num_occ_al, num_vir_al)];
        energy = y_aaaa_iajb * t_aaaa_iabj;
    }

    accumulate_block_sum_2d(energy, &s_energy_3h3p, g_energy_3h3p);
}


static __global__ void contract_4h2p_2h4p_aaaaaa(
    double* g_energy_4h2p_2h4p, const double* g_x_aaaa_ovov, 
    const double* g_t_aaaa_oovv, const double* g_t_aaaa_vvoo,
    const int num_occ_al, const int num_vir_al)
{
    __shared__ double s_energy_4h2p_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_4h2p_2h4p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_al * num_vir_al * num_vir_al;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_al * num_vir_al);
        const int ab = ijab % (num_vir_al * num_vir_al);
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int a = ab / num_vir_al + num_occ_al;
        const int b = ab % num_vir_al + num_occ_al;

        const double x_aaaa_iajb = g_x_aaaa_ovov[ovov2s(i, a, j, b, num_occ_al, num_vir_al)];
        const double t_aaaa_ijab = g_t_aaaa_oovv[oovv2s(i, j, a, b, num_occ_al, num_vir_al)];
        const double t_aaaa_abij = g_t_aaaa_vvoo[vvoo2s(a, b, i, j, num_occ_al, num_vir_al)];
        energy = (0.5) * x_aaaa_iajb * (t_aaaa_ijab + t_aaaa_abij);
    }

    accumulate_block_sum_2d(energy, &s_energy_4h2p_2h4p, g_energy_4h2p_2h4p);
}






static __global__ void contract_3h3p_aabaab_abaaba(
    double* g_energy_3h3p, 
    const double* g_y_aaaa_ovov, const double* g_t_aaaa_ovvo,
    const int num_occ_al, const int num_vir_al)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_al * num_vir_al * num_vir_al;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_al * num_vir_al);
        const int ab = ijab % (num_vir_al * num_vir_al);
        const int i = ij / num_occ_al;
        const int j = ij % num_occ_al;
        const int a = ab / num_vir_al + num_occ_al;
        const int b = ab % num_vir_al + num_occ_al;

        const double y_aaaa_iajb = g_y_aaaa_ovov[ovov2s(i, a, j, b, num_occ_al, num_vir_al)];
        const double t_aaaa_iabj = g_t_aaaa_ovvo[ovvo2s(i, a, b, j, num_occ_al, num_vir_al)];
        energy = 2.0 * y_aaaa_iajb * t_aaaa_iabj;
    }

    accumulate_block_sum_2d(energy, &s_energy_3h3p, g_energy_3h3p);
}


static __global__ void contract_3h3p_abbabb(
    double* g_energy_3h3p, 
    const double* g_x_aabb_ovov, const double* g_t_aabb_ovvo,
    const int num_occ_al, const int num_occ_be, const int num_vir_al, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_be * num_vir_al * num_vir_be;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_al * num_vir_be);
        const int ab = ijab % (num_vir_al * num_vir_be);
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int a = ab / num_vir_be + num_occ_al;
        const int b = ab % num_vir_be + num_occ_be;

        const double x_aabb_iajb = g_x_aabb_ovov[ovov2s_aabb(i, a, j, b, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
        const double t_aabb_iabj = g_t_aabb_ovvo[ovvo2s_aabb(i, a, b, j, num_occ_al, num_vir_al, num_vir_be, num_occ_be)];
        energy = x_aabb_iajb * t_aabb_iabj;
    }

    accumulate_block_sum_2d(energy, &s_energy_3h3p, g_energy_3h3p);
}


static __global__ void aabb_icka2abba_iakc(
    double* d_abba_ovov, const double* d_aabb_ovov,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t icka = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    if (icka >= num_ovov) {
        return;
    }

    const int ic = icka / (num_occ_be * num_vir_be);
    const int ka = icka % (num_occ_be * num_vir_be);
    const int i = ic / num_vir_al;
    const int c = ic % num_vir_al + num_occ_al;
    const int k = ka / num_vir_be;
    const int a = ka % num_vir_be + num_occ_be;

    const size_t iakc =
        ((size_t)num_vir_be * num_occ_be * num_vir_al) * i +
        ((size_t)num_occ_be * num_vir_al) * (a - num_occ_be) +
        (num_vir_al) * k + (c - num_occ_al);

    d_abba_ovov[iakc] = d_aabb_ovov[ovov2s_aabb(i, c, k, a, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
}


static __global__ void contract_3h3p_abbbaa(
    double* g_energy_3h3p,
    const double* g_x_aabb_ovov, const double* g_t_abab_ovvo,
    const int num_occ_al, const int num_occ_be, const int num_vir_al, const int num_vir_be)
{
    __shared__ double s_energy_3h3p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_3h3p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_be * num_vir_be * num_vir_al;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_be * num_vir_al);
        const int ab = ijab % (num_vir_be * num_vir_al);
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int a = ab / num_vir_al + num_occ_be;
        const int b = ab % num_vir_al + num_occ_al;

        const double x_aabb_ibja = g_x_aabb_ovov[ovov2s_aabb(i, b, j, a, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
        const size_t iabj =
            ((size_t)num_vir_be * num_vir_al * num_occ_be) * i +
            ((size_t)num_vir_al * num_occ_be) * (a - num_occ_be) +
            (num_occ_be) * (b - num_occ_al) + j;
        const double t_abab_iabj = g_t_abab_ovvo[iabj];
        energy = (-1) * x_aabb_ibja * t_abab_iabj;
    }

    accumulate_block_sum_2d(energy, &s_energy_3h3p, g_energy_3h3p);
}


static __global__ void aabb_kalb2abab_klab(
    double* d_abab_oovv, const double* d_aabb_ovov,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t kalb = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    if (kalb >= num_ovov) {
        return;
    }

    const int ka = kalb / (num_occ_be * num_vir_be);
    const int lb = kalb % (num_occ_be * num_vir_be);
    const int k = ka / num_vir_al;
    const int a = ka % num_vir_al + num_occ_al;
    const int l = lb / num_vir_be;
    const int b = lb % num_vir_be + num_occ_be;

    const size_t klab =
        ((size_t)num_occ_be * num_vir_al * num_vir_be) * k +
        ((size_t)num_vir_al * num_vir_be) * l +
        (num_vir_be) * (a - num_occ_al) + (b - num_occ_be);

    d_abab_oovv[klab] = d_aabb_ovov[ovov2s_aabb(k, a, l, b, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
}


static __global__ void aabb_icjd2abab_cdij(
    double* d_abab_vvoo, const double* d_aabb_ovov,
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t icjd = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_ovov = (size_t)num_occ_al * num_vir_al * num_occ_be * num_vir_be;
    if (icjd >= num_ovov) {
        return;
    }

    const int ic = icjd / (num_occ_be * num_vir_be);
    const int jd = icjd % (num_occ_be * num_vir_be);
    const int i = ic / num_vir_al;
    const int c = ic % num_vir_al + num_occ_al;
    const int j = jd / num_vir_be;
    const int d = jd % num_vir_be + num_occ_be;

    const size_t cdij =
        ((size_t)num_vir_be * num_occ_al * num_occ_be) * (c - num_occ_al) +
        ((size_t)num_occ_al * num_occ_be) * (d - num_occ_be) +
        (num_occ_be) * i + j;

    d_abab_vvoo[cdij] = d_aabb_ovov[ovov2s_aabb(i, c, j, d, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
}



__device__ inline size_t oovv2s_abab(
    int i, int j, int a, int b,
    int num_occ_al, int num_occ_be, int num_vir_al, int num_vir_be)
{
    return ((size_t)num_occ_be * num_vir_al * num_vir_be) * i +
           ((size_t)num_vir_al * num_vir_be) * j +
           (num_vir_be) * (a - num_occ_al) + (b - num_occ_be);
}

__device__ inline size_t vvoo2s_abab(
    int a, int b, int i, int j,
    int num_occ_al, int num_occ_be, int num_vir_al, int num_vir_be)
{
    return ((size_t)num_vir_be * num_occ_al * num_occ_be) * (a - num_occ_al) +
           ((size_t)num_occ_al * num_occ_be) * (b - num_occ_be) +
           (num_occ_be) * i + j;
}


static __global__ void contract_4h2p_2h4p_ababab_bababa(
    double* g_energy_4h2p_2h4p, 
    const double* g_x_aabb_ovov, const double* g_t_abab_oovv, const double* g_t_abab_vvoo,
    const int num_occ_al, const int num_occ_be, const int num_vir_al, const int num_vir_be)
{
    __shared__ double s_energy_4h2p_2h4p;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_energy_4h2p_2h4p = 0;
    }
    __syncthreads();

    const size_t ijab = ((size_t)blockDim.x * blockDim.y) * (gridDim.x * blockIdx.y + blockIdx.x) + (blockDim.x * threadIdx.y + threadIdx.x);
    const size_t num_oovv = (size_t)num_occ_al * num_occ_be * num_vir_al * num_vir_be;
    double energy = 0.0;
    if (ijab < num_oovv) {
        const int ij = ijab / (num_vir_al * num_vir_be);
        const int ab = ijab % (num_vir_al * num_vir_be);
        const int i = ij / num_occ_be;
        const int j = ij % num_occ_be;
        const int a = ab / num_vir_be + num_occ_al;
        const int b = ab % num_vir_be + num_occ_be;

        const double x_aabb_iajb = g_x_aabb_ovov[ovov2s_aabb(i, a, j, b, num_occ_al, num_vir_al, num_occ_be, num_vir_be)];
        const double t_abab_ijab = g_t_abab_oovv[oovv2s_abab(i, j, a, b, num_occ_al, num_occ_be, num_vir_al, num_vir_be)];
        const double t_abab_abij = g_t_abab_vvoo[vvoo2s_abab(a, b, i, j, num_occ_al, num_occ_be, num_vir_al, num_vir_be)];
        energy = x_aabb_iajb * (t_abab_ijab + t_abab_abij);
    }

    accumulate_block_sum_2d(energy, &s_energy_4h2p_2h4p, g_energy_4h2p_2h4p);
}












static inline cublasStatus_t dgemm_device_row_major(
    cublasHandle_t h,
    cublasOperation_t opA_rm, cublasOperation_t opB_rm,
    int m, int n, int k,
    const double* alpha,
    const double* A_rm, int lda_rm,
    const double* B_rm, int ldb_rm,
    const double* beta,
    double* C_rm, int ldc_rm
){
    // Row-major: C = opA(A) * opB(B)  (m×n)
    // Column-major view: C^T = opB(B)^T * opA(A)^T  (n×m)
    //
    // Memory trick:
    //   A_rm (row-major) is A_cm = A_rm^T (col-major)
    //   B_rm (row-major) is B_cm = B_rm^T (col-major)
    //   C_rm (row-major) is C_cm = C_rm^T (col-major)
    //
    // Then:
    //   C_cm = op(B_cm) * op(A_cm)
    // where op(B_cm) uses the SAME op flag as opB_rm (not flipped),
    // and op(A_cm) uses the SAME op flag as opA_rm.

    return cublasDgemm(
        h,
        /*transa=*/opB_rm,  /*transb=*/opA_rm,
        /*m=*/n, /*n=*/m, /*k=*/k,
        alpha,
        B_rm, ldb_rm,
        A_rm, lda_rm,
        beta,
        C_rm, ldc_rm
    );
}


static inline cublasStatus_t dgemm_strided_batched_device_row_major(
    cublasHandle_t h,
    cublasOperation_t opA_rm, cublasOperation_t opB_rm,
    int m, int n, int k,
    const double* alpha,
    const double* A_rm, int lda_rm, long long int strideA_rm,
    const double* B_rm, int ldb_rm, long long int strideB_rm,
    const double* beta,
    double* C_rm, int ldc_rm, long long int strideC_rm,
    int batchCount
){
    // Row-major:  C = opA(A) * opB(B)     (m×n)
    // Transpose:  C^T = opB(B)^T * opA(A)^T  (n×m)
    //
    // Memory trick:
    //   A_rm memory == A_cm = A_rm^T in column-major
    //   B_rm memory == B_cm = B_rm^T in column-major
    //   C_rm memory == C_cm = C_rm^T in column-major
    //
    // Therefore in column-major we compute:
    //   C_cm (n×m) = op(B_cm) * op(A_cm)
    // where op flags are the same as row-major op flags (not flipped).
    //
    // Leading dimensions and strides in elements carry over unchanged.

    return cublasDgemmStridedBatched(
        h,
        /*transa=*/opB_rm,  /*transb=*/opA_rm,
        /*m=*/n, /*n=*/m, /*k=*/k,
        alpha,
        /*A=*/B_rm, /*lda=*/ldb_rm, /*strideA=*/strideB_rm,
        /*B=*/A_rm, /*ldb=*/lda_rm, /*strideB=*/strideA_rm,
        beta,
        /*C=*/C_rm, /*ldc=*/ldc_rm, /*strideC=*/strideC_rm,
        batchCount
    );
}

static inline cublasStatus_t dgeam_transpose_row_major(
    cublasHandle_t h,
    int rows_rm, int cols_rm,          // A_rm is rows_rm x cols_rm (row-major)
    const double* A_rm,
    double* AT_rm                     // AT_rm is cols_rm x rows_rm (row-major)
){
    const double alpha = 1.0;
    const double beta  = 0.0;

    // row-major A_rm(rows,cols) == column-major A_cm(cols,rows)
    // want AT_rm(cols,rows) == column-major AT_cm(rows,cols)
    // AT_cm = A_cm^T

    return cublasDgeam(
        h,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/rows_rm,              // rows(AT_cm)
        /*n=*/cols_rm,              // cols(AT_cm)
        &alpha,
        A_rm, /*lda=*/cols_rm,      // lda = rows(A_cm) = cols_rm
        &beta,
        A_rm, /*ldb=*/cols_rm,
        AT_rm, /*ldc=*/rows_rm      // ldc = rows(AT_cm) = rows_rm
    );
}


// two normal dgemms and two stridedbatched dgemms
//*
inline void transform_eri_ao2mo_dgemm_full(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, const int num_basis)
{
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix, num_basis, 0,
        d_eri_mo, num_basis_2, num_basis_3,
        &beta,
        d_eri_ao, num_basis_2, num_basis_3,
        num_basis
    );

    dgeam_transpose_row_major(
        cublasH, 
        num_basis_2, num_basis_2, 
        d_eri_ao, 
        d_eri_mo);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_mo, num_basis_3, 
        &beta, 
        d_eri_ao, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix, num_basis, 0,
        d_eri_ao, num_basis_2, num_basis_3,
        &beta,
        d_eri_mo, num_basis_2, num_basis_3,
        num_basis
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/


inline void transform_eri_ao2mo_dgemm_full_os(
    double* d_eri_ao, 
    double* d_eri_mo, 
    const double* d_coefficient_matrix_al, 
    const double* d_coefficient_matrix_be, 
    const int num_basis)
{
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix_al, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix_al, num_basis, 0,
        d_eri_mo, num_basis_2, num_basis_3,
        &beta,
        d_eri_ao, num_basis_2, num_basis_3,
        num_basis
    );

    dgeam_transpose_row_major(
        cublasH, 
        num_basis_2, num_basis_2, 
        d_eri_ao, 
        d_eri_mo);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix_be, num_basis, 
        d_eri_mo, num_basis_3, 
        &beta, 
        d_eri_ao, num_basis_3
    );
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_basis, num_basis_2, num_basis,
        &alpha,
        d_coefficient_matrix_be, num_basis, 0,
        d_eri_ao, num_basis_2, num_basis_3,
        &beta,
        d_eri_mo, num_basis_2, num_basis_3,
        num_basis
    );

    // debug
    dgeam_transpose_row_major(
        cublasH, 
        num_basis_2, num_basis_2, 
        d_eri_mo, 
        d_eri_ao);
    cudaMemcpy(d_eri_mo, d_eri_ao, sizeof(double) * num_basis_2 * num_basis_2, cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}











__device__ inline size_t ovov2seq(
    const int i, const int a, const int j, const int b, 
    const int num_occupied, const int num_virtual) 
{
    return (((size_t(i) * num_virtual + a) * num_occupied + j) * num_virtual + b);
}



__device__ inline size_t ovov2seq_aabb(
    const int i, const int a, const int j, const int b, 
    const int num_occupied_al, const int num_virtual_al, 
    const int num_occupied_be, const int num_virtual_be) 
{
    return (((size_t(i) * num_virtual_al + a) * num_occupied_be + j) * num_virtual_be + b);
}



// for rmp2 and ump2 (same spin)
//*
inline void transform_eri_ao2mo_dgemm_ovov(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, 
    const int num_occ, const int num_vir)
{
    const size_t num_basis = num_occ + num_vir;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ * num_basis, 
        num_basis * num_basis, 
        d_eri_mo, 
        d_eri_ao);
    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ, num_basis_2 * num_occ, num_basis,
        &alpha, 
        d_coefficient_matrix, num_basis, 
        d_eri_ao, num_basis_2 * num_occ, 
        &beta, 
        d_eri_mo, num_basis_2 * num_occ
    );

    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_basis, num_basis,
        &alpha,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_basis, num_basis_2 * num_occ,
        &beta,
        d_eri_ao, num_occ * num_basis, num_vir * num_occ * num_basis,
        num_occ
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ * num_vir, 
        num_occ * num_basis, 
        d_eri_ao, 
        d_eri_mo);
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir, num_occ * num_vir, num_basis,
        &alpha,
        d_coefficient_matrix + num_occ, num_basis, 0,
        d_eri_mo, num_occ * num_vir, num_basis * num_occ * num_vir,
        &beta,
        d_eri_ao, num_occ * num_vir, num_vir * num_occ * num_vir,
        num_occ
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/



// for ump2 (opposite spin)
//*
inline void transform_eri_ao2mo_dgemm_ovov_os(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix_al, 
    const double* d_coefficient_matrix_be, 
    const int num_occ_al, const int num_vir_al,
    const int num_occ_be, const int num_vir_be)
{
    const size_t num_basis = num_occ_al + num_vir_al;
    const size_t num_basis_2 = num_basis * num_basis;
    const size_t num_basis_3 = num_basis_2 * num_basis;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_al, num_basis_3, num_basis,
        &alpha, 
        d_coefficient_matrix_al, num_basis, 
        d_eri_ao, num_basis_3, 
        &beta, 
        d_eri_mo, num_basis_3
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ_al * num_basis, 
        num_basis * num_basis, 
        d_eri_mo, 
        d_eri_ao);
    dgemm_device_row_major(
        cublasH, 
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_occ_be, num_basis_2 * num_occ_al, num_basis,
        &alpha, 
        d_coefficient_matrix_be, num_basis, 
        d_eri_ao, num_basis_2 * num_occ_al, 
        &beta, 
        d_eri_mo, num_basis_2 * num_occ_al
    );

    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_be, num_occ_al * num_basis, num_basis,
        &alpha,
        d_coefficient_matrix_be + num_occ_be, num_basis, 0,
        d_eri_mo, num_occ_al * num_basis, num_basis_2 * num_occ_al,
        &beta,
        d_eri_ao, num_occ_al * num_basis, num_vir_be * num_occ_al * num_basis,
        num_occ_be
    );
    dgeam_transpose_row_major(
        cublasH, 
        num_occ_be * num_vir_be, 
        num_occ_al * num_basis, 
        d_eri_ao, 
        d_eri_mo);
    dgemm_strided_batched_device_row_major(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_vir_al, num_occ_be * num_vir_be, num_basis,
        &alpha,
        d_coefficient_matrix_al + num_occ_al, num_basis, 0,
        d_eri_mo, num_occ_be * num_vir_be, num_basis * num_occ_be * num_vir_be,
        &beta,
        d_eri_ao, num_occ_be * num_vir_be, num_vir_al * num_occ_be * num_vir_be,
        num_occ_al
    );

    cudaDeviceSynchronize();
    cublasDestroy(cublasH);
}
/**/




























} // namespace gansu













/*
void transform_eri_ao2mo_dgemm_full(
    double* d_eri_ao, double* d_eri_mo, 
    const double* d_coefficient_matrix, const int num_basis)
{
    double* d_G1;
    double* d_G2;
    const int num_threads_per_block = 256;
    const size_t num_basis_sq = num_basis * num_basis;
    const size_t num_blocks = ((size_t)num_basis_sq * num_basis_sq + num_threads_per_block - 1) / num_threads_per_block;

    const double alpha = 1.0;
    const double beta = 0.0;
    cublasHandle_t cublasH = NULL;
    cublasCreate(&cublasH);

    // mu2i for (i, nu, la, si)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    swap_bra_index<<<num_blocks, num_threads_per_block>>>(d_eri_mo, d_eri_ao, num_basis);
    cudaDeviceSynchronize();

    // nu2j for (i, j, k, si)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }

    // la2k for (i, nu, k, si)
    cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, num_basis_sq, num_basis_sq,
                &alpha, d_eri_mo, num_basis_sq, &beta, d_eri_mo, num_basis_sq, d_eri_ao, num_basis_sq);
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    swap_bra_index<<<num_blocks, num_threads_per_block>>>(d_eri_mo, d_eri_ao, num_basis);
    cudaDeviceSynchronize();

    // si2l for (i, j, k, l)
    for (int i = 0; i < num_basis; ++i) {
        d_G1 = d_eri_ao + num_basis_sq * num_basis * i;
        d_G2 = d_eri_mo + num_basis_sq * num_basis * i;
        cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, num_basis_sq, num_basis, num_basis, 
                    &alpha, d_G1, num_basis_sq, d_coefficient_matrix, num_basis, &beta, d_G2, num_basis_sq);
    }
    //cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, num_basis_sq, num_basis_sq,
    //            &alpha, d_eri_mo, num_basis_sq, &beta, d_eri_mo, num_basis_sq, d_eri_ao, num_basis_sq);

    cublasDestroy(cublasH);
}
/**/






#endif // AO2MO_CUH
