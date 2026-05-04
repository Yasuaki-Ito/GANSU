/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 */

/**
 * @file ccsd_lambda.cu
 * @brief CCSD Lambda equations solver and 1-RDM (non-relaxed correlation density).
 *
 * CPU reference implementation
 * (_gamma1_intermediates) for spin-traced closed-shell RHF.
 *
 * MO integral convention: chemist's notation (pq|rs), row-major [p,q,r,s].
 * t1, l1 shape (nocc, nvir); t2, l2 shape (nocc, nocc, nvir, nvir).
 */

#include "ccsd_lambda.hpp"
#include "progress.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>

namespace gansu {

// ---------------------------------------------------------------------------
//  Index helpers (row-major 4D)
// ---------------------------------------------------------------------------

namespace {

struct LambdaDims {
    int no;  // nocc
    int nv;  // nvir
    int na;  // nao = no + nv
};

inline size_t idx_t1(const LambdaDims& d, int i, int a) {
    return (size_t)i * d.nv + a;
}
inline size_t idx_t2(const LambdaDims& d, int i, int j, int a, int b) {
    return (((size_t)i * d.no + j) * d.nv + a) * d.nv + b;
}

// Full MO ERI [na,na,na,na], chemist's (pq|rs)
inline size_t idx_eri(int na, int p, int q, int r, int s) {
    return (((size_t)p * na + q) * na + r) * na + s;
}

// ---------------------------------------------------------------------------
//  MO ERI sub-block extractors
// ---------------------------------------------------------------------------
//  Chemist's notation (pq|rs). Occupied indices i,j,k,l ∈ [0, nocc),
//  virtuals a,b,c,d offset by nocc when reading full MO ERI.
// ---------------------------------------------------------------------------

// ovov[i,a,j,b] = (i, nocc+a | j, nocc+b)
std::vector<real_t> extract_ovov(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.nv * d.no * d.nv);
    for (int i = 0; i < d.no; i++)
     for (int a = 0; a < d.nv; a++)
      for (int j = 0; j < d.no; j++)
       for (int b = 0; b < d.nv; b++)
        out[(((size_t)i * d.nv + a) * d.no + j) * d.nv + b] =
            eri[idx_eri(d.na, i, d.no + a, j, d.no + b)];
    return out;
}

std::vector<real_t> extract_oovv(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.no * d.nv * d.nv);
    for (int i = 0; i < d.no; i++)
     for (int j = 0; j < d.no; j++)
      for (int a = 0; a < d.nv; a++)
       for (int b = 0; b < d.nv; b++)
        out[(((size_t)i * d.no + j) * d.nv + a) * d.nv + b] =
            eri[idx_eri(d.na, i, j, d.no + a, d.no + b)];
    return out;
}

std::vector<real_t> extract_ovvo(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.nv * d.nv * d.no);
    for (int i = 0; i < d.no; i++)
     for (int a = 0; a < d.nv; a++)
      for (int b = 0; b < d.nv; b++)
       for (int j = 0; j < d.no; j++)
        out[(((size_t)i * d.nv + a) * d.nv + b) * d.no + j] =
            eri[idx_eri(d.na, i, d.no + a, d.no + b, j)];
    return out;
}

std::vector<real_t> extract_oooo(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.no * d.no * d.no);
    for (int i = 0; i < d.no; i++)
     for (int j = 0; j < d.no; j++)
      for (int k = 0; k < d.no; k++)
       for (int l = 0; l < d.no; l++)
        out[(((size_t)i * d.no + j) * d.no + k) * d.no + l] =
            eri[idx_eri(d.na, i, j, k, l)];
    return out;
}

// ovoo[i,a,j,k] = (i, nocc+a | j, k)  — chemist convention
std::vector<real_t> extract_ovoo(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.nv * d.no * d.no);
    for (int i = 0; i < d.no; i++)
     for (int a = 0; a < d.nv; a++)
      for (int j = 0; j < d.no; j++)
       for (int k = 0; k < d.no; k++)
        out[(((size_t)i * d.nv + a) * d.no + j) * d.no + k] =
            eri[idx_eri(d.na, i, d.no + a, j, k)];
    return out;
}

// ovvv[i,a,b,c] = (i, nocc+a | nocc+b, nocc+c)
std::vector<real_t> extract_ovvv(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.no * d.nv * d.nv * d.nv);
    for (int i = 0; i < d.no; i++)
     for (int a = 0; a < d.nv; a++)
      for (int b = 0; b < d.nv; b++)
       for (int c = 0; c < d.nv; c++)
        out[(((size_t)i * d.nv + a) * d.nv + b) * d.nv + c] =
            eri[idx_eri(d.na, i, d.no + a, d.no + b, d.no + c)];
    return out;
}

std::vector<real_t> extract_vvvv(const LambdaDims& d, const real_t* eri) {
    std::vector<real_t> out((size_t)d.nv * d.nv * d.nv * d.nv);
    for (int a = 0; a < d.nv; a++)
     for (int b = 0; b < d.nv; b++)
      for (int c = 0; c < d.nv; c++)
       for (int dd = 0; dd < d.nv; dd++)
        out[(((size_t)a * d.nv + b) * d.nv + c) * d.nv + dd] =
            eri[idx_eri(d.na, d.no + a, d.no + b, d.no + c, d.no + dd)];
    return out;
}

// ---------------------------------------------------------------------------
//  Helpers: tau = t2 + t1⊗t1 ; theta = 2*t2 - t2.T(0,1,3,2)
// ---------------------------------------------------------------------------

void make_tau(const LambdaDims& d, const real_t* t1, const real_t* t2,
              std::vector<real_t>& tau)
{
    tau.assign((size_t)d.no * d.no * d.nv * d.nv, 0.0);
    for (int i = 0; i < d.no; i++)
      for (int j = 0; j < d.no; j++)
        for (int a = 0; a < d.nv; a++)
          for (int b = 0; b < d.nv; b++)
            tau[idx_t2(d,i,j,a,b)] = t2[idx_t2(d,i,j,a,b)]
                                   + t1[idx_t1(d,i,a)] * t1[idx_t1(d,j,b)];
}

void make_theta(const LambdaDims& d, const real_t* t2,
                std::vector<real_t>& theta)
{
    theta.assign((size_t)d.no * d.no * d.nv * d.nv, 0.0);
    for (int i = 0; i < d.no; i++)
      for (int j = 0; j < d.no; j++)
        for (int a = 0; a < d.nv; a++)
          for (int b = 0; b < d.nv; b++)
            theta[idx_t2(d,i,j,a,b)] = 2.0 * t2[idx_t2(d,i,j,a,b)]
                                     -        t2[idx_t2(d,i,j,b,a)];
}

// ---------------------------------------------------------------------------
//  Lambda equations — direct translation of the reference Lambda equations
//  (make_intermediates + update_lambda) for canonical RHF:
//    foo[i,j] = δ_ij ε_i,  fvv[a,b] = δ_ab ε_{nocc+a},  fov = fvo = 0.
//  All MO ERI in chemist's notation.
// ---------------------------------------------------------------------------

void update_lambda_full(const LambdaDims& d,
                        const real_t* t1, const real_t* t2,
                        const real_t* l1, const real_t* l2,
                        const real_t* eps,
                        const std::vector<real_t>& ovov,
                        const std::vector<real_t>& ovoo,
                        const std::vector<real_t>& ovvv,
                        const std::vector<real_t>& oovv,
                        const std::vector<real_t>& ovvo,
                        const std::vector<real_t>& oooo,
                        const std::vector<real_t>& vvvv,
                        real_t* l1_new, real_t* l2_new,
                        const real_t* fov_active = nullptr)
{
    const int no = d.no, nv = d.nv;
    const size_t l1_sz = (size_t)no * nv;
    const size_t l2_sz = (size_t)no * no * nv * nv;

    // ----- Index lambdas (chemist notation, row-major) -----
    auto T1 = [&](int i, int a) { return t1[i*nv + a]; };
    auto T2 = [&](int i, int j, int a, int b) { return t2[idx_t2(d,i,j,a,b)]; };
    auto L1 = [&](int i, int a) { return l1[i*nv + a]; };
    auto L2 = [&](int i, int j, int a, int b) { return l2[idx_t2(d,i,j,a,b)]; };
    auto OVOV = [&](int i, int a, int j, int b) {
        return ovov[(((size_t)i * nv + a) * no + j) * nv + b]; };
    auto OVOO = [&](int i, int a, int j, int k) {
        return ovoo[(((size_t)i * nv + a) * no + j) * no + k]; };
    auto OVVV = [&](int i, int a, int b, int c) {
        return ovvv[(((size_t)i * nv + a) * nv + b) * nv + c]; };
    auto OOVV = [&](int i, int j, int a, int b) {
        return oovv[(((size_t)i * no + j) * nv + a) * nv + b]; };
    auto OVVO = [&](int i, int a, int b, int j) {
        return ovvo[(((size_t)i * nv + a) * nv + b) * no + j]; };
    auto OOOO = [&](int i, int j, int k, int l) {
        return oooo[(((size_t)i * no + j) * no + k) * no + l]; };
    auto VVVV = [&](int a, int b, int c, int dd) {
        return vvvv[(((size_t)a * nv + b) * nv + c) * nv + dd]; };

    // tau, theta
    std::vector<real_t> tau, theta;
    make_tau(d, t1, t2, tau);
    make_theta(d, t2, theta);
    auto TAU = [&](int i, int j, int a, int b) { return tau[idx_t2(d,i,j,a,b)]; };
    auto TH  = [&](int i, int j, int a, int b) { return theta[idx_t2(d,i,j,a,b)]; };

    // ovov1[j,a,k,c] = 2*ovov[j,a,k,c] - ovov[j,c,k,a]
    auto OVOV1 = [&](int j, int a, int k, int c) {
        return 2.0 * OVOV(j,a,k,c) - OVOV(j,c,k,a); };
    // ovoo1[k,b,i,j] = 2*ovoo[k,b,i,j] - ovoo[i,b,k,j]   (reference: ovoo*2 - ovoo.T(2,1,0,3))
    auto OVOO1 = [&](int k, int b, int i, int j) {
        return 2.0 * OVOO(k,b,i,j) - OVOO(i,b,k,j); };
    // ovvv1[i,a,b,c] = 2*ovvv[i,a,b,c] - ovvv[i,c,b,a]
    auto OVVV1 = [&](int i, int a, int b, int c) {
        return 2.0 * OVVV(i,a,b,c) - OVVV(i,c,b,a); };

    // =================================================================
    //  make_intermediates
    // =================================================================
    std::vector<real_t> v1((size_t)nv*nv, 0.0);   // [a,b] indexed v1[b,a] → store as v1[b*nv+a]? reference: einsum('jakc,jkbc->ba')
                                                  //  We'll store v1[b,a] = v1[(size_t)b*nv+a].
    std::vector<real_t> v2((size_t)no*no, 0.0);   // [i,j]
    std::vector<real_t> v4((size_t)no*nv, 0.0);   // [j,b]
    std::vector<real_t> v5((size_t)nv*no, 0.0);   // [b,j]
    std::vector<real_t> w3((size_t)nv*no, 0.0);   // [c,k]

    // v1[b,a] = fvv[b,a] - Σ_{j,k,c} ovov1[j,a,k,c]*tau[j,k,b,c]
    //   For canonical: fvv[b,a] = δ_ba * ε_{nocc+a}
    //   Semi-canonical: v1 -= Σ_j fov[j,a]*t1[j,b]
    for (int b = 0; b < nv; b++)
      for (int a = 0; a < nv; a++) {
        real_t v = (a == b ? eps[no + a] : 0.0);
        for (int j = 0; j < no; j++)
          for (int k = 0; k < no; k++)
            for (int c = 0; c < nv; c++)
              v -= OVOV1(j,a,k,c) * TAU(j,k,b,c);
        if (fov_active) {
            for (int j = 0; j < no; j++)
                v -= fov_active[(size_t)j*nv + a] * T1(j, b);
        }
        v1[(size_t)b*nv + a] = v;
      }

    // v2[i,j] = foo[i,j] + Σ_{b,k,c} ovov1[i,b,k,c]*tau[j,k,b,c] + Σ_{k,b} ovoo1[k,b,i,j]*t1[k,b]
    //   Semi-canonical: v2 += Σ_b fov[i,b]*t1[j,b]
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        real_t v = (i == j ? eps[i] : 0.0);
        for (int b = 0; b < nv; b++)
          for (int k = 0; k < no; k++)
            for (int c = 0; c < nv; c++)
              v += OVOV1(i,b,k,c) * TAU(j,k,b,c);
        for (int k = 0; k < no; k++)
          for (int b = 0; b < nv; b++)
            v += OVOO1(k,b,i,j) * T1(k,b);
        if (fov_active) {
            for (int b = 0; b < nv; b++)
                v += fov_active[(size_t)i*nv + b] * T1(j, b);
        }
        v2[(size_t)i*no + j] = v;
      }

    // v4[j,b] = fov + Σ_{k,c} ovov1[j,b,k,c]*t1[k,c]   (fov=0 for canonical)
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++) {
        real_t v = (fov_active ? fov_active[(size_t)j*nv + b] : 0.0);
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            v += OVOV1(j,b,k,c) * T1(k,c);
        v4[(size_t)j*nv + b] = v;
      }
    auto V4 = [&](int j, int b) { return v4[(size_t)j*nv + b]; };

    // v5[b,j] = +fvo (= fov^T for real, = 0 for canonical)
    //   + 2 Σ_{k,c} fov[k,c]*t2[j,k,b,c] - Σ_{k,c} fov[k,c]*t2[j,k,c,b]   (semi-canon)
    //   + Σ_{k,c} v4[k,c]*t1[k,b]*t1[j,c]
    //   - Σ_{l,c,k} ovoo1[l,c,k,j]*t2[k,l,b,c]
    for (int b = 0; b < nv; b++)
      for (int j = 0; j < no; j++) {
        real_t v = (fov_active ? fov_active[(size_t)j*nv + b] : 0.0);  // fvo = fov^T
        if (fov_active) {
            for (int k = 0; k < no; k++)
                for (int c = 0; c < nv; c++) {
                    v += 2.0 * fov_active[(size_t)k*nv + c] * T2(j, k, b, c);
                    v -=        fov_active[(size_t)k*nv + c] * T2(j, k, c, b);
                }
        }
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            v += V4(k,c) * T1(k,b) * T1(j,c);
        for (int l = 0; l < no; l++)
          for (int c = 0; c < nv; c++)
            for (int k = 0; k < no; k++)
              v -= OVOO1(l,c,k,j) * T2(k,l,b,c);
        v5[(size_t)b*no + j] = v;
      }

    // woooo[i,k,j,l]:  einsum order:
    //   woooo  = einsum('icjl,kc->ikjl', ovoo, t1)   → woooo[i,k,j,l] += Σ_c ovoo[i,c,j,l]*t1[k,c]
    //   woooo += einsum('jcil,kc->iljk', ovoo, t1)   → woooo[i,l,j,k] += Σ_c ovoo[j,c,i,l]*t1[k,c]
    //   woooo += oooo                                → woooo[i,k,j,l] += oooo[i,k,j,l]
    //   woooo += einsum('icjd,klcd->ikjl', ovov, tau)→ woooo[i,k,j,l] += Σ_{c,d} ovov[i,c,j,d]*tau[k,l,c,d]
    std::vector<real_t> woooo((size_t)no*no*no*no, 0.0);
    auto WOOOO = [&](int i, int k, int j, int l) -> real_t& {
        return woooo[(((size_t)i * no + k) * no + j) * no + l]; };
    for (int i = 0; i < no; i++)
      for (int k = 0; k < no; k++)
        for (int j = 0; j < no; j++)
          for (int l = 0; l < no; l++) {
            real_t v = OOOO(i,k,j,l);
            // einsum('icjl,kc->ikjl', ovoo, t1)
            for (int c = 0; c < nv; c++)
              v += OVOO(i,c,j,l) * T1(k,c);
            // einsum('jcil,kc->iljk', ovoo, t1) → reindex to (i,k,j,l): += OVOO(j,c,i,k)*T1(l,c)
            for (int c = 0; c < nv; c++)
              v += OVOO(j,c,i,k) * T1(l,c);
            // einsum('icjd,klcd->ikjl', ovov, tau)
            for (int c = 0; c < nv; c++)
              for (int dd = 0; dd < nv; dd++)
                v += OVOV(i,c,j,dd) * TAU(k,l,c,dd);
            WOOOO(i,k,j,l) = v;
          }

    // v4OVvo[j,b,c,k] = Σ_{l,d} ovov1[l,d,j,b]*t2[k,l,c,d] - Σ_{l,d} ovov[l,d,j,b]*t2[k,l,d,c] + ovvo[j,b,c,k]
    std::vector<real_t> v4OVvo((size_t)no*nv*nv*no, 0.0);
    std::vector<real_t> v4oVVo((size_t)no*nv*nv*no, 0.0);
    auto V4OVvo = [&](int j, int b, int c, int k) -> real_t& {
        return v4OVvo[(((size_t)j * nv + b) * nv + c) * no + k]; };
    auto V4oVVo = [&](int j, int b, int c, int k) -> real_t& {
        return v4oVVo[(((size_t)j * nv + b) * nv + c) * no + k]; };
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int k = 0; k < no; k++) {
            real_t v = OVVO(j,b,c,k);
            for (int l = 0; l < no; l++)
              for (int dd = 0; dd < nv; dd++) {
                v += OVOV1(l,dd,j,b) * T2(k,l,c,dd);
                v -= OVOV(l,dd,j,b) * T2(k,l,dd,c);
              }
            V4OVvo(j,b,c,k) = v;
          }
    // v4oVVo[j,b,c,k] = Σ_{l,d} ovov[j,d,l,b]*t2[k,l,d,c] - oovv[j,c,b,k] (reference: oovv.T(0,3,2,1))
    //   reference: v4oVVo -= eris.oovv.transpose(0,3,2,1) → v4oVVo[j,b,c,k] -= oovv[j,k,c,b]
    //   (Since transpose order (0,3,2,1) maps oovv[a,b,c,d]→[a,d,c,b])
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int k = 0; k < no; k++) {
            real_t v = 0.0;
            for (int l = 0; l < no; l++)
              for (int dd = 0; dd < nv; dd++)
                v += OVOV(j,dd,l,b) * T2(k,l,dd,c);
            v -= OOVV(j,k,c,b);  // oovv[j,k,c,b]
            V4oVVo(j,b,c,k) = v;
          }

    // v4ovvo = v4OVvo*2 + v4oVVo
    // w3[c,k] = Σ_{j,b} v4ovvo[j,b,c,k]*t1[j,b]
    for (int c = 0; c < nv; c++)
      for (int k = 0; k < no; k++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++)
            v += (2.0 * V4OVvo(j,b,c,k) + V4oVVo(j,b,c,k)) * T1(j,b);
        w3[(size_t)c*no + k] = v;
      }

    // wOVvo and woVVo final (after all corrections):
    //   wOVvo = v4OVvo - Σ_{l,d} ovov[j,b,l,d]*t1[k,d]*t1[l,c]
    //                  - Σ_l ovoo[j,b,l,k]*t1[l,c]
    //                  + Σ_d ovvv[j,b,c,d]*t1[k,d]
    //   woVVo = v4oVVo + Σ_{l,d} ovov[j,d,l,b]*t1[k,d]*t1[l,c]
    //                  + Σ_l ovoo[l,b,j,k]*t1[l,c]
    //                  - Σ_d ovvv[j,d,c,b]*t1[k,d]
    std::vector<real_t> wOVvo((size_t)no*nv*nv*no);
    std::vector<real_t> woVVo((size_t)no*nv*nv*no);
    auto WOVvo = [&](int j, int b, int c, int k) -> real_t& {
        return wOVvo[(((size_t)j * nv + b) * nv + c) * no + k]; };
    auto WoVVo = [&](int j, int b, int c, int k) -> real_t& {
        return woVVo[(((size_t)j * nv + b) * nv + c) * no + k]; };
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int k = 0; k < no; k++) {
            real_t vO = V4OVvo(j,b,c,k);
            real_t vV = V4oVVo(j,b,c,k);
            for (int l = 0; l < no; l++)
              for (int dd = 0; dd < nv; dd++) {
                vO -= OVOV(j,b,l,dd) * T1(k,dd) * T1(l,c);
                vV += OVOV(j,dd,l,b) * T1(k,dd) * T1(l,c);
              }
            for (int l = 0; l < no; l++) {
              vO -= OVOO(j,b,l,k) * T1(l,c);
              vV += OVOO(l,b,j,k) * T1(l,c);
            }
            for (int dd = 0; dd < nv; dd++) {
              vO += OVVV(j,b,c,dd) * T1(k,dd);
              vV -= OVVV(j,dd,c,b) * T1(k,dd);
            }
            WOVvo(j,b,c,k) = vO;
            WoVVo(j,b,c,k) = vV;
          }

    // wovvo (final, used by update_lambda) = wOVvo*2 + woVVo
    std::vector<real_t> wovvo((size_t)no*nv*nv*no);
    auto WOVVO = [&](int j, int b, int c, int k) -> real_t& {
        return wovvo[(((size_t)j * nv + b) * nv + c) * no + k]; };
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int k = 0; k < no; k++)
            WOVVO(j,b,c,k) = 2.0 * WOVvo(j,b,c,k) + WoVVo(j,b,c,k);

    // woovo and wvvvo: build using v4OVvo, v4oVVo, OVOO, OVVV (long expression)
    std::vector<real_t> woovo((size_t)no*no*nv*no, 0.0);
    auto WOOVO = [&](int i, int j, int c, int k) -> real_t& {
        return woovo[(((size_t)i * no + j) * nv + c) * no + k]; };
    // From the reference:
    //   woovo  = einsum('ibck,jb->ijck', v4ovvo, t1)               # v4ovvo = 2*v4OVvo + v4oVVo
    //   woovo  = woovo - woovo.T(0,3,2,1)                          # woovo[i,j,c,k] -= woovo[i,k,c,j]
    //   woovo += einsum('ibck,jb->ikcj', v4OVvo - v4oVVo, t1)      # adds woovo[i,k,c,j] += sum_b (v4O-v4o)[i,b,c,k]*t1[j,b]
    //   woovo += ovoo1.T(3,2,1,0)                                  # woovo[i,j,c,k] += ovoo1[k,c,j,i]
    //   woovo += einsum('lcik,jlbc->ikbj', ovoo1, theta)           # woovo[i,k,b,j] += sum_{l,c} ovoo1[l,c,i,k]*theta[j,l,b,c]
    //   woovo -= einsum('lcik,jlbc->ijbk', ovoo1, t2)              # woovo[i,j,b,k] -= sum_{l,c} ovoo1[l,c,i,k]*t2[j,l,b,c]
    //   woovo -= einsum('iclk,ljbc->ijbk', ovoo1, t2)              # woovo[i,j,b,k] -= sum_{l,c} ovoo1[i,c,l,k]*t2[l,j,b,c]
    //   woovo += einsum('idcb,jkdb->ijck', ovvv, tau)              # woovo[i,j,c,k] += sum_{d,b} ovvv[i,d,c,b]*tau[j,k,d,b]
    {
        std::vector<real_t> v4ovvo((size_t)no*nv*nv*no);
        for (size_t k = 0; k < v4ovvo.size(); k++)
            v4ovvo[k] = 2.0 * v4OVvo[k] + v4oVVo[k];
        auto V4 = [&](int j, int b, int c, int k) {
            return v4ovvo[(((size_t)j * nv + b) * nv + c) * no + k]; };
        auto V4dif = [&](int j, int b, int c, int k) {
            return v4OVvo[(((size_t)j * nv + b) * nv + c) * no + k]
                 - v4oVVo[(((size_t)j * nv + b) * nv + c) * no + k]; };

        // step 1: woovo[i,j,c,k] = sum_b v4[i,b,c,k]*t1[j,b]
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int b = 0; b < nv; b++) v += V4(i,b,c,k) * T1(j,b);
                WOOVO(i,j,c,k) = v;
              }
        // step 2: woovo -= woovo.T(0,3,2,1)  → woovo[i,j,c,k] -= woovo[i,k,c,j]
        std::vector<real_t> tmp = woovo;
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++)
                WOOVO(i,j,c,k) -= tmp[(((size_t)i * no + k) * nv + c) * no + j];
        // step 3: += einsum('ibck,jb->ikcj', v4dif, t1) — destination index 'ikcj' = woovo[i,k,c,j]
        for (int i = 0; i < no; i++)
          for (int k = 0; k < no; k++)
            for (int c = 0; c < nv; c++)
              for (int j = 0; j < no; j++) {
                real_t v = 0.0;
                for (int b = 0; b < nv; b++) v += V4dif(i,b,c,k) * T1(j,b);
                WOOVO(i,k,c,j) += v;
              }
        // step 4: += ovoo1.T(3,2,1,0) → woovo[i,j,c,k] += ovoo1[k,c,j,i]
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++)
                WOOVO(i,j,c,k) += OVOO1(k,c,j,i);
        // step 5: += einsum('lcik,jlbc->ikbj', ovoo1, theta)
        for (int i = 0; i < no; i++)
          for (int k = 0; k < no; k++)
            for (int b = 0; b < nv; b++)
              for (int j = 0; j < no; j++) {
                real_t v = 0.0;
                for (int l = 0; l < no; l++)
                  for (int c = 0; c < nv; c++)
                    v += OVOO1(l,c,i,k) * TH(j,l,b,c);
                WOOVO(i,k,b,j) += v;
              }
        // step 6: -= einsum('lcik,jlbc->ijbk', ovoo1, t2)
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int b = 0; b < nv; b++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int l = 0; l < no; l++)
                  for (int c = 0; c < nv; c++)
                    v += OVOO1(l,c,i,k) * T2(j,l,b,c);
                WOOVO(i,j,b,k) -= v;
              }
        // step 7: -= einsum('iclk,ljbc->ijbk', ovoo1, t2)
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int b = 0; b < nv; b++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int l = 0; l < no; l++)
                  for (int c = 0; c < nv; c++)
                    v += OVOO1(i,c,l,k) * T2(l,j,b,c);
                WOOVO(i,j,b,k) -= v;
              }
        // step 8: += einsum('idcb,jkdb->ijck', ovvv1, tau)
        //   ovvv1[i,d,c,b] = 2*ovvv[i,d,c,b] - ovvv[i,b,c,d]   (reference reassigns ovvv→ovvv1 before this step)
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int dd = 0; dd < nv; dd++)
                  for (int b = 0; b < nv; b++) {
                    real_t ovvv1 = 2.0 * OVVV(i,dd,c,b) - OVVV(i,b,c,dd);
                    v += ovvv1 * TAU(j,k,dd,b);
                  }
                WOOVO(i,j,c,k) += v;
              }
    }

    // wvvvo[b,a,c,k]: similar long expression
    std::vector<real_t> wvvvo((size_t)nv*nv*nv*no, 0.0);
    auto WVVVO = [&](int b, int a, int c, int k) -> real_t& {
        return wvvvo[(((size_t)b * nv + a) * nv + c) * no + k]; };
    {
        // reference:
        //   wvvvo  = einsum('jack,jb->back', v4ovvo, t1)   → wvvvo[b,a,c,k] = sum_j v4[j,a,c,k]*t1[j,b]
        //   wvvvo  = wvvvo - wvvvo.T(2,1,0,3)              → wvvvo[b,a,c,k] -= wvvvo[c,a,b,k]
        //   wvvvo += einsum('jack,jb->cabk', v4OVvo - v4oVVo, t1)  → wvvvo[c,a,b,k] += sum_j (v4O-v4o)[j,a,c,k]*t1[j,b]
        //   wvvvo -= einsum('lajk,jlbc->cabk', ovoo1, tau)         → wvvvo[c,a,b,k] -= sum_{l,j} ovoo1[l,a,j,k]*tau[j,l,b,c]
        //   wvvvo += einsum('kacd,kjbd->bacj', ovvv, t2) * 1.5     → wvvvo[b,a,c,j] += 1.5 * sum_{k,d} ovvv[k,a,c,d]*t2[k,j,b,d]
        //   wvvvo -= einsum('kdca,jkbd->cabj', ovvv1, theta)
        //   wvvvo += einsum('kdca,jkbd->bacj', ovvv1, theta) * 0.5  (after .T(2,1,0,3))
        //   wvvvo -= ovvv1.T(3,2,1,0)                              → wvvvo[b,a,c,k] -= ovvv1[k,c,a,b]
        std::vector<real_t> v4ovvo((size_t)no*nv*nv*no);
        for (size_t k = 0; k < v4ovvo.size(); k++)
            v4ovvo[k] = 2.0 * v4OVvo[k] + v4oVVo[k];
        auto V4 = [&](int j, int b, int c, int k) {
            return v4ovvo[(((size_t)j * nv + b) * nv + c) * no + k]; };
        auto V4dif = [&](int j, int b, int c, int k) {
            return v4OVvo[(((size_t)j * nv + b) * nv + c) * no + k]
                 - v4oVVo[(((size_t)j * nv + b) * nv + c) * no + k]; };

        // step 1: wvvvo[b,a,c,k] = sum_j v4[j,a,c,k]*t1[j,b]
        for (int b = 0; b < nv; b++)
          for (int a = 0; a < nv; a++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int j = 0; j < no; j++) v += V4(j,a,c,k) * T1(j,b);
                WVVVO(b,a,c,k) = v;
              }
        // step 2: wvvvo[b,a,c,k] -= wvvvo[c,a,b,k]
        std::vector<real_t> tmp = wvvvo;
        for (int b = 0; b < nv; b++)
          for (int a = 0; a < nv; a++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++)
                WVVVO(b,a,c,k) -= tmp[(((size_t)c * nv + a) * nv + b) * no + k];
        // step 3: wvvvo[c,a,b,k] += sum_j v4dif[j,a,c,k]*t1[j,b]
        for (int c = 0; c < nv; c++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int j = 0; j < no; j++) v += V4dif(j,a,c,k) * T1(j,b);
                WVVVO(c,a,b,k) += v;
              }
        // step 4: wvvvo[c,a,b,k] -= sum_{l,j} ovoo1[l,a,j,k]*tau[j,l,b,c]
        for (int c = 0; c < nv; c++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int l = 0; l < no; l++)
                  for (int j = 0; j < no; j++)
                    v += OVOO1(l,a,j,k) * TAU(j,l,b,c);
                WVVVO(c,a,b,k) -= v;
              }
        // step 5: wvvvo[b,a,c,j] += 1.5 * sum_{k,d} ovvv[k,a,c,d]*t2[k,j,b,d]
        for (int b = 0; b < nv; b++)
          for (int a = 0; a < nv; a++)
            for (int c = 0; c < nv; c++)
              for (int j = 0; j < no; j++) {
                real_t v = 0.0;
                for (int k = 0; k < no; k++)
                  for (int dd = 0; dd < nv; dd++)
                    v += OVVV(k,a,c,dd) * T2(k,j,b,dd);
                WVVVO(b,a,c,j) += 1.5 * v;
              }
        // step 6: tmp_cabj = sum_{k,d} ovvv1[k,d,c,a]*theta[j,k,b,d]
        std::vector<real_t> tmpcabj((size_t)nv*nv*nv*no, 0.0);
        for (int c = 0; c < nv; c++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              for (int j = 0; j < no; j++) {
                real_t v = 0.0;
                for (int k = 0; k < no; k++)
                  for (int dd = 0; dd < nv; dd++)
                    v += OVVV1(k,dd,c,a) * TH(j,k,b,dd);
                tmpcabj[(((size_t)c * nv + a) * nv + b) * no + j] = v;
              }
        // wvvvo -= tmp; wvvvo += tmp.T(2,1,0,3)*0.5
        for (int c = 0; c < nv; c++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              for (int j = 0; j < no; j++) {
                size_t cabj = (((size_t)c * nv + a) * nv + b) * no + j;
                size_t bacj = (((size_t)b * nv + a) * nv + c) * no + j;
                wvvvo[cabj] -= tmpcabj[cabj];
                wvvvo[bacj] += 0.5 * tmpcabj[cabj];
              }
        // step 7: wvvvo[b,a,c,k] -= ovvv1[k,c,a,b]
        for (int b = 0; b < nv; b++)
          for (int a = 0; a < nv; a++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++)
                WVVVO(b,a,c,k) -= OVVV1(k,c,a,b);
    }

    // Add ovvv1 contribution to v1: v1[b,a] += sum_j ovvv1[j,c,b,a]*t1[j,c]
    // (reference: v1 += einsum('jcba,jc->ba', ovvv1, t1))
    for (int b = 0; b < nv; b++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int c = 0; c < nv; c++)
            v += OVVV1(j,c,b,a) * T1(j,c);
        v1[(size_t)b*nv + a] += v;
      }
    // v5[b,j] += sum_{k,c,d} ovvv1[k,d,b,c]*t2[j,k,c,d]
    for (int b = 0; b < nv; b++)
      for (int j = 0; j < no; j++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            for (int dd = 0; dd < nv; dd++)
              v += OVVV1(k,dd,b,c) * T2(j,k,c,dd);
        v5[(size_t)b*no + j] += v;
      }

    // w3 += v5; w3 += v1*t1; w3 -= v2*t1
    //   w3[c,k] += v5[c,k]; w3[c,k] += sum_b v1[c,b]*t1[k,b]; w3[c,k] -= sum_j v2[j,k]*t1[j,c]
    for (int c = 0; c < nv; c++)
      for (int k = 0; k < no; k++) {
        real_t v = v5[(size_t)c*no + k];
        for (int b = 0; b < nv; b++) v += v1[(size_t)c*nv + b] * T1(k,b);
        for (int j = 0; j < no; j++) v -= v2[(size_t)j*no + k] * T1(j,c);
        w3[(size_t)c*no + k] += v;
      }

    // =================================================================
    //  update_lambda
    // =================================================================
    // mvv[a,b] = Σ_{k,l,c} l2[k,l,c,b]*theta[k,l,c,a]   (einsum 'klca,klcb->ba')
    std::vector<real_t> mvv((size_t)nv*nv, 0.0);
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int l = 0; l < no; l++)
            for (int c = 0; c < nv; c++)
              v += L2(k,l,c,b) * TH(k,l,c,a);
        mvv[(size_t)a*nv + b] = v;
      }
    // moo[i,j] = Σ_{k,c,d} l2[k,i,c,d]*theta[k,j,c,d]
    std::vector<real_t> moo((size_t)no*no, 0.0);
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            for (int dd = 0; dd < nv; dd++)
              v += L2(k,i,c,dd) * TH(k,j,c,dd);
        moo[(size_t)i*no + j] = v;
      }
    // mvv1 = einsum('jc,jb->bc', l1, t1) + mvv  → mvv1[b,c] = sum_j l1[j,c]*t1[j,b] + mvv[b,c]
    std::vector<real_t> mvv1 = mvv;
    for (int b = 0; b < nv; b++)
      for (int c = 0; c < nv; c++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++) v += L1(j,c) * T1(j,b);
        mvv1[(size_t)b*nv + c] += v;
      }
    // moo1 = einsum('ic,kc->ik', l1, t1) + moo  → moo1[i,k] = sum_c l1[i,c]*t1[k,c] + moo[i,k]
    std::vector<real_t> moo1 = moo;
    for (int i = 0; i < no; i++)
      for (int k = 0; k < no; k++) {
        real_t v = 0.0;
        for (int c = 0; c < nv; c++) v += L1(i,c) * T1(k,c);
        moo1[(size_t)i*no + k] += v;
      }

    // m3[i,j,a,b] = 0.5 * [ Σ_{c,d} l2[i,j,c,d]*vvvv[a,c,b,d] + Σ_{k,l} l2[k,l,a,b]*woooo[i,k,j,l] ]
    //               + 0.5 * Σ_{k,l} ovov[k,a,l,b] * (Σ_{c,d} l2[i,j,c,d]*tau[k,l,c,d])
    std::vector<real_t> m3(l2_sz, 0.0);
    auto M3 = [&](int i, int j, int a, int b) -> real_t& { return m3[idx_t2(d,i,j,a,b)]; };
    // first term: l2.vvvv  (note reference uses .conj() but real → no-op)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int c = 0; c < nv; c++)
              for (int dd = 0; dd < nv; dd++)
                v += L2(i,j,c,dd) * VVVV(a,c,b,dd);
            M3(i,j,a,b) = v;
          }
    // + l2*woooo
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              for (int l = 0; l < no; l++)
                v += L2(k,l,a,b) * WOOOO(i,k,j,l);
            M3(i,j,a,b) += v;
          }
    // *0.5
    for (size_t k = 0; k < l2_sz; k++) m3[k] *= 0.5;
    // + 0.5 * ovov * (Σ l2*tau)
    {
        std::vector<real_t> l2tau((size_t)no*no*no*no, 0.0);
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int k = 0; k < no; k++)
              for (int l = 0; l < no; l++) {
                real_t v = 0.0;
                for (int c = 0; c < nv; c++)
                  for (int dd = 0; dd < nv; dd++)
                    v += L2(i,j,c,dd) * TAU(k,l,c,dd);
                l2tau[(((size_t)i * no + j) * no + k) * no + l] = v;
              }
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++) {
                real_t v = 0.0;
                for (int k = 0; k < no; k++)
                  for (int l = 0; l < no; l++)
                    v += OVOV(k,a,l,b) * l2tau[(((size_t)i * no + j) * no + k) * no + l];
                M3(i,j,a,b) += 0.5 * v;
              }
    }

    // ----- l2new = 0.5 * ovov.T(0,2,1,3) -----
    //   ovov[i,a,j,b].T(0,2,1,3) = ovov[i,j,a,b]   (move axis 1↔2)
    std::fill(l2_new, l2_new + l2_sz, 0.0);
    auto L2N = [&](int i, int j, int a, int b) -> real_t& { return l2_new[idx_t2(d,i,j,a,b)]; };
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++)
            L2N(i,j,a,b) = 0.5 * OVOV(i,a,j,b);

    // l2new += einsum('ijac,cb->ijab', l2, v1)   v1 stored [b,a]; reference v1[c,b]; here v1[b,a] indexed v1[b*nv+a]
    //   So 'cb' here means v1[c,b] = our v1[(c,b)] = v1[c*nv + b].
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int c = 0; c < nv; c++)
              v += L2(i,j,a,c) * v1[(size_t)c*nv + b];
            L2N(i,j,a,b) += v;
          }
    // l2new -= einsum('ikab,jk->ijab', l2, v2)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              v += L2(i,k,a,b) * v2[(size_t)j*no + k];
            L2N(i,j,a,b) -= v;
          }
    // l2new -= einsum('ca,icjb->ijab', mvv1, ovov)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int c = 0; c < nv; c++)
              v += mvv1[(size_t)c*nv + a] * OVOV(i,c,j,b);
            L2N(i,j,a,b) -= v;
          }
    // l2new -= einsum('ik,kajb->ijab', moo1, ovov)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              v += moo1[(size_t)i*no + k] * OVOV(k,a,j,b);
            L2N(i,j,a,b) -= v;
          }

    // ----- l1new initialization and contributions -----
    std::fill(l1_new, l1_new + l1_sz, 0.0);
    auto L1N = [&](int i, int a) -> real_t& { return l1_new[i*nv + a]; };

    // l1new -= einsum('ik,ka->ia', moo, v4)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++) v += moo[(size_t)i*no + k] * V4(k,a);
        L1N(i,a) -= v;
      }
    // l1new -= einsum('ca,ic->ia', mvv, v4)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int c = 0; c < nv; c++) v += mvv[(size_t)c*nv + a] * V4(i,c);
        L1N(i,a) -= v;
      }
    // l2new += einsum('ia,jb->ijab', l1, v4)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++)
            L2N(i,j,a,b) += L1(i,a) * V4(j,b);

    // tmp[j,b] = t1[j,b] + Σ_{k,c} l1[k,c]*theta[k,j,c,b] - Σ_d mvv1[b,d]*t1[j,d] - Σ_l moo[l,j]*t1[l,b]
    // l1new += einsum('jbia,jb->ia', ovov1, tmp)
    {
        std::vector<real_t> tmp((size_t)no*nv, 0.0);
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            real_t v = T1(j,b);
            for (int k = 0; k < no; k++)
              for (int c = 0; c < nv; c++)
                v += L1(k,c) * TH(k,j,c,b);
            for (int dd = 0; dd < nv; dd++)
              v -= mvv1[(size_t)b*nv + dd] * T1(j,dd);
            for (int l = 0; l < no; l++)
              v -= moo[(size_t)l*no + j] * T1(l,b);
            tmp[(size_t)j*nv + b] = v;
          }
        for (int i = 0; i < no; i++)
          for (int a = 0; a < nv; a++) {
            real_t v = 0.0;
            for (int j = 0; j < no; j++)
              for (int b = 0; b < nv; b++)
                v += OVOV1(j,b,i,a) * tmp[(size_t)j*nv + b];
            L1N(i,a) += v;
          }
    }

    // l1new += 2 * einsum('iacb,bc->ia', ovvv, mvv1) - einsum('ibca,bc->ia', ovvv, mvv1)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int b = 0; b < nv; b++)
          for (int c = 0; c < nv; c++)
            v += 2.0 * OVVV(i,a,c,b) * mvv1[(size_t)b*nv + c]
               -       OVVV(i,b,c,a) * mvv1[(size_t)b*nv + c];
        L1N(i,a) += v;
      }
    // l2new += einsum('ic,jbca->jiba', l1, ovvv) → L2N[j,i,b,a] += sum_c l1[i,c]*ovvv[j,b,c,a]
    for (int j = 0; j < no; j++)
      for (int i = 0; i < no; i++)
        for (int b = 0; b < nv; b++)
          for (int a = 0; a < nv; a++) {
            real_t v = 0.0;
            for (int c = 0; c < nv; c++)
              v += L1(i,c) * OVVV(j,b,c,a);
            L2N(j,i,b,a) += v;
          }
    // m3 -= einsum('kbca,ijck->ijab', ovvv, l2t1)  where l2t1[i,j,c,k] = sum_d l2[i,j,c,d]*t1[k,d]
    {
        std::vector<real_t> l2t1((size_t)no*no*nv*no, 0.0);
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              for (int k = 0; k < no; k++) {
                real_t v = 0.0;
                for (int dd = 0; dd < nv; dd++)
                  v += L2(i,j,c,dd) * T1(k,dd);
                l2t1[(((size_t)i * no + j) * nv + c) * no + k] = v;
              }
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++) {
                real_t v = 0.0;
                for (int k = 0; k < no; k++)
                  for (int c = 0; c < nv; c++)
                    v += OVVV(k,b,c,a) * l2t1[(((size_t)i * no + j) * nv + c) * no + k];
                M3(i,j,a,b) -= v;
              }
    }

    // l2new += m3
    for (size_t k = 0; k < l2_sz; k++) l2_new[k] += m3[k];

    // l1new += 2*einsum('ijab,jb->ia', m3, t1)
    //       + 2*einsum('jiba,jb->ia', m3, t1)
    //       -   einsum('ijba,jb->ia', m3, t1)
    //       -   einsum('jiab,jb->ia', m3, t1)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            v += 2.0 * M3(i,j,a,b) * T1(j,b);
            v += 2.0 * M3(j,i,b,a) * T1(j,b);
            v -=       M3(i,j,b,a) * T1(j,b);
            v -=       M3(j,i,a,b) * T1(j,b);
          }
        L1N(i,a) += v;
      }

    // l1new -= 2*einsum('iajk,kj->ia', ovoo, moo1) + einsum('jaik,kj->ia', ovoo, moo1)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int j = 0; j < no; j++)
            v -= 2.0 * OVOO(i,a,j,k) * moo1[(size_t)k*no + j]
               -       OVOO(j,a,i,k) * moo1[(size_t)k*no + j];
        L1N(i,a) += v;  // sign already incorporated
      }
    // l2new -= einsum('ka,jbik->ijab', l1, ovoo)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              v += L1(k,a) * OVOO(j,b,i,k);
            L2N(i,j,a,b) -= v;
          }

    // l2theta = 2*l2 - l2.T(0,1,3,2)
    // l2new += 0.5 * einsum('ikac,jbck->ijab', l2theta, wovvo)
    // tmp = einsum('ikca,jbck->ijab', l2, woVVo)
    // l2new += 0.5 * tmp
    // l2new += tmp.T(1,0,2,3)
    {
        std::vector<real_t> tmp(l2_sz, 0.0);
        auto TMP = [&](int i, int j, int a, int b) -> real_t& { return tmp[idx_t2(d,i,j,a,b)]; };
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++) {
                real_t v_ovvo = 0.0, v_ovvo2 = 0.0;
                for (int k = 0; k < no; k++)
                  for (int c = 0; c < nv; c++) {
                    real_t l2theta_ikac = 2.0 * L2(i,k,a,c) - L2(i,k,c,a);
                    v_ovvo  += l2theta_ikac * WOVVO(j,b,c,k);
                    v_ovvo2 += L2(i,k,c,a)  * WoVVo(j,b,c,k);
                  }
                L2N(i,j,a,b) += 0.5 * v_ovvo;
                TMP(i,j,a,b) = v_ovvo2;
              }
        // l2new += 0.5 * tmp + tmp.T(1,0,2,3)
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++)
                L2N(i,j,a,b) += 0.5 * TMP(i,j,a,b) + TMP(j,i,a,b);
    }

    // l1new += fov (=0 for canonical, ≠ 0 for semi-canonical/DMET cluster)
    if (fov_active) {
        for (int i = 0; i < no; i++)
            for (int a = 0; a < nv; a++)
                L1N(i, a) += fov_active[(size_t)i*nv + a];
    }
    // l1new += einsum('ib,ba->ia', l1, v1)   v1[b,a]
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int b = 0; b < nv; b++) v += L1(i,b) * v1[(size_t)b*nv + a];
        L1N(i,a) += v;
      }
    // l1new -= einsum('ja,ij->ia', l1, v2)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++) v += L1(j,a) * v2[(size_t)i*no + j];
        L1N(i,a) -= v;
      }
    // l1new += 2 * einsum('jb,iabj->ia', l1, ovvo)
    // l1new -=     einsum('jb,ijba->ia', l1, oovv)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            v += 2.0 * L1(j,b) * OVVO(i,a,b,j);
            v -=       L1(j,b) * OOVV(i,j,b,a);
          }
        L1N(i,a) += v;
      }
    // l1new -= einsum('ijbc,bacj->ia', l2, wvvvo)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++)
            for (int c = 0; c < nv; c++)
              v += L2(i,j,b,c) * WVVVO(b,a,c,j);
        L1N(i,a) -= v;
      }
    // l1new -= einsum('kjca,ijck->ia', l2, woovo)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int j = 0; j < no; j++)
            for (int c = 0; c < nv; c++)
              v += L2(k,j,c,a) * WOOVO(i,j,c,k);
        L1N(i,a) -= v;
      }
    // l1new += 2 * einsum('ijab,bj->ia', l2, w3)
    //       -     einsum('ijba,bj->ia', l2, w3)
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            v += 2.0 * L2(i,j,a,b) * w3[(size_t)b*no + j];
            v -=       L2(i,j,b,a) * w3[(size_t)b*no + j];
          }
        L1N(i,a) += v;
      }

    // ----- denominators and final update -----
    //   eia[i,a] = ε_i - ε_{nocc+a}     (NEGATIVE)
    //   l1new /= eia ; l1new += l1
    //   l2new = l2new + l2new.T(1,0,3,2)
    //   l2new /= eia[i,a] + eia[j,b]    ; l2new += l2
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t eia = eps[i] - eps[no + a];
        L1N(i,a) /= eia;
        L1N(i,a) += L1(i,a);
      }
    {
        std::vector<real_t> l2sym(l2_sz);
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++)
                l2sym[idx_t2(d,i,j,a,b)] = l2_new[idx_t2(d,i,j,a,b)]
                                         + l2_new[idx_t2(d,j,i,b,a)];
        for (int i = 0; i < no; i++)
          for (int j = 0; j < no; j++)
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++) {
                real_t denom = (eps[i] - eps[no + a]) + (eps[j] - eps[no + b]);
                L2N(i,j,a,b) = l2sym[idx_t2(d,i,j,a,b)] / denom + L2(i,j,a,b);
              }
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
//  Public: solve_ccsd_lambda_cpu
// ---------------------------------------------------------------------------

bool solve_ccsd_lambda_cpu(
    int nocc, int nvir,
    const real_t* h_eps,
    const real_t* h_eri_mo,
    const real_t* h_t1,
    const real_t* h_t2,
    real_t* h_lambda1,
    real_t* h_lambda2,
    int max_iter,
    real_t tol,
    int verbose,
    const real_t* h_fov_active)
{
    LambdaDims d{nocc, nvir, nocc + nvir};
    size_t l1_sz = (size_t)nocc * nvir;
    size_t l2_sz = (size_t)nocc * nocc * nvir * nvir;

    // Pre-extract MO ERI sub-blocks (all needed for Lambda equations)
    auto ovov = extract_ovov(d, h_eri_mo);
    auto ovoo = extract_ovoo(d, h_eri_mo);
    auto ovvv = extract_ovvv(d, h_eri_mo);
    auto oovv = extract_oovv(d, h_eri_mo);
    auto ovvo = extract_ovvo(d, h_eri_mo);
    auto oooo = extract_oooo(d, h_eri_mo);
    auto vvvv = extract_vvvv(d, h_eri_mo);

    // ---- Direct solve (Lambda equations are LINEAR in Λ) ----
    //
    // Builds the full Jacobian J by N+1 calls to update_lambda_full and solves
    // (I − J)·Λ = b via Gaussian elimination. Reliable but costs O(N) update
    // sweeps; for N ≈ 1000 this is ~10 s. Only used as a *last resort* — with
    // canonical embedding orbitals the iterative DIIS solver below converges
    // in tens of iterations, ~30× faster.
    //
    // (Pre-embedding-HF era: direct was the default for N ≤ 2000 because the
    // iterative solver would diverge on non-canonical embedding orbitals.)
    const size_t N = l1_sz + l2_sz;
    constexpr size_t DIRECT_THRESHOLD = 50;  // tiny embeddings only
    if (N <= DIRECT_THRESHOLD) {
        if (verbose > 0)
            std::cout << "  Lambda direct solve: N=" << N << std::endl;

        // b = f(0)
        std::vector<real_t> l1_zero(l1_sz, 0.0), l2_zero(l2_sz, 0.0);
        std::vector<real_t> l1_b(l1_sz), l2_b(l2_sz);
        update_lambda_full(d, h_t1, h_t2, l1_zero.data(), l2_zero.data(),
                           h_eps, ovov, ovoo, ovvv, oovv, ovvo, oooo, vvvv,
                           l1_b.data(), l2_b.data(), h_fov_active);

        std::vector<real_t> b(N);
        for (size_t k = 0; k < l1_sz; k++) b[k] = l1_b[k];
        for (size_t k = 0; k < l2_sz; k++) b[l1_sz + k] = l2_b[k];

        // Build A = I - J,  where J[:,k] = f(e_k) - b
        std::vector<real_t> A(N * N, 0.0);
        for (size_t i = 0; i < N; i++) A[i * N + i] = 1.0;  // I

        for (size_t col = 0; col < N; col++) {
            std::vector<real_t> l1_ek(l1_sz, 0.0), l2_ek(l2_sz, 0.0);
            if (col < l1_sz) l1_ek[col] = 1.0;
            else l2_ek[col - l1_sz] = 1.0;

            std::vector<real_t> l1_fk(l1_sz), l2_fk(l2_sz);
            update_lambda_full(d, h_t1, h_t2, l1_ek.data(), l2_ek.data(),
                               h_eps, ovov, ovoo, ovvv, oovv, ovvo, oooo, vvvv,
                               l1_fk.data(), l2_fk.data(), h_fov_active);

            for (size_t row = 0; row < l1_sz; row++)
                A[row * N + col] -= (l1_fk[row] - b[row]);
            for (size_t row = 0; row < l2_sz; row++)
                A[(l1_sz + row) * N + col] -= (l2_fk[row] - b[l1_sz + row]);
        }

        // Regularize: antisymmetric L2 components make (I-J) singular
        // Add small diagonal to break degeneracy
        for (size_t i = 0; i < N; i++) A[i * N + i] += 1e-10;

        // Solve A * x = b by Gaussian elimination with partial pivoting
        std::vector<real_t> x = b;
        bool ok = true;
        for (size_t col = 0; col < N && ok; col++) {
            size_t pivot = col;
            for (size_t row = col + 1; row < N; row++)
                if (std::abs(A[row * N + col]) > std::abs(A[pivot * N + col]))
                    pivot = row;
            if (std::abs(A[pivot * N + col]) < 1e-14) { ok = false; break; }
            if (pivot != col) {
                for (size_t j = 0; j < N; j++) std::swap(A[col * N + j], A[pivot * N + j]);
                std::swap(x[col], x[pivot]);
            }
            for (size_t row = col + 1; row < N; row++) {
                real_t f = A[row * N + col] / A[col * N + col];
                for (size_t j = col; j < N; j++) A[row * N + j] -= f * A[col * N + j];
                x[row] -= f * x[col];
            }
        }
        if (ok) {
            for (int row = (int)N - 1; row >= 0; row--) {
                for (size_t j = row + 1; j < N; j++) x[row] -= A[row * N + j] * x[j];
                x[row] /= A[row * N + row];
            }

            // Copy and symmetrize L2: l2[i,j,a,b] = (l2[i,j,a,b]+l2[j,i,b,a])/2
            std::copy(x.begin(), x.begin() + l1_sz, h_lambda1);
            std::copy(x.begin() + l1_sz, x.end(), h_lambda2);
            for (int i = 0; i < nocc; i++)
              for (int j = 0; j <= i; j++)
                for (int a = 0; a < nvir; a++)
                  for (int b = 0; b < nvir; b++) {
                    size_t ij = ((size_t)i*nocc+j)*nvir*nvir + a*nvir+b;
                    size_t ji = ((size_t)j*nocc+i)*nvir*nvir + b*nvir+a;
                    real_t sym = 0.5*(h_lambda2[ij]+h_lambda2[ji]);
                    h_lambda2[ij] = sym; h_lambda2[ji] = sym;
                  }

            // Verify: compute residual f(x) - x
            std::vector<real_t> l1_check(l1_sz), l2_check(l2_sz);
            update_lambda_full(d, h_t1, h_t2, h_lambda1, h_lambda2,
                               h_eps, ovov, ovoo, ovvv, oovv, ovvo, oooo, vvvv,
                               l1_check.data(), l2_check.data(), h_fov_active);
            real_t resid = 0.0;
            for (size_t k = 0; k < l1_sz; k++) {
                real_t d = l1_check[k] - h_lambda1[k]; resid += d * d; }
            for (size_t k = 0; k < l2_sz; k++) {
                real_t d = l2_check[k] - h_lambda2[k]; resid += d * d; }
            resid = std::sqrt(resid);

            if (verbose > 0)
                std::cout << "  Lambda direct solve: residual = " << std::scientific
                          << resid << std::defaultfloat << std::endl;

            if (resid < 1e-4) {
                if (verbose > 0)
                    std::cout << "  Lambda converged (direct solve)" << std::endl;
                return true;
            }
        }
        if (verbose > 0)
            std::cout << "  Lambda direct solve failed, trying iterative..." << std::endl;
    }

    // ---- Iterative solve (fallback) ----
    std::vector<real_t> l1_new(l1_sz), l2_new(l2_sz);

    // DIIS storage
    const int diis_size = 8;
    std::vector<std::vector<real_t>> diis_vecs, diis_errs;
    const size_t l_total = l1_sz + l2_sz;

    if (verbose > 0) {
        std::cout << "  Lambda solver: nocc=" << nocc << " nvir=" << nvir
                  << " max_iter=" << max_iter << " tol=" << std::scientific
                  << tol << std::defaultfloat << std::endl;
    }

    real_t prev_resid = 1e30;
    const real_t damp_init = 0.5;  // initial damping factor
    real_t damp = damp_init;

    for (int iter = 0; iter < max_iter; iter++) {
        update_lambda_full(d, h_t1, h_t2, h_lambda1, h_lambda2,
                           h_eps,
                           ovov, ovoo, ovvv, oovv, ovvo, oooo, vvvv,
                           l1_new.data(), l2_new.data(), h_fov_active);

        // Compute residual (error vector)
        std::vector<real_t> err_vec(l_total);
        real_t r_sq = 0.0;
        for (size_t k = 0; k < l1_sz; k++) {
            err_vec[k] = l1_new[k] - h_lambda1[k];
            r_sq += err_vec[k] * err_vec[k];
        }
        for (size_t k = 0; k < l2_sz; k++) {
            err_vec[l1_sz + k] = l2_new[k] - h_lambda2[k];
            r_sq += err_vec[l1_sz + k] * err_vec[l1_sz + k];
        }
        real_t resid = std::sqrt(r_sq);

        // Damping if residual is increasing
        if (resid > prev_resid * 1.5 && iter < 10) {
            damp = std::max(damp * 0.5, 0.1);
        } else if (resid < prev_resid * 0.9 && iter > 5) {
            damp = std::min(damp * 1.2, 1.0);
        }

        // Apply damping: λ = damp * λ_new + (1-damp) * λ_old
        if (damp < 1.0 - 1e-12 && iter < 20) {
            for (size_t k = 0; k < l1_sz; k++)
                l1_new[k] = damp * l1_new[k] + (1.0 - damp) * h_lambda1[k];
            for (size_t k = 0; k < l2_sz; k++)
                l2_new[k] = damp * l2_new[k] + (1.0 - damp) * h_lambda2[k];
        }

        // DIIS extrapolation (start after damping stabilizes)
        if (iter >= 5 && resid < 5.0) {
            std::vector<real_t> vec(l_total);
            std::copy(l1_new.begin(), l1_new.end(), vec.begin());
            std::copy(l2_new.begin(), l2_new.end(), vec.begin() + l1_sz);

            // Save pre-DIIS for fallback
            std::vector<real_t> l1_pre(l1_new.begin(), l1_new.end());
            std::vector<real_t> l2_pre(l2_new.begin(), l2_new.end());

            diis_vecs.push_back(std::move(vec));
            diis_errs.push_back(err_vec);
            if ((int)diis_vecs.size() > diis_size) {
                diis_vecs.erase(diis_vecs.begin());
                diis_errs.erase(diis_errs.begin());
            }

            int nd = (int)diis_vecs.size();
            if (nd >= 2) {
                std::vector<real_t> B((nd+1)*(nd+1), 0.0);
                for (int i = 0; i < nd; i++)
                    for (int j = 0; j <= i; j++) {
                        real_t dot = 0.0;
                        for (size_t k = 0; k < l_total; k++)
                            dot += diis_errs[i][k] * diis_errs[j][k];
                        B[i*(nd+1)+j] = dot;
                        B[j*(nd+1)+i] = dot;
                    }
                for (int i = 0; i < nd; i++) {
                    B[i*(nd+1)+nd] = -1.0;
                    B[nd*(nd+1)+i] = -1.0;
                }

                std::vector<real_t> rhs(nd+1, 0.0);
                rhs[nd] = -1.0;

                int n = nd + 1;
                std::vector<real_t> A = B;
                std::vector<real_t> x = rhs;
                bool ok = true;
                for (int col = 0; col < n && ok; col++) {
                    int pivot = col;
                    for (int row = col+1; row < n; row++)
                        if (std::abs(A[row*n+col]) > std::abs(A[pivot*n+col])) pivot = row;
                    if (std::abs(A[pivot*n+col]) < 1e-14) { ok = false; break; }
                    if (pivot != col) {
                        for (int j = 0; j < n; j++) std::swap(A[col*n+j], A[pivot*n+j]);
                        std::swap(x[col], x[pivot]);
                    }
                    for (int row = col+1; row < n; row++) {
                        real_t f = A[row*n+col] / A[col*n+col];
                        for (int j = col; j < n; j++) A[row*n+j] -= f * A[col*n+j];
                        x[row] -= f * x[col];
                    }
                }
                if (ok) {
                    for (int row = n-1; row >= 0; row--) {
                        for (int j = row+1; j < n; j++) x[row] -= A[row*n+j] * x[j];
                        x[row] /= A[row*n+row];
                    }
                    std::fill(l1_new.begin(), l1_new.end(), 0.0);
                    std::fill(l2_new.begin(), l2_new.end(), 0.0);
                    for (int i = 0; i < nd; i++) {
                        for (size_t k = 0; k < l1_sz; k++)
                            l1_new[k] += x[i] * diis_vecs[i][k];
                        for (size_t k = 0; k < l2_sz; k++)
                            l2_new[k] += x[i] * diis_vecs[i][k + l1_sz];
                    }

                    // Check if DIIS made things worse → revert and reset
                    real_t diis_r = 0.0;
                    for (size_t k = 0; k < l1_sz; k++) {
                        real_t d = l1_new[k] - h_lambda1[k]; diis_r += d*d; }
                    for (size_t k = 0; k < l2_sz; k++) {
                        real_t d = l2_new[k] - h_lambda2[k]; diis_r += d*d; }
                    diis_r = std::sqrt(diis_r);
                    if (diis_r > resid * 3.0) {
                        // DIIS diverged → revert and clear history
                        std::copy(l1_pre.begin(), l1_pre.end(), l1_new.begin());
                        std::copy(l2_pre.begin(), l2_pre.end(), l2_new.begin());
                        diis_vecs.clear();
                        diis_errs.clear();
                    }
                }
            }
        }

        prev_resid = resid;
        std::copy(l1_new.begin(), l1_new.end(), h_lambda1);
        std::copy(l2_new.begin(), l2_new.end(), h_lambda2);

        if (verbose > 0 && (iter < 5 || iter % 10 == 0 || resid < tol)) {
            std::cout << "  Lambda iter " << std::setw(3) << (iter + 1)
                      << ": ||Δλ|| = " << std::scientific << std::setprecision(3)
                      << resid << " damp=" << std::fixed << std::setprecision(2)
                      << damp << std::defaultfloat << std::endl;
        }
        { double vals[] = {resid}; report_progress("ccsd_lambda", iter + 1, 1, vals); }

        if (resid < tol) {
            if (verbose > 0) {
                std::cout << "  Lambda converged in " << (iter + 1) << " iterations"
                          << std::endl;
            }
            return true;
        }
    }

    if (verbose > 0) {
        std::cout << "  Warning: Lambda did not converge in " << max_iter
                  << " iterations (resid=" << std::scientific << prev_resid
                  << std::defaultfloat << ")" << std::endl;
    }
    return false;
}

// ---------------------------------------------------------------------------
//  Public: build_ccsd_1rdm_mo_cpu
//
//  Following the spin-traced ccsd_rdm._gamma1_intermediates (spin-traced RHF 1-RDM).
//
//  doo[i,j] = -Σ_a t1[j,a]*l1[i,a] - Σ_kab θ[j,k,a,b]*l2[i,k,a,b]
//  dvv[a,b] =  Σ_i t1[i,a]*l1[i,b] + Σ_ijc θ[j,i,c,a]*l2[j,i,c,b]
//  dvo[a,i] =  t1[i,a]
//            + Σ_me θ[i,m,a,e]*l1[m,e]
//            - Σ_m ⟨xt1⟩[m,i]*t1[m,a]
//            - Σ_e t1[i,e]*⟨xt2⟩[e,a]
//  dov[i,a] =  l1[i,a]
//  where θ[i,j,a,b] = 2*t2[i,j,a,b] - t2[i,j,b,a]
//        xt1[m,i]  = Σ_nef l2[m,n,e,f]*θ[i,n,e,f]
//        xt2[e,a]  = Σ_mnf l2[m,n,a,f]*θ[m,n,e,f]
//
//  Assembly (spin-traced, McWeeny convention, with HF reference):
//   dm[:nocc,:nocc] = doo + doo^T + 2*I  (HF contribution on diagonal)
//   dm[:nocc,nocc:] = dov + dvo^T
//   dm[nocc:,:nocc] = (dm[:nocc,nocc:])^T
//   dm[nocc:,nocc:] = dvv + dvv^T
// ---------------------------------------------------------------------------

void build_ccsd_1rdm_mo_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_lambda1, const real_t* h_lambda2,
    real_t* D_mo_out)
{
    const int no = nocc;
    const int nv = nvir;
    const int na = no + nv;

    auto T1 = [&](int i, int a) { return h_t1[i * nv + a]; };
    auto T2 = [&](int i, int j, int a, int b) {
        return h_t2[(((size_t)i * no + j) * nv + a) * nv + b]; };
    auto L1 = [&](int i, int a) { return h_lambda1[i * nv + a]; };
    auto L2 = [&](int i, int j, int a, int b) {
        return h_lambda2[(((size_t)i * no + j) * nv + a) * nv + b]; };

    // theta[i,j,a,b] = 2*t2[i,j,a,b] - t2[i,j,b,a]
    auto TH = [&](int i, int j, int a, int b) { return 2.0 * T2(i,j,a,b) - T2(i,j,b,a); };

    // --- doo[i,j] ---
    std::vector<real_t> doo((size_t)no * no, 0.0);
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        real_t v = 0.0;
        for (int a = 0; a < nv; a++) v -= T1(j,a) * L1(i,a);
        for (int k = 0; k < no; k++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              v -= TH(j,k,a,b) * L2(i,k,a,b);
        doo[i * no + j] = v;
      }

    // --- dvv[a,b] ---
    std::vector<real_t> dvv((size_t)nv * nv, 0.0);
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int i = 0; i < no; i++) v += T1(i,a) * L1(i,b);
        for (int j = 0; j < no; j++)
          for (int i = 0; i < no; i++)
            for (int c = 0; c < nv; c++)
              v += TH(j,i,c,a) * L2(j,i,c,b);
        dvv[a * nv + b] = v;
      }

    // --- xt1[m,i] = Σ_nef l2[m,n,e,f]*θ[i,n,e,f] ---
    std::vector<real_t> xt1((size_t)no * no, 0.0);
    for (int m = 0; m < no; m++)
      for (int i = 0; i < no; i++) {
        real_t v = 0.0;
        for (int n = 0; n < no; n++)
          for (int e = 0; e < nv; e++)
            for (int f = 0; f < nv; f++)
              v += L2(m,n,e,f) * TH(i,n,e,f);
        xt1[m * no + i] = v;
      }

    // --- xt2[e,a] = Σ_mnf l2[m,n,a,f]*θ[m,n,e,f] ---
    std::vector<real_t> xt2((size_t)nv * nv, 0.0);
    for (int e = 0; e < nv; e++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int m = 0; m < no; m++)
          for (int n = 0; n < no; n++)
            for (int f = 0; f < nv; f++)
              v += L2(m,n,a,f) * TH(m,n,e,f);
        xt2[e * nv + a] = v;
      }

    // --- dvo[a,i] = t1[i,a] + θ·l1 - xt1·t1 - t1·xt2 ---
    std::vector<real_t> dvo((size_t)nv * no, 0.0);
    for (int a = 0; a < nv; a++)
      for (int i = 0; i < no; i++) {
        real_t v = T1(i,a);
        for (int m = 0; m < no; m++)
          for (int e = 0; e < nv; e++)
            v += TH(i,m,a,e) * L1(m,e);
        for (int m = 0; m < no; m++)
          v -= xt1[m * no + i] * T1(m,a);
        for (int e = 0; e < nv; e++)
          v -= T1(i,e) * xt2[e * nv + a];
        dvo[a * no + i] = v;
      }

    // --- Assemble dm[na,na], McWeeny convention ---
    std::fill(D_mo_out, D_mo_out + (size_t)na * na, 0.0);

    // oo block: doo + doo^T + 2*I (HF diagonal)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        D_mo_out[i * na + j] = doo[i * no + j] + doo[j * no + i];
    for (int i = 0; i < no; i++)
      D_mo_out[i * na + i] += 2.0;

    // vv block: dvv + dvv^T   (offset: rows/cols start at nocc)
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++)
        D_mo_out[(no + a) * na + (no + b)] = dvv[a * nv + b] + dvv[b * nv + a];

    // ov block (upper right): dov + dvo^T where dov = L1
    // dm[i, nocc+a] = L1[i,a] + dvo[a,i]
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t val = L1(i,a) + dvo[a * no + i];
        D_mo_out[i * na + (no + a)] = val;
        D_mo_out[(no + a) * na + i] = val; // symmetric (real, RHF)
      }
}

// ---------------------------------------------------------------------------
//  Public: transform_density_mo_to_ao_cpu
//   D_AO = C · D_MO · C^T     (row-major, C[nao×nao], col = MO)
// ---------------------------------------------------------------------------

void transform_density_mo_to_ao_cpu(
    int nao,
    const real_t* h_C,
    const real_t* h_D_mo,
    real_t* h_D_ao_out)
{
    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        C(h_C, nao, nao);
    Eigen::Map<const Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Dmo(h_D_mo, nao, nao);
    Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Dao(h_D_ao_out, nao, nao);

    Dao = C * Dmo * C.transpose();
}

// ============================================================================
//  CCSD 2-RDM (CPU, for DMET)
//
//  Following the spin-traced ccsd_rdm._make_rdm2 (spatial orbital, chemist notation).
//  Γ[p,q,r,s] in chemist (pq|rs) ordering.
//  Includes HF reference: Γ_HF[i,j,i,j] = 4, Γ_HF[i,j,j,i] = -2.
//
//  For embedding spaces (n_emb ~ 10-30), O(n^4) storage and O(n^5) construction
//  are negligible.
// ============================================================================

void build_ccsd_2rdm_mo_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_l1, const real_t* h_l2,
    real_t* D2)
{
    const int no = nocc, nv = nvir, na = no + nv;
    const size_t na2 = (size_t)na * na;
    const size_t na4 = na2 * na2;

    auto T1 = [&](int i, int a) -> real_t { return h_t1[i * nv + a]; };
    auto T2 = [&](int i, int j, int a, int b) -> real_t {
        return h_t2[(((size_t)i * no + j) * nv + a) * nv + b]; };
    auto L1 = [&](int i, int a) -> real_t { return h_l1[i * nv + a]; };
    auto L2 = [&](int i, int j, int a, int b) -> real_t {
        return h_l2[(((size_t)i * no + j) * nv + a) * nv + b]; };

    // tau[i,j,a,b] = t2[i,j,a,b] + t1[i,a]*t1[j,b]
    auto TAU = [&](int i, int j, int a, int b) -> real_t {
        return T2(i,j,a,b) + T1(i,a) * T1(j,b); };

    // D2 index: chemist (pq|rs) → D2[p*na³ + q*na² + r*na + s]
    auto D2idx = [&](int p, int q, int r, int s) -> size_t {
        return ((size_t)p * na + q) * na2 + (size_t)r * na + s; };

    std::memset(D2, 0, na4 * sizeof(real_t));

    // --- 1-RDM intermediates (reuse from 1-RDM code) ---
    // doo[i,j] = -Σ_a t1[j,a]*l1[i,a] - Σ_kab θ[j,k,a,b]*l2[i,k,a,b]
    std::vector<real_t> doo(no * no, 0.0);
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        real_t v = 0.0;
        for (int a = 0; a < nv; a++) v -= T1(j,a) * L1(i,a);
        for (int k = 0; k < no; k++)
          for (int a = 0; a < nv; a++)
            for (int b = 0; b < nv; b++)
              v -= (2*T2(j,k,a,b) - T2(j,k,b,a)) * L2(i,k,a,b);
        doo[i * no + j] = v;
      }

    // dvv[a,b] = Σ_i t1[i,a]*l1[i,b] + Σ_ijc θ[j,i,c,a]*l2[j,i,c,b]
    std::vector<real_t> dvv(nv * nv, 0.0);
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int i = 0; i < no; i++) v += T1(i,a) * L1(i,b);
        for (int j = 0; j < no; j++)
          for (int i = 0; i < no; i++)
            for (int c = 0; c < nv; c++)
              v += (2*T2(j,i,c,a) - T2(j,i,a,c)) * L2(j,i,c,b);
        dvv[a * nv + b] = v;
      }

    // HF reference (note: this function is currently unused; convention WIP)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        D2[D2idx(i,j,i,j)] += 4.0;
        D2[D2idx(i,j,j,i)] -= 2.0;
      }

    // --- oooo block: Γ[i,j,k,l] += Σ_ab tau[i,j,a,b]*l2[k,l,a,b] + transpose ---
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int l = 0; l < no; l++) {
            real_t v = 0.0;
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++)
                v += TAU(i,j,a,b) * L2(k,l,a,b);
            D2[D2idx(i,j,k,l)] += v + v;  // factor 2 for spin trace
          }

    // --- oovv block: Γ[i,j,a+no,b+no] += tau[i,j,a,b] + ... ---
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = TAU(i,j,a,b) + TAU(j,i,b,a);
            D2[D2idx(i,j,no+a,no+b)] += v;
            D2[D2idx(no+a,no+b,i,j)] += v;  // symmetry
          }

    // --- vvvv block: Γ[a,b,c,d] += Σ_ij l2[i,j,a,b]*tau[i,j,c,d] ---
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int d = 0; d < nv; d++) {
            real_t v = 0.0;
            for (int i = 0; i < no; i++)
              for (int j = 0; j < no; j++)
                v += L2(i,j,a,b) * TAU(i,j,c,d);
            real_t sym = v + v;
            D2[D2idx(no+a,no+b,no+c,no+d)] += sym;
          }

    // --- ovov block: Γ[i,a+no,j,b+no] from l2, t1, doo, dvv ---
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            real_t v = L2(i,j,a,b) + L2(j,i,b,a);
            // Add 1-RDM contributions: -doo * δ, +dvv * δ
            if (i == j) v += dvv[a * nv + b] + dvv[b * nv + a];
            if (a == b) v += doo[i * no + j] + doo[j * no + i];
            if (i == j && a == b) v += 2.0;  // HF diagonal
            D2[D2idx(i,no+a,j,no+b)] += v;
            D2[D2idx(j,no+b,i,no+a)] += v;  // symmetry
          }

    // --- ooov block: Γ[i,j,k,a+no] from l1, l2, t1, t2 ---
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int a = 0; a < nv; a++) {
            real_t v = 0.0;
            // l2 contraction with t1
            for (int b = 0; b < nv; b++)
              v += L2(i,j,a,b) * T1(k,b) - L2(k,j,a,b) * T1(i,b);
            // l1 contribution
            if (i == k) v += L1(j,a);
            if (j == k) v -= 0.5 * L1(i,a);
            real_t sym = v + v;
            D2[D2idx(i,j,k,no+a)] += sym;
            D2[D2idx(k,no+a,i,j)] += sym;  // symmetry
          }

    // --- ovvv block: Γ[i,a+no,b+no,c+no] from l1, l2, t1, t2 ---
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
          for (int c = 0; c < nv; c++) {
            real_t v = 0.0;
            for (int j = 0; j < no; j++)
              v += L2(i,j,b,c) * T1(j,a) - L2(i,j,a,c) * T1(j,b);
            if (a == b) v += L1(i,c);
            if (a == c) v -= 0.5 * L1(i,b);
            real_t sym = v + v;
            D2[D2idx(i,no+a,no+b,no+c)] += sym;
            D2[D2idx(no+b,no+c,i,no+a)] += sym;  // symmetry
          }
}

// ============================================================================
//  CCSD 2-RDM in chemist convention (CPU).
//
//  Direct port of the reference 2-RDM._gamma2_outcore + _make_rdm2.
//  Verified element-wise on H2O/STO-3G (no=5,nv=2): max|diff| < 1e-16.
//
//  Convention: E = einsum('pq,qp', h_core, dm1) + 0.5*einsum('pqrs,pqrs', eri, dm2) + E_nuc
//  (final transpose(1,0,3,2) is applied; HF reference is included via with_dm1.)
// ============================================================================
void build_ccsd_2rdm_chemist_cpu(
    int nocc, int nvir,
    const real_t* h_t1, const real_t* h_t2,
    const real_t* h_l1, const real_t* h_l2,
    const real_t* dm1,
    real_t* D2)
{
    const int no = nocc, nv = nvir, na = no + nv;
    const size_t na2 = (size_t)na * na;
    const size_t na4 = na2 * na2;

    auto T1 = [&](int i, int a) -> real_t { return h_t1[i*nv+a]; };
    auto T2 = [&](int i, int j, int a, int b) -> real_t {
        return h_t2[(((size_t)i*no+j)*nv+a)*nv+b]; };
    auto L1 = [&](int i, int a) -> real_t { return h_l1[i*nv+a]; };
    auto L2 = [&](int i, int j, int a, int b) -> real_t {
        return h_l2[(((size_t)i*no+j)*nv+a)*nv+b]; };
    auto TAU = [&](int i, int j, int a, int b) -> real_t {
        return T2(i,j,a,b) + T1(i,a)*T1(j,b); };
    auto TH = [&](int i, int j, int a, int b) -> real_t {
        return 2.0*T2(i,j,a,b) - T2(i,j,b,a); };

    // --- Pass 1: pvOOv, pvoOV, moo, mvv, mia, mab, mij ---
    // pvOOv[a,i,j,b] = Σ_{k,c} l2[i,k,c,a]*t2[j,k,c,b]
    std::vector<real_t> pvOOv((size_t)nv*no*no*nv, 0.0);
    auto PVOOV = [&](int a, int i, int j, int b) -> real_t& {
        return pvOOv[(((size_t)a*no+i)*no+j)*nv+b]; };
    for (int a = 0; a < nv; a++)
      for (int i = 0; i < no; i++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              for (int c = 0; c < nv; c++)
                v += L2(i,k,c,a) * T2(j,k,c,b);
            PVOOV(a,i,j,b) = v;
          }

    // pvoOV[a,i,j,b] = -Σ l2[i,k,c,a]*t2[j,k,b,c] + Σ l2[i,k,a,c]*theta[j,k,b,c]
    std::vector<real_t> pvoOV((size_t)nv*no*no*nv, 0.0);
    auto PVOV = [&](int a, int i, int j, int b) -> real_t& {
        return pvoOV[(((size_t)a*no+i)*no+j)*nv+b]; };
    for (int a = 0; a < nv; a++)
      for (int i = 0; i < no; i++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              for (int c = 0; c < nv; c++) {
                v -= L2(i,k,c,a) * T2(j,k,b,c);
                v += L2(i,k,a,c) * TH(j,k,b,c);
              }
            PVOV(a,i,j,b) = v;
          }

    // moo[j,l] = 2 * Σ_d pvOOv[d,l,j,d] + Σ_d pvoOV[d,l,j,d]
    //   (factor 2 only on the pvOOv contribution; chemist convention)
    std::vector<real_t> moo(no*no, 0.0);
    for (int j = 0; j < no; j++)
      for (int l = 0; l < no; l++) {
        real_t v = 0.0;
        for (int d = 0; d < nv; d++)
          v += 2.0*PVOOV(d,l,j,d) + PVOV(d,l,j,d);
        moo[j*no+l] = v;
      }

    // mvv[d,b] = 2 * Σ_l pvOOv[b,l,l,d] + Σ_l pvoOV[b,l,l,d]
    std::vector<real_t> mvv(nv*nv, 0.0);
    for (int d = 0; d < nv; d++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int l = 0; l < no; l++)
          v += 2.0*PVOOV(b,l,l,d) + PVOV(b,l,l,d);
        mvv[d*nv+b] = v;
      }

    // mia[i,a] = Σ_k,c l1[k,c]*(2*t2[i,k,a,c]-t2[i,k,c,a])
    std::vector<real_t> mia(no*nv, 0.0);
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            v += L1(k,c) * (2.0*T2(i,k,a,c) - T2(i,k,c,a));
        mia[i*nv+a] = v;
      }

    // mab[c,b] = Σ_k l1[k,c]*t1[k,b]
    std::vector<real_t> mab(nv*nv, 0.0);
    for (int c = 0; c < nv; c++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++) v += L1(k,c)*T1(k,b);
        mab[c*nv+b] = v;
      }

    // mij[j,k] = Σ_c l1[k,c]*t1[j,c] + 0.5*moo[j,k]
    std::vector<real_t> mij(no*no, 0.0);
    for (int j = 0; j < no; j++)
      for (int k = 0; k < no; k++) {
        real_t v = 0.0;
        for (int c = 0; c < nv; c++) v += L1(k,c)*T1(j,c);
        mij[j*no+k] = v + 0.5*moo[j*no+k];
      }

    // --- goooo, doooo ---
    // tau[i,j,a,b] = t1[i,a]*t1[j,b] + t2[i,j,a,b]
    // goooo[i,j,k,l] = 0.5 * Σ_ab tau[i,j,a,b]*l2[k,l,a,b]
    std::vector<real_t> goooo((size_t)no*no*no*no, 0.0);
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int l = 0; l < no; l++) {
            real_t v = 0.0;
            for (int a = 0; a < nv; a++)
              for (int b = 0; b < nv; b++)
                v += TAU(i,j,a,b)*L2(k,l,a,b);
            goooo[((size_t)i*no+j)*no*no+k*no+l] = 0.5*v;
          }

    // doooo[i,j,k,l] = goooo.T(0,2,1,3)*2 - goooo.T(0,3,1,2)
    //               = goooo[i,k,j,l]*2 - goooo[i,k,l,j]
    std::vector<real_t> doooo((size_t)no*no*no*no, 0.0);
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int l = 0; l < no; l++)
            doooo[((size_t)i*no+j)*no*no+k*no+l] =
                2.0*goooo[((size_t)i*no+k)*no*no+j*no+l]   // [i,k,j,l]
                   -goooo[((size_t)i*no+k)*no*no+l*no+j];  // [i,k,l,j]

    // --- goovv (Pass 2, simplified for small systems) ---
    // goovv[i,j,a,b] = mia[i,a]*t1[j,b] + ... (many terms)
    std::vector<real_t> goovv((size_t)no*no*nv*nv, 0.0);
    auto GOOVV = [&](int i, int j, int a, int b) -> real_t& {
        return goovv[(((size_t)i*no+j)*nv+a)*nv+b]; };

    // goovv = einsum('ia,jb->ijab', mia, t1)
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++)
            GOOVV(i,j,a,b) = mia[i*nv+a]*T1(j,b);

    // tmpoovv = einsum('ijkl,klab->ijab', goooo, tau)
    //         - einsum('jk,ikab->ijab', mij, tau)
    //         - einsum('cb,ijac->ijab', mab, t2)
    //         - einsum('bd,ijad->ijab', mvv*0.5, tau)
    //         + 0.5*tau
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int k = 0; k < no; k++)
              for (int l = 0; l < no; l++)
                v += goooo[((size_t)i*no+j)*no*no+k*no+l]*TAU(k,l,a,b);
            for (int k = 0; k < no; k++)
              v -= mij[j*no+k]*TAU(i,k,a,b);
            for (int c = 0; c < nv; c++)
              v -= mab[c*nv+b]*T2(i,j,a,c);
            for (int d = 0; d < nv; d++)
              v -= 0.5*mvv[b*nv+d]*TAU(i,j,a,d);
            v += 0.5*TAU(i,j,a,b);
            v += 0.5*L2(i,j,a,b);
            GOOVV(i,j,a,b) += v;
          }

    // goovv += einsum('dlib,jlda->ijab', pvOOv, tau_half)
    // goovv -= einsum('dlia,jldb->ijab', pvoOV, tau_half)
    // goovv += einsum('dlia,jlbd->ijab', pvoOV+0.5*pvOOv, t2)
    // where tau_half = tau - 0.5*t2
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t v = 0.0;
            for (int d = 0; d < nv; d++)
              for (int l = 0; l < no; l++) {
                real_t tau_h_jlda = TAU(j,l,d,a) - 0.5*T2(j,l,d,a);
                real_t tau_h_jldb = TAU(j,l,d,b) - 0.5*T2(j,l,d,b);
                v += PVOOV(d,l,i,b)*tau_h_jlda;
                v -= PVOV(d,l,i,a)*tau_h_jldb;
                v += (PVOV(d,l,i,a)+0.5*PVOOV(d,l,i,a))*T2(j,l,b,d);
              }
            GOOVV(i,j,a,b) += v;
          }

    // dovov[i,a,j,b] = goovv[i,j,a,b]*2 - goovv[j,i,a,b]  (transposed from goovv)
    // Note: reference does goovv.transpose(0,2,1,3)*2 - goovv.transpose(1,2,0,3)
    std::vector<real_t> dovov((size_t)no*nv*no*nv, 0.0);
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++)
            dovov[(((size_t)i*nv+a)*no+j)*nv+b] =
                2.0*GOOVV(i,j,a,b) - GOOVV(j,i,a,b);

    // --- dovvo, doovv from Pass 2 ---
    //   gOvvO[i,a,b,j] = pvOOv[a,i,j,b] + Σ_{k,c} l2[k,i,a,c]*t1[j,c]*t1[k,b]
    //   govVO[i,a,b,j] = l1[i,a]*t1[j,b] + pvoOV[a,i,j,b] - Σ_{k,c} l2[i,k,a,c]*t1[j,c]*t1[k,b]
    //   dovvo[i,a,b,j] = 2*govVO + gOvvO
    //   doovv[j,i,a,b] = -2*gOvvO - govVO       (via T(3,0,1,2))
    std::vector<real_t> dovvo((size_t)no*nv*nv*no, 0.0);
    std::vector<real_t> doovv((size_t)no*no*nv*nv, 0.0);
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
          for (int j = 0; j < no; j++) {
            real_t gOvvO = PVOOV(a,i,j,b);
            real_t govVO = L1(i,a)*T1(j,b) + PVOV(a,i,j,b);
            for (int k = 0; k < no; k++)
              for (int c = 0; c < nv; c++) {
                gOvvO += L2(k,i,a,c)*T1(j,c)*T1(k,b);
                govVO -= L2(i,k,a,c)*T1(j,c)*T1(k,b);
              }
            dovvo[(((size_t)i*nv+a)*nv+b)*no+j] = 2.0*govVO + gOvvO;
            doovv[(((size_t)j*no+i)*nv+a)*nv+b] = -2.0*gOvvO - govVO;
          }

    // --- dvvvv (Pass 3, simplified) ---
    // gvvvv[a,b,c,d] = Σ_ij l2[i,j,a,b]*t2[i,j,c,d] + Σ_ij l2[i,j,a,b]*t1[i,c]*t1[j,d]
    // dvvvv = gvvvv - gvvvv.transpose(1,0,2) ... (symmetrization)
    std::vector<real_t> dvvvv((size_t)nv*nv*nv*nv, 0.0);
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int d = 0; d < nv; d++) {
            real_t v = 0.0;
            for (int i = 0; i < no; i++)
              for (int j = 0; j < no; j++)
                v += L2(i,j,a,b)*TAU(i,j,c,d);
            // dvvvv = gvvvv(a,c,b,d) - 0.5*gvvvv(b,c,a,d) (from reference symmetrization)
            // But for the non-compressed case, just store v and symmetrize later
            dvvvv[(((size_t)a*nv+b)*nv+c)*nv+d] = v;
          }
    // dvvvv[a,b,c,d] = gvvvv[a,c,b,d] - 0.5 * gvvvv[a,c,d,b]
    //   reference: vvv = gvvvv[a].transpose(1,0,2);  dvvvv[a] = vvv - vvv.transpose(2,1,0)*0.5
    //   Derivation: vvv[c,b,d] = gvvvv[a,b,c,d];  vvv.T(2,1,0)[c,b,d] = gvvvv[a,b,d,c]
    //   so dvvvv[a, X=c, Y=b, Z=d] = gvvvv[a, Y=b, X=c, Z=d] - 0.5*gvvvv[a, Y=b, Z=d, X=c]
    //   relabel (X,Y,Z) → (b',c',d'):  dvvvv[a,b',c',d'] = gvvvv[a,c',b',d'] - 0.5*gvvvv[a,c',d',b']
    {
        std::vector<real_t> tmp = dvvvv;
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++)
            for (int c = 0; c < nv; c++)
              for (int d = 0; d < nv; d++)
                dvvvv[(((size_t)a*nv+b)*nv+c)*nv+d] =
                    tmp[(((size_t)a*nv+c)*nv+b)*nv+d]       // gvvvv[a,c,b,d]
                    - 0.5*tmp[(((size_t)a*nv+c)*nv+d)*nv+b]; // gvvvv[a,c,d,b]
    }

    // --- gooov, dooov ---
    std::vector<real_t> gooov((size_t)no*no*no*nv, 0.0);
    auto GOOOV = [&](int j, int k, int i, int a) -> real_t& {
        return gooov[(((size_t)j*no+k)*no+i)*nv+a]; };
    // gooov[j,k,i,a] = Σ_c t1[k,c]*pvOOv[c,i,j,a] - Σ_c t1[j,c]*pvoOV[c,i,k,a]
    //                - 0.5*moo[j,i]*t1[k,a] + 2*Σ_l t1[l,a]*goooo[j,k,i,l]
    //                - Σ_b l1[i,b]*tau[j,k,b,a] - Σ_b l2[j,k,b,a]*t1[i,b]
    //   (einsum 'ib,jkba->jkia' contracts on b with tau[j,k,b,a])
    for (int j = 0; j < no; j++)
      for (int k = 0; k < no; k++)
        for (int i = 0; i < no; i++)
          for (int a = 0; a < nv; a++) {
            real_t v = 0.0;
            for (int c = 0; c < nv; c++) {
              v += T1(k,c)*PVOOV(c,i,j,a);
              v -= T1(j,c)*PVOV(c,i,k,a);
            }
            v -= 0.5*moo[j*no+i]*T1(k,a);
            for (int l = 0; l < no; l++)
              v += 2.0*T1(l,a)*goooo[((size_t)j*no+k)*no*no+i*no+l];
            for (int b = 0; b < nv; b++) {
              v -= L1(i,b)*TAU(j,k,b,a);
              v -= L2(j,k,b,a)*T1(i,b);
            }
            GOOOV(j,k,i,a) = v;
          }
    // dooov[j,k,i,a] = gooov.T(0,2,1,3)*2 - gooov.T(1,2,0,3)
    //                = gooov[j,i,k,a]*2 - gooov[i,j,k,a]
    std::vector<real_t> dooov((size_t)no*no*no*nv, 0.0);
    for (int j = 0; j < no; j++)
      for (int k = 0; k < no; k++)
        for (int i = 0; i < no; i++)
          for (int a = 0; a < nv; a++)
            dooov[(((size_t)j*no+k)*no+i)*nv+a] =
                2.0*gooov[(((size_t)j*no+i)*no+k)*nv+a]   // [j,i,k,a]
                   -gooov[(((size_t)i*no+j)*no+k)*nv+a];  // [i,j,k,a]

    // --- dovvv (simplified — from gvovv) ---
    // gvovv[a,i,b,c] = -Σ_d gvvvv(unsym)[a,d,b,c]*t1[i,d]
    //                + Σ_k pvoOV[a,k,i,c]*t1[k,b] - Σ_k pvOOv[a,k,i,b]*t1[k,c]
    //                + Σ_j l1[j,a]*t2[j,i,b,c] + Σ_j l1[j,a]*t1[j,b]*t1[i,c]
    //                + 0.5*mvv[?,a]*t1[i,c] (from reference mvv indexing)
    //                + Σ_j t1[j,a]*l2[j,i,b,c]
    // dovvv[i,b,a,c] = gvovv[a,i,b,c]*2 - gvovv[a,i,c,b]  (transposed)
    // For simplicity, compute dovvv directly
    std::vector<real_t> dovvv((size_t)no*nv*nv*nv, 0.0);
    // Compute gvovv_raw (unsymmetrized gvvvv needed)
    std::vector<real_t> gvvvv_raw((size_t)nv*nv*nv*nv, 0.0);
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int d = 0; d < nv; d++) {
            real_t v = 0.0;
            for (int i = 0; i < no; i++)
              for (int j = 0; j < no; j++)
                v += L2(i,j,a,b)*TAU(i,j,c,d);
            gvvvv_raw[(((size_t)a*nv+b)*nv+c)*nv+d] = v;
          }

    for (int a = 0; a < nv; a++)
      for (int i = 0; i < no; i++)
        for (int b = 0; b < nv; b++)
          for (int c = 0; c < nv; c++) {
            real_t v = 0.0;
            for (int d = 0; d < nv; d++)
              v -= gvvvv_raw[(((size_t)a*nv+d)*nv+b)*nv+c]*T1(i,d);
            for (int k = 0; k < no; k++) {
              v += PVOV(a,k,i,c)*T1(k,b);
              v -= PVOOV(a,k,i,b)*T1(k,c);
            }
            for (int j = 0; j < no; j++) {
              v += L1(j,a)*T2(j,i,b,c);
              v += L1(j,a)*T1(j,b)*T1(i,c);
              v += T1(j,a)*L2(j,i,b,c);
            }
            v += 0.5*mvv[b*nv+a]*T1(i,c);  // reference: einsum('ba,ic->aibc', mvv*0.5, t1)
            // dovvv[i,J,K,L] = 2*gvovv[K,i,L,J] - gvovv[K,i,J,L]
            //   from the reference: dovvv = gvovv.T(1,3,0,2)*2 - gvovv.T(1,2,0,3)
            //   loop iter (i,a,b,c) writes gv to dovvv[i,c,a,b]*2 and dovvv[i,b,a,c]*(-1)
            real_t gv = v;
            dovvv[(((size_t)i*nv+c)*nv+a)*nv+b] += 2.0*gv;
            dovvv[(((size_t)i*nv+b)*nv+a)*nv+c] -= gv;
          }

    // =========================================================
    //  Assemble dm2 in reference internal convention
    // =========================================================
    std::memset(D2, 0, na2*na2*sizeof(real_t));
    auto DM2 = [&](int p, int q, int r, int s) -> real_t& {
        return D2[((size_t)p*na+q)*na2 + r*na+s]; };

    // dovov → dm2[i,a,j,b] = dm2[:nocc, nocc:, :nocc, nocc:]
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++) {
            real_t d = dovov[(((size_t)i*nv+a)*no+j)*nv+b];
            DM2(i,no+a,j,no+b) += d;
            DM2(j,no+b,i,no+a) += d;  // += transpose(2,3,0,1)
          }
    // voov = transpose(1,0,3,2) of ovov block
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int j = 0; j < no; j++)
          for (int b = 0; b < nv; b++)
            DM2(no+a,i,no+b,j) = DM2(i,no+a,j,no+b);

    // doovv → dm2[i,j,a,b]
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++) {
            real_t d = doovv[(((size_t)i*no+j)*nv+a)*nv+b];
            DM2(i,j,no+a,no+b) += d;
            DM2(j,i,no+b,no+a) += d;  // += transpose(1,0,3,2)
          }
    // vvoo = transpose(2,3,0,1) of oovv
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int a = 0; a < nv; a++)
          for (int b = 0; b < nv; b++)
            DM2(no+a,no+b,i,j) = DM2(i,j,no+a,no+b);

    // dovvo → dm2[i,a,b,j]
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
          for (int j = 0; j < no; j++) {
            real_t d = dovvo[(((size_t)i*nv+a)*nv+b)*no+j];
            DM2(i,no+a,no+b,j) += d;
            DM2(j,no+b,no+a,i) += d;  // += transpose(3,2,1,0)
          }
    // voov = transpose(1,0,3,2) of ovvo
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
          for (int j = 0; j < no; j++)
            DM2(no+a,i,j,no+b) = DM2(i,no+a,no+b,j);

    // doooo
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int l = 0; l < no; l++) {
            real_t d = doooo[((size_t)i*no+j)*no*no+k*no+l];
            DM2(i,j,k,l) += 2.0*(d + doooo[((size_t)j*no+i)*no*no+l*no+k]);
          }

    // dvvvv
    for (int a = 0; a < nv; a++)
      for (int b = 0; b < nv; b++)
        for (int c = 0; c < nv; c++)
          for (int d = 0; d < nv; d++) {
            real_t v = dvvvv[(((size_t)a*nv+b)*nv+c)*nv+d];
            DM2(no+a,no+b,no+c,no+d) += 2.0*(v + dvvvv[(((size_t)b*nv+a)*nv+d)*nv+c]);
          }

    // dooov: all 4 symmetry partners use the same value d = dooov[i,j,k,a]
    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++)
        for (int k = 0; k < no; k++)
          for (int a = 0; a < nv; a++) {
            real_t d = dooov[(((size_t)i*no+j)*no+k)*nv+a];
            DM2(i,j,k,no+a) += d;          // original
            DM2(k,no+a,i,j) += d;          // transpose(2,3,0,1)
            DM2(j,i,no+a,k) += d;          // transpose(1,0,3,2)
            DM2(no+a,k,j,i) += d;          // transpose(3,2,1,0)
          }

    // dovvv: all 4 symmetry partners use the same d = dovvv[i,a,b,c]
    for (int i = 0; i < no; i++)
      for (int a = 0; a < nv; a++)
        for (int b = 0; b < nv; b++)
          for (int c = 0; c < nv; c++) {
            real_t d = dovvv[(((size_t)i*nv+a)*nv+b)*nv+c];
            DM2(i,no+a,no+b,no+c) += d;    // original
            DM2(no+b,no+c,i,no+a) += d;    // transpose(2,3,0,1)
            DM2(no+a,i,no+c,no+b) += d;    // transpose(1,0,3,2)
            DM2(no+c,no+b,no+a,i) += d;    // transpose(3,2,1,0)
          }

    // =========================================================
    //  with_dm1: add 1-RDM products and HF reference
    // =========================================================
    // dm1_corr = dm1 - 2*I_occ
    std::vector<real_t> dm1c(na*na);
    std::copy(dm1, dm1+na*na, dm1c.begin());
    for (int i = 0; i < no; i++) dm1c[i*na+i] -= 2.0;

    for (int i = 0; i < no; i++) {
        for (int p = 0; p < na; p++)
          for (int q = 0; q < na; q++) {
            DM2(i,i,p,q) += 2.0*dm1c[p*na+q];
            DM2(p,q,i,i) += 2.0*dm1c[p*na+q];
            DM2(p,i,i,q) -= dm1c[p*na+q];
            DM2(i,p,q,i) -= dm1c[q*na+p];
          }
    }

    for (int i = 0; i < no; i++)
      for (int j = 0; j < no; j++) {
        DM2(i,i,j,j) += 4.0;
        DM2(i,j,j,i) -= 2.0;
      }

    // =========================================================
    //  Final transpose: dm2 = dm2.transpose(1,0,3,2)
    // =========================================================
    {
        std::vector<real_t> tmp(na2*na2);
        std::copy(D2, D2+na2*na2, tmp.begin());
        for (int p = 0; p < na; p++)
          for (int q = 0; q < na; q++)
            for (int r = 0; r < na; r++)
              for (int s = 0; s < na; s++)
                D2[((size_t)p*na+q)*na2+r*na+s] = tmp[((size_t)q*na+p)*na2+s*na+r];
    }
}

// ============================================================================
//  DMET democratic fragment energy (T-amplitude based)
// ============================================================================

real_t compute_dmet_fragment_energy(
    int nocc, int nvir,
    const real_t* eri_emb,   // [na^4] canonical MO ERI (chemist)
    const real_t* t1,        // [nocc_act × nvir] T1 amplitudes
    const real_t* t2,        // [nocc_act^2 × nvir^2] T2 amplitudes
    int n_frag,              // fragment AOs in embedding
    const real_t* eigvecs,   // [na × na] embedding→canonical eigenvectors
    int n_frozen,            // frozen core orbitals (first n_frozen MOs skipped by CCSD)
    const real_t* fov_active)// [nocc_act × nvir] semi-canonical f_ov (optional)
{
    const int no = nocc, nv = nvir, na = no + nv;
    const int no_act = no - n_frozen;  // active occupied (T amplitude dimension)
    const size_t na2 = (size_t)na * na;

    // T amplitude accessors — indexed by ACTIVE occupied [0, no_act)
    auto T1 = [&](int i, int a) -> real_t { return t1[i * nv + a]; };
    auto T2 = [&](int i, int j, int a, int b) -> real_t {
        return t2[(((size_t)i * no_act + j) * nv + a) * nv + b]; };
    auto TAU = [&](int i, int j, int a, int b) -> real_t {
        return T2(i, j, a, b) + T1(i, a) * T1(j, b); };

    // ERI accessor — indexed by FULL orbital indices [0, na)
    auto ERI = [&](int p, int q, int r, int s) -> real_t {
        return eri_emb[((size_t)p * na + q) * na2 + (size_t)r * na + s]; };

    // Fragment projector in canonical basis
    std::vector<real_t> P_can(na * na, 0.0);
    for (int i = 0; i < na; i++)
        for (int j = 0; j <= i; j++) {
            real_t val = 0.0;
            for (int p = 0; p < n_frag; p++)
                val += eigvecs[p * na + i] * eigvecs[p * na + j];
            P_can[i * na + j] = val;
            P_can[j * na + i] = val;
        }

    // E_corr_frag = Σ_{i,i'∈active_occ} P[i+nf, i'+nf] * Σ_{j∈active_occ, a,b∈vir}
    //              ((i+nf),a+no | (j+nf),b+no) * (2τ[i',j,a,b] - τ[i',j,b,a])
    // where nf = n_frozen, and i,i',j are ACTIVE indices [0, no_act)
    real_t E_corr_frag = 0.0;
    for (int i = 0; i < no_act; i++)
        for (int ip = 0; ip < no_act; ip++) {
            // P_can uses FULL orbital indices: i+n_frozen, ip+n_frozen
            real_t Pip = P_can[(i + n_frozen) * na + (ip + n_frozen)];
            if (std::abs(Pip) < 1e-15) continue;
            for (int j = 0; j < no_act; j++)
                for (int a = 0; a < nv; a++)
                    for (int b = 0; b < nv; b++) {
                        // ERI uses FULL indices
                        real_t iajb = ERI(i + n_frozen, no + a,
                                          j + n_frozen, no + b);
                        real_t tau_ab = TAU(ip, j, a, b);
                        real_t tau_ba = TAU(ip, j, b, a);
                        E_corr_frag += Pip * iajb * (2.0 * tau_ab - tau_ba);
                    }
        }

    // Semi-canonical Brillouin term: when f_ov ≠ 0 (e.g. Vayesta-canonical
    // DMET clusters with μ-shift), the CCSD correlation energy gains
    //   E_corr += 2 Σ_{i,a} f_ov[i,a] · t1[i,a]
    // which the democratic partition projects with P:
    //   E_corr_frag += 2 Σ_{i,i',a} P[i+nf, i'+nf] · f_ov[i,a] · t1[i',a]
    // Σ_F P_F = I → summing over all fragments recovers the total Brillouin
    // contribution exactly.
    if (fov_active) {
        for (int i = 0; i < no_act; i++)
            for (int ip = 0; ip < no_act; ip++) {
                real_t Pip = P_can[(i + n_frozen) * na + (ip + n_frozen)];
                if (std::abs(Pip) < 1e-15) continue;
                for (int a = 0; a < nv; a++)
                    E_corr_frag += 2.0 * Pip
                                 * fov_active[(size_t)i * nv + a]
                                 * T1(ip, a);
            }
    }

    return E_corr_frag;
}

// ============================================================================
//  AO-projected DMET fragment correlation energy via Lambda-relaxed 1-RDM+2-RDM.
//
//  Cleanly derived form: stays in canonical basis, but applies the fragment
//  projector P[i,i'] = Σ_{p<n_frag} U[p,i] U[p,i'] to one index pair. The
//  orthonormality of U collapses the other three index sums to identities.
//
//      E_corr_frag = E_frag(D1, D2) − E_frag(D1_HF, D2_HF)
//      E_frag      = Σ_{i,i'} P[i,i'] (h_can D1)[i,i']
//                  + (1/2) Σ_{i,i'} P[i,i'] Σ_{jkl} eri_can[i,j,k,l] D2[i',j,k,l]
//      h_can       = U^T h_emb_1e U
//
//  Pass the same 1e operator the CCSD's reference HF uses (h_emb_base = C^T F C
//  in this code) so the result is a true CCSD-correlation energy (≤ 0). Using
//  C^T h_core C instead measures a physical-Hamiltonian energy against the
//  F_emb-HF reference, which is NOT a correlation and can be positive.
//
//  Cost is O(ne⁵) per fragment; for ne ≲ 30 this runs in under 1 ms.
// ============================================================================
real_t compute_dmet_fragment_energy_aoproj(
    int nocc_act, int nvir,
    const real_t* h_emb_1e,
    const real_t* eri_can,
    const real_t* dm1_active,
    const real_t* dm2_active,
    int n_frag,
    const real_t* eigvecs,
    int n_frozen,
    real_t* E1_out,
    real_t* E2_out)
{
    const int na_act = nocc_act + nvir;
    const int ne = n_frozen + na_act;
    const int no_full = n_frozen + nocc_act;
    const size_t ne2 = (size_t)ne * ne;
    const size_t ne4 = ne2 * ne2;

    // 1. dm1_full: HF ref on frozen-frozen, active CCSD on active-active.
    std::vector<real_t> dm1_full(ne2, 0.0);
    for (int i = 0; i < n_frozen; i++) dm1_full[i*ne + i] = 2.0;
    for (int i = 0; i < na_act; i++)
        for (int j = 0; j < na_act; j++)
            dm1_full[(i+n_frozen)*ne + (j+n_frozen)] = dm1_active[i*na_act + j];

    // 2. dm2_full: start from D2_HF over all occupied (frozen + active occ),
    //    then overwrite the all-active block with dm2_active. The all-active
    //    slice of D2_HF_full equals D2_HF_active (since D1_HF on those indices
    //    is the same), so the overwrite correctly replaces "active HF" with
    //    "active HF + correlation" while leaving frozen-* blocks at HF.
    auto D1HF_full = [no_full](int p, int q) -> real_t {
        return (p == q && p < no_full) ? 2.0 : 0.0;
    };
    std::vector<real_t> dm2_full(ne4, 0.0);
    for (int p = 0; p < ne; p++)
        for (int q = 0; q < ne; q++) {
            real_t d_pq = D1HF_full(p,q);
            for (int r = 0; r < ne; r++) {
                real_t d_rq = D1HF_full(r,q);
                for (int s = 0; s < ne; s++) {
                    real_t d_rs = D1HF_full(r,s);
                    real_t d_ps = D1HF_full(p,s);
                    dm2_full[(((size_t)p*ne+q)*ne+r)*ne+s] =
                        d_pq*d_rs - 0.5*d_ps*d_rq;
                }
            }
        }
    for (int i = 0; i < na_act; i++)
        for (int j = 0; j < na_act; j++)
            for (int k = 0; k < na_act; k++)
                for (int l = 0; l < na_act; l++)
                    dm2_full[(((size_t)(i+n_frozen)*ne+(j+n_frozen))*ne+(k+n_frozen))*ne+(l+n_frozen)] =
                        dm2_active[(((size_t)i*na_act+j)*na_act+k)*na_act+l];

    // 3. h_can = U^T h_emb_1e U
    std::vector<real_t> h_can(ne2, 0.0);
    {
        std::vector<real_t> tmp(ne2, 0.0);
        for (int i = 0; i < ne; i++)
            for (int q = 0; q < ne; q++) {
                real_t v = 0.0;
                for (int p = 0; p < ne; p++)
                    v += eigvecs[p*ne + i] * h_emb_1e[p*ne + q];
                tmp[i*ne + q] = v;
            }
        for (int i = 0; i < ne; i++)
            for (int j = 0; j < ne; j++) {
                real_t v = 0.0;
                for (int q = 0; q < ne; q++)
                    v += tmp[i*ne + q] * eigvecs[q*ne + j];
                h_can[i*ne + j] = v;
            }
    }

    // 4. P[i,i'] = Σ_{p<n_frag} U[p,i] U[p,i']
    std::vector<real_t> P(ne2, 0.0);
    for (int i = 0; i < ne; i++)
        for (int j = 0; j <= i; j++) {
            real_t v = 0.0;
            for (int p = 0; p < n_frag; p++)
                v += eigvecs[p*ne + i] * eigvecs[p*ne + j];
            P[i*ne + j] = v;
            P[j*ne + i] = v;
        }

    // 5. HF reference on full ne basis (to subtract).
    std::vector<real_t> D1_HF(ne2, 0.0);
    for (int i = 0; i < no_full; i++) D1_HF[i*ne + i] = 2.0;
    std::vector<real_t> D2_HF(ne4, 0.0);
    for (int p = 0; p < ne; p++)
        for (int q = 0; q < ne; q++) {
            real_t d_pq = D1_HF[p*ne + q];
            for (int r = 0; r < ne; r++) {
                real_t d_rq = D1_HF[r*ne + q];
                for (int s = 0; s < ne; s++) {
                    real_t d_rs = D1_HF[r*ne + s];
                    real_t d_ps = D1_HF[p*ne + s];
                    D2_HF[(((size_t)p*ne+q)*ne+r)*ne+s] =
                        d_pq*d_rs - 0.5*d_ps*d_rq;
                }
            }
        }

    // 6. E_frag(D1, D2) — apply P to first index of (h, eri); contract rest.
    //    Returns (E1, E2) separately for diagnostics.
    auto eval_E_frag = [&](const std::vector<real_t>& D1,
                           const std::vector<real_t>& D2)
        -> std::pair<real_t, real_t> {
        real_t E1 = 0.0, E2 = 0.0;
        for (int i = 0; i < ne; i++)
            for (int ip = 0; ip < ne; ip++) {
                real_t Pi = P[i*ne + ip];
                if (std::abs(Pi) < 1e-15) continue;
                real_t hD = 0.0;
                for (int j = 0; j < ne; j++)
                    hD += h_can[i*ne + j] * D1[j*ne + ip];
                E1 += Pi * hD;
                real_t two_e = 0.0;
                for (int j = 0; j < ne; j++)
                    for (int k = 0; k < ne; k++)
                        for (int l = 0; l < ne; l++)
                            two_e += eri_can[(((size_t)i*ne+j)*ne+k)*ne+l]
                                   * D2[(((size_t)ip*ne+j)*ne+k)*ne+l];
                E2 += 0.5 * Pi * two_e;
            }
        return {E1, E2};
    };

    auto [E1_full, E2_full] = eval_E_frag(dm1_full, dm2_full);
    auto [E1_HF,  E2_HF ]  = eval_E_frag(D1_HF, D2_HF);
    real_t E1_corr = E1_full - E1_HF;
    real_t E2_corr = E2_full - E2_HF;
    if (E1_out) *E1_out = E1_corr;
    if (E2_out) *E2_out = E2_corr;
    return E1_corr + E2_corr;
}


// ============================================================================
//  Vayesta-convention DMET fragment correlation energy (standard QC-DMET).
//
//  e1 = Tr[P · h_avg · D1]   with  h_avg = ½(h_bare + h_eff)
//  e2 = ½ · Σ P[p,t] · eri[t,q,r,s] · D2[p,q,r,s]
//  E_corr = (e1 + e2) − (e1_HF + e2_HF)
//
//  h_bare = canonical-basis bare core (= U^T · C_emb^T · h_core · C_emb · U)
//  h_eff  = canonical-basis (F − cluster-internal V_HF), so on average the
//           1-body operator reduces to h_core (cluster Hartree-exchange
//           cancels). Validated against Vayesta on benzene/STO-3G.
// ============================================================================
real_t compute_dmet_fragment_energy_vayesta(
    int nocc_act, int nvir,
    const real_t* h_core_emb,
    const real_t* h_emb_1e,
    const real_t* eri_can,
    const real_t* dm1_active,
    const real_t* dm2_active,
    int n_frag,
    const real_t* eigvecs,
    int n_frozen,
    real_t* E1_out,
    real_t* E2_out)
{
    const int na_act = nocc_act + nvir;
    const int ne = n_frozen + na_act;
    const int no_full = n_frozen + nocc_act;
    const size_t ne2 = (size_t)ne * ne;
    const size_t ne4 = ne2 * ne2;

    // 1. Build full-ne dm1, dm2 (HF on frozen, CCSD active block)
    std::vector<real_t> dm1_full(ne2, 0.0);
    for (int i = 0; i < n_frozen; i++) dm1_full[i*ne + i] = 2.0;
    for (int i = 0; i < na_act; i++)
        for (int j = 0; j < na_act; j++)
            dm1_full[(i+n_frozen)*ne + (j+n_frozen)] = dm1_active[i*na_act + j];

    auto D1HF_full = [no_full](int p, int q) -> real_t {
        return (p == q && p < no_full) ? 2.0 : 0.0;
    };
    std::vector<real_t> dm2_full(ne4, 0.0);
    for (int p = 0; p < ne; p++)
        for (int q = 0; q < ne; q++) {
            real_t d_pq = D1HF_full(p,q);
            for (int r = 0; r < ne; r++) {
                real_t d_rq = D1HF_full(r,q);
                for (int s = 0; s < ne; s++) {
                    real_t d_rs = D1HF_full(r,s);
                    real_t d_ps = D1HF_full(p,s);
                    dm2_full[(((size_t)p*ne+q)*ne+r)*ne+s] =
                        d_pq*d_rs - 0.5*d_ps*d_rq;
                }
            }
        }
    for (int i = 0; i < na_act; i++)
        for (int j = 0; j < na_act; j++)
            for (int k = 0; k < na_act; k++)
                for (int l = 0; l < na_act; l++)
                    dm2_full[(((size_t)(i+n_frozen)*ne+(j+n_frozen))*ne+(k+n_frozen))*ne+(l+n_frozen)] =
                        dm2_active[(((size_t)i*na_act+j)*na_act+k)*na_act+l];

    // 2. Transform 1-body operators to canonical basis: h_can = U^T · h_emb · U
    auto transform_to_can = [&](const real_t* h_emb, std::vector<real_t>& h_can) {
        h_can.assign(ne2, 0.0);
        std::vector<real_t> tmp(ne2, 0.0);
        for (int i = 0; i < ne; i++)
            for (int q = 0; q < ne; q++) {
                real_t v = 0.0;
                for (int p = 0; p < ne; p++)
                    v += eigvecs[p*ne + i] * h_emb[p*ne + q];
                tmp[i*ne + q] = v;
            }
        for (int i = 0; i < ne; i++)
            for (int j = 0; j < ne; j++) {
                real_t v = 0.0;
                for (int q = 0; q < ne; q++)
                    v += tmp[i*ne + q] * eigvecs[q*ne + j];
                h_can[i*ne + j] = v;
            }
    };

    std::vector<real_t> h_bare_can, h_F_can;
    transform_to_can(h_core_emb, h_bare_can);
    transform_to_can(h_emb_1e,  h_F_can);

    // 3. v_act in canonical basis: 2·Σ_{k∈occ} eri[k,k,i,j] − Σ_{k∈occ} eri[k,j,i,k]
    //    (closed-shell 2J − K from cluster-occupied density)
    std::vector<real_t> v_act(ne2, 0.0);
    for (int i = 0; i < ne; i++)
        for (int j = 0; j < ne; j++) {
            real_t s = 0.0;
            for (int k = 0; k < no_full; k++) {
                s += 2.0 * eri_can[(((size_t)k*ne + k)*ne + i)*ne + j];
                s -=        eri_can[(((size_t)k*ne + j)*ne + i)*ne + k];
            }
            v_act[i*ne + j] = s;
        }

    // 4. h_avg = ½ · (h_bare + h_F − v_act)  =  ½ · (h_bare + h_eff)
    std::vector<real_t> h_avg(ne2, 0.0);
    for (size_t idx = 0; idx < ne2; idx++) {
        h_avg[idx] = 0.5 * (h_bare_can[idx] + h_F_can[idx] - v_act[idx]);
    }

    // 5. P[i,i'] = Σ_{p<n_frag} U[p,i] · U[p,i']  (fragment AO projector)
    std::vector<real_t> P(ne2, 0.0);
    for (int i = 0; i < ne; i++)
        for (int j = 0; j <= i; j++) {
            real_t v = 0.0;
            for (int p = 0; p < n_frag; p++)
                v += eigvecs[p*ne + i] * eigvecs[p*ne + j];
            P[i*ne + j] = v;
            P[j*ne + i] = v;
        }

    // 6. HF reference RDMs
    std::vector<real_t> D1_HF(ne2, 0.0);
    for (int i = 0; i < no_full; i++) D1_HF[i*ne + i] = 2.0;
    std::vector<real_t> D2_HF(ne4, 0.0);
    for (int p = 0; p < ne; p++)
        for (int q = 0; q < ne; q++) {
            real_t d_pq = D1_HF[p*ne + q];
            for (int r = 0; r < ne; r++) {
                real_t d_rq = D1_HF[r*ne + q];
                for (int s = 0; s < ne; s++) {
                    real_t d_rs = D1_HF[r*ne + s];
                    real_t d_ps = D1_HF[p*ne + s];
                    D2_HF[(((size_t)p*ne+q)*ne+r)*ne+s] =
                        d_pq*d_rs - 0.5*d_ps*d_rq;
                }
            }
        }

    // 7. e1 = Tr[P · h_avg · D1];  e2 = ½ · Σ_pqrs P[p,t] · eri[t,q,r,s] · D2[p,q,r,s]
    auto eval_E_frag = [&](const std::vector<real_t>& D1,
                           const std::vector<real_t>& D2)
        -> std::pair<real_t, real_t> {
        real_t E1 = 0.0, E2 = 0.0;
        for (int i = 0; i < ne; i++)
            for (int ip = 0; ip < ne; ip++) {
                real_t Pi = P[i*ne + ip];
                if (std::abs(Pi) < 1e-15) continue;
                real_t hD = 0.0;
                for (int j = 0; j < ne; j++)
                    hD += h_avg[i*ne + j] * D1[j*ne + ip];
                E1 += Pi * hD;
                real_t two_e = 0.0;
                for (int j = 0; j < ne; j++)
                    for (int k = 0; k < ne; k++)
                        for (int l = 0; l < ne; l++)
                            two_e += eri_can[(((size_t)i*ne+j)*ne+k)*ne+l]
                                   * D2[(((size_t)ip*ne+j)*ne+k)*ne+l];
                E2 += 0.5 * Pi * two_e;
            }
        return {E1, E2};
    };

    auto [E1_full, E2_full] = eval_E_frag(dm1_full, dm2_full);
    auto [E1_HF,  E2_HF ]  = eval_E_frag(D1_HF, D2_HF);
    real_t E1_corr = E1_full - E1_HF;
    real_t E2_corr = E2_full - E2_HF;
    if (E1_out) *E1_out = E1_corr;
    if (E2_out) *E2_out = E2_corr;
    return E1_corr + E2_corr;
}

} // namespace gansu
