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
 * CPU reference implementation. Follows PySCF's rccsd_lambda.py and ccsd_rdm.py
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

// ovoo[i,a,j,k] = (i, nocc+a | j, k)  — PySCF convention
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
//  Lambda equations — direct translation of PySCF rccsd_lambda.py
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
                        real_t* l1_new, real_t* l2_new)
{
    const int no = d.no, nv = d.nv;
    const size_t l1_sz = (size_t)no * nv;
    const size_t l2_sz = (size_t)no * no * nv * nv;

    // ----- Index lambdas (PySCF chemist notation, row-major) -----
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
    // ovoo1[k,b,i,j] = 2*ovoo[k,b,i,j] - ovoo[i,b,k,j]   (PySCF: ovoo*2 - ovoo.T(2,1,0,3))
    auto OVOO1 = [&](int k, int b, int i, int j) {
        return 2.0 * OVOO(k,b,i,j) - OVOO(i,b,k,j); };
    // ovvv1[i,a,b,c] = 2*ovvv[i,a,b,c] - ovvv[i,c,b,a]
    auto OVVV1 = [&](int i, int a, int b, int c) {
        return 2.0 * OVVV(i,a,b,c) - OVVV(i,c,b,a); };

    // =================================================================
    //  make_intermediates
    // =================================================================
    std::vector<real_t> v1((size_t)nv*nv, 0.0);   // [a,b] indexed v1[b,a] (PySCF) → store as v1[b*nv+a]? PySCF: einsum('jakc,jkbc->ba')
                                                  //  We'll store v1[b,a] = v1[(size_t)b*nv+a].
    std::vector<real_t> v2((size_t)no*no, 0.0);   // [i,j]
    std::vector<real_t> v4((size_t)no*nv, 0.0);   // [j,b]
    std::vector<real_t> v5((size_t)nv*no, 0.0);   // [b,j]
    std::vector<real_t> w3((size_t)nv*no, 0.0);   // [c,k]

    // v1[b,a] = fvv[b,a] - Σ_{j,k,c} ovov1[j,a,k,c]*tau[j,k,b,c]
    //   For canonical: fvv[b,a] = δ_ba * ε_{nocc+a}
    for (int b = 0; b < nv; b++)
      for (int a = 0; a < nv; a++) {
        real_t v = (a == b ? eps[no + a] : 0.0);
        for (int j = 0; j < no; j++)
          for (int k = 0; k < no; k++)
            for (int c = 0; c < nv; c++)
              v -= OVOV1(j,a,k,c) * TAU(j,k,b,c);
        v1[(size_t)b*nv + a] = v;
      }

    // v2[i,j] = foo[i,j] + Σ_{b,k,c} ovov1[i,b,k,c]*tau[j,k,b,c] + Σ_{k,b} ovoo1[k,b,i,j]*t1[k,b]
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
        v2[(size_t)i*no + j] = v;
      }

    // v4[j,b] = fov + Σ_{k,c} ovov1[j,b,k,c]*t1[k,c]   (fov=0 for canonical)
    for (int j = 0; j < no; j++)
      for (int b = 0; b < nv; b++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            v += OVOV1(j,b,k,c) * T1(k,c);
        v4[(size_t)j*nv + b] = v;
      }
    auto V4 = [&](int j, int b) { return v4[(size_t)j*nv + b]; };

    // v5[b,j] = +fvo (=0)
    //   + Σ_{k,c} v4[k,c]*t1[k,b]*t1[j,c]
    //   - Σ_{l,c,k} ovoo1[l,c,k,j]*t2[k,l,b,c]
    for (int b = 0; b < nv; b++)
      for (int j = 0; j < no; j++) {
        real_t v = 0.0;
        for (int k = 0; k < no; k++)
          for (int c = 0; c < nv; c++)
            v += V4(k,c) * T1(k,b) * T1(j,c);
        for (int l = 0; l < no; l++)
          for (int c = 0; c < nv; c++)
            for (int k = 0; k < no; k++)
              v -= OVOO1(l,c,k,j) * T2(k,l,b,c);
        v5[(size_t)b*no + j] = v;
      }

    // woooo[i,k,j,l]:  PySCF einsum order:
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
    // v4oVVo[j,b,c,k] = Σ_{l,d} ovov[j,d,l,b]*t2[k,l,d,c] - oovv[j,c,b,k] (PySCF: oovv.T(0,3,2,1))
    //   PySCF: v4oVVo -= eris.oovv.transpose(0,3,2,1) → v4oVVo[j,b,c,k] -= oovv[j,k,c,b]
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
    // From PySCF:
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
        //   ovvv1[i,d,c,b] = 2*ovvv[i,d,c,b] - ovvv[i,b,c,d]   (PySCF reassigns ovvv→ovvv1 before this step)
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
        // PySCF:
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
    // (PySCF: v1 += einsum('jcba,jc->ba', ovvv1, t1))
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
    // mvv[a,b] = Σ_{k,l,c} l2[k,l,c,b]*theta[k,l,c,a]   (PySCF einsum 'klca,klcb->ba')
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
    // first term: l2.vvvv  (note PySCF uses .conj() but real → no-op)
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

    // l2new += einsum('ijac,cb->ijab', l2, v1)   v1 stored [b,a]; PySCF v1[c,b]; here v1[b,a] indexed v1[b*nv+a]
    //   So 'cb' here means PySCF's v1[c,b] = our v1[(c,b)] = v1[c*nv + b].
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

    // l1new += fov (=0)
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
    int verbose)
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

    // Initial guess: Λ = 0 (simplest, same as PySCF default)
    std::fill(h_lambda1, h_lambda1 + l1_sz, 0.0);
    std::fill(h_lambda2, h_lambda2 + l2_sz, 0.0);

    std::vector<real_t> l1_new(l1_sz), l2_new(l2_sz);

    if (verbose > 0) {
        std::cout << "CCSD Lambda solver: nocc=" << nocc << " nvir=" << nvir
                  << " max_iter=" << max_iter << " tol=" << std::scientific
                  << tol << std::defaultfloat << std::endl;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        update_lambda_full(d, h_t1, h_t2, h_lambda1, h_lambda2,
                           h_eps,
                           ovov, ovoo, ovvv, oovv, ovvo, oooo, vvvv,
                           l1_new.data(), l2_new.data());

        // Residual = ||λ_new - λ_old||
        real_t r1 = 0.0, r2 = 0.0;
        for (size_t k = 0; k < l1_sz; k++) r1 += (l1_new[k] - h_lambda1[k]) * (l1_new[k] - h_lambda1[k]);
        for (size_t k = 0; k < l2_sz; k++) r2 += (l2_new[k] - h_lambda2[k]) * (l2_new[k] - h_lambda2[k]);
        real_t resid = std::sqrt(r1 + r2);

        std::copy(l1_new.begin(), l1_new.end(), h_lambda1);
        std::copy(l2_new.begin(), l2_new.end(), h_lambda2);

        if (verbose > 0) {
            std::cout << "  Lambda iter " << std::setw(3) << (iter + 1)
                      << ": ||Δλ|| = " << std::scientific << std::setprecision(3)
                      << resid << std::defaultfloat << std::endl;
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
                  << " iterations" << std::endl;
    }
    return false;
}

// ---------------------------------------------------------------------------
//  Public: build_ccsd_1rdm_mo_cpu
//
//  Following PySCF ccsd_rdm._gamma1_intermediates (spin-traced RHF 1-RDM).
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

} // namespace gansu
