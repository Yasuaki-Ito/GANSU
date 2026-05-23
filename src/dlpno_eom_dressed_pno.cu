/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/**
 * @file dlpno_eom_dressed_pno.cu
 * @brief Per-pair PNO-basis intermediates for the native DLPNO-EOM σ operators.
 *        B-a.6c (a): Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij). See the .hpp for the contract.
 */

#include "dlpno_eom_dressed_pno.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <Eigen/Dense>

#include "dlpno_mp2.hpp"        // DLPNOLMP2Result (setups / pairs / pair_lookup / phase24)
#include "dlpno_pair_data.hpp" // PairSetup / PairData / Phase24Integrals

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace

DressedPnoIP build_dressed_pno_ip(const std::vector<real_t>& h_Lvv,
                                  const std::vector<std::vector<real_t>>& Uall,
                                  const std::vector<int>& n_pno,
                                  int nvir)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    DressedPnoIP out;
    out.Lvv_pno.resize(n_pairs);

    Eigen::Map<const RowMatXd> Lvv(h_Lvv.data(), nvir, nvir);   // Lvv[a,d]

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        // U^(ij): [nvir × n_pno], flat row-major.
        Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
        // Lvv^(ij) = U^(ij)ᵀ Lvv U^(ij)   [n × n].
        const RowMatXd Lpno = U.transpose() * Lvv * U;
        out.Lvv_pno[idx].assign(static_cast<size_t>(n) * n, 0.0);
        for (int a = 0; a < n; ++a)
            for (int d = 0; d < n; ++d)
                out.Lvv_pno[idx][static_cast<size_t>(a) * n + d] = Lpno(a, d);
    }
    return out;
}

void build_dressed_pno_ip_phladder(DressedPnoIP& io,
                                   const std::vector<real_t>& h_Wovvo_lmo,
                                   const std::vector<real_t>& h_Wovov_lmo,
                                   const std::vector<std::vector<real_t>>& Uall,
                                   const std::vector<int>& pair_i,
                                   const std::vector<int>& pair_j,
                                   const std::vector<int>& n_pno,
                                   int nocc, int nvir)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    io.Wovvo_pno_occi.assign(n_pairs, {});
    io.Wovvo_pno_occj.assign(n_pairs, {});
    io.Wovov_pno_occi.assign(n_pairs, {});
    io.Wovov_pno_occj.assign(n_pairs, {});

    const size_t nv = static_cast<size_t>(nvir);

    // For one fixed occupied I and one LMO m, gather the dense [nvir × nvir]
    // virtual block, congruence it U^(ij)ᵀ M U^(ij) → [n × n], scatter to
    // out[(m·n + a')·n + d']. `ovvo` selects the layout (Wovvo vs Wovov).
    auto build_one = [&](std::vector<std::vector<real_t>>& dst, int idx, int n,
                         const Eigen::Map<const RowMatXd>& U, int I, bool ovvo) {
        dst[idx].assign(static_cast<size_t>(nocc) * n * n, 0.0);
        RowMatXd M(nvir, nvir);
        for (int m = 0; m < nocc; ++m) {
            for (int a = 0; a < nvir; ++a)
                for (int d = 0; d < nvir; ++d) {
                    // Wovvo layout ((m·nvir+a)·nvir+d)·nocc + I
                    // Wovov layout ((m·nvir+a)·nocc+I)·nvir + d
                    const size_t off = ovvo
                        ? (((static_cast<size_t>(m) * nv + a) * nv + d) * nocc + I)
                        : (((static_cast<size_t>(m) * nv + a) * nocc + I) * nv + d);
                    M(a, d) = (ovvo ? h_Wovvo_lmo : h_Wovov_lmo)[off];
                }
            const RowMatXd C = U.transpose() * M * U;        // [n × n]
            const size_t base = static_cast<size_t>(m) * n * n;
            for (int ap = 0; ap < n; ++ap)
                for (int dp = 0; dp < n; ++dp)
                    dst[idx][base + static_cast<size_t>(ap) * n + dp] = C(ap, dp);
        }
    };

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
        const int i = pair_i[idx];
        const int j = pair_j[idx];
        build_one(io.Wovvo_pno_occi, idx, n, U, i, /*ovvo=*/true);
        build_one(io.Wovvo_pno_occj, idx, n, U, j, /*ovvo=*/true);
        build_one(io.Wovov_pno_occi, idx, n, U, i, /*ovvo=*/false);
        build_one(io.Wovov_pno_occj, idx, n, U, j, /*ovvo=*/false);
    }
}

// ---------------------------------------------------------------------------
//  B-a.6c IP dense-free bare seed (Phase24 W_ovvo_bare / W_oovv_bare → the per-
//  pair PNO Wovvo_pno/Wovov_pno bare, NO dense nocc²·nvir²). EA B-EA.6e analog.
// ---------------------------------------------------------------------------
void build_dressed_pno_ip_bare(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<int>& n_pno,
                               int nocc)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    const Phase24Integrals& ph = res.phase24;
    io.Wovvo_pno_occi.assign(n_pairs, {});
    io.Wovvo_pno_occj.assign(n_pairs, {});
    io.Wovov_pno_occi.assign(n_pairs, {});
    io.Wovov_pno_occj.assign(n_pairs, {});

    // Copy one Phase24 bare block (layout (m·n+a')·n+d', size nocc·n²) into the
    // matching native Wovvo_pno/Wovov_pno slot, or zero it if Phase24 lacks it.
    auto seed = [&](std::vector<std::vector<real_t>>& dst, int idx, size_t need,
                    const std::vector<std::vector<real_t>>& src) {
        if (idx < static_cast<int>(src.size()) && src[idx].size() == need)
            dst[idx] = src[idx];                 // direct copy (same layout)
        else
            dst[idx].assign(need, 0.0);          // missing block → bare = 0 (ring still added)
    };

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        const size_t need = static_cast<size_t>(nocc) * n * n;
        seed(io.Wovvo_pno_occi, idx, need, ph.W_ovvo_bare_i);
        seed(io.Wovvo_pno_occj, idx, need, ph.W_ovvo_bare_j);
        seed(io.Wovov_pno_occi, idx, need, ph.W_oovv_bare_i);
        seed(io.Wovov_pno_occj, idx, need, ph.W_oovv_bare_j);
    }
}

// ---------------------------------------------------------------------------
//  B-a.6c(b2): native ring build (Phase24 + two-sided barS) − dense-ring
//  congruence subtraction. See the .hpp for the contract and index derivation.
// ---------------------------------------------------------------------------
void build_dressed_pno_ip_ring(DressedPnoIP& io,
                               const DLPNOLMP2Result& res,
                               const std::vector<real_t>& h_ovov_lmo,
                               const std::vector<real_t>& h_t2_lmo,
                               const std::vector<std::vector<real_t>>& Uall,
                               const std::vector<real_t>& h_S,
                               const std::vector<int>& n_pno,
                               int nao, int nocc, int nvir,
                               real_t& max_delta, real_t& max_ring,
                               bool subtract_dense)
{
    const int n_pairs = static_cast<int>(n_pno.size());
    const size_t NV = static_cast<size_t>(nvir);
    const size_t NO = static_cast<size_t>(nocc);
    const Phase24Integrals& ph = res.phase24;
    max_delta = 0.0;
    max_ring  = 0.0;

    // ovov(p,a,q,b) = (pa|qb), layout ((p·nvir+a)·nocc+q)·nvir+b.
    auto OVOV = [&](int p, int a, int q, int b) -> real_t {
        return h_ovov_lmo[(((static_cast<size_t>(p) * NV + a) * NO + q) * NV + b)];
    };
    // t2(p,q,a,b), layout ((p·nocc+q)·nvir+a)·nvir+b.
    auto T2 = [&](int p, int q, int a, int b) -> real_t {
        return h_t2_lmo[(((static_cast<size_t>(p) * NO + q) * NV + a) * NV + b)];
    };

    // --- Dense ring per fixed occupied I (canonical virtual basis), occ→LMO. ---
    // DRvvo[I][m,a,c] = Σ_{l,e}[ 2 ovov(m,c,l,e) t2(I,l,a,e)
    //                          -   ovov(m,c,l,e) t2(l,I,a,e)
    //                          -   ovov(m,e,l,c) t2(I,l,a,e) ]   (Wovvo ring; a=res, c=contr)
    // DRvov[I][m,b,e] = -Σ_{c,l} ovov(m,c,l,e) t2(I,l,c,b)        (Wovov ring; b=res, e=contr)
    // Only for the B-a.6c(b2) validate gate (subtract_dense); the IP dense-free
    // path (subtract_dense=false) NEVER materialises these nocc²·nvir² buffers.
    const size_t dr_per_I = NO * NV * NV;                     // [m,a,c]
    std::vector<std::vector<real_t>> DRvvo(subtract_dense ? nocc : 0),
                                     DRvov(subtract_dense ? nocc : 0);
    if (subtract_dense)
    for (int I = 0; I < nocc; ++I) {
        DRvvo[I].assign(dr_per_I, 0.0);
        DRvov[I].assign(dr_per_I, 0.0);
        for (int m = 0; m < nocc; ++m)
            for (int a = 0; a < nvir; ++a)
                for (int c = 0; c < nvir; ++c) {
                    real_t svvo = 0.0, svov = 0.0;
                    for (int l = 0; l < nocc; ++l)
                        for (int e = 0; e < nvir; ++e) {
                            const real_t mcle = OVOV(m, c, l, e);   // (mc|le)
                            const real_t mecl = OVOV(m, e, l, c);   // (me|lc)
                            svvo += 2.0 * mcle * T2(I, l, a, e)
                                  -       mcle * T2(l, I, a, e)
                                  -       mecl * T2(I, l, a, e);
                            // Wovov ring DR[I][m,a(result),c(contracted)]
                            //   = -Σ_{e,l} ovov(m,e,l,c)·t2(I,l,e,a)   (e = internal vir)
                            svov += -mecl * T2(I, l, e, a);
                        }
                    DRvvo[I][(static_cast<size_t>(m) * NV + a) * NV + c] = svvo;
                    DRvov[I][(static_cast<size_t>(m) * NV + a) * NV + c] = svov;
                }
    }

    Eigen::Map<const RowMatXd> S(h_S.data(), nao, nao);

    // Apply the (native_ring − congruence(dense_ring)) update for one occ role.
    //   dst[idx][(m·n+ap)·n+dp]   ap = result PNO, dp = contracted PNO.
    auto apply_role = [&](int idx, int I, int n,
                          const Eigen::Map<const RowMatXd>& U,
                          const RowMatXd& Bij,            // bar_Q_ijᵀ · S  [n × nao]
                          std::vector<std::vector<real_t>>& dst_vvo,
                          std::vector<std::vector<real_t>>& dst_vov) {
        // V[A,B,P,Q]=(AP|BQ), layout ((A·nocc+B)·n+P)·n+Q. No Phase24 ovov for this
        // pair → leave the b1 congruence seed (incl. its dense ring) intact rather
        // than dropping the ring entirely.
        const bool have_V = (static_cast<int>(idx) < static_cast<int>(ph.V_ovov_pair.size())) &&
                            (static_cast<int>(ph.V_ovov_pair[idx].size()) ==
                             static_cast<int>(NO * NO * static_cast<size_t>(n) * n));
        if (!have_V) return;
        const std::vector<real_t>& V = ph.V_ovov_pair[idx];

        // t2_proj[I,l] in target PNO [n×n] for every l (t2_proj[l,I] = transpose).
        std::vector<RowMatXd> t2Il(nocc, RowMatXd::Zero(n, n));
        for (int l = 0; l < nocc; ++l) {
            const int src = res.pair_lookup[static_cast<size_t>(I) * nocc + l];
            if (src < 0) continue;
            const int ns = n_pno[src];
            if (ns == 0) continue;
            if (static_cast<int>(res.pairs[src].Y.size()) != ns * ns) continue;
            Eigen::Map<const RowMatXd> barQ_src(res.pairs[src].bar_Q.data(), nao, ns);
            const RowMatXd barS = Bij * barQ_src;                      // [n × ns]
            Eigen::Map<const RowMatXd> Ysrc(res.pairs[src].Y.data(), ns, ns);
            const RowMatXd M = barS * Ysrc * barS.transpose();          // [n × n], ordering (src.i,src.j)
            if (res.setups[src].i == I) t2Il[l] = M;                     // occ order (I,l) == storage
            else                        t2Il[l] = M.transpose();          // (l,I) → transpose
        }

        RowMatXd nat_vvo(n, n), nat_vov(n, n);
        for (int m = 0; m < nocc; ++m) {
            // native ring (PNO space) for this (idx,I,m).
            nat_vvo.setZero();
            nat_vov.setZero();
            {
                const size_t mV = static_cast<size_t>(m) * NO;        // (m·nocc + l) row base
                for (int l = 0; l < nocc; ++l) {
                    const RowMatXd& T = t2Il[l];                       // T(ap,e) = t2_proj[I,l]
                    const size_t vbase = (mV + l) * static_cast<size_t>(n) * n;  // V[m,l,·,·]
                    for (int ap = 0; ap < n; ++ap)
                        for (int dp = 0; dp < n; ++dp) {
                            real_t svvo = 0.0, svov = 0.0;
                            for (int e = 0; e < n; ++e) {
                                const real_t V1 = V[vbase + static_cast<size_t>(dp) * n + e]; // (m dp|l e)
                                const real_t V3 = V[vbase + static_cast<size_t>(e) * n + dp]; // (m e|l dp)
                                svvo += 2.0 * V1 * T(ap, e) - V1 * T(e, ap) - V3 * T(ap, e);
                                // Wovov: -Σ_{c'} (m c'|l dp)·t2_proj[I,l](c',ap), c'≡e → (m e|l dp)=V3.
                                svov += -V3 * T(e, ap);
                            }
                            nat_vvo(ap, dp) += svvo;   // accumulate over l (zeroed per m)
                            nat_vov(ap, dp) += svov;
                        }
                }
            }
            const size_t base = static_cast<size_t>(m) * n * n;
            if (!subtract_dense) {
                // IP dense-free: add native ring alone on top of the Phase24 bare
                // seed (no dense DR built / no congruence subtraction).
                for (int ap = 0; ap < n; ++ap)
                    for (int dp = 0; dp < n; ++dp) {
                        const real_t vvo = nat_vvo(ap, dp), vov = nat_vov(ap, dp);
                        dst_vvo[idx][base + static_cast<size_t>(ap) * n + dp] += vvo;
                        dst_vov[idx][base + static_cast<size_t>(ap) * n + dp] += vov;
                        max_ring = std::max(max_ring, std::max(std::fabs(vvo), std::fabs(vov)));
                    }
                continue;
            }
            // B-a.6c(b2) validate: Wovvo_pno += native_ring − congruence(dense_ring).
            // congruence of the dense ring: U^(ij)ᵀ · DR[I][m] · U^(ij)  [n×n].
            Eigen::Map<const RowMatXd> DRvm(DRvvo[I].data() + static_cast<size_t>(m) * NV * NV, nvir, nvir);
            Eigen::Map<const RowMatXd> DRom(DRvov[I].data() + static_cast<size_t>(m) * NV * NV, nvir, nvir);
            const RowMatXd cong_vvo = U.transpose() * DRvm * U;        // [n×n]
            const RowMatXd cong_vov = U.transpose() * DRom * U;

            for (int ap = 0; ap < n; ++ap)
                for (int dp = 0; dp < n; ++dp) {
                    const real_t dvvo = nat_vvo(ap, dp) - cong_vvo(ap, dp);
                    const real_t dvov = nat_vov(ap, dp) - cong_vov(ap, dp);
                    dst_vvo[idx][base + static_cast<size_t>(ap) * n + dp] += dvvo;
                    dst_vov[idx][base + static_cast<size_t>(ap) * n + dp] += dvov;
                    max_delta = std::max(max_delta, std::max(std::fabs(dvvo), std::fabs(dvov)));
                    max_ring  = std::max(max_ring,  std::max(std::fabs(cong_vvo(ap, dp)),
                                                             std::fabs(cong_vov(ap, dp))));
                }
        }
    };

    for (int idx = 0; idx < n_pairs; ++idx) {
        const int n = n_pno[idx];
        if (n == 0) continue;
        Eigen::Map<const RowMatXd> U(Uall[idx].data(), nvir, n);
        Eigen::Map<const RowMatXd> barQ_ij(res.pairs[idx].bar_Q.data(), nao, n);
        const RowMatXd Bij = barQ_ij.transpose() * S;                 // [n × nao]
        const int i = res.setups[idx].i;
        const int j = res.setups[idx].j;
        apply_role(idx, i, n, U, Bij, io.Wovvo_pno_occi, io.Wovov_pno_occi);
        apply_role(idx, j, n, U, Bij, io.Wovvo_pno_occj, io.Wovov_pno_occj);
    }
}

// ===========================================================================
//  EA mirror (B-EA.6d): per-pair PNO Wvvvv.
// ===========================================================================

namespace {
// 4-index congruence W_pno[a'b'c'd'] = Σ_{abcd} U[a,a']U[b,b']U[c,c']U[d,d'] W[abcd],
// as four sequential single-index transforms. Uflat is row-major [nv × n];
// W and out are row-major (((·)*…)). Runs once in the ctor — gate scale.
void congruence4(const std::vector<real_t>& W,
                 const std::vector<real_t>& Uflat, int nv, int n,
                 std::vector<real_t>& out) {
    auto Uat = [&](int a, int ap) -> real_t { return Uflat[static_cast<size_t>(a) * n + ap]; };
    const size_t NV = static_cast<size_t>(nv);
    // step 1: a→a'   A[a',b,c,d]   (dims n·nv·nv·nv)
    std::vector<real_t> A(static_cast<size_t>(n) * NV * NV * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int b = 0; b < nv; ++b)
            for (int c = 0; c < nv; ++c)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int a = 0; a < nv; ++a)
                        s += Uat(a, ap) * W[(((static_cast<size_t>(a) * NV + b) * NV + c) * NV + d)];
                    A[(((static_cast<size_t>(ap) * NV + b) * NV + c) * NV + d)] = s;
                }
    // step 2: b→b'   B[a',b',c,d]   (dims n·n·nv·nv)
    std::vector<real_t> B(static_cast<size_t>(n) * n * NV * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int c = 0; c < nv; ++c)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int b = 0; b < nv; ++b)
                        s += Uat(b, bp) * A[(((static_cast<size_t>(ap) * NV + b) * NV + c) * NV + d)];
                    B[(((static_cast<size_t>(ap) * n + bp) * NV + c) * NV + d)] = s;
                }
    // step 3: c→c'   C[a',b',c',d]  (dims n·n·n·nv)
    std::vector<real_t> C(static_cast<size_t>(n) * n * n * NV, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int cp = 0; cp < n; ++cp)
                for (int d = 0; d < nv; ++d) {
                    real_t s = 0.0;
                    for (int c = 0; c < nv; ++c)
                        s += Uat(c, cp) * B[(((static_cast<size_t>(ap) * n + bp) * NV + c) * NV + d)];
                    C[(((static_cast<size_t>(ap) * n + bp) * n + cp) * NV + d)] = s;
                }
    // step 4: d→d'   out[a',b',c',d']  (dims n⁴)
    out.assign(static_cast<size_t>(n) * n * n * n, 0.0);
    #pragma omp parallel for
    for (int ap = 0; ap < n; ++ap)
        for (int bp = 0; bp < n; ++bp)
            for (int cp = 0; cp < n; ++cp)
                for (int dp = 0; dp < n; ++dp) {
                    real_t s = 0.0;
                    for (int d = 0; d < nv; ++d)
                        s += Uat(d, dp) * C[(((static_cast<size_t>(ap) * n + bp) * n + cp) * NV + d)];
                    out[(((static_cast<size_t>(ap) * n + bp) * n + cp) * n + dp)] = s;
                }
}
}  // namespace

DressedPnoEA build_dressed_pno_ea_vvvv(const std::vector<real_t>& h_Wvvvv,
                                       const std::vector<std::vector<real_t>>& Uocc,
                                       const std::vector<int>& n_pno_ii,
                                       int nvir)
{
    const int nocc = static_cast<int>(n_pno_ii.size());
    DressedPnoEA out;
    out.Wvvvv_pno.assign(nocc, {});
    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        congruence4(h_Wvvvv, Uocc[j], nvir, n, out.Wvvvv_pno[j]);
    }
    return out;
}

DressedPnoEA build_dressed_pno_ea_vvvv_bare(const DLPNOLMP2Result& res,
                                            const std::vector<int>& n_pno_ii)
{
    const int nocc = static_cast<int>(n_pno_ii.size());
    const Phase24Integrals& ph = res.phase24;
    DressedPnoEA out;
    out.Wvvvv_pno.assign(nocc, {});
    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        const int idx_jj = res.pair_lookup[static_cast<size_t>(j) * nocc + j];
        if (idx_jj < 0) continue;
        const size_t n4 = static_cast<size_t>(n) * n * n * n;
        // W_pair^(jj)[a,b,c,d] = (ac|bd) in PNO(jj), layout ((a·n+b)·n+c)·n+d —
        // the bare Wvvvv contribution, with NO dense nvir⁴ materialisation.
        if (idx_jj < static_cast<int>(ph.W_pair.size()) &&
            ph.W_pair[idx_jj].size() == n4) {
            out.Wvvvv_pno[j] = ph.W_pair[idx_jj];   // direct copy (same [a,b,c,d] layout)
        } else {
            out.Wvvvv_pno[j].assign(n4, 0.0);       // missing block → bare = 0 (ring still added)
        }
    }
    return out;
}

void build_dressed_pno_ea_vvvv_ring(DressedPnoEA& io,
                                    const DLPNOLMP2Result& res,
                                    const std::vector<real_t>& h_ovov_lmo,
                                    const std::vector<real_t>& h_t2_lmo,
                                    const std::vector<std::vector<real_t>>& Uocc,
                                    const std::vector<real_t>& h_S,
                                    const std::vector<int>& n_pno_ii,
                                    int nao, int nocc, int nvir,
                                    real_t& max_delta, real_t& max_ring,
                                    bool subtract_dense)
{
    const size_t NV = static_cast<size_t>(nvir);
    const size_t NO = static_cast<size_t>(nocc);
    const Phase24Integrals& ph = res.phase24;
    max_delta = 0.0;
    max_ring  = 0.0;

    // ovov(k,c,l,d) = (kc|ld), layout ((k·nvir+c)·nocc+l)·nvir+d.
    auto OVOV = [&](int k, int c, int l, int d) -> real_t {
        return h_ovov_lmo[(((static_cast<size_t>(k) * NV + c) * NO + l) * NV + d)];
    };
    // t2(k,l,a,b), layout ((k·nocc+l)·nvir+a)·nvir+b.
    auto T2 = [&](int k, int l, int a, int b) -> real_t {
        return h_t2_lmo[(((static_cast<size_t>(k) * NO + l) * NV + a) * NV + b)];
    };

    // --- Dense Wvvvv ring (canonical virtual basis), built once ---
    //   DR[a,b,c,d] = Σ_{k,l} (kc|ld) t2(k,l,a,b).
    // Only for the B-EA.6d validate gate (subtract_dense); the true-scaling
    // path (subtract_dense=false) NEVER materialises this nvir⁴ buffer.
    std::vector<real_t> DR;
    if (subtract_dense) {
        DR.assign(NV * NV * NV * NV, 0.0);
        #pragma omp parallel for
        for (int a = 0; a < nvir; ++a)
            for (int b = 0; b < nvir; ++b)
                for (int c = 0; c < nvir; ++c)
                    for (int d = 0; d < nvir; ++d) {
                        real_t s = 0.0;
                        for (int k = 0; k < nocc; ++k)
                            for (int l = 0; l < nocc; ++l)
                                s += OVOV(k, c, l, d) * T2(k, l, a, b);
                        DR[(((static_cast<size_t>(a) * NV + b) * NV + c) * NV + d)] = s;
                    }
    }

    Eigen::Map<const RowMatXd> S(h_S.data(), nao, nao);

    for (int j = 0; j < nocc; ++j) {
        const int n = n_pno_ii[j];
        if (n == 0) continue;
        const int idx_jj = res.pair_lookup[static_cast<size_t>(j) * nocc + j];
        if (idx_jj < 0) continue;
        // V_ovov_pair^(jj)[k,l,c',d'] = (k c'|l d'), layout ((k·nocc+l)·n+c')·n+d'.
        // No Phase24 V for this pair → leave the b1 congruence seed intact.
        const bool have_V = (idx_jj < static_cast<int>(ph.V_ovov_pair.size())) &&
                            (static_cast<int>(ph.V_ovov_pair[idx_jj].size()) ==
                             static_cast<int>(NO * NO * static_cast<size_t>(n) * n));
        if (!have_V) continue;
        const std::vector<real_t>& V = ph.V_ovov_pair[idx_jj];

        Eigen::Map<const RowMatXd> barQ_jj(res.pairs[idx_jj].bar_Q.data(), nao, n);
        const RowMatXd Bjj = barQ_jj.transpose() * S;        // [n × nao]

        // t2_proj^(jj←kl)[a',b'] = oriented(barS Y_src barSᵀ) [n×n] for every (k,l).
        std::vector<RowMatXd> t2kl(static_cast<size_t>(nocc) * nocc, RowMatXd::Zero(n, n));
        for (int k = 0; k < nocc; ++k)
            for (int l = 0; l < nocc; ++l) {
                const int src = res.pair_lookup[static_cast<size_t>(k) * nocc + l];
                if (src < 0) continue;
                const int ns = res.pairs[src].n_pno;
                if (ns == 0) continue;
                if (static_cast<int>(res.pairs[src].Y.size()) != ns * ns) continue;
                Eigen::Map<const RowMatXd> barQ_src(res.pairs[src].bar_Q.data(), nao, ns);
                const RowMatXd barS = Bjj * barQ_src;                       // [n × ns]
                Eigen::Map<const RowMatXd> Ysrc(res.pairs[src].Y.data(), ns, ns);
                const RowMatXd M = barS * Ysrc * barS.transpose();           // [n × n], order (src.i,src.j)
                t2kl[static_cast<size_t>(k) * nocc + l] =
                    (res.setups[src].i == k) ? M : RowMatXd(M.transpose());   // (k,l) order
            }

        // native_ring[a',b',c',d'] = Σ_{k,l} V[k,l,c',d'] · t2_proj[k,l][a',b'].
        std::vector<real_t> nat(static_cast<size_t>(n) * n * n * n, 0.0);
        for (int k = 0; k < nocc; ++k)
            for (int l = 0; l < nocc; ++l) {
                const RowMatXd& T = t2kl[static_cast<size_t>(k) * nocc + l];
                const size_t vbase = (static_cast<size_t>(k) * nocc + l) * n * n;  // V[k,l,·,·]
                #pragma omp parallel for
                for (int ap = 0; ap < n; ++ap)
                    for (int bp = 0; bp < n; ++bp) {
                        const real_t Tab = T(ap, bp);
                        if (Tab == 0.0) continue;
                        real_t* row = nat.data() + (((static_cast<size_t>(ap) * n + bp) * n) * n);
                        for (int cp = 0; cp < n; ++cp)
                            for (int dp = 0; dp < n; ++dp)
                                row[static_cast<size_t>(cp) * n + dp] +=
                                    Tab * V[vbase + static_cast<size_t>(cp) * n + dp];
                    }
            }

        std::vector<real_t>& W = io.Wvvvv_pno[j];
        const size_t n4 = static_cast<size_t>(n) * n * n * n;
        if (subtract_dense) {
            // B-EA.6d validate: Wvvvv_pno += native_ring − congruence(dense_ring).
            std::vector<real_t> cong;
            congruence4(DR, Uocc[j], nvir, n, cong);
            for (size_t e = 0; e < n4; ++e) {
                const real_t delta = nat[e] - cong[e];
                W[e] += delta;
                max_delta = std::max(max_delta, std::fabs(delta));
                max_ring  = std::max(max_ring,  std::fabs(cong[e]));
            }
        } else {
            // B-EA.6e true scaling: Wvvvv_pno += native_ring (no dense reference).
            for (size_t e = 0; e < n4; ++e) {
                W[e] += nat[e];
                max_ring = std::max(max_ring, std::fabs(nat[e]));
            }
        }
    }
}

} // namespace gansu
