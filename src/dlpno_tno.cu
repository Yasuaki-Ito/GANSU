/*
 * GANSU: GPU Accelerated Numerical Simulation Utility
 *
 * Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
 * All rights reserved.
 *
 * This software is licensed under the BSD 3-Clause License.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "dlpno_tno.hpp"

#include <Eigen/Dense>
#include <stdexcept>

namespace gansu {

namespace {
using RowMatXd = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
} // namespace

TNOBuilder::TNOBuilder(const std::vector<PairData>& pairs,
                       const real_t* F_AO,
                       const real_t* S_AO,
                       int nao,
                       real_t tol_lin_dep,
                       real_t t_cut_tno)
    : pairs_(pairs), F_AO_(F_AO), S_AO_(S_AO), nao_(nao),
      tol_lin_dep_(tol_lin_dep), t_cut_tno_(t_cut_tno) {
    if (F_AO == nullptr || S_AO == nullptr)
        throw std::invalid_argument("TNOBuilder: F_AO/S_AO null");
    if (nao <= 0)
        throw std::invalid_argument("TNOBuilder: nao must be positive");
}

TNOData TNOBuilder::build_for_triple(int idx_ij,
                                     int idx_ik,
                                     int idx_jk) const {
    TNOData out;

    const PairData& p_ij = pairs_[idx_ij];
    const PairData& p_ik = pairs_[idx_ik];
    const PairData& p_jk = pairs_[idx_jk];

    const int n_ij = p_ij.n_pno;
    const int n_ik = p_ik.n_pno;
    const int n_jk = p_jk.n_pno;
    if (n_ij == 0 || n_ik == 0 || n_jk == 0) {
        return out; // empty
    }
    const int m = n_ij + n_ik + n_jk; // raw union dimension

    Eigen::Map<const RowMatXd> S_map(S_AO_, nao_, nao_);
    Eigen::Map<const RowMatXd> F_map(F_AO_, nao_, nao_);

    // Step 1: stack the three pair PNO blocks into Q_union (nao × m).
    RowMatXd Q_union(nao_, m);
    {
        Eigen::Map<const RowMatXd> bQ_ij(p_ij.bar_Q.data(), nao_, n_ij);
        Eigen::Map<const RowMatXd> bQ_ik(p_ik.bar_Q.data(), nao_, n_ik);
        Eigen::Map<const RowMatXd> bQ_jk(p_jk.bar_Q.data(), nao_, n_jk);
        Q_union.leftCols(n_ij)                 = bQ_ij;
        Q_union.middleCols(n_ij, n_ik)         = bQ_ik;
        Q_union.rightCols(n_jk)                = bQ_jk;
    }

    // Step 2: union overlap S_u = Q_union^T · S_AO · Q_union (m × m).
    // Each pair's bar_Q is individually S-orthonormal so the diagonal blocks
    // of S_u are the identity; off-diagonal blocks measure redundancy
    // between PNOs from different pairs.
    const RowMatXd S_QU = S_map * Q_union;            // (nao × m)
    RowMatXd S_u = Q_union.transpose() * S_QU;        // (m × m)
    S_u = 0.5 * (S_u + S_u.transpose());

    // Step 3: eigendecomposition of S_u (ascending eigenvalues).
    Eigen::SelfAdjointEigenSolver<RowMatXd> es_S(S_u);
    if (es_S.info() != Eigen::Success)
        throw std::runtime_error("TNOBuilder: union overlap eigendecomp failed");
    const Eigen::VectorXd s_eigvals = es_S.eigenvalues();
    const RowMatXd        s_eigvecs = es_S.eigenvectors();

    // Step 4: drop redundant directions (λ < tol_lin_dep).
    int n_keep = 0;
    for (int k = 0; k < m; ++k)
        if (s_eigvals(k) > tol_lin_dep_) ++n_keep;
    out.n_dropped_overlap = m - n_keep;
    if (n_keep == 0) return out;

    // Step 5: build the orthonormal mixing matrix M = V_kept · diag(1/√λ).
    // Kept eigenvalues are the largest n_keep, at the tail (ascending).
    RowMatXd M(m, n_keep);
    for (int k = 0; k < n_keep; ++k) {
        const int src = m - n_keep + k;
        M.col(k) = s_eigvecs.col(src) / std::sqrt(s_eigvals(src));
    }

    // Step 6: orthonormal raw TNO basis in AO basis: Q_orth = Q_union · M.
    // By construction: Q_orth^T · S_AO · Q_orth = I_n_keep.
    RowMatXd Q_orth = Q_union * M; // (nao × n_keep)

    // Step 6b: TNO occupation-number truncation (t_cut_tno > 0). Build the
    // triple pair density D in the orthonormal union span — the sum of the
    // three pair PNO densities Dp = T̃·Tᵀ + T̃ᵀ·T (T̃ = 2T − Tᵀ, T = Y_pair),
    // each mapped into the union via R_p = Q_orthᵀ·S·bar_Q_p — diagonalise it,
    // and keep only directions whose occupation exceeds t_cut_tno. This is the
    // ORCA-style TNO truncation that makes per-triple cost (~n_tno³) affordable;
    // the union span (Q_orth, n_keep) is the most-complete/slowest limit (off).
    int      n_use = n_keep;
    RowMatXd Q_use = Q_orth;   // (nao × n_use), S-orthonormal
    if (t_cut_tno_ > 0.0) {
        RowMatXd D = RowMatXd::Zero(n_keep, n_keep);
        const PairData* trip[3] = {&p_ij, &p_ik, &p_jk};
        for (int t = 0; t < 3; ++t) {
            const PairData& p = *trip[t];
            const int n_p = p.n_pno;
            if (n_p == 0 || p.Y.empty()) continue;
            Eigen::Map<const RowMatXd> bQ_p(p.bar_Q.data(), nao_, n_p);
            const RowMatXd R_p =
                Q_orth.transpose() * (S_map * bQ_p);          // (n_keep × n_p)
            Eigen::Map<const RowMatXd> T(p.Y.data(), n_p, n_p);
            const RowMatXd Ttil = 2.0 * T - T.transpose();
            RowMatXd Dp = Ttil * T.transpose()
                        + Ttil.transpose() * T;               // (n_p × n_p)
            Dp = 0.5 * (Dp + Dp.transpose());
            D.noalias() += R_p * Dp * R_p.transpose();
        }
        D = 0.5 * (D + D.transpose());
        Eigen::SelfAdjointEigenSolver<RowMatXd> es_D(D);
        if (es_D.info() == Eigen::Success) {
            const Eigen::VectorXd d_eig = es_D.eigenvalues();   // ascending
            const RowMatXd        d_vec = es_D.eigenvectors();
            int n_tno = 0;
            for (int k = 0; k < n_keep; ++k)
                if (d_eig(k) > t_cut_tno_) ++n_tno;
            if (n_tno > 0 && n_tno < n_keep) {
                // Keep the largest-occupation directions (tail, ascending order).
                RowMatXd U_keep(n_keep, n_tno);
                for (int k = 0; k < n_tno; ++k)
                    U_keep.col(k) = d_vec.col(n_keep - n_tno + k);
                Q_use = Q_orth * U_keep;   // (nao × n_tno), still S-orthonormal
                n_use = n_tno;
            }
        }
    }

    // Step 7: project Fock onto the (possibly truncated) orthonormal subspace.
    // F_sub[a, b] = Q_use_col_a^T · F_AO · Q_use_col_b.
    const RowMatXd FQ = F_map * Q_use;              // (nao × n_use)
    RowMatXd F_sub = Q_use.transpose() * FQ;        // (n_use × n_use)
    F_sub = 0.5 * (F_sub + F_sub.transpose());

    // Step 8: diagonalise F_sub to get semi-canonical TNOs.
    Eigen::SelfAdjointEigenSolver<RowMatXd> es_F(F_sub);
    if (es_F.info() != Eigen::Success)
        throw std::runtime_error("TNOBuilder: F-subspace eigendecomp failed");
    const Eigen::VectorXd f_eigvals = es_F.eigenvalues(); // ascending
    const RowMatXd        V_can     = es_F.eigenvectors();

    // Step 9: pack results. Q_tno = Q_use · V_can (nao × n_use), eps in
    // ascending order (matches LMO Fock convention used elsewhere).
    RowMatXd Q_tno = Q_use * V_can;

    out.n_tno = n_use;
    out.Q_tno.assign(static_cast<size_t>(nao_) * n_use, 0.0);
    out.eps_tno.assign(n_use, 0.0);
    Eigen::Map<RowMatXd>(out.Q_tno.data(), nao_, n_use) = Q_tno;
    for (int k = 0; k < n_use; ++k) out.eps_tno[k] = f_eigvals(k);

    return out;
}

// ---------------------------------------------------------------------------
//  Phase 3.2.1 — T2 projection from pair PNO basis to TNO basis.
// ---------------------------------------------------------------------------

std::vector<real_t> project_pno_to_tno(const TNOData& tno,
                                       const PairData& pair,
                                       const real_t* S_AO,
                                       int nao) {
    const int n_tno = tno.n_tno;
    const int n_p   = pair.n_pno;
    if (n_tno == 0 || n_p == 0)
        return std::vector<real_t>(static_cast<size_t>(n_tno) * n_tno, 0.0);

    Eigen::Map<const RowMatXd> Qtno(tno.Q_tno.data(), nao, n_tno);
    Eigen::Map<const RowMatXd> Smap(S_AO,             nao, nao);
    Eigen::Map<const RowMatXd> bQ  (pair.bar_Q.data(), nao, n_p);
    Eigen::Map<const RowMatXd> Y   (pair.Y.data(),     n_p, n_p);

    // R = Q_tno^T · S · bar_Q   (n_tno × n_p)
    const RowMatXd S_bQ = Smap * bQ;            // (nao × n_p)
    const RowMatXd R    = Qtno.transpose() * S_bQ; // (n_tno × n_p)

    // T_tno = R · Y · R^T   (n_tno × n_tno)
    const RowMatXd RY  = R * Y;                  // (n_tno × n_p)
    const RowMatXd T   = RY * R.transpose();     // (n_tno × n_tno)

    std::vector<real_t> out(static_cast<size_t>(n_tno) * n_tno, 0.0);
    Eigen::Map<RowMatXd>(out.data(), n_tno, n_tno) = T;
    return out;
}

T2InTNO project_triple_t2_to_tno(const TNOData& tno,
                                 const PairData& p_ij,
                                 const PairData& p_ik,
                                 const PairData& p_jk,
                                 const real_t* S_AO,
                                 int nao) {
    T2InTNO out;
    out.T_ij = project_pno_to_tno(tno, p_ij, S_AO, nao);
    out.T_ik = project_pno_to_tno(tno, p_ik, S_AO, nao);
    out.T_jk = project_pno_to_tno(tno, p_jk, S_AO, nao);
    return out;
}

// ---------------------------------------------------------------------------
//  Phase 3.2.3b — W[a,b,c] tensor builder.
// ---------------------------------------------------------------------------

std::vector<real_t> build_W_tensor_for_triple(
    const TNOData& tno,
    const ERIInTNO& eri,
    const T2InTNO& t2_triple,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc) {
    const int n = tno.n_tno;
    const size_t n3 = static_cast<size_t>(n) * n * n;
    std::vector<real_t> W(n3, 0.0);
    if (n == 0) return W;

    // Local accessors for readability.
    const real_t* K = eri.K_iadc.data();        // 3 × n × n × n
    const real_t* L = eri.L_lcmn.data();        // nocc × n × 9
    const real_t* T_ij = t2_triple.T_ij.data(); // n × n
    const real_t* T_ik = t2_triple.T_ik.data();
    const real_t* T_jk = t2_triple.T_jk.data();

    auto K_at = [&](int loc, int x, int d, int y) -> real_t {
        return K[((static_cast<size_t>(loc) * n + x) * n + d) * n + y];
    };
    auto L_at = [&](int l, int x, int m_loc, int n_loc) -> real_t {
        return L[(static_cast<size_t>(l) * n + x) * 9 + m_loc * 3 + n_loc];
    };
    auto T_at = [&](const real_t* T, int x, int y) -> real_t {
        return T[static_cast<size_t>(x) * n + y];
    };

    // 6 particle permutation terms + 6 hole terms, indexed below by the
    // "slot mapping" laid out in dlpno_tno's design notes:
    //   slot_1 contributes the (lmo, vir) used in K's outer two indices
    //   slot_2 contributes the (lmo, vir) used in T2's first index
    //   slot_3 contributes the (lmo, vir) used in T2's second-LMO index
    // No inner OMP pragma — the caller (run_phase33_triple_loop) already
    // parallelises across triples, and nested OMP regions are typically
    // disabled.
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
                real_t v = 0.0;

                // Particle terms.
                for (int d = 0; d < n; ++d) {
                    v += T_at(T_jk, b, d) * K_at(0, a, d, c); // p0
                    v += T_at(T_jk, d, c) * K_at(0, a, d, b); // p1: t_kj^cd = T_jk[d,c]
                    v += T_at(T_ik, a, d) * K_at(1, b, d, c); // p2
                    v += T_at(T_ij, d, b) * K_at(2, c, d, a); // p3: t_ji^bd = T_ij[d,b]
                    v += T_at(T_ij, a, d) * K_at(2, c, d, b); // p4
                    v += T_at(T_ik, d, c) * K_at(1, b, d, a); // p5: t_ki^cd = T_ik[d,c]
                }

                // Hole terms.
                for (int l = 0; l < nocc; ++l) {
                    const real_t* Til = T_il_ext[l].data();
                    const real_t* Tjl = T_jl_ext[l].data();
                    const real_t* Tkl = T_kl_ext[l].data();
                    if (T_il_ext[l].empty() && T_jl_ext[l].empty()
                        && T_kl_ext[l].empty()) continue;
                    if (!T_il_ext[l].empty()) {
                        v -= T_at(Til, a, b) * L_at(l, c, 1, 2); // h0
                        v -= T_at(Til, a, c) * L_at(l, b, 2, 1); // h1
                    }
                    if (!T_jl_ext[l].empty()) {
                        v -= T_at(Tjl, b, a) * L_at(l, c, 0, 2); // h2: t_jl^ba
                        v -= T_at(Tjl, b, c) * L_at(l, a, 2, 0); // h5: t_jl^bc
                    }
                    if (!T_kl_ext[l].empty()) {
                        v -= T_at(Tkl, c, b) * L_at(l, a, 1, 0); // h3: t_kl^cb
                        v -= T_at(Tkl, c, a) * L_at(l, b, 0, 1); // h4: t_kl^ca
                    }
                }

                W[(static_cast<size_t>(a) * n + b) * n + c] = v;
            }
    return W;
}

// ---------------------------------------------------------------------------
//  Phase 3.2.3c — closed-shell (T) energy for one triple.
// ---------------------------------------------------------------------------

real_t compute_triple_t_energy(int i, int j, int k,
                               real_t eps_i, real_t eps_j, real_t eps_k,
                               const TNOData& tno,
                               const std::vector<real_t>& W) {
    const int n = tno.n_tno;
    if (n == 0) return 0.0;

    auto W_at = [&](int a, int b, int c) -> real_t {
        return W[(static_cast<size_t>(a) * n + b) * n + c];
    };

    // Phase 3.2.5 fix (2026-05-09): Riplinger DLPNO-(T) summation rule.
    //   ω(W)[a,b,c] is the closed-shell P̂_abc projector with all six S_3 perms,
    //   verified bit-exact against PySCF ccsd_t_slow's r3() function:
    //     ω = 4 W_abc − 2 W_acb − 2 W_bac − 2 W_cba + W_bca + W_cab
    //   The previous form was missing the −2 W_cba term (sum of weights = 2,
    //   not 0 as required for a proper irreducible projector). On H2O sto-3g
    //   pseudo-canonical TNO this changed the per-triple factor by ~9×.
    //   Caller (run_phase33_triple_loop) is also expected to skip partial-
    //   equal (i=j or j=k) triples per Riplinger 2013 — those over-count in
    //   the 12-term W formulation and the strict i<j<k restriction is the
    //   standard DLPNO-(T) treatment.
    const real_t e_occ_sum = eps_i + eps_j + eps_k;
    real_t E = 0.0;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
                const real_t W_abc = W_at(a, b, c);
                const real_t W_acb = W_at(a, c, b);
                const real_t W_bac = W_at(b, a, c);
                const real_t W_bca = W_at(b, c, a);
                const real_t W_cab = W_at(c, a, b);
                const real_t W_cba = W_at(c, b, a);
                const real_t omega = 4.0 * W_abc - 2.0 * W_acb - 2.0 * W_bac
                                   - 2.0 * W_cba + W_bca + W_cab;
                const real_t D = e_occ_sum
                               - tno.eps_tno[a] - tno.eps_tno[b] - tno.eps_tno[c];
                E += W_abc * omega / D;
            }

    // Riplinger 2013 strict i<j<k convention: factor = 1 for distinct triples.
    // Partial-equal (i=j or j=k) and all-equal triples are filtered upstream.
    return E;
}

std::vector<real_t> project_pair_t2_oriented_to_tno(
    const TNOData& tno,
    const std::vector<PairData>& pairs,
    const std::vector<PairSetup>& setups,
    const std::vector<int>& pair_lookup,
    int lmo_p, int lmo_q,
    const real_t* S_AO,
    int nao, int nocc) {
    const int n = tno.n_tno;
    std::vector<real_t> out(static_cast<size_t>(n) * n, 0.0);
    if (n == 0) return out;
    if (lmo_p < 0 || lmo_p >= nocc || lmo_q < 0 || lmo_q >= nocc) return out;

    const int idx = pair_lookup[lmo_p * nocc + lmo_q];
    if (idx < 0) return out;

    const PairData&  pair  = pairs[idx];
    const PairSetup& setup = setups[idx];
    if (pair.n_pno == 0) return out;

    auto T_canonical = project_pno_to_tno(tno, pair, S_AO, nao);

    // Canonical storage convention: setup.i ≤ setup.j. If the caller asked
    // for (q, p) when storage is (p, q), the requested amplitude is the
    // transpose: t_{q,p}^{ab} = t_{p,q}^{ba}.
    const bool need_transpose = (setup.i == lmo_q && setup.j == lmo_p);

    if (need_transpose) {
        // out[a, b] = T_canonical[b, a]
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < n; ++b)
                out[static_cast<size_t>(a) * n + b] =
                    T_canonical[static_cast<size_t>(b) * n + a];
    } else {
        out = std::move(T_canonical);
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Phase 3.2.2 — ERI in TNO basis (per triple).
// ---------------------------------------------------------------------------

ERIInTNO build_eri_in_tno(const TNOData& tno,
                          const int triple_lmos[3],
                          const real_t* B_lmo_ao,
                          const real_t* B_ao_ao,
                          const real_t* B_lmo_lmo,
                          int nao,
                          int nocc,
                          int naux,
                          bool build_L,
                          const real_t* B_TTQ_precomputed) {
    ERIInTNO out;
    out.n_tno = tno.n_tno;
    out.nocc  = nocc;
    if (tno.n_tno == 0) return out;
    const int n = tno.n_tno;

    // Q_tno is stored row-major (nao × n).
    Eigen::Map<const RowMatXd> Qtno(tno.Q_tno.data(), nao, n);

    // -----------------------------------------------------------------
    // Step 1: B_l_tno[l, a, Q] = Σ_ν Q_tno[ν, a] · B_lmo_ao[l, ν, Q]
    //         shape (nocc × n × naux)
    //
    // For each (l, Q) slab: B_lmo_ao[(l, *, Q)] is a length-nao vector.
    // We need to contract its ν-axis against Q_tno's ν-axis.
    //
    // Cheaper layout: per l, view B_lmo_ao_l as (nao × naux) row-major,
    // then B_lTQ_l = Q_tno^T · B_lmo_ao_l  (n × naux).
    // -----------------------------------------------------------------
    std::vector<real_t> B_lTQ(static_cast<size_t>(nocc) * n * naux, 0.0);
    for (int l = 0; l < nocc; ++l) {
        Eigen::Map<const RowMatXd> Bl(
            B_lmo_ao + static_cast<size_t>(l) * nao * naux, nao, naux);
        const RowMatXd Bl_tno = Qtno.transpose() * Bl;     // (n × naux)
        Eigen::Map<RowMatXd>(
            B_lTQ.data() + static_cast<size_t>(l) * n * naux, n, naux) = Bl_tno;
    }

    // -----------------------------------------------------------------
    // Step 2: B_tno_tno[a, b, Q] = Σ_μν Q_tno[μ,a] Q_tno[ν,b] B_ao_ao[μ ν Q]
    //         shape (n × n × naux)
    //
    // If the caller pre-computed B_TTQ on GPU (via EriBuildGpu::build_b_ttq),
    // reuse it directly — that's the dominant CPU cost on large systems.
    // -----------------------------------------------------------------
    std::vector<real_t> B_TTQ;
    const real_t* B_TTQ_ptr = B_TTQ_precomputed;
    if (B_TTQ_ptr == nullptr) {
        B_TTQ.assign(static_cast<size_t>(n) * n * naux, 0.0);
        RowMatXd M_Q(nao, nao);
        for (int Q = 0; Q < naux; ++Q) {
            for (int mu = 0; mu < nao; ++mu)
                for (int nu = 0; nu < nao; ++nu)
                    M_Q(mu, nu) =
                        B_ao_ao[(static_cast<size_t>(mu) * nao + nu) * naux + Q];
            const RowMatXd MQ_T = M_Q * Qtno;
            const RowMatXd TT   = Qtno.transpose() * MQ_T;
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    B_TTQ[(static_cast<size_t>(a) * n + b) * naux + Q] = TT(a, b);
        }
        B_TTQ_ptr = B_TTQ.data();
    }

    // -----------------------------------------------------------------
    // Step 3: K[i_loc, a, d, c] = Σ_Q B_lTQ[i_lmo, a, Q] · B_TTQ[d, c, Q]
    //         shape 3 × n × n × n
    //
    // Flatten (d, c) → dc index of length n²:
    //   B_TTQ_flat: (n² × naux)
    //   K_loc[a, dc] = B_lTQ_l[a, Q] · B_TTQ_flat[dc, Q]^T
    //                = B_lTQ_l (n × naux) · B_TTQ_flat^T (naux × n²)
    // -----------------------------------------------------------------
    out.K_iadc.assign(static_cast<size_t>(3) * n * n * n, 0.0);
    Eigen::Map<const RowMatXd> B_TTQ_flat(B_TTQ_ptr, n * n, naux);
    for (int i_loc = 0; i_loc < 3; ++i_loc) {
        const int l = triple_lmos[i_loc];
        Eigen::Map<const RowMatXd> Bl(
            B_lTQ.data() + static_cast<size_t>(l) * n * naux, n, naux);
        const RowMatXd Kloc = Bl * B_TTQ_flat.transpose();   // (n × n²)
        // Pack into out.K_iadc[i_loc, a, d, c] = Kloc[a, d*n + c]
        const size_t base = static_cast<size_t>(i_loc) * n * n * n;
        for (int a = 0; a < n; ++a)
            for (int dc = 0; dc < n * n; ++dc)
                out.K_iadc[base + static_cast<size_t>(a) * n * n + dc] =
                    Kloc(a, dc);
    }

    // -----------------------------------------------------------------
    // Step 4: L[l, c, m_loc, n_loc] = Σ_Q B_lTQ[l, c, Q] · B_lmo_lmo[m, n', Q]
    //         shape (nocc × n × 9)
    //
    // Only needed by the legacy 12-term W path (build_W_tensor_for_triple).
    // Phase 3.2.6 closed-shell (T) (compute_triple_t_energy_pyscf) uses M
    // tensors (build_hole_m_tensors) instead, so skip when build_L == false.
    // -----------------------------------------------------------------
    if (build_L) {
        out.L_lcmn.assign(static_cast<size_t>(nocc) * n * 9, 0.0);
        Eigen::Map<const RowMatXd> B_lTQ_flat(B_lTQ.data(), nocc * n, naux);
        Eigen::MatrixXd v_mn(naux, 9);
        for (int m_loc = 0; m_loc < 3; ++m_loc)
            for (int n_loc = 0; n_loc < 3; ++n_loc) {
                const int m = triple_lmos[m_loc];
                const int nn = triple_lmos[n_loc];
                for (int Q = 0; Q < naux; ++Q)
                    v_mn(Q, m_loc * 3 + n_loc) =
                        B_lmo_lmo[(static_cast<size_t>(m) * nocc + nn) * naux + Q];
            }
        // L_flat = B_lTQ_flat · v_mn  shape (nocc*n × 9)
        const Eigen::MatrixXd L_flat = B_lTQ_flat * v_mn;
        for (int lc = 0; lc < nocc * n; ++lc)
            for (int mn = 0; mn < 9; ++mn)
                out.L_lcmn[static_cast<size_t>(lc) * 9 + mn] = L_flat(lc, mn);
    }

    return out;
}

// ---------------------------------------------------------------------------
//  Phase 3.2.6 — hole-side M tensors and PySCF-equivalent (T) energy.
// ---------------------------------------------------------------------------

HoleMTensors build_hole_m_tensors(const TNOData& tno,
                                  const int triple_lmos[3],
                                  const real_t* B_lmo_ao,
                                  const real_t* B_lmo_lmo,
                                  int nao,
                                  int nocc,
                                  int naux) {
    HoleMTensors out;
    out.n_tno = tno.n_tno;
    out.nocc  = nocc;
    if (tno.n_tno == 0) return out;
    const int n = tno.n_tno;

    // Step 1: B_lTQ_triple[loc, a, Q] = Σ_ν Q_tno[ν, a] · B_lmo_ao[lmo_loc, ν, Q]
    // for loc ∈ {0, 1, 2} (the three triple LMOs).
    Eigen::Map<const RowMatXd> Qtno(tno.Q_tno.data(), nao, n);
    std::array<std::vector<real_t>, 3> B_lTQ_triple;
    for (int loc = 0; loc < 3; ++loc) {
        const int lmo = triple_lmos[loc];
        Eigen::Map<const RowMatXd> Bl(
            B_lmo_ao + static_cast<size_t>(lmo) * nao * naux, nao, naux);
        const RowMatXd Bl_tno = Qtno.transpose() * Bl;     // (n × naux)
        B_lTQ_triple[loc].assign(static_cast<size_t>(n) * naux, 0.0);
        Eigen::Map<RowMatXd>(B_lTQ_triple[loc].data(), n, naux) = Bl_tno;
    }

    // Step 2: for each ordered pair (slot_p, slot_q) with slot_p ≠ slot_q,
    // M_pq[l, a] = Σ_Q B_lTQ_triple[slot_p][a, Q] · B_lmo_lmo[l, lmo_q, Q]
    // Reformulate as M_pq^T (a, l) = (B_lTQ_triple[slot_p] (n × naux))
    //                              · (B_lmo_lmo_slice[lmo_q]^T (naux × nocc))
    for (int sp = 0; sp < 3; ++sp) {
        for (int sq = 0; sq < 3; ++sq) {
            if (sp == sq) continue;
            const int lmo_q = triple_lmos[sq];
            // Extract B_lmo_lmo[:, lmo_q, :] as (nocc × naux).
            // Layout of B_lmo_lmo: (l * nocc + m) * naux + Q ; we want m == lmo_q.
            RowMatXd Bq(nocc, naux);
            for (int l = 0; l < nocc; ++l)
                for (int Q = 0; Q < naux; ++Q)
                    Bq(l, Q) = B_lmo_lmo[(static_cast<size_t>(l) * nocc + lmo_q) * naux + Q];
            // M_pq has shape (nocc × n_tno), row-major: M[l, a].
            Eigen::Map<const RowMatXd> Bp(B_lTQ_triple[sp].data(), n, naux);
            const RowMatXd M_la = Bq * Bp.transpose();  // (nocc × n)
            const int slot = sp * 3 + sq;
            out.M[slot].assign(static_cast<size_t>(nocc) * n, 0.0);
            Eigen::Map<RowMatXd>(out.M[slot].data(), nocc, n) = M_la;
        }
    }
    return out;
}

namespace {

/// Build a single simple-W tensor for one permutation σ of (i,j,k).
///   w(a,b,c) = Σ_d K_loc0(a, b, d) · T_part(c, d)   [particle]
///            − Σ_l M_pq(l, a)        · T_lp_ext[l](b, c)   [hole]
/// Inputs are already perm-resolved by the caller.
inline std::vector<real_t> build_simple_w_perm(
    int n_tno, int nocc,
    const real_t* K_loc0,                                 ///< n_tno³ slab for σ[0]
    const real_t* T_part,                                 ///< n_tno² (c, d) for (σ[2], σ[1])
    const real_t* M_pq,                                   ///< nocc × n_tno (l, a)
    const std::vector<std::vector<real_t>>& T_lp_ext      ///< per l (n_tno × n_tno)
) {
    const int n = n_tno;
    std::vector<real_t> w(static_cast<size_t>(n) * n * n, 0.0);

    // Particle: w[a, b, c] += K_loc0[a, b, d] · T_part[c, d], summed over d.
    // Reshape K as (n*n × n) (rows: (a,b), cols: d). T_part (n × n). Then
    // (n*n × n) · (n × n)^T → (n*n × n), reshape back to (a, b, c).
    if (T_part != nullptr) {
        Eigen::Map<const RowMatXd> Kab_d(K_loc0, n * n, n);   // (ab, d)
        Eigen::Map<const RowMatXd> Tcd(T_part, n, n);         // (c, d)
        const RowMatXd part_ab_c = Kab_d * Tcd.transpose();   // (ab, c)
        // Pack: w[a,b,c] = part_ab_c[(a, b), c].
        Eigen::Map<RowMatXd>(w.data(), n * n, n) += part_ab_c;
    }

    // Hole: w[a, b, c] -= Σ_l M_pq[l, a] · t_{l, lmo_σ[2]}^{bc}
    // The caller stores T_lp_ext[l] = project_pair_t2_oriented_to_tno(lmo_σ[2], l)
    // = t_{lmo_σ[2], l}^{ab} (axes a, b). By t2 antisymmetry under particle
    // exchange, t_{l, lmo_σ[2]}^{bc} = t_{lmo_σ[2], l}^{cb} = T_lp_ext[l].T at (b, c).
    Eigen::Map<const RowMatXd> Mla(M_pq, nocc, n);   // (l, a)
    for (int l = 0; l < nocc; ++l) {
        if (T_lp_ext[l].empty()) continue;
        Eigen::Map<const RowMatXd> Tl_stored(T_lp_ext[l].data(), n, n);
        // Compute Tl_used = Tl_stored^T to align with t_{l, lmo_σ[2]}^{bc}.
        const RowMatXd Tl_used = Tl_stored.transpose();   // (b, c)
        for (int a = 0; a < n; ++a) {
            const real_t mla = Mla(l, a);
            if (mla == 0.0) continue;
            Eigen::Map<RowMatXd> w_a(w.data() + static_cast<size_t>(a) * n * n, n, n);
            w_a.noalias() -= mla * Tl_used;
        }
    }
    return w;
}

/// Apply r3-style 6-permutation operator on (a,b,c) of W:
///   r3(W)[a,b,c] = 4 W[a,b,c] + W[c,a,b] + W[b,c,a]
///                − 2 W[c,b,a] − 2 W[a,c,b] − 2 W[b,a,c]
inline std::vector<real_t> apply_r3_abc(const std::vector<real_t>& W, int n) {
    std::vector<real_t> R(W.size(), 0.0);
    auto IDX = [&](int a, int b, int c) {
        return (static_cast<size_t>(a) * n + b) * n + c;
    };
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
                R[IDX(a, b, c)] =
                    4.0 * W[IDX(a, b, c)]
                    +     W[IDX(c, a, b)]
                    +     W[IDX(b, c, a)]
                    - 2.0 * W[IDX(c, b, a)]
                    - 2.0 * W[IDX(a, c, b)]
                    - 2.0 * W[IDX(b, a, c)];
            }
    return R;
}

/// Sum_{abc} W_perm_w[π(a,b,c)] · z[a,b,c] / D[a,b,c], with z and D varying
/// in the caller. π is a permutation of (a,b,c) determined by perm_code:
///   0 = identity, 1 = (a,c,b), 2 = (b,a,c), 3 = (b,c,a),
///   4 = (c,a,b), 5 = (c,b,a)
inline real_t contract_w_z_perm(const std::vector<real_t>& W,
                                const std::vector<real_t>& z,
                                const std::vector<real_t>& D_inv,
                                int n, int perm_code) {
    real_t sum = 0.0;
    auto IDX = [&](int a, int b, int c) {
        return (static_cast<size_t>(a) * n + b) * n + c;
    };
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
                size_t w_idx;
                switch (perm_code) {
                    case 0: w_idx = IDX(a, b, c); break;
                    case 1: w_idx = IDX(a, c, b); break;
                    case 2: w_idx = IDX(b, a, c); break;
                    case 3: w_idx = IDX(b, c, a); break;
                    case 4: w_idx = IDX(c, a, b); break;
                    case 5: w_idx = IDX(c, b, a); break;
                    default: w_idx = IDX(a, b, c);
                }
                sum += W[w_idx] * z[IDX(a, b, c)] * D_inv[IDX(a, b, c)];
            }
    return sum;
}

} // namespace

real_t compute_triple_t_energy_pyscf(
    int i, int j, int k,
    real_t eps_i, real_t eps_j, real_t eps_k,
    const TNOData& tno,
    const real_t* K_iadc,
    const std::array<std::vector<real_t>, 9>& M,
    const std::array<std::vector<real_t>, 9>& T_part_oriented,
    const std::vector<std::vector<real_t>>& T_il_ext,
    const std::vector<std::vector<real_t>>& T_jl_ext,
    const std::vector<std::vector<real_t>>& T_kl_ext,
    int nocc)
{
    const int n = tno.n_tno;
    if (n == 0) return 0.0;

    // The 6 perms of (i,j,k) in slot-space: each row is (slot[0], slot[1], slot[2])
    // where slot ∈ {0=i, 1=j, 2=k}. We use these to look up which K, M,
    // T_part, and T_lp_ext to use.
    constexpr int perm_table[6][3] = {
        {0, 1, 2}, // (i,j,k)
        {0, 2, 1}, // (i,k,j)
        {1, 0, 2}, // (j,i,k)
        {1, 2, 0}, // (j,k,i)
        {2, 0, 1}, // (k,i,j)
        {2, 1, 0}, // (k,j,i)
    };

    // Pointers to T_lp_ext for each "third slot" σ[2] ∈ {0, 1, 2}.
    const std::vector<std::vector<real_t>>* T_ext_by_slot[3] = {
        &T_il_ext, &T_jl_ext, &T_kl_ext
    };

    // Build the 6 simple W tensors w_σ[a,b,c].
    const size_t n3 = static_cast<size_t>(n) * n * n;
    std::array<std::vector<real_t>, 6> w_tensors;
    for (int p = 0; p < 6; ++p) {
        const int s0 = perm_table[p][0];
        const int s1 = perm_table[p][1];
        const int s2 = perm_table[p][2];
        const real_t* K_s0 = K_iadc + static_cast<size_t>(s0) * n * n * n;
        // Particle t: t_{lmo_{s2}, lmo_{s1}}^{cd} → slot (s2, s1) of T_part_oriented.
        const std::vector<real_t>& Tp = T_part_oriented[s2 * 3 + s1];
        const real_t* Tp_ptr = Tp.empty() ? nullptr : Tp.data();
        // Hole M: M[(s0, s1)][l, a].
        const std::vector<real_t>& Mpq = M[s0 * 3 + s1];
        // Hole t: t_{l, lmo_{s2}}^{bc} → T_ext[s2][l].
        const std::vector<std::vector<real_t>>& T_ext = *T_ext_by_slot[s2];
        w_tensors[p] = build_simple_w_perm(n, nocc, K_s0, Tp_ptr,
                                            Mpq.data(), T_ext);
    }

    // Build inverse-denominator tensor 1/D[a,b,c] (scaled by d3_ijk).
    int d3_factor;
    if (i == j && j == k)        d3_factor = 6;
    else if (i == j || j == k)   d3_factor = 2;
    else                         d3_factor = 1;
    const real_t e_occ_sum = eps_i + eps_j + eps_k;
    std::vector<real_t> D_inv(n3, 0.0);
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < n; ++c) {
                const real_t D = e_occ_sum
                               - tno.eps_tno[a] - tno.eps_tno[b] - tno.eps_tno[c];
                D_inv[(static_cast<size_t>(a) * n + b) * n + c] =
                    1.0 / (D * d3_factor);
            }

    // Build z_σ = r3(w_σ) for each σ. (1/D applied inside the contraction.)
    std::array<std::vector<real_t>, 6> z_tensors;
    for (int p = 0; p < 6; ++p) z_tensors[p] = apply_r3_abc(w_tensors[p], n);

    // 36 contractions: for each (σ_z, σ_w), contract w_σw under permutation
    // pattern π_{σw,σz} with z_σz/D. Each perm_codes[sz][sw] gives the
    // index permutation π applied to w_σw before contracting with z_σz.
    // Verified bit-exact against PySCF on H2O sto-3g and cc-pVDZ via
    // verify_perm_table.py (ratio = 1.000000, diff < 1e-11).
    constexpr int perm_codes[6][6] = {
        {0, 1, 2, 3, 4, 5}, // sz=0 (zijk)
        {1, 0, 4, 5, 2, 3}, // sz=1 (zikj)
        {2, 3, 0, 1, 5, 4}, // sz=2 (zjik)
        {4, 5, 1, 0, 3, 2}, // sz=3 (zjki)
        {3, 2, 5, 4, 0, 1}, // sz=4 (zkij)
        {5, 4, 3, 2, 1, 0}, // sz=5 (zkji)
    };

    real_t et = 0.0;
    for (int sz = 0; sz < 6; ++sz)
        for (int sw = 0; sw < 6; ++sw)
            et += contract_w_z_perm(w_tensors[sw], z_tensors[sz], D_inv,
                                     n, perm_codes[sz][sw]);
    return 2.0 * et;
}

} // namespace gansu
