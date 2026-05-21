#!/usr/bin/env python3
"""
PySCF reference for STEOM-CCSD G^{1h1p} construction
=====================================================

STATUS (2026-05-21): sub-phase 3.4 (F^eff_oo + F^eff_vv) ship + sub-phase
3.5 (W^eff partial: bar W + ujaei·X_EA - umabi·X_MI; cross umaei in 3.6).

   Reference: megansimons/steom_ccsd-ct (GitHub) — CFOUR-style closed-shell
   RHF STEOM-CCSD with explicit working equations. Specifically:
     • `steom_intermediates.cxx` lines 7-81 = `gmi_steom_rhf` = F^eff_oo
     • `steom.cxx` lines 23-49 = `renormalize` = X(MI) matrix build

   Current H2O sto-3g state:
     • 11/11 bar-H bit-exact vs GANSU (view-aliasing bug fix landed)
     • X(MI) per CFOUR canonical (active R1 matrix inverse)
     • U(M,I) per CFOUR `gmi_steom_rhf` (PySCF σ1-style spin-adapted from
       IP-EOM matvec: +2Fov·R2 - Fov·R2 - 2Wooov·R2 + Wooov·R2)
     • F^eff_oo active-row dressing, inactive block = bar Loo (no dressing)
     • G^{1h1p} eigenvalues vs ORCA: state 2 at +4.5 mHa, states 0/1 at
       ~-45 mHa undershoot (expected — W^eff + cross + G(EM)/G(Mn,Ie) of
       sub-phase 3.5/3.6 still missing)

   ★ The F^eff_oo Frobenius value emitted by this script is the reference
     against which GANSU C++ `STEOMCCSDOperator::build_dressed_intermediates`
     extension should be validated (sub-phase 3.4 ship).

Phase P3 sub-phase 3.4 (F^eff dressing) の reference 実装。GANSU `STEOMCCSDOperator`
が出力する Frobenius と element-wise 比較するための ground truth を生成する。

P1 IP-EOM + P2 EA-EOM の reference (`pyscf_ip_eom_ccsd_reference.py` /
`pyscf_ea_eom_ccsd_reference.py`) は PySCF の `eom_rccsd.EOMIP/EOMEA` をそのまま
呼ぶだけで済んだが、PySCF は STEOM-CCSD を実装していないので、bar-H 中間体 +
Ŝ amplitudes + G^{1h1p} の組み立てを numpy で陽に行う。

実装方針:
  Step 1. PySCF で RHF + CCSD → T1, T2
  Step 2. PySCF rintermediates.py の 11 種 bar-H 中間体を再構築 (Loo, Lvv, Fov,
          Woooo, Wooov, Wovov, Wovvo, Wovoo, Wvovv, Wvvvv, Wvvvo)
  Step 3. PySCF EOMIP / EOMEA で n_act_occ IP root + n_act_vir EA root を取得
  Step 4. Ŝ intermediate normalization (STEOM.md §6.3):
            Ŝ^IP[ñ, i, j, a] = R2_IP[ñ, i, j, a] / R1_IP[ñ, ñ]
            Ŝ^EA[ẽ, j, a, b] = R2_EA[ẽ, j, a, b] / R1_EA[ẽ, ẽ]
          (canonical では active NTO = active MO なので diagonal R1 を直接使う)
  Step 5. G^{1h1p}_{ia,jb} を term-by-term で組み立てて explicit な matrix を作る
          (full nocc · nvir 空間、closed-shell singlet block)
  Step 6. 各 contribution の Frobenius を出力 (F^eff_oo, F^eff_vv, W^eff, cross)
  Step 7. G を非エルミート diagonalize して singlet excitation energies を出力
          → ORCA reference (H2O sto-3g: 0.4354200/0.4998300/0.5916380 Ha) と比較

使い方:
    wsl python3 script/pyscf_steom_feff_reference.py xyz/H2O.xyz sto-3g 3 2 3

  arg 1: xyz path
  arg 2: basis name
  arg 3: n_act_occ (number of IP roots = active occupied count, e.g. 3)
  arg 4: n_act_vir (number of EA roots = active virtual count, e.g. 2)
  arg 5: n_steom_roots (number of final STEOM excited states to print, e.g. 3)

[[pyscf-run-locally]] [[pyscf-cartesian]] [[careful-verification]]
"""

import sys
import numpy as np


def read_xyz(path):
    with open(path) as f:
        natoms = int(f.readline().strip())
        f.readline()
        atoms = []
        for _ in range(natoms):
            parts = f.readline().split()
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "; ".join(atoms)


# ----------------------------------------------------------------------
# Bar-H intermediates (PySCF rintermediates.py literal port for closed-shell
# RHF reference; same 11 intermediates GANSU's IP/EA/STEOM operator builds).
# ----------------------------------------------------------------------
def build_bar_h(eri, t1, t2, f_oo, f_vv, nocc, nvir):
    """Build the 11 bar-H intermediates used by IP/EA/STEOM-CCSD.

    All ERIs are in chemist notation (ij|ab) — same as GANSU.
    Index slices: o=[:nocc], v=[nocc:].
    """
    o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    eri_oooo = np.ascontiguousarray(eri[o, o, o, o])
    eri_ooov = np.ascontiguousarray(eri[o, o, o, v])
    eri_oovv = np.ascontiguousarray(eri[o, o, v, v])
    eri_ovov = np.ascontiguousarray(eri[o, v, o, v])
    eri_ovvo = np.ascontiguousarray(eri[o, v, v, o])
    eri_ovvv = np.ascontiguousarray(eri[o, v, v, v])
    eri_vvvv = np.ascontiguousarray(eri[v, v, v, v])

    # cc_Fov = Σ_{l,d} (2 (kc|ld) - (kd|lc)) t1[l,d]
    Fov = 2.0 * np.einsum("kcld,ld->kc", eri_ovov, t1) \
        -       np.einsum("kdlc,ld->kc", eri_ovov, t1)

    # cc_Foo  (k,i)
    ccFoo = np.diag(f_oo).copy() if f_oo.ndim == 1 else f_oo.copy()
    ccFoo += 2.0 * np.einsum("kcld,ilcd->ki", eri_ovov, t2) \
          -        np.einsum("kdlc,ilcd->ki", eri_ovov, t2)
    ccFoo += 2.0 * np.einsum("kcld,ic,ld->ki", eri_ovov, t1, t1) \
          -        np.einsum("kdlc,ic,ld->ki", eri_ovov, t1, t1)

    # cc_Fvv  (a,c)
    ccFvv = np.diag(f_vv).copy() if f_vv.ndim == 1 else f_vv.copy()
    ccFvv -= 2.0 * np.einsum("kcld,klad->ac", eri_ovov, t2) \
          -        np.einsum("kdlc,klad->ac", eri_ovov, t2)
    ccFvv -= 2.0 * np.einsum("kcld,ka,ld->ac", eri_ovov, t1, t1) \
          -        np.einsum("kdlc,ka,ld->ac", eri_ovov, t1, t1)

    # Loo / Lvv
    Loo = ccFoo + np.einsum("kc,ic->ki", Fov, t1) \
        + 2.0 * np.einsum("kilc,lc->ki", eri_ooov, t1) \
        -       np.einsum("likc,lc->ki", eri_ooov, t1)
    Lvv = ccFvv - np.einsum("kc,ka->ac", Fov, t1) \
        + 2.0 * np.einsum("kdac,kd->ac", eri_ovvv, t1) \
        -       np.einsum("kcad,kd->ac", eri_ovvv, t1)

    # Wooov [k,l,i,d] (IP version, also used by Woooo/Wovoo/Wovov/Wovvo)
    Wooov = eri_ooov.transpose(0, 2, 1, 3).copy()  # (k,l,i,d) from (k,i,l,d)
    Wooov += np.einsum("ic,kcld->klid", t1, eri_ovov)

    # Woooo  (PySCF IP version, no t1·t1 symmetrization)
    Woooo = eri_oooo.transpose(0, 2, 1, 3).copy()  # (k,l,i,j) from (k,i,l,j)
    Woooo += np.einsum("kild,jd->klij", eri_ooov, t1)
    Woooo += np.einsum("ljkc,ic->klij", eri_ooov, t1)
    Woooo += np.einsum("kcld,ijcd->klij", eri_ovov, t2)
    Woooo += np.einsum("kcld,ic,jd->klij", eri_ovov, t1, t1)

    # W1ovov + Wovov
    W1ovov = eri_oovv.transpose(0, 2, 1, 3).copy()  # (k,b,i,d) from (k,i,b,d)
    W1ovov -= np.einsum("kcld,ilcb->kbid", eri_ovov, t2)
    Wovov = W1ovov.copy()
    Wovov -= np.einsum("klid,lb->kbid", Wooov, t1)
    Wovov += np.einsum("kcbd,ic->kbid", eri_ovvv, t1)

    # W1ovvo + Wovvo
    W1ovvo = eri_ovvo.transpose(0, 1, 2, 3).copy()  # (k,c,b,j) → (k,b,c,j)
    W1ovvo = eri_ovvo.transpose(0, 2, 1, 3).copy()
    # W1ovvo[k,b,c,j] += sum_{ld} [2(kc|ld) - (kd|lc)] t2[jl,bd] - (kd|lc) t2[lj,bd]
    W1ovvo += 2.0 * np.einsum("kcld,jlbd->kbcj", eri_ovov, t2)
    W1ovvo -=       np.einsum("kcld,ljbd->kbcj", eri_ovov, t2)
    W1ovvo -=       np.einsum("kdlc,jlbd->kbcj", eri_ovov, t2)
    Wovvo = W1ovvo.copy()
    Wovvo -= np.einsum("lb,lkjc->kbcj", t1, Wooov)
    Wovvo += np.einsum("kcbd,jd->kbcj", eri_ovvv, t1)

    # Wovoo (IP-side full PySCF formula, 11 terms) — .copy() forced (see Wvovv note above)
    Wovoo = np.einsum("ikjb->kbij", eri_ooov).copy()
    Wovoo += np.einsum("kbid,jd->kbij", W1ovov, t1)
    Wovoo -= np.einsum("klij,lb->kbij", Woooo, t1)
    Wovoo += np.einsum("kbcj,ic->kbij", W1ovvo, t1)
    Wovoo += 2.0 * np.einsum("kild,ljdb->kbij", eri_ooov, t2)
    Wovoo -=       np.einsum("kild,jldb->kbij", eri_ooov, t2)
    Wovoo -=       np.einsum("likd,ljdb->kbij", eri_ooov, t2)
    Wovoo += np.einsum("kcbd,jidc->kbij", eri_ovvv, t2)
    Wovoo += np.einsum("kcbd,jd,ic->kbij", eri_ovvv, t1, t1)
    Wovoo -= np.einsum("ljkc,libc->kbij", eri_ooov, t2)
    Wovoo += np.einsum("kc,ijcb->kbij", Fov, t2)

    # Wvovv (EA-side)
    # NOTE: np.einsum can return a VIEW on the input when the operation is a
    # pure permutation; in-place `-=` would then mutate eri_ovvv, corrupting
    # all subsequent intermediates that read eri_ovvv (Wvvvv, Wvvvo, Wovoo,
    # ...). Force a contiguous copy.
    Wvovv = np.einsum("ldac->alcd", eri_ovvv).copy()
    Wvovv -= np.einsum("ka,kcld->alcd", t1, eri_ovov)

    # Wvvvv (EA-side) — .copy() forced (see Wvovv note above)
    Wvvvv = np.einsum("acbd->abcd", eri_vvvv).copy()
    Wvvvv -= np.einsum("kcbd,ka->abcd", eri_ovvv, t1)
    Wvvvv -= np.einsum("ldac,lb->abcd", eri_ovvv, t1)
    Wvvvv += np.einsum("kcld,klab->abcd", eri_ovov, t2)
    Wvvvv += np.einsum("kcld,ka,lb->abcd", eri_ovov, t1, t1)

    # Wvvvo (EA-side, 11-term) — .copy() forced (see Wvovv note above)
    Wvvvo = np.einsum("jbca->abcj", eri_ovvv).copy()
    Wvvvo -= np.einsum("lajc,lb->abcj", W1ovov, t1)
    Wvvvo -= np.einsum("kbcj,ka->abcj", W1ovvo, t1)
    Wvvvo += 2.0 * np.einsum("ldac,ljdb->abcj", eri_ovvv, t2)
    Wvvvo -=       np.einsum("ldac,ljbd->abcj", eri_ovvv, t2)
    Wvvvo -=       np.einsum("lcad,ljdb->abcj", eri_ovvv, t2)
    Wvvvo -=       np.einsum("kcbd,jkda->abcj", eri_ovvv, t2)
    Wvvvo += np.einsum("ljkc,lkba->abcj", eri_ooov, t2)
    Wvvvo += np.einsum("ljkc,lb,ka->abcj", eri_ooov, t1, t1)
    Wvvvo -= np.einsum("kc,kjab->abcj", Fov, t2)
    Wvvvo += np.einsum("abcd,jd->abcj", Wvvvv, t1)

    return {
        "Loo": Loo, "Lvv": Lvv, "Fov": Fov,
        "Woooo": Woooo, "Wooov": Wooov,
        "Wovov": Wovov, "Wovvo": Wovvo, "Wovoo": Wovoo,
        "Wvovv": Wvovv, "Wvvvv": Wvvvv, "Wvvvo": Wvvvo,
        "eri_oooo": eri_oooo, "eri_ooov": eri_ooov, "eri_oovv": eri_oovv,
        "eri_ovov": eri_ovov, "eri_ovvo": eri_ovvo, "eri_ovvv": eri_ovvv,
        "eri_vvvv": eri_vvvv,
    }


# ----------------------------------------------------------------------
# X(MI) / X(EA) matrix construction (CFOUR `renormalize` function, lines 23-49
# of steom.cxx). For canonical STEOM with "trans" matrix = identity on the
# active orbital subset:
#   R1_active[m_idx, n_root]   = R1^(n_root)[m_idx]
#   X = inv(R1_active)
# Returns the matrix that, when contracted with U(MI), produces the F^eff
# active-row correction.
# ----------------------------------------------------------------------
def assign_active_1to1(r1_list, n_orb):
    """1:1 assignment of each EOM root to a DISTINCT dominant orbital index.

    The old `argmax(|R1|)` per-root selection can map two (near-)degenerate
    roots to the SAME orbital index, producing a singular active R1 matrix
    (and a meaningless X inverse). The Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) instead finds the global 1:1
    assignment that maximizes Σ_root |R1[root, idx(root)]|, guaranteeing a
    permutation-like (collision-free) active index set.

    Returns active_idx[root] = orbital index assigned to that root.
    """
    from scipy.optimize import linear_sum_assignment
    n_root = len(r1_list)
    cost = np.zeros((n_root, n_orb))
    for r, r1 in enumerate(r1_list):
        cost[r, :] = -np.abs(np.asarray(r1).ravel())   # maximize |R1| → min -|R1|
    row, col = linear_sum_assignment(cost)
    idx = [0] * n_root
    for r, c in zip(row, col):
        idx[int(r)] = int(c)
    return idx


def build_x_matrices(r1_ip_list, r1_ea_list, active_occ_idx, active_vir_idx):
    n_act_occ = len(r1_ip_list)
    n_act_vir = len(r1_ea_list)

    R1_IP_active = np.zeros((n_act_occ, n_act_occ))
    for n_root in range(n_act_occ):
        for m, m_idx in enumerate(active_occ_idx):
            R1_IP_active[m, n_root] = r1_ip_list[n_root][m_idx]
    X_IP = np.linalg.inv(R1_IP_active)  # X_IP[n_root, m_active]

    R1_EA_active = np.zeros((n_act_vir, n_act_vir))
    for e_root in range(n_act_vir):
        for a, a_idx in enumerate(active_vir_idx):
            R1_EA_active[a, e_root] = r1_ea_list[e_root][a_idx]
    X_EA = np.linalg.inv(R1_EA_active)  # X_EA[e_root, a_active]

    return X_IP, X_EA


# ----------------------------------------------------------------------
# Sub-phase 3.5: W^eff dressing intermediates (CFOUR `ujaei_steom_rhf` +
# `umabi_steom_rhf` + partial `gmaei_steom_rhf`).
#
# IMPORTANT CONVENTION NOTES (derived from CFOUR steom_intermediates.cxx +
# axpby algebra):
#
# 1. CFOUR sabej[A,B,E,J] (vv|va|o spec) is the "raw" (alpha-beta direct)
#    closed-shell stored amplitude. In PySCF terms:
#       sabej_direct[A,B,E,J] = R2_EA_list[E][J, A, B]
#       sabej_swap  [A,B,E,J] = R2_EA_list[E][J, B, A]      (first 2 swapped)
#
# 2. CFOUR `.spinad()` toggles the stored form to the singlet-adapted
#    combination (= 2·direct - swap). After `.spinad()`:
#       sabej_after_spinad[A,B,E,J] = 2·sabej_direct - sabej_swap
#
# 3. The CFOUR ujaei build has two phases:
#       Phase 1 (raw): ujaei_raw  ← 3 contractions with sabej_direct
#       Phase 2 (spin):ujaie_raw  ← 3 contractions with sabej_swap
#       Phase 3: ujaei := 1.5·ujaei_raw - 0.5·ujaie_raw   (axpby)
#       Phase 5: ujaie := 1/3·ujaei    + 2/3·ujaie_raw    (axpby)
#    Algebraic substitution shows:
#       ujaei_final = 2·(swap-contraction) - (direct-contraction)
#       ujaie_final = direct-contraction      (single term)
#    where "swap-contraction" = contractions using sabej_swap as source,
#    "direct-contraction" = contractions using sabej_direct as source.
#    Wait — verify by F·S contraction:
#       Phase 1 direct: ujaei_raw += Σ_F sabej_direct[F,B,E,J]·Fov[I,F]
#                                  = Σ_F R2[J,F,B]·Fov[I,F]    ≡ DD
#       Phase 2 swap:   ujaie_raw += Σ_F sabej_swap[B,F,E,J]·Fov[I,F]
#                                  = Σ_F R2[J,F,B]·Fov[I,F]    ≡ DD     (same!)
#    Hmm. Actually `sabej_swap[B,F,E,J] = R2[J,F,B]` is the same as
#    `sabej_direct[F,B,E,J] = R2[J,F,B]`. So Phase 1 vs Phase 2 differ ONLY
#    in the index pattern, but for "FBEJ" vs "BFEJ" contractions on the
#    same physical tensor, the result IS different because the contraction
#    sums a different slot of sabej.
#
#    Let me redo carefully. We have sabej[A,B,E,J] = R2[J,A,B] (canonical
#    storage = "direct"). After .spinad(), the storage becomes the singlet
#    combination: sabej_S[A,B,E,J] = 2·R2[J,A,B] - R2[J,B,A].
#
#       Phase 1 (no spinad) ujaei_raw[E,I,B,J] = Σ_F sabej[F,B,E,J]·Fov[I,F]
#                                              = Σ_F R2[J,F,B]·Fov[I,F]
#       Phase 2 (spinad)    ujaie_raw[E,I,B,J] = Σ_F sabej_S[B,F,E,J]·Fov[I,F]
#                                              = Σ_F (2·R2[J,B,F]-R2[J,F,B])·Fov[I,F]
#       Phase 3 ujaei := 1.5·ujaei_raw - 0.5·ujaie_raw
#                     = Σ_F Fov[I,F] · [1.5·R2[J,F,B] - 0.5·(2·R2[J,B,F]-R2[J,F,B])]
#                     = Σ_F Fov[I,F] · [2·R2[J,F,B] - R2[J,B,F]]
#                     = "PySCF singlet form" (2·direct in F-summed slot - exchange)
#       Phase 5 ujaie := 1/3·ujaei + 2/3·ujaie_raw
#                     = 1/3·Σ Fov·(2R2[J,F,B] - R2[J,B,F]) + 2/3·Σ Fov·(2R2[J,B,F] - R2[J,F,B])
#                     = Σ Fov · [(2/3)R2[J,F,B] - (1/3)R2[J,B,F] + (4/3)R2[J,B,F] - (2/3)R2[J,F,B]]
#                     = Σ Fov · R2[J,B,F]
#                     = "pure exchange form" (R2 with B and F swapped vs direct)
#
#    So in PySCF native form:
#       ujaei_final = (PySCF singlet 2·direct - exchange)  contributions
#       ujaie_final = (pure exchange)                       contributions
#
# 4. Singlet G^{1h1p} matvec uses (2·gmaei - gmaie) per closed-shell
#    convention. Substituting CFOUR form:
#       σ1[I,A_act] += Σ_{J,B} [2·(W_bar_ovvo + ujaei_final·X_EA - umabi·X_MI)
#                              - (W_bar_ovov + ujaie_final·X_EA - umaib·X_MI)] · R1[J,B]
#    where the bar W combination `2·W_bar_ovvo - W_bar_ovov` is the
#    standard PySCF singlet bar-W. The Δ part:
#       Δσ1[I,A_act] = Σ [2·ujaei_final·X_EA - ujaie_final·X_EA] R1 (EA route)
#                    - Σ [2·umabi      ·X_MI - umaib      ·X_MI] R1 (IP route)
#
# 5. Index pattern (PySCF) for U intermediates:
#       ujaei_final[e_root, i, b, j]  (n_act_vir × nocc × nvir × nocc)
#       ujaie_final[e_root, i, b, j]  (same shape, "exchange" partner)
#       umabi_final[a, m_root, b, j]  (nvir × n_act_occ × nvir × nocc)
#       umaib_final[a, m_root, b, j]  (same shape)
#       X_EA[e_root, k_NTO]           (n_act_vir × n_act_vir)
#       X_MI[m_root, k_NTO]           (n_act_occ × n_act_occ)
# ----------------------------------------------------------------------


def build_ujaei(bar_h, r2_ea_list, nocc, nvir):
    """Build ujaei_final + ujaie_final per CFOUR `ujaei_steom_rhf`.

    Canonical CCSD only (R3/triples block skipped).

    SUB-PHASE 3.5/3.6 (2026-05-21): axpby-algebraic form (hypothesis A
    equivalent for single-level singletization at contraction time). The
    explicit hypothesis B literal port (Option A 2026-05-21) was TRIED
    and gave MUCH WORSE results (state 0/1/2 = 0.3145/0.3733/0.5420 Ha,
    gap -120/-127/-50 mHa, vs hypothesis A's -33/-45/+11 mHa) → hypothesis
    B REJECTED. Hypothesis A (this implementation) is closer to correct
    canonical STEOM, even though it has its own caveats for (3) term.

    3 contraction families (CFOUR ↔ PySCF):
      (1) F·S:        sabej·fem    → Fov
      (2) W(vovv)·S:  sabej·wamef  → Wvovv (mapping wamef[E,F,A,M] = Wvovv[A,M,E,F])
      (3) W(ooov)·S:  sabej·wmnie  → Wooov (direct mapping)
    """
    n_act_vir = len(r2_ea_list)
    Fov   = bar_h["Fov"]      # [k, c]   nocc × nvir
    Wvovv = bar_h["Wvovv"]    # [a, l, c, d] nvir × nocc × nvir × nvir
    Wooov = bar_h["Wooov"]    # [k, l, i, d] nocc × nocc × nocc × nvir

    # Build "direct" and "swap" components separately. We compute each
    # contraction with both raw R2 and swapped R2 (first two virtual slots
    # swapped) and combine via axpby chain.
    ujaei_direct = np.zeros((n_act_vir, nocc, nvir, nocc))  # [E,I,B,J] — Phase 1 raw
    ujaei_swap   = np.zeros((n_act_vir, nocc, nvir, nocc))  # Phase 2 swap-only

    for e_root in range(n_act_vir):
        r2 = r2_ea_list[e_root]    # [j, a, b] PySCF EOMEA storage

        # Phase 1 (raw sabej): sabej[A,B,E,J] = R2[J,A,B]
        # (1) Σ_F sabej[F,B,E,J]·Fov[I,F] = Σ_F R2[J,F,B]·Fov[I,F]
        ujaei_direct[e_root] += np.einsum("IF,JFB->IBJ", Fov, r2)
        # (2) Σ_{F,G} sabej[G,F,E,J]·wamef[F,G,B,I] = Σ Wvovv[B,I,F,G]·R2[J,G,F]
        ujaei_direct[e_root] += np.einsum("BIFG,JGF->IBJ", Wvovv, r2)
        # (3) -Σ_{F,N} sabej[F,B,E,N]·wmnie[N,I,J,F] = -Σ Wooov[N,I,J,F]·R2[N,F,B]
        ujaei_direct[e_root] -= np.einsum("NIJF,NFB->IBJ", Wooov, r2)

        # Phase 2 (swap pattern, single-term per-contraction interpretation):
        # (1') Σ_F sabej[B,F,E,J]·Fov[I,F] = Σ R2[J,B,F]·Fov[I,F]
        ujaei_swap[e_root]   += np.einsum("IF,JBF->IBJ", Fov, r2)
        # (2') Σ_{F,G} sabej[G,F,E,J]·wamef[G,F,B,I] = Σ Wvovv[B,I,G,F]·R2[J,G,F]
        ujaei_swap[e_root]   += np.einsum("BIGF,JGF->IBJ", Wvovv, r2)
        # (3') Σ_{F,N} sabej[B,F,E,N]·wmnie[I,N,J,F] = Σ Wooov[I,N,J,F]·R2[N,B,F]
        ujaei_swap[e_root]   += np.einsum("INJF,NBF->IBJ", Wooov, r2)

    # axpby chain: ujaei := 1.5·D - 0.5·(2S - D) = 2D - S  (per F·R2 algebra)
    ujaie_raw = 2.0 * ujaei_swap - ujaei_direct
    ujaei_final = 1.5 * ujaei_direct - 0.5 * ujaie_raw   # = 2D - S
    ujaie_final = (1.0 / 3.0) * ujaei_final + (2.0 / 3.0) * ujaie_raw
    return ujaei_final, ujaie_final


def build_umabi(bar_h, r2_ip_list, nocc, nvir):
    """Build umabi_final + umaib_final per CFOUR `umabi_steom_rhf`.

    Symmetric structure to build_ujaei for S^IP route. Output shape:
    [A_full, M_root, B_full, J_full] = (nvir, n_act_occ, nvir, nocc).

    Hypothesis A (axpby-algebraic single-level singletization). See
    build_ujaei docstring; Option A hypothesis B literal port was tried
    and rejected (gave -120/-127/-50 mHa gap vs hypothesis A's
    -33/-45/+11 mHa).
    """
    n_act_occ = len(r2_ip_list)
    Fov   = bar_h["Fov"]
    Wooov = bar_h["Wooov"]
    Wvovv = bar_h["Wvovv"]

    umabi_direct = np.zeros((nvir, n_act_occ, nvir, nocc))
    umabi_swap   = np.zeros((nvir, n_act_occ, nvir, nocc))

    for m_root in range(n_act_occ):
        r2 = r2_ip_list[m_root]   # [i, j, a] PySCF EOMIP storage
        # smbij[I,J,M,B] = r2[J,I,B] per CFOUR driver "JIB"→"IJB" transpose.

        # Phase 1 (raw smbij):
        umabi_direct[:, m_root, :, :] -= np.einsum("NA,JNB->ABJ", Fov, r2)
        umabi_direct[:, m_root, :, :] += np.einsum("NOJA,NOB->ABJ", Wooov, r2)
        umabi_direct[:, m_root, :, :] -= np.einsum("BNFA,JNF->ABJ", Wvovv, r2)

        # Phase 2 (swap-only single-term interpretation):
        umabi_swap[:, m_root, :, :]   -= np.einsum("NA,NJB->ABJ", Fov, r2)
        umabi_swap[:, m_root, :, :]   += np.einsum("ONJA,NOB->ABJ", Wooov, r2)
        umabi_swap[:, m_root, :, :]   += np.einsum("BNAF,NJF->ABJ", Wvovv, r2)

    # axpby chain (mirror of ujaei)
    umaib_raw = 2.0 * umabi_swap - umabi_direct
    umabi_final = 1.5 * umabi_direct - 0.5 * umaib_raw
    umaib_final = (1.0 / 3.0) * umabi_final + (2.0 / 3.0) * umaib_raw
    return umabi_final, umaib_final


# ----------------------------------------------------------------------
# DIAGNOSTIC: Nooijen-Bartlett 1997 JCP 107, 6812 Eq.(56)-(59) direct port.
#
# This is a "ground truth" implementation of the canonical STEOM phph
# (W^eff_ovvo) intermediates, using the formulas from the original
# Nooijen-Bartlett paper (Section B, Two-particle matrix elements,
# Eq.(56)-(59) for phph block).
#
# Used to verify the CFOUR-based build_umabi/build_ujaei translations
# and identify any convention mismatch.
#
# Nooijen index convention (Eq.(56)):
#   u_{amci}[a, m, c, i]:  a=vir, m=active_occ, c=vir, i=occ
#   - a: "result vir" (output for σ1[i, a])
#   - m: active occ root index (S^IP)
#   - c: "summed vir" (sum index for matvec with R1)
#   - i: "result occ" (output for σ1[i, a])
#
# Tilde notation: $\tilde s^{md}_{il} = 2 s^{md}_{il} - s^{md}_{li}$
#   (singlet spin adaptation, swap of i↔l holes)
#
# Storage in PySCF: s^{ma}_{ik} = r2_ip[m_root][i, k, a]
# ----------------------------------------------------------------------


def build_u_amci_nooijen(bar_h, r2_ip_list, nocc, nvir):
    """Direct port of Nooijen-Bartlett 1997 Eq.(56) — phph S^IP route.

    Returns u_amci[a, m, c, i] in Nooijen index convention.

    Formula:
      u_{amci} = -Σ_k Fov[k,c]·s^{ma}_{ik}
               + Σ_{l,d} Wvovv[a,l,c,d]·tilde_s^{md}_{il}
               - Σ_{l,d} Wvovv[a,l,d,c]·s^{md}_{il}
               + Σ_{k,l} Wooov[k,l,i,c]·s^{ma}_{lk}

    where s^{ma}_{ik} = r2_ip[m][i,k,a] and
    tilde_s^{md}_{il} = 2·r2[i,l,d] - r2[l,i,d].
    """
    n_act_occ = len(r2_ip_list)
    Fov   = bar_h["Fov"]      # [k, c]
    Wvovv = bar_h["Wvovv"]    # [a, l, c, d]  = (ac|ld) chemist = <al|cd> 1212
    Wovoo = bar_h["Wovoo"]    # [k, c, l, i]  = (kc|li) chemist = <kl|ci> 1212

    u_amci = np.zeros((nvir, n_act_occ, nvir, nocc))   # [a, m, c, i]

    for m_root in range(n_act_occ):
        r2 = r2_ip_list[m_root]   # [i, k, a]; s^{ma}_{ik} = r2[i,k,a]

        # Term 1: -Σ_k w_{kc} s^{ma}_{ik} = -Σ_k Fov[k,c]·r2[i,k,a]
        u_amci[:, m_root, :, :] -= np.einsum("kc,ika->aci", Fov, r2)

        # Term 2: +Σ_{l,d} w_{alcd} s̃^{md}_{il} = +Σ Wvovv[a,l,c,d]·(2·r2[i,l,d] - r2[l,i,d])
        u_amci[:, m_root, :, :] += 2.0 * np.einsum("alcd,ild->aci", Wvovv, r2)
        u_amci[:, m_root, :, :] -=       np.einsum("alcd,lid->aci", Wvovv, r2)

        # Term 3: -Σ_{l,d} w_{aldc} s^{md}_{il} = -Σ Wvovv[a,l,d,c]·r2[i,l,d]
        u_amci[:, m_root, :, :] -= np.einsum("aldc,ild->aci", Wvovv, r2)

        # Term 4: +Σ_{k,l} w_{klci} s^{ma}_{lk} = +Σ Wovoo[k,c,l,i]·r2[l,k,a]
        u_amci[:, m_root, :, :] += np.einsum("kcli,lka->aci", Wovoo, r2)

    return u_amci


def build_u_akei_nooijen(bar_h, r2_ea_list, nocc, nvir):
    """Direct port of Nooijen-Bartlett 1997 Eq.(57) — phph S^EA route.

    Returns u_akei[a, k, e, i] in Nooijen index convention.

    Formula:
      u_{akei} = -Σ_k Fov[k,c=?]... no, indices are k (occ-sum), e (active vir).
                More carefully Eq.(57):
      u_{akei} = -Σ_d Fov[k,d]·s^{ad}_{ei}
               + Σ_{l,d} Wooov[l,k,i,d]·tilde_s^{ad}_{el}     (Wooov here = w_{lkdi}?)
               - Σ_{l,d} Wooov[l,k,d,i]·s^{ad}_{el}            (Wooov here = w_{lkid}?)
               + Σ_{c,d} Wvovv[a,k,c,d]·s^{cd}_{ei}             (w_{akcd})

    where s^{ad}_{ei} = r2_ea[e_root][i, a, d] (PySCF EA storage) for active e,
    and tilde_s^{ad}_{el} = 2·r2[l,a,d] - r2[l,d,a].

    Note: w_{kc} in Eq.(57) Term 1 uses index c but the contracted index is d
    (re-read paper: "−Σ_k w_{kc} s^{ac}_{ei}"). Let me re-check via the paper.
    Actually Eq.(57) shows "-Σ_k w_{kc} s^{ac}_{ei}" where output is u_akei,
    so contraction index for Term 1 must be k (sum), with c free in output? But
    output is [a,k,e,i], no c. Wait — looking at the paper Eq.(57) image again:
       u_{akei} = -Σ_k w_{kc} s^{ac}_{ei} + Σ_{l,d} w_{lkdi} tilde_s^{ad}_{el}
                  - Σ_{l,d} w_{lkid} s^{ad}_{el} + Σ_{c,d} w_{akcd} s^{cd}_{ei}
    Output [a, k, e, i]. The c in Term 1 must be the "summed" index — but k
    is also summed.

    Wait, the output of Eq.(57) is u_{akei} but Term 1 is -Σ_k w_{kc} s^{ac}_{ei}
    — this has c in the s amplitude (s^{ac}_{ei}) but no c in output. Re-reading
    the image: it might be Σ_c (not Σ_k), with k a free output index. Let me
    write it as:
      Term 1: -Σ_c w_{kc}·s^{ac}_{ei}   (sum over c, with k, e, i, a as free output)

    Actually no — looking again at Eq.(57): "-Σ_k w_{kc} s^{ac}_{ei}".
    But this leaves c summed too implicitly if both k and c are present. The
    paper image is unclear here. Let me err toward "the sum is over c" since
    k is the output occ slot.

    DEFERRED: implementation TBD after re-reading Eq.(57) closely.
    """
    n_act_vir = len(r2_ea_list)
    Fov   = bar_h["Fov"]
    Wvovv = bar_h["Wvovv"]
    Wooov = bar_h["Wooov"]

    u_akei = np.zeros((nvir, nocc, n_act_vir, nocc))   # [a, k, e, i]
    # Implementation TBD — see docstring.
    return u_akei


def compare_umabi_vs_nooijen(umabi_cfour, u_amci_nooijen, label=""):
    """Compare CFOUR-derived umabi[A,M,B,J] against Nooijen u_amci[a,m,c,i]
    under various candidate index permutations to identify convention bug.
    """
    print(f"\n--- {label} CFOUR vs Nooijen convention check ---")
    print(f"  ‖umabi (CFOUR)‖    = {np.linalg.norm(umabi_cfour):.8f}")
    print(f"  ‖u_amci (Nooijen)‖ = {np.linalg.norm(u_amci_nooijen):.8f}")
    # Candidate permutations:
    # H0: identity (CFOUR slot [0,1,2,3] == Nooijen slot [a,m,c,i])
    # H1: swap virtuals (CFOUR [A,M,B,J] = Nooijen [c,m,a,i] = u_amci[B,M,A,J])
    # H2: rotate (CFOUR [A,M,B,J] = Nooijen [a,m,?,?] with i,c swap)
    # Only permutations that preserve the (nvir, n_act_occ, nvir, nocc) shape
    cands = {
        "H0 identity (A=a, B=c, J=i)":   umabi_cfour,
        "H1 swap_AB    (A=c, B=a, J=i)": umabi_cfour.transpose(2, 1, 0, 3),
    }
    for name, perm in cands.items():
        diff = np.linalg.norm(perm - u_amci_nooijen)
        print(f"  {name}: ‖umabi_perm - u_amci_Nooijen‖ = {diff:.8e}")
    print("  NOTE: norm mismatch (0.061 vs 0.088) → not a pure permutation;")
    print("        CFOUR axpby translation diverges from canonical Eq.(56).")
    print("        Nooijen bar-W mapping (w_klci = oovo) needs careful PySCF")
    print("        correspondence work — u_amci_Nooijen itself unverified.")


# ----------------------------------------------------------------------
# CANONICAL Nooijen-Bartlett 1997 phph (Eq.56-59) + phhp (Eq.60-63)
# implementation, using the verified w→PySCF mapping table:
#   w_kc   = Fov[k,c]
#   w_klid = Wooov[k,l,i,d]   (= (ki|ld) chemist)
#   w_alcd = Wvovv[a,l,c,d]   (= (ac|ld))
#   w_klci = Wovoo[k,c,l,i]   (= (kc|li))
#   w_akci = Wovov[k,a,i,c]   (phph bare, exchange-like (ac|ki))
#   w_bkjc = Wovvo[k,b,c,j]   (phhp bare, Coulomb-like (kc|bj))
# s storage: s^{ma}_{ik}=r2_ip[m][i,k,a],  s^{ab}_{ej}=r2_ea[e][j,a,b]
# tilde: s̃^{md}_{il}=2 r2_ip[m][i,l,d]-r2_ip[m][l,i,d] (IP hole swap)
#        s̃^{ad}_{el}=2 r2_ea[e][l,a,d]-r2_ea[e][l,d,a] (EA particle swap)
#
# NOTE: cross terms (Eq.58 u_amei, Eq.62 u_bmje) NOT yet included — this
# is the "canonical no-cross" baseline. Cross requires hp (Eq.38-40) +
# hhhp (Eq.42-44) intermediates.
# ----------------------------------------------------------------------


def build_u_akei_canonical(bar_h, r2_ea_list, nocc, nvir):
    """Nooijen Eq.(57) — phph S^EA route. Returns u_akei[a, k, e, i]."""
    n_act_vir = len(r2_ea_list)
    Fov   = bar_h["Fov"]
    Wvovv = bar_h["Wvovv"]
    Wooov = bar_h["Wooov"]
    Wovoo = bar_h["Wovoo"]

    u_akei = np.zeros((nvir, nocc, n_act_vir, nocc))   # [a, k, e, i]

    for e_root in range(n_act_vir):
        r2 = r2_ea_list[e_root]   # [j, a, b]; s^{ac}_{ei}=r2[i,a,c]

        # T1: -Σ_c Fov[k,c]·r2[i,a,c]
        u_akei[:, :, e_root, :] -= np.einsum("kc,iac->aki", Fov, r2)
        # T2: +Σ_{l,d} Wovoo[l,d,k,i]·(2 r2[l,a,d] - r2[l,d,a])
        u_akei[:, :, e_root, :] += 2.0 * np.einsum("ldki,lad->aki", Wovoo, r2)
        u_akei[:, :, e_root, :] -=       np.einsum("ldki,lda->aki", Wovoo, r2)
        # T3: -Σ_{l,d} Wooov[l,k,i,d]·r2[l,a,d]
        u_akei[:, :, e_root, :] -= np.einsum("lkid,lad->aki", Wooov, r2)
        # T4: +Σ_{c,d} Wvovv[a,k,c,d]·r2[i,c,d]
        u_akei[:, :, e_root, :] += np.einsum("akcd,icd->aki", Wvovv, r2)

    return u_akei


def build_u_bmjc_canonical(bar_h, r2_ip_list, nocc, nvir):
    """Nooijen Eq.(60) — phhp S^IP route. Returns u_bmjc[b, m, j, c].

    Eq.(60) (verified from 300 DPI crop):
      u_{bmjc} = -Σ_k w_{kc} s^{mb}_{kj} + Σ_{k,l} w_{klcj} s^{mb}_{kl}
               - Σ_{k,d} w_{kbcd} s^{md}_{kj}

    Mapping:
      w_kc   = Fov[k,c]
      w_klcj = Wovoo[k,c,l,j]   (= (kc|lj) chemist = <kl|cj> 1212)
      w_kbcd = Wvovv[b,k,d,c]   (= (bd|kc) chemist = <kb|cd> 1212)
    s storage: s^{mb}_{kj}=r2_ip[m][k,j,b], s^{mb}_{kl}=r2_ip[m][k,l,b],
               s^{md}_{kj}=r2_ip[m][k,j,d]
    """
    n_act_occ = len(r2_ip_list)
    Fov   = bar_h["Fov"]
    Wovoo = bar_h["Wovoo"]
    Wvovv = bar_h["Wvovv"]

    u_bmjc = np.zeros((nvir, n_act_occ, nocc, nvir))   # [b, m, j, c]

    for m_root in range(n_act_occ):
        r2 = r2_ip_list[m_root]   # [i, k, a]

        # T1: -Σ_k Fov[k,c]·r2[k,j,b]
        u_bmjc[:, m_root, :, :] -= np.einsum("kc,kjb->bjc", Fov, r2)
        # T2: +Σ_{k,l} Wovoo[k,c,l,j]·r2[k,l,b]
        u_bmjc[:, m_root, :, :] += np.einsum("kclj,klb->bjc", Wovoo, r2)
        # T3: -Σ_{k,d} Wvovv[b,k,d,c]·r2[k,j,d]
        u_bmjc[:, m_root, :, :] -= np.einsum("bkdc,kjd->bjc", Wvovv, r2)

    return u_bmjc


def build_u_bkje_canonical(bar_h, r2_ea_list, nocc, nvir):
    """Nooijen Eq.(61) — phhp S^EA route. Returns u_bkje[b, k, j, e].

    Eq.(61):
      u_{bkje} = Σ_d w_{kd} s^{db}_{ej} + Σ_{c,d} w_{bkdc} s^{cd}_{ej}
               - Σ_{l,d} w_{lkjd} s^{db}_{el}

    Mapping:
      w_kd  = Fov[k,d]
      w_bkdc (1212)=<bk|dc>=(bd|kc) chemist. b,d=vir(e1); k,c=occ?,vir(e2).
             (bd|kc): b,d vir; k occ, c vir. (vv|ov). = Wvovv[b,k,d,c]?
             Wvovv[a,l,c,d]=(ac|ld). For (bd|kc): match a→b,c→d,l→k,d→c:
             Wvovv[b,k,d,c]=(bd|kc). ✓ So w_bkdc=Wvovv[b,k,d,c].
      w_lkjd (1212)=<lk|jd>=(lj|kd) chemist. l,j=occ(e1); k,d=occ,vir(e2).
             (lj|kd)=(oo|ov). = Wooov[l,k,j,d]? Wooov[k,l,i,d]=(ki|ld).
             For (lj|kd): match k→l,i→j,l→k,d→d: Wooov[l,k,j,d]=(lj|kd). ✓
      s^{db}_{ej} = r2_ea[e][j,d,b];  s^{db}_{el}=r2_ea[e][l,d,b]
    """
    n_act_vir = len(r2_ea_list)
    Fov   = bar_h["Fov"]
    Wvovv = bar_h["Wvovv"]
    Wooov = bar_h["Wooov"]

    u_bkje = np.zeros((nvir, nocc, nocc, n_act_vir))   # [b, k, j, e]

    for e_root in range(n_act_vir):
        r2 = r2_ea_list[e_root]   # [j, a, b]

        # T1: +Σ_d Fov[k,d]·s^{db}_{ej} = +Σ_d Fov[k,d]·r2[j,d,b]
        u_bkje[:, :, :, e_root] += np.einsum("kd,jdb->bkj", Fov, r2)
        # T2: +Σ_{c,d} Wvovv[b,k,d,c]·s^{cd}_{ej} = +Σ Wvovv[b,k,d,c]·r2[j,c,d]
        u_bkje[:, :, :, e_root] += np.einsum("bkdc,jcd->bkj", Wvovv, r2)
        # T3: -Σ_{l,d} Wooov[l,k,j,d]·s^{db}_{el} = -Σ Wooov[l,k,j,d]·r2[l,d,b]
        u_bkje[:, :, :, e_root] -= np.einsum("lkjd,ldb->bkj", Wooov, r2)

    return u_bkje


def build_g_phph_phhp_canonical(
    bar_h, r2_ip_list, r2_ea_list,
    F_eff_oo, F_eff_vv, X_IP, X_EA,
    active_occ_idx, active_vir_idx, nocc, nvir,
):
    """Assemble canonical G^{1h1p} singlet matrix from Nooijen Eq.(56)-(63),
    NO-CROSS baseline (Eq.58/62 cross terms omitted).

    G^{1h1p}[ia, jb] = F_eff_vv[a,b]·δ_ij - F_eff_oo[i,j]·δ_ab
                     + 2·g_phhp[Coulomb] - g_phph[exchange]

    phph g_akci[a,k,c,i] = Wovov[k,a,i,c]
                         + (S^IP: Σ_m u_amci·X_IP at active k)
                         + (S^EA: Σ_e u_akei·X_EA at active c)
    phhp g_bkjc[b,k,j,c] = Wovvo[k,b,c,j]
                         + (S^IP: u_bmjc·X_IP at active k)
                         + (S^EA: Σ_e u_bkje·X_EA at active c)
    """
    n_act_occ = len(active_occ_idx)
    n_act_vir = len(active_vir_idx)
    Wovov = bar_h["Wovov"]   # [k,b,i,d] = (ki|bd)
    Wovvo = bar_h["Wovvo"]   # [k,b,c,j]

    # --- phph: u_amci (S^IP) + u_akei (S^EA) ---
    u_amci = build_u_amci_nooijen(bar_h, r2_ip_list, nocc, nvir)   # [a,m,c,i]
    u_akei = build_u_akei_canonical(bar_h, r2_ea_list, nocc, nvir)  # [a,k,e,i]

    # g_phph[a, k, c, i] base = Wovov[k,a,i,c]
    g_phph = np.einsum("kaic->akci", Wovov).copy()

    # s-normalization signs per Nooijen Eq.(29)-(30):
    #   S^IP: s = -Σ_λ r(λ)·r⁻¹_{λm}  (extra minus, hole-line contraction)
    #   S^EA: s = +Σ_λ r(λ)·r⁻¹_{λe}  (plus)
    IP_SIGN = -1.0
    EA_SIGN = +1.0
    # S^IP route: active-k rows.  Σ_m u_amci[a,m,c,i]·X_IP[m, k_NTO] for k=active_occ_idx[k_NTO]
    for k_NTO in range(n_act_occ):
        k_full = active_occ_idx[k_NTO]
        g_phph[:, k_full, :, :] += IP_SIGN * np.einsum("amci,m->aci", u_amci, X_IP[:, k_NTO])
    # S^EA route: active-c cols.  Σ_e u_akei[a,k,e,i]·X_EA[e, c_NTO] for c=active_vir_idx[c_NTO]
    for c_NTO in range(n_act_vir):
        c_full = active_vir_idx[c_NTO]
        g_phph[:, :, c_full, :] += EA_SIGN * np.einsum("akei,e->aki", u_akei, X_EA[:, c_NTO])

    # --- phhp: u_bkje (S^EA); u_bmjc (S^IP, DEFERRED=0) ---
    u_bkje = build_u_bkje_canonical(bar_h, r2_ea_list, nocc, nvir)  # [b,k,j,e]
    u_bmjc = build_u_bmjc_canonical(bar_h, r2_ip_list, nocc, nvir)  # [b,m,j,c] (zeros)

    # g_phhp[b, k, j, c] base = Wovvo[k,b,c,j]
    g_phhp = np.einsum("kbcj->bkjc", Wovvo).copy()
    # S^EA route: active-c.  Σ_e u_bkje[b,k,j,e]·X_EA[e, c_NTO]
    for c_NTO in range(n_act_vir):
        c_full = active_vir_idx[c_NTO]
        g_phhp[:, :, :, c_full] += EA_SIGN * np.einsum("bkje,e->bkj", u_bkje, X_EA[:, c_NTO])
    # S^IP route: active-k.  Σ_m u_bmjc[b,m,j,c]·X_IP[m, k_NTO]
    for k_NTO in range(n_act_occ):
        k_full = active_occ_idx[k_NTO]
        g_phhp[:, k_full, :, :] += IP_SIGN * np.einsum("bmjc,m->bjc", u_bmjc, X_IP[:, k_NTO])

    # --- Assemble G^{1h1p} singlet ---
    dim = nocc * nvir
    G = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            row = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    col = j * nvir + b
                    val = 0.0
                    if i == j:
                        val += F_eff_vv[a, b]
                    if a == b:
                        val -= F_eff_oo[i, j]
                    # 2·Coulomb(phhp) - exchange(phph), matching bar-W baseline
                    #   2·Wovvo[j,b,a,i] - Wovov[j,a,i,b]
                    # g_phhp[b,k,j,c]=Wovvo[k,b,c,j] → Wovvo[j,b,a,i]=g_phhp[b,j,i,a]
                    # g_phph[a,k,c,i]=Wovov[k,a,i,c] → Wovov[j,a,i,b]=g_phph[a,j,b,i]
                    val += 2.0 * g_phhp[b, j, i, a]
                    val -=       g_phph[a, j, b, i]
                    G[row, col] += val
    return G, g_phph, g_phhp, u_amci, u_akei, u_bkje


# ======================================================================
# CANONICAL FULL (with cross): pre-normalized s amplitudes + Eq.(34)-(63).
#
# Normalized s (Eq.29-30): the sign convention (IP minus / EA plus) and the
# active-root inverse are baked into s_IP/s_EA, so all u/g formulas use them
# directly with simple δ selection (no separate X application).
#   s_IP[m,i,j,b] = -Σ_λ r2_ip[λ][i,j,b]·rinv_IP[λ,m]
#   s_EA[e,i,a,b] = +Σ_λ r2_ea[λ][i,a,b]·rinv_EA[λ,e]
# rinv = inv(R1mat), R1mat[λ,m]=r1[λ][active_idx[m]].
#
# Tilde (symmetrized) per Eq.33:
#   IP: s̃_IP[m,i,j,b] = 2 s_IP[m,i,j,b] - s_IP[m,j,i,b]  (hole swap)
#   EA: s̃_EA[e,i,a,b] = 2 s_EA[e,i,a,b] - s_EA[e,i,b,a]  (particle swap)
# ======================================================================


def build_normalized_s(r2_ip_list, r2_ea_list, r1_ip_list, r1_ea_list,
                        active_occ_idx, active_vir_idx, nocc, nvir):
    n_act_occ = len(r2_ip_list)
    n_act_vir = len(r2_ea_list)

    # R1mat[λ, m] = r1[λ][active_idx[m]];  rinv = inv(R1mat)
    R1mat_IP = np.array([[r1_ip_list[lam][active_occ_idx[m]]
                          for m in range(n_act_occ)]
                         for lam in range(n_act_occ)])
    rinv_IP = np.linalg.inv(R1mat_IP)
    R1mat_EA = np.array([[r1_ea_list[lam][active_vir_idx[e]]
                          for e in range(n_act_vir)]
                         for lam in range(n_act_vir)])
    rinv_EA = np.linalg.inv(R1mat_EA)

    # s_IP[m,i,j,b] = -Σ_λ r2_ip[λ][i,j,b]·rinv_IP[λ,m]
    s_IP = np.zeros((n_act_occ, nocc, nocc, nvir))
    for m in range(n_act_occ):
        for lam in range(n_act_occ):
            s_IP[m] -= r2_ip_list[lam] * rinv_IP[lam, m]
    # s_EA[e,i,a,b] = +Σ_λ r2_ea[λ][i,a,b]·rinv_EA[λ,e]
    s_EA = np.zeros((n_act_vir, nocc, nvir, nvir))
    for e in range(n_act_vir):
        for lam in range(n_act_vir):
            s_EA[e] += r2_ea_list[lam] * rinv_EA[lam, e]
    return s_IP, s_EA


def build_g_canonical_full(
    bar_h, r2_ip_list, r2_ea_list, r1_ip_list, r1_ea_list,
    active_occ_idx, active_vir_idx, nocc, nvir,
):
    """Full canonical STEOM G^{1h1p} with cross terms (Eq.56-63), using
    pre-normalized s amplitudes (Eq.29-30 signs baked in). F^eff_oo/vv
    (Eq.34-37) rebuilt internally with normalized s for consistency.
    """
    n_act_occ = len(active_occ_idx)
    n_act_vir = len(active_vir_idx)
    Fov   = bar_h["Fov"]
    Wvovv = bar_h["Wvovv"]
    Wooov = bar_h["Wooov"]
    Wovoo = bar_h["Wovoo"]
    Wovov = bar_h["Wovov"]
    Wovvo = bar_h["Wovvo"]
    eri_ovov = bar_h["eri_ovov"]

    Loo = bar_h["Loo"]
    Lvv = bar_h["Lvv"]

    s_IP, s_EA = build_normalized_s(
        r2_ip_list, r2_ea_list, r1_ip_list, r1_ea_list,
        active_occ_idx, active_vir_idx, nocc, nvir)

    # ---------- F^eff_oo (Eq.34-35) + F^eff_vv (Eq.36-37) with normalized s ----------
    # (rebuilt consistently with W^eff to avoid normalization mismatch)
    F_eff_oo = Loo.copy()
    for m in range(n_act_occ):
        s = s_IP[m]                                  # [i,k,c]
        st = 2.0 * s - s.transpose(1, 0, 2)          # s̃_IP[m][i,k,c]
        # u_mi[i] = Σ_{k,c} Fov[k,c] s̃[i,k,c] - Σ_{k,l,d} Wooov[k,l,i,d] s̃[k,l,d]
        u_mi = np.einsum("kc,ikc->i", Fov, st) \
             - np.einsum("klid,kld->i", Wooov, st)
        F_eff_oo[active_occ_idx[m], :] += u_mi
    F_eff_vv = Lvv.copy()
    for e in range(n_act_vir):
        s = s_EA[e]                                  # [k,a,c]
        st = 2.0 * s - s.transpose(0, 2, 1)          # s̃_EA[e][k,a,c]
        # u_ae[a] = Σ_{k,c} Fov[k,c] s̃[k,a,c] + Σ_{c,d,l} Wvovv[a,l,c,d] s̃[l,c,d]
        u_ae = np.einsum("kc,kac->a", Fov, st) \
             + np.einsum("alcd,lcd->a", Wvovv, st)
        F_eff_vv[active_vir_idx[e], :] += u_ae

    # ---------- hp intermediates (Eq.38-39) ----------
    # u_ma[m,a] = -Σ_{k,l,d} Wovvo[k,d,a,l]·s̃_IP[m][k,l,d]
    # u_ie[i,e] = +Σ_{c,d,l} Wovvo[i,d,c,l]·s̃_EA[e][l,c,d]
    u_ma = np.zeros((n_act_occ, nvir))
    for m in range(n_act_occ):
        st = 2.0 * s_IP[m] - s_IP[m].transpose(1, 0, 2)   # s̃_IP[m][k,l,d]
        u_ma[m] = -np.einsum("kdal,kld->a", Wovvo, st)
    u_ie = np.zeros((nocc, n_act_vir))
    for e in range(n_act_vir):
        st = 2.0 * s_EA[e] - s_EA[e].transpose(0, 2, 1)   # s̃_EA[e][l,c,d]
        u_ie[:, e] = np.einsum("idcl,lcd->i", Wovvo, st)

    # ---------- hhhp intermediates (Eq.42-44, bare v=eri_ovov) ----------
    # u_mlid[m,l,i,d] = Σ_{j,b}(eri_ovov[j,b,l,d]·s̃_IP[m][i,j,b] - eri_ovov[l,b,j,d]·s_IP[m][i,j,b])
    u_mlid = np.zeros((n_act_occ, nocc, nocc, nvir))
    for m in range(n_act_occ):
        st = 2.0 * s_IP[m] - s_IP[m].transpose(1, 0, 2)   # s̃_IP[m][i,j,b]
        u_mlid[m] = np.einsum("jbld,ijb->lid", eri_ovov, st) \
                  - np.einsum("lbjd,ijb->lid", eri_ovov, s_IP[m])
    # u_kmid[k,m,i,d] = -Σ_{j,b} eri_ovov[j,d,k,b]·s_IP[m][j,i,b]
    u_kmid = np.zeros((nocc, n_act_occ, nocc, nvir))
    for m in range(n_act_occ):
        u_kmid[:, m] = -np.einsum("jdkb,jib->kid", eri_ovov, s_IP[m])
    # u_klie[k,l,i,e] = Σ_{a,b} eri_ovov[k,a,l,b]·s_EA[e][i,a,b]
    u_klie = np.zeros((nocc, nocc, nocc, n_act_vir))
    for e in range(n_act_vir):
        u_klie[:, :, :, e] = np.einsum("kalb,iab->kli", eri_ovov, s_EA[e])

    # ---------- phph intermediates (Eq.56-58) ----------
    # u_amci[a,m,c,i] (Eq.56, S^IP)
    u_amci = np.zeros((nvir, n_act_occ, nvir, nocc))
    for m in range(n_act_occ):
        s = s_IP[m]                                  # [i,k,a]
        st = 2.0 * s - s.transpose(1, 0, 2)          # s̃_IP[m][i,l,d]
        u_amci[:, m, :, :] = -np.einsum("kc,ika->aci", Fov, s) \
            + np.einsum("alcd,ild->aci", Wvovv, st) \
            - np.einsum("aldc,ild->aci", Wvovv, s) \
            + np.einsum("kcli,lka->aci", Wovoo, s)
    # u_akei[a,k,e,i] (Eq.57, S^EA)
    u_akei = np.zeros((nvir, nocc, n_act_vir, nocc))
    for e in range(n_act_vir):
        s = s_EA[e]                                  # [i,a,c]
        st = 2.0 * s - s.transpose(0, 2, 1)          # s̃_EA[e][l,a,d]
        u_akei[:, :, e, :] = -np.einsum("kc,iac->aki", Fov, s) \
            + np.einsum("ldki,lad->aki", Wovoo, st) \
            - np.einsum("lkid,lad->aki", Wooov, s) \
            + np.einsum("akcd,icd->aki", Wvovv, s)
    # u_amei[a,m,e,i] (Eq.58, cross)
    u_amei = np.zeros((nvir, n_act_occ, n_act_vir, nocc))
    for m in range(n_act_occ):
        sIP = s_IP[m]                                # [i,k,a]
        for e in range(n_act_vir):
            sEA = s_EA[e]                            # [i,a,c]
            stEA = 2.0 * sEA - sEA.transpose(0, 2, 1)  # s̃_EA[e][l,a,d]
            # T1 +Σ_c u_ma[m,c]·sEA[i,a,c]
            t = np.einsum("c,iac->ai", u_ma[m], sEA)
            # T2 -Σ_k u_ie[k,e]·sIP[i,k,a]
            t -= np.einsum("k,ika->ai", u_ie[:, e], sIP)
            # T3 +Σ_{l,d} u_mlid[m,l,i,d]·stEA[l,a,d]
            t += np.einsum("lid,lad->ai", u_mlid[m], stEA)
            # T4 -Σ_{l,d} u_kmid[l,m,i,d]·sEA[l,a,d]
            t -= np.einsum("lid,lad->ai", u_kmid[:, m], sEA)
            # T5 +Σ_{k,l} u_klie[k,l,i,e]·sIP[l,k,a]
            t += np.einsum("kli,lka->ai", u_klie[:, :, :, e], sIP)
            u_amei[:, m, e, :] = t

    # ---------- phhp intermediates (Eq.60-62) ----------
    # u_bmjc[b,m,j,c] (Eq.60, S^IP)
    u_bmjc = np.zeros((nvir, n_act_occ, nocc, nvir))
    for m in range(n_act_occ):
        s = s_IP[m]                                  # [k,j,b]
        u_bmjc[:, m, :, :] = -np.einsum("kc,kjb->bjc", Fov, s) \
            + np.einsum("kclj,klb->bjc", Wovoo, s) \
            - np.einsum("bkdc,kjd->bjc", Wvovv, s)
    # u_bkje[b,k,j,e] (Eq.61, S^EA)
    u_bkje = np.zeros((nvir, nocc, nocc, n_act_vir))
    for e in range(n_act_vir):
        s = s_EA[e]                                  # [j,d,b]
        u_bkje[:, :, :, e] = np.einsum("kd,jdb->bkj", Fov, s) \
            + np.einsum("bkdc,jcd->bkj", Wvovv, s) \
            - np.einsum("lkjd,ldb->bkj", Wooov, s)
    # u_bmje[b,m,j,e] (Eq.62, cross)
    u_bmje = np.zeros((nvir, n_act_occ, nocc, n_act_vir))
    for m in range(n_act_occ):
        sIP = s_IP[m]                                # [k,j,b]
        for e in range(n_act_vir):
            sEA = s_EA[e]                            # [j,d,b]
            # T1 +Σ_d u_ma[m,d]·sEA[j,d,b]
            t = np.einsum("d,jdb->bj", u_ma[m], sEA)
            # T2 -Σ_k u_ie[k,e]·sIP[k,j,b]
            t -= np.einsum("k,kjb->bj", u_ie[:, e], sIP)
            # T3 +Σ_{k,l} u_klie[k,l,j,e]·sIP[k,l,b]
            t += np.einsum("klj,klb->bj", u_klie[:, :, :, e], sIP)
            # T4 -Σ_{l,d} u_mlid[m,l,j,d]·sEA[l,d,b]
            t -= np.einsum("ljd,ldb->bj", u_mlid[m], sEA)
            u_bmje[:, m, :, e] = t

    # ---------- assemble g_phph (Eq.59) + g_phhp (Eq.63) ----------
    g_phph = np.einsum("kaic->akci", Wovov).copy()   # [a,k,c,i]
    for m in range(n_act_occ):
        k_full = active_occ_idx[m]
        g_phph[:, k_full, :, :] += u_amci[:, m, :, :]          # δ_mk u_amci
    for e in range(n_act_vir):
        c_full = active_vir_idx[e]
        g_phph[:, :, c_full, :] += u_akei[:, :, e, :]          # δ_ce u_akei
    for m in range(n_act_occ):
        k_full = active_occ_idx[m]
        for e in range(n_act_vir):
            c_full = active_vir_idx[e]
            g_phph[:, k_full, c_full, :] += u_amei[:, m, e, :]  # δ_ec δ_km u_amei

    g_phhp = np.einsum("kbcj->bkjc", Wovvo).copy()   # [b,k,j,c]
    for m in range(n_act_occ):
        k_full = active_occ_idx[m]
        g_phhp[:, k_full, :, :] += u_bmjc[:, m, :, :]          # δ_km u_bmjc
    for e in range(n_act_vir):
        c_full = active_vir_idx[e]
        g_phhp[:, :, :, c_full] += u_bkje[:, :, :, e]          # δ_ce u_bkje
    for m in range(n_act_occ):
        k_full = active_occ_idx[m]
        for e in range(n_act_vir):
            c_full = active_vir_idx[e]
            g_phhp[:, k_full, :, c_full] += u_bmje[:, m, :, e]  # cross

    # ---------- G^{1h1p} singlet ----------
    dim = nocc * nvir
    G = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            row = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    col = j * nvir + b
                    val = 0.0
                    if i == j:
                        val += F_eff_vv[a, b]
                    if a == b:
                        val -= F_eff_oo[i, j]
                    val += 2.0 * g_phhp[b, j, i, a]
                    val -=       g_phph[a, j, b, i]
                    G[row, col] += val
    return G, g_phph, g_phhp, u_amei, u_bmje


# ----------------------------------------------------------------------
# Sub-phase 3.6 — Cross IP×EA dressing (CFOUR `uei` + `uam` + `uijke`
# + `uajim` + `umaei` + cross gmaei dressing).
#
# Index mapping (recap):
#   wmnef[A,B,I,J] = <Ab|Ij>  ≡  PySCF eri_ovov[I,A,J,B]   (chemist (AI|BJ))
#   sabej[A,B,E,J] = R2_EA[E][J,A,B]
#   smbij[I,J,M,B] = R2_IP[M][J,I,B]
# ----------------------------------------------------------------------


def build_uei_uam(bar_h, r2_ea_list, r2_ip_list, nocc, nvir):
    """Build uei[E,I] (S^EA, CFOUR `uei_steom_rhf`) and uam[A,M] (S^IP,
    CFOUR `uam_steom_rhf`). Single contraction each, spinad'd wmnef.

    PySCF native singlet form:
      uei[E,I]   = +2·Σ eri_ovov[I,G,M,F]·R2_EA[E][M,G,F]  -  Σ eri_ovov[I,F,M,G]·R2_EA[E][M,G,F]
      uam[A,M]   = -2·Σ eri_ovov[O,A,N,E]·R2_IP[M][N,O,E]  +  Σ eri_ovov[O,E,N,A]·R2_IP[M][N,O,E]
    """
    n_act_vir = len(r2_ea_list)
    n_act_occ = len(r2_ip_list)
    eri_ovov = bar_h["eri_ovov"]   # [k,b,i,d]  (o,v,o,v)

    uei = np.zeros((n_act_vir, nocc))
    for e_root, r2 in enumerate(r2_ea_list):    # r2[J,G,F]
        # +2·Σ eri_ovov[I,G,M,F]·r2[M,G,F]
        uei[e_root] += 2.0 * np.einsum("IGMF,MGF->I", eri_ovov, r2)
        # -Σ eri_ovov[I,F,M,G]·r2[M,G,F]
        uei[e_root] -=       np.einsum("IFMG,MGF->I", eri_ovov, r2)

    uam = np.zeros((nvir, n_act_occ))
    for m_root, r2 in enumerate(r2_ip_list):    # r2[i,j,a]
        # -2·Σ eri_ovov[O,A,N,E]·smbij[O,N,M,E] with smbij[O,N,M,E] = r2[N,O,E]
        uam[:, m_root] -= 2.0 * np.einsum("OANE,NOE->A", eri_ovov, r2)
        uam[:, m_root] +=       np.einsum("OENA,NOE->A", eri_ovov, r2)

    return uei, uam


def build_uijke_uajim(bar_h, r2_ea_list, r2_ip_list, nocc, nvir):
    """Build uijke[I,J,K,E] (CFOUR `uijke_steom_rhf`, single non-spinad
    contraction) and uajim/uajmi[A,I,M,J] (CFOUR `uajim_steom_rhf`,
    2 contractions + axpby averaging).
    """
    n_act_vir = len(r2_ea_list)
    n_act_occ = len(r2_ip_list)
    eri_ovov = bar_h["eri_ovov"]

    # ----- uijke[I, J, K, E] = Σ_{G,F} eri_ovov[J,G,I,F]·R2_EA[E][K,G,F]
    uijke = np.zeros((nocc, nocc, nocc, n_act_vir))
    for e_root, r2 in enumerate(r2_ea_list):    # r2[K,G,F]
        uijke[:, :, :, e_root] = np.einsum("JGIF,KGF->IJK", eri_ovov, r2)

    # ----- uajim, uajmi: [A, I, M, J] shape (nvir, nocc, n_act_occ, nocc)
    # CFOUR contraction (a): wmnef_S · smbij_S (both spinad'd) → uajim
    # CFOUR contraction (b): wmnef · smbij (raw) → uajmi, sign -1
    # Then axpby(0.5, uajmi, 0.5, uajim): uajim := 0.5·uajmi + 0.5·uajim
    #
    # We compute via 4-term explicit expansion for (a) (under hypothesis B),
    # and single-term for (b).
    uajim = np.zeros((nvir, nocc, n_act_occ, nocc))   # [A, I, M, J]
    uajmi = np.zeros((nvir, nocc, n_act_occ, nocc))

    for m_root, r2 in enumerate(r2_ip_list):    # r2[i,j,a]
        # smbij[J,N,M,E] = r2[N,J,E];  smbij[N,J,M,E] = r2[J,N,E]
        # smbij_S[J,N,M,E] = 2·smbij[J,N,M,E] - smbij[N,J,M,E] = 2·r2[N,J,E] - r2[J,N,E]
        # wmnef[A,E,I,N] = eri_ovov[I,A,N,E]
        # wmnef_S[A,E,I,N] = 2·eri_ovov[I,A,N,E] - eri_ovov[I,E,N,A]
        #
        # (a) Σ_{E,N} wmnef_S[A,E,I,N] · smbij_S[J,N,M,E]
        #   = Σ (2·eri_ovov[I,A,N,E] - eri_ovov[I,E,N,A]) · (2·r2[N,J,E] - r2[J,N,E])
        #   = 4·Σ eri_ovov[I,A,N,E]·r2[N,J,E]
        #     -2·Σ eri_ovov[I,A,N,E]·r2[J,N,E]
        #     -2·Σ eri_ovov[I,E,N,A]·r2[N,J,E]
        #     +1·Σ eri_ovov[I,E,N,A]·r2[J,N,E]
        uajim[:, :, m_root, :] += 4.0 * np.einsum("IANE,NJE->AIJ", eri_ovov, r2)
        uajim[:, :, m_root, :] -= 2.0 * np.einsum("IANE,JNE->AIJ", eri_ovov, r2)
        uajim[:, :, m_root, :] -= 2.0 * np.einsum("IENA,NJE->AIJ", eri_ovov, r2)
        uajim[:, :, m_root, :] +=       np.einsum("IENA,JNE->AIJ", eri_ovov, r2)

        # (b) -Σ_{E,N} wmnef[A,E,N,I] · smbij[N,J,M,E]
        #     wmnef[A,E,N,I] = eri_ovov[N,A,I,E]
        #     smbij[N,J,M,E] = r2[J,N,E]
        #     = -Σ eri_ovov[N,A,I,E]·r2[J,N,E]
        uajmi[:, :, m_root, :] -= np.einsum("NAIE,JNE->AIJ", eri_ovov, r2)

    # Final averaging: uajim := 0.5·uajmi + 0.5·uajim
    uajim_avg = 0.5 * uajim + 0.5 * uajmi
    # uajmi unchanged
    return uijke, uajim_avg, uajmi


def build_umaei(bar_h, uei, uam, uijke, uajim, uajmi,
                r2_ea_list, r2_ip_list, nocc, nvir):
    """Build umaei + umaie per CFOUR `umaei_steom_rhf` (cross IP×EA, 8
    contractions + 2 axpby).

    Output shape: [E_root, M_root, B_full, J_full] = (n_act_vir, n_act_occ, nvir, nocc).

    Contractions (signs as per CFOUR source):
      (1) +Σ_F sabej[F,B,E,J]·uam[F,M]              → umaei
      (2) -Σ_{F,N} sabej[F,B,E,N]·uajmi[F,N,M,J]    → umaei
      (3) +Σ_F sabej_S[B,F,E,J]·uam[F,M]            → umaie
      (4) +Σ_{F,N} sabej_S[B,F,E,N]·tmp[F,N,M,J]    → umaie  (tmp = 2·uajim - uajmi)
      (5) -Σ_N smbij[N,J,M,B]·uei[E,N]              → umaei
      (6) +Σ_{O,N} smbij[O,N,M,B]·uijke[N,O,J,E]    → umaei
      (7) -Σ_N smbij_S[J,N,M,B]·uei[E,N]            → umaie
      (8) +Σ_{O,N} smbij_S[O,N,M,B]·uijke[O,N,J,E]  → umaie
    Then:
      axpby(-0.5, umaie, 1.5, umaei):  umaei := 1.5·umaei - 0.5·umaie
      axpby(1/3,  umaei, 2/3,  umaie): umaie := 1/3·umaei + 2/3·umaie
    """
    n_act_vir = len(r2_ea_list)
    n_act_occ = len(r2_ip_list)

    umaei = np.zeros((n_act_vir, n_act_occ, nvir, nocc))   # [E,M,B,J]
    umaie = np.zeros((n_act_vir, n_act_occ, nvir, nocc))

    # Pre-compute (2·uajim - uajmi) for (4)
    tmp_2uajim_minus_uajmi = 2.0 * uajim - uajmi   # [A,I,M,J]

    for e_root, r2_ea in enumerate(r2_ea_list):    # r2_ea[J,A,B]
        for m_root, r2_ip in enumerate(r2_ip_list):  # r2_ip[I,J,A]

            # (1) +Σ_F r2_ea[J,F,B]·uam[F, m_root]
            umaei[e_root, m_root] += np.einsum("JFB,F->BJ", r2_ea, uam[:, m_root])

            # (2) -Σ_{F,N} r2_ea[N,F,B]·uajmi[F,N, m_root, J]
            umaei[e_root, m_root] -= np.einsum("NFB,FNJ->BJ", r2_ea, uajmi[:, :, m_root, :])

            # (3) +Σ_F sabej_S[B,F,E,J]·uam[F,M]
            #     sabej_S[B,F,E,J] = 2·r2_ea[J,B,F] - r2_ea[J,F,B]
            umaie[e_root, m_root] += 2.0 * np.einsum("JBF,F->BJ", r2_ea, uam[:, m_root])
            umaie[e_root, m_root] -=       np.einsum("JFB,F->BJ", r2_ea, uam[:, m_root])

            # (4) +Σ_{F,N} sabej_S[B,F,E,N]·tmp[F,N,M,J]
            #     sabej_S[B,F,E,N] = 2·r2_ea[N,B,F] - r2_ea[N,F,B]
            umaie[e_root, m_root] += 2.0 * np.einsum("NBF,FNJ->BJ", r2_ea, tmp_2uajim_minus_uajmi[:, :, m_root, :])
            umaie[e_root, m_root] -=       np.einsum("NFB,FNJ->BJ", r2_ea, tmp_2uajim_minus_uajmi[:, :, m_root, :])

            # (5) -Σ_N smbij[N,J,M,B]·uei[E,N];  smbij[N,J,M,B] = r2_ip[J,N,B]
            umaei[e_root, m_root] -= np.einsum("JNB,N->BJ", r2_ip, uei[e_root, :])

            # (6) +Σ_{O,N} smbij[O,N,M,B]·uijke[N,O,J,E]; smbij[O,N,M,B] = r2_ip[N,O,B]
            umaei[e_root, m_root] += np.einsum("NOB,NOJ->BJ", r2_ip, uijke[:, :, :, e_root])

            # (7) -Σ_N smbij_S[J,N,M,B]·uei[E,N]
            #     smbij_S[J,N,M,B] = 2·r2_ip[N,J,B] - r2_ip[J,N,B]
            umaie[e_root, m_root] -= 2.0 * np.einsum("NJB,N->BJ", r2_ip, uei[e_root, :])
            umaie[e_root, m_root] +=       np.einsum("JNB,N->BJ", r2_ip, uei[e_root, :])

            # (8) +Σ_{O,N} smbij_S[O,N,M,B]·uijke[O,N,J,E]
            #     smbij_S[O,N,M,B] = 2·r2_ip[N,O,B] - r2_ip[O,N,B]
            umaie[e_root, m_root] += 2.0 * np.einsum("NOB,ONJ->BJ", r2_ip, uijke[:, :, :, e_root])
            umaie[e_root, m_root] -=       np.einsum("ONB,ONJ->BJ", r2_ip, uijke[:, :, :, e_root])

    # axpby chain:
    umaei_after_3 = 1.5 * umaei - 0.5 * umaie
    umaie_after_5 = (1.0 / 3.0) * umaei_after_3 + (2.0 / 3.0) * umaie
    return umaei_after_3, umaie_after_5


def build_full_w_eff_delta_g(
    ujaei, ujaie, umabi, umaib,
    umaei, umaie,
    X_IP, X_EA,
    active_occ_idx, active_vir_idx, nocc, nvir,
):
    """Sub-phase 3.6 extension of build_partial_w_eff_delta_g.

    Adds cross umaei·X_EA·X_MI dressing to Δgmaei (and umaie equivalent
    for Δgmaie):

        Δgmaei[A,I,B,J] = +Σ_E ujaei[E,I,B,J]·X_EA[E,A]            (S^EA)
                          -Σ_M umabi[A,M,B,J]·X_MI[M,I]            (S^IP)
                          -Σ_{E,M} umaei[E,M,B,J]·X_EA[E,A]·X_MI[M,I]  (cross)

    The cross term contributes only when BOTH A is active vir AND I is
    active occ (intersection of the two structured-row sets).

    Returns (delta_G_full, Dgmaei, Dgmaie) for diagnostics.
    """
    n_act_occ = len(active_occ_idx)
    n_act_vir = len(active_vir_idx)
    dim = nocc * nvir
    delta_G = np.zeros((dim, dim))

    Dgmaei = np.zeros((nvir, nocc, nvir, nocc))
    Dgmaie = np.zeros((nvir, nocc, nvir, nocc))

    # ----- S^EA route only (A active vir, full I) -----
    for k_NTO in range(n_act_vir):
        a_full = active_vir_idx[k_NTO]
        Dgmaei[a_full, :, :, :] += np.einsum("EIBJ,E->IBJ", ujaei, X_EA[:, k_NTO])
        Dgmaie[a_full, :, :, :] += np.einsum("EIBJ,E->IBJ", ujaie, X_EA[:, k_NTO])

    # ----- S^IP route only (I active occ, full A) -----
    for m_NTO in range(n_act_occ):
        i_full = active_occ_idx[m_NTO]
        Dgmaei[:, i_full, :, :] -= np.einsum("AMBJ,M->ABJ", umabi, X_IP[:, m_NTO])
        Dgmaie[:, i_full, :, :] -= np.einsum("AMBJ,M->ABJ", umaib, X_IP[:, m_NTO])

    # ----- Cross IP×EA route (both A active vir AND I active occ) -----
    # umabi_cross[A, m_NTO, B, J] = Σ_E umaei[E, m_root, B, J] · X_EA[E, A]
    # Then subtract this contracted with X_MI[m_root, i_NTO] to give:
    # ΔDgmaei[A=active, I=active, B, J] -= Σ_{E,m_root} umaei[E,m_root,B,J]·X_EA[E,a_NTO]·X_MI[m_root,i_NTO]
    for k_NTO in range(n_act_vir):
        a_full = active_vir_idx[k_NTO]
        for m_NTO in range(n_act_occ):
            i_full = active_occ_idx[m_NTO]
            # cross contribution: -Σ_{E,M_root} umaei[E,M_root,B,J]·X_EA[E,a_NTO]·X_MI[M_root,i_NTO]
            Dgmaei[a_full, i_full, :, :] -= np.einsum(
                "EMBJ,E,M->BJ", umaei, X_EA[:, k_NTO], X_IP[:, m_NTO]
            )
            Dgmaie[a_full, i_full, :, :] -= np.einsum(
                "EMBJ,E,M->BJ", umaie, X_EA[:, k_NTO], X_IP[:, m_NTO]
            )

    # Combine to singlet G^{1h1p}: ΔG[I,A,J,B] = 2·Δgmaei[A,I,B,J] - Δgmaie[A,I,B,J]
    for i_full in range(nocc):
        for a_full in range(nvir):
            row = i_full * nvir + a_full
            for j_full in range(nocc):
                for b_full in range(nvir):
                    col = j_full * nvir + b_full
                    delta_G[row, col] += (
                        2.0 * Dgmaei[a_full, i_full, b_full, j_full]
                        -     Dgmaie[a_full, i_full, b_full, j_full]
                    )

    return delta_G, Dgmaei, Dgmaie


def build_partial_w_eff_delta_g(
    ujaei, ujaie, umabi, umaib,
    X_IP, X_EA,
    active_occ_idx, active_vir_idx, nocc, nvir,
):
    """Compute the partial W^eff dressing contribution to G^{1h1p} matrix
    (sub-phase 3.5, cross umaei in 3.6 excluded).

    Returns a dense matrix ΔG[(I,A), (J,B)] which is to be ADDED to the
    skeleton G^{1h1p} from sub-phase 3.4. The CFOUR-equivalent formula:

        Δgmaei[A,I,B,J] = +Σ_E ujaei[E,I,B,J]·X_EA[E,A]   (A active vir only)
                          -Σ_M umabi[A,M,B,J]·X_MI[M,I]   (I active occ only)
        Δgmaie[A,I,B,J] = +Σ_E ujaie[E,I,B,J]·X_EA[E,A]
                          -Σ_M umaib[A,M,B,J]·X_MI[M,I]
        ΔG[I,A,J,B]     = 2·Δgmaei[A,I,B,J] - Δgmaie[A,I,B,J]

    Where X_EA[E,A_active_NTO] and X_MI[M,I_active_NTO] are the small
    (n_act_vir,n_act_vir) and (n_act_occ,n_act_occ) canonical inverses.
    """
    n_act_occ = len(active_occ_idx)
    n_act_vir = len(active_vir_idx)
    dim = nocc * nvir
    delta_G = np.zeros((dim, dim))

    # Build full-dimension Δgmaei and Δgmaie tensors [A_full, I_full, B_full, J_full]
    # but only populated for the active-A or active-I "structured" sub-blocks.
    Dgmaei = np.zeros((nvir, nocc, nvir, nocc))  # [A,I,B,J]
    Dgmaie = np.zeros((nvir, nocc, nvir, nocc))

    # S^EA route: A in active vir gets dressed (full I,B,J)
    for k_NTO in range(n_act_vir):
        a_full = active_vir_idx[k_NTO]
        # Δgmaei[a_full, :, :, :] += sum_E ujaei[E,:,:,:] · X_EA[E, k_NTO]
        Dgmaei[a_full, :, :, :] += np.einsum("EIBJ,E->IBJ", ujaei, X_EA[:, k_NTO])
        Dgmaie[a_full, :, :, :] += np.einsum("EIBJ,E->IBJ", ujaie, X_EA[:, k_NTO])

    # S^IP route: I in active occ gets dressed (full A,B,J)
    for m_NTO in range(n_act_occ):
        i_full = active_occ_idx[m_NTO]
        # Δgmaei[:, i_full, :, :] -= sum_M umabi[:,M,:,:] · X_MI[M, m_NTO]
        Dgmaei[:, i_full, :, :] -= np.einsum("AMBJ,M->ABJ", umabi, X_IP[:, m_NTO])
        Dgmaie[:, i_full, :, :] -= np.einsum("AMBJ,M->ABJ", umaib, X_IP[:, m_NTO])

    # Combine to singlet G^{1h1p} contribution: ΔG[I,A,J,B] = 2·Δgmaei - Δgmaie
    for i_full in range(nocc):
        for a_full in range(nvir):
            row = i_full * nvir + a_full
            for j_full in range(nocc):
                for b_full in range(nvir):
                    col = j_full * nvir + b_full
                    delta_G[row, col] += (
                        2.0 * Dgmaei[a_full, i_full, b_full, j_full]
                        -     Dgmaie[a_full, i_full, b_full, j_full]
                    )

    return delta_G, Dgmaei, Dgmaie


# ----------------------------------------------------------------------
# G^{1h1p} construction using CFOUR formulas (megansimons/steom_ccsd-ct
# steom_intermediates.cxx, lines 7-157 for F^eff_oo + F^eff_vv).
# ----------------------------------------------------------------------
def build_g_singlet(bar_h, r2_ip_list, r2_ea_list, X_IP, X_EA,
                    active_occ_idx, active_vir_idx, nocc, nvir):
    """Build STEOM singlet G^{1h1p} matrix using CFOUR working equations.

    F^eff_oo (G(MI) in CFOUR notation), per `gmi_steom_rhf`:
      Step 1 (closed-shell spin-adapted form from PySCF σ1[i] of IP-EOM matvec):
        U(M,I) = + 2 Σ_{l,d} Fov[l,d] · R2_IP^(M)[I,l,d]
                 -   Σ_{k,d} Fov[k,d] · R2_IP^(M)[k,I,d]
                 - 2 Σ_{k,l,d} Wooov[k,l,I,d] · R2_IP^(M)[k,l,d]
                 +   Σ_{k,l,d} Wooov[l,k,I,d] · R2_IP^(M)[k,l,d]
      Step 2:
        F^eff[M, I] = F^bar[M, I] - Σ_N U(N, I) · X(N, M)
        (only active rows are dressed; inactive rows = bare F^bar)

    F^eff_vv (G(EA)), per `gea_steom_rhf` — symmetric structure with S^EA.

    For closed-shell singlet G^{1h1p}_{ia, jb}:
      G[ia, jb] = F^bar_vv[a, b] δ_ij  -  F^eff_full[i, j] δ_ab
                + 2 Wovvo[j, b, a, i]  -  Wovov[j, a, i, b]      (CIS-like bar-W
                                                                   placeholder)
    NOTE: W^eff cross-terms (sub-phase 3.5/3.6) are not yet included; ORCA
    eigenvalues will not be matched until those land.
    """
    n_act_occ = len(active_occ_idx)
    n_act_vir = len(active_vir_idx)
    Loo   = bar_h["Loo"]
    Lvv   = bar_h["Lvv"]
    Fov   = bar_h["Fov"]      # [k, d]
    Wooov = bar_h["Wooov"]    # [k, l, i, d]
    Wvovv = bar_h["Wvovv"]    # [a, l, c, d]
    Wovov = bar_h["Wovov"]    # [k, b, i, d]
    Wovvo = bar_h["Wovvo"]    # [k, b, c, j]

    # --- F^eff_oo (sub-phase 3.4 focus, CFOUR `gmi_steom_rhf`) ----------
    # Step 1: U[M, I]
    U_MI = np.zeros((n_act_occ, nocc))
    for m_root in range(n_act_occ):
        r2 = r2_ip_list[m_root]                    # [k, l, d]
        # +2 Σ Fov[l,d] r2[I,l,d]   -   Σ Fov[k,d] r2[k,I,d]
        U_MI[m_root] += 2.0 * np.einsum("ld,Ild->I", Fov, r2)
        U_MI[m_root] -=       np.einsum("kd,kId->I", Fov, r2)
        # -2 Σ Wooov[k,l,I,d] r2[k,l,d] + Σ Wooov[l,k,I,d] r2[k,l,d]
        U_MI[m_root] -= 2.0 * np.einsum("klId,kld->I", Wooov, r2)
        U_MI[m_root] +=       np.einsum("lkId,kld->I", Wooov, r2)

    # Step 2: F^eff active rows = F^bar[M, :] - Σ_N U[N, :] · X[N, M]
    #   X_IP has shape (n_root, m_active) with X_IP[n_root, m_active]
    #   F^eff_active[m, i] = Loo[m_idx, i] - Σ_n U[n, i] · X[n, m]
    F_eff_active_rows = np.zeros((n_act_occ, nocc))
    for m_active in range(n_act_occ):
        m_idx = active_occ_idx[m_active]
        F_eff_active_rows[m_active] = Loo[m_idx, :].copy()
        for n_root in range(n_act_occ):
            F_eff_active_rows[m_active] -= U_MI[n_root] * X_IP[n_root, m_active]

    # Build full F^eff_oo (nocc × nocc): active rows dressed, others = bar F
    F_eff_oo_full = Loo.copy()
    for m_active in range(n_act_occ):
        m_idx = active_occ_idx[m_active]
        F_eff_oo_full[m_idx, :] = F_eff_active_rows[m_active]

    # --- F^eff_vv (CFOUR `gea_steom_rhf`) -- sub-phase 3.4 (extended) ---
    # PySCF EA-EOM σ1[a] from r2 contributions (src/ea_eom_ccsd_operator.cu
    # lines 124-163):
    #   σ1[a] += +2 Σ Fov[l,d] r2[l,a,d]  -  Σ Fov[l,d] r2[l,d,a]
    #           + Σ_{l,c,d} (2 Wvovv[a,l,c,d] - Wvovv[a,l,d,c]) r2[l,c,d]
    # Replacing r2 with R2_EA^{(e)}, this yields U(E, A):
    n_act_vir_e = len(active_vir_idx)
    U_EA = np.zeros((n_act_vir_e, nvir))
    for e_root in range(n_act_vir_e):
        r2 = r2_ea_list[e_root]    # [l, a, b]
        # +2 Σ Fov[l,d] r2[l,A,d]  -  Σ Fov[l,d] r2[l,d,A]
        U_EA[e_root] += 2.0 * np.einsum("ld,lAd->A", Fov, r2)
        U_EA[e_root] -=       np.einsum("ld,ldA->A", Fov, r2)
        # +2 Wvovv[A,l,c,d] r2[l,c,d]  -  Wvovv[A,l,d,c] r2[l,c,d]
        U_EA[e_root] += 2.0 * np.einsum("Alcd,lcd->A", Wvovv, r2)
        U_EA[e_root] -=       np.einsum("Aldc,lcd->A", Wvovv, r2)

    # F^eff_vv[A_active, A_inner] = Lvv[A_idx, A_inner] + Σ_E U(E, A_inner) · X(E, A_NTO)
    # Active rows get dressed; inactive rows = bar Lvv.
    F_eff_vv_full = Lvv.copy()
    for a_NTO in range(n_act_vir_e):
        a_idx = active_vir_idx[a_NTO]
        for A in range(nvir):
            s = Lvv[a_idx, A]
            for e_root in range(n_act_vir_e):
                s += U_EA[e_root, A] * X_EA[e_root, a_NTO]
            F_eff_vv_full[a_idx, A] = s

    # --- Build G^{1h1p} dense matrix in singlet basis ----------------
    dim = nocc * nvir
    G = np.zeros((dim, dim))

    # F^bar_vv[a, b] δ_ij
    for i in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                G[i * nvir + a, i * nvir + b] += F_eff_vv_full[a, b]

    # -F^eff_oo[i, j] δ_ab
    for a in range(nvir):
        for i in range(nocc):
            for j in range(nocc):
                G[i * nvir + a, j * nvir + a] -= F_eff_oo_full[i, j]

    # 2 Wovvo[j,a,b,i] - Wovov[j,a,i,b] closed-shell singlet ph-coupling.
    # NOTE (2026-05-22): the Wovvo virtual indices are [j,a,b,i] (a in 2nd slot),
    # NOT [j,b,a,i]. Verified against PySCF EE-EOM-CCSD singles-singles block
    # (‖Hss−G‖ 0.1887 → 0.0032). The swap only affects i≠j AND a≠b elements,
    # so it was masked on a=b-dominated low states but corrupted state 2+.
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    G[i * nvir + a, j * nvir + b] += 2.0 * Wovvo[j, a, b, i]
                    G[i * nvir + a, j * nvir + b] -=       Wovov[j, a, i, b]

    return G, F_eff_oo_full, F_eff_vv_full, U_MI, U_EA


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------
def main(xyz_path, basis, n_act_occ, n_act_vir, n_steom_roots):
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd

    print("=" * 78)
    print("STEOM-CCSD G^{1h1p} reference (PySCF + numpy)")
    print(f"  xyz={xyz_path}  basis={basis}")
    print(f"  n_act_occ={n_act_occ}  n_act_vir={n_act_vir}  n_steom={n_steom_roots}")
    print("=" * 78)

    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf  = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2

    nocc = mycc.nocc
    nmo  = mycc.nmo
    nvir = nmo - nocc

    # MO ERI (chemist notation, full 4D) — for small system convenience
    mo = mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mo, compact=False).reshape(nmo, nmo, nmo, nmo)

    # Diagonal Fock matrices in MO basis (canonical RHF → already diagonal)
    f_oo = np.diag(mf.mo_energy[:nocc])
    f_vv = np.diag(mf.mo_energy[nocc:])

    print(f"\n--- System ---\n  nocc={nocc}  nvir={nvir}  nmo={nmo}")
    print(f"  HF E   = {mf.e_tot:.10f}  Ha")
    print(f"  CCSD E = {mf.e_tot + mycc.e_corr:.10f}  Ha (corr={mycc.e_corr:.10f})")

    # EOM-EE-CCSD reference (the exact target STEOM should approximate; with a
    # FULL active space STEOM must reduce to these eigenvalues bit-for-bit).
    eom_ee = eom_rccsd.EOMEESinglet(mycc)
    e_ee, _ = eom_ee.kernel(nroots=n_steom_roots)
    e_ee = np.atleast_1d(np.asarray(e_ee))
    print(f"\n--- PySCF EOM-EE-CCSD (singlet) reference ---")
    for k in range(len(e_ee)):
        print(f"  EE state {k}:  ω = {e_ee[k]:.10f} Ha  ({e_ee[k]*27.2114:.4f} eV)")

    # Step 1: build 11 bar-H intermediates
    bar_h = build_bar_h(eri_mo, t1, t2, f_oo, f_vv, nocc, nvir)
    print("\n--- bar-H intermediate Frobenius (PySCF reference) ---")
    for k in ("Loo", "Lvv", "Fov",
              "Woooo", "Wooov", "Wovov", "Wovvo", "Wovoo",
              "Wvovv", "Wvvvv", "Wvvvo"):
        print(f"  ‖{k:7s}‖ = {np.linalg.norm(bar_h[k]):.8f}")

    # Step 2: PySCF IP-EOM / EA-EOM (canonical, n_act_* lowest roots)
    eom_ip = eom_rccsd.EOMIP(mycc)
    e_ip, r_ip = eom_ip.kernel(nroots=n_act_occ)
    if np.isscalar(e_ip):
        e_ip = np.asarray([e_ip]); r_ip = [r_ip]
    print("\n--- PySCF IP-EOM-CCSD ---")
    r1_ip_list, r2_ip_list = [], []
    for k in range(n_act_occ):
        rv = np.asarray(r_ip[k]).ravel()
        r1 = rv[:nocc]
        r2 = rv[nocc:].reshape(nocc, nocc, nvir)
        r1_ip_list.append(r1); r2_ip_list.append(r2)
        n1 = np.einsum("i,i->", r1, r1)
        n2 = np.einsum("ija,ija->", r2, r2)
        print(f"  IP root {k}:  ω={e_ip[k]:.10f} Ha  %singles={n1/(n1+n2):.4f}  ‖R1‖={np.sqrt(n1):.4f}")

    eom_ea = eom_rccsd.EOMEA(mycc)
    e_ea, r_ea = eom_ea.kernel(nroots=n_act_vir)
    if np.isscalar(e_ea):
        e_ea = np.asarray([e_ea]); r_ea = [r_ea]
    print("\n--- PySCF EA-EOM-CCSD ---")
    r1_ea_list, r2_ea_list = [], []
    for k in range(n_act_vir):
        rv = np.asarray(r_ea[k]).ravel()
        r1 = rv[:nvir]
        r2 = rv[nvir:].reshape(nocc, nvir, nvir)
        r1_ea_list.append(r1); r2_ea_list.append(r2)
        n1 = np.einsum("a,a->", r1, r1)
        n2 = np.einsum("iab,iab->", r2, r2)
        print(f"  EA root {k}:  ω={e_ea[k]:.10f} Ha  %singles={n1/(n1+n2):.4f}  ‖R1‖={np.sqrt(n1):.4f}")

    # Step 3: active NTO indices via Hungarian 1:1 assignment (collision-free).
    # Each IP/EA root is mapped to a DISTINCT dominant orbital index, so the
    # active R1 matrix is non-singular even for (near-)degenerate roots.
    active_occ_idx = assign_active_1to1(r1_ip_list, nocc)
    active_vir_idx = assign_active_1to1(r1_ea_list, nvir)
    # Compare against the legacy per-root argmax to flag collisions.
    argmax_occ = [int(np.argmax(np.abs(r1))) for r1 in r1_ip_list]
    argmax_vir = [int(np.argmax(np.abs(r1))) for r1 in r1_ea_list]
    print(f"\n--- Active NTO selection (Hungarian 1:1 assignment) ---")
    print(f"  active occ indices (IP, 1:1): {active_occ_idx}   (argmax: {argmax_occ})")
    print(f"  active vir indices (EA, 1:1): {active_vir_idx}   (argmax: {argmax_vir})")
    if len(set(argmax_occ)) != len(argmax_occ):
        print(f"  ⚠ argmax IP had COLLISIONS (degenerate roots) — 1:1 fixes this")
    if len(set(argmax_vir)) != len(argmax_vir):
        print(f"  ⚠ argmax EA had COLLISIONS (degenerate roots) — 1:1 fixes this")
    full_active = (n_act_occ == nocc and n_act_vir == nvir)
    if full_active:
        print(f"  ★ FULL-ACTIVE MODE (n_act_occ=nocc={nocc}, n_act_vir=nvir={nvir})")
        print(f"    → definitive correctness test: STEOM must reduce to EOM-CCSD")

    # Step 4: X(MI), X(EA) matrices — active R1 inverse (CFOUR `renormalize`)
    X_IP, X_EA = build_x_matrices(r1_ip_list, r1_ea_list,
                                  active_occ_idx, active_vir_idx)
    print(f"\n--- X matrices (active R1 inverse, CFOUR canonical) ---")
    print(f"  X_IP shape={X_IP.shape}  ‖X_IP‖={np.linalg.norm(X_IP):.8f}")
    print(f"  X_EA shape={X_EA.shape}  ‖X_EA‖={np.linalg.norm(X_EA):.8f}")
    print(f"  X_IP diagonal = {np.diag(X_IP)}")
    print(f"  X_EA diagonal = {np.diag(X_EA)}")

    # Step 5: build G^{1h1p} singlet matrix using CFOUR formulas
    G, F_eff_oo, F_eff_vv, U_MI, U_EA = build_g_singlet(
        bar_h, r2_ip_list, r2_ea_list, X_IP, X_EA,
        active_occ_idx, active_vir_idx, nocc, nvir)
    print(f"\n--- F^eff Frobenius (★ sub-phase 3.4 deliverable, CFOUR formula) ---")
    print(f"  ‖bar F_oo (= Loo)‖ = {np.linalg.norm(bar_h['Loo']):.8f}")
    print(f"  ‖U(M,I)‖           = {np.linalg.norm(U_MI):.8f}  (intermediate, shape={U_MI.shape})")
    print(f"  ‖F^eff_oo (full)‖  = {np.linalg.norm(F_eff_oo):.8f}  (active rows dressed)")
    print(f"  ‖F^eff_oo − Loo‖   = {np.linalg.norm(F_eff_oo - bar_h['Loo']):.8f}  (Ŝ^IP contribution)")
    print(f"  ‖bar F_vv (= Lvv)‖ = {np.linalg.norm(bar_h['Lvv']):.8f}")
    print(f"  ‖U(E,A)‖           = {np.linalg.norm(U_EA):.8f}  (intermediate, shape={U_EA.shape})")
    print(f"  ‖F^eff_vv (full)‖  = {np.linalg.norm(F_eff_vv):.8f}  (active rows dressed)")
    print(f"  ‖F^eff_vv − Lvv‖   = {np.linalg.norm(F_eff_vv - bar_h['Lvv']):.8f}  (Ŝ^EA contribution)")

    # Step 5b: build W^eff partial dressing (sub-phase 3.5)
    ujaei, ujaie = build_ujaei(bar_h, r2_ea_list, nocc, nvir)
    umabi, umaib = build_umabi(bar_h, r2_ip_list, nocc, nvir)

    # DIAGNOSTIC: compare CFOUR-derived umabi vs Nooijen Eq.(56) direct port
    u_amci_NB = build_u_amci_nooijen(bar_h, r2_ip_list, nocc, nvir)
    compare_umabi_vs_nooijen(umabi, u_amci_NB, label="phph S^IP route (Eq.56)")
    delta_G_35, Dgmaei_35, Dgmaie_35 = build_partial_w_eff_delta_g(
        ujaei, ujaie, umabi, umaib, X_IP, X_EA,
        active_occ_idx, active_vir_idx, nocc, nvir,
    )

    print(f"\n--- W^eff partial Frobenius (sub-phase 3.5 deliverable, CFOUR formula) ---")
    print(f"  ‖ujaei (final, 2·dir-swap)‖ = {np.linalg.norm(ujaei):.8f}  shape={ujaei.shape}")
    print(f"  ‖ujaie (final, swap-only) ‖ = {np.linalg.norm(ujaie):.8f}  shape={ujaie.shape}")
    print(f"  ‖umabi (final, 2·dir-swap)‖ = {np.linalg.norm(umabi):.8f}  shape={umabi.shape}")
    print(f"  ‖umaib (final, swap-only) ‖ = {np.linalg.norm(umaib):.8f}  shape={umaib.shape}")
    print(f"  ‖Δgmaei (ovvo dressing)   ‖ = {np.linalg.norm(Dgmaei_35):.8f}  (3.5 partial, active-A or active-I rows)")
    print(f"  ‖Δgmaie (ovov dressing)   ‖ = {np.linalg.norm(Dgmaie_35):.8f}")
    print(f"  ‖ΔG^{{1h1p}} (3.5 partial) ‖ = {np.linalg.norm(delta_G_35):.8f}")

    # Step 5c: build cross umaei + full W^eff dressing (sub-phase 3.6)
    uei, uam = build_uei_uam(bar_h, r2_ea_list, r2_ip_list, nocc, nvir)
    uijke, uajim, uajmi = build_uijke_uajim(bar_h, r2_ea_list, r2_ip_list, nocc, nvir)
    umaei, umaie = build_umaei(bar_h, uei, uam, uijke, uajim, uajmi,
                               r2_ea_list, r2_ip_list, nocc, nvir)
    delta_G_36, Dgmaei_36, Dgmaie_36 = build_full_w_eff_delta_g(
        ujaei, ujaie, umabi, umaib, umaei, umaie, X_IP, X_EA,
        active_occ_idx, active_vir_idx, nocc, nvir,
    )

    print(f"\n--- Cross IP×EA helpers Frobenius (★ sub-phase 3.6 deliverable) ---")
    print(f"  ‖uei (E,I)‖    = {np.linalg.norm(uei):.8f}  shape={uei.shape}")
    print(f"  ‖uam (A,M)‖    = {np.linalg.norm(uam):.8f}  shape={uam.shape}")
    print(f"  ‖uijke‖        = {np.linalg.norm(uijke):.8f}  shape={uijke.shape}")
    print(f"  ‖uajim (avg)‖  = {np.linalg.norm(uajim):.8f}  shape={uajim.shape}")
    print(f"  ‖uajmi‖        = {np.linalg.norm(uajmi):.8f}  shape={uajmi.shape}")
    print(f"  ‖umaei‖        = {np.linalg.norm(umaei):.8f}  shape={umaei.shape}")
    print(f"  ‖umaie‖        = {np.linalg.norm(umaie):.8f}  shape={umaie.shape}")
    print(f"  ‖Δgmaei (3.6 full) ‖ = {np.linalg.norm(Dgmaei_36):.8f}")
    print(f"  ‖Δgmaie (3.6 full) ‖ = {np.linalg.norm(Dgmaie_36):.8f}")
    print(f"  ‖ΔG^{{1h1p}} (3.6 full)‖ = {np.linalg.norm(delta_G_36):.8f}")
    print(f"  Δ(cross only)  = ‖ΔG_36 - ΔG_35‖ = {np.linalg.norm(delta_G_36 - delta_G_35):.8f}")

    # Combine into final G matrices
    G_dressed_35 = G + delta_G_35
    G_dressed_36 = G + delta_G_36

    # Step 6: diagonalize all three (3.4 only, 3.5 partial, 3.6 full)
    print(f"\n--- STEOM G^{{1h1p}} eigenvalues — sub-phase 3.4 (F^eff only) ---")
    e34 = np.sort(np.real(np.linalg.eigvals(G)))
    for k in range(min(n_steom_roots, len(e34))):
        print(f"  state {k}:  ω = {e34[k]:.10f} Ha  ({e34[k]*27.2114:.4f} eV)")

    print(f"\n--- STEOM G^{{1h1p}} eigenvalues — sub-phase 3.5 (+W^eff partial) ---")
    e35 = np.sort(np.real(np.linalg.eigvals(G_dressed_35)))
    for k in range(min(n_steom_roots, len(e35))):
        print(f"  state {k}:  ω = {e35[k]:.10f} Ha  ({e35[k]*27.2114:.4f} eV)")

    print(f"\n--- STEOM G^{{1h1p}} eigenvalues — sub-phase 3.6 (+W^eff full, cross umaei) ---")
    e36 = np.sort(np.real(np.linalg.eigvals(G_dressed_36)))
    for k in range(min(n_steom_roots, len(e36))):
        print(f"  state {k}:  ω = {e36[k]:.10f} Ha  ({e36[k]*27.2114:.4f} eV)")

    # Step 5d: CANONICAL Nooijen Eq.(56)-(63) no-cross baseline
    Gc, g_phph, g_phhp, u_amci_c, u_akei_c, u_bkje_c = build_g_phph_phhp_canonical(
        bar_h, r2_ip_list, r2_ea_list, F_eff_oo, F_eff_vv, X_IP, X_EA,
        active_occ_idx, active_vir_idx, nocc, nvir,
    )
    print(f"\n--- CANONICAL Nooijen Eq.(56)-(63) Frobenius (no-cross baseline) ---")
    print(f"  ‖u_amci (Eq.56, S^IP)‖ = {np.linalg.norm(u_amci_c):.8f}")
    print(f"  ‖u_akei (Eq.57, S^EA)‖ = {np.linalg.norm(u_akei_c):.8f}")
    print(f"  ‖u_bkje (Eq.61, S^EA)‖ = {np.linalg.norm(u_bkje_c):.8f}")
    print(f"  ‖g_phph (Eq.59)‖       = {np.linalg.norm(g_phph):.8f}")
    print(f"  ‖g_phhp (Eq.63)‖       = {np.linalg.norm(g_phhp):.8f}")
    print(f"\n--- STEOM G^{{1h1p}} eigenvalues — CANONICAL Nooijen no-cross ---")
    ec = np.sort(np.real(np.linalg.eigvals(Gc)))
    for k in range(min(n_steom_roots, len(ec))):
        print(f"  state {k}:  ω = {ec[k]:.10f} Ha  ({ec[k]*27.2114:.4f} eV)")

    # Step 5e: CANONICAL FULL with cross terms (Eq.56-63, pre-normalized s)
    Gf, gf_phph, gf_phhp, u_amei_f, u_bmje_f = build_g_canonical_full(
        bar_h, r2_ip_list, r2_ea_list, r1_ip_list, r1_ea_list,
        active_occ_idx, active_vir_idx, nocc, nvir,
    )
    print(f"\n--- CANONICAL FULL (with cross Eq.58/62) Frobenius ---")
    print(f"  ‖u_amei (cross phph)‖ = {np.linalg.norm(u_amei_f):.8f}")
    print(f"  ‖u_bmje (cross phhp)‖ = {np.linalg.norm(u_bmje_f):.8f}")
    print(f"\n--- STEOM G^{{1h1p}} eigenvalues — CANONICAL Nooijen FULL (cross) ---")
    ef = np.sort(np.real(np.linalg.eigvals(Gf)))
    for k in range(min(n_steom_roots, len(ef))):
        print(f"  state {k}:  ω = {ef[k]:.10f} Ha  ({ef[k]*27.2114:.4f} eV)")

    # Step 7: definitive comparison table — STEOM (canonical full) vs EOM-EE-CCSD
    print("\n" + "=" * 78)
    print("DEFINITIVE COMPARISON: STEOM canonical-full vs PySCF EOM-EE-CCSD")
    if full_active:
        print("  (FULL-ACTIVE MODE — STEOM should EQUAL EOM-EE-CCSD if formulas correct)")
    else:
        print(f"  (truncated active: n_act_occ={n_act_occ}, n_act_vir={n_act_vir})")
    print("=" * 78)
    n_cmp = min(n_steom_roots, len(ef), len(e_ee))
    print(f"  {'state':>5} {'EOM-EE':>14} {'STEOM-full':>14} {'Δ(mHa)':>10}")
    max_abs_dev = 0.0
    for k in range(n_cmp):
        dev_mHa = (ef[k] - e_ee[k]) * 1000.0
        max_abs_dev = max(max_abs_dev, abs(dev_mHa))
        print(f"  {k:>5} {e_ee[k]:>14.8f} {ef[k]:>14.8f} {dev_mHa:>+10.3f}")
    print(f"  max|Δ| = {max_abs_dev:.3f} mHa")
    if full_active:
        if max_abs_dev < 1.0:
            print("  ✅ PASS: STEOM == EOM-CCSD in full active → formulas CORRECT;")
            print("     over-correction at truncation is an active-space artifact.")
        else:
            print("  ❌ FAIL: STEOM ≠ EOM-CCSD in full active → FORMULA BUG remains.")
            print("     Next: term-by-term check of F^eff/W^eff vs eom_rccsd matvec.")

    print("\n--- ORCA reference targets (H2O sto-3g) ---")
    print("  state 0:  ω = 0.4354200000  Ha")
    print("  state 1:  ω = 0.4998300000  Ha")
    print("  state 2:  ω = 0.5916380000  Ha")
    print("\n  *** sub-phase 3.4 status (F^eff_oo only, CFOUR formula): ***")
    print("    11/11 bar-H bit-exact vs GANSU (view-bug fix landed)")
    print("    X(MI) matrix built per CFOUR `renormalize` (active R1 inverse)")
    print("    U(M,I) per CFOUR `gmi_steom_rhf` (PySCF σ1-style spin-adapted)")
    print("    F^eff_oo dressing active-rows-only, full = bar Loo otherwise")
    print("")
    print("    Eigenvalue gap vs ORCA depends on W^eff (3.5) + cross (3.6)")
    print("    + G(EM)/G(Mn,Ie) (3.5/3.6) which are still missing. State 2")
    print("    typically at +5 mHa, state 0/1 at ~-45 mHa undershoot when")
    print("    only F^eff_oo is dressed — closes after subsequent sub-phases.")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__); sys.exit(1)
    xyz   = sys.argv[1]
    basis = sys.argv[2]
    n_ao  = int(sys.argv[3])
    n_av  = int(sys.argv[4])
    n_st  = int(sys.argv[5])
    main(xyz, basis, n_ao, n_av, n_st)
