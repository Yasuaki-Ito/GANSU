#!/usr/bin/env python3
"""
Faithful CFOUR W^eff (gmaei) port — spinad storage-toggle emulation
===================================================================

Purpose (2026-05-21): the over-correction in STEOM was DECISIVELY localized to
the W^eff dressing (F^eff verified correct: dressed diag = IP/EA eigenvalues).
The two existing W^eff implementations DISAGREE (Nooijen build_g_canonical_full
+24.2 mHa vs CFOUR build_full_w_eff_delta_g +33.5 mHa on H2O sto-3g state 0),
and BOTH under-correct (EOM needs +61.8 mHa). The prime suspect is the
`.spinad()` storage-toggle semantics in CFOUR steom_intermediates.cxx, which the
existing Python ports approximate with an axpby algebra that may be wrong for
the DOUBLE-spinad terms (both operands toggled).

This script ports `ujaei_steom_rhf` + `umabi_steom_rhf` + `umaei_steom_rhf` +
`gmaei_steom_rhf` LITERALLY: every `.spinad()` is emulated as an explicit
in-place tensor toggle T → 2·T − T_swap on the documented index pair, and each
`contract(...)` is reproduced in source order. This removes the axpby ambiguity.

CFOUR tensor layouts (verified against steom.cxx assembly + memory mappings):
  fem[E,M]        = F_vo[E,M]      = Fov[M,E]                       (v,o)
  wmnie[M,N,I,E]  = W(Mn,Ie)       = Wooov[M,N,I,E]                 (o,o,o,v)
  wamef[E,f,A,m]  = W(Ef,Am)       = Wvovv[A,m,E,f]                 (v,v,v,o)
  wmnef[A,B,I,J]  = <Ab|Ij>        = eri_ovov[I,A,J,B]              (v,v,o,o)
  smbij[I,J,M,b]  = S(Ij,Mb)       = r2_ip[M][J,I,b]               (o,o,act_o,v)
  sabej[A,B,E,j]  = S(Ab,Ej)       = r2_ea[E][j,A,B]               (v,v,act_v,o)
  wmaei[A,I,B,J]  = W(EM,bj)       = Wovvo[I,A,J,B]  (bare ph-hp Coulomb)
  wmaie[A,I,B,J]  = W(Em,Bj)       = Wovov[I,A,J,B]  (bare ph-ph exchange)

spinad swap axes (closed-shell singlet, 2·T − T_swap):
  smbij : swap I↔J  (two holes)         smbij_S[I,J,M,b]=2 s[I,J,M,b]-s[J,I,M,b]
  sabej : swap A↔B  (two particles)     sabej_S[A,B,E,j]=2 s[A,B,E,j]-s[B,A,E,j]
  wmnie : swap M↔N  (two holes)         wmnie_S[M,N,I,E]=2 w[M,N,I,E]-w[N,M,I,E]
  wamef : swap E↔f  (two particles)     wamef_S[E,f,A,m]=2 w[E,f,A,m]-w[f,E,A,m]
  wmnef : swap A↔B  (two particles)     wmnef_S[A,B,I,J]=2 w[A,B,I,J]-w[B,A,I,J]

X(MI)[M_act, I_full] and X(EA)[E_act, A_full]: nonzero only on active columns
(= active R1 inverse mapped through the selection matrix). Built here as full
(act × full) matrices with the active-orbital columns filled.

RESOLVED (2026-05-22): smbij uses the TRANSPOSE layout r2_ip[M][J,I,b], which is
the verbatim CFOUR convention (`trans "JIB"→"IJB"` ⇒ sijb[I,J,b]=r2[J,I,b]). With
it the faithful W^eff AGREES with the independent Nooijen build_g_canonical_full
(H2O sto-3g: 0.38864/0.45311/0.58108 vs 0.39289/0.44906/0.57869, ‖umaei‖=0.0035).
The earlier "umi(slot0) vs umabi(slot1) inconsistency" was a FALSE coupling: the
umi diagnostic reproduces -IP only with DIRECT smbij, but umi is NOT the production
F^eff route (F^eff comes from build_g_singlet, PySCF σ1-style, independent of this
smbij). The faithful smbij is consumed ONLY by umabi/umaei → transpose is correct.
DIRECT was the regression (over-shoots W^eff by ~+150 mHa: 0.586/0.679/0.701).

★ The slot question is NOT the physics gap. Both correct, independent W^eff impls
(this CFOUR port + Nooijen Eq.56-63) land at ~0.39, ~38 mHa below EOM-EE-CCSD and
~43 below ORCA STEOM (0.4354). The residual gap is COMMON to both → the next
suspect is active-space mismatch with ORCA (CIS-NTO vs argmax/Hungarian), not the
S^IP route. EA-route (ujaei, F^eff_vv) and bare-W are verified correct.

Usage:
    wsl python3 script/steom_weff_cfour_faithful.py xyz/H2O.xyz sto-3g 3 2 3

[[pyscf-run-locally]] [[pyscf-cartesian]] [[no-adhoc-fixes]] [[careful-verification]]
"""

import sys
import numpy as np

sys.path.insert(0, "script")
from pyscf_steom_feff_reference import (
    read_xyz, build_bar_h, build_g_singlet, assign_active_1to1,
)


def build_cfour_tensors(bar_h, r2_ip_list, r2_ea_list, nocc, nvir,
                        n_act_occ, n_act_vir):
    """Assemble the CFOUR-layout dpd tensors from PySCF bar-H + R2 amplitudes."""
    Fov = bar_h["Fov"]            # [m,e]
    Wooov = bar_h["Wooov"]        # [m,n,i,e]
    Wvovv = bar_h["Wvovv"]        # [a,m,e,f]
    Wovvo = bar_h["Wovvo"]        # [k,b,c,j]
    Wovov = bar_h["Wovov"]        # [k,b,i,d]
    eri_ovov = bar_h["eri_ovov"]  # [i,a,j,b]

    fem = Fov.T.copy()                                  # [E,M] = Fov[M,E]
    wmnie = Wooov.copy()                                # [M,N,I,E]
    wamef = np.einsum("amef->efam", Wvovv).copy()       # [E,f,A,m]=Wvovv[A,m,E,f]
    wmnef = np.einsum("iajb->abij", eri_ovov).copy()    # [A,B,I,J]=eri_ovov[I,A,J,B]

    # smbij[I,J,M,b] = r2_ip[M][J,I,b] (TRANSPOSE, verbatim CFOUR). The CFOUR
    # driver builds S(Ij,Mb) via `trans(1.0, r2, "JIB", 0.0, sijb, "IJB")`, i.e.
    # sijb[I,J,b] = r2_cfour[J,I,b]. This (transpose) orientation is the one the
    # umabi/umaei contractions are written against; it makes the faithful W^eff
    # AGREE with the independent Nooijen build_g_canonical_full (~0.389 vs 0.393
    # on H2O sto-3g, ‖umaei‖=0.0035). The DIRECT orientation r2[I,J,b] was tried
    # because the faithful-umi diagnostic reproduces -IP eigenvalues with it, but
    # that path is NOT the production F^eff route (F^eff comes from build_g_singlet,
    # PySCF σ1-style, independent of this smbij) — direct over-shoots W^eff by
    # ~+150 mHa. Override with SMBIJ_TRANSPOSE=0 only for the umi diagnostic.
    import os as _os
    smbij = np.zeros((nocc, nocc, n_act_occ, nvir))     # [I,J,M,b]
    _SMBIJ_TRANSPOSE = _os.environ.get("SMBIJ_TRANSPOSE", "1") == "1"
    for m in range(n_act_occ):
        if _SMBIJ_TRANSPOSE:
            smbij[:, :, m, :] = np.swapaxes(r2_ip_list[m], 0, 1)   # r2[J,I,b] (CFOUR)
        else:
            smbij[:, :, m, :] = r2_ip_list[m]                      # r2[I,J,b] direct (umi diag only)
    sabej = np.zeros((nvir, nvir, n_act_vir, nocc))     # [A,B,E,j]=r2_ea[E][j,A,B]
    for e in range(n_act_vir):
        sabej[:, :, e, :] = np.einsum("jab->abj", r2_ea_list[e])

    # bare ph-hp / ph-ph (W(EM,bj) Coulomb / W(Em,Bj) exchange), CFOUR [A,I,B,J]
    # (vir,occ,vir,occ). bare-match (verified vs PySCF EE singles block):
    #   gmaei[a,i,b,j]=Wovvo[j,a,b,i] (NOT [j,b,a,i] — a,b swap bug fixed),
    #   gmaie[a,i,b,j]=Wovov[j,a,i,b].  Singlet coupling 2·gmaei - gmaie.
    wmaei = np.einsum("kbcj->bjck", Wovvo).copy()       # [A,I,B,J]=Wovvo[J,A,B,I]
    wmaie = np.einsum("kbid->bidk", Wovov).copy()       # [A,I,B,J]=Wovov[J,A,I,B]
    return dict(fem=fem, wmnie=wmnie, wamef=wamef, wmnef=wmnef,
                smbij=smbij, sabej=sabej, wmaei=wmaei, wmaie=wmaie)


def spinad(T, ax0, ax1):
    """CFOUR .spinad() storage toggle: T -> 2·T - swapaxes(T, ax0, ax1)."""
    return 2.0 * T - np.swapaxes(T, ax0, ax1)


def build_X(r1_list, active_idx, n_full):
    """X[act_root, full] = active R1 inverse, nonzero only on active columns."""
    n_act = len(r1_list)
    R1act = np.array([[r1_list[lam][active_idx[r]] for r in range(n_act)]
                      for lam in range(n_act)])   # [lam, r]
    Xinv = np.linalg.inv(R1act)                   # [r, lam]?  inv: Xinv@R1act=I
    # CFOUR: U=trans^T·r1 (act×act); X = inv(U)·trans^T (act×full).
    # trans[full, r]=δ(full, active_idx[r]); so X[r, full] nonzero only at
    # full∈active_idx, with X[r, active_idx[s]] = inv(U)[r, s].
    # U[r, s] = Σ_full trans[full,r]·r1[s? ] ... use R1act convention:
    # U[r, s] = r1_lam? Build U[r,s] = Σ_lam trans[lam_orb? ]. Simplest: the
    # active block of X equals inv(R1_active^T) consistent with build_x_matrices.
    X = np.zeros((n_act, n_full))
    Xact = np.linalg.inv(R1act.T)                 # matches build_x_matrices X_*
    for r in range(n_act):
        for s in range(n_act):
            X[r, active_idx[s]] = Xact[r, s]
    return X


def build_ujaei_faithful(T, n_act_vir, nocc, nvir):
    """Literal port of ujaei_steom_rhf (CCSD, r3 skipped)."""
    fem, wamef, wmnie, wmnef = T["fem"], T["wamef"], T["wmnie"], T["wmnef"]
    sabej = T["sabej"]
    ujaei = np.zeros((n_act_vir, nocc, nvir, nocc))   # [E,I,B,J]
    ujaie = np.zeros((n_act_vir, nocc, nvir, nocc))

    # --- non-spinad block (ujaei) ---
    # F(fi) R(fb,ej) -> U(ib,ej):  Σ_F sabej[F,B,E,J]·fem[F,I]
    ujaei += np.einsum("FBEJ,FI->EIBJ", sabej, fem)
    # W(bi,fg) R(gf,ej):  Σ_{G,F} sabej[G,F,E,J]·wamef[F,G,B,I]
    ujaei += np.einsum("GFEJ,FGBI->EIBJ", sabej, wamef)
    # - W(in,fj) R(fb,en): -Σ_{F,N} sabej[F,B,E,N]·wmnie[N,I,J,F]
    ujaei -= np.einsum("FBEN,NIJF->EIBJ", sabej, wmnie)

    # --- spinad block (ujaie): sabej spinad (A↔B), then wmnie spinad on term3 ---
    sabej_S = spinad(sabej, 0, 1)
    # F(fi) R(bf,ej): Σ_F sabej_S[B,F,E,J]·fem[F,I]
    ujaie += np.einsum("BFEJ,FI->EIBJ", sabej_S, fem)
    # W(ib,fg) R(gf,ej): Σ sabej_S[G,F,E,J]·wamef[G,F,B,I]
    ujaie += np.einsum("GFEJ,GFBI->EIBJ", sabej_S, wamef)
    # W(in,jf) R(bf,en): Σ sabej_S[B,F,E,N]·wmnie_S[I,N,J,F]
    wmnie_S = spinad(wmnie, 0, 1)
    ujaie += np.einsum("BFEN,INJF->EIBJ", sabej_S, wmnie_S)

    # axpby(-0.5, ujaie, 1.5, ujaei): ujaei := 1.5·ujaei - 0.5·ujaie
    ujaei = 1.5 * ujaei - 0.5 * ujaie
    # axpby(1/3, ujaei, 2/3, ujaie): ujaie := 1/3·ujaei + 2/3·ujaie
    ujaie = (1.0 / 3.0) * ujaei + (2.0 / 3.0) * ujaie
    return ujaei, ujaie


def build_umabi_faithful(T, n_act_occ, nocc, nvir):
    """Literal port of umabi_steom_rhf (CCSD, r3 skipped)."""
    fem, wamef, wmnie = T["fem"], T["wamef"], T["wmnie"]
    smbij = T["smbij"]
    umabi = np.zeros((nvir, n_act_occ, nvir, nocc))   # [A,M,B,J]
    umaib = np.zeros((nvir, n_act_occ, nvir, nocc))

    # --- non-spinad (umabi) ---
    # - F(an) R(mb,nj): -Σ_N smbij[N,J,M,B]·fem[A,N]
    umabi -= np.einsum("NJMB,AN->AMBJ", smbij, fem)
    # W(no,ja) R(mb,on): Σ_{O,N} smbij[O,N,M,B]·wmnie[N,O,J,A]
    umabi += np.einsum("ONMB,NOJA->AMBJ", smbij, wmnie)
    # - W(nb,af) R(mf,nj): -Σ_{N,F} smbij[N,J,M,F]·wamef[F,A,B,N]
    umabi -= np.einsum("NJMF,FABN->AMBJ", smbij, wamef)

    # --- spinad (umaib): smbij spinad (I↔J), then wamef spinad on term3 ---
    smbij_S = spinad(smbij, 0, 1)
    # - F(an) R(mb,jn): -Σ_N smbij_S[J,N,M,B]·fem[A,N]
    umaib -= np.einsum("JNMB,AN->AMBJ", smbij_S, fem)
    # W(no,aj) R(mb,on): Σ smbij_S[O,N,M,B]·wmnie[O,N,J,A]
    umaib += np.einsum("ONMB,ONJA->AMBJ", smbij_S, wmnie)
    # W(nb,fa) R(mf,jn): Σ smbij_S[J,N,M,F]·wamef_S[A,F,B,N]
    wamef_S = spinad(wamef, 0, 1)
    umaib += np.einsum("JNMF,AFBN->AMBJ", smbij_S, wamef_S)

    umabi = 1.5 * umabi - 0.5 * umaib
    umaib = (1.0 / 3.0) * umabi + (2.0 / 3.0) * umaib
    return umabi, umaib


def build_uei_uam_uijke_uajim_faithful(T, n_act_occ, n_act_vir, nocc, nvir):
    fem, wamef, wmnie, wmnef = T["fem"], T["wamef"], T["wmnie"], T["wmnef"]
    sabej, smbij = T["sabej"], T["smbij"]

    # uei[E,I] = Σ wmnef_S[G,F,I,M]·sabej[G,F,E,M]
    wmnef_S = spinad(wmnef, 0, 1)
    uei = np.einsum("GFIM,GFEM->EI", wmnef_S, sabej)
    # uam[A,M] = -Σ wmnef_S[A,E,O,N]·smbij[O,N,M,E]
    uam = -np.einsum("AEON,ONME->AM", wmnef_S, smbij)

    # uijke[I,J,K,E] = Σ wmnef[G,F,J,I]·sabej[G,F,E,K]
    uijke = np.einsum("GFJI,GFEK->IJKE", wmnef, sabej)

    # uajim/uajmi:
    uajim = np.zeros((nvir, nocc, n_act_occ, nocc))   # [A,I,M,J]
    uajmi = np.zeros((nvir, nocc, n_act_occ, nocc))
    # W(in,ae) R(me,jn) -> uajim: wmnef_S & smbij_S, Σ wmnef_S[A,E,I,N]·smbij_S[J,N,M,E]
    smbij_S = spinad(smbij, 0, 1)
    uajim += np.einsum("AEIN,JNME->AIMJ", wmnef_S, smbij_S)
    # - W(in,ea) R(e,jn) -> uajmi: -Σ wmnef[A,E,N,I]·smbij[N,J,M,E]
    uajmi -= np.einsum("AENI,NJME->AIMJ", wmnef, smbij)
    # axpby(0.5, uajmi, 0.5, uajim): uajim := 0.5·uajmi + 0.5·uajim
    uajim = 0.5 * uajmi + 0.5 * uajim
    return uei, uam, uijke, uajim, uajmi


def build_umaei_faithful(T, uei, uam, uijke, uajim, uajmi,
                         n_act_occ, n_act_vir, nocc, nvir):
    """Literal port of umaei_steom_rhf."""
    sabej, smbij = T["sabej"], T["smbij"]
    umaei = np.zeros((n_act_vir, n_act_occ, nvir, nocc))   # [E,M,B,J]
    umaie = np.zeros((n_act_vir, n_act_occ, nvir, nocc))

    # U(fm) R(fb,ej) -> umaei: Σ_F sabej[F,B,E,J]·uam[F,M]
    umaei += np.einsum("FBEJ,FM->EMBJ", sabej, uam)
    # - U(mn,fj) R(fb,en) -> umaei: -Σ sabej[F,B,E,N]·uajmi[F,N,M,J]
    umaei -= np.einsum("FBEN,FNMJ->EMBJ", sabej, uajmi)

    # sabej spinad
    sabej_S = spinad(sabej, 0, 1)
    # U(fm) R(bf,ej) -> umaie: Σ sabej_S[B,F,E,J]·uam[F,M]
    umaie += np.einsum("BFEJ,FM->EMBJ", sabej_S, uam)
    # U(mn,jf) R(bf,en) -> umaie: with tmp = 2·uajim - uajmi
    tmp = 2.0 * uajim - uajmi
    umaie += np.einsum("BFEN,FNMJ->EMBJ", sabej_S, tmp)

    # - U(en) R(mb,nj) -> umaei: -Σ_N smbij[N,J,M,B]·uei[E,N]
    umaei -= np.einsum("NJMB,EN->EMBJ", smbij, uei)
    # U(no,je) R(mb,on) -> umaei: Σ smbij[O,N,M,B]·uijke[N,O,J,E]
    umaei += np.einsum("ONMB,NOJE->EMBJ", smbij, uijke)

    # smbij spinad
    smbij_S = spinad(smbij, 0, 1)
    # - U(en) R(mb,jn) -> umaie: -Σ smbij_S[J,N,M,B]·uei[E,N]
    umaie -= np.einsum("JNMB,EN->EMBJ", smbij_S, uei)
    # U(no,ej) R(mb,on) -> umaie: Σ smbij_S[O,N,M,B]·uijke[O,N,J,E]
    umaie += np.einsum("ONMB,ONJE->EMBJ", smbij_S, uijke)

    umaei = 1.5 * umaei - 0.5 * umaie
    umaie = (1.0 / 3.0) * umaei + (2.0 / 3.0) * umaie
    return umaei, umaie


def build_gmaei_faithful(T, ujaei, ujaie, umabi, umaib, umaei, umaie,
                         X_MI, X_EA, nocc, nvir):
    """Literal port of gmaei_steom_rhf → returns G(EM,bj)=gmaei, G(Em,Bj)=gmaie
    as full [A,I,B,J] tensors."""
    gmaei = T["wmaei"].copy()   # [A,I,B,J] bare Wovvo
    gmaie = T["wmaie"].copy()   # [A,I,B,J] bare Wovov

    # + U(ej,ib) X(ea) -> gmaei:  Σ_E ujaei[E,I,B,J]·X_EA[E,A]
    gmaei += np.einsum("EIBJ,EA->AIBJ", ujaei, X_EA)
    gmaie += np.einsum("EIBJ,EA->AIBJ", ujaie, X_EA)

    # fold cross: umabi += umaei·X_EA ; umaib += umaie·X_EA  (per CFOUR)
    umabi_tot = umabi + np.einsum("EMBJ,EA->AMBJ", umaei, X_EA)
    umaib_tot = umaib + np.einsum("EMBJ,EA->AMBJ", umaie, X_EA)

    # - umabi_tot X(mi) -> gmaei:  -Σ_M umabi_tot[A,M,B,J]·X_MI[M,I]
    gmaei -= np.einsum("AMBJ,MI->AIBJ", umabi_tot, X_MI)
    gmaie -= np.einsum("AMBJ,MI->AIBJ", umaib_tot, X_MI)
    return gmaei, gmaie


def assemble_G(F_eff_oo, F_eff_vv, gmaei, gmaie, nocc, nvir):
    """Singlet G^{1h1p}[ia,jb] = F_eff_vv δ_ij - F_eff_oo δ_ab
                               + 2·gmaei[a,i,b,j] - gmaie[a,i,b,j]."""
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
                    val += 2.0 * gmaei[a, i, b, j]
                    val -= gmaie[a, i, b, j]
                    G[row, col] += val
    return G


def main(xyz_path, basis, n_act_occ, n_act_vir, n_steom):
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd

    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    mycc = cc.CCSD(mf); mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([nmo]*4)
    bar_h = build_bar_h(eri, mycc.t1, mycc.t2,
                        np.diag(mf.mo_energy[:nocc]), np.diag(mf.mo_energy[nocc:]),
                        nocc, nvir)

    eom_ip = eom_rccsd.EOMIP(mycc); e_ip, r_ip = eom_ip.kernel(nroots=n_act_occ)
    eom_ea = eom_rccsd.EOMEA(mycc); e_ea, r_ea = eom_ea.kernel(nroots=n_act_vir)
    eom_ee = eom_rccsd.EOMEESinglet(mycc); e_ee, _ = eom_ee.kernel(nroots=n_steom)
    e_ee = np.atleast_1d(np.asarray(e_ee))
    r1_ip, r2_ip = [], []
    for k in range(n_act_occ):
        rv = np.asarray(r_ip[k]).ravel()
        r1_ip.append(rv[:nocc]); r2_ip.append(rv[nocc:].reshape(nocc, nocc, nvir))
    r1_ea, r2_ea = [], []
    for k in range(n_act_vir):
        rv = np.asarray(r_ea[k]).ravel()
        r1_ea.append(rv[:nvir]); r2_ea.append(rv[nvir:].reshape(nocc, nvir, nvir))

    aocc = assign_active_1to1(r1_ip, nocc)
    avir = assign_active_1to1(r1_ea, nvir)

    # F^eff (verified correct) from build_g_singlet
    from pyscf_steom_feff_reference import build_x_matrices
    X_IP, X_EA_old = build_x_matrices(r1_ip, r1_ea, aocc, avir)
    _, F_eff_oo, F_eff_vv, _, _ = build_g_singlet(
        bar_h, r2_ip, r2_ea, X_IP, X_EA_old, aocc, avir, nocc, nvir)

    # CFOUR-layout X (full columns)
    X_MI = build_X(r1_ip, aocc, nocc)   # [M_act, I_full]
    X_EA = build_X(r1_ea, avir, nvir)   # [E_act, A_full]

    # Faithful W^eff
    T = build_cfour_tensors(bar_h, r2_ip, r2_ea, nocc, nvir, n_act_occ, n_act_vir)
    ujaei, ujaie = build_ujaei_faithful(T, n_act_vir, nocc, nvir)
    umabi, umaib = build_umabi_faithful(T, n_act_occ, nocc, nvir)
    uei, uam, uijke, uajim, uajmi = build_uei_uam_uijke_uajim_faithful(
        T, n_act_occ, n_act_vir, nocc, nvir)
    umaei, umaie = build_umaei_faithful(T, uei, uam, uijke, uajim, uajmi,
                                        n_act_occ, n_act_vir, nocc, nvir)
    gmaei, gmaie = build_gmaei_faithful(T, ujaei, ujaie, umabi, umaib,
                                        umaei, umaie, X_MI, X_EA, nocc, nvir)

    G = assemble_G(F_eff_oo, F_eff_vv, gmaei, gmaie, nocc, nvir)
    ef = np.sort(np.real(np.linalg.eigvals(G)))

    print("=" * 70)
    print(f"Faithful CFOUR W^eff:  {xyz_path} {basis} act=({n_act_occ},{n_act_vir})")
    print("=" * 70)
    print(f"  ‖ujaei‖={np.linalg.norm(ujaei):.6f}  ‖umabi‖={np.linalg.norm(umabi):.6f}"
          f"  ‖umaei‖={np.linalg.norm(umaei):.6f}")
    print(f"  {'st':>3} {'EOM-EE':>13} {'STEOM(faith)':>13} {'Δ(mHa)':>9}")
    nc = min(n_steom, len(ef), len(e_ee))
    for k in range(nc):
        print(f"  {k:>3} {e_ee[k]:>13.8f} {ef[k]:>13.8f} {1000*(ef[k]-e_ee[k]):>+9.2f}")
    print("\n  ORCA H2O sto-3g: 0.4354200 0.4998300 0.5916380")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
