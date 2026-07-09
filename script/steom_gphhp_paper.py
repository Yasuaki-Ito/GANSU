#!/usr/bin/env python3
"""Faithful transcription of Nooijen & Bartlett JCP 107, 6812 (1997) Eqs. (38)-(63)
for the COMPLETE phhp (g_phhp) block, gated against ORCA 8+8 truth (続50).

Conventions (verified via Eq.(56) == shipped u_amci rosetta):
  - bar tensors are physicist 1212 as named: Wvovv[a,l,c,d]=<al|cd>,
    Wooov[k,l,i,c]=<kl|ic>, Fov[k,c]; eri_ovov[i,a,j,b]=(ia|jb)=<ij|ab> (bare).
  - sM[m][i,j,b] = s^{mb}_{ij}  (pairs (m,i),(b,j); rinv/X folded in)
  - sE[e][j,a,b] = s^{ab}_{ej}  (pairs (a,e),(b,j))
  - s~ IP: 2s - s.transpose(1,0,2) ; s~ EA: 2s - s.transpose(0,2,1)
  - paper g_bkjc -> GANSU g_phhp[b, k, j, c] wait: mapping g_phhp[b,jG,iG,aG] with
    jG=k (scattered occ), iG=j, aG=c (scattered vir); base w_bkjc=Wovvo[k,b,c,j].

Ambiguity toggles (GATE decides):
  A: (62) term4 u_lmjd: 0 = (43)-family (root slot2), 1 = (42) pair-symmetric
  B: (62) term3 u_klej: 0 = pair-reading (k,e),(l,j), 1 = literal (44) slots
  C: base: 0 = paper w_bkjc = Wovvo[k,b,c,j], 1 = EE-fix Wovvo[k,c,b,j]

Run:  wsl python3 script/steom_gphhp_paper.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full, build_normalized_s
from steom_gphhp_gate import restrict, assemble, eig_lab, ORCA_S, ORCA_T, Ha2eV


def build_paper_gphhp(d, ambA=0, ambB=0, ambC=0, fcross=1.0, fea=1.0, fip=1.0,
                      vswap=()):
    """vswap: subset of {'ip','ea','cross'} — place that route with the two
    VIRTUAL roles exchanged (amplitude-vir -> bra slot, integral/scattered vir
    -> ket slot), i.e. the ST-CC2-appendix convention.  Our g layout:
    g[ketvir b, kethole k, brahole j, bravir c]."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    Wovvo = bar["Wovvo"]; eri = bar["eri_ovov"]      # eri[i,a,j,b] = <ij|ab>
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    sM, sE = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                occ_idx, vir_idx, nocc, nvir)
    nM = len(sM); nE = len(sE)

    # ---- base, Eq.(63) w_bkjc ----
    if ambC == 0:
        g = np.einsum("kbcj->bkjc", Wovvo).copy()       # w_bkjc = Wovvo[k,b,c,j]
    else:
        g = np.einsum("kcbj->bkjc", Wovvo).copy()       # EE-fix variant
    # reorder to g_phhp[b, jG=k, iG=j, aG=c]
    g = np.einsum("bkjc->bkjc", g)                       # already [b,k,j,c]

    # ---- Eq.(60): u_bmjc (S^IP linear), scatter k=occ_idx[m] ----
    for m in range(nM):
        s = sM[m]                                        # s^{mb}_{ij} = s[i,j,b]
        u = -np.einsum("kc,kjb->bjc", Fov, s)            # -w_kc s^{mb}_{kj}
        u += np.einsum("lkjc,klb->bjc", Wooov, s)        # +w_klcj s^{mb}_{kl} (<kl|cj>=Wooov[l,k,j,c])
        u -= np.einsum("bkdc,kjd->bjc", Wvovv, s)        # -w_kbcd s^{md}_{kj} (<kb|cd>=Wvovv[b,k,d,c])
        if "ip" in vswap:
            g[:, occ_idx[m], :, :] += fip * u.transpose(2, 1, 0)   # amp-vir b -> bra slot
        else:
            g[:, occ_idx[m], :, :] += fip * u
    # ---- Eq.(61): u_bkje (S^EA linear), scatter c=vir_idx[e] ----
    for e in range(nE):
        s = sE[e]                                        # s^{ab}_{ej} = s[j,a,b]
        u = np.einsum("kd,jdb->bkj", Fov, s)             # +w_kd s^{db}_{ej}
        u += np.einsum("bkdc,jcd->bkj", Wvovv, s)        # +w_bkdc s^{cd}_{ej}
        u -= np.einsum("lkjd,ldb->bkj", Wooov, s)        # -w_lkjd s^{db}_{el}
        if "ea" in vswap:
            g[vir_idx[e], :, :, :] += fea * u.transpose(1, 2, 0)   # e -> ket slot
        else:
            g[:, :, :, vir_idx[e]] += fea * u
    # ---- Eq.(62): u_bmje (cross), scatter k=occ_idx[m], c=vir_idx[e] ----
    # helpers (38)/(39)/(44)/(42)/(43) with bare v = eri_ovov
    u_ma = np.zeros((nM, nvir))
    for m in range(nM):
        st = 2.0 * sM[m] - sM[m].transpose(1, 0, 2)
        u_ma[m] = -np.einsum("kald,kld->a", eri, st)     # (38)
    u_ie = np.zeros((nE, nocc))
    for e in range(nE):
        st = 2.0 * sE[e] - sE[e].transpose(0, 2, 1)
        u_ie[e] = np.einsum("kcld,lcd->k", eri, st)      # (39) u_ke
    for m in range(nM):
        sm = sM[m]
        smt = 2.0 * sm - sm.transpose(1, 0, 2)
        # u_lmjd per (42)/(43) family (toggle A)
        if ambA == 0:
            # (43)-family root slot2: u_lmjd = -sum_{p,b} v_pldb s^{mb}_{pj}
            u_lmjd = -np.einsum("pdlb,pjb->ljd", eri, sm)
        else:
            # (42) with pair symmetry: u_lmjd = u_mldj?? use (42) tilde structure:
            # u_mlid = sum_{j,b} ( v_jlbd s~^{mb}_{ij} - v_ljbd s^{mb}_{ij} )
            # read u_lmjd = (42) tensor with (l,d) pair and free occ j:
            u_lmjd = (np.einsum("pblc,pjb->ljc", eri, smt)
                      - np.einsum("lbpc,pjb->ljc", eri, sm))
        for e in range(nE):
            se = sE[e]
            # (44): u_klej with pair-reading toggle B
            if ambB == 0:
                # pairs (k,e),(l,j): u_klej = sum_ab v_klab s^{ab}_{?}
                # rootpair virtual on k side: <kl|ab>: a with k -> a pairs root e
                u_klej = np.einsum("kalb,jab->klj", eri, se)
            else:
                u_klej = np.einsum("kalb,jba->klj", eri, se)
            u = np.einsum("d,jdb->bj", u_ma[m], se)          # +u_md s^{db}_{ej}
            u -= np.einsum("k,kjb->bj", u_ie[e], sm)         # -u_ke s^{mb}_{kj}
            u += np.einsum("klj,klb->bj", u_klej, sm)        # +u_klej s^{mb}_{kl}
            u -= np.einsum("ljd,ldb->bj", u_lmjd, se)        # -u_lmjd s^{db}_{el}
            if "cross" in vswap:
                g[vir_idx[e], occ_idx[m], :, :] += fcross * u.T
            else:
                g[:, occ_idx[m], :, vir_idx[e]] += fcross * u
    return g


def build_paper_gphph(d, with_cross=True):
    """Paper Eqs.(56)-(59): complete phph block g_phph[a,j,b,i] (=g_akci with
    k->j-slot, c->b-slot).  Base w_akci = Wovov[k,a,i,c] (shipped-validated)."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    Wovov = bar["Wovov"]; eri = bar["eri_ovov"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    sM, sE = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                occ_idx, vir_idx, nocc, nvir)
    nM = len(sM); nE = len(sE)
    g = np.einsum("kaic->akci", Wovov).copy()            # [a,k,c,i]
    # (56) u_amci, scatter k=occ_idx[m]
    for m in range(nM):
        s = sM[m]; st = 2.0 * s - s.transpose(1, 0, 2)
        u = -np.einsum("kc,ika->aci", Fov, s)
        u += np.einsum("alcd,ild->aci", Wvovv, st)
        u -= np.einsum("aldc,ild->aci", Wvovv, s)
        u += np.einsum("lkic,lka->aci", Wooov, s)
        g[:, occ_idx[m], :, :] += u
    # (57) u_akei (misprint-fixed), scatter c=vir_idx[e]
    for e in range(nE):
        s = sE[e]; st = 2.0 * s - s.transpose(0, 2, 1)
        u = -np.einsum("kc,iac->aki", Fov, s)
        u += np.einsum("klid,lad->aki", Wooov, st)
        u -= np.einsum("lkid,lad->aki", Wooov, s)
        u += np.einsum("akcd,icd->aki", Wvovv, s)
        g[:, :, vir_idx[e], :] += u
    if not with_cross:
        return g
    # (58) u_amei cross, scatter k=occ_idx[m], c=vir_idx[e]
    u_ma = np.zeros((nM, nvir)); u42 = []; u43 = []
    for m in range(nM):
        s = sM[m]; st = 2.0 * s - s.transpose(1, 0, 2)
        u_ma[m] = -np.einsum("kald,kld->a", eri, st)                     # (38)
        u42.append(np.einsum("pbld,ipb->lid", eri, st)
                   - np.einsum("lbpd,ipb->lid", eri, s))                 # (42) u_mlid
        u43.append(-np.einsum("pdlb,pib->lid", eri, s))                  # (43) u_lmid
    u_ie = np.zeros((nE, nocc)); u44 = []
    for e in range(nE):
        s = sE[e]; st = 2.0 * s - s.transpose(0, 2, 1)
        u_ie[e] = np.einsum("kcld,lcd->k", eri, st)                      # (39)
        u44.append(np.einsum("kalb,iba->kli", eri, s))                   # (44) u_klie
    for m in range(nM):
        sm = sM[m]
        for e in range(nE):
            se = sE[e]; set_ = 2.0 * se - se.transpose(0, 2, 1)
            u = np.einsum("c,iac->ai", u_ma[m], se)          # +u_mc s^{ac}_{ei}
            u -= np.einsum("k,ika->ai", u_ie[e], sm)         # -u_ke s^{ma}_{ik}
            u += np.einsum("lid,lad->ai", u42[m], set_)      # +u_lmdi s~^{ad}_{el}
            u -= np.einsum("lid,lad->ai", u43[m], se)        # -u_lmid s^{ad}_{el}
            T = u44[e].transpose(1, 0, 2)                    # u_klei[k,l,i]=u44[l,k,i]
            u += np.einsum("kli,lka->ai", T, sm)             # +u_klei s^{ma}_{lk}
            g[:, occ_idx[m], vir_idx[e], :] += u
    return g


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    d = C.load("xyz/H2O.xyz", "sto-3g", 1)
    d = restrict(d, occ_keep=(1, 2, 3), vir_keep=(0, 1))
    nocc = d["nocc"]; nvir = d["nvir"]
    Foo, Fvv = C.build_feff(d)
    target = np.array(ORCA_S[:6])

    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)

    targT = np.array(ORCA_T[:6])

    def score(name, ghp, gph):
        Gs, Gt = assemble(d, ghp, gph, Foo, Fvv)
        ws, ims, _ = eig_lab(Gs, nvir)
        wt, imt, _ = eig_lab(Gt, nvir)
        rms = np.sqrt(np.mean((ws[:6] - target) ** 2))
        rmt = np.sqrt(np.mean((wt[:6] - targT) ** 2))
        im = f" Im={ims:.3f}" if ims > 1e-3 else ""
        print(f"{name:28s} S: " + " ".join(f"{x:7.3f}" for x in ws)
              + f"  rms6={rms:.4f} rmsT={rmt:.4f}{im}")
        return rms

    print("ORCA    S:", " ".join(f"{x:7.3f}" for x in ORCA_S))
    print("ORCA    T:", " ".join(f"{x:7.3f}" for x in ORCA_T))
    score("shipped", g_hp_a, g_ph_a)
    for ambA in (0, 1):
        for ambB in (0, 1):
            ghp = build_paper_gphhp(d, ambA, ambB, 1)
            score(f"pap-hp A{ambA}B{ambB} + ship-ph", ghp, g_ph_a)
    gph_nc = build_paper_gphph(d, with_cross=False)
    gph_full = build_paper_gphph(d, with_cross=True)
    for ambA in (0, 1):
        ghp = build_paper_gphhp(d, ambA, 0, 1)
        score(f"pap-hp A{ambA} + pap-ph nocr", ghp, gph_nc)
        score(f"pap-hp A{ambA} + pap-ph full", ghp, gph_full)


if __name__ == "__main__":
    main()
