#!/usr/bin/env python3
"""Bounded slot-reading scan for the paper g_phhp transcription (続52 residual).

Suspicious spots (amplitude pair-slot readings, one toggle at a time):
  S1: (61) term1  fov(k,d)*sE[e][j,d,b]  vs  sE[e][j,b,d]
  S2: (61) term3  wooov*sE[e][l,d,b]     vs  sE[e][l,b,d]
  S3: (62) term1  u_ma[d]*sE[e][j,d,b]   vs  sE[e][j,b,d]
  S4: (62) term2  u_ie[k]*sM[m][k,j,b]   vs  sM[m][j,k,b]
  S5: (61) term2  wvovv(b,k,d,c)*sE[e][j,c,d] vs sE[e][j,d,c]
Judged on BOTH gates: H2O FC1 (ORCA 8 singlets) and zigzag C6H8 (ORCA 5 roots).
A variant must improve BOTH to be accepted.

Run:  wsl python3 script/steom_gphhp_slotscan.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full, build_normalized_s
from steom_gphhp_gate import restrict, assemble, eig_lab, ORCA_S, Ha2eV
from steom_gphhp_paper import build_paper_gphph
from steom_hex_gate import light_load, ATOM, ORCA_HEX_S


def build_gphhp_var(d, tog=()):
    """paper g_phhp with slot toggles; base=EE, A0/B0."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    Wovvo = bar["Wovvo"]; eri = bar["eri_ovov"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    sM, sE = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                occ_idx, vir_idx, nocc, nvir)
    nM = len(sM); nE = len(sE)
    T = lambda k: k in tog
    g = np.einsum("kcbj->bkjc", Wovvo).copy()
    for m in range(nM):
        s = sM[m]
        u = -np.einsum("kc,kjb->bjc", Fov, s)
        u += np.einsum("lkjc,klb->bjc", Wooov, s)
        u -= np.einsum("bkdc,kjd->bjc", Wvovv, s)
        g[:, occ_idx[m], :, :] += u
    for e in range(nE):
        s = sE[e]
        u = np.einsum("kd,jdb->bkj", Fov, s if not T("S1") else s.transpose(0, 2, 1))
        u += np.einsum("bkdc,jcd->bkj", Wvovv, s if not T("S5") else s.transpose(0, 2, 1))
        u -= np.einsum("lkjd,ldb->bkj", Wooov, s if not T("S2") else s.transpose(0, 2, 1))
        g[:, :, :, vir_idx[e]] += u
    u_ma = np.zeros((nM, nvir)); u_ie = np.zeros((nE, nocc))
    for m in range(nM):
        st = 2.0 * sM[m] - sM[m].transpose(1, 0, 2)
        u_ma[m] = -np.einsum("kald,kld->a", eri, st)
    for e in range(nE):
        st = 2.0 * sE[e] - sE[e].transpose(0, 2, 1)
        u_ie[e] = np.einsum("kcld,lcd->k", eri, st)
    for m in range(nM):
        sm = sM[m]
        u_lmjd = -np.einsum("pdlb,pjb->ljd", eri, sm)          # A0 = UKMID
        for e in range(nE):
            se = sE[e]
            u_klej = np.einsum("kalb,jab->klj", eri, se)       # B0 = UKLIE
            u = np.einsum("d,jdb->bj", u_ma[m],
                          se if not T("S3") else se.transpose(0, 2, 1))
            u -= np.einsum("k,kjb->bj", u_ie[e],
                           sm if not T("S4") else sm.transpose(1, 0, 2))
            u += np.einsum("klj,klb->bj", u_klej, sm)
            u -= np.einsum("ljd,ldb->bj", u_lmjd, se)
            g[:, occ_idx[m], :, vir_idx[e]] += u
    return g


def gate_rms(d, ghp, gph, Foo, Fvv, target, n):
    Gs, _ = assemble(d, ghp, gph, Foo, Fvv)
    ws = np.sort(np.linalg.eigvals(Gs).real) * Ha2eV
    return float(np.sqrt(np.mean((ws[:n] - np.array(target[:n])) ** 2)))


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    # gate 1: H2O
    d1 = C.load("xyz/H2O.xyz", "sto-3g", 1)
    d1 = restrict(d1, occ_keep=(1, 2, 3), vir_keep=(0, 1))
    F1 = C.build_feff(d1); gph1 = build_paper_gphph(d1, with_cross=True)
    # gate 2: hex
    d2 = light_load(ATOM)
    d2 = restrict(d2, occ_keep=tuple(range(d2["nocc"] - 6, d2["nocc"])),
                  vir_keep=tuple(range(8)))
    F2 = C.build_feff(d2); gph2 = build_paper_gphph(d2, with_cross=True)

    variants = [(), ("S1",), ("S2",), ("S3",), ("S4",), ("S5",)]
    for tog in variants:
        g1 = build_gphhp_var(d1, tog)
        g2 = build_gphhp_var(d2, tog)
        r1 = gate_rms(d1, g1, gph1, F1[0], F1[1], ORCA_S, 6)
        r2 = gate_rms(d2, g2, gph2, F2[0], F2[1], ORCA_HEX_S, 5)
        print(f"tog={str(tog):10s}  H2O rms6={r1:.4f}   hex rms5={r2:.4f}")


if __name__ == "__main__":
    main()
