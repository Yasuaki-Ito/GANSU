#!/usr/bin/env python3
"""Eigenvalue-driven harness for the STEOM g_phph EA route (u_akei) derivation.

The per-route oracle isolation diagonal is unreliable (normalization mismatch
for low-%singles / near-degenerate roots — even the VALIDATED IP u_amci diagonal
is off at o2). The ROBUST acceptance metric is the full-G eigenvalues vs the
determinant {e^S} oracle (== ORCA for H2O sto-3g FC1: IROOT1 11.849, IROOT2 13.60).

This harness assembles the analytic singlet/triplet G with swappable route
tensors and reports eigenvalues, so candidate u_akei forms are scored on
eigenvalues (+ the trustworthy diagonal elements o1/o3 only).

  wsl python3 script/steom_eig_harness.py [ncore] [xyz]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full, build_normalized_s
from steom_route_probe import route_tensors
from steom_es_oracle import _amci_wooov

Ha = 27.211386245988
np.set_printoptions(precision=4, suppress=True, linewidth=170)


def ea_term(d, fn):
    """scatter an EA-route block fn(s,st)->[a,k,i] into g_phph[a,k,e=vir_idx,i]."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]; vir_idx = d["vir_idx"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], vir_idx, nocc, nvir)
    T = np.zeros((nvir, nocc, nvir, nocc))
    for e in range(len(s_EA)):
        s = s_EA[e]; st = 2.0 * s - s.transpose(0, 2, 1)
        T[:, :, vir_idx[e], :] += fn(s, st, bar)
    return T


def assemble(d, u_akei_new):
    """Build singlet & triplet G with: base + u_amci(Wooov) + u_akei_new (g_phph),
    keeping the current g_phhp + cross. Returns (Gs, Gt)."""
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    # current full analytic
    Gs_cur, g_phph_cur, g_phhp_cur, u_amei, u_bmje, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    base, t_amci_bug, t_akei_cur = route_tensors(d)
    t_amci_fix = _amci_wooov(d)
    # swap IP route (bug->fix) and EA route (cur->new) in g_phph
    g_phph = g_phph_cur - t_amci_bug + t_amci_fix - t_akei_cur + u_akei_new
    g_phhp = g_phhp_cur
    Foo, Fvv = C.build_feff(d)
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    Gs[r, c] = fdg + 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                    Gt[r, c] = fdg - g_phph[a, j, b, i]
    return Gs, Gt


def eigs(G):
    return np.sort(np.linalg.eigvals(G).real)


def determinant_gold(ncore, xyz, basis):
    """determinant {e^S} oracle eigenvalues (gold standard == ORCA)."""
    import steom_es_oracle as O
    res = O.run(ncore=ncore, xyz=xyz, basis=basis)
    return res["es_e"], res["et_e"], res["es_p"], res["et_p"]


# library of candidate u_akei EA-route blocks fn(s,st,bar)->[a,k,i]
# s = s_EA[e][i,a,c] (occ i, vir a,c) ; st = 2s - s.swap(vir pair)
def CAND():
    lib = {}
    lib["cur_T1 Fov.s"]   = lambda s, st, b: -np.einsum("kc,iac->aki", b["Fov"], s)
    lib["cur_T2 Wovoo.st"]= lambda s, st, b:  np.einsum("ldki,lad->aki", b["Wovoo"], st)
    lib["cur_T3 Wooov.s"] = lambda s, st, b: -np.einsum("lkid,lad->aki", b["Wooov"], s)
    lib["cur_T4 Wvovv.s"] = lambda s, st, b:  np.einsum("akcd,icd->aki", b["Wvovv"], s)
    return lib


def current_uakei(d):
    _, _, t_akei_cur = route_tensors(d)
    return t_akei_cur


def cfour_ujaie(d):
    """CFOUR ujaie (0.5A+0.5B) folded into g_phph[a,j,b,i] EA route."""
    T = C.cfour_tensors(d)
    _, ujaie = C.build_ujaei(T)            # [E,I,B,J]
    X_EA = d["X_EA"]
    gmaie_ea = np.einsum("EIBJ,EA->AIBJ", ujaie, X_EA)
    return np.einsum("AIBJ->AJBI", gmaie_ea)   # [a,j,b,i]


def diag_o(T, d):
    nocc = d["nocc"]; nvir = d["nvir"]
    return np.array([[T[a, i, a, i] for a in range(nvir)] for i in range(nocc)])


def cfour_ujaei_phhp(d):
    """CFOUR ujaei (1.5A-0.5B) folded into g_phhp[b,j,i,a] EA route."""
    T = C.cfour_tensors(d)
    ujaei, _ = C.build_ujaei(T)            # [E,I,B,J]
    X_EA = d["X_EA"]
    gmaei_ea = np.einsum("EIBJ,EA->AIBJ", ujaei, X_EA)
    return np.einsum("AIBJ->BJIA", gmaei_ea)   # [b,j,i,a]


def bmjc_routes(d):
    """IP g_phhp route u_bmjc current (Wovoo) and Wooov-fixed, as [b,j,i,a] tensors."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wovoo = bar["Wovoo"]; Wooov = bar["Wooov"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    occ_idx, vir_idx, nocc, nvir)
    cur = np.zeros((nvir, nocc, nocc, nvir))
    fix = np.zeros((nvir, nocc, nocc, nvir))
    for m in range(len(s_IP)):
        s = s_IP[m]                                  # [k,j,b]
        blk_cur = (-np.einsum("kc,kjb->bjc", Fov, s)
                   + np.einsum("kclj,klb->bjc", Wovoo, s)
                   - np.einsum("bkdc,kjd->bjc", Wvovv, s))   # [b,j,c]
        # Wooov fix mirror: replace Wovoo term with Wooov
        blk_fix = (-np.einsum("kc,kjb->bjc", Fov, s)
                   + np.einsum("kljc,klb->bjc", Wooov, s)
                   - np.einsum("bkdc,kjd->bjc", Wvovv, s))
        # u_bmjc[b,m,j,c] -> g_phhp[b, k_full=m, j, c]; g_phhp layout [b,k,j,c]=[b,j,i,a]?
        # build_g_canonical_full: g_phhp[:, k_full, :, :] += u_bmjc[:, m, :, :]; g_phhp is [b,k,j,c]
        cur[:, occ_idx[m], :, :] += blk_cur
        fix[:, occ_idx[m], :, :] += blk_fix
    return cur, fix


def assemble_full(d, ea_phph, ea_phhp, ip_phhp_fix=True):
    """Full singlet/triplet G with all 4 routes swappable.
    ea_phph: EA g_phph route [a,j,b,i].  ea_phhp: EA g_phhp route [b,j,i,a].
    ip_phhp_fix: use Wooov fix for IP g_phhp route u_bmjc."""
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    Gs_cur, g_phph_cur, g_phhp_cur, u_amei, u_bmje, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    base, t_amci_bug, t_akei_cur = route_tensors(d)
    t_amci_fix = _amci_wooov(d)
    # g_phph: swap IP bug->fix, EA cur->ea_phph
    g_phph = g_phph_cur - t_amci_bug + t_amci_fix - t_akei_cur + ea_phph
    # g_phhp: need current EA route and IP route to subtract. Reconstruct from decomp.
    _, _, _, _, _, decomp = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    hp_base, hp_ip_cur, hp_ea_cur, hp_cross = decomp  # [b,k,j,c]=[b,j,i,a]
    bmjc_cur, bmjc_fix = bmjc_routes(d)
    ip_phhp = bmjc_fix if ip_phhp_fix else bmjc_cur
    g_phhp = hp_base + ip_phhp + ea_phhp + hp_cross
    Foo, Fvv = C.build_feff(d)
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    Gs[r, c] = fdg + 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                    Gt[r, c] = fdg - g_phph[a, j, b, i]
    return Gs, Gt


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    xyz = sys.argv[2] if len(sys.argv) > 2 else "xyz/H2O_asym.xyz"
    basis = "sto-3g"
    d = C.load(xyz, basis, ncore)
    nocc, nvir = d["nocc"], d["nvir"]
    print(f"\n== {xyz} {basis} ncore={ncore}  nocc={nocc} nvir={nvir} ==")

    es_e, et_e, es_p, et_p = determinant_gold(ncore, xyz, basis)
    print("GOLD determinant {e^S} singlet:", np.round(es_e, 3))
    print("GOLD determinant plain  singlet:", np.round(es_p, 3))
    print("GOLD determinant {e^S} triplet:", np.round(et_e, 3))
    if ncore == 1 and "asym" not in xyz:
        print("ORCA (h2o FC1): IROOT1=11.849 IROOT2=13.60")

    print(f"\n  GOLD plain  triplet[0]={et_p[0]:.3f}  singlet[0]={es_p[0]:.3f}")
    print(f"  GOLD {{e^S}} triplet[0]={et_e[0]:.3f}  singlet[0]={es_e[0]:.3f}")
    H = nocc - 1; L = 0
    print(f"  (HOMO o{H} -> LUMO v{L} diagonal target ~ -0.524 trustworthy)\n")

    cands = {
        "current": current_uakei(d),
        "current x2": 2.0 * current_uakei(d),
        "CFOUR ujaie": cfour_ujaie(d),
        "CFOUR x0.85": 0.85 * cfour_ujaie(d),
    }
    print(f"  {'candidate':18s} {'trip[0]':>8s} {'sing[0]':>8s} {'o3 diag(HL)':>12s} {'o1 diag':>9s}")
    for nm, u in cands.items():
        Gs, Gt = assemble(d, u)
        es = eigs(Gs) * Ha; et = eigs(Gt) * Ha
        dg = diag_o(u, d) * Ha
        print(f"  {nm:18s} {et[0]:8.3f} {es[0]:8.3f} {dg[H, L]:12.3f} {dg[1, L]:9.3f}")
    print(f"  {'TARGET(gold plain)':18s} {et_p[0]:8.3f} {es_p[0]:8.3f} {-0.524:12.3f} {-0.730:9.3f}")


if __name__ == "__main__":
    main()
