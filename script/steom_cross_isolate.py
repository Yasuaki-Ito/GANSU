#!/usr/bin/env python3
"""Exactly isolate the 2nd-order g_phhp components (cross S^IP·S^EA, same-type
S^IP·S^IP and S^EA·S^EA) from the determinant {e^S} oracle, to characterise which
piece GANSU is missing and whether it has a clean closed-shell form.

2nd-order analytic operator (what Nooijen Eq.56-63 truncates to):
    Q(S) = 1/2 [S,[S,H]] - 1/2 (K H + H K),   K = S^2 - {S^2}   (depends on S)
Q is EXACTLY quadratic in S ⇒ bilinear isolation is exact:
    cross    = Q(sIP+sEA) - Q(sIP) - Q(sEA)      (S^IP·S^EA)
    quad_IP  = Q(sIP)                             (S^IP·S^IP)
    quad_EA  = Q(sEA)                             (S^EA·S^EA)
g_phhp piece of each = 0.5(Qs - Qt) projected to 1h1p, layout [b,j,i,a].

Run:  wsl python3 script/steom_cross_isolate.py 1
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
from steom_es_oracle import s_terms, build_K, build_S_from_terms

Ha = 27.211386245988
np.set_printoptions(precision=4, suppress=True, linewidth=170)


def Q_of(data, dets, index, H, sIP, sEA):
    """2nd-order operator Q(S) = 1/2[S,[S,H]] - 1/2(KH+HK)."""
    terms = s_terms(data, sIP, sEA)
    S = build_S(data, dets, index, sIP, sEA)
    K = build_K(data, dets, index, terms)
    SH = S @ H - H @ S                 # [S,H]
    comm2 = S @ SH - SH @ S            # [S,[S,H]]
    return 0.5 * comm2 - 0.5 * (K @ H + H @ K)


def gphhp_of(data, dets, index, Qfull, nocc, nvir):
    """extract g_phhp[b,j,i,a] = 0.5(Qs - Qt) from a 2nd-order op (no E_N shift)."""
    Qs, Qt = project_1h1p(data, dets, index, Qfull)
    g = np.zeros((nvir, nocc, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    g[b, j, i, a] = 0.5 * (Qs[r, c] - Qt[r, c])
    return g


def off_norm(T, nocc, nvir):
    s = 0.0
    for b in range(nvir):
        for j in range(nocc):
            for i in range(nocc):
                for a in range(nvir):
                    if (i != j) and (a != b):
                        s += T[b, j, i, a] ** 2
    return np.sqrt(s)


def report(name, T, nocc, nvir):
    print(f"  {name:14s} ||T||={np.linalg.norm(T):.4f}  off-diag(i!=j&a!=b)={off_norm(T,nocc,nvir):.4f}")


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    xyz = "xyz/H2O.xyz"; basis = "sto-3g"
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nocc = data["nocc"]; nvir = data["nvir"]
    print(f"== {xyz} {basis} ncore={ncore}  nocc={nocc} nvir={nvir}")
    dets, index, H = build_sector(data, data["nelec"])
    hf = hf_det(data); E_N = H[index[hf], index[hf]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    z_ip = {m: np.zeros_like(sIP[m]) for m in sIP}
    z_ea = {e: np.zeros_like(sEA[e]) for e in sEA}

    print("\n  building Q(both), Q(ip), Q(ea) ...")
    Q_both = Q_of(data, dets, index, H, sIP, sEA)
    Q_ip = Q_of(data, dets, index, H, sIP, z_ea)
    Q_ea = Q_of(data, dets, index, H, z_ip, sEA)

    g_cross = gphhp_of(data, dets, index, Q_both - Q_ip - Q_ea, nocc, nvir)
    g_qip = gphhp_of(data, dets, index, Q_ip, nocc, nvir)
    g_qea = gphhp_of(data, dets, index, Q_ea, nocc, nvir)
    g_2nd = g_cross + g_qip + g_qea

    print("\n##### exact 2nd-order g_phhp components (H2O sto-3g FC%d) #####" % ncore)
    report("cross(IP*EA)", g_cross, nocc, nvir)
    report("same-type IP", g_qip, nocc, nvir)
    report("same-type EA", g_qea, nocc, nvir)
    report("TOTAL 2nd", g_2nd, nocc, nvir)

    # eigenvalue impact: what does adding the 2nd-order do to the singlet?
    # (rough: report Frobenius contributions; eigenvalue test is in es_oracle)
    print("\n  interpretation: GANSU shipped has base+IP-linear (u_bmjc); it is MISSING")
    print("  EA-linear (u_bkje, small) + these 2nd-order pieces. The one with large")
    print("  off-diagonal weight drives config-mixed states (naphthalene Lb).")


if __name__ == "__main__":
    main()
