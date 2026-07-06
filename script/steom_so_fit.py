#!/usr/bin/env python3
"""Derive the correct g_phhp EA s-route by fitting the det-oracle target in SPIN-ORBITALS.

g_phhp = Mc (cross-spin 1h1p block, <i^a_alpha|G|j^b_beta>). The EA route is the part
linear in s_EA:  target = Mc(Hbar + [S_ea,Hbar]) - Mc(Hbar).  Everything is in the
determinant oracle's own SO convention (oracle s_EA), so there is NO SO-vs-spatial
mismatch (the confound that blocks tensor-fitting GANSU's spatial routes). We fit this
target against SO contractions of s_EA with bar-H, read off the formula, then spin-adapt.

Stage 1 here: extract the target + sanity-check its norm vs the spin-adapted g_phhp EA
route (steom_cross_locate). Run: wsl python3 script/steom_so_fit.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p, occ_so, vir_so, so_index)
from steom_so_derive import det_singles_block, build_hbar, build_direct


def mc_block(data, dets, index, G):
    """cross-spin 1h1p block Mc[i,a,j,b] = <i^a_a|G|j^b_b> (spatial, dressing-relative)."""
    _, Mc, _ = det_singles_block(data, dets, index, G)
    return Mc


def main():
    atom = "; ".join(f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(6))
    data = get_active_data(atom=atom, basis="sto-3g", ncore=0)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    zEA = {e: np.zeros_like(sEA[e]) for e in sEA}
    S_ea = build_S(data, dets, index, zIP, sEA)

    # target: linear-in-s_EA cross-spin route.  [S_ea,Hbar] is exactly linear in s_EA.
    comm = S_ea @ Hbar - Hbar @ S_ea
    Mc_base = mc_block(data, dets, index, Hbar)
    Mc_lin = mc_block(data, dets, index, Hbar + comm)
    target = Mc_lin - Mc_base              # g_phhp EA route (SO cross-spin, [i,a,j,b])

    # cross-check: full only-S_ea g_phhp EA route (all orders) via 0.5(Gs-Gt)
    from scipy.linalg import expm
    def gphhp(S):
        Gs, Gt = project_1h1p(data, dets, index, expm(S) @ Hbar @ expm(-S))
        GsD = Gs - E_N*np.eye(nocc*nvir); GtD = Gt - E_N*np.eye(nocc*nvir)
        g = np.zeros((nvir, nocc, nocc, nvir))
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        g[b, j, i, a] = 0.5*(GsD[i*nvir+a, j*nvir+b] - GtD[i*nvir+a, j*nvir+b])
        return g
    ea_full = gphhp(S_ea) - gphhp(build_S(data, dets, index, zIP, zEA))
    # map target Mc[i,a,j,b] to g_phhp[b,j,i,a] layout for comparison
    tgt_g = np.einsum("iajb->bjia", target)

    print(f"H6  nocc={nocc} nvir={nvir}")
    print(f"  ||Mc_base (g_phhp base, cross-spin)|| = {np.linalg.norm(Mc_base):.4f}")
    print(f"  ||target (EA route, linear s_EA)||    = {np.linalg.norm(target):.4f}")
    print(f"  ||ea_full (all-order only-S_ea)||     = {np.linalg.norm(ea_full):.4f}")

    # ---------- Stage 2: fit target against SO candidate contractions ----------
    from steom_so_derive import build_direct, build_so_integrals
    nact = data["nact"]
    g, f = build_so_integrals(data); D = build_direct(data)
    occ = occ_so(data); vir = vir_so(data)
    nso = 2 * nact
    sea = np.zeros((nso, nso, nso, nso))         # sea[E,J,A,B], E,A,B vir-so, J occ-so
    for e in sEA:
        sea[e] = sEA[e]
    Ia = lambda i: so_index(i, 0, nact); Aa = lambda a: so_index(a+nocc, 0, nact)
    Jb = lambda j: so_index(j, 1, nact); Bb = lambda b: so_index(b+nocc, 1, nact)

    def build(fn):
        out = np.zeros((nocc, nvir, nocc, nvir))
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        out[i, a, j, b] = fn(Ia(i), Aa(a), Jb(j), Bb(b))
        return out

    V = vir; O = occ
    # candidate contractions (sea on beta/ket side: root E=B_beta, occ=J_beta); direct D + f.
    pats = {
        "f[I,C]sea[B,J,A,C]":       lambda I,A,J,B: sum(f[I,C]*sea[B,J,A,C] for C in V),
        "f[I,C]sea[B,J,C,A]":       lambda I,A,J,B: sum(f[I,C]*sea[B,J,C,A] for C in V),
        "D[I,A,C,Dd]sea[B,J,C,Dd]": lambda I,A,J,B: sum(D[I,A,C,Dd]*sea[B,J,C,Dd] for C in V for Dd in V),
        "D[I,C,A,Dd]sea[B,J,C,Dd]": lambda I,A,J,B: sum(D[I,C,A,Dd]*sea[B,J,C,Dd] for C in V for Dd in V),
        "D[I,C,Dd,A]sea[B,J,C,Dd]": lambda I,A,J,B: sum(D[I,C,Dd,A]*sea[B,J,C,Dd] for C in V for Dd in V),
        "D[L,I,J,C]sea[B,L,A,C]":   lambda I,A,J,B: sum(D[L,I,J,C]*sea[B,L,A,C] for L in O for C in V),
        "D[I,L,J,C]sea[B,L,A,C]":   lambda I,A,J,B: sum(D[I,L,J,C]*sea[B,L,A,C] for L in O for C in V),
        "D[I,A,L,C]sea[B,L,J?,C]xx":lambda I,A,J,B: sum(D[I,A,L,C]*sea[B,L,C,C] for L in O for C in V),
    }
    cols = []; names = []
    for nm, fn in pats.items():
        t = build(fn)
        if np.linalg.norm(t) < 1e-12:
            continue
        cols.append(t.ravel()); names.append(nm)
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, target.ravel(), rcond=None)
    resid = np.linalg.norm(A @ coef - target.ravel()) / np.linalg.norm(target.ravel())
    print(f"\n  SO fit: {len(names)} candidates, rel-resid = {resid:.3e}")
    for nm, c in sorted(zip(names, coef), key=lambda z: -abs(z[1])):
        if abs(c) > 1e-3:
            print(f"    {nm:34s} c={c:+.4f}")


if __name__ == "__main__":
    main()
