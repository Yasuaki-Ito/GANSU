#!/usr/bin/env python3
"""Pin the config-mixed overshoot to the g_phhp CROSS route, in the det oracle.

Reconstructs the singlet 1h1p spectrum from EXACT (route-isolated) g_phph/g_phhp:
  target        = F + 2 g_phhp(full)          - g_phph(full)    [= plain oracle = ORCA]
  drop_gphhp_X  = F + 2 g_phhp(base+IP+EA)    - g_phph(full)    [zero only cross]
  drop_gphhp_EAX= F + 2 g_phhp(base+IP)       - g_phph(full)    [zero EA+cross = GANSU g_phhp]
Route isolation via only-S^IP / only-S^EA transforms (F,base cancel in differences).
If drop_gphhp_EAX reproduces the config-mixed overshoot => the fix is the g_phhp cross
term u_bmje (underived). Run: wsl python3 script/steom_cross_locate.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
Ha = 27.211386245988


def blocks(data, dets, index, S, Hbar, E_N):
    """return F-free g_phph_true[a,j,b,i], g_phhp_true[b,j,i,a] and F (Foo,Fvv-diag)
    from GsD/GtD of the transform e^S Hbar e^-S."""
    nocc, nvir = data["nocc"], data["nvir"]
    G = expm(S) @ Hbar @ expm(-S)
    Gs, Gt = project_1h1p(data, dets, index, G)
    GsD = Gs - E_N * np.eye(nocc * nvir)
    GtD = Gt - E_N * np.eye(nocc * nvir)
    gpp = np.zeros((nvir, nocc, nvir, nocc))
    gph = np.zeros((nvir, nocc, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    gpp[a, j, b, i] = -GtD[r, c]              # includes F on diag
                    gph[b, j, i, a] = 0.5 * (GsD[r, c] - GtD[r, c])
    return GsD, GtD, gpp, gph


def spec_from(gpp, gph, Fdiag, nocc, nvir):
    """assemble singlet 1h1p from g_phph(a,j,b,i)+F and g_phhp(b,j,i,a); eigenvalues eV."""
    dim = nocc * nvir
    Gs = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    Gs[r, c] = 2.0 * gph[b, j, i, a] - gpp[a, j, b, i]
    w, vr = np.linalg.eig(Gs)
    o = np.argsort(w.real)
    return w[o].real * Ha, vr[:, o]


def part_ratio(v):
    p = np.abs(v) ** 2; s = p.sum()
    return float(1.0 / np.sum((p / s) ** 2)) if s > 0 else 0.0


def follow(vref, V):
    ov = np.abs(vref.conj() @ V) / (np.linalg.norm(vref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov))


def run(label, atom, ncore=0, active=None):
    data = (get_active_data(atom=atom, basis="sto-3g", active=active) if active is not None
            else get_active_data(atom=atom, basis="sto-3g", ncore=ncore))
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    nsip = sum(np.linalg.norm(v) for v in sIP.values())
    nsea = sum(np.linalg.norm(v) for v in sEA.values())
    print(f"    ||sIP||={nsip:.4f} ||sEA||={nsea:.4f}  (STEOM transform nontrivial if >0)")
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}
    zEA = {e: np.zeros_like(sEA[e]) for e in sEA}
    S_full = build_S(data, dets, index, sIP, sEA)
    S_ip = build_S(data, dets, index, sIP, zEA)
    S_ea = build_S(data, dets, index, zIP, sEA)
    S_0 = build_S(data, dets, index, zIP, zEA)

    _, _, gpp_f, gph_f = blocks(data, dets, index, S_full, Hbar, E_N)
    _, _, gpp_ip, gph_ip = blocks(data, dets, index, S_ip, Hbar, E_N)
    _, _, gpp_ea, gph_ea = blocks(data, dets, index, S_ea, Hbar, E_N)
    _, _, gpp_0, gph_0 = blocks(data, dets, index, S_0, Hbar, E_N)

    # g_phhp route decomposition (F-free): base + IP + EA + cross
    gph_base = gph_0
    gph_IP = gph_ip - gph_0
    gph_EA = gph_ea - gph_0
    gph_cross = gph_f - gph_ip - gph_ea + gph_0

    # g_phph route decomposition (subtract F-diagonal: use S=0 as base incl F)
    # gpp includes F on the (i==j,a==b) diagonal; routes are the S-dependent deltas.
    gpp_base = gpp_0                      # = F + bare wovov
    gpp_IP = gpp_ip - gpp_0              # u_amci route (S^IP)
    gpp_EA = gpp_ea - gpp_0              # u_akei route (S^EA)
    gpp_cross = gpp_f - gpp_ip - gpp_ea + gpp_0   # u_amei cross (S^IP x S^EA)
    def offn(X):
        s = 0.0
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i != j and a != b:
                            s += X[a, j, b, i] ** 2
        return s ** 0.5
    print(f"    ||g_phph routes|| (tot / off-diag):")
    for nm, X in [("base(F+wovov)", gpp_base), ("IP u_amci", gpp_IP),
                  ("EA u_akei", gpp_EA), ("cross u_amei", gpp_cross)]:
        print(f"       {nm:16s} {np.linalg.norm(X):8.4f} / {offn(X):8.4f}")

    # F on the diagonal lives in gpp_f (a==b,i==j); keep full g_phph everywhere (GANSU does)
    e_tar, V_tar = spec_from(gpp_f, gph_f, None, nocc, nvir)                       # plain=ORCA
    e_noX, _ = spec_from(gpp_f, gph_base + gph_IP + gph_EA, None, nocc, nvir)      # zero cross
    e_noEAX, _ = spec_from(gpp_f, gph_base + gph_IP, None, nocc, nvir)             # zero EA+cross

    print(f"\n=== {label}  nocc={nocc} nvir={nvir}")
    print(f"    ||g_phhp routes||: base={np.linalg.norm(gph_base):.4f} "
          f"IP={np.linalg.norm(gph_IP):.4f} EA={np.linalg.norm(gph_EA):.4f} "
          f"cross={np.linalg.norm(gph_cross):.4f}")
    print(f"    {'root':>4} {'target':>9} {'zeroX':>9} {'zeroEAX':>9} "
          f"{'dX':>7} {'dEAX':>7}  {'PR':>5}  mix")
    for i in range(len(e_tar)):
        vt = V_tar[:, i]; pr = part_ratio(vt)
        dX = e_noX[follow(vt, spec_from(gpp_f, gph_base+gph_IP+gph_EA, None, nocc, nvir)[1])] - e_tar[i]
        dEAX = e_noEAX[follow(vt, spec_from(gpp_f, gph_base+gph_IP, None, nocc, nvir)[1])] - e_tar[i]
        mix = "MIXED" if pr >= 1.5 else "single"
        print(f"    {i:>4} {e_tar[i]:9.3f} {e_tar[i]+dX:9.3f} {e_tar[i]+dEAX:9.3f} "
              f"{dX:+7.3f} {dEAX:+7.3f}  {pr:5.2f}  {mix}")


def main():
    run("H4 rect d=2.0 h=1.4 (config-mixed)", "H 0 0 0; H 2.0 0 0; H 0 1.4 0; H 2.0 1.4 0")
    # H6 twisted ladder: 3 occ / 3 vir, STEOM transform nontrivial + possible mixing
    run("H6 rect ladder d=2.0 h=1.4", "; ".join(
        f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(6)))
    # H8 for more room
    run("H8 rect ladder d=2.0 h=1.4", "; ".join(
        f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(8)))


if __name__ == "__main__":
    main()
