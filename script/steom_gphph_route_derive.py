#!/usr/bin/env python3
"""Derive the plain-projection g_phph (exchange) IP/EA routes (linear in S).

Off-diagonal identity (singlet Gs = F + 2 g_phhp - g_phph, oracle GsD = Ms + Mc,
g_phhp = Mc):   g_phph(i,a,j,b) = Mc - Ms   (F-free off-diagonal).
Target the linear-in-S part of (Mc - Ms), SO-enumerate s x {D, g_antisym}, read off
the formula, then spin-integrate to spatial. Same method as the g_phhp IP route.

Run: wsl python3 script/steom_gphph_route_derive.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, itertools
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, so_index, vir_so, occ_so)
from steom_so_derive import det_singles_block, build_direct, build_so_integrals


def enumerate_fit(sip, nso, nocc, nvir, nact, target, g, D, f, label, spin="cross"):
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    # cross = alpha-bra/beta-ket [oa,va,ob,vb]; same = alpha-bra/alpha-ket [oa,va,oa,va]
    if spin == "same":
        ixj, ixb = oa, va
    else:
        ixj, ixb = ob, vb
    def slc(T): return T[np.ix_(oa, va, ixj, ixb)]
    cand = {}
    sax = ['p', 'q', 'r']
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            c = slc(Tp)
            if np.linalg.norm(c) < 1e-9:
                continue
            cand[f"{tag}:perm{perm}"] = c.ravel()
    for Xname, X in [("D", D), ("g", g)]:
        for sc in itertools.combinations(range(3), 2):
            sf = [k for k in range(3) if k not in sc][0]
            for xc in itertools.permutations(range(4), 2):
                lab_s = list(sax); lab_x = list("wxyz")
                for si, xi in zip(sc, xc):
                    lab_x[xi] = lab_s[si]
                xf = [k for k in range(4) if k not in xc]
                out = [lab_x[xf[0]], lab_x[xf[1]], lab_s[sf], 'e']
                es = f"e{''.join(lab_s)},{''.join(lab_x)}->{''.join(out)}"
                try:
                    T = np.einsum(es, sip, X, optimize=True)
                except Exception:
                    continue
                add(T, f"{Xname}:{es}")
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 9))
        uniq.setdefault(key, n)
    names = list(uniq.values())
    tv = target.ravel(); tn = np.linalg.norm(tv)
    print(f"  [{label}] SO enum: {len(cand)}->{len(names)} uniq  ||target||={tn:.4f}")
    chosen = []; res = tv.copy()
    for step in range(6):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n])) * cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac @ co
        print(f"    step {step + 1}: rel-resid={np.linalg.norm(res) / tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"        {cc:+.4f}  {m}")
        if np.linalg.norm(res) / tn < 3e-3:
            break


def run(atom=None, xyz=None, active=None, ncore=0, label=""):
    print(f"\n===== {label} =====")
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
    elif active:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso))
           for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    Hc = Hbar + (S_ip @ Hbar - Hbar @ S_ip)
    Ms0, Mc0, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, Mc1, _ = det_singles_block(data, dets, index, Hc)
    # g_phph(i,a,j,b) = Mc - Ms  (off-diag, F-free);  linear-IP route:
    tgt_gphph = (Mc1 - Ms1) - (Mc0 - Ms0)
    tgt_ms = Ms1 - Ms0                          # same-spin block IP route (for reference)
    g, f = build_so_integrals(data); D = build_direct(data)
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP:
        sip[m] = SIP[m]
    print(f"  ||g_phph IP route (Mc-Ms)||={np.linalg.norm(tgt_gphph):.4f}  ||Ms IP route||={np.linalg.norm(tgt_ms):.4f}")
    enumerate_fit(sip, nso, nocc, nvir, nact, tgt_ms, g, D, f, "Ms (same-spin)", spin="same")


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(6)), label="H6 rect ladder")
    import steom_cas_verify as V
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
