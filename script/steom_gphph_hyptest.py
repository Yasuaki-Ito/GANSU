#!/usr/bin/env python3
"""Direct hypothesis test (no blind fit): the g_phhp IP route is 2 SO terms
(memory pt599, direct D).  Test whether the SAME 2 terms with antisymmetrized g
reproduce BOTH the cross block Mc (=> direct only) AND the same block Ms
(=> direct-exchange).  If yes: g_phph = Mc - Ms = exchange part, done.

Term1 (SO): perm(0,2,3,1) of einsum('epqr,pqyz->yzre', sip, X)
Term2 (SO): perm(2,1,3,0) of einsum('epqr,pryz->yzqe', sip, X)
root e = ket hole; output mapped to [i,a,j,b].

Run: wsl python3 script/steom_gphph_hyptest.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip,
                                 build_S, hf_det, so_index)
from steom_so_derive import det_singles_block, build_direct, build_so_integrals


def two_terms(sip, X):
    T1 = np.transpose(np.einsum("epqr,pqyz->yzre", sip, X, optimize=True), (0, 2, 3, 1))
    T2 = np.transpose(np.einsum("epqr,pryz->yzqe", sip, X, optimize=True), (2, 1, 3, 0))
    return T1, T2


def slc(T, data, block):
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    if block == "cross":
        j = [so_index(i, 1, nact) for i in range(nocc)]
        b = [so_index(a + nocc, 1, nact) for a in range(nvir)]
    else:
        j = oa; b = va
    return T[np.ix_(oa, va, j, b)]


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
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, Mc0, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, Mc1, _ = det_singles_block(data, dets, index, Hbar + comm)
    Mc_route = Mc1 - Mc0
    Ms_route = Ms1 - Ms0
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP:
        sip[m] = SIP[m]
    g, f = build_so_integrals(data); D = build_direct(data)

    def report(X, name):
        T1, T2 = two_terms(sip, X)
        for cross_tgt, same_tgt in [(Mc_route, Ms_route)]:
            for tag, tgt, blk in [("cross(Mc)", Mc_route, "cross"), ("same(Ms)", Ms_route, "same")]:
                s = slc(T1 + T2, data, blk)
                # best single scale for the 2-term sum
                c = (s.ravel() @ tgt.ravel()) / (s.ravel() @ s.ravel() + 1e-30)
                r = np.linalg.norm(s * c - tgt) / (np.linalg.norm(tgt) + 1e-30)
                # also free 2-coeff fit
                A = np.stack([slc(T1, data, blk).ravel(), slc(T2, data, blk).ravel()], 1)
                co, *_ = np.linalg.lstsq(A, tgt.ravel(), rcond=None)
                r2 = np.linalg.norm(A @ co - tgt.ravel()) / (np.linalg.norm(tgt) + 1e-30)
                print(f"    [{name}] {tag:10s}: scale c={c:+.3f} resid={r:.3e} | "
                      f"free(c1,c2)=({co[0]:+.3f},{co[1]:+.3f}) resid={r2:.3e}")
    print(f"  ||Mc||={np.linalg.norm(Mc_route):.4f} ||Ms||={np.linalg.norm(Ms_route):.4f}")
    report(D, "D  ")
    report(g, "g  ")


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(6)), label="H6 rect ladder")
    import steom_cas_verify as V
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
