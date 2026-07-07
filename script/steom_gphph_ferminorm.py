#!/usr/bin/env python3
"""Validate the sympy Fermi-NO 2-body route formula V by removing the F_eff
contraction from the bare route:  op2(vp)_bare = W2_FermiNO + op1(F_c) + const,
so route(W2_FermiNO) = Ms_2(bare) - route(op1(F_c)).  Test F_c candidates
(Fermi mean-field contraction of vp) so that Ms_2 - route(op1(F_c)) == V (sympy).

Run: wsl python3 script/steom_gphph_ferminorm.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    SIP = {m: v for m, v in IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data).items() if m in occ_so(data)}
    zEA = {so_index(a+nocc, s, nact): np.zeros((nso,)*3) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    occ = occ_so(data); vir = vir_so(data)
    sp = np.zeros((nso,)*4)
    for m in SIP: sp[m] = SIP[m]
    dab = np.eye(nso)

    def route1(fmat):
        Op = H3.opk_matrix(dets, index, fmat, 1)
        c = S_ip @ Op - Op @ S_ip
        Ms, _, _ = det_singles_block(data, dets, index, c); return Ms

    op2vp = H3.opk_matrix(dets, index, vp, 2)
    c2 = S_ip @ op2vp - op2vp @ S_ip
    Ms2, _, _ = det_singles_block(data, dets, index, c2)   # bare 2-body route

    # sympy Fermi-NO 2-body route formula V
    V = (-0.5*np.einsum("IKbi,jIKa->iajb", vp, sp, optimize=True)
         -0.5*np.einsum("aIbA,jIiA->iajb", vp, sp, optimize=True)
         +0.5*np.einsum("aIbA,jiIA->iajb", vp, sp, optimize=True)
         -0.5*np.einsum("IKiA,jIKA,ab->iajb", vp, sp, dab, optimize=True))
    oa = [so_index(i, 0, nact) for i in range(nocc)]; va = [so_index(a+nocc, 0, nact) for a in range(nvir)]
    Vs = V[np.ix_(oa, va, oa, va)]

    # F_c candidate: Fermi mean-field contraction of op2(vp) over occupied
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    # standard: F_c[p,q] = Σ_{I occ} vp[p,I,q,I]  (=<pI||qI> summed)
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    for lbl, fc, sgn in [("+Σ vp[p,I,q,I]", Fc, 1.0), ("-Σ vp[p,I,q,I]", Fc, -1.0),
                         ("+Σ vp[I,p,I,q]", np.einsum("IpIq,I->pq", vp, occ_mask, optimize=True), 1.0)]:
        r1 = route1(sgn*fc)
        w2route = Ms2 - r1
        d = np.linalg.norm(w2route - Vs)/np.linalg.norm(Vs)
        c = np.vdot(w2route, Vs)/(np.vdot(w2route, w2route)+1e-30)
        print(f"  F_c={lbl}: ||Ms2 - route(op1(Fc)) - V||/||V|| = {d:.3e}  (best-scale {1/c if abs(c)>1e-9 else 0:.3f})")
    # also try 0.5 factor
    for f in (0.5, 1.0, 2.0, -0.5, -1.0):
        r1 = route1(f*Fc); d = np.linalg.norm((Ms2-r1)-Vs)/np.linalg.norm(Vs)
        print(f"  factor {f:+.1f} on Fc: resid={d:.3e}")


if __name__ == "__main__":
    main()
