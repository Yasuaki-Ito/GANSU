#!/usr/bin/env python3
"""Verify the sympy S_ea route (2-body) machine-exact via F_c correction:
route_ea(W2_FermiNO) = Ms_ea2(bare) - route_ea(op1(F_c)),  F_c[p,q]=Σ_I vp[p,I,q,I].
sympy 2-body S_ea formula (root=b, se=s_ea[b][J,A,B]):
  +1/2 v[(a,j),(A0,A1)] se[b,i,A0,A1]  -1/2 v[(j,I0),(i,A0)] se[b,I0,A0,a]
  +1/2 v[(j,I0),(i,A0)] se[b,I0,a,A0]  -1/2 dij v[(a,I0),(A0,A1)] se[b,I0,A0,A1]
Run: wsl python3 script/steom_gphph_sea_verify.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_gphph_hbar3 as H3
import steom_ea_spinadapt as EA
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ea,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    sEA = solve_ea(data)
    s_sp = EA.extract_spatial_amp(sEA, data)
    sea_rec = EA.build_sea_recon(s_sp, data)              # SO s_ea per active vir
    SEA = {e: sea_rec[e] for e in vir_so(data)}
    zIP = {m: np.zeros((nso,)*3) for m in occ_so(data)}
    S_ea = build_S(data, dets, index, zIP, SEA)
    occ = occ_so(data); vir = vir_so(data)
    # se SO tensor: se[E,J,A,B] with E active-vir root
    se = np.zeros((nso,)*4)
    for e in SEA:
        se[e] = SEA[e]
    dab = np.eye(nso); dij = np.eye(nso)

    def route1(fmat):
        Op = H3.opk_matrix(dets, index, fmat, 1)
        c = S_ea @ Op - Op @ S_ea; Ms, _, _ = det_singles_block(data, dets, index, c); return Ms

    op2vp = H3.opk_matrix(dets, index, vp, 2)
    Ms2, _, _ = det_singles_block(data, dets, index, S_ea @ op2vp - op2vp @ S_ea)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    w2route = Ms2 - route1(Fc)

    # sympy 2-body S_ea formula (root b): se[b, ...]
    # +1/2 v[a,j,A0,A1] se[b,i,A0,A1] -1/2 v[j,I0,i,A0] se[b,I0,A0,a]
    # +1/2 v[j,I0,i,A0] se[b,I0,a,A0] -1/2 dij v[a,I0,A0,A1] se[b,I0,A0,A1]
    V = ( 0.5*np.einsum("ajAB,biAB->iajb", vp, se, optimize=True)
         -0.5*np.einsum("jIiA,bIAa->iajb", vp, se, optimize=True)
         +0.5*np.einsum("jIiA,bIaA->iajb", vp, se, optimize=True)
         -0.5*np.einsum("aIAB,bIAB,ij->iajb", vp, se, dij, optimize=True))
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    Vs = V[np.ix_(oa, va, oa, va)]
    d = np.linalg.norm(w2route - Vs)/np.linalg.norm(Vs)
    c = np.vdot(w2route, Vs)/(np.vdot(w2route, w2route)+1e-30)
    print(f"S_ea 2-body route: ||Ms_ea2 - route(op1(Fc)) - V_sympy||/||V|| = {d:.3e}  best-scale {1/c if abs(c)>1e-9 else 0:.4f}")


if __name__ == "__main__":
    main()
