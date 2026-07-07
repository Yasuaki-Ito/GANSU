#!/usr/bin/env python3
"""Explicit-loop evaluation of the sympy general-Dummy 2-body formula (no einsum
ambiguity), vs exact Ms_2.  vp=<pq||rs> confirmed. Formula (I,K occ ; A vir):
 Ms2[i,a,j,b] = -1/2 dab Σ vp[I,K,i,A] s[j,I,K,A]
                -1/2 Σ vp[I,K,b,i] s[j,I,K,a]
                -1/2 Σ vp[a,I,b,A] s[j,I,i,A]
                +1/2 Σ vp[a,I,b,A] s[j,i,I,A]
Run: wsl python3 script/steom_gphph_explicit.py
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
    c2 = S_ip @ H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, vp, 2) @ S_ip
    Ms2, _, _ = det_singles_block(data, dets, index, c2)   # compact (nocc,nvir,nocc,nvir)
    sp = np.zeros((nso,)*4)
    for m in SIP: sp[m] = SIP[m]
    occ = occ_so(data); vir = vir_so(data)
    oaf = lambda x: so_index(x, 0, nact)          # alpha occ SO
    vaf = lambda x: so_index(x+nocc, 0, nact)     # alpha vir SO

    form = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for jj in range(nocc):
                for bb in range(nvir):
                    iS, aS, jS, bS = oaf(i), vaf(a), oaf(jj), vaf(bb)
                    val = 0.0
                    for I in occ:
                        for K in occ:
                            for A in vir:
                                # term1 dab
                                if aS == bS:
                                    val += -0.5 * vp[I, K, iS, A] * sp[jS, I, K, A]
                                # term3 -1/2 vp[a,I,b,A] s[j,I,i,A]  (K unused -> only once)
                            # terms without A-only: do below
                    for I in occ:
                        for K in occ:
                            val += -0.5 * vp[I, K, bS, iS] * sp[jS, I, K, aS]   # term2
                        for A in vir:
                            val += -0.5 * vp[aS, I, bS, A] * sp[jS, I, iS, A]    # term3
                            val += +0.5 * vp[aS, I, bS, A] * sp[jS, iS, I, A]    # term4
                    form[i, a, jj, bb] = val
    d = np.linalg.norm(form - Ms2) / np.linalg.norm(Ms2)
    print(f"explicit-loop formula vs Ms_2: resid = {d:.3e}  (||Ms2||={np.linalg.norm(Ms2):.4f} ||form||={np.linalg.norm(form):.4f})")
    c = np.vdot(form, Ms2) / np.vdot(form, form)
    print(f"  best-scale Ms2={1/c:.4f}*form post-resid={np.linalg.norm(c*form-Ms2)/np.linalg.norm(Ms2):.3e}")


if __name__ == "__main__":
    main()
