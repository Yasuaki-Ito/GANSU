#!/usr/bin/env python3
"""Spin-adapt the remaining LINEAR routes (Fov 1-body for S_ip & S_ea; V2 for S_ea)
to spatial closed-shell, verify machine-exact vs SO alpha-block. Same Kramers method
as the V2 S_ip route (verified 3.3e-17).

Run: wsl python3 script/steom_gphph_spinadapt2.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sip_sp = IPD.extract_sip(solve_ip(data, E_N), data); SIP = IPD.build_sip_recon(sip_sp, data)
    sea = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)
    sp = np.zeros((nso,)*4); se = np.zeros((nso,)*4)
    for m in occ_so(data): sp[m] = SIP[m]
    for e in vir_so(data): se[e] = sea[e]
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]
    I4 = np.eye(nso)
    rx = np.stack([sip_sp[m] for m in range(nocc)], 0)   # s_ip[m][i,j,b]
    saaa = np.einsum("mjib->mijb", rx) - rx; sbab = np.einsum("mjib->mijb", rx); sabb = -rx

    # ===== Fov S_ip route =====
    Fso = (0.5*np.einsum("Ib,jIia->iajb", fp, sp, optimize=True)
           - 0.5*np.einsum("Jb,jiJa->iajb", fp, sp, optimize=True)
           - 0.5*np.einsum("IB,jIiB,ab->iajb", fp, sp, I4, optimize=True)
           + 0.5*np.einsum("JB,jiJB,ab->iajb", fp, sp, I4, optimize=True))
    tgt = Fso[np.ix_(oa, va, oa, va)]
    fov = fp[np.ix_(oa, va)]                              # f alpha ov block (spin-diagonal)
    # f spin-diagonal -> internal I alpha only. s all-alpha = saaa.
    Fsp = (0.5*np.einsum("Ib,jIia->iajb", fov, saaa, optimize=True)
           - 0.5*np.einsum("Jb,jiJa->iajb", fov, saaa, optimize=True))
    # delta terms (a=b): -1/2 f[I,B]saaa[j,I,i,B] +1/2 f[J,B]saaa[j,i,J,B]
    dl = (-0.5*np.einsum("IB,jIiB->ij", fov, saaa, optimize=True) + 0.5*np.einsum("JB,jiJB->ij", fov, saaa, optimize=True))
    Fd = np.zeros((nocc, nvir, nocc, nvir))
    for aa in range(nvir): Fd[:, aa, :, aa] = dl
    print(f"Fov S_ip route: full resid = {np.linalg.norm(Fsp+Fd-tgt)/np.linalg.norm(tgt):.3e}")

    # ===== V2 S_ea route (root e=b; se[e][J,A,B]) =====
    # SO: +1/2 v[a,j,A,B]se[b,i,A,B] -1/2 v[j,I,i,A]se[b,I,A,a] +1/2 v[j,I,i,A]se[b,I,a,A] -1/2 dij v[a,I,A,B]se[b,I,A,B]
    Vso = (0.5*np.einsum("ajAB,biAB->iajb", vp, se, optimize=True)
           - 0.5*np.einsum("jIiA,bIAa->iajb", vp, se, optimize=True)
           + 0.5*np.einsum("jIiA,bIaA->iajb", vp, se, optimize=True)
           - 0.5*np.einsum("aIAB,bIAB,ij->iajb", vp, se, I4, optimize=True))
    tgt2 = Vso[np.ix_(oa, va, oa, va)]
    # se spin blocks -> spatial ry (analogous to rx). extract ry and its reductions.
    ry = np.zeros((nvir, nocc, nvir, nvir))              # se[e][j,a,b] spatial
    for e in range(nvir):
        ry[e] = se[so_index(e+nocc, 0, nact)][np.ix_(oa, va, va)]
    print(f"  (se spatial ry norm {np.linalg.norm(ry):.3f}, ||tgt2||={np.linalg.norm(tgt2):.4f})")
    # numeric spin-decomposition check of each V2-Sea term: enumerate internal spins
    # (report only that SO alpha-block is captured by same aa/ab pattern; detailed spatial
    #  reduction of se mirrors s_ip -- done next iteration.)


if __name__ == "__main__":
    main()
