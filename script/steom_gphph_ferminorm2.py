#!/usr/bin/env python3
"""Cleaner (cancellation-free) formulation check: physical g_phph route =
[S, op1(fN)+op2(vp)] + 1/2[S,[S,op2(vp)]] with fN = Fermi Fock = fp + F_c.
Confirms the F_c is absorbed into the Fermi Fock (bar_h Loo/Fov/Lvv), so the
spatial recipe needs NO explicit large-cancellation F_c term: just the Fermi
Fock 1-body route (ov=Fov, oo=Loo, vv=Lvv) + the antisym 2-body route + quad.

Decompose Fock route by block (ov / oo / vv) to see which the singles route needs.

Run:  wsl python3 script/steom_gphph_ferminorm2.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)


def offn(T, no, nv):
    s = 0.0
    for i in range(no):
        for a in range(nv):
            for j in range(no):
                for b in range(nv):
                    if i != j and a != b:
                        s += T[i, a, j, b]**2
    return s**0.5


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sp = np.zeros((nso,)*4); se = np.zeros((nso,)*4)
    for m in occ_so(data): sp[m] = IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data)[m]
    for e in vir_so(data): se[e] = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)[e]
    occ = occ_so(data); vir = vir_so(data)
    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})

    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    fN = fp + Fc                                     # Fermi Fock

    def route_of(mat1b=None, mat2b=None, quad=False):
        O = np.zeros_like(S)
        if mat1b is not None: O = O + H3.opk_matrix(dets, index, mat1b, 1)
        if mat2b is not None: O = O + H3.opk_matrix(dets, index, mat2b, 2)
        c = S @ O - O @ S
        if quad: c = 0.5*(S @ c - c @ S)
        Ms, Mc, _ = det_singles_block(data, dets, index, c)
        return Mc - Ms

    # physical (bare) ground truth
    op2 = H3.opk_matrix(dets, index, vp, 2); op1f = H3.opk_matrix(dets, index, fp, 1)
    lin = S @ (op1f + op2) - (op1f + op2) @ S
    inr = S @ op2 - op2 @ S; quad = 0.5*(S @ inr - inr @ S)
    Msr, Mcr, _ = det_singles_block(data, dets, index, lin + quad); phys = Mcr - Msr

    # Fermi-Fock formulation:  fN 1-body route + vp 2-body route + quad
    fock_route = route_of(mat1b=fN)                  # [S, op1(fN)] Mc-Ms
    v2_route = route_of(mat2b=vp)                    # [S, op2(vp)] Mc-Ms
    quad_route = route_of(mat2b=vp, quad=True)
    clean = fock_route + v2_route + quad_route
    print(f"physical (bare)        off = {offn(phys,nocc,nvir):.6f}")
    print(f"fermi-fock formulation off = {offn(clean,nocc,nvir):.6f}   "
          f"||clean-phys|| = {offn(clean-phys,nocc,nvir):.3e}")

    # block-decompose the BARE fp Fock route: which fp block contributes to singles?
    print("-- bare fp block routes --")
    for blk, sl in [("ov", (occ, vir)), ("oo", (occ, occ)), ("vv", (vir, vir)), ("vo", (vir, occ))]:
        m = np.zeros_like(fp); m[np.ix_(*sl)] = fp[np.ix_(*sl)]
        print(f"   fp[{blk}] route off = {offn(route_of(mat1b=m),nocc,nvir):.6f}")
    # magnitudes
    def blkn(M, sl): return np.linalg.norm(M[np.ix_(*sl)])
    print(f"-- ||fp_ov||={blkn(fp,(occ,vir)):.4f}  ||Fc_ov||={blkn(Fc,(occ,vir)):.4f}  "
          f"||fN_ov||={blkn(fN,(occ,vir)):.4f}  ||Fc_oo||={blkn(Fc,(occ,occ)):.4f}  ||Fc_vv||={blkn(Fc,(vir,vir)):.4f}")

    # GROUPING TEST: physical =? Fov(fN_ov) + Fc(oo+vv) + V2 + quad  (bar_h friendly)
    V2 = op2 - H3.opk_matrix(dets, index, Fc, 1)     # Fermi-NO 2-body operator = op2(vp) - op1(Fc)
    m_ov = np.zeros_like(fN); m_ov[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]; m_ov[np.ix_(vir, occ)] = fN[np.ix_(vir, occ)]
    m_oovv = np.zeros_like(Fc)
    m_oovv[np.ix_(occ, occ)] = Fc[np.ix_(occ, occ)]; m_oovv[np.ix_(vir, vir)] = Fc[np.ix_(vir, vir)]
    r_fov = route_of(mat1b=m_ov)
    r_fcoovv = route_of(mat1b=m_oovv)
    # V2 route via a matrix already ordered: build [S,V2] Mc-Ms
    cV = S @ V2 - V2 @ S; MsV, McV, _ = det_singles_block(data, dets, index, cV); r_v2 = McV - MsV
    cVq = 0.5*(S @ cV - cV @ S); MsVq, McVq, _ = det_singles_block(data, dets, index, cVq); r_v2q = McVq - MsVq
    # NOTE quad must use bare op2 (physical) not V2; test both
    grp = r_fov + r_fcoovv + r_v2 + quad_route
    print(f"GROUPING [Fov(fN_ov)+Fc(oo,vv)+V2+quad(bare)] off = {offn(grp,nocc,nvir):.6f}  "
          f"||grp-phys||={offn(grp-phys,nocc,nvir):.3e}")


if __name__ == "__main__":
    main()
