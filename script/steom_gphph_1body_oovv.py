#!/usr/bin/env python3
"""Derive the SO terms of the 1-body route [S, op1(g)] for the oo and vv blocks
(the Fc mean-field route), between singles.  Prints the fully-contracted sympy
expression (so terms can be transcribed to the spatial adapter / C++) and verifies
the generic-evaluator SO result vs the det oracle (op1(g_oo)+op1(g_vv)).

Run:  wsl python3 script/steom_gphph_1body_oovv.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, IndexedBase
from sympy.physics.secondquant import F, Fd, NO
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)


def main():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    g = IndexedBase('g'); s = IndexedBase('s'); sea = IndexedBase('se')
    # S_ip and S_ea operators
    M, I, J = symbols('M I J', below_fermi=True); B = symbols('B', above_fermi=True)
    Sip = Rational(1, 2)*s[M, I, J, B]*NO(Fd(M)*F(I)*Fd(B)*F(J))
    E, A, Bp = symbols('E A Bp', above_fermi=True); Jp = symbols('Jp', below_fermi=True)
    Sea = Rational(1, 2)*sea[E, Jp, A, Bp]*NO(Fd(A)*F(E)*Fd(Bp)*F(Jp))
    # 1-body operators for oo and vv blocks
    k, l = symbols('k l', below_fermi=True); c, d = symbols('c d', above_fermi=True)
    op_oo = g[k, l]*NO(Fd(k)*F(l))
    op_vv = g[c, d]*NO(Fd(c)*F(d))

    for name, op in [("oo", op_oo), ("vv", op_vv)]:
        for sname, Sop in [("S_ip", Sip), ("S_ea", Sea)]:
            expr = WE.contract(Fd(i)*F(a)*(Sop*op - op*Sop)*Fd(b)*F(j))
            print(f"--- 1-body {name} route, {sname} ---")
            print("   ", expr)

    # numerical verification of the whole oo+vv route vs det
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    zz = np.load("/tmp/hbar_mbody.npz"); vp = zz["vp"]
    sp = np.zeros((nso,)*4); se = np.zeros((nso,)*4)
    for m in occ_so(data): sp[m] = IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data)[m]
    for e in vir_so(data): se[e] = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)[e]
    occ = occ_so(data); vir = vir_so(data)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    g_oovv = np.zeros((nso, nso))
    g_oovv[np.ix_(occ, occ)] = Fc[np.ix_(occ, occ)]; g_oovv[np.ix_(vir, vir)] = Fc[np.ix_(vir, vir)]
    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]
    op1 = H3.opk_matrix(dets, index, g_oovv, 1)
    cmt = S @ op1 - op1 @ S
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, cmt); det_r = Mc_d - Ms_d

    # generic-eval SO route (sum of oo+vv, S_ip+S_ea) using g_oovv as the 'g' array
    tot = np.zeros((nso,)*4)
    for op in [op_oo, op_vv]:
        for Sop in [Sip, Sea]:
            expr = WE.contract(Fd(i)*F(a)*(Sop*op - op*Sop)*Fd(b)*F(j))
            if expr == 0:
                continue
            tot += WE.eval_expr(expr, {'v': vp, 'g': g_oovv, 's': sp, 'se': se}, [i, a, j, b], nso, occ, vir)
    ev_r = tot[np.ix_(oa, va, ob, vb)] - tot[np.ix_(oa, va, oa, va)]

    def offn(T):
        ss = 0.
        for x in range(nocc):
            for y in range(nvir):
                for u in range(nocc):
                    for w in range(nvir):
                        if x != u and y != w: ss += T[x, y, u, w]**2
        return ss**0.5
    print(f"\n[oo+vv Fc route] det off={offn(det_r):.6f}  eval off={offn(ev_r):.6f}  ||diff||={offn(ev_r-det_r):.3e}")


if __name__ == "__main__":
    main()
