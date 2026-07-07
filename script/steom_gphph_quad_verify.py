#!/usr/bin/env python3
"""Verify quadratic 1/2[S_ip,[S_ip,V2]] machine-exact via F_c correction:
det 1/2[Sip,[Sip,op2(vp)_bare]] = sympy 1/2[Sip,[Sip,V2_FermiNO]] + 1/2[Sip,[Sip,op1(F_c)]].
Uses the generic evaluator. Proves quadratic works -> rest is mechanical.

Run: wsl python3 script/steom_gphph_quad_verify.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_ip_route_derive as IPD, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    # --- sympy: 1/2[Sip,[Sip,V2]] ---
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    s = IndexedBase('s')
    M = symbols('M', below_fermi=True); I = symbols('I', below_fermi=True)
    Js = symbols('J', below_fermi=True); B = symbols('B', above_fermi=True)
    Sip = Rational(1, 2) * s[M, I, Js, B] * NO(Fd(M) * F(I) * Fd(B) * F(Js))
    M2 = symbols('M2', below_fermi=True); I2 = symbols('I2', below_fermi=True)
    J2 = symbols('J2', below_fermi=True); B2 = symbols('B2', above_fermi=True)
    Sip2 = Rational(1, 2) * s[M2, I2, J2, B2] * NO(Fd(M2) * F(I2) * Fd(B2) * F(J2))
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))
    inner = Sip2 * V2 - V2 * Sip2
    quad = Rational(1, 2) * (Sip * inner - inner * Sip)
    expr = WE.contract(Fd(i) * F(a) * quad * Fd(b) * F(j))

    # --- numerics ---
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    SIP = {m: vv for m, vv in IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data).items() if m in occ_so(data)}
    zEA = {so_index(x+nocc, sp2, nact): np.zeros((nso,)*3) for sp2 in range(2) for x in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    occ = occ_so(data); vir = vir_so(data)
    spt = np.zeros((nso,)*4)
    for m in SIP: spt[m] = SIP[m]

    def quad_route(Op):
        inr = S_ip @ Op - Op @ S_ip
        c = 0.5 * (S_ip @ inr - inr @ S_ip)
        Ms, _, _ = det_singles_block(data, dets, index, c); return Ms
    op2vp = H3.opk_matrix(dets, index, vp, 2)
    Ms_q2 = quad_route(op2vp)                    # det 1/2[Sip,[Sip,op2(vp)]]
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    Ms_qFc = quad_route(H3.opk_matrix(dets, index, Fc, 1))   # 1/2[Sip,[Sip,op1(Fc)]]
    target = Ms_q2 - Ms_qFc                       # should = sympy V2 quadratic

    V = WE.eval_expr(expr, {'v': vp, 's': spt}, [i, a, j, b], nso, occ, vir)
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    Vs = V[np.ix_(oa, va, oa, va)]
    print(f"quadratic 1/2[Sip,[Sip,V2]]: ||sympy - (det-Fc)||/||.|| = {np.linalg.norm(Vs-target)/np.linalg.norm(target):.3e}")
    print(f"  (||target||={np.linalg.norm(target):.4f} ||sympy||={np.linalg.norm(Vs):.4f})")


if __name__ == "__main__":
    main()
