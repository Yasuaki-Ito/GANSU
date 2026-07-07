#!/usr/bin/env python3
"""Full V2-quadratic: 1/2[S,[S,V2]], S=S_ip+S_ea = sum of Sip^2, Sea^2, cross.
Evaluate all via generic evaluator, + F_c correction, compare to det
1/2[S,[S,op2(vp)]] (full S).  Completes the 2-body quadratic.

Run: wsl python3 script/steom_gphph_quad_full.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)


def sops():
    s = IndexedBase('s'); se = IndexedBase('se')
    def Sip(tag):
        M, I, J = symbols(f'M{tag} I{tag} J{tag}', below_fermi=True); B = symbols(f'B{tag}', above_fermi=True)
        return Rational(1, 2) * s[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))
    def Sea(tag):
        E, A, B = symbols(f'E{tag} A{tag} B{tag}', above_fermi=True); J = symbols(f'J{tag}', below_fermi=True)
        return Rational(1, 2) * se[E, J, A, B] * NO(Fd(A) * F(E) * Fd(B) * F(J))
    return Sip, Sea


def main():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))
    Sip, Sea = sops()
    bra = Fd(i) * F(a); ket = Fd(b) * F(j)

    def nested(SA, SB):
        inner = SB * V2 - V2 * SB
        return Rational(1, 2) * (SA * inner - inner * SA)

    print("deriving 4 quadratic pieces (slow)...")
    exprs = []
    for SA, SB, nm in [(Sip('1'), Sip('2'), "SipSip"), (Sea('1'), Sea('2'), "SeaSea"),
                       (Sip('1'), Sea('2'), "SipSea"), (Sea('1'), Sip('2'), "SeaSip")]:
        exprs.append(WE.contract(bra * nested(SA, SB) * ket))
        print(f"  {nm} done")

    # numerics
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    SIP = {m: vv for m, vv in IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data).items() if m in occ_so(data)}
    sea = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)
    SEA = {e: sea[e] for e in vir_so(data)}
    S = build_S(data, dets, index, SIP, SEA)
    occ = occ_so(data); vir = vir_so(data)
    spt = np.zeros((nso,)*4); set_ = np.zeros((nso,)*4)
    for m in SIP: spt[m] = SIP[m]
    for e in SEA: set_[e] = SEA[e]

    def qroute(Op):
        inr = S @ Op - Op @ S; c = 0.5*(S @ inr - inr @ S)
        M, _, _ = det_singles_block(data, dets, index, c); return M
    op2vp = H3.opk_matrix(dets, index, vp, 2)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    target = qroute(op2vp) - qroute(H3.opk_matrix(dets, index, Fc, 1))

    tot = np.zeros((nso,)*4)
    for e in exprs:
        tot += WE.eval_expr(e, {'v': vp, 's': spt, 'se': set_}, [i, a, j, b], nso, occ, vir)
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ts = tot[np.ix_(oa, va, oa, va)]
    print(f"full V2-quadratic: ||sympy - (det-Fc)||/||.|| = {np.linalg.norm(ts-target)/np.linalg.norm(target):.3e}"
          f"  (||target||={np.linalg.norm(target):.4f})")


if __name__ == "__main__":
    main()
