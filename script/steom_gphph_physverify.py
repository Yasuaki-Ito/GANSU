#!/usr/bin/env python3
"""FINAL C++ blueprint verification: the PHYSICAL g_phph = Mc - Ms of the full
order-2 bare route (det oracle = ground truth), reproduced by my recipe.

Ground truth (det):  Ostep = [S, op1(fp)+op2(vp)] + 1/2 [S,[S, op2(vp)]]  (bare)
  det_singles_block(Ostep) -> Ms_route (aa), Mc_route (ab);  g_phph = Mc - Ms.

Recipe under test:  my Fermi-NO linear formula (Fov ov + V2), sliced for same-spin
(ket alpha) and cross-spin (ket beta), PLUS the F_c mean-field 1-body route
(linear+quad), PLUS the quadratic V2 route.  F_c[p,q] = sum_I vp[p,I,q,I].

Run:  wsl python3 script/steom_gphph_physverify.py
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


def off_norm(T, nocc, nvir):
    s = 0.0
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
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
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]

    # ---------- GROUND TRUTH via det oracle ----------
    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})
    op1f = H3.opk_matrix(dets, index, fp, 1)
    op2 = H3.opk_matrix(dets, index, vp, 2)
    lin = S @ (op1f + op2) - (op1f + op2) @ S
    inr = S @ op2 - op2 @ S
    quad = 0.5*(S @ inr - inr @ S)
    Ms_r, Mc_r, _ = det_singles_block(data, dets, index, lin + quad)
    phys = Mc_r - Ms_r
    print(f"[ground truth] physical g_phph route  off ||.|| = {off_norm(phys, nocc, nvir):.6f}")

    # ---------- MY RECIPE ----------
    arr = {'v': vp, 'f': fp, 's': sp, 'e': se}
    lin_terms = [
        (+0.5, "Ib,jIia->iajb", 'f', 's'), (-0.5, "Jb,jiJa->iajb", 'f', 's'),
        (-0.5, "IKbi,jIKa->iajb", 'v', 's'), (-0.5, "aIbA,jIiA->iajb", 'v', 's'), (+0.5, "aIbA,jiIA->iajb", 'v', 's'),
        (-0.5, "jA,biAa->iajb", 'f', 'e'), (+0.5, "jB,biaB->iajb", 'f', 'e'),
        (+0.5, "ajAB,biAB->iajb", 'v', 'e'), (-0.5, "jIiA,bIAa->iajb", 'v', 'e'), (+0.5, "jIiA,bIaA->iajb", 'v', 'e'),
    ]
    def eval_lin(jo, jv):
        out = np.zeros((nocc, nvir, nocc, nvir))
        for c, sub, o1, o2 in lin_terms:
            T = c*np.einsum(sub, arr[o1], arr[o2], optimize=True)
            out += T[np.ix_(oa, va, jo, jv)]
        return out
    lin_route = eval_lin(ob, vb) - eval_lin(oa, va)          # Mc - Ms (linear Fov+V2)

    # F_c mean-field 1-body route (linear + quad)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    op1Fc = H3.opk_matrix(dets, index, Fc, 1)
    lFc = S @ op1Fc - op1Fc @ S
    qFc = 0.5*(S @ lFc - lFc @ S)
    Ms_fc, Mc_fc, _ = det_singles_block(data, dets, index, lFc + qFc)
    fc_route = Mc_fc - Ms_fc

    # quadratic V2 route via sympy generic evaluator -> full SO tensor, slice both blocks
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4) * v * NO(Fd(p)*Fd(q)*F(t)*F(r))
    s = IndexedBase('s'); sea = IndexedBase('se')
    def Sip(tg):
        M, I, J = symbols(f'M{tg} I{tg} J{tg}', below_fermi=True); B = symbols(f'B{tg}', above_fermi=True)
        return Rational(1, 2)*s[M, I, J, B]*NO(Fd(M)*F(I)*Fd(B)*F(J))
    def Sea(tg):
        E, A, B = symbols(f'E{tg} A{tg} B{tg}', above_fermi=True); J = symbols(f'J{tg}', below_fermi=True)
        return Rational(1, 2)*sea[E, J, A, B]*NO(Fd(A)*F(E)*Fd(B)*F(J))
    print("deriving quadratic V2 route (slow)...")
    quadV2 = np.zeros((nso,)*4)
    for SA, SB in [(Sip('1'), Sip('2')), (Sea('1'), Sea('2')), (Sip('1'), Sea('2')), (Sea('1'), Sip('2'))]:
        inner = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inner - inner*SA)
        expr = WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))
        quadV2 += WE.eval_expr(expr, {'v': vp, 's': sp, 'se': se}, [i, a, j, b], nso, occ, vir)
    quad_route = quadV2[np.ix_(oa, va, ob, vb)] - quadV2[np.ix_(oa, va, oa, va)]

    recipe = lin_route + fc_route + quad_route
    print(f"[breakdown] ||lin(Mc-Ms)||off={off_norm(lin_route,nocc,nvir):.6f}  "
          f"||Fc||off={off_norm(fc_route,nocc,nvir):.6f}  ||quad(Mc-Ms)||off={off_norm(quad_route,nocc,nvir):.6f}")
    print(f"[recipe] lin(Mc-Ms)+Fc+quad(Mc-Ms)  off ||.|| = {off_norm(recipe, nocc, nvir):.6f}")
    print(f"[CHECK]  ||recipe - physical|| off = {off_norm(recipe - phys, nocc, nvir):.3e}")


if __name__ == "__main__":
    main()
