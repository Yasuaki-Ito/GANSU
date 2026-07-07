#!/usr/bin/env python3
"""Combined build_g_phph_projection: base(Wovov) + linear(Fov+V2) + quad(V2), the
route part = route_Mc - route_Ms.  Compare to the TRUE g_phph (Mc-Ms of the full
det singles block = base+route) both off-diagonal AND full (incl diagonal), to
settle the diagonal / F_eff decomposition for the C++ port.

Run:  wsl python3 script/steom_gphph_combined.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_gphph_spatial_quad as SQ
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block, build_so_integrals
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)

LSPACE = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v', 'I': 'o', 'J': 'o', 'K': 'o', 'A': 'v', 'B': 'v'}
EXT = set('iajb')


def offdiag_full_norm(T, nocc, nvir):
    off = full = 0.
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    full += T[i, a, j, b]**2
                    if i != j and a != b: off += T[i, a, j, b]**2
    return off**0.5, full**0.5


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
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True); fN = fp + Fc
    g1 = np.zeros((nso, nso))
    g1[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]; g1[np.ix_(vir, occ)] = fN[np.ix_(vir, occ)]
    arr = {'v': vp, 'g': g1, 's': sp, 'e': se}

    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]

    def olist(s): return oa if s == 0 else ob
    def vlist(s): return va if s == 0 else vb

    def spatial_term(coeff, sub, o1, o2, ket_spin):
        lhs, out = sub.split("->"); A, B = lhs.split(",")
        internal = sorted(set(lhs.replace(",", "")) - EXT)
        acc = np.zeros((nocc, nvir, nocc, nvir))
        for spins in itertools.product([0, 1], repeat=len(internal)):
            sm = dict(zip(internal, spins))
            def ixl(opstr):
                r = []
                for ch in opstr:
                    s = (0 if ch in ('i', 'a') else ket_spin) if ch in EXT else sm[ch]
                    r.append(olist(s) if LSPACE[ch] == 'o' else vlist(s))
                return r
            acc += coeff*np.einsum(sub, arr[o1][np.ix_(*ixl(A))], arr[o2][np.ix_(*ixl(B))], optimize=True)
        return acc

    lin_terms = [
        (+0.5, "Ib,jIia->iajb", 'g', 's'), (-0.5, "Jb,jiJa->iajb", 'g', 's'),
        (-0.5, "IKbi,jIKa->iajb", 'v', 's'), (-0.5, "aIbA,jIiA->iajb", 'v', 's'), (+0.5, "aIbA,jiIA->iajb", 'v', 's'),
        (-0.5, "jA,biAa->iajb", 'g', 'e'), (+0.5, "jB,biaB->iajb", 'g', 'e'),
        (+0.5, "ajAB,biAB->iajb", 'v', 'e'), (-0.5, "jIiA,bIAa->iajb", 'v', 'e'), (+0.5, "jIiA,bIaA->iajb", 'v', 'e'),
    ]
    def lin_route(ks):
        r = np.zeros((nocc, nvir, nocc, nvir))
        for c, sub, o1, o2 in lin_terms: r += spatial_term(c, sub, o1, o2, ks)
        return r

    # quadratic
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    p, q, r_, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r_, t))
    V2 = Rational(1, 4)*v*NO(Fd(p)*Fd(q)*F(t)*F(r_))
    ssb = IndexedBase('s'); seab = IndexedBase('se')
    def Sip(tg):
        M, I, J = symbols(f'M{tg} I{tg} J{tg}', below_fermi=True); B = symbols(f'B{tg}', above_fermi=True)
        return Rational(1, 2)*ssb[M, I, J, B]*NO(Fd(M)*F(I)*Fd(B)*F(J))
    def Sea(tg):
        E, A, B = symbols(f'E{tg} A{tg} B{tg}', above_fermi=True); J = symbols(f'J{tg}', below_fermi=True)
        return Rational(1, 2)*seab[E, J, A, B]*NO(Fd(A)*F(E)*Fd(B)*F(J))
    print("deriving quadratic (slow)...")
    qexpr = 0
    for SA, SB in [(Sip('1'), Sip('2')), (Sea('1'), Sea('2')), (Sip('1'), Sea('2')), (Sea('1'), Sip('2'))]:
        inr = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inr - inr*SA)
        qexpr += WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))
    ev = SQ.build_spatial_evaluator(nact, nocc, nvir, {'v': vp, 's': sp, 'se': se})

    def route(ks): return lin_route(ks) + ev(qexpr, [i, a, j, b], ks)
    route_diff = route(1) - route(0)                     # route_Mc - route_Ms

    # base Wovov (alpha-alpha singles Hbar base): g_phph base = Wovov[k,a,i,c]->[i,a,j,b] i.e.
    # my[i,a,j,b] base = <i^a_a|Hbar|j^b_a>_base.  Extract from SO g integrals directly.
    g_so, f_so = build_so_integrals(data)
    # base for the ROUTE convention: det_singles_block Ms/Mc with route only excludes base.
    # So the full g_phph = base + route.  TRUE g_phph = Mc_full - Ms_full (full singles block).
    Ms_full, Mc_full, _ = det_singles_block(data, dets, index, Hbar)
    true_gphph = Mc_full - Ms_full
    # my projection full = base(Wovov) + route_diff.  base = Ms_full,Mc_full at route=0 -> use their
    # difference base = (Mc-Ms) with S=0 == det_singles_block base part. Compute base via det on 0-route:
    base_Ms, base_Mc, _ = det_singles_block(data, dets, index, np.zeros_like(Hbar))
    # ^ zero operator gives -E_N on diagonal for Ms only; not the base. Instead base = true - route(det).
    # Proper: TRUE = base + det_route. Compare my spatial route_diff to det_route (Mc-Ms).
    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    op1g = H3.opk_matrix(dets, index, g1, 1)
    lin = S @ (op1g + V2mat) - (op1g + V2mat) @ S
    innr = S @ V2mat - V2mat @ S; qd = 0.5*(S @ innr - innr @ S)
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, lin + qd)
    det_route = Mc_d - Ms_d

    off_r, full_r = offdiag_full_norm(route_diff - det_route, nocc, nvir)
    print(f"[route Mc-Ms]  spatial vs det:  off-diag ||diff||={off_r:.3e}  full ||diff||={full_r:.3e}")
    o1, f1 = offdiag_full_norm(det_route, nocc, nvir)
    print(f"   ||det_route|| off={o1:.6f} full={f1:.6f}   (diagonal magnitude={ (f1**2-o1**2)**0.5:.6f})")


if __name__ == "__main__":
    main()
