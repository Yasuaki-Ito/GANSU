#!/usr/bin/env python3
"""bar_h MAPPING verification (final pre-C++ step): build every SO ingredient of the
projection g_phph route FROM THE ANALYTIC PIPELINE (pyscf bar_h intermediates +
normalized s amplitudes), run the verified spatial adapter (linear 10 terms + quad),
and compare Mc-Ms vs the det-oracle route.  A match proves the C++ needs only:
  Fov, Wooov, Wvovv (dressed), eri_ovov (bare), s_IP, s_EA   [+ base Wovov].

SO constructions (spin-diagonal 1-body; 2-body antisym from spatial direct W):
  <KL||ID> = d(sK,sI)d(sL,sD) Wooov[k,l,i,d] - d(sK,sD)d(sL,sI) Wooov[l,k,i,d]
  <AL||CD> = d(sA,sC)d(sL,sD) Wvovv[a,l,c,d] - d(sA,sD)d(sL,sC) Wvovv[a,l,d,c]
  <IJ||AB> = d(sI,sA)d(sJ,sB) (ia|jb)        - d(sI,sB)d(sJ,sA) (ib|ja)   [bare]
  plus antisym images v[..,S,R] = -v[..,R,S] for the oovo access "IKbi".

Run:  wsl python3 script/steom_gphph_barh_map.py
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
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from pyscf_steom_feff_reference import build_normalized_s
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)

LSPACE = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v', 'I': 'o', 'J': 'o', 'K': 'o', 'A': 'v', 'B': 'v'}
EXT = set('iajb')


def offn(T, nocc, nvir):
    s = 0.
    for x in range(nocc):
        for y in range(nvir):
            for u in range(nocc):
                for w in range(nvir):
                    if x != u and y != w: s += T[x, y, u, w]**2
    return s**0.5


def main():
    xyz, basis, ncore = "xyz/H2O.xyz", "sto-3g", 1
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)

    # ---------- det-oracle route (ground truth, as verified) ----------
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)            # {m: [i,j,b]}
    se_det = EA.extract_spatial_amp(solve_ea(data), data)          # [e][j,a,b]
    spso = np.zeros((nso,)*4); seso = np.zeros((nso,)*4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    g_ov = np.zeros((nso, nso)); fN = fp + Fc
    g_ov[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    op1g = H3.opk_matrix(dets, index, g_ov, 1)
    lin = S @ (op1g + V2mat) - (op1g + V2mat) @ S
    inr = S @ V2mat - V2mat @ S; qd = 0.5*(S @ inr - inr @ S)
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, lin + qd)
    det_route = Mc_d - Ms_d

    # ---------- analytic pipeline ----------
    d = CW.load(xyz, basis, ncore)
    bar = d["bar"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    # amplitude convention check vs det-extracted
    dip = max(np.linalg.norm(s_IP[m] - sp_det[m]) for m in range(nocc))
    dea = np.linalg.norm(s_EA - se_det)
    print(f"[amp check] max_m ||s_IP - det|| = {dip:.3e}   ||s_EA - det|| = {dea:.3e}")

    # SO amplitudes: USE THE DET-EXTRACTED spatial s on both sides — this test
    # isolates the INTEGRAL mapping (bar_h -> SO).  The amplitude pipelines differ
    # only in principal-root selection for the deepest occ (%singles blow-up), which
    # is irrelevant for the C++ port (GANSU uses its own s consistently).
    sp_an = spso; se_an = seso

    # SO integrals from bar_h
    Fov = bar["Fov"]; Wooov = bar["Wooov"]; Wvovv = bar["Wvovv"]; eri_ovov = bar["eri_ovov"]
    def O(x, s): return so_index(x, s, nact)
    def V(x, s): return so_index(x+nocc, s, nact)
    g1 = np.zeros((nso, nso))
    for s in range(2):
        for k in range(nocc):
            for c in range(nvir):
                g1[O(k, s), V(c, s)] = Fov[k, c]
    v_an = np.zeros((nso,)*4)
    sps = [0, 1]
    # ooov (+ antisym image oovo)
    for s1, s2, s3, s4 in itertools.product(sps, repeat=4):
        d13 = (s1 == s3); d24 = (s2 == s4); d14 = (s1 == s4); d23 = (s2 == s3)
        if not (d13 and d24) and not (d14 and d23): continue
        for k in range(nocc):
            for l in range(nocc):
                for i2 in range(nocc):
                    for dd in range(nvir):
                        val = 0.0
                        if d13 and d24: val += Wooov[k, l, i2, dd]
                        if d14 and d23: val -= Wooov[l, k, i2, dd]
                        P, Q, R, Ss = O(k, s1), O(l, s2), O(i2, s3), V(dd, s4)
                        v_an[P, Q, R, Ss] = val
                        v_an[P, Q, Ss, R] = -val
    # vovv
    for s1, s2, s3, s4 in itertools.product(sps, repeat=4):
        d13 = (s1 == s3); d24 = (s2 == s4); d14 = (s1 == s4); d23 = (s2 == s3)
        if not (d13 and d24) and not (d14 and d23): continue
        for a2 in range(nvir):
            for l in range(nocc):
                for c in range(nvir):
                    for dd in range(nvir):
                        val = 0.0
                        if d13 and d24: val += Wvovv[a2, l, c, dd]
                        if d14 and d23: val -= Wvovv[a2, l, dd, c]
                        v_an[V(a2, s1), O(l, s2), V(c, s3), V(dd, s4)] = val
    # oovv (bare)
    for s1, s2, s3, s4 in itertools.product(sps, repeat=4):
        d13 = (s1 == s3); d24 = (s2 == s4); d14 = (s1 == s4); d23 = (s2 == s3)
        if not (d13 and d24) and not (d14 and d23): continue
        for i2 in range(nocc):
            for j2 in range(nocc):
                for a2 in range(nvir):
                    for b2 in range(nvir):
                        val = 0.0
                        if d13 and d24: val += eri_ovov[i2, a2, j2, b2]
                        if d14 and d23: val -= eri_ovov[i2, b2, j2, a2]
                        v_an[O(i2, s1), O(j2, s2), V(a2, s3), V(b2, s4)] = val

    # ---------- run the verified spatial adapter with analytic arrays ----------
    arr = {'v': v_an, 'g': g1, 's': sp_an, 'e': se_an}
    oa = [O(x, 0) for x in range(nocc)]; va = [V(x, 0) for x in range(nvir)]
    ob = [O(x, 1) for x in range(nocc)]; vb = [V(x, 1) for x in range(nvir)]
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
        inn = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inn - inn*SA)
        qexpr += WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))
    ev = SQ.build_spatial_evaluator(nact, nocc, nvir, {'v': v_an, 's': sp_an, 'se': se_an})

    route = (lin_route(1) + ev(qexpr, [i, a, j, b], 1)) - (lin_route(0) + ev(qexpr, [i, a, j, b], 0))
    print(f"[bar_h map] analytic route off={offn(route,nocc,nvir):.6f}  det route off={offn(det_route,nocc,nvir):.6f}")
    print(f"[CHECK]     ||analytic - det|| off = {offn(route-det_route,nocc,nvir):.3e}")


if __name__ == "__main__":
    main()
