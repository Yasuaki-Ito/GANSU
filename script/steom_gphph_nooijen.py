#!/usr/bin/env python3
"""NOOIJEN CONNECTED OBJECT derivation seed (post-続48 campaign start).

Facts (続49): ORCA STEOM is ACTIVE-SPACE-INVARIANT (H2O 11.848/13.601 identical
for 3 vs 2 active IP roots) => ORCA implements the normal-ordered/connected
object, not the plain partial-S-fragile projection.  The det {e^S}-K oracle
shows the invariance mechanism (root0 immobile under root drops) but its
2nd-order K form is 0.23 eV off ORCA at root0.

This script derives with sympy the CONNECTED similarity
    G = ({Hbar} {e^S})_C = Hbar + (Hbar S)_C + 1/2 (Hbar {S^2})_C
on the 1h1p block (leg counting: a 2-body Hbar can connect at most 2 S factors
with 4 externals => the series terminates exactly at n=2).  Connectedness
filter on the fully-contracted Wick terms:
  * every amplitude tensor (s/se) shares >=1 index with the Hbar tensor (x/v)
  * no two amplitude tensors share an index ({S^2} = normal-ordered product,
    no S-S contractions)
Evaluate at SO level on H2O FC1 and compare eigenvalues vs ORCA
(11.848 / 13.601 / 16.101 / 18.238 eV) and vs shipped connected
(11.773 / 13.522) — and test root-drop invariance.

Run:  wsl python3 script/steom_gphph_nooijen.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor, KroneckerDelta
import steom_gphph_wickeval as WE
import steom_gphph_diagsemi as DS
import steom_ip_route_derive as IPD
import steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 hf_det, so_index, occ_so, vir_so)

Ha = 27.211386245988


def make_pieces():
    x = IndexedBase('x'); ss = IndexedBase('s'); sea = IndexedBase('se')

    def O1():
        p, q = symbols('p q', cls=Dummy)
        return x[p, q] * NO(Fd(p) * F(q))

    def V2():
        p, q, r, t = symbols('p q r t', cls=Dummy)
        v = AntiSymmetricTensor('v', (p, q), (r, t))
        return Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))

    def Sip(tg):
        M, I, J = symbols(f'M{tg} I{tg} J{tg}', below_fermi=True)
        B = symbols(f'B{tg}', above_fermi=True)
        return Rational(1, 2) * ss[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))

    def Sea(tg):
        E, A, B = symbols(f'E{tg} A{tg} B{tg}', above_fermi=True)
        J = symbols(f'J{tg}', below_fermi=True)
        return Rational(1, 2) * sea[E, J, A, B] * NO(Fd(A) * F(E) * Fd(B) * F(J))
    return O1, V2, Sip, Sea


def term_tensors(term):
    """-> (amp_tensor_indexsets, op_tensor_indexset) for connectivity analysis."""
    amps = []; opidx = set()
    for f in Mul.make_args(term):
        if isinstance(f, AntiSymmetricTensor):
            opidx |= set(list(f.upper) + list(f.lower))
        elif getattr(f, 'is_Indexed', False):
            nm = str(f.base)
            if nm in ('s', 'se'):
                amps.append(set(f.indices))
            elif nm == 'x':
                opidx |= set(f.indices)
    return amps, opidx


def connected_filter(expr):
    """keep only terms where every amp touches the O tensor and no amp-amp link."""
    kept = []
    for term in Add.make_args(expr):
        if term == 0:
            continue
        amps, opidx = term_tensors(term)
        ok = all(len(a & opidx) >= 1 for a in amps)
        for i in range(len(amps)):
            for j in range(i + 1, len(amps)):
                if amps[i] & amps[j]:
                    ok = False
        if ok:
            kept.append(term)
    return Add(*kept) if kept else 0


def derive_connected():
    i, j = symbols('i j', below_fermi=True)
    a, b = symbols('a b', above_fermi=True)
    bra = Fd(i) * F(a); ket = Fd(b) * F(j)
    O1, V2, Sip, Sea = make_pieces()
    exprs = []
    for Of in (O1, V2):
        for S1f in (lambda: Sip('1'), lambda: Sea('1')):
            e = WE.contract(bra * Of() * S1f() * ket)
            exprs.append(connected_filter(e))
        for SAf, SBf in [(lambda: Sip('1'), lambda: Sip('2')),
                         (lambda: Sea('1'), lambda: Sea('2')),
                         (lambda: Sip('1'), lambda: Sea('2')),
                         (lambda: Sea('1'), lambda: Sip('2'))]:
            e = WE.contract(bra * Rational(1, 2) * Of() * SAf() * SBf() * ket)
            exprs.append(connected_filter(e))
    return (i, a, j, b), exprs


def main():
    xyz, basis, ncore = "xyz/H2O.xyz", "sto-3g", 1
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    fN = fp + Fc

    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)

    print("deriving connected (Hbar {e^S})_C exprs (slow)...")
    ext, exprs = derive_connected()
    i, a, j, b = ext
    nz = sum(1 for e in exprs if e != 0)
    print(f"[sympy] {len(exprs)} pieces, {nz} nonzero after connected filter")

    def spectrum(drop_occ=(), drop_vir=()):
        spso = np.zeros((nso,) * 4); seso = np.zeros((nso,) * 4)
        for m in occ:
            if (m % nact) not in drop_occ: spso[m] = rec_ip[m]
        for e in vir:
            if (e % nact) - nocc not in drop_vir: seso[e] = rec_ea[e]
        arrs = {'v': vp, 's': spso, 'se': seso, 'x': fN}
        V = np.zeros((nso,) * 4)
        for e_ in exprs:
            if e_ == 0:
                continue
            V += WE.eval_expr(e_, arrs, [i, a, j, b], nso, occ, vir)
        oa = [so_index(k, 0, nact) for k in range(nocc)]
        ob = [so_index(k, 1, nact) for k in range(nocc)]
        va = [so_index(k + nocc, 0, nact) for k in range(nvir)]
        vb = [so_index(k + nocc, 1, nact) for k in range(nvir)]
        Ms_r = V[np.ix_(oa, va, oa, va)]; Mc_r = V[np.ix_(oa, va, ob, vb)]
        # base = raw Hbar blocks (E_N-shifted)
        Ms_h, Mc_h = DS.det_block_raw(data, dets, index, Hbar)
        dim = nocc * nvir
        Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
        for i2 in range(nocc):
            for a2 in range(nvir):
                r = i2 * nvir + a2
                for j2 in range(nocc):
                    for b2 in range(nvir):
                        c = j2 * nvir + b2
                        een = E_N if (i2 == j2 and a2 == b2) else 0.0
                        ms = Ms_h[i2, a2, j2, b2] - een + Ms_r[i2, a2, j2, b2]
                        mc = Mc_h[i2, a2, j2, b2] + Mc_r[i2, a2, j2, b2]
                        Gs[r, c] = ms + mc
                        Gt[r, c] = ms - mc
        es = np.sort(np.linalg.eigvals(Gs).real) * Ha
        et = np.sort(np.linalg.eigvals(Gt).real) * Ha
        return es, et

    es0, et0 = spectrum()
    print("connected G (full S)  singlet:", np.round(es0, 3))
    print("                      triplet:", np.round(et0, 3))
    print("ORCA                 singlet: [11.848 13.601 16.101 18.238]")
    print("GANSU shipped conn   singlet: [11.773 13.522 16.963 16.997]")
    for dro, drv in [((0,), ()), ((0,), (1,)), ((0, 1), (1,))]:
        es, et = spectrum(dro, drv)
        print(f"drop occ={dro} vir={drv}: singlet:", np.round(es, 3))


if __name__ == "__main__":
    main()
