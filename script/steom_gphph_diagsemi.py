#!/usr/bin/env python3
"""DIAG/SEMI-DIAG projection route derivation (step 2 of the franken-G fix).

The shipped C++ projection fix replaces only the off-diag (i!=j & a!=b) class;
franken-G (proven in steom_gphph_franken.py) requires replacing ALL classes
consistently.  This script derives the COMPLETE projection route (all element
classes) with sympy.secondquant, using the Fermi-NO regrouping

    op1(fp) + op2(vp) = op1(fN) + V2_NO + const ,   fN = fp + Fc  (dressed Fock)

so the one-body operand is the full dressed Fock fN with blocks
{ov=Fov, oo=Loo, vv=Lvv, vo=Fvo} — i.e. everything is bar_h-expressible.
Route = [S, op1(fN)+V2] + 1/2[S,[S, op1(fN)+V2]]  (order-2 exact on 1h1p).

Verification (H2O FC1 sto-3g, det oracle):
  truth  = raw singles blocks of (G - Hbar),  G = expm(S) Hbar expm(-S)
  layerA = sympy expr evaluated at SO level (WE.eval_expr)      -> machine-exact
  layerB = spatial expansion (GEN.expand w/ delta + x alts)     -> machine-exact
           with det-sliced spatial arrays; then analytic bar_h  -> ~1e-9
Per element class (off / semi / diag).  Saves the full spatial blueprint
(Ms and Mc separately) to script/gphph_projection_full_terms.json.

Run:  wsl python3 script/steom_gphph_diagsemi.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor, KroneckerDelta
import steom_gphph_wickeval as WE
import steom_gphph_spatial_gen as GEN
import steom_gphph_hbar3 as H3
import steom_ip_route_derive as IPD
import steom_ea_spinadapt as EA
import steom_cfour_weff as CW
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                 solve_ea, hf_det, so_index, occ_so, vir_so,
                                 apply_string)

Ha = 27.211386245988


# ------------------------------------------------------------------ sympy setup
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


def derive_full_exprs():
    """-> list of sympy exprs: linear [S,O] and quadratic 1/2[S,[S,O]] for
    O in {O1(fN), V2} and all S combos.  Externals i,a,j,b."""
    i, j = symbols('i j', below_fermi=True)
    a, b = symbols('a b', above_fermi=True)
    bra = Fd(i) * F(a); ket = Fd(b) * F(j)
    O1, V2, Sip, Sea = make_pieces()
    exprs = []
    for Of in (O1, V2):
        for S1f in (lambda: Sip('1'), lambda: Sea('1')):
            O = Of(); S1 = S1f()
            exprs.append(WE.contract(bra * (S1 * O - O * S1) * ket))
        for SAf, SBf in [(lambda: Sip('1'), lambda: Sip('2')),
                         (lambda: Sea('1'), lambda: Sea('2')),
                         (lambda: Sip('1'), lambda: Sea('2')),
                         (lambda: Sea('1'), lambda: Sip('2'))]:
            O = Of(); SA = SAf(); SB = SBf()
            inn = SB * O - O * SB
            exprs.append(WE.contract(bra * Rational(1, 2) * (SA * inn - inn * SA) * ket))
    return (i, a, j, b), exprs


# ---------------------------------------------- struct extraction (GEN format)
def struct_from_expr(expr, ext):
    """sympy expr -> GEN term list [(coeff, [(kind, [(letter, space)...])...])]."""
    i, a, j, b = ext
    extmap = {i: 'i', a: 'a', j: 'j', b: 'b'}
    terms = []
    for term in Add.make_args(expr):
        if term == 0:
            continue
        coeff = 1.0; ops = []; names = {}

        def tok(sym):
            if sym in extmap:
                nm = extmap[sym]
                return (nm, 'o' if nm in ('i', 'j') else 'v')
            bf = sym.assumptions0.get('below_fermi')
            af = sym.assumptions0.get('above_fermi')
            if not bf and not af:
                raise RuntimeError(f"index {sym} has no fermi assumption in {term}")
            if sym not in names:
                names[sym] = f"d{len(names)}"
            return (names[sym], 'o' if bf else 'v')

        for f in Mul.make_args(term):
            if f.is_number:
                coeff *= float(f)
            elif isinstance(f, AntiSymmetricTensor):
                ops.append(('v', [tok(s) for s in list(f.upper) + list(f.lower)]))
            elif isinstance(f, KroneckerDelta):
                ops.append(('delta', [tok(s) for s in f.args]))
            elif f.is_Indexed:
                base = str(f.base)
                kind = {'s': 's', 'se': 'se', 'x': 'x'}[base]
                ops.append((kind, [tok(s) for s in f.indices]))
            else:
                coeff *= float(f)
        terms.append((coeff, ops))
    return terms


# ------------------------------------------------------- spatial x alternative
def x_alts(toks, sp):
    """one-body fN operand, spin-diagonal, block-dispatched to bar_h arrays."""
    P, Q = toks
    if sp[P[0]] != sp[Q[0]]:
        return []
    blk = P[1] + Q[1]
    arr = {'oo': 'Xoo', 'ov': 'Xov', 'vo': 'Xvo', 'vv': 'Xvv'}[blk]
    return [(+1, arr, [P, Q])]


GEN.ALTS['x'] = x_alts


# ------------------------------------------------------------------ det truth
def det_block_raw(data, dets, index, O):
    """raw <i^a s|O|j^b s'> blocks (NO E_N subtraction). Ms = alpha/alpha,
    Mc = alpha/beta."""
    nact = data["nact"]; nocc = data["nocc"]; nvir = data["nvir"]
    hf = hf_det(data)

    def sdet(i, a, s):
        I = so_index(i, s, nact); A = so_index(a + nocc, s, nact)
        sg, dd = apply_string(hf, [("c", A), ("a", I)])
        return index[dd], sg
    Ms = np.zeros((nocc, nvir, nocc, nvir)); Mc = np.zeros_like(Ms)
    for i in range(nocc):
        for a in range(nvir):
            ida, sga = sdet(i, a, 0)
            for j in range(nocc):
                for b in range(nvir):
                    jda, ta = sdet(j, b, 0); jdb, tb = sdet(j, b, 1)
                    Ms[i, a, j, b] = O[ida, jda] * sga * ta
                    Mc[i, a, j, b] = O[ida, jdb] * sga * tb
    return Ms, Mc


def cls_norm(T, nocc, nvir):
    off = semi = diag = 0.
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    v2 = T[i, a, j, b] ** 2
                    if i != j and a != b:
                        off += v2
                    elif i == j and a == b:
                        diag += v2
                    else:
                        semi += v2
    return off ** 0.5, semi ** 0.5, diag ** 0.5


def report_cls(tag, D, T, nocc, nvir):
    do = cls_norm(D, nocc, nvir); dt = cls_norm(T, nocc, nvir)
    print(f"  [{tag}] diff off={do[0]:.3e} semi={do[1]:.3e} diag={do[2]:.3e}   "
          f"(truth norms off={dt[0]:.4f} semi={dt[1]:.4f} diag={dt[2]:.4f})")


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
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    spso = np.zeros((nso,) * 4); seso = np.zeros((nso,) * 4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})

    # ---- TRUTH: raw blocks of (G - Hbar), all orders/bodies ----
    G = expm(S) @ Hbar @ expm(-S)
    R = G - Hbar
    Ms_t, Mc_t = det_block_raw(data, dets, index, R)
    c_hf = R[index[hf_det(data)], index[hf_det(data)]]
    print(f"<HF|G-Hbar|HF> = {c_hf:.6e}  (route HF constant)")

    # cross-check: lin+quad of op1(fp)+op2(vp) vs full expm (3/4-body residual)
    Ofull = H3.opk_matrix(dets, index, fp, 1) + H3.opk_matrix(dets, index, vp, 2)
    linc = S @ Ofull - Ofull @ S
    inr = S @ Ofull - Ofull @ S; qd = 0.5 * (S @ inr - inr @ S)
    Ms_lq, Mc_lq = det_block_raw(data, dets, index, linc + qd)
    print("cross-check lin+quad(op1(fp)+op2(vp)) vs expm truth:")
    report_cls("Ms lq-vs-expm", Ms_lq - Ms_t, Ms_t, nocc, nvir)
    report_cls("Mc lq-vs-expm", Mc_lq - Mc_t, Mc_t, nocc, nvir)

    # ---- derive full sympy exprs ----
    print("deriving full linear+quadratic sympy exprs (slow)...")
    ext, exprs = derive_full_exprs()
    i, a, j, b = ext

    # ---- Layer A: SO-level evaluation ----
    arrs = {'v': vp, 's': spso, 'se': seso, 'x': fN}
    nzero = sum(1 for e_ in exprs if e_ == 0)
    print(f"[sympy] {len(exprs)} exprs, {nzero} identically zero "
          f"(all quad one-body pieces vanish on the 1h1p block)")
    V = np.zeros((nso,) * 4)
    for e_ in exprs:
        if e_ == 0:
            continue
        V += WE.eval_expr(e_, arrs, [i, a, j, b], nso, occ, vir)
    oa = [so_index(k, 0, nact) for k in range(nocc)]
    ob = [so_index(k, 1, nact) for k in range(nocc)]
    va = [so_index(k + nocc, 0, nact) for k in range(nvir)]
    vb = [so_index(k + nocc, 1, nact) for k in range(nvir)]
    A_Ms = V[np.ix_(oa, va, oa, va)]
    A_Mc = V[np.ix_(oa, va, ob, vb)]
    print("Layer A (SO eval) vs raw det truth:")
    report_cls("Ms A", A_Ms - Ms_t, Ms_t, nocc, nvir)
    report_cls("Mc A", A_Mc - Mc_t, Mc_t, nocc, nvir)

    # ---- Layer B: spatial expansion ----
    terms = []
    for e_ in exprs:
        if e_ == 0:
            continue
        terms += struct_from_expr(e_, ext)
    Ms_terms = GEN.expand(terms, 0)
    Mc_terms = GEN.expand(terms, 1)
    print(f"[gen-full] raw={len(terms)}  Ms spatial terms={len(Ms_terms)}  "
          f"Mc spatial terms={len(Mc_terms)}")

    # det-sliced spatial arrays (machine-exact route to the same truth)
    det_arrays = {
        'Fov':      fN[np.ix_(oa, va)],
        'Xov':      fN[np.ix_(oa, va)],
        'Xoo':      fN[np.ix_(oa, oa)],
        'Xvo':      fN[np.ix_(va, oa)],
        'Xvv':      fN[np.ix_(va, va)],
        'Wooov':    vp[np.ix_(oa, ob, oa, vb)],
        'Wvovv':    vp[np.ix_(va, ob, va, vb)],
        'eri_ovov': np.einsum("IJAB->IAJB", vp[np.ix_(oa, ob, va, vb)]).copy(),
        'rx': rx, 'ry': ry,
    }
    B_Ms = GEN.evaluate(Ms_terms, det_arrays, nocc, nvir)
    B_Mc = GEN.evaluate(Mc_terms, det_arrays, nocc, nvir)
    print("Layer B (spatial, det-sliced arrays) vs raw det truth:")
    report_cls("Ms B", B_Ms - Ms_t, Ms_t, nocc, nvir)
    report_cls("Mc B", B_Mc - Mc_t, Mc_t, nocc, nvir)

    # ---- Layer B': analytic bar_h arrays (C++ availability check) ----
    d = CW.load(xyz, basis, ncore); bar = d["bar"]
    print("bar_h vs det fN blocks:  ||Loo-fN_oo||=%.3e  ||Lvv-fN_vv||=%.3e  "
          "||Fov-fN_ov||=%.3e  ||fN_vo||=%.3f" % (
              np.linalg.norm(bar["Loo"] - det_arrays['Xoo']),
              np.linalg.norm(bar["Lvv"] - det_arrays['Xvv']),
              np.linalg.norm(bar["Fov"] - det_arrays['Xov']),
              np.linalg.norm(det_arrays['Xvo'])))
    ana_arrays = {
        'Fov': bar["Fov"], 'Xov': bar["Fov"], 'Xoo': bar["Loo"],
        'Xvo': det_arrays['Xvo'], 'Xvv': bar["Lvv"],
        'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"], 'eri_ovov': bar["eri_ovov"],
        'rx': rx, 'ry': ry,
    }
    C_Ms = GEN.evaluate(Ms_terms, ana_arrays, nocc, nvir)
    C_Mc = GEN.evaluate(Mc_terms, ana_arrays, nocc, nvir)
    print("Layer B' (spatial, analytic bar_h arrays) vs raw det truth:")
    report_cls("Ms B'", C_Ms - Ms_t, Ms_t, nocc, nvir)
    report_cls("Mc B'", C_Mc - Mc_t, Mc_t, nocc, nvir)

    # which arrays does the final term list actually use?
    used = sorted({arr for _, ops in (Ms_terms + Mc_terms) for arr, _ in ops})
    print("used arrays:", used)

    # ---- save blueprint ----
    def blob(terms):
        return [{"coeff": c, "ops": [[arr, ["".join(t[0]) for t in toks]]
                                     for arr, toks in ops]} for c, ops in terms]
    with open("script/gphph_projection_full_terms.json", "w") as f:
        json.dump({"ms": blob(Ms_terms), "mc": blob(Mc_terms)}, f, indent=1)
    print(f"[saved] script/gphph_projection_full_terms.json  "
          f"(Ms {len(Ms_terms)} + Mc {len(Mc_terms)} spatial terms)")


if __name__ == "__main__":
    main()
