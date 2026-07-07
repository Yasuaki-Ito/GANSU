#!/usr/bin/env python3
"""SPATIAL EINSUM GENERATOR (the C++ blueprint emitter).  Expands every SO term of
the projection g_phph route (linear 10 + quadratic sympy) over internal spin
assignments, replaces each SO operand slice by its spatial bar_h / amplitude
expression, merges duplicates, and verifies the merged spatial term list against
the det-oracle route (target ~2.7e-10 with analytic bar_h arrays).

Spatial arrays:
  Fov[k,c]           dressed 1-body ov
  Wooov[k,l,i,d]     dressed <kl|id> (direct)
  Wvovv[a,l,c,d]     dressed <al|cd> (direct)
  eri_ovov[i,a,j,b]  bare (ia|jb) chemist
  rx[m,i,j,b]        spatial s_IP (root m)
  ry[e,j,a,b]        spatial s_EA (root e)

Output: merged (coeff, [(arr, idx-letters)...]) lists for route = Mc - Ms.

Run:  wsl python3 script/steom_gphph_spatial_gen.py
"""
import os, sys, itertools, json
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor, KroneckerDelta
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)

# ---------------------------------------------------------------- term structures
# index token = (name:str, space:'o'|'v');  op = (kind, [tokens]);  term = (coeff, [ops])
LETTER_SPACE = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v', 'I': 'o', 'J': 'o', 'K': 'o',
                'A': 'v', 'B': 'v'}
EXT_NAMES = ('i', 'a', 'j', 'b')


def linear_struct():
    lin = [
        (+0.5, "Ib,jIia", 'g', 's'), (-0.5, "Jb,jiJa", 'g', 's'),
        (-0.5, "IKbi,jIKa", 'v', 's'), (-0.5, "aIbA,jIiA", 'v', 's'), (+0.5, "aIbA,jiIA", 'v', 's'),
        (-0.5, "jA,biAa", 'g', 'e'), (+0.5, "jB,biaB", 'g', 'e'),
        (+0.5, "ajAB,biAB", 'v', 'e'), (-0.5, "jIiA,bIAa", 'v', 'e'), (+0.5, "jIiA,bIaA", 'v', 'e'),
    ]
    out = []
    for c, sub, k1, k2 in lin:
        A, B = sub.split(",")
        op1 = (k1, [(ch, LETTER_SPACE[ch]) for ch in A])
        op2 = ('s' if k2 == 's' else 'se', [(ch, LETTER_SPACE[ch]) for ch in B])
        out.append((c, [op1, op2]))
    return out


def quad_struct():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4)*v*NO(Fd(p)*Fd(q)*F(t)*F(r))
    ss = IndexedBase('s'); sea = IndexedBase('se')
    def Sip(tg):
        M, I, J = symbols(f'M{tg} I{tg} J{tg}', below_fermi=True); B = symbols(f'B{tg}', above_fermi=True)
        return Rational(1, 2)*ss[M, I, J, B]*NO(Fd(M)*F(I)*Fd(B)*F(J))
    def Sea(tg):
        E, A, B = symbols(f'E{tg} A{tg} B{tg}', above_fermi=True); J = symbols(f'J{tg}', below_fermi=True)
        return Rational(1, 2)*sea[E, J, A, B]*NO(Fd(A)*F(E)*Fd(B)*F(J))
    ext = {i: 'i', a: 'a', j: 'j', b: 'b'}
    terms = []
    for SA, SB in [(Sip('1'), Sip('2')), (Sea('1'), Sea('2')), (Sip('1'), Sea('2')), (Sea('1'), Sip('2'))]:
        inn = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inn - inn*SA)
        expr = WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))
        for term in Add.make_args(expr):
            if term == 0: continue
            coeff = 1.0; ops = []; names = {}
            def tok(sym):
                if sym in ext: return (ext[sym], 'o' if ext[sym] in ('i', 'j') else 'v')
                if sym not in names:
                    names[sym] = f"d{len(names)}"
                sp = 'o' if sym.assumptions0.get('below_fermi') else 'v'
                return (names[sym], sp)
            for f in Mul.make_args(term):
                if f.is_number:
                    coeff *= float(f)
                elif isinstance(f, AntiSymmetricTensor):
                    ops.append(('v', [tok(s) for s in list(f.upper)+list(f.lower)]))
                elif isinstance(f, KroneckerDelta):
                    ops.append(('delta', [tok(s) for s in f.args]))
                elif f.is_Indexed:
                    ops.append(('s' if str(f.base) == 's' else 'se', [tok(s) for s in f.indices]))
                else:
                    coeff *= float(f)
            terms.append((coeff, ops))
    return terms


# ------------------------------------------------------- spin-alternative tables
def v_alts(toks, sp):
    """SO antisym v operand -> spatial alternatives [(sign, arr, [tokens])].
    Blocks: ooov, oovo(image), vovv, oovv."""
    spaces = "".join(t[1] for t in toks)
    s = [sp[t[0]] for t in toks]
    P, Q, R, S = toks
    if spaces == "ooov":
        out = []
        if s[0] == s[2] and s[1] == s[3]: out.append((+1, 'Wooov', [P, Q, R, S]))
        if s[0] == s[3] and s[1] == s[2]: out.append((-1, 'Wooov', [Q, P, R, S]))
        return out
    if spaces == "oovo":   # v[K,L,D,I] = -v[K,L,I,D]
        neg = v_alts([P, Q, S, R], sp)
        return [(-g, arr, o) for g, arr, o in neg]
    if spaces == "vovv":
        out = []
        if s[0] == s[2] and s[1] == s[3]: out.append((+1, 'Wvovv', [P, Q, R, S]))
        if s[0] == s[3] and s[1] == s[2]: out.append((-1, 'Wvovv', [P, Q, S, R]))
        return out
    if spaces == "oovv":
        out = []
        if s[0] == s[2] and s[1] == s[3]: out.append((+1, 'eri_ovov', [P, R, Q, S]))
        if s[0] == s[3] and s[1] == s[2]: out.append((-1, 'eri_ovov', [P, S, Q, R]))
        return out
    raise RuntimeError(f"v block {spaces} unsupported")


def s_alts(toks, sp):
    """sp[M,I,J,B] -> rx alternatives (root M).  Patterns from build_sip_recon."""
    M, I, J, B = toks
    sm, si, sj, sb = sp[M[0]], sp[I[0]], sp[J[0]], sp[B[0]]
    pat = (si ^ sm, sj ^ sm, sb ^ sm)   # relative to root spin (Kramers mirror)
    if pat == (0, 0, 0): return [(+1, 'rx', [M, J, I, B]), (-1, 'rx', [M, I, J, B])]
    if pat == (0, 1, 1): return [(-1, 'rx', [M, I, J, B])]
    if pat == (1, 0, 1): return [(+1, 'rx', [M, J, I, B])]
    return []


def se_alts(toks, sp):
    """se[E,J,A,B] -> ry alternatives (root E).  Patterns from build_sea_recon."""
    E, J, A, B = toks
    se_, sj, sa, sb = sp[E[0]], sp[J[0]], sp[A[0]], sp[B[0]]
    pat = (sj ^ se_, sa ^ se_, sb ^ se_)
    if pat == (0, 0, 0): return [(+1, 'ry', [E, J, A, B]), (-1, 'ry', [E, J, B, A])]
    if pat == (1, 0, 1): return [(+1, 'ry', [E, J, A, B])]
    if pat == (1, 1, 0): return [(-1, 'ry', [E, J, A, B])]
    return []


def g_alts(toks, sp):
    K, C = toks
    return [(+1, 'Fov', [K, C])] if sp[K[0]] == sp[C[0]] else []


def delta_alts(toks, sp):
    P, Q = toks
    if sp[P[0]] != sp[Q[0]]: return []
    return [(+1, 'delta_o' if P[1] == 'o' else 'delta_v', [P, Q])]


ALTS = {'v': v_alts, 's': s_alts, 'se': se_alts, 'g': g_alts, 'delta': delta_alts}


# ---------------------------------------------------------------- expansion+merge
def expand(terms, ket_spin):
    """-> list of (coeff, ((arr, (letters...)), ...)) canonically renamed/sorted."""
    out = []
    for coeff, ops in terms:
        internal = []
        for _, toks in ops:
            for t in toks:
                if t[0] not in EXT_NAMES and t[0] not in internal: internal.append(t[0])
        for spins in itertools.product([0, 1], repeat=len(internal)):
            sp = dict(zip(internal, spins))
            sp.update({'i': 0, 'a': 0, 'j': ket_spin, 'b': ket_spin})
            alt_lists = [ALTS[k](toks, sp) for k, toks in ops]
            if any(len(al) == 0 for al in alt_lists): continue
            for combo in itertools.product(*alt_lists):
                sgn = 1.0; sops = []
                for g, arr, toks in combo:
                    sgn *= g; sops.append((arr, toks))
                out.append(canon(coeff*sgn, sops))
    return merge(out)


def canon(coeff, sops):
    """sort ops (stable by arr name + external pattern), rename internals."""
    def extpat(op): return (op[0], "".join(t[0] if t[0] in EXT_NAMES else '.' for t in op[1]))
    sops = sorted(sops, key=extpat)
    ren = {}; letters = "klmncdefgh"
    def rn(t):
        nm = t[0]
        if nm in EXT_NAMES: return nm
        if nm not in ren:
            pool = [c for c in ("klmn" if t[1] == 'o' else "cdef") if c not in ren.values()]
            ren[nm] = pool[0]
        return ren[nm]
    canon_ops = tuple((arr, tuple((rn(t), t[1]) for t in toks)) for arr, toks in sops)
    return (coeff, canon_ops)


def merge(terms):
    acc = {}
    for c, ops in terms:
        acc[ops] = acc.get(ops, 0.0) + c
    return [(c, ops) for ops, c in acc.items() if abs(c) > 1e-12]


def subtract(mc, ms):
    acc = {}
    for c, ops in mc: acc[ops] = acc.get(ops, 0.0) + c
    for c, ops in ms: acc[ops] = acc.get(ops, 0.0) - c
    return [(c, ops) for ops, c in acc.items() if abs(c) > 1e-12]


# ---------------------------------------------------------------- evaluation
def evaluate(terms, arrays, nocc, nvir):
    res = np.zeros((nocc, nvir, nocc, nvir))
    dims = {'o': nocc, 'v': nvir}
    for c, ops in terms:
        operands = []; subs = []
        for arr, toks in ops:
            if arr == 'delta_o': operands.append(np.eye(nocc))
            elif arr == 'delta_v': operands.append(np.eye(nvir))
            else: operands.append(arrays[arr])
            subs.append("".join(t[0] for t in toks))
        res += c*np.einsum(",".join(subs)+"->iajb", *operands, optimize=True)
    return res


def offn(T, nocc, nvir):
    s = 0.
    for x in range(nocc):
        for y in range(nvir):
            for u in range(nocc):
                for w in range(nvir):
                    if x != u and y != w: s += T[x, y, u, w]**2
    return s**0.5


def main():
    lin = linear_struct()
    print("deriving quadratic (slow)...")
    quad = quad_struct()
    allt = lin + quad
    Ms = expand(allt, 0); Mc = expand(allt, 1); route = subtract(Mc, Ms)
    print(f"[gen] Ms terms={len(Ms)}  Mc terms={len(Mc)}  route(Mc-Ms) terms={len(route)}")

    # ------------- numerical verification vs det route (analytic bar_h arrays)
    xyz, basis, ncore = "xyz/H2O.xyz", "sto-3g", 1
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0)
    ry = se_det
    spso = np.zeros((nso,)*4); seso = np.zeros((nso,)*4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True); fN = fp + Fc
    g_ov = np.zeros((nso, nso)); g_ov[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    op1g = H3.opk_matrix(dets, index, g_ov, 1)
    linc = S @ (op1g + V2mat) - (op1g + V2mat) @ S
    inr = S @ V2mat - V2mat @ S; qd = 0.5*(S @ inr - inr @ S)
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, linc + qd)
    det_route = Mc_d - Ms_d

    d = CW.load(xyz, basis, ncore); bar = d["bar"]
    arrays = {'Fov': bar["Fov"], 'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"],
              'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    gen_route = evaluate(route, arrays, nocc, nvir)
    print(f"[verify] gen off={offn(gen_route,nocc,nvir):.6f}  det off={offn(det_route,nocc,nvir):.6f}  "
          f"||diff|| off={offn(gen_route-det_route,nocc,nvir):.3e}")
    full = np.linalg.norm(gen_route - det_route)
    print(f"[verify] full ||diff||={full:.3e}  (incl diagonal)")

    # save blueprint
    blob = [{"coeff": c, "ops": [[arr, ["".join(t[0]) for t in toks]] for arr, toks in ops]}
            for c, ops in route]
    with open("script/gphph_projection_terms.json", "w") as f:
        json.dump(blob, f, indent=1)
    print(f"[saved] script/gphph_projection_terms.json  ({len(route)} spatial terms)")


if __name__ == "__main__":
    main()
