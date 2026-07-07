#!/usr/bin/env python3
"""Generic evaluator: sympy Wick expression (AntiSymmetricTensor v, IndexedBase s/se,
KroneckerDelta) -> numerical [i,a,j,b] tensor.  Lets us evaluate ANY derived route
(linear or quadratic) without hand transcription.  Validate on the known linear
S_ip route (machine-exact), then use for the quadratic.

Run: wsl python3 script/steom_gphph_wickeval.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul, Integer
from sympy.physics.secondquant import (F, Fd, wicks, NO, AntiSymmetricTensor,
                                        KroneckerDelta, evaluate_deltas, substitute_dummies)
sys.path.insert(0, "script")

pretty = dict(above_fermi='abcdef', below_fermi='ijklmn')


def contract(expr):
    e = wicks(expr, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
    return substitute_dummies(evaluate_deltas(e), new_indices=True, pretty_indices=pretty)


def below(sym):
    return bool(sym.assumptions0.get('below_fermi'))


def eval_expr(expr, arrs, ext_syms, nso, occ, vir):
    """arrs: {'v':vp,'s':sp,'se':se}. ext_syms=[i,a,j,b] output order.
    Returns [nso,nso,nso,nso] over full SO (slice externals later)."""
    from numpy import einsum
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    vir_mask = np.zeros(nso); vir_mask[vir] = 1.0
    out = np.zeros((nso,)*4)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for term in Add.make_args(expr):
        factors = Mul.make_args(term)
        coeff = 1.0; tensors = []; deltas = []
        for f in factors:
            if f.is_number:
                coeff *= float(f)
            elif isinstance(f, AntiSymmetricTensor):
                tensors.append(('v', list(f.upper) + list(f.lower)))
            elif isinstance(f, KroneckerDelta):
                deltas.append(list(f.args))
            elif f.is_Indexed:
                nm = str(f.base); tensors.append((nm, list(f.indices)))
            else:
                # could be Pow or sign; try number
                try: coeff *= float(f)
                except Exception: raise RuntimeError(f"unhandled factor {f}")
        # collect all index symbols
        allidx = []
        for _, ids in tensors: allidx += ids
        for d in deltas: allidx += d
        uniq = []
        for s in allidx:
            if s not in uniq: uniq.append(s)
        # assign letters
        lab = {s: letters[k] for k, s in enumerate(uniq)}
        # build operands and subscripts
        operands = []; subs = []
        for nm, ids in tensors:
            operands.append(arrs[nm]); subs.append("".join(lab[s] for s in ids))
        # deltas -> identity operands
        for d in deltas:
            operands.append(np.eye(nso)); subs.append("".join(lab[s] for s in d))
        # dummy masks: any index NOT in ext_syms is summed; restrict its space
        dummies = [s for s in uniq if s not in ext_syms]
        for s in dummies:
            operands.append(occ_mask if below(s) else vir_mask); subs.append(lab[s])
        outsub = "".join(lab[s] for s in ext_syms)
        es = ",".join(subs) + "->" + outsub
        out += coeff * einsum(es, *operands, optimize=True)
    return out


def main():
    i, j = symbols('i j', below_fermi=True); a, b = symbols('a b', above_fermi=True)
    s = IndexedBase('s')
    M = symbols('M', below_fermi=True); I = symbols('I', below_fermi=True)
    J = symbols('J', below_fermi=True); B = symbols('B', above_fermi=True)
    Sip = Rational(1, 2) * s[M, I, J, B] * NO(Fd(M) * F(I) * Fd(B) * F(J))
    p, q, r, t = symbols('p q r t', cls=Dummy); v = AntiSymmetricTensor('v', (p, q), (r, t))
    V2 = Rational(1, 4) * v * NO(Fd(p) * Fd(q) * F(t) * F(r))
    expr = contract(Fd(i) * F(a) * (Sip * V2 - V2 * Sip) * Fd(b) * F(j))

    # numerical validate vs hand einsum (known machine-exact)
    import steom_ip_route_derive as IPD, steom_gphph_hbar3 as H3
    from steom_so_derive import det_singles_block
    from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                     hf_det, so_index, occ_so, vir_so)
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    SIP = {m: v for m, v in IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data).items() if m in occ_so(data)}
    zEA = {so_index(x+nocc, sp2, nact): np.zeros((nso,)*3) for sp2 in range(2) for x in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    occ = occ_so(data); vir = vir_so(data)
    spt = np.zeros((nso,)*4)
    for m in SIP: spt[m] = SIP[m]
    op2vp = H3.opk_matrix(dets, index, vp, 2)
    Ms2, _, _ = det_singles_block(data, dets, index, S_ip @ op2vp - op2vp @ S_ip)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    def route1(fmat):
        Op = H3.opk_matrix(dets, index, fmat, 1); c = S_ip @ Op - Op @ S_ip
        M, _, _ = det_singles_block(data, dets, index, c); return M
    w2route = Ms2 - route1(Fc)

    V = eval_expr(expr, {'v': vp, 's': spt}, [i, a, j, b], nso, occ, vir)
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    Vs = V[np.ix_(oa, va, oa, va)]
    print(f"generic-eval S_ip 2b route vs (Ms2-Fc): resid = {np.linalg.norm(Vs-w2route)/np.linalg.norm(w2route):.3e}")


if __name__ == "__main__":
    main()
