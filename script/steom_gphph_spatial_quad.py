#!/usr/bin/env python3
"""Spatial closed-shell adaptation of the QUADRATIC g_phph route (1/2[S,[S,V2]],
S_ip^2 + S_ea^2 + cross).  Generalizes the spin-block adapter to arbitrary sympy
terms (integral v + two amplitudes s/se).  Each term: enumerate internal-dummy
spins, slice the SO arrays to spin blocks, compact einsum; external bra (i,a)=alpha,
ket (j,b)=spin param.  Mc-Ms = eval(beta)-eval(alpha).  Verify vs det oracle.

Run:  wsl python3 script/steom_gphph_spatial_quad.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from sympy import symbols, Rational, Dummy, IndexedBase, Add, Mul
from sympy.physics.secondquant import F, Fd, NO, AntiSymmetricTensor, KroneckerDelta
sys.path.insert(0, "script")
import steom_gphph_wickeval as WE
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)


def below(sym):
    return bool(sym.assumptions0.get('below_fermi'))


def build_spatial_evaluator(nact, nocc, nvir, arrs):
    nso = 2*nact
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]

    def olist(s): return oa if s == 0 else ob
    def vlist(s): return va if s == 0 else vb

    def eval_expr_spatial(expr, ext_syms, ket_spin):
        """Compact [nocc,nvir,nocc,nvir]. ext_syms=[i,a,j,b]; i,a alpha, j,b=ket_spin."""
        ext = set(ext_syms)
        out = np.zeros((nocc, nvir, nocc, nvir))
        for term in Add.make_args(expr):
            if term == 0:
                continue
            coeff = 1.0; tensors = []; deltas = []
            for f in Mul.make_args(term):
                if f.is_number:
                    coeff *= float(f)
                elif isinstance(f, AntiSymmetricTensor):
                    tensors.append(('v', list(f.upper)+list(f.lower)))
                elif isinstance(f, KroneckerDelta):
                    deltas.append(list(f.args))
                elif f.is_Indexed:
                    tensors.append((str(f.base), list(f.indices)))
                else:
                    try: coeff *= float(f)
                    except Exception: raise RuntimeError(f"factor {f}")
            allidx = []
            for _, ids in tensors: allidx += ids
            for d in deltas: allidx += d
            uniq = []
            for s in allidx:
                if s not in uniq: uniq.append(s)
            internal = [s for s in uniq if s not in ext]

            def spc(sym):   # 'o' or 'v'
                return 'o' if below(sym) else 'v'

            def spin_of_ext(sym):
                nm = sym.name
                return 0 if nm in ('i', 'a') else ket_spin

            acc = np.zeros((nocc, nvir, nocc, nvir))
            for spins in itertools.product([0, 1], repeat=len(internal)):
                sm = dict(zip(internal, spins))

                def idxlist(sym):
                    s = spin_of_ext(sym) if sym in ext else sm[sym]
                    return olist(s) if spc(sym) == 'o' else vlist(s)
                letters = "PQRSTUVWXYZABCDEFGHIJKLMNO"
                lab = {s: letters[k] for k, s in enumerate(uniq)}
                operands = []; subs = []
                for nm, ids in tensors:
                    operands.append(arrs[nm][np.ix_(*[idxlist(s) for s in ids])])
                    subs.append("".join(lab[s] for s in ids))
                for d in deltas:
                    # delta over compact space of its two (equal-space) indices
                    sp0 = spc(d[0])
                    n = nocc if sp0 == 'o' else nvir
                    operands.append(np.eye(n)); subs.append("".join(lab[s] for s in d))
                outsub = "".join(lab[s] for s in ext_syms)
                acc += coeff*np.einsum(",".join(subs)+"->"+outsub, *operands, optimize=True)
            out += acc
        return out
    return eval_expr_spatial


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]
    sp = np.zeros((nso,)*4); se = np.zeros((nso,)*4)
    for m in occ_so(data): sp[m] = IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data)[m]
    for e in vir_so(data): se[e] = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)[e]
    occ = occ_so(data); vir = vir_so(data)
    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]

    # det ground-truth quadratic (Mc - Ms)
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    inr = S @ V2mat - V2mat @ S; quad = 0.5*(S @ inr - inr @ S)
    Msq, Mcq, _ = det_singles_block(data, dets, index, quad); det_quad = Mcq - Msq

    # sympy quadratic expression (same as physverify)
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
    print("deriving quadratic (slow)...")
    full_expr = 0
    for SA, SB in [(Sip('1'), Sip('2')), (Sea('1'), Sea('2')), (Sip('1'), Sea('2')), (Sea('1'), Sip('2'))]:
        inner = SB*V2 - V2*SB; nq = Rational(1, 2)*(SA*inner - inner*SA)
        full_expr += WE.contract(Fd(i)*F(a)*nq*Fd(b)*F(j))

    ev = build_spatial_evaluator(nact, nocc, nvir, {'v': vp, 's': sp, 'se': se})
    Ms = ev(full_expr, [i, a, j, b], 0); Mc = ev(full_expr, [i, a, j, b], 1)
    spat_quad = Mc - Ms

    def offn(T):
        s = 0.
        for x in range(nocc):
            for y in range(nvir):
                for u in range(nocc):
                    for w in range(nvir):
                        if x != u and y != w: s += T[x, y, u, w]**2
        return s**0.5
    print(f"[quad spatial] det off={offn(det_quad):.6f}  spatial off={offn(spat_quad):.6f}  "
          f"||diff||={offn(spat_quad-det_quad):.3e}")


if __name__ == "__main__":
    main()
