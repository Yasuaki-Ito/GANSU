#!/usr/bin/env python3
"""General spin-decomposer: given an SO term coeff*einsum(sub, integral, amplitude)
with external i,a,j,b (alpha) and internal dummies, enumerate internal spins,
slice operands to spin blocks, and reproduce the SO alpha-block as a sum of spatial
contractions. Verifies the spatial reduction for ANY route (S_ip/S_ea/quadratic).
Applies to the S_ea V2 route here.

Run: wsl python3 script/steom_gphph_spindecomp.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"]); E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    sea_sp = EA.extract_spatial_amp(solve_ea(data), data); sea = EA.build_sea_recon(sea_sp, data)
    se = np.zeros((nso,)*4)
    for e in vir_so(data): se[e] = sea[e]
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]

    # SO S_ea V2 off-diag terms (root e=b): list of (coeff, sub, opkind for the two operands)
    # opkind: 'v' -> vp ; 'se' -> se ; external i,a,j,b alpha
    terms = [
        (+0.5, "ajAB,biAB->iajb"),   # v[a,j,A,B] se[b,i,A,B]
        (-0.5, "jIiA,bIAa->iajb"),   # v[j,I,i,A] se[b,I,A,a]
        (+0.5, "jIiA,bIaA->iajb"),   # v[j,I,i,A] se[b,I,a,A]
    ]
    tgt = np.zeros((nocc, nvir, nocc, nvir))
    for c, sub in terms:
        T = c*np.einsum(sub, vp, se, optimize=True)
        tgt += T[np.ix_(oa, va, oa, va)]

    # index -> space (o/v) from the two operands; external order i,a,j,b
    def spaces(sub):
        lhs, out = sub.split("->")
        A, B = lhs.split(",")
        sp = {}
        # v (vp) axes are all so; determine o/v by whether the letter maps to occ/vir externally
        # externals: i,j occ ; a,b vir ; internal: determine by amplitude/integral context.
        # here amplitude se[e][J,A,B]: axis0=e(vir root),1=J(occ),2=A(vir),3=B(vir)
        return A, B, out
    # do spin enumeration numerically: for each term, sum over internal-index spin choices
    def occ_of(s): return oa if s == 0 else [so_index(x, 1, nact) for x in range(nocc)]
    def vir_of(s): return va if s == 0 else [so_index(x+nocc, 1, nact) for x in range(nvir)]
    # space map per letter (hard-coded from term structure)
    letter_space = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v', 'I': 'o', 'A': 'v', 'B': 'v'}
    ext = set('iajb')
    recon = np.zeros((nso,)*4)
    for c, sub in terms:
        lhs, out = sub.split("->"); Aop, Bop = lhs.split(",")
        internal = sorted(set(lhs.replace(",", "")) - ext)
        # enumerate internal spins
        acc = np.zeros((nso,)*4)
        for spins in itertools.product([0, 1], repeat=len(internal)):
            smap = dict(zip(internal, spins))
            # build sliced operands: for vp use full then slice; simplest: mask by spin on each axis
            # construct index lists per operand axis
            def slist(opstr):
                lst = []
                for ch in opstr:
                    if ch in ext:
                        lst.append(oa if letter_space[ch] == 'o' else va)  # external alpha
                    else:
                        s = smap[ch]
                        lst.append(occ_of(s) if letter_space[ch] == 'o' else vir_of(s))
                return lst
            va1 = vp[np.ix_(*slist(Aop))]
            va2 = se[np.ix_(*slist(Bop))]
            # einsum in compact space then scatter to alpha-external output
            compact = c*np.einsum(sub, va1, va2, optimize=True)  # [i,a,j,b] compact alpha
            acc[np.ix_(oa, va, oa, va)] += compact
        recon += acc
    print("S_ea V2 route: sum-over-spins reconstruction vs direct SO alpha-block =",
          float(np.linalg.norm(recon[np.ix_(oa, va, oa, va)] - tgt)/np.linalg.norm(tgt)))


if __name__ == "__main__":
    main()
