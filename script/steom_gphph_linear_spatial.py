#!/usr/bin/env python3
"""C++ blueprint: compute the FULL LINEAR g_phph projection route (off-diagonal)
in spatial closed-shell via the general spin-decomposer, verify vs the SO linear
route (S_ip + S_ea). All linear terms (Fov + V2, both S ops).

Run: wsl python3 script/steom_gphph_linear_spatial.py
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
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sp = np.zeros((nso,)*4); se = np.zeros((nso,)*4)
    for m in occ_so(data): sp[m] = IPD.build_sip_recon(IPD.extract_sip(solve_ip(data, E_N), data), data)[m]
    for e in vir_so(data): se[e] = EA.build_sea_recon(EA.extract_spatial_amp(solve_ea(data), data), data)[e]
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    arr = {'v': vp, 'f': fp, 's': sp, 'e': se}
    lsp = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v', 'I': 'o', 'J': 'o', 'K': 'o', 'A': 'v', 'B': 'v'}
    ext = set('iajb')

    def slist(opstr, smap):
        out = []
        for ch in opstr:
            s = 0 if ch in ext else smap[ch]
            base = range(nocc) if lsp[ch] == 'o' else range(nocc, nact) if False else range(nvir)
            if lsp[ch] == 'o':
                out.append([so_index(x, s, nact) for x in range(nocc)])
            else:
                out.append([so_index(x+nocc, s, nact) for x in range(nvir)])
        return out

    def route_spatial(terms):
        res = np.zeros((nso,)*4)
        for c, sub, o1, o2 in terms:
            lhs, o = sub.split("->"); A, B = lhs.split(",")
            internal = sorted(set(lhs.replace(",", "")) - ext)
            for spins in itertools.product([0, 1], repeat=len(internal)):
                sm = dict(zip(internal, spins))
                t = c*np.einsum(sub, arr[o1][np.ix_(*slist(A, sm))], arr[o2][np.ix_(*slist(B, sm))], optimize=True)
                res[np.ix_(oa, va, oa, va)] += t
        return res[np.ix_(oa, va, oa, va)]

    # all LINEAR off-diagonal terms (SO), (coeff, subscript, integral, amplitude)
    terms = [
        # S_ip Fov
        (+0.5, "Ib,jIia->iajb", 'f', 's'), (-0.5, "Jb,jiJa->iajb", 'f', 's'),
        # S_ip V2
        (-0.5, "IKbi,jIKa->iajb", 'v', 's'), (-0.5, "aIbA,jIiA->iajb", 'v', 's'), (+0.5, "aIbA,jiIA->iajb", 'v', 's'),
        # S_ea Fov
        (-0.5, "jA,biAa->iajb", 'f', 'e'), (+0.5, "jB,biaB->iajb", 'f', 'e'),
        # S_ea V2
        (+0.5, "ajAB,biAB->iajb", 'v', 'e'), (-0.5, "jIiA,bIAa->iajb", 'v', 'e'), (+0.5, "jIiA,bIaA->iajb", 'v', 'e'),
    ]
    spat = route_spatial(terms)

    # reference: direct SO evaluation of the SAME validated terms (Fermi-NO g_phph route)
    ref = np.zeros((nocc, nvir, nocc, nvir))
    for c, sub, o1, o2 in terms:
        ref += (c*np.einsum(sub, arr[o1], arr[o2], optimize=True))[np.ix_(oa, va, oa, va)]
    # off-diagonal comparison
    off = offr = 0.
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i != j and a != b:
                        off += (spat-ref)[i, a, j, b]**2; offr += ref[i, a, j, b]**2
    print(f"FULL LINEAR g_phph route (spatial spin-decomposed) vs SO off-diag: {(off/offr)**0.5:.3e}")
    print(f"  (||ref off-diag||={offr**0.5:.4f})  [C++ blueprint: 10 spatial terms, spin-enumerated]")


if __name__ == "__main__":
    main()
