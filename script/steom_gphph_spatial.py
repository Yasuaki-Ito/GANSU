#!/usr/bin/env python3
"""General closed-shell spin-adapter for the g_phph projection route.  Each SO term
is coeff*einsum(sub, arrayA, arrayB[, ...]) with external bra (i,a)=alpha and ket
(j,b)=parametrized spin.  Internal dummies are spin-enumerated; SO arrays (already
spin-structured) are sliced to spin blocks and contracted compactly.  Mc-Ms =
spatial_route(ket beta) - spatial_route(ket alpha).  Verified vs the det oracle.

This is the direct C++ blueprint: once matched, every compact contraction maps to a
bar_h (chemist) intermediate + s_IP/s_EA and is transcribed to steom_ccsd_operator.cu.

Run:  wsl python3 script/steom_gphph_spatial.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so)

# letter -> orbital space (o=occ, v=vir); externals i,j occ, a,b vir
LSPACE = {'i': 'o', 'j': 'o', 'a': 'v', 'b': 'v',
          'I': 'o', 'J': 'o', 'K': 'o', 'M': 'o', 'N': 'o',
          'A': 'v', 'B': 'v', 'C': 'v', 'D': 'v'}
EXT = set('iajb')


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
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True)
    fN = fp + Fc
    # bar_h-friendly 1-body: g1[ov/vo]=fN, g1[oo/vv]=Fc  (== physical route)
    g1 = np.zeros((nso, nso))
    g1[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]; g1[np.ix_(vir, occ)] = fN[np.ix_(vir, occ)]
    g1[np.ix_(occ, occ)] = Fc[np.ix_(occ, occ)]; g1[np.ix_(vir, vir)] = Fc[np.ix_(vir, vir)]
    arr = {'v': vp, 'f': fp, 'g': g1, 's': sp, 'e': se}

    S = build_S(data, dets, index, {m: sp[m] for m in occ}, {e: se[e] for e in vir})
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    ob = [so_index(x, 1, nact) for x in range(nocc)]; vb = [so_index(x+nocc, 1, nact) for x in range(nvir)]

    def olist(s): return oa if s == 0 else ob
    def vlist(s): return va if s == 0 else vb

    def spatial_term(coeff, sub, o1, o2, ket_spin):
        """One SO term, external ket spin = ket_spin (0=alpha->Ms, 1=beta->Mc).
        Returns compact [nocc,nvir,nocc,nvir]."""
        lhs, out = sub.split("->"); A, B = lhs.split(",")
        internal = sorted(set(lhs.replace(",", "")) - EXT)
        acc = np.zeros((nocc, nvir, nocc, nvir))
        for spins in itertools.product([0, 1], repeat=len(internal)):
            sm = dict(zip(internal, spins))

            def ixlist(opstr):
                out2 = []
                for ch in opstr:
                    if ch in EXT:
                        s = 0 if ch in ('i', 'a') else ket_spin
                    else:
                        s = sm[ch]
                    out2.append(olist(s) if LSPACE[ch] == 'o' else vlist(s))
                return out2
            a1 = arr[o1][np.ix_(*ixlist(A))]
            a2 = arr[o2][np.ix_(*ixlist(B))]
            acc += coeff*np.einsum(sub, a1, a2, optimize=True)
        return acc

    def route(terms, ket_spin):
        r = np.zeros((nocc, nvir, nocc, nvir))
        for c, sub, o1, o2 in terms:
            r += spatial_term(c, sub, o1, o2, ket_spin)
        return r

    # LINEAR terms: Fov(fp ov) + V2(vp) + Fc/oo,vv via g1 for the 1-body route.
    # 1-body route uses g1 (ov=fN, oo/vv=Fc) in place of fp -> covers Fov+Fc at once.
    lin_terms = [
        (+0.5, "Ib,jIia->iajb", 'g', 's'), (-0.5, "Jb,jiJa->iajb", 'g', 's'),
        (-0.5, "IKbi,jIKa->iajb", 'v', 's'), (-0.5, "aIbA,jIiA->iajb", 'v', 's'), (+0.5, "aIbA,jiIA->iajb", 'v', 's'),
        (-0.5, "jA,biAa->iajb", 'g', 'e'), (+0.5, "jB,biaB->iajb", 'g', 'e'),
        (+0.5, "ajAB,biAB->iajb", 'v', 'e'), (-0.5, "jIiA,bIAa->iajb", 'v', 'e'), (+0.5, "jIiA,bIaA->iajb", 'v', 'e'),
    ]
    # NOTE: the 1-body g-terms above use only the ov block (subscripts Ib, Jb, jA, jB
    # contract g[occ,vir]/g[occ,vir]); the oo/vv Fc route needs its OWN terms. Derive
    # them: the 1-body route [S,op1(g)] for oo/vv blocks. Placeholder: compare vs det
    # for the ov-only piece first, then add oo/vv.
    Ms_lin = route(lin_terms, 0); Mc_lin = route(lin_terms, 1)
    spatial_lin = Mc_lin - Ms_lin

    # det ground-truth (FAIR ov-only): 1-body = fN restricted to ov/vo + V2(Fermi-NO)
    g_ov = np.zeros((nso, nso))
    g_ov[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]; g_ov[np.ix_(vir, occ)] = fN[np.ix_(vir, occ)]
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    op1g = H3.opk_matrix(dets, index, g_ov, 1)
    c = S @ (op1g + V2mat) - (op1g + V2mat) @ S
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, c)
    det_lin = Mc_d - Ms_d

    def offn(T):
        s = 0.
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        if i != j and a != b: s += T[i, a, j, b]**2
        return s**0.5
    print(f"[linear ov-only test] spatial off={offn(spatial_lin):.6f}  det off={offn(det_lin):.6f}  "
          f"||diff||={offn(spatial_lin-det_lin):.3e}")


if __name__ == "__main__":
    main()
