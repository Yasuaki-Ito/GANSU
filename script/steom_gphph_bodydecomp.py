#!/usr/bin/env python3
"""EXACT body-rank decomposition of the g_phph Ms route (no fitting):
  Ms = det_singles([S_ip, Hbar]).same ;  Hbar = E0 + op1(f') + op2(v') + op3(w')
  Ms_k = det_singles([S_ip, op_k]).same
Report ||Ms_1||,||Ms_2||,||Ms_3|| and Ms_1+Ms_2+Ms_3 vs Ms.  Tells us exactly how
much of the route is 1/2/3-body (settles the 2-body-vs-3-body question definitively).
Uses saved /tmp/hbar_mbody.npz (from steom_gphph_hbar3.py) for H2O FC1.

Run: wsl python3 script/steom_gphph_bodydecomp.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    z = np.load("/tmp/hbar_mbody.npz")
    fp, vp, wp, E0 = z["fp"], z["vp"], z["wp"], float(z["E0"])
    print(f"loaded vertices: ||f'||={np.linalg.norm(fp):.3f} ||v'||={np.linalg.norm(vp):.3f} "
          f"||w'||={np.linalg.norm(wp):.3f}")

    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso)) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)

    def route(Op):
        comm = S_ip @ Op - Op @ S_ip
        Ms0, _, _ = det_singles_block(data, dets, index, np.zeros_like(Op))
        Ms1, _, _ = det_singles_block(data, dets, index, comm)
        return Ms1  # base (Hbar=0) gives 0; comm is the route directly

    print("building op1(f')..."); O1 = H3.opk_matrix(dets, index, fp, 1)
    print("building op2(v')..."); O2 = H3.opk_matrix(dets, index, vp, 2)
    Ms1 = route(O1); Ms2 = route(O2)
    print(f"||Ms_1 (1-body)||={np.linalg.norm(Ms1):.5f}  ||Ms_2 (2-body)||={np.linalg.norm(Ms2):.5f}")

    # ---- fit Ms_1 ~ s x f'  and  Ms_2 ~ s x v'  (each clean, no cancellation) ----
    import itertools
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean: sip[m] = SIP_clean[m]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    slegs = ['e', 'p', 'q', 'r']
    # delta-correction traces from build_S: u1[J,B]=Σ_I s[I,I,J,B] (m=I), u2[I,B]=Σ_J s[J,I,J,B] (m=J)
    u1 = np.einsum("iijb->jb", sip, optimize=True)   # [J,B]  occ,vir
    u2 = np.einsum("jijb->ib", sip, optimize=True)   # [I,B]  occ,vir
    ulegs = ['p', 'q']
    def addperm(C, T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso,)*4: continue
            v = slc(Tp)
            if np.linalg.norm(v) > 1e-9: C[f"{tag}:p{perm}"] = v
    def enum_f(fp):
        C = {}
        for s1 in range(4):
            for w1 in range(2):
                sl = list(slegs); wl = ['P', 'Q']; sl[s1] = 'X'; wl[w1] = 'X'
                out = [c for c in sl if c != 'X'] + [wl[t] for t in range(2) if t != w1]
                es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
                addperm(C, np.einsum(es, sip, fp, optimize=True), f"f:{es}")
        # delta traces u x f': outer (k=0) and contract-1
        for un, uu in [("u1", u1), ("u2", u2)]:
            addperm(C, np.einsum("pq,rs->pqrs", uu, fp, optimize=True), f"f{un}:outer")
            for a1 in range(2):
                for b1 in range(2):
                    sl = list(ulegs); wl = ['R', 'S']; sl[a1] = 'X'; wl[b1] = 'X'
                    out = [c for c in sl if c != 'X'] + [wl[t] for t in range(2) if t != b1]
                    if len(out) != 2: continue
                    es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
                    Tc = np.einsum(es, uu, fp, optimize=True)   # 2-index -> broadcast? skip if not 4d
        return C
    def enum_v(vp):
        C = {}
        for s2 in itertools.combinations(range(4), 2):
            for w2 in itertools.permutations(range(4), 2):
                sl = list(slegs); wl = ['P', 'Q', 'R', 'S']
                for pos, (si, wi) in enumerate(zip(s2, w2)):
                    lab = "XY"[pos]; sl[si] = lab; wl[wi] = lab
                out = [c for c in sl if c not in ('X', 'Y')] + [wl[t] for t in range(4) if t not in w2]
                es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
                addperm(C, np.einsum(es, sip, vp, optimize=True), f"v:{es}")
        # delta traces u x v': contract 1 of u's 2 legs with 1 of v's 4 -> [1 u + 3 v] = 4
        for un, uu in [("u1", u1), ("u2", u2)]:
            for a1 in range(2):
                for w1 in range(4):
                    sl = list(ulegs); wl = ['P', 'Q', 'R', 'S']; sl[a1] = 'X'; wl[w1] = 'X'
                    out = [c for c in sl if c != 'X'] + [wl[t] for t in range(4) if t != w1]
                    es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
                    addperm(C, np.einsum(es, uu, vp, optimize=True), f"v{un}:{es}")
        return C
    for lbl, tgt, C in [("Ms_1 ~ s x f'", Ms1, enum_f(fp)), ("Ms_2 ~ s x v'", Ms2, enum_v(vp))]:
        names = list(C); A = np.stack([C[n] for n in names], 1); tv = tgt.ravel()
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        print(f"\n{lbl}: ncand={len(names)} FULL resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.3e}")
        res = tv.copy(); chosen = []
        for step in range(5):
            best = min(range(len(names)),
                       key=lambda i: np.linalg.norm(res - (A[:, i]@res/(A[:, i]@A[:, i]+1e-30))*A[:, i]))
            if best in chosen: break
            chosen.append(best); cc, *_ = np.linalg.lstsq(A[:, chosen], tv, rcond=None); res = tv - A[:, chosen]@cc
            print(f"  step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(tv):.3e}  " +
                  "  ".join(f"{c:+.3f}[{names[j]}]" for c, j in zip(cc, chosen)))
            if np.linalg.norm(res)/np.linalg.norm(tv) < 3e-3: break


if __name__ == "__main__":
    main()
