#!/usr/bin/env python3
"""Empirically pin the 2-body g_phph route: fit Ms_2 (= [S_ip, op2(vp)] route,
clean/no-cancellation) to comprehensive s x vp SO contractions, JOINTLY over many
C1 systems (data >> params breaks collinearity).  vp = full dressed 2-body Hbar.
Reads sparse coefficients directly -> the 2-body formula.

Run: wsl python3 script/steom_gphph_2b_joint.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_dets, build_sector, build_S,
                                  solve_ip, hf_det, so_index, occ_so, vir_so)


def extract_fv(data):
    nso = 2 * data["nact"]
    def di(idx):
        d = [None]*len(idx)
        for k, v in idx.items(): d[v] = k
        return d, idx
    _, _, H0 = H3.hbar_in_sector(data, 0); E0 = H0[0, 0]
    d1, i1, H1 = H3.hbar_in_sector(data, 1)
    fp = H3.read_coeff(d1, i1, H1 - E0*np.eye(len(d1)), 1, nso)
    d2, i2, H2 = H3.hbar_in_sector(data, 2)
    vp = H3.read_coeff(d2, i2, H2 - E0*np.eye(len(d2)) - H3.opk_matrix(d2, i2, fp, 1), 2, nso)
    return fp, vp


def system(data):
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    fp, vp = extract_fv(data)
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a+nocc, s, nact): np.zeros((nso,)*3) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    c2 = S_ip @ H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, vp, 2) @ S_ip
    Ms2, _, _ = det_singles_block(data, dets, index, c2)
    sp = np.zeros((nso,)*4)
    for m in SIP_clean: sp[m] = SIP_clean[m]
    u1 = np.einsum("iijb->jb", sp); u2 = np.einsum("jijb->ib", sp)
    occ = occ_so(data); vir = vir_so(data)
    oa = [so_index(i, 0, nact) for i in range(nocc)]; va = [so_index(a+nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    C = {}
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso,)*4: continue
            v = slc(Tp)
            if np.linalg.norm(v) > 1e-9: C[f"{tag}:p{perm}"] = v
    # s (4 legs) x vp (4 legs), contract 2
    for s2 in itertools.combinations(range(4), 2):
        for w2 in itertools.permutations(range(4), 2):
            sl = list("abcd"); wl = list("PQRS")
            for pos, (si, wi) in enumerate(zip(s2, w2)):
                lab = "XY"[pos]; sl[si] = lab; wl[wi] = lab
            free = [x for x in sl if x not in "XY"] + [wl[t] for t in range(4) if t not in w2]
            if len(free) != 4: continue
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(free)}"
            try: add(np.einsum(es, sp, vp, optimize=True), f"s:{es}")
            except Exception: pass
    # u1,u2 (2 legs) x vp, contract 1
    for un, uu in [("u1", u1), ("u2", u2)]:
        for a1 in range(2):
            for w1 in range(4):
                sl = list("ab"); wl = list("PQRS"); sl[a1] = 'X'; wl[w1] = 'X'
                free = [x for x in sl if x != 'X'] + [wl[t] for t in range(4) if t != w1]
                es = f"{''.join(sl)},{''.join(wl)}->{''.join(free)}"
                try: add(np.einsum(es, uu, vp, optimize=True), f"{un}:{es}")
                except Exception: pass
    return C, Ms2.ravel()


def main():
    defs = [
        dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=1),
        dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=1),
        dict(atom="O 0 0 0; H 0.88 0.45 -0.20; H -0.20 0.96 0.30", ncore=1),
        dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=0),
        dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=0),
        dict(atom="N 0 0 0; H 0.95 0.05 0.30; H -0.45 0.83 0.28; H -0.52 -0.78 0.35", ncore=1),
        dict(atom="N 0 0 0; H 1.02 0.10 0.20; H -0.50 0.78 0.33; H -0.40 -0.85 0.28", ncore=1),
    ]
    systems = []
    for kw in defs:
        print(f"loading {kw['atom'][:20]}...")
        systems.append(system(get_active_data(atom=kw["atom"], basis="sto-3g", ncore=kw["ncore"])))
    common = sorted(set.intersection(*[set(C) for C, _ in systems]))
    ndata = sum(len(t) for _, t in systems)
    print(f"common cand={len(common)} data={ndata}")
    A = np.concatenate([np.stack([C[n] for n in common], 1) for C, _ in systems], 0)
    t = np.concatenate([tv for _, tv in systems])
    co, *_ = np.linalg.lstsq(A, t, rcond=None)
    print(f"JOINT Ms_2 full resid={np.linalg.norm(A@co-t)/np.linalg.norm(t):.3e}")
    res = t.copy(); chosen = []
    for step in range(8):
        best = min(range(len(common)), key=lambda k: np.linalg.norm(res-(A[:,k]@res/(A[:,k]@A[:,k]+1e-30))*A[:,k]))
        if best in chosen: break
        chosen.append(best); cc, *_ = np.linalg.lstsq(A[:,chosen], t, rcond=None); res = t-A[:,chosen]@cc
        print(f"  step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(t):.3e}")
        for c, k in sorted(zip(cc, chosen), key=lambda x:-abs(x[0])):
            print(f"      {c:+.4f}  {common[k]}")
        if np.linalg.norm(res)/np.linalg.norm(t) < 5e-3: break


if __name__ == "__main__":
    main()
