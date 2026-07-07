#!/usr/bin/env python3
"""Confirm (multi-system, non-overfit) that the g_phph Ms route = s x {f', v'}
(1-body + FULL dressed 2-body Hbar vertex, ALL blocks) with NO 3-body.  Single
system was machine-zero but overfit (456 cand > 64 data).  Joint over many C1
systems (data >> params, shared coeffs) is the real test.  f',v' need only
sectors 0-2 (fast).

Run: wsl python3 script/steom_gphph_2body_joint.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  hf_det, so_index, occ_so, vir_so)


def extract_fv(data):
    nso = 2 * data["nact"]
    _, _, H0 = H3.hbar_in_sector(data, 0); E0 = H0[0, 0]
    _, i1, H1 = H3.hbar_in_sector(data, 1)
    fp = H3.read_coeff(*_di(i1), H1 - E0 * np.eye(H1.shape[0]), 1, nso)
    d2, i2, H2 = H3.hbar_in_sector(data, 2)
    Rem2 = H2 - E0 * np.eye(H2.shape[0]) - H3.opk_matrix(d2, i2, fp, 1)
    vp = H3.read_coeff(d2, i2, Rem2, 2, nso)
    return fp, vp


def _di(index):
    # read_coeff needs (dets,index); reconstruct dets list from index dict
    dets = [None] * len(index)
    for d, k in index.items():
        dets[k] = d
    return dets, index


def system_cands(data):
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    fp, vp = extract_fv(data)
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso)) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, _, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, _, _ = det_singles_block(data, dets, index, Hbar + comm)
    Ms = (Ms1 - Ms0).ravel()
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean: sip[m] = SIP_clean[m]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    C = {}
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso): continue
            v = slc(Tp)
            if np.linalg.norm(v) < 1e-9: continue
            C[f"{tag}:p{perm}"] = v
    slegs = ['e', 'p', 'q', 'r']
    for s1 in range(4):
        for w1 in range(2):
            sl = list(slegs); wl = ['P', 'Q']; sl[s1] = 'X'; wl[w1] = 'X'
            out = [c for c in sl if c != 'X'] + [wl[t] for t in range(2) if t != w1]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try: add(np.einsum(es, sip, fp, optimize=True), f"f:{es}")
            except Exception: pass
    for s2 in itertools.combinations(range(4), 2):
        for w2 in itertools.permutations(range(4), 2):
            sl = list(slegs); wl = ['P', 'Q', 'R', 'S']
            for pos, (si, wi) in enumerate(zip(s2, w2)):
                lab = "XY"[pos]; sl[si] = lab; wl[wi] = lab
            out = [c for c in sl if c not in ('X', 'Y')] + [wl[t] for t in range(4) if t not in w2]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try: add(np.einsum(es, sip, vp, optimize=True), f"v:{es}")
            except Exception: pass
    return C, Ms


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
        print(f"loading {kw['atom'][:22]}... ncore={kw['ncore']}")
        data = get_active_data(atom=kw["atom"], basis="sto-3g", ncore=kw["ncore"])
        systems.append(system_cands(data))
    common = set(systems[0][0])
    for C, _ in systems[1:]: common &= set(C)
    common = sorted(common)
    ndata = sum(len(Ms) for _, Ms in systems)
    print(f"\ncommon candidates={len(common)}  total data rows={ndata}")
    A = np.concatenate([np.stack([C[n] for n in common], 1) for C, _ in systems], 0)
    t = np.concatenate([Ms for _, Ms in systems])
    co, *_ = np.linalg.lstsq(A, t, rcond=None)
    print(f"JOINT s x {{f',v'}} resid = {np.linalg.norm(A@co-t)/np.linalg.norm(t):.3e}  "
          f"(->0 means Ms is 2-body-complete, NO 3-body)")
    # greedy sparse
    res = t.copy(); chosen = []
    for step in range(8):
        best = min(range(len(common)),
                   key=lambda i: np.linalg.norm(res - (A[:, i]@res/(A[:, i]@A[:, i]+1e-30))*A[:, i]))
        if best in chosen: break
        chosen.append(best); Ac = A[:, chosen]
        cc, *_ = np.linalg.lstsq(Ac, t, rcond=None); res = t - Ac@cc
        print(f"  step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(t):.3e}  " +
              "  ".join(f"{c:+.3f}[{common[j]}]" for c, j in zip(cc, chosen)))
        if np.linalg.norm(res)/np.linalg.norm(t) < 3e-3: break


if __name__ == "__main__":
    main()
