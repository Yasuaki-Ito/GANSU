#!/usr/bin/env python3
"""Extract the FULL normal-ordered (true-vacuum) many-body parts of Hbar via
low-particle sectors (a k-body operator vanishes on <k-particle states):
  E0' = <0|Hbar|0>              (0e sector)
  f'  = 1-body coeff            (1e sector, minus E0)
  v'  = 2-body coeff            (2e sector, minus E0,f')
  w'  = 3-body coeff (ALL blocks, incl mixed) (3e sector remainder)
Verify E0'+op1(f')+op2(v')+op3(w') reproduces Hbar_N (remainder=4-body).
Then the g_phph Ms route's 3-body part uses the FULL w' (not the (T)-driver block,
which was orthogonal).  Fit Ms to s x {f', v', w'} in SO.

Run: wsl python3 script/steom_gphph_hbar3.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
from steom_fockspace_ref import (get_active_data, build_dets, build_H, build_t_so,
                                  build_excitation_matrix, apply_string, hf_det,
                                  so_index, occ_so, vir_so, build_sector)


def hbar_in_sector(data, nelec):
    nso = 2 * data["nact"]
    dets, index = build_dets(nso, nelec)
    H, _ = build_H(data, dets, index)
    t1so, t2so = build_t_so(data)
    T = (build_excitation_matrix(data, dets, index, t1so, "t1")
         + build_excitation_matrix(data, dets, index, t2so, "t2"))
    Hbar = expm(-T) @ H @ expm(T)
    return dets, index, Hbar


def opk_matrix(dets, index, coeff, k):
    """det matrix of (1/(k!)^2) Σ coeff[p1..pk,q1..qk] a†_p1..a†_pk a_qk..a_q1.
    Iterates only nonzero coeff entries."""
    import math
    N = len(dets); M = np.zeros((N, N))
    fac = 1.0 / (math.factorial(k) ** 2)
    nz = np.argwhere(np.abs(coeff) > 1e-13)
    for Jc, det in enumerate(dets):
        for entry in nz:
            ps = tuple(entry[:k]); qs = tuple(entry[k:])
            ops = [("c", p) for p in ps] + [("a", q) for q in reversed(qs)]
            sg, d = apply_string(det, ops)
            if sg == 0:
                continue
            M[index[d], Jc] += fac * sg * coeff[tuple(entry)]
    return M


def read_coeff(dets, index, Rem, k, nso):
    """read the antisym k-body coeff w[p1..pk,q1..qk] = <p1..pk|Rem|q1..qk> from a
    remainder matrix that is PURELY k-body in this sector (k-particle dets)."""
    W = np.zeros((nso,) * (2 * k))
    # map: k-subset (sorted) -> (det, canonical sign for a†_{sorted}|0>)
    subs = {}
    for combo in itertools.combinations(range(nso), k):
        sg, d = apply_string(0, [("c", p) for p in combo])
        if sg != 0 and d in index:
            subs[combo] = (index[d], sg)
    for cb, (rc, sc) in subs.items():
        for kb, (cc, sk) in subs.items():
            val = Rem[rc, cc] * sc * sk
            if abs(val) < 1e-13:
                continue
            # scatter with full antisymmetry over both groups
            for pp, sp in _perms(cb):
                for qq, sq in _perms(kb):
                    W[tuple(pp) + tuple(qq)] = sp * sq * val
    return W


def _perms(combo):
    out = []
    base = list(combo)
    for perm in itertools.permutations(range(len(combo))):
        sign = _parity(perm)
        out.append(([base[i] for i in perm], sign))
    return out


def _parity(perm):
    perm = list(perm); s = 1
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                s = -s
    return s


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    print(f"H2O FC1: nso={nso}  extracting Hbar many-body parts via sectors 0..3")
    # sector 0,1,2,3
    d0, i0, H0 = hbar_in_sector(data, 0)
    E0 = H0[0, 0]
    d1, i1, H1 = hbar_in_sector(data, 1)
    fp = read_coeff(d1, i1, H1 - E0 * np.eye(len(d1)), 1, nso)
    print(f"  E0'={E0:.6f}  ||f'||={np.linalg.norm(fp):.4f}")
    d2, i2, H2 = hbar_in_sector(data, 2)
    Rem2 = H2 - E0 * np.eye(len(d2)) - opk_matrix(d2, i2, fp, 1)
    vp = read_coeff(d2, i2, Rem2, 2, nso)
    print(f"  ||v'||={np.linalg.norm(vp):.4f}")
    d3, i3, H3 = hbar_in_sector(data, 3)
    Rem3 = H3 - E0 * np.eye(len(d3)) - opk_matrix(d3, i3, fp, 1) - opk_matrix(d3, i3, vp, 2)
    wp = read_coeff(d3, i3, Rem3, 3, nso)
    print(f"  ||w'||={np.linalg.norm(wp):.4f}  (FULL 3-body vertex, all blocks)")
    np.savez("/tmp/hbar_mbody.npz", fp=fp, vp=vp, wp=wp, E0=E0)
    route_fit(data, fp, vp, wp)


def route_fit(data, fp, vp, wp):
    """Fit the det Ms same-spin route to SO candidates s x {f', v', w'} (full vertices)."""
    import steom_ip_route_derive as IPD
    from steom_so_derive import det_singles_block
    from steom_fockspace_ref import build_S, solve_ip
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso)) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, _, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, _, _ = det_singles_block(data, dets, index, Hbar + comm)
    Ms = Ms1 - Ms0
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean:
        sip[m] = SIP_clean[m]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    cand = {}
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            v = slc(Tp)
            if np.linalg.norm(v) < 1e-9:
                continue
            cand[f"{tag}:p{perm}"] = v
    slegs = ['e', 'p', 'q', 'r']    # sip axes = m,I,J,B
    # s x f' : contract 1 of 4 s-legs with 1 of 2 f'-legs -> [3 s-free + 1 f'-free]
    for s1 in range(4):
        for w1 in range(2):
            sl = list(slegs); wl = ['P', 'Q']; sl[s1] = 'X'; wl[w1] = 'X'
            out = [c for c in sl if c != 'X'] + [wl[t] for t in range(2) if t != w1]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try: add(np.einsum(es, sip, fp, optimize=True), f"f:{es}")
            except Exception: pass
    # s x v' : contract 2 of 4 s-legs with 2 of 4 v'-legs -> [2 s-free + 2 v'-free]
    for s2 in itertools.combinations(range(4), 2):
        for w2 in itertools.permutations(range(4), 2):
            sl = list(slegs); wl = ['P', 'Q', 'R', 'S']
            for si, wi in zip(s2, w2): sl[si] = {0: 'X', 1: 'Y'}[s2.index(si)]; wl[wi] = sl[si]
            out = [c for c in sl if c not in ('X', 'Y')] + [wl[t] for t in range(4) if t not in w2]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try: add(np.einsum(es, sip, vp, optimize=True), f"v:{es}")
            except Exception: pass
    # s x w' : contract 3 of 4 s-legs with 3 of 6 w'-legs -> [1 s-free + 3 w'-free]
    for s3 in itertools.combinations(range(4), 3):
        for w3 in itertools.permutations(range(6), 3):
            sl = list(slegs); wl = list("UVWXYZ")
            for pos, (si, wi) in enumerate(zip(s3, w3)):
                lab = "XYZ"[pos]; sl[si] = lab; wl[wi] = lab
            out = [c for c in sl if c not in ("X", "Y", "Z")] + [wl[t] for t in range(6) if t not in w3]
            es = f"{''.join(sl)},{''.join(wl)}->{''.join(out)}"
            try: add(np.einsum(es, sip, wp, optimize=True), f"w:{es}")
            except Exception: pass
    print(f"  ||Ms route||={np.linalg.norm(Ms):.4f}  candidates={len(cand)}")
    def fit(pred, label):
        names = [n for n in cand if pred(n)]
        if not names: print(f"    [{label}] none"); return
        A = np.stack([cand[n] for n in names], 1); tv = Ms.ravel()
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        print(f"    [{label}] ncand={len(names)} resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.3e}")
    fit(lambda n: n.startswith("f:"), "s x f' (1-body)")
    fit(lambda n: n.startswith(("f:", "v:")), "s x f',v' (dressed 2-body)")
    fit(lambda n: n.startswith("w:"), "s x w' (full 3-body) only")
    fit(lambda n: True, "s x f',v',w' (ALL)")


if __name__ == "__main__":
    main()
