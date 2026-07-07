#!/usr/bin/env python3
"""Decompose Ms_2 (2-body route) by which 2-body Hbar block it comes from:
for each occ/vir block of v', compute ||[S_ip, op2(v'_block)].Ms|| exactly.
Reveals the formula's ingredient blocks (only a few contribute).  Same for Ms_1
by Fock block.  Uses /tmp/hbar_mbody.npz (H2O FC1).

Run: wsl python3 script/steom_gphph_blockdecomp.py
"""
import os, sys, itertools
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
    z = np.load("/tmp/hbar_mbody.npz"); fp, vp = z["fp"], z["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso)) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    occ = occ_so(data); vir = vir_so(data)
    ovlabel = {}
    for P in range(nso):
        ovlabel[P] = 'o' if P in occ else 'v'

    def route(Op):
        comm = S_ip @ Op - Op @ S_ip
        Ms1, _, _ = det_singles_block(data, dets, index, comm)
        return Ms1

    Ms_full = route(H3.opk_matrix(dets, index, vp, 2)) + route(H3.opk_matrix(dets, index, fp, 1))
    # actually recompute reference route from Hbar
    comm_full = S_ip @ Hbar - Hbar @ S_ip
    Ms0, _, _ = det_singles_block(data, dets, index, Hbar)
    Ms1r, _, _ = det_singles_block(data, dets, index, Hbar + comm_full)
    Ms_ref = Ms1r - Ms0
    print(f"||Ms_ref||={np.linalg.norm(Ms_ref):.5f}")

    # ---- 2-body block decomposition ----
    print("\n2-body block contributions ||[S_ip, op2(v'_block)].Ms||:")
    tot2 = np.zeros_like(Ms_ref)
    for blk in itertools.product("ov", repeat=4):
        tag = "".join(blk)
        mask = np.zeros((nso, nso, nso, nso))
        # build block by index membership
        idxs = [occ if c == 'o' else vir for c in blk]
        vblk = np.zeros_like(vp)
        vblk[np.ix_(*idxs)] = vp[np.ix_(*idxs)]
        if np.linalg.norm(vblk) < 1e-10:
            continue
        Op = H3.opk_matrix(dets, index, vblk, 2)
        contrib = route(Op); tot2 += contrib
        nrm = np.linalg.norm(contrib)
        if nrm > 1e-6:
            print(f"  v'[{tag}]: ||contrib||={nrm:.5f}  (||v'blk||={np.linalg.norm(vblk):.3f})")
    print(f"  sum of 2-body blocks ||={np.linalg.norm(tot2):.5f}")

    # ---- 1-body block decomposition ----
    print("\n1-body block contributions ||[S_ip, op1(f'_block)].Ms||:")
    tot1 = np.zeros_like(Ms_ref)
    for blk in itertools.product("ov", repeat=2):
        tag = "".join(blk)
        idxs = [occ if c == 'o' else vir for c in blk]
        fblk = np.zeros_like(fp); fblk[np.ix_(*idxs)] = fp[np.ix_(*idxs)]
        if np.linalg.norm(fblk) < 1e-10:
            continue
        Op = H3.opk_matrix(dets, index, fblk, 1)
        contrib = route(Op); tot1 += contrib
        nrm = np.linalg.norm(contrib)
        print(f"  f'[{tag}]: ||contrib||={nrm:.5f}  (||f'blk||={np.linalg.norm(fblk):.3f})")
    print(f"  sum of 1-body blocks ||={np.linalg.norm(tot1):.5f}")
    print(f"\n||Ms_ref - (tot1+tot2)|| = {np.linalg.norm(Ms_ref-(tot1+tot2)):.3e}")

    # ================= per-block CLEAN fits (single intermediate each) =================
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean: sip[m] = SIP_clean[m]
    u1 = np.einsum("iijb->jb", sip, optimize=True)   # delta trace m=I
    u2 = np.einsum("jijb->ib", sip, optimize=True)   # delta trace m=J
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, oa, va)].ravel()
    def contrib_block(vp_or_fp, idxs, k):
        blk = np.zeros_like(vp_or_fp); blk[np.ix_(*idxs)] = vp_or_fp[np.ix_(*idxs)]
        return route(H3.opk_matrix(dets, index, blk, k))
    def fit(tgt, srcs, label):
        # srcs: list of (name, tensor, [amp-leg-count-with-that-tensor via all einsum contractions])
        cand = {}
        def addall(amp, amp_axes, W, waxes, tag):
            # contract min(len available) ... enumerate contracting c axes (c=amp_axes∩ up to)
            import itertools as it
            for c in range(1, min(len(amp_axes), len(waxes)) + 1):
                for sset in it.combinations(range(len(amp_axes)), c):
                    for wset in it.permutations(range(len(waxes)), c):
                        sl = list("abcd")[:len(amp_axes)]; wl = list("PQRS")[:len(waxes)]
                        for pos, (si, wi) in enumerate(zip(sset, wset)):
                            lab = "XYZW"[pos]; sl[si] = lab; wl[wi] = lab
                        out = [x for x in sl if x in "XYZW" and False]  # placeholder
                        free = [x for x in sl if x not in "XYZW"] + [wl[t] for t in range(len(waxes)) if t not in wset]
                        if len(free) != 4: continue
                        es = f"{''.join(sl)},{''.join(wl)}->{''.join(free)}"
                        try: T = np.einsum(es, amp, W, optimize=True)
                        except Exception: continue
                        for perm in it.permutations(range(4)):
                            Tp = np.transpose(T, perm)
                            if Tp.shape != (nso,)*4: continue
                            v = slc(Tp)
                            if np.linalg.norm(v) > 1e-9: cand[f"{tag}:{es}:p{perm}"] = v
        for nm, W in srcs:
            addall(sip, [0, 1, 2, 3], W, list(range(W.ndim)), f"s.{nm}")
            addall(u1, [0, 1], W, list(range(W.ndim)), f"u1.{nm}")
            addall(u2, [0, 1], W, list(range(W.ndim)), f"u2.{nm}")
        names = list(cand)
        if not names: print(f"  [{label}] no cand"); return
        A = np.stack([cand[n] for n in names], 1); tv = tgt.ravel()
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        print(f"\n  [{label}] ncand={len(names)} FULL resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.2e}")
        res = tv.copy(); chosen = []
        for step in range(4):
            best = min(range(len(names)), key=lambda i: np.linalg.norm(res-(A[:,i]@res/(A[:,i]@A[:,i]+1e-30))*A[:,i]))
            if best in chosen: break
            chosen.append(best); cc, *_ = np.linalg.lstsq(A[:,chosen], tv, rcond=None); res = tv-A[:,chosen]@cc
            print(f"    step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(tv):.2e}  " +
                  "  ".join(f"{c:+.3f}[{names[j]}]" for c,j in zip(cc,chosen)))
            if np.linalg.norm(res)/np.linalg.norm(tv) < 5e-3: break
    fov = np.zeros_like(fp); fov[np.ix_(occ, vir)] = fp[np.ix_(occ, vir)]
    wooov = np.zeros_like(vp); wooov[np.ix_(occ, occ, occ, vir)] = vp[np.ix_(occ, occ, occ, vir)]
    wovvv = np.zeros_like(vp); wovvv[np.ix_(occ, vir, vir, vir)] = vp[np.ix_(occ, vir, vir, vir)]
    fit(contrib_block(fp, [occ, vir], 1), [("Fov", fov)], "Ms1 = [S_ip, Fov]")
    fit(contrib_block(vp, [occ, occ, occ, vir], 2), [("Wooov", wooov)], "Ms2a = [S_ip, Wooov]")
    fit(contrib_block(vp, [occ, vir, vir, vir], 2), [("Wovvv", wovvv)], "Ms2b = [S_ip, Wovvv]")


if __name__ == "__main__":
    main()
