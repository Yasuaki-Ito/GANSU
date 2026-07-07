#!/usr/bin/env python3
"""OPERATOR-level derivation (avoids the singles-projection ambiguity):
For each block B in {Fov, Wooov, Wvovv}, extract the 2-body normal-ordered coeff
c2 of the commutator [S_ip, op_k(B)] via low sectors (0/1/2).  c2 IS the operator;
the off-diagonal g_phph route = c2 projected, so matching c2 element-wise to
s x B contractions gives the exact formula (no projection needed).

Run: wsl python3 script/steom_gphph_c2match.py
"""
import os, sys, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_gphph_hbar3 as H3
from steom_fockspace_ref import (get_active_data, build_dets, build_S, solve_ip,
                                  build_sector, hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    z = np.load("/tmp/hbar_mbody.npz"); fp, vp = z["fp"], z["vp"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    occ = occ_so(data); vir = vir_so(data)
    sip = np.zeros((nso,)*4)
    for m in SIP_clean: sip[m] = SIP_clean[m]
    u1 = np.einsum("iijb->jb", sip); u2 = np.einsum("jijb->ib", sip)

    # per-sector dets/index/S_ip
    sect = {}
    for ne in (0, 1, 2):
        d, i = build_dets(nso, ne)
        zEA = {so_index(a + nocc, s, nact): np.zeros((nso,)*3) for s in range(2) for a in range(nvir)}
        Sop = build_S(data, d, i, SIP_clean, zEA)
        sect[ne] = (d, i, Sop)

    def comm_c2(block, k):
        """2-body coeff of [S_ip, op_k(block)] via sectors 0,1,2."""
        E0 = None; fco = None
        for ne in (0, 1, 2):
            d, i, Sop = sect[ne]
            Ob = H3.opk_matrix(d, i, block, k)
            C = Sop @ Ob - Ob @ Sop
            if ne == 0:
                E0 = C[0, 0]
            elif ne == 1:
                fco = H3.read_coeff(d, i, C - E0 * np.eye(len(d)), 1, nso)
            else:
                Rem = C - E0 * np.eye(len(d)) - H3.opk_matrix(d, i, fco, 1)
                return H3.read_coeff(d, i, Rem, 2, nso)

    def anti(T): return T - T.transpose(1, 0, 2, 3) - T.transpose(0, 1, 3, 2) + T.transpose(1, 0, 3, 2)

    def match(c2, srcs, blk, label):
        nw = blk.ndim
        c2 = anti(c2); tv = c2.ravel(); tn = np.linalg.norm(tv)
        cand = {}
        for an, amp in srcs:
            aax = list(range(amp.ndim))
            for c in range(1, min(len(aax), nw) + 1):
                for sset in itertools.combinations(range(len(aax)), c):
                    for wset in itertools.permutations(range(nw), c):
                        sl = list("abcd")[:len(aax)]; wl = list("PQRS")[:nw]
                        for pos, (si, wi) in enumerate(zip(sset, wset)):
                            lab = "XYZW"[pos]; sl[si] = lab; wl[wi] = lab
                        free = [x for x in sl if x not in "XYZW"] + [wl[t] for t in range(nw) if t not in wset]
                        if len(free) != 4: continue
                        es = f"{''.join(sl)},{''.join(wl)}->{''.join(free)}"
                        try: T = np.einsum(es, amp, blk, optimize=True)
                        except Exception: continue
                        if T.shape != (nso,)*4: continue
                        Ta = anti(T)
                        if np.linalg.norm(Ta) < 1e-9: continue
                        cand[f"{an}:{es}"] = Ta.ravel()
        names = list(cand)
        if not names: print(f"[{label}] no cand"); return
        A = np.stack([cand[n] for n in names], 1)
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        print(f"\n[{label}] ||c2||={tn:.4f} ncand={len(names)} FULL resid={np.linalg.norm(A@co-tv)/tn:.2e}")
        res = tv.copy(); chosen = []
        for step in range(4):
            best = min(range(len(names)), key=lambda k: np.linalg.norm(res-(A[:,k]@res/(A[:,k]@A[:,k]+1e-30))*A[:,k]))
            if best in chosen: break
            chosen.append(best); cc, *_ = np.linalg.lstsq(A[:,chosen], tv, rcond=None); res = tv-A[:,chosen]@cc
            print(f"  step{step+1}: resid={np.linalg.norm(res)/tn:.2e}  " +
                  "  ".join(f"{c:+.3f}[{names[j]}]" for c,j in zip(cc,chosen)))
            if np.linalg.norm(res)/tn < 5e-3: break

    fovF = np.zeros((nso, nso)); fovF[np.ix_(occ, vir)] = fp[np.ix_(occ, vir)]
    wooov = np.zeros((nso,)*4); wooov[np.ix_(occ, occ, occ, vir)] = vp[np.ix_(occ, occ, occ, vir)]
    wovvv = np.zeros((nso,)*4); wovvv[np.ix_(occ, vir, vir, vir)] = vp[np.ix_(occ, vir, vir, vir)]
    for name, blk, k in [("Fov", fovF, 1), ("Wooov", wooov, 2), ("Wvovv", wovvv, 2)]:
        c2 = comm_c2(blk, k)
        srcs = [("s", sip), ("u1", u1), ("u2", u2)]
        match(c2, srcs, blk, f"[S_ip, {name}] c2")


if __name__ == "__main__":
    main()
