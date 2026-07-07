#!/usr/bin/env python3
"""Nail Ms1 (Fov) & Ms2a (Wooov) contractions by projecting candidates through
the EXACT det_singles projection (op2(T) then det_singles) instead of raw slices.
Candidate T = canonical s/u contraction with the single intermediate block; the
projection supplies the singles antisymmetrization.  Ms2b already = -0.5*one term.

Run: wsl python3 script/steom_gphph_projfit.py
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
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP_clean: sip[m] = SIP_clean[m]
    u1 = np.einsum("iijb->jb", sip); u2 = np.einsum("jijb->ib", sip)

    def route(Op):
        comm = S_ip @ Op - Op @ S_ip
        Ms1, _, _ = det_singles_block(data, dets, index, comm)
        return Ms1.ravel()

    def antisym2(T):
        return T - T.transpose(1, 0, 2, 3) - T.transpose(0, 1, 3, 2) + T.transpose(1, 0, 3, 2)

    def proj(Tcoeff, k):
        """<i^a|op_k(Tcoeff)|j^b> same-spin, as vector.  k=2 coeff antisymmetrized."""
        if k == 2:
            Tcoeff = antisym2(Tcoeff)
        Op = H3.opk_matrix(dets, index, Tcoeff, k)
        Ms, _, _ = det_singles_block(data, dets, index, Op)
        return Ms.ravel()

    def gen_cands(amp_list, W, block_idx):
        """canonical contractions amp x W -> 4-index coeff T (full nso), for op2/op1 projection."""
        cands = {}
        for an, amp in amp_list:
            aax = list(range(amp.ndim)); wax = list(range(W.ndim))
            for c in range(1, min(len(aax), len(wax)) + 1):
                for sset in itertools.combinations(range(len(aax)), c):
                    for wset in itertools.permutations(range(len(wax)), c):
                        sl = list("abcd")[:len(aax)]; wl = list("PQRS")[:len(wax)]
                        for pos, (si, wi) in enumerate(zip(sset, wset)):
                            lab = "XYZW"[pos]; sl[si] = lab; wl[wi] = lab
                        free = [x for x in sl if x not in "XYZW"] + [wl[t] for t in range(len(wax)) if t not in wset]
                        if len(free) != 4: continue
                        es = f"{''.join(sl)},{''.join(wl)}->{''.join(free)}"
                        try: T = np.einsum(es, amp, W, optimize=True)
                        except Exception: continue
                        if T.shape != (nso,)*4 or np.linalg.norm(T) < 1e-9: continue
                        cands[f"{an}:{es}"] = T
        return cands

    def do(target, amp_list, W, k, label):
        cd = gen_cands(amp_list, W, None)
        names = list(cd)
        # project each candidate coeff through op_k + det_singles
        cols = []
        for n in names:
            cols.append(proj(cd[n], 2))   # op2 projection (2-body form)
        A = np.stack(cols, 1); tv = target
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        print(f"\n[{label}] ncand={len(names)} FULL resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.2e}")
        res = tv.copy(); chosen = []
        for step in range(4):
            best = min(range(len(names)), key=lambda i: np.linalg.norm(res-(A[:,i]@res/(A[:,i]@A[:,i]+1e-30))*A[:,i]))
            if best in chosen: break
            chosen.append(best); cc, *_ = np.linalg.lstsq(A[:,chosen], tv, rcond=None); res = tv-A[:,chosen]@cc
            print(f"  step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(tv):.2e}  " +
                  "  ".join(f"{c:+.3f}[{names[j]}]" for c,j in zip(cc,chosen)))
            if np.linalg.norm(res)/np.linalg.norm(tv) < 5e-3: break

    fovF = np.zeros((nso, nso)); fovF[np.ix_(occ, vir)] = fp[np.ix_(occ, vir)]
    wooov = np.zeros((nso,)*4); wooov[np.ix_(occ, occ, occ, vir)] = vp[np.ix_(occ, occ, occ, vir)]
    wovvv = np.zeros((nso,)*4); wovvv[np.ix_(occ, vir, vir, vir)] = vp[np.ix_(occ, vir, vir, vir)]
    # targets (exact route contributions)
    t_fov = route(H3.opk_matrix(dets, index, fovF, 1))
    t_ooov = route(H3.opk_matrix(dets, index, wooov, 2))
    print(f"||Ms1(Fov)||={np.linalg.norm(t_fov):.4f}  ||Ms2a(Wooov)||={np.linalg.norm(t_ooov):.4f}")
    # For Fov (1-body block) the route coeff is 2-body; candidates = s x Fov / u x Fov (produce 4-index)
    do(t_fov, [("s", sip), ("u1", u1), ("u2", u2)], fovF, 2, "Ms1 = [S_ip, Fov]")
    do(t_ooov, [("s", sip), ("u1", u1), ("u2", u2)], wooov, 2, "Ms2a = [S_ip, Wooov]")


if __name__ == "__main__":
    main()
