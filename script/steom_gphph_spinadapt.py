#!/usr/bin/env python3
"""Spin-adapt the SO g_phph linear S_ip V2 route to spatial closed-shell (for C++).
Target = machine-exact SO route (sympy V) sliced to the alpha-alpha block.
Candidates = spatial s_ip[m][i,k,c] contracted with bar_h dressed intermediates
(Fermi-NO), the known {Wooov,Wvovv,Fov,Wovov} structure. Fit -> spatial formula.

Run: wsl python3 script/steom_gphph_spinadapt.py
"""
import os, sys, itertools, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD, steom_cfour_weff as CW
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, hf_det,
                                  so_index, occ_so, vir_so)


def main():
    xyz = "xyz/H2O.xyz"; ncore = 1
    data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
    d = CW.load(xyz, "sto-3g", ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)     # spatial s_ip[m][i,k,c]
    SIP = IPD.build_sip_recon(sip_sp, data)
    vp = np.load("/tmp/hbar_mbody.npz")["vp"]
    sp = np.zeros((nso,)*4)
    for m in occ_so(data):
        sp[m] = SIP[m]
    occ = occ_so(data); vir = vir_so(data)
    # SO Fermi-NO linear S_ip V2 route (sympy V)
    V = (-0.5*np.einsum("IKbi,jIKa->iajb", vp, sp, optimize=True)
         -0.5*np.einsum("aIbA,jIiA->iajb", vp, sp, optimize=True)
         +0.5*np.einsum("aIbA,jiIA->iajb", vp, sp, optimize=True)
         -0.5*np.einsum("IKiA,jIKA,ab->iajb", vp, sp, np.eye(nso), optimize=True))
    oa = [so_index(x, 0, nact) for x in range(nocc)]; va = [so_index(x+nocc, 0, nact) for x in range(nvir)]
    # target: g_phph[a,k,c,i] route layout = V[i,a,j=k(root),b=c] -> [a,k,c,i]
    tgt = np.einsum("iajb->ajbi", V[np.ix_(oa, va, oa, va)])   # [a,k,c,i], k=root=j
    print(f"||spatial g_phph S_ip V2 route target||={np.linalg.norm(tgt):.4f}")

    # spatial amplitudes and intermediates (bar_h = Fermi-NO spatial)
    s = np.stack([sip_sp[m] for m in range(nocc)], 0)   # [m,i,k,c]
    st = 2*s - s.transpose(0, 2, 1, 3)                   # s~ (swap holes)
    bar = d["bar"]; Fov = bar["Fov"]; Wooov = bar["Wooov"]; Wvovv = bar["Wvovv"]; Wovov = bar["Wovov"]
    # candidate contractions producing [a,k=root m,c,i] (root m == occ in full-active)
    C = {}
    def add(T, nm):
        # T is [m,...]; place per-root m at k=m -> build [a,k,c,i]
        if T.shape != (nocc, nvir, nvir, nocc): return
        C[nm] = T.reshape(-1)
    # Wooov-type: s[m] x Wooov -> [a? no vir from Wooov]. Wooov[k,l,i,d]; s[m][i,k,c].
    for es, nm in [("mIKc,IKid->mdci", "Wooov:A"), ("mIKc,KIid->mdci", "Wooov:B"),
                   ("mIKa,IKic->maci", "Wooov:C")]:
        try: add(np.einsum(es.replace("d", "d"), s, Wooov, optimize=True), nm)
        except Exception: pass
    # Wvovv-type: Wvovv[a,l,c,d]; s[m][i,l,d]
    for es, nm in [("aIcd,mIid->maci", "Wvovv:A"), ("aIdc,mIid->maci", "Wvovv:B"),
                   ("aIcd,mIid->maci", "Wvovv:C2"), ("aIcd,miId->maci", "Wvovv:D")]:
        try:
            T = np.einsum(es, Wvovv, s, optimize=True); add(T, nm)
        except Exception: pass
    for es, nm in [("aIcd,mIid->maci", "Wvovv~A")]:
        add(np.einsum("aIcd,mIid->maci", Wvovv, st, optimize=True), "Wvovv:st")
    # Fov: Fov[k,c]; s[m]
    add(np.einsum("Ic,miIa->maci" if False else "Ic,miIa->maci", Fov, s, optimize=True) if False else np.zeros((nocc,nvir,nvir,nocc)), "z")
    # brute enumerate s x {Wooov,Wvovv,Fov,Wovov} contractions -> [m,a,c,i]
    C.clear()
    W = {"Wooov": Wooov, "Wvovv": Wvovv, "Wovov": Wovov, "Fov": Fov}
    o, v = nocc, nvir
    saxk = [(0, o), (1, o), (2, v)]  # s[m] axes i,k,c
    def addall(nm, Wt):
        wsh = Wt.shape
        for c in range(1, min(3, len(wsh))+1):
            for sset in itertools.combinations(range(3), c):
                for wset in itertools.permutations(range(len(wsh)), c):
                    sl = list("abc"); wl = list("PQRS")[:len(wsh)]
                    for pos, (si, wi) in enumerate(zip(sset, wset)):
                        lab = "XYZ"[pos]; sl[si] = lab; wl[wi] = lab
                    free = [x for x in sl if x not in "XYZ"] + [wl[t] for t in range(len(wsh)) if t not in wset]
                    if len(free) != 3: continue
                    es = f"m{''.join(sl)},{''.join(wl)}->m{''.join(free)}"
                    for src, tag in [(s, ""), (st, "~")]:
                        try: T = np.einsum(es, src, Wt, optimize=True)
                        except Exception: continue
                        for perm in itertools.permutations(range(4)):
                            Tp = np.transpose(T, perm)
                            if Tp.shape != (nocc, nvir, nvir, nocc): continue
                            vvv = Tp.reshape(-1)
                            if np.linalg.norm(vvv) > 1e-9: C[f"{nm}{tag}:{es}:p{perm}"] = vvv
    for nm, Wt in W.items():
        if nm == "Fov": continue
        addall(nm, Wt)
    names = list(C); A = np.stack([C[n] for n in names], 1); tv = tgt.reshape(-1)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"spatial fit: ncand={len(names)} FULL resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.3e}")
    res = tv.copy(); ch = []
    for step in range(6):
        best = min(range(len(names)), key=lambda k: np.linalg.norm(res-(A[:,k]@res/(A[:,k]@A[:,k]+1e-30))*A[:,k]))
        if best in ch: break
        ch.append(best); cc, *_ = np.linalg.lstsq(A[:,ch], tv, rcond=None); res = tv-A[:,ch]@cc
        print(f"  step{step+1}: resid={np.linalg.norm(res)/np.linalg.norm(tv):.3e}  " +
              "  ".join(f"{c:+.3f}[{names[k]}]" for c,k in zip(cc,ch)))
        if np.linalg.norm(res)/np.linalg.norm(tv) < 5e-3: break


if __name__ == "__main__":
    main()
