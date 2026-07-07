#!/usr/bin/env python3
"""Discriminating test: does the KNOWN dressed g_phhp IP route (memory pt619,
coeff +1,+1) reproduce the ORACLE g_phhp IP route across systems?
  g_phhp[b,j,i,a] += Σ_kl s_ip[j][l,k,a] Wooov[k,l,i,b] - Σ_kc s_ip[j][k,i,c] Wvovv[a,k,c,b]
If resid is small+consistent on all systems -> CW.load bar_h matches the oracle
dressing -> the g_phph joint-fit failure means MISSING candidates (not bar_h bug).
If resid is large -> bar_h/oracle mismatch is the real blocker.

Also prints oracle g_phhp base vs dressed Wovvo[k,c,b,j] (EE-base) for reference.

Run: wsl python3 script/steom_gphhp_check.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_cas_verify as V
import steom_cfour_weff as CW
import steom_ea_spinadapt as EA
import steom_ip_route_derive as IPD
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p, occ_so, vir_so)


def oracle_phhp_routes(data):
    """spatial clean-gauge g_phhp(=0.5(Gs-Gt),F-free) base + IP route [b,j,i,a]."""
    nocc, nvir = data["nocc"], data["nvir"]; nact = data["nact"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zIP = {m: np.zeros((nso, nso, nso)) for m in occ_so(data)}
    zEA = {e: np.zeros((nso, nso, nso)) for e in vir_so(data)}

    def gph(SIPx):
        Sm = build_S(data, dets, index, SIPx, zEA)
        G = expm(Sm) @ Hbar @ expm(-Sm)
        Gs, Gt = project_1h1p(data, dets, index, G)
        GsD = Gs - E_N * np.eye(nocc * nvir); GtD = Gt - E_N * np.eye(nocc * nvir)
        g = np.zeros((nvir, nocc, nocc, nvir))
        for i in range(nocc):
            for a in range(nvir):
                r = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        c = j * nvir + b
                        g[b, j, i, a] = 0.5 * (GsD[r, c] - GtD[r, c])
        return g
    base = gph(zIP); ip = gph(SIP_clean) - base
    return base, ip, sip_sp


def run(label, xyz=None, atom=None, active=None, ncore=0):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore); d = CW.load(xyz, "sto-3g", ncore)
    elif active is not None:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        xyzf = os.path.join(tempfile.gettempdir(), "gc.xyz")
        L = [a.strip() for a in atom.split(";")]; open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
        xyzf = os.path.join(tempfile.gettempdir(), "gc.xyz")
        L = [a.strip() for a in atom.split(";")]; open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", ncore, atom=atom)
    nocc, nvir = data["nocc"], data["nvir"]
    base, ip, sip_sp = oracle_phhp_routes(data)
    bar = d["bar"]; Wooov = bar["Wooov"]; Wvovv = bar["Wvovv"]; Wovvo = bar["Wovvo"]
    s = np.stack([sip_sp[m] for m in range(nocc)], 0)   # [m,i,k,c]
    # known dressed g_phhp IP route (memory pt619), root m
    T1 = np.einsum("mlka,klib->mbia", s, Wooov, optimize=True)   # +Σ_kl s[l,k,a] Wooov[k,l,i,b]
    T2 = -np.einsum("mkic,akcb->mbia", s, Wvovv, optimize=True)  # -Σ_kc s[k,i,c] Wvovv[a,k,c,b]
    # place per-root m into [b,m,i,a] (route nonzero at k=m)
    # place per-root m into [b,j,i,a] (route nonzero at j=root=m)
    P1 = np.zeros_like(ip); P2 = np.zeros_like(ip)
    for m in range(nocc):
        P1[:, m, :, :] = T1[m]; P2[:, m, :, :] = T2[m]
    tv = ip.ravel(); tn = np.linalg.norm(tv)
    r_fixed = np.linalg.norm((P1 + P2).ravel() - tv) / tn
    A = np.stack([P1.ravel(), P2.ravel()], 1)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    r_free = np.linalg.norm(A @ co - tv) / tn
    print(f"  [{label}] ||g_phhp IP oracle||={tn:.4f}  fixed(1,1) resid={r_fixed:.3e}  "
          f"free=({co[0]:+.3f},{co[1]:+.3f}) resid={r_free:.3e}")


def main():
    run("H2O FC1 (4,2)", xyz="xyz/H2O.xyz", ncore=1)
    run("H2O full (5,2)", xyz="xyz/H2O.xyz", ncore=0)
    run("H2O distort (4,2)", atom="O 0 0 0; H 0.95 0.30 0.10; H -0.35 0.88 -0.15", ncore=1)
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run("hexatriene (3,3)", atom=at, active=ac)


if __name__ == "__main__":
    main()
