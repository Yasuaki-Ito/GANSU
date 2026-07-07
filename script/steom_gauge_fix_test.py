#!/usr/bin/env python3
"""Hypothesis: building the oracle route target through GANSU's s_ip (sN =
build_normalized_s, via build_sip_recon) instead of the oracle's own solve_ip
amplitudes puts target+candidates+C++ in ONE s-gauge, closing the 7% g_phhp gap.

Compare the known dressed g_phhp IP route (coeff 1,1) residual with the target
built from (a) oracle sip_sp gauge  vs  (b) GANSU sN gauge.

Run: wsl python3 script/steom_gauge_fix_test.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD
import steom_cas_verify as V
from pyscf_steom_feff_reference import build_normalized_s
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip,
                                 build_S, hf_det, project_1h1p, occ_so, vir_so)


def phhp_ip_route(data, sip_sp_dict):
    """g_phhp(=0.5(Gs-Gt),F-free) linear-s_ip route [b,j,i,a] for a given spatial
    s_ip dict {m:[i,k,c]} (built into SO via build_sip_recon)."""
    nocc, nvir = data["nocc"], data["nvir"]; nact = data["nact"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    SIP = IPD.build_sip_recon(sip_sp_dict, data)
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
                        g[b, j, i, a] = 0.5 * (GsD[r, c] if False else (GsD[r, j*nvir+b] - GtD[r, j*nvir+b]))
        return g
    return gph(SIP_clean) - gph(zIP)


def known_formula_resid(data, d, sip_dict, ip_target):
    nocc, nvir = data["nocc"], data["nvir"]
    eri = data["eri"]; o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    eri_ooov = eri[o, o, o, v]     # [k,i,l,b] = (ki|lb)
    eri_ovvv = eri[o, v, v, v]     # [k,b,c,a] = (kb|ca)
    s = np.stack([np.asarray(sip_dict[m]) for m in range(nocc)], 0)  # [m,i,k,c]
    # pt619 RAW projection formula (validated 0.36% H2O):
    #   g_phhp[b,j,i,a] += Σ_kl s[j][l,k,a]·eri_ooov[k,i,l,b] − Σ_kc s[j][k,i,c]·eri_ovvv[k,b,c,a]
    T1 = np.einsum("mlka,kilb->mbia", s, eri_ooov, optimize=True)
    T2 = -np.einsum("mkic,kbca->mbia", s, eri_ovvv, optimize=True)
    P1 = np.zeros_like(ip_target); P2 = np.zeros_like(ip_target)
    for m in range(nocc):
        P1[:, m, :, :] = T1[m]; P2[:, m, :, :] = T2[m]
    tv = ip_target.ravel(); tn = np.linalg.norm(tv)
    A = np.stack([P1.ravel(), P2.ravel()], 1)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    return (np.linalg.norm((P1 + P2).ravel() - tv) / tn,
            co, np.linalg.norm(A @ co - tv) / tn)


def run(label, xyz=None, atom=None, active=None, ncore=0):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore); d = CW.load(xyz, "sto-3g", ncore)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        xyzf = os.path.join(tempfile.gettempdir(), "gf.xyz")
        L = [a.strip() for a in atom.split(";")]; open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"]); E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sip_sp = IPD.extract_sip(solve_ip(data, E_N), data)             # oracle gauge dict
    sN, _ = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                               d["occ_idx"], d["vir_idx"], nocc, nvir)
    sN_dict = {m: np.asarray(sN[m]) for m in range(nocc)}           # GANSU gauge dict
    print(f"\n===== {label} =====")
    for tag, sd in [("oracle sip_sp gauge", sip_sp), ("GANSU sN gauge", sN_dict)]:
        ip = phhp_ip_route(data, sd)
        rf, co, rfree = known_formula_resid(data, d, sd, ip)
        print(f"  [{tag}] ||ip||={np.linalg.norm(ip):.4f}  fixed(1,1)={rf:.3e}  "
              f"free=({co[0]:+.3f},{co[1]:+.3f}) resid={rfree:.3e}")


def main():
    run("H2O FC1", xyz="xyz/H2O.xyz", ncore=1)
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run("hexatriene", atom=at, active=ac)


if __name__ == "__main__":
    main()
