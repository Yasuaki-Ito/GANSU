#!/usr/bin/env python3
"""Derive/validate the ZEROED g_phhp EA + cross s-routes against the det oracle.

g_phhp[b,j,i,a] = 0.5(Gs - Gt) is a convention-independent physical tensor, so the
oracle and GANSU-analytic g_phhp are directly comparable. GANSU keeps base+IP and
ZEROES EA (u_bkje) + cross (u_bmje). This extracts the EXACT EA and cross route
tensors from the oracle (via only-S^EA / only-S^IP transforms, both full and
1st-order-in-s) and compares them to GANSU's candidate analytic forms
(build_u_bkje_canonical, pt41 SO EA). Degeneracy-immune (tensor-level, no eig-following).

Run: wsl python3 script/steom_sroute_derive.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
import steom_cas_verify as V
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_g_canonical_full


def oracle_gphhp(data, dets, index, Hbar, E_N, S, order=None):
    """g_phhp[b,j,i,a] from e^S Hbar e^-S; order=1 -> keep only Hbar+[S,Hbar]."""
    nocc, nvir = data["nocc"], data["nvir"]
    if order == 1:
        G = Hbar + (S @ Hbar - Hbar @ S)
    else:
        G = expm(S) @ Hbar @ expm(-S)
    Gs, Gt = project_1h1p(data, dets, index, G)
    GsD = Gs - E_N * np.eye(nocc * nvir); GtD = Gt - E_N * np.eye(nocc * nvir)
    gph = np.zeros((nvir, nocc, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    gph[b, j, i, a] = 0.5 * (GsD[i*nvir+a, j*nvir+b] - GtD[i*nvir+a, j*nvir+b])
    return gph


def rel(x, y):
    return np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30)


def main():
    n, n_pi = 6, 3
    atom = V.polyene(n)
    active, _ = V.detect_pi(atom, "sto-3g", n_pi, n_pi)
    data = get_active_data(atom=atom, basis="sto-3g", active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}; zEA = {e: np.zeros_like(sEA[e]) for e in sEA}
    S_full = build_S(data, dets, index, sIP, sEA)
    S_ip = build_S(data, dets, index, sIP, zEA)
    S_ea = build_S(data, dets, index, zIP, sEA)
    S_0 = build_S(data, dets, index, zIP, zEA)

    gph_0 = oracle_gphhp(data, dets, index, Hbar, E_N, S_0)
    gph_ea_full = oracle_gphhp(data, dets, index, Hbar, E_N, S_ea)
    gph_ea_1 = oracle_gphhp(data, dets, index, Hbar, E_N, S_ea, order=1)
    gph_ip_full = oracle_gphhp(data, dets, index, Hbar, E_N, S_ip)
    gph_ip_1 = oracle_gphhp(data, dets, index, Hbar, E_N, S_ip, order=1)
    gph_full = oracle_gphhp(data, dets, index, Hbar, E_N, S_full)
    EA_route = gph_ea_full - gph_0
    EA_route_1 = gph_ea_1 - gph_0          # exact linear-in-s_EA part = [S_EA,Hbar]
    IP_route = gph_ip_full - gph_0
    IP_route_1 = gph_ip_1 - gph_0          # exact linear-in-s_IP part = [S_IP,Hbar]
    cross = gph_full - gph_ip_full - gph_ea_full + gph_0
    # verify g_phhp is order-2-exact as a tensor (justifies route-by-order decomposition)
    G2 = Hbar + (S_full@Hbar-Hbar@S_full) + 0.5*(S_full@(S_full@Hbar-Hbar@S_full)
                                                 - (S_full@Hbar-Hbar@S_full)@S_full)
    gph_2nd = oracle_gphhp(data, dets, index, Hbar, E_N, S_full)  # full
    # (full already used; compare full vs explicit 2nd-order partial sum)
    Gs2, Gt2 = project_1h1p(data, dets, index, G2)
    gph_from2 = np.zeros_like(gph_full)
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    gph_from2[b,j,i,a] = 0.5*((Gs2[i*nvir+a,j*nvir+b]-E_N*(i*nvir+a==j*nvir+b))
                                              -(Gt2[i*nvir+a,j*nvir+b]-E_N*(i*nvir+a==j*nvir+b)))
    print(f"  g_phhp order-2-exact check: ||full - 2nd-order|| rel = {rel(gph_from2, gph_full):.3e}")
    print(f"hexatriene pi-CAS(6,6)  nocc={nocc} nvir={nvir}")
    print(f"  ||g_phhp||: base={np.linalg.norm(gph_0):.4f} IP={np.linalg.norm(IP_route):.4f} "
          f"EA(full)={np.linalg.norm(EA_route):.4f} EA(1st)={np.linalg.norm(EA_route_1):.4f} "
          f"cross={np.linalg.norm(cross):.4f}")
    print(f"  EA route: 1st-order vs full rel-diff = {rel(EA_route_1, EA_route):.3e}")

    # ---- GANSU candidate EA routes (build_u_bkje_canonical via env probe) ----
    xyzf = os.path.join(tempfile.gettempdir(), "sr.xyz")
    lines = [a.strip() for a in atom.split(";")]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    # base+IP g_phhp (shipped, EA/cross=0) and the decomposition
    _, _, g_hp_sh, _, _, decomp_sh = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    g_base, g_ip, g_ea_z, g_cr_z = decomp_sh
    # EE-base variant (sanity: EE base must match oracle base ~machine precision)
    os.environ["STEOM_EE_BASE"] = "1"
    _, _, _, _, _, dec_ee = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    os.environ.pop("STEOM_EE_BASE", None)
    g_base_ee = dec_ee[0]
    print(f"\n  GANSU vs oracle (abs norms: oracle base={np.linalg.norm(gph_0):.4f} "
          f"IP={np.linalg.norm(IP_route):.4f} base+IP={np.linalg.norm(gph_0+IP_route):.4f})")
    print(f"    shipped base : rel-diff = {rel(g_base, gph_0):.3e}  (||{np.linalg.norm(g_base):.4f}||)")
    print(f"    EE base      : rel-diff = {rel(g_base_ee, gph_0):.3e}  <== must be ~0")
    print(f"    IP route vs full  : rel-diff = {rel(g_ip, IP_route):.3e}  (GANSU ||{np.linalg.norm(g_ip):.4f}|| oracle-full ||{np.linalg.norm(IP_route):.4f}||)")
    print(f"    IP route vs 1st   : rel-diff = {rel(g_ip, IP_route_1):.3e}  (oracle-1st ||{np.linalg.norm(IP_route_1):.4f}||)  <== GANSU u_bmjc is linear-in-s")
    print(f"    IP 1st vs full    : rel-diff = {rel(IP_route_1, IP_route):.3e}  (higher-order-in-s_IP content)")
    # EA candidates
    for mode in ("1", "2"):
        os.environ["STEOM_TEST_GPHHP_EA"] = mode
        _, _, _, _, _, dec = build_g_canonical_full(
            d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
            d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
        os.environ.pop("STEOM_TEST_GPHHP_EA", None)
        ea_cand = dec[2]
        print(f"    EA cand mode{mode}: ||{np.linalg.norm(ea_cand):.4f}||  "
              f"vs oracle EA(full) rel={rel(ea_cand, EA_route):.3e}  "
              f"vs EA(1st) rel={rel(ea_cand, EA_route_1):.3e}")

    # save oracle targets for a fit stage
    np.savez(os.path.join(tempfile.gettempdir(), "hex_sroutes.npz"),
             EA_full=EA_route, EA_1=EA_route_1, IP=IP_route, cross=cross, base=gph_0)
    print("  saved oracle route tensors to /tmp/hex_sroutes.npz")


if __name__ == "__main__":
    main()
