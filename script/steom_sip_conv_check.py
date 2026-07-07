#!/usr/bin/env python3
"""Verify oracle clean-gauge s_ip (build_sip_recon/extract_sip) == GANSU
build_normalized_s s_ip (the s used by build_g_canonical_full).  pt436 verified
this for s_EA (machine-exact); s_IP was never checked.  If they differ, that is
the root cause of the ~10% g_phhp residual AND the g_phph joint-fit failure.

Run: wsl python3 script/steom_sip_conv_check.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD
import steom_cas_verify as V
from pyscf_steom_feff_reference import build_normalized_s
from steom_fockspace_ref import get_active_data, build_sector, solve_ip, hf_det


def check(label, xyz=None, atom=None, active=None, ncore=0):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
        d = CW.load(xyz, "sto-3g", ncore)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        xyzf = os.path.join(tempfile.gettempdir(), "sc.xyz")
        L = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sip_sp = IPD.extract_sip(sIP, data)          # oracle clean [m][i,k,c]
    sN, _ = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                               d["occ_idx"], d["vir_idx"], nocc, nvir)
    nao = len(d["occ_idx"])
    print(f"  [{label}] nocc={nocc} nvir={nvir}  n_act_occ={nao}  len(sN)={len(sN)}")
    for m in range(min(nocc, len(sN))):
        a = sip_sp[m]; b = np.asarray(sN[m])
        if a.shape != b.shape:
            print(f"    root{m}: SHAPE {a.shape} vs {b.shape}"); continue
        num = np.vdot(b, a); den = np.vdot(b, b) + 1e-30; sc = num / den
        rel = np.linalg.norm(a - sc * b) / (np.linalg.norm(a) + 1e-30)
        print(f"    root{m}: best-scale={sc:+.4f} rel-resid={rel:.2e}  "
              f"||sip_sp||={np.linalg.norm(a):.4f} ||sN||={np.linalg.norm(b):.4f}")


def main():
    check("H2O FC1", xyz="xyz/H2O.xyz", ncore=1)
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    check("hexatriene", atom=at, active=ac)


if __name__ == "__main__":
    main()
