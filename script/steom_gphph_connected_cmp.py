#!/usr/bin/env python3
"""Compare the KNOWN connected u_amci (Nooijen Eq.56, g_phph IP route) against the
PROJECTION oracle Ms route (det target), IN THE SAME s-gauge (sip_sp).  If the
difference is small or matches the bilinear s.t2.eri correction, we get g_phph =
connected + (simple correction) instead of a from-scratch SO Wick derivation.

connected u_amci[a,m,c,i] =
  -Fov[k,c] s[i,k,a] + Wvovv[a,l,c,d] s~[i,l,d] - Wvovv[a,l,d,c] s[i,l,d]
  + Wooov[k,l,i,c] s[k,l,a]     (s=sip_sp[m][i,k,c], s~=2s-s.T(holes))

Run: wsl python3 script/steom_gphph_connected_cmp.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as CW
import steom_gphph_spatial_dressed as SD
from steom_fockspace_ref import get_active_data


def connected_uamci(sip_sp, bar_h, nocc, nvir):
    Fov = bar_h["Fov"]; Wvovv = bar_h["Wvovv"]; Wooov = bar_h["Wooov"]
    out = np.zeros((nvir, nocc, nvir, nocc))   # [a,m,c,i]
    for m in range(nocc):
        s = sip_sp[m]                          # [i,k,c]
        st = 2.0 * s - s.transpose(1, 0, 2)    # s~ swap holes i,k
        aci = (-np.einsum("kc,ika->aci", Fov, s)
               + np.einsum("alcd,ild->aci", Wvovv, st)
               - np.einsum("aldc,ild->aci", Wvovv, s)
               + np.einsum("klic,kla->aci", Wooov, s))
        out[:, m, :, :] = aci
    return out


def run(label, xyz=None, atom=None, ncore=0):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore); d = CW.load(xyz, "sto-3g", ncore)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
        xyzf = os.path.join(tempfile.gettempdir(), "cc.xyz")
        L = [a.strip() for a in atom.split(";")]; open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", ncore, atom=atom)
    nocc, nvir = data["nocc"], data["nvir"]
    base, opp_IP, sip_sp = SD.oracle_gphph_routes(data, atom, None)  # projection target [a,m,c,i]
    sipd = {m: sip_sp[m] for m in range(nocc)}
    conn = connected_uamci(sipd, d["bar"], nocc, nvir)
    tn = np.linalg.norm(opp_IP)
    # best scale of connected vs projection + residual = projection - connected
    num = np.vdot(conn, opp_IP); den = np.vdot(conn, conn) + 1e-30; sc = num / den
    diff = opp_IP - conn
    diff_sc = opp_IP - sc * conn
    print(f"  [{label}] ||proj||={tn:.4f} ||conn||={np.linalg.norm(conn):.4f}  "
          f"||proj-conn||={np.linalg.norm(diff):.4f} ({np.linalg.norm(diff)/tn:.2%})  "
          f"best-scale={sc:.3f} ||resid||={np.linalg.norm(diff_sc)/tn:.2%}")
    return diff, tn


def main():
    print("connected u_amci (Nooijen Eq.56) vs PROJECTION g_phph IP route (same sip_sp gauge):")
    run("H2O distort FC1", atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=1)
    run("H2O distort full", atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=0)
    run("NH3-C1 FC1", atom="N 0 0 0; H 0.95 0.05 0.30; H -0.45 0.83 0.28; H -0.52 -0.78 0.35", ncore=1)


if __name__ == "__main__":
    main()
