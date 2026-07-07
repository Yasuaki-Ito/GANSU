#!/usr/bin/env python3
"""g_phph IP route (projection, dressed): fit to a SMALL physically-motivated
candidate set (the connected u_amci terms {Fov,Wvovv,Wooov} + exchange partners
{Wovov,Wovvo}), requiring coefficients CONSISTENT across H2O (nocc4,nvir2) and
hexatriene (3,3).  A consistent coeff = real term; an inconsistent one = collinear
artifact.  Target = det-oracle clean-gauge g_phph(=-Gt,F-free) linear-s_ip route.

Run: wsl python3 script/steom_gphph_physical.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_cas_verify as V
import steom_cfour_weff as CW
from steom_fockspace_ref import get_active_data
import steom_gphph_spatial_dressed as SD


def candidates(sip, bar_h, nocc, nvir):
    """Return dict name-> tensor [m,a,c,i] (per-root m).  s = s_ip[m][i,k,c]."""
    s = sip                                   # [m,i,k,c]
    st = 2 * s - s.transpose(0, 2, 1, 3)      # s̃: swap the two holes (i,k)
    Fov = bar_h["Fov"]        # [k,c]
    Wvovv = bar_h["Wvovv"]    # [a,l,c,d]
    Wooov = bar_h["Wooov"]    # [k,l,i,d]
    Wovov = bar_h["Wovov"]    # [k,b,i,d] = (ki|bd)
    Wovvo = bar_h["Wovvo"]    # [k,b,c,j] = (kc|bj)
    C = {}
    # --- Fov (1-body) ---  output [a,c,i]
    C["Fov.s"]  = np.einsum("kc,mika->maci", Fov, s)      # -Σ_k Fov[k,c] s[i,k,a]
    C["Fov.st"] = np.einsum("kc,mika->maci", Fov, st)
    # --- Wvovv terms (as in u_amci) ---
    C["Wvovv.st_alcd_ild"] = np.einsum("alcd,mild->maci", Wvovv, st)   # +Wvovv[a,l,c,d] s̃[i,l,d]
    C["Wvovv.s_aldc_ild"]  = np.einsum("aldc,mild->maci", Wvovv, s)    # -Wvovv[a,l,d,c] s[i,l,d]
    C["Wvovv.s_alcd_ild"]  = np.einsum("alcd,mild->maci", Wvovv, s)
    # --- Wooov terms ---  Σ_{jl} Wooov[j,l,i,a] s[j,l,c]-style -> [a,c,i]
    C["Wooov.s_klic_kla"]  = np.einsum("klic,mkla->maci", Wooov, s)    # (as u_amci Wovoo->Wooov fix)
    C["Wooov.st_klic_kla"] = np.einsum("klic,mkla->maci", Wooov, st)
    C["Wooov.s_jlia_jlc"]  = np.einsum("jlia,mjlc->maci", Wooov, s)    # exchange-ish (a from Wooov vir)
    # --- Wovov (exchange ph) partners ---  [k,b,i,d]=(ki|bd)
    C["Wovov.s_kaid_kdc?"] = np.einsum("kaid,mkdc->maci", Wovov, s)    # a=b index of Wovov
    C["Wovov.s_kcid_kda"]  = np.einsum("kcid,mkda->maci", Wovov, s)
    C["Wovov.st_kaid_kdc"] = np.einsum("kaid,mkdc->maci", Wovov, st)
    # --- Wovvo (direct ph) partners --- [k,b,c,j]=(kc|bj)
    C["Wovvo.s_kacj_jic?"] = np.einsum("kacj,mjic->maci", Wovvo, s)
    C["Wovvo.s_kcaj_?"]    = np.einsum("kacj,mkic->maci", Wovvo, s)
    return C


def fit(sip, bar_h, target, nocc, nvir):
    C = candidates(sip, bar_h, nocc, nvir)
    names = list(C)
    A = np.stack([C[n].reshape(-1) for n in names], 1)
    tv = target.transpose(1, 0, 2, 3).reshape(-1)   # [m,a,c,i]
    tn = np.linalg.norm(tv)
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    resid = np.linalg.norm(A @ co - tv) / tn
    return names, co, resid, {n: np.linalg.norm(C[n]) for n in names}


def run(atom=None, xyz=None, active=None, ncore=0, label=""):
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
        d = CW.load(xyz, "sto-3g", ncore)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        xyzf = os.path.join(tempfile.gettempdir(), "g.xyz")
        lines = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    base, opp_IP, sip_sp = SD.oracle_gphph_routes(data, atom, active)
    sip = np.stack([sip_sp[m] for m in range(nocc)], 0)
    names, co, resid, norms = fit(sip, d["bar"], opp_IP, nocc, nvir)
    print(f"\n===== {label} =====  ||tgt||={np.linalg.norm(opp_IP):.4f}  FULL-fit resid={resid:.3e}")
    return names, co, norms, resid


def main():
    r1 = run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    r2 = run(atom=at, active=ac, label="hexatriene pi-CAS(6,6)")
    names = r1[0]
    print(f"\n{'candidate':28s} {'H2O coeff':>12s} {'hex coeff':>12s}  consistent?")
    for i, n in enumerate(names):
        c1, c2 = r1[1][i], r2[1][i]
        cons = "YES" if abs(c1 - c2) < 0.05 and abs(c1) > 0.03 else ("~0" if abs(c1) < 0.03 and abs(c2) < 0.03 else "no")
        print(f"{n:28s} {c1:>12.4f} {c2:>12.4f}  {cons}")


if __name__ == "__main__":
    main()
