#!/usr/bin/env python3
"""Extract the exact 2nd-order {e^S} g_phhp cross+same-type residual and characterise
its diagonal vs off-diagonal structure. This is the LAST undone W^eff piece.

residual = g_phhp_true({e^S}) - (base + IP route) - EA route(pt41)
         = cross (S^IP·S^EA) + same-type (S^IP·S^IP, S^EA·S^EA), all 2nd-order.

New lead (2026-07-06): naphthalene Lb (config-mixed) overshoot is governed by the
OFF-DIAGONAL (i!=j & a!=b) of g_phhp, which n->pi* (H2O) can't test. So we look at
the residual's off-diagonal block specifically.

Run:  wsl python3 script/steom_cross_extract.py 1
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from steom_es_oracle import analyze
from pyscf_steom_feff_reference import build_normalized_s

Ha = 27.211386245988
np.set_printoptions(precision=4, suppress=True, linewidth=170)


def ea_route_pt41(bar, r2_ip, r2_ea, r1_ip, r1_ea, occ_idx, vir_idx, nocc, nvir):
    """pt41 SO-derived g_phhp EA route (root=ket-particle b=vir_idx[e])."""
    Wovoo = bar["Wovoo"]; Wvovv = bar["Wvovv"]
    _, s_EA = build_normalized_s(r2_ip, r2_ea, r1_ip, r1_ea, occ_idx, vir_idx, nocc, nvir)
    g = np.zeros((nvir, nocc, nocc, nvir))   # [b,k,j,c] accessed as [b,j,i,a]
    for e in range(len(s_EA)):
        bt = vir_idx[e]
        s = s_EA[e]                               # [i,a,c] (occ,vir,vir)
        block = (0.5 * np.einsum("lcji,lac->jia", Wovoo, s)
                 + 0.5 * np.einsum("ajcd,icd->jia", Wvovv, s))   # [j,i,a]
        g[bt] += block
    return g


def diag_offdiag_split(T, nocc, nvir, label):
    """T is g_phhp[b,j,i,a] (accessed physical). Split by (i==j) & (a==b?) — but
    g_phhp has no b==a natural pairing; use the G off-diagonal notion: the coupling
    that matters is g_phhp[b,j,i,a] with (i,a) != (j,b) i.e. it enters G[ia,jb].
    We report ||T|| and the fraction on 'aligned' (i==j or a==b) vs 'fully-off'
    (i!=j AND a!=b) index patterns."""
    n_al = 0.0; n_off = 0.0
    for b in range(nvir):
        for j in range(nocc):
            for i in range(nocc):
                for a in range(nvir):
                    v = T[b, j, i, a]
                    # G row=(i,a) col=(j,b): fully-off = i!=j AND a!=b
                    if (i != j) and (a != b):
                        n_off += v * v
                    else:
                        n_al += v * v
    print(f"  [{label}] ||T||={np.linalg.norm(T):.4f}  aligned(i=j or a=b)={np.sqrt(n_al):.4f}  "
          f"fully-off(i!=j & a!=b)={np.sqrt(n_off):.4f}")


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res, D = analyze(ncore=ncore)
    d = D["d"]; nocc = d["nocc"]; nvir = d["nvir"]
    gphhp_true = D["gphhp_true"]            # {e^S} exact g_phhp[b,j,i,a]

    # base + IP route = the shipped g_phhp (build_g_canonical_full returns g_hp with EA/cross=0)
    from pyscf_steom_feff_reference import build_g_canonical_full
    _, _, g_hp_shipped, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)   # = base + u_bmjc (IP)

    ea = ea_route_pt41(d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                       d["occ_idx"], d["vir_idx"], nocc, nvir)

    print("\n##### g_phhp DECOMPOSITION (ncore=%d) #####" % ncore)
    print(f"  ||g_phhp_true ({{e^S}})|| = {np.linalg.norm(gphhp_true):.4f}")
    print(f"  ||base+IP (shipped)||    = {np.linalg.norm(g_hp_shipped):.4f}")
    print(f"  ||EA route (pt41)||      = {np.linalg.norm(ea):.4f}")

    resid = gphhp_true - g_hp_shipped - ea      # = cross + same-type (2nd order)
    resid_noea = gphhp_true - g_hp_shipped      # = EA + cross + same-type
    print(f"\n  residual = true - base - IP - EA  (= cross + same-type):")
    diag_offdiag_split(resid, nocc, nvir, "cross+sametype")
    print(f"  residual = true - base - IP        (= EA + cross + same-type):")
    diag_offdiag_split(resid_noea, nocc, nvir, "EA+cross+sametype")
    print(f"  base+IP alone vs true:")
    diag_offdiag_split(gphhp_true - g_hp_shipped, nocc, nvir, "true-(base+IP)")

    # How much of the fully-off-diagonal g_phhp does base+IP miss?
    def off_norm(T):
        s = 0.0
        for b in range(nvir):
            for j in range(nocc):
                for i in range(nocc):
                    for a in range(nvir):
                        if (i != j) and (a != b):
                            s += T[b, j, i, a] ** 2
        return np.sqrt(s)
    print(f"\n  OFF-DIAGONAL (i!=j & a!=b) norms:")
    print(f"    true            = {off_norm(gphhp_true):.4f}")
    print(f"    base+IP         = {off_norm(g_hp_shipped):.4f}")
    print(f"    base+IP+EA      = {off_norm(g_hp_shipped + ea):.4f}")
    print(f"    missing (resid) = {off_norm(resid):.4f}")


if __name__ == "__main__":
    main()
