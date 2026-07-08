#!/usr/bin/env python3
"""FRANKEN-G mechanism test (step 0 of the diag/semi projection-consistency fix).

The C++ projection fix (GANSU_STEOM_GPHPH_PROJECTION=1) replaces only the
off-diagonal (i!=j & a!=b) elements of g_phph/g_phhp with the oracle-exact
projection base+route, while the diag/semi-diag elements stay shipped
(F_eff + connected, ZEROED routes included).  naphthalene cc-pVDZ shows the fix
repairs Lb (+1.13 -> +0.40 vs ORCA) but breaks La (-0.09 -> -0.48).
Hypothesis (franken-G): the shipped off-diag and shipped diag/semi errors
partially cancel on single-config roots; mixing EXACT off-diag with shipped
diag/semi breaks that cancellation.

Test on det-oracle systems, comparing per-root (composition-followed, PR-tagged):
  (a) shipped      : connected G (all classes analytic/shipped)     [= GANSU base]
  (b) franken      : shipped diag/semi + oracle off-diag            [= current C++ fix]
  (d) anti-franken : oracle diag/semi + shipped off-diag            [control]
  (c) full oracle  : det GsD                                        [truth]
Expected: (b) improves config-mixed roots (PR>1.5) but WORSENS single-config
roots vs (a); only full replacement fixes both.

Run:  wsl python3 script/steom_gphph_franken.py        (H2O FC1 + hexatriene)
      wsl python3 script/steom_gphph_franken.py h2o    (H2O only, fast)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                 solve_ea, hf_det, project_1h1p)
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_cas_verify import polyene, detect_pi

Ha = 27.211386245988


def _eig(G):
    w, vr = np.linalg.eig(G)
    o = np.argsort(w.real)
    return w[o].real * Ha, vr[:, o]


def _pr(v):
    p = np.abs(v) ** 2
    s = p.sum()
    return float(1.0 / np.sum((p / s) ** 2)) if s > 0 else 0.0


def _follow(vref, V):
    ov = np.abs(vref.conj() @ V) / (np.linalg.norm(vref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov))


def class_masks(nocc, nvir):
    """boolean (dim,dim) masks: off (i!=j & a!=b), rest (semi+diag)."""
    dim = nocc * nvir
    off = np.zeros((dim, dim), dtype=bool)
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    if i != j and a != b:
                        off[r, j * nvir + b] = True
    return off


def run(tag, xyz=None, atom=None, basis="sto-3g", ncore=None, active=None):
    print(f"\n================ {tag} ================")
    # ---- det oracle ----
    data = get_active_data(xyz=xyz, basis=basis, ncore=(ncore if ncore is not None else 2),
                           atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dim = nocc * nvir
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sEA = solve_ea(data)
    S = build_S(data, dets, index, sIP, sEA)
    G = expm(S) @ Hbar @ expm(-S)
    Gs, _ = project_1h1p(data, dets, index, G)
    GsD = Gs - E_N * np.eye(dim)

    # ---- shipped connected G (analytic, same active space) ----
    if atom is not None:
        xyzf = os.path.join(tempfile.gettempdir(), "franken.xyz")
        lines = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, basis, 0, atom=atom, active=active)
    else:
        d = CW.load(xyz, basis, ncore)
    Ga, *_ = build_g_canonical_full(d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"],
                                    d["r1_ea"], d["occ_idx"], d["vir_idx"],
                                    d["nocc"], d["nvir"])
    assert d["nocc"] == nocc and d["nvir"] == nvir

    off = class_masks(nocc, nvir)
    # variants
    G_frank = Ga.copy();  G_frank[off]  = GsD[off]    # shipped diag/semi + oracle off-diag
    G_anti  = GsD.copy(); G_anti[off]   = Ga[off]     # oracle diag/semi + shipped off-diag
    variants = [("shipped(conn)", Ga), ("franken(C++ fix)", G_frank),
                ("anti-franken", G_anti)]

    e_or, V_or = _eig(GsD)
    print(f"nocc={nocc} nvir={nvir} dim={dim}   ||GsD off||={np.linalg.norm(GsD[off]):.4f} "
          f"||Ga off||={np.linalg.norm(Ga[off]):.4f} ||diff off||={np.linalg.norm((Ga-GsD)[off]):.4f} "
          f"||diff rest||={np.linalg.norm((Ga-GsD)[~off]):.4f}")
    res = {nm: _eig(Gx) for nm, Gx in variants}
    hdr = " ".join(f"{nm:>17}" for nm, _ in variants)
    print(f"  {'root':>4}  {'PR':>5} {'oracle':>8} | " + hdr + "   (signed error vs oracle, eV)")
    nshow = min(dim, 10)
    for k in range(nshow):
        vo = V_or[:, k]
        pr = _pr(vo)
        mix = "*" if pr >= 1.5 else " "
        cells = []
        for nm, _ in variants:
            e_x, V_x = res[nm]
            jx = _follow(vo, V_x)
            cells.append(f"{e_x[jx]:8.3f} ({e_x[jx]-e_or[k]:+.3f})")
        print(f"  {k:>4}{mix} {pr:5.2f} {e_or[k]:8.3f} | " + " ".join(cells))
    # summary: worst error on single-config vs mixed roots
    for nm, _ in variants:
        e_x, V_x = res[nm]
        err_s, err_m = [], []
        for k in range(dim):
            vo = V_or[:, k]
            jx = _follow(vo, V_x)
            (err_m if _pr(vo) >= 1.5 else err_s).append(abs(e_x[jx] - e_or[k]))
        print(f"  [{nm:>17}] max|err| single-config={max(err_s) if err_s else 0:.3f} eV"
              f"   config-mixed={max(err_m) if err_m else 0:.3f} eV")


def main():
    args = sys.argv[1:]
    run("H2O FC1 sto-3g", xyz="xyz/H2O.xyz", ncore=1)
    if "h2o" not in args:
        atom = polyene(6, distort=0.0)
        active, _ = detect_pi(atom, "sto-3g", 3, 3)
        run("hexatriene pi-CAS(6,6) sto-3g", atom=atom, active=active)


if __name__ == "__main__":
    main()
