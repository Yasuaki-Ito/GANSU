#!/usr/bin/env python3
"""Probe whether the ANALYTIC STEOM G (build_g_canonical_full routes = what GANSU
C++ computes) has COMPLEX eigenvalues for small molecules, and whether the {e^S}
implicit-triples correction (from steom_es_oracle) would change them.

The complex-root / near-defective-G robustness problem appears in GANSU C++ for
medium molecules (2-octanone). To test if implicit triples fix it, we first need
a SMALL molecule whose analytic G already goes complex. This probe scans a few
small xyz geometries and reports max|Im(eig)| of the singlet & triplet G.

  wsl OMP_NUM_THREADS=1 python3 script/steom_complex_probe.py [xyz ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full

Ha = 27.211386245988


def analytic_G(d):
    """Assemble singlet & triplet analytic G (the shipped GANSU routes) and
    return their complex eigenvalues."""
    nocc, nvir, dim = d["nocc"], d["nvir"], d["dim"]
    _, g_phph, g_phhp, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    Foo, Fvv = C.build_feff(d)
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    Gs[r, c] = fdg + 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                    Gt[r, c] = fdg - g_phph[a, j, b, i]
    ws = np.linalg.eigvals(Gs)
    wt = np.linalg.eigvals(Gt)
    return ws, wt


def report(xyz, basis, ncore):
    try:
        d = C.load(xyz, basis, ncore)
    except Exception as e:
        print(f"  {xyz:32s} LOAD FAILED: {e}")
        return
    ws, wt = analytic_G(d)
    # lowest few by real part
    ss = ws[np.argsort(ws.real)]
    st = wt[np.argsort(wt.real)]
    max_im_s = np.max(np.abs(ws.imag)) * Ha
    max_im_t = np.max(np.abs(wt.imag)) * Ha
    n_cplx_s = int(np.sum(np.abs(ws.imag) * Ha > 1e-4))
    n_cplx_t = int(np.sum(np.abs(wt.imag) * Ha > 1e-4))
    print(f"  {os.path.basename(xyz):28s} dim={d['dim']:4d}  "
          f"SING max|Im|={max_im_s:7.3f}eV n_cplx={n_cplx_s:3d}  "
          f"TRIP max|Im|={max_im_t:7.3f}eV n_cplx={n_cplx_t:3d}")
    print(f"      singlet Re(eV) lowest5: {np.round(ss.real[:5]*Ha, 3)}")
    if n_cplx_s > 0:
        # show the complex ones
        cpx = ss[np.abs(ss.imag) * Ha > 1e-4]
        print(f"      singlet COMPLEX roots (eV): "
              f"{[f'{z.real*Ha:.2f}{z.imag*Ha:+.2f}j' for z in cpx[:6]]}")


def main():
    basis = "sto-3g"
    ncore = 1
    xyzs = sys.argv[1:] if len(sys.argv) > 1 else [
        "xyz/H2O.xyz", "xyz/Formaldehyde.xyz", "xyz/2-butanone.xyz",
        "xyz/Acetone.xyz", "xyz/pyridine.xyz",
    ]
    print(f"\n== analytic STEOM G complex-eigenvalue probe ({basis}, ncore={ncore}) ==")
    for x in xyzs:
        report(x, basis, ncore)


if __name__ == "__main__":
    main()
