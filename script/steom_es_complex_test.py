#!/usr/bin/env python3
"""KEY TEST: does the {e^S} (implicit-triples) correction turn the COMPLEX
triplet STEOM eigenvalues REAL?

H2O sto-3g FC1 TRIPLET analytic G (the GANSU-ported routes) has complex roots
(max|Im|~0.78 eV). This probe compares max|Im(eig)| of the TRIPLET G for:
  (1) analytic plain G   = build_g_canonical_full (what GANSU C++ computes)
  (2) determinant plain G = expm(+S) Hbar expm(-S)      (2nd-transform, all orders in expm)
  (3) determinant {e^S} G = (2) - 1/2 (K Hbar + Hbar K) (normal-ordered exp, the TRUE STEOM)
If (3) is REAL while (1)/(2) are complex, implicit triples FIX the complex-root
robustness -> justifies the C++ port. If (3) is still complex, they don't.

  wsl OMP_NUM_THREADS=1 python3 script/steom_es_complex_test.py [ncore] [xyz]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")

import steom_es_oracle as O
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full

Ha = 27.211386245988


def maxim(G):
    w = np.linalg.eigvals(G)
    return np.max(np.abs(w.imag)) * Ha, int(np.sum(np.abs(w.imag) * Ha > 1e-4)), \
        np.sort(w.real)[:4] * Ha


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    xyz = sys.argv[2] if len(sys.argv) > 2 else "xyz/H2O.xyz"
    basis = "sto-3g"

    # ---- (1) analytic plain G (GANSU routes) ----
    d = C.load(xyz, basis, ncore)
    nocc, nvir, dim = d["nocc"], d["nvir"], d["dim"]
    _, g_phph, g_phhp, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    Foo, Fvv = C.build_feff(d)
    GtA = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    GtA[r, c] = fdg - g_phph[a, j, b, i]

    # ---- (2)/(3) determinant plain + {e^S} G ----
    data = O.get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    dets, index, HbarN = O.build_sector(data, data["nelec"])
    hf = O.hf_det(data); iHF = index[hf]; E_N = HbarN[iHF, iHF]
    sIP = O.solve_ip(data, E_N); sEA = O.solve_ea(data)
    S = O.build_S(data, dets, index, sIP, sEA)
    terms = O.s_terms(data, sIP, sEA)
    K = O.build_K(data, dets, index, terms)
    Gp_full = expm(S) @ HbarN @ expm(-S)
    GeS_full = Gp_full - 0.5 * (K @ HbarN + HbarN @ K)
    _, GtP = O.project_1h1p(data, dets, index, Gp_full)
    _, GtE = O.project_1h1p(data, dets, index, GeS_full)
    GtP = GtP - E_N * np.eye(GtP.shape[0])
    GtE = GtE - E_N * np.eye(GtE.shape[0])

    print(f"\n== {xyz} {basis} ncore={ncore}  TRIPLET G complex test ==")
    for name, G in [("(1) analytic plain (GANSU C++)", GtA),
                    ("(2) determinant plain (expm)  ", GtP),
                    ("(3) determinant {e^S} (TRUE)  ", GtE)]:
        mi, nc, lo = maxim(G)
        print(f"  {name}: max|Im|={mi:7.3f}eV  n_cplx={nc:2d}  Re_lowest4={np.round(lo,3)}")
    print("\n  -> if (3) real but (1)/(2) complex, implicit triples FIX the robustness.")


if __name__ == "__main__":
    main()
