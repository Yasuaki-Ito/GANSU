#!/usr/bin/env python3
"""Order-by-order BCH decomposition of the plain STEOM transform.
G = e^S H e^-S = sum_n (1/n!) ad_S^n(H). Shows each order's contribution to
singlet[0] and where it terminates. Run: wsl python3 script/steom_order_scan.py 1
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
from scipy.linalg import expm
Ha = 27.211386245988


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=ncore)
    nocc = data["nocc"]; nvir = data["nvir"]
    dets, index, H = build_sector(data, data["nelec"])
    E_N = H[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    S = build_S(data, dets, index, sIP, sEA)
    print(f"== H2O sto-3g FC{ncore}  nocc={nocc} nvir={nvir} dim={len(dets)}")

    def sing0(G):
        Gs, _ = project_1h1p(data, dets, index, G)
        return np.sort(np.linalg.eigvals(Gs).real - E_N)[0] * Ha

    ad = H.copy(); G = H.copy(); fact = 1.0
    print("\n  order  ||term||       partial-sum singlet[0] (eV)")
    print(f"   0     {np.linalg.norm(ad):.4e}   {sing0(G):.4f}")
    for n in range(1, 9):
        ad = S @ ad - ad @ S; fact *= n; term = ad / fact; G = G + term
        print(f"   {n}     {np.linalg.norm(term):.4e}   {sing0(G):.4f}")

    print(f"\n  full expm singlet[0] = {sing0(expm(S) @ H @ expm(-S)):.4f} eV")
    if ncore == 1:
        print("  ORCA STEOM H2O sto-3g FC1 IROOT1 = 11.849 eV")


if __name__ == "__main__":
    main()
