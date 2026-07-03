#!/usr/bin/env python3
"""Reproduce GANSU STEOM singlet/triplet H2O FC1 numbers (log197/log198) from the
Python reference, including the IP 4->3 %singles active-set filter.

GANSU (2026-07-03, H200) with the current IP-EOM Davidson finds a 2h1p satellite
(28.63 eV, %singles 0.001) instead of the deep-valence o0 principal IP root, so
the %singles filter drops that NTO -> Ŝ^IP built from 3 roots (X(MI) 3x3). This
script shows the Python reference with the SAME 3-root active set reproduces the
GANSU eigenvalues, i.e. the shift vs the 2026-06-20 4-root reference values
(singlet 11.773 -> 11.82, triplet 10.25 -> 10.33) is the active-set effect, not
a regression from the triplet-block change.

  GANSU log197 (singlet): 0.43446605 / 0.49911594 Ha = 11.8224 / 13.5816 eV
  GANSU log198 (triplet): 0.37976759 / 0.47020918 Ha = 10.3340 / 12.7950 eV

  wsl python3 script/steom_triplet_activeset_check.py
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
np.set_printoptions(precision=4, suppress=True, linewidth=170)


def spectrum(d):
    """Assemble singlet & triplet G from the shipped build_g_canonical_full
    (all corrected routes) + F_eff, exactly as GANSU build_W_eff_and_G:
      Gs = Fdiag + 2 g_phhp - g_phph,  Gt = Fdiag - g_phph."""
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
    es = np.sort(np.linalg.eigvals(Gs).real)
    et = np.sort(np.linalg.eigvals(Gt).real)
    return es, et


def drop_ip_roots(d, drop_occ):
    """Return a copy of d with the IP roots ionizing the given occ orbitals
    removed from the Ŝ^IP active set (mimics GANSU's %singles NTO filter)."""
    keep = [k for k, o in enumerate(d["occ_idx"]) if o not in drop_occ]
    d2 = dict(d)
    d2["r1_ip"] = [d["r1_ip"][k] for k in keep]
    d2["r2_ip"] = [d["r2_ip"][k] for k in keep]
    d2["occ_idx"] = [d["occ_idx"][k] for k in keep]
    return d2


def main():
    xyz = sys.argv[1] if len(sys.argv) > 1 else "xyz/H2O.xyz"
    ncore = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    d = C.load(xyz, "sto-3g", ncore)
    print(f"\n== {xyz} sto-3g ncore={ncore}  nocc={d['nocc']} nvir={d['nvir']} "
          f"occ_idx={list(d['occ_idx'])} vir_idx={list(d['vir_idx'])} ==\n")

    es, et = spectrum(d)
    print(f"complete active (IP {len(d['occ_idx'])} roots):")
    print("  singlet (eV):", np.round(es * Ha, 4))
    print("  triplet (eV):", np.round(et * Ha, 4))

    d3 = drop_ip_roots(d, drop_occ={0})
    es3, et3 = spectrum(d3)
    print(f"\nGANSU-matched active (o0 IP root dropped, IP {len(d3['occ_idx'])} roots):")
    print("  singlet (eV):", np.round(es3 * Ha, 4))
    print("  triplet (eV):", np.round(et3 * Ha, 4))
    print("\n  GANSU log197 singlet: [11.8224 13.5816 17.0334 17.0643 23.3032 27.3932]")
    print("  GANSU log198 triplet: [10.3340 12.7950 13.1496 14.8515 17.9619 19.2951]")


if __name__ == "__main__":
    main()
