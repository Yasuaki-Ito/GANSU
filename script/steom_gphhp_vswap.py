#!/usr/bin/env python3
"""VSWAP scan (続55): place phhp routes with the two VIRTUAL roles exchanged
(ST-CC2-appendix convention: amplitude-vir -> bra, integral/scattered vir -> ket),
consistent with the EE-base fix which already swapped the BASE virtuals.

Judged on the three ORCA gates + the butadiene pair-2x2 razor
(needed: ghp_diag -0.23, ghp_off -0.43 eV vs unswapped paper).

Run:  wsl python3 script/steom_gphhp_vswap.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from steom_gphhp_gate import restrict, assemble, Ha2eV, ORCA_S
from steom_gphhp_paper import build_paper_gphhp, build_paper_gphph
from steom_hex_gate import light_load, ATOM as ATOM_HEX, ORCA_HEX_S
from steom_buta_gate import ATOM as ATOM_BUTA, ORCA_BUTA_S


def prep():
    gates = {}
    d = C.load("xyz/H2O.xyz", "sto-3g", 1)
    d = restrict(d, occ_keep=(1, 2, 3), vir_keep=(0, 1))
    gates["h2o"] = (d, C.build_feff(d), build_paper_gphph(d, True), ORCA_S, 6)
    d = light_load(ATOM_HEX, ncore=6)
    d = restrict(d, occ_keep=tuple(range(d["nocc"] - 6, d["nocc"])),
                 vir_keep=tuple(range(8)))
    gates["hex"] = (d, C.build_feff(d), build_paper_gphph(d, True), ORCA_HEX_S, 5)
    d = light_load(ATOM_BUTA, ncore=4)
    d = restrict(d, occ_keep=tuple(range(3, d["nocc"])), vir_keep=tuple(range(8)))
    gates["buta"] = (d, C.build_feff(d), build_paper_gphph(d, True), ORCA_BUTA_S, 6)
    return gates


def gate_rms(gate, vswap):
    d, (Foo, Fvv), gph, target, n = gate
    ghp = build_paper_gphhp(d, 0, 0, 1, vswap=vswap)
    Gs, _ = assemble(d, ghp, gph, Foo, Fvv)
    ws = np.sort(np.linalg.eigvals(Gs).real) * Ha2eV
    return float(np.sqrt(np.mean((ws[:n] - np.array(target[:n])) ** 2)))


def buta_2x2(gate, vswap):
    d, (Foo, Fvv), gph, _, _ = gate
    ghp = build_paper_gphhp(d, 0, 0, 1, vswap=vswap)
    Gs, _ = assemble(d, ghp, gph, Foo, Fvv)
    nvir = d["nvir"]
    r1 = 9 * nvir + 0; r2 = 10 * nvir + 1
    return np.array([[Gs[r1, r1], Gs[r1, r2]], [Gs[r2, r1], Gs[r2, r2]]]) * Ha2eV


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    gates = prep()
    print("ORCA buta pair 2x2 target: [[10.74, -1.61], [-1.44, 11.26]]")
    variants = [(), ("ip",), ("ea",), ("cross",), ("ea", "cross"),
                ("ip", "ea"), ("ip", "cross"), ("ip", "ea", "cross")]
    print(f"{'vswap':22s} " + " ".join(f"{k:>8s}" for k in gates) + "   buta 2x2")
    for vs in variants:
        rr = [gate_rms(g, vs) for g in gates.values()]
        M = buta_2x2(gates["buta"], vs)
        print(f"{str(vs):22s} " + " ".join(f"{r:8.4f}" for r in rr)
              + f"   [[{M[0,0]:6.2f},{M[0,1]:6.2f}],[{M[1,0]:6.2f},{M[1,1]:6.2f}]]")


if __name__ == "__main__":
    main()
