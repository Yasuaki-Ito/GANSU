#!/usr/bin/env python3
"""Route-scale scan on THREE ORCA gates (H2O / zigzag-C6H8 / butadiene).

Hypothesis (続53+): the config-mixing pair splitting deficit (naphthalene Lb +0.5,
butadiene minus-pair +1.3) is a wrong overall factor on the Eq.(62) cross route
(e.g. a missing ordering factor 2 in the {S_ip S_ea} bilinear).  Scan fcross
(and fea) and require ALL THREE gates to improve simultaneously.

Run:  wsl python3 script/steom_gphhp_fscan.py
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


def rms_of(gate, **kw):
    d, (Foo, Fvv), gph, target, n = gate
    ghp = build_paper_gphhp(d, 0, 0, 1, **kw)
    Gs, _ = assemble(d, ghp, gph, Foo, Fvv)
    ws = np.sort(np.linalg.eigvals(Gs).real) * Ha2eV
    return float(np.sqrt(np.mean((ws[:n] - np.array(target[:n])) ** 2)))


def main():
    gates = prep()
    print(f"{'variant':28s} " + " ".join(f"{k:>8s}" for k in gates))
    for kw in [dict(), dict(fcross=1.5), dict(fcross=2.0), dict(fcross=3.0),
               dict(fcross=0.5), dict(fea=1.5), dict(fea=2.0), dict(fea=0.5),
               dict(fip=1.5), dict(fip=0.5),
               dict(fcross=2.0, fea=2.0), dict(fcross=2.0, fea=1.5)]:
        tag = ",".join(f"{k}={v}" for k, v in kw.items()) or "paper(1,1,1)"
        rr = [rms_of(g, **kw) for g in gates.values()]
        print(f"{tag:28s} " + " ".join(f"{r:8.4f}" for r in rr))


if __name__ == "__main__":
    main()
