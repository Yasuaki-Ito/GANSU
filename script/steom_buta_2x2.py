#!/usr/bin/env python3
"""Dissect the butadiene minus-pair 2x2 block of Gs (basis {(9>0),(10>1)}).

ORCA implied (from eigpair 9.457/12.548 + printed eigvecs):
    [[10.74, -1.61], [-1.44, 11.26]]  eV   (diag split 0.52, coupling ~1.5)
Ours: print the same elements for paper build, decomposed by route
(F/base/IP60/EA61/cross62 for g_phhp; g_phph separately), plus shipped.

Run:  wsl python3 script/steom_buta_2x2.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_gphhp_gate import restrict, assemble, Ha2eV
from steom_gphhp_paper import build_paper_gphhp, build_paper_gphph
from steom_hex_gate import light_load
from steom_buta_gate import ATOM


def two_by_two(d, ghp, gph, Foo, Fvv):
    Gs, _ = assemble(d, ghp, gph, Foo, Fvv)
    nvir = d["nvir"]
    r1 = 9 * nvir + 0    # (9->0)
    r2 = 10 * nvir + 1   # (10->1)
    M = np.array([[Gs[r1, r1], Gs[r1, r2]], [Gs[r2, r1], Gs[r2, r2]]]) * Ha2eV
    return M


def main():
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    d = light_load(ATOM, ncore=4)
    d = restrict(d, occ_keep=tuple(range(3, d["nocc"])), vir_keep=tuple(range(8)))
    nocc = d["nocc"]; nvir = d["nvir"]
    Foo, Fvv = C.build_feff(d)
    gph = build_paper_gphph(d, with_cross=True)

    print("ORCA implied 2x2: [[10.74, -1.61], [-1.44, 11.26]]")
    # paper full
    ghp = build_paper_gphhp(d, 0, 0, 1)
    print("paper full      :", two_by_two(d, ghp, gph, Foo, Fvv).tolist())
    # decompose g_phhp routes: base only (fip=fea=fcross=0)
    for tag, kw in [("base only", dict(fip=0, fea=0, fcross=0)),
                    ("+IP(60)", dict(fea=0, fcross=0)),
                    ("+EA(61)", dict(fip=0, fcross=0)),
                    ("+cross(62)", dict(fip=0, fea=0))]:
        g = build_paper_gphhp(d, 0, 0, 1, **kw)
        print(f"{tag:16s}:", two_by_two(d, g, gph, Foo, Fvv).tolist())
    # zero g_phhp entirely -> shows F + g_phph part
    print("ghp=0 (F-gph)   :", two_by_two(d, np.zeros_like(ghp), gph, Foo, Fvv).tolist())
    # shipped
    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    print("shipped         :", two_by_two(d, g_hp_a, g_ph_a, Foo, Fvv).tolist())
    print("paper-hp+ship-ph:", two_by_two(d, ghp, g_ph_a, Foo, Fvv).tolist())


if __name__ == "__main__":
    main()
