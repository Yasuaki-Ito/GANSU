#!/usr/bin/env python3
"""Third ORCA GATE: C2h s-trans butadiene STO-3G FC4 — minimal Lb-analog lab.

ORCA (s177 ~/steom_ref/buta/buta_steom.out, E_SCF=-153.01427765, E_corr=-0.307188):
  root1  9.457 eV = -0.781(13->15) -0.624(14->16)   <- 50/50 minus pair (Lb analog)
  root2 10.201 eV = -0.982(14->15)                  <- single-config (La analog)
  root3 10.298, root4 11.275, root5 11.801, root6 12.548, root7 12.891, root8 13.949
ORCA MO: frozen 4; occ<=14 (HOMO=14), vir>=15.  Our labels: occ-4, vir-15.
Target question: does the naphthalene Lb (+0.5 eV) residual reproduce here at
full-healthy-active, small scale?

Run:  wsl python3 script/steom_buta_gate.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_gphhp_gate import restrict, assemble, eig_lab, Ha2eV
from steom_gphhp_paper import build_paper_gphhp, build_paper_gphph
from steom_hex_gate import light_load

ATOM = """
C -0.7300 0.0000 0.0
C 0.7300 0.0000 0.0
C -1.4000 1.1605 0.0
C 1.4000 -1.1605 0.0
H -1.2750 -0.9440 0.0
H 1.2750 0.9440 0.0
H -2.4900 1.1605 0.0
H 2.4900 -1.1605 0.0
H -0.8550 2.1045 0.0
H 0.8550 -2.1045 0.0
"""
ORCA_BUTA_S = [9.457, 10.201, 10.298, 11.275, 11.801, 12.548, 12.891, 13.949]


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    d0 = light_load(ATOM, basis="sto-3g", ncore=4)
    nocc = d0["nocc"]; nvir = d0["nvir"]
    # healthy channels per psingles print (adjust after first run if needed)
    occ_keep = tuple(range(nocc - 8, nocc))
    vir_keep = tuple(range(8))
    d = restrict(d0, occ_keep=occ_keep, vir_keep=vir_keep)
    print(f"nocc={nocc} nvir={nvir} active occ={d['occ_idx']} vir={d['vir_idx']}")
    Foo, Fvv = C.build_feff(d)
    target = np.array(ORCA_BUTA_S)

    def score(name, ghp, gph, n=6):
        Gs, Gt = assemble(d, ghp, gph, Foo, Fvv)
        w, v = np.linalg.eig(Gs)
        idx = np.argsort(w.real)
        w = w[idx]; v = v[:, idx]
        ws = w.real * Ha2eV
        rms = np.sqrt(np.mean((ws[:n] - target[:n]) ** 2))
        print(f"{name:22s} rms{n}={rms:.4f}")
        for k in range(8):
            comps = np.argsort(-np.abs(v[:, k]))[:3]
            cs = " ".join(f"{v[p, k].real:+.2f}({p // nvir}>{p % nvir})" for p in comps
                          if abs(v[p, k]) > 0.15)
            imtag = f" Im={abs(w[k].imag)*Ha2eV:.2f}" if abs(w[k].imag) > 4e-5 else ""
            print(f"   {ws[k]:8.3f}{imtag}  {cs}")
        return rms

    print("ORCA S:", " ".join(f"{x:7.3f}" for x in ORCA_BUTA_S))
    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    score("shipped", g_hp_a, g_ph_a)
    gph = build_paper_gphph(d, with_cross=True)
    ghp = build_paper_gphhp(d, 0, 0, 1)
    score("paper A0B0 + pap-ph", ghp, gph)
    score("paper A0B0 + ship-ph", ghp, g_ph_a)


if __name__ == "__main__":
    main()
