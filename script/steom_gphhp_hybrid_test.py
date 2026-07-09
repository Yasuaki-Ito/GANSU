#!/usr/bin/env python3
"""Hybrid test: shipped F_eff + shipped g_phph (triplet-validated) combined with
the det-exact g_phhp coupling of candidate objects (plain projection / NOsim).

If ORCA = shipped(F,g_phph) + object-X g_phhp, the singlet spectrum snaps to the
ORCA truth.  g_phhp is extracted det-exactly as (Gs-Gt)/2 in the folded (ia,jb)
matrix rep, so no formula transcription is involved.

Alignment check: det GtD vs harness Gt(shipped) element-wise (both should be
~ORCA triplet; also catches MO phase/order mismatches between the two runs).

Run:  wsl python3 script/steom_gphhp_hybrid_test.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_gphhp_gate import restrict, assemble, ORCA_S, ORCA_T, Ha2eV
from steom_nosim_exact import setup
from steom_fockspace_ref import project_1h1p


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    # ---- harness side (shipped) ----
    d = C.load("xyz/H2O.xyz", "sto-3g", 1)
    d = restrict(d, occ_keep=(1, 2, 3), vir_keep=(0, 1))
    nocc = d["nocc"]; nvir = d["nvir"]
    Foo, Fvv = C.build_feff(d)
    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    Gs_ship, Gt_ship = assemble(d, g_hp_a, g_ph_a, Foo, Fvv)

    # ---- det oracle side (same active restriction: drop deepest occ) ----
    ctx = setup(ncore=1, drop_occ=(0,))
    H = ctx["HbarN"]; S = ctx["S"]; K = ctx["K"]; E_N = ctx["E_N"]
    I = np.eye(H.shape[0])
    objs = {
        "plainproj": expm(S) @ H @ expm(-S),
        "NOsim": H @ (I - S + 0.5 * (S @ S) - 0.5 * K),
    }
    dim = nocc * nvir
    def det_blocks(M):
        Gs, Gt = project_1h1p(ctx["data"], ctx["dets"], ctx["index"], M)
        return Gs - E_N * np.eye(dim), Gt - E_N * np.eye(dim)

    # alignment check: det plain GtD vs harness Gt_ship
    GsD_p, GtD_p = det_blocks(objs["plainproj"])
    print("align: ||GtD(plain det) - Gt(shipped)|| =",
          round(float(np.linalg.norm(GtD_p - Gt_ship)), 4),
          " (diag diff)", np.round(np.diag(GtD_p - Gt_ship) * Ha2eV, 3))

    def spec(G, n=8):
        return np.sort(np.linalg.eigvals(G).real)[:n] * Ha2eV

    print("ORCA S :", ORCA_S)
    print("ship  S:", np.round(spec(Gs_ship), 3))
    for name, M in objs.items():
        GsD, GtD = det_blocks(M)
        ghp2 = GsD - GtD          # = 2*g_phhp(object) in folded rep
        cand = Gt_ship + ghp2     # shipped F/g_phph + object g_phhp
        w = spec(cand)
        rms = np.sqrt(np.mean((w[:6] - np.array(ORCA_S[:6])) ** 2))
        print(f"cand[{name:9s}] S:", np.round(w, 3), f" rms6={rms:.4f}")
        # object's own singlet for reference
        print(f"  (pure {name} S:", np.round(spec(GsD), 3), ")")


if __name__ == "__main__":
    main()
