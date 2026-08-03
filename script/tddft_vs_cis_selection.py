#!/usr/bin/env python3
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
"""Robustness of the automatic fragment SELECTION to the preprocessing model.

The method locates the excitation with a cheap CIS and selects the real-space
fragment from the per-atom localization of the (state-averaged) transition
density. Section 4.9 argues that this localization is a low-order, robust
property, so a different cheap excited-state model should return the same
fragment. This script tests that directly: it computes the same per-atom
localization score and the same greedy selection (floor f, coverage T) from

    CIS  ( = TDA / Hartree-Fock ),
    TDA / B3LYP        (a global hybrid, no long-range correction),
    TDA / CAM-B3LYP    (range-separated),

and reports whether the selected atom set is the same. The per-atom score is
built transparently from the TDA amplitudes X_{ia}:

    hole weight of occupied MO i = sum_a X_{ia}^2
    particle weight of virtual a = sum_i X_{ia}^2
    s_A = sum_states [ sum_i n^hole_i pop_A(i) + sum_a n^part_a pop_A(a) ]

with pop_A the Loewdin per-atom population of the MO, state-averaged uniformly
and normalised to sum to one over atoms (cf. Eqs 9/11 of the paper).

Requires PySCF (+ numpy, scipy).  Usage:
    python tddft_vs_cis_selection.py [molecule.xyz]
default molecule is ../xyz/anisaldehyde.xyz relative to this script.
"""
import os, sys
import numpy as np
from scipy.linalg import sqrtm
from pyscf import gto, scf, dft, tdscf

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "xyz", "anisaldehyde.xyz")
BASIS   = "cc-pvdz"
NSTATES = 10          # states averaged (uniform weights), cf. n_cis in the paper
FLOOR   = 0.01        # per-atom floor f
COVER   = 0.92        # cumulative coverage target T

# --- molecule ---
L = open(XYZ).read().splitlines()
nat = int(L[0].split()[0])
mol = gto.M(atom="\n".join(L[2:2 + nat]), basis=BASIS, verbose=0)
nocc = mol.nelectron // 2
aos  = mol.aoslice_by_atom()
S    = mol.intor("int1e_ovlp")
S12  = sqrtm(S).real

def per_atom_pop(C):
    """Loewdin per-atom population of every MO column of C (AO x nmo)."""
    P = (S12 @ C) ** 2                         # AO x nmo
    out = np.zeros((mol.natm, C.shape[1]))
    for A in range(mol.natm):
        out[A] = P[aos[A, 2]:aos[A, 3]].sum(axis=0)
    return out

def selection(mo_coeff, td):
    pop = per_atom_pop(mo_coeff)               # natm x nmo
    s = np.zeros(mol.natm)
    for k in range(len(td.e)):
        X = np.asarray(td.xy[k][0])            # nocc x nvir  (TDA: Y = 0)
        nhole = (X ** 2).sum(axis=1)           # per occupied MO
        npart = (X ** 2).sum(axis=0)           # per virtual MO
        s += pop[:, :nocc] @ nhole
        s += pop[:, nocc:nocc + X.shape[1]] @ npart
    s /= s.sum()
    order, sel, acc = np.argsort(-s), [], 0.0
    for A in order:
        if s[A] < FLOOR:
            break
        sel.append(int(A)); acc += s[A]
        if acc >= COVER:
            break
    return sorted(sel), s

def build(kind):
    if kind == "CIS (TDA/HF)":
        mf = scf.RHF(mol).run()
    else:
        mf = dft.RKS(mol); mf.xc = kind.split("/")[1]; mf.run()
    td = tdscf.TDA(mf); td.nstates = NSTATES; td.kernel()
    return mf, td

HARTREE2EV = 27.211386
res = {}
for kind in ("CIS (TDA/HF)", "TDA/b3lyp", "TDA/camb3lyp"):
    mf, td = build(kind)
    sel, s = selection(mf.mo_coeff, td)
    res[kind] = (sel, s, td.e[:5] * HARTREE2EV)

ref = set(res["CIS (TDA/HF)"][0])
print(f"molecule : {os.path.basename(XYZ)}   basis {BASIS}   {NSTATES} states averaged"
      f"   floor {FLOOR}  coverage {COVER}\n")
for kind, (sel, s, e) in res.items():
    ss = set(sel)
    inter, uni = len(ss & ref), len(ss | ref)
    print(f"{kind:14s} lowest 5 (eV): {np.round(e,2)}")
    print(f"{'':14s} selected {len(sel):2d} atoms: {sel}")
    print(f"{'':14s} vs CIS  ->  shared {inter}   CIS-only {len(ref-ss)}   "
          f"{kind}-only {len(ss-ref)}   Jaccard {inter/uni:.3f}\n")

print("Interpretation: identical (or near-identical) selected atom sets across the "
      "three preprocessing models demonstrate that the fragment is set by the "
      "localization, not by the particular cheap excited-state method (Sec 4.9).")
