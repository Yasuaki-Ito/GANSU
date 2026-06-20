#!/usr/bin/env python3
"""
Eigenvalue-metric debugging harness for STEOM W^eff (the ONLY valid metric;
element-wise localizer is invalid, see AQUA/STEOM_WEFF_CFOUR_REF.md notes).

Reports max|Δeig| (mHa) of STEOM-G eigenvalues vs EOM-EE-CCSD over a small
molecule set, for both the current Nooijen build_g_canonical_full and the
faithful CFOUR port, so changes can be scored quickly.

Usage: wsl python3 script/steom_eig_debug.py
"""
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full

np.set_printoptions(precision=5, suppress=True, linewidth=170)

MOLS = [("xyz/H2O.xyz", "sto-3g", 2), ("xyz/CH4.xyz", "sto-3g", 1)]


def nooijen_g(d):
    """g_phhp/g_phph from the current code path."""
    G, g_phph, g_phhp, *_ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    return g_phhp, g_phph, G


def cfour_g(d, uajmi_variant=1, cross_sign=+1.0, with_cross=True):
    T = C.cfour_tensors(d)
    uj, uje = C.build_ujaei(T)
    um, umb = C.build_umabi(T)
    if with_cross:
        uei, uam, uijke = C.build_uei_uam_uijke(T)
        ua = C.build_uajmi(T, uajmi_variant)
        ume, umie = C.build_umaei(T, uei, uam, uijke, ua, ua)
        um = um + cross_sign * np.einsum("EMBJ,EA->AMBJ", ume, d["X_EA"])
        umb = umb + cross_sign * np.einsum("EMBJ,EA->AMBJ", umie, d["X_EA"])
    g_hp, g_ph = C.assemble_g(d, T, uj, uje, um, umb)
    return g_hp, g_ph


def eigdelta(d, g_hp, g_ph):
    e = C.full_G_eigs(d, g_hp, g_ph)
    return e, np.max(np.abs(e - d["e_s"])) * 1000


def triplet_eigs(d, g_ph):
    """Triplet effective H = δ_ij Fvv - δ_ab Foo - g_phph  (no g_phhp).
    Eigenvalues must match EOM triplet e_t if g_phph is correct."""
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    Foo, Fvv = C.build_feff(d)
    G = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    v = -g_ph[a, j, b, i]
                    if i == j: v += Fvv[a, b]
                    if a == b: v -= Foo[i, j]
                    G[r, c] = v
    return np.sort(np.linalg.eigvals(G).real)


def main():
    data = [(m, C.load(*m)) for m in MOLS]
    print("=== Nooijen build_g_canonical_full (current code) ===")
    print("    singlet(full) vs e_s  |  triplet(g_phph only) vs e_t  -> isolates block")
    for m, d in data:
        g_hp, g_ph, G = nooijen_g(d)
        eN = np.sort(np.real(np.linalg.eigvals(G)))
        dN = np.max(np.abs(eN - d["e_s"])) * 1000
        et = triplet_eigs(d, g_ph)
        dT = np.max(np.abs(et - d["e_t"])) * 1000
        print(f"  {m[0]:18s} singlet Δ={dN:7.2f}  triplet Δ={dT:7.2f} mHa")
    print("=== CFOUR port (faithful) ===")
    for m, d in data:
        g_hp, g_ph = cfour_g(d)
        e, dd = eigdelta(d, g_hp, g_ph)
        # also base-only
        T = C.cfour_tensors(d)
        z1 = np.zeros((len(d["r2_ea"]), d["nocc"], d["nvir"], d["nocc"]))
        z2 = np.zeros((d["nvir"], len(d["r2_ip"]), d["nvir"], d["nocc"]))
        gb_hp, gb_ph = C.assemble_g(d, T, z1, z1, z2, z2)
        _, db = eigdelta(d, gb_hp, gb_ph)
        print(f"  {m[0]:18s} base Δ={db:7.2f}  full Δ={dd:7.2f} mHa")


if __name__ == "__main__":
    main()
