#!/usr/bin/env python3
"""Find the CORRECT index permutation for the STEOM base off-diagonal.

steom_gansu_vs_oracle localized the naphthalene-class overshoot to the s=0 BASE
1h1p block: GANSU's `2 Wovvo[j,b,a,i] - Wovov[j,a,i,b]` disagrees with the exact
<ia|Hbar|jb> ONLY on the i!=j & a!=b (config-coupling) elements. Diagonal and
semi-diagonal match. So Loo/Lvv/Wovvo/Wovov are fine; only the assembly's index
map into the fully-off-diagonal block is wrong.

This brute-forces every axis permutation of Wovvo and Wovov in
    T[i,a,j,b] = 2 * Wovvo.transpose(P1)[i,a,j,b] - Wovov.transpose(P2)[i,a,j,b]
against the exact two-body block T_exact = <ia|Hbar|jb> - (dij Lvv[a,b] - dab Loo[i,j]),
and reports the permutations that reproduce it (esp. the i!=j&a!=b block). If none
match => the bug is in build_bar_h's Wovvo/Wovov themselves, not the assembly.

Run: wsl python3 script/steom_base_permfit.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile, itertools
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, hf_det, project_1h1p)
import steom_cfour_weff as CW
Ha = 27.211386245988


def oracle_base_block(atom, basis, ncore):
    """exact <ia|Hbar|jb> (s=0 STEOM base) as B[i,a,j,b] in Hartree (ref-shifted)."""
    data = get_active_data(atom=atom, basis=basis, ncore=ncore)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    Gs, _ = project_1h1p(data, dets, index, Hbar)
    B = (Gs - E_N * np.eye(nocc * nvir)).reshape(nocc, nvir, nocc, nvir)
    return B, nocc, nvir


def run_one(label, atom, basis, ncore, natom):
    B, nocc, nvir = oracle_base_block(atom, basis, ncore)

    xyzf = os.path.join(tempfile.gettempdir(), "pf.xyz")
    open(xyzf, "w").write(f"{natom}\n\n" + "\n".join(a.strip() for a in atom.split(";")) + "\n")
    d = CW.load(xyzf, basis, ncore)
    bar = d["bar"]
    Loo, Lvv = bar["Loo"], bar["Lvv"]
    Wovvo, Wovov = bar["Wovvo"], bar["Wovov"]

    # exact two-body block = B - (dij Lvv[a,b] - dab Loo[i,j])
    T = B.copy()
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    diag = (Lvv[a, b] if i == j else 0.0) - (Loo[i, j] if a == b else 0.0)
                    T[i, a, j, b] -= diag

    # current GANSU expression, as sanity
    cur = 2.0 * np.einsum("jbai->iajb", Wovvo) - np.einsum("jaib->iajb", Wovov)
    # mask: i!=j & a!=b (the block that's wrong)
    mask = np.zeros((nocc, nvir, nocc, nvir), bool)
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i != j and a != b:
                        mask[i, a, j, b] = True
    print(f"\n=== {label}  nocc={nocc} nvir={nvir}")
    print(f"  ||T_exact(off)||={np.linalg.norm(T[mask]):.4f}  "
          f"||current-Texact(off)||={np.linalg.norm((cur-T)[mask]):.4f}  "
          f"(full: {np.linalg.norm(cur-T):.4f})")

    # Wovvo source axes (k,b,c,j)=(occ,vir,vir,occ); Wovov (k,b,i,d)=(occ,vir,occ,vir).
    # Map each source axis to a target letter (occ->{i,j}, vir->{a,b}); output "iajb".
    # einsum(<input labels in source-axis order> + "->iajb").
    def assignments(src_types):
        # src_types: list of 'o'/'v' per source axis. Yield input-label strings.
        occ_slots = ["i", "j"]; vir_slots = ["a", "b"]
        occ_axes = [k for k, t in enumerate(src_types) if t == "o"]
        vir_axes = [k for k, t in enumerate(src_types) if t == "v"]
        outs = []
        for op in itertools.permutations(occ_slots):
            for vp in itertools.permutations(vir_slots):
                lab = [None] * 4
                for ax, s in zip(occ_axes, op): lab[ax] = s
                for ax, s in zip(vir_axes, vp): lab[ax] = s
                outs.append("".join(lab))
        return outs

    A_vvo = assignments(["o", "v", "v", "o"])   # Wovvo
    A_vov = assignments(["o", "v", "o", "v"])   # Wovov
    print(f"  assignments: Wovvo {len(A_vvo)}, Wovov {len(A_vov)}")

    best = []
    for a1 in A_vvo:
        Wv1 = np.einsum(f"{a1}->iajb", Wovvo)
        for a2 in A_vov:
            Wv2 = np.einsum(f"{a2}->iajb", Wovov)
            cand = 2.0 * Wv1 - Wv2
            best.append((np.linalg.norm((cand - T)[mask]),
                         np.linalg.norm(cand - T), a1, a2))
    best.sort()
    print("  current GANSU = 2*Wovvo'jbai' - Wovov'jaib'")
    print("  top (2,-1) matches (err on i!=j&a!=b block / full):")
    for err_off, err_full, a1, a2 in best[:3]:
        tag = "  <== EXACT" if err_full < 1e-8 else ("  (off-diag exact)" if err_off < 1e-8 else "")
        print(f"    2*Wovvo'{a1}' - Wovov'{a2}':  off={err_off:.4e} full={err_full:.4e}{tag}")

    # ---- rigorous: lstsq fit of the pure config-coupling block (i!=j&a!=b) against
    #      ALL available raw dressed-integral permutations. Residual ~0 => the correct
    #      formula is a linear combo of these; nonzero => build_bar_h lacks a needed term.
    cols = []; names = []
    for a1 in A_vvo:
        cols.append(np.einsum(f"{a1}->iajb", Wovvo)[mask]); names.append(f"Wovvo'{a1}'")
    for a2 in A_vov:
        cols.append(np.einsum(f"{a2}->iajb", Wovov)[mask]); names.append(f"Wovov'{a2}'")
    # also raw eri_ovvo / eri_ovov (undressed) in case base should be bare integrals
    er_vvo = bar["eri_ovvo"]; er_vov = bar["eri_ovov"]
    for a1 in A_vvo:
        cols.append(np.einsum(f"{a1}->iajb", er_vvo)[mask]); names.append(f"eri_ovvo'{a1}'")
    for a2 in A_vov:
        cols.append(np.einsum(f"{a2}->iajb", er_vov)[mask]); names.append(f"eri_ovov'{a2}'")
    Amat = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(Amat, T[mask], rcond=None)
    resid = np.linalg.norm(Amat @ coef - T[mask]) / (np.linalg.norm(T[mask]) + 1e-30)
    print(f"  lstsq over {len(names)} dressed/bare perms: rel-resid={resid:.3e}")
    for nm, c in sorted(zip(names, coef), key=lambda z: -abs(z[1]))[:6]:
        if abs(c) > 1e-3:
            print(f"      {nm:16s} c={c:+.4f}")


def main():
    run_one("H4 rect h=1.4", "H 0 0 0; H 2.0 0 0; H 0 1.4 0; H 2.0 1.4 0", "sto-3g", 0, 4)
    run_one("H6 ladder h=1.4",
            "; ".join(f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(6)), "sto-3g", 0, 6)
    lines = [l.strip() for l in open("xyz/H2O.xyz").read().splitlines()[2:5]]
    run_one("H2O FC1", "; ".join(lines), "sto-3g", 1, 3)


if __name__ == "__main__":
    main()
