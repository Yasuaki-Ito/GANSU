#!/usr/bin/env python3
"""Coefficient-fit diagnostic against the ORCA razor GATE (続50).

Fix F_eff + g_phph + shipped g_phhp(base+IP) [all triplet-validated], and fit
coefficients of candidate g_phhp dressing terms (CFOUR EA raw/spinad parts,
cross variants) to the 6 ORCA singlet eigenvalues (holes 1-3).  If the true
closed form lies in the span, coefficients snap to simple rationals and the
residual drops to ~meV (like the triplet).

Run:  wsl python3 script/steom_gphhp_fit.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
from steom_gphhp_gate import restrict, assemble, eig_lab, ORCA_S, ORCA_T, Ha2eV


def ea_parts(T):
    """raw (A) and spinad (B) parts of the CFOUR EA dressing, before axpby."""
    fem = T["fem"]; wamef = T["wamef"]; wmnie = T["wmnie"]; sabej = T["sabej"]
    A = np.einsum("FBEJ,FI->EIBJ", sabej, fem)
    A += np.einsum("GFEJ,FGBI->EIBJ", sabej, wamef)
    A -= np.einsum("FBEN,NIJF->EIBJ", sabej, wmnie)
    ss = C.spinad(sabej); wn = C.spinad(wmnie)
    B = np.einsum("BFEJ,FI->EIBJ", ss, fem)
    B += np.einsum("GFEJ,GFBI->EIBJ", ss, wamef)
    B += np.einsum("BFEN,INJF->EIBJ", ss, wn)
    return A, B


def cross_parts(T, variant):
    """raw (A) and spinad (B) parts of the CFOUR cross umaei, before axpby."""
    uei, uam, uijke = C.build_uei_uam_uijke(T)
    uajmi = C.build_uajmi(T, variant); uajim = C.build_uajmi(T, variant)
    sabej = T["sabej"]; smbij = T["smbij"]
    A = np.einsum("FBEJ,FM->EMBJ", sabej, uam)
    A -= np.einsum("FBEN,FNMJ->EMBJ", sabej, uajmi)
    A -= np.einsum("NJMB,EN->EMBJ", smbij, uei)
    A += np.einsum("ONMB,NOJE->EMBJ", smbij, uijke)
    ss = C.spinad(sabej); sm = C.spinad(smbij)
    B = np.einsum("BFEJ,FM->EMBJ", ss, uam)
    B += np.einsum("BFEN,FNMJ->EMBJ", ss, 2.0 * uajim - uajmi)
    B -= np.einsum("JNMB,EN->EMBJ", sm, uei)
    B += np.einsum("ONMB,ONJE->EMBJ", sm, uijke)
    return A, B


def fold_ea(t, X_EA):
    """[E,I,B,J] EA-dressing tensor -> g_phhp[b,j,i,a] contribution."""
    g = np.einsum("EIBJ,EA->AIBJ", t, X_EA)
    return np.einsum("AIBJ->BJIA", g)


def fold_cross(t, X_EA, X_IP):
    """[E,M,B,J] cross tensor -> g_phhp[b,j,i,a] contribution (CFOUR fold)."""
    u = np.einsum("EMBJ,EA->AMBJ", t, X_EA)
    g = -np.einsum("AMBJ,MI->AIBJ", u, X_IP)
    return np.einsum("AIBJ->BJIA", g)


def spectrum(d, g_phhp, g_phph, Foo, Fvv, n=6):
    Gs, _ = assemble(d, g_phhp, g_phph, Foo, Fvv)
    w = np.sort(np.linalg.eigvals(Gs).real) * Ha2eV
    return w[:n]


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    d = C.load("xyz/H2O.xyz", "sto-3g", 1)
    d = restrict(d, occ_keep=(1, 2, 3), vir_keep=(0, 1))
    nocc = d["nocc"]; nvir = d["nvir"]
    Foo, Fvv = C.build_feff(d)
    T = C.cfour_tensors(d)
    target = np.array(ORCA_S[:6])

    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)

    w0 = spectrum(d, g_hp_a, g_ph_a, Foo, Fvv)
    print("shipped :", np.round(w0, 3), " rms=", round(float(np.sqrt(np.mean((w0-target)**2))), 4))
    print("ORCA    :", target)

    A_ea, B_ea = ea_parts(T)
    terms = {"EA_A": fold_ea(A_ea, d["X_EA"]), "EA_B": fold_ea(B_ea, d["X_EA"])}
    for v in range(4):
        A_c, B_c = cross_parts(T, v)
        terms[f"crX{v}_A"] = fold_cross(A_c, d["X_EA"], d["X_IP"])
        terms[f"crX{v}_B"] = fold_cross(B_c, d["X_EA"], d["X_IP"])

    from scipy.optimize import least_squares

    def run_fit(names, x0=None):
        def resid(c):
            g = g_hp_a.copy()
            for k, nme in enumerate(names):
                g = g + c[k] * terms[nme]
            return spectrum(d, g, g_ph_a, Foo, Fvv) - target
        x0v = np.zeros(len(names)) if x0 is None else np.array(x0, float)
        r = least_squares(resid, x0v, method="lm", xtol=1e-12, ftol=1e-12)
        rms = np.sqrt(np.mean(r.fun ** 2))
        print(f"fit [{','.join(names)}]  rms={rms:.4f} eV  c={np.round(r.x, 4)}")
        return rms, r.x

    # EA only
    run_fit(["EA_A", "EA_B"])
    # EA + each cross variant
    for v in range(4):
        run_fit(["EA_A", "EA_B", f"crX{v}_A", f"crX{v}_B"])
    # CFOUR-predicted combos as fixed sanity: gmaei gets ujaei=1.5A-0.5B
    g_cfour = g_hp_a + 1.5 * terms["EA_A"] - 0.5 * terms["EA_B"]
    wc = spectrum(d, g_cfour, g_ph_a, Foo, Fvv)
    print("shipped+CFOUR-EA(1.5,-0.5):", np.round(wc, 3),
          " rms=", round(float(np.sqrt(np.mean((wc-target)**2))), 4))


if __name__ == "__main__":
    main()
