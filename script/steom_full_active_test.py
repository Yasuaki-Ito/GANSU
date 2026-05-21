#!/usr/bin/env python3
"""
STEOM full-active definitive correctness test
==============================================

Purpose (2026-05-21): answer the single remaining question for P3 STEOM-CCSD —
is the canonical-full G^{1h1p} implementation (Nooijen-Bartlett Eq.34-63, all
F^eff + W^eff + cross terms in pyscf_steom_feff_reference.py) CORRECT, with the
observed over-correction being purely an active-space-truncation artifact, or
is there a residual FORMULA BUG?

The definitive test: STEOM with a COMPLETE active space must reduce to
EOM-EE-CCSD bit-for-bit. "Complete" = one PRINCIPAL (singles-dominated) IP
state per occupied MO + one principal EA state per virtual MO, so that the
active R1 matrix is non-singular (the active orbitals span the full o / v space).

Why the naive "lowest nocc IP roots" fails (see session diagnosis):
  H2O/sto-3g lowest 5 IP roots = 3 principal (orbitals 4,3,2) + 2 SATELLITES
  (2h1p-dominated, %singles≈0.001/0.23). The deep-core (O 1s) and 2s principal
  IP states sit at very high energy, beyond the lowest 5 roots. The resulting
  active R1 matrix is singular (det≈1e-20) → rinv blows up → garbage eigenvalues.

Fix here: build the DENSE IP-EOM / EA-EOM matrices (cheap for sto-3g), take ALL
eigenvectors, then select the principal root per orbital via a rectangular
Hungarian assignment over singles-dominated candidates. This yields a complete,
well-conditioned principal set including the core IP.

Usage:
    wsl python3 script/steom_full_active_test.py xyz/H2O.xyz sto-3g 6

  arg1 xyz, arg2 basis, arg3 n_steom_roots (states to compare).

[[pyscf-run-locally]] [[pyscf-cartesian]] [[careful-verification]]
"""

import sys
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "script")
from pyscf_steom_feff_reference import (
    read_xyz, build_bar_h, build_x_matrices,
    build_g_canonical_full,
)


def dense_eom_matrix(eom, imds):
    """Form the dense (non-symmetric) EOM matrix by applying matvec to each
    unit vector. Only practical for small (minimal-basis) systems."""
    size = eom.vector_size()
    H = np.zeros((size, size))
    for j in range(size):
        v = np.zeros(size)
        v[j] = 1.0
        H[:, j] = np.asarray(eom.matvec(v, imds)).ravel()
    return H


def principal_roots_ip(mycc, nocc, nvir):
    """Return (r1_list, r2_list, omega_list, active_idx) — one PRINCIPAL IP
    state per occupied orbital, from a dense diagonalization of IP-EOM-CCSD.
    """
    from pyscf.cc import eom_rccsd
    eom = eom_rccsd.EOMIP(mycc)
    imds = eom.make_imds()
    H = dense_eom_matrix(eom, imds)
    w, vr = scipy.linalg.eig(H)
    w = np.real(w)
    vr = np.real(vr)
    nroot = vr.shape[1]
    # split each eigenvector, compute %singles + dominant-orbital |R1|
    R1 = np.zeros((nroot, nocc))
    psingles = np.zeros(nroot)
    r2_all = []
    for k in range(nroot):
        v = vr[:, k]
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        r1 = v[:nocc]
        r2 = v[nocc:].reshape(nocc, nocc, nvir)
        n1 = r1 @ r1
        n2 = np.einsum("ija,ija->", r2, r2)
        R1[k] = r1
        psingles[k] = n1 / (n1 + n2 + 1e-300)
        r2_all.append(r2)
    # rectangular Hungarian: assign each occupied orbital to a distinct root,
    # maximizing |R1[root, orb]| but only over singles-dominated candidates.
    SING = 0.5
    cost = np.full((nocc, nroot), 1e6)
    for orb in range(nocc):
        for k in range(nroot):
            if psingles[k] >= SING:
                cost[orb, k] = -abs(R1[k, orb])
    orb_idx, root_idx = linear_sum_assignment(cost)  # len = nocc
    sel_root = np.zeros(nocc, dtype=int)
    for o, r in zip(orb_idx, root_idx):
        sel_root[o] = r
    r1_list = [R1[sel_root[o]].copy() for o in range(nocc)]
    r2_list = [r2_all[sel_root[o]].copy() for o in range(nocc)]
    omega = [w[sel_root[o]] for o in range(nocc)]
    active_idx = list(range(nocc))  # orbital o ↔ root assigned to orbital o
    return r1_list, r2_list, omega, active_idx, psingles[sel_root]


def principal_roots_ea(mycc, nocc, nvir):
    """Return one PRINCIPAL EA state per virtual orbital (dense diag)."""
    from pyscf.cc import eom_rccsd
    eom = eom_rccsd.EOMEA(mycc)
    imds = eom.make_imds()
    H = dense_eom_matrix(eom, imds)
    w, vr = scipy.linalg.eig(H)
    w = np.real(w)
    vr = np.real(vr)
    nroot = vr.shape[1]
    R1 = np.zeros((nroot, nvir))
    psingles = np.zeros(nroot)
    r2_all = []
    for k in range(nroot):
        v = vr[:, k]
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        r1 = v[:nvir]
        r2 = v[nvir:].reshape(nocc, nvir, nvir)
        n1 = r1 @ r1
        n2 = np.einsum("iab,iab->", r2, r2)
        R1[k] = r1
        psingles[k] = n1 / (n1 + n2 + 1e-300)
        r2_all.append(r2)
    SING = 0.5
    cost = np.full((nvir, nroot), 1e6)
    for orb in range(nvir):
        for k in range(nroot):
            if psingles[k] >= SING:
                cost[orb, k] = -abs(R1[k, orb])
    orb_idx, root_idx = linear_sum_assignment(cost)
    sel_root = np.zeros(nvir, dtype=int)
    for o, r in zip(orb_idx, root_idx):
        sel_root[o] = r
    r1_list = [R1[sel_root[o]].copy() for o in range(nvir)]
    r2_list = [r2_all[sel_root[o]].copy() for o in range(nvir)]
    omega = [w[sel_root[o]] for o in range(nvir)]
    active_idx = list(range(nvir))
    return r1_list, r2_list, omega, active_idx, psingles[sel_root]


def main(xyz_path, basis, n_steom_roots):
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd

    print("=" * 78)
    print("STEOM FULL-ACTIVE definitive correctness test")
    print(f"  xyz={xyz_path}  basis={basis}  n_steom={n_steom_roots}")
    print("=" * 78)

    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    mycc = cc.CCSD(mf); mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    mo = mf.mo_coeff
    eri_mo = ao2mo.kernel(mol, mo, compact=False).reshape(nmo, nmo, nmo, nmo)
    f_oo = np.diag(mf.mo_energy[:nocc]); f_vv = np.diag(mf.mo_energy[nocc:])
    print(f"\n  nocc={nocc} nvir={nvir} nmo={nmo}")
    print(f"  CCSD E_corr = {mycc.e_corr:.10f}")

    bar_h = build_bar_h(eri_mo, t1, t2, f_oo, f_vv, nocc, nvir)

    # EOM-EE-CCSD exact reference
    eom_ee = eom_rccsd.EOMEESinglet(mycc)
    e_ee, _ = eom_ee.kernel(nroots=n_steom_roots)
    e_ee = np.atleast_1d(np.asarray(e_ee))

    # Complete PRINCIPAL IP / EA sets (one per orbital, dense diag)
    r1_ip, r2_ip, w_ip, occ_idx, ps_ip = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, w_ea, vir_idx, ps_ea = principal_roots_ea(mycc, nocc, nvir)

    print("\n--- Complete principal IP set (one per occ orbital) ---")
    for o in range(nocc):
        print(f"  orb {o}: ω={w_ip[o]:.6f} %singles={ps_ip[o]:.4f} "
              f"R1[orb]={r1_ip[o][o]:+.4f}")
    print("--- Complete principal EA set (one per vir orbital) ---")
    for o in range(nvir):
        print(f"  orb {o}: ω={w_ea[o]:.6f} %singles={ps_ea[o]:.4f} "
              f"R1[orb]={r1_ea[o][o]:+.4f}")

    R1mat_ip = np.array([[r1_ip[l][occ_idx[m]] for m in range(nocc)]
                         for l in range(nocc)])
    R1mat_ea = np.array([[r1_ea[l][vir_idx[e]] for e in range(nvir)]
                         for l in range(nvir)])
    print(f"\n  R1mat_IP cond={np.linalg.cond(R1mat_ip):.3e} "
          f"det={np.linalg.det(R1mat_ip):.3e}")
    print(f"  R1mat_EA cond={np.linalg.cond(R1mat_ea):.3e} "
          f"det={np.linalg.det(R1mat_ea):.3e}")

    # Full canonical STEOM G^{1h1p} (all terms, pre-normalized s)
    Gf, _, _, _, _ = build_g_canonical_full(
        bar_h, r2_ip, r2_ea, r1_ip, r1_ea,
        occ_idx, vir_idx, nocc, nvir,
    )
    ef = np.sort(np.real(np.linalg.eigvals(Gf)))

    print("\n" + "=" * 78)
    print("DEFINITIVE: STEOM(full principal active) vs EOM-EE-CCSD")
    print("=" * 78)
    n_cmp = min(n_steom_roots, len(ef), len(e_ee))
    print(f"  {'state':>5} {'EOM-EE':>14} {'STEOM-full':>14} {'Δ(mHa)':>10}")
    maxdev = 0.0
    for k in range(n_cmp):
        dev = (ef[k] - e_ee[k]) * 1000.0
        maxdev = max(maxdev, abs(dev))
        print(f"  {k:>5} {e_ee[k]:>14.8f} {ef[k]:>14.8f} {dev:>+10.3f}")
    print(f"  max|Δ| = {maxdev:.3f} mHa")
    if maxdev < 1.0:
        print("  ✅ PASS: STEOM == EOM-CCSD in full active → FORMULAS CORRECT.")
        print("     Over-correction at truncated active = active-space artifact.")
    else:
        print("  ❌ FAIL: residual deviation in full active → FORMULA BUG remains.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(1)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
