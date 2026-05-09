#!/usr/bin/env python3
"""
H2 dissociation curve via PySCF: RHF / UHF / FCI.

UHF is solved from long R (5.0 Å) down to short R, reusing the previous
broken-symmetry solution as the initial guess. At 5.0 Å we initialize
UHF with a localized alpha-on-H1 / beta-on-H2 density to trigger the
symmetry-broken branch.

Usage:
  wsl python3 h2_dissociation_pyscf.py [--basis cc-pvdz] [--rmin 0.4 --rmax 5.0 --npts 24]

Outputs:
  h2_dissociation_pyscf.csv
  h2_dissociation_pyscf.png
"""

import argparse
import csv
import sys
import numpy as np

from pyscf import gto, scf, fci


def build_mol(R, basis):
    return gto.M(atom=f"H 0 0 0; H 0 0 {R}", basis=basis, unit="Angstrom",
                 verbose=0)


def run_rhf(R, basis):
    mol = build_mol(R, basis)
    mf = scf.RHF(mol).run()
    return mf.e_tot


def run_fci(R, basis):
    mol = build_mol(R, basis)
    mf = scf.RHF(mol).run()
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    return e_fci + mol.energy_nuc() if False else e_fci  # FCI returns total


def init_broken_symmetry_dm(mol):
    """Localize alpha on H1 and beta on H2 to break spatial symmetry."""
    # Minimal bonding/antibonding guess, then mix to localize
    mf0 = scf.RHF(mol).run()
    mo = mf0.mo_coeff
    # bonding (MO 0), antibonding (MO 1) for H2 minimal
    # Rotate: localized1 = (bonding + antibonding)/sqrt(2), localized2 = (bonding - antibonding)/sqrt(2)
    alpha_mo = mo.copy()
    beta_mo  = mo.copy()
    if mo.shape[1] >= 2:
        b = mo[:, 0].copy()
        a = mo[:, 1].copy()
        alpha_mo[:, 0] = (b + a) / np.sqrt(2)  # localized on one H
        beta_mo[:, 0]  = (b - a) / np.sqrt(2)  # localized on other H
    occ = np.zeros((2, mol.nao))
    occ[0, 0] = 1.0  # alpha in first localized
    occ[1, 0] = 1.0  # beta in second localized
    dm_a = alpha_mo @ np.diag(occ[0]) @ alpha_mo.T
    dm_b = beta_mo  @ np.diag(occ[1]) @ beta_mo.T
    return np.array([dm_a, dm_b])


def run_uhf_chain(Rs, basis):
    """Solve UHF at each R from large to small, reusing previous DM as guess."""
    energies = [None] * len(Rs)
    # start from largest R with a symmetry-broken DM
    order = sorted(range(len(Rs)), key=lambda i: -Rs[i])

    prev_dm = None
    for idx in order:
        R = Rs[idx]
        mol = build_mol(R, basis)
        mf = scf.UHF(mol)
        mf.verbose = 0
        if prev_dm is None:
            # First (largest R): explicit broken-symmetry guess
            dm0 = init_broken_symmetry_dm(mol)
        else:
            # Reuse previous DM (may need to be re-projected if basis differs,
            # but same basis here — just use as-is)
            dm0 = prev_dm
        mf.kernel(dm0=dm0)
        if not mf.converged:
            mf = mf.newton().run(dm0=dm0)
        energies[idx] = mf.e_tot
        prev_dm = mf.make_rdm1()

        # Report S² for diagnostic
        try:
            s2, mult = mf.spin_square()
        except Exception:
            s2, mult = float("nan"), float("nan")
        print(f"  UHF R={R:.3f}: E={mf.e_tot:.8f}  <S²>={s2:.3f}")
    return energies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--rmin", type=float, default=0.4)
    ap.add_argument("--rmax", type=float, default=5.0)
    ap.add_argument("--npts", type=int, default=24)
    ap.add_argument("--out", default="h2_dissociation_pyscf")
    args = ap.parse_args()

    Rs = np.linspace(args.rmin, args.rmax, args.npts)

    print("=== RHF ===")
    rhf_E = []
    for R in Rs:
        E = run_rhf(R, args.basis)
        rhf_E.append(E)
        print(f"  RHF R={R:.3f}: E={E:.8f}")

    print("\n=== FCI ===")
    fci_E = []
    for R in Rs:
        E = run_fci(R, args.basis)
        fci_E.append(E)
        print(f"  FCI R={R:.3f}: E={E:.8f}")

    print("\n=== UHF (chain from large R) ===")
    uhf_E = run_uhf_chain(Rs, args.basis)

    # CSV
    csv_path = f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R_A", "RHF", "UHF", "FCI"])
        for i, R in enumerate(Rs):
            w.writerow([f"{R:.4f}", f"{rhf_E[i]:.10f}",
                        f"{uhf_E[i]:.10f}" if uhf_E[i] is not None else "",
                        f"{fci_E[i]:.10f}"])
    print(f"\nwrote {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Rs, rhf_E, "o-",  color="tab:blue",  label="RHF")
    ax.plot(Rs, uhf_E, "s--", color="tab:cyan",  label="UHF")
    ax.plot(Rs, fci_E, "*-",  color="black",     label="Full CI", markersize=10)
    ax.set_xlabel("H-H distance (Å)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title(f"H$_2$ dissociation (PySCF, basis: {args.basis})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"{args.out}.png"
    plt.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
