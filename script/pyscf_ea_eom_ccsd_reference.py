#!/usr/bin/env python3
"""
PySCF reference: closed-shell canonical EA-EOM-CCSD eigenvalues + R1 / R2 norms.

Compared against GANSU `--post_hf_method ea_eom_ccsd` (Phase P2 sub-phase 2.6
validation gate). Targets:
  * |ω_GANSU − ω_PySCF| < 1 mHa for every reported root
  * %singles agreement to ~1e-2

Usage (run inside WSL Ubuntu — see [[pyscf-run-locally]]):
    wsl python3 script/pyscf_ea_eom_ccsd_reference.py xyz/H2O.xyz sto-3g 2
    wsl python3 script/pyscf_ea_eom_ccsd_reference.py xyz/H2O.xyz cc-pVDZ 4
    wsl python3 script/pyscf_ea_eom_ccsd_reference.py xyz/N2.xyz   cc-pVDZ 4

Notes:
  * Uses `cart=True` (GANSU is Cartesian GTO; required for d+ basis). See
    [[pyscf-cartesian]].
  * EA-EOM-CCSD probes N+1 electron states (electron attachment), so the
    smallest root is the negative electron affinity (or the lowest unbound
    state) — for H2O it lies above all virtual orbital energies.
"""

import sys
import numpy as np


def read_xyz(path):
    with open(path) as f:
        natoms = int(f.readline().strip())
        f.readline()
        atoms = []
        for _ in range(natoms):
            parts = f.readline().split()
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "; ".join(atoms)


def main(xyz_path: str, basis: str, n_roots: int, frozen_core: int = 0) -> None:
    from pyscf import gto, scf, cc
    from pyscf.cc import eom_rccsd

    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf  = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    mycc = cc.CCSD(mf)
    if frozen_core > 0:
        mycc.frozen = frozen_core
    mycc.conv_tol = 1e-9
    mycc.conv_tol_normt = 1e-7
    mycc.kernel()

    myeom = eom_rccsd.EOMEA(mycc)
    e_ea, r_ea = myeom.kernel(nroots=n_roots)
    if np.isscalar(e_ea):
        e_ea = np.asarray([e_ea])
        r_ea = [r_ea]

    nocc_active = mycc.nocc
    nvir        = mycc.nmo - mycc.nocc

    print("--- PySCF EA-EOM-CCSD reference -------------------------------------")
    print(f"  xyz             = {xyz_path}")
    print(f"  basis           = {basis} (cart=True)")
    print(f"  frozen_core     = {frozen_core}")
    print(f"  HF energy       = {mf.e_tot:.10f} Ha")
    print(f"  CCSD corr       = {mycc.e_corr:.10f} Ha")
    print(f"  nocc_active     = {nocc_active}")
    print(f"  nvir            = {nvir}")
    print(f"  nroots          = {n_roots}")
    print(f"  EA roots (sorted ascending):")
    print(f"   k    omega (Ha)         omega (eV)     %singles")
    for k in range(len(e_ea)):
        # PySCF EA EOM stores r as concat of (r1, r2) where r1 has shape (nvir,)
        # and r2 has shape (nocc, nvir, nvir).
        r_vec = np.asarray(r_ea[k]).ravel()
        r1 = r_vec[:nvir]
        r2 = r_vec[nvir:].reshape(nocc_active, nvir, nvir)
        n1 = float(np.einsum("a,a->", r1, r1))
        n2 = float(np.einsum("jab,jab->", r2, r2))
        pct = n1 / (n1 + n2) if (n1 + n2) > 0.0 else 0.0
        print(f"  {k:2d}    {e_ea[k]:.10f}    {e_ea[k]*27.2114:10.4f}    {pct:.4f}")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        sys.exit(1)
    xyz   = sys.argv[1]
    basis = sys.argv[2]
    n_rt  = int(sys.argv[3])
    fc    = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    main(xyz, basis, n_rt, fc)
