#!/usr/bin/env python3
"""
Generate PySCF reference values for RHF nuclear gradient validation.
Uses Cartesian Gaussians (cart=True) to match GANSU's convention.
Prints per-component gradients via finite differences.
"""

import numpy as np
from pyscf import gto, scf, grad


def run_gradient(name, atoms_str, basis_name):
    """Run RHF gradient calculation with PySCF and print per-component gradients."""
    mol = gto.M(atom=atoms_str, basis=basis_name, cart=True, unit='Angstrom')
    mf = scf.RHF(mol)
    mf.kernel()

    print(f"\n{'='*60}")
    print(f"  {name} / {basis_name}")
    print(f"{'='*60}")
    print(f"  RHF Energy: {mf.e_tot:.10f} Hartree")

    D = mf.make_rdm1()
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    occ_idx = mo_occ > 0
    C_occ = mo_coeff[:, occ_idx]
    e_occ = mo_energy[occ_idx]
    W = 2.0 * C_occ @ np.diag(e_occ) @ C_occ.T

    # Total gradient
    g = mf.nuc_grad_method()
    grad_total = g.kernel()

    print(f"\n  Total Gradient (Hartree/Bohr):")
    print(f"  {'Atom':>6s} {'dE/dx':>18s} {'dE/dy':>18s} {'dE/dz':>18s}")
    for i in range(mol.natm):
        print(f"  {i:>6d} {grad_total[i,0]:>18.10f} {grad_total[i,1]:>18.10f} {grad_total[i,2]:>18.10f}")

    # Nuclear repulsion gradient
    gobj = grad.rhf.Gradients(mf)
    grad_nuc = gobj.grad_nuc(mol)
    print(f"\n  Nuclear repulsion gradient:")
    for i in range(mol.natm):
        print(f"  {i:>6d} {grad_nuc[i,0]:>18.10f} {grad_nuc[i,1]:>18.10f} {grad_nuc[i,2]:>18.10f}")

    # Per-component gradients via finite differences
    dx = 1e-5  # Bohr
    coords = mol.atom_coords().copy()
    natm = mol.natm

    grad_s_fd = np.zeros((natm, 3))
    grad_t_fd = np.zeros((natm, 3))
    grad_v_fd = np.zeros((natm, 3))
    grad_2e_fd = np.zeros((natm, 3))

    for atm_id in range(natm):
        for axis in range(3):
            coords_p = coords.copy()
            coords_p[atm_id, axis] += dx
            mol_p = mol.copy()
            mol_p.set_geom_(coords_p, unit='Bohr')
            mol_p.build()

            coords_m = coords.copy()
            coords_m[atm_id, axis] -= dx
            mol_m = mol.copy()
            mol_m.set_geom_(coords_m, unit='Bohr')
            mol_m.build()

            S_p = mol_p.intor('int1e_ovlp')
            S_m = mol_m.intor('int1e_ovlp')
            dS = (S_p - S_m) / (2 * dx)
            grad_s_fd[atm_id, axis] = np.einsum('ij,ij->', W, dS)

            T_p = mol_p.intor('int1e_kin')
            T_m = mol_m.intor('int1e_kin')
            dT = (T_p - T_m) / (2 * dx)
            grad_t_fd[atm_id, axis] = np.einsum('ij,ij->', D, dT)

            V_p = mol_p.intor('int1e_nuc')
            V_m = mol_m.intor('int1e_nuc')
            dV = (V_p - V_m) / (2 * dx)
            grad_v_fd[atm_id, axis] = np.einsum('ij,ij->', D, dV)

            eri_p = mol_p.intor('int2e')
            eri_m = mol_m.intor('int2e')
            deri = (eri_p - eri_m) / (2 * dx)
            J_grad = 0.5 * np.einsum('ij,kl,ijkl->', D, D, deri)
            K_grad = 0.25 * np.einsum('ij,kl,ikjl->', D, D, deri)
            grad_2e_fd[atm_id, axis] = J_grad - K_grad

    print(f"\n  Overlap gradient (W*dS/dR):")
    for i in range(natm):
        print(f"  {i:>6d} {grad_s_fd[i,0]:>18.10f} {grad_s_fd[i,1]:>18.10f} {grad_s_fd[i,2]:>18.10f}")

    print(f"\n  Kinetic gradient (D*dT/dR):")
    for i in range(natm):
        print(f"  {i:>6d} {grad_t_fd[i,0]:>18.10f} {grad_t_fd[i,1]:>18.10f} {grad_t_fd[i,2]:>18.10f}")

    print(f"\n  Nuclear attraction gradient (D*dV/dR):")
    for i in range(natm):
        print(f"  {i:>6d} {grad_v_fd[i,0]:>18.10f} {grad_v_fd[i,1]:>18.10f} {grad_v_fd[i,2]:>18.10f}")

    print(f"\n  Two-electron gradient (0.5*D*dG/dR):")
    for i in range(natm):
        print(f"  {i:>6d} {grad_2e_fd[i,0]:>18.10f} {grad_2e_fd[i,1]:>18.10f} {grad_2e_fd[i,2]:>18.10f}")

    # Reconstruct: Total = Nuc + T + V + 2e - W*dS
    grad_sum = grad_nuc + grad_t_fd + grad_v_fd + grad_2e_fd - grad_s_fd
    print(f"\n  Sum (Nuc + T + V + 2e - W*dS):")
    for i in range(natm):
        print(f"  {i:>6d} {grad_sum[i,0]:>18.10f} {grad_sum[i,1]:>18.10f} {grad_sum[i,2]:>18.10f}")

    return mf.e_tot, grad_total


if __name__ == "__main__":
    # H2 / STO-3G at 0.7122 Angstrom (matching GANSU's H2.xyz)
    run_gradient("H2 (0.7122A)", "H 0 0 0; H 0 0 0.7122", "sto-3g")

    # H2O / STO-3G (matching GANSU's H2O.xyz)
    run_gradient("H2O", """
        O  0.000  0.000  0.127
        H  0.000  0.758 -0.509
        H  0.000 -0.758 -0.509
    """, "sto-3g")

    # H2O / cc-pVDZ (includes d-type basis functions, Cartesian 6d)
    run_gradient("H2O", """
        O  0.000  0.000  0.127
        H  0.000  0.758 -0.509
        H  0.000 -0.758 -0.509
    """, "cc-pvdz")
