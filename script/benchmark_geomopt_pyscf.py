#!/usr/bin/env python3
"""
Benchmark: PySCF vs GANSU geometry optimization performance comparison.
Uses Cartesian Gaussians (cart=True) to match GANSU's convention.

Usage:
    python benchmark_geomopt_pyscf.py
"""

import time
import numpy as np
from pyscf import gto, scf, grad
from scipy.optimize import minimize


def read_xyz(filepath):
    """Read xyz file and return atom string for PySCF."""
    with open(filepath) as f:
        lines = f.readlines()
    natom = int(lines[0].strip())
    atoms = []
    for line in lines[2:2+natom]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}")
    return "; ".join(atoms)


def run_pyscf_geomopt(name, atoms_str, basis_name, max_steps=50, conv_tol=1e-4):
    """Run PySCF RHF geometry optimization and return timing/results."""
    print(f"\n{'='*70}")
    print(f"  {name} / {basis_name}")
    print(f"{'='*70}")

    # Build molecule
    mol = gto.M(atom=atoms_str, basis=basis_name, cart=True, unit='Angstrom', verbose=0)
    print(f"  Atoms: {mol.natm},  Basis functions: {mol.nao_nr()}")

    # --- Benchmark: Single-point SCF ---
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    t0 = time.perf_counter()
    mf.kernel()
    t_scf = time.perf_counter() - t0
    print(f"  Initial RHF energy: {mf.e_tot:.10f} Hartree")
    print(f"  SCF time: {t_scf*1000:.1f} ms")

    # --- Benchmark: Single gradient ---
    g = mf.nuc_grad_method()
    t0 = time.perf_counter()
    grad_val = g.kernel()
    t_grad = time.perf_counter() - t0
    max_grad = np.max(np.abs(grad_val))
    print(f"  Gradient time: {t_grad*1000:.1f} ms")
    print(f"  Max gradient: {max_grad:.6e} Hartree/Bohr")

    # --- Benchmark: Full geometry optimization (BFGS via scipy) ---
    mol_opt = gto.M(atom=atoms_str, basis=basis_name, cart=True, unit='Angstrom', verbose=0)
    coords0 = mol_opt.atom_coords(unit='Bohr').flatten()
    opt_nsteps = [0]

    def energy_and_grad(x):
        coords_2d = x.reshape(-1, 3)
        mol_opt.set_geom_(coords_2d, unit='Bohr')
        mol_opt.build()
        mf_tmp = scf.RHF(mol_opt)
        mf_tmp.conv_tol = 1e-10
        mf_tmp.verbose = 0
        mf_tmp.kernel()
        g_tmp = mf_tmp.nuc_grad_method()
        g_tmp.verbose = 0
        gv = g_tmp.kernel()
        opt_nsteps[0] += 1
        return mf_tmp.e_tot, gv.flatten()

    t0 = time.perf_counter()
    res = minimize(energy_and_grad, coords0, method='BFGS', jac=True,
                   options={'maxiter': max_steps, 'gtol': conv_tol})
    t_opt = time.perf_counter() - t0
    converged = res.success
    final_energy = res.fun

    print(f"\n  Optimization time: {t_opt*1000:.1f} ms")
    print(f"  Final energy: {final_energy:.10f} Hartree")
    print(f"  Steps: {opt_nsteps[0]}")
    print(f"  Converged: {converged}")

    # Print optimized geometry
    if converged:
        coords_final = res.x.reshape(-1, 3)
        print(f"\n  Optimized Geometry (Bohr):")
        for i in range(mol_opt.natm):
            sym = mol_opt.atom_symbol(i)
            print(f"    {sym:>2s} {coords_final[i,0]:>16.10f} {coords_final[i,1]:>16.10f} {coords_final[i,2]:>16.10f}")

    return {
        'name': name,
        'basis': basis_name,
        'natom': mol.natm,
        'nbasis': mol.nao_nr(),
        'energy_init': mf.e_tot,
        'energy_final': final_energy,
        'scf_time_ms': t_scf * 1000,
        'grad_time_ms': t_grad * 1000,
        'opt_time_ms': t_opt * 1000,
        'converged': converged,
    }


def main():
    print("=" * 70)
    print("  PySCF Geometry Optimization Benchmark")
    print("  (Cartesian Gaussians, RHF, for comparison with GANSU)")
    print("=" * 70)

    benchmarks = [
        # (name, atoms_str, basis)
        ("H2O_distorted", """
            O  0.000  0.000  0.200
            H  0.000  0.900 -0.400
            H  0.000 -0.650 -0.600
        """, "sto-3g"),

        ("H2O", """
            O  0.000  0.000  0.127
            H  0.000  0.758 -0.509
            H  0.000 -0.758 -0.509
        """, "sto-3g"),

        ("H2O", """
            O  0.000  0.000  0.127
            H  0.000  0.758 -0.509
            H  0.000 -0.758 -0.509
        """, "cc-pvdz"),

        ("CH4", """
            C  0.000  0.000  0.000
            H  0.626  0.626  0.626
            H -0.626 -0.626  0.626
            H -0.626  0.626 -0.626
            H  0.626 -0.626 -0.626
        """, "sto-3g"),

        ("CH4", """
            C  0.000  0.000  0.000
            H  0.626  0.626  0.626
            H -0.626 -0.626  0.626
            H -0.626  0.626 -0.626
            H  0.626 -0.626 -0.626
        """, "cc-pvdz"),

        ("Formaldehyde", """
            O  0.000  0.000  0.683
            C  0.000  0.000 -0.534
            H  0.000  0.926 -1.129
            H  0.000 -0.926 -1.129
        """, "sto-3g"),

        ("Formaldehyde", """
            O  0.000  0.000  0.683
            C  0.000  0.000 -0.534
            H  0.000  0.926 -1.129
            H  0.000 -0.926 -1.129
        """, "cc-pvdz"),

        ("Ethanol", """
            C  1.187 -0.429  0.000
            C  0.000  0.555  0.000
            O -1.218 -0.204  0.000
            H -1.930  0.485  0.000
            H  2.126  0.116  0.000
            H  1.151 -1.062  0.881
            H  1.151 -1.062 -0.881
            H  0.063  1.203  0.883
            H  0.063  1.203 -0.883
        """, "sto-3g"),

        ("Ethanol", """
            C  1.187 -0.429  0.000
            C  0.000  0.555  0.000
            O -1.218 -0.204  0.000
            H -1.930  0.485  0.000
            H  2.126  0.116  0.000
            H  1.151 -1.062  0.881
            H  1.151 -1.062 -0.881
            H  0.063  1.203  0.883
            H  0.063  1.203 -0.883
        """, "cc-pvdz"),

        ("NH3", """
            N  0.000  0.000  0.116
            H  0.000  0.939 -0.272
            H  0.813 -0.470 -0.272
            H -0.813 -0.470 -0.272
        """, "sto-3g"),

        ("Benzene", """
            C  1.398  0.000  0.000
            C  0.699  1.210  0.000
            C -0.699  1.210  0.000
            C -1.398  0.000  0.000
            C -0.699 -1.210  0.000
            C  0.699 -1.210  0.000
            H  2.478  0.000  0.000
            H  1.239  2.146  0.000
            H -1.239  2.146  0.000
            H -2.478  0.000  0.000
            H -1.239 -2.146  0.000
            H  1.239 -2.146  0.000
        """, "sto-3g"),

        ("Naphthalene", """
            C  0.000  0.000  1.225
            C  0.000  1.242  0.701
            C  0.000  1.242 -0.701
            C  0.000  0.000 -1.225
            C  0.000 -1.242 -0.701
            C  0.000 -1.242  0.701
            C  0.000  2.428  1.399
            C  0.000  3.610  0.695
            C  0.000  3.610 -0.695
            C  0.000  2.428 -1.399
            H  0.000  0.000  2.309
            H  0.000  0.000 -2.309
            H  0.000 -2.157 -1.207
            H  0.000 -2.157  1.207
            H  0.000  2.428  2.483
            H  0.000  4.546  1.207
            H  0.000  4.546 -1.207
            H  0.000  2.428 -2.483
        """, "sto-3g"),
    ]

    results = []
    for name, atoms_str, basis in benchmarks:
        try:
            r = run_pyscf_geomopt(name, atoms_str, basis)
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary table
    print(f"\n\n{'='*90}")
    print(f"  Summary: PySCF Geometry Optimization Benchmark")
    print(f"{'='*90}")
    print(f"  {'Molecule':<18s} {'Basis':<10s} {'Atoms':>5s} {'Nbas':>5s} "
          f"{'SCF(ms)':>9s} {'Grad(ms)':>9s} {'Opt(ms)':>10s} {'Final E':>16s}")
    print(f"  {'-'*86}")
    for r in results:
        print(f"  {r['name']:<18s} {r['basis']:<10s} {r['natom']:>5d} {r['nbasis']:>5d} "
              f"{r['scf_time_ms']:>9.1f} {r['grad_time_ms']:>9.1f} {r['opt_time_ms']:>10.1f} "
              f"{r['energy_final']:>16.10f}")

    print(f"\n  Note: Run GANSU with the same molecules/basis sets for comparison:")
    print(f"  ./HF_main -x ../xyz/<mol>.xyz -g ../basis/<basis>.gbs -m RHF --geometry_optimization 1")
    print()


if __name__ == "__main__":
    main()
