#!/usr/bin/env python3
"""
Benchmark: Excited state calculations (PySCF CPU / GPU4PySCF)

Matches the GANSU benchmark for direct comparison.
Outputs CSV with timing and excitation energies.

Usage:
  wsl python3 ../script/benchmark_excited_states_pyscf.py cpu
  wsl python3 ../script/benchmark_excited_states_pyscf.py gpu4pyscf
"""

import sys
import os
import time
import csv
import numpy as np

# =============================================================================
# Configuration
# =============================================================================
NSTATES = 3

MODE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
OUTDIR = "../benchmark_results"
os.makedirs(OUTDIR, exist_ok=True)
RESULT_FILE = os.path.join(OUTDIR, f"excited_states_pyscf_{MODE}_{TIMESTAMP}.csv")

# =============================================================================
# Read GANSU xyz file (Bohr) and convert to Angstrom for PySCF
# =============================================================================
def read_xyz(filepath):
    """Read standard xyz file (coordinates in Angstrom) and return PySCF atom string."""
    with open(filepath) as f:
        natoms = int(f.readline().strip())
        f.readline()  # comment
        atoms = []
        for _ in range(natoms):
            parts = f.readline().split()
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return "; ".join(atoms), natoms

# =============================================================================
# Benchmark runners
# =============================================================================
def run_pyscf_cis(mol, nstates=NSTATES):
    """CIS via TDA. Returns (total_time, posthf_time, energies)."""
    from pyscf import scf, tdscf
    t0 = time.time()
    mf = scf.RHF(mol).run(verbose=0)
    t1 = time.time()
    td = tdscf.TDA(mf)
    td.nstates = nstates
    td.run(verbose=0)
    t2 = time.time()
    energies_ev = [e * 27.211386 for e in td.e]
    return t2 - t0, t2 - t1, energies_ev[:nstates]

def run_pyscf_adc2(mol, nstates=NSTATES):
    """ADC(2) via pyscf.adc. Returns (total_time, posthf_time, energies)."""
    from pyscf import scf, adc
    t0 = time.time()
    mf = scf.RHF(mol).run(verbose=0)
    t1 = time.time()
    myadc = adc.ADC(mf)
    myadc.method_type = "ee"
    myadc.method = "adc(2)"
    result = myadc.kernel(nroots=nstates)
    t2 = time.time()
    energies = result[0] if isinstance(result, tuple) else result
    energies_ev = [e * 27.211386 for e in energies]
    return t2 - t0, t2 - t1, energies_ev[:nstates]

def run_pyscf_eom_ccsd(mol, nstates=NSTATES):
    """EOM-CCSD via pyscf.cc. Returns (total_time, posthf_time, energies)."""
    from pyscf import scf, cc
    t0 = time.time()
    mf = scf.RHF(mol).run(verbose=0)
    t1 = time.time()
    mycc = cc.CCSD(mf).run(verbose=0)
    e_ee, _ = mycc.eomee_ccsd_singlet(nroots=nstates)
    t2 = time.time()
    energies_ev = [e * 27.211386 for e in e_ee]
    return t2 - t0, t2 - t1, energies_ev[:nstates]

def run_gpu4pyscf_cis(mol, nstates=NSTATES):
    """CIS via GPU4PySCF TDA. Returns (total_time, posthf_time, energies)."""
    try:
        from gpu4pyscf import tdscf as gpu_tdscf
        from gpu4pyscf import scf as gpu_scf
    except ImportError:
        print("    GPU4PySCF not available, skipping")
        return None, None, None
    t0 = time.time()
    mf = gpu_scf.RHF(mol).run(verbose=0)
    t1 = time.time()
    td = gpu_tdscf.TDA(mf)
    td.nstates = nstates
    td.run(verbose=0)
    t2 = time.time()
    energies_ev = [e * 27.211386 for e in td.e]
    return t2 - t0, t2 - t1, energies_ev[:nstates]

# =============================================================================
# Method dispatch
# =============================================================================
METHODS = {
    "cis":      {"cpu": run_pyscf_cis,      "gpu4pyscf": run_gpu4pyscf_cis},
    "adc2":     {"cpu": run_pyscf_adc2,      "gpu4pyscf": None},
    "eom_ccsd": {"cpu": run_pyscf_eom_ccsd,  "gpu4pyscf": None},
}

# =============================================================================
# Test cases
# =============================================================================
XYZ_DIR = "../xyz"

# (label, xyz_path, basis, methods_to_run)
test_cases = [
    # Part 1: Accuracy (H2O)
    ("H2O", f"{XYZ_DIR}/H2O.xyz", "sto-3g", ["cis", "adc2", "eom_ccsd"]),
    ("H2O", f"{XYZ_DIR}/H2O.xyz", "cc-pvdz", ["cis", "adc2", "eom_ccsd"]),
    # Part 2: Scaling - CIS
    ("Benzene",     f"{XYZ_DIR}/Benzene.xyz",     "sto-3g",  ["cis"]),
    ("Benzene",     f"{XYZ_DIR}/Benzene.xyz",     "cc-pvdz", ["cis"]),
    ("Naphthalene", f"{XYZ_DIR}/Naphthalene.xyz", "sto-3g",  ["cis"]),
    ("Naphthalene", f"{XYZ_DIR}/Naphthalene.xyz", "cc-pvdz", ["cis"]),
    ("Anthracene",  f"{XYZ_DIR}/Anthracene.xyz",  "sto-3g",  ["cis"]),
    ("Anthracene",  f"{XYZ_DIR}/Anthracene.xyz",  "cc-pvdz", ["cis"]),
    ("Tetracene",   f"{XYZ_DIR}/Tetracene.xyz",   "sto-3g",  ["cis"]),
    ("Tetracene",   f"{XYZ_DIR}/Tetracene.xyz",   "cc-pvdz", ["cis"]),
    ("Pentacene",   f"{XYZ_DIR}/Pentacene.xyz",   "sto-3g",  ["cis"]),
    ("Pentacene",   f"{XYZ_DIR}/Pentacene.xyz",   "cc-pvdz", ["cis"]),
    # Part 2: Scaling - ADC(2)
    ("Benzene",     f"{XYZ_DIR}/Benzene.xyz",     "sto-3g",  ["adc2"]),
    ("Benzene",     f"{XYZ_DIR}/Benzene.xyz",     "cc-pvdz", ["adc2"]),
    ("Naphthalene", f"{XYZ_DIR}/Naphthalene.xyz", "sto-3g",  ["adc2"]),
    ("Naphthalene", f"{XYZ_DIR}/Naphthalene.xyz", "cc-pvdz", ["adc2"]),
    ("Anthracene",  f"{XYZ_DIR}/Anthracene.xyz",  "sto-3g",  ["adc2"]),
    ("Tetracene",   f"{XYZ_DIR}/Tetracene.xyz",   "sto-3g",  ["adc2"]),
    # Part 2: Scaling - EOM-CCSD
    ("Benzene",     f"{XYZ_DIR}/Benzene.xyz",     "sto-3g",  ["eom_ccsd"]),
    ("Naphthalene", f"{XYZ_DIR}/Naphthalene.xyz", "sto-3g",  ["eom_ccsd"]),
]

# =============================================================================
# Main
# =============================================================================
def main():
    from pyscf import gto

    with open(RESULT_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["mode", "molecule", "basis", "method", "solver", "natoms", "nbasis",
                         "n_states", "time_total_sec", "time_posthf_sec", "excitation_energies_eV"])

        for mol_name, xyz_path, basis, methods in test_cases:
            if not os.path.exists(xyz_path):
                print(f"  SKIP {mol_name} ({xyz_path} not found)")
                continue

            atom_str, natoms = read_xyz(xyz_path)
            mol = gto.M(atom=atom_str, basis=basis, verbose=0)
            nbasis = mol.nao

            for method in methods:
                runner = METHODS.get(method, {}).get(MODE)
                if runner is None:
                    print(f"  SKIP {mol_name}/{basis}/{method} (not available for {MODE})")
                    continue

                print(f"  {mol_name} / {basis} / {method} ...", end="", flush=True)
                try:
                    total_time, posthf_time, energies = runner(mol, NSTATES)
                    if total_time is None:
                        continue
                    e_str = ";".join(f"{e:.4f}" for e in energies)
                    writer.writerow([f"pyscf_{MODE}", mol_name, basis, method, "default",
                                     natoms, nbasis, NSTATES,
                                     f"{total_time:.3f}", f"{posthf_time:.3f}", e_str])
                    print(f" total={total_time:.1f}s, posthf={posthf_time:.1f}s, E = {e_str} eV")
                except Exception as ex:
                    print(f" FAILED: {ex}")
                    writer.writerow([f"pyscf_{MODE}", mol_name, basis, method, "default",
                                     natoms, nbasis, NSTATES, "FAILED", "", ""])

    print(f"\nResults saved to: {RESULT_FILE}")

if __name__ == "__main__":
    main()
