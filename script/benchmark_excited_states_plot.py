#!/usr/bin/env python3
"""
Compare and plot excited state benchmark results.

Reads CSV files from benchmark_results/ and generates:
  1. Accuracy table: excitation energies comparison
  2. Scaling plot: wall time vs number of basis functions

Usage:
  wsl python3 ../script/benchmark_excited_states_plot.py
"""

import os
import glob
import csv
import sys

OUTDIR = "../benchmark_results"

def load_results():
    """Load all CSV result files and merge."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUTDIR, "excited_states_*.csv"))):
        with open(f) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                rows.append(row)
    return rows

def print_accuracy_table(rows):
    """Print accuracy comparison for H2O."""
    print("=" * 80)
    print("  ACCURACY COMPARISON (H2O)")
    print("=" * 80)

    h2o = [r for r in rows if r["molecule"] == "H2O" and r.get("time_total_sec", r.get("time_sec", "")) != "FAILED"]

    for basis in ["sto-3g", "cc-pvdz"]:
        for method in ["cis", "adc2", "eom_mp2", "eom_ccsd"]:
            subset = [r for r in h2o if r["basis"] == basis and r["method"] == method
                      and r.get("solver", "default") == "default"]
            if not subset:
                continue
            print(f"\n  {method.upper()} / {basis}:")
            for r in subset:
                energies = r.get("excitation_energies_eV", "")
                print(f"    {r['mode']:20s}  E = {energies:40s}  ({r['time_sec']}s)")

def print_solver_table(rows):
    """Print solver comparison table."""
    print("\n" + "=" * 80)
    print("  SOLVER COMPARISON (Schur vs Full)")
    print("=" * 80)

    for method in ["adc2", "eom_mp2", "eom_cc2"]:
        method_rows = [r for r in rows if r["method"] == method and r.get("time_total_sec", r.get("time_sec", "")) != "FAILED"
                       and r.get("solver", "default") != "default"]
        if not method_rows:
            continue

        for basis in ["sto-3g", "cc-pvdz"]:
            basis_rows = [r for r in method_rows if r["basis"] == basis]
            if not basis_rows:
                continue

            solvers = sorted(set(r.get("solver", "") for r in basis_rows))
            molecules = sorted(set(r["molecule"] for r in basis_rows),
                               key=lambda m: int([r for r in basis_rows if r["molecule"]==m][0].get("nbasis", 0) or 0))

            print(f"\n  {method.upper()} / {basis}:")
            print(f"    {'Molecule':15s} {'nao':>5s}  ", end="")
            for s in solvers:
                print(f"{s:>15s}", end="")
            print()
            print("    " + "-" * (22 + 15 * len(solvers)))

            for mol_name in molecules:
                mol_rows = [r for r in basis_rows if r["molecule"] == mol_name]
                if not mol_rows:
                    continue
                nbasis = mol_rows[0].get("nbasis", "?")
                print(f"    {mol_name:15s} {str(nbasis):>5s}  ", end="")
                for s in solvers:
                    match = [r for r in mol_rows if r.get("solver") == s]
                    if match:
                        print(f"{match[0]['time_sec']:>15s}", end="")
                    else:
                        print(f"{'---':>15s}", end="")
                print()

def print_scaling_table(rows):
    """Print scaling table: time vs basis size."""
    print("\n" + "=" * 80)
    print("  SCALING (wall time in seconds)")
    print("=" * 80)

    for method in ["cis", "adc2", "eom_mp2", "eom_ccsd"]:
        method_rows = [r for r in rows if r["method"] == method and r.get("time_total_sec", r.get("time_sec", "")) != "FAILED"
                       and r.get("solver", "default") == "default"]
        if not method_rows:
            continue

        for basis in ["sto-3g", "cc-pvdz"]:
            basis_rows = [r for r in method_rows if r["basis"] == basis]
            if not basis_rows:
                continue

            print(f"\n  {method.upper()} / {basis}:")
            print(f"    {'Molecule':15s} {'nao':>5s}  ", end="")
            modes = sorted(set(r["mode"] for r in basis_rows))
            for m in modes:
                print(f"{m:>15s}", end="")
            print()
            print("    " + "-" * (22 + 15 * len(modes)))

            molecules = []
            for r in basis_rows:
                key = (r["molecule"], int(r.get("nbasis", 0) or 0))
                if key not in molecules:
                    molecules.append(key)
            molecules.sort(key=lambda x: x[1])

            for mol_name, nbasis in molecules:
                print(f"    {mol_name:15s} {nbasis:5d}  ", end="")
                for m in modes:
                    match = [r for r in basis_rows if r["molecule"] == mol_name and r["mode"] == m]
                    if match:
                        t = match[0]["time_sec"]
                        print(f"{t:>15s}", end="")
                    else:
                        print(f"{'---':>15s}", end="")
                print()

def main():
    rows = load_results()
    if not rows:
        print(f"No CSV files found in {OUTDIR}/")
        print("Run the benchmark scripts first:")
        print("  bash ../script/benchmark_excited_states.sh gpu")
        print("  bash ../script/benchmark_excited_states.sh cpu")
        print("  wsl python3 ../script/benchmark_excited_states_pyscf.py cpu")
        sys.exit(1)

    print(f"Loaded {len(rows)} data points from {OUTDIR}/\n")
    print_accuracy_table(rows)
    print_solver_table(rows)
    print_scaling_table(rows)
    print()

if __name__ == "__main__":
    main()
