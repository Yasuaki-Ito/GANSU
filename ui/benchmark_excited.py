#!/usr/bin/env python3
"""
Benchmark: GANSU GPU excited state calculations (direct Python API, no server).
Outputs CSV with molecule, basis, method, solver, time, energy results.

Usage: python benchmark_excited.py [output.csv]
"""

import sys
import os
import time
import csv
import traceback

OUTFILE = sys.argv[1] if len(sys.argv) > 1 else "benchmark_gansu_excited.csv"

# Setup gansu
GANSU_PATH = os.environ.get("GANSU_PATH", os.path.expanduser("~/GANSU"))
gansu_python = os.path.join(GANSU_PATH, "python")
if gansu_python not in sys.path:
    sys.path.insert(0, gansu_python)
if "GANSU_LIB" not in os.environ:
    for lib in ["build/libgansu.so", "build/libgansu.dylib"]:
        p = os.path.join(GANSU_PATH, lib)
        if os.path.exists(p):
            os.environ["GANSU_LIB"] = p
            break

import gansu
gansu.init()

# ── Molecules ──
XYZ_DIR = os.path.join(GANSU_PATH, "xyz")
MOLECULES = {}
for name in ["H2O", "Benzene", "Naphthalene", "Anthracene", "Tetracene"]:
    fpath = os.path.join(XYZ_DIR, f"{name}.xyz")
    if os.path.exists(fpath):
        MOLECULES[name] = fpath
    else:
        print(f"WARNING: {fpath} not found, skipping {name}")

BASIS_SETS = ["sto-3g", "cc-pvdz"]

METHODS = [
    ("CIS", "cis", "auto"),
    ("ADC(2) static", "adc2", "schur_static"),
    ("ADC(2) omega", "adc2", "schur_omega"),
    ("ADC(2) full", "adc2", "full"),
    ("EOM-MP2 static", "eom_mp2", "schur_static"),
    ("EOM-MP2 omega", "eom_mp2", "schur_omega"),
    ("EOM-MP2 full", "eom_mp2", "full"),
    ("EOM-CC2 static", "eom_cc2", "schur_static"),
    ("EOM-CC2 omega", "eom_cc2", "schur_omega"),
    ("EOM-CC2 full", "eom_cc2", "full"),
    ("EOM-CCSD", "eom_ccsd", "auto"),
]

N_STATES = 3
TIMEOUT = 300  # 5 min


def run_calc(xyz_path, basis, post_hf, solver, n_states):
    """Run a single calculation, return (result_dict, elapsed_seconds)."""
    # Build solver param name
    solver_params = {}
    if solver != "auto":
        solver_key = {
            "adc2": "adc2_solver", "adc2x": "adc2_solver",
            "eom_mp2": "eom_mp2_solver", "eom_cc2": "eom_cc2_solver",
        }.get(post_hf)
        if solver_key:
            solver_params[solver_key] = solver

    mol = gansu.Molecule(
        xyz_path, basis=basis,
        initial_guess="sad",
        n_excited_states=str(n_states),
        **solver_params,
    )

    t0 = time.time()
    r = mol.run(method="RHF", post_hf=post_hf, quiet=True)
    elapsed = time.time() - t0

    result = {
        "hf_energy": r.total_energy,
        "post_hf_correction": r.post_hf_energy if post_hf not in ("cis",) else 0.0,
        "n_basis": r.num_basis,
    }

    # Parse excited states from report
    es_report = r.excited_state_report
    if es_report:
        import re
        es_list = []
        for line in es_report.split('\n'):
            m = re.match(
                r'\s*(\d+)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+([-+]?\d+\.?\d*)\s+(.*)',
                line)
            if m:
                es_list.append({
                    "state": int(m.group(1)),
                    "energy_ha": float(m.group(2)),
                    "energy_ev": float(m.group(3)),
                    "osc_strength": float(m.group(4)),
                    "transitions": m.group(5).strip(),
                })
        result["excited_states"] = es_list

    # Explicitly delete to free GPU memory
    del r, mol

    return result, elapsed


# ── Run benchmark ──
rows = []
total = len(MOLECULES) * len(BASIS_SETS) * len(METHODS)
count = 0

# Warm up
print("Warming up GPU...")
try:
    run_calc(MOLECULES["H2O"], "sto-3g", "cis", "auto", 1)
except Exception:
    pass

for mol_name, xyz_path in MOLECULES.items():
    for basis in BASIS_SETS:
        for method_label, post_hf, solver in METHODS:
            count += 1
            label = f"[{count}/{total}] {mol_name}/{basis}/{method_label}"
            print(f"{label}...", end=" ", flush=True)

            row = {
                "molecule": mol_name,
                "basis": basis,
                "method": method_label,
                "post_hf": post_hf,
                "solver": solver,
                "n_states": N_STATES,
                "n_basis": "",
                "time_s": 0,
                "status": "error",
                "error": "",
                "hf_energy": "",
                "post_hf_correction": "",
            }

            try:
                result, elapsed = run_calc(xyz_path, basis, post_hf, solver, N_STATES)
                row["time_s"] = round(elapsed, 3)
                row["status"] = "ok"
                row["hf_energy"] = result["hf_energy"]
                row["post_hf_correction"] = result.get("post_hf_correction", "")
                row["n_basis"] = result.get("n_basis", "")

                es = result.get("excited_states", [])
                for i, s in enumerate(es):
                    row[f"E{i+1}_ha"] = s["energy_ha"]
                    row[f"E{i+1}_ev"] = s["energy_ev"]
                    row[f"f{i+1}"] = s["osc_strength"]

                print(f"ok ({elapsed:.2f}s, nbasis={result.get('n_basis','')})")

            except Exception as e:
                row["error"] = str(e)[:100]
                print(f"ERROR: {e}")
                traceback.print_exc()

            rows.append(row)

# ── Write CSV ──
if rows:
    fieldnames = list(rows[0].keys())
    for i in range(N_STATES):
        for col in [f"E{i+1}_ha", f"E{i+1}_ev", f"f{i+1}"]:
            if col not in fieldnames:
                fieldnames.append(col)

    with open(OUTFILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTFILE}")
    ok = sum(1 for r in rows if r['status'] == 'ok')
    err = sum(1 for r in rows if r['status'] == 'error')
    print(f"Total: {ok} ok, {err} errors")

gansu.finalize()
