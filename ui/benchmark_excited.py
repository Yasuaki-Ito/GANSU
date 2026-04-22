#!/usr/bin/env python3
"""
Benchmark: GANSU GPU excited state calculations.
Outputs CSV with molecule, basis, method, solver, n_states, time, energy results.

Usage: python benchmark_excited.py [base_url] [output.csv]
"""

import sys
import json
import time
import csv
import urllib.request
import traceback

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "benchmark_gansu_excited.csv"

# ── Molecules ──
MOLECULES = {
    "H2O": """3
H2O
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200""",

    "Benzene": """12
Benzene
C  0.000000  1.396000  0.000000
C  1.209031  0.698000  0.000000
C  1.209031 -0.698000  0.000000
C  0.000000 -1.396000  0.000000
C -1.209031 -0.698000  0.000000
C -1.209031  0.698000  0.000000
H  0.000000  2.484000  0.000000
H  2.151421  1.242000  0.000000
H  2.151421 -1.242000  0.000000
H  0.000000 -2.484000  0.000000
H -2.151421 -1.242000  0.000000
H -2.151421  1.242000  0.000000""",
}

# Load from xyz files for larger molecules
import os
GANSU_XYZ = os.environ.get("GANSU_PATH", os.path.expanduser("~/GANSU")) + "/xyz"
for name in ["Naphthalene", "Anthracene", "Tetracene"]:
    fpath = os.path.join(GANSU_XYZ, f"{name}.xyz")
    if os.path.exists(fpath):
        MOLECULES[name] = open(fpath).read()
    else:
        print(f"WARNING: {fpath} not found, skipping {name}")

BASIS_SETS = ["sto-3g", "cc-pvdz"]

# method: (post_hf, solver)
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


def run_calc(xyz, basis, post_hf, solver, n_states, timeout=1200):
    data = json.dumps({
        "xyz_text": xyz,
        "basis": basis,
        "method": "RHF",
        "post_hf_method": post_hf,
        "excited_solver": solver,
        "n_excited_states": n_states,
        "initial_guess": "sad",
        "timeout": timeout,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/run/inprocess",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout + 30) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.time() - t0
        return result, elapsed
    except Exception as e:
        return {"ok": False, "error": str(e)}, time.time() - t0


# ── Run benchmark ──
rows = []
total = len(MOLECULES) * len(BASIS_SETS) * len(METHODS)
count = 0

# Warm up GPU
print("Warming up GPU...")
run_calc(MOLECULES["H2O"], "sto-3g", "cis", "auto", 1, timeout=60)

for mol_name, xyz in MOLECULES.items():
    for basis in BASIS_SETS:
        for method_label, post_hf, solver in METHODS:
            count += 1
            label = f"[{count}/{total}] {mol_name}/{basis}/{method_label}"
            print(f"{label}...", end=" ", flush=True)

            result, elapsed = run_calc(xyz, basis, post_hf, solver, N_STATES)

            row = {
                "molecule": mol_name,
                "basis": basis,
                "method": method_label,
                "post_hf": post_hf,
                "solver": solver,
                "n_states": N_STATES,
                "time_s": round(elapsed, 3),
                "status": "ok" if result.get("ok") else "error",
                "error": result.get("error", "")[:100] if not result.get("ok") else "",
                "hf_energy": "",
                "post_hf_correction": "",
            }

            if result.get("ok"):
                row["hf_energy"] = result.get("summary", {}).get("total_energy", "")
                if result.get("post_hf"):
                    row["post_hf_correction"] = result["post_hf"].get("correction", "")

                es = result.get("excited_states", [])
                for i, s in enumerate(es):
                    row[f"E{i+1}_ha"] = s["energy_ha"]
                    row[f"E{i+1}_ev"] = s["energy_ev"]
                    row[f"f{i+1}"] = s["osc_strength"]

            rows.append(row)
            status = "ok" if result.get("ok") else f"ERROR: {result.get('error', '?')[:50]}"
            print(f"{status} ({elapsed:.2f}s)")

# ── Write CSV ──
if rows:
    fieldnames = list(rows[0].keys())
    # Add E/f columns for all states
    for i in range(N_STATES):
        for col in [f"E{i+1}_ha", f"E{i+1}_ev", f"f{i+1}"]:
            if col not in fieldnames:
                fieldnames.append(col)

    with open(OUTFILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTFILE}")
    print(f"Total: {sum(1 for r in rows if r['status']=='ok')} ok, {sum(1 for r in rows if r['status']=='error')} errors")
