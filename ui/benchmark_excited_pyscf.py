#!/usr/bin/env python3
"""
Benchmark: PySCF excited state calculations (CPU reference).
Outputs CSV with molecule, basis, method, time, energy results.

Usage: python benchmark_excited_pyscf.py [output.csv]
Run in WSL where PySCF is installed.
"""

import sys
import time
import csv
import os
import traceback
import resource

OUTFILE = sys.argv[1] if len(sys.argv) > 1 else "benchmark_pyscf_excited.csv"

# ── Molecules (atom, coords in Angstrom) ──
MOLECULES = {
    "H2O": """
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200""",

    "Benzene": """
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

# Load from xyz files
GANSU_XYZ = os.environ.get("GANSU_PATH", os.path.expanduser("~/GANSU")) + "/xyz"
for name in ["Naphthalene", "Anthracene", "Tetracene"]:
    fpath = os.path.join(GANSU_XYZ, f"{name}.xyz")
    if os.path.exists(fpath):
        lines = open(fpath).readlines()
        # Skip first 2 lines (atom count + comment)
        atom_lines = []
        for line in lines[2:]:
            parts = line.strip().split()
            if len(parts) >= 4:
                atom_lines.append(f"{parts[0]}  {parts[1]}  {parts[2]}  {parts[3]}")
        MOLECULES[name] = "\n".join(atom_lines)

BASIS_SETS = ["sto-3g", "cc-pvdz"]
N_STATES = 3

# PySCF methods to benchmark
# (label, function)
def run_cis(mol, mf, n_states):
    from pyscf import tdscf
    td = tdscf.TDA(mf)
    td.nstates = n_states
    td.kernel()
    return td.e, None

def run_adc2(mol, mf, n_states):
    from pyscf import adc
    myadc = adc.ADC(mf)
    myadc.method = "adc(2)"
    myadc.method_type = "ee"
    myadc.kernel(nroots=n_states)
    return myadc.e_ee, None

def run_eom_ccsd(mol, mf, n_states):
    from pyscf import cc
    mycc = cc.CCSD(mf)
    mycc.kernel()
    from pyscf.cc import eom_rccsd
    eom = eom_rccsd.EOMEESinglet(mycc)
    e_ee, _ = eom.kernel(nroots=n_states)
    return e_ee, mycc.e_corr

METHODS = [
    ("CIS (TDA)", run_cis),
    ("ADC(2)", run_adc2),
    ("EOM-CCSD", run_eom_ccsd),
]


def get_mem_mb():
    """Get peak memory usage in MB (Linux only)."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB
    except Exception:
        return 0


rows = []
total = len(MOLECULES) * len(BASIS_SETS) * len(METHODS)
count = 0

for mol_name, atom_str in MOLECULES.items():
    for basis in BASIS_SETS:
        for method_label, method_fn in METHODS:
            count += 1
            label = f"[{count}/{total}] {mol_name}/{basis}/{method_label}"
            print(f"{label}...", end=" ", flush=True)

            row = {
                "molecule": mol_name,
                "basis": basis,
                "method": method_label,
                "n_states": N_STATES,
                "time_s": 0,
                "mem_mb": 0,
                "status": "error",
                "error": "",
                "hf_energy": "",
                "post_hf_correction": "",
            }

            try:
                from pyscf import gto, scf
                mol = gto.M(atom=atom_str.strip(), basis=basis, verbose=0)
                mf = scf.RHF(mol)
                mf.kernel()
                row["hf_energy"] = mf.e_tot

                t0 = time.time()
                mem0 = get_mem_mb()

                excitation_energies, corr_energy = method_fn(mol, mf, N_STATES)

                elapsed = time.time() - t0
                mem1 = get_mem_mb()

                row["time_s"] = round(elapsed, 3)
                row["mem_mb"] = round(mem1 - mem0, 1) if mem1 > mem0 else round(mem1, 1)
                row["status"] = "ok"
                if corr_energy is not None:
                    row["post_hf_correction"] = corr_energy

                for i, e in enumerate(excitation_energies[:N_STATES]):
                    row[f"E{i+1}_ha"] = float(e)
                    row[f"E{i+1}_ev"] = float(e) * 27.21138

                print(f"ok ({elapsed:.2f}s)")

            except Exception as e:
                row["error"] = str(e)[:100]
                print(f"ERROR: {e}")
                traceback.print_exc()

            rows.append(row)

# ── Write CSV ──
if rows:
    fieldnames = list(rows[0].keys())
    for i in range(N_STATES):
        for col in [f"E{i+1}_ha", f"E{i+1}_ev"]:
            if col not in fieldnames:
                fieldnames.append(col)

    with open(OUTFILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTFILE}")
    print(f"Total: {sum(1 for r in rows if r['status']=='ok')} ok, "
          f"{sum(1 for r in rows if r['status']=='error')} errors")
