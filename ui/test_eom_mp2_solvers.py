#!/usr/bin/env python3
"""
Test: EOM-MP2 solver comparison (schur_static vs schur_omega vs full).
Verifies that the refactored schur_static (dense diag) gives consistent results.

Usage: python test_eom_mp2_solvers.py [base_url]
"""

import sys
import json
import time
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

H2O_XYZ = """3
H2O
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200"""


def run_calc(xyz, basis, method, post_hf, solver="auto", n_states=5):
    data = json.dumps({
        "xyz_text": xyz,
        "basis": basis,
        "method": method,
        "post_hf_method": post_hf,
        "excited_solver": solver,
        "n_excited_states": n_states,
        "initial_guess": "sad",
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/api/run/inprocess",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    return result, elapsed


def print_excited_states(label, result, elapsed):
    print(f"\n{'='*60}")
    print(f"  {label}  ({elapsed:.2f}s)")
    print(f"{'='*60}")
    if not result.get("ok"):
        print(f"  ERROR: {result.get('error')}")
        return []
    es = result.get("excited_states", [])
    if not es:
        print("  No excited states found")
        return []
    print(f"  {'State':>5}  {'E (Ha)':>12}  {'E (eV)':>10}  {'f':>8}  Transitions")
    print(f"  {'-'*5}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*20}")
    for s in es:
        print(f"  {s['state']:5d}  {s['energy_ha']:12.6f}  {s['energy_ev']:10.4f}  {s['osc_strength']:8.4f}  {s['transitions']}")
    return es


def compare(label1, es1, label2, es2):
    if not es1 or not es2:
        print(f"\n  Cannot compare: missing data")
        return
    n = min(len(es1), len(es2))
    print(f"\n  Comparison: {label1} vs {label2}")
    print(f"  {'State':>5}  {'dE (Ha)':>12}  {'dE (eV)':>10}  {'df':>8}")
    max_de = 0
    for i in range(n):
        de_ha = es1[i]["energy_ha"] - es2[i]["energy_ha"]
        de_ev = es1[i]["energy_ev"] - es2[i]["energy_ev"]
        df = es1[i]["osc_strength"] - es2[i]["osc_strength"]
        max_de = max(max_de, abs(de_ha))
        print(f"  {i+1:5d}  {de_ha:12.6f}  {de_ev:10.4f}  {df:8.4f}")
    print(f"  Max |dE|: {max_de:.6f} Ha ({max_de*27.211:.4f} eV)")
    if max_de < 0.02:
        print("  => PASS (< 0.02 Ha)")
    else:
        print("  => WARNING: large deviation")


# Run with each solver
print("Testing EOM-MP2 solvers on H2O/STO-3G...")

r_static, t_static = run_calc(H2O_XYZ, "sto-3g", "RHF", "eom_mp2", "schur_static")
es_static = print_excited_states("schur_static (dense diag)", r_static, t_static)

r_omega, t_omega = run_calc(H2O_XYZ, "sto-3g", "RHF", "eom_mp2", "schur_omega")
es_omega = print_excited_states("schur_omega (omega iteration)", r_omega, t_omega)

r_full, t_full = run_calc(H2O_XYZ, "sto-3g", "RHF", "eom_mp2", "full")
es_full = print_excited_states("full (Davidson)", r_full, t_full)

# Compare
compare("schur_static", es_static, "schur_omega", es_omega)
compare("schur_static", es_static, "full", es_full)
compare("schur_omega", es_omega, "full", es_full)

print(f"\nTiming: static={t_static:.2f}s  omega={t_omega:.2f}s  full={t_full:.2f}s")
