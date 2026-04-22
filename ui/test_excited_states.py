#!/usr/bin/env python3
"""
Test: All excited state methods via in-process API.
Usage: python test_excited_states.py [base_url]
"""

import sys
import json
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

H2O_XYZ = """3
H2O
O  0.000000  0.000000  0.117300
H  0.000000  0.756950 -0.469200
H  0.000000 -0.756950 -0.469200"""

PASS = 0
FAIL = 0


def run_calc(post_hf, solver="auto", basis="sto-3g", n_states=3):
    data = json.dumps({
        "xyz_text": H2O_XYZ,
        "basis": basis,
        "method": "RHF",
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
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        return result, time.time() - t0
    except Exception as e:
        return {"ok": False, "error": str(e)}, time.time() - t0


def test_method(label, post_hf, solver="auto", basis="sto-3g"):
    global PASS, FAIL
    print(f"\n  {label}...", end=" ", flush=True)
    result, elapsed = run_calc(post_hf, solver, basis)
    if not result.get("ok"):
        FAIL += 1
        print(f"FAIL ({elapsed:.2f}s) — {result.get('error', '?')[:80]}")
        return
    es = result.get("excited_states", [])
    if not es:
        FAIL += 1
        print(f"FAIL ({elapsed:.2f}s) — no excited states returned")
        return
    PASS += 1
    print(f"PASS ({elapsed:.2f}s) — {len(es)} states")
    for s in es:
        print(f"    State {s['state']:2d}: {s['energy_ha']:10.6f} Ha  {s['energy_ev']:8.4f} eV  f={s['osc_strength']:.4f}  {s['transitions'][:40]}")


print("=" * 60)
print("  Excited state methods — H2O/STO-3G")
print("=" * 60)

test_method("CIS", "cis")
test_method("ADC(2) schur_static", "adc2", "schur_static")
test_method("ADC(2) schur_omega", "adc2", "schur_omega")
test_method("ADC(2) full", "adc2", "full")
test_method("ADC(2)-x", "adc2x")
test_method("EOM-MP2 schur_static", "eom_mp2", "schur_static")
test_method("EOM-MP2 schur_omega", "eom_mp2", "schur_omega")
test_method("EOM-MP2 full", "eom_mp2", "full")
test_method("EOM-CC2", "eom_cc2")
test_method("EOM-CCSD", "eom_ccsd")

print(f"\n{'=' * 60}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("All tests passed!")
else:
    print("Some tests FAILED")
    sys.exit(1)
