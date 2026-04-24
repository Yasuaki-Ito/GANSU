#!/usr/bin/env python3
"""
Test: Frozen core approximation — compare GANSU vs PySCF.
Usage: python test_frozen_core.py

Runs both GANSU (via Python API) and PySCF, compares MP2 correlation energies.
"""

import sys
import os
import time

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

XYZ_DIR = os.path.join(GANSU_PATH, "xyz")

# ── Test cases ──
TESTS = [
    # (molecule, xyz_file, basis, n_frozen_expected_auto)
    ("H2O", "H2O.xyz", "cc-pvdz", 1),      # O has 1s frozen
    ("N2", "N2.xyz", "cc-pvdz", 2),          # 2x N 1s
    ("Benzene", "Benzene.xyz", "sto-3g", 6), # 6x C 1s
]

PASS = 0
FAIL = 0


def test_gansu_frozen(mol_name, xyz_file, basis, expected_frozen, post_hf="mp2"):
    global PASS, FAIL
    import gansu

    xyz_path = os.path.join(XYZ_DIR, xyz_file)
    if not os.path.exists(xyz_path):
        print(f"  SKIP {mol_name}: {xyz_path} not found")
        return None, None

    method_upper = post_hf.upper()

    is_excited = post_hf in ("cis", "adc2", "eom_ccsd")

    # Without frozen core
    mol = gansu.Molecule(xyz_path, basis=basis, initial_guess="sad")
    r = mol.run(method="RHF", post_hf=post_hf, quiet=True)
    e_hf = r.total_energy
    if is_excited:
        import re
        report = r.excited_state_report or ""
        e_nofrozen = None
        for line in report.split('\n'):
            m = re.match(r'\s*1\s+([-+]?\d+\.?\d*)', line)
            if m:
                e_nofrozen = float(m.group(1))
                break
    else:
        e_nofrozen = r.post_hf_energy
    del r, mol

    # With frozen core (auto)
    mol = gansu.Molecule(xyz_path, basis=basis, initial_guess="sad", frozen_core="auto")
    r = mol.run(method="RHF", post_hf=post_hf, quiet=True)
    if is_excited:
        report = r.excited_state_report or ""
        e_frozen = None
        for line in report.split('\n'):
            m = re.match(r'\s*1\s+([-+]?\d+\.?\d*)', line)
            if m:
                e_frozen = float(m.group(1))
                break
    else:
        e_frozen = r.post_hf_energy
    del r, mol

    print(f"  GANSU {mol_name}/{basis}/{method_upper}:")
    print(f"    HF energy:                {e_hf:.10f}")
    if e_nofrozen is not None and e_frozen is not None:
        label = "E1 excit." if is_excited else method_upper
        print(f"    {label} (all electron): {e_nofrozen:.10f}")
        print(f"    {label} (frozen core):  {e_frozen:.10f}")
        print(f"    Difference:               {abs(e_nofrozen - e_frozen):.10f}")
    else:
        print(f"    Could not parse excited state energies")

    return e_hf, e_frozen


def test_pyscf_frozen(mol_name, xyz_file, basis, n_frozen, post_hf="mp2"):
    global PASS, FAIL
    try:
        from pyscf import gto, scf, mp, cc
    except ImportError:
        print(f"  SKIP PySCF: not installed")
        return None

    xyz_path = os.path.join(XYZ_DIR, xyz_file)
    if not os.path.exists(xyz_path):
        return None

    # Read xyz
    lines = open(xyz_path).readlines()
    atom_str = ""
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 4:
            atom_str += f"{parts[0]}  {parts[1]}  {parts[2]}  {parts[3]}\n"

    mol = gto.M(atom=atom_str.strip(), basis=basis, verbose=0, cart=True)
    mf = scf.RHF(mol)
    mf.kernel()

    method_upper = post_hf.upper()

    if post_hf == "mp2":
        calc = mp.MP2(mf, frozen=n_frozen)
        calc.kernel()
        e_corr = calc.e_corr
    elif post_hf == "ccsd":
        calc = cc.CCSD(mf, frozen=n_frozen)
        calc.kernel()
        e_corr = calc.e_corr
    elif post_hf == "ccsd_t":
        calc = cc.CCSD(mf, frozen=n_frozen)
        calc.kernel()
        et = calc.ccsd_t()
        e_corr = calc.e_corr + et
    elif post_hf == "cis":
        from pyscf import tdscf
        td = tdscf.TDA(mf)
        td.nstates = 3
        # TDA doesn't support frozen core directly — compare excitation energies instead
        td.kernel()
        e_corr = td.e[0]  # first excitation energy
    else:
        print(f"  PySCF: {post_hf} not supported in test")
        return None

    print(f"  PySCF {mol_name}/{basis}/{method_upper} (frozen={n_frozen}):")
    print(f"    HF energy:         {mf.e_tot:.10f}")
    print(f"    {method_upper} correlation:   {e_corr:.10f}")

    return e_corr


# ── Main ──
print("=" * 60)
print("  Frozen Core Approximation: GANSU vs PySCF")
print("=" * 60)

import gansu
gansu.init()

METHODS_TO_TEST = ["mp2", "ccsd", "ccsd_t", "cis"]

for mol_name, xyz_file, basis, n_frozen in TESTS:
    for post_hf in METHODS_TO_TEST:
        # Skip CCSD for Benzene (too slow for test)
        if post_hf == "ccsd" and mol_name == "Benzene":
            continue

        print(f"\n--- {mol_name} / {basis} / {post_hf.upper()} (frozen: {n_frozen}) ---")

        e_hf, e_gansu = test_gansu_frozen(mol_name, xyz_file, basis, n_frozen, post_hf)

        e_pyscf = test_pyscf_frozen(mol_name, xyz_file, basis, n_frozen, post_hf)

        if e_gansu is not None and e_pyscf is not None:
            diff = abs(e_gansu - e_pyscf)
            label = post_hf.upper()
            print(f"\n  Comparison:")
            print(f"    GANSU frozen {label}:  {e_gansu:.10f}")
            print(f"    PySCF frozen {label}:  {e_pyscf:.10f}")
            print(f"    |diff|:               {diff:.2e} Ha")
            if diff < 1e-5:
                PASS += 1
                print(f"    => PASS")
            else:
                FAIL += 1
                print(f"    => FAIL (diff > 1e-5)")

gansu.finalize()

print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed")
