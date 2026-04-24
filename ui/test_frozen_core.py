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


def test_gansu_mp2_frozen(mol_name, xyz_file, basis, expected_frozen):
    global PASS, FAIL
    import gansu

    xyz_path = os.path.join(XYZ_DIR, xyz_file)
    if not os.path.exists(xyz_path):
        print(f"  SKIP {mol_name}: {xyz_path} not found")
        return None, None

    # Without frozen core
    mol = gansu.Molecule(xyz_path, basis=basis, initial_guess="sad")
    r = mol.run(method="RHF", post_hf="mp2", quiet=True)
    e_nofrozen = r.post_hf_energy
    e_hf = r.total_energy
    del r, mol

    # With frozen core (auto)
    mol = gansu.Molecule(xyz_path, basis=basis, initial_guess="sad", frozen_core="auto")
    r = mol.run(method="RHF", post_hf="mp2", quiet=True)
    e_frozen = r.post_hf_energy
    del r, mol

    print(f"  GANSU {mol_name}/{basis}:")
    print(f"    HF energy:          {e_hf:.10f}")
    print(f"    MP2 (all electron): {e_nofrozen:.10f}")
    print(f"    MP2 (frozen core):  {e_frozen:.10f}")
    print(f"    Difference:         {abs(e_nofrozen - e_frozen):.10f}")

    return e_hf, e_frozen


def test_pyscf_mp2_frozen(mol_name, xyz_file, basis, n_frozen):
    global PASS, FAIL
    try:
        from pyscf import gto, scf, mp
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

    # MP2 with frozen core
    mp2 = mp.MP2(mf, frozen=n_frozen)
    mp2.kernel()

    print(f"  PySCF {mol_name}/{basis} (frozen={n_frozen}):")
    print(f"    HF energy:         {mf.e_tot:.10f}")
    print(f"    MP2 correlation:   {mp2.e_corr:.10f}")

    return mp2.e_corr


# ── Main ──
print("=" * 60)
print("  Frozen Core Approximation: GANSU vs PySCF")
print("=" * 60)

import gansu
gansu.init()

for mol_name, xyz_file, basis, n_frozen in TESTS:
    print(f"\n--- {mol_name} / {basis} (expected frozen: {n_frozen}) ---")

    e_hf, e_gansu = test_gansu_mp2_frozen(mol_name, xyz_file, basis, n_frozen)

    e_pyscf = test_pyscf_mp2_frozen(mol_name, xyz_file, basis, n_frozen)

    if e_gansu is not None and e_pyscf is not None:
        diff = abs(e_gansu - e_pyscf)
        print(f"\n  Comparison:")
        print(f"    GANSU frozen MP2:  {e_gansu:.10f}")
        print(f"    PySCF frozen MP2:  {e_pyscf:.10f}")
        print(f"    |diff|:            {diff:.2e} Ha")
        if diff < 1e-6:
            PASS += 1
            print(f"    => PASS")
        else:
            FAIL += 1
            print(f"    => FAIL (diff > 1e-6)")

gansu.finalize()

print(f"\n{'='*60}")
print(f"Results: {PASS} passed, {FAIL} failed")
