#!/usr/bin/env python3
"""
Validate GANSU CC2 correlation energy against Psi4 reference.

Usage:
    pip install psi4  (or conda install psi4 -c conda-forge)
    python validate_cc2_psi4.py

GANSU uses Cartesian d-functions (6d) — Psi4 default is spherical (5d).
Set puream=False for Cartesian to match GANSU.
"""

import psi4

psi4.set_memory('2 GB')
psi4.core.set_output_file('cc2_validation.out', False)

# H2O geometry (same as GANSU's H2O.xyz)
mol = psi4.geometry("""
    O   0.000000   0.000000   0.127118
    H   0.000000   0.757814  -0.508873
    H   0.000000  -0.757814  -0.508873
""")

# --- H2O / cc-pVDZ (Cartesian) ---
psi4.set_options({
    'basis': 'cc-pvdz',
    'puream': False,       # Cartesian (6d) to match GANSU
    'scf_type': 'pk',
    'e_convergence': 1e-12,
    'r_convergence': 1e-10,
    'freeze_core': False,
})

# RHF
e_rhf = psi4.energy('scf')
print(f"RHF energy:  {e_rhf:.10f}")

# MP2
e_mp2 = psi4.energy('mp2')
print(f"MP2 energy:  {e_mp2:.10f}")
print(f"MP2 corr:    {e_mp2 - e_rhf:.10f}")

# CC2
e_cc2 = psi4.energy('cc2')
print(f"CC2 energy:  {e_cc2:.10f}")
print(f"CC2 corr:    {e_cc2 - e_rhf:.10f}")

# CCSD
e_ccsd = psi4.energy('ccsd')
print(f"CCSD energy: {e_ccsd:.10f}")
print(f"CCSD corr:   {e_ccsd - e_rhf:.10f}")

print("\n--- Summary ---")
print(f"MP2  corr: {e_mp2 - e_rhf:.10f}")
print(f"CC2  corr: {e_cc2 - e_rhf:.10f}")
print(f"CCSD corr: {e_ccsd - e_rhf:.10f}")
print(f"MP2 < CC2 < CCSD: {(e_mp2 - e_rhf) > (e_cc2 - e_rhf) > (e_ccsd - e_rhf)}")

print("\n--- GANSU values for comparison ---")
print(f"GANSU MP2  corr: -0.2102563235")
print(f"GANSU CC2  corr: -0.2104510390")
print(f"GANSU CCSD corr: (from test) -0.2192016272")

# --- H2 / STO-3G ---
print("\n\n=== H2 / STO-3G ===")
mol_h2 = psi4.geometry("""
    H   0.000000   0.000000   0.000000
    H   0.000000   0.000000   0.712200
""")

psi4.set_options({
    'basis': 'sto-3g',
    'puream': False,
})

e_rhf_h2 = psi4.energy('scf', molecule=mol_h2)
e_cc2_h2 = psi4.energy('cc2', molecule=mol_h2)
e_ccsd_h2 = psi4.energy('ccsd', molecule=mol_h2)
print(f"H2 RHF:  {e_rhf_h2:.10f}")
print(f"H2 CC2 corr:  {e_cc2_h2 - e_rhf_h2:.10f}")
print(f"H2 CCSD corr: {e_ccsd_h2 - e_rhf_h2:.10f}")
