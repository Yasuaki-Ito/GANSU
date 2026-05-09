#!/usr/bin/env python3
"""
Benchmark PySCF CIS and ADC(2) execution times.

Compare with GANSU:
  ./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs --post_hf_method cis --n_excited_states 5
  ./HF_main -x ../xyz/H2O.xyz -g ../basis/sto-3g.gbs --post_hf_method adc2 --n_excited_states 5
  ./HF_main -x ../xyz/Benzene.xyz -g ../basis/sto-3g.gbs --post_hf_method cis --n_excited_states 5
  etc.
"""

import time
import numpy as np
from pyscf import gto, scf, tdscf, adc

# ============================================================
#  Molecular geometries (same as GANSU xyz files)
# ============================================================

molecules = {
    "H2O/STO-3G": {
        "atom": "O 0.000 0.000 0.127; H 0.000 0.758 -0.509; H 0.000 -0.758 -0.509",
        "basis": "sto-3g",
    },
    "H2O/cc-pVDZ": {
        "atom": "O 0.000 0.000 0.127; H 0.000 0.758 -0.509; H 0.000 -0.758 -0.509",
        "basis": "cc-pvdz",
    },
    "Benzene/STO-3G": {
        "atom": """
            C   0.000    1.387    0.000
            C   1.201    0.693    0.000
            C   1.201   -0.693    0.000
            C   0.000   -1.387    0.000
            C  -1.201   -0.693    0.000
            C  -1.201    0.693    0.000
            H   0.000    2.469    0.000
            H   2.139    1.235    0.000
            H   2.139   -1.235    0.000
            H   0.000   -2.469    0.000
            H  -2.139   -1.235    0.000
            H  -2.139    1.235    0.000
        """,
        "basis": "sto-3g",
    },
    "Benzene/cc-pVDZ": {
        "atom": """
            C   0.000    1.387    0.000
            C   1.201    0.693    0.000
            C   1.201   -0.693    0.000
            C   0.000   -1.387    0.000
            C  -1.201   -0.693    0.000
            C  -1.201    0.693    0.000
            H   0.000    2.469    0.000
            H   2.139    1.235    0.000
            H   2.139   -1.235    0.000
            H   0.000   -2.469    0.000
            H  -2.139   -1.235    0.000
            H  -2.139    1.235    0.000
        """,
        "basis": "cc-pvdz",
    },
    "Naphthalene/STO-3G": {
        "atom": """
            C  0  1.253  1.395
            C  0  2.421  0.713
            C  0  2.421 -0.713
            C  0  1.253 -1.395
            C  0 -1.253 -1.395
            C  0 -2.421 -0.713
            C  0 -2.421  0.713
            C  0 -1.253  1.395
            C  0  0      0.702
            C  0  0     -0.702
            H  0  1.244  2.478
            H  0  3.367  1.240
            H  0  3.367 -1.240
            H  0  1.244 -2.478
            H  0 -1.244 -2.478
            H  0 -3.367 -1.240
            H  0 -3.367  1.240
            H  0 -1.244  2.478
        """,
        "basis": "sto-3g",
    },
    "Naphthalene/cc-pVDZ": {
        "atom": """
            C  0  1.253  1.395
            C  0  2.421  0.713
            C  0  2.421 -0.713
            C  0  1.253 -1.395
            C  0 -1.253 -1.395
            C  0 -2.421 -0.713
            C  0 -2.421  0.713
            C  0 -1.253  1.395
            C  0  0      0.702
            C  0  0     -0.702
            H  0  1.244  2.478
            H  0  3.367  1.240
            H  0  3.367 -1.240
            H  0  1.244 -2.478
            H  0 -1.244 -2.478
            H  0 -3.367 -1.240
            H  0 -3.367  1.240
            H  0 -1.244  2.478
        """,
        "basis": "cc-pvdz",
    },
}

n_states = 5

print("=" * 80)
print(f"PySCF CIS / ADC(2) Benchmark  (n_states={n_states})")
print("=" * 80)
print()

results = []

for name, mol_params in molecules.items():
    print(f"--- {name} ---")

    mol = gto.M(atom=mol_params["atom"], basis=mol_params["basis"], cart=True, verbose=0)
    nao = mol.nao_nr()
    print(f"  nao = {nao}")

    # RHF
    t0 = time.perf_counter()
    mf = scf.RHF(mol).run()
    t_hf = time.perf_counter() - t0
    print(f"  RHF energy = {mf.e_tot:.10f}  ({t_hf:.3f} s)")

    nocc = mol.nelectron // 2
    nvir = nao - nocc
    print(f"  nocc={nocc}, nvir={nvir}, singles={nocc*nvir}, doubles={nocc**2 * nvir**2}")

    # CIS (= TDA)
    t0 = time.perf_counter()
    td = tdscf.TDA(mf)
    td.nstates = n_states
    td.run()
    t_cis = time.perf_counter() - t0
    print(f"  CIS time = {t_cis:.3f} s")
    cis_energies = td.e
    for k, e in enumerate(cis_energies):
        print(f"    CIS[{k}] = {e:.10f} Ha = {e*27.2114:.4f} eV")

    # ADC(2)
    t0 = time.perf_counter()
    myadc = adc.ADC(mf)
    myadc.method = "adc(2)"
    myadc.method_type = "ee"
    e_adc, v_adc, p_adc, x_adc = myadc.kernel(nroots=n_states)
    t_adc = time.perf_counter() - t0
    print(f"  ADC(2) time = {t_adc:.3f} s")
    for k, e in enumerate(e_adc):
        print(f"    ADC(2)[{k}] = {e:.10f} Ha = {e*27.2114:.4f} eV")

    results.append({
        "name": name,
        "nao": nao,
        "nocc": nocc,
        "nvir": nvir,
        "t_hf": t_hf,
        "t_cis": t_cis,
        "t_adc2": t_adc,
        "cis_energies": cis_energies,
        "adc2_energies": e_adc,
    })
    print()

# Summary table
print("=" * 80)
print("Summary")
print("=" * 80)
print(f"{'System':<25s} {'nao':>5s} {'nocc':>5s} {'nvir':>5s} {'HF(s)':>8s} {'CIS(s)':>8s} {'ADC2(s)':>8s}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<25s} {r['nao']:5d} {r['nocc']:5d} {r['nvir']:5d} "
          f"{r['t_hf']:8.3f} {r['t_cis']:8.3f} {r['t_adc2']:8.3f}")
print()
print("GANSU commands:")
for r in results:
    parts = r['name'].split('/')
    mol_name = parts[0]
    basis = parts[1].lower().replace('-', '-')
    basis_file = basis + ".gbs"
    print(f"  ./HF_main -x ../xyz/{mol_name}.xyz -g ../basis/{basis_file} --post_hf_method cis --n_excited_states {n_states}")
    print(f"  ./HF_main -x ../xyz/{mol_name}.xyz -g ../basis/{basis_file} --post_hf_method adc2 --n_excited_states {n_states}")
