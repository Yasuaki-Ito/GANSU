"""
Comprehensive ADC(2) diagnostic for H2O/cc-pVDZ (Cartesian).
Compares CIS/TDA, ADC(2) M11, and full ADC(2).

Usage: python verify_adc2_h2o_diagnostic.py
"""
import numpy as np
from pyscf import gto, scf, adc, tdscf

mol = gto.M(
    atom='''
    O   0.000   0.000   0.117
    H   0.000   0.757  -0.470
    H   0.000  -0.757  -0.470
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    cart=True
)

mf = scf.RHF(mol).run()
print(f"RHF energy: {mf.e_tot:.10f}")
print(f"Num occ: {mol.nelectron // 2}, Num basis: {mol.nao_nr()}")

# 1. CIS/TDA (Tamm-Dancoff Approximation)
print("\n=== CIS/TDA ===")
td = tdscf.TDA(mf)
td.nstates = 10
td.run()
print("TDA excitation energies:")
for i, ei in enumerate(td.e):
    print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")

# 2. ADC(2)
print("\n=== ADC(2) ===")
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=10)
e_adc2 = result[0]
print("ADC(2) excitation energies:")
for i, ei in enumerate(e_adc2):
    print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")

# 3. MP2 correlation energy
print(f"\nMP2 correlation energy: {myadc.e_corr:.10f}")

# 4. Build ADC(2) matrix explicitly for diagnostics
print("\n=== ADC(2) matrix diagnostics ===")
try:
    # Access the ADC matrix
    myadc2 = adc.radc.RADCIP(mf)  # this may not work
except:
    pass

# 5. Compare with GANSU
print("\n=== GANSU comparison ===")
gansu_m11 = []  # Will be filled after running GANSU with diagnostics
gansu_adc2 = [0.284460, 0.360289, 0.391786, 0.469268, 0.514541]

print("GANSU ADC(2) vs PySCF ADC(2):")
for i in range(min(5, len(e_adc2))):
    gi = gansu_adc2[i] if i < len(gansu_adc2) else float('nan')
    diff = abs(gi - e_adc2[i])
    print(f"  State {i+1}: GANSU={gi:.6f}  PySCF={e_adc2[i]:.6f}  diff={diff:.6e} Ha")

# 6. Also try ADC(2)-x for comparison
print("\n=== ADC(2)-x (extended) ===")
try:
    myadc_x = adc.ADC(mf)
    myadc_x.method = "adc(2)-x"
    result_x = myadc_x.kernel(nroots=5)
    e_x = result_x[0]
    print("ADC(2)-x excitation energies:")
    for i, ei in enumerate(e_x):
        print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")
except Exception as ex:
    print(f"  ADC(2)-x failed: {ex}")
