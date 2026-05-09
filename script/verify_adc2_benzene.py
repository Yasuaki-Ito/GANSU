"""
Verify ADC(2) excitation energies for benzene/cc-pVDZ against PySCF.

Usage: python verify_adc2_benzene.py
"""
from pyscf import gto, scf, adc

mol = gto.M(
    atom='''
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
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    cart=True
)

mf = scf.RHF(mol).run()
print(f"RHF energy: {mf.e_tot:.10f}")
print(f"Num occ: {mol.nelectron // 2}, Num basis: {mol.nao_nr()}")

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e = result[0]

print("\nPySCF ADC(2) excitation energies:")
for i, ei in enumerate(e):
    print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")

print("\nGANSU results for comparison:")
gansu = [0.201134, 0.250873, 0.283050, 0.283142, 0.302001]
for i, gi in enumerate(gansu):
    diff = abs(gi - e[i]) if i < len(e) else float('nan')
    print(f"  State {i+1}: GANSU={gi:.6f}  PySCF={e[i]:.6f}  diff={diff:.6e} Ha")
