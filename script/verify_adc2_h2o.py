"""
Verify ADC(2) excitation energies for H2O/cc-pVDZ against PySCF.
Small molecule where both dense and on-the-fly paths can run.

Usage: python verify_adc2_h2o.py
"""
from pyscf import gto, scf, adc

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

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e = result[0]

print("\nPySCF ADC(2)/cc-pVDZ (Cartesian) excitation energies for H2O:")
for i, ei in enumerate(e):
    print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")
