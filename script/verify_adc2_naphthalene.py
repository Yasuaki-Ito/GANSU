"""
Verify ADC(2) excitation energies for naphthalene/cc-pVDZ against PySCF.

Usage: python verify_adc2_naphthalene.py
"""
from pyscf import gto, scf, adc

# Naphthalene geometry (from GANSU xyz file)
mol = gto.M(
    atom='''
    C   0   1.253   1.395
    C   0   2.421   0.713
    C   0   2.421  -0.713
    C   0   1.253  -1.395
    C   0  -1.253  -1.395
    C   0  -2.421  -0.713
    C   0  -2.421   0.713
    C   0  -1.253   1.395
    C   0   0       0.702
    C   0   0      -0.702
    H   0   1.244   2.478
    H   0   3.367   1.24
    H   0   3.367  -1.24
    H   0   1.244  -2.478
    H   0  -1.244  -2.478
    H   0  -3.367  -1.24
    H   0  -3.367   1.24
    H   0  -1.244   2.478
    ''',
    basis='cc-pvdz',
    unit='Angstrom'
)

mf = scf.RHF(mol).run()
print(f"RHF energy: {mf.e_tot:.10f}")
print(f"Num occ: {mol.nelectron // 2}, Num basis: {mol.nao_nr()}")

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
e, v, _ = myadc.kernel(nroots=5)

print("\nPySCF ADC(2) excitation energies:")
for i, ei in enumerate(e):
    print(f"  State {i+1}: {ei:.10f} Ha  ({ei*27.211386:.4f} eV)")

print("\nGANSU results for comparison:")
gansu = [0.201134, 0.250873, 0.283050, 0.283142, 0.302001]
for i, gi in enumerate(gansu):
    diff = abs(gi - e[i]) if i < len(e) else float('nan')
    print(f"  State {i+1}: GANSU={gi:.6f}  PySCF={e[i]:.6f}  diff={diff:.6e} Ha")
