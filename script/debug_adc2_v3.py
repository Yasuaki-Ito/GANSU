"""
Debug ADC(2): use PySCF matvec to build full matrix.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee

np.set_printoptions(precision=10, linewidth=200, suppress=True)

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
eps = mf.mo_energy
sd = nocc * nvir
dd = nocc**2 * nvir**2

print(f"nocc={nocc}, nvir={nvir}, sd={sd}, dd={dd}")

myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=5)
e_ee = result[0]
print(f"PySCF ADC(2) eigenvalues: {e_ee}")

# Get M_ab from get_imds
M_ab = radc_ee.get_imds(myadc)
print(f"get_imds returned: type={type(M_ab)}, shape={M_ab.shape if hasattr(M_ab,'shape') else 'N/A'}")

# Try matvec with M_ab
import inspect
src = inspect.getsource(radc_ee.matvec)
print(f"\nmatvec source (first 800 chars):\n{src[:800]}")
