"""
Get the CORRECT EE-ADC(2) reference values from PySCF.
The default method_type is "ip", not "ee"!
We need to explicitly set method_type = "ee".

Usage: python correct_ee_adc2_reference.py
"""
import numpy as np
from pyscf import gto, scf, adc, lib
from pyscf.adc import radc_ee, radc
from pyscf.lib import linalg_helper

mol = gto.M(
    atom='''
    O   0.000   0.000   0.117
    H   0.000   0.757  -0.470
    H   0.000  -0.757  -0.470
    ''',
    basis='sto-3g',
    unit='Angstrom',
    cart=True
)

mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
print(f"nocc={nocc}, nvir={nvir}")

# ============================================================
# Method 1: Default (IP-ADC(2))
# ============================================================
print("\n=== Method 1: Default myadc.kernel() (method_type=ip) ===")
myadc_ip = adc.ADC(mf)
myadc_ip.method = "adc(2)"
print(f"method_type before kernel: {myadc_ip.method_type}")
result_ip = myadc_ip.kernel(nroots=5)
e_ip = result_ip[0]
print(f"method_type after kernel: {myadc_ip.method_type}")
print(f"IP-ADC(2): {e_ip}")
print(f"IP-ADC(2) in eV: {e_ip * 27.2114}")

# ============================================================
# Method 2: EE-ADC(2) via method_type = "ee"
# ============================================================
print("\n=== Method 2: myadc.kernel() with method_type='ee' ===")
myadc_ee = adc.ADC(mf)
myadc_ee.method = "adc(2)"
myadc_ee.method_type = "ee"
print(f"method_type before kernel: {myadc_ee.method_type}")

# Trace Davidson to check dimensions
orig_dav = linalg_helper.davidson_nosym1
def traced_dav(matvec, x0, diag, **kwargs):
    print(f"  [Davidson] diag={diag.shape}, x0[0]={x0[0].shape if isinstance(x0, list) else x0.shape}")
    return orig_dav(matvec, x0, diag, **kwargs)
linalg_helper.davidson_nosym1 = traced_dav

result_ee = myadc_ee.kernel(nroots=5)
e_ee = result_ee[0]
print(f"EE-ADC(2): {e_ee}")
print(f"EE-ADC(2) in eV: {e_ee * 27.2114}")

linalg_helper.davidson_nosym1 = orig_dav

# ============================================================
# Method 3: Direct RADCEE approach
# ============================================================
print("\n=== Method 3: Direct RADCEE ===")
myadc3 = adc.ADC(mf)
myadc3.method = "adc(2)"
# Need to compute amplitudes first
eris = myadc3.transform_integrals()
from pyscf.adc import radc_amplitudes
myadc3.e_corr, myadc3.t1, myadc3.t2 = radc_amplitudes.compute_amplitudes_energy(
    myadc3, eris=eris, verbose=0)

adc_es = radc_ee.RADCEE(myadc3)
result3 = adc_es.kernel(nroots=5, eris=eris)
e_ee3 = result3[0]
print(f"Direct RADCEE EE-ADC(2): {e_ee3}")

# ============================================================
# Comparison
# ============================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"IP-ADC(2):       {e_ip}")
print(f"EE-ADC(2) (v1):  {e_ee}")
print(f"EE-ADC(2) (v2):  {e_ee3}")

# Expected dimensions:
# IP: singles = nocc = 5, doubles = nocc*nocc*nvir = 50, total = 55
# EE: singles = nocc*nvir = 10, doubles = nocc^2*nvir^2 = 100, total = 110
print(f"\nExpected dimensions:")
print(f"  IP: {nocc} + {nocc*nocc*nvir} = {nocc + nocc*nocc*nvir}")
print(f"  EE: {nocc*nvir} + {nocc*nocc*nvir*nvir} = {nocc*nvir + nocc*nocc*nvir*nvir}")

# Compare EE-ADC(2) methods
if np.allclose(e_ee, e_ee3, atol=1e-6):
    print(f"\n*** EE-ADC(2) methods agree! ***")
    print(f"Max diff: {np.max(np.abs(e_ee - e_ee3)):.6e}")
else:
    print(f"\nEE-ADC(2) methods DISAGREE!")
    print(f"Diffs: {e_ee - e_ee3}")

# ============================================================
# GANSU reference: the 110-dim eigenvalues
# ============================================================
print(f"\n=== GANSU's 110-dim eigenvalues (from previous runs) ===")
gansu_110 = np.array([0.47224258, 0.55468986, 0.60818392, 0.70837176, 0.82540528])
print(f"GANSU 110-dim: {gansu_110}")
if e_ee is not None:
    print(f"EE-ADC(2):     {e_ee[:5]}")
    diffs = gansu_110[:min(5, len(e_ee))] - e_ee[:min(5, len(gansu_110))]
    print(f"Differences:   {diffs}")
    print(f"Max abs diff:  {np.max(np.abs(diffs)):.6e}")
    if np.allclose(gansu_110[:5], e_ee[:5], atol=1e-5):
        print(f"\n*** GANSU MATCHES EE-ADC(2)! ***")
    else:
        print(f"\nGANSU does NOT match EE-ADC(2)")

# Also try CIS for sanity check
print(f"\n=== CIS reference ===")
from pyscf import tdscf
td = tdscf.CIS(mf)
td.nstates = 5
td.run()
print(f"CIS excitation energies: {td.e}")
print(f"CIS in eV: {td.e * 27.2114}")
