"""
Build the full ADC(2) matrix numerically using PySCF's sigma-vector routine.
Extract it column-by-column to get the explicit matrix, then analyze.

Usage: python build_adc2_from_pyscf.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
import pyscf

mol = gto.M(
    atom='''
    O   0.000   0.000   0.117
    H   0.000   0.757  -0.470
    H   0.000  -0.757  -0.470
    ''',
    basis='sto-3g',  # Use STO-3G for small matrix
    unit='Angstrom',
    cart=True
)

mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
print(f"nocc={nocc}, nvir={nvir}, nao={nao}, ov={ov}, dd={dd}")
print(f"Total dimension: {ov + dd}")

eps = mf.mo_energy
C = mf.mo_coeff

# Get MO ERIs
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)

# T2 amplitudes
t2 = np.zeros((nocc, nocc, nvir, nvir))
D2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                denom = eps[i] + eps[j] - eps[a + nocc] - eps[b + nocc]
                t2[i, j, a, b] = eri_ovov[i, a, j, b] / denom
                D2[i, j, a, b] = eps[a + nocc] + eps[b + nocc] - eps[i] - eps[j]

# MP2 energy
e_mp2 = np.einsum('ijab,iajb->', t2, 2*eri_ovov - eri_ovov.transpose(0,3,2,1))
print(f"MP2 energy: {e_mp2:.10f}")

# Run PySCF ADC(2) for reference
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=min(5, ov))
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

# Now try to extract the ADC matrix
# PySCF stores intermediates in get_imds()
try:
    # Access the internal sigma vector function
    import pyscf.adc.radc_amplitudes as radc_amp

    # Build intermediates
    imds = myadc.get_imds()

    # Check what imds contains
    print(f"\nimds type: {type(imds)}")
    print(f"imds attributes: {[a for a in dir(imds) if not a.startswith('_')]}")

    # Look at method-specific attributes
    if hasattr(imds, 't2'):
        print(f"imds.t2 shape: {np.array(imds.t2[0]).shape if isinstance(imds.t2, tuple) else imds.t2.shape}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Alternative: try to use the matvec directly
try:
    print("\n=== Trying matvec approach ===")
    # PySCF ADC uses a LinearOperator-like approach
    # The key function is compute_amplitudes or get_trans_moments

    # Try to find the matrix-vector product function
    from pyscf.adc import radc as radc_mod
    src_lines = open(radc_mod.__file__).readlines()

    # Find matvec or sigma-related functions
    for i, line in enumerate(src_lines):
        if any(kw in line.lower() for kw in ['def matvec', 'def sigma', 'def get_diag', 'def contract']):
            print(f"  Line {i+1}: {line.rstrip()}")
            # Print next few lines
            for j in range(1, 10):
                if i+j < len(src_lines):
                    print(f"  Line {i+j+1}: {src_lines[i+j].rstrip()}")
            print()
except Exception as e:
    print(f"Error: {e}")

# Try yet another approach - use the ADC(2) sigma vector to build the matrix
print("\n=== Building ADC(2) matrix from sigma vectors ===")
try:
    # Find the right function
    from pyscf.adc import radc_amplitudes as ramp

    # List all functions in radc_amplitudes
    all_funcs = [f for f in dir(ramp) if callable(getattr(ramp, f)) and not f.startswith('_')]
    print(f"Functions in radc_amplitudes: {all_funcs}")

    # Try to use compute_amplitudes
    if hasattr(ramp, 'compute_amplitudes'):
        print("Found compute_amplitudes")

    # Look for the sigma vector function signature
    for func_name in all_funcs:
        func = getattr(ramp, func_name)
        try:
            sig = str(func.__doc__[:200]) if func.__doc__ else "no doc"
            print(f"  {func_name}: {sig}")
        except:
            pass

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Direct approach: find and read the PySCF ADC source for the coupling terms
print("\n=== Looking for coupling formulas in PySCF source ===")
try:
    import pyscf.adc as adc_pkg
    adc_dir = os.path.dirname(adc_pkg.__file__)
    import os

    for fname in sorted(os.listdir(adc_dir)):
        if fname.endswith('.py'):
            fpath = os.path.join(adc_dir, fname)
            with open(fpath) as f:
                content = f.read()
            # Look for einsum calls related to coupling
            lines = content.split('\n')
            found = False
            for i, line in enumerate(lines):
                if 'einsum' in line and any(kw in line for kw in ['ijab', 'iajb', 'abij', 'jabi']):
                    if not found:
                        print(f"\n--- {fname} ---")
                        found = True
                    # Print context
                    for j in range(max(0,i-2), min(len(lines), i+3)):
                        print(f"  {j+1:4d}: {lines[j]}")
                    print()
except Exception as e:
    print(f"Error: {e}")
