"""
Extract the full ADC(2) matrix from PySCF by probing with unit vectors.
This reveals the correct M11, M12, M21, D2 structure.

Usage: python extract_pyscf_adc2_matrix.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo

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
nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
print(f"nocc={nocc}, nvir={nvir}, ov={ov}")

# Run ADC(2) to get the object
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
myadc.kernel(nroots=5)

# Try to access the matvec function
print("\nADC object attributes:")
print([attr for attr in dir(myadc) if not attr.startswith('_')])

# Check if we can get the sigma vector function
print(f"\nMethod type: {type(myadc)}")

# Try to look at PySCF ADC source
import pyscf.adc as adc_mod
import inspect
print(f"\nADC module location: {adc_mod.__file__}")

# List files in the ADC directory
import os
adc_dir = os.path.dirname(adc_mod.__file__)
print(f"Files in {adc_dir}:")
for f in sorted(os.listdir(adc_dir)):
    print(f"  {f}")

# Try to read the matvec source
try:
    # For EE-ADC(2), PySCF uses radc_ea_amplitudes or similar
    from pyscf.adc import radc_amplitudes
    print(f"\nradc_amplitudes location: {radc_amplitudes.__file__}")

    # Look for the sigma vector computation
    src = inspect.getsource(radc_amplitudes)
    # Find the coupling terms
    for line in src.split('\n'):
        if 'einsum' in line.lower() and ('ov' in line or 'vv' in line or 'oo' in line):
            print(f"  {line.strip()}")
except Exception as e:
    print(f"  Error: {e}")

# Try to find the EE-ADC sigma vector
try:
    from pyscf.adc import radc
    print(f"\nradc location: {radc.__file__}")
except:
    pass

# Try the uadc or dfadc modules
for modname in ['radc', 'uadc', 'dfadc', 'radc_amplitudes']:
    try:
        mod = getattr(adc_mod, modname, None)
        if mod:
            print(f"\n{modname}: {mod.__file__}")
    except:
        pass

# Try to build the full ADC matrix by probing
# PySCF's ADC uses get_imds() to get intermediates
try:
    print("\n\nTrying to access ADC matrix-vector product...")
    # Get method-specific implementation
    from pyscf.adc import radc_amplitudes as radc_amp

    # Look for the sigma function
    funcs = [f for f in dir(radc_amp) if 'sigma' in f.lower() or 'matvec' in f.lower() or 'contract' in f.lower()]
    print(f"Functions with sigma/matvec/contract: {funcs}")

    funcs2 = [f for f in dir(radc_amp) if not f.startswith('_')]
    print(f"All public functions: {funcs2}")
except Exception as e:
    print(f"Error: {e}")

# Try a different approach - look at the ADC method's compute_amplitudes
try:
    from pyscf.adc import radc_amplitudes
    src = inspect.getsource(radc_amplitudes)
    print(f"\n=== radc_amplitudes source (first 200 lines) ===")
    for i, line in enumerate(src.split('\n')[:200]):
        print(f"  {i+1:3d}: {line}")
except Exception as e:
    print(f"Error reading source: {e}")
