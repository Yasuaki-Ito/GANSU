"""
Find exactly where PySCF converts from 110-dim to 55-dim.
Patch gen_matvec and get_init_guess to trace the transformation.

Usage: python trace_pyscf_55dim_source.py
"""
import numpy as np
from pyscf import gto, scf, adc, lib
from pyscf.adc import radc_ee, radc, radc_amplitudes
from pyscf.lib import linalg_helper
import inspect

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
nvir = mf.mo_coeff.shape[1] - nocc
print(f"nocc={nocc}, nvir={nvir}")

# Read the radc.kernel standalone function COMPLETELY
print("\n=== COMPLETE radc.kernel source ===")
func = radc.kernel
src = inspect.getsource(func)
print(src)

# Read the COMPLETE RADCEE.gen_matvec source
print("\n=== COMPLETE RADCEE.gen_matvec source ===")
src = inspect.getsource(radc_ee.RADCEE.gen_matvec)
print(src)

# Check if RADCEE overrides matvec or get_diag
print("\n=== Checking method resolution order ===")
for method_name in ['matvec', 'get_diag', 'gen_matvec', 'get_init_guess', 'get_imds']:
    method = getattr(radc_ee.RADCEE, method_name, None)
    if method:
        # Check if it's defined in RADCEE or inherited
        for cls in radc_ee.RADCEE.__mro__:
            if method_name in cls.__dict__:
                print(f"  {method_name}: defined in {cls.__name__}")
                break

# Now patch RADCEE methods to trace the flow
print("\n=== Tracing the actual kernel execution ===")

original_gen_matvec = radc_ee.RADCEE.gen_matvec
def traced_gen_matvec(self, imds=None, eris=None):
    print(f"  [gen_matvec] called")
    result = original_gen_matvec(self, imds, eris)
    matvec_fn, diag = result
    print(f"  [gen_matvec] returned diag.shape={diag.shape}")

    # Test matvec dimension
    test = np.zeros(len(diag))
    test[0] = 1.0
    out = matvec_fn(test)
    print(f"  [gen_matvec] matvec: input_dim={len(diag)}, output_dim={len(out)}")
    return result

radc_ee.RADCEE.gen_matvec = traced_gen_matvec

original_get_init = radc_ee.RADCEE.get_init_guess
def traced_get_init(self, nroots=1, diag=None, ascending=True, type=None, eris=None, ini=None):
    print(f"  [get_init_guess] called, type={type}, diag.shape={diag.shape if diag is not None else 'None'}")
    result = original_get_init(self, nroots, diag, ascending, type, eris, ini)
    if isinstance(result, list) and len(result) > 0:
        print(f"  [get_init_guess] returned {len(result)} vectors, shape={result[0].shape}")
    return result

radc_ee.RADCEE.get_init_guess = traced_get_init

# Also patch the standalone kernel to see everything
original_kernel = radc.kernel
def traced_kernel(adc_obj, nroots=1, guess=None, eris=None, verbose=None):
    print(f"  [kernel] called, type(adc)={type(adc_obj).__name__}")

    # Call the original kernel but with our traces
    return original_kernel(adc_obj, nroots, guess, eris, verbose)

radc.kernel = traced_kernel
# Also update RADCEE.kernel since it references radc.kernel
radc_ee.RADCEE.kernel = traced_kernel

# Also patch Davidson
original_dav = linalg_helper.davidson_nosym1
def traced_dav(matvec, x0, diag, **kwargs):
    print(f"  [davidson] called, diag.shape={diag.shape}")
    if isinstance(x0, (list, tuple)):
        print(f"  [davidson] x0: {len(x0)} vectors, x0[0].shape={x0[0].shape}")

    # Test matvec with one vector
    test_dim = len(diag)
    test = np.zeros(test_dim)
    test[0] = 1.0
    out = matvec([test])
    print(f"  [davidson] matvec test: input_dim={test_dim}, output[0].shape={out[0].shape}")

    return original_dav(matvec, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = traced_dav

# Run ADC(2)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"\nPySCF ADC(2): {e_adc2}")

# Restore
radc.kernel = original_kernel
radc_ee.RADCEE.kernel = original_kernel
radc_ee.RADCEE.gen_matvec = original_gen_matvec
radc_ee.RADCEE.get_init_guess = original_get_init
linalg_helper.davidson_nosym1 = original_dav
