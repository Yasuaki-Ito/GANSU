"""
Minimal test to find where 110-dim becomes 55-dim.
Compare myadc.kernel() vs manual RADCEE.kernel() in the same script.

Usage: python trace_minimal.py
"""
import numpy as np
from pyscf import gto, scf, adc, lib
from pyscf.adc import radc_ee, radc, radc_amplitudes, radc_ao2mo
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

# Read __init__.py
print("=== pyscf/adc/__init__.py ===")
import pyscf.adc as adc_module
init_path = adc_module.__file__
with open(init_path) as f:
    print(f.read())

# ============================================================
# Test 1: Use myadc.kernel() with Davidson trace
# ============================================================
print("\n" + "="*60)
print("Test 1: myadc.kernel() with Davidson trace")
print("="*60)

dav_calls = []
orig_dav = linalg_helper.davidson_nosym1

def traced_dav(matvec, x0, diag, **kwargs):
    info = {
        'diag_shape': diag.shape,
        'x0_shape': x0[0].shape if isinstance(x0, list) else x0.shape,
        'caller': inspect.stack()[1]
    }
    dav_calls.append(info)
    print(f"  [Davidson #{len(dav_calls)}] diag={diag.shape}, "
          f"x0[0]={info['x0_shape']}, "
          f"caller={info['caller'].filename.split('/')[-1]}:{info['caller'].lineno}")
    return orig_dav(matvec, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = traced_dav

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result1 = myadc.kernel(nroots=5)
print(f"Result 1: {result1[0]}")
print(f"Total Davidson calls: {len(dav_calls)}")
for i, info in enumerate(dav_calls):
    print(f"  Call {i+1}: diag={info['diag_shape']}, x0={info['x0_shape']}")

linalg_helper.davidson_nosym1 = orig_dav

# ============================================================
# Test 2: Manual RADCEE.kernel() with Davidson trace
# ============================================================
print("\n" + "="*60)
print("Test 2: Manual RADCEE.kernel() with Davidson trace")
print("="*60)

dav_calls2 = []
linalg_helper.davidson_nosym1 = lambda *args, **kwargs: traced_dav2(*args, **kwargs)

def traced_dav2(matvec, x0, diag, **kwargs):
    info = {
        'diag_shape': diag.shape,
        'x0_shape': x0[0].shape if isinstance(x0, list) else x0.shape,
        'caller': inspect.stack()[1]
    }
    dav_calls2.append(info)
    print(f"  [Davidson #{len(dav_calls2)}] diag={diag.shape}, "
          f"x0[0]={info['x0_shape']}, "
          f"caller={info['caller'].filename.split('/')[-1]}:{info['caller'].lineno}")
    return orig_dav(matvec, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = traced_dav2

# Manually follow RADC.kernel steps
myadc2 = adc.ADC(mf)
myadc2.method = "adc(2)"

# Step 1: Transform integrals
eris2 = myadc2.transform_integrals()

# Step 2: Compute amplitudes
myadc2.e_corr, myadc2.t1, myadc2.t2 = radc_amplitudes.compute_amplitudes_energy(
    myadc2, eris=eris2, verbose=0)

# Step 3: Check _finalize
if hasattr(myadc2, '_finalize'):
    print(f"Calling _finalize...")
    myadc2._finalize()
    print(f"After _finalize:")
    print(f"  method_type: {myadc2.method_type}")

# Step 4: Create RADCEE
adc_es2 = radc_ee.RADCEE(myadc2)

# Step 5: Check what gen_matvec returns
imds2 = adc_es2.get_imds(eris2)
mv2, diag2 = adc_es2.gen_matvec(imds2, eris2)
print(f"gen_matvec: diag={diag2.shape}")

# Step 6: Call kernel WITH eris (like ee_adc does)
print(f"\nCalling adc_es2.kernel(nroots=5, eris=eris2)...")
result2 = adc_es2.kernel(nroots=5, eris=eris2)
print(f"Result 2: {result2[0]}")

# Step 7: Call kernel WITHOUT eris
print(f"\nCalling adc_es2b.kernel(nroots=5) WITHOUT eris...")
adc_es2b = radc_ee.RADCEE(myadc2)
dav_calls2.clear()
result2b = adc_es2b.kernel(nroots=5)
print(f"Result 2b: {result2b[0]}")

linalg_helper.davidson_nosym1 = orig_dav

# ============================================================
# Test 3: Check if RADC.kernel modifies something
# ============================================================
print("\n" + "="*60)
print("Test 3: Checking what RADC.kernel modifies")
print("="*60)

myadc3 = adc.ADC(mf)
myadc3.method = "adc(2)"

# Check attributes before kernel
attrs_before = set(dir(myadc3))
vals_before = {}
for attr in ['t1', 't2', 'e_corr', 'method_type', 'transform_integrals']:
    if hasattr(myadc3, attr):
        val = getattr(myadc3, attr)
        vals_before[attr] = type(val).__name__
        print(f"  Before: {attr} = {type(val).__name__}")

# Temporarily patch RADCEE.__init__ to see what it receives
orig_init = radc_ee.RADCEE.__init__
def traced_init(self, adc_obj):
    print(f"\n  RADCEE.__init__ called:")
    print(f"    type(adc_obj): {type(adc_obj)}")
    print(f"    adc_obj.method: {adc_obj.method}")
    print(f"    adc_obj.method_type: {adc_obj.method_type}")
    if hasattr(adc_obj, 't1') and adc_obj.t1 is not None:
        print(f"    t1: list of {len(adc_obj.t1)} arrays")
        for k, t in enumerate(adc_obj.t1):
            print(f"      t1[{k}] shape: {t.shape}")
    if hasattr(adc_obj, 't2') and adc_obj.t2 is not None:
        print(f"    t2: list of {len(adc_obj.t2)} arrays")
        for k, t in enumerate(adc_obj.t2):
            print(f"      t2[{k}] shape: {t.shape}")
    orig_init(self, adc_obj)
    print(f"    After init - self.t1: list of {len(self.t1)} arrays")
    for k, t in enumerate(self.t1):
        print(f"      self.t1[{k}] shape: {t.shape}")
    print(f"    After init - self.t2: list of {len(self.t2)} arrays")
    for k, t in enumerate(self.t2):
        print(f"      self.t2[{k}] shape: {t.shape}")

radc_ee.RADCEE.__init__ = traced_init

# Also patch gen_matvec to see what it does
orig_gen = radc_ee.RADCEE.gen_matvec
def traced_gen(self, imds=None, eris=None):
    print(f"\n  gen_matvec called:")
    print(f"    self.t1: list of {len(self.t1)} arrays")
    for k, t in enumerate(self.t1):
        print(f"      self.t1[{k}] shape: {t.shape}")
    result = orig_gen(self, imds, eris)
    mv, diag = result
    print(f"    Returned diag shape: {diag.shape}")
    return result

radc_ee.RADCEE.gen_matvec = traced_gen

dav_calls3 = []
def traced_dav3(matvec, x0, diag, **kwargs):
    info = {'diag_shape': diag.shape}
    dav_calls3.append(info)
    print(f"  [Davidson] diag={diag.shape}")
    return orig_dav(matvec, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = traced_dav3

print("\nRunning myadc3.kernel()...")
result3 = myadc3.kernel(nroots=5)
print(f"Result 3: {result3[0]}")
print(f"Davidson calls: {len(dav_calls3)}")

# Restore everything
radc_ee.RADCEE.__init__ = orig_init
radc_ee.RADCEE.gen_matvec = orig_gen
linalg_helper.davidson_nosym1 = orig_dav
