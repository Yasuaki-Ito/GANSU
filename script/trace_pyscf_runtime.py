"""
Trace the EXACT runtime code path in PySCF's ADC(2) to find
where the 55-dim representation is constructed.

Print PySCF version, actual method resolution, and source code.

Usage: python trace_pyscf_runtime.py
"""
import numpy as np
import pyscf
from pyscf import gto, scf, adc, lib
from pyscf.adc import radc_ee, radc
from pyscf.lib import linalg_helper
import inspect
import sys

print(f"PySCF version: {pyscf.__version__}")
print(f"PySCF location: {pyscf.__file__}")
print(f"Python version: {sys.version}")

# Check radc_ee location
print(f"\nradc_ee location: {radc_ee.__file__}")
print(f"radc location: {radc.__file__}")

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

# Create the ADC object and prepare for EE
myadc = adc.ADC(mf)
myadc.method = "adc(2)"

# Step 1: Check RADC class and method resolution
print(f"\n=== Class hierarchy ===")
print(f"type(myadc): {type(myadc)}")
print(f"MRO: {[c.__name__ for c in type(myadc).__mro__]}")

# Step 2: Check what ee_adc does
print(f"\n=== ee_adc source ===")
src = inspect.getsource(type(myadc).ee_adc)
print(src)

# Step 3: Manually follow the path that kernel takes
print(f"\n=== Following kernel path ===")
# kernel computes amplitudes then calls ee_adc
eris = myadc.transform_integrals()
myadc.e_corr, myadc.t1, myadc.t2 = radc.radc_amplitudes.compute_amplitudes_energy(
    myadc, eris=eris, verbose=0)

# ee_adc creates RADCEE
adc_es = radc_ee.RADCEE(myadc)
print(f"type(adc_es): {type(adc_es)}")
print(f"MRO: {[c.__name__ for c in type(adc_es).__mro__]}")

# Step 4: Check ACTUAL gen_matvec method
print(f"\n=== Actual gen_matvec ===")
for cls in type(adc_es).__mro__:
    if 'gen_matvec' in cls.__dict__:
        print(f"gen_matvec defined in: {cls.__name__}")
        print(f"Source file: {inspect.getfile(cls.gen_matvec)}")
        src = inspect.getsource(cls.gen_matvec)
        print(src)
        break

# Step 5: Check ACTUAL get_diag method
print(f"\n=== Actual get_diag ===")
for cls in type(adc_es).__mro__:
    if 'get_diag' in cls.__dict__:
        print(f"get_diag defined in: {cls.__name__}")
        print(f"Source file: {inspect.getfile(cls.__dict__['get_diag'])}")
        # Check if it's a function or descriptor
        gd = cls.__dict__['get_diag']
        print(f"Type: {type(gd)}")
        if callable(gd):
            src = inspect.getsource(gd)
            print(src[:3000])
        break

# Step 6: Check ACTUAL matvec method
print(f"\n=== Actual matvec ===")
for cls in type(adc_es).__mro__:
    if 'matvec' in cls.__dict__:
        print(f"matvec defined in: {cls.__name__}")
        print(f"Source file: {inspect.getfile(cls.__dict__['matvec'])}")
        mv = cls.__dict__['matvec']
        print(f"Type: {type(mv)}")
        if callable(mv):
            src = inspect.getsource(mv)
            print(src[:5000])
        break

# Step 7: Actually call gen_matvec and check dimensions
print(f"\n=== Calling gen_matvec ===")
imds = adc_es.get_imds(eris)
print(f"imds type: {type(imds)}")
if hasattr(imds, 'shape'):
    print(f"imds shape: {imds.shape}")

matvec_fn, diag = adc_es.gen_matvec(imds, eris)
print(f"diag shape: {diag.shape}")
print(f"diag[:15]: {diag[:15]}")

# Test matvec
test = np.zeros(len(diag))
test[0] = 1.0
out = matvec_fn(test)
print(f"matvec: input_dim={len(diag)}, output_dim={len(out)}")

# Step 8: Check kernel (standalone function)
print(f"\n=== Standalone kernel ===")
print(f"RADCEE.kernel is radc.kernel: {adc_es.kernel is radc.kernel}")
kernel_src = inspect.getsource(radc.kernel)
print(kernel_src[:3000])

# Step 9: Check if there's a different davidson
print(f"\n=== Davidson function ===")
dav_fn = linalg_helper.davidson_nosym1
print(f"davidson_nosym1 location: {inspect.getfile(dav_fn)}")

# Step 10: Direct inspection - what does matvec return?
print(f"\n=== Direct matvec inspection ===")
# Get the actual sigma_ closure
sigma_ = radc_ee.matvec(adc_es, M_ab=imds, eris=eris)
print(f"Module-level matvec returned: {type(sigma_)}")

# Test with 110-dim vector
nocc = adc_es._nocc
nvir = adc_es._nvir
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
total_110 = ov + dd
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}, total_110={total_110}")

test110 = np.zeros(total_110)
test110[0] = 1.0
try:
    out110 = sigma_(test110)
    print(f"Module sigma_(110-dim vector): output dim={len(out110)}")
except Exception as e:
    print(f"Module sigma_(110-dim vector) failed: {e}")

# Test with 55-dim vector
total_55 = len(diag)
test55 = np.zeros(total_55)
test55[0] = 1.0
try:
    out55 = sigma_(test55)
    print(f"Module sigma_(55-dim vector): output dim={len(out55)}")
except Exception as e:
    print(f"Module sigma_(55-dim vector) failed: {e}")

# Test the gen_matvec's returned matvec with both dimensions
try:
    out_gm_110 = matvec_fn(np.zeros(total_110))
    print(f"gen_matvec's matvec(110-dim): output dim={len(out_gm_110)}")
except Exception as e:
    print(f"gen_matvec's matvec(110-dim) failed: {e}")

try:
    out_gm_55 = matvec_fn(np.zeros(total_55))
    print(f"gen_matvec's matvec(55-dim): output dim={len(out_gm_55)}")
except Exception as e:
    print(f"gen_matvec's matvec(55-dim) failed: {e}")

# Step 11: Check if the matvec_fn IS the same as sigma_
print(f"\n=== Function identity ===")
print(f"matvec_fn is sigma_: {matvec_fn is sigma_}")
print(f"matvec_fn: {matvec_fn}")
print(f"sigma_: {sigma_}")

# If they're closures, look at their cells
if hasattr(matvec_fn, '__closure__') and matvec_fn.__closure__:
    print(f"matvec_fn closure vars: {[c.cell_contents if hasattr(c, 'cell_contents') else '?' for c in matvec_fn.__closure__[:3]]}")
if hasattr(sigma_, '__closure__') and sigma_.__closure__:
    print(f"sigma_ closure vars (types): {[type(c.cell_contents).__name__ if hasattr(c, 'cell_contents') else '?' for c in sigma_.__closure__[:5]]}")
