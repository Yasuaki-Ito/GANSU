"""
Three-pronged approach to understand the correct ADC(2):

1. Check if PySCF ADC(2) eigenvalues exist in the 110-dim matrix spectrum
2. Read the ACTUAL installed radc_ee.py and radc.py source files to find
   any 55-dim transformation code
3. Check if davidson_nosym1 does any internal transformation

Usage: python find_correct_adc2.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, lib
from pyscf.adc import radc_ee, radc
from pyscf.lib import linalg_helper
import inspect
import os

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
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
total = ov + dd
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}, total={total}")

# Get PySCF reference
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

# ============================================================
# Part 1: Full 110-dim spectrum
# ============================================================
print(f"\n{'='*60}")
print(f"Part 1: Full 110-dim matrix spectrum")
print(f"{'='*60}")

eris = myadc.transform_integrals()
M_ab = radc_ee.get_imds(myadc, eris=eris)
sigma_fn = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)

print(f"Building {total}x{total} matrix...")
M110 = np.zeros((total, total))
for col in range(total):
    e = np.zeros(total)
    e[col] = 1.0
    M110[:, col] = sigma_fn(e)

# General eigenvalues (non-symmetric matrix)
print(f"Diagonalizing 110-dim matrix...")
evals_110_complex = np.linalg.eig(M110)[0]
# Check if eigenvalues are real
max_imag = np.max(np.abs(evals_110_complex.imag))
print(f"Max imaginary part: {max_imag:.6e}")
evals_110 = np.sort(evals_110_complex.real)

print(f"\nAll 110 eigenvalues (real parts):")
for i in range(0, len(evals_110), 10):
    chunk = evals_110[i:i+10]
    print(f"  [{i:3d}-{i+len(chunk)-1:3d}]: {chunk}")

print(f"\nPySCF ADC(2) eigenvalues: {e_adc2}")

# Check if any of the 110 eigenvalues match PySCF's
print(f"\nSearching for PySCF eigenvalues in 110-dim spectrum:")
for k, e_ref in enumerate(e_adc2):
    diffs = np.abs(evals_110 - e_ref)
    closest_idx = np.argmin(diffs)
    closest_val = evals_110[closest_idx]
    print(f"  PySCF state {k}: {e_ref:.10f}  closest 110-dim: {closest_val:.10f}  diff: {diffs[closest_idx]:.6e}")


# ============================================================
# Part 2: Read installed source files
# ============================================================
print(f"\n{'='*60}")
print(f"Part 2: Searching installed PySCF source for 55-dim code")
print(f"{'='*60}")

# Read the installed radc_ee.py
radc_ee_path = inspect.getfile(radc_ee)
print(f"\nReading: {radc_ee_path}")
with open(radc_ee_path) as f:
    radc_ee_src = f.read()
radc_ee_lines = radc_ee_src.split('\n')
print(f"Total lines: {len(radc_ee_lines)}")

# Search for keywords related to spin adaptation / dimension reduction
keywords = ['spin_adapt', 'singlet_adapt', 'pack', 'unpack', 'tril', 'triu',
            'n_doubles_red', 'n_packed', 'symmetr', 'fold', 'compact',
            'spin_cas', 'spin_block', 'ab_sym', 'ij_sym',
            'n_doubles =', 'dim =', 'shape']
print(f"\nSearching for relevant keywords:")
for kw in keywords:
    matches = [(i+1, line.strip()) for i, line in enumerate(radc_ee_lines)
               if kw.lower() in line.lower() and not line.strip().startswith('#')]
    if matches:
        print(f"\n  '{kw}' ({len(matches)} matches):")
        for lineno, line in matches[:5]:
            print(f"    L{lineno}: {line[:120]}")

# Read the installed radc.py
radc_path = inspect.getfile(radc)
print(f"\n\nReading: {radc_path}")
with open(radc_path) as f:
    radc_src = f.read()
radc_lines = radc_src.split('\n')
print(f"Total lines: {len(radc_lines)}")

# Search for symmetry/packing code
print(f"\nSearching radc.py:")
for kw in ['spin_adapt', 'pack', 'fold', 'compact', '55', 'singlet']:
    matches = [(i+1, line.strip()) for i, line in enumerate(radc_lines)
               if kw.lower() in line.lower() and not line.strip().startswith('#')]
    if matches:
        print(f"\n  '{kw}' ({len(matches)} matches):")
        for lineno, line in matches[:5]:
            print(f"    L{lineno}: {line[:120]}")

# Check all files in the adc directory
adc_dir = os.path.dirname(inspect.getfile(radc_ee))
print(f"\n\nAll files in {adc_dir}:")
for fname in sorted(os.listdir(adc_dir)):
    if fname.endswith('.py'):
        fpath = os.path.join(adc_dir, fname)
        with open(fpath) as f:
            content = f.read()
        n_lines = len(content.split('\n'))
        has_spin = 'spin_adapt' in content.lower() or 'singlet_adapt' in content.lower()
        has_pack = 'pack' in content.lower()
        has_fold = 'fold' in content.lower()
        has_sym = 'symmetr' in content.lower() and 'spin' in content.lower()
        flags = []
        if has_spin: flags.append('SPIN_ADAPT')
        if has_pack: flags.append('PACK')
        if has_fold: flags.append('FOLD')
        if has_sym: flags.append('SPIN_SYM')
        print(f"  {fname:30s} ({n_lines:5d} lines) {' '.join(flags)}")


# ============================================================
# Part 3: Check davidson_nosym1 source
# ============================================================
print(f"\n{'='*60}")
print(f"Part 3: Davidson solver inspection")
print(f"{'='*60}")

dav_src = inspect.getsource(linalg_helper.davidson_nosym1)
print(f"davidson_nosym1 source ({len(dav_src)} chars, {len(dav_src.split(chr(10)))} lines)")

# Search for any transformation/spin adaptation in Davidson
for kw in ['spin', 'adapt', 'symmetr', 'pack', 'fold', 'compact', 'project', 'singlet']:
    if kw.lower() in dav_src.lower():
        print(f"  WARNING: '{kw}' found in davidson_nosym1!")
        for i, line in enumerate(dav_src.split('\n')):
            if kw.lower() in line.lower():
                print(f"    L{i+1}: {line.strip()[:100]}")

# Check the first 50 lines for any dimension transformation
dav_lines = dav_src.split('\n')
print(f"\nFirst 30 lines of davidson_nosym1:")
for i, line in enumerate(dav_lines[:30]):
    print(f"  {i+1:3d}: {line}")


# ============================================================
# Part 4: Trace the actual kernel execution
# ============================================================
print(f"\n{'='*60}")
print(f"Part 4: Tracing actual kernel with dimension checks")
print(f"{'='*60}")

# Patch get_diag, matvec, get_init_guess, and davidson
adc_es = radc_ee.RADCEE(myadc)

# Patch get_diag
orig_get_diag = type(adc_es).get_diag
def traced_get_diag(adc_obj, M_ab=None, eris=None):
    result = orig_get_diag(adc_obj, M_ab, eris)
    print(f"  [get_diag] returned shape={result.shape}")
    return result

# Patch matvec
orig_matvec = type(adc_es).matvec
def traced_matvec(adc_obj, M_ab=None, eris=None):
    sigma = orig_matvec(adc_obj, M_ab, eris)
    # Wrap sigma to trace calls
    def traced_sigma(r):
        result = sigma(r)
        print(f"  [sigma] input={len(r)}, output={len(result)}")
        return result
    return traced_sigma

# Patch get_init_guess
orig_get_init = type(adc_es).get_init_guess
def traced_get_init(self, nroots=1, diag=None, ascending=True, type=None, eris=None, ini=None):
    print(f"  [get_init_guess] called, type={type}, diag.shape={diag.shape if diag is not None else None}")
    result = orig_get_init(self, nroots, diag, ascending, type, eris, ini)
    if isinstance(result, list) and len(result) > 0:
        print(f"  [get_init_guess] returned {len(result)} vectors, shape={result[0].shape}")
    return result

# Patch davidson
orig_dav = linalg_helper.davidson_nosym1
call_count = [0]
def traced_dav(matvec, x0, diag, **kwargs):
    call_count[0] += 1
    print(f"  [davidson #{call_count[0]}] diag.shape={diag.shape}")
    if isinstance(x0, list):
        print(f"  [davidson #{call_count[0]}] x0: {len(x0)} vectors, x0[0].shape={x0[0].shape}")
    # Test matvec dimension
    test = [np.zeros(len(diag))]
    test[0][0] = 1.0
    out = matvec(test)
    print(f"  [davidson #{call_count[0]}] matvec test: in={len(test[0])}, out={len(out[0])}")
    return orig_dav(matvec, x0, diag, **kwargs)

type(adc_es).get_diag = traced_get_diag
type(adc_es).matvec = traced_matvec
type(adc_es).get_init_guess = traced_get_init
linalg_helper.davidson_nosym1 = traced_dav

print("Running kernel with traces...")
try:
    result2 = adc_es.kernel(nroots=5)
    print(f"Result: {result2[0]}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Restore
type(adc_es).get_diag = orig_get_diag
type(adc_es).matvec = orig_matvec
type(adc_es).get_init_guess = orig_get_init
linalg_helper.davidson_nosym1 = orig_dav
