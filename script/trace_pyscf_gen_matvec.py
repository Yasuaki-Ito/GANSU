"""
Find and read PySCF's gen_matvec function which produces the 55-dim sigma.
Then probe the actual sigma function used by Davidson.

Usage: python trace_pyscf_gen_matvec.py
"""
import numpy as np
from pyscf import gto, scf, adc, lib
from pyscf.adc import radc_ee, radc
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
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
total = ov + dd
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}, total={total}")

# Find gen_matvec
print("\n=== Searching for gen_matvec ===")
for mod_name, mod in [('radc', radc), ('radc_ee', radc_ee)]:
    if hasattr(mod, 'gen_matvec'):
        print(f"Found gen_matvec in {mod_name}")
        src = inspect.getsource(mod.gen_matvec)
        print(src[:8000])
        if len(src) > 8000:
            print(f"... ({len(src)} chars total)")

# Search in RADCEE class
print("\n=== RADCEE.gen_matvec ===")
if hasattr(radc_ee.RADCEE, 'gen_matvec'):
    src = inspect.getsource(radc_ee.RADCEE.gen_matvec)
    print(src[:8000])

# Search in RADC class
print("\n=== RADC.gen_matvec ===")
if hasattr(radc.RADC, 'gen_matvec'):
    src = inspect.getsource(radc.RADC.gen_matvec)
    print(src[:8000])

# Search everywhere
print("\n=== Searching all ADC modules ===")
import pyscf.adc
adc_modules = []
import os
adc_dir = os.path.dirname(pyscf.adc.__file__)
for fname in sorted(os.listdir(adc_dir)):
    if fname.endswith('.py'):
        fpath = os.path.join(adc_dir, fname)
        with open(fpath) as f:
            content = f.read()
        if 'gen_matvec' in content:
            print(f"  gen_matvec found in {fname}")
            for i, line in enumerate(content.split('\n')):
                if 'gen_matvec' in line:
                    lines = content.split('\n')
                    start = max(0, i-1)
                    end = min(len(lines), i+3)
                    for j in range(start, end):
                        print(f"    {j+1}: {lines[j]}")
                    print()

# Now actually get the sigma function and probe it
print("\n=== Getting the actual sigma function ===")
myadc = adc.ADC(mf)
myadc.method = "adc(2)"

# Compute amplitudes first (needed for matvec)
from pyscf.adc import radc_amplitudes
eris = myadc.transform_integrals()
myadc.e_corr, myadc.t1, myadc.t2 = radc_amplitudes.compute_amplitudes_energy(
    myadc, eris=eris, verbose=0)

# Now get gen_matvec
adc_es = radc_ee.RADCEE(myadc)

# Try gen_matvec
try:
    imds = adc_es.get_imds(eris)
    print(f"imds type: {type(imds)}, shape: {imds.shape if hasattr(imds, 'shape') else 'N/A'}")
    matvec_fn, diag = adc_es.gen_matvec(imds, eris)
    print(f"gen_matvec returned!")
    print(f"matvec_fn type: {type(matvec_fn)}")
    print(f"diag shape: {diag.shape}")
    print(f"diag[:15]: {diag[:15]}")

    # Test dimension
    test = np.zeros(len(diag))
    test[0] = 1.0
    out = matvec_fn(test)
    dim = len(out)
    print(f"sigma dimension: {dim}")

    # Probe the matrix
    print(f"\nProbing {dim}x{dim} matrix...")
    M = np.zeros((dim, dim))
    for col in range(dim):
        e = np.zeros(dim)
        e[col] = 1.0
        M[:, col] = matvec_fn(e)

    print(f"Symmetric: {np.allclose(M, M.T)}")
    print(f"Max asymmetry: {np.max(np.abs(M - M.T)):.6e}")

    if np.allclose(M, M.T):
        evals = np.linalg.eigvalsh(M)
    else:
        evals = np.sort(np.linalg.eig(M)[0].real)
    print(f"Eigenvalues[:10]: {evals[:10]}")

    # Compare with PySCF
    result = myadc.kernel(nroots=5)
    e_adc2 = result[0]
    print(f"PySCF ADC(2):     {e_adc2}")

    n = min(5, len(e_adc2))
    print(f"\nDifferences:")
    for i in range(n):
        print(f"  State {i+1}: {evals[i]:.10f} vs {e_adc2[i]:.10f}, diff={evals[i]-e_adc2[i]:.2e}")

    match = np.allclose(evals[:n], e_adc2[:n], atol=1e-6)
    print(f"\nMatch: {match}")

    if match:
        print("\n*** SUCCESS: gen_matvec in 55-dim space gives correct ADC(2) eigenvalues! ***")

        # Analyze the structure
        print(f"\n=== Analyzing 55-dim space structure ===")
        print(f"Singles part: 0..{ov-1} (dim={ov})")
        doubles_dim = dim - ov
        print(f"Doubles part: {ov}..{dim-1} (dim={doubles_dim})")
        print(f"Expected: nocc*(nocc+1)/2 * nvir*(nvir+1)/2 = {nocc*(nocc+1)//2 * nvir*(nvir+1)//2}")
        print(f"  + nocc*(nocc-1)/2 * nvir*(nvir-1)/2 = {nocc*(nocc-1)//2 * nvir*(nvir-1)//2}")
        print(f"  = {nocc*(nocc+1)//2 * nvir*(nvir+1)//2 + nocc*(nocc-1)//2 * nvir*(nvir-1)//2}")

        # Extract blocks
        M11 = M[:ov, :ov]
        M12_55 = M[:ov, ov:]
        M21_55 = M[ov:, :ov]
        D2_55 = M[ov:, ov:]

        print(f"\nM11 (55-dim): shape={M11.shape}, sym={np.allclose(M11, M11.T)}")
        print(f"M12_55: shape={M12_55.shape}")
        print(f"M21_55: shape={M21_55.shape}")
        print(f"M12_55 == M21_55^T: {np.allclose(M12_55, M21_55.T)}")
        print(f"Max |M12 - M21^T|: {np.max(np.abs(M12_55 - M21_55.T)):.6e}")
        print(f"D2_55 diagonal: {np.allclose(D2_55, np.diag(np.diag(D2_55)))}")

except Exception as e:
    print(f"gen_matvec failed: {e}")
    import traceback
    traceback.print_exc()
