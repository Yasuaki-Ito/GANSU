"""
Understand why PySCF's Davidson uses 55-dim vectors, not 110-dim.
Read get_diag and get_init_guess source, and properly probe the captured sigma.

Usage: python trace_pyscf_dim55.py
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
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}")

# Read get_diag source
print("\n=== radc_ee.get_diag source ===")
src = inspect.getsource(radc_ee.get_diag)
print(src[:5000])

# Read get_init_guess source
print("\n=== RADCEE.get_init_guess source (first part) ===")
src = inspect.getsource(radc_ee.RADCEE.get_init_guess)
print(src[:5000])

# Read get_diag to understand dimension
print("\n=== radc_ee.get_diag ===")
src = inspect.getsource(radc_ee.get_diag)
# Look for dimension-related lines
for i, line in enumerate(src.split('\n')):
    if any(kw in line for kw in ['n_singles', 'n_doubles', 'dim', 'shape', 'zeros', 'arange']):
        print(f"  {i+1}: {line}")

# Now properly intercept Davidson and probe
print("\n=== Proper Davidson interception ===")

captured = {'batched_sigma': None, 'x0': None, 'diag': None}

original_dav = linalg_helper.davidson_nosym1
def patched_dav(matvec_batch, x0, diag, **kwargs):
    captured['batched_sigma'] = matvec_batch
    captured['x0'] = x0
    captured['diag'] = diag
    print(f"  Davidson called:")
    print(f"    diag shape: {diag.shape}")
    if isinstance(x0, (list, tuple)):
        print(f"    x0: list of {len(x0)} vectors, x0[0].shape={x0[0].shape}")
    elif hasattr(x0, 'shape'):
        print(f"    x0 shape: {x0.shape}")
    return original_dav(matvec_batch, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = patched_dav

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

linalg_helper.davidson_nosym1 = original_dav

# Probe the captured batched sigma
if captured['batched_sigma'] is not None:
    batched_sigma = captured['batched_sigma']
    diag = captured['diag']
    dim = len(diag)
    print(f"\n=== Probing captured batched sigma (dim={dim}) ===")

    # The batched sigma takes a LIST of vectors and returns a LIST
    # Test with a single unit vector
    test = np.zeros(dim)
    test[0] = 1.0
    result_list = batched_sigma([test])
    print(f"batched_sigma([e_0]) returned list of len {len(result_list)}")
    print(f"result[0] shape: {result_list[0].shape}")
    print(f"result[0][:5]: {result_list[0][:5]}")

    # Build full matrix
    print(f"Building {dim}x{dim} matrix...")
    M = np.zeros((dim, dim))
    for col in range(dim):
        e = np.zeros(dim)
        e[col] = 1.0
        sigma_col = batched_sigma([e])[0]
        M[:, col] = sigma_col

    print(f"Symmetric: {np.allclose(M, M.T)}")
    print(f"Max asymmetry: {np.max(np.abs(M - M.T)):.6e}")

    if np.allclose(M, M.T):
        evals = np.linalg.eigvalsh(M)
    else:
        evals = np.sort(np.linalg.eig(M)[0].real)

    print(f"Eigenvalues[:10]: {evals[:10]}")
    print(f"PySCF ADC(2):     {e_adc2}")

    n = min(5, len(e_adc2))
    match = np.allclose(evals[:n], e_adc2[:n], atol=1e-6)
    print(f"Match: {match}")

    if match:
        print("\n*** MATCH! The captured sigma gives correct eigenvalues! ***")

        # Analyze structure
        M11 = M[:ov, :ov]
        M12 = M[:ov, ov:]
        M21 = M[ov:, :ov]
        D2_block = M[ov:, ov:]
        dd_actual = dim - ov
        print(f"\nSingles dim: {ov}")
        print(f"Doubles dim: {dd_actual}")
        print(f"M11: sym={np.allclose(M11, M11.T)}")
        print(f"M12 == M21^T: {np.allclose(M12, M21.T)}")
        print(f"Max |M12-M21^T|: {np.max(np.abs(M12 - M21.T)):.6e}")
        print(f"D2 diagonal: {np.allclose(D2_block, np.diag(np.diag(D2_block)))}")
