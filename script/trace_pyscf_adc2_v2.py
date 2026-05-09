"""
Trace PySCF's actual ADC(2) sigma function by reading radc.kernel source
and intercepting the correct Davidson call.

Usage: python trace_pyscf_adc2_v2.py
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

# Read radc.kernel (standalone function, not class method)
print("\n=== radc.kernel standalone function ===")
# Find it
if hasattr(radc, 'kernel') and not isinstance(radc.kernel, type):
    func = radc.kernel
    if callable(func):
        src = inspect.getsource(func)
        print(src[:5000])
        print("..." if len(src) > 5000 else "")

# Also check what Davidson function is actually used
print("\n=== Searching for Davidson calls in radc and radc_ee ===")
for mod_name, mod in [('radc', radc), ('radc_ee', radc_ee)]:
    src = inspect.getsource(mod)
    for i, line in enumerate(src.split('\n')):
        if 'davidson' in line.lower() or 'eigsh' in line.lower() or 'eigh' in line.lower():
            lines = src.split('\n')
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"  {mod_name}:{j+1}: {lines[j]}")
            print()

# Now let's properly intercept by patching ALL possible Davidson functions
print("\n=== Intercepting ALL Davidson variants ===")
from pyscf.lib import linalg_helper

captured = {'sigma': None, 'x0': None}

# List all Davidson-like functions
for name in dir(linalg_helper):
    obj = getattr(linalg_helper, name)
    if callable(obj) and 'davidson' in name.lower():
        print(f"  linalg_helper.{name}: {type(obj)}")

# Patch all of them
originals = {}
for name in dir(linalg_helper):
    if 'davidson' in name.lower() and callable(getattr(linalg_helper, name)):
        originals[name] = getattr(linalg_helper, name)

        def make_patch(orig_func, func_name):
            def patched(*args, **kwargs):
                print(f"  >>> {func_name} called!")
                if len(args) > 0:
                    captured['sigma'] = args[0]
                    print(f"      sigma type: {type(args[0])}")
                if len(args) > 1:
                    x0 = args[1]
                    if hasattr(x0, 'shape'):
                        captured['x0'] = x0
                        print(f"      x0 shape: {x0.shape}")
                    elif isinstance(x0, (list, tuple)):
                        print(f"      x0 is list/tuple of len {len(x0)}")
                        if len(x0) > 0 and hasattr(x0[0], 'shape'):
                            print(f"      x0[0] shape: {x0[0].shape}")
                return orig_func(*args, **kwargs)
            return patched

        setattr(linalg_helper, name, make_patch(originals[name], name))

# Also patch lib level
for name in dir(lib):
    if 'davidson' in name.lower() and callable(getattr(lib, name)):
        if name not in originals:
            originals[f'lib.{name}'] = getattr(lib, name)
            def make_lib_patch(orig_func, func_name):
                def patched(*args, **kwargs):
                    print(f"  >>> lib.{func_name} called!")
                    if len(args) > 0:
                        captured['sigma'] = args[0]
                    if len(args) > 1:
                        x0 = args[1]
                        if hasattr(x0, 'shape'):
                            captured['x0'] = x0
                            print(f"      x0 shape: {x0.shape}")
                        elif isinstance(x0, (list, tuple)):
                            print(f"      x0 is list/tuple of len {len(x0)}")
                    return orig_func(*args, **kwargs)
                return patched
            setattr(lib, name, make_lib_patch(originals[f'lib.{name}'], name))

# Run ADC(2)
print("\nRunning ADC(2)...")
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"\nPySCF ADC(2): {e_adc2}")

# Restore originals
for name, func in originals.items():
    if name.startswith('lib.'):
        setattr(lib, name[4:], func)
    else:
        setattr(linalg_helper, name, func)

# Probe the captured sigma
if captured['sigma'] is not None:
    sigma_fn = captured['sigma']
    print(f"\n=== Probing captured sigma ===")

    # Determine dimension
    for test_dim in [total, ov]:
        try:
            test = np.zeros(test_dim)
            test[0] = 1.0
            out = sigma_fn(test)
            print(f"sigma works with dim={test_dim}, output dim={len(out)}")
            dim = test_dim
            break
        except Exception as e:
            print(f"sigma with dim={test_dim} failed: {e}")
            dim = 0

    if dim > 0:
        print(f"Building {dim}x{dim} matrix...")
        M_cap = np.zeros((dim, dim))
        for col in range(dim):
            e_vec = np.zeros(dim)
            e_vec[col] = 1.0
            M_cap[:, col] = sigma_fn(e_vec)

        print(f"Symmetric: {np.allclose(M_cap, M_cap.T)}")
        print(f"Max asymmetry: {np.max(np.abs(M_cap - M_cap.T)):.6e}")

        if np.allclose(M_cap, M_cap.T):
            evals = np.linalg.eigvalsh(M_cap)
        else:
            evals = np.sort(np.linalg.eig(M_cap)[0].real)
        print(f"Eigenvalues[:10]: {evals[:10]}")
        print(f"PySCF ADC(2):     {e_adc2}")
        print(f"Match: {np.allclose(evals[:len(e_adc2)], e_adc2, atol=1e-6)}")
else:
    print("\nDavidson was NOT intercepted! Checking alternative solvers...")

    # Maybe PySCF uses scipy or numpy directly
    src = inspect.getsource(radc)
    for i, line in enumerate(src.split('\n')):
        if any(kw in line.lower() for kw in ['eigh', 'eigsh', 'eigvals', 'solver', 'precond']):
            lines = src.split('\n')
            start = max(0, i-2)
            end = min(len(lines), i+5)
            for j in range(start, end):
                print(f"  radc:{j+1}: {lines[j]}")
            print()
