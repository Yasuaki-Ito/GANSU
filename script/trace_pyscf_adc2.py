"""
Trace PySCF's actual ADC(2) computation to find the real sigma function.
Intercept the Davidson solver to capture the actual matvec.

Usage: python trace_pyscf_adc2.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, lib
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

# ======================================================================
# Step 1: Read radc_ee.matvec source from the INSTALLED version
# ======================================================================
print("\n=== Step 1: Actual matvec source ===")
src = inspect.getsource(radc_ee.matvec)
print(src[:3000])

# ======================================================================
# Step 2: Read radc.RADC.kernel source
# ======================================================================
print("\n=== Step 2: RADC.kernel source ===")
try:
    kernel_src = inspect.getsource(radc.RADC.kernel)
    print(kernel_src[:3000])
except:
    # kernel might be inherited
    kernel_src = inspect.getsource(radc.RADC.ee_adc)
    print(kernel_src[:3000])

# ======================================================================
# Step 3: Find the actual ee_adc method
# ======================================================================
print("\n=== Step 3: ee_adc source ===")
try:
    ee_src = inspect.getsource(radc.RADC.ee_adc)
    print(ee_src[:5000])
except Exception as e:
    print(f"ee_adc not found: {e}")
    # Look for compute_amplitudes or similar
    for name in ['compute_amplitudes', 'kernel', 'ee_adc', 'compute_excitation']:
        try:
            func = getattr(radc.RADC, name)
            src = inspect.getsource(func)
            print(f"\n--- {name} ---")
            print(src[:3000])
        except:
            pass

# ======================================================================
# Step 4: Monkey-patch to intercept the actual sigma function
# ======================================================================
print("\n=== Step 4: Intercepting actual sigma function ===")

captured_sigma = [None]  # Use list for mutability in closure
captured_dim = [None]

# Patch lib.davidson_nosym1 or whatever Davidson PySCF uses
original_davidson = None
for dav_name in ['davidson_nosym1', 'davidson1', 'davidson_nosym', 'davidson']:
    if hasattr(lib, dav_name):
        original_davidson = getattr(lib, dav_name)
        print(f"Found Davidson solver: lib.{dav_name}")

        def patched_davidson(matvec, *args, **kwargs):
            print(f"  Davidson called! matvec={matvec}")
            captured_sigma[0] = matvec
            # Determine dimension from args
            if len(args) > 1:
                x0 = args[1]
                if hasattr(x0, 'shape'):
                    captured_dim[0] = x0.shape[-1] if x0.ndim > 1 else x0.shape[0]
                    print(f"  x0 shape: {x0.shape}")
            return original_davidson(matvec, *args, **kwargs)

        setattr(lib, dav_name, patched_davidson)
        break

if original_davidson is None:
    # Try linalg_helper
    from pyscf.lib import linalg_helper
    for dav_name in ['davidson_nosym1', 'davidson1', 'davidson_nosym', 'davidson']:
        if hasattr(linalg_helper, dav_name):
            original_davidson = getattr(linalg_helper, dav_name)
            print(f"Found Davidson solver: linalg_helper.{dav_name}")

            def patched_davidson(matvec, *args, **kwargs):
                print(f"  Davidson called! matvec={matvec}")
                captured_sigma[0] = matvec
                if len(args) > 1:
                    x0 = args[1]
                    if hasattr(x0, 'shape'):
                        print(f"  x0 shape: {x0.shape}")
                return original_davidson(matvec, *args, **kwargs)

            setattr(linalg_helper, dav_name, patched_davidson)
            break

# Run ADC(2) with the patched Davidson
myadc2 = adc.ADC(mf)
myadc2.method = "adc(2)"
result2 = myadc2.kernel(nroots=5)
e_adc2 = result2[0]
print(f"\nPySCF ADC(2) eigenvalues: {e_adc2}")

# Restore original
if original_davidson is not None:
    if hasattr(lib, dav_name):
        setattr(lib, dav_name, original_davidson)
    else:
        setattr(linalg_helper, dav_name, original_davidson)

# ======================================================================
# Step 5: Probe the captured sigma function
# ======================================================================
if captured_sigma[0] is not None:
    actual_sigma = captured_sigma[0]
    print(f"\n=== Step 5: Probing captured sigma function ===")

    # Test with unit vector to get dimension
    test = np.zeros(total)
    test[0] = 1.0
    try:
        out = actual_sigma(test)
        dim_actual = len(out)
        print(f"Actual sigma: input dim={total}, output dim={dim_actual}")
    except Exception as e:
        print(f"sigma(total) failed: {e}")
        # Try with just ov dimension
        test = np.zeros(ov)
        test[0] = 1.0
        try:
            out = actual_sigma(test)
            dim_actual = len(out)
            print(f"Actual sigma: input dim={ov}, output dim={dim_actual}")
        except Exception as e2:
            print(f"sigma(ov) also failed: {e2}")
            dim_actual = 0

    if dim_actual > 0:
        print(f"\nProbing with {dim_actual} unit vectors...")
        M_actual = np.zeros((dim_actual, dim_actual))
        for col in range(dim_actual):
            e_vec = np.zeros(dim_actual)
            e_vec[col] = 1.0
            M_actual[:, col] = actual_sigma(e_vec)

        print(f"Actual matrix: shape={M_actual.shape}")
        print(f"Symmetric: {np.allclose(M_actual, M_actual.T)}")
        print(f"Max asymmetry: {np.max(np.abs(M_actual - M_actual.T)):.6e}")

        if np.allclose(M_actual, M_actual.T):
            evals_actual = np.linalg.eigvalsh(M_actual)
        else:
            evals_actual = np.sort(np.linalg.eig(M_actual)[0].real)

        print(f"Actual matrix eigenvalues[:10]: {evals_actual[:10]}")
        print(f"PySCF ADC(2) eigenvalues:       {e_adc2}")

        if dim_actual == total:
            # Compare with our probed matrix
            M_ext = np.zeros((total, total))
            M_ab = radc_ee.get_imds(myadc2)
            eris = myadc2.transform_integrals()
            sigma_ext = radc_ee.matvec(myadc2, M_ab=M_ab, eris=eris)
            for col in range(total):
                e_vec = np.zeros(total)
                e_vec[col] = 1.0
                M_ext[:, col] = sigma_ext(e_vec)

            print(f"\nMax diff actual vs radc_ee.matvec: {np.max(np.abs(M_actual - M_ext)):.6e}")
else:
    print("Could not capture sigma function!")

    # Alternative: look at what radc_ee.RADCEE does
    print("\n=== Alternative: check RADCEE class ===")
    try:
        radcee_src = inspect.getsource(radc_ee.RADCEE)
        print(radcee_src[:5000])
    except Exception as e:
        print(f"Error: {e}")
