"""
Extract the 55-dim ADC(2) matrix blocks from PySCF's captured sigma.
Build M11_55, M12_55, M21_55, D2_55, then test the Schur complement.
Also determine the exact packing convention and derive the relationship
between 55-dim and 110-dim formulas.

Usage: python extract_55dim_blocks.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, lib
from pyscf.adc import radc_ee, radc
from pyscf.lib import linalg_helper

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
dd_full = nocc * nocc * nvir * nvir
n_packed = nocc * (nocc + 1) // 2 * nvir * (nvir + 1) // 2
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd_full={dd_full}, n_packed={n_packed}")
print(f"Expected 55-dim total: {ov + n_packed}")

# ============================================================
# Step 1: Capture PySCF's 55-dim sigma function
# ============================================================
captured = {'sigma': None, 'diag': None}

original_dav = linalg_helper.davidson_nosym1
def patched_dav(matvec_batch, x0, diag, **kwargs):
    captured['sigma'] = matvec_batch
    captured['diag'] = diag
    return original_dav(matvec_batch, x0, diag, **kwargs)

linalg_helper.davidson_nosym1 = patched_dav

myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

linalg_helper.davidson_nosym1 = original_dav

dim = len(captured['diag'])
print(f"\nCaptured dim: {dim}")

# ============================================================
# Step 2: Build the full dim x dim matrix by probing
# ============================================================
sigma_fn = captured['sigma']
print(f"Building {dim}x{dim} matrix...")
M55 = np.zeros((dim, dim))
for col in range(dim):
    e = np.zeros(dim)
    e[col] = 1.0
    M55[:, col] = sigma_fn([e])[0]

# Extract blocks
M11_55 = M55[:ov, :ov]
M12_55 = M55[:ov, ov:]
M21_55 = M55[ov:, :ov]
D2_55  = M55[ov:, ov:]
dd_55 = dim - ov

print(f"\n=== Block shapes ===")
print(f"M11_55: {M11_55.shape}")
print(f"M12_55: {M12_55.shape}")
print(f"M21_55: {M21_55.shape}")
print(f"D2_55:  {D2_55.shape}")

print(f"\n=== Block properties ===")
print(f"M11_55 sym: {np.allclose(M11_55, M11_55.T)}")
print(f"M11_55 max asym: {np.max(np.abs(M11_55 - M11_55.T)):.6e}")
print(f"M12_55 == M21_55^T: {np.allclose(M12_55, M21_55.T)}")
print(f"Max |M12_55 - M21_55^T|: {np.max(np.abs(M12_55 - M21_55.T)):.6e}")
print(f"D2_55 diagonal: {np.allclose(D2_55, np.diag(np.diag(D2_55)))}")
D2_55_diag = np.diag(D2_55)

# ============================================================
# Step 3: Also get the 110-dim sigma
# ============================================================
eris = myadc.transform_integrals()
M_ab = radc_ee.get_imds(myadc, eris=eris)
sigma_110 = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)

print(f"\nBuilding 110x110 matrix...")
M110 = np.zeros((ov + dd_full, ov + dd_full))
for col in range(ov + dd_full):
    e = np.zeros(ov + dd_full)
    e[col] = 1.0
    M110[:, col] = sigma_110(e)

M11_110 = M110[:ov, :ov]
M12_110 = M110[:ov, ov:]
M21_110 = M110[ov:, :ov]
D2_110  = M110[ov:, ov:]

# ============================================================
# Step 4: Compare M11 blocks
# ============================================================
print(f"\n=== M11 comparison (55-dim vs 110-dim) ===")
print(f"Max diff: {np.max(np.abs(M11_55 - M11_110)):.6e}")
if np.allclose(M11_55, M11_110, atol=1e-10):
    print("M11 blocks are IDENTICAL!")
else:
    print("M11 blocks DIFFER!")
    diff = M11_55 - M11_110
    print(f"Diff:\n{diff}")

# ============================================================
# Step 5: Analyze D2 packing
# ============================================================
eps = mf.mo_energy
D2_full_arr = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                D2_full_arr[i,j,a,b] = eps[a+nocc] + eps[b+nocc] - eps[i] - eps[j]

print(f"\n=== D2 packing analysis ===")
print(f"D2_55 diagonal values ({dd_55} values):")
print(f"  {D2_55_diag}")

# Try different packing conventions for D2
# Convention 1: (I<=J, A<=B) with IJ = J*(J+1)/2+I, AB = B*(B+1)/2+A
print(f"\nD2 full unique values (I<=J, A<=B):")
packed_d2_v1 = []
for J in range(nocc):
    for I in range(J+1):  # I <= J
        for B in range(nvir):
            for A in range(B+1):  # A <= B
                d2_val = eps[A+nocc] + eps[B+nocc] - eps[I] - eps[J]
                packed_d2_v1.append(d2_val)
packed_d2_v1 = np.array(packed_d2_v1)
print(f"  Convention 1 (J,I,B,A loop): {len(packed_d2_v1)} values")
if len(packed_d2_v1) == dd_55:
    match = np.allclose(D2_55_diag, packed_d2_v1)
    print(f"  Match: {match}")
    if not match:
        print(f"  Max diff: {np.max(np.abs(D2_55_diag - packed_d2_v1)):.6e}")
        # Try to find the permutation
        for i in range(min(10, dd_55)):
            print(f"  D2_55[{i}]={D2_55_diag[i]:.8f}  v1[{i}]={packed_d2_v1[i]:.8f}")

# Convention 2: (I<=J, A<=B) with IJ = I*nocc - I*(I+1)/2 + J, etc
packed_d2_v2 = []
for I in range(nocc):
    for J in range(I, nocc):  # J >= I
        for A in range(nvir):
            for B in range(A, nvir):  # B >= A
                d2_val = eps[A+nocc] + eps[B+nocc] - eps[I] - eps[J]
                packed_d2_v2.append(d2_val)
packed_d2_v2 = np.array(packed_d2_v2)
print(f"\n  Convention 2 (I,J,A,B loop I<=J A<=B): {len(packed_d2_v2)} values")
if len(packed_d2_v2) == dd_55:
    match = np.allclose(D2_55_diag, packed_d2_v2)
    print(f"  Match: {match}")
    if not match:
        for i in range(min(10, dd_55)):
            print(f"  D2_55[{i}]={D2_55_diag[i]:.8f}  v2[{i}]={packed_d2_v2[i]:.8f}")

# Convention 3: same as full ordering but with tril indexing
packed_d2_v3 = []
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                if j <= i and b <= a:  # lower triangular in both
                    packed_d2_v3.append(D2_full_arr[i,j,a,b])
packed_d2_v3 = np.array(packed_d2_v3)
print(f"\n  Convention 3 (i,j,a,b with j<=i b<=a): {len(packed_d2_v3)} values")
if len(packed_d2_v3) == dd_55:
    match = np.allclose(D2_55_diag, packed_d2_v3)
    print(f"  Match: {match}")
    if not match:
        for i in range(min(10, dd_55)):
            print(f"  D2_55[{i}]={D2_55_diag[i]:.8f}  v3[{i}]={packed_d2_v3[i]:.8f}")

# If none match, try to find the mapping by matching values
print(f"\n=== Finding D2 mapping ===")
D2_full_flat = D2_full_arr.reshape(-1)
perm = []
for k in range(dd_55):
    target = D2_55_diag[k]
    matches = np.where(np.abs(D2_full_flat - target) < 1e-10)[0]
    perm.append(matches.tolist())
    if k < 10:
        # Decode the full-space indices
        match_indices = []
        for m in matches:
            i = m // (nocc * nvir * nvir)
            rem = m % (nocc * nvir * nvir)
            j = rem // (nvir * nvir)
            rem2 = rem % (nvir * nvir)
            a = rem2 // nvir
            b = rem2 % nvir
            match_indices.append((i,j,a,b))
        print(f"  packed[{k}] = {target:.8f} matches full indices: {match_indices}")


# ============================================================
# Step 6: Analyze M12 relationship
# ============================================================
print(f"\n=== M12 relationship (55-dim vs 110-dim) ===")

# For each packed doubles index, check which full doubles columns it corresponds to
# M12_55[:, k] should be some combination of M12_110 columns
for k in range(min(10, dd_55)):
    m12_55_col = M12_55[:, k]
    # Find which full columns match or combine to this
    best_match = -1
    best_diff = 1e10
    for j in range(dd_full):
        diff = np.max(np.abs(m12_55_col - M12_110[:, j]))
        if diff < best_diff:
            best_diff = diff
            best_match = j

    # Also try sum of matching D2 indices
    if perm[k]:
        m12_sum = np.zeros(ov)
        for j in perm[k]:
            m12_sum += M12_110[:, j]
        sum_diff = np.max(np.abs(m12_55_col - m12_sum))

        # Try average
        m12_avg = m12_sum / len(perm[k])
        avg_diff = np.max(np.abs(m12_55_col - m12_avg))

        # Try with sqrt
        m12_sqrt = m12_sum / np.sqrt(len(perm[k]))
        sqrt_diff = np.max(np.abs(m12_55_col - m12_sqrt))

        print(f"  M12_55[:,{k}]: best_single={best_diff:.6e}, sum={sum_diff:.6e}, avg={avg_diff:.6e}, sqrt={sqrt_diff:.6e} (m={len(perm[k])})")


# ============================================================
# Step 7: Analyze M21 relationship
# ============================================================
print(f"\n=== M21 relationship (55-dim vs 110-dim) ===")
for k in range(min(10, dd_55)):
    m21_55_row = M21_55[k, :]
    best_match = -1
    best_diff = 1e10
    for j in range(dd_full):
        diff = np.max(np.abs(m21_55_row - M21_110[j, :]))
        if diff < best_diff:
            best_diff = diff
            best_match = j

    if perm[k]:
        m21_sum = np.zeros(ov)
        for j in perm[k]:
            m21_sum += M21_110[j, :]
        sum_diff = np.max(np.abs(m21_55_row - m21_sum))
        m21_avg = m21_sum / len(perm[k])
        avg_diff = np.max(np.abs(m21_55_row - m21_avg))
        m21_sqrt = m21_sum / np.sqrt(len(perm[k]))
        sqrt_diff = np.max(np.abs(m21_55_row - m21_sqrt))

        print(f"  M21_55[{k},:]: best_single={best_diff:.6e}, sum={sum_diff:.6e}, avg={avg_diff:.6e}, sqrt={sqrt_diff:.6e} (m={len(perm[k])})")


# ============================================================
# Step 8: Test Schur complement with the 55-dim blocks
# ============================================================
print(f"\n=== Schur complement with 55-dim blocks ===")
def schur_55(omega):
    M_eff = M11_55.copy()
    for k in range(dd_55):
        M_eff += np.outer(M12_55[:, k], M21_55[k, :]) / (omega - D2_55_diag[k])
    return M_eff

# Iterate omega
omega = 0.0
for it in range(200):
    M_eff = schur_55(omega)
    if np.allclose(M_eff, M_eff.T, atol=1e-12):
        evals = np.linalg.eigvalsh(M_eff)
    else:
        evals = np.sort(np.linalg.eig(M_eff)[0].real)
    omega_new = evals[0]
    if abs(omega_new - omega) < 1e-10:
        break
    omega = omega_new

print(f"  Converged in {it+1} iterations")
print(f"  Eigenvalues[:5]: {evals[:5]}")
print(f"  PySCF ADC(2):    {e_adc2}")
diffs = evals[:5] - e_adc2[:5]
print(f"  Differences:     {diffs}")
print(f"  Max abs diff:    {np.max(np.abs(diffs)):.6e}")
print(f"  MATCH: {np.allclose(evals[:5], e_adc2[:5], atol=1e-6)}")

# Symmetry check
M_eff_test = schur_55(e_adc2[0])
print(f"\n  M_eff symmetric: {np.allclose(M_eff_test, M_eff_test.T, atol=1e-12)}")
print(f"  M_eff max asym: {np.max(np.abs(M_eff_test - M_eff_test.T)):.6e}")


# ============================================================
# Step 9: Check if M12_55 * D2inv * M21_55 is symmetric
# ============================================================
print(f"\n=== Symmetry of Schur coupling ===")
# At omega = 0
coupling = np.zeros((ov, ov))
for k in range(dd_55):
    coupling += np.outer(M12_55[:, k], M21_55[k, :]) / (-D2_55_diag[k])
print(f"M12_55 * D2inv * M21_55 symmetric: {np.allclose(coupling, coupling.T, atol=1e-12)}")
print(f"Max asym: {np.max(np.abs(coupling - coupling.T)):.6e}")


# ============================================================
# Step 10: Summary - what transformation gives M12_55 from M12_110?
# ============================================================
print(f"\n=== Summary: Transformation from 110-dim to 55-dim ===")
print(f"  The 55-dim representation uses {dd_55} packed doubles")
print(f"  Expected: nocc*(nocc+1)/2 * nvir*(nvir+1)/2 = {n_packed}")

# Check if we can construct a transformation matrix T (dd_full x dd_55)
# such that M12_55 = M12_110 @ T and M21_55 = T^T @ M21_110
# or M21_55 = some_other_transform

# Build T from the D2 mapping
print(f"\nTrying M12_55 = M12_110 @ T where T maps packed to full...")
# For each packed index k, T[:,k] has 1's at the full indices in perm[k]
T = np.zeros((dd_full, dd_55))
for k in range(dd_55):
    for j in perm[k]:
        T[j, k] = 1.0

M12_test = M12_110 @ T
print(f"M12_110 @ T vs M12_55: max diff = {np.max(np.abs(M12_test - M12_55)):.6e}")

# Try with normalization
for norm_name, norm_fn in [("1/m", lambda m: 1.0/m), ("1/sqrt(m)", lambda m: 1.0/np.sqrt(m))]:
    T_norm = np.zeros((dd_full, dd_55))
    for k in range(dd_55):
        m = len(perm[k])
        for j in perm[k]:
            T_norm[j, k] = norm_fn(m)
    M12_norm = M12_110 @ T_norm
    print(f"M12_110 @ T({norm_name}) vs M12_55: max diff = {np.max(np.abs(M12_norm - M12_55)):.6e}")

# Try T^T for M21
M21_test = T.T @ M21_110
print(f"\nT^T @ M21_110 vs M21_55: max diff = {np.max(np.abs(M21_test - M21_55)):.6e}")

for norm_name, norm_fn in [("1/m", lambda m: 1.0/m), ("1/sqrt(m)", lambda m: 1.0/np.sqrt(m))]:
    T_norm = np.zeros((dd_full, dd_55))
    for k in range(dd_55):
        m = len(perm[k])
        for j in perm[k]:
            T_norm[j, k] = norm_fn(m)
    M21_norm = T_norm.T @ M21_110
    print(f"T({norm_name})^T @ M21_110 vs M21_55: max diff = {np.max(np.abs(M21_norm - M21_55)):.6e}")
