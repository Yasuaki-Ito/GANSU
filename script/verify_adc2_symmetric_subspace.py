"""
Verify that ADC(2) eigenvalues are correct when restricted to the
symmetric doubles subspace where r2[I,J,A,B] = r2[J,I,B,A].

The full ADC(2) matrix (probed from PySCF) is non-symmetric in the
redundant o^2*v^2 doubles space. But in the physical subspace where
doubles satisfy the simultaneous swap symmetry r2[IJAB] = r2[JIBA],
the matrix IS Hermitian and gives correct eigenvalues.

Usage: python verify_adc2_symmetric_subspace.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee

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

# Run ADC(2) and get matvec
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

M_ab = radc_ee.get_imds(myadc)
eris = myadc.transform_integrals()
sigma_fn = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)

# Build the full matrix by probing
print(f"\nBuilding full {total}x{total} matrix...")
M_full = np.zeros((total, total))
for col in range(total):
    e_vec = np.zeros(total)
    e_vec[col] = 1.0
    M_full[:, col] = sigma_fn(e_vec)

print(f"Full matrix symmetric: {np.allclose(M_full, M_full.T)}")
print(f"Full matrix eigenvalues[:5]: {np.sort(np.linalg.eig(M_full)[0].real)[:5]}")

# ======================================================================
# Build the swap operator S: r2[I,J,A,B] -> r2[J,I,B,A]
# ======================================================================
# S acts as identity on r1, and swaps (I,J) and (A,B) simultaneously on r2
def swap_r2(r2_flat):
    """Apply swap: r2[I,J,A,B] -> r2[J,I,B,A]"""
    r2 = r2_flat.reshape(nocc, nocc, nvir, nvir)
    r2_swapped = r2.transpose(1, 0, 3, 2)  # [J,I,B,A]
    return r2_swapped.reshape(-1)

def apply_swap(v):
    """Apply swap operator to full vector [r1, r2]"""
    result = np.zeros_like(v)
    result[:ov] = v[:ov]  # r1 unchanged
    result[ov:] = swap_r2(v[ov:])
    return result

# Verify S^2 = I
test_v = np.random.randn(total)
print(f"\nS^2 = I check: {np.allclose(apply_swap(apply_swap(test_v)), test_v)}")

# Check if M commutes with S: M*S*v should equal S*M*v
# (This means the symmetric subspace is invariant under M)
print("Checking M*S = S*M (matrix commutes with swap)...")
max_comm = 0.0
for _ in range(10):
    v = np.random.randn(total)
    Ms_v = M_full @ apply_swap(v)
    sM_v = apply_swap(M_full @ v)
    max_comm = max(max_comm, np.max(np.abs(Ms_v - sM_v)))
print(f"Max |M*S*v - S*M*v|: {max_comm:.6e}")

# ======================================================================
# Build projector onto symmetric subspace
# ======================================================================
# P = (I + S) / 2
# Build the projection matrix explicitly
S_matrix = np.zeros((total, total))
for col in range(total):
    e_vec = np.zeros(total)
    e_vec[col] = 1.0
    S_matrix[:, col] = apply_swap(e_vec)

P = 0.5 * (np.eye(total) + S_matrix)
print(f"\nProjector P: rank = {np.linalg.matrix_rank(P, tol=1e-10)}")

# Symmetric subspace dimension
sym_dim = np.linalg.matrix_rank(P, tol=1e-10)
print(f"Symmetric subspace dim: {sym_dim} (ov={ov}, expected doubles={(nocc*(nocc+1)//2)*(nvir*(nvir+1)//2) + (nocc*(nocc-1)//2)*(nvir*(nvir-1)//2)})")

# ======================================================================
# Method 1: Project M onto symmetric subspace and diagonalize
# ======================================================================
# M_proj = P * M * P
M_proj = P @ M_full @ P
print(f"\nProjected matrix symmetric: {np.allclose(M_proj, M_proj.T)}")
print(f"Max asymmetry of projected: {np.max(np.abs(M_proj - M_proj.T)):.6e}")

# Get eigenvalues (non-zero ones)
evals_proj = np.linalg.eigvalsh(M_proj)
# Filter out near-zero eigenvalues (from null space of P)
nonzero_mask = np.abs(evals_proj) > 1e-10
evals_physical = evals_proj[nonzero_mask]
print(f"Projected eigenvalues (non-zero): {evals_physical[:10]}")

# ======================================================================
# Method 2: Build an orthonormal basis for the symmetric subspace
# and compute the matrix in that basis
# ======================================================================
# Use SVD of P to get the basis vectors
U, S_vals, Vt = np.linalg.svd(P)
# Columns of U with singular value ≈ 1 form the symmetric subspace basis
basis_idx = S_vals > 0.5
Q = U[:, basis_idx]  # orthonormal basis for symmetric subspace
print(f"\nBasis Q: shape={Q.shape}")

# M in symmetric subspace basis: M_sym = Q^T * M * Q
M_sym = Q.T @ M_full @ Q
print(f"M_sym symmetric: {np.allclose(M_sym, M_sym.T)}")
print(f"Max asymmetry: {np.max(np.abs(M_sym - M_sym.T)):.6e}")

evals_sym = np.linalg.eigvalsh(M_sym)
print(f"Symmetric subspace eigenvalues[:10]: {evals_sym[:10]}")

# ======================================================================
# Method 3: Schur complement in symmetric subspace
# ======================================================================
print("\n=== Schur complement in symmetric subspace ===")
# Build M11, M12_sym, M21_sym, D2 in the symmetric basis
# For this, we need the symmetric-doubles basis
Q1 = Q[:ov, :]      # singles part of basis vectors
Q2 = Q[ov:, :]      # doubles part of basis vectors

# Actually, the singles part is trivially the identity (no swap for r1)
# Let's separate the symmetric doubles basis
# First ov columns of Q should be [I_ov; 0_dd] (singles part)
print(f"Q1 (singles part): shape={Q1.shape}")
print(f"Q2 (doubles part): shape={Q2.shape}")

# ======================================================================
# Comparison
# ======================================================================
print("\n" + "=" * 70)
print(f"{'Method':>30}  {'Eigenvalue 1':>12}  {'Eigenvalue 2':>12}  {'Eigenvalue 3':>12}")
print("-" * 70)
full_evals = np.sort(np.linalg.eig(M_full)[0].real)
print(f"{'Full matrix (eig)':>30}  {full_evals[0]:12.8f}  {full_evals[1]:12.8f}  {full_evals[2]:12.8f}")
print(f"{'Sym subspace (eigvalsh)':>30}  {evals_sym[0]:12.8f}  {evals_sym[1]:12.8f}  {evals_sym[2]:12.8f}")
print(f"{'PySCF ADC(2)':>30}  {e_adc2[0]:12.8f}  {e_adc2[1]:12.8f}  {e_adc2[2]:12.8f}")

print(f"\n{'State':>5}  {'Sym subspace':>14}  {'PySCF':>14}  {'Diff':>14}")
print("-" * 55)
n_compare = min(5, len(evals_sym), len(e_adc2))
for i in range(n_compare):
    diff = evals_sym[i] - e_adc2[i]
    print(f"{i+1:5d}  {evals_sym[i]:14.8f}  {e_adc2[i]:14.8f}  {diff:14.6e}")

match = np.allclose(evals_sym[:n_compare], e_adc2[:n_compare], atol=1e-6)
print(f"\nSymmetric subspace matches PySCF: {match}")

if match:
    print("\n*** CONFIRMED: ADC(2) must operate in the symmetric doubles subspace ***")
    print("*** r2[I,J,A,B] = r2[J,I,B,A] constraint is essential ***")
