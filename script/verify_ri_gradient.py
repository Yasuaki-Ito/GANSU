"""
Verify RI-HF gradient components using PySCF.

1. Compute 3-center integrals and their derivatives
2. Compute 2-center integrals and their derivatives
3. Build effective densities and contract with derivatives
4. Compare with GANSU's numerical gradient
"""
import numpy as np
from pyscf import gto, scf, df

# H2O with same geometry as GANSU (Angstrom)
mol = gto.M(
    atom='''
    O  0.000  0.000  0.127
    H  0.000  0.758 -0.509
    H  0.000 -0.758 -0.509
    ''',
    basis='sto-3g',
    unit='Angstrom'
)

auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-jkfit')

# Run RHF with density fitting
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.kernel()
print(f"RI-HF energy: {mf.e_tot:.10f}")

# Get density matrix
D = mf.make_rdm1()
print(f"D shape: {D.shape}")
print(f"D trace: {np.trace(D):.6f} (should be {mol.nelectron})")

# === 3-center integrals ===
# (μν|P) using PySCF
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
print(f"3-center shape: {int3c.shape}")  # (nao, nao, naux)

# 3-center derivatives: d(μν|P)/dR_A
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1')
print(f"3-center deriv shape: {int3c_ip1.shape}")  # (3, nao, nao, naux)

# === 2-center integrals ===
int2c = auxmol.intor('int2c2e')
print(f"2-center shape: {int2c.shape}")  # (naux, naux)

int2c_ip1 = auxmol.intor('int2c2e_ip1')
print(f"2-center deriv shape: {int2c_ip1.shape}")  # (3, naux, naux)

# === Build B matrix ===
# B = L^{-1} V^{(3)} where L L^T = (P|Q)
L = np.linalg.cholesky(int2c)  # Lower triangular
# B_{P,μν} = Σ_Q L^{-1}_{PQ} (μν|Q)
# int3c is (nao, nao, naux), reshape to (nao*nao, naux)
nao = mol.nao
naux = auxmol.nao
V3 = int3c.reshape(nao*nao, naux)  # (nao², naux)
B = np.linalg.solve(L, V3.T).T  # B: (nao², naux) ... no
# Actually: B_{P,μν} = Σ_Q (L^{-1})_{PQ} V3_{μν,Q}
# B(P, μν) = Σ_Q L_inv(P,Q) V3(μν, Q)
# B = (L^{-1} @ V3.T) → shape (naux, nao²)
L_inv = np.linalg.inv(L)
B_matrix = L_inv @ V3.T  # (naux, nao²)
print(f"B shape: {B_matrix.shape}")

# === w_P = Σ_{μν} D_{μν} B_{P,μν} ===
D_flat = D.flatten()
w = B_matrix @ D_flat  # (naux,)
print(f"w shape: {w.shape}, w norm: {np.linalg.norm(w):.6f}")

# === d_bar = (P|Q)^{-1} d where d_Q = Σ_{μν} D_{μν} (μν|Q) ===
d_vec = V3.T @ D_flat  # (naux,)
d_bar = np.linalg.solve(int2c, d_vec)  # (naux,)
print(f"d_bar norm: {np.linalg.norm(d_bar):.6f}")
print(f"w vs d_bar max diff: {np.max(np.abs(w - d_bar)):.2e}")
# w and d_bar should NOT be the same in general

# === Verify J energy ===
E_J = 0.5 * np.dot(w, w)  # = 0.5 Σ_P w_P^2 = 0.5 d^T A^{-1} d
E_J_alt = 0.5 * np.dot(d_vec, d_bar)
print(f"E_J (from w): {E_J:.10f}")
print(f"E_J (from d_bar): {E_J_alt:.10f}")

# === Analytical RI-HF gradient via PySCF ===
grad_obj = mf.nuc_grad_method()
g = grad_obj.kernel()
print("\nPySCF RI-HF gradient (Hartree/Bohr):")
for i, atom in enumerate(mol.atom):
    print(f"  {atom[0]:2s}  {g[i,0]:16.10e}  {g[i,1]:16.10e}  {g[i,2]:16.10e}")

# === Numerical gradient for comparison ===
print("\nNumerical gradient (central diff, delta=1e-5 Bohr):")
delta = 1e-5
coords = mol.atom_coords().copy()  # Bohr
num_grad = np.zeros_like(coords)
for i in range(len(coords)):
    for j in range(3):
        coords_p = coords.copy()
        coords_p[i, j] += delta
        mol_p = mol.copy()
        mol_p.set_geom_(coords_p, unit='Bohr')
        mol_p.build()
        mf_p = scf.RHF(mol_p).density_fit(auxbasis='cc-pvdz-jkfit')
        mf_p.kernel()

        coords_m = coords.copy()
        coords_m[i, j] -= delta
        mol_m = mol.copy()
        mol_m.set_geom_(coords_m, unit='Bohr')
        mol_m.build()
        mf_m = scf.RHF(mol_m).density_fit(auxbasis='cc-pvdz-jkfit')
        mf_m.kernel()

        num_grad[i, j] = (mf_p.e_tot - mf_m.e_tot) / (2 * delta)

for i, atom in enumerate(mol.atom):
    print(f"  {atom[0]:2s}  {num_grad[i,0]:16.10e}  {num_grad[i,1]:16.10e}  {num_grad[i,2]:16.10e}")
