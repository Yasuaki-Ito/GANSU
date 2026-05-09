"""Quick check: compute d_bar with cc-pvdz-rifit to match GANSU."""
import numpy as np
from pyscf import gto, scf, df

mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()
D = mf.make_rdm1()
nao = mol.nao
auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-ri')
naux = auxmol.nao
print(f"nao={nao}, naux={naux}")

int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)
B = L_inv @ int3c.T  # (naux, nao²)
w = B @ D.flatten()

d_bar = np.linalg.solve(L.T, w)  # d̄ = L^{-T} w

print(f"w[0:5] = {w[:5]}")
print(f"d_bar[0:5] = {d_bar[:5]}")
print(f"w norm = {np.linalg.norm(w):.6f}")
print(f"d_bar norm = {np.linalg.norm(d_bar):.6f}")

# Also check: d̄ via A^{-1} d where d = L^T w
d_vec = L.T @ w  # d = L^T w (= V3^T D)
d_bar2 = np.linalg.solve(int2c, d_vec)
print(f"\nd_bar2[0:5] = {d_bar2[:5]} (via A^{{-1}} d)")
print(f"d_bar2 norm = {np.linalg.norm(d_bar2):.6f}")
print(f"d_bar == d_bar2? max diff = {np.max(np.abs(d_bar - d_bar2)):.2e}")
