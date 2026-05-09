"""
Verify RI-HF gradient formula step by step using PySCF.

Build every intermediate and check against PySCF's analytical gradient.
This tells us EXACTLY what formula to use.
"""
import numpy as np
from pyscf import gto, scf, df, grad

# H2O same geometry
mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)
auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-jkfit')

mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.kernel()

D = mf.make_rdm1()
nao = mol.nao
naux = auxmol.nao
print(f"nao={nao}, naux={naux}, nocc={mol.nelectron//2}")

# === Integrals ===
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e')  # (nao, nao, naux)
int2c = auxmol.intor('int2c2e')  # (naux, naux)

# Derivatives
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1')  # (3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2')  # (3, nao, nao, naux) deriv on aux center
int2c_ip1 = auxmol.intor('int2c2e_ip1')  # (3, naux, naux)

# === Build B ===
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)
V3 = int3c.reshape(nao*nao, naux)  # (nao², naux)
B = (L_inv @ V3.T)  # (naux, nao²)

# === Intermediates ===
D_flat = D.flatten()
w = B @ D_flat  # w_P = Σ D B

# d_bar = (P|Q)^{-1} d  where d = V3^T @ D_flat
d_vec = V3.T @ D_flat
d_bar = np.linalg.solve(int2c, d_vec)

print(f"w norm: {np.linalg.norm(w):.6f}")
print(f"d_bar norm: {np.linalg.norm(d_bar):.6f}")

# === PySCF gradient for reference ===
g_obj = mf.nuc_grad_method()
g_anal = g_obj.kernel()
print(f"\nPySCF analytical gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_anal[i,0]:16.10e}  {g_anal[i,1]:16.10e}  {g_anal[i,2]:16.10e}")

# === Now compute each gradient term manually ===

# Get MO coefficients and orbital energies
C = mf.mo_coeff
e = mf.mo_energy
nocc = mol.nelectron // 2

# W matrix: W_μν = 2 Σ_i ε_i C_μi C_νi
W = 2.0 * C[:, :nocc] @ np.diag(e[:nocc]) @ C[:, :nocc].T

# 1-electron gradient (kinetic + nuclear + overlap)
h1_grad = grad.rhf.grad_nuc(mol)  # nuclear repulsion gradient
print(f"\nNuclear repulsion gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {h1_grad[i,0]:16.10e}  {h1_grad[i,1]:16.10e}  {h1_grad[i,2]:16.10e}")

# === 2-electron RI gradient: compute step by step ===
#
# dE_2e/dR_A = Σ_Q d_bar_Q Σ_{μν} D_{μν} d(μν|Q)/dR_A
#            - 0.5 Σ_{PQ} d_bar_P d(P|Q)/dR_A d_bar_Q
#            - 0.5 Σ_P Σ_{μν} Γ^K_{P,μν} d(μν|P)/dR_A  (exchange 3-center)
#            + 0.25 Σ_{PQ} γ^K_{PQ} d(P|Q)/dR_A          (exchange 2-center)
#
# For simplicity, let's compute the J contribution first.

# Coulomb 3-center: Σ_Q d_bar_Q Σ_{μν} D_{μν} d(μν|Q)/dR_A
# int3c_ip1 shape: (3, nao, nao, naux) — derivative on FIRST center of (μν|P)
# int3c_ip2 shape: (3, nao, nao, naux) — derivative on THIRD center (auxiliary)

# For atom A, we need derivatives of (μν|P) where μ is on A, ν is on A, or P is on A.
# d(μν|P)/dR_A = d(μν|P)/dR_μ if μ on A, + d(μν|P)/dR_ν if ν on A, + d(μν|P)/dR_P if P on A

# PySCF conventions:
# int3c2e_ip1 gives d/dR_1 where R_1 is the center of the FIRST basis function
# By translational invariance: d/dR_1 + d/dR_2 + d/dR_3 = 0

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

grad_J_3c = np.zeros((mol.natm, 3))
grad_J_2c = np.zeros((mol.natm, 3))
grad_K_3c = np.zeros((mol.natm, 3))
grad_K_2c = np.zeros((mol.natm, 3))

# === Coulomb 3-center ===
# Σ_Q d_bar_Q D_{μν} d(μν|Q)/dR_A
# ip1: d/dR_μ, ip2: d/dR_P, by transl. inv: d/dR_ν = -d/dR_μ - d/dR_P
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]

    # d/dR_μ contribution (μ on atom A)
    # int3c_ip1[:, μ, ν, Q] for μ in [p0,p1)
    for xyz in range(3):
        val = np.einsum('Q,uv,uvQ->', d_bar, D, int3c_ip1[xyz])
        # Only μ on this atom
        val_atom = np.einsum('Q,uv,uvQ->', d_bar, D[p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
        grad_J_3c[iatm, xyz] += val_atom

        # d/dR_ν contribution (ν on atom A, using symmetry)
        val_atom2 = np.einsum('Q,uv,uvQ->', d_bar, D[:, p0:p1], int3c_ip1[xyz, :, p0:p1, :])
        # But wait: int3c_ip1 is d/dR_1 (first center), not d/dR_2.
        # For d/dR_ν we need a different integral...
        # Actually: (μν|P) is symmetric in μ,ν: (μν|P) = (νμ|P)
        # So d(μν|P)/dR_ν = d(νμ|P)/dR_ν = int3c_ip1[xyz, ν, μ, P] evaluated at atom of ν
        val_atom2 = np.einsum('Q,uv,vuQ->', d_bar, D[:, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_J_3c[iatm, xyz] += val_atom2

# Auxiliary center contribution for 3-center: d/dR_P = -d/dR_μ - d/dR_ν (translational invariance)
# Or use int3c_ip2 directly
for iatm in range(mol.natm):
    # Find aux functions on this atom
    # auxmol atoms include ghost atoms for auxiliary basis
    # Actually auxmol.aoslice_by_atom() gives slices for the ORIGINAL atoms
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        val = np.einsum('Q,uv,uvQ->', d_bar[q0:q1], D, int3c_ip2[xyz, :, :, q0:q1])
        grad_J_3c[iatm, xyz] += val

print(f"\nCoulomb 3-center gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {grad_J_3c[i,0]:16.10e}  {grad_J_3c[i,1]:16.10e}  {grad_J_3c[i,2]:16.10e}")

# === Coulomb 2-center ===
# -0.5 Σ_{PQ} d_bar_P d(P|Q)/dR_A d_bar_Q
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # d(P|Q)/dR_A: derivative when P is on atom A
        val = -0.5 * np.einsum('P,PQ,Q->', d_bar[q0:q1], int2c_ip1[xyz, q0:q1, :], d_bar)
        # Also when Q is on atom A (by symmetry of (P|Q))
        val += -0.5 * np.einsum('P,QP,Q->', d_bar, int2c_ip1[xyz, q0:q1, :], d_bar[q0:q1])
        grad_J_2c[iatm, xyz] += val

print(f"\nCoulomb 2-center gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {grad_J_2c[i,0]:16.10e}  {grad_J_2c[i,1]:16.10e}  {grad_J_2c[i,2]:16.10e}")

print(f"\nTotal Coulomb gradient (3c + 2c):")
g_J = grad_J_3c + grad_J_2c
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_J[i,0]:16.10e}  {g_J[i,1]:16.10e}  {g_J[i,2]:16.10e}")

# === Total 2e = J - 0.5*K ===
# For now just print J to see if it's reasonable
print(f"\nFor reference, PySCF total gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_anal[i,0]:16.10e}  {g_anal[i,1]:16.10e}  {g_anal[i,2]:16.10e}")
