"""Dump D2_eff (Coulomb + Exchange) from PySCF and verify 2c gradient."""
import numpy as np
from pyscf import gto, scf, df
from pyscf.gto.basis import parse_nwchem

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
aux_dict = {'O': parse_nwchem.parse(raw, 'O'), 'H': parse_nwchem.parse(raw, 'H')}

mol = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
            basis='sto-3g', unit='Angstrom')
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao, naux, nocc = mol.nao, auxmol.nao, mol.nelectron // 2

mf = scf.RHF(mol).density_fit(auxbasis=aux_dict)
mf.kernel()
D = mf.make_rdm1()
C = mf.mo_coeff[:, :nocc]

int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)
L = np.linalg.cholesky(int2c)
B = np.linalg.inv(L) @ int3c.T
B3d = B.reshape(naux, nao, nao)
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)

# Coulomb D2
D2_J = np.outer(d_bar, d_bar)

# Exchange D2: T_oo[P,i,j] = Σ_{μν} B[P,μ,ν] C[ν,i] C[μ,j]
T_half = np.einsum('Pmn,ni->Pmi', B3d, C)  # (naux, nao, nocc)
T_oo = np.einsum('Pmi,mj->Pij', T_half, C)  # (naux, nocc, nocc)

# D2_K[P,Q] = Σ_{ij} T_oo[P,i,j] T_oo[Q,j,i]
D2_K = np.einsum('Pij,Qji->PQ', T_oo, T_oo)

D2_eff = D2_J - D2_K  # Coulomb - Exchange

print(f"D2_J norm = {np.linalg.norm(D2_J):.6f}")
print(f"D2_K norm = {np.linalg.norm(D2_K):.6f}")
print(f"D2_eff norm = {np.linalg.norm(D2_eff):.6f}")
print(f"D2_J[0,0] = {D2_J[0,0]:.6e}")
print(f"D2_K[0,0] = {D2_K[0,0]:.6e}")
print(f"D2_eff[0,0] = {D2_eff[0,0]:.6e}")

# 2c gradient with D2_eff
int2c_ip1 = auxmol.intor('int2c2e_ip1')
auxslices = auxmol.aoslice_by_atom()

grad_2c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_2c[iatm, xyz] += np.einsum('PQ,PQ->', D2_eff[q0:q1, :], int2c_ip1[xyz, q0:q1, :])
        grad_2c[iatm, xyz] += np.einsum('PQ,QP->', D2_eff[:, q0:q1], int2c_ip1[xyz, q0:q1, :])

print(f"\n2c gradient (O dz):")
print(f"  J only:     {np.einsum('PQ,PQ->', D2_J[:56,:], int2c_ip1[2,:56,:]) + np.einsum('PQ,QP->', D2_J[:,:56], int2c_ip1[2,:56,:]):.6e} (O aux 0:56)")

print(f"  Full D2:    {grad_2c[0,2]:.6e}")
print(f"  J part:     {D2_J.sum():.6e}")
print(f"  K part:     {D2_K.sum():.6e}")

# Also: what is the correct total 2c?
# 3c without 2c gave residual -0.033 in GANSU.
# PySCF aux part was -0.0016.
# So correct 2c ≈ +0.03 (for this aux basis, different naux)

# From PySCF get_jk
g_ref = mf.nuc_grad_method().kernel()
print(f"\nFull gradient (O dz) = {g_ref[0,2]:.6e}")
