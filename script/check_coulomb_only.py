"""Check Coulomb-only RI gradient with cc-pvdz-ri (matching GANSU's aux basis)."""
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
B = L_inv @ int3c.T
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)

print(f"w norm = {np.linalg.norm(w):.6f}")
print(f"d_bar norm = {np.linalg.norm(d_bar):.6f}")

# 3c and 2c derivative integrals
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)
int2c_ip1 = auxmol.intor('int2c2e_ip1')

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

# Coulomb 3c: Σ d̄_P D_μν d(μν|P)/dR_A
# PySCF ip1 = -d/dR_1, ip2 = -d/dR_aux
grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # μ on atom: d̄_P D_μν (-ip1[μ,ν,P])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar, D[p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        # ν on atom: d̄_P D_μν d(μν|P)/dR_ν = d̄_P D_μν (-ip1[ν,μ,P])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,vuP->', d_bar, D[:, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        # aux P on atom
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar[q0:q1], D, int3c_ip2[xyz, :, :, q0:q1])

# Coulomb 2c: -Σ d̄_P d(P|Q)/dR_A d̄_Q
# int2c_ip1 = -d(P|Q)/dR_P
grad_2c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # P on atom: -d̄_P (-ip1[P,Q]) d̄_Q = +d̄_P ip1[P,Q] d̄_Q
        grad_2c[iatm, xyz] += np.einsum('P,PQ,Q->', d_bar[q0:q1], int2c_ip1[xyz, q0:q1, :], d_bar)
        # Q on atom: -d̄_P d(P|Q)/dR_Q d̄_Q = -d̄_P (-ip1[Q,P]) d̄_Q = +d̄_P ip1[Q,P] d̄_Q
        grad_2c[iatm, xyz] += np.einsum('P,QP,Q->', d_bar, int2c_ip1[xyz, q0:q1, :], d_bar[q0:q1])

print(f"\nCoulomb gradient (O dz):")
print(f"  3c = {grad_3c[0,2]:.6e}")
print(f"  2c = {grad_2c[0,2]:.6e}")
print(f"  total = {grad_3c[0,2]+grad_2c[0,2]:.6e}")

# Full gradient for comparison
g_ref = mf.nuc_grad_method().kernel()
print(f"\nFull RI-HF gradient (O dz) = {g_ref[0,2]:.6e}")

# 1e gradient
from pyscf.grad import rhf as rhf_grad
g_nuc = rhf_grad.grad_nuc(mol)
nocc = mol.nelectron // 2
C = mf.mo_coeff
e = mf.mo_energy
W = 2.0 * C[:, :nocc] @ np.diag(e[:nocc]) @ C[:, :nocc].T
h1 = -(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
s1 = -mol.intor('int1e_ipovlp', comp=3)
g_1e = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    for xyz in range(3):
        g_1e[iatm, xyz] = (np.einsum('uv,uv->', D[p0:p1], h1[xyz, p0:p1])
                          + np.einsum('uv,vu->', D[:, p0:p1], h1[xyz, p0:p1])
                          + np.einsum('uv,uv->', W[p0:p1], s1[xyz, p0:p1])
                          + np.einsum('uv,vu->', W[:, p0:p1], s1[xyz, p0:p1]))
    with mol.with_rinv_origin(mol.atom_coord(iatm)):
        vrinv = mol.intor('int1e_iprinv', comp=3)
    g_1e[iatm] -= np.einsum('xij,ij->x', vrinv, D) * mol.atom_charge(iatm)
g_1e += g_nuc

g_2e_ref = g_ref - g_1e
print(f"\n1e (O dz) = {g_1e[0,2]:.6e}")
print(f"2e ref (O dz) = {g_2e_ref[0,2]:.6e}")
print(f"Coulomb-only 2e (O dz) = {grad_3c[0,2]+grad_2c[0,2]:.6e}")
print(f"Exchange 2e (O dz) = {g_2e_ref[0,2] - (grad_3c[0,2]+grad_2c[0,2]):.6e}")
