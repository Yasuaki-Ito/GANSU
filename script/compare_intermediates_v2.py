"""
Compare RI-HF gradient intermediates — load GANSU's exact aux basis.
"""
import numpy as np
from pyscf import gto, scf, df

mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

# Load GANSU's exact aux basis (NWChem format from BSE)
from pyscf.gto.basis import parse_nwchem
with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
auxbasis_o = parse_nwchem.parse(raw, 'O')
auxbasis_h = parse_nwchem.parse(raw, 'H')
aux_dict = {'O': auxbasis_o, 'H': auxbasis_h}

auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao = mol.nao
naux = auxmol.nao
print(f"nao={nao}, naux={naux}")

if naux != 96:
    print(f"WARNING: naux={naux}, expected 96")

# DF-RHF
mf = scf.RHF(mol).density_fit(auxbasis=aux_dict)
mf.kernel()
D = mf.make_rdm1()
print(f"E = {mf.e_tot:.10f}")

# Integrals
int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)

# B and w
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)
B = L_inv @ int3c.T
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)

print(f"\nw[0:5] = {w[:5]}")
print(f"d_bar[0:5] = {d_bar[:5]}")
print(f"w norm = {np.linalg.norm(w):.6f}")
print(f"d_bar norm = {np.linalg.norm(d_bar):.6f}")

# Derivative integrals
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)
int2c_ip1 = auxmol.intor('int2c2e_ip1')

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

# === Coulomb 3c ===
grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar, D[p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,vuP->', d_bar, D[:, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar[q0:q1], D, int3c_ip2[xyz, :, :, q0:q1])

# === Coulomb 2c ===
grad_2c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_2c[iatm, xyz] += np.einsum('P,PQ,Q->', d_bar[q0:q1], int2c_ip1[xyz, q0:q1, :], d_bar)
        grad_2c[iatm, xyz] += np.einsum('P,QP,Q->', d_bar, int2c_ip1[xyz, q0:q1, :], d_bar[q0:q1])

g_ref = mf.nuc_grad_method().kernel()

print(f"\n=== Coulomb gradient (all atoms, dz) ===")
print(f"{'Atom':4s} {'3c':>12s} {'2c':>12s} {'Coul total':>12s} {'Full ref':>12s}")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {grad_3c[i,2]:12.6e}  {grad_2c[i,2]:12.6e}  {grad_3c[i,2]+grad_2c[i,2]:12.6e}  {g_ref[i,2]:12.6e}")
