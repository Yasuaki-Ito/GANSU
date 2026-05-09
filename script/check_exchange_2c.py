"""Compute Exchange 2c contribution to verify formula."""
import numpy as np
from pyscf import gto, scf, df
from pyscf.gto.basis import parse_nwchem

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
aux_dict = {'O': parse_nwchem.parse(raw, 'O'), 'H': parse_nwchem.parse(raw, 'H')}

mol = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
            basis='sto-3g', unit='Angstrom')
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
nao, naux = mol.nao, auxmol.nao
nocc = mol.nelectron // 2

mf = scf.RHF(mol).density_fit(auxbasis=aux_dict)
mf.kernel()
D = mf.make_rdm1()
C = mf.mo_coeff
C_occ = C[:, :nocc]

int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)
L = np.linalg.cholesky(int2c)
B = np.linalg.inv(L) @ int3c.T  # (naux, nao²)
B3d = B.reshape(naux, nao, nao)
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)

# Exchange 2c: need rhok_oo[P,i,j] = Σ_μν B_Pμν C_μi C_νj ... no
# PySCF: rhok_oo[P,i,j] = Σ_μ (Σ_ν B_Pμν C_νi * sqrt(n_i)) × C_μj
# For RHF, n_i = 2 for occupied.
# Actually PySCF uses orbol = C_occ * sqrt(2) and orbor = C_occ
# rhok[P,μ,i] = Σ_ν B_Pμν C_νi × sqrt(2)
# rhok_oo[P,i,j] = Σ_μ rhok[P,μ,i] × C_μj

# Let me compute directly:
# T[P,μ,i] = Σ_ν B[P,μ,ν] C_occ[ν,i]  (half-transform)
T = np.einsum('Pmn,ni->Pmi', B3d, C_occ)  # (naux, nao, nocc)

# rhok_oo[P,i,j] = Σ_μ T[P,μ,i] C_occ[μ,j]
rhok_oo = np.einsum('Pmi,mj->Pij', T, C_occ)  # (naux, nocc, nocc)

# Exchange 2c density: d2k[P,Q] = Σ_{ij} rhok_oo[P,i,j] rhok_oo[Q,j,i]
d2k = np.einsum('Pij,Qji->PQ', rhok_oo, rhok_oo)  # (naux, naux)

# But PySCF has factor of 2 from orbol = C*sqrt(2):
# d2k_pyscf = 4 * d2k (2 from each orbol)
# Actually let me check PySCF's exact convention...
# PySCF: orbol = C_occ * sqrt(n) where n = occupation = 2 for RHF
# So rhok has sqrt(2) factor in each orbol → rhok_oo has 2 factor (sqrt(2)^2)
# d2k_pyscf = 2^2 * d2k_mine = 4 * d2k? No...
# PySCF: orbol[i] = C_occ * sqrt(2), orbor[i] = C_occ
# rhok[P,μ,i] = Σ_ν B[P,μ,ν] (C_occ[ν,i] * sqrt(2))
# rhok_oo[P,i,j] = Σ_μ rhok[P,μ,i] C_occ[μ,j]  (orbor = C_occ without sqrt(2))
# = sqrt(2) Σ_μν B[P,μ,ν] C[ν,i] C[μ,j]
# d2k[P,Q] = Σ_ij rhok_oo[P,i,j] rhok_oo[Q,j,i]
#   = 2 Σ_ij (Σ B C C)_P (Σ B C C)_Q
# So factor of 2.

# For the 2c contribution:
# PySCF: vkaux -= einsum('xpq,pq->xp', int2c_ip1, d2k_pyscf)
# = vkaux -= 2 einsum('xpq,pq->xp', int2c_ip1, my_d2k)
# Then vjaux = -vjaux → contribution = +2 d2k × ip1 per atom sum

# Let me just compute it:
int2c_ip1 = auxmol.intor('int2c2e_ip1')  # (3, naux, naux) = -d(P|Q)/dR_P
auxslices = auxmol.aoslice_by_atom()

# PySCF Exchange 2c:
grad_2c_K = np.zeros((mol.natm, 3))
d2k_full = 2.0 * d2k  # factor of 2 from sqrt(2) in orbol
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # PySCF: vkaux -= einsum('pq,xpq->xp', d2k, ip1) per P-slice, then -vkaux → gradient
        # Per atom: gradient = +Σ_{P on atom} Σ_Q d2k[P,Q] ip1[x,P,Q]
        grad_2c_K[iatm, xyz] += np.einsum('PQ,PQ->', d2k_full[q0:q1, :], int2c_ip1[xyz, q0:q1, :])
        # Also Q on atom (by symmetry of int2c):
        # d(P|Q)/dR_Q = d(Q|P)/dR_Q → ip1[Q,P]
        grad_2c_K[iatm, xyz] += np.einsum('PQ,QP->', d2k_full[:, q0:q1], int2c_ip1[xyz, q0:q1, :])

# Also Coulomb 2c for comparison
grad_2c_J = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_2c_J[iatm, xyz] += np.einsum('P,PQ,Q->', d_bar[q0:q1], int2c_ip1[xyz, q0:q1, :], d_bar)
        grad_2c_J[iatm, xyz] += np.einsum('P,QP,Q->', d_bar, int2c_ip1[xyz, q0:q1, :], d_bar[q0:q1])

print(f"=== 2c gradient (O dz) ===")
print(f"  Coulomb:   {grad_2c_J[0,2]:.6e}")
print(f"  Exchange:  {grad_2c_K[0,2]:.6e}")
print(f"  Total 2c:  {grad_2c_J[0,2]+grad_2c_K[0,2]:.6e}")

# Full reference
g_ref = mf.nuc_grad_method().kernel()
# 1e
from pyscf.grad import rhf as rhf_grad
g_nuc = rhf_grad.grad_nuc(mol)
h1 = -(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
s1 = -mol.intor('int1e_ipovlp', comp=3)
e = mf.mo_energy
W = 2.0 * C[:, :nocc] @ np.diag(e[:nocc]) @ C[:, :nocc].T
aoslices = mol.aoslice_by_atom()
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

# 3c from earlier
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)
D3 = np.outer(w, D.flatten()) - 0.5 * np.array([D @ B3d[P] @ D for P in range(naux)]).reshape(naux, nao*nao)
Z = np.linalg.solve(L.T, D3)
Z3d = Z.reshape(naux, nao, nao)
grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_3c[iatm, xyz] -= np.einsum('Quv,uvQ->', Z3d[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('Quv,vuQ->', Z3d[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('Quv,uvQ->', Z3d[q0:q1, :, :], int3c_ip2[xyz, :, :, q0:q1])

print(f"\n=== Full breakdown (O dz) ===")
print(f"  1e:        {g_1e[0,2]:.6e}")
print(f"  3c:        {grad_3c[0,2]:.6e}")
print(f"  2c J:      {grad_2c_J[0,2]:.6e}")
print(f"  2c K:      {grad_2c_K[0,2]:.6e}")
g_my = g_1e[0,2] + grad_3c[0,2] + grad_2c_J[0,2] + grad_2c_K[0,2]
print(f"  My total:  {g_my:.6e}")
print(f"  Ref total: {g_ref[0,2]:.6e}")
print(f"  Diff:      {g_my - g_ref[0,2]:.6e}")
