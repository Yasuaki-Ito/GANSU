"""
Trace PySCF's DF-RHF gradient to get the exact 2e breakdown.
Use PySCF's own get_jk to get vj and vk, then compute per-atom contributions.
"""
import numpy as np
from pyscf import gto, scf, df
from pyscf.gto.basis import parse_nwchem
from pyscf.df.grad import rhf as df_rhf_grad

with open('/tmp/cc-pvdz-rifit.nwchem') as f:
    raw = f.read()
aux_dict = {'O': parse_nwchem.parse(raw, 'O'), 'H': parse_nwchem.parse(raw, 'H')}

mol = gto.M(atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
            basis='sto-3g', unit='Angstrom')

mf = scf.RHF(mol).density_fit(auxbasis=aux_dict)
mf.kernel()
D = mf.make_rdm1()

# Get the gradient object
g_obj = mf.nuc_grad_method()
g_obj.auxbasis_response = True  # include aux basis response (2c terms)

# Call get_jk to get vj, vk with aux response
vj, vk = df_rhf_grad.get_jk(g_obj)

# vj and vk have shape (3, nao, nao) + .aux attribute (natm, 3)
print(f"vj shape: {vj.shape}")
print(f"vk shape: {vk.shape}")
print(f"vj.aux shape: {vj.aux.shape}")
print(f"vk.aux shape: {vk.aux.shape}")

# 2e gradient = Tr[D × (vj - 0.5*vk)] per atom
nao = mol.nao
aoslices = mol.aoslice_by_atom()

# vj and vk have shape (3, nao, nao) — these are the AO part (d/dR_μ contributions)
# The gradient is: g_2e[A,x] = Σ_{μ∈A,ν} D_μν (vj - 0.5*vk)[x,μ,ν]
#                            + Σ_{μ,ν∈A} D_μν (vj - 0.5*vk)[x,ν,μ]  (by symmetry)
#                            + vj.aux[A,x] - 0.5*vk.aux[A,x]  (aux center response)

# Actually, PySCF grad code (grad_elec function) does:
# g[A,x] = Tr[D @ (vj[x] - 0.5*vk[x])] restricted to μ on atom A
# = Σ_{μ∈A} Σ_ν D[ν,μ] vhf[x,μ,ν]  ... let me check PySCF's grad_elec

# PySCF's grad_elec:
# for each atom A with AO slice p0:p1:
#   vhf = vj - 0.5*vk
#   de[A] += einsum('xij,ij->x', vhf[:,p0:p1], dm[:,p0:p1]) * 2
# (factor 2 for double counting of upper/lower triangle)

vhf = vj - 0.5 * vk  # (3, nao, nao)
g_2e_ao = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    g_2e_ao[iatm] = np.einsum('xij,ji->x', vhf[:, p0:p1], D[:, p0:p1]) * 2

# Aux response
g_2e_aux = np.zeros((mol.natm, 3))
auxmol = df.addons.make_auxmol(mol, auxbasis=aux_dict)
auxslices = auxmol.aoslice_by_atom()
vjaux = vj.aux[0, 0]  # (natm, 3)
vkaux = vk.aux[0, 0]  # (natm, 3)
print(f"vjaux shape: {vjaux.shape}")
print(f"vkaux shape: {vkaux.shape}")

g_2e_aux = vjaux - 0.5 * vkaux

g_2e = g_2e_ao + g_2e_aux
print(f"\n=== 2e gradient breakdown (O dz) ===")
print(f"  AO part (d/dR_μ):  {g_2e_ao[0,2]:.10e}")
print(f"  Aux part (d/dR_P): {g_2e_aux[0,2]:.10e}")
print(f"  Total 2e:          {g_2e[0,2]:.10e}")

# Also print J and K separately
g_J_ao = np.zeros((mol.natm, 3))
g_K_ao = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    g_J_ao[iatm] = np.einsum('xij,ji->x', vj[:, p0:p1], D[:, p0:p1]) * 2
    g_K_ao[iatm] = np.einsum('xij,ji->x', vk[:, p0:p1], D[:, p0:p1]) * 2

print(f"\n=== J/K breakdown (O dz) ===")
print(f"  J AO:  {g_J_ao[0,2]:.10e}")
print(f"  K AO:  {g_K_ao[0,2]:.10e}")
print(f"  J aux: {vjaux[0,2]:.10e}")
print(f"  K aux: {vkaux[0,2]:.10e}")
print(f"  2e = J_ao - 0.5*K_ao + J_aux - 0.5*K_aux")
print(f"     = {g_J_ao[0,2] - 0.5*g_K_ao[0,2] + vjaux[0,2] - 0.5*vkaux[0,2]:.10e}")

# Reference
g_ref = mf.nuc_grad_method().kernel()
from pyscf.grad import rhf as rhf_grad
g_nuc = rhf_grad.grad_nuc(mol)
C = mf.mo_coeff; e = mf.mo_energy; nocc = mol.nelectron//2
W = 2.0 * C[:,:nocc] @ np.diag(e[:nocc]) @ C[:,:nocc].T
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

print(f"\n=== Comparison (O dz) ===")
print(f"  My 2e:  {g_2e[0,2]:.10e}")
print(f"  Ref 2e: {g_2e_ref[0,2]:.10e}")
print(f"  Diff:   {g_2e[0,2]-g_2e_ref[0,2]:.10e}")
