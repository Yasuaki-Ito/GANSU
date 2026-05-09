"""
RI-HF gradient: complete breakdown of all terms.
Compare each term with GANSU to find the discrepancy.
"""
import numpy as np
from pyscf import gto, scf, df, grad
from pyscf.grad import rhf as rhf_grad

mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

# Standard (non-DF) RHF for 1-electron gradient comparison
mf_std = scf.RHF(mol)
mf_std.kernel()
D_std = mf_std.make_rdm1()
C_std = mf_std.mo_coeff
e_std = mf_std.mo_energy
nocc = mol.nelectron // 2
nao = mol.nao

print(f"Standard RHF energy: {mf_std.e_tot:.10f}")
print(f"nao={nao}, nocc={nocc}")

# W matrix
W_std = 2.0 * C_std[:, :nocc] @ np.diag(e_std[:nocc]) @ C_std[:, :nocc].T

# === 1-electron gradient terms (these are the SAME for Stored and RI) ===
# Nuclear repulsion gradient
g_nuc = rhf_grad.grad_nuc(mol)

# Kinetic energy derivative: Σ D_μν dT_μν/dR
# Nuclear attraction derivative: Σ D_μν dV_μν/dR
# Overlap derivative: -Σ W_μν dS_μν/dR

aoslices = mol.aoslice_by_atom()

# Get derivative integrals
s1 = -mol.intor('int1e_ipovlp', comp=3)  # (3, nao, nao) — NOTE: PySCF returns -dS/dR_1
t1 = -mol.intor('int1e_ipkin', comp=3)   # (3, nao, nao) — -dT/dR_1
v1 = -mol.intor('int1e_ipnuc', comp=3)   # (3, nao, nao) — -dV/dR_1

# h1 = t1 + v1 (core Hamiltonian derivative)
h1 = t1 + v1

# Per-atom 1-electron gradient
g_kinetic = np.zeros((mol.natm, 3))
g_nuclear_attr = np.zeros((mol.natm, 3))
g_overlap = np.zeros((mol.natm, 3))
g_hcore = np.zeros((mol.natm, 3))

for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    for xyz in range(3):
        # PySCF ip integrals: -d<μ|O|ν>/dR_1 where R_1 is center of μ.
        # Contribution when μ on atom A: Σ_{μ∈A,ν} D_μν (-ip[μ,ν])
        # Contribution when ν on atom A: by symmetry of real integrals, ip[μ,ν] = ip[ν,μ]^T
        #   so: Σ_{μ,ν∈A} D_μν (-ip[ν,μ]) = Σ_{μ,ν∈A} D_μν (-ip[ν,μ])
        g_hcore[iatm, xyz] = (np.einsum('uv,uv->', D_std[p0:p1], h1[xyz, p0:p1])
                             + np.einsum('uv,vu->', D_std[:, p0:p1], h1[xyz, p0:p1]))

        g_overlap[iatm, xyz] = (np.einsum('uv,uv->', W_std[p0:p1], s1[xyz, p0:p1])
                               + np.einsum('uv,vu->', W_std[:, p0:p1], s1[xyz, p0:p1]))

    # Nuclear attraction: derivative on nuclear center R_C
    with mol.with_rinv_origin(mol.atom_coord(iatm)):
        vrinv = mol.intor('int1e_iprinv', comp=3)
    g_hcore[iatm] -= np.einsum('xij,ij->x', vrinv, D_std) * mol.atom_charge(iatm)

print("\n=== Per-term gradient (O atom, dz component) ===")
print(f"Nuclear repulsion:    {g_nuc[0,2]:16.10e}")
print(f"Core Hamiltonian:     {g_hcore[0,2]:16.10e}")
print(f"Overlap (W):          {g_overlap[0,2]:16.10e}")
print(f"1e total (hcore+ovlp+nuc): {(g_hcore[0,2]+g_overlap[0,2]+g_nuc[0,2]):16.10e}")

# === Now compute 2-electron (4-center) gradient for Stored HF ===
g_std = mf_std.nuc_grad_method().kernel()
g_2e_stored = g_std - g_nuc - g_hcore - g_overlap
print(f"\n2-electron (Stored):  {g_2e_stored[0,2]:16.10e}")
print(f"Total (Stored):       {g_std[0,2]:16.10e}")

# === RI-HF for comparison ===
mf_ri = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf_ri.kernel()
g_ri = mf_ri.nuc_grad_method().kernel()
D_ri = mf_ri.make_rdm1()
C_ri = mf_ri.mo_coeff
e_ri = mf_ri.mo_energy
W_ri = 2.0 * C_ri[:, :nocc] @ np.diag(e_ri[:nocc]) @ C_ri[:, :nocc].T

# 1e terms with RI density/W
g_hcore_ri = np.zeros((mol.natm, 3))
g_overlap_ri = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    for xyz in range(3):
        g_hcore_ri[iatm, xyz] = (np.einsum('ij,ij->', D_ri[p0:p1, :], h1[xyz, p0:p1, :])
                                + np.einsum('ij,ji->', D_ri[:, p0:p1], h1[xyz, p0:p1, :]))
        g_overlap_ri[iatm, xyz] = (np.einsum('ij,ij->', W_ri[p0:p1, :], s1[xyz, p0:p1, :])
                                  + np.einsum('ij,ji->', W_ri[:, p0:p1], s1[xyz, p0:p1, :]))
    with mol.with_rinv_origin(mol.atom_coord(iatm)):
        vrinv = mol.intor('int1e_iprinv', comp=3)
    g_hcore_ri[iatm] -= np.einsum('xij,ij->x', vrinv, D_ri) * mol.atom_charge(iatm)

g_1e_ri = g_hcore_ri + g_overlap_ri + g_nuc
g_2e_ri = g_ri - g_1e_ri

print(f"\n=== RI-HF breakdown (O atom, dz) ===")
print(f"Nuclear repulsion:    {g_nuc[0,2]:16.10e}")
print(f"Core Hamiltonian:     {g_hcore_ri[0,2]:16.10e}")
print(f"Overlap (W):          {g_overlap_ri[0,2]:16.10e}")
print(f"1e total:             {g_1e_ri[0,2]:16.10e}")
print(f"2e (RI):              {g_2e_ri[0,2]:16.10e}")
print(f"Total (RI):           {g_ri[0,2]:16.10e}")

print(f"\n=== Full gradient (all atoms) ===")
print("RI-HF:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_ri[i,0]:16.10e}  {g_ri[i,1]:16.10e}  {g_ri[i,2]:16.10e}")
print("  1e only:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_1e_ri[i,0]:16.10e}  {g_1e_ri[i,1]:16.10e}  {g_1e_ri[i,2]:16.10e}")
print("  2e only:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_2e_ri[i,0]:16.10e}  {g_2e_ri[i,1]:16.10e}  {g_2e_ri[i,2]:16.10e}")
