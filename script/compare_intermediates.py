"""
Compare RI-HF gradient intermediates using GANSU's exact auxiliary basis file.
Dump w, d_bar, D3_eff, D2_eff, 3c contribution, 2c contribution.
"""
import numpy as np
from pyscf import gto, scf, df

# Same geometry as GANSU
mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

# Load GANSU's auxiliary basis
auxmol = gto.M(atom=mol.atom, basis='cc-pvdz-ri', unit='Angstrom')
# Try different aux basis names until naux=96
for name in ['cc-pvdz-ri', 'cc-pvdz-rifit', 'cc-pvdz-jkfit', 'weigend']:
    try:
        auxmol = df.addons.make_auxmol(mol, auxbasis=name)
        print(f"  {name}: naux = {auxmol.nao}")
        if auxmol.nao == 96:
            print(f"  → MATCH! Using {name}")
            break
    except:
        pass

nao = mol.nao
naux = auxmol.nao
print(f"nao={nao}, naux={naux}")

if naux != 96:
    print("WARNING: Could not find matching aux basis with naux=96")
    print("Proceeding with naux =", naux)

# DF-RHF with this aux basis
auxbasis_name = name
mf = scf.RHF(mol).density_fit(auxbasis=auxbasis_name)
mf.kernel()
D = mf.make_rdm1()
print(f"E = {mf.e_tot:.10f}")

# Recompute auxmol to match
auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis_name)
naux = auxmol.nao

# Integrals
int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)

# B matrix
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)
B = L_inv @ int3c.T  # (naux, nao²)

# w
w = B @ D.flatten()
d_bar = np.linalg.solve(L.T, w)  # d̄ = L^{-T} w

print(f"\nw[0:5] = {w[:5]}")
print(f"d_bar[0:5] = {d_bar[:5]}")
print(f"w norm = {np.linalg.norm(w):.6f}")
print(f"d_bar norm = {np.linalg.norm(d_bar):.6f}")

# Verify: A d̄ = d where d = L^T w
d_vec = L.T @ w
d_bar_check = np.linalg.solve(int2c, d_vec)
print(f"d_bar via A^-1 d: norm = {np.linalg.norm(d_bar_check):.6f}")
print(f"d_bar vs A^-1 d: max diff = {np.max(np.abs(d_bar - d_bar_check)):.2e}")

# Coulomb D3 = d̄ ⊗ D
D3_coulomb = np.outer(d_bar, D.flatten())  # (naux, nao²)
print(f"D3_coulomb norm = {np.linalg.norm(D3_coulomb):.6f}")

# Coulomb D2 = d̄ ⊗ d̄
D2_coulomb = np.outer(d_bar, d_bar)  # (naux, naux)
print(f"D2_coulomb norm = {np.linalg.norm(D2_coulomb):.6f}")

# Derivative integrals
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)
int2c_ip1 = auxmol.intor('int2c2e_ip1')  # (3, naux, naux) = -d(P|Q)/dR_P

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

# === 3c Coulomb gradient ===
# PySCF convention: ip1 = -d/dR_1
# Contribution: -Σ d̄_P D_μν ip1[μ,ν,P] (when μ on atom)
#             + -Σ d̄_P D_μν ip1[ν,μ,P] (when ν on atom, by symmetry)
#             + -Σ d̄_P D_μν ip2[μ,ν,P] (when P on atom)
grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar, D[p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,vuP->', d_bar, D[:, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        grad_3c[iatm, xyz] -= np.einsum('P,uv,uvP->', d_bar[q0:q1], D, int3c_ip2[xyz, :, :, q0:q1])

# === 2c Coulomb gradient ===
# PySCF: vjaux -= einsum('xpq,mp,nq', int2c_e1, rhoj, rhoj)
#   then vjaux per-atom-sum, then -vjaux → gradient
# int2c_e1 = -d(P|Q)/dR_P
# vjaux[P] -= Σ_Q d̄_P (-d(P|Q)/dR) d̄_Q = +d̄_P d(P|Q)/dR d̄_Q
# gradient[atom] = -Σ_{P on atom} vjaux[P]
# = -Σ_{P on atom} d̄_P d(P|Q)/dR d̄_Q
# = +Σ_{P on atom} d̄_P (-d(P|Q)/dR) d̄_Q  (using ip1 = -d/dR)
# = +Σ_{P on atom} d̄_P ip1[P,Q] d̄_Q

grad_2c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # P on this atom
        grad_2c[iatm, xyz] += np.einsum('P,PQ,Q->', d_bar[q0:q1], int2c_ip1[xyz, q0:q1, :], d_bar)
        # Q on this atom: by symmetry (P|Q)=(Q|P), d(P|Q)/dR_Q = d(Q|P)/dR_Q
        # int2c_ip1 gives -d(P|Q)/dR_P. For Q center: -d(Q|P)/dR_Q = int2c_ip1[xyz, Q, P]
        grad_2c[iatm, xyz] += np.einsum('P,QP,Q->', d_bar, int2c_ip1[xyz, q0:q1, :], d_bar[q0:q1])

print(f"\n=== Coulomb gradient (O atom, dz) ===")
print(f"  3c = {grad_3c[0,2]:.10e}")
print(f"  2c = {grad_2c[0,2]:.10e}")
print(f"  total = {grad_3c[0,2]+grad_2c[0,2]:.10e}")

# Full reference
g_ref = mf.nuc_grad_method().kernel()
print(f"\n  Full RI-HF (O dz) = {g_ref[0,2]:.10e}")

# All atoms
print(f"\n=== Coulomb gradient (all atoms) ===")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  3c: {grad_3c[i,2]:12.6e}  2c: {grad_2c[i,2]:12.6e}  total: {grad_3c[i,2]+grad_2c[i,2]:12.6e}")
