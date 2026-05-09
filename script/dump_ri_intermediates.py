"""
Dump all RI-HF gradient intermediates for comparison with GANSU.
Uses same geometry and basis as GANSU test.
"""
import numpy as np
from pyscf import gto, scf, df

mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

# Use cc-pvdz-jkfit (PySCF default for density fitting)
mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.kernel()

D = mf.make_rdm1()
C = mf.mo_coeff
e = mf.mo_energy
nao = mol.nao
nocc = mol.nelectron // 2
auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-jkfit')
naux = auxmol.nao

print(f"nao={nao}, naux={naux}, nocc={nocc}")
print(f"E_tot = {mf.e_tot:.12f}")

# === Raw integrals ===
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e')  # (nao, nao, naux)
int2c = auxmol.intor('int2c2e')  # (naux, naux)

# === B matrix ===
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)
V3 = int3c.reshape(nao*nao, naux)  # (nao², naux)
B = (L_inv @ V3.T)  # (naux, nao²) — B[P, μν]

# === w_P ===
D_flat = D.flatten()
w = B @ D_flat  # (naux,)

# === D̃(3) = w ⊗ D - 0.5 D·B·D ===
D3_coulomb = np.outer(w, D_flat)  # (naux, nao²)

# D·B_P·D for each P
B_reshaped = B.reshape(naux, nao, nao)  # B[P, μ, ν]
D3_exchange = np.zeros((naux, nao, nao))
for P in range(naux):
    D3_exchange[P] = D @ B_reshaped[P] @ D
D3_exchange = D3_exchange.reshape(naux, nao*nao)

D3_eff = D3_coulomb - 0.5 * D3_exchange

# === Z = L^{-T} D̃(3) ===
# Z[Q, μν] = Σ_P (L^{-T})_{QP} D3_eff[P, μν]
# = Σ_P (L^{-1})_{PQ} D3_eff[P, μν]
# = (L^{-T} D3)  where L^{-T} = (L^{-1})^T = (L^T)^{-1}
Z = np.linalg.solve(L.T, D3_eff)  # L^T Z = D3 → Z = L^{-T} D3

# === D̃(2) = -Z · B^T ===
D2_eff = -Z @ B.T  # (naux, naux)

print(f"\n=== Intermediate norms ===")
print(f"D norm: {np.linalg.norm(D):.6f}")
print(f"B norm: {np.linalg.norm(B):.6f}")
print(f"w norm: {np.linalg.norm(w):.6f}")
print(f"D3_coulomb norm: {np.linalg.norm(D3_coulomb):.6f}")
print(f"D3_exchange norm: {np.linalg.norm(D3_exchange):.6f}")
print(f"D3_eff norm: {np.linalg.norm(D3_eff):.6f}")
print(f"Z norm: {np.linalg.norm(Z):.6f}")
print(f"D2_eff norm: {np.linalg.norm(D2_eff):.6f}")

# === Contract with derivatives ===
int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1')  # (3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2')  # (3, nao, nao, naux)
int2c_ip1 = auxmol.intor('int2c2e_ip1')  # (3, naux, naux)

# Negate Z for the sign convention: kernels compute -d/dR, so -Z × (-d/dR) = Z × d/dR
Z_neg = -Z

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

grad_3c = np.zeros((mol.natm, 3))
grad_2c = np.zeros((mol.natm, 3))

# 3-center: Σ_{Q,μν} Z_neg[Q,μν] * d(μν|Q)/dR_A
# ip1 gives -d/dR_1 (first center)
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    Z_neg_3d = Z_neg.reshape(naux, nao, nao)
    for xyz in range(3):
        # μ on atom A: contract Z_neg[Q, μ, ν] with ip1[xyz, μ, ν, Q]
        grad_3c[iatm, xyz] += np.einsum('Quv,uvQ->', Z_neg_3d[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
        # ν on atom A (use symmetry of (μν|Q)):
        grad_3c[iatm, xyz] += np.einsum('Quv,vuQ->', Z_neg_3d[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])

# Auxiliary center contribution
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    Z_neg_3d = Z_neg.reshape(naux, nao, nao)
    for xyz in range(3):
        grad_3c[iatm, xyz] += np.einsum('Quv,uvQ->', Z_neg_3d[q0:q1, :, :], int3c_ip2[xyz, :, :, q0:q1])

# 2-center: Σ_{PQ} D2_eff[PQ] * d(P|Q)/dR_A
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # P on atom A
        grad_2c[iatm, xyz] += np.einsum('PQ,PQ->', D2_eff[q0:q1, :], int2c_ip1[xyz, q0:q1, :])
        # Q on atom A (use ip1 symmetry: d(P|Q)/dR_Q)
        # By translational invariance: d(P|Q)/dR_Q = -d(P|Q)/dR_P when P≠Q center
        # Actually int2c_ip1 gives -d/dR_1 for the first center.
        # For Q on atom A, we need d(P|Q)/dR_Q. Use: int2c_ip1[xyz, Q, P] gives -d(Q|P)/dR_Q = -d(P|Q)/dR_Q
        # So contribution = -Σ D2[P,Q] int2c_ip1[xyz, Q, P] for Q in [q0,q1)
        # But D2 may not be symmetric, so careful:
        grad_2c[iatm, xyz] += np.einsum('PQ,QP->', D2_eff[:, q0:q1], int2c_ip1[xyz, q0:q1, :])

print(f"\n=== RI 2e gradient (with -Z convention) ===")
g_ri_2e = grad_3c + grad_2c
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  3c: {grad_3c[i,2]:12.6e}  2c: {grad_2c[i,2]:12.6e}  total: {g_ri_2e[i,2]:12.6e}")

# Reference: PySCF analytical gradient
g_ref = mf.nuc_grad_method().kernel()
# 1e terms from breakdown
h1 = -(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
s1 = -mol.intor('int1e_ipovlp', comp=3)
W = 2.0 * C[:, :nocc] @ np.diag(e[:nocc]) @ C[:, :nocc].T
g_1e = np.zeros((mol.natm, 3))
g_ovlp = np.zeros((mol.natm, 3))
from pyscf.grad import rhf as rhf_grad
g_nuc = rhf_grad.grad_nuc(mol)

for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    for xyz in range(3):
        g_1e[iatm, xyz] = (np.einsum('uv,uv->', D[p0:p1], h1[xyz, p0:p1])
                          + np.einsum('uv,vu->', D[:, p0:p1], h1[xyz, p0:p1]))
        g_ovlp[iatm, xyz] = (np.einsum('uv,uv->', W[p0:p1], s1[xyz, p0:p1])
                             + np.einsum('uv,vu->', W[:, p0:p1], s1[xyz, p0:p1]))
    with mol.with_rinv_origin(mol.atom_coord(iatm)):
        vrinv = mol.intor('int1e_iprinv', comp=3)
    g_1e[iatm] -= np.einsum('xij,ij->x', vrinv, D) * mol.atom_charge(iatm)

g_1e_total = g_1e + g_ovlp + g_nuc
g_2e_ref = g_ref - g_1e_total

print(f"\n=== Reference 2e gradient (from PySCF total - 1e) ===")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_2e_ref[i,0]:12.6e}  {g_2e_ref[i,1]:12.6e}  {g_2e_ref[i,2]:12.6e}")

print(f"\n=== My 2e gradient ===")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_ri_2e[i,0]:12.6e}  {g_ri_2e[i,1]:12.6e}  {g_ri_2e[i,2]:12.6e}")

print(f"\n=== Diff (my - ref) ===")
for i in range(mol.natm):
    d = g_ri_2e[i] - g_2e_ref[i]
    print(f"  {mol.atom_symbol(i):2s}  {d[0]:12.6e}  {d[1]:12.6e}  {d[2]:12.6e}")
