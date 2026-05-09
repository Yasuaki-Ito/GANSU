"""
Dump RI-HF gradient intermediates — use PySCF's own gradient code as reference.
Trace through pyscf.df.grad.rhf to understand the exact formula.
"""
import numpy as np
from pyscf import gto, scf, df, lib
from pyscf.df.grad import rhf as df_rhf_grad

mol = gto.M(
    atom='O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509',
    basis='sto-3g', unit='Angstrom'
)

mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')
mf.kernel()
D = mf.make_rdm1()
nao = mol.nao
nocc = mol.nelectron // 2
auxmol = df.addons.make_auxmol(mol, auxbasis='cc-pvdz-jkfit')
naux = auxmol.nao

print(f"nao={nao}, naux={naux}")

# === PySCF's own RI gradient code ===
# Let's look at what PySCF does internally
g_obj = mf.nuc_grad_method()

# PySCF DF gradient uses get_veff() which internally computes:
# vj, vk contributions via 3-center and 2-center derivative integrals
# Let's trace through the code

# The key function is pyscf.df.grad.rhf.get_veff
# It computes: vhfopt contributions to gradient

# Let's use PySCF's internal machinery directly
print("\n=== Tracing PySCF df.grad.rhf ===")

# From pyscf source: the DF-RHF gradient 2e part is:
#   g_2e = Σ_P (Σ_μν d3_Pμν d(μν|P)/dR)  + Σ_PQ d2_PQ d(P|Q)/dR
#
# where d3 and d2 are effective densities built from D and the DF coefficients.

# PySCF approach (from df/grad/rhf.py):
# 1. Build DF coefficients: (μν|P) fitted
# 2. Compute rho (= density fitted coefficients)
# 3. Contract with derivative integrals

# Let's just use PySCF's gradient directly and print per-atom contributions
g_anal = g_obj.kernel()

# Also compute via PySCF's internal get_jk function for verification
print(f"\nPySCF total gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  {g_anal[i,0]:16.10e}  {g_anal[i,1]:16.10e}  {g_anal[i,2]:16.10e}")

# === Now manually compute 2e gradient using PySCF's integral machinery ===
# Following Weigend & Ahlrichs, PCCP 2005
#
# The RI-JK 2e gradient has 3 terms:
#   (I)   Σ_Q d̄_Q Σ_μν D_μν d(μν|Q)/dR
#   (II)  -1/2 Σ_PQ d̄_P d(P|Q)/dR d̄_Q
#   (III) Exchange terms (similar structure)
#
# Combined: use "response density" approach
# 3-center: Σ_Q Σ_μν ρ^J_Qμν d(μν|Q)/dR + exchange
# 2-center: Σ_PQ σ_PQ d(P|Q)/dR

# Build intermediates using PySCF
int2c = auxmol.intor('int2c2e')
int3c = df.incore.aux_e2(mol, auxmol, intor='int3c2e').reshape(nao*nao, naux)

# Cholesky
L = np.linalg.cholesky(int2c)
L_inv = np.linalg.inv(L)

# B = L^{-1} V3^T → B[P, μν]
B = L_inv @ int3c.T  # (naux, nao²)

# Coulomb
w = B @ D.flatten()  # w_P = Σ D B_P
d_bar = np.linalg.solve(int2c, int3c.T @ D.flatten())  # d̄ = A^{-1} d

print(f"\nw[0:5] = {w[:5]}")
print(f"d_bar[0:5] = {d_bar[:5]}")

# For Coulomb: ρ^J_μν = Σ_Q d̄_Q δ_{μν-like} → actually it's simpler
# 3c Coulomb contribution density: d̄_Q * D_μν for each Q
# But contracted with d(μν|Q)/dR, this becomes:
#   Σ_Q d̄_Q Σ_μν D_μν d(μν|Q)/dR

# For Exchange:
# ρ^K_{Q,μν} = Σ_λ (d̄·B_Q)_μλ D_λν ... complex
# Let's use the effective density approach from the paper

# Effective 3-center density (Coulomb + Exchange combined):
# ρ_{P,μν} = d̄_P D_μν   (J part)
#           - 0.5 Σ_λ [D B_P^{-1}... ] ... (K part, complex)

# Actually let's just compute the CORRECT Z the simple way:
# dE_2e/dR = Σ_{P,μν} dE/dB_{P,μν} × dB_{P,μν}/dR
# dE/dB = w D - 0.5 D·B·D  (from chain rule of E = 0.5 w² - 0.25 Tr[DBDB])
# dB/dR = L^{-1} dV3/dR  +  dL^{-1}/dR × V3

# The 3-center part: Σ Ztilde × dV3/dR  where Ztilde = L^{-T} (dE/dB) = L^{-T} D̃(3)
# The 2-center part: Σ d2 × dA/dR  where d2 comes from dL^{-1}/dR terms

# Let me compute Z correctly and then use PySCF's integrals
D3_eff = np.outer(w, D.flatten()) - 0.5 * np.array([D @ B.reshape(naux,nao,nao)[P] @ D for P in range(naux)]).reshape(naux, nao*nao)
Z = np.linalg.solve(L.T, D3_eff)  # Z = L^{-T} D̃(3)

print(f"\nD3_eff[0,:5] = {D3_eff[0,:5]}")
print(f"Z[0,:5] = {Z[0,:5]}")
print(f"Z norm = {np.linalg.norm(Z):.6f}")

# 3-center gradient: Σ_{Q,μν} Z_{Q,μν} d(μν|Q)/dR_A
# PySCF int3c2e_ip1: -d(μν|Q)/dR_μ  (derivative on center of μ)
# PySCF int3c2e_ip2: -d(μν|Q)/dR_Q  (derivative on auxiliary center)
# By translational invariance: d/dR_μ + d/dR_ν + d/dR_Q = 0

int3c_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1').reshape(3, nao, nao, naux)
int3c_ip2 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip2').reshape(3, nao, nao, naux)
int2c_ip1 = auxmol.intor('int2c2e_ip1')  # -d(P|Q)/dR_P

aoslices = mol.aoslice_by_atom()
auxslices = auxmol.aoslice_by_atom()

Z_3d = Z.reshape(naux, nao, nao)

# PySCF convention: ip1 = -d/dR_1
# So Σ Z d(μν|P)/dR_A = Σ Z × (-ip1) when μ is on A
# = -Σ Z × ip1

grad_3c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    p0, p1 = aoslices[iatm, 2], aoslices[iatm, 3]
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # μ on atom A: d(μν|Q)/dR_A = -int3c_ip1[xyz, μ, ν, Q]
        grad_3c[iatm, xyz] -= np.einsum('Quv,uvQ->', Z_3d[:, p0:p1, :], int3c_ip1[xyz, p0:p1, :, :])
        # ν on atom A: d(μν|Q)/dR_A for ν center
        # (μν|Q) = (νμ|Q) by symmetry, so d/dR_ν of (μν|Q) = d/dR_1 of (νμ|Q) = -int3c_ip1[xyz, ν, μ, Q]
        # But Z[Q,μ,ν] ≠ Z[Q,ν,μ] in general, so:
        grad_3c[iatm, xyz] -= np.einsum('Quv,vuQ->', Z_3d[:, :, p0:p1], int3c_ip1[xyz, p0:p1, :, :])
        # Q on atom A: d(μν|Q)/dR_A = -int3c_ip2[xyz, μ, ν, Q]
        grad_3c[iatm, xyz] -= np.einsum('Quv,uvQ->', Z_3d[q0:q1, :, :], int3c_ip2[xyz, :, :, q0:q1])

# 2-center: Σ D2_PQ d(P|Q)/dR_A
D2_eff = -Z @ B.T  # (naux, naux)
grad_2c = np.zeros((mol.natm, 3))
for iatm in range(mol.natm):
    q0, q1 = auxslices[iatm, 2], auxslices[iatm, 3]
    for xyz in range(3):
        # P on atom A: d(P|Q)/dR_A = -int2c_ip1[xyz, P, Q]
        grad_2c[iatm, xyz] -= np.einsum('PQ,PQ->', D2_eff[q0:q1, :], int2c_ip1[xyz, q0:q1, :])
        # Q on atom A: d(P|Q)/dR_Q = -int2c_ip1[xyz, Q, P] (derivative on Q center for (Q|P) integral)
        # But (P|Q) = (Q|P), so d(P|Q)/dR_Q = d(Q|P)/dR_Q = -int2c_ip1[xyz, Q, P]
        grad_2c[iatm, xyz] -= np.einsum('PQ,QP->', D2_eff[:, q0:q1], int2c_ip1[xyz, q0:q1, :])

g_2e_my = grad_3c + grad_2c

# Reference
from pyscf.grad import rhf as rhf_grad
g_nuc = rhf_grad.grad_nuc(mol)
h1 = -(mol.intor('int1e_ipkin', comp=3) + mol.intor('int1e_ipnuc', comp=3))
s1 = -mol.intor('int1e_ipovlp', comp=3)
C = mf.mo_coeff
e = mf.mo_energy
W = 2.0 * C[:, :nocc] @ np.diag(e[:nocc]) @ C[:, :nocc].T
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

g_1e_total = g_1e + g_nuc
g_2e_ref = g_anal - g_1e_total

print(f"\n=== 2e gradient comparison ===")
print(f"{'Atom':4s}  {'my 3c':>12s}  {'my 2c':>12s}  {'my total':>12s}  {'ref':>12s}  {'diff':>12s}")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s} z {grad_3c[i,2]:12.6e} {grad_2c[i,2]:12.6e} {g_2e_my[i,2]:12.6e} {g_2e_ref[i,2]:12.6e} {g_2e_my[i,2]-g_2e_ref[i,2]:12.6e}")

print(f"\n=== Check: total gradient = 1e + 2e ===")
g_total_my = g_1e_total + g_2e_my
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):2s}  my: {g_total_my[i,2]:12.6e}  ref: {g_anal[i,2]:12.6e}  diff: {g_total_my[i,2]-g_anal[i,2]:12.6e}")
