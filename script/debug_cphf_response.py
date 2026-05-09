"""
Debug CPHF response: compute all intermediates and verify 3-term formula.
Uses finite difference for h1ao/s1ao (same as GANSU).
"""
import numpy as np
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess

mol = gto.Mole()
mol.atom = "O 0 0 0.127; H 0 0.758 -0.509; H 0 -0.758 -0.509"
mol.basis = "sto-3g"
mol.unit = "Angstrom"
mol.verbose = 0
mol.build()

mf = scf.RHF(mol).run()
print(f"E = {mf.e_tot:.10f}")

natm = mol.natm
nao = mol.nao_nr()
nocc = mol.nelectron // 2
nvir = nao - nocc
nmo = nao
ndim = 3 * natm
C = mf.mo_coeff
eps = mf.mo_energy
D = mf.make_rdm1()  # 2 * C_occ @ C_occ.T

# ---------------------------------------------------------------
# Step 1: h1ao and s1ao via finite difference (same method as GANSU)
# ---------------------------------------------------------------
h_fd = 1e-4

def rebuild_fock(mol_disp):
    """Rebuild Fock matrix at displaced geometry using converged density."""
    mf_disp = scf.RHF(mol_disp)
    mf_disp.verbose = 0
    # Get integrals at displaced geometry
    S = mf_disp.get_ovlp()
    h_core = mf_disp.get_hcore()
    # Build Fock with ORIGINAL density D
    vj, vk = mf_disp.get_jk(dm=D)
    F = h_core + vj - 0.5 * vk
    return F, S

h1ao = np.zeros((ndim, nao, nao))
s1ao = np.zeros((ndim, nao, nao))

coords = mol.atom_coords().copy()  # in Bohr
for coord in range(ndim):
    ai, di = divmod(coord, 3)
    for sign, delta in [(+1, +h_fd), (-1, -h_fd)]:
        new_coords = coords.copy()
        new_coords[ai, di] += delta
        mol_d = mol.copy()
        mol_d.set_geom_(new_coords, unit='Bohr')
        mol_d.build()
        F_d, S_d = rebuild_fock(mol_d)
        if sign > 0:
            Fp, Sp = F_d, S_d
        else:
            Fm, Sm = F_d, S_d
    h1ao[coord] = (Fp - Fm) / (2 * h_fd)
    s1ao[coord] = (Sp - Sm) / (2 * h_fd)

print(f"\nh1ao[Oy] diagonal: {np.diag(h1ao[1])}")
print(f"s1ao[Oy] diagonal: {np.diag(s1ao[1])}")

# ---------------------------------------------------------------
# Step 2: Transform to MO basis, build CPHF RHS
# ---------------------------------------------------------------
C_occ = C[:, :nocc]
eps_occ = eps[:nocc]

def ao2mo(M):
    return C.T @ M @ C

h1_mo = np.array([ao2mo(h1ao[p]) for p in range(ndim)])
s1_mo = np.array([ao2mo(s1ao[p]) for p in range(ndim)])
s1oo = s1_mo[:, :nocc, :nocc]

# CPHF RHS: rhs[ai] = -(F^x[a,i] - eps_i * S^x[a,i])
rhs = np.zeros((ndim, nocc * nvir))
for p in range(ndim):
    for i in range(nocc):
        for a in range(nvir):
            a_mo = nocc + a
            rhs[p, i * nvir + a] = -(h1_mo[p, a_mo, i] - eps[i] * s1_mo[p, a_mo, i])

# ---------------------------------------------------------------
# Step 3: Solve CPHF using PySCF's infrastructure
# ---------------------------------------------------------------
from pyscf import ao2mo as pyscf_ao2mo

# Full MO ERI
eri_ao = mol.intor('int2e')
eri_mo = pyscf_ao2mo.full(eri_ao.reshape(nao,nao,nao,nao), C)

def cphf_matvec(U_flat):
    """A * U where A is the CPHF operator."""
    U = U_flat.reshape(nocc, nvir)
    AU = np.zeros_like(U)
    for i in range(nocc):
        for a in range(nvir):
            a_mo = nocc + a
            AU[i, a] = (eps[a_mo] - eps[i]) * U[i, a]
            for j in range(nocc):
                for b in range(nvir):
                    b_mo = nocc + b
                    aibj = eri_mo[a_mo, i, b_mo, j]
                    abij = eri_mo[a_mo, b_mo, i, j]
                    ajib = eri_mo[a_mo, j, i, b_mo]
                    AU[i, a] += (4*aibj - abij - ajib) * U[j, b]
    return AU.flatten()

# Solve A*U = rhs (rhs already contains -B)
from scipy.sparse.linalg import LinearOperator, cg
n_cphf = nocc * nvir
A_op = LinearOperator((n_cphf, n_cphf), matvec=cphf_matvec)

U_all = np.zeros((ndim, nocc, nvir))
for p in range(ndim):
    sol, info = cg(A_op, rhs[p], atol=1e-10, maxiter=200)
    U_all[p] = sol.reshape(nocc, nvir)
    if info != 0:
        print(f"  CPHF pert {p}: NOT converged (info={info})")

print(f"\nU[Oy] (first few): {U_all[1].flatten()[:5]}")

# ---------------------------------------------------------------
# Step 4: Build response quantities
# ---------------------------------------------------------------
# mo1[p,i]: vir-occ from U, occ-occ from -0.5*s1oo
mo1 = np.zeros((ndim, nmo, nocc))
for p in range(ndim):
    mo1[p, nocc:, :] = U_all[p].T  # mo1[a,i] = U[i,a].T → U_all[p][:,a] for a-th vir
    # Actually U_all[p] is (nocc, nvir), so U_all[p][i,a] = U[i,a]
    # mo1[p, nocc+a, i] = U_all[p][i, a]
    for i in range(nocc):
        for a in range(nvir):
            mo1[p, nocc + a, i] = U_all[p][i, a]
    mo1[p, :nocc, :] = -0.5 * s1oo[p]

# dm1 = C * mo1 * C_occ^T (one-sided, nao x nao)
dm1 = np.zeros((ndim, nao, nao))
dm1e = np.zeros((ndim, nao, nao))
for p in range(ndim):
    temp = C @ mo1[p]  # (nao, nocc)
    dm1[p] = temp @ C_occ.T
    dm1e[p] = (temp * eps_occ) @ C_occ.T

print(f"\ndm1[Oy] trace: {np.trace(dm1[1]):.8f}")
print(f"dm1e[Oy] trace: {np.trace(dm1e[1]):.8f}")

# ---------------------------------------------------------------
# Step 5: Compute vhf1 and mo_e1
# ---------------------------------------------------------------
# D1 = 2*(dm1 + dm1^T)
D1 = np.zeros((ndim, nao, nao))
for p in range(ndim):
    D1[p] = 2 * (dm1[p] + dm1[p].T)

# vhf1 = G(D1) = J(D1) - 0.5*K(D1)
vhf1 = np.zeros((ndim, nao, nao))
for p in range(ndim):
    vj = np.einsum('ijkl,kl->ij', eri_ao.reshape(nao,nao,nao,nao), D1[p])
    vk = np.einsum('ikjl,kl->ij', eri_ao.reshape(nao,nao,nao,nao), D1[p])
    vhf1[p] = vj - 0.5 * vk

# F_tot = h1ao + vhf1
F_tot = h1ao + vhf1
F_tot_mo = np.array([ao2mo(F_tot[p]) for p in range(ndim)])

# mo_e1 with different formulas
mo_e1_v1 = np.zeros((ndim, nocc, nocc))  # F_tot + eps_j * mo1_oo
mo_e1_v2 = np.zeros((ndim, nocc, nocc))  # F_tot + eps_j * s1oo (without -0.5)
for p in range(ndim):
    for i in range(nocc):
        for j in range(nocc):
            mo_e1_v1[p, i, j] = F_tot_mo[p, i, j] + eps[j] * (-0.5 * s1oo[p, i, j])
            mo_e1_v2[p, i, j] = F_tot_mo[p, i, j] + eps[j] * s1oo[p, i, j]

# ---------------------------------------------------------------
# Step 6: Assemble response with different formulas
# ---------------------------------------------------------------
# Get PySCF reference
hess_obj = rhf_hess.Hessian(mf)
H_full = np.zeros((ndim, ndim))
hf = hess_obj.kernel()
for i in range(natm):
    for j in range(natm):
        H_full[3*i:3*i+3, 3*j:3*j+3] = hf[i, j]

hess_skel = hess_obj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
hess_nuc = rhf_hess.hess_nuc(mol)
H_skel = np.zeros((ndim, ndim))
for i in range(natm):
    for j in range(natm):
        H_skel[3*i:3*i+3, 3*j:3*j+3] = hess_skel[i,j] + hess_nuc[i,j]

H_resp_ref = H_full - H_skel

def assemble_response(h1, s1, dm1_, dm1e_, s1oo_, mo_e1_):
    """3-term formula."""
    R = np.zeros((ndim, ndim))
    for x in range(ndim):
        for y in range(ndim):
            t1 = 4 * np.sum(h1[x] * dm1_[y])
            t2 = -4 * np.sum(s1[x] * dm1e_[y])
            t3 = -2 * np.sum(s1oo_[x] * mo_e1_[y])
            R[x, y] = t1 + t2 + t3
    return R

def assemble_response_4term(h1, s1, dm1_):
    """PySCF-style: 4*tr(h1ao*dm1) - 4*tr(s1ao*dm1e) - 2*tr(s1ao*dm1)"""
    R = np.zeros((ndim, ndim))
    for x in range(ndim):
        for y in range(ndim):
            t1 = 4 * np.sum(h1[x] * dm1_[y])
            t2 = -4 * np.sum(s1[x] * dm1e[y])
            t3 = -2 * np.sum(s1[x] * dm1_[y])
            R[x, y] = t1 + t2 + t3
    return R

R_v1 = assemble_response(h1ao, s1ao, dm1, dm1e, s1oo, mo_e1_v1)
R_v2 = assemble_response(h1ao, s1ao, dm1, dm1e, s1oo, mo_e1_v2)
R_4t = assemble_response_4term(h1ao, s1ao, dm1)

# Also try using PySCF's own CPHF solution
hess2 = rhf_hess.Hessian(mf)
h1ao_pyscf = rhf_hess.make_h1(hess2, C, mf.mo_occ)
mo1_pyscf, mo_e1_pyscf = hess2.solve_mo1(eps, C, mf.mo_occ, h1ao_pyscf)

# Build dm1 from PySCF's mo1
dm1_pyscf = np.zeros((ndim, nao, nao))
dm1e_pyscf = np.zeros((ndim, nao, nao))
s1oo_pyscf = np.zeros((ndim, nocc, nocc))
mo_e1_flat_pyscf = np.zeros((ndim, nocc, nocc))
for ia in range(natm):
    for x in range(3):
        idx = 3 * ia + x
        mo1_ix = mo1_pyscf[ia][x]  # (nmo, nocc)
        temp = C @ mo1_ix  # (nao, nocc)
        dm1_pyscf[idx] = temp @ C_occ.T
        dm1e_pyscf[idx] = (temp * eps_occ) @ C_occ.T
        s1oo_pyscf[idx] = C_occ.T @ s1ao[idx] @ C_occ
        mo_e1_flat_pyscf[idx] = mo_e1_pyscf[ia][x]

R_pyscf = assemble_response(h1ao, s1ao, dm1_pyscf, dm1e_pyscf, s1oo_pyscf, mo_e1_flat_pyscf)

# ---------------------------------------------------------------
# Print comparison
# ---------------------------------------------------------------
print(f"\n{'='*80}")
print("Response Hessian diagonal comparison:")
print(f"{'coord':>6}  {'PySCF ref':>10}  {'v1(mo1_oo)':>10}  {'v2(s1oo)':>10}  {'4-term':>10}  {'PySCF mo1':>10}")
labels = []
for i in range(natm):
    for c in 'xyz':
        labels.append(f"{mol.atom_symbol(i)}{i}_{c}")

for i in range(ndim):
    print(f"{labels[i]:>6}  {H_resp_ref[i,i]:10.6f}  {R_v1[i,i]:10.6f}  {R_v2[i,i]:10.6f}  {R_4t[i,i]:10.6f}  {R_pyscf[i,i]:10.6f}")

print(f"\nMax |diff| from PySCF ref:")
print(f"  v1 (mo_e1 = F_tot + eps*mo1_oo):  {np.max(np.abs(R_v1 - H_resp_ref)):.6f}")
print(f"  v2 (mo_e1 = F_tot + eps*s1oo):     {np.max(np.abs(R_v2 - H_resp_ref)):.6f}")
print(f"  4-term (no mo_e1):                  {np.max(np.abs(R_4t - H_resp_ref)):.6f}")
print(f"  PySCF mo1/mo_e1:                    {np.max(np.abs(R_pyscf - H_resp_ref)):.6f}")

# Print PySCF mo_e1 vs our mo_e1 for pert Oy
print(f"\nmo_e1 comparison for Oy (pert=1):")
print(f"  PySCF mo_e1:\n{mo_e1_flat_pyscf[1]}")
print(f"  v1 (F_tot + eps*mo1_oo):\n{mo_e1_v1[1]}")
print(f"  v2 (F_tot + eps*s1oo):\n{mo_e1_v2[1]}")

# Compare mo1
print(f"\nmo1 comparison for Oy (pert=1), vir-occ block:")
print(f"  Our U:    {U_all[1].flatten()}")
print(f"  PySCF mo1 vir-occ: {mo1_pyscf[0][1][nocc:,:].flatten()}")

print(f"\nmo1 occ-occ block:")
print(f"  Our (-0.5*s1oo): {(-0.5*s1oo[1]).flatten()}")
print(f"  PySCF mo1 occ-occ: {mo1_pyscf[0][1][:nocc,:].flatten()}")
