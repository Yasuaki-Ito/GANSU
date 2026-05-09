"""
PySCF verification: full analytical Hessian for H2O/STO-3G
Decomposes into skeleton + response (CPHF) terms.

Run with: wsl python3 script/verify_hessian_cphf.py
"""

import numpy as np
from pyscf import gto, scf
from pyscf.hessian import rhf as rhf_hess

# -----------------------------------------------------------------------
# Molecule — coordinates must match GANSU xyz/H2O.xyz exactly
# -----------------------------------------------------------------------
mol = gto.Mole()
mol.atom = """
O   0.000   0.000   0.127
H   0.000   0.758  -0.509
H   0.000  -0.758  -0.509
"""
mol.basis = "sto-3g"
mol.unit = "Angstrom"
mol.verbose = 0
mol.build()

# -----------------------------------------------------------------------
# RHF SCF
# -----------------------------------------------------------------------
mf = scf.RHF(mol)
mf.kernel()
print(f"RHF energy: {mf.e_tot:.10f} Hartree")

natm = mol.natm
ndim = 3 * natm
nao = mol.nao_nr()

# -----------------------------------------------------------------------
# Full analytical Hessian
# -----------------------------------------------------------------------
hess_obj = rhf_hess.Hessian(mf)
hess_full = hess_obj.kernel()

def hess_to_matrix(h):
    M = np.zeros((ndim, ndim))
    for i in range(natm):
        for j in range(natm):
            M[3*i:3*i+3, 3*j:3*j+3] = h[i, j]
    return M

H_full = hess_to_matrix(hess_full)

# -----------------------------------------------------------------------
# Skeleton Hessian
# -----------------------------------------------------------------------
hess_scanner = rhf_hess.Hessian(mf)
hess_elec_skel = hess_scanner.partial_hess_elec(mo_energy=mf.mo_energy,
                                                 mo_coeff=mf.mo_coeff,
                                                 mo_occ=mf.mo_occ)
hess_nuc = rhf_hess.hess_nuc(mol)

H_skel_elec = hess_to_matrix(hess_elec_skel)
H_skel_nuc  = hess_to_matrix(hess_nuc)
H_skel      = H_skel_elec + H_skel_nuc
H_resp      = H_full - H_skel

# -----------------------------------------------------------------------
# CPHF response decomposition into 3 terms
# -----------------------------------------------------------------------
mo_coeff  = mf.mo_coeff
mo_energy = mf.mo_energy
mo_occ    = mf.mo_occ
nmo = mo_coeff.shape[1]
nocc = int(mo_occ.sum()) // 2
nvir = nmo - nocc
C_occ = mo_coeff[:, :nocc]
eps_occ = mo_energy[:nocc]

# h1ao: Fock derivative per atom (list of (3, nao, nao))
h1ao_all = rhf_hess.make_h1(hess_scanner, mo_coeff, mo_occ)

# s1ao: overlap derivative per atom
# PySCF's int1e_ipovlp gives dS/dR_A for basis functions on atom A
# Shape: (3, nao, nao), antisymmetric: s1[x,mu,nu] for mu on atom A
s1ao_all = []
aoslices = mol.aoslice_by_atom()
s1_raw = -mol.intor('int1e_ipovlp', comp=3)  # (3, nao, nao), minus sign for d/dR_A
for ia in range(natm):
    p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
    s1 = np.zeros((3, nao, nao))
    s1[:, p0:p1] += s1_raw[:, p0:p1]
    s1 = s1 + s1.transpose(0, 2, 1)  # symmetrize: s1[x,mu,nu] + s1[x,nu,mu]
    s1ao_all.append(s1)

# CPHF solve
mo1_all, mo_e1_all_pyscf = hess_scanner.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao_all)
# mo1_all[ia]: (3, nmo, nocc)
# mo_e1_all_pyscf[ia]: (3, nocc, nocc)

# Build response terms
H_resp_t1 = np.zeros((ndim, ndim))
H_resp_t2 = np.zeros((ndim, ndim))
H_resp_t3 = np.zeros((ndim, ndim))

for ja in range(natm):
    for y in range(3):
        By = 3 * ja + y
        mo1_jy   = mo1_all[ja][y]        # (nmo, nocc)
        mo_e1_jy = mo_e1_all_pyscf[ja][y] # (nocc, nocc)

        # dm1 = C * mo1 * C_occ^T  (one-sided response density in AO)
        dm1_jy = (mo_coeff @ mo1_jy) @ C_occ.T     # (nao, nao)
        # dm1e = C * mo1 * diag(eps) * C_occ^T
        dm1e_jy = (mo_coeff @ (mo1_jy * eps_occ)) @ C_occ.T  # (nao, nao)

        for ia in range(natm):
            for x in range(3):
                Ax = 3 * ia + x
                h1ao_ix = h1ao_all[ia][x]  # (nao, nao)
                s1ao_ix = s1ao_all[ia][x]  # (nao, nao)

                t1 = 4.0 * np.einsum('pq,pq->', h1ao_ix, dm1_jy)
                t2 = -4.0 * np.einsum('pq,pq->', s1ao_ix, dm1e_jy)

                s1oo_ix = C_occ.T @ s1ao_ix @ C_occ  # (nocc, nocc)
                t3 = -2.0 * np.einsum('ij,ij->', s1oo_ix, mo_e1_jy)

                H_resp_t1[Ax, By] += t1
                H_resp_t2[Ax, By] += t2
                H_resp_t3[Ax, By] += t3

H_resp_check = H_resp_t1 + H_resp_t2 + H_resp_t3

# -----------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------
atom_labels = [mol.atom_symbol(i) for i in range(natm)]
coord_labels = []
for i, sym in enumerate(atom_labels):
    for c in ['x', 'y', 'z']:
        coord_labels.append(f"{sym}{i}_{c}")

def print_matrix(label, M):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    header = f"{'':>8}" + "".join(f"{l:>12}" for l in coord_labels)
    print(header)
    for i in range(ndim):
        row = f"{coord_labels[i]:>8}" + "".join(f"{M[i,j]:12.6f}" for j in range(ndim))
        print(row)

def print_upper_triangle(label, M):
    print(f"\n  {label}:")
    for i in range(ndim):
        for j in range(i, ndim):
            print(f"  [{i},{j}] {M[i,j]:14.8f}")

print("\n" + "="*70)
print("  H2O / STO-3G  RHF Hessian (PySCF)")
print("="*70)

print_matrix("Skeleton Hessian", H_skel)
print_matrix("Response Hessian (full - skel)", H_resp)
print_matrix("Full Hessian", H_full)

print_matrix("Response T1: +4*tr[h1ao(x)*dm1(y)]", H_resp_t1)
print_matrix("Response T2: -4*tr[s1ao(x)*dm1e(y)]", H_resp_t2)
print_matrix("Response T3: -2*tr[s1oo(x)*mo_e1(y)]", H_resp_t3)
print_matrix("Response check = T1+T2+T3", H_resp_check)

# Consistency checks
print(f"\n{'='*70}")
print("  Consistency checks")
print(f"{'='*70}")
diff_resp = np.max(np.abs(H_resp - H_resp_check))
diff_full = np.max(np.abs(H_full - (H_skel + H_resp_check)))
diff_sym  = np.max(np.abs(H_full - H_full.T))
print(f"  max|resp - (T1+T2+T3)|        = {diff_resp:.3e}")
print(f"  max|full - (skel + T1+T2+T3)| = {diff_full:.3e}")
print(f"  max|H_full - H_full.T|        = {diff_sym:.3e}")

# Diagonal summary
print(f"\n{'='*70}")
print("  Diagonal summary")
print(f"{'='*70}")
print(f"  {'coord':>8}  {'skel':>12}  {'resp':>12}  {'full':>12}  {'T1':>12}  {'T2':>12}  {'T3':>12}")
for i, l in enumerate(coord_labels):
    print(f"  {l:>8}  {H_skel[i,i]:12.6f}  {H_resp[i,i]:12.6f}  {H_full[i,i]:12.6f}  "
          f"{H_resp_t1[i,i]:12.6f}  {H_resp_t2[i,i]:12.6f}  {H_resp_t3[i,i]:12.6f}")

# Upper triangle for GANSU comparison
print_upper_triangle("Full Hessian (upper triangle)", H_full)
print_upper_triangle("Skeleton Hessian (upper triangle)", H_skel)
print_upper_triangle("Response Hessian (upper triangle)", H_resp)
print()
