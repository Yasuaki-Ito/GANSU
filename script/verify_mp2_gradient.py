#!/usr/bin/env python3
"""
PySCF verification script for MP2 gradient debugging.
Matches GANSU H2O/STO-3G calculation for element-by-element comparison.

Usage: python3 verify_mp2_gradient.py
"""

import numpy as np
from pyscf import gto, scf, mp, lib
from pyscf.mp import mp2 as mp2_mod
from pyscf.grad import mp2 as mp2_grad

np.set_printoptions(precision=12, linewidth=200)

# -------------------------------------------------------
# 1. Build molecule (matching GANSU xyz/H2O.xyz exactly)
# -------------------------------------------------------
mol = gto.Mole()
mol.atom = '''
O   0.000   0.000   0.127
H   0.000   0.758  -0.509
H   0.000  -0.758  -0.509
'''
mol.basis = 'sto-3g'
mol.unit = 'Angstrom'
mol.verbose = 4
mol.build()

print("=" * 60)
print("Molecule: H2O / STO-3G")
print(f"nao = {mol.nao_nr()}, nelec = {mol.nelectron}")
print("=" * 60)

# -------------------------------------------------------
# 2. RHF
# -------------------------------------------------------
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
print(f"\nRHF Energy = {mf.e_tot:.12f}")

nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc

print(f"\nnocc = {nocc}, nvir = {nvir}, nmo = {nmo}")

# -------------------------------------------------------
# 3. Orbital energies
# -------------------------------------------------------
eps = mf.mo_energy
print(f"\nOrbital energies:")
for p in range(nmo):
    print(f"  eps[{p}] = {eps[p]:.12f}")

# -------------------------------------------------------
# 4. MO integrals (ia|jb) -- chemist's notation
# -------------------------------------------------------
# Full MO ERI
eri_mo = mol.ao2mo(mf.mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)

print(f"\nOVOV integrals (ia|jb) -- chemist's notation:")
print("  GANSU layout: h_ovov[i*nvir*nocc*nvir + a*nocc*nvir + j*nvir + b] = (ia|jb)")
for i in range(nocc):
    for a in range(nvir):
        for j in range(nocc):
            for b in range(nvir):
                val = eri_mo[i, nocc+a, j, nocc+b]
                print(f"  ({i},{a},{j},{b}) = {val:.12f}")

# OVOV norm
ovov_block = eri_mo[:nocc, nocc:, :nocc, nocc:]
print(f"\n|OVOV| = {np.linalg.norm(ovov_block):.12f}")

# -------------------------------------------------------
# 5. MP2 energy and T2 amplitudes
# -------------------------------------------------------
mymp2 = mp.MP2(mf)
mymp2.kernel()
print(f"\nMP2 correlation energy = {mymp2.e_corr:.12f}")
print(f"MP2 total energy = {mymp2.e_tot:.12f}")

# Get T2 amplitudes: PySCF t2[i,j,a,b]
t2 = mymp2.t2

# Verify MP2 energy from T2
e_mp2_check = 0.0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                iajb = eri_mo[i, nocc+a, j, nocc+b]
                iajb_swap = eri_mo[i, nocc+b, j, nocc+a]
                D = eps[i] + eps[j] - eps[nocc+a] - eps[nocc+b]
                e_mp2_check += iajb * (2*iajb - iajb_swap) / D
print(f"E_MP2 recomputed = {e_mp2_check:.12f}")

# T2 norms
print(f"\n|T2| (PySCF) = {np.linalg.norm(t2):.12f}")

# Build GANSU-layout T2 for comparison
t2_gansu = np.zeros((nocc, nvir, nocc, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                D = eps[i] + eps[j] - eps[nocc+a] - eps[nocc+b]
                t2_gansu[i, a, j, b] = eri_mo[i, nocc+a, j, nocc+b] / D
print(f"|T2| (GANSU layout) = {np.linalg.norm(t2_gansu):.12f}")

# T2tilde: 2*T - T(swap a,b)
t2tilde_gansu = np.zeros_like(t2_gansu)
for i in range(nocc):
    for a in range(nvir):
        for j in range(nocc):
            for b in range(nvir):
                t2tilde_gansu[i, a, j, b] = 2.0 * t2_gansu[i, a, j, b] - t2_gansu[i, b, j, a]
print(f"|T2tilde| (GANSU layout) = {np.linalg.norm(t2tilde_gansu):.12f}")

# -------------------------------------------------------
# 6. Unrelaxed MP2 density
# -------------------------------------------------------
P_oo = np.zeros((nocc, nocc))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for k in range(nocc):
                for b in range(nvir):
                    P_oo[i, j] += -t2tilde_gansu[i, a, k, b] * t2_gansu[j, a, k, b]

P_vv = np.zeros((nvir, nvir))
for a in range(nvir):
    for b in range(nvir):
        for i in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    P_vv[a, b] += t2tilde_gansu[i, a, j, c] * t2_gansu[i, b, j, c]

print(f"\n|P_oo| = {np.linalg.norm(P_oo):.12f}")
print(f"|P_vv| = {np.linalg.norm(P_vv):.12f}")
print(f"P_oo diag: {np.diag(P_oo)}")
print(f"P_vv diag: {np.diag(P_vv)}")

# dm1mo_unrelaxed = P_oo(sym) in oo, P_vv(sym) in vv
dm1_unrelax = np.zeros((nmo, nmo))
for i in range(nocc):
    for j in range(nocc):
        dm1_unrelax[i, j] = P_oo[i, j] + P_oo[j, i]
for a in range(nvir):
    for b in range(nvir):
        dm1_unrelax[nocc+a, nocc+b] = P_vv[a, b] + P_vv[b, a]
print(f"|dm1_unrelax| = {np.linalg.norm(dm1_unrelax):.12f}")

# PySCF's rdm1 for comparison
dm1_pyscf = mymp2.make_rdm1()  # in MO basis, includes HF part (2*delta_ij)
print(f"\nPySCF dm1 (MO, with HF):")
print(dm1_pyscf)

# -------------------------------------------------------
# 7. Z-vector extraction from PySCF
# -------------------------------------------------------
dm1_mo_corr = dm1_pyscf - np.diag([2.0]*nocc + [0.0]*nvir)
print(f"\n|dm1mo| (MP2 corr only, PySCF) = {np.linalg.norm(dm1_mo_corr):.12f}")
print(f"dm1mo (MO, MP2 correction):")
print(dm1_mo_corr)

# Z-vector = ov block of dm1_corr
z_pyscf = dm1_mo_corr[nocc:, :nocc]  # z(a,i)
print(f"\nZ-vector from PySCF (ov block of dm1_corr):")
print(f"|z| (PySCF) = {np.linalg.norm(z_pyscf):.12f}")
print("z(a,i) elements:")
for a in range(nvir):
    for i in range(nocc):
        print(f"  z({a},{i}) = {z_pyscf[a,i]:.12f}")

# -------------------------------------------------------
# 7b. Manually solve CPHF for comparison
# -------------------------------------------------------
print(f"\n{'='*60}")
print("Manual CPHF solve (matching GANSU)")
print(f"{'='*60}")

# Build CPHF RHS: Xvo(a,i) = (2J-K)(dm1_unrelax)_MO in the vo block
# NOTE: PySCF get_veff returns (J - 0.5*K), NOT (2J - K).
# GANSU uses (2J - K) convention. Use get_jk for explicit control.
C = mf.mo_coeff
dm_ao_unrelax = C @ dm1_unrelax @ C.T
vj, vk = mf.get_jk(mol, dm_ao_unrelax)
vhf_2jk_ao = 2*vj - vk   # (2J-K) convention, matching GANSU
vhf_jk05_ao = vj - 0.5*vk  # (J-0.5K) convention, PySCF get_veff
vhf_2jk_mo = C.T @ vhf_2jk_ao @ C
vhf_jk05_mo = C.T @ vhf_jk05_ao @ C
print(f"|vhf_MO(2J-K)| = {np.linalg.norm(vhf_2jk_mo):.12f}")
print(f"|vhf_MO(J-0.5K)| = {np.linalg.norm(vhf_jk05_mo):.12f}")
print(f"Ratio (2J-K)/(J-0.5K) = {np.linalg.norm(vhf_2jk_mo)/np.linalg.norm(vhf_jk05_mo):.6f}")

# Extract Xvo with (2J-K) convention (matching GANSU)
nvo = nvir * nocc
Xvo_2jk = np.zeros(nvo)
for a in range(nvir):
    for i in range(nocc):
        Xvo_2jk[a * nocc + i] = vhf_2jk_mo[nocc+a, i]
print(f"|Xvo(2J-K)| = {np.linalg.norm(Xvo_2jk):.12f}")

# Build CPHF matrix: A_{ai,bj} = 4*(ai|bj) - (ab|ij) - (aj|bi) [GANSU convention]
# M1 = A + diag(eps_a - eps_i)   [GANSU current, factor 1]
# M2 = A + 2*diag(eps_a - eps_i)  [factor 2]
M1 = np.zeros((nvo, nvo))
for a in range(nvir):
    for i in range(nocc):
        ai = a * nocc + i
        for b in range(nvir):
            for j in range(nocc):
                bj = b * nocc + j
                A_val = (4.0 * eri_mo[nocc+a, i, nocc+b, j]
                       - eri_mo[nocc+a, nocc+b, i, j]
                       - eri_mo[nocc+a, j, nocc+b, i])
                M1[ai, bj] = A_val
                if ai == bj:
                    M1[ai, bj] += (eps[nocc+a] - eps[i])

M2 = M1.copy()
for a in range(nvir):
    for i in range(nocc):
        ai = a * nocc + i
        M2[ai, ai] += (eps[nocc+a] - eps[i])  # add second factor

# Test all 4 combinations: {factor 1, factor 2} x {+Xvo, -Xvo}
z_1x_pos = np.linalg.solve(M1, Xvo_2jk)     # GANSU current
z_1x_neg = np.linalg.solve(M1, -Xvo_2jk)    # factor 1, negated RHS
z_2x_pos = np.linalg.solve(M2, Xvo_2jk)     # factor 2, positive RHS (previously tried)
z_2x_neg = np.linalg.solve(M2, -Xvo_2jk)    # factor 2, negated RHS

print(f"\nCPHF solutions (all use (2J-K) Xvo):")
print(f"|z_1x_pos| (factor1, +Xvo) = {np.linalg.norm(z_1x_pos):.12f}  ← GANSU current")
print(f"|z_1x_neg| (factor1, -Xvo) = {np.linalg.norm(z_1x_neg):.12f}")
print(f"|z_2x_pos| (factor2, +Xvo) = {np.linalg.norm(z_2x_pos):.12f}  ← previously tried")
print(f"|z_2x_neg| (factor2, -Xvo) = {np.linalg.norm(z_2x_neg):.12f}")

# Compare each with PySCF's z
z_pyscf_flat = np.zeros(nvo)
for a in range(nvir):
    for i in range(nocc):
        z_pyscf_flat[a * nocc + i] = z_pyscf[a, i]
print(f"\n|z_pyscf| = {np.linalg.norm(z_pyscf_flat):.12f}")

diffs = {
    '1x_pos': np.linalg.norm(z_1x_pos - z_pyscf_flat),
    '1x_neg': np.linalg.norm(z_1x_neg - z_pyscf_flat),
    '2x_pos': np.linalg.norm(z_2x_pos - z_pyscf_flat),
    '2x_neg': np.linalg.norm(z_2x_neg - z_pyscf_flat),
}
print(f"\nDifference from PySCF z:")
for key, val in diffs.items():
    marker = " ★ BEST" if val == min(diffs.values()) else ""
    print(f"  |z_{key} - z_pyscf| = {val:.6e}{marker}")

# Print element-by-element comparison
print(f"\nZ-vector element comparison:")
print(f"  {'(a,i)':>8} {'PySCF':>16} {'1x_pos':>16} {'1x_neg':>16} {'2x_pos':>16} {'2x_neg':>16}")
for a in range(nvir):
    for i in range(nocc):
        ai = a * nocc + i
        print(f"  ({a},{i})    {z_pyscf_flat[ai]:16.12f} {z_1x_pos[ai]:16.12f} {z_1x_neg[ai]:16.12f} {z_2x_pos[ai]:16.12f} {z_2x_neg[ai]:16.12f}")

# -------------------------------------------------------
# 7c. I_mat computation (Lagrangian terms)
# -------------------------------------------------------
print(f"\n{'='*60}")
print("I_mat (Lagrangian terms from 2-RDM)")
print(f"{'='*60}")

# I_oo(i,j) = sum_{akb} T_tilde(i,a,k,b) * (ja|kb)
I_oo = np.zeros((nocc, nocc))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for k in range(nocc):
                for b in range(nvir):
                    I_oo[i, j] += t2tilde_gansu[i, a, k, b] * eri_mo[j, nocc+a, k, nocc+b]

# I_vv(a,b) = sum_{ijc} T_tilde(i,a,j,c) * (ib|jc)
I_vv = np.zeros((nvir, nvir))
for a in range(nvir):
    for b in range(nvir):
        for i in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    I_vv[a, b] += t2tilde_gansu[i, a, j, c] * eri_mo[i, nocc+b, j, nocc+c]

print(f"|I_oo| = {np.linalg.norm(I_oo):.12f}")
print(f"|I_vv| = {np.linalg.norm(I_vv):.12f}")
print(f"I_oo:")
print(I_oo)
print(f"I_vv:")
print(I_vv)

# -------------------------------------------------------
# 8. Build W matrix variants for comparison
# -------------------------------------------------------
print(f"\n{'='*60}")
print("W matrix variants")
print(f"{'='*60}")

# Variant 1: GANSU current (no I_mat)
# W_MO(i,i) = 2*eps_i  (HF)
# W_MO(p,q) += eps_q * dm1mo(p,q)  (MP2 correction)
# Then symmetrize
W_noI_MO = np.zeros((nmo, nmo))
for i in range(nocc):
    W_noI_MO[i, i] = 2.0 * eps[i]
for p in range(nmo):
    for q in range(nmo):
        W_noI_MO[p, q] += eps[q] * dm1_mo_corr[p, q]
# Symmetrize
W_noI_MO = (W_noI_MO + W_noI_MO.T) / 2

W_noI_AO = C @ W_noI_MO @ C.T
print(f"\nW_noI (GANSU current, no I_mat):")
print(f"|W_noI_MO| = {np.linalg.norm(W_noI_MO):.12f}")
print(f"|W_noI_AO| = {np.linalg.norm(W_noI_AO):.12f}")
if nmo <= 10:
    print(f"W_noI_MO:")
    print(W_noI_MO)
    print(f"W_noI_AO:")
    print(W_noI_AO)

# Variant 2: With I_mat
# W_withI_MO = W_noI + I_oo(sym) + I_vv(sym)
W_withI_MO = W_noI_MO.copy()
I_oo_sym = (I_oo + I_oo.T) / 2
I_vv_sym = (I_vv + I_vv.T) / 2
W_withI_MO[:nocc, :nocc] += I_oo_sym
W_withI_MO[nocc:, nocc:] += I_vv_sym

W_withI_AO = C @ W_withI_MO @ C.T
print(f"\nW_withI (with I_oo, I_vv):")
print(f"|W_withI_MO| = {np.linalg.norm(W_withI_MO):.12f}")
print(f"|W_withI_AO| = {np.linalg.norm(W_withI_AO):.12f}")
if nmo <= 10:
    print(f"W_withI_MO:")
    print(W_withI_MO)
    print(f"W_withI_AO:")
    print(W_withI_AO)

# Variant 3: PySCF's formula (eps_p + eps_q) * dm1_total(p,q)
# This is what PySCF might use internally
W_pyscf_MO = np.zeros((nmo, nmo))
for p in range(nmo):
    for q in range(nmo):
        W_pyscf_MO[p, q] = (eps[p] + eps[q]) * dm1_pyscf[p, q]

W_pyscf_AO = C @ W_pyscf_MO @ C.T
print(f"\nW_pyscf_style ((eps_p+eps_q)*dm1_total):")
print(f"|W_pyscf_MO| = {np.linalg.norm(W_pyscf_MO):.12f}")
print(f"|W_pyscf_AO| = {np.linalg.norm(W_pyscf_AO):.12f}")
if nmo <= 10:
    print(f"W_pyscf_MO:")
    print(W_pyscf_MO)

# Compare W variants
print(f"\nW difference norms:")
print(f"|W_noI - W_pyscf_style| = {np.linalg.norm(W_noI_MO - W_pyscf_MO):.12f}")
print(f"|W_withI - W_pyscf_style| = {np.linalg.norm(W_withI_MO - W_pyscf_MO):.12f}")

# Check: is GANSU's W = PySCF's zeta / 2?
# PySCF: zeta(p,q) = (eps_p+eps_q) * dm1_total(p,q)
# GANSU: W(p,q) = 2*eps_i*delta_ij + (eps_p+eps_q)/2 * dm1_corr(p,q)
# Let's compute explicitly:
W_gansu_explicit = np.zeros((nmo, nmo))
for i in range(nocc):
    W_gansu_explicit[i, i] = 2.0 * eps[i]
for p in range(nmo):
    for q in range(nmo):
        W_gansu_explicit[p, q] += (eps[p] + eps[q]) / 2 * dm1_mo_corr[p, q]
print(f"\nW_gansu_explicit = 2*eps_i*delta + (eps_p+eps_q)/2 * dm1_corr:")
print(f"|W_gansu_explicit| = {np.linalg.norm(W_gansu_explicit):.12f}")
# This should match W_noI exactly
print(f"|W_gansu_explicit - W_noI| = {np.linalg.norm(W_gansu_explicit - W_noI_MO):.12f}")

# -------------------------------------------------------
# 8b. PySCF internal W: get the actual energy-weighted density
# -------------------------------------------------------
print(f"\n{'='*60}")
print("PySCF actual energy-weighted density")
print(f"{'='*60}")

# PySCF's zeta: (eps_p + eps_q) * dm1_total(p,q)
dm1_total_mo_pyscf = mymp2.make_rdm1()
zeta_mo = lib.direct_sum('i+j->ij', eps, eps) * dm1_total_mo_pyscf
zeta_ao = C @ zeta_mo @ C.T
print(f"|zeta_MO| (PySCF direct_sum) = {np.linalg.norm(zeta_mo):.12f}")
print(f"|zeta_AO| = {np.linalg.norm(zeta_ao):.12f}")

# Compare with 2*W variants:
print(f"|zeta_MO - 2*W_noI|    = {np.linalg.norm(zeta_mo - 2*W_noI_MO):.12f}")
print(f"|zeta_MO - 2*W_withI|  = {np.linalg.norm(zeta_mo - 2*W_withI_MO):.12f}")
print(f"|zeta_MO - W_pyscf_MO| = {np.linalg.norm(zeta_mo - W_pyscf_MO):.12f}")

if nmo <= 10:
    print(f"\nzeta_MO:")
    print(zeta_mo)

# PySCF's overlap gradient uses:
#   g_S = einsum('xij,ij', s1[:,p0:p1], dme0[p0:p1]) * 2
# where dme0 = make_rdm1e (energy-weighted density, half of zeta)
# So effective W = zeta / 2 = (eps_p+eps_q)/2 * dm1_total
print(f"\nPySCF convention: g_S uses dme0*2 where dme0 = zeta/2")
print(f"So effective W = zeta/2 = (eps_p+eps_q)/2 * dm1_total")
print(f"|zeta/2 - W_noI|  = {np.linalg.norm(zeta_mo/2 - W_noI_MO):.12f}")
print(f"|zeta/2 - W_withI| = {np.linalg.norm(zeta_mo/2 - W_withI_MO):.12f}")

# -------------------------------------------------------
# 9. Analytical MP2 gradient
# -------------------------------------------------------
mp2_g = mymp2.nuc_grad_method()
g_analytical = mp2_g.kernel()
print(f"\nMP2 Analytical Gradient (Hartree/Bohr):")
print(f"{'Atom':>6} {'x':>16} {'y':>16} {'z':>16}")
for i in range(mol.natm):
    print(f"{mol.atom_symbol(i):>6} {g_analytical[i,0]:16.10f} {g_analytical[i,1]:16.10f} {g_analytical[i,2]:16.10f}")

# -------------------------------------------------------
# 10. Numerical gradient (finite difference, gold standard)
# -------------------------------------------------------
print(f"\nComputing numerical MP2 gradient (finite difference)...")
delta = 1e-4  # Bohr
g_numerical = np.zeros((mol.natm, 3))

for iatom in range(mol.natm):
    for idir in range(3):
        mol_p = mol.copy()
        coords_p = mol.atom_coords().copy()
        coords_p[iatom, idir] += delta
        mol_p.set_geom_(coords_p, unit='Bohr')
        mol_p.build()
        mf_p = scf.RHF(mol_p)
        mf_p.conv_tol = 1e-12
        mf_p.verbose = 0
        mf_p.kernel()
        mp2_p = mp.MP2(mf_p)
        mp2_p.kernel()
        e_p = mp2_p.e_tot

        mol_m = mol.copy()
        coords_m = mol.atom_coords().copy()
        coords_m[iatom, idir] -= delta
        mol_m.set_geom_(coords_m, unit='Bohr')
        mol_m.build()
        mf_m = scf.RHF(mol_m)
        mf_m.conv_tol = 1e-12
        mf_m.verbose = 0
        mf_m.kernel()
        mp2_m = mp.MP2(mf_m)
        mp2_m.kernel()
        e_m = mp2_m.e_tot

        g_numerical[iatom, idir] = (e_p - e_m) / (2 * delta)

print(f"\nMP2 Numerical Gradient (Hartree/Bohr):")
print(f"{'Atom':>6} {'x':>16} {'y':>16} {'z':>16}")
for i in range(mol.natm):
    print(f"{mol.atom_symbol(i):>6} {g_numerical[i,0]:16.10f} {g_numerical[i,1]:16.10f} {g_numerical[i,2]:16.10f}")

# -------------------------------------------------------
# 11. Compare analytical vs numerical
# -------------------------------------------------------
diff = g_analytical - g_numerical
print(f"\nAnalytical - Numerical (should be ~0):")
print(f"{'Atom':>6} {'x':>16} {'y':>16} {'z':>16}")
for i in range(mol.natm):
    print(f"{mol.atom_symbol(i):>6} {diff[i,0]:16.10f} {diff[i,1]:16.10f} {diff[i,2]:16.10f}")
print(f"Max |diff| = {np.max(np.abs(diff)):.2e}")

# -------------------------------------------------------
# 12. Gradient component breakdown
# -------------------------------------------------------
print(f"\n{'='*60}")
print(f"Gradient component breakdown")
print(f"{'='*60}")

# HF gradient
hf_g = mf.nuc_grad_method()
g_hf = hf_g.kernel()
print(f"\nHF Gradient (Hartree/Bohr):")
print(f"{'Atom':>6} {'x':>16} {'y':>16} {'z':>16}")
for i in range(mol.natm):
    print(f"{mol.atom_symbol(i):>6} {g_hf[i,0]:16.10f} {g_hf[i,1]:16.10f} {g_hf[i,2]:16.10f}")

g_mp2_corr = g_analytical - g_hf
print(f"\nMP2 correction to gradient:")
print(f"{'Atom':>6} {'x':>16} {'y':>16} {'z':>16}")
for i in range(mol.natm):
    print(f"{mol.atom_symbol(i):>6} {g_mp2_corr[i,0]:16.10f} {g_mp2_corr[i,1]:16.10f} {g_mp2_corr[i,2]:16.10f}")

# -------------------------------------------------------
# 13. AO-basis quantities for GANSU comparison
# -------------------------------------------------------
print(f"\n{'='*60}")
print(f"AO-basis quantities for GANSU comparison")
print(f"{'='*60}")

dm1_total_ao = C @ dm1_pyscf @ C.T
D_hf = mf.make_rdm1()
print(f"\n|P_eff_AO| = {np.linalg.norm(dm1_total_ao):.12f}")
print(f"|D_HF_AO| = {np.linalg.norm(D_hf):.12f}")
print(f"|P_eff - D_HF| = {np.linalg.norm(dm1_total_ao - D_hf):.12f}")

if nmo <= 10:
    print(f"\nP_eff_AO:")
    print(dm1_total_ao)
    print(f"\nW_noI_AO (GANSU style):")
    print(W_noI_AO)
    print(f"\nW_withI_AO:")
    print(W_withI_AO)

# -------------------------------------------------------
# 14. Gamma (2-PDM) construction and comparison
# -------------------------------------------------------
print(f"\n{'='*60}")
print("Gamma (non-separable 2-PDM) construction")
print(f"{'='*60}")

nao = mol.nao_nr()
nao2 = nao * nao

# Build Gamma_T2 in AO: Gamma(mu,nu,la,si) = sum_{ijab} C(mu,i)*C(nu,a+n)*T_tilde(i,a,j,b)*C(la,j)*C(si,b+n)
Gamma_raw = np.zeros((nao, nao, nao, nao))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                t_val = t2tilde_gansu[i, a, j, b]
                if abs(t_val) < 1e-15:
                    continue
                for mu in range(nao):
                    Cmi = C[mu, i]
                    if abs(Cmi) < 1e-15:
                        continue
                    for nu in range(nao):
                        CnA = C[nu, nocc+a]
                        if abs(CnA) < 1e-15:
                            continue
                        for la in range(nao):
                            ClJ = C[la, j]
                            if abs(ClJ) < 1e-15:
                                continue
                            factor = t_val * Cmi * CnA * ClJ
                            for si in range(nao):
                                Gamma_raw[mu, nu, la, si] += factor * C[si, nocc+b]

print(f"|Gamma_raw| = {np.linalg.norm(Gamma_raw):.12f}")

# 4-fold symmetrize
Gamma_sym = np.zeros_like(Gamma_raw)
for mu in range(nao):
    for nu in range(nao):
        for la in range(nao):
            for si in range(nao):
                Gamma_sym[mu,nu,la,si] = 0.25 * (
                    Gamma_raw[mu,nu,la,si] + Gamma_raw[nu,mu,la,si]
                    + Gamma_raw[mu,nu,si,la] + Gamma_raw[nu,mu,si,la])
print(f"|Gamma_sym| (4-fold) = {np.linalg.norm(Gamma_sym):.12f}")

# sep(deltaP)
deltaP = dm1_total_ao - D_hf
sep_deltaP = np.zeros((nao, nao, nao, nao))
for mu in range(nao):
    for nu in range(nao):
        for la in range(nao):
            for si in range(nao):
                sep_deltaP[mu,nu,la,si] = (
                    0.5 * deltaP[mu,nu] * deltaP[la,si]
                    - 0.125 * (deltaP[mu,la] * deltaP[nu,si]
                             + deltaP[mu,si] * deltaP[nu,la]))
print(f"|sep(deltaP)| = {np.linalg.norm(sep_deltaP):.12f}")

Gamma_final = Gamma_sym - sep_deltaP
print(f"|Gamma_final| = {np.linalg.norm(Gamma_final):.12f}")

# -------------------------------------------------------
# 15. Manual gradient computation for each component
# -------------------------------------------------------
print(f"\n{'='*60}")
print("Manual gradient computation (component-by-component)")
print(f"{'='*60}")

# We need derivative integrals. PySCF provides these via grad module.
from pyscf.grad import rhf as rhf_grad_mod

# Nuclear repulsion gradient
g_nuc = rhf_grad_mod.grad_nuc(mol)
print(f"\nNuclear repulsion gradient:")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):>6} {g_nuc[i,0]:16.10f} {g_nuc[i,1]:16.10f} {g_nuc[i,2]:16.10f}")

# 1-electron integral derivatives: h^x
hcore_deriv = mf.nuc_grad_method().hcore_generator(mol)
g_1el = np.zeros((mol.natm, 3))
for iatom in range(mol.natm):
    h1 = hcore_deriv(iatom)  # (3, nao, nao)
    for idir in range(3):
        g_1el[iatom, idir] = np.einsum('ij,ij', h1[idir], dm1_total_ao)

print(f"\n1-electron gradient (h^x . P_eff):")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):>6} {g_1el[i,0]:16.10f} {g_1el[i,1]:16.10f} {g_1el[i,2]:16.10f}")

# Overlap derivative: S^x (for W contribution)
s1 = mf.nuc_grad_method().get_ovlp(mol)  # (3, nao, nao) per atom? No, need per atom
# Actually, s1 is computed differently. Let me use the lower-level interface.
aoslices = mol.aoslice_by_atom()

# For overlap gradient, we need per-atom derivative
g_S_noI = np.zeros((mol.natm, 3))
g_S_withI = np.zeros((mol.natm, 3))
g_S_pyscf_style = np.zeros((mol.natm, 3))

s1_full = -mol.intor('int1e_ipovlp', comp=3)  # (3, nao, nao)
for iatom in range(mol.natm):
    p0, p1 = aoslices[iatom, 2], aoslices[iatom, 3]
    for idir in range(3):
        # Overlap gradient: -tr(S^x * W) = sum_{mu in atom} S^x(mu,nu) * W(mu,nu)
        # s1_full[idir] has derivatives for all basis functions
        # For atom iatom, only basis functions p0:p1 contribute
        # g_S += einsum('ij,ij->', s1[idir, p0:p1, :], W[p0:p1, :]) * 2
        # The factor 2 accounts for mu<->nu symmetry
        g_S_noI[iatom, idir] = np.einsum('j,j', s1_full[idir, p0:p1].ravel(),
                                          W_noI_AO[p0:p1].ravel()) * 2
        g_S_withI[iatom, idir] = np.einsum('j,j', s1_full[idir, p0:p1].ravel(),
                                            W_withI_AO[p0:p1].ravel()) * 2
        # For PySCF-style W (zeta/2):
        W_pyscf_half = zeta_ao / 2
        g_S_pyscf_style[iatom, idir] = np.einsum('j,j', s1_full[idir, p0:p1].ravel(),
                                                   W_pyscf_half[p0:p1].ravel()) * 2

print(f"\nOverlap gradient (S^x . W) contributions:")
print(f"  W_noI:")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_S_noI[i,0]:16.10f} {g_S_noI[i,1]:16.10f} {g_S_noI[i,2]:16.10f}")
print(f"  W_withI:")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_S_withI[i,0]:16.10f} {g_S_withI[i,1]:16.10f} {g_S_withI[i,2]:16.10f}")
print(f"  W_pyscf_style (zeta/2):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_S_pyscf_style[i,0]:16.10f} {g_S_pyscf_style[i,1]:16.10f} {g_S_pyscf_style[i,2]:16.10f}")

# 2-electron integral derivatives
# For the full 2-el gradient, we need (mu nu|la si)^x contracted with the full dm2.
# This is too expensive to compute manually for each atom.
# Instead, let's compute g_total - g_nuc - g_1el - g_S = g_2el (by difference)
g_2el_pyscf = g_analytical - g_nuc - g_1el - g_S_pyscf_style
print(f"\n2-electron gradient (by difference, PySCF W):")
for i in range(mol.natm):
    print(f"  {mol.atom_symbol(i):>6} {g_2el_pyscf[i,0]:16.10f} {g_2el_pyscf[i,1]:16.10f} {g_2el_pyscf[i,2]:16.10f}")

# Also compute what GANSU's total would be with different W choices:
g_total_noI = g_nuc + g_1el + g_S_noI + g_2el_pyscf
g_total_withI = g_nuc + g_1el + g_S_withI + g_2el_pyscf
g_total_pyscf_W = g_nuc + g_1el + g_S_pyscf_style + g_2el_pyscf

print(f"\nTotal gradient with different W choices (assuming same 2-el):")
print(f"  PySCF W: (should match analytical)")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_total_pyscf_W[i,0]:16.10f} {g_total_pyscf_W[i,1]:16.10f} {g_total_pyscf_W[i,2]:16.10f}")
print(f"  W_noI (GANSU current):")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_total_noI[i,0]:16.10f} {g_total_noI[i,1]:16.10f} {g_total_noI[i,2]:16.10f}")
print(f"  W_withI:")
for i in range(mol.natm):
    print(f"    {mol.atom_symbol(i):>6} {g_total_withI[i,0]:16.10f} {g_total_withI[i,1]:16.10f} {g_total_withI[i,2]:16.10f}")

# Difference from analytical
print(f"\n  Diff from analytical (W_noI): max = {np.max(np.abs(g_total_noI - g_analytical)):.6e}")
print(f"  Diff from analytical (W_withI): max = {np.max(np.abs(g_total_withI - g_analytical)):.6e}")
print(f"  Diff from analytical (W_pyscf): max = {np.max(np.abs(g_total_pyscf_W - g_analytical)):.6e}")

# -------------------------------------------------------
# 16. Summary
# -------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"|W_noI - zeta/2|  = {np.linalg.norm(W_noI_MO - zeta_mo/2):.6e}  (GANSU current vs PySCF)")
print(f"|W_withI - zeta/2| = {np.linalg.norm(W_withI_MO - zeta_mo/2):.6e}  (with I_mat vs PySCF)")
print(f"|z_manual(1x) - z_pyscf| = {np.linalg.norm(diff_1x):.6e}  (CPHF factor 1)")
print(f"|z_manual(2x,-) - z_pyscf| = {np.linalg.norm(diff_2x_neg):.6e}  (CPHF factor 2, -RHS)")

print("\n" + "=" * 60)
print("Done. Compare GANSU output with values above.")
print("=" * 60)
