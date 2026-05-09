"""
Analyze ADC(2) Schur complement M12·(ω-D2)⁻¹·M21 decomposition.

Decomposes the full Schur complement into 8 terms from:
  M12: Group A (δ_{IK}) + Group B (δ_{CE})
  M21: 4 terms (δ_{LI}, δ_{LJ}, δ_{FC}, δ_{FD})

Reports the eigenvalue contribution of each term and identifies
which terms are responsible for the SOS vs full ADC(2) gap.

Usage: wsl python3 script/analyze_adc2_schur_terms.py
"""

import numpy as np
from pyscf import gto, scf, adc

# H2O / STO-3G (same geometry as GANSU xyz/H2O.xyz)
mol = gto.M(
    atom='''
    O   0.000000   0.000000   0.117176
    H   0.000000   0.756950  -0.468706
    H   0.000000  -0.756950  -0.468706
    ''',
    basis='sto-3g',
    cart=True,
    unit='Angstrom'
)

mf = scf.RHF(mol).run(verbose=0)
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
ov = nocc * nvir
print(f"nocc={nocc}, nvir={nvir}, nmo={nmo}, ov={ov}")

# MO-basis ERIs: chemist notation (pq|rs)
eri_ao = mol.intor('int2e')
C = mf.mo_coeff
eri_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri_ao, C, C, C, C)

eps = mf.mo_energy
eps_o = eps[:nocc]
eps_v = eps[nocc:]

# ---- Build ADC(2) matrices ----
# D2[I,J,C,D] = eps_C + eps_D - eps_I - eps_J
D2 = np.zeros((nocc, nocc, nvir, nvir))
for I in range(nocc):
    for J in range(nocc):
        for C in range(nvir):
            for D in range(nvir):
                D2[I,J,C,D] = eps_v[C] + eps_v[D] - eps_o[I] - eps_o[J]

# VVOV: (EC|JD) = eri_mo[nocc+E, nocc+C, J, nocc+D]
def vvov(E,C,J,D):
    return eri_mo[nocc+E, nocc+C, J, nocc+D]

# OOOV: (JK|ID) = eri_mo[J, K, I, nocc+D]
def ooov(J,K,I,D):
    return eri_mo[J, K, I, nocc+D]

# Build M12 [ov x dd] and M21 [dd x ov]
dd = nocc * nocc * nvir * nvir
M12 = np.zeros((ov, dd))
M21 = np.zeros((dd, ov))

for ke_idx in range(ov):
    K = ke_idx // nvir
    E = ke_idx % nvir
    for I in range(nocc):
        for J in range(nocc):
            for C_ in range(nvir):
                for D_ in range(nvir):
                    dd_idx = I*(nocc*nvir*nvir) + J*(nvir*nvir) + C_*nvir + D_
                    val12 = 0.0
                    if I == K:
                        val12 += 2.0*vvov(E,C_,J,D_) - vvov(D_,E,J,C_)
                    if C_ == E:
                        val12 += ooov(J,K,I,D_) - 2.0*ooov(I,K,J,D_)
                    M12[ke_idx, dd_idx] = val12

                    val21 = 0.0
                    if K == I:
                        val21 += vvov(E,C_,J,D_)
                    if K == J:
                        val21 += vvov(E,D_,I,C_)
                    if E == C_:
                        val21 -= ooov(I,K,J,D_)
                    if E == D_:
                        val21 -= ooov(J,K,I,C_)
                    M21[dd_idx, ke_idx] = val21

# ---- PySCF ADC(2) reference ----
myadc = adc.ADC(mf)
myadc.method_type = "ee"
evals_pyscf, evecs_pyscf, _, _ = myadc.kernel(nroots=3)
print("\nPySCF ADC(2) excitation energies:")
for i, e in enumerate(evals_pyscf):
    print(f"  State {i+1}: {e:.8f} Ha = {e*27.2114:.4f} eV")

# ---- Analyze Schur complement terms ----
omega = evals_pyscf[0]  # Use first excitation energy as omega
print(f"\nAnalyzing Schur complement at omega = {omega:.8f} Ha")

# Full Schur complement
D2_flat = D2.reshape(-1)
inv_denom = 1.0 / (omega - D2_flat)
Schur_full = M12 @ np.diag(inv_denom) @ M21

# Decompose M12 and M21 into sub-matrices
# M12 = M12_A (delta_IK, vvov terms) + M12_B (delta_CE, ooov terms)
M12_A = np.zeros_like(M12)  # delta_{IK} terms
M12_B = np.zeros_like(M12)  # delta_{CE} terms

for ke_idx in range(ov):
    K = ke_idx // nvir
    E = ke_idx % nvir
    for I in range(nocc):
        for J in range(nocc):
            for C_ in range(nvir):
                for D_ in range(nvir):
                    dd_idx = I*(nocc*nvir*nvir) + J*(nvir*nvir) + C_*nvir + D_
                    if I == K:
                        M12_A[ke_idx, dd_idx] = 2.0*vvov(E,C_,J,D_) - vvov(D_,E,J,C_)
                    if C_ == E:
                        M12_B[ke_idx, dd_idx] = ooov(J,K,I,D_) - 2.0*ooov(I,K,J,D_)

# M21 = M21_1 (delta_LI) + M21_2 (delta_LJ) + M21_3 (delta_FC) + M21_4 (delta_FD)
M21_1 = np.zeros_like(M21)  # delta_{LI}, vvov
M21_2 = np.zeros_like(M21)  # delta_{LJ}, vvov
M21_3 = np.zeros_like(M21)  # delta_{FC}, ooov
M21_4 = np.zeros_like(M21)  # delta_{FD}, ooov

for lf_idx in range(ov):
    L = lf_idx // nvir
    F = lf_idx % nvir
    for I in range(nocc):
        for J in range(nocc):
            for C_ in range(nvir):
                for D_ in range(nvir):
                    dd_idx = I*(nocc*nvir*nvir) + J*(nvir*nvir) + C_*nvir + D_
                    if L == I:
                        M21_1[dd_idx, lf_idx] = vvov(F,C_,J,D_)
                    if L == J:
                        M21_2[dd_idx, lf_idx] = vvov(F,D_,I,C_)
                    if F == C_:
                        M21_3[dd_idx, lf_idx] = -ooov(I,L,J,D_)
                    if F == D_:
                        M21_4[dd_idx, lf_idx] = -ooov(J,L,I,C_)

# Verify decomposition
assert np.allclose(M12, M12_A + M12_B), "M12 decomposition error"
assert np.allclose(M21, M21_1 + M21_2 + M21_3 + M21_4), "M21 decomposition error"

# Compute 8 cross-terms
D_inv = np.diag(inv_denom)
labels = []
schur_terms = []

for m12_label, m12_mat in [("A(dIK,vvov)", M12_A), ("B(dCE,ooov)", M12_B)]:
    for m21_label, m21_mat in [("1(dLI,vvov)", M21_1), ("2(dLJ,vvov)", M21_2),
                                ("3(dFC,ooov)", M21_3), ("4(dFD,ooov)", M21_4)]:
        term = m12_mat @ D_inv @ m21_mat
        labels.append(f"{m12_label} x {m21_label}")
        schur_terms.append(term)

# Verify sum of 8 terms = full Schur
Schur_sum = sum(schur_terms)
print(f"  ||Schur_full - sum(8 terms)||_F = {np.linalg.norm(Schur_full - Schur_sum):.2e}")

# Eigenvalue analysis: contribution of each term to lowest eigenvalue
# Build M11 (from PySCF or from our construction)
# For simplicity, compute M_eff = M11 + Schur and extract eigenvalues

# Build M11 from eri_mo
M11 = np.zeros((ov, ov))
# D1 + 2(ia|jb) - (ij|ab)
for ia in range(ov):
    i = ia // nvir
    a = ia % nvir
    for jb in range(ov):
        j = jb // nvir
        b = jb % nvir
        val = 0.0
        if ia == jb:
            val += eps_v[a] - eps_o[i]
        val += 2.0 * eri_mo[i, nocc+a, j, nocc+b]  # 2(ia|jb)
        val -= eri_mo[i, j, nocc+a, nocc+b]          # -(ij|ab)
        M11[ia, jb] = val

# Add ISR correction (2nd order)
t2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                t2[i,j,a,b] = eri_mo[i,nocc+a,j,nocc+b] / (eps_o[i]+eps_o[j]-eps_v[a]-eps_v[b])

# ISR + Sigma_oo + Sigma_vv (match GANSU's ADC2Operator::build_M11)
for ia in range(ov):
    i = ia // nvir
    a = ia % nvir
    for jb in range(ov):
        j = jb // nvir
        b = jb % nvir
        isr = 0.0
        for k in range(nocc):
            for c in range(nvir):
                t2_kiac = t2[k,i,a,c]
                t2_ikac = t2[i,k,a,c]
                kb_jc = eri_mo[k,nocc+b,j,nocc+c]
                jb_kc = eri_mo[j,nocc+b,k,nocc+c]
                isr += 0.5*t2_kiac*kb_jc + 2.0*t2_ikac*jb_kc - t2_ikac*kb_jc - t2_kiac*jb_kc
                # Transpose
                t2_kjbc = t2[k,j,b,c]
                t2_jkbc = t2[j,k,b,c]
                ka_ic = eri_mo[k,nocc+a,i,nocc+c]
                ia_kc = eri_mo[i,nocc+a,k,nocc+c]
                isr += 0.5*t2_kjbc*ka_ic + 2.0*t2_jkbc*ia_kc - t2_jkbc*ka_ic - t2_kjbc*ia_kc
        M11[ia,jb] += isr

# Sigma_oo
sigma_oo = np.zeros((nocc, nocc))
for i in range(nocc):
    for j in range(nocc):
        for k in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2_ikab = t2[i,k,a,b]
                    t2_jkab = t2[j,k,a,b]
                    ja_kb = eri_mo[j,nocc+a,k,nocc+b]
                    jb_ka = eri_mo[j,nocc+b,k,nocc+a]
                    ia_kb = eri_mo[i,nocc+a,k,nocc+b]
                    ib_ka = eri_mo[i,nocc+b,k,nocc+a]
                    sigma_oo[i,j] += t2_ikab*(ja_kb-0.5*jb_ka) + t2_jkab*(ia_kb-0.5*ib_ka)

sigma_vv = np.zeros((nvir, nvir))
for a in range(nvir):
    for b in range(nvir):
        for i in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    t2_ijac = t2[i,j,a,c]
                    t2_ijbc = t2[i,j,b,c]
                    ib_jc = eri_mo[i,nocc+b,j,nocc+c]
                    jb_ic = eri_mo[j,nocc+b,i,nocc+c]
                    ia_jc = eri_mo[i,nocc+a,j,nocc+c]
                    ja_ic = eri_mo[j,nocc+a,i,nocc+c]
                    sigma_vv[a,b] -= t2_ijac*(ib_jc-0.5*jb_ic) + t2_ijbc*(ia_jc-0.5*ja_ic)

for ia in range(ov):
    i = ia // nvir
    a = ia % nvir
    for jb in range(ov):
        j = jb // nvir
        b = jb % nvir
        if a == b:
            M11[ia,jb] -= sigma_oo[i,j]
        if i == j:
            M11[ia,jb] += sigma_vv[a,b]

# Eigenvalues of M_eff = M11 + Schur_full
M_eff = M11 + Schur_full
evals_full = np.sort(np.linalg.eigvalsh(M_eff))

print(f"\n  M_eff eigenvalues (omega={omega:.6f}):")
for k in range(min(3, len(evals_full))):
    print(f"    State {k+1}: {evals_full[k]:.8f} Ha = {evals_full[k]*27.2114:.4f} eV")

# Now analyze: eigenvalue shift from each Schur term
print("\n  ===== Schur Complement Term Analysis =====")
print(f"  {'Term':<30s}  {'trace':>12s}  {'||F||':>10s}  {'eval_1 shift':>14s}")
print(f"  {'-'*30}  {'-'*12}  {'-'*10}  {'-'*14}")

for i, (label, term) in enumerate(zip(labels, schur_terms)):
    M_eff_i = M11 + term
    evals_i = np.sort(np.linalg.eigvalsh(M_eff_i))
    shift = evals_i[0] - np.sort(np.linalg.eigvalsh(M11))[0]
    print(f"  {label:<30s}  {np.trace(term):>12.6f}  {np.linalg.norm(term):>10.4f}  {shift*27.2114:>+12.4f} eV")

# Current SOS approximation: (ia|jb)^2 / (omega - D2)
# This is: Schur_SOS[ia,jb] = sum_{kc} (ia|kc)(kc|jb)/(omega-D2_{ijkc})??
# No: current code computes (ia|jb)^2 / (omega - D2_{ijab})
Schur_SOS = np.zeros((ov, ov))
for ia in range(ov):
    i = ia // nvir
    a = ia % nvir
    for jb in range(ov):
        j = jb // nvir
        b = jb % nvir
        iajb = eri_mo[i, nocc+a, j, nocc+b]
        denom = omega - (eps_v[a] + eps_v[b] - eps_o[i] - eps_o[j])
        Schur_SOS[ia,jb] = iajb * iajb / denom

M_eff_sos = M11 + 1.3 * Schur_SOS  # c_os = 1.3
evals_sos = np.sort(np.linalg.eigvalsh(M_eff_sos))
M_eff_sos_1 = M11 + 1.0 * Schur_SOS  # c_os = 1.0
evals_sos_1 = np.sort(np.linalg.eigvalsh(M_eff_sos_1))

print(f"\n  ===== Comparison =====")
print(f"  {'Method':<35s}  {'State 1 [eV]':>14s}  {'State 2 [eV]':>14s}  {'State 3 [eV]':>14s}")
print(f"  {'-'*35}  {'-'*14}  {'-'*14}  {'-'*14}")

evals_m11 = np.sort(np.linalg.eigvalsh(M11))
print(f"  {'M11 only (CIS+ISR)':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_m11[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'M11 + Full Schur (ADC(2))':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_full[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'M11 + (ia|jb)^2 SOS c_os=1.0':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_sos_1[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'M11 + (ia|jb)^2 SOS c_os=1.3':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_sos[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'PySCF ADC(2)':35s}", end="")
for k in range(min(3, len(evals_pyscf))):
    print(f"  {evals_pyscf[k]*27.2114:>14.4f}", end="")
print()

# Identify which M12 group dominates the missing contribution
Schur_A = sum(schur_terms[:4])  # Group A terms
Schur_B = sum(schur_terms[4:])  # Group B terms
M_eff_A = M11 + Schur_A
M_eff_B = M11 + Schur_B
evals_A = np.sort(np.linalg.eigvalsh(M_eff_A))
evals_B = np.sort(np.linalg.eigvalsh(M_eff_B))

print(f"\n  {'M11 + Group A (dIK, vvov M12)':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_A[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'M11 + Group B (dCE, ooov M12)':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_B[k]*27.2114:>14.4f}", end="")
print()

# vvov-only vs ooov-only in M21
Schur_vvov_m21 = M12 @ D_inv @ (M21_1 + M21_2)  # vvov M21 terms
Schur_ooov_m21 = M12 @ D_inv @ (M21_3 + M21_4)  # ooov M21 terms
M_eff_vvov_m21 = M11 + Schur_vvov_m21
M_eff_ooov_m21 = M11 + Schur_ooov_m21
evals_vvov_m21 = np.sort(np.linalg.eigvalsh(M_eff_vvov_m21))
evals_ooov_m21 = np.sort(np.linalg.eigvalsh(M_eff_ooov_m21))

print(f"\n  {'M11 + M12_full * D * M21_vvov':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_vvov_m21[k]*27.2114:>14.4f}", end="")
print()

print(f"  {'M11 + M12_full * D * M21_ooov':35s}", end="")
for k in range(min(3, ov)):
    print(f"  {evals_ooov_m21[k]*27.2114:>14.4f}", end="")
print()
