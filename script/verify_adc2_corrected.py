"""
Test corrected ADC(2) Schur complement formulas.

Three approaches compared:
A) GANSU's current M12/M21 (known wrong)
B) Corrected M12 from derivation (÷2), M21 = M12^T (Hermitian)
C) Direct Schur complement from ADC2_SCHUR_RHF.md einsum formulas

Usage: python verify_adc2_corrected.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, tdscf

mol = gto.M(
    atom='''
    O   0.000   0.000   0.117
    H   0.000   0.757  -0.470
    H   0.000  -0.757  -0.470
    ''',
    basis='cc-pvdz',
    unit='Angstrom',
    cart=True
)

mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}")

C = mf.mo_coeff
eps = mf.mo_energy

# MO ERIs
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)
eri_vvov = eri_mo[nocc:, nocc:, :nocc, nocc:]  # (ab|ic)
eri_ooov = eri_mo[:nocc, :nocc, :nocc, nocc:]  # (ji|kb)

# CIS matrix
CIS = np.zeros((ov, ov))
for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b
                val = 0.0
                if i == j and a == b:
                    val += eps[a + nocc] - eps[i]
                val += 2.0 * eri_ovov[i, a, j, b]
                val -= eri_mo[i, j, a + nocc, b + nocc]
                CIS[ia, jb] = val

# T2 and D2
t2 = np.zeros((nocc, nocc, nvir, nvir))
D2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                denom = eps[i] + eps[j] - eps[a + nocc] - eps[b + nocc]
                t2[i, j, a, b] = eri_ovov[i, a, j, b] / denom
                D2[i, j, a, b] = eps[a + nocc] + eps[b + nocc] - eps[i] - eps[j]

# ISR correction
ISR = np.zeros((ov, ov))
for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b
                val = 0.0
                for k in range(nocc):
                    for c in range(nvir):
                        t2_ikac = t2[i, k, a, c]
                        t2_kiac = t2[k, i, a, c]
                        jbkc = eri_ovov[j, b, k, c]
                        kbjc = eri_ovov[k, b, j, c]
                        val += 2.0 * t2_ikac * jbkc - t2_ikac * kbjc
                        val -= t2_kiac * jbkc - 0.5 * t2_kiac * kbjc
                        t2_jkbc = t2[j, k, b, c]
                        t2_kjbc = t2[k, j, b, c]
                        iakc = eri_ovov[i, a, k, c]
                        kaic = eri_ovov[k, a, i, c]
                        val += 2.0 * t2_jkbc * iakc - t2_jkbc * kaic
                        val -= t2_kjbc * iakc - 0.5 * t2_kjbc * kaic
                ISR[ia, jb] = val

# Self-energy
sigma_oo = np.zeros((nocc, nocc))
for i in range(nocc):
    for j in range(nocc):
        val_ij = 0.0
        val_ji = 0.0
        for k in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    val_ij += t2[i, k, a, b] * (eri_ovov[j, a, k, b] - 0.5 * eri_ovov[j, b, k, a])
                    val_ji += t2[j, k, a, b] * (eri_ovov[i, a, k, b] - 0.5 * eri_ovov[i, b, k, a])
        sigma_oo[i, j] = val_ij + val_ji

sigma_vv = np.zeros((nvir, nvir))
for a in range(nvir):
    for b in range(nvir):
        val_ab = 0.0
        val_ba = 0.0
        for i in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    val_ab += t2[i, j, a, c] * (-eri_ovov[i, b, j, c] + 0.5 * eri_ovov[j, b, i, c])
                    val_ba += t2[i, j, b, c] * (-eri_ovov[i, a, j, c] + 0.5 * eri_ovov[j, a, i, c])
        sigma_vv[a, b] = val_ab + val_ba

# M11
M11 = CIS.copy() + ISR
for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b
                if a == b:
                    M11[ia, jb] -= sigma_oo[i, j]
                if i == j:
                    M11[ia, jb] += sigma_vv[a, b]

print(f"M11 eigenvalues: {np.linalg.eigvalsh(M11)[:5]}")

# ======================================================================
# Approach A: GANSU's current M12/M21 (for reference)
# ======================================================================
print("\n=== Approach A: GANSU formulas ===")
M12_gansu = np.zeros((ov, dd))
M21_gansu = np.zeros((dd, ov))
for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for C_ in range(nvir):
                    for D in range(nvir):
                        ijcd = I * nocc * nvir * nvir + J * nvir * nvir + C_ * nvir + D
                        # M12
                        val12 = 0.0
                        if I == K:
                            val12 += 2.0 * eri_vvov[E, C_, J, D] - eri_vvov[D, E, J, C_]
                        if C_ == E:
                            val12 += eri_ooov[J, K, I, D] - 2.0 * eri_ooov[I, K, J, D]
                        M12_gansu[ke, ijcd] = val12
                        # M21
                        val21 = 0.0
                        if K == I:
                            val21 += eri_vvov[E, C_, J, D]
                        if K == J:
                            val21 += eri_vvov[E, D, I, C_]
                        if E == C_:
                            val21 -= eri_ooov[I, K, J, D]
                        if E == D:
                            val21 -= eri_ooov[J, K, I, C_]
                        M21_gansu[ijcd, ke] = val21

D2_flat = D2.reshape(-1)

def solve_schur(M11, M12, M21, D2_flat, n_states=5, label=""):
    def build_M_eff(omega):
        inv_denom = 1.0 / (omega - D2_flat)
        return M11 + M12 @ (M21 * inv_denom[:, np.newaxis])

    energies = np.zeros(n_states)
    for k in range(n_states):
        omega = 0.0
        for it in range(30):
            M_eff = build_M_eff(omega)
            evals = np.linalg.eigvalsh(M_eff)
            omega_new = evals[k]
            if abs(omega_new - omega) < 1e-8:
                energies[k] = omega_new
                break
            omega = omega_new
        else:
            energies[k] = omega_new
    return energies

e_gansu = solve_schur(M11, M12_gansu, M21_gansu, D2_flat, label="GANSU")
print(f"GANSU eigenvalues:     {e_gansu}")

# ======================================================================
# Approach B: Corrected M12 from derivation/2, M21 = M12^T
# ======================================================================
print("\n=== Approach B: M12 from derivation/2, M21=M12^T ===")
M12_corr = np.zeros((ov, dd))
for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for C_ in range(nvir):
                    for D in range(nvir):
                        ijcd = I * nocc * nvir * nvir + J * nvir * nvir + C_ * nvir + D
                        val = 0.0
                        if I == K:  # delta_{I,K}
                            # From derivation: 4(EC|JD) - 4(ED|JC), divided by 2
                            val += 2.0 * eri_vvov[E, C_, J, D] - 2.0 * eri_vvov[E, D, J, C_]
                        if C_ == E:  # delta_{C,E}
                            # From derivation: -4*ooov[I,K,J,D] + 4*ooov[J,K,I,D], divided by 2
                            val += -2.0 * eri_ooov[I, K, J, D] + 2.0 * eri_ooov[J, K, I, D]
                        M12_corr[ke, ijcd] = val

M21_corr = M12_corr.T  # Hermitian: M21 = M12^T
print(f"M12_corr == M21_corr^T: {np.allclose(M12_corr, M21_corr.T)}")

e_corr = solve_schur(M11, M12_corr, M21_corr, D2_flat, label="Corrected")
print(f"Corrected eigenvalues: {e_corr}")

# ======================================================================
# Approach C: Direct Schur complement from ADC2_SCHUR_RHF.md einsums
# ======================================================================
print("\n=== Approach C: Direct Schur complement (einsum) ===")

def build_M_eff_direct(omega, M11):
    """Build M_eff using direct einsum formulas from derivation."""
    # Dw[a,b,i,j] = 1/(omega - (eps_a + eps_b - eps_i - eps_j))
    Dw = np.zeros((nvir, nvir, nocc, nocc))
    for a in range(nvir):
        for b in range(nvir):
            for i in range(nocc):
                for j in range(nocc):
                    Dw[a, b, i, j] = 1.0 / (omega - eps[a+nocc] - eps[b+nocc] + eps[i] + eps[j])

    # The Schur complement [M_eff - M11] as a matrix
    # Set r1 = e_{jb} and compute sigma for each (j,b)
    Schur = np.zeros((ov, ov))

    for j0 in range(nocc):
        for b0 in range(nvir):
            jb = j0 * nvir + b0
            # r1[K,E] = delta_{K,j0} delta_{E,b0}

            sigma = np.zeros(ov)  # sigma[a*nocc + i] or sigma[i*nvir + a]

            # Einsum 1: "iajb,bj,abij->ai", coeff 4
            # sigma[a,i] += 4 * ovov[i,a,j0,b0] * Dw[a,b0,i,j0]
            for i in range(nocc):
                for a in range(nvir):
                    sigma[i * nvir + a] += 4.0 * eri_ovov[i, a, j0, b0] * Dw[a, b0, i, j0]

            # Einsum 2: "ibja,bj,abij->ai", coeff -2
            # sigma[a,i] += -2 * ovov[i,b0,j0,a] * Dw[a,b0,i,j0]
            for i in range(nocc):
                for a in range(nvir):
                    sigma[i * nvir + a] += -2.0 * eri_ovov[i, b0, j0, a] * Dw[a, b0, i, j0]

            # Einsum 3: "kajb,ak,abij->bj", coeff -8
            # sigma[B,J] += -8 * sum_{k,a',i'} ovov[k,a',J,B] * r1[a',k] * Dw[a',B,i',J]
            # With r1 = e_{j0,b0}: r1[a',k] = delta_{k,j0} delta_{a',b0}
            # sigma[B,J] += -8 * sum_{i'} ovov[j0,b0,J,B] * Dw[b0,B,i',J]
            for J in range(nocc):
                for B in range(nvir):
                    s = 0.0
                    for ip in range(nocc):
                        s += Dw[b0, B, ip, J]
                    sigma[J * nvir + B] += -8.0 * eri_ovov[j0, b0, J, B] * s

            # Einsum 4: "kbja,ak,abij->bj", coeff 4
            # sigma[B,J] += 4 * sum_{k,a',i'} ovov[k,B,J,a'] * r1[a',k] * Dw[a',B,i',J]
            # = 4 * sum_{i'} ovov[j0,B,J,b0] * Dw[b0,B,i',J]
            for J in range(nocc):
                for B in range(nvir):
                    s = 0.0
                    for ip in range(nocc):
                        s += Dw[b0, B, ip, J]
                    sigma[J * nvir + B] += 4.0 * eri_ovov[j0, B, J, b0] * s

            Schur[:, jb] = sigma

    # Divide by N_singles = 2 to get the correct normalization
    Schur /= 2.0
    return M11 + Schur

# Solve with omega iteration using direct Schur complement
energies_direct = np.zeros(5)
for k in range(5):
    omega = 0.0
    for it in range(30):
        M_eff = build_M_eff_direct(omega, M11)
        # Symmetrize (should already be symmetric or nearly so)
        M_eff = 0.5 * (M_eff + M_eff.T)
        evals = np.linalg.eigvalsh(M_eff)
        omega_new = evals[k]
        if abs(omega_new - omega) < 1e-8:
            energies_direct[k] = omega_new
            break
        omega = omega_new
    else:
        energies_direct[k] = omega_new

print(f"Direct eigenvalues:    {energies_direct}")

# Check symmetry of direct Schur complement at omega=0
M_eff_test = build_M_eff_direct(0.0, M11)
print(f"Direct M_eff symmetric: {np.allclose(M_eff_test, M_eff_test.T, atol=1e-10)}")
print(f"Max asymmetry: {np.max(np.abs(M_eff_test - M_eff_test.T)):.6e}")

# ======================================================================
# Compare all approaches with PySCF
# ======================================================================
print("\n" + "=" * 70)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_pyscf = result[0]

print(f"\n{'State':>5}  {'GANSU':>12}  {'Corrected':>12}  {'Direct':>12}  {'PySCF':>12}")
print("-" * 65)
for i in range(5):
    print(f"{i+1:5d}  {e_gansu[i]:12.8f}  {e_corr[i]:12.8f}  {energies_direct[i]:12.8f}  {e_pyscf[i]:12.8f}")

print(f"\nMax |Corrected - PySCF|: {np.max(np.abs(e_corr - e_pyscf)):.6e}")
print(f"Max |Direct - PySCF|:    {np.max(np.abs(energies_direct - e_pyscf)):.6e}")
print(f"Max |GANSU - PySCF|:     {np.max(np.abs(e_gansu - e_pyscf)):.6e}")
