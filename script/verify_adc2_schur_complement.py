"""
Verify ADC(2) Schur complement formulas against PySCF.
Implements GANSU's exact M11/M12/M21/D2 formulas in Python
to isolate formula errors from CUDA implementation bugs.

Usage: python verify_adc2_schur_complement.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo

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
print(f"RHF energy: {mf.e_tot:.10f}")

nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
print(f"nocc={nocc}, nvir={nvir}, nao={nao}, ov={ov}, dd={dd}")

# MO coefficients and orbital energies
C = mf.mo_coeff
eps = mf.mo_energy
print(f"Orbital energies: {eps}")

# Transform ERIs to MO basis: (pq|rs) chemist notation
# Full N^4 tensor
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)
print(f"MO ERI tensor shape: {eri_mo.shape}")

# Extract ERI blocks
# ovov[i,a,j,b] = (ia|jb) where a,b are relative to nocc
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]
print(f"eri_ovov shape: {eri_ovov.shape}")  # (nocc, nvir, nocc, nvir)

# vvov[a,b,i,c] = (ab|ic)
eri_vvov = eri_mo[nocc:, nocc:, :nocc, nocc:]
print(f"eri_vvov shape: {eri_vvov.shape}")  # (nvir, nvir, nocc, nvir)

# ooov[j,i,k,b] = (ji|kb)
eri_ooov = eri_mo[:nocc, :nocc, :nocc, nocc:]
print(f"eri_ooov shape: {eri_ooov.shape}")  # (nocc, nocc, nocc, nvir)

# ======================================================================
# 1. CIS matrix: A[ia,jb] = δ_ij·δ_ab·(eps_a - eps_i) + 2(ia|jb) - (ij|ab)
# ======================================================================
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
                val += 2.0 * eri_ovov[i, a, j, b]  # 2(ia|jb)
                val -= eri_mo[i, j, a + nocc, b + nocc]  # -(ij|ab)
                CIS[ia, jb] = val

cis_evals = np.linalg.eigvalsh(CIS)
print(f"\nCIS eigenvalues: {cis_evals[:10]}")

# Compare with PySCF TDA
from pyscf import tdscf
td = tdscf.TDA(mf)
td.nstates = 10
td.run()
print(f"PySCF TDA:        {td.e[:10]}")
print(f"CIS - TDA diff:   {np.abs(cis_evals[:5] - td.e[:5])}")

# ======================================================================
# 2. MP1 T2 amplitudes: t2[i,j,a,b] = (ia|jb) / (eps_i + eps_j - eps_a - eps_b)
# ======================================================================
t2 = np.zeros((nocc, nocc, nvir, nvir))
D2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                denom = eps[i] + eps[j] - eps[a + nocc] - eps[b + nocc]
                t2[i, j, a, b] = eri_ovov[i, a, j, b] / denom
                D2[i, j, a, b] = eps[a + nocc] + eps[b + nocc] - eps[i] - eps[j]

# MP2 energy
e_mp2 = 0.0
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                e_mp2 += t2[i, j, a, b] * (2.0 * eri_ovov[i, a, j, b] - eri_ovov[i, b, j, a])
print(f"\nMP2 correlation energy: {e_mp2:.10f}")
print(f"PySCF MP2 ref:         -0.2075782409")

# ======================================================================
# 3. ISR correction to M11
# ISR_corr[ia,jb] = (2P1 - P2 - P5 + 0.5*P6) + transpose
# ======================================================================
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
                        ovov_jbkc = eri_ovov[j, b, k, c]
                        ovov_kbjc = eri_ovov[k, b, j, c]
                        # Direct: 2P1 - P2 - P5 + 0.5P6
                        val += 2.0 * t2_ikac * ovov_jbkc
                        val -= t2_ikac * ovov_kbjc
                        val -= t2_kiac * ovov_jbkc
                        val += 0.5 * t2_kiac * ovov_kbjc
                        # Transpose: (i,a) <-> (j,b)
                        t2_jkbc = t2[j, k, b, c]
                        t2_kjbc = t2[k, j, b, c]
                        ovov_iakc = eri_ovov[i, a, k, c]
                        ovov_kaic = eri_ovov[k, a, i, c]
                        val += 2.0 * t2_jkbc * ovov_iakc
                        val -= t2_jkbc * ovov_kaic
                        val -= t2_kjbc * ovov_iakc
                        val += 0.5 * t2_kjbc * ovov_kaic
                ISR[ia, jb] = val

print(f"\nISR correction: norm={np.linalg.norm(ISR):.6f}, max={np.max(np.abs(ISR)):.6f}")
print(f"ISR symmetric: {np.allclose(ISR, ISR.T)}")

# ======================================================================
# 4. Self-energy corrections Sigma_oo and Sigma_vv
# ======================================================================
# Sigma_oo[i,j] = sum_{k,a,b} t2[i,k,a,b] * (ovov[j,a,k,b] - 0.5*ovov[j,b,k,a])
# Symmetrized: Sigma_oo[i,j] = raw[i,j] + raw[j,i] (no 0.5 factor)
sigma_oo_raw = np.zeros((nocc, nocc))
for i in range(nocc):
    for j in range(nocc):
        val = 0.0
        for k in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    val += t2[i, k, a, b] * (eri_ovov[j, a, k, b] - 0.5 * eri_ovov[j, b, k, a])
        sigma_oo_raw[i, j] = val
sigma_oo = sigma_oo_raw + sigma_oo_raw.T
print(f"\nSigma_oo: norm={np.linalg.norm(sigma_oo):.6f}")

# Sigma_vv[a,b] = sum_{i,j,c} t2[i,j,a,c] * (-ovov[i,b,j,c] + 0.5*ovov[j,b,i,c])
sigma_vv_raw = np.zeros((nvir, nvir))
for a in range(nvir):
    for b in range(nvir):
        val = 0.0
        for i in range(nocc):
            for j in range(nocc):
                for c in range(nvir):
                    val += t2[i, j, a, c] * (-eri_ovov[i, b, j, c] + 0.5 * eri_ovov[j, b, i, c])
        sigma_vv_raw[a, b] = val
sigma_vv = sigma_vv_raw + sigma_vv_raw.T
print(f"Sigma_vv: norm={np.linalg.norm(sigma_vv):.6f}")

# ======================================================================
# 5. Build M11 = CIS + ISR + self-energy
# ======================================================================
M11 = CIS.copy()
M11 += ISR
# Add self-energy: M11[ia,jb] += -delta_ab * sigma_oo[i,j] + delta_ij * sigma_vv[a,b]
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

m11_evals = np.linalg.eigvalsh(M11)
print(f"\nM11 eigenvalues: {m11_evals[:10]}")
print(f"M11 symmetric: {np.allclose(M11, M11.T)}")

# ======================================================================
# 6. Build M12 and M21 using GANSU's formulas
# ======================================================================
print("\nBuilding M12 and M21 (GANSU formulas)...")

# M12[KE, IJCD] = delta_{I,K}*[2*(EC|JD) - (DE|JC)] + delta_{C,E}*[(JK|ID) - 2*(IK|JD)]
M12 = np.zeros((ov, dd))
for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for C_ in range(nvir):
                    for D in range(nvir):
                        ijcd = I * nocc * nvir * nvir + J * nvir * nvir + C_ * nvir + D
                        val = 0.0
                        if I == K:
                            EC_JD = eri_vvov[E, C_, J, D]
                            DE_JC = eri_vvov[D, E, J, C_]
                            val += 2.0 * EC_JD - DE_JC
                        if C_ == E:
                            JK_ID = eri_ooov[J, K, I, D]
                            IK_JD = eri_ooov[I, K, J, D]
                            val += JK_ID - 2.0 * IK_JD
                        M12[ke, ijcd] = val

# M21[IJCD, KE] = delta_{K,I}*(EC|JD) + delta_{K,J}*(ED|IC) - delta_{E,C}*(IK|JD) - delta_{E,D}*(JK|IC)
M21 = np.zeros((dd, ov))
for I in range(nocc):
    for J in range(nocc):
        for C_ in range(nvir):
            for D in range(nvir):
                ijcd = I * nocc * nvir * nvir + J * nvir * nvir + C_ * nvir + D
                for K in range(nocc):
                    for E in range(nvir):
                        ke = K * nvir + E
                        val = 0.0
                        if K == I:
                            val += eri_vvov[E, C_, J, D]
                        if K == J:
                            val += eri_vvov[E, D, I, C_]
                        if E == C_:
                            val -= eri_ooov[I, K, J, D]
                        if E == D:
                            val -= eri_ooov[J, K, I, C_]
                        M21[ijcd, ke] = val

print(f"M12 shape: {M12.shape}, norm: {np.linalg.norm(M12):.6f}")
print(f"M21 shape: {M21.shape}, norm: {np.linalg.norm(M21):.6f}")
print(f"M12 == M21^T: {np.allclose(M12, M21.T)}")
print(f"Max |M12 - M21^T|: {np.max(np.abs(M12 - M21.T)):.6e}")

# ======================================================================
# 7. Build M_eff(omega) and solve with omega iteration
# ======================================================================
D2_flat = D2.reshape(-1)

def build_M_eff(omega):
    """M_eff(omega) = M11 + M12 * diag(1/(omega - D2)) * M21"""
    inv_denom = 1.0 / (omega - D2_flat)
    scaled_M21 = M21 * inv_denom[:, np.newaxis]  # diag(1/(omega-D2)) * M21
    M_eff = M11 + M12 @ scaled_M21
    return M_eff

# Initial: omega = 0
M_eff_0 = build_M_eff(0.0)
print(f"\nM_eff(omega=0) symmetric: {np.allclose(M_eff_0, M_eff_0.T)}")
evals_0 = np.linalg.eigvalsh(M_eff_0)
print(f"M_eff(omega=0) eigenvalues: {evals_0[:10]}")

# Omega iteration
n_states = 5
omega_threshold = 1e-8
max_iter = 30

print(f"\nOmega iteration (GANSU formulas):")
excitation_energies = np.zeros(n_states)
for k in range(n_states):
    omega = 0.0
    for it in range(max_iter):
        M_eff = build_M_eff(omega)
        evals = np.linalg.eigvalsh(M_eff)
        omega_new = evals[k]
        delta = abs(omega_new - omega)
        if delta < omega_threshold:
            excitation_energies[k] = omega_new
            print(f"  Root {k+1}: converged in {it+1} iter, omega={omega_new:.8f}")
            break
        omega = omega_new
    else:
        excitation_energies[k] = omega_new
        print(f"  Root {k+1}: NOT converged, omega={omega_new:.8f}")

# ======================================================================
# 8. Compare with PySCF ADC(2)
# ======================================================================
print("\n" + "=" * 60)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]

print(f"\n{'State':>5}  {'Python Schur':>14}  {'PySCF ADC(2)':>14}  {'Diff':>12}")
print("-" * 50)
for i in range(n_states):
    diff = excitation_energies[i] - e_adc2[i]
    print(f"{i+1:5d}  {excitation_energies[i]:14.8f}  {e_adc2[i]:14.8f}  {diff:12.6e}")

print("\n" + "=" * 60)
print("Summary:")
print(f"  CIS eigenvalues match PySCF TDA: {np.allclose(cis_evals[:5], td.e[:5], atol=1e-8)}")
print(f"  MP2 energy: {e_mp2:.10f}")
print(f"  Schur complement ADC(2) matches PySCF: {np.allclose(excitation_energies, e_adc2, atol=1e-4)}")
