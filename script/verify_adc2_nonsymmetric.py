"""
Verify ADC(2) Schur complement with non-symmetric eigensolver.

The full ADC(2) matrix in the redundant spatial orbital doubles space is
NON-symmetric (M12 != M21^T). Using eigvalsh (symmetric) on M_eff gives
wrong results. This script tests whether eig (general) gives correct results.

Also tests the full-space diagonalization approach.

Usage: python verify_adc2_nonsymmetric.py
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo

mol = gto.M(
    atom='''
    O   0.000   0.000   0.117
    H   0.000   0.757  -0.470
    H   0.000  -0.757  -0.470
    ''',
    basis='sto-3g',
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
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}")

C = mf.mo_coeff
eps = mf.mo_energy

# MO ERIs (chemist notation)
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)
eri_vvov = eri_mo[nocc:, nocc:, :nocc, nocc:]  # (ab|ic)
eri_ooov = eri_mo[:nocc, :nocc, :nocc, nocc:]  # (ij|ka)

# T2 and D2
t2 = np.zeros((nocc, nocc, nvir, nvir))
D2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                D2[i, j, a, b] = eps[a + nocc] + eps[b + nocc] - eps[i] - eps[j]
                t2[i, j, a, b] = eri_ovov[i, a, j, b] / (-D2[i, j, a, b])

# Build M11 (CIS + ISR + self-energy) — known to be correct
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
                        val += 2.0 * t2[i,k,a,c] * eri_ovov[j,b,k,c]
                        val -= t2[i,k,a,c] * eri_ovov[k,b,j,c]
                        val -= t2[k,i,a,c] * eri_ovov[j,b,k,c]
                        val += 0.5 * t2[k,i,a,c] * eri_ovov[k,b,j,c]
                        val += 2.0 * t2[j,k,b,c] * eri_ovov[i,a,k,c]
                        val -= t2[j,k,b,c] * eri_ovov[k,a,i,c]
                        val -= t2[k,j,b,c] * eri_ovov[i,a,k,c]
                        val += 0.5 * t2[k,j,b,c] * eri_ovov[k,a,i,c]
                ISR[ia, jb] = val

sigma_oo_raw = np.einsum('ikab,jakb->ij', t2, eri_ovov) - 0.5 * np.einsum('ikab,jbka->ij', t2, eri_ovov)
sigma_oo = sigma_oo_raw + sigma_oo_raw.T

sigma_vv_raw = -np.einsum('ijac,ibjc->ab', t2, eri_ovov) + 0.5 * np.einsum('ijac,jbic->ab', t2, eri_ovov)
sigma_vv = sigma_vv_raw + sigma_vv_raw.T

M11 = CIS + ISR
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

print(f"M11 symmetric: {np.allclose(M11, M11.T)}")

# Build M12 [ov x dd] and M21 [dd x ov] — PySCF formulas
print("\nBuilding M12 and M21...")
M12 = np.zeros((ov, dd))
M21 = np.zeros((dd, ov))

for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for Cv in range(nvir):
                    for D in range(nvir):
                        ijcd = I * nocc * nvir * nvir + J * nvir * nvir + Cv * nvir + D
                        # M12
                        m12_val = 0.0
                        if I == K:
                            m12_val += 2.0 * eri_vvov[E, Cv, J, D] - eri_vvov[D, E, J, Cv]
                        if Cv == E:
                            m12_val += eri_ooov[J, K, I, D] - 2.0 * eri_ooov[I, K, J, D]
                        M12[ke, ijcd] = m12_val

                        # M21
                        m21_val = 0.0
                        if K == I:
                            m21_val += eri_vvov[E, Cv, J, D]
                        if K == J:
                            m21_val += eri_vvov[E, D, I, Cv]
                        if E == Cv:
                            m21_val -= eri_ooov[I, K, J, D]
                        if E == D:
                            m21_val -= eri_ooov[J, K, I, Cv]
                        M21[ijcd, ke] = m21_val

print(f"M12 == M21^T: {np.allclose(M12, M21.T)}")
print(f"Max |M12 - M21^T|: {np.max(np.abs(M12 - M21.T)):.6e}")

# ======================================================================
# Test 1: Full matrix diagonalization (ov + dd dimensional)
# ======================================================================
print("\n=== Test 1: Full matrix diagonalization ===")
full_dim = ov + dd
M_full = np.zeros((full_dim, full_dim))
M_full[:ov, :ov] = M11
M_full[:ov, ov:] = M12
M_full[ov:, :ov] = M21
M_full[ov:, ov:] = np.diag(D2.reshape(-1))

print(f"Full matrix symmetric: {np.allclose(M_full, M_full.T)}")
print(f"Max asymmetry: {np.max(np.abs(M_full - M_full.T)):.6e}")

# General eigensolver (non-symmetric)
evals_full, evecs_full = np.linalg.eig(M_full)
# Sort by real part
idx_sort = np.argsort(evals_full.real)
evals_full_sorted = evals_full[idx_sort]

print(f"Imaginary parts: max={np.max(np.abs(evals_full.imag)):.6e}")
print(f"Full matrix eigenvalues (real, first 10): {evals_full_sorted.real[:10]}")

# ======================================================================
# Test 2: Schur complement with SYMMETRIC eigensolver (current GANSU approach)
# ======================================================================
print("\n=== Test 2: Schur complement + eigvalsh (GANSU's current approach) ===")
D2_flat = D2.reshape(-1)

n_states = 5
omega_threshold = 1e-8
max_iter = 30

def build_M_eff(omega):
    inv_denom = 1.0 / (omega - D2_flat)
    scaled_M21 = M21 * inv_denom[:, np.newaxis]
    return M11 + M12 @ scaled_M21

# Check symmetry
M_eff_0 = build_M_eff(0.0)
print(f"M_eff(0) symmetric: {np.allclose(M_eff_0, M_eff_0.T)}")
print(f"Max asymmetry: {np.max(np.abs(M_eff_0 - M_eff_0.T)):.6e}")

print("Omega iteration with eigvalsh:")
for k in range(n_states):
    omega = 0.0
    for it in range(max_iter):
        M_eff = build_M_eff(omega)
        evals = np.linalg.eigvalsh(M_eff)
        omega_new = evals[k]
        if abs(omega_new - omega) < omega_threshold:
            print(f"  Root {k+1}: converged iter {it+1}, omega={omega_new:.8f}")
            break
        omega = omega_new
    else:
        print(f"  Root {k+1}: NOT converged, omega={omega_new:.8f}")

# ======================================================================
# Test 3: Schur complement with GENERAL eigensolver
# ======================================================================
print("\n=== Test 3: Schur complement + eig (general eigensolver) ===")
print("Omega iteration with eig:")
energies_eig = []
for k in range(n_states):
    omega = 0.0
    for it in range(max_iter):
        M_eff = build_M_eff(omega)
        evals_complex = np.linalg.eig(M_eff)[0]
        # Take only real eigenvalues (or nearly real)
        real_mask = np.abs(evals_complex.imag) < 1e-6
        evals_real = np.sort(evals_complex[real_mask].real)
        if k >= len(evals_real):
            print(f"  Root {k+1}: not enough real eigenvalues")
            break
        omega_new = evals_real[k]
        if abs(omega_new - omega) < omega_threshold:
            energies_eig.append(omega_new)
            print(f"  Root {k+1}: converged iter {it+1}, omega={omega_new:.8f}")
            break
        omega = omega_new
    else:
        energies_eig.append(omega_new)
        print(f"  Root {k+1}: NOT converged, omega={omega_new:.8f}")

# ======================================================================
# Test 4: Schur complement with SYMMETRIZED M_eff
# ======================================================================
print("\n=== Test 4: Schur complement + symmetrized M_eff ===")
print("Omega iteration with (M_eff + M_eff^T)/2 + eigvalsh:")
energies_sym = []
for k in range(n_states):
    omega = 0.0
    for it in range(max_iter):
        M_eff = build_M_eff(omega)
        M_eff_sym = 0.5 * (M_eff + M_eff.T)
        evals = np.linalg.eigvalsh(M_eff_sym)
        omega_new = evals[k]
        if abs(omega_new - omega) < omega_threshold:
            energies_sym.append(omega_new)
            print(f"  Root {k+1}: converged iter {it+1}, omega={omega_new:.8f}")
            break
        omega = omega_new
    else:
        energies_sym.append(omega_new)
        print(f"  Root {k+1}: NOT converged, omega={omega_new:.8f}")

# ======================================================================
# PySCF reference
# ======================================================================
print("\n=== PySCF ADC(2) reference ===")
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

# ======================================================================
# Comparison
# ======================================================================
print("\n" + "=" * 70)
print(f"{'State':>5}  {'eigvalsh':>12}  {'eig':>12}  {'symmetrized':>12}  {'PySCF':>12}")
print("-" * 70)

# Re-run eigvalsh approach to get stored values
energies_eigvalsh = []
for k in range(n_states):
    omega = 0.0
    for it in range(max_iter):
        M_eff = build_M_eff(omega)
        evals = np.linalg.eigvalsh(M_eff)
        omega_new = evals[k]
        if abs(omega_new - omega) < omega_threshold:
            break
        omega = omega_new
    energies_eigvalsh.append(omega_new)

for i in range(n_states):
    ev = energies_eigvalsh[i] if i < len(energies_eigvalsh) else float('nan')
    eg = energies_eig[i] if i < len(energies_eig) else float('nan')
    es = energies_sym[i] if i < len(energies_sym) else float('nan')
    ep = e_adc2[i]
    print(f"{i+1:5d}  {ev:12.8f}  {eg:12.8f}  {es:12.8f}  {ep:12.8f}")

print("\nDifferences from PySCF:")
print(f"{'State':>5}  {'eigvalsh diff':>14}  {'eig diff':>14}  {'sym diff':>14}")
print("-" * 55)
for i in range(n_states):
    ev = energies_eigvalsh[i] if i < len(energies_eigvalsh) else float('nan')
    eg = energies_eig[i] if i < len(energies_eig) else float('nan')
    es = energies_sym[i] if i < len(energies_sym) else float('nan')
    ep = e_adc2[i]
    print(f"{i+1:5d}  {ev-ep:14.6e}  {eg-ep:14.6e}  {es-ep:14.6e}")
