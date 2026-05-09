"""
Verify the correct Schur complement formula for singlet EE-ADC(2)
using packed (symmetry-adapted) doubles.

PySCF uses 45-dim packed doubles (I<=J, A<=B) instead of full 100-dim.
This script tests different normalizations to find the correct one.

Usage: python verify_packed_schur.py
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
nocc = mol.nelectron // 2
nao = mol.nao_nr()
nvir = nao - nocc
ov = nocc * nvir
dd = nocc * nocc * nvir * nvir
n_packed = nocc * (nocc + 1) // 2 * nvir * (nvir + 1) // 2
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd_full={dd}, dd_packed={n_packed}")

# Get PySCF reference
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

# Build MO integrals
C = mf.mo_coeff
eps = mf.mo_energy
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)

# ERI blocks
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]  # (ia|jb)
eri_vvov = eri_mo[nocc:, nocc:, :nocc, nocc:]  # (ab|ic)
eri_ooov = eri_mo[:nocc, :nocc, :nocc, nocc:]  # (ij|ka)

# T2 amplitudes and D2 denominators
t2 = np.zeros((nocc, nocc, nvir, nvir))
D2_arr = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                D2_arr[i, j, a, b] = eps[a + nocc] + eps[b + nocc] - eps[i] - eps[j]
                t2[i, j, a, b] = eri_ovov[i, a, j, b] / (-D2_arr[i, j, a, b])

# Build M11 (CIS + ISR + self-energy)
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
                val += 2.0 * eri_ovov[i, a, j, b] - eri_mo[i, j, a + nocc, b + nocc]
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
                        val += 2.0 * t2[i, k, a, c] * eri_ovov[j, b, k, c]
                        val -= t2[i, k, a, c] * eri_ovov[k, b, j, c]
                        val -= t2[k, i, a, c] * eri_ovov[j, b, k, c]
                        val += 0.5 * t2[k, i, a, c] * eri_ovov[k, b, j, c]
                        val += 2.0 * t2[j, k, b, c] * eri_ovov[i, a, k, c]
                        val -= t2[j, k, b, c] * eri_ovov[k, a, i, c]
                        val -= t2[k, j, b, c] * eri_ovov[i, a, k, c]
                        val += 0.5 * t2[k, j, b, c] * eri_ovov[k, a, i, c]
                ISR[ia, jb] = val

sigma_oo_raw = np.einsum('ikab,jakb->ij', t2, eri_ovov) \
             - 0.5 * np.einsum('ikab,jbka->ij', t2, eri_ovov)
sigma_oo = sigma_oo_raw + sigma_oo_raw.T
sigma_vv_raw = -np.einsum('ijac,ibjc->ab', t2, eri_ovov) \
             + 0.5 * np.einsum('ijac,jbic->ab', t2, eri_ovov)
sigma_vv = sigma_vv_raw + sigma_vv_raw.T

M11 = CIS + ISR.copy()
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

# Build full M12 and M21
M12 = np.zeros((ov, dd))
for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for Cv in range(nvir):
                    for D in range(nvir):
                        ijcd = I * nocc * nvir * nvir + J * nvir * nvir + Cv * nvir + D
                        val = 0.0
                        if I == K:
                            val += 2.0 * eri_vvov[E, Cv, J, D] - eri_vvov[D, E, J, Cv]
                        if Cv == E:
                            val += eri_ooov[J, K, I, D] - 2.0 * eri_ooov[I, K, J, D]
                        M12[ke, ijcd] = val

M21 = np.zeros((dd, ov))
for I in range(nocc):
    for J in range(nocc):
        for Cv in range(nvir):
            for D in range(nvir):
                ijcd = I * nocc * nvir * nvir + J * nvir * nvir + Cv * nvir + D
                for K in range(nocc):
                    for E in range(nvir):
                        ke = K * nvir + E
                        val = 0.0
                        if K == I:
                            val += eri_vvov[E, Cv, J, D]
                        if K == J:
                            val += eri_vvov[E, D, I, Cv]
                        if E == Cv:
                            val -= eri_ooov[I, K, J, D]
                        if E == D:
                            val -= eri_ooov[J, K, I, Cv]
                        M21[ijcd, ke] = val


# ============================================================
# Helper: packed index mapping
# ============================================================
def packed_ij(i, j):
    """Map (i,j) to triangular index with min(i,j) <= max(i,j)"""
    p, q = min(i, j), max(i, j)
    return q * (q + 1) // 2 + p

def orbit_elements(I, J, A, B):
    """Return all (i,j,a,b) in the orbit of packed index (IJ, AB)"""
    elems = set()
    for ii, jj in [(I, J), (J, I)]:
        for aa, bb in [(A, B), (B, A)]:
            elems.add((ii, jj, aa, bb))
    return list(elems)


# ============================================================
# Approach 1: Full Schur complement (current GANSU, WRONG)
# ============================================================
def schur_complement_full(omega):
    M_eff = M11.copy()
    for idx in range(dd):
        I = idx // (nocc * nvir * nvir)
        rem = idx % (nocc * nvir * nvir)
        J = rem // (nvir * nvir)
        rem2 = rem % (nvir * nvir)
        C = rem2 // nvir
        D = rem2 % nvir
        d2 = eps[C + nocc] + eps[D + nocc] - eps[I] - eps[J]
        M_eff += np.outer(M12[:, idx], M21[idx, :]) / (omega - d2)
    return M_eff


# ============================================================
# Approach 2: Packed Schur complement with various normalizations
# ============================================================
def schur_complement_packed(omega, norm_type="sum_over_m"):
    """
    Sum over packed doubles orbits instead of all 100 doubles.

    norm_type options:
    - "sum_sum_over_m": M12_p = sum, M21_p = sum/m  (average M21)
    - "sum_over_m_both": M12_p = sum/m, M21_p = sum/m
    - "sum_sum_over_m2": divide by m^2
    - "sqrt_m": M12_p = sum/sqrt(m), M21_p = sum/sqrt(m)
    - "sum_sum": M12_p = sum, M21_p = sum (no normalization)
    """
    M_eff = M11.copy()

    n_ij = nocc * (nocc + 1) // 2
    n_ab = nvir * (nvir + 1) // 2

    for I in range(nocc):
        for J in range(I, nocc):  # J >= I
            for A in range(nvir):
                for B in range(A, nvir):  # B >= A
                    d2 = eps[A + nocc] + eps[B + nocc] - eps[I] - eps[J]
                    orbit = orbit_elements(I, J, A, B)
                    m = len(orbit)

                    # Sum M12 columns and M21 rows over orbit
                    m12_sum = np.zeros(ov)
                    m21_sum = np.zeros(ov)
                    for (i, j, a, b) in orbit:
                        idx = i * nocc * nvir * nvir + j * nvir * nvir + a * nvir + b
                        m12_sum += M12[:, idx]
                        m21_sum += M21[idx, :]

                    if norm_type == "sum_sum_over_m":
                        M_eff += np.outer(m12_sum, m21_sum) / (m * (omega - d2))
                    elif norm_type == "sum_over_m_both":
                        M_eff += np.outer(m12_sum / m, m21_sum / m) / (omega - d2)
                    elif norm_type == "sum_sum_over_m2":
                        M_eff += np.outer(m12_sum, m21_sum) / (m * m * (omega - d2))
                    elif norm_type == "sqrt_m":
                        M_eff += np.outer(m12_sum / np.sqrt(m), m21_sum / np.sqrt(m)) / (omega - d2)
                    elif norm_type == "sum_sum":
                        M_eff += np.outer(m12_sum, m21_sum) / (omega - d2)
                    else:
                        raise ValueError(f"Unknown norm_type: {norm_type}")

    return M_eff


# ============================================================
# Approach 3: Symmetrize M12 in the full space
# ============================================================
def symmetrize_M12(M12_in):
    """Symmetrize M12 so M12[:,ijab] = M12[:,jiab] = M12[:,ijba] = M12[:,jiba]"""
    M12_sym = np.zeros_like(M12_in)
    for I in range(nocc):
        for J in range(nocc):
            for A in range(nvir):
                for B in range(nvir):
                    idx = I * nocc * nvir * nvir + J * nvir * nvir + A * nvir + B
                    orbit = orbit_elements(I, J, A, B)
                    m = len(orbit)
                    avg = np.zeros(ov)
                    for (i, j, a, b) in orbit:
                        oidx = i * nocc * nvir * nvir + j * nvir * nvir + a * nvir + b
                        avg += M12_in[:, oidx]
                    M12_sym[:, idx] = avg / m
    return M12_sym


def symmetrize_M21(M21_in):
    """Symmetrize M21 so M21[ijab,:] = M21[jiab,:] = M21[ijba,:] = M21[jiba,:]"""
    M21_sym = np.zeros_like(M21_in)
    for I in range(nocc):
        for J in range(nocc):
            for A in range(nvir):
                for B in range(nvir):
                    idx = I * nocc * nvir * nvir + J * nvir * nvir + A * nvir + B
                    orbit = orbit_elements(I, J, A, B)
                    m = len(orbit)
                    avg = np.zeros(ov)
                    for (i, j, a, b) in orbit:
                        oidx = i * nocc * nvir * nvir + j * nvir * nvir + a * nvir + b
                        avg += M21_in[oidx, :]
                    M21_sym[idx, :] = avg / m
    return M21_sym


def schur_complement_sym(omega, sym_m12=True, sym_m21=True):
    """Schur complement using symmetrized M12 and/or M21 in the FULL space"""
    M12_use = symmetrize_M12(M12) if sym_m12 else M12
    M21_use = symmetrize_M21(M21) if sym_m21 else M21

    M_eff = M11.copy()
    for idx in range(dd):
        I = idx // (nocc * nvir * nvir)
        rem = idx % (nocc * nvir * nvir)
        J = rem // (nvir * nvir)
        rem2 = rem % (nvir * nvir)
        C = rem2 // nvir
        D = rem2 % nvir
        d2 = eps[C + nocc] + eps[D + nocc] - eps[I] - eps[J]
        M_eff += np.outer(M12_use[:, idx], M21_use[idx, :]) / (omega - d2)
    return M_eff


# ============================================================
# Test all approaches with iterative omega convergence
# ============================================================
def converge_schur(build_fn, label, nstates=5, max_iter=200, tol=1e-10):
    """Iterate omega until eigenvalues converge"""
    omega = 0.0  # initial guess
    for it in range(max_iter):
        M_eff = build_fn(omega)
        # Use symmetric eigensolver if symmetric
        if np.allclose(M_eff, M_eff.T, atol=1e-12):
            evals = np.linalg.eigvalsh(M_eff)
        else:
            evals = np.sort(np.linalg.eig(M_eff)[0].real)
        omega_new = evals[0]
        if abs(omega_new - omega) < tol:
            break
        omega = omega_new

    print(f"\n=== {label} ===")
    print(f"  Converged in {it+1} iterations")
    print(f"  Eigenvalues[:5]: {evals[:5]}")
    print(f"  PySCF ADC(2):    {e_adc2}")
    diffs = evals[:nstates] - e_adc2[:nstates]
    print(f"  Differences:     {diffs}")
    print(f"  Max abs diff:    {np.max(np.abs(diffs)):.6e}")
    match = np.allclose(evals[:nstates], e_adc2[:nstates], atol=1e-6)
    print(f"  MATCH: {match}")
    return match, evals


# Test 1: Full Schur complement (current GANSU)
converge_schur(schur_complement_full, "Full Schur (current GANSU)")

# Test 2: Packed with different normalizations
for norm in ["sum_sum_over_m", "sqrt_m", "sum_sum", "sum_over_m_both", "sum_sum_over_m2"]:
    converge_schur(lambda omega, n=norm: schur_complement_packed(omega, n),
                   f"Packed, norm={norm}")

# Test 3: Symmetrized M12 and/or M21 in full space
converge_schur(lambda omega: schur_complement_sym(omega, True, True),
               "Full + sym(M12) + sym(M21)")
converge_schur(lambda omega: schur_complement_sym(omega, True, False),
               "Full + sym(M12) only")
converge_schur(lambda omega: schur_complement_sym(omega, False, True),
               "Full + sym(M21) only")


# ============================================================
# Also try without iterating omega (omega=0, just eigenvalues)
# ============================================================
print("\n" + "="*60)
print("Non-iterative check (omega=0):")

M_eff_full = schur_complement_full(0.0)
evals_full = np.linalg.eigvalsh(M_eff_full) if np.allclose(M_eff_full, M_eff_full.T) else np.sort(np.linalg.eig(M_eff_full)[0].real)
print(f"Full:     {evals_full[:5]}")

for norm in ["sum_sum_over_m", "sqrt_m"]:
    M_eff_p = schur_complement_packed(0.0, norm)
    is_sym = np.allclose(M_eff_p, M_eff_p.T, atol=1e-12)
    evals_p = np.linalg.eigvalsh(M_eff_p) if is_sym else np.sort(np.linalg.eig(M_eff_p)[0].real)
    print(f"Packed({norm}): {evals_p[:5]}  sym={is_sym}")

M_eff_ss = schur_complement_sym(0.0, True, True)
evals_ss = np.linalg.eigvalsh(M_eff_ss) if np.allclose(M_eff_ss, M_eff_ss.T) else np.sort(np.linalg.eig(M_eff_ss)[0].real)
print(f"Sym both: {evals_ss[:5]}")

print(f"PySCF:    {e_adc2}")
