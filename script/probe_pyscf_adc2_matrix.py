"""
Probe PySCF's ADC(2) matvec with unit vectors to extract the full matrix.
Compare block-by-block (M11, M12, M21, D2) with our formulas.

Usage: python probe_pyscf_adc2_matrix.py
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
total = ov + dd
print(f"nocc={nocc}, nvir={nvir}, ov={ov}, dd={dd}, total={total}")

# Run ADC(2)
myadc = adc.ADC(mf)
myadc.method = "adc(2)"
result = myadc.kernel(nroots=5)
e_adc2 = result[0]
print(f"PySCF ADC(2): {e_adc2}")

# Get the matvec function
from pyscf.adc import radc_ee

# Get M_ab (the M11 matrix) and eris
M_ab = radc_ee.get_imds(myadc)
print(f"M_ab type: {type(M_ab)}, shape: {np.array(M_ab).shape if hasattr(M_ab, 'shape') else 'unknown'}")

eris = myadc.transform_integrals()
print(f"eris type: {type(eris)}")
print(f"eris attrs: {[a for a in dir(eris) if not a.startswith('_') and hasattr(getattr(eris, a), 'shape')]}")

# Get sigma_ function
sigma_fn = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)
print(f"sigma_fn type: {type(sigma_fn)}")

# Test with a random vector
r_test = np.random.randn(total)
sigma_test = sigma_fn(r_test)
print(f"sigma_fn works! Input shape: {r_test.shape}, output shape: {sigma_test.shape}")

# Build the full matrix by probing with unit vectors
print(f"\nProbing with {total} unit vectors...")
M_pyscf = np.zeros((total, total))
for col in range(total):
    e_vec = np.zeros(total)
    e_vec[col] = 1.0
    M_pyscf[:, col] = sigma_fn(e_vec)

print(f"PySCF matrix built.")
print(f"Symmetric: {np.allclose(M_pyscf, M_pyscf.T)}")
print(f"Max asymmetry: {np.max(np.abs(M_pyscf - M_pyscf.T)):.6e}")

# Diagonalize
if np.allclose(M_pyscf, M_pyscf.T):
    evals_pyscf = np.linalg.eigvalsh(M_pyscf)
else:
    evals_pyscf = np.sort(np.linalg.eig(M_pyscf)[0].real)
print(f"Matrix eigenvalues[:10]: {evals_pyscf[:10]}")
print(f"PySCF ADC(2) kernel:    {e_adc2}")

# Extract blocks
M11_pyscf = M_pyscf[:ov, :ov]
M12_pyscf = M_pyscf[:ov, ov:]
M21_pyscf = M_pyscf[ov:, :ov]
D2_block_pyscf = M_pyscf[ov:, ov:]

print(f"\n--- Block analysis ---")
print(f"M11: sym={np.allclose(M11_pyscf, M11_pyscf.T)}, norm={np.linalg.norm(M11_pyscf):.6f}")
print(f"M12: norm={np.linalg.norm(M12_pyscf):.6f}")
print(f"M21: norm={np.linalg.norm(M21_pyscf):.6f}")
print(f"M12==M21^T: {np.allclose(M12_pyscf, M21_pyscf.T)}")
print(f"Max |M12-M21^T|: {np.max(np.abs(M12_pyscf - M21_pyscf.T)):.6e}")
print(f"D2 diagonal: {np.allclose(D2_block_pyscf, np.diag(np.diag(D2_block_pyscf)))}")
print(f"D2 off-diag max: {np.max(np.abs(D2_block_pyscf - np.diag(np.diag(D2_block_pyscf)))):.6e}")

# Now build our version and compare
C = mf.mo_coeff
eps = mf.mo_energy
eri_mo = ao2mo.kernel(mol, C, compact=False).reshape(nao, nao, nao, nao)
eri_ovov = eri_mo[:nocc, nocc:, :nocc, nocc:]
eri_vvov = eri_mo[nocc:, nocc:, :nocc, nocc:]
eri_ooov = eri_mo[:nocc, :nocc, :nocc, nocc:]

t2 = np.zeros((nocc, nocc, nvir, nvir))
D2_arr = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                D2_arr[i,j,a,b] = eps[a+nocc] + eps[b+nocc] - eps[i] - eps[j]
                t2[i,j,a,b] = eri_ovov[i,a,j,b] / (-D2_arr[i,j,a,b])

# D2 comparison
D2_diag_pyscf = np.diag(D2_block_pyscf)
print(f"\n=== Block comparisons ===")
print(f"D2: max_diff={np.max(np.abs(D2_diag_pyscf - D2_arr.reshape(-1))):.6e}")

# M12 comparison
M12_ours = np.zeros((ov, dd))
for K in range(nocc):
    for E in range(nvir):
        ke = K * nvir + E
        for I in range(nocc):
            for J in range(nocc):
                for Cv in range(nvir):
                    for D in range(nvir):
                        ijcd = I*nocc*nvir*nvir + J*nvir*nvir + Cv*nvir + D
                        val = 0.0
                        if I == K:
                            val += 2.0*eri_vvov[E,Cv,J,D] - eri_vvov[D,E,J,Cv]
                        if Cv == E:
                            val += eri_ooov[J,K,I,D] - 2.0*eri_ooov[I,K,J,D]
                        M12_ours[ke, ijcd] = val

M21_ours = np.zeros((dd, ov))
for I in range(nocc):
    for J in range(nocc):
        for Cv in range(nvir):
            for D in range(nvir):
                ijcd = I*nocc*nvir*nvir + J*nvir*nvir + Cv*nvir + D
                for K in range(nocc):
                    for E in range(nvir):
                        ke = K * nvir + E
                        val = 0.0
                        if K == I: val += eri_vvov[E,Cv,J,D]
                        if K == J: val += eri_vvov[E,D,I,Cv]
                        if E == Cv: val -= eri_ooov[I,K,J,D]
                        if E == D: val -= eri_ooov[J,K,I,Cv]
                        M21_ours[ijcd, ke] = val

print(f"M12: max_diff={np.max(np.abs(M12_pyscf - M12_ours)):.6e}")
print(f"M21: max_diff={np.max(np.abs(M21_pyscf - M21_ours)):.6e}")

# M11 comparison - build all components
CIS = np.zeros((ov, ov))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                val = 0.0
                if i == j and a == b: val += eps[a+nocc] - eps[i]
                val += 2.0*eri_ovov[i,a,j,b] - eri_mo[i,j,a+nocc,b+nocc]
                CIS[ia, jb] = val

ISR = np.zeros((ov, ov))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                val = 0.0
                for k in range(nocc):
                    for c in range(nvir):
                        val += 2.0*t2[i,k,a,c]*eri_ovov[j,b,k,c]
                        val -= t2[i,k,a,c]*eri_ovov[k,b,j,c]
                        val -= t2[k,i,a,c]*eri_ovov[j,b,k,c]
                        val += 0.5*t2[k,i,a,c]*eri_ovov[k,b,j,c]
                        val += 2.0*t2[j,k,b,c]*eri_ovov[i,a,k,c]
                        val -= t2[j,k,b,c]*eri_ovov[k,a,i,c]
                        val -= t2[k,j,b,c]*eri_ovov[i,a,k,c]
                        val += 0.5*t2[k,j,b,c]*eri_ovov[k,a,i,c]
                ISR[ia, jb] = val

sigma_oo_raw = np.einsum('ikab,jakb->ij', t2, eri_ovov) \
             - 0.5*np.einsum('ikab,jbka->ij', t2, eri_ovov)
sigma_oo = sigma_oo_raw + sigma_oo_raw.T
sigma_vv_raw = -np.einsum('ijac,ibjc->ab', t2, eri_ovov) \
             + 0.5*np.einsum('ijac,jbic->ab', t2, eri_ovov)
sigma_vv = sigma_vv_raw + sigma_vv_raw.T

M11_ours = CIS + ISR
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                if a == b: M11_ours[ia, jb] -= sigma_oo[i, j]
                if i == j: M11_ours[ia, jb] += sigma_vv[a, b]

print(f"M11: max_diff={np.max(np.abs(M11_pyscf - M11_ours)):.6e}")

# Component analysis
print(f"\n=== M11 component analysis ===")
print(f"CIS vs PySCF M11:           max_diff={np.max(np.abs(M11_pyscf - CIS)):.6e}")
print(f"CIS+ISR vs PySCF M11:       max_diff={np.max(np.abs(M11_pyscf - CIS - ISR)):.6e}")

# What does PySCF's M_ab look like compared to our M11?
if hasattr(M_ab, 'shape'):
    M_ab_2d = np.array(M_ab).reshape(ov, ov) if M_ab.ndim != 2 else np.array(M_ab)
    print(f"\nPySCF M_ab (get_imds): shape={M_ab_2d.shape}")
    print(f"M_ab vs probed M11:   max_diff={np.max(np.abs(M_ab_2d - M11_pyscf)):.6e}")
    print(f"M_ab vs our M11:      max_diff={np.max(np.abs(M_ab_2d - M11_ours)):.6e}")
    print(f"M_ab vs CIS:          max_diff={np.max(np.abs(M_ab_2d - CIS)):.6e}")

# Print difference if significant
if np.max(np.abs(M11_pyscf - M11_ours)) > 1e-8:
    print(f"\n  M11 DIFFERS!")
    diff = M11_pyscf - M11_ours
    print(f"  Diff matrix (PySCF - ours):")
    for i in range(ov):
        row = " ".join(f"{diff[i,j]:10.6f}" for j in range(ov))
        print(f"    [{i}]: {row}")
