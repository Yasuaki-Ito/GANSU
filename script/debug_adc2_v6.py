"""
ADC(2) final diagnostic: verify non-sym eigenvalues, fix M11.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee

np.set_printoptions(precision=10, linewidth=200, suppress=True)

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
eps = mf.mo_energy
sd = nocc * nvir
dd = nocc**2 * nvir**2

myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=5)
e_ee = result[0]

# Build full matrix from matvec
M_ab = radc_ee.get_imds(myadc)
eris = myadc.transform_integrals()
mv = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)

dim = sd + dd
M = np.zeros((dim, dim))
for col in range(dim):
    e = np.zeros(dim); e[col] = 1.0
    M[:, col] = mv(e)

# Non-sym eigensolve
evals = np.sort(np.linalg.eigvals(M).real)
print(f"Non-sym eigenvalues: {evals[:5]}")
print(f"PySCF kernel:        {e_ee[:5]}")
print(f"Diff: {evals[:5] - e_ee[:5]}")

# Extract blocks
M11 = M[:sd, :sd]
M12 = M[:sd, sd:]
M21 = M[sd:, :sd]
D2m = M[sd:, sd:]

print(f"\n|M11 - M11^T| = {np.max(np.abs(M11 - M11.T)):.2e}")
print(f"|M12 - M21^T| = {np.max(np.abs(M12 - M21.T)):.2e}")

# Check: is M12^T * D2^{-1} * M12 = M21 * D2^{-1} * M21^T?
# i.e., does the Schur complement M11 + M12*(w-D2)^{-1}*M21 give symmetric result?
d2d = np.diag(D2m)
omega = 0.5
inv_oD2 = np.diag(1.0 / (omega - d2d))
M_eff = M11 + M12 @ inv_oD2 @ M21
print(f"\nSchur complement at omega={omega}:")
print(f"|M_eff - M_eff^T| = {np.max(np.abs(M_eff - M_eff.T)):.2e}")

# Also check at other omega values
for w in [0.0, 0.3, 0.5, 0.8, 1.0]:
    inv_oD2 = np.diag(1.0 / (w - d2d))
    M_eff = M11 + M12 @ inv_oD2 @ M21
    sym_err = np.max(np.abs(M_eff - M_eff.T))
    print(f"  omega={w}: |M_eff - M_eff^T| = {sym_err:.2e}")

# Now check our M11 vs PySCF M11
eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
t2_ours = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                t2_ours[i,j,a,b] = eri[i, nocc+a, j, nocc+b] / (eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

# Check PySCF's T2
t2_pyscf = myadc.t2[0]  # shape (nocc, nocc, nvir, nvir)
print(f"\nt2 shapes: ours={t2_ours.shape}, PySCF={t2_pyscf.shape}")
print(f"|t2_ours - t2_pyscf| = {np.max(np.abs(t2_ours - t2_pyscf)):.2e}")

# Check PySCF's T1
t1_pyscf = myadc.t1[0]  # shape (nocc, nvir)
print(f"t1_pyscf shape: {t1_pyscf.shape}")
print(f"|t1_pyscf| max = {np.max(np.abs(t1_pyscf)):.2e}")
# For MP2, t1 should be zero. For ADC(2) ISR, t1 comes from the ISR transformation.

# Build our M11 and compare element-by-element with PySCF
our_M11 = np.zeros((sd, sd))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                val = 0.0
                # CIS
                if i==j and a==b: val += eps[nocc+a] - eps[i]
                val += 2.0*eri[i,nocc+a,j,nocc+b] - eri[i,j,nocc+a,nocc+b]
                # T2 vertex (1/2)
                for k in range(nocc):
                    for c in range(nvir):
                        val += 0.5*t2_ours[i,k,a,c]*(2.0*eri[j,nocc+b,k,nocc+c] - eri[j,nocc+c,k,nocc+b])
                        val += 0.5*t2_ours[j,k,b,c]*(2.0*eri[i,nocc+a,k,nocc+c] - eri[i,nocc+c,k,nocc+a])
                # Self-energy
                if a == b:
                    s = 0.0
                    for k in range(nocc):
                        for ap in range(nvir):
                            for bp in range(nvir):
                                s += t2_ours[i,k,ap,bp]*(2.0*eri[j,nocc+ap,k,nocc+bp]-eri[j,nocc+bp,k,nocc+ap])
                    s2 = 0.0
                    for k in range(nocc):
                        for ap in range(nvir):
                            for bp in range(nvir):
                                s2 += t2_ours[j,k,ap,bp]*(2.0*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                    val -= 0.5*(s + s2)
                if i == j:
                    s = 0.0
                    for ip in range(nocc):
                        for jp in range(nocc):
                            for c in range(nvir):
                                s -= t2_ours[ip,jp,a,c]*(2.0*eri[ip,nocc+b,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+b])
                    s2 = 0.0
                    for ip in range(nocc):
                        for jp in range(nocc):
                            for c in range(nvir):
                                s2 -= t2_ours[ip,jp,b,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                    val += 0.5*(s + s2)
                our_M11[ia, jb] = val

err11 = np.max(np.abs(our_M11 - M11))
print(f"\n|our_M11 - PySCF_M11| = {err11:.2e}")

if err11 > 1e-6:
    print("M11 elements with largest diff:")
    diff11 = our_M11 - M11
    for ia in range(sd):
        for jb in range(sd):
            if abs(diff11[ia,jb]) > 1e-4:
                i,a = ia//nvir, ia%nvir
                j,b = jb//nvir, jb%nvir
                print(f"  [{i},{a}|{j},{b}]: ours={our_M11[ia,jb]:.8f} pyscf={M11[ia,jb]:.8f} diff={diff11[ia,jb]:.6f}")

# Now check: if we use PySCF's exact M11 with our M12/M21/D2,
# do non-sym eigenvalues match?
# Build M12/M21 from our formulas (which are algebraically same as PySCF)
our_D2 = np.array([eps[nocc+a]+eps[nocc+b]-eps[i]-eps[j]
                    for i in range(nocc) for j in range(nocc)
                    for a in range(nvir) for b in range(nvir)])

# Use matvec to get exact M12 and M21
# M12: apply to doubles-only vectors
# M21: apply to singles-only vectors
M12_exact = np.zeros((sd, dd))
for col in range(dd):
    e = np.zeros(dim); e[sd + col] = 1.0
    sv = mv(e)
    M12_exact[:, col] = sv[:sd]

M21_exact = np.zeros((dd, sd))
for col in range(sd):
    e = np.zeros(dim); e[col] = 1.0
    sv = mv(e)
    M21_exact[:, col] = sv[sd:]  # but this includes D2 contribution too...

# Actually M21 from unit singles vectors also gets D2*0 = 0 contribution from doubles part
# So sv[sd:] = M21 * e_col is correct since r2 = 0
print(f"\n|M12_exact - M12| = {np.max(np.abs(M12_exact - M12)):.2e}")  # should be 0
print(f"|M21_exact - M21| = {np.max(np.abs(M21_exact - M21)):.2e}")  # should be 0

# Now build GANSU's M12 and M21 from code formulas
# GANSU M12: delta(I,K)*[2*vvov[E,C,J,D] - vvov[D,E,J,C]] + delta(C,E)*[ooov[J,K,I,D] - 2*ooov[I,K,J,D]]
# where vvov[E,C,J,D] = (EC|JD) = eri[nocc+E, nocc+C, J, nocc+D]
# and ooov[J,K,I,D] = (JK|ID) = eri[J, K, I, nocc+D]

gansu_M12 = np.zeros((sd, dd))
gansu_M21 = np.zeros((dd, sd))
for I in range(nocc):
    for J in range(nocc):
        for C in range(nvir):
            for D in range(nvir):
                ijcd = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
                for K in range(nocc):
                    for E in range(nvir):
                        ke = K*nvir + E
                        # GANSU M12
                        m12 = 0.0
                        if I == K:
                            m12 += 2.0*eri[nocc+E,nocc+C,J,nocc+D] - eri[nocc+D,nocc+E,J,nocc+C]
                        if C == E:
                            m12 += eri[J,K,I,nocc+D] - 2.0*eri[I,K,J,nocc+D]
                        gansu_M12[ke, ijcd] = m12

                        # GANSU M21
                        m21 = 0.0
                        if K == I: m21 += eri[nocc+E,nocc+C,J,nocc+D]
                        if K == J: m21 += eri[nocc+E,nocc+D,I,nocc+C]
                        if E == C: m21 -= eri[I,K,J,nocc+D]
                        if E == D: m21 -= eri[J,K,I,nocc+C]
                        gansu_M21[ijcd, ke] = m21

print(f"\n|gansu_M12 - PySCF_M12| = {np.max(np.abs(gansu_M12 - M12)):.2e}")
print(f"|gansu_M21 - PySCF_M21| = {np.max(np.abs(gansu_M21 - M21)):.2e}")

# If gansu M12/M21 match PySCF, the ONLY issue is M11
# Build full matrix with PySCF's M11 + gansu's M12/M21/D2
M_hybrid = np.zeros((dim, dim))
M_hybrid[:sd, :sd] = M11  # PySCF M11
M_hybrid[:sd, sd:] = gansu_M12
M_hybrid[sd:, :sd] = gansu_M21
M_hybrid[sd:, sd:] = np.diag(our_D2)

hybrid_evals = np.sort(np.linalg.eigvals(M_hybrid).real)
print(f"\nHybrid (PySCF_M11 + GANSU_M12/M21/D2):")
print(f"  eigenvalues: {hybrid_evals[:5]}")
print(f"  PySCF:       {evals[:5]}")
print(f"  diff:        {hybrid_evals[:5] - evals[:5]}")

# Also: omega SC with PySCF data
print(f"\n=== Schur complement convergence ===")
omega = 0.0
for iteration in range(10):
    inv_oD2 = np.diag(1.0 / (omega - d2d))
    M_eff = M11 + M12 @ inv_oD2 @ M21
    # Non-sym eigen
    eff_evals = np.sort(np.linalg.eigvals(M_eff).real)
    omega_new = eff_evals[0]
    print(f"  iter {iteration}: omega={omega:.8f} -> {omega_new:.8f} (PySCF: {e_ee[0]:.8f})")
    if abs(omega_new - omega) < 1e-10:
        break
    omega = omega_new
