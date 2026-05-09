"""
Verify Schur complement approach for H2O/cc-pVDZ.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='cc-pvdz', verbose=0)
mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
eps = mf.mo_energy
sd = nocc * nvir
dd = nocc**2 * nvir**2

print(f"nocc={nocc}, nvir={nvir}, sd={sd}, dd={dd}")
print(f"Matrix dimension: {sd+dd}")

myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=3)
e_ee = result[0]
print(f"PySCF eigenvalues: {e_ee}")

# This is too large to build full matrix (sd+dd = 100 + 10000 = 10100)
# But we can build M11 (100x100) and test Schur complement with GANSU's formulas

# Build M11 from PySCF
M_ab = radc_ee.get_imds(myadc)
print(f"M11 shape: {M_ab.shape}")

# Build our M11
eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
t2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                t2[i,j,a,b] = eri[i,nocc+a,j,nocc+b]/(eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

our_M11 = np.zeros((sd, sd))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir+a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir+b
                val = 0.0
                if i==j and a==b: val += eps[nocc+a]-eps[i]
                val += 2.0*eri[i,nocc+a,j,nocc+b] - eri[i,j,nocc+a,nocc+b]
                for k in range(nocc):
                    for c in range(nvir):
                        val += 0.5*t2[i,k,a,c]*(2.0*eri[j,nocc+b,k,nocc+c]-eri[j,nocc+c,k,nocc+b])
                        val += 0.5*t2[j,k,b,c]*(2.0*eri[i,nocc+a,k,nocc+c]-eri[i,nocc+c,k,nocc+a])
                if a==b:
                    s1=sum(t2[i,k,ap,bp]*(2*eri[j,nocc+ap,k,nocc+bp]-eri[j,nocc+bp,k,nocc+ap])
                           for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    s2=sum(t2[j,k,ap,bp]*(2*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                           for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    val -= 0.5*(s1+s2)
                if i==j:
                    s1=-sum(t2[ip,jp,a,c]*(2*eri[ip,nocc+b,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+b])
                            for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    s2=-sum(t2[ip,jp,b,c]*(2*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                            for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    val += 0.5*(s1+s2)
                our_M11[ia,jb] = val

print(f"|our_M11 - PySCF_M11| = {np.max(np.abs(our_M11 - M_ab)):.2e}")

# Build M12*x and M21*x using GANSU's formulas to verify
# M12[KE, IJCD] = delta(I,K)*[2*(EC|JD)-(DE|JC)] + delta(C,E)*[(JK|ID)-2*(IK|JD)]
# M21[IJCD, KE] = delta(K,I)*(EC|JD) + delta(K,J)*(ED|IC) - delta(E,C)*(IK|JD) - delta(E,D)*(JK|IC)

D2 = np.array([eps[nocc+a]+eps[nocc+b]-eps[i]-eps[j]
               for i in range(nocc) for j in range(nocc)
               for a in range(nvir) for b in range(nvir)])

# Build M_eff(omega) = M11 + M12 * diag(1/(omega-D2)) * M21
# Do this by computing M12*diag(1/(omega-D2))*M21 using the delta structure
# For each (ke, jpbp) element: sum over IJCD of M12[ke,IJCD] * (1/(w-D2[IJCD])) * M21[IJCD,jpbp]

def build_M_eff(M11_mat, omega, eri_full, eps_arr, nocc_val, nvir_val):
    sd_val = nocc_val * nvir_val
    M_eff = M11_mat.copy()

    # For efficiency, compute the Schur complement term
    for ke in range(sd_val):
        K = ke // nvir_val
        E = ke % nvir_val
        for jpbp in range(sd_val):
            Jp = jpbp // nvir_val
            Bp = jpbp % nvir_val
            schur_val = 0.0

            for I in range(nocc_val):
                for J in range(nocc_val):
                    for C in range(nvir_val):
                        for D in range(nvir_val):
                            # M12[KE, IJCD]
                            m12 = 0.0
                            if I == K:
                                m12 += 2.0*eri_full[nocc_val+E,nocc_val+C,J,nocc_val+D] - eri_full[nocc_val+D,nocc_val+E,J,nocc_val+C]
                            if C == E:
                                m12 += eri_full[J,K,I,nocc_val+D] - 2.0*eri_full[I,K,J,nocc_val+D]

                            if abs(m12) < 1e-15: continue

                            # M21[IJCD, JpBp]
                            m21 = 0.0
                            if Jp == I: m21 += eri_full[nocc_val+Bp,nocc_val+C,J,nocc_val+D]
                            if Jp == J: m21 += eri_full[nocc_val+Bp,nocc_val+D,I,nocc_val+C]
                            if Bp == C: m21 -= eri_full[I,Jp,J,nocc_val+D]
                            if Bp == D: m21 -= eri_full[J,Jp,I,nocc_val+C]

                            if abs(m21) < 1e-15: continue

                            idx = I*nocc_val*nvir_val*nvir_val + J*nvir_val*nvir_val + C*nvir_val + D
                            w = 1.0 / (omega - D2[idx])
                            schur_val += m12 * w * m21

            M_eff[ke, jpbp] += schur_val
    return M_eff

# This is too slow for nvir=20... let me use vectorized approach
print("\nBuilding Schur complement vectorized...")

def build_schur_fast(M11_mat, omega, eri_full, eps_arr, no, nv):
    sd_val = no * nv
    dd_val = no * no * nv * nv
    n = eri_full.shape[0]

    # Build M12 and M21 as dense matrices
    M12 = np.zeros((sd_val, dd_val))
    M21 = np.zeros((dd_val, sd_val))

    for I in range(no):
        for J in range(no):
            for C in range(nv):
                for D in range(nv):
                    ijcd = I*no*nv*nv + J*nv*nv + C*nv + D
                    for K in range(no):
                        for E in range(nv):
                            ke = K*nv + E
                            # M12
                            m12 = 0.0
                            if I==K: m12 += 2*eri_full[no+E,no+C,J,no+D]-eri_full[no+D,no+E,J,no+C]
                            if C==E: m12 += eri_full[J,K,I,no+D]-2*eri_full[I,K,J,no+D]
                            M12[ke, ijcd] = m12
                            # M21
                            m21 = 0.0
                            if K==I: m21 += eri_full[no+E,no+C,J,no+D]
                            if K==J: m21 += eri_full[no+E,no+D,I,no+C]
                            if E==C: m21 -= eri_full[I,K,J,no+D]
                            if E==D: m21 -= eri_full[J,K,I,no+C]
                            M21[ijcd, ke] = m21

    D2_arr = np.array([eps_arr[no+a]+eps_arr[no+b]-eps_arr[i]-eps_arr[j]
                        for i in range(no) for j in range(no)
                        for a in range(nv) for b in range(nv)])

    inv_oD2 = 1.0 / (omega - D2_arr)
    scaled_M21 = M21 * inv_oD2[:, np.newaxis]
    M_eff = M11_mat + M12 @ scaled_M21
    return M_eff, M12, M21, D2_arr

# Build once
M_eff0, M12, M21, D2 = build_schur_fast(our_M11, 0.0, eri, eps, nocc, nvir)
print(f"M12 shape: {M12.shape}, M21 shape: {M21.shape}")
print(f"|M_eff(0) - M_eff(0)^T| = {np.max(np.abs(M_eff0 - M_eff0.T)):.2e}")

# Omega iteration
for root in range(3):
    omega = 0.0
    print(f"\nRoot {root+1}:")
    for it in range(15):
        inv_oD2 = 1.0 / (omega - D2)
        scaled_M21 = M21 * inv_oD2[:, np.newaxis]
        M_eff = our_M11 + M12 @ scaled_M21
        evals = np.linalg.eigvalsh(M_eff)
        omega_new = evals[root]
        delta = abs(omega_new - omega)
        if it < 3 or delta < 1e-6:
            print(f"  iter {it}: omega={omega:.10f} -> {omega_new:.10f} (delta={delta:.2e})")
        if delta < 1e-10:
            break
        omega = omega_new
    print(f"  FINAL: {omega:.10f} (PySCF: {e_ee[root]:.10f}, diff: {omega-e_ee[root]:.2e})")
