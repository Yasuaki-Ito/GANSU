"""
Get full PySCF ADC(2) matvec source and build full matrix.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo
from pyscf.adc import radc_ee
import inspect

np.set_printoptions(precision=10, linewidth=200, suppress=True)

# Print full matvec source
src = inspect.getsource(radc_ee.matvec)
print("=== FULL MATVEC SOURCE ===")
print(src)
print("=== END SOURCE ===")

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
print(f"\nPySCF ADC(2) eigenvalues: {e_ee}")

# Get M_ab and eris
M_ab = radc_ee.get_imds(myadc)
eris = myadc.transform_integrals()

# Build matvec function
mv = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)
print(f"matvec type: {type(mv)}")

# Try applying it
dim = sd + dd
v = np.zeros(dim); v[0] = 1.0
try:
    sv = mv(v)
    print(f"mv(v) shape: {np.shape(sv)}")
    vec_size = len(sv)
except Exception as ex:
    print(f"mv(v) failed: {ex}")
    vec_size = None

if vec_size:
    print(f"\nBuilding full matrix ({vec_size}x{vec_size})...")
    M = np.zeros((vec_size, vec_size))
    for col in range(vec_size):
        e = np.zeros(vec_size); e[col] = 1.0
        M[:, col] = mv(e)

    print(f"|M - M^T| = {np.max(np.abs(M - M.T)):.2e}")
    evals = np.linalg.eigvalsh(M)
    print(f"Matrix eigenvalues (first 5): {evals[:5]}")
    print(f"PySCF kernel:                 {e_ee[:5]}")

    # Extract blocks
    M11 = M[:sd, :sd]
    M12 = M[:sd, sd:]
    M21 = M[sd:, :sd]
    D2m = M[sd:, sd:]

    print(f"\n|M12 - M21^T| = {np.max(np.abs(M12 - M21.T)):.2e}")
    d2_offdiag = np.max(np.abs(D2m - np.diag(np.diag(D2m))))
    print(f"|D2 off-diag| = {d2_offdiag:.2e}")

    # D2 comparison
    our_D2 = np.zeros(dd)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    idx = i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
                    our_D2[idx] = eps[nocc+a] + eps[nocc+b] - eps[i] - eps[j]

    d2d = np.diag(D2m)
    print(f"|D2 - our_D2| = {np.max(np.abs(d2d - our_D2)):.2e}")

    # M11 comparison: build our M11
    eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2[i,j,a,b] = eri[i, nocc+a, j, nocc+b] / (eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

    our_M11 = np.zeros((sd, sd))
    for i in range(nocc):
        for a in range(nvir):
            ia = i*nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    jb = j*nvir + b
                    val = 0.0
                    if i==j and a==b: val += eps[nocc+a] - eps[i]
                    val += 2.0*eri[i,nocc+a,j,nocc+b] - eri[i,j,nocc+a,nocc+b]
                    # T2 vertex (1/2 factor)
                    for k in range(nocc):
                        for c in range(nvir):
                            K1 = 2.0*eri[j,nocc+b,k,nocc+c] - eri[j,nocc+c,k,nocc+b]
                            val += 0.5 * t2[i,k,a,c] * K1
                            K2 = 2.0*eri[i,nocc+a,k,nocc+c] - eri[i,nocc+c,k,nocc+a]
                            val += 0.5 * t2[j,k,b,c] * K2
                    # Self-energy
                    if a == b:
                        s_ij = sum(t2[i,k,ap,bp]*(2.0*eri[j,nocc+ap,k,nocc+bp]-eri[j,nocc+bp,k,nocc+ap])
                                   for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                        s_ji = sum(t2[j,k,ap,bp]*(2.0*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                                   for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                        val -= 0.5*(s_ij + s_ji)
                    if i == j:
                        s_ab = -sum(t2[ip,jp,a,c]*(2.0*eri[ip,nocc+b,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+b])
                                    for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                        s_ba = -sum(t2[ip,jp,b,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                                    for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                        val += 0.5*(s_ab + s_ba)
                    our_M11[ia, jb] = val

    print(f"\n|our_M11 - PySCF M11| = {np.max(np.abs(our_M11 - M11)):.2e}")

    # Reverse-engineer M21 from PySCF
    print(f"\n=== Reverse-engineering M21 ===")
    # For elements with only delta(E,A) (K!=I, E==A):
    count = 0
    for idx2 in range(dd):
        I = idx2 // (nocc*nvir*nvir)
        r = idx2 % (nocc*nvir*nvir)
        J = r // (nvir*nvir)
        r2 = r % (nvir*nvir)
        A = r2 // nvir
        B = r2 % nvir
        for ke in range(sd):
            K = ke // nvir
            E = ke % nvir
            pval = M21[idx2, ke]
            if abs(pval) < 1e-10: continue
            if E == A and K != I:
                e1 = eri[K,I,nocc+B,J]  # (ki|bj)
                e2 = eri[K,J,nocc+B,I]  # (kj|bi)
                if abs(e1) > 1e-12 or abs(e2) > 1e-12:
                    print(f"  δ(E=A) [{I},{J},{A},{B}|{K},{E}] val={pval:+.8f}  -(ki|bj)={-e1:.8f} +(kj|bi)={+e2:.8f}  ratio_e1={pval/(-e1) if abs(e1)>1e-12 else 'inf'}  ratio_e2={pval/e2 if abs(e2)>1e-12 else 'inf'}")
                    count += 1
            if K == I and E != A:
                e1 = eri[nocc+A,J,nocc+B,nocc+E]  # (aj|be)
                e2 = eri[nocc+A,nocc+E,nocc+B,J]  # (ae|bj)
                if abs(e1) > 1e-12 or abs(e2) > 1e-12:
                    print(f"  δ(K=I) [{I},{J},{A},{B}|{K},{E}] val={pval:+.8f}  -(aj|be)={-e1:.8f} +(ae|bj)={+e2:.8f}  ratio_e1={pval/(-e1) if abs(e1)>1e-12 else 'inf'}  ratio_e2={pval/e2 if abs(e2)>1e-12 else 'inf'}")
                    count += 1
            if count >= 20: break
        if count >= 20: break
