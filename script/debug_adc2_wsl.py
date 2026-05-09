"""
Debug ADC(2) matrix blocks vs PySCF. H2O/STO-3G.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo

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

print(f"RHF energy: {mf.e_tot:.10f}")
print(f"nocc={nocc}, nvir={nvir}, sd={sd}, dd={dd}")

# PySCF ADC(2)
myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=5)
e_ee = result[0]
print(f"PySCF ADC(2) eigenvalues: {e_ee}")

# Get matvec
from pyscf.adc import radc_ee
# Try to get matvec function
try:
    matvec_fn = radc_ee.get_matvec(myadc)
    print("Got matvec from radc_ee.get_matvec")
except Exception as ex:
    print(f"radc_ee.get_matvec failed: {ex}")
    try:
        # Try through the ADC object
        print(f"ADC methods: {[m for m in dir(myadc) if 'matvec' in m.lower() or 'sigma' in m.lower()]}")
        matvec_fn = None
    except:
        matvec_fn = None

if matvec_fn is None:
    # Alternative: build matrix from ADC internals
    print("Trying alternative approach...")
    print(f"radc_ee exports: {[x for x in dir(radc_ee) if not x.startswith('_')]}")

    # Try get_trans_moments or compute_amplitudes
    try:
        imds = myadc.get_imds()
        print(f"imds type: {type(imds)}")
        print(f"imds attrs: {[x for x in dir(imds) if not x.startswith('_')]}")
    except Exception as ex:
        print(f"get_imds failed: {ex}")

# Try building the matvec manually
print("\n=== Trying to extract matvec ===")
try:
    # PySCF 2.x may store sigma fn differently
    sigma_fn = radc_ee.get_matvec_fn
    print(f"Found: {sigma_fn}")
except:
    pass

# Let me try a different approach: look at what compute_trans_moments returns
try:
    # Some versions expose it as kernel/matvec
    adc_obj = adc.ADC(mf)
    adc_obj.method_type = 'ee'
    # Access the sigma vector computation
    from pyscf.adc import radc_ee as ree

    # List all functions
    fns = [x for x in dir(ree) if callable(getattr(ree, x)) and not x.startswith('_')]
    print(f"radc_ee functions: {fns}")

    # Try ea_adc_matvec or similar
    for fn_name in fns:
        if 'matvec' in fn_name.lower() or 'sigma' in fn_name.lower():
            print(f"  Found: {fn_name}")
except Exception as ex:
    print(f"Error: {ex}")

# Alternative: build our own full matrix and compare eigenvalues directly
print("\n=== Direct eigenvalue comparison ===")
eri_mo = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)

# T2 amplitudes
t2 = np.zeros((nocc, nocc, nvir, nvir))
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                t2[i,j,a,b] = eri_mo[i, nocc+a, j, nocc+b] / (eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

# Build M11: CIS + T2 correction (use standard normalization, factor 1)
M11 = np.zeros((sd, sd))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                val = 0.0
                # CIS
                if i==j and a==b: val += eps[nocc+a] - eps[i]
                val += 2.0*eri_mo[i,nocc+a,j,nocc+b] - eri_mo[i,j,nocc+a,nocc+b]

                # T2 vertex correction (1/2 factor)
                for k in range(nocc):
                    for c in range(nvir):
                        K = 2.0*eri_mo[j,nocc+b,k,nocc+c] - eri_mo[j,nocc+c,k,nocc+b]
                        val += 0.5 * t2[i,k,a,c] * K
                        K2 = 2.0*eri_mo[i,nocc+a,k,nocc+c] - eri_mo[i,nocc+c,k,nocc+a]
                        val += 0.5 * t2[j,k,b,c] * K2

                # Self-energy corrections
                if a == b:  # -Sigma_oo[i,j]
                    s = 0.0
                    for k in range(nocc):
                        for ap in range(nvir):
                            for bp in range(nvir):
                                s += t2[i,k,ap,bp] * (2.0*eri_mo[j,nocc+ap,k,nocc+bp] - eri_mo[j,nocc+bp,k,nocc+ap])
                    val -= 0.5*(s + sum(t2[j,k2,ap2,bp2]*(2.0*eri_mo[i,nocc+ap2,k2,nocc+bp2]-eri_mo[i,nocc+bp2,k2,nocc+ap2]) for k2 in range(nocc) for ap2 in range(nvir) for bp2 in range(nvir)))

                if i == j:  # +Sigma_vv[a,b]
                    s = 0.0
                    for ip in range(nocc):
                        for jp in range(nocc):
                            for c in range(nvir):
                                s -= t2[ip,jp,a,c] * (2.0*eri_mo[ip,nocc+b,jp,nocc+c] - eri_mo[ip,nocc+c,jp,nocc+b])
                    s2 = 0.0
                    for ip in range(nocc):
                        for jp in range(nocc):
                            for c in range(nvir):
                                s2 -= t2[ip,jp,b,c] * (2.0*eri_mo[ip,nocc+a,jp,nocc+c] - eri_mo[ip,nocc+c,jp,nocc+a])
                    val += 0.5*(s + s2)

                M11[ia, jb] = val

print(f"M11 symmetric? |M11-M11^T| = {np.max(np.abs(M11 - M11.T)):.2e}")

# D2 diagonal
D2 = np.zeros(dd)
for i in range(nocc):
    for j in range(nocc):
        for a in range(nvir):
            for b in range(nvir):
                idx = i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
                D2[idx] = eps[nocc+a] + eps[nocc+b] - eps[i] - eps[j]

# Build M12 from new sigma1 equation:
# sigma^a_i += 8(ab|jc)r2^bc_ij - 8(ac|jb)r2^bc_ij
# = 8 sum_{j,bc} r2^{bc}_{ij} [(ab|jc) - (ac|jb)]
# M12[ia, ijbc] = 8[(a,b|j,c) - (a,c|j,b)] when K=I (sigma's i)
#
# Also from: -8(ji|kb)r2^ab_jk + 8(jb|ki)r2^ab_jk
# = 8 sum_{jk,b} r2^{ab}_{jk} [-(ji|kb) + (jb|ki)]
# M12[ia, jkab] ... wait this has different structure

# Let me re-read the new sigma1:
# -8 (ji|kb) r2^{ab}_{jk} + 8 (jb|ki) r2^{ab}_{jk}
# For sigma^A_I from r2^{AB}_{JK}: (using A,I as sigma indices, J,K,B as r2 indices)
# Wait: r2^{ab}_{jk} has a,b vir and j,k occ. In sigma^A_I:
# a=A (sigma vir), b=B (r2 vir, but B appears in both sigma and r2?)
# Actually a in r2^{ab}_{jk} is the SAME a as in sigma^a_i. So this is:
# M12[IA, JKAB] = -8(JI|KB) + 8(JB|KI)   ... but this has delta(A,A)
# Hmm no, r2^{ab}_{jk}: a is the first vir index matching sigma's a=A
# So M12[IA, JKAB] += -8(JI|KB) + 8(JB|KI)  when the first vir of doubles = A
# i.e. doubles index is [J,K,A,B] => delta(C=A) in M12[ia, ijCD]

# Let me be very explicit. doubles index: r2^{CD}_{JK} with idx = J*nocc*nvir*nvir + K*nvir*nvir + C*nvir + D
# sigma^A_I contributions from r2^{CD}_{JK}:
#
# From "-8(ji|kb) r2^{ab}_{jk}": j->J, i->I(free), k->K, b->D(?), a->A(free)=C
#   So C=A (delta), and this is -8(JI|KD) when C=A
# From "+8(jb|ki) r2^{ab}_{jk}": j->J, b->D, k->K, i->I(free), a->A(free)=C
#   C=A, +8(JD|KI)
# From "+8(ab|jc) r2^{bc}_{ij}": a->A(free), b->C, j->J, c->D, i->I(free)
#   So doubles index is r2^{CD}_{IJ}: J->K in doubles? No.
#   r2^{bc}_{ij}: b->C, c->D, i->I(free), j->J
#   doubles idx = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
#   Hmm, but the first occ index of doubles = I (same as sigma's I).
#   M12[IA, IJCD] += 8(AC|JD)   (with delta I,I - first occ matches)
# From "-8(ac|jb) r2^{bc}_{ij}": a->A, c->D, j->J, b->C
#   M12[IA, IJCD] += -8(AD|JC)

# So M12 has TWO structural contributions:
# 1) delta(C,A): M12[IA, JKAD] = -8(JI|KD) + 8(JD|KI)  [from r2^{ab}_{jk} terms]
# 2) delta(1st_occ, I): M12[IA, IJCD] = 8(AC|JD) - 8(AD|JC)  [from r2^{bc}_{ij} terms]

print("\n=== Building M12 from NEW sigma1 ===")
M12_new = np.zeros((sd, dd))
for I in range(nocc):
    for A in range(nvir):
        ia = I*nvir + A
        for J in range(nocc):
            for K in range(nocc):
                for C in range(nvir):
                    for D in range(nvir):
                        jkcd = J*nocc*nvir*nvir + K*nvir*nvir + C*nvir + D
                        val = 0.0

                        # From r2^{ab}_{jk} terms: delta(C, A)
                        # -8(JI|KD) + 8(JD|KI) when C=A
                        if C == A:
                            val += -8.0*eri_mo[J,I,K,nocc+D] + 8.0*eri_mo[J,nocc+D,K,I]

                        # From r2^{bc}_{ij} terms: delta(J, I) (first occ of doubles = sigma's I)
                        # 8(AC|KD) - 8(AD|KC)
                        # Wait: r2^{bc}_{ij} with i=I, j=K(2nd occ of doubles)
                        # doubles idx = I*nocc*nvir*nvir + K*nvir*nvir + C*nvir + D
                        # So first occ = J in our loop, must equal I
                        if J == I:
                            val += 8.0*eri_mo[nocc+A,nocc+C,K,nocc+D] - 8.0*eri_mo[nocc+A,nocc+D,K,nocc+C]

                        M12_new[ia, jkcd] = val

# Build M21 from new sigma2:
# -8(ki|bj)r1^a_k + 8(kj|bi)r1^a_k - 8(aj|bc)r1^c_i + 8(ac|bj)r1^c_i
print("Building M21 from NEW sigma2...")
M21_new = np.zeros((dd, sd))
for I in range(nocc):
    for J in range(nocc):
        for A in range(nvir):
            for B in range(nvir):
                ijab = I*nocc*nvir*nvir + J*nvir*nvir + A*nvir + B
                for K in range(nocc):
                    for E in range(nvir):
                        ke = K*nvir + E
                        val = 0.0
                        # -8(ki|bj)r1^a_k: a is sigma's A, k is r1's K, => E=A
                        # (ki|bj) with k=K,i=I,b=B,j=J => (KI|B+nocc,J)
                        if E == A:
                            val += -8.0*eri_mo[K,I,nocc+B,J]
                        # +8(kj|bi)r1^a_k: E=A, (KJ|B+nocc,I)
                        if E == A:
                            val += 8.0*eri_mo[K,J,nocc+B,I]
                        # -8(aj|bc)r1^c_i: a=A,j=J,b=B,c=E, i=I => K=I
                        if K == I:
                            val += -8.0*eri_mo[nocc+A,J,nocc+B,nocc+E]
                        # +8(ac|bj)r1^c_i: a=A,c=E,b=B,j=J, i=I => K=I
                        if K == I:
                            val += 8.0*eri_mo[nocc+A,nocc+E,nocc+B,J]
                        M21_new[ijab, ke] = val

# Check M12 = M21^T (Hermiticity)
print(f"\n=== Hermiticity check ===")
print(f"|M12_new - M21_new^T| = {np.max(np.abs(M12_new - M21_new.T)):.2e}")

# Build full matrix
full_dim = sd + dd
M_our = np.zeros((full_dim, full_dim))
M_our[:sd, :sd] = M11
M_our[:sd, sd:] = M12_new
M_our[sd:, :sd] = M21_new
M_our[sd:, sd:] = np.diag(D2)

print(f"|M_our - M_our^T| = {np.max(np.abs(M_our - M_our.T)):.2e}")

our_evals = np.linalg.eigvalsh(M_our)
print(f"\nOur eigenvalues (first 5):   {our_evals[:5]}")
print(f"PySCF eigenvalues (first 5): {e_ee[:5]}")
print(f"Differences: {our_evals[:5] - e_ee[:5]}")

# Try with normalization factor on M12/M21
for norm in [0.5, 0.25, 0.125, 1.0/8, 1.0/16, 2.0, 4.0]:
    M_test = np.zeros((full_dim, full_dim))
    M_test[:sd, :sd] = M11
    M_test[:sd, sd:] = M12_new * norm
    M_test[sd:, :sd] = M21_new * norm
    M_test[sd:, sd:] = np.diag(D2)
    test_evals = np.linalg.eigvalsh(M_test)
    err = np.max(np.abs(test_evals[:5] - e_ee[:5]))
    if err < 0.01:
        print(f"  norm={norm:.4f}: error={err:.6f} evals={test_evals[:5]}")

# Also try normalizing D2
for d2_norm in [2, 4, 8, 16]:
    M_test = np.zeros((full_dim, full_dim))
    M_test[:sd, :sd] = M11
    M_test[:sd, sd:] = M12_new
    M_test[sd:, :sd] = M21_new
    M_test[sd:, sd:] = np.diag(D2) * d2_norm
    for m_norm in [0.125, 0.25, 0.5, 1.0]:
        M_test2 = M_test.copy()
        M_test2[:sd, sd:] *= m_norm
        M_test2[sd:, :sd] *= m_norm
        test_evals = np.linalg.eigvalsh(M_test2)
        err = np.max(np.abs(test_evals[:5] - e_ee[:5]))
        if err < 0.01:
            print(f"  D2*{d2_norm}, M12*{m_norm:.3f}: error={err:.6f}")
