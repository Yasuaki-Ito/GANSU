"""
ADC(2) non-symmetric analysis.
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

# Build full matrix
M_ab = radc_ee.get_imds(myadc)
eris = myadc.transform_integrals()
mv = radc_ee.matvec(myadc, M_ab=M_ab, eris=eris)

dim = sd + dd
M = np.zeros((dim, dim))
for col in range(dim):
    e = np.zeros(dim); e[col] = 1.0
    M[:, col] = mv(e)

# Non-symmetric eigensolve
evals_nonsym = np.linalg.eigvals(M)
evals_real = np.sort(evals_nonsym.real)
print(f"Non-sym eigenvalues (first 5):  {evals_real[:5]}")
print(f"PySCF kernel eigenvalues:       {e_ee[:5]}")
print(f"Max imaginary part: {np.max(np.abs(evals_nonsym.imag)):.2e}")
print(f"Diff (nonsym vs PySCF): {evals_real[:5] - e_ee[:5]}")

# Check: are eigenvalues real?
print(f"\nAll eigenvalues real? {np.allclose(evals_nonsym.imag, 0, atol=1e-10)}")

# Extract blocks
M11 = M[:sd, :sd]
M12 = M[:sd, sd:]
M21 = M[sd:, :sd]
D2m = M[sd:, sd:]

print(f"\n|M11 - M11^T| = {np.max(np.abs(M11 - M11.T)):.2e}")
print(f"|M12 - M21^T| = {np.max(np.abs(M12 - M21.T)):.2e}")

# What is M_ab (the get_imds output)?
print(f"\nM_ab shape: {M_ab.shape}")
print(f"|M_ab - M11| = {np.max(np.abs(M_ab - M11)):.2e}")
# M_ab might be just the CIS part (without T2 correction fully baked in)

# Check: does PySCF store T2 correction in M_ab already?
# Build CIS matrix for comparison
eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
CIS = np.zeros((sd, sd))
for i in range(nocc):
    for a in range(nvir):
        ia = i*nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j*nvir + b
                val = 0.0
                if i==j and a==b: val += eps[nocc+a] - eps[i]
                val += 2.0*eri[i,nocc+a,j,nocc+b] - eri[i,j,nocc+a,nocc+b]
                CIS[ia, jb] = val

print(f"|M_ab - CIS| = {np.max(np.abs(M_ab - CIS)):.2e}")
print(f"|M11 - CIS| = {np.max(np.abs(M11 - CIS)):.2e}")

# So M11 from matvec is CIS + T2 correction applied through r2 intermediates?
# Or is M11 just CIS?
# Let's see what the matvec does for singles-only input

# Check: what PySCF variables store
print(f"\nt1 type: {type(myadc.t1)}, len: {len(myadc.t1)}")
print(f"t2 type: {type(myadc.t2)}, len: {len(myadc.t2)}")
if isinstance(myadc.t1, (list, tuple)):
    print(f"t1[0] shape: {myadc.t1[0].shape}")
if isinstance(myadc.t2, (list, tuple)):
    print(f"t2[0] shape: {myadc.t2[0].shape}")

# PySCF convention: t1_ccee = t2[0] means occ,occ,vir,vir (t2 amplitudes)
# t2_ce = t1[0] means something like "t1" = (occ, vir) singles amplitudes (MP2 t1?)

# Now let's analyze M21 more carefully
# From the source, the key ADC(2) terms for sigma2 (doubles block) are:
#
# 1) D2 * r2 (diagonal term)
# 2) from v_ceee (ovvv): einsum('Ia,JDaC->IJCD', Y, v_ceee) + einsum('Ja,ICaD->IJCD', Y, v_ceee)
# 3) from v_cecc (ovoo): -einsum('iC,JDIi->IJCD', Y, v_cecc) - einsum('iD,ICJi->IJCD', Y, v_cecc)
#
# v_ceee = eris.ovvv[i,a,b,c] = (ia|bc) chemist notation
# v_cecc = eris.ovoo[i,a,j,k] = (ia|jk) chemist notation

# So the M21 terms from PySCF are:
# sigma2[I,J,C,D] += einsum('Ia,JDaC->IJCD', r1, v_ceee)
#                  = sum_a r1[I,a] * v_ceee[J,D,a,C]
#                  = sum_a r1[I,a] * (JD|aC) = sum_a r1[I,a] * eri_ovvv[J,D,a,C]
#
# This has delta K=I (r1's occ index = sigma's I index)
# M21[IJCD, Ia] += (JD|aC) = eri[J, nocc+D, nocc+a, nocc+C]... wait
# Actually v_ceee uses PySCF's notation where c=occ, e=vir
# v_ceee = ovvv: index [occ, vir, vir, vir]
# v_ceee[J,D,a,C] = (ovvv)[J,D,a,C] which in chemist is... let me check

# PySCF's eris.ovvv[i,a,b,c] = (ia|bc) in chemist notation?
# Let me verify
ovvv = eris.ovvv
print(f"\neris.ovvv shape: {ovvv.shape if ovvv is not None else 'None'}")
if ovvv is not None:
    # Check: does eris.ovvv[i,a,b,c] = (ia|bc)?
    # Compare with full eri
    i,a,b,c = 0,0,0,0
    print(f"eris.ovvv[{i},{a},{b},{c}] = {ovvv[i,a,b,c]:.10f}")
    print(f"eri_full(i,a+nocc|b+nocc,c+nocc) = {eri[i,nocc+a,nocc+b,nocc+c]:.10f}")
    i,a,b,c = 1,0,1,0
    print(f"eris.ovvv[{i},{a},{b},{c}] = {ovvv[i,a,b,c]:.10f}")
    print(f"eri_full({i},{a}+nocc|{b}+nocc,{c}+nocc) = {eri[i,nocc+a,nocc+b,nocc+c]:.10f}")

ovoo = eris.ovoo
print(f"\neris.ovoo shape: {ovoo.shape if ovoo is not None else 'None'}")
if ovoo is not None:
    i,a,j,k = 0,0,0,0
    print(f"eris.ovoo[{i},{a},{j},{k}] = {ovoo[i,a,j,k]:.10f}")
    print(f"eri_full({i},{a}+nocc|{j},{k}) = {eri[i,nocc+a,j,k]:.10f}")
    i,a,j,k = 1,0,2,3
    print(f"eris.ovoo[{i},{a},{j},{k}] = {ovoo[i,a,j,k]:.10f}")
    print(f"eri_full({i},{a}+nocc|{j},{k}) = {eri[i,nocc+a,j,k]:.10f}")

# Now let's carefully build M21 from PySCF's matvec formulas
# For ADC(2), the sigma2 terms involving r1 (Y) are:
#
# From ovvv: sigma2[I,J,C,D] += Y[I,a]*v_ovvv[J,D,a,C] + Y[J,a]*v_ovvv[I,C,a,D]
# From ovoo: sigma2[I,J,C,D] -= Y[i,C]*v_ovoo[J,D,I,i] + Y[i,D]*v_ovoo[I,C,J,i]
#
# = einsum('Ia,JDaC->IJCD', Y, ovvv) + einsum('Ja,ICaD->IJCD', Y, ovvv)
#   -einsum('iC,JDIi->IJCD', Y, ovoo) - einsum('iD,ICJi->IJCD', Y, ovoo)
#
# Translating to M21[IJCD, KE]:
# Term 1: delta(K,I) * ovvv[J,D,E,C]
# Term 2: delta(K,J) * ovvv[I,C,E,D]
# Term 3: -delta(E,C) * ovoo[J,D,I,K]
# Term 4: -delta(E,D) * ovoo[I,C,J,K]

if ovvv is not None and ovoo is not None:
    print(f"\n=== Building M21 from PySCF formulas ===")
    M21_test = np.zeros((dd, sd))
    for I in range(nocc):
        for J in range(nocc):
            for C in range(nvir):
                for D in range(nvir):
                    ijcd = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
                    for K in range(nocc):
                        for E in range(nvir):
                            ke = K*nvir + E
                            val = 0.0
                            if K == I: val += ovvv[J,D,E,C]
                            if K == J: val += ovvv[I,C,E,D]
                            if E == C: val -= ovoo[J,D,I,K]
                            if E == D: val -= ovoo[I,C,J,K]
                            M21_test[ijcd, ke] = val

    err = np.max(np.abs(M21_test - M21))
    print(f"|M21_test - M21_pyscf| = {err:.2e}")

    # Similarly M12 from PySCF source:
    # From ovvv: sigma1[I,D] += -einsum('Iiab,iabD->ID', r2, v_ceee)
    #                         + 2*einsum('Iiab,ibDa->ID', r2, v_ceee)
    # From ovoo: sigma1[I,D] -= 2*einsum('ijDa,jaiI->ID', r2, v_cecc)
    #                         + einsum('ijDa,iajI->ID', r2, v_cecc)
    #
    # M12[ID, IJab]:
    # Term1: -v_ovvv[I,a,b,D]  (sum over I with delta K=first_occ_of_r2=I? No...)
    # Actually let me be more careful:
    # -einsum('Iiab,iabD', r2, ovvv) means:
    #  sigma1[I,D] += sum_{i,a,b} r2[I,i,a,b] * (-ovvv[i,a,b,D])
    #  So M12[ID, Iiab] += -ovvv[i,a,b,D]   (first occ of doubles = I = sigma's occ)
    # +2*einsum('Iiab,ibDa', r2, ovvv) means:
    #  sigma1[I,D] += sum_{i,a,b} r2[I,i,a,b] * 2*ovvv[i,b,D,a]
    #  M12[ID, Iiab] += 2*ovvv[i,b,D,a]

    # -2*einsum('ijDa,jaiI', r2, ovoo):
    #  sigma1[I,D] += sum_{i,j,a} r2[i,j,D,a] * (-2*ovoo[j,a,i,I])
    #  But r2 index is [i,j,D,a] where D is sigma's vir index, fixed!
    #  doubles index: [i,j,D,a], so 3rd vir index = D = sigma's vir
    #  M12[ID, ijDa] += -2*ovoo[j,a,i,I]    (delta: 3rd vir of doubles = D = sigma's vir)

    # +einsum('ijDa,iajI', r2, ovoo):
    #  sigma1[I,D] += sum_{i,j,a} r2[i,j,D,a] * ovoo[i,a,j,I]
    #  M12[ID, ijDa] += ovoo[i,a,j,I]

    print(f"\n=== Building M12 from PySCF formulas ===")
    M12_test = np.zeros((sd, dd))
    for K in range(nocc):  # sigma occ
        for E in range(nvir):  # sigma vir
            ke = K*nvir + E
            for I in range(nocc):
                for J in range(nocc):
                    for C in range(nvir):
                        for D in range(nvir):
                            ijcd = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
                            val = 0.0
                            # From -ovvv[i,a,b,D]: K=I(1st occ), i=J, a=C, b=D... wait
                            # r2[I,i,a,b] -> doubles index [I,J,C,D]
                            # So I=K(sigma occ), i=J(2nd occ), a=C, b=D
                            if K == I:
                                val += -ovvv[J,C,D,E]  # -ovvv[i,a,b,D] with D->E(sigma vir)
                                val += 2.0*ovvv[J,D,E,C]  # +2*ovvv[i,b,D,a] -> ovvv[J,D,E,C]

                            # From r2[i,j,D,a]: 3rd index of doubles = E(sigma vir)
                            # doubles [I,J,C,D]: C must = E (sigma's vir)
                            if C == E:
                                val += -2.0*ovoo[J,D,I,K]  # -2*ovoo[j,a,i,I] with I->K(sigma occ)
                                val += ovoo[I,D,J,K]  # +ovoo[i,a,j,I] -> ovoo[I,D,J,K]

                            M12_test[ke, ijcd] = val

    err12 = np.max(np.abs(M12_test - M12))
    print(f"|M12_test - M12_pyscf| = {err12:.2e}")

    # Check Hermiticity
    print(f"|M12_test - M21_test^T| = {np.max(np.abs(M12_test - M21_test.T)):.2e}")

    # Now convert to our ERI notation
    # ovvv[i,a,b,c] = eri[i, nocc+a, nocc+b, nocc+c] in chemist notation
    # ovoo[i,a,j,k] = eri[i, nocc+a, j, k]
    #
    # M21[IJCD, KE]:
    # delta(K,I): ovvv[J,D,E,C] = eri[J, nocc+D, nocc+E, nocc+C] = (J,D+o | E+o,C+o)
    # delta(K,J): ovvv[I,C,E,D] = eri[I, nocc+C, nocc+E, nocc+D] = (I,C+o | E+o,D+o)
    # delta(E,C): -ovoo[J,D,I,K] = -eri[J, nocc+D, I, K] = -(J,D+o | I,K)
    # delta(E,D): -ovoo[I,C,J,K] = -eri[I, nocc+C, J, K] = -(I,C+o | J,K)
    #
    # M12[KE, IJCD]:
    # delta(K,I): -ovvv[J,C,D,E] + 2*ovvv[J,D,E,C]
    #           = -eri[J,C+o,D+o,E+o] + 2*eri[J,D+o,E+o,C+o]
    #           = -(J,C+o|D+o,E+o) + 2*(J,D+o|E+o,C+o)
    # delta(C,E): -2*ovoo[J,D,I,K] + ovoo[I,D,J,K]
    #           = -2*eri[J,D+o,I,K] + eri[I,D+o,J,K]
    #           = -2*(J,D+o|I,K) + (I,D+o|J,K)
    print(f"\n=== Converting to chemist notation ERIs ===")
    print("M21[IJCD, KE]:")
    print("  delta(K,I): +(JD|EC) = eri[J, D+o, E+o, C+o]")
    print("  delta(K,J): +(IC|ED) = eri[I, C+o, E+o, D+o]")
    print("  delta(E,C): -(JD|IK) = -eri[J, D+o, I, K]")
    print("  delta(E,D): -(IC|JK) = -eri[I, C+o, J, K]")
    print("")
    print("M12[KE, IJCD]:")
    print("  delta(K,I): -(JC|DE) + 2(JD|EC) = -eri[J,C+o,D+o,E+o] + 2*eri[J,D+o,E+o,C+o]")
    print("  delta(C,E): -2(JD|IK) + (ID|JK) = -2*eri[J,D+o,I,K] + eri[I,D+o,J,K]")

    # Now compare with GANSU's code formulas:
    # GANSU M12: delta(I,K)*[2*(EC|JD) - (DE|JC)] + delta(C,E)*[(JK|ID) - 2*(IK|JD)]
    # PySCF M12: delta(K,I)*[-(JC|DE) + 2*(JD|EC)] + delta(C,E)*[-2*(JD|IK) + (ID|JK)]
    #
    # Using (pq|rs) = (rs|pq) (8-fold symmetry):
    # (EC|JD) = (JD|EC), (DE|JC) = (JC|DE), (JK|ID) = (ID|JK), (IK|JD) = (JD|IK)
    #
    # GANSU: delta(I,K)*[2*(JD|EC) - (JC|DE)] + delta(C,E)*[(ID|JK) - 2*(JD|IK)]
    # PySCF: delta(K,I)*[2*(JD|EC) - (JC|DE)] + delta(C,E)*[(ID|JK) - 2*(JD|IK)]
    #
    # THEY ARE THE SAME!

    print("\n=== GANSU M12 vs PySCF M12: IDENTICAL after ERI symmetry! ===")

    # Now check M21:
    # GANSU M21: delta(K,I)*(EC|JD) + delta(K,J)*(ED|IC) - delta(E,C)*(IK|JD) - delta(E,D)*(JK|IC)
    # PySCF M21: delta(K,I)*(JD|EC) + delta(K,J)*(IC|ED) - delta(E,C)*(JD|IK) - delta(E,D)*(IC|JK)
    #
    # Using symmetry: (EC|JD)=(JD|EC), (ED|IC)=(IC|ED), (IK|JD)=(JD|IK), (JK|IC)=(IC|JK)
    # GANSU = PySCF for M21 too!

    print("=== GANSU M21 vs PySCF M21: IDENTICAL after ERI symmetry! ===")

    # So the formulas are the same! The problem must be elsewhere.
    # Let me check M11 more carefully.
    print(f"\n=== M11 comparison ===")
    print(f"|M_ab - CIS| = {np.max(np.abs(M_ab - CIS)):.2e}")
    print(f"|M11 - M_ab| = {np.max(np.abs(M11 - M_ab)):.2e}")
    # M11 in matvec is just M_ab applied to r1: s[s1:f1] = M_ab @ r1
    # So M11 = M_ab. But does M_ab include T2 corrections?

    # Build our M11 (CIS + T2 correction)
    t2_ours = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2_ours[i,j,a,b] = eri[i, nocc+a, j, nocc+b] / (eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

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
                    for k in range(nocc):
                        for c in range(nvir):
                            K1 = 2.0*eri[j,nocc+b,k,nocc+c] - eri[j,nocc+c,k,nocc+b]
                            val += 0.5*t2_ours[i,k,a,c]*K1
                            K2 = 2.0*eri[i,nocc+a,k,nocc+c] - eri[i,nocc+c,k,nocc+a]
                            val += 0.5*t2_ours[j,k,b,c]*K2
                    if a == b:
                        s_ij = sum(t2_ours[i,k,ap,bp]*(2.0*eri[j,nocc+ap,k,nocc+bp]-eri[j,nocc+bp,k,nocc+ap])
                                   for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                        s_ji = sum(t2_ours[j,k,ap,bp]*(2.0*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                                   for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                        val -= 0.5*(s_ij + s_ji)
                    if i == j:
                        s_ab = -sum(t2_ours[ip,jp,a,c]*(2.0*eri[ip,nocc+b,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+b])
                                    for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                        s_ba = -sum(t2_ours[ip,jp,b,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                                    for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                        val += 0.5*(s_ab + s_ba)
                    our_M11[ia, jb] = val

    print(f"|our_M11 - M_ab| = {np.max(np.abs(our_M11 - M_ab)):.2e}")

    # If M_ab = our_M11, then the issue is just that M_ab includes T2 corrections
    # But from the output above: |M_ab - CIS| was nonzero, |our_M11 - PySCF M11| was 3e-3
    # This means M_ab IS our_M11, but there's something else in the matvec that contributes to M11

    # Actually wait - the matvec source shows:
    # s[s1:f1] = M_ @ r1  (only this for singles-singles)
    # So M11 from the full matrix build SHOULD equal M_ab.
    # But the output said |M11 - CIS| was nonzero...
    # That means M_ab includes the T2 corrections already.
    # And |our_M11 - M11| = 3e-3 means our T2 correction formula differs from PySCF's.

    # Build full matrix with our formulas and compare
    M_ours = np.zeros((dim, dim))
    M_ours[:sd, :sd] = our_M11
    M_ours[:sd, sd:] = M12_test  # confirmed same as PySCF
    M_ours[sd:, :sd] = M21_test  # confirmed same as PySCF
    M_ours[sd:, sd:] = np.diag(D2_ours := np.array([eps[nocc+a]+eps[nocc+b]-eps[i]-eps[j]
                                                      for i in range(nocc) for j in range(nocc)
                                                      for a in range(nvir) for b in range(nvir)]))

    our_evals = np.sort(np.linalg.eigvals(M_ours).real)
    print(f"\nOur full matrix evals (first 5):    {our_evals[:5]}")
    print(f"PySCF full matrix evals (first 5):  {evals_real[:5]}")
    print(f"PySCF kernel evals (first 5):       {e_ee[:5]}")
    print(f"Diff (ours - pyscf_full): {our_evals[:5] - evals_real[:5]}")

    # The error should come from M11 only (since M12, M21, D2 match)
    # Let's use PySCF's exact M_ab and see if we get exact match
    M_mixed = np.zeros((dim, dim))
    M_mixed[:sd, :sd] = M_ab  # PySCF's M11
    M_mixed[:sd, sd:] = M12_test
    M_mixed[sd:, :sd] = M21_test
    M_mixed[sd:, sd:] = np.diag(D2_ours)

    mixed_evals = np.sort(np.linalg.eigvals(M_mixed).real)
    print(f"\nMixed (PySCF M11 + our M12/M21/D2): {mixed_evals[:5]}")
    print(f"Diff from PySCF full:               {mixed_evals[:5] - evals_real[:5]}")
