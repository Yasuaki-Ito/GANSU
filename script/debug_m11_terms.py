"""
Identify which M11 terms are missing by building PySCF's M11 step by step.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, lib
from pyscf.adc import radc_ee, radc_ao2mo

np.set_printoptions(precision=10, linewidth=200, suppress=True)

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
            basis='sto-3g', verbose=0)
mf = scf.RHF(mol).run()
nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
eps = mf.mo_energy
einsum = lib.einsum

myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=5)

# Get PySCF internals
t1_ccee = myadc.t2[0][:]  # T2 amplitudes (nocc,nocc,nvir,nvir)
t2_ce = myadc.t1[0]       # "T1" amplitudes (nocc,nvir)
eris = myadc.transform_integrals()

v_ccee = eris.oovv  # (ij|ab)
v_cece = eris.ovvo  # (ia|bj)... or (ia|jb)?
v_ceec = eris.ovvo
v_cecc = eris.ovoo  # (ia|jk)

occ_list = np.array(range(nocc))
vir_list = np.array(range(nvir))

# Step 1: Section 000 (diagonal)
M_000 = np.zeros((nocc, nvir, nocc, nvir))
d_ai_a = eps[nocc:][:,None] - eps[:nocc]
diag_flat = np.zeros(nocc * nvir)
np.fill_diagonal(diag_flat.reshape(nocc*nvir, -1) if False else np.diag(np.ones(1)), 0)
# Actually:
M_flat = np.zeros((nocc*nvir, nocc*nvir))
np.fill_diagonal(M_flat, d_ai_a.transpose().reshape(-1))
M_000 = M_flat.reshape(nocc, nvir, nocc, nvir).copy()

# Step 2: Section 010 (CIS-like)
M_010 = np.zeros((nocc, nvir, nocc, nvir))
M_010 -= einsum('ILAD->IDLA', v_ccee)
M_010 += einsum('LADI->IDLA', v_ceec)
M_010 += einsum('LADI->IDLA', v_ceec)

# Step 3: Section 020 (T2 correction)
M_020 = np.zeros((nocc, nvir, nocc, nvir))
M_020 += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece)
M_020 -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece)
M_020 += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece)
M_020 -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece)
M_020 -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece)
M_020 += einsum('iIDa,iAaL->IDLA', t1_ccee, v_cece)
M_020 -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece)
M_020 += einsum('iLAa,iDaI->IDLA', t1_ccee, v_cece)

# Additional 020 terms (with ## marks)
M_020b = np.zeros((nocc, nvir, nocc, nvir))
M_020b -= einsum('LAai,IiDa->IDLA', v_ceec, t1_ccee)
M_020b += 0.5 * einsum('LAai,iIDa->IDLA', v_ceec, t1_ccee)
M_020b += 0.5 * einsum('iAaL,IiDa->IDLA', v_ceec, t1_ccee)
M_020b -= 0.5 * einsum('iAaL,iIDa->IDLA', v_ceec, t1_ccee)
M_020b -= einsum('LiAa,IDai->IDLA', t1_ccee, v_ceec)
M_020b += 0.5 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec)
M_020b += 0.5 * einsum('iLAa,IDai->IDLA', t1_ccee, v_ceec)
M_020b -= 0.5 * einsum('iLAa,iDaI->IDLA', t1_ccee, v_ceec)
M_020b -= einsum('IiDa,LAai->IDLA', t1_ccee, v_ceec)
M_020b += 0.5 * einsum('IiDa,iAaL->IDLA', t1_ccee, v_ceec)
M_020b += 0.5 * einsum('iIDa,LAai->IDLA', t1_ccee, v_ceec)

# Self-energy like terms
M_020c = np.zeros((nocc, nvir, nocc, nvir))
M_020c[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t1_ccee, v_cece)
M_020c[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t1_ccee, v_cece)
M_020c[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t1_ccee, v_cece)
M_020c[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL', t1_ccee, v_cece)
M_020c[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t1_ccee, v_cece)
M_020c[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t1_ccee, v_cece)
M_020c[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t1_ccee, v_cece)
M_020c[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA', t1_ccee, v_cece)

M_020d = np.zeros((nocc, nvir, nocc, nvir))
M_020d[occ_list,:,occ_list,:] += einsum('iAaj,ijDa->DA', v_ceec, t1_ccee)
M_020d[occ_list,:,occ_list,:] -= 0.5 * einsum('iAaj,jiDa->DA', v_ceec, t1_ccee)
M_020d[occ_list,:,occ_list,:] += einsum('ijAa,iDaj->DA', t1_ccee, v_ceec)
M_020d[occ_list,:,occ_list,:] -= 0.5 * einsum('ijAa,jDai->DA', t1_ccee, v_ceec)
M_020d[:,vir_list,:,vir_list] += einsum('Iabi,Liab->IL', v_ceec, t1_ccee)
M_020d[:,vir_list,:,vir_list] -= 0.5 * einsum('Iabi,Liba->IL', v_ceec, t1_ccee)
M_020d[:,vir_list,:,vir_list] += einsum('Iiab,Labi->IL', t1_ccee, v_ceec)
M_020d[:,vir_list,:,vir_list] -= 0.5 * einsum('Iiab,Lbai->IL', t1_ccee, v_ceec)

# More terms after the ## blocks
M_020e = np.zeros((nocc, nvir, nocc, nvir))
M_020e += 2 * einsum('IiDa,LAai->IDLA', t1_ccee, v_cece)
M_020e -= einsum('IiDa,iAaL->IDLA', t1_ccee, v_cece)
M_020e += 2 * einsum('LiAa,IDai->IDLA', t1_ccee, v_cece)
M_020e -= einsum('LiAa,iDaI->IDLA', t1_ccee, v_cece)
M_020e -= einsum('iIDa,LAai->IDLA', t1_ccee, v_cece)
M_020e -= einsum('iLAa,IDai->IDLA', t1_ccee, v_cece)

M_020f = np.zeros((nocc, nvir, nocc, nvir))
M_020f -= einsum('LiAa,IDai->IDLA', t1_ccee, v_ceec)
M_020f += 0.5 * einsum('LiAa,iDaI->IDLA', t1_ccee, v_ceec)
M_020f += 0.5 * einsum('iLAa,IDai->IDLA', t1_ccee, v_ceec)

# Total
M_total = M_000 + M_010 + M_020 + M_020b + M_020c + M_020d + M_020e + M_020f
M_total_flat = M_total.reshape(nocc*nvir, nocc*nvir)

# PySCF reference
M_pyscf = radc_ee.get_imds(myadc, eris=eris).reshape(nocc*nvir, nocc*nvir)

print(f"|M_total - M_pyscf| = {np.max(np.abs(M_total_flat - M_pyscf)):.2e}")

# Check individual contributions
print(f"\nSection norms:")
print(f"  000 (diag):  {np.linalg.norm(M_000):.6f}")
print(f"  010 (CIS):   {np.linalg.norm(M_010):.6f}")
print(f"  020:         {np.linalg.norm(M_020):.6f}")
print(f"  020b:        {np.linalg.norm(M_020b):.6f}")
print(f"  020c:        {np.linalg.norm(M_020c):.6f}")
print(f"  020d:        {np.linalg.norm(M_020d):.6f}")
print(f"  020e:        {np.linalg.norm(M_020e):.6f}")
print(f"  020f:        {np.linalg.norm(M_020f):.6f}")

# Now build OUR M11 and compare
eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
t2 = t1_ccee  # same as our t2

our_CIS = M_000 + M_010
our_CIS_flat = our_CIS.reshape(nocc*nvir, nocc*nvir)

# Our T2 correction = (1/2) * sum_kc t2[ik,ac]*K(jb,kc) + t2[jk,bc]*K(ia,kc)
# + self-energy
our_T2corr = np.zeros((nocc, nvir, nocc, nvir))
for i in range(nocc):
    for d in range(nvir):
        for l in range(nocc):
            for a in range(nvir):
                val = 0.0
                for k in range(nocc):
                    for c in range(nvir):
                        # (1/2) t2[i,k,d,c] * K(l,a,k,c)
                        K_lakc = 2.0*eri[l,nocc+a,k,nocc+c] - eri[l,nocc+c,k,nocc+a]
                        val += 0.5 * t2[i,k,d,c] * K_lakc
                        # (1/2) t2[l,k,a,c] * K(i,d,k,c)
                        K_idkc = 2.0*eri[i,nocc+d,k,nocc+c] - eri[i,nocc+c,k,nocc+d]
                        val += 0.5 * t2[l,k,a,c] * K_idkc
                our_T2corr[i,d,l,a] = val

our_SE = np.zeros((nocc, nvir, nocc, nvir))
# -delta_ab * Sigma_oo[i,j] + delta_ij * Sigma_vv[a,b]
for i in range(nocc):
    for l in range(nocc):
        sigma_oo = 0.0
        sigma_oo_sym = 0.0
        for k in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    sigma_oo += t2[i,k,a,b]*(2.0*eri[l,nocc+a,k,nocc+b]-eri[l,nocc+b,k,nocc+a])
                    sigma_oo_sym += t2[l,k,a,b]*(2.0*eri[i,nocc+a,k,nocc+b]-eri[i,nocc+b,k,nocc+a])
        for d in range(nvir):
            our_SE[i,d,l,d] -= 0.5*(sigma_oo + sigma_oo_sym)

for d in range(nvir):
    for a in range(nvir):
        sigma_vv = 0.0
        sigma_vv_sym = 0.0
        for ip in range(nocc):
            for jp in range(nocc):
                for c in range(nvir):
                    sigma_vv -= t2[ip,jp,d,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                    sigma_vv_sym -= t2[ip,jp,a,c]*(2.0*eri[ip,nocc+d,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+d])
        for i in range(nocc):
            our_SE[i,d,i,a] += 0.5*(sigma_vv + sigma_vv_sym)

our_M11 = our_CIS + our_T2corr + our_SE
our_M11_flat = our_M11.reshape(nocc*nvir, nocc*nvir)

print(f"\n|our_CIS - PySCF_000+010| = {np.max(np.abs(our_CIS_flat - (M_000+M_010).reshape(-1,nocc*nvir))):.2e}")
print(f"|our_M11 - M_pyscf| = {np.max(np.abs(our_M11_flat - M_pyscf)):.2e}")

# What PySCF adds beyond our formula
pyscf_T2 = (M_020 + M_020b + M_020c + M_020d + M_020e + M_020f).reshape(nocc*nvir, nocc*nvir)
our_T2_total = (our_T2corr + our_SE).reshape(nocc*nvir, nocc*nvir)
print(f"|our_T2+SE - PySCF_020| = {np.max(np.abs(our_T2_total - pyscf_T2)):.2e}")

# The difference should tell us what terms we're missing
diff_T2 = pyscf_T2 - our_T2_total
print(f"\nT2 correction difference norm: {np.linalg.norm(diff_T2):.6f}")
print(f"Our T2 correction norm: {np.linalg.norm(our_T2_total):.6f}")
print(f"PySCF T2 correction norm: {np.linalg.norm(pyscf_T2):.6f}")

# Check: v_cece vs v_ceec
# PySCF: v_cece = eris.ovvo = (ia|bj)? Let's verify
print(f"\neris.ovvo shape: {eris.ovvo.shape}")
# (ia|bj) in chemist: eri[i, nocc+a, nocc+b, j]
i,a,b,j = 0,0,0,0
print(f"eris.ovvo[{i},{a},{b},{j}] = {eris.ovvo[i,a,b,j]:.10f}")
print(f"eri[{i},{nocc+a},{nocc+b},{j}] = {eri[i,nocc+a,nocc+b,j]:.10f}")
i,a,b,j = 1,0,1,2
print(f"eris.ovvo[{i},{a},{b},{j}] = {eris.ovvo[i,a,b,j]:.10f}")
print(f"eri[{i},{nocc+a},{nocc+b},{j}] = {eri[i,nocc+a,nocc+b,j]:.10f}")

# NOTE: In PySCF, v_cece and v_ceec are both eris.ovvo!
# So v_cece[I,i,D,a] = ovvo[I,i,D,a] = (Ii|Da) = eri[I, nocc+i... wait
# PySCF notation: c=occ, e=vir
# v_cece = v[occ,vir,occ,vir] but ovvo has shape (nocc, nvir, nvir, nocc)
# So v_cece[i,a,b,j] where i,j occ, a,b vir = (ia|bj) chemist
# einsum('IiDa,LAai->IDLA', t1_ccee, v_cece):
#   t1_ccee[I,i,D,a] * v_cece[L,A,a,i] = t2[I,i,D,a] * (LA|ai) = t2[I,i,D,a] * eri[L,nocc+A,nocc+a,i]
# This is a contraction over i,a: sum_{i,a} t2[I,i,D,a] * eri[L,A+o,a+o,i]

# Our formula: sum_{k,c} t2[i,k,a,c] * K(j,b,k,c) where K = 2(jb|kc) - (jc|kb)
# = sum_{k,c} t2[i,k,a,c] * [2*eri[j,b+o,k,c+o] - eri[j,c+o,k,b+o]]

# PySCF's first term: 2*einsum('IiDa,LAai->IDLA', t2, ovvo)
# = 2 * sum_{i,a} t2[I,i,D,a] * ovvo[L,A,a,i]
# = 2 * sum_{i,a} t2[I,i,D,a] * eri[L,A+o,a+o,i]
# In our notation: 2 * sum_{k,c} t2[I,k,D,c] * (L,A+o|c+o,k) = 2*(LA|ck) permuted
# = 2 * sum_{k,c} t2[I,k,D,c] * eri[L,A+o,c+o,k]
# But our K has eri[j,b+o,k,c+o] not eri[j,b+o,c+o,k]!
# These differ by (pq|rs) vs (pr|qs)... no, they're the same by 8-fold symmetry:
# (L,A+o|c+o,k) = eri_chemist[L, A+o, c+o, k]
# Our formula uses eri[j,b+o,k,c+o] = (j,b+o|k,c+o) = eri_chemist[j, b+o, k, c+o]
# These are DIFFERENT integrals! (pq|rs) != (ps|rq) in general!
# (LA|ck) != (Lk|cA) in general

# Actually (pq|rs) = ∫ p(1)q(1) 1/r12 r(2)s(2) and (pq|rs) = (rs|pq) by symmetry
# So eri[L,A+o,c+o,k] = (L,A+o|c+o,k) and eri[L,A+o,k,c+o] = (L,A+o|k,c+o)
# These differ: (LA|ck) vs (LA|kc). Only equal if c==k (same index).
# Wait no: (pq|rs) = (pq|rs), and eris.ovvo stores (ia|bj).
# But in PySCF's notation, ovvo[I,a,b,j] means the integral with
# first index=occ I, second=vir a, third=vir b, fourth=occ j.
# In chemist notation: (Ia|bj) = ∫ I(1)a(1) 1/r12 b(2)j(2)
# = eri_full[I, a+nocc, b+nocc, j]

# So PySCF's term: t2[I,i,D,a] * ovvo[L,A,a,i] = t2[I,i,D,a] * (LA|ai)
# Our term: t2[I,k,D,c] * (Lb|kc) with b=A, j=L
# = t2[I,k,D,c] * eri[L, A+o, k, c+o]

# These are: (LA|ai) = eri[L, A+o, a+o, i] vs eri[L, A+o, k, c+o]
# With k=i, c=a: eri[L, A+o, i, a+o] = (L, A+o | i, a+o)
# But PySCF has eri[L, A+o, a+o, i] = (L, A+o | a+o, i)

# (L,A+o | a+o,i) vs (L,A+o | i,a+o):
# By 2-electron integral symmetry: (pq|rs) = (pq|sr)? NO!
# (pq|rs) = ∫ p(1)q(1) 1/r12 r(2)s(2)
# (pq|sr) = ∫ p(1)q(1) 1/r12 s(2)r(2)
# Since r(2)s(2) = s(2)r(2) (product of functions), these ARE equal!
# So (pq|rs) = (pq|sr) = (rs|pq) = (sr|pq) = (qp|rs) = (qp|sr) = (rs|qp) = (sr|qp)

# So eri[L, A+o, a+o, i] = eri[L, A+o, i, a+o]! They're the SAME!
# Great, so PySCF's formula and ours should be equivalent.
# Let me verify:
print(f"\nVerifying ERI symmetry:")
i,j,a,b = 0,1,0,1
print(f"eri[{i},{nocc+a},{nocc+b},{j}] = {eri[i,nocc+a,nocc+b,j]:.10f}")
print(f"eri[{i},{nocc+a},{j},{nocc+b}] = {eri[i,nocc+a,j,nocc+b]:.10f}")
# These should be equal: (i,a+o|b+o,j) = (i,a+o|j,b+o) since (pq|rs)=(pq|sr)

i,j,a,b = 2,3,0,1
print(f"eri[{i},{nocc+a},{nocc+b},{j}] = {eri[i,nocc+a,nocc+b,j]:.10f}")
print(f"eri[{i},{nocc+a},{j},{nocc+b}] = {eri[i,nocc+a,j,nocc+b]:.10f}")
