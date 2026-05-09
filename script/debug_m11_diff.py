"""
Identify the exact M11 difference between our formula and PySCF's.
"""
import numpy as np
from pyscf import gto, scf, adc, ao2mo, lib
from pyscf.adc import radc_ee

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

t2 = myadc.t2[0][:]  # T2[i,j,a,b] = (ia|jb) / denom
eris = myadc.transform_integrals()
eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)

# PySCF M11
M_pyscf = radc_ee.get_imds(myadc, eris=eris)  # shape (nocc*nvir, nocc*nvir)

# Our M11 (CIS + vertex + self-energy)
sd = nocc * nvir
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
                # Vertex correction (1/2)
                for k in range(nocc):
                    for c in range(nvir):
                        K_jbkc = 2.0*eri[j,nocc+b,k,nocc+c] - eri[j,nocc+c,k,nocc+b]
                        val += 0.5*t2[i,k,a,c]*K_jbkc
                        K_iakc = 2.0*eri[i,nocc+a,k,nocc+c] - eri[i,nocc+c,k,nocc+a]
                        val += 0.5*t2[j,k,b,c]*K_iakc
                # Self-energy
                if a == b:
                    s1 = sum(t2[i,k,ap,bp]*(2.0*eri[j,nocc+ap,k,nocc+bp]-eri[j,nocc+bp,k,nocc+ap])
                             for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    s2 = sum(t2[j,k,ap,bp]*(2.0*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                             for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    val -= 0.5*(s1+s2)
                if i == j:
                    s1 = -sum(t2[ip,jp,a,c]*(2.0*eri[ip,nocc+b,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+b])
                              for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    s2 = -sum(t2[ip,jp,b,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                              for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    val += 0.5*(s1+s2)
                our_M11[ia,jb] = val

diff = M_pyscf - our_M11
print(f"|diff| = {np.max(np.abs(diff)):.2e}")

# The difference must be the "extra" terms in PySCF that we don't have.
# These involve 020b, 020d, 020e, 020f which have 1/2 factors and v_ceec.
# Many of these terms look like they're renormalization corrections from the ISR.

# Let's try to express the difference as einsum contractions of t2 with ERIs.
# diff[id,la] represents the missing terms in M11[(i,d),(l,a)] notation.

# PySCF's 020 section uses v_cece = ovvo and v_ceec = ovvo (same!)
# Our formula uses ovov integrals in K() function.
# The difference should be expressible as contractions of t2 with ovvo or similar.

# Let's try: does the difference equal some simple contraction?
# Compute: sum_kc t2[ikdc] * (ovvo - K terms)

# Actually, let me try a direct approach:
# Compute PySCF's T2 correction using their 020 terms exactly
# and see what differs from our formula.

ovvo = eris.ovvo  # (ia|bj) = eri[i, a+nocc, b+nocc, j]
# In terms of our eri_ovov: ovov[i,a,j,b] = eri[i, a+nocc, j, b+nocc]
# ovvo[i,a,b,j] = eri[i, a+nocc, b+nocc, j] = ovov with last two indices swapped

# Build PySCF's T2 correction (020 section only, without b/c/d/e/f)
pyscf_020_only = np.zeros((nocc, nvir, nocc, nvir))
pyscf_020_only += 2 * einsum('IiDa,LAai->IDLA', t2, ovvo)
pyscf_020_only -= einsum('IiDa,iAaL->IDLA', t2, ovvo)
pyscf_020_only += 2 * einsum('LiAa,IDai->IDLA', t2, ovvo)
pyscf_020_only -= einsum('LiAa,iDaI->IDLA', t2, ovvo)
pyscf_020_only -= einsum('iIDa,LAai->IDLA', t2, ovvo)
pyscf_020_only += einsum('iIDa,iAaL->IDLA', t2, ovvo)
pyscf_020_only -= einsum('iLAa,IDai->IDLA', t2, ovvo)
pyscf_020_only += einsum('iLAa,iDaI->IDLA', t2, ovvo)
p020_flat = pyscf_020_only.reshape(sd, sd)

# Our vertex correction (without self-energy)
our_vertex = np.zeros((sd, sd))
for i in range(nocc):
    for d in range(nvir):
        id_ = i*nvir + d
        for l in range(nocc):
            for a in range(nvir):
                la = l*nvir + a
                val = 0.0
                for k in range(nocc):
                    for c in range(nvir):
                        K_lakc = 2.0*eri[l,nocc+a,k,nocc+c] - eri[l,nocc+c,k,nocc+a]
                        val += 0.5*t2[i,k,d,c]*K_lakc
                        K_idkc = 2.0*eri[i,nocc+d,k,nocc+c] - eri[i,nocc+c,k,nocc+d]
                        val += 0.5*t2[l,k,a,c]*K_idkc
                our_vertex[id_,la] = val

print(f"\n|PySCF_020 - our_vertex| = {np.max(np.abs(p020_flat - our_vertex)):.2e}")

# If they're the same, the diff comes from 020b-020f only
# If they differ, there's a formula issue in the vertex correction too

# Check: our self-energy terms vs PySCF's 020c
pyscf_020c = np.zeros((nocc, nvir, nocc, nvir))
occ_list = np.array(range(nocc))
vir_list = np.array(range(nvir))
pyscf_020c[:,vir_list,:,vir_list] -= 2 * einsum('Iiab,Labi->IL', t2, ovvo)
pyscf_020c[:,vir_list,:,vir_list] += einsum('Iiab,Lbai->IL', t2, ovvo)
pyscf_020c[:,vir_list,:,vir_list] -= 2 * einsum('Liab,Iabi->IL', t2, ovvo)
pyscf_020c[:,vir_list,:,vir_list] += einsum('Liab,Ibai->IL', t2, ovvo)
pyscf_020c[occ_list,:,occ_list,:] -= 2 * einsum('ijAa,iDaj->DA', t2, ovvo)
pyscf_020c[occ_list,:,occ_list,:] += einsum('ijAa,jDai->DA', t2, ovvo)
pyscf_020c[occ_list,:,occ_list,:] -= 2 * einsum('ijDa,iAaj->DA', t2, ovvo)
pyscf_020c[occ_list,:,occ_list,:] += einsum('ijDa,jAai->DA', t2, ovvo)
p020c_flat = pyscf_020c.reshape(sd, sd)

# Our self-energy
our_SE = np.zeros((sd, sd))
for i in range(nocc):
    for d in range(nvir):
        id_ = i*nvir + d
        for l in range(nocc):
            for a in range(nvir):
                la = l*nvir + a
                if d == a:
                    s1 = sum(t2[i,k,ap,bp]*(2.0*eri[l,nocc+ap,k,nocc+bp]-eri[l,nocc+bp,k,nocc+ap])
                             for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    s2 = sum(t2[l,k,ap,bp]*(2.0*eri[i,nocc+ap,k,nocc+bp]-eri[i,nocc+bp,k,nocc+ap])
                             for k in range(nocc) for ap in range(nvir) for bp in range(nvir))
                    our_SE[id_,la] -= 0.5*(s1+s2)
                if i == l:
                    s1 = -sum(t2[ip,jp,d,c]*(2.0*eri[ip,nocc+a,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+a])
                              for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    s2 = -sum(t2[ip,jp,a,c]*(2.0*eri[ip,nocc+d,jp,nocc+c]-eri[ip,nocc+c,jp,nocc+d])
                              for ip in range(nocc) for jp in range(nocc) for c in range(nvir))
                    our_SE[id_,la] += 0.5*(s1+s2)

print(f"|PySCF_020c - our_SE| = {np.max(np.abs(p020c_flat - our_SE)):.2e}")

# So the difference must be in 020b + 020d + 020e + 020f
# These are the "extra" ISR renormalization terms
pyscf_extra = (diff - (p020_flat - our_vertex) - (p020c_flat - our_SE))
print(f"\nExtra terms norm: {np.linalg.norm(pyscf_extra):.6f}")
print(f"|diff| total: {np.linalg.norm(diff):.6f}")

# The extra terms in PySCF (020b, 020d, 020e, 020f) are:
# These involve contractions like t2 * v_ceec (= t2 * ovvo) with 1/2 factors
# They represent the "renormalization" or "non-diagonal" ISR corrections

# CONCLUSION: Our M11 is the "diagonal ISR" approximation.
# PySCF includes additional ISR renormalization terms.
# For ADC(2), these are second-order corrections to M11.

# How much does this affect eigenvalues?
dd = nocc**2 * nvir**2
D2 = np.array([eps[nocc+a]+eps[nocc+b]-eps[i]-eps[j]
               for i in range(nocc) for j in range(nocc)
               for a in range(nvir) for b in range(nvir)])

e_ee = result[0]

# Build M12/M21 from PySCF matvec
mv = radc_ee.matvec(myadc, M_ab=M_pyscf, eris=eris)
dim = sd + dd
M_full = np.zeros((dim, dim))
for col in range(dim):
    e = np.zeros(dim); e[col] = 1.0
    M_full[:, col] = mv(e)
M12 = M_full[:sd, sd:]
M21 = M_full[sd:, :sd]

# With our M11
M_ours = np.zeros((dim, dim))
M_ours[:sd, :sd] = our_M11
M_ours[:sd, sd:] = M12
M_ours[sd:, :sd] = M21
M_ours[sd:, sd:] = np.diag(D2)
our_evals = np.sort(np.linalg.eigvals(M_ours).real)

# With PySCF M11
M_pyscf_full = np.zeros((dim, dim))
M_pyscf_full[:sd, :sd] = M_pyscf
M_pyscf_full[:sd, sd:] = M12
M_pyscf_full[sd:, :sd] = M21
M_pyscf_full[sd:, sd:] = np.diag(D2)
pyscf_evals = np.sort(np.linalg.eigvals(M_pyscf_full).real)

print(f"\n=== Eigenvalue comparison ===")
print(f"{'i':>3}  {'Our M11':>14}  {'PySCF M11':>14}  {'PySCF kernel':>14}  {'Our diff':>10}")
for i in range(5):
    print(f"{i:3d}  {our_evals[i]:14.8f}  {pyscf_evals[i]:14.8f}  {e_ee[i]:14.8f}  {our_evals[i]-e_ee[i]:10.2e}")

# Schur complement with our M11 vs PySCF M11
print(f"\n=== Schur complement convergence ===")
for label, m11 in [("PySCF M11", M_pyscf), ("Our M11", our_M11)]:
    omega = 0.0
    for it in range(10):
        inv_oD2 = np.diag(1.0 / (omega - D2))
        M_eff = m11 + M12 @ inv_oD2 @ M21
        eff_evals = np.sort(np.linalg.eigvals(M_eff).real)
        omega_new = eff_evals[0]
        if it < 3 or abs(omega_new - omega) < 1e-8:
            print(f"  {label} iter {it}: {omega:.8f} -> {omega_new:.8f}")
        if abs(omega_new - omega) < 1e-10:
            break
        omega = omega_new
    print(f"  {label} FINAL: {omega:.8f} (PySCF: {e_ee[0]:.8f}, diff: {omega-e_ee[0]:.2e})")
