"""
Verify EOM-CCSD sigma vector: implement PySCF's algorithm with explicit
intermediates and compare with PySCF's matvec output.

Usage: python verify_eom_ccsd_sigma.py
"""

import numpy as np
from pyscf import gto, scf, cc, ao2mo
from pyscf.cc import eom_rccsd

# ======================================================================
#  Setup
# ======================================================================
mol = gto.M(
    atom='''
    O  0.000  0.000  0.127
    H  0.000  0.758 -0.509
    H  0.000 -0.758 -0.509
    ''',
    basis='sto-3g',
    cart=True,
    verbose=0
)

mf = scf.RHF(mol).run()
print(f"RHF energy: {mf.e_tot:.10f}")

mycc = cc.CCSD(mf)
mycc.kernel()
print(f"CCSD correlation energy: {mycc.e_corr:.10f}")

nocc = mol.nelectron // 2
nmo = mf.mo_coeff.shape[1]
nvir = nmo - nocc
eps = mf.mo_energy

print(f"nocc={nocc}, nvir={nvir}, nmo={nmo}")

t1 = mycc.t1
t2 = mycc.t2

# MO ERI in chemist notation: eri[p,q,r,s] = (pq|rs)
eri_full = ao2mo.restore(1, ao2mo.full(mf._eri, mf.mo_coeff), nmo)

# Extract ERI blocks in PySCF convention: ovov[i,a,j,b] = (ia|jb)
o, v = slice(0, nocc), slice(nocc, nmo)
ovov = eri_full[o, v, o, v]  # (ia|jb)
oooo = eri_full[o, o, o, o]  # (ij|kl)
ovoo = eri_full[o, v, o, o]  # (ia|jk)
oovv = eri_full[o, o, v, v]  # (ij|ab)
ovvo = eri_full[o, v, v, o]  # (ia|bj)
ovvv = eri_full[o, v, v, v]  # (ia|bc)
vvvv = eri_full[v, v, v, v]  # (ab|cd)

fock = np.diag(eps)
foo = fock[:nocc, :nocc]
fvv = fock[nocc:, nocc:]
fov = fock[:nocc, nocc:]  # zero for canonical MOs


# ======================================================================
#  Build dressed intermediates (following PySCF's make_ee)
# ======================================================================

def make_tau(t2, t1a, t1b):
    """tau(ijab) = t2(ijab) + t1a(i,a)*t1b(j,b) + t1b(j,b)*t1a(i,a) symmetrized"""
    tau = np.einsum('ia,jb->ijab', t1a, t1b)
    tau = tau + tau.transpose(1,0,3,2)
    tau *= 0.5
    tau += t2
    return tau

def make_response_tau(r2, r1, t1):
    """tau2 = r2 + r1*t1 + t1*r1 (response tau, fac=2)"""
    tau = np.einsum('ia,jb->ijab', r1, t1)
    tau = tau + tau.transpose(1,0,3,2)
    tau += r2
    return tau

tau = make_tau(t2, t1, t1)
theta_t = 2*t2 - t2.transpose(0,1,3,2)

# Dressed Fock
# Fov
Fov = fov.copy()
ovov_2m1 = 2*ovov - ovov.transpose(0,3,2,1)  # 2(ia|jb) - (ib|ja)
Fov += np.einsum('nf,menf->me', t1, ovov_2m1)

# Foo (following PySCF's _IMDS.make_ee exactly)
tilab = np.einsum('ia,jb->ijab', t1, t1) * 0.5 + t2
Foo = np.einsum('mief,menf->ni', tilab, ovov_2m1)
# ovoo contribution (ovoo_2m1 = 2*ovoo - ovoo.transpose(2,1,0,3))
ovoo_2m1 = 2*ovoo - ovoo.transpose(2,1,0,3)
Foo += np.einsum('ne,nemi->mi', t1, ovoo_2m1)
Foo += foo + 0.5 * np.einsum('me,ie->mi', Fov + fov, t1)

# Fvv
Fvv = fvv.copy()
Fvv -= np.einsum('mnaf,menf->ae', tilab, ovov_2m1)
Fvv += np.einsum('mf,mfae->ae', t1, ovvv) * 2
Fvv -= np.einsum('mf,meaf->ae', t1, ovvv)
Fvv -= 0.5 * np.einsum('me,ma->ae', Fov + fov, t1)

# Woooo: woOoO[m,n,i,j] = oooo[m,i,n,j] + T corrections
Woooo = oooo.transpose(0,2,1,3).copy()  # (mi|nj) -> W[m,n,i,j]
Woooo += np.einsum('je,nemi->mnij', t1, ovoo)
Woooo += Woooo.transpose(1,0,3,2)  # symmetrize the T1 part
# Wait, that's wrong. Let me re-do following PySCF exactly.
Woooo = oooo.transpose(0,2,1,3).copy()
tmp = np.einsum('je,nemi->mnij', t1, ovoo)
Woooo += tmp + tmp.transpose(1,0,3,2)
Woooo += np.einsum('menf,ijef->mnij', ovov, tau)

# woOoV: Wooov[m,n,i,e] = ovoo[n,e,m,i]^T + T1*ovov
Wooov = np.einsum('if,mfne->mnie', t1, ovov)
Wooov += ovoo.transpose(2,0,3,1)  # ovoo[i,a,j,k] -> [j,i,k,a] reindex

# WoVVo (exchange type): woVVo[m,b,e,j]
WoVVo = -oovv.transpose(0,2,3,1).copy()  # -(ij|ab) -> -[i,a,b,j]
WoVVo += np.einsum('jf,mfbe->mbej', -t1, ovvv)
tmp_vvo = np.einsum('njbf,mfne->mbej', t2, ovov)  # not menf
WoVVo += tmp_vvo
tmp2 = np.einsum('mfne,jf->menj', ovov, t1)
WoVVo += np.einsum('nb,menj->mbej', t1, tmp2)
# Missing ovoo term from make_ee
WoVVo += np.einsum('nb,nemj->mbej', t1, ovoo)

# WoVvO (direct type): woVvO[m,b,e,j]
WoVvO = ovvo.transpose(0,2,1,3).copy()  # (ia|bj) -> [i,b,a,j]
WoVvO += np.einsum('jf,mebf->mbej', t1, ovvv)
WoVvO -= tmp_vvo * 0.5
tmp3 = np.einsum('menf,jf->menj', ovov, t1)
WoVvO -= np.einsum('nb,menj->mbej', t1, tmp3)
# Missing ovoo term from make_ee
WoVvO -= np.einsum('nb,menj->mbej', t1, ovoo)

ovov_sym = 2*ovov - ovov.transpose(0,3,2,1)
WoVvO += np.einsum('njfb,menf->mbej', theta_t, ovov_sym) * 0.5

# woVoO: complex, skip building explicitly — use PySCF's imds
# wvOvV: complex, skip building explicitly — use PySCF's imds

# ======================================================================
#  σ2 computation following PySCF's algorithm
# ======================================================================

def correct_sigma2(r1, r2, imds, use_pyscf_imds=False):
    """Compute σ2 following PySCF's eeccsd_matvec_singlet exactly.
    If use_pyscf_imds=True, use PySCF's intermediates for ALL terms."""

    if use_pyscf_imds:
        _Foo = imds.Foo
        _Fvv = imds.Fvv
        _Woooo = np.asarray(imds.woOoO)
        _WoVVo = np.asarray(imds.woVVo)
        _WoVvO = np.asarray(imds.woVvO)
        _Wooov = np.asarray(imds.woOoV)
    else:
        _Foo = Foo
        _Fvv = Fvv
        _Woooo = Woooo
        _WoVVo = WoVVo
        _WoVvO = WoVvO
        _Wooov = Wooov

    tau2 = make_response_tau(r2, r1, t1)

    # VVVV * tau2
    Hr2 = np.einsum('ijef,aebf->ijab', tau2, vvvv)

    # Woooo * r2
    Hr2 += np.einsum('mnij,mnab->ijab', _Woooo, r2)

    # *= 0.5 (applied to both VVVV and OOOO terms)
    Hr2 *= 0.5

    # Dressed Fock
    Hr2 += np.einsum('be,ijae->ijab', _Fvv, r2)
    Hr2 -= np.einsum('mj,imab->ijab', _Foo, r2)

    # OVVV-based terms
    tmp = np.einsum('meaf,ijef->maij', ovvv, tau2)
    Hr2 -= np.einsum('ma,mbij->ijab', t1, tmp)
    tmp = np.einsum('meaf,me->af', ovvv, r1) * 2
    tmp -= np.einsum('mfae,me->af', ovvv, r1)
    Hr2 += np.einsum('af,ijfb->ijab', tmp, t2)

    # WoVoO * r1 (use PySCF's imds)
    Hr2 -= np.einsum('mbij,ma->ijab', np.asarray(imds.woVoO), r1)

    # WvOvV * r1 (use PySCF's imds)
    Hr2 += np.einsum('ejab,ie->ijab', np.asarray(imds.wvOvV), r1)

    # WoVVo * r2
    tmp = np.einsum('mbej,imea->jiab', _WoVVo, r2)
    Hr2 += tmp
    Hr2 += 0.5 * tmp.transpose(0,1,3,2)

    # WoVvO * theta
    woVvO_combined = _WoVVo * 0.5 + _WoVvO
    theta = 2*r2 - r2.transpose(0,1,3,2)
    Hr2 += np.einsum('mbej,imae->ijab', woVvO_combined, theta)

    # woOoV corrections
    tmp = np.einsum('nmie,me->ni', _Wooov, r1) * 2
    tmp -= np.einsum('mnie,me->ni', _Wooov, r1)
    Hr2 -= np.einsum('ni,njab->ijab', tmp, t2)

    # OVOV corrections
    tmp = np.einsum('mfne,mf->en', ovov, r1) * 2
    tmp -= np.einsum('menf,mf->en', ovov, r1)
    tmp = np.einsum('en,nb->eb', tmp, t1)
    tmp += np.einsum('menf,mnbf->eb', ovov, theta)
    Hr2 -= np.einsum('eb,ijea->jiab', tmp, t2)

    tmp = np.einsum('nemf,imef->ni', ovov, theta)
    Hr2 -= np.einsum('mj,miab->ijba', tmp, t2)

    # tau-ovov correction
    tmp = np.einsum('menf,ijef->mnij', ovov, tau2)
    tau_half = make_tau(t2, t1, t1) * 0.5
    Hr2 += np.einsum('mnij,mnab->ijab', tmp, tau_half)

    # Symmetrize (singlet)
    Hr2 = Hr2 + Hr2.transpose(1,0,3,2)

    return Hr2


# ======================================================================
#  PySCF reference
# ======================================================================
eom = eom_rccsd.EOMEESinglet(mycc)
imds = eom.make_imds()

# ======================================================================
#  Compare hand-built intermediates vs PySCF's imds
# ======================================================================
print("\n" + "="*70)
print("  Intermediate comparison (hand-built vs PySCF imds)")
print("="*70)
Foo_adj = Foo - np.diag(np.diag(foo))
Fvv_adj = Fvv - np.diag(np.diag(fvv))
print(f"Foo diff (raw):      {np.max(np.abs(Foo - imds.Foo)):.2e}")
print(f"Foo diff (diag-sub): {np.max(np.abs(Foo_adj - imds.Foo)):.2e}")
print(f"Fvv diff (raw):      {np.max(np.abs(Fvv - imds.Fvv)):.2e}")
print(f"Fvv diff (diag-sub): {np.max(np.abs(Fvv_adj - imds.Fvv)):.2e}")
print(f"Fov diff:            {np.max(np.abs(Fov - imds.Fov)):.2e}")
print(f"Woooo diff:          {np.max(np.abs(Woooo - imds.woOoO)):.2e}")
# Show Foo element-wise
print("\nFoo (mine) vs imds.Foo:")
for i in range(nocc):
    for j in range(nocc):
        d = Foo[i,j] - imds.Foo[i,j]
        if abs(d) > 1e-10:
            print(f"  [{i},{j}]: mine={Foo[i,j]:.8e} adj={Foo_adj[i,j]:.8e} ref={imds.Foo[i,j]:.8e} diff={d:.2e}")

# WoVVo: imds.woVVo
try:
    ref_WoVVo = np.asarray(imds.woVVo)
    print(f"WoVVo diff:  {np.max(np.abs(WoVVo - ref_WoVVo)):.2e}")
except:
    print(f"WoVVo: PySCF imds.woVVo not available")

# WoVvO: imds.woVvO
try:
    ref_WoVvO = np.asarray(imds.woVvO)
    print(f"WoVvO diff:  {np.max(np.abs(WoVvO - ref_WoVvO)):.2e}")
except:
    print(f"WoVvO: PySCF imds.woVvO not available")

# Wooov
try:
    ref_Wooov = np.asarray(imds.woOoV)
    print(f"Wooov diff:  {np.max(np.abs(Wooov - ref_Wooov)):.2e}")
except Exception as e:
    print(f"Wooov: comparison failed ({e})")

def pyscf_sigma(r1, r2):
    vec = eom_rccsd.amplitudes_to_vector_ee(r1, r2)
    sigma_vec = eom.matvec(vec, imds=imds)
    s1, s2 = eom_rccsd.vector_to_amplitudes_ee(sigma_vec, nmo, nocc)
    return s1, s2


# ======================================================================
#  Test vectors
# ======================================================================
r1_test = np.zeros((nocc, nvir))
r1_test[nocc-1, 0] = 1.0
r2_test = np.zeros((nocc, nocc, nvir, nvir))

r1_test2 = np.zeros((nocc, nvir))
r2_test2 = np.zeros((nocc, nocc, nvir, nvir))
r2_test2[nocc-1, nocc-1, 0, 0] = 1.0

np.random.seed(42)
r1_rand = np.random.randn(nocc, nvir) * 0.1
r2_rand = np.random.randn(nocc, nocc, nvir, nvir) * 0.01
r2_rand = 0.5 * (r2_rand + r2_rand.transpose(1,0,3,2))

def run_test(name, r1, r2, use_pyscf_imds):
    s1_ref, s2_ref = pyscf_sigma(r1, r2)
    s2_my = correct_sigma2(r1, r2, imds, use_pyscf_imds=use_pyscf_imds)
    diff = s2_my - s2_ref
    maxd = np.max(np.abs(diff))
    tag = "MATCH" if maxd < 1e-8 else "MISMATCH"
    print(f"  {name:20s}: max_diff={maxd:.2e}  [{tag}]")
    if maxd > 1e-8:
        # Show worst element
        idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"    worst at {idx}: mine={s2_my[idx]:.8e} ref={s2_ref[idx]:.8e}")

# --- Using ALL PySCF imds ---
print("\n" + "="*70)
print("  σ2 tests with use_pyscf_imds=True (formula validation)")
print("="*70)
run_test("R1-only", r1_test, r2_test, True)
run_test("R2-only", r1_test2, r2_test2, True)
run_test("Random R1+R2", r1_rand, r2_rand, True)

# --- Using hand-built intermediates ---
print("\n" + "="*70)
print("  σ2 tests with use_pyscf_imds=False (hand-built intermediates)")
print("="*70)
run_test("R1-only", r1_test, r2_test, False)
run_test("R2-only", r1_test2, r2_test2, False)
run_test("Random R1+R2", r1_rand, r2_rand, False)

# ======================================================================
#  σ1 verification: our kernel formula vs PySCF
# ======================================================================

def our_sigma1(r1, r2):
    """Compute σ1 using our eom_cc2_sigma1_kernel + T1² shift formulas.
    r2 should already be singlet-symmetrized: r2[ijab]=r2[jiba].
    """
    NO, NV = nocc, nvir
    sigma = np.zeros((NO, NV))

    # f_oo, f_vv: bare orbital energies
    f_oo = np.diag(foo)  # [NO]
    f_vv = np.diag(fvv)  # [NV]

    # T1: -f_oo * r1 (diagonal)
    for i in range(NO):
        sigma[i, :] -= f_oo[i] * r1[i, :]

    # T2: +f_vv * r1 (diagonal)
    for a in range(NV):
        sigma[:, a] += f_vv[a] * r1[:, a]

    # T3: CIS-like: Σ_{m,e} r1[m,e] * [2*(ia|me) - (mi|ae)]
    #   (ia|me) = ovov[i,a,m,e], (mi|ae) = oovv[m,i,a,e]
    sigma += 2.0 * np.einsum('me,iame->ia', r1, ovov) \
           - np.einsum('me,miae->ia', r1, oovv)

    # T1-dep: Σ_m r1[m,a] * Σ_{n,e} t1[n,e] * [-2*ooov[m,i,n,e] + ooov[n,i,m,e]]
    #   ooov[m,i,n,e] = (mi|ne)
    ooov_kernel = -2*eri_full[o,o,o,v] + eri_full[o,o,o,v].transpose(2,1,0,3)
    tmp = np.einsum('ne,mine->mi', t1, ooov_kernel)
    sigma += np.einsum('ma,mi->ia', r1, tmp)

    # T1-dep: Σ_e r1[i,e] * Σ_{m,f} t1[m,f] * [2*vvov[a,e,m,f] - vvov[a,f,m,e]]
    #   vvov[a,e,m,f] = (ae|mf) = ovov[m,f,a,e] (by bra-ket) ... wait
    #   Actually vvov[a,b,i,c] = (ab|ic) in our notation
    #   So vvov[a,e,m,f] = (ae|mf)
    vvov = eri_full[v, v, o, v]  # (ab|ic)
    vvov_kernel = 2*vvov - vvov.transpose(0,3,2,1)  # 2*(ae|mf) - (af|me)
    tmp = np.einsum('mf,aemf->ae', t1, vvov_kernel)
    sigma += np.einsum('ie,ae->ia', r1, tmp)

    # T4: Σ_m r1[m,a] * Σ_{n,e,f} t2[i,n,e,f] * [-(me|nf) + (mf|ne)]
    #   = Σ_m r1[m,a] * Σ_{n,e,f} t2[i,n,e,f] * [-ovov[m,e,n,f] + ovov[m,f,n,e]]
    K = -ovov + ovov.transpose(0,3,2,1)  # [-ovov[m,e,n,f] + ovov[m,f,n,e]]
    tmp = np.einsum('inef,menf->mi', t2, K)
    sigma += np.einsum('ma,mi->ia', r1, tmp)

    # T5: Σ_e r1[i,e] * Σ_{m,n,f} t2[m,n,a,f] * [-(me|nf) + (mf|ne)]
    tmp = np.einsum('mnaf,menf->ae', t2, K)
    sigma += np.einsum('ie,ae->ia', r1, tmp)

    # T6: Σ_{m,n,e} r2[m,n,a,e] * [-(mi|ne) + (me|ni)]
    #   -(mi|ne) = -ooov[m,i,n,e], (me|ni) = ovov_alt?
    #   Actually: (me|ni) = eri_full[m+0, e+nocc, n+0, i+0] which is ovoo[m,e,n,i]
    #   ovoo[m,e,n,i] = (me|ni) but in our block notation: eri_full[o,v,o,o][m,e,n,i]
    #   We use: ooov[n,i,m,e] = (ni|me) = (me|ni) by bra-ket
    ooov_K = -eri_full[o,o,o,v] + eri_full[o,o,o,v].transpose(2,1,0,3)
    # ooov_K[m,i,n,e] = -ooov[m,i,n,e] + ooov[n,i,m,e] = -(mi|ne) + (ni|me) = -(mi|ne) + (me|ni)
    sigma += np.einsum('mnae,mine->ia', r2, ooov_K)

    # T7: 2 Σ_{m,e,f} r2[i,m,e,f] * [(ae|mf) - (af|me)]
    vvov_K = vvov - vvov.transpose(0,3,2,1)  # (ae|mf) - (af|me)
    sigma += 2.0 * np.einsum('imef,aemf->ia', r2, vvov_K)

    # T11: 2 Σ_{m,e} r2[i,m,a,e] * Σ_{n,f} t1[n,f] * [-2*(me|nf) + (mf|ne)]
    #   = 2 Σ_{m,e} r2[i,m,a,e] * Σ_{n,f} t1[n,f] * [-2*ovov[m,e,n,f] + ovov[m,f,n,e]]
    K2 = -2*ovov + ovov.transpose(0,3,2,1)
    tmp = np.einsum('nf,menf->me', t1, K2)
    sigma += 2.0 * np.einsum('imae,me->ia', r2, tmp)

    # T1² shift: shift = 2 * Σ_{m,n,e,f} t1[n,e]*t1[m,f]*[-ovov[m,e,n,f] + 2*ovov[m,f,n,e]]
    K3 = -ovov + 2*ovov.transpose(0,3,2,1)
    C = np.einsum('ne,mf,menf->', t1, t1, K3)
    shift = 2.0 * C
    sigma += shift * r1

    return sigma

def pyscf_sigma1(r1, r2):
    """Get σ1 from PySCF's eeccsd_matvec_singlet."""
    vec = eom_rccsd.amplitudes_to_vector_ee(r1, r2)
    sigma_vec = eom.matvec(vec, imds=imds)
    s1, s2 = eom_rccsd.vector_to_amplitudes_ee(sigma_vec, nmo, nocc)
    return s1

print("\n" + "="*70)
print("  σ1 verification: our kernel formula vs PySCF")
print("="*70)

def run_sigma1_test(name, r1, r2):
    # Symmetrize r2 for singlet
    r2_sym = 0.5 * (r2 + r2.transpose(1,0,3,2))
    s1_ours = our_sigma1(r1, r2_sym)
    s1_ref = pyscf_sigma1(r1, r2_sym)
    diff = s1_ours - s1_ref
    maxd = np.max(np.abs(diff))
    tag = "MATCH" if maxd < 1e-8 else "MISMATCH"
    print(f"  {name:20s}: max_diff={maxd:.2e}  [{tag}]")
    if maxd > 1e-8:
        idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"    worst at {idx}: ours={s1_ours[idx]:.8e} ref={s1_ref[idx]:.8e}")
        # Print all elements for small systems
        if NO * NV <= 30:
            print(f"    Our σ1:")
            for i in range(NO):
                for a in range(NV):
                    print(f"      [{i},{a}]: ours={s1_ours[i,a]:+.8e} ref={s1_ref[i,a]:+.8e} diff={s1_ours[i,a]-s1_ref[i,a]:+.8e}")
    return maxd

NO, NV = nocc, nvir
run_sigma1_test("R1-only", r1_test, r2_test)
run_sigma1_test("R2-only", r1_test2, r2_test2)
run_sigma1_test("Random R1+R2", r1_rand, r2_rand)


# ======================================================================
#  σ1 using PySCF's formula (dressed intermediates)
# ======================================================================

import inspect

# ======================================================================
#  Print FULL eeccsd_matvec_singlet source
# ======================================================================
print("\n" + "="*70)
print("  FULL eeccsd_matvec_singlet source")
print("="*70)
try:
    source = inspect.getsource(eom_rccsd.eeccsd_matvec_singlet)
    print(source)
except Exception as e:
    print(f"  Could not get source: {e}")

# Print make_ee FULL source
print("\n" + "="*70)
print("  FULL make_ee source")
print("="*70)
try:
    source = inspect.getsource(type(imds).make_ee)
    print(source)
except Exception as e:
    print(f"  Could not get make_ee source: {e}")

# ======================================================================
#  Check if imds.Foo includes bare Fock diagonal
# ======================================================================
print("\n" + "="*70)
print("  imds.Foo diagonal analysis")
print("="*70)
_Foo = np.asarray(imds.Foo)
_Fvv = np.asarray(imds.Fvv)
_Fov = np.asarray(imds.Fov)
print("imds.Foo diagonal:", np.diag(_Foo))
print("bare foo diagonal:", np.diag(foo))
print("imds.Foo diag - bare foo:", np.diag(_Foo) - np.diag(foo))
print("imds.Fvv diagonal:", np.diag(_Fvv))
print("bare fvv diagonal:", np.diag(fvv))
print("imds.Fvv diag - bare fvv:", np.diag(_Fvv) - np.diag(fvv))
print("imds.Fov:", _Fov)

# ======================================================================
#  Test with correct vector packing
# ======================================================================
print("\n" + "="*70)
print("  σ1 with eom's own vector packing")
print("="*70)

r1_t = r1_test.copy()
r2_t = np.zeros((nocc, nocc, nvir, nvir))

# Use eom's own packing
vec = eom.amplitudes_to_vector(r1_t, r2_t)
sigma_vec = eom.matvec(vec, imds=imds)
s1_packed, s2_packed = eom.vector_to_amplitudes(sigma_vec, nmo, nocc)
print(f"eom packing: σ1 shape={s1_packed.shape}, σ2 shape={s2_packed.shape}")
print(f"σ1[4,0] = {s1_packed[4,0]:.10f}")

# Compare with ee packing
vec_ee = eom_rccsd.amplitudes_to_vector_ee(r1_t, r2_t)
print(f"vec lengths: eom={len(vec)}, ee={len(vec_ee)}")

# Use eom packing for R2-only test
r1_zero = np.zeros((nocc, nvir))
r2_t2 = np.zeros((nocc, nocc, nvir, nvir))
r2_t2[nocc-1, nocc-1, 0, 0] = 1.0
vec2 = eom.amplitudes_to_vector(r1_zero, r2_t2)
sigma2 = eom.matvec(vec2, imds=imds)
s1_r2, s2_r2 = eom.vector_to_amplitudes(sigma2, nmo, nocc)
print(f"\nR2-only with eom packing:")
for i in range(nocc):
    for a in range(nvir):
        if abs(s1_r2[i,a]) > 1e-10:
            print(f"  σ1[{i},{a}] = {s1_r2[i,a]:+.10e}")

# ======================================================================
#  Correct σ1 formula (all 8 terms from PySCF source)
# ======================================================================

def correct_sigma1(r1, r2):
    """Compute σ1 following PySCF's eeccsd_matvec_singlet exactly.
    r2 should be singlet-symmetrized.
    Uses PySCF imds for dressed intermediates.
    """
    _Foo = np.asarray(imds.Foo)
    _Fvv = np.asarray(imds.Fvv)
    _Fov = np.asarray(imds.Fov)
    _woVVo = np.asarray(imds.woVVo)
    _woVvO = np.asarray(imds.woVvO)
    _woOoV = np.asarray(imds.woOoV)

    theta_r2 = r2 * 2 - r2.transpose(0,1,3,2)

    # Term 1: Fvv * r1
    Hr1  = np.einsum('ae,ie->ia', _Fvv, r1)
    # Term 2: -Foo * r1
    Hr1 -= np.einsum('mi,ma->ia', _Foo, r1)
    # Term 3: +2 * Fov * r2[imae]
    Hr1 += np.einsum('me,imae->ia', _Fov, r2) * 2
    # Term 4: -Fov * r2[imea]
    Hr1 -= np.einsum('me,imea->ia', _Fov, r2)
    # Term 5: +ovvv * theta_r2  (ovvv[m,f,a,e] = (mf|ae))
    #   einsum('mfae,mife->ia', ovvv, theta)
    #   where theta[m,i,f,e] = 2*r2[m,i,f,e] - r2[m,i,e,f]
    Hr1 += np.einsum('mfae,mife->ia', ovvv, theta_r2)
    # Term 6: +2 * (0.5*WoVVo + WoVvO) * r1
    #   einsum('maei,me->ia', combined, r1) * 2
    combined = _woVVo * 0.5 + _woVvO
    Hr1 += np.einsum('maei,me->ia', combined, r1) * 2
    # Term 7: -woOoV * theta_r2
    #   einsum('mnie,mnae->ia', woOoV, theta)
    Hr1 -= np.einsum('mnie,mnae->ia', _woOoV, theta_r2)
    # Term 8: -t1 * (ovov * theta_r2)
    #   tmp = einsum('nemf,imef->ni', ovov, theta)
    #   Hr1 -= einsum('na,ni->ia', t1, tmp)
    tmp = np.einsum('nemf,imef->ni', ovov, theta_r2)
    Hr1 -= np.einsum('na,ni->ia', t1, tmp)

    return Hr1

print("\n" + "="*70)
print("  Correct σ1 (8 terms from PySCF source) vs PySCF matvec")
print("="*70)

def run_correct_sigma1_test(name, r1, r2):
    r2_sym = 0.5 * (r2 + r2.transpose(1,0,3,2))
    s1_formula = correct_sigma1(r1, r2_sym)
    vec = eom.amplitudes_to_vector(r1, r2_sym)
    sigma_vec = eom.matvec(vec, imds=imds)
    s1_ref, _ = eom.vector_to_amplitudes(sigma_vec, nmo, nocc)
    diff = s1_formula - s1_ref
    maxd = np.max(np.abs(diff))
    tag = "MATCH" if maxd < 1e-8 else "MISMATCH"
    print(f"  {name:20s}: max_diff={maxd:.2e}  [{tag}]")
    if maxd > 1e-8 and nocc * nvir <= 30:
        for i in range(nocc):
            for a in range(nvir):
                d = s1_formula[i,a] - s1_ref[i,a]
                if abs(d) > 1e-10:
                    print(f"    [{i},{a}]: formula={s1_formula[i,a]:+.8e} ref={s1_ref[i,a]:+.8e} diff={d:+.8e}")

run_correct_sigma1_test("R1-only", r1_test, r2_test)
run_correct_sigma1_test("R2-only", r1_test2, r2_test2)
run_correct_sigma1_test("Random R1+R2", r1_rand, r2_rand)

# ======================================================================
#  Eigenvalues
# ======================================================================
print("\n" + "="*70)
print("  PySCF EOM-CCSD Excitation Energies")
print("="*70)

ee = eom_rccsd.EOMEESinglet(mycc)
ee.nroots = 5
e_ee, v_ee = ee.kernel()
if isinstance(e_ee, np.ndarray):
    for i, e in enumerate(e_ee):
        print(f"  State {i+1}: {e:.10f}")
else:
    print(f"  State 1: {e_ee:.10f}")
