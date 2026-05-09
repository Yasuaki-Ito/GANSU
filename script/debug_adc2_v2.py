"""
Debug ADC(2): use PySCF matvec to build full matrix, then compare blocks.
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

print(f"nocc={nocc}, nvir={nvir}, sd={sd}, dd={dd}")

myadc = adc.ADC(mf)
myadc.method_type = 'ee'
myadc.verbose = 0
result = myadc.kernel(nroots=5)
e_ee = result[0]
print(f"PySCF ADC(2) eigenvalues: {e_ee}")

# Use radc_ee.matvec
# Check its signature
import inspect
sig = inspect.signature(radc_ee.matvec)
print(f"matvec signature: {sig}")

# Try calling it
# Typically: matvec(adc, M_ij=None, eris=None)
# Returns a function
try:
    imds = radc_ee.get_imds(myadc)
    print(f"imds type: {type(imds)}")
except Exception as ex:
    print(f"get_imds: {ex}")
    imds = None

try:
    mv_fn = radc_ee.matvec(myadc, imds=imds)
    print(f"matvec returned: {type(mv_fn)}")
except Exception as ex:
    print(f"matvec call failed: {ex}")
    try:
        mv_fn = radc_ee.matvec(myadc)
        print(f"matvec(adc) returned: {type(mv_fn)}")
    except Exception as ex2:
        print(f"matvec(adc) also failed: {ex2}")
        mv_fn = None

if mv_fn is not None:
    # Try vector sizes
    for sz in [sd + dd, sd + nocc*(nocc+1)//2*nvir*(nvir+1)//2,
               sd + nocc*nocc*nvir*(nvir+1)//2,
               sd + nocc*(nocc+1)//2*nvir*nvir]:
        try:
            v = np.zeros(sz); v[0] = 1.0
            sv = mv_fn(v)
            print(f"  size {sz} -> output size {len(sv)}")
            vec_size = sz
            break
        except Exception as ex:
            print(f"  size {sz} failed: {type(ex).__name__}: {ex}")

    if vec_size:
        print(f"\nBuilding full matrix ({vec_size}x{vec_size})...")
        M = np.zeros((vec_size, vec_size))
        for col in range(vec_size):
            e = np.zeros(vec_size); e[col] = 1.0
            M[:, col] = mv_fn(e)

        print(f"|M - M^T| = {np.max(np.abs(M - M.T)):.2e}")
        evals = np.linalg.eigvalsh(M)
        print(f"Eigenvalues (first 5): {evals[:5]}")
        print(f"PySCF kernel:          {e_ee[:5]}")
        print(f"Diff: {evals[:5] - e_ee[:5]}")

        # Extract blocks
        dd_actual = vec_size - sd
        print(f"\nsingles_dim={sd}, doubles_dim_actual={dd_actual}, our_dd={dd}")

        M11 = M[:sd, :sd]
        M12 = M[:sd, sd:]
        M21 = M[sd:, :sd]
        D2m = M[sd:, sd:]

        print(f"|M12 - M21^T| = {np.max(np.abs(M12 - M21.T)):.2e}")
        print(f"|D2 off-diag| = {np.max(np.abs(D2m - np.diag(np.diag(D2m)))):.2e}")

        # D2 diagonal values
        d2d = np.diag(D2m)
        print(f"\nD2 diagonal (first 20): {d2d[:20]}")

        # Our D2
        our_D2 = np.zeros(dd)
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        idx = i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
                        our_D2[idx] = eps[nocc+a] + eps[nocc+b] - eps[i] - eps[j]

        if dd_actual == dd:
            print(f"|D2_pyscf - our_D2| = {np.max(np.abs(d2d - our_D2)):.2e}")
            if np.max(np.abs(d2d - our_D2)) > 1e-6:
                # Check ratio
                ratio = d2d / our_D2
                print(f"D2 ratio: {np.unique(np.round(ratio[np.abs(our_D2)>1e-10], 6))}")

        # Print M12 nonzero structure
        print(f"\n=== M12 from PySCF (nonzero, first 30) ===")
        count = 0
        for ke in range(sd):
            k, e = ke // nvir, ke % nvir
            for idx2 in range(dd_actual):
                val = M12[ke, idx2]
                if abs(val) > 1e-10:
                    if dd_actual == dd:
                        i = idx2 // (nocc*nvir*nvir)
                        r = idx2 % (nocc*nvir*nvir)
                        j = r // (nvir*nvir)
                        r2 = r % (nvir*nvir)
                        c = r2 // nvir
                        d = r2 % nvir
                        deltas = []
                        if k==i: deltas.append("k=i")
                        if k==j: deltas.append("k=j")
                        if e==c: deltas.append("e=c")
                        if e==d: deltas.append("e=d")
                        print(f"  M12[({k},{e}),({i},{j},{c},{d})] = {val:+.10f}  {' '.join(deltas)}")
                    else:
                        print(f"  M12[({k},{e}),{idx2}] = {val:+.10f}")
                    count += 1
                    if count >= 30: break
            if count >= 30: break

        # M21 structure
        print(f"\n=== M21 from PySCF (nonzero, first 30) ===")
        count = 0
        for idx2 in range(min(dd_actual, 30)):
            if dd_actual == dd:
                i = idx2 // (nocc*nvir*nvir)
                r = idx2 % (nocc*nvir*nvir)
                j = r // (nvir*nvir)
                r2 = r % (nvir*nvir)
                a = r2 // nvir
                b = r2 % nvir
            for ke in range(sd):
                k, e = ke // nvir, ke % nvir
                val = M21[idx2, ke]
                if abs(val) > 1e-10:
                    if dd_actual == dd:
                        deltas = []
                        if k==i: deltas.append("k=i")
                        if k==j: deltas.append("k=j")
                        if e==a: deltas.append("e=a")
                        if e==b: deltas.append("e=b")
                        # Identify ERI
                        # Try (ac|bj) = eri[nocc+a, nocc+e, nocc+b, j]
                        eri_acbj = eri_mo[nocc+a, nocc+e, nocc+b, j]
                        eri_ajbc = eri_mo[nocc+a, j, nocc+b, nocc+e]
                        eri_kibj = eri_mo[k, i, nocc+b, j]
                        eri_kjbi = eri_mo[k, j, nocc+b, i]
                        print(f"  M21[({i},{j},{a},{b}),({k},{e})] = {val:+.10f}  {' '.join(deltas)}  (ac|bj)={eri_acbj:.6f} (aj|bc)={eri_ajbc:.6f} (ki|bj)={eri_kibj:.6f} (kj|bi)={eri_kjbi:.6f}")
                    else:
                        print(f"  M21[{idx2},({k},{e})] = {val:+.10f}")
                    count += 1
                    if count >= 30: break
            if count >= 30: break

        # Now try building M12 from new formula and compare
        eri = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)

        if dd_actual == dd:
            print(f"\n=== Testing M21 formulas ===")
            # New sigma2: <ab||cj>r1^c_i - <kb||ij>r1^a_k
            # Spatial: -8(ki|bj)r1^a_k + 8(kj|bi)r1^a_k - 8(aj|bc)r1^c_i + 8(ac|bj)r1^c_i
            # Without the overall factor:
            # M21[ijab, ke] = delta(e,a)*[-8(ki|bj)+8(kj|bi)] + delta(k,i)*[-8(aj|be)+8(ae|bj)]
            # Try factor 1 (just the ERIs without normalization)

            # Actually, the spin2spatial conversion includes a normalization factor
            # from the number of spin-orbital contributions. Let me try factor 1 (just eri values)
            for overall_fac in [1.0, 0.5, 0.25, 0.125]:
                test_M21 = np.zeros((dd, sd))
                for I in range(nocc):
                    for J in range(nocc):
                        for A in range(nvir):
                            for B in range(nvir):
                                ijab = I*nocc*nvir*nvir + J*nvir*nvir + A*nvir + B
                                for K in range(nocc):
                                    for E in range(nvir):
                                        ke = K*nvir + E
                                        val = 0.0
                                        if E == A:
                                            val += (-eri[K,I,nocc+B,J] + eri[K,J,nocc+B,I])
                                        if K == I:
                                            val += (-eri[nocc+A,J,nocc+B,nocc+E] + eri[nocc+A,nocc+E,nocc+B,J])
                                        test_M21[ijab, ke] = val * overall_fac

                err = np.max(np.abs(test_M21 - M21))
                if err < 0.1:
                    print(f"  factor {overall_fac}: |err| = {err:.2e}")
                    if err < 1e-6:
                        print(f"  *** MATCH with factor {overall_fac}! ***")

            # Also try with the spin-orbital formula directly
            # <ab||cj>r1^c_i - <kb||ij>r1^a_k in spatial:
            # Sum over spin gives different factors depending on structure
            # Let me try:
            for f1, f2 in [(1,1), (2,-1), (1,-1), (2,2), (-1,2)]:
                test_M21 = np.zeros((dd, sd))
                for I in range(nocc):
                    for J in range(nocc):
                        for A in range(nvir):
                            for B in range(nvir):
                                ijab = I*nocc*nvir*nvir + J*nvir*nvir + A*nvir + B
                                for K in range(nocc):
                                    for E in range(nvir):
                                        ke = K*nvir + E
                                        val = 0.0
                                        if E == A:
                                            val += f1*eri[K,I,nocc+B,J] + f2*eri[K,J,nocc+B,I]
                                        if K == I:
                                            val += f1*eri[nocc+A,J,nocc+B,nocc+E] + f2*eri[nocc+A,nocc+E,nocc+B,J]
                                        test_M21[ijab, ke] = val

                err = np.max(np.abs(test_M21 - M21))
                if err < 0.1:
                    print(f"  (f1={f1},f2={f2}): |err| = {err:.2e}")

            # Try to reverse-engineer from a specific element
            print("\n  Reverse engineering M21 from specific elements:")
            for idx2 in range(min(20, dd)):
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
                    if abs(pval) > 1e-10:
                        # Print all possible ERI values
                        errs = {}
                        if E == A and K != I:  # pure delta(E,A) term
                            e1 = eri[K,I,nocc+B,J]
                            e2 = eri[K,J,nocc+B,I]
                            print(f"  [{I},{J},{A},{B}|{K},{E}] pyscf={pval:+.8f} δ(E=A)  (KI|BJ)={e1:.8f} (KJ|BI)={e2:.8f}  ratio1={pval/e1 if abs(e1)>1e-12 else 'inf':.4f}  ratio2={pval/e2 if abs(e2)>1e-12 else 'inf':.4f}")
                        if K == I and E != A:  # pure delta(K,I) term
                            e1 = eri[nocc+A,J,nocc+B,nocc+E]
                            e2 = eri[nocc+A,nocc+E,nocc+B,J]
                            print(f"  [{I},{J},{A},{B}|{K},{E}] pyscf={pval:+.8f} δ(K=I)  (AJ|BE)={e1:.8f} (AE|BJ)={e2:.8f}  ratio1={pval/e1 if abs(e1)>1e-12 else 'inf':.4f}  ratio2={pval/e2 if abs(e2)>1e-12 else 'inf':.4f}")
