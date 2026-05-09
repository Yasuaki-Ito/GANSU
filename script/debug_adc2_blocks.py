"""
Debug ADC(2) matrix blocks: build PySCF's full ADC(2) matrix numerically,
extract M11/M12/M21/D2 blocks, and compare with GANSU formulas.

H2O/STO-3G: nocc=5, nvir=2 => singles=10, doubles=100, total=110
"""

import numpy as np
from pyscf import gto, scf, adc, ao2mo

np.set_printoptions(precision=10, linewidth=200, suppress=True)

def main():
    mol = gto.M(
        atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
        basis='sto-3g', verbose=0
    )
    mf = scf.RHF(mol).run()
    nocc = mol.nelectron // 2
    nmo = mf.mo_coeff.shape[1]
    nvir = nmo - nocc
    eps = mf.mo_energy
    sd = nocc * nvir        # singles dim
    dd = nocc**2 * nvir**2  # doubles dim

    print(f"RHF energy: {mf.e_tot:.10f}")
    print(f"nocc={nocc}, nvir={nvir}, singles_dim={sd}, doubles_dim={dd}")
    print(f"Orbital energies: {eps}")

    # --- PySCF ADC(2) reference ---
    myadc = adc.ADC(mf)
    myadc.method_type = 'ee'
    myadc.verbose = 0
    result = myadc.kernel(nroots=5)
    e_ee = result[0]
    print(f"\nPySCF ADC(2) excitation energies (Ha): {e_ee}")

    # --- Build full ADC(2) matrix from PySCF matvec ---
    from pyscf.adc import radc2
    matvec_fn = radc2.get_matvec(myadc)

    # Try to find vector size
    vec_size = None
    for sz in [sd + dd, sd + dd // 2, sd + dd * 2]:
        try:
            v = np.zeros(sz)
            v[0] = 1.0
            sv = matvec_fn(v)
            if len(sv) == sz:
                vec_size = sz
                print(f"PySCF vector size: {vec_size}")
                break
        except:
            pass

    if vec_size is None:
        # Try from eigenvector
        v_ee = result[1]
        if isinstance(v_ee, (list, tuple)):
            vec_size = len(v_ee[0])
        else:
            vec_size = v_ee.shape[-1]
        print(f"PySCF vector size (from eigvec): {vec_size}")

    print(f"Expected: singles({sd}) + doubles({dd}) = {sd + dd}")
    print(f"Actual PySCF vector size: {vec_size}")

    # Build full matrix
    M_full = np.zeros((vec_size, vec_size))
    for col in range(vec_size):
        e_col = np.zeros(vec_size)
        e_col[col] = 1.0
        M_full[:, col] = matvec_fn(e_col)

    sym_err = np.max(np.abs(M_full - M_full.T))
    print(f"\nFull matrix symmetry: max|M - M^T| = {sym_err:.2e}")

    # Eigenvalues of full matrix
    evals_full = np.linalg.eigvalsh(M_full)
    print(f"\nFull matrix eigenvalues (lowest 10):")
    for i in range(min(10, len(evals_full))):
        print(f"  {i}: {evals_full[i]:.10f}")

    print(f"\nPySCF kernel eigenvalues:")
    for i, e in enumerate(e_ee):
        print(f"  {i}: {e:.10f}")

    # --- Extract blocks ---
    # PySCF vector layout: [r1(sd), r2(dd_pyscf)]
    dd_pyscf = vec_size - sd
    print(f"\nPySCF doubles dim: {dd_pyscf}")
    print(f"Our doubles dim:   {dd}")

    M11_p = M_full[:sd, :sd]
    M12_p = M_full[:sd, sd:]
    M21_p = M_full[sd:, :sd]
    D2_p  = M_full[sd:, sd:]

    print(f"\n--- Block shapes ---")
    print(f"M11: {M11_p.shape}")
    print(f"M12: {M12_p.shape}")
    print(f"M21: {M21_p.shape}")
    print(f"D2:  {D2_p.shape}")

    # M12 = M21^T check
    if M12_p.shape[1] == M21_p.shape[0]:
        m12_m21t = np.max(np.abs(M12_p - M21_p.T))
        print(f"\n|M12 - M21^T| = {m12_m21t:.2e}")

    # D2 diagonal check
    D2_offdiag = np.max(np.abs(D2_p - np.diag(np.diag(D2_p))))
    print(f"|D2 off-diagonal| = {D2_offdiag:.2e}")

    # --- Build OUR M11 ---
    eri_mo = ao2mo.full(mf._eri, mf.mo_coeff, compact=False).reshape(nmo,nmo,nmo,nmo)
    o, v = slice(0, nocc), slice(nocc, nmo)

    t2 = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2[i,j,a,b] = eri_mo[i, nocc+a, j, nocc+b] / (eps[i]+eps[j]-eps[nocc+a]-eps[nocc+b])

    # Build M11 using einsum
    # sigma^a_i = -2 f_ji r^a_j + 2 f_ab r^b_i
    #           + 2 r^b_j [2(ai|jb) - (ab|ji)]
    #           + 2 t2^{bc}_{ik} r^a_j [(jb|kc) - (jc|kb)]
    #           + 2 t2^{ac}_{jk} r^b_i [-(jb|kc) + (jc|kb)]
    #           + 4 t2^{ac}_{ik} r^b_j [2(jb|kc) - (jc|kb)]
    f_oo = np.diag(eps[:nocc])
    f_vv = np.diag(eps[nocc:])
    eri_ovov = eri_mo[o, v, o, v]  # (ia|jb)

    our_M11 = np.zeros((sd, sd))
    for i in range(nocc):
        for a in range(nvir):
            ia = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    jb = j * nvir + b
                    val = 0.0

                    # CIS terms
                    if a == b: val += -2.0 * f_oo[j, i]
                    if i == j: val += 2.0 * f_vv[a, b]
                    val += 2.0 * (2.0 * eri_mo[nocc+a, i, j, nocc+b] - eri_mo[nocc+a, nocc+b, j, i])

                    # T2 correction term 4: delta_{ab} part
                    if a == b:
                        for k in range(nocc):
                            for c in range(nvir):
                                for bp in range(nvir):
                                    val += 2.0 * t2[i,k,bp,c] * (eri_ovov[j,bp,k,c] - eri_mo[j, nocc+c, k, nocc+bp])

                    # T2 correction term 5: delta_{ij} part
                    if i == j:
                        for jp in range(nocc):
                            for kp in range(nocc):
                                for c in range(nvir):
                                    val += 2.0 * t2[jp,kp,a,c] * (-eri_ovov[jp,b,kp,c] + eri_mo[jp, nocc+c, kp, nocc+b])

                    # T2 correction term 6: full contraction
                    for k in range(nocc):
                        for c in range(nvir):
                            val += 4.0 * t2[i,k,a,c] * (2.0 * eri_ovov[j,b,k,c] - eri_mo[j, nocc+c, k, nocc+b])

                    our_M11[ia, jb] = val

    print(f"\n--- M11 Comparison ---")
    print(f"Our M11 symmetric? max|M11 - M11^T| = {np.max(np.abs(our_M11 - our_M11.T)):.2e}")
    m11_err = np.max(np.abs(our_M11 - M11_p))
    print(f"|our_M11 - pyscf_M11| = {m11_err:.2e}")

    if m11_err > 1e-6:
        print("  M11 MISMATCH! Largest differences:")
        diff = our_M11 - M11_p
        flat_idx = np.argsort(np.abs(diff).ravel())[::-1][:10]
        for fi in flat_idx:
            ia, jb = fi // sd, fi % sd
            i, a = ia // nvir, ia % nvir
            j, b = jb // nvir, jb % nvir
            if abs(diff[ia, jb]) > 1e-8:
                print(f"    [{i},{a}|{j},{b}]: ours={our_M11[ia,jb]:.8f} pyscf={M11_p[ia,jb]:.8f} diff={diff[ia,jb]:.2e}")
    else:
        print("  M11 matches PySCF!")

    # --- D2 comparison ---
    print(f"\n--- D2 Comparison ---")
    d2_diag_p = np.diag(D2_p)
    if dd_pyscf == dd:
        our_D2 = np.zeros(dd)
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        idx = i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
                        our_D2[idx] = eps[nocc+a] + eps[nocc+b] - eps[i] - eps[j]
        d2_err = np.max(np.abs(our_D2 - d2_diag_p))
        print(f"|our_D2 - pyscf_D2_diag| = {d2_err:.2e}")

        if d2_err > 1e-6:
            # Check if PySCF uses different ordering
            our_D2_sorted = np.sort(our_D2)
            pyscf_D2_sorted = np.sort(d2_diag_p)
            d2_sorted_err = np.max(np.abs(our_D2_sorted - pyscf_D2_sorted))
            print(f"|sorted D2 error| = {d2_sorted_err:.2e}")

            # Print first few
            print("  First 10 D2 values:")
            for idx in range(min(10, dd)):
                i = idx // (nocc * nvir * nvir)
                rem = idx % (nocc * nvir * nvir)
                j = rem // (nvir * nvir)
                rem2 = rem % (nvir * nvir)
                a = rem2 // nvir
                b = rem2 % nvir
                print(f"    [{i},{j},{a},{b}]: ours={our_D2[idx]:.8f} pyscf={d2_diag_p[idx]:.8f}")

    # --- M12 inspection ---
    print(f"\n--- M12 from PySCF (nonzero elements) ---")
    count = 0
    for ke in range(sd):
        k, e = ke // nvir, ke % nvir
        for idx2 in range(dd_pyscf):
            if abs(M12_p[ke, idx2]) > 1e-10:
                if dd_pyscf == dd:
                    i = idx2 // (nocc * nvir * nvir)
                    rem = idx2 % (nocc * nvir * nvir)
                    j = rem // (nvir * nvir)
                    rem2 = rem % (nvir * nvir)
                    c = rem2 // nvir
                    d = rem2 % nvir
                    print(f"  M12[({k},{e}), ({i},{j},{c},{d})] = {M12_p[ke, idx2]:.10f}")
                else:
                    print(f"  M12[({k},{e}), {idx2}] = {M12_p[ke, idx2]:.10f}")
                count += 1
                if count > 40:
                    print("  ... (truncated)")
                    break
        if count > 40:
            break

    # --- Build OUR M12 from GANSU formulas ---
    # From ADC2_RHF.md sigma1 factored:
    # +4 sum_{j,bc} r^{bc}_{ij} [(ab|jc) - (ac|jb)]
    # This means: sigma^a_i gets contribution from r2^{bc}_{ij}
    # M12[ia, ijbc] += 4 * [(a,b|j,c) - (a,c|j,b)]  (ONLY when outer i matches)
    if dd_pyscf == dd:
        our_M12 = np.zeros((sd, dd))
        for K in range(nocc):
            for E in range(nvir):
                ke = K * nvir + E
                for I in range(nocc):
                    for J in range(nocc):
                        for C in range(nvir):
                            for D in range(nvir):
                                ijcd = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
                                # sigma^E_K += 4 * sum_{J,CD} r2^{CD}_{KJ} * [(E,C|J,D) - (E,D|J,C)]
                                if K == I:
                                    our_M12[ke, ijcd] = 4.0 * (
                                        eri_mo[nocc+E, nocc+C, J, nocc+D] -
                                        eri_mo[nocc+E, nocc+D, J, nocc+C]
                                    )

        m12_err = np.max(np.abs(our_M12 - M12_p))
        print(f"\n--- M12 Comparison ---")
        print(f"|our_M12 - pyscf_M12| = {m12_err:.2e}")

        if m12_err > 1e-6:
            print("  M12 MISMATCH! Largest differences:")
            diff12 = our_M12 - M12_p
            flat_idx = np.argsort(np.abs(diff12).ravel())[::-1][:10]
            for fi in flat_idx:
                ke = fi // dd
                idx2 = fi % dd
                k, e = ke // nvir, ke % nvir
                i = idx2 // (nocc * nvir * nvir)
                rem = idx2 % (nocc * nvir * nvir)
                j = rem // (nvir * nvir)
                rem2 = rem % (nvir * nvir)
                c = rem2 // nvir
                d = rem2 % nvir
                if abs(diff12[ke, idx2]) > 1e-8:
                    print(f"    M12[({k},{e}),({i},{j},{c},{d})]: ours={our_M12[ke,idx2]:.8f} pyscf={M12_p[ke,idx2]:.8f} diff={diff12[ke,idx2]:.2e}")

            # Check if M12_p = M21_p^T (should be true for Hermitian)
            # and show what PySCF M21 looks like in terms of our indices
            print("\n  Analyzing PySCF M12 structure...")
            # For each nonzero, identify which delta structure it corresponds to
            for ke in range(min(3, sd)):
                k, e = ke // nvir, ke % nvir
                for idx2 in range(dd):
                    if abs(M12_p[ke, idx2]) > 1e-10:
                        i = idx2 // (nocc * nvir * nvir)
                        rem = idx2 % (nocc * nvir * nvir)
                        j = rem // (nvir * nvir)
                        rem2 = rem % (nvir * nvir)
                        c = rem2 // nvir
                        d = rem2 % nvir

                        # Identify which delta contributes
                        deltas = []
                        if k == i: deltas.append(f"δ(k={k},i={i})")
                        if k == j: deltas.append(f"δ(k={k},j={j})")
                        if e == c: deltas.append(f"δ(e={e},c={c})")
                        if e == d: deltas.append(f"δ(e={e},d={d})")
                        delta_str = " ".join(deltas) if deltas else "NO DELTA"
                        print(f"    M12_pyscf[({k},{e}),({i},{j},{c},{d})] = {M12_p[ke,idx2]:+.8f}  {delta_str}")
        else:
            print("  M12 matches PySCF!")

    # --- Build OUR M21 from GANSU formulas ---
    # From ADC2_RHF.md sigma2 unfactored:
    # +4(ia|jb) r1^b_j - 2(ib|ja) r1^b_j - 4(ka|jb) r1^a_k + 2(kb|ja) r1^a_k
    # For sigma^{CD}_{IJ} from r1^E_K:
    # We need to figure out: which terms contribute and with what delta structure
    if dd_pyscf == dd:
        our_M21 = np.zeros((dd, sd))
        for I in range(nocc):
            for J in range(nocc):
                for C in range(nvir):
                    for D in range(nvir):
                        ijcd = I*nocc*nvir*nvir + J*nvir*nvir + C*nvir + D
                        for K in range(nocc):
                            for E in range(nvir):
                                ke = K * nvir + E
                                val = 0.0
                                # sigma^{ab}_{ij} = terms with r1
                                # Term 1: +4(ia|jb) r1^b_j
                                # In sigma^{CD}_{IJ}: i=I, a=C, j=?, b=?
                                # r1^b_j: b=E, j=K
                                # So: (I, C+nocc | K, E+nocc) * delta(b=E, j=K) ... but j is external!
                                # Wait: j and b in the original equation are the FREE indices
                                # of sigma^{ab}_{ij}. The term "+4(ia|jb) r1^b_j" means:
                                # The FREE b and j of sigma^{ab}_{ij} are used in r1^b_j.
                                # So for sigma^{CD}_{IJ}: a=C, b=D, i=I, j=J
                                # r1^b_j = r1^D_J, which means K=J and E=D
                                if K == J and E == D:
                                    val += 4.0 * eri_mo[I, nocc+C, J, nocc+D]

                                # Term 2: -2(ib|ja) r1^b_j
                                # i=I, b=D, j=J, a=C, r1^b_j = r1^D_J => K=J, E=D
                                if K == J and E == D:
                                    val += -2.0 * eri_mo[I, nocc+D, J, nocc+C]

                                # Term 3: -4(ka|jb) r1^a_k
                                # In sigma^{CD}_{IJ}: a=C, b=D, i=I, j=J
                                # r1^a_k = r1^C_K, k=K in (ka|jb) => E=C
                                # j=J, b=D are free from sigma
                                if E == C:
                                    val += -4.0 * eri_mo[K, nocc+C, J, nocc+D]

                                # Term 4: +2(kb|ja) r1^a_k
                                # k=K, b=D, j=J, a=C, r1^a_k = r1^C_K => E=C
                                if E == C:
                                    val += 2.0 * eri_mo[K, nocc+D, J, nocc+C]

                                our_M21[ijcd, ke] = val

        m21_err = np.max(np.abs(our_M21 - M21_p))
        print(f"\n--- M21 Comparison ---")
        print(f"|our_M21 - pyscf_M21| = {m21_err:.2e}")

        if m21_err > 1e-6:
            print("  M21 MISMATCH!")
            diff21 = our_M21 - M21_p
            flat_idx = np.argsort(np.abs(diff21).ravel())[::-1][:10]
            for fi in flat_idx:
                ijcd = fi // sd
                ke = fi % sd
                k, e = ke // nvir, ke % nvir
                i = ijcd // (nocc * nvir * nvir)
                rem = ijcd % (nocc * nvir * nvir)
                j = rem // (nvir * nvir)
                rem2 = rem % (nvir * nvir)
                c = rem2 // nvir
                d = rem2 % nvir
                if abs(diff21[ijcd, ke]) > 1e-8:
                    print(f"    M21[({i},{j},{c},{d}),({k},{e})]: ours={our_M21[ijcd,ke]:.8f} pyscf={M21_p[ijcd,ke]:.8f}")

        # Check M12 = M21^T (Hermiticity)
        our_m12_m21t = np.max(np.abs(our_M12 - our_M21.T))
        print(f"\nOur |M12 - M21^T| = {our_m12_m21t:.2e}")

        # Build our full matrix and compare eigenvalues
        print(f"\n--- Full matrix eigenvalue comparison ---")
        our_M_full = np.zeros((sd + dd, sd + dd))
        our_M_full[:sd, :sd] = our_M11
        our_M_full[:sd, sd:] = our_M12
        our_M_full[sd:, :sd] = our_M21
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    for b in range(nvir):
                        idx = i*nocc*nvir*nvir + j*nvir*nvir + a*nvir + b
                        our_M_full[sd + idx, sd + idx] = eps[nocc+a] + eps[nocc+b] - eps[i] - eps[j]

        our_evals = np.linalg.eigvalsh(our_M_full)
        pyscf_evals = evals_full

        print(f"{'i':>3}  {'Our eigenvalue':>16}  {'PySCF eigenvalue':>16}  {'Diff':>12}")
        for i in range(min(10, len(our_evals))):
            pe = pyscf_evals[i] if i < len(pyscf_evals) else float('nan')
            print(f"{i:3d}  {our_evals[i]:16.10f}  {pe:16.10f}  {our_evals[i]-pe:12.2e}")


if __name__ == '__main__':
    main()
