#!/usr/bin/env python3
"""
Localize the STEOM W^eff formula bug by ELEMENT-WISE comparison of
build_g_canonical_full's G^{1h1p} against the EXACT effective Hamiltonian
  G_exact = R1 · diag(ω_EOM) · R1^{-1}
where (ω_EOM, R1) are EOM-EE-CCSD eigenvalues + singles part of the right
eigenvectors. At complete active space STEOM(G) must equal G_exact, so the
element-wise residual G - G_exact pinpoints the wrong W^eff contribution.

Usage: wsl python3 script/steom_weff_localize.py xyz/H2O.xyz sto-3g 2
       arg1 xyz  arg2 basis  arg3 ncore(frozen)
"""
import sys
import numpy as np
sys.path.insert(0, "script")
from pyscf_steom_feff_reference import read_xyz, build_bar_h, build_g_canonical_full
from steom_full_active_test import principal_roots_ip, principal_roots_ea


def main(xyz_path, basis, ncore):
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd
    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    mycc = cc.CCSD(mf, frozen=ncore); mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    mo_c = mf.mo_coeff[:, ncore:]; moe_c = mf.mo_energy[ncore:]
    eri_mo = ao2mo.kernel(mol, mo_c, compact=False).reshape(nmo, nmo, nmo, nmo)
    f_oo = np.diag(moe_c[:nocc]); f_vv = np.diag(moe_c[nocc:])
    dim = nocc * nvir
    print(f"nocc={nocc} nvir={nvir} singles_dim={dim}")

    bar_h = build_bar_h(eri_mo, t1, t2, f_oo, f_vv, nocc, nvir)

    # ---- EOM-EE-CCSD: eigenvalues + singles part of right eigenvectors ----
    eom = eom_rccsd.EOMEESinglet(mycc)
    e_ee, v_ee = eom.kernel(nroots=dim)
    e_ee = np.atleast_1d(np.asarray(e_ee))
    R1 = np.zeros((dim, dim))           # [ia, state]
    for n in range(dim):
        r1, r2 = eom.vector_to_amplitudes(v_ee[n])   # r1 [nocc,nvir] spin-adapted
        R1[:, n] = np.asarray(r1).reshape(dim)
    print(f"EOM-EE first {dim} ω = " + " ".join(f"{e:.6f}" for e in e_ee[:dim]))
    print(f"R1mat cond = {np.linalg.cond(R1):.3e}")
    G_exact = R1 @ np.diag(e_ee[:dim]) @ np.linalg.inv(R1)

    # ---- triplet EOM-EE → G_t_exact (separates g_phph from g_phhp) ----
    eomT = eom_rccsd.EOMEETriplet(mycc)
    e_t, v_t = eomT.kernel(nroots=dim)
    e_t = np.atleast_1d(np.asarray(e_t))
    R1t = np.zeros((dim, dim))
    for n in range(dim):
        amp = eomT.vector_to_amplitudes(v_t[n])
        r1t = amp[0] if isinstance(amp, (tuple, list)) else amp
        R1t[:, n] = np.asarray(r1t).reshape(dim)
    print(f"triplet R1mat cond = {np.linalg.cond(R1t):.3e}")
    G_t_exact = R1t @ np.diag(e_t[:dim]) @ np.linalg.inv(R1t)

    # ---- STEOM G + separated g_phph / g_phhp from working equations ----
    r1_ip, r2_ip, w_ip, occ_idx, _ = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, w_ea, vir_idx, _ = principal_roots_ea(mycc, nocc, nvir)
    G, g_phph, g_phhp, _uam, _ubm, hp_decomp = build_g_canonical_full(
        bar_h, r2_ip, r2_ea, r1_ip, r1_ea, occ_idx, vir_idx, nocc, nvir)
    hp_base, hp_bmjc, hp_bkje, hp_bmje = hp_decomp

    # exact g_phph/g_phhp on the (i!=j,a!=b) off-diagonal block:
    #   g_phph[a,j,b,i] = -G_t_exact[ia,jb];  g_phhp[b,j,i,a]=(G_s-G_t)/2[ia,jb]
    print("\n  (i,a)|(j,b)  [i!=j,a!=b]   g_phph code/exact      g_phhp code/exact")
    worst_ph = worst_hp = 0.0
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i == j or a == b:
                        continue
                    r = i*nvir+a; c = j*nvir+b
                    ph_code = g_phph[a, j, b, i]
                    ph_ex   = -G_t_exact[r, c]
                    hp_code = g_phhp[b, j, i, a]
                    hp_ex   = (G_exact[r, c] - G_t_exact[r, c]) / 2.0
                    worst_ph = max(worst_ph, abs(ph_code-ph_ex))
                    worst_hp = max(worst_hp, abs(hp_code-hp_ex))
                    print(f"  ({i},{a})|({j},{b})   "
                          f"{ph_code:+.5f}/{ph_ex:+.5f} (Δ{ph_code-ph_ex:+.4f})   "
                          f"{hp_code:+.5f}/{hp_ex:+.5f} (Δ{hp_code-hp_ex:+.4f})")
    print(f"  worst |Δ g_phph| = {worst_ph:.5f}   worst |Δ g_phhp| = {worst_hp:.5f}")

    # ---- g_phhp[b,j,i,a] term decomposition for (i!=j,a!=b) elements ----
    print("\n  g_phhp[b,j,i,a] decomposition (i!=j,a!=b):")
    print("  (i,a)|(j,b)     base     +u_bmjc    +u_bkje   +u_bmje(X)   sum     exact")
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i == j or a == b:
                        continue
                    r = i*nvir+a; c = j*nvir+b
                    ex = (G_exact[r, c] - G_t_exact[r, c]) / 2.0
                    if abs(ex) < 1e-6 and abs(g_phhp[b,j,i,a]) < 1e-6:
                        continue
                    vals = (hp_base[b,j,i,a], hp_bmjc[b,j,i,a],
                            hp_bkje[b,j,i,a], hp_bmje[b,j,i,a])
                    print(f"  ({i},{a})|({j},{b})  {vals[0]:+.5f} {vals[1]:+.5f} "
                          f"{vals[2]:+.5f} {vals[3]:+.5f}  {sum(vals):+.5f} {ex:+.5f}")

    # eigenvalue check
    eg = np.sort(np.real(np.linalg.eigvals(G)))
    ex = np.sort(np.real(np.linalg.eigvals(G_exact)))
    print("\n state   G_exact(=EOM)   STEOM-G     Δ(mHa)")
    for k in range(dim):
        print(f"  {k:>3}  {ex[k]:>13.8f} {eg[k]:>12.8f} {(eg[k]-ex[k])*1000:>+9.3f}")

    # element-wise residual (note: both are effective H over the SAME singles
    # basis ia=i*nvir+a; G_exact eigenvectors = EOM R1, G eigenvectors = STEOM R1.
    # If conventions match they are directly comparable.)
    D = G - G_exact
    print(f"\n‖G - G_exact‖_F = {np.linalg.norm(D):.6e}   max|elem| = {np.abs(D).max():.6e}")
    print("largest residual elements  (i,a)|(j,b)   G        G_exact     Δ:")
    idx = np.argsort(-np.abs(D).ravel())[:12]
    for p in idx:
        r, c = divmod(p, dim)
        i, a = divmod(r, nvir); j, b = divmod(c, nvir)
        print(f"  ({i},{a})|({j},{b})  {G[r,c]:+.6f} {G_exact[r,c]:+.6f}  {D[r,c]:+.6e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
