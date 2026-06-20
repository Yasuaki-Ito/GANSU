#!/usr/bin/env python3
"""
Robust STEOM full-active test with FROZEN CORE.
Freezing inner-valence/core (no clean principal IP) removes the ill-conditioned
R1mat that made steom_full_active_test.py non-reproducible. With ncore chosen so
every correlated occupied has a clean principal IP, STEOM(complete active) must
equal EOM-EE-CCSD bit-for-bit → clean harness to debug the W^eff formula.

Usage: wsl python3 script/steom_fulltest_frozen.py xyz/H2O.xyz sto-3g 4 2
       arg1 xyz  arg2 basis  arg3 n_states  arg4 ncore(frozen)
"""
import sys
import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "script")
from pyscf_steom_feff_reference import read_xyz, build_bar_h, build_g_canonical_full
from steom_full_active_test import principal_roots_ip, principal_roots_ea


def main(xyz_path, basis, n_states, ncore):
    from pyscf import gto, scf, cc, ao2mo
    from pyscf.cc import eom_rccsd
    print("=" * 78)
    print(f"STEOM FROZEN-CORE full-active test  xyz={xyz_path} basis={basis} "
          f"nstates={n_states} ncore={ncore}")
    print("=" * 78)
    mol = gto.M(atom=read_xyz(xyz_path), basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    mycc = cc.CCSD(mf, frozen=ncore); mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc   # correlated space
    mo_c = mf.mo_coeff[:, ncore:]                          # drop frozen cols
    moe_c = mf.mo_energy[ncore:]
    eri_mo = ao2mo.kernel(mol, mo_c, compact=False).reshape(nmo, nmo, nmo, nmo)
    f_oo = np.diag(moe_c[:nocc]); f_vv = np.diag(moe_c[nocc:])
    print(f"  correlated: nocc={nocc} nvir={nvir} nmo={nmo}  E_corr={mycc.e_corr:.10f}")

    bar_h = build_bar_h(eri_mo, t1, t2, f_oo, f_vv, nocc, nvir)
    eom_ee = eom_rccsd.EOMEESinglet(mycc)
    e_ee, _ = eom_ee.kernel(nroots=n_states)
    e_ee = np.atleast_1d(np.asarray(e_ee))

    r1_ip, r2_ip, w_ip, occ_idx, ps_ip = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, w_ea, vir_idx, ps_ea = principal_roots_ea(mycc, nocc, nvir)
    print("  IP %singles:", " ".join(f"{p:.3f}" for p in ps_ip))
    print("  EA %singles:", " ".join(f"{p:.3f}" for p in ps_ea))
    R1mat_ip = np.array([[r1_ip[l][occ_idx[m]] for m in range(nocc)] for l in range(nocc)])
    R1mat_ea = np.array([[r1_ea[l][vir_idx[e]] for e in range(nvir)] for l in range(nvir)])
    print(f"  R1mat_IP cond={np.linalg.cond(R1mat_ip):.3e}  "
          f"R1mat_EA cond={np.linalg.cond(R1mat_ea):.3e}")

    Gf, *_ = build_g_canonical_full(bar_h, r2_ip, r2_ea, r1_ip, r1_ea,
                                    occ_idx, vir_idx, nocc, nvir)
    ef = np.sort(np.real(np.linalg.eigvals(Gf)))
    print(f"\n  {'state':>5} {'EOM-EE':>14} {'STEOM-full':>14} {'Δ(mHa)':>10}")
    maxdev = 0.0
    n_cmp = min(n_states, len(ef), len(e_ee))
    for k in range(n_cmp):
        dev = (ef[k] - e_ee[k]) * 1000.0
        maxdev = max(maxdev, abs(dev))
        print(f"  {k:>5} {e_ee[k]:>14.8f} {ef[k]:>14.8f} {dev:>+10.3f}")
    print(f"  max|Δ| = {maxdev:.3f} mHa  "
          f"{'PASS ✅' if maxdev < 1.0 else 'FAIL ❌ (W^eff formula bug)'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
