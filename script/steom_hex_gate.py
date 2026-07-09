#!/usr/bin/env python3
"""Second ORCA GATE (続51): synthetic zigzag C6H8 chain, sto-3g, FC6 — strongly
config-mixed pi system (ORCA STEOM singlets 0.317/1.508/3.878/4.153/4.311/6.680 eV,
s177 ~/steom_ref/hex/hex_steom.out, active IP8/EA8).

Light loader (skips the exact EOM-EE downfold of steom_cfour_weff.load) + the
paper-complete g_phhp/g_phph builds (steom_gphhp_paper).  Discriminates the
Eq.(62) ambiguity toggles A/B on a system where cross terms are large.

Run:  wsl python3 script/steom_hex_gate.py   (takes ~10 min: dense IP/EA EOM eig)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_bar_h, build_g_canonical_full
from steom_full_active_test import principal_roots_ip, principal_roots_ea
from steom_gphhp_gate import restrict, assemble, eig_lab, Ha2eV
from steom_gphhp_paper import build_paper_gphhp, build_paper_gphph

ATOM = """
C 0.0000 0.0000 0.0
C 1.2400 0.7000 0.0
C 2.4800 0.0000 0.0
C 3.7200 0.7000 0.0
C 4.9600 0.0000 0.0
C 6.2000 0.7000 0.0
H 0.0000 1.0800 0.0
H 1.2400 -0.3800 0.0
H 2.4800 1.0800 0.0
H 3.7200 -0.3800 0.0
H 4.9600 1.0800 0.0
H 6.2000 -0.3800 0.0
H -1.0200 0.0000 0.0
H 7.2200 0.7000 0.0
"""
ORCA_HEX_S = [0.317, 1.508, 3.878, 4.153, 4.311, 6.680]
# ORCA labels (frozen 6; occ - 6, vir - 22): root1: 15->1(-0.98) 14->2(-0.18) etc.


def light_load(atom, basis="sto-3g", ncore=6):
    from pyscf import gto, scf, cc, ao2mo
    mol = gto.M(atom=atom, basis=basis, cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel()
    # this metallic-ish chain has multiple RHF solutions (ORCA lands 61 mHa lower):
    # follow internal instabilities until stable
    for _ in range(6):
        mo_i, _, stable_i, _ = mf.stability(internal=True, external=False,
                                            return_status=True)
        if stable_i:
            break
        dm = mf.make_rdm1(mo_i, mf.mo_occ)
        mf = scf.RHF(mol); mf.conv_tol = 1e-10; mf.kernel(dm0=dm)
    print("SCF (stable) =", mf.e_tot, " (ORCA -227.30485887)")
    nmo_tot = mf.mo_coeff.shape[1]
    frozen = list(range(ncore))
    mycc = cc.CCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-9; mycc.conv_tol_normt = 1e-7
    mycc.kernel()
    print("E_corr =", mycc.e_corr, " (ORCA -0.521764295)")
    t1, t2 = mycc.t1, mycc.t2
    nocc = mycc.nocc; nmo = mycc.nmo; nvir = nmo - nocc
    active = list(range(ncore, nmo_tot))
    mo_c = mf.mo_coeff[:, active]; moe = mf.mo_energy[active]
    eri = ao2mo.kernel(mol, mo_c, compact=False).reshape(nmo, nmo, nmo, nmo)
    bar = build_bar_h(eri, t1, t2, np.diag(moe[:nocc]), np.diag(moe[nocc:]), nocc, nvir)
    r1_ip, r2_ip, w_ip, occ_idx, ps_ip = principal_roots_ip(mycc, nocc, nvir)
    r1_ea, r2_ea, w_ea, vir_idx, ps_ea = principal_roots_ea(mycc, nocc, nvir)
    print("IP %singles per occ:", np.round(ps_ip, 2))
    print("EA %singles per vir:", np.round(ps_ea, 2))
    return dict(bar=bar, nocc=nocc, nvir=nvir, dim=nocc * nvir,
                r1_ip=r1_ip, r2_ip=r2_ip, r1_ea=r1_ea, r2_ea=r2_ea,
                occ_idx=occ_idx, vir_idx=vir_idx)


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    d0 = light_load(ATOM)
    nocc = d0["nocc"]; nvir = d0["nvir"]
    # active: only %singles-healthy channels (occ 10-15, vir 0-7 per psingles print)
    d = restrict(d0, occ_keep=tuple(range(nocc - 6, nocc)), vir_keep=tuple(range(8)))
    print(f"nocc={nocc} nvir={nvir} active occ={d['occ_idx']} vir={d['vir_idx']}")
    Foo, Fvv = C.build_feff(d)
    target = np.array(ORCA_HEX_S)

    def score(name, ghp, gph):
        Gs, Gt = assemble(d, ghp, gph, Foo, Fvv)
        ws, ims, ls = eig_lab(Gs, nvir)
        rms = np.sqrt(np.mean((ws[:6] - target) ** 2))
        im = f" Im={ims:.3f}" if ims > 1e-3 else ""
        print(f"{name:26s} S: " + " ".join(f"{x:7.3f}" for x in ws[:8])
              + f"  rms6={rms:.4f}{im}")
        print(f"{'':26s} lab " + " ".join(f"{x:>7s}" for x in ls[:8]))
        return rms

    print("ORCA S:", " ".join(f"{x:7.3f}" for x in ORCA_HEX_S),
          "  (root1 lab 15>0-ish mix, see out)")
    # shipped baseline
    _, g_ph_a, g_hp_a, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    score("shipped", g_hp_a, g_ph_a)
    gph_full = build_paper_gphph(d, with_cross=True)
    for ambA in (0, 1):
        for ambB in (0, 1):
            ghp = build_paper_gphhp(d, ambA, ambB, 1)
            score(f"paper A{ambA}B{ambB} + pap-ph", ghp, gph_full)
    ghp = build_paper_gphhp(d, 1, 0, 1)
    score("paper A1B0 + ship-ph", ghp, g_ph_a)


if __name__ == "__main__":
    main()
