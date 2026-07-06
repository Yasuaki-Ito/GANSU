#!/usr/bin/env python3
"""Does 2nd-order STEOM systematically miss CONFIG-MIXED valence states?

Decisive experiment for the naphthalene Lb +1.0 eV overshoot (memory
project_dlpno_steom_orca_reconsider). Uses the determinant Fock-space oracle
(steom_fockspace_ref) to build the *plain* STEOM transform G = e^S Hbar e^-S and
decompose it order-by-order in the BCH series:
    G_n = sum_{k<=n} (1/k!) ad_S^k(Hbar)
GANSU's build_W_eff_and_G is a 2nd-order analytic form (no K normal ordering,
confirmed in code 2026-07-06). So per 1h1p singlet root:
    dE(3+) = E_full(expm) - E_order2      [eV]   <- what a 2nd-order form CANNOT reach
and we correlate dE(3+) with the root's CONFIG-MIXING (participation ratio of its
1h1p eigenvector). Hypothesis: single-config roots have dE(3+)~0 (like H2O), but
config-mixed roots (acene La/Lb analog) have large dE(3+) => the overshoot is a
2nd-order method ceiling, not a fixable formula gap.

Run:  wsl python3 script/steom_configmix_scan.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
from scipy.linalg import expm
Ha = 27.211386245988


def singlet_spectrum(data, dets, index, G, E_N):
    """1h1p singlet eigenvalues (eV, ref-shifted) + right eigenvectors (columns)."""
    Gs, _ = project_1h1p(data, dets, index, G)
    w, vr = np.linalg.eig(Gs)
    order = np.argsort(w.real)
    return (w[order].real - E_N) * Ha, vr[:, order]


def part_ratio(v):
    """participation ratio of a (complex) 1h1p vector: 1=single config, >1=mixed."""
    p = np.abs(v) ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(1.0 / np.sum(p ** 2))


def follow(v_ref, V):
    """index of the column of V with max |<v_ref|col>| (root following)."""
    ov = np.abs(v_ref.conj() @ V) / (np.linalg.norm(v_ref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov)), float(ov.max())


def scan(label, n_report=6, orca=None, **akw):
    data = get_active_data(**akw)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    S = build_S(data, dets, index, sIP, sEA)

    # order-2 partial sum G2 = Hbar + [S,Hbar] + 1/2 [S,[S,Hbar]]
    ad1 = S @ Hbar - Hbar @ S
    ad2 = S @ ad1 - ad1 @ S
    G2 = Hbar + ad1 + 0.5 * ad2
    ad3 = S @ ad2 - ad2 @ S
    G3 = G2 + ad3 / 6.0
    Gfull = expm(S) @ Hbar @ expm(-S)

    e2, V2 = singlet_spectrum(data, dets, index, G2, E_N)
    e3, V3 = singlet_spectrum(data, dets, index, G3, E_N)
    ef, Vf = singlet_spectrum(data, dets, index, Gfull, E_N)

    print(f"\n=== {label}   nocc={nocc} nvir={nvir}  1h1p-dim={nocc*nvir}  "
          f"HF={data['Ehf']:.5f} CCSD={data['Eccsd']:.5f} FCI={data['Efci']:.5f}")
    print(f"    {'root':>4} {'E_full':>9} {'E_ord2':>9} {'E_ord3':>9} "
          f"{'dE(3+)':>8} {'dE(4+)':>8}  {'PR':>5}  mix")
    nrep = min(n_report, len(ef))
    for i in range(nrep):
        vf = Vf[:, i]
        j2, o2 = follow(vf, V2)
        j3, o3 = follow(vf, V3)
        pr = part_ratio(vf)
        dE3 = ef[i] - e2[j2]          # order 3+ contribution (2nd-order cannot reach)
        dE4 = ef[i] - e3[j3]          # order 4+ contribution
        mix = "MIXED" if pr >= 1.5 else "single"
        flag = " <=" if pr >= 1.5 and abs(dE3) >= 0.3 else ""
        print(f"    {i:>4} {ef[i]:9.3f} {e2[j2]:9.3f} {e3[j3]:9.3f} "
              f"{dE3:+8.3f} {dE4:+8.3f}  {pr:5.2f}  {mix}{flag}")
    if orca is not None:
        print(f"    ORCA STEOM ref: {orca}")
    return data


def h4_rect(d=2.0, h=1.6):
    """H4 rectangle: two H2 rails length d, separation h. h->d = square (max mixing)."""
    return (f"H 0 0 0; H {d} 0 0; H 0 {h} 0; H {d} {h} 0")


def detect_pi(mf, mol, nfrz_occ_max=None):
    """active pi-CAS: pick the 2 highest-pz occ + 2 lowest-pz vir MOs (planar molecule
    in xy-plane => pi = pz-dominated)."""
    aoslices = mol.aoslice_by_atom()
    # pz AO indices (cart label 'z' with single z, i.e. 'pz' / 'z' component)
    labels = mol.cart_labels() if mol.cart else mol.ao_labels()
    pz = [k for k, lb in enumerate(labels) if lb.strip().endswith(" z") or lb.strip().endswith("pz")]
    C = mf.mo_coeff
    S = mf.get_ovlp()
    # Mulliken-ish pz weight per MO: sum over pz AOs of C*(S C)
    SC = S @ C
    wpz = np.einsum("ki,ki->i", C[pz, :], SC[pz, :])
    nocc = mol.nelectron // 2
    occ = list(range(nocc)); vir = list(range(nocc, C.shape[1]))
    occ_pi = sorted(occ, key=lambda p: -wpz[p])[:2]
    vir_pi = sorted(vir, key=lambda p: -wpz[p])[:2]
    return sorted(occ_pi + vir_pi), wpz


def butadiene_active():
    """s-trans butadiene (roughly planar, xy) + pi-CAS(4,4) via pz detection."""
    from pyscf import gto, scf
    atom = ("C -0.606 0.400 0; C 0.606 -0.400 0; C 1.812 0.140 0; C -1.812 -0.140 0; "
            "H -0.556 1.486 0; H 0.556 -1.486 0; H 1.788 1.226 0; H -1.788 -1.226 0; "
            "H 2.760 -0.400 0; H -2.760 0.400 0")
    mol = gto.M(atom=atom, basis="sto-3g", cart=True, unit="Angstrom")
    mf = scf.RHF(mol); mf.conv_tol = 1e-12; mf.kernel()
    active, wpz = detect_pi(mf, mol)
    print(f"[butadiene] pi-CAS active MOs = {active}  (pz weights "
          f"{np.round(wpz[active],2)})  nocc_tot={mol.nelectron//2}")
    return atom, active


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"

    if which in ("all", "h2o"):
        # single-config baseline: order3+ ~ 0 expected (matches order_scan 11.847)
        scan("H2O sto-3g FC1 (single-config baseline)",
             xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1,
             orca="lowest 11.849 eV (single config)")

    if which in ("all", "h4"):
        # config-mixing sweep: rectangle (h<d) -> square (h=d, max mixing)
        for h in (1.0, 1.4, 1.8, 2.0):
            scan(f"H4 rect d=2.0 h={h}  ({'SQUARE' if h==2.0 else 'rect'})",
                 atom=h4_rect(2.0, h), basis="sto-3g", ncore=0)

    if which in ("all", "h6"):
        scan("H6 chain sto-3g (R=1.8)",
             atom="; ".join(f"H {1.8*i} 0 0" for i in range(6)),
             basis="sto-3g", ncore=0)

    if which in ("all", "butadiene"):
        atom, active = butadiene_active()
        scan("butadiene pi-CAS(4,4) sto-3g (2Ag config-mixed)",
             atom=atom, basis="sto-3g", active=active,
             orca="2Ag is the classic doubly-excited/config-mixed valence state")


if __name__ == "__main__":
    main()
