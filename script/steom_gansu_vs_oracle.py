#!/usr/bin/env python3
"""GANSU 2nd-order analytic STEOM vs the plain-full oracle, on config-mixed systems.

steom_configmix_scan proved plain STEOM is order-2-exact at the 1h1p level even for
strongly config-mixed roots (order-3+ = 0). So a COMPLETE 2nd-order form reaches the
ORCA/plain target. This script measures whether GANSU's ACTUAL analytic form
(build_g_canonical_full, with g_phhp EA/cross ZEROED) falls short specifically on
config-mixed roots -> i.e. is the naphthalene +1.0 eV overshoot a fixable formula gap.

Per singlet root (matched by 1h1p eigenvector overlap):
   E_oracle : det-space plain e^S STEOM  (= ORCA-equivalent target)
   E_gansu  : build_g_canonical_full eigenvalue (shipped GANSU analytic, EA/cross=0)
   gap      : E_gansu - E_oracle   <- the fixable 2nd-order incompleteness
   PR       : participation ratio (config mixing)

Run:  wsl python3 script/steom_gansu_vs_oracle.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_g_canonical_full
Ha = 27.211386245988


def write_xyz(atoms, path):
    lines = atoms.replace(";", "\n").strip().splitlines()
    with open(path, "w") as f:
        f.write(f"{len(lines)}\n\n")
        for ln in lines:
            f.write(ln.strip() + "\n")
    return path


def oracle_plain(atom, basis, ncore):
    data = get_active_data(atom=atom, basis=basis, ncore=ncore)
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    S = build_S(data, dets, index, sIP, sEA)
    Gfull = expm(S) @ Hbar @ expm(-S)
    Gs, _ = project_1h1p(data, dets, index, Gfull)
    w, vr = np.linalg.eig(Gs)
    o = np.argsort(w.real)
    return (w[o].real - E_N) * Ha, vr[:, o], data["nocc"], data["nvir"]


def gansu_analytic(xyzfile, basis, ncore):
    d = CW.load(xyzfile, basis, ncore)
    Ga, *_ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    w, vr = np.linalg.eig(Ga)
    o = np.argsort(w.real)
    return w[o].real * Ha, vr[:, o], d["e_s"] * Ha


def part_ratio(v):
    p = np.abs(v) ** 2; s = p.sum()
    return float(1.0 / np.sum((p / s) ** 2)) if s > 0 else 0.0


def follow(vref, V):
    ov = np.abs(vref.conj() @ V) / (np.linalg.norm(vref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov))


def compare(label, atom, basis="sto-3g", ncore=0, n_report=6):
    xyzf = write_xyz(atom, os.path.join(tempfile.gettempdir(), "cmp.xyz"))
    e_or, V_or, nocc, nvir = oracle_plain(atom, basis, ncore)
    e_ga, V_ga, e_eom = gansu_analytic(xyzf, basis, ncore)
    # sorted-spectrum RMS (root-following-free) — the headline fix metric
    n = min(len(e_or), len(e_ga))
    rms = float(np.sqrt(np.mean((np.sort(e_or)[:n] - np.sort(e_ga)[:n]) ** 2)))
    mx = float(np.max(np.abs(np.sort(e_or)[:n] - np.sort(e_ga)[:n])))
    base_tag = "EE-BASE" if os.environ.get("STEOM_EE_BASE") else "shipped-base"
    print(f"\n=== {label}   nocc={nocc} nvir={nvir}  [{base_tag}]  "
          f"sorted-spectrum RMS={rms:.3f} max={mx:.3f} eV")
    print(f"    {'root':>4} {'E_oracle':>9} {'E_gansu':>9} {'E_eomEE':>9} "
          f"{'gap':>8}  {'PR':>5}  mix")
    nrep = min(n_report, len(e_or))
    for i in range(nrep):
        vo = V_or[:, i]
        jg = follow(vo, V_ga)
        pr = part_ratio(vo)
        gap = e_ga[jg] - e_or[i]
        eom = e_eom[i] if i < len(e_eom) else float("nan")
        mix = "MIXED" if pr >= 1.5 else "single"
        flag = " <=" if pr >= 1.5 and abs(gap) >= 0.3 else ""
        print(f"    {i:>4} {e_or[i]:9.3f} {e_ga[jg]:9.3f} {eom:9.3f} "
              f"{gap:+8.3f}  {pr:5.2f}  {mix}{flag}")


def h4_rect(d=2.0, h=1.4):
    return f"H 0 0 0; H {d} 0 0; H 0 {h} 0; H {d} {h} 0"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "h2o"):
        # single-config control: gap must be ~0 (validated STEOM path)
        compare("H2O sto-3g FC1 (single-config control)",
                open("xyz/H2O.xyz").read().splitlines()[2:5], ncore=1) \
            if False else compare(
                "H2O sto-3g FC1 (single-config control)",
                "; ".join(l.strip() for l in open("xyz/H2O.xyz").read().splitlines()[2:5]),
                ncore=1)
    if which in ("all", "h4"):
        for h in (1.4, 1.8):
            compare(f"H4 rect d=2.0 h={h} (config-mixed present)", h4_rect(2.0, h), ncore=0)
    if which in ("all", "h6"):
        # S != 0 (nontrivial STEOM) + more room for mixing
        compare("H6 rect ladder d=2.0 h=1.4 (S!=0)",
                "; ".join(f"H {2.0*(k%2)} {1.4*(k//2)} 0" for k in range(6)), ncore=0, n_report=9)


if __name__ == "__main__":
    main()
