#!/usr/bin/env python3
"""PARTIAL-ACTIVE-SPACE GATE for the full-projection G (franken-G fix, step 6 debug).

The naphthalene PROJ_FULL run shows a ~uniform +4.5 eV spectrum shift (HOMO-LUMO
diag 0.217 -> 0.402 Ha): the S-dressing diagonal is not reproduced.  All previous
GATES ran with EVERY orbital active (NMo=NO, NMv=NV).  Production has
NMo<<NO / NMv<<NV.  This gate restricts S to a subset of roots on hexatriene
pi-CAS(6,6) (drop the deepest occ root and the highest vir root), builds the
det oracle with the restricted S (truth), and assembles the projection G with
zero-padded amplitudes (which is exactly what the C++ root loops compute).

If this gate FAILS -> the formula/evaluation itself breaks under partial-active
(root-orbital identification in the delta terms).  If it PASSES -> C++ bug.

Run:  wsl python3 script/steom_gphph_partial_gate.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, tempfile
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_gphph_fullgate as FG
import steom_ip_route_derive as IPD
import steom_ea_spinadapt as EA
import steom_cfour_weff as CW
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                 solve_ea, hf_det, so_index, occ_so, vir_so,
                                 project_1h1p)
from steom_cas_verify import polyene, detect_pi

Ha = 27.211386245988


def main():
    atom = polyene(6, distort=0.0)
    active, _ = detect_pi(atom, "sto-3g", 3, 3)
    data = get_active_data(atom=atom, basis="sto-3g", active=active)
    nocc, nvir = data["nocc"], data["nvir"]; nact = data["nact"]
    nso = 2 * nact; dim = nocc * nvir
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sp_det = IPD.extract_sip(sIP, data)
    se_det = EA.extract_spatial_amp(sEA, data)

    # ---- restrict the active roots: drop occ root 0 (deepest) + vir root nvir-1 ----
    act_occ = list(range(1, nocc))            # spatial occ roots kept
    act_vir = list(range(0, nvir - 1))        # spatial vir roots kept
    print(f"partial active: occ roots {act_occ} / {nocc},  vir roots {act_vir} / {nvir}")
    rx = np.zeros((nocc, nocc, nocc, nvir))
    for m in act_occ: rx[m] = sp_det[m]
    ry = np.zeros((nvir, nocc, nvir, nvir))
    for e in act_vir: ry[e] = se_det[e]

    # restricted SO S (zero the dropped roots, both spins)
    spso = np.zeros((nso,) * 4); seso = np.zeros((nso,) * 4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ:
        if (m % nact) in act_occ:
            spso[m] = rec_ip[m]
    for e in vir:
        if (e % nact) - nocc in act_vir:
            seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})
    G = expm(S) @ Hbar @ expm(-S)
    Gs_d, Gt_d = project_1h1p(data, dets, index, G)
    es_det = np.sort(np.linalg.eigvals(Gs_d).real - E_N) * Ha
    et_det = np.sort(np.linalg.eigvals(Gt_d).real - E_N) * Ha

    # ---- analytic side ----
    xyzf = os.path.join(tempfile.gettempdir(), "partgate.xyz")
    lines = [x.strip() for x in atom.split(";")]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    bar = d["bar"]

    Ms_terms, Mc_terms = FG.load_terms()
    arrays = {'Fov': bar["Fov"], 'Xov': bar["Fov"], 'Wooov': bar["Wooov"],
              'Wvovv': bar["Wvovv"], 'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    gph, ghp = FG.build_proj_g(arrays, Ms_terms, Mc_terms, bar["Wovov"], bar["Wovvo"],
                               nocc, nvir)
    Gs, Gt = FG.assemble(bar["Foo_ee"], bar["Fvv_ee"], gph, ghp, nocc, nvir)
    es = np.sort(np.linalg.eigvals(Gs).real) * Ha
    et = np.sort(np.linalg.eigvals(Gt).real) * Ha
    print(f"[partial GATE] max|d-det| singlet={np.max(np.abs(es-es_det)):.4f} eV  "
          f"triplet={np.max(np.abs(et-et_det)):.4f} eV")
    print("   proj:", np.round(es, 4))
    print("   det :", np.round(es_det, 4))
    # diag comparison (HOMO->LUMO analog): element (i=nocc-1,a=0)
    r = (nocc - 1) * nvir + 0
    print(f"   diag(HOMO->LUMO): proj={Gs[r, r]:.6f}  det={Gs_d[r, r]-E_N:.6f}")


if __name__ == "__main__":
    main()
