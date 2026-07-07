#!/usr/bin/env python3
"""Second-system GATE: hexatriene pi-CAS(6,6) (C1-asymmetric, nocc=nvir=3).
diag/semi = det truth, off-diag = generated projection base+route -> eigenvalues
must match the det oracle STEOM.  Guards against H2O(nvir=2) degeneracies.

Run:  wsl python3 script/steom_gphph_gate2.py
"""
import os, sys, tempfile
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_cas_verify as V
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so, project_1h1p)

Ha = 27.211386245988


def main():
    atom = V.polyene(6, 0.0)
    active, _ = V.detect_pi(atom, "sto-3g", 3, 3)
    data = get_active_data(atom=atom, basis="sto-3g", active=active)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    spso = np.zeros((nso,)*4); seso = np.zeros((nso,)*4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})

    G = expm(S) @ Hbar @ expm(-S)
    Gs, Gt = project_1h1p(data, dets, index, G)
    es_det = np.sort(np.linalg.eigvals(Gs).real - E_N)*Ha
    et_det = np.sort(np.linalg.eigvals(Gt).real - E_N)*Ha

    # analytic bar_h for the same CAS
    xyzf = os.path.join(tempfile.gettempdir(), "hexa.xyz")
    lines = [a.strip() for a in atom.split(";") if a.strip()]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    bar = d["bar"]
    arrays = {'Fov': bar["Fov"], 'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"],
              'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}

    lin = GEN.linear_struct()
    print("deriving quadratic (slow)...")
    quad = GEN.quad_struct()
    allt = lin + quad
    Ms_t = GEN.expand(allt, 0); Mc_t = GEN.expand(allt, 1)
    gen_Ms = GEN.evaluate(Ms_t, arrays, nocc, nvir)
    gen_Mc = GEN.evaluate(Mc_t, arrays, nocc, nvir)
    route_ph = gen_Mc - gen_Ms; route_hp = gen_Mc

    Wovov = bar["Wovov"]; Wovvo = bar["Wovvo"]
    base_ph = np.einsum("kaic->akci", Wovov).copy()
    base_hp = np.einsum("kcbj->bkjc", Wovvo).copy()
    dim = nocc*nvir
    Gs_hyb = (Gs - E_N*np.eye(dim)).copy(); Gt_hyb = (Gt - E_N*np.eye(dim)).copy()
    for i in range(nocc):
        for a in range(nvir):
            r = i*nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j*nvir + b
                    if i != j and a != b:
                        gph = base_ph[a, j, b, i] + route_ph[i, a, j, b]
                        ghp = base_hp[b, j, i, a] + route_hp[i, a, j, b]
                        Gs_hyb[r, c] = 2.0*ghp - gph
                        Gt_hyb[r, c] = -gph
    es_h = np.sort(np.linalg.eigvals(Gs_hyb).real)*Ha
    et_h = np.sort(np.linalg.eigvals(Gt_hyb).real)*Ha
    print("det oracle singlet:", np.round(es_det, 4))
    print("gate       singlet:", np.round(es_h, 4))
    print("det oracle triplet:", np.round(et_det, 4))
    print("gate       triplet:", np.round(et_h, 4))
    print(f"max|gate-det| singlet={np.max(np.abs(es_h-es_det)):.2e} eV  "
          f"triplet={np.max(np.abs(et_h-et_det)):.2e} eV")


if __name__ == "__main__":
    main()
