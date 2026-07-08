#!/usr/bin/env python3
"""HYBRID CLASS-REPLACEMENT SCAN under PARTIAL active space (production regime).

Finding: the plain-projection diagonal carries almost no relaxation dressing
(hexatriene det: -0.009 vs connected u-dressing -0.049); with a PARTIAL S the
projection diagonal loses the inactive orbitals' relaxation entirely
(naphthalene PROJ_FULL: uniform +4.5 eV shift).  The projection replacement is
therefore only viable on classes whose physics is carried by the ACTIVE-pair
config coupling.  This scan tests, on hexatriene pi-CAS(6,6) with a PARTIAL S
(drop 1 occ + 1 vir root, mimicking production):

   variants (per element class, conn = shipped connected, proj = projection):
     A: all conn                          (baseline)
     B: off=proj, semi+diag conn          (= current GANSU_STEOM_GPHPH_PROJECTION)
     C: off+semi=proj, diag conn          (candidate fix)
     D: all proj                          (= GANSU_STEOM_PROJ_FULL, expected shifted)

target = FULL-S det oracle (the physical STEOM answer), root-followed by
composition (PR tag).  Also prints the PARTIAL-S det oracle for reference.

Run:  wsl python3 script/steom_gphph_hybrid_scan.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile
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
from pyscf_steom_feff_reference import build_g_canonical_full

Ha = 27.211386245988


def _eig(G):
    w, vr = np.linalg.eig(G)
    o = np.argsort(w.real)
    return w[o].real * Ha, vr[:, o]


def _pr(v):
    p = np.abs(v) ** 2; s = p.sum()
    return float(1.0 / np.sum((p / s) ** 2)) if s > 0 else 0.0


def _follow(vref, V):
    ov = np.abs(vref.conj() @ V) / (np.linalg.norm(vref) * np.linalg.norm(V, axis=0) + 1e-30)
    return int(np.argmax(ov))


def main():
    atom = polyene(6, distort=0.0)
    active, _ = detect_pi(atom, "sto-3g", 3, 3)
    data = get_active_data(atom=atom, basis="sto-3g", active=active)
    nocc, nvir, nact = data["nocc"], data["nvir"], data["nact"]
    nso = 2 * nact; dim = nocc * nvir
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sp_det = IPD.extract_sip(sIP, data)
    se_det = EA.extract_spatial_amp(sEA, data)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)

    def det_G(occ_roots, vir_roots):
        spso = np.zeros((nso,) * 4); seso = np.zeros((nso,) * 4)
        for m in occ:
            if (m % nact) in occ_roots: spso[m] = rec_ip[m]
        for e in vir:
            if (e % nact) - nocc in vir_roots: seso[e] = rec_ea[e]
        S = build_S(data, dets, index, {m: spso[m] for m in occ},
                    {e: seso[e] for e in vir})
        G = expm(S) @ Hbar @ expm(-S)
        Gs, _ = project_1h1p(data, dets, index, G)
        return Gs - E_N * np.eye(dim)

    # FULL-S oracle = physical target; PARTIAL-S for reference
    Gs_full = det_G(list(range(nocc)), list(range(nvir)))
    act_occ = list(range(1, nocc)); act_vir = list(range(0, nvir - 1))
    Gs_part = det_G(act_occ, act_vir)
    e_full, V_full = _eig(Gs_full)
    e_part, _ = _eig(Gs_part)
    print("FULL-S det oracle :", np.round(e_full, 3))
    print("PART-S det oracle :", np.round(e_part, 3))

    # ---- analytic pieces, PARTIAL amplitudes (production regime) ----
    xyzf = os.path.join(tempfile.gettempdir(), "hyb.xyz")
    lines = [x.strip() for x in atom.split(";")]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    bar = d["bar"]
    rx = np.zeros((nocc, nocc, nocc, nvir))
    for m in act_occ: rx[m] = sp_det[m]
    ry = np.zeros((nvir, nocc, nvir, nvir))
    for e in act_vir: ry[e] = se_det[e]

    # connected G with the SAME partial amplitudes: restrict r2/r1 lists
    occ_idx_p = [d["occ_idx"][m] for m in range(nocc) if d["occ_idx"][m] in act_occ]
    keep_o = [m for m in range(nocc) if d["occ_idx"][m] in act_occ]
    vir_idx_p = [d["vir_idx"][e] for e in range(nvir) if d["vir_idx"][e] in act_vir]
    keep_v = [e for e in range(nvir) if d["vir_idx"][e] in act_vir]
    Ga, *_ = build_g_canonical_full(
        bar, [d["r2_ip"][m] for m in keep_o], [d["r2_ea"][e] for e in keep_v],
        [d["r1_ip"][m] for m in keep_o], [d["r1_ea"][e] for e in keep_v],
        occ_idx_p, vir_idx_p, nocc, nvir)

    # projection G with partial amplitudes
    Ms_terms, Mc_terms = FG.load_terms()
    arrays = {'Fov': bar["Fov"], 'Xov': bar["Fov"], 'Wooov': bar["Wooov"],
              'Wvovv': bar["Wvovv"], 'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    gph, ghp = FG.build_proj_g(arrays, Ms_terms, Mc_terms, bar["Wovov"], bar["Wovvo"],
                               nocc, nvir)
    Gp, _ = FG.assemble(bar["Foo_ee"], bar["Fvv_ee"], gph, ghp, nocc, nvir)

    def mask_mix(Gc, Gpj, use_proj):
        """use_proj: set of classes ('off','semi','diag') taken from projection."""
        Gx = Gc.copy()
        for i in range(nocc):
            for a in range(nvir):
                r = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        c = j * nvir + b
                        cls = ('off' if (i != j and a != b) else
                               'diag' if (i == j and a == b) else 'semi')
                        if cls in use_proj:
                            Gx[r, c] = Gpj[r, c]
        return Gx

    variants = [("A all-conn      ", set()),
                ("B off-proj      ", {'off'}),
                ("C off+semi-proj ", {'off', 'semi'}),
                ("D all-proj      ", {'off', 'semi', 'diag'})]
    print(f"\n  {'root':>4}  {'PR':>5} {'FULL-S':>8} | " +
          " ".join(f"{nm.split()[0]:>16}" for nm, _ in variants))
    res = {}
    for nm, up in variants:
        res[nm] = _eig(mask_mix(Ga, Gp, up))
    for k in range(dim):
        vo = V_full[:, k]; pr = _pr(vo)
        mix = "*" if pr >= 1.5 else " "
        cells = []
        for nm, _ in variants:
            e_x, V_x = res[nm]
            jx = _follow(vo, V_x)
            cells.append(f"{e_x[jx]:7.3f} ({e_x[jx]-e_full[k]:+.2f})")
        print(f"  {k:>4}{mix} {pr:5.2f} {e_full[k]:8.3f} | " + " ".join(cells))
    for nm, _ in variants:
        e_x, V_x = res[nm]
        es, em = [], []
        for k in range(dim):
            vo = V_full[:, k]
            jx = _follow(vo, V_x)
            (em if _pr(vo) >= 1.5 else es).append(abs(e_x[jx] - e_full[k]))
        print(f"  [{nm}] max|err| single={max(es):.3f} eV  mixed={max(em) if em else 0:.3f} eV")


if __name__ == "__main__":
    main()
