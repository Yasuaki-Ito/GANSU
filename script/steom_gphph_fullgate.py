#!/usr/bin/env python3
"""FULL-PROJECTION G eigenvalue GATE (step 3 of the franken-G fix).

Assembles the STEOM G with projection base+route on ALL element classes
(off/semi/diag) and no F_eff s-dressing:

  Gs[r,c] = d_ij Fvvb[a,b] - d_ab Foob[i,j]
            + 2*(base_hp + route_mc)[b,j,i,a] - (base_ph + route_mc - route_ms)[a,j,b,i]
  Gt[r,c] = d_ij Fvvb - d_ab Foob - (base_ph + route_mc - route_ms)

  base_ph = Wovov[k,a,i,c]->[a,k,c,i],  base_hp = Wovvo[k,c,b,j]->[b,k,j,c] (EE fix)
  routes  = full-class spatial terms (script/gphph_projection_full_terms.json,
            derived+verified machine-exact in steom_gphph_diagsemi.py)
  Foob/Fvvb = fN blocks (Fermi-NO 1-body of Hbar).  bar_h identification is
            probed numerically here (Loo/Lvv + t1*Fov-type corrections).

GATES:
  1. H2O FC1: base F identification + element-wise + eigenvalues vs det oracle
  2. hexatriene pi-CAS(6,6): eigenvalues vs det oracle (config-mixed, the
     naphthalene analog; this is where franken-G broke root1)
Both singlet and triplet.  Amplitudes: det gauge (gauge-consistent with oracle).

Run:  wsl python3 script/steom_gphph_fullgate.py
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
import steom_gphph_diagsemi as DS
import steom_ip_route_derive as IPD
import steom_ea_spinadapt as EA
import steom_cfour_weff as CW
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                 solve_ea, hf_det, so_index, occ_so, vir_so,
                                 project_1h1p)
from steom_cas_verify import polyene, detect_pi
from pyscf_steom_feff_reference import build_g_canonical_full

Ha = 27.211386245988


def load_terms():
    blob = json.load(open("script/gphph_projection_full_terms.json"))
    def parse(lst):
        out = []
        for t in lst:
            ops = tuple((arr, tuple((nm, ('o' if (nm in ('i', 'j') or nm in 'klmn')
                                          else 'v')) for nm in toks))
                        for arr, toks in t["ops"])
            out.append((t["coeff"], ops))
        return out
    return parse(blob["ms"]), parse(blob["mc"])


def assemble(Foob, Fvvb, gph, ghp, nocc, nvir):
    dim = nocc * nvir
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    # NB non-Hermitian Hbar: the oo one-body enters as -F[j,i]
                    # (ket hole is the ROW of Foo; cf. PySCF Hr1 -= 'mi,ma->ia').
                    f = (Fvvb[a, b] if i == j else 0.0) - (Foob[j, i] if a == b else 0.0)
                    Gs[r, c] = f + 2.0 * ghp[b, j, i, a] - gph[a, j, b, i]
                    Gt[r, c] = f - gph[a, j, b, i]
    return Gs, Gt


def build_proj_g(bar_arrays, Ms_terms, Mc_terms, Wovov, Wovvo, nocc, nvir):
    """projection g_phph[a,j,b,i], g_phhp[b,j,i,a] (base + full-class route)."""
    ms = GEN.evaluate(Ms_terms, bar_arrays, nocc, nvir)
    mc = GEN.evaluate(Mc_terms, bar_arrays, nocc, nvir)
    base_ph = np.einsum("kaic->akci", Wovov).copy()
    base_hp = np.einsum("kcbj->bkjc", Wovvo).copy()
    gph = base_ph.copy(); ghp = base_hp.copy()
    route_ph = mc - ms; route_hp = mc
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    gph[a, j, b, i] += route_ph[i, a, j, b]
                    ghp[b, j, i, a] += route_hp[i, a, j, b]
    return gph, ghp


def run_system(tag, xyz=None, atom=None, basis="sto-3g", ncore=None, active=None,
               probe_fbase=False):
    print(f"\n================ {tag} ================")
    data = get_active_data(xyz=xyz, basis=basis, ncore=(ncore if ncore is not None else 2),
                           atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]; nact = data["nact"]
    nso = 2 * nact; dim = nocc * nvir
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sp_det = IPD.extract_sip(sIP, data)
    se_det = EA.extract_spatial_amp(sEA, data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    spso = np.zeros((nso,) * 4); seso = np.zeros((nso,) * 4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})
    G = expm(S) @ Hbar @ expm(-S)
    Gs_d, Gt_d = project_1h1p(data, dets, index, G)
    es_det = np.sort(np.linalg.eigvals(Gs_d).real - E_N) * Ha
    et_det = np.sort(np.linalg.eigvals(Gt_d).real - E_N) * Ha

    # ---- det fN (truth-level base F) ----
    oa = [so_index(k, 0, nact) for k in range(nocc)]
    va = [so_index(k + nocc, 0, nact) for k in range(nvir)]
    Ms_h, Mc_h = DS.det_block_raw(data, dets, index, Hbar)

    # ---- analytic side ----
    if atom is not None:
        xyzf = os.path.join(tempfile.gettempdir(), "fullgate.xyz")
        lines = [x.strip() for x in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, basis, 0, atom=atom, active=active)
    else:
        d = CW.load(xyz, basis, ncore)
    bar = d["bar"]; t1 = d["t1"]

    # ---- base identity: raw det Hbar blocks vs bar_h base variants ----
    base_ph = np.einsum("kaic->akci", bar["Wovov"])
    base_hp = np.einsum("kcbj->bkjc", bar["Wovvo"])
    Hs = np.zeros((dim, dim)); Ht = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    een = E_N if (i == j and a == b) else 0.0
                    Hs[r, c] = Ms_h[i, a, j, b] + Mc_h[i, a, j, b] - een
                    Ht[r, c] = Ms_h[i, a, j, b] - Mc_h[i, a, j, b] - een
    for fb_tag, Fo, Fv in [("Loo/Lvv", bar["Loo"], bar["Lvv"]),
                           ("Foo_ee/Fvv_ee", bar["Foo_ee"], bar["Fvv_ee"])]:
        Bs = np.zeros((dim, dim)); Bt = np.zeros((dim, dim))
        for i in range(nocc):
            for a in range(nvir):
                r = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        c = j * nvir + b
                        f = (Fv[a, b] if i == j else 0.0) - (Fo[j, i] if a == b else 0.0)
                        Bs[r, c] = f + 2.0 * base_hp[b, j, i, a] - base_ph[a, j, b, i]
                        Bt[r, c] = f - base_ph[a, j, b, i]
        print(f"[base identity {fb_tag}] ||Bs-Hbar_s||={np.linalg.norm(Bs-Hs):.3e}  "
              f"||Bt-Hbar_t||={np.linalg.norm(Bt-Ht):.3e}")
    if probe_fbase:
        # closed-form candidates for (Foo_ee - Loo) / (Fvv_ee - Lvv) from bar + t1
        Fov = bar["Fov"]; eooov = bar["eri_ooov"]; eovvv = bar["eri_ovvv"]
        d_oo = (bar["Foo_ee"] - bar["Loo"]).ravel()
        c_oo = {"Fov.t1[me,ie]": np.einsum("me,ie->mi", Fov, t1),
                "ooov.t1 theta": 2.0 * np.einsum("kilc,lc->ki", eooov, t1)
                                 - np.einsum("likc,lc->ki", eooov, t1)}
        A = np.stack([v.ravel() for v in c_oo.values()], 1)
        coef, res, *_ = np.linalg.lstsq(A, d_oo, rcond=None)
        resid = np.linalg.norm(A @ coef - d_oo)
        print("[Foo_ee-Loo fit] " + "  ".join(f"{k}:{c:+.4f}" for k, c in
              zip(c_oo, coef)) + f"  resid={resid:.3e} (||d||={np.linalg.norm(d_oo):.3e})")
        d_vv = (bar["Fvv_ee"] - bar["Lvv"]).ravel()
        c_vv = {"Fov.t1[me,ma]": np.einsum("me,ma->ae", Fov, t1),
                "ovvv.t1 theta": 2.0 * np.einsum("kdac,kd->ac", eovvv, t1)
                                 - np.einsum("kcad,kd->ac", eovvv, t1)}
        A = np.stack([v.ravel() for v in c_vv.values()], 1)
        coef, res, *_ = np.linalg.lstsq(A, d_vv, rcond=None)
        resid = np.linalg.norm(A @ coef - d_vv)
        print("[Fvv_ee-Lvv fit] " + "  ".join(f"{k}:{c:+.4f}" for k, c in
              zip(c_vv, coef)) + f"  resid={resid:.3e} (||d||={np.linalg.norm(d_vv):.3e})")

    # ---- projection assembly (analytic bar_h + det-gauge amplitudes) ----
    Ms_terms, Mc_terms = load_terms()
    arrays = {'Fov': bar["Fov"], 'Xov': bar["Fov"], 'Wooov': bar["Wooov"],
              'Wvovv': bar["Wvovv"], 'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    gph, ghp = build_proj_g(arrays, Ms_terms, Mc_terms, bar["Wovov"], bar["Wovvo"],
                            nocc, nvir)
    for fb_tag, Foob, Fvvb in [("Loo/Lvv", bar["Loo"], bar["Lvv"]),
                               ("Foo_ee/Fvv_ee", bar["Foo_ee"], bar["Fvv_ee"])]:
        Gs, Gt = assemble(Foob, Fvvb, gph, ghp, nocc, nvir)
        es = np.sort(np.linalg.eigvals(Gs).real) * Ha
        et = np.sort(np.linalg.eigvals(Gt).real) * Ha
        print(f"[GATE F-base={fb_tag}] max|d-det| singlet={np.max(np.abs(es-es_det)):.4f} eV  "
              f"triplet={np.max(np.abs(et-et_det)):.4f} eV")
        print("   singlet:", np.round(es, 4))
        print("   det    :", np.round(es_det, 4))

    # ---- reference: shipped connected (for context) ----
    Ga, *_ = build_g_canonical_full(bar, d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], d["vir_idx"], nocc, nvir)
    es_c = np.sort(np.linalg.eigvals(Ga).real) * Ha
    print(f"[shipped connected] max|d-det| singlet={np.max(np.abs(es_c-es_det)):.4f} eV")


def main():
    run_system("H2O FC1 sto-3g", xyz="xyz/H2O.xyz", ncore=1, probe_fbase=True)
    if "h2o" not in sys.argv[1:]:
        atom = polyene(6, distort=0.0)
        active, _ = detect_pi(atom, "sto-3g", 3, 3)
        run_system("hexatriene pi-CAS(6,6) sto-3g", atom=atom, active=active)


if __name__ == "__main__":
    main()
