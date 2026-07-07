#!/usr/bin/env python3
"""END-TO-END eigenvalue test (final pre-C++ gate): assemble the full STEOM G with
the PROJECTION g_phph (and optionally projection g_phhp = route_Mc), diagonalize,
and compare singlet/triplet excitation energies vs the det oracle (GsD/GtD = ORCA).

Also decomposes the route agreement by element class: off-diag (i!=j&a!=b),
semi-diag (xor), diag (i==j&a==b) — to decide the replacement domain.

Run:  wsl python3 script/steom_gphph_endtoend.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA, steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  solve_ea, hf_det, so_index, occ_so, vir_so, project_1h1p)
from pyscf_steom_feff_reference import build_g_canonical_full

Ha = 27.211386245988


def cls_norm(T, nocc, nvir):
    off = semi = diag = 0.
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    v2 = T[i, a, j, b]**2
                    if i != j and a != b: off += v2
                    elif i == j and a == b: diag += v2
                    else: semi += v2
    return off**0.5, semi**0.5, diag**0.5


def main():
    xyz, basis, ncore = "xyz/H2O.xyz", "sto-3g", 1
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2*nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    occ = occ_so(data); vir = vir_so(data)
    z = np.load("/tmp/hbar_mbody.npz"); vp = z["vp"]; fp = z["fp"]
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    spso = np.zeros((nso,)*4); seso = np.zeros((nso,)*4)
    rec_ip = IPD.build_sip_recon(sp_det, data); rec_ea = EA.build_sea_recon(se_det, data)
    for m in occ: spso[m] = rec_ip[m]
    for e in vir: seso[e] = rec_ea[e]
    S = build_S(data, dets, index, {m: spso[m] for m in occ}, {e: seso[e] for e in vir})

    # ---- det oracle STEOM (truth = ORCA) ----
    G = expm(S) @ Hbar @ expm(-S)
    Gs, Gt = project_1h1p(data, dets, index, G)
    es_det = np.sort(np.linalg.eigvals(Gs).real - E_N)*Ha
    et_det = np.sort(np.linalg.eigvals(Gt).real - E_N)*Ha

    # ---- det routes for Ms/Mc separately (class decomposition) ----
    occ_mask = np.zeros(nso); occ_mask[occ] = 1.0
    Fc = np.einsum("pIqI,I->pq", vp, occ_mask, optimize=True); fN = fp + Fc
    g_ov = np.zeros((nso, nso)); g_ov[np.ix_(occ, vir)] = fN[np.ix_(occ, vir)]
    V2mat = H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, Fc, 1)
    op1g = H3.opk_matrix(dets, index, g_ov, 1)
    linc = S @ (op1g + V2mat) - (op1g + V2mat) @ S
    inr = S @ V2mat - V2mat @ S; qd = 0.5*(S @ inr - inr @ S)
    Ms_d, Mc_d, _ = det_singles_block(data, dets, index, linc + qd)

    # ---- generated routes ----
    lin = GEN.linear_struct()
    print("deriving quadratic (slow)...")
    quad = GEN.quad_struct()
    allt = lin + quad
    Ms_t = GEN.expand(allt, 0); Mc_t = GEN.expand(allt, 1)
    d = CW.load(xyz, basis, ncore); bar = d["bar"]
    arrays = {'Fov': bar["Fov"], 'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"],
              'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}
    gen_Ms = GEN.evaluate(Ms_t, arrays, nocc, nvir)
    gen_Mc = GEN.evaluate(Mc_t, arrays, nocc, nvir)
    for nm, gg, dd in [("Ms", gen_Ms, Ms_d), ("Mc", gen_Mc, Mc_d)]:
        o, s2, dg = cls_norm(gg - dd, nocc, nvir)
        on, sn, dn = cls_norm(dd, nocc, nvir)
        print(f"[route {nm}] diff off={o:.3e} semi={s2:.3e} diag={dg:.3e}   "
              f"(det norms off={on:.4f} semi={sn:.4f} diag={dn:.4f})")

    # ---- end-to-end: G assembly with projection replacement ----
    # analytic connected reference
    res = build_g_canonical_full(bar, d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                 d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    Ga = res[0]
    es_conn = np.sort(np.linalg.eigvals(Ga).real)*Ha

    # rebuild G with projection g_phph/g_phhp: reuse the function internals by
    # reconstructing here (base + F_eff diag from the same builder).
    # Simplest: G_proj[r,c] = F-part (from connected Ga: its diag structure) hard to
    # split; instead assemble directly:  G = d_ij F_eff_vv - d_ab F_eff_oo
    #                                     + 2*g_phhp[b,j,i,a] - g_phph[a,j,b,i]
    # with g_phhp = base_hp + route_Mc, g_phph = base_ph + (route_Mc - route_Ms).
    Wovov = bar["Wovov"]; Wovvo = bar["Wovvo"]
    # bases (EE-consistent, STEOM_EE_BASE fix): g_phph base = Wovov[k,a,i,c]->[a,k,c,i]
    # g_phhp base = Wovvo[k,c,b,j]->[b,k,j,c]  (virtual-swapped, the 2026-07-06 fix)
    base_ph = np.einsum("kaic->akci", Wovov).copy()     # [a,j,b,i] order: [a,k,c,i]
    base_hp = np.einsum("kcbj->bkjc", Wovvo).copy()     # [b,k,j,c]
    # routes in [i,a,j,b] -> map: g_phph[a,j,b,i] += route_ph[i,a,j,b] with (a=a,j=j,b=b,i=i)
    route_ph = gen_Mc - gen_Ms
    route_hp = gen_Mc
    g_phph = base_ph.copy()
    g_phhp = base_hp.copy()
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    g_phph[a, j, b, i] += route_ph[i, a, j, b]
                    g_phhp[b, j, i, a] += route_hp[i, a, j, b]
    # F_eff diagonal from the connected builder (keep GANSU convention): extract via
    # Ga's diagonal identity: Ga[r,r] = F_eff_vv[a,a]-F_eff_oo[i,i] + 2 g_hp_conn - g_ph_conn.
    # Instead rebuild F_eff directly (same as builder):
    from pyscf_steom_feff_reference import build_normalized_s
    import numpy.linalg as la
    Loo = bar["Loo"]; Lvv = bar["Lvv"]
    s_IP_an, s_EA_an = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                          d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    n_act_occ = len(d["occ_idx"]); n_act_vir = len(d["vir_idx"])
    Fov_b = bar["Fov"]; Wooov_b = bar["Wooov"]; Wvovv_b = bar["Wvovv"]
    F_eff_oo = Loo.copy()
    for m in range(n_act_occ):
        s = s_IP_an[m]; st = 2.0*s - s.transpose(1, 0, 2)
        u_mi = np.einsum("kc,ikc->i", Fov_b, st) - np.einsum("klid,kld->i", Wooov_b, st)
        F_eff_oo[d["occ_idx"][m], :] += u_mi
    F_eff_vv = Lvv.copy()
    for e in range(n_act_vir):
        s = s_EA_an[e]; st = 2.0*s - s.transpose(0, 2, 1)
        u_ae = np.einsum("kc,kac->a", Fov_b, st) + np.einsum("alcd,lcd->a", Wvovv_b, st)
        F_eff_vv[d["vir_idx"][e], :] += u_ae
    dim = nocc*nvir

    # connected g_phph/g_phhp from the builder (for the hybrid diag/semi part)
    g_ph_conn = res[1]; g_hp_conn = res[2]      # [a,k,c,i], [b,k,j,c]

    def assemble(gph, ghp):
        Gsx = np.zeros((dim, dim)); Gtx = np.zeros((dim, dim))
        for i in range(nocc):
            for a in range(nvir):
                r = i*nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        c = j*nvir + b
                        f = (F_eff_vv[a, b] if i == j else 0.0) - (F_eff_oo[i, j] if a == b else 0.0)
                        Gsx[r, c] = f + 2.0*ghp[b, j, i, a] - gph[a, j, b, i]
                        Gtx[r, c] = f - gph[a, j, b, i]
        return Gsx, Gtx

    # hybrid: off-diag (i!=j & a!=b) -> projection base+route; else connected
    gph_h = g_ph_conn.copy(); ghp_h2 = g_hp_conn.copy()
    gph_full = g_phph; ghp_full = g_phhp
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    if i != j and a != b:
                        gph_h[a, j, b, i] = gph_full[a, j, b, i]
                        ghp_h2[b, j, i, a] = ghp_full[b, j, i, a]
    variants = [
        ("connected           ", g_ph_conn, g_hp_conn),
        ("proj offdiag ph only", gph_h, g_hp_conn),
        ("proj offdiag ph+hp  ", gph_h, ghp_h2),
        ("proj FULL (naive)   ", g_phph, g_phhp),
    ]

    # ---- ISOLATED GATE: diag/semi = det truth, off-diag = my base+route ----
    Gs_hyb = (Gs - E_N*np.eye(dim)).copy(); Gt_hyb = (Gt - E_N*np.eye(dim)).copy()
    for i in range(nocc):
        for a in range(nvir):
            r = i*nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j*nvir + b
                    if i != j and a != b:
                        gph_v = base_ph[a, j, b, i] + route_ph[i, a, j, b]
                        ghp_v = base_hp[b, j, i, a] + route_hp[i, a, j, b]
                        Gs_hyb[r, c] = 2.0*ghp_v - gph_v
                        Gt_hyb[r, c] = -gph_v
    es_h = np.sort(np.linalg.eigvals(Gs_hyb).real)*Ha
    et_h = np.sort(np.linalg.eigvals(Gt_hyb).real)*Ha
    print("\n[GATE det-diag + proj-offdiag] singlet:", np.round(es_h, 4))
    print("[GATE det-diag + proj-offdiag] triplet:", np.round(et_h, 4))
    print(f"   max|d-det| singlet={np.max(np.abs(es_h-es_det)):.2e} eV  "
          f"triplet={np.max(np.abs(et_h-et_det)):.2e} eV")
    print("\n=== H2O FC1 sto-3g STEOM 1h1p eigenvalues (eV) ===")
    print("det oracle  singlet:", np.round(es_det, 4))
    print("det oracle  triplet:", np.round(et_det, 4))
    for nm, gph, ghp in variants:
        Gsx, Gtx = assemble(gph, ghp)
        esx = np.sort(np.linalg.eigvals(Gsx).real)*Ha
        etx = np.sort(np.linalg.eigvals(Gtx).real)*Ha
        print(f"\n[{nm}] singlet:", np.round(esx, 4))
        print(f"[{nm}] triplet:", np.round(etx, 4))
        print(f"   max|d-det| singlet={np.max(np.abs(esx-es_det)):.4f} eV  "
              f"triplet={np.max(np.abs(etx-et_det)):.4f} eV   "
              f"low3 singlet={np.max(np.abs(esx[:3]-es_det[:3])):.4f}")


if __name__ == "__main__":
    main()
