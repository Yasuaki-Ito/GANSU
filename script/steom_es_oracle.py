#!/usr/bin/env python3
"""
{e^S} normal-ordered-exponential STEOM oracle.

The plain-expm determinant oracle (steom_fockspace_ref.py) contaminates the EA
route because S^IP (q-ann a†_m) and S^EA (q-ann a_e) SELF-CONTRACT in S^2, so the
true STEOM uses the NORMAL-ORDERED exponential {e^S} (Nooijen Eq.8), not e^S.

Leading 2nd-order correction (derived; cross terms S·Hbar·S cancel between G and
G_plain):
        G_{e^S} = G_plain - 1/2 (K Hbar + Hbar K),   K = S^2 - {S^2}
K = sum of all SELF-CONTRACTIONS of S^2.  KEY FACT: every S operator carries
EXACTLY ONE q-annihilation (IP: a†_m ; EA: a_e), so K has only SINGLE contractions
=> K is a pure 3-body operator, and after one contraction the lone remaining q-ann
sits rightmost => the 6 leftover ladder operators are already normal-ordered =>
plain product == normal product => apply_string builds them directly.

This makes the EA route clean and lets us fit u_akei / u_bkje / cross.

Run:  wsl python3 script/steom_es_oracle.py [ncore]   (OMP_NUM_THREADS forced to 1)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")

import steom_fockspace_ref as F
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                  build_S, project_1h1p, apply_string, occ_so, vir_so,
                                  hf_det)

Ha2eV = 27.211386245988


# --------------------------------------------------------------------- S as terms
def s_terms(data, sIP, sEA, thr=1e-13):
    """S = S^IP + S^EA written as a list of (coeff, normal_op_list).

    normal_op_list = ladder ops in NORMAL ORDER (q-creations left, q-ann right),
    op = (dag, p): dag True => a†_p, False => a_p.

    {a†_m a_I a†_B a_J} = -1 * (a_I a†_B a_J a†_m)   [move a†_m past 3 ops]
    {a†_A a_e a†_B a_J} = +1 * (a†_A a†_B a_J a_e)   [move a_e past 2 ops]
    build_S applies coeff 0.5*v to each {operator}, so coeff = 0.5*v*sign_perm.
    """
    occ = occ_so(data); vir = vir_so(data)
    terms = []
    for m in occ:
        amp = sIP[m]
        for I in occ:
            for J in occ:
                for B in vir:
                    v = amp[I, J, B]
                    if abs(v) < thr:
                        continue
                    terms.append((-0.5 * v,
                                  [(False, I), (True, B), (False, J), (True, m)]))
    for e in vir:
        amp = sEA[e]
        for J in occ:
            for A in vir:
                for B in vir:
                    v = amp[J, A, B]
                    if abs(v) < thr:
                        continue
                    terms.append((+0.5 * v,
                                  [(True, A), (True, B), (False, J), (False, e)]))
    return terms


def _ks(ops):
    """convert internal (dag, p) ladder ops to apply_string's ('c'/'a', p) form."""
    return [("c" if dag else "a", p) for dag, p in ops]


def build_S_from_terms(data, dets, index, terms):
    """Reconstruct S as a matrix from the normal-ordered term list (validation)."""
    N = len(dets)
    S = np.zeros((N, N))
    for coeff, ops in terms:
        ks = _ks(ops)
        for Jc, det in enumerate(dets):
            sg, d = apply_string(det, ks)
            if sg == 0:
                continue
            S[index[d], Jc] += coeff * sg
    return S


# ------------------------------------------------------------------ contraction K
def _is_occ(p, nact, nocc):
    return (p % nact) < nocc


def _contract_val(opL, opR, nact, nocc):
    """nonzero single contraction <opL opR> (opL left, opR right), HF reference.
    <a†_I a_J> = δ_IJ  (both occ) ; <a_A a†_B> = δ_AB (both vir).  returns True/False."""
    dagL, pL = opL
    dagR, pR = opR
    if pL != pR:
        return False
    occL = _is_occ(pL, nact, nocc)
    if dagL and (not dagR) and occL:        # a†_occ (q-ann) · a_occ (q-cre)
        return True
    if (not dagL) and dagR and (not occL):  # a_vir (q-ann) · a†_vir (q-cre)
        return True
    return False


def build_K(data, dets, index, terms):
    """K = S^2 - {S^2} = Σ single self-contractions of S^2.
    terms = list of (coeff, normal_op_list) for S (q-cre left, q-ann right)."""
    nact = data["nact"]; nocc = data["nocc"]; N = len(dets)
    K = np.zeros((N, N))
    nt = len(terms)
    for ia in range(nt):
        ca, La = terms[ia]
        na = len(La)
        for ib in range(nt):
            cb, Lb = terms[ib]
            nb = len(Lb)
            cc = ca * cb
            for i in range(na):
                opi = La[i]
                for j in range(nb):
                    if not _contract_val(opi, Lb[j], nact, nocc):
                        continue
                    sign = -1.0 if ((na - 1 - i) + j) % 2 else 1.0
                    remaining = _ks(La[:i] + La[i + 1:] + Lb[:j] + Lb[j + 1:])
                    coeff = cc * sign
                    for Jc, det in enumerate(dets):
                        sg, d2 = apply_string(det, remaining)
                        if sg == 0:
                            continue
                        K[index[d2], Jc] += coeff * sg
    return K


# ---------------------------------------------------------- {e^S} transform of an op
def transform_eS(Hmat, S, K, sign=+1):
    """G = {e^{sign·S}}^{-1?} ...  here returns {e^{-S}} H {e^{S}} to 2nd order:
       e^{-S} H e^{S} - 1/2 (K H + H K).   (sign kept for symmetry tests.)"""
    Gp = expm(-sign * S) @ Hmat @ expm(sign * S)
    return Gp - 0.5 * (K @ Hmat + Hmat @ K)


# ------------------------------------------------------------------------ main run
def _drop_roots(data, sIP, sEA, drop_occ=(), drop_vir=()):
    """zero out S^IP roots on spatial-occ in drop_occ (ORCA NActIP exclusion of a
    low-%singles deep root) and S^EA roots on spatial-vir in drop_vir."""
    from steom_fockspace_ref import spat_of
    nact = data["nact"]; nocc = data["nocc"]
    sIP = {m: (np.zeros_like(sIP[m]) if spat_of(m, nact) in drop_occ else sIP[m])
           for m in sIP}
    sEA = {e: (np.zeros_like(sEA[e]) if (spat_of(e, nact) - nocc) in drop_vir else sEA[e])
           for e in sEA}
    return sIP, sEA


def run(ncore=2, xyz="xyz/H2O.xyz", basis="sto-3g", drop_occ=(), drop_vir=()):
    np.set_printoptions(precision=4, suppress=True, linewidth=170)
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nocc = data["nocc"]; nvir = data["nvir"]
    print(f"== {xyz} {basis} ncore={ncore}  nact={data['nact']} nocc={nocc} nvir={nvir}"
          f"  drop_occ={drop_occ} drop_vir={drop_vir}")
    print(f"   HF={data['Ehf']:.6f} CCSD={data['Eccsd']:.6f}")

    dets, index, HbarN = build_sector(data, data["nelec"])
    hf = hf_det(data); iHF = index[hf]; E_N = HbarN[iHF, iHF]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sIP, sEA = _drop_roots(data, sIP, sEA, drop_occ, drop_vir)

    # ---- validation A: term-list reconstructs S ----
    S = build_S(data, dets, index, sIP, sEA)
    terms = s_terms(data, sIP, sEA)
    S2 = build_S_from_terms(data, dets, index, terms)
    print(f"\n[A] ||S_terms - build_S|| = {np.linalg.norm(S2 - S):.2e}   "
          f"(n_terms={len(terms)})")

    # ---- build K, validation B: {S^2}|HF> = 0  (strong: normal-ordered S kills
    #      the q-vacuum HF, so S|HF>=0 and every {S^n>=1}|HF>=0 => S^2|HF>=K|HF>) ----
    K = build_K(data, dets, index, terms)
    SS = S @ S
    eHF = np.zeros(len(dets)); eHF[iHF] = 1.0
    print(f"[B] ||S|HF>|| = {np.linalg.norm(S @ eHF):.2e}  (normal-ordered S kills HF)")
    NS2 = SS - K                                          # = {S^2}
    print(f"    ||S^2|HF>|| = {np.linalg.norm(SS @ eHF):.2e}   "
          f"||K|HF>|| = {np.linalg.norm(K @ eHF):.2e}   "
          f"||{{S^2}}|HF>|| = {np.linalg.norm(NS2 @ eHF):.2e}  (must be ~0)")
    print(f"    ||K||={np.linalg.norm(K):.4e}  ||S^2||={np.linalg.norm(SS):.4e}")

    # ---- G_plain and G_{e^S} ----   (correct STEOM sign: e^{+S} Hbar e^{-S})
    Gp_full = expm(S) @ HbarN @ expm(-S)
    GeS_full = Gp_full - 0.5 * (K @ HbarN + HbarN @ K)

    def eigs_of(Gfull):
        Gs, Gt = project_1h1p(data, dets, index, Gfull)
        es = np.sort(np.linalg.eigvals(Gs).real - E_N) * Ha2eV
        et = np.sort(np.linalg.eigvals(Gt).real - E_N) * Ha2eV
        return es, et, Gs - E_N * np.eye(nocc * nvir), Gt - E_N * np.eye(nocc * nvir)

    es_p, et_p, _, _ = eigs_of(Gp_full)
    es_e, et_e, GsD, GtD = eigs_of(GeS_full)
    print("\n[C] STEOM 1h1p eigenvalues (eV):")
    print("    plain  singlet:", np.round(es_p, 3))
    print("    {e^S}  singlet:", np.round(es_e, 3))
    print("    plain  triplet:", np.round(et_p, 3))
    print("    {e^S}  triplet:", np.round(et_e, 3))

    res = dict(data=data, dets=dets, index=index, HbarN=HbarN, E_N=E_N,
               sIP=sIP, sEA=sEA, S=S, K=K, terms=terms,
               Gp_full=Gp_full, GeS_full=GeS_full, GsD=GsD, GtD=GtD,
               es_e=es_e, et_e=et_e, es_p=es_p, et_p=et_p)
    return res


def analyze(ncore=2, xyz="xyz/H2O.xyz", basis="sto-3g"):
    """Extract clean g_phph_true / g_phhp_true from the {e^S} oracle and fit them
    against the analytic per-route tensors (base / u_amci / u_akei / cross)."""
    import steom_cfour_weff as C
    from steom_route_probe import route_tensors
    from pyscf_steom_feff_reference import build_g_canonical_full
    np.set_printoptions(precision=4, suppress=True, linewidth=170)

    res = run(ncore=ncore, xyz=xyz, basis=basis)
    nocc = res["data"]["nocc"]; nvir = res["data"]["nvir"]

    d = C.load(xyz, basis, ncore)
    Foo, Fvv = C.build_feff(d)
    base, t_amci, t_akei = route_tensors(d)
    Ga, g_ph_a, g_hp_a, u_amei, u_bmje, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    mycross = g_ph_a - (base + t_amci + t_akei)        # analytic cross in g_phph

    def fit(GsD, GtD, tag):
        g_phph_true = np.zeros((nvir, nocc, nvir, nocc))
        g_phhp_true = np.zeros((nvir, nocc, nocc, nvir))
        for i in range(nocc):
            for a in range(nvir):
                r = i * nvir + a
                for j in range(nocc):
                    for b in range(nvir):
                        c = j * nvir + b
                        Fp = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                        g_phph_true[a, j, b, i] = Fp - GtD[r, c]
                        g_phhp_true[b, j, i, a] = 0.5 * (GsD[r, c] - GtD[r, c])
        cand = {"base": base, "u_amci": t_amci, "u_akei": t_akei, "cross": mycross}
        names = list(cand)
        A = np.stack([cand[n].ravel() for n in names], 1)
        coef, *_ = np.linalg.lstsq(A, g_phph_true.ravel(), rcond=None)
        resid = np.linalg.norm(A @ coef - g_phph_true.ravel()) / np.linalg.norm(g_phph_true.ravel())
        print(f"\n  ===== [{tag}] g_phph_true = Σ c_k route_k  (resid={resid:.3f}) =====")
        for n, c in zip(names, coef):
            print(f"    {n:8s} c={c:+.3f}")
        R = g_phph_true - sum(coef[k] * cand[names[k]] for k in range(len(names)))
        print(f"    ||g_phph_true||={np.linalg.norm(g_phph_true):.4f}  "
              f"||resid tensor||={np.linalg.norm(R):.4f}")
        # off-diagonal route-isolation ratios (cleanest single-route magnitudes)
        return g_phph_true, g_phhp_true

    print("\n##### ROUTE FIT: plain expm vs {e^S} #####")
    GsD_p, GtD_p = _project_shift(res, res["Gp_full"])
    fit(GsD_p, GtD_p, "plain")
    gphph_e, gphhp_e = fit(res["GsD"], res["GtD"], "{e^S}")

    # eigenvalue comparison vs EOM and (ncore=1) ORCA
    print("\n  EOM-EE singlet:", np.round(np.sort(d["e_s"]) * Ha2eV, 3))
    print("  {e^S}  singlet:", np.round(res["es_e"], 3))
    if ncore == 1:
        print("  ORCA STEOM (h2o sto-3g FC1): IROOT1=11.849  IROOT2=13.60")
    return res, dict(gphph_true=gphph_e, gphhp_true=gphhp_e,
                     base=base, t_amci=t_amci, t_akei=t_akei, cross=mycross,
                     Foo=Foo, Fvv=Fvv, d=d)


def _route_S(data, dets, index, sIP, sEA, route):
    z_ip = {m: np.zeros_like(sIP[m]) for m in sIP}
    z_ea = {e: np.zeros_like(sEA[e]) for e in sEA}
    if route == "ip":
        return build_S(data, dets, index, sIP, z_ea)
    if route == "ea":
        return build_S(data, dets, index, z_ip, sEA)
    return build_S(data, dets, index, sIP, sEA)


def route_isolation(ncore=2, xyz="xyz/H2O.xyz", basis="sto-3g", use_eS=True,
                    excl_o0=True):
    """Extract per-route g_phph dressing tensors u_amci(S^IP) / u_akei(S^EA) from
    the determinant oracle the way pt19/21 validated:  g_phph[a,j,b,i] = δF_eff − Gt,
    with the ROUTE-CONSISTENT F_eff (IP route dresses Foo only; EA route dresses
    Fvv only).  This separates the F_eff dressing from g_phph on the diagonal.
    use_eS: add the {e^S} 2nd-order K correction (matters for the EA route).
    excl_o0: drop the pathological deep-occ IP root (analytic F mishandles it)."""
    import steom_cfour_weff as C
    from steom_route_probe import route_tensors
    np.set_printoptions(precision=4, suppress=True, linewidth=170)

    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nocc = data["nocc"]; nvir = data["nvir"]
    dets, index, HbarN = build_sector(data, data["nelec"])
    iHF = index[hf_det(data)]; E_N = HbarN[iHF, iHF]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)

    d = C.load(xyz, basis, ncore)
    bar = d["bar"]
    Loo = bar["Loo"]; Lvv = bar["Lvv"]
    Foo_full, Fvv_full = C.build_feff(d)     # Foo=Loo+IP-dressing, Fvv=Lvv+EA-dressing
    base, t_amci_bug, t_akei = route_tensors(d)
    t_amci_fix = _amci_wooov(d)

    # g_phph via δF_eff − Gt, with optional {e^S} 2nd-order K correction
    # correct STEOM sign: e^{+S} Hbar e^{-S}
    def gphph_K(S_route, K_route, Foo, Fvv):
        G = expm(S_route) @ HbarN @ expm(-S_route)
        if K_route is not None:
            G = G - 0.5 * (K_route @ HbarN + HbarN @ K_route)
        _, Gt = project_1h1p(data, dets, index, G)
        GtD = Gt - E_N * np.eye(nocc * nvir)
        g = np.zeros((nvir, nocc, nvir, nocc))
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        Fp = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                        g[a, j, b, i] = Fp - GtD[i * nvir + a, j * nvir + b]
        return g

    z_ip = {m: np.zeros_like(sIP[m]) for m in sIP}
    z_ea = {e: np.zeros_like(sEA[e]) for e in sEA}
    S0 = build_S(data, dets, index, z_ip, z_ea)
    S_ip = build_S(data, dets, index, sIP, z_ea)
    S_ea = build_S(data, dets, index, z_ip, sEA)
    K_ip = build_K(data, dets, index, s_terms(data, sIP, z_ea)) if use_eS else None
    K_ea = build_K(data, dets, index, s_terms(data, z_ip, sEA)) if use_eS else None

    g0 = gphph_K(S0, None, Loo, Lvv)
    g_ip = gphph_K(S_ip, K_ip, Foo_full, Lvv)      # IP route: Foo dressed, Fvv bare
    g_ea = gphph_K(S_ea, K_ea, Loo, Fvv_full)      # EA route: Fvv dressed, Foo bare
    u_amci_true = g_ip - g0
    u_akei_true = g_ea - g0

    # pure LINEAR extraction by finite difference (eps*S): {e^S}==e^S at O(eps),
    # F-dressing is linear in s -> use eps*(Foo_full-Loo) etc.
    eps = 1e-3
    g_ip_lin = gphph_K(eps * S_ip, None, Loo + eps * (Foo_full - Loo), Lvv)
    g_ea_lin = gphph_K(eps * S_ea, None, Loo, Lvv + eps * (Fvv_full - Lvv))
    u_amci_lin = (g_ip_lin - g0) / eps
    u_akei_lin = (g_ea_lin - g0) / eps

    print(f"\n== ROUTE ISOLATION  {xyz} {basis} ncore={ncore}  use_eS={use_eS} "
          f"excl_o0={excl_o0}  (nocc={nocc} nvir={nvir}) ==")
    print(f"  sanity ||g_phph(0) - base(Wovov)|| = {np.linalg.norm(g0 - base):.2e}")

    def mask(T):
        T = T.copy()
        if excl_o0:
            T[:, 0, :, :] = 0.0   # drop pathological deep-occ row (i index = 0)
            T[:, :, :, 0] = 0.0
        return T

    def cmp(name, true, ana):
        true = mask(true); ana = mask(ana)
        num = float(np.vdot(ana, true)); den = float(np.vdot(ana, ana))
        k = num / den if den else 0.0
        res = np.linalg.norm(true - ana) / max(np.linalg.norm(true), 1e-30)
        resk = np.linalg.norm(true - k * ana) / max(np.linalg.norm(true), 1e-30)
        print(f"    {name:20s}: ||true||={np.linalg.norm(true):.4f} ||ana||={np.linalg.norm(ana):.4f}"
              f"  true≈{k:+.3f}×ana  resid(1×)={res:.3f}  resid(bestk)={resk:.3f}")

    print("  IP route — FULL transform true:")
    cmp("u_amci (Wovoo bug)", u_amci_true, t_amci_bug)
    cmp("u_amci (Wooov fix)", u_amci_true, t_amci_fix)
    print("  IP route — LINEAR (finite-diff) true:")
    cmp("u_amci (Wovoo bug)", u_amci_lin, t_amci_bug)
    cmp("u_amci (Wooov fix)", u_amci_lin, t_amci_fix)
    print("  EA route:")
    cmp("u_akei FULL", u_akei_true, t_akei)
    cmp("u_akei LINEAR", u_akei_lin, t_akei)
    # HOMO->LUMO diagonal in eV (pt21: true u_amci = -1.547; analytic Wooov should match)
    H = nocc - 1; L = 0
    print(f"  HOMO(o{H})->LUMO(v{L}) g_phph diagonal [eV]:")
    print(f"    u_amci: true(full)={u_amci_true[L,H,L,H]*Ha2eV:+.3f} "
          f"true(lin)={u_amci_lin[L,H,L,H]*Ha2eV:+.3f} "
          f"Wovoo={t_amci_bug[L,H,L,H]*Ha2eV:+.3f} Wooov={t_amci_fix[L,H,L,H]*Ha2eV:+.3f}")
    print(f"    u_akei: true(full)={u_akei_true[L,H,L,H]*Ha2eV:+.3f} "
          f"true(lin)={u_akei_lin[L,H,L,H]*Ha2eV:+.3f} ana={t_akei[L,H,L,H]*Ha2eV:+.3f}")
    return dict(u_amci_true=u_amci_true, u_akei_true=u_akei_true,
                u_amci_lin=u_amci_lin, u_akei_lin=u_akei_lin, g0=g0,
                base=base, t_amci_bug=t_amci_bug, t_amci_fix=t_amci_fix,
                t_akei=t_akei, d=d, data=data, bar=bar)


def ip_sector(ncore=2, xyz="xyz/H2O.xyz", basis="sto-3g"):
    """IP (N-1) sector decoupling + F_eff_oo, plain expm vs {e^S}.
    Transform sign (pt19/22): G_IP = e^{+S^IP} Hbar e^{-S^IP} - 1/2{K,Hbar}.
    Validations: (1) active-1h <-> 2h1p coupling residual -> ~0 ;
                 (2) G_IP[active-1h] eigenvalues == E_N - omega_IP and the block
                     matches analytic F_eff_oo (verifies s amplitude SCALE)."""
    import steom_cfour_weff as C
    from steom_fockspace_ref import occ_so, vir_so, apply_string, build_sector
    np.set_printoptions(precision=4, suppress=True, linewidth=170)

    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    nocc = data["nocc"]; nvir = data["nvir"]
    dets, index, HbarN = build_sector(data, data["nelec"])
    iHF = index[hf_det(data)]; E_N = HbarN[iHF, iHF]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    z_ea = {e: np.zeros_like(sEA[e]) for e in sEA}

    # (N-1) sector
    detsM, indexM, HbarM = build_sector(data, data["nelec"] - 1)
    occ = occ_so(data); vir = vir_so(data); hf = hf_det(data)
    p_list = []; p_orb = []
    for I in occ:
        s, dd = apply_string(hf, [("a", I)])
        p_list.append(indexM[dd]); p_orb.append((I, s))
    q_list = []; seen = set(p_list)
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            I, J = occ[ii], occ[jj]
            for B in vir:
                from steom_fockspace_ref import apply_string as _as
                sg, dd = _as(hf, [("c", B), ("a", I), ("a", J)])
                if sg == 0 or indexM[dd] in seen:
                    continue
                seen.add(indexM[dd]); q_list.append(indexM[dd])
    S_ipM = build_S(data, detsM, indexM, sIP, z_ea)
    K_ipM = build_K(data, detsM, indexM, s_terms(data, sIP, z_ea))

    def transform(useK):
        G = expm(S_ipM) @ HbarM @ expm(-S_ipM)        # e^{+S} Hbar e^{-S}
        if useK:
            G = G - 0.5 * (K_ipM @ HbarM + HbarM @ K_ipM)
        return G

    sub = p_list + q_list
    npd = len(p_list)
    for tag, useK in [("plain", False), ("{e^S}", True)]:
        G = transform(useK)
        Gsub = G[np.ix_(sub, sub)]
        coupling = Gsub[:npd, npd:]
        res = np.linalg.norm(coupling) / np.linalg.norm(Gsub[:npd, :npd])
        # 1h block eigenvalues -> -omega_IP relative to E_N
        w = np.sort(np.linalg.eigvals(Gsub[:npd, :npd]).real - E_N) * Ha2eV
        print(f"  [{tag:5s}] 1h<->2h1p coupling residual = {res:.4f}   "
              f"1h eigs (E-E_N, eV)= {np.round(w, 2)}")
    # analytic reference: F_eff_oo eigenvalues == -omega_IP (build_feff validated)
    d = C.load(xyz, basis, ncore)
    Foo, Fvv = C.build_feff(d)
    wa = np.sort(np.linalg.eigvals(Foo).real - E_N) * Ha2eV
    print(f"  analytic F_eff_oo eigs (E-E_N, eV) = {np.round(wa, 2)}")
    return


def _ea_candidates(d):
    """candidate [a,k,c,i] tensors for the u_akei (Eq57, S^EA) g_phph route,
    each scattered into c=vir_idx[e]. s_EA[e]=[i,a,c]. spinad st=2s-s.swap(1,2)."""
    from pyscf_steom_feff_reference import build_normalized_s
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]; Wovoo = bar["Wovoo"]
    Wovvo = bar["Wovvo"]; Wovov = bar["Wovov"]; Wvvvo = bar["Wvvvo"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    occ_idx, vir_idx, nocc, nvir)
    nE = len(s_EA)

    def scat(fn):
        T = np.zeros((nvir, nocc, nvir, nocc))
        for e in range(nE):
            s = s_EA[e]; st = 2.0 * s - s.transpose(0, 2, 1)
            T[:, :, vir_idx[e], :] += fn(s, st)
        return T

    cand = {}
    cand["Fov_s"]    = scat(lambda s, st: -np.einsum("kc,iac->aki", Fov, s))
    cand["Fov_st"]   = scat(lambda s, st: -np.einsum("kc,iac->aki", Fov, st))
    cand["Wovoo_st"] = scat(lambda s, st: np.einsum("ldki,lad->aki", Wovoo, st))
    cand["Wovoo_s"]  = scat(lambda s, st: np.einsum("ldki,lad->aki", Wovoo, s))
    cand["Wooov_s"]  = scat(lambda s, st: -np.einsum("lkid,lad->aki", Wooov, s))
    cand["Wooov_st"] = scat(lambda s, st: -np.einsum("lkid,lad->aki", Wooov, st))
    cand["Wvovv_s"]  = scat(lambda s, st: np.einsum("akcd,icd->aki", Wvovv, s))
    cand["Wvovv_st"] = scat(lambda s, st: np.einsum("akcd,icd->aki", Wvovv, st))
    # Wvvvo = p-h mirror of Wovoo/Wooov (the u_amci fix mirror): contract 2 vir of s_EA
    cand["Wvvvo_s"]  = scat(lambda s, st: np.einsum("abck,ibc->aki", Wvvvo, s))
    cand["Wvvvo_st"] = scat(lambda s, st: np.einsum("abck,ibc->aki", Wvvvo, st))
    return cand, s_EA, vir_idx


def fit_ea(ncore=2, xyz="xyz/H2O.xyz", basis="sto-3g"):
    """fit the clean linear u_akei_true (from {e^S}-correct route isolation) against
    candidate Eq57 terms to identify the EA-route fix (pt21 method, EA side)."""
    np.set_printoptions(precision=4, suppress=True, linewidth=170)
    from steom_route_probe import route_tensors
    R = route_isolation(ncore=ncore, xyz=xyz, basis=basis, use_eS=False, excl_o0=True)
    true = R["u_akei_lin"]; d = R["d"]
    cand, s_EA, vir_idx = _ea_candidates(d)
    _, _, t_akei_cur = route_tensors(d)            # current analytic u_akei

    def msk(T):
        T = T.copy(); T[:, 0, :, :] = 0; T[:, :, :, 0] = 0; return T  # excl o0

    # ERROR that the fix must supply = true - current analytic (pt21 method)
    err = msk(true - t_akei_cur)
    print(f"\n== EA fix search  ncore={ncore} ==")
    print(f"  ||true||={np.linalg.norm(msk(true)):.4f}  ||current ana||={np.linalg.norm(msk(t_akei_cur)):.4f}"
          f"  ||error||={np.linalg.norm(err):.4f}")
    print("  per-candidate single correlation with ERROR (coeff≈±1 & resid<<1 = the term):")
    for n in cand:
        T = msk(cand[n])
        den = float(np.vdot(T, T))
        if den < 1e-14:
            continue
        k = float(np.vdot(T, err)) / den
        res = np.linalg.norm(err - k * T) / np.linalg.norm(err)
        print(f"    {n:10s}: coeff={k:+.3f}  resid={res:.3f}")

    # decisive: CFOUR ujaie EA-route (authoritative working eqs) vs true u_akei
    import steom_cfour_weff as C2
    T = C2.cfour_tensors(d)
    ujaei, ujaie = C2.build_ujaei(T)            # [E,I,B,J]
    X_EA = d["X_EA"]
    nocc = d["nocc"]; nvir = d["nvir"]
    # gmaie[A,I,B,J] += ujaie·X_EA ; g_phph[a,j,b,i] = gmaie[a,i,b,j]
    gmaie_ea = np.einsum("EIBJ,EA->AIBJ", ujaie, X_EA)
    cfour_uakei = np.einsum("AIBJ->AJBI", gmaie_ea)   # [a,j,b,i]
    H = d["nocc"] - 1; L = 0
    res_c = np.linalg.norm(msk(cfour_uakei - true)) / np.linalg.norm(msk(true))
    den = float(np.vdot(msk(cfour_uakei), msk(cfour_uakei)))
    kc = float(np.vdot(msk(cfour_uakei), msk(true))) / den if den else 0
    print(f"  CFOUR ujaie EA-route vs true: resid(1x)={res_c:.3f}  true≈{kc:+.3f}×CFOUR  "
          f"HOMO->LUMO CFOUR={cfour_uakei[L,H,L,H]*Ha2eV:+.3f} true={true[L,H,L,H]*Ha2eV:+.3f}")
    # decompose CFOUR ujaie into its 6 components, fit true = Σ c_k comp_k
    fem = T["fem"]; wamef = T["wamef"]; wmnie = T["wmnie"]; sabej = T["sabej"]
    ss = C2.spinad(sabej); wn = C2.spinad(wmnie)
    comps = {
        "A1 sab·fem":  np.einsum("FBEJ,FI->EIBJ", sabej, fem),
        "A2 sab·wam":  np.einsum("GFEJ,FGBI->EIBJ", sabej, wamef),
        "A3 sab·wmn":  -np.einsum("FBEN,NIJF->EIBJ", sabej, wmnie),
        "B1 ss·fem":   np.einsum("BFEJ,FI->EIBJ", ss, fem),
        "B2 ss·wam":   np.einsum("GFEJ,GFBI->EIBJ", ss, wamef),
        "B3 ss·wn":    np.einsum("BFEN,INJF->EIBJ", ss, wn),
    }
    def fold(g):  # [E,I,B,J] -> g_phph [a,j,b,i]
        return np.einsum("AIBJ->AJBI", np.einsum("EIBJ,EA->AIBJ", g, X_EA))
    # grouped: A_W = A2+A3, B_W = B2+B3, FEM = A1+B1 (collinear block)
    A_W = fold(comps["A2 sab·wam"] + comps["A3 sab·wmn"])
    B_W = fold(comps["B2 ss·wam"] + comps["B3 ss·wn"])
    FEM = fold(comps["A1 sab·fem"] + comps["B1 ss·fem"])
    Amat = np.stack([msk(A_W).ravel(), msk(B_W).ravel(), msk(FEM).ravel()], 1)
    coef, *_ = np.linalg.lstsq(Amat, msk(true).ravel(), rcond=None)
    rfit = np.linalg.norm(Amat @ coef - msk(true).ravel()) / np.linalg.norm(msk(true).ravel())
    print(f"  GROUPED fit true = wA·(A2+A3) + wB·(B2+B3) + wF·(fem)  (resid={rfit:.3f}):")
    print(f"    wA(W-raw)={coef[0]:+.3f}  wB(W-spinad)={coef[1]:+.3f}  wF(fem)={coef[2]:+.3f}")
    print("    (CFOUR ujaie = 0.5A+0.5B => wA=wB=wF=0.5)")
    return R, cand, err, cfour_uakei


def _amci_wooov(d):
    """analytic u_amci with the pt21 fix: Wovoo term -> Wooov term."""
    from pyscf_steom_feff_reference import build_normalized_s
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    occ_idx, vir_idx, nocc, nvir)
    t = np.zeros((nvir, nocc, nvir, nocc))
    for m in range(len(s_IP)):
        s = s_IP[m]
        st = 2.0 * s - s.transpose(1, 0, 2)
        blk = (-np.einsum("kc,ika->aci", Fov, s)
               + np.einsum("alcd,ild->aci", Wvovv, st)
               - np.einsum("aldc,ild->aci", Wvovv, s)
               + np.einsum("klic,kla->aci", Wooov, s))      # <-- Wooov fix
        t[:, occ_idx[m], :, :] += blk
    return t


def _project_shift(res, Gfull):
    data = res["data"]; dets = res["dets"]; index = res["index"]; E_N = res["E_N"]
    nocc = data["nocc"]; nvir = data["nvir"]
    Gs, Gt = project_1h1p(data, dets, index, Gfull)
    return Gs - E_N * np.eye(nocc * nvir), Gt - E_N * np.eye(nocc * nvir)


if __name__ == "__main__":
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    analyze(ncore=nc)
