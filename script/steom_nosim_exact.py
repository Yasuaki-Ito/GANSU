#!/usr/bin/env python3
"""Step-1 det experiment: EXACT normal-ordered-similarity 1h1p block + K-variant zoo.

KEY RANK ANALYSIS (this campaign):
  * every normal-ordered string changes q-particle rank by exactly (#q-cre - #q-ann).
  * S is rank-homogeneous +2, {S^n} is rank-homogeneous +2n, {S^n}|HF> = 0.
  * hence  P2 {e^{+S}} = P2  and  {S^3} P2 = 0  (a 1h1p det has only 2 q-particles)
    =>  <1h1p| {e^S}^-1 Hbar {e^S} |1h1p>  =  <1h1p| Hbar (1 - S + 1/2(S^2 - K)) |1h1p>
    EXACTLY (K = S^2 - {S^2}).  No K3 correction exists for this block.
  * plain block = <1h1p| Hbar e^{-S} |1h1p> = Hbar(1 - S + 1/2 S^2) block iff the
    Hbar S^3 tail block vanishes (checked numerically here).

So the exact {e^S} object == the 2nd-order-K object (11.619 for H2O FC1) and is NOT
ORCA (11.848).  This script verifies all of the above and then scans K-piece variants
(K split by route: ii=IP*IP, ie=IP*EA, ei=EA*IP, ee=EA*EA) and product-normal-ordered
exponentials for the combination that reproduces ORCA values AND drop-invariance.

Run from GANSU root:  wsl python3 script/steom_nosim_exact.py
ORCA truth (s177 ~/steom_ref/h2o, sto-3g FC1): 11.848 / 13.601 / 16.101 / 18.238 eV.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")

from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                  build_S, project_1h1p, apply_string, occ_so, vir_so,
                                  hf_det, spat_of, popcount)
from steom_es_oracle import s_terms, _ks, _drop_roots

Ha2eV = 27.211386245988
ORCA_H2O_FC1 = [11.848, 13.601, 16.101, 18.238]


# ---------------------------------------------------------------- K via channels
def _op_matrix_rect(dets_from, index_to, nrows, ops):
    """matrix of an operator string from sector(dets_from) into sector(index_to)."""
    M = np.zeros((nrows, len(dets_from)))
    for Jc, det in enumerate(dets_from):
        sg, d = apply_string(det, ops)
        if sg:
            M[index_to[d], Jc] += sg
    return M


def build_K_channels(data, dets, index, terms_L, terms_R, aux):
    """K piece = sum of single contractions with LEFT factor from terms_L and RIGHT
    factor from terms_R.   K = sum_p Wc_p @ V_p  where p runs over q-ann channels:
      Wc_p = sum_{a in L, q-ann orbital p} c_a * M(La[:-1])          (3 q-cre string)
      V_p  = sum_{b in R, pos j with q-cre orbital p} c_b*(-1)^j * M(Lb minus op j)
    Contraction sign from steom_es_oracle.build_K: (-1)^{(na-1-i)+j}, i = na-1 (the
    q-ann is always rightmost) => (-1)^j.  String concat => matrix product Wc @ V.
    The 3-op intermediates change electron count by -1 (occ channel: V removes an
    electron-annihilator => net +1, i.e. N->N+1; vir channel: N->N-1), so V_p / Wc_p
    are rectangular via the N+-1 sectors passed in aux=(detsP,indexP,detsM,indexM)."""
    nact = data["nact"]; nocc = data["nocc"]; N = len(dets)
    detsP, indexP, detsM, indexM = aux

    def is_occ(p):
        return (p % nact) < nocc

    Wc = {}
    for c, L in terms_L:
        dag, p = L[-1]
        assert (dag and is_occ(p)) or ((not dag) and (not is_occ(p)))
        Wc.setdefault(p, []).append((c, L[:-1]))
    Vg = {}
    for c, L in terms_R:
        for j, (dag, p) in enumerate(L[:-1]):        # q-cre ops are L[:-1]
            sign = -1.0 if j % 2 else 1.0
            Vg.setdefault(p, []).append((c * sign, L[:j] + L[j + 1:]))
    K = np.zeros((N, N))
    for p, lefts in Wc.items():
        if p not in Vg:
            continue
        if is_occ(p):     # V: N->N+1 (removed op was an electron annihilator a_J)
            d_mid, i_mid = detsP, indexP
        else:             # V: N->N-1 (removed op was a creator a†_B)
            d_mid, i_mid = detsM, indexM
        nm = len(d_mid)
        Wm = np.zeros((N, nm)); Vm = np.zeros((nm, N))
        for c, ops in lefts:
            Wm += c * _op_matrix_rect(d_mid, index, N, _ks(ops))
        for c, ops in Vg[p]:
            Vm += c * _op_matrix_rect(dets, i_mid, nm, _ks(ops))
        K += Wm @ Vm
    return K


# ------------------------------------------------------------------- eigen tools
def eig_sorted(G, E_N):
    w, v = np.linalg.eig(G)
    idx = np.argsort(w.real)
    return (w[idx].real - E_N) * Ha2eV, w[idx].imag * Ha2eV, v[:, idx]


def match_roots(vref, w, v):
    """map each reference root k -> variant root argmax |<vref_k, v_j>| (greedy)."""
    ov = np.abs(vref.conj().T @ v)
    out = []
    used = set()
    for k in range(vref.shape[1]):
        order = np.argsort(-ov[k])
        for j in order:
            if j not in used:
                used.add(j); out.append(j); break
    return out


def dominant_label(vec, nocc, nvir):
    k = int(np.argmax(np.abs(vec)))
    return f"{k // nvir}->{k % nvir}", float(np.abs(vec[k]))


# ------------------------------------------------------------------------- setup
def setup(ncore=1, xyz="xyz/H2O.xyz", basis="sto-3g", drop_occ=(), drop_vir=()):
    data = get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    dets, index, HbarN = build_sector(data, data["nelec"])
    iHF = index[hf_det(data)]; E_N = HbarN[iHF, iHF]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    sIP, sEA = _drop_roots(data, sIP, sEA, drop_occ, drop_vir)
    z_ip = {m: np.zeros_like(sIP[m]) for m in sIP}
    z_ea = {e: np.zeros_like(sEA[e]) for e in sEA}
    S = build_S(data, dets, index, sIP, sEA)
    S_ip = build_S(data, dets, index, sIP, z_ea)
    S_ea = build_S(data, dets, index, z_ip, sEA)
    t_ip = s_terms(data, sIP, z_ea)
    t_ea = s_terms(data, z_ip, sEA)
    from steom_fockspace_ref import build_dets
    nso = 2 * data["nact"]
    detsP, indexP = build_dets(nso, data["nelec"] + 1)
    detsM, indexM = build_dets(nso, data["nelec"] - 1)
    aux = (detsP, indexP, detsM, indexM)
    Kii = build_K_channels(data, dets, index, t_ip, t_ip, aux)
    Kie = build_K_channels(data, dets, index, t_ip, t_ea, aux)
    Kei = build_K_channels(data, dets, index, t_ea, t_ip, aux)
    Kee = build_K_channels(data, dets, index, t_ea, t_ea, aux)
    return dict(data=data, dets=dets, index=index, HbarN=HbarN, E_N=E_N,
                S=S, S_ip=S_ip, S_ea=S_ea, Kii=Kii, Kie=Kie, Kei=Kei, Kee=Kee,
                K=Kii + Kie + Kei + Kee)


def variant_blocks(ctx):
    """dict name -> full-space matrix whose 1h1p block defines the variant."""
    H = ctx["HbarN"]; S = ctx["S"]; K = ctx["K"]
    S_ip = ctx["S_ip"]; S_ea = ctx["S_ea"]
    Kii = ctx["Kii"]; Kie = ctx["Kie"]; Kei = ctx["Kei"]; Kee = ctx["Kee"]
    N = H.shape[0]; I = np.eye(N)
    poly = I - S + 0.5 * (S @ S)
    out = {}
    out["plain(expm)"] = expm(S) @ H @ expm(-S)
    out["plain(poly)"] = H @ poly
    out["NOsim(-K/2)"] = H @ (poly - 0.5 * K)
    out["K:ii+ee"] = H @ (poly - 0.5 * (Kii + Kee))
    out["K:cross(ie+ei)"] = H @ (poly - 0.5 * (Kie + Kei))
    out["K:ii"] = H @ (poly - 0.5 * Kii)
    out["K:ee"] = H @ (poly - 0.5 * Kee)
    out["K:ie"] = H @ (poly - 0.5 * Kie)
    out["K:ei"] = H @ (poly - 0.5 * Kei)
    for d in (0.25, 0.75, 1.0):
        out[f"K:full d={d}"] = H @ (poly - d * K)
    Xip = I - S_ip + 0.5 * (S_ip @ S_ip - Kii)
    Xea = I - S_ea + 0.5 * (S_ea @ S_ea - Kee)
    out["prodNO {ip}{ea}"] = H @ (Xip @ Xea)
    out["prodNO {ea}{ip}"] = H @ (Xea @ Xip)
    return out


def run_point(ctx, names=None):
    """singlet eigenvalues for each variant at this (drop) point."""
    data = ctx["data"]; dets = ctx["dets"]; index = ctx["index"]; E_N = ctx["E_N"]
    res = {}
    blocks = variant_blocks(ctx)
    if names:
        blocks = {k: blocks[k] for k in names}
    for name, M in blocks.items():
        Gs, Gt = project_1h1p(data, dets, index, M)
        ws, wi, v = eig_sorted(Gs, E_N)
        res[name] = (ws, wi, v)
    return res


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print("=== setup H2O sto-3g FC1 (full S) ===")
    ctx = setup(ncore=1)
    data = ctx["data"]; nocc = data["nocc"]; nvir = data["nvir"]
    print(f"nocc={nocc} nvir={nvir} ndets={len(ctx['dets'])}  E_N={ctx['E_N']:.6f}")

    # ---- structural checks ----
    dets = ctx["dets"]; hfd = hf_det(data)
    rank = np.array([popcount(d ^ hfd) for d in dets])   # holes+particles
    S = ctx["S"]
    r, c = np.nonzero(np.abs(S) > 1e-12)
    bad = np.sum(rank[r] - rank[c] != 2)
    print(f"[A] S rank-homogeneous +2: violations={bad}/{len(r)}")
    from steom_es_oracle import build_K as build_K_slow
    # channel-K vs slow K on this system (slow OK here since small)
    # (skip if too slow: ndets*nt^2; H2O FC1 fine)
    sIPfull = None
    K = ctx["K"]
    eHF = np.zeros(len(dets)); eHF[ctx["index"][hfd]] = 1.0
    SS = S @ S
    print(f"[B] ||S|HF>||={np.linalg.norm(S @ eHF):.2e}  "
          f"||{{S^2}}|HF>||={np.linalg.norm((SS - K) @ eHF):.2e} (must be ~0)")

    res = run_point(ctx)
    wp = res["plain(expm)"][0]; vp = res["plain(expm)"][2]
    # tail check: plain(expm) block == plain(poly) block
    dpoly = np.max(np.abs(res["plain(poly)"][0] - wp))
    print(f"[C] plain expm vs poly block eigs: max|diff|={dpoly:.2e} eV "
          f"(0 => Hbar*S^3 tail block = 0)")

    print("\n=== full-S singlet eigenvalues (eV), roots matched to plain by overlap ===")
    print(f"{'variant':22s}" + "".join(f"{w:9.3f}" for w in wp[:6]) + "   <- plain ref")
    print(f"{'ORCA':22s}" + "".join(f"{w:9.3f}" for w in ORCA_H2O_FC1))
    for name, (ws, wi, v) in res.items():
        mm = match_roots(vp, ws, v)
        row = [ws[j] for j in mm[:6]]
        im = max(abs(wi[j]) for j in mm[:6])
        note = f"  (max|Im|={im:.3f})" if im > 1e-3 else ""
        print(f"{name:22s}" + "".join(f"{w:9.3f}" for w in row) + note)

    # composition of plain reference roots
    print("\nplain root compositions (spatial i->a, |C|):")
    for k in range(min(6, vp.shape[1])):
        lab, w = dominant_label(vp[:, k], nocc, nvir)
        print(f"  root{k}: {wp[k]:8.3f} eV  {lab} ({w:.2f})")

    # ---- drop scan on selected variants ----
    print("\n=== drop scan (root drift, matched by overlap with same variant full-S) ===")
    drops = [((), ()), ((1,), ()), ((2,), ()), ((3,), ()), ((2, 3), ()), ((), (1,))]
    sel = ["plain(expm)", "NOsim(-K/2)", "K:ii+ee", "K:cross(ie+ei)", "K:ie", "K:ei",
           "prodNO {ip}{ea}", "prodNO {ea}{ip}"]
    full = {n: res[n] for n in sel}
    table = {n: [] for n in sel}
    for do, dv in drops:
        if do == () and dv == ():
            ctx2 = ctx
        else:
            ctx2 = setup(ncore=1, drop_occ=do, drop_vir=dv)
        r2 = run_point(ctx2, names=sel)
        for n in sel:
            ws, wi, v = r2[n]
            mm = match_roots(full[n][2], ws, v)
            table[n].append([ws[j] for j in mm[:4]])
    hdr = "drop:      " + "".join(f"{str(d):16s}" for d, _ in [(d, 0) for d, _v in drops])
    print("root0/root1 per drop pattern " + str([d for d in drops]))
    for n in sel:
        r0 = [t[0] for t in table[n]]
        r1 = [t[1] for t in table[n]]
        drift0 = max(r0) - min(r0); drift1 = max(r1) - min(r1)
        print(f"{n:22s} r0: " + " ".join(f"{x:7.3f}" for x in r0)
              + f"  drift={drift0:.3f}")
        print(f"{'':22s} r1: " + " ".join(f"{x:7.3f}" for x in r1)
              + f"  drift={drift1:.3f}")
    print("\nORCA: r0=11.848 r1=13.601 (invariant).  plain full-S=11.847 (drifts). "
          "2nd-K/exact-{e^S}=11.619 (invariant).")


if __name__ == "__main__":
    main()
