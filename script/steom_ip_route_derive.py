#!/usr/bin/env python3
"""Derive the plain-projection g_phhp IP route (linear in S_ip) analytic form.

Target = the PLAIN 2nd-order BCH projection (= ORCA = det oracle), NOT GANSU's
connected Nooijen form (which deviates for config-mixed). Following the EA-route
method (steom_ea_spinadapt): extract the cross-block linear-in-S_ip contribution
from the det oracle, SO-enumerate s_ip x integral candidates, read off the formula,
then spin-integrate to spatial via the confirmed spatial2spin_ip convention.

Run: wsl python3 script/steom_ip_route_derive.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, itertools
import numpy as np
sys.path.insert(0, "script")
import steom_ea_spinadapt as EA
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, so_index, vir_so, occ_so)
from steom_so_derive import det_singles_block, build_direct, build_so_integrals


def extract_sip(sIP, data):
    """clean s_ip[m][i,j,b] = -sIP[m_alpha][I_a,J_b,B_b]  (abb block = -rx)."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    res = {}
    for mm in range(nocc):
        mA = so_index(mm, 0, nact); r = np.zeros((nocc, nocc, nvir))
        for i in range(nocc):
            for j in range(nocc):
                for b in range(nvir):
                    r[i, j, b] = -sIP[mA][so_index(i, 0, nact), so_index(j, 1, nact), so_index(b + nocc, 1, nact)]
        res[mm] = r
    return res


def build_sip_recon(sip_sp, data):
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    def O(i, s): return so_index(i, s, nact)
    def Vv(a, s): return so_index(a + nocc, s, nact)
    res = {}
    for mm in range(nocc):
        rx = sip_sp[mm]; aa = rx.transpose(1, 0, 2) - rx
        for (root, sr) in [(O(mm, 0), 0), (O(mm, 1), 1)]:
            r2 = np.zeros((nso, nso, nso))
            for i in range(nocc):
                for j in range(nocc):
                    for b in range(nvir):
                        if sr == 0:
                            r2[O(i, 0), O(j, 0), Vv(b, 0)] = aa[i, j, b]
                            r2[O(i, 0), O(j, 1), Vv(b, 1)] = -rx[i, j, b]
                            r2[O(j, 1), O(i, 0), Vv(b, 1)] = rx[i, j, b]
                        else:
                            r2[O(i, 1), O(j, 1), Vv(b, 1)] = aa[i, j, b]
                            r2[O(i, 1), O(j, 0), Vv(b, 0)] = -rx[i, j, b]
                            r2[O(j, 0), O(i, 1), Vv(b, 0)] = rx[i, j, b]
            res[root] = r2
    return res


def run(atom=None, xyz=None, active=None, ncore=0, label=""):
    print(f"\n===== {label} =====")
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
    elif active:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sip_sp = extract_sip(sIP, data)
    SIP = build_sip_recon(sip_sp, data)                 # clean-gauge SO s_ip dict
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso))
           for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    base = det_singles_block(data, dets, index, Hbar)[1]
    lin = det_singles_block(data, dets, index, Hbar + comm)[1]
    target = lin - base                                 # [i,a,j,b] cross, g_phhp IP route

    # s_ip SO tensor [M,I,J,B]
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP:
        sip[m] = SIP[m]
    g, f = build_so_integrals(data); D = build_direct(data)
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    def slc(T): return T[np.ix_(oa, va, ob, vb)]
    # enumerate: sip[e,I,J,B] contract 2 of {I,J,B} with integral X (4 legs);
    # free sip axis -> output, root axis e -> output. Output mapped to [i,a,j,b].
    cand = {}
    sax = ['p', 'q', 'r']            # sip's I,J,B axes
    # root axis e (occ SO) may map to ANY output slot -> permute all 4 output axes.
    def add_cands(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            c = slc(Tp)
            if np.linalg.norm(c) < 1e-9:
                continue
            cand[f"{tag}:perm{perm}"] = c.ravel()
    for Xname, X in [("D", D), ("g", g)]:
        for sc in itertools.combinations(range(3), 2):
            sf = [k for k in range(3) if k not in sc][0]
            for xc in itertools.permutations(range(4), 2):
                lab_s = list(sax); lab_x = list("wxyz")
                for si, xi in zip(sc, xc):
                    lab_x[xi] = lab_s[si]
                xf = [k for k in range(4) if k not in xc]
                out = [lab_x[xf[0]], lab_x[xf[1]], lab_s[sf], 'e']
                es = f"e{''.join(lab_s)},{''.join(lab_x)}->{''.join(out)}"
                try:
                    T = np.einsum(es, sip, X, optimize=True)
                except Exception:
                    continue
                add_cands(T, f"{Xname}:{es}")
    # f-based candidates (1-body): sip contract 1 axis with f, output has 2 sip-free + f-free + e
    for fc in itertools.product(range(3), range(2)):
        si, fi = fc
        lab_s = list(sax); lab_f = ['w', 'x']; lab_f[fi] = lab_s[si]
        sfree = [k for k in range(3) if k != si]; ffree = [k for k in range(2) if k != fi][0]
        out = [lab_f[ffree], lab_s[sfree[0]], lab_s[sfree[1]], 'e']
        es = f"e{''.join(lab_s)},{''.join(lab_f)}->{''.join(out)}"
        T = np.einsum(es, sip, f, optimize=True)
        add_cands(T, f"f:{es}")

    # ---- spatial form (hand spin-integration of the two SO terms) ----
    #   g_phhp[b,j,i,a] = Mc[i,a,j,b]
    #     Term1 +Sum_kl s_ip[j][l,k,a]*(ki|lb)   (chemist eri_ooov[k,i,l,b])
    #     Term2 -Sum_kc s_ip[j][k,i,c]*(kb|ca)   (chemist eri_ovvv[k,b,c,a])
    eri = data["eri"]; o = slice(0, nocc); v = slice(nocc, nocc + nvir)
    s_ip_arr = np.zeros((nocc, nocc, nocc, nvir))     # [M,I,J,B]
    for mm in range(nocc):
        s_ip_arr[mm] = sip_sp[mm]
    T1 = np.einsum("jlka,kilb->iajb", s_ip_arr, eri[o, o, o, v], optimize=True)
    T2 = -np.einsum("jkic,kbca->iajb", s_ip_arr, eri[o, v, v, v], optimize=True)
    tvv = target.ravel(); tnn = np.linalg.norm(tvv)
    for lbl, arr in [("T1", T1), ("T2", T2), ("T1+T2", T1 + T2)]:
        dd = arr.ravel()
        cc = dd @ tvv / (dd @ dd)
        print(f"    [spatial] {lbl:6s}: c={cc:+.4f}  rel-resid(c=1)={np.linalg.norm(tvv - dd) / tnn:.3e}")
    A = np.stack([T1.ravel(), T2.ravel()], 1)
    co, *_ = np.linalg.lstsq(A, tvv, rcond=None)
    print(f"    [spatial] free lstsq=({co[0]:+.4f},{co[1]:+.4f}) rel-resid={np.linalg.norm(tvv - A @ co) / tnn:.3e}")

    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 9))
        uniq.setdefault(key, n)
    names = list(uniq.values())
    tv = target.ravel(); tn = np.linalg.norm(tv)
    print(f"  SO enum: {len(cand)}->{len(names)} uniq  ||target(lin_ip)||={tn:.4f}")
    chosen = []; res = tv.copy()
    for step in range(6):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n])) * cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac @ co
        print(f"    step {step + 1}: rel-resid={np.linalg.norm(res) / tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"        {cc:+.4f}  {m}")
        if np.linalg.norm(res) / tn < 3e-3:
            break


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(6)), label="H6 rect ladder")
    import steom_cas_verify as V
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
