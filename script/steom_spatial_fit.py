#!/usr/bin/env python3
"""Read off the g_phhp EA route directly in GANSU's SPATIAL convention.

The SO derivation (steom_so_fit2) proved the route is 2 direct-Coulomb terms, but the
SO->spatial spin-adaptation is error-prone by hand (3 wrong attempts). Instead: enumerate
spatial candidates = GANSU's s_EA[b][p,q,r] contracted with GANSU's chemist eri blocks,
and lstsq-fit the oracle's physical g_phhp EA-route tensor (convention-independent as a
tensor). If GANSU's s_EA matches the oracle's physically, this yields the implementable
spatial formula. Per-root-diagonal (root b scatters to vir b in full-active).
Run: wsl python3 script/steom_spatial_fit.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, tempfile, itertools
import numpy as np
sys.path.insert(0, "script")
from scipy.linalg import expm
import steom_cas_verify as V
import steom_cfour_weff as CW
from pyscf_steom_feff_reference import build_normalized_s
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p)


def oracle_ea(atom, active):
    data = get_active_data(atom=atom, basis="sto-3g", active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sEA = solve_ea(data)
    zIP = {m: np.zeros_like(sIP[m]) for m in sIP}; zEA = {e: np.zeros_like(sEA[e]) for e in sEA}
    def gp(S, o1=False):
        G = (Hbar + (S@Hbar - Hbar@S)) if o1 else expm(S)@Hbar@expm(-S)
        Gs, Gt = project_1h1p(data, dets, index, G)
        GsD = Gs - E_N*np.eye(nocc*nvir); GtD = Gt - E_N*np.eye(nocc*nvir)
        g = np.zeros((nvir, nocc, nocc, nvir))
        for i in range(nocc):
            for a in range(nvir):
                for j in range(nocc):
                    for b in range(nvir):
                        g[b, j, i, a] = 0.5*(GsD[i*nvir+a, j*nvir+b] - GtD[i*nvir+a, j*nvir+b])
        return g
    S_ea = build_S(data, dets, index, zIP, sEA); S0 = build_S(data, dets, index, zIP, zEA)
    return gp(S_ea, o1=True) - gp(S0, o1=True), nocc, nvir   # linear-in-s_EA target


def main():
    atom = V.polyene(6); active, _ = V.detect_pi(atom, "sto-3g", 3, 3)
    ea, nocc, nvir = oracle_ea(atom, active)
    xyzf = os.path.join(tempfile.gettempdir(), "sf.xyz")
    lines = [a.strip() for a in atom.split(";")]
    open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
    d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    d["occ_idx"], d["vir_idx"], d["nocc"], d["nvir"])
    # full-active: root e -> vir active_vir_idx[e]; build target restricted to that diag
    avi = d["vir_idx"]
    bar = d["bar"]
    # chemist eri blocks (o,v labels) available in bar
    B = {k: bar[k] for k in ("eri_ovvv", "eri_ooov", "eri_ovov", "eri_oovv",
                             "eri_ovvo", "eri_vvvv", "eri_oooo")}
    # target per active root e: tgt[e][j,i,a] = ea[avi[e], j, i, a]
    n_act_vir = len(s_EA)
    tgt = np.stack([ea[avi[e]] for e in range(n_act_vir)])   # [e,j,i,a]

    # candidates: s_EA[e][p,q,r] (occ,vir,vir) contracted with one eri block -> [e,j,i,a].
    # enumerate: which eri block, and an einsum assigning (p,q,r)+block indices to (j,i,a)+dummies.
    letters = "pqr"          # s_EA free axis labels
    cand = {}
    for bname, blk in B.items():
        nd = blk.ndim
        # eri block index spaces (o/v) from name
        spaces = bname.split("_")[1]   # e.g. 'ovvv'
        # assign: 2 of s_EA's {p,q,r} contract with 2 of block's axes; rest -> outputs j,i,a
        for s_contract in itertools.combinations(range(3), 2):
            s_free = [k for k in range(3) if k not in s_contract][0]
            for b_contract in itertools.permutations(range(nd), 2):
                b_free = [k for k in range(nd) if k not in b_contract]
                if len(b_free) != 2:
                    continue
                lab_s = list("pqr"); lab_b = list("wxyz"[:nd])
                for si, bi in zip(s_contract, b_contract):
                    lab_b[bi] = lab_s[si]
                out = [lab_s[s_free], lab_b[b_free[0]], lab_b[b_free[1]]]
                # try all assignments of out -> (j,i,a)
                for perm in itertools.permutations(range(3)):
                    o = [out[perm[0]], out[perm[1]], out[perm[2]]]
                    es = f"e{''.join(lab_s)},{''.join(lab_b)}->e{''.join(o)}"
                    try:
                        T = np.einsum(es, s_EA, blk, optimize=True)
                    except Exception:
                        continue
                    if T.shape != (n_act_vir, nocc, nocc, nvir):
                        continue
                    if np.linalg.norm(T) < 1e-9:
                        continue
                    cand[f"{bname}:{es}"] = T.ravel()

    # dedup
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v/ (np.linalg.norm(v)+1e-30), 8))
        if key not in uniq:
            uniq[key] = n
    names = list(uniq.values())
    tv = tgt.ravel(); tn = np.linalg.norm(tv)
    print(f"hexatriene spatial EA fit: {len(cand)} raw -> {len(names)} unique  ||target||={tn:.4f}")
    A = np.stack([cand[n] for n in names], 1)
    coef, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"  full lstsq rel-resid = {np.linalg.norm(A@coef-tv)/tn:.3e}")
    # greedy sparse
    chosen = []; res = tv.copy()
    for step in range(4):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n]@res/(cand[n]@cand[n]))*cand[n]))
        if n in chosen: break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tv, rcond=None)
        res = tv - Ac@co
        print(f"  step{step+1} rel-resid={np.linalg.norm(res)/tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"      {cc:+.4f}  {m}")
        if np.linalg.norm(res)/tn < 5e-2:
            break


if __name__ == "__main__":
    main()
