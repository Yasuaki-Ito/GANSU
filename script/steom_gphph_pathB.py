#!/usr/bin/env python3
"""Path B probe: g_phph route needs 3-body / bilinear s.t2.eri terms (path A showed
the complete 2-body family fits only to joint-resid 0.215 over 6 C1 systems).
Add bilinear candidates = (s_ip dressed by t2) contracted with raw eri, joint-fit
over the same systems, and check whether resid drops well below 0.215.

Bilinear intermediates (spatial, closed-shell t2[i,j,a,b]):
  sd_A[m][i,l,d] = Σ_kc s[m][i,k,c] t2[k,l,c,d]
  sd_B[m][i,l,d] = Σ_kc s[m][i,k,c] t2[l,k,c,d]
  sd_C[m][k,l,a] = Σ_c   s[m][?]... (occ-pair dressings)
then contract sd with eri blocks -> [a,c,i].

Run: wsl python3 script/steom_gphph_pathB.py
"""
import os, sys, tempfile, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as CW
import steom_gphph_joint as J
import steom_gphph_spatial_dressed as SD
from steom_fockspace_ref import get_active_data, so_index, occ_so, vir_so
from steom_so_derive import build_t_arrays


def spatial_amps(data):
    """closed-shell spatial t1[i,a] and t2[i,j,a,b] from SO amplitudes."""
    from steom_fockspace_ref import build_t_so
    t1so, t2so = build_t_so(data)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]
    t1 = np.zeros((nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            t1[i, a] = t1so[so_index(i, 0, nact), so_index(a + nocc, 0, nact)]
    t2 = np.zeros((nocc, nocc, nvir, nvir))
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    t2[i, j, a, b] = t2so[so_index(i, 0, nact), so_index(j, 1, nact),
                                          so_index(a + nocc, 0, nact), so_index(b + nocc, 1, nact)]
    return t1, t2


def bilinear_cands(sip, t1, t2, bar_h, nocc, nvir):
    """FULL s.t2.eri + s.t1.eri bilinear family: enumerate all s[m][i,k,c] x t{1,2}
    contractions -> 3-index sd, then sd x raw-eri 2-index -> [m,a,c,i]."""
    o, v = nocc, nvir
    C = {}
    W = {"eri_ooov": bar_h["eri_ooov"], "eri_ovvv": bar_h["eri_ovvv"],
         "eri_ovov": bar_h["eri_ovov"], "eri_oovv": bar_h["eri_oovv"],
         "eri_oooo": bar_h["eri_oooo"], "eri_vvvv": bar_h["eri_vvvv"], "eri_ovvo": bar_h["eri_ovvo"]}
    s_axes = [(0, o), (1, o), (2, v)]              # s: i,k,c
    t_axes = [(0, o), (1, o), (2, v), (3, v)]      # t2: i,j,a,b
    t1_axes = [(0, o), (1, v)]                     # t1: i,a

    def add(es, arrs, name):
        try:
            T = np.einsum(es, *arrs, optimize=True)
        except Exception:
            return
        if T.ndim != 4:
            return
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nocc, nvir, nvir, nocc):
                continue
            vv = Tp.reshape(-1)
            if np.linalg.norm(vv) < 1e-9:
                continue
            C[f"{name}:p{perm}"] = vv

    # step 1: build all 3-index sd = s x t2 (contract 2 matching-kind axes)
    sds = {}
    for (s1, k1), (s2, k2) in itertools.combinations(s_axes, 2):
        for (t1a, tk1), (t2a, tk2) in itertools.permutations(t_axes, 2):
            if tk1 != k1 or tk2 != k2:
                continue
            sl = ['m', 'A', 'B', 'C']; tl = ['P', 'Q', 'R', 'S']
            sl[1 + s1] = 'X'; tl[t1a] = 'X'; sl[1 + s2] = 'Y'; tl[t2a] = 'Y'
            free = [c for c in sl[1:] if c not in ('X', 'Y')] + \
                   [tl[t] for t in range(4) if t not in (t1a, t2a)]
            es = f"{''.join(sl)},{''.join(tl)}->m{''.join(free)}"
            try:
                sd = np.einsum(es, sip, t2, optimize=True)
            except Exception:
                continue
            # kinds of sd's 3 free axes
            kinds = []
            for c in free:
                if c in ('A', 'B'):
                    kinds.append(o)
                elif c == 'C':
                    kinds.append(v)
                else:  # from t2
                    ti = tl.index(c); kinds.append(t_axes[ti][1])
            sds[es] = (sd, kinds)
    # t1 dressing: contract 1 s-axis with 1 t1-axis -> 3-index sd (2 s-free + 1 t1-free)
    for (s1, k1) in s_axes:
        for (t1a, tk1) in t1_axes:
            if tk1 != k1:
                continue
            sl = ['m', 'A', 'B', 'C']; tl = ['P', 'Q']
            sl[1 + s1] = 'X'; tl[t1a] = 'X'
            free = [c for c in sl[1:] if c != 'X'] + [tl[t] for t in range(2) if t != t1a]
            es = f"{''.join(sl)},{''.join(tl)}->m{''.join(free)}"
            try:
                sd = np.einsum(es, sip, t1, optimize=True)
            except Exception:
                continue
            kinds = []
            for c in free:
                if c in ('A', 'B'):
                    kinds.append(o)
                elif c == 'C':
                    kinds.append(v)
                else:
                    ti = tl.index(c); kinds.append(t1_axes[ti][1])
            sds["t1:" + es] = (sd, kinds)

    # step 2: sd (3-index) x eri (contract 2 matching-kind) -> output
    for sdn, (sd, kinds) in sds.items():
        sd_axes = [(k, kinds[k]) for k in range(3)]
        for wn, Wt in W.items():
            wsh = Wt.shape
            for (a1, k1), (a2, k2) in itertools.combinations(sd_axes, 2):
                for w1, w2 in itertools.permutations(range(len(wsh)), 2):
                    if wsh[w1] != k1 or wsh[w2] != k2:
                        continue
                    sl = ['m', 'A', 'B', 'C']; wl = [chr(ord('P') + t) for t in range(len(wsh))]
                    sl[1 + a1] = 'X'; wl[w1] = 'X'; sl[1 + a2] = 'Y'; wl[w2] = 'Y'
                    free = [c for c in sl[1:] if c not in ('X', 'Y')] + \
                           [wl[t] for t in range(len(wsh)) if t not in (w1, w2)]
                    es = f"{''.join(sl)},{''.join(wl)}->m{''.join(free)}"
                    add(es, [sd, Wt], f"BL:[{sdn}]:[{es}]:{wn}")
    return C


def load_full(**kw):
    if "xyz" in kw:
        data = get_active_data(xyz=kw["xyz"], basis="sto-3g", ncore=kw.get("ncore", 0))
        d = CW.load(kw["xyz"], "sto-3g", kw.get("ncore", 0))
    else:
        data = get_active_data(atom=kw["atom"], basis="sto-3g", ncore=kw.get("ncore", 0))
        xyzf = os.path.join(tempfile.gettempdir(), "pb.xyz")
        L = [a.strip() for a in kw["atom"].split(";")]
        open(xyzf, "w").write(f"{len(L)}\n\n" + "\n".join(L) + "\n")
        d = CW.load(xyzf, "sto-3g", kw.get("ncore", 0), atom=kw["atom"])
    nocc, nvir = data["nocc"], data["nvir"]
    base, opp_IP, sip_sp = SD.oracle_gphph_routes(data, kw.get("atom"), None)
    sip = np.stack([sip_sp[m] for m in range(nocc)], 0)
    C2, tv = J.cand_dict(sip, d["bar"], opp_IP, nocc, nvir)   # 2-body candidates
    t1, t2 = spatial_amps(data)
    Cb = bilinear_cands(sip, t1, t2, d["bar"], nocc, nvir)    # bilinear candidates
    C = {**C2, **Cb}
    return C, tv


def main():
    defs = [
        dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=1),
        dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=1),
        dict(atom="O 0 0 0; H 0.88 0.45 -0.20; H -0.20 0.96 0.30", ncore=1),
        dict(atom="O 0 0 0; H 1.12 0.20 -0.15; H -0.42 0.70 0.35", ncore=1),
        dict(atom="O 0 0 0; H 0.90 0.55 0.05; H 0.10 -0.95 0.25", ncore=1),
        dict(atom="O 0 0 0; H 0.97 0.31 0.11; H -0.33 0.89 -0.17", ncore=0),
        dict(atom="O 0 0 0; H 1.05 0.10 0.22; H -0.50 0.75 0.13", ncore=0),
        dict(atom="O 0 0 0; H 0.88 0.45 -0.20; H -0.20 0.96 0.30", ncore=0),
        dict(atom="N 0 0 0; H 0.95 0.05 0.30; H -0.45 0.83 0.28; H -0.52 -0.78 0.35", ncore=1),
        dict(atom="N 0 0 0; H 1.02 0.10 0.20; H -0.50 0.78 0.33; H -0.40 -0.85 0.28", ncore=1),
    ]
    systems = []
    for kw in defs:
        print(f"loading {kw['atom'][:20]}... ncore={kw['ncore']}")
        try:
            systems.append(load_full(**kw))
        except Exception as e:
            print(f"  SKIP {e}")
    common = set(systems[0][0])
    for C, _ in systems[1:]:
        common &= set(C)
    common = sorted(common)
    n2 = sum(1 for n in common if not n.startswith("BL:"))
    print(f"\ncommon candidates: {len(common)} (2-body={n2}, bilinear={len(common)-n2})")
    for tag, filt in [("2-body only", lambda n: not n.startswith("BL:")),
                      ("2-body + bilinear", lambda n: True)]:
        cols = [n for n in common if filt(n)]
        rows = []; tgt = []
        for C, tv in systems:
            rows.append(np.stack([C[n] for n in cols], 1)); tgt.append(tv)
        A = np.concatenate(rows, 0); t = np.concatenate(tgt, 0)
        co, *_ = np.linalg.lstsq(A, t, rcond=None)
        print(f"  [{tag}] ncol={len(cols)}  JOINT resid={np.linalg.norm(A@co-t)/np.linalg.norm(t):.3e}")


if __name__ == "__main__":
    main()
