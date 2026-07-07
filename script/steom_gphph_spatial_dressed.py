#!/usr/bin/env python3
"""g_phph IP route (linear S_ip), projection target, DRESSED spatial intermediates.

Ms route is NOT representable by bare 2-body integrals (steom_gphph_probe: full
lstsq resid 0.68) -- it needs the t-dressed Hbar intermediates.  Work directly in
the spatial closed-shell rep (= what the C++ build_g_canonical_full uses): oracle
clean-gauge g_phph route (blocks()) as target, s_ip[m] x {dressed bar_h W} as
candidates.  If it closes we read off the C++-ready formula in dressed W.

Target: opp_IP = g_phph(=-Gt, F-free) linear-in-s_ip route from the det oracle,
clean gauge (GANSU s convention).  Active space = full occ in the test systems,
so root m == occ index (no scatter).

Run: wsl python3 script/steom_gphph_spatial_dressed.py
"""
import os, sys, tempfile, itertools
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_cas_verify as V
import steom_cfour_weff as CW
import steom_ea_spinadapt as EA
import steom_ip_route_derive as IPD
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip, solve_ea,
                                 build_S, hf_det, project_1h1p, so_index, vir_so, occ_so)
Ha = 27.211386245988


def oracle_gphph_routes(data, atom, active):
    """g_phph route via the VALIDATED det_singles_block target (same as the
    g_phhp IP route derivation), NOT project_1h1p (whose Gt is a subtly different
    normalization that breaks the fit).  g_phph[a,k,c,i] = (Mc - Ms)[i,a,k,c]
    with Mc,Ms the det cross/same singles blocks.  Returns base, IP route [a,m,c,i]
    (linear in s_ip), and s_ip[m][i,k,c]."""
    from steom_so_derive import det_singles_block
    nocc, nvir = data["nocc"], data["nvir"]; nact = data["nact"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {e: np.zeros((nso, nso, nso)) for e in vir_so(data)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, Mc0, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, Mc1, _ = det_singles_block(data, dets, index, Hbar + comm)
    gphph_base = (Mc0 - Ms0)              # [i,a,j,b]
    gphph_ip   = (Mc1 - Ms1) - gphph_base
    # remap [i,a,j,b] -> [a,k=j,c=b,i]
    base = np.einsum("iajb->ajbi", gphph_base)
    opp_IP = np.einsum("iajb->ajbi", gphph_ip)
    return base, opp_IP, sip_sp


def enumerate_spatial(sip_sp, bar_h, target, nocc, nvir, label):
    """target[a,m,c,i] (per-root m == occ). candidates: s_ip[m][i,k,c] x dressed W."""
    # dressed intermediates (spatial, chemist-dressed)
    W = {
        "Wooov": bar_h["Wooov"],   # [k,l,i,d] oo ov
        "Wvovv": bar_h["Wvovv"],   # [a,l,c,d] v o vv
        "Wovvo": bar_h["Wovvo"],   # [k,b,c,j] o v v o
        "Wovov": bar_h["Wovov"],   # [k,b,i,d] o v o v
        "Wovoo": bar_h["Wovoo"],   # [k,c,l,i] o v o o
        "Woooo": bar_h["Woooo"],   # [k,l,i,j] oooo
        "Fov":   bar_h["Fov"],     # [k,c]
    }
    # build target vector stacked over roots m
    tv = target.transpose(1, 0, 2, 3).reshape(nocc, -1)  # [m, (a,c,i)]
    tvec = tv.ravel(); tn = np.linalg.norm(tvec)
    # letters for occ/vir axis kinds so we only contract compatible spaces
    #   s_ip[m] axes: 0=i(occ) 1=k(occ) 2=c(vir)
    cand = {}
    sip = np.stack([sip_sp[m] for m in range(nocc)], 0)   # [m,i,k,c]

    def try_es(es, arrs, name):
        try:
            T = np.einsum(es, *arrs, optimize=True)
        except Exception:
            return
        # T must have axes we can map to (m,a,c,i). Expect shape (nocc, nvir, nvir, nocc) after perm
        if T.ndim != 4:
            return
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nocc, nvir, nvir, nocc):
                continue
            v = Tp.reshape(nocc, -1).ravel()
            if np.linalg.norm(v) < 1e-9:
                continue
            cand[f"{name}:{es}:p{perm}"] = v

    o, v = nocc, nvir
    kinds = {'i': o, 'k': o, 'c': v}     # sip free-axis kinds
    # 2-body W: contract 2 of sip's (i,k,c) with 2 of W's axes (matching occ/vir)
    for wn, Wt in W.items():
        if wn == "Fov":
            continue
        wshape = Wt.shape
        wkind = ['o' if s == o else 'v' for s in wshape]  # crude: assume o=nocc,v=nvir distinct
        # only valid if nocc != nvir (else ambiguous); guard
        if o == v:
            pass
        # enumerate: pick 2 sip axes (0,1,2) and 2 W axes to contract, matching kind
        sip_axes = [('i', 0, o), ('k', 1, o), ('c', 2, v)]
        for (n1, s1, k1), (n2, s2, k2) in itertools.combinations(sip_axes, 2):
            for w1, w2 in itertools.permutations(range(len(wshape)), 2):
                if wshape[w1] != k1 or wshape[w2] != k2:
                    continue
                # build einsum: sip 'm???' with contracted axes labelled, W with its axes
                slab = ['m', 'a', 'b', 'c']  # dummy sip labels m + 3
                # assign sip axis labels: contracted get shared letters, free gets unique
                sl = ['m', 'A', 'B', 'C']    # m + i,k,c
                wl = [chr(ord('P') + t) for t in range(len(wshape))]
                sl[1 + s1] = 'X'; wl[w1] = 'X'
                sl[1 + s2] = 'Y'; wl[w2] = 'Y'
                es = f"{''.join(sl)},{''.join(wl)}->"
                # output = m + all free letters
                free = [c for c in sl[1:] if c not in ('X', 'Y')] + [wl[t] for t in range(len(wshape)) if t not in (w1, w2)]
                es += 'm' + ''.join(free)
                try_es(es, [sip, Wt], wn)
    # 1-body Fov: contract 1 sip axis with Fov (o,v)
    Fov = W["Fov"]
    sip_axes = [('i', 0, o), ('k', 1, o), ('c', 2, v)]
    for (n1, s1, k1) in sip_axes:
        for w1 in range(2):
            if Fov.shape[w1] != k1:
                continue
            sl = ['m', 'A', 'B', 'C']; wl = ['P', 'Q']
            sl[1 + s1] = 'X'; wl[w1] = 'X'
            es = f"{''.join(sl)},{''.join(wl)}->"
            free = [c for c in sl[1:] if c != 'X'] + [wl[t] for t in range(2) if t != w1]
            es += 'm' + ''.join(free)
            try_es(es, [sip, Fov], "Fov")

    uniq = {}
    for n, vv in cand.items():
        key = tuple(np.round(vv / (np.linalg.norm(vv) + 1e-30), 8))
        uniq.setdefault(key, n)
    names = list(uniq.values())
    if not names:
        print(f"  [{label}] NO candidates"); return
    A = np.stack([cand[n] for n in names], 1)
    co, *_ = np.linalg.lstsq(A, tvec, rcond=None)
    full = np.linalg.norm(A @ co - tvec) / tn
    print(f"  [{label}] ||tgt||={tn:.4f} ncand={len(names)} FULL-lstsq resid={full:.3e}")
    # greedy sparse
    chosen = []; res = tvec.copy()
    for step in range(6):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n] + 1e-30)) * cand[n]))
        if n in chosen: break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        cc, *_ = np.linalg.lstsq(Ac, tvec, rcond=None)
        res = tvec - Ac @ cc
        print(f"    step{step+1}: resid={np.linalg.norm(res)/tn:.3e}  " +
              "  ".join(f"{c:+.3f}[{m}]" for c, m in zip(cc, chosen)))
        if np.linalg.norm(res) / tn < 3e-3: break


def run(atom=None, xyz=None, active=None, ncore=0, label=""):
    print(f"\n===== {label} =====")
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
        d = CW.load(xyz, "sto-3g", ncore)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
        import tempfile
        xyzf = os.path.join(tempfile.gettempdir(), "g.xyz")
        lines = [a.strip() for a in atom.split(";")]
        open(xyzf, "w").write(f"{len(lines)}\n\n" + "\n".join(lines) + "\n")
        d = CW.load(xyzf, "sto-3g", 0, atom=atom, active=active)
    nocc, nvir = data["nocc"], data["nvir"]
    base, opp_IP, sip_sp = oracle_gphph_routes(data, atom, active)
    print(f"  ||g_phph base||={np.linalg.norm(base):.4f}  ||g_phph IP route||={np.linalg.norm(opp_IP):.4f}")
    enumerate_spatial(sip_sp, d["bar"], opp_IP, nocc, nvir, "g_phph IP route (dressed)")


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(atom=at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
