#!/usr/bin/env python3
"""Is the Ms same-spin route representable by s_ip x {g,D,f}? Full-lstsq residual
+ rank candidates. If full-resid ~0 the target is representable (hunt sparse on an
asymmetric system to break collinearity); if not, Ms needs dressed W / other struct.

Run: wsl python3 script/steom_gphph_probe.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, itertools
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
from steom_fockspace_ref import (get_active_data, build_sector, solve_ip,
                                 build_S, hf_det, so_index)
from steom_so_derive import det_singles_block, build_direct, build_so_integrals


def build_cands(sip, f, integrals, data, block):
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    if block == "cross":
        jo = [so_index(i, 1, nact) for i in range(nocc)]
        jv = [so_index(a + nocc, 1, nact) for a in range(nvir)]
    else:
        jo, jv = oa, va
    def slc(T): return T[np.ix_(oa, va, jo, jv)]
    cand = {}
    sax = ['p', 'q', 'r']
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            c = slc(Tp)
            if np.linalg.norm(c) < 1e-9:
                continue
            cand[f"{tag}:p{perm}"] = c.ravel()
    for Xname, X in integrals:
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
                add(T, f"{Xname}:{es}")
    # 1-body f candidates: contract 1 sip axis with f (2 remaining sip + f-free + e)
    for si in range(3):
        for fi in range(2):
            lab_s = list(sax); lab_f = ['w', 'x']; lab_f[fi] = lab_s[si]
            sfree = [k for k in range(3) if k != si]; ffree = 1 - fi
            out = [lab_f[ffree], lab_s[sfree[0]], lab_s[sfree[1]], 'e']
            es = f"e{''.join(lab_s)},{''.join(lab_f)}->{''.join(out)}"
            T = np.einsum(es, sip, f, optimize=True)
            add(T, f"f:{es}")
    # dedup
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 8))
        uniq.setdefault(key, n)
    return {n: cand[n] for n in uniq.values()}


def run(atom=None, xyz=None, active=None, ncore=0, label=""):
    print(f"\n===== {label} =====")
    if xyz:
        data = get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
    elif active:
        data = get_active_data(atom=atom, basis="sto-3g", active=active)
    else:
        data = get_active_data(atom=atom, basis="sto-3g", ncore=ncore)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N)
    sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data)
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso, nso, nso))
           for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP, zEA)
    comm = S_ip @ Hbar - Hbar @ S_ip
    Ms0, Mc0, _ = det_singles_block(data, dets, index, Hbar)
    Ms1, Mc1, _ = det_singles_block(data, dets, index, Hbar + comm)
    Mc_route = Mc1 - Mc0; Ms_route = Ms1 - Ms0
    gphph = Mc_route - Ms_route
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP:
        sip[m] = SIP[m]
    g, f = build_so_integrals(data); D = build_direct(data)

    for tgt_name, tgt, blk in [("Mc(cross)", Mc_route, "cross"),
                               ("Ms(same)", Ms_route, "same"),
                               ("g_phph=Mc-Ms(same)", gphph, "same")]:
        cand = build_cands(sip, f, [("g", g), ("D", D)], data, blk)
        names = list(cand); A = np.stack([cand[n] for n in names], 1)
        tv = tgt.ravel(); tn = np.linalg.norm(tv)
        co, *_ = np.linalg.lstsq(A, tv, rcond=None)
        full_resid = np.linalg.norm(A @ co - tv) / tn
        # rank by |contribution| = |coeff|*||cand||
        contrib = sorted(((abs(co[i]) * np.linalg.norm(A[:, i]), names[i], co[i])
                          for i in range(len(names))), reverse=True)
        print(f"  [{tgt_name}] ||tgt||={tn:.4f}  ncand={len(names)}  FULL-lstsq resid={full_resid:.3e}")
        for w, n, c in contrib[:6]:
            print(f"       contrib={w:.4f} coeff={c:+.4f}  {n}")


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    # asymmetric H6 to break collinearity
    run("; ".join(f"H {2.0*(k%2)+0.13*k} {1.4*(k//2)} {0.05*k}" for k in range(6)),
        label="H6 asymmetric")
    import steom_cas_verify as V
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
