#!/usr/bin/env python3
"""Derive the plain-projection g_phph route by fitting ONE antisymmetrized-g SO
formula to BOTH spin blocks (Ms same-spin AND Mc cross-spin) simultaneously.

Rationale (steom_so_derive.validate_singles):
  Ms_base = fpart + Wmbej[j,a,b,i]      (antisym W  = direct - exchange)
  Mc_base = fpart + Coulomb(Wmbej)      (direct only)
=> Ms and Mc are two slices of ONE spin-orbital route tensor built from the
   antisymmetrized <pq||rs>.  The cross slice (aa/bb) collapses to direct only
   (why the g_phhp fit was clean with D); the same slice (aa/aa) keeps
   direct-exchange (why blind same-spin enumeration is collinear).

Fix: enumerate candidates s_ip x {antisym g, direct D}, but SLICE EACH CANDIDATE
ON BOTH blocks and STACK (Mc | Ms) targets.  The cross block pins the direct
coefficient; requiring the same antisym formula to also reproduce Ms fixes the
exchange -- no collinearity.  Then g_phph = Mc - Ms reads off as the exchange
counterpart of the g_phhp direct route.

Run: wsl python3 script/steom_gphph_fullso.py
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


def build_targets(data, dets, index, Hbar):
    """Linear-in-S_ip route targets for BOTH spin blocks (clean gauge SO s_ip).
    Returns Mc_route, Ms_route (each [i,a,j,b]) and the SO s_ip tensor + integrals."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
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
    Mc_route = Mc1 - Mc0
    Ms_route = Ms1 - Ms0
    sip = np.zeros((nso, nso, nso, nso))
    for m in SIP:
        sip[m] = SIP[m]
    return Mc_route, Ms_route, sip, sip_sp


def enumerate_stacked(sip, data, Mc_t, Ms_t, integrals):
    """Fit candidates s_ip x integral to the STACKED (Mc | Ms) target.
    integrals: list of (name, tensor)."""
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    ob = [so_index(j, 1, nact) for j in range(nocc)]
    vb = [so_index(b + nocc, 1, nact) for b in range(nvir)]
    def slc_cross(T): return T[np.ix_(oa, va, ob, vb)]
    def slc_same(T):  return T[np.ix_(oa, va, oa, va)]
    def stacked(T):   return np.concatenate([slc_cross(T).ravel(), slc_same(T).ravel()])
    cand = {}
    sax = ['p', 'q', 'r']
    def add(T, tag):
        for perm in itertools.permutations(range(4)):
            Tp = np.transpose(T, perm)
            if Tp.shape != (nso, nso, nso, nso):
                continue
            v = stacked(Tp)
            if np.linalg.norm(v) < 1e-9:
                continue
            cand[f"{tag}:perm{perm}"] = v
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
    uniq = {}
    for n, v in cand.items():
        key = tuple(np.round(v / (np.linalg.norm(v) + 1e-30), 8))
        uniq.setdefault(key, n)
    names = list(uniq.values())
    tgt = np.concatenate([Mc_t.ravel(), Ms_t.ravel()])
    tn = np.linalg.norm(tgt)
    print(f"  STACKED SO enum: {len(cand)}->{len(names)} uniq  ||target(Mc|Ms)||={tn:.4f}"
          f"  (||Mc||={np.linalg.norm(Mc_t):.3f} ||Ms||={np.linalg.norm(Ms_t):.3f})")
    chosen = []; res = tgt.copy()
    for step in range(8):
        n = min(names, key=lambda n: np.linalg.norm(res - (cand[n] @ res / (cand[n] @ cand[n])) * cand[n]))
        if n in chosen:
            break
        chosen.append(n)
        Ac = np.stack([cand[m] for m in chosen], 1)
        co, *_ = np.linalg.lstsq(Ac, tgt, rcond=None)
        res = tgt - Ac @ co
        print(f"    step {step + 1}: rel-resid={np.linalg.norm(res) / tn:.3e}")
        for m, cc in zip(chosen, co):
            print(f"        {cc:+.4f}  {m}")
        if np.linalg.norm(res) / tn < 3e-3:
            break
    return chosen


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
    Mc_t, Ms_t, sip, sip_sp = build_targets(data, dets, index, Hbar)
    g, f = build_so_integrals(data); D = build_direct(data)
    # sanity: g_phhp cross route already known to be direct-Coulomb 2-term
    print(f"  ||Mc route||={np.linalg.norm(Mc_t):.4f}  ||Ms route||={np.linalg.norm(Ms_t):.4f}"
          f"  ||g_phph=Mc-Ms||={np.linalg.norm(Mc_t - Ms_t):.4f}")
    print("  --- fit with antisym g only (physical SO Hamiltonian) ---")
    enumerate_stacked(sip, data, Mc_t, Ms_t, [("g", g)])
    print("  --- fit with g + D (mixed basis) ---")
    enumerate_stacked(sip, data, Mc_t, Ms_t, [("g", g), ("D", D)])


def main():
    run(xyz="xyz/H2O.xyz", ncore=1, label="H2O FC1")
    run("; ".join(f"H {2.0 * (k % 2)} {1.4 * (k // 2)} 0" for k in range(6)), label="H6 rect ladder")
    import steom_cas_verify as V
    at = V.polyene(6, 0.0); ac, _ = V.detect_pi(at, "sto-3g", 3, 3)
    run(at, active=ac, label="hexatriene pi-CAS(6,6)")


if __name__ == "__main__":
    main()
