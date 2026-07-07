#!/usr/bin/env python3
"""Isolate Hbar's many-body parts as det-basis operators, so the route's PURE
3-body contribution = [S_ip, Hbar - Hbar_{0,1,2}].Ms can be computed by difference
and fit to the general SO bilinear s.t2.g (not the (T)-driver, which was orthogonal).

Hbar_1 (1-body) and Hbar_2 (2-body) are rebuilt from their normal-ordered
coefficients f_pq, v_pqrs, extracted from Hbar via connected matrix elements:
  E0     = <HF|Hbar|HF>
  f_pq   : {p†q} coefficient   (rebuild operator, subtract, iterate)
  v_pqrs : {p†q†sr} coefficient
Then Hbar3 = Hbar - E0 - Op(f) - Op(v);  route_k = [S_ip, Hbar_k].Ms.

Run: wsl python3 script/steom_gphph_hbar_split.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import itertools
import numpy as np
sys.path.insert(0, "script")
from steom_fockspace_ref import (get_active_data, build_sector, hf_det, apply_string,
                                 so_index, occ_so, vir_so, build_dets)


def op1(data, dets, index, f):
    """det matrix of Σ_pq f[p,q] {p†q}  (normal-ordered a†_p a_q, ref |HF>)."""
    nso = 2 * data["nact"]; N = len(dets); M = np.zeros((N, N))
    for Jc, det in enumerate(dets):
        for p in range(nso):
            for q in range(nso):
                if f[p, q] == 0.0:
                    continue
                sg, d = apply_string(det, [("c", p), ("a", q)])
                if sg == 0:
                    continue
                M[index[d], Jc] += sg * f[p, q]
    # normal order: subtract the c-number <HF|Op|HF> already handled separately;
    # {p†q}=a†_p a_q - δ_pq (q occ). For our use we keep bare a†a and remove the
    # HF-diagonal expectation via E0 bookkeeping in the caller.
    return M


def op2(data, dets, index, v):
    """det matrix of ¼ Σ v[p,q,r,s] a†_p a†_q a_s a_r."""
    nso = 2 * data["nact"]; N = len(dets); M = np.zeros((N, N))
    for Jc, det in enumerate(dets):
        for p in range(nso):
            for q in range(nso):
                for r in range(nso):
                    for s in range(nso):
                        val = v[p, q, r, s]
                        if val == 0.0:
                            continue
                        sg, d = apply_string(det, [("c", p), ("c", q), ("a", s), ("a", r)])
                        if sg == 0:
                            continue
                        M[index[d], Jc] += 0.25 * sg * val
    return M


def extract_f_v(data, Hbar):
    """extract bare-a†a 1-body f and antisym 2-body v so that
    Op0 + op1(f) + op2(v) reproduces all <D'|Hbar|D> with |D'-D| <= 2 excitations.
    Uses HF-referenced connected matrix elements (single sweep, exact for <=2-body)."""
    nso = 2 * data["nact"]
    dets, index, Hbar_full = None, None, None
    return None  # placeholder; see main (uses direct 1h1p/2h2p projection instead)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nocc, nvir, nact = data["nocc"], data["nvir"], data["nact"]; nso = 2 * nact
    print(f"H2O FC1: nso={nso}")
    dets, index, Hbar = build_sector(data, data["nelec"])
    hf = hf_det(data); iHF = index[hf]; E0 = Hbar[iHF, iHF]
    occ = occ_so(data); vir = vir_so(data)

    # ---- extract 1-body f[p,q] (bare a†_p a_q coefficient) ----
    # f_pq is read from single-excitation matrix elements relative to reference in a
    # way that is exact for the 1-body part; we get it from <HF|a†_q a_p Hbar|HF> style.
    # Practical route: build all 1- and 2-body coefficients by projecting Hbar onto
    # the operator basis {a†a},{a†a†aa} restricted to <=2-excitation dets (linear solve).
    # For nso=12 this basis is small.
    # Build design: rows = (D,D') pairs with excitation rank <=2; cols = operator params.
    # -- 1-body params: f[p,q] for all p,q  (nso^2)
    # -- 2-body params: v[p,q,r,s] antisym p<q, r<s
    # Rather than a huge solve, extract analytically block by block below.

    # f from <phi_p^q | Hbar | HF>-type is messy with occ/vir; instead use the fact
    # that for a normal-ordered operator, <HF| a_a† a_i ... > gives 1-body pieces.
    # We extract the FULL 1+2-body by matching the reference block exactly:
    #   E0 = <HF|Hbar|HF>
    #   1h1p block M1[ia,jb] = <phi_i^a|Hbar|phi_j^b> = E0 d + (f_ab d_ij - f_ij d_ab) + v_iajb-ish
    # This still entangles f and v. Given complexity, we take the DIRECT approach:
    # build Hbar_012 by keeping only <=2-excitation-connected structure via a clean
    # projector is nontrivial; STOP here and report what this needs.
    print("NOTE: clean f/v extraction needs the full <=2-body operator-basis linear")
    print("solve (rows=det pairs rank<=2, cols=f,v params). nso=12 => feasible but")
    print("substantial. Building that solve is the concrete next step for Hbar3 isolation.")
    print(f"E0=<HF|Hbar|HF>={E0:.6f}")


if __name__ == "__main__":
    main()
