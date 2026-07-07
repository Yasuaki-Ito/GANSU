#!/usr/bin/env python3
"""Numerically verify the sympy-derived SO formulas against the EXACT route
contributions (Ms_1 from Fov, Ms_2a from Wooov, Ms_2b from Wvovv).
sympy raw output (root m=j, dummies I,J occ, B vir; delta = KroneckerDelta):
  Fov:   -dab/2(f[I,B]s[j,I,i,B]-f[J,B]s[j,i,J,B]) + 1/2(f[I,b]s[j,I,i,a]-f[J,b]s[j,i,J,a])
  Wooov: -dab/2(s[j,I,J,B]w[I,J,i,B]-s[j,I,J,B]w[J,I,i,B]) + 1/2(s[j,I,J,a]w[I,J,i,b]-s[j,I,J,a]w[J,I,i,b])
  Wvovv:  1/2(s[j,I,i,B](w[a,I,B,b]-w[a,I,b,B]) - s[j,i,J,B](w[a,J,B,b]-w[a,J,b,B]))
w = Wooov=v'[o,o,o,v] / Wvovv=v'[v,o,v,v] block ; f = f'[o,v].

Run: wsl python3 script/steom_gphph_sympy_verify.py
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_ip_route_derive as IPD
import steom_gphph_hbar3 as H3
from steom_so_derive import det_singles_block
from steom_fockspace_ref import (get_active_data, build_sector, build_S, solve_ip,
                                  hf_det, so_index, occ_so, vir_so)


def main():
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nact, nocc, nvir = data["nact"], data["nocc"], data["nvir"]; nso = 2 * nact
    dets, index, Hbar = build_sector(data, data["nelec"])
    z = np.load("/tmp/hbar_mbody.npz"); fp, vp = z["fp"], z["vp"]
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sIP = solve_ip(data, E_N); sip_sp = IPD.extract_sip(sIP, data)
    SIP = IPD.build_sip_recon(sip_sp, data); SIP_clean = {m: SIP[m] for m in occ_so(data)}
    zEA = {so_index(a + nocc, s, nact): np.zeros((nso,)*3) for s in range(2) for a in range(nvir)}
    S_ip = build_S(data, dets, index, SIP_clean, zEA)
    occ = occ_so(data); vir = vir_so(data)
    sp = np.zeros((nso,)*4)
    for m in SIP_clean: sp[m] = SIP_clean[m]

    def route(Op):
        comm = S_ip @ Op - Op @ S_ip
        Ms, _, _ = det_singles_block(data, dets, index, comm)
        return Ms   # [i,a,j,b]

    Io = np.eye(nso)   # for full-index einsums we keep SO indices; occ/vir handled by block masks
    # block tensors (full nso, zero outside block)
    fov = np.zeros((nso, nso)); fov[np.ix_(occ, vir)] = fp[np.ix_(occ, vir)]
    wooov = np.zeros((nso,)*4); wooov[np.ix_(occ, occ, occ, vir)] = vp[np.ix_(occ, occ, occ, vir)]
    wvovv = np.zeros((nso,)*4); wvovv[np.ix_(vir, occ, vir, vir)] = vp[np.ix_(vir, occ, vir, vir)]
    dab = np.eye(nso)  # KroneckerDelta on full SO indices; a,b are vir externals

    # s[m,I,J,B] = sp ; sum over dummies via einsum over full nso (blocks zero elsewhere)
    # Fov:  Ms[i,a,j,b]
    F1 = 0.5 * np.einsum("jIia,Ib->iajb", sp, fov, optimize=True)          # f[I,b]s[j,I,i,a]
    F2 = -0.5 * np.einsum("jiJa,Jb->iajb", sp, fov, optimize=True)         # -f[J,b]s[j,i,J,a]
    # delta_ab terms:
    Fd1 = -0.5 * np.einsum("jIiB,IB->ij", sp, fov, optimize=True)          # ->[i,j], times dab[a,b]
    Fd2 = 0.5 * np.einsum("jiJB,JB->ij", sp, fov, optimize=True)
    Fdiag = np.einsum("ij,ab->iajb", Fd1 + Fd2, dab, optimize=True)
    Ms_fov_form = F1 + F2 + Fdiag

    # Wooov: w[I,J,i,B]
    W1 = 0.5 * np.einsum("jIJa,IJib->iajb", sp, wooov, optimize=True)
    W2 = -0.5 * np.einsum("jIJa,JIib->iajb", sp, wooov, optimize=True)
    Wd1 = -0.5 * np.einsum("jIJB,IJiB->ij", sp, wooov, optimize=True)
    Wd2 = 0.5 * np.einsum("jIJB,JIiB->ij", sp, wooov, optimize=True)
    Wdiag = np.einsum("ij,ab->iajb", Wd1 + Wd2, dab, optimize=True)
    Ms_wooov_form = W1 + W2 + Wdiag

    # Wvovv:
    V1 = 0.5 * np.einsum("jIiB,aIBb->iajb", sp, wvovv, optimize=True)
    V2 = -0.5 * np.einsum("jIiB,aIbB->iajb", sp, wvovv, optimize=True)
    V3 = -0.5 * np.einsum("jiJB,aJBb->iajb", sp, wvovv, optimize=True)
    V4 = 0.5 * np.einsum("jiJB,aJbB->iajb", sp, wvovv, optimize=True)
    Ms_wvovv_form = V1 + V2 + V3 + V4

    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    for label, form, blk, k in [
        ("Fov", Ms_fov_form, fov, 1),
        ("Wooov", Ms_wooov_form, wooov, 2),
        ("Wvovv", Ms_wvovv_form, wvovv, 2)]:
        exact = route(H3.opk_matrix(dets, index, blk, k))       # compact (nocc,nvir,nocc,nvir)
        fs = form[np.ix_(oa, va, oa, va)]                        # slice formula to alpha externals
        d = np.linalg.norm(fs - exact) / (np.linalg.norm(exact) + 1e-30)
        c = np.vdot(fs, exact) / (np.vdot(fs, fs) + 1e-30)       # best scale exact ~ c*fs
        dsc = np.linalg.norm(c * fs - exact) / (np.linalg.norm(exact) + 1e-30)
        print(f"  [{label}] raw resid={d:.3e}   best-scale exact={1/c if abs(c)>1e-9 else 0:.4f}*form  "
              f"post-scale resid={dsc:.3e}")

    # ---- pin Wooov coefficients: fit exact to its 4 structural pieces ----
    exact_w = route(H3.opk_matrix(dets, index, wooov, 2))
    pieces = {
        "W1 s[jIJa]w[IJib]": np.einsum("jIJa,IJib->iajb", sp, wooov, optimize=True)[np.ix_(oa, va, oa, va)],
        "W2 s[jIJa]w[JIib]": np.einsum("jIJa,JIib->iajb", sp, wooov, optimize=True)[np.ix_(oa, va, oa, va)],
        "Wd1 dab s[jIJB]w[IJiB]": np.einsum("jIJB,IJiB,ab->iajb", sp, wooov, dab, optimize=True)[np.ix_(oa, va, oa, va)],
        "Wd2 dab s[jIJB]w[JIiB]": np.einsum("jIJB,JIiB,ab->iajb", sp, wooov, dab, optimize=True)[np.ix_(oa, va, oa, va)],
    }
    names = list(pieces); A = np.stack([pieces[n].ravel() for n in names], 1); tv = exact_w.ravel()
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"\n  Wooov exact fit resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.2e}:")
    for n, cc in zip(names, co):
        print(f"    {cc:+.4f}  {n}")


if __name__ == "__main__":
    main()
