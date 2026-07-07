#!/usr/bin/env python3
"""Verify the COMPLETE sympy-derived g_phph route (Fov 1-body + full 2-body) is
machine-exact vs the exact det route [S_ip, Hbar_{1,2}].  v = AntiSymmetricTensor
coeff = vp (full antisym 2-body Hbar).  s antisymmetry restricts dummy kinds.

Run: wsl python3 script/steom_gphph_final_verify.py
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
    dab = np.eye(nso)
    fov = np.zeros((nso, nso)); fov[np.ix_(occ, vir)] = fp[np.ix_(occ, vir)]

    # exact total route [S_ip, Hbar_{1,2}] = [S_ip, op1(fp)+op2(vp)]
    O12 = H3.opk_matrix(dets, index, fp, 1) + H3.opk_matrix(dets, index, vp, 2)
    comm = S_ip @ O12 - O12 @ S_ip
    exact, _, _ = det_singles_block(data, dets, index, comm)   # (nocc,nvir,nocc,nvir)

    # ---- sympy formula (SO, full nso; s antisym restricts dummy kinds) ----
    # Fov (1-body):  1/2(f[I,b]s[j,I,i,a] - f[J,b]s[j,i,J,a]) - dab/2(f[I,B]s[j,I,i,B]-f[J,B]s[j,i,J,B])
    F = (0.5 * np.einsum("jIia,Ib->iajb", sp, fov, optimize=True)
         - 0.5 * np.einsum("jiJa,Jb->iajb", sp, fov, optimize=True)
         + np.einsum("ij,ab->iajb", -0.5 * np.einsum("jIiB,IB->ij", sp, fov, optimize=True)
                     + 0.5 * np.einsum("jiJB,JB->ij", sp, fov, optimize=True), dab, optimize=True))
    # 2-body (from AntiSymmetricTensor derivation):
    V = (-0.5 * np.einsum("jIKa,IKbi->iajb", sp, vp, optimize=True)              # -1/2 v[IK,bi]s[j,IK,a]
         - 0.5 * np.einsum("jIiA,aIbA->iajb", sp, vp, optimize=True)            # -1/2 v[aI,bA]s[j,I,i,A]
         + 0.5 * np.einsum("jiIA,aIbA->iajb", sp, vp, optimize=True)            # +1/2 v[aI,bA]s[j,i,I,A]
         - 0.5 * np.einsum("ij,ab->iajb",
                           np.einsum("jIKA,IKiA->ij", sp, vp, optimize=True), dab, optimize=True))  # -dab/2 v[IK,iA]s[j,IK,A]
    oa = [so_index(i, 0, nact) for i in range(nocc)]
    va = [so_index(a + nocc, 0, nact) for a in range(nvir)]
    def sl(T): return T[np.ix_(oa, va, oa, va)]
    # separate Ms_1, Ms_2 exact
    c1 = S_ip @ H3.opk_matrix(dets, index, fp, 1) - H3.opk_matrix(dets, index, fp, 1) @ S_ip
    Ms1, _, _ = det_singles_block(data, dets, index, c1)
    c2 = S_ip @ H3.opk_matrix(dets, index, vp, 2) - H3.opk_matrix(dets, index, vp, 2) @ S_ip
    Ms2, _, _ = det_singles_block(data, dets, index, c2)
    Ms2_alt = exact - Ms1   # = Ms_full - Ms_1  (true Hbar 2-body route, Ms_3=0)
    print(f"  F vs Ms_1: resid={np.linalg.norm(sl(F)-Ms1)/np.linalg.norm(Ms1):.3e}")
    print(f"  V vs Ms_2(op2): resid={np.linalg.norm(sl(V)-Ms2)/np.linalg.norm(Ms2):.3e}")
    print(f"  V vs Ms_2(full-Ms1): resid={np.linalg.norm(sl(V)-Ms2_alt)/np.linalg.norm(Ms2_alt):.3e}")
    print(f"  ||Ms2(op2)-Ms2(full-Ms1)||={np.linalg.norm(Ms2-Ms2_alt):.3e}")
    # scale check
    cV = np.vdot(sl(V), Ms2) / (np.vdot(sl(V), sl(V)) + 1e-30)
    print(f"  best-scale Ms2(op2) = {1/cV if abs(cV)>1e-9 else 0:.4f} * V   post={np.linalg.norm(cV*sl(V)-Ms2)/np.linalg.norm(Ms2):.3e}")
    # ---- find vp index convention matching sympy AntiSymmetricTensor v[(p,q),(r,s)] ----
    def Vform(w):
        return (-0.5 * np.einsum("jIKa,IKbi->iajb", sp, w, optimize=True)
                - 0.5 * np.einsum("jIiA,aIbA->iajb", sp, w, optimize=True)
                + 0.5 * np.einsum("jiIA,aIbA->iajb", sp, w, optimize=True)
                - 0.5 * np.einsum("ij,ab->iajb", np.einsum("jIKA,IKiA->ij", sp, w, optimize=True), dab, optimize=True))
    print("\n  vp-convention scan (V vs Ms_2):")
    for nm, perm in [("pqrs", (0,1,2,3)), ("pqsr", (0,1,3,2)), ("qprs", (1,0,2,3)),
                     ("rspq", (2,3,0,1)), ("qpsr", (1,0,3,2)), ("prqs", (0,2,1,3))]:
        w = np.transpose(vp, perm)
        r = np.linalg.norm(sl(Vform(w)) - Ms2) / np.linalg.norm(Ms2)
        print(f"    vp[{nm}]: resid={r:.3e}")

    # ---- pin 2-body coeffs: fit Ms_2 to the sympy structural terms (free coeffs) ----
    P = {
        "v[IK,bi]s[j,IK,a]": sl(np.einsum("jIKa,IKbi->iajb", sp, vp, optimize=True)),
        "v[aI,bA]s[j,I,i,A]": sl(np.einsum("jIiA,aIbA->iajb", sp, vp, optimize=True)),
        "v[aI,bA]s[j,i,I,A]": sl(np.einsum("jiIA,aIbA->iajb", sp, vp, optimize=True)),
        "dab v[IK,iA]s[j,IK,A]": sl(np.einsum("jIKA,IKiA,ab->iajb", sp, vp, dab, optimize=True)),
    }
    names = list(P); A = np.stack([P[n].ravel() for n in names], 1); tv = Ms2.ravel()
    co, *_ = np.linalg.lstsq(A, tv, rcond=None)
    print(f"\n  Ms_2 fit to sympy structure: resid={np.linalg.norm(A@co-tv)/np.linalg.norm(tv):.3e}")
    for n, cc in zip(names, co):
        print(f"    {cc:+.4f}  {n}")


if __name__ == "__main__":
    main()
