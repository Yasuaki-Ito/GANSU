#!/usr/bin/env python3
"""
VALID linear fit of the g_phph dressing against the ORCA-reconstructed TRUE
target (not the invalid EOM-derived one). target_g_phph = F_diag - V diag(w_ORCA)
V^-1 with V = my Gt eigenvectors (verified to match ORCA). Then fit
  target_g_phph - base = sum_k c_k T_k
over candidate single-route contraction tensors. Clean coefficients / a large
coefficient on a term ABSENT from build_g_canonical_full reveals the missing term.

g_phph tensor layout (as in build_g_canonical_full): [a,k,c,i] = [vir,occ,vir,occ];
used as g_phph[a,j,b,i]. S^IP route -> [a,c,i] per active-occ m (placed at k slot);
S^EA route -> [a,k,i] per active-vir e (placed at c slot).
"""
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as C
from steom_eig_debug import nooijen_g
from pyscf_steom_feff_reference import build_normalized_s
np.set_printoptions(precision=4, suppress=True, linewidth=170)
Ha2eV = 27.211386245988

# ORCA triplet STEOM eigenvalues (ch2o sto-3g frozen=2, complete active), 20 roots, eV
ORCA_T = np.array([4.197, 5.959, 9.160, 13.336, 13.547, 15.508, 17.331, 17.973,
                   19.311, 19.367, 19.925, 21.043, 21.128, 22.065, 22.441, 23.490,
                   26.085, 27.480, 28.417, 30.617]) / Ha2eV


def reconstruct_target_gphph(d, gphN, Foo, Fvv):
    nocc, nvir, dim = d["nocc"], d["nvir"], d["dim"]
    Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    v = -gphN[a, j, b, i]
                    if i == j: v += Fvv[a, b]
                    if a == b: v -= Foo[i, j]
                    Gt[r, c] = v
    w, V = np.linalg.eig(Gt)
    order = np.argsort(w.real)
    wt = w.real.copy()
    for k in range(min(len(ORCA_T), dim)):
        wt[order[k]] = ORCA_T[k]
    tG = (V @ np.diag(wt) @ np.linalg.inv(V)).real
    tgt = np.zeros((nvir, nocc, nvir, nocc))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    Fpart = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    tgt[a, j, b, i] = Fpart - tG[i * nvir + a, j * nvir + b]
    return tgt


def phph_candidates(d):
    """name -> full [a,k,c,i] tensor (vir,occ,vir,occ) dressing candidate."""
    bar = d["bar"]; nocc, nvir = d["nocc"], d["nvir"]
    occ_idx, vir_idx = d["occ_idx"], d["vir_idx"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    occ_idx, vir_idx, nocc, nvir)
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wovoo = bar["Wovoo"]; Wooov = bar["Wooov"]
    nM, nE = len(s_IP), len(s_EA)
    cand = {}

    def IP(fn):  # per m builder -> [a,c,i]; scatter to [a, occ_idx[m], c, i]
        out = np.zeros((nvir, nocc, nvir, nocc))
        for m in range(nM):
            out[:, occ_idx[m], :, :] += fn(s_IP[m])
        return out

    def IPt(fn):  # tilde over the two occ of s_IP
        out = np.zeros((nvir, nocc, nvir, nocc))
        for m in range(nM):
            out[:, occ_idx[m], :, :] += fn(2 * s_IP[m] - s_IP[m].transpose(1, 0, 2))
        return out

    # s_IP[m] = [k,i,d] (occ,occ,vir).  outputs [a,c,i]
    cand["IP_F"]   = IP(lambda s: -np.einsum("kc,kia->aci", Fov, s))            # Fov
    cand["IP_Ft"]  = IPt(lambda s: -np.einsum("kc,kia->aci", Fov, s))
    cand["IP_Vd"]  = IP(lambda s:  np.einsum("alcd,ild->aci", Wvovv, s))        # Wvovv direct
    cand["IP_Vt"]  = IPt(lambda s: np.einsum("alcd,ild->aci", Wvovv, s))        # Wvovv tilde
    cand["IP_Vs"]  = IP(lambda s: -np.einsum("aldc,ild->aci", Wvovv, s))        # Wvovv swap dc
    cand["IP_O"]   = IP(lambda s:  np.einsum("kcli,kla->aci", Wovoo, s))        # Wovoo
    cand["IP_Ot"]  = IPt(lambda s: np.einsum("kcli,kla->aci", Wovoo, s))
    cand["IP_W"]   = IP(lambda s:  np.einsum("klic,kla->aci", Wooov, s))        # Wooov (MISSING?)
    cand["IP_Wt"]  = IPt(lambda s: np.einsum("klic,kla->aci", Wooov, s))

    def EA(fn):  # per e -> [a,k,i]; scatter to [a,k, vir_idx[e], i]
        out = np.zeros((nvir, nocc, nvir, nocc))
        for e in range(nE):
            out[:, :, vir_idx[e], :] += fn(s_EA[e])
        return out

    def EAt(fn):  # tilde over the two vir of s_EA
        out = np.zeros((nvir, nocc, nvir, nocc))
        for e in range(nE):
            out[:, :, vir_idx[e], :] += fn(2 * s_EA[e] - s_EA[e].transpose(0, 2, 1))
        return out

    # s_EA[e] = [i,a,d] (occ,vir,vir). outputs [a,k,i]
    cand["EA_F"]   = EA(lambda s: -np.einsum("kd,iad->aki", Fov, s))            # Fov
    cand["EA_Ft"]  = EAt(lambda s: -np.einsum("kd,iad->aki", Fov, s))
    cand["EA_Vd"]  = EA(lambda s:  np.einsum("akcd,icd->aki", Wvovv, s))        # Wvovv
    cand["EA_O"]   = EA(lambda s:  np.einsum("lkid,lad->aki", Wovoo, s) if False else np.einsum("ldki,lad->aki", Wovoo, s))  # Wovoo
    cand["EA_Ot"]  = EAt(lambda s: np.einsum("ldki,lad->aki", Wovoo, s))
    cand["EA_W"]   = EA(lambda s: -np.einsum("lkid,lad->aki", Wooov, s))        # Wooov
    cand["EA_Wt"]  = EAt(lambda s: -np.einsum("lkid,lad->aki", Wooov, s))
    return cand


def main():
    d = C.load("xyz/Formaldehyde.xyz", "sto-3g", 2)
    nocc, nvir = d["nocc"], d["nvir"]
    _, gphN, _ = nooijen_g(d)
    Foo, Fvv = C.build_feff(d)
    tgt = reconstruct_target_gphph(d, gphN, Foo, Fvv)
    base = np.einsum("kaic->akci", d["bar"]["Wovov"])
    miss = tgt - gphN                       # full missing dressing (target - mine)
    needed = tgt - base                     # total dressing needed (target - base)
    cand = phph_candidates(d)
    names = list(cand.keys())
    M = np.stack([cand[n].ravel() for n in names], axis=1)
    # fit the TOTAL needed dressing (target - base) with all candidates
    y = needed.ravel()
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = np.linalg.norm(M @ c - y) / np.linalg.norm(y)
    print("FIT target_dressing (target_g_phph - base) = sum c_k T_k")
    print(f"  rel residual = {resid:.3f}   (‖needed‖={np.linalg.norm(y):.4f})")
    for n, ci in zip(names, c):
        mark = "  <==" if abs(ci) > 0.3 else ""
        print(f"    {n:7s} {ci:+.4f}{mark}")
    # how much does my current Nooijen miss?
    print(f"\n  ‖miss (target-Nooijen)‖ = {np.linalg.norm(miss):.4f}  "
          f"‖needed‖ = {np.linalg.norm(needed):.4f}")
    # also report norm of each candidate (scale awareness)
    print("\n  candidate norms:", {n: round(float(np.linalg.norm(cand[n])), 3) for n in names})


if __name__ == "__main__":
    main()
