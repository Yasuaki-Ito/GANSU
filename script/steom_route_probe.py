#!/usr/bin/env python3
"""
Per-ROUTE deficiency probe for the STEOM g_phph dressing, on a molecule where
the PySCF EOM-EE-Triplet downfold (exact_phph = -G_t) is VALID (H2O sto-3g),
so the element-wise target is noise-free (triplet => g_phph only, no g_phhp).

Strategy (eigenvalue metric is primary; element-wise projection is the localizer):
  1. Rebuild the g_phph route decomposition (base Wovov, +u_amci [S^IP],
     +u_akei [S^EA], +u_amei [cross]) by calling the SAME code path
     build_g_canonical_full but reconstructing the per-route tensors here so we
     can add them one at a time.
  2. Triplet G = diag(Fvv) - diag(Foo) - g_phph; eigenvalues at each stage vs
     EOM-triplet e_t. Tells which route is too weak (eigenvalue = valid metric).
  3. exact_phph - g_phph(full) = missing tensor M. Project M onto each route's
     tensor T_k:  alpha_k = <M,T_k>/<T_k,T_k>. A large alpha_k on a route that is
     already present => that route is under-scaled (factor bug). alpha for a route
     ~0 with big residual => missing term orthogonal to all routes.

Usage:  wsl python3 script/steom_route_probe.py
        wsl python3 script/steom_route_probe.py xyz/H2O.xyz sto-3g 2
"""
import sys
import numpy as np
sys.path.insert(0, "script")
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full, build_normalized_s

np.set_printoptions(precision=4, suppress=True, linewidth=170)
Ha2eV = 27.211386245988


def route_tensors(d):
    """Reproduce the g_phph per-route tensors exactly as build_g_canonical_full,
    returned scattered into full [a,k,c,i] = [vir,occ,vir,occ] layout."""
    bar = d["bar"]; nocc = d["nocc"]; nvir = d["nvir"]
    occ_idx = d["occ_idx"]; vir_idx = d["vir_idx"]
    Fov = bar["Fov"]; Wvovv = bar["Wvovv"]; Wooov = bar["Wooov"]; Wovoo = bar["Wovoo"]
    Wovov = bar["Wovov"]
    s_IP, s_EA = build_normalized_s(d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
                                    occ_idx, vir_idx, nocc, nvir)
    nM = len(s_IP); nE = len(s_EA)

    base = np.einsum("kaic->akci", Wovov).copy()

    t_amci = np.zeros((nvir, nocc, nvir, nocc))
    for m in range(nM):
        s = s_IP[m]                                  # [i,k,a]
        st = 2.0 * s - s.transpose(1, 0, 2)
        blk = (-np.einsum("kc,ika->aci", Fov, s)
               + np.einsum("alcd,ild->aci", Wvovv, st)
               - np.einsum("aldc,ild->aci", Wvovv, s)
               + np.einsum("kcli,lka->aci", Wovoo, s))
        t_amci[:, occ_idx[m], :, :] += blk

    t_akei = np.zeros((nvir, nocc, nvir, nocc))
    for e in range(nE):
        s = s_EA[e]                                  # [i,a,c]
        st = 2.0 * s - s.transpose(0, 2, 1)
        blk = (-np.einsum("kc,iac->aki", Fov, s)
               + np.einsum("ldki,lad->aki", Wovoo, st)
               - np.einsum("lkid,lad->aki", Wooov, s)
               + np.einsum("akcd,icd->aki", Wvovv, s))
        t_akei[:, :, vir_idx[e], :] += blk

    return base, t_amci, t_akei


def triplet_eig(d, g_ph):
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    Foo, Fvv = C.build_feff(d)
    G = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    v = -g_ph[a, j, b, i]
                    if i == j: v += Fvv[a, b]
                    if a == b: v -= Foo[i, j]
                    G[r, c] = v
    return np.sort(np.linalg.eigvals(G).real)


def main(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=2):
    d = C.load(xyz, basis, ncore)
    nocc, nvir = d["nocc"], d["nvir"]
    print(f"{xyz} {basis} ncore={ncore}  nocc={nocc} nvir={nvir} dim={d['dim']}")

    # full Nooijen g_phph (includes cross u_amei) for reference
    _, g_full, _, _, _, _ = (lambda r: r)(build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir))
    base, t_amci, t_akei = route_tensors(d)
    g_amci = base + t_amci
    g_both = base + t_amci + t_akei

    et = d["e_t"]                       # EOM triplet (valid for H2O)
    print("\n--- TRIPLET eigenvalues (eV), stage vs EOM-triplet ---")
    for tag, g in [("base       ", base), ("+u_amci(IP)", g_amci),
                   ("+u_akei(EA)", g_both), ("full(+cross)", g_full)]:
        e = triplet_eig(d, g) * Ha2eV
        dmax = np.max(np.abs(triplet_eig(d, g) - et)) * 1000
        print(f"  {tag}: {e[:6]}   Δmax(EOM)={dmax:6.2f} mHa")
    print(f"  EOM e_t (eV): {np.sort(et)[:6] * Ha2eV}")

    # exact target (triplet downfold). sanity: must reproduce e_t.
    exact = d["exact_phph"]            # [a,j,b,i]
    e_exact = triplet_eig(d, exact) * Ha2eV
    print(f"\n  sanity exact_phph triplet eigs (eV): {e_exact[:6]}")

    # missing tensor and per-route projection
    M = exact - g_full
    print(f"\n--- element-wise missing M = exact_phph - g_full ---")
    print(f"  ||M|| = {np.linalg.norm(M):.4f}   ||base|| = {np.linalg.norm(base):.4f}"
          f"   ||u_amci|| = {np.linalg.norm(t_amci):.4f}   ||u_akei|| = {np.linalg.norm(t_akei):.4f}")
    for nm, T in [("u_amci", t_amci), ("u_akei", t_akei), ("base", base)]:
        denom = float(np.vdot(T, T))
        alpha = float(np.vdot(T, M)) / denom if denom > 0 else 0.0
        # residual after removing best multiple of this route
        res = np.linalg.norm(M - alpha * T) / np.linalg.norm(M)
        print(f"  project M onto {nm:7s}: alpha={alpha:+.3f}  (M already needs {alpha:+.2f}x more of it); "
              f"resid after = {res:.3f}")

    # 2-parameter best-fit: M ~ a*u_amci + b*u_akei
    A = np.stack([t_amci.ravel(), t_akei.ravel()], axis=1)
    coef, *_ = np.linalg.lstsq(A, M.ravel(), rcond=None)
    r2 = np.linalg.norm(A @ coef - M.ravel()) / np.linalg.norm(M.ravel())
    print(f"\n  2-param fit M ~ a*u_amci + b*u_akei: a={coef[0]:+.3f} b={coef[1]:+.3f}  resid={r2:.3f}")
    print(f"    (a,b ~ small => routes already ~right; large => those routes under-produced;")
    print(f"     resid ~1 => missing term is ORTHOGONAL to both single routes)")

    # where is M largest? diagonal vs off-diagonal
    diag = 0.0; off = 0.0
    big = []
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    v = M[a, j, b, i]
                    if i == j and a == b:
                        diag = max(diag, abs(v)); big.append((abs(v), i, a, j, b, v))
                    else:
                        off = max(off, abs(v))
    print(f"\n  max|M| diagonal(i=j,a=b) = {diag:.4f}   off-diagonal = {off:.4f}")
    big.sort(reverse=True)
    print("  top diagonal M[a,i,a,i]: " +
          ", ".join(f"(o{i},v{a})={v:+.3f}" for _, i, a, j, b, v in big[:6]))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*([a[0]] if a else []),
         **({"basis": a[1]} if len(a) > 1 else {}),
         **({"ncore": int(a[2])} if len(a) > 2 else {}))
