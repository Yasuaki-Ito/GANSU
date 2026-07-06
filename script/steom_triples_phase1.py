#!/usr/bin/env python3
"""Phase 1 of implicit-triples: characterise Δ = G_true({e^S}) − G_analytic(GANSU
routes) in the 1h1p space. Δ is exactly the "implicit triples" correction GANSU
lacks. We measure its magnitude, its symmetric/antisymmetric split (the antisym
part of G_analytic is what makes it complex; how does Δ relate?), and whether the
existing routes span it (route-fit residual = derivation difficulty).

  wsl OMP_NUM_THREADS=1 python3 script/steom_triples_phase1.py [ncore] [xyz]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, "script")
import steom_es_oracle as O
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
Ha = 27.211386245988


def analytic_G(d):
    nocc, nvir, dim = d["nocc"], d["nvir"], d["dim"]
    _, g_phph, g_phhp, _, _, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    Foo, Fvv = C.build_feff(d)
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    Gs[r, c] = fdg + 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                    Gt[r, c] = fdg - g_phph[a, j, b, i]
    return Gs, Gt


def true_G(xyz, basis, ncore):
    data = O.get_active_data(xyz=xyz, basis=basis, ncore=ncore)
    dets, index, HbarN = O.build_sector(data, data["nelec"])
    hf = O.hf_det(data); iHF = index[hf]; E_N = HbarN[iHF, iHF]
    sIP = O.solve_ip(data, E_N); sEA = O.solve_ea(data)
    S = O.build_S(data, dets, index, sIP, sEA)
    terms = O.s_terms(data, sIP, sEA); K = O.build_K(data, dets, index, terms)
    GeS = expm(S) @ HbarN @ expm(-S) - 0.5 * (K @ HbarN + HbarN @ K)
    Gp = expm(S) @ HbarN @ expm(-S)
    dim = data["nocc"] * data["nvir"]
    GsE, GtE = O.project_1h1p(data, dets, index, GeS)
    GsP, GtP = O.project_1h1p(data, dets, index, Gp)
    return (GsE - E_N*np.eye(dim), GtE - E_N*np.eye(dim),
            GsP - E_N*np.eye(dim), GtP - E_N*np.eye(dim))


def anal(tag, GA, GE, GP):
    dA = GA - 0.5*(GA+GA.T)   # antisym part of analytic (source of complex)
    dTrue = GE - GP           # {e^S} normal-ordering correction (K-term)
    Delta = GE - GA           # full implicit-triples correction GANSU lacks
    Delta_p = GP - GA         # higher-order-in-expm (plain, no {e^S}) correction
    print(f"\n[{tag}]")
    print(f"  ||G_analytic||={np.linalg.norm(GA):.4f}  ||G_true(eS)||={np.linalg.norm(GE):.4f}")
    print(f"  ||Δ = G_true − G_analytic|| = {np.linalg.norm(Delta):.4f}  "
          f"({100*np.linalg.norm(Delta)/np.linalg.norm(GA):.1f}% of G)")
    print(f"    of which: Δ_plain(GP−GA)={np.linalg.norm(Delta_p):.4f}  "
          f"{{e^S}}-K-term(GE−GP)={np.linalg.norm(dTrue):.4f}")
    print(f"  antisym(G_analytic)={np.linalg.norm(GA-GA.T)/2:.4f} (→complex)  "
          f"antisym(G_true)={np.linalg.norm(GE-GE.T)/2:.4f}  "
          f"antisym(Δ)={np.linalg.norm(Delta-Delta.T)/2:.4f}")
    # eigenvalue effect
    def lo(G): return np.sort(np.linalg.eigvals(G).real)[:4]*Ha
    def mi(G): return np.max(np.abs(np.linalg.eigvals(G).imag))*Ha
    print(f"  eig analytic (eV): {np.round(lo(GA),3)}  max|Im|={mi(GA):.3f}")
    print(f"  eig true    (eV): {np.round(lo(GE),3)}  max|Im|={mi(GE):.3f}")


def main():
    ncore = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    xyz = sys.argv[2] if len(sys.argv) > 2 else "xyz/H2O.xyz"
    basis = "sto-3g"
    d = C.load(xyz, basis, ncore)
    GsA, GtA = analytic_G(d)
    GsE, GtE, GsP, GtP = true_G(xyz, basis, ncore)
    print(f"== {xyz} {basis} ncore={ncore} — implicit-triples Δ characterisation ==")
    anal("SINGLET", GsA, GsE, GsP)
    anal("TRIPLET", GtA, GtE, GtP)


if __name__ == "__main__":
    main()
