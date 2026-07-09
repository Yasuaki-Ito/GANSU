#!/usr/bin/env python3
"""ORCA 8+8 GATE for the complete g_phhp (STEOM singlet coupling block).

続50 finding: GANSU triplet == ORCA triplet (3-9 meV) => F_eff & g_phph correct;
the entire deficit is the singlet-only g_phhp (ZEROED EA/cross).  This gate runs
H2O sto-3g FC1 FULL-S (ORCA is active-space invariant, so full-S comparison is
legitimate) and scores candidate g_phhp builds against the ORCA truth:

  singlet: 11.849 13.601 16.102 18.238 21.367 27.379 39.955 40.523  eV
  triplet: 10.257 12.684 13.073 14.743 17.855 19.207 34.549         eV
  (s177 ~/steom_ref/h2o/h2o_full8{,t}.out, compositions in memory 続50)

Candidates: shipped (build_g_canonical_full, GANSU bit-exact), CFOUR single-route
(ujaei/umabi, no cross), CFOUR full with cross variants v0..3 (uajmi ambiguity).

Run:  wsl python3 script/steom_gphhp_gate.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np
sys.path.insert(0, "script")

import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full

Ha2eV = 27.211386245988
ORCA_S = [11.849, 13.601, 16.102, 18.238, 21.367, 27.379, 39.955, 40.523]
ORCA_T = [10.257, 12.684, 13.073, 14.743, 17.855, 19.207, 34.549]
GANSU_S = [11.7726, 13.5225, 16.9632, 16.9975]
GANSU_T = [10.2539, 12.6753, 13.0750, 14.7470]


def restrict(d, occ_keep, vir_keep):
    """ORCA/GANSU-style active-space restriction: keep only IP roots on occ_keep
    and EA roots on vir_keep (drops e.g. the pathological deepest-occ channel).
    The 1h1p space stays FULL; only the S dressing shrinks."""
    from pyscf_steom_feff_reference import build_x_matrices
    d2 = dict(d)
    so = [k for k, o in enumerate(d["occ_idx"]) if o in occ_keep]
    sv = [k for k, v in enumerate(d["vir_idx"]) if v in vir_keep]
    d2["r1_ip"] = [d["r1_ip"][k] for k in so]
    d2["r2_ip"] = [d["r2_ip"][k] for k in so]
    d2["occ_idx"] = [d["occ_idx"][k] for k in so]
    d2["r1_ea"] = [d["r1_ea"][k] for k in sv]
    d2["r2_ea"] = [d["r2_ea"][k] for k in sv]
    d2["vir_idx"] = [d["vir_idx"][k] for k in sv]
    Xi_a, Xe_a = build_x_matrices(d2["r1_ip"], d2["r1_ea"],
                                  d2["occ_idx"], d2["vir_idx"])
    # scatter active-square X[root, m_active] -> (act x FULL) as assemble_g expects
    Xi = np.zeros((len(so), d["nocc"])); Xi[:, d2["occ_idx"]] = Xi_a
    Xe = np.zeros((len(sv), d["nvir"])); Xe[:, d2["vir_idx"]] = Xe_a
    d2["X_IP"] = Xi; d2["X_EA"] = Xe
    return d2


def assemble(d, g_phhp, g_phph, Foo, Fvv):
    nocc = d["nocc"]; nvir = d["nvir"]; dim = d["dim"]
    Gs = np.zeros((dim, dim)); Gt = np.zeros((dim, dim))
    for i in range(nocc):
        for a in range(nvir):
            r = i * nvir + a
            for j in range(nocc):
                for b in range(nvir):
                    c = j * nvir + b
                    hp = g_phhp[b, j, i, a]; ph = g_phph[a, j, b, i]
                    f = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                    Gs[r, c] = f + 2.0 * hp - ph
                    Gt[r, c] = f - ph
    return Gs, Gt


def eig_lab(G, nvir):
    w, v = np.linalg.eig(G)
    idx = np.argsort(w.real)
    w = w[idx]; v = v[:, idx]
    labs = []
    for k in range(len(w)):
        p = int(np.argmax(np.abs(v[:, k])))
        labs.append(f"{p // nvir}>{p % nvir}")
    return w.real * Ha2eV, np.abs(w.imag).max() * Ha2eV, labs


def score(name, d, g_phhp, g_phph, Foo, Fvv):
    Gs, Gt = assemble(d, g_phhp, g_phph, Foo, Fvv)
    ws, ims, ls = eig_lab(Gs, d["nvir"])
    wt, imt, lt = eig_lab(Gt, d["nvir"])
    # RMS vs ORCA on the first NCMP roots (holes 1-3; hole-0 states excluded
    # because the (3,2) active dressing leaves them undressed)
    NCMP = 6
    rms_s = np.sqrt(np.mean((ws[:NCMP] - np.array(ORCA_S[:NCMP])) ** 2))
    rms_t = np.sqrt(np.mean((wt[:NCMP] - np.array(ORCA_T[:NCMP])) ** 2))
    print(f"--- {name} ---")
    print("  S:", " ".join(f"{x:7.3f}" for x in ws), f" rmsS={rms_s:.3f}",
          f"(Im {ims:.3f})" if ims > 1e-3 else "")
    print("  lab", " ".join(f"{x:>7s}" for x in ls))
    print("  T:", " ".join(f"{x:7.3f}" for x in wt), f" rmsT={rms_t:.3f}",
          f"(Im {imt:.3f})" if imt > 1e-3 else "")
    return rms_s, rms_t


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    d = C.load("xyz/H2O.xyz", "sto-3g", 1)      # FC1
    d = restrict(d, occ_keep=(1, 2, 3), vir_keep=(0, 1))   # ORCA IP3/EA2-equivalent
    nocc = d["nocc"]; nvir = d["nvir"]
    print(f"nocc={nocc} nvir={nvir}  active occ={d['occ_idx']} vir={d['vir_idx']}")
    print("ORCA S:", " ".join(f"{x:7.3f}" for x in ORCA_S))
    print("ORCA T:", " ".join(f"{x:7.3f}" for x in ORCA_T))
    print("GANSU shipped S:", GANSU_S, " T:", GANSU_T)

    Foo, Fvv = C.build_feff(d)
    T = C.cfour_tensors(d)

    # shipped (GANSU bit-exact Nooijen port)
    Ga, g_ph_a, g_hp_a, u_amei, u_bmje, _ = build_g_canonical_full(
        d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
        d["occ_idx"], d["vir_idx"], nocc, nvir)
    score("shipped (build_g_canonical_full)", d, g_hp_a, g_ph_a, Foo, Fvv)

    # CFOUR single-route (no cross)
    ujaei, ujaie = C.build_ujaei(T)
    umabi, umaib = C.build_umabi(T)
    g_hp, g_ph = C.assemble_g(d, T, ujaei, ujaie, umabi, umaib)
    score("CFOUR single-route", d, g_hp, g_ph, Foo, Fvv)

    # keep shipped g_phph (triplet-validated), CFOUR g_phhp
    score("CFOUR g_phhp + shipped g_phph", d, g_hp, g_ph_a, Foo, Fvv)

    # CFOUR full with cross variants
    uei, uam, uijke = C.build_uei_uam_uijke(T)
    X_EA = d["X_EA"]
    for v in range(4):
        uajmi = C.build_uajmi(T, v); uajim = C.build_uajmi(T, v)
        umaei, umaie = C.build_umaei(T, uei, uam, uijke, uajmi, uajim)
        umabi_t = umabi + np.einsum("EMBJ,EA->AMBJ", umaei, X_EA)
        umaib_t = umaib + np.einsum("EMBJ,EA->AMBJ", umaie, X_EA)
        g_hp2, g_ph2 = C.assemble_g(d, T, ujaei, ujaie, umabi_t, umaib_t)
        score(f"CFOUR full cross v{v}", d, g_hp2, g_ph2, Foo, Fvv)
        score(f"CFOUR full cross v{v} + shipped g_phph", d, g_hp2, g_ph_a, Foo, Fvv)



if __name__ == "__main__":
    main()
