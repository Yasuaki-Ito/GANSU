import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
sys.path.insert(0, "script")
import numpy as np
from scipy.linalg import expm
import steom_es_oracle as O
import steom_cfour_weff as C
from pyscf_steom_feff_reference import build_g_canonical_full
Ha = 27.211386245988

xyz = sys.argv[1] if len(sys.argv) > 1 else "xyz/H2O.xyz"
ncore = int(sys.argv[2]) if len(sys.argv) > 2 else 1
d = C.load(xyz, "sto-3g", ncore)
nocc, nvir, dim = d["nocc"], d["nvir"], d["dim"]
_, g_phph, g_phhp, _, _, _ = build_g_canonical_full(
    d["bar"], d["r2_ip"], d["r2_ea"], d["r1_ip"], d["r1_ea"],
    d["occ_idx"], d["vir_idx"], nocc, nvir)
Foo, Fvv = C.build_feff(d)
GsA = np.zeros((dim, dim)); GtA = np.zeros((dim, dim))
for i in range(nocc):
    for a in range(nvir):
        r = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                c = j * nvir + b
                fdg = (Fvv[a, b] if i == j else 0.0) - (Foo[i, j] if a == b else 0.0)
                GsA[r, c] = fdg + 2.0 * g_phhp[b, j, i, a] - g_phph[a, j, b, i]
                GtA[r, c] = fdg - g_phph[a, j, b, i]
data = O.get_active_data(xyz=xyz, basis="sto-3g", ncore=ncore)
dets, index, HbarN = O.build_sector(data, data["nelec"])
hf = O.hf_det(data); iHF = index[hf]; E_N = HbarN[iHF, iHF]
sIP = O.solve_ip(data, E_N); sEA = O.solve_ea(data)
S = O.build_S(data, dets, index, sIP, sEA)
terms = O.s_terms(data, sIP, sEA); K = O.build_K(data, dets, index, terms)
GeS = expm(S) @ HbarN @ expm(-S) - 0.5 * (K @ HbarN + HbarN @ K)
GsE, GtE = O.project_1h1p(data, dets, index, GeS)
GsE = GsE - E_N * np.eye(dim); GtE = GtE - E_N * np.eye(dim)


def lo(G):
    return np.sort(np.linalg.eigvals(G).real)[:5] * Ha


def mi(G):
    return round(np.max(np.abs(np.linalg.eigvals(G).imag)) * Ha, 4)


print("RESULT", xyz)
for tag, GA, GE in [("SINGLET", GsA, GsE), ("TRIPLET", GtA, GtE)]:
    sym = 0.5 * (GA + GA.T)
    print(f"  [{tag}] analytic.real : {np.round(lo(GA), 3)}  max|Im|={mi(GA)}")
    print(f"  [{tag}] SYMMETRIZED   : {np.round(lo(sym), 3)}  max|Im|={mi(sym)}")
    print(f"  [{tag}] TRUE {{e^S}}    : {np.round(lo(GE), 3)}")
