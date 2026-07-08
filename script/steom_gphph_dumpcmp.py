#!/usr/bin/env python3
"""Compare the C++ PROJ_FULL route arrays (GANSU_STEOM_PROJ_DUMP) against the
Python zero-padded evaluation with the SAME inputs.  Localizes the C++
emission bug under partial-active space (root-embedded amplitudes).

Run:  wsl python3 script/steom_gphph_dumpcmp.py /tmp/projdump_local/projdump
"""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_gphph_fullgate as FG


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "/tmp/projdump_local/projdump"
    dims = np.fromfile(pre + "_dims.bin", dtype=np.int32)
    NO, NV, NMo, NMv = [int(x) for x in dims]
    aocc = np.fromfile(pre + "_aocc.bin", dtype=np.int32)
    avir = np.fromfile(pre + "_avir.bin", dtype=np.int32)
    print(f"NO={NO} NV={NV} NMo={NMo} NMv={NMv}  aocc={aocc} avir={avir}")
    rd = lambda nm, shape: np.fromfile(pre + nm, dtype=np.float64).reshape(shape)
    Fov   = rd("_fov.bin",   (NO, NV))
    Wooov = rd("_wooov.bin", (NO, NO, NO, NV))
    Wvovv = rd("_wvovv.bin", (NV, NO, NV, NV))
    ERIov = rd("_eriov.bin", (NO, NV, NO, NV))
    sIP   = rd("_sip.bin",   (NMo, NO, NO, NV))
    sEA   = rd("_sea.bin",   (NMv, NO, NV, NV))
    rph_c = rd("_rph.bin",   (NO, NV, NO, NV))
    rmc_c = rd("_rmc.bin",   (NO, NV, NO, NV))

    # zero-padded (orbital-embedded) amplitudes = the C++ root-loop semantics
    rx = np.zeros((NO, NO, NO, NV))
    for r, m in enumerate(aocc): rx[m] = sIP[r]
    ry = np.zeros((NV, NO, NV, NV))
    for r, e in enumerate(avir): ry[e] = sEA[r]

    arrays = {'Fov': Fov, 'Xov': Fov, 'Wooov': Wooov, 'Wvovv': Wvovv,
              'eri_ovov': ERIov, 'rx': rx, 'ry': ry}
    Ms_terms, Mc_terms = FG.load_terms()
    import steom_gphph_cppgen_full as CGF
    Msj, Mcj = CGF.parse_json()
    route_ph_terms = GEN.subtract(Mcj, Msj)
    ph_py = GEN.evaluate(route_ph_terms, arrays, NO, NV)
    mc_py = GEN.evaluate(Mcj, arrays, NO, NV)
    print(f"route_ph:  ||C++||={np.linalg.norm(rph_c):.6f}  ||py||={np.linalg.norm(ph_py):.6f}"
          f"  max|d|={np.max(np.abs(rph_c - ph_py)):.3e}")
    print(f"route_mc:  ||C++||={np.linalg.norm(rmc_c):.6f}  ||py||={np.linalg.norm(mc_py):.6f}"
          f"  max|d|={np.max(np.abs(rmc_c - mc_py)):.3e}")

    # localize: delta terms only vs non-delta
    isdelta = lambda t: any(a.startswith('delta') for a, _ in t[1])
    for tag, terms in [("ph delta-only", [t for t in route_ph_terms if isdelta(t)]),
                       ("ph nodelta   ", [t for t in route_ph_terms if not isdelta(t)])]:
        v = GEN.evaluate(terms, arrays, NO, NV)
        print(f"  [{tag}] ||py||={np.linalg.norm(v):.6f}")
    D = rph_c - ph_py
    # class decomposition of the diff
    off = semi = diag = 0.0
    for i in range(NO):
        for a in range(NV):
            for j in range(NO):
                for b in range(NV):
                    v2 = D[i, a, j, b] ** 2
                    if i != j and a != b: off += v2
                    elif i == j and a == b: diag += v2
                    else: semi += v2
    print(f"  diff classes: off={off**.5:.3e} semi={semi**.5:.3e} diag={diag**.5:.3e}")
    # largest offenders
    idx = np.unravel_index(np.argsort(-np.abs(D).ravel())[:8], D.shape)
    for k in range(8):
        i, a, j, b = idx[0][k], idx[1][k], idx[2][k], idx[3][k]
        print(f"   d[{i},{a},{j},{b}] = {D[i,a,j,b]:+.6e}  (C++ {rph_c[i,a,j,b]:+.6e}  py {ph_py[i,a,j,b]:+.6e})")


if __name__ == "__main__":
    main()


def diag_analysis(pre):
    """route diagonal decomposition by term class + per-root amplitude norms."""
    import steom_gphph_cppgen_full as CGF
    dims = np.fromfile(pre + "_dims.bin", dtype=np.int32)
    NO, NV, NMo, NMv = [int(x) for x in dims]
    aocc = np.fromfile(pre + "_aocc.bin", dtype=np.int32)
    avir = np.fromfile(pre + "_avir.bin", dtype=np.int32)
    rd = lambda nm, shape: np.fromfile(pre + nm, dtype=np.float64).reshape(shape)
    Fov = rd("_fov.bin", (NO, NV)); Wooov = rd("_wooov.bin", (NO, NO, NO, NV))
    Wvovv = rd("_wvovv.bin", (NV, NO, NV, NV)); ERIov = rd("_eriov.bin", (NO, NV, NO, NV))
    sIP = rd("_sip.bin", (NMo, NO, NO, NV)); sEA = rd("_sea.bin", (NMv, NO, NV, NV))
    rph_c = rd("_rph.bin", (NO, NV, NO, NV))
    print("per-root ||siP||:", np.round([np.linalg.norm(sIP[r]) for r in range(NMo)], 2))
    print("per-root ||seA||:", np.round([np.linalg.norm(sEA[r]) for r in range(NMv)], 2))
    rx = np.zeros((NO, NO, NO, NV))
    for r, m in enumerate(aocc): rx[m] = sIP[r]
    ry = np.zeros((NV, NO, NV, NV))
    for r, e in enumerate(avir): ry[e] = sEA[r]
    arrays = {'Fov': Fov, 'Xov': Fov, 'Wooov': Wooov, 'Wvovv': Wvovv,
              'eri_ovov': ERIov, 'rx': rx, 'ry': ry}
    Msj, Mcj = CGF.parse_json()
    route_ph_terms = GEN.subtract(Mcj, Msj)
    isdelta = lambda t: any(a.startswith('delta') for a, _ in t[1])
    isquad = lambda t: sum(1 for a, _ in t[1] if a in ('rx', 'ry')) >= 2
    classes = {
        'lin-delta ': [t for t in route_ph_terms if isdelta(t) and not isquad(t)],
        'quad-delta': [t for t in route_ph_terms if isdelta(t) and isquad(t)],
        'lin-nodel ': [t for t in route_ph_terms if not isdelta(t) and not isquad(t)],
        'quad-nodel': [t for t in route_ph_terms if not isdelta(t) and isquad(t)],
    }
    # HOMO index = highest aocc orbital, LUMO = lowest avir
    H = int(max(aocc)); L = int(min(avir))
    print(f"HOMO(win)={H} LUMO(win)={L}   C++ route_ph diag[H,L]={rph_c[H, L, H, L]:+.6f}")
    tot = 0.0
    for nm, terms in classes.items():
        v = GEN.evaluate(terms, arrays, NO, NV)
        d = v[H, L, H, L]
        tot += d
        dsum = np.mean([v[i, a, i, a] for i in range(NO) for a in range(NV)])
        print(f"  [{nm}] {len(terms):3d} terms  diag[H,L]={d:+.6f}  mean-diag={dsum:+.6f}")
    print(f"  sum diag[H,L] = {tot:+.6f}")


if len(sys.argv) > 2 and sys.argv[2] == "diag":
    diag_analysis(sys.argv[1])
