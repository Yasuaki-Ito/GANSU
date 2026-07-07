#!/usr/bin/env python3
"""Local (WSL g++) numerical test of the EMITTED include/steom_gphph_projection.inc:
dump the H2O arrays + Python-reference routes, compile a host-only harness that
includes the .inc verbatim, run, compare.  Catches any emission bug before the
remote CUDA build.

Run:  wsl python3 script/steom_gphph_inc_test.py
"""
import os, sys, subprocess, struct
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
sys.path.insert(0, "script")
import steom_gphph_spatial_gen as GEN
import steom_cfour_weff as CW
import steom_ip_route_derive as IPD, steom_ea_spinadapt as EA
from steom_fockspace_ref import get_active_data, build_sector, solve_ip, solve_ea, hf_det

WORK = "/tmp/proj_inc_test"

HARNESS = r"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
typedef double real_t;

static std::vector<real_t> readbin(const char* p, size_t n) {
    std::vector<real_t> v(n);
    FILE* f = fopen(p, "rb");
    if (!f || fread(v.data(), sizeof(real_t), n, f) != n) { fprintf(stderr, "read fail %s\n", p); exit(1); }
    fclose(f); return v;
}

int main(int argc, char** argv) {
    int NO, NV;
    { FILE* f = fopen(DATA_DIR "/dims.txt", "r"); if (fscanf(f, "%d %d", &NO, &NV) != 2) return 1; fclose(f); }
    const int NMo = NO, NMv = NV;
    std::vector<int> active_occ_idx_(NMo), active_vir_idx_(NMv);
    for (int m = 0; m < NMo; ++m) active_occ_idx_[m] = m;
    for (int e = 0; e < NMv; ++e) active_vir_idx_[e] = e;
    std::vector<real_t> Fov   = readbin(DATA_DIR "/fov.bin",   (size_t)NO*NV);
    std::vector<real_t> Wooov = readbin(DATA_DIR "/wooov.bin", (size_t)NO*NO*NO*NV);
    std::vector<real_t> Wvovv = readbin(DATA_DIR "/wvovv.bin", (size_t)NV*NO*NV*NV);
    std::vector<real_t> ERIov = readbin(DATA_DIR "/eriov.bin", (size_t)NO*NV*NO*NV);
    std::vector<real_t> sIP   = readbin(DATA_DIR "/rx.bin",    (size_t)NMo*NO*NO*NV);
    std::vector<real_t> sEA   = readbin(DATA_DIR "/ry.bin",    (size_t)NMv*NO*NV*NV);
    auto fov  = [&](int k,int c){ return Fov[(size_t)k*NV+c]; };
    auto wooov= [&](int k,int l,int i,int d){ return Wooov[(((size_t)k*NO+l)*NO+i)*NV+d]; };
    auto wvovv= [&](int a,int l,int c,int d){ return Wvovv[(((size_t)a*NO+l)*NV+c)*NV+d]; };
    auto eriov= [&](int k,int b,int i,int d){ return ERIov[(((size_t)k*NV+b)*NO+i)*NV+d]; };
    auto siP  = [&](int m,int i,int j,int a)->real_t&{ return sIP[(((size_t)m*NO+i)*NO+j)*NV+a]; };
    auto seA  = [&](int e,int i,int a,int b)->real_t&{ return sEA[(((size_t)e*NO+i)*NV+a)*NV+b]; };
    std::vector<real_t> proj_route_ph((size_t)NO*NV*NO*NV, 0.0);
    std::vector<real_t> proj_route_mc((size_t)NO*NV*NO*NV, 0.0);
    #include "steom_gphph_projection.inc"
    std::vector<real_t> ref_ph2 = readbin(DATA_DIR "/ref_ph.bin", proj_route_ph.size());
    std::vector<real_t> ref_mc2 = readbin(DATA_DIR "/ref_mc.bin", proj_route_mc.size());
    real_t dph = 0.0, dmc = 0.0;
    for (size_t x = 0; x < proj_route_ph.size(); ++x) dph = std::max(dph, std::fabs(proj_route_ph[x]-ref_ph2[x]));
    for (size_t x = 0; x < proj_route_mc.size(); ++x) dmc = std::max(dmc, std::fabs(proj_route_mc[x]-ref_mc2[x]));
    printf("[inc test] route_ph max|d| = %.3e   route_mc max|d| = %.3e  (expect <=1e-12)\n", dph, dmc);
    return (dph < 1e-10 && dmc < 1e-10) ? 0 : 2;
}
"""


def main():
    os.makedirs(WORK, exist_ok=True)
    data = get_active_data(xyz="xyz/H2O.xyz", basis="sto-3g", ncore=1)
    nocc, nvir = data["nocc"], data["nvir"]
    dets, index, Hbar = build_sector(data, data["nelec"])
    E_N = Hbar[index[hf_det(data)], index[hf_det(data)]]
    sp_det = IPD.extract_sip(solve_ip(data, E_N), data)
    se_det = EA.extract_spatial_amp(solve_ea(data), data)
    rx = np.stack([sp_det[m] for m in range(nocc)], 0); ry = se_det
    d = CW.load("xyz/H2O.xyz", "sto-3g", 1); bar = d["bar"]
    arrays = {'Fov': bar["Fov"], 'Wooov': bar["Wooov"], 'Wvovv': bar["Wvovv"],
              'eri_ovov': bar["eri_ovov"], 'rx': rx, 'ry': ry}

    lin = GEN.linear_struct()
    print("deriving quadratic (slow)...")
    quad = GEN.quad_struct()
    allt = lin + quad
    Ms = GEN.expand(allt, 0); Mc = GEN.expand(allt, 1)
    route = GEN.subtract(Mc, Ms)
    nodelta = lambda lst: [(c, ops) for c, ops in lst
                           if not any(a.startswith('delta') for a, _ in ops)]
    ref_ph = GEN.evaluate(nodelta(route), arrays, nocc, nvir)
    ref_mc = GEN.evaluate(nodelta(Mc), arrays, nocc, nvir)

    open(f"{WORK}/dims.txt", "w").write(f"{nocc} {nvir}\n")
    np.ascontiguousarray(bar["Fov"]).tofile(f"{WORK}/fov.bin")
    np.ascontiguousarray(bar["Wooov"]).tofile(f"{WORK}/wooov.bin")
    np.ascontiguousarray(bar["Wvovv"]).tofile(f"{WORK}/wvovv.bin")
    np.ascontiguousarray(bar["eri_ovov"]).tofile(f"{WORK}/eriov.bin")
    np.ascontiguousarray(rx).tofile(f"{WORK}/rx.bin")
    np.ascontiguousarray(ry).tofile(f"{WORK}/ry.bin")
    np.ascontiguousarray(ref_ph).tofile(f"{WORK}/ref_ph.bin")
    np.ascontiguousarray(ref_mc).tofile(f"{WORK}/ref_mc.bin")
    open(f"{WORK}/harness.cpp", "w").write(HARNESS)

    inc = os.path.abspath("include")
    cmd = ["g++", "-O2", "-fopenmp", f"-DDATA_DIR=\"{WORK}\"", f"-I{inc}",
           f"{WORK}/harness.cpp", "-o", f"{WORK}/harness"]
    print("compiling:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:4000]); sys.exit(1)
    os.environ["GANSU_STEOM_PROJ_VALIDATE"] = "1"   # also exercise the naive self-check
    r = subprocess.run([f"{WORK}/harness"], capture_output=True, text=True,
                       env=dict(os.environ))
    print(r.stdout, r.stderr)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
