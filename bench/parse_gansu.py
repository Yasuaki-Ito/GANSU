#!/usr/bin/env python3
"""Parse GANSU benchmark logs (bench/logs/*.log) into a tidy CSV.

Usage (from anywhere):
    python3 bench/parse_gansu.py [bench/logs] > bench/gansu_bench.csv

Filename convention written by run_gansu.sh:  <method>_g<N>_<molecule>.log
Columns: method,num_gpus,molecule,natoms,nbf,status,wall_s,<phase times in s>
Phases extracted (NA if absent for that method):
  ri_Bbuild  scf  dlpno_mp2  dlpno_ccsd_t2  cis  ip_eom  ea_eom  steom  postproc
"""
import sys, os, re, glob, csv

LOGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "logs")

def grab(rx, text, group=1, last=True, cast=float):
    ms = re.findall(rx, text)
    if not ms:
        return None
    v = ms[-1] if last else ms[0]
    if isinstance(v, tuple):
        v = v[group-1]
    try:
        return cast(v)
    except Exception:
        return None

def status_of(text, rc_hint=None):
    if re.search(r"out of memory|cudaMalloc failed|bad_alloc", text, re.I):
        return "OOM"
    if re.search(r"\bSegmentation fault\b|core dumped", text, re.I):
        return "CRASH"
    # a completed run prints the final Calculation Summary
    if "Total Energy:" in text or "STEOM excited-state energies" in text or "Computing time:" in text:
        return "OK"
    return "INCOMPLETE"

rows = []
for log in sorted(glob.glob(os.path.join(LOGDIR, "*.log"))):
    base = os.path.basename(log)[:-4]
    m = re.match(r"(?P<method>.+?)_g(?P<ng>\d+)_(?P<mol>.+)$", base)
    if not m:
        continue
    with open(log, errors="replace") as fh:
        t = fh.read()

    natoms = grab(r"Number of atoms:\s*(\d+)", t, cast=int)
    nbf    = grab(r"Number of basis functions:\s*(\d+)", t, cast=int)

    # phase timings → seconds
    ri_b   = grab(r"precompute_eri_matrix after ([0-9.]+) ms", t)            # ms
    ri_b   = ri_b/1000 if ri_b is not None else None
    scf_ms = grab(r"Computing time:\s*([0-9]+)\s*\[ms\]", t, cast=float)
    scf    = scf_ms/1000 if scf_ms is not None else None
    mp2    = grab(r"\[DLPNO-MP2-PROF\]\s*total=([0-9.]+)s", t)               # s
    if mp2 is None:
        mp2 = grab(r"solve_dlpno_lmp2 = ([0-9.]+) s", t)
    ccsd   = grab(r"iterate_dlpno_ccsd_t2 = ([0-9.]+) s", t)                # s
    # Timing Summary block: "<fn>: <val> microseconds total" (values are ms)
    def summ(fn):
        v = grab(rf"{fn}:\s*([0-9.]+)\s*microseconds", t)
        return v/1000 if v is not None else None
    cis   = summ("compute_cis_ri_impl") or summ("compute_cis_nto")
    ip    = summ("compute_ip_eom_ccsd_impl")
    ea    = summ("compute_ea_eom_ccsd_impl")
    steom = summ("compute_steom_ccsd_impl")
    pp    = summ("post_process_after_scf")

    # SCF convergence path vs per-iteration cost. The total SCF wall depends on the
    # iteration count (guess-dependent); the per-iteration Fock build is the
    # iteration-independent GPU metric. Timing Summary: "compute_fock_matrix: <ms>
    # microseconds total, called <N> times".
    scf_iters = grab(r"Number of iterations:\s*(\d+)", t, cast=int)
    fk = re.search(r"compute_fock_matrix:\s*([0-9.]+)\s*microseconds total, called\s*(\d+)", t)
    fock_per_iter = (float(fk.group(1))/int(fk.group(2))) if fk and int(fk.group(2)) else None

    rows.append(dict(
        method=m["method"], num_gpus=int(m["ng"]), molecule=m["mol"],
        natoms=natoms, nbf=nbf, status=status_of(t),
        scf_iters=scf_iters, fock_per_iter_ms=fock_per_iter,
        ri_Bbuild=ri_b, scf=scf, dlpno_mp2=mp2, dlpno_ccsd_t2=ccsd,
        cis=cis, ip_eom=ip, ea_eom=ea, steom=steom, postproc=pp,
    ))

cols = ["method","num_gpus","molecule","natoms","nbf","status","wall_s",
        "scf_iters","fock_per_iter_ms","ri_Bbuild","scf","dlpno_mp2","dlpno_ccsd_t2",
        "cis","ip_eom","ea_eom","steom","postproc"]

# pull the authoritative wall_s from results_*.tsv (actual /SECONDS wall)
wall = {}
for r in glob.glob(os.path.join(os.path.dirname(LOGDIR), "results_*.tsv")):
    mm = re.search(r"results_(.+?)_g(\d+)\.tsv$", os.path.basename(r))
    if not mm: continue
    with open(r) as fh:
        next(fh, None)
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                wall[(mm[1], int(mm[2]), p[0])] = p[3]

w = csv.DictWriter(sys.stdout, fieldnames=cols)
w.writeheader()
for r in sorted(rows, key=lambda x: (x["method"], x["num_gpus"], x["natoms"] or 0)):
    r["wall_s"] = wall.get((r["method"], r["num_gpus"], r["molecule"]), "")
    for k, v in list(r.items()):
        if isinstance(v, float):
            r[k] = f"{v:.2f}"
    w.writerow({c: r.get(c, "") for c in cols})
