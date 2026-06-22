#!/usr/bin/env python3
"""GANSU (GPU) vs ORCA (CPU) wall-time benchmark -- single Python driver.

Replaces the run_gansu.sh / run_orca.sh / make_orca_inputs.py / parse_*.py shell
pipeline with one script. Each (molecule, method) runs as an ISOLATED subprocess
(fresh GPU memory, OOM/crash isolation, full CLI knob coverage), is wall-timed and
parsed, and the results go to a tidy CSV + a printed table.

Run from the build/ directory (where ./gansu lives):

    # GANSU only (timing/scaling):
    python3 ../bench/bench.py --family aldehyde --methods dlpno_ccsd,dlpno_steom --num_gpus 4

    # GANSU vs ORCA:
    python3 ../bench/bench.py --family aldehyde --methods dlpno_ccsd \
        --num_gpus 4 --orca /opt/orca6/orca --orca-nprocs 64

    # arbitrary molecule list / custom series file:
    python3 ../bench/bench.py --series ../bench/series_alkylbenzene.tsv --methods dlpno_ccsd
    python3 ../bench/bench.py --xyz ../xyz/bench_steom/pentanal.xyz ../xyz/Naphthalene.xyz ...

Molecules are processed in series order (increasing size); by default the sweep
stops at the first OOM/timeout/crash per package+method (--no-stop to run all).
"""
import argparse, csv, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

# method -> (gansu post-HF/eri args, orca '!' keyword, needs cc-pVDZ/C aux, extra block)
METHODS = {
    "rihf":         (["-m", "RHF"],                                          "RHF RIJCOSX",      False, ""),
    "rimp2":        (["--post_hf_method", "MP2"],                            "RI-MP2",           True,  ""),
    "dlpno_ccsd":   (["--post_hf_method", "dlpno_ccsd",   "--frozen_core", "auto"], "DLPNO-CCSD",    True, ""),
    "dlpno_ccsd_t": (["--post_hf_method", "dlpno_ccsd_t", "--frozen_core", "auto"], "DLPNO-CCSD(T)", True, ""),
    "dlpno_steom":  (["--post_hf_method", "dlpno_steom_ccsd", "--frozen_core", "auto"],
                     "STEOM-DLPNO-CCSD", True, "%mdci\n  nroots {nroots}\nend\n"),
}

# ----------------------------------------------------------------- parsing
def gansu_status(out):
    if re.search(r"out of memory|cudaMalloc failed|bad_alloc|CUDA error", out, re.I): return "OOM"
    if re.search(r"Segmentation fault|core dumped|Aborted", out, re.I): return "CRASH"
    if ("Total Energy:" in out or "STEOM excited-state energies" in out
            or "Computing time:" in out): return "OK"
    return "INCOMPLETE"

def gansu_energy(out, method):
    if method == "dlpno_steom":                       # report lowest excitation (eV)
        m = re.search(r"STEOM excited-state energies:.*?\n(?:.*\n)*?\s*\d+\s+[-\d.]+\s+([-\d.]+)", out)
        return float(m.group(1)) if m else None
    m = re.search(r"Post-HF energy correction:\s*([-\d.]+)", out)
    if m: return float(m.group(1))
    m = re.search(r"Total Energy:\s*([-\d.]+)", out)
    return float(m.group(1)) if m else None

def orca_status(out):
    if re.search(r"not enough memory|insufficient memory|out of memory|finished by error|aborting the run", out, re.I):
        return "OOM_or_ERR"
    return "OK" if "ORCA TERMINATED NORMALLY" in out else "INCOMPLETE"

def orca_seconds(out):
    m = re.search(r"TOTAL RUN TIME:\s*(\d+)\s*days?\s*(\d+)\s*hours?\s*(\d+)\s*minutes?\s*(\d+)\s*seconds?\s*(\d+)\s*msec", out)
    if not m: return None
    d, h, mi, s, ms = map(int, m.groups()); return d*86400 + h*3600 + mi*60 + s + ms/1000.0

def orca_corr(out):
    """ORCA correlation energy (CCSD + (T) if present), comparable to GANSU's
    post-HF correction. None for non-correlated methods / unparsed output."""
    m = re.search(r"E\(CORR\)[^\n]*?(-\d+\.\d+)", out)
    if not m: return None
    ec = float(m.group(1))
    t = re.search(r"Triples Correction \(T\)[^\n]*?(-\d+\.\d+)", out)
    return ec + float(t.group(1)) if t else ec

CM1_PER_EV = 8065.54429  # cm^-1 per eV (NIST)

def orca_steom_exc(out):
    """Lowest STEOM-CCSD singlet excitation energy (eV) from an ORCA output.

    ORCA's final STEOM excited-state spectrum is the LAST
    'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS' block (earlier
    electric-dipole blocks are the CIS/CIS(D) active-space guess). Rows are

        0-1A  ->  1-1A    4.187232   33772.3   296.1   0.000000001  ...
        <ground> -> <state>  E(eV)     E(cm-1)  lambda(nm)  fosc ...

    so the first float after the '-> <label>' is the excitation in eV. Take the
    smallest positive one in the last block. Returns eV, or None."""
    heads = [m.start() for m in re.finditer(
        r"ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS", out)]
    if not heads:
        return None
    seg = out[heads[-1]:]                       # last electric-dipole block = final STEOM spectrum
    nxt = seg.find("ABSORPTION SPECTRUM VIA TRANSITION VELOCITY")
    if nxt != -1:                               # bound to this block (its velocity twin repeats the rows)
        seg = seg[:nxt]
    evs = [float(m.group(1)) for m in
           re.finditer(r"->\s+\S+\s+(-?\d+\.\d+)\s+\d+\.\d+\s+\d+\.\d+", seg)]
    evs = [e for e in evs if e > 0.05]
    return min(evs) if evs else None

# ----------------------------------------------------------------- runners
def run_cmd(cmd, timeout_s, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s,
                           env=env, errors="replace")
        return time.time() - t0, p.stdout + p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        if isinstance(out, bytes): out = out.decode("utf-8", "replace")
        return time.time() - t0, out, True

# GANSU dlpno preset -> ORCA PNO keyword (ORCA has no VeryTightPNO preset; TightPNO is its tightest).
ORCA_PNO = {"loose": "LoosePNO", "normal": "NormalPNO", "tight": "TightPNO", "very_tight": "TightPNO"}

def run_gansu(xyz, method, a):
    extra = ["--n_excited_states", str(a.nroots)] if method == "dlpno_steom" else []
    if method.startswith("dlpno"):                       # STEOM excitations need tight+ on large mols
        extra += ["--dlpno_preset", a.dlpno_preset]
    cmd = [a.gansu, "-x", xyz, "-g", a.basis, "--use_spherical", "1",
           "--initial_guess", "sad", "--eri_method", "ri", "-ag", a.aux,
           "--num_gpus", str(a.num_gpus)] + METHODS[method][0] + extra
    env = dict(os.environ); env.setdefault("OPENBLAS_NUM_THREADS", str(a.orca_nprocs))
    wall, out, to = run_cmd(cmd, a.timeout*60, env)
    st = "TIMEOUT" if to else gansu_status(out)
    return dict(wall=wall, status=st, energy=gansu_energy(out, method))

def orca_nprocs_for(nat, a, method=None):
    # ORCA errors when MPI ranks exceed what a small molecule can parallelise.
    # Cap to a size-appropriate value (large molecules still get the full count).
    # STEOM-DLPNO additionally crashes inside orca_mdci_mpi at high rank counts on
    # small systems (MDCI parallel efficiency plateaus ~8-16), so cap it harder.
    if nat <= 0 or not a.orca_cap: return a.orca_nprocs
    cap = min(a.orca_nprocs, 2*nat)
    if method == "dlpno_steom":
        cap = min(cap, a.orca_steom_nprocs)
    return max(2, cap)

def orca_inp(xyz_abs, method, a, nprocs):
    kw, need_aux, extra = METHODS[method][1], METHODS[method][2], METHODS[method][3]
    pno = (" " + ORCA_PNO.get(a.dlpno_preset, "NormalPNO")) if method.startswith("dlpno") else ""
    head = f"! {kw} {a.basis}" + (f" {a.basis}/C" if need_aux else "") + pno + " TightSCF"
    body = f"%pal nprocs {nprocs} end\n%maxcore {a.orca_maxcore}\n"
    body += extra.format(nroots=a.nroots)
    body += f"* xyzfile {a.charge} {a.mult} {xyz_abs}\n"
    return head + "\n" + body

def run_orca(xyz_abs, method, a, inpdir, nat):
    np_eff = orca_nprocs_for(nat, a, method)
    inp = os.path.join(inpdir, f"{method}__{os.path.splitext(os.path.basename(xyz_abs))[0]}.inp")
    with open(inp, "w") as f: f.write(orca_inp(xyz_abs, method, a, np_eff))
    wall, out, to = run_cmd([a.orca, inp], a.timeout*60)
    with open(inp[:-4] + ".out", "w") as f: f.write(out)
    st = "TIMEOUT" if to else orca_status(out)
    return dict(wall=wall, status=st, orca_rt=orca_seconds(out), nprocs=np_eff,
                ecorr=orca_corr(out),
                steom_eV=(orca_steom_exc(out) if method == "dlpno_steom" else None))

# ----------------------------------------------------------------- molecules
def load_series(path):
    mols = []
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or not ln.strip(): continue
            c = ln.rstrip("\n").split("\t")
            mols.append((c[0], c[1], int(c[2]) if len(c) > 2 and c[2].isdigit() else 0,
                         c[3] if len(c) > 3 else ""))
    return mols

def main():
    ap = argparse.ArgumentParser(description="GANSU vs ORCA wall-time benchmark")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--series", help="series TSV (name<TAB>xyz<TAB>natoms<TAB>tags)")
    g.add_argument("--family", help="local-chromophore family -> bench/series_<family>.tsv")
    g.add_argument("--xyz", nargs="+", help="explicit xyz files")
    ap.add_argument("--reparse", metavar="CSV",
                    help="no runs: merge ORCA energies (from existing .out) into this CSV "
                         "(which has gansu_E) -> *_E.csv with orca_ecorr + dE_mEh")
    ap.add_argument("--methods", default="dlpno_ccsd",
                    help="comma list: " + ",".join(METHODS))
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--aux", default="../auxiliary_basis/cc-pvdz-rifit.gbs")
    ap.add_argument("--gansu", default="./gansu")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--orca", default=None, help="path to orca binary (omit = GANSU only)")
    ap.add_argument("--orca-nprocs", type=int, default=64)
    ap.add_argument("--orca-steom-nprocs", type=int, default=16,
                    help="extra ORCA rank cap for STEOM-DLPNO (orca_mdci_mpi crashes at "
                         "high ranks on small systems); default 16")
    ap.add_argument("--dlpno-preset", default="normal",
                    choices=["loose", "normal", "tight", "very_tight"],
                    help="DLPNO truncation preset for GANSU dlpno_* methods. STEOM excitations need "
                         "tight+ on large molecules (normal under-truncates the PNO space the W^eff "
                         "dressing is built from); ORCA gets the matching *PNO keyword. Default normal.")
    ap.add_argument("--no-orca-cap", dest="orca_cap", action="store_false",
                    help="do NOT cap ORCA nprocs to ~2*natoms for small molecules")
    ap.set_defaults(orca_cap=True)
    ap.add_argument("--orca-maxcore", type=int, default=3000)
    ap.add_argument("--nroots", type=int, default=5)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=120, help="per-job cap (minutes)")
    ap.add_argument("--no-stop", action="store_true", help="do not stop at first failure")
    ap.add_argument("--out", default=None, help="output CSV path")
    a = ap.parse_args()

    # ---- reparse mode: merge ORCA energies from saved .out into an existing CSV ----
    if a.reparse:
        inpdir = os.path.join(HERE, "orca", "inp")
        with open(a.reparse) as f: rd = list(csv.DictReader(f))
        out = os.path.splitext(a.reparse)[0] + "_E.csv"
        fld = list(rd[0].keys())
        for c in ("orca_ecorr", "orca_exc_eV", "dE_mEh", "dE_eV"):
            if c not in fld: fld.append(c)
        for r in rd:
            op = os.path.join(inpdir, f"{r['method']}__{r['molecule']}.out")
            txt = open(op, errors="replace").read() if os.path.exists(op) else ""
            if r["method"] == "dlpno_steom":
                ev = orca_steom_exc(txt) if txt else None
                r["orca_exc_eV"] = f"{ev:.4f}" if ev is not None else ""
                try:
                    r["dE_eV"] = f"{float(r['gansu_E'])-ev:+.4f}" if ev is not None and r.get('gansu_E') else ""
                except Exception:
                    r["dE_eV"] = ""
                r["orca_ecorr"] = r.get("orca_ecorr", ""); r["dE_mEh"] = r.get("dE_mEh", "")
            else:
                ec = orca_corr(txt) if txt else None
                r["orca_ecorr"] = f"{ec:.6f}" if ec is not None else ""
                try:
                    r["dE_mEh"] = f"{(float(r['gansu_E'])-ec)*1000:+.2f}" if ec is not None and r.get('gansu_E') else ""
                except Exception:
                    r["dE_mEh"] = ""
                r["orca_exc_eV"] = r.get("orca_exc_eV", ""); r["dE_eV"] = r.get("dE_eV", "")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fld); w.writeheader()
            for r in rd: w.writerow(r)
        print(f"wrote {out}  ({len(rd)} rows; ORCA energies from {inpdir})")
        return

    if not (a.series or a.family or a.xyz):
        ap.error("need one of --series / --family / --xyz (or --reparse CSV)")

    if a.family:  a.series = os.path.join(HERE, f"series_{a.family}.tsv")
    if a.series:
        mols = load_series(a.series); tag = os.path.splitext(os.path.basename(a.series))[0]
    else:
        mols = []
        for x in a.xyz:
            try: nat = int(open(x).readline().split()[0])
            except Exception: nat = 0
            mols.append((os.path.splitext(os.path.basename(x))[0], x, nat, ",".join(METHODS)))
        tag = "custom"
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS: sys.exit(f"unknown method '{m}' (have {list(METHODS)})")

    inpdir = os.path.join(HERE, "orca", "inp"); os.makedirs(inpdir, exist_ok=True)
    out_csv = a.out or os.path.join(HERE, f"bench_{tag}.csv")
    cols = ["method","molecule","natoms","gansu_s","gansu_status","gansu_E"]
    if a.orca: cols += ["orca_s","orca_rt","orca_status","orca_np","orca_ecorr",
                        "orca_exc_eV","dE_mEh","dE_eV","speedup"]
    rows = []
    def flush_csv():
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"# molecules={len(mols)}  methods={methods}  num_gpus={a.num_gpus}"
          f"  orca={'yes' if a.orca else 'no'}  timeout={a.timeout}min")
    print(f"# CSV (updated after every molecule, Ctrl-C safe): {out_csv}\n")
    hdr = f"{'method':12s} {'molecule':18s} {'nat':>4s} {'gansu_s':>9s} {'g_stat':>8s} {'gansu_E':>11s}"
    if a.orca: hdr += f" {'orca_s':>9s} {'o_stat':>10s} {'orca_E':>11s} {'dE(eV/mEh)':>11s} {'orca/gansu':>10s}"
    print(hdr); print("-"*len(hdr))

    for method in methods:
        # Stop the sweep only on a failure that occurs AFTER at least one success
        # (a genuine size ceiling) -- an early failure on the smallest molecule
        # (e.g. ORCA refusing too many ranks) must NOT abort the whole series.
        g_dead = o_dead = False; g_ok = o_ok = 0
        for name, xyz, nat, tags in mols:
            if tags and method not in tags.split(","): continue
            xyz_abs = xyz if os.path.isabs(xyz) else os.path.normpath(os.path.join(os.getcwd(), xyz))
            row = dict(method=method, molecule=name, natoms=nat)
            if not g_dead:
                r = run_gansu(xyz, method, a)
                row.update(gansu_s=f"{r['wall']:.1f}", gansu_status=r["status"], gansu_E=r["energy"])
                if r["status"] == "OK": g_ok += 1
                elif not a.no_stop and g_ok >= 1: g_dead = True
            else:
                row.update(gansu_s="", gansu_status="skip", gansu_E=None)
            gE = row.get('gansu_E')
            gE_disp = (f"{gE:.4f}" if isinstance(gE, float) else "")
            line = f"{method:12s} {name:18s} {nat:4d} {row.get('gansu_s',''):>9s} {row['gansu_status']:>8s} {gE_disp:>11s}"
            if a.orca:
                if not o_dead:
                    r = run_orca(xyz_abs, method, a, inpdir, nat)
                    sp = (r["orca_rt"] / float(row["gansu_s"])) if (r.get("orca_rt") and row.get("gansu_s")) else None
                    # STEOM compares excitation energies (eV); correlated methods compare corr (mEh).
                    # A crashed/incomplete ORCA run leaves only the CIS guess block, so its parsed
                    # energy is garbage -- only report orca_E / dE when ORCA terminated normally.
                    ok = (r["status"] == "OK")
                    gE = row.get("gansu_E")
                    de_meh = ((gE - r["ecorr"]) * 1000.0
                              if (ok and method != "dlpno_steom" and gE is not None and r.get("ecorr") is not None) else None)
                    de_ev = ((gE - r["steom_eV"])
                             if (ok and method == "dlpno_steom" and gE is not None and r.get("steom_eV") is not None) else None)
                    row.update(orca_s=f"{r['wall']:.1f}", orca_rt=(f"{r['orca_rt']:.1f}" if r["orca_rt"] else ""),
                               orca_status=r["status"], orca_np=r["nprocs"],
                               orca_ecorr=(f"{r['ecorr']:.6f}" if (ok and r["ecorr"] is not None) else ""),
                               orca_exc_eV=(f"{r['steom_eV']:.4f}" if (ok and r.get("steom_eV") is not None) else ""),
                               dE_mEh=(f"{de_meh:+.2f}" if de_meh is not None else ""),
                               dE_eV=(f"{de_ev:+.4f}" if de_ev is not None else ""),
                               speedup=(f"{sp:.2f}" if sp else ""))
                    if r["status"] == "OK": o_ok += 1
                    elif not a.no_stop and o_ok >= 1: o_dead = True
                else:
                    row.update(orca_s="", orca_rt="", orca_status="skip", speedup="")
                de_disp = row.get('dE_eV','') or row.get('dE_mEh','')
                oE_disp = row.get('orca_exc_eV','') or row.get('orca_ecorr','')
                line += f" {row.get('orca_s',''):>9s} {row['orca_status']:>10s} {oE_disp:>11s} {de_disp:>11s} {row.get('speedup',''):>10s}"
            print(line, flush=True)
            rows.append(row)
            flush_csv()        # persist after every molecule -> Ctrl-C never loses results

    print(f"\nwrote {out_csv}  ({len(rows)} rows)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] partial results already saved to the CSV "
              "(updated after each molecule).", file=sys.stderr)
        sys.exit(130)
