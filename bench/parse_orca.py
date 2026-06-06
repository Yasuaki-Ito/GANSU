#!/usr/bin/env python3
"""Parse ORCA .out files (bench/orca/inp/*.out) → CSV, and (optionally) merge with
the GANSU CSV into a GANSU-vs-ORCA comparison table.

    python3 bench/parse_orca.py                         > bench/orca_bench.csv
    python3 bench/parse_orca.py bench/gansu_bench.csv   > bench/compare.csv   # merged

ORCA wall = parsed from "TOTAL RUN TIME:" (authoritative, package-internal).
Comparison columns: per (method,molecule): natoms, gansu_wall_s(@num_gpus), orca_wall_s,
speedup = orca/gansu, plus status of each.
"""
import sys, os, re, glob, csv

HERE = os.path.dirname(os.path.abspath(__file__))
INP  = os.path.join(HERE, "orca", "inp")

def orca_seconds(text):
    m = re.search(r"TOTAL RUN TIME:\s*(\d+)\s*days?\s*(\d+)\s*hours?\s*(\d+)\s*minutes?\s*"
                  r"(\d+)\s*seconds?\s*(\d+)\s*msec", text)
    if not m:
        return None
    d, h, mi, s, ms = map(int, m.groups())
    return d*86400 + h*3600 + mi*60 + s + ms/1000.0

def orca_status(text):
    if re.search(r"not enough memory|insufficient memory|out of memory|aborting the run|finished by error", text, re.I):
        return "OOM_or_ERR"
    if "ORCA TERMINATED NORMALLY" in text:
        return "OK"
    return "INCOMPLETE"

orca = {}   # (method,molecule) -> dict
for out in sorted(glob.glob(os.path.join(INP, "*.out"))):
    b = os.path.basename(out)[:-4]
    m = re.match(r"(?P<method>.+?)__(?P<mol>.+)$", b)
    if not m:
        continue
    with open(out, errors="replace") as fh:
        t = fh.read()
    nat = re.search(r"Number of atoms\s*\.+\s*(\d+)", t)
    orca[(m["method"], m["mol"])] = dict(
        method=m["method"], molecule=m["mol"],
        natoms=int(nat.group(1)) if nat else "",
        orca_status=orca_status(t),
        orca_wall_s=("%.1f" % orca_seconds(t)) if orca_seconds(t) is not None else "",
    )

if len(sys.argv) > 1:  # merge with GANSU CSV → comparison
    gan = {}
    with open(sys.argv[1]) as fh:
        for r in csv.DictReader(fh):
            gan[(r["method"], r["molecule"])] = r
    keys = sorted(set(orca) | set(gan), key=lambda k: (k[0], int(gan.get(k, {}).get("natoms") or
                  orca.get(k, {}).get("natoms") or 0)))
    cols = ["method","molecule","natoms","num_gpus","gansu_status","gansu_wall_s",
            "orca_status","orca_wall_s","speedup_orca_over_gansu"]
    w = csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()
    for k in keys:
        g, o = gan.get(k, {}), orca.get(k, {})
        gw = g.get("wall_s") or ""
        ow = o.get("orca_wall_s") or ""
        sp = ""
        try:
            if gw and ow and float(gw) > 0:
                sp = "%.1f" % (float(ow)/float(gw))
        except ValueError:
            pass
        w.writerow(dict(method=k[0], molecule=k[1],
            natoms=g.get("natoms") or o.get("natoms") or "",
            num_gpus=g.get("num_gpus",""), gansu_status=g.get("status",""), gansu_wall_s=gw,
            orca_status=o.get("orca_status",""), orca_wall_s=ow, speedup_orca_over_gansu=sp))
else:  # ORCA-only CSV
    cols = ["method","molecule","natoms","orca_status","orca_wall_s"]
    w = csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()
    for k in sorted(orca, key=lambda k: (k[0], orca[k]["natoms"] or 0)):
        w.writerow({c: orca[k].get(c, "") for c in cols})
