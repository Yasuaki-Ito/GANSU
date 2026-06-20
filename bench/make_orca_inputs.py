#!/usr/bin/env python3
"""Generate ORCA input files for the same size series + methods as the GANSU run,
for a wall-time / max-size comparison (GANSU GPU vs ORCA CPU).

Usage:
    python3 bench/make_orca_inputs.py [NPROCS] [MAXCORE_MB]
        NPROCS    : MPI ranks for ORCA (%pal). Default 64.
        MAXCORE_MB: per-rank memory in MB (%maxcore). Default 3000.

Writes bench/orca/inp/<method>__<molecule>.inp using "* xyzfile 0 1 <abs xyz>".
Method header lines (cc-pVDZ everywhere; frozen core = ORCA default = matches GANSU):
  rihf        : ! RHF cc-pVDZ RIJCOSX             (ORCA production fast HF)
  rimp2       : ! RI-MP2 cc-pVDZ cc-pVDZ/C        (cc-pVDZ/C ≈ cc-pvdz-rifit aux)
  dlpno_ccsd  : ! DLPNO-CCSD cc-pVDZ cc-pVDZ/C
  dlpno_ccsd_t: ! DLPNO-CCSD(T) cc-pVDZ cc-pVDZ/C   (semi-canonical (T0), ORCA default)
  dlpno_steom : ! STEOM-DLPNO-CCSD cc-pVDZ cc-pVDZ/C  + %mdci nroots 5
NOTE: algorithms are not byte-identical to GANSU (e.g. ORCA HF=RIJCOSX vs GANSU full
RI-JK); this measures *production-setting* wall time per package. Adjust headers to taste.
"""
import sys, os

NPROCS  = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MAXCORE = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)                       # repo root
OUT     = os.path.join(HERE, "orca", "inp"); os.makedirs(OUT, exist_ok=True)

HEADERS = {
    "rihf":        "! RHF cc-pVDZ RIJCOSX TightSCF",
    "rimp2":       "! RI-MP2 cc-pVDZ cc-pVDZ/C TightSCF",
    "dlpno_ccsd":  "! DLPNO-CCSD cc-pVDZ cc-pVDZ/C TightSCF",
    "dlpno_ccsd_t": "! DLPNO-CCSD(T) cc-pVDZ cc-pVDZ/C TightSCF",
    "dlpno_steom": "! STEOM-DLPNO-CCSD cc-pVDZ cc-pVDZ/C TightSCF",
}
EXTRA = {  # method-specific blocks
    "dlpno_steom": "%mdci\n  nroots 5\nend\n",
}

count = 0
SERIES = os.environ.get("SERIES", os.path.join(HERE, "series.tsv"))  # override series file
with open(SERIES) as fh:
    for line in fh:
        if line.startswith("#") or not line.strip():
            continue
        name, xyzrel, nat, tags = (line.rstrip("\n").split("\t") + ["", "", "", ""])[:4]
        xyz_abs = os.path.normpath(os.path.join(ROOT, "build", xyzrel))  # series paths are build-relative
        if not os.path.exists(xyz_abs):
            xyz_abs = os.path.normpath(os.path.join(ROOT, xyzrel.replace("../", "", 1)))
        if not os.path.exists(xyz_abs):
            print(f"  skip {name}: xyz not found ({xyzrel})"); continue
        # Read geometry and re-emit it SPACE-separated inline. GANSU's xyz files are
        # TAB-separated, which ORCA's coordinate scanner rejects ("Expected the y
        # coordinate"); split() on any whitespace + a clean "sym x y z" rewrite fixes
        # it (and makes each .inp self-contained — no external xyz path dependency).
        with open(xyz_abs, errors="replace") as xf:
            body = xf.read().splitlines()
        geom = []
        for ln in body[2:]:                       # skip count + comment
            p = ln.split()                        # any whitespace incl. tabs
            if len(p) >= 4:
                geom.append(f"{p[0]:<3s} {p[1]:>18s} {p[2]:>18s} {p[3]:>18s}")
        if not geom:
            print(f"  skip {name}: no atoms parsed from {xyz_abs}"); continue
        for method in HEADERS:
            if f",{method}," not in f",{tags},":
                continue
            inp = os.path.join(OUT, f"{method}__{name}.inp")
            with open(inp, "w") as o:
                o.write(HEADERS[method] + "\n")
                o.write(f"%pal nprocs {NPROCS} end\n")
                o.write(f"%maxcore {MAXCORE}\n")
                o.write(EXTRA.get(method, ""))
                o.write("* xyz 0 1\n")
                o.write("\n".join(geom) + "\n")
                o.write("*\n")
            count += 1
print(f"wrote {count} ORCA inputs to {OUT}  (nprocs={NPROCS}, maxcore={MAXCORE} MB)")
print("run e.g.:  cd bench/orca && for f in inp/dlpno_ccsd__*.inp; do "
      "$ORCA/orca $f > ${f%.inp}.out; done")
