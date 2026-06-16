#!/usr/bin/env python3
"""STEOM-CCSD accuracy benchmark: GANSU excitation energies vs ORCA, root by root.

This is the *accuracy* counterpart to parse_gansu.py / parse_orca.py (which compare
wall time). It pairs the STEOM excited-state energies (eV) printed by each package
on the same molecule and tabulates the differences.

    # from anywhere (defaults: GANSU logs in bench/logs, ORCA outs in bench/orca/inp)
    python3 bench/compare_steom_roots.py > bench/steom_roots.csv
    # custom dirs:
    python3 bench/compare_steom_roots.py <gansu_logdir> <orca_outdir> > out.csv

Inputs
  GANSU : bench/logs/dlpno_steom_g<N>_<Molecule>.log   (written by run_gansu.sh)
          root table printed by eri_stored_steom_ccsd.cu:
              STEOM excited-state energies:
               k   omega (Ha)        omega (eV)
               0   0.12345678   3.3594
               ...
  ORCA  : bench/orca/inp/dlpno_steom__<Molecule>.out
          FINAL excited states under the "STEOM-CCSD RESULTS (SINGLETS)" header,
          whose IROOT lines uniquely carry an <S**2> tag:
              IROOT=  1:  0.162757 au   4.429 eV   35721.1 cm**-1 <S**2> = 0.000000
          (The earlier IROOT blocks in the file are the IP-EOM and EA-EOM active-state
          solves — 8-17 eV ionizations etc. — and must NOT be picked up. They have no
          <S**2> tag and sit before the RESULTS header.)

Output (CSV, one row per paired state)
  molecule, natoms, state, gansu_k, gansu_eV, orca_iroot, orca_eV, diff_eV, abs_diff_meV

Pairing
  Roots are sorted ascending in energy within each package and paired by position
  (state 1 = lowest, etc.). NOTE: STEOM root ordering can differ between packages
  for near-degenerate states (e.g. D2h acenes — see the known root jitter), so an
  index pairing may misalign a near-degenerate pair; inspect the table, don't trust
  a single large diff blindly. The per-molecule MAD/MAX summary goes to stderr.
"""
import sys, os, re, glob, csv

HERE = os.path.dirname(os.path.abspath(__file__))
GAN_LOGDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "logs")
ORCA_DIR   = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "orca", "inp")

# natoms per molecule (for ordering the table by size); optional.
def load_natoms():
    d = {}
    series = os.path.join(HERE, "series.tsv")
    if os.path.exists(series):
        with open(series, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3:
                    try:
                        d[p[0]] = int(p[2])
                    except ValueError:
                        pass
    return d

NATOMS = load_natoms()

# ---- GANSU: last "STEOM excited-state energies" block → [(k, eV), ...] -----------
GAN_ROW = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")

def gansu_roots(text):
    if "STEOM excited-state energies" not in text:
        return []
    seg = text.split("STEOM excited-state energies")[-1]
    roots = []
    for line in seg.splitlines()[1:]:          # skip the header line itself
        if "omega" in line:                    # the "k  omega(Ha)  omega(eV)" header
            continue
        m = GAN_ROW.match(line)
        if m:
            roots.append((int(m.group(1)), float(m.group(3))))   # (k, eV)
        elif roots:                            # first non-row after rows started → done
            break
    return roots

# ---- ORCA: final STEOM-CCSD RESULTS (SINGLETS) only --------------------------------
# The whole .out also contains IROOT lines from the IP-EOM and EA-EOM active-state
# solves (no <S**2> tag, before the RESULTS header). Scope to the SINGLETS results
# section AND require the <S**2> tag so only the genuine excitations are taken.
ORCA_ROW = re.compile(
    r"IROOT\s*=\s*(\d+)\s*:\s*[-0-9.]+\s*au\s+([-0-9.]+)\s*eV\s+[0-9.]+\s*cm\*\*-1\s*<S\*\*2>",
    re.I)

def orca_roots(text):
    m = re.search(r"STEOM-CCSD RESULTS \(SINGLETS\)(.*?)(?:STEOM-CCSD RESULTS|\Z)",
                  text, re.S)
    scope = m.group(1) if m else ""
    seen = {}
    for mm in ORCA_ROW.finditer(scope):
        k = int(mm.group(1))
        if k not in seen:
            seen[k] = float(mm.group(2))
    return sorted(seen.items())                # [(iroot, eV), ...]

# ---- collect per molecule --------------------------------------------------------
gan = {}   # mol -> [(k, eV)]
for log in glob.glob(os.path.join(GAN_LOGDIR, "dlpno_steom_g*_*.log")):
    m = re.match(r"dlpno_steom_g\d+_(.+)\.log$", os.path.basename(log))
    if not m:
        continue
    with open(log, encoding="utf-8", errors="replace") as fh:
        r = gansu_roots(fh.read())
    if r:
        gan[m.group(1)] = r

orca = {}  # mol -> [(iroot, eV)]
for out in glob.glob(os.path.join(ORCA_DIR, "dlpno_steom__*.out")):
    m = re.match(r"dlpno_steom__(.+)\.out$", os.path.basename(out))
    if not m:
        continue
    with open(out, encoding="utf-8", errors="replace") as fh:
        r = orca_roots(fh.read())
    if r:
        orca[m.group(1)] = r

# ---- pair + tabulate -------------------------------------------------------------
cols = ["molecule","natoms","state","gansu_k","gansu_eV","orca_iroot","orca_eV",
        "diff_eV","abs_diff_meV"]
w = csv.DictWriter(sys.stdout, fieldnames=cols); w.writeheader()

mols = sorted(set(gan) | set(orca), key=lambda x: (NATOMS.get(x, 1 << 30), x))
for mol in mols:
    g = sorted(gan.get(mol, []), key=lambda t: t[1])   # ascending eV
    o = sorted(orca.get(mol, []), key=lambda t: t[1])
    diffs = []
    n = max(len(g), len(o))
    for i in range(n):
        gk, gev = g[i] if i < len(g) else ("", "")
        oi, oev = o[i] if i < len(o) else ("", "")
        diff = absmev = ""
        if gev != "" and oev != "":
            diff = gev - oev
            absmev = abs(diff) * 1000.0
            diffs.append(abs(diff))
        w.writerow(dict(
            molecule=mol, natoms=NATOMS.get(mol, ""), state=i + 1,
            gansu_k=gk, gansu_eV=("%.4f" % gev) if gev != "" else "",
            orca_iroot=oi, orca_eV=("%.4f" % oev) if oev != "" else "",
            diff_eV=("%+.4f" % diff) if diff != "" else "",
            abs_diff_meV=("%.1f" % absmev) if absmev != "" else "",
        ))
    if diffs:
        mad = sum(diffs) / len(diffs)
        sys.stderr.write("[%-16s] paired=%d  MAD=%.0f meV  MAX=%.0f meV\n"
                         % (mol, len(diffs), mad * 1000, max(diffs) * 1000))
    else:
        miss = "GANSU" if mol not in gan else "ORCA"
        sys.stderr.write("[%-16s] no pair (missing %s roots)\n" % (mol, miss))
