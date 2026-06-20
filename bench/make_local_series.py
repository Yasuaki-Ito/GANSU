#!/usr/bin/env python3
"""Generate per-chromophore size-series TSV files for the local-excitation STEOM
benchmark (xyz/bench_steom/*.xyz), in the same format run_gansu.sh / run_orca.sh /
make_orca_inputs.py consume (name  xyz_relpath  natoms  tags).

    python3 bench/make_local_series.py
  -> bench/series_aldehyde.tsv  series_ketone.tsv  series_amide.tsv
     series_nitrile.tsv  series_alkylbenzene.tsv

Then benchmark a family GANSU-vs-ORCA, e.g.:
    cd build
    SERIES=../bench/series_aldehyde.tsv bash ../bench/run_gansu.sh dlpno_ccsd 4 120
    SERIES=bench/series_aldehyde.tsv python3 bench/make_orca_inputs.py 64 3000
    SERIES=bench/series_aldehyde.tsv ORCA=/opt/orca6 bash bench/run_orca.sh dlpno_ccsd 240
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
XYZDIR = os.path.join(ROOT, "xyz", "bench_steom")
TAGS = "rihf,rimp2,dlpno_ccsd,dlpno_ccsd_t,dlpno_steom"

# ordered (small -> large) molecule names per family (must match xyz/bench_steom/*.xyz)
FAMILIES = {
    "aldehyde": ["formaldehyde","acetaldehyde","propanal","butanal","pentanal","hexanal",
                 "heptanal","octanal","nonanal","decanal","undecanal","dodecanal",
                 "tridecanal","tetradecanal","hexadecanal","octadecanal","icosanal"],
    "ketone":   ["acetone","2-butanone","2-pentanone","2-hexanone","2-octanone",
                 "2-decanone","2-dodecanone"],
    "amide":    ["formamide","acetamide","propanamide","pentanamide","heptanamide",
                 "nonanamide","undecanamide"],
    "nitrile":  ["acetonitrile","propionitrile","butyronitrile","hexanenitrile",
                 "octanenitrile","decanenitrile","dodecanenitrile"],
    "alkylbenzene": ["benzene","toluene","ethylbenzene","propylbenzene","butylbenzene",
                     "pentylbenzene","hexylbenzene","heptylbenzene","octylbenzene",
                     "nonylbenzene","decylbenzene","dodecylbenzene","tetradecylbenzene",
                     "hexadecylbenzene"],
}

def natoms(name):
    p = os.path.join(XYZDIR, name + ".xyz")
    with open(p) as f:
        return int(f.readline().split()[0])

for fam, names in FAMILIES.items():
    out = os.path.join(HERE, f"series_{fam}.tsv")
    n = 0
    with open(out, "w") as f:
        f.write(f"# {fam} local-chromophore size series (xyz/bench_steom). "
                f"Columns: name  xyz_relpath  natoms  tags\n")
        for nm in names:
            xp = os.path.join(XYZDIR, nm + ".xyz")
            if not os.path.exists(xp):
                print(f"  skip {nm} (no {xp})"); continue
            f.write(f"{nm}\t../xyz/bench_steom/{nm}.xyz\t{natoms(nm)}\t{TAGS}\n")
            n += 1
    print(f"wrote {out}  ({n} molecules)")
