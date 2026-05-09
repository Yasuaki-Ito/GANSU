#!/usr/bin/env python3
"""DLPNO benchmark — timing + recovery on a series of molecules.

Compares DLPNO-MP2 / DLPNO-CCSD against canonical RI-MP2 / RI-CCSD
on a small set of molecules and emits a markdown table.

Usage (from build/ directory):
    python3 ../script/bench_dlpno.py --num_gpus 1 --out bench_a100.md
    python3 ../script/bench_dlpno.py --num_gpus 4 --out bench_h200x4.md

Energies are parsed from the gansu CLI's
    "Post-HF energy correction: <value> [hartree]"
print line. Wall time is measured with time.perf_counter() around the
gansu subprocess.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


MOLECULES = [
    # (display name, xyz file relative to xyz_dir)
    ("H2O",         "H2O.xyz"),
    ("H2O_dimer",   "H2O_dimer.xyz"),
    ("H2O_trimer",  "H2O_trimer_cyclic.xyz"),
    ("H2O_hexamer", "H2O_hexamer_prism.xyz"),
    ("Benzene",     "Benzene.xyz"),
]

METHODS = [
    # (display name, --post_hf_method)
    ("DLPNO-MP2",  "dlpno_mp2"),
    ("RI-MP2",     "mp2"),
    ("DLPNO-CCSD", "dlpno_ccsd"),
    ("RI-CCSD",    "ccsd"),
]


_ENERGY_RE = re.compile(
    r"Post-HF\s+energy\s+correction:\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(
    r"Total\s+Energy\s+\(including\s+post-HF\s+correction\):\s*"
    r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_energy(text):
    """Return (E_corr, E_total) or (None, None) if not found."""
    m_corr = _ENERGY_RE.search(text)
    m_tot  = _TOTAL_RE.search(text)
    e_corr = float(m_corr.group(1)) if m_corr else None
    e_tot  = float(m_tot.group(1))  if m_tot  else None
    return e_corr, e_tot


def run_one(gansu_bin, xyz, basis, aux, method, num_gpus, preset, dry):
    args = [
        str(gansu_bin),
        "-x", str(xyz),
        "-g", str(basis),
        "-ag", str(aux),
        "--eri_method", "ri",
        "--post_hf_method", method,
        "--dlpno_preset", preset,
        "--dlpno_verbose", "0",
        "--convergence_energy_threshold", "1e-10",
    ]
    if num_gpus > 1:
        args += ["--num_gpus", str(num_gpus)]

    if dry:
        print("    [dry-run]", " ".join(args))
        return {"ok": True, "wall_s": 0.0, "e_corr": 0.0, "e_total": 0.0}

    t0 = time.perf_counter()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {"ok": False, "wall_s": 7200.0, "err": "TIMEOUT (>2h)"}
    t = time.perf_counter() - t0

    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-500:]
        return {"ok": False, "wall_s": t, "err": tail.strip()}

    e_corr, e_total = parse_energy(p.stdout)
    return {"ok": True, "wall_s": t,
            "e_corr": e_corr, "e_total": e_total}


def write_markdown(rows, out_path, num_gpus, preset, basis, gansu_bin):
    lines = []
    lines.append(f"# DLPNO benchmark ({preset} preset, num_gpus={num_gpus})\n")
    lines.append(f"- gansu binary: `{gansu_bin}`")
    lines.append(f"- basis: `{basis}`")
    lines.append("")
    lines.append("## Timing & correlation energies\n")
    lines.append("| Molecule | Method | Wall (s) | E_corr (Ha) | "
                 "Recovery (%) | E_total (Ha) |")
    lines.append("|---|---|---:|---:|---:|---:|")
    seen_mols = []
    for r in rows:
        if r["mol"] not in seen_mols:
            seen_mols.append(r["mol"])

    for mol in seen_mols:
        mol_rows = [r for r in rows if r["mol"] == mol]
        ri_mp2  = next((r for r in mol_rows
                        if r["method"] == "RI-MP2"  and r["ok"]), None)
        ri_ccsd = next((r for r in mol_rows
                        if r["method"] == "RI-CCSD" and r["ok"]), None)
        for r in mol_rows:
            if not r["ok"]:
                lines.append(
                    f"| {mol} | {r['method']} | "
                    f"{r['wall_s']:.1f} | FAIL: {r.get('err','?')[:40]} | — | — |"
                )
                continue
            ec = r.get("e_corr")
            et = r.get("e_total")
            ec_s = f"{ec:.6f}" if ec is not None else "—"
            et_s = f"{et:.6f}" if et is not None else "—"
            ref = ri_mp2 if "MP2" in r["method"] else ri_ccsd
            if ref is None or ec is None or ref.get("e_corr") is None:
                rec_s = "—"
            else:
                rec_s = f"{100.0 * ec / ref['e_corr']:.2f}"
            if r["method"] in ("RI-MP2", "RI-CCSD"):
                rec_s = "ref"
            lines.append(
                f"| {mol} | {r['method']} | {r['wall_s']:.1f} | "
                f"{ec_s} | {rec_s} | {et_s} |"
            )
        lines.append("")  # blank line between molecules

    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gansu", default="./gansu",
                    help="path to gansu binary (relative to cwd)")
    ap.add_argument("--xyz_dir", default="../xyz",
                    help="directory containing the .xyz files")
    ap.add_argument("--basis", default="../basis/cc-pvdz.gbs")
    ap.add_argument("--aux",   default="../auxiliary_basis/cc-pvdz-rifit.gbs")
    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--preset", default="normal")
    ap.add_argument("--out", default="bench_dlpno.md")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without executing")
    ap.add_argument("--methods", nargs="+", default=None,
                    help="restrict to a subset of method display names")
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="restrict to a subset of molecule display names")
    args = ap.parse_args()

    gansu = Path(args.gansu)
    if not args.dry_run and not gansu.exists():
        print(f"ERROR: gansu binary not found at {gansu}", file=sys.stderr)
        sys.exit(2)
    # Resolve to absolute path so subprocess does not fall back to $PATH
    # lookup (Path("./gansu") gets normalised to "gansu" on Linux, which
    # then misses the cwd-relative binary).
    gansu = gansu.resolve()

    xyz_dir = Path(args.xyz_dir)
    basis = Path(args.basis)
    aux = Path(args.aux)

    mols = MOLECULES
    if args.molecules:
        keep = set(args.molecules)
        mols = [m for m in MOLECULES if m[0] in keep]
    methods = METHODS
    if args.methods:
        keep = set(args.methods)
        methods = [m for m in METHODS if m[0] in keep]

    rows = []
    for mol_name, xyz_file in mols:
        xyz_path = xyz_dir / xyz_file
        if not xyz_path.exists():
            print(f"  SKIP: {mol_name} ({xyz_path} not found)")
            continue
        for method_name, method in methods:
            print(f"  [{mol_name}] {method_name}  num_gpus={args.num_gpus} ...",
                  flush=True)
            r = run_one(gansu, xyz_path, basis, aux, method,
                        args.num_gpus, args.preset, args.dry_run)
            r["mol"] = mol_name
            r["method"] = method_name
            rows.append(r)
            if r["ok"]:
                ec = r.get("e_corr")
                ec_s = f"{ec:.6f}" if ec is not None else "—"
                print(f"      ok  wall={r['wall_s']:.1f}s  E_corr={ec_s}")
            else:
                print(f"      FAILED  wall={r['wall_s']:.1f}s  err={r.get('err','?')[:120]}")

    out_path = Path(args.out)
    write_markdown(rows, out_path, args.num_gpus, args.preset,
                   args.basis, args.gansu)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
