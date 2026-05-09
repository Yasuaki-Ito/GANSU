#!/usr/bin/env python3
"""
H2 dissociation curve: HF / UHF / FCI / MP2 / MP3 / CCSD / CCSD(T)

Runs GANSU's HF_main for each method at various bond distances and plots.

Usage (from build/):
    wsl python3 ../script/h2_dissociation.py
    wsl python3 ../script/h2_dissociation.py --basis cc-pvdz --rmin 0.4 --rmax 5.0 --npts 24

Output:
    h2_dissociation.csv
    h2_dissociation.png
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

# (label, method_flag, post_hf_flag)
METHODS = [
    ("RHF",    "RHF",  "none"),
    ("UHF",    "UHF",  "none"),
    ("MP2",    "RHF",  "mp2"),
    ("MP3",    "RHF",  "mp3"),
    ("CCSD",   "RHF",  "ccsd"),
    ("CCSD(T)","RHF",  "ccsd_t"),
    ("FCI",    "RHF",  "fci"),
]


def run_gansu(exe, xyz_path, basis_path, method, post_hf):
    cmd = [exe, "-x", xyz_path, "-g", basis_path, "-m", method,
           "--post_hf_method", post_hf]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if res.returncode != 0:
        sys.stderr.write(f"  FAILED {method}/{post_hf}: exit={res.returncode}\n")
        sys.stderr.write(res.stdout[-400:] + "\n")
        return None
    # Match numbers including scientific notation (e.g. -9.9856e-01)
    num = r"(-?\d+\.\d+(?:[eE][+-]?\d+)?)"
    txt = res.stdout
    m = re.search(r"Total Energy \(including post-HF correction\):\s+" + num, txt)
    if m is None:
        m = re.search(r"Total Energy:\s+" + num, txt)
    if m is None:
        sys.stderr.write(f"  could not parse energy for {method}/{post_hf}\n")
        return None
    return float(m.group(1))


def make_xyz(path, R):
    with open(path, "w") as f:
        f.write("2\n")
        f.write(f"H2 at R={R:.4f} A\n")
        f.write("H 0.0 0.0 0.0\n")
        f.write(f"H 0.0 0.0 {R:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="./HF_main")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--basis-dir", default="../basis")
    ap.add_argument("--rmin", type=float, default=0.4)
    ap.add_argument("--rmax", type=float, default=5.0)
    ap.add_argument("--npts", type=int, default=24)
    ap.add_argument("--out", default="h2_dissociation")
    args = ap.parse_args()

    basis_path = os.path.join(args.basis_dir, f"{args.basis}.gbs")
    if not os.path.isfile(basis_path):
        sys.exit(f"basis not found: {basis_path}")
    if not os.path.isfile(args.exe):
        sys.exit(f"exe not found: {args.exe}")

    # Distances (uniform grid)
    import numpy as np
    Rs = np.linspace(args.rmin, args.rmax, args.npts)

    results = {lbl: [] for lbl, _, _ in METHODS}

    with tempfile.TemporaryDirectory() as tmpd:
        xyz_path = os.path.join(tmpd, "h2.xyz")
        for R in Rs:
            make_xyz(xyz_path, R)
            sys.stdout.write(f"R = {R:.3f} A :")
            sys.stdout.flush()
            for lbl, hf, post in METHODS:
                e = run_gansu(args.exe, xyz_path, basis_path, hf, post)
                results[lbl].append(e)
                sys.stdout.write(f" {lbl}={e if e is None else f'{e:.6f}'}")
                sys.stdout.flush()
            sys.stdout.write("\n")

    # CSV
    csv_path = f"{args.out}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["R_A"] + [lbl for lbl, _, _ in METHODS])
        for i, R in enumerate(Rs):
            row = [f"{R:.4f}"] + [
                "" if results[lbl][i] is None else f"{results[lbl][i]:.10f}"
                for lbl, _, _ in METHODS
            ]
            w.writerow(row)
    print(f"wrote {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    styles = {
        "RHF":     {"color": "tab:blue",   "ls": "-",  "marker": "o"},
        "UHF":     {"color": "tab:cyan",   "ls": "--", "marker": "s"},
        "MP2":     {"color": "tab:green",  "ls": "-",  "marker": "^"},
        "MP3":     {"color": "tab:olive",  "ls": "-",  "marker": "v"},
        "CCSD":    {"color": "tab:orange", "ls": "-",  "marker": "D"},
        "CCSD(T)": {"color": "tab:red",    "ls": "-",  "marker": "P"},
        "FCI":     {"color": "black",      "ls": "-",  "marker": "*"},
    }
    for lbl, _, _ in METHODS:
        ys = results[lbl]
        xs_plot = [R for R, y in zip(Rs, ys) if y is not None]
        ys_plot = [y for y in ys if y is not None]
        if not ys_plot:
            continue
        ax.plot(xs_plot, ys_plot, label=lbl, **styles.get(lbl, {}))

    ax.set_xlabel("H-H distance (Å)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title(f"H$_2$ dissociation curve  (basis: {args.basis})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"{args.out}.png"
    plt.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
