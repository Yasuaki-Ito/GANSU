#!/usr/bin/env python3
"""DLPNO-CCSD scaling benchmark for Step 6 series (post-Step 6.6).

Sweeps a graduated series of molecules and captures:
  - Wall time + GPU peak memory
  - DLPNO-CCSD sub-timers (LMP2 driver, T1, phase24, T2 iter)
  - Per-iter CCSD breakdown (picache / resid / "other" rgpu compute / etc.)
  - Pair statistics (N_pair, avg n_pno, avg n_pao, max_n estimate)
  - Energy (E_corr, E_total) for bit-exact tracking across re-runs

Comparison vs RI-CCSD is OPT-IN per molecule via `ref` column in MOLECULES
(set to True for small molecules where RI-CCSD finishes; False for large
ones where it would OOM or time out).

Usage (from build/ directory):
    python3 ../script/bench_dlpno_step6.py --num_gpus 8 --out bench_step6.md
    python3 ../script/bench_dlpno_step6.py --molecules Phthalocyanine Decacene
    python3 ../script/bench_dlpno_step6.py --dry-run

Note: For large molecules (>~80 atoms in cc-pVDZ), the Step 6.2/6.5/6.6
ResidGpu padded buffers may exceed GPU memory and silently fall back to
CPU. The per-iter "other" column reveals this (CPU fallback shows
significantly different patterns than GPU-active runs).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MOLECULES = [
    # (display, xyz, ref_RI_CCSD)
    # ---- Water cluster scaling (small → medium) ----
    ("H2O_dimer",     "H2O_dimer.xyz",          True),
    ("H2O_trimer",    "H2O_trimer_cyclic.xyz",  True),
    ("H2O_hexamer",   "H2O_hexamer_prism.xyz",  True),
    # ---- Aromatic / saturated organic (small → medium) ----
    ("Benzene",       "Benzene.xyz",            True),
    ("Anthracene",    "Anthracene.xyz",         True),
    ("Decane",        "Decane.xyz",             True),
    # ---- Medium (50-80 atoms) ----
    ("Dioxin",        "large_molecular/Dioxin.xyz",            False),
    ("Phthalocyanine","large_molecular/Phthalocyanine.xyz",    False),
    ("Decacene",      "large_molecular/Decacene.xyz",          False),
    ("Cholesterol",   "large_molecular/Cholesterol.xyz",       False),
    # ---- Larger (80-100 atoms) — likely to push ResidGpu memory ----
    ("Dicoronylene",  "large_molecular/Dicoronylene.xyz",      False),
    ("Tetraphenylporphyrin",
                      "large_molecular/Tetraphenylporphyrin.xyz", False),
    ("C90",           "large_molecular/C90.xyz",               False),
    ("Beta-Carotene", "large_molecular/Beta-Carotene.xyz",     False),
]


# ---------------------------------------------------------------------------
# Output regex parsers — extract sub-timers + statistics from gansu's stdout.
# ---------------------------------------------------------------------------
_RE_E_CORR  = re.compile(
    r"Post-HF\s+energy\s+correction:\s*(-?\d+\.\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE)
_RE_E_TOTAL = re.compile(
    r"Total\s+Energy\s+\(including\s+post-HF\s+correction\):\s*"
    r"(-?\d+\.\d+(?:[eE][-+]?\d+)?)", re.IGNORECASE)

_RE_NAO     = re.compile(r"Number of basis functions:\s*(\d+)")
_RE_NOCC    = re.compile(r"Occupied orbitals:\s*(\d+)")

_RE_PEAK_MEM = re.compile(r"Peak usage:\s*([\d.]+)\s*([KMG]?)B")
_RE_POST     = re.compile(r"END:\s+post_process_after_scf after\s+([\d.eE+-]+)\s*ms")

# DLPNO sub-timers
_RE_LMP2     = re.compile(r"\[DLPNO-CCSD-PROF\]\s+solve_dlpno_lmp2\s*=\s*([\d.]+)\s*s")
_RE_PHASE24  = re.compile(r"\[DLPNO-CCSD-PROF\]\s+precompute_phase24_integrals\s*=\s*([\d.]+)\s*s")
_RE_T2_ITER  = re.compile(r"\[DLPNO-CCSD-PROF\]\s+iterate_dlpno_ccsd_t2\s*=\s*([\d.]+)\s*s\s*\((\d+)\s*iter\)")

# Pair statistics from the [DLPNO-MP2] setup line
_RE_PAIRS    = re.compile(r"setup\s+pairs=(\d+)\s+avg n_pao=([\d.]+)\s+avg n_pno=([\d.]+)")

# CCSD T2 iter PROF line: barS=… vmeta=… picache=… dFki=… DFpair=… resid=… diis=… other=…
_RE_ITER_PROF = re.compile(
    r"\[DLPNO-ITER-PROF\]\s+CCSD T2.*?\s+total=([\d.]+)s"
    r".*?picache=([\d.]+).*?resid=([\d.]+).*?other=([\d.]+)",
    re.DOTALL)


def parse_run_output(text):
    out = {}
    m = _RE_E_CORR.search(text);   out["e_corr"]  = float(m.group(1)) if m else None
    m = _RE_E_TOTAL.search(text);  out["e_total"] = float(m.group(1)) if m else None
    m = _RE_NAO.search(text);      out["nao"]     = int(m.group(1))   if m else None
    m = _RE_NOCC.search(text);     out["nocc"]    = int(m.group(1))   if m else None

    m = _RE_PEAK_MEM.search(text)
    if m:
        v, unit = float(m.group(1)), m.group(2)
        scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[unit]
        out["peak_mem_gb"] = v * scale / (1024**3)
    else:
        out["peak_mem_gb"] = None

    m = _RE_POST.search(text)
    out["post_s"] = float(m.group(1)) / 1000.0 if m else None
    m = _RE_LMP2.search(text);     out["lmp2_s"]    = float(m.group(1)) if m else None
    m = _RE_PHASE24.search(text);  out["phase24_s"] = float(m.group(1)) if m else None
    m = _RE_T2_ITER.search(text)
    out["t2_iter_s"] = float(m.group(1)) if m else None
    out["n_iters"]   = int(m.group(2))   if m else None

    m = _RE_PAIRS.search(text)
    if m:
        out["n_pair"]    = int(m.group(1))
        out["avg_n_pao"] = float(m.group(2))
        out["avg_n_pno"] = float(m.group(3))
    else:
        out["n_pair"] = out["avg_n_pao"] = out["avg_n_pno"] = None

    m = _RE_ITER_PROF.search(text)
    if m:
        out["iter_total_s"] = float(m.group(1))
        out["iter_picache"] = float(m.group(2))
        out["iter_resid"]   = float(m.group(3))
        out["iter_other"]   = float(m.group(4))
    else:
        out["iter_total_s"] = out["iter_picache"] = out["iter_resid"] = out["iter_other"] = None
    return out


def run_one(gansu_bin, xyz, basis, aux, method, num_gpus, preset, dry, timeout):
    args = [
        str(gansu_bin),
        "-x", str(xyz),
        "-g", str(basis),
        "-ag", str(aux),
        "--eri_method", "ri",
        "--post_hf_method", method,
        "--dlpno_preset", preset,
        "--dlpno_verbose", "1",
        "--convergence_energy_threshold", "1e-10",
    ]
    if num_gpus > 1:
        args += ["--num_gpus", str(num_gpus)]

    if dry:
        print("    [dry-run]", " ".join(args))
        return {"ok": True, "wall_s": 0.0}

    t0 = time.perf_counter()
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "wall_s": float(timeout), "err": f"TIMEOUT (>{timeout}s)"}
    t = time.perf_counter() - t0

    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "")[-800:]
        return {"ok": False, "wall_s": t, "err": tail.strip()}

    parsed = parse_run_output(p.stdout)
    parsed["ok"]      = True
    parsed["wall_s"]  = t
    return parsed


def fmt(v, prec=2, default="—"):
    if v is None:
        return default
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def write_markdown(rows, out_path, num_gpus, preset, basis, gansu_bin):
    lines = []
    lines.append(f"# DLPNO-CCSD Step 6 scaling benchmark\n")
    lines.append(f"- gansu binary: `{gansu_bin}`")
    lines.append(f"- basis: `{basis}`")
    lines.append(f"- preset: `{preset}`, num_gpus={num_gpus}\n")

    lines.append("## DLPNO-CCSD timings\n")
    lines.append("| Molecule | nao | nocc | N_pair | avg n_pno | "
                 "post (s) | LMP2 (s) | T2 iter (s) | iter (s/iter) | "
                 "picache (s) | resid (s) | other (s) | peak GPU (GB) | E_corr (Ha) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        if not r.get("ok"):
            lines.append(f"| **{r['mol']}** | FAIL: {r.get('err','?')[:60]} ||||||||||||")
            continue
        n_iter = r.get("n_iters")
        per_iter = (r["iter_total_s"] / n_iter) if (r.get("iter_total_s") and n_iter) else None
        lines.append(
            f"| **{r['mol']}** | "
            f"{fmt(r.get('nao'),0)} | "
            f"{fmt(r.get('nocc'),0)} | "
            f"{fmt(r.get('n_pair'),0)} | "
            f"{fmt(r.get('avg_n_pno'),2)} | "
            f"{fmt(r.get('post_s'),2)} | "
            f"{fmt(r.get('lmp2_s'),2)} | "
            f"{fmt(r.get('t2_iter_s'),2)} | "
            f"{fmt(per_iter,3)} | "
            f"{fmt(r.get('iter_picache'),2)} | "
            f"{fmt(r.get('iter_resid'),3)} | "
            f"{fmt(r.get('iter_other'),2)} | "
            f"{fmt(r.get('peak_mem_gb'),2)} | "
            f"{fmt(r.get('e_corr'),8)} |"
        )

    # Reference RI-CCSD section (only molecules where ref=True attempted)
    ri_rows = [r for r in rows if r.get("method") == "RI-CCSD"]
    if ri_rows:
        lines.append("\n## RI-CCSD reference (small molecules only)\n")
        lines.append("| Molecule | post (s) | E_corr (Ha) | DLPNO recovery (%) |")
        lines.append("|---|---:|---:|---:|")
        for r in ri_rows:
            if not r.get("ok"):
                lines.append(f"| {r['mol']} | FAIL ||")
                continue
            dlpno = next((x for x in rows
                          if x.get("ok") and x["mol"] == r["mol"]
                             and x.get("method") == "DLPNO-CCSD"), None)
            rec = (100 * dlpno["e_corr"] / r["e_corr"]) if (dlpno and dlpno.get("e_corr") and r.get("e_corr")) else None
            lines.append(f"| {r['mol']} | "
                         f"{fmt(r.get('post_s'),2)} | "
                         f"{fmt(r.get('e_corr'),8)} | "
                         f"{fmt(rec,2)} |")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\n[bench] Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gansu",     default="./gansu")
    ap.add_argument("--xyz_dir",   default="../xyz")
    ap.add_argument("--basis",     default="../basis/cc-pvdz.gbs")
    ap.add_argument("--aux",       default="../auxiliary_basis/cc-pvdz-rifit.gbs")
    ap.add_argument("--num_gpus",  type=int, default=1)
    ap.add_argument("--preset",    default="normal")
    ap.add_argument("--out",       default="bench_step6.md")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--molecules", nargs="+", default=None,
                    help="restrict to a subset")
    ap.add_argument("--skip-ri",   action="store_true",
                    help="skip RI-CCSD reference even for small molecules")
    ap.add_argument("--timeout",   type=int, default=7200,
                    help="per-run timeout in seconds")
    args = ap.parse_args()

    gansu = Path(args.gansu).resolve()
    if not args.dry_run and not gansu.exists():
        print(f"ERROR: gansu binary not found at {gansu}", file=sys.stderr)
        sys.exit(2)

    xyz_dir = Path(args.xyz_dir)
    basis   = Path(args.basis)
    aux     = Path(args.aux)

    mols = MOLECULES
    if args.molecules:
        wanted = set(args.molecules)
        mols = [m for m in MOLECULES if m[0] in wanted]
        missing = wanted - {m[0] for m in mols}
        if missing:
            print(f"WARNING: unknown molecules ignored: {sorted(missing)}",
                  file=sys.stderr)

    print(f"[bench] gansu = {gansu}")
    print(f"[bench] basis = {basis}")
    print(f"[bench] num_gpus = {args.num_gpus}, preset = {args.preset}")
    print(f"[bench] {len(mols)} molecules to run")

    rows = []
    for mol_name, xyz_rel, do_ri in mols:
        xyz = xyz_dir / xyz_rel
        if not args.dry_run and not xyz.exists():
            print(f"  SKIP {mol_name}: {xyz} not found")
            continue

        print(f"\n[bench] {mol_name}  ({xyz_rel})")
        # DLPNO-CCSD (always)
        print("  -> DLPNO-CCSD ...", end="", flush=True)
        r = run_one(gansu, xyz, basis, aux, "dlpno_ccsd",
                    args.num_gpus, args.preset, args.dry_run, args.timeout)
        r["mol"] = mol_name
        r["method"] = "DLPNO-CCSD"
        if r.get("ok"):
            print(f" {r.get('wall_s', 0):.1f} s, post={fmt(r.get('post_s'))} s")
        else:
            print(f" FAIL: {r.get('err','?')[:80]}")
        rows.append(r)

        # RI-CCSD reference (only if requested and small)
        if do_ri and not args.skip_ri:
            print("  -> RI-CCSD  ...", end="", flush=True)
            r2 = run_one(gansu, xyz, basis, aux, "ccsd",
                         args.num_gpus, args.preset, args.dry_run, args.timeout)
            r2["mol"] = mol_name
            r2["method"] = "RI-CCSD"
            if r2.get("ok"):
                print(f" {r2.get('wall_s', 0):.1f} s, post={fmt(r2.get('post_s'))} s")
            else:
                print(f" FAIL: {r2.get('err','?')[:80]}")
            rows.append(r2)

        # Tag DLPNO method (post-hoc) so output sorting works
        r["method"] = "DLPNO-CCSD"

    write_markdown(rows, Path(args.out), args.num_gpus,
                   args.preset, basis, gansu)


if __name__ == "__main__":
    main()
