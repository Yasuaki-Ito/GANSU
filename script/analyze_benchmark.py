#!/usr/bin/env python3
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
"""Analyze the auto-fragment vs full-STEOM benchmark logs.

Usage: analyze_benchmark.py <OUTDIR> "name|xyz|basis|nstate|ncis" ...

For each molecule it reads <OUTDIR>/<name>_full.log (whole-molecule STEOM
reference) and <OUTDIR>/<name>_auto.log (auto-fragment DMET-STEOM), and reports
per state the reference energy, the auto energy, their difference, the bath
gauge, and the STEOM active character eta, plus MAE / RMSE / max deviation.
"""
import re, sys, os, math

def steom_roots(path):
    """Return list of (eV, eta) from the LAST STEOM energy block, or [] ."""
    if not os.path.exists(path):
        return []
    txt = open(path, errors="ignore").read().splitlines()
    blocks, cur, inblk = [], [], False
    for ln in txt:
        if "STEOM excited-state energies" in ln:
            cur, inblk = [], True; continue
        if inblk and "active-space health" in ln:
            blocks.append(cur); inblk = False; continue
        if inblk:
            m = re.match(r"\s*\d+\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", ln)
            if m:
                cur.append((float(m.group(2)), float(m.group(3))))  # (eV, eta)
    if inblk and cur:
        blocks.append(cur)
    return blocks[-1] if blocks else []

def gauge(path):
    if not os.path.exists(path): return None
    g = re.findall(r"occupation-weighted uncaptured = ([\d.]+)", open(path, errors="ignore").read())
    return float(g[-1]) if g else None

def selected(path):
    if not os.path.exists(path): return None, None
    m = re.findall(r"selected (\d+) atom\(s\) \(coverage=([\d.]+)", open(path, errors="ignore").read())
    return (int(m[-1][0]), float(m[-1][1])) if m else (None, None)

def _phase_s(txt, name):
    """Largest 'END: <name>... after X ms' duration, in seconds (0 if absent)."""
    best = 0.0
    for m in re.finditer(rf"END:\s+{re.escape(name)}\w* after ([\d.]+) ms", txt):
        best = max(best, float(m.group(1)))
    return best / 1000.0

def timing(path):
    """Wall-clock split: full-system preprocessing (CIS-NTO) vs cluster solver.
    Returns (t_pre_s, t_solver_s) or (None, None)."""
    if not os.path.exists(path): return None, None
    txt = open(path, errors="ignore").read()
    t_pre = _phase_s(txt, "compute_cis_nto")                 # includes the full-system CIS
    t_sol = _phase_s(txt, "compute_steom_ccsd_impl")         # STEOM cluster solve
    if t_sol == 0.0:                                          # ADC(2) path
        t_sol = max(_phase_s(txt, "compute_adc2_impl"),
                    _phase_s(txt, "post_process_after_scf") - t_pre)
    return (t_pre or None), (t_sol or None)

def main():
    outdir = sys.argv[1]
    mols = [e.split("|")[0] for e in sys.argv[2:]]
    all_abs = []                    # |error| over high-eta matched states, all molecules
    print("="*78)
    print("Auto-fragment DMET-STEOM  vs  full (whole-molecule) STEOM-CCSD")
    print("="*78)
    for name in mols:
        full = steom_roots(os.path.join(outdir, f"{name}_full.log"))
        auto = steom_roots(os.path.join(outdir, f"{name}_auto.log"))
        nat, cov = selected(os.path.join(outdir, f"{name}_auto.log"))
        u = gauge(os.path.join(outdir, f"{name}_auto.log"))
        tpre, tsol = timing(os.path.join(outdir, f"{name}_auto.log"))
        head = f"[{name}]  fragment: {nat} atoms (coverage {cov})  bath gauge u={u}"
        print("\n"+head)
        if tpre and tsol:
            print(f"  wall-clock: preprocessing (full-system CIS-NTO) {tpre:.1f} s"
                  f"  vs  cluster solver {tsol:.1f} s  (ratio 1:{tsol/tpre:.1f})")
        if not auto:
            print("  (no auto energies parsed)"); continue
        if not full:
            print("  full reference not available (too large?) -- auto-only:")
            print("   k   auto(eV)   eta")
            for k,(e,et) in enumerate(auto):
                print(f"  {k:2d}   {e:8.4f}   {et:.3f}")
            continue
        n = min(len(full), len(auto))
        print("   k   full(eV)   auto(eV)     dE(meV)   eta     flag")
        errs_all, errs_val = [], []
        for k in range(n):
            fe, _ = full[k]; ae, aeta = auto[k]
            d = (ae-fe)*1000.0
            flag = "" if aeta >= 0.96 else "low-eta"
            print(f"  {k:2d}   {fe:8.4f}   {ae:8.4f}   {d:8.1f}   {aeta:.3f}  {flag}")
            errs_all.append(abs(ae-fe))
            if aeta >= 0.96: errs_val.append(abs(ae-fe)); all_abs.append(abs(ae-fe))
        def stats(x):
            if not x: return None
            mae=sum(x)/len(x); rmse=math.sqrt(sum(v*v for v in x)/len(x)); mx=max(x)
            return mae,rmse,mx
        sa, sv = stats(errs_all), stats(errs_val)
        if sa: print(f"   all {n} states : MAE {sa[0]*1000:6.1f}  RMSE {sa[1]*1000:6.1f}  max {sa[2]*1000:6.1f} meV")
        if sv: print(f"   eta>=0.96     : MAE {sv[0]*1000:6.1f}  RMSE {sv[1]*1000:6.1f}  max {sv[2]*1000:6.1f} meV")
    if all_abs:
        mae=sum(all_abs)/len(all_abs); rmse=math.sqrt(sum(v*v for v in all_abs)/len(all_abs))
        print("\n"+"="*78)
        print(f"OVERALL (eta>=0.96 valence states, {len(all_abs)} states): "
              f"MAE {mae*1000:.1f} meV  RMSE {rmse*1000:.1f} meV  max {max(all_abs)*1000:.1f} meV")
        print("="*78)

if __name__ == "__main__":
    main()
