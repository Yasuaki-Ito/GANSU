#!/usr/bin/env python3
# GANSU: GPU Accelerated Numerical Simulation Utility
# Copyright (c) 2025-2026, Hiroshima University and Fujitsu Limited
# SPDX-License-Identifier: BSD-3-Clause
"""
Phase C external grouping driver for CIS-guided DMET-STEOM.

Reads the per-state per-atom localization JSON written by GANSU
(`--dmet_steom_auto_json <path>`) together with the molecular `.xyz`, and
clusters the excited states by *where* they localize — the cosine similarity of
their per-atom weight vectors. Spatially distinct excitations (e.g. an
anthraquinone pi->pi* vs. an n->pi* on a remote carbonyl) fall into different
clusters. For each cluster it:

  1. averages the per-atom localization over the cluster's states,
  2. greedily selects the fragment atoms (per-atom floor, cumulative coverage),
  3. writes a fragment `.xyz` (paper / visualization ready), and
  4. emits a DMET-STEOM job line with that fragment's `--dmet_fragments` spec.

This replaces one scattered state-average fragment with several clean, region-
specific fragments — the "split into per-chromophore jobs" idea. Each emitted job
is independent (embarrassingly parallel across a GPU cluster).

Usage:
    python3 dmet_steom_group_states.py --json dox_states.json --xyz Doxorubicin.xyz \\
            --outdir dox_groups --sim 0.5 --floor 0.01 --coverage 0.92 \\
            --gansu-args "-g cc-pvdz --eri_method ri -ag ../auxiliary_basis/cc-pvdz-rifit.gbs \\
                          --frozen_core auto --num_gpus 4 --initial_guess sad --n_excited_states 5"
"""
import argparse
import json
import math
import os


def cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da * db > 0.0 else 0.0


def cluster_states(vectors, sim_thresh):
    """Single-linkage connected components: states linked when cosine >= sim_thresh."""
    n = len(vectors)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if cosine(vectors[i], vectors[j]) >= sim_thresh:
                union(i, j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Sort groups by their lowest state index (lowest excitation first).
    return [sorted(g) for _, g in sorted(groups.items(), key=lambda kv: min(kv[1]))]


def select_fragment(score, floor, coverage, max_atoms):
    """Greedy: atoms with score >= floor, in descending order, until coverage."""
    order = sorted(range(len(score)), key=lambda a: -score[a])
    picked, cum = [], 0.0
    for a in order:
        if score[a] < floor:
            break
        if max_atoms and len(picked) >= max_atoms:
            break
        picked.append(a)
        cum += score[a]
        if cum >= coverage:
            break
    return sorted(picked), cum


def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    comment = lines[1] if len(lines) > 1 else ""
    atoms = []
    for ln in lines[2:2 + n]:
        p = ln.split()
        atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return comment, atoms


def write_fragment_xyz(path, atoms, idx, header):
    with open(path, "w") as f:
        f.write(f"{len(idx)}\n{header}\n")
        for a in idx:
            el, x, y, z = atoms[a]
            f.write(f"{el:2s} {x:14.8f} {y:14.8f} {z:14.8f}\n")


def brace_spec(idx):
    return "{" + ",".join(str(a) for a in idx) + "}"


def main():
    ap = argparse.ArgumentParser(description="DMET-STEOM per-state grouping + per-fragment xyz")
    ap.add_argument("--json", required=True, help="per-state JSON from --dmet_steom_auto_json")
    ap.add_argument("--xyz", help="molecular .xyz (for fragment geometry output; "
                                  "optional if the JSON carries coordinates)")
    ap.add_argument("--outdir", default="dmet_groups", help="output directory")
    ap.add_argument("--sim", type=float, default=0.5, help="cosine link threshold for clustering")
    ap.add_argument("--floor", type=float, default=0.01, help="per-atom score floor")
    ap.add_argument("--coverage", type=float, default=0.92, help="cumulative coverage target")
    ap.add_argument("--max-atoms", type=int, default=0, help="cap atoms per fragment (0 = none)")
    ap.add_argument("--xyz-path", default="../xyz/molecule.xyz",
                    help="xyz path to embed in the emitted gansu job lines")
    ap.add_argument("--gansu-args", default="",
                    help="common gansu args appended to every emitted job line")
    args = ap.parse_args()

    d = json.load(open(args.json))
    elements = d["elements"]
    per_state = d["per_state"]
    vectors = [s["atom_scores"] for s in per_state]
    n_atoms = d["num_atoms"]

    def top_atoms(v, k=3):
        o = sorted(range(len(v)), key=lambda i: -v[i])[:k]
        return "+".join(f"{i}{elements[i]}" for i in o if v[i] > 0.03)

    # Charge-transfer character per state: hole (occupied) vs particle (virtual)
    # per-atom weight. A low hole–particle cosine means the excitation moves charge
    # between spatially disjoint regions (donor -> acceptor).
    ct_info = {}
    if per_state and "hole_scores" in per_state[0]:
        for s in per_state:
            c = cosine(s["hole_scores"], s["part_scores"])
            ct_info[s["state"]] = (c < 0.5, c, top_atoms(s["hole_scores"]), top_atoms(s["part_scores"]))

    # Geometry: prefer coordinates embedded in the JSON, else read the .xyz.
    comment = "fragment"
    if "coords_angstrom" in d:
        atoms = [(elements[i], *d["coords_angstrom"][i]) for i in range(n_atoms)]
    elif args.xyz:
        comment, atoms = read_xyz(args.xyz)
        if len(atoms) != n_atoms:
            raise SystemExit(f"xyz has {len(atoms)} atoms, JSON has {n_atoms}")
    else:
        atoms = None  # no geometry → skip xyz output

    os.makedirs(args.outdir, exist_ok=True)
    groups = cluster_states(vectors, args.sim)

    print(f"# {len(per_state)} states -> {len(groups)} spatial group(s) "
          f"(cosine link >= {args.sim})")
    job_lines = ["#!/bin/bash", "set -e"]
    summary = []
    for gi, states in enumerate(groups):
        # Mean per-atom localization over the group's states, renormalized.
        mean = [sum(vectors[s][a] for s in states) / len(states) for a in range(n_atoms)]
        tot = sum(mean) or 1.0
        mean = [x / tot for x in mean]
        idx, cum = select_fragment(mean, args.floor, args.coverage, args.max_atoms)
        sig = {}
        for a in idx:
            sig[elements[a]] = sig.get(elements[a], 0) + 1
        sig_str = "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(sig.items()))
        summary.append((gi, states, idx, cum, sig_str))
        print(f"#  group {gi}: states {states} -> {len(idx)} atoms "
              f"({sig_str}, coverage {cum:.3f}): {brace_spec(idx)}")
        if ct_info:
            ct_states = [s for s in states if ct_info.get(s, (False,))[0]]
            if ct_states:
                s0 = ct_states[0]
                _, c, don, acc = ct_info[s0]
                print(f"#    charge-transfer states {ct_states}: donor {don} -> acceptor {acc} "
                      f"(state {s0} hole-particle cos {c:.2f})")

        if atoms is not None:
            xyz_path = os.path.join(args.outdir, f"fragment_{gi}.xyz")
            hdr = f"DMET-STEOM group {gi} fragment ({sig_str}); states {states}; coverage {cum:.3f}"
            write_fragment_xyz(xyz_path, atoms, idx, hdr)

        job = (f"./gansu -x {args.xyz_path} {args.gansu_args} "
               f"--post_hf_method dmet_steom --dmet_fragments \"{brace_spec(idx)}\" "
               f"> group_{gi}.log 2>&1")
        job_lines.append(f"# group {gi}: states {states} ({sig_str})")
        job_lines.append(job)

    with open(os.path.join(args.outdir, "jobs.sh"), "w") as f:
        f.write("\n".join(job_lines) + "\n")
    print(f"# wrote {len(groups)} fragment xyz + jobs.sh to {args.outdir}/")


if __name__ == "__main__":
    main()
