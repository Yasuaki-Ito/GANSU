"""Parse HF_main stdout into structured data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedResult:
    raw_output: str = ""
    molecule: dict[str, Any] = field(default_factory=dict)
    basis_set: dict[str, Any] = field(default_factory=dict)
    scf_iterations: list[dict[str, float]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    post_hf: dict[str, Any] | None = None
    orbital_energies: list[dict[str, Any]] = field(default_factory=list)
    orbital_energies_beta: list[dict[str, Any]] = field(default_factory=list)
    mulliken: list[dict[str, Any]] = field(default_factory=list)
    mayer_bond_order: list[list[float]] = field(default_factory=list)
    wiberg_bond_order: list[list[float]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "raw_output": self.raw_output,
            "molecule": self.molecule,
            "basis_set": self.basis_set,
            "scf_iterations": self.scf_iterations,
            "summary": self.summary,
            "orbital_energies": self.orbital_energies,
            "orbital_energies_beta": self.orbital_energies_beta,
            "mulliken": self.mulliken,
            "mayer_bond_order": self.mayer_bond_order,
            "wiberg_bond_order": self.wiberg_bond_order,
            "timing": self.timing,
        }
        if self.post_hf:
            d["post_hf"] = self.post_hf
        return d


# Regex for the single-line iteration format:
# ---- Iteration: 0 ----  Energy: -1.860... Total energy: -1.117... Energy difference: 0
_ITER_RE = re.compile(
    r"^-+\s*Iteration:\s*(\d+)\s*-+"
    r"\s+Energy:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"\s+Total energy:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"(?:\s+Energy difference:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?))?"
)


def parse_output(text: str) -> ParsedResult:
    """Parse the full stdout of HF_main into structured data."""
    result = ParsedResult(raw_output=text)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # --- Single-line iteration ---
        m = _ITER_RE.match(line)
        if m:
            iteration: dict[str, Any] = {
                "iteration": int(m.group(1)),
                "energy": float(m.group(2)),
                "total_energy": float(m.group(3)),
            }
            if m.group(4) is not None:
                iteration["delta_e"] = float(m.group(4))
            result.scf_iterations.append(iteration)
            i += 1
            continue

        # --- Molecule / Atom Summary ---
        if line in ("[Molecule Summary]", "[Atom Summary]"):
            atoms: list[dict[str, Any]] = []
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("[") and l not in ("[Molecule Summary]", "[Atom Summary]"):
                    break
                if l == "":
                    i += 1
                    continue
                for pat, key, typ in [
                    (r"Number of atoms:\s*(\d+)", "num_atoms", int),
                    (r"Number of electrons:\s*(\d+)", "num_electrons", int),
                    (r"Number of alpha-spin electrons:\s*(\d+)", "alpha_electrons", int),
                    (r"Number of beta-spin electrons:\s*(\d+)", "beta_electrons", int),
                ]:
                    m2 = re.match(pat, l)
                    if m2:
                        result.molecule[key] = typ(m2.group(1))
                        break
                else:
                    # Atom line
                    m2 = re.match(r"Atom\s+(\d+):\s*(\w+)\s*\(([^)]+)\)", l)
                    if m2:
                        coords = [float(x.strip()) for x in m2.group(3).split(",")]
                        atoms.append({"index": int(m2.group(1)), "element": m2.group(2), "coords": coords})
                i += 1
            if atoms:
                result.molecule["atoms"] = atoms
            continue

        # --- Basis Set Summary ---
        if line == "[Basis Set Summary]":
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("[") and l != "[Basis Set Summary]":
                    break
                if l == "":
                    i += 1
                    continue
                for pat, key in [
                    (r"Number of basis functions:\s*(\d+)", "num_basis"),
                    (r"Number of primitive basis functions:\s*(\d+)", "num_primitives"),
                    (r"Number of auxiliary basis functions:\s*(\d+)", "num_auxiliary"),
                ]:
                    m2 = re.match(pat, l)
                    if m2:
                        result.basis_set[key] = int(m2.group(1))
                i += 1
            continue

        # --- Orbital Energies (new format) ---
        if line in ("[Orbital Energies]", "[Orbital Energies (Alpha)]", "[Orbital Energies (Beta)]"):
            is_beta = "(Beta)" in line
            target = result.orbital_energies_beta if is_beta else result.orbital_energies
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l == "" or l.startswith("[") or l.startswith("---"):
                    break
                m2 = re.match(r"MO\s+(\d+)\s+\((\w+)\)\s+([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)", l)
                if m2:
                    target.append({
                        "index": int(m2.group(1)),
                        "occupation": m2.group(2),
                        "energy": float(m2.group(3)),
                    })
                i += 1
            continue

        # --- Orbital Energies (old format, fallback) ---
        if line == "Orbital energies:":
            i += 1
            idx = 1
            while i < len(lines):
                l = lines[i].strip()
                if l == "" or l.startswith("[") or l.startswith("---"):
                    break
                for val in l.split():
                    try:
                        result.orbital_energies.append({"index": idx, "occupation": "?", "energy": float(val)})
                        idx += 1
                    except ValueError:
                        pass
                i += 1
            continue

        # --- Mulliken Population ---
        if line == "[Mulliken population]":
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("["):
                    break
                if l == "":
                    i += 1
                    continue
                m2 = re.match(r"Atom\s+(\d+)\s+(\w+):\s*([-+]?\d+\.?\d*)", l)
                if m2:
                    result.mulliken.append({
                        "index": int(m2.group(1)),
                        "element": m2.group(2),
                        "charge": float(m2.group(3)),
                    })
                i += 1
            continue

        # --- Mayer Bond Order ---
        if line == "[Mayer bond order]":
            result.mayer_bond_order = _parse_matrix(lines, i + 1)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                i += 1
            continue

        # --- Wiberg Bond Order ---
        if line == "[Wiberg bond order]":
            result.wiberg_bond_order = _parse_matrix(lines, i + 1)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                i += 1
            continue

        # --- Calculation Summary ---
        if line == "[Calculation Summary]":
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("[") and l != "[Calculation Summary]":
                    break
                if l == "":
                    i += 1
                    continue
                _parse_summary_line(l, result.summary)
                i += 1
            continue

        # --- Post-HF Summary ---
        if line == "[Calculation Summary (Post-HF)]":
            result.post_hf = {}
            i += 1
            while i < len(lines):
                l = lines[i].strip()
                if l.startswith("[") and l != "[Calculation Summary (Post-HF)]":
                    break
                if l == "":
                    i += 1
                    continue
                _parse_post_hf_line(l, result.post_hf)
                i += 1
            continue

        # --- Timing Summary ---
        if line == "[Timing Summary]":
            i += 1
            entries = []
            while i < len(lines):
                l = lines[i].strip()
                if l == "" or l.startswith("["):
                    break
                # e.g. "compute_energy: 1.003 microseconds total, called 2 times."
                m2 = re.match(r"(\w+):\s*([\d.]+)\s+(\w+)\s+total,\s+called\s+(\d+)\s+times", l)
                if m2:
                    entries.append({
                        "name": m2.group(1),
                        "time": float(m2.group(2)),
                        "unit": m2.group(3),
                        "calls": int(m2.group(4)),
                    })
                i += 1
            result.timing["entries"] = entries
            continue

        i += 1

    # Build summary from iterations if [Calculation Summary] was not present
    if not result.summary and result.scf_iterations:
        last = result.scf_iterations[-1]
        result.summary["total_energy"] = last.get("total_energy")
        result.summary["electronic_energy"] = last.get("energy")
        result.summary["iterations"] = len(result.scf_iterations)
        if "delta_e" in last:
            result.summary["energy_difference"] = last["delta_e"]

    return result


def _parse_summary_line(line: str, summary: dict[str, Any]) -> None:
    mapping = [
        (r"Method:\s*(.+)", "method", str),
        (r"Schwarz screening threshold:\s*([\d.eE+-]+)", "schwarz_threshold", float),
        (r"Initial guess method:\s*(.+)", "initial_guess", str),
        (r"Convergence algorithm:\s*(.+)", "convergence_algorithm", str),
        (r"Number of iterations:\s*(\d+)", "iterations", int),
        (r"Convergence criterion:\s*([\d.eE+-]+)", "convergence_criterion", float),
        (r"Energy difference:\s*([\d.eE+-]+)", "energy_difference", float),
        (r"Energy \(without nuclear repulsion\):\s*([-+]?\d+\.?\d+)", "electronic_energy", float),
        (r"Total Energy:\s*([-+]?\d+\.?\d+)", "total_energy", float),
        (r"Computing time:\s*([\d.eE+-]+)", "computing_time_ms", float),
    ]
    for pattern, key, typ in mapping:
        m = re.match(pattern, line)
        if m:
            summary[key] = typ(m.group(1).strip())
            return


def _parse_post_hf_line(line: str, post_hf: dict[str, Any]) -> None:
    mapping = [
        (r"Post-HF method:\s*(.+)", "method", str),
        (r"Post-HF energy correction:\s*([-+]?\d+\.?\d+)", "correction", float),
        (r"Total Energy \(including post-HF correction\):\s*([-+]?\d+\.?\d+)", "total_energy", float),
    ]
    for pattern, key, typ in mapping:
        m = re.match(pattern, line)
        if m:
            post_hf[key] = typ(m.group(1).strip())
            return


def _parse_matrix(lines: list[str], start: int) -> list[list[float]]:
    """Parse a space-separated matrix block."""
    matrix: list[list[float]] = []
    i = start
    while i < len(lines):
        l = lines[i].strip()
        if l == "" or l.startswith("["):
            break
        row: list[float] = []
        for val in l.split():
            try:
                row.append(float(val))
            except ValueError:
                break
        if row:
            matrix.append(row)
        i += 1
    return matrix
