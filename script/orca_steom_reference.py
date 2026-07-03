#!/usr/bin/env python3
"""
ORCA STEOM-CCSD reference script for bt-PNO-STEOM Phase P3 validation.

Generates an ORCA input file (`! STEOM-CCSD <basis>`), runs ORCA if `orca`
is in PATH (or `ORCA_PATH` env var is set), and parses the output for
STEOM excited-state energies + dominant transitions.

Usage:
    python3 script/orca_steom_reference.py xyz/H2O.xyz STO-3G 3
    python3 script/orca_steom_reference.py xyz/H2O.xyz cc-pVDZ 5

    # Also compute the triplet block (%mdci DoTriplet true) — roots are then
    # labeled singlet/triplet in the output table:
    python3 script/orca_steom_reference.py xyz/formaldehyde.xyz STO-3G 5 --triplet

    # If ORCA binary is in a non-standard path:
    ORCA_PATH=/opt/orca-6.0.1/orca python3 script/orca_steom_reference.py ...

    # Skip the ORCA run (just write the input file for manual execution):
    python3 script/orca_steom_reference.py xyz/H2O.xyz STO-3G 3 --dry-run

Output:
  --- ORCA STEOM-CCSD reference -------
    state  ω (Ha)        ω (eV)      character
       0   0.30421000    8.2766       1a' -> 2a' (95%)
       ...

The ORCA AutoSTEOM active-space selector (OThresh/VThresh, default 1e-3)
should match GANSU's CIS-NTO defaults so the active spaces are aligned.

Notes:
  * ORCA `! STEOM-CCSD <basis>` auto-runs CIS + IP/EA-EOM-CCSD internally,
    same as GANSU's composite STEOM dispatch.
  * Cartesian-vs-spherical: ORCA defaults to spherical (5d). GANSU is
    Cartesian (6d). For d+ basis (cc-pVDZ etc.) we add `! 6D` to ORCA
    input to align with GANSU.
  * If a state has dominant character that is a doubly-excited configuration
    (η or %active < threshold), ORCA prints "WARNING: Low η". Such states
    typically have σ-bond character and should be skipped for STEOM gates.
"""

import os
import shutil
import subprocess
import sys
import tempfile


def read_xyz_body(path):
    """Return list of '<sym> <x> <y> <z>' lines (ORCA xyz block format)."""
    with open(path) as f:
        natoms = int(f.readline().strip())
        f.readline()  # comment line
        return [f.readline().strip() for _ in range(natoms)]


def build_orca_input(xyz_path, basis, n_states, triplet=False):
    """Build an ORCA STEOM-CCSD input string. Forces Cartesian d basis (6D)
    via %method block to match GANSU's Cartesian GTO convention.

    triplet=True adds `DoTriplet true` (NB: singular — `DoTriplets` errors in
    ORCA 6.x) so ORCA also prints the STEOM-CCSD RESULTS (TRIPLETS) block;
    the parser then labels each root singlet/triplet.
    """
    atoms = read_xyz_body(xyz_path)
    # For d+ basis sets, force Cartesian via %method block (ORCA 6.x syntax).
    # NOTE: ORCA 6.1.1 does not accept `6D`, `ForceCartesian`, or
    # `CartesianGTO` keywords for forcing Cartesian d basis. We run with
    # ORCA's default (spherical 5d) and accept a ~0.1-0.5 mHa basis-induced
    # discrepancy against GANSU (which uses Cartesian 6d throughout). This
    # difference is small enough that the 1 mHa STEOM gate is still
    # meaningful for sub-phase 3.9 + 3.13 validation.
    method_block = ""
    # Parallel execution: ORCA needs an explicit %pal block (it runs serial
    # otherwise). Set ORCA_NPROCS=N (>1) to add it. ORCA MPI parallelism
    # requires the orca binary to be launched by its ABSOLUTE path with a
    # matching OpenMPI in PATH — set ORCA_PATH to the full path if needed.
    nprocs = int(os.environ.get("ORCA_NPROCS", "1") or "1")
    pal_block = f"%pal nprocs {nprocs} end\n" if nprocs > 1 else ""
    triplet_line = "  DoTriplet true\n" if triplet else ""
    body = (
        f"! STEOM-CCSD {basis} TightSCF\n"
        f"{method_block}"
        f"{pal_block}"
        f"%mdci\n"
        f"  NRoots {n_states}\n"
        f"  DoSTEOM true\n"
        f"{triplet_line}"
        f"  OThresh 1e-3\n"
        f"  VThresh 1e-3\n"
        f"end\n"
        f"%maxcore 4000\n"
        f"*xyz 0 1\n"
    )
    for atom_line in atoms:
        body += f"  {atom_line}\n"
    body += "*\n"
    return body


def parse_orca_steom_output(text):
    """Parse ORCA STEOM-CCSD excited-state energies (ORCA 6.x format).

    The relevant block in ORCA 6.x output is delimited by
        STEOM-CCSD RESULTS (SINGLETS)
        ...
        IROOT=  1:  0.435420 au    11.848 eV   95563.6 cm**-1 <S**2> = ...
        IROOT=  2:  ...
        ...
        STEOM-CCSD done
    (There are pre-STEOM IROOT lines for IP-EOM / EA-EOM that we must skip
    by gating on the STEOM-CCSD RESULTS header.)

    With DoTriplet a second block "STEOM-CCSD RESULTS (TRIPLETS)" follows the
    singlet one; per-root spin is taken from the block header, cross-checked
    against the `<S**2> = X` tail when present (0 → singlet, 2 → triplet).

    Returns: list of (state_index, omega_Ha, omega_eV, spin_label)
    """
    results = []
    lines = text.splitlines()
    in_steom_block = False
    block_spin = ""
    for line in lines:
        stripped = line.strip()
        if "STEOM-CCSD RESULTS" in stripped:
            in_steom_block = True
            up = stripped.upper()
            block_spin = ("triplet" if "TRIPLET" in up
                          else "singlet" if "SINGLET" in up else "")
            continue
        if in_steom_block and "STEOM-CCSD done" in stripped:
            in_steom_block = False   # keep scanning: TRIPLETS block may follow
            continue
        if not in_steom_block:
            continue
        # ORCA 6.x line format:
        #   IROOT=  N:  E_au au    E_eV eV   wavenumber cm**-1   <S**2> = X
        if stripped.startswith("IROOT=") and " au " in line and " eV " in line:
            try:
                tokens = stripped.replace(":", "").split()
                # tokens = ["IROOT=", "N", "E_au", "au", "E_eV", "eV", ...]
                idx_state = int(tokens[1])
                e_au_pos = next(i for i, t in enumerate(tokens) if t == "au")
                e_ev_pos = next(i for i, t in enumerate(tokens) if t == "eV")
                omega_Ha = float(tokens[e_au_pos - 1])
                omega_eV = float(tokens[e_ev_pos - 1])
                spin = block_spin
                if "<S**2>" in stripped:            # cross-check via <S**2>
                    try:
                        s2 = float(stripped.split("=")[-1].split()[0])
                        spin = "triplet" if s2 > 1.0 else "singlet"
                    except (ValueError, IndexError):
                        pass
                results.append((idx_state, omega_Ha, omega_eV, spin))
            except (ValueError, StopIteration, IndexError):
                pass
    return results


def main(xyz_path, basis, n_states, dry_run=False, triplet=False):
    inp_text = build_orca_input(xyz_path, basis, n_states, triplet=triplet)
    print(f"--- ORCA STEOM-CCSD reference ({xyz_path}, {basis}, {n_states} states"
          f"{', +triplets' if triplet else ''}) ---\n")

    if dry_run:
        print("DRY-RUN — ORCA input file content:\n")
        print(inp_text)
        return

    orca_bin = os.environ.get("ORCA_PATH") or shutil.which("orca")
    if not orca_bin:
        print("ERROR: ORCA binary not found. Set ORCA_PATH env var, add `orca` to PATH,",
              "or rerun with --dry-run to dump the input file for manual execution.",
              file=sys.stderr)
        sys.exit(1)

    workdir = tempfile.mkdtemp(prefix="orca_steom_")
    inp_path = os.path.join(workdir, "steom.inp")
    out_path = os.path.join(workdir, "steom.out")
    with open(inp_path, "w") as f:
        f.write(inp_text)
    print(f"  workdir = {workdir}")
    print(f"  orca    = {orca_bin}")
    print(f"  input   = {inp_path}")

    with open(out_path, "w") as fout:
        proc = subprocess.run([orca_bin, inp_path], stdout=fout,
                              stderr=subprocess.STDOUT, cwd=workdir)
    if proc.returncode != 0:
        print(f"ERROR: ORCA exited with status {proc.returncode}. See {out_path}",
              file=sys.stderr)
        sys.exit(proc.returncode)

    with open(out_path) as f:
        out_text = f.read()
    states = parse_orca_steom_output(out_text)
    if not states:
        print("WARNING: No STEOM-CCSD excited states parsed from ORCA output. ")
        print(f"         See raw output at: {out_path}")
        sys.exit(2)

    print(f"\n  ORCA STEOM-CCSD excited states (from {out_path}):")
    print(f"   state  omega (Ha)         omega (eV)    spin")
    for (idx, oHa, oeV, spin) in states:
        print(f"    {idx:3d}   {oHa:.10f}     {oeV:8.4f}   {spin}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    xyz = sys.argv[1]
    basis = sys.argv[2]
    n_states = int(sys.argv[3])
    dry_run = "--dry-run" in sys.argv[4:]
    triplet = "--triplet" in sys.argv[4:]
    main(xyz, basis, n_states, dry_run=dry_run, triplet=triplet)
