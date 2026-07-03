#!/usr/bin/env python3
"""Smoke + correctness test for the derivative / property C-API getters.

Exercises the getters added alongside gansu_get_energy_gradient / _hessian /
_frequencies / _dipole / _excited_states, through the Python wrapper:

  1. dipole        — shape (3,), plausible magnitude for a polar molecule (H2O)
  2. gradient      — shape (natoms, 3); validated against a central finite
                     difference of the total energy (the strong correctness test)
  3. hessian       — shape (3N, 3N), symmetric
  4. frequencies   — count is 3N-6 for a non-linear molecule (H2O -> 3)
  5. excited_states— energies/oscillator strengths arrays after a CIS run

Usage (from repo root, with libgansu.so built, e.g. via GANSU_LIB):
  GANSU_LIB=.../libgansu.so python3 script/check_property_api.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, "python")
import gansu

BOHR = 0.52917721067          # Angstrom per Bohr
AU_TO_DEBYE = 2.5417464157
BASIS = "sto-3g"
TMP = os.path.join(os.environ.get("TMPDIR", "."), "_prop_test.xyz")

# Near-equilibrium water geometry, Angstrom (symbol, x, y, z).
WATER = [
    ("O", 0.0000000000,  0.0000000000,  0.1173000000),
    ("H", 0.0000000000,  0.7572000000, -0.4692000000),
    ("H", 0.0000000000, -0.7572000000, -0.4692000000),
]


def write_xyz(path, atoms):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\nmol\n")
        for s, x, y, z in atoms:
            f.write(f"{s} {x:.10f} {y:.10f} {z:.10f}\n")


def energy_of(atoms):
    write_xyz(TMP, atoms)
    return gansu.Molecule(TMP, basis=BASIS).run(method="RHF").total_energy


gansu.init()
ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        ok = False


# --- Base RHF run --------------------------------------------------------
write_xyz(TMP, WATER)
r = gansu.Molecule(TMP, basis=BASIS).run(method="RHF")
natoms = r.num_atoms
print(f"H2O / {BASIS}: E(RHF) = {r.total_energy:.10f} Ha,  natoms = {natoms}")

# --- 1. Dipole -----------------------------------------------------------
mu = r.dipole
mu_debye = np.linalg.norm(mu) * AU_TO_DEBYE
check("dipole shape (3,)", mu.shape == (3,))
check("dipole points along water C2 axis (z)",
      abs(mu[0]) < 1e-6 and abs(mu[1]) < 1e-6 and abs(mu[2]) > 1e-3,
      f"mu={np.round(mu,4)} a.u.")
check("dipole magnitude plausible for H2O", 0.2 < mu_debye < 3.0,
      f"|mu|={mu_debye:.3f} Debye")

# --- 2. Gradient + finite-difference validation --------------------------
g = r.energy_gradient          # (natoms, 3), Ha/Bohr  (captured before FD runs)
check("gradient shape (natoms,3)", g.shape == (natoms, 3))

h_bohr = 1.0e-3
h_ang = h_bohr * BOHR
comps = [(0, 2), (1, 1), (1, 2)]   # O-z, H-y, H-z
max_err = 0.0
for (a, c) in comps:
    plus = [list(t) for t in WATER]
    minus = [list(t) for t in WATER]
    plus[a][1 + c] += h_ang
    minus[a][1 + c] -= h_ang
    e_plus = energy_of([tuple(t) for t in plus])
    e_minus = energy_of([tuple(t) for t in minus])
    num = (e_plus - e_minus) / (2.0 * h_bohr)
    err = abs(num - g[a, c])
    max_err = max(max_err, err)
    print(f"    grad[atom {a}, comp {c}]  analytic={g[a,c]:+.6f}  "
          f"fd={num:+.6f}  |diff|={err:.2e}")
check("gradient matches finite difference", max_err < 1.0e-3,
      f"max|diff|={max_err:.2e} Ha/Bohr")

# --- 3. Hessian ----------------------------------------------------------
H = r.hessian
ndim = 3 * natoms
check("hessian shape (3N,3N)", H.shape == (ndim, ndim))
check("hessian symmetric", np.allclose(H, H.T, atol=1e-6),
      f"max asym={np.abs(H - H.T).max():.2e}")

# --- 4. Frequencies ------------------------------------------------------
freqs = r.frequencies
# Non-linear triatomic: 3N - 6 = 3 vibrational modes.
check("frequency count (3N-6 for non-linear H2O)", len(freqs) == ndim - 6,
      f"n={len(freqs)}")
print("    frequencies (cm^-1):", np.round(freqs, 1))

# --- 5. Excited states ---------------------------------------------------
write_xyz(TMP, WATER)
n_states = 5
rx = gansu.Molecule(TMP, basis=BASIS).run(
        method="RHF", post_hf="cis", n_excited_states=str(n_states))
es = rx.excited_states
check("excited_states keys", set(es) == {"energies", "oscillator_strengths"})
check("excitation-energy count", len(es["energies"]) == n_states,
      f"n={len(es['energies'])}")
check("excitation energies positive and ascending",
      len(es["energies"]) > 0 and np.all(es["energies"] > 0)
      and np.all(np.diff(es["energies"]) >= -1e-8))
check("oscillator strengths non-negative",
      np.all(es["oscillator_strengths"] >= -1e-8))
ev = es["energies"] * 27.211386
print("    excitation energies (eV):", np.round(ev, 3))
print("    oscillator strengths:    ", np.round(es["oscillator_strengths"], 4))

# --- cleanup -------------------------------------------------------------
try:
    os.remove(TMP)
except OSError:
    pass

gansu.finalize()
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
