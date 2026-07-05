#!/usr/bin/env python3
"""Smoke test for the FMQA SCF-free UHF evaluation API (Phase A).

Covers the acceptance criteria of the UHF spec that need no FMQA package:
  1. RHF consistency: energy_of_uhf(Ca=Cb=C_RHF occ) == RHF total_energy
  2. NumPy reference: energy_of_uhf matches a pure-NumPy UHF functional
  3. Density route == MO route
  4. Batch == loop
  5. Spin properties: RHF solution has <S^2> ~ 0 and per-atom spin ~ 0
  6. Reproducibility (bit-exact) + timing
  7. SCF polish: symmetric guess stays at RHF; spin-alternating guess on a
     stretched H4 relaxes to a lower broken-symmetry solution with <S^2> > 0

Usage (from repo root, after building libgansu.so in build/):
  python3 script/check_uhf_energy_api.py [xyz/H2O.xyz] [sto-3g]
"""
import os
import sys
import time
import tempfile
import numpy as np

sys.path.insert(0, "python")
import gansu

xyz = sys.argv[1] if len(sys.argv) > 1 else "xyz/H2O.xyz"
basis = sys.argv[2] if len(sys.argv) > 2 else "sto-3g"

gansu.init()
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {detail}")


def uhf_energy_numpy(Ca_occ, Cb_occ, h, eri, enn):
    """Reference UHF energy from occupied alpha/beta MOs (single occupancy)."""
    Pa = Ca_occ @ Ca_occ.T
    Pb = Cb_occ @ Cb_occ.T
    Pt = Pa + Pb
    J = np.einsum("pqrs,rs->pq", eri, Pt)
    Ka = np.einsum("prqs,rs->pq", eri, Pa)
    Kb = np.einsum("prqs,rs->pq", eri, Pb)
    return (np.sum(Pt * h) + 0.5 * np.sum(Pt * J)
            - 0.5 * (np.sum(Pa * Ka) + np.sum(Pb * Kb)) + enn)


# --- RHF reference wavefunction ---------------------------------------------
mol_rhf = gansu.Molecule(xyz, basis=basis)
res = mol_rhf.run(method="RHF")
e_rhf = res.total_energy
n = mol_rhf.num_basis
nocc = mol_rhf.nocc
C = res.mo_coefficients
C_occ = np.ascontiguousarray(C[:, :nocc])

# UHF-method handle for the SCF-free evaluators.
mol = gansu.Molecule(xyz, basis=basis, method="UHF")
h = mol.hcore
eri = mol.eri
enn = mol.nuclear_repulsion_energy

# 1. RHF consistency (closed shell: na = nb = nocc, Ca = Cb = C_RHF).
e1 = mol.energy_of_uhf(Ca_occ=C_occ, Cb_occ=C_occ)
check("RHF consistency", abs(e1 - e_rhf) < 1e-10, f"|dE|={abs(e1 - e_rhf):.3e} Ha")

# 2. NumPy reference.
e_np = uhf_energy_numpy(C_occ, C_occ, h, eri, enn)
check("NumPy reference", abs(e1 - e_np) < 1e-8, f"|dE|={abs(e1 - e_np):.3e} Ha")

# 3. Density route == MO route.
Pa = C_occ @ C_occ.T
Pb = C_occ @ C_occ.T
e3 = mol.energy_of_uhf(Pa=Pa, Pb=Pb)
check("density route", abs(e3 - e1) < 1e-11, f"|dE|={abs(e3 - e1):.3e} Ha")

# 4. Batch == loop.
ca_b = np.stack([C_occ] * 8)
cb_b = np.stack([C_occ] * 8)
eb = mol.energy_of_uhf_batch(ca_b, cb_b)
check("batch", eb.shape == (8,) and np.all(eb == e1))

# 5. Spin properties of the RHF (symmetric) solution: <S^2> ~ 0, spins ~ 0.
s2, spins = mol.spin_properties(Pa, Pb)
check("RHF <S^2> ~ 0", abs(s2) < 1e-8, f"<S^2>={s2:.3e}")
check("RHF atom spins ~ 0", np.max(np.abs(spins)) < 1e-8,
      f"max|m_A|={np.max(np.abs(spins)):.3e}")

# 6. Reproducibility (bit-exact) + timing.
t0 = time.time()
vals = {mol.energy_of_uhf(Ca_occ=C_occ, Cb_occ=C_occ) for _ in range(100)}
dt = (time.time() - t0) / 100
check("bit-exact over 100 evals", len(vals) == 1, f"{dt * 1e3:.2f} ms/eval")

# 7. SCF polish on a stretched H4 chain (broken symmetry).
h4_xyz = os.path.join(tempfile.gettempdir(), "gansu_h4_stretched.xyz")
R = 1.8  # Angstrom, stretched so UHF breaks symmetry
with open(h4_xyz, "w") as f:
    f.write("4\nstretched H4\n")
    for i in range(4):
        f.write(f"H 0.0 0.0 {i * R:.4f}\n")

mol_h4 = gansu.Molecule(h4_xyz, basis="sto-3g", method="UHF")
res_h4 = gansu.Molecule(h4_xyz, basis="sto-3g").run(method="RHF")
e_rhf_h4 = res_h4.total_energy
n4 = mol_h4.num_basis
nocc4 = mol_h4.num_electrons // 2
C4 = res_h4.mo_coefficients
Cocc4 = np.ascontiguousarray(C4[:, :nocc4])

# Symmetric guess must stay at the RHF stationary point.
e_sym = mol_h4.uhf_scf_from(Cocc4, Cocc4)
check("polish symmetric -> RHF", abs(e_sym - e_rhf_h4) < 1e-6,
      f"E={e_sym:.6f} vs RHF {e_rhf_h4:.6f}")

# Spin-alternating guess: rotate HOMO/LUMO oppositely for alpha/beta.
if n4 > nocc4:
    th = 0.5
    homo, lumo = nocc4 - 1, nocc4
    Ca = C4.copy()
    Cb = C4.copy()
    Ca[:, homo] = np.cos(th) * C4[:, homo] + np.sin(th) * C4[:, lumo]
    Cb[:, homo] = np.cos(th) * C4[:, homo] - np.sin(th) * C4[:, lumo]
    Ca_occ4 = np.ascontiguousarray(Ca[:, :nocc4])
    Cb_occ4 = np.ascontiguousarray(Cb[:, :nocc4])
    e_bs, Pa_bs, Pb_bs = mol_h4.uhf_scf_from(Ca_occ4, Cb_occ4, return_density=True)
    s2_bs, spins_bs = mol_h4.spin_properties(Pa_bs, Pb_bs)
    check("polish broken-symmetry E <= RHF", e_bs <= e_rhf_h4 + 1e-9,
          f"E_bs={e_bs:.6f} <= RHF {e_rhf_h4:.6f}")
    check("broken-symmetry <S^2> > 0", s2_bs > 1e-3,
          f"<S^2>={s2_bs:.4f}, spins={np.round(spins_bs, 3)}")

gansu.finalize()
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
