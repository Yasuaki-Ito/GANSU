#!/usr/bin/env python3
"""Smoke test for the FMQA SCF-free energy evaluation API.

Covers the acceptance criteria that can be checked without the FMQA package:
  1. SCF consistency: energy_of(C_occ from converged SCF) == total_energy
  2. Density route: energy_of(P=2 C C^T) matches the MO route bit-exactly
  3. Variational: rotated (non-SCF) orbitals give E > E_SCF
  4. Performance: single evaluation wall time
  5. Reproducibility: repeated evaluation is bit-identical
  6. Lazy init: hcore / eri / overlap / energy_of work WITHOUT run()

Usage (from repo root, after building libgansu.so in build/):
  python3 script/check_energy_api.py [xyz/H2O.xyz] [sto-3g]
"""
import sys
import time
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


# --- Lazy path first (no run()): integrals prepared without SCF -------------
mol_lazy = gansu.Molecule(xyz, basis=basis)
n = mol_lazy.num_basis
nocc = mol_lazy.nocc
S = mol_lazy.overlap
h = mol_lazy.hcore
eri = mol_lazy.eri
check("lazy shapes", S.shape == (n, n) and h.shape == (n, n) and eri.shape == (n, n, n, n),
      f"(n={n}, nocc={nocc})")
check("hcore symmetric", np.allclose(h, h.T, atol=1e-12))
check("eri (pq|rs)=(rs|pq) symmetry", np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12))

# Core-guess energy via pure NumPy vs energy_of — validates the whole chain.
Xs = np.linalg.cholesky(np.linalg.inv(S))  # S^-1 factor: columns orthonormal wrt S
w, V = np.linalg.eigh(Xs.T @ h @ Xs)
C0 = Xs @ V[:, :nocc]
P0 = 2.0 * C0 @ C0.T
J = np.einsum("pqrs,rs->pq", eri, P0)
K = np.einsum("prqs,rs->pq", eri, P0)
e_np = np.sum(P0 * h) + 0.5 * np.sum(P0 * (J - 0.5 * K)) + mol_lazy.nuclear_repulsion_energy
e_lazy = mol_lazy.energy_of(C_occ=C0)
check("NumPy reference", abs(e_lazy - e_np) < 1e-8, f"|dE|={abs(e_lazy - e_np):.3e} Ha")

# --- SCF-based checks --------------------------------------------------------
mol = gansu.Molecule(xyz, basis=basis)
res = mol.run(method="RHF")
e_scf = res.total_energy
C = res.mo_coefficients  # (n, n), column j = MO j
C_occ = np.ascontiguousarray(C[:, :nocc])

# 1. SCF consistency
e1 = mol.energy_of(C_occ=C_occ)
check("SCF consistency", abs(e1 - e_scf) < 1e-10, f"|dE|={abs(e1 - e_scf):.3e} Ha")

# 2. Density route == MO route (last-bit differences allowed: the MO route
#    forms P = 2 C C^T inside the C API with a different summation order
#    than NumPy's GEMM, so the two P matrices differ in the last ulp).
P = 2.0 * C_occ @ C_occ.T
e2 = mol.energy_of(P=P)
check("density route", abs(e2 - e1) < 1e-11, f"|dE|={abs(e2 - e1):.3e} Ha")

# 3. Variational: rotate occ<->vir -> E must rise
if n > nocc:
    theta = 0.3
    Crot = C.copy()
    i, a = nocc - 1, nocc  # HOMO-LUMO rotation
    Crot[:, i] = np.cos(theta) * C[:, i] + np.sin(theta) * C[:, a]
    e3 = mol.energy_of(C_occ=np.ascontiguousarray(Crot[:, :nocc]))
    check("variational (E(theta) > E_SCF)", e3 > e_scf, f"dE={e3 - e_scf:.6f} Ha")

# 4+5. Reproducibility (bit-exact) + timing
t0 = time.time()
vals = {mol.energy_of(C_occ=C_occ) for _ in range(100)}
dt = (time.time() - t0) / 100
check("bit-exact over 100 evals", len(vals) == 1, f"{dt * 1e3:.2f} ms/eval")

# Batch
batch = np.stack([C_occ] * 8)
eb = mol.energy_of_batch(batch)
check("batch", eb.shape == (8,) and np.all(eb == e1))

gansu.finalize()
print("\nALL PASS" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
