"""
GANSU vs PySCF CCSD/CCSD(T) Benchmark Script

Usage:
    pip install pyscf
    python benchmark_ccsd_pyscf.py

Generates reference CCSD/CCSD(T) energies and timing for comparison with GANSU.
All basis sets use cart=True to match GANSU's Cartesian Gaussians (6d, 10f).
"""

import time
import numpy as np

try:
    from pyscf import gto, scf, cc
except ImportError:
    print("PySCF is not installed. Install with: pip install pyscf")
    exit(1)


# ============================================================
#  Molecule definitions (matching GANSU xyz files)
# ============================================================

molecules = {
    "H2O": dict(
        atom="O 0.0 0.0 0.127; H 0.0 0.758 -0.509; H 0.0 -0.758 -0.509",
        charge=0, spin=0
    ),
    "NH3": dict(
        atom="N 0.0 0.0 0.128; H 0.0 0.941 -0.298; H 0.815 -0.470 -0.298; H -0.815 -0.470 -0.298",
        charge=0, spin=0
    ),
    "CH4": dict(
        atom="C 0.0 0.0 0.0; H 0.625 -0.625 -0.625; H 0.625 0.625 0.625; H -0.625 0.625 -0.625; H -0.625 -0.625 0.625",
        charge=0, spin=0
    ),
    "HF": dict(
        atom="F 0.0 0.0 0.096; H 0.0 0.0 -0.860",
        charge=0, spin=0
    ),
    "N2": dict(
        atom="N 0.0 0.0 0.567; N 0.0 0.0 -0.567",
        charge=0, spin=0
    ),
    "HCl": dict(
        atom="Cl 0.0 0.0 0.073; H 0.0 0.0 -1.240",
        charge=0, spin=0
    ),
    "Formaldehyde": dict(
        atom="O 0.0 0.0 0.683; C 0.0 0.0 -0.534; H 0.0 0.926 -1.129; H 0.0 -0.926 -1.129",
        charge=0, spin=0
    ),
    "Methanol": dict(
        atom="C -0.050 0.665 0.0; O -0.050 -0.768 0.0; H -1.090 0.996 0.0; H 0.439 1.082 0.886; H 0.439 1.082 -0.886; H 0.912 -1.005 0.0",
        charge=0, spin=0
    ),
    "Ethanol": dict(
        atom="C 1.187 -0.429 0.0; C 0.0 0.555 0.0; O -1.218 -0.204 0.0; H -1.930 0.485 0.0; H 2.126 0.116 0.0; H 1.151 -1.062 0.881; H 1.151 -1.062 -0.881; H 0.063 1.203 0.883; H 0.063 1.203 -0.883",
        charge=0, spin=0
    ),
}

basis_sets = ["sto-3g", "cc-pvdz", "cc-pvtz"]

# Representative combinations with good contrast (small → large)
benchmarks = [
    ("H2O",          "sto-3g"),    # small:  nocc=5, nvir=2
    ("H2O",          "cc-pvdz"),   # medium: nocc=5, nvir=19
    ("Formaldehyde", "cc-pvdz"),   # medium: nocc=8, nvir=32
    ("NH3",          "cc-pvtz"),   # large:  nocc=5, nvir=53
]


def run_benchmark(mol_name, basis_name):
    """Run RHF -> CCSD -> CCSD(T) and return results with timing."""
    mol_args = molecules[mol_name]
    cart = (basis_name != "sto-3g")  # Cartesian for DZ/TZ

    mol = gto.M(**mol_args, basis=basis_name, cart=cart, unit='Angstrom')
    n_basis = mol.nao_nr()
    n_elec = mol.nelectron
    nocc = n_elec // 2
    nvir = n_basis - nocc

    # RHF
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    t0 = time.time()
    mf.kernel()
    t_hf = time.time() - t0

    # CCSD
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    t0 = time.time()
    mycc.kernel()
    t_ccsd = time.time() - t0

    # CCSD(T)
    t0 = time.time()
    et = mycc.ccsd_t()
    t_ccsd_t = time.time() - t0

    return {
        "mol": mol_name,
        "basis": basis_name,
        "n_basis": n_basis,
        "nocc": nocc,
        "nvir": nvir,
        "e_hf": mf.e_tot,
        "e_ccsd_corr": mycc.e_corr,
        "e_ccsd_total": mycc.e_tot,
        "e_t": et,
        "e_ccsd_t_total": mycc.e_tot + et,
        "t_hf": t_hf,
        "t_ccsd": t_ccsd,
        "t_ccsd_t": t_ccsd_t,
        "t_total": t_hf + t_ccsd + t_ccsd_t,
    }


# ============================================================
#  Run all benchmarks
# ============================================================

print("=" * 90)
print("  GANSU vs PySCF Benchmark: CCSD / CCSD(T) Reference Values")
print("=" * 90)
print()

results = []
for mol_name, basis_name in benchmarks:
    label = f"{mol_name}/{basis_name}"
    print(f"Running {label} ...", end=" ", flush=True)
    try:
        r = run_benchmark(mol_name, basis_name)
        results.append(r)
        print(f"done ({r['t_total']:.1f}s)")
    except Exception as e:
        print(f"FAILED: {e}")

# ============================================================
#  Output: Table for presentation
# ============================================================

print()
print("=" * 90)
print("  Results Table")
print("=" * 90)
print()

# Table 1: Energies
print(f"{'Molecule':<14s} {'Basis':<10s} {'N_bas':>5s} {'nocc':>4s} {'nvir':>4s} "
      f"{'E(HF)':>16s} {'E(CCSD) corr':>16s} {'E(CCSD(T)) total':>18s}")
print("-" * 90)
for r in results:
    print(f"{r['mol']:<14s} {r['basis']:<10s} {r['n_basis']:>5d} {r['nocc']:>4d} {r['nvir']:>4d} "
          f"{r['e_hf']:>16.10f} {r['e_ccsd_corr']:>16.10f} {r['e_ccsd_t_total']:>18.10f}")

print()

# Table 2: Timing (PySCF on CPU)
print(f"{'Molecule':<14s} {'Basis':<10s} {'N_bas':>5s} {'nocc':>4s} {'nvir':>4s} "
      f"{'HF (s)':>8s} {'CCSD (s)':>10s} {'(T) (s)':>10s} {'Total (s)':>10s}")
print("-" * 90)
for r in results:
    print(f"{r['mol']:<14s} {r['basis']:<10s} {r['n_basis']:>5d} {r['nocc']:>4d} {r['nvir']:>4d} "
          f"{r['t_hf']:>8.2f} {r['t_ccsd']:>10.2f} {r['t_ccsd_t']:>10.2f} {r['t_total']:>10.2f}")

print()

# Table 3: C++ constexpr reference values for test_validation.cu
print("=" * 90)
print("  Reference values for test_validation.cu")
print("=" * 90)
print()
for r in results:
    tag = r['mol'].replace(' ', '_')
    basis_tag = r['basis'].replace('-', '').replace('*', 's')
    prefix = f"REF_{tag}"
    print(f"// {r['mol']} / {r['basis']}  (nocc={r['nocc']}, nvir={r['nvir']}, N={r['n_basis']})")
    print(f"constexpr real_t {prefix}_RHF_{basis_tag} = {r['e_hf']:.10f};")
    print(f"constexpr real_t {prefix}_CCSD_{basis_tag}_corr = {r['e_ccsd_corr']:.10f};")
    print(f"constexpr real_t {prefix}_CCSDT_{basis_tag}_total = {r['e_ccsd_t_total']:.10f};")
    print()
